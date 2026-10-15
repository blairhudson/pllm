//! Bounded research fixture only: exact one-input ring32 head via signed limbs.
use pllm_core::kernels::{Executor, Matrix};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Instant;

const MAX_BYTES: usize = 256 << 20;
// Full signed dot (including horizontal reduction) must fit i32.
const MAX_COLS: usize = i32::MAX as usize / (128 * 128);

#[cfg(unix)]
extern "C" {
    fn clock() -> i64;
}
fn cpu() -> f64 {
    #[cfg(unix)]
    {
        unsafe { clock() as f64 / 1_000_000.0 }
    }
    #[cfg(not(unix))]
    {
        0.0
    }
}
fn stamp() -> (Instant, f64) {
    (Instant::now(), cpu())
}
fn elapsed(s: (Instant, f64)) -> serde_json::Value {
    json!({"elapsed_seconds": s.0.elapsed().as_secs_f64(), "cpu_seconds": cpu()-s.1})
}
fn dotprod_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}
fn pack(x: &[u32], limbs: &mut [i8]) {
    let n = x.len();
    for (i, &value) in x.iter().enumerate() {
        let mut carry = value as i64;
        for k in 0..4 {
            let digit = ((carry + 128) & 255) - 128;
            limbs[k * n + i] = digit as i8;
            carry = (carry - digit) >> 8;
        }
        // Remaining carry is a multiple of 2^32 and deliberately discarded.
    }
}
fn scalar_limbs(w: &[i8], limbs: &[i8]) -> u32 {
    let n = w.len();
    (0..4).fold(0u32, |total, k| {
        let sum: i64 = w
            .iter()
            .zip(&limbs[k * n..(k + 1) * n])
            .map(|(&a, &b)| a as i64 * b as i64)
            .sum();
        total.wrapping_add((sum as u32).wrapping_shl(8 * k as u32))
    })
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn sdot(w: &[i8], limbs: &[i8]) -> u32 {
    use std::arch::aarch64::*;
    let n = w.len();
    let mut acc = [vdupq_n_s32(0); 4];
    let mut i = 0;
    while i + 16 <= n {
        let weights = vld1q_s8(w.as_ptr().add(i));
        for k in 0..4 {
            let input = vld1q_s8(limbs.as_ptr().add(k * n + i));
            // Stable inline assembly avoids unstable stdarch dotprod intrinsics
            // on the repository's Rust 1.85 toolchain. Feature-gated entry only.
            std::arch::asm!("sdot {acc:v}.4s, {w:v}.16b, {x:v}.16b",
                acc = inout(vreg) acc[k], w = in(vreg) weights,
                x = in(vreg) input, options(pure, nomem, nostack));
        }
        i += 16;
    }
    let mut result = 0u32;
    for k in 0..4 {
        let mut sum = vaddvq_s32(acc[k]);
        for j in i..n {
            sum += w[j] as i32 * limbs[k * n + j] as i32;
        }
        result = result.wrapping_add((sum as u32).wrapping_shl(8 * k as u32));
    }
    result
}
fn candidate(matrix: &Matrix, limbs: &[i8], output: &mut [u32], use_sdot: bool) {
    let (_, n) = matrix.shape();
    for (w, out) in matrix.as_signed_slice().chunks_exact(n).zip(output) {
        #[cfg(target_arch = "aarch64")]
        if use_sdot {
            // Runtime feature check, dimensions, and accumulator bound preflighted.
            *out = unsafe { sdot(w, limbs) };
            continue;
        }
        let _ = use_sdot;
        *out = scalar_limbs(w, limbs);
    }
}
fn oracle(matrix: &Matrix, x: &[u32]) -> Vec<u32> {
    // Independent signed i64 products and u64 wrapping accumulation; no limbs.
    matrix
        .as_signed_slice()
        .chunks_exact(x.len())
        .map(|row| {
            row.iter().zip(x).fold(0u64, |s, (&w, &v)| {
                s.wrapping_add((w as i64 * v as i64) as u64)
            }) as u32
        })
        .collect()
}
fn rng(s: &mut u64) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s as u32
}
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 && args.len() != 8 {
        return Err(
            "usage: boundary_head_dotprod ROWS COLS REPEATS SEED auto|scalar emit|quiet [ORDER_OFFSET=0|1]".into(),
        );
    }
    let rows: usize = args[1].parse().map_err(|_| "bad rows")?;
    let cols: usize = args[2].parse().map_err(|_| "bad cols")?;
    let repeats: usize = args[3].parse().map_err(|_| "bad repeats")?;
    let mut seed: u64 = args[4].parse().map_err(|_| "bad seed")?;
    let offset: usize = args
        .get(7)
        .map(|s| s.parse())
        .transpose()
        .map_err(|_| "bad order offset")?
        .unwrap_or(0);
    if offset > 1 {
        return Err("order offset must be 0 or 1".into());
    }
    if !["auto", "scalar"].contains(&args[5].as_str())
        || !["emit", "quiet"].contains(&args[6].as_str())
    {
        return Err("bad mode".into());
    }
    let bytes = rows.checked_mul(cols).ok_or("geometry overflow")?;
    let live_bound = (bytes as u128) * 2 + (rows as u128) * 20 + (cols as u128) * 8;
    if rows == 0
        || cols == 0
        || cols > MAX_COLS
        || bytes > MAX_BYTES
        || live_bound > (512u128 << 20)
        || (bytes as u128) * (repeats as u128) > (2u128 << 30)
        || !(1..=100).contains(&repeats)
        || seed == 0
    {
        return Err("geometry/repeats/seed exceeds bounded fixture limits".into());
    }
    if args[6] == "emit" && (bytes > 65536 || rows > 1024) {
        return Err("emit payload too large".into());
    }
    let setup = stamp();
    let mut raw = vec![0u8; bytes];
    for (i, w) in raw.iter_mut().enumerate() {
        *w = if i % 17 == 0 {
            128
        } else {
            rng(&mut seed) as u8
        };
    }
    let matrix = Matrix::new(&raw, rows, cols)?;
    drop(raw);
    let edges = [
        0,
        1,
        u32::MAX,
        0x7fffffff,
        0x80000000,
        127,
        128,
        255,
        256,
        0x7f7f7f7f,
        0x80808080,
        0xffffff80,
        0x00007fff,
        0x00008000,
    ];
    let x: Vec<u32> = (0..cols)
        .map(|i| {
            if i % 3 != 2 {
                edges[i % edges.len()]
            } else {
                rng(&mut seed)
            }
        })
        .collect();
    let executor = Executor::new(1, true)?;
    let mut limbs = vec![0i8; 4 * cols];
    let mut result = vec![0u32; rows];
    let setup_time = elapsed(setup);
    let feature = dotprod_available();
    let selected = feature && args[5] == "auto";
    let mut packing = 0.0;
    let mut packing_cpu = 0.0;
    let mut kernel = 0.0;
    let mut kernel_cpu = 0.0;
    let mut baseline_time = 0.0;
    let mut baseline_cpu = 0.0;
    // Warmups also validate full vectors, never hashes.
    pack(&x, &mut limbs);
    candidate(&matrix, &limbs, &mut result, selected);
    let expected = oracle(&matrix, &x);
    let baseline = matrix.wrap32(&executor, &x, 1)?;
    if result != expected || baseline != expected {
        return Err("oracle mismatch".into());
    }
    let mut samples = Vec::with_capacity(repeats);
    for index in 0..repeats {
        let candidate_first = (index + offset) % 2 == 0;
        let mut p = (0.0, 0.0);
        let mut k = (0.0, 0.0);
        let mut b = (0.0, 0.0);
        let mut control = Vec::new();
        for phase in 0..2 {
            if (phase == 0) == candidate_first {
                let t = stamp();
                pack(std::hint::black_box(&x), &mut limbs);
                p = (t.0.elapsed().as_secs_f64(), cpu() - t.1);
                let t = stamp();
                candidate(std::hint::black_box(&matrix), &limbs, &mut result, selected);
                std::hint::black_box(&result);
                k = (t.0.elapsed().as_secs_f64(), cpu() - t.1);
            } else {
                let t = stamp();
                control = matrix.wrap32(&executor, std::hint::black_box(&x), 1)?;
                b = (t.0.elapsed().as_secs_f64(), cpu() - t.1);
            }
        }
        // Validation and JSON construction stay outside both timed calls.
        if control != expected || result != expected {
            return Err("timed oracle mismatch".into());
        }
        packing += p.0;
        packing_cpu += p.1;
        kernel += k.0;
        kernel_cpu += k.1;
        baseline_time += b.0;
        baseline_cpu += b.1;
        samples.push(json!({"index": index,
            "order": if candidate_first { "candidate-first" } else { "baseline-first" },
            "packing": {"elapsed_seconds": p.0, "cpu_seconds": p.1},
            "kernel": {"elapsed_seconds": k.0, "cpu_seconds": k.1},
            "candidate_including_packing": {"elapsed_seconds": p.0+k.0, "cpu_seconds": p.1+k.1},
            "baseline": {"elapsed_seconds": b.0, "cpu_seconds": b.1}, "exact": true}));
    }
    let emit = args[6] == "emit";
    println!(
        "{}",
        json!({
            "schema": "boundary-head-dotprod-v2", "arch": std::env::consts::ARCH,
            "order_offset": offset, "samples": samples,
            "comparison": "packing+preallocated-candidate-output versus allocating Matrix::wrap32 API",
            "compiled_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("boundary_head_dotprod.rs"))),
            "compiled_kernels_sha256": format!("{:x}", Sha256::digest(include_bytes!("../src/kernels.rs"))),
            "build_provenance": option_env!("BOUNDARY_DOTPROD_BUILD_PROVENANCE"),
            "rows": rows, "cols": cols, "repeats": repeats, "dotprod_available": feature,
            "selected": if selected { "arm-sdot-inline-asm" } else { "scalar-balanced-limbs" },
            "baseline_simd": executor.simd(), "exact": true, "setup": setup_time,
            "packing": {"elapsed_seconds": packing, "cpu_seconds": packing_cpu},
            "kernel": {"elapsed_seconds": kernel, "cpu_seconds": kernel_cpu},
            "candidate_including_packing": {"elapsed_seconds": packing+kernel, "cpu_seconds": packing_cpu+kernel_cpu},
            "baseline": {"elapsed_seconds": baseline_time, "cpu_seconds": baseline_cpu},
            "owned_buffers": {"weights_bytes": bytes, "input_bytes": 4*cols,
                "input_limb_bytes": 4*cols, "candidate_output_bytes": 4*rows,
                "oracle_output_bytes": 4*rows, "baseline_retained_output_bytes": 4*rows,
                "baseline_call_temporary_upper_bytes": 8*rows, "setup_raw_temporary_bytes": bytes,
                "extra_persistent_weight_bytes": 0},
            "weights": if emit { Some(matrix.as_signed_slice()) } else { None },
            "input": if emit { Some(&x) } else { None },
            "output": if emit { Some(&result) } else { None },
            "baseline_output": if emit { Some(&baseline) } else { None }
        })
    );
    Ok(())
}
