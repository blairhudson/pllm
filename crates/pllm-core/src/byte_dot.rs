//! Bounded single-row ring32 dot products over immutable signed-byte weights.
use crate::kernels::{Executor, Matrix};

pub(crate) const MAX_WIDTH: usize = 131071;
const MAX_WEIGHTS: usize = 256 << 20;

pub(crate) fn multiply(
    matrix: &Matrix,
    executor: &Executor,
    x: &[u32],
) -> Result<Vec<u32>, String> {
    let (rows, width) = matrix.shape();
    // Check all geometry before allocating the invocation-local slab/output.
    let live_bytes = matrix.weight_bytes() as u128 + 8 * width as u128 + 4 * rows as u128;
    if width > MAX_WIDTH
        || x.len() != width
        || matrix.weight_bytes() > MAX_WEIGHTS
        || live_bytes > 512 << 20
    {
        return Err("byte-dot requires one row, width <= 131071, weights <= 256 MiB and live payload <= 512 MiB".into());
    }
    let mut limbs = Vec::new();
    limbs
        .try_reserve_exact(4 * width)
        .map_err(|_| "byte-dot limb allocation failed")?;
    limbs.resize(4 * width, 0i8);
    for (i, &value) in x.iter().enumerate() {
        let mut carry = i64::from(value);
        for k in 0..4 {
            let digit = ((carry + 128) & 255) - 128;
            limbs[k * width + i] = digit as i8;
            carry = (carry - digit) >> 8;
        }
        // The remaining carry multiplies 2^32, hence vanishes in this ring.
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(rows)
        .map_err(|_| "byte-dot output allocation failed")?;
    output.resize(rows, 0);
    let use_dotprod = executor.simd() && available();
    executor.rows(&mut output, 1, |row, out| {
        let weights = &matrix.as_signed_slice()[row * width..(row + 1) * width];
        #[cfg(target_arch = "aarch64")]
        if use_dotprod {
            // SAFETY: runtime dotprod check above; slices contain width and
            // 4*width elements. width*128*128 <= i32::MAX bounds every sum.
            out[0] = unsafe { sdot(weights, &limbs) };
            return;
        }
        let _ = use_dotprod;
        out[0] = scalar(weights, &limbs);
    });
    Ok(output)
}

fn available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

fn scalar(weights: &[i8], limbs: &[i8]) -> u32 {
    let n = weights.len();
    (0..4).fold(0u32, |total, k| {
        let sum: i32 = weights
            .iter()
            .zip(&limbs[k * n..(k + 1) * n])
            .map(|(&w, &x)| i32::from(w) * i32::from(x))
            .sum();
        total.wrapping_add((sum as u32).wrapping_shl(8 * k as u32))
    })
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn sdot(weights: &[i8], limbs: &[i8]) -> u32 {
    use std::arch::aarch64::*;
    let n = weights.len();
    let mut acc = [vdupq_n_s32(0); 4];
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: the loop guard and slab geometry bound all sixteen-byte loads.
        let w = unsafe { vld1q_s8(weights.as_ptr().add(i)) };
        for k in 0..4 {
            let x = unsafe { vld1q_s8(limbs.as_ptr().add(k * n + i)) };
            // Stable Rust 1.85 lacks stable dotprod intrinsics. No global flags.
            unsafe {
                std::arch::asm!("sdot {acc:v}.4s, {w:v}.16b, {x:v}.16b",
                    acc = inout(vreg) acc[k], w = in(vreg) w,
                    x = in(vreg) x, options(pure, nomem, nostack));
            }
        }
        i += 16;
    }
    let mut total = 0u32;
    for k in 0..4 {
        let mut sum = vaddvq_s32(acc[k]);
        for j in i..n {
            sum += i32::from(weights[j]) * i32::from(limbs[k * n + j]);
        }
        total = total.wrapping_add((sum as u32).wrapping_shl(8 * k as u32));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_full_ring_extremes_tails_and_forced_scalar() {
        let values = [
            0,
            127,
            128,
            255,
            256,
            0x7fffffff,
            0x80000000,
            0xffffff80,
            u32::MAX,
        ];
        for width in [1, 3, 4, 15, 16, 17, 31, 32, 33, 127, 896, MAX_WIDTH] {
            let bytes: Vec<u8> = (0..3 * width)
                .map(|i| [-128i8, 127, -1, 0, 1][i % 5] as u8)
                .collect();
            let matrix = Matrix::new(&bytes, 3, width).unwrap();
            let x: Vec<u32> = (0..width).map(|i| values[i % values.len()]).collect();
            let expected: Vec<u32> = matrix
                .as_signed_slice()
                .chunks_exact(width)
                .map(|w| {
                    w.iter()
                        .zip(&x)
                        .map(|(&w, &x)| i128::from(w) * i128::from(x))
                        .sum::<i128>() as u32
                })
                .collect();
            for simd in [false, true] {
                for threads in [1, 2] {
                    let executor = Executor::new(threads, simd).unwrap();
                    assert_eq!(matrix.wrap32_byte_dot(&executor, &x).unwrap(), expected);
                    assert_eq!(matrix.wrap32(&executor, &x, 1).unwrap(), expected);
                }
            }
        }
    }

    #[test]
    fn rejects_malformed_or_unbounded_geometry() {
        let matrix = Matrix::new(&[128; 17], 1, 17).unwrap();
        let executor = Executor::new(1, false).unwrap();
        for len in [0, 16, 18, 34] {
            assert!(matrix.wrap32_byte_dot(&executor, &vec![0; len]).is_err());
        }
        let wide = Matrix::new(&vec![128; MAX_WIDTH + 1], 1, MAX_WIDTH + 1).unwrap();
        assert!(wide
            .wrap32_byte_dot(&executor, &vec![0; MAX_WIDTH + 1])
            .is_err());
    }

    #[test]
    fn maximum_width_positive_limb_accumulator_is_bounded() {
        let matrix = Matrix::new(&vec![128; MAX_WIDTH], 1, MAX_WIDTH).unwrap();
        let x = vec![128; MAX_WIDTH];
        for simd in [false, true] {
            let executor = Executor::new(1, simd).unwrap();
            let expected = (-128i64 * 128 * MAX_WIDTH as i64) as u32;
            assert_eq!(
                matrix.wrap32_byte_dot(&executor, &x).unwrap(),
                vec![expected]
            );
        }
    }

    #[test]
    fn unavailable_architecture_uses_exact_scalar() {
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!available());
        let matrix = Matrix::new(&[128, 127, 255], 1, 3).unwrap();
        let executor = Executor::new(1, true).unwrap();
        assert_eq!(
            matrix.wrap32_byte_dot(&executor, &[u32::MAX; 3]).unwrap(),
            vec![2]
        );
    }
}
