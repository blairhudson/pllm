//! Exact ring products through balanced base-256 digits and CPU SGEMM.
//!
//! Signed i8 weights and digits have magnitude at most 128. Blocks of 1023
//! products have absolute sum < 2^24, so every float32 product and partial sum
//! is exact, regardless of reduction order. Reconstruction uses wrapping u32.
//! No persistent float weight copy or reduced-precision GPU backend is used.

use crate::kernels::Matrix;

pub const MAX_ELEMENTS: usize = 4 << 20;
pub const WORKSPACE_BYTES: usize = 64 << 20;
const BLOCK: usize = 1023;
const TILE: usize = 1024;

pub fn available() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

fn product(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // SAFETY: callers provide contiguous m*k, n*k and m*n buffers, all
        // dimensions are bounded below i32::MAX, and output aliases neither input.
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                k as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        Ok(())
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (a, b, c, m, n, k);
        Err("digit8 requires Apple Silicon CPU Accelerate".into())
    }
}

pub fn multiply(matrix: &Matrix, x: &[u32], batch: usize, bits: u32) -> Result<Vec<u32>, String> {
    let (rows, cols) = matrix.shape();
    if !available() {
        return Err("digit8 requires Apple Silicon CPU Accelerate".into());
    }
    if !matches!(bits, 16 | 24 | 32)
        || !(1..=4096).contains(&batch)
        || batch.checked_mul(cols) != Some(x.len())
        || x.len() > MAX_ELEMENTS
        || batch.checked_mul(rows).is_none_or(|n| n > MAX_ELEMENTS)
        || (bits < 32 && x.iter().any(|&v| v >= 1 << bits))
    {
        return Err("digit8 input exceeds ring or geometry bounds".into());
    }
    // Tiles and full output together stay below 64 MiB. Caller input/binding
    // copies and paged weights are separate owners priced by role admission.
    let tile = rows.min(TILE);
    let block = cols.min(BLOCK);
    let work = 4 * (batch * rows + batch * block + tile * block + batch * tile);
    if work > WORKSPACE_BYTES {
        return Err("digit8 workspace exceeds 64 MiB".into());
    }
    let weights = matrix.as_signed_slice();
    let mut out = vec![0u32; batch * rows];
    let mut a = vec![0f32; batch * block];
    let mut b = vec![0f32; tile * block];
    let mut c = vec![0f32; batch * tile];
    for start in (0..cols).step_by(BLOCK) {
        let k = (cols - start).min(BLOCK);
        for first in (0..rows).step_by(TILE) {
            let n = (rows - first).min(TILE);
            for r in 0..n {
                for i in 0..k {
                    b[r * k + i] = weights[(first + r) * cols + start + i] as f32;
                }
            }
            let mut carry = 0u64;
            for shift in (0..bits).step_by(8) {
                for r in 0..batch {
                    for i in 0..k {
                        let remaining = (u64::from(x[r * cols + start + i]) + carry) >> shift;
                        a[r * k + i] = (((remaining + 128) & 255) as i32 - 128) as f32;
                    }
                }
                product(
                    &a[..batch * k],
                    &b[..n * k],
                    &mut c[..batch * n],
                    batch,
                    n,
                    k,
                )?;
                for r in 0..batch {
                    for j in 0..n {
                        let v = c[r * n + j];
                        if !v.is_finite() || v.abs() >= (1u32 << 24) as f32 || v.fract() != 0.0 {
                            return Err("digit8 SGEMM violated the exact integer bound".into());
                        }
                        let index = r * rows + first + j;
                        out[index] = out[index].wrapping_add((v as i32 as u32).wrapping_shl(shift));
                    }
                }
                carry += 128u64 << shift;
            }
        }
    }
    if bits < 32 {
        out.iter_mut().for_each(|v| *v &= (1 << bits) - 1);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_extremes_carries_and_tiles() {
        if !available() {
            return;
        }
        for cols in [1, 1023, 1024, 2051] {
            let rows = 1031;
            let w: Vec<u8> = (0..rows * cols)
                .map(|i| [128, 127, 0, 255][i % 4])
                .collect();
            let m = Matrix::new(&w, rows, cols).unwrap();
            for bits in [16, 24, 32] {
                let mask = ((1u64 << bits) - 1) as u32;
                let x: Vec<u32> = (0..3 * cols)
                    .map(|i| [0, u32::MAX, 128, 255, 32768, 0x7fff_ffff][i % 6] & mask)
                    .collect();
                let actual = multiply(&m, &x, 3, bits).unwrap();
                for r in 0..3 {
                    for j in 0..rows {
                        let expected: i64 = (0..cols)
                            .map(|i| i64::from(x[r * cols + i]) * i64::from(w[j * cols + i] as i8))
                            .sum();
                        assert_eq!(actual[r * rows + j], expected as u32 & mask);
                    }
                }
            }
        }
    }
    #[test]
    fn rejects_invalid_domain_before_allocation() {
        let m = Matrix::new(&[1], 1, 1).unwrap();
        for (x, batch, bits) in [
            (vec![65536], 1, 16),
            (vec![0], 1, 17),
            (vec![], 0, 32),
            (vec![0], usize::MAX, 32),
        ] {
            assert!(multiply(&m, &x, batch, bits).is_err());
        }
    }
}
