//! Bounded public SIMD layouts and resource admission, not an HE implementation.
//! SEAL separately validates cryptographic parameters and executes ciphertexts.
use serde::Serialize;

pub const MAX_WIDTH: usize = 32;
pub const MAX_ALLOCATION: usize = 512 << 20;

#[derive(Debug, Serialize)]
pub struct Parameters {
    pub degree: usize,
    pub coefficient_bits: Vec<u32>,
    pub scale_bits: u32,
    pub depth: usize,
    pub rotation_keys: usize,
    pub relinearization: bool,
    pub live_ciphertexts: usize,
    pub ciphertext_frame_bytes: usize,
    pub public_key_frame_bytes: usize,
    pub switching_key_frame_bytes: usize,
    pub allocation_bytes_estimate: usize,
}

pub fn parameters(
    depth: usize,
    rotation_keys: usize,
    relinearization: bool,
    live_ciphertexts: usize,
) -> Result<Parameters, String> {
    if !(1..=8).contains(&depth)
        || rotation_keys > MAX_WIDTH - 1
        || !(1..=64).contains(&live_ciphertexts)
    {
        return Err("CKKS reference parameter geometry exceeds bounds".into());
    }
    // TC128 ceilings: 218 bits at N=8192, 438 at N=16384. The final
    // 50-bit prime is for key switching, not an extra multiplication level.
    let total_bits = 100 + 40 * depth;
    let degree = if total_bits <= 218 { 8192 } else { 16384 };
    let mut coefficient_bits = vec![50];
    coefficient_bits.extend(std::iter::repeat_n(40, depth));
    coefficient_bits.push(50);
    let limbs = depth + 2;
    let ciphertext_frame_bytes = 2 * degree * (limbs - 1) * 8 + 4096;
    let public_key_frame_bytes = 2 * degree * limbs * 8 + 4096;
    let switching_key_frame_bytes = (limbs - 1) * public_key_frame_bytes + 4096;
    let key_count = rotation_keys + usize::from(relinearization);
    // Client and provider keys plus bounded serialized delivery overlap;
    // contexts/RNS scratch, live ciphertexts, Python bindings and pool slack.
    let allocation_bytes_estimate = (32 << 20)
        + 3 * (key_count * switching_key_frame_bytes + public_key_frame_bytes)
        + live_ciphertexts * ciphertext_frame_bytes
        + 64 * degree * limbs * 8;
    if allocation_bytes_estimate > MAX_ALLOCATION {
        return Err("CKKS reference allocation estimate exceeds 512 MiB".into());
    }
    Ok(Parameters {
        degree,
        coefficient_bits,
        scale_bits: 40,
        depth,
        rotation_keys,
        relinearization,
        live_ciphertexts,
        ciphertext_frame_bytes,
        public_key_frame_bytes,
        switching_key_frame_bytes,
        allocation_bytes_estimate,
    })
}

pub fn padded_width(rows: usize, columns: usize) -> Result<usize, String> {
    if !(1..=MAX_WIDTH).contains(&rows) || !(1..=MAX_WIDTH).contains(&columns) {
        return Err("CKKS matrix dimensions must be between 1 and 32".into());
    }
    Ok(rows.max(columns).next_power_of_two())
}

pub fn diagonals(weights: &[f64], rows: usize, columns: usize) -> Result<Vec<Vec<f64>>, String> {
    let width = padded_width(rows, columns)?;
    if weights.len() != rows * columns || weights.iter().any(|v| !v.is_finite() || v.abs() > 1.0) {
        return Err("CKKS public weights require exact shape and finite [-1,1] values".into());
    }
    Ok((0..width)
        .map(|shift| {
            (0..width)
                .map(|row| {
                    let column = (row + shift) % width;
                    if row < rows && column < columns {
                        weights[row * columns + column]
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_diagonal_rotations_match_independent_dense_products() {
        for rows in [1, 3, 8, 17, 32] {
            for columns in [1, 5, 16, 31, 32] {
                let w = (0..rows * columns)
                    .map(|i| ((i * 7 % 19) as f64 - 9.0) / 16.0)
                    .collect::<Vec<_>>();
                let x = (0..columns)
                    .map(|i| (i as f64 - 16.0) / 32.0)
                    .collect::<Vec<_>>();
                let d = diagonals(&w, rows, columns).unwrap();
                let width = d.len();
                for row in 0..width {
                    let actual: f64 = d
                        .iter()
                        .enumerate()
                        .map(|(shift, d)| {
                            d[row] * x.get((row + shift) % width).copied().unwrap_or(0.0)
                        })
                        .sum();
                    let expected: f64 = if row < rows {
                        w[row * columns..(row + 1) * columns]
                            .iter()
                            .zip(&x)
                            .map(|(a, b)| a * b)
                            .sum()
                    } else {
                        0.0
                    };
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn parameter_depth_security_ceiling_and_allocation_precede_material() {
        for depth in 1..=8 {
            let p = parameters(depth, 1, true, 16).unwrap();
            let ceiling = if p.degree == 8192 { 218 } else { 438 };
            assert!(p.coefficient_bits.iter().sum::<u32>() <= ceiling);
            assert_eq!(p.coefficient_bits.len() - 2, depth);
            assert!(p.allocation_bytes_estimate <= MAX_ALLOCATION);
        }
        for (depth, rotations, live) in [(0, 1, 1), (9, 1, 1), (8, 31, 64), (1, 32, 1), (1, 1, 0)] {
            assert!(parameters(depth, rotations, true, live).is_err());
        }
        assert!(diagonals(&[f64::NAN], 1, 1).is_err());
        assert!(diagonals(&[1.01], 1, 1).is_err());
        assert!(diagonals(&[], usize::MAX, 2).is_err());
        assert!(diagonals(&[0.0], 1, 2).is_err());
    }
}
