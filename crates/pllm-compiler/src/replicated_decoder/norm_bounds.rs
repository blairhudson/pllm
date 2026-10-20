//! Integer-only certificates for complete public matrix blocks and tensor rows.
//!
//! These routines bound ordinary integer products and every floor/ceiling choice
//! of stochastic truncation. They neither inspect private values nor establish
//! the provenance of a caller-supplied row certificate. The compiler must bind
//! that provenance and workers must recompute public matrix certificates.
//! Shared arithmetic lives in `pllm_core::integer_bounds`; these adapters retain
//! the original admission limits, public certificate fields, and error mapping.

use pllm_core::{integer_bounds, puma::Error};

const MAX_ROWS: usize = 128;
const MAX_COLUMNS: usize = 16_384;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MatrixCertificate {
    pub rows: usize,
    pub columns: usize,
    pub maximum_row_l1: u128,
    pub maximum_column_l1: u128,
    pub frobenius_squared: u128,
}

impl MatrixCertificate {
    fn shared(self) -> Result<integer_bounds::MatrixCertificate, Error> {
        shape(self.rows, self.columns)?;
        integer_bounds::MatrixCertificate::from_bounds(
            self.rows,
            self.columns,
            self.maximum_row_l1,
            self.maximum_column_l1,
            self.frobenius_squared,
        )
        .map_err(error)
    }

    /// ||W||_2^2 <= min(||W||_infinity * ||W||_1, ||W||_F^2).
    pub fn operator_squared(self) -> Result<u128, Error> {
        self.shared()?.operator_squared().map_err(error)
    }
}

/// Ordered, one-use public coefficient scan. Any error retires the scan.
#[repr(transparent)]
pub(super) struct MatrixScan(integer_bounds::MatrixScan);

impl MatrixScan {
    pub fn payload_bytes(rows: usize, columns: usize) -> Result<usize, Error> {
        shape(rows, columns)?;
        integer_bounds::MatrixScan::payload_bytes(rows, columns).map_err(error)
    }

    pub fn new(rows: usize, columns: usize) -> Result<Self, Error> {
        let payload = Self::payload_bytes(rows, columns)?;
        integer_bounds::MatrixScan::new(rows, columns, payload)
            .map(Self)
            .map_err(error)
    }

    pub fn append(&mut self, offset: usize, values: &[i64]) -> Result<(), Error> {
        self.0.append(offset, values).map_err(error)
    }

    pub fn finish(self) -> Result<MatrixCertificate, Error> {
        let certificate = self.0.finish().map_err(error)?;
        Ok(MatrixCertificate {
            rows: certificate.rows(),
            columns: certificate.columns(),
            maximum_row_l1: certificate.maximum_row_l1(),
            maximum_column_l1: certificate.maximum_column_l1(),
            frobenius_squared: certificate.frobenius_squared(),
        })
    }
}

fn shape(rows: usize, columns: usize) -> Result<(), Error> {
    if rows == 0 || rows > MAX_ROWS || columns == 0 || columns > MAX_COLUMNS {
        return Err(Error::Shape);
    }
    Ok(())
}

fn error(error: integer_bounds::Error) -> Error {
    match error {
        integer_bounds::Error::Shape => Error::Shape,
        integer_bounds::Error::Bounds => Error::Bounds,
        integer_bounds::Error::Allocation => Error::Allocation,
        integer_bounds::Error::Retired => Error::Binding,
    }
}

/// Complete output row after one truncation per coordinate. Each truncation's
/// absolute rounding error is strictly below one integer unit.
pub(super) fn linear_energy(
    input_squared: u128,
    matrix: MatrixCertificate,
    shift: u32,
) -> Result<u128, Error> {
    if shift == 0 || shift >= 64 {
        return Err(Error::Bounds);
    }
    integer_bounds::linear_energy(input_squared, matrix.shared()?, shift, 1).map_err(error)
}

pub(super) fn sum_energy(left: u128, right: u128) -> Result<u128, Error> {
    integer_bounds::sum_energy(left, right).map_err(error)
}

/// Half-split rotation with separate signed coefficient-product truncations.
/// Each two-coordinate block has operator norm sqrt(c^2+s^2)/2^shift.
/// Each final coordinate has two rounding errors, each strictly below one.
pub(super) fn rotary_energy(
    input_squared: u128,
    cos: &[i64],
    sin: &[i64],
    shift: u32,
) -> Result<u128, Error> {
    if cos.is_empty()
        || cos.len() != sin.len()
        || cos.len() > MAX_ROWS / 2
        || shift == 0
        || shift >= 64
    {
        return Err(Error::Shape);
    }
    integer_bounds::rotary_energy(input_squared, cos, sin, shift, 2).map_err(error)
}

/// Cauchy--Schwarz bound for the complete untruncated dot, before ring narrowing.
/// The caller must additionally prove the primitive's strict quarter-ring bound.
#[cfg(test)]
pub(super) fn dot_radius(left_squared: u128, right_squared: u128) -> Result<u128, Error> {
    integer_bounds::dot_radius(left_squared, right_squared).map_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn energy(values: &[i64]) -> u128 {
        values
            .iter()
            .map(|v| u128::from(v.unsigned_abs()).pow(2))
            .sum()
    }

    fn scan(rows: usize, columns: usize, values: &[i64], chunk: usize) -> MatrixCertificate {
        let mut owner = MatrixScan::new(rows, columns).unwrap();
        for (i, values) in values.chunks(chunk).enumerate() {
            owner.append(i * chunk, values).unwrap();
        }
        owner.finish().unwrap()
    }

    #[test]
    fn shared_adapters_preserve_legacy_envelopes_and_admission_limits() {
        let cert = scan(2, 3, &[1, -2, 3, -4, 0, 5], 4);
        assert_eq!(cert.operator_squared().unwrap(), 55);
        assert_eq!(linear_energy(25, cert, 1).unwrap(), 441);
        assert_eq!(linear_energy(25, cert, 3).unwrap(), 49);
        assert_eq!(linear_energy(0, cert, 18).unwrap(), 4);
        assert_eq!(linear_energy(17, cert, 18).unwrap(), 9);
        assert_eq!(sum_energy(2, 3).unwrap(), 16);
        assert_eq!(rotary_energy(25, &[5], &[6], 3).unwrap(), 81);
        assert_eq!(dot_radius(25, 61).unwrap(), 40);
        assert_eq!(linear_energy(1, cert, 0), Err(Error::Bounds));
        assert_eq!(rotary_energy(1, &[1], &[0], 0), Err(Error::Shape));
        assert_eq!(rotary_energy(1, &[1; 65], &[0; 65], 18), Err(Error::Shape));
        let invalid = MatrixCertificate { rows: 129, ..cert };
        assert_eq!(invalid.operator_squared(), Err(Error::Shape));
        let mut owner = MatrixScan::new(1, 1).unwrap();
        assert_eq!(owner.append(1, &[1]), Err(Error::Shape));
        assert_eq!(owner.append(0, &[1]), Err(Error::Binding));
        assert_eq!(owner.finish(), Err(Error::Binding));
    }

    #[test]
    fn independent_exhaustive_small_matrices_and_rounding_choices() {
        for code in 0..625usize {
            let mut code = code;
            let mut w = [0i64; 4];
            for value in &mut w {
                *value = (code % 5) as i64 - 2;
                code /= 5;
            }
            let cert = scan(2, 2, &w, 3);
            let row_l1 = w
                .chunks_exact(2)
                .map(|r| r.iter().map(|v| v.unsigned_abs() as u128).sum::<u128>())
                .max()
                .unwrap();
            let col_l1 = (0..2)
                .map(|c| u128::from(w[c].unsigned_abs()) + u128::from(w[2 + c].unsigned_abs()))
                .max()
                .unwrap();
            assert_eq!(cert.maximum_row_l1, row_l1);
            assert_eq!(cert.maximum_column_l1, col_l1);
            assert_eq!(cert.frobenius_squared, energy(&w));
            for x0 in -3..=3i64 {
                for x1 in -3..=3i64 {
                    let raw = [w[0] * x0 + w[1] * x1, w[2] * x0 + w[3] * x1];
                    let input = energy(&[x0, x1]);
                    assert!(energy(&raw) <= cert.operator_squared().unwrap() * input);
                    for shift in [1, 3] {
                        for choices in 0..4 {
                            let mut out = [0; 2];
                            for i in 0..2 {
                                let floor = raw[i].div_euclid(1 << shift);
                                let ceiling = floor + i64::from(raw[i].rem_euclid(1 << shift) != 0);
                                out[i] = if choices & (1 << i) == 0 {
                                    floor
                                } else {
                                    ceiling
                                };
                            }
                            assert!(
                                energy(&out) <= linear_energy(input, cert, shift as u32).unwrap()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_and_signed_blocks_keep_tighter_operator_bounds() {
        let mut identity = vec![0; 64 * 64];
        for i in 0..64 {
            identity[i * 64 + i] = 1 << 18;
        }
        let cert = scan(64, 64, &identity, 127);
        assert_eq!(cert.operator_squared().unwrap(), 1 << 36);
        assert_eq!(cert.frobenius_squared, 64 << 36);
        assert_eq!(scan(64, 64, &identity, 1), cert);
        assert_eq!(scan(64, 64, &identity, identity.len()), cert);
        let negative = identity.iter().map(|v| -v).collect::<Vec<_>>();
        assert_eq!(scan(64, 64, &negative, 129), cert);
    }

    #[test]
    fn every_rotary_rounding_choice_and_bias_energy_is_bounded() {
        for x in -5..=5i64 {
            for y in -5..=5i64 {
                for (c, s) in [(8, 0), (0, 8), (5, 6), (-5, 6), (-8, 0)] {
                    let bound = rotary_energy(energy(&[x, y]), &[c], &[s], 3).unwrap();
                    for choices in 0..16 {
                        let raw = [x * c, -y * s, y * c, x * s];
                        let mut terms = [0; 4];
                        for i in 0..4 {
                            terms[i] = raw[i].div_euclid(8)
                                + i64::from(choices & (1 << i) != 0 && raw[i].rem_euclid(8) != 0);
                        }
                        let out = [terms[0] + terms[1], terms[2] + terms[3]];
                        assert!(energy(&out) <= bound);
                        let bias = [-2, 3];
                        assert!(
                            energy(&[out[0] + bias[0], out[1] + bias[1]])
                                <= sum_energy(bound, energy(&bias)).unwrap()
                        );
                        for key in [[-3, 1], [0, 0], [2, -4]] {
                            let dot = i128::from(out[0]) * i128::from(key[0])
                                + i128::from(out[1]) * i128::from(key[1]);
                            assert!(dot.unsigned_abs() <= dot_radius(bound, energy(&key)).unwrap());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn incomplete_malformed_and_overflowed_scans_are_terminal() {
        for (offset, values) in [(1, vec![0]), (0, vec![]), (0, vec![0; 5])] {
            let mut owner = MatrixScan::new(2, 2).unwrap();
            assert!(owner.append(offset, &values).is_err());
            assert!(owner.append(0, &[0; 4]).is_err());
            assert!(owner.finish().is_err());
        }
        let mut owner = MatrixScan::new(2, 2).unwrap();
        owner.append(0, &[1, -1, 2]).unwrap();
        assert!(owner.finish().is_err());
        let mut owner = MatrixScan::new(2, 2).unwrap();
        assert!(owner.append(0, &[i64::MIN; 4]).is_err());
        assert!(owner.finish().is_err());
        assert!(MatrixScan::new(129, 1).is_err());
        assert!(MatrixScan::new(1, MAX_COLUMNS + 1).is_err());
        assert!(MatrixScan::payload_bytes(1, usize::MAX).is_err());
        assert!(MatrixScan::payload_bytes(128, MAX_COLUMNS).unwrap() < (257 << 10));
        assert!(dot_radius(u128::MAX, 2).is_err());
        assert!(sum_energy(u128::MAX, 1).is_err());
        assert!(rotary_energy(u128::MAX, &[2], &[0], 18).is_err());
        assert!(rotary_energy(1, &[1], &[], 18).is_err());
        assert!(linear_energy(1, scan(1, 1, &[1], 1), 64).is_err());
    }
}
