//! Integer-only certificates for complete public matrix blocks and tensor rows.
//!
//! These routines bound ordinary integer products and every floor/ceiling choice
//! of stochastic truncation. They neither inspect private values nor establish
//! the provenance of a caller-supplied row certificate. The compiler must bind
//! that provenance and workers must recompute public matrix certificates.

use pllm_core::puma::Error;

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
    /// ||W||_2^2 <= min(||W||_infinity * ||W||_1, ||W||_F^2).
    pub fn operator_squared(self) -> Result<u128, Error> {
        if self.rows == 0 || self.rows > MAX_ROWS || self.columns == 0 || self.columns > MAX_COLUMNS
        {
            return Err(Error::Shape);
        }
        // An overflowing product exceeds every representable Frobenius bound.
        Ok(self
            .maximum_row_l1
            .checked_mul(self.maximum_column_l1)
            .map_or(self.frobenius_squared, |bound| {
                bound.min(self.frobenius_squared)
            }))
    }
}

struct Pending {
    columns: Vec<u128>,
    rows: usize,
    offset: usize,
    current_row: u128,
    maximum_row: u128,
    frobenius_squared: u128,
}

/// Ordered, one-use public coefficient scan. Any error retires the scan.
pub(super) struct MatrixScan(Option<Pending>);

impl MatrixScan {
    pub fn payload_bytes(rows: usize, columns: usize) -> Result<usize, Error> {
        if rows == 0 || rows > MAX_ROWS || columns == 0 || columns > MAX_COLUMNS {
            return Err(Error::Shape);
        }
        columns
            .checked_mul(std::mem::size_of::<u128>())
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<Self>() + std::mem::size_of::<Pending>())
            })
            .ok_or(Error::Bounds)
    }

    pub fn new(rows: usize, columns: usize) -> Result<Self, Error> {
        Self::payload_bytes(rows, columns)?;
        let mut sums = Vec::new();
        sums.try_reserve_exact(columns)
            .map_err(|_| Error::Allocation)?;
        sums.resize(columns, 0);
        Ok(Self(Some(Pending {
            columns: sums,
            rows,
            offset: 0,
            current_row: 0,
            maximum_row: 0,
            frobenius_squared: 0,
        })))
    }

    pub fn append(&mut self, offset: usize, values: &[i64]) -> Result<(), Error> {
        let mut pending = self.0.take().ok_or(Error::Binding)?;
        let columns = pending.columns.len();
        if values.is_empty()
            || offset != pending.offset
            || offset
                .checked_add(values.len())
                .is_none_or(|n| n > pending.rows * columns)
        {
            return Err(Error::Shape);
        }
        for value in values {
            let magnitude = u128::from(value.unsigned_abs());
            let column = pending.offset % columns;
            pending.columns[column] = pending.columns[column]
                .checked_add(magnitude)
                .ok_or(Error::Bounds)?;
            pending.current_row = pending
                .current_row
                .checked_add(magnitude)
                .ok_or(Error::Bounds)?;
            pending.frobenius_squared = pending
                .frobenius_squared
                .checked_add(magnitude * magnitude)
                .ok_or(Error::Bounds)?;
            pending.offset += 1;
            if pending.offset % columns == 0 {
                pending.maximum_row = pending.maximum_row.max(pending.current_row);
                pending.current_row = 0;
            }
        }
        self.0 = Some(pending);
        Ok(())
    }

    pub fn finish(mut self) -> Result<MatrixCertificate, Error> {
        let pending = self.0.take().ok_or(Error::Binding)?;
        if pending.offset != pending.rows * pending.columns.len() || pending.current_row != 0 {
            return Err(Error::Shape);
        }
        Ok(MatrixCertificate {
            rows: pending.rows,
            columns: pending.columns.len(),
            maximum_row_l1: pending.maximum_row,
            maximum_column_l1: pending.columns.into_iter().max().ok_or(Error::Shape)?,
            frobenius_squared: pending.frobenius_squared,
        })
    }
}

fn root(value: u128) -> u128 {
    let floor = value.isqrt();
    floor + u128::from(floor * floor != value)
}

fn squared(radius: u128) -> Result<u128, Error> {
    radius.checked_mul(radius).ok_or(Error::Bounds)
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
    let raw = input_squared
        .checked_mul(matrix.operator_squared()?)
        .ok_or(Error::Bounds)?;
    let radius = root(raw)
        .div_ceil(1u128 << shift)
        .checked_add(root(matrix.rows as u128))
        .ok_or(Error::Bounds)?;
    squared(radius)
}

pub(super) fn sum_energy(left: u128, right: u128) -> Result<u128, Error> {
    squared(root(left).checked_add(root(right)).ok_or(Error::Bounds)?)
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
    let mut maximum = 0u128;
    for (&c, &s) in cos.iter().zip(sin) {
        let c = u128::from(c.unsigned_abs());
        let s = u128::from(s.unsigned_abs());
        maximum = maximum.max(
            c.checked_mul(c)
                .and_then(|v| v.checked_add(s * s))
                .ok_or(Error::Bounds)?,
        );
    }
    let raw = input_squared.checked_mul(maximum).ok_or(Error::Bounds)?;
    let rounding = 2 * root((2 * cos.len()) as u128);
    squared(
        root(raw)
            .div_ceil(1u128 << shift)
            .checked_add(rounding)
            .ok_or(Error::Bounds)?,
    )
}

/// Cauchy--Schwarz bound for the complete untruncated dot, before ring narrowing.
/// The caller must additionally prove the primitive's strict quarter-ring bound.
#[cfg(test)]
pub(super) fn dot_radius(left_squared: u128, right_squared: u128) -> Result<u128, Error> {
    Ok(root(
        left_squared
            .checked_mul(right_squared)
            .ok_or(Error::Bounds)?,
    ))
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
