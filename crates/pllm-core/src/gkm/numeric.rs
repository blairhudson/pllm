use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use super::{payload, Error, MAX_DIMENSION, NUMERIC_PROFILE};

fn rounded(value: f64) -> Result<f64, Error> {
    if !value.is_finite() {
        return Err(Error::Numeric);
    }
    // Multiplication by a binary power is exact for every admitted result.
    let scaled = value * 2048.0;
    if !scaled.is_finite() {
        return Err(Error::Numeric);
    }
    Ok(scaled.round_ties_even())
}

pub fn encode_weight(value: f64) -> Result<i16, Error> {
    let value = rounded(value)?;
    if value < f64::from(i16::MIN) || value > f64::from(i16::MAX) {
        return Err(Error::Numeric);
    }
    Ok(value as i16)
}

pub fn encode_input(value: f64) -> Result<i64, Error> {
    let value = rounded(value)?;
    // i64::MAX rounds UP to 2^63 as f64; an inclusive cast-based check is wrong.
    if !(-((1u64 << 63) as f64)..(1u64 << 63) as f64).contains(&value) {
        return Err(Error::Numeric);
    }
    Ok(value as i64)
}

/// Defined ring addition, then two's-complement signed interpretation.
pub fn reconstruct_signed(left: u64, right: u64) -> i64 {
    left.wrapping_add(right) as i64
}

/// Rust's integer-to-float conversion rounds to nearest, ties to even.
/// Do not move the division ahead of this f32 conversion.
pub fn rescale_signed(value: i64) -> f32 {
    (value as f32) / 4_194_304.0_f32
}

/// Immutable row-major signed-i16 weights with conservative public input bounds.
/// All row sums `sum(abs(w[j]) * bound[j])` must be strictly below 2^63.
/// Bounds are on quantized inputs, not floats and not uniform worker shares.
pub struct I16Matrix {
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) weights: Vec<i16>,
    pub(super) input_bounds: Vec<u64>,
    pub(super) output_bounds: Vec<u64>,
    pub(super) digest: [u8; 32],
}

impl I16Matrix {
    fn validate_shape(rows: usize, columns: usize, length: usize) -> Result<(), Error> {
        if rows == 0 || columns == 0 || rows > MAX_DIMENSION || columns > MAX_DIMENSION {
            return Err(Error::Limit);
        }
        if rows.checked_mul(columns) != Some(length) {
            return Err(Error::Shape);
        }
        Ok(())
    }

    pub fn from_encoded(
        rows: usize,
        columns: usize,
        weights: &[i16],
        input_bounds: &[u64],
    ) -> Result<Self, Error> {
        Self::validate_encoded(rows, columns, weights, input_bounds)?;
        Self::from_validated_owned(
            rows,
            columns,
            payload::copy(weights)?,
            payload::copy(input_bounds)?,
        )
    }

    fn validate_encoded(
        rows: usize,
        columns: usize,
        weights: &[i16],
        input_bounds: &[u64],
    ) -> Result<(), Error> {
        Self::validate_shape(rows, columns, weights.len())?;
        if input_bounds.len() != columns {
            return Err(Error::Shape);
        }
        if input_bounds.iter().any(|&b| b > 1u64 << 63) {
            return Err(Error::Bound);
        }
        // Validate every public row before allocating owned weights/bounds.
        for row in weights.chunks_exact(columns) {
            if row_bound(row, input_bounds) >= 1u128 << 63 {
                return Err(Error::Bound);
            }
        }
        Ok(())
    }

    /// Takes an exactly sized, already allocated import buffer. This retains the
    /// same numeric validation and digest as `from_encoded`, without another
    /// complete weight copy. Spare vector capacity is not an unpriced owner.
    pub fn from_encoded_owned(
        rows: usize,
        columns: usize,
        weights: Vec<i16>,
        input_bounds: Vec<u64>,
    ) -> Result<Self, Error> {
        Self::validate_encoded(rows, columns, &weights, &input_bounds)?;
        Self::from_validated_owned(rows, columns, weights, input_bounds)
    }

    fn from_validated_owned(
        rows: usize,
        columns: usize,
        weights: Vec<i16>,
        input_bounds: Vec<u64>,
    ) -> Result<Self, Error> {
        if weights.capacity() != weights.len() || input_bounds.capacity() != input_bounds.len() {
            return Err(Error::Limit);
        }
        let output_bounds = payload::collect(
            weights
                .chunks_exact(columns)
                .map(|row| row_bound(row, &input_bounds) as u64),
        )?;
        let mut hash = Sha256::new();
        hash.update(b"pllm.gkm.matrix.v1");
        hash.update(NUMERIC_PROFILE);
        hash.update((rows as u64).to_le_bytes());
        hash.update((columns as u64).to_le_bytes());
        for &bound in &input_bounds {
            hash.update(bound.to_le_bytes());
        }
        for &weight in &weights {
            hash.update(weight.to_le_bytes());
        }
        Ok(Self {
            rows,
            columns,
            weights,
            input_bounds,
            output_bounds,
            digest: hash.finalize().into(),
        })
    }

    pub fn from_f64(
        rows: usize,
        columns: usize,
        weights: &[f64],
        input_bounds: &[u64],
    ) -> Result<Self, Error> {
        Self::validate_shape(rows, columns, weights.len())?;
        if input_bounds.len() != columns {
            return Err(Error::Shape);
        }
        if input_bounds.iter().any(|&b| b > 1u64 << 63) {
            return Err(Error::Bound);
        }
        // Complete numeric/bound admission before allocating the import buffer.
        for row in weights.chunks_exact(columns) {
            let mut bound = 0u128;
            for (&weight, &input_bound) in row.iter().zip(input_bounds) {
                bound +=
                    u128::from(encode_weight(weight)?.unsigned_abs()) * u128::from(input_bound);
            }
            if bound >= 1u128 << 63 {
                return Err(Error::Bound);
            }
        }
        let mut encoded = payload::capacity(weights.len())?;
        for &weight in weights {
            encoded.push(encode_weight(weight)?);
        }
        Self::from_encoded(rows, columns, &encoded, input_bounds)
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Exact scalar modular kernel. No signed overflow or float accumulation.
    /// This operator alone does not verify or accept a client result.
    pub fn multiply_share(&self, share: &[u64]) -> Result<Zeroizing<Vec<u64>>, Error> {
        if share.len() != self.columns {
            return Err(Error::Shape);
        }
        Ok(Zeroizing::new(payload::collect(
            self.weights.chunks_exact(self.columns).map(|row| {
                row.iter().zip(share).fold(0u64, |sum, (&w, &x)| {
                    sum.wrapping_add((i64::from(w) as u64).wrapping_mul(x))
                })
            }),
        )?))
    }
}

fn row_bound(row: &[i16], bounds: &[u64]) -> u128 {
    // Public geometry limits make this sum < 2^87, well inside u128.
    row.iter()
        .zip(bounds)
        .map(|(&w, &b)| u128::from(w.unsigned_abs()) * u128::from(b))
        .sum()
}
