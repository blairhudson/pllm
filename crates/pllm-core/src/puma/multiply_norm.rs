//! Public whole-dot-row energy contract for the original ring64 MatMulAA.
//!
//! This is a caller obligation, not a verification of secret shares: every left
//! entry must satisfy `bounds.left`, every right entry `bounds.right`, and every
//! left row / right column (the two inner-axis vectors in each dot product) must
//! have squared integer energy at most its declared maximum. All obligations
//! hold simultaneously for the intended, unwrapped signed integers. Never derive
//! this ticket from private observations. A compiler must authenticate source and
//! operation provenance separately before admitting these public promises.

use super::*;

/// Bounded public Cauchy-Schwarz proof, with no arbitrary raw-range constructor.
/// Construction uses only checked integer arithmetic, before any random draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowEnergyProof {
    inner: usize,
    bounds: InputBounds,
    left_squared_max: u128,
    right_squared_max: u128,
    raw: Interval,
}

impl RowEnergyProof {
    pub fn new(
        inner: usize,
        bounds: InputBounds,
        left_squared_max: u128,
        right_squared_max: u128,
    ) -> Result<Self, Error> {
        if inner == 0 || inner > crate::puma::MAX_LANES {
            return Err(Error::Bounds);
        }
        // Reject inconsistent scalar/energy promises, including overflow of the
        // minimum possible energy. Upper scalar bounds remain simultaneous
        // caller obligations; they are not replaced by the energy promise.
        for (interval, energy) in [
            (bounds.left, left_squared_max),
            (bounds.right, right_squared_max),
        ] {
            let minimum = if interval.lower > 0 {
                interval.lower as u128
            } else if interval.upper < 0 {
                i128::from(interval.upper).unsigned_abs()
            } else {
                0
            };
            let minimum_energy = minimum
                .checked_mul(minimum)
                .and_then(|v| v.checked_mul(inner as u128))
                .ok_or(Error::Bounds)?;
            if minimum_energy > energy {
                return Err(Error::Bounds);
            }
        }
        let squared = left_squared_max
            .checked_mul(right_squared_max)
            .ok_or(Error::Bounds)?;
        let radius = i64::try_from(ceil_sqrt(squared)).map_err(|_| Error::Bounds)?;
        let mut raw = Interval::new(-radius, radius)?;
        // Both promises hold simultaneously. A loose public energy certificate
        // must never widen an already admitted scalar-product hull. A rejected
        // scalar hull supplies no bound; Cauchy-Schwarz remains independently
        // sufficient in that case, including sparse full-width contractions.
        if let Ok(scalar) = bounds.dot_product(inner) {
            raw = Interval::new(raw.lower.max(scalar.lower), raw.upper.min(scalar.upper))?;
        }
        Ok(Self {
            inner,
            bounds,
            left_squared_max,
            right_squared_max,
            raw,
        })
    }

    pub fn inner(self) -> usize {
        self.inner
    }

    pub fn input_bounds(self) -> InputBounds {
        self.bounds
    }

    pub fn squared_energy_maxima(self) -> [u128; 2] {
        [self.left_squared_max, self.right_squared_max]
    }

    /// Cauchy-Schwarz envelope intersected with any admitted scalar-product hull.
    pub fn raw_bounds(self) -> Interval {
        self.raw
    }

    /// Inclusive envelope covering every stochastic Q18 rounding outcome.
    /// Rejects raw bounds outside the existing TruncAPr quarter-gap contract.
    pub fn output_bounds(self) -> Result<Interval, Error> {
        arithmetic::rescaled_bounds(self.raw, FRACTION_BITS)
    }

    pub fn digest(self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"pllm.puma.public-row-energy-proof.v1");
        hash.update((self.inner as u64).to_le_bytes());
        hash.update(self.left_squared_max.to_le_bytes());
        hash.update(self.right_squared_max.to_le_bytes());
        for interval in [self.bounds.left, self.bounds.right, self.raw] {
            hash.update(interval.lower.to_le_bytes());
            hash.update(interval.upper.to_le_bytes());
        }
        hash.finalize().into()
    }
}

// Binary search with division rather than potentially overflowing squares.
fn ceil_sqrt(value: u128) -> u128 {
    if value == 0 {
        return 0;
    }
    let (mut low, mut high) = (1_u128, 1_u128 << 64);
    while low < high {
        let mid = low + (high - low) / 2;
        if mid >= value / mid + u128::from(value % mid != 0) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low
}

impl PartyRuntime {
    /// Original MatMulAA followed by one full-dot Q18 TruncAPr, with public
    /// row-energy admission intersected with any admitted scalar-product hull.
    pub fn plan_norm_matmul(
        &mut self,
        shape: MatrixShape,
        proof: RowEnergyProof,
    ) -> Result<Plan, Error> {
        self.plan_norm_matmul_inner(shape, proof, true)
    }

    /// Original raw ring64 MatMulAA, with no rescale or truncation randomness.
    pub fn plan_ring_norm_matmul(
        &mut self,
        shape: MatrixShape,
        proof: RowEnergyProof,
    ) -> Result<Plan, Error> {
        self.plan_norm_matmul_inner(shape, proof, false)
    }

    fn plan_norm_matmul_inner(
        &mut self,
        shape: MatrixShape,
        proof: RowEnergyProof,
        rescale: bool,
    ) -> Result<Plan, Error> {
        let result = (|| {
            // Recompute all proof obligations even for an internal malformed
            // ticket. No caller-controlled interval bypass reaches plan_product.
            let checked = RowEnergyProof::new(
                proof.inner,
                proof.bounds,
                proof.left_squared_max,
                proof.right_squared_max,
            )?;
            if checked != proof {
                return Err(Error::Binding);
            }
            if shape.inner != proof.inner {
                return Err(Error::Shape);
            }
            self.plan_product(
                Product::Matrix(shape),
                &[proof.bounds],
                rescale,
                Some((proof.raw, proof.digest())),
            )
        })();
        self.checked(result)
    }
}

#[cfg(test)]
#[path = "multiply_norm_tests.rs"]
mod tests;
