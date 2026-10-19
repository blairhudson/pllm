//! Complete-row compact schedules. Equations/proofs share the legacy builders;
//! only explicit constructors select compact Boolean conversion/predicates.
use super::*;

impl Program {
    // Bind exact public proof obligations even when two energy bounds happen to
    // produce the same rounded intermediate envelope. No secret data is hashed.
    fn bind_compact_row(mut self, kind: u8, input: Option<u128>, output: Option<u128>) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"puma.compact.complete-row-proof.q18.v1");
        hash.update(self.digest);
        hash.update([kind]);
        for certificate in [input, output] {
            hash.update([u8::from(certificate.is_some())]);
            if let Some(bound) = certificate {
                hash.update(bound.to_le_bytes());
            }
        }
        self.digest = hash.finalize().into();
        self
    }

    pub fn softmax_compact(rows: usize, width: usize, input: Interval) -> Result<Self, Error> {
        let program = Self::softmax_schedule(rows, width, input, false, false, true)?.0;
        Ok(program.bind_compact_row(0, None, None))
    }

    /// Guard clipped tails BEFORE powers, preserving the strict -14 cutoff.
    pub fn softmax_guarded_compact(
        rows: usize,
        width: usize,
        input: Interval,
    ) -> Result<Self, Error> {
        let program = Self::softmax_schedule(rows, width, input, true, false, true)?.0;
        Ok(program.bind_compact_row(1, None, None))
    }

    /// Nonnegative complete-row encoded mass, under the original reciprocal
    /// partition proof covering every admitted stochastic rounding choice.
    pub fn softmax_certified_compact(
        rows: usize,
        width: usize,
        input: Interval,
    ) -> Result<(Self, u64), Error> {
        let (program, mass) = Self::softmax_schedule(rows, width, input, true, true, true)?;
        let mass = mass.ok_or(Error::Binding)?;
        Ok((
            program.bind_compact_row(2, None, Some(u128::from(mass))),
            mass,
        ))
    }

    pub fn layer_norm_compact(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        beta: &[i64],
    ) -> Result<Self, Error> {
        Ok(
            Self::layer_norm_schedule(rows, width, input, gamma, beta, true)?
                .bind_compact_row(3, None, None),
        )
    }

    /// Conservative complete output-row energy, including gamma/beta and every
    /// rounding error. This uses the proved final interval, not an RMS identity;
    /// LayerNorm centering and epsilon-two arithmetic are unchanged.
    pub fn layer_norm_certified_compact(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        beta: &[i64],
    ) -> Result<(Self, u128), Error> {
        let program = Self::layer_norm_schedule(rows, width, input, gamma, beta, true)?;
        let bounds = program.output_bounds();
        let magnitude = u128::from(
            bounds
                .lower()
                .unsigned_abs()
                .max(bounds.upper().unsigned_abs()),
        );
        let energy = magnitude
            .checked_mul(magnitude)
            .and_then(|square| square.checked_mul(width as u128))
            .ok_or(Error::Bounds)?;
        Ok((program.bind_compact_row(4, None, Some(energy)), energy))
    }

    pub fn rms_norm_compact(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
    ) -> Result<Self, Error> {
        let (program, _) = Self::rms_norm_proof(rows, width, input, gamma, false, None, true)?;
        Ok(program.bind_compact_row(5, None, None))
    }

    pub fn rms_norm_certified_compact(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
    ) -> Result<(Self, u128), Error> {
        let (program, energy) = Self::rms_norm_proof(rows, width, input, gamma, true, None, true)?;
        Ok((program.bind_compact_row(6, None, Some(energy)), energy))
    }

    /// `squared_norm` is a caller-proved bound for EACH complete input row,
    /// not for a tile or the whole batch. It is never opened/checked privately.
    pub fn rms_norm_with_input_norm_compact(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        squared_norm: u128,
    ) -> Result<(Self, u128), Error> {
        let (program, energy) =
            Self::rms_norm_proof(rows, width, input, gamma, true, Some(squared_norm), true)?;
        Ok((
            program.bind_compact_row(7, Some(squared_norm), Some(energy)),
            energy,
        ))
    }
}
