//! Arithmetic for complete-row energy through coordinatewise checked safe-zero.
//!
//! This module neither validates provenance nor mints certificates. Callers must
//! obtain B from their existing complete, ordered producer-row ledger and bind
//! any retained result to the exact corresponding checked output row. In
//! particular, source statistics and attention-only proofs are not substitutes
//! for a general input-row certificate.
use super::Error;

/// The scalar interval alone proves n*C^2. Keep checked overflow even when a
/// smaller producer bound would make that product unnecessary mathematically:
/// this preserves the original outer constructor's admission behavior.
pub(super) fn cap_energy(width: usize, bound: i64) -> Result<u128, Error> {
    if width == 0 {
        return Err(Error::Shape);
    }
    let cap = u128::try_from(bound).map_err(|_| Error::Bounds)?;
    cap.checked_mul(cap)
        .and_then(|square| square.checked_mul(width as u128))
        .ok_or(Error::Bounds)
}

/// If sum(x_i^2) <= B and y_i is x_i inside inclusive [-C,C], zero otherwise,
/// then sum(y_i^2) <= min(B,n*C^2), including rejected requests. No rounding is
/// involved. Only strict improvements need retained certificate metadata.
///
/// These numeric arguments are NOT evidence that either premise is true.
pub(super) fn strict_improvement(certified_energy: u128, cap_energy: u128) -> Option<u128> {
    (certified_energy < cap_energy).then_some(certified_energy.min(cap_energy))
}

#[cfg(test)]
#[path = "checked_row_energy_tests.rs"]
mod tests;
