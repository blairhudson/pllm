//! Exact rounded removal of one odd coefficient limb, using caller-owned buffers.
//! Public, variable-time algebra only: no transforms, ciphertexts or secret issuer.

use super::modular::{InverseError, inverse, mul, sub};
use super::{CoefficientRing, LayoutError};

/// One retained modulus q and one removed modulus p, of equal degree.
/// Coprime odd composites are legal; no primality, roots or HE profile is implied.
/// Construction and application allocate no storage and retain no buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoundedLimbDrop {
    retained: CoefficientRing,
    removed: CoefficientRing,
    inverse_p: u64,
}

impl RoundedLimbDrop {
    /// Uses validated ring bounds (legacy 4096, or explicitly public 8192).
    /// Degree mismatch is InvalidDegree; noncoprime moduli are InvalidModulus.
    /// The extracted inverse's iteration-cap failure maps to Overflow.
    pub fn new(
        retained: CoefficientRing,
        removed: CoefficientRing,
    ) -> Result<Self, LayoutError> {
        if retained.degree() != removed.degree() {
            return Err(LayoutError::InvalidDegree);
        }
        let inverse_p = inverse(removed.modulus() % retained.modulus(), retained.modulus())
            .map_err(|error| match error {
                InverseError::NotCoprime => LayoutError::InvalidModulus,
                InverseError::Capacity => LayoutError::Overflow,
            })?;
        Ok(Self { retained, removed, inverse_p })
    }

    /// For x in [0,p*q) represented by (vq,vp), returns
    /// floor((x+(p-1)/2)/p) mod q coefficientwise. Odd p excludes half ties.
    /// All three lengths and both complete canonical inputs are checked before
    /// any write. Atomicity covers this one retained-limb projection only:
    /// callers projecting several limbs must preflight the whole operation.
    pub fn apply_into(
        self,
        vq: &[u64],
        vp: &[u64],
        output: &mut [u64],
    ) -> Result<(), LayoutError> {
        let degree = self.retained.degree();
        if vq.len() != degree || vp.len() != degree || output.len() != degree {
            return Err(LayoutError::BufferLength);
        }
        self.retained.canonical(vq)?;
        self.removed.canonical(vp)?;
        let q = self.retained.modulus();
        let p = self.removed.modulus();
        let half = (p - 1) / 2;
        for ((&aq, &ap), out) in vq.iter().zip(vp).zip(output) {
            // Reduce the centered remainder BEFORE subtraction: p may exceed q.
            // Negative center is represented modulo q without a signed cast.
            let center_q = if ap <= half {
                ap % q
            } else {
                sub(0, (p - ap) % q, q)
            };
            *out = mul(sub(aq, center_q, q), self.inverse_p, q);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
