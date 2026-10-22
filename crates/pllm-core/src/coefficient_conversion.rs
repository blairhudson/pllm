//! Bounded deterministic coefficient-phase conversion, not encryption or H2A.
//!
//! Matches `paper/reproduction-contracts/shared_profile_oracle.py`'s
//! `coefficient_h2a` and shares its `down` contract. No HE backend, randomness,
//! sampling, freshness enforcement, privacy, or reconstruction-error bound is
//! supplied. BumbleBee B-N1 remains open; D21's layout corrections are separate.
//!
//! Scalar, allocation-free O(1) workspace and work; no retained buffers or batch
//! admission. Not constant-time. Inputs and results are clear algebra values.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConversionError {
    InvalidBits,
    InvalidModulus,
    /// Exact worst-case `down` numerator exceeds u128. A wider-integer backend
    /// needs a separate implementation; no truncation or parameter substitution.
    ArithmeticCapacity,
    NonCanonical,
    InvalidDeltaMessage,
    PhaseOutOfRange,
}

/// Caller-supplied phases. `delta_message` must be produced by `delta_encode`,
/// not nearest modulus-up (`lift`). Both errors are signed and bounded by q/2.
/// The name `fresh_zero_error` describes the oracle phase, not enforced freshness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoefficientPhaseInputs {
    pub delta_message: u128,
    pub error: i128,
    pub fresh_zero_error: i128,
    pub mask: u128,
}

/// Canonical residues modulo t. These are algebraic outputs, not private shares.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoefficientPhaseOutputs {
    pub receiver: u64,
    pub sender: u64,
}

/// Public scalar domain: 1<=bits<=64, t=2^bits<q, odd q, and
/// `2*t*(q-1)+q <= u128::MAX`. This last bound is an implementation-capacity
/// gate, NOT an approved numeric/HE profile. In particular bits=64 has no
/// admissible q with this exact fixed-width numerator. Oracle q up to 256 bits
/// requires wider arithmetic and is deliberately not silently approximated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoefficientConversion {
    bits: u8,
    t: u128,
    q: u128,
    delta: u128,
}

impl CoefficientConversion {
    pub fn new(bits: u8, odd_modulus: u128) -> Result<Self, ConversionError> {
        if !(1..=64).contains(&bits) {
            return Err(ConversionError::InvalidBits);
        }
        let t = 1_u128 << bits;
        let q = odd_modulus;
        if q <= t || q & 1 == 0 {
            return Err(ConversionError::InvalidModulus);
        }
        // Admit every canonical coefficient, not just a convenient input.
        t.checked_mul(2)
            .and_then(|v| v.checked_mul(q - 1))
            .and_then(|v| v.checked_add(q))
            .ok_or(ConversionError::ArithmeticCapacity)?;
        Ok(Self {
            bits,
            t,
            q,
            delta: q / t,
        })
    }

    pub fn bits(self) -> u8 {
        self.bits
    }

    pub fn odd_modulus(self) -> u128 {
        self.q
    }

    /// Delta*m, Delta=floor(q/t), for a canonical ring-t message. Not lift(m).
    pub fn delta_encode(self, message: u64) -> Result<u128, ConversionError> {
        if u128::from(message) >= self.t {
            return Err(ConversionError::NonCanonical);
        }
        self.delta
            .checked_mul(u128::from(message))
            .ok_or(ConversionError::ArithmeticCapacity)
    }

    /// Shared down contract: nearest(t*(x mod q)/q) mod t. Odd q excludes
    /// exact half ties. Accepts any u128 phase and canonicalizes before scaling.
    /// This deliberately does not reuse signed ties-even fixed-point division,
    /// whose domain and overflow contract differ from modulus conversion.
    pub fn down(self, phase: u128) -> Result<u64, ConversionError> {
        let numerator = self
            .t
            .checked_mul(2)
            .and_then(|v| v.checked_mul(phase % self.q))
            .and_then(|v| v.checked_add(self.q))
            .ok_or(ConversionError::ArithmeticCapacity)?;
        let denominator = self
            .q
            .checked_mul(2)
            .ok_or(ConversionError::ArithmeticCapacity)?;
        Ok(((numerator / denominator) % self.t) as u64)
    }

    fn add(self, a: u128, b: u128) -> Result<u128, ConversionError> {
        a.checked_add(b)
            .map(|v| v % self.q)
            .ok_or(ConversionError::ArithmeticCapacity)
    }

    fn signed_phase(self, error: i128) -> Result<u128, ConversionError> {
        let magnitude = error.unsigned_abs();
        if magnitude > self.q / 2 {
            return Err(ConversionError::PhaseOutOfRange);
        }
        Ok(if error < 0 {
            self.q - magnitude
        } else {
            magnitude
        })
    }

    /// receiver=down(Delta*m+error+fresh_zero_error+mask), sender=-down(mask).
    /// No claim that their sum equals m, or differs by at most one.
    pub fn evaluate(
        self,
        inputs: CoefficientPhaseInputs,
    ) -> Result<CoefficientPhaseOutputs, ConversionError> {
        if inputs.delta_message % self.delta != 0 || inputs.delta_message / self.delta >= self.t {
            return Err(ConversionError::InvalidDeltaMessage);
        }
        if inputs.mask >= self.q {
            return Err(ConversionError::NonCanonical);
        }
        let error = self.signed_phase(inputs.error)?;
        let zero_error = self.signed_phase(inputs.fresh_zero_error)?;
        let phase = self.add(inputs.delta_message, error)?;
        let phase = self.add(phase, zero_error)?;
        let phase = self.add(phase, inputs.mask)?;
        let receiver = self.down(phase)?;
        let sender = ((self.t - u128::from(self.down(inputs.mask)?)) % self.t) as u64;
        Ok(CoefficientPhaseOutputs { receiver, sender })
    }
}

#[cfg(test)]
#[path = "coefficient_conversion/tests.rs"]
mod tests;
