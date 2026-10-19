//! Explicit compact schedules; legacy constructors never select these paths.
use super::*;

impl Program {
    /// Positive reciprocal with compact exponent extraction only. Factor/reversal
    /// widths, PPA conversion and every stochastic arithmetic equation are kept.
    pub fn reciprocal_compact(lanes: usize, input: Interval) -> Result<Self, Error> {
        let mut b = Builder::new(lanes, input)?;
        b.0.numeric_profile = COMPACT_NONLINEAR_PROFILE;
        let output = numeric::reciprocal(&mut b, Value(0), input)?;
        b.finish(output)
    }

    /// Positive rsqrt with the original 35-bit normalization/compensation widths.
    pub fn rsqrt_compact(lanes: usize, input: Interval) -> Result<Self, Error> {
        let mut b = Builder::new(lanes, input)?;
        b.0.numeric_profile = COMPACT_NONLINEAR_PROFILE;
        let output = numeric::rsqrt(&mut b, Value(0), input)?;
        b.finish(output)
    }

    /// Smallest native bucket whose signed range contains the entire interval.
    /// Equality modulo a smaller power of two is NOT a signed-order proof.
    pub fn signed_width(input: Interval) -> u8 {
        for bits in [8, 16, 32, 64] {
            let half = 1_i128 << (bits - 1);
            if i128::from(input.lower()) >= -half && i128::from(input.upper()) < half {
                return bits;
            }
        }
        unreachable!("every i64 interval fits signed64")
    }

    /// Prove mathematical subtraction fits ring64 BEFORE emitting arithmetic.
    /// A full64 MSB of a wrapped difference is never a valid fallback.
    pub fn signed_difference_bounds(left: Interval, right: Interval) -> Result<Interval, Error> {
        Interval::new(
            i64::try_from(i128::from(left.lower()) - i128::from(right.upper()))
                .map_err(|_| Error::Bounds)?,
            i64::try_from(i128::from(left.upper()) - i128::from(right.lower()))
                .map_err(|_| Error::Bounds)?,
        )
    }

    /// Source-proven signed predicate, not a membership guard. Incoming values
    /// outside `input` may alias; check them with full-word checked_range first.
    pub fn negative_compact(lanes: usize, input: Interval) -> Result<Self, Error> {
        let mut b = Builder::new(lanes, input)?;
        let bit = b.sign_compact(b.input())?;
        let out = b.from_bit(bit)?;
        b.finish(out)
    }

    /// Original guarded polynomial equations and exact tails, with compact
    /// predicates justified by the incoming public proof. This does not clip
    /// accepted outputs or establish private membership in that input interval.
    pub fn activation_guarded_compact(
        kind: Activation,
        lanes: usize,
        input: Interval,
    ) -> Result<Self, Error> {
        Self::activation_impl(kind, lanes, input, true, true)
    }
}

impl Builder {
    pub fn sign_compact(&mut self, input: Value) -> Result<Value, Error> {
        let width = Program::signed_width(self.node(input, false)?.bounds);
        let bit = self.push(
            Instruction::CompactSign(input, width),
            Interval::new(0, 1)?,
            true,
        )?;
        self.0.numeric_profile = COMPACT_NONLINEAR_PROFILE;
        Ok(bit)
    }

    /// Strict comparison; equality preserves the earlier candidate in selection.
    /// Public proof failure rejects before even appending a subtraction node.
    pub fn less_than_compact(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        Program::signed_difference_bounds(
            self.node(left, false)?.bounds,
            self.node(right, false)?.bounds,
        )?;
        let delta = self.affine(
            left,
            Some(right),
            Affine {
                left: 1,
                right: -1,
                constant: 0,
            },
        )?;
        self.sign_compact(delta)
    }
}
