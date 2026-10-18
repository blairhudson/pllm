//! Bounded local ring arithmetic and standalone source rescaling.
//! Public coefficients never select private branches or reveal share values.

use super::{boolean, InputBounds, Interval, PartyRuntime, Phase};
use crate::puma::prss::blocks;
use crate::puma::*;
use sha2::{Digest, Sha256};

pub const AFFINE_PROFILE: &str = "puma.spu-a42c261.local-affine.ring64.replicated3.v1";
pub const RESCALE_PROFILE: &str = "puma.spu-a42c261.bound-truncapr.ring64.replicated3.v1";
pub const XOR_PROFILE: &str = "puma.spu-a42c261.local-xor.ring64.replicated3.v1";

/// Exact integer coefficients, with no implicit fixed-point rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Affine {
    pub left: i64,
    pub right: i64,
    pub constant: i64,
}

impl Affine {
    pub fn output_bounds(self, bounds: InputBounds) -> Result<Interval, Error> {
        let term = |coefficient: i64, value: Interval| {
            let a = i128::from(coefficient) * i128::from(value.lower);
            let b = i128::from(coefficient) * i128::from(value.upper);
            (a.min(b), a.max(b))
        };
        let (a, b) = term(self.left, bounds.left);
        let (c, d) = term(self.right, bounds.right);
        let lower = a
            .checked_add(c)
            .and_then(|v| v.checked_add(i128::from(self.constant)))
            .ok_or(Error::Bounds)?;
        let upper = b
            .checked_add(d)
            .and_then(|v| v.checked_add(i128::from(self.constant)))
            .ok_or(Error::Bounds)?;
        Interval::new(
            i64::try_from(lower).map_err(|_| Error::Bounds)?,
            i64::try_from(upper).map_err(|_| Error::Bounds)?,
        )
    }
}

/// Complete stochastic output hull, conditional on the compiler-proved input
/// interval. No private runtime value is reconstructed to establish membership.
pub fn rescaled_bounds(bounds: Interval, shift: u8) -> Result<Interval, Error> {
    if !(1..=62).contains(&shift) || bounds.lower < INPUT_MIN || bounds.upper >= INPUT_MAX_EXCLUSIVE
    {
        return Err(Error::Bounds);
    }
    let unit = 1_i64 << shift;
    let lower = bounds.lower.div_euclid(unit);
    let upper = bounds.upper.div_euclid(unit) + i64::from(bounds.upper.rem_euclid(unit) != 0);
    Interval::new(lower, upper)
}

fn validate(source: &PartyRuntime, value: &ReplicatedShare) -> Result<(), Error> {
    if value.party != source.party() {
        return Err(Error::Role);
    }
    if value.operation.context != source.context() {
        return Err(Error::Binding);
    }
    if value.first.len() != value.operation.lanes || value.second.len() != value.operation.lanes {
        return Err(Error::Shape);
    }
    source.source.limits.core.admit(value.operation.lanes)?;
    Ok(())
}

fn reserve(source: &PartyRuntime, additional: usize) -> Result<(), Error> {
    let usage = source.payload_usage();
    if usage
        .live_bytes()
        .checked_add(additional)
        .is_none_or(|v| v > usage.limit_bytes())
    {
        return Err(Error::Bounds);
    }
    Ok(())
}

impl PartyRuntime {
    /// Source XorBB followed by XorBP. The public word is XORed into each
    /// component as in the three-party source; nominal-width padding survives.
    pub fn boolean_xor(
        &mut self,
        left: &mut boolean::Share,
        mut right: Option<&mut boolean::Share>,
        public: u64,
    ) -> Result<boolean::Share, Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            validate(self, &left.value)?;
            let n = left.value.operation.lanes;
            if let Some(value) = &right {
                validate(self, &value.value)?;
                if value.value.operation.lanes != n {
                    return Err(Error::Shape);
                }
            }
            let bits = left
                .bits()
                .max(right.as_ref().map_or(0, |v| v.bits()))
                .max((64 - public.leading_zeros()) as u8);
            if bits > 64 {
                return Err(Error::Bounds);
            }
            reserve(self, (if right.is_some() { 6 } else { 4 }) * n * 8)?;
            let mut operation = Operation::new(
                self.context(),
                self.source.runtime.next_sequence(),
                Party::P0,
                n,
                1,
                self.source.limits.core,
            )?;
            let mut hash = Sha256::new();
            hash.update(XOR_PROFILE.as_bytes());
            hash.update(self.source.setup);
            hash.update(left.binding());
            hash.update([left.bits(), bits, u8::from(right.is_some())]);
            if let Some(value) = &right {
                hash.update(value.binding());
                hash.update([value.bits()]);
            }
            hash.update(public.to_le_bytes());
            operation.consumer = Some(hash.finalize().into());
            self.source.runtime.consume_ordinal(operation)?;
            let usage = self.payload_usage();
            left.value.charge_to(&usage)?;
            if let Some(value) = &mut right {
                value.value.charge_to(&usage)?;
            }
            let mut first = usage.zeros(n)?;
            let mut second = usage.zeros(n)?;
            for i in 0..n {
                first[i] = left.first()[i] ^ right.as_ref().map_or(0, |r| r.first()[i]) ^ public;
                second[i] = left.second()[i] ^ right.as_ref().map_or(0, |r| r.second()[i]) ^ public;
            }
            Ok(boolean::Share {
                value: ReplicatedShare {
                    operation,
                    party: self.party(),
                    first,
                    second,
                },
                bits,
            })
        })();
        self.checked(result)
    }

    /// Retains the input owners and returns a separately charged output pair.
    /// Mutable borrows attach this runtime's payload lease before arithmetic;
    /// the share contents stay unchanged. Public interval arithmetic proves
    /// absence of signed overflow, conditional on the caller/compiler bounds.
    pub fn affine(
        &mut self,
        left: &mut ReplicatedShare,
        mut right: Option<&mut ReplicatedShare>,
        coefficients: Affine,
        bounds: InputBounds,
    ) -> Result<(ReplicatedShare, Interval), Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            validate(self, left)?;
            if let Some(value) = &right {
                validate(self, value)?;
                if value.operation.lanes != left.operation.lanes {
                    return Err(Error::Shape);
                }
            } else if coefficients.right != 0 || bounds.right != Interval::new(0, 0)? {
                return Err(Error::Bounds);
            }
            let output_bounds = coefficients.output_bounds(bounds)?;
            let n = left.operation.lanes;
            // Includes uncharged caller inputs as well as the new output pair.
            reserve(self, (if right.is_some() { 6 } else { 4 }) * n * 8)?;
            let mut operation = Operation::new(
                self.context(),
                self.source.runtime.next_sequence(),
                Party::P0,
                n,
                1,
                self.source.limits.core,
            )?;
            let mut hash = Sha256::new();
            hash.update(AFFINE_PROFILE.as_bytes());
            hash.update(self.source.setup);
            hash.update(left.operation.digest());
            hash.update([u8::from(right.is_some())]);
            if let Some(value) = &right {
                hash.update(value.operation.digest());
            }
            for value in [
                coefficients.left,
                coefficients.right,
                coefficients.constant,
                bounds.left.lower,
                bounds.left.upper,
                bounds.right.lower,
                bounds.right.upper,
            ] {
                hash.update(value.to_le_bytes());
            }
            operation.consumer = Some(hash.finalize().into());
            self.source.runtime.consume_ordinal(operation)?;
            let usage = self.payload_usage();
            left.charge_to(&usage)?;
            if let Some(value) = &mut right {
                value.charge_to(&usage)?;
            }
            let mut first = usage.zeros(n)?;
            let mut second = usage.zeros(n)?;
            for i in 0..n {
                let term = |x: u64, y: u64| {
                    x.wrapping_mul(coefficients.left as u64)
                        .wrapping_add(y.wrapping_mul(coefficients.right as u64))
                };
                first[i] = term(left.first[i], right.as_ref().map_or(0, |r| r.first[i]));
                second[i] = term(left.second[i], right.as_ref().map_or(0, |r| r.second[i]));
                if self.party() == Party::P0 {
                    first[i] = first[i].wrapping_add(coefficients.constant as u64);
                }
                if self.party() == Party::P2 {
                    second[i] = second[i].wrapping_add(coefficients.constant as u64);
                }
            }
            Ok((
                ReplicatedShare {
                    operation,
                    party: self.party(),
                    first,
                    second,
                },
                output_bounds,
            ))
        })();
        self.checked(result)
    }

    /// Starts exactly one source TruncAPr, preserving the current source cursor.
    /// Public products can use local affine multiplication then this rescale;
    /// they do not incur an extra private MulAA or silently shift shares locally.
    pub fn begin_rescale(
        &mut self,
        mut input: ReplicatedShare,
        shift: u8,
        bounds: Interval,
    ) -> Result<(Operation, [Option<Message>; 2]), Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            validate(self, &input)?;
            rescaled_bounds(bounds, shift)?;
            let n = input.operation.lanes;
            reserve(
                self,
                self.source.limits.core.admit(n)? - FIXED_RESERVE_BYTES,
            )?;
            let q = blocks(n)?;
            self.source.work(4 * q + n as u64 + 1)?;
            self.source
                .cursors
                .private_blocks
                .checked_add(n as u64)
                .ok_or(Error::Bounds)?;
            let mut hash = Sha256::new();
            hash.update(RESCALE_PROFILE.as_bytes());
            hash.update(input.operation.digest());
            hash.update(bounds.lower.to_le_bytes());
            hash.update(bounds.upper.to_le_bytes());
            hash.update([shift]);
            let mut operation = self.source.plan(n, shift)?;
            operation.consumer = Some(hash.finalize().into());
            self.source.pending = Some(operation);
            input.operation = operation;
            let messages = self.source.begin(input)?;
            self.phase = Phase::Truncating(operation);
            Ok((operation, messages))
        })();
        self.checked(result)
    }
}

#[cfg(test)]
#[path = "arithmetic_tests.rs"]
mod tests;
