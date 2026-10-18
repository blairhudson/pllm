//! Bounded local ring arithmetic and standalone source rescaling.
//! Public coefficients never select private branches or reveal share values.

use super::{boolean, InputBounds, Interval, PartyRuntime, Phase};
use crate::puma::prss::blocks;
use crate::puma::*;
use sha2::{Digest, Sha256};

pub const AFFINE_PROFILE: &str = "puma.spu-a42c261.local-affine.ring64.replicated3.v1";
pub const RESCALE_PROFILE: &str = "puma.spu-a42c261.bound-truncapr.ring64.replicated3.v1";
pub const XOR_PROFILE: &str = "puma.spu-a42c261.local-xor.ring64.replicated3.v1";
pub const BOOLEAN_MAP_PROFILE: &str = "puma.public-bit-linear-map.replicated3.v1";
pub const ROW_MAP_PROFILE: &str = "puma.public-row-linear-map.replicated3.v1";
pub const LANE_AFFINE_PROFILE: &str = "puma.public-lane-affine.replicated3.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowMap {
    Sum,
    Rotate(usize),
    Broadcast,
}

impl RowMap {
    pub fn output_bounds(self, width: usize, bounds: Interval) -> Result<Interval, Error> {
        if !(1..=128).contains(&width) {
            return Err(Error::Bounds);
        }
        match self {
            Self::Sum => Affine {
                left: width as i64,
                right: 0,
                constant: 0,
            }
            .output_bounds(InputBounds {
                left: bounds,
                right: Interval::new(0, 0)?,
            }),
            Self::Rotate(n) if n < width => Ok(bounds),
            Self::Broadcast => Ok(bounds),
            _ => Err(Error::Bounds),
        }
    }
}

/// Public XOR-linear maps. Each local component uses the same fixed map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitMap {
    LowBits(u8),
    ShiftRight(u8),
    ShiftLeft(u8),
    ReverseLowBits(u8),
    FoldPairs,
    Parity(u64),
}

impl BitMap {
    pub(super) fn width(self, input: u8) -> Result<u8, Error> {
        if input > 64 {
            return Err(Error::Bounds);
        }
        Ok(match self {
            Self::LowBits(n) | Self::ReverseLowBits(n) if (1..=64).contains(&n) => n,
            Self::ShiftRight(n) if n < 64 => input.saturating_sub(n),
            Self::ShiftLeft(n) if n < 64 => input.saturating_add(n).min(64),
            Self::FoldPairs => input.div_ceil(2),
            Self::Parity(_) => 1,
            _ => return Err(Error::Bounds),
        })
    }

    pub(super) fn apply(self, word: u64) -> u64 {
        match self {
            Self::LowBits(n) => word & (u64::MAX >> (64 - n)),
            Self::ShiftRight(n) => word >> n,
            Self::ShiftLeft(n) => word << n,
            Self::ReverseLowBits(n) => word.reverse_bits() >> (64 - n),
            Self::FoldPairs => {
                let mut out = 0;
                for i in 0..32 {
                    out |= (((word >> (2 * i)) ^ (word >> (2 * i + 1))) & 1) << i;
                }
                out
            }
            Self::Parity(mask) => u64::from((word & mask).count_ones() & 1),
        }
    }

    pub(super) fn hash(self, hash: &mut Sha256) {
        let (tag, parameter) = match self {
            Self::LowBits(n) => (0, u64::from(n)),
            Self::ShiftRight(n) => (1, u64::from(n)),
            Self::ShiftLeft(n) => (2, u64::from(n)),
            Self::ReverseLowBits(n) => (3, u64::from(n)),
            Self::FoldPairs => (4, 0),
            Self::Parity(mask) => (5, mask),
        };
        hash.update([tag]);
        hash.update(parameter.to_le_bytes());
    }
}

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
    pub fn row_map(
        &mut self,
        input: &mut ReplicatedShare,
        width: usize,
        map: RowMap,
        bounds: Interval,
    ) -> Result<ReplicatedShare, Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            validate(self, input)?;
            map.output_bounds(width, bounds)?;
            let n = input.operation.lanes;
            let output_lanes = match map {
                RowMap::Broadcast => n.checked_mul(width).ok_or(Error::Bounds)?,
                RowMap::Sum if n % width == 0 => n / width,
                RowMap::Rotate(_) if n % width == 0 => n,
                _ => return Err(Error::Shape),
            };
            self.source.limits.core.admit(output_lanes)?;
            reserve(self, 2 * (n + output_lanes) * 8)?;
            let mut operation = Operation::new(
                self.context(),
                self.source.runtime.next_sequence(),
                Party::P0,
                output_lanes,
                1,
                self.source.limits.core,
            )?;
            let mut hash = Sha256::new();
            hash.update(ROW_MAP_PROFILE.as_bytes());
            hash.update(self.source.setup);
            hash.update(input.operation.digest());
            hash.update((width as u64).to_le_bytes());
            hash.update(
                match map {
                    RowMap::Sum => u64::MAX,
                    RowMap::Broadcast => u64::MAX - 1,
                    RowMap::Rotate(n) => n as u64,
                }
                .to_le_bytes(),
            );
            hash.update(bounds.lower.to_le_bytes());
            hash.update(bounds.upper.to_le_bytes());
            operation.consumer = Some(hash.finalize().into());
            self.source.runtime.consume_ordinal(operation)?;
            let usage = self.payload_usage();
            input.charge_to(&usage)?;
            let mut first = usage.zeros(output_lanes)?;
            let mut second = usage.zeros(output_lanes)?;
            for (source, target) in [(&input.first, &mut first), (&input.second, &mut second)] {
                match map {
                    RowMap::Sum => {
                        for (row, out) in source.chunks_exact(width).zip(target.iter_mut()) {
                            *out = row.iter().fold(0_u64, |s, &v| s.wrapping_add(v));
                        }
                    }
                    RowMap::Broadcast => {
                        for (&word, out) in source.iter().zip(target.chunks_exact_mut(width)) {
                            out.fill(word);
                        }
                    }
                    RowMap::Rotate(offset) => {
                        for (row, out) in source
                            .chunks_exact(width)
                            .zip(target.chunks_exact_mut(width))
                        {
                            for (i, word) in out.iter_mut().enumerate() {
                                *word = row[(i + offset) % width];
                            }
                        }
                    }
                }
            }
            Ok(ReplicatedShare {
                operation,
                party: self.party(),
                first,
                second,
            })
        })();
        self.checked(result)
    }

    pub fn affine_lanes(
        &mut self,
        input: &mut ReplicatedShare,
        coefficients: &[Affine],
        bounds: Interval,
    ) -> Result<ReplicatedShare, Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            validate(self, input)?;
            let n = input.operation.lanes;
            if coefficients.len() != n {
                return Err(Error::Shape);
            }
            reserve(self, 4 * n * 8)?;
            let mut operation = Operation::new(
                self.context(),
                self.source.runtime.next_sequence(),
                Party::P0,
                n,
                1,
                self.source.limits.core,
            )?;
            let mut hash = Sha256::new();
            hash.update(LANE_AFFINE_PROFILE.as_bytes());
            hash.update(self.source.setup);
            hash.update(input.operation.digest());
            hash.update(bounds.lower.to_le_bytes());
            hash.update(bounds.upper.to_le_bytes());
            for c in coefficients {
                if c.right != 0 {
                    return Err(Error::Bounds);
                }
                c.output_bounds(InputBounds {
                    left: bounds,
                    right: Interval::new(0, 0)?,
                })?;
                hash.update(c.left.to_le_bytes());
                hash.update(c.constant.to_le_bytes());
            }
            operation.consumer = Some(hash.finalize().into());
            self.source.runtime.consume_ordinal(operation)?;
            let usage = self.payload_usage();
            input.charge_to(&usage)?;
            let mut first = usage.zeros(n)?;
            let mut second = usage.zeros(n)?;
            for (i, c) in coefficients.iter().enumerate() {
                first[i] = input.first[i].wrapping_mul(c.left as u64);
                second[i] = input.second[i].wrapping_mul(c.left as u64);
                if self.party() == Party::P0 {
                    first[i] = first[i].wrapping_add(c.constant as u64);
                }
                if self.party() == Party::P2 {
                    second[i] = second[i].wrapping_add(c.constant as u64);
                }
            }
            Ok(ReplicatedShare {
                operation,
                party: self.party(),
                first,
                second,
            })
        })();
        self.checked(result)
    }

    pub fn boolean_map(
        &mut self,
        mut input: boolean::Share,
        map: BitMap,
    ) -> Result<boolean::Share, Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            validate(self, &input.value)?;
            let bits = map.width(input.bits)?;
            let n = input.value.operation.lanes;
            self.source.limits.core.admit(n)?;
            input.value.charge_to(&self.payload_usage())?;
            let mut operation = Operation::new(
                self.context(),
                self.source.runtime.next_sequence(),
                Party::P0,
                n,
                1,
                self.source.limits.core,
            )?;
            let mut hash = Sha256::new();
            hash.update(BOOLEAN_MAP_PROFILE.as_bytes());
            hash.update(self.source.setup);
            hash.update(input.binding());
            map.hash(&mut hash);
            operation.consumer = Some(hash.finalize().into());
            self.source.runtime.consume_ordinal(operation)?;
            for word in input
                .value
                .first
                .iter_mut()
                .chain(input.value.second.iter_mut())
            {
                *word = map.apply(*word)
                    & if bits == 0 {
                        0
                    } else {
                        u64::MAX >> (64 - bits)
                    };
            }
            input.value.operation = operation;
            input.bits = bits;
            Ok(input)
        })();
        self.checked(result)
    }

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
