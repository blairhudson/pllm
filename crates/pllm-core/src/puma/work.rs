//! Public schedule-derived sequence and PRSS bounds, independent of secret data.
//!
//! Operations count consumed top-level ordinals, not messages or gate rounds.
//! Pair/public cursors are exact. Private draws and AES expansions bound each
//! party independently over every possible public pivot. Fill tails are rounded
//! separately, matching the source's call-reset AES schedule.

use super::{prss::Cursors, prss::PrssLimits, Error, MAX_LANES};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkBudget {
    operations: u64,
    public_blocks: u64,
    pair_blocks: u64,
    private_blocks: u64,
    aes_blocks: u64,
}

impl WorkBudget {
    pub fn operations(self) -> u64 {
        self.operations
    }

    pub fn cursors(self) -> Cursors {
        Cursors {
            public_blocks: self.public_blocks,
            pair_blocks: self.pair_blocks,
            private_blocks: self.private_blocks,
            aes_work: self.aes_blocks,
        }
    }

    pub fn checked_add(self, other: Self) -> Result<Self, Error> {
        Ok(Self {
            operations: self
                .operations
                .checked_add(other.operations)
                .ok_or(Error::Bounds)?,
            public_blocks: self
                .public_blocks
                .checked_add(other.public_blocks)
                .ok_or(Error::Bounds)?,
            pair_blocks: self
                .pair_blocks
                .checked_add(other.pair_blocks)
                .ok_or(Error::Bounds)?,
            private_blocks: self
                .private_blocks
                .checked_add(other.private_blocks)
                .ok_or(Error::Bounds)?,
            aes_blocks: self
                .aes_blocks
                .checked_add(other.aes_blocks)
                .ok_or(Error::Bounds)?,
        })
    }

    pub fn checked_mul(self, count: u64) -> Result<Self, Error> {
        Ok(Self {
            operations: self.operations.checked_mul(count).ok_or(Error::Bounds)?,
            public_blocks: self.public_blocks.checked_mul(count).ok_or(Error::Bounds)?,
            pair_blocks: self.pair_blocks.checked_mul(count).ok_or(Error::Bounds)?,
            private_blocks: self
                .private_blocks
                .checked_mul(count)
                .ok_or(Error::Bounds)?,
            aes_blocks: self.aes_blocks.checked_mul(count).ok_or(Error::Bounds)?,
        })
    }

    /// Admit the original complete-program ceilings independently. A purely
    /// local schedule needs no AES allowance and retains one ordinal capacity.
    pub fn limits(self) -> Result<PrssLimits, Error> {
        PrssLimits::program(self.operations.max(1), self.aes_blocks)
    }

    pub(super) const fn local() -> Self {
        Self {
            operations: 1,
            public_blocks: 0,
            pair_blocks: 0,
            private_blocks: 0,
            aes_blocks: 0,
        }
    }

    /// Complete product, including optional one final source truncation.
    /// Matrix contractions draw by output lanes, not by contraction width.
    pub fn product(lanes: usize, rescale: bool) -> Result<Self, Error> {
        let q = blocks(lanes, 8)?;
        let mut result = Self {
            pair_blocks: q,
            aes_blocks: 2 * q,
            ..Self::default()
        };
        if rescale {
            result = result.checked_add(Self::rescale(lanes)?)?;
        } else {
            result.operations = 1;
        }
        Ok(result)
    }

    /// Read-only source rescale schedule estimate; does not admit an operand,
    /// reserve randomness, or relax the independent complete-program ceilings.
    pub fn rescale(lanes: usize) -> Result<Self, Error> {
        let q = blocks(lanes, 8)?;
        Ok(Self {
            operations: 1,
            public_blocks: 1,
            pair_blocks: 2 * q,
            private_blocks: lanes as u64,
            aes_blocks: 4 * q + lanes as u64 + 1,
        })
    }

    pub(super) fn conversion(lanes: usize, sign: bool) -> Result<Self, Error> {
        let double = lanes.checked_mul(2).ok_or(Error::Bounds)?;
        let mut pairs = 2 * blocks(lanes, 8)?; // split and first AND
        for round in 1..=6 {
            let bits = if sign { 64 >> round } else { 64 };
            pairs += blocks(double, backing(bits)?)?;
        }
        Ok(Self {
            pair_blocks: pairs,
            aes_blocks: 2 * pairs,
            ..Self::local()
        })
    }

    /// Exact public-width quotient-ring A2B schedule: one fresh split, one
    /// initial AND, and ceil(log2(bits)) doubled-lane carry gates. Each fill
    /// rounds its AES tail separately, including odd lane counts.
    pub fn conversion_bounded(lanes: usize, bits: u8) -> Result<Self, Error> {
        if !(1..=64).contains(&bits) {
            return Err(Error::Bounds);
        }
        let double = lanes.checked_mul(2).ok_or(Error::Bounds)?;
        let bytes = backing(bits)?;
        let rounds = u8::BITS - (bits - 1).leading_zeros();
        let pairs = 2 * blocks(lanes, bytes)? + u64::from(rounds) * blocks(double, bytes)?;
        Ok(Self {
            pair_blocks: pairs,
            aes_blocks: 2 * pairs,
            ..Self::local()
        })
    }

    pub(super) fn from_boolean(lanes: usize, bits: u8) -> Result<Self, Error> {
        blocks(lanes, 8)?;
        match bits {
            0 => Ok(Self::local()),
            1..=8 => {
                let q = blocks(lanes.checked_mul(bits as usize).ok_or(Error::Bounds)?, 8)?;
                Ok(Self {
                    public_blocks: 1,
                    pair_blocks: 3 * q,
                    aes_blocks: 4 * q + 1,
                    ..Self::local()
                })
            }
            9..=64 => {
                let pairs = 3 * blocks(lanes, 8)?
                    + 6 * blocks(lanes.checked_mul(2).ok_or(Error::Bounds)?, 8)?;
                Ok(Self {
                    pair_blocks: pairs,
                    aes_blocks: 2 * pairs,
                    ..Self::local()
                })
            }
            _ => Err(Error::Bounds),
        }
    }

    pub(super) fn bit_or(lanes: usize, bits: u8) -> Result<Self, Error> {
        let q = blocks(lanes, backing(bits)?)?;
        Ok(Self {
            operations: 3, // one AND, then two local XORs
            pair_blocks: q,
            aes_blocks: 2 * q,
            ..Self::default()
        })
    }

    pub(super) fn admit(
        self,
        sequence: u64,
        cursors: Cursors,
        max_operations: u64,
        max_aes_blocks: u64,
    ) -> Result<(), Error> {
        if sequence
            .checked_add(self.operations)
            .is_none_or(|n| n > max_operations)
            || cursors
                .aes_work
                .checked_add(self.aes_blocks)
                .is_none_or(|n| n > max_aes_blocks)
            || cursors
                .public_blocks
                .checked_add(self.public_blocks)
                .is_none()
            || cursors.pair_blocks.checked_add(self.pair_blocks).is_none()
            || cursors
                .private_blocks
                .checked_add(self.private_blocks)
                .is_none()
        {
            return Err(Error::Bounds);
        }
        Ok(())
    }
}

fn blocks(lanes: usize, bytes: u64) -> Result<u64, Error> {
    if !(1..=MAX_LANES).contains(&lanes) {
        return Err(Error::Bounds);
    }
    Ok((lanes as u64 * bytes).div_ceil(16))
}

fn backing(bits: u8) -> Result<u64, Error> {
    match bits {
        0..=8 => Ok(1),
        9..=16 => Ok(2),
        17..=32 => Ok(4),
        33..=64 => Ok(8),
        _ => Err(Error::Bounds),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puma::{prss::MAX_PROGRAM_AES_BLOCKS, MAX_PROGRAM_OPERATIONS};

    #[test]
    fn bounded_conversion_exact_backings_and_fill_tails() {
        for lanes in [1, 3, 17, 128, 512] {
            for bits in 1_u8..=64 {
                let bytes = if bits <= 8 {
                    1
                } else if bits <= 16 {
                    2
                } else if bits <= 32 {
                    4
                } else {
                    8
                };
                let rounds = (0..=6).find(|&n| (1_u8 << n) >= bits).unwrap();
                let expected_pairs = 2 * ((lanes * bytes) as u64).div_ceil(16)
                    + rounds * ((2 * lanes * bytes) as u64).div_ceil(16);
                let budget = WorkBudget::conversion_bounded(lanes, bits).unwrap();
                assert_eq!(budget.operations(), 1);
                assert_eq!(budget.cursors().pair_blocks, expected_pairs);
                assert_eq!(budget.cursors().aes_work, 2 * expected_pairs);
                assert_eq!(budget.cursors().public_blocks, 0);
                assert_eq!(budget.cursors().private_blocks, 0);
                if bits == 64 {
                    assert_eq!(budget, WorkBudget::conversion(lanes, false).unwrap());
                }
            }
        }
        for (lanes, bits) in [(0, 18), (513, 18), (usize::MAX, 18), (1, 0), (1, 65)] {
            assert!(WorkBudget::conversion_bounded(lanes, bits).is_err());
        }
    }

    #[test]
    fn independent_program_ceilings_and_checked_composition() {
        let wide_conversion = WorkBudget::conversion(512, false).unwrap();
        assert_eq!(wide_conversion.operations(), 1);
        assert_eq!(wide_conversion.cursors().aes_work, 7168);
        wide_conversion.limits().unwrap();
        PrssLimits::program(MAX_PROGRAM_OPERATIONS, MAX_PROGRAM_AES_BLOCKS).unwrap();
        assert!(PrssLimits::program(MAX_PROGRAM_OPERATIONS + 1, 0).is_err());
        assert!(PrssLimits::program(1, MAX_PROGRAM_AES_BLOCKS + 1).is_err());
        PrssLimits::program(1, 0).unwrap();
        let local = WorkBudget::local().checked_mul(u64::MAX).unwrap();
        assert!(local.checked_add(WorkBudget::local()).is_err());
        assert!(local.checked_mul(2).is_err());
        assert!(wide_conversion.checked_mul(u64::MAX).is_err());
        assert!(WorkBudget::product(0, true).is_err());
        assert!(WorkBudget::product(usize::MAX, true).is_err());
        assert!(WorkBudget::conversion(usize::MAX, false).is_err());
        assert!(WorkBudget::from_boolean(1, 65).is_err());
    }

    #[test]
    fn continuation_budget_checks_each_cursor_and_sequence_before_randomness() {
        let work = WorkBudget::product(3, true).unwrap();
        let start = Cursors {
            public_blocks: 11,
            pair_blocks: 13,
            private_blocks: 17,
            aes_work: 19,
        };
        work.admit(23, start, 24, 35).unwrap(); // 6*ceil(3/2)+3+1 = 16 AES
        assert!(work.admit(23, start, 23, 35).is_err());
        assert!(work.admit(23, start, 24, 34).is_err());
        for current in [
            Cursors {
                public_blocks: u64::MAX,
                ..start
            },
            Cursors {
                pair_blocks: u64::MAX,
                ..start
            },
            Cursors {
                private_blocks: u64::MAX,
                ..start
            },
            Cursors {
                aes_work: u64::MAX,
                ..start
            },
        ] {
            assert!(work.admit(23, current, u64::MAX, u64::MAX).is_err());
        }
        assert!(work.admit(u64::MAX, start, u64::MAX, u64::MAX).is_err());
    }
}
