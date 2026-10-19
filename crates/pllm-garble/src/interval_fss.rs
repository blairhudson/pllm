//! Public-boundary key sharing from FSS for Mixed-Mode Secure Computation,
//! ePrint 2020/1392, §4 Lemma 1 and Figures 3/14. Independent implementation.
//!
//! One DCF for gamma = r-1 serves every PUBLIC boundary at this input width.
//! The dealer folds the missing shift corrections into the helper's secret
//! constant shares. Keys are internal to one one-use gate, never shared across
//! phases, lanes or issuances. This is not independent cryptographic review.

use crate::{compact_dcf, shared_rescale_reference::Error};

pub(crate) struct Comparison {
    bits: u8,
    key: compact_dcf::Comparison,
}

fn mask(bits: u8) -> u32 {
    ((1u64 << bits) - 1) as u32
}

/// Dealer-only correction c_p = [((r+p) mod 2^k) > p]. All callers provide
/// validated public widths/boundaries and a mask already in that input ring.
pub(crate) fn correction(bits: u8, input_mask: u32, boundary: u32) -> u32 {
    u32::from((input_mask.wrapping_add(boundary) & mask(bits)) > boundary)
}

impl Comparison {
    pub(crate) fn issue(bits: u8, input_mask: u32) -> Result<[Self; 2], Error> {
        if !(1..=32).contains(&bits) || u64::from(input_mask) >= 1u64 << bits {
            return Err(Error::Domain);
        }
        Ok(
            compact_dcf::Comparison::issue(bits, input_mask.wrapping_sub(1) & mask(bits))?
                .map(|key| Self { bits, key }),
        )
    }

    /// Shares of [u < (r+p) mod 2^k] - c_p, from Lemma 1. The public term
    /// belongs to party zero; Figure 14 assigns it to party one, equivalently.
    pub(crate) fn eval(&self, party: u8, opened: u32, boundary: u32) -> u32 {
        let shifted = opened.wrapping_add(mask(self.bits)).wrapping_sub(boundary) & mask(self.bits);
        self.key
            .eval(party, shifted)
            .wrapping_sub(u32::from(party == 0 && opened > boundary))
    }

    pub(crate) fn payload_bytes(&self) -> usize {
        self.key.payload_bytes()
    }
}

/// Ring-aware universal PUBLIC-original-boundary algebra. Never pass private
/// translated Protocol-3 boundaries to an evaluator. This is not a raw codec,
/// affine descriptor adapter, or consuming online gate.
pub(crate) struct RingComparison {
    bits: u8,
    output_bits: u8,
    key: compact_dcf::RingComparison,
}

impl RingComparison {
    pub(crate) fn resources(bits: u8, output_bits: u8) -> Result<compact_dcf::Resources, Error> {
        let mut cost = compact_dcf::RingComparison::resources(bits, output_bits)?;
        let extra = std::mem::size_of::<Self>() - cost.key_owner_bytes;
        cost.key_owner_bytes += extra;
        cost.issuance_scratch_bytes += 2 * extra;
        cost.combined_allocation_bytes += 4 * extra;
        Ok(cost)
    }

    pub(crate) fn issue(
        bits: u8,
        output_bits: u8,
        input_mask: u64,
        allocation_limit: usize,
    ) -> Result<[Self; 2], Error> {
        let cost = Self::resources(bits, output_bits)?;
        if input_mask >= 1u64 << bits {
            return Err(Error::Domain);
        }
        if cost.combined_allocation_bytes > allocation_limit {
            return Err(Error::Resource);
        }
        let extra =
            4 * (std::mem::size_of::<Self>() - std::mem::size_of::<compact_dcf::RingComparison>());
        let keys = compact_dcf::RingComparison::issue(
            bits,
            output_bits,
            input_mask.wrapping_sub(1) & ((1u64 << bits) - 1),
            1,
            allocation_limit - extra,
        )?;
        Ok(keys.map(|key| Self {
            bits,
            output_bits,
            key,
        }))
    }

    /// A share of [opened < (r+p) mod N] - c_p. The N sentinel is
    /// canonicalized to 0 here; use lt_correction for the full [x<p] identity.
    pub(crate) fn eval(&self, party: u8, opened: u64, boundary: u64) -> Result<u64, Error> {
        let n = 1u64 << self.bits;
        if party > 1 || opened >= n || boundary > n {
            return Err(Error::Domain);
        }
        let p = boundary & (n - 1);
        let shifted = opened.wrapping_sub(1).wrapping_sub(p) & (n - 1);
        Ok(self
            .key
            .eval(party, shifted)?
            .wrapping_sub(u64::from(party == 0 && opened > p))
            & ((1u64 << self.output_bits) - 1))
    }

    /// Shares of the masked comparison difference; dealer keeps the remaining
    /// correction secret-shared. No selected interval or correction is opened.
    #[cfg(test)]
    pub(crate) fn eval_lt_difference(
        &self,
        party: u8,
        opened: u64,
        boundary: u64,
    ) -> Result<u64, Error> {
        Ok(self
            .eval(party, opened, boundary)?
            .wrapping_sub(self.eval(party, opened, 0)?)
            & ((1u64 << self.output_bits) - 1))
    }

    /// Dealer-only correction for eval_lt_difference. Supports both descriptor
    /// sentinels exactly: boundary 0 returns 0, boundary N returns 1. This clear
    /// value must be freshly shared or folded into secret coefficient constants.
    pub(crate) fn lt_correction(
        bits: u8,
        output_bits: u8,
        input_mask: u64,
        boundary: u64,
    ) -> Result<u64, Error> {
        compact_dcf::RingComparison::resources(bits, output_bits)?;
        let n = 1u64 << bits;
        if input_mask >= n || boundary > n {
            return Err(Error::Domain);
        }
        let p = boundary & (n - 1);
        let correction = u64::from(((input_mask + p) & (n - 1)) > p);
        Ok(correction
            .wrapping_sub(u64::from(input_mask > 0))
            .wrapping_add(u64::from(input_mask + boundary >= n))
            & ((1u64 << output_bits) - 1))
    }

    pub(crate) fn allocation(&self) -> compact_dcf::Resources {
        let mut cost = self.key.allocation();
        let extra = std::mem::size_of::<Self>() - cost.key_owner_bytes;
        cost.key_owner_bytes += extra;
        cost.issuance_scratch_bytes += 2 * extra;
        cost.combined_allocation_bytes += 4 * extra;
        cost
    }
}

#[cfg(test)]
mod ring_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_public_shifts_with_every_mask_including_zero() {
        for bits in 1..=5 {
            let max = mask(bits);
            for r in 0..=max {
                let [a, b] = Comparison::issue(bits, r).unwrap();
                assert_eq!(a.payload_bytes(), compact_dcf::Comparison::bytes(bits));
                assert_eq!(a.payload_bytes(), b.payload_bytes());
                for p in 0..=max {
                    for u in 0..=max {
                        assert_eq!(
                            a.eval(0, u, p)
                                .wrapping_add(b.eval(1, u, p))
                                .wrapping_add(correction(bits, r, p)),
                            u32::from(u < (r + p) & max),
                            "bits={bits} r={r} p={p} u={u}"
                        );
                    }
                }
            }
        }
    }

    fn check_intervals(bits: u8, r: u32, intervals: &[(u32, u32)], inputs: &[u32]) {
        let max = mask(bits);
        let [a, b] = Comparison::issue(bits, r).unwrap();
        for &(p, q) in intervals {
            let end = q.wrapping_add(1) & max;
            let alpha_p = p.wrapping_add(r) & max;
            let alpha_q = q.wrapping_add(r) & max;
            // Figure 14's z, lifted to the helper's Z/(2^32) output group.
            let z = u32::from(alpha_p > alpha_q)
                .wrapping_sub(correction(bits, r, p))
                .wrapping_add(correction(bits, r, end))
                .wrapping_add(u32::from(alpha_q == max));
            let mut random = [0; 4];
            getrandom::fill(&mut random).unwrap();
            let za = u32::from_le_bytes(random);
            let zb = z.wrapping_sub(za);
            for &u in inputs {
                let ya = a
                    .eval(0, u, end)
                    .wrapping_sub(a.eval(0, u, p))
                    .wrapping_add(za);
                let yb = b
                    .eval(1, u, end)
                    .wrapping_sub(b.eval(1, u, p))
                    .wrapping_add(zb);
                let x = u.wrapping_sub(r) & max;
                assert_eq!(
                    ya.wrapping_add(yb),
                    u32::from(p <= x && x <= q),
                    "bits={bits} r={r} interval=[{p},{q}] u={u}"
                );
            }
        }
    }

    #[test]
    fn figure_14_all_small_intervals_and_full_width_endpoints() {
        for bits in 1..=4 {
            let values: Vec<_> = (0..=mask(bits)).collect();
            let intervals: Vec<_> = values
                .iter()
                .flat_map(|&p| (p..=mask(bits)).map(move |q| (p, q)))
                .collect();
            for r in 0..=mask(bits) {
                check_intervals(bits, r, &intervals, &values);
            }
        }
        for bits in [16, 24, 32] {
            let max = mask(bits);
            let h = 1 << (bits - 1);
            let inputs = [0, 1, h - 1, h, h + 1, max - 1, max];
            for r in inputs {
                check_intervals(
                    bits,
                    r,
                    &[(0, max), (0, 0), (max, max), (h, h), (0, h - 1), (h, max)],
                    &inputs,
                );
            }
        }
        for (bits, r) in [(0, 0), (33, 0), (1, 2), (31, u32::MAX)] {
            assert!(matches!(Comparison::issue(bits, r), Err(Error::Domain)));
        }
    }
}
