//! Bounded integer certificates for `floor(scale * exp(-x))`, 0 <= x <= 16.
//!
//! Public mathematics only: no table/source, range-capability, Budget, protocol,
//! entropy, profile or security authority. No floating point, heap, big integers,
//! adaptive precision, cache or fallback. Mount as `mod exp_floor;` in lib.rs.
//!
//! This is the existing positive Taylor/geometric-tail proof from
//! paper/reproduction-contracts/fusefss/gate-set/reference.py:138-160, with
//! bounded outward dyadic arithmetic instead of unbounded Fraction arithmetic.
//! For y=x/16 in [0,1], enclose exp(y) by 24 updates (t0..t24) and its tail.
//! Terms after t25 have successive ratio <=1/26<1/2, so the omitted tail is
//! <=2*t25. Reciprocal reverses endpoints; four outward squarings enclose
//! exp(-x). Matching endpoint floors certify the result; disagreement rejects.

use std::mem::size_of;

pub(crate) const FRACTION_BITS: u8 = 60;
pub(crate) const ONE: u128 = 1u128 << FRACTION_BITS;
const TERMS: u32 = 24;
const SQUARINGS: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Error {
    Domain,
    Overflow,
    Unresolved { lower_floor: u64, upper_floor: u64 },
}

/// Dyadic x=n/2^b, b<=12 and 0<=x<=16. Not secret-value provenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Argument {
    numerator: u32,
    denominator_bits: u8,
}

impl Argument {
    pub(crate) fn new(numerator: u32, denominator_bits: u8) -> Result<Self, Error> {
        if denominator_bits > 12 {
            return Err(Error::Domain);
        }
        let maximum = 1u32
            .checked_shl(u32::from(denominator_bits) + 4)
            .ok_or(Error::Overflow)?;
        if numerator > maximum {
            return Err(Error::Domain);
        }
        Ok(Self {
            numerator,
            denominator_bits,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Bounds {
    lower: u128,
    upper: u128,
}

/// Immutable arithmetic evidence, constructible only by the bounded verifier.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Certificate {
    argument: Argument,
    scale: u64,
    bounds: Bounds,
    value: u64,
}

impl Certificate {
    pub(crate) fn floor(&self) -> u64 {
        self.value
    }

    /// Inclusive Q60 bounds on exp(-x), not on a secret or a source invocation.
    #[cfg(test)]
    pub(crate) fn bounds_q60(&self) -> (u128, u128) {
        (self.bounds.lower, self.bounds.upper)
    }
}

/// Explicit reusable scratch, constant size regardless of entry count.
#[derive(Debug, Default)]
pub(crate) struct Workspace {
    term: Bounds,
    sum: Bounds,
    interval: Bounds,
    reduced_denominator: u128,
    divisor: u128,
    following: u128,
    reciprocal_numerator: u128,
}

// Named logical call/return overlap, not a physical compiler-stack/RSS claim.
// Caller input/table storage, quote-call storage, hash states and native owners
// remain additional. A primitive's checked Result/Option slots overlap in time.
#[allow(dead_code)]
struct CallSlots {
    argument: Argument,
    scale: u64,
    step: u32,
    argument_numerator: u128,
    arithmetic_inputs: [u128; 3],
    arithmetic_result: Result<u128, Error>,
    checked_primitive: Option<u128>,
    quotient: u128,
    remainder: u128,
    resolving_interval: Bounds,
    floor_endpoints: [u64; 2],
    resolved_floor: Result<u64, Error>,
    returned_certificate: Result<Certificate, Error>,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Quote {
    pub(crate) entries: usize,
    pub(crate) workspace_and_call_bytes: usize,
    pub(crate) taylor_steps: usize,
    pub(crate) squarings: usize,
    pub(crate) checked_multiplications: usize,
    pub(crate) checked_divisions: usize,
    pub(crate) checked_remainders: usize,
    pub(crate) checked_additions: usize,
    pub(crate) checked_shifts: usize,
}

/// Allocation-/hash-free quote; overflow rejects. Reuses one scratch/call slot.
/// Operation counts bound checked arithmetic including Argument construction,
/// excluding conversions, loop/control and caller hash/ownership work.
/// Zero-argument special cases do less work.
pub(crate) fn quote(entries: usize) -> Result<Quote, Error> {
    let times = |per_entry: usize| entries.checked_mul(per_entry).ok_or(Error::Overflow);
    Ok(Quote {
        entries,
        workspace_and_call_bytes: size_of::<Workspace>()
            .checked_add(size_of::<CallSlots>())
            .ok_or(Error::Overflow)?,
        taylor_steps: times(TERMS as usize)?,
        squarings: times(SQUARINGS as usize)?,
        checked_multiplications: times(3 * TERMS as usize + 14)?,
        checked_divisions: times(2 * TERMS as usize + 13)?,
        checked_remainders: times(TERMS as usize + 6)?,
        checked_additions: times(3 * TERMS as usize + 7)?,
        checked_shifts: times(2)?,
    })
}

fn multiply(a: u128, b: u128) -> Result<u128, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}

fn down(numerator: u128, denominator: u128) -> Result<u128, Error> {
    numerator.checked_div(denominator).ok_or(Error::Overflow)
}

fn up(numerator: u128, denominator: u128) -> Result<u128, Error> {
    let quotient = down(numerator, denominator)?;
    let remainder = numerator.checked_rem(denominator).ok_or(Error::Overflow)?;
    quotient
        .checked_add(u128::from(remainder != 0))
        .ok_or(Error::Overflow)
}

fn resolve(scale: u64, interval: Bounds) -> Result<u64, Error> {
    if interval.lower > interval.upper || interval.upper > ONE {
        return Err(Error::Domain);
    }
    let lower_floor = u64::try_from(down(multiply(u128::from(scale), interval.lower)?, ONE)?)
        .map_err(|_| Error::Overflow)?;
    let upper_floor = u64::try_from(down(multiply(u128::from(scale), interval.upper)?, ONE)?)
        .map_err(|_| Error::Overflow)?;
    if lower_floor != upper_floor {
        return Err(Error::Unresolved {
            lower_floor,
            upper_floor,
        });
    }
    Ok(lower_floor)
}

impl Workspace {
    pub(crate) fn certify(&mut self, argument: Argument, scale: u64) -> Result<Certificate, Error> {
        // Exact x=0 is a literal integer boundary; do not widen it by a tail.
        if argument.numerator == 0 {
            return Ok(Certificate {
                argument,
                scale,
                bounds: Bounds {
                    lower: ONE,
                    upper: ONE,
                },
                value: scale,
            });
        }
        self.reduced_denominator = 1u128
            .checked_shl(u32::from(argument.denominator_bits) + 4)
            .ok_or(Error::Overflow)?;
        let numerator = u128::from(argument.numerator);
        self.term = Bounds {
            lower: ONE,
            upper: ONE,
        };
        self.sum = self.term;
        for n in 1..=TERMS {
            self.divisor = multiply(self.reduced_denominator, u128::from(n))?;
            self.term.lower = down(multiply(self.term.lower, numerator)?, self.divisor)?;
            self.term.upper = up(multiply(self.term.upper, numerator)?, self.divisor)?;
            self.sum.lower = self
                .sum
                .lower
                .checked_add(self.term.lower)
                .ok_or(Error::Overflow)?;
            self.sum.upper = self
                .sum
                .upper
                .checked_add(self.term.upper)
                .ok_or(Error::Overflow)?;
        }
        self.divisor = multiply(self.reduced_denominator, u128::from(TERMS + 1))?;
        self.following = up(multiply(self.term.upper, numerator)?, self.divisor)?;
        self.sum.upper = self
            .sum
            .upper
            .checked_add(multiply(2, self.following)?)
            .ok_or(Error::Overflow)?;
        self.reciprocal_numerator = multiply(ONE, ONE)?;
        self.interval.lower = down(self.reciprocal_numerator, self.sum.upper)?;
        self.interval.upper = up(self.reciprocal_numerator, self.sum.lower)?;
        for _ in 0..SQUARINGS {
            self.interval.lower = down(multiply(self.interval.lower, self.interval.lower)?, ONE)?;
            self.interval.upper = up(multiply(self.interval.upper, self.interval.upper)?, ONE)?;
        }
        let value = resolve(scale, self.interval)?;
        Ok(Certificate {
            argument,
            scale,
            bounds: self.interval,
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_zero_and_domain_endpoints() {
        let mut work = Workspace::default();
        let zero = work
            .certify(Argument::new(0, 12).unwrap(), u64::MAX)
            .unwrap();
        assert_eq!(zero.floor(), u64::MAX);
        assert_eq!(zero.bounds_q60(), (ONE, ONE));
        assert!(Argument::new(65536, 12).is_ok());
        assert_eq!(Argument::new(65537, 12), Err(Error::Domain));
        assert_eq!(Argument::new(1, 13), Err(Error::Domain));
    }

    #[test]
    fn boundary_ambiguity_and_overflow_fail_closed() {
        assert_eq!(
            resolve(
                2,
                Bounds {
                    lower: ONE / 2 - 1,
                    upper: ONE / 2
                }
            ),
            Err(Error::Unresolved {
                lower_floor: 0,
                upper_floor: 1
            })
        );
        assert_eq!(multiply(u128::MAX, 2), Err(Error::Overflow));
        assert_eq!(up(u128::MAX, 1), Ok(u128::MAX));
        assert_eq!(up(1, 0), Err(Error::Overflow));
        assert_eq!(
            resolve(
                1,
                Bounds {
                    lower: ONE,
                    upper: ONE - 1
                }
            ),
            Err(Error::Domain)
        );
        assert_eq!(quote(usize::MAX), Err(Error::Overflow));
    }

    #[test]
    fn checked_quote_uses_one_workspace_not_512_copies() {
        let one = quote(1).unwrap();
        let all = quote(512).unwrap();
        assert_eq!(all.workspace_and_call_bytes, one.workspace_and_call_bytes);
        assert_eq!(all.taylor_steps, 512 * 24);
        assert_eq!(all.squarings, 512 * 4);
        assert_eq!(all.checked_multiplications, 512 * 86);
        assert_eq!(all.checked_divisions, 512 * 61);
        assert_eq!(all.checked_remainders, 512 * 30);
        assert_eq!(all.checked_additions, 512 * 79);
        assert_eq!(all.checked_shifts, 512 * 2);
    }
}
