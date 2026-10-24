//! D36 fixed q8 inverse arithmetic, not coordinate/source/admission authority.
//!
//! T[0]=0; T[z]=floor(2^18/z), z=1..255. The sentinel is not reciprocal zero.
//! Global payload maximum is 262144, NOT 4096. A genuine inherited coordinate
//! proof must independently establish reachable z=64..192 before tightening R.
//! This module adds no cache, table Vec, private-index oracle or exp table.
//! B mounts `mod fixed_inverse_table;` and retains Verified in its charged source
//! owner. Every call verifies all 256 values before exposing selected bytes.

use std::mem::size_of;

pub(crate) const ENTRY_COUNT: usize = 256;
pub(crate) const NUMERATOR: u64 = 1 << 18;
pub(crate) const GLOBAL_MAX: u64 = NUMERATOR;
pub(crate) const SHA256_U64LE: [u8; 32] = [
    0xb0, 0x3a, 0xf8, 0x99, 0x17, 0xaf, 0xce, 0x4c, 0xad, 0x1e, 0x19, 0x23, 0xbc, 0x09, 0x06, 0x4e,
    0x7d, 0x0b, 0x19, 0xe7, 0xf5, 0xb9, 0x06, 0xf7, 0x69, 0x4c, 0x3a, 0x42, 0x1c, 0x3e, 0x8f, 0x3d,
];

// Pure compile-time public constants. No runtime generation/allocation or cache.
const fn selected_values() -> [u64; ENTRY_COUNT] {
    let mut values = [0; ENTRY_COUNT];
    let mut index = 1;
    while index < ENTRY_COUNT {
        values[index] = NUMERATOR / index as u64;
        index += 1;
    }
    values
}
static SELECTED: [u64; ENTRY_COUNT] = selected_values();

/// Declarations only. The index is the unsigned Q0 lookup word of a Q6 code;
/// changing that interpretation cannot be justified by these table declarations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Definition {
    pub(crate) entries: usize,
    pub(crate) numerator: u64,
    pub(crate) sentinel: u64,
    pub(crate) coordinate_bits: u8,
    pub(crate) coordinate_fractional_bits: u8,
    pub(crate) denominator_fractional_bits: u8,
    pub(crate) payload_bits: u8,
    pub(crate) payload_fractional_bits: u8,
    pub(crate) global_payload_max: u64,
    pub(crate) sha256_u64le: [u8; 32],
}

pub(crate) fn definition() -> Definition {
    Definition {
        entries: ENTRY_COUNT,
        numerator: NUMERATOR,
        sentinel: 0,
        coordinate_bits: 8,
        coordinate_fractional_bits: 0,
        denominator_fractional_bits: 6,
        payload_bits: 50,
        payload_fractional_bits: 12,
        global_payload_max: GLOBAL_MAX,
        sha256_u64le: SHA256_U64LE,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Error {
    Overflow,
    Literal { index: usize, selected: u64 },
}

/// No clone/caller constructor/raw-table override. Only these fixed static bytes.
/// Arithmetic evidence alone grants no source, range, Budget or privacy promise.
#[derive(Debug)]
pub(crate) struct Verified {
    _all_256_checked: (),
}

impl Verified {
    pub(crate) fn values(&self) -> &'static [u64; ENTRY_COUNT] {
        &SELECTED
    }
}

#[allow(dead_code)]
struct ValidationSlots {
    selected: &'static [u64; ENTRY_COUNT],
    iterator: std::iter::Enumerate<std::slice::Iter<'static, u64>>,
    index: usize,
    literal: u64,
    denominator: u128,
    lower: u128,
    upper: u128,
    successor: u128,
    arithmetic_result: Option<u128>,
    check_result: Result<(), Error>,
    returned: Result<Verified, Error>,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Quote {
    pub(crate) entries: usize,
    pub(crate) sentinel_checks: usize,
    pub(crate) quotient_checks: usize,
    pub(crate) checked_u128_products: usize,
    pub(crate) checked_u128_additions: usize,
    pub(crate) validation_slots_bytes: usize,
    /// Preexisting immutable public storage; native copies are separately charged.
    pub(crate) selected_literal_bytes: usize,
}

/// No hash, allocator, native construction, entropy, Budget or new authority.
pub(crate) fn quote() -> Result<Quote, Error> {
    Ok(Quote {
        entries: ENTRY_COUNT,
        sentinel_checks: 1,
        quotient_checks: ENTRY_COUNT - 1,
        checked_u128_products: (ENTRY_COUNT - 1).checked_mul(2).ok_or(Error::Overflow)?,
        checked_u128_additions: ENTRY_COUNT - 1,
        validation_slots_bytes: size_of::<ValidationSlots>(),
        selected_literal_bytes: size_of::<[u64; ENTRY_COUNT]>(),
    })
}

// Independent of the compile-time quotient generator: checked multiplication
// proves z*T[z] <= N < z*(T[z]+1). No float, precision fallback or supplied proof.
fn check(selected: &[u64; ENTRY_COUNT]) -> Result<(), Error> {
    for (index, &literal) in selected.iter().enumerate() {
        if index == 0 {
            if literal != 0 {
                return Err(Error::Literal {
                    index,
                    selected: literal,
                });
            }
            continue;
        }
        let denominator = u128::try_from(index).map_err(|_| Error::Overflow)?;
        let lower = denominator
            .checked_mul(u128::from(literal))
            .ok_or(Error::Overflow)?;
        let successor = u128::from(literal).checked_add(1).ok_or(Error::Overflow)?;
        let upper = denominator.checked_mul(successor).ok_or(Error::Overflow)?;
        if lower > u128::from(NUMERATOR) || u128::from(NUMERATOR) >= upper {
            return Err(Error::Literal {
                index,
                selected: literal,
            });
        }
    }
    Ok(())
}

/// Containing owner MUST admit validation/return/input overlap before this call.
/// No cache: every successful token has checked sentinel AND all 255 quotients.
pub(crate) fn verify() -> Result<Verified, Error> {
    check(&SELECTED)?;
    Ok(Verified {
        _all_256_checked: (),
    })
}

#[cfg(test)]
#[path = "fixed_inverse_table_tests.rs"]
mod tests;
