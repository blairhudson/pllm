//! Source-locked public D8 tables. These are public integer constants, not
//! correlated material, a private-index oracle, or checkpoint range evidence.
//! SiLU/exp constants come from the independently rational-certified frozen
//! PLLM fixture. Reciprocal and rsqrt use the exact integer source equations.
use super::super::{allocate, Budget, Error, Lease};
use sha2::{Digest, Sha256};
use std::mem::size_of;

#[path = "source_table_values.rs"]
mod values;

#[cfg(test)]
pub(crate) const SOURCE_LOCK_HEX: &str =
    "8080069e2edf58fb0b6b72a0a27bcbfa636db25018a0696681335a7dfe1ddcbd";

pub(crate) fn source_digest() -> [u8; 32] {
    Sha256::digest(include_bytes!(
        "../../../../../paper/reproduction-contracts/fusefss/gate-set/source-lock.json"
    ))
    .into()
}

pub(crate) fn source_lock_bytes() -> usize {
    include_bytes!("../../../../../paper/reproduction-contracts/fusefss/gate-set/source-lock.json")
        .len()
}

pub(crate) fn silu() -> &'static [u64] {
    crate::fixed_activation_tables::silu_q12()
}
pub(crate) fn exp0() -> &'static [u64] {
    &values::EXP0
}
pub(crate) fn exp1() -> &'static [u64] {
    &values::EXP1
}

/// The generator's public array remains charged until the final copy retires.
/// A lookup Table additionally owns/charges its own immutable copy. No secret
/// value is used to choose a table or generate an entry.
pub(crate) struct Values<'a> {
    values: Vec<u64>,
    _lease: Lease<'a>,
}
impl AsRef<[u64]> for Values<'_> {
    fn as_ref(&self) -> &[u64] {
        &self.values
    }
}
impl std::ops::Deref for Values<'_> {
    type Target = [u64];
    fn deref(&self) -> &[u64] {
        &self.values
    }
}

pub(crate) fn reciprocal_q(k: usize) -> Result<u8, Error> {
    if !(1..=1024).contains(&k) {
        return Err(Error::Domain);
    }
    Ok(6 + (usize::BITS - k.leading_zeros()) as u8)
}

pub(crate) fn reciprocal(k: usize, budget: &Budget) -> Result<Values<'_>, Error> {
    let count = 1usize << reciprocal_q(k)?;
    public_values(count, budget, |index| {
        if index == 0 {
            0
        } else {
            (1u64 << 18) / index as u64
        }
    })
}

pub(crate) fn rsqrt(k: usize, budget: &Budget) -> Result<Values<'_>, Error> {
    if !(1..=4096).contains(&k) {
        return Err(Error::Domain);
    }
    // All 8192 slots are formula-defined, even exponents 50..63 which a valid
    // unsigned50 input cannot select. u128 covers the largest denominator.
    public_values(8192, budget, |index| {
        let mantissa = index >> 6;
        let exponent = index & 63;
        let denominator = (128u128 + mantissa as u128) << exponent;
        let radicand = ((1u128 << 55) * k as u128) / denominator;
        radicand.isqrt() as u64
    })
}

fn public_values(
    count: usize,
    budget: &Budget,
    value: impl Fn(usize) -> u64,
) -> Result<Values<'_>, Error> {
    let bytes = count
        .checked_mul(8)
        .and_then(|v| v.checked_add(size_of::<Values<'_>>()))
        .ok_or(Error::Resource)?;
    let lease = budget.reserve(bytes)?;
    let mut values = allocate(count, budget)?;
    for index in 0..count {
        let entry = value(index);
        if entry >= 1u64 << 50 {
            return Err(Error::Domain);
        }
        values.push(entry);
    }
    Ok(Values {
        values,
        _lease: lease,
    })
}

#[cfg(test)]
#[path = "source_tables_tests.rs"]
mod tests;
