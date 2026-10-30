//! Shared canonical-residue arithmetic. Callers validate modulus and operands.

pub(crate) fn mul(a: u64, b: u64, p: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(p)) as u64
}
pub(crate) fn sub(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { p - (b - a) }
}
