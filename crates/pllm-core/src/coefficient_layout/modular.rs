//! Shared canonical-residue arithmetic. Callers validate modulus and operands.

pub(crate) fn mul(a: u64, b: u64, p: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(p)) as u64
}
pub(crate) fn sub(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { p - (b - a) }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InverseError {
    NotCoprime,
    Capacity,
}

/// Bounded Euclid extracted from the clear Rns3 owner, unchanged numerically.
/// Callers supply odd 3 <= p < 2^61 and 0 <= a < p. Stack-only,
/// variable-time public arithmetic; no full-u64 or even-modulus contract.
pub(crate) fn inverse(a: u64, p: u64) -> Result<u64, InverseError> {
    let (mut r, mut s, mut x, mut y) = (i128::from(p), i128::from(a), 0_i128, 1_i128);
    for _ in 0..128 {
        if s == 0 {
            return if r == 1 {
                Ok(x.rem_euclid(i128::from(p)) as u64)
            } else {
                Err(InverseError::NotCoprime)
            };
        }
        let d = r / s;
        (r, s) = (s, r - d * s);
        (x, y) = (y, x - d * y);
    }
    Err(InverseError::Capacity)
}
