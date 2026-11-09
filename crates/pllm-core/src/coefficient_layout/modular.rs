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

/// Private arithmetic domain, not a key, security parameter set, or owner.
/// Public setup accepts odd composite moduli too. Every value operand of
/// encode/decode/add/sub/mul must be canonical below q. Fixed source patterns
/// do not certify target-machine constant time or register/copy erasure.
pub(crate) struct Montgomery64 {
    q: u64,
    qinv: u64,
    r2: u64,
}

impl Montgomery64 {
    /// PUBLIC modulus only. Six wrapping Newton steps double the inverse's
    /// correct low bits from one to 64. R and R^2 setup use public remainders.
    pub(crate) fn new(q: u64) -> Option<Self> {
        if !(3..(1_u64 << 61)).contains(&q) || q & 1 == 0 {
            return None;
        }
        let mut inverse = 1_u64;
        for _ in 0..6 {
            inverse = inverse.wrapping_mul(2_u64.wrapping_sub(q.wrapping_mul(inverse)));
        }
        let r = (1_u128 << 64) % u128::from(q);
        let r2 = (r * r % u128::from(q)) as u64;
        Some(Self { q, qinv: 0_u64.wrapping_sub(inverse), r2 })
    }

    fn subtract_once(&self, value: u64) -> u64 {
        let (reduced, borrow) = value.overflowing_sub(self.q);
        let mask = 0_u64.wrapping_sub(u64::from(borrow));
        (value & mask) | (reduced & !mask)
    }

    /// INTERNAL bounded REDC only: T<q^2, never an arbitrary-u128 interface.
    /// m<2^64, mq<2^125, T<2^122, so the sum fits u128. The cancellation
    /// makes it divisible by R; v<q+q^2/R<2q needs one masked subtraction.
    fn reduce(&self, product: u128) -> u64 {
        let m = (product as u64).wrapping_mul(self.qinv);
        let sum = product.wrapping_add(u128::from(m).wrapping_mul(u128::from(self.q)));
        self.subtract_once((sum >> 64) as u64)
    }

    /// Ordinary canonical coefficient -> canonical Montgomery coefficient xR.
    pub(crate) fn encode(&self, value: u64) -> u64 {
        self.reduce(u128::from(value).wrapping_mul(u128::from(self.r2)))
    }

    /// Only the documented external boundary leaves the Montgomery domain.
    pub(crate) fn decode(&self, value: u64) -> u64 {
        self.reduce(u128::from(value))
    }

    pub(crate) fn mul(&self, lhs: u64, rhs: u64) -> u64 {
        self.reduce(u128::from(lhs).wrapping_mul(u128::from(rhs)))
    }

    pub(crate) fn add(&self, lhs: u64, rhs: u64) -> u64 {
        self.subtract_once(lhs.wrapping_add(rhs))
    }

    pub(crate) fn sub(&self, lhs: u64, rhs: u64) -> u64 {
        let (difference, borrow) = lhs.overflowing_sub(rhs);
        difference.wrapping_add(self.q & 0_u64.wrapping_sub(u64::from(borrow)))
    }

    /// Full scan, including values following an invalid coefficient.
    pub(crate) fn invalid(&self, values: &[u64]) -> u64 {
        let mut invalid = 0;
        for &value in values {
            invalid |= u64::from(!value.overflowing_sub(self.q).1);
        }
        invalid
    }
}
