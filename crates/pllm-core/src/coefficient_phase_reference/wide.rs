//! Fixed-capacity, non-constant-time clear integer/CRT arithmetic. No BigInt.
use super::Error;
use std::cmp::Ordering;

/// Little-endian four-word unsigned integer. All four words are significant.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Wide(pub [u64; 4]);

impl Ord for Wide {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}
impl PartialOrd for Wide {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl From<u64> for Wide {
    fn from(x: u64) -> Self {
        Self([x, 0, 0, 0])
    }
}
impl Wide {
    pub fn checked_add(self, b: Self) -> Result<Self, Error> {
        let mut out = [0; 4];
        let mut carry = false;
        for (i, v) in out.iter_mut().enumerate() {
            let (s, a) = self.0[i].overflowing_add(b.0[i]);
            let (s, c) = s.overflowing_add(u64::from(carry));
            *v = s;
            carry = a || c;
        }
        if carry {
            Err(Error::Capacity)
        } else {
            Ok(Self(out))
        }
    }
    pub fn checked_sub(self, b: Self) -> Result<Self, Error> {
        let mut out = [0; 4];
        let mut borrow = false;
        for (i, v) in out.iter_mut().enumerate() {
            let (s, a) = self.0[i].overflowing_sub(b.0[i]);
            let (s, c) = s.overflowing_sub(u64::from(borrow));
            *v = s;
            borrow = a || c;
        }
        if borrow {
            Err(Error::Capacity)
        } else {
            Ok(Self(out))
        }
    }
    pub fn checked_mul_word(self, b: u64) -> Result<Self, Error> {
        let mut out = [0; 4];
        let mut carry = 0_u128;
        for (i, v) in out.iter_mut().enumerate() {
            let p = u128::from(self.0[i]) * u128::from(b) + carry;
            *v = p as u64;
            carry = p >> 64;
        }
        if carry != 0 {
            Err(Error::Capacity)
        } else {
            Ok(Self(out))
        }
    }
    pub fn checked_shl(self, bits: u32) -> Result<Self, Error> {
        if bits >= 256 {
            return Err(Error::Capacity);
        }
        let words = bits as usize / 64;
        let shift = bits % 64;
        let mut out = [0; 4];
        for (i, value) in self.0.into_iter().enumerate() {
            let at = i + words;
            if at >= 4 {
                if value != 0 {
                    return Err(Error::Capacity);
                }
                continue;
            }
            out[at] |= value << shift;
            if shift != 0 {
                let high = value >> (64 - shift);
                if at + 1 == 4 {
                    if high != 0 {
                        return Err(Error::Capacity);
                    }
                } else {
                    out[at + 1] |= high;
                }
            }
        }
        Ok(Self(out))
    }
    pub fn shr_word(self) -> Self {
        Self([self.0[1], self.0[2], self.0[3], 0])
    }
    pub fn half(self) -> Self {
        Self(std::array::from_fn(|i| {
            (self.0[i] >> 1) | if i == 3 { 0 } else { self.0[i + 1] << 63 }
        }))
    }
    pub(crate) fn rem_word(self, p: u64) -> u64 {
        let mut r = 0_u128;
        for x in self.0.into_iter().rev() {
            r = ((r << 64) | u128::from(x)) % u128::from(p);
        }
        r as u64
    }
    pub fn add_mod(self, b: Self, q: Self) -> Result<Self, Error> {
        if self >= q || b >= q {
            return Err(Error::NonCanonical);
        }
        let s = self.checked_add(b)?;
        if s >= q { s.checked_sub(q) } else { Ok(s) }
    }
}

pub(super) fn mul(a: u64, b: u64, p: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(p)) as u64
}
pub(super) fn sub(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { p - (b - a) }
}
pub(super) fn inverse(a: u64, p: u64) -> Result<u64, Error> {
    let (mut r, mut s, mut x, mut y) = (i128::from(p), i128::from(a), 0_i128, 1_i128);
    for _ in 0..128 {
        if s == 0 {
            return if r == 1 {
                Ok(x.rem_euclid(i128::from(p)) as u64)
            } else {
                Err(Error::Unsupported)
            };
        }
        let d = r / s;
        (r, s) = (s, r - d * s);
        (x, y) = (y, x - d * y);
    }
    Err(Error::Capacity)
}

/// Three ordered, pairwise-coprime odd limbs. Composite coprime limbs work too.
/// Capacity is explicit: 2^128 <= q < 2^191. No primality/security claim.
/// Validated fields cannot be changed after construction:
/// ```compile_fail
/// use pllm_core::coefficient_phase_reference::{Rns3, LIMBS};
/// let mut r = Rns3::new(LIMBS).unwrap();
/// r.primes[0] = 3;
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rns3 {
    primes: [u64; 3],
    modulus: Wide,
    p01: Wide,
    inverse01: u64,
    inverse012: u64,
}
impl Rns3 {
    pub fn new(primes: [u64; 3]) -> Result<Self, Error> {
        if primes
            .iter()
            .any(|p| !(3..(1_u64 << 61)).contains(p) || p & 1 == 0)
        {
            return Err(Error::Unsupported);
        }
        let p01 = Wide::from(primes[0]).checked_mul_word(primes[1])?;
        let modulus = p01.checked_mul_word(primes[2])?;
        if modulus.0[2] == 0 || modulus.0[3] != 0 || modulus.0[2] >> 63 != 0 {
            return Err(Error::Capacity);
        }
        Ok(Self {
            primes,
            modulus,
            p01,
            inverse01: inverse(primes[0] % primes[1], primes[1])?,
            inverse012: inverse(p01.rem_word(primes[2]), primes[2])?,
        })
    }
    pub fn primes(self) -> [u64; 3] {
        self.primes
    }
    pub fn modulus(self) -> Wide {
        self.modulus
    }
    pub fn reconstruct(self, a: [u64; 3]) -> Result<Wide, Error> {
        if a.iter().zip(self.primes).any(|(&a, p)| a >= p) {
            return Err(Error::NonCanonical);
        }
        let d1 = mul(
            sub(a[1], a[0] % self.primes[1], self.primes[1]),
            self.inverse01,
            self.primes[1],
        );
        let x = Wide::from(a[0]).checked_add(Wide::from(self.primes[0]).checked_mul_word(d1)?)?;
        let d2 = mul(
            sub(a[2], x.rem_word(self.primes[2]), self.primes[2]),
            self.inverse012,
            self.primes[2],
        );
        x.checked_add(self.p01.checked_mul_word(d2)?)
    }
    /// Observed floor-whole encoding. This does NOT replace delta_encode.
    pub fn encode(self, m: u64) -> Result<Wide, Error> {
        Ok(self.modulus.checked_mul_word(m)?.shr_word())
    }
    /// Exact nearest(2^64*x/q), reduced modulo 2^64. Odd q precludes half ties.
    /// Normalized 4-by-3-word division, at most two downward corrections.
    /// Final quotient/remainder check makes capacity failure explicit, never an
    /// approximate rounding result. No backend scaler equivalence is implied.
    pub fn down(self, x: Wide) -> Result<u64, Error> {
        if x >= self.modulus {
            return Err(Error::NonCanonical);
        }
        let numerator = x.checked_shl(64)?.checked_add(self.modulus.half())?;
        // Canonical x makes the rounded quotient at most 2^64. Handle that
        // wrap before normalizing, so the normalized numerator fits four words.
        if numerator >= self.modulus.checked_shl(64)? {
            return Ok(0);
        }
        let shift = self.modulus.0[2].leading_zeros();
        let divisor = self.modulus.checked_shl(shift)?;
        let n = numerator.checked_shl(shift)?;
        let top = (u128::from(n.0[3]) << 64) | u128::from(n.0[2]);
        let mut quotient = if n.0[3] >= divisor.0[2] {
            u64::MAX
        } else {
            (top / u128::from(divisor.0[2])) as u64
        };
        let mut product = divisor.checked_mul_word(quotient)?;
        // For B=2^64 and d2>=B/2, truncating the two low divisor words
        // overestimates floor(n/d) by at most two; capping at B-1 preserves it.
        for _ in 0..2 {
            if product <= n {
                break;
            }
            quotient = quotient.checked_sub(1).ok_or(Error::Capacity)?;
            product = product.checked_sub(divisor)?;
        }
        if product > n || n.checked_sub(product)? >= divisor {
            return Err(Error::Capacity);
        }
        Ok(quotient)
    }
}
