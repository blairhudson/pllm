//! Pure finite binary32 arithmetic; no live-owner or admission authority.
use super::Error;
use std::cmp::Ordering;

/// Logical scalar-work allowance for EACH add, multiply or divide invocation,
/// including errors. Callers must prepay; these helpers do not charge a ledger.
/// At most 320 inspected bits * 16 units + 3072 fixed control/limb/round units.
/// Not a CPU, RSS, wall-time or machine-fuel bound. Future row producers must
/// re-quote their work; this does not fit an old 512*d+8192 norm-row allowance.
pub const FINITE_R32_WORK: usize = 8192;

#[derive(Clone, Copy)]
struct Parts {
    negative: bool,
    m: u32,
    e: i32,
}

fn decode(value: f32) -> Result<Parts, Error> {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 255;
    if exponent == 255 {
        return Err(Error::NonFinite);
    }
    let fraction = bits & 0x007f_ffff;
    Ok(Parts {
        negative: bits >> 31 != 0,
        m: if exponent == 0 {
            fraction
        } else {
            fraction | 0x0080_0000
        },
        e: if exponent == 0 {
            -149
        } else {
            exponent as i32 - 150
        },
    })
}

// Exact magnitudes, not floating scratch. Five limbs hold the sum of two
// largest finite magnitudes on the 2^-149 grid (highest possible bit 277).
#[derive(Clone, Copy)]
struct Magnitude([u64; 5]);

impl Magnitude {
    fn shifted(value: u64, shift: usize) -> Self {
        // Add: shift=e+149 in [0,253], value<2^24. Mul: shift=0,
        // value<2^48. No caller can overflow the five-limb capacity.
        let mut out = Self([0; 5]);
        let limb = shift / 64;
        let offset = shift % 64;
        out.0[limb] = value << offset;
        if offset != 0 && limb + 1 < 5 {
            out.0[limb + 1] = value >> (64 - offset);
        }
        out
    }

    fn cmp(&self, other: &Self) -> Ordering {
        for (left, right) in self.0.iter().zip(&other.0).rev() {
            match left.cmp(right) {
                Ordering::Equal => {}
                order => return order,
            }
        }
        Ordering::Equal
    }

    fn add(self, other: Self) -> Self {
        let mut out = Self([0; 5]);
        let mut carry = false;
        for (i, slot) in out.0.iter_mut().enumerate() {
            let (sum, first) = self.0[i].overflowing_add(other.0[i]);
            let (sum, second) = sum.overflowing_add(u64::from(carry));
            *slot = sum;
            carry = first || second;
        }
        debug_assert!(!carry);
        out
    }

    fn sub(self, other: Self) -> Self {
        let mut out = Self([0; 5]);
        let mut borrow = false;
        for (i, slot) in out.0.iter_mut().enumerate() {
            let (difference, first) = self.0[i].overflowing_sub(other.0[i]);
            let (difference, second) = difference.overflowing_sub(u64::from(borrow));
            *slot = difference;
            borrow = first || second;
        }
        debug_assert!(!borrow); // Callers subtract only smaller from larger.
        out
    }

    fn high(&self) -> Option<i32> {
        for (i, limb) in self.0.iter().enumerate().rev() {
            if *limb != 0 {
                return Some((i * 64) as i32 + (63 - limb.leading_zeros()) as i32);
            }
        }
        None
    }
}

// Single rounding of exact magnitude*2^base, retaining every guard/sticky bit.
fn round_dyadic(value: Magnitude, base: i32, negative: bool) -> Result<f32, Error> {
    let Some(high) = value.high() else {
        return pack(0, -149, negative);
    };
    let h = high + base;
    if h > 127 {
        return Err(Error::NonFinite);
    }
    let k = (h - 23).max(-149);
    let shift = k - base;
    let mut quotient = 0_u64;
    let mut guard = false;
    let mut sticky = false;
    // When reached: exactly 320 bit inspections, including tiny products and
    // partial cancellation. Exact zero and early overflow bypass this scan.
    // Chosen quantum guarantees <=24 retained bits; left shifts <=23.
    for bit in 0..320 {
        if value.0[bit / 64] & (1_u64 << (bit % 64)) == 0 {
            continue;
        }
        let position = bit as i32 - shift;
        if position >= 0 {
            debug_assert!(position < 24);
            quotient |= 1_u64 << position;
        } else if position == -1 {
            guard = true;
        } else {
            sticky = true;
        }
    }
    if guard && (sticky || quotient & 1 != 0) {
        quotient += 1;
    }
    pack(quotient, k, negative)
}

// quotient*2^k after the ONLY rounding, quotient<=2^24 and k>=-149.
fn pack(mut quotient: u64, mut k: i32, negative: bool) -> Result<f32, Error> {
    let sign = u32::from(negative) << 31;
    if quotient == 0 {
        return Ok(f32::from_bits(sign));
    }
    if quotient == 1 << 24 {
        quotient >>= 1;
        k += 1;
    }
    if quotient < 1 << 23 {
        debug_assert_eq!(k, -149);
        return Ok(f32::from_bits(sign | quotient as u32));
    }
    let exponent = k + 150;
    if exponent >= 255 {
        return Err(Error::NonFinite);
    }
    debug_assert!((1..255).contains(&exponent));
    Ok(f32::from_bits(
        sign | ((exponent as u32) << 23) | (quotient as u32 & 0x007f_ffff),
    ))
}

/// RN32(a+b), nearest-even with gradual underflow. Finite operands only;
/// nonfinite inputs or rounded overflow return `NonFinite`. Exact opposite-sign
/// cancellation is +0; -0 + -0 is -0. No floating arithmetic or allocation.
pub fn add_rn32(a: f32, b: f32) -> Result<f32, Error> {
    let a = decode(a)?;
    let b = decode(b)?;
    let left = Magnitude::shifted(u64::from(a.m), (a.e + 149) as usize);
    let right = Magnitude::shifted(u64::from(b.m), (b.e + 149) as usize);
    if a.negative == b.negative {
        round_dyadic(left.add(right), -149, a.negative)
    } else {
        match left.cmp(&right) {
            Ordering::Greater => round_dyadic(left.sub(right), -149, a.negative),
            Ordering::Less => round_dyadic(right.sub(left), -149, b.negative),
            Ordering::Equal => pack(0, -149, false),
        }
    }
}

/// RN32(a*b), nearest-even with gradual underflow and XOR sign (including zero).
/// Nonfinite inputs or rounded overflow return `NonFinite`. Integer-only, no
/// allocation, libm, FTZ, FMA or FPU-environment dependence.
pub fn mul_rn32(a: f32, b: f32) -> Result<f32, Error> {
    let a = decode(a)?;
    let b = decode(b)?;
    round_dyadic(
        Magnitude::shifted(u64::from(a.m) * u64::from(b.m), 0),
        a.e + b.e,
        a.negative ^ b.negative,
    )
}

/// RN32(a/b), nearest-even with gradual underflow and XOR sign (including zero).
/// Any nonfinite input, ZERO DENOMINATOR of either sign, or rounded overflow
/// returns `NonFinite`. Reject the denominator before zero-numerator handling.
/// Pure scalar numeric helper; does not authorize inputs, owners or admission.
pub fn div_rn32(a: f32, b: f32) -> Result<f32, Error> {
    let a = decode(a)?;
    let b = decode(b)?;
    if b.m == 0 {
        return Err(Error::NonFinite);
    }
    let negative = a.negative ^ b.negative;
    if a.m == 0 {
        return pack(0, -149, negative);
    }
    let mut t = b.m.leading_zeros() as i32 - a.m.leading_zeros() as i32;
    // Correct bit-length estimate once to floor(log2(ma/mb)); t in [-24,23].
    let below = if t >= 0 {
        u64::from(a.m) < (u64::from(b.m) << t)
    } else {
        (u64::from(a.m) << -t) < u64::from(b.m)
    };
    if below {
        t -= 1;
    }
    let h = t + a.e - b.e;
    if h > 127 {
        return Err(Error::NonFinite);
    }
    let k = (h - 23).max(-149);
    let s = a.e - b.e - k;
    // If k=H-23, s=23-t<=47. If k=-149, H<=-126 gives
    // s<=23-t<=47 too. Also ea-eb>=-253 => s>=-104.
    // Numerator <2^71 or denominator <2^128; neither shift can be oversized.
    let (numerator, denominator) = if s >= 0 {
        debug_assert!(s <= 47);
        (u128::from(a.m) << s, u128::from(b.m))
    } else {
        debug_assert!(-s <= 104);
        (u128::from(a.m), u128::from(b.m) << -s)
    };
    let mut quotient = numerator / denominator;
    let remainder = numerator % denominator;
    // Compare remainder to half without overflowing via 2*remainder.
    match remainder.cmp(&(denominator - remainder)) {
        Ordering::Greater => quotient += 1,
        Ordering::Equal if quotient & 1 != 0 => quotient += 1,
        _ => {}
    }
    debug_assert!(quotient <= 1 << 24);
    pack(quotient as u64, k, negative)
}

#[cfg(test)]
#[path = "finite_r32_tests.rs"]
mod tests;
