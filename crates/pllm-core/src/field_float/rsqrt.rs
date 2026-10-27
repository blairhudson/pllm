//! Exact single-round reciprocal square root. Pure numeric helper only.
use super::Error;
use std::cmp::Ordering;

/// Conservative logical scalar-work allowance per invocation, including rejected
/// inputs: at most 32 comparisons at 64 units plus 2048 control/decode units.
/// Later callers must prepay this allowance; this helper does not charge a ledger.
/// This is not a CPU-instruction, elapsed-time, allocator or RSS bound.
pub const RSQRT_WORK: usize = 4096;

/// Exactly RN32(1/sqrt(a)), with round-to-nearest, ties-to-even. This is NOT a
/// rounded square root followed by division. Positive finite inputs, including
/// subnormals, produce positive finite normal outputs. No floating arithmetic,
/// libm, FTZ, FMA, allocation or source/admission authority is involved.
///
/// Nonfinite inputs return `NonFinite`; finite nonpositive inputs (both zero
/// signs included) return the existing `NonPositiveScale` diagnostic.
pub fn rsqrt_rn32(a: f32) -> Result<f32, Error> {
    let bits = a.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return Err(Error::NonFinite);
    }
    if bits & 0x8000_0000 != 0 || bits == 0 {
        return Err(Error::NonPositiveScale);
    }
    let (m, e) = parts(bits);
    // 0 < a < 2^128 implies 1/sqrt(a) > 2^-64; a >= 2^-149
    // implies 1/sqrt(a) <= 2^74.5 < 2^75. Thus both endpoints are
    // strict, normal finite brackets, and positive encoding order is value order.
    let mut lo = 0x1f80_0000_u32;
    let mut hi = 0x6500_0000_u32;
    // Initial encoding distance < 2^31: at most 31 binary-search comparisons.
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let (n, k) = parts(mid);
        if compare(m, e, n, k) != Ordering::Greater {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (n, k) = parts(lo);
    // Adjacent normals have midpoint (2*n+1)*2^(k-1), also when hi
    // crosses a power of two. Round using the exact midpoint, not sqrt(a).
    let result = match compare(m, e, 2 * n + 1, k - 1) {
        Ordering::Less => hi,
        Ordering::Greater => lo,
        Ordering::Equal => {
            if lo & 1 == 0 {
                lo
            } else {
                hi
            }
        }
    };
    // Midpoint ties are in fact impossible: its odd numerator 2*n+1 > 1
    // cannot divide a power of two in m*(2*n+1)^2*2^q = 1. Keep RNE
    // handling explicit rather than relying on that fact for rounding policy.
    Ok(f32::from_bits(result))
}

// Positive finite encoding -> exact m*2^e. Search candidates are all normal;
// source inputs may be subnormal. Only masked, fixed-width shifts occur.
fn parts(bits: u32) -> (u32, i32) {
    let fraction = bits & 0x007f_ffff;
    let exponent = (bits >> 23) & 255;
    if exponent == 0 {
        (fraction, -149)
    } else {
        (0x0080_0000 | fraction, exponent as i32 - 150)
    }
}

// Compare a*(n*2^k)^2 with one exactly. m has <=24 bits, n <=25 bits,
// so P < 2^74 fits u128. q is conservatively in [-325,208]; i32 suffices.
// No shifting by q, floating operations or magnitude-dependent loop is needed.
fn compare(m: u32, e: i32, n: u32, k: i32) -> Ordering {
    #[cfg(test)]
    COMPARISONS.with(|count| count.set(count.get() + 1));
    let p = u128::from(m) * u128::from(n) * u128::from(n);
    if p == 0 {
        return Ordering::Less;
    }
    let q = e + 2 * k;
    let h = (127 - p.leading_zeros()) as i32 + q;
    match h.cmp(&0) {
        Ordering::Equal if !p.is_power_of_two() => Ordering::Greater,
        order => order,
    }
}

#[cfg(test)]
std::thread_local! {
    static COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
#[path = "rsqrt_tests.rs"]
mod tests;
