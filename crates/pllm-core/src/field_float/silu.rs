//! Scalar BF16 SiLU: RN32(exact real SiLU), then optional existing RN16 cast.
//! P256/N64 is fixed; ambiguity is an internal invariant error, not a supported
//! finite-input refusal, tolerance, refinement, or alternative numeric method.
//! Scratch is 122 caller words (976 bytes), NOT total stack/live memory/RSS.
//! Quotes are logical source-event ceilings, not payment, reservation, a ledger,
//! CPU instructions, security, or source funding. Callers separately pay quoting,
//! acquisition, entry, frame and debit work. New source needs independent audit.
//! Only the active caller prefix is promised wiped, not scalar locals/registers
//! or compiler-generated/hidden Rust copies; ordinary unwind is not abort.

use super::{finite_r32, fixed_uint as wide, f32_to_bf16, Error};
use crate::kernel_support::WipeGuard;
use wide::{U1024, U512};

const WORDS: usize = 122;

/// Caller scratch only. Spare tail is never borrowed by the wipe guard.
pub fn silu_bf16_scratch_words() -> usize {
    WORDS
}

fn price(total: usize, count: usize, unit: usize) -> Result<usize, Error> {
    total.checked_add(count.checked_mul(unit).ok_or(Error::WorkLimit)?)
        .ok_or(Error::WorkLimit)
}

/// Proposed source-event ceiling, pending independent actual-source audit.
/// Does not acquire or debit any work account.
pub fn silu_bf16_rn32_work() -> Result<usize, Error> {
    // F: 1024 dispatch allowances and 1024 glue-block allowances, each 128.
    // All helper-local operations are inside their unit prices, not free work.
    // Outer arithmetic: 142 scaled-product dispatches (128 Taylor, 14 square),
    // two endpoint dispatches, and the priced helper calls below. Copies are
    // ONLY q->term/sum/end (128+14+2); sparse initialization is in F/init.
    // Actual increment calls <=64+7+1+1=73; 145 is the retained ceiling.
    // Other outer glue: nine checked price terms, mode/public result dispatch,
    // preflight, split/guard/init, 14 typed takes, two sparse coefficients,
    // 64 Taylor/7 square loop controls, 128 sum carry checks, two endpoint
    // remainder folds/ceil checks, two endpoint rounds/equality and guard Drop.
    // These are an inventory for review, not a claim the F audit has passed.
    let work = price(262_144, 144, 4_992)?;
    let work = price(work, 142, 4_736)?;
    let work = price(work, 2, 1_016_768)?;
    let work = price(work, 145, 384)?;
    let work = price(work, 128, 384)?;
    let work = price(work, 2, 384)?;
    let work = price(work, 144, 384)?;
    let work = price(work, 2, 16_512)?;
    price(work, 1, 7_808)
}

/// Same RN32 computation, plus existing RN16 cast and result-control ceiling.
pub fn silu_bf16_work() -> Result<usize, Error> {
    silu_bf16_rn32_work()?.checked_add(256).ok_or(Error::WorkLimit)
}

/// RN32 of exact real SiLU on a finite BF16 input, including signed zeros.
/// Refusal priority is NonFinite, Capacity, WorkLimit; refusals preserve ALL
/// scratch. After admission, Drop wipes exactly the active 122-word prefix.
pub fn silu_bf16_rn32_with_scratch(
    word: u16,
    scratch: &mut [u64],
    work_limit: usize,
) -> Result<f32, Error> {
    match kernel::<0>(word, scratch, work_limit, false)? {
        Scalar::Rn32(value) => Ok(value),
        Scalar::Bf16(_) => Err(Error::Geometry),
    }
}

/// RN16(RN32(exact real SiLU)), NOT direct real-to-BF16 rounding.
/// Same scratch/refusal lifecycle, with a single BF16-mode kernel admission.
pub fn silu_bf16_with_scratch(
    word: u16,
    scratch: &mut [u64],
    work_limit: usize,
) -> Result<u16, Error> {
    match kernel::<0>(word, scratch, work_limit, true)? {
        Scalar::Bf16(value) => Ok(value),
        Scalar::Rn32(_) => Err(Error::Geometry),
    }
}

enum Scalar {
    Rn32(f32),
    Bf16(u16),
}

fn scalar(value: f32, bf16: bool) -> Result<Scalar, Error> {
    if bf16 { Ok(Scalar::Bf16(f32_to_bf16(value)?)) } else { Ok(Scalar::Rn32(value)) }
}

// Returns references only. No owner wrapper or wide backing outside scratch.
fn take<const N: usize>(words: &mut [u64]) -> Result<(&mut [u64; N], &mut [u64]), Error> {
    if words.len() < N { return Err(Error::Geometry); }
    let (head, rest) = words.split_at_mut(N);
    Ok((head.try_into().map_err(|_| Error::Geometry)?, rest))
}

// Sparse coefficient*2^shift, into the ALREADY initialized prefix.
fn coefficient(out: &mut U512, value: u64, shift: usize) {
    let limb = shift / 64;
    let offset = shift % 64;
    out[limb] = value << offset;
    if offset != 0 && limb + 1 < 8 {
        out[limb + 1] = value >> (64 - offset);
    }
}

fn scaled_product(
    a: &U512, b: &U512, d: u64, ceil: bool,
    product: &mut U1024, q: &mut U512,
) -> Result<(), Error> {
    wide::mul_into(a, b, product);
    let sticky = wide::shr_words_div_word_into(product, 4, d, q)
        .map_err(|_| Error::IntegerRange)?;
    if ceil && sticky && wide::increment(q) { return Err(Error::IntegerRange); }
    Ok(())
}

fn endpoint(
    a: &U512, scale: &U512, exp: &U512, negative: bool, ceil: bool,
    den: &mut U512, out: &mut U512, product: &mut U1024, q: &mut U512,
    rem: &mut [u64; 9], trial: &mut [u64; 9],
) -> Result<(), Error> {
    if wide::add_into(scale, exp, den) { return Err(Error::IntegerRange); }
    wide::mul_into(a, if negative { scale } else { exp }, product);
    wide::div_rem_into(product, den, q, rem, trial).map_err(|_| Error::IntegerRange)?;
    let mut remainder = 0_u64;
    for limb in rem.iter() { remainder |= *limb; }
    if ceil && remainder != 0 && wide::increment(q) { return Err(Error::IntegerRange); }
    out.copy_from_slice(q);
    Ok(())
}

// Test-only const parameter reaches the ACTUAL arithmetic/lifecycle kernel.
// Non-test instantiation has no fault controls or persistent instrumentation.
fn kernel<const FAULT: u8>(
    word: u16, scratch: &mut [u64], work_limit: usize, bf16: bool,
) -> Result<Scalar, Error> {
    // All refusal checks precede even splitting/arming: entire scratch unchanged.
    if word & 0x7f80 == 0x7f80 { return Err(Error::NonFinite); }
    if scratch.len() < WORDS { return Err(Error::Capacity); }
    let required = if bf16 { silu_bf16_work()? } else { silu_bf16_rn32_work()? };
    if work_limit < required { return Err(Error::WorkLimit); }
    let (active, _) = scratch.split_at_mut(WORDS);
    let guard = WipeGuard::new(active);
    guard.words.fill(0);
    #[cfg(test)]
    fault::<FAULT>(1)?;

    let mag = word & 0x7fff;
    let negative = word & 0x8000 != 0;
    let sign = u32::from(negative) << 31;
    // Every fast path still constructs its scalar result with the guard armed.
    if mag == 0 { return scalar(f32::from_bits(sign), bf16); }
    if mag >= 0x4300 {
        return scalar(f32::from_bits(if negative { sign } else { u32::from(word) << 16 }), bf16);
    }
    if mag <= 0x3300 {
        // BF16 promotion has 16 zero low bits. Its exact half is on the F32
        // grid even across the normal/subnormal transition (min is 2^-134).
        // Independent strict upper-cell proof covers both signs/asymmetric ulps.
        // Bit logic avoids invoking/pricing the old 8192-unit multiply branch.
        let bits = u32::from(mag) << 16;
        let half = if bits < 0x0100_0000 { bits >> 1 } else { bits - 0x0080_0000 };
        return scalar(f32::from_bits(sign | half), bf16);
    }

    // Exact 122-word layout; every wide backing is a typed borrow of the prefix.
    let (a, rest) = take::<8>(&mut *guard.words)?; // 0..7
    let (t, rest) = take::<8>(rest)?;             // 8..15
    let (term_lo, rest) = take::<8>(rest)?;       // 16..23
    let (term_hi, rest) = take::<8>(rest)?;       // 24..31
    let (sum_lo, rest) = take::<8>(rest)?;        // 32..39
    let (sum_hi, rest) = take::<8>(rest)?;        // 40..47
    let (den, rest) = take::<8>(rest)?;           // 48..55
    let (end_lo, rest) = take::<8>(rest)?;        // 56..63
    let (end_hi, rest) = take::<8>(rest)?;        // 64..71
    let (scale, rest) = take::<8>(rest)?;         // 72..79
    let (product, rest) = take::<16>(rest)?;      // 80..95
    let (q, rest) = take::<8>(rest)?;             // 96..103
    let (rem, rest) = take::<9>(rest)?;           // 104..112
    let (trial, _) = take::<9>(rest)?;           // 113..121
    let exponent = usize::from(mag >> 7);
    let mantissa = u64::from(128 + (mag & 127));
    coefficient(a, mantissa, exponent + 122); // Q*a, a=m*2^(exponent-134)
    coefficient(t, mantissa, exponent + 115); // Exact A/128, not rounded.
    scale[4] = 1;
    term_lo[4] = 1;
    term_hi[4] = 1;
    sum_lo[4] = 1;
    sum_hi[4] = 1;

    // A<2^263, T<Q. Taylor directed upper term error <2 quanta, summed
    // error <128+tail1; exp(t)<11/4 gives initial upper/Q<3. Seven squares
    // obey u_next<=u^2+1/Q and u_j<4^(2^j)-1: final E_hi<2^512-Q.
    // Thus Q+E fits 512, products fit 1024; endpoint numerators <2^775
    // (negative <2^519), and ceil quotient<=A<2^263. Overflow errors below
    // are fail-closed invariant diagnostics, not allowed finite-domain refusals.
    for k in 1..=64_u64 {
        scaled_product(term_lo, t, k, false, product, q)?;
        term_lo.copy_from_slice(q);
        if wide::add_assign(sum_lo, term_lo) { return Err(Error::IntegerRange); }
        scaled_product(term_hi, t, k, true, product, q)?;
        term_hi.copy_from_slice(q);
        if wide::add_assign(sum_hi, term_hi) { return Err(Error::IntegerRange); }
    }
    if wide::increment(sum_hi) { return Err(Error::IntegerRange); } // Omitted tail <1/Q.
    #[cfg(test)]
    fault::<FAULT>(2)?;
    for _ in 0..7 {
        scaled_product(sum_lo, sum_lo, 1, false, product, q)?;
        sum_lo.copy_from_slice(q);
        scaled_product(sum_hi, sum_hi, 1, true, product, q)?;
        sum_hi.copy_from_slice(q);
    }
    // Positive a*E/(Q+E) increases with E; negative magnitude a*Q/(Q+E)
    // decreases. Both rounded endpoints stay live until RN32 equality is checked.
    endpoint(a, scale, if negative { sum_hi } else { sum_lo }, negative, false,
        den, end_lo, product, q, rem, trial)?;
    endpoint(a, scale, if negative { sum_lo } else { sum_hi }, negative, true,
        den, end_hi, product, q, rem, trial)?;
    #[cfg(test)]
    fault::<FAULT>(3)?;
    let lo = finite_r32::round_q256(end_lo, negative)?;
    let hi = finite_r32::round_q256(end_hi, negative)?;
    if lo.to_bits() != hi.to_bits() { return Err(Error::TieMismatch); }
    // Never finish(): Drop wipes exactly active on success/error/ordinary unwind.
    scalar(lo, bf16)
}

#[cfg(test)]
fn fault<const FAULT: u8>(phase: u8) -> Result<(), Error> {
    if FAULT == phase { return Err(Error::IntegerRange); }
    if FAULT == phase + 3 { panic!("local SiLU phase {phase} unwind"); }
    Ok(())
}

#[cfg(test)]
pub(super) fn with_fault(
    word: u16, scratch: &mut [u64], work: usize, bf16: bool, injection: u8,
) -> Result<(), Error> {
    let result = match injection {
        1 => kernel::<1>(word, scratch, work, bf16),
        2 => kernel::<2>(word, scratch, work, bf16),
        3 => kernel::<3>(word, scratch, work, bf16),
        4 => kernel::<4>(word, scratch, work, bf16),
        5 => kernel::<5>(word, scratch, work, bf16),
        6 => kernel::<6>(word, scratch, work, bf16),
        _ => return Err(Error::Geometry),
    };
    result.map(|_| ())
}
