//! Allocation-free clear BF16 row composition, not a source/controller capability.
//!
//! This implements the increasing-coordinate, cast-before-gamma numeric rule.
//! Raw slices confer no source authority, private execution, ledger or profile.
use super::{
    add_rn32, div_rn32, f32_to_bf16, mul_rn32, promote_bf16, rsqrt_rn32, Error, FINITE_R32_WORK,
    RSQRT_WORK,
};

const EPSILON_BITS: u32 = 0x3586_37bd;

// Separately auditable OTHER inventory, not charged again inside primitives:
// 4 promotions * 64 = 256; 2 casts * 64 = 128; two-pass loop/control = 256;
// per-coordinate row checks/glue = 128; full-row cleanup = 256. Total 1024*d.
// Fixed 4096 covers shape/width/checked-quote/control and error-dispatch glue.
// Promotion: classification mask/compare/branch + shift/pack, <64 logical ops.
// Cast: mask, two truncations, remainder/tie/parity checks, increment, overflow
// check, sign pack, <64 even on its longest branch. Two iterator advances and
// destructuring visits fit 256; ten per-coordinate Result propagations (four
// promotions, two casts, four RN32 calls) plus row glue fit 128. Cleanup reserves
// a full d stores/advances even when failure follows the final coordinate. Fixed
// checks are length-only, with bounded integer shifts and checked quote math.
const OTHER_PER_COORDINATE: usize = 1024;
const OTHER_FIXED: usize = 4096;

// PROVISIONAL logical-content envelope, NOT measured/guaranteed physical stack:
// row/control: 80 eight-byte slots = 640 (slice refs, iterator state, scalar
// values, results, width/quote temporaries); finite primitive path: at most eight
// five-u64 Magnitudes + 40 eight-byte scalar/Parts slots = 640; rsqrt path:
// 32 eight-byte slots = 256; unassigned margin = 512. Total 2048. The primitive
// paths are mutually exclusive, but both are reserved. No recursive calls,
// heap/row scratch, retained row, or allocating error value. ABI, compiler spills,
// debug instrumentation and physical backing/stack still need separate audit.
// Input + gamma + output have 6*d logical bytes; actual caller backing capacity
// remains separately owned and is not certified by these three slice lengths.

/// Proposed logical scalar-work quote, including error/full-cleanup allowance.
/// Four RN32 calls per coordinate plus two fixed RN32 calls and one rsqrt:
/// `(4*d+2)*8192 + 4096 + (1024*d+4096) = 33792*d+24576`.
/// Checked independently against current primitive allowances, never saturating.
/// Caller must separately admit/prepay this quote; no ledger is minted/charged,
/// no work limit is accepted by the numeric helper, and no execution grant is
/// implied. Not a CPU/time/RSS/physical-stack bound or a cleared producer quote.
pub fn rms_norm_bf16_row_work(width: usize) -> Result<usize, Error> {
    exact_width(width)?;
    quote(width)
}

/// Logical allowance for [`rms_norm_bf16_bound`], deliberately identical to the
/// checked row quote: `(4*d+2)*FINITE_R32_WORK + RSQRT_WORK + 1024*d+4096`.
/// The bound itself calls d+5 finite RN32 primitives and one rsqrt; d+5 <= 4*d+2
/// for d>=1. Three promotions and two casts fit the first coordinate's existing
/// 256+128 allowance; each iteration reserves 256 control + 128 glue. Fixed 4096
/// covers width/quote/budget, two classifications, sequencing and refusals.
/// This is only a logical allowance, not a SourceWork proof, execution grant,
/// ledger charge, or CPU/physical-resource guarantee.
pub fn rms_norm_bf16_bound_work(width: usize) -> Result<usize, Error> {
    rms_norm_bf16_row_work(width)
}

/// Sufficient, conservative absolute BF16 output envelope for the clear RMS row.
/// Caller supplies public nonnegative finite BF16 bounds H and G on every input
/// and gamma magnitude. Both zero signs are accepted and canonicalized; positive
/// subnormals are preserved. Width/exactness, checked quote, and budget precede
/// inspection of H then G. Arithmetic or BF16 cast overflow returns NonFinite.
///
/// Rounded positive energy is monotone: each actual square and increasing-order
/// partial sum is bounded by the same recurrence with H. Its mean and adjusted
/// mean therefore remain finite. Actual adjusted mean >= epsilon, so its rounded
/// inverse root <= beta=RN32(rsqrt(epsilon)). Sign symmetry and monotonicity of
/// RN32 and R16 then bound the actual cast-before-gamma output by the returned N.
/// The energy recurrence is literal ordered addition, never width*square.
/// Failure is NOT proof that all rows inside the envelope fail.
///
/// Numeric helper only: no source authentication, private-input range proof,
/// whole-profile/global no-wrap/adoption claim, or physical funding guarantee.
/// Even success does not imply signed-i64 Q8 representability: separately check
/// `quantize_q8(promote_bf16(N)?)` on the positive returned bound.
pub fn rms_norm_bf16_bound(
    width: usize,
    input_abs: u16,
    gamma_abs: u16,
    work_limit: usize,
) -> Result<u16, Error> {
    let exact = exact_width(width)?;
    let required = quote(width)?;
    if required > work_limit {
        return Err(Error::WorkLimit);
    }
    let h = bound_magnitude(input_abs)?;
    let g = bound_magnitude(gamma_abs)?;
    let square = mul_rn32(h, h)?;
    let mut sum = f32::from_bits(0);
    for _ in 0..width {
        sum = add_rn32(sum, square)?;
    }
    let mean = div_rn32(sum, exact)?;
    // Check this even though the conservative inverse uses epsilon alone.
    let _adjusted = add_rn32(mean, f32::from_bits(EPSILON_BITS))?;
    let beta = rsqrt_rn32(f32::from_bits(EPSILON_BITS))?;
    let normalized = mul_rn32(h, beta)?;
    let cast = f32_to_bf16(normalized)?;
    let product = mul_rn32(promote_bf16(cast)?, g)?;
    Ok(f32_to_bf16(product)? & 0x7fff)
}

fn bound_magnitude(word: u16) -> Result<f32, Error> {
    // Classify nonfinite first, including negative infinity and negative NaNs.
    let value = promote_bf16(word)?;
    if word & 0x8000 != 0 && word & 0x7fff != 0 {
        return Err(Error::IntegerRange);
    }
    Ok(if word & 0x7fff == 0 {
        f32::from_bits(0)
    } else {
        value
    })
}

/// Checked worst-case logical work for [`rms_norm_bf16_relational_bound`].
/// With M=0x7f80=32640 nonnegative finite BF16 encodings, the quote is
/// `(d+4+4*M)*FINITE_R32_WORK + M*RSQRT_WORK + 1024*(d+M) + 4096`.
/// There are d+3 energy RN32 calls, four per encoding, and one final gamma call;
/// each encoding has one rsqrt. All quote additions/multiplications are checked.
/// The quote is independent of H/G, even when the actual prefix is smaller.
///
/// OTHER reserves 1024 per encoding for promotion, first cast, loop, maximum,
/// control and Result dispatch. The first d allowance covers fixed H/G/C
/// promotions and the final cast because d>=1; the remaining d allowance covers
/// energy-loop control/dispatch. Fixed 4096 covers preflight, classification and
/// checked arithmetic. These are logical allowances, not measured CPU/stack/RSS
/// bounds, SourceWork issuance, ledger charges or execution grants.
pub fn rms_norm_bf16_relational_bound_work(width: usize) -> Result<usize, Error> {
    exact_width(width)?;
    relational_quote(width)
}

fn relational_quote(width: usize) -> Result<usize, Error> {
    const MAGNITUDES: usize = 0x7f80;
    let per_encoding = MAGNITUDES.checked_mul(4).ok_or(Error::WorkLimit)?;
    let roots = MAGNITUDES.checked_mul(RSQRT_WORK).ok_or(Error::WorkLimit)?;
    let arithmetic = width
        .checked_add(4)
        .and_then(|n| n.checked_add(per_encoding))
        .and_then(|n| n.checked_mul(FINITE_R32_WORK))
        .and_then(|n| n.checked_add(roots))
        .ok_or(Error::WorkLimit)?;
    let other = width
        .checked_add(MAGNITUDES)
        .and_then(|n| n.checked_mul(OTHER_PER_COORDINATE))
        .and_then(|n| n.checked_add(OTHER_FIXED))
        .ok_or(Error::WorkLimit)?;
    arithmetic.checked_add(other).ok_or(Error::WorkLimit)
}

/// Offline relational absolute BF16 output certificate for the clear RMS row.
/// H/G are public nonnegative finite BF16 magnitude bounds; either zero sign is
/// accepted and canonicalized. Precedence is exact width, checked worst-case
/// quote, budget, H classification, G classification, repeated-H energy check,
/// encoding enumeration, then final gamma. Negative finite bounds return
/// IntegerRange; nonfinite bounds/arithmetic/casts return NonFinite; quote
/// overflow or insufficient budget returns WorkLimit. Zero G skips no checks.
///
/// For every encoding u=0..=(H&0x7fff), compute s=RN32(u*u),
/// a=RN32(RN32(s/d)+epsilon), b=rsqrt_rn32(a), c=R16(RN32(u*b)).
/// C is the largest canonical BF16 c AFTER this first cast. Return
/// R16(RN32(promote(C)*promote(G))), with canonical +0. Casts never combine.
/// Positive finite BF16 encoding order is value order; no heap is used.
///
/// Proof: each nonnegative sequential rounded sum is >= every individual square,
/// so the actual adjusted mean is >= a for that coordinate and its inverse root
/// is <= b. Monotone multiply/cast and sign symmetry bound each coordinate by C
/// before gamma and the returned bound after gamma. The literal repeated-H
/// energy recurrence bounds all actual energy intermediates; enumeration and
/// final cast establish finite output. Refusal need not mean every row fails.
///
/// Numeric certificate only, not a runtime replacement, provenance/private-range
/// proof, global no-wrap, funding or profile guarantee. Success does not certify
/// signed-i64 Q8: separately check quantize_q8 on the promoted positive bound.
pub fn rms_norm_bf16_relational_bound(
    width: usize,
    input_abs: u16,
    gamma_abs: u16,
    work_limit: usize,
) -> Result<u16, Error> {
    let exact = exact_width(width)?;
    let required = relational_quote(width)?;
    if required > work_limit {
        return Err(Error::WorkLimit);
    }
    let h = bound_magnitude(input_abs)?;
    let g = bound_magnitude(gamma_abs)?;
    let square = mul_rn32(h, h)?;
    let mut sum = f32::from_bits(0);
    for _ in 0..width {
        sum = add_rn32(sum, square)?;
    }
    let mean = div_rn32(sum, exact)?;
    let _adjusted = add_rn32(mean, f32::from_bits(EPSILON_BITS))?;
    let mut maximum = 0_u16;
    for word in 0..=(input_abs & 0x7fff) {
        let value = promote_bf16(word)?;
        let square = mul_rn32(value, value)?;
        let mean = div_rn32(square, exact)?;
        let adjusted = add_rn32(mean, f32::from_bits(EPSILON_BITS))?;
        let beta = rsqrt_rn32(adjusted)?;
        let cast = f32_to_bf16(mul_rn32(value, beta)?)? & 0x7fff;
        maximum = maximum.max(cast);
    }
    let product = mul_rn32(promote_bf16(maximum)?, g)?;
    Ok(f32_to_bf16(product)? & 0x7fff)
}

/// Logical work for [`rms_norm_bf16_q8_energy_bound`]: the existing checked
/// relational quote plus 4096. Only this new API requires 1..=MAX_COLUMNS.
/// Geometry precedes quote arithmetic; quote overflow is unreachable on this
/// domain with the current allowances, but the addition remains checked.
///
/// Additional logical model: eight units per scalar operation/control site,
/// four loop-free blocks reserved at 128 sites each: 8*4*128=4096. Conservative
/// source audit (operations, bindings, branches, conversions, call/Result sites;
/// alternative branches counted together, not opaque machine instructions):
/// (1) wrapper preflight/quote/budget/dispatch <=128: geometry/wrapper glue <=32,
///     exact_width <=40, relational_quote and its public wrapper <=56;
/// (2) promotion/Q8 <=112: promote_bf16 <=16, quantize_q8 <=72, its loop-free
///     div_round_ties_even_i128 (including debug assertion) <=24;
/// (3) canonical G decode/ceiling <=96: decode/multiply <=32, dyadic shift/ceil
///     guards, bindings, branches and Result dispatch <=64;
/// (4) final energy/return <=48: checked +1/square/width product <=24,
///     helper/caller Result and final return glue <=24.
/// The relational bound's own body is charged ONLY by its existing quote.
/// These allowances include error prefixes; no added block has a loop or heap.
/// This is not CPU/physical-resource pricing, SourceWork or a funding grant.
pub fn rms_norm_bf16_q8_energy_bound_work(width: usize) -> Result<usize, Error> {
    q8_energy_quotes(width).map(|(_, energy)| energy)
}

fn q8_energy_quotes(width: usize) -> Result<(usize, usize), Error> {
    if width == 0 || width > crate::integer_bounds::MAX_COLUMNS {
        return Err(Error::Geometry);
    }
    let relational = rms_norm_bf16_relational_bound_work(width)?;
    let energy = relational.checked_add(4096).ok_or(Error::WorkLimit)?;
    Ok((relational, energy))
}

/// Conditional public u128 bound on sum(q_i^2) for the cast-before-gamma RMS
/// row followed by signed-i64 Q8. This is NOT a RowEnergyCertificate: it does not
/// scan/authenticate a row, establish no-wrap, select a profile or grant funding.
/// Premises are the caller's public H/G magnitude bounds on all row coordinates.
/// Existing RMS APIs retain their wider width domain.
///
/// Precedence: width 0 or >MAX_COLUMNS -> Geometry; checked relational quote
/// (including exact width), checked +4096 -> WorkLimit; full budget check;
/// relational bound with EXACTLY its own quote; promote returned positive bound
/// and quantize_q8; only then decode canonical G and calculate coarse energy.
/// The positive Q8 check establishes symmetric i64 coverage (sufficient, not
/// necessary); zero G skips no finite/arithmetic/cast/Q8 check. Coarse checked
/// arithmetic overflow returns IntegerRange, even if some actual rows fit.
///
/// Proof, in real arithmetic: let u=2^-24, a=1-u, b=1+u, eta=2^-150 and
/// eps=exact F32(0x358637bd). Ordered RN32 squares/adds give
/// z >= a^(d+1)*sum(x_i^2)-2*d*eta. The actual adjusted mean satisfies
/// D >= a^2*z/d+eps/2. It is normal, so r=rsqrt_rn32(D)<=b/sqrt(D).
/// Since 4*a^2*eta<=eps, sum((x_i*r)^2)<=d*b^2/a^(d+3)<=2*d for
/// d<=16384: Bernoulli gives a^(d+3)>=1-(d+3)*u>=b^2/2.
///
/// Put v=2^-8, eta16=2^-134, A=(1+v)*(1+u), B=(1+v)*eta+eta16.
/// Each RN32-multiply/BF16-cast stage has absolute envelope A*|t|+B.
/// Applying BOTH stages and Minkowski gives
/// ||output|| <= A^2*G*sqrt(2*d)+A*B*G*sqrt(d)+B*sqrt(d).
/// Q8 rounding adds sqrt(d)/2 after scaling by 256. Now A^2<65/64,
/// A<2, B<2^-133 and sqrt(2)<3/2 imply
/// ||q||/sqrt(d) <= 390*G+2^-125*(2*G+1)+1/2 <= 392*G+1.
/// Thus K=ceil(392*G)+1 and E=d*K^2 suffice. Integer dyadic decoding uses
/// no host floating arithmetic, square root or allocation. A caller combining E
/// with a scanned matrix must separately check geometry and centered-field
/// thresholds; neither this helper nor dot_radius performs global admission.
pub fn rms_norm_bf16_q8_energy_bound(
    width: usize,
    input_abs: u16,
    gamma_abs: u16,
    work_limit: usize,
) -> Result<u128, Error> {
    let (relational, required) = q8_energy_quotes(width)?;
    if required > work_limit {
        return Err(Error::WorkLimit);
    }
    let bound = rms_norm_bf16_relational_bound(width, input_abs, gamma_abs, relational)?;
    super::quantize_q8(promote_bf16(bound)?)?;
    // Relational success already validated G's sign/finiteness. Only now may
    // signed zero be canonicalized for the coarse dyadic calculation.
    let ceiling = ceil_392_gamma(gamma_abs & 0x7fff)?;
    q8_coarse_energy(width, ceiling)
}

// Private contract: canonical, validated nonnegative finite BF16 word.
fn ceil_392_gamma(word: u16) -> Result<u128, Error> {
    let exponent = (word >> 7) & 255;
    let fraction = u128::from(word & 127);
    let significand = if exponent == 0 {
        fraction
    } else {
        128 + fraction
    };
    let shift = i32::from(exponent.max(1)) - 134; // -133..=120.
    let numerator = 392_u128.checked_mul(significand).ok_or(Error::IntegerRange)?;
    ceil_positive_dyadic(numerator, shift)
}

fn ceil_positive_dyadic(numerator: u128, shift: i32) -> Result<u128, Error> {
    if numerator == 0 {
        return Ok(0);
    }
    if shift >= 0 {
        let left = shift as u32;
        // checked_shl alone checks the shift count, NOT lost high bits.
        if left >= 128 || numerator > (u128::MAX >> left) {
            return Err(Error::IntegerRange);
        }
        Ok(numerator << left)
    } else {
        let right = shift.unsigned_abs();
        if right >= 128 {
            return Ok(1);
        }
        let quotient = numerator >> right;
        let remainder = numerator & ((1_u128 << right) - 1);
        quotient
            .checked_add(u128::from(remainder != 0))
            .ok_or(Error::IntegerRange)
    }
}

fn q8_coarse_energy(width: usize, ceiling: u128) -> Result<u128, Error> {
    let k = ceiling.checked_add(1).ok_or(Error::IntegerRange)?;
    let square = k.checked_mul(k).ok_or(Error::IntegerRange)?;
    square.checked_mul(width as u128).ok_or(Error::IntegerRange)
}

fn quote(width: usize) -> Result<usize, Error> {
    let arithmetic = width
        .checked_mul(4)
        .and_then(|n| n.checked_add(2))
        .and_then(|n| n.checked_mul(FINITE_R32_WORK))
        .and_then(|n| n.checked_add(RSQRT_WORK))
        .ok_or(Error::WorkLimit)?;
    let other = width
        .checked_mul(OTHER_PER_COORDINATE)
        .and_then(|n| n.checked_add(OTHER_FIXED))
        .ok_or(Error::WorkLimit)?;
    arithmetic.checked_add(other).ok_or(Error::WorkLimit)
}

// Shape/exact-width/quote preflight takes lengths ONLY: no coordinate can be
// visited or output mutated here. Test huge widths without allocating fake rows.
fn preflight(input: usize, gamma: usize, output: usize) -> Result<f32, Error> {
    if input == 0 {
        return Err(Error::Geometry);
    }
    if input != gamma || input != output {
        return Err(Error::Length);
    }
    let width = exact_width(input)?;
    quote(input)?;
    Ok(width)
}

// Positive usize -> EXACT finite F32, by integer leading bit/divisibility only.
// No `as f32`, host rounding, iterative float conversion or element inspection.
fn exact_width(width: usize) -> Result<f32, Error> {
    if width == 0 {
        return Err(Error::Geometry);
    }
    let high = usize::BITS - 1 - width.leading_zeros();
    if high > 127 {
        return Err(Error::IntegerRange);
    }
    let discarded = high.saturating_sub(23);
    if width & ((1_usize << discarded) - 1) != 0 {
        return Err(Error::IntegerRange);
    }
    let significand = if high <= 23 {
        width << (23 - high)
    } else {
        width >> discarded
    };
    let fraction = (significand as u32) & 0x007f_ffff;
    Ok(f32::from_bits(((high + 127) << 23) | fraction))
}

/// Clear/offline RMS row with fixed epsilon `0x358637bd` and exact width.
/// Shape, integer width and quote-overflow errors leave output UNTOUCHED, without
/// visiting any element. After preflight, ANY arithmetic/conversion error wipes
/// ALL output words (including unwritten suffix), not just the written prefix.
/// Output is tentative until success: caller must publish only on `Ok(())`.
/// No floating arithmetic, libm, reassociation, allocation or scratch row.
pub fn rms_norm_bf16_row_into(
    input: &[u16],
    gamma: &[u16],
    output: &mut [u16],
) -> Result<(), Error> {
    let width = preflight(input.len(), gamma.len(), output.len())?;
    let result = compute(input, gamma, output, width);
    if result.is_err() {
        output.fill(0);
    }
    result
}

fn compute(input: &[u16], gamma: &[u16], output: &mut [u16], width: f32) -> Result<(), Error> {
    let mut sum = f32::from_bits(0); // Explicit +0, in increasing coordinate order.
    for &word in input {
        let value = promote_bf16(word)?;
        let square = mul_rn32(value, value)?;
        sum = add_rn32(sum, square)?;
    }
    let mean = div_rn32(sum, width)?;
    let adjusted = add_rn32(mean, f32::from_bits(EPSILON_BITS))?;
    let normalizer = rsqrt_rn32(adjusted)?;
    for ((&word, &weight), slot) in input.iter().zip(gamma).zip(output.iter_mut()) {
        let normalized = mul_rn32(promote_bf16(word)?, normalizer)?;
        let cast = f32_to_bf16(normalized)?;
        let product = mul_rn32(promote_bf16(cast)?, promote_bf16(weight)?)?;
        *slot = f32_to_bf16(product)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "rms_norm_tests.rs"]
mod tests;
