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
