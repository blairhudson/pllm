//! Clear, model-neutral attention contractions, not an attention executor.
//!
//! Implements numeric-source-recovery §3.2 D32 and §3.3 steps 3/4 only. Caller
//! selects visible rows and supplies alpha/probabilities; none establish source
//! authority, softmax authenticity, GQA/mask/cache policy or a model profile.
use super::{
    add_rn32, dot_bf16_rn32, dot_bf16_rn32_work, f32_to_bf16, mul_rn32,
    promote_bf16, Error, FINITE_R32_WORK,
};

// Reuse the complete local-dot allowance for each output, including its two
// primitive calls and 2048 OTHER units per term. Its fixed 4096 covers inner
// loop setup/dispatch even for the strided context loop. Additional 2048 per
// output covers two score casts, promotion, indexing, Result dispatch and full
// active-prefix cleanup (<64 each cast/promotion, <256 glue, <256 cleanup).
// Outer fixed 4096 covers checked shape/quote, slicing and error dispatch.
// Logical work only, not CPU/time/stack bounds or prepaid original-ledger credit.
const OTHER_PER_OUTPUT: usize = 2048;
const OTHER_FIXED: usize = 4096;

fn geometry(rows: usize, width: usize) -> Result<usize, Error> {
    if rows == 0 || width == 0 {
        return Err(Error::Geometry);
    }
    rows.checked_mul(width).ok_or(Error::WorkLimit)
}

fn quote(terms: usize, outputs: usize, scale: bool) -> Result<usize, Error> {
    dot_bf16_rn32_work(terms)?
        .checked_add(if scale { FINITE_R32_WORK } else { 0 })
        .and_then(|n| n.checked_add(OTHER_PER_OUTPUT))
        .and_then(|n| n.checked_mul(outputs))
        .and_then(|n| n.checked_add(OTHER_FIXED))
        .ok_or(Error::WorkLimit)
}

/// Checked logical allowance for `rows` score outputs, each with `width` terms.
/// Both dimensions must be positive; checked geometry/work overflow refuses.
/// Includes a separate RN32 alpha multiply and full active-prefix cleanup.
pub fn score_row_bf16_work(rows: usize, width: usize) -> Result<usize, Error> {
    geometry(rows, width)?;
    quote(width, rows, true)
}

/// Checked logical allowance for `width` context outputs over `rows` visible
/// rows. Both dimensions must be positive. No normalization work is performed.
pub fn context_row_bf16_work(rows: usize, width: usize) -> Result<usize, Error> {
    geometry(rows, width)?;
    quote(rows, width, false)
}

fn preflight(work: usize, active: usize, capacity: usize, limit: usize) -> Result<(), Error> {
    if capacity < active {
        return Err(Error::Capacity);
    }
    if work > limit {
        return Err(Error::WorkLimit);
    }
    Ok(())
}

/// For each increasing row j: R16(RN32(promote(R16(D32(query, key[j]))) * alpha)).
/// D32 is increasing-coordinate NONFUSED RN32 multiply then add, starting +0.
/// `keys_row_major` must contain a positive integral number of query-width rows.
/// Alpha is supplied finite F32 bits, NOT a selected/authenticated coefficient.
///
/// Length/geometry/capacity/work preflight visits no elements and leaves output
/// untouched. After preflight, any nonfinite input/alpha, arithmetic overflow or
/// BF16-cast overflow clears ALL active output words (including unwritten ones).
/// Spare tail is always untouched. Publish only on Ok; errors are offline numeric
/// diagnostics, not permission for private online abort. No allocation/host FMA.
pub fn score_row_bf16_into(
    query: &[u16],
    keys_row_major: &[u16],
    alpha_f32_bits: u32,
    output: &mut [u16],
    work_limit: usize,
) -> Result<(), Error> {
    let width = query.len();
    if width == 0 {
        return Err(Error::Geometry);
    }
    if keys_row_major.len() % width != 0 {
        return Err(Error::Length);
    }
    let rows = keys_row_major.len() / width;
    let work = score_row_bf16_work(rows, width)?;
    preflight(work, rows, output.len(), work_limit)?;
    let dot_work = dot_bf16_rn32_work(width)?;
    let active = &mut output[..rows];
    let result = (|| {
        for (key, slot) in keys_row_major.chunks_exact(width).zip(active.iter_mut()) {
            let dot = dot_bf16_rn32(query, key, dot_work)?;
            let rounded_dot = promote_bf16(f32_to_bf16(dot)?)?;
            *slot = f32_to_bf16(mul_rn32(rounded_dot, f32::from_bits(alpha_f32_bits))?)?;
        }
        Ok(())
    })();
    if result.is_err() {
        active.fill(0);
    }
    result
}

/// Each context coordinate k is R16(D32(probabilities, values[:, k])). Values
/// are tightly row-major, so each coordinate contracts a STRIDED column in
/// increasing visible-row order. No probability renormalization/range/sum check.
/// Positive row count comes from probabilities.len(); positive width is inferred
/// from values.len()/rows, which must divide exactly. No implicit masked rows.
/// Same preflight/active-cleanup/spare-tail contract as `score_row_bf16_into`.
pub fn context_row_bf16_into(
    probabilities: &[u16],
    values_row_major: &[u16],
    output: &mut [u16],
    work_limit: usize,
) -> Result<(), Error> {
    let rows = probabilities.len();
    if rows == 0 {
        return Err(Error::Geometry);
    }
    if values_row_major.len() % rows != 0 {
        return Err(Error::Length);
    }
    let width = values_row_major.len() / rows;
    let work = context_row_bf16_work(rows, width)?;
    preflight(work, width, output.len(), work_limit)?;
    let active = &mut output[..width];
    let result = (|| {
        for (coordinate, slot) in active.iter_mut().enumerate() {
            let mut sum = f32::from_bits(0);
            for (&probability, row) in probabilities.iter().zip(values_row_major.chunks_exact(width)) {
                let product = mul_rn32(promote_bf16(probability)?, promote_bf16(row[coordinate])?)?;
                sum = add_rn32(sum, product)?;
            }
            *slot = f32_to_bf16(sum)?;
        }
        Ok(())
    })();
    if result.is_err() {
        active.fill(0);
    }
    result
}

#[cfg(test)]
#[path = "attention_bf16_tests.rs"]
mod tests;
