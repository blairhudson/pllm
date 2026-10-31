//! Clear numeric conversion/import primitives, not a serving/admission profile.
//!
//! E4M3FN -> exactly promoted positive BF16 block scale -> R32 -> signed Q8;
//! canonical BabyBear arithmetic; DIRECT rational-to-BF16 reply conversion.
//! No clipping, FTZ, inverse scales, dynamic quantization or F32 decode bridge.
//! Raw slices here do not establish source identity, shard integrity, head tying,
//! honest no-wrap, speculative totality, privacy, or acceptance authority.
//! Errors are offline/oracle diagnostics, NOT a private online abort policy.

use crate::{
    coded_linear::{field_add, field_from_signed, field_mul, CODED_LINEAR_FIELD},
    fixed_point::div_round_ties_even_i128,
};

pub mod window_linear;

mod rsqrt;
pub use rsqrt::{rsqrt_rn32, RSQRT_WORK};

mod finite_r32;
pub use finite_r32::{add_rn32, div_rn32, mul_rn32, FINITE_R32_WORK};

mod local_bf16;
pub use local_bf16::{
    add_bf16_into, add_bf16_work, dot_bf16_rn32, dot_bf16_rn32_work,
    mul_bf16_into, mul_bf16_work, rope_bf16_supplied_into, rope_bf16_supplied_work,
};

mod attention_bf16;
pub use attention_bf16::{
    context_row_bf16_into, context_row_bf16_work, score_row_bf16_into, score_row_bf16_work,
};

mod rms_norm;
pub use rms_norm::{rms_norm_bf16_row_into, rms_norm_bf16_row_work};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    NonFinite,
    NonPositiveScale,
    IntegerRange,
    NonCanonical,
    Geometry,
    Length,
    TieMismatch,
    Allocation,
    Capacity,
    WorkLimit,
    Coverage,
    Incomplete,
    SessionMismatch,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

/// Exact finite logical BF16 promotion, including subnormals and signed zero.
pub fn promote_bf16(word: u16) -> Result<f32, Error> {
    if word & 0x7f80 == 0x7f80 {
        return Err(Error::NonFinite);
    }
    Ok(f32::from_bits(u32::from(word) << 16))
}

pub fn positive_bf16_scale(word: u16) -> Result<f32, Error> {
    let value = promote_bf16(word)?;
    if word & 0x8000 != 0 || word & 0x7fff == 0 {
        return Err(Error::NonPositiveScale);
    }
    Ok(value)
}

/// RNE with gradual underflow, preserving zero signs; overflow rejects, not clamps.
pub fn f32_to_bf16(value: f32) -> Result<u16, Error> {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return Err(Error::NonFinite);
    }
    let magnitude = bits & 0x7fff_ffff;
    let mut upper = (magnitude >> 16) as u16;
    let remainder = magnitude & 0xffff;
    if remainder > 0x8000 || (remainder == 0x8000 && upper & 1 != 0) {
        upper += 1;
    }
    if upper & 0x7f80 == 0x7f80 {
        return Err(Error::NonFinite);
    }
    Ok(((bits >> 16) as u16 & 0x8000) | upper)
}

fn fp8_parts(byte: u8) -> Result<(bool, u32, i32), Error> {
    let exponent = (byte >> 3) & 15;
    let fraction = byte & 7;
    if exponent == 15 && fraction == 7 {
        return Err(Error::NonFinite);
    }
    Ok((
        byte & 128 != 0,
        if exponent == 0 {
            u32::from(fraction)
        } else {
            8 + u32::from(fraction)
        },
        if exponent == 0 {
            -9
        } else {
            i32::from(exponent) - 10
        },
    ))
}

/// Encode an already-exact dyadic (<=12 significant bits, quantum >=2^-149).
/// These are the complete product domain's proven limits, not a general cast.
fn exact_product_f32(negative: bool, coefficient: u32, exponent: i32) -> Result<f32, Error> {
    let sign = u32::from(negative) << 31;
    if coefficient == 0 {
        return Ok(f32::from_bits(sign));
    }
    let high = 31 - coefficient.leading_zeros();
    let unbiased = high as i32 + exponent;
    if unbiased >= 128 {
        return Err(Error::NonFinite);
    }
    // Every finite BF16 * finite E4M3FN product has <=12 bits and exponent>=-142.
    debug_assert!(high < 24 && exponent >= -149);
    let bits = if unbiased < -126 {
        coefficient << (exponent + 149) as u32
    } else {
        (((unbiased + 127) as u32) << 23) | ((coefficient << (23 - high)) & 0x7f_ffff)
    };
    Ok(f32::from_bits(sign | bits))
}

pub fn decode_e4m3fn(byte: u8) -> Result<f32, Error> {
    let (sign, coefficient, exponent) = fp8_parts(byte)?;
    exact_product_f32(sign, coefficient, exponent)
}

/// Exact BF16 promotion then R32 multiplication, without host FTZ/libm dependence.
/// Finite products need <=12 significant bits and are on the F32 subnormal grid;
/// they are therefore exact. This also handles scales outside sufficient proof
/// envelopes, rejecting actual overflow rather than imposing a new scale policy.
pub fn dequantize_e4m3fn(byte: u8, scale_word: u16) -> Result<f32, Error> {
    positive_bf16_scale(scale_word)?;
    let (negative, coefficient, exponent) = fp8_parts(byte)?;
    let scale_exponent = (scale_word >> 7) & 255;
    let fraction = u32::from(scale_word & 127);
    let (scale_coefficient, scale_power) = if scale_exponent == 0 {
        (fraction, -133)
    } else {
        (128 + fraction, i32::from(scale_exponent) - 134)
    };
    exact_product_f32(
        negative,
        coefficient * scale_coefficient,
        exponent + scale_power,
    )
}

/// I(256*exact_value(value)), NO extra F32 multiplication and NO saturating cast.
pub fn quantize_q8(value: f32) -> Result<i64, Error> {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 255;
    let fraction = bits & 0x7f_ffff;
    if exponent == 255 {
        return Err(Error::NonFinite);
    }
    let negative = bits >> 31 != 0;
    let coefficient = if exponent == 0 {
        fraction
    } else {
        fraction | (1 << 23)
    };
    let shift = if exponent == 0 {
        -141
    } else {
        exponent as i32 - 142
    };
    let magnitude = if shift < 0 {
        let right = (-shift) as u32;
        if right > 24 {
            0
        } else {
            div_round_ties_even_i128(i128::from(coefficient), 1_i128 << right) as u128
        }
    } else {
        let limit = if negative {
            1_u128 << 63
        } else {
            i64::MAX as u128
        };
        if shift > 63 || u128::from(coefficient) > limit >> shift {
            return Err(Error::IntegerRange);
        }
        u128::from(coefficient) << shift
    };
    let signed = if negative {
        -(magnitude as i128)
    } else {
        magnitude as i128
    };
    i64::try_from(signed).map_err(|_| Error::IntegerRange)
}

pub fn q8_to_field(value: i64) -> u32 {
    field_from_signed(value)
}

/// BF16-derived coefficients, without inventing FP8 scales or another rounding.
pub fn import_bf16_q8(words: &[u16]) -> Result<Vec<i64>, Error> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(words.len())
        .map_err(|_| Error::Allocation)?;
    for &word in words {
        output.push(bf16_q8(word)?);
    }
    Ok(output)
}

/// Caller-buffered version of `import_bf16_q8`, with identical scalar semantics.
/// Validate every coefficient before writing; errors leave `output` unchanged.
/// Work counts scalar visits, not CPU instructions or serving admission.
pub fn import_bf16_q8_into(
    words: &[u16],
    output: &mut [i64],
    work_limit: usize,
) -> Result<(), Error> {
    if output.len() < words.len() {
        return Err(Error::Capacity);
    }
    let work = words.len().checked_mul(2).ok_or(Error::WorkLimit)?;
    if work > work_limit {
        return Err(Error::WorkLimit);
    }
    for &word in words {
        bf16_q8(word)?;
    }
    for (&word, slot) in words.iter().zip(output) {
        *slot = bf16_q8(word)?;
    }
    Ok(())
}

fn bf16_q8(word: u16) -> Result<i64, Error> {
    quantize_q8(promote_bf16(word)?)
}

fn fp8_q8(byte: u8, scale: u16) -> Result<i64, Error> {
    quantize_q8(dequantize_e4m3fn(byte, scale)?)
}

/// All supplied values must be finite and both logical BF16 copies bit-identical.
/// Logical words are decoded by the source owner from little-endian fields.
/// This does not authenticate those copies or establish checkpoint-wide tying.
pub fn validate_bf16_tie(left: &[u16], right: &[u16]) -> Result<(), Error> {
    if left.is_empty() || left.len() != right.len() {
        return Err(Error::Length);
    }
    for (&a, &b) in left.iter().zip(right) {
        promote_bf16(a)?;
        promote_bf16(b)?;
        if a != b {
            return Err(Error::TieMismatch);
        }
    }
    Ok(())
}

/// Direct R16(centered(residue)/65536). Integer zero becomes positive zero.
pub fn decode_reply_bf16(residue: u32) -> Result<u16, Error> {
    if residue >= CODED_LINEAR_FIELD {
        return Err(Error::NonCanonical);
    }
    let centered = if residue <= (CODED_LINEAR_FIELD - 1) / 2 {
        i64::from(residue)
    } else {
        i64::from(residue) - i64::from(CODED_LINEAR_FIELD)
    };
    if centered == 0 {
        return Ok(0);
    }
    let magnitude = centered.unsigned_abs();
    let high = 63 - magnitude.leading_zeros();
    let mut unbiased = high as i32 - 16;
    let mut significand = if high > 7 {
        div_round_ties_even_i128(i128::from(magnitude), 1_i128 << (high - 7)) as u16
    } else {
        (magnitude << (7 - high)) as u16
    };
    if significand == 256 {
        significand = 128;
        unbiased += 1;
    }
    // Every nonzero canonical reply /65536 is BF16-normal, within [-15360,15360].
    Ok((u16::from(centered < 0) << 15) | (((unbiased + 127) as u16) << 7) | (significand & 127))
}

/// Method-neutral [output,input] tiled source geometry. No edge-block padding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Block128Geometry {
    rows: usize,
    columns: usize,
}

impl Block128Geometry {
    pub fn new(rows: usize, columns: usize) -> Result<Self, Error> {
        if rows == 0 || columns == 0 || rows % 128 != 0 || columns % 128 != 0 {
            return Err(Error::Geometry);
        }
        rows.checked_mul(columns).ok_or(Error::Geometry)?;
        Ok(Self { rows, columns })
    }

    /// Caller still owes verified source identity, exact names/dtypes/extents,
    /// config agreement, head tying and complete public admission certificates.
    pub fn import_q8(self, weights: &[u8], scales: &[u16]) -> Result<Vec<i64>, Error> {
        if weights.len() != self.rows * self.columns
            || scales.len() != (self.rows / 128) * (self.columns / 128)
        {
            return Err(Error::Length);
        }
        for &scale in scales {
            positive_bf16_scale(scale)?;
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(weights.len())
            .map_err(|_| Error::Allocation)?;
        for (index, &byte) in weights.iter().enumerate() {
            let row = index / self.columns;
            let column = index % self.columns;
            let scale = scales[(row / 128) * (self.columns / 128) + column / 128];
            output.push(fp8_q8(byte, scale)?);
        }
        Ok(output)
    }
}

/// Clear linear correctness oracle only, NOT the full eight-ID decoder oracle.
/// Does not assert honest integer no-wrap: modular behavior is intentional here.
pub fn linear_bf16(weights: &[i64], rows: usize, input: &[u16]) -> Result<Vec<u16>, Error> {
    if rows == 0 || input.is_empty() || rows.checked_mul(input.len()) != Some(weights.len()) {
        return Err(Error::Geometry);
    }
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(input.len())
        .map_err(|_| Error::Allocation)?;
    for &word in input {
        encoded.push(q8_to_field(bf16_q8(word)?));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(rows)
        .map_err(|_| Error::Allocation)?;
    for row in weights.chunks_exact(input.len()) {
        let mut sum = 0;
        for (&weight, &x) in row.iter().zip(&encoded) {
            sum = field_add(sum, field_mul(q8_to_field(weight), x));
        }
        output.push(decode_reply_bf16(sum)?);
    }
    Ok(output)
}

#[cfg(test)]
#[path = "field_float/tests.rs"]
mod tests;
