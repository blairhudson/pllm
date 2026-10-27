//! Allocation-free raw scalar codec and logical capacity arithmetic ONLY.
//!
//! No expected pin, source identity, owner, hashing, publication or admission is
//! created here. Caller must reserve before constructing the actual workspace
//! and decoding. Quote excludes caller buffers, scalar evaluation temporaries,
//! moves, allocator/RSS and other call-frame state; it is not an opaque permit.
//! Repeated passes, scale visits, Q8 conversions and later publication inventory
//! must be admitted separately by their caller. Later publication still owes
//! chunk replay/hash/mutation checks, head ties, owner retention and all-path
//! cleanup BEFORE reservation refund. No BF16-to-FP8 conversion exists here.

use super::{checked_sum_bytes, SourceDecodeWorkspace, SourceError};
use pllm_core::field_float::{self, Block128Geometry};
use std::mem::size_of;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RawScalarType {
    F32Le,
    Bf16Le,
    E4m3Fn,
}
impl RawScalarType {
    fn width(self) -> usize {
        match self {
            Self::F32Le => 4,
            Self::Bf16Le => 2,
            Self::E4m3Fn => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawCodecQuote {
    pub(crate) source_bytes: usize,
    pub(crate) scalar_visits: usize,
    pub(crate) workspace_bytes: usize,
}

/// One decode visit per raw scalar and one reusable actual workspace. `byte_cap`
/// bounds source_bytes + workspace_bytes, including when input bytes are borrowed;
/// it does NOT reserve either. No default/selected-source cap is changed. Empty
/// selections reject consistently with the existing ordered-selection guard.
pub(crate) fn quote(
    dtype: RawScalarType,
    elements: usize,
    byte_cap: usize,
    visit_cap: usize,
) -> Result<RawCodecQuote, SourceError> {
    if elements == 0 {
        return Err("invalid ordered source selection");
    }
    let source_bytes = elements
        .checked_mul(dtype.width())
        .ok_or("source extent overflow")?;
    let workspace_bytes = size_of::<SourceDecodeWorkspace>();
    if checked_sum_bytes([source_bytes, workspace_bytes])? > byte_cap {
        return Err("source codec byte capacity exceeded");
    }
    if elements > visit_cap {
        return Err("source codec scalar visits exceeded");
    }
    Ok(RawCodecQuote {
        source_bytes,
        scalar_visits: elements,
        workspace_bytes,
    })
}

fn numeric_error(error: field_float::Error) -> SourceError {
    match error {
        field_float::Error::NonFinite => "nonfinite source scalar",
        field_float::Error::NonPositiveScale => "nonpositive source scale",
        field_float::Error::IntegerRange => "source Q8 integer range",
        field_float::Error::Geometry => "invalid source block geometry",
        _ => "invalid source scalar",
    }
}

/// Exact scalar width: truncated AND trailing bytes reject before workspace
/// mutation. Successful-width reads overwrite all four workspace bytes, including
/// on numeric rejection, matching the existing selected-source decoder contract.
/// E4M3FN is decoded through the unchanged exact dequantizer with BF16 scale one.
pub(crate) fn decode_finite(
    dtype: RawScalarType,
    bytes: &[u8],
    workspace: &mut super::SourceDecodeWorkspace,
) -> Result<f32, SourceError> {
    if bytes.len() != dtype.width() {
        return Err("source scalar width");
    }
    workspace.bytes.fill(0);
    workspace.bytes[..bytes.len()].copy_from_slice(bytes);
    match dtype {
        RawScalarType::F32Le => {
            let value = f32::from_le_bytes(workspace.bytes);
            if !value.is_finite() {
                return Err("nonfinite source scalar");
            }
            Ok(value)
        }
        RawScalarType::Bf16Le => {
            field_float::promote_bf16(u16::from_le_bytes([workspace.bytes[0], workspace.bytes[1]]))
                .map_err(numeric_error)
        }
        RawScalarType::E4m3Fn => scaled_e4m3fn(workspace.bytes[0], [0x80, 0x3f]),
    }
}

pub(crate) fn positive_bf16_scale(
    bytes: &[u8],
    workspace: &mut super::SourceDecodeWorkspace,
) -> Result<f32, SourceError> {
    if bytes.len() != 2 {
        return Err("source scalar width");
    }
    workspace.bytes.fill(0);
    workspace.bytes[..2].copy_from_slice(bytes);
    field_float::positive_bf16_scale(u16::from_le_bytes([workspace.bytes[0], workspace.bytes[1]]))
        .map_err(numeric_error)
}

/// Exact positive BF16 promotion then shared E4M3FN/R32 multiplication. No scale
/// envelope, clipping, inverse scale, FTZ or alternate rounding policy is added.
pub(crate) fn scaled_e4m3fn(byte: u8, scale_le: [u8; 2]) -> Result<f32, SourceError> {
    field_float::dequantize_e4m3fn(byte, u16::from_le_bytes(scale_le)).map_err(numeric_error)
}

/// Shared ties-even signed Q8, including its asymmetric i64 endpoint bounds.
pub(crate) fn q8(value: f32) -> Result<i64, SourceError> {
    field_float::quantize_q8(value).map_err(numeric_error)
}

/// Original row-major [output,input] axes, full 128x128 tiles, no edge padding.
/// Only the index is returned; source/scale extent validation remains caller work.
pub(crate) fn block128_scale_index(
    rows: usize,
    columns: usize,
    row: usize,
    column: usize,
) -> Result<usize, SourceError> {
    Block128Geometry::new(rows, columns).map_err(numeric_error)?;
    if row >= rows || column >= columns {
        return Err("source block coordinate out of bounds");
    }
    let block_row = (row / 128)
        .checked_mul(columns / 128)
        .ok_or("source coordinate overflow")?;
    checked_sum_bytes([block_row, column / 128])
}

#[cfg(test)]
#[path = "raw_tests.rs"]
mod tests;
