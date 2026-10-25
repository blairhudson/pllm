//! Private collection component, NOT an authenticated source/Manifest issuer.
//! The original E begin owner must preadmit these inline capacities and all
//! external gamma/norm/file owners BEFORE invoking this component. It owns
//! source-order/plan association, cumulative work and disk guards throughout.
//! No factory, alternate budget, source JSON parse or expanded graph lives here.
//! Encoding reuses F32 multiplication then checked i64 cast TOWARD ZERO, not
//! nearest-even. Snapshot completion/digest alone never authenticates its source.
use super::*;
use pllm_core::public_file::PublicFileWriter;
use std::mem::size_of;

#[cfg_attr(not(test), allow(dead_code))] // Inactive until original Header join.
const ENCODED_BYTES: usize = CHUNK * size_of::<i64>();

#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct Scan {
    elements: usize,
    columns: Option<usize>,
    offset: usize,
    hash: Sha256,
    lower: i64,
    upper: i64,
    row_sum: u64,
    row_l1: u64,
    row_square_sum: u128,
    row_squared_norm: u128,
    row_l1_squared_sum: u128,
    retired: bool,
}

/// Computed statistics only. This does not authenticate a source or grant a
/// SourceRecord/SourceRegistry/Program; E checks the original plan association.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct Summary {
    pub(super) lower: i64,
    pub(super) upper: i64,
    pub(super) digest: [u8; 32],
    pub(super) row_l1: Option<u64>,
    pub(super) row_squared_norm: Option<u128>,
    pub(super) row_l1_squared_sum: Option<u128>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl Scan {
    /// Layout only, not reservation authority or a complete publisher quote.
    /// Includes old/new scanner transition and return slots. Caller FFI bytes,
    /// gamma backing, BlockScan, snapshot writer/files/tables/guards are separate.
    pub(super) const fn inline_peak_bytes() -> usize {
        2 * size_of::<Self>()
            + 2 * size_of::<Zeroizing<[u8; ENCODED_BYTES]>>()
            + size_of::<Result<Summary, String>>()
            + 16 * size_of::<usize>()
    }

    pub(super) fn new(elements: usize, columns: Option<usize>) -> Result<Self, String> {
        if elements == 0 || columns.is_some_and(|c| c == 0 || elements % c != 0) {
            return Err("invalid compact source scan shape".into());
        }
        Ok(Self {
            elements,
            columns,
            offset: 0,
            hash: Sha256::new(),
            lower: i64::MAX,
            upper: i64::MIN,
            row_sum: 0,
            row_l1: 0,
            row_square_sum: 0,
            row_squared_norm: 0,
            row_l1_squared_sum: 0,
            retired: false,
        })
    }

    /// All coefficients, INCLUDING gamma, go to the same snapshot byte stream.
    /// Gamma is an already admitted exact destination, never grown by this scan.
    /// Caller must destroy collection on any error/unwind: earlier writes are
    /// provisional and cannot be retried or exposed as a completed source.
    /// This scan does NOT own or close writer/gamma/norms: conversion/norm
    /// failure can leave a live writer and partially mutated borrowed payloads.
    /// The enclosing collector must destroy them BEFORE external guard refunds,
    /// with stable original tensor associations and prepaid per-frame work.
    pub(super) fn append(
        &mut self,
        offset: usize,
        bytes: &[u8],
        writer: &mut PublicFileWriter,
        mut gamma: Option<&mut [i64]>,
        mut norms: Option<&mut source_norms::BlockScan>,
    ) -> Result<(), String> {
        if self.retired {
            return Err("retired compact source scan".into());
        }
        // Burn before touching any external owner; success alone reopens scan.
        self.retired = true;
        if offset != self.offset
            || bytes.is_empty()
            || bytes.len() > CHUNK * 4
            || bytes.len() % 4 != 0
            || offset
                .checked_add(bytes.len() / 4)
                .is_none_or(|end| end > self.elements)
            || gamma.as_ref().is_some_and(|g| g.len() != self.elements)
        {
            return Err("invalid ordered replicated source frame".into());
        }
        let mut encoded = Zeroizing::new([0u8; ENCODED_BYTES]);
        for (i, raw) in bytes.chunks_exact(4).enumerate() {
            let value = encode_q18(raw)?;
            if let Some(scan) = norms.as_mut() {
                scan.append(offset + i, std::slice::from_ref(&value))
                    .map_err(|e| e.to_string())?;
            }
            if let Some(g) = gamma.as_mut() {
                g[offset + i] = value;
            }
            let word = value.to_le_bytes();
            encoded[i * 8..i * 8 + 8].copy_from_slice(&word);
            self.hash.update(word);
            self.lower = self.lower.min(value);
            self.upper = self.upper.max(value);
            if let Some(columns) = self.columns {
                self.row_sum = self
                    .row_sum
                    .checked_add(value.unsigned_abs())
                    .ok_or("public row bound overflow")?;
                self.row_square_sum = self
                    .row_square_sum
                    .checked_add(u128::from(value.unsigned_abs()).pow(2))
                    .ok_or("public row square bound overflow")?;
                if (offset + i + 1) % columns == 0 {
                    self.row_l1 = self.row_l1.max(self.row_sum);
                    self.row_l1_squared_sum = self
                        .row_l1_squared_sum
                        .checked_add(u128::from(self.row_sum).pow(2))
                        .ok_or("public row energy overflow")?;
                    self.row_squared_norm = self.row_squared_norm.max(self.row_square_sum);
                    self.row_sum = 0;
                    self.row_square_sum = 0;
                }
            }
        }
        writer.write(&encoded[..bytes.len() * 2])?;
        self.offset += bytes.len() / 4;
        self.retired = false;
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<Summary, String> {
        if self.retired
            || self.offset != self.elements
            || self.row_sum != 0
            || self.row_square_sum != 0
        {
            return Err("incomplete compact source scan".into());
        }
        self.retired = true;
        Ok(Summary {
            lower: self.lower,
            upper: self.upper,
            digest: self.hash.finalize().into(),
            row_l1: self.columns.map(|_| self.row_l1),
            row_squared_norm: self.columns.map(|_| self.row_squared_norm),
            row_l1_squared_sum: self.columns.map(|_| self.row_l1_squared_sum),
        })
    }
}

#[cfg(test)]
#[path = "compact_publisher_scan_tests.rs"]
mod tests;
