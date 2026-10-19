//! Bounded source-ordered row-block certificates. Each worker recomputes these
//! public records before the owning immutable source can enter execution.

use super::norm_bounds::{MatrixCertificate, MatrixScan};
use pllm_core::puma::Error;
use pllm_models::{DecoderGraph, ModelOperator as Op};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

const MAX_BLOCKS: usize = 4096;
pub(super) const MAX_METADATA_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Block {
    maximum_row_l1: u128,
    maximum_column_l1: u128,
    frobenius_squared: u128,
}

/// Public source metadata, never a proof of a caller's private values. Shape,
/// exact source contents and every statistic are verified by the source owner.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockNorms {
    rows: usize,
    columns: usize,
    block_rows: usize,
    blocks: Vec<Block>,
}

pub type SourceNorms = BTreeMap<String, BlockNorms>;

/// Native norm-metadata payload envelope, not allocator overhead or process RSS.
/// Covers four whole-set owners (input, construction/import clones and overlap),
/// one geometry-request map, and the largest scan plus its completed output.
/// Caller-owned graph, source/frame buffers and serialized manifest bytes are
/// excluded. Call before cloning records or constructing a scan. Existing
/// geometry maps are borrowed; this calculation performs no allocations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ImportEnvelope {
    pub retained_bytes: usize,
    pub request_bytes: usize,
    pub largest_scan_bytes: usize,
    pub largest_output_bytes: usize,
    pub total_bytes: usize,
}

// Charge a full, generously sized tree node PER entry, including unused slots
// and child edges. Current native std BTree nodes have at most 11 key/value
// slots and 12 edges; 16 slots and 128 bookkeeping bytes deliberately overcount.
// No allocator headers, caller buffers or empirical RSS enter this allowance.
fn tree_entry_bytes<T>() -> Result<usize, Error> {
    std::mem::size_of::<T>()
        .checked_add(std::mem::size_of::<usize>())
        .and_then(|n| n.checked_mul(16))
        .and_then(|n| n.checked_add(128))
        .ok_or(Error::Bounds)
}

impl ImportEnvelope {
    fn add(
        &mut self,
        key_bytes: usize,
        rows: usize,
        columns: usize,
        width: usize,
        record_bytes: usize,
    ) -> Result<(), Error> {
        let scan = BlockScan::payload_bytes(rows, columns, width)?;
        let norm_node = tree_entry_bytes::<(String, BlockNorms)>()?;
        let record = record_bytes
            .checked_add(key_bytes)
            .and_then(|n| n.checked_add(norm_node))
            .ok_or(Error::Bounds)?;
        let request = key_bytes
            .checked_add(tree_entry_bytes::<(String, (usize, usize, usize))>()?)
            .ok_or(Error::Bounds)?;
        if self.retained_bytes == 0 {
            self.retained_bytes = std::mem::size_of::<SourceNorms>();
            self.request_bytes = std::mem::size_of::<BTreeMap<String, (usize, usize, usize)>>();
        }
        self.retained_bytes = self
            .retained_bytes
            .checked_add(record)
            .ok_or(Error::Bounds)?;
        self.request_bytes = self
            .request_bytes
            .checked_add(request)
            .ok_or(Error::Bounds)?;
        self.largest_scan_bytes = self.largest_scan_bytes.max(scan);
        self.largest_output_bytes = self.largest_output_bytes.max(record_bytes);
        self.total_bytes = self
            .retained_bytes
            .checked_mul(4)
            .and_then(|n| n.checked_add(self.request_bytes))
            .and_then(|n| n.checked_add(self.largest_scan_bytes))
            .and_then(|n| n.checked_add(self.largest_output_bytes))
            .filter(|n| *n <= MAX_METADATA_BYTES)
            .ok_or(Error::Bounds)?;
        Ok(())
    }
}

/// Builder and worker-import preflight. Includes actual Vec/String capacities,
/// so oversized caller reservations cannot hide behind their logical lengths.
pub(super) fn import_envelope(norms: &SourceNorms) -> Result<ImportEnvelope, Error> {
    metadata_bytes(norms)?;
    let mut envelope = ImportEnvelope::default();
    for (key, record) in norms {
        record.validate(record.rows, record.columns)?;
        envelope.add(
            key.capacity(),
            record.rows,
            record.columns,
            record.block_rows,
            record.metadata_bytes()?,
        )?;
    }
    Ok(envelope)
}

/// Publisher preflight before the first BlockScan or retained norm-map clone.
/// BlockScan reserves the complete planned record vector exactly. Requests are
/// normally produced by `requests`; malformed/overflowing geometries reject too.
pub(super) fn requests_envelope(
    requests: &BTreeMap<String, (usize, usize, usize)>,
) -> Result<ImportEnvelope, Error> {
    let mut envelope = ImportEnvelope::default();
    for (key, &(rows, columns, width)) in requests {
        // Validate before division, including width==0 and product overflow.
        BlockScan::payload_bytes(rows, columns, width)?;
        let record = rows
            .div_ceil(width)
            .checked_mul(std::mem::size_of::<Block>())
            .and_then(|n| n.checked_add(std::mem::size_of::<BlockNorms>()))
            .ok_or(Error::Bounds)?;
        envelope.add(key.capacity(), rows, columns, width, record)?;
    }
    Ok(envelope)
}

pub(super) fn digest(norms: &SourceNorms) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"pllm.replicated-decoder.public-source-block-norms.v1");
    h.update((norms.len() as u64).to_le_bytes());
    for (key, norms) in norms {
        h.update((key.len() as u64).to_le_bytes());
        h.update(key);
        for value in [
            norms.rows,
            norms.columns,
            norms.block_rows,
            norms.blocks.len(),
        ] {
            h.update((value as u64).to_le_bytes());
        }
        for block in &norms.blocks {
            for value in [
                block.maximum_row_l1,
                block.maximum_column_l1,
                block.frobenius_squared,
            ] {
                h.update(value.to_le_bytes());
            }
        }
    }
    h.finalize().into()
}

pub(super) fn metadata_bytes(norms: &SourceNorms) -> Result<usize, Error> {
    norms.iter().try_fold(0usize, |bytes, (key, record)| {
        bytes
            .checked_add(record.metadata_bytes()?)
            .and_then(|n| n.checked_add(key.capacity()))
            .and_then(|n| n.checked_add(256))
            .filter(|n| *n <= MAX_METADATA_BYTES)
            .ok_or(Error::Bounds)
    })
}

pub(super) fn validate_set(
    graph: &DecoderGraph,
    shapes: &BTreeMap<String, Vec<usize>>,
    norms: &SourceNorms,
) -> Result<(), Error> {
    metadata_bytes(norms)?;
    if norms.is_empty() {
        return Ok(());
    }
    let required = requests(graph)?;
    if required.len() != norms.len() {
        return Err(Error::Binding);
    }
    for (key, (rows, columns, block_rows)) in required {
        let record = norms.get(&key).ok_or(Error::Binding)?;
        let shape = shapes.get(&key).ok_or(Error::Binding)?;
        if record.geometry() != (rows, columns, block_rows)
            || (shape.as_slice() != [rows, columns]
                && !(columns == 1 && shape.as_slice() == [rows]))
        {
            return Err(Error::Shape);
        }
        record.validate(rows, columns)?;
    }
    Ok(())
}

impl BlockNorms {
    pub(super) fn validate(&self, rows: usize, columns: usize) -> Result<(), Error> {
        MatrixScan::payload_bytes(self.block_rows, columns)?;
        if (self.rows, self.columns) != (rows, columns)
            || rows == 0
            || self.blocks.len() != rows.div_ceil(self.block_rows)
            || self.blocks.len() > MAX_BLOCKS
            || self.metadata_bytes()? > MAX_METADATA_BYTES
        {
            return Err(Error::Shape);
        }
        for index in 0..self.blocks.len() {
            let rows = self.block_rows.min(self.rows - index * self.block_rows);
            self.block(index * self.block_rows, rows)
                .ok_or(Error::Shape)?
                .operator_squared()?;
        }
        Ok(())
    }

    pub(super) fn geometry(&self) -> (usize, usize, usize) {
        (self.rows, self.columns, self.block_rows)
    }

    pub(super) fn block(&self, start: usize, rows: usize) -> Option<MatrixCertificate> {
        if self.block_rows == 0 || start % self.block_rows != 0 || rows == 0 || start >= self.rows {
            return None;
        }
        let record = *self.blocks.get(start / self.block_rows)?;
        if rows != self.block_rows.min(self.rows - start) {
            return None;
        }
        Some(MatrixCertificate {
            rows,
            columns: self.columns,
            maximum_row_l1: record.maximum_row_l1,
            maximum_column_l1: record.maximum_column_l1,
            frobenius_squared: record.frobenius_squared,
        })
    }

    pub(super) fn metadata_bytes(&self) -> Result<usize, Error> {
        self.blocks
            .capacity()
            .checked_mul(std::mem::size_of::<Block>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(Error::Bounds)
    }
}

struct Pending {
    rows: usize,
    columns: usize,
    block_rows: usize,
    offset: usize,
    scan: Option<MatrixScan>,
    blocks: Vec<Block>,
}

/// Holds only column totals for the current block and completed public records.
/// Errors retire the entire scan; source import owns the enclosing transaction.
pub(super) struct BlockScan(Option<Pending>);

impl BlockScan {
    pub(super) fn payload_bytes(
        rows: usize,
        columns: usize,
        block_rows: usize,
    ) -> Result<usize, Error> {
        let scratch = MatrixScan::payload_bytes(block_rows, columns)?;
        if rows == 0
            || rows.div_ceil(block_rows) > MAX_BLOCKS
            || rows.checked_mul(columns).is_none()
        {
            return Err(Error::Bounds);
        }
        let retained = rows
            .div_ceil(block_rows)
            .checked_mul(std::mem::size_of::<Block>())
            .and_then(|n| n.checked_add(std::mem::size_of::<BlockNorms>()))
            .filter(|n| *n <= MAX_METADATA_BYTES)
            .ok_or(Error::Bounds)?;
        retained
            .checked_add(scratch)
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<Self>() + std::mem::size_of::<Pending>())
            })
            .ok_or(Error::Bounds)
    }

    pub(super) fn new(rows: usize, columns: usize, block_rows: usize) -> Result<Self, Error> {
        Self::payload_bytes(rows, columns, block_rows)?;
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(rows.div_ceil(block_rows))
            .map_err(|_| Error::Allocation)?;
        Ok(Self(Some(Pending {
            rows,
            columns,
            block_rows,
            offset: 0,
            scan: Some(MatrixScan::new(rows.min(block_rows), columns)?),
            blocks,
        })))
    }

    pub(super) fn append(&mut self, offset: usize, mut values: &[i64]) -> Result<(), Error> {
        let mut pending = self.0.take().ok_or(Error::Binding)?;
        if offset != pending.offset
            || values.is_empty()
            || offset
                .checked_add(values.len())
                .is_none_or(|n| n > pending.rows * pending.columns)
        {
            return Err(Error::Shape);
        }
        while !values.is_empty() {
            let first_row = pending.blocks.len() * pending.block_rows;
            let rows = (pending.rows - first_row).min(pending.block_rows);
            let local = pending.offset - first_row * pending.columns;
            let count = values.len().min(rows * pending.columns - local);
            pending
                .scan
                .as_mut()
                .ok_or(Error::Binding)?
                .append(local, &values[..count])?;
            pending.offset += count;
            values = &values[count..];
            if local + count == rows * pending.columns {
                let record = pending.scan.take().ok_or(Error::Binding)?.finish()?;
                pending.blocks.push(Block {
                    maximum_row_l1: record.maximum_row_l1,
                    maximum_column_l1: record.maximum_column_l1,
                    frobenius_squared: record.frobenius_squared,
                });
                let next = first_row + rows;
                if next < pending.rows {
                    pending.scan = Some(MatrixScan::new(
                        (pending.rows - next).min(pending.block_rows),
                        pending.columns,
                    )?);
                }
            }
        }
        self.0 = Some(pending);
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<BlockNorms, Error> {
        let pending = self.0.take().ok_or(Error::Binding)?;
        if pending.offset != pending.rows * pending.columns || pending.scan.is_some() {
            return Err(Error::Shape);
        }
        let result = BlockNorms {
            rows: pending.rows,
            columns: pending.columns,
            block_rows: pending.block_rows,
            blocks: pending.blocks,
        };
        result.validate(pending.rows, pending.columns)?;
        Ok(result)
    }
}

/// Discover original Q/K projection matrices through declared semantic producers,
/// not checkpoint path strings. Biases use rows-by-one source geometry. Unknown
/// transforms reject rather than assigning a certificate to the wrong tensor.
pub(super) fn requests(
    graph: &DecoderGraph,
) -> Result<BTreeMap<String, (usize, usize, usize)>, Error> {
    let mut result = BTreeMap::new();
    let producers: BTreeMap<_, _> = graph
        .operations
        .iter()
        .map(|op| (op.id.as_str(), op))
        .collect();
    for rotary in graph
        .operations
        .iter()
        .filter(|op| op.operator == Op::RotaryEmbedding)
    {
        if rotary.output_shape.len() != 4 {
            return Err(Error::Shape);
        }
        let width = usize::try_from(rotary.output_shape[3]).map_err(|_| Error::Bounds)?;
        let mut source = rotary.inputs.first().ok_or(Error::Binding)?.as_str();
        let mut projection = None;
        for _ in 0..graph.operations.len() {
            let op = producers.get(source).ok_or(Error::Binding)?;
            match op.operator {
                Op::Linear => {
                    projection = Some(*op);
                    break;
                }
                Op::Reshape | Op::RmsNorm => {
                    source = op.inputs.first().ok_or(Error::Binding)?.as_str();
                }
                _ => return Err(Error::Binding),
            }
        }
        let op = projection.ok_or(Error::Binding)?;
        let key = op
            .attributes
            .get("weight")
            .and_then(|v| v.as_str())
            .ok_or(Error::Binding)?;
        let input = producers
            .get(op.inputs.first().ok_or(Error::Binding)?.as_str())
            .ok_or(Error::Binding)?;
        let rows = usize::try_from(*op.output_shape.last().ok_or(Error::Shape)?)
            .map_err(|_| Error::Bounds)?;
        let columns = usize::try_from(*input.output_shape.last().ok_or(Error::Shape)?)
            .map_err(|_| Error::Bounds)?;
        if rows % width.max(1) != 0 {
            return Err(Error::Shape);
        }
        BlockScan::payload_bytes(rows, columns, width)?;
        for (key, geometry) in std::iter::once((key, (rows, columns, width))).chain(
            op.attributes
                .get("bias")
                .and_then(|v| v.as_str())
                .map(|key| (key, (rows, 1, width))),
        ) {
            if result
                .insert(key.to_owned(), geometry)
                .is_some_and(|old| old != geometry)
            {
                return Err(Error::Binding);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "source_norms_tests.rs"]
mod tests;
