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

/// Logical source-event/typed-slot work, not cycles, RSS, authority or funding.
/// Evaluation itself needs future caller sponsorship. Only singleton, nonempty,
/// one-value append attempts are qualified; arbitrary chunks are not. Delegated
/// adapter/core work and completed/replacement bodies are separate components.
/// No live state, geometry, graph counts, admission or workload composition here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SourceScanWorkRates {
    pub preflight_p: u64,
    pub preflight_a: u64,
    pub preflight_s: u64,
    pub source_construct_j: u64,
    pub source_construct_reserve_error_k: u64,
    pub adapter_construct_u: u64,
    pub adapter_append_v: u64,
    pub adapter_finish_f: u64,
    pub adapter_terminal_error_mapper: u64,
    pub core_append: pllm_core::integer_bounds::MatrixScanWorkRates,
    pub core_lifecycle: pllm_core::integer_bounds::MatrixScanLifecycleRates,
    pub validation_fixed: u64,
    pub validation_record: u64,
    pub validation_block: u64,
    pub validation_terminal_error: u64,
    /// Includes scalar allowance 48, in addition to KN's typed extents.
    pub local_k_n: u64,
    /// Includes scalar allowance 16, in addition to KB's typed extents.
    pub local_k_b: u64,
    /// Includes scalar allowance 8, in addition to KR's typed extents.
    pub local_k_r: u64,
    /// Includes scalar allowance 16. Finish validation traversal is separate;
    /// this covers only consuming finish's receiver/result/Flow boundary.
    pub local_k_t: u64,
    pub source_retire_local: SourceRetireLocalRates,
    /// One terminal extra per source path, not per value. Inner core retirement
    /// and semantic retained blocks remain separate, each charged once.
    pub source_terminal_error: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SourceRetireLocalRates {
    pub source_shell: u64,
    pub record_shell: u64,
    pub retained_block: u64,
}

fn source_work_narrow(value: u128) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| Error::Bounds)
}

// Only fixed inventories supplied below; no collection, allocation or live scan.
fn source_work_sum<const N: usize>(terms: [u64; N]) -> Result<u64, Error> {
    terms.into_iter().try_fold(0u64, |sum, term| {
        sum.checked_add(term).ok_or(Error::Bounds)
    })
}

fn source_work_h<const N: usize>(k: u64, c: u64, types: [usize; N]) -> Result<u64, Error> {
    let sizes = types.into_iter().try_fold(0u64, |sum, bytes| {
        let bytes = source_work_narrow(u128::try_from(bytes).map_err(|_| Error::Bounds)?)?;
        sum.checked_add(bytes).ok_or(Error::Bounds)
    })?;
    sizes.checked_mul(k).and_then(|n| c.checked_add(n)).ok_or(Error::Bounds)
}

/// Pure checked fixed inventories. Distinct types remain distinct even when
/// their sizes coincide. No private core layout or adapter internals exposed.
/// Successful caller counts (not computed here): P*(2B+2T+Tv), A*2B,
/// S*2T, J*2T, U*2B, V*2N, F*2B; core construction is separate.
/// Local per-pass counts N,B,B-T,T require checked subtraction. N bounds both
/// append calls and forwarded loop iterations in the singleton domain only.
/// Validation counts are independent attempts/records/block iterations, plus
/// at most one terminal; whole-map traversal/lookup and payload P are excluded.
/// ResidentBlockFinish::finish_into has different ownership, not qualified.
pub(super) fn scan_work_rates() -> Result<SourceScanWorkRates, Error> {
    use pllm_core::integer_bounds::{Error as K, MatrixCertificate as CC, MatrixScan as CM};
    use std::{collections::TryReserveError, convert::Infallible, mem::size_of, ops::ControlFlow};
    type E = Error;
    type AM = MatrixScan;
    type AC = MatrixCertificate;
    type P = Pending;
    type W = BlockScan;
    type D = BlockNorms;
    type RE<X> = Result<X, E>;
    type RK<X> = Result<X, K>;
    type ZE = RE<Infallible>;
    type ZK = RK<Infallible>;
    type FE<X> = ControlFlow<ZE, X>;
    type FK<X> = ControlFlow<ZK, X>;
    // Each macro expands to a fixed array, never a layout stand-in or heap bank.
    macro_rules! h {
        ($k:expr, $c:expr; $($ty:ty),+ $(,)?) => {
            source_work_h($k, $c, [$(size_of::<$ty>()),+])
        };
    }
    macro_rules! extent {
        ($count:expr; $ty:ty) => { h!($count, 0; $ty) };
    }
    let cs = h!(256, 512; usize, bool, (), K, Option<usize>, RK<usize>, RK<()>, FK<usize>, ZK)?;
    let cp = h!(256, 512; usize, (), K, Option<usize>, RK<usize>, RK<()>, FK<()>, ZK)?;
    let as_ = h!(256, 512; usize, bool, (), E, RE<()>)?;
    let ap = h!(256, 512; usize, (), E, K, RE<()>, FE<()>, ZE, RK<usize>, RE<usize>)?;
    let preflight_p = source_work_sum([cs, cp, as_, ap, h!(256, 512; K, E)?])?;
    // A explicitly owns norm_bounds::new's payload caller/propagation (53-54).
    // U therefore must NOT duplicate RE<usize>/FE<usize>/ZE here.
    let preflight_a = h!(256, 512; usize, E, RE<usize>, FE<usize>, ZE, RE<AM>)?;
    let preflight_s = h!(256, 512; usize, bool, &usize, E, Option<usize>, RE<usize>, FE<usize>, ZE, RE<W>)?;
    let source_construct_j = h!(64, 512;
        usize, bool, (), E, Vec<Block>, &mut Vec<Block>, P, Option<P>, W,
        RE<W>, AM, Option<AM>, RE<AM>, FE<AM>, Option<usize>, RE<usize>,
        FE<usize>, RE<()>, FE<()>, ZE, Result<(), TryReserveError>)?;
    // Only reserve failure's one local Vec cleanup, before any live Pending.
    let source_construct_reserve_error_k = h!(8, 64;
        TryReserveError, E, Vec<Block>, &mut Vec<Block>, usize, bool)?;
    let adapter_construct_u = h!(8, 64; usize, CM, AM, RK<CM>, RK<AM>, RE<AM>)?;
    let adapter_append_v = source_work_sum([
        h!(1, 0; &mut AM, &mut CM, RK<()>)?, extent!(2; usize)?,
        extent!(2; &[i64])?, extent!(2; RE<()>)?, 8,
    ])?;
    let adapter_finish_f = h!(32, 256;
        AM, CM, CC, AC, usize, u128, E, RK<CC>, RE<CC>, FE<CC>, ZE, RE<AC>)?;
    // Actual terminal mapping only; P already owns its own preflight mapping.
    let adapter_terminal_error_mapper = h!(4, 16; K, E)?;

    let kn = source_work_sum([
        h!(1, 0; P, RE<P>, FE<P>, Option<&mut AM>, RE<&mut AM>, FE<&mut AM>,
            &mut AM, FE<()>, &mut W, Option<usize>)?,
        extent!(2; Option<P>)?, extent!(2; RE<()>)?, extent!(3; &[i64])?,
        extent!(2; E)?, extent!(32; usize)?, extent!(16; bool)?,
    ])?;
    let kb = source_work_sum([
        h!(1, 0; Option<AM>, RE<AM>, FE<AM>, AM, RE<AC>, FE<AC>, AC, Block, E)?,
        extent!(8; usize)?, extent!(4; bool)?,
    ])?;
    let kr = source_work_sum([
        h!(1, 0; RE<AM>, FE<AM>, AM, Option<AM>)?, extent!(8; usize)?,
    ])?;
    let kt = source_work_sum([
        h!(1, 0; W, Option<P>, RE<P>, FE<P>, P, D, RE<D>, RE<()>, FE<()>, E, &D)?,
        extent!(8; usize)?, extent!(4; bool)?,
    ])?;
    // Union by TYPE of KN/KB/KR/KT, once each; not weighted KN totals.
    let terminal_inventory = h!(1, 0;
        P, Option<P>, RE<P>, FE<P>, Option<&mut AM>, RE<&mut AM>, FE<&mut AM>,
        &mut AM, RE<()>, FE<()>, &mut W, &[i64], E, Option<usize>, usize, bool,
        Option<AM>, RE<AM>, FE<AM>, AM, RE<AC>, FE<AC>, AC, Block,
        W, D, RE<D>, &D)?;
    let source_terminal_error = source_work_sum([terminal_inventory, h!(1, 0; ZE)?])?
        .checked_mul(32).and_then(|n| n.checked_add(64)).ok_or(Error::Bounds)?;

    let validation_h = h!(1, 0;
        Block, AC, CC, (usize, usize), &D, &Vec<Block>, &[Block], &Block,
        Option<&Block>, Option<AC>, Option<usize>, Option<u128>, Option<Infallible>,
        ControlFlow<Option<Infallible>, &Block>, RE<usize>, RE<()>, RE<AC>, RE<CC>,
        RE<u128>, RK<usize>, RK<()>, RK<CC>, RK<u128>, FE<usize>, FE<()>, FE<AC>,
        FE<CC>, FE<u128>, FK<usize>, FK<()>, FK<CC>, ZE, ZK, E, K)?;
    let validation_j = h!(1, 0; usize, u128, bool, ())?;
    let validation_u = source_work_sum([
        256, validation_h.checked_mul(64).ok_or(Error::Bounds)?,
        validation_j.checked_mul(128).ok_or(Error::Bounds)?,
    ])?;
    let validation_two = validation_u.checked_mul(2).ok_or(Error::Bounds)?;
    let validation_nine = validation_u.checked_mul(9).ok_or(Error::Bounds)?;
    Ok(SourceScanWorkRates {
        preflight_p, preflight_a, preflight_s, source_construct_j,
        source_construct_reserve_error_k, adapter_construct_u, adapter_append_v,
        adapter_finish_f, adapter_terminal_error_mapper,
        core_append: CM::work_rates().map_err(|_| Error::Bounds)?,
        core_lifecycle: CM::lifecycle_work_rates().map_err(|_| Error::Bounds)?,
        validation_fixed: validation_two,
        validation_record: validation_two,
        validation_block: validation_nine,
        validation_terminal_error: validation_nine,
        local_k_n: kn.checked_add(48).ok_or(Error::Bounds)?,
        local_k_b: kb.checked_add(16).ok_or(Error::Bounds)?,
        local_k_r: kr.checked_add(8).ok_or(Error::Bounds)?,
        local_k_t: kt.checked_add(16).ok_or(Error::Bounds)?,
        source_retire_local: SourceRetireLocalRates {
            source_shell: h!(1, 8; P, W)?,
            record_shell: h!(1, 8; D, Vec<Block>)?,
            retained_block: h!(1, 0; Block)?,
        },
        source_terminal_error,
    })
}

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
        validate_parts(
            self.rows,
            self.columns,
            self.block_rows,
            &self.blocks,
            rows,
            columns,
        )
    }

    pub(super) fn geometry(&self) -> (usize, usize, usize) {
        (self.rows, self.columns, self.block_rows)
    }

    pub(super) fn block(&self, start: usize, rows: usize) -> Option<MatrixCertificate> {
        block_parts(
            self.rows,
            self.columns,
            self.block_rows,
            &self.blocks,
            start,
            rows,
        )
    }

    pub(super) fn metadata_bytes(&self) -> Result<usize, Error> {
        block_metadata_bytes(&self.blocks)
    }
}

// Vec, rather than a slice: retained capacity is part of the metadata bound.
fn block_metadata_bytes(blocks: &Vec<Block>) -> Result<usize, Error> {
    blocks
        .capacity()
        .checked_mul(std::mem::size_of::<Block>())
        .and_then(|n| n.checked_add(std::mem::size_of::<BlockNorms>()))
        .ok_or(Error::Bounds)
}

fn block_parts(
    actual_rows: usize,
    actual_columns: usize,
    block_rows: usize,
    blocks: &[Block],
    start: usize,
    rows: usize,
) -> Option<MatrixCertificate> {
    if block_rows == 0 || start % block_rows != 0 || rows == 0 || start >= actual_rows {
        return None;
    }
    let record = *blocks.get(start / block_rows)?;
    if rows != block_rows.min(actual_rows - start) {
        return None;
    }
    Some(MatrixCertificate {
        rows,
        columns: actual_columns,
        maximum_row_l1: record.maximum_row_l1,
        maximum_column_l1: record.maximum_column_l1,
        frobenius_squared: record.frobenius_squared,
    })
}

fn validate_parts(
    actual_rows: usize,
    actual_columns: usize,
    block_rows: usize,
    blocks: &Vec<Block>,
    rows: usize,
    columns: usize,
) -> Result<(), Error> {
    MatrixScan::payload_bytes(block_rows, columns)?;
    if (actual_rows, actual_columns) != (rows, columns)
        || rows == 0
        || blocks.len() != rows.div_ceil(block_rows)
        || blocks.len() > MAX_BLOCKS
        || block_metadata_bytes(blocks)? > MAX_METADATA_BYTES
    {
        return Err(Error::Shape);
    }
    for index in 0..blocks.len() {
        let rows = block_rows.min(actual_rows - index * block_rows);
        block_parts(
            actual_rows,
            actual_columns,
            block_rows,
            blocks,
            index * block_rows,
            rows,
        )
        .ok_or(Error::Shape)?
        .operator_squared()?;
    }
    Ok(())
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

/// One-use finish custody. The caller must retain this owner and its output slot
/// outside the fallible/unwinding operation. Refusal or unwind leaves the scan
/// resident for cleanup only, never for retry. This is not a charge/guard proof;
/// wrapper/state/output-slot quotation and native integration remain open.
pub(super) struct ResidentBlockFinish {
    scan: BlockScan,
    state: FinishState,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FinishState {
    Ready,
    CleanupOnly,
    Complete,
}

impl ResidentBlockFinish {
    pub(super) fn finish_into(&mut self, out: &mut Option<BlockNorms>) -> Result<(), Error> {
        let prior = std::mem::replace(&mut self.state, FinishState::CleanupOnly);
        if out.is_some() {
            return Err(Error::Binding);
        }
        if prior != FinishState::Ready {
            return Err(Error::Binding);
        }
        let pending = self.scan.0.as_mut().ok_or(Error::Binding)?;
        if pending.offset != pending.rows * pending.columns || pending.scan.is_some() {
            return Err(Error::Shape);
        }
        validate_parts(
            pending.rows,
            pending.columns,
            pending.block_rows,
            &pending.blocks,
            pending.rows,
            pending.columns,
        )?;
        #[cfg(test)]
        PANIC_FINISH_PRECOMMIT.with(|flag| {
            if flag.replace(false) {
                panic!("test resident block finish precommit unwind");
            }
        });

        let rows = pending.rows;
        let columns = pending.columns;
        let block_rows = pending.block_rows;
        // All preflight is complete. Transfer the original Vec into the known
        // empty slot; no allocation, callback or fallible operation in commit.
        let blocks = std::mem::take(&mut pending.blocks);
        *out = Some(BlockNorms {
            rows,
            columns,
            block_rows,
            blocks,
        });
        self.state = FinishState::Complete;
        Ok(())
    }
}

impl BlockScan {
    pub(super) fn into_resident_finish(self) -> ResidentBlockFinish {
        ResidentBlockFinish {
            scan: self,
            state: FinishState::Ready,
        }
    }

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
                Op::GeluErf | Op::Tanh | Op::AttentionMask => return Err(Error::Binding),
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

#[cfg(test)]
thread_local! {
    static PANIC_FINISH_PRECOMMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
#[path = "source_norms_resident_finish_tests.rs"]
mod resident_finish_tests;
