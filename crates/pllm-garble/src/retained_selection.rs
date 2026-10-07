//! Bounded client-garbled stable top-k and oblivious retained-KV movement.
//!
//! This is a research substrate, not MPCache's distributed scoring protocol.
//! The trusted client supplies finite nonnegative float32 scores and all KV
//! bits. Fixed networks move complete records or sort compact indices followed
//! by an oblivious full-scan gather, always under opaque labels;
//! only the client can decode positions or values. No provider-held persistent
//! state, protected score calculation, transport codec or malicious proof exists.

use crate::boolean::{
    BooleanCircuitBuilder, BooleanCircuitDecoder, BooleanCircuitInputs, BooleanCircuitOutputs,
    BooleanCircuitProgram, BooleanWire, LABEL_BYTES,
};
use pllm_core::retained_kv::{RetainedKvCache, RetainedKvView};
use zeroize::Zeroizing;

pub const MAX_ROWS: usize = 512;
pub const MAX_WIDTH: usize = 8192;
pub const MAX_ALLOCATION_BYTES: usize = 64 << 20;

mod cost;
mod indexed;
pub use cost::{project, Projection};

/// Public circuit choice, bound into each one-use receipt. Neither layout
/// reveals the selected index to the evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    FullRecords,
    CompactIndices,
}

fn index_bits(rows: usize) -> usize {
    (rows.next_power_of_two().trailing_zeros() as usize).max(4)
}

fn rank_bits(rows: usize) -> usize {
    index_bits(rows) + 34 // float32 score, recent, valid
}

/// Caller-declared public context. Its authenticity is the client's obligation;
/// this Rust-only reference does not load or validate a compiler plan itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Context {
    pub plan: [u8; 32],
    pub source: [u8; 32],
    pub state_contract: [u8; 32],
    pub layer: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    pub keep: usize,
    /// Mandatory newest active rows, independent of their scores.
    pub recent: usize,
}

/// Contains only public context, random owner/issuance identities and geometry.
/// It deliberately contains no digest of private scores, KV or selected indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub context: Context,
    pub owner: [u8; 32],
    pub revision: u64,
    pub issuance: [u8; 32],
    pub rows: usize,
    pub width: usize,
    pub next_position: usize,
    pub policy: Policy,
    pub layout: Layout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Estimate {
    pub wire_bound: usize,
    pub combined_allocation_bound_bytes: usize,
    pub input_bits: usize,
    pub output_bits: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub estimate: Estimate,
    pub public_schedule_digest: [u8; 32],
    pub and_gates: usize,
    pub wires: usize,
    pub evaluator_ciphertext_bytes: u64,
    pub constant_label_bytes: usize,
    pub input_label_bytes: usize,
    pub output_label_bytes: usize,
    pub client_encoding_bytes: usize,
    /// Fixed-width public instruction/index estimate, not serialized traffic.
    pub public_metadata_bytes_estimate: usize,
}

fn comparisons(rows: usize) -> usize {
    let n = rows.next_power_of_two();
    let levels = n.trailing_zeros() as usize;
    n * levels * (levels + 1) / 4
}

pub fn estimate(
    rows: usize,
    width: usize,
    policy: Policy,
    state_allocation_bytes: usize,
) -> Result<Estimate, String> {
    estimate_layout(
        rows,
        width,
        policy,
        state_allocation_bytes,
        Layout::FullRecords,
    )
}

pub fn estimate_layout(
    rows: usize,
    width: usize,
    policy: Policy,
    state_allocation_bytes: usize,
    layout: Layout,
) -> Result<Estimate, String> {
    let projection = project(rows, width, policy, state_allocation_bytes, layout)?;
    if !projection.admitted() {
        return Err("retained selection exceeds its allocation or circuit boundary limit".into());
    }
    Ok(projection.estimate)
}

/// Client-only private owner. Moving the cache here prevents external mutation.
/// One outstanding issuance per owner; no Clone, serialization or secret Debug.
pub struct SelectionOwner {
    cache: RetainedKvCache,
    context: Context,
    identity: [u8; 32],
    revision: u64,
    pending: Option<[u8; 32]>,
}

impl SelectionOwner {
    pub fn new(context: Context, cache: RetainedKvCache) -> Result<Self, String> {
        if context.plan == [0; 32]
            || context.source == [0; 32]
            || context.state_contract == [0; 32]
            || context.layer >= 128
        {
            return Err("retained selection context is invalid".into());
        }
        cache.active()?;
        let mut identity = [0; 32];
        getrandom::fill(&mut identity).map_err(|_| "selection owner randomness failed")?;
        Ok(Self {
            cache,
            context,
            identity,
            revision: 0,
            pending: None,
        })
    }

    pub fn active(&self) -> Result<RetainedKvView<'_>, String> {
        self.cache.active().map_err(Into::into)
    }

    pub fn allocated_bytes(&self) -> usize {
        self.cache.allocated_bytes()
    }

    pub fn append(&mut self, first: usize, keys: &[f32], values: &[f32]) -> Result<(), String> {
        let revision = self
            .revision
            .checked_add(1)
            .ok_or("selection revision exhausted")?;
        self.cache.append(first, keys, values)?;
        self.revision = revision;
        self.pending = None;
        Ok(())
    }

    /// Invalidates the outstanding receipt; dropping its two handles erases keys.
    pub fn cancel(&mut self) {
        self.pending = None;
    }

    pub fn close(&mut self) {
        self.cancel();
        self.cache.close();
    }

    pub fn issue(
        &mut self,
        scores: &[f32],
        policy: Policy,
    ) -> Result<(SelectionClient, SelectionEvaluator), String> {
        self.issue_with_layout(scores, policy, Layout::FullRecords)
    }

    pub fn issue_with_layout(
        &mut self,
        scores: &[f32],
        policy: Policy,
        layout: Layout,
    ) -> Result<(SelectionClient, SelectionEvaluator), String> {
        let (keys, values, positions) = self.cache.active()?;
        if self.pending.is_some() {
            return Err("retained selection already has an outstanding query".into());
        }
        if scores.len() != positions.len() || scores.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("retained selection requires one finite nonnegative score per row".into());
        }
        let bound = estimate_layout(
            positions.len(),
            self.cache.width(),
            policy,
            self.cache.allocated_bytes(),
            layout,
        )?;
        self.revision
            .checked_add(1)
            .ok_or("selection revision exhausted")?;
        let width = self.cache.width();
        let (encoder, program) = match layout {
            Layout::FullRecords => circuit(positions.len(), width, policy)?,
            Layout::CompactIndices => indexed::circuit(positions.len(), width, policy)?,
        };
        if program.wire_count() > bound.wire_bound {
            return Err("retained selection circuit exceeds preflight".into());
        }
        let binding = Binding {
            context: self.context,
            owner: self.identity,
            revision: self.revision,
            issuance: program.issuance_id(),
            rows: positions.len(),
            width,
            next_position: self.cache.next_position(),
            policy,
            layout,
        };
        let resources = Resources {
            estimate: bound,
            public_schedule_digest: program.public_schedule_digest(),
            and_gates: program.and_gate_count(),
            wires: program.wire_count(),
            evaluator_ciphertext_bytes: program.evaluator_ciphertext_bytes()?,
            constant_label_bytes: program.constant_label_bytes(),
            input_label_bytes: bound.input_bits * LABEL_BYTES,
            output_label_bytes: bound.output_bits * LABEL_BYTES,
            client_encoding_bytes: (bound.input_bits + bound.output_bits) * 2 * LABEL_BYTES,
            public_metadata_bytes_estimate: program.public_metadata_bytes_estimate(),
        };
        let mut bits = Zeroizing::new(Vec::with_capacity(bound.input_bits));
        for (row, (&position, &score)) in positions.iter().zip(scores).enumerate() {
            push_word(&mut bits, if score == 0.0 { 0 } else { score.to_bits() });
            push_word(&mut bits, position);
            for v in keys[row * width..(row + 1) * width]
                .iter()
                .chain(&values[row * width..(row + 1) * width])
            {
                push_word(&mut bits, v.to_bits());
            }
        }
        let (inputs, decoder) = encoder.encode(&bits)?;
        self.pending = Some(binding.issuance);
        Ok((
            SelectionClient { binding, decoder },
            SelectionEvaluator {
                binding,
                program,
                inputs,
                resources,
            },
        ))
    }
}

pub struct SelectionClient {
    binding: Binding,
    decoder: BooleanCircuitDecoder,
}

pub struct SelectionEvaluator {
    binding: Binding,
    program: BooleanCircuitProgram,
    inputs: BooleanCircuitInputs,
    resources: Resources,
}

pub struct SelectionOutput {
    binding: Binding,
    outputs: BooleanCircuitOutputs,
}

impl SelectionEvaluator {
    pub fn binding(&self) -> Binding {
        self.binding
    }
    pub fn resources(&self) -> Resources {
        self.resources
    }

    /// Consumes material even when context validation fails. The worker receives
    /// no output decoder and performs the same accesses for every private input.
    pub fn evaluate(self, expected: Binding) -> Result<SelectionOutput, String> {
        if self.binding != expected {
            return Err("retained selection evaluator context mismatch".into());
        }
        Ok(SelectionOutput {
            binding: self.binding,
            outputs: self.program.evaluate(self.inputs)?,
        })
    }
}

impl SelectionClient {
    /// Commit only to the exact live owner/revision that issued this query.
    /// Validation/decoding consumes this handle; successful commit preserves
    /// remaining decode capacity and atomically retires discarded native rows.
    pub fn finish(self, owner: &mut SelectionOwner, output: SelectionOutput) -> Result<(), String> {
        let b = self.binding;
        if owner.identity != b.owner || owner.pending != Some(b.issuance) {
            return Err("retained selection owner or issuance is stale".into());
        }
        owner.pending = None; // burn before any subsequent validation or decoding
        if output.binding != b
            || owner.context != b.context
            || owner.revision != b.revision
            || owner.cache.len() != b.rows
            || owner.cache.width() != b.width
            || owner.cache.next_position() != b.next_position
        {
            return Err("retained selection state or output context mismatch".into());
        }
        let bits = Zeroizing::new(self.decoder.decode(output.outputs)?);
        let mut selected = Zeroizing::new(Vec::with_capacity(b.policy.keep));
        let (keys, values, positions) = owner.cache.active()?;
        let row_bits = 32 + 64 * b.width;
        if bits.len() != b.policy.keep * row_bits {
            return Err("retained selection output shape mismatch".into());
        }
        for record in bits.chunks_exact(row_bits) {
            let position = read_word(&record[..32]);
            if selected.last().is_some_and(|&p| p >= position as usize) {
                return Err("retained selection output positions are not ordered".into());
            }
            let row = positions
                .binary_search(&position)
                .map_err(|_| "retained selection output position is not owned")?;
            for (encoded, value) in record[32..].chunks_exact(32).zip(
                keys[row * b.width..(row + 1) * b.width]
                    .iter()
                    .chain(&values[row * b.width..(row + 1) * b.width]),
            ) {
                if read_word(encoded) != value.to_bits() {
                    return Err("retained selection output KV does not match its owner".into());
                }
            }
            selected.push(position as usize);
        }
        let capacity = b.policy.keep + owner.cache.capacity() - owner.cache.len();
        let revision = owner
            .revision
            .checked_add(1)
            .ok_or("selection revision exhausted")?;
        owner.cache.retain(&selected, capacity)?;
        owner.revision = revision;
        Ok(())
    }
}

fn push_word(bits: &mut Vec<bool>, value: u32) {
    bits.extend((0..32).map(|bit| value & (1 << bit) != 0));
}

fn read_word(bits: &[bool]) -> u32 {
    bits.iter()
        .enumerate()
        .fold(0, |n, (bit, &v)| n | ((v as u32) << bit))
}

fn circuit(
    rows: usize,
    width: usize,
    policy: Policy,
) -> Result<(crate::boolean::BooleanCircuitClient, BooleanCircuitProgram), String> {
    let mut builder = BooleanCircuitBuilder::new()?;
    let mut inputs = Vec::with_capacity(rows);
    // Declare all private inputs before any gates or public constants.
    for _ in 0..rows {
        let score = builder.input_word(32)?;
        let mut payload = builder.input_word(32)?;
        for _ in 0..2 * width {
            payload.extend(builder.input_word(32)?);
        }
        inputs.push((score, payload));
    }
    let rank = rank_bits(rows);
    let index = index_bits(rows);
    let record_bits = rank + 32 + 64 * width;
    let mut records = Vec::with_capacity(rows.next_power_of_two());
    for (slot, (score, payload)) in inputs.into_iter().enumerate() {
        let mut row = builder.constant_word(((1 << index) - 1 - slot) as u128, index)?;
        row.extend(score);
        row.push(builder.constant(slot >= rows - policy.recent)?);
        row.push(builder.constant(true)?);
        row.extend(payload);
        records.push(row);
    }
    while records.len() < rows.next_power_of_two() {
        records.push(
            (0..record_bits)
                .map(|_| builder.constant(false))
                .collect::<Result<_, _>>()?,
        );
    }
    sort(&mut builder, &mut records, 0..rank, false)?;
    records.truncate(policy.keep);
    while records.len() < policy.keep.next_power_of_two() {
        // Dummy positions sort after every real absolute token position.
        records.push(
            (0..record_bits)
                .map(|_| builder.constant(true))
                .collect::<Result<_, _>>()?,
        );
    }
    sort(&mut builder, &mut records, rank..rank + 32, true)?;
    let output = records
        .iter()
        .take(policy.keep)
        .flat_map(|row| row[rank..].iter().copied())
        .collect::<Vec<_>>();
    builder.finish(&output)
}

fn sort(
    builder: &mut BooleanCircuitBuilder,
    records: &mut [Vec<BooleanWire>],
    key: std::ops::Range<usize>,
    ascending: bool,
) -> Result<(), String> {
    let mut group = 2;
    while group <= records.len() {
        let mut distance = group / 2;
        while distance != 0 {
            for left in 0..records.len() {
                let right = left ^ distance;
                if right <= left {
                    continue;
                }
                let increasing = ((left & group) == 0) == ascending;
                let (a, b) = if increasing {
                    (right, left)
                } else {
                    (left, right)
                };
                let (swap, _) =
                    builder.compare_unsigned(&records[a][key.clone()], &records[b][key.clone()])?;
                for bit in 0..records[left].len() {
                    let different = builder.xor(records[left][bit], records[right][bit])?;
                    let move_bit = builder.and(swap, different)?;
                    records[left][bit] = builder.xor(records[left][bit], move_bit)?;
                    records[right][bit] = builder.xor(records[right][bit], move_bit)?;
                }
            }
            distance /= 2;
        }
        group *= 2;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
