//! M-X-I metadata only. No matrices, randomness, claims, protocol owners or output.
//!
//! Input is an ordered immutable original/source catalog supplied by its owner;
//! this module neither traverses a decoder nor authenticates that catalog. Core
//! cannot import the compiler's private traversal (compiler depends on core).
//! Reuse that traversal upstream, not operation-name parsing or a model builder.
//! The 256/128/B=2 proposal is NOT a supported `AllocationPlan::for_shape`.
//! Metadata reservations never authorize preprocessing or a protocol bridge.

use super::{ALLOCATION_CEILING, BATCH_SIZE, CHECK_WEIGHT, REPEAT, REPETITIONS};

pub const MAX_RECORDS: usize = 4096;
const ROW: u64 = 256;
const COLUMN: u64 = 128;
const TX: u64 = 164; // Proposed horizon only; not the capsule's privacy parameter.
const CLIENT_CAP: u64 = 2 << 30;
const STORE_CAP: u64 = 6 << 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    Geometry,
    Binding,
    Records,
    Overflow,
    UnknownAttachment,
    Capacity,
    Work,
    Order,
    OwnerBridgeMissing,
}
type Result<T> = std::result::Result<T, Refused>;
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or(Refused::Overflow)
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or(Refused::Overflow)
}
fn sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0, |a, &b| add(a, b))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Use {
    EightPositions,
    FinalPosition,
}

/// Digests are supplied references, NOT authenticated source/OriginalSourcePlan.
#[derive(Debug, Clone, Copy)]
pub struct OriginalMatrix {
    pub operation: [u8; 32],
    pub source: [u8; 32],
    pub rows: u64,
    pub columns: u64,
    pub usage: Use,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Summary {
    pub originals: u64,
    pub unpartitioned_claims: u64,
    pub tiles: u64,
    pub groups: u64,
    pub batches: u64,
    pub claims: u64,
    pub frames: u64,
    pub original_bytes: u64,
    pub spool_bytes: u64,
    pub wire_bytes: u64,
    pub public_pass_bytes: u64,
    pub evaluator_q_bytes: u64,
    pub matrix_terms: u64,
    pub online_work: u64,
    pub derived_work: u64,
    pub accepted_sparse_entries: u64,
    pub noise_samples: u64,
    pub primary_challenges: u64,
    pub secondary_challenges: u64,
    pub original_scalar_checks: u64,
    pub secondary_scalar_checks: u64,
    pub code_instances: u64,
    pub code_description_bytes: u64,
    pub logical_io_floor: u64,
    pub active_payload_floor: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct Inventory<'a> {
    plan: [u8; 32],
    matrices: &'a [OriginalMatrix],
    summary: Summary,
}

impl<'a> Inventory<'a> {
    /// Checked summary, O(originals) memory borrowed; NEVER allocates tile records.
    /// `complete_originals` must come from the same original traversal as catalog.
    pub fn new(
        plan: [u8; 32],
        matrices: &'a [OriginalMatrix],
        complete_originals: u64,
    ) -> Result<Self> {
        if matrices.is_empty() || matrices.len() > MAX_RECORDS {
            return Err(Refused::Records);
        }
        if plan == [0; 32] || complete_originals != matrices.len() as u64 {
            return Err(Refused::Binding);
        }
        let mut s = Summary {
            originals: complete_originals,
            ..Summary::default()
        };
        s.code_instances = 5;
        s.code_description_bytes = 16 * REPEAT as u64 * (2 * COLUMN + ROW + COLUMN + 2 * COLUMN);
        for (i, matrix) in matrices.iter().enumerate() {
            if matrix.source == [0; 32]
                || matrix.operation == [0; 32]
                || matrices[..i]
                    .iter()
                    .any(|m| m.operation == matrix.operation)
            {
                return Err(Refused::Binding);
            }
            let columns = matrix.columns / COLUMN;
            s.unpartitioned_claims = add(
                s.unpartitioned_claims,
                if matrix.usage == Use::EightPositions {
                    8
                } else {
                    1
                },
            )?;
            if matrix.rows == 0
                || matrix.rows % COLUMN != 0
                || columns == 0
                || matrix.columns % COLUMN != 0
                || (matrix.usage == Use::FinalPosition && columns % 2 != 0)
            {
                return Err(Refused::Geometry);
            }
            for (rows, intervals) in [
                (ROW, matrix.rows / ROW),
                (COLUMN, (matrix.rows % ROW) / COLUMN),
            ] {
                let tiles = mul(intervals, columns)?;
                let groups = if matrix.usage == Use::EightPositions {
                    tiles
                } else {
                    tiles / 2
                };
                let batches = mul(
                    groups,
                    if matrix.usage == Use::EightPositions {
                        4
                    } else {
                        1
                    },
                )?;
                let claims = mul(batches, BATCH_SIZE as u64)?;
                let input_bytes = mul(claims, COLUMN * 4)?;
                let reply_bytes = mul(claims, rows * 4)?;
                let p = rows * COLUMN * REPEAT as u64 * BATCH_SIZE as u64 * 4;
                let q = p;
                let auxiliary =
                    rows * REPEAT as u64 * (BATCH_SIZE as u64 * COLUMN * REPEAT as u64) * 4;
                s.tiles = add(s.tiles, tiles)?;
                s.groups = add(s.groups, groups)?;
                s.batches = add(s.batches, batches)?;
                s.claims = add(s.claims, claims)?;
                s.original_bytes = add(s.original_bytes, add(input_bytes, reply_bytes)?)?;
                s.public_pass_bytes =
                    add(s.public_pass_bytes, mul(groups, sum(&[p, q, auxiliary])?)?)?;
                s.evaluator_q_bytes = add(s.evaluator_q_bytes, mul(groups, q)?)?;
                s.matrix_terms = add(s.matrix_terms, mul(claims, rows * COLUMN)?)?;
                let r = REPETITIONS as u64;
                let t = CHECK_WEIGHT as u64;
                let online_per_batch = sum(&[
                    2 * r * COLUMN * t,                 // Auxiliary projection products.
                    2 * rows * TX,                      // Privacy correction products.
                    2 * rows * REPEAT as u64 * 4,       // Encoded y: two mul/two add.
                    2 * COLUMN * REPEAT as u64 * 5,     // Privacy syndromes.
                    r * 2 * COLUMN * REPEAT as u64 * 5, // Secondary syndromes.
                    r * r * t * t,                      // Secondary bilinear products.
                    r * r * 2 * COLUMN,                 // Secondary right products.
                    r * 2 * COLUMN,                     // Original left products.
                    r * 2 * t,                          // Original sparse right products.
                ])?;
                s.online_work = add(s.online_work, mul(batches, online_per_batch)?)?;
                if groups != 0 {
                    let active = sum(&[
                        4 << 20,
                        p,
                        q,
                        2 * (2 * rows * COLUMN * 4),
                        2 * s.code_description_bytes,
                    ])?;
                    s.active_payload_floor = s.active_payload_floor.max(active);
                }
            }
        }
        s.frames = mul(s.batches, 6)?;
        s.spool_bytes = mul(
            s.batches,
            (REPETITIONS * CHECK_WEIGHT * 8 + REPETITIONS * 2 * COLUMN as usize * 4) as u64,
        )?;
        s.wire_bytes = add(s.original_bytes, s.spool_bytes)?;
        // Full construction + full validation, issuance AND final verification.
        s.derived_work = mul(s.public_pass_bytes, 4)?;
        // Both primary AND secondary sparse draws, plus privacy noise.
        s.accepted_sparse_entries = mul(s.claims, TX + 2 * CHECK_WEIGHT as u64)?;
        s.noise_samples = s.claims;
        s.primary_challenges = s.claims;
        s.secondary_challenges = s.claims;
        s.original_scalar_checks = mul(s.claims, REPETITIONS as u64)?;
        s.secondary_scalar_checks = s.original_scalar_checks;
        // Plus complete evaluator-Q acquisition/validation and one spool write/read.
        s.logical_io_floor = sum(&[
            s.derived_work,
            mul(s.evaluator_q_bytes, 2)?,
            mul(s.spool_bytes, 2)?,
        ])?;
        Ok(Self {
            plan,
            matrices,
            summary: s,
        })
    }

    pub fn summary(&self) -> Summary {
        self.summary
    }

    /// Derive one ordinal on demand, never a full claim catalog. Positions are
    /// original request positions, NOT token IDs or numerical activation values.
    pub fn claim_slice(&self, group: u64, batch: u64, slot: u64) -> Result<ClaimSlice> {
        let g = self.group(group)?;
        if slot >= BATCH_SIZE as u64
            || batch < g.first_batch
            || batch >= g.first_batch + g.batch_count
        {
            return Err(Refused::Order);
        }
        let m = self.matrices[g.original];
        let (column, position) = match m.usage {
            Use::EightPositions => (g.column_start, 2 * (batch - g.first_batch) + slot),
            Use::FinalPosition => (g.column_start + slot * COLUMN, 7),
        };
        Ok(ClaimSlice {
            operation: m.operation,
            source: m.source,
            row: g.row_start,
            column,
            rows: g.rows,
            columns: COLUMN,
            position,
        })
    }

    /// Canonical FIFO: original order, row interval, column interval, token pair.
    /// Head pairs adjacent columns; tail is an actual 128-row group, never padded.
    pub fn group(&self, ordinal: u64) -> Result<Group> {
        let mut left = ordinal;
        let mut first_batch = 0;
        for (original, matrix) in self.matrices.iter().enumerate() {
            let columns = matrix.columns / COLUMN;
            let per_row = if matrix.usage == Use::EightPositions {
                columns
            } else {
                columns / 2
            };
            let count = mul(matrix.rows.div_ceil(ROW), per_row)?;
            let batch_count = if matrix.usage == Use::EightPositions {
                4
            } else {
                1
            };
            if left < count {
                let row_start = (left / per_row) * ROW;
                let rows = ROW.min(matrix.rows - row_start);
                let column_start = (left % per_row) * COLUMN * if batch_count == 4 { 1 } else { 2 };
                let p = rows * COLUMN * REPEAT as u64 * 2 * 4;
                let auxiliary = rows * REPEAT as u64 * (2 * COLUMN * REPEAT as u64) * 4;
                return Ok(Group {
                    original,
                    row_start,
                    column_start,
                    rows,
                    first_batch: add(first_batch, mul(left, batch_count)?)?,
                    batch_count,
                    public_bytes: sum(&[p, p, auxiliary])?,
                    q_bytes: p,
                });
            }
            left -= count;
            first_batch = add(first_batch, mul(count, batch_count)?)?;
        }
        Err(Refused::Order)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Group {
    pub original: usize,
    pub row_start: u64,
    pub column_start: u64,
    pub rows: u64,
    pub first_batch: u64,
    pub batch_count: u64,
    pub public_bytes: u64,
    pub q_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaimSlice {
    pub operation: [u8; 32],
    pub source: [u8; 32],
    pub row: u64,
    pub column: u64,
    pub rows: u64,
    pub columns: u64,
    pub position: u64,
}

/// Supplied attachment references are not validators/certificates. Unknowns
/// refuse even METADATA reservation; execution always needs actual sealed owners.
#[derive(Debug, Clone, Copy)]
pub struct Attachments {
    pub security: Option<[u8; 32]>,
    pub backend: Option<[u8; 32]>,
    pub source_binding: Option<[u8; 32]>,
    pub lifetime_binding: Option<[u8; 32]>,
    pub client_other_bytes: Option<u64>,
    pub active_other_bytes: Option<u64>,
    pub source_backing_bytes: Option<u64>,
    pub store_other_bytes: Option<u64>,
    /// Source, code/distance, local numeric/scalar, entropy rejection, transport,
    /// hashing/copying/product reductions, cleanup work; EVERY term requires a
    /// finite complete quote. These are caller-supplied conditional metadata,
    /// not independently validated runtime bounds or proof certificates.
    pub additional_work: [Option<u64>; 7],
    pub additional_io: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub client_bytes: u64,
    pub store_bytes: u64,
    pub active_bytes: u64,
    pub work: u64,
    pub io: u64,
}

/// Immutable arithmetic reservation, never a live memory/protocol owner.
#[derive(Debug)]
pub struct MetadataReservation<'a> {
    inventory: Inventory<'a>,
    client_bytes: u64,
    store_bytes: u64,
    active_bytes: u64,
    work: u64,
    io: u64,
    attachments: Attachments,
}

impl<'a> MetadataReservation<'a> {
    pub fn admit(inventory: Inventory<'a>, a: Attachments, limits: Limits) -> Result<Self> {
        for id in [a.security, a.backend, a.source_binding, a.lifetime_binding] {
            if id.is_none_or(|v| v == [0; 32]) {
                return Err(Refused::UnknownAttachment);
            }
        }
        let known = |v: Option<u64>| v.ok_or(Refused::UnknownAttachment);
        let s = inventory.summary;
        let active_bytes = add(s.active_payload_floor, known(a.active_other_bytes)?)?;
        let client_bytes = sum(&[s.original_bytes, active_bytes, known(a.client_other_bytes)?])?;
        let store_bytes = sum(&[
            s.spool_bytes,
            known(a.source_backing_bytes)?,
            known(a.store_other_bytes)?,
        ])?;
        let mut work = sum(&[
            s.derived_work,
            mul(s.evaluator_q_bytes, 2)?,
            s.matrix_terms,
            s.online_work,
        ])?;
        for term in a.additional_work {
            work = add(work, known(term)?)?;
        }
        let io = add(s.logical_io_floor, known(a.additional_io)?)?;
        if limits.client_bytes > CLIENT_CAP
            || limits.store_bytes > STORE_CAP
            || limits.active_bytes > ALLOCATION_CEILING as u64
            || client_bytes > limits.client_bytes
            || store_bytes > limits.store_bytes
            || active_bytes > limits.active_bytes
        {
            return Err(Refused::Capacity);
        }
        if work > limits.work || io > limits.io {
            return Err(Refused::Work);
        }
        Ok(Self {
            inventory,
            client_bytes,
            store_bytes,
            active_bytes,
            work,
            io,
            attachments: a,
        })
    }

    pub fn execution_admission(&self) -> Result<()> {
        Err(Refused::OwnerBridgeMissing)
    }

    pub fn reserved(&self) -> Limits {
        Limits {
            client_bytes: self.client_bytes,
            store_bytes: self.store_bytes,
            active_bytes: self.active_bytes,
            work: self.work,
            io: self.io,
        }
    }

    pub fn supplied_bindings(&self) -> ([u8; 32], Attachments) {
        (self.inventory.plan, self.attachments)
    }

    pub fn cursor(&self) -> Cursor<'a> {
        Cursor {
            inventory: self.inventory,
            phase: Phase::Original,
            group: 0,
            batch: 0,
            window: None,
            read: false,
            validated: false,
            primary: false,
            original_bytes: 0,
            spool_bytes: 0,
            work: 0,
            io: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Original,
    Primary,
    Check,
    Erase,
    Release,
    Spent,
}

#[derive(Debug, Clone, Copy)]
pub enum Event {
    Cancel,
    Open(u64),
    Read(u64),
    Validate(u64),
    Original(u64),
    Primary(u64),
    Auxiliary(u64),
    Check(u64),
    Close,
    OriginalBarrier,
    AuxiliaryBarrier,
    Finish,
    Erase,
    Release,
}

/// Consuming METADATA simulator. No Clone and no private/accepted values.
/// Reconstructing a cursor does not reset any real request: there is no bridge.
/// Dropping a window never resets batch ordinals or the one-attempt horizon.
#[derive(Debug)]
pub struct Cursor<'a> {
    inventory: Inventory<'a>,
    phase: Phase,
    group: u64,
    batch: u64,
    window: Option<Group>,
    read: bool,
    validated: bool,
    primary: bool,
    original_bytes: u64,
    spool_bytes: u64,
    work: u64,
    io: u64,
}

impl Cursor<'_> {
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn retained_bytes(&self) -> (u64, u64) {
        (self.original_bytes, self.spool_bytes)
    }
    pub fn cumulative_visits(&self) -> (u64, u64) {
        (self.work, self.io)
    }
    pub fn horizon_attempts(&self) -> u64 {
        1
    }

    /// Failure consumes the simulator, just as terminal cancellation. No retry
    /// or early acceptance object is returned; these are NOT custody guarantees.
    pub fn advance(mut self, event: Event) -> Result<Self> {
        match event {
            Event::Cancel
                if !matches!(self.phase, Phase::Spent | Phase::Erase | Phase::Release) =>
            {
                // Model cancellation retains backing/accounting until ordered
                // erase then release; there is no new attempt or accepted value.
                self.window = None;
                self.phase = Phase::Erase;
            }
            Event::Open(ordinal)
                if self.window.is_none()
                    && ordinal == self.group
                    && matches!(self.phase, Phase::Original | Phase::Primary | Phase::Check) =>
            {
                let g = self.inventory.group(ordinal)?;
                if self.batch != g.first_batch {
                    return Err(Refused::Order);
                }
                self.window = Some(g);
                self.read = false;
                self.validated = false;
            }
            Event::Read(bytes) if self.window.is_some() && !self.read => {
                let g = self.window.ok_or(Refused::Order)?;
                let expected = if self.phase == Phase::Primary {
                    g.q_bytes
                } else {
                    g.public_bytes
                };
                if bytes != expected {
                    return Err(Refused::Order);
                }
                self.io = add(self.io, bytes)?;
                self.work = add(self.work, bytes)?;
                self.read = true;
            }
            Event::Validate(bytes) if self.read && !self.validated => {
                let g = self.window.ok_or(Refused::Order)?;
                let expected = if self.phase == Phase::Primary {
                    g.q_bytes
                } else {
                    g.public_bytes
                };
                if bytes != expected {
                    return Err(Refused::Order);
                }
                self.io = add(self.io, bytes)?;
                self.work = add(self.work, bytes)?;
                self.validated = true;
            }
            Event::Original(batch)
                if self.phase == Phase::Original && self.validated && batch == self.batch =>
            {
                let g = self.window.ok_or(Refused::Order)?;
                if batch >= g.first_batch + g.batch_count {
                    return Err(Refused::Order);
                }
                self.original_bytes = add(self.original_bytes, 2 * (COLUMN + g.rows) * 4)?;
                self.batch = add(self.batch, 1)?;
            }
            Event::Primary(batch)
                if self.phase == Phase::Primary
                    && self.validated
                    && !self.primary
                    && batch == self.batch =>
            {
                let g = self.window.ok_or(Refused::Order)?;
                if batch >= g.first_batch + g.batch_count {
                    return Err(Refused::Order);
                }
                self.primary = true;
            }
            Event::Auxiliary(batch)
                if self.phase == Phase::Primary && self.primary && batch == self.batch =>
            {
                let per = self.inventory.summary.spool_bytes / self.inventory.summary.batches;
                self.spool_bytes = add(self.spool_bytes, per)?;
                self.batch = add(self.batch, 1)?;
                self.primary = false;
            }
            Event::Check(batch)
                if self.phase == Phase::Check && self.validated && batch == self.batch =>
            {
                let g = self.window.ok_or(Refused::Order)?;
                if batch >= g.first_batch + g.batch_count {
                    return Err(Refused::Order);
                }
                self.batch = add(self.batch, 1)?;
            }
            Event::Close if self.window.is_some() && !self.primary => {
                let g = self.window.ok_or(Refused::Order)?;
                if self.batch != g.first_batch + g.batch_count || !self.validated {
                    return Err(Refused::Order);
                }
                self.window = None;
                self.group = add(self.group, 1)?;
                self.read = false;
                self.validated = false;
            }
            Event::OriginalBarrier
                if self.phase == Phase::Original
                    && self.window.is_none()
                    && self.batch == self.inventory.summary.batches
                    && self.group == self.inventory.summary.groups =>
            {
                self.phase = Phase::Primary;
                self.group = 0;
                self.batch = 0;
            }
            Event::AuxiliaryBarrier
                if self.phase == Phase::Primary
                    && self.window.is_none()
                    && self.batch == self.inventory.summary.batches
                    && self.group == self.inventory.summary.groups =>
            {
                self.phase = Phase::Check;
                self.group = 0;
                self.batch = 0;
            }
            Event::Finish
                if self.phase == Phase::Check
                    && self.window.is_none()
                    && self.batch == self.inventory.summary.batches
                    && self.group == self.inventory.summary.groups =>
            {
                self.phase = Phase::Erase;
            }
            Event::Erase if self.phase == Phase::Erase => {
                self.phase = Phase::Release;
            }
            Event::Release if self.phase == Phase::Release => {
                self.original_bytes = 0;
                self.spool_bytes = 0;
                self.phase = Phase::Spent;
            }
            _ => return Err(Refused::Order),
        }
        Ok(self)
    }
}

#[cfg(test)]
#[path = "window_plan_tests.rs"]
mod tests;
