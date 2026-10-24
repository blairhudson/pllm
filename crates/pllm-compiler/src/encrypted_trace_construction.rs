//! Borrowed immutable preconstruction evidence for the shared SSA validator.
//! Quotes never grant allocation, native execution, or metadata work. Pre-existing
//! upstream owners must already cover manifest acquisition and original capacity.
//! Unknown allocator/copy bounds require refusal; no allocating factory lives here.

use super::*;
use pllm_core::row_fold::WorkLedger;
use std::mem::{align_of, size_of};

pub const METADATA_BYTES: usize = 8 << 20;
pub const OWNER_BYTES: usize = 64 << 20;
pub const TRACE_WORK_CEILING: u64 = 30_068_627_962_880;
pub const VALIDATOR_BYTES_CEILING: usize = 1_991_168;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    UnknownOriginalCapacity,
    IncompleteConstructionManifest,
    ArithmeticOverflow,
    ShortStorage,
    MetadataWorkRefused,
    MetadataStorageRefused,
    ControlWorkExhausted,
    MissingAmbientTraceAdmission,
    LifetimeMismatch,
    Poisoned,
    InvalidTrace,
}
fn plus(a: usize, b: usize) -> Result<usize, Refusal> {
    a.checked_add(b).ok_or(Refusal::ArithmeticOverflow)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Execution {
    Ckks,
    Hybrid,
    View,
}
#[derive(Clone, Copy, Debug)]
pub struct OperationCount {
    pub inputs: usize,
    pub output_rank: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct CapabilityCount<'a> {
    pub input_ranks: &'a [usize],
    pub output_rank: usize,
    pub execution: Execution,
}

/// Element type fixes actual target header/enum layout, not a caller byte guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    Byte,
    String,
    Shape,
    Owner,
    Value,
    Operation,
    Capability,
    Step,
    State,
}
impl Element {
    fn layout(self) -> (usize, usize) {
        match self {
            Self::Byte => (1, 1),
            Self::String => (size_of::<String>(), align_of::<String>()),
            Self::Shape => (size_of::<u64>(), align_of::<u64>()),
            Self::Owner => (
                size_of::<PublicCiphertextOwner>(),
                align_of::<PublicCiphertextOwner>(),
            ),
            Self::Value => (
                size_of::<PublicCiphertextValue>(),
                align_of::<PublicCiphertextValue>(),
            ),
            Self::Operation => (
                size_of::<PublicCiphertextGraphOperation>(),
                align_of::<PublicCiphertextGraphOperation>(),
            ),
            Self::Capability => (
                size_of::<PublicCiphertextCapability>(),
                align_of::<PublicCiphertextCapability>(),
            ),
            Self::Step => (
                size_of::<PublicCiphertextTraceStep>(),
                align_of::<PublicCiphertextTraceStep>(),
            ),
            Self::State => (
                size_of::<PublicCiphertextState>(),
                align_of::<PublicCiphertextState>(),
            ),
        }
    }
}
/// One original backing. Inline headers are part of their containing allocation.
/// Capacity is original, never merely a slice length. Bounded half-open phases.
#[derive(Clone, Copy, Debug)]
pub struct Backing {
    pub element: Element,
    pub length: usize,
    pub capacity: Option<usize>,
    pub first_phase: usize,
    pub end_phase: usize,
}
/// Other independently admitted original owners, including source/staging/copy,
/// input, public vectors, physical pool, oracle, diagnostics and quote scratch.
/// Metadata includes manifest original storage. Native owners are out of scope.
#[derive(Clone, Copy, Debug)]
pub struct OtherOwner {
    pub bytes: Option<usize>,
    pub metadata: bool,
    pub first_phase: usize,
    pub end_phase: usize,
}

pub struct PublicCiphertextConstructionManifest<'a> {
    pub owners: usize,
    pub persistent_owners: usize,
    pub input_ranks: &'a [usize],
    pub operations: &'a [OperationCount],
    pub capabilities: &'a [CapabilityCount<'a>],
    pub outputs: usize,
    pub retained_inputs: usize,
    pub release_lengths: &'a [usize],
    /// Ordered allocation inventory: graph cost strings, outer arrays, then nested
    /// owners/persistent IDs/inputs/operations/capabilities/outputs/retained/steps.
    pub backings: &'a [Backing],
    pub other_owners: &'a [OtherOwner],
    pub phases: usize,
    /// Original admitted metadata region containing this manifest and all its
    /// borrowed descriptions. This is evidence, not retroactive acquisition.
    pub manifest_original_bytes: Option<usize>,
}

#[derive(Debug)]
pub struct ControlLedger<'a> {
    work: &'a mut WorkLedger,
}
impl<'a> ControlLedger<'a> {
    /// Compiler control and core/oracle work debit the same original cumulative
    /// ledger. No separate allowance can overlap or be refunded into clear work.
    pub fn new(work: &'a mut WorkLedger) -> Self {
        Self { work }
    }
    fn debit(&mut self, n: usize) -> Result<(), Refusal> {
        self.work
            .debit(0, n)
            .map_err(|_| Refusal::ControlWorkExhausted)
    }
    pub(crate) fn core_work(&mut self) -> &mut WorkLedger {
        self.work
    }
}
#[derive(Debug)]
pub struct MetadataLedger {
    remaining: usize,
}
impl MetadataLedger {
    /// Independently authorized cumulative ambient allowance, NOT a requirement.
    pub fn new(ambient: usize) -> Self {
        Self {
            remaining: ambient.min(usize::try_from(TRACE_WORK_CEILING).unwrap_or(0)),
        }
    }
    pub fn remaining(&self) -> usize {
        self.remaining
    }
}

/// Computed evidence is immutable to callers; never an ambient grant.
///
/// ```compile_fail
/// use pllm_compiler::encrypted_level_schedule::construction::PublicCiphertextConstructionQuote;
/// fn forge(q: &mut PublicCiphertextConstructionQuote<'_>) { q.trace_work = 0; }
/// ```
pub struct PublicCiphertextConstructionQuote<'a> {
    manifest: &'a PublicCiphertextConstructionManifest<'a>,
    records: usize,
    trace_work: usize,
    validator_payload_bytes: usize,
    metadata_peak: usize,
    owner_peak: usize,
}

/// One already-debited validation, not new allocation/native authority. Opaque,
/// non-Clone, consumed on use; dropping on construction failure never refunds.
/// Its exclusive original-ledger borrow lasts through validation/diagnostics.
///
/// ```compile_fail
/// use pllm_compiler::encrypted_level_schedule::construction::TraceAdmission;
/// fn replay(token: TraceAdmission<'_, '_, '_>) { let first = token; let second = token; }
/// ```
pub struct TraceAdmission<'q, 'm, 'l> {
    quote: &'q PublicCiphertextConstructionQuote<'m>,
    _ledger: &'l mut MetadataLedger,
    capacity: PublicCiphertextTraceCapacity,
}

// Enumerate exactly the allocations demanded by the structural manifest. The
// same sequence is checked against actual graph capacities before validation.
fn inventory(
    m: &PublicCiphertextConstructionManifest<'_>,
    mut visit: impl FnMut(Element, Option<usize>, usize) -> Result<(), Refusal>,
) -> Result<(), Refusal> {
    fn state(
        v: &mut impl FnMut(Element, Option<usize>, usize) -> Result<(), Refusal>,
        rank: usize,
    ) -> Result<(), Refusal> {
        if !(1..=2).contains(&rank) {
            return Err(Refusal::IncompleteConstructionManifest);
        }
        for _ in 0..5 {
            v(Element::Byte, None, 64)?;
        }
        v(Element::Shape, Some(rank), 2)
    }
    fn value(
        v: &mut impl FnMut(Element, Option<usize>, usize) -> Result<(), Refusal>,
        rank: usize,
    ) -> Result<(), Refusal> {
        v(Element::Byte, None, 64)?;
        v(Element::Byte, None, 64)?;
        state(v, rank)
    }
    visit(Element::Byte, None, 64)?;
    visit(Element::Byte, None, 64)?;
    for (e, n, cap) in [
        (Element::Owner, m.owners, 149),
        (Element::String, m.persistent_owners, 22),
        (Element::Value, m.input_ranks.len(), 1),
        (Element::Operation, m.operations.len(), 128),
        (Element::Capability, m.capabilities.len(), 128),
        (Element::String, m.outputs, 1),
        (Element::String, m.retained_inputs, 1),
        (Element::Step, m.release_lengths.len(), 128),
    ] {
        visit(e, Some(n), cap)?;
    }
    for _ in 0..m.owners + m.persistent_owners {
        visit(Element::Byte, None, 64)?;
    }
    for &rank in m.input_ranks {
        value(&mut visit, rank)?;
    }
    for op in m.operations {
        visit(Element::Byte, None, 64)?;
        visit(Element::Byte, None, 64)?;
        visit(Element::String, Some(op.inputs), 2)?;
        for _ in 0..op.inputs {
            visit(Element::Byte, None, 64)?;
        }
        value(&mut visit, op.output_rank)?;
    }
    for cap in m.capabilities {
        visit(Element::Byte, None, 64)?;
        visit(Element::Byte, None, 64)?;
        visit(Element::State, Some(cap.input_ranks.len()), 2)?;
        for &rank in cap.input_ranks {
            state(&mut visit, rank)?;
        }
        state(&mut visit, cap.output_rank)?;
        visit(Element::Byte, None, 64)?; // Required error assumption.
        let strings = match cap.execution {
            Execution::Ckks => 2,
            Execution::Hybrid => 4,
            Execution::View => 1,
        };
        for _ in 0..strings {
            visit(Element::Byte, None, 64)?;
        }
    }
    for _ in 0..m.outputs + m.retained_inputs {
        visit(Element::Byte, None, 64)?;
    }
    for &n in m.release_lengths {
        visit(Element::Byte, None, 64)?;
        visit(Element::Byte, None, 64)?;
        visit(Element::String, Some(n), 2)?;
        for _ in 0..n {
            visit(Element::Byte, None, 64)?;
        }
    }
    Ok(())
}

pub fn quote_public_ciphertext_construction<'a>(
    m: &'a PublicCiphertextConstructionManifest<'a>,
    control: &mut ControlLedger,
) -> Result<PublicCiphertextConstructionQuote<'a>, Refusal> {
    // Bound every traversal before entering it. Prepay quote + actual-capacity
    // checking; no refund on malformed manifests or repeated attempts.
    control.debit(256)?;
    if m.owners > 149
        || m.persistent_owners > 22
        || m.input_ranks.len() > 1
        || m.operations.len() > 128
        || m.capabilities.len() > 128
        || m.outputs > 1
        || m.retained_inputs > 1
        || m.release_lengths.len() > 128
        || m.backings.len() > 8192
        || m.other_owners.len() > 8192
        || m.backings.len() + m.other_owners.len() > 8192
        || m.phases == 0
        || m.phases > 256
    {
        return Err(Refusal::IncompleteConstructionManifest);
    }
    control.debit(plus(
        32768,
        plus(m.backings.len() * 8, m.other_owners.len() * 8)?,
    )?)?;
    let mut counter = TraceRecordCounter::new();
    for n in [
        m.owners,
        m.persistent_owners,
        m.input_ranks.len(),
        m.operations.len(),
        m.capabilities.len(),
        m.outputs,
        m.retained_inputs,
        m.release_lengths.len(),
    ] {
        counter.top(n).ok_or(Refusal::ArithmeticOverflow)?;
    }
    for &rank in m.input_ranks {
        counter.state(rank).ok_or(Refusal::ArithmeticOverflow)?;
    }
    for op in m.operations {
        if !(1..=2).contains(&op.inputs) {
            return Err(Refusal::IncompleteConstructionManifest);
        }
        counter
            .operation(op.inputs)
            .ok_or(Refusal::ArithmeticOverflow)?;
        counter
            .state(op.output_rank)
            .ok_or(Refusal::ArithmeticOverflow)?;
    }
    for cap in m.capabilities {
        if !(1..=2).contains(&cap.input_ranks.len()) {
            return Err(Refusal::IncompleteConstructionManifest);
        }
        counter
            .capability(cap.input_ranks.len())
            .ok_or(Refusal::ArithmeticOverflow)?;
        counter
            .state(cap.output_rank)
            .ok_or(Refusal::ArithmeticOverflow)?;
        for &rank in cap.input_ranks {
            counter.state(rank).ok_or(Refusal::ArithmeticOverflow)?;
        }
    }
    let mut releases = 0;
    for &n in m.release_lengths {
        if n > 2 {
            return Err(Refusal::IncompleteConstructionManifest);
        }
        releases = plus(releases, n)?;
        counter.step(n).ok_or(Refusal::ArithmeticOverflow)?;
    }
    if releases > 125 {
        return Err(Refusal::IncompleteConstructionManifest);
    }
    let records = counter.records;
    if records > 7774 {
        return Err(Refusal::MetadataWorkRefused);
    }
    let (trace_work, validator_payload_bytes) =
        trace_resource_bound(records).ok_or(Refusal::ArithmeticOverflow)?;
    if trace_work as u64 > TRACE_WORK_CEILING || validator_payload_bytes > VALIDATOR_BYTES_CEILING {
        return Err(Refusal::MetadataWorkRefused);
    }
    let mut index = 0;
    inventory(m, |element, length, cap| {
        let b = m
            .backings
            .get(index)
            .ok_or(Refusal::IncompleteConstructionManifest)?;
        index += 1;
        let capacity = b.capacity.ok_or(Refusal::UnknownOriginalCapacity)?;
        if b.element != element
            || length.is_some_and(|n| n != b.length)
            || b.length > capacity
            || capacity > cap
            || (element == Element::Byte && b.length == 0)
            || b.end_phase != m.phases
        {
            return Err(Refusal::IncompleteConstructionManifest);
        }
        Ok(())
    })?;
    if index != m.backings.len() {
        return Err(Refusal::IncompleteConstructionManifest);
    }
    let original = m
        .manifest_original_bytes
        .ok_or(Refusal::UnknownOriginalCapacity)?;
    let minimum = size_of::<PublicCiphertextConstructionManifest<'_>>()
        + size_of_val(m.backings)
        + size_of_val(m.other_owners)
        + size_of_val(m.input_ranks)
        + size_of_val(m.operations)
        + size_of_val(m.capabilities)
        + size_of_val(m.release_lengths)
        + m.capabilities
            .iter()
            .map(|c| size_of_val(c.input_ranks))
            .sum::<usize>();
    if original < minimum || original > METADATA_BYTES {
        return Err(Refusal::MetadataStorageRefused);
    }
    // Four event arrays, not O(phases*backings) scans. Actual quote scratch and
    // inline graph/trace/quote control are charged even before graph exists.
    let scratch = 4 * size_of::<[usize; 257]>()
        + size_of::<PublicCiphertextGraph>()
        + size_of::<Vec<PublicCiphertextTraceStep>>()
        + size_of::<PublicCiphertextConstructionQuote<'_>>();
    if scratch > 65536 {
        return Err(Refusal::MetadataStorageRefused);
    }
    let base = plus(plus(original, scratch)?, validator_payload_bytes)?;
    let mut starts = [0usize; 257];
    let mut ends = [0usize; 257];
    let mut meta_starts = [0usize; 257];
    let mut meta_ends = [0usize; 257];
    let mut event =
        |bytes: usize, metadata: bool, first: usize, end: usize| -> Result<(), Refusal> {
            if first >= end || end > m.phases {
                return Err(Refusal::LifetimeMismatch);
            }
            starts[first] = plus(starts[first], bytes)?;
            ends[end] = plus(ends[end], bytes)?;
            if metadata {
                meta_starts[first] = plus(meta_starts[first], bytes)?;
                meta_ends[end] = plus(meta_ends[end], bytes)?;
            }
            Ok(())
        };
    for b in m.backings {
        let (size, _align) = b.element.layout();
        let bytes = b
            .capacity
            .ok_or(Refusal::UnknownOriginalCapacity)?
            .checked_mul(size)
            .ok_or(Refusal::ArithmeticOverflow)?;
        event(bytes, true, b.first_phase, b.end_phase)?;
    }
    for o in m.other_owners {
        event(
            o.bytes.ok_or(Refusal::UnknownOriginalCapacity)?,
            o.metadata,
            o.first_phase,
            o.end_phase,
        )?;
    }
    let (mut live, mut meta, mut owner_peak, mut metadata_peak) = (base, base, base, base);
    for phase in 0..m.phases {
        live = live
            .checked_sub(ends[phase])
            .ok_or(Refusal::LifetimeMismatch)?;
        meta = meta
            .checked_sub(meta_ends[phase])
            .ok_or(Refusal::LifetimeMismatch)?;
        live = plus(live, starts[phase])?;
        meta = plus(meta, meta_starts[phase])?;
        owner_peak = owner_peak.max(live);
        metadata_peak = metadata_peak.max(meta);
    }
    if owner_peak > OWNER_BYTES || metadata_peak > METADATA_BYTES {
        return Err(Refusal::MetadataStorageRefused);
    }
    Ok(PublicCiphertextConstructionQuote {
        manifest: m,
        records,
        trace_work,
        validator_payload_bytes,
        metadata_peak,
        owner_peak,
    })
}

impl<'m> PublicCiphertextConstructionQuote<'m> {
    pub fn records(&self) -> usize {
        self.records
    }
    pub fn trace_work(&self) -> usize {
        self.trace_work
    }
    pub fn validator_payload_bytes(&self) -> usize {
        self.validator_payload_bytes
    }
    pub fn metadata_peak(&self) -> usize {
        self.metadata_peak
    }
    pub fn owner_peak(&self) -> usize {
        self.owner_peak
    }
    /// Validation-only entry for already constructed, upstream-admitted metadata.
    /// Complete validator bound is debited BEFORE either capacity verification or
    /// validation. Failure does not refund. Ordinary String diagnostics are
    /// discarded within the prepaid metadata envelope; no private values escape.
    #[allow(clippy::ptr_arg)] // Preserve original trace allocation capacity.
    pub fn validate_constructed(
        &self,
        admission: TraceAdmission<'_, '_, '_>,
        g: &PublicCiphertextGraph,
        trace: &Vec<PublicCiphertextTraceStep>,
        control: &mut ControlLedger,
    ) -> Result<PublicCiphertextTraceSummary, Refusal> {
        if !std::ptr::eq(self, admission.quote) {
            return Err(Refusal::LifetimeMismatch);
        }
        self.check_constructed(g, trace, control)?;
        validate_public_ciphertext_trace(g, trace, admission.capacity)
            .map_err(|_| Refusal::InvalidTrace)
    }
    /// Pre-factory gate. Capacity/ledger are independently supplied finite ambient
    /// grants; requirement/local ceiling is never used to construct either grant.
    /// This does not authorize an allocator. Existing owner guards must additionally
    /// cover all simultaneous original allocations through payload cleanup.
    pub fn admit_trace<'q, 'l>(
        &'q self,
        ambient: Option<PublicCiphertextTraceCapacity>,
        ledger: &'l mut MetadataLedger,
    ) -> Result<TraceAdmission<'q, 'm, 'l>, Refusal> {
        let c = ambient.ok_or(Refusal::MissingAmbientTraceAdmission)?;
        if self.trace_work > c.max_work
            || self.trace_work > ledger.remaining
            || self.records > c.max_records
        {
            return Err(Refusal::MetadataWorkRefused);
        }
        if self.validator_payload_bytes > c.max_buffer_bytes
            || self.owner_peak as u64 > c.max_live_owner_bytes
        {
            return Err(Refusal::MetadataStorageRefused);
        }
        ledger.remaining -= self.trace_work;
        Ok(TraceAdmission {
            quote: self,
            _ledger: ledger,
            capacity: c,
        })
    }

    /// Check original capacities/lengths against immutable manifest, then use the
    /// ordinary shared SSA validator. No quote can skip its consistency checks.
    #[allow(clippy::ptr_arg)] // Original Vec capacity, deliberately not a slice-size quote.
    pub fn check_constructed(
        &self,
        g: &PublicCiphertextGraph,
        trace: &Vec<PublicCiphertextTraceStep>,
        control: &mut ControlLedger,
    ) -> Result<(), Refusal> {
        let m = self.manifest;
        control.debit(1024 + 4 * m.backings.len())?;
        if [
            g.owners.len(),
            g.persistent_owners.len(),
            g.inputs.len(),
            g.operations.len(),
            g.capabilities.len(),
            g.outputs.len(),
            g.retained_inputs.len(),
            trace.len(),
        ] != [
            m.owners,
            m.persistent_owners,
            m.input_ranks.len(),
            m.operations.len(),
            m.capabilities.len(),
            m.outputs,
            m.retained_inputs,
            m.release_lengths.len(),
        ] {
            return Err(Refusal::IncompleteConstructionManifest);
        }
        let mut index = 0;
        let mut actual =
            |element: Element, length: usize, capacity: usize| -> Result<(), Refusal> {
                let b = m
                    .backings
                    .get(index)
                    .ok_or(Refusal::IncompleteConstructionManifest)?;
                index += 1;
                if b.element != element || b.length != length || b.capacity != Some(capacity) {
                    return Err(Refusal::LifetimeMismatch);
                }
                Ok(())
            };
        macro_rules! string {
            ($s:expr) => {
                actual(Element::Byte, $s.len(), $s.capacity())?
            };
        }
        macro_rules! vector {
            ($v:expr,$e:expr) => {
                actual($e, $v.len(), $v.capacity())?
            };
        }
        fn state(
            s: &PublicCiphertextState,
            a: &mut impl FnMut(Element, usize, usize) -> Result<(), Refusal>,
        ) -> Result<(), Refusal> {
            for id in [
                &s.scheme,
                &s.key_domain,
                &s.layout,
                &s.scale,
                &s.error.model_identity,
            ] {
                a(Element::Byte, id.len(), id.capacity())?;
            }
            a(Element::Shape, s.shape.len(), s.shape.capacity())
        }
        fn value(
            v: &PublicCiphertextValue,
            a: &mut impl FnMut(Element, usize, usize) -> Result<(), Refusal>,
        ) -> Result<(), Refusal> {
            a(Element::Byte, v.identity.len(), v.identity.capacity())?;
            a(Element::Byte, v.owner.len(), v.owner.capacity())?;
            state(&v.state, a)
        }
        string!(g.cost_model_identity);
        string!(g.cost_unit);
        vector!(g.owners, Element::Owner);
        vector!(g.persistent_owners, Element::String);
        vector!(g.inputs, Element::Value);
        vector!(g.operations, Element::Operation);
        vector!(g.capabilities, Element::Capability);
        vector!(g.outputs, Element::String);
        vector!(g.retained_inputs, Element::String);
        actual(Element::Step, trace.len(), trace.capacity())?;
        for o in &g.owners {
            string!(o.identity);
        }
        for id in &g.persistent_owners {
            string!(id);
        }
        for v in &g.inputs {
            value(v, &mut actual)?;
        }
        for op in &g.operations {
            string!(op.identity);
            string!(op.kernel_identity);
            vector!(op.inputs, Element::String);
            for id in &op.inputs {
                string!(id);
            }
            value(&op.output, &mut actual)?;
        }
        for (i, cap) in g.capabilities.iter().enumerate() {
            let execution = match cap.execution {
                PublicCiphertextExecution::Ckks { .. } => Execution::Ckks,
                PublicCiphertextExecution::Hybrid { .. } => Execution::Hybrid,
                PublicCiphertextExecution::View { .. } => Execution::View,
            };
            if execution != m.capabilities[i].execution {
                return Err(Refusal::LifetimeMismatch);
            }
            string!(cap.identity);
            string!(cap.kernel_identity);
            vector!(cap.inputs, Element::State);
            for s in &cap.inputs {
                state(s, &mut actual)?;
            }
            state(&cap.output, &mut actual)?;
            let id = cap
                .error_assumption
                .as_ref()
                .ok_or(Refusal::IncompleteConstructionManifest)?;
            string!(id);
            match &cap.execution {
                PublicCiphertextExecution::Ckks {
                    compute,
                    transition,
                    ..
                } => {
                    string!(compute.cost.identity);
                    string!(transition.cost.identity);
                }
                PublicCiphertextExecution::Hybrid { round_trip: h } => {
                    string!(h.identity);
                    string!(h.entry.identity);
                    string!(h.evaluate.identity);
                    string!(h.recover.identity);
                }
                PublicCiphertextExecution::View { cost } => {
                    string!(cost.identity);
                }
            }
        }
        for id in g.outputs.iter().chain(&g.retained_inputs) {
            string!(id);
        }
        for step in trace {
            string!(step.operation);
            string!(step.capability);
            vector!(step.release, Element::String);
            for id in &step.release {
                string!(id);
            }
        }
        if index != m.backings.len() {
            return Err(Refusal::IncompleteConstructionManifest);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "encrypted_trace_construction_tests.rs"]
mod tests;
