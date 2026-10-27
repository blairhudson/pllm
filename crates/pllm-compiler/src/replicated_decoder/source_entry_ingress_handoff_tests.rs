//! Source-only custody tests. These do not claim checkpoint scanner provenance
//! or exercise a projection/numeric backend.
use super::*;
use std::cell::Cell;
use std::rc::Rc;

fn begin() -> SourceHeaderEntry {
    let config = br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}"#;
    let experiment = serde_json::to_vec(&serde_json::json!({
        "schema": "pllm.experiment.v1", "name": "private-collector-handoff",
        "pipeline": {"model": {"source": "original-tiny-source"}, "components": {
            "linear": {"component": crate::replicated_decoder::sdk::PROTOCOL,
                "params": {"weights": "public", "range_policy": "checked",
                    "max_abs_activation": 1, "storage": "compact"}}}},
        "deployment": {"kind": "local", "root": "."},
        "budget": {"requests": 1, "max_input_tokens": 4, "max_new_tokens": 2}
    })).unwrap();
    SourceHeaderEntry::begin(&experiment, config, 4, 2, [1; 32], [2; 32], [3; 32]).unwrap()
}

#[test]
fn handoff_moves_original_plan_header_payload_and_covering_lease() {
    let header = begin();
    let counter = header.root.try_clone().unwrap();
    let plan = header.original.plan() as *const _;
    let directory = Arc::as_ptr(header.directory());
    let source = header.manifest.source;
    let configuration = header.manifest.config.as_ptr();
    // This is a heap-allocation identity check, not the address of a moved Vec
    // header and not a checkpoint provenance or runtime authority assertion.
    struct HeapPayload {
        bytes: Vec<u8>,
        owner: Rc<()>,
    }
    let mut bytes = Vec::with_capacity(29);
    bytes.extend_from_slice(&[7u8; 13]);
    let backing = (bytes.as_ptr(), bytes.len(), bytes.capacity());
    let owner = Rc::new(());
    let unique_owner = Rc::downgrade(&owner);
    let payload = HeapPayload { bytes, owner };
    let collector = header.collect(1024, 4096, 1, |_, _| Ok(payload)).unwrap();
    let old_payload_bytes = collector.header.collection_charge.payload_bytes();
    let retained = collector.into_retained().unwrap();
    assert_eq!(retained.header.original.plan() as *const _, plan);
    assert_eq!(Arc::as_ptr(retained.header.directory()), directory);
    assert_eq!(retained.header.manifest.source, source);
    assert_eq!(retained.header.manifest.config.as_ptr(), configuration);
    let moved = &retained.retained.payload;
    assert_eq!((moved.bytes.as_ptr(), moved.bytes.len(), moved.bytes.capacity()), backing);
    assert_eq!(moved.bytes.as_slice(), &[7; 13]);
    assert_eq!(Rc::as_ptr(&moved.owner), unique_owner.as_ptr());
    assert_eq!(Rc::strong_count(&moved.owner), 1);
    assert!(retained.retained.charge.is_for(counter.budget()));
    assert!(retained.retained.charge.payload_bytes() >= old_payload_bytes);
    assert_eq!(retained.header.collection_charge.payload_bytes(), 0);
    drop(retained);
    assert!(unique_owner.upgrade().is_none());
}

#[test]
fn foreign_collection_lease_is_not_adopted_even_with_equal_source_bytes() {
    let header = begin();
    let foreign = begin();
    assert_eq!(header.manifest.source, foreign.manifest.source);
    assert_eq!(header.manifest.config, foreign.manifest.config);
    let mut collector = header.collect(1024, 4096, 1, |_, _| Ok(())).unwrap();
    collector.header.collection_charge = foreign.root.charge_external(4096).unwrap();
    assert!(matches!(collector.into_retained(), Err(Error::Binding)));
}

#[test]
fn foreign_original_source_is_not_adopted_even_with_equal_header() {
    let header = begin();
    let foreign = begin();
    let mut collector = header.collect(1024, 4096, 1, |_, _| Ok(())).unwrap();
    collector.header.original = foreign.original;
    assert!(matches!(collector.into_retained(), Err(Error::Binding)));
}

#[derive(Clone, Copy, Debug)]
struct PayloadDrop {
    at: usize,
    covering_lease_live: bool,
    header_live: bool,
    directory_live: bool,
}

struct DropTrace {
    counter: OwnedBudget,
    next: Cell<usize>,
    covering_lease_live: Cell<bool>,
    header_live: Cell<bool>,
    payload: Cell<Option<PayloadDrop>>,
    lease_retiring_at: Cell<Option<usize>>,
    header_retiring_at: Cell<Option<usize>>,
    before_guard_used: Cell<Option<usize>>,
    at_header_used: Cell<Option<usize>>,
    after_guard_used: Cell<Option<usize>>,
    lease_retired_at: Cell<Option<usize>>,
}
impl DropTrace {
    fn tick(&self) -> usize {
        let at = self.next.get();
        self.next.set(at + 1);
        at
    }
}

enum ProbeKind {
    Header,
    CoveringLease,
    AfterCoveringLease,
}

/// Observation only, never a lease or runtime capability. The non-cloneable
/// covering-lease witness occupies the field immediately BEFORE that exact guard;
/// Rust field-drop order makes its LIVE state conservative: while LIVE, that
/// guard's destructor cannot have begun. Ingress moves it with the actual guard,
/// not with the replacement zero lease. No aggregate budget inequality is used.
pub(super) struct DropProbe {
    trace: Rc<DropTrace>,
    kind: ProbeKind,
}
impl Drop for DropProbe {
    fn drop(&mut self) {
        match self.kind {
            ProbeKind::Header => {
                self.trace.at_header_used.set(Some(self.trace.counter.budget().used_bytes()));
                assert!(self.trace.header_live.replace(false));
                self.trace.header_retiring_at.set(Some(self.trace.tick()));
            }
            ProbeKind::CoveringLease => {
                self.trace.before_guard_used.set(Some(self.trace.counter.budget().used_bytes()));
                assert!(self.trace.covering_lease_live.replace(false));
                self.trace.lease_retiring_at.set(Some(self.trace.tick()));
            }
            ProbeKind::AfterCoveringLease => {
                self.trace.after_guard_used.set(Some(self.trace.counter.budget().used_bytes()));
                self.trace.lease_retired_at.set(Some(self.trace.tick()));
            }
        }
    }
}

struct Observe {
    directory: std::sync::Weak<snapshot_directory::OwnedSnapshotDirectory>,
    trace: Rc<DropTrace>,
}
impl Drop for Observe {
    fn drop(&mut self) {
        assert!(self.trace.payload.get().is_none());
        self.trace.payload.set(Some(PayloadDrop {
            at: self.trace.tick(),
            covering_lease_live: self.trace.covering_lease_live.get(),
            header_live: self.trace.header_live.get(),
            directory_live: self.directory.upgrade().is_some(),
        }));
    }
}

fn observed() -> (SourceCollector<Observe>, OwnedBudget, Rc<DropTrace>) {
    let mut header = begin();
    let counter = header.root.try_clone().unwrap();
    let directory = Arc::downgrade(header.directory());
    let trace = Rc::new(DropTrace {
        counter: counter.try_clone().unwrap(),
        next: Cell::new(0),
        covering_lease_live: Cell::new(true),
        header_live: Cell::new(true),
        payload: Cell::new(None),
        lease_retiring_at: Cell::new(None),
        header_retiring_at: Cell::new(None),
        before_guard_used: Cell::new(None),
        at_header_used: Cell::new(None),
        after_guard_used: Cell::new(None),
        lease_retired_at: Cell::new(None),
    });
    header.header_drop_probe = Some(DropProbe { trace: Rc::clone(&trace), kind: ProbeKind::Header });
    header.collection_drop_probe = Some(DropProbe { trace: Rc::clone(&trace), kind: ProbeKind::CoveringLease });
    let observed = Rc::clone(&trace);
    let collector = header.collect(1024, 4096, 1, |_, _| Ok(Observe {
        directory, trace: observed,
    })).unwrap();
    assert!(collector.header.collection_charge.is_for(counter.budget()));
    (collector, counter, trace)
}

fn assert_payload_preceded_covering_guard(trace: &DropTrace) -> PayloadDrop {
    let payload = trace.payload.get().expect("payload destructor was observed");
    assert!(payload.covering_lease_live, "exact covering guard must still be LIVE");
    assert!(payload.at < trace.lease_retiring_at.get().unwrap());
    assert!(!trace.covering_lease_live.get());
    payload
}

#[test]
fn handoff_refusal_drops_payload_before_header_and_original_lease() {
    let (collector, counter, trace) = observed();
    let outside = counter.charge_external(
        MAX_PROGRAM_BYTES - counter.budget().used_bytes() - size_of::<OwnedExternalCharge>(),
    ).unwrap();
    assert!(matches!(collector.into_retained(), Err(Error::Bounds)));
    let payload = assert_payload_preceded_covering_guard(&trace);
    assert!(payload.directory_live);
    assert!(payload.header_live);
    // Header's first-field witness has not retired when payload runs; its
    // controller/charges are later fields and therefore have not retired either.
    assert!(payload.at < trace.header_retiring_at.get().unwrap());
    drop(outside);
}

#[test]
fn capsule_only_unwind_drops_payload_before_header_and_original_lease() {
    let (collector, counter, trace) = observed();
    let retained = collector.into_retained().unwrap();
    assert!(retained.retained.charge.is_for(counter.budget()));
    assert!(retained.retained.charge_drop_probe.is_some());
    assert!(Rc::ptr_eq(&retained.retained.charge_drop_probe.as_ref().unwrap().trace, &trace));
    assert!(retained.header.collection_drop_probe.is_none());
    assert_eq!(retained.header.collection_charge.payload_bytes(), 0);
    let exact_covering_charge = retained.retained.charge.charged_bytes();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _retained = retained;
        panic!("capsule-only cancellation, not downstream feed unwind");
    }));
    assert!(unwind.is_err());
    let payload = assert_payload_preceded_covering_guard(&trace);
    assert!(payload.directory_live);
    assert!(payload.header_live);
    assert!(payload.at < trace.header_retiring_at.get().unwrap());
    assert!(trace.lease_retiring_at.get().unwrap() < trace.header_retiring_at.get().unwrap());
    // These probes are adjacent to the one actual guard: PayloadLease's last
    // field drops, then the header's first-field probe runs. No header fields,
    // controller or other reservations retire between these two observations.
    assert_eq!(
        trace.before_guard_used.get().unwrap().checked_sub(trace.at_header_used.get().unwrap()),
        Some(exact_covering_charge),
    );
}

#[test]
fn real_finish_pre_feed_failure_retains_lease_not_header_until_payload_drop() {
    let (collector, _counter, trace) = observed();
    let fed = Cell::new(false);
    // Real consuming finish with empty records: observe PRE-FEED failure only.
    // No independent site/factory probe here: funding can also refuse before
    // registry validation. Neither validation nor importer creation is proved.
    let result = collector.finish(|_, _| {
        fed.set(true);
        panic!("pre-feed fixture unexpectedly reached feed");
    });
    assert!(result.is_err());
    assert!(!fed.get());
    let payload = assert_payload_preceded_covering_guard(&trace);
    assert!(!payload.header_live, "ordinary finish does not retain the header capsule on error");
    assert!(!payload.directory_live, "this test payload owns no directory reference");
    assert!(trace.header_retiring_at.get().unwrap() < payload.at);
    // Actual Collection retains its own directory. That separate ownership is
    // not a guarantee that future projection outputs retain header/controller.
}

// The independent real-publisher cases below require successful importer append;
// the pre-feed case above does not stand in for those later stages.

#[derive(Clone, Copy)]
enum FeedFault {
    Error,
    Unwind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingAllocations {
    raw_heap: (usize, usize, usize),
    original_config: (usize, usize),
    manifest_config_heap: (usize, usize, usize),
    controller_bytes: usize,
    budget: usize,
}

struct RealFeedState {
    fault: FeedFault,
    trace: std::cell::RefCell<Option<Rc<DropTrace>>>,
    norm_names: std::cell::RefCell<std::collections::BTreeSet<String>>,
    reached: Cell<bool>,
    allocations: Cell<Option<PendingAllocations>>,
    retirement_allocations: Cell<Option<PendingAllocations>>,
    first_raw_backing: Cell<Option<(u64, u64)>>,
    dropped_raw_backing: Cell<Option<(u64, u64)>>,
    snapshot_heap: Cell<Option<(usize, usize, usize)>>,
    dropped_snapshot_heap: Cell<Option<(usize, usize, usize)>>,
    directory: std::cell::RefCell<Option<std::sync::Weak<snapshot_directory::OwnedSnapshotDirectory>>>,
    exact_covering_charge: Cell<Option<usize>>,
    used_before_debit: Cell<Option<u64>>,
    used_after_debit: Cell<Option<u64>>,
    used_at_pending_retirement: Cell<Option<u64>>,
    pending_retired_at: Cell<Option<usize>>,
}

thread_local! {
    // At most one explicitly armed fixture on this test thread; no cross-test
    // failure switch, owner registry, source factory or retained Pending handle.
    static REAL_FEED: std::cell::RefCell<Option<Rc<RealFeedState>>> = const { std::cell::RefCell::new(None) };
}

struct RealFeedScope(Rc<RealFeedState>);
impl RealFeedScope {
    fn arm(fault: FeedFault) -> Self {
        let state = Rc::new(RealFeedState {
            fault,
            trace: std::cell::RefCell::new(None),
            norm_names: std::cell::RefCell::new(std::collections::BTreeSet::new()),
            reached: Cell::new(false),
            allocations: Cell::new(None),
            retirement_allocations: Cell::new(None),
            first_raw_backing: Cell::new(None),
            dropped_raw_backing: Cell::new(None),
            snapshot_heap: Cell::new(None),
            dropped_snapshot_heap: Cell::new(None),
            directory: std::cell::RefCell::new(None),
            exact_covering_charge: Cell::new(None),
            used_before_debit: Cell::new(None),
            used_after_debit: Cell::new(None),
            used_at_pending_retirement: Cell::new(None),
            pending_retired_at: Cell::new(None),
        });
        REAL_FEED.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none(), "nested real feed fixture is forbidden");
            *slot = Some(Rc::clone(&state));
        });
        Self(state)
    }
}
impl Drop for RealFeedScope {
    fn drop(&mut self) {
        REAL_FEED.with(|slot| {
            let armed = slot.borrow_mut().take().unwrap();
            assert!(Rc::ptr_eq(&armed, &self.0));
        });
    }
}

fn real_state(budget: &pllm_garble::shared_allocation::Budget) -> Option<Rc<RealFeedState>> {
    REAL_FEED.with(|slot| {
        let state = slot.borrow().as_ref().map(Rc::clone)?;
        let matches = state.trace.borrow().as_ref()
            .is_some_and(|trace| std::ptr::eq(trace.counter.budget(), budget));
        matches.then_some(state)
    })
}

pub(in crate::replicated_decoder) fn real_fixture_matches(budget: &pllm_garble::shared_allocation::Budget) -> bool {
    real_state(budget).is_some()
}

pub(in crate::replicated_decoder) fn real_payload_probe_armed(budget: &pllm_garble::shared_allocation::Budget) -> bool {
    real_state(budget).is_some_and(|state| state.reached.get())
}

pub(in crate::replicated_decoder) fn observe_real_collector<T>(owner: &mut SourceCollector<T>) {
    REAL_FEED.with(|slot| {
        let state = slot.borrow().as_ref().map(Rc::clone);
        let Some(state) = state else { return };
        assert!(state.trace.borrow().is_none(), "one real collector per fixture");
        let trace = Rc::new(DropTrace {
            // This real same-counter handle's control charge stays in both sides
            // of every measured baseline; it is not free capacity or a new root.
            counter: owner.header.clone_collection_budget().unwrap(),
            next: Cell::new(0),
            covering_lease_live: Cell::new(true),
            header_live: Cell::new(true),
            payload: Cell::new(None),
            lease_retiring_at: Cell::new(None),
            header_retiring_at: Cell::new(None),
            before_guard_used: Cell::new(None),
            at_header_used: Cell::new(None),
            after_guard_used: Cell::new(None),
            lease_retired_at: Cell::new(None),
        });
        let names = owner.header.original.plan().prefill.operations.iter()
            .filter(|op| op.operator == pllm_models::ModelOperator::RmsNorm)
            .map(|op| op.attributes["weight"].as_str().unwrap().to_owned())
            .collect();
        *state.norm_names.borrow_mut() = names;
        *state.directory.borrow_mut() = Some(Arc::downgrade(owner.header.directory()));
        owner.header.header_drop_probe = Some(DropProbe { trace: Rc::clone(&trace), kind: ProbeKind::Header });
        owner.header.collection_drop_probe = Some(DropProbe { trace: Rc::clone(&trace), kind: ProbeKind::CoveringLease });
        owner.header.collection_after_drop_probe = Some(DropProbe { trace: Rc::clone(&trace), kind: ProbeKind::AfterCoveringLease });
        *state.trace.borrow_mut() = Some(trace);
    });
}

pub(super) fn note_real_covering_charge(charge: &OwnedExternalCharge) {
    if let Some(state) = real_state(charge.budget()) {
        assert!(state.exact_covering_charge.replace(Some(charge.charged_bytes())).is_none());
    }
}

fn pending_allocations(pending: &super::super::Pending) -> Option<PendingAllocations> {
    let raw = match &pending.raw {
        SourceIngressRaw::CollectedHeader(raw) => raw,
        SourceIngressRaw::Wire(_) => return None,
    };
    if !std::ptr::eq(raw.budget(), pending.root.budget())
        || !std::ptr::eq(pending.original.budget(), pending.root.budget())
        || !pending.controller.is_for(pending.root.budget())
    {
        return None;
    }
    let raw_heap = raw.payload().raw_owner().payload();
    let config = pending.original.raw_configuration();
    let manifest_config = &pending.manifest.payload().config;
    if raw_heap.is_empty() || config.is_empty() || manifest_config.is_empty() { return None }
    Some(PendingAllocations {
        raw_heap: (raw_heap.as_ptr() as usize, raw_heap.len(), raw_heap.capacity()),
        original_config: (config.as_ptr() as usize, config.len()),
        manifest_config_heap: (manifest_config.as_ptr() as usize, manifest_config.len(), manifest_config.capacity()),
        controller_bytes: pending.controller.charged_bytes(),
        budget: pending.root.budget() as *const _ as usize,
    })
}

/// Exact allocation-free readback through the existing SAME-source bootstrap
/// query. Only reduce the query's work ceiling; never raise limits or construct
/// anything. Work means ceiling <= used; Ok/Quote mean ceiling > used. Quote
/// is mapped only after the remaining-ledger check, not itself a debit or proof
/// of a particular bootstrap quote failure reason.
fn cumulative_used(original: &OriginalSourcePlan, limits: original_source::Limits) -> Result<u64, original_source::Error> {
    fn above_used(original: &OriginalSourcePlan, mut limits: original_source::Limits, ceiling: u64) -> Result<bool, original_source::Error> {
        limits.max_work = ceiling;
        match original.lower_bootstrap_bytes(limits) {
            Err(original_source::Error::Work) => Ok(false),
            Ok(_) | Err(original_source::Error::Quote) => Ok(true),
            Err(other) => Err(other),
        }
    }
    if !above_used(original, limits, limits.max_work)? { return Err(original_source::Error::Work) }
    let (mut low, mut high) = (0, limits.max_work);
    while low < high {
        let mid = low + (high - low) / 2;
        if above_used(original, limits, mid)? { high = mid } else { low = mid + 1 }
    }
    low.checked_sub(1).ok_or(original_source::Error::Work)
}

const FEED_DEBIT: u64 = 7;
const FEED_PANIC: &str = "real first successful F32 read and importer append";

pub(in crate::replicated_decoder) fn after_real_import_append(
    import: &mut SourceRequestImport,
    backing: pllm_core::public_file::FileBacking,
    snapshot_heap: (usize, usize, usize),
) -> Result<(), Error> {
    let pending = import.0.as_ref().expect("successful append retains real Pending");
    let Some(state) = real_state(pending.root.budget()) else { return Ok(()) };
    assert!(!state.reached.replace(true), "inject only after the FIRST successful append");
    assert!(snapshot_heap.0 != 0 && snapshot_heap.1 > 0 && snapshot_heap.2 >= snapshot_heap.1);
    state.allocations.set(Some(pending_allocations(pending).expect("genuine same-root Pending allocations")));
    state.first_raw_backing.set(Some((backing.device, backing.file)));
    state.snapshot_heap.set(Some(snapshot_heap));
    let used = cumulative_used(&pending.original, pending.limits).unwrap();
    state.used_before_debit.set(Some(used));
    // Positive genuine debit on the original ledger, using its original
    // controller. This test-only observation does not claim late-error burn.
    pending.original.prepay_primitive_work(&pending.controller, FEED_DEBIT).unwrap();
    state.used_after_debit.set(Some(cumulative_used(&pending.original, pending.limits).unwrap()));
    match state.fault {
        FeedFault::Error => Err(Error::Binding),
        FeedFault::Unwind => panic!("{FEED_PANIC}"),
    }
}

pub(in crate::replicated_decoder) fn observe_real_payload_drop(
    budget: &pllm_garble::shared_allocation::Budget,
    backing: Option<pllm_core::public_file::FileBacking>,
    heap: (usize, usize, usize),
) {
    let state = real_state(budget).unwrap();
    state.dropped_raw_backing.set(backing.map(|backing| (backing.device, backing.file)));
    state.dropped_snapshot_heap.set(Some(heap));
    let trace = state.trace.borrow();
    let trace = trace.as_ref().unwrap();
    assert!(trace.payload.get().is_none());
    trace.payload.set(Some(PayloadDrop {
        at: trace.tick(),
        covering_lease_live: trace.covering_lease_live.get(),
        header_live: trace.header_live.get(),
        // Actual Collection and its snapshots retain their directory, unlike
        // the generic pre-feed fixture. No header-custody inference follows.
        directory_live: state.directory.borrow().as_ref().and_then(std::sync::Weak::upgrade).is_some(),
    }));
}

struct PendingRetiredProbe(Rc<RealFeedState>);
impl Drop for PendingRetiredProbe {
    fn drop(&mut self) {
        let trace = self.0.trace.borrow();
        self.0.pending_retired_at.set(Some(trace.as_ref().unwrap().tick()));
    }
}

// For these two injected cases only: move a NONOWNING witness beside the actual
// whole Pending. The tuple drops Pending first using its existing field order
// (including original config, raw and controller), then records its retirement.
// Normal unarmed test drops remain ordinary Option field drops. This does not
// observe an append error that has already dropped Pending internally.
impl Drop for SourceRequestImport {
    fn drop(&mut self) {
        let state = self.0.as_ref().and_then(|pending| real_state(pending.root.budget()))
            .filter(|state| state.reached.get());
        if let Some(state) = state {
            let pending = self.0.as_ref().unwrap();
            // Record, do not assert/panic during the injected unwind.
            state.retirement_allocations.set(pending_allocations(pending));
            state.used_at_pending_retirement.set(cumulative_used(&pending.original, pending.limits).ok());
            let owners_and_witness = (self.0.take().unwrap(), PendingRetiredProbe(state));
            drop(owners_and_witness);
        }
    }
}

fn real_publisher(fault: FeedFault) -> (crate::replicated_decoder::source::Publisher, RealFeedScope) {
    let scope = RealFeedScope::arm(fault);
    let config = br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}"#;
    let experiment = serde_json::to_vec(&serde_json::json!({
        "schema": "pllm.experiment.v1", "name": "real-compact-feed-custody",
        "pipeline": {"model": {"source": "original-tiny-source"}, "components": {
            "linear": {"component": crate::replicated_decoder::sdk::PROTOCOL,
                "params": {"weights": "public", "range_policy": "checked",
                    "max_abs_activation": 1, "storage": "compact"}}}},
        "deployment": {"kind": "local", "root": "."},
        "budget": {"requests": 1, "max_input_tokens": 1, "max_new_tokens": 2}
    })).unwrap();
    let mut publisher = crate::replicated_decoder::source::Publisher::from_compact_source_request(
        &experiment, config, 1, 2, [2; 32], [3; 32],
    ).expect("tiny REAL compact fixture must fit existing quotes; do not raise caps");
    for index in 0..publisher.shapes().len() {
        let (name, shape) = &publisher.shapes()[index];
        let name = name.clone();
        let count = shape.iter().product::<usize>();
        assert!((1..=4096).contains(&count), "bounded tiny fixture only");
        let value = if scope.0.norm_names.borrow().contains(&name) { 1f32 } else { 0f32 };
        let mut frame = Vec::with_capacity(count * 4);
        for _ in 0..count { frame.extend_from_slice(&value.to_le_bytes()); }
        publisher.append(&name, 0, &frame).expect("genuine original-F32 collection must fit existing quote");
    }
    (publisher, scope)
}

fn assert_real_feed_failure(scope: &RealFeedScope) {
    let state = &scope.0;
    assert!(state.reached.get(), "unresolved real quote/factory/append failure: injection was NOT reached");
    assert!(state.allocations.get().is_some());
    assert!(state.allocations.get().unwrap().controller_bytes > 0);
    assert_eq!(state.retirement_allocations.get(), state.allocations.get());
    assert!(state.first_raw_backing.get().is_some());
    assert_eq!(state.dropped_raw_backing.get(), state.first_raw_backing.get());
    assert_eq!(state.dropped_snapshot_heap.get(), state.snapshot_heap.get());
    let trace = state.trace.borrow();
    let trace = trace.as_ref().unwrap();
    let payload = assert_payload_preceded_covering_guard(trace);
    assert!(!payload.header_live);
    assert!(payload.directory_live, "Collection independently retains its own directory");
    assert!(trace.header_retiring_at.get().unwrap() < payload.at);
    assert!(state.pending_retired_at.get().unwrap() < payload.at,
        "real Pending/config/raw/controller retire before payload: custody limitation");
    assert!(trace.header_retiring_at.get().unwrap() < state.pending_retired_at.get().unwrap());
    assert!(payload.at < trace.lease_retired_at.get().unwrap());
    assert!(trace.lease_retiring_at.get().unwrap() < trace.lease_retired_at.get().unwrap());
    assert_eq!(trace.before_guard_used.get().unwrap().checked_sub(trace.after_guard_used.get().unwrap()),
        state.exact_covering_charge.get());
    assert!(state.exact_covering_charge.get().unwrap() > 0);
    let before = state.used_before_debit.get().unwrap();
    let after = state.used_after_debit.get().unwrap();
    assert_eq!(after.checked_sub(before), Some(FEED_DEBIT));
    assert_eq!(state.used_at_pending_retirement.get(), Some(after), "positive debit was not refunded");
    // No assertion of shared-ledger burn on late error, nor of config/controller
    // survival through payload/escaped output. Those stronger contracts are not
    // established by this existing ordinary finish path.
}

#[test]
fn real_finish_feed_error_keeps_covering_lease() {
    let (mut publisher, scope) = real_publisher(FeedFault::Error);
    let result = publisher.seal(&[0]);
    assert_real_feed_failure(&scope);
    assert_eq!(result, Err(Error::Binding.to_string()));
}

#[test]
fn real_finish_feed_unwind_keeps_covering_lease() {
    let (mut publisher, scope) = real_publisher(FeedFault::Unwind);
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| publisher.seal(&[0])));
    assert_real_feed_failure(&scope);
    let panic = unwind.expect_err("injected feed must unwind");
    let message = panic.downcast_ref::<String>().map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied());
    assert_eq!(message, Some(FEED_PANIC));
}
