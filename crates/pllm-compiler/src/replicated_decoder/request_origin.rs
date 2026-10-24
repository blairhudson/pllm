//! Linear, compiler-logical request history. NOT runtime acceptance or drain.
//!
//! Origin comes from the actual guarded config factory AND the actual token Input
//! creation. Feedback comes only from sealed events inside original graph lowering.
//! A five-token prefix aliases four original tokens plus one selected producer;
//! it is never a newly asserted private input or a digest/interval certificate.
use super::*;
use crate::original_source::{self, OriginalSourcePlan};
use crate::replicated_decoder::source_registry::{self, SourceRegistry};
use pllm_garble::shared_allocation::{OwnedCharged, OwnedExternalCharge};
use pllm_models::DecoderPlan;
use std::{
    mem::size_of,
    sync::atomic::{AtomicU8, Ordering},
};

const OPEN: u8 = 0;
const BURNED: u8 = 1;
const COMPLETE: u8 = 2;

#[path = "compact_request_storage.rs"]
pub(super) mod compact_storage;
#[path = "rotary_creation.rs"]
pub(in crate::replicated_decoder) mod rotary_creation;
#[path = "request_schedule.rs"]
pub(super) mod schedule;
#[path = "source_bias.rs"]
pub(in crate::replicated_decoder) mod source_bias;

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn bytes<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
}
fn reserve<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    if values.capacity() != n {
        return Err(Error::Allocation);
    }
    Ok(values)
}

/// Opaque, single-use authorization for program's actual Input creation hook.
pub(super) struct InputMint {
    rows: usize,
    bounds: Interval,
}
impl InputMint {
    pub(super) fn parts(self) -> (usize, Interval) {
        (self.rows, self.bounds)
    }
}

struct Origin {
    input: Arc<original_import::InputRecord>,
    source: Arc<OriginalSourcePlan>,
    public: Arc<PublicSource>,
    registry_identity: source_registry::Instance,
    source_records: usize,
    bounds: Interval,
    state: AtomicU8,
}

/// Borrows the exact validated descriptor maps and retains the authentic files.
/// Config-factory lineage alone is not checkpoint/source authentication.
pub(super) struct RequestSources<'s> {
    manifest: &'s Manifest,
    registry: &'s SourceRegistry<'s>,
    public: Arc<PublicSource>,
}
impl<'s> RequestSources<'s> {
    pub(super) fn new(
        manifest: &'s Manifest,
        registry: &'s SourceRegistry<'s>,
        public: Arc<PublicSource>,
    ) -> Result<Self, Error> {
        if manifest.weights != WeightMode::Public
            || !std::ptr::eq(&manifest.tensors, registry.source_records())
            || !std::ptr::eq(&manifest.norms, registry.source_norms())
        {
            return Err(Error::Binding);
        }
        // Actual binding/file validation occurs only after declaration's quote
        // has admitted its named workspace. No replacement source is created.
        Ok(Self {
            manifest,
            registry,
            public,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Resources {
    pub payload_bytes: usize,
    /// Additional live counter bytes at declaration, including the external
    /// guard and OwnedCharged's real pre-factory inline growth. Original source
    /// guards and later lowering controllers remain separately retained.
    pub required_counter_bytes: usize,
    pub metadata_work: u64,
    pub rows: usize,
    pub outputs: usize,
    pub max_operations: usize,
    pub projection_slots: usize,
}

/// Metadata quote only. Source/model factories retain their own guarded work;
/// native arithmetic and full request CPU/stack/RSS are NOT admitted by this quote.
pub(super) fn quote(
    builder: &Builder,
    source: &OriginalSourcePlan,
    sources: &RequestSources<'_>,
) -> Result<Resources, Error> {
    let workload = source.workload();
    let rows = usize::try_from(workload.max_input_tokens).map_err(|_| Error::Bounds)?;
    let outputs = usize::try_from(workload.max_new_tokens).map_err(|_| Error::Bounds)?;
    let total = add(rows, outputs)?;
    if workload.batch != 1
        || rows == 0
        || outputs == 0
        || total > 40
        || outputs > 8
        || source.raw_configuration().is_empty()
        || source.raw_configuration().len() > 16384
        || source.plan().prefill.query_sequence != workload.max_input_tokens
        || !source.plan().token_feedback
        || builder.request_emission.is_some()
        || builder.checked_bound.is_none()
    {
        return Err(Error::Binding);
    }
    let graph = &source.plan().prefill;
    let output = graph
        .operations
        .iter()
        .find(|op| op.id == graph.output)
        .ok_or(Error::Binding)?;
    let selected = output
        .inputs
        .first()
        .and_then(|name| graph.operations.iter().find(|op| &op.id == name))
        .ok_or(Error::Binding)?;
    if output.operator != Op::TokenFeedback
        || output.inputs.len() != 1
        || selected.operator != Op::GreedyTokenSelection
    {
        return Err(Error::Binding);
    }
    let max_operations = graph.operations.len();
    // One semantic producer plus, at most, one specialized creation per op.
    let projection_slots = max_operations
        .checked_mul(outputs)
        .and_then(|n| n.checked_mul(2))
        .ok_or(Error::Bounds)?;
    let mut payload = 0;
    // Full existing creation-record capacity remains charged conservatively;
    // destination backing overlaps old backing during any required reserve.
    let input_next = add(builder.original_inputs.len(), 1)?;
    let record_slots = if input_next > builder.original_inputs.capacity() {
        input_next
    } else {
        0
    };
    let node_slots = if builder.nodes.len() == builder.nodes.capacity() {
        add(builder.nodes.len(), 1)?
    } else {
        0
    };
    for n in [
        bytes::<Arc<OriginalSourcePlan>>(outputs)?,
        // Actual owned source-result wrapper per epoch, not Arc-from-borrow.
        bytes::<OriginalSourcePlan>(outputs)?,
        bytes::<usize>(outputs.checked_mul(2).ok_or(Error::Bounds)?)?,
        bytes::<Fragment>(outputs)?,
        source_bias::creation_bytes(projection_slots, sources.registry.source_records())?,
        // Same-counter metadata controllers moved into actual graph emission.
        bytes::<OwnedExternalCharge>(outputs)?,
        bytes::<(usize, usize)>(add(outputs, 1)?)?,
        // Two preallocated pools: growing tokens and active epoch snapshot.
        Words::span_pool_bytes(add(outputs, 1)?)
            .and_then(|n| n.checked_mul(2))
            .ok_or(Error::Bounds)?,
        // Existing acceptance's selected-token and broadcast-flag temporaries.
        Words::span_pool_bytes(outputs.checked_next_power_of_two().ok_or(Error::Bounds)?)
            .and_then(|n| n.checked_add(Words::span_pool_bytes(1)?))
            .ok_or(Error::Bounds)?,
        bytes::<usize>(12)?,
        bytes::<Arc<original_import::InputRecord>>(add(
            builder.original_inputs.capacity(),
            record_slots,
        )?)?,
        bytes::<Node>(node_slots)?,
        add(
            size_of::<original_import::InputRecord>(),
            bytes::<usize>(2)?,
        )?,
        add(size_of::<Origin>(), bytes::<usize>(2)?)?,
        size_of::<OwnedExternalCharge>(),
        size_of::<OwnedCharged<Origin>>(),
        bytes::<u64>(rows.checked_mul(2).ok_or(Error::Bounds)?)?,
        add(size_of::<Node>(), "input.tokens".len())?,
        size_of::<History<'_>>(),
        size_of::<RequestInputPermit<'_>>(),
        size_of::<PendingRequest<'_>>(),
        size_of::<LogicalComplete<'_>>(),
        size_of::<Active>(),
        size_of::<BoundExpanded<'_, '_>>(),
        size_of::<graph::SelectionEvent<'_>>(),
        size_of::<graph::FeedbackEvent<'_>>(),
        bytes::<request_driver::Plan>(2)?,
        bytes::<HistoryInput<'_, '_>>(2)?,
        bytes::<Result<HistoryInput<'_, '_>, Error>>(2)?,
        bytes::<Result<Resources, Error>>(2)?,
        bytes::<OwnedExternalCharge>(2)?,
        size_of::<InputMint>(),
        size_of::<super::CreatedRequestInput>(),
        // Capture/return/move slots before storage into Active/Fragment.
        size_of::<Result<Record, Error>>(),
        size_of::<Selection>(),
        size_of::<Fragment>(),
        size_of::<Option<Fragment>>(),
        bytes::<Result<(), Error>>(4)?,
        bytes::<SourcePlanSlot>(2)?,
        OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
        size_of::<EmissionGuard<'_>>(),
        size_of::<RequestSources<'_>>(),
        // check_registry lends cached binding and file descriptors; no SHA
        // state or checkpoint re-encoding is created by this transition.
        size_of::<[u8; 32]>(),
        std::mem::size_of_val(&sources.registry.files()),
        size_of::<source_registry::FileDescriptor<'_>>(),
        size_of::<Option<&pllm_core::public_file::PublicFile>>(),
    ] {
        payload = add(payload, n)?;
    }
    // Full retained external descriptor/file metadata: actual capacities, no
    // disk-length or alias credit. Source factory/config guards remain separate.
    let manifest = sources.manifest;
    let mut retained = add(size_of::<Manifest>(), manifest.config.capacity())?;
    retained = add(retained, manifest.schema.capacity())?;
    retained = add(
        retained,
        source_registry::source_metadata_bytes(&manifest.tensors, &manifest.norms)?,
    )?;
    retained = add(retained, sources.public.resident_metadata_bytes()?)?;
    retained = add(retained, bytes::<usize>(2)?)?; // public Arc counts
    retained = add(retained, size_of::<SourceRegistry<'_>>())?;
    payload = add(payload, retained)?;
    // Named logical inspections, NOT CPU fuel or native recipe work. Charge
    // byte visits for retained binding/parameter data plus original graph scans;
    // each later factory epoch is charged separately from its actual owner.
    let graph_bytes = source_registry::semantic_metadata_bytes(graph)?;
    // Prepay all fixed history checks, including paths which fail before a
    // subsequent source factory or epoch debit. Logical visits, not magic CPU
    // multipliers: origin checks, token checks and graph-entry coordinate checks.
    let input_records = add(builder.original_inputs.len(), 1)?;
    let origin_checks = add(outputs, 2)?;
    let token_checks = add(outputs.checked_mul(2).ok_or(Error::Bounds)?, 2)?;
    let origin_visits = origin_checks
        .checked_mul(add(
            input_records,
            add(outputs.checked_mul(2).ok_or(Error::Bounds)?, 1)?,
        )?)
        .ok_or(Error::Bounds)?;
    let token_visits = token_checks
        .checked_mul(add(rows, outputs.checked_mul(2).ok_or(Error::Bounds)?)?)
        .ok_or(Error::Bounds)?;
    let entry_visits = outputs
        .checked_mul(add(input_records, total)?)
        .ok_or(Error::Bounds)?;
    let work = add(
        add(add(origin_visits, token_visits)?, entry_visits)?,
        add(
            add(retained, manifest.config.len())?,
            add(
                graph_bytes,
                add(
                    max_operations.checked_mul(2).ok_or(Error::Bounds)?,
                    add(builder.original_inputs.len(), total)?,
                )?,
            )?,
        )?,
    )?;
    if payload > MAX_PROGRAM_BYTES {
        return Err(Error::Bounds);
    }
    Ok(Resources {
        payload_bytes: payload,
        required_counter_bytes: add(payload, size_of::<OwnedCharged<Origin>>())?,
        metadata_work: u64::try_from(work).map_err(|_| Error::Bounds)?,
        rows,
        outputs,
        max_operations,
        projection_slots,
    })
}

// Named quote/control slot; no plan document is copied into this slot.
type SourcePlanSlot = Result<OriginalSourcePlan, original_source::Error>;

/// No Clone, no constructor from a View/seal, no caller-supplied history baseline.
pub(super) struct RequestInputPermit<'s> {
    history: History<'s>,
    tokens: View,
}
impl RequestInputPermit<'_> {
    /// SAME creation-owned plan allocation, for the consuming source-custody
    /// transfer. This borrow cannot mint/replay a permit or an operator owner.
    pub(super) fn retained_original(&self) -> Result<&Arc<OriginalSourcePlan>, Error> {
        self.history.epochs.first().ok_or(Error::Binding)
    }
    pub(super) fn lowering_control_bytes(
        &self,
        limits: original_source::Limits,
    ) -> Result<usize, Error> {
        if self.history.resources.outputs == 1 {
            return Ok(0);
        }
        let bootstrap = self
            .history
            .origin
            .payload()
            .source
            .lower_bootstrap_bytes(limits)
            .map_err(|_| Error::Bounds)?;
        add(bootstrap, size_of::<OwnedExternalCharge>())?
            .checked_mul(self.history.resources.outputs - 1)
            .ok_or(Error::Bounds)
    }
}

struct History<'s> {
    epochs: Vec<Arc<OriginalSourcePlan>>,
    fragments: Vec<Fragment>,
    projections: Vec<Option<rotary_creation::CreatedOperator>>,
    scratch: Words,
    origin: HistoryOrigin,
    sources: RequestSources<'s>,
    resources: Resources,
    sticky: Option<usize>,
    charge: OwnedExternalCharge,
}

/// Burning is attached to the origin, rather than to the container, so a
/// completed history can MOVE its original guards into its final Program.
struct HistoryOrigin(Arc<OwnedCharged<Origin>>);
impl std::ops::Deref for HistoryOrigin {
    type Target = Arc<OwnedCharged<Origin>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for HistoryOrigin {
    fn drop(&mut self) {
        let _ = self.0.payload().state.compare_exchange(
            OPEN,
            BURNED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

/// Exact original selected/sticky records. Scalar records copy no lane arrays;
/// Tensor/Selection/Product records retain their SAME original native Arcs.
enum RecordCode {
    Fixed(Code),
    PublicScalar(i64),
}
struct Record {
    code: RecordCode,
    lanes: usize,
    bounds: Interval,
}
impl Record {
    fn capture(node: &Node) -> Result<Self, Error> {
        let code = match &node.code {
            Code::Affine { .. }
            | Code::Product { .. }
            | Code::Tensor { .. }
            | Code::CheckedRange { .. }
            | Code::Selection { .. } => RecordCode::Fixed(node.code.clone()),
            Code::PublicValues(v) if v.len() == 1 => RecordCode::PublicScalar(v[0]),
            _ => return Err(Error::Binding),
        };
        Ok(Self {
            code,
            lanes: node.lanes,
            bounds: node.bounds,
        })
    }
    fn matches(&self, node: &Node) -> bool {
        if node.lanes != self.lanes || node.bounds != self.bounds {
            return false;
        }
        match (&self.code, &node.code) {
            (RecordCode::PublicScalar(a), Code::PublicValues(b)) => b.len() == 1 && b[0] == *a,
            (
                RecordCode::Fixed(Code::Affine {
                    left: a,
                    right: b,
                    coefficients: c,
                }),
                Code::Affine {
                    left: x,
                    right: y,
                    coefficients: z,
                },
            ) => a == x && b == y && (c.left, c.right, c.constant) == (z.left, z.right, z.constant),
            (
                RecordCode::Fixed(Code::Product {
                    left: a,
                    right: b,
                    shape: c,
                    rescale: d,
                    norm: e,
                }),
                Code::Product {
                    left: v,
                    right: w,
                    shape: x,
                    rescale: y,
                    norm: z,
                },
            ) => {
                a == v
                    && b == w
                    && c == x
                    && d == y
                    && match (e, z) {
                        (None, None) => true,
                        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                        _ => false,
                    }
            }
            (
                RecordCode::Fixed(Code::Tensor {
                    input: a,
                    program: b,
                }),
                Code::Tensor {
                    input: x,
                    program: y,
                },
            )
            | (
                RecordCode::Fixed(Code::CheckedRange {
                    input: a,
                    program: b,
                }),
                Code::CheckedRange {
                    input: x,
                    program: y,
                },
            ) => a == x && Arc::ptr_eq(b, y),
            (
                RecordCode::Fixed(Code::Selection { input: a, plan: b }),
                Code::Selection { input: x, plan: y },
            ) => a == x && Arc::ptr_eq(b, y),
            _ => false,
        }
    }
}

struct Selection {
    word: (usize, usize),
    operation: usize,
    record: Record,
}
struct Fragment {
    selection: Selection,
    plan: Arc<OriginalSourcePlan>,
    epoch: usize,
    sticky_in: Option<usize>,
    sticky_out: Option<usize>,
    sticky_record: Option<Record>,
}

/// Exists only while this original Builder emits this actual source-plan epoch.
pub(super) struct Active {
    origin: Arc<OwnedCharged<Origin>>,
    plan: Arc<OriginalSourcePlan>,
    epoch: usize,
    sticky_in: Option<usize>,
    selected: Option<Selection>,
    feedback: Option<Fragment>,
    input_words: Words,
    projections: Vec<Option<rotary_creation::CreatedOperator>>,
    rotary: Option<rotary_creation::Emission>,
    produced_charge: Option<OwnedExternalCharge>,
    creation_charge: Option<OwnedExternalCharge>,
    source_records: usize,
    registry_identity: source_registry::Instance,
    snapshot_remaining: usize,
    committed: bool,
}
impl Active {
    pub(in crate::replicated_decoder::program) fn semantic_plan(&self) -> &Arc<OriginalSourcePlan> {
        &self.plan
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.origin.payload().state.compare_exchange(
                OPEN,
                BURNED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

/// Clear active ownership on both error and panic, before the borrowed Builder
/// escapes. Successful graph emission leaves it only for append_feedback.
struct EmissionGuard<'b> {
    builder: &'b mut Builder,
    keep: bool,
}
impl Drop for EmissionGuard<'_> {
    fn drop(&mut self) {
        if !self.keep {
            self.builder.request_emission = None;
        }
    }
}

pub(super) fn graph_entry(
    builder: &Builder,
    graph: &DecoderGraph,
    tokens: &View,
) -> Result<(), Error> {
    let Some(active) = &builder.request_emission else {
        return Ok(());
    };
    if active.origin.payload().state.load(Ordering::Acquire) != OPEN
        || !std::ptr::eq(graph, &active.plan.plan().prefill)
        || tokens.words != active.input_words
        || !builder
            .original_inputs
            .iter()
            .any(|record| Arc::ptr_eq(record, &active.origin.payload().input))
    {
        return Err(Error::Binding);
    }
    Ok(())
}

pub(super) fn declare<'s>(
    builder: &mut Builder,
    source: OriginalSourcePlan,
    sources: RequestSources<'s>,
    mut charge: OwnedExternalCharge,
) -> Result<RequestInputPermit<'s>, Error> {
    if !charge.is_for(source.budget()) {
        return Err(Error::Binding);
    }
    let resources = quote(builder, &source, &sources)?;
    if charge.payload_bytes() < resources.payload_bytes {
        return Err(Error::Bounds);
    }
    let manifest = sources.manifest;
    let workload = source.workload();
    if source.raw_configuration() != manifest.config.as_slice()
        || workload.max_input_tokens != manifest.rows as u64
        || workload.max_new_tokens != manifest.outputs as u64
        || !builder.public_weights
        || builder.checked_bound != manifest.execution.checked_bound()
        || builder.lookup != manifest.execution.lookup
        || builder.selection != manifest.execution.selection
        || builder.token_source_binding
            != if builder.lookup == LookupPolicy::Trie {
                sources.registry.binding()
            } else {
                [0; 32]
            }
    {
        return Err(Error::Binding);
    }
    let vocabulary = source
        .plan()
        .prefill
        .operations
        .iter()
        .find(|op| op.operator == Op::OutputHead)
        .and_then(|op| op.output_shape.last())
        .copied()
        .ok_or(Error::Binding)?;
    let bounds = Interval::new(
        0,
        i64::try_from(vocabulary)
            .map_err(|_| Error::Bounds)?
            .checked_sub(1)
            .ok_or(Error::Bounds)?,
    )?;
    source
        .prepay_compiler_metadata(&charge, resources.metadata_work)
        .map_err(|_| Error::Bounds)?;
    sources.public.check_registry(sources.registry)?;
    // Both children keep the SAME original counter; no guard is detached.
    let root_charge = charge
        .split_owned(add(size_of::<Origin>(), bytes::<usize>(2)?)?)
        .map_err(|_| Error::Bounds)?;
    let mut epochs = reserve(resources.outputs)?;
    let fragments = reserve(resources.outputs)?;
    let projections = reserve(resources.projection_slots)?;
    let mut words = Words::try_with_spans(add(resources.outputs, 1)?)?;
    let scratch = Words::try_with_spans(add(resources.outputs, 1)?)?;
    let source = Arc::new(source);
    let mut failure = None;
    // try_build admits its inline/padding before the genuine Input factory too.
    let origin = root_charge
        .try_build(|_| {
            let record = builder
                .create_request_input(InputMint {
                    rows: resources.rows,
                    bounds,
                })
                .map_err(|error| {
                    failure = Some(error);
                    pllm_garble::shared_allocation::Error::Resource
                })?;
            words.extend((0..resources.rows).map(|lane| (record.owner(), lane)));
            epochs.push(Arc::clone(&source));
            Ok(Origin {
                input: record,
                source,
                public: Arc::clone(&sources.public),
                registry_identity: sources.registry.instance(),
                source_records: std::ptr::from_ref(sources.registry.source_records()) as usize,
                bounds,
                state: AtomicU8::new(OPEN),
            })
        })
        .map_err(|_| failure.unwrap_or(Error::Bounds))?;
    let origin = Arc::new(origin);
    #[cfg(test)]
    MINTS.with(|v| v.set(v.get() + 1));
    Ok(RequestInputPermit {
        tokens: View {
            words,
            shape: vec![1, resources.rows],
        },
        history: History {
            epochs,
            fragments,
            projections,
            scratch,
            origin: HistoryOrigin(origin),
            sources,
            resources,
            sticky: builder.validity,
            charge,
        },
    })
}

fn check_origin(history: &History<'_>, builder: &Builder) -> Result<(), Error> {
    check_origin_state(history, builder, OPEN)
}

fn check_origin_state(history: &History<'_>, builder: &Builder, state: u8) -> Result<(), Error> {
    let origin = history.origin.payload();
    if origin.state.load(Ordering::Acquire) != state
        || !Arc::ptr_eq(&origin.public, &history.sources.public)
        || !builder
            .original_inputs
            .iter()
            .any(|record| Arc::ptr_eq(record, &origin.input))
    {
        return Err(Error::Binding);
    }
    let node = builder
        .nodes
        .get(origin.input.owner())
        .ok_or(Error::Binding)?;
    if !matches!(&node.code, Code::Input(key) if key == "input.tokens")
        || node.lanes != history.resources.rows
        || node.bounds != origin.bounds
    {
        return Err(Error::Binding);
    }
    for fragment in &history.fragments {
        let selected = builder
            .nodes
            .get(fragment.selection.word.0)
            .ok_or(Error::Binding)?;
        if !fragment.selection.record.matches(selected) {
            return Err(Error::Binding);
        }
        match (fragment.sticky_out, &fragment.sticky_record) {
            (Some(id), Some(record))
                if record.matches(builder.nodes.get(id).ok_or(Error::Binding)?) => {}
            (None, None) => {}
            _ => return Err(Error::Binding),
        }
    }
    Ok(())
}

fn check_source_inputs(
    history: &History<'_>,
    builder: &Builder,
    inputs: &BTreeMap<String, SourceView>,
) -> Result<(), Error> {
    let registry = history.sources.registry;
    if inputs.len() != registry.records().count()
        || builder.public_norms.len() != registry.source_norms().len()
    {
        return Err(Error::Binding);
    }
    for (_, key, record, norm) in registry.records() {
        let view = inputs.get(key).ok_or(Error::Binding)?;
        let node = builder.nodes.get(view.owner).ok_or(Error::Binding)?;
        if view.shape != record.shape
            || node.lanes != record.shape.iter().product::<usize>()
            || node.bounds != Interval::new(record.lower, record.upper)?
            || builder.public_norms.get(&view.owner) != norm
        {
            return Err(Error::Binding);
        }
        match &node.code {
            Code::PublicInput {
                key: actual,
                digest,
                row_l1,
                row_squared_norm,
                row_l1_squared_sum,
            } if actual == key
                && *digest == record.digest
                && *row_l1 == record.row_l1
                && *row_squared_norm == record.row_squared_norm
                && *row_l1_squared_sum == record.row_l1_squared_sum => {}
            _ => return Err(Error::Binding),
        }
    }
    Ok(())
}

impl History<'_> {
    /// Logical metadata visits through actual retained owners, not a native work
    /// formula/opaque CPU bound. Future plan construction has its own original
    /// guarded factory quote; it does not borrow this admission.
    fn prepay_epoch(&mut self, epoch: usize) -> Result<(), Error> {
        let plan = self.epochs.get(epoch).ok_or(Error::Binding)?;
        let graph = &plan.plan().prefill;
        let visits = add(
            source_registry::semantic_metadata_bytes(graph)?,
            graph.operations.len().checked_mul(2).ok_or(Error::Bounds)?,
        )?;
        let visits = add(
            visits,
            source_bias::creation_work(graph, self.sources.registry.source_records())?,
        )?;
        // Original mapping snapshots coexist with source/native owners and all
        // prior epochs. Grow same guard BEFORE operator emission; never refund.
        let snapshot = source_bias::snapshot_bytes(graph)?;
        self.charge
            .resize_payload(add(self.charge.payload_bytes(), snapshot)?)
            .map_err(|_| Error::Bounds)?;
        self.origin
            .payload()
            .source
            .prepay_compiler_metadata(
                &self.charge,
                u64::try_from(visits).map_err(|_| Error::Bounds)?,
            )
            .map_err(|_| Error::Bounds)
    }

    fn check_tokens(&self, tokens: &View) -> Result<(), Error> {
        if self
            .epochs
            .first()
            .is_none_or(|plan| !Arc::ptr_eq(plan, &self.origin.payload().source))
        {
            return Err(Error::Binding);
        }
        for (epoch, source) in self.epochs.iter().enumerate() {
            let workload = source.workload();
            if !source.same_origin(&self.origin.payload().source)
                || workload.batch != 1
                || workload.max_input_tokens
                    != u64::try_from(add(self.resources.rows, epoch)?).map_err(|_| Error::Bounds)?
                || workload.max_new_tokens
                    != u64::try_from(
                        self.resources
                            .outputs
                            .checked_sub(epoch)
                            .ok_or(Error::Bounds)?,
                    )
                    .map_err(|_| Error::Bounds)?
            {
                return Err(Error::Binding);
            }
        }
        let rows = add(self.resources.rows, self.fragments.len())?;
        if tokens.shape != [1, rows] || tokens.words.len() != rows {
            return Err(Error::Binding);
        }
        for lane in 0..self.resources.rows {
            if tokens.words.at(lane) != (self.origin.payload().input.owner(), lane) {
                return Err(Error::Binding);
            }
        }
        for (i, fragment) in self.fragments.iter().enumerate() {
            if fragment.epoch != i
                || self
                    .epochs
                    .get(i)
                    .is_none_or(|plan| !Arc::ptr_eq(plan, &fragment.plan))
                || tokens.words.at(self.resources.rows + i) != fragment.selection.word
                || (i > 0 && fragment.sticky_in != self.fragments[i - 1].sticky_out)
            {
                return Err(Error::Binding);
            }
        }
        Ok(())
    }
}

/// Borrowed compiler creation/alias seal. Indices are interpreted only against
/// retained immutable actual owners, never dereferenced through raw pointers.
pub(super) struct HistoryInput<'a, 's> {
    history: &'a History<'s>,
    epoch: usize,
}
impl HistoryInput<'_, '_> {
    pub(super) fn plan(&self) -> &DecoderPlan {
        self.history.epochs[self.epoch].plan()
    }
    pub(super) fn rows(&self) -> usize {
        self.history.resources.rows + self.epoch
    }
    pub(super) fn word(&self, lane: usize) -> Result<(usize, usize), Error> {
        if lane < self.history.resources.rows {
            return Ok((self.history.origin.payload().input.owner(), lane));
        }
        let fragment = self
            .history
            .fragments
            .get(lane - self.history.resources.rows)
            .ok_or(Error::Binding)?;
        if fragment.epoch >= self.epoch {
            return Err(Error::Binding);
        }
        Ok(fragment.selection.word)
    }
    pub(super) fn same_origin(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.history.origin, &other.history.origin)
    }
    /// No runtime/copy/import authority follows from this logical link.
    pub(super) fn follows(&self, previous: &Self) -> bool {
        self.same_origin(previous)
            && self.epoch == previous.epoch + 1
            && self
                .history
                .fragments
                .get(previous.epoch)
                .is_some_and(|f| f.epoch == previous.epoch)
            && self.history.epochs[self.epoch].same_origin(&previous.history.epochs[previous.epoch])
    }
}

pub(super) struct PendingRequest<'s> {
    selected: Vec<(usize, usize)>,
    tokens: View,
    history: History<'s>,
}
impl<'s> PendingRequest<'s> {
    pub(super) fn input(&self, epoch: usize) -> Result<HistoryInput<'_, 's>, Error> {
        if epoch >= self.history.epochs.len() {
            return Err(Error::Binding);
        }
        Ok(HistoryInput {
            history: &self.history,
            epoch,
        })
    }
    pub(super) fn cancel(self) {
        drop(self);
    }
}

/// Logical checked-epilogue construction only. NOT runtime acceptance, reveal,
/// proof, completion/drain capability, or an SDK result.
pub(super) struct LogicalComplete<'s> {
    outputs: Vec<(usize, usize)>,
    pending: PendingRequest<'s>,
    accepted: [(usize, Record); 2],
    nodes: usize,
}
impl<'s> LogicalComplete<'s> {
    pub(super) fn outputs(&self) -> &[(usize, usize)] {
        &self.outputs
    }
    pub(super) fn input(&self, epoch: usize) -> Result<HistoryInput<'_, 's>, Error> {
        self.pending.input(epoch)
    }
}

pub(super) fn selected(
    builder: &mut Builder,
    event: graph::SelectionEvent<'_>,
) -> Result<(), Error> {
    let Some(active) = builder.request_emission.as_mut() else {
        return Ok(());
    };
    if active.origin.payload().state.load(Ordering::Acquire) != OPEN
        || !std::ptr::eq(event.graph(), &active.plan.plan().prefill)
        || active.selected.is_some()
        || event.view().words.len() != 1
    {
        return Err(Error::Binding);
    }
    let operation = active
        .plan
        .plan()
        .prefill
        .operations
        .iter()
        .position(|op| std::ptr::eq(op, event.operation()))
        .ok_or(Error::Binding)?;
    let word = event.view().words.at(0);
    let node = builder.nodes.get(word.0).ok_or(Error::Binding)?;
    if word.1 >= node.lanes {
        return Err(Error::Binding);
    }
    active.selected = Some(Selection {
        word,
        operation,
        record: Record::capture(node)?,
    });
    #[cfg(test)]
    match SELECTION_FAULT.with(|fault| fault.get()) {
        1 => return Err(Error::Binding),
        2 => panic!("test unwind after genuine greedy producer publication"),
        _ => {}
    }
    Ok(())
}

pub(super) fn feedback(
    builder: &mut Builder,
    event: graph::FeedbackEvent<'_>,
) -> Result<(), Error> {
    let Some(active) = builder.request_emission.as_mut() else {
        return Ok(());
    };
    if active.origin.payload().state.load(Ordering::Acquire) != OPEN
        || !std::ptr::eq(event.graph(), &active.plan.plan().prefill)
        || active.feedback.is_some()
        || event.view().words.len() != 1
    {
        return Err(Error::Binding);
    }
    let graph = &active.plan.plan().prefill;
    if event.operation().id != graph.output || event.operation().inputs.len() != 1 {
        return Err(Error::Binding);
    }
    let selection = active.selected.take().ok_or(Error::Binding)?;
    let producer = graph
        .operations
        .get(selection.operation)
        .ok_or(Error::Binding)?;
    if producer.operator != Op::GreedyTokenSelection
        || event.operation().inputs[0] != producer.id
        || event.view().words.at(0) != selection.word
    {
        return Err(Error::Binding);
    }
    let sticky_out = builder.validity;
    if active.sticky_in.is_some() && sticky_out.is_none() {
        return Err(Error::Binding);
    }
    let sticky_record = sticky_out
        .map(|id| Record::capture(builder.nodes.get(id).ok_or(Error::Binding)?))
        .transpose()?;
    active.feedback = Some(Fragment {
        selection,
        plan: Arc::clone(&active.plan),
        epoch: active.epoch,
        sticky_in: active.sticky_in,
        sticky_out,
        sticky_record,
    });
    Ok(())
}

/// Existing driver loop owns all history. Caller supplies already admitted source
/// construction controllers; no implicit/unpriced lowering or alternate runner.
pub(super) fn drive_context<'s>(
    permit: RequestInputPermit<'s>,
    builder: &mut Builder,
    inputs: &BTreeMap<String, SourceView>,
    sources: &BTreeMap<String, SourceTensor>,
    mut lowering: OwnedExternalCharge,
    limits: original_source::Limits,
    site: &mut AdmissionSite,
) -> Result<PendingRequest<'s>, Error> {
    let RequestInputPermit {
        mut history,
        mut tokens,
    } = permit;
    check_origin(&history, builder)?;
    if !std::ptr::eq(sources, history.sources.registry.source_records()) {
        return Err(Error::Binding);
    }
    check_source_inputs(&history, builder, inputs)?;
    if !lowering.is_for(history.origin.payload().source.budget()) {
        return Err(Error::Binding);
    }
    let bootstrap = if history.resources.outputs == 1 {
        0
    } else {
        history
            .origin
            .payload()
            .source
            .lower_bootstrap_bytes(limits)
            .map_err(|_| Error::Bounds)?
    };
    let control = add(bootstrap, size_of::<OwnedExternalCharge>())?
        .checked_mul(history.resources.outputs - 1)
        .ok_or(Error::Bounds)?;
    if lowering.payload_bytes() < control {
        return Err(Error::Bounds);
    }
    let rows = history.resources.rows;
    let outputs = history.resources.outputs;
    let mut sink = BoundExpanded {
        builder,
        inputs,
        sources,
        history: &mut history,
    };
    let result = request_driver::drive_with(
        &mut sink,
        rows,
        outputs,
        &mut tokens,
        site,
        |sink, selection| {
            check_origin(sink.history, sink.builder)?;
            if selection != sink.history.fragments.len() {
                return Err(Error::Binding);
            }
            if selection != 0 {
                let workload = pllm_models::DecoderWorkload {
                    batch: 1,
                    max_input_tokens: u64::try_from(add(rows, selection)?)
                        .map_err(|_| Error::Bounds)?,
                    max_new_tokens: u64::try_from(outputs - selection)
                        .map_err(|_| Error::Bounds)?,
                };
                let required = sink
                    .history
                    .origin
                    .payload()
                    .source
                    .lower_bootstrap_bytes(limits)
                    .map_err(|_| Error::Bounds)?;
                if required > bootstrap {
                    return Err(Error::Bounds);
                }
                let guard = lowering.split_owned(bootstrap).map_err(|_| Error::Bounds)?;
                let plan = sink
                    .history
                    .origin
                    .payload()
                    .source
                    .lower(guard, workload, limits)
                    .map_err(|_| Error::Bounds)?;
                if plan.plan().prefill.operations.len() > sink.history.resources.max_operations
                    || !plan.same_origin(&sink.history.origin.payload().source)
                {
                    return Err(Error::Binding);
                }
                sink.history.epochs.push(Arc::new(plan));
            }
            sink.history.prepay_epoch(selection)?;
            Ok(request_driver::Plan::Original(Arc::clone(
                &sink.history.epochs[selection],
            )))
        },
    );
    sink.builder.request_emission = None;
    let selected = result?;
    history.check_tokens(&tokens)?;
    Ok(PendingRequest {
        selected,
        tokens,
        history,
    })
}

struct BoundExpanded<'a, 's> {
    builder: &'a mut Builder,
    inputs: &'a BTreeMap<String, SourceView>,
    sources: &'a BTreeMap<String, SourceTensor>,
    history: &'a mut History<'s>,
}
impl request_driver::RequestSink for BoundExpanded<'_, '_> {
    type View = View;
    type Selected = Vec<(usize, usize)>;
    fn selected(&mut self) -> Result<Self::Selected, Error> {
        reserve(self.history.resources.outputs + 1)
    }
    fn lanes(view: &View) -> usize {
        view.words.len()
    }
    fn prefill(
        &mut self,
        plan: &DecoderPlan,
        tokens: &View,
        position: usize,
        state: &mut BTreeMap<(u64, pllm_models::StateKind), View>,
        site: &mut AdmissionSite,
    ) -> Result<View, Error> {
        let epoch = self.history.fragments.len();
        self.history.check_tokens(tokens)?;
        if position != 0
            || !state.is_empty()
            || self.builder.validity != self.history.sticky
            || !std::ptr::eq(plan, self.history.epochs[epoch].plan())
            || self.builder.request_emission.is_some()
        {
            return Err(Error::Binding);
        }
        let mut input_words = std::mem::take(&mut self.history.scratch);
        input_words.clear_preserving_capacity();
        input_words.extend(tokens.words.iter());
        self.builder.request_emission = Some(Active {
            origin: Arc::clone(&self.history.origin),
            plan: Arc::clone(&self.history.epochs[epoch]),
            epoch,
            sticky_in: self.history.sticky,
            selected: None,
            feedback: None,
            input_words,
            projections: std::mem::take(&mut self.history.projections),
            rotary: None,
            produced_charge: None,
            creation_charge: Some(
                self.history
                    .charge
                    .split_owned(0)
                    .map_err(|_| Error::Bounds)?,
            ),
            source_records: std::ptr::from_ref(self.sources) as usize,
            registry_identity: self.history.sources.registry.instance(),
            snapshot_remaining: source_bias::snapshot_bytes(&plan.prefill)?,
            committed: false,
        });
        let vocabulary = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::OutputHead)
            .and_then(|op| op.output_shape.last())
            .copied()
            .ok_or(Error::Binding)?;
        let mut guard = EmissionGuard {
            builder: &mut *self.builder,
            keep: false,
        };
        let result = guard.builder.graph(
            &plan.prefill,
            tokens,
            self.inputs,
            self.sources,
            usize::try_from(vocabulary).map_err(|_| Error::Bounds)?,
            position,
            state,
            site,
        );
        guard.keep = result.is_ok();
        result
    }
    fn append_feedback(
        &mut self,
        tokens: &mut View,
        result: &View,
        selected: &mut Self::Selected,
    ) -> Result<(), Error> {
        let mut active = self.builder.request_emission.take().ok_or(Error::Binding)?;
        let fragment = active.feedback.take().ok_or(Error::Binding)?;
        if result.words.len() != 1
            || result.words.at(0) != fragment.selection.word
            || fragment.epoch != self.history.fragments.len()
            || self.builder.validity != fragment.sticky_out
        {
            return Err(Error::Binding);
        }
        if selected.capacity() != self.history.resources.outputs + 1 {
            return Err(Error::Bounds);
        }
        selected.push(fragment.selection.word);
        tokens.words.push(fragment.selection.word);
        tokens.shape[1] = tokens.words.len();
        self.history.sticky = fragment.sticky_out;
        self.history.fragments.push(fragment);
        self.history.scratch = std::mem::take(&mut active.input_words);
        self.history.projections = std::mem::take(&mut active.projections);
        self.history.check_tokens(tokens)?;
        active.committed = true;
        Ok(())
    }
    #[cfg(test)]
    fn boundary(&self, selection: usize, id: &str, view: &View) {
        if super::super::runtime::oracle_trace::enabled() {
            super::super::runtime::oracle_trace::boundary(
                selection,
                id,
                &view.shape,
                view.words.iter().collect(),
            );
        }
    }
}

/// Called only by the existing checked acceptance module, consuming all pending
/// history. No arbitrary selected vector or sticky scalar can substitute.
pub(super) fn accept<'s>(
    builder: &mut Builder,
    mut pending: PendingRequest<'s>,
) -> Result<LogicalComplete<'s>, Error> {
    check_origin(&pending.history, builder)?;
    pending.history.check_tokens(&pending.tokens)?;
    let n = pending.history.resources.outputs;
    if pending.history.fragments.len() != n
        || pending.selected.len() != n
        || pending.history.epochs.len() != n
        || builder.validity != pending.history.sticky
        || builder.checked_bound.is_none()
        || builder.request_emission.is_some()
    {
        return Err(Error::Binding);
    }
    for (word, fragment) in pending.selected.iter().zip(&pending.history.fragments) {
        if *word != fragment.selection.word {
            return Err(Error::Binding);
        }
    }
    let input = pending.history.origin.payload().input.owner();
    if pending.selected.capacity() < n + 1 {
        return Err(Error::Bounds);
    }
    let (masked, valid) = builder.acceptance_nodes(input, n, &pending.selected)?;
    pending.selected.clear();
    pending.selected.extend((0..n).map(|lane| (masked, lane)));
    pending.selected.push((valid, 0));
    let accepted = [
        (masked, Record::capture(&builder.nodes[masked])?),
        (valid, Record::capture(&builder.nodes[valid])?),
    ];
    pending
        .history
        .origin
        .payload()
        .state
        .compare_exchange(OPEN, COMPLETE, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| Error::Binding)?;
    let outputs = std::mem::take(&mut pending.selected);
    Ok(LogicalComplete {
        outputs,
        pending,
        accepted,
        nodes: builder.nodes.len(),
    })
}

#[cfg(test)]
thread_local! { static MINTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
thread_local! { static SELECTION_FAULT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
#[path = "request_origin_tests.rs"]
mod tests;
