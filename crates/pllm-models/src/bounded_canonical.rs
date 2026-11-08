//! Bounded canonical identities for immutable model schemas, not execution admission.
//!
//! No JSON tree, descriptor tree, serializer, output buffer, or heap workspace is
//! constructed. Fixed schema keys are emitted in lexical order; attributes use
//! the shared iterative JSON walker. Each public write/hash first runs the same
//! traversal into `io::sink()`. Admission failures never touch external output.
//! Writer errors during replay can leave partial output.
//!
//! The resource quote describes logical owned Rust state, not total native stack,
//! allocator usage, RSS, or private dependency scratch. Borrowed model/JSON/string
//! payload capacity and external writer backing are excluded. Compiled call frames
//! and numeric library scratch remain explicitly unpriced in `Resources`; this module is not
//! an execution gate, authority decision, or certification of a machine bound.
//! Typed schema calls have fixed nesting and never recurse. Their reusable
//! collection cursors live in `Cursors`; attribute nesting uses only the supplied
//! `StreamFrame` slice. Resource categories quote actual `size_of` types, not an
//! estimated allowance per model node; they are not a machine peak-stack total.
//!
//! Logical slots follow ownership, not ABI temporaries: moving an argument,
//! constructor output, or consumed receiver transfers its slot. A Result includes
//! its payload; it is not added to that payload again. Sequential representations
//! reuse a logical slot (their maximum size), even if compiled stack slots do not.
//! Copy arguments follow the same transfer convention unless the caller retains
//! a separately stored copy. Admission retains its limits and counters; adjusted
//! pass limits transfer into Session's meter. Borrowed aliases and scalar
//! evaluation locals are not owned workspace. JSON numbers are traversed by
//! reference, never copied into an owned Number. Dependency/native call stacks
//! are outside this logical quote, not an unknown component of its total.

use crate::{
    AppliedMethod, DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator,
    StateKind, StateTensor, DECODER_PLAN_SCHEMA_VERSION,
};
use pllm_types::bounded_canonical::{
    DomainHash, Resources, Session, StreamError, StreamFrame, StreamLimits, StreamUsage,
};
use std::{fmt, io, mem::size_of};

/// Collection caps are global across both graphs; rank is per tensor.
/// String, JSON node/depth/output and algorithmic work caps include schema AND
/// attributes, under the exact shared stream accounting contract.
/// Work also includes one unit per graph-width check and per operation/state
/// inspected by collection admission. Returned work is one traversal plus this
/// admission scan, not the sum of both replay passes. Domain hashing is separately
/// bounded by `max_domain_bytes`, outside JSON work.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub canonical: StreamLimits,
    pub max_operations: usize,
    pub max_states: usize,
    pub max_inputs: usize,
    pub max_rank: usize,
    pub max_transformations: usize,
    /// Hash domain bytes are outside JSON counters, so receive a separate cap.
    pub max_domain_bytes: usize,
}

#[derive(Debug)]
pub enum Error {
    Operations,
    States,
    Inputs,
    Rank,
    Transformations,
    Domain,
    Canonical(StreamError),
}

impl From<StreamError> for Error {
    fn from(value: StreamError) -> Self {
        Self::Canonical(value)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canonical(error) => fmt::Display::fmt(error, f),
            error => write!(f, "bounded model canonical limit: {error:?}"),
        }
    }
}
impl std::error::Error for Error {}

/// Concrete logical state categories, intentionally NOT a native-stack total.
#[derive(Clone, Copy, Debug)]
pub struct ResourceQuote {
    pub stream: Resources,
    /// Actual reusable schema collection cursors (item and scalar).
    pub schema_cursor_bytes: usize,
    /// Actual admitted schema view, immutable limits and retained usage.
    pub admission_state_bytes: usize,
    /// Actual length/rank admission accumulator and collection cursors.
    pub collection_state_bytes: usize,
    /// Adjusted stream-limit type size (transfers into Session, not added again).
    pub pass_limits_bytes: usize,
    /// Completed counters returned by the current pass, alongside admission.
    pub pass_usage_bytes: usize,
    /// Complete write phases, each including admission exactly once.
    pub phases: ResourcePhases,
    /// Hash-only phases after admission; shared admission phases are in `phases`.
    pub hash_phases: HashResourcePhases,
    /// Entire supplied frame capacity plus the maximum write phase.
    pub write_workspace_bytes: usize,
    /// Entire supplied frame capacity plus the maximum hash phase (not write replay).
    pub hash_workspace_bytes: usize,
}

/// Full phase slots excluding supplied frame capacity. Collection and sink are
/// sequential; replay retains admission, not the earlier sink session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourcePhases {
    pub admission_construction_bytes: usize,
    pub collection_bytes: usize,
    pub sink_pass_bytes: usize,
    pub replay_bytes: usize,
    pub finish_return_bytes: usize,
}

/// Full hash-only phase slots excluding frames, each retaining admission once.
/// Construction transfers DomainHash's output slot; finish consumes that slot
/// before returning its Result (the larger representation is charged).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HashResourcePhases {
    pub construction_bytes: usize,
    pub replay_bytes: usize,
    pub finish_return_bytes: usize,
}

fn sum(parts: &[usize]) -> Result<usize, Error> {
    parts.iter().try_fold(0usize, |total, part| {
        total
            .checked_add(*part)
            .ok_or(Error::Canonical(StreamError::StackLimit))
    })
}

fn peak(parts: &[usize]) -> usize {
    parts.iter().copied().max().unwrap_or(0)
}

// These are actual pass slots, not allowances for unspecified call frames.
struct PassConstruction<'a, 'w, W> {
    session: Result<Session<'a, 'w, W>, StreamError>,
}

struct PassWorkspace<'a, 'w, W> {
    stream: Session<'a, 'w, W>,
    cursors: Cursors,
    result: Result<(), StreamError>,
}

struct PassFinish {
    result: Result<StreamUsage, StreamError>,
}

struct CollectionWorkspace {
    counts: Collections,
    result: Result<usize, Error>,
}

struct HashConstruction {
    output: DomainHash,
}

struct HashFinish {
    result: Result<[u8; 32], Error>,
}

fn pass_workspace_bytes<W: io::Write>() -> Result<usize, Error> {
    let stream = Session::<W>::resources_for_slots(0)?;
    // String escape and numeric formatting are sequential. No owned Number:
    // stream.numeric_value_bytes is a diagnostic type size, not live storage.
    let formatting = peak(&[
        stream.escape_buffer_bytes,
        sum(&[
            stream.format_arguments_bytes,
            stream.compact_formatter_bytes,
        ])?,
    ]);
    Ok(peak(&[
        size_of::<PassConstruction<'_, '_, W>>(),
        sum(&[
            size_of::<PassWorkspace<'_, '_, W>>(),
            stream.emitter_bytes,
            formatting,
        ])?,
        size_of::<Session<'_, '_, W>>(),
        size_of::<PassFinish>(),
        size_of::<Result<StreamUsage, Error>>(),
    ]))
}

/// Quotes supplied stack capacity, even unused slots. Writer-owned allocations
/// and machine call frames are not included. No allocation is performed.
/// Legacy category fields are diagnostics, not quantities to sum: `stream`
/// includes type sizes (notably Number and hash) that need not be live in a pass.
/// Overflow of either composed total is `Canonical(StackLimit)`.
///
/// Write = frames + max(admission construction, collection, sink, replay, return).
/// Hash = frames + max(admission construction, collection, sink, hash construction,
/// hash replay, hash finish/return). No hash exists during admission. Constructor
/// output Results include their payload slots; successful moves reuse those slots.
pub fn resources<W: io::Write>(slots: usize) -> Result<ResourceQuote, Error> {
    let stream = Session::<W>::resources_for_slots(slots)?;
    let admission = size_of::<Admission<'_>>();
    let phases = ResourcePhases {
        admission_construction_bytes: size_of::<Result<Admission<'_>, Error>>(),
        collection_bytes: sum(&[admission, size_of::<CollectionWorkspace>()])?,
        sink_pass_bytes: sum(&[admission, pass_workspace_bytes::<io::Sink>()?])?,
        replay_bytes: sum(&[admission, pass_workspace_bytes::<W>()?])?,
        finish_return_bytes: sum(&[admission, size_of::<Result<StreamUsage, Error>>()])?,
    };
    let hash_phases = HashResourcePhases {
        construction_bytes: sum(&[admission, size_of::<HashConstruction>()])?,
        replay_bytes: sum(&[
            admission,
            size_of::<DomainHash>(),
            pass_workspace_bytes::<DomainHash>()?,
        ])?,
        finish_return_bytes: sum(&[
            admission,
            peak(&[size_of::<DomainHash>(), size_of::<HashFinish>()]),
        ])?,
    };
    let admission_peak = peak(&[
        phases.admission_construction_bytes,
        phases.collection_bytes,
        phases.sink_pass_bytes,
    ]);
    let write_workspace_bytes = sum(&[
        stream.stack_bytes,
        peak(&[
            admission_peak,
            phases.replay_bytes,
            phases.finish_return_bytes,
        ]),
    ])?;
    let hash_workspace_bytes = sum(&[
        stream.stack_bytes,
        peak(&[
            admission_peak,
            hash_phases.construction_bytes,
            hash_phases.replay_bytes,
            hash_phases.finish_return_bytes,
        ]),
    ])?;
    Ok(ResourceQuote {
        stream,
        schema_cursor_bytes: size_of::<Cursors>(),
        admission_state_bytes: size_of::<Admission<'_>>(),
        collection_state_bytes: size_of::<Collections>(),
        pass_limits_bytes: size_of::<StreamLimits>(),
        pass_usage_bytes: size_of::<StreamUsage>(),
        phases,
        hash_phases,
        write_workspace_bytes,
        hash_workspace_bytes,
    })
}

#[derive(Clone, Copy)]
enum Schema<'a> {
    Plan(&'a DecoderPlan),
    Operation(&'a ModelOperation),
}

struct Admission<'a> {
    schema: Schema<'a>,
    limits: Limits,
    collection_work: usize,
    usage: StreamUsage,
}

impl<'a> Admission<'a> {
    fn new(
        schema: Schema<'a>,
        limits: Limits,
        stack: &mut [StreamFrame<'a>],
    ) -> Result<Self, Error> {
        let mut admission = Self {
            schema,
            limits,
            collection_work: 0,
            usage: StreamUsage::default(),
        };
        admission.collection_work = {
            let mut collection = CollectionWorkspace {
                counts: Collections::default(),
                result: Ok(0),
            };
            collection.result = admission
                .schema
                .check(&admission.limits, &mut collection.counts);
            collection.result?
        };
        // Collection and sink pass do not overlap, even as logical slots.
        admission.usage = pass(&admission, stack, &mut io::sink())?;
        Ok(admission)
    }
}

#[derive(Default)]
struct Collections {
    operations: usize,
    states: usize,
    inputs: usize,
    graph: usize,
    item: usize,
    side: usize,
    work: usize,
}

impl Collections {
    fn step(&mut self, limits: &Limits) -> Result<(), Error> {
        add(
            &mut self.work,
            1,
            limits.canonical.max_work,
            Error::Canonical(StreamError::WorkLimit),
        )
    }
}

#[derive(Default)]
struct Cursors {
    item: usize,
    scalar: usize,
}

fn add(current: &mut usize, amount: usize, limit: usize, error: Error) -> Result<(), Error> {
    *current = current
        .checked_add(amount)
        .filter(|n| *n <= limit)
        .ok_or(error)?;
    Ok(())
}

impl Schema<'_> {
    /// Inspect lengths before any attribute traversal. No strings or JSON scanned.
    fn check(&self, limits: &Limits, counts: &mut Collections) -> Result<usize, Error> {
        match self {
            Self::Operation(operation) => {
                check_operation(operation, limits, counts)?;
            }
            Self::Plan(plan) => {
                if plan.transformations.len() > limits.max_transformations {
                    return Err(Error::Transformations);
                }
                // Reject global graph widths before visiting even the first item.
                while counts.graph < 2 {
                    counts.step(limits)?;
                    let graph = if counts.graph == 0 {
                        &plan.decode
                    } else {
                        &plan.prefill
                    };
                    add(
                        &mut counts.operations,
                        graph.operations.len(),
                        limits.max_operations,
                        Error::Operations,
                    )?;
                    add(
                        &mut counts.states,
                        graph.state_inputs.len(),
                        limits.max_states,
                        Error::States,
                    )?;
                    add(
                        &mut counts.states,
                        graph.state_outputs.len(),
                        limits.max_states,
                        Error::States,
                    )?;
                    counts.graph += 1;
                }
                counts.graph = 0;
                while counts.graph < 2 {
                    let graph = if counts.graph == 0 {
                        &plan.decode
                    } else {
                        &plan.prefill
                    };
                    counts.item = 0;
                    while counts.item < graph.operations.len() {
                        check_operation_fields(&graph.operations[counts.item], limits, counts)?;
                        counts.item += 1;
                    }
                    counts.side = 0;
                    while counts.side < 2 {
                        let states = if counts.side == 0 {
                            &graph.state_inputs
                        } else {
                            &graph.state_outputs
                        };
                        counts.item = 0;
                        while counts.item < states.len() {
                            counts.step(limits)?;
                            if states[counts.item].shape.len() > limits.max_rank {
                                return Err(Error::Rank);
                            }
                            counts.item += 1;
                        }
                        counts.side += 1;
                    }
                    counts.graph += 1;
                }
            }
        }
        Ok(counts.work)
    }
}

fn check_operation(
    operation: &ModelOperation,
    limits: &Limits,
    counts: &mut Collections,
) -> Result<(), Error> {
    add(
        &mut counts.operations,
        1,
        limits.max_operations,
        Error::Operations,
    )?;
    check_operation_fields(operation, limits, counts)
}

fn check_operation_fields(
    operation: &ModelOperation,
    limits: &Limits,
    counts: &mut Collections,
) -> Result<(), Error> {
    counts.step(limits)?;
    add(
        &mut counts.inputs,
        operation.inputs.len(),
        limits.max_inputs,
        Error::Inputs,
    )?;
    if operation.output_shape.len() > limits.max_rank {
        return Err(Error::Rank);
    }
    Ok(())
}

fn pass<'a, W: io::Write>(
    admission: &Admission<'a>,
    stack: &mut [StreamFrame<'a>],
    output: &mut W,
) -> Result<StreamUsage, Error> {
    let session = {
        let mut canonical = admission.limits.canonical;
        canonical.max_work -= admission.collection_work; // Already admitted.
        let construction = PassConstruction {
            session: Session::new(output, stack, canonical),
        };
        construction.session
    };
    let mut workspace = PassWorkspace {
        stream: session?,
        cursors: Cursors::default(),
        result: Ok(()),
    };
    workspace.result = match admission.schema {
        Schema::Plan(plan) => emit_plan(&mut workspace.stream, plan, &mut workspace.cursors),
        Schema::Operation(operation) => {
            emit_operation(&mut workspace.stream, operation, &mut workspace.cursors)
        }
    };
    let PassWorkspace { stream, result, .. } = workspace;
    result?;
    let finish = PassFinish {
        result: stream.finish(),
    };
    let mut usage = finish.result?;
    usage.work += admission.collection_work; // Cannot overflow: stream used the remainder.
    Ok(usage)
}

fn write<'a, W: io::Write>(
    schema: Schema<'a>,
    limits: Limits,
    stack: &mut [StreamFrame<'a>],
    output: &mut W,
) -> Result<StreamUsage, Error> {
    let admission = Admission::new(schema, limits, stack)?;
    pass(&admission, stack, output)?;
    Ok(admission.usage)
}

fn hash<'a>(
    schema: Schema<'a>,
    domain: &str,
    limits: Limits,
    stack: &mut [StreamFrame<'a>],
) -> Result<[u8; 32], Error> {
    if domain.len() > limits.max_domain_bytes {
        return Err(Error::Domain);
    }
    let admission = Admission::new(schema, limits, stack)?;
    let mut construction = HashConstruction {
        output: DomainHash::new(domain),
    };
    pass(&admission, stack, &mut construction.output)?;
    let finish = HashFinish {
        result: Ok(construction.output.finish()),
    };
    finish.result
}

pub fn bounded_plan_digest<'a>(
    plan: &'a DecoderPlan,
    limits: Limits,
    stack: &mut [StreamFrame<'a>],
) -> Result<[u8; 32], Error> {
    hash(
        Schema::Plan(plan),
        DECODER_PLAN_SCHEMA_VERSION,
        limits,
        stack,
    )
}

/// Caller supplies existing operation identity domain; no new domain is invented.
pub fn bounded_model_operation_digest<'a>(
    operation: &'a ModelOperation,
    domain: &str,
    limits: Limits,
    stack: &mut [StreamFrame<'a>],
) -> Result<[u8; 32], Error> {
    hash(Schema::Operation(operation), domain, limits, stack)
}

pub fn bounded_plan_write<'a, W: io::Write>(
    plan: &'a DecoderPlan,
    limits: Limits,
    stack: &mut [StreamFrame<'a>],
    output: &mut W,
) -> Result<StreamUsage, Error> {
    write(Schema::Plan(plan), limits, stack, output)
}

pub fn bounded_model_operation_write<'a, W: io::Write>(
    operation: &'a ModelOperation,
    limits: Limits,
    stack: &mut [StreamFrame<'a>],
    output: &mut W,
) -> Result<StreamUsage, Error> {
    write(Schema::Operation(operation), limits, stack, output)
}

fn optional_u64<W: io::Write>(
    s: &mut Session<'_, '_, W>,
    value: Option<u64>,
) -> Result<(), StreamError> {
    match value {
        Some(value) => s.u64(value),
        None => s.null(),
    }
}

fn shape<W: io::Write>(
    s: &mut Session<'_, '_, W>,
    values: &[u64],
    c: &mut Cursors,
) -> Result<(), StreamError> {
    s.begin_array()?;
    c.scalar = 0;
    while c.scalar < values.len() {
        s.u64(values[c.scalar])?;
        c.scalar += 1;
    }
    s.end_array()
}

fn emit_operation<'a, W: io::Write>(
    s: &mut Session<'a, '_, W>,
    op: &'a ModelOperation,
    c: &mut Cursors,
) -> Result<(), StreamError> {
    s.begin_object()?;
    s.key("attributes")?;
    s.json(&op.attributes)?;
    s.key("id")?;
    s.string(&op.id)?;
    s.key("inputs")?;
    s.begin_array()?;
    c.scalar = 0;
    while c.scalar < op.inputs.len() {
        s.string(&op.inputs[c.scalar])?;
        c.scalar += 1;
    }
    s.end_array()?;
    s.key("layer")?;
    optional_u64(s, op.layer)?;
    s.key("operator")?;
    s.string(operator_name(op.operator))?;
    s.key("output_shape")?;
    shape(s, &op.output_shape, c)?;
    s.key("state_kind")?;
    match op.state_kind {
        Some(kind) => s.string(state_name(kind))?,
        None => s.null()?,
    }
    s.end_object()
}

fn emit_state<W: io::Write>(
    s: &mut Session<'_, '_, W>,
    state: &StateTensor,
    c: &mut Cursors,
) -> Result<(), StreamError> {
    s.begin_object()?;
    s.key("id")?;
    s.string(&state.id)?;
    s.key("kind")?;
    s.string(state_name(state.kind))?;
    s.key("layer")?;
    optional_u64(s, state.layer)?;
    s.key("maximum_sequence")?;
    s.u64(state.maximum_sequence)?;
    s.key("shape")?;
    shape(s, &state.shape, c)?;
    s.end_object()
}

fn emit_graph<'a, W: io::Write>(
    s: &mut Session<'a, '_, W>,
    graph: &'a DecoderGraph,
    c: &mut Cursors,
) -> Result<(), StreamError> {
    s.begin_object()?;
    s.key("batch")?;
    s.u64(graph.batch)?;
    s.key("maximum_key_sequence")?;
    s.u64(graph.maximum_key_sequence)?;
    s.key("mode")?;
    s.string(match graph.mode {
        DecoderMode::Prefill => "prefill",
        DecoderMode::Decode => "decode",
    })?;
    s.key("operations")?;
    s.begin_array()?;
    c.item = 0;
    while c.item < graph.operations.len() {
        emit_operation(s, &graph.operations[c.item], c)?;
        c.item += 1;
    }
    s.end_array()?;
    s.key("output")?;
    s.string(&graph.output)?;
    s.key("query_sequence")?;
    s.u64(graph.query_sequence)?;
    s.key("state_inputs")?;
    s.begin_array()?;
    c.item = 0;
    while c.item < graph.state_inputs.len() {
        emit_state(s, &graph.state_inputs[c.item], c)?;
        c.item += 1;
    }
    s.end_array()?;
    s.key("state_outputs")?;
    s.begin_array()?;
    c.item = 0;
    while c.item < graph.state_outputs.len() {
        emit_state(s, &graph.state_outputs[c.item], c)?;
        c.item += 1;
    }
    s.end_array()?;
    s.end_object()
}

fn emit_method<W: io::Write>(
    s: &mut Session<'_, '_, W>,
    method: &AppliedMethod,
) -> Result<(), StreamError> {
    s.begin_object()?;
    s.key("component")?;
    s.string(&method.component)?;
    s.key("configuration_digest")?;
    s.string(method.configuration_digest.as_str())?;
    s.key("implementation")?;
    s.string(&method.implementation)?;
    s.key("input_digest")?;
    s.string(method.input_digest.as_str())?;
    s.key("method_id")?;
    s.string(&method.method_id)?;
    s.end_object()
}

fn emit_plan<'a, W: io::Write>(
    s: &mut Session<'a, '_, W>,
    plan: &'a DecoderPlan,
    c: &mut Cursors,
) -> Result<(), StreamError> {
    s.begin_object()?;
    s.key("adapter")?;
    s.string(&plan.adapter)?;
    s.key("config_digest")?;
    s.string(plan.config_digest.as_str())?;
    s.key("decode")?;
    emit_graph(s, &plan.decode, c)?;
    s.key("model_family")?;
    s.string(&plan.model_family)?;
    s.key("prefill")?;
    emit_graph(s, &plan.prefill, c)?;
    s.key("schema_version")?;
    s.string(&plan.schema_version)?;
    s.key("token_feedback")?;
    s.bool(plan.token_feedback)?;
    s.key("transformations")?;
    s.begin_array()?;
    c.item = 0;
    while c.item < plan.transformations.len() {
        emit_method(s, &plan.transformations[c.item])?;
        c.item += 1;
    }
    s.end_array()?;
    s.end_object()
}

fn state_name(kind: StateKind) -> &'static str {
    match kind {
        StateKind::Key => "key",
        StateKind::Value => "value",
        StateKind::CacheIndices => "cache_indices",
        StateKind::Recurrent => "recurrent",
        StateKind::Convolution => "convolution",
    }
}

fn operator_name(operator: ModelOperator) -> &'static str {
    match operator {
        ModelOperator::TokenLookup => "token_lookup",
        ModelOperator::Reshape => "reshape",
        ModelOperator::RmsNorm => "rms_norm",
        ModelOperator::Linear => "linear",
        ModelOperator::RotaryEmbedding => "rotary_embedding",
        ModelOperator::KvCacheAppend => "kv_cache_append",
        ModelOperator::AttentionScores => "attention_scores",
        ModelOperator::AttentionScale => "attention_scale",
        ModelOperator::CausalMask => "causal_mask",
        ModelOperator::Softmax => "softmax",
        ModelOperator::AttentionValues => "attention_values",
        ModelOperator::ResidualAdd => "residual_add",
        ModelOperator::Silu => "silu",
        ModelOperator::Multiply => "multiply",
        ModelOperator::OutputHead => "output_head",
        ModelOperator::LastToken => "last_token",
        ModelOperator::GreedyTokenSelection => "greedy_token_selection",
        ModelOperator::TokenFeedback => "token_feedback",
        ModelOperator::AttentionImportance => "attention_importance",
        ModelOperator::SecureTopK => "secure_top_k",
        ModelOperator::SecureGather => "secure_gather",
        ModelOperator::ClusterBounds => "cluster_bounds",
        ModelOperator::ClusterSimilarity => "cluster_similarity",
        ModelOperator::SharedIndices => "shared_indices",
        ModelOperator::CacheActiveIndices => "cache_active_indices",
        ModelOperator::Scale => "scale",
        ModelOperator::Slice => "slice",
        ModelOperator::GeluTanh => "gelu_tanh",
        ModelOperator::Softcap => "softcap",
        ModelOperator::CacheSuffix => "cache_suffix",
        ModelOperator::StateInitialize => "state_initialize",
        ModelOperator::Permute => "permute",
        ModelOperator::Sigmoid => "sigmoid",
        ModelOperator::GatedDeltaDecay => "gated_delta_decay",
        ModelOperator::CausalConvolution => "causal_convolution",
        ModelOperator::ConvolutionStateUpdate => "convolution_state_update",
        ModelOperator::GatedDeltaRule => "gated_delta_rule",
        ModelOperator::GatedDeltaStateUpdate => "gated_delta_state_update",
        ModelOperator::RmsNormGated => "rms_norm_gated",
        ModelOperator::LayerNorm => "layer_norm",
        ModelOperator::GeluErf => "gelu_erf",
        ModelOperator::Tanh => "tanh",
        ModelOperator::AttentionMask => "attention_mask",
    }
}

#[cfg(test)]
#[path = "bounded_canonical_tests.rs"]
mod tests;
