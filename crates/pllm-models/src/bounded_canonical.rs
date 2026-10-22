//! Bounded canonical identities for immutable model schemas, not execution admission.
//!
//! No JSON tree, descriptor tree, serializer, output buffer, or heap workspace is
//! constructed. Fixed schema keys are emitted in lexical order; attributes use
//! the shared iterative JSON walker. Each public write/hash first runs the same
//! traversal into `io::sink()`. Admission failures never touch external output.
//! Writer errors during replay can leave partial output.
//!
//! The resource quote describes logical owned Rust state, not total native stack,
//! allocator usage, RSS, or private dependency scratch. Borrowed model capacity
//! and external writer storage are excluded. Compiled call frames and numeric
//! library scratch remain explicitly unpriced in `Resources`; this module is not
//! an execution gate, authority decision, or certification of a machine bound.
//! Typed schema calls have fixed nesting and never recurse. Their reusable
//! collection cursors live in `Cursors`; attribute nesting uses only the supplied
//! `StreamFrame` slice. Resource categories quote actual `size_of` types, not an
//! estimated allowance per model node; they are not a machine peak-stack total.

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
    /// Actual per-pass adjusted copy of stream limits.
    pub pass_limits_bytes: usize,
    /// Completed counters returned by the current pass, alongside admission.
    pub pass_usage_bytes: usize,
}

/// Quotes supplied stack capacity, even unused slots. Writer-owned allocations
/// and machine call frames are not included. No allocation is performed.
pub fn resources<W: io::Write>(slots: usize) -> Result<ResourceQuote, Error> {
    Ok(ResourceQuote {
        stream: Session::<W>::resources_for_slots(slots)?,
        schema_cursor_bytes: size_of::<Cursors>(),
        admission_state_bytes: size_of::<Admission<'_>>(),
        collection_state_bytes: size_of::<Collections>(),
        pass_limits_bytes: size_of::<StreamLimits>(),
        pass_usage_bytes: size_of::<StreamUsage>(),
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
        admission.collection_work = schema.check(&admission.limits)?;
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
    fn check(self, limits: &Limits) -> Result<usize, Error> {
        let mut counts = Collections::default();
        match self {
            Self::Operation(operation) => {
                check_operation(operation, limits, &mut counts)?;
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
                        check_operation_fields(
                            &graph.operations[counts.item],
                            limits,
                            &mut counts,
                        )?;
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
    let mut canonical = admission.limits.canonical;
    canonical.max_work -= admission.collection_work; // Collection admission checked this bound.
    let mut stream = Session::new(output, stack, canonical)?;
    let mut cursors = Cursors::default();
    match admission.schema {
        Schema::Plan(plan) => emit_plan(&mut stream, plan, &mut cursors)?,
        Schema::Operation(operation) => emit_operation(&mut stream, operation, &mut cursors)?,
    }
    let mut usage = stream.finish()?;
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
    let mut output = DomainHash::new(domain);
    pass(&admission, stack, &mut output)?;
    Ok(output.finish())
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
    }
}

#[cfg(test)]
#[path = "bounded_canonical_tests.rs"]
mod tests;
