//! Descriptive checkpoint commitments, not graph validation or execution authority.
//!
//! Quote before constructing caller storage. `check` never allocates, copies a
//! plan/root, validates a graph, or authenticates source provenance. A future
//! source owner must prepay and bind this result itself. Shapes follow
//! runtime/semantic_tensors.py, including independent phase vocabularies.
//! Unknown attribute keys fail closed; recognized non-artifact metadata is only
//! censused, NOT semantically validated. Shared embedding and centered affine
//! LayerNorm tensor contracts also serve native graph validation.

use crate::{DecoderPlan, ModelOperation, ModelOperator};
use crate::transformer_blocks::{self, EmbeddingDomain, EmbeddingError};
use pllm_types::bounded_canonical::borrowed::{self as json, NativeFrame, Traversal};
use serde_json::Value;
use std::{cmp::Ordering, fmt, mem::size_of};

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Totals across both phases (not per graph).
    pub max_operations: usize,
    pub max_inputs: usize,
    pub max_dimensions: usize,
    pub max_rank: usize,
    pub max_string_bytes: usize,
    pub max_attribute_nodes: usize,
    pub max_attribute_bytes: usize,
    /// One-based native JSON depth; caller supplies this many frames.
    pub max_attribute_depth: usize,
    pub max_work: usize,
    pub max_scratch_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Overflow, Operations, Inputs, Dimensions, Rank, String, Attributes,
    Work, Scratch, DuplicateId, Head, TokenSource, ProjectionSource,
    Shape, Artifact, Conflict, UnsupportedOperator, UnknownAttribute,
    Contract, Json(json::Error),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Overflow => "checkpoint requirement arithmetic overflow",
            Self::Operations => "checkpoint operation limit",
            Self::Inputs => "checkpoint input limit",
            Self::Dimensions => "checkpoint dimension limit",
            Self::Rank => "checkpoint rank limit",
            Self::String => "checkpoint string limit",
            Self::Attributes => "checkpoint attribute limit",
            Self::Work => "checkpoint work limit",
            Self::Scratch => "checkpoint scratch limit",
            Self::DuplicateId => "duplicate phase operation identity",
            Self::Head => "checkpoint needs one head per phase",
            Self::TokenSource => "embedding lookup input/domain invalid",
            Self::ProjectionSource => "projection needs one operation input",
            Self::Shape => "checkpoint dimensions must be positive",
            Self::Artifact => "checkpoint artifact identity invalid",
            Self::Conflict => "checkpoint artifact shapes conflict",
            Self::UnsupportedOperator => "checkpoint operator unsupported",
            Self::UnknownAttribute => "checkpoint attribute unsupported",
            Self::Contract => "checkpoint local tensor contract unsupported",
            Self::Json(_) => "checkpoint attribute census refused",
        })
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy)]
enum Contract { Free, Projection, Norm, LayerNorm, GatedNorm, Decay, Scalar, Convolution }
macro_rules! contracts {
    ($($op:ident => $kind:ident / $count:literal),+ $(,)?) => {
        fn contract(op: ModelOperator) -> Contract {
            match op { $(ModelOperator::$op => Contract::$kind),+ }
        }
        /// Derived from the same exhaustive operator table as classification.
        pub const MAX_REQUIREMENTS_PER_OPERATION: usize = {
            let counts: &[usize] = &[$($count),+];
            let mut i = 0;
            let mut maximum = 0;
            while i < counts.len() {
                if counts[i] > maximum { maximum = counts[i]; }
                i += 1;
            }
            maximum
        };
    }
}
contracts! {
    TokenLookup => Projection / 2, Reshape => Free / 0,
    RmsNorm => Norm / 1, Linear => Projection / 2,
    RotaryEmbedding => Free / 0, KvCacheAppend => Free / 0,
    AttentionScores => Free / 0, AttentionScale => Free / 0,
    CausalMask => Free / 0, Softmax => Free / 0,
    AttentionValues => Free / 0, ResidualAdd => Free / 0,
    Silu => Free / 0, Multiply => Free / 0, OutputHead => Projection / 2,
    LastToken => Free / 0, GreedyTokenSelection => Free / 0,
    TokenFeedback => Free / 0, AttentionImportance => Free / 0,
    SecureTopK => Free / 0, SecureGather => Free / 0,
    ClusterBounds => Free / 0, ClusterSimilarity => Free / 0,
    SharedIndices => Free / 0, CacheActiveIndices => Free / 0,
    Scale => Scalar / 1, Slice => Free / 0, GeluTanh => Free / 0,
    Softcap => Free / 0, CacheSuffix => Free / 0,
    StateInitialize => Free / 0, Permute => Free / 0, Sigmoid => Free / 0,
    GatedDeltaDecay => Decay / 2, CausalConvolution => Convolution / 1,
    ConvolutionStateUpdate => Convolution / 1, GatedDeltaRule => Free / 0,
    GatedDeltaStateUpdate => Free / 0, RmsNormGated => GatedNorm / 1,
    LayerNorm => LayerNorm / 2,
    GeluErf => Free / 0, Tanh => Free / 0, AttentionMask => Free / 0,
}

/// Actual supported requirement ranks: vector, matrix, depthwise convolution.
/// Operation ranks are bounded separately and never truncated into this shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape { dimensions: [u64; 3], rank: u8 }
impl Shape {
    fn vector(n: u64) -> Self { Self { dimensions: [n, 0, 0], rank: 1 } }
    fn matrix(a: u64, b: u64) -> Self { Self { dimensions: [a, b, 0], rank: 2 } }
    fn convolution(a: u64, b: u64) -> Self { Self { dimensions: [a, 1, b], rank: 3 } }
    pub fn dimensions(&self) -> &[u64] { &self.dimensions[..usize::from(self.rank)] }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Requirement<'plan> { pub name: &'plan str, pub shape: Shape }
impl Requirement<'_> {
    pub const fn empty() -> Self {
        Self { name: "", shape: Shape { dimensions: [0; 3], rank: 0 } }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct OperationSlot<'plan> { phase: u8, operation: Option<&'plan ModelOperation> }
impl OperationSlot<'_> {
    pub const fn empty() -> Self { Self { phase: 0, operation: None } }
    fn key(&self) -> (u8, &str) {
        (self.phase, self.operation.map_or("", |op| op.id.as_str()))
    }
}
/// Storage belongs to caller before `check`; inactive entries may retain borrows.
pub struct Scratch<'plan, 'scratch> {
    pub operations: &'scratch mut [OperationSlot<'plan>],
    pub requirements: &'scratch mut [Requirement<'plan>],
    pub attributes: &'scratch mut [NativeFrame<'plan>],
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Census {
    pub operations: usize, pub inputs: usize, pub dimensions: usize,
    pub attribute_nodes: usize, pub attribute_bytes: usize,
    pub materialized_records: usize, pub unique_records: usize,
    /// Conservative source-level charged work; not machine instructions/time.
    pub work: usize,
}
#[derive(Debug)]
pub struct RequirementsView<'plan, 'scratch> {
    records: &'scratch [Requirement<'plan>],
    pub prefill_vocabulary: u64,
    pub decode_vocabulary: u64,
    pub census: Census,
}
impl<'plan> RequirementsView<'plan, '_> {
    pub fn records(&self) -> &[Requirement<'plan>] { self.records }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Quote {
    pub operation_slots: usize, pub requirement_slots: usize,
    pub attribute_frame_slots: usize,
    /// Typed caller pools only; excludes plan backing and compiled callframes.
    pub scratch_bytes: usize,
    /// Shared walker includes its concrete control/iterator/formatter envelope.
    pub attribute_workspace_bytes: usize,
    pub work_ceiling: usize,
}
fn add(a: usize, b: usize) -> Result<usize, Error> { a.checked_add(b).ok_or(Error::Overflow) }
fn mul(a: usize, b: usize) -> Result<usize, Error> { a.checked_mul(b).ok_or(Error::Overflow) }
fn log2(n: usize) -> usize {
    let n = n.max(2);
    (usize::BITS - (n - 1).leading_zeros()) as usize
}
/// Comparison envelope H(k) = 4*k*(ceilLog2(max(k,2))+1).
fn heap_envelope(k: usize) -> Result<usize, Error> { mul(mul(4, k)?, add(log2(k), 1)?) }

/// No plan access, scan, allocation or authority. All arithmetic checked.
/// Work counts byte-pair comparisons + termination, walker work, rank/input
/// visits, fills/swaps/census and conservative fixed attribute lookup overhead.
/// Each op pays 512*(a_i+1)*(L+1) after its bounded attribute walk. This
/// envelope covers root/nested factor key classification, fixed lookups and
/// literal comparisons, including bounded B-tree/string comparisons. The
/// cumulative walker limit guarantees sum(a_i) <= A, so sum(a_i+1) <= A+N;
/// A is the global attribute-node cap, N the operation cap, L the global string
/// cap. Walker funding still precedes traversal. No opaque sort is used.
pub fn quote(limits: Limits) -> Result<Quote, Error> {
    let n = limits.max_operations;
    let r = mul(n, MAX_REQUIREMENTS_PER_OPERATION)?;
    let walker = Traversal::resources_for_native_slots(limits.max_attribute_depth)
        .map_err(Error::Json)?;
    let scratch_bytes = add(add(mul(n, size_of::<OperationSlot<'_>>())?,
        mul(r, size_of::<Requirement<'_>>())?)?,
        mul(limits.max_attribute_depth, size_of::<NativeFrame<'_>>())?)?;
    let comparisons = add(add(heap_envelope(n)?, heap_envelope(r)?)?,
        add(mul(n, add(log2(n), 1)?)?, add(n, r)?)?)?;
    let fixed = mul(mul(512, add(limits.max_attribute_nodes, n)?)?,
        add(limits.max_string_bytes, 1)?)?;
    // Native walker transitions/number formatting: source work cap, not CPU.
    let attributes = mul(64, add(add(limits.max_attribute_nodes,
        limits.max_attribute_bytes)?, n)?)?;
    let structural = mul(32, add(add(add(n, r)?, limits.max_inputs)?,
        add(limits.max_dimensions, 1)?)?)?;
    let work_ceiling = add(add(fixed, attributes)?, add(structural,
        mul(comparisons, add(limits.max_string_bytes, 16)?)?)?)?;
    Ok(Quote { operation_slots: n, requirement_slots: r,
        attribute_frame_slots: limits.max_attribute_depth, scratch_bytes,
        attribute_workspace_bytes: walker.logical_workspace_bytes, work_ceiling })
}

struct Meter { work: usize, ceiling: usize }
impl Meter {
    fn charge(&mut self, n: usize) -> Result<(), Error> {
        self.work = add(self.work, n)?;
        if self.work > self.ceiling { return Err(Error::Work); }
        Ok(())
    }
    fn compare(&mut self, a: &str, b: &str) -> Result<Ordering, Error> {
        for (&a, &b) in a.as_bytes().iter().zip(b.as_bytes()) {
            self.charge(1)?;
            match a.cmp(&b) { Ordering::Equal => {}, other => return Ok(other) }
        }
        self.charge(1)?;
        Ok(a.len().cmp(&b.len()))
    }
}
// In-place max-heapsort, no recursion, allocation, std sort or reconstruction.
fn heapsort<T: Copy>(items: &mut [T], meter: &mut Meter,
    compare: fn(&T, &T, &mut Meter) -> Result<Ordering, Error>) -> Result<(), Error> {
    fn sift<T: Copy>(a: &mut [T], mut root: usize, end: usize, m: &mut Meter,
        cmp: fn(&T, &T, &mut Meter) -> Result<Ordering, Error>) -> Result<(), Error> {
        while root < end / 2 {
            let mut child = root * 2 + 1;
            if child + 1 < end && cmp(&a[child], &a[child + 1], m)? == Ordering::Less {
                child += 1;
            }
            if cmp(&a[root], &a[child], m)? != Ordering::Less { break; }
            m.charge(1)?;
            a.swap(root, child);
            root = child;
        }
        Ok(())
    }
    let length = items.len();
    for root in (0..length / 2).rev() { sift(items, root, length, meter, compare)?; }
    for end in (1..length).rev() {
        meter.charge(1)?;
        items.swap(0, end);
        sift(items, 0, end, meter, compare)?;
    }
    Ok(())
}
fn compare_operations(a: &OperationSlot<'_>, b: &OperationSlot<'_>, m: &mut Meter)
    -> Result<Ordering, Error> {
    m.charge(1)?;
    if a.phase != b.phase { return Ok(a.phase.cmp(&b.phase)); }
    m.compare(a.key().1, b.key().1)
}
fn compare_records(a: &Requirement<'_>, b: &Requirement<'_>, m: &mut Meter)
    -> Result<Ordering, Error> { m.compare(a.name, b.name) }
fn width(op: &ModelOperation) -> Result<u64, Error> {
    op.output_shape.last().copied().filter(|n| *n > 0).ok_or(Error::Shape)
}
fn source<'p>(index: &[OperationSlot<'p>], phase: u8, name: &str, meter: &mut Meter)
    -> Result<&'p ModelOperation, Error> {
    let (mut low, mut high) = (0, index.len());
    while low < high {
        let mid = low + (high - low) / 2;
        meter.charge(1)?;
        let order = if index[mid].phase == phase {
            meter.compare(index[mid].key().1, name)?
        } else { index[mid].phase.cmp(&phase) };
        match order {
            Ordering::Less => low = mid + 1,
            Ordering::Greater => high = mid,
            Ordering::Equal => return index[mid].operation.ok_or(Error::ProjectionSource),
        }
    }
    Err(Error::ProjectionSource)
}

fn present(value: Option<&Value>) -> Option<&Value> { value.filter(|v| !v.is_null()) }
fn text(value: Option<&Value>) -> Result<&str, Error> {
    value.and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or(Error::Artifact)
}
fn bind<'p>(out: &mut [Requirement<'p>], used: &mut usize, value: Option<&'p Value>,
    shape: Shape, optional: bool, meter: &mut Meter) -> Result<(), Error> {
    let value = present(value);
    if optional && value.is_none() { return Ok(()); }
    let name = text(value)?;
    meter.charge(1)?;
    let slot = out.get_mut(*used).ok_or(Error::Scratch)?;
    *slot = Requirement { name, shape };
    *used = add(*used, 1)?;
    Ok(())
}

/// Strict keys for artifact-bearing contracts; no hidden artifact declarations.
/// Tensor-free metadata keys are a closed vocabulary, not execution validation.
fn known_key(kind: Contract, key: &str) -> bool {
    let keys: &[&str] = match kind {
        Contract::Projection => &["weight", "bias", "tied", "input_layout", "output_layout",
            "compute_dtype", "output_dtype", "weight_shape", "bias_shape", "weight_layout", "formula",
            "checkpoint_layout", "adapter_reference_revision", "index_domain", "position_policy",
            "out_of_bounds", "semantic_config"],
        Contract::Norm => &["weight", "epsilon", "weight_offset", "with_scale", "output_dtype", "compute_dtype"],
        Contract::LayerNorm => &["weight", "bias", "weight_shape", "bias_shape", "epsilon", "axis",
            "variance", "epsilon_placement", "affine_order", "compute_dtype", "output_dtype"],
        Contract::GatedNorm => &["weight", "epsilon", "activation", "weight_offset", "norm_before_gate"],
        Contract::Decay => &["a_log", "dt_bias", "formula", "compute_dtype"],
        Contract::Scalar => &["factor", "kind", "radicand", "numerator", "denominator",
            "factor_source_dtype", "factor_rounding_dtype", "compute_dtype", "output_dtype"],
        Contract::Convolution => &["weight", "bias", "groups", "kernel_size", "activation", "mode"],
        Contract::Free => &["axis", "layout", "input_layout", "output_layout", "compute_dtype",
            "output_dtype", "group_size", "head_dim", "theta", "rotary_dimensions", "pairing",
            "position_policy", "coefficient_profile", "tail_policy", "frequency_scaling",
            "rope_type", "partial_rotary_factor", "mrope_interleaved", "mrope_section",
            "state", "mode", "state_capacity", "state_layout", "absolute_write_positions_input",
            "padding_mask_input", "valid_lengths_input", "attention_domain", "maximum_sequence",
            "semantics", "kind", "maximum_position_embeddings", "absolute_positions_input",
            "cache_validity", "key_domain", "masked_value", "selection", "policy", "source",
            "target", "tie_break", "start", "end", "squeeze", "formula", "cap", "softcap",
            "initial_value", "dtype", "state_kind", "normalization_epsilon", "query_scale",
            "key_head_repeats", "chunk_size", "state_update", "permutation", "axes",
            "window", "window_size", "sink_tokens", "budget", "k", "indices", "selection_axis",
            "key_value_source_layer", "key_layout", "sliding_window", "left_context", "includes_current",
            "observation_window", "reduction", "keep", "keep_ratio", "index_domain", "output_order",
            "padding", "static_keep", "prefill_maximum_sequence", "candidate_bound", "index_axis",
            "index_unit_size", "output_semantics", "cluster_size", "level", "alpha", "dropout"],
    };
    keys.contains(&key)
}

/// Shared, borrowed per-operation artifact core. Decoder caller retains its
/// independent phase/head/source indexing and meter. Forward caller supplies
/// its own admitted topology. No ModelOperation/Value/shape/name is cloned.
fn artifact_slots(operator:ModelOperator,inputs:&[String],attrs:&Value,input:u64,output:u64)
    ->Result<[(&'static str,Shape,bool);2],Error>{
    match operator {
        ModelOperator::TokenLookup|ModelOperator::Linear|ModelOperator::OutputHead=>{
            let shape=if operator==ModelOperator::TokenLookup {Shape::matrix(input,output)} else {
                let [a,b]=transformer_blocks::projection_dimensions_parts(attrs,input,output).ok_or(Error::Contract)?;
                Shape::matrix(a,b)
            };
            if attrs.get("weight_shape").is_some()&&!transformer_blocks::declared_shape(attrs.get("weight_shape"),shape.dimensions())
                ||attrs.get("bias_shape").is_some()&&!transformer_blocks::declared_shape(attrs.get("bias_shape"),&[output]){
                return Err(Error::Contract);
            }
            Ok([("weight",shape,true),("bias",Shape::vector(output),true)])
        }
        ModelOperator::LayerNorm=>{
            if !transformer_blocks::centered_affine_layer_norm_parts(inputs.len(),attrs,output){return Err(Error::Contract);}
            Ok([("weight",Shape::vector(output),false),("bias",Shape::vector(output),false)])
        }
        _=>Err(Error::UnsupportedOperator),
    }
}

/// Checks limits/work/pool sizes before touching scratch or walking operations.
/// Semantic errors may leave partial scratch; no view is returned on failure.
pub fn check<'plan, 'scratch>(plan: &'plan DecoderPlan, limits: Limits,
    scratch: Scratch<'plan, 'scratch>) -> Result<RequirementsView<'plan, 'scratch>, Error> {
    let q = quote(limits)?;
    if q.work_ceiling > limits.max_work { return Err(Error::Work); }
    if q.scratch_bytes > limits.max_scratch_bytes { return Err(Error::Scratch); }
    // Exact pools: surplus typed storage is not silently omitted from quotation.
    if scratch.operations.len() != q.operation_slots ||
        scratch.requirements.len() != q.requirement_slots ||
        scratch.attributes.len() != q.attribute_frame_slots || q.attribute_frame_slots == 0 {
        return Err(Error::Scratch);
    }
    let n = add(plan.prefill.operations.len(), plan.decode.operations.len())?;
    if n > limits.max_operations { return Err(Error::Operations); }
    let mut census = Census::default();
    let mut meter = Meter { work: 0, ceiling: q.work_ceiling };
    meter.charge(32)?;
    // Bounded census precedes index fill. Attribute walker is caller-backed.
    for graph in [&plan.prefill, &plan.decode] {
        for op in &graph.operations {
            meter.charge(32)?;
            if op.id.is_empty() || op.id.len() > limits.max_string_bytes { return Err(Error::String); }
            census.operations = add(census.operations, 1)?;
            census.inputs = add(census.inputs, op.inputs.len())?;
            if census.inputs > limits.max_inputs { return Err(Error::Inputs); }
            census.dimensions = add(census.dimensions, op.output_shape.len())?;
            if census.dimensions > limits.max_dimensions { return Err(Error::Dimensions); }
            if op.output_shape.len() > limits.max_rank { return Err(Error::Rank); }
            for input in &op.inputs {
                meter.charge(1)?;
                if input.len() > limits.max_string_bytes { return Err(Error::String); }
            }
            for dimension in &op.output_shape {
                meter.charge(1)?;
                if *dimension == 0 { return Err(Error::Shape); }
            }
            let kind = contract(op.operator);
            let mut walker = Traversal::new_native(&op.attributes, scratch.attributes, json::Limits {
                max_nodes: limits.max_attribute_nodes - census.attribute_nodes,
                max_bytes: limits.max_attribute_bytes - census.attribute_bytes,
                max_depth: limits.max_attribute_depth,
                max_string_bytes: limits.max_string_bytes, max_key_bytes: limits.max_string_bytes,
                max_work: q.work_ceiling - meter.work,
                max_workspace_bytes: q.attribute_workspace_bytes,
            }).map_err(Error::Json)?;
            while walker.next_event().map_err(Error::Json)?.is_some() {}
            let usage = walker.usage();
            census.attribute_nodes = add(census.attribute_nodes, usage.nodes)?;
            census.attribute_bytes = add(census.attribute_bytes, usage.bytes)?;
            meter.charge(usage.work)?;
            let attrs = op.attributes.as_object().ok_or(Error::Attributes)?;
            meter.charge(mul(mul(512, add(usage.nodes, 1)?)?,
                add(limits.max_string_bytes, 1)?)?)?;
            for key in attrs.keys() {
                if !known_key(kind, key) { return Err(Error::UnknownAttribute); }
            }
        }
    }
    let mut used = 0;
    for (phase, graph) in [(0, &plan.prefill), (1, &plan.decode)] {
        for op in &graph.operations {
            meter.charge(1)?;
            scratch.operations[used] = OperationSlot { phase, operation: Some(op) };
            used += 1;
        }
    }
    let index = &mut scratch.operations[..n];
    heapsort(index, &mut meter, compare_operations)?;
    for pair in index.windows(2) {
        if compare_operations(&pair[0], &pair[1], &mut meter)? == Ordering::Equal {
            return Err(Error::DuplicateId);
        }
    }
    let mut vocabularies = [0_u64; 2];
    let mut heads = [0_usize; 2];
    for entry in index.iter() {
        meter.charge(1)?;
        let op = entry.operation.ok_or(Error::ProjectionSource)?;
        if op.operator == ModelOperator::OutputHead {
            let p = usize::from(entry.phase);
            heads[p] = add(heads[p], 1)?;
            vocabularies[p] = width(op)?;
        }
    }
    if heads != [1, 1] { return Err(Error::Head); }
    let mut records = 0;
    for entry in index.iter() {
        meter.charge(16)?;
        let op = entry.operation.ok_or(Error::ProjectionSource)?;
        if matches!(op.operator, ModelOperator::GeluErf | ModelOperator::Tanh | ModelOperator::AttentionMask) {
            return Err(Error::UnsupportedOperator);
        }
        let attrs = &op.attributes;
        match contract(op.operator) {
            Contract::Projection => {
                let output = width(op)?;
                let input = if op.operator == ModelOperator::TokenLookup {
                    let embedding = transformer_blocks::embedding(op).map_err(|error| match error {
                        EmbeddingError::Source => Error::TokenSource,
                        EmbeddingError::Contract => Error::Contract,
                    })?;
                    match (embedding.domain, embedding.cardinality) {
                        (_, Some(cardinality)) => cardinality,
                        (EmbeddingDomain::Tokens, None) => vocabularies[usize::from(entry.phase)],
                        _ => return Err(Error::Contract),
                    }
                } else {
                    if op.inputs.len() != 1 { return Err(Error::ProjectionSource); }
                    width(source(index, entry.phase, &op.inputs[0], &mut meter)?)?
                };
                for (key,shape,optional) in artifact_slots(op.operator,&op.inputs,attrs,input,output)? {
                    bind(scratch.requirements,&mut records,attrs.get(key),shape,optional,&mut meter)?;
                }
            }
            Contract::LayerNorm => {
                // Twelve allowlist literals and thirteen root lookups (eleven
                // in the shared predicate plus these two binds), including the
                // optional output dtype, remain within the existing prepaid
                // 512*(a_i+1)*(L+1) envelope. Epsilon parsing is the same direct
                // native-FP32 operation as before; no new traversal or tariff.
                let output=op.output_shape.last().copied().unwrap_or(0);
                for (key,shape,optional) in artifact_slots(op.operator,&op.inputs,attrs,0,output)? {
                    bind(scratch.requirements,&mut records,attrs.get(key),shape,optional,&mut meter)?;
                }
            }
            Contract::Norm => {
                if present(attrs.get("weight")).is_none() {
                    if attrs.get("with_scale").and_then(Value::as_bool) != Some(false) ||
                        attrs.get("output_dtype").and_then(Value::as_str) != Some("bfloat16") {
                        return Err(Error::Contract);
                    }
                } else {
                    bind(scratch.requirements, &mut records, attrs.get("weight"), Shape::vector(width(op)?), false, &mut meter)?;
                }
            }
            Contract::GatedNorm => {
                if attrs.get("activation").and_then(Value::as_str) != Some("silu") ||
                    attrs.get("weight_offset").and_then(Value::as_f64) != Some(0.0) ||
                    attrs.get("norm_before_gate").and_then(Value::as_bool) != Some(true) { return Err(Error::Contract); }
                bind(scratch.requirements, &mut records, attrs.get("weight"), Shape::vector(width(op)?), false, &mut meter)?;
            }
            Contract::Decay => {
                if attrs.get("formula").and_then(Value::as_str) != Some("-exp(A_log)*softplus(a+dt_bias)") ||
                    attrs.get("compute_dtype").and_then(Value::as_str) != Some("float32") { return Err(Error::Contract); }
                let shape = Shape::vector(width(op)?);
                bind(scratch.requirements, &mut records, attrs.get("a_log"), shape, false, &mut meter)?;
                bind(scratch.requirements, &mut records, attrs.get("dt_bias"), shape, false, &mut meter)?;
            }
            Contract::Scalar => {
                if let Some(factor) = attrs.get("factor").and_then(Value::as_object) {
                    for key in factor.keys() {
                        if !["kind", "weight", "weight_shape", "numerator", "denominator", "sqrt_denominator"].contains(&key.as_str()) {
                            return Err(Error::UnknownAttribute);
                        }
                    }
                    if factor.get("kind").and_then(Value::as_str) == Some("checkpoint_scalar") {
                        let shape = factor.get("weight_shape").and_then(Value::as_array).ok_or(Error::Contract)?;
                        if shape.len() != 1 || shape[0].as_f64() != Some(1.0) { return Err(Error::Contract); }
                        bind(scratch.requirements, &mut records, factor.get("weight"), Shape::vector(1), false, &mut meter)?;
                    } else if factor.contains_key("weight") { return Err(Error::Contract); }
                }
            }
            Contract::Convolution => {
                let groups = attrs.get("groups").and_then(Value::as_u64).filter(|n| (1..=8192).contains(n)).ok_or(Error::Contract)?;
                let kernel = attrs.get("kernel_size").and_then(Value::as_u64).filter(|n| (1..=16).contains(n)).ok_or(Error::Contract)?;
                if present(attrs.get("bias")).is_some() { return Err(Error::Contract); }
                bind(scratch.requirements, &mut records, attrs.get("weight"), Shape::convolution(groups, kernel), false, &mut meter)?;
            }
            Contract::Free => {},
        }
    }
    census.materialized_records = records;
    let out = &mut scratch.requirements[..records];
    heapsort(out, &mut meter, compare_records)?;
    let mut unique = 0;
    for read in 0..records {
        meter.charge(1)?;
        if unique > 0 && meter.compare(out[unique - 1].name, out[read].name)? == Ordering::Equal {
            if out[unique - 1].shape != out[read].shape { return Err(Error::Conflict); }
        } else {
            out[unique] = out[read];
            unique += 1;
        }
    }
    census.unique_records = unique;
    census.work = meter.work;
    Ok(RequirementsView { records: &out[..unique], prefill_vocabulary: vocabularies[0],
        decode_vocabulary: vocabularies[1], census })
}

// Allocation boundary: the private child owns forward-only result/index storage.
// This parent and shared transformer predicates remain wholly source-guarded;
// the child calls this parent's artifact interpretation, never a parallel copy.
mod forward_owned;
pub(crate) use forward_owned::forward_records;
