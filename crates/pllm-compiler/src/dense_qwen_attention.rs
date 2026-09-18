use std::collections::BTreeMap;

use pllm_types::digest_bytes;
use serde::Serialize;
use zeroize::{Zeroize, Zeroizing};

use crate::provenance_primitives::{
    BoundedKvCacheViewQ10, ProvenanceBoundKvCacheQ10, PROVENANCE_PRIMITIVE_HARD_MAX_CACHE_BYTES,
    PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_BYTES, PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_DEPTH,
    PROVENANCE_PRIMITIVE_HARD_MAX_OPERATIONS, PROVENANCE_PRIMITIVE_HARD_MAX_ROPE_ELEMENTS,
    PROVENANCE_PRIMITIVE_HARD_MAX_STATES, PROVENANCE_PRIMITIVE_HARD_MAX_VIEW_BYTES,
};

use super::{
    append_model_kv_cache_q10, canonical_digest, execute_model_attention_scores_q20,
    execute_model_attention_values_q10, execute_model_kv_cache_view_q10, execute_model_linear,
    execute_model_reshape, execute_model_residual, execute_model_rope_q10,
    execute_model_softmax_q30, execute_q14_to_q10_rescale, execute_rms_norm_q10_direct,
    initialize_model_kv_cache_q10, lower_model_attention_scores_q20_regions,
    lower_model_attention_values_q10_regions, lower_model_kv_cache_append_q10_regions,
    lower_model_kv_cache_view_q10_regions, lower_model_linear_region,
    lower_model_q14_to_q10_rescale_regions, lower_model_reshape_region,
    lower_model_residual_region, lower_model_rope_q10_regions, lower_model_softmax_q30_regions,
    lower_rms_norm_q10_direct_region, model_graph, tensor_elements, DecoderGraph, DecoderMode,
    DecoderPlan, Digest, FixedPointRounding, KvCacheQ10ExecutionInputs,
    ModelAttentionScoresQ20Region, ModelAttentionValuesQ10Region, ModelKvCacheAppendQ10Region,
    ModelKvCacheViewQ10Region, ModelLinearRegion, ModelOperation, ModelOperator,
    ModelReshapeLayout, ModelReshapeRegion, ModelResidualRegion, ModelRmsNormQ10DirectRegion,
    ModelRopeQ10Region, ModelSoftmaxQ30Region, ProvenancePrimitiveResourcePolicy,
    Q14ToQ10RescaleRegion, StateKind,
};

pub const DENSE_QWEN_ATTENTION_BLOCK_SCHEMA_VERSION: &str = "pllm.dense_qwen_attention_block.v1";
pub const DENSE_QWEN_ATTENTION_CLEAR_PROFILE: &str =
    "pllm.clear_exact.dense_qwen_attention.q10_q4.v1";
pub const DENSE_QWEN_ATTENTION_WEIGHT_MANIFEST_SCHEMA_VERSION: &str =
    "pllm.dense_qwen_attention_weight_manifest.v1";
pub const DENSE_QWEN_ATTENTION_HARD_MAX_TOTAL_WEIGHT_BYTES: u64 = 16 * 1024 * 1024 * 1024;
pub const DENSE_QWEN_ATTENTION_HARD_MAX_ACTIVATION_ELEMENTS: u64 = 16 * 1024 * 1024;
const DENSE_QWEN_ATTENTION_WEIGHT_BYTES_DOMAIN: &str = "pllm.dense_qwen_attention.weight_bytes.v1";
const DENSE_QWEN_ATTENTION_WEIGHT_MANIFEST_DOMAIN: &str =
    "pllm.dense_qwen_attention.weight_manifest.v1";
const DENSE_QWEN_ATTENTION_BINDING_DOMAIN: &str = "pllm.dense_qwen_attention.binding.v1";

#[derive(Clone, Copy, Debug)]
pub struct DenseQwenAttentionWeightBytes<'a> {
    pub weight_id: &'a str,
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct DenseQwenAttentionWeights<'a> {
    pub norm_q10: DenseQwenAttentionWeightBytes<'a>,
    pub q_q4: DenseQwenAttentionWeightBytes<'a>,
    pub k_q4: DenseQwenAttentionWeightBytes<'a>,
    pub v_q4: DenseQwenAttentionWeightBytes<'a>,
    pub o_q4: DenseQwenAttentionWeightBytes<'a>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseQwenAttentionWeightManifest {
    schema_version: String,
    entries: BTreeMap<String, Digest>,
}

impl DenseQwenAttentionWeightManifest {
    pub fn from_weights(weights: DenseQwenAttentionWeights<'_>) -> Result<Self, String> {
        let mut entries = BTreeMap::new();
        for weight in [
            weights.norm_q10,
            weights.q_q4,
            weights.k_q4,
            weights.v_q4,
            weights.o_q4,
        ] {
            if weight.weight_id.is_empty() {
                return Err("dense-qwen attention weight ids must be nonempty".into());
            }
            let digest = digest_bytes(DENSE_QWEN_ATTENTION_WEIGHT_BYTES_DOMAIN, weight.bytes);
            if entries
                .insert(weight.weight_id.to_owned(), digest)
                .is_some()
            {
                return Err("dense-qwen attention weight ids must be unique".into());
            }
        }
        if entries.len() != 5 {
            return Err("dense-qwen attention requires exactly five weights".into());
        }
        Ok(Self {
            schema_version: DENSE_QWEN_ATTENTION_WEIGHT_MANIFEST_SCHEMA_VERSION.to_owned(),
            entries,
        })
    }

    pub fn digest(&self) -> Digest {
        canonical_digest(DENSE_QWEN_ATTENTION_WEIGHT_MANIFEST_DOMAIN, &self.entries)
    }

    pub fn entries(&self) -> &BTreeMap<String, Digest> {
        &self.entries
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseQwenAttentionResourcePolicy {
    pub max_total_weight_bytes: u64,
    pub max_activation_elements: u64,
}

impl DenseQwenAttentionResourcePolicy {
    fn validate(&self) -> Result<(), String> {
        if self.max_total_weight_bytes == 0
            || self.max_total_weight_bytes > DENSE_QWEN_ATTENTION_HARD_MAX_TOTAL_WEIGHT_BYTES
        {
            return Err("dense-qwen attention weight-byte cap exceeds hard cap".into());
        }
        if self.max_activation_elements == 0
            || self.max_activation_elements > DENSE_QWEN_ATTENTION_HARD_MAX_ACTIVATION_ELEMENTS
        {
            return Err("dense-qwen attention activation cap exceeds hard cap".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQwenAttentionElementType {
    SignedI16,
    SignedI8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQwenAttentionLayout {
    RowMajorVector,
    RowMajorOutputInput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQwenAttentionRangePolicy {
    RejectNoSaturation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseQwenAttentionWeightArtifact {
    pub weight_id: String,
    pub data_digest: Digest,
    pub shape: Vec<u64>,
    pub element_type: DenseQwenAttentionElementType,
    pub fractional_bits: u8,
    pub layout: DenseQwenAttentionLayout,
    pub rounding: FixedPointRounding,
    pub range_policy: DenseQwenAttentionRangePolicy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseQwenAttentionComposite {
    pub schema_version: String,
    pub numeric_profile: String,
    pub model_plan_digest: Digest,
    pub weight_manifest_digest: Digest,
    pub resource_policy: DenseQwenAttentionResourcePolicy,
    pub mode: DecoderMode,
    pub layer: u64,
    pub input_shape: Vec<u64>,
    pub q_shape: Vec<u64>,
    pub kv_shape: Vec<u64>,
    pub output_shape: Vec<u64>,
    pub semantic_guard_model_family: String,
    pub semantic_guard_adapter: String,
    pub norm: ModelRmsNormQ10DirectRegion,
    pub q: ModelLinearRegion,
    pub k: ModelLinearRegion,
    pub v: ModelLinearRegion,
    pub o: ModelLinearRegion,
    pub q_rescale: Q14ToQ10RescaleRegion,
    pub k_rescale: Q14ToQ10RescaleRegion,
    pub v_rescale: Q14ToQ10RescaleRegion,
    pub o_rescale: Q14ToQ10RescaleRegion,
    pub q_reshape: ModelReshapeRegion,
    pub k_reshape: ModelReshapeRegion,
    pub v_reshape: ModelReshapeRegion,
    pub q_rope: ModelRopeQ10Region,
    pub k_rope: ModelRopeQ10Region,
    pub key_append: ModelKvCacheAppendQ10Region,
    pub value_append: ModelKvCacheAppendQ10Region,
    pub key_view: ModelKvCacheViewQ10Region,
    pub value_view: ModelKvCacheViewQ10Region,
    pub scores: ModelAttentionScoresQ20Region,
    pub softmax: ModelSoftmaxQ30Region,
    pub attention_values: ModelAttentionValuesQ10Region,
    pub attention_hidden_reshape: ModelReshapeRegion,
    pub residual: ModelResidualRegion,
    pub norm_weight: DenseQwenAttentionWeightArtifact,
    pub q_weight: DenseQwenAttentionWeightArtifact,
    pub k_weight: DenseQwenAttentionWeightArtifact,
    pub v_weight: DenseQwenAttentionWeightArtifact,
    pub o_weight: DenseQwenAttentionWeightArtifact,
    pub protected_execution: bool,
    pub complete_decoder: bool,
}

struct AttentionStructure {
    input_norm: ModelOperation,
    q_linear: ModelOperation,
    k_linear: ModelOperation,
    v_linear: ModelOperation,
    o_linear: ModelOperation,
    q_reshape: ModelOperation,
    k_reshape: ModelOperation,
    v_reshape: ModelOperation,
    rope_q: ModelOperation,
    rope_k: ModelOperation,
    key_append: ModelOperation,
    value_append: ModelOperation,
    key_view: ModelOperation,
    value_view: ModelOperation,
    scores: ModelOperation,
    softmax: ModelOperation,
    values: ModelOperation,
    attention_hidden_reshape: ModelOperation,
    residual: ModelOperation,
}

struct AttentionRegions {
    norm: ModelRmsNormQ10DirectRegion,
    q: ModelLinearRegion,
    k: ModelLinearRegion,
    v: ModelLinearRegion,
    o: ModelLinearRegion,
    q_rescale: Q14ToQ10RescaleRegion,
    k_rescale: Q14ToQ10RescaleRegion,
    v_rescale: Q14ToQ10RescaleRegion,
    o_rescale: Q14ToQ10RescaleRegion,
    q_reshape: ModelReshapeRegion,
    k_reshape: ModelReshapeRegion,
    v_reshape: ModelReshapeRegion,
    q_rope: ModelRopeQ10Region,
    k_rope: ModelRopeQ10Region,
    key_append: ModelKvCacheAppendQ10Region,
    value_append: ModelKvCacheAppendQ10Region,
    key_view: ModelKvCacheViewQ10Region,
    value_view: ModelKvCacheViewQ10Region,
    scores: ModelAttentionScoresQ20Region,
    softmax: ModelSoftmaxQ30Region,
    attention_values: ModelAttentionValuesQ10Region,
    attention_hidden_reshape: ModelReshapeRegion,
    residual: ModelResidualRegion,
}

fn operation<'a>(
    operations: &'a BTreeMap<String, &'a ModelOperation>,
    operation_id: &str,
) -> Result<&'a ModelOperation, String> {
    operations.get(operation_id).copied().ok_or_else(|| {
        format!("dense-qwen attention block dependency {operation_id} is not an operation")
    })
}

fn require_operator<'a>(
    operation: &'a ModelOperation,
    expected: ModelOperator,
    role: &str,
) -> Result<&'a ModelOperation, String> {
    if operation.operator != expected {
        return Err(format!(
            "dense-qwen attention block {role} requires {expected:?}, found {:?} for {}",
            operation.operator, operation.id
        ));
    }
    Ok(operation)
}

fn require_layer<'a>(
    operation: &'a ModelOperation,
    layer: u64,
    role: &str,
) -> Result<&'a ModelOperation, String> {
    if operation.layer != Some(layer) {
        return Err(format!(
            "dense-qwen attention block {role} {} is outside layer {layer}",
            operation.id
        ));
    }
    Ok(operation)
}

fn producer<'a>(
    operations: &'a BTreeMap<String, &'a ModelOperation>,
    consumer: &ModelOperation,
    index: usize,
    role: &str,
) -> Result<&'a ModelOperation, String> {
    let input_id = consumer.inputs.get(index).ok_or_else(|| {
        format!(
            "dense-qwen attention block {role} {} is missing input {index}",
            consumer.id
        )
    })?;
    operation(operations, input_id).map_err(|_| {
        format!(
            "dense-qwen attention block {role} {} input {index} must resolve to an operation",
            consumer.id
        )
    })
}

fn tensor_input<'a>(
    operations: &'a BTreeMap<String, &'a ModelOperation>,
    operation: &ModelOperation,
    role: &str,
) -> Result<&'a ModelOperation, String> {
    let mut resolved = operation
        .inputs
        .iter()
        .filter_map(|input_id| operations.get(input_id.as_str()).copied());
    let first = resolved.next().ok_or_else(|| {
        format!(
            "dense-qwen attention block {role} {} has no tensor input",
            operation.id
        )
    })?;
    if resolved.next().is_some() {
        return Err(format!(
            "dense-qwen attention block {role} {} must have exactly one tensor input",
            operation.id
        ));
    }
    Ok(first)
}

fn exactly_one<'a>(
    operations: &'a BTreeMap<String, &'a ModelOperation>,
    layer: u64,
    operator: ModelOperator,
    role: &str,
) -> Result<&'a ModelOperation, String> {
    let mut matches = operations
        .values()
        .copied()
        .filter(|operation| operation.layer == Some(layer) && operation.operator == operator);
    let operation = matches
        .next()
        .ok_or_else(|| format!("dense-qwen attention block requires a {role} in layer {layer}"))?;
    if matches.next().is_some() {
        return Err(format!(
            "dense-qwen attention block requires exactly one {role} in layer {layer}"
        ));
    }
    Ok(operation)
}

fn single_consumer<'a>(
    operations: &'a BTreeMap<String, &'a ModelOperation>,
    producer_id: &str,
    layer: u64,
    operator: ModelOperator,
    role: &str,
) -> Result<&'a ModelOperation, String> {
    let mut matches = operations.values().copied().filter(|operation| {
        operation.layer == Some(layer)
            && operation.operator == operator
            && operation.inputs.iter().any(|input| input == producer_id)
    });
    let operation = matches.next().ok_or_else(|| {
        format!("dense-qwen attention block requires a {role} consuming {producer_id}")
    })?;
    if matches.next().is_some() {
        return Err(format!(
            "dense-qwen attention block requires exactly one {role} consuming {producer_id}"
        ));
    }
    Ok(operation)
}

fn provenance_cap() -> Result<ProvenancePrimitiveResourcePolicy, String> {
    ProvenancePrimitiveResourcePolicy::new(
        PROVENANCE_PRIMITIVE_HARD_MAX_ROPE_ELEMENTS,
        PROVENANCE_PRIMITIVE_HARD_MAX_CACHE_BYTES,
        PROVENANCE_PRIMITIVE_HARD_MAX_VIEW_BYTES,
        PROVENANCE_PRIMITIVE_HARD_MAX_OPERATIONS,
        PROVENANCE_PRIMITIVE_HARD_MAX_STATES,
        PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_BYTES,
        PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_DEPTH,
    )
}

fn lower_structure(
    plan: &DecoderPlan,
    mode: DecoderMode,
    layer: u64,
) -> Result<(AttentionStructure, AttentionRegions), String> {
    plan.validate().map_err(|error| error.to_string())?;
    if plan.model_family != "qwen2" {
        return Err(format!(
            "dense-qwen attention block requires model_family qwen2, found {}",
            plan.model_family
        ));
    }
    if !plan.transformations.is_empty() {
        return Err("dense-qwen attention block does not support transformed plans".into());
    }
    let graph = model_graph(plan, mode);
    if graph
        .operations
        .iter()
        .any(|operation| operation.operator == ModelOperator::SecureGather)
        || graph
            .state_inputs
            .iter()
            .chain(&graph.state_outputs)
            .any(|state| state.kind == StateKind::CacheIndices)
    {
        return Err(
            "dense-qwen attention block does not support MPCache selection operators".into(),
        );
    }
    let operations: BTreeMap<String, &ModelOperation> = graph
        .operations
        .iter()
        .map(|operation| (operation.id.clone(), operation))
        .collect();
    let scores = require_layer(
        exactly_one(
            &operations,
            layer,
            ModelOperator::AttentionScores,
            "attention scores",
        )?,
        layer,
        "attention scores",
    )?;
    let values = require_layer(
        exactly_one(
            &operations,
            layer,
            ModelOperator::AttentionValues,
            "attention values",
        )?,
        layer,
        "attention values",
    )?;
    let rope_q = require_layer(
        require_operator(
            producer(&operations, scores, 0, "attention scores query")?,
            ModelOperator::RotaryEmbedding,
            "attention scores query",
        )?,
        layer,
        "query rotary",
    )?;
    if rope_q.inputs.len() != 2 || rope_q.inputs[1] != "input.positions" {
        return Err("dense-qwen attention block query rotary requires input.positions".into());
    }
    let q_reshape = require_layer(
        require_operator(
            producer(&operations, rope_q, 0, "query rotary")?,
            ModelOperator::Reshape,
            "query rotary",
        )?,
        layer,
        "query head reshape",
    )?;
    let q_linear = require_layer(
        require_operator(
            producer(&operations, q_reshape, 0, "query head reshape")?,
            ModelOperator::Linear,
            "query head reshape",
        )?,
        layer,
        "query projection",
    )?;
    let key_view = require_layer(
        require_operator(
            producer(&operations, scores, 1, "attention scores key")?,
            ModelOperator::CacheSuffix,
            "attention scores key",
        )?,
        layer,
        "key cache view",
    )?;
    if key_view.state_kind != Some(StateKind::Key) {
        return Err("dense-qwen attention block key view must carry key state kind".into());
    }
    let key_append = require_layer(
        require_operator(
            producer(&operations, key_view, 0, "key cache view")?,
            ModelOperator::KvCacheAppend,
            "key cache view",
        )?,
        layer,
        "key cache append",
    )?;
    if key_append.state_kind != Some(StateKind::Key) {
        return Err("dense-qwen attention block key append must carry key state kind".into());
    }
    let rope_k = require_layer(
        require_operator(
            tensor_input(&operations, key_append, "key cache append")?,
            ModelOperator::RotaryEmbedding,
            "key cache append current tensor",
        )?,
        layer,
        "key rotary",
    )?;
    if rope_k.inputs.len() != 2 || rope_k.inputs[1] != "input.positions" {
        return Err("dense-qwen attention block key rotary requires input.positions".into());
    }
    let k_reshape = require_layer(
        require_operator(
            producer(&operations, rope_k, 0, "key rotary")?,
            ModelOperator::Reshape,
            "key rotary",
        )?,
        layer,
        "key head reshape",
    )?;
    let k_linear = require_layer(
        require_operator(
            producer(&operations, k_reshape, 0, "key head reshape")?,
            ModelOperator::Linear,
            "key head reshape",
        )?,
        layer,
        "key projection",
    )?;
    let softmax = require_layer(
        require_operator(
            producer(&operations, values, 0, "attention values probabilities")?,
            ModelOperator::Softmax,
            "attention values probabilities",
        )?,
        layer,
        "attention softmax",
    )?;
    let mask = require_operator(
        producer(&operations, softmax, 0, "attention softmax")?,
        ModelOperator::CausalMask,
        "attention softmax",
    )?;
    let scale = require_operator(
        producer(&operations, mask, 0, "attention causal mask")?,
        ModelOperator::AttentionScale,
        "attention causal mask",
    )?;
    if producer(&operations, scale, 0, "attention scale")?.id != scores.id {
        return Err(
            "dense-qwen attention block softmax chain must consume the traced scores".into(),
        );
    }
    let value_view = require_layer(
        require_operator(
            producer(&operations, values, 1, "attention values value")?,
            ModelOperator::CacheSuffix,
            "attention values value",
        )?,
        layer,
        "value cache view",
    )?;
    if value_view.state_kind != Some(StateKind::Value) {
        return Err("dense-qwen attention block value view must carry value state kind".into());
    }
    let value_append = require_layer(
        require_operator(
            producer(&operations, value_view, 0, "value cache view")?,
            ModelOperator::KvCacheAppend,
            "value cache view",
        )?,
        layer,
        "value cache append",
    )?;
    if value_append.state_kind != Some(StateKind::Value) {
        return Err("dense-qwen attention block value append must carry value state kind".into());
    }
    let v_reshape = require_layer(
        require_operator(
            tensor_input(&operations, value_append, "value cache append")?,
            ModelOperator::Reshape,
            "value cache append current tensor",
        )?,
        layer,
        "value head reshape",
    )?;
    let v_linear = require_layer(
        require_operator(
            producer(&operations, v_reshape, 0, "value head reshape")?,
            ModelOperator::Linear,
            "value head reshape",
        )?,
        layer,
        "value projection",
    )?;
    if q_linear.inputs != k_linear.inputs || q_linear.inputs != v_linear.inputs {
        return Err(
            "dense-qwen attention block requires q/k/v projections to share one norm input".into(),
        );
    }
    let input_norm = require_layer(
        require_operator(
            producer(&operations, q_linear, 0, "query projection")?,
            ModelOperator::RmsNorm,
            "attention input norm",
        )?,
        layer,
        "attention input norm",
    )?;
    let attention_hidden_reshape = require_layer(
        single_consumer(
            &operations,
            &values.id,
            layer,
            ModelOperator::Reshape,
            "attention hidden reshape",
        )?,
        layer,
        "attention hidden reshape",
    )?;
    if attention_hidden_reshape.inputs != [values.id.clone()] {
        return Err(
            "dense-qwen attention block attention hidden reshape must consume only attention values"
                .into(),
        );
    }
    let o_linear = require_layer(
        single_consumer(
            &operations,
            &attention_hidden_reshape.id,
            layer,
            ModelOperator::Linear,
            "output projection",
        )?,
        layer,
        "output projection",
    )?;
    if o_linear.inputs != [attention_hidden_reshape.id.clone()] {
        return Err(
            "dense-qwen attention block output projection must consume only attention hidden"
                .into(),
        );
    }
    let residual = require_layer(
        single_consumer(
            &operations,
            &o_linear.id,
            layer,
            ModelOperator::ResidualAdd,
            "attention residual",
        )?,
        layer,
        "attention residual",
    )?;
    if residual.inputs.len() != 2 || residual.inputs[1] != o_linear.id {
        return Err(
            "dense-qwen attention block attention residual must take the output projection at input 1"
                .into(),
        );
    }
    if input_norm.inputs.len() != 1 || residual.inputs[0] != input_norm.inputs[0] {
        return Err("dense-qwen attention block residual branch must equal the norm input".into());
    }
    let structure = AttentionStructure {
        input_norm: input_norm.clone(),
        q_linear: q_linear.clone(),
        k_linear: k_linear.clone(),
        v_linear: v_linear.clone(),
        o_linear: o_linear.clone(),
        q_reshape: q_reshape.clone(),
        k_reshape: k_reshape.clone(),
        v_reshape: v_reshape.clone(),
        rope_q: rope_q.clone(),
        rope_k: rope_k.clone(),
        key_append: key_append.clone(),
        value_append: value_append.clone(),
        key_view: key_view.clone(),
        value_view: value_view.clone(),
        scores: scores.clone(),
        softmax: softmax.clone(),
        values: values.clone(),
        attention_hidden_reshape: attention_hidden_reshape.clone(),
        residual: residual.clone(),
    };
    let regions = lower_regions(plan, mode, layer, graph, &structure)?;
    Ok((structure, regions))
}

fn find_q14_to_q10_region(
    regions: &[Q14ToQ10RescaleRegion],
    source: &ModelOperation,
    target: &ModelOperation,
    target_input_index: u32,
    role: &str,
) -> Result<Q14ToQ10RescaleRegion, String> {
    let mut matches = regions.iter().filter(|region| {
        region.source_operation_id == source.id
            && region.target_operation_id == target.id
            && region.target_input_index == target_input_index
    });
    let region = matches.next().ok_or_else(|| {
        format!("dense-qwen attention block is missing the {role} Q14-to-Q10 region")
    })?;
    if matches.next().is_some() {
        return Err(format!(
            "dense-qwen attention block has duplicate {role} Q14-to-Q10 regions"
        ));
    }
    Ok(region.clone())
}

fn lower_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
    layer: u64,
    graph: &DecoderGraph,
    structure: &AttentionStructure,
) -> Result<AttentionRegions, String> {
    let provenance = provenance_cap()?;
    let plan_digest = plan.digest();
    let norm = lower_rms_norm_q10_direct_region(&plan_digest, graph, mode, &structure.input_norm)?;
    let q = lower_model_linear_region(graph, mode, &structure.q_linear)?;
    let k = lower_model_linear_region(graph, mode, &structure.k_linear)?;
    let v = lower_model_linear_region(graph, mode, &structure.v_linear)?;
    let o = lower_model_linear_region(graph, mode, &structure.o_linear)?;
    let rescale_regions = lower_model_q14_to_q10_rescale_regions(plan, mode)?;
    let operations: BTreeMap<String, &ModelOperation> = graph
        .operations
        .iter()
        .map(|operation| (operation.id.clone(), operation))
        .collect();
    let layer_rescales = rescale_regions
        .iter()
        .filter(|region| {
            operations
                .get(region.source_operation_id.as_str())
                .is_some_and(|source| source.layer == Some(layer))
        })
        .count();
    if layer_rescales != 4 {
        return Err(format!(
            "dense-qwen attention block requires exactly four Q14-to-Q10 edges in layer {layer}, found {layer_rescales}"
        ));
    }
    let q_rescale = find_q14_to_q10_region(
        &rescale_regions,
        &structure.q_linear,
        &structure.q_reshape,
        0,
        "query",
    )?;
    let k_rescale = find_q14_to_q10_region(
        &rescale_regions,
        &structure.k_linear,
        &structure.k_reshape,
        0,
        "key",
    )?;
    let v_rescale = find_q14_to_q10_region(
        &rescale_regions,
        &structure.v_linear,
        &structure.v_reshape,
        0,
        "value",
    )?;
    let o_rescale = find_q14_to_q10_region(
        &rescale_regions,
        &structure.o_linear,
        &structure.residual,
        1,
        "output",
    )?;
    let q_reshape = lower_model_reshape_region(graph, mode, &structure.q_reshape)?;
    let k_reshape = lower_model_reshape_region(graph, mode, &structure.k_reshape)?;
    let v_reshape = lower_model_reshape_region(graph, mode, &structure.v_reshape)?;
    let attention_hidden_reshape =
        lower_model_reshape_region(graph, mode, &structure.attention_hidden_reshape)?;
    if q_reshape.layout != ModelReshapeLayout::BatchHeadsSequenceFeature
        || k_reshape.layout != ModelReshapeLayout::BatchHeadsSequenceFeature
        || v_reshape.layout != ModelReshapeLayout::BatchHeadsSequenceFeature
    {
        return Err("dense-qwen attention block requires head-layout q/k/v reshapes".into());
    }
    if attention_hidden_reshape.layout != ModelReshapeLayout::BatchSequenceHidden {
        return Err("dense-qwen attention block requires a hidden-layout attention reshape".into());
    }
    let rope_regions = lower_model_rope_q10_regions(plan, mode, &provenance)?;
    let q_rope = rope_regions
        .iter()
        .find(|region| region.operation_id == structure.rope_q.id)
        .cloned()
        .ok_or_else(|| {
            "dense-qwen attention block is missing the query rotary region".to_owned()
        })?;
    let k_rope = rope_regions
        .iter()
        .find(|region| region.operation_id == structure.rope_k.id)
        .cloned()
        .ok_or_else(|| "dense-qwen attention block is missing the key rotary region".to_owned())?;
    if q_rope.input_id != structure.q_reshape.id
        || k_rope.input_id != structure.k_reshape.id
        || q_rope.positions_id != "input.positions"
        || k_rope.positions_id != "input.positions"
    {
        return Err("dense-qwen attention block rotary regions do not match topology".into());
    }
    let append_regions = lower_model_kv_cache_append_q10_regions(plan, mode, &provenance)?;
    let key_append = append_regions
        .iter()
        .find(|region| region.operation_id == structure.key_append.id)
        .cloned()
        .ok_or_else(|| "dense-qwen attention block is missing the key append region".to_owned())?;
    let value_append = append_regions
        .iter()
        .find(|region| region.operation_id == structure.value_append.id)
        .cloned()
        .ok_or_else(|| {
            "dense-qwen attention block is missing the value append region".to_owned()
        })?;
    if key_append.current_input_id != structure.rope_k.id
        || value_append.current_input_id != structure.v_reshape.id
    {
        return Err("dense-qwen attention block cache append regions do not match topology".into());
    }
    let view_regions = lower_model_kv_cache_view_q10_regions(plan, mode, &provenance)?;
    let key_view = view_regions
        .iter()
        .find(|region| region.operation_id == structure.key_view.id)
        .cloned()
        .ok_or_else(|| "dense-qwen attention block is missing the key view region".to_owned())?;
    let value_view = view_regions
        .iter()
        .find(|region| region.operation_id == structure.value_view.id)
        .cloned()
        .ok_or_else(|| "dense-qwen attention block is missing the value view region".to_owned())?;
    if key_view.append_producer_id != structure.key_append.id
        || value_view.append_producer_id != structure.value_append.id
    {
        return Err("dense-qwen attention block cache view regions do not match topology".into());
    }
    let scores = lower_model_attention_scores_q20_regions(plan, mode)?
        .into_iter()
        .find(|region| region.score_operation_id == structure.scores.id)
        .ok_or_else(|| "dense-qwen attention block is missing the scores region".to_owned())?;
    if scores.layout != pllm_core::AttentionScoreQ20Layout::GroupedQueryCache
        || scores.query_input_id != structure.rope_q.id
        || scores.key_input_id != structure.key_view.id
    {
        return Err(
            "dense-qwen attention block requires a grouped-query cache scores region bound to the traced topology"
                .into(),
        );
    }
    let softmax = lower_model_softmax_q30_regions(plan, mode)?
        .into_iter()
        .find(|region| region.operation_id == structure.softmax.id)
        .ok_or_else(|| "dense-qwen attention block is missing the softmax region".to_owned())?;
    let attention_values = lower_model_attention_values_q10_regions(plan, mode)?
        .into_iter()
        .find(|region| region.operation_id == structure.values.id)
        .ok_or_else(|| {
            "dense-qwen attention block is missing the attention values region".to_owned()
        })?;
    if attention_values.layout != pllm_core::AttentionValueQ10Layout::GroupedQuery
        || attention_values.probabilities_input_id != structure.softmax.id
        || attention_values.values_input_id != structure.value_view.id
    {
        return Err(
            "dense-qwen attention block requires a grouped-query attention values region bound to the traced topology"
                .into(),
        );
    }
    let residual = lower_model_residual_region(graph, mode, &structure.residual)?;
    if residual.input_ids
        != [
            structure.input_norm.inputs[0].clone(),
            structure.o_linear.id.clone(),
        ]
    {
        return Err("dense-qwen attention block residual region does not match topology".into());
    }
    let regions = AttentionRegions {
        norm,
        q,
        k,
        v,
        o,
        q_rescale,
        k_rescale,
        v_rescale,
        o_rescale,
        q_reshape,
        k_reshape,
        v_reshape,
        q_rope,
        k_rope,
        key_append,
        value_append,
        key_view,
        value_view,
        scores,
        softmax,
        attention_values,
        attention_hidden_reshape,
        residual,
    };
    validate_shape_contract(structure, &regions)?;
    Ok(regions)
}

fn shape4(shape: &[u64], role: &str) -> Result<[u64; 4], String> {
    let parsed: [u64; 4] = shape
        .try_into()
        .map_err(|_| format!("dense-qwen attention block {role} must be rank-4"))?;
    if parsed.contains(&0) {
        return Err(format!("dense-qwen attention block {role} must be nonzero"));
    }
    Ok(parsed)
}

fn view_shape4(shape: &[u64], role: &str) -> Result<[usize; 4], String> {
    let parsed = shape4(shape, role)?;
    let mut converted = [0_usize; 4];
    for (target, source) in converted.iter_mut().zip(parsed) {
        *target = usize::try_from(source)
            .map_err(|_| format!("dense-qwen attention block {role} exceeds usize"))?;
    }
    Ok(converted)
}

fn validate_shape_contract(
    structure: &AttentionStructure,
    regions: &AttentionRegions,
) -> Result<(), String> {
    let input_shape = &structure.input_norm.output_shape;
    if input_shape.len() != 3 || input_shape.contains(&0) {
        return Err("dense-qwen attention block requires a nonzero rank-3 input shape".into());
    }
    if regions.norm.shape != *input_shape {
        return Err("dense-qwen attention block norm shape mismatch".into());
    }
    let q_shape = shape4(&structure.q_reshape.output_shape, "query head shape")?;
    let k_shape = shape4(&structure.k_reshape.output_shape, "key head shape")?;
    let v_shape = shape4(&structure.v_reshape.output_shape, "value head shape")?;
    if k_shape != v_shape {
        return Err("dense-qwen attention block key/value head shapes differ".into());
    }
    if q_shape[0] != input_shape[0]
        || q_shape[2] != input_shape[1]
        || k_shape[0] != input_shape[0]
        || k_shape[2] != input_shape[1]
    {
        return Err(
            "dense-qwen attention block head reshapes must preserve batch and sequence".into(),
        );
    }
    if q_shape[1] % k_shape[1] != 0 {
        return Err(
            "dense-qwen attention block requires query heads to be a multiple of kv heads".into(),
        );
    }
    if structure.values.output_shape != structure.q_reshape.output_shape {
        return Err(
            "dense-qwen attention block attention values must match the query head shape".into(),
        );
    }
    if structure.attention_hidden_reshape.output_shape.last()
        != structure.o_linear.output_shape.last()
        || structure.o_linear.output_shape != *input_shape
    {
        return Err(
            "dense-qwen attention block output projection must restore the hidden shape".into(),
        );
    }
    if structure.residual.output_shape != *input_shape {
        return Err("dense-qwen attention block residual output must match the input shape".into());
    }
    if regions.scores.output.shape.len() != 4 || regions.scores.output.shape[..3] != q_shape[..3] {
        return Err("dense-qwen attention block scores shape mismatch".into());
    }
    if regions.key_view.output_shape.len() != 4
        || regions.key_view.output_shape[0] != k_shape[0]
        || regions.key_view.output_shape[1] != k_shape[1]
        || regions.key_view.output_shape[3] != k_shape[3]
        || regions.value_view.output_shape != regions.key_view.output_shape
    {
        return Err("dense-qwen attention block cache view shape mismatch".into());
    }
    Ok(())
}

fn check_weight_ids(
    regions: &AttentionRegions,
    weights: DenseQwenAttentionWeights<'_>,
) -> Result<(), String> {
    if weights.norm_q10.weight_id != regions.norm.weight_id
        || weights.q_q4.weight_id != regions.q.weight_id
        || weights.k_q4.weight_id != regions.k.weight_id
        || weights.v_q4.weight_id != regions.v.weight_id
        || weights.o_q4.weight_id != regions.o.weight_id
    {
        return Err(
            "dense-qwen attention weight ids must match the traced region weight ids".into(),
        );
    }
    Ok(())
}

fn check_weight_lengths(
    regions: &AttentionRegions,
    weights: DenseQwenAttentionWeights<'_>,
    hidden: u64,
) -> Result<(), String> {
    let hidden =
        usize::try_from(hidden).map_err(|_| "dense-qwen attention hidden width exceeds usize")?;
    let norm_expected = hidden
        .checked_mul(2)
        .ok_or("dense-qwen attention norm weight length overflows usize")?;
    if weights.norm_q10.bytes.len() != norm_expected {
        return Err("dense-qwen attention norm weight must hold hidden i16 values".into());
    }
    let expected = |region: &ModelLinearRegion| -> Result<usize, String> {
        let input = usize::try_from(region.input.shape[1])
            .map_err(|_| "dense-qwen attention linear width exceeds usize")?;
        let output = usize::try_from(region.operation.output.shape[1])
            .map_err(|_| "dense-qwen attention linear width exceeds usize")?;
        input
            .checked_mul(output)
            .ok_or_else(|| "dense-qwen attention linear weight shape overflows usize".into())
    };
    if weights.q_q4.bytes.len() != expected(&regions.q)?
        || weights.k_q4.bytes.len() != expected(&regions.k)?
        || weights.v_q4.bytes.len() != expected(&regions.v)?
        || weights.o_q4.bytes.len() != expected(&regions.o)?
    {
        return Err(
            "dense-qwen attention projection weights must match [output,input] byte lengths".into(),
        );
    }
    Ok(())
}

fn validate_q4_row_bounds(
    weights: &[u8],
    rows: usize,
    columns: usize,
    role: &str,
) -> Result<(), String> {
    const MAX_Q10_INPUT_ABS: u64 = 32_768;
    for row in 0..rows {
        let absolute_sum = weights[row * columns..(row + 1) * columns]
            .iter()
            .enumerate()
            .try_fold(0_u64, |total, (column, byte)| {
                let value = i8::from_ne_bytes([*byte]);
                if !(-8..=7).contains(&value) {
                    return Err(format!(
                        "dense-qwen attention {role} row {row} column {column} holds {value}, outside signed q4"
                    ));
                }
                total
                    .checked_add(u64::from(i16::from(value).unsigned_abs()))
                    .ok_or_else(|| {
                        format!("dense-qwen attention {role} row {row} absolute sum overflows u64")
                    })
            })?;
        let bound = absolute_sum
            .checked_mul(MAX_Q10_INPUT_ABS)
            .ok_or_else(|| format!("dense-qwen attention {role} row {row} bound overflows u64"))?;
        if bound > i32::MAX as u64 {
            return Err(format!(
                "dense-qwen attention {role} row {row} cannot keep its exact dot inside signed i32"
            ));
        }
    }
    Ok(())
}

fn weight_artifacts(
    regions: &AttentionRegions,
    weights: DenseQwenAttentionWeights<'_>,
) -> Result<
    (
        DenseQwenAttentionWeightArtifact,
        DenseQwenAttentionWeightArtifact,
        DenseQwenAttentionWeightArtifact,
        DenseQwenAttentionWeightArtifact,
        DenseQwenAttentionWeightArtifact,
    ),
    String,
> {
    let artifact = |weight: DenseQwenAttentionWeightBytes<'_>,
                    shape: Vec<u64>,
                    element_type: DenseQwenAttentionElementType,
                    fractional_bits: u8,
                    layout: DenseQwenAttentionLayout| {
        DenseQwenAttentionWeightArtifact {
            weight_id: weight.weight_id.to_owned(),
            data_digest: digest_bytes(DENSE_QWEN_ATTENTION_WEIGHT_BYTES_DOMAIN, weight.bytes),
            shape,
            element_type,
            fractional_bits,
            layout,
            rounding: FixedPointRounding::TiesToEven,
            range_policy: DenseQwenAttentionRangePolicy::RejectNoSaturation,
        }
    };
    let matrix_shape =
        |region: &ModelLinearRegion| vec![region.operation.output.shape[1], region.input.shape[1]];
    Ok((
        artifact(
            weights.norm_q10,
            vec![regions.norm.shape[2]],
            DenseQwenAttentionElementType::SignedI16,
            regions.norm.weight_fractional_bits,
            DenseQwenAttentionLayout::RowMajorVector,
        ),
        artifact(
            weights.q_q4,
            matrix_shape(&regions.q),
            DenseQwenAttentionElementType::SignedI8,
            4,
            DenseQwenAttentionLayout::RowMajorOutputInput,
        ),
        artifact(
            weights.k_q4,
            matrix_shape(&regions.k),
            DenseQwenAttentionElementType::SignedI8,
            4,
            DenseQwenAttentionLayout::RowMajorOutputInput,
        ),
        artifact(
            weights.v_q4,
            matrix_shape(&regions.v),
            DenseQwenAttentionElementType::SignedI8,
            4,
            DenseQwenAttentionLayout::RowMajorOutputInput,
        ),
        artifact(
            weights.o_q4,
            matrix_shape(&regions.o),
            DenseQwenAttentionElementType::SignedI8,
            4,
            DenseQwenAttentionLayout::RowMajorOutputInput,
        ),
    ))
}

fn check_resource_policy(
    regions: &AttentionRegions,
    weights: DenseQwenAttentionWeights<'_>,
    resource_policy: &DenseQwenAttentionResourcePolicy,
) -> Result<(), String> {
    let total: u64 = [
        weights.norm_q10.bytes.len(),
        weights.q_q4.bytes.len(),
        weights.k_q4.bytes.len(),
        weights.v_q4.bytes.len(),
        weights.o_q4.bytes.len(),
    ]
    .iter()
    .try_fold(0_u64, |total, bytes| {
        let bytes =
            u64::try_from(*bytes).map_err(|_| "dense-qwen attention weight bytes exceed u64")?;
        total
            .checked_add(bytes)
            .ok_or("dense-qwen attention weight bytes overflow u64")
    })?;
    if total > resource_policy.max_total_weight_bytes {
        return Err("dense-qwen attention weights exceed the resource policy".into());
    }
    for (shape, role) in [
        (&regions.norm.shape, "input"),
        (&regions.q.operation.output.shape, "query projection"),
        (&regions.k.operation.output.shape, "key projection"),
        (&regions.v.operation.output.shape, "value projection"),
        (&regions.o.operation.output.shape, "output projection"),
        (&regions.q_reshape.output.shape, "query heads"),
        (&regions.scores.output.shape, "attention scores"),
        (&regions.attention_values.output.shape, "attention values"),
    ] {
        let elements = u64::try_from(tensor_elements(shape)?)
            .map_err(|_| format!("dense-qwen attention {role} elements exceed u64"))?;
        if elements > resource_policy.max_activation_elements {
            return Err(format!(
                "dense-qwen attention {role} exceeds the activation resource policy"
            ));
        }
    }
    Ok(())
}

fn build_composite(
    plan: &DecoderPlan,
    mode: DecoderMode,
    layer: u64,
    weight_manifest_digest: Digest,
    resource_policy: &DenseQwenAttentionResourcePolicy,
    weights: DenseQwenAttentionWeights<'_>,
) -> Result<DenseQwenAttentionComposite, String> {
    resource_policy.validate()?;
    let (structure, regions) = lower_structure(plan, mode, layer)?;
    check_weight_ids(&regions, weights)?;
    let hidden = structure.input_norm.output_shape[2];
    check_weight_lengths(&regions, weights, hidden)?;
    for (region, bytes, role) in [
        (&regions.q, weights.q_q4.bytes, "query projection"),
        (&regions.k, weights.k_q4.bytes, "key projection"),
        (&regions.v, weights.v_q4.bytes, "value projection"),
        (&regions.o, weights.o_q4.bytes, "output projection"),
    ] {
        let rows = usize::try_from(region.operation.output.shape[1])
            .map_err(|_| format!("dense-qwen attention {role} rows exceed usize"))?;
        let columns = usize::try_from(region.input.shape[1])
            .map_err(|_| format!("dense-qwen attention {role} columns exceed usize"))?;
        validate_q4_row_bounds(bytes, rows, columns, role)?;
    }
    check_resource_policy(&regions, weights, resource_policy)?;
    let (norm_weight, q_weight, k_weight, v_weight, o_weight) =
        weight_artifacts(&regions, weights)?;
    Ok(DenseQwenAttentionComposite {
        schema_version: DENSE_QWEN_ATTENTION_BLOCK_SCHEMA_VERSION.to_owned(),
        numeric_profile: DENSE_QWEN_ATTENTION_CLEAR_PROFILE.to_owned(),
        model_plan_digest: plan.digest(),
        weight_manifest_digest,
        resource_policy: resource_policy.clone(),
        mode,
        layer,
        input_shape: structure.input_norm.output_shape.clone(),
        q_shape: structure.q_reshape.output_shape.clone(),
        kv_shape: structure.k_reshape.output_shape.clone(),
        output_shape: structure.residual.output_shape.clone(),
        semantic_guard_model_family: plan.model_family.clone(),
        semantic_guard_adapter: plan.adapter.clone(),
        norm: regions.norm,
        q: regions.q,
        k: regions.k,
        v: regions.v,
        o: regions.o,
        q_rescale: regions.q_rescale,
        k_rescale: regions.k_rescale,
        v_rescale: regions.v_rescale,
        o_rescale: regions.o_rescale,
        q_reshape: regions.q_reshape,
        k_reshape: regions.k_reshape,
        v_reshape: regions.v_reshape,
        q_rope: regions.q_rope,
        k_rope: regions.k_rope,
        key_append: regions.key_append,
        value_append: regions.value_append,
        key_view: regions.key_view,
        value_view: regions.value_view,
        scores: regions.scores,
        softmax: regions.softmax,
        attention_values: regions.attention_values,
        attention_hidden_reshape: regions.attention_hidden_reshape,
        residual: regions.residual,
        norm_weight,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        protected_execution: false,
        complete_decoder: false,
    })
}

struct OwnedWeights {
    norm_bytes: Vec<u8>,
    norm_q10: Vec<i16>,
    q_q4: Vec<u8>,
    k_q4: Vec<u8>,
    v_q4: Vec<u8>,
    o_q4: Vec<u8>,
}

impl OwnedWeights {
    fn copy(weights: DenseQwenAttentionWeights<'_>) -> Result<Self, String> {
        let copy_bytes = |bytes: &[u8], role: &str| -> Result<Vec<u8>, String> {
            let mut copied = Vec::new();
            copied.try_reserve_exact(bytes.len()).map_err(|_| {
                format!("dense-qwen attention {role} weight copy allocation failed")
            })?;
            copied.extend_from_slice(bytes);
            Ok(copied)
        };
        if weights.norm_q10.bytes.len() % 2 != 0 {
            return Err("dense-qwen attention norm weight requires even byte length".into());
        }
        let mut norm_q10 = Vec::new();
        norm_q10
            .try_reserve_exact(weights.norm_q10.bytes.len() / 2)
            .map_err(|_| "dense-qwen attention norm weight decode allocation failed")?;
        for chunk in weights.norm_q10.bytes.chunks_exact(2) {
            norm_q10.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        Ok(Self {
            norm_bytes: copy_bytes(weights.norm_q10.bytes, "norm")?,
            norm_q10,
            q_q4: copy_bytes(weights.q_q4.bytes, "query projection")?,
            k_q4: copy_bytes(weights.k_q4.bytes, "key projection")?,
            v_q4: copy_bytes(weights.v_q4.bytes, "value projection")?,
            o_q4: copy_bytes(weights.o_q4.bytes, "output projection")?,
        })
    }

    fn borrowed(&self) -> DenseQwenAttentionWeights<'_> {
        DenseQwenAttentionWeights {
            norm_q10: DenseQwenAttentionWeightBytes {
                weight_id: "",
                bytes: &self.norm_bytes,
            },
            q_q4: DenseQwenAttentionWeightBytes {
                weight_id: "",
                bytes: &self.q_q4,
            },
            k_q4: DenseQwenAttentionWeightBytes {
                weight_id: "",
                bytes: &self.k_q4,
            },
            v_q4: DenseQwenAttentionWeightBytes {
                weight_id: "",
                bytes: &self.v_q4,
            },
            o_q4: DenseQwenAttentionWeightBytes {
                weight_id: "",
                bytes: &self.o_q4,
            },
        }
    }
}

pub struct CompiledDenseQwenAttentionBlock {
    pub composite: DenseQwenAttentionComposite,
    weights: OwnedWeights,
    manifest: DenseQwenAttentionWeightManifest,
    binding_digest: Digest,
}

impl CompiledDenseQwenAttentionBlock {
    pub fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compile_dense_qwen_attention_block(
    plan: &DecoderPlan,
    mode: DecoderMode,
    layer: u64,
    weight_manifest: DenseQwenAttentionWeightManifest,
    resource_policy: DenseQwenAttentionResourcePolicy,
    weights: DenseQwenAttentionWeights<'_>,
) -> Result<CompiledDenseQwenAttentionBlock, String> {
    resource_policy.validate()?;
    if weight_manifest.schema_version != DENSE_QWEN_ATTENTION_WEIGHT_MANIFEST_SCHEMA_VERSION
        || weight_manifest.entries.len() != 5
    {
        return Err("dense-qwen attention requires a five-entry weight manifest".into());
    }
    let composite = build_composite(
        plan,
        mode,
        layer,
        weight_manifest.digest(),
        &resource_policy,
        weights,
    )?;
    for artifact in [
        &composite.norm_weight,
        &composite.q_weight,
        &composite.k_weight,
        &composite.v_weight,
        &composite.o_weight,
    ] {
        if weight_manifest.entries.get(&artifact.weight_id) != Some(&artifact.data_digest) {
            return Err(
                "dense-qwen attention weight manifest does not match the supplied weights".into(),
            );
        }
    }
    let owned = OwnedWeights::copy(weights)?;
    let binding_digest = canonical_digest(DENSE_QWEN_ATTENTION_BINDING_DOMAIN, &composite);
    Ok(CompiledDenseQwenAttentionBlock {
        composite,
        weights: owned,
        manifest: weight_manifest,
        binding_digest,
    })
}

fn validate_compiled_block(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenAttentionBlock,
) -> Result<(), String> {
    if canonical_digest(DENSE_QWEN_ATTENTION_BINDING_DOMAIN, &block.composite)
        != block.binding_digest
    {
        return Err("dense-qwen attention binding digest mismatch".into());
    }
    if block.manifest.digest() != block.composite.weight_manifest_digest {
        return Err("dense-qwen attention manifest digest mismatch".into());
    }
    block.composite.resource_policy.validate()?;
    let borrowed = block.weights.borrowed();
    let expected = build_composite(
        plan,
        block.composite.mode,
        block.composite.layer,
        block.composite.weight_manifest_digest.clone(),
        &block.composite.resource_policy,
        DenseQwenAttentionWeights {
            norm_q10: DenseQwenAttentionWeightBytes {
                weight_id: &block.composite.norm_weight.weight_id,
                bytes: borrowed.norm_q10.bytes,
            },
            q_q4: DenseQwenAttentionWeightBytes {
                weight_id: &block.composite.q_weight.weight_id,
                bytes: borrowed.q_q4.bytes,
            },
            k_q4: DenseQwenAttentionWeightBytes {
                weight_id: &block.composite.k_weight.weight_id,
                bytes: borrowed.k_q4.bytes,
            },
            v_q4: DenseQwenAttentionWeightBytes {
                weight_id: &block.composite.v_weight.weight_id,
                bytes: borrowed.v_q4.bytes,
            },
            o_q4: DenseQwenAttentionWeightBytes {
                weight_id: &block.composite.o_weight.weight_id,
                bytes: borrowed.o_q4.bytes,
            },
        },
    )?;
    if expected != block.composite {
        return Err("dense-qwen attention composite does not match plan and weights".into());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseQwenAttentionExecutionPolicy {
    pub threads: usize,
    pub simd: bool,
    pub provenance: ProvenancePrimitiveResourcePolicy,
    pub scores: pllm_core::AttentionScoreQ20Policy,
    pub softmax: pllm_core::SoftmaxQ30Policy,
    pub values: pllm_core::AttentionValueQ10Policy,
}

pub struct DenseQwenAttentionOutputQ10 {
    values: Vec<i16>,
    shape: [usize; 3],
}

impl DenseQwenAttentionOutputQ10 {
    pub fn values(&self) -> &[i16] {
        &self.values
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }
}

impl Drop for DenseQwenAttentionOutputQ10 {
    fn drop(&mut self) {
        self.values.zeroize();
    }
}

pub struct DenseQwenAttentionState {
    key: ProvenanceBoundKvCacheQ10,
    value: ProvenanceBoundKvCacheQ10,
    plan_digest: Digest,
    weight_manifest_digest: Digest,
    layer: u64,
    usable: bool,
}

impl DenseQwenAttentionState {
    pub fn is_usable(&self) -> bool {
        self.usable
    }
}

fn decode_centered_q10(input: &[u32], role: &str) -> Result<Zeroizing<Vec<i16>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| format!("dense-qwen attention {role} decode allocation failed"))?;
    for value in input {
        output.push(
            i16::try_from(*value as i32)
                .map_err(|_| format!("dense-qwen attention {role} is outside centered q10"))?,
        );
    }
    Ok(Zeroizing::new(output))
}

fn encode_centered_i16(input: &[i16]) -> Result<Zeroizing<Vec<u32>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| "dense-qwen attention wrap32 encode allocation failed")?;
    output.extend(input.iter().map(|value| *value as i32 as u32));
    Ok(Zeroizing::new(output))
}

fn shape3(shape: &[u64], role: &str) -> Result<[usize; 3], String> {
    let parsed: [u64; 3] = shape
        .try_into()
        .map_err(|_| format!("dense-qwen attention {role} must be rank-3"))?;
    if parsed.contains(&0) {
        return Err(format!("dense-qwen attention {role} must be nonzero"));
    }
    let mut converted = [0usize; 3];
    for (target, source) in converted.iter_mut().zip(parsed) {
        *target = usize::try_from(source)
            .map_err(|_| format!("dense-qwen attention {role} exceeds usize"))?;
    }
    Ok(converted)
}

fn check_activation(
    block: &CompiledDenseQwenAttentionBlock,
    elements: usize,
    role: &str,
) -> Result<(), String> {
    let elements = u64::try_from(elements)
        .map_err(|_| format!("dense-qwen attention {role} elements exceed u64"))?;
    if elements > block.composite.resource_policy.max_activation_elements {
        return Err(format!(
            "dense-qwen attention {role} exceeds the activation resource policy"
        ));
    }
    Ok(())
}

fn validate_execution_inputs(
    block: &CompiledDenseQwenAttentionBlock,
    input_q10: &[u32],
    positions: &[u32],
    attention_mask: &[bool],
    valid_lengths: &[usize],
) -> Result<[usize; 3], String> {
    let [batch, query, hidden] = shape3(&block.composite.input_shape, "input shape")?;
    let elements = batch
        .checked_mul(query)
        .and_then(|value| value.checked_mul(hidden))
        .ok_or_else(|| "dense-qwen attention input elements overflow usize".to_owned())?;
    check_activation(block, elements, "input")?;
    if input_q10.len() != elements {
        return Err(format!(
            "dense-qwen attention input requires {elements} elements, received {}",
            input_q10.len()
        ));
    }
    let queries = batch
        .checked_mul(query)
        .ok_or_else(|| "dense-qwen attention query count overflows usize".to_owned())?;
    if positions.len() != queries || attention_mask.len() != queries {
        return Err(
            "dense-qwen attention positions and attention mask must cover every query".into(),
        );
    }
    if valid_lengths.len() != batch {
        return Err("dense-qwen attention valid lengths must provide one value per batch".into());
    }
    Ok([batch, query, hidden])
}

fn flatten_view(
    view: &BoundedKvCacheViewQ10,
    view_shape: [usize; 4],
    role: &str,
) -> Result<Zeroizing<Vec<i16>>, String> {
    let [batch_count, head_count, expected_sequence, expected_dim] = view_shape;
    let maximum_sequence = view.maximum_sequence();
    let head_dim = view.head_dim();
    if maximum_sequence != expected_sequence || head_dim != expected_dim {
        return Err(format!(
            "dense-qwen attention {role} view dimensions do not match its cache shape"
        ));
    }
    let valid_lengths = view.valid_lengths();
    if valid_lengths.len() != batch_count {
        return Err(format!(
            "dense-qwen attention {role} view must provide one valid length per batch"
        ));
    }
    let total = batch_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(maximum_sequence))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "dense-qwen attention view elements overflow usize".to_owned())?;
    let mut flat = Vec::new();
    flat.try_reserve_exact(total)
        .map_err(|_| "dense-qwen attention view flatten allocation failed")?;
    flat.resize(total, 0_i16);
    for (batch, &valid_length) in valid_lengths.iter().enumerate() {
        let expected_prefix = valid_length
            .checked_mul(head_dim)
            .ok_or_else(|| "dense-qwen attention view valid length overflows usize".to_owned())?;
        for head in 0..head_count {
            let prefix = view.prefix(batch, head).ok_or_else(|| {
                format!("dense-qwen attention {role} view is missing a batch/head prefix")
            })?;
            if prefix.len() != expected_prefix {
                return Err(format!(
                    "dense-qwen attention {role} view prefix is inconsistent with its valid lengths"
                ));
            }
            let offset = batch
                .checked_mul(head_count)
                .and_then(|value| value.checked_add(head))
                .and_then(|value| value.checked_mul(maximum_sequence))
                .and_then(|value| value.checked_mul(head_dim))
                .ok_or_else(|| "dense-qwen attention view offset overflows usize".to_owned())?;
            flat[offset..offset + prefix.len()].copy_from_slice(prefix);
        }
    }
    Ok(Zeroizing::new(flat))
}

struct PreCacheTensors {
    q_rope: Zeroizing<Vec<i16>>,
    k_rope: Zeroizing<Vec<i16>>,
    v: Zeroizing<Vec<i16>>,
}

fn pre_cache_forward(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenAttentionBlock,
    input_q10: &[u32],
    positions: &[u32],
    policy: DenseQwenAttentionExecutionPolicy,
) -> Result<PreCacheTensors, String> {
    let composite = &block.composite;
    let input = decode_centered_q10(input_q10, "input")?;
    let normalized_i32 = Zeroizing::new(execute_rms_norm_q10_direct(
        plan,
        &composite.norm,
        &input,
        &block.weights.norm_q10,
    )?);
    check_activation(block, normalized_i32.len(), "normalized input")?;
    let mut normalized = Vec::new();
    normalized
        .try_reserve_exact(normalized_i32.len())
        .map_err(|_| "dense-qwen attention normalized input allocation failed")?;
    for value in normalized_i32.iter() {
        normalized.push(
            i16::try_from(*value)
                .map_err(|_| "dense-qwen attention RMSNorm output is outside signed Q10 i16")?,
        );
    }
    let normalized = Zeroizing::new(normalized);
    let normalized_wrap = encode_centered_i16(&normalized)?;
    let projection = |linear: &ModelLinearRegion,
                      rescale: &Q14ToQ10RescaleRegion,
                      reshape: &ModelReshapeRegion,
                      weights: &[u8],
                      role: &str|
     -> Result<Zeroizing<Vec<i16>>, String> {
        let q14 = Zeroizing::new(execute_model_linear(
            linear,
            weights,
            &normalized_wrap,
            policy.threads,
            policy.simd,
        )?);
        check_activation(block, q14.len(), role)?;
        let q10 = Zeroizing::new(execute_q14_to_q10_rescale(rescale, &q14)?);
        let reshaped = Zeroizing::new(execute_model_reshape(reshape, &encode_centered_i16(&q10)?)?);
        decode_centered_q10(&reshaped, role)
    };
    let q_heads = projection(
        &composite.q,
        &composite.q_rescale,
        &composite.q_reshape,
        &block.weights.q_q4,
        "query projection",
    )?;
    let k_heads = projection(
        &composite.k,
        &composite.k_rescale,
        &composite.k_reshape,
        &block.weights.k_q4,
        "key projection",
    )?;
    let v_heads = projection(
        &composite.v,
        &composite.v_rescale,
        &composite.v_reshape,
        &block.weights.v_q4,
        "value projection",
    )?;
    let q_rope = Zeroizing::new(execute_model_rope_q10(
        plan,
        &composite.q_rope,
        &q_heads,
        positions,
        &policy.provenance,
    )?);
    check_activation(block, q_rope.len(), "query rotary")?;
    let k_rope = Zeroizing::new(execute_model_rope_q10(
        plan,
        &composite.k_rope,
        &k_heads,
        positions,
        &policy.provenance,
    )?);
    check_activation(block, k_rope.len(), "key rotary")?;
    Ok(PreCacheTensors {
        q_rope,
        k_rope,
        v: v_heads,
    })
}

#[allow(clippy::too_many_arguments)]
fn post_cache_forward(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenAttentionBlock,
    input_q10: &[u32],
    positions: &[u32],
    attention_mask: &[bool],
    tensors: &PreCacheTensors,
    key_cache: &ProvenanceBoundKvCacheQ10,
    value_cache: &ProvenanceBoundKvCacheQ10,
    policy: DenseQwenAttentionExecutionPolicy,
) -> Result<DenseQwenAttentionOutputQ10, String> {
    let composite = &block.composite;
    let key_view =
        execute_model_kv_cache_view_q10(key_cache, plan, &composite.key_view, &policy.provenance)?;
    let value_view = execute_model_kv_cache_view_q10(
        value_cache,
        plan,
        &composite.value_view,
        &policy.provenance,
    )?;
    let key_flat = flatten_view(
        &key_view,
        view_shape4(&composite.key_view.output_shape, "key view")?,
        "key",
    )?;
    check_activation(block, key_flat.len(), "key view")?;
    let value_flat = flatten_view(
        &value_view,
        view_shape4(&composite.value_view.output_shape, "value view")?,
        "value",
    )?;
    check_activation(block, value_flat.len(), "value view")?;
    let mut query_mask = Vec::new();
    query_mask
        .try_reserve_exact(attention_mask.len())
        .map_err(|_| "dense-qwen attention query mask allocation failed")?;
    query_mask.extend(attention_mask.iter().map(|value| u8::from(*value)));
    let query_mask = Zeroizing::new(query_mask);
    let scores = execute_model_attention_scores_q20(
        &composite.scores,
        &tensors.q_rope,
        &key_flat,
        None,
        positions,
        &query_mask,
        Some(key_view.valid_lengths()),
        policy.scores,
    )?;
    check_activation(block, scores.scores().len(), "attention scores")?;
    let probabilities = execute_model_softmax_q30(
        &composite.softmax,
        scores.scores(),
        scores.allowed(),
        policy.softmax,
    )?;
    let values = execute_model_attention_values_q10(
        &composite.attention_values,
        &probabilities,
        &value_flat,
        policy.values,
    )?;
    check_activation(block, values.values().len(), "attention values")?;
    let hidden_wrap = Zeroizing::new(execute_model_reshape(
        &composite.attention_hidden_reshape,
        &encode_centered_i16(values.values())?,
    )?);
    let hidden = decode_centered_q10(&hidden_wrap, "attention hidden")?;
    let hidden_wrap = encode_centered_i16(&hidden)?;
    let o_q14 = Zeroizing::new(execute_model_linear(
        &composite.o,
        &block.weights.o_q4,
        &hidden_wrap,
        policy.threads,
        policy.simd,
    )?);
    check_activation(block, o_q14.len(), "output projection")?;
    let o_q10 = Zeroizing::new(execute_q14_to_q10_rescale(&composite.o_rescale, &o_q14)?);
    let output_wrap = Zeroizing::new(execute_model_residual(
        &composite.residual,
        input_q10,
        &encode_centered_i16(&o_q10)?,
    )?);
    let mut output = decode_centered_q10(&output_wrap, "residual output")?;
    let shape = shape3(&composite.output_shape, "output shape")?;
    if output.len()
        != shape
            .iter()
            .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))
            .ok_or_else(|| "dense-qwen attention output elements overflow usize".to_owned())?
    {
        return Err("dense-qwen attention output length mismatch".into());
    }
    let values = std::mem::take(&mut *output);
    Ok(DenseQwenAttentionOutputQ10 { values, shape })
}

#[allow(clippy::too_many_arguments)]
pub fn execute_dense_qwen_attention_prefill(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenAttentionBlock,
    input_q10: &[u32],
    positions: &[u32],
    attention_mask: &[bool],
    valid_lengths: &[usize],
    policy: DenseQwenAttentionExecutionPolicy,
) -> Result<(DenseQwenAttentionOutputQ10, DenseQwenAttentionState), String> {
    validate_compiled_block(plan, block)?;
    if block.composite.mode != DecoderMode::Prefill {
        return Err("dense-qwen attention prefill requires a prefill block".into());
    }
    validate_execution_inputs(block, input_q10, positions, attention_mask, valid_lengths)?;
    let tensors = pre_cache_forward(plan, block, input_q10, positions, policy)?;
    let key_cache = initialize_model_kv_cache_q10(
        plan,
        &block.composite.key_append,
        KvCacheQ10ExecutionInputs::new(&tensors.k_rope, positions, attention_mask, valid_lengths),
        &policy.provenance,
    )?;
    let value_cache = initialize_model_kv_cache_q10(
        plan,
        &block.composite.value_append,
        KvCacheQ10ExecutionInputs::new(&tensors.v, positions, attention_mask, valid_lengths),
        &policy.provenance,
    )?;
    let output = post_cache_forward(
        plan,
        block,
        input_q10,
        positions,
        attention_mask,
        &tensors,
        &key_cache,
        &value_cache,
        policy,
    )?;
    Ok((
        output,
        DenseQwenAttentionState {
            key: key_cache,
            value: value_cache,
            plan_digest: block.composite.model_plan_digest.clone(),
            weight_manifest_digest: block.composite.weight_manifest_digest.clone(),
            layer: block.composite.layer,
            usable: true,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn execute_dense_qwen_attention_decode(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenAttentionBlock,
    state: &mut DenseQwenAttentionState,
    input_q10: &[u32],
    positions: &[u32],
    attention_mask: &[bool],
    new_valid_lengths: &[usize],
    policy: DenseQwenAttentionExecutionPolicy,
) -> Result<DenseQwenAttentionOutputQ10, String> {
    validate_compiled_block(plan, block)?;
    if block.composite.mode != DecoderMode::Decode {
        return Err("dense-qwen attention decode requires a decode block".into());
    }
    if !state.usable {
        return Err("dense-qwen attention state is unusable".into());
    }
    if state.plan_digest != block.composite.model_plan_digest
        || state.weight_manifest_digest != block.composite.weight_manifest_digest
        || state.layer != block.composite.layer
    {
        return Err("dense-qwen attention state does not match the block binding".into());
    }
    validate_execution_inputs(
        block,
        input_q10,
        positions,
        attention_mask,
        new_valid_lengths,
    )?;
    let tensors = pre_cache_forward(plan, block, input_q10, positions, policy)?;
    state.usable = false;
    append_model_kv_cache_q10(
        &mut state.key,
        plan,
        &block.composite.key_append,
        KvCacheQ10ExecutionInputs::new(
            &tensors.k_rope,
            positions,
            attention_mask,
            new_valid_lengths,
        ),
        &policy.provenance,
    )?;
    append_model_kv_cache_q10(
        &mut state.value,
        plan,
        &block.composite.value_append,
        KvCacheQ10ExecutionInputs::new(&tensors.v, positions, attention_mask, new_valid_lengths),
        &policy.provenance,
    )?;
    let output = post_cache_forward(
        plan,
        block,
        input_q10,
        positions,
        attention_mask,
        &tensors,
        &state.key,
        &state.value,
        policy,
    )?;
    state.usable = true;
    Ok(output)
}
