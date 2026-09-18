use super::{
    execute_q14_to_q7_rescale, lower_model_gated_multiply_q7_regions_with_components,
    lower_model_linear_region, lower_model_residual_region, lower_rms_norm_q10_direct_region,
    model_graph, tensor_elements, FixedPointRounding, ModelGatedMultiplyQ7Region,
    ModelLinearRegion, ModelResidualRegion, ModelRmsNormQ10DirectRegion,
    GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
    GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS, GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
};
use pllm_models::{DecoderMode, DecoderPlan, ModelOperation, ModelOperator};
use pllm_types::{canonical_digest, digest_bytes, Digest};
use serde::Serialize;
use std::collections::BTreeMap;

pub const DENSE_QWEN_MLP_BLOCK_SCHEMA_VERSION: &str = "pllm.dense_qwen_mlp_block.v2";
pub const DENSE_QWEN_MLP_CLEAR_PROFILE: &str = "pllm.clear_exact.dense_qwen_mlp.q10_q7.v2";
pub const DENSE_QWEN_MLP_WEIGHT_MANIFEST_SCHEMA_VERSION: &str =
    "pllm.dense_qwen_mlp_weight_manifest.v1";
pub const DENSE_QWEN_MLP_HARD_MAX_TOTAL_WEIGHT_BYTES: u64 = 16 * 1024 * 1024 * 1024;
pub const DENSE_QWEN_MLP_HARD_MAX_ACTIVATION_ELEMENTS: u64 =
    GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS as u64;
const WEIGHT_DIGEST_DOMAIN: &str = "pllm.dense_qwen_mlp.weight_bytes.v1";
const WEIGHT_MANIFEST_DIGEST_DOMAIN: &str = "pllm.dense_qwen_mlp.weight_manifest.v1";
const BINDING_DIGEST_DOMAIN: &str = "pllm.dense_qwen_mlp.binding.v1";

/// Exact supplied bytes for one semantic weight. I16 values use little-endian encoding.
#[derive(Clone, Copy, Debug)]
pub struct DenseQwenMlpWeightBytes<'a> {
    pub weight_id: &'a str,
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct DenseQwenMlpWeights<'a> {
    pub norm_q10: DenseQwenMlpWeightBytes<'a>,
    pub gate_q4: DenseQwenMlpWeightBytes<'a>,
    pub up_q4: DenseQwenMlpWeightBytes<'a>,
    pub down_q3: DenseQwenMlpWeightBytes<'a>,
}

/// Immutable byte commitment for exactly named MLP tensors. This binds supplied data; it does
/// not establish checkpoint provenance.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseQwenMlpWeightManifest {
    schema_version: String,
    entries: BTreeMap<String, Digest>,
}

impl DenseQwenMlpWeightManifest {
    pub fn from_weights(weights: DenseQwenMlpWeights<'_>) -> Result<Self, String> {
        let mut entries = BTreeMap::new();
        for weight in [
            weights.norm_q10,
            weights.gate_q4,
            weights.up_q4,
            weights.down_q3,
        ] {
            if weight.weight_id.is_empty()
                || entries
                    .insert(
                        weight.weight_id.to_owned(),
                        digest_bytes(WEIGHT_DIGEST_DOMAIN, weight.bytes),
                    )
                    .is_some()
            {
                return Err(
                    "dense Qwen MLP weight manifest requires four unique weight IDs".into(),
                );
            }
        }
        Ok(Self {
            schema_version: DENSE_QWEN_MLP_WEIGHT_MANIFEST_SCHEMA_VERSION.into(),
            entries,
        })
    }

    pub fn digest(&self) -> Digest {
        canonical_digest(WEIGHT_MANIFEST_DIGEST_DOMAIN, self)
    }

    pub fn entries(&self) -> &BTreeMap<String, Digest> {
        &self.entries
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseQwenMlpResourcePolicy {
    /// Sum of four serialized supplied weight tensors, before owned copies are allocated.
    pub max_total_weight_bytes: u64,
    /// Maximum element count of any input, intermediate, or output activation tensor.
    pub max_activation_elements: u64,
}

impl DenseQwenMlpResourcePolicy {
    fn validate(&self) -> Result<(), String> {
        if self.max_total_weight_bytes == 0
            || self.max_total_weight_bytes > DENSE_QWEN_MLP_HARD_MAX_TOTAL_WEIGHT_BYTES
            || self.max_activation_elements == 0
            || self.max_activation_elements > DENSE_QWEN_MLP_HARD_MAX_ACTIVATION_ELEMENTS
        {
            return Err("dense Qwen MLP resource policy exceeds hard caps or contains zero".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQwenMlpElementType {
    SignedI16,
    SignedI8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQwenMlpLayout {
    RowMajorVector,
    RowMajorOutputInput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQwenMlpRangePolicy {
    RejectNoSaturation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQwenMlpResidualStorage {
    CenteredWrap32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQwenMlpResidualOutputRangePolicy {
    RejectOutsideSignedI16ForNextRmsNorm,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
/// Numeric refinement of [`ModelResidualRegion`]'s wrap32 contract. Residual output is interpreted
/// as centered signed Q10, then checked and bridged to i16 for the following RMSNorm input.
pub struct DenseQwenMlpResidualRefinement {
    pub output_fractional_bits: u8,
    pub storage: DenseQwenMlpResidualStorage,
    pub output_range_policy: DenseQwenMlpResidualOutputRangePolicy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseQwenMlpWeightArtifact {
    pub weight_id: String,
    pub data_digest: Digest,
    pub shape: Vec<u64>,
    pub element_type: DenseQwenMlpElementType,
    pub fractional_bits: u8,
    pub layout: DenseQwenMlpLayout,
    pub rounding: FixedPointRounding,
    pub range_policy: DenseQwenMlpRangePolicy,
}

/// Structural and numeric contract for one clear dense Qwen2/Qwen3 MLP block.
/// `protected_execution` and `complete_decoder` remain false: this is only a clear oracle slice.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseQwenMlpComposite {
    pub schema_version: String,
    pub numeric_profile: String,
    pub model_plan_digest: Digest,
    pub weight_manifest_digest: Digest,
    /// Immutable-plan label guard only. Topology checks remain decisive; this is not provenance.
    pub semantic_guard_model_family: String,
    /// Immutable-plan adapter guard only. This does not authenticate checkpoint bytes.
    pub semantic_guard_adapter: String,
    pub resource_policy: DenseQwenMlpResourcePolicy,
    pub mode: DecoderMode,
    pub layer: u64,
    pub input_shape: Vec<u64>,
    pub intermediate_shape: Vec<u64>,
    pub norm: ModelRmsNormQ10DirectRegion,
    pub gate: ModelLinearRegion,
    pub up: ModelLinearRegion,
    pub gated_multiply: ModelGatedMultiplyQ7Region,
    pub down: ModelLinearRegion,
    pub residual: ModelResidualRegion,
    pub residual_refinement: DenseQwenMlpResidualRefinement,
    pub norm_weight: DenseQwenMlpWeightArtifact,
    pub gate_weight: DenseQwenMlpWeightArtifact,
    pub up_weight: DenseQwenMlpWeightArtifact,
    pub down_weight: DenseQwenMlpWeightArtifact,
    pub protected_execution: bool,
    pub complete_decoder: bool,
}

#[derive(Clone, Debug)]
struct OwnedWeights {
    norm_bytes: Vec<u8>,
    norm_q10: Vec<i16>,
    gate_q4: Vec<u8>,
    up_q4: Vec<u8>,
    down_q3: Vec<u8>,
}

/// Compiled block owns all supplied weight bytes. Public contract may be inspected, but execution
/// revalidates it against both plan and private binding before arithmetic.
#[derive(Clone, Debug)]
pub struct CompiledDenseQwenMlpBlock {
    pub composite: DenseQwenMlpComposite,
    weights: OwnedWeights,
    weight_manifest: DenseQwenMlpWeightManifest,
    binding_digest: Digest,
}

struct Structure {
    input_shape: Vec<u64>,
    intermediate_shape: Vec<u64>,
    hidden: usize,
    intermediate: usize,
    norm: ModelRmsNormQ10DirectRegion,
    gate: ModelLinearRegion,
    up: ModelLinearRegion,
    gated_multiply: ModelGatedMultiplyQ7Region,
    down: ModelLinearRegion,
    residual: ModelResidualRegion,
}

pub(crate) struct DenseQwenMlpInputStage {
    pub(crate) input_i16: Vec<i16>,
    pub(crate) rows: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
}

pub(crate) struct DenseQwenMlpGateUpStage {
    pub(crate) gate_q7: Vec<i16>,
    pub(crate) up_q7: Vec<i16>,
}

pub fn compile_dense_qwen_mlp_block(
    plan: &DecoderPlan,
    mode: DecoderMode,
    layer: u64,
    weight_manifest: DenseQwenMlpWeightManifest,
    resource_policy: DenseQwenMlpResourcePolicy,
    weights: DenseQwenMlpWeights<'_>,
) -> Result<CompiledDenseQwenMlpBlock, String> {
    resource_policy.validate()?;
    let structure = lower_structure(plan, mode, layer)?;
    validate_weight_id(weights.norm_q10, &structure.norm.weight_id, "norm")?;
    validate_weight_id(weights.gate_q4, &structure.gate.weight_id, "gate")?;
    validate_weight_id(weights.up_q4, &structure.up.weight_id, "up")?;
    validate_weight_id(weights.down_q3, &structure.down.weight_id, "down")?;

    let norm_bytes = expected_bytes(structure.hidden, 2, "norm Q10")?;
    let gate_bytes = expected_bytes(structure.intermediate, structure.hidden, "gate Q4")?;
    let up_bytes = expected_bytes(structure.intermediate, structure.hidden, "up Q4")?;
    let down_bytes = expected_bytes(structure.hidden, structure.intermediate, "down Q3")?;
    validate_byte_len(weights.norm_q10.bytes, norm_bytes, "norm Q10")?;
    validate_byte_len(weights.gate_q4.bytes, gate_bytes, "gate Q4")?;
    validate_byte_len(weights.up_q4.bytes, up_bytes, "up Q4")?;
    validate_byte_len(weights.down_q3.bytes, down_bytes, "down Q3")?;
    validate_resource_preflight(&resource_policy, &structure, weights)?;
    validate_weight_manifest(&weight_manifest, &structure, weights)?;

    let mut norm_q10 = try_vec(structure.hidden, "norm Q10")?;
    norm_q10.extend(
        weights
            .norm_q10
            .bytes
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]])),
    );
    let owned = OwnedWeights {
        norm_bytes: try_copy(weights.norm_q10.bytes, "norm weight")?,
        norm_q10,
        gate_q4: try_copy(weights.gate_q4.bytes, "gate weight")?,
        up_q4: try_copy(weights.up_q4.bytes, "up weight")?,
        down_q3: try_copy(weights.down_q3.bytes, "down weight")?,
    };
    let composite = make_composite(
        plan,
        mode,
        layer,
        &weight_manifest,
        resource_policy,
        structure,
        &owned,
    )?;
    let binding_digest = canonical_digest(BINDING_DIGEST_DOMAIN, &composite);
    Ok(CompiledDenseQwenMlpBlock {
        composite,
        weights: owned,
        weight_manifest,
        binding_digest,
    })
}

/// Execute clear exact fixed-scale arithmetic. No protected scheduler is invoked.
pub fn execute_dense_qwen_mlp_block(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenMlpBlock,
    input_q10: &[u32],
    threads: usize,
    simd: bool,
) -> Result<Vec<i16>, String> {
    let input = validate_dense_qwen_mlp_input_stage(plan, block, input_q10)?;
    let normalized = pllm_core::rms_norm_q10_direct(&input.input_i16, block.norm_weights())
        .map_err(|error| error.to_string())?;
    let executor = pllm_core::Executor::new(threads, simd)?;
    let projections = dense_qwen_mlp_rms_to_gate_up_stage(block, &input, &normalized, &executor)?;
    let mut multiplied_q7 = try_vec(projections.gate_q7.len(), "gated Q7 output")?;
    for (gate, up) in projections.gate_q7.iter().zip(&projections.up_q7) {
        multiplied_q7
            .push(pllm_core::gated_multiply_q7(*gate, *up).map_err(|error| error.to_string())?);
    }
    dense_qwen_mlp_down_residual_stage(block, &input, &multiplied_q7, &executor)
}

pub(crate) fn validate_dense_qwen_mlp_input_stage(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenMlpBlock,
    input_q10: &[u32],
) -> Result<DenseQwenMlpInputStage, String> {
    validate_compiled_block(plan, block)?;
    let composite = &block.composite;
    let [batch, sequence, hidden] = rank3(&composite.input_shape, "MLP input")?;
    let [intermediate_batch, intermediate_sequence, intermediate] =
        rank3(&composite.intermediate_shape, "MLP intermediate")?;
    if batch != intermediate_batch || sequence != intermediate_sequence {
        return Err("dense Qwen MLP leading dimensions differ".into());
    }
    let rows = batch
        .checked_mul(sequence)
        .ok_or("dense Qwen MLP row count overflows usize")?;
    checked_rank3_rank2(input_q10.len(), rows, hidden, "MLP input")?;
    validate_activation_elements(&composite.resource_policy, input_q10.len(), "MLP input")?;
    let mut input_i16 = try_vec(input_q10.len(), "MLP input")?;
    for value in input_q10 {
        input_i16.push(
            i16::try_from(*value as i32)
                .map_err(|_| "centered wrap32 Q10 input is outside i16".to_owned())?,
        );
    }
    Ok(DenseQwenMlpInputStage {
        input_i16,
        rows,
        hidden,
        intermediate,
    })
}

pub(crate) fn dense_qwen_mlp_rms_to_gate_up_stage(
    block: &CompiledDenseQwenMlpBlock,
    input: &DenseQwenMlpInputStage,
    normalized: &[i32],
    executor: &pllm_core::Executor,
) -> Result<DenseQwenMlpGateUpStage, String> {
    let composite = &block.composite;
    checked_rank3_rank2(normalized.len(), input.rows, input.hidden, "RMSNorm output")?;
    validate_activation_elements(
        &composite.resource_policy,
        normalized.len(),
        "RMSNorm output",
    )?;
    let mut normalized_wrap = try_vec(normalized.len(), "RMSNorm wrap32 bridge")?;
    normalized_wrap.extend(normalized.iter().map(|value| *value as u32));
    let gate_q14 = checked_matrix_wrap32(
        &block.weights.gate_q4,
        input.intermediate,
        input.hidden,
        &normalized_wrap,
        input.rows,
        executor,
        "gate",
    )?;
    let up_q14 = checked_matrix_wrap32(
        &block.weights.up_q4,
        input.intermediate,
        input.hidden,
        &normalized_wrap,
        input.rows,
        executor,
        "up",
    )?;
    let gate_q7 = execute_q14_to_q7_rescale(&composite.gated_multiply.gate_rescale, &gate_q14)?;
    let up_q7 = execute_q14_to_q7_rescale(&composite.gated_multiply.up_rescale, &up_q14)?;
    checked_rank3_rank2(gate_q7.len(), input.rows, input.intermediate, "gate Q7")?;
    checked_rank3_rank2(up_q7.len(), input.rows, input.intermediate, "up Q7")?;
    validate_activation_elements(
        &composite.resource_policy,
        gate_q7.len(),
        "gated intermediate",
    )?;
    Ok(DenseQwenMlpGateUpStage { gate_q7, up_q7 })
}

pub(crate) fn dense_qwen_mlp_down_residual_stage(
    block: &CompiledDenseQwenMlpBlock,
    input: &DenseQwenMlpInputStage,
    multiplied_q7: &[i16],
    executor: &pllm_core::Executor,
) -> Result<Vec<i16>, String> {
    let composite = &block.composite;
    checked_rank3_rank2(
        multiplied_q7.len(),
        input.rows,
        input.intermediate,
        "multiplied Q7",
    )?;
    validate_activation_elements(
        &composite.resource_policy,
        multiplied_q7.len(),
        "gated output",
    )?;
    let mut multiplied_wrap = try_vec(multiplied_q7.len(), "Q7 wrap32 bridge")?;
    multiplied_wrap.extend(
        multiplied_q7
            .iter()
            .map(|value| u32::from_ne_bytes(i32::from(*value).to_ne_bytes())),
    );
    let down_q10 = checked_matrix_wrap32(
        &block.weights.down_q3,
        input.hidden,
        input.intermediate,
        &multiplied_wrap,
        input.rows,
        executor,
        "down",
    )?;
    checked_rank3_rank2(down_q10.len(), input.rows, input.hidden, "down Q10")?;
    validate_activation_elements(&composite.resource_policy, down_q10.len(), "MLP output")?;

    let mut output = try_vec(input.input_i16.len(), "MLP residual output")?;
    for (residual, down) in input.input_i16.iter().zip(down_q10) {
        let exact = i64::from(*residual) + i64::from(down as i32);
        let centered_wrap32 = u32::from_ne_bytes(
            i32::try_from(exact)
                .map_err(|_| "dense Qwen MLP residual Q10 result is outside centered wrap32")?
                .to_ne_bytes(),
        );
        output.push(
            i16::try_from(centered_wrap32 as i32)
                .map_err(|_| "dense Qwen MLP residual Q10 result is outside i16".to_owned())?,
        );
    }
    Ok(output)
}

impl CompiledDenseQwenMlpBlock {
    pub(crate) fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }

    pub(crate) fn norm_weights(&self) -> &[i16] {
        &self.weights.norm_q10
    }
}

fn lower_structure(plan: &DecoderPlan, mode: DecoderMode, layer: u64) -> Result<Structure, String> {
    plan.validate().map_err(|error| error.to_string())?;
    // Family and adapter labels guard immutable plan origin only. Semantic edge/operator/shape
    // checks below determine eligibility; labels do not authenticate weight provenance.
    if !matches!(
        (plan.model_family.as_str(), plan.adapter.as_str()),
        ("qwen2", "pllm.qwen2.v1") | ("qwen3", "pllm.qwen3.v1")
    ) {
        return Err("dense Qwen MLP requires the Qwen2 or Qwen3 decoder adapter".into());
    }
    let graph = model_graph(plan, mode);
    let multiply = exactly_one(
        graph.operations.iter().filter(|operation| {
            operation.layer == Some(layer) && operation.operator == ModelOperator::Multiply
        }),
        "same-layer multiply",
    )?;
    let elements = tensor_elements(&multiply.output_shape)?;
    if elements > GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS {
        return Err("dense Qwen MLP intermediate exceeds chunked schedule bound".into());
    }
    let gated_multiply = exactly_one(
        lower_model_gated_multiply_q7_regions_with_components(
            plan,
            mode,
            GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
            GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
            elements,
        )?
        .into_iter()
        .filter(|region| region.layer == layer && region.multiply_operation_id == multiply.id),
        "same-layer gated multiply region",
    )?
    .clone();
    let gate_operation = operation(
        graph.operations.as_slice(),
        &gated_multiply.gate_linear_operation_id,
    )?;
    let up_operation = operation(
        graph.operations.as_slice(),
        &gated_multiply.up_linear_operation_id,
    )?;
    let [gate_input] = gate_operation.inputs.as_slice() else {
        return Err("dense Qwen MLP gate linear must have one input".into());
    };
    let [up_input] = up_operation.inputs.as_slice() else {
        return Err("dense Qwen MLP up linear must have one input".into());
    };
    if gate_input != up_input {
        return Err("dense Qwen MLP gate and up linears must share one RMSNorm input".into());
    }
    let norm_operation = operation(graph.operations.as_slice(), gate_input)?;
    if norm_operation.operator != ModelOperator::RmsNorm || norm_operation.layer != Some(layer) {
        return Err("dense Qwen MLP gate/up producer must be same-layer RMSNorm".into());
    }
    let [residual_input] = norm_operation.inputs.as_slice() else {
        return Err("dense Qwen MLP RMSNorm must have one input".into());
    };
    let down_operation = exactly_one(
        graph.operations.iter().filter(|candidate| {
            candidate.layer == Some(layer)
                && candidate.operator == ModelOperator::Linear
                && candidate.inputs.as_slice() == [multiply.id.as_str()]
        }),
        "same-layer down linear",
    )?;
    let residual_operation = exactly_one(
        graph.operations.iter().filter(|candidate| {
            candidate.layer == Some(layer)
                && candidate.operator == ModelOperator::ResidualAdd
                && candidate.inputs.len() == 2
                && candidate
                    .inputs
                    .iter()
                    .any(|input| input == &down_operation.id)
                && candidate.inputs.iter().any(|input| input == residual_input)
        }),
        "same-layer MLP residual",
    )?;
    let gate = lower_model_linear_region(graph, mode, gate_operation)?;
    let up = lower_model_linear_region(graph, mode, up_operation)?;
    let down = lower_model_linear_region(graph, mode, down_operation)?;
    if gate.bias_id.is_some() || up.bias_id.is_some() || down.bias_id.is_some() {
        return Err("dense Qwen MLP linears do not support bias".into());
    }
    let norm = lower_rms_norm_q10_direct_region(&plan.digest(), graph, mode, norm_operation)?;
    let residual = lower_model_residual_region(graph, mode, residual_operation)?;
    let [batch, sequence, hidden] = rank3_u64(&norm.shape, "dense Qwen MLP input")?;
    let [gate_batch, gate_sequence, intermediate] =
        rank3_u64(&gate_operation.output_shape, "dense Qwen MLP intermediate")?;
    if batch != graph.batch
        || sequence != graph.query_sequence
        || gate_batch != batch
        || gate_sequence != sequence
        || up_operation.output_shape != gate_operation.output_shape
        || down_operation.output_shape != norm.shape
        || residual_operation.output_shape != norm.shape
        || gated_multiply.input.shape != gate_operation.output_shape
    {
        return Err("dense Qwen MLP graph has incompatible rank-3 shapes".into());
    }
    Ok(Structure {
        input_shape: norm.shape.clone(),
        intermediate_shape: gate_operation.output_shape.clone(),
        hidden: usize::try_from(hidden).map_err(|_| "hidden width exceeds usize")?,
        intermediate: usize::try_from(intermediate)
            .map_err(|_| "intermediate width exceeds usize")?,
        norm,
        gate,
        up,
        gated_multiply,
        down,
        residual,
    })
}

fn make_composite(
    plan: &DecoderPlan,
    mode: DecoderMode,
    layer: u64,
    weight_manifest: &DenseQwenMlpWeightManifest,
    resource_policy: DenseQwenMlpResourcePolicy,
    structure: Structure,
    weights: &OwnedWeights,
) -> Result<DenseQwenMlpComposite, String> {
    let hidden = u64::try_from(structure.hidden).map_err(|_| "hidden width exceeds u64")?;
    let intermediate =
        u64::try_from(structure.intermediate).map_err(|_| "intermediate width exceeds u64")?;
    Ok(DenseQwenMlpComposite {
        schema_version: DENSE_QWEN_MLP_BLOCK_SCHEMA_VERSION.into(),
        numeric_profile: DENSE_QWEN_MLP_CLEAR_PROFILE.into(),
        model_plan_digest: plan.digest(),
        weight_manifest_digest: weight_manifest.digest(),
        semantic_guard_model_family: plan.model_family.clone(),
        semantic_guard_adapter: plan.adapter.clone(),
        resource_policy,
        mode,
        layer,
        input_shape: structure.input_shape,
        intermediate_shape: structure.intermediate_shape,
        norm_weight: artifact(
            &structure.norm.weight_id,
            &weights.norm_bytes,
            &[hidden],
            DenseQwenMlpElementType::SignedI16,
            10,
            DenseQwenMlpLayout::RowMajorVector,
        )?,
        gate_weight: artifact(
            &structure.gate.weight_id,
            &weights.gate_q4,
            &[intermediate, hidden],
            DenseQwenMlpElementType::SignedI8,
            4,
            DenseQwenMlpLayout::RowMajorOutputInput,
        )?,
        up_weight: artifact(
            &structure.up.weight_id,
            &weights.up_q4,
            &[intermediate, hidden],
            DenseQwenMlpElementType::SignedI8,
            4,
            DenseQwenMlpLayout::RowMajorOutputInput,
        )?,
        down_weight: artifact(
            &structure.down.weight_id,
            &weights.down_q3,
            &[hidden, intermediate],
            DenseQwenMlpElementType::SignedI8,
            3,
            DenseQwenMlpLayout::RowMajorOutputInput,
        )?,
        norm: structure.norm,
        gate: structure.gate,
        up: structure.up,
        gated_multiply: structure.gated_multiply,
        down: structure.down,
        residual: structure.residual,
        residual_refinement: DenseQwenMlpResidualRefinement {
            output_fractional_bits: 10,
            storage: DenseQwenMlpResidualStorage::CenteredWrap32,
            output_range_policy:
                DenseQwenMlpResidualOutputRangePolicy::RejectOutsideSignedI16ForNextRmsNorm,
        },
        protected_execution: false,
        complete_decoder: false,
    })
}

pub(crate) fn validate_compiled_block(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenMlpBlock,
) -> Result<(), String> {
    if canonical_digest(BINDING_DIGEST_DOMAIN, &block.composite) != block.binding_digest {
        return Err("dense Qwen MLP composite binding was modified".into());
    }
    block.composite.resource_policy.validate()?;
    if block.weight_manifest.digest() != block.composite.weight_manifest_digest {
        return Err("dense Qwen MLP weight manifest digest differs from its binding".into());
    }
    let structure = lower_structure(plan, block.composite.mode, block.composite.layer)?;
    let weights = DenseQwenMlpWeights {
        norm_q10: DenseQwenMlpWeightBytes {
            weight_id: &block.composite.norm_weight.weight_id,
            bytes: &block.weights.norm_bytes,
        },
        gate_q4: DenseQwenMlpWeightBytes {
            weight_id: &block.composite.gate_weight.weight_id,
            bytes: &block.weights.gate_q4,
        },
        up_q4: DenseQwenMlpWeightBytes {
            weight_id: &block.composite.up_weight.weight_id,
            bytes: &block.weights.up_q4,
        },
        down_q3: DenseQwenMlpWeightBytes {
            weight_id: &block.composite.down_weight.weight_id,
            bytes: &block.weights.down_q3,
        },
    };
    validate_resource_preflight(&block.composite.resource_policy, &structure, weights)?;
    validate_weight_manifest(&block.weight_manifest, &structure, weights)?;
    let authentic = make_composite(
        plan,
        block.composite.mode,
        block.composite.layer,
        &block.weight_manifest,
        block.composite.resource_policy.clone(),
        structure,
        &block.weights,
    )?;
    if authentic != block.composite {
        return Err("dense Qwen MLP composite differs from plan or weight bytes".into());
    }
    Ok(())
}

fn checked_matrix_wrap32(
    weights: &[u8],
    output_width: usize,
    input_width: usize,
    input: &[u32],
    rows: usize,
    executor: &pllm_core::Executor,
    name: &str,
) -> Result<Vec<u32>, String> {
    checked_rank3_rank2(input.len(), rows, input_width, name)?;
    let max_abs = input
        .iter()
        .map(|value| i64::from(*value as i32).unsigned_abs())
        .max()
        .unwrap_or(0);
    for row in weights.chunks_exact(input_width) {
        let row_sum = row.iter().try_fold(0_u64, |sum, weight| {
            sum.checked_add(u64::from(i8::from_ne_bytes([*weight]).unsigned_abs()))
                .ok_or_else(|| format!("{name} weight row absolute sum overflows u64"))
        })?;
        if max_abs
            .checked_mul(row_sum)
            .is_none_or(|bound| bound > i32::MAX as u64)
        {
            return Err(format!(
                "{name} linear signed dot-product bound exceeds i32 before wrap32"
            ));
        }
    }
    pllm_core::Matrix::new(weights, output_width, input_width)?.wrap32(executor, input, rows)
}

fn artifact(
    weight_id: &str,
    bytes: &[u8],
    shape: &[u64],
    element_type: DenseQwenMlpElementType,
    fractional_bits: u8,
    layout: DenseQwenMlpLayout,
) -> Result<DenseQwenMlpWeightArtifact, String> {
    Ok(DenseQwenMlpWeightArtifact {
        weight_id: weight_id.into(),
        data_digest: digest_bytes(WEIGHT_DIGEST_DOMAIN, bytes),
        shape: try_copy(shape, "weight artifact shape")?,
        element_type,
        fractional_bits,
        layout,
        rounding: FixedPointRounding::TiesToEven,
        range_policy: DenseQwenMlpRangePolicy::RejectNoSaturation,
    })
}

fn validate_weight_id(
    weight: DenseQwenMlpWeightBytes<'_>,
    expected: &str,
    name: &str,
) -> Result<(), String> {
    if weight.weight_id != expected {
        return Err(format!(
            "dense Qwen MLP {name} weight identity differs from plan"
        ));
    }
    Ok(())
}

fn validate_weight_manifest(
    manifest: &DenseQwenMlpWeightManifest,
    structure: &Structure,
    weights: DenseQwenMlpWeights<'_>,
) -> Result<(), String> {
    if manifest.schema_version != DENSE_QWEN_MLP_WEIGHT_MANIFEST_SCHEMA_VERSION
        || manifest.entries.len() != 4
    {
        return Err("dense Qwen MLP weight manifest contract is invalid".into());
    }
    for (weight, expected_id) in [
        (weights.norm_q10, structure.norm.weight_id.as_str()),
        (weights.gate_q4, structure.gate.weight_id.as_str()),
        (weights.up_q4, structure.up.weight_id.as_str()),
        (weights.down_q3, structure.down.weight_id.as_str()),
    ] {
        if weight.weight_id != expected_id
            || manifest.entries.get(expected_id)
                != Some(&digest_bytes(WEIGHT_DIGEST_DOMAIN, weight.bytes))
        {
            return Err(
                "dense Qwen MLP supplied weight bytes differ from their manifest entry".into(),
            );
        }
    }
    Ok(())
}

fn validate_resource_preflight(
    policy: &DenseQwenMlpResourcePolicy,
    structure: &Structure,
    weights: DenseQwenMlpWeights<'_>,
) -> Result<(), String> {
    policy.validate()?;
    let total_weight_bytes = [
        weights.norm_q10.bytes.len(),
        weights.gate_q4.bytes.len(),
        weights.up_q4.bytes.len(),
        weights.down_q3.bytes.len(),
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        total.checked_add(u64::try_from(bytes).ok()?)
    })
    .ok_or("dense Qwen MLP total weight byte count overflows u64")?;
    if total_weight_bytes > policy.max_total_weight_bytes {
        return Err("dense Qwen MLP total weight bytes exceed resource policy".into());
    }
    for (shape, name) in [
        (structure.input_shape.as_slice(), "input/output"),
        (
            structure.intermediate_shape.as_slice(),
            "intermediate activation",
        ),
    ] {
        validate_activation_elements(policy, tensor_elements(shape)?, name)?;
    }
    Ok(())
}

fn validate_activation_elements(
    policy: &DenseQwenMlpResourcePolicy,
    elements: usize,
    name: &str,
) -> Result<(), String> {
    if u64::try_from(elements)
        .map_err(|_| format!("dense Qwen MLP {name} element count exceeds u64"))?
        > policy.max_activation_elements
    {
        return Err(format!(
            "dense Qwen MLP {name} elements exceed resource policy"
        ));
    }
    Ok(())
}

fn try_vec<T>(capacity: usize, name: &str) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| format!("dense Qwen MLP {name} allocation failed"))?;
    Ok(values)
}

fn try_copy<T: Clone>(values: &[T], name: &str) -> Result<Vec<T>, String> {
    let mut copy = try_vec(values.len(), name)?;
    copy.extend_from_slice(values);
    Ok(copy)
}

fn expected_bytes(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("dense Qwen MLP {name} byte length overflows usize"))
}

fn validate_byte_len(bytes: &[u8], expected: usize, name: &str) -> Result<(), String> {
    if bytes.len() != expected {
        return Err(format!(
            "dense Qwen MLP {name} requires {expected} bytes, received {}",
            bytes.len()
        ));
    }
    Ok(())
}

fn operation<'a>(operations: &'a [ModelOperation], id: &str) -> Result<&'a ModelOperation, String> {
    operations
        .iter()
        .find(|operation| operation.id == id)
        .ok_or_else(|| format!("dense Qwen MLP references missing semantic input {id}"))
}

fn exactly_one<T>(mut values: impl Iterator<Item = T>, description: &str) -> Result<T, String> {
    let value = values
        .next()
        .ok_or_else(|| format!("dense Qwen MLP has no {description}"))?;
    if values.next().is_some() {
        return Err(format!(
            "dense Qwen MLP has multiple {description} candidates"
        ));
    }
    Ok(value)
}

fn rank3(shape: &[u64], name: &str) -> Result<[usize; 3], String> {
    let [first, second, third] = rank3_u64(shape, name)?;
    Ok([
        usize::try_from(first).map_err(|_| format!("{name} dimension exceeds usize"))?,
        usize::try_from(second).map_err(|_| format!("{name} dimension exceeds usize"))?,
        usize::try_from(third).map_err(|_| format!("{name} dimension exceeds usize"))?,
    ])
}

fn rank3_u64(shape: &[u64], name: &str) -> Result<[u64; 3], String> {
    let [first, second, third] = shape else {
        return Err(format!("{name} must be rank 3"));
    };
    if *first == 0 || *second == 0 || *third == 0 {
        return Err(format!("{name} has a zero dimension"));
    }
    Ok([*first, *second, *third])
}

fn checked_rank3_rank2(
    elements: usize,
    rows: usize,
    width: usize,
    name: &str,
) -> Result<(), String> {
    if rows.checked_mul(width) != Some(elements) {
        return Err(format!("{name} rank-3/rank-2 row-major view is invalid"));
    }
    Ok(())
}
