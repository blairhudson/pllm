use super::dense_qwen_mlp::{
    dense_qwen_mlp_down_residual_stage, dense_qwen_mlp_rms_to_gate_up_stage,
    validate_compiled_block, validate_dense_qwen_mlp_input_stage,
};
use super::gated_tensor::{
    preflight_bound_gated_multiply_q7_tensor_material,
    prepare_bound_gated_multiply_q7_tensor_material_with_preflight,
};
use super::rms_norm_stream_protected::{
    preflight_rms_norm_q10_stream_registry, rms_norm_q10_stream_body_bytes,
};
use super::{
    canonical_digest, prepare_bound_rms_norm_q10_stream_row, BoundGatedMultiplyQ7TensorMaterial,
    BoundRmsNormQ10StreamEvaluator, BoundRmsNormQ10StreamMaterial, CompiledDenseQwenMlpBlock,
    Digest, GatedMultiplyQ7TensorEvaluator, RmsNormQ10StreamResourcePolicy, TensorResourcePolicy,
};
use pllm_garble::boolean_stream::{RMS_NORM_Q10_STREAM_METHOD_ID, RMS_NORM_Q10_STREAM_TOPOLOGY_ID};
use pllm_models::DecoderPlan;
use serde::Serialize;
use std::io::{Read, Write};

const PROTECTED_NONLINEAR_BINDING_SCHEMA_VERSION: &str =
    "pllm.dense_qwen_mlp.protected_nonlinear_binding.v2";
const PROTECTED_NONLINEAR_BINDING_DIGEST_DOMAIN: &str =
    "pllm.dense_qwen_mlp.protected_nonlinear_binding.digest.v2";
const ROW_ORDER: &str = "batch_sequence_row_major";
pub const DENSE_QWEN_MLP_PROTECTED_HARD_MAX_ROWS: u64 = 64;
pub const DENSE_QWEN_MLP_PROTECTED_HARD_MAX_TOTAL_BODY_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseQwenMlpProtectedResourcePolicy {
    pub max_rows: u64,
    pub max_total_body_bytes: u64,
}

impl DenseQwenMlpProtectedResourcePolicy {
    fn validate(&self) -> Result<(), String> {
        if self.max_rows == 0
            || self.max_rows > DENSE_QWEN_MLP_PROTECTED_HARD_MAX_ROWS
            || self.max_total_body_bytes == 0
            || self.max_total_body_bytes > DENSE_QWEN_MLP_PROTECTED_HARD_MAX_TOTAL_BODY_BYTES
        {
            return Err("protected nonlinear MLP aggregate resource policy is invalid".into());
        }
        Ok(())
    }
}

/// Explicit acknowledgement that this composes unreviewed reference cryptography and remains
/// experimental. It does not approve protected linear layers or full-block production security.
#[derive(Clone, Copy)]
pub struct ExperimentalDenseQwenMlpProtectedNonlinearApproval {
    _private: (),
}

impl ExperimentalDenseQwenMlpProtectedNonlinearApproval {
    pub fn acknowledge_unreviewed_experimental_components() -> Self {
        Self { _private: () }
    }
}

/// Immutable framing and component binding for one aggregate one-use material.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseQwenMlpProtectedNonlinearBinding {
    schema_version: String,
    block_binding_digest: Digest,
    weight_manifest_digest: Digest,
    rms_method_id: String,
    rms_topology_id: String,
    rms_artifact_digest: Digest,
    gated_method_component_id: String,
    gated_schedule_component_id: String,
    row_order: String,
    row_count: usize,
    rms_body_bytes: Vec<u64>,
    gated_body_bytes: u64,
    total_body_bytes: u64,
    rms_max_body_bytes: u64,
    tensor_policy: TensorResourcePolicy,
    aggregate_policy: DenseQwenMlpProtectedResourcePolicy,
}

impl DenseQwenMlpProtectedNonlinearBinding {
    pub fn digest(&self) -> Digest {
        canonical_digest(PROTECTED_NONLINEAR_BINDING_DIGEST_DOMAIN, self)
    }

    pub fn total_body_bytes(&self) -> u64 {
        self.total_body_bytes
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn row_order(&self) -> &str {
        &self.row_order
    }

    pub fn rms_body_bytes(&self, row: usize) -> Option<u64> {
        self.rms_body_bytes.get(row).copied()
    }

    pub fn gated_body_bytes(&self) -> u64 {
        self.gated_body_bytes
    }

    pub fn resource_policy(&self) -> &DenseQwenMlpProtectedResourcePolicy {
        &self.aggregate_policy
    }
}

/// Opaque process-local material for streamed protected nonlinear stages only.
pub struct BoundDenseQwenMlpProtectedMaterial {
    binding: DenseQwenMlpProtectedNonlinearBinding,
    binding_digest: Digest,
    rms_materials: Option<Vec<BoundRmsNormQ10StreamMaterial>>,
    gated_material: Option<BoundGatedMultiplyQ7TensorMaterial>,
}

pub struct DenseQwenMlpProtectedNonlinearExecution<'a> {
    pub rms_policy: &'a RmsNormQ10StreamResourcePolicy,
    pub tensor_policy: &'a TensorResourcePolicy,
    pub aggregate_policy: &'a DenseQwenMlpProtectedResourcePolicy,
    pub body_bytes: u64,
    pub input_q10: &'a [u32],
    pub threads: usize,
    pub simd: bool,
}

impl BoundDenseQwenMlpProtectedMaterial {
    pub fn binding(&self) -> &DenseQwenMlpProtectedNonlinearBinding {
        &self.binding
    }
}

/// Publish row-major RMSNorm bodies followed by one chunked gated body. On any error, caller must
/// discard sink contents; dropping local materials cancels every issuance already published.
pub fn prepare_bound_dense_qwen_mlp_protected_nonlinear<W: Write>(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenMlpBlock,
    _approval: &ExperimentalDenseQwenMlpProtectedNonlinearApproval,
    rms_policy: &RmsNormQ10StreamResourcePolicy,
    tensor_policy: &TensorResourcePolicy,
    aggregate_policy: &DenseQwenMlpProtectedResourcePolicy,
    writer: &mut W,
) -> Result<BoundDenseQwenMlpProtectedMaterial, String> {
    validate_compiled_block(plan, block)?;
    aggregate_policy.validate()?;
    let [batch, sequence, _hidden] = rank3(&block.composite.input_shape)?;
    let row_count = batch
        .checked_mul(sequence)
        .ok_or("protected nonlinear MLP row count overflows usize")?;
    let row_count_u64 =
        u64::try_from(row_count).map_err(|_| "protected nonlinear MLP row count exceeds u64")?;
    if row_count_u64 > aggregate_policy.max_rows
        || row_count_u64 > DENSE_QWEN_MLP_PROTECTED_HARD_MAX_ROWS
    {
        return Err("protected nonlinear MLP row count exceeds its aggregate bound".into());
    }
    let rms_body_bytes_per_row = rms_norm_q10_stream_body_bytes(block.norm_weights(), rms_policy)?;
    let total_rms_body_bytes = rms_body_bytes_per_row
        .checked_mul(row_count_u64)
        .ok_or("protected nonlinear MLP RMS body total overflows u64")?;
    let conservative_total_body_bytes = total_rms_body_bytes
        .checked_add(tensor_policy.max_body_bytes)
        .ok_or("protected nonlinear MLP conservative body total overflows u64")?;
    if conservative_total_body_bytes > aggregate_policy.max_total_body_bytes
        || conservative_total_body_bytes > DENSE_QWEN_MLP_PROTECTED_HARD_MAX_TOTAL_BODY_BYTES
    {
        return Err("protected nonlinear MLP body total exceeds its aggregate bound".into());
    }
    preflight_rms_norm_q10_stream_registry(row_count, total_rms_body_bytes)?;
    let gated_preflight = preflight_bound_gated_multiply_q7_tensor_material(
        plan,
        &block.composite.gated_multiply,
        tensor_policy,
    )?;
    let mut rms_materials = try_vec(row_count, "RMSNorm material")?;
    let mut rms_body_bytes = try_vec(row_count, "RMSNorm body metadata")?;
    let rms_method_id = try_string(RMS_NORM_Q10_STREAM_METHOD_ID, "RMSNorm method ID")?;
    let rms_topology_id = try_string(RMS_NORM_Q10_STREAM_TOPOLOGY_ID, "RMSNorm topology ID")?;
    let gated_method_component_id = try_string(
        &block.composite.gated_multiply.method_component_id,
        "gated method component ID",
    )?;
    let gated_schedule_component_id = try_string(
        &block.composite.gated_multiply.schedule_component_id,
        "gated schedule component ID",
    )?;
    let schema_version = try_string(
        PROTECTED_NONLINEAR_BINDING_SCHEMA_VERSION,
        "binding schema version",
    )?;
    let row_order = try_string(ROW_ORDER, "row order")?;
    let mut total_body_bytes = 0_u64;

    for row in 0..row_count {
        let material = prepare_bound_rms_norm_q10_stream_row(
            plan,
            &block.composite.norm,
            row,
            block.norm_weights(),
            rms_policy,
            writer,
        )
        .map_err(|error| {
            format!(
                "protected nonlinear MLP RMSNorm publication failed; discard sink contents: {error}"
            )
        })?;
        let body_bytes = material.ticket().body_bytes();
        if body_bytes != rms_body_bytes_per_row {
            return Err(
                "protected nonlinear MLP RMS body differs from its estimator; discard sink contents"
                    .into(),
            );
        }
        total_body_bytes = total_body_bytes.checked_add(body_bytes).ok_or(
            "protected nonlinear MLP total body length overflows u64; discard sink contents",
        )?;
        rms_body_bytes.push(body_bytes);
        rms_materials.push(material);
    }

    let gated_material = prepare_bound_gated_multiply_q7_tensor_material_with_preflight(
        &block.composite.gated_multiply,
        tensor_policy.clone(),
        gated_preflight,
        writer,
    )
    .map_err(|error| {
        format!("protected nonlinear MLP gated publication failed; discard sink contents: {error}")
    })?;
    let gated_body_bytes = gated_material.ticket().body_bytes();
    total_body_bytes = total_body_bytes
        .checked_add(gated_body_bytes)
        .ok_or("protected nonlinear MLP total body length overflows u64; discard sink contents")?;
    if total_body_bytes > aggregate_policy.max_total_body_bytes {
        return Err(
            "protected nonlinear MLP published body exceeds its aggregate bound; discard sink contents"
                .into(),
        );
    }
    let binding = DenseQwenMlpProtectedNonlinearBinding {
        schema_version,
        block_binding_digest: block.binding_digest().clone(),
        weight_manifest_digest: block.composite.weight_manifest_digest.clone(),
        rms_method_id,
        rms_topology_id,
        rms_artifact_digest: super::protected_rms_norm_q10_artifact_digest(),
        gated_method_component_id,
        gated_schedule_component_id,
        row_order,
        row_count,
        rms_body_bytes,
        gated_body_bytes,
        total_body_bytes,
        rms_max_body_bytes: rms_policy.max_body_bytes(),
        tensor_policy: tensor_policy.clone(),
        aggregate_policy: aggregate_policy.clone(),
    };
    let binding_digest = binding.digest();
    Ok(BoundDenseQwenMlpProtectedMaterial {
        binding,
        binding_digest,
        rms_materials: Some(rms_materials),
        gated_material: Some(gated_material),
    })
}

/// Consume one aggregate protected-nonlinear frame. Clear linears remain exact local arithmetic;
/// only RMSNorm and gated multiplication use streamed protected reference components.
pub fn execute_bound_dense_qwen_mlp_protected_nonlinear<R: Read>(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenMlpBlock,
    material: &mut BoundDenseQwenMlpProtectedMaterial,
    reader: &mut R,
    execution: DenseQwenMlpProtectedNonlinearExecution<'_>,
) -> Result<Vec<i16>, String> {
    let input = validate_dense_qwen_mlp_input_stage(plan, block, execution.input_q10)?;
    validate_material_binding(
        block,
        material,
        execution.rms_policy,
        execution.tensor_policy,
        execution.aggregate_policy,
        execution.body_bytes,
    )?;
    if input.rows != material.binding.row_count {
        return Err("protected nonlinear MLP input row count differs from material".into());
    }
    let rms_materials = material
        .rms_materials
        .take()
        .ok_or("protected nonlinear MLP material was already consumed")?;
    let gated_material = material
        .gated_material
        .take()
        .ok_or("protected nonlinear MLP gated material was already consumed")?;
    let mut normalized = try_vec(input.input_i16.len(), "normalized activation")?;

    for (row, rms_material) in rms_materials.into_iter().enumerate() {
        let start = row
            .checked_mul(input.hidden)
            .ok_or("protected nonlinear MLP row offset overflows usize")?;
        let end = start
            .checked_add(input.hidden)
            .ok_or("protected nonlinear MLP row end overflows usize")?;
        let body_bytes = material.binding.rms_body_bytes[row];
        let (ticket, encoded, decoder) = rms_material.encode(&input.input_i16[start..end])?;
        let evaluator = BoundRmsNormQ10StreamEvaluator::claim(
            plan,
            &block.composite.norm,
            row,
            block.norm_weights(),
            execution.rms_policy,
            ticket,
            encoded,
        )?;
        let outputs = evaluator.evaluate_with_reader(|| Ok(reader.by_ref().take(body_bytes)))?;
        let row_output = decoder.decode(outputs)?;
        if row_output.len() != input.hidden {
            return Err("protected nonlinear MLP RMSNorm row output width differs".into());
        }
        normalized.extend(row_output);
    }

    let executor = pllm_core::Executor::new(execution.threads, execution.simd)?;
    let projections = dense_qwen_mlp_rms_to_gate_up_stage(block, &input, &normalized, &executor)?;
    let gate_labels = gated_material
        .encode_gates(&projections.gate_q7)
        .map_err(|error| error.to_string())?;
    let up_labels = gated_material
        .encode_ups(&projections.up_q7)
        .map_err(|error| error.to_string())?;
    let mut evaluator = GatedMultiplyQ7TensorEvaluator::claim(
        &block.composite.gated_multiply,
        gated_material.ticket(),
        execution.tensor_policy,
    )?;
    let mut gated_frame = reader.by_ref().take(material.binding.gated_body_bytes);
    let output_labels = evaluator.evaluate(&mut gated_frame, &gate_labels, &up_labels)?;
    let multiplied_q7 = gated_material
        .decode_tensor(&output_labels)
        .map_err(|error| error.to_string())?;
    ensure_eof(reader)?;
    dense_qwen_mlp_down_residual_stage(block, &input, &multiplied_q7, &executor)
}

fn validate_material_binding(
    block: &CompiledDenseQwenMlpBlock,
    material: &BoundDenseQwenMlpProtectedMaterial,
    rms_policy: &RmsNormQ10StreamResourcePolicy,
    tensor_policy: &TensorResourcePolicy,
    aggregate_policy: &DenseQwenMlpProtectedResourcePolicy,
    frame_body_bytes: u64,
) -> Result<(), String> {
    aggregate_policy.validate()?;
    let binding = &material.binding;
    let [batch, sequence, _hidden] = rank3(&block.composite.input_shape)?;
    let row_count = batch
        .checked_mul(sequence)
        .ok_or("protected nonlinear MLP row count overflows usize")?;
    let rms_materials = material
        .rms_materials
        .as_ref()
        .ok_or("protected nonlinear MLP material was already consumed")?;
    let gated_material = material
        .gated_material
        .as_ref()
        .ok_or("protected nonlinear MLP material was already consumed")?;
    let summed = binding
        .rms_body_bytes
        .iter()
        .try_fold(0_u64, |total, bytes| total.checked_add(*bytes))
        .and_then(|total| total.checked_add(binding.gated_body_bytes))
        .ok_or("protected nonlinear MLP total body length overflows u64")?;
    let tickets_match = rms_materials.len() == row_count
        && binding.rms_body_bytes.len() == row_count
        && binding
            .rms_body_bytes
            .iter()
            .all(|bytes| *bytes <= binding.rms_max_body_bytes)
        && rms_materials
            .iter()
            .zip(&binding.rms_body_bytes)
            .all(|(material, bytes)| material.ticket().body_bytes() == *bytes)
        && gated_material.ticket().body_bytes() == binding.gated_body_bytes
        && binding.gated_body_bytes <= binding.tensor_policy.max_body_bytes;
    if binding.schema_version != PROTECTED_NONLINEAR_BINDING_SCHEMA_VERSION
        || binding.digest() != material.binding_digest
        || binding.block_binding_digest != *block.binding_digest()
        || binding.weight_manifest_digest != block.composite.weight_manifest_digest
        || binding.rms_method_id != RMS_NORM_Q10_STREAM_METHOD_ID
        || binding.rms_topology_id != RMS_NORM_Q10_STREAM_TOPOLOGY_ID
        || binding.rms_artifact_digest != super::protected_rms_norm_q10_artifact_digest()
        || binding.gated_method_component_id != block.composite.gated_multiply.method_component_id
        || binding.gated_schedule_component_id
            != block.composite.gated_multiply.schedule_component_id
        || binding.row_order != ROW_ORDER
        || binding.row_count != row_count
        || u64::try_from(binding.row_count).map_err(|_| "row count exceeds u64")?
            > aggregate_policy.max_rows
        || binding.rms_max_body_bytes != rms_policy.max_body_bytes()
        || binding.tensor_policy != *tensor_policy
        || binding.aggregate_policy != *aggregate_policy
        || binding.total_body_bytes != summed
        || binding.total_body_bytes > aggregate_policy.max_total_body_bytes
        || binding.total_body_bytes != frame_body_bytes
        || !tickets_match
    {
        return Err("protected nonlinear MLP material binding or policy differs".into());
    }
    Ok(())
}

fn rank3(shape: &[u64]) -> Result<[usize; 3], String> {
    let [batch, sequence, hidden] = shape else {
        return Err("protected nonlinear MLP input must be rank 3".into());
    };
    Ok([
        usize::try_from(*batch).map_err(|_| "protected nonlinear MLP batch exceeds usize")?,
        usize::try_from(*sequence).map_err(|_| "protected nonlinear MLP sequence exceeds usize")?,
        usize::try_from(*hidden)
            .map_err(|_| "protected nonlinear MLP hidden width exceeds usize")?,
    ])
}

fn try_vec<T>(capacity: usize, name: &str) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| format!("protected nonlinear MLP {name} allocation failed"))?;
    Ok(values)
}

fn try_string(value: &str, name: &str) -> Result<String, String> {
    let mut string = String::new();
    string
        .try_reserve_exact(value.len())
        .map_err(|_| format!("protected nonlinear MLP {name} allocation failed"))?;
    string.push_str(value);
    Ok(string)
}

fn ensure_eof<R: Read>(reader: &mut R) -> Result<(), String> {
    let mut byte = [0_u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err("protected nonlinear MLP frame has trailing bytes".into()),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.to_string()),
        }
    }
}

#[cfg(test)]
#[path = "dense_qwen_mlp_protected_registry_tests.rs"]
mod registry_tests;
