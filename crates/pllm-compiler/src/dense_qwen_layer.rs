use serde::Serialize;
use zeroize::{Zeroize, Zeroizing};

use super::{
    canonical_digest, execute_dense_qwen_attention_decode, execute_dense_qwen_attention_prefill,
    execute_dense_qwen_mlp_block, CompiledDenseQwenAttentionBlock, CompiledDenseQwenMlpBlock,
    DecoderMode, DecoderPlan, DenseQwenAttentionComposite, DenseQwenAttentionExecutionPolicy,
    DenseQwenAttentionOutputQ10, DenseQwenAttentionState, DenseQwenMlpComposite,
    DenseQwenMlpResidualOutputRangePolicy, DenseQwenMlpResidualStorage, Digest,
    DENSE_QWEN_ATTENTION_CLEAR_PROFILE, DENSE_QWEN_MLP_CLEAR_PROFILE,
};

pub const DENSE_QWEN_LAYER_SCHEMA_VERSION: &str = "pllm.dense_qwen_layer.v1";
pub const DENSE_QWEN_LAYER_CLEAR_PROFILE: &str = "pllm.clear_exact.dense_qwen_layer.q10_q4_q7.v1";
const DENSE_QWEN_LAYER_BINDING_DOMAIN: &str = "pllm.dense_qwen_layer.binding.v1";
const DENSE_QWEN_LAYER_STATE_DOMAIN: &str = "pllm.dense_qwen_layer.state.v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseQwenLayerComposite {
    pub schema_version: String,
    pub numeric_profile: String,
    pub model_plan_digest: Digest,
    pub mode: DecoderMode,
    pub layer: u64,
    pub input_shape: Vec<u64>,
    pub output_shape: Vec<u64>,
    pub attention: DenseQwenAttentionComposite,
    pub mlp: DenseQwenMlpComposite,
    pub protected_execution: bool,
    pub complete_decoder: bool,
}

pub struct CompiledDenseQwenLayerBlock {
    pub composite: DenseQwenLayerComposite,
    attention: CompiledDenseQwenAttentionBlock,
    mlp: CompiledDenseQwenMlpBlock,
    binding_digest: Digest,
}

impl CompiledDenseQwenLayerBlock {
    pub fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }
}

fn require_rank3(shape: &[u64], role: &str) -> Result<(), String> {
    if shape.len() != 3 || shape.contains(&0) {
        return Err(format!(
            "dense-qwen layer {role} must be a nonzero rank-3 shape"
        ));
    }
    Ok(())
}

fn build_layer_composite(
    plan: &DecoderPlan,
    attention: &CompiledDenseQwenAttentionBlock,
    mlp: &CompiledDenseQwenMlpBlock,
) -> Result<DenseQwenLayerComposite, String> {
    plan.validate().map_err(|error| error.to_string())?;
    if plan.model_family != "qwen2" {
        return Err(format!(
            "dense-qwen layer requires model_family qwen2, found {}",
            plan.model_family
        ));
    }
    if !plan.transformations.is_empty() {
        return Err("dense-qwen layer does not support transformed plans".into());
    }
    let attention = &attention.composite;
    let mlp = &mlp.composite;
    if attention.model_plan_digest != plan.digest() || mlp.model_plan_digest != plan.digest() {
        return Err("dense-qwen layer blocks must bind the supplied decoder plan".into());
    }
    if attention.mode != mlp.mode || attention.layer != mlp.layer {
        return Err("dense-qwen layer blocks must share one mode and layer".into());
    }
    if attention.numeric_profile != DENSE_QWEN_ATTENTION_CLEAR_PROFILE {
        return Err("dense-qwen layer requires the clear dense-qwen attention profile".into());
    }
    if mlp.numeric_profile != DENSE_QWEN_MLP_CLEAR_PROFILE {
        return Err("dense-qwen layer requires the clear dense-qwen MLP profile".into());
    }
    if attention.protected_execution
        || attention.complete_decoder
        || mlp.protected_execution
        || mlp.complete_decoder
    {
        return Err("dense-qwen layer requires clear composite blocks".into());
    }
    let output_shape = mlp.residual.output.shape.clone();
    require_rank3(&attention.input_shape, "input shape")?;
    require_rank3(&attention.output_shape, "attention output shape")?;
    require_rank3(&output_shape, "output shape")?;
    if attention.output_shape != mlp.input_shape {
        return Err("dense-qwen layer attention output must feed the MLP input".into());
    }
    if attention.input_shape != output_shape {
        return Err("dense-qwen layer input and output shapes must match".into());
    }
    if attention.residual.operation_id != mlp.norm.input_id {
        return Err("dense-qwen layer MLP norm must consume the attention residual".into());
    }
    if attention.residual.input_ids[0] != attention.norm.input_id {
        return Err("dense-qwen layer attention residual must carry the layer input".into());
    }
    if mlp.residual.input_ids[0] != mlp.norm.input_id {
        return Err("dense-qwen layer MLP residual must carry the attention output".into());
    }
    if mlp.residual.input_ids[1] != mlp.down.operation_id {
        return Err("dense-qwen layer MLP residual must add the down projection".into());
    }
    if mlp.residual_refinement.output_fractional_bits != 10
        || mlp.residual_refinement.storage != DenseQwenMlpResidualStorage::CenteredWrap32
        || mlp.residual_refinement.output_range_policy
            != DenseQwenMlpResidualOutputRangePolicy::RejectOutsideSignedI16ForNextRmsNorm
    {
        return Err(
            "dense-qwen layer requires centered-wrap32 Q10 residual output for the next layer"
                .into(),
        );
    }
    Ok(DenseQwenLayerComposite {
        schema_version: DENSE_QWEN_LAYER_SCHEMA_VERSION.into(),
        numeric_profile: DENSE_QWEN_LAYER_CLEAR_PROFILE.into(),
        model_plan_digest: plan.digest(),
        mode: attention.mode,
        layer: attention.layer,
        input_shape: attention.input_shape.clone(),
        output_shape,
        attention: attention.clone(),
        mlp: mlp.clone(),
        protected_execution: false,
        complete_decoder: false,
    })
}

pub fn compose_dense_qwen_layer_block(
    plan: &DecoderPlan,
    attention: CompiledDenseQwenAttentionBlock,
    mlp: CompiledDenseQwenMlpBlock,
) -> Result<CompiledDenseQwenLayerBlock, String> {
    let composite = build_layer_composite(plan, &attention, &mlp)?;
    let binding_digest = canonical_digest(DENSE_QWEN_LAYER_BINDING_DOMAIN, &composite);
    Ok(CompiledDenseQwenLayerBlock {
        composite,
        attention,
        mlp,
        binding_digest,
    })
}

fn validate_layer_block(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenLayerBlock,
) -> Result<(), String> {
    if canonical_digest(DENSE_QWEN_LAYER_BINDING_DOMAIN, &block.composite) != block.binding_digest {
        return Err("dense-qwen layer composite does not match its binding".into());
    }
    if build_layer_composite(plan, &block.attention, &block.mlp)? != block.composite {
        return Err("dense-qwen layer composite differs from its bound blocks".into());
    }
    Ok(())
}

pub struct DenseQwenLayerOutputQ10 {
    values: Vec<i16>,
    shape: [usize; 3],
}

impl DenseQwenLayerOutputQ10 {
    pub fn values(&self) -> &[i16] {
        &self.values
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }
}

impl Drop for DenseQwenLayerOutputQ10 {
    fn drop(&mut self) {
        self.values.zeroize();
    }
}

pub struct DenseQwenLayerState {
    attention: DenseQwenAttentionState,
    plan_digest: Digest,
    layer: u64,
    binding_digest: Digest,
    usable: bool,
}

impl DenseQwenLayerState {
    pub fn is_usable(&self) -> bool {
        self.usable
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseQwenLayerExecutionPolicy {
    pub attention: DenseQwenAttentionExecutionPolicy,
    pub mlp_threads: usize,
    pub mlp_simd: bool,
}

fn layer_state_binding(composite: &DenseQwenLayerComposite) -> Digest {
    canonical_digest(
        DENSE_QWEN_LAYER_STATE_DOMAIN,
        &(
            composite.model_plan_digest.clone(),
            composite.attention.weight_manifest_digest.clone(),
            composite.mlp.weight_manifest_digest.clone(),
            composite.layer,
        ),
    )
}

fn shape3(shape: &[u64], role: &str) -> Result<[usize; 3], String> {
    let parsed: [u64; 3] = shape
        .try_into()
        .map_err(|_| format!("dense-qwen layer {role} must be rank-3"))?;
    if parsed.contains(&0) {
        return Err(format!("dense-qwen layer {role} must be nonzero"));
    }
    let mut converted = [0_usize; 3];
    for (target, source) in converted.iter_mut().zip(parsed) {
        *target = usize::try_from(source)
            .map_err(|_| format!("dense-qwen layer {role} exceeds usize"))?;
    }
    Ok(converted)
}

fn bridge_attention_output(
    composite: &DenseQwenLayerComposite,
    output: DenseQwenAttentionOutputQ10,
) -> Result<Zeroizing<Vec<u32>>, String> {
    let expected = composite
        .mlp
        .input_shape
        .iter()
        .try_fold(1_usize, |total, dimension| {
            let dimension = usize::try_from(*dimension).ok()?;
            total.checked_mul(dimension)
        })
        .ok_or_else(|| "dense-qwen layer bridge elements overflow usize".to_owned())?;
    if output.values().len() != expected {
        return Err("dense-qwen layer attention output length mismatch".into());
    }
    let mut bridge = Vec::new();
    bridge
        .try_reserve_exact(output.values().len())
        .map_err(|_| "dense-qwen layer attention bridge allocation failed")?;
    bridge.extend(output.values().iter().map(|value| i32::from(*value) as u32));
    Ok(Zeroizing::new(bridge))
}

fn layer_output(
    composite: &DenseQwenLayerComposite,
    mut mlp_output: Zeroizing<Vec<i16>>,
) -> Result<DenseQwenLayerOutputQ10, String> {
    let shape = shape3(&composite.output_shape, "output shape")?;
    let elements = shape
        .iter()
        .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| "dense-qwen layer output elements overflow usize".to_owned())?;
    if mlp_output.len() != elements {
        return Err("dense-qwen layer output length mismatch".into());
    }
    let values = std::mem::take(&mut *mlp_output);
    Ok(DenseQwenLayerOutputQ10 { values, shape })
}

#[allow(clippy::too_many_arguments)]
pub fn execute_dense_qwen_layer_prefill(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenLayerBlock,
    input_q10: &[u32],
    positions: &[u32],
    attention_mask: &[bool],
    valid_lengths: &[usize],
    policy: DenseQwenLayerExecutionPolicy,
) -> Result<(DenseQwenLayerOutputQ10, DenseQwenLayerState), String> {
    validate_layer_block(plan, block)?;
    if block.composite.mode != DecoderMode::Prefill {
        return Err("dense-qwen layer prefill requires a prefill block".into());
    }
    let (attention_output, attention_state) = execute_dense_qwen_attention_prefill(
        plan,
        &block.attention,
        input_q10,
        positions,
        attention_mask,
        valid_lengths,
        policy.attention,
    )?;
    let bridge = bridge_attention_output(&block.composite, attention_output)?;
    let mlp_output = Zeroizing::new(execute_dense_qwen_mlp_block(
        plan,
        &block.mlp,
        &bridge,
        policy.mlp_threads,
        policy.mlp_simd,
    )?);
    let output = layer_output(&block.composite, mlp_output)?;
    Ok((
        output,
        DenseQwenLayerState {
            attention: attention_state,
            plan_digest: block.composite.model_plan_digest.clone(),
            layer: block.composite.layer,
            binding_digest: layer_state_binding(&block.composite),
            usable: true,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn execute_dense_qwen_layer_decode(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenLayerBlock,
    state: &mut DenseQwenLayerState,
    input_q10: &[u32],
    positions: &[u32],
    attention_mask: &[bool],
    new_valid_lengths: &[usize],
    policy: DenseQwenLayerExecutionPolicy,
) -> Result<DenseQwenLayerOutputQ10, String> {
    validate_layer_block(plan, block)?;
    if block.composite.mode != DecoderMode::Decode {
        return Err("dense-qwen layer decode requires a decode block".into());
    }
    if !state.usable {
        return Err("dense-qwen layer state is unusable".into());
    }
    if state.plan_digest != block.composite.model_plan_digest
        || state.layer != block.composite.layer
        || state.binding_digest != layer_state_binding(&block.composite)
    {
        return Err("dense-qwen layer state does not match the block binding".into());
    }
    let attention_output = match execute_dense_qwen_attention_decode(
        plan,
        &block.attention,
        &mut state.attention,
        input_q10,
        positions,
        attention_mask,
        new_valid_lengths,
        policy.attention,
    ) {
        Ok(output) => output,
        Err(error) => {
            state.usable = state.attention.is_usable();
            return Err(error);
        }
    };
    state.usable = false;
    let bridge = bridge_attention_output(&block.composite, attention_output)?;
    let mlp_output = Zeroizing::new(execute_dense_qwen_mlp_block(
        plan,
        &block.mlp,
        &bridge,
        policy.mlp_threads,
        policy.mlp_simd,
    )?);
    let output = layer_output(&block.composite, mlp_output)?;
    state.usable = true;
    Ok(output)
}
