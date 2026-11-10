//! Model-neutral implementation codes. Classification grants no execution credit.

use pllm_models::{DecoderGraph, DecoderMode, ModelOperation, ModelOperator};
use crate::decoder_runtime_schedule::{DecoderRuntimeExecutor, DecoderRuntimePhaseSchedule};

/// Borrowed admission-time descriptors only. Never retained in a runtime contract.
#[derive(Clone, Copy)]
pub(crate) enum SelectedComponents<'a> {
    Legacy,
    Explicit(&'a [(&'a str, &'a str)]),
}

/// Resolve the selected reusable capability to the implementation consumed by
/// bound_step. This is not attestation of trusted client-side Python arithmetic.
pub(crate) fn bind_selected(
    operation: ModelOperator,
    selected: SelectedComponents<'_>,
) -> Result<Implementation, InvalidBoundStep> {
    let SelectedComponents::Explicit(descriptors) = selected else {
        return classify(operation);
    };
    use Implementation as I;
    use ModelOperator as O;
    let (slot, identity, implementation) = match operation {
        O::TokenLookup => ("lookup", "pllm/verified-public-f32-lookup/v1", I::Lookup),
        O::Reshape => ("attention", "pllm/f32-causal-attention/v1", I::Reshape),
        O::RmsNorm => ("rmsnorm", "pllm/f32-rmsnorm/v1", I::RmsNorm),
        O::Linear => ("linear", "pllm/open-weight-verifiable-outsourcing/v1", I::Linear),
        O::RotaryEmbedding => ("rotary", "pllm/f32-rotary/v1", I::RotaryEmbedding),
        O::KvCacheAppend => ("state", "pllm/client-local-kv", I::KvCacheAppend),
        O::AttentionScores => ("attention", "pllm/f32-causal-attention/v1", I::AttentionScores),
        O::AttentionScale => ("attention", "pllm/f32-causal-attention/v1", I::AttentionScale),
        O::CausalMask => ("attention", "pllm/f32-causal-attention/v1", I::CausalMask),
        O::Softmax => ("attention", "pllm/f32-causal-attention/v1", I::Softmax),
        O::AttentionValues => ("attention", "pllm/f32-causal-attention/v1", I::AttentionValues),
        O::ResidualAdd => ("add", "pllm/f32-add/v1", I::ResidualAdd),
        O::Silu => ("activation", "pllm/f32-activation/v1", I::Silu),
        O::Multiply => ("multiply", "pllm/f32-multiply/v1", I::Multiply),
        O::OutputHead => ("head", "pllm/full-vocabulary-head/v1", I::OutputHead),
        O::LastToken => ("head", "pllm/full-vocabulary-head/v1", I::LastToken),
        O::GreedyTokenSelection => ("selection", "pllm/greedy-selection/v1", I::GreedyTokenSelection),
        O::TokenFeedback => ("selection", "pllm/greedy-selection/v1", I::TokenFeedback),
        O::CacheSuffix => ("state", "pllm/client-local-kv", I::CacheSuffix),
        _ => return Err(InvalidBoundStep),
    };
    if descriptors.iter().filter(|(name, _)| *name == slot).count() != 1
        || !descriptors.iter().any(|(name, id)| *name == slot && *id == identity)
    {
        return Err(InvalidBoundStep);
    }
    Ok(implementation)
}

/// The selected f32 implementations consume only these semantics. General
/// model validation already checks reshape/rotary/GQA/cache geometry; reject
/// attributes that those implementations would otherwise silently ignore.
/// Called within the existing operation scan, before tensor/source admission.
pub(crate) fn validate_selected_operation(
    graph: &DecoderGraph,
    op: &ModelOperation,
    implementation: Implementation,
) -> Result<(), InvalidBoundStep> {
    use Implementation as I;
    let attributes = op.attributes.as_object().ok_or(InvalidBoundStep)?;
    let keys: &[&str] = match implementation {
        I::Lookup => &["weight"],
        I::Reshape => &["layout"],
        I::RmsNorm => &["epsilon", "weight", "weight_offset"],
        I::Linear => &["weight", "bias"],
        I::RotaryEmbedding => &["theta", "rotary_dimensions", "pairing", "position_policy",
            "coefficient_profile", "input_layout", "output_layout", "tail_policy"],
        I::KvCacheAppend => &["state", "mode", "state_capacity", "state_layout",
            "absolute_write_positions_input", "padding_mask_input", "valid_lengths_input", "attention_domain"],
        I::AttentionScores | I::AttentionValues => &["group_size"],
        I::AttentionScale => &["head_dim"],
        I::CausalMask | I::ResidualAdd | I::Silu | I::Multiply => &[],
        I::Softmax => &["axis"],
        I::OutputHead => &["weight", "tied", "input_layout", "output_layout", "compute_dtype", "output_dtype"],
        I::LastToken => &["axis", "selection", "valid_lengths_input"],
        I::GreedyTokenSelection | I::TokenFeedback => &["policy", "source"],
        I::CacheSuffix => &["axis", "maximum_sequence", "semantics", "absolute_write_positions_input",
            "padding_mask_input", "valid_lengths_input"],
    };
    if attributes.keys().any(|key| !keys.contains(&key.as_str())) {
        return Err(InvalidBoundStep);
    }
    let input = |index: usize| {
        op.inputs.get(index).and_then(|id| graph.operations.iter().find(|prior| prior.id == *id))
    };
    let same_shape = |index| input(index).is_some_and(|prior| prior.output_shape == op.output_shape);
    let text = |key: &str, value: &str| op.attributes[key].as_str() == Some(value);
    let policy = || text("policy", "pllm.greedy.v1") && text("source", "execution_policy");
    let full_state_inputs = || text("absolute_write_positions_input", "input.positions")
        && text("padding_mask_input", "input.attention_mask")
        && text("valid_lengths_input", "input.sequence_lengths");
    let valid = match implementation {
        I::Lookup => op.inputs.as_slice() == ["input.tokens"] && op.output_shape.len() == 3
            && op.output_shape[0] == graph.batch && op.output_shape[1] == graph.query_sequence,
        I::Reshape => matches!(op.attributes["layout"].as_str(),
            Some("batch_heads_sequence_feature" | "batch_sequence_hidden")),
        I::RmsNorm => same_shape(0) && op.attributes["weight_offset"].as_u64() == Some(0)
            && op.attributes["epsilon"].as_str().and_then(|value| value.parse::<f32>().ok())
                .is_some_and(|epsilon| epsilon.is_finite() && epsilon > 0.0),
        I::Linear => input(0).is_some_and(|prior| prior.output_shape.len() == 3
            && op.output_shape.len() == 3 && prior.output_shape[..2] == op.output_shape[..2])
            && attributes.get("bias").is_none_or(|bias| bias.is_null()
                || bias.as_str().is_some_and(|id| !id.is_empty())),
        I::RotaryEmbedding => same_shape(0)
            && op.inputs.get(1).is_some_and(|id| id == "input.positions")
            && op.attributes["theta"].as_u64().is_some_and(|theta| theta > 0)
            && op.attributes["rotary_dimensions"].as_u64() == op.output_shape.last().copied()
            && text("pairing", "split_half") && text("position_policy", "sequential_absolute")
            && text("coefficient_profile", "pllm.numeric.rope.q30.libm.v1")
            && text("input_layout", "batch_heads_sequence_feature")
            && text("output_layout", "batch_heads_sequence_feature") && text("tail_policy", "unchanged"),
        I::KvCacheAppend => full_state_inputs() && op.layer.is_some()
            && op.attributes["state"].as_str().is_some_and(|state| !state.is_empty()
                && (graph.mode == DecoderMode::Prefill || graph.state_inputs.iter().any(|prior|
                    prior.id == state && prior.layer == op.layer && Some(prior.kind) == op.state_kind
                    && prior.shape == op.output_shape)))
            && text("mode", if graph.mode == DecoderMode::Prefill { "initialize" } else { "append" })
            && text("state_layout", "batch_kv_heads_sequence_feature")
            && op.output_shape.get(2).copied() == op.attributes["state_capacity"].as_u64()
            && op.attributes["attention_domain"].as_object().is_some_and(|domain| domain.len() == 3
                && domain.get("layout").and_then(serde_json::Value::as_str) == Some("batch_kv_heads_sequence_feature")
                && domain.get("maximum_sequence").and_then(serde_json::Value::as_u64) == Some(graph.maximum_key_sequence)
                && domain.get("includes_current").and_then(serde_json::Value::as_bool) == Some(true))
            && graph.state_outputs.iter().any(|state| state.id == op.id && state.layer == op.layer
                && Some(state.kind) == op.state_kind && state.shape == op.output_shape),
        I::AttentionScores | I::AttentionValues => op.attributes["group_size"].as_u64().is_some_and(|n| n > 0)
            && input(0).is_some_and(|prior| prior.layer == op.layer && prior.operator ==
                if implementation == I::AttentionScores { ModelOperator::RotaryEmbedding } else { ModelOperator::Softmax })
            && input(1).is_some_and(|prior| prior.layer == op.layer
                && prior.operator == ModelOperator::CacheSuffix && prior.state_kind == Some(
                    if implementation == I::AttentionScores { pllm_models::StateKind::Key } else { pllm_models::StateKind::Value })),
        I::AttentionScale => same_shape(0) && input(0).is_some_and(|scores|
            scores.operator == ModelOperator::AttentionScores && scores.inputs.first()
                .and_then(|id| graph.operations.iter().find(|query| query.id == *id))
                .and_then(|query| query.output_shape.last()).copied()
                == op.attributes["head_dim"].as_u64()),
        I::CausalMask => same_shape(0) && op.inputs.len() == 2
            && op.inputs[1] == "input.positions"
            && input(0).is_some_and(|prior| prior.operator == ModelOperator::AttentionScale && prior.layer == op.layer),
        I::Softmax => same_shape(0) && op.attributes["axis"].as_i64() == Some(-1)
            && input(0).is_some_and(|prior| prior.operator == ModelOperator::CausalMask && prior.layer == op.layer),
        I::ResidualAdd | I::Multiply => same_shape(0) && same_shape(1),
        I::Silu => same_shape(0),
        I::OutputHead => input(0).is_some_and(|prior| prior.operator == ModelOperator::LastToken
            && prior.output_shape.len() == 2 && prior.output_shape[0] == graph.batch)
            && op.output_shape.len() == 2 && op.output_shape[0] == graph.batch
            && op.attributes["tied"].is_boolean() && text("input_layout", "batch_hidden")
            && text("output_layout", "batch_vocabulary") && text("compute_dtype", "model_native")
            && text("output_dtype", "model_native"),
        I::LastToken => op.attributes["axis"].as_u64() == Some(1)
            && if graph.mode == DecoderMode::Prefill {
                text("selection", "last_valid") && text("valid_lengths_input", "input.sequence_lengths")
            } else { attributes.len() == 1 },
        I::GreedyTokenSelection => policy() && op.output_shape.as_slice() == [graph.batch]
            && input(0).is_some_and(|prior| prior.operator == ModelOperator::OutputHead),
        I::TokenFeedback => policy() && op.output_shape.as_slice() == [graph.batch, 1]
            && graph.output == op.id
            && input(0).is_some_and(|prior| prior.operator == ModelOperator::GreedyTokenSelection),
        I::CacheSuffix => full_state_inputs() && op.attributes["axis"].as_u64() == Some(2)
            && text("semantics", "visible_valid_prefix")
            && op.attributes["maximum_sequence"].as_u64() == Some(graph.maximum_key_sequence)
            && input(0).is_some_and(|prior| prior.operator == ModelOperator::KvCacheAppend
                && prior.state_kind == op.state_kind && prior.layer == op.layer),
    };
    if !valid { return Err(InvalidBoundStep); }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidBoundStep;

impl std::fmt::Display for InvalidBoundStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid bound decoder step")
    }
}

impl std::error::Error for InvalidBoundStep {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum Implementation {
    Lookup = 1,
    Reshape = 2,
    RmsNorm = 3,
    Linear = 4,
    RotaryEmbedding = 5,
    KvCacheAppend = 6,
    AttentionScores = 7,
    AttentionScale = 8,
    CausalMask = 9,
    Softmax = 10,
    AttentionValues = 11,
    ResidualAdd = 12,
    Silu = 13,
    Multiply = 14,
    OutputHead = 15,
    LastToken = 16,
    GreedyTokenSelection = 17,
    TokenFeedback = 18,
    CacheSuffix = 19,
}

impl Implementation {
    pub(crate) fn matrix(self) -> bool {
        matches!(self, Self::Linear | Self::OutputHead)
    }
    pub(crate) fn local(self) -> bool {
        !self.matrix() && self != Self::Lookup
    }
}

pub(crate) fn classify(op: ModelOperator) -> Result<Implementation, InvalidBoundStep> {
    use Implementation as I;
    use ModelOperator as O;
    Ok(match op {
        O::TokenLookup => I::Lookup,
        O::Reshape => I::Reshape,
        O::RmsNorm => I::RmsNorm,
        O::Linear => I::Linear,
        O::RotaryEmbedding => I::RotaryEmbedding,
        O::KvCacheAppend => I::KvCacheAppend,
        O::AttentionScores => I::AttentionScores,
        O::AttentionScale => I::AttentionScale,
        O::CausalMask => I::CausalMask,
        O::Softmax => I::Softmax,
        O::AttentionValues => I::AttentionValues,
        O::ResidualAdd => I::ResidualAdd,
        O::Silu => I::Silu,
        O::Multiply => I::Multiply,
        O::OutputHead => I::OutputHead,
        O::LastToken => I::LastToken,
        O::GreedyTokenSelection => I::GreedyTokenSelection,
        O::TokenFeedback => I::TokenFeedback,
        O::CacheSuffix => I::CacheSuffix,
        O::AttentionImportance | O::SecureTopK | O::SecureGather | O::ClusterBounds
        | O::ClusterSimilarity | O::SharedIndices | O::CacheActiveIndices | O::Scale
        | O::Slice | O::GeluTanh | O::Softcap | O::StateInitialize | O::Permute
        | O::Sigmoid | O::GatedDeltaDecay | O::CausalConvolution
        | O::ConvolutionStateUpdate | O::GatedDeltaRule | O::GatedDeltaStateUpdate
        | O::RmsNormGated | O::LayerNorm | O::GeluErf | O::Tanh | O::AttentionMask => {
            return Err(InvalidBoundStep);
        }
    })
}

/// Bounded borrowed scans: every graph operation occurs exactly once, with the
/// exact operator and executor. No retained index, cloned graph or claim cursor.
pub(crate) fn validate_phase(
    graph: &DecoderGraph,
    schedule: &DecoderRuntimePhaseSchedule,
) -> Result<(), InvalidBoundStep> {
    if graph.mode != schedule.mode || graph.batch != schedule.batch
        || graph.query_sequence != schedule.query_sequence
        || graph.maximum_key_sequence != schedule.maximum_key_sequence
        || graph.state_inputs != schedule.state_inputs || graph.state_outputs != schedule.state_outputs
        || graph.output != schedule.output
    {
        return Err(InvalidBoundStep);
    }
    for operation in &graph.operations {
        classify(operation.operator)?;
        if graph.operations.iter().filter(|op| op.id == operation.id).count() != 1
            || schedule.steps.iter().flat_map(|step| &step.operation_ids)
                .filter(|id| **id == operation.id).count() != 1
        {
            return Err(InvalidBoundStep);
        }
    }
    for step in &schedule.steps {
        if step.operation_ids.is_empty() || step.operation_ids.len() != step.operators.len()
            || step.operation_ids.len() != step.outputs.len()
        {
            return Err(InvalidBoundStep);
        }
        for (index, id) in step.operation_ids.iter().enumerate() {
            let op = graph.operations.iter().find(|op| op.id == *id).ok_or(InvalidBoundStep)?;
            let implementation = classify(op.operator)?;
            if step.operators[index] != op.operator || step.outputs[index].operation_id != *id
                || step.outputs[index].output_shape != op.output_shape
                || step.input_ids != op.inputs || step.layer != op.layer
                || step.executor != if implementation.local() {
                    DecoderRuntimeExecutor::ClientLocal
                } else { DecoderRuntimeExecutor::RemoteStage }
                || (implementation == Implementation::Lookup && step.operation_ids.len() != 1)
                || (implementation.local() && step.operation_ids.len() != 1)
                || (implementation.matrix() && step.operators.iter().any(|other|
                    !classify(*other).is_ok_and(|code| code.matrix())))
            {
                return Err(InvalidBoundStep);
            }
        }
    }
    Ok(())
}
