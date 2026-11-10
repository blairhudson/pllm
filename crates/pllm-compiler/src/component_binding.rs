//! Model-neutral implementation codes. Classification grants no execution credit.

use pllm_models::{DecoderGraph, ModelOperator};
use crate::decoder_runtime_schedule::{DecoderRuntimeExecutor, DecoderRuntimePhaseSchedule};

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
