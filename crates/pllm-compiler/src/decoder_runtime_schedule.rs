use pllm_models::{
    DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator, StateKind, StateTensor,
};
use pllm_types::{canonical_digest, pipeline_digest_bytes, Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DECODER_RUNTIME_SCHEDULE_SCHEMA_VERSION: &str = "pllm.decoder_runtime_schedule.v2";
const DECODER_RUNTIME_SCHEDULE_DIGEST_DOMAIN: &str = "pllm.decoder_runtime_schedule.v2";
const MAX_WINDOW_STATE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_WINDOW_VIEW_ELEMENTS: u64 = 1 << 24;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderRuntimeExecutor {
    ClientLocal,
    ClientLinear,
    RemoteStage,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderRuntimeOutput {
    pub operation_id: String,
    pub output_shape: Vec<u64>,
    pub stage_offset: u64,
    pub stage_width: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderRuntimeStep {
    pub order: u64,
    pub operation_ids: Vec<String>,
    pub operators: Vec<ModelOperator>,
    pub layer: Option<u64>,
    pub input_ids: Vec<String>,
    pub executor: DecoderRuntimeExecutor,
    pub weight_ids: Vec<String>,
    pub outputs: Vec<DecoderRuntimeOutput>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderRuntimePhaseSchedule {
    pub mode: DecoderMode,
    pub batch: u64,
    pub query_sequence: u64,
    pub maximum_key_sequence: u64,
    pub state_inputs: Vec<StateTensor>,
    pub state_outputs: Vec<StateTensor>,
    pub steps: Vec<DecoderRuntimeStep>,
    pub output: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderRuntimeSchedule {
    pub schema_version: String,
    pub composition_digest: Digest,
    pub model_plan_digest: Digest,
    pub model_config_digest: Digest,
    pub prefill: DecoderRuntimePhaseSchedule,
    pub decode: DecoderRuntimePhaseSchedule,
    pub protected_execution: bool,
    pub complete: bool,
}

impl DecoderRuntimeSchedule {
    pub fn digest(&self) -> Digest {
        canonical_digest(DECODER_RUNTIME_SCHEDULE_DIGEST_DOMAIN, self)
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RemoteGroupKey {
    operator: ModelOperator,
    layer: Option<u64>,
    inputs: Vec<String>,
    singleton: Option<String>,
}

struct RemoteGroup<'a> {
    operations: Vec<&'a ModelOperation>,
}

fn remote_operator(operator: ModelOperator) -> bool {
    matches!(
        operator,
        ModelOperator::TokenLookup | ModelOperator::Linear | ModelOperator::OutputHead
    )
}

fn local_operator(operation: &ModelOperation) -> bool {
    let bfloat16 = |value: &serde_json::Value| {
        value
            .get("compute_dtype")
            .and_then(serde_json::Value::as_str)
            == Some("bfloat16")
            && value
                .get("output_dtype")
                .and_then(serde_json::Value::as_str)
                == Some("bfloat16")
    };
    match operation.operator {
        ModelOperator::Reshape
        | ModelOperator::KvCacheAppend
        | ModelOperator::Silu
        | ModelOperator::LastToken
        | ModelOperator::GreedyTokenSelection
        | ModelOperator::TokenFeedback => true,
        ModelOperator::ResidualAdd | ModelOperator::Multiply => {
            let attrs = &operation.attributes;
            attrs.as_object().is_some_and(serde_json::Map::is_empty)
                || attrs
                    .get("output_dtype")
                    .and_then(serde_json::Value::as_str)
                    == Some("bfloat16")
        }
        ModelOperator::Softmax => {
            let attrs = &operation.attributes;
            (attrs.get("output_dtype").is_none() && attrs.get("compute_dtype").is_none())
                || (attrs
                    .get("output_dtype")
                    .and_then(serde_json::Value::as_str)
                    == Some("bfloat16")
                    && attrs
                        .get("compute_dtype")
                        .and_then(serde_json::Value::as_str)
                        == Some("float32")
                    && attrs.get("axis").and_then(serde_json::Value::as_i64) == Some(-1))
        }
        ModelOperator::RmsNorm => {
            let attrs = &operation.attributes;
            if attrs.get("output_dtype").is_none() && attrs.get("compute_dtype").is_none() {
                return true;
            }
            let weight = attrs.get("weight").and_then(serde_json::Value::as_str);
            let offset = attrs
                .get("weight_offset")
                .and_then(serde_json::Value::as_i64);
            attrs
                .get("output_dtype")
                .and_then(serde_json::Value::as_str)
                == Some("bfloat16")
                && attrs
                    .get("compute_dtype")
                    .and_then(serde_json::Value::as_str)
                    == Some("float32")
                && attrs
                    .get("epsilon")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|value| !value.is_empty())
                && attrs.get("with_scale").and_then(serde_json::Value::as_bool)
                    == Some(weight.is_some())
                && matches!(offset, Some(0 | 1))
                && (weight.is_some() || offset == Some(0))
        }
        ModelOperator::CacheSuffix => {
            let attrs = &operation.attributes;
            if attrs.get("axis").and_then(serde_json::Value::as_u64) == Some(2)
                && attrs.get("semantics").and_then(serde_json::Value::as_str)
                    == Some("visible_valid_prefix")
            {
                return true;
            }
            attrs.get("axis").and_then(serde_json::Value::as_u64) == Some(3)
                && attrs.get("output_axis").and_then(serde_json::Value::as_u64) == Some(2)
                && attrs.get("semantics").and_then(serde_json::Value::as_str)
                    == Some("persist_last_valid_past_tokens")
                && attrs
                    .get("absolute_write_positions_input")
                    .and_then(serde_json::Value::as_str)
                    == Some("input.positions")
                && attrs
                    .get("padding_mask_input")
                    .and_then(serde_json::Value::as_str)
                    == Some("input.attention_mask")
                && attrs
                    .get("valid_lengths_input")
                    .and_then(serde_json::Value::as_str)
                    == Some("input.sequence_lengths")
                && operation.inputs.len() == 4
                && operation.inputs[1..]
                    == [
                        "input.positions",
                        "input.attention_mask",
                        "input.sequence_lengths",
                    ]
                && operation.output_shape.len() == 4
                && attrs
                    .get("maximum_sequence")
                    .and_then(serde_json::Value::as_u64)
                    == Some(operation.output_shape[2])
                && matches!(
                    operation.state_kind.as_ref(),
                    Some(StateKind::Key | StateKind::Value)
                )
        }
        ModelOperator::AttentionScores | ModelOperator::AttentionValues => {
            let key = if operation.operator == ModelOperator::AttentionScores {
                "key_layout"
            } else {
                "value_layout"
            };
            let attrs = &operation.attributes;
            match attrs.get(key).and_then(serde_json::Value::as_str) {
                None => {
                    attrs.get("output_dtype").is_none()
                        && attrs
                            .get("group_size")
                            .and_then(serde_json::Value::as_u64)
                            .is_some_and(|value| value > 0)
                }
                Some("batch_kv_heads_sequence_feature" | "batch_kv_heads_query_window_feature") => {
                    attrs
                        .get("group_size")
                        .and_then(serde_json::Value::as_u64)
                        .is_some_and(|value| value > 0)
                        && attrs
                            .get("output_dtype")
                            .and_then(serde_json::Value::as_str)
                            == Some("bfloat16")
                        && operation.output_shape.len() == 4
                        && (operation.operator != ModelOperator::AttentionScores
                            || (operation.layer.is_some_and(|layer| {
                                attrs
                                    .get("key_value_source_layer")
                                    .and_then(serde_json::Value::as_u64)
                                    .is_some_and(|source| source <= layer)
                            })))
                }
                _ => false,
            }
        }
        ModelOperator::CausalMask => {
            let attrs = &operation.attributes;
            if attrs.as_object().is_some_and(serde_json::Map::is_empty) {
                return true;
            }
            if attrs
                .get("absolute_positions_input")
                .and_then(serde_json::Value::as_str)
                != Some("input.positions")
                || attrs
                    .get("padding_mask_input")
                    .and_then(serde_json::Value::as_str)
                    != Some("input.attention_mask")
                || attrs
                    .get("valid_lengths_input")
                    .and_then(serde_json::Value::as_str)
                    != Some("input.sequence_lengths")
            {
                return false;
            }
            match attrs.get("kind").and_then(serde_json::Value::as_str) {
                Some("full_causal") => {
                    attrs
                        .get("maximum_position_embeddings")
                        .and_then(serde_json::Value::as_u64)
                        .is_some_and(|value| value > 0)
                        && attrs.get("key_domain").and_then(serde_json::Value::as_str)
                            == Some("fixed_capacity")
                        && attrs
                            .get("cache_validity")
                            .and_then(serde_json::Value::as_str)
                            == Some("valid_lengths_fixed_capacity")
                }
                Some("sliding_causal") => {
                    let window = attrs
                        .get("sliding_window")
                        .and_then(serde_json::Value::as_u64);
                    window.is_some_and(|value| {
                        value > 0
                            && attrs
                                .get("left_context")
                                .and_then(serde_json::Value::as_u64)
                                == Some(value - 1)
                    }) && attrs
                        .get("includes_current")
                        .and_then(serde_json::Value::as_bool)
                        == Some(true)
                        && attrs.get("key_domain").and_then(serde_json::Value::as_str)
                            == Some("query_relative_window")
                        && attrs
                            .get("cache_validity")
                            .and_then(serde_json::Value::as_str)
                            == Some("valid_lengths_bounded_suffix")
                }
                _ => false,
            }
        }
        ModelOperator::Scale | ModelOperator::AttentionScale => {
            if operation.operator == ModelOperator::AttentionScale
                && operation.attributes.get("factor").is_none()
            {
                return operation
                    .attributes
                    .get("head_dim")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|value| value > 0);
            }
            let Some(factor) = operation.attributes.get("factor") else {
                return false;
            };
            if !bfloat16(factor)
                || factor
                    .get("factor_rounding_dtype")
                    .and_then(serde_json::Value::as_str)
                    != Some("bfloat16")
            {
                return false;
            }
            match factor.get("kind").and_then(serde_json::Value::as_str) {
                Some("sqrt" | "inverse_sqrt") => {
                    factor
                        .get("radicand")
                        .and_then(serde_json::Value::as_u64)
                        .is_some_and(|value| value > 0)
                        && matches!(
                            factor
                                .get("factor_source_dtype")
                                .and_then(serde_json::Value::as_str),
                            Some("float32_buffer" | "python_float64")
                        )
                }
                Some("rational") => {
                    factor
                        .get("numerator")
                        .and_then(serde_json::Value::as_i64)
                        .is_some()
                        && factor
                            .get("denominator")
                            .and_then(serde_json::Value::as_i64)
                            .is_some_and(|value| value > 0)
                        && factor
                            .get("factor_source_dtype")
                            .and_then(serde_json::Value::as_str)
                            == Some("exact_integer")
                }
                Some("checkpoint_scalar") => {
                    factor
                        .get("weight")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|value| !value.is_empty())
                        && factor.get("weight_shape") == Some(&serde_json::json!([1]))
                        && factor
                            .get("factor_source_dtype")
                            .and_then(serde_json::Value::as_str)
                            == Some("bfloat16")
                }
                _ => false,
            }
        }
        ModelOperator::GeluTanh => {
            bfloat16(&operation.attributes)
                && operation
                    .attributes
                    .get("approximation")
                    .and_then(serde_json::Value::as_str)
                    == Some("tanh")
        }
        ModelOperator::Softcap => {
            bfloat16(&operation.attributes)
                && operation
                    .attributes
                    .get("cap")
                    .and_then(serde_json::Value::as_i64)
                    .is_some_and(|value| value > 0)
                && operation
                    .attributes
                    .get("formula")
                    .and_then(serde_json::Value::as_str)
                    == Some("cap*tanh(input/cap)")
        }
        ModelOperator::Permute => {
            operation.attributes.get("permutation") == Some(&serde_json::json!([0, 2, 1, 3]))
                && operation.output_shape.len() == 4
        }
        ModelOperator::Slice => {
            let start = operation
                .attributes
                .get("start")
                .and_then(serde_json::Value::as_u64);
            start.is_some_and(|start| {
                operation
                    .attributes
                    .get("end")
                    .and_then(serde_json::Value::as_u64)
                    == start.checked_add(1)
            }) && operation
                .attributes
                .get("axis")
                .and_then(serde_json::Value::as_i64)
                == Some(2)
                && operation
                    .attributes
                    .get("squeeze")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
                && operation.output_shape.len() == 3
        }
        ModelOperator::RotaryEmbedding => {
            operation
                .attributes
                .get("rope_type")
                .and_then(serde_json::Value::as_str)
                .is_none_or(|kind| matches!(kind, "default" | "proportional"))
                && operation
                    .attributes
                    .get("frequency_scaling")
                    .is_none_or(|scale| {
                        scale.get("kind").and_then(serde_json::Value::as_str)
                            == Some("wavelength_transition")
                    })
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum NumericDomain {
    Bfloat16,
    MaskedBfloat16,
    Integer,
}

/// Mixed-precision plans must bind every semantic numeric edge, not merely
/// round individual activations. The client may represent BF16 as float32
/// storage only when every producer and consumer preserves its boundary.
fn bind_bfloat16_graph(graph: &DecoderGraph) -> Result<(), String> {
    let declares_bfloat16 = graph.operations.iter().any(|operation| {
        let attrs = &operation.attributes;
        ["output_dtype", "input_dtype", "compute_dtype"]
            .iter()
            .any(|field| attrs.get(*field).and_then(serde_json::Value::as_str) == Some("bfloat16"))
            || attrs
                .get("factor")
                .and_then(|factor| factor.get("output_dtype"))
                .and_then(serde_json::Value::as_str)
                == Some("bfloat16")
    });
    if !declares_bfloat16 {
        return Ok(());
    }
    let mut domains = BTreeMap::new();
    domains.insert("input.tokens".to_owned(), NumericDomain::Integer);
    domains.insert("input.positions".to_owned(), NumericDomain::Integer);
    domains.insert("input.sequence_lengths".to_owned(), NumericDomain::Integer);
    domains.insert("input.attention_mask".to_owned(), NumericDomain::Integer);
    for state in &graph.state_inputs {
        domains.insert(state.id.clone(), NumericDomain::Bfloat16);
    }
    for operation in &graph.operations {
        let inputs: Vec<_> = operation
            .inputs
            .iter()
            .map(|id| domains.get(id).copied())
            .collect();
        let uses = |indices: &[usize], domain| {
            indices
                .iter()
                .all(|index| inputs.get(*index) == Some(&Some(domain)))
        };
        let attrs = &operation.attributes;
        let output = attrs
            .get("output_dtype")
            .and_then(serde_json::Value::as_str)
            == Some("bfloat16");
        let factor_output = attrs
            .get("factor")
            .and_then(|factor| factor.get("output_dtype"))
            .and_then(serde_json::Value::as_str)
            == Some("bfloat16");
        let result = match operation.operator {
            ModelOperator::TokenLookup if uses(&[0], NumericDomain::Integer) && output => {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::Linear
                if uses(&[0], NumericDomain::Bfloat16)
                    && attrs.get("input_dtype").and_then(serde_json::Value::as_str)
                        == Some("bfloat16")
                    && output =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::OutputHead
            | ModelOperator::RmsNorm
            | ModelOperator::RotaryEmbedding
            | ModelOperator::GeluTanh
            | ModelOperator::Softcap
                if uses(&[0], NumericDomain::Bfloat16) && output =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::Scale | ModelOperator::AttentionScale
                if uses(&[0], NumericDomain::Bfloat16) && factor_output =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::AttentionScores
            | ModelOperator::AttentionValues
            | ModelOperator::ResidualAdd
            | ModelOperator::Multiply
                if uses(&[0, 1], NumericDomain::Bfloat16) && output =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::Reshape
            | ModelOperator::Permute
            | ModelOperator::Slice
            | ModelOperator::LastToken
            | ModelOperator::CacheSuffix
                if uses(&[0], NumericDomain::Bfloat16) =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::KvCacheAppend
                if operation.inputs.iter().enumerate().any(|(index, id)| {
                    id != "input.positions"
                        && id != "input.attention_mask"
                        && id != "input.sequence_lengths"
                        && inputs[index] == Some(NumericDomain::Bfloat16)
                }) && operation.inputs.iter().enumerate().all(|(index, id)| {
                    matches!(
                        id.as_str(),
                        "input.positions" | "input.attention_mask" | "input.sequence_lengths"
                    ) || inputs[index] == Some(NumericDomain::Bfloat16)
                }) =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::CausalMask if uses(&[0], NumericDomain::Bfloat16) => {
                Some(NumericDomain::MaskedBfloat16)
            }
            ModelOperator::Softmax if uses(&[0], NumericDomain::MaskedBfloat16) && output => {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::GreedyTokenSelection if uses(&[0], NumericDomain::Bfloat16) => {
                Some(NumericDomain::Integer)
            }
            ModelOperator::TokenFeedback if uses(&[0], NumericDomain::Integer) => {
                Some(NumericDomain::Integer)
            }
            _ => None,
        };
        let Some(result) = result else {
            return Err(format!(
                "BF16 numeric edge at {} has no complete operator contract",
                operation.id
            ));
        };
        if domains.insert(operation.id.clone(), result).is_some() {
            return Err(format!(
                "BF16 operation {} was declared twice",
                operation.id
            ));
        }
    }
    Ok(())
}

fn weight_id(operation: &ModelOperation) -> Result<String, String> {
    operation
        .attributes
        .get("weight")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            format!(
                "remote operation {} requires one declared weight artifact",
                operation.id
            )
        })
}

fn last_width(operation: &ModelOperation) -> Result<u64, String> {
    operation
        .output_shape
        .last()
        .copied()
        .filter(|width| *width > 0)
        .ok_or_else(|| format!("operation {} has no output width", operation.id))
}

fn classify_remote_groups(
    graph: &DecoderGraph,
) -> Result<(BTreeMap<&str, RemoteGroup<'_>>, BTreeSet<&str>), String> {
    let layers: BTreeSet<_> = graph
        .operations
        .iter()
        .filter_map(|operation| operation.layer)
        .collect();
    if layers.is_empty()
        || layers
            .iter()
            .copied()
            .ne(0..u64::try_from(layers.len()).map_err(|_| "layer count exceeds u64")?)
    {
        return Err("decoder runtime layers must be contiguous from zero".into());
    }

    let mut grouped = BTreeMap::<RemoteGroupKey, Vec<&ModelOperation>>::new();
    for operation in &graph.operations {
        if remote_operator(operation.operator) {
            weight_id(operation)?;
            grouped
                .entry(RemoteGroupKey {
                    operator: operation.operator,
                    layer: operation.layer,
                    inputs: operation.inputs.clone(),
                    // Token tables with the same input belong to one lookup
                    // stage, preserving each table as an ordered artifact.
                    singleton: (operation.operator == ModelOperator::OutputHead)
                        .then(|| operation.id.clone()),
                })
                .or_default()
                .push(operation);
        }
    }

    let token_lookups = graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::TokenLookup)
        .count();
    let output_heads = graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::OutputHead)
        .count();
    if token_lookups == 0 || output_heads != 1 {
        return Err("decoder runtime requires token lookup and one output head".into());
    }

    let positions: BTreeMap<_, _> = graph
        .operations
        .iter()
        .enumerate()
        .map(|(index, operation)| (operation.id.as_str(), index))
        .collect();
    let mut leaders = BTreeMap::new();
    let mut members = BTreeSet::new();
    for operations in grouped.into_values() {
        let mut operations = operations;
        operations.sort_by_key(|operation| positions[operation.id.as_str()]);
        let leader = operations
            .first()
            .ok_or("remote runtime group has no operations")?
            .id
            .as_str();
        for operation in &operations {
            if !members.insert(operation.id.as_str()) {
                return Err(format!(
                    "operation {} belongs to multiple runtime stages",
                    operation.id
                ));
            }
        }
        if leaders.insert(leader, RemoteGroup { operations }).is_some() {
            return Err(format!("duplicate runtime stage leader {leader}"));
        }
    }
    Ok((leaders, members))
}

fn lower_phase(
    graph: &DecoderGraph,
    linear_executor: DecoderRuntimeExecutor,
) -> Result<DecoderRuntimePhaseSchedule, String> {
    let (leaders, remote_members) = classify_remote_groups(graph)?;
    let mut scheduled = BTreeSet::new();
    let mut steps = Vec::new();
    for operation in &graph.operations {
        if remote_members.contains(operation.id.as_str()) {
            let Some(group) = leaders.get(operation.id.as_str()) else {
                continue;
            };
            let input_ids = group.operations[0].inputs.clone();
            let layer = group.operations[0].layer;
            if group
                .operations
                .iter()
                .any(|member| member.inputs != input_ids || member.layer != layer)
            {
                return Err("runtime stage operations do not share inputs and layer".into());
            }
            let mut offset = 0_u64;
            let mut outputs = Vec::new();
            let mut weight_ids = Vec::new();
            for member in &group.operations {
                let width = last_width(member)?;
                outputs.push(DecoderRuntimeOutput {
                    operation_id: member.id.clone(),
                    output_shape: member.output_shape.clone(),
                    stage_offset: offset,
                    stage_width: width,
                });
                offset = offset
                    .checked_add(width)
                    .ok_or("runtime stage width overflowed")?;
                weight_ids.push(weight_id(member)?);
                if !scheduled.insert(member.id.as_str()) {
                    return Err(format!("operation {} was scheduled twice", member.id));
                }
            }
            steps.push(DecoderRuntimeStep {
                order: u64::try_from(steps.len()).map_err(|_| "runtime step count exceeds u64")?,
                operation_ids: group
                    .operations
                    .iter()
                    .map(|member| member.id.clone())
                    .collect(),
                operators: group
                    .operations
                    .iter()
                    .map(|member| member.operator)
                    .collect(),
                layer,
                input_ids,
                executor: linear_executor,
                weight_ids,
                outputs,
            });
            continue;
        }
        if !local_operator(operation) {
            return Err(format!(
                "operation {} ({:?}) has no decoder runtime executor",
                operation.id, operation.operator
            ));
        }
        if !scheduled.insert(operation.id.as_str()) {
            return Err(format!("operation {} was scheduled twice", operation.id));
        }
        steps.push(DecoderRuntimeStep {
            order: u64::try_from(steps.len()).map_err(|_| "runtime step count exceeds u64")?,
            operation_ids: vec![operation.id.clone()],
            operators: vec![operation.operator],
            layer: operation.layer,
            input_ids: operation.inputs.clone(),
            executor: DecoderRuntimeExecutor::ClientLocal,
            weight_ids: Vec::new(),
            outputs: vec![DecoderRuntimeOutput {
                operation_id: operation.id.clone(),
                output_shape: operation.output_shape.clone(),
                stage_offset: 0,
                stage_width: last_width(operation)?,
            }],
        });
    }
    let expected: BTreeSet<_> = graph
        .operations
        .iter()
        .map(|operation| operation.id.as_str())
        .collect();
    if scheduled != expected {
        return Err("decoder runtime schedule does not cover every operation".into());
    }
    Ok(DecoderRuntimePhaseSchedule {
        mode: graph.mode,
        batch: graph.batch,
        query_sequence: graph.query_sequence,
        maximum_key_sequence: graph.maximum_key_sequence,
        state_inputs: graph.state_inputs.clone(),
        state_outputs: graph.state_outputs.clone(),
        steps,
        output: graph.output.clone(),
    })
}

type LinearStageSignature = (Option<u64>, Vec<String>, Vec<String>, Vec<u64>);

fn linear_stage_signature(
    phase: &DecoderRuntimePhaseSchedule,
    linear_executor: DecoderRuntimeExecutor,
) -> Vec<LinearStageSignature> {
    phase
        .steps
        .iter()
        .filter(|step| step.executor == linear_executor)
        .map(|step| {
            (
                step.layer,
                step.operation_ids.clone(),
                step.weight_ids.clone(),
                step.outputs
                    .iter()
                    .map(|output| output.stage_width)
                    .collect(),
            )
        })
        .collect()
}

fn validate_window_resources(graph: &DecoderGraph) -> Result<(), String> {
    if !graph.operations.iter().any(|operation| {
        operation.operator == ModelOperator::CacheSuffix
            && operation
                .attributes
                .get("axis")
                .and_then(serde_json::Value::as_u64)
                == Some(3)
    }) {
        return Ok(());
    }
    let mut state_bytes = 0_u64;
    for state in &graph.state_outputs {
        let bytes = state
            .shape
            .iter()
            .try_fold(4_u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or("bounded decoder state size overflowed")?;
        state_bytes = state_bytes
            .checked_add(bytes)
            .ok_or("bounded decoder state size overflowed")?;
        if state_bytes > MAX_WINDOW_STATE_BYTES {
            return Err("bounded decoder state exceeds the client memory ceiling".into());
        }
    }
    for operation in &graph.operations {
        if operation.operator != ModelOperator::KvCacheAppend || operation.output_shape.len() != 5 {
            continue;
        }
        let shape = &operation.output_shape;
        let view_elements = [shape[0], shape[1], shape[3], shape[4]]
            .into_iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(dimension))
            .ok_or("bounded window view size overflowed")?;
        if view_elements > MAX_WINDOW_VIEW_ELEMENTS {
            return Err("bounded window view exceeds the client memory ceiling".into());
        }
    }
    Ok(())
}

pub fn lower_decoder_runtime_schedule(
    plan: &DecoderPlan,
    canonical_composition: &[u8],
) -> Result<DecoderRuntimeSchedule, String> {
    let linear_executor = match super::classify_decoder_composition(canonical_composition)? {
        super::DecoderCompositionKind::MaskedLinear => DecoderRuntimeExecutor::RemoteStage,
        super::DecoderCompositionKind::TwoOnlineOffsetLinear => DecoderRuntimeExecutor::RemoteStage,
        super::DecoderCompositionKind::ClientOnlyLinear => DecoderRuntimeExecutor::ClientLinear,
        super::DecoderCompositionKind::VerifiedMaskedLinear => {
            return Err(
                "verified runtime scheduling requires verifier-bound execution evidence".into(),
            );
        }
        super::DecoderCompositionKind::Other => {
            return Err(
                "decoder runtime schedule requires an admitted linear component composition".into(),
            );
        }
    };
    plan.validate().map_err(|error| error.to_string())?;
    bind_bfloat16_graph(&plan.prefill)?;
    bind_bfloat16_graph(&plan.decode)?;
    if !plan.transformations.is_empty() {
        return Err("baseline decoder runtime schedule does not support transformed plans".into());
    }
    if plan.prefill.batch != 1 || plan.decode.batch != 1 {
        return Err("baseline decoder runtime schedule requires batch one".into());
    }
    validate_window_resources(&plan.prefill)?;
    validate_window_resources(&plan.decode)?;
    let prefill = lower_phase(&plan.prefill, linear_executor)?;
    let decode = lower_phase(&plan.decode, linear_executor)?;
    if linear_stage_signature(&prefill, linear_executor)
        != linear_stage_signature(&decode, linear_executor)
    {
        return Err("prefill/decode linear stage contracts differ".into());
    }
    Ok(DecoderRuntimeSchedule {
        schema_version: DECODER_RUNTIME_SCHEDULE_SCHEMA_VERSION.into(),
        composition_digest: pipeline_digest_bytes(canonical_composition),
        model_plan_digest: plan.digest(),
        model_config_digest: plan.config_digest.clone(),
        prefill,
        decode,
        protected_execution: false,
        complete: true,
    })
}

#[cfg(test)]
mod numeric_contract_tests {
    use super::*;
    use pllm_models::{lower_model_json, DecoderWorkload};

    #[test]
    fn checked_semantic_numeric_operations_require_bound_contracts() {
        let plan = lower_model_json(
            include_bytes!("../../pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 2,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        let (groups, _) = classify_remote_groups(&plan.prefill).unwrap();
        assert_eq!(groups["main_embedding"].operations.len(), 2);
        assert_ne!(
            weight_id(groups["main_embedding"].operations[0]).unwrap(),
            weight_id(groups["main_embedding"].operations[1]).unwrap()
        );
        bind_bfloat16_graph(&plan.prefill).unwrap();
        bind_bfloat16_graph(&plan.decode).unwrap();
        let mut unbound_linear = plan.prefill.clone();
        unbound_linear
            .operations
            .iter_mut()
            .find(|row| row.id == "layer.0.q_linear")
            .unwrap()
            .attributes["output_dtype"] = serde_json::Value::Null;
        assert!(bind_bfloat16_graph(&unbound_linear)
            .unwrap_err()
            .contains("layer.0.q_linear"));
        let mut unbound_residual = plan.prefill.clone();
        unbound_residual
            .operations
            .iter_mut()
            .find(|row| row.id == "layer.0.attention_residual")
            .unwrap()
            .attributes["output_dtype"] = serde_json::Value::Null;
        assert!(bind_bfloat16_graph(&unbound_residual)
            .unwrap_err()
            .contains("attention_residual"));
        let mut stripped = plan.prefill.clone();
        for operation in &mut stripped.operations {
            if let Some(attributes) = operation.attributes.as_object_mut() {
                attributes.remove("output_dtype");
            }
        }
        assert!(bind_bfloat16_graph(&stripped)
            .unwrap_err()
            .contains("main_embedding"));
        let operations = &plan.prefill.operations;
        for id in [
            "main_embedding_scaled",
            "ple_context_scaled",
            "layer.0.attention_scale",
            "layer.0.gelu_tanh",
            "layer.0.q_permute",
            "layer.0.ple_slice",
            "logit_softcap",
        ] {
            let operation = operations.iter().find(|row| row.id == id).unwrap();
            assert!(
                local_operator(operation),
                "{id} needs a generic client-local contract"
            );
        }
        let scalar = operations
            .iter()
            .find(|row| row.id == "layer.0.layer_scalar")
            .unwrap();
        assert!(local_operator(scalar));
        let mut forged_scalar = scalar.clone();
        forged_scalar.attributes["factor"]["weight_shape"] = serde_json::json!([2]);
        assert!(!local_operator(&forged_scalar));
        let suffix = operations
            .iter()
            .find(|row| row.id == "layer.0.key_suffix")
            .unwrap();
        assert!(local_operator(suffix));
        let mut forged_suffix = suffix.clone();
        forged_suffix.attributes["maximum_sequence"] = serde_json::json!(0);
        assert!(!local_operator(&forged_suffix));
        validate_window_resources(&plan.prefill).unwrap();
        validate_window_resources(&plan.decode).unwrap();
        let mut oversize = plan.decode.clone();
        oversize.state_outputs[0].shape[2] = 1 << 30;
        assert!(validate_window_resources(&oversize)
            .unwrap_err()
            .contains("memory ceiling"));
        let sliding_mask = operations
            .iter()
            .find(|row| row.id == "layer.0.causal_mask")
            .unwrap();
        assert!(local_operator(sliding_mask));
        let full_mask = operations
            .iter()
            .find(|row| {
                row.operator == ModelOperator::CausalMask && row.attributes["kind"] == "full_causal"
            })
            .unwrap();
        assert!(local_operator(full_mask));
        for id in ["layer.0.input_norm", "layer.0.v_norm", "layer.0.softmax"] {
            let operation = operations.iter().find(|row| row.id == id).unwrap();
            assert!(local_operator(operation));
        }
        let mut forged_norm = operations
            .iter()
            .find(|row| row.id == "layer.0.input_norm")
            .unwrap()
            .clone();
        forged_norm.attributes["with_scale"] = serde_json::json!(false);
        assert!(!local_operator(&forged_norm));
        for id in ["layer.0.attention_scores", "layer.0.attention_values"] {
            let operation = operations.iter().find(|row| row.id == id).unwrap();
            assert!(local_operator(operation));
            assert_eq!(operation.attributes["output_dtype"], "bfloat16");
        }
        let mut forged_layout = operations
            .iter()
            .find(|row| row.id == "layer.0.attention_scores")
            .unwrap()
            .clone();
        forged_layout.attributes["key_layout"] = serde_json::json!("unbound_layout");
        assert!(!local_operator(&forged_layout));
        let mut forged_window = sliding_mask.clone();
        forged_window.attributes["left_context"] = serde_json::json!(0);
        assert!(!local_operator(&forged_window));
        let mut forged = operations
            .iter()
            .find(|row| row.id == "main_embedding_scaled")
            .unwrap()
            .clone();
        forged.attributes["factor"]["factor_rounding_dtype"] = serde_json::json!("float16");
        assert!(!local_operator(&forged));
    }
}
