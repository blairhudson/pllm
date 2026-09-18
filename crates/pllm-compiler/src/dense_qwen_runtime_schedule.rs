use pllm_models::{
    DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator, StateKind, StateTensor,
};
use pllm_types::{canonical_digest, Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DENSE_QWEN_RUNTIME_SCHEDULE_SCHEMA_VERSION: &str = "pllm.dense_qwen_runtime_schedule.v1";
pub const DENSE_QWEN_MASKED_RUNTIME_PROFILE: &str = "baseline.masked_linear_cpu";
const DENSE_QWEN_RUNTIME_SCHEDULE_DIGEST_DOMAIN: &str = "pllm.dense_qwen_runtime_schedule.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQwenRuntimeExecutor {
    ClientLocal,
    RemoteStage,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQwenRuntimeOutput {
    pub operation_id: String,
    pub output_shape: Vec<u64>,
    pub stage_offset: u64,
    pub stage_width: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQwenRuntimeStep {
    pub order: u64,
    pub operation_ids: Vec<String>,
    pub operators: Vec<ModelOperator>,
    pub layer: Option<u64>,
    pub input_ids: Vec<String>,
    pub executor: DenseQwenRuntimeExecutor,
    pub stage_role: Option<String>,
    pub outputs: Vec<DenseQwenRuntimeOutput>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQwenRuntimePhaseSchedule {
    pub mode: DecoderMode,
    pub batch: u64,
    pub query_sequence: u64,
    pub maximum_key_sequence: u64,
    pub state_inputs: Vec<StateTensor>,
    pub state_outputs: Vec<StateTensor>,
    pub steps: Vec<DenseQwenRuntimeStep>,
    pub output: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQwenRuntimeSchedule {
    pub schema_version: String,
    pub profile: String,
    pub model_plan_digest: Digest,
    pub model_config_digest: Digest,
    pub prefill: DenseQwenRuntimePhaseSchedule,
    pub decode: DenseQwenRuntimePhaseSchedule,
    pub protected_execution: bool,
    pub complete: bool,
}

impl DenseQwenRuntimeSchedule {
    pub fn digest(&self) -> Digest {
        canonical_digest(DENSE_QWEN_RUNTIME_SCHEDULE_DIGEST_DOMAIN, self)
    }
}

struct RemoteGroup<'a> {
    role: &'static str,
    layer: Option<u64>,
    operations: Vec<&'a ModelOperation>,
}

fn by_id(graph: &DecoderGraph) -> Result<BTreeMap<&str, &ModelOperation>, String> {
    let mut result = BTreeMap::new();
    for operation in &graph.operations {
        if result.insert(operation.id.as_str(), operation).is_some() {
            return Err(format!(
                "duplicate runtime-schedule operation {}",
                operation.id
            ));
        }
    }
    Ok(result)
}

fn positions(graph: &DecoderGraph) -> BTreeMap<&str, usize> {
    graph
        .operations
        .iter()
        .enumerate()
        .map(|(index, operation)| (operation.id.as_str(), index))
        .collect()
}

fn producers<'a>(
    operation: &ModelOperation,
    operations: &BTreeMap<&'a str, &'a ModelOperation>,
) -> Vec<&'a ModelOperation> {
    operation
        .inputs
        .iter()
        .filter_map(|input| operations.get(input.as_str()).copied())
        .collect()
}

fn producer<'a>(
    operation: &ModelOperation,
    operations: &BTreeMap<&'a str, &'a ModelOperation>,
    operator: ModelOperator,
    layer: u64,
    role: &str,
) -> Result<&'a ModelOperation, String> {
    let found: Vec<_> = producers(operation, operations)
        .into_iter()
        .filter(|candidate| candidate.operator == operator && candidate.layer == Some(layer))
        .collect();
    match found.as_slice() {
        [operation] => Ok(operation),
        _ => Err(format!(
            "dense-qwen runtime layer {layer} requires exactly one {role}"
        )),
    }
}

fn consumers<'a>(
    operation: &ModelOperation,
    graph: &'a DecoderGraph,
    operator: ModelOperator,
    layer: u64,
) -> Vec<&'a ModelOperation> {
    graph
        .operations
        .iter()
        .filter(|candidate| {
            candidate.operator == operator
                && candidate.layer == Some(layer)
                && candidate.inputs.contains(&operation.id)
        })
        .collect()
}

fn consumer<'a>(
    operation: &ModelOperation,
    graph: &'a DecoderGraph,
    operator: ModelOperator,
    layer: u64,
    role: &str,
) -> Result<&'a ModelOperation, String> {
    let found = consumers(operation, graph, operator, layer);
    match found.as_slice() {
        [operation] => Ok(operation),
        _ => Err(format!(
            "dense-qwen runtime layer {layer} requires exactly one {role}"
        )),
    }
}

fn unique_layer_operator(
    graph: &DecoderGraph,
    layer: u64,
    operator: ModelOperator,
) -> Result<&ModelOperation, String> {
    let found: Vec<_> = graph
        .operations
        .iter()
        .filter(|operation| operation.layer == Some(layer) && operation.operator == operator)
        .collect();
    match found.as_slice() {
        [operation] => Ok(operation),
        _ => Err(format!(
            "dense-qwen runtime layer {layer} requires exactly one {operator:?}"
        )),
    }
}

fn unique_global_operator(
    graph: &DecoderGraph,
    operator: ModelOperator,
) -> Result<&ModelOperation, String> {
    let found: Vec<_> = graph
        .operations
        .iter()
        .filter(|operation| operation.layer.is_none() && operation.operator == operator)
        .collect();
    match found.as_slice() {
        [operation] => Ok(operation),
        _ => Err(format!(
            "dense-qwen runtime requires exactly one global {operator:?}"
        )),
    }
}

fn state_view_producer<'a>(
    operation: &ModelOperation,
    operations: &BTreeMap<&'a str, &'a ModelOperation>,
    layer: u64,
    kind: StateKind,
    role: &str,
) -> Result<&'a ModelOperation, String> {
    let found: Vec<_> = producers(operation, operations)
        .into_iter()
        .filter(|candidate| {
            candidate.operator == ModelOperator::CacheSuffix
                && candidate.layer == Some(layer)
                && candidate.state_kind == Some(kind)
        })
        .collect();
    match found.as_slice() {
        [operation] => Ok(operation),
        _ => Err(format!(
            "dense-qwen runtime layer {layer} requires exactly one {role}"
        )),
    }
}

fn insert_group<'a>(
    leaders: &mut BTreeMap<&'a str, RemoteGroup<'a>>,
    members: &mut BTreeSet<&'a str>,
    role: &'static str,
    layer: Option<u64>,
    operations: Vec<&'a ModelOperation>,
) -> Result<(), String> {
    let leader_id = operations
        .first()
        .map(|operation| operation.id.as_str())
        .ok_or_else(|| format!("runtime stage {role} has no operations"))?;
    for operation in &operations {
        if !members.insert(operation.id.as_str()) {
            return Err(format!(
                "operation {} belongs to multiple runtime stages",
                operation.id
            ));
        }
    }
    if leaders
        .insert(
            leader_id,
            RemoteGroup {
                role,
                layer,
                operations,
            },
        )
        .is_some()
    {
        return Err(format!("duplicate runtime stage leader {leader_id}"));
    }
    Ok(())
}

fn local_operator(operator: ModelOperator) -> bool {
    matches!(
        operator,
        ModelOperator::Reshape
            | ModelOperator::RmsNorm
            | ModelOperator::RotaryEmbedding
            | ModelOperator::KvCacheAppend
            | ModelOperator::CacheSuffix
            | ModelOperator::AttentionScores
            | ModelOperator::AttentionScale
            | ModelOperator::CausalMask
            | ModelOperator::Softmax
            | ModelOperator::AttentionValues
            | ModelOperator::ResidualAdd
            | ModelOperator::Silu
            | ModelOperator::Multiply
            | ModelOperator::LastToken
            | ModelOperator::GreedyTokenSelection
            | ModelOperator::TokenFeedback
    )
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
    let operations = by_id(graph)?;
    let indices = positions(graph);
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
        return Err("dense-qwen runtime layers must be contiguous from zero".into());
    }
    let mut leaders = BTreeMap::new();
    let mut members = BTreeSet::new();
    let token_lookup = unique_global_operator(graph, ModelOperator::TokenLookup)?;
    let output_head = unique_global_operator(graph, ModelOperator::OutputHead)?;
    insert_group(
        &mut leaders,
        &mut members,
        "token_lookup",
        None,
        vec![token_lookup],
    )?;
    for layer in layers {
        let scores = unique_layer_operator(graph, layer, ModelOperator::AttentionScores)?;
        let query_rope = producer(
            scores,
            &operations,
            ModelOperator::RotaryEmbedding,
            layer,
            "query rotary input",
        )?;
        let query_heads = producer(
            query_rope,
            &operations,
            ModelOperator::Reshape,
            layer,
            "query head reshape",
        )?;
        let query = producer(
            query_heads,
            &operations,
            ModelOperator::Linear,
            layer,
            "query projection",
        )?;
        let key_view =
            state_view_producer(scores, &operations, layer, StateKind::Key, "key cache view")?;
        let key_append = producer(
            key_view,
            &operations,
            ModelOperator::KvCacheAppend,
            layer,
            "key cache append",
        )?;
        if key_append.state_kind != Some(StateKind::Key) {
            return Err(format!(
                "dense-qwen runtime layer {layer} key append has the wrong state kind"
            ));
        }
        let key_rope = producer(
            key_append,
            &operations,
            ModelOperator::RotaryEmbedding,
            layer,
            "key rotary input",
        )?;
        let key_heads = producer(
            key_rope,
            &operations,
            ModelOperator::Reshape,
            layer,
            "key head reshape",
        )?;
        let key = producer(
            key_heads,
            &operations,
            ModelOperator::Linear,
            layer,
            "key projection",
        )?;
        let attention_values = unique_layer_operator(graph, layer, ModelOperator::AttentionValues)?;
        let value_view = state_view_producer(
            attention_values,
            &operations,
            layer,
            StateKind::Value,
            "value cache view",
        )?;
        let value_append = producer(
            value_view,
            &operations,
            ModelOperator::KvCacheAppend,
            layer,
            "value cache append",
        )?;
        if value_append.state_kind != Some(StateKind::Value) {
            return Err(format!(
                "dense-qwen runtime layer {layer} value append has the wrong state kind"
            ));
        }
        let value_heads = producer(
            value_append,
            &operations,
            ModelOperator::Reshape,
            layer,
            "value head reshape",
        )?;
        let value = producer(
            value_heads,
            &operations,
            ModelOperator::Linear,
            layer,
            "value projection",
        )?;
        if query.inputs.len() != 1 || query.inputs != key.inputs || key.inputs != value.inputs {
            return Err(format!(
                "dense-qwen runtime layer {layer} q/k/v projections must share one input"
            ));
        }
        let query_index = *indices
            .get(query.id.as_str())
            .ok_or_else(|| format!("query operation {} is not in the graph", query.id))?;
        let key_index = *indices
            .get(key.id.as_str())
            .ok_or_else(|| format!("key operation {} is not in the graph", key.id))?;
        let value_index = *indices
            .get(value.id.as_str())
            .ok_or_else(|| format!("value operation {} is not in the graph", value.id))?;
        if query_index.checked_add(1) != Some(key_index)
            || key_index.checked_add(1) != Some(value_index)
        {
            return Err(format!(
                "dense-qwen runtime layer {layer} q/k/v projections are not fuseable in order"
            ));
        }
        insert_group(
            &mut leaders,
            &mut members,
            "qkv_projection",
            Some(layer),
            vec![query, key, value],
        )?;

        let attention_hidden = consumer(
            attention_values,
            graph,
            ModelOperator::Reshape,
            layer,
            "attention output reshape",
        )?;
        let attention_output = consumer(
            attention_hidden,
            graph,
            ModelOperator::Linear,
            layer,
            "attention output projection",
        )?;
        insert_group(
            &mut leaders,
            &mut members,
            "attention_output",
            Some(layer),
            vec![attention_output],
        )?;

        let silu = unique_layer_operator(graph, layer, ModelOperator::Silu)?;
        let gate = producer(
            silu,
            &operations,
            ModelOperator::Linear,
            layer,
            "MLP gate projection",
        )?;
        let multiply = unique_layer_operator(graph, layer, ModelOperator::Multiply)?;
        if !multiply.inputs.contains(&silu.id) {
            return Err(format!(
                "dense-qwen runtime layer {layer} multiply does not consume SiLU"
            ));
        }
        let up_candidates: Vec<_> = producers(multiply, &operations)
            .into_iter()
            .filter(|candidate| candidate.operator == ModelOperator::Linear)
            .collect();
        let up = match up_candidates.as_slice() {
            [operation] if operation.layer == Some(layer) => *operation,
            _ => {
                return Err(format!(
                    "dense-qwen runtime layer {layer} requires exactly one MLP up projection"
                ))
            }
        };
        if gate.inputs.len() != 1 || gate.inputs != up.inputs {
            return Err(format!(
                "dense-qwen runtime layer {layer} gate/up projections must share one input"
            ));
        }
        if last_width(gate)? != last_width(up)? {
            return Err(format!(
                "dense-qwen runtime layer {layer} gate/up projection widths differ"
            ));
        }
        let gate_index = *indices
            .get(gate.id.as_str())
            .ok_or_else(|| format!("gate operation {} is not in the graph", gate.id))?;
        let up_index = *indices
            .get(up.id.as_str())
            .ok_or_else(|| format!("up operation {} is not in the graph", up.id))?;
        if gate_index.checked_add(1) != Some(up_index) {
            return Err(format!(
                "dense-qwen runtime layer {layer} gate/up projections are not fuseable in order"
            ));
        }
        insert_group(
            &mut leaders,
            &mut members,
            "mlp_gate_up",
            Some(layer),
            vec![gate, up],
        )?;
        let down = consumer(
            multiply,
            graph,
            ModelOperator::Linear,
            layer,
            "MLP down projection",
        )?;
        insert_group(
            &mut leaders,
            &mut members,
            "mlp_down",
            Some(layer),
            vec![down],
        )?;

        let classified: BTreeSet<_> = [query, key, value, attention_output, gate, up, down]
            .into_iter()
            .map(|operation| operation.id.as_str())
            .collect();
        let linears: BTreeSet<_> = graph
            .operations
            .iter()
            .filter(|operation| {
                operation.layer == Some(layer) && operation.operator == ModelOperator::Linear
            })
            .map(|operation| operation.id.as_str())
            .collect();
        if classified != linears {
            return Err(format!(
                "dense-qwen runtime layer {layer} linear topology is incomplete"
            ));
        }
    }
    insert_group(
        &mut leaders,
        &mut members,
        "lm_head",
        None,
        vec![output_head],
    )?;
    Ok((leaders, members))
}

fn lower_phase(graph: &DecoderGraph) -> Result<DenseQwenRuntimePhaseSchedule, String> {
    let (leaders, remote_members) = classify_remote_groups(graph)?;
    let mut scheduled = BTreeSet::new();
    let mut steps = Vec::new();
    for operation in &graph.operations {
        if remote_members.contains(operation.id.as_str()) {
            let Some(group) = leaders.get(operation.id.as_str()) else {
                continue;
            };
            let input_ids = group.operations[0].inputs.clone();
            if group
                .operations
                .iter()
                .any(|member| member.inputs != input_ids)
            {
                return Err(format!(
                    "runtime stage {} operations do not share inputs",
                    group.role
                ));
            }
            let mut offset = 0_u64;
            let mut outputs = Vec::new();
            for member in &group.operations {
                let width = last_width(member)?;
                outputs.push(DenseQwenRuntimeOutput {
                    operation_id: member.id.clone(),
                    output_shape: member.output_shape.clone(),
                    stage_offset: offset,
                    stage_width: width,
                });
                offset = offset
                    .checked_add(width)
                    .ok_or_else(|| format!("runtime stage {} width overflowed", group.role))?;
                if !scheduled.insert(member.id.as_str()) {
                    return Err(format!("operation {} was scheduled twice", member.id));
                }
            }
            steps.push(DenseQwenRuntimeStep {
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
                layer: group.layer,
                input_ids,
                executor: DenseQwenRuntimeExecutor::RemoteStage,
                stage_role: Some(group.role.into()),
                outputs,
            });
            continue;
        }
        if !local_operator(operation.operator) {
            return Err(format!(
                "operation {} has no dense-qwen runtime executor",
                operation.id
            ));
        }
        if !scheduled.insert(operation.id.as_str()) {
            return Err(format!("operation {} was scheduled twice", operation.id));
        }
        steps.push(DenseQwenRuntimeStep {
            order: u64::try_from(steps.len()).map_err(|_| "runtime step count exceeds u64")?,
            operation_ids: vec![operation.id.clone()],
            operators: vec![operation.operator],
            layer: operation.layer,
            input_ids: operation.inputs.clone(),
            executor: DenseQwenRuntimeExecutor::ClientLocal,
            stage_role: None,
            outputs: vec![DenseQwenRuntimeOutput {
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
        return Err("dense-qwen runtime schedule does not cover every operation".into());
    }
    if steps
        .iter()
        .enumerate()
        .any(|(index, step)| step.order != u64::try_from(index).unwrap_or(u64::MAX))
    {
        return Err("dense-qwen runtime step order is not contiguous".into());
    }
    Ok(DenseQwenRuntimePhaseSchedule {
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

type RemoteSignature = (String, Option<u64>, Vec<String>, Vec<u64>);

fn remote_signature(phase: &DenseQwenRuntimePhaseSchedule) -> Vec<RemoteSignature> {
    phase
        .steps
        .iter()
        .filter(|step| step.executor == DenseQwenRuntimeExecutor::RemoteStage)
        .map(|step| {
            (
                step.stage_role.clone().unwrap_or_default(),
                step.layer,
                step.operation_ids.clone(),
                step.outputs
                    .iter()
                    .map(|output| output.stage_width)
                    .collect(),
            )
        })
        .collect()
}

pub fn lower_dense_qwen_runtime_schedule(
    plan: &DecoderPlan,
) -> Result<DenseQwenRuntimeSchedule, String> {
    plan.validate().map_err(|error| error.to_string())?;
    if plan.model_family != "qwen2" || plan.adapter != "pllm.qwen2.v1" {
        return Err("dense-qwen masked runtime schedule requires the Qwen2 adapter".into());
    }
    if !plan.transformations.is_empty() {
        return Err("dense-qwen masked runtime schedule does not support transformed plans".into());
    }
    if plan.prefill.batch != 1 || plan.decode.batch != 1 {
        return Err("dense-qwen masked runtime schedule requires batch one".into());
    }
    let prefill = lower_phase(&plan.prefill)?;
    let decode = lower_phase(&plan.decode)?;
    if remote_signature(&prefill) != remote_signature(&decode) {
        return Err("dense-qwen prefill/decode remote stage contracts differ".into());
    }
    Ok(DenseQwenRuntimeSchedule {
        schema_version: DENSE_QWEN_RUNTIME_SCHEDULE_SCHEMA_VERSION.into(),
        profile: DENSE_QWEN_MASKED_RUNTIME_PROFILE.into(),
        model_plan_digest: plan.digest(),
        model_config_digest: plan.config_digest.clone(),
        prefill,
        decode,
        protected_execution: false,
        complete: true,
    })
}
