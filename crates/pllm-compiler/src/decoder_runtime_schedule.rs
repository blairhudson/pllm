use pllm_models::{
    DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator, StateTensor,
};
use pllm_types::{canonical_digest, Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DECODER_RUNTIME_SCHEDULE_SCHEMA_VERSION: &str = "pllm.decoder_runtime_schedule.v1";
pub const MASKED_LINEAR_RUNTIME_PROFILE: &str = "baseline.masked_linear_cpu";
pub const VERIFIED_MASKED_LINEAR_RUNTIME_PROFILE: &str = "research.verified_masked_linear_cpu";
const DECODER_RUNTIME_SCHEDULE_DIGEST_DOMAIN: &str = "pllm.decoder_runtime_schedule.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderRuntimeExecutor {
    ClientLocal,
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
    pub profile: String,
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
    match operation.operator {
        ModelOperator::Reshape
        | ModelOperator::RmsNorm
        | ModelOperator::CacheSuffix
        | ModelOperator::KvCacheAppend
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
        | ModelOperator::TokenFeedback => true,
        ModelOperator::RotaryEmbedding => operation
            .attributes
            .get("rope_type")
            .and_then(serde_json::Value::as_str)
            .is_none_or(|kind| matches!(kind, "default" | "proportional")),
        _ => false,
    }
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

fn lower_phase(graph: &DecoderGraph) -> Result<DecoderRuntimePhaseSchedule, String> {
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
                executor: DecoderRuntimeExecutor::RemoteStage,
                weight_ids,
                outputs,
            });
            continue;
        }
        if !local_operator(operation) {
            return Err(format!(
                "operation {} ({:?}) has no masked-linear runtime executor",
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

type RemoteSignature = (Option<u64>, Vec<String>, Vec<String>, Vec<u64>);

fn remote_signature(phase: &DecoderRuntimePhaseSchedule) -> Vec<RemoteSignature> {
    phase
        .steps
        .iter()
        .filter(|step| step.executor == DecoderRuntimeExecutor::RemoteStage)
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

pub fn lower_decoder_runtime_schedule(
    plan: &DecoderPlan,
) -> Result<DecoderRuntimeSchedule, String> {
    lower_decoder_runtime_schedule_for_profile(plan, MASKED_LINEAR_RUNTIME_PROFILE)
}

pub fn lower_decoder_runtime_schedule_for_profile(
    plan: &DecoderPlan,
    profile: &str,
) -> Result<DecoderRuntimeSchedule, String> {
    if profile == VERIFIED_MASKED_LINEAR_RUNTIME_PROFILE {
        return Err(
            "verified runtime scheduling requires verifier-bound execution evidence".into(),
        );
    }
    if profile != MASKED_LINEAR_RUNTIME_PROFILE {
        return Err("decoder runtime schedule requires the baseline masked-linear profile".into());
    }
    plan.validate().map_err(|error| error.to_string())?;
    if !plan.transformations.is_empty() {
        return Err("masked-linear runtime schedule does not support transformed plans".into());
    }
    if plan.prefill.batch != 1 || plan.decode.batch != 1 {
        return Err("masked-linear runtime schedule requires batch one".into());
    }
    let prefill = lower_phase(&plan.prefill)?;
    let decode = lower_phase(&plan.decode)?;
    if remote_signature(&prefill) != remote_signature(&decode) {
        return Err("prefill/decode remote stage contracts differ".into());
    }
    Ok(DecoderRuntimeSchedule {
        schema_version: DECODER_RUNTIME_SCHEDULE_SCHEMA_VERSION.into(),
        profile: profile.into(),
        model_plan_digest: plan.digest(),
        model_config_digest: plan.config_digest.clone(),
        prefill,
        decode,
        protected_execution: false,
        complete: true,
    })
}
