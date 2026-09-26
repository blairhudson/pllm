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
const MAX_CONVOLUTION_CHANNELS: u64 = 8192;
const MAX_CONVOLUTION_KERNEL: u64 = 16;
const MAX_CONVOLUTION_TOKENS: u64 = 256;

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

fn bounded_convolution_operator(operation: &ModelOperation) -> bool {
    let attrs = &operation.attributes;
    let Some(fields) = attrs.as_object() else {
        return false;
    };
    let channels = attrs.get("groups").and_then(serde_json::Value::as_u64);
    let kernel = attrs.get("kernel_size").and_then(serde_json::Value::as_u64);
    let mode = attrs.get("mode").and_then(serde_json::Value::as_str);
    if fields.len() != 6
        || !channels.is_some_and(|width| (1..=MAX_CONVOLUTION_CHANNELS).contains(&width))
        || !kernel.is_some_and(|width| (1..=MAX_CONVOLUTION_KERNEL).contains(&width))
        || attrs
            .get("weight")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        || attrs.get("bias") != Some(&serde_json::Value::Null)
        || attrs.get("activation").and_then(serde_json::Value::as_str) != Some("silu")
        || !matches!(mode, Some("chunk" | "recurrent"))
        || operation.inputs.len() != 2
        || operation.layer.is_none()
    {
        return false;
    }
    let channels = channels.expect("bounded channel count");
    let kernel = kernel.expect("bounded kernel size");
    match operation.operator {
        ModelOperator::CausalConvolution => {
            operation.state_kind.is_none()
                && operation.output_shape.len() == 3
                && operation.output_shape[0] == 1
                && operation.output_shape[2] == channels
                && (1..=MAX_CONVOLUTION_TOKENS).contains(&operation.output_shape[1])
                && operation.output_shape[1]
                    .checked_mul(channels)
                    .is_some_and(|elements| elements <= 1 << 22)
        }
        ModelOperator::ConvolutionStateUpdate => {
            operation.state_kind == Some(StateKind::Convolution)
                && operation.output_shape == [1, channels, kernel]
        }
        _ => false,
    }
}

fn bounded_delta_decay(operation: &ModelOperation) -> bool {
    let attrs = &operation.attributes;
    let Some(fields) = attrs.as_object() else {
        return false;
    };
    fields.len() == 4
        && attrs
            .get("a_log")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| !v.is_empty())
        && attrs
            .get("dt_bias")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| !v.is_empty())
        && attrs["formula"] == "-exp(A_log)*softplus(a+dt_bias)"
        && attrs["compute_dtype"] == "float32"
        && operation.inputs.len() == 1
        && operation.layer.is_some()
        && operation.state_kind.is_none()
        && matches!(operation.output_shape.as_slice(), [1, tokens, heads]
            if (1..=MAX_CONVOLUTION_TOKENS).contains(tokens) && (1..=64).contains(heads))
}

fn bounded_delta_operator(operation: &ModelOperation) -> bool {
    let attrs = &operation.attributes;
    let Some(fields) = attrs.as_object() else {
        return false;
    };
    let Some(scale) = attrs
        .get("query_scale")
        .and_then(serde_json::Value::as_object)
    else {
        return false;
    };
    let denominator = scale
        .get("sqrt_denominator")
        .and_then(serde_json::Value::as_u64);
    let repeats = attrs
        .get("key_head_repeats")
        .and_then(serde_json::Value::as_u64);
    let valid_shape = match operation.operator {
        ModelOperator::GatedDeltaStateUpdate => matches!(operation.output_shape.as_slice(),
            [1, heads, key, value] if (1..=64).contains(heads)
                && (1..=256).contains(key) && (1..=256).contains(value)
                && heads.checked_mul(*key).and_then(|v| v.checked_mul(*value))
                    .is_some_and(|count| count <= 2 * 1024 * 1024)),
        ModelOperator::GatedDeltaRule => matches!(operation.output_shape.as_slice(),
            [1, tokens, heads, value] if (1..=MAX_CONVOLUTION_TOKENS).contains(tokens)
                && (1..=64).contains(heads) && (1..=256).contains(value)),
        _ => false,
    };
    fields.len() == 7
        && scale.len() == 2
        && scale.get("numerator").and_then(serde_json::Value::as_u64) == Some(1)
        && denominator.is_some_and(|width| (1..=256).contains(&width))
        && repeats.is_some_and(|width| (1..=16).contains(&width))
        && attrs["qk_l2_normalize"] == true
        && attrs["normalization_epsilon"] == "0.000001"
        && attrs["chunk_size"] == 64
        && matches!(
            attrs.get("mode").and_then(serde_json::Value::as_str),
            Some("chunked" | "recurrent")
        )
        && attrs["state_update"] == "S'=exp(g)*S+k*(v-S^T*k)*beta"
        && operation.inputs.len() == 6
        && operation.layer.is_some()
        && valid_shape
        && match operation.operator {
            ModelOperator::GatedDeltaStateUpdate => {
                operation.state_kind == Some(StateKind::Recurrent)
            }
            ModelOperator::GatedDeltaRule => operation.state_kind.is_none(),
            _ => false,
        }
}

fn bounded_state_initializer(operation: &ModelOperation) -> bool {
    let state_kind = operation
        .attributes
        .get("state_kind")
        .and_then(serde_json::Value::as_str);
    if operation.operator != ModelOperator::StateInitialize
        || !operation.inputs.is_empty()
        || operation.layer.is_none()
        || operation.state_kind.is_some()
        || !matches!(state_kind, Some("convolution" | "recurrent"))
        || operation.attributes
            != serde_json::json!({
                "initial_value": 0,
                "dtype": "float32",
                "state_kind": state_kind
            })
    {
        return false;
    }
    match (state_kind, operation.output_shape.as_slice()) {
        (Some("convolution"), [batch, channels, kernel]) => {
            *batch == 1
                && (1..=MAX_CONVOLUTION_CHANNELS).contains(channels)
                && (1..=MAX_CONVOLUTION_KERNEL).contains(kernel)
        }
        (Some("recurrent"), [batch, heads, key, value]) => {
            *batch == 1
                && (1..=64).contains(heads)
                && (1..=256).contains(key)
                && (1..=256).contains(value)
                && heads
                    .checked_mul(*key)
                    .and_then(|count| count.checked_mul(*value))
                    .is_some_and(|count| count <= 2 * 1024 * 1024)
        }
        _ => false,
    }
}

fn validate_causal_convolution_contract(graph: &DecoderGraph) -> Result<(), String> {
    let convolutions: Vec<_> = graph
        .operations
        .iter()
        .filter(|row| row.operator == ModelOperator::CausalConvolution)
        .collect();
    let updates: Vec<_> = graph
        .operations
        .iter()
        .filter(|row| row.operator == ModelOperator::ConvolutionStateUpdate)
        .collect();
    if convolutions.len() != updates.len() {
        return Err("causal convolution lacks an exact state-update partner".into());
    }
    let expected_mode = match graph.mode {
        DecoderMode::Prefill => "chunk",
        DecoderMode::Decode => "recurrent",
    };
    for convolution in convolutions {
        if !bounded_convolution_operator(convolution) {
            return Err("causal convolution numeric contract is unsupported".into());
        }
        let state = graph
            .state_inputs
            .iter()
            .find(|state| state.id == convolution.inputs[1]);
        let initialized = graph
            .operations
            .iter()
            .find(|row| row.id == convolution.inputs[1]);
        let paired: Vec<_> = updates
            .iter()
            .filter(|row| row.inputs == convolution.inputs && row.layer == convolution.layer)
            .collect();
        if paired.len() != 1 {
            return Err("causal convolution has no unique state update".into());
        }
        let update = paired[0];
        let result = graph
            .state_outputs
            .iter()
            .find(|row| row.id == update.id)
            .ok_or("causal convolution state has no persistent output")?;
        let channels = convolution.attributes["groups"]
            .as_u64()
            .ok_or("causal convolution channel count is invalid")?;
        let kernel = convolution.attributes["kernel_size"]
            .as_u64()
            .ok_or("causal convolution kernel size is invalid")?;
        let projection = graph
            .operations
            .iter()
            .find(|row| row.id == convolution.inputs[0])
            .ok_or("causal convolution has no declared projection")?;
        if !bounded_convolution_operator(convolution)
            || !bounded_convolution_operator(update)
            || convolution.attributes != update.attributes
            || convolution.attributes["mode"] != expected_mode
            || convolution.output_shape != [1, graph.query_sequence, channels]
            || projection.output_shape != convolution.output_shape
            || result.kind != StateKind::Convolution
            || result.layer != convolution.layer
            || result.maximum_sequence != kernel
            || update.output_shape != result.shape
            || result.shape != [1, channels, kernel]
        {
            return Err("causal convolution and state shape or numeric contract differ".into());
        }
        match graph.mode {
            DecoderMode::Prefill => {
                let initializer =
                    initialized.ok_or("causal convolution prefill omits zero state")?;
                if state.is_some()
                    || !bounded_state_initializer(initializer)
                    || initializer.attributes["state_kind"] != "convolution"
                    || initializer.layer != convolution.layer
                    || initializer.output_shape != result.shape
                {
                    return Err(
                        "causal convolution prefill state is not explicitly initialized".into(),
                    );
                }
            }
            DecoderMode::Decode => {
                let prior = state.ok_or("causal convolution decode lacks retained state")?;
                if initialized.is_some()
                    || prior.kind != StateKind::Convolution
                    || prior.layer != convolution.layer
                    || prior.shape != result.shape
                    || prior.maximum_sequence != kernel
                {
                    return Err(
                        "causal convolution decode state does not match its producer".into(),
                    );
                }
            }
        }
    }
    let bytes = graph
        .state_outputs
        .iter()
        .filter(|state| matches!(state.kind, StateKind::Convolution | StateKind::Recurrent))
        .try_fold(0_u64, |total, state| {
            let size = state
                .shape
                .iter()
                .try_fold(4_u64, |product, width| product.checked_mul(*width))?;
            total.checked_add(size)
        })
        .ok_or("hybrid state capacity overflowed")?;
    if bytes > MAX_WINDOW_STATE_BYTES {
        return Err("hybrid decoder state exceeds the client memory ceiling".into());
    }
    Ok(())
}

fn validate_gated_delta_contract(graph: &DecoderGraph) -> Result<(), String> {
    let rules: Vec<_> = graph
        .operations
        .iter()
        .filter(|row| row.operator == ModelOperator::GatedDeltaRule)
        .collect();
    let updates: Vec<_> = graph
        .operations
        .iter()
        .filter(|row| row.operator == ModelOperator::GatedDeltaStateUpdate)
        .collect();
    if rules.len() != updates.len() {
        return Err("gated-delta rule lacks an exact state-update partner".into());
    }
    let expected_mode = match graph.mode {
        DecoderMode::Prefill => "chunked",
        DecoderMode::Decode => "recurrent",
    };
    for rule in rules {
        if !bounded_delta_operator(rule) {
            return Err("gated-delta numeric contract is unsupported".into());
        }
        let paired: Vec<_> = updates
            .iter()
            .filter(|row| row.inputs == rule.inputs && row.layer == rule.layer)
            .collect();
        if paired.len() != 1 {
            return Err("gated-delta rule has no unique state update".into());
        }
        let update = paired[0];
        let [_, batch_tokens, value_heads, value_dim] = rule.output_shape.as_slice() else {
            return Err("gated-delta output shape is invalid".into());
        };
        let [_, state_heads, key_dim, state_value_dim] = update.output_shape.as_slice() else {
            return Err("gated-delta state shape is invalid".into());
        };
        let repeats = rule.attributes["key_head_repeats"]
            .as_u64()
            .ok_or("gated-delta head ratio is invalid")?;
        let key_heads = value_heads
            .checked_div(repeats)
            .filter(|heads| *heads > 0 && heads * repeats == *value_heads)
            .ok_or("gated-delta head geometry is invalid")?;
        let expected_width = key_heads
            .checked_mul(*key_dim)
            .ok_or("gated-delta key width overflowed")?;
        let expected_value_width = value_heads
            .checked_mul(*value_dim)
            .ok_or("gated-delta value width overflowed")?;
        let producers: Vec<_> = rule.inputs[..5]
            .iter()
            .map(|id| graph.operations.iter().find(|row| &row.id == id))
            .collect();
        if producers.iter().any(Option::is_none) {
            return Err("gated-delta input has no declared producer".into());
        }
        let producer = |index: usize| producers[index].expect("all producers were checked");
        let state = graph
            .state_inputs
            .iter()
            .find(|row| row.id == rule.inputs[5]);
        let initialized = graph.operations.iter().find(|row| row.id == rule.inputs[5]);
        let output = graph
            .state_outputs
            .iter()
            .find(|row| row.id == update.id)
            .ok_or("gated-delta state has no persistent output")?;
        if !bounded_delta_operator(update)
            || update.attributes != rule.attributes
            || rule.attributes["mode"] != expected_mode
            || *batch_tokens != graph.query_sequence
            || *value_heads != *state_heads
            || *value_dim != *state_value_dim
            || rule.attributes["query_scale"]["sqrt_denominator"] != *key_dim
            || producer(0).output_shape != [1, *batch_tokens, expected_width]
            || producer(1).output_shape != producer(0).output_shape
            || producer(2).output_shape != [1, *batch_tokens, expected_value_width]
            || producer(3).operator != ModelOperator::GatedDeltaDecay
            || !bounded_delta_decay(producer(3))
            || producer(4).operator != ModelOperator::Sigmoid
            || producer(4).attributes != serde_json::json!({})
            || producer(3).output_shape != [1, *batch_tokens, *value_heads]
            || producer(4).output_shape != producer(3).output_shape
            || output.kind != StateKind::Recurrent
            || output.layer != rule.layer
            || output.maximum_sequence != 1
            || output.shape != update.output_shape
        {
            return Err(
                "gated-delta operators or retained state violate their declared numeric contract"
                    .into(),
            );
        }
        match graph.mode {
            DecoderMode::Prefill => {
                let initial = initialized.ok_or("gated-delta prefill omits zero state")?;
                if state.is_some()
                    || !bounded_state_initializer(initial)
                    || initial.attributes["state_kind"] != "recurrent"
                    || initial.layer != rule.layer
                    || initial.output_shape != output.shape
                {
                    return Err("gated-delta prefill state is not explicitly initialized".into());
                }
            }
            DecoderMode::Decode => {
                let prior = state.ok_or("gated-delta decode lacks retained state")?;
                if initialized.is_some()
                    || prior.kind != StateKind::Recurrent
                    || prior.layer != rule.layer
                    || prior.shape != output.shape
                    || prior.maximum_sequence != 1
                {
                    return Err("gated-delta decode state does not match its producer".into());
                }
            }
        }
    }
    Ok(())
}

fn validate_causal_convolution_handoff(plan: &DecoderPlan) -> Result<(), String> {
    let slots = |states: &[StateTensor]| -> Result<BTreeMap<u64, (Vec<u64>, u64)>, String> {
        let mut bound = BTreeMap::new();
        for state in states
            .iter()
            .filter(|row| row.kind == StateKind::Convolution)
        {
            let layer = state.layer.ok_or("convolution state lacks a layer")?;
            if bound
                .insert(layer, (state.shape.clone(), state.maximum_sequence))
                .is_some()
            {
                return Err("convolution state has duplicate layer ownership".into());
            }
        }
        Ok(bound)
    };
    if slots(&plan.prefill.state_outputs)? != slots(&plan.decode.state_inputs)? {
        return Err("convolution prefill and decode state contracts differ".into());
    }
    let kernels = |graph: &DecoderGraph| -> Result<BTreeMap<u64, serde_json::Map<String, serde_json::Value>>, String> {
        let mut bound = BTreeMap::new();
        for operation in graph
            .operations
            .iter()
            .filter(|row| row.operator == ModelOperator::CausalConvolution)
        {
            let layer = operation.layer.ok_or("causal convolution lacks a layer")?;
            let mut attrs = operation
                .attributes
                .as_object()
                .ok_or("causal convolution has no numeric descriptor")?
                .clone();
            attrs.remove("mode");
            if bound.insert(layer, attrs).is_some() {
                return Err("causal convolution has duplicate layer ownership".into());
            }
        }
        Ok(bound)
    };
    if kernels(&plan.prefill)? != kernels(&plan.decode)? {
        return Err("convolution prefill and decode kernels differ".into());
    }
    let recurrent_slots =
        |states: &[StateTensor]| -> Result<BTreeMap<u64, (Vec<u64>, u64)>, String> {
            let mut bound = BTreeMap::new();
            for state in states.iter().filter(|row| row.kind == StateKind::Recurrent) {
                let layer = state.layer.ok_or("gated-delta state lacks a layer")?;
                if bound
                    .insert(layer, (state.shape.clone(), state.maximum_sequence))
                    .is_some()
                {
                    return Err("gated-delta state has duplicate layer ownership".into());
                }
            }
            Ok(bound)
        };
    if recurrent_slots(&plan.prefill.state_outputs)? != recurrent_slots(&plan.decode.state_inputs)?
    {
        return Err("gated-delta prefill and decode state contracts differ".into());
    }
    for operator in [
        ModelOperator::GatedDeltaRule,
        ModelOperator::GatedDeltaDecay,
    ] {
        let attrs_by_layer = |graph: &DecoderGraph| -> Result<
            BTreeMap<u64, serde_json::Map<String, serde_json::Value>>,
            String,
        > {
            let mut bound = BTreeMap::new();
            for row in graph
                .operations
                .iter()
                .filter(|row| row.operator == operator)
            {
                let layer = row.layer.ok_or("gated-delta operator lacks a layer")?;
                let mut attrs = row
                    .attributes
                    .as_object()
                    .ok_or("gated-delta attrs are invalid")?
                    .clone();
                attrs.remove("mode");
                if bound.insert(layer, attrs).is_some() {
                    return Err("gated-delta operator repeats its layer".into());
                }
            }
            Ok(bound)
        };
        if attrs_by_layer(&plan.prefill)? != attrs_by_layer(&plan.decode)? {
            return Err("gated-delta coefficients or rule change across decoder phases".into());
        }
    }
    Ok(())
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
        ModelOperator::StateInitialize => bounded_state_initializer(operation),
        ModelOperator::Sigmoid => {
            operation.attributes == serde_json::json!({})
                && operation.inputs.len() == 1
                && matches!(operation.output_shape.as_slice(), [1, tokens, width]
                    if (1..=MAX_CONVOLUTION_TOKENS).contains(tokens)
                        && (1..=8192).contains(width))
        }
        ModelOperator::GatedDeltaDecay => bounded_delta_decay(operation),
        ModelOperator::GatedDeltaRule | ModelOperator::GatedDeltaStateUpdate => {
            bounded_delta_operator(operation)
        }
        ModelOperator::RmsNormGated => {
            let attrs = &operation.attributes;
            attrs.as_object().is_some_and(|fields| fields.len() == 5)
                && attrs
                    .get("epsilon")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|value| value.parse::<f64>().ok())
                    .is_some_and(|epsilon| epsilon.is_finite() && epsilon > 0.0 && epsilon <= 0.01)
                && attrs
                    .get("weight")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|weight| !weight.is_empty())
                && attrs["activation"] == "silu"
                && attrs["weight_offset"] == 0
                && attrs["norm_before_gate"] == true
                && operation.inputs.len() == 2
                && operation.layer.is_some()
                && operation.state_kind.is_none()
                && matches!(operation.output_shape.as_slice(), [1, tokens, heads, value]
                    if (1..=MAX_CONVOLUTION_TOKENS).contains(tokens)
                        && (1..=64).contains(heads) && (1..=256).contains(value))
        }
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
            attrs.as_object().is_some_and(|fields| {
                fields.is_empty()
                    || (fields.len() == 1
                        && attrs.get("axis").and_then(serde_json::Value::as_i64) == Some(-1))
            }) || (attrs.as_object().is_some_and(|fields| fields.len() == 2)
                && attrs.get("axis").and_then(serde_json::Value::as_i64) == Some(-1)
                && attrs
                    .get("compute_dtype")
                    .and_then(serde_json::Value::as_str)
                    == Some("float32"))
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
            let end = operation
                .attributes
                .get("end")
                .and_then(serde_json::Value::as_u64);
            let axis = operation
                .attributes
                .get("axis")
                .and_then(serde_json::Value::as_i64);
            let squeeze = operation
                .attributes
                .get("squeeze")
                .and_then(serde_json::Value::as_bool);
            matches!(operation.output_shape.len(), 3 | 4)
                && match (start, end, axis, squeeze) {
                    (Some(start), Some(end), Some(2), Some(true)) => {
                        operation.output_shape.len() == 3 && start.checked_add(1) == Some(end)
                    }
                    (Some(start), Some(end), Some(-1), Some(false)) => end > start,
                    _ => false,
                }
        }
        ModelOperator::CausalConvolution | ModelOperator::ConvolutionStateUpdate => {
            bounded_convolution_operator(operation)
        }
        ModelOperator::RotaryEmbedding => {
            let attrs = &operation.attributes;
            if ["mrope_interleaved", "mrope_section"]
                .iter()
                .any(|field| attrs.get(*field).is_some())
            {
                let Some(sections) = attrs
                    .get("mrope_section")
                    .and_then(serde_json::Value::as_array)
                else {
                    return false;
                };
                let [1, heads, tokens, width] = operation.output_shape.as_slice() else {
                    return false;
                };
                let Some(rotary) = attrs
                    .get("rotary_dimensions")
                    .and_then(serde_json::Value::as_u64)
                else {
                    return false;
                };
                let Some(theta) = attrs.get("theta").and_then(serde_json::Value::as_f64) else {
                    return false;
                };
                let factor = attrs
                    .get("partial_rotary_factor")
                    .and_then(serde_json::Value::as_str);
                let scaled = match factor {
                    Some("0.25") if width % 4 == 0 => width / 4,
                    Some("1") => *width,
                    _ => return false,
                };
                return attrs.as_object().is_some_and(|fields| fields.len() == 7)
                    && attrs["rope_type"] == "default"
                    && attrs["mrope_interleaved"] == true
                    && attrs["position_policy"] == "text_replicated_axes"
                    && operation.inputs.len() == 2
                    && operation.inputs[1] == "input.positions"
                    && (1..=64).contains(heads)
                    && (1..=MAX_CONVOLUTION_TOKENS).contains(tokens)
                    && (2..=256).contains(width)
                    && scaled == rotary
                    && rotary % 2 == 0
                    && theta.is_finite()
                    && (1.0..=1e12).contains(&theta)
                    && sections.len() == 3
                    && sections.iter().all(|value| value.as_u64().is_some())
                    && sections
                        .iter()
                        .try_fold(0_u64, |sum, item| sum.checked_add(item.as_u64()?))
                        == Some(rotary / 2);
            }
            let scaling = attrs
                .get("frequency_scaling")
                .and_then(|scale| scale.get("kind"))
                .and_then(serde_json::Value::as_str);
            match attrs.get("rope_type").and_then(serde_json::Value::as_str) {
                None => matches!(scaling, None | Some("wavelength_transition")),
                Some("longrope") => scaling == Some("per_frequency_context"),
                Some("default" | "proportional") => scaling.is_none(),
                _ => false,
            }
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
        if operation.operator == ModelOperator::RotaryEmbedding
            && operation
                .attributes
                .get("frequency_scaling")
                .and_then(|scale| scale.get("kind"))
                .and_then(serde_json::Value::as_str)
                == Some("per_frequency_context")
        {
            let original = operation.attributes["frequency_scaling"]
                ["original_max_position_embeddings"]
                .as_u64()
                .ok_or("per-frequency rotary context is not bound")?;
            if graph.maximum_key_sequence > original {
                return Err("per-frequency rotary crossing requires cache re-rotation".into());
            }
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
    if !plan.prefill.state_inputs.is_empty() {
        return Err("prefill declares persistent state without an initialization step".into());
    }
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
    validate_causal_convolution_contract(&plan.prefill)?;
    validate_causal_convolution_contract(&plan.decode)?;
    validate_gated_delta_contract(&plan.prefill)?;
    validate_gated_delta_contract(&plan.decode)?;
    validate_causal_convolution_handoff(plan)?;
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
    use pllm_types::canonical_bytes;

    #[test]
    fn bounded_text_only_hybrid_decoder_builds_a_semantic_schedule() {
        let plan = lower_model_json(
            include_bytes!("../../pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json"),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 5,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        for graph in [&plan.prefill, &plan.decode] {
            validate_causal_convolution_contract(graph).unwrap();
            validate_gated_delta_contract(graph).unwrap();
            let recurrence = graph
                .operations
                .iter()
                .find(|row| row.operator == ModelOperator::GatedDeltaRule)
                .unwrap();
            let retained = graph
                .operations
                .iter()
                .find(|row| row.operator == ModelOperator::GatedDeltaStateUpdate)
                .unwrap();
            assert!(local_operator(recurrence));
            assert!(local_operator(retained));
            let mut wrong_scale = recurrence.clone();
            wrong_scale.attributes["query_scale"]["sqrt_denominator"] = serde_json::json!(256);
            assert!(local_operator(&wrong_scale));
            let mut forged_scale = graph.clone();
            forged_scale
                .operations
                .iter_mut()
                .find(|row| row.id == recurrence.id)
                .unwrap()
                .attributes["query_scale"]["sqrt_denominator"] = serde_json::json!(256);
            assert!(validate_gated_delta_contract(&forged_scale).is_err());
            let mut incorrect_pair = graph.clone();
            incorrect_pair
                .operations
                .iter_mut()
                .find(|row| row.id == retained.id)
                .unwrap()
                .inputs[5] = "foreign.state".into();
            assert!(validate_gated_delta_contract(&incorrect_pair).is_err());
            let mut unbound_decay = graph.clone();
            unbound_decay
                .operations
                .iter_mut()
                .find(|row| row.operator == ModelOperator::GatedDeltaDecay)
                .unwrap()
                .attributes["formula"] = serde_json::json!("unknown");
            assert!(validate_gated_delta_contract(&unbound_decay).is_err());
            let convolution = graph
                .operations
                .iter()
                .find(|row| row.operator == ModelOperator::CausalConvolution)
                .unwrap();
            let update = graph
                .operations
                .iter()
                .find(|row| row.operator == ModelOperator::ConvolutionStateUpdate)
                .unwrap();
            assert!(local_operator(convolution));
            assert!(local_operator(update));
            let mut incorrect_mode = convolution.clone();
            incorrect_mode.attributes["mode"] = serde_json::json!("unbound");
            assert!(!local_operator(&incorrect_mode));
            let mut incorrect_shape = update.clone();
            incorrect_shape.output_shape[1] += 1;
            assert!(!local_operator(&incorrect_shape));
            let mut incorrect_pair = graph.clone();
            let changed = incorrect_pair
                .operations
                .iter_mut()
                .find(|row| row.id == update.id)
                .unwrap();
            changed.attributes["kernel_size"] = serde_json::json!(3);
            assert!(validate_causal_convolution_contract(&incorrect_pair).is_err());
            let mut forged = graph.clone();
            let changed = forged
                .operations
                .iter_mut()
                .find(|row| row.id == convolution.id)
                .unwrap();
            changed.inputs.clear();
            assert!(validate_causal_convolution_contract(&forged).is_err());
            let partial_rotary = graph
                .operations
                .iter()
                .find(|row| row.operator == ModelOperator::RotaryEmbedding)
                .unwrap();
            assert!(local_operator(partial_rotary));
            let mut external_axes = partial_rotary.clone();
            external_axes.attributes["position_policy"] =
                serde_json::json!("three_independent_axes");
            assert!(!local_operator(&external_axes));
            let mut different_section = partial_rotary.clone();
            different_section.attributes["mrope_section"] = serde_json::json!([11, 11, 11]);
            assert!(!local_operator(&different_section));
        }
        validate_causal_convolution_handoff(&plan).unwrap();
        let mut wrong_handoff = plan.clone();
        wrong_handoff.decode.state_inputs[0].shape[1] += 1;
        assert!(validate_causal_convolution_handoff(&wrong_handoff).is_err());
        let mut wrong_kernel = plan.clone();
        wrong_kernel
            .decode
            .operations
            .iter_mut()
            .find(|row| row.operator == ModelOperator::CausalConvolution)
            .unwrap()
            .attributes["weight"] = serde_json::json!("forged.other_weight");
        assert!(validate_causal_convolution_handoff(&wrong_kernel).is_err());
        let mut wrong_recurrent = plan.clone();
        wrong_recurrent
            .decode
            .state_inputs
            .iter_mut()
            .find(|row| row.kind == StateKind::Recurrent)
            .unwrap()
            .shape[2] += 1;
        assert!(validate_causal_convolution_handoff(&wrong_recurrent).is_err());
        let mut wrong_decay_weight = plan.clone();
        wrong_decay_weight
            .decode
            .operations
            .iter_mut()
            .find(|row| row.operator == ModelOperator::GatedDeltaDecay)
            .unwrap()
            .attributes["a_log"] = serde_json::json!("foreign.coefficient");
        assert!(validate_causal_convolution_handoff(&wrong_decay_weight).is_err());
        let composition = canonical_bytes(&serde_json::json!({
            "components": {
                "inference": {"component": "pllm/inference", "params": {}},
                "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
                "linear": {"component": "pllm/masked-linear", "params": {}},
                "preparation": {"component": "pllm/model-aware-corrections", "params": {}}
            },
            "model": {"source": "model.fixture"}
        }));
        let schedule = lower_decoder_runtime_schedule(&plan, &composition).unwrap();
        assert_eq!(schedule.prefill.state_inputs.len(), 0);
        assert_eq!(
            schedule.prefill.state_outputs.len(),
            schedule.decode.state_inputs.len()
        );
        assert_eq!(schedule.prefill.state_outputs.len(), 64);
        assert!(schedule.prefill.steps.iter().any(|step| step
            .operation_ids
            .iter()
            .any(|id| id.ends_with("gated_delta_rule"))));
        let state_operation = plan
            .decode
            .operations
            .iter()
            .find(|row| row.operator == ModelOperator::ConvolutionStateUpdate)
            .unwrap();
        assert!(schedule
            .decode
            .steps
            .iter()
            .any(|step| step.operation_ids.contains(&state_operation.id)));
    }

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
