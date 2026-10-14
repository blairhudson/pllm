//! Explicit client-local retained-state extension of a bound dense decoder.
//! The protected eviction transform is a different, still unimplemented executor.
use crate::{decoder_continuation::live_working_bytes, lower_decoder_runtime_schedule};
use pllm_models::{
    cache::MpcachePolicy, DecoderGraph, DecoderPlan, ModelOperation, ModelOperator, StateKind,
};
use pllm_types::{canonical_bytes, Digest};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;

const SCHEMA: &str = "pllm.retained_decoder.v1";

pub(crate) fn validate_state_component(pipeline: &crate::ExperimentPipeline) -> Result<(), String> {
    let Some(state) = pipeline.components.get("state") else {
        return Ok(());
    };
    if state.component != "pllm/client-local-kv"
        || state.params.keys().any(|key| key != "retention")
        || !pipeline.components.get("linear").is_some_and(|linear| {
            matches!(
                linear.component.as_str(),
                "pllm/masked-linear" | "pllm/cleartext-linear"
            )
        })
        || ["cache", "public_prefix", "verification", "nonlinear"]
            .iter()
            .any(|slot| pipeline.components.contains_key(*slot))
        || pipeline
            .components
            .get("quantization")
            .is_some_and(|q| q.params.contains_key("causal_reduction"))
    {
        return Err("client state requires unverified response-owned dense execution".into());
    }
    if let Some(policy) = state.params.get("retention") {
        selected_policy(policy)?;
    }
    Ok(())
}

fn selected_policy(policy: &Value) -> Result<MpcachePolicy, String> {
    if policy.as_object().is_none_or(|map| map.len() != 2)
        || policy["component"] != "pllm/kv-cache-eviction"
    {
        return Err("retained decoder requires a KV eviction component".into());
    }
    let mut params = policy["params"]
        .as_object()
        .cloned()
        .ok_or("retained policy parameters are missing")?;
    if params.remove("implementation") != Some(Value::from("pllm/importance-kv-cache-eviction/v1"))
    {
        return Err("retained policy implementation is unsupported".into());
    }
    let selected: MpcachePolicy =
        serde_json::from_value(Value::Object(params)).map_err(|e| e.to_string())?;
    selected.validate().map_err(|e| e.to_string())?;
    if selected.dynamic_keep.numerator != selected.dynamic_keep.denominator
        || selected.share_adjacent_layers
    {
        return Err("retained decoder admits static client-local selection only".into());
    }
    Ok(selected)
}

#[derive(Serialize)]
struct PhaseBinding {
    key: String,
    value: String,
    scores: String,
    mask: String,
    probabilities: String,
    commit_after: String,
}

#[derive(Serialize)]
struct LayerBinding {
    layer: u64,
    kv_heads: u64,
    head_dim: u64,
    prefill: PhaseBinding,
    decode: PhaseBinding,
}

#[derive(Serialize)]
struct RetainedDecoder {
    schema_version: &'static str,
    source_plan_digest: Digest,
    source_schedule_digest: Digest,
    composition_digest: Digest,
    policy: Value,
    selection_executor: &'static str,
    importance_reduction: &'static str,
    state_layout: &'static str,
    max_input_tokens: u64,
    max_decode_steps: u64,
    retained_payload_bound_bytes: u64,
    dense_prefill_payload_bound_bytes: u64,
    conversion_export_bound_bytes: u64,
    workspace_bound_bytes: u64,
    owned_array_bound_bytes: u64,
    layers: Vec<LayerBinding>,
}

fn one(graph: &DecoderGraph, layer: u64, kind: ModelOperator) -> Result<&ModelOperation, String> {
    let mut found = graph
        .operations
        .iter()
        .filter(|op| op.layer == Some(layer) && op.operator == kind);
    let op = found
        .next()
        .ok_or("retained decoder lacks a required layer operation")?;
    if found.next().is_some() {
        return Err("retained decoder requires one operation per semantic layer role".into());
    }
    Ok(op)
}

fn phase(
    graph: &DecoderGraph,
    layer: u64,
    heads: u64,
    width: u64,
    state_capacity: u64,
    prefill: bool,
) -> Result<PhaseBinding, String> {
    let state = |kind| -> Result<&ModelOperation, String> {
        let outputs: Vec<_> = graph
            .state_outputs
            .iter()
            .filter(|s| s.layer == Some(layer) && s.kind == kind)
            .collect();
        if outputs.len() != 1 {
            return Err("retained decoder requires exclusive key/value outputs".into());
        }
        let out = outputs[0];
        let op = graph
            .operations
            .iter()
            .find(|op| op.id == out.id)
            .ok_or("retained state producer is missing")?;
        let mode = if prefill { "initialize" } else { "append" };
        if op.operator != ModelOperator::KvCacheAppend
            || op.layer != Some(layer)
            || op.state_kind != Some(kind)
            || out.shape != [1, heads, state_capacity, width]
            || op.output_shape != out.shape
            || out.maximum_sequence != state_capacity
            || op.attributes["mode"] != mode
            || op.attributes["attention_domain"]
                != serde_json::json!({"includes_current": true, "layout": "batch_kv_heads_sequence_feature", "maximum_sequence": graph.maximum_key_sequence})
            || op.attributes["state_capacity"] != state_capacity
            || op.attributes["state_layout"] != "batch_kv_heads_sequence_feature"
            || op.attributes["absolute_write_positions_input"] != "input.positions"
            || op.attributes["padding_mask_input"] != "input.attention_mask"
            || op.attributes["valid_lengths_input"] != "input.sequence_lengths"
            || op.attributes.get("output_dtype").is_some()
            || op.inputs.len() != if prefill { 4 } else { 5 }
            || op.inputs[op.inputs.len() - 3..]
                != [
                    "input.positions",
                    "input.attention_mask",
                    "input.sequence_lengths",
                ]
        {
            return Err("retained decoder requires explicit full causal KV append geometry".into());
        }
        if !prefill {
            let input = graph
                .state_inputs
                .iter()
                .find(|s| s.id == op.inputs[0])
                .ok_or("retained decoder lacks prior state")?;
            if input.layer != Some(layer)
                || input.kind != kind
                || input.shape != [1, heads, state_capacity, width]
                || input.maximum_sequence != state_capacity
            {
                return Err("retained decoder prior-state ownership differs".into());
            }
        }
        Ok(op)
    };
    let key = state(StateKind::Key)?;
    let value = state(StateKind::Value)?;
    let view = |producer: &ModelOperation| -> Result<&ModelOperation, String> {
        let consumers: Vec<_> = graph
            .operations
            .iter()
            .filter(|op| op.inputs.contains(&producer.id))
            .collect();
        if consumers.len() != 1 {
            return Err("retained state lacks a unique bounded view".into());
        }
        let op = consumers[0];
        if op.operator != ModelOperator::CacheSuffix
            || op.layer != Some(layer)
            || op.state_kind != producer.state_kind
            || op.inputs
                != [
                    producer.id.clone(),
                    "input.positions".into(),
                    "input.attention_mask".into(),
                    "input.sequence_lengths".into(),
                ]
            || op.output_shape != [1, heads, graph.maximum_key_sequence, width]
            || op.attributes
                != serde_json::json!({"axis":2,"maximum_sequence":graph.maximum_key_sequence,
                "semantics":"visible_valid_prefix", "absolute_write_positions_input":"input.positions",
                "padding_mask_input":"input.attention_mask", "valid_lengths_input":"input.sequence_lengths"})
        {
            return Err("retained state view is not a full causal prefix".into());
        }
        Ok(op)
    };
    let key_view = view(key)?;
    let value_view = view(value)?;
    let scores = one(graph, layer, ModelOperator::AttentionScores)?;
    let scale = one(graph, layer, ModelOperator::AttentionScale)?;
    let mask = one(graph, layer, ModelOperator::CausalMask)?;
    let probabilities = one(graph, layer, ModelOperator::Softmax)?;
    let values = one(graph, layer, ModelOperator::AttentionValues)?;
    let group = scores.attributes["group_size"]
        .as_u64()
        .ok_or("retained attention group is missing")?;
    if !(1..=128).contains(&group)
        || heads * group > 128
        || scores.attributes != serde_json::json!({"group_size": group})
        || values.attributes != scores.attributes
        || scores.inputs.len() != 2
        || scores.inputs[1] != key_view.id
        || scale.inputs != [scores.id.clone()]
        || scale.attributes != serde_json::json!({"head_dim": width})
        || mask.inputs != [scale.id.clone(), "input.positions".into()]
        || mask.attributes != serde_json::json!({})
        || probabilities.inputs != [mask.id.clone()]
        || ![
            serde_json::json!({}),
            serde_json::json!({"axis": -1}),
            serde_json::json!({"axis": -1, "compute_dtype": "float32"}),
        ]
        .contains(&probabilities.attributes)
        || values.inputs != [probabilities.id.clone(), value_view.id.clone()]
        || scores.output_shape
            != [
                1,
                heads * group,
                graph.query_sequence,
                graph.maximum_key_sequence,
            ]
        || scale.output_shape != scores.output_shape
        || mask.output_shape != scores.output_shape
        || probabilities.output_shape != scores.output_shape
        || values.output_shape != [1, heads * group, graph.query_sequence, width]
    {
        return Err("retained decoder requires a complete dense float32 attention chain".into());
    }
    // The executor commits both K/V together before scores and retires dense state
    // only after weighted values. Reject additional consumers of persistent views.
    for (id, consumer) in [(&key_view.id, &scores.id), (&value_view.id, &values.id)] {
        if graph
            .operations
            .iter()
            .filter(|op| op.inputs.contains(id))
            .any(|op| &op.id != consumer)
        {
            return Err("retained state has an unbound extra consumer".into());
        }
    }
    let order = |id: &str| {
        graph
            .operations
            .iter()
            .position(|op| op.id == id)
            .unwrap_or(usize::MAX)
    };
    if order(&key.id) >= order(&value.id)
        || order(&value.id) >= order(&key_view.id)
        || order(&value.id) >= order(&scores.id)
    {
        return Err("retained key/value commit order differs".into());
    }
    Ok(PhaseBinding {
        key: key.id.clone(),
        value: value.id.clone(),
        scores: scores.id.clone(),
        mask: mask.id.clone(),
        probabilities: probabilities.id.clone(),
        commit_after: values.id.clone(),
    })
}

pub fn retained_decoder_bytes(
    plan_bytes: &[u8],
    composition: &[u8],
    component: &[u8],
    memory_bytes: u64,
) -> Result<Vec<u8>, String> {
    if plan_bytes.len() > 16 << 20 || composition.len() > 1 << 20 || component.len() > 4096 {
        return Err("retained decoder metadata exceeds its byte bound".into());
    }
    let plan: DecoderPlan = serde_json::from_slice(plan_bytes).map_err(|e| e.to_string())?;
    let mut source = lower_decoder_runtime_schedule(&plan, composition)?;
    source.retained_state = None;
    let policy: Value = serde_json::from_slice(component).map_err(|e| e.to_string())?;
    let numeric: Value = serde_json::from_slice(composition).map_err(|e| e.to_string())?;
    if let Some(bound) = numeric["components"]["state"]["params"].get("retention") {
        if bound != &policy {
            return Err("retained policy differs from Pipeline state".into());
        }
    }
    build_contract(&plan, &source, composition, policy, memory_bytes)
}

pub(crate) fn build_contract(
    plan: &DecoderPlan,
    source: &crate::DecoderRuntimeSchedule,
    composition: &[u8],
    policy: Value,
    memory_bytes: u64,
) -> Result<Vec<u8>, String> {
    let numeric: Value = serde_json::from_slice(composition).map_err(|e| e.to_string())?;
    if source.private_boundary.is_some() {
        return Err("private boundary retained state is not supported".into());
    }
    if !matches!(
        crate::classify_decoder_composition(composition)?,
        crate::DecoderCompositionKind::MaskedLinear
            | crate::DecoderCompositionKind::ClientOnlyLinear
    ) || ["cache", "public_prefix", "nonlinear"]
        .iter()
        .any(|slot| numeric["components"].get(slot).is_some())
        || numeric["components"]["quantization"]["params"]
            .get("causal_reduction")
            .is_some()
    {
        return Err("retained decoder needs an ordinary unverified dense numeric source".into());
    }
    let selected = selected_policy(&policy)?;
    let inputs = plan.prefill.query_sequence;
    let decode = plan
        .decode
        .maximum_key_sequence
        .checked_sub(inputs)
        .ok_or("retained decode bound is invalid")?;
    if !(1..=512).contains(&inputs) || decode > 64 || plan.prefill.maximum_key_sequence != inputs {
        return Err("retained decoder workload exceeds 512 prefill / 64 decode rows".into());
    }
    let ceil = |ratio: &pllm_models::cache::Ratio| {
        ((u128::from(inputs) * u128::from(ratio.numerator)).div_ceil(u128::from(ratio.denominator)))
            as u64
    };
    let capacity = ceil(&selected.static_keep).max(ceil(&selected.observation_window)) + decode;
    let owners: BTreeSet<_> = plan
        .prefill
        .state_outputs
        .iter()
        .filter_map(|state| state.layer)
        .collect();
    if owners.is_empty()
        || owners.len() > 128
        || owners.iter().copied().ne(0..owners.len() as u64)
        || [
            &plan.prefill.state_outputs,
            &plan.decode.state_inputs,
            &plan.decode.state_outputs,
        ]
        .iter()
        .any(|states| {
            states.len() != 2 * owners.len()
                || states.iter().any(|s| {
                    s.layer.is_none() || !matches!(s.kind, StateKind::Key | StateKind::Value)
                })
        })
    {
        return Err("retained decoder requires exclusive paired dense state owners".into());
    }
    let mut layers = Vec::new();
    let mut retained_payload = 0;
    let mut dense_payload = 0;
    let mut conversions = 0;
    for layer in owners {
        let shape = &plan
            .prefill
            .state_outputs
            .iter()
            .find(|s| s.layer == Some(layer))
            .ok_or("state owner missing")?
            .shape;
        if shape.len() != 4 || shape[1] == 0 || shape[1] > 128 || shape[3] == 0 || shape[3] > 8192 {
            return Err("retained state feature layout is invalid".into());
        }
        let width = shape[1] * shape[3];
        retained_payload += pllm_core::retained_kv::storage_bytes(capacity as usize, width as usize)
            .map_err(str::to_string)? as u64;
        dense_payload += inputs.next_power_of_two().max(64) * width * 8;
        // One active layer: contiguous source + immutable bytes + decoded native
        // floats; exports and their positions. Other retained owners are above.
        conversions = conversions.max(3 * inputs * width * 8 + 3 * capacity * (width * 8 + 4));
        layers.push(LayerBinding {
            layer,
            kv_heads: shape[1],
            head_dim: shape[3],
            prefill: phase(
                &plan.prefill,
                layer,
                shape[1],
                shape[3],
                inputs + decode,
                true,
            )?,
            decode: phase(
                &plan.decode,
                layer,
                shape[1],
                shape[3],
                inputs + decode,
                false,
            )?,
        });
    }
    let workspace = live_working_bytes(&source.prefill)?.max(live_working_bytes(&source.decode)?);
    let owned = [retained_payload, dense_payload, conversions, workspace]
        .into_iter()
        .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or("retained memory accounting overflow")?;
    if owned > memory_bytes || owned > 2 << 30 {
        return Err("retained decoder exceeds its state/workspace owned-array budget".into());
    }
    let contract = RetainedDecoder {
        schema_version: SCHEMA,
        source_plan_digest: plan.digest(),
        source_schedule_digest: source.digest(),
        composition_digest: source.composition_digest.clone(),
        policy,
        selection_executor: "trusted_client",
        importance_reduction:
            "mean_last_observation_queries_and_all_heads_f32_stable_topk_with_recent_suffix",
        state_layout: "retained_float32_kv_with_absolute_u32_positions",
        max_input_tokens: inputs,
        max_decode_steps: decode,
        retained_payload_bound_bytes: retained_payload,
        dense_prefill_payload_bound_bytes: dense_payload,
        conversion_export_bound_bytes: conversions,
        workspace_bound_bytes: workspace,
        owned_array_bound_bytes: owned,
        layers,
    };
    Ok(canonical_bytes(&contract))
}
