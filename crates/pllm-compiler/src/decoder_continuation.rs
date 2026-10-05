//! Full-KV continuation is an extension of an admitted prefill schedule, not decode batching.
use crate::{lower_decoder_runtime_schedule, DecoderRuntimePhaseSchedule};
use pllm_models::{DecoderGraph, DecoderPlan, ModelOperator, StateKind};
use pllm_types::{canonical_bytes, canonical_digest, Digest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const DECODER_CONTINUATION_SCHEMA: &str = "pllm.decoder_continuation.v1";

fn is_false(value: &bool) -> bool {
    !value
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderContinuation {
    pub schema_version: String,
    pub mode: String,
    pub source_plan_digest: Digest,
    pub model_config_digest: Digest,
    pub composition_digest: Digest,
    pub source_schedule_digest: Digest,
    pub numeric_digest: Digest,
    pub token_bound: u64,
    pub state_row_bytes: u64,
    pub working_bytes_bound: u64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub generated_prefix_canonical: bool,
    pub graph: DecoderGraph,
    pub schedule: DecoderRuntimePhaseSchedule,
}

impl DecoderContinuation {
    pub fn digest(&self) -> Digest {
        canonical_digest(DECODER_CONTINUATION_SCHEMA, self)
    }

    /// Preflight includes independently restored prefix, geometrically allocated KV,
    /// append views, live scheduled outputs and bounded per-step scratch allowance.
    /// This is an owned-array admission bound, not measured RSS or a weight bound.
    pub fn admit(&self, prefix: u64, query: u64, memory_bytes: u64) -> Result<u64, String> {
        let total = prefix
            .checked_add(query)
            .ok_or("continuation token count overflow")?;
        if prefix == 0 || query == 0 || total > self.token_bound {
            return Err(
                "continuation prefix plus suffix exceeds fixed compiler input bound".into(),
            );
        }
        let capacity = total
            .checked_next_power_of_two()
            .ok_or("continuation capacity overflow")?
            .max(64);
        let rows = prefix
            .checked_add(capacity)
            .and_then(|n| n.checked_add(total))
            .ok_or("continuation state memory overflow")?;
        let required = rows
            .checked_mul(self.state_row_bytes)
            .and_then(|n| n.checked_add(self.working_bytes_bound))
            .ok_or("continuation memory overflow")?;
        if required > memory_bytes || required > 2 << 30 {
            return Err("continuation exceeds client owned-array memory budget".into());
        }
        Ok(required)
    }
}

pub fn lower_decoder_continuation(
    plan: &DecoderPlan,
    composition: &[u8],
) -> Result<DecoderContinuation, String> {
    if !matches!(
        crate::classify_decoder_composition(composition)?,
        crate::DecoderCompositionKind::MaskedLinear
            | crate::DecoderCompositionKind::TwoOnlineOffsetLinear
            | crate::DecoderCompositionKind::VerifiedMaskedLinear
    ) {
        return Err("continuation requires admitted public linear composition".into());
    }
    let source = lower_decoder_runtime_schedule(plan, composition)?;
    let bound = plan.prefill.query_sequence;
    if !(2..=4096).contains(&bound) || plan.prefill.maximum_key_sequence != bound {
        return Err("continuation requires fixed compiler input token bound in [2, 4096]".into());
    }
    let mut owners = BTreeSet::new();
    let mut row_bytes = 0_u64;
    for state in &plan.prefill.state_outputs {
        let layer = state
            .layer
            .ok_or("continuation state lacks exclusive layer owner")?;
        if !matches!(state.kind, StateKind::Key | StateKind::Value)
            || state.shape.len() != 4
            || state.shape[0] != 1
            || state.shape[2] != state.maximum_sequence
            || state.maximum_sequence < bound
            || !owners.insert((layer, state.kind))
        {
            return Err("continuation supports exclusively owned full key/value state only".into());
        }
        row_bytes = row_bytes
            .checked_add(
                state.shape[1]
                    .checked_mul(state.shape[3])
                    .and_then(|n| n.checked_mul(4))
                    .ok_or("continuation state width overflow")?,
            )
            .ok_or("continuation state width overflow")?;
    }
    if owners.is_empty()
        || owners.iter().any(|(layer, _)| {
            !owners.contains(&(*layer, StateKind::Key))
                || !owners.contains(&(*layer, StateKind::Value))
        })
    {
        return Err("continuation requires complete key/value owner pairs".into());
    }
    for state in plan
        .decode
        .state_inputs
        .iter()
        .chain(&plan.decode.state_outputs)
    {
        if !owners.contains(&(
            state.layer.ok_or("continuation state owner missing")?,
            state.kind,
        )) {
            return Err("continuation rejects hybrid, recurrent, sliding or shared state".into());
        }
        let original = plan
            .prefill
            .state_outputs
            .iter()
            .find(|s| s.layer == state.layer && s.kind == state.kind)
            .ok_or("continuation decode state owner mismatch")?;
        if state.shape.len() != 4
            || state.shape[0] != 1
            || state.shape[1] != original.shape[1]
            || state.shape[3] != original.shape[3]
            || state.maximum_sequence < bound
            || state.shape[2] != state.maximum_sequence
        {
            return Err("continuation decode full-KV geometry mismatch".into());
        }
    }
    if plan.decode.state_inputs.len() != owners.len()
        || plan.decode.state_outputs.len() != owners.len()
    {
        return Err("continuation state ownership differs across phases".into());
    }
    let mut graph = plan.prefill.clone();
    let mut schedule = source.prefill.clone();
    // Continuation exposes per-row logits/checkpoints, including intermediate
    // prefix boundaries. Its demand contract therefore retains every row.
    for step in &mut schedule.steps {
        step.terminal_row_only = false;
    }
    let mut appended = BTreeSet::new();
    for op in &mut graph.operations {
        if matches!(
            op.operator,
            ModelOperator::AttentionScores | ModelOperator::AttentionValues
        ) && !op.layer.is_some_and(|layer| {
            owners.contains(&(layer, StateKind::Key)) && owners.contains(&(layer, StateKind::Value))
        }) {
            return Err("continuation rejects shared attention state".into());
        }
        if (op.operator == ModelOperator::CacheSuffix
            && (op.attributes["semantics"] != "visible_valid_prefix" || op.attributes["axis"] != 2))
            || matches!(
                op.operator,
                ModelOperator::StateInitialize
                    | ModelOperator::ConvolutionStateUpdate
                    | ModelOperator::GatedDeltaStateUpdate
            )
        {
            return Err(
                "continuation rejects hybrid, recurrent, sliding or shared state operators".into(),
            );
        }
        if op.operator != ModelOperator::KvCacheAppend {
            continue;
        }
        let owner = (
            op.layer.ok_or("continuation append owner missing")?,
            op.state_kind.ok_or("continuation append kind missing")?,
        );
        if !owners.contains(&owner)
            || !appended.insert(owner)
            || op.inputs.is_empty()
            || op.attributes["mode"] != "initialize"
            || op.attributes["attention_domain"]["layout"] != "batch_kv_heads_sequence_feature"
        {
            return Err(
                "continuation requires one full-KV initialization per exclusive state owner".into(),
            );
        }
        let output = plan
            .prefill
            .state_outputs
            .iter()
            .find(|s| s.id == op.id)
            .ok_or("continuation append lacks declared state output")?;
        let mut input = output.clone();
        input.id = format!("continuation.input.{}", op.id);
        if graph.state_inputs.iter().any(|s| s.id == input.id) {
            return Err("continuation state identity collision".into());
        }
        op.inputs.insert(0, input.id.clone());
        op.attributes["mode"] = serde_json::json!("append");
        graph.state_inputs.push(input);
        let step = schedule
            .steps
            .iter_mut()
            .find(|s| s.operation_ids == [op.id.clone()])
            .ok_or("continuation state step missing")?;
        step.input_ids = op.inputs.clone();
    }
    if appended != owners {
        return Err("continuation state has no unique append producer".into());
    }
    schedule.state_inputs = graph.state_inputs.clone();
    let working_bytes = live_working_bytes(&schedule)?;
    let numeric: serde_json::Value = serde_json::from_slice(composition)
        .map_err(|e| format!("invalid continuation composition: {e}"))?;
    let generated =
        numeric["components"]["cache"]["params"]["generated_prefixes"].as_bool() == Some(true);
    if generated
        && numeric["components"]["quantization"]["params"]["causal_reduction"].as_str()
            != Some("prefix_f32")
    {
        return Err("generated prefix state requires canonical prefix_f32 arithmetic".into());
    }
    Ok(DecoderContinuation {
        schema_version: DECODER_CONTINUATION_SCHEMA.into(),
        mode: "full_kv_batched_suffix".into(),
        source_plan_digest: plan.digest(),
        model_config_digest: plan.config_digest.clone(),
        composition_digest: source.composition_digest.clone(),
        source_schedule_digest: source.digest(),
        numeric_digest: canonical_digest(
            "pllm.decoder_continuation.numeric.v1",
            &serde_json::json!({
                "quantization": numeric["components"]["quantization"],
                "prefill_operations": plan.prefill.operations,
                "decode_operations": plan.decode.operations,
                "attention_order": "legacy_per_query_row",
            }),
        ),
        token_bound: bound,
        state_row_bytes: row_bytes,
        working_bytes_bound: working_bytes,
        generated_prefix_canonical: generated,
        graph,
        schedule,
    })
}

/// Mirrors the executor's step-input reference counting, including fused outputs.
/// Views are charged as copies. Each live output is charged eight bytes per
/// element; a step additionally receives eight four-byte output-sized scratch
/// arrays. This bounds declared array ownership, not allocator/process RSS,
/// immutable weights, tokenizers or cryptographic inventory storage.
fn live_working_bytes(schedule: &DecoderRuntimePhaseSchedule) -> Result<u64, String> {
    let mut remaining = std::collections::BTreeMap::<&str, u64>::new();
    for input in schedule.steps.iter().flat_map(|step| &step.input_ids) {
        *remaining.entry(input).or_default() += 1;
    }
    let mut live = std::collections::BTreeMap::<&str, u64>::new();
    let mut resident = 0_u64;
    let mut peak = 0_u64;
    for step in &schedule.steps {
        let mut scratch = 0_u64;
        for output in &step.outputs {
            let elements = output
                .output_shape
                .iter()
                .try_fold(1_u64, |n, dim| n.checked_mul(*dim))
                .ok_or("continuation output geometry overflow")?;
            let bytes = elements
                .checked_mul(8)
                .ok_or("continuation resident output overflow")?;
            if live.insert(&output.operation_id, bytes).is_some() {
                return Err("continuation output is produced more than once".into());
            }
            resident = resident
                .checked_add(bytes)
                .ok_or("continuation live workspace overflow")?;
            scratch = scratch
                .checked_add(
                    elements
                        .checked_mul(32)
                        .ok_or("continuation scratch overflow")?,
                )
                .ok_or("continuation step scratch overflow")?;
        }
        peak = peak.max(
            resident
                .checked_add(scratch)
                .ok_or("continuation workspace overflow")?,
        );
        for input in &step.input_ids {
            let count = remaining
                .get_mut(input.as_str())
                .ok_or("continuation input count missing")?;
            *count -= 1;
            if *count == 0 && input != &schedule.output {
                if let Some(bytes) = live.remove(input.as_str()) {
                    resident = resident
                        .checked_sub(bytes)
                        .ok_or("continuation liveness underflow")?;
                }
            }
        }
    }
    Ok(peak)
}

pub fn decoder_continuation_bytes(
    plan_bytes: &[u8],
    composition: &[u8],
) -> Result<Vec<u8>, String> {
    if plan_bytes.len() > 16 << 20 || composition.len() > 1 << 20 {
        return Err("continuation metadata exceeds byte bound".into());
    }
    let plan: DecoderPlan = serde_json::from_slice(plan_bytes)
        .map_err(|e| format!("invalid continuation source plan: {e}"))?;
    Ok(canonical_bytes(&lower_decoder_continuation(
        &plan,
        composition,
    )?))
}

pub fn admit_decoder_continuation(
    plan: &[u8],
    composition: &[u8],
    contract: &[u8],
    prefix: u64,
    query: u64,
    memory_bytes: u64,
) -> Result<u64, String> {
    let expected = decoder_continuation_bytes(plan, composition)?;
    if contract != expected {
        return Err("continuation source/numeric/schedule contract mismatch".into());
    }
    let bound: DecoderContinuation =
        serde_json::from_slice(&expected).map_err(|e| e.to_string())?;
    bound.admit(prefix, query, memory_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_models::{lower_model_json, DecoderWorkload};

    fn fixture() -> (DecoderPlan, Vec<u8>) {
        let plan = lower_model_json(
            br#"{"model_type":"qwen2","hidden_size":8,
            "intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,
            "num_key_value_heads":1,"vocab_size":32,"max_position_embeddings":64,
            "hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000.0,
            "tie_word_embeddings":true}"#,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 32,
                max_new_tokens: 4,
            },
        )
        .unwrap();
        let pipeline = canonical_bytes(&serde_json::json!({
            "model":{"source":"model.fixture"}, "components":{
                "inference":{"component":"pllm/inference","params":{}},
                "kernels":{"component":"pllm/cpu","params":{"threads":1}},
                "linear":{"component":"pllm/masked-linear","params":{}},
                "preparation":{"component":"pllm/model-aware-corrections","params":{}}
            }
        }));
        (plan, pipeline)
    }

    #[test]
    fn continuation_is_independently_bound_and_keeps_linear_geometry() {
        let (plan, pipeline) = fixture();
        let source = lower_decoder_runtime_schedule(&plan, &pipeline).unwrap();
        let extension = lower_decoder_continuation(&plan, &pipeline).unwrap();
        assert_ne!(extension.digest(), source.digest());
        assert_eq!(extension.source_schedule_digest, source.digest());
        assert_eq!(extension.graph.state_inputs.len(), 4);
        for (before, after) in source.prefill.steps.iter().zip(&extension.schedule.steps) {
            assert_eq!(before.outputs, after.outputs);
            assert_eq!(before.executor, after.executor);
            assert_eq!(before.operation_ids, after.operation_ids);
        }
        assert!(extension.admit(9, 23, 2 << 30).is_ok());
        assert!(extension.admit(10, 23, 2 << 30).is_err());
        assert!(extension.admit(9, 23, 1).is_err());
        assert!(extension.admit(u64::MAX, 1, u64::MAX).is_err());
        let mut forged = extension.clone();
        forged.token_bound = 4096;
        assert!(admit_decoder_continuation(
            &canonical_bytes(&plan),
            &pipeline,
            &canonical_bytes(&forged),
            9,
            23,
            2 << 30
        )
        .is_err());
    }

    #[test]
    fn continuation_rejects_nonexclusive_state_and_unbounded_width() {
        let (mut plan, pipeline) = fixture();
        plan.prefill.state_outputs[0].shape[1] = u64::MAX;
        assert!(lower_decoder_continuation(&plan, &pipeline).is_err());
        let (mut plan, pipeline) = fixture();
        plan.decode.state_inputs[0].kind = StateKind::Recurrent;
        assert!(lower_decoder_continuation(&plan, &pipeline).is_err());
    }

    #[test]
    fn workspace_keeps_branches_and_fused_outputs_until_last_consumer() {
        let (plan, pipeline) = fixture();
        let mut schedule = lower_decoder_continuation(&plan, &pipeline)
            .unwrap()
            .schedule;
        schedule.steps.truncate(3);
        for (step, id) in schedule.steps.iter_mut().zip(["a", "b", "c"]) {
            step.outputs.truncate(1);
            step.outputs[0].operation_id = id.into();
            step.outputs[0].output_shape = vec![100];
        }
        schedule.output = "c".into();
        schedule.steps[0].input_ids.clear();
        schedule.steps[1].input_ids = vec!["a".into()];
        schedule.steps[2].input_ids = vec!["b".into()];
        assert_eq!(live_working_bytes(&schedule).unwrap(), 4800);
        schedule.steps[2].input_ids.push("a".into());
        assert_eq!(live_working_bytes(&schedule).unwrap(), 5600);
        let mut fused = schedule.steps[1].outputs[0].clone();
        fused.operation_id = "d".into();
        schedule.steps[1].outputs.push(fused);
        schedule.steps[2].input_ids.push("d".into());
        assert_eq!(live_working_bytes(&schedule).unwrap(), 8800);
        schedule.steps[0].outputs[0].output_shape = vec![u64::MAX];
        assert!(live_working_bytes(&schedule).is_err());
    }
}
