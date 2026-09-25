use pllm_compiler::{
    lower_decoder_runtime_schedule, DecoderRuntimeExecutor, DecoderRuntimeSchedule,
    DECODER_RUNTIME_SCHEDULE_SCHEMA_VERSION,
};
use pllm_models::{lower_model_json, DecoderPlan, DecoderWorkload, ModelOperator};
use pllm_types::{canonical_bytes, pipeline_digest_bytes};
use serde_json::json;
use std::collections::BTreeSet;

const QWEN2: &[u8] = br#"{
    "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
    "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
    "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
    "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
}"#;

const QWEN3: &[u8] = br#"{
    "model_type":"qwen3","hidden_size":8,"intermediate_size":16,
    "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
    "vocab_size":32,"max_position_embeddings":32,"head_dim":4,
    "hidden_act":"silu","rms_norm_eps":1e-6,"rope_theta":10000.0,
    "tie_word_embeddings":true,"attention_bias":false,"attention_dropout":0.0,
    "rope_scaling":null,"sliding_window":null,"use_sliding_window":false,
    "use_cache":true,"max_window_layers":2,
    "layer_types":["full_attention","full_attention"]
}"#;

const GEMMA4_E2B: &[u8] =
    include_bytes!("../../pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json");
const PHI4_MINI: &[u8] =
    include_bytes!("../../pllm-models/tests/fixtures/Phi-4-mini-instruct-cfbefac-config.json");

fn plan(config: &[u8]) -> DecoderPlan {
    lower_model_json(
        config,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 3,
            max_new_tokens: 2,
        },
    )
    .unwrap()
}

fn composition(verified: bool) -> Vec<u8> {
    let mut pipeline = json!({
        "components": {
            "inference": {"component": "pllm/inference", "params": {}},
            "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
            "linear": {"component": "pllm/masked-linear", "params": {}},
            "preparation": {"component": "pllm/model-aware-corrections", "params": {}}
        },
        "model": {"source": "model.fixture"}
    });
    if verified {
        pipeline["components"]["verification"] = json!({
            "component": "pllm/freivalds-verify/v1",
            "params": {"target_failure_bits": 40}
        });
    }
    canonical_bytes(&pipeline)
}

fn lower_schedule(plan: &DecoderPlan) -> Result<DecoderRuntimeSchedule, String> {
    lower_decoder_runtime_schedule(plan, &composition(false))
}

#[test]
fn wavelength_rotary_stays_local_in_both_compiled_decoder_phases() {
    let scaled = json!({
        "model_type": "llama", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 256, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "rope_scaling": {"rope_type": "llama3", "factor": 8.0,
            "original_max_position_embeddings": 32,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0}
    });
    let plan = plan(&serde_json::to_vec(&scaled).unwrap());
    let schedule = lower_schedule(&plan).unwrap();
    for phase in [&schedule.prefill, &schedule.decode] {
        for rotary in phase
            .steps
            .iter()
            .filter(|step| step.operators.contains(&ModelOperator::RotaryEmbedding))
        {
            assert_eq!(rotary.executor, DecoderRuntimeExecutor::ClientLocal);
        }
        assert_eq!(
            phase
                .steps
                .iter()
                .filter(|step| step.operators.contains(&ModelOperator::RotaryEmbedding))
                .count(),
            2
        );
        assert!(phase
            .steps
            .iter()
            .any(|step| step.executor == DecoderRuntimeExecutor::RemoteStage));
    }
    let mut tampered = plan;
    let rotary = tampered
        .decode
        .operations
        .iter_mut()
        .find(|op| op.operator == ModelOperator::RotaryEmbedding)
        .unwrap();
    rotary.attributes["frequency_scaling"]["kind"] = json!("unreviewed");
    assert!(lower_schedule(&tampered).is_err());
}

#[test]
fn bounded_per_frequency_rotary_and_fused_slices_compile_without_extended_context() {
    let source = plan(PHI4_MINI);
    let schedule = lower_schedule(&source).unwrap();
    for phase in [&schedule.prefill, &schedule.decode] {
        let slices = phase
            .steps
            .iter()
            .filter(|step| step.operators == [ModelOperator::Slice])
            .count();
        let rotary = phase
            .steps
            .iter()
            .filter(|step| step.operators == [ModelOperator::RotaryEmbedding])
            .count();
        assert_eq!(slices, 5 * 32);
        assert_eq!(rotary, 2 * 32);
        assert!(phase.steps.iter().any(|step| {
            step.executor == DecoderRuntimeExecutor::RemoteStage
                && step
                    .operation_ids
                    .iter()
                    .any(|id| id.ends_with("qkv_linear"))
        }));
    }
    let mut forged = source.clone();
    let query = forged
        .prefill
        .operations
        .iter_mut()
        .find(|op| op.id == "layer.0.q_slice")
        .unwrap();
    query.attributes["end"] = json!(5000);
    assert!(lower_schedule(&forged).is_err());

    let extended = lower_model_json(
        PHI4_MINI,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 4096,
            max_new_tokens: 2,
        },
    )
    .unwrap();
    assert!(lower_schedule(&extended)
        .unwrap_err()
        .contains("requires cache re-rotation"));
}

#[test]
fn two_online_offset_topology_binds_remote_stages_for_shared_decoder_operators() {
    let composition = canonical_bytes(&json!({
        "components": {
            "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
            "linear": {"component": "pllm/two-online-offset-linear/v1", "params": {}},
            "topology": {"component": "pllm/two-online-offset-workers/v1", "params": {}},
            "quantization": {"component": "pllm/symmetric-per-row-quantization/v1",
                "params": {"weight_bits": 8, "activation_bits": 8}}
        },
        "model": {"source": "model.fixture"}
    }));
    for config in [QWEN2, QWEN3, GEMMA4_E2B] {
        let schedule = lower_decoder_runtime_schedule(&plan(config), &composition).unwrap();
        assert_eq!(
            schedule.composition_digest,
            pipeline_digest_bytes(&composition)
        );
        for phase in [&schedule.prefill, &schedule.decode] {
            assert!(phase
                .steps
                .iter()
                .any(|step| step.executor == DecoderRuntimeExecutor::RemoteStage));
            assert!(phase
                .steps
                .iter()
                .all(|step| step.executor != DecoderRuntimeExecutor::ClientLinear));
        }
    }
    let mut invalid: serde_json::Value = serde_json::from_slice(&composition).unwrap();
    invalid["components"]["topology"]["component"] = json!("pllm/client-only/v1");
    assert!(lower_decoder_runtime_schedule(&plan(QWEN2), &canonical_bytes(&invalid)).is_err());
}

#[test]
fn client_only_topology_places_all_linear_stages_with_the_client() {
    let composition = canonical_bytes(&json!({
        "components": {
            "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
            "linear": {"component": "pllm/cleartext-linear", "params": {}},
            "topology": {"component": "pllm/client-only/v1", "params": {}},
            "quantization": {"component": "pllm/symmetric-per-row-quantization/v1",
                "params": {"weight_bits": 8, "activation_bits": 8}}
        },
        "model": {"source": "model.fixture"}
    }));
    for config in [QWEN2, QWEN3, GEMMA4_E2B] {
        let plan = plan(config);
        let schedule = lower_decoder_runtime_schedule(&plan, &composition).unwrap();
        assert_eq!(
            schedule.composition_digest,
            pipeline_digest_bytes(&composition)
        );
        for phase in [&schedule.prefill, &schedule.decode] {
            assert!(phase
                .steps
                .iter()
                .any(|step| step.executor == DecoderRuntimeExecutor::ClientLinear));
            assert!(phase
                .steps
                .iter()
                .all(|step| step.executor != DecoderRuntimeExecutor::RemoteStage));
            assert!(phase
                .steps
                .iter()
                .filter(|step| step.executor == DecoderRuntimeExecutor::ClientLinear)
                .all(|step| !step.weight_ids.is_empty()));
        }
    }
    let plan = plan(QWEN2);
    let mut invalid: serde_json::Value = serde_json::from_slice(&composition).unwrap();
    invalid["components"]["preparation"] = json!({
        "component": "pllm/model-aware-corrections", "params": {}
    });
    assert!(lower_decoder_runtime_schedule(&plan, &canonical_bytes(&invalid)).is_err());
    invalid["components"]
        .as_object_mut()
        .unwrap()
        .remove("preparation");
    invalid["components"]["linear"]["component"] = json!("pllm/masked-linear");
    assert!(lower_decoder_runtime_schedule(&plan, &canonical_bytes(&invalid)).is_err());
}

#[test]
fn explicit_prepared_topology_keeps_coverage_but_changes_plan_identity() {
    let plan = plan(QWEN2);
    let original = composition(false);
    let mut pipeline: serde_json::Value = serde_json::from_slice(&original).unwrap();
    pipeline["components"]["topology"] = json!({
        "component": "pllm/one-online-provider-offline-preparation/v1", "params": {}
    });
    let selected = canonical_bytes(&pipeline);
    let baseline = lower_decoder_runtime_schedule(&plan, &original).unwrap();
    let bound = lower_decoder_runtime_schedule(&plan, &selected).unwrap();
    assert!(bound.complete);
    assert_ne!(bound.composition_digest, baseline.composition_digest);
    assert_eq!(bound.composition_digest, pipeline_digest_bytes(&selected));
    assert_eq!(bound.prefill.steps, baseline.prefill.steps);
    assert_eq!(bound.decode.steps, baseline.decode.steps);

    pipeline["components"]["topology"]["component"] = json!("pllm/two-online-workers/v1");
    assert!(lower_decoder_runtime_schedule(&plan, &canonical_bytes(&pipeline)).is_err());
    pipeline["components"]["topology"]["component"] =
        json!("pllm/one-online-provider-offline-preparation/v1");
    pipeline["components"]["topology"]["params"] = json!({"operator": "inference"});
    assert!(lower_decoder_runtime_schedule(&plan, &canonical_bytes(&pipeline)).is_err());
}

#[test]
fn lowers_complete_prefill_and_decode_schedules() {
    let plan = plan(QWEN2);
    let composition = composition(false);
    let schedule = lower_decoder_runtime_schedule(&plan, &composition).unwrap();
    assert_eq!(
        schedule.schema_version,
        DECODER_RUNTIME_SCHEDULE_SCHEMA_VERSION
    );
    assert_eq!(
        schedule.composition_digest,
        pipeline_digest_bytes(&composition)
    );
    assert_eq!(schedule.model_plan_digest, plan.digest());
    assert_eq!(schedule.model_config_digest, plan.config_digest);
    assert!(schedule.complete);
    assert!(!schedule.protected_execution);
    assert_eq!(schedule.prefill.mode, pllm_models::DecoderMode::Prefill);
    assert_eq!(schedule.decode.mode, pllm_models::DecoderMode::Decode);
    assert!(schedule.prefill.state_inputs.is_empty());
    assert_eq!(schedule.prefill.state_outputs.len(), 4);
    assert_eq!(schedule.decode.state_inputs.len(), 4);
    assert_eq!(schedule.decode.state_outputs.len(), 4);

    for (phase, graph) in [
        (&schedule.prefill, &plan.prefill),
        (&schedule.decode, &plan.decode),
    ] {
        let scheduled: Vec<_> = phase
            .steps
            .iter()
            .flat_map(|step| step.operation_ids.iter().map(String::as_str))
            .collect();
        let expected: Vec<_> = graph
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect();
        assert_eq!(scheduled.len(), expected.len());
        assert_eq!(
            scheduled.iter().copied().collect::<BTreeSet<_>>(),
            expected.iter().copied().collect::<BTreeSet<_>>()
        );
        assert!(phase
            .steps
            .iter()
            .enumerate()
            .all(|(index, step)| step.order == index as u64));
        assert_eq!(phase.output, "token_feedback");
        assert_eq!(
            phase.steps.last().unwrap().operators,
            vec![ModelOperator::TokenFeedback]
        );
        assert_eq!(
            phase
                .steps
                .iter()
                .filter(|step| step.executor == DecoderRuntimeExecutor::RemoteStage)
                .count(),
            10
        );
    }
}

#[test]
fn schedules_multiple_dense_decoder_adapters_through_one_contract() {
    let qwen2 = lower_schedule(&plan(QWEN2)).unwrap();
    let qwen3 = lower_schedule(&plan(QWEN3)).unwrap();

    assert_eq!(qwen2.schema_version, qwen3.schema_version);
    assert_eq!(qwen2.composition_digest, qwen3.composition_digest);
    for schedule in [&qwen2, &qwen3] {
        assert!(schedule.complete);
        assert_eq!(schedule.prefill.steps.len(), schedule.decode.steps.len());
        assert!(schedule.prefill.steps.iter().all(|step| {
            step.executor == DecoderRuntimeExecutor::ClientLocal || !step.weight_ids.is_empty()
        }));
    }
}

#[test]
fn bounded_gemma_text_decoder_uses_the_same_schedule_contract() {
    let plan = plan(GEMMA4_E2B);
    let schedule = lower_schedule(&plan).unwrap();
    assert!(schedule.complete);
    let first = schedule
        .prefill
        .steps
        .iter()
        .find(|step| step.executor == DecoderRuntimeExecutor::RemoteStage)
        .unwrap();
    assert_eq!(
        first.operation_ids,
        ["main_embedding", "ple_token_embedding"]
    );
    assert_eq!(first.outputs[0].stage_offset, 0);
    assert_eq!(first.outputs[1].stage_offset, first.outputs[0].stage_width);
    assert_eq!(
        schedule
            .prefill
            .steps
            .iter()
            .filter(|step| step.executor == DecoderRuntimeExecutor::RemoteStage)
            .count(),
        schedule
            .decode
            .steps
            .iter()
            .filter(|step| step.executor == DecoderRuntimeExecutor::RemoteStage)
            .count()
    );
    assert!(schedule.prefill.state_inputs.is_empty());
    assert!(!schedule.decode.state_inputs.is_empty());
}

#[test]
fn verified_composition_requires_verifier_bound_execution_evidence() {
    let error = lower_decoder_runtime_schedule(&plan(QWEN2), &composition(true)).unwrap_err();
    assert!(error.contains("verifier-bound"));
}

#[test]
fn rejects_unimplemented_local_operator_capabilities() {
    let mut plan = plan(QWEN2);
    plan.prefill.operations[1].operator = ModelOperator::Scale;
    assert!(lower_schedule(&plan).is_err());
}

#[test]
fn fuses_remote_stages_with_deterministic_slices() {
    let schedule = lower_schedule(&plan(QWEN2)).unwrap();
    for phase in [&schedule.prefill, &schedule.decode] {
        let layer_zero: Vec<_> = phase
            .steps
            .iter()
            .filter(|step| {
                step.executor == DecoderRuntimeExecutor::RemoteStage && step.layer == Some(0)
            })
            .collect();
        assert_eq!(layer_zero.len(), 4);
        let qkv = layer_zero[0];
        assert_eq!(
            qkv.operators,
            vec![
                ModelOperator::Linear,
                ModelOperator::Linear,
                ModelOperator::Linear
            ]
        );
        assert_eq!(qkv.input_ids.len(), 1);
        assert_eq!(qkv.weight_ids.len(), 3);
        assert_eq!(
            qkv.outputs
                .iter()
                .map(|output| (output.stage_offset, output.stage_width))
                .collect::<Vec<_>>(),
            vec![(0, 8), (8, 4), (12, 4)]
        );
        let gate_up = layer_zero[2];
        assert_eq!(
            gate_up
                .outputs
                .iter()
                .map(|output| (output.stage_offset, output.stage_width))
                .collect::<Vec<_>>(),
            vec![(0, 16), (16, 16)]
        );
    }
    assert_eq!(
        lower_schedule(&plan(QWEN2)).unwrap().digest(),
        schedule.digest()
    );
}

#[test]
fn includes_final_norm_head_and_decoder_tail() {
    let schedule = lower_schedule(&plan(QWEN2)).unwrap();
    for phase in [&schedule.prefill, &schedule.decode] {
        let final_norm = phase
            .steps
            .iter()
            .find(|step| step.operation_ids == ["final_norm"])
            .unwrap();
        assert_eq!(final_norm.executor, DecoderRuntimeExecutor::ClientLocal);
        assert_eq!(final_norm.operators, [ModelOperator::RmsNorm]);
        let output_head = phase
            .steps
            .iter()
            .find(|step| step.operators == [ModelOperator::OutputHead])
            .unwrap();
        assert_eq!(output_head.input_ids, ["last_hidden"]);
        let tail: Vec<_> = phase.steps[phase.steps.len() - 4..]
            .iter()
            .flat_map(|step| step.operators.iter().copied())
            .collect();
        assert_eq!(
            tail,
            vec![
                ModelOperator::LastToken,
                ModelOperator::OutputHead,
                ModelOperator::GreedyTokenSelection,
                ModelOperator::TokenFeedback
            ]
        );
    }
}

#[test]
fn exact_composition_enables_complete_coverage_without_promoting_arbitrary_compositions() {
    let qwen = plan(QWEN2);
    let baseline_composition = composition(false);
    let baseline = pllm_compiler::decoder_coverage(&qwen, Some(&baseline_composition)).unwrap();
    assert!(baseline.complete);
    assert_eq!(
        baseline.composition_digest,
        Some(pipeline_digest_bytes(&baseline_composition))
    );
    assert!(baseline.operators.iter().all(|row| {
        row.level == pllm_compiler::CapabilityLevel::ExecutableRegion && row.component.is_some()
    }));
    assert!(baseline.operators.iter().any(|row| {
        row.operator == ModelOperator::Softmax
            && row
                .blocker
                .contains("client-local execution is outside provider protection")
    }));

    let arbitrary = canonical_bytes(&json!({
        "components": {},
        "model": {"source": "model.fixture"}
    }));
    let primitive = pllm_compiler::decoder_coverage(&qwen, Some(&arbitrary)).unwrap();
    assert!(!primitive.complete);
    assert!(primitive.operators.iter().all(|row| row
        .blocker
        .contains("whole-decoder scheduling is unavailable")));
    assert!(lower_decoder_runtime_schedule(&qwen, &arbitrary)
        .unwrap_err()
        .contains("admitted linear component composition"));
    assert!(pllm_compiler::decoder_coverage(&qwen, None)
        .unwrap()
        .operators
        .iter()
        .all(|row| row.level != pllm_compiler::CapabilityLevel::Missing));
    assert!(
        pllm_compiler::decoder_coverage(&plan(QWEN3), Some(&baseline_composition))
            .unwrap()
            .complete
    );

    let verified = pllm_compiler::decoder_coverage(&qwen, Some(&composition(true))).unwrap();
    assert!(!verified.complete);
    assert!(verified
        .operators
        .iter()
        .all(|row| row.blocker.contains("verifier-bound execution evidence")));

    let noncanonical = serde_json::to_vec_pretty(
        &serde_json::from_slice::<serde_json::Value>(&baseline_composition).unwrap(),
    )
    .unwrap();
    assert!(pllm_compiler::decoder_coverage(&qwen, Some(&noncanonical)).is_err());
    assert!(lower_decoder_runtime_schedule(&qwen, &noncanonical).is_err());
}

#[test]
fn accepts_supported_semantics_and_rejects_transformations_and_batches() {
    assert!(lower_schedule(&plan(QWEN3)).is_ok());
    let batched = lower_model_json(
        QWEN2,
        DecoderWorkload {
            batch: 2,
            max_input_tokens: 3,
            max_new_tokens: 2,
        },
    )
    .unwrap();
    assert!(lower_schedule(&batched).is_err());
    assert!(
        !pllm_compiler::decoder_coverage(&batched, Some(&composition(false)))
            .unwrap()
            .complete
    );

    let mut transformed = plan(QWEN2);
    transformed
        .transformations
        .push(pllm_models::AppliedMethod {
            component: "pllm/test".into(),
            implementation: "test".into(),
            method_id: "test".into(),
            input_digest: transformed.digest(),
            configuration_digest: transformed.config_digest.clone(),
        });
    assert!(lower_schedule(&transformed).is_err());

    let mut reordered = plan(QWEN2);
    let query = reordered
        .prefill
        .operations
        .iter()
        .position(|operation| operation.id == "layer.0.q_linear")
        .unwrap();
    let key = reordered
        .prefill
        .operations
        .iter()
        .position(|operation| operation.id == "layer.0.k_linear")
        .unwrap();
    reordered.prefill.operations.swap(query, key);
    let query = reordered
        .decode
        .operations
        .iter()
        .position(|operation| operation.id == "layer.0.q_linear")
        .unwrap();
    let key = reordered
        .decode
        .operations
        .iter()
        .position(|operation| operation.id == "layer.0.k_linear")
        .unwrap();
    reordered.decode.operations.swap(query, key);
    assert!(reordered.validate().is_ok());
    assert!(lower_schedule(&reordered).is_ok());
}
