use pllm_compiler::{
    lower_dense_qwen_runtime_schedule, DenseQwenRuntimeExecutor, DENSE_QWEN_MASKED_RUNTIME_PROFILE,
    DENSE_QWEN_RUNTIME_SCHEDULE_SCHEMA_VERSION,
};
use pllm_models::{lower_model_json, DecoderPlan, DecoderWorkload, ModelOperator};
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

#[test]
fn lowers_complete_prefill_and_decode_schedules() {
    let plan = plan(QWEN2);
    let schedule = lower_dense_qwen_runtime_schedule(&plan).unwrap();
    assert_eq!(
        schedule.schema_version,
        DENSE_QWEN_RUNTIME_SCHEDULE_SCHEMA_VERSION
    );
    assert_eq!(schedule.profile, DENSE_QWEN_MASKED_RUNTIME_PROFILE);
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
                .filter(|step| step.executor == DenseQwenRuntimeExecutor::RemoteStage)
                .count(),
            10
        );
    }
}

#[test]
fn fuses_remote_stages_with_deterministic_slices() {
    let schedule = lower_dense_qwen_runtime_schedule(&plan(QWEN2)).unwrap();
    for phase in [&schedule.prefill, &schedule.decode] {
        let layer_zero: Vec<_> = phase
            .steps
            .iter()
            .filter(|step| {
                step.executor == DenseQwenRuntimeExecutor::RemoteStage && step.layer == Some(0)
            })
            .collect();
        assert_eq!(
            layer_zero
                .iter()
                .map(|step| step.stage_role.as_deref().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "qkv_projection",
                "attention_output",
                "mlp_gate_up",
                "mlp_down"
            ]
        );
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
        lower_dense_qwen_runtime_schedule(&plan(QWEN2))
            .unwrap()
            .digest(),
        schedule.digest()
    );
}

#[test]
fn includes_final_norm_head_and_decoder_tail() {
    let schedule = lower_dense_qwen_runtime_schedule(&plan(QWEN2)).unwrap();
    for phase in [&schedule.prefill, &schedule.decode] {
        let final_norm = phase
            .steps
            .iter()
            .find(|step| step.operation_ids == ["final_norm"])
            .unwrap();
        assert_eq!(final_norm.executor, DenseQwenRuntimeExecutor::ClientLocal);
        assert_eq!(final_norm.operators, [ModelOperator::RmsNorm]);
        let output_head = phase
            .steps
            .iter()
            .find(|step| step.stage_role.as_deref() == Some("lm_head"))
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
fn baseline_coverage_is_complete_without_promoting_research_profile() {
    let qwen = plan(QWEN2);
    let baseline = pllm_compiler::decoder_coverage(&qwen, DENSE_QWEN_MASKED_RUNTIME_PROFILE);
    assert!(baseline.complete);
    assert!(baseline.operators.iter().all(|row| {
        row.level == pllm_compiler::CapabilityLevel::ExecutableRegion && row.component.is_some()
    }));
    assert!(baseline.operators.iter().any(|row| {
        row.operator == ModelOperator::Softmax
            && row
                .blocker
                .contains("client-local execution is outside provider protection")
    }));

    let research = pllm_compiler::decoder_coverage(&qwen, "research.single_evaluator");
    assert!(!research.complete);
    assert!(research.operators.iter().all(|row| row
        .blocker
        .contains("whole-decoder scheduling is unavailable")));
    assert!(
        !pllm_compiler::decoder_coverage(&plan(QWEN3), DENSE_QWEN_MASKED_RUNTIME_PROFILE).complete
    );
}

#[test]
fn rejects_other_families_transformations_batches_and_unfuseable_order() {
    assert!(lower_dense_qwen_runtime_schedule(&plan(QWEN3)).is_err());
    let batched = lower_model_json(
        QWEN2,
        DecoderWorkload {
            batch: 2,
            max_input_tokens: 3,
            max_new_tokens: 2,
        },
    )
    .unwrap();
    assert!(lower_dense_qwen_runtime_schedule(&batched).is_err());
    assert!(!pllm_compiler::decoder_coverage(&batched, DENSE_QWEN_MASKED_RUNTIME_PROFILE).complete);

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
    assert!(lower_dense_qwen_runtime_schedule(&transformed).is_err());

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
    assert!(reordered.validate().is_ok());
    assert!(lower_dense_qwen_runtime_schedule(&reordered).is_err());
}
