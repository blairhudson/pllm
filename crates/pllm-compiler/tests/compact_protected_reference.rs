use pllm_compiler::{prepare_bound_compact_q7_element, ExperimentalCompactQ7Policy};
use pllm_core::fit_compact_silu_q7;
use pllm_models::{lower_model_json, DecoderMode, DecoderPlan, DecoderWorkload, ModelOperator};

const QWEN2: &[u8] = br#"{
    "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
    "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
    "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
    "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
}"#;

const QWEN3: &[u8] = br#"{
    "model_type":"qwen3","hidden_size":8,"intermediate_size":16,
    "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
    "vocab_size":32,"max_position_embeddings":32,"head_dim":4,
    "hidden_act":"silu","rms_norm_eps":1e-6,"rope_theta":10000.0,
    "tie_word_embeddings":true,"attention_bias":false,"attention_dropout":0.0,
    "rope_scaling":null,"sliding_window":null,"use_sliding_window":false,
    "use_cache":true,"max_window_layers":1,"layer_types":["full_attention"]
}"#;

fn plan(tokens: u64) -> DecoderPlan {
    plan_with_config(QWEN2, tokens)
}

fn plan_with_config(config: &[u8], tokens: u64) -> DecoderPlan {
    lower_model_json(
        config,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: tokens,
            max_new_tokens: 1,
        },
    )
    .unwrap()
}

#[test]
fn semantic_binding_is_model_neutral_for_dense_qwen3_silu() {
    let plan = plan_with_config(QWEN3, 2);
    let profile = fit_compact_silu_q7(&[0; 257], 2).unwrap();
    let operation_id = silu(&plan);
    let material = prepare_bound_compact_q7_element(
        &plan,
        DecoderMode::Prefill,
        operation_id,
        4,
        &profile,
        &policy(),
    )
    .unwrap();
    let (evaluation, decoder) = material.encode(64).unwrap();
    let outputs = evaluation
        .evaluate(
            &plan,
            DecoderMode::Prefill,
            operation_id,
            4,
            &profile,
            &policy(),
        )
        .unwrap();
    assert_eq!(
        decoder.decode(outputs).unwrap().1,
        profile.evaluate(64).unwrap()
    );
}

fn silu(plan: &DecoderPlan) -> &str {
    &plan
        .prefill
        .operations
        .iter()
        .find(|operation| operation.operator == ModelOperator::Silu)
        .unwrap()
        .id
}

fn policy() -> ExperimentalCompactQ7Policy {
    ExperimentalCompactQ7Policy::acknowledge_unreviewed_public_profile(8 * 1024 * 1024).unwrap()
}

#[test]
fn semantic_silu_element_is_bound_to_plan_profile_mode_and_one_use_material() {
    let plan = plan(2);
    let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    let operation_id = silu(&plan);
    let material = prepare_bound_compact_q7_element(
        &plan,
        DecoderMode::Prefill,
        operation_id,
        0,
        &profile,
        &policy(),
    )
    .unwrap();
    assert!(material.evaluator_ciphertext_bytes().unwrap() > 0);
    let bound_digest = material.binding_digest().clone();
    let (evaluation, decoder) = material.encode(-87).unwrap();
    assert_eq!(evaluation.binding_digest(), &bound_digest);
    let output = evaluation
        .evaluate(
            &plan,
            DecoderMode::Prefill,
            operation_id,
            0,
            &profile,
            &policy(),
        )
        .unwrap();
    let (piece, value) = decoder.decode(output).unwrap();
    assert!((profile.pieces()[piece].lower()..=profile.pieces()[piece].upper()).contains(&-87));
    assert_eq!(value, profile.evaluate(-87).unwrap());

    let next = prepare_bound_compact_q7_element(
        &plan,
        DecoderMode::Prefill,
        operation_id,
        0,
        &profile,
        &policy(),
    )
    .unwrap();
    assert_ne!(&bound_digest, next.binding_digest());
}

#[test]
fn semantic_context_and_resource_policy_reject_mismatches_before_evaluation() {
    let altered_plan = plan(3);
    let qwen3_plan = plan_with_config(QWEN3, 2);
    let plan = plan(2);
    let profile = fit_compact_silu_q7(&[0; 257], 2).unwrap();
    let mut counts = [0; 257];
    counts[128] = 1;
    let other_profile = fit_compact_silu_q7(&counts, 2).unwrap();
    let operation_id = silu(&plan);
    let small = ExperimentalCompactQ7Policy::acknowledge_unreviewed_public_profile(1).unwrap();
    assert!(prepare_bound_compact_q7_element(
        &plan,
        DecoderMode::Prefill,
        operation_id,
        0,
        &profile,
        &small,
    )
    .is_err());
    assert!(prepare_bound_compact_q7_element(
        &plan,
        DecoderMode::Prefill,
        operation_id,
        usize::MAX,
        &profile,
        &policy(),
    )
    .is_err());
    let wrong_operator = &plan.prefill.operations[0].id;
    assert!(prepare_bound_compact_q7_element(
        &plan,
        DecoderMode::Prefill,
        wrong_operator,
        0,
        &profile,
        &policy(),
    )
    .is_err());
    for mismatch in 0..6 {
        let material = prepare_bound_compact_q7_element(
            &plan,
            DecoderMode::Prefill,
            operation_id,
            0,
            &profile,
            &policy(),
        )
        .unwrap();
        let (evaluation, _) = material.encode(12).unwrap();
        let result = match mismatch {
            0 => evaluation.evaluate(
                &altered_plan,
                DecoderMode::Prefill,
                operation_id,
                0,
                &profile,
                &policy(),
            ),
            1 => evaluation.evaluate(
                &plan,
                DecoderMode::Decode,
                operation_id,
                0,
                &profile,
                &policy(),
            ),
            2 => evaluation.evaluate(
                &plan,
                DecoderMode::Prefill,
                operation_id,
                1,
                &profile,
                &policy(),
            ),
            3 => evaluation.evaluate(
                &plan,
                DecoderMode::Prefill,
                operation_id,
                0,
                &other_profile,
                &policy(),
            ),
            4 => evaluation.evaluate(
                &qwen3_plan,
                DecoderMode::Prefill,
                operation_id,
                0,
                &profile,
                &policy(),
            ),
            _ => evaluation.evaluate(
                &plan,
                DecoderMode::Prefill,
                operation_id,
                0,
                &profile,
                &small,
            ),
        };
        assert!(result.is_err(), "context mismatch {mismatch} was accepted");
    }
    let material = prepare_bound_compact_q7_element(
        &plan,
        DecoderMode::Prefill,
        operation_id,
        0,
        &profile,
        &policy(),
    )
    .unwrap();
    assert!(material.encode(-129).is_err());
    assert!(ExperimentalCompactQ7Policy::acknowledge_unreviewed_public_profile(0).is_err());
    assert!(
        ExperimentalCompactQ7Policy::acknowledge_unreviewed_public_profile(8 * 1024 * 1024 + 1)
            .is_err()
    );
}
