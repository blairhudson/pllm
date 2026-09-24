use pllm_compiler::{
    prepare_bound_logrow_q7_element, prepare_bound_logrow_q7_tensor, ExperimentalLogRowQ7Policy,
    ExperimentalLogRowQ7TensorPolicy,
};
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

fn plan(config: &[u8], tokens: u64) -> DecoderPlan {
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

fn silu(plan: &DecoderPlan) -> &str {
    &plan
        .prefill
        .operations
        .iter()
        .find(|op| op.operator == ModelOperator::Silu)
        .unwrap()
        .id
}

fn policy() -> ExperimentalLogRowQ7Policy {
    ExperimentalLogRowQ7Policy::acknowledge_unreviewed_public_profile(2_144).unwrap()
}

#[test]
fn each_dense_qwen_family_binds_one_signed_q7_element_to_the_same_semantic_operator() {
    let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    for model in [QWEN2, QWEN3] {
        let plan = plan(model, 2);
        let op = silu(&plan);
        for value in [-128, -87, -1, 0, 64, 127, 128] {
            let material = prepare_bound_logrow_q7_element(
                &plan,
                DecoderMode::Prefill,
                op,
                3,
                &profile,
                &policy(),
            )
            .unwrap();
            assert_eq!(material.evaluator_material_bytes(), 2_144);
            let binding = material.binding_digest().clone();
            let (evaluation, decoder) = material.encode(value).unwrap();
            assert_eq!(evaluation.binding_digest(), &binding);
            let output = evaluation
                .evaluate(&plan, DecoderMode::Prefill, op, 3, &profile, &policy())
                .unwrap();
            assert_eq!(decoder.decode(output), Ok(profile.evaluate(value).unwrap()));
        }
    }
}

#[test]
fn policy_semantic_bounds_and_fresh_issuance_fail_closed() {
    let plan = plan(QWEN2, 2);
    let profile = fit_compact_silu_q7(&[0; 257], 2).unwrap();
    let op = silu(&plan);
    assert!(ExperimentalLogRowQ7Policy::acknowledge_unreviewed_public_profile(0).is_err());
    assert!(ExperimentalLogRowQ7Policy::acknowledge_unreviewed_public_profile(65_537).is_err());
    let small = ExperimentalLogRowQ7Policy::acknowledge_unreviewed_public_profile(2_143).unwrap();
    assert!(
        prepare_bound_logrow_q7_element(&plan, DecoderMode::Prefill, op, 0, &profile, &small)
            .is_err()
    );
    assert!(prepare_bound_logrow_q7_element(
        &plan,
        DecoderMode::Prefill,
        op,
        usize::MAX,
        &profile,
        &policy()
    )
    .is_err());
    assert!(prepare_bound_logrow_q7_element(
        &plan,
        DecoderMode::Prefill,
        &plan.prefill.operations[0].id,
        0,
        &profile,
        &policy()
    )
    .is_err());
    let material =
        prepare_bound_logrow_q7_element(&plan, DecoderMode::Prefill, op, 0, &profile, &policy())
            .unwrap();
    let binding = material.binding_digest().clone();
    assert!(material.encode(129).is_err());
    let next =
        prepare_bound_logrow_q7_element(&plan, DecoderMode::Prefill, op, 0, &profile, &policy())
            .unwrap();
    assert_ne!(next.binding_digest(), &binding);
}

#[test]
fn revalidation_burns_plan_mode_profile_operator_index_and_policy_mismatches() {
    let changed = plan(QWEN2, 3);
    let other = plan(QWEN3, 2);
    let plan = plan(QWEN2, 2);
    let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    let mut counts = [0; 257];
    counts[128] = 5;
    let other_profile = fit_compact_silu_q7(&counts, 4).unwrap();
    let op = silu(&plan);
    let wrong_op = &plan.prefill.operations[0].id;
    let small = ExperimentalLogRowQ7Policy::acknowledge_unreviewed_public_profile(2_143).unwrap();
    for mismatch in 0..7 {
        let material = prepare_bound_logrow_q7_element(
            &plan,
            DecoderMode::Prefill,
            op,
            0,
            &profile,
            &policy(),
        )
        .unwrap();
        let (evaluation, _) = material.encode(-64).unwrap();
        let result = match mismatch {
            0 => evaluation.evaluate(&changed, DecoderMode::Prefill, op, 0, &profile, &policy()),
            1 => evaluation.evaluate(&plan, DecoderMode::Decode, op, 0, &profile, &policy()),
            2 => evaluation.evaluate(
                &plan,
                DecoderMode::Prefill,
                wrong_op,
                0,
                &profile,
                &policy(),
            ),
            3 => evaluation.evaluate(&plan, DecoderMode::Prefill, op, 1, &profile, &policy()),
            4 => evaluation.evaluate(
                &plan,
                DecoderMode::Prefill,
                op,
                0,
                &other_profile,
                &policy(),
            ),
            5 => evaluation.evaluate(&plan, DecoderMode::Prefill, op, 0, &profile, &small),
            _ => evaluation.evaluate(&other, DecoderMode::Prefill, op, 0, &profile, &policy()),
        };
        assert!(result.is_err(), "context mismatch {mismatch} escaped");
    }
}

#[test]
fn complete_q7_silu_tensors_bind_across_dense_model_sources_and_both_phases() {
    let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    for model in [QWEN2, QWEN3] {
        let plan = plan(model, 2);
        let operation = silu(&plan);
        for (mode, count) in [(DecoderMode::Prefill, 32), (DecoderMode::Decode, 16)] {
            let limit = count * 2_144;
            let policy = ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(
                count, limit,
            )
            .unwrap();
            let material =
                prepare_bound_logrow_q7_tensor(&plan, mode, operation, &profile, &policy).unwrap();
            assert_eq!(material.elements(), count);
            assert_eq!(material.evaluator_material_bytes(), limit);
            let binding = material.binding_digest().clone();
            let values = (0..count)
                .map(|index| [-128, -71, -1, 0, 64, 128][index % 6])
                .collect::<Vec<_>>();
            let (evaluation, decoder) = material.encode(&values).unwrap();
            assert_eq!(evaluation.binding_digest(), &binding);
            let outputs = evaluation
                .evaluate(&plan, mode, operation, &profile, &policy)
                .unwrap();
            assert_eq!(
                decoder.decode(outputs).unwrap(),
                values
                    .iter()
                    .map(|&value| profile.evaluate(value).unwrap())
                    .collect::<Vec<_>>()
            );
            let different =
                prepare_bound_logrow_q7_tensor(&plan, mode, operation, &profile, &policy).unwrap();
            assert_ne!(different.binding_digest(), &binding);
        }
    }
}

#[test]
fn tensor_admission_and_invalid_inputs_fail_before_any_partial_evaluation() {
    let plan = plan(QWEN2, 2);
    let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    let operation = silu(&plan);
    assert!(ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(0, 1).is_err());
    assert!(ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(1, 0).is_err());
    assert!(
        ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(
            1,
            64 * 1024 * 1024 + 1
        )
        .is_err()
    );
    assert!(pllm_garble::logrow::prepare_compact_q7_logrow_elements(&profile, 0).is_err());
    assert!(pllm_garble::logrow::prepare_compact_q7_logrow_elements(&profile, usize::MAX).is_err());
    for (elements, bytes) in [(31, 68_608), (32, 68_607)] {
        let policy = ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(
            elements, bytes,
        )
        .unwrap();
        assert!(prepare_bound_logrow_q7_tensor(
            &plan,
            DecoderMode::Prefill,
            operation,
            &profile,
            &policy
        )
        .is_err());
    }
    let policy =
        ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(32, 68_608)
            .unwrap();
    let material =
        prepare_bound_logrow_q7_tensor(&plan, DecoderMode::Prefill, operation, &profile, &policy)
            .unwrap();
    assert!(material.encode(&[0; 31]).is_err());
    let material =
        prepare_bound_logrow_q7_tensor(&plan, DecoderMode::Prefill, operation, &profile, &policy)
            .unwrap();
    let mut out_of_range = vec![0; 32];
    out_of_range[31] = 129;
    assert!(material.encode(&out_of_range).is_err());
    let material =
        prepare_bound_logrow_q7_tensor(&plan, DecoderMode::Prefill, operation, &profile, &policy)
            .unwrap();
    let (evaluation, decoder) = material.encode(&[0; 32]).unwrap();
    assert!(evaluation
        .evaluate(&plan, DecoderMode::Decode, operation, &profile, &policy)
        .is_err());
    drop(decoder);
}

#[test]
fn tensor_material_never_crosses_plan_profile_operator_or_decoder_contexts() {
    let other = plan(QWEN3, 2);
    let changed = plan(QWEN2, 3);
    let plan = plan(QWEN2, 2);
    let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    let mut counts = [0; 257];
    counts[128] = 5;
    let other_profile = fit_compact_silu_q7(&counts, 4).unwrap();
    let operation = silu(&plan);
    let wrong = &plan.prefill.operations[0].id;
    let policy =
        ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(32, 68_608)
            .unwrap();
    let altered_policy =
        ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(33, 68_608)
            .unwrap();
    for mismatch in 0..5 {
        let material = prepare_bound_logrow_q7_tensor(
            &plan,
            DecoderMode::Prefill,
            operation,
            &profile,
            &policy,
        )
        .unwrap();
        let (evaluation, _) = material.encode(&[0; 32]).unwrap();
        let result = match mismatch {
            0 => evaluation.evaluate(&other, DecoderMode::Prefill, operation, &profile, &policy),
            1 => evaluation.evaluate(&changed, DecoderMode::Prefill, operation, &profile, &policy),
            2 => evaluation.evaluate(&plan, DecoderMode::Prefill, wrong, &profile, &policy),
            3 => evaluation.evaluate(
                &plan,
                DecoderMode::Prefill,
                operation,
                &other_profile,
                &policy,
            ),
            _ => evaluation.evaluate(
                &plan,
                DecoderMode::Prefill,
                operation,
                &profile,
                &altered_policy,
            ),
        };
        assert!(result.is_err(), "context mismatch {mismatch} escaped");
    }
    let first =
        prepare_bound_logrow_q7_tensor(&plan, DecoderMode::Decode, operation, &profile, &policy)
            .unwrap();
    let second =
        prepare_bound_logrow_q7_tensor(&plan, DecoderMode::Decode, operation, &profile, &policy)
            .unwrap();
    let (first_eval, first_decoder) = first.encode(&[0; 16]).unwrap();
    let (_, other_decoder) = second.encode(&[0; 16]).unwrap();
    let first_output = first_eval
        .evaluate(&plan, DecoderMode::Decode, operation, &profile, &policy)
        .unwrap();
    assert!(other_decoder.decode(first_output).is_err());
    drop(first_decoder);
}
