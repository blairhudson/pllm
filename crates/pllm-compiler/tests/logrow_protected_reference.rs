use pllm_compiler::{
    estimate_bound_logrow_profile_session, estimate_bound_logrow_q7_session,
    prepare_bound_logrow_profile_tensor, prepare_bound_logrow_q7_element,
    prepare_bound_logrow_q7_session, prepare_bound_logrow_q7_tensor,
    prepare_bound_logrow_scaled_session, ExperimentalLogRowQ7Policy,
    ExperimentalLogRowQ7TensorPolicy, LogRowSiluProfile,
};
use pllm_core::fit_compact_silu_q7;
use pllm_core::logrow_numeric::ScaledSiluQ7Profile;
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
    plan_with_decode(config, tokens, 1)
}

fn plan_with_decode(config: &[u8], tokens: u64, max_new_tokens: u64) -> DecoderPlan {
    lower_model_json(
        config,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: tokens,
            max_new_tokens,
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

#[test]
fn entire_response_material_is_admitted_before_any_tensor_issuance() {
    let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    let mut skewed = [0; 257];
    skewed[128] = 5;
    let other_profile = fit_compact_silu_q7(&skewed, 4).unwrap();
    let policy =
        ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(32, 68_608)
            .unwrap();
    for model in [QWEN2, QWEN3] {
        let plan = plan_with_decode(model, 2, 2);
        let estimate =
            estimate_bound_logrow_q7_session(&plan, &profile, &policy, 2, 137_216).unwrap();
        assert_eq!(estimate.prefill_elements, 32);
        assert_eq!(estimate.decode_elements_per_step, 16);
        assert_eq!(estimate.reserved_evaluator_material_bytes, 137_216);
        assert_eq!(estimate.largest_tensor_material_bytes, 68_608);
        assert_eq!(estimate.plan_digest, plan.digest());
        assert_eq!(estimate.profile_digest, profile.digest());
        assert_eq!(estimate.max_decode_steps, 2);
        assert_ne!(
            estimate.estimate_digest,
            estimate_bound_logrow_q7_session(&plan, &profile, &policy, 1, 137_216)
                .unwrap()
                .estimate_digest
        );
        assert_ne!(
            estimate.estimate_digest,
            estimate_bound_logrow_q7_session(&plan, &other_profile, &policy, 2, 137_216)
                .unwrap()
                .estimate_digest
        );
        for (steps, bytes) in [
            (0, 137_216),
            (3, 137_216),
            (2, 137_215),
            (2, 0),
            (2, usize::MAX),
        ] {
            assert!(
                estimate_bound_logrow_q7_session(&plan, &profile, &policy, steps, bytes).is_err(),
                "invalid session bound ({steps}, {bytes}) passed"
            );
        }
        let tighter =
            ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(16, 68_608)
                .unwrap();
        assert!(estimate_bound_logrow_q7_session(&plan, &profile, &tighter, 2, 137_216).is_err());
    }
}

#[test]
fn offline_session_issues_all_material_once_and_consumes_semantic_order() {
    let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    let policy =
        ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(32, 68_608)
            .unwrap();
    for model in [QWEN2, QWEN3] {
        let plan = plan_with_decode(model, 2, 2);
        let operation = silu(&plan);
        let mut session =
            prepare_bound_logrow_q7_session(&plan, &profile, &policy, 2, 137_216).unwrap();
        assert_eq!(
            session.estimate().reserved_evaluator_material_bytes,
            137_216
        );
        assert_eq!(session.remaining_tensors(), 3);
        let other = prepare_bound_logrow_q7_session(&plan, &profile, &policy, 2, 137_216).unwrap();
        assert_ne!(session.issuance_digest(), other.issuance_digest());
        let inputs = [-1.0, -0.5, 0.0, 0.5, 1.0];
        for (mode, step, width) in [
            (DecoderMode::Prefill, 0, 32),
            (DecoderMode::Decode, 0, 16),
            (DecoderMode::Decode, 1, 16),
        ] {
            let values = (0..width)
                .map(|index| inputs[index % inputs.len()])
                .collect::<Vec<_>>();
            let output = session
                .evaluate_float32(mode, step, operation, &values)
                .unwrap();
            assert_eq!(output.len(), width);
            for (value, result) in values.iter().zip(output) {
                assert_eq!(
                    result,
                    f32::from(profile.evaluate((value * 128.0) as i16).unwrap()) / 128.0
                );
            }
        }
        assert_eq!(session.remaining_tensors(), 0);
        assert!(session
            .evaluate_float32(DecoderMode::Decode, 1, operation, &[0.0; 16])
            .is_err());
    }
}

#[test]
fn offline_session_burns_every_remaining_tensor_on_mismatch_failure_or_abort() {
    let plan = plan_with_decode(QWEN2, 2, 2);
    let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    let policy =
        ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(32, 68_608)
            .unwrap();
    let operation = silu(&plan);
    let mut session =
        prepare_bound_logrow_q7_session(&plan, &profile, &policy, 2, 137_216).unwrap();
    assert!(session
        .evaluate_float32(DecoderMode::Decode, 0, operation, &[0.0; 16])
        .is_err());
    assert_eq!(session.remaining_tensors(), 0);
    assert!(session
        .evaluate_float32(DecoderMode::Prefill, 0, operation, &[0.0; 32])
        .is_err());
    let mut session =
        prepare_bound_logrow_q7_session(&plan, &profile, &policy, 2, 137_216).unwrap();
    let mut invalid = vec![0.0; 32];
    invalid[31] = 1.01;
    assert!(session
        .evaluate_float32(DecoderMode::Prefill, 0, operation, &invalid)
        .is_err());
    assert_eq!(session.remaining_tensors(), 0);
    let mut session =
        prepare_bound_logrow_q7_session(&plan, &profile, &policy, 2, 137_216).unwrap();
    assert!(session
        .evaluate_float32(DecoderMode::Prefill, 0, operation, &[0.0; 31])
        .is_err());
    assert_eq!(session.remaining_tensors(), 0);
    let mut session =
        prepare_bound_logrow_q7_session(&plan, &profile, &policy, 2, 137_216).unwrap();
    session
        .evaluate_float32(DecoderMode::Prefill, 0, operation, &[0.0; 32])
        .unwrap();
    session.abort();
    assert_eq!(session.remaining_tensors(), 0);
    assert!(session
        .evaluate_float32(DecoderMode::Decode, 0, operation, &[0.0; 16])
        .is_err());
}

#[test]
fn scaled_public_range_reuses_one_use_session_without_compact_profile_confusion() {
    let compact = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    let policy =
        ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(32, 68_608)
            .unwrap();
    for model in [QWEN2, QWEN3] {
        let plan = plan_with_decode(model, 2, 2);
        let profile = ScaledSiluQ7Profile::new(4).unwrap();
        let other = ScaledSiluQ7Profile::new(8).unwrap();
        let budget = 137_216;
        let estimate = estimate_bound_logrow_profile_session(
            &plan,
            &LogRowSiluProfile::Scaled(profile.clone()),
            &policy,
            2,
            budget,
        )
        .unwrap();
        let changed = estimate_bound_logrow_profile_session(
            &plan,
            &LogRowSiluProfile::Scaled(other),
            &policy,
            2,
            budget,
        )
        .unwrap();
        let compact_estimate =
            estimate_bound_logrow_q7_session(&plan, &compact, &policy, 2, budget).unwrap();
        assert_ne!(estimate.estimate_digest, changed.estimate_digest);
        assert_ne!(estimate.estimate_digest, compact_estimate.estimate_digest);
        let prefill = silu(&plan);
        let tensor = prepare_bound_logrow_profile_tensor(
            &plan,
            DecoderMode::Prefill,
            prefill,
            &LogRowSiluProfile::Scaled(profile.clone()),
            &policy,
        )
        .unwrap();
        let (evaluation, _decoder) = tensor.encode(&[0; 32]).unwrap();
        assert!(evaluation
            .evaluate(&plan, DecoderMode::Prefill, prefill, &compact, &policy)
            .is_err());

        let mut session =
            prepare_bound_logrow_scaled_session(&plan, &profile, &policy, 2, budget).unwrap();
        assert_eq!(session.estimate().estimate_digest, estimate.estimate_digest);
        for (mode, step, count) in [
            (DecoderMode::Prefill, 0, 32),
            (DecoderMode::Decode, 0, 16),
            (DecoderMode::Decode, 1, 16),
        ] {
            let values = (0..count)
                .map(|index| (index as f32 / count as f32 * 6.0) - 3.0)
                .collect::<Vec<_>>();
            let output = session
                .evaluate_float32(mode, step, prefill, &values)
                .unwrap();
            for (actual, value) in output.into_iter().zip(values) {
                let exact = f64::from(value) / (1.0 + (-f64::from(value)).exp());
                assert!(
                    (f64::from(actual) - exact).abs() <= profile.maximum_absolute_error_bound()
                );
            }
        }
        assert_eq!(session.remaining_tensors(), 0);
        let mut invalid =
            prepare_bound_logrow_scaled_session(&plan, &profile, &policy, 2, budget).unwrap();
        assert!(invalid
            .evaluate_float32(DecoderMode::Prefill, 0, prefill, &[4.1; 32])
            .is_err());
        assert_eq!(invalid.remaining_tensors(), 0);
    }
}
