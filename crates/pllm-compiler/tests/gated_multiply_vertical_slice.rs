use pllm_compiler::{
    decoder_coverage, define_q14_to_q7_rescale_region, lower_model_gated_multiply_q7_regions,
    prepare_bound_gated_multiply_q7_material, CapabilityLevel, GatedMultiplyQ7Evaluator,
    GATED_MULTIPLY_Q7_MAX_EVALUATOR_PAYLOAD_BYTES, GATED_MULTIPLY_Q7_NUMERIC_GRAPH_ID,
    GATED_MULTIPLY_Q7_PROTECTED_GRAPH_ID,
};
use pllm_models::{lower_model_json, DecoderMode, DecoderPlan, DecoderWorkload};

fn qwen_plan(intermediate_size: u64) -> DecoderPlan {
    let config = format!(
        r#"{{
            "model_type":"qwen2",
            "hidden_size":2,
            "intermediate_size":{intermediate_size},
            "num_hidden_layers":1,
            "num_attention_heads":1,
            "num_key_value_heads":1,
            "vocab_size":8,
            "max_position_embeddings":8,
            "hidden_act":"silu",
            "rms_norm_eps":1e-6,
            "rope_theta":10000.0,
            "tie_word_embeddings":true
        }}"#
    );
    lower_model_json(
        config.as_bytes(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 1,
        },
    )
    .unwrap()
}

#[test]
fn semantic_qwen_gated_multiply_lowers_with_exact_provenance() {
    let plan = qwen_plan(1);
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        let regions = lower_model_gated_multiply_q7_regions(&plan, mode).unwrap();
        let [region] = regions.as_slice() else {
            panic!("expected one gated MLP region")
        };
        assert_eq!(region.layer, 0);
        assert_eq!(region.gate_linear_operation_id, "layer.0.gate_proj");
        assert_eq!(region.silu_operation_id, "layer.0.silu");
        assert_eq!(region.up_linear_operation_id, "layer.0.up_proj");
        assert_eq!(region.multiply_operation_id, "layer.0.gated_multiply");
        assert_eq!(region.gate_rescale.target_input_index, 0);
        assert_eq!(region.up_rescale.target_input_index, 1);
        assert_eq!(region.input.shape, vec![1, 1, 1]);
        assert_eq!(region.numeric_graph_id, GATED_MULTIPLY_Q7_NUMERIC_GRAPH_ID);
        assert_eq!(
            region.protected_graph_id,
            GATED_MULTIPLY_Q7_PROTECTED_GRAPH_ID
        );
    }
    let coverage = decoder_coverage(&plan, "research.single_evaluator");
    let multiply = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == pllm_models::ModelOperator::Multiply)
        .unwrap();
    assert_eq!(multiply.level, CapabilityLevel::Primitive);
    assert_eq!(
        multiply.component.as_deref(),
        Some("pllm/agc-gated-multiply-q7@0.1.0-alpha.1-experimental")
    );
    assert!(multiply.blocker.contains("one scalar"));
}

#[test]
fn composed_gate_executes_without_exposing_the_silu_value() {
    let plan = qwen_plan(1);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let payload = material.evaluator_payload();
    assert_eq!(payload.len(), 649_160);
    assert!(payload.len() <= GATED_MULTIPLY_Q7_MAX_EVALUATOR_PAYLOAD_BYTES);
    let mut evaluator = GatedMultiplyQ7Evaluator::new(&region, &payload).unwrap();
    let output = evaluator
        .evaluate(
            &material.encode_gate(-65).unwrap(),
            &material.encode_up(127).unwrap(),
        )
        .unwrap();
    assert_eq!(
        material.decode(&output).unwrap(),
        pllm_core::gated_multiply_q7(-65, 127).unwrap()
    );
    assert_eq!(
        evaluator
            .evaluate(
                &material.encode_gate(-65).unwrap(),
                &material.encode_up(127).unwrap(),
            )
            .unwrap_err(),
        "gated Q7 multiply evaluator material was already consumed"
    );
}

#[test]
fn forged_payload_does_not_burn_authentic_material() {
    let plan = qwen_plan(1);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let payload = material.evaluator_payload();
    let mut forged = payload.clone();
    *forged.last_mut().unwrap() ^= 1;
    assert!(GatedMultiplyQ7Evaluator::new(&region, &forged).is_err());

    let mut evaluator = GatedMultiplyQ7Evaluator::new(&region, &payload).unwrap();
    let output = evaluator
        .evaluate(
            &material.encode_gate(64).unwrap(),
            &material.encode_up(-96).unwrap(),
        )
        .unwrap();
    assert_eq!(
        material.decode(&output).unwrap(),
        pllm_core::gated_multiply_q7(64, -96).unwrap()
    );
}

#[test]
fn authentic_wrong_region_binding_burns_material() {
    let plan = qwen_plan(1);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let payload = material.evaluator_payload();
    let mut other_region = region.clone();
    other_region.mode = DecoderMode::Decode;
    assert_eq!(
        GatedMultiplyQ7Evaluator::new(&other_region, &payload)
            .err()
            .unwrap(),
        "gated Q7 multiply commitment does not match region"
    );
    assert_eq!(
        GatedMultiplyQ7Evaluator::new(&region, &payload)
            .err()
            .unwrap(),
        "gated Q7 multiply evaluator material was not issued or was already bound"
    );
}

#[test]
fn malformed_input_and_cross_lane_labels_burn_the_program() {
    let plan = qwen_plan(1);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let mut evaluator =
        GatedMultiplyQ7Evaluator::new(&region, &material.evaluator_payload()).unwrap();
    assert!(evaluator
        .evaluate(
            &material.encode_up(3).unwrap(),
            &material.encode_gate(5).unwrap(),
        )
        .is_err());
    assert_eq!(
        evaluator
            .evaluate(
                &material.encode_gate(5).unwrap(),
                &material.encode_up(3).unwrap(),
            )
            .unwrap_err(),
        "gated Q7 multiply evaluator material was already consumed"
    );
}

#[test]
fn executor_rejects_non_scalar_and_mutated_regions() {
    let plan = qwen_plan(2);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    assert_eq!(
        prepare_bound_gated_multiply_q7_material(&plan, &region)
            .err()
            .unwrap(),
        "gated Q7 multiply reference executor requires exactly one element, received 2"
    );

    let plan = qwen_plan(1);
    let mut region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    region.method.id = "forged-method".into();
    assert_eq!(
        prepare_bound_gated_multiply_q7_material(&plan, &region)
            .err()
            .unwrap(),
        "gated Q7 multiply region does not match the installed contract"
    );

    let mut region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    region.multiply_operation_id = "layer.0.attention_scores".into();
    region.up_rescale = define_q14_to_q7_rescale_region(
        &region.up_linear_operation_id,
        &region.multiply_operation_id,
        1,
        region.input.shape.clone(),
    )
    .unwrap();
    assert_eq!(
        prepare_bound_gated_multiply_q7_material(&plan, &region)
            .err()
            .unwrap(),
        "gated Q7 multiply region was not lowered from the decoder plan"
    );
}
