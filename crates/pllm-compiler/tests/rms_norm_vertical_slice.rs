use pllm_compiler::{
    decoder_coverage, execute_rms_norm_f32_direct, execute_rms_norm_q10_direct,
    lower_rms_norm_f32_direct_regions, lower_rms_norm_q10_direct_regions,
    prepare_bound_rms_norm_q10_row, CapabilityLevel, ExperimentalRmsNormQ10Policy,
    RmsNormWeightPolicy, RMS_NORM_F32_DIRECT_REGION_SCHEMA_VERSION,
    RMS_NORM_Q10_DIRECT_REGION_SCHEMA_VERSION,
};
use pllm_models::{lower_model_json, DecoderMode, DecoderWorkload, ModelOperator};

fn qwen2_plan() -> pllm_models::DecoderPlan {
    lower_model_json(
        br#"{
            "model_type":"qwen2","hidden_size":8,"intermediate_size":4,
            "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
            "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
            "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
        }"#,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 2,
            max_new_tokens: 1,
        },
    )
    .unwrap()
}

#[test]
fn direct_weight_norm_regions_bind_semantics_and_execute_reference() {
    let plan = qwen2_plan();
    let prefill = lower_rms_norm_f32_direct_regions(&plan, DecoderMode::Prefill).unwrap();
    assert_eq!(prefill.len(), 3);
    assert!(prefill.iter().all(|region| {
        region.schema_version == RMS_NORM_F32_DIRECT_REGION_SCHEMA_VERSION
            && region.model_plan_digest == plan.digest()
            && region.mode == DecoderMode::Prefill
            && region.shape == vec![1, 2, 8]
            && region.epsilon == "1e-6"
            && region.weight_policy == RmsNormWeightPolicy::Direct
            && region.weight_id.ends_with(".weight")
    }));
    let input_norm = prefill
        .iter()
        .find(|region| region.operation_id == "layer.0.input_norm")
        .unwrap();
    let input = [1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]
        .into_iter()
        .cycle()
        .take(16)
        .collect::<Vec<_>>();
    let output = execute_rms_norm_f32_direct(&plan, input_norm, &input, &[1.0; 8]).unwrap();
    let denominator = (0.5f32 + 0.000_001).sqrt();
    assert_eq!(output.len(), input.len());
    for (actual, source) in output.into_iter().zip(input) {
        assert!((actual - source / denominator).abs() <= f32::EPSILON * 2.0);
    }
    let decode = lower_rms_norm_f32_direct_regions(&plan, DecoderMode::Decode).unwrap();
    assert_eq!(decode.len(), 3);
    assert!(decode.iter().all(|region| region.shape == vec![1, 1, 8]));

    let coverage = decoder_coverage(&plan, "research.single_evaluator");
    let rms_norm = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == ModelOperator::RmsNorm)
        .unwrap();
    assert_eq!(rms_norm.occurrences, 6);
    assert_eq!(rms_norm.level, CapabilityLevel::Primitive);
    assert!(rms_norm
        .component
        .as_deref()
        .is_some_and(|component| component.starts_with("pllm/core")));
    assert!(rms_norm
        .blocker
        .contains("no protected numeric decomposition"));
}

#[test]
fn direct_weight_norm_execution_rejects_tampering_and_bad_buffers() {
    let plan = qwen2_plan();
    let region = lower_rms_norm_f32_direct_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
    assert!(execute_rms_norm_f32_direct(&plan, &region, &[1.0; 8], &[1.0; 8]).is_ok());
    assert!(execute_rms_norm_f32_direct(&plan, &region, &[1.0; 7], &[1.0; 8]).is_err());
    assert!(execute_rms_norm_f32_direct(&plan, &region, &[1.0; 8], &[1.0; 7]).is_err());

    let mut forged = region.clone();
    forged.weight_id = "forged.weight".into();
    assert!(execute_rms_norm_f32_direct(&plan, &forged, &[1.0; 8], &[1.0; 8]).is_err());
    let mut forged = region.clone();
    forged.model_plan_digest = pllm_types::digest_bytes("test.forged.v1", b"forged");
    assert!(execute_rms_norm_f32_direct(&plan, &forged, &[1.0; 8], &[1.0; 8]).is_err());
    let mut forged = region;
    forged.kernel_artifact_digest = pllm_types::digest_bytes("test.forged.v1", b"kernel");
    assert!(execute_rms_norm_f32_direct(&plan, &forged, &[1.0; 8], &[1.0; 8]).is_err());
}

#[test]
fn qwen3_per_head_norms_use_direct_weight_regions() {
    let plan = lower_model_json(
        include_bytes!("../../pllm-models/tests/fixtures/mini-coder-4b-c87892d-config.json"),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 1,
        },
    )
    .unwrap();
    let regions = lower_rms_norm_f32_direct_regions(&plan, DecoderMode::Decode).unwrap();
    assert_eq!(regions.len(), 145);
    let query = regions
        .iter()
        .find(|region| region.operation_id == "layer.0.q_norm")
        .unwrap();
    assert_eq!(query.shape, vec![1, 32, 1, 128]);
    assert_eq!(query.weight_id, "model.layers.0.self_attn.q_norm.weight");
    let key = regions
        .iter()
        .find(|region| region.operation_id == "layer.0.k_norm")
        .unwrap();
    assert_eq!(key.shape, vec![1, 8, 1, 128]);
    assert_eq!(key.weight_id, "model.layers.0.self_attn.k_norm.weight");
}

#[test]
fn direct_weight_lowering_is_semantic_not_qwen_named() {
    let workload = DecoderWorkload {
        batch: 1,
        max_input_tokens: 1,
        max_new_tokens: 1,
    };
    let phi = lower_model_json(
        include_bytes!("../../pllm-models/tests/fixtures/Phi-4-mini-instruct-cfbefac-config.json"),
        workload,
    )
    .unwrap();
    let regions = lower_rms_norm_f32_direct_regions(&phi, DecoderMode::Decode).unwrap();
    assert!(!regions.is_empty());
    assert!(regions
        .iter()
        .all(|region| region.weight_policy == RmsNormWeightPolicy::Direct));

    let offset_weight = lower_model_json(
        include_bytes!("../../pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json"),
        workload,
    )
    .unwrap();
    let error = lower_rms_norm_f32_direct_regions(&offset_weight, DecoderMode::Decode).unwrap_err();
    assert!(error.contains("weight offset 1"));

    let gemma = lower_model_json(
        include_bytes!("../../pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"),
        workload,
    )
    .unwrap();
    let error = lower_rms_norm_f32_direct_regions(&gemma, DecoderMode::Decode).unwrap_err();
    assert!(error.contains("weight offset 1"));
}

#[test]
fn q10_direct_profile_binds_exact_numeric_contract_and_plan() {
    let plan = qwen2_plan();
    let regions = lower_rms_norm_q10_direct_regions(&plan, DecoderMode::Decode).unwrap();
    let region = &regions[0];
    assert_eq!(
        region.schema_version,
        RMS_NORM_Q10_DIRECT_REGION_SCHEMA_VERSION
    );
    assert_eq!(region.epsilon_numerator, 1);
    assert_eq!(region.epsilon_denominator, 1_000_000);
    assert_eq!(region.maximum_encoded_error, 1);
    assert_eq!(region.shape, vec![1, 1, 8]);
    assert_eq!(
        execute_rms_norm_q10_direct(&plan, region, &[1024; 8], &[1024; 8]).unwrap(),
        vec![1024; 8]
    );

    let mut forged = region.clone();
    forged.maximum_encoded_error = 2;
    assert!(execute_rms_norm_q10_direct(&plan, &forged, &[1024; 8], &[1024; 8]).is_err());

    let workload = DecoderWorkload {
        batch: 1,
        max_input_tokens: 2,
        max_new_tokens: 1,
    };
    let phi = lower_model_json(
        include_bytes!("../../pllm-models/tests/fixtures/Phi-4-mini-instruct-cfbefac-config.json"),
        workload,
    )
    .unwrap();
    assert!(lower_rms_norm_q10_direct_regions(&phi, DecoderMode::Decode)
        .unwrap_err()
        .contains("unsupported by the exact 1/1000000 Q10 profile"));
}

#[test]
fn protected_q10_row_is_plan_bound_one_use_material() {
    let plan = qwen2_plan();
    let region = lower_rms_norm_q10_direct_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
    let input = [1536_i16, -768, 256, -128, 1024, 2048, -1536, 512];
    let weights = [1024_i16, 896, 960, 1088, 1024, 1000, 1040, 992];
    let expected = execute_rms_norm_q10_direct(&plan, &region, &input, &weights).unwrap();
    let policy =
        ExperimentalRmsNormQ10Policy::acknowledge_unreviewed_public_weights(8_000_000).unwrap();
    let material = prepare_bound_rms_norm_q10_row(&plan, &region, 0, &weights, &policy).unwrap();
    assert_eq!(material.and_gate_count(), 236_200);
    assert_eq!(
        material.evaluator_ciphertext_bytes().unwrap(),
        u64::try_from(material.and_gate_count()).unwrap() * 32
    );
    let (evaluation, decoder) = material.encode(&input).unwrap();
    let outputs = evaluation.evaluate(&plan, &region, 0).unwrap();
    assert_eq!(decoder.decode(outputs).unwrap(), expected);

    let bounded =
        ExperimentalRmsNormQ10Policy::acknowledge_unreviewed_public_weights(1_000_000).unwrap();
    let error = match prepare_bound_rms_norm_q10_row(&plan, &region, 0, &weights, &bounded) {
        Ok(_) => panic!("undersized protected RMSNorm policy was accepted"),
        Err(error) => error,
    };
    assert!(error.contains("exceeding policy limit"));

    let material = prepare_bound_rms_norm_q10_row(&plan, &region, 0, &weights, &policy).unwrap();
    let (evaluation, _decoder) = material.encode(&input).unwrap();
    let mut forged = region.clone();
    forged.maximum_encoded_error = 2;
    assert!(evaluation.evaluate(&plan, &forged, 0).is_err());
    assert!(prepare_bound_rms_norm_q10_row(&plan, &region, 1, &weights, &policy).is_err());
}
