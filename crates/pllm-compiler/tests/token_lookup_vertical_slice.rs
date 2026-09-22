use pllm_compiler::{
    decoder_coverage, execute_model_token_lookup_q10, lower_model_token_lookup_q10_regions,
    CapabilityLevel, NumericType,
};
use pllm_core::{
    token_lookup_q10, TokenLookupQ10Policy, TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS,
    TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS, TOKEN_LOOKUP_Q10_PROFILE,
};
use pllm_models::{lower_model_json, DecoderMode, DecoderWorkload, ModelOperator};

const CONFIG: &[u8] = br#"{
    "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
    "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
    "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
    "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
}"#;

fn qwen_plan() -> pllm_models::DecoderPlan {
    lower_model_json(
        CONFIG,
        DecoderWorkload {
            batch: 2,
            max_input_tokens: 3,
            max_new_tokens: 2,
        },
    )
    .unwrap()
}

fn operation_mut<'a>(
    graph: &'a mut pllm_models::DecoderGraph,
    id: &str,
) -> &'a mut pllm_models::ModelOperation {
    graph
        .operations
        .iter_mut()
        .find(|operation| operation.id == id)
        .unwrap()
}

fn policy() -> TokenLookupQ10Policy {
    TokenLookupQ10Policy::new(
        TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS,
        TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS,
    )
    .unwrap()
}

#[test]
fn lowers_one_token_lookup_region_per_decoder_phase() {
    let plan = qwen_plan();
    let prefill = lower_model_token_lookup_q10_regions(&plan, DecoderMode::Prefill).unwrap();
    let decode = lower_model_token_lookup_q10_regions(&plan, DecoderMode::Decode).unwrap();
    assert_eq!(prefill.len(), 1);
    assert_eq!(decode.len(), 1);
    let region = &prefill[0];
    assert_eq!(region.mode, DecoderMode::Prefill);
    assert_eq!(region.layer, None);
    assert_eq!(region.operation_id, "token_lookup");
    assert_eq!(region.tokens_input_id, "input.tokens");
    assert_eq!(region.weight_id, "model.embed_tokens.weight");
    assert_eq!(region.numeric_profile, TOKEN_LOOKUP_Q10_PROFILE);
    assert_eq!(region.tokens.numeric, NumericType::TokenIdU32);
    assert_eq!(region.tokens.shape, vec![2, 3]);
    assert_eq!(region.weight_shape, vec![32, 8]);
    assert_eq!(region.output.numeric, NumericType::SignedFixedQ10);
    assert_eq!(region.output.shape, vec![2, 3, 8]);
    let region = &decode[0];
    assert_eq!(region.mode, DecoderMode::Decode);
    assert_eq!(region.tokens.shape, vec![2, 1]);
    assert_eq!(region.weight_shape, vec![32, 8]);
    assert_eq!(region.output.shape, vec![2, 1, 8]);
}

#[test]
fn executes_distinct_table_rows_against_core() {
    let plan = qwen_plan();
    let region = &lower_model_token_lookup_q10_regions(&plan, DecoderMode::Prefill).unwrap()[0];
    let weights: Vec<i16> = (0..32 * 8)
        .map(|index| ((index * 41) % 509 - 254) as i16)
        .collect();
    let token_ids = [31_u32, 0, 7, 7, 15, 2];
    let output = execute_model_token_lookup_q10(region, &token_ids, &weights, policy()).unwrap();
    assert_eq!(output.shape(), [2, 3, 8]);
    let expected = token_lookup_q10(&token_ids, [2, 3], &weights, [32, 8], policy()).unwrap();
    assert_eq!(output.values(), expected.values());
    for (row, &token) in token_ids.iter().enumerate() {
        let token = token as usize;
        assert_eq!(
            &output.values()[row * 8..row * 8 + 8],
            &weights[token * 8..token * 8 + 8]
        );
    }
}

#[test]
fn rejects_out_of_range_tokens_and_forged_regions() {
    let plan = qwen_plan();
    let region = lower_model_token_lookup_q10_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let weights = vec![0_i16; 32 * 8];
    let tokens = vec![0_u32; 6];
    let error = execute_model_token_lookup_q10(&region, &[0, 1, 2, 3, 4, 32], &weights, policy())
        .err()
        .unwrap();
    assert!(error.contains("outside vocabulary"));
    assert!(execute_model_token_lookup_q10(&region, &tokens[..5], &weights, policy()).is_err());
    assert!(execute_model_token_lookup_q10(&region, &tokens, &weights[..255], policy()).is_err());
    for mutate in [
        (|region: &mut pllm_compiler::ModelTokenLookupQ10Region| {
            region.numeric_profile = "forged".into();
        }) as fn(&mut pllm_compiler::ModelTokenLookupQ10Region),
        |region| region.tokens.numeric = NumericType::SignedFixedQ10,
        |region| region.output.numeric = NumericType::Wrap32,
        |region| region.tokens_input_id = "input.positions".into(),
        |region| region.weight_id = String::new(),
        |region| region.weight_shape = vec![32, 7],
        |region| region.layer = Some(0),
    ] {
        let mut forged = region.clone();
        mutate(&mut forged);
        assert!(execute_model_token_lookup_q10(&forged, &tokens, &weights, policy()).is_err());
    }
}

#[test]
fn reports_token_lookup_as_executable_region() {
    let plan = qwen_plan();
    let coverage = decoder_coverage(&plan, None).unwrap();
    let row = coverage
        .operators
        .iter()
        .find(|row| row.operator == ModelOperator::TokenLookup)
        .unwrap();
    assert_eq!(row.occurrences, 2);
    assert_eq!(row.level, CapabilityLevel::ExecutableRegion);
    assert_eq!(
        row.component.as_deref(),
        Some("pllm/client-token-lookup-q10@0.1.0-alpha.1")
    );
    assert_eq!(
        row.blocker,
        "client-local Q10 token embedding lookup executes with bounded vocabulary checks, but whole-decoder scheduling is unavailable"
    );
    assert!(!coverage.complete);
}

#[test]
fn rejects_tampered_lookup_plans() {
    let mut tampered = qwen_plan();
    operation_mut(&mut tampered.prefill, "token_lookup").inputs[0] = "input.positions".into();
    assert!(lower_model_token_lookup_q10_regions(&tampered, DecoderMode::Prefill).is_err());

    let mut tampered = qwen_plan();
    operation_mut(&mut tampered.prefill, "token_lookup").output_shape = vec![2, 3];
    assert!(lower_model_token_lookup_q10_regions(&tampered, DecoderMode::Prefill).is_err());

    let mut tampered = qwen_plan();
    operation_mut(&mut tampered.prefill, "token_lookup").output_shape[0] = 1;
    assert!(lower_model_token_lookup_q10_regions(&tampered, DecoderMode::Prefill).is_err());

    let mut tampered = qwen_plan();
    operation_mut(&mut tampered.prefill, "token_lookup")
        .attributes
        .as_object_mut()
        .unwrap()
        .remove("weight");
    assert!(lower_model_token_lookup_q10_regions(&tampered, DecoderMode::Prefill).is_err());

    let mut tampered = qwen_plan();
    operation_mut(&mut tampered.prefill, "token_lookup").layer = Some(0);
    assert!(lower_model_token_lookup_q10_regions(&tampered, DecoderMode::Prefill).is_err());

    let mut tampered = qwen_plan();
    operation_mut(&mut tampered.prefill, "output_head").output_shape[1] = 0;
    assert!(lower_model_token_lookup_q10_regions(&tampered, DecoderMode::Prefill).is_err());
}

#[test]
fn gemma_per_layer_lookup_stays_primitive() {
    let gemma = lower_model_json(
        include_bytes!("../../pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 2,
            max_new_tokens: 1,
        },
    )
    .unwrap();
    assert!(lower_model_token_lookup_q10_regions(&gemma, DecoderMode::Prefill).is_err());
    let coverage = decoder_coverage(&gemma, None).unwrap();
    let row = coverage
        .operators
        .iter()
        .find(|row| row.operator == ModelOperator::TokenLookup)
        .unwrap();
    assert_eq!(row.level, CapabilityLevel::Primitive);
    assert_eq!(
        row.component.as_deref(),
        Some("pllm/agc-project@0.1.0-alpha.1-reference")
    );
    assert!(!coverage.complete);
}
