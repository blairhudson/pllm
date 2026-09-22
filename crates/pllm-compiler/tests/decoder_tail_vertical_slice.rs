use pllm_compiler::{
    decoder_coverage, execute_model_greedy_token_selection, execute_model_last_token,
    execute_model_token_feedback, lower_model_greedy_token_selection_regions,
    lower_model_last_token_regions, lower_model_token_feedback_regions, CapabilityLevel,
    ModelLastTokenSelection, NumericType,
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

fn wrap32(value: i32) -> u32 {
    value as u32
}

#[test]
fn prefill_last_token_is_length_aware() {
    let plan = qwen_plan();
    let regions = lower_model_last_token_regions(&plan, DecoderMode::Prefill).unwrap();
    assert_eq!(regions.len(), 1);
    let prefill = &regions[0];
    assert_eq!(prefill.selection, ModelLastTokenSelection::LastValid);
    assert_eq!(
        prefill.sequence_lengths_input_id.as_deref(),
        Some("input.sequence_lengths")
    );
    assert_eq!(prefill.axis, 1);
    assert_eq!(prefill.input.shape, vec![2, 3, 8]);
    assert_eq!(prefill.output.shape, vec![2, 8]);

    let input: Vec<u32> = (0..48).collect();
    let output = execute_model_last_token(prefill, &input, Some(&[1, 3])).unwrap();
    let mut expected: Vec<u32> = (0..8).collect();
    expected.extend(40..48);
    assert_eq!(output, expected);

    assert!(execute_model_last_token(prefill, &input, None).is_err());
    assert!(execute_model_last_token(prefill, &input, Some(&[])).is_err());
    assert!(execute_model_last_token(prefill, &input, Some(&[0, 3])).is_err());
    assert!(execute_model_last_token(prefill, &input, Some(&[1, 4])).is_err());
}

#[test]
fn decode_last_token_is_physical_last() {
    let plan = qwen_plan();
    let regions = lower_model_last_token_regions(&plan, DecoderMode::Decode).unwrap();
    assert_eq!(regions.len(), 1);
    let decode = &regions[0];
    assert_eq!(decode.selection, ModelLastTokenSelection::PhysicalLast);
    assert_eq!(decode.sequence_lengths_input_id, None);
    assert_eq!(decode.input.shape, vec![2, 1, 8]);
    assert_eq!(decode.output.shape, vec![2, 8]);

    let input: Vec<u32> = (0..16).collect();
    assert_eq!(
        execute_model_last_token(decode, &input, None).unwrap(),
        input
    );
    assert!(execute_model_last_token(decode, &input, Some(&[1, 1])).is_err());
    let mut forged = decode.clone();
    forged.sequence_lengths_input_id = Some("input.sequence_lengths".into());
    assert!(execute_model_last_token(&forged, &input, None).is_err());
}

#[test]
fn greedy_token_selection_is_signed_with_lowest_index_ties() {
    let plan = qwen_plan();
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        let regions = lower_model_greedy_token_selection_regions(&plan, mode).unwrap();
        assert_eq!(regions.len(), 1);
        let region = &regions[0];
        assert_eq!(region.input.numeric, NumericType::Wrap32);
        assert_eq!(region.input.shape, vec![2, 32]);
        assert_eq!(region.output.numeric, NumericType::TokenIdU32);
        assert_eq!(region.output.shape, vec![2]);

        let mut logits = vec![u32::MAX; 64];
        logits[2] = wrap32(7);
        logits[5] = wrap32(7);
        for value in &mut logits[32..64] {
            *value = wrap32(-5);
        }
        logits[63] = wrap32(-1);
        assert_eq!(
            execute_model_greedy_token_selection(region, &logits).unwrap(),
            vec![2, 31]
        );

        assert!(execute_model_greedy_token_selection(region, &logits[..63]).is_err());
        let mut forged = region.clone();
        forged.output.numeric = NumericType::Wrap32;
        assert!(execute_model_greedy_token_selection(&forged, &logits).is_err());
        let mut forged = region.clone();
        forged.input.shape = vec![u64::MAX, 2];
        forged.output.shape = vec![u64::MAX];
        assert!(execute_model_greedy_token_selection(&forged, &[]).is_err());
    }
}

#[test]
fn token_feedback_passes_token_ids_through() {
    let plan = qwen_plan();
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        let regions = lower_model_token_feedback_regions(&plan, mode).unwrap();
        assert_eq!(regions.len(), 1);
        let region = &regions[0];
        assert_eq!(region.input.numeric, NumericType::TokenIdU32);
        assert_eq!(region.input.shape, vec![2]);
        assert_eq!(region.output.numeric, NumericType::TokenIdU32);
        assert_eq!(region.output.shape, vec![2, 1]);
        assert_eq!(
            execute_model_token_feedback(region, &[2, 31]).unwrap(),
            vec![2, 31]
        );
        assert!(execute_model_token_feedback(region, &[2]).is_err());
        let mut forged = region.clone();
        forged.output.shape = vec![2, 2];
        assert!(execute_model_token_feedback(&forged, &[2, 31]).is_err());
    }
}

#[test]
fn tail_operators_report_executable_coverage() {
    let plan = qwen_plan();
    let coverage = decoder_coverage(&plan, None).unwrap();
    for operator in [
        ModelOperator::LastToken,
        ModelOperator::GreedyTokenSelection,
        ModelOperator::TokenFeedback,
    ] {
        let row = coverage
            .operators
            .iter()
            .find(|row| row.operator == operator)
            .unwrap();
        assert_eq!(row.occurrences, 2, "{operator:?}");
        assert_eq!(row.level, CapabilityLevel::ExecutableRegion, "{operator:?}");
    }
    assert!(!coverage.complete);
}

#[test]
fn lowering_rejects_tampered_tail_operations() {
    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "token_selection").attributes["policy"] =
        serde_json::json!("pllm.greedy.v0");
    assert!(plan.validate().is_err());
    assert!(lower_model_greedy_token_selection_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "token_selection").inputs[0] = "last_hidden".into();
    assert!(plan.validate().is_err());
    assert!(lower_model_greedy_token_selection_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "token_feedback").output_shape = vec![2, 2];
    assert!(plan.validate().is_err());
    assert!(lower_model_token_feedback_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "last_hidden").attributes["selection"] =
        serde_json::json!("last_valid");
    assert!(plan.validate().is_err());
    assert!(lower_model_last_token_regions(&plan, DecoderMode::Decode).is_err());
}
