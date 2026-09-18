use pllm_compiler::{
    decoder_coverage, execute_model_attention_scores_q20, lower_model_attention_scores_q20_regions,
    CapabilityLevel, NumericType,
};
use pllm_core::{AttentionScoreQ20Layout, AttentionScoreQ20Policy, ATTENTION_SCORE_Q20_PROFILE};
use pllm_models::{lower_model_json, DecoderMode, DecoderWorkload, ModelOperator, StateKind};

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

fn policy() -> AttentionScoreQ20Policy {
    AttentionScoreQ20Policy::new(
        pllm_core::ATTENTION_Q20_MAX_SCORE_ELEMENTS,
        pllm_core::ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES,
    )
    .unwrap()
}

#[test]
fn lowers_one_score_region_per_decoder_phase() {
    let plan = qwen_plan();
    let prefill = lower_model_attention_scores_q20_regions(&plan, DecoderMode::Prefill).unwrap();
    let decode = lower_model_attention_scores_q20_regions(&plan, DecoderMode::Decode).unwrap();
    assert_eq!(prefill.len(), 1);
    assert_eq!(decode.len(), 1);
    for region in prefill.iter().chain(&decode) {
        assert_eq!(region.numeric_profile, ATTENTION_SCORE_Q20_PROFILE);
        assert_eq!(region.layout, AttentionScoreQ20Layout::GroupedQueryCache);
        assert_eq!(region.query.numeric, NumericType::SignedFixedQ10);
        assert_eq!(region.key.numeric, NumericType::SignedFixedQ10);
        assert_eq!(region.output.numeric, NumericType::SignedFixedQ20);
        assert_eq!(region.score_operation_id, "layer.0.attention_scores");
        assert_eq!(region.scale_operation_id, "layer.0.attention_scale");
        assert_eq!(region.mask_operation_id, "layer.0.causal_mask");
        assert_eq!(region.query_input_id, "layer.0.rope_q");
        assert_eq!(region.key_input_id, "layer.0.key_view");
        assert_eq!(region.selected_positions_input_id, None);
        assert_eq!(region.layer, Some(0));
    }
    assert_eq!(prefill[0].query.shape, vec![2, 2, 3, 4]);
    assert_eq!(prefill[0].key.shape, vec![2, 1, 3, 4]);
    assert_eq!(prefill[0].output.shape, vec![2, 2, 3, 3]);
    assert_eq!(decode[0].query.shape, vec![2, 2, 1, 4]);
    assert_eq!(decode[0].key.shape, vec![2, 1, 4, 4]);
    assert_eq!(decode[0].output.shape, vec![2, 2, 1, 4]);
}

#[test]
fn executes_grouped_query_scores_against_core() {
    let plan = qwen_plan();
    let region = &lower_model_attention_scores_q20_regions(&plan, DecoderMode::Decode).unwrap()[0];
    let query: Vec<i16> = (0..16)
        .map(|index| (index as i64 * 5 % 9 - 4) as i16)
        .collect();
    let key: Vec<i16> = (0..32)
        .map(|index| (index as i64 * 7 % 11 - 5) as i16)
        .collect();
    let positions = [3_u32, 2];
    let query_mask = [1_u8, 1];
    let valid_lengths = [4_usize, 3];
    let output = execute_model_attention_scores_q20(
        region,
        &query,
        &key,
        None,
        &positions,
        &query_mask,
        Some(&valid_lengths),
        policy(),
    )
    .unwrap();
    let direct = pllm_core::attention_scores_q20(
        &query,
        [2, 2, 1, 4],
        &key,
        [2, 1, 4, 4],
        &positions,
        &query_mask,
        &valid_lengths,
        policy(),
    )
    .unwrap();
    assert_eq!(output.shape(), [2, 2, 1, 4]);
    assert_eq!(output.scores(), direct.scores());
    assert_eq!(output.allowed(), direct.allowed());
    assert_eq!(
        output.allowed(),
        &[
            true, true, true, true, true, true, true, true, true, true, true, false, true, true,
            true, false
        ]
    );
}

#[test]
fn composes_with_mpcache_per_query_head_windows() {
    let plan = qwen_plan();
    let base =
        lower_model_attention_scores_q20_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
    let optimized =
        pllm_models::cache::optimize(&plan, pllm_models::cache::MpcachePolicy::paper_profile())
            .unwrap();
    let regions =
        lower_model_attention_scores_q20_regions(&optimized, DecoderMode::Decode).unwrap();
    assert_eq!(regions.len(), 1);
    let region = &regions[0];
    assert_eq!(region.layout, AttentionScoreQ20Layout::PerQueryHeadWindow);
    assert_eq!(region.key.shape.len(), 5);
    assert!(region.key.shape[3] < base.key.shape[2]);
    assert_eq!(region.output.shape[3], region.key.shape[3]);
    assert_eq!(
        region.selected_positions_input_id.as_deref(),
        Some("method.mpcache.layer.0.selected_positions")
    );
    let gather = optimized
        .decode
        .operations
        .iter()
        .find(|operation| operation.id == region.key_input_id)
        .unwrap();
    assert_eq!(
        gather.inputs[1].as_str(),
        region.selected_positions_input_id.as_deref().unwrap()
    );
    let window = region.key.shape[3] as usize;
    let query_elements: usize = region.query.shape.iter().product::<u64>() as usize;
    let key_elements: usize = region.key.shape.iter().product::<u64>() as usize;
    let query: Vec<i16> = (0..query_elements)
        .map(|index| (index as i64 * 3 % 7 - 3) as i16)
        .collect();
    let key: Vec<i16> = (0..key_elements)
        .map(|index| (index as i64 * 5 % 13 - 6) as i16)
        .collect();
    let mut selected = vec![0_u32; 4 * window];
    for row in 0..4 {
        for k in 0..window {
            selected[row * window + k] = k as u32;
        }
    }
    let positions = [3_u32, 0];
    let query_mask = [1_u8, 1];
    let output = execute_model_attention_scores_q20(
        region,
        &query,
        &key,
        Some(&selected),
        &positions,
        &query_mask,
        None,
        policy(),
    )
    .unwrap();
    assert_eq!(output.shape(), [2, 2, 1, window]);
    for row in 0..4 {
        for k in 0..window {
            let index = row * window + k;
            let expected_allowed = row < 2 || k == 0;
            assert_eq!(
                output.allowed()[index],
                expected_allowed,
                "row {row} key {k}"
            );
        }
    }
}

#[test]
fn reports_score_chain_as_executable_regions() {
    let plan = qwen_plan();
    let coverage = decoder_coverage(&plan, "research.single_evaluator");
    let expected = [
        (
            ModelOperator::AttentionScores,
            "client-local Q10 attention scoring executes with exact Q20 scaling and masking, but whole-decoder scheduling is unavailable",
        ),
        (
            ModelOperator::AttentionScale,
            "attention scaling is fused into the plan-bound Q20 score region, but whole-decoder scheduling is unavailable",
        ),
        (
            ModelOperator::CausalMask,
            "causal and validity masking is fused into the plan-bound Q20 score region, but whole-decoder scheduling is unavailable",
        ),
    ];
    for (operator, blocker) in expected {
        let row = coverage
            .operators
            .iter()
            .find(|entry| entry.operator == operator)
            .unwrap();
        assert_eq!(row.occurrences, 2);
        assert_eq!(row.level, CapabilityLevel::ExecutableRegion);
        assert_eq!(
            row.component.as_deref(),
            Some("pllm/client-attention-scores-q20@0.1.0-alpha.1")
        );
        assert_eq!(row.blocker, blocker);
    }
    assert!(!coverage.complete);
}

#[test]
fn rejects_tampered_score_chain_plans() {
    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.attention_scores").inputs =
        vec!["layer.0.key_view".to_owned(), "layer.0.key_view".to_owned()];
    assert!(lower_model_attention_scores_q20_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.attention_scale").output_shape = vec![2, 2, 1, 8];
    assert!(lower_model_attention_scores_q20_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.attention_scale")
        .attributes
        .as_object_mut()
        .unwrap()
        .insert("head_dim".into(), serde_json::json!(8));
    assert!(lower_model_attention_scores_q20_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.causal_mask").inputs =
        vec!["layer.0.attention_scale".to_owned()];
    assert!(lower_model_attention_scores_q20_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.key_view").state_kind = Some(StateKind::Value);
    assert!(lower_model_attention_scores_q20_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.attention_scores")
        .attributes
        .as_object_mut()
        .unwrap()
        .insert("group_size".into(), serde_json::json!(1));
    assert!(lower_model_attention_scores_q20_regions(&plan, DecoderMode::Decode).is_err());

    let mut tampered = pllm_models::cache::optimize(
        &qwen_plan(),
        pllm_models::cache::MpcachePolicy::paper_profile(),
    )
    .unwrap();
    operation_mut(
        &mut tampered.decode,
        "method.mpcache.layer.0.dynamic_key_gather",
    )
    .state_kind = Some(StateKind::Value);
    assert!(lower_model_attention_scores_q20_regions(&tampered, DecoderMode::Decode).is_err());

    let mut tampered = pllm_models::cache::optimize(
        &qwen_plan(),
        pllm_models::cache::MpcachePolicy::paper_profile(),
    )
    .unwrap();
    operation_mut(&mut tampered.decode, "layer.0.causal_mask").inputs = vec![
        "layer.0.attention_scale".to_owned(),
        "input.positions".to_owned(),
    ];
    assert!(lower_model_attention_scores_q20_regions(&tampered, DecoderMode::Decode).is_err());
}

#[test]
fn rejects_forged_regions_options_and_buffers() {
    let plan = qwen_plan();
    let region =
        lower_model_attention_scores_q20_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
    let query = vec![1_i16; 16];
    let key = vec![1_i16; 32];
    let positions = [3_u32, 2];
    let query_mask = [1_u8, 1];
    let valid_lengths = [4_usize, 3];

    let mut forged = region.clone();
    forged.numeric_profile = "forged".to_owned();
    assert!(execute_model_attention_scores_q20(
        &forged,
        &query,
        &key,
        None,
        &positions,
        &query_mask,
        Some(&valid_lengths),
        policy(),
    )
    .is_err());
    let mut forged = region.clone();
    forged.query.numeric = NumericType::SignedFixedQ20;
    assert!(execute_model_attention_scores_q20(
        &forged,
        &query,
        &key,
        None,
        &positions,
        &query_mask,
        Some(&valid_lengths),
        policy(),
    )
    .is_err());
    let mut forged = region.clone();
    forged.layout = AttentionScoreQ20Layout::PerQueryHeadWindow;
    forged.selected_positions_input_id = Some("forged".to_owned());
    assert!(execute_model_attention_scores_q20(
        &forged,
        &query,
        &key,
        Some(&[0_u32; 16]),
        &positions,
        &query_mask,
        None,
        policy(),
    )
    .is_err());
    let mut forged = region.clone();
    forged.output.shape = vec![2, 2, 1, 8];
    assert!(execute_model_attention_scores_q20(
        &forged,
        &query,
        &key,
        None,
        &positions,
        &query_mask,
        Some(&valid_lengths),
        policy(),
    )
    .is_err());
    assert!(execute_model_attention_scores_q20(
        &region,
        &query,
        &key,
        Some(&[0_u32; 16]),
        &positions,
        &query_mask,
        Some(&valid_lengths),
        policy(),
    )
    .is_err());
    assert!(execute_model_attention_scores_q20(
        &region,
        &query,
        &key,
        None,
        &positions,
        &query_mask,
        None,
        policy(),
    )
    .is_err());
    assert!(execute_model_attention_scores_q20(
        &region,
        &query,
        &key[..key.len() - 1],
        None,
        &positions,
        &query_mask,
        Some(&valid_lengths),
        policy(),
    )
    .is_err());
    assert!(execute_model_attention_scores_q20(
        &region,
        &query,
        &key,
        None,
        &positions[..1],
        &query_mask,
        Some(&valid_lengths),
        policy(),
    )
    .is_err());

    let optimized =
        pllm_models::cache::optimize(&plan, pllm_models::cache::MpcachePolicy::paper_profile())
            .unwrap();
    let window_region = lower_model_attention_scores_q20_regions(&optimized, DecoderMode::Decode)
        .unwrap()[0]
        .clone();
    let window = window_region.key.shape[3] as usize;
    let query: Vec<i16> = vec![1; window_region.query.shape.iter().product::<u64>() as usize];
    let key: Vec<i16> = vec![1; window_region.key.shape.iter().product::<u64>() as usize];
    let selected = vec![0_u32; 4 * window];
    assert!(execute_model_attention_scores_q20(
        &window_region,
        &query,
        &key,
        None,
        &[3, 2],
        &[1, 1],
        None,
        policy(),
    )
    .is_err());
    assert!(execute_model_attention_scores_q20(
        &window_region,
        &query,
        &key,
        Some(&selected),
        &[3, 2],
        &[1, 1],
        Some(&[4, 4]),
        policy(),
    )
    .is_err());
    let mut unsorted = selected.clone();
    unsorted[0] = 2;
    unsorted[1] = 1;
    assert!(execute_model_attention_scores_q20(
        &window_region,
        &query,
        &key,
        Some(&unsorted),
        &[3, 2],
        &[1, 1],
        None,
        policy(),
    )
    .is_err());
}
