use pllm_compiler::{
    decoder_coverage, execute_model_softmax_q30, lower_model_softmax_q30_regions, CapabilityLevel,
    NumericType,
};
use pllm_core::{SoftmaxQ30Policy, SOFTMAX_Q20_TO_Q30_PROFILE, SOFTMAX_Q30_ONE};
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

fn policy() -> SoftmaxQ30Policy {
    SoftmaxQ30Policy::new(
        pllm_core::SOFTMAX_Q30_MAX_ELEMENTS,
        pllm_core::SOFTMAX_Q30_MAX_ROW_LENGTH,
    )
    .unwrap()
}

#[test]
fn lowers_one_softmax_region_per_decoder_phase() {
    let plan = qwen_plan();
    let prefill = lower_model_softmax_q30_regions(&plan, DecoderMode::Prefill).unwrap();
    let decode = lower_model_softmax_q30_regions(&plan, DecoderMode::Decode).unwrap();
    assert_eq!(prefill.len(), 1);
    assert_eq!(decode.len(), 1);
    for region in prefill.iter().chain(&decode) {
        assert_eq!(region.axis, 3);
        assert_eq!(region.numeric_profile, SOFTMAX_Q20_TO_Q30_PROFILE);
        assert_eq!(region.input.numeric, NumericType::SignedFixedQ20);
        assert_eq!(region.output.numeric, NumericType::UnsignedFixedQ30);
        assert_eq!(region.input.shape, region.output.shape);
    }
    assert_eq!(prefill[0].input.shape, vec![2, 2, 3, 3]);
    assert_eq!(prefill[0].operation_id, "layer.0.softmax");
    assert_eq!(decode[0].input.shape, vec![2, 2, 1, 4]);
    assert_eq!(decode[0].mode, DecoderMode::Decode);
    assert_eq!(decode[0].layer, Some(0));
}

#[test]
fn executes_masked_prefix_rows_with_exact_q30_sums() {
    let plan = qwen_plan();
    let region = &lower_model_softmax_q30_regions(&plan, DecoderMode::Prefill).unwrap()[0];
    let row_length = 3_usize;
    let elements = 2 * 2 * 3 * row_length;
    let scores = vec![0_i64; elements];
    let mut allowed = vec![false; elements];
    for row in 0..elements / row_length {
        if row == 4 {
            continue;
        }
        let valid = row % row_length + 1;
        for index in 0..valid {
            allowed[row * row_length + index] = true;
        }
    }
    let output = execute_model_softmax_q30(region, &scores, &allowed, policy()).unwrap();
    assert_eq!(output.shape(), [2, 2, 3, 3]);
    let probabilities = output.probabilities();
    for row in 0..elements / row_length {
        let slice = &probabilities[row * row_length..(row + 1) * row_length];
        if row == 4 {
            assert!(slice.iter().all(|&value| value == 0));
            continue;
        }
        assert_eq!(
            slice.iter().copied().map(i64::from).sum::<i64>(),
            i64::from(SOFTMAX_Q30_ONE)
        );
        let valid = row % row_length + 1;
        for (index, &value) in slice.iter().enumerate() {
            if index >= valid {
                assert_eq!(value, 0);
            }
        }
        let minimum = slice[..valid].iter().copied().min().unwrap();
        let maximum = slice[..valid].iter().copied().max().unwrap();
        assert!(maximum - minimum <= 1);
    }
}

#[test]
fn executes_extreme_scores_through_the_compiler_region() {
    let plan = qwen_plan();
    let region = &lower_model_softmax_q30_regions(&plan, DecoderMode::Decode).unwrap()[0];
    let elements: usize = region.input.shape.iter().product::<u64>() as usize;
    let row_length = *region.input.shape.last().unwrap() as usize;
    let mut scores = vec![0_i64; elements];
    for (index, value) in scores[..row_length].iter_mut().enumerate() {
        *value = if index % 2 == 0 { i64::MIN } else { i64::MAX };
    }
    let allowed = vec![true; elements];
    let output = execute_model_softmax_q30(region, &scores, &allowed, policy()).unwrap();
    for row in output.probabilities().chunks_exact(row_length) {
        assert_eq!(
            row.iter().copied().map(i64::from).sum::<i64>(),
            i64::from(SOFTMAX_Q30_ONE)
        );
        assert!(row.iter().all(|&value| value >= 0));
    }
    assert_eq!(output.probabilities()[0], 0);
}

#[test]
fn rejects_tampered_plans_regions_and_buffers() {
    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.softmax")
        .attributes
        .as_object_mut()
        .unwrap()
        .insert("axis".into(), serde_json::json!(0));
    assert!(lower_model_softmax_q30_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.softmax").inputs =
        vec!["layer.0.attention_scale".to_owned()];
    assert!(lower_model_softmax_q30_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.softmax").output_shape = vec![2, 2, 1, 3];
    assert!(lower_model_softmax_q30_regions(&plan, DecoderMode::Decode).is_err());

    let plan = qwen_plan();
    let region = lower_model_softmax_q30_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
    let elements: usize = region.input.shape.iter().product::<u64>() as usize;
    let scores = vec![0_i64; elements];
    let allowed = vec![true; elements];
    let mut forged = region.clone();
    forged.numeric_profile = "forged".to_owned();
    assert!(execute_model_softmax_q30(&forged, &scores, &allowed, policy()).is_err());
    let mut forged = region.clone();
    forged.input.numeric = NumericType::Wrap32;
    assert!(execute_model_softmax_q30(&forged, &scores, &allowed, policy()).is_err());
    let mut forged = region.clone();
    forged.output.numeric = NumericType::SignedFixedQ20;
    assert!(execute_model_softmax_q30(&forged, &scores, &allowed, policy()).is_err());
    let mut forged = region.clone();
    forged.axis = 0;
    assert!(execute_model_softmax_q30(&forged, &scores, &allowed, policy()).is_err());
    let mut forged = region.clone();
    forged.output.shape = vec![2, 2, 1, 3];
    assert!(execute_model_softmax_q30(&forged, &scores, &allowed, policy()).is_err());
    assert!(
        execute_model_softmax_q30(&region, &scores[..elements - 1], &allowed, policy()).is_err()
    );
    assert!(
        execute_model_softmax_q30(&region, &scores, &allowed[..elements - 1], policy()).is_err()
    );
}

#[test]
fn reports_softmax_as_an_executable_region_with_scheduling_blocker() {
    let plan = qwen_plan();
    let coverage = decoder_coverage(&plan, None).unwrap();
    let softmax = coverage
        .operators
        .iter()
        .find(|row| row.operator == ModelOperator::Softmax)
        .unwrap();
    assert_eq!(softmax.occurrences, 2);
    assert_eq!(softmax.level, CapabilityLevel::ExecutableRegion);
    assert_eq!(
        softmax.component.as_deref(),
        Some("pllm/client-softmax-q30@0.1.0-alpha.1")
    );
    assert!(softmax.blocker.contains("exact row sums"));
    assert!(!coverage.complete);
}

#[test]
fn composes_with_the_mpcache_structural_transform() {
    let plan = qwen_plan();
    let base_decode =
        lower_model_softmax_q30_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
    let optimized = pllm_models::cache::optimize(
        &plan,
        pllm_models::cache::MpcachePolicy::r23_reference_policy(),
    )
    .unwrap();
    let transformed = lower_model_softmax_q30_regions(&optimized, DecoderMode::Decode).unwrap();
    assert_eq!(transformed.len(), 1);
    let region = &transformed[0];
    assert_eq!(region.numeric_profile, SOFTMAX_Q20_TO_Q30_PROFILE);
    assert!(region.input.shape[3] < base_decode.input.shape[3]);
    let elements: usize = region.input.shape.iter().product::<u64>() as usize;
    let row_length = region.input.shape[3] as usize;
    let scores = vec![0_i64; elements];
    let allowed = vec![true; elements];
    let output = execute_model_softmax_q30(region, &scores, &allowed, policy()).unwrap();
    for row in output.probabilities().chunks_exact(row_length) {
        assert_eq!(
            row.iter().copied().map(i64::from).sum::<i64>(),
            i64::from(SOFTMAX_Q30_ONE)
        );
    }
}
