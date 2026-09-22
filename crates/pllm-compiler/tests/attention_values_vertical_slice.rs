use pllm_compiler::{
    decoder_coverage, execute_model_attention_values_q10, lower_model_attention_values_q10_regions,
    CapabilityLevel, NumericType,
};
use pllm_core::{
    softmax_q20_to_q30, AttentionValueQ10Layout, AttentionValueQ10Policy, SoftmaxQ30Policy,
    ATTENTION_VALUE_Q30_Q10_PROFILE,
};
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

fn policy() -> AttentionValueQ10Policy {
    AttentionValueQ10Policy::new(
        pllm_core::ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS,
        pllm_core::ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES,
    )
    .unwrap()
}

fn softmax_policy() -> SoftmaxQ30Policy {
    SoftmaxQ30Policy::new(
        pllm_core::SOFTMAX_Q30_MAX_ELEMENTS,
        pllm_core::SOFTMAX_Q30_MAX_ROW_LENGTH,
    )
    .unwrap()
}

fn uniform_probabilities(shape: [usize; 4]) -> pllm_core::SoftmaxProbabilitiesQ30 {
    let elements: usize = shape.iter().product();
    softmax_q20_to_q30(
        &vec![0_i64; elements],
        &vec![true; elements],
        shape,
        softmax_policy(),
    )
    .unwrap()
}

fn ties_even_div_i128(numerator: i128, denominator: i128) -> i128 {
    let quotient = numerator.div_euclid(denominator);
    let remainder = numerator.rem_euclid(denominator);
    match remainder.cmp(&(denominator - remainder)) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal if quotient % 2 == 0 => quotient,
        std::cmp::Ordering::Equal => quotient + 1,
    }
}

#[test]
fn lowers_one_attention_values_region_per_decoder_phase() {
    let plan = qwen_plan();
    let prefill = lower_model_attention_values_q10_regions(&plan, DecoderMode::Prefill).unwrap();
    let decode = lower_model_attention_values_q10_regions(&plan, DecoderMode::Decode).unwrap();
    assert_eq!(prefill.len(), 1);
    assert_eq!(decode.len(), 1);
    for region in prefill.iter().chain(&decode) {
        assert_eq!(region.numeric_profile, ATTENTION_VALUE_Q30_Q10_PROFILE);
        assert_eq!(region.layout, AttentionValueQ10Layout::GroupedQuery);
        assert_eq!(region.probabilities.numeric, NumericType::UnsignedFixedQ30);
        assert_eq!(region.values.numeric, NumericType::SignedFixedQ10);
        assert_eq!(region.output.numeric, NumericType::SignedFixedQ10);
        assert_eq!(region.probabilities_input_id, "layer.0.softmax");
        assert_eq!(region.values_input_id, "layer.0.value_view");
    }
    assert_eq!(prefill[0].probabilities.shape, vec![2, 2, 3, 3]);
    assert_eq!(prefill[0].values.shape, vec![2, 1, 3, 4]);
    assert_eq!(prefill[0].output.shape, vec![2, 2, 3, 4]);
    assert_eq!(decode[0].probabilities.shape, vec![2, 2, 1, 4]);
    assert_eq!(decode[0].values.shape, vec![2, 1, 4, 4]);
    assert_eq!(decode[0].output.shape, vec![2, 2, 1, 4]);
}

#[test]
fn executes_grouped_query_contraction_against_reference() {
    let plan = qwen_plan();
    let region = &lower_model_attention_values_q10_regions(&plan, DecoderMode::Decode).unwrap()[0];
    let shape = [2_usize, 2, 1, 4];
    let elements: usize = shape.iter().product();
    let mut allowed = vec![true; elements];
    allowed[4..8].fill(false);
    let probabilities =
        softmax_q20_to_q30(&vec![0_i64; elements], &allowed, shape, softmax_policy()).unwrap();
    let keys = shape[3];
    let depth = region.values.shape[3] as usize;
    let value_elements: usize = region.values.shape.iter().product::<u64>() as usize;
    let values: Vec<i16> = (0..value_elements)
        .map(|index| ((index as i64 * 37) % 41 - 20) as i16)
        .collect();
    let output =
        execute_model_attention_values_q10(region, &probabilities, &values, policy()).unwrap();
    assert_eq!(output.shape(), [2, 2, 1, 4]);
    let probabilities_slice = probabilities.probabilities();
    for row in 0..probabilities_slice.len() / keys {
        let inactive = probabilities_slice[row * keys..(row + 1) * keys]
            .iter()
            .all(|&probability| probability == 0);
        for feature in 0..depth {
            let expected = if inactive {
                0_i16
            } else {
                let accumulate: i128 = probabilities_slice[row * keys..(row + 1) * keys]
                    .iter()
                    .enumerate()
                    .map(|(key, &probability)| {
                        i128::from(probability)
                            * i128::from(values[((row / 2) * keys + key) * depth + feature])
                    })
                    .sum();
                i16::try_from(ties_even_div_i128(accumulate, 1_i128 << 30)).unwrap()
            };
            assert_eq!(
                output.values()[row * depth + feature],
                expected,
                "row {row} feature {feature}"
            );
        }
    }
}

#[test]
fn composes_with_mpcache_per_query_head_windows() {
    let plan = qwen_plan();
    let base =
        lower_model_attention_values_q10_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
    let optimized = pllm_models::cache::optimize(
        &plan,
        pllm_models::cache::MpcachePolicy::r23_reference_policy(),
    )
    .unwrap();
    let regions =
        lower_model_attention_values_q10_regions(&optimized, DecoderMode::Decode).unwrap();
    assert_eq!(regions.len(), 1);
    let region = &regions[0];
    assert_eq!(region.layout, AttentionValueQ10Layout::PerQueryHeadWindow);
    assert_eq!(region.values.shape.len(), 5);
    assert!(region.values.shape[3] < base.values.shape[2]);
    assert_eq!(region.probabilities.shape[3], region.values.shape[3]);
    let probabilities = uniform_probabilities([
        region.probabilities.shape[0] as usize,
        region.probabilities.shape[1] as usize,
        region.probabilities.shape[2] as usize,
        region.probabilities.shape[3] as usize,
    ]);
    let value_elements: usize = region.values.shape.iter().product::<u64>() as usize;
    let values: Vec<i16> = (0..value_elements)
        .map(|index| ((index as i64 * 11) % 17 - 8) as i16)
        .collect();
    let output =
        execute_model_attention_values_q10(region, &probabilities, &values, policy()).unwrap();
    assert_eq!(output.shape(), [2, 2, 1, 4]);
    assert!(output.values().iter().all(|&value| value.abs() <= 8));
}

#[test]
fn reports_attention_values_as_executable_region() {
    let plan = qwen_plan();
    let coverage = decoder_coverage(&plan, None).unwrap();
    let row = coverage
        .operators
        .iter()
        .find(|entry| entry.operator == ModelOperator::AttentionValues)
        .unwrap();
    assert_eq!(row.occurrences, 2);
    assert_eq!(row.level, CapabilityLevel::ExecutableRegion);
    assert_eq!(
        row.component.as_deref(),
        Some("pllm/client-attention-values-q10@0.1.0-alpha.1")
    );
    assert_eq!(
        row.blocker,
        "client-local Q30-probability by Q10-value attention contraction executes, but whole-decoder scheduling is unavailable"
    );
    assert!(!coverage.complete);
}

#[test]
fn rejects_tampered_attention_values_plans() {
    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.attention_values").inputs =
        vec!["layer.0.softmax".to_owned(), "layer.0.softmax".to_owned()];
    assert!(lower_model_attention_values_q10_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.attention_values").inputs =
        vec!["layer.0.softmax".to_owned(), "layer.0.key_view".to_owned()];
    assert!(lower_model_attention_values_q10_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.attention_values")
        .attributes
        .as_object_mut()
        .unwrap()
        .insert("group_size".into(), serde_json::json!(1));
    assert!(lower_model_attention_values_q10_regions(&plan, DecoderMode::Decode).is_err());

    let mut plan = qwen_plan();
    operation_mut(&mut plan.decode, "layer.0.attention_values").output_shape = vec![2, 2, 1, 8];
    assert!(lower_model_attention_values_q10_regions(&plan, DecoderMode::Decode).is_err());

    let mut tampered = pllm_models::cache::optimize(
        &qwen_plan(),
        pllm_models::cache::MpcachePolicy::r23_reference_policy(),
    )
    .unwrap();
    operation_mut(
        &mut tampered.decode,
        "method.kv_cache_eviction.layer.0.dynamic_value_gather",
    )
    .state_kind = Some(StateKind::Key);
    assert!(lower_model_attention_values_q10_regions(&tampered, DecoderMode::Decode).is_err());
}

#[test]
fn rejects_forged_regions_buffers_and_invalid_probabilities() {
    let plan = qwen_plan();
    let region =
        lower_model_attention_values_q10_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
    let probabilities = uniform_probabilities([2, 2, 1, 4]);
    let value_elements: usize = region.values.shape.iter().product::<u64>() as usize;
    let values = vec![1_i16; value_elements];
    let mut forged = region.clone();
    forged.numeric_profile = "forged".to_owned();
    assert!(
        execute_model_attention_values_q10(&forged, &probabilities, &values, policy()).is_err()
    );
    let mut forged = region.clone();
    forged.values.numeric = NumericType::Wrap32;
    assert!(
        execute_model_attention_values_q10(&forged, &probabilities, &values, policy()).is_err()
    );
    let mut forged = region.clone();
    forged.layout = AttentionValueQ10Layout::PerQueryHeadWindow;
    assert!(
        execute_model_attention_values_q10(&forged, &probabilities, &values, policy()).is_err()
    );
    let mut forged = region.clone();
    forged.probabilities.shape = vec![2, 2, 1, 3];
    assert!(
        execute_model_attention_values_q10(&forged, &probabilities, &values, policy()).is_err()
    );
    assert!(execute_model_attention_values_q10(
        &region,
        &probabilities,
        &values[..value_elements - 1],
        policy()
    )
    .is_err());
    let wrong_shape = uniform_probabilities([2, 2, 1, 2]);
    assert!(execute_model_attention_values_q10(&region, &wrong_shape, &values, policy()).is_err());
    assert!(matches!(
        pllm_core::attention_values_q10(
            &[1_i32, 1, 0, 0],
            [1, 1, 1, 4],
            &[0_i16; 4],
            &[1, 1, 4, 1],
            policy(),
        ),
        Err(pllm_core::AttentionValueQ10Error::ProbabilityRowSumInvalid { .. })
    ));
}
