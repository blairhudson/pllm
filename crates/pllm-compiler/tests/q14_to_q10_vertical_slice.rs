use pllm_compiler::{
    define_q14_to_q10_rescale_region, execute_q14_to_q10_rescale,
    lower_model_q14_to_q10_rescale_regions, q14_to_q10_rescale_region_digest, FixedPointRounding,
    NumericType, Q14ToQ10RangePolicy, Q14ToQ10RescaleRegion, Q14_TO_Q10_REGION_SCHEMA_VERSION,
};
use pllm_core::{rescale_q14_to_q10_centered_u32_tensor, Q14_TO_Q10_PROFILE};
use pllm_models::{lower_model_json, DecoderMode, DecoderPlan, DecoderWorkload};

const CONFIG: &[u8] = br#"{
    "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
    "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
    "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
    "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
}"#;

fn qwen_plan() -> DecoderPlan {
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

fn centered(value: i32) -> u32 {
    u32::from_ne_bytes(value.to_ne_bytes())
}

#[test]
fn lowers_four_graph_derived_rescale_regions_per_phase() {
    let plan = qwen_plan();
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        let regions = lower_model_q14_to_q10_rescale_regions(&plan, mode).unwrap();
        assert_eq!(regions.len(), 4);
        let expected = [
            ("layer.0.q_linear", "layer.0.q_heads", 0, vec![2, 3, 8]),
            ("layer.0.k_linear", "layer.0.k_heads", 0, vec![2, 3, 4]),
            ("layer.0.v_linear", "layer.0.v_heads", 0, vec![2, 3, 4]),
            (
                "layer.0.o_proj",
                "layer.0.attention_residual",
                1,
                vec![2, 3, 8],
            ),
        ];
        for (region, (source, target, input_index, shape)) in regions.iter().zip(expected.iter()) {
            let shape = if mode == DecoderMode::Prefill {
                shape.clone()
            } else {
                let mut decode_shape = shape.clone();
                decode_shape[1] = 1;
                decode_shape
            };
            assert_eq!(region.schema_version, Q14_TO_Q10_REGION_SCHEMA_VERSION);
            assert_eq!(region.source_operation_id, *source);
            assert_eq!(region.target_operation_id, *target);
            assert_eq!(region.target_input_index, *input_index);
            assert_eq!(
                region.operation_id,
                format!("{target}.input.{input_index}.from.{source}.q14_to_q10")
            );
            assert_eq!(region.numeric_profile, Q14_TO_Q10_PROFILE);
            assert_eq!(region.input.numeric, NumericType::Wrap32);
            assert_eq!(region.output.numeric, NumericType::SignedFixedQ10);
            assert_eq!(region.input.shape, shape);
            assert_eq!(region.output.shape, shape);
            assert_eq!(region.input_fractional_bits, 14);
            assert_eq!(region.output_fractional_bits, 10);
            assert_eq!(region.divisor, 16);
            assert_eq!(region.rounding, FixedPointRounding::TiesToEven);
            assert_eq!(
                region.range_policy,
                Q14ToQ10RangePolicy::RejectOutsideSignedI16
            );
        }
        assert!(regions.iter().all(|region| {
            ![
                "layer.0.gate_proj",
                "layer.0.up_proj",
                "layer.0.down_proj",
                "output_head",
            ]
            .contains(&region.source_operation_id.as_str())
        }));
    }
}

#[test]
fn executes_centered_signed_values_against_core() {
    let plan = qwen_plan();
    let region = &lower_model_q14_to_q10_rescale_regions(&plan, DecoderMode::Prefill).unwrap()[0];
    let input: Vec<u32> = [
        0,
        8,
        -8,
        24,
        -24,
        40,
        -40,
        pllm_core::Q14_TO_Q10_INPUT_MIN,
        pllm_core::Q14_TO_Q10_INPUT_MAX,
    ]
    .into_iter()
    .chain(std::iter::repeat(0).take(39))
    .map(centered)
    .collect();
    let expected = rescale_q14_to_q10_centered_u32_tensor(&input).unwrap();
    assert_eq!(execute_q14_to_q10_rescale(region, &input), Ok(expected));
    assert_eq!(
        &execute_q14_to_q10_rescale(region, &input).unwrap()[..9],
        &[0, 0, 0, 2, -2, 2, -2, i16::MIN, i16::MAX]
    );
    for value in [
        pllm_core::Q14_TO_Q10_INPUT_MIN - 1,
        pllm_core::Q14_TO_Q10_INPUT_MAX + 1,
    ] {
        let mut out_of_domain = input.clone();
        out_of_domain[0] = centered(value);
        assert!(execute_q14_to_q10_rescale(region, &out_of_domain).is_err());
    }
    assert!(execute_q14_to_q10_rescale(region, &input[..47]).is_err());
    assert!(execute_q14_to_q10_rescale(region, &[]).is_err());
}

#[test]
fn digest_is_deterministic_and_forged_regions_are_rejected() {
    let plan = qwen_plan();
    let region = lower_model_q14_to_q10_rescale_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let digest = q14_to_q10_rescale_region_digest(&region);
    assert_eq!(digest, q14_to_q10_rescale_region_digest(&region));
    let input = vec![centered(0); 48];
    let mutations: Vec<fn(&mut Q14ToQ10RescaleRegion)> = vec![
        |region| region.schema_version = "forged.schema".into(),
        |region| region.operation_id = "forged.operation".into(),
        |region| region.source_operation_id = "layer.0.up_proj".into(),
        |region| region.target_operation_id = "layer.0.k_heads".into(),
        |region| region.target_input_index = 1,
        |region| region.numeric_profile = "forged.profile".into(),
        |region| region.input.numeric = NumericType::SignedFixedQ10,
        |region| region.output.numeric = NumericType::Wrap32,
        |region| region.input.shape = vec![2, 3, 4],
        |region| region.output.shape = vec![2, 3, 9],
        |region| region.input_fractional_bits = 15,
        |region| region.output_fractional_bits = 7,
        |region| region.divisor = 128,
        |region| {
            region.source_operation_id.clear();
        },
    ];
    for mutate in mutations {
        let mut forged = region.clone();
        mutate(&mut forged);
        assert_ne!(q14_to_q10_rescale_region_digest(&forged), digest);
        assert!(execute_q14_to_q10_rescale(&forged, &input).is_err());
    }
    assert!(define_q14_to_q10_rescale_region("bad id", "valid.id", 0, vec![1]).is_err());
    assert!(define_q14_to_q10_rescale_region("valid.id", "valid.id", 0, vec![]).is_err());
    assert!(define_q14_to_q10_rescale_region("valid.id", "valid.id", 0, vec![1, 0]).is_err());
}

#[test]
fn qwen3_fixture_lowers_despite_qk_norm_between_reshape_and_rope() {
    let plan = lower_model_json(
        include_bytes!("../../pllm-models/tests/fixtures/mini-coder-4b-c87892d-config.json"),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 1,
        },
    )
    .unwrap();
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        let regions = lower_model_q14_to_q10_rescale_regions(&plan, mode).unwrap();
        assert_eq!(regions.len(), 36 * 4);
        let first = regions
            .iter()
            .find(|region| region.source_operation_id == "layer.0.q_linear")
            .unwrap();
        assert_eq!(first.target_operation_id, "layer.0.q_heads");
        assert_eq!(first.target_input_index, 0);
        assert!(regions.iter().all(|region| {
            region.schema_version == Q14_TO_Q10_REGION_SCHEMA_VERSION
                && region.numeric_profile == Q14_TO_Q10_PROFILE
                && region.input.numeric == NumericType::Wrap32
                && region.output.numeric == NumericType::SignedFixedQ10
        }));
    }
}

#[test]
fn tampered_attention_topology_is_rejected() {
    let mut bypassed = qwen_plan();
    operation_mut(&mut bypassed.prefill, "layer.0.o_proj").inputs =
        vec!["layer.0.attention_values".into()];
    assert!(lower_model_q14_to_q10_rescale_regions(&bypassed, DecoderMode::Prefill).is_err());

    let mut layout = qwen_plan();
    operation_mut(&mut layout.prefill, "layer.0.q_heads")
        .attributes
        .as_object_mut()
        .unwrap()
        .insert("layout".into(), "batch_sequence_hidden".into());
    assert!(lower_model_q14_to_q10_rescale_regions(&layout, DecoderMode::Prefill).is_err());

    let mut wrong_index = qwen_plan();
    operation_mut(&mut wrong_index.prefill, "layer.0.attention_residual").inputs =
        vec!["layer.0.o_proj".into(), "layer.0.input_norm".into()];
    assert!(lower_model_q14_to_q10_rescale_regions(&wrong_index, DecoderMode::Prefill).is_err());

    let mut duplicated = qwen_plan();
    let extra = {
        let template = operation_mut(&mut duplicated.prefill, "layer.0.q_heads");
        pllm_models::ModelOperation {
            id: "layer.0.q_heads_duplicate".into(),
            operator: template.operator,
            layer: template.layer,
            state_kind: template.state_kind,
            inputs: template.inputs.clone(),
            output_shape: template.output_shape.clone(),
            attributes: template.attributes.clone(),
        }
    };
    duplicated.prefill.operations.push(extra);
    assert!(lower_model_q14_to_q10_rescale_regions(&duplicated, DecoderMode::Prefill).is_err());
}

#[test]
fn unsupported_model_families_reject_instead_of_returning_subsets() {
    let workload = DecoderWorkload {
        batch: 1,
        max_input_tokens: 2,
        max_new_tokens: 1,
    };
    let gemma = lower_model_json(
        include_bytes!("../../pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"),
        workload,
    )
    .unwrap();
    assert!(lower_model_q14_to_q10_rescale_regions(&gemma, DecoderMode::Prefill).is_err());
    assert!(lower_model_q14_to_q10_rescale_regions(&gemma, DecoderMode::Decode).is_err());

    let phi = lower_model_json(
        include_bytes!("../../pllm-models/tests/fixtures/Phi-4-mini-instruct-cfbefac-config.json"),
        workload,
    )
    .unwrap();
    assert!(lower_model_q14_to_q10_rescale_regions(&phi, DecoderMode::Prefill).is_err());
    assert!(lower_model_q14_to_q10_rescale_regions(&phi, DecoderMode::Decode).is_err());
}
