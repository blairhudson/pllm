use pllm_compiler::{
    compile_dense_qwen_mlp_block as compile_block, execute_dense_qwen_mlp_block as execute_block,
    CompiledDenseQwenMlpBlock, DenseQwenMlpElementType, DenseQwenMlpLayout,
    DenseQwenMlpRangePolicy, DenseQwenMlpResidualOutputRangePolicy, DenseQwenMlpResidualStorage,
    DenseQwenMlpResourcePolicy, DenseQwenMlpWeightBytes, DenseQwenMlpWeightManifest,
    DenseQwenMlpWeights, FixedPointRounding,
    GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
    GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
};
use pllm_models::{lower_model_json, DecoderMode, DecoderPlan, DecoderWorkload};
use pllm_types::digest_bytes;

struct TestWeights {
    norm: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

impl TestWeights {
    fn new(norm: &[i16], gate: &[i8], up: &[i8], down: &[i8]) -> Self {
        Self {
            norm: norm.iter().flat_map(|value| value.to_le_bytes()).collect(),
            gate: gate.iter().map(|value| value.to_ne_bytes()[0]).collect(),
            up: up.iter().map(|value| value.to_ne_bytes()[0]).collect(),
            down: down.iter().map(|value| value.to_ne_bytes()[0]).collect(),
        }
    }

    fn input(&self) -> DenseQwenMlpWeights<'_> {
        DenseQwenMlpWeights {
            norm_q10: weight("model.layers.0.post_attention_layernorm.weight", &self.norm),
            gate_q4: weight("model.layers.0.mlp.gate_proj.weight", &self.gate),
            up_q4: weight("model.layers.0.mlp.up_proj.weight", &self.up),
            down_q3: weight("model.layers.0.mlp.down_proj.weight", &self.down),
        }
    }
}

fn weight<'a>(weight_id: &'a str, bytes: &'a [u8]) -> DenseQwenMlpWeightBytes<'a> {
    DenseQwenMlpWeightBytes { weight_id, bytes }
}

fn policy() -> DenseQwenMlpResourcePolicy {
    DenseQwenMlpResourcePolicy {
        max_total_weight_bytes: 10_000,
        max_activation_elements: 10_000,
    }
}

fn compile(
    plan: &DecoderPlan,
    mode: DecoderMode,
    layer: u64,
    weights: &TestWeights,
) -> Result<CompiledDenseQwenMlpBlock, String> {
    let input = weights.input();
    compile_block(
        plan,
        mode,
        layer,
        DenseQwenMlpWeightManifest::from_weights(input)?,
        policy(),
        input,
    )
}

fn qwen2_plan(hidden: usize, intermediate: usize, prefill_tokens: u64) -> DecoderPlan {
    let config = format!(
        r#"{{
            "model_type":"qwen2","hidden_size":{hidden},"intermediate_size":{intermediate},
            "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
            "vocab_size":8,"max_position_embeddings":16,"hidden_act":"silu",
            "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
        }}"#
    );
    lower_model_json(
        config.as_bytes(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: prefill_tokens,
            max_new_tokens: 1,
        },
    )
    .unwrap()
}

fn qwen3_plan() -> DecoderPlan {
    lower_model_json(
        br#"{
            "model_type":"qwen3","hidden_size":2,"intermediate_size":3,
            "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
            "vocab_size":8,"max_position_embeddings":16,"head_dim":2,"hidden_act":"silu",
            "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true,
            "attention_bias":false,"attention_dropout":0.0,"rope_scaling":null,
            "sliding_window":null,"use_sliding_window":false,"use_cache":true,
            "max_window_layers":1,"layer_types":["full_attention"]
        }"#,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 1,
        },
    )
    .unwrap()
}

fn ordinary_weights() -> TestWeights {
    TestWeights::new(
        &[1024, 1024],
        &[4, -2, -3, 5, 2, 2],
        &[3, 1, 2, -4, -2, 3],
        &[2, -1, 3, -2, 2, 1],
    )
}

fn wrap(values: &[i16]) -> Vec<u32> {
    values
        .iter()
        .map(|value| i32::from(*value) as u32)
        .collect()
}

fn manual(
    input: &[i16],
    norm: &[i16],
    gate: &[i8],
    up: &[i8],
    down: &[i8],
    intermediate: usize,
) -> Vec<i16> {
    let hidden = norm.len();
    let rows = input.len() / hidden;
    let normalized = pllm_core::rms_norm_q10_direct(input, norm).unwrap();
    let gate = scalar_linear(gate, intermediate, hidden, &normalized, rows)
        .into_iter()
        .map(|value| pllm_core::rescale_q14_to_q7(value).unwrap())
        .collect::<Vec<_>>();
    let up = scalar_linear(up, intermediate, hidden, &normalized, rows)
        .into_iter()
        .map(|value| pllm_core::rescale_q14_to_q7(value).unwrap())
        .collect::<Vec<_>>();
    let multiplied = gate
        .iter()
        .zip(up)
        .map(|(gate, up)| pllm_core::gated_multiply_q7(*gate, up).unwrap())
        .map(i32::from)
        .collect::<Vec<_>>();
    let down = scalar_linear(down, hidden, intermediate, &multiplied, rows);
    input
        .iter()
        .zip(down)
        .map(|(residual, down)| i16::try_from(i32::from(*residual) + down).unwrap())
        .collect()
}

fn scalar_linear(
    weights: &[i8],
    outputs: usize,
    inputs: usize,
    values: &[i32],
    rows: usize,
) -> Vec<i32> {
    let mut result = Vec::with_capacity(rows * outputs);
    for input in values.chunks_exact(inputs) {
        for weights in weights.chunks_exact(inputs).take(outputs) {
            result.push(
                weights
                    .iter()
                    .zip(input)
                    .map(|(weight, value)| i32::from(*weight) * *value)
                    .sum(),
            );
        }
    }
    result
}

#[test]
fn decode_b1_t1_matches_manual_and_owns_weight_bytes() {
    let plan = qwen2_plan(2, 3, 1);
    let mut weights = ordinary_weights();
    let original = ordinary_weights();
    let block = compile(&plan, DecoderMode::Decode, 0, &weights).unwrap();
    weights.norm.fill(0);
    weights.gate.fill(0);
    weights.up.fill(0);
    weights.down.fill(0);
    let input = [300, -500];
    let output = execute_block(&plan, &block, &wrap(&input), 1, false).unwrap();
    assert_eq!(
        output,
        manual(
            &input,
            &[1024, 1024],
            &original
                .gate
                .iter()
                .map(|value| *value as i8)
                .collect::<Vec<_>>(),
            &original
                .up
                .iter()
                .map(|value| *value as i8)
                .collect::<Vec<_>>(),
            &original
                .down
                .iter()
                .map(|value| *value as i8)
                .collect::<Vec<_>>(),
            3,
        )
    );
    let composite = &block.composite;
    assert_eq!(composite.gate_weight.shape, [3, 2]);
    assert_eq!(composite.down_weight.shape, [2, 3]);
    assert_eq!(
        composite.norm_weight.element_type,
        DenseQwenMlpElementType::SignedI16
    );
    assert_eq!(
        composite.gate_weight.layout,
        DenseQwenMlpLayout::RowMajorOutputInput
    );
    assert_eq!(
        composite.gate_weight.rounding,
        FixedPointRounding::TiesToEven
    );
    assert_eq!(
        composite.gate_weight.range_policy,
        DenseQwenMlpRangePolicy::RejectNoSaturation
    );
    assert_eq!(composite.residual_refinement.output_fractional_bits, 10);
    assert_eq!(
        composite.residual_refinement.storage,
        DenseQwenMlpResidualStorage::CenteredWrap32
    );
    assert_eq!(
        composite.residual_refinement.output_range_policy,
        DenseQwenMlpResidualOutputRangePolicy::RejectOutsideSignedI16ForNextRmsNorm
    );
    assert_eq!(composite.resource_policy, policy());
    assert_eq!(
        composite.gated_multiply.method_component_id,
        GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID
    );
    assert_eq!(
        composite.gated_multiply.schedule_component_id,
        GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID
    );
    assert!(!composite.protected_execution);
    assert!(!composite.complete_decoder);
}

#[test]
fn prefill_more_than_four_intermediates_matches_manual_and_qwen3_compiles() {
    let plan = qwen2_plan(2, 3, 2);
    let weights = ordinary_weights();
    let block = compile(&plan, DecoderMode::Prefill, 0, &weights).unwrap();
    assert_eq!(block.composite.gated_multiply.max_tensor_elements, 6);
    let input = [300, -500, -700, 200];
    let output = execute_block(&plan, &block, &wrap(&input), 2, true).unwrap();
    assert_eq!(
        output,
        manual(
            &input,
            &[1024, 1024],
            &[4, -2, -3, 5, 2, 2],
            &[3, 1, 2, -4, -2, 3],
            &[2, -1, 3, -2, 2, 1],
            3,
        )
    );
    assert_eq!(output.len(), 4);

    compile(&qwen3_plan(), DecoderMode::Decode, 0, &weights).unwrap();
}

#[test]
fn rejects_wrong_layer_weights_dimensions_and_tampering() {
    let plan = qwen2_plan(2, 3, 1);
    let weights = ordinary_weights();
    assert!(compile(&plan, DecoderMode::Decode, 1, &weights).is_err());

    let wrong_id = DenseQwenMlpWeights {
        gate_q4: weight("wrong.weight", &weights.gate),
        ..weights.input()
    };
    let wrong_manifest = DenseQwenMlpWeightManifest::from_weights(wrong_id).unwrap();
    assert!(compile_block(
        &plan,
        DecoderMode::Decode,
        0,
        wrong_manifest,
        policy(),
        wrong_id,
    )
    .is_err());
    let short = DenseQwenMlpWeights {
        down_q3: weight("model.layers.0.mlp.down_proj.weight", &weights.down[..5]),
        ..weights.input()
    };
    let short_manifest = DenseQwenMlpWeightManifest::from_weights(short).unwrap();
    assert!(compile_block(
        &plan,
        DecoderMode::Decode,
        0,
        short_manifest,
        policy(),
        short,
    )
    .is_err());

    let block = compile(&plan, DecoderMode::Decode, 0, &weights).unwrap();
    let mut tampered = block.clone();
    tampered.composite.gate.operation_id = "tampered".into();
    assert!(execute_block(&plan, &tampered, &wrap(&[1, 2]), 1, false).is_err());
    let mut wrong_scale = block;
    wrong_scale.composite.gate_weight.fractional_bits = 5;
    assert!(execute_block(&plan, &wrong_scale, &wrap(&[1, 2]), 1, false).is_err());
}

#[test]
fn rejects_bias_family_and_wrong_semantic_graph() {
    let weights = ordinary_weights();
    let mut biased = qwen2_plan(2, 3, 1);
    biased
        .decode
        .operations
        .iter_mut()
        .find(|operation| operation.id == "layer.0.gate_proj")
        .unwrap()
        .attributes["bias"] = "model.layers.0.mlp.gate_proj.bias".into();
    assert!(compile(&biased, DecoderMode::Decode, 0, &weights)
        .unwrap_err()
        .contains("bias"));

    let mut family = qwen2_plan(2, 3, 1);
    family.model_family = "not_qwen".into();
    assert!(compile(&family, DecoderMode::Decode, 0, &weights).is_err());

    let mut graph = qwen2_plan(2, 3, 1);
    graph
        .decode
        .operations
        .iter_mut()
        .find(|operation| operation.id == "layer.0.up_proj")
        .unwrap()
        .inputs = vec!["layer.0.attention_residual".into()];
    assert!(compile(&graph, DecoderMode::Decode, 0, &weights).is_err());
}

#[test]
fn rejects_q14_range_and_linear_overflow_before_wrap32() {
    let plan = qwen2_plan(2, 1, 1);
    let q14 = TestWeights::new(&[1024, 1024], &[127, 127], &[0, 0], &[0, 0]);
    let block = compile(&plan, DecoderMode::Decode, 0, &q14).unwrap();
    let error = execute_block(&plan, &block, &wrap(&[1024, 1024]), 1, false).unwrap_err();
    assert!(error.contains("signed Q14 input"));

    let plan = qwen2_plan(528, 1, 1);
    let overflow = TestWeights::new(
        &vec![i16::MAX; 528],
        &vec![i8::MIN; 528],
        &vec![0; 528],
        &vec![0; 528],
    );
    let block = compile(&plan, DecoderMode::Decode, 0, &overflow).unwrap();
    let error = execute_block(&plan, &block, &wrap(&vec![1024; 528]), 1, false).unwrap_err();
    assert!(error.contains("bound exceeds i32 before wrap32"));
}

#[test]
fn residual_rejects_out_of_i16_and_normal_output_is_next_block_i16() {
    let plan = qwen2_plan(2, 1, 1);
    let overflowing = TestWeights::new(&[1024, 1024], &[8, 0], &[8, 0], &[127, 127]);
    let block = compile(&plan, DecoderMode::Decode, 0, &overflowing).unwrap();
    let error = execute_block(&plan, &block, &wrap(&[32_000, 32_000]), 1, false).unwrap_err();
    assert!(error.contains("residual Q10 result is outside i16"));

    let weights = TestWeights::new(&[1024, 1024], &[1, 0], &[1, 0], &[1, 1]);
    let block = compile(&plan, DecoderMode::Decode, 0, &weights).unwrap();
    let next_block_input: Vec<i16> =
        execute_block(&plan, &block, &wrap(&[100, -100]), 1, false).unwrap();
    assert_eq!(next_block_input.len(), 2);
}

#[test]
fn i8_min_weight_matrix_path_succeeds() {
    let plan = qwen2_plan(2, 1, 1);
    let weights = TestWeights::new(&[64, 64], &[i8::MIN, 0], &[0, 0], &[0, 0]);
    let block = compile(&plan, DecoderMode::Decode, 0, &weights).unwrap();
    assert_eq!(
        execute_block(&plan, &block, &wrap(&[100, 0]), 1, false).unwrap(),
        [100, 0]
    );
}

#[test]
fn centered_i16_min_input_survives_zero_mlp() {
    let plan = qwen2_plan(2, 1, 1);
    let weights = TestWeights::new(&[0, 0], &[0, 0], &[0, 0], &[0, 0]);
    let block = compile(&plan, DecoderMode::Decode, 0, &weights).unwrap();
    assert_eq!(
        execute_block(&plan, &block, &wrap(&[i16::MIN, 0]), 1, false).unwrap(),
        [i16::MIN, 0]
    );
}

#[test]
fn manifest_digest_changes_on_one_byte_and_stale_manifest_is_rejected() {
    let plan = qwen2_plan(2, 3, 1);
    let original = ordinary_weights();
    let original_manifest = DenseQwenMlpWeightManifest::from_weights(original.input()).unwrap();
    let mut changed = ordinary_weights();
    changed.gate[0] ^= 1;
    let changed_manifest = DenseQwenMlpWeightManifest::from_weights(changed.input()).unwrap();
    assert_ne!(original_manifest.digest(), changed_manifest.digest());
    assert!(compile_block(
        &plan,
        DecoderMode::Decode,
        0,
        original_manifest,
        policy(),
        changed.input(),
    )
    .unwrap_err()
    .contains("manifest entry"));
}

#[test]
fn altered_plan_is_rejected_at_execution() {
    let plan = qwen2_plan(2, 3, 1);
    let weights = ordinary_weights();
    let block = compile(&plan, DecoderMode::Decode, 0, &weights).unwrap();
    let mut altered = plan.clone();
    altered.config_digest = digest_bytes("test.altered.config.v1", b"altered");
    assert!(execute_block(&altered, &block, &wrap(&[1, 2]), 1, false).is_err());
}

#[test]
fn resource_policy_rejects_before_weight_copy_or_activation_execution() {
    let decode = qwen2_plan(2, 3, 1);
    let weights = ordinary_weights();
    let input = weights.input();
    let manifest = DenseQwenMlpWeightManifest::from_weights(input).unwrap();
    let exact_bytes = input.norm_q10.bytes.len()
        + input.gate_q4.bytes.len()
        + input.up_q4.bytes.len()
        + input.down_q3.bytes.len();
    let error = compile_block(
        &decode,
        DecoderMode::Decode,
        0,
        manifest,
        DenseQwenMlpResourcePolicy {
            max_total_weight_bytes: u64::try_from(exact_bytes - 1).unwrap(),
            max_activation_elements: 10,
        },
        input,
    )
    .unwrap_err();
    assert!(error.contains("weight bytes exceed resource policy"));

    let prefill = qwen2_plan(2, 3, 2);
    let input = weights.input();
    let manifest = DenseQwenMlpWeightManifest::from_weights(input).unwrap();
    let error = compile_block(
        &prefill,
        DecoderMode::Prefill,
        0,
        manifest,
        DenseQwenMlpResourcePolicy {
            max_total_weight_bytes: 100,
            max_activation_elements: 5,
        },
        input,
    )
    .unwrap_err();
    assert!(error.contains("activation elements exceed resource policy"));

    let input = weights.input();
    let manifest = DenseQwenMlpWeightManifest::from_weights(input).unwrap();
    let error = compile_block(
        &decode,
        DecoderMode::Decode,
        0,
        manifest,
        DenseQwenMlpResourcePolicy {
            max_total_weight_bytes: u64::MAX,
            max_activation_elements: 1,
        },
        input,
    )
    .unwrap_err();
    assert!(error.contains("hard caps"));
}
