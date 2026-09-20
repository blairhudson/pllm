use pllm_compiler::{
    compile_dense_qwen_attention_block, compile_dense_qwen_mlp_block,
    compose_dense_qwen_layer_block, execute_dense_qwen_attention_decode,
    execute_dense_qwen_attention_prefill, execute_dense_qwen_layer_decode,
    execute_dense_qwen_layer_prefill, execute_dense_qwen_mlp_block,
    CompiledDenseQwenAttentionBlock, CompiledDenseQwenLayerBlock, CompiledDenseQwenMlpBlock,
    DenseQwenAttentionExecutionPolicy, DenseQwenAttentionResourcePolicy,
    DenseQwenAttentionWeightBytes, DenseQwenAttentionWeightManifest, DenseQwenAttentionWeights,
    DenseQwenLayerExecutionPolicy, DenseQwenMlpResourcePolicy, DenseQwenMlpWeightBytes,
    DenseQwenMlpWeightManifest, DenseQwenMlpWeights, ProvenancePrimitiveResourcePolicy,
    DENSE_QWEN_LAYER_CLEAR_PROFILE, DENSE_QWEN_LAYER_SCHEMA_VERSION,
};
use pllm_models::{lower_model_json, AppliedMethod, DecoderMode, DecoderPlan, DecoderWorkload};
use pllm_types::digest_bytes;

const CONFIG: &[u8] = br#"{
    "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
    "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
    "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
    "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
}"#;

fn qwen_plan(batch: u64, input_tokens: u64, new_tokens: u64) -> DecoderPlan {
    lower_model_json(
        CONFIG,
        DecoderWorkload {
            batch,
            max_input_tokens: input_tokens,
            max_new_tokens: new_tokens,
        },
    )
    .unwrap()
}

fn q4_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    (0..rows * cols)
        .map(|index| ((((index + seed) % 3) as i8) - 1) as u8)
        .collect()
}

struct AttentionWeights {
    norm: Vec<u8>,
    q: Vec<u8>,
    q_bias: Vec<u8>,
    k: Vec<u8>,
    k_bias: Vec<u8>,
    v: Vec<u8>,
    v_bias: Vec<u8>,
    o: Vec<u8>,
}

impl AttentionWeights {
    fn new() -> Self {
        Self {
            norm: [1024_i16; 8]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            q: q4_matrix(8, 8, 0),
            q_bias: vec![0; 16],
            k: q4_matrix(4, 8, 1),
            k_bias: vec![0; 8],
            v: q4_matrix(4, 8, 2),
            v_bias: vec![0; 8],
            o: q4_matrix(8, 8, 3),
        }
    }

    fn input(&self) -> DenseQwenAttentionWeights<'_> {
        DenseQwenAttentionWeights {
            norm_q10: attention_weight("model.layers.0.input_layernorm.weight", &self.norm),
            q_q4: attention_weight("model.layers.0.self_attn.q_proj.weight", &self.q),
            q_bias_q10: attention_weight("model.layers.0.self_attn.q_proj.bias", &self.q_bias),
            k_q4: attention_weight("model.layers.0.self_attn.k_proj.weight", &self.k),
            k_bias_q10: attention_weight("model.layers.0.self_attn.k_proj.bias", &self.k_bias),
            v_q4: attention_weight("model.layers.0.self_attn.v_proj.weight", &self.v),
            v_bias_q10: attention_weight("model.layers.0.self_attn.v_proj.bias", &self.v_bias),
            o_q4: attention_weight("model.layers.0.self_attn.o_proj.weight", &self.o),
        }
    }
}

fn attention_weight<'a>(weight_id: &'a str, bytes: &'a [u8]) -> DenseQwenAttentionWeightBytes<'a> {
    DenseQwenAttentionWeightBytes { weight_id, bytes }
}

struct MlpWeights {
    norm: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

impl MlpWeights {
    fn new() -> Self {
        Self {
            norm: [1024_i16; 8]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            gate: (0..128).map(|index| u8::from(index % 8 < 4)).collect(),
            up: (0..128).map(|index| u8::from(index % 8 < 4)).collect(),
            down: (0..128)
                .map(|index| (((index % 5) as i8) - 2).to_ne_bytes()[0])
                .collect(),
        }
    }

    fn over_domain_gate() -> Self {
        let mut weights = Self::new();
        weights.gate = (0..128).map(|index| u8::from(index % 8 == 0) * 7).collect();
        weights
    }

    fn input(&self) -> DenseQwenMlpWeights<'_> {
        DenseQwenMlpWeights {
            norm_q10: mlp_weight("model.layers.0.post_attention_layernorm.weight", &self.norm),
            gate_q4: mlp_weight("model.layers.0.mlp.gate_proj.weight", &self.gate),
            up_q4: mlp_weight("model.layers.0.mlp.up_proj.weight", &self.up),
            down_q3: mlp_weight("model.layers.0.mlp.down_proj.weight", &self.down),
        }
    }
}

fn mlp_weight<'a>(weight_id: &'a str, bytes: &'a [u8]) -> DenseQwenMlpWeightBytes<'a> {
    DenseQwenMlpWeightBytes { weight_id, bytes }
}

fn compile_attention(
    plan: &DecoderPlan,
    mode: DecoderMode,
    weights: &AttentionWeights,
) -> Result<CompiledDenseQwenAttentionBlock, String> {
    let input = weights.input();
    compile_dense_qwen_attention_block(
        plan,
        mode,
        0,
        DenseQwenAttentionWeightManifest::from_weights(input)?,
        DenseQwenAttentionResourcePolicy {
            max_total_weight_bytes: 1 << 20,
            max_activation_elements: 1 << 20,
        },
        input,
    )
}

fn compile_mlp(
    plan: &DecoderPlan,
    mode: DecoderMode,
    weights: &MlpWeights,
) -> Result<CompiledDenseQwenMlpBlock, String> {
    let input = weights.input();
    compile_dense_qwen_mlp_block(
        plan,
        mode,
        0,
        DenseQwenMlpWeightManifest::from_weights(input)?,
        DenseQwenMlpResourcePolicy {
            max_total_weight_bytes: 1 << 20,
            max_activation_elements: 1 << 20,
        },
        input,
    )
}

fn compose(
    plan: &DecoderPlan,
    mode: DecoderMode,
    attention_weights: &AttentionWeights,
    mlp_weights: &MlpWeights,
) -> Result<CompiledDenseQwenLayerBlock, String> {
    let attention = compile_attention(plan, mode, attention_weights)?;
    let mlp = compile_mlp(plan, mode, mlp_weights)?;
    compose_dense_qwen_layer_block(plan, attention, mlp)
}

fn attention_policy() -> DenseQwenAttentionExecutionPolicy {
    DenseQwenAttentionExecutionPolicy {
        threads: 1,
        simd: false,
        provenance: ProvenancePrimitiveResourcePolicy::new(
            1 << 20,
            1 << 24,
            1 << 24,
            1 << 16,
            1 << 12,
            1 << 20,
            32,
        )
        .unwrap(),
        scores: pllm_core::AttentionScoreQ20Policy::new(
            pllm_core::ATTENTION_Q20_MAX_SCORE_ELEMENTS,
            pllm_core::ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES,
        )
        .unwrap(),
        softmax: pllm_core::SoftmaxQ30Policy::new(
            pllm_core::SOFTMAX_Q30_MAX_ELEMENTS,
            pllm_core::SOFTMAX_Q30_MAX_ROW_LENGTH,
        )
        .unwrap(),
        values: pllm_core::AttentionValueQ10Policy::new(
            pllm_core::ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS,
            pllm_core::ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES,
        )
        .unwrap(),
    }
}

fn execution_policy() -> DenseQwenLayerExecutionPolicy {
    DenseQwenLayerExecutionPolicy {
        attention: attention_policy(),
        mlp_threads: 1,
        mlp_simd: false,
    }
}

fn wrap_i16(values: &[i16]) -> Vec<u32> {
    values.iter().map(|value| *value as i32 as u32).collect()
}

fn sequential_layer(
    plan: &DecoderPlan,
    attention_block: &CompiledDenseQwenAttentionBlock,
    mlp_block: &CompiledDenseQwenMlpBlock,
    input: &[i16],
    positions: &[u32],
    mask: &[bool],
    valid: &[usize],
) -> (Vec<i16>, pllm_compiler::DenseQwenAttentionState) {
    let (attention_output, attention_state) = execute_dense_qwen_attention_prefill(
        plan,
        attention_block,
        &wrap_i16(input),
        positions,
        mask,
        valid,
        attention_policy(),
    )
    .unwrap();
    let output = execute_dense_qwen_mlp_block(
        plan,
        mlp_block,
        &wrap_i16(attention_output.values()),
        1,
        false,
    )
    .unwrap();
    (output, attention_state)
}

#[test]
fn composes_prefill_and_decode_blocks_with_bound_composite() {
    let plan = qwen_plan(1, 2, 3);
    let attention_weights = AttentionWeights::new();
    let mlp_weights = MlpWeights::new();
    for (mode, query) in [(DecoderMode::Prefill, 2_u64), (DecoderMode::Decode, 1)] {
        let block = compose(&plan, mode, &attention_weights, &mlp_weights).unwrap();
        let composite = &block.composite;
        assert_eq!(composite.schema_version, DENSE_QWEN_LAYER_SCHEMA_VERSION);
        assert_eq!(composite.numeric_profile, DENSE_QWEN_LAYER_CLEAR_PROFILE);
        assert_eq!(composite.model_plan_digest, plan.digest());
        assert_eq!(composite.mode, mode);
        assert_eq!(composite.layer, 0);
        assert_eq!(composite.input_shape, vec![1, query, 8]);
        assert_eq!(composite.output_shape, vec![1, query, 8]);
        assert_eq!(composite.input_shape, composite.attention.input_shape);
        assert_eq!(composite.output_shape, composite.mlp.input_shape);
        assert_eq!(composite.attention.output_shape, composite.mlp.input_shape);
        assert_eq!(
            composite.attention.residual.operation_id,
            composite.mlp.norm.input_id
        );
        assert_eq!(
            composite.mlp.residual.input_ids[0],
            composite.mlp.norm.input_id
        );
        assert_eq!(
            composite.mlp.residual.input_ids[1],
            composite.mlp.down.operation_id
        );
        assert_eq!(
            composite.attention.residual.input_ids[0],
            composite.attention.norm.input_id
        );
        assert!(!composite.protected_execution);
        assert!(!composite.complete_decoder);
        let second = compose(&plan, mode, &attention_weights, &mlp_weights).unwrap();
        assert_eq!(block.binding_digest(), second.binding_digest());
    }
}

#[test]
fn prefill_layer_output_matches_sequential_attention_mlp_oracle() {
    let plan = qwen_plan(1, 2, 3);
    let attention_weights = AttentionWeights::new();
    let mlp_weights = MlpWeights::new();
    let block = compose(
        &plan,
        DecoderMode::Prefill,
        &attention_weights,
        &mlp_weights,
    )
    .unwrap();
    let input = [
        300_i16, -500, 200, -100, 64, 900, -750, 420, -210, 88, 512, -640, 130, -95, 260, -480,
    ];
    let (output, state) = execute_dense_qwen_layer_prefill(
        &plan,
        &block,
        &wrap_i16(&input),
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    )
    .unwrap();
    assert_eq!(output.shape(), [1, 2, 8]);
    assert!(state.is_usable());
    let attention_block =
        compile_attention(&plan, DecoderMode::Prefill, &attention_weights).unwrap();
    let mlp_block = compile_mlp(&plan, DecoderMode::Prefill, &mlp_weights).unwrap();
    let (expected, _oracle_state) = sequential_layer(
        &plan,
        &attention_block,
        &mlp_block,
        &input,
        &[0, 1],
        &[true, true],
        &[2],
    );
    assert_eq!(output.values(), expected.as_slice());
}

#[test]
fn decode_after_prefill_advances_layer_state_and_matches_oracle() {
    let plan = qwen_plan(1, 2, 3);
    let attention_weights = AttentionWeights::new();
    let mlp_weights = MlpWeights::new();
    let prefill_block = compose(
        &plan,
        DecoderMode::Prefill,
        &attention_weights,
        &mlp_weights,
    )
    .unwrap();
    let decode_block =
        compose(&plan, DecoderMode::Decode, &attention_weights, &mlp_weights).unwrap();
    let input = [
        300_i16, -500, 200, -100, 64, 900, -750, 420, -210, 88, 512, -640, 130, -95, 260, -480,
    ];
    let (_prefill_output, mut state) = execute_dense_qwen_layer_prefill(
        &plan,
        &prefill_block,
        &wrap_i16(&input),
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    )
    .unwrap();

    let attention_prefill =
        compile_attention(&plan, DecoderMode::Prefill, &attention_weights).unwrap();
    let attention_decode =
        compile_attention(&plan, DecoderMode::Decode, &attention_weights).unwrap();
    let mlp_decode = compile_mlp(&plan, DecoderMode::Decode, &mlp_weights).unwrap();
    let (_oracle_prefill, mut oracle_attention_state) = sequential_layer(
        &plan,
        &attention_prefill,
        &compile_mlp(&plan, DecoderMode::Prefill, &mlp_weights).unwrap(),
        &input,
        &[0, 1],
        &[true, true],
        &[2],
    );

    let first = [128_i16, -64, 300, 200, -420, 512, -256, 96];
    let output = execute_dense_qwen_layer_decode(
        &plan,
        &decode_block,
        &mut state,
        &wrap_i16(&first),
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .unwrap();
    assert_eq!(output.shape(), [1, 1, 8]);
    assert!(state.is_usable());
    let oracle_attention = execute_dense_qwen_attention_decode(
        &plan,
        &attention_decode,
        &mut oracle_attention_state,
        &wrap_i16(&first),
        &[2],
        &[true],
        &[3],
        attention_policy(),
    )
    .unwrap();
    let expected = execute_dense_qwen_mlp_block(
        &plan,
        &mlp_decode,
        &wrap_i16(oracle_attention.values()),
        1,
        false,
    )
    .unwrap();
    assert_eq!(output.values(), expected.as_slice());

    let second = [77_i16, -33, 410, -120, 88, -512, 640, -48];
    let output = execute_dense_qwen_layer_decode(
        &plan,
        &decode_block,
        &mut state,
        &wrap_i16(&second),
        &[3],
        &[true],
        &[4],
        execution_policy(),
    )
    .unwrap();
    assert_eq!(output.shape(), [1, 1, 8]);
    assert!(state.is_usable());
    let oracle_attention = execute_dense_qwen_attention_decode(
        &plan,
        &attention_decode,
        &mut oracle_attention_state,
        &wrap_i16(&second),
        &[3],
        &[true],
        &[4],
        attention_policy(),
    )
    .unwrap();
    let expected = execute_dense_qwen_mlp_block(
        &plan,
        &mlp_decode,
        &wrap_i16(oracle_attention.values()),
        1,
        false,
    )
    .unwrap();
    assert_eq!(output.values(), expected.as_slice());
}

#[test]
fn compose_rejects_mismatched_blocks_and_plans() {
    let plan = qwen_plan(1, 2, 3);
    let attention_weights = AttentionWeights::new();
    let mlp_weights = MlpWeights::new();

    let attention_prefill =
        compile_attention(&plan, DecoderMode::Prefill, &attention_weights).unwrap();
    let mlp_decode = compile_mlp(&plan, DecoderMode::Decode, &mlp_weights).unwrap();
    assert!(compose_dense_qwen_layer_block(&plan, attention_prefill, mlp_decode).is_err());

    let other_plan = qwen_plan(1, 3, 3);
    let attention = compile_attention(&plan, DecoderMode::Prefill, &attention_weights).unwrap();
    let mlp = compile_mlp(&other_plan, DecoderMode::Prefill, &mlp_weights).unwrap();
    assert!(compose_dense_qwen_layer_block(&plan, attention, mlp).is_err());

    let mut transformed = plan.clone();
    transformed.transformations.push(AppliedMethod {
        component: "cache".into(),
        implementation: "mpcache".into(),
        method_id: "mpcache".into(),
        input_digest: digest_bytes("test", b"input"),
        configuration_digest: digest_bytes("test", b"config"),
    });
    let attention = compile_attention(&plan, DecoderMode::Prefill, &attention_weights).unwrap();
    let mlp = compile_mlp(&plan, DecoderMode::Prefill, &mlp_weights).unwrap();
    assert!(compose_dense_qwen_layer_block(&transformed, attention, mlp).is_err());

    let mut wrong_family = plan.clone();
    wrong_family.model_family = "qwen3".into();
    let attention = compile_attention(&plan, DecoderMode::Prefill, &attention_weights).unwrap();
    let mlp = compile_mlp(&plan, DecoderMode::Prefill, &mlp_weights).unwrap();
    assert!(compose_dense_qwen_layer_block(&wrong_family, attention, mlp).is_err());

    let attention = compile_attention(&plan, DecoderMode::Prefill, &attention_weights).unwrap();
    let mut mlp = compile_mlp(&plan, DecoderMode::Prefill, &mlp_weights).unwrap();
    mlp.composite.input_shape = vec![1, 2, 4];
    assert!(compose_dense_qwen_layer_block(&plan, attention, mlp).is_err());

    let attention = compile_attention(&plan, DecoderMode::Prefill, &attention_weights).unwrap();
    let mut mlp = compile_mlp(&plan, DecoderMode::Prefill, &mlp_weights).unwrap();
    mlp.composite.norm.input_id = "layer.0.other".into();
    assert!(compose_dense_qwen_layer_block(&plan, attention, mlp).is_err());

    let mut block = compose(
        &plan,
        DecoderMode::Prefill,
        &attention_weights,
        &mlp_weights,
    )
    .unwrap();
    block.composite.layer = 7;
    assert!(execute_dense_qwen_layer_prefill(
        &plan,
        &block,
        &wrap_i16(&[0_i16; 16]),
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    )
    .is_err());
}

#[test]
fn execution_rejects_wrong_mode_plan_binding_and_inputs() {
    let plan = qwen_plan(1, 2, 3);
    let attention_weights = AttentionWeights::new();
    let mlp_weights = MlpWeights::new();
    let prefill_block = compose(
        &plan,
        DecoderMode::Prefill,
        &attention_weights,
        &mlp_weights,
    )
    .unwrap();
    let decode_block =
        compose(&plan, DecoderMode::Decode, &attention_weights, &mlp_weights).unwrap();
    let input = wrap_i16(&[
        300_i16, -500, 200, -100, 64, 900, -750, 420, -210, 88, 512, -640, 130, -95, 260, -480,
    ]);
    let (_output, mut state) = execute_dense_qwen_layer_prefill(
        &plan,
        &prefill_block,
        &input,
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    )
    .unwrap();
    let decode_input = wrap_i16(&[64_i16, -128, 256, -32, 512, -96, 192, -48]);

    assert!(execute_dense_qwen_layer_decode(
        &plan,
        &prefill_block,
        &mut state,
        &decode_input,
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
    assert!(state.is_usable());

    let other_plan = qwen_plan(1, 2, 4);
    assert!(execute_dense_qwen_layer_decode(
        &other_plan,
        &decode_block,
        &mut state,
        &decode_input,
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
    assert!(state.is_usable());

    let other_attention = AttentionWeights {
        norm: [2048_i16; 8]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        ..AttentionWeights::new()
    };
    let other_block = compose(&plan, DecoderMode::Decode, &other_attention, &mlp_weights).unwrap();
    assert!(execute_dense_qwen_layer_decode(
        &plan,
        &other_block,
        &mut state,
        &decode_input,
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
    assert!(state.is_usable());

    assert!(execute_dense_qwen_layer_decode(
        &plan,
        &decode_block,
        &mut state,
        &decode_input[..7],
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
    assert!(state.is_usable());

    assert!(execute_dense_qwen_layer_decode(
        &plan,
        &decode_block,
        &mut state,
        &decode_input,
        &[2, 2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
    assert!(state.is_usable());

    assert!(execute_dense_qwen_layer_decode(
        &plan,
        &decode_block,
        &mut state,
        &decode_input,
        &[2],
        &[true, true],
        &[3],
        execution_policy(),
    )
    .is_err());
    assert!(state.is_usable());

    assert!(execute_dense_qwen_layer_decode(
        &plan,
        &decode_block,
        &mut state,
        &decode_input,
        &[2],
        &[true],
        &[3, 3],
        execution_policy(),
    )
    .is_err());
    assert!(state.is_usable());
}

#[test]
fn attention_post_append_failure_marks_wrapper_unusable() {
    let plan = qwen_plan(1, 2, 3);
    let attention_weights = AttentionWeights::new();
    let mlp_weights = MlpWeights::new();
    let prefill_block = compose(
        &plan,
        DecoderMode::Prefill,
        &attention_weights,
        &mlp_weights,
    )
    .unwrap();
    let decode_block =
        compose(&plan, DecoderMode::Decode, &attention_weights, &mlp_weights).unwrap();
    let input = wrap_i16(&[
        300_i16, -500, 200, -100, 64, 900, -750, 420, -210, 88, 512, -640, 130, -95, 260, -480,
    ]);
    let (_output, mut state) = execute_dense_qwen_layer_prefill(
        &plan,
        &prefill_block,
        &input,
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    )
    .unwrap();
    let mut policy = execution_policy();
    policy.attention.softmax = pllm_core::SoftmaxQ30Policy::new(1, 1).unwrap();
    assert!(execute_dense_qwen_layer_decode(
        &plan,
        &decode_block,
        &mut state,
        &wrap_i16(&[64_i16; 8]),
        &[2],
        &[true],
        &[3],
        policy,
    )
    .is_err());
    assert!(!state.is_usable());
    assert!(execute_dense_qwen_layer_decode(
        &plan,
        &decode_block,
        &mut state,
        &wrap_i16(&[64_i16; 8]),
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
}

#[test]
fn mlp_failure_after_attention_append_marks_wrapper_unusable() {
    let plan = qwen_plan(1, 2, 3);
    let attention_weights = AttentionWeights::new();
    let mlp_weights = MlpWeights::over_domain_gate();
    let prefill_block = compose(
        &plan,
        DecoderMode::Prefill,
        &attention_weights,
        &mlp_weights,
    )
    .unwrap();
    let decode_block =
        compose(&plan, DecoderMode::Decode, &attention_weights, &mlp_weights).unwrap();
    let input = wrap_i16(&[
        300_i16, -500, 200, -100, 64, 900, -750, 420, -210, 88, 512, -640, 130, -95, 260, -480,
    ]);
    let (_output, mut state) = execute_dense_qwen_layer_prefill(
        &plan,
        &prefill_block,
        &input,
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    )
    .unwrap();
    assert!(state.is_usable());
    let skewed = wrap_i16(&[30_000_i16, 0, 0, 0, 0, 0, 0, 0]);
    assert!(execute_dense_qwen_layer_decode(
        &plan,
        &decode_block,
        &mut state,
        &skewed,
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
    assert!(!state.is_usable());
    assert!(execute_dense_qwen_layer_decode(
        &plan,
        &decode_block,
        &mut state,
        &wrap_i16(&[64_i16; 8]),
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
}
