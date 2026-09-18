use pllm_compiler::{
    compile_dense_qwen_attention_block, execute_dense_qwen_attention_decode,
    execute_dense_qwen_attention_prefill, execute_model_linear, lower_model_linear_regions,
    CompiledDenseQwenAttentionBlock, DenseQwenAttentionElementType,
    DenseQwenAttentionExecutionPolicy, DenseQwenAttentionLayout, DenseQwenAttentionRangePolicy,
    DenseQwenAttentionResourcePolicy, DenseQwenAttentionWeightBytes,
    DenseQwenAttentionWeightManifest, DenseQwenAttentionWeights, FixedPointRounding,
    KvCacheAppendMode, NumericType, ProvenancePrimitiveResourcePolicy,
    DENSE_QWEN_ATTENTION_BLOCK_SCHEMA_VERSION, DENSE_QWEN_ATTENTION_CLEAR_PROFILE,
};
use pllm_models::{
    lower_model_json, AppliedMethod, DecoderMode, DecoderPlan, DecoderWorkload, StateKind,
    StateTensor,
};
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

struct TestWeights {
    norm: Vec<u8>,
    norm_i16: Vec<i16>,
    q: Vec<u8>,
    k: Vec<u8>,
    v: Vec<u8>,
    o: Vec<u8>,
}

impl TestWeights {
    fn new() -> Self {
        let norm_i16 = vec![1024_i16; 8];
        Self {
            norm: norm_i16
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            norm_i16,
            q: q4_matrix(8, 8, 0),
            k: q4_matrix(4, 8, 1),
            v: q4_matrix(4, 8, 2),
            o: q4_matrix(8, 8, 3),
        }
    }

    fn input(&self) -> DenseQwenAttentionWeights<'_> {
        DenseQwenAttentionWeights {
            norm_q10: weight("model.layers.0.input_layernorm.weight", &self.norm),
            q_q4: weight("model.layers.0.self_attn.q_proj.weight", &self.q),
            k_q4: weight("model.layers.0.self_attn.k_proj.weight", &self.k),
            v_q4: weight("model.layers.0.self_attn.v_proj.weight", &self.v),
            o_q4: weight("model.layers.0.self_attn.o_proj.weight", &self.o),
        }
    }
}

fn weight<'a>(weight_id: &'a str, bytes: &'a [u8]) -> DenseQwenAttentionWeightBytes<'a> {
    DenseQwenAttentionWeightBytes { weight_id, bytes }
}

fn resource_policy() -> DenseQwenAttentionResourcePolicy {
    DenseQwenAttentionResourcePolicy {
        max_total_weight_bytes: 1 << 20,
        max_activation_elements: 1 << 20,
    }
}

fn provenance_policy() -> ProvenancePrimitiveResourcePolicy {
    ProvenancePrimitiveResourcePolicy::new(1 << 20, 1 << 24, 1 << 24, 1 << 16, 1 << 12, 1 << 20, 32)
        .unwrap()
}

fn execution_policy() -> DenseQwenAttentionExecutionPolicy {
    DenseQwenAttentionExecutionPolicy {
        threads: 1,
        simd: false,
        provenance: provenance_policy(),
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

fn compile(
    plan: &DecoderPlan,
    mode: DecoderMode,
    weights: &TestWeights,
) -> Result<CompiledDenseQwenAttentionBlock, String> {
    let input = weights.input();
    compile_dense_qwen_attention_block(
        plan,
        mode,
        0,
        DenseQwenAttentionWeightManifest::from_weights(input)?,
        resource_policy(),
        input,
    )
}

fn wrap_i16(values: &[i16]) -> Vec<u32> {
    values.iter().map(|value| *value as i32 as u32).collect()
}

fn split_heads(input: &[i16], batch: usize, sequence: usize, heads: usize, dim: usize) -> Vec<i16> {
    let mut output = vec![0_i16; input.len()];
    for b in 0..batch {
        for s in 0..sequence {
            for h in 0..heads {
                for d in 0..dim {
                    output[((b * heads + h) * sequence + s) * dim + d] =
                        input[((b * sequence + s) * heads + h) * dim + d];
                }
            }
        }
    }
    output
}

fn merge_heads(input: &[i16], batch: usize, sequence: usize, heads: usize, dim: usize) -> Vec<i16> {
    let mut output = vec![0_i16; input.len()];
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..sequence {
                for d in 0..dim {
                    output[((b * sequence + s) * heads + h) * dim + d] =
                        input[((b * heads + h) * sequence + s) * dim + d];
                }
            }
        }
    }
    output
}

fn oracle_linear(
    weights: &[u8],
    output_width: usize,
    input_width: usize,
    input: &[i32],
) -> Vec<i16> {
    let wrap: Vec<u32> = input.iter().map(|value| *value as u32).collect();
    let matrix = pllm_core::Matrix::new(weights, output_width, input_width).unwrap();
    let executor = pllm_core::Executor::new(1, false).unwrap();
    let rows = input.len() / input_width;
    let q14 = matrix.wrap32(&executor, &wrap, rows).unwrap();
    pllm_core::rescale_q14_to_q10_centered_u32_tensor(&q14).unwrap()
}

fn flatten_cache(cache: &pllm_core::BoundedKvCacheQ10, maximum_sequence: usize) -> Vec<i16> {
    let [batch, heads, _, head_dim] = cache.shape();
    let mut flat = vec![0_i16; batch * heads * maximum_sequence * head_dim];
    for b in 0..batch {
        for h in 0..heads {
            let prefix = cache.visible_prefix(b, h, maximum_sequence).unwrap();
            let offset = (b * heads + h) * maximum_sequence * head_dim;
            flat[offset..offset + prefix.len()].copy_from_slice(prefix);
        }
    }
    flat
}

struct OracleCaches {
    key: pllm_core::BoundedKvCacheQ10,
    value: pllm_core::BoundedKvCacheQ10,
}

fn oracle_forward(
    weights: &TestWeights,
    block: &CompiledDenseQwenAttentionBlock,
    input: &[i16],
    positions: &[u32],
    attention_mask: &[bool],
    valid_lengths: &[usize],
    caches: &mut Option<OracleCaches>,
) -> Vec<i16> {
    let composite = &block.composite;
    let [batch, query, hidden] = composite.input_shape.as_slice() else {
        panic!("rank three input")
    };
    let (batch, query, hidden) = (*batch as usize, *query as usize, *hidden as usize);
    let heads = composite.q_shape[1] as usize;
    let kv_heads = composite.kv_shape[1] as usize;
    let head_dim = composite.q_shape[3] as usize;
    let maximum_sequence = composite.key_view.maximum_sequence as usize;
    let capacity = composite.key_append.state_shape[2] as usize;
    let normalized = pllm_core::rms_norm_q10_direct(input, &weights.norm_i16).unwrap();
    let normalized_i32: Vec<i32> = normalized;
    let q = oracle_linear(&weights.q, heads * head_dim, hidden, &normalized_i32);
    let k = oracle_linear(&weights.k, kv_heads * head_dim, hidden, &normalized_i32);
    let v = oracle_linear(&weights.v, kv_heads * head_dim, hidden, &normalized_i32);
    let q_heads = split_heads(&q, batch, query, heads, head_dim);
    let k_heads = split_heads(&k, batch, query, kv_heads, head_dim);
    let v_heads = split_heads(&v, batch, query, kv_heads, head_dim);
    let rope_config = pllm_core::RopeQ10Config {
        theta: composite.q_rope.theta.parse().unwrap(),
        rotary_dimensions: usize::try_from(composite.q_rope.rotary_dimensions).unwrap(),
        maximum_position: composite.q_rope.maximum_position,
    };
    let q_rope = pllm_core::rope_q10(
        &q_heads,
        positions,
        [batch, heads, query, head_dim],
        rope_config,
    )
    .unwrap();
    let k_rope = pllm_core::rope_q10(
        &k_heads,
        positions,
        [batch, kv_heads, query, head_dim],
        rope_config,
    )
    .unwrap();
    match caches {
        None => {
            *caches = Some(OracleCaches {
                key: pllm_core::BoundedKvCacheQ10::initialize(
                    batch,
                    kv_heads,
                    capacity,
                    head_dim,
                    &k_rope,
                    query,
                    positions,
                    attention_mask,
                    valid_lengths,
                )
                .unwrap(),
                value: pllm_core::BoundedKvCacheQ10::initialize(
                    batch,
                    kv_heads,
                    capacity,
                    head_dim,
                    &v_heads,
                    query,
                    positions,
                    attention_mask,
                    valid_lengths,
                )
                .unwrap(),
            });
        }
        Some(caches) => {
            caches
                .key
                .append(&k_rope, query, positions, attention_mask, valid_lengths)
                .unwrap();
            caches
                .value
                .append(&v_heads, query, positions, attention_mask, valid_lengths)
                .unwrap();
        }
    }
    let caches = caches.as_ref().unwrap();
    let key_flat = flatten_cache(&caches.key, maximum_sequence);
    let value_flat = flatten_cache(&caches.value, maximum_sequence);
    let query_mask: Vec<u8> = attention_mask
        .iter()
        .map(|value| u8::from(*value))
        .collect();
    let policy = execution_policy();
    let scores = pllm_core::attention_scores_q20(
        &q_rope,
        [batch, heads, query, head_dim],
        &key_flat,
        [batch, kv_heads, maximum_sequence, head_dim],
        positions,
        &query_mask,
        caches.key.valid_lengths(),
        policy.scores,
    )
    .unwrap();
    let probabilities = pllm_core::softmax_q20_to_q30(
        scores.scores(),
        scores.allowed(),
        [batch, heads, query, maximum_sequence],
        policy.softmax,
    )
    .unwrap();
    let values = pllm_core::attention_values_q10(
        probabilities.probabilities(),
        [batch, heads, query, maximum_sequence],
        &value_flat,
        &[batch, kv_heads, maximum_sequence, head_dim],
        policy.values,
    )
    .unwrap();
    let hidden_flat = merge_heads(values.values(), batch, query, heads, head_dim);
    let hidden_i32: Vec<i32> = hidden_flat.iter().map(|value| i32::from(*value)).collect();
    let o = oracle_linear(&weights.o, hidden, heads * head_dim, &hidden_i32);
    input
        .iter()
        .zip(&o)
        .map(|(residual, projected)| {
            i16::try_from(i32::from(*residual) + i32::from(*projected)).unwrap()
        })
        .collect()
}

#[test]
fn compiles_prefill_and_decode_composites_with_traced_topology() {
    let plan = qwen_plan(1, 2, 2);
    let weights = TestWeights::new();
    for (mode, query) in [(DecoderMode::Prefill, 2_u64), (DecoderMode::Decode, 1)] {
        let block = compile(&plan, mode, &weights).unwrap();
        let composite = &block.composite;
        assert_eq!(
            composite.schema_version,
            DENSE_QWEN_ATTENTION_BLOCK_SCHEMA_VERSION
        );
        assert_eq!(
            composite.numeric_profile,
            DENSE_QWEN_ATTENTION_CLEAR_PROFILE
        );
        assert_eq!(composite.mode, mode);
        assert_eq!(composite.layer, 0);
        assert_eq!(composite.input_shape, vec![1, query, 8]);
        assert_eq!(composite.q_shape, vec![1, 2, query, 4]);
        assert_eq!(composite.kv_shape, vec![1, 1, query, 4]);
        assert_eq!(composite.output_shape, vec![1, query, 8]);
        assert_eq!(composite.norm.operation_id, "layer.0.input_norm");
        assert_eq!(
            composite.norm.weight_id,
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(composite.q.operation_id, "layer.0.q_linear");
        assert_eq!(composite.k.operation_id, "layer.0.k_linear");
        assert_eq!(composite.v.operation_id, "layer.0.v_linear");
        assert_eq!(composite.o.operation_id, "layer.0.o_proj");
        assert_eq!(
            composite.q.weight_id,
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            composite.o.weight_id,
            "model.layers.0.self_attn.o_proj.weight"
        );
        for (rescale, source, target, index) in [
            (
                &composite.q_rescale,
                "layer.0.q_linear",
                "layer.0.q_heads",
                0,
            ),
            (
                &composite.k_rescale,
                "layer.0.k_linear",
                "layer.0.k_heads",
                0,
            ),
            (
                &composite.v_rescale,
                "layer.0.v_linear",
                "layer.0.v_heads",
                0,
            ),
            (
                &composite.o_rescale,
                "layer.0.o_proj",
                "layer.0.attention_residual",
                1,
            ),
        ] {
            assert_eq!(rescale.source_operation_id, source);
            assert_eq!(rescale.target_operation_id, target);
            assert_eq!(rescale.target_input_index, index);
            assert_eq!(rescale.input_fractional_bits, 14);
            assert_eq!(rescale.output_fractional_bits, 10);
            assert_eq!(rescale.divisor, 16);
        }
        assert_eq!(composite.q_reshape.operation_id, "layer.0.q_heads");
        assert_eq!(composite.k_reshape.operation_id, "layer.0.k_heads");
        assert_eq!(composite.v_reshape.operation_id, "layer.0.v_heads");
        assert_eq!(composite.q_rope.operation_id, "layer.0.rope_q");
        assert_eq!(composite.k_rope.operation_id, "layer.0.rope_k");
        assert_eq!(composite.key_append.operation_id, "layer.0.key_append");
        assert_eq!(composite.value_append.operation_id, "layer.0.value_append");
        assert_eq!(composite.key_append.state_kind, StateKind::Key);
        assert_eq!(composite.value_append.state_kind, StateKind::Value);
        assert_eq!(composite.key_append.current_input_id, "layer.0.rope_k");
        assert_eq!(composite.value_append.current_input_id, "layer.0.v_heads");
        assert_eq!(composite.key_view.operation_id, "layer.0.key_view");
        assert_eq!(composite.value_view.operation_id, "layer.0.value_view");
        assert_eq!(composite.key_view.append_producer_id, "layer.0.key_append");
        assert_eq!(
            composite.value_view.append_producer_id,
            "layer.0.value_append"
        );
        assert_eq!(
            composite.scores.score_operation_id,
            "layer.0.attention_scores"
        );
        assert_eq!(composite.scores.query_input_id, "layer.0.rope_q");
        assert_eq!(composite.scores.key_input_id, "layer.0.key_view");
        assert_eq!(
            composite.scores.layout,
            pllm_core::AttentionScoreQ20Layout::GroupedQueryCache
        );
        assert_eq!(composite.softmax.operation_id, "layer.0.softmax");
        assert_eq!(
            composite.attention_values.operation_id,
            "layer.0.attention_values"
        );
        assert_eq!(
            composite.attention_values.probabilities_input_id,
            "layer.0.softmax"
        );
        assert_eq!(
            composite.attention_values.values_input_id,
            "layer.0.value_view"
        );
        assert_eq!(
            composite.attention_values.layout,
            pllm_core::AttentionValueQ10Layout::GroupedQuery
        );
        assert_eq!(
            composite.attention_hidden_reshape.operation_id,
            "layer.0.attention_hidden"
        );
        assert_eq!(
            composite.residual.operation_id,
            "layer.0.attention_residual"
        );
        assert_eq!(composite.residual.input_ids[1], "layer.0.o_proj");
        assert_eq!(composite.residual.input_ids[0], composite.norm.input_id);
        let expected_append_mode = if mode == DecoderMode::Prefill {
            KvCacheAppendMode::Initialize
        } else {
            KvCacheAppendMode::Append
        };
        assert_eq!(composite.key_append.append_mode, expected_append_mode);
        assert_eq!(
            composite.norm_weight.element_type,
            DenseQwenAttentionElementType::SignedI16
        );
        assert_eq!(composite.norm_weight.fractional_bits, 10);
        assert_eq!(composite.norm_weight.shape, vec![8]);
        for (artifact, shape) in [
            (&composite.q_weight, vec![8, 8]),
            (&composite.k_weight, vec![4, 8]),
            (&composite.v_weight, vec![4, 8]),
            (&composite.o_weight, vec![8, 8]),
        ] {
            assert_eq!(
                artifact.element_type,
                DenseQwenAttentionElementType::SignedI8
            );
            assert_eq!(artifact.fractional_bits, 4);
            assert_eq!(
                artifact.layout,
                DenseQwenAttentionLayout::RowMajorOutputInput
            );
            assert_eq!(artifact.rounding, FixedPointRounding::TiesToEven);
            assert_eq!(
                artifact.range_policy,
                DenseQwenAttentionRangePolicy::RejectNoSaturation
            );
            assert_eq!(artifact.shape, shape);
        }
        for (artifact, bytes) in [
            (&composite.norm_weight, weights.norm.as_slice()),
            (&composite.q_weight, weights.q.as_slice()),
            (&composite.k_weight, weights.k.as_slice()),
            (&composite.v_weight, weights.v.as_slice()),
            (&composite.o_weight, weights.o.as_slice()),
        ] {
            assert_eq!(
                artifact.data_digest,
                digest_bytes("pllm.dense_qwen_attention.weight_bytes.v1", bytes)
            );
        }
        assert_eq!(composite.q.input.numeric, NumericType::Wrap32);
        assert_eq!(composite.q_rope.mode, mode);
        assert!(!composite.protected_execution);
        assert!(!composite.complete_decoder);
        let second = compile(&plan, mode, &weights).unwrap();
        assert_eq!(block.binding_digest(), second.binding_digest());
    }
}

#[test]
fn prefill_executes_full_pipeline_and_creates_usable_state() {
    let plan = qwen_plan(1, 2, 2);
    let weights = TestWeights::new();
    let block = compile(&plan, DecoderMode::Prefill, &weights).unwrap();
    let input = [
        300_i16, -500, 200, -100, 64, 900, -750, 420, -210, 88, 512, -640, 130, -95, 260, -480,
    ];
    let positions = [0_u32, 1];
    let mask = [true, true];
    let valid = [2_usize];
    let (output, state) = execute_dense_qwen_attention_prefill(
        &plan,
        &block,
        &wrap_i16(&input),
        &positions,
        &mask,
        &valid,
        execution_policy(),
    )
    .unwrap();
    assert_eq!(output.shape(), [1, 2, 8]);
    assert!(state.is_usable());
    let mut caches = None;
    let expected = oracle_forward(
        &weights,
        &block,
        &input,
        &positions,
        &mask,
        &valid,
        &mut caches,
    );
    assert_eq!(output.values(), expected.as_slice());
}

#[test]
fn decode_after_prefill_advances_state_and_matches_oracle() {
    let plan = qwen_plan(1, 2, 3);
    let weights = TestWeights::new();
    let prefill_block = compile(&plan, DecoderMode::Prefill, &weights).unwrap();
    let decode_block = compile(&plan, DecoderMode::Decode, &weights).unwrap();
    let input = [
        300_i16, -500, 200, -100, 64, 900, -750, 420, -210, 88, 512, -640, 130, -95, 260, -480,
    ];
    let (prefill_output, mut state) = execute_dense_qwen_attention_prefill(
        &plan,
        &prefill_block,
        &wrap_i16(&input),
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    )
    .unwrap();
    assert!(state.is_usable());
    let mut caches = None;
    oracle_forward(
        &weights,
        &prefill_block,
        &input,
        &[0, 1],
        &[true, true],
        &[2],
        &mut caches,
    );
    drop(prefill_output);
    let step_one = [180_i16, -220, 90, 410, -330, 75, 610, -145];
    let first = execute_dense_qwen_attention_decode(
        &plan,
        &decode_block,
        &mut state,
        &wrap_i16(&step_one),
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .unwrap();
    assert_eq!(first.shape(), [1, 1, 8]);
    assert!(state.is_usable());
    let expected_first = oracle_forward(
        &weights,
        &decode_block,
        &step_one,
        &[2],
        &[true],
        &[3],
        &mut caches,
    );
    assert_eq!(first.values(), expected_first.as_slice());
    let step_two = [-40_i16, 95, -510, 240, 330, -60, 150, -270];
    let second = execute_dense_qwen_attention_decode(
        &plan,
        &decode_block,
        &mut state,
        &wrap_i16(&step_two),
        &[3],
        &[true],
        &[4],
        execution_policy(),
    )
    .unwrap();
    assert_eq!(second.shape(), [1, 1, 8]);
    assert!(state.is_usable());
    let expected_second = oracle_forward(
        &weights,
        &decode_block,
        &step_two,
        &[3],
        &[true],
        &[4],
        &mut caches,
    );
    assert_eq!(second.values(), expected_second.as_slice());
}

#[test]
fn padded_prefill_hides_unwritten_slots_and_rejects_poisoned_padding() {
    let plan = qwen_plan(2, 2, 1);
    let weights = TestWeights::new();
    let block = compile(&plan, DecoderMode::Prefill, &weights).unwrap();
    let mut input = vec![0_i16; 32];
    input[..8].copy_from_slice(&[300, -500, 200, -100, 64, 900, -750, 420]);
    input[8..16].copy_from_slice(&[-210, 88, 512, -640, 130, -95, 260, -480]);
    input[16..24].copy_from_slice(&[150, -320, 480, 90, -220, 640, -510, 275]);
    let positions = [0_u32, 1, 0, 1];
    let mask = [true, true, true, false];
    let valid = [2_usize, 1];
    let (output, state) = execute_dense_qwen_attention_prefill(
        &plan,
        &block,
        &wrap_i16(&input),
        &positions,
        &mask,
        &valid,
        execution_policy(),
    )
    .unwrap();
    assert_eq!(output.shape(), [2, 2, 8]);
    assert!(state.is_usable());
    let mut caches = None;
    let expected = oracle_forward(
        &weights,
        &block,
        &input,
        &positions,
        &mask,
        &valid,
        &mut caches,
    );
    assert_eq!(output.values(), expected.as_slice());
    let mut poisoned = input.clone();
    poisoned[24] = 7;
    assert!(execute_dense_qwen_attention_prefill(
        &plan,
        &block,
        &wrap_i16(&poisoned),
        &positions,
        &mask,
        &valid,
        execution_policy(),
    )
    .is_err());
}

#[test]
fn compile_rejects_wrong_family_transformations_weights_and_bounds() {
    let plan = qwen_plan(1, 2, 2);
    let weights = TestWeights::new();

    let mut family = plan.clone();
    family.model_family = "qwen3".into();
    assert!(compile(&family, DecoderMode::Prefill, &weights).is_err());

    let mut transformed = plan.clone();
    transformed.transformations.push(AppliedMethod {
        component: "cache".into(),
        implementation: "mpcache".into(),
        method_id: "mpcache".into(),
        input_digest: digest_bytes("test", b"input"),
        configuration_digest: digest_bytes("test", b"config"),
    });
    assert!(compile(&transformed, DecoderMode::Prefill, &weights).is_err());

    let mut gathered = plan.clone();
    gathered.decode.state_inputs.push(StateTensor {
        id: "state.layer.0.cache_indices".into(),
        layer: Some(0),
        kind: StateKind::CacheIndices,
        shape: vec![1, 1, 1],
        maximum_sequence: 1,
    });
    assert!(compile(&gathered, DecoderMode::Decode, &weights).is_err());

    let wrong_id = DenseQwenAttentionWeights {
        q_q4: weight("model.layers.0.self_attn.q_proj.weight.backup", &weights.q),
        ..weights.input()
    };
    let wrong_manifest = DenseQwenAttentionWeightManifest::from_weights(wrong_id).unwrap();
    assert!(compile_dense_qwen_attention_block(
        &plan,
        DecoderMode::Prefill,
        0,
        wrong_manifest,
        resource_policy(),
        wrong_id,
    )
    .is_err());

    let short = DenseQwenAttentionWeights {
        o_q4: weight("model.layers.0.self_attn.o_proj.weight", &weights.o[..63]),
        ..weights.input()
    };
    let short_manifest = DenseQwenAttentionWeightManifest::from_weights(short).unwrap();
    assert!(compile_dense_qwen_attention_block(
        &plan,
        DecoderMode::Prefill,
        0,
        short_manifest,
        resource_policy(),
        short,
    )
    .is_err());

    let mut altered = TestWeights::new();
    altered.q[0] ^= 1;
    let stale_manifest = DenseQwenAttentionWeightManifest::from_weights(weights.input()).unwrap();
    assert!(compile_dense_qwen_attention_block(
        &plan,
        DecoderMode::Prefill,
        0,
        stale_manifest,
        resource_policy(),
        altered.input(),
    )
    .is_err());

    let input = weights.input();
    let manifest = DenseQwenAttentionWeightManifest::from_weights(input).unwrap();
    let total = input.norm_q10.bytes.len()
        + input.q_q4.bytes.len()
        + input.k_q4.bytes.len()
        + input.v_q4.bytes.len()
        + input.o_q4.bytes.len();
    assert!(compile_dense_qwen_attention_block(
        &plan,
        DecoderMode::Prefill,
        0,
        manifest,
        DenseQwenAttentionResourcePolicy {
            max_total_weight_bytes: u64::try_from(total - 1).unwrap(),
            max_activation_elements: 1 << 20,
        },
        input,
    )
    .is_err());
    let input = weights.input();
    let manifest = DenseQwenAttentionWeightManifest::from_weights(input).unwrap();
    assert!(compile_dense_qwen_attention_block(
        &plan,
        DecoderMode::Prefill,
        0,
        manifest,
        DenseQwenAttentionResourcePolicy {
            max_total_weight_bytes: 1 << 20,
            max_activation_elements: 4,
        },
        input,
    )
    .is_err());

    let mut above_range = TestWeights::new();
    above_range.q[0] = 8;
    assert!(compile(&plan, DecoderMode::Prefill, &above_range).is_err());
    let mut below_range = TestWeights::new();
    below_range.k[0] = (-9_i8) as u8;
    assert!(compile(&plan, DecoderMode::Prefill, &below_range).is_err());

    let mut full_scale = TestWeights::new();
    full_scale.q = vec![7_u8; 64];
    full_scale.k = vec![7_u8; 32];
    full_scale.v = vec![7_u8; 32];
    full_scale.o = vec![7_u8; 64];
    assert!(compile(&plan, DecoderMode::Prefill, &full_scale).is_ok());
}

#[test]
fn execution_rejects_wrong_mode_plan_binding_and_inputs() {
    let plan = qwen_plan(1, 2, 2);
    let weights = TestWeights::new();
    let prefill_block = compile(&plan, DecoderMode::Prefill, &weights).unwrap();
    let decode_block = compile(&plan, DecoderMode::Decode, &weights).unwrap();
    let input = wrap_i16(&[
        300_i16, -500, 200, -100, 64, 900, -750, 420, -210, 88, 512, -640, 130, -95, 260, -480,
    ]);
    assert!(execute_dense_qwen_attention_prefill(
        &plan,
        &decode_block,
        &input,
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    )
    .is_err());
    let (_, mut state) = execute_dense_qwen_attention_prefill(
        &plan,
        &prefill_block,
        &input,
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    )
    .unwrap();
    let step = wrap_i16(&[1_i16; 8]);
    assert!(execute_dense_qwen_attention_decode(
        &plan,
        &prefill_block,
        &mut state,
        &step,
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
    let other_block = compile(&plan, DecoderMode::Decode, &{
        let mut changed = TestWeights::new();
        changed.q[3] ^= 1;
        changed
    })
    .unwrap();
    assert!(execute_dense_qwen_attention_decode(
        &plan,
        &other_block,
        &mut state,
        &step,
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
    let mut wrong_plan = plan.clone();
    wrong_plan.config_digest = digest_bytes("test.altered", b"altered");
    assert!(execute_dense_qwen_attention_decode(
        &wrong_plan,
        &decode_block,
        &mut state,
        &step,
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
    assert!(state.is_usable());
    for (input_len, positions, mask, valid) in [
        (15, &[2][..], &[true][..], &[3_usize][..]),
        (8, &[2, 9][..], &[true][..], &[3_usize][..]),
        (8, &[2][..], &[true, false][..], &[3_usize][..]),
        (8, &[2][..], &[true][..], &[3_usize, 4][..]),
    ] {
        assert!(execute_dense_qwen_attention_decode(
            &plan,
            &decode_block,
            &mut state,
            &input[..input_len],
            positions,
            mask,
            valid,
            execution_policy(),
        )
        .is_err());
        assert!(state.is_usable());
    }
}

#[test]
fn post_append_failure_poisons_state_and_blocks_later_decode() {
    let plan = qwen_plan(1, 2, 2);
    let weights = TestWeights::new();
    let prefill_block = compile(&plan, DecoderMode::Prefill, &weights).unwrap();
    let decode_block = compile(&plan, DecoderMode::Decode, &weights).unwrap();
    let input = wrap_i16(&[
        300_i16, -500, 200, -100, 64, 900, -750, 420, -210, 88, 512, -640, 130, -95, 260, -480,
    ]);
    let (_, mut state) = execute_dense_qwen_attention_prefill(
        &plan,
        &prefill_block,
        &input,
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    )
    .unwrap();
    let mut tight = execution_policy();
    tight.softmax = pllm_core::SoftmaxQ30Policy::new(1 << 20, 2).unwrap();
    let step = wrap_i16(&[1_i16; 8]);
    assert!(execute_dense_qwen_attention_decode(
        &plan,
        &decode_block,
        &mut state,
        &step,
        &[2],
        &[true],
        &[3],
        tight,
    )
    .is_err());
    assert!(!state.is_usable());
    assert!(execute_dense_qwen_attention_decode(
        &plan,
        &decode_block,
        &mut state,
        &step,
        &[2],
        &[true],
        &[3],
        execution_policy(),
    )
    .is_err());
}

#[test]
fn tampered_composite_regions_digests_and_flags_are_rejected() {
    let plan = qwen_plan(1, 2, 2);
    let weights = TestWeights::new();
    let input = wrap_i16(&[
        300_i16, -500, 200, -100, 64, 900, -750, 420, -210, 88, 512, -640, 130, -95, 260, -480,
    ]);
    let mutations: Vec<fn(&mut CompiledDenseQwenAttentionBlock)> = vec![
        |block| block.composite.q.operation_id = "tampered".into(),
        |block| block.composite.scores.query_input_id = "tampered".into(),
        |block| block.composite.residual.input_ids[0] = "tampered".into(),
        |block| block.composite.q_weight.fractional_bits = 5,
        |block| block.composite.weight_manifest_digest = digest_bytes("test", b"forged"),
        |block| block.composite.protected_execution = true,
        |block| block.composite.complete_decoder = true,
        |block| block.composite.schema_version = "forged.schema".into(),
    ];
    for mutate in mutations {
        let mut tampered = compile(&plan, DecoderMode::Prefill, &weights).unwrap();
        mutate(&mut tampered);
        assert!(execute_dense_qwen_attention_prefill(
            &plan,
            &tampered,
            &input,
            &[0, 1],
            &[true, true],
            &[2],
            execution_policy(),
        )
        .is_err());
    }
}

#[test]
fn execute_model_linear_runs_multirow_and_rejects_malformed() {
    let plan = qwen_plan(1, 2, 2);
    let region = lower_model_linear_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .into_iter()
        .find(|region| region.operation_id == "layer.0.q_linear")
        .unwrap();
    let weights = q4_matrix(8, 8, 0);
    let input: Vec<u32> = (0..16).map(|value| (value * 7 + 3) as u32).collect();
    let output = execute_model_linear(&region, &weights, &input, 1, false).unwrap();
    assert_eq!(output.len(), 16);
    let expected = |row: usize, column: usize| -> u32 {
        (0..8_usize).fold(0_u32, |total, inner| {
            let weight = weights[column * 8 + inner] as i8 as i32 as u32;
            total.wrapping_add(weight.wrapping_mul(input[row * 8 + inner]))
        })
    };
    for row in 0..2 {
        for column in 0..8 {
            assert_eq!(output[row * 8 + column], expected(row, column));
        }
    }
    assert!(execute_model_linear(&region, &weights[..63], &input, 1, false).is_err());
    assert!(execute_model_linear(&region, &weights, &input[..15], 1, false).is_err());
    let mut forged = region.clone();
    forged.input.numeric = NumericType::SignedFixedQ10;
    assert!(execute_model_linear(&forged, &weights, &input, 1, false).is_err());
    let mut forged = region;
    forged.operation.output.shape = vec![2, 4, 2];
    assert!(execute_model_linear(&forged, &weights, &input, 1, false).is_err());
}

#[test]
fn prefill_rejects_rms_norm_output_outside_signed_i16() {
    let plan = qwen_plan(1, 2, 2);
    let mut weights = TestWeights::new();
    weights.norm = [i16::MAX; 8]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    weights.norm_i16 = vec![i16::MAX; 8];
    let block = compile(&plan, DecoderMode::Prefill, &weights).unwrap();
    let input = wrap_i16(&[30_000_i16, 0, 0, 0, 0, 0, 0, 0, 30_000, 0, 0, 0, 0, 0, 0, 0]);
    let result = execute_dense_qwen_attention_prefill(
        &plan,
        &block,
        &input,
        &[0, 1],
        &[true, true],
        &[2],
        execution_policy(),
    );
    assert!(result.is_err());
    let error = result.err().unwrap();
    assert!(error.contains("RMSNorm output is outside signed Q10 i16"));
}
