use pllm_compiler::{
    append_model_kv_cache_q10, decoder_coverage, execute_model_kv_cache_view_q10,
    execute_model_rope_q10, initialize_model_kv_cache_q10, lower_model_kv_cache_append_q10_regions,
    lower_model_kv_cache_view_q10_regions, lower_model_rope_q10_regions,
    provenance_primitives_artifact_digest, CapabilityLevel, KvCacheAppendMode,
    KvCacheQ10ExecutionInputs, ProvenancePrimitiveResourcePolicy,
    KV_CACHE_APPEND_Q10_REGION_SCHEMA_VERSION, KV_CACHE_VIEW_Q10_REGION_SCHEMA_VERSION,
    PROVENANCE_PRIMITIVE_HARD_MAX_CACHE_BYTES, PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_BYTES,
    PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_DEPTH, PROVENANCE_PRIMITIVE_HARD_MAX_OPERATIONS,
    PROVENANCE_PRIMITIVE_HARD_MAX_ROPE_ELEMENTS, PROVENANCE_PRIMITIVE_HARD_MAX_STATES,
    PROVENANCE_PRIMITIVE_HARD_MAX_VIEW_BYTES, ROPE_Q10_REGION_SCHEMA_VERSION,
};
use pllm_models::{
    lower_model_json, DecoderMode, DecoderPlan, DecoderWorkload, ModelOperator, StateKind,
};

const QWEN2: &[u8] = br#"{
    "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
    "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
    "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
    "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
}"#;

const QWEN3: &[u8] = br#"{
    "model_type":"qwen3","hidden_size":8,"intermediate_size":16,
    "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
    "vocab_size":32,"max_position_embeddings":32,"head_dim":4,"hidden_act":"silu",
    "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":false,
    "attention_bias":false,"attention_dropout":0.0,"rope_scaling":null,
    "sliding_window":null,"use_sliding_window":false,"use_cache":true,
    "max_window_layers":1,"layer_types":["full_attention"]
}"#;

const QWEN2_LARGE_POSITION: &[u8] = br#"{
    "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
    "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
    "vocab_size":32,"max_position_embeddings":3000000,"hidden_act":"silu",
    "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
}"#;

fn plan(config: &[u8], batch: u64, input: u64, new: u64) -> DecoderPlan {
    lower_model_json(
        config,
        DecoderWorkload {
            batch,
            max_input_tokens: input,
            max_new_tokens: new,
        },
    )
    .unwrap()
}

fn resources() -> ProvenancePrimitiveResourcePolicy {
    constrained_resources(1_024, 1_024, 1_024)
}

fn constrained_resources(
    max_rope_elements: usize,
    max_cache_bytes: usize,
    max_view_bytes: usize,
) -> ProvenancePrimitiveResourcePolicy {
    ProvenancePrimitiveResourcePolicy::new(
        max_rope_elements,
        max_cache_bytes,
        max_view_bytes,
        1_000,
        1_000,
        1_048_576,
        32,
    )
    .unwrap()
}

fn kv_inputs<'a>(
    current: &'a [i16],
    positions: &'a [u32],
    attention_mask: &'a [bool],
    valid_lengths: &'a [usize],
) -> KvCacheQ10ExecutionInputs<'a> {
    KvCacheQ10ExecutionInputs::new(current, positions, attention_mask, valid_lengths)
}

#[test]
fn compact_qwen_regions_bind_exact_semantic_provenance() {
    assert!(std::ptr::eq(
        provenance_primitives_artifact_digest(),
        provenance_primitives_artifact_digest()
    ));
    for decoder in [plan(QWEN2, 1, 2, 2), plan(QWEN3, 1, 2, 2)] {
        for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
            let expected_plan_digest = decoder.digest();
            let ropes = lower_model_rope_q10_regions(&decoder, mode, &resources()).unwrap();
            let appends =
                lower_model_kv_cache_append_q10_regions(&decoder, mode, &resources()).unwrap();
            let views =
                lower_model_kv_cache_view_q10_regions(&decoder, mode, &resources()).unwrap();
            assert_eq!(ropes.len(), 2);
            assert_eq!(appends.len(), 2);
            assert_eq!(views.len(), 2);
            assert!(ropes.iter().all(|region| {
                region.schema_version == ROPE_Q10_REGION_SCHEMA_VERSION
                    && region.model_plan_digest == expected_plan_digest
                    && region.positions_id == "input.positions"
                    && region.positions_shape
                        == vec![1, decoder_graph(&decoder, mode).query_sequence]
                    && region.theta == "10000"
                    && region.compiler_core_artifact_digest
                        == *provenance_primitives_artifact_digest()
            }));
            assert!(appends.iter().all(|region| {
                region.schema_version == KV_CACHE_APPEND_Q10_REGION_SCHEMA_VERSION
                    && region.model_plan_digest == expected_plan_digest
                    && region.state_shape == vec![1, 1, 3, 4]
                    && region.state_capacity == 3
                    && region.append_mode
                        == if mode == DecoderMode::Prefill {
                            KvCacheAppendMode::Initialize
                        } else {
                            KvCacheAppendMode::Append
                        }
            }));
            assert!(views.iter().all(|region| {
                region.schema_version == KV_CACHE_VIEW_Q10_REGION_SCHEMA_VERSION
                    && region.model_plan_digest == expected_plan_digest
                    && region.axis == 2
                    && region.semantics == "visible_valid_prefix"
            }));
        }
    }

    let qwen3 = plan(QWEN3, 1, 2, 2);
    let ropes = lower_model_rope_q10_regions(&qwen3, DecoderMode::Prefill, &resources()).unwrap();
    let query = ropes
        .iter()
        .find(|region| region.operation_id == "layer.0.rope_q")
        .unwrap();
    let key = ropes
        .iter()
        .find(|region| region.operation_id == "layer.0.rope_k")
        .unwrap();
    assert_eq!(query.input_id, "layer.0.q_norm");
    assert_eq!(key.input_id, "layer.0.k_norm");

    let encoded = serde_json::to_vec(query).unwrap();
    let decoded = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(query, &decoded);
    assert!(include_str!("../../../Cargo.lock").contains("name = \"libm\"\nversion = \"0.2.16\""));
}

#[test]
fn rope_execution_matches_core_and_rejects_all_provenance_tampering() {
    let decoder = plan(QWEN2, 1, 2, 2);
    let region = lower_model_rope_q10_regions(&decoder, DecoderMode::Prefill, &resources())
        .unwrap()
        .into_iter()
        .find(|region| region.operation_id == "layer.0.rope_q")
        .unwrap();
    let input = (1_i16..=16).collect::<Vec<_>>();
    let positions = [0, 1];
    let policy = resources();
    let output = execute_model_rope_q10(&decoder, &region, &input, &positions, &policy).unwrap();
    let expected = pllm_core::rope_q10(
        &input,
        &positions,
        [1, 2, 2, 4],
        pllm_core::RopeQ10Config {
            theta: 10_000.0,
            rotary_dimensions: 4,
            maximum_position: 3,
        },
    )
    .unwrap();
    assert_eq!(output, expected);
    assert_eq!(&output[0..4], &input[0..4]);
    assert_eq!(&output[8..12], &input[8..12]);
    assert_ne!(&output[4..8], &input[4..8]);
    assert!(execute_model_rope_q10(&decoder, &region, &input, &[0, 3], &policy).is_err());

    let mut forged = region.clone();
    forged.theta = "10001".into();
    assert!(execute_model_rope_q10(&decoder, &forged, &input, &positions, &policy).is_err());
    let mut forged = region.clone();
    forged.coefficient_profile = "forged".into();
    assert!(execute_model_rope_q10(&decoder, &forged, &input, &positions, &policy).is_err());
    let mut forged = region.clone();
    forged.tensor_shape[3] = 2;
    assert!(execute_model_rope_q10(&decoder, &forged, &input, &positions, &policy).is_err());
    let mut forged = region.clone();
    forged.operation_id = "layer.0.rope_k".into();
    assert!(execute_model_rope_q10(&decoder, &forged, &input, &positions, &policy).is_err());
    let mut forged = region.clone();
    forged.input_id = "layer.0.k_heads".into();
    assert!(execute_model_rope_q10(&decoder, &forged, &input, &positions, &policy).is_err());
    let other_plan = plan(QWEN2, 1, 3, 1);
    assert!(execute_model_rope_q10(&other_plan, &region, &input, &positions, &policy).is_err());
}

#[test]
fn rope_requires_sequential_absolute_rows_and_explicit_resources() {
    let decoder = plan(QWEN2, 2, 3, 2);
    let region = lower_model_rope_q10_regions(&decoder, DecoderMode::Prefill, &resources())
        .unwrap()
        .into_iter()
        .find(|region| region.operation_id == "layer.0.rope_q")
        .unwrap();
    let input = vec![1_i16; 2 * 2 * 3 * 4];
    let policy = resources();
    assert!(
        execute_model_rope_q10(&decoder, &region, &input, &[0, 1, 2, 1, 2, 3], &policy,).is_ok()
    );
    for positions in [
        [0, 0, 1, 1, 2, 3],
        [0, 2, 3, 1, 2, 3],
        [2, 1, 0, 1, 2, 3],
        [0, 1, 2, 1, 3, 4],
    ] {
        assert!(execute_model_rope_q10(&decoder, &region, &input, &positions, &policy).is_err());
    }

    let decode = lower_model_rope_q10_regions(&decoder, DecoderMode::Decode, &resources())
        .unwrap()
        .into_iter()
        .find(|region| region.operation_id == "layer.0.rope_q")
        .unwrap();
    assert!(execute_model_rope_q10(&decoder, &decode, &[1; 16], &[3, 0], &policy,).is_ok());

    let too_small = constrained_resources(47, 1_024, 1_024);
    assert!(
        execute_model_rope_q10(&decoder, &region, &input, &[0, 1, 2, 1, 2, 3], &too_small,)
            .unwrap_err()
            .contains("resource policy")
    );
}

#[test]
fn kv_prefill_decode_views_are_bounded_and_fail_atomically() {
    let decoder = plan(QWEN2, 2, 3, 2);
    let prefill =
        lower_model_kv_cache_append_q10_regions(&decoder, DecoderMode::Prefill, &resources())
            .unwrap();
    let decode =
        lower_model_kv_cache_append_q10_regions(&decoder, DecoderMode::Decode, &resources())
            .unwrap();
    let prefill_views =
        lower_model_kv_cache_view_q10_regions(&decoder, DecoderMode::Prefill, &resources())
            .unwrap();
    let decode_views =
        lower_model_kv_cache_view_q10_regions(&decoder, DecoderMode::Decode, &resources()).unwrap();
    let prefill_key = by_kind(&prefill, StateKind::Key);
    let prefill_value = by_kind(&prefill, StateKind::Value);
    let decode_key = by_kind(&decode, StateKind::Key);
    let decode_value = by_kind(&decode, StateKind::Value);
    let prefill_key_view = view_by_kind(&prefill_views, StateKind::Key);
    let decode_key_view = view_by_kind(&decode_views, StateKind::Key);
    let policy = resources();

    let positions = [0, 1, 99, 0, 99, 99];
    let mask = [true, true, false, true, false, false];
    let lengths = [2, 1];
    let key_current = [
        1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let value_current = [
        101, 102, 103, 104, 105, 106, 107, 108, 0, 0, 0, 0, 121, 122, 123, 124, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let mut key_cache = initialize_model_kv_cache_q10(
        &decoder,
        prefill_key,
        kv_inputs(&key_current, &positions, &mask, &lengths),
        &policy,
    )
    .unwrap();
    let value_cache = initialize_model_kv_cache_q10(
        &decoder,
        prefill_value,
        kv_inputs(&value_current, &positions, &mask, &lengths),
        &policy,
    )
    .unwrap();
    assert!(!format!("{key_cache:?}").contains("valid_lengths"));
    assert!(!format!("{key_cache:?}").contains("1, 2, 3, 4"));

    let visible =
        execute_model_kv_cache_view_q10(&key_cache, &decoder, prefill_key_view, &policy).unwrap();
    assert_eq!(visible.valid_lengths(), &[2, 1]);
    assert_eq!(visible.maximum_sequence(), 3);
    assert_eq!(visible.prefix(0, 0), Some(&key_current[..8]));
    assert_eq!(visible.prefix(1, 0), Some(&key_current[12..16]));
    assert_eq!(visible.prefix(0, 0).unwrap().len(), 8);
    assert_eq!(visible.prefix(1, 0).unwrap().len(), 4);

    drop(visible);
    assert!(append_model_kv_cache_q10(
        &mut key_cache,
        &decoder,
        decode_value,
        kv_inputs(
            &[31, 32, 33, 34, 41, 42, 43, 44],
            &[2, 1],
            &[true, true],
            &[3, 2],
        ),
        &policy,
    )
    .is_err());
    assert_eq!(key_cache.valid_lengths(), &[2, 1]);
    let unchanged =
        execute_model_kv_cache_view_q10(&key_cache, &decoder, prefill_key_view, &policy).unwrap();
    assert_eq!(unchanged.prefix(0, 0), Some(&key_current[..8]));
    assert_eq!(unchanged.prefix(1, 0), Some(&key_current[12..16]));
    drop(unchanged);

    for forged in [
        {
            let mut region = decode_key.clone();
            region.state_id = decode_value.state_id.clone();
            region
        },
        {
            let mut region = decode_key.clone();
            region.state_capacity += 1;
            region
        },
        {
            let mut region = decode_key.clone();
            region.operation_id = "forged.append".into();
            region
        },
        {
            let mut region = decode_key.clone();
            region.current_input_id = decode_value.current_input_id.clone();
            region
        },
    ] {
        assert!(append_model_kv_cache_q10(
            &mut key_cache,
            &decoder,
            &forged,
            kv_inputs(
                &[31, 32, 33, 34, 41, 42, 43, 44],
                &[2, 1],
                &[true, true],
                &[3, 2],
            ),
            &policy,
        )
        .is_err());
        assert_eq!(key_cache.valid_lengths(), &[2, 1]);
    }

    let mismatched_plan = plan(QWEN2, 2, 2, 3);
    assert!(append_model_kv_cache_q10(
        &mut key_cache,
        &mismatched_plan,
        decode_key,
        kv_inputs(
            &[31, 32, 33, 34, 41, 42, 43, 44],
            &[2, 1],
            &[true, true],
            &[3, 2],
        ),
        &policy,
    )
    .is_err());
    assert_eq!(key_cache.valid_lengths(), &[2, 1]);

    append_model_kv_cache_q10(
        &mut key_cache,
        &decoder,
        decode_key,
        kv_inputs(
            &[31, 32, 33, 34, 41, 42, 43, 44],
            &[2, 1],
            &[true, true],
            &[3, 2],
        ),
        &policy,
    )
    .unwrap();
    let visible =
        execute_model_kv_cache_view_q10(&key_cache, &decoder, decode_key_view, &policy).unwrap();
    assert_eq!(visible.valid_lengths(), &[3, 2]);
    assert_eq!(visible.maximum_sequence(), 4);
    assert_eq!(
        visible.prefix(0, 0),
        Some(&[1, 2, 3, 4, 5, 6, 7, 8, 31, 32, 33, 34][..])
    );
    assert_eq!(
        visible.prefix(1, 0),
        Some(&[21, 22, 23, 24, 41, 42, 43, 44][..])
    );
    assert_eq!(visible.prefix(0, 0).unwrap().len(), 3 * visible.head_dim());
    assert_eq!(visible.prefix(1, 0).unwrap().len(), 2 * visible.head_dim());

    assert!(append_model_kv_cache_q10(
        &mut key_cache,
        &decoder,
        decode_key,
        kv_inputs(
            &[31, 32, 33, 34, 41, 42, 43, 44],
            &[2, 1],
            &[true, true],
            &[3, 2],
        ),
        &policy,
    )
    .is_err());
    assert!(append_model_kv_cache_q10(
        &mut key_cache,
        &decoder,
        decode_key,
        kv_inputs(
            &[51, 52, 53, 54, 61, 62, 63, 64],
            &[2, 2],
            &[true, true],
            &[4, 3],
        ),
        &policy,
    )
    .is_err());
    assert_eq!(key_cache.valid_lengths(), &[3, 2]);
    let unchanged =
        execute_model_kv_cache_view_q10(&key_cache, &decoder, decode_key_view, &policy).unwrap();
    assert_eq!(unchanged.valid_lengths(), visible.valid_lengths());
    assert_eq!(unchanged.prefix(0, 0), visible.prefix(0, 0));
    assert_eq!(unchanged.prefix(1, 0), visible.prefix(1, 0));

    let mut forged_view = decode_key_view.clone();
    forged_view.maximum_sequence -= 1;
    assert!(execute_model_kv_cache_view_q10(&key_cache, &decoder, &forged_view, &policy).is_err());
    assert!(
        execute_model_kv_cache_view_q10(&value_cache, &decoder, prefill_key_view, &policy).is_err()
    );
}

#[test]
fn resource_policy_is_positive_hard_capped_and_precedes_cache_mutation() {
    assert!(ProvenancePrimitiveResourcePolicy::new(0, 1, 1, 1, 1, 1, 1).is_err());
    assert!(ProvenancePrimitiveResourcePolicy::new(
        PROVENANCE_PRIMITIVE_HARD_MAX_ROPE_ELEMENTS + 1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    .is_err());
    assert!(ProvenancePrimitiveResourcePolicy::new(
        1,
        1,
        1,
        PROVENANCE_PRIMITIVE_HARD_MAX_OPERATIONS + 1,
        1,
        1,
        1,
    )
    .is_err());
    assert!(ProvenancePrimitiveResourcePolicy::new(
        1,
        1,
        1,
        1,
        PROVENANCE_PRIMITIVE_HARD_MAX_STATES + 1,
        1,
        1,
    )
    .is_err());
    assert!(ProvenancePrimitiveResourcePolicy::new(
        1,
        1,
        1,
        1,
        1,
        PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_BYTES + 1,
        1,
    )
    .is_err());
    assert!(ProvenancePrimitiveResourcePolicy::new(
        1,
        1,
        1,
        1,
        1,
        1,
        PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_DEPTH + 1,
    )
    .is_err());
    assert!(ProvenancePrimitiveResourcePolicy::new(
        1,
        PROVENANCE_PRIMITIVE_HARD_MAX_CACHE_BYTES + 1,
        1,
        1,
        1,
        1,
        1,
    )
    .is_err());
    assert!(ProvenancePrimitiveResourcePolicy::new(
        1,
        1,
        PROVENANCE_PRIMITIVE_HARD_MAX_VIEW_BYTES + 1,
        1,
        1,
        1,
        1,
    )
    .is_err());

    let decoder = plan(QWEN2, 1, 2, 2);
    let prefill =
        lower_model_kv_cache_append_q10_regions(&decoder, DecoderMode::Prefill, &resources())
            .unwrap();
    let decode =
        lower_model_kv_cache_append_q10_regions(&decoder, DecoderMode::Decode, &resources())
            .unwrap();
    let views = lower_model_kv_cache_view_q10_regions(&decoder, DecoderMode::Prefill, &resources())
        .unwrap();
    let prefill_key = by_kind(&prefill, StateKind::Key);
    let decode_key = by_kind(&decode, StateKind::Key);
    let view = view_by_kind(&views, StateKind::Key);
    let too_small_cache = constrained_resources(1_024, 31, 1_024);
    assert!(initialize_model_kv_cache_q10(
        &decoder,
        prefill_key,
        kv_inputs(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1], &[true, true], &[2],),
        &too_small_cache,
    )
    .is_err());

    let policy = resources();
    let mut cache = initialize_model_kv_cache_q10(
        &decoder,
        prefill_key,
        kv_inputs(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1], &[true, true], &[2]),
        &policy,
    )
    .unwrap();
    assert!(append_model_kv_cache_q10(
        &mut cache,
        &decoder,
        decode_key,
        kv_inputs(&[9, 10, 11, 12], &[2], &[true], &[3]),
        &too_small_cache,
    )
    .is_err());
    assert_eq!(cache.valid_lengths(), &[2]);

    let too_small_view = constrained_resources(1_024, 1_024, 23);
    assert!(execute_model_kv_cache_view_q10(&cache, &decoder, view, &too_small_view).is_err());
}

#[test]
fn plan_authentication_policy_rejects_before_relowering_or_core_work() {
    let decoder = plan(QWEN2, 1, 2, 2);
    let normal = resources();
    let region = lower_model_rope_q10_regions(&decoder, DecoderMode::Prefill, &normal)
        .unwrap()
        .remove(0);
    let operation_limited =
        ProvenancePrimitiveResourcePolicy::new(1_024, 1_024, 1_024, 1, 1_000, 1_048_576, 32)
            .unwrap();
    let error =
        execute_model_rope_q10(&decoder, &region, &[], &[], &operation_limited).unwrap_err();
    assert!(error.contains("operations"), "{error}");

    let metadata_limited =
        ProvenancePrimitiveResourcePolicy::new(1_024, 1_024, 1_024, 1_000, 1_000, 1, 32).unwrap();
    let error = lower_model_rope_q10_regions(&decoder, DecoderMode::Prefill, &metadata_limited)
        .unwrap_err();
    assert!(error.contains("metadata requires"), "{error}");

    let depth_limited =
        ProvenancePrimitiveResourcePolicy::new(1_024, 1_024, 1_024, 1_000, 1_000, 1_048_576, 1)
            .unwrap();
    let error =
        lower_model_rope_q10_regions(&decoder, DecoderMode::Prefill, &depth_limited).unwrap_err();
    assert!(error.contains("metadata depth"));
}

#[test]
fn coverage_marks_valid_q10_descriptors_executable() {
    let decoder = plan(QWEN2, 1, 2, 2);
    let coverage = decoder_coverage(&decoder, "research.single_evaluator");
    for (operator, component) in [
        (
            ModelOperator::RotaryEmbedding,
            "pllm/core-rope-q10@0.1.0-alpha.1",
        ),
        (
            ModelOperator::KvCacheAppend,
            "pllm/core-kv-cache-q10@0.1.0-alpha.1",
        ),
        (
            ModelOperator::CacheSuffix,
            "pllm/core-kv-cache-q10@0.1.0-alpha.1",
        ),
    ] {
        let item = coverage
            .operators
            .iter()
            .find(|item| item.operator == operator)
            .unwrap();
        assert_eq!(item.level, CapabilityLevel::ExecutableRegion);
        assert_eq!(item.component.as_deref(), Some(component));
        assert!(item.blocker.contains("whole-decoder scheduling"));
    }
    assert!(!coverage.complete);
}

#[test]
fn coverage_rejects_legacy_and_malformed_q10_descriptors() {
    for config in [
        include_bytes!("../../pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json")
            .as_slice(),
        include_bytes!("../../pllm-models/tests/fixtures/Phi-4-mini-instruct-cfbefac-config.json")
            .as_slice(),
        include_bytes!("../../pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json")
            .as_slice(),
    ] {
        let legacy = plan(config, 1, 2, 1);
        let coverage = decoder_coverage(&legacy, "research.single_evaluator");
        for item in coverage.operators.iter().filter(|item| {
            matches!(
                item.operator,
                ModelOperator::RotaryEmbedding
                    | ModelOperator::KvCacheAppend
                    | ModelOperator::CacheSuffix
            )
        }) {
            assert_eq!(item.level, CapabilityLevel::Missing);
            assert!(item.blocker.contains("primitive is unavailable"));
        }
    }

    let mut malformed = plan(QWEN2, 1, 2, 2);
    malformed
        .prefill
        .operations
        .iter_mut()
        .find(|operation| operation.operator == ModelOperator::RotaryEmbedding)
        .unwrap()
        .attributes["coefficient_profile"] = serde_json::json!("forged.profile");
    let coverage = decoder_coverage(&malformed, "research.single_evaluator");
    let rope = coverage
        .operators
        .iter()
        .find(|item| item.operator == ModelOperator::RotaryEmbedding)
        .unwrap();
    assert_eq!(rope.level, CapabilityLevel::Missing);
    assert!(rope.blocker.contains("plan validation failed"));
    for operator in [ModelOperator::KvCacheAppend, ModelOperator::CacheSuffix] {
        let item = coverage
            .operators
            .iter()
            .find(|item| item.operator == operator)
            .unwrap();
        assert_eq!(item.level, CapabilityLevel::Missing);
        assert!(item.blocker.contains("plan validation failed"));
    }

    let oversized = plan(
        QWEN2_LARGE_POSITION,
        1,
        u64::try_from(PROVENANCE_PRIMITIVE_HARD_MAX_ROPE_ELEMENTS / 8 + 1).unwrap(),
        1,
    );
    let coverage = decoder_coverage(&oversized, "research.single_evaluator");
    let rope = coverage
        .operators
        .iter()
        .find(|item| item.operator == ModelOperator::RotaryEmbedding)
        .unwrap();
    assert_eq!(rope.level, CapabilityLevel::Missing);
    assert!(rope.blocker.contains("immutable hard maximum"));
}

fn decoder_graph(plan: &DecoderPlan, mode: DecoderMode) -> &pllm_models::DecoderGraph {
    match mode {
        DecoderMode::Prefill => &plan.prefill,
        DecoderMode::Decode => &plan.decode,
    }
}

fn by_kind(
    regions: &[pllm_compiler::ModelKvCacheAppendQ10Region],
    kind: StateKind,
) -> &pllm_compiler::ModelKvCacheAppendQ10Region {
    regions
        .iter()
        .find(|region| region.state_kind == kind)
        .unwrap()
}

fn view_by_kind(
    regions: &[pllm_compiler::ModelKvCacheViewQ10Region],
    kind: StateKind,
) -> &pllm_compiler::ModelKvCacheViewQ10Region {
    regions
        .iter()
        .find(|region| region.state_kind == kind)
        .unwrap()
}
