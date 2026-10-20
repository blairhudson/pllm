use super::*;

fn qwen() -> Value {
    json!({"model_type":"qwen2", "hidden_size":896, "intermediate_size":4864,
        "num_hidden_layers":24,"num_attention_heads":14,"num_key_value_heads":2,
        "vocab_size":151936,"max_position_embeddings":32768,"hidden_act":"silu",
        "rms_norm_eps":0.000001,"rope_theta":1000000.0,"tie_word_embeddings":true,
        "attention_dropout":0.0,"use_cache":true,"use_sliding_window":false})
}

#[test]
fn decoder_source_geometry_admits_complete_qwen_without_tensor_payloads() {
    let config = canonical_bytes(&qwen());
    assert!(preflight_source(&config, 1, 1, DENSE_ROPE_NUMERIC).is_err());
    for (rows, outputs) in [(1, 1), (2, 2)] {
        let a = preflight_source_with_capacity(
            &config,
            rows,
            outputs,
            DENSE_ROPE_NUMERIC,
            Capacity::Decoder,
        )
        .unwrap();
        assert_eq!(a.capacity, Capacity::Decoder);
        assert_eq!(a.plan.prefill.operations.len(), 678);
        assert_eq!(a.source_shapes.len(), 290);
        assert_eq!(a.source_shapes["model.embed_tokens.weight"], [151936, 896]);
        assert_eq!(a.source_q12_value_bytes, 3_952_262_144);
        assert_eq!(a.source_float32_bytes, 1_976_131_072);
        assert_eq!(a.source_disk_bytes, 3_973_580_800);
        assert!(a.source_minimum_owner_bytes < 8 << 20);
        assert!(a.phase_admission.ingestion_peak_bound <= 64 << 20);
        assert!(a.phase_admission.compile_prepare_peak_bound <= 64 << 20);
    }
    assert!(
        preflight_source_with_capacity(&config, 64, 8, DENSE_ROPE_NUMERIC, Capacity::Decoder)
            .is_err()
    );
}

#[test]
fn decoder_geometry_rejects_unbounded_sources_before_graph_expansion() {
    for (field, value) in [
        ("num_hidden_layers", 1000000u64),
        ("vocab_size", u64::MAX),
        ("hidden_size", 1 << 31),
        ("intermediate_size", 1 << 32),
    ] {
        let mut config = qwen();
        config[field] = json!(value);
        let error = preflight_source_with_capacity(
            &canonical_bytes(&config),
            1,
            1,
            DENSE_ROPE_NUMERIC,
            Capacity::Decoder,
        )
        .err()
        .unwrap();
        assert!(error.contains("expansion"), "{error}");
    }
}
