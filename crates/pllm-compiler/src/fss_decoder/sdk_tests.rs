use super::*;

#[test]
fn explicit_source_capacity_is_bound_and_legacy_contract_has_no_capacity_fields() {
    let config = dense();
    let old = contract(&experiment(DENSE_ROPE_NUMERIC), &config, 3, 2).unwrap();
    let old_record: Value = serde_json::from_slice(&old).unwrap();
    for field in [
        "capacity",
        "source_disk_bytes",
        "source_resident_bound_bytes",
    ] {
        assert!(old_record.get(field).is_none());
    }
    let capsule = preflight(&config, DENSE_ROPE_NUMERIC).unwrap();
    let capsule_binding = SourceExecutionBinding::new(&capsule, 2).unwrap();
    capsule_binding.checked_contract_digest(&old).unwrap();
    let mut experiment = experiment(DENSE_ROPE_NUMERIC);
    experiment["pipeline"]["components"]["linear"]["params"]["capacity"] = json!("decoder");
    let encoded = contract(&experiment, &config, 3, 2).unwrap();
    let decoder = preflight_source_with_capacity(
        &canonical_bytes(&config),
        3,
        2,
        DENSE_ROPE_NUMERIC,
        Capacity::Decoder,
    )
    .unwrap();
    let binding = SourceExecutionBinding::new(&decoder, 2).unwrap();
    binding.checked_contract_digest(&encoded).unwrap();
    assert_ne!(binding.identity, capsule_binding.identity);
    assert!(binding.checked_contract_digest(&old).is_err());
    assert!(capsule_binding.checked_contract_digest(&encoded).is_err());
    let record: Value = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(record["capacity"], "decoder");
    assert_eq!(record["admission"]["numeric"], "not_performed");
    assert_eq!(record["admission"]["serving_qualified"], false);
    for (field, replacement) in [
        ("capacity", json!("capsule")),
        ("source_disk_bytes", json!(0)),
        ("source_resident_bound_bytes", json!(1)),
    ] {
        let mut changed = record.clone();
        changed[field] = replacement;
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&changed))
            .is_err());
    }
}

fn experiment(profile: &str) -> Value {
    json!({"schema":"pllm.experiment.v2","name":"fss-metadata-only",
        "pipeline":{"model":{"source":"generated/public-source"},"components":{
            "linear":{"component":PROTOCOL,"params":{"numeric_profile":profile}},
            "topology":{"component":TOPOLOGY,"params":{}},
            "state":{"component":STATE,"params":{}},
            "kernels":{"component":"pllm/cpu","params":{"threads":1}}}},
        "deployment":{"kind":"local","root":"local://fss-metadata-only"},
        "budget":{"requests":1,"max_input_tokens":3,"max_new_tokens":2}})
}
fn document(value: &Value) -> ExperimentDocument {
    serde_json::from_value(value.clone()).unwrap()
}
fn absolute() -> Value {
    json!({"model_type":"gpt2","n_embd":8,"n_head":2,"n_layer":2,
        "n_positions":16,"vocab_size":17,"tie_word_embeddings":true})
}
fn dense() -> Value {
    json!({"model_type":"qwen2","hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,
        "num_attention_heads":2,"num_key_value_heads":1,"vocab_size":17,
        "max_position_embeddings":32,"hidden_act":"silu","rms_norm_eps":0.000001,
        "rope_theta":10000,"tie_word_embeddings":true})
}
fn preflight(value: &Value, profile: &str) -> Result<SourcePreflight, String> {
    preflight_source(&canonical_bytes(value), 3, 2, profile)
}
fn contract(e: &Value, c: &Value, rows: usize, outputs: usize) -> Result<Vec<u8>, String> {
    source_contract(
        &canonical_bytes(e),
        &canonical_bytes(c),
        rows,
        outputs,
        &"a".repeat(64),
        &"b".repeat(64),
    )
}

#[test]
fn audit_metadata_amplification_and_late_string_configs_reject_before_expansion() {
    for layers in [1024, 36] {
        let malicious = json!({"model_type":"qwen2","hidden_size":2,"intermediate_size":2,
            "num_hidden_layers":layers,"num_attention_heads":1,"num_key_value_heads":1,
            "vocab_size":3,"max_position_embeddings":16,"hidden_act":"silu",
            "rope_theta":10000,"rms_norm_eps":format!("0.000001{}", "0".repeat(15000)),
            "tie_word_embeddings":true});
        let bytes = canonical_bytes(&malicious);
        assert!(bytes.len() <= MAX_CONFIG_BYTES);
        let error = match preflight_source(&bytes, 1, 1, DENSE_ROPE_NUMERIC) {
            Ok(_) => panic!("amplification accepted"),
            Err(e) => e,
        };
        assert!(error.contains("string"), "{error}");
        // The raw 16 KiB config is the only sizeable owner. Never lower these
        // 1024 layers in a regression merely to observe the old rejection.
        assert!(bounded_json_value(&bytes).is_err());
    }
    let mut large = dense();
    large["num_hidden_layers"] = json!(1024);
    let budget = DecoderExpansionBudget {
        max_layers: 1024,
        max_feature_width: 32768,
        max_attention_heads: 32768,
        max_vocabulary: 32768,
        max_tensor_elements: 32768,
    };
    assert!(budget
        .preflight_metadata(
            &large,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 1,
                max_new_tokens: 1
            },
            MAX_NODES,
            1024,
            32768,
            1 << 20
        )
        .is_err());
}

#[test]
fn bounded_canonical_counting_rejects_before_building_output_and_preserves_identity() {
    let value = json!({"z":[true,1,1.5,"escaped\n"],"a":{"b":null}});
    assert_eq!(
        bounded_canonical(&value, 1024).unwrap(),
        canonical_bytes(&value)
    );
    assert!(bounded_canonical(&value, 4).is_err());
    // No giant output allocation: the counting writer stops at the first limit.
    struct Oversize;
    impl Serialize for Oversize {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeSeq;
            let mut seq = s.serialize_seq(None)?;
            for _ in 0..2_000_000 {
                seq.serialize_element(&0u8)?;
            }
            seq.end()
        }
    }
    assert!(bounded_canonical(&Oversize, 1024).is_err());
    let mut large = dense();
    large["num_hidden_layers"] = json!(36);
    large["rms_norm_eps"] = json!(format!("0.000001{}", "0".repeat(1016)));
    assert!(preflight(&large, DENSE_ROPE_NUMERIC).is_err());
}

#[test]
fn immutable_execution_binding_rejects_contract_substitution_not_just_digest_mismatch() {
    for (config, profile) in [(absolute(), GPT2_NUMERIC), (dense(), DENSE_ROPE_NUMERIC)] {
        let admitted = preflight(&config, profile).unwrap();
        let binding = SourceExecutionBinding::new(&admitted, 2).unwrap();
        let bytes = contract(&experiment(profile), &config, 3, 2).unwrap();
        assert_eq!(
            binding.checked_contract_digest(&bytes).unwrap(),
            <[u8; 32]>::from(Sha256::digest(&bytes))
        );
        // Inert metadata leaves semantic lowering unchanged, but exact config
        // bytes are available to this native constructor and must also bind.
        let mut other_config = config.clone();
        other_config["_name_or_path"] = json!("different-inert-source-label");
        let other_contract = contract(&experiment(profile), &other_config, 3, 2).unwrap();
        assert!(binding.checked_contract_digest(&other_contract).is_err());
        assert!(binding.checked_contract_digest(b"x").is_err());
        assert!(binding.checked_contract_digest(b"{}").is_err());
        assert!(binding
            .checked_contract_digest(&vec![b'x'; MAX_CONTRACT_BYTES + 1])
            .is_err());
        assert!(binding
            .checked_contract_digest(
                &serde_json::to_vec_pretty(&serde_json::from_slice::<Value>(&bytes).unwrap())
                    .unwrap()
            )
            .is_err());
        for (path, value) in [
            ("numeric_profile", json!("wrong-profile")),
            ("input_rows", json!(2)),
            ("output_tokens", json!(1)),
            ("width", json!(999)),
            ("vocabulary", json!(999)),
            ("schema_version", json!("other")),
            ("state_policy", json!("kv-cache")),
            ("setup_owner", json!("client")),
            ("ring_bits", json!(64)),
            ("weights", json!("private")),
            ("source_float32_bytes", json!(1)),
        ] {
            let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
            changed[path] = value;
            assert!(
                binding
                    .checked_contract_digest(&canonical_bytes(&changed))
                    .is_err(),
                "{path}"
            );
        }
        let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
        changed["plan"]["config_digest"] = json!("f".repeat(64));
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&changed))
            .is_err());
        let mut inventory: Value = serde_json::from_slice(&bytes).unwrap();
        *inventory["source_shapes"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap() = json!([1]);
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&inventory))
            .is_err());
        let mut rotary: Value = serde_json::from_slice(&bytes).unwrap();
        rotary["rotary_shapes"] = json!({"forged":[1,2]});
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&rotary))
            .is_err());
        let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
        let layout = changed["source_layouts"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap();
        *layout = json!("wrong");
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&changed))
            .is_err());
        let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
        changed["unexpected"] = json!(true);
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&changed))
            .is_err());
        // Pins are format-checked authenticated-caller claims, not proven by a
        // tensor stream. Their valid substitution needs caller authentication.
        changed.as_object_mut().unwrap().remove("unexpected");
        changed["source_lock"] = json!("c".repeat(64));
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&changed))
            .is_ok());
    }
}

#[test]
fn exact_sdk_identities_and_compact_public_profile_params() {
    assert_eq!(PROTOCOL, "pllm/fusefss-source-decoder/v1");
    assert_eq!(TOPOLOGY, "pllm/trusted-dealer-two-share/v1");
    assert_eq!(STATE, "pllm/fixed-count-private-prefix/v1");
    for p in [GPT2_NUMERIC, DENSE_ROPE_NUMERIC] {
        let e = document(&experiment(p));
        assert!(selected(&e.pipeline));
        validate(&e).unwrap();
    }
    let mut old = experiment(GPT2_NUMERIC);
    old["pipeline"]["components"] =
        json!({"linear":{"component":"pllm/masked-linear","params":{}}});
    assert!(!selected(&document(&old).pipeline));
    for slot in ["linear", "topology", "state", "wrong-slot"] {
        let mut partial = old.clone();
        partial["pipeline"]["components"][slot] = json!({"component":STATE,"params":{}});
        assert!(selected(&document(&partial).pipeline));
        assert!(validate(&document(&partial)).is_err());
    }
}

#[test]
fn rejects_noncanonical_params_client_dealer_extra_slots_and_wrong_cpu() {
    for (slot, key, value) in [
        (
            "linear",
            "numeric_profile",
            json!("fusefss.source-derived-ring50-q12.v1"),
        ),
        ("linear", "numeric_profile", json!(false)),
        ("linear", "weights", json!("private")),
        ("linear", "weights", json!("public")), // SDK omits the immutable public default.
        ("linear", "range_policy", json!("unchecked")),
        ("topology", "setup_owner", json!("client")),
        ("topology", "client_is_dealer", json!(true)),
        ("state", "kv_cache", json!(true)),
        ("state", "streaming", json!(true)),
        ("state", "early_exit", json!(true)),
        ("state", "generation_count", json!(2)),
        ("kernels", "threads", json!(2)),
        ("kernels", "threads", json!(1.0)),
        ("kernels", "threads", json!(true)),
    ] {
        let mut e = experiment(GPT2_NUMERIC);
        e["pipeline"]["components"][slot]["params"][key] = value;
        assert!(validate(&document(&e)).is_err(), "{slot}:{key}");
    }
    for slot in [
        "cache",
        "verification",
        "preparation",
        "dealer",
        "nonlinear",
        "extra",
    ] {
        let mut e = experiment(GPT2_NUMERIC);
        e["pipeline"]["components"][slot] = json!({"component":STATE,"params":{}});
        assert!(validate(&document(&e)).is_err(), "{slot}");
    }
    for (slot, id) in [
        ("linear", "pllm/puma-three-party-decoder/v1"),
        ("topology", "pllm/two-online-offset-workers/v1"),
        ("state", "pllm/client-local-kv"),
        ("kernels", "pllm/apple-metal-int8/v1"),
    ] {
        let mut e = experiment(GPT2_NUMERIC);
        e["pipeline"]["components"][slot]["component"] = json!(id);
        assert!(validate(&document(&e)).is_err());
    }
    let mut missing = experiment(GPT2_NUMERIC);
    missing["pipeline"]["components"]["linear"]["params"] = json!({});
    assert!(validate(&document(&missing)).is_err());
}

#[test]
fn native_request_source_and_local_topology_envelope_is_bounded() {
    for kind in ["huggingface", "safetensors"] {
        let mut e = experiment(GPT2_NUMERIC);
        e["pipeline"]["model"]["kind"] = json!(kind);
        validate(&document(&e)).unwrap();
    }
    for kind in ["tiny", "vllm", "gguf", "mlx", "ollama"] {
        let mut e = experiment(GPT2_NUMERIC);
        e["pipeline"]["model"]["kind"] = json!(kind);
        assert!(validate(&document(&e)).is_err());
    }
    for (field, n) in [
        ("requests", 0),
        ("requests", 65),
        ("max_input_tokens", 0),
        ("max_input_tokens", 17),
        ("max_new_tokens", 0),
        ("max_new_tokens", 5),
        ("max_input_tokens", u64::MAX),
    ] {
        let mut e = experiment(GPT2_NUMERIC);
        e["budget"][field] = json!(n);
        assert!(validate(&document(&e)).is_err(), "{field}:{n}");
    }
    let mut max = experiment(GPT2_NUMERIC);
    max["budget"] = json!({"requests":64,"max_input_tokens":13,"max_new_tokens":4});
    validate(&document(&max)).unwrap();
    max["budget"]["max_input_tokens"] = json!(14);
    assert!(validate(&document(&max)).is_err());
    let mut e = experiment(GPT2_NUMERIC);
    e["schema"] = json!("pllm.experiment.v3");
    assert!(validate(&document(&e)).is_err());
    e["deployment"] = json!({"kind":"network","network_id":"fss-test-network","network_spec_digest":"a".repeat(64),"snapshot_digest":null});
    assert!(validate(&document(&e)).is_err());
    let mut e = experiment(GPT2_NUMERIC);
    e["pipeline"]["model"]["endpoint"] = json!("http://remote.invalid");
    assert!(validate(&document(&e)).is_err());
    let mut e = experiment(GPT2_NUMERIC);
    e["deployment"]["root"] = json!("");
    assert!(validate(&document(&e)).is_err());
}

#[test]
fn learned_positions_conv1d_bias_norm_and_tied_source_shapes_are_explicit() {
    for tied in [true, false] {
        let mut c = absolute();
        c["tie_word_embeddings"] = json!(tied);
        let p = preflight(&c, GPT2_NUMERIC).unwrap();
        assert_eq!((p.width, p.vocabulary, p.maximum_sequence), (8, 17, 4));
        assert_eq!(p.source_shapes["transformer.wte.weight"], vec![17, 8]);
        assert_eq!(p.source_shapes["transformer.wpe.weight"], vec![16, 8]);
        assert_eq!(
            p.source_shapes["transformer.h.0.attn.c_attn.weight"],
            vec![8, 24]
        );
        assert_eq!(
            p.source_shapes["transformer.h.0.attn.c_attn.bias"],
            vec![24]
        );
        assert_eq!(
            p.source_layouts["transformer.h.0.attn.c_attn.weight"],
            "input_output"
        );
        assert_eq!(
            p.source_shapes["transformer.h.0.mlp.c_fc.weight"],
            vec![8, 32]
        );
        assert_eq!(
            p.source_shapes["transformer.h.0.mlp.c_proj.weight"],
            vec![32, 8]
        );
        for name in [
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.ln_1.bias",
            "transformer.h.0.ln_2.bias",
            "transformer.ln_f.bias",
        ] {
            assert_eq!(p.source_shapes[name], vec![8]);
        }
        assert_eq!(p.source_shapes.contains_key("lm_head.weight"), !tied);
        if !tied {
            assert_eq!(p.source_shapes["lm_head.weight"], vec![17, 8]);
            assert_eq!(p.source_layouts["lm_head.weight"], "output_input");
        }
        assert!(p.rotary_shapes.is_empty());
        assert_eq!(p.rotary_float32_bytes, 0);
        let values: usize = p
            .source_shapes
            .values()
            .map(|s| s.iter().product::<usize>())
            .sum();
        assert_eq!(p.source_float32_bytes, values * 4);
        assert_eq!(p.source_q12_value_bytes, values * 8);
        assert!(p.source_minimum_owner_bytes > p.source_q12_value_bytes);
    }
}

#[test]
fn dense_rope_gqa_qk_norm_and_bias_free_controls_use_semantic_capabilities() {
    let c = dense();
    let p = preflight(&c, DENSE_ROPE_NUMERIC).unwrap();
    assert_eq!(
        p.source_shapes["model.layers.0.self_attn.q_proj.weight"],
        vec![8, 8]
    );
    assert_eq!(
        p.source_shapes["model.layers.0.self_attn.k_proj.weight"],
        vec![4, 8]
    );
    assert_eq!(
        p.source_shapes["model.layers.0.self_attn.k_proj.bias"],
        vec![4]
    );
    assert_eq!(
        p.source_shapes["model.layers.0.input_layernorm.weight"],
        vec![8]
    );
    assert_eq!(
        p.source_layouts["model.layers.0.self_attn.k_proj.weight"],
        "output_input"
    );
    assert_eq!(p.rotary_shapes["layer.0.rope_q"], vec![4, 4]);
    assert_eq!(p.rotary_shapes["layer.0.rope_k"], vec![4, 4]);
    assert_eq!(p.rotary_float32_bytes, 4 * 4 * 4 * 2 * 4);
    let mut qk = c.clone();
    qk["model_type"] = json!("qwen3");
    for (key, value) in [
        ("head_dim", json!(4)),
        ("attention_bias", json!(false)),
        ("attention_dropout", json!(0.0)),
        ("rope_scaling", Value::Null),
        ("sliding_window", Value::Null),
        ("use_sliding_window", json!(false)),
        ("use_cache", json!(true)),
        ("max_window_layers", json!(2)),
    ] {
        qk[key] = value;
    }
    let p = preflight(&qk, DENSE_ROPE_NUMERIC).unwrap();
    assert_eq!(
        p.source_shapes["model.layers.0.self_attn.q_norm.weight"],
        vec![4]
    );
    assert_eq!(
        p.source_shapes["model.layers.0.self_attn.k_norm.weight"],
        vec![4]
    );
    assert!(!p
        .source_shapes
        .contains_key("model.layers.0.self_attn.q_proj.bias"));
    let mut bias_free = c;
    bias_free["model_type"] = json!("llama");
    let p = preflight(&bias_free, DENSE_ROPE_NUMERIC).unwrap();
    assert!(!p.source_shapes.keys().any(|s| s.ends_with(".bias")));
    assert!(!p.source_shapes.keys().any(|s| s.contains("q_norm")));
}

#[test]
fn rejects_profile_mismatch_unapproved_source_attributes_dtype_and_rotary() {
    assert!(preflight(&absolute(), DENSE_ROPE_NUMERIC).is_err());
    assert!(preflight(&dense(), GPT2_NUMERIC).is_err());
    assert!(preflight(&dense(), "fusefss.source-derived-ring50-q12.v1").is_err());
    for (key, value) in [
        ("rope_scaling", json!({"type":"linear","factor":2})),
        ("partial_rotary_factor", json!(0.5)),
        ("use_cache", json!(false)),
        ("attention_dropout", json!(0.1)),
        ("torch_dtype", json!("int8")),
        ("mlp_bias", json!(true)),
        ("extra_norm", json!(true)),
        ("attention_bias", json!(false)),
        ("head_dim", json!(8)),
        (
            "layer_types",
            json!(["sliding_attention", "full_attention"]),
        ),
    ] {
        let mut c = dense();
        c[key] = value;
        assert!(preflight(&c, DENSE_ROPE_NUMERIC).is_err(), "{key}");
    }
    let mut odd = dense();
    odd["hidden_size"] = json!(6);
    assert!(preflight(&odd, DENSE_ROPE_NUMERIC).is_err());
}

#[test]
fn tensor_geometry_total_source_owner_and_expanded_views_reject_before_reads() {
    assert!(preflight_source(&[], 3, 2, GPT2_NUMERIC).is_err());
    assert!(preflight_source(&vec![b' '; MAX_CONFIG_BYTES + 1], 3, 2, GPT2_NUMERIC).is_err());
    let mut exact = canonical_bytes(&absolute());
    exact.resize(MAX_CONFIG_BYTES, b' ');
    assert!(preflight_source(&exact, 3, 2, GPT2_NUMERIC).is_ok());
    let mut full_size = absolute();
    full_size["n_embd"] = json!(768);
    full_size["n_head"] = json!(12);
    full_size["n_layer"] = json!(12);
    full_size["vocab_size"] = json!(50257);
    full_size["n_positions"] = json!(1024);
    assert!(preflight(&full_size, GPT2_NUMERIC).is_err()); // Metadata only; no checkpoint reads.
    for (rows, outputs) in [(0, 1), (1, 0), (1, 5), (16, 2), (usize::MAX, 4)] {
        assert!(
            preflight_source(&canonical_bytes(&absolute()), rows, outputs, GPT2_NUMERIC).is_err()
        );
    }
    let mut c = absolute();
    c["n_embd"] = json!(128);
    c["n_head"] = json!(8);
    c["n_inner"] = json!(128);
    assert!(preflight(&c, GPT2_NUMERIC)
        .err()
        .unwrap()
        .contains("MAX_VALUES"));
    let mut c = dense();
    c["hidden_size"] = json!(80);
    c["num_attention_heads"] = json!(4);
    c["num_key_value_heads"] = json!(4);
    c["intermediate_size"] = json!(160);
    c["num_hidden_layers"] = json!(20);
    assert!(preflight(&c, DENSE_ROPE_NUMERIC)
        .err()
        .unwrap()
        .contains("MAX_BYTES"));
    let mut c = absolute();
    c["n_embd"] = json!(129);
    c["n_head"] = json!(129);
    c["n_inner"] = json!(8);
    c["n_layer"] = json!(1);
    assert!(preflight_source(&canonical_bytes(&c), 13, 4, GPT2_NUMERIC)
        .err()
        .unwrap()
        .contains("MAX_VALUES"));
    assert!(count(&[u64::MAX, 2]).is_err());
    let mut shapes = BTreeMap::new();
    let mut layouts = BTreeMap::new();
    bind(
        Capacity::Capsule,
        &mut shapes,
        &mut layouts,
        "shared",
        &[2, 4],
        "output_input",
    )
    .unwrap();
    assert!(bind(
        Capacity::Capsule,
        &mut shapes,
        &mut layouts,
        "shared",
        &[4, 2],
        "output_input"
    )
    .is_err());
    assert!(bind(
        Capacity::Capsule,
        &mut shapes,
        &mut layouts,
        "shared",
        &[2, 4],
        "input_output"
    )
    .is_err());
}

#[test]
fn mutated_semantic_ops_shapes_dtype_orientation_and_unapproved_numeric_attrs_reject() {
    let p = preflight(&absolute(), GPT2_NUMERIC).unwrap().plan;
    for (id, key, value) in [
        ("position_lookup", "weight_shape", json!([3, 8])),
        ("layer.0.qkv_linear", "weight_layout", json!("output_input")),
        ("layer.0.input_norm", "bias_shape", json!([7])),
        ("output_head", "tied", json!(false)),
        ("layer.0.qkv_linear", "compute_dtype", json!("int8")),
        ("layer.0.gelu", "numeric_profile", json!("other")),
    ] {
        let mut bad = p.clone();
        bad.prefill
            .operations
            .iter_mut()
            .find(|v| v.id == id)
            .unwrap()
            .attributes[key] = value;
        assert!(preflight_plan(bad, 2, GPT2_NUMERIC).is_err(), "{id}:{key}");
    }
    let mut bad = p;
    bad.prefill.operations[0].operator = Op::SecureGather;
    assert!(preflight_plan(bad, 2, GPT2_NUMERIC).is_err());
    let p = preflight(&dense(), DENSE_ROPE_NUMERIC).unwrap().plan;
    for (key, value) in [
        ("frequency_scaling", json!({"factor":2})),
        ("coefficient_profile", json!("arbitrary")),
        ("rotary_dimensions", json!(2)),
        ("pairing", json!("interleaved")),
    ] {
        let mut bad = p.clone();
        bad.prefill
            .operations
            .iter_mut()
            .find(|v| v.operator == Op::RotaryEmbedding)
            .unwrap()
            .attributes[key] = value;
        assert!(preflight_plan(bad, 2, DENSE_ROPE_NUMERIC).is_err());
    }
}

#[test]
fn canonical_metadata_binds_pins_composition_source_shapes_and_explicit_unknowns() {
    for (profile, c) in [(GPT2_NUMERIC, absolute()), (DENSE_ROPE_NUMERIC, dense())] {
        let e = experiment(profile);
        let bytes = contract(&e, &c, 3, 2).unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(canonical_bytes(&v), bytes);
        assert_eq!(v["schema_version"], "pllm.fss.sdk-contract.v1");
        assert_eq!(v["numeric_profile"], profile);
        assert_eq!(v["weights"], "public");
        assert_eq!(v["setup_owner"], "dealer");
        assert_eq!(v["source_lock"], "a".repeat(64));
        assert_eq!(v["tokenizer"], "b".repeat(64));
        assert_eq!(
            v["composition"],
            serde_json::to_value(crate::pipeline_digest(&document(&e).pipeline)).unwrap()
        );
        assert_eq!(
            v["source_shapes"],
            serde_json::to_value(preflight(&c, profile).unwrap().source_shapes).unwrap()
        );
        assert_eq!(v["selection_prefix_rows"], json!([3, 4]));
        assert_eq!(v["output_tokens"], 2);
        assert_eq!(v["vocabulary"], 17);
        assert_eq!(v["width"], 8);
        assert_eq!(v["persistent_kv"], false);
        assert_eq!(v["admission"]["numeric"], "not_performed");
        assert_eq!(v["admission"]["serving_qualified"], false);
        for key in [
            "source_owner_reservation_bytes",
            "rotary_f32_payloads",
            "range_obligations",
            "domain_admission",
            "program_resources",
            "transport_resources",
            "complete_live_accounting",
        ] {
            assert_eq!(v["admission"][key], Value::Null);
        }
        assert_eq!(bytes, contract(&e, &c, 3, 2).unwrap());
        assert_ne!(
            bytes,
            source_contract(
                &canonical_bytes(&e),
                &canonical_bytes(&c),
                3,
                2,
                &"c".repeat(64),
                &"b".repeat(64)
            )
            .unwrap()
        );
        let mut changed = e;
        changed["name"] = json!("changed-configuration");
        assert_ne!(bytes, contract(&changed, &c, 3, 2).unwrap());
    }
}

#[test]
fn metadata_rejects_token_source_pins_noncanonical_json_and_oversized_saved_envelopes() {
    let e = experiment(GPT2_NUMERIC);
    let c = absolute();
    assert!(source_contract(
        &[],
        &canonical_bytes(&c),
        3,
        2,
        &"a".repeat(64),
        &"b".repeat(64)
    )
    .is_err());
    assert!(source_contract(
        &vec![b' '; MAX_EXPERIMENT_BYTES + 1],
        &canonical_bytes(&c),
        3,
        2,
        &"a".repeat(64),
        &"b".repeat(64)
    )
    .is_err());
    for bad in [
        "".to_owned(),
        "a".repeat(63),
        "A".repeat(64),
        "0".repeat(64),
        "g".repeat(64),
    ] {
        assert!(source_contract(
            &canonical_bytes(&e),
            &canonical_bytes(&c),
            3,
            2,
            &bad,
            &"b".repeat(64)
        )
        .is_err());
        assert!(source_contract(
            &canonical_bytes(&e),
            &canonical_bytes(&c),
            3,
            2,
            &"a".repeat(64),
            &bad
        )
        .is_err());
    }
    for (rows, outputs) in [(4, 2), (3, 3), (0, 1), (1, 0)] {
        assert!(contract(&e, &c, rows, outputs).is_err());
    }
    let mut bytes = canonical_bytes(&e);
    bytes.push(b'\n');
    assert!(source_contract(
        &bytes,
        &canonical_bytes(&c),
        3,
        2,
        &"a".repeat(64),
        &"b".repeat(64)
    )
    .is_err());
    let mut small_table = c;
    small_table["n_positions"] = json!(3);
    assert!(contract(&e, &small_table, 1, 1).is_err()); // Actual query fits; declared budget does not.
    assert!(contract(&e, &absolute(), 1, 1).is_ok());
}
