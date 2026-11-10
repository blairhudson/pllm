use pllm_compiler::{resolve_experiment, GkmDecoderContract, GkmRuntimeContract};
use pllm_types::{canonical_bytes, digest_bytes, Digest};
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};

fn experiment() -> Value {
    json!({"schema":"pllm.experiment.v2", "name":"delegation-control",
        "pipeline":{"model":{"source":"fixture", "kind":"safetensors"},
        "components":{
            "kernels":{"component":"pllm/cpu", "params":{"threads":1}},
            "linear":{"component":"pllm/open-weight-verifiable-outsourcing/v1",
                "params":{"sharing":"full", "generation":"fixed_length_greedy"}},
            "verification":{"component":"pllm/published-linear-verify/v1", "params":{"acceptance":"request"}},
            "topology":{"component":"pllm/two-online-delegation-workers/v1", "params":{}},
            "state":{"component":"pllm/client-local-kv", "params":{}}
        }}, "deployment":{"kind":"local", "root":"./fixture-run"},
        "budget":{"requests":1, "max_input_tokens":2, "max_new_tokens":2}})
}

fn config() -> Value {
    json!({"model_type":"qwen2", "hidden_size":8, "intermediate_size":16,
        "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
        "vocab_size":32,"max_position_embeddings":16,"hidden_act":"silu",
        "rms_norm_eps":1e-6,"rope_theta":10000.,"tie_word_embeddings":true})
}

fn explicit_bindings(mut e: Value) -> Value {
    for (slot, identity) in [
        ("input", "pllm/two-share-input/v1"),
        ("lookup", "pllm/verified-public-f32-lookup/v1"),
        ("regions", "pllm/client-f32-decoder-regions/v1"),
        ("numeric", "pllm/q11-i16-ring64-boundary/v1"),
    ] {
        e["pipeline"]["components"][slot] = json!({"component": identity, "params": {}});
    }
    e
}

#[test]
fn explicit_bindings_preserve_decoder_and_source_but_change_sdk_pins() {
    for sharing in ["full", "seeded"] {
        let mut legacy = experiment();
        legacy["pipeline"]["components"]["linear"]["params"]["sharing"] = json!(sharing);
        let original = canonical_bytes(&legacy);
        let explicit = explicit_bindings(legacy.clone());
        let old_resolved = resolve_experiment(&original).unwrap();
        let new_resolved = resolve_experiment(&canonical_bytes(&explicit)).unwrap();
        assert_ne!(old_resolved.configuration_digest(), new_resolved.configuration_digest());
        assert_ne!(old_resolved.composition_digest(), new_resolved.composition_digest());
        let old = compile(&legacy, &config(), 2, 2).unwrap();
        let new = compile(&explicit, &config(), 2, 2).unwrap();
        assert_eq!(old.decoder().canonical_bytes(), new.decoder().canonical_bytes());
        assert_eq!(old.decoder().source_binding_digest(), new.decoder().source_binding_digest());
        assert_ne!(old.digest(), new.digest());
        assert_eq!(canonical_bytes(&legacy), original);
        assert_eq!(old.canonical_bytes(), compile(&legacy, &config(), 2, 2).unwrap().canonical_bytes());
    }
}

#[test]
fn partial_unknown_parameterized_or_incompatible_bindings_reject_before_config() {
    for slot in ["input", "lookup", "regions", "numeric", "linear", "state", "kernels", "verification", "topology"] {
        let mut e = explicit_bindings(experiment());
        e["pipeline"]["components"].as_object_mut().unwrap().remove(slot);
        assert!(resolve_experiment(&canonical_bytes(&e)).is_err());
        let bad_config = compile(&e, &json!({}), 1, 1).err().unwrap();
        assert_eq!(bad_config, compile(&e, &config(), 1, 1).err().unwrap());
    }
    for slot in ["input", "lookup", "regions", "numeric"] {
        let mut e = explicit_bindings(experiment());
        e["pipeline"]["components"][slot]["params"]["sharing"] = json!("seeded");
        assert!(resolve_experiment(&canonical_bytes(&e)).is_err());
        let mut e = explicit_bindings(experiment());
        e["pipeline"]["components"][slot]["component"] = json!("unknown/implementation/v1");
        assert!(resolve_experiment(&canonical_bytes(&e)).is_err());
    }
    let mut mixed = explicit_bindings(experiment());
    mixed["pipeline"]["components"]["numeric"]["component"] = json!("pllm/source-float32/v1");
    assert!(resolve_experiment(&canonical_bytes(&mixed)).is_err());
}

fn compile(e: &Value, c: &Value, rows: usize, tokens: usize) -> Result<GkmRuntimeContract, String> {
    GkmRuntimeContract::compile(
        &canonical_bytes(e),
        &canonical_bytes(c),
        rows,
        tokens,
        Digest::from_sha256([1; 32]),
        Digest::from_sha256([2; 32]),
    )
}

#[test]
fn canonical_experiment_and_source_bound_actual_request() {
    let e = experiment();
    let resolved = resolve_experiment(&canonical_bytes(&e)).unwrap();
    let contract = compile(&e, &config(), 2, 2).unwrap();
    let value: Value = serde_json::from_slice(&contract.canonical_bytes()).unwrap();
    assert_eq!(value["composition"], json!(resolved.composition_digest()));
    assert_eq!(
        value["configuration"],
        json!(resolved.configuration_digest())
    );
    assert_eq!(value["decoder"]["decode_steps"], 1);
    assert_ne!(
        compile(&e, &config(), 1, 1).unwrap().digest(),
        contract.digest()
    );
    let changed = GkmRuntimeContract::compile(
        &canonical_bytes(&e),
        &canonical_bytes(&config()),
        2,
        2,
        Digest::from_sha256([3; 32]),
        Digest::from_sha256([2; 32]),
    )
    .unwrap();
    assert_ne!(contract.digest(), changed.digest());
}

#[test]
fn incompatible_and_incomplete_compositions_never_select_int8_runtime() {
    for slot in ["linear", "verification", "topology", "kernels", "state"] {
        let mut e = experiment();
        e["pipeline"]["components"]
            .as_object_mut()
            .unwrap()
            .remove(slot);
        assert!(resolve_experiment(&canonical_bytes(&e)).is_err(), "{slot}");
    }
    for (slot, key, value) in [
        ("linear", "sharing", json!("unknown")),
        ("linear", "generation", json!("sampled")),
        ("verification", "acceptance", json!("token")),
        ("kernels", "threads", json!(2)),
    ] {
        let mut e = experiment();
        e["pipeline"]["components"][slot]["params"][key] = value;
        assert!(resolve_experiment(&canonical_bytes(&e)).is_err());
    }
    let mut e = experiment();
    e["pipeline"]["components"]["quantization"] =
        json!({"component":"pllm/symmetric-per-row", "params":{}});
    assert!(resolve_experiment(&canonical_bytes(&e)).is_err());
}

#[test]
fn declared_envelope_and_source_preflight_precede_actual_small_query() {
    let mut e = experiment();
    e["budget"]["max_input_tokens"] = json!(33);
    assert!(compile(&e, &config(), 1, 1).is_err());
    let mut c = config();
    c["num_hidden_layers"] = json!(1_000_000_000);
    assert!(compile(&experiment(), &c, 1, 1).is_err());
    for (rows, tokens) in [(0, 1), (1, 0), (3, 1), (1, 3)] {
        assert!(compile(&experiment(), &config(), rows, tokens).is_err());
    }
    let mut e = experiment();
    e["budget"]["max_input_tokens"] = json!(32);
    e["budget"]["max_new_tokens"] = json!(9);
    // Per-field limits fit, but the complete native scheduled window does not.
    assert!(compile(&e, &config(), 1, 1).is_err());
}

#[test]
fn explicit_decoder_capacity_preserves_default_and_admits_full_source_geometry() {
    let original = experiment();
    let legacy = compile(&original, &config(), 2, 2).unwrap();
    let legacy_json: Value = serde_json::from_slice(&legacy.canonical_bytes()).unwrap();
    assert!(legacy_json["decoder"].get("decoder_resources").is_none());
    let mut expanded = original.clone();
    expanded["pipeline"]["components"]["linear"]["params"]["capacity"] = json!("decoder");
    let changed = compile(&expanded, &config(), 2, 2).unwrap();
    assert_ne!(legacy.digest(), changed.digest());
    assert_eq!(
        legacy.digest(),
        compile(&original, &config(), 2, 2).unwrap().digest()
    );
    let full = json!({"model_type":"qwen2", "hidden_size":896, "intermediate_size":4864,
        "num_hidden_layers":24,"num_attention_heads":14,"num_key_value_heads":2,
        "vocab_size":151936,"max_position_embeddings":32768,"hidden_act":"silu",
        "rms_norm_eps":1e-6,"rope_theta":1000000.,"tie_word_embeddings":true});
    expanded["budget"]["max_input_tokens"] = json!(150);
    expanded["budget"]["max_new_tokens"] = json!(8);
    let admitted = compile(&expanded, &full, 150, 8).unwrap();
    let value: Value = serde_json::from_slice(&admitted.canonical_bytes()).unwrap();
    let decoder = &value["decoder"];
    let resources = &decoder["decoder_resources"];
    assert_eq!(decoder["matrices"].as_array().unwrap().len(), 97);
    assert_eq!(decoder["source_shapes"].as_object().unwrap().len(), 290);
    assert_eq!(resources["profile"], "pllm.gkm.decoder-capacity.v1");
    assert!(resources["source_import_bytes"].as_u64().unwrap() > 1_532_461_056);
    assert!(resources["worker_payload_bytes"].as_u64().unwrap() > 1_532_461_056);
    assert!(
        resources["client_payload_bytes"].as_u64().unwrap()
            > resources["worker_payload_bytes"].as_u64().unwrap()
    );
    assert!(resources["client_payload_bytes"].as_u64().unwrap() < 4u64 << 30);
    assert_eq!(resources["source_chunk_bytes"], 65536);
    assert!(compile(&original, &full, 1, 1).is_err());
    expanded["budget"]["max_input_tokens"] = json!(513);
    assert!(compile(&expanded, &full, 1, 1).is_err());
}

#[test]
fn decoder_capacity_must_be_explicit_and_seeded_sharing_is_separately_bound() {
    for capacity in [json!("capsule"), json!("unknown"), json!(false), json!(1)] {
        let mut e = experiment();
        e["pipeline"]["components"]["linear"]["params"]["capacity"] = capacity;
        assert!(resolve_experiment(&canonical_bytes(&e)).is_err());
    }
    let mut e = experiment();
    e["pipeline"]["components"]["linear"]["params"]["capacity"] = json!("decoder");
    e["pipeline"]["components"]["linear"]["params"]["sharing"] = json!("seeded");
    assert!(resolve_experiment(&canonical_bytes(&e)).is_ok());
    let seeded = compile(&e, &config(), 2, 2).unwrap();
    assert_eq!(
        seeded.sharing_profile(),
        pllm_core::gkm::SharingProfile::Aes128WindowV1
    );
    let value: Value = serde_json::from_slice(&seeded.canonical_bytes()).unwrap();
    assert_eq!(
        value["sharing_profile"],
        pllm_core::gkm::SEEDED_SHARING_PROFILE
    );
    e["pipeline"]["components"]["linear"]["params"]["sharing"] = json!("full");
    let full = compile(&e, &config(), 2, 2).unwrap();
    assert_ne!(full.digest(), seeded.digest());
    let value: Value = serde_json::from_slice(&full.canonical_bytes()).unwrap();
    assert!(value.get("sharing_profile").is_none());
}

// Fixed metadata inputs and pins from independent installed-wheel baseline capture.
// Synthetic source-lock/tokenizer pins do not validate source weights. No tensors run.
const LEGACY_TINY_FULL: &[u8] = br#"{"budget":{"max_input_tokens":3,"max_new_tokens":3,"requests":1},"deployment":{"kind":"local","root":".pllm/golden"},"name":"gkm-golden-full","pipeline":{"components":{"kernels":{"component":"pllm/cpu","params":{"threads":1}},"linear":{"component":"pllm/open-weight-verifiable-outsourcing/v1","params":{"capacity":"decoder","generation":"fixed_length_greedy","sharing":"full"}},"state":{"component":"pllm/client-local-kv","params":{}},"topology":{"component":"pllm/two-online-delegation-workers/v1","params":{}},"verification":{"component":"pllm/published-linear-verify/v1","params":{"acceptance":"request"}}},"model":{"local_files_only":true,"source":"fixture"}},"schema":"pllm.experiment.v2"}"#;
const LEGACY_TINY_SEEDED: &[u8] = br#"{"budget":{"max_input_tokens":3,"max_new_tokens":3,"requests":1},"deployment":{"kind":"local","root":".pllm/golden"},"name":"gkm-golden-seeded","pipeline":{"components":{"kernels":{"component":"pllm/cpu","params":{"threads":1}},"linear":{"component":"pllm/open-weight-verifiable-outsourcing/v1","params":{"capacity":"decoder","generation":"fixed_length_greedy","sharing":"seeded"}},"state":{"component":"pllm/client-local-kv","params":{}},"topology":{"component":"pllm/two-online-delegation-workers/v1","params":{}},"verification":{"component":"pllm/published-linear-verify/v1","params":{"acceptance":"request"}}},"model":{"local_files_only":true,"source":"fixture"}},"schema":"pllm.experiment.v2"}"#;

#[test]
fn legacy_factory_documents_match_fixed_baseline_pins() {
    for (tiny, name, raw_sha, configuration, composition) in [
        (LEGACY_TINY_FULL, "gkm",
         "348ac9eb0e2db6f9dd380d7dea22f11570620dd8a58675f9a1e5ae1fde228b3b",
         "aa831fd8c943917f7e6feab8573daca1246afc44361e4a32fe64b6382615a69b",
         "864edfe83172d2b39d4c0c801e0dae6b3edc24f3714ad0d28d36a4272707be90"),
        (LEGACY_TINY_SEEDED, "gkm-seeded",
         "e298986a2d649e32637df1ea6084e978b33d6410d52331f700518dd09b7178b8",
         "a2f067b9363670c6b2a3df668504e8b0adf563b70efb617b5297eec01935ac3a",
         "c7a81a5eaeb518f192dc2d2e92d58f0220b2d4d10a4797c02f0e1ce95897bc4e"),
    ] {
        let mut e: Value = serde_json::from_slice(tiny).unwrap();
        e["name"] = json!(name);
        e["budget"] = json!({"requests":32,"max_input_tokens":64,"max_new_tokens":8});
        e["deployment"]["root"] = json!(".pllm/papers/gkm");
        e["pipeline"]["model"] = json!({
            "local_files_only":true,
            "revision":"7ae557604adf67be50417f59c2c2f167def9a775",
            "source":"Qwen/Qwen2.5-0.5B-Instruct"
        });
        let raw = canonical_bytes(&e);
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), raw_sha);
        let resolved = resolve_experiment(&raw).unwrap();
        assert_eq!(json!(resolved.configuration_digest()), json!(configuration));
        assert_eq!(json!(resolved.composition_digest()), json!(composition));
    }
}

#[test]
fn legacy_metadata_contracts_match_fixed_baseline_bytes_and_domains() {
    for (tiny, rows, outputs, sdk_sha, sdk_digest) in [
        (LEGACY_TINY_FULL, 3, 3,
         "04c7199f61784a81b49d91badbddb449cae39aeb26a3d7bd140a613a8a0178c6",
         "950598afb3c92ab5f03e09a29a9cf751021507d01010289bf00f94d5a40b5894"),
        (LEGACY_TINY_FULL, 1, 1,
         "ddddb2489860018d90b8e3736c1d9be07ad9799125cb1003a881093aacdeeded",
         "48480b83a96b25f6befcf000b1572d6ec904e7e41cee509eefb2588ade7e3e59"),
        (LEGACY_TINY_SEEDED, 3, 3,
         "e3676a27d3d7cce37d47baa1ccf94ff396e7052061080ec8139335b3e716c3d3",
         "c2c4558c4f892eab91b83c909c1dbe0ff42bdf89020c8a7f63ad2402db347cf6"),
        (LEGACY_TINY_SEEDED, 1, 1,
         "2cbbe6330849209e557381594ecc4206d1e4e5fcfeef22c8f7cc75d373c3e285",
         "9c9eee1b8e636bd5f0ac01d5be942bc770155afadcbdeda5c524b51788d98397"),
    ] {
        // Source binding is PRE-streamed, not PRE-request-narrowing. Each geometry
        // has its own independently captured ordinary-decoder identity.
        let (ordinary_sha, ordinary_digest, streamed_sha, streamed_digest) = match (rows, outputs) {
            (3, 3) => (
                "4ef4eb5c61d8819e84e2f55303b375755936268d20e1c6f639672525824987b6",
                "8cd32aba63cc37b2f9d219f6ab7fe30c83c576450fa06436218bf4f566166a7e",
                "bafeb266ab504da232224740b5c5da11838e09aa55c24413e1089907ca20c1eb",
                "cf45d1561515d01ecd3fbf2922196e2a9a0b6b6160cd5d8b5b795044d3c68e69",
            ),
            (1, 1) => (
                "aa67bf6b1f60ba8518d02796de5dbeaa2a0a2f0e6e03dda2f379802e6e646483",
                "7ad2dc7c19707a2451c9f24089b79c6428818a53d3355eee4b538da1d0b634b8",
                "44d0e9f9678c1725c6a9623c096dae223410fbbf08a247a64854cec886cc21ad",
                "4c4bbd17baa8f9e944405c16ac81ce1a3131ef1e66cea6c38c93ae8753362940",
            ),
            _ => unreachable!(),
        };
        let cfg = canonical_bytes(&config());
        let ordinary = GkmDecoderContract::compile_with_capacity(
            &cfg, rows, outputs - 1, pllm_core::gkm::Capacity::Decoder,
        ).unwrap();
        let streamed = GkmDecoderContract::compile_streamed_with_capacity(
            &cfg, rows, outputs - 1, pllm_core::gkm::Capacity::Decoder,
        ).unwrap();
        let sdk = GkmRuntimeContract::compile(
            tiny, &cfg, rows, outputs,
            Digest::from_sha256([1; 32]), Digest::from_sha256([2; 32]),
        ).unwrap();
        for (contract, raw_sha, domain_digest) in [
            (&ordinary, ordinary_sha, ordinary_digest),
            (&streamed, streamed_sha, streamed_digest),
            (sdk.decoder(), streamed_sha, streamed_digest),
        ] {
            let raw = contract.canonical_bytes();
            assert_eq!(
                format!("{:x}", Sha256::digest(&raw)), raw_sha,
                "legacy decoder bytes: rows={rows} outputs={outputs} document={} kind={} input_config={} canonical_contract={}",
                if tiny == LEGACY_TINY_FULL { "gkm-golden-full" } else { "gkm-golden-seeded" },
                if std::ptr::eq(contract, &ordinary) {
                    "ordinary"
                } else if std::ptr::eq(contract, &streamed) {
                    "streamed"
                } else {
                    "sdk.decoder"
                },
                std::str::from_utf8(&cfg).unwrap(),
                std::str::from_utf8(&raw).unwrap(),
            );
            assert_eq!(json!(contract.digest()), json!(domain_digest));
            assert_eq!(json!(digest_bytes("pllm.gkm.decoder-reference.v1", &raw)), json!(domain_digest));
        }
        assert_eq!(json!(streamed.source_binding_digest()), json!(ordinary_digest));
        assert_eq!(json!(sdk.decoder().source_binding_digest()), json!(ordinary_digest));
        let raw = sdk.canonical_bytes();
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), sdk_sha);
        assert_eq!(json!(sdk.digest()), json!(sdk_digest));
        assert_eq!(json!(digest_bytes("pllm.gkm.sdk-contract.v1", &raw)), json!(sdk_digest));
    }
}
