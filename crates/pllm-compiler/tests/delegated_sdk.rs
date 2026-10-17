use pllm_compiler::{resolve_experiment, GkmRuntimeContract};
use pllm_types::{canonical_bytes, Digest};
use serde_json::{json, Value};

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
        ("linear", "sharing", json!("seeded")),
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
