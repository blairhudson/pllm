use pllm_compiler::{
    lower_decoder_runtime_schedule, network_placement_requirements, resolve_experiment,
    DecoderRuntimeSchedule,
};
use pllm_models::{lower_model_json, DecoderPlan, DecoderWorkload};
use pllm_types::canonical_bytes;
use serde_json::{json, Value};

fn plan() -> DecoderPlan {
    lower_model_json(
        br#"{"model_type":"qwen2","hidden_size":8,"intermediate_size":16,
        "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
        "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
        "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true}"#,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 3,
            max_new_tokens: 2,
        },
    )
    .unwrap()
}

fn composition(decode: bool, bytes: u64) -> Value {
    let mut value = json!({"model":{"source":"model.fixture"}, "components":{
        "inference":{"component":"pllm/inference","params":{}},
        "kernels":{"component":"pllm/cpu","params":{"threads":1}},
        "linear":{"component":"pllm/masked-linear","params":{}},
        "preparation":{"component":"pllm/model-aware-corrections","params":{}},
        "quantization":{"component":"pllm/symmetric-per-row-quantization/v1",
            "params":{"weight_bits":8,"activation_bits":8,"causal_reduction":"prefix_f32"}},
        "inventory":{"component":"pllm/prepared-inventory-policy/v1", "params":{
            "policy":"request-sized","rows":1,"refill":"on-demand","allocation":"demand","decode_epoch_tokens":1}},
        "cache":{"component":"pllm/client-prefix-reuse/v1", "params":{
            "max_bytes":1048576,"fixed_input_tokens":3,"generated_prefixes":true}}
    }});
    if decode {
        value["components"]["cache"]["params"]["decode"] = json!(true);
        value["components"]["cache"]["params"]["decode_max_bytes"] = json!(bytes);
    }
    value
}

fn experiment(composition: &Value) -> Value {
    json!({"schema":"pllm.experiment.v2","name":"decode-contract","pipeline":composition,
        "deployment":{"kind":"local","root":"local://decode"},
        "budget":{"requests":8,"max_input_tokens":3,"max_new_tokens":2}})
}

#[test]
fn decode_reuse_binds_immutable_policy_and_additional_memory_without_reducing_full_cap() {
    let model = plan();
    let baseline =
        lower_decoder_runtime_schedule(&model, &canonical_bytes(&composition(false, 0))).unwrap();
    assert!(baseline.decode_reuse.is_none());
    assert!(serde_json::to_value(&baseline)
        .unwrap()
        .get("decode_reuse")
        .is_none());
    let mut digests = std::collections::BTreeSet::new();
    for bytes in [8 << 20, 32 << 20, 128 << 20, 256 << 20] {
        let composition = composition(true, bytes);
        assert!(resolve_experiment(&canonical_bytes(&experiment(&composition))).is_ok());
        let schedule =
            lower_decoder_runtime_schedule(&model, &canonical_bytes(&composition)).unwrap();
        assert_eq!(schedule.prefill, baseline.prefill);
        assert_eq!(schedule.decode, baseline.decode);
        assert_ne!(schedule.digest(), baseline.digest());
        assert!(digests.insert(schedule.digest()));
        let contract = schedule.decode_reuse.as_ref().unwrap();
        assert_eq!(contract["decode_max_bytes"], json!(bytes));
        assert_eq!(contract["client_additional_memory_bytes"], json!(bytes));
        assert_eq!(
            contract["maximum_key_sequence"],
            json!(model.decode.maximum_key_sequence)
        );
        assert_eq!(contract["model_plan_digest"], json!(model.digest()));
        assert_eq!(contract["semantic_bounds"], json!("unchanged_full_cap"));
        let replay: DecoderRuntimeSchedule =
            serde_json::from_slice(&canonical_bytes(&schedule)).unwrap();
        assert_eq!(schedule, replay);
    }
}

#[test]
fn rejects_bad_limits_and_noncanonical_decode_options() {
    let model = plan();
    for (field, invalid) in [
        ("decode", json!(false)),
        ("decode", json!(1)),
        ("decode", json!(null)),
        ("generated_prefixes", json!(false)),
        ("decode_max_bytes", json!(0)),
        ("decode_max_bytes", json!(7 << 20)),
        ("decode_max_bytes", json!(257 << 20)),
        ("decode_max_bytes", json!((32 << 20) + 1)),
        ("decode_max_bytes", json!(33554432.0)),
        ("decode_max_bytes", json!(true)),
    ] {
        let mut value = composition(true, 128 << 20);
        value["components"]["cache"]["params"][field] = invalid;
        assert!(
            lower_decoder_runtime_schedule(&model, &canonical_bytes(&value)).is_err(),
            "{field}"
        );
        assert!(
            resolve_experiment(&canonical_bytes(&experiment(&value))).is_err(),
            "{field}"
        );
    }
    for field in ["decode", "decode_max_bytes", "generated_prefixes"] {
        let mut value = composition(true, 128 << 20);
        value["components"]["cache"]["params"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(
            lower_decoder_runtime_schedule(&model, &canonical_bytes(&value)).is_err(),
            "{field}"
        );
    }
}

#[test]
fn rejects_non_one_token_epochs_and_incompatible_compositions() {
    let model = plan();
    for tokens in [json!(0), json!(2), json!(true), json!(1.0)] {
        let mut value = composition(true, 128 << 20);
        value["components"]["inventory"]["params"]["decode_epoch_tokens"] = tokens;
        assert!(lower_decoder_runtime_schedule(&model, &canonical_bytes(&value)).is_err());
    }
    for key in ["decode_epoch_tokens", "allocation", "refill"] {
        let mut value = composition(true, 128 << 20);
        value["components"]["inventory"]["params"]
            .as_object_mut()
            .unwrap()
            .remove(key);
        assert!(lower_decoder_runtime_schedule(&model, &canonical_bytes(&value)).is_err());
    }
    let mut growing = composition(true, 128 << 20);
    growing["components"]["inventory"]["params"]
        .as_object_mut()
        .unwrap()
        .remove("decode_epoch_tokens");
    growing["components"]["inventory"]["params"]["decode_epoch_schedule"] = json!([1]);
    assert!(lower_decoder_runtime_schedule(&model, &canonical_bytes(&growing)).is_err());
    for slot in [
        "verification",
        "state",
        "scheduler",
        "nonlinear",
        "public_prefix",
    ] {
        let mut value = composition(true, 128 << 20);
        value["components"][slot] = json!({"component":"unsupported","params":{}});
        assert!(
            lower_decoder_runtime_schedule(&model, &canonical_bytes(&value)).is_err(),
            "{slot}"
        );
    }
    for (slot, component) in [
        ("kernels", "pllm/apple-metal-int8/v1"),
        ("linear", "pllm/two-online-offset-linear/v1"),
        ("linear", "pllm/cleartext-linear"),
    ] {
        let mut value = composition(true, 128 << 20);
        value["components"][slot]["component"] = json!(component);
        assert!(
            lower_decoder_runtime_schedule(&model, &canonical_bytes(&value)).is_err(),
            "{component}"
        );
    }
    let mut inflight = composition(true, 128 << 20);
    inflight["components"]["cache"]["params"]["inflight"] = json!(true);
    inflight["components"]["cache"]["params"]["inflight_max_requests"] = json!(8);
    inflight["components"]["cache"]["params"]["inflight_max_bytes"] = json!(1048576);
    assert!(lower_decoder_runtime_schedule(&model, &canonical_bytes(&inflight)).is_err());
}

#[test]
fn network_auto_planning_rejects_unowned_unpriced_decode_resources() {
    let model = plan();
    let composition = composition(true, 32 << 20);
    let error =
        network_placement_requirements(&canonical_bytes(&model), &canonical_bytes(&composition))
            .err()
            .expect("caller-owned decode cache cannot be automatically placed");
    assert!(
        error.contains("caller-owned decode cache resources"),
        "{error}"
    );
    let mut document = experiment(&composition);
    document["schema"] = json!("pllm.experiment.v3");
    document["deployment"] = json!({"kind":"network","network_id":"decode-network", "network_spec_digest":"1".repeat(64)});
    assert!(resolve_experiment(&canonical_bytes(&document)).is_err());
}

#[test]
fn rejects_nonexclusive_or_partial_native_kv_before_decode_reuse_admission() {
    let composition = canonical_bytes(&composition(true, 32 << 20));
    let mut shared = plan();
    shared.prefill.state_outputs[0].layer = shared.prefill.state_outputs[2].layer;
    assert!(lower_decoder_runtime_schedule(&shared, &composition).is_err());
    let mut partial = plan();
    partial.decode.state_inputs.pop();
    assert!(lower_decoder_runtime_schedule(&partial, &composition).is_err());
    let mut sliding = plan();
    sliding.prefill.state_outputs[0].maximum_sequence = 1;
    assert!(lower_decoder_runtime_schedule(&sliding, &composition).is_err());
}
