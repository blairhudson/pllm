use super::*;
use crate::{Qwen3Config, Qwen3ConfigField, QwenConfig, QwenConfigField};
use serde_json::json;

fn qwen2() -> Value {
    json!({
        "model_type": "qwen2", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "vocab_size": 258, "max_position_embeddings": 256, "hidden_act": "silu",
        "rms_norm_eps": "0.000001", "rope_theta": 10000, "tie_word_embeddings": false
    })
}

fn qwen3() -> Value {
    let mut value = qwen2();
    value["model_type"] = json!("qwen3");
    for (key, field) in [
        ("head_dim", json!(8)), ("attention_bias", json!(false)),
        ("attention_dropout", json!("0")), ("use_sliding_window", json!(false)),
        ("use_cache", json!(true)), ("max_window_layers", json!(2)),
    ] {
        value[key] = field;
    }
    value
}

fn llama() -> Value {
    json!({
        "model_type": "llama", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4,
        "vocab_size": 258, "max_position_embeddings": 256
    })
}

#[test]
fn emitted_types_keep_original_traits_and_field_types() {
    fn config<T: Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de>>() {}
    fn eq_config<T: Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de> + Eq + PartialEq>() {}
    eq_config::<QwenConfig>();
    eq_config::<Qwen3Config>();
    config::<SourceConfig>(); // No extra Eq/PartialEq contract on the f64 role.
    let q: QwenConfig = serde_json::from_value(qwen2()).unwrap();
    let _: (&String, &u64, &u32, &bool, &Option<u64>, &Option<bool>) = (
        &q.model_type, &q.hidden_size, &q.num_hidden_layers,
        &q.tie_word_embeddings, &q.sliding_window, &q.use_sliding_window,
    );
    let q: Qwen3Config = serde_json::from_value(qwen3()).unwrap();
    let _: (&String, &u64, &Option<Value>, &Option<Vec<String>>) = (
        &q.attention_dropout, &q.head_dim, &q.rope_scaling, &q.layer_types,
    );
    let l: SourceConfig = serde_json::from_value(llama()).unwrap();
    let _: (&f64, &Option<u32>, &Option<u64>, &Option<bool>) = (
        &l.attention_dropout, &l.num_key_value_heads, &l.head_dim, &l.use_cache,
    );
}

#[test]
fn qwen2_generated_private_field_ids_keep_source_order_and_names() {
    use QwenConfigField as F;
    let ids = [F::model_type, F::hidden_size, F::intermediate_size,
        F::num_hidden_layers, F::num_attention_heads, F::num_key_value_heads,
        F::vocab_size, F::max_position_embeddings, F::hidden_act, F::rms_norm_eps,
        F::rope_theta, F::tie_word_embeddings, F::sliding_window, F::use_sliding_window];
    let names = ["model_type", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "vocab_size", "max_position_embeddings", "hidden_act", "rms_norm_eps",
        "rope_theta", "tie_word_embeddings", "sliding_window", "use_sliding_window"];
    for (index, id) in ids.into_iter().enumerate() {
        assert_eq!(id as usize, index);
        assert_eq!(id.as_str(), names[index]);
    }
}

#[test]
fn qwen3_generated_private_field_ids_keep_source_order_and_names() {
    use Qwen3ConfigField as F;
    let ids = [F::model_type, F::hidden_size, F::intermediate_size,
        F::num_hidden_layers, F::num_attention_heads, F::num_key_value_heads,
        F::vocab_size, F::max_position_embeddings, F::head_dim, F::hidden_act,
        F::rms_norm_eps, F::rope_theta, F::tie_word_embeddings, F::attention_bias,
        F::attention_dropout, F::rope_scaling, F::sliding_window,
        F::use_sliding_window, F::use_cache, F::max_window_layers, F::layer_types];
    let names = ["model_type", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "vocab_size", "max_position_embeddings", "head_dim", "hidden_act",
        "rms_norm_eps", "rope_theta", "tie_word_embeddings", "attention_bias",
        "attention_dropout", "rope_scaling", "sliding_window",
        "use_sliding_window", "use_cache", "max_window_layers", "layer_types"];
    for (index, id) in ids.into_iter().enumerate() {
        assert_eq!(id as usize, index);
        assert_eq!(id.as_str(), names[index]);
    }
}

#[test]
fn llama_generated_private_field_ids_keep_source_order_and_names() {
    use SourceConfigField as F;
    let ids = [F::model_type, F::hidden_size, F::intermediate_size,
        F::num_hidden_layers, F::num_attention_heads, F::num_key_value_heads,
        F::vocab_size, F::max_position_embeddings, F::hidden_act, F::rms_norm_eps,
        F::rope_theta, F::tie_word_embeddings, F::head_dim, F::attention_bias,
        F::mlp_bias, F::attention_dropout, F::rope_scaling, F::sliding_window,
        F::use_cache, F::pretraining_tp];
    let names = ["model_type", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "vocab_size", "max_position_embeddings", "hidden_act", "rms_norm_eps",
        "rope_theta", "tie_word_embeddings", "head_dim", "attention_bias",
        "mlp_bias", "attention_dropout", "rope_scaling", "sliding_window",
        "use_cache", "pretraining_tp"];
    for (index, id) in ids.into_iter().enumerate() {
        assert_eq!(id as usize, index);
        assert_eq!(id.as_str(), names[index]);
    }
}

#[test]
fn qwen2_explicit_optional_none_still_omits_serialized_keys() {
    for explicit_null in [false, true] {
        let mut document = qwen2();
        if explicit_null {
            document["sliding_window"] = Value::Null;
            document["use_sliding_window"] = Value::Null;
        }
        let q: QwenConfig = serde_json::from_value(document).unwrap();
        assert_eq!(q.sliding_window, None);
        assert_eq!(q.use_sliding_window, None);
        assert_eq!(serde_json::to_value(q).unwrap(), qwen2());
    }
}

#[test]
fn qwen3_implicit_optional_none_still_serializes_null() {
    for explicit_null in [false, true] {
        let mut document = qwen3();
        let mut expected = qwen3();
        for name in ["rope_scaling", "sliding_window", "layer_types"] {
            expected[name] = Value::Null;
            if explicit_null { document[name] = Value::Null; }
        }
        let q: Qwen3Config = serde_json::from_value(document).unwrap();
        assert!(q.rope_scaling.is_none() && q.sliding_window.is_none() && q.layer_types.is_none());
        assert_eq!(serde_json::to_value(q).unwrap(), expected);
    }
}

#[test]
fn llama_existing_named_and_bool_float_defaults_still_keep_optional_nulls() {
    let l: SourceConfig = serde_json::from_value(llama()).unwrap();
    // Independent original-value oracles, not calls to the default functions.
    assert_eq!(l.hidden_act, "silu");
    assert_eq!(l.rms_norm_eps, "0.000001");
    assert_eq!(l.rope_theta, 10_000);
    assert!(!l.tie_word_embeddings && !l.attention_bias && !l.mlp_bias);
    assert_eq!(l.attention_dropout.to_bits(), 0.0_f64.to_bits());
    let mut expected = llama();
    for (key, value) in [
        ("hidden_act", json!("silu")), ("rms_norm_eps", json!("0.000001")),
        ("rope_theta", json!(10_000)), ("tie_word_embeddings", json!(false)),
        ("attention_bias", json!(false)), ("mlp_bias", json!(false)),
        ("attention_dropout", json!(0.0)),
    ] { expected[key] = value; }
    for key in ["num_key_value_heads", "head_dim", "rope_scaling", "sliding_window", "use_cache", "pretraining_tp"] {
        expected[key] = Value::Null;
    }
    assert_eq!(serde_json::to_value(l).unwrap(), expected);
    let mut explicit_nulls = llama();
    for key in ["num_key_value_heads", "head_dim", "rope_scaling", "sliding_window", "use_cache", "pretraining_tp"] {
        explicit_nulls[key] = Value::Null;
    }
    let l: SourceConfig = serde_json::from_value(explicit_nulls).unwrap();
    assert_eq!(serde_json::to_value(l).unwrap(), expected);
}

#[test]
fn populated_optionals_keep_original_keys_and_values() {
    let mut q = qwen2();
    q["sliding_window"] = json!(16);
    q["use_sliding_window"] = json!(false);
    assert_eq!(serde_json::to_value(serde_json::from_value::<QwenConfig>(q.clone()).unwrap()).unwrap(), q);
    let mut q = qwen3();
    q["rope_scaling"] = json!({"rope_type": "default"});
    q["sliding_window"] = json!(16);
    q["layer_types"] = json!(["full_attention", "full_attention"]);
    assert_eq!(serde_json::to_value(serde_json::from_value::<Qwen3Config>(q.clone()).unwrap()).unwrap(), q);
    let mut l = llama();
    for (key, value) in [("num_key_value_heads", json!(2)), ("head_dim", json!(8)),
        ("rope_scaling", json!({"rope_type": "default"})), ("sliding_window", json!(16)),
        ("use_cache", json!(true)), ("pretraining_tp", json!(1))] {
        l[key] = value;
    }
    let source: SourceConfig = serde_json::from_value(l.clone()).unwrap();
    let serialized = serde_json::to_value(source).unwrap();
    for (key, value) in l.as_object().unwrap() { assert_eq!(&serialized[key], value); }
}

fn rejects(role: usize, document: Value) -> bool {
    match role {
        0 => serde_json::from_value::<QwenConfig>(document).is_err(),
        1 => serde_json::from_value::<Qwen3Config>(document).is_err(),
        2 => serde_json::from_value::<SourceConfig>(document).is_err(),
        _ => panic!("unknown test role"),
    }
}

#[test]
fn every_required_field_still_rejects_missing_and_null() {
    // These three minimal documents contain only the independently listed
    // required fields; no generated field-ID traversal derives this oracle.
    for (role, base) in [qwen2(), qwen3(), llama()].into_iter().enumerate() {
        assert!(!rejects(role, base.clone()));
        for key in base.as_object().unwrap().keys() {
            let mut missing = base.clone();
            missing.as_object_mut().unwrap().remove(key);
            assert!(rejects(role, missing), "role {role}, missing {key}");
            let mut null = base.clone();
            null[key] = Value::Null;
            assert!(rejects(role, null), "role {role}, null {key}");
        }
    }
}

#[test]
fn ordinary_u64_dimensions_do_not_gain_the_integral_float_codec() {
    for (role, base) in [qwen2(), qwen3(), llama()].into_iter().enumerate() {
        for key in ["hidden_size", "intermediate_size", "vocab_size", "max_position_embeddings"] {
            let mut document = base.clone();
            document[key] = serde_json::from_str("1.0").unwrap();
            assert!(rejects(role, document), "role {role}, {key}");
        }
        let mut document = base;
        document["head_dim"] = serde_json::from_str("1.0").unwrap();
        if role != 0 { assert!(rejects(role, document)); }
    }
}

fn decoded_rope(role: usize, document: Value) -> u64 {
    match role {
        0 => serde_json::from_value::<QwenConfig>(document).unwrap().rope_theta,
        1 => serde_json::from_value::<Qwen3Config>(document).unwrap().rope_theta,
        2 => serde_json::from_value::<SourceConfig>(document).unwrap().rope_theta,
        _ => panic!("unknown test role"),
    }
}

#[test]
fn integral_codec_keeps_float_integer_negative_zero_and_integer_max_acceptance() {
    // Deserialization contract ONLY: subsequent semantic validation may reject
    // zero, unrelated to the unchanged custom integral codec.
    for (role, base) in [qwen2(), qwen3(), llama()].into_iter().enumerate() {
        for (raw, expected) in [("1.0", 1), ("-0.0", 0), ("18446744073709551615", u64::MAX)] {
            let mut document = base.clone();
            document["rope_theta"] = serde_json::from_str(raw).unwrap();
            assert_eq!(decoded_rope(role, document), expected);
        }
    }
}

#[test]
fn integral_codec_keeps_float_two_to_64_fractional_negative_and_nonnumber_rejections() {
    for (role, base) in [qwen2(), qwen3(), llama()].into_iter().enumerate() {
        for raw in ["18446744073709551616.0", "1.5", "-1", "null", "true", "\"10000\""] {
            let mut document = base.clone();
            document["rope_theta"] = serde_json::from_str(raw).unwrap();
            assert!(rejects(role, document), "role {role}, {raw}");
        }
    }
}

#[test]
fn decimal_fields_keep_verbatim_string_and_number_stringification() {
    for (role, base) in [qwen2(), qwen3(), llama()].into_iter().enumerate() {
        for (value, expected) in [(json!("1e-6"), "1e-6"), (json!(0.125), "0.125")] {
            let mut document = base.clone();
            document["rms_norm_eps"] = value.clone();
            if role == 1 { document["attention_dropout"] = value; }
            let serialized = match role {
                0 => serde_json::to_value(serde_json::from_value::<QwenConfig>(document).unwrap()).unwrap(),
                1 => serde_json::to_value(serde_json::from_value::<Qwen3Config>(document).unwrap()).unwrap(),
                2 => serde_json::to_value(serde_json::from_value::<SourceConfig>(document).unwrap()).unwrap(),
                _ => unreachable!(),
            };
            assert_eq!(serialized["rms_norm_eps"], json!(expected));
            if role == 1 { assert_eq!(serialized["attention_dropout"], json!(expected)); }
        }
    }
}

#[test]
fn decimal_fields_reject_null_bool_list_and_object_without_default_fallback() {
    for (role, base) in [qwen2(), qwen3(), llama()].into_iter().enumerate() {
        for bad in [Value::Null, json!(false), json!([]), json!({})] {
            let mut document = base.clone();
            document["rms_norm_eps"] = bad.clone();
            assert!(rejects(role, document));
            if role == 1 {
                let mut document = base.clone();
                document["attention_dropout"] = bad;
                assert!(rejects(role, document));
            }
        }
    }
}

#[test]
fn llama_defaulted_scalar_null_is_not_missing_and_f64_is_not_decimal_text() {
    for key in ["hidden_act", "rope_theta", "tie_word_embeddings", "attention_bias", "mlp_bias", "attention_dropout"] {
        let mut document = llama();
        document[key] = Value::Null;
        assert!(serde_json::from_value::<SourceConfig>(document).is_err(), "{key}");
    }
    let mut document = llama();
    document["attention_dropout"] = json!(0.125);
    document["hidden_act"] = json!("caller-supplied-act");
    document["tie_word_embeddings"] = json!(true);
    document["attention_bias"] = json!(true);
    document["mlp_bias"] = json!(true);
    // Deserialization only, not a claim these values pass model validation.
    let l: SourceConfig = serde_json::from_value(document.clone()).unwrap();
    assert_eq!(l.attention_dropout, 0.125);
    assert_eq!(l.hidden_act, "caller-supplied-act");
    assert!(l.tie_word_embeddings && l.attention_bias && l.mlp_bias);
    document["attention_dropout"] = json!("0.125");
    assert!(serde_json::from_value::<SourceConfig>(document).is_err());
}
