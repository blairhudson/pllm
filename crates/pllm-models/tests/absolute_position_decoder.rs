use pllm_models::{
    lower_model_json, lower_model_json_with_budget, DecoderExpansionBudget, DecoderPlan,
    DecoderWorkload, ModelOperator,
};
use serde_json::{json, Value};

fn source() -> Value {
    json!({"model_type":"gpt2", "n_embd":8,"n_head":2,"n_layer":2,"n_positions":32,"vocab_size":19})
}
fn workload() -> DecoderWorkload {
    DecoderWorkload {
        batch: 2,
        max_input_tokens: 3,
        max_new_tokens: 3,
    }
}
fn lower(v: &Value) -> DecoderPlan {
    lower_model_json(&serde_json::to_vec(v).unwrap(), workload()).unwrap()
}
fn budget() -> DecoderExpansionBudget {
    DecoderExpansionBudget {
        max_layers: 4,
        max_feature_width: 64,
        max_attention_heads: 4,
        max_vocabulary: 32,
        max_tensor_elements: 4096,
    }
}

#[test]
fn complete_source_graph_has_positions_centered_affine_norms_and_conv1d_orientation() {
    for tied in [true, false] {
        let mut v = source();
        v["tie_word_embeddings"] = json!(tied);
        let p = lower(&v);
        assert_eq!(p.adapter, "pllm.absolute_position_decoder.v1");
        assert_eq!(p.prefill.state_outputs.len(), 4);
        assert_eq!(p.decode.state_inputs.len(), 4);
        assert!(p.prefill.state_inputs.is_empty());
        for g in [&p.prefill, &p.decode] {
            let op = |id: &str| g.operations.iter().find(|o| o.id == id).unwrap();
            assert_eq!(op("position_lookup").inputs, ["input.positions"]);
            assert_eq!(
                op("position_lookup").attributes["weight_shape"],
                json!([32, 8])
            );
            assert_eq!(
                op("position_lookup").attributes["weight"],
                "transformer.wpe.weight"
            );
            assert_eq!(op("layer.0.input_norm").operator, ModelOperator::LayerNorm);
            assert_eq!(
                op("layer.0.input_norm").attributes["bias"],
                "transformer.h.0.ln_1.bias"
            );
            assert_eq!(
                op("layer.0.qkv_linear").attributes["weight_shape"],
                json!([8, 24])
            );
            assert_eq!(
                op("layer.0.qkv_linear").attributes["weight_layout"],
                "input_output"
            );
            assert_eq!(
                op("layer.0.qkv_linear").attributes["bias_shape"],
                json!([24])
            );
            for (role, start, end) in [("q", 0, 8), ("k", 8, 16), ("v", 16, 24)] {
                let slice = op(&format!("layer.0.{role}_slice"));
                assert_eq!(slice.attributes["start"], start);
                assert_eq!(slice.attributes["end"], end);
            }
            assert_eq!(
                op("layer.0.up_proj").attributes["weight_shape"],
                json!([8, 32])
            );
            assert_eq!(
                op("layer.0.down_proj").attributes["weight_shape"],
                json!([32, 8])
            );
            assert_eq!(op("layer.0.gelu").operator, ModelOperator::GeluTanh);
            assert!(!g.operations.iter().any(|o| matches!(
                o.operator,
                ModelOperator::RotaryEmbedding
                    | ModelOperator::RmsNorm
                    | ModelOperator::Multiply
                    | ModelOperator::Silu
            )));
            assert_eq!(
                op("output_head").attributes["weight"],
                if tied {
                    "transformer.wte.weight"
                } else {
                    "lm_head.weight"
                }
            );
            assert_eq!(op("output_head").attributes["weight_shape"], json!([19, 8]));
            assert_eq!(
                op("output_head").attributes["weight_layout"],
                "output_input"
            );
            assert_eq!(op("output_head").attributes["bias"], Value::Null);
            assert_eq!(
                op("token_selection").attributes["tie_break"],
                "lowest_index"
            );
            assert_eq!(op("layer.0.causal_mask").inputs.len(), 4);
            assert_eq!(op("layer.0.key_append").attributes["state_capacity"], 5);
            assert_eq!(op("layer.0.key_append").output_shape, vec![2, 2, 5, 4]);
            assert_eq!(
                op("layer.0.key_append").attributes["attention_domain"]["includes_current"],
                true
            );
        }
        let restored: DecoderPlan =
            serde_json::from_slice(&serde_json::to_vec(&p).unwrap()).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.digest(), p.digest());
    }
}

#[test]
fn actual_open_config_is_metadata_only_and_keeps_1024_positions() {
    let bytes = include_bytes!("fixtures/openai-community-gpt2-config.json");
    let p = lower_model_json(
        bytes,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1023,
            max_new_tokens: 2,
        },
    )
    .unwrap();
    assert_eq!(p.prefill.state_outputs.len(), 24);
    let qkv = p
        .prefill
        .operations
        .iter()
        .find(|o| o.id == "layer.0.qkv_linear")
        .unwrap();
    assert_eq!(qkv.attributes["weight_shape"], json!([768, 2304]));
    assert_eq!(p.decode.state_inputs[0].shape, vec![1, 12, 1024, 64]);
    assert!(lower_model_json(
        bytes,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1024,
            max_new_tokens: 2
        }
    )
    .is_err());
    // Larger standard source controls only allocate graph descriptors, never weights.
    let mut xl = source();
    xl["n_embd"] = json!(1600);
    xl["n_head"] = json!(25);
    xl["n_layer"] = json!(48);
    xl["n_positions"] = json!(1024);
    xl["vocab_size"] = json!(50257);
    assert_eq!(lower(&xl).decode.state_inputs.len(), 96);
}

#[test]
fn rejects_unsupported_or_invalid_source_variants_before_graph_construction() {
    for (key, value) in [
        ("activation_function", json!("gelu")),
        ("n_embd", json!(0)),
        ("n_head", json!(3)),
        ("n_layer", json!(u32::MAX)),
        ("n_positions", json!(1025)),
        ("n_inner", json!(0)),
        ("layer_norm_epsilon", json!("NaN")),
        ("layer_norm_epsilon", json!("1e-100")),
        ("scale_attn_weights", json!(false)),
        ("scale_attn_by_inverse_layer_idx", json!(true)),
        ("reorder_and_upcast_attn", json!(true)),
        ("add_cross_attention", json!(true)),
        ("is_encoder_decoder", json!(true)),
        ("pruned_heads", json!({"0":[0]})),
        ("use_cache", json!(false)),
        ("n_ctx", json!(31)),
        ("architectures", json!(["GPT2Model"])),
        ("attn_pdrop", json!(1.1)),
        ("torch_dtype", json!("float16")),
        ("rope_scaling", json!({})),
        ("_attn_implementation", json!("flash_attention_2")),
    ] {
        let mut v = source();
        v[key] = value;
        assert!(
            lower_model_json(&serde_json::to_vec(&v).unwrap(), workload()).is_err(),
            "{key}"
        );
    }
    let mut custom = source();
    custom["n_inner"] = json!(13);
    assert!(lower(&custom).validate().is_ok());
    for w in [
        DecoderWorkload {
            batch: 0,
            ..workload()
        },
        DecoderWorkload {
            max_input_tokens: u64::MAX,
            ..workload()
        },
        DecoderWorkload {
            max_new_tokens: 0,
            ..workload()
        },
    ] {
        assert!(lower_model_json(&serde_json::to_vec(&source()).unwrap(), w).is_err());
    }
}

#[test]
fn native_budget_prices_fused_projection_before_expansion() {
    let bytes = serde_json::to_vec(&source()).unwrap();
    let p = lower_model_json_with_budget(&bytes, workload(), budget()).unwrap();
    assert_eq!(p, lower(&source()));
    for b in [
        DecoderExpansionBudget {
            max_layers: 1,
            ..budget()
        },
        DecoderExpansionBudget {
            max_feature_width: 23,
            ..budget()
        },
        DecoderExpansionBudget {
            max_tensor_elements: 143,
            ..budget()
        },
        DecoderExpansionBudget {
            max_attention_heads: 1,
            ..budget()
        },
        DecoderExpansionBudget {
            max_vocabulary: 18,
            ..budget()
        },
    ] {
        assert!(lower_model_json_with_budget(&bytes, workload(), b).is_err());
    }
    let mut v = source();
    v["n_inner"] = json!(8);
    let b = DecoderExpansionBudget {
        max_feature_width: 23,
        ..budget()
    };
    let e =
        lower_model_json_with_budget(&serde_json::to_vec(&v).unwrap(), workload(), b).unwrap_err();
    assert!(e.to_string().contains("fused QKV"));
}

#[test]
fn validator_rejects_bias_shape_orientation_slice_state_and_head_drift() {
    let p = lower(&source());
    for (id, key, value) in [
        ("position_lookup", "weight_shape", json!([31, 8])),
        ("position_lookup", "out_of_bounds", json!("wrap")),
        ("layer.0.input_norm", "variance", json!("uncentered")),
        ("layer.0.input_norm", "bias", Value::Null),
        ("layer.0.qkv_linear", "weight_layout", json!("output_input")),
        ("layer.0.qkv_linear", "bias_shape", json!([8])),
        ("layer.0.q_slice", "start", json!(1)),
        ("layer.0.key_append", "state_capacity", json!(4)),
        ("output_head", "weight", json!("lm_head.weight")),
        ("output_head", "weight_shape", json!([8, 19])),
        ("layer.0.down_proj", "bias", Value::Null),
    ] {
        let mut bad = p.clone();
        bad.prefill
            .operations
            .iter_mut()
            .find(|o| o.id == id)
            .unwrap()
            .attributes[key] = value;
        assert!(bad.validate().is_err(), "{id}:{key}");
    }
    let mut bad = p.clone();
    bad.decode.state_inputs[0].shape[2] = 4;
    assert!(bad.validate().is_err());
    let mut bad = p.clone();
    bad.decode
        .operations
        .iter_mut()
        .find(|o| o.id == "layer.0.attention_scores")
        .unwrap()
        .inputs[0] = "layer.0.v_heads".into();
    assert!(bad.validate().is_err());
    let mut bad = p;
    bad.prefill
        .operations
        .iter_mut()
        .find(|o| o.id == "final_norm")
        .unwrap()
        .inputs[0] = "embedding_sum".into();
    assert!(bad.validate().is_err());
}

#[test]
fn existing_operator_serializations_stay_unchanged_and_layer_norm_is_additive() {
    for (op, label) in [
        (ModelOperator::RmsNorm, "rms_norm"),
        (ModelOperator::TokenLookup, "token_lookup"),
        (ModelOperator::GeluTanh, "gelu_tanh"),
        (ModelOperator::LayerNorm, "layer_norm"),
    ] {
        assert_eq!(serde_json::to_value(op).unwrap(), label);
    }
}
