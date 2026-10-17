use pllm_compiler::{GkmDecoderContract, GkmDecoderSession};
use pllm_types::canonical_bytes;
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn config(family: &str, tied: bool) -> Value {
    let mut value = json!({
        "model_type": family, "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 32, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": tied
    });
    if family == "qwen3" {
        value["head_dim"] = json!(4);
        value["attention_bias"] = json!(false);
        value["attention_dropout"] = json!(0.0);
        value["rope_scaling"] = Value::Null;
        value["sliding_window"] = Value::Null;
        value["use_sliding_window"] = json!(false);
        value["use_cache"] = json!(true);
        value["max_window_layers"] = json!(2);
        value["layer_types"] = json!(["full_attention", "full_attention"]);
    }
    value
}

#[test]
fn generic_dense_contracts_bind_original_boundaries_and_complete_window() {
    for family in ["qwen2", "qwen3", "llama"] {
        for tied in [true, false] {
            let source = canonical_bytes(&config(family, tied));
            let contract = GkmDecoderContract::compile(&source, 3, 2).unwrap();
            let doc: Value = serde_json::from_slice(&contract.canonical_bytes()).unwrap();
            assert_eq!(doc["window_steps"], 48); // 3 lookups + 24 body + 1 head + 2*10
            assert_eq!(doc["matrices"].as_array().unwrap().len(), 9);
            assert_eq!(doc["matrices"][0]["rows"], 16);
            assert_eq!(doc["matrices"][8]["rows"], 32);
            assert_eq!(doc["numeric_profile"], pllm_core::gkm::NUMERIC_PROFILE);
            assert_eq!(
                doc["local_numeric_profile"],
                "semantic.float32.dense-full-kv.client-bias-after-div22.v1"
            );
            assert!(doc["execution_payload_estimate_bytes"].as_u64().unwrap() < 64 << 20);
            assert_eq!(
                contract.canonical_bytes(),
                GkmDecoderContract::compile(&source, 3, 2)
                    .unwrap()
                    .canonical_bytes()
            );
        }
    }
}

fn tensors(contract: &GkmDecoderContract) -> BTreeMap<String, Vec<u8>> {
    contract
        .source_shapes()
        .iter()
        .map(|(name, shape)| {
            let mut bytes = Vec::new();
            for index in 0..shape.iter().product::<usize>() {
                let value = if index == 0 {
                    -0.0f32
                } else {
                    (index as f32 % 13.0 - 6.0) / 64.0
                };
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            (name.clone(), bytes)
        })
        .collect()
}

#[test]
fn all_scheduled_calls_pass_integer_and_original_byte_oracles() {
    for seeded in [false, true] {
        let contract =
            GkmDecoderContract::compile(&canonical_bytes(&config("qwen2", true)), 2, 1).unwrap();
        let doc: Value = serde_json::from_slice(&contract.canonical_bytes()).unwrap();
        let source = tensors(&contract);
        let borrowed = source
            .iter()
            .map(|(k, v)| (k.clone(), v.as_slice()))
            .collect();
        let mut session = GkmDecoderSession::new(contract, &borrowed, seeded).unwrap();
        for (phase, rows) in [("prefill", 2), ("decode", 1)] {
            for token in 0..rows {
                let got = session.lookup(token).unwrap();
                let table = &source[doc["table"].as_str().unwrap()];
                assert_eq!(&got, &table[token * 32..(token + 1) * 32]);
            }
            for (stage, matrix) in doc["matrices"].as_array().unwrap().iter().enumerate() {
                let count = if stage == 8 { 1 } else { rows };
                let columns = matrix["columns"].as_u64().unwrap() as usize;
                for _ in 0..count {
                    // Integral and exactly representable: independent integer
                    // dot products, with no core numeric helper in the oracle.
                    let input: Vec<f32> = (0..columns).map(|i| (i as i32 % 7 - 3) as f32).collect();
                    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let got = session.linear(stage, &bytes).unwrap();
                    let mut expected = Vec::new();
                    for member in matrix["members"].as_array().unwrap() {
                        for row in
                            source[member["weight"].as_str().unwrap()].chunks_exact(columns * 4)
                        {
                            let dot: i64 = row
                                .chunks_exact(4)
                                .zip(&input)
                                .map(|(b, x)| {
                                    let w = f32::from_le_bytes(b.try_into().unwrap());
                                    (w as f64 * 2048.0).round_ties_even() as i64
                                        * (*x as i64 * 2048)
                                })
                                .sum();
                            expected.extend_from_slice(&((dot as f32) / 4194304.0).to_le_bytes());
                        }
                    }
                    assert_eq!(got, expected, "{phase} stage {stage}");
                }
            }
        }
        let receipt: Value = serde_json::from_slice(&session.finish().unwrap()).unwrap();
        assert_eq!(receipt["steps"], doc["window_steps"]);
        assert_eq!(receipt["seeded"], seeded);
        assert!(session.finish().is_err());
        assert!(session.lookup(0).is_err());
    }
}

#[test]
fn source_and_geometry_reject_before_execution() {
    let mut large = config("qwen2", true);
    large["hidden_size"] = json!(896);
    large["intermediate_size"] = json!(4864);
    large["vocab_size"] = json!(151936);
    assert!(GkmDecoderContract::compile(&canonical_bytes(&large), 3, 1).is_err());
    let config = canonical_bytes(&config("qwen2", true));
    for (rows, decode) in [(0, 1), (33, 0), (1, 9), (32, 8)] {
        assert!(GkmDecoderContract::compile(&config, rows, decode).is_err());
    }
    let contract = GkmDecoderContract::compile(&config, 2, 1).unwrap();
    let mut source = tensors(&contract);
    let table = serde_json::from_slice::<Value>(&contract.canonical_bytes()).unwrap()["table"]
        .as_str()
        .unwrap()
        .to_owned();
    source.get_mut(&table).unwrap()[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    let borrowed = source
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect();
    assert!(GkmDecoderSession::new(contract, &borrowed, false).is_err());
}

#[test]
fn wrong_stage_range_shape_and_incomplete_calls_burn_request() {
    for failure in 0..5 {
        let contract =
            GkmDecoderContract::compile(&canonical_bytes(&config("qwen2", true)), 1, 0).unwrap();
        let source = tensors(&contract);
        let borrowed = source
            .iter()
            .map(|(k, v)| (k.clone(), v.as_slice()))
            .collect();
        let mut session = GkmDecoderSession::new(contract, &borrowed, true).unwrap();
        match failure {
            0 => {
                assert!(session.finish().is_err());
            }
            1 => {
                assert!(session.linear(0, &[0; 32]).is_err());
            }
            2 => {
                assert!(session.lookup(32).is_err());
            }
            3 => {
                session.lookup(1).unwrap();
                assert!(session.linear(0, &[0; 31]).is_err());
            }
            _ => {
                session.lookup(1).unwrap();
                let bytes: Vec<u8> = [2f32.powi(31); 8]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                assert!(session.linear(0, &bytes).is_err());
            }
        }
        assert!(session.lookup(0).is_err());
        assert!(session.finish().is_err());
    }
}
