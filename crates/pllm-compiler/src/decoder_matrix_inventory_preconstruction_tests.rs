use super::*;
use pllm_models::{lower_model_json, DecoderWorkload};
use pllm_types::canonical_bytes;
use serde_json::json;
use std::collections::BTreeMap;

pub(super) fn plan(family: &str, tied: bool) -> DecoderPlan {
    let mut config = json!({
        "model_type": family, "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 32, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": tied
    });
    if family == "qwen3" {
        for (key, value) in [
            ("head_dim", json!(4)),
            ("attention_bias", json!(false)),
            ("attention_dropout", json!(0.0)),
            ("rope_scaling", Value::Null),
            ("sliding_window", Value::Null),
            ("use_sliding_window", json!(false)),
            ("use_cache", json!(true)),
            ("max_window_layers", json!(2)),
            ("layer_types", json!(["full_attention", "full_attention"])),
        ] {
            config[key] = value;
        }
    }
    lower_model_json(
        &canonical_bytes(&config),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 3,
            max_new_tokens: 2,
        },
    )
    .unwrap()
}

// Independent reference for the old B-tree grouping, including original owned
// keys, position sort and error precedence. Test-only, never an alternate path.
fn legacy_groups(graph: &DecoderGraph) -> Result<BTreeMap<&str, Vec<&ModelOperation>>, String> {
    let layers: BTreeSet<_> = graph.operations.iter().filter_map(|op| op.layer).collect();
    if layers.is_empty() || layers.iter().copied().ne(0..layers.len() as u64) {
        return Err("decoder runtime layers must be contiguous from zero".into());
    }
    let mut grouped = BTreeMap::new();
    for op in &graph.operations {
        if remote(op.operator) {
            if op
                .attributes
                .get("weight")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            {
                return Err(format!(
                    "remote operation {} requires one declared weight artifact",
                    op.id
                ));
            }
            grouped
                .entry((
                    op.operator,
                    op.layer,
                    op.inputs.clone(),
                    (op.operator == ModelOperator::OutputHead).then(|| op.id.clone()),
                ))
                .or_insert_with(Vec::new)
                .push(op);
        }
    }
    if !graph
        .operations
        .iter()
        .any(|op| op.operator == ModelOperator::TokenLookup)
        || graph
            .operations
            .iter()
            .filter(|op| op.operator == ModelOperator::OutputHead)
            .count()
            != 1
    {
        return Err("decoder runtime requires token lookup and one output head".into());
    }
    let positions: BTreeMap<_, _> = graph
        .operations
        .iter()
        .enumerate()
        .map(|(i, op)| (op.id.as_str(), i))
        .collect();
    let mut result = BTreeMap::new();
    let mut members = BTreeSet::new();
    for mut group in grouped.into_values() {
        group.sort_by_key(|op| positions[op.id.as_str()]);
        for op in &group {
            if !members.insert(op.id.as_str()) {
                return Err(format!(
                    "operation {} belongs to multiple runtime stages",
                    op.id
                ));
            }
        }
        let leader = group[0].id.as_str();
        assert!(result.insert(leader, group).is_none());
    }
    Ok(result)
}

// Valid-input oracle, independent of new borrowed structs, lowering and stream
// emitter. Recreates the original owned JSON schema and original B-tree order.
pub(super) fn legacy_inventory_json(model: &DecoderPlan) -> Value {
    let phase = |graph: &DecoderGraph| {
        let vocabulary = graph
            .operations
            .iter()
            .find(|op| op.operator == ModelOperator::OutputHead)
            .unwrap()
            .output_shape[1];
        let reference = legacy_groups(graph).unwrap();
        let mut groups = Vec::new();
        for op in &graph.operations {
            let Some(members) = reference.get(op.id.as_str()) else {
                continue;
            };
            let token = op.operator == ModelOperator::TokenLookup;
            let input_shape = if token {
                vec![graph.batch, graph.query_sequence]
            } else {
                graph
                    .operations
                    .iter()
                    .find(|producer| producer.id == op.inputs[0])
                    .unwrap()
                    .output_shape
                    .clone()
            };
            let mut offset = 0;
            let members: Vec<_> = members.iter().map(|member| {
                let width = *member.output_shape.last().unwrap();
                let bias = member.attributes.get("bias").and_then(Value::as_str);
                let value = json!({
                    "operation_id": member.id, "weight_id": member.attributes["weight"],
                    "bias_id": bias, "source_shape": if token { [vocabulary, width] } else { [width, *input_shape.last().unwrap()] },
                    "bias_shape": bias.map(|_| [width]), "output_shape": member.output_shape,
                    "output_offset": offset, "output_width": width,
                });
                offset += width;
                value
            }).collect();
            groups.push(json!({
                "order": groups.len(), "operator": op.operator, "layer": op.layer,
                "input_id": op.inputs[0], "input_shape": input_shape,
                "artifact_kind": if token { "token_table" } else { "matrix" },
                "orientation": if token { "vocabulary_feature" } else { "output_input" },
                "logical_shape": if token { [vocabulary, offset] } else { [offset, *input_shape.last().unwrap()] },
                "members": members,
            }));
        }
        json!({ "mode": graph.mode, "batch": graph.batch, "query_sequence": graph.query_sequence,
            "maximum_key_sequence": graph.maximum_key_sequence, "groups": groups })
    };
    json!({ "schema_version": DECODER_MATRIX_INVENTORY_SCHEMA_VERSION,
        "model_plan_digest": model.digest(), "model_config_digest": model.config_digest,
        "prefill": phase(&model.prefill), "decode": phase(&model.decode) })
}

#[test]
fn stream_bytes_and_digests_match_original_canonical_implementation() {
    for family in ["llama", "qwen2", "qwen3"] {
        for tied in [false, true] {
            let mut model = plan(family, tied);
            for graph in [&mut model.prefill, &mut model.decode] {
                for op in &mut graph.operations {
                    if remote(op.operator) {
                        for key in ["weight", "bias"] {
                            if let Some(id) = op.attributes.get(key).and_then(Value::as_str) {
                                op.attributes[key] = json!(format!("{id}\nλ\"\\"));
                            }
                        }
                    } else {
                        op.attributes["canonical-floats"] = json!([-0.0, 1e-30, 1.5, "\u{0001}"]);
                    }
                }
            }
            let lowered = lower_inventory(&model).unwrap();
            let legacy = canonical_bytes(&legacy_inventory_json(&model));
            assert_eq!(canonical_bytes(&lowered), legacy);
            let streamed = inventory_canonical_bytes(&lowered).unwrap();
            assert_eq!(streamed, legacy);
            assert_eq!(streamed.capacity(), streamed.len());
            assert_eq!(lowered.model_plan_digest.as_str(), model.digest().as_str());
            let inventory = lower_decoder_matrix_inventory(&model).unwrap();
            assert_eq!(inventory.canonical_bytes(), legacy);
            assert_eq!(
                *inventory.digest(),
                digest_bytes(DECODER_MATRIX_INVENTORY_SCHEMA_VERSION, &legacy)
            );
        }
    }
}

#[test]
fn metadata_uses_original_id_and_shape_backing_and_exact_buffer_layouts() {
    let model = plan("llama", true);
    let metadata = lower_inventory(&model).unwrap();
    for (phase, graph) in [
        (&metadata.prefill, &model.prefill),
        (&metadata.decode, &model.decode),
    ] {
        assert!(phase.groups.len() <= graph.operations.len());
        for group in phase.groups.iter() {
            let leader = graph
                .operations
                .iter()
                .find(|op| op.id == group.members[0].operation_id)
                .unwrap();
            assert_eq!(group.input_id.as_ptr(), leader.inputs[0].as_ptr());
            if let InputShape::Produced(shape) = group.input_shape {
                let producer = graph
                    .operations
                    .iter()
                    .find(|op| op.id == group.input_id)
                    .unwrap();
                assert_eq!(shape.as_ptr(), producer.output_shape.as_ptr());
            }
            for member in group.members.iter() {
                let op = graph
                    .operations
                    .iter()
                    .find(|op| op.id == member.operation_id)
                    .unwrap();
                assert_eq!(member.operation_id.as_ptr(), op.id.as_ptr());
                assert_eq!(
                    member.weight_id.as_ptr(),
                    op.attributes["weight"].as_str().unwrap().as_ptr()
                );
                assert_eq!(member.output_shape.as_ptr(), op.output_shape.as_ptr());
            }
        }
    }
    let raw = model.digest().as_sha256();
    let exact = HexDigest::new(raw).into_digest();
    assert_eq!(exact, model.digest());
    assert_eq!(exact.backing_capacity_bytes(), 64);
}
