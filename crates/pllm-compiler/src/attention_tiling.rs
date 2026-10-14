//! Initial-prefill lifetime regions; complete reductions and KV owners are unchanged.
use crate::{
    lower_decoder_runtime_schedule, DecoderCompositionKind, DecoderRuntimeExecutor,
    DecoderRuntimeSchedule, ExperimentPipeline,
};
use pllm_models::{DecoderPlan, ModelOperator, StateKind};
use pllm_types::{canonical_bytes, Digest};
use serde::Serialize;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Serialize)]
struct Region {
    first_step: usize,
    end_step: usize,
    layer: u64,
    operation_ids: Vec<String>,
    query_id: String,
    key_id: String,
    value_id: String,
    positions_id: String,
    output_id: String,
    query_heads: u64,
    kv_heads: u64,
    group_size: u64,
    head_dim: u64,
}

#[derive(Serialize)]
struct Contract {
    schema: &'static str,
    source_plan_digest: Digest,
    source_schedule_digest: Digest,
    composition_digest: Digest,
    query_rows: u64,
    kv_heads: u64,
    maximum_query_sequence: u64,
    maximum_key_sequence: u64,
    regions: Vec<Region>,
}

pub fn attention_tiling_bytes(
    plan: &[u8],
    composition: &[u8],
    query_rows: u64,
    kv_heads: u64,
) -> Result<Vec<u8>, String> {
    if plan.len() > 16 << 20
        || composition.len() > 2 << 20
        || !(1..=256).contains(&query_rows)
        || !(1..=64).contains(&kv_heads)
    {
        return Err("attention tiling exceeds bounded public admission".into());
    }
    if !matches!(
        crate::classify_decoder_composition(composition)?,
        DecoderCompositionKind::MaskedLinear | DecoderCompositionKind::ClientOnlyLinear
    ) {
        return Err("attention tiling requires unverified public linear execution".into());
    }
    let pipeline: ExperimentPipeline =
        serde_json::from_slice(composition).map_err(|e| e.to_string())?;
    if pipeline
        .components
        .get("kernels")
        .is_none_or(|k| k.component != "pllm/cpu")
    {
        return Err("attention tiling requires CPU execution".into());
    }
    let plan: DecoderPlan = serde_json::from_slice(plan).map_err(|e| e.to_string())?;
    let source = lower_decoder_runtime_schedule(&plan, composition)?;
    build_contract(&plan, &source, query_rows, kv_heads)
}

fn build_contract(
    plan: &DecoderPlan,
    source: &DecoderRuntimeSchedule,
    query_rows: u64,
    kv_heads: u64,
) -> Result<Vec<u8>, String> {
    let graph = &plan.prefill;
    if graph.batch != 1
        || source.private_boundary.is_some()
        || source.nonlinear.is_some()
        || source.retained_state.is_some()
        || source.decode_epochs.is_some()
        || source.inflight_prefix.is_some()
        || source.decode_reuse.is_some()
        || graph
            .state_outputs
            .iter()
            .any(|s| !matches!(s.kind, StateKind::Key | StateKind::Value))
    {
        return Err("attention tiling requires ordinary initial prefill and dense full KV".into());
    }
    let operations: BTreeMap<_, _> = graph
        .operations
        .iter()
        .map(|op| (op.id.as_str(), op))
        .collect();
    let expected = [
        ModelOperator::AttentionScores,
        ModelOperator::AttentionScale,
        ModelOperator::CausalMask,
        ModelOperator::Softmax,
        ModelOperator::AttentionValues,
    ];
    let mut regions = Vec::new();
    for (first, window) in source.prefill.steps.windows(5).enumerate() {
        if window.iter().zip(expected).any(|(step, kind)| {
            step.executor != DecoderRuntimeExecutor::ClientLocal
                || step.operators != [kind]
                || step.operation_ids.len() != 1
                || step.layer != window[0].layer
                || step.terminal_row_only
        }) || window[0].layer.is_none()
        {
            continue;
        }
        let ops: Vec<_> = window
            .iter()
            .map(|step| operations[step.operation_ids[0].as_str()])
            .collect();
        let (score, scale, mask, softmax, value) = (ops[0], ops[1], ops[2], ops[3], ops[4]);
        let Some(group) = score
            .attributes
            .get("group_size")
            .and_then(serde_json::Value::as_u64)
        else {
            continue;
        };
        if group == 0
            || group > 128
            || score.inputs.len() != 2
            || scale.inputs != [score.id.clone()]
            || mask.inputs.len() != 2
            || mask.inputs[0] != scale.id
            || softmax.inputs != [mask.id.clone()]
            || value.inputs.len() != 2
            || value.inputs[0] != softmax.id
            || score.attributes != json!({"group_size": group})
            || value.attributes != json!({"group_size": group})
            || mask.attributes != json!({})
            || ![
                json!({}),
                json!({"axis": -1}),
                json!({"axis": -1, "compute_dtype": "float32"}),
            ]
            .contains(&softmax.attributes)
            || ops.iter().any(|op| op.state_kind.is_some())
        {
            continue;
        }
        let (Some(query), Some(keys), Some(values)) = (
            operations.get(score.inputs[0].as_str()),
            operations.get(score.inputs[1].as_str()),
            operations.get(value.inputs[1].as_str()),
        ) else {
            continue;
        };
        let [1, heads, rows, width] = query.output_shape.as_slice() else {
            continue;
        };
        let [1, key_heads, key_rows, key_width] = keys.output_shape.as_slice() else {
            continue;
        };
        if *heads == 0
            || *heads > 128
            || *key_heads == 0
            || *key_heads > 64
            || *width == 0
            || *width > 256
            || width != key_width
            || *heads != key_heads * group
            || *rows != graph.query_sequence
            || *key_rows != graph.maximum_key_sequence
            || values.output_shape != keys.output_shape
            || value.output_shape != query.output_shape
            || scale.attributes != json!({"head_dim": width})
            || ops[..4]
                .iter()
                .any(|op| op.output_shape != [1, *heads, *rows, *key_rows])
            || heads
                .checked_mul(*rows)
                .and_then(|n| n.checked_mul(plan.decode.maximum_key_sequence))
                .is_none_or(|n| n > 16_777_216)
            || heads
                .checked_mul(*rows)
                .and_then(|n| n.checked_mul(*width))
                .is_none_or(|n| n > 16_777_216)
        {
            continue;
        }
        let ids: Vec<_> = ops.iter().map(|op| op.id.clone()).collect();
        let interior: BTreeSet<_> = ids[..4].iter().collect();
        if graph
            .operations
            .iter()
            .filter(|op| !ids.contains(&op.id))
            .any(|op| op.inputs.iter().any(|id| interior.contains(id)))
            || interior.iter().any(|id| {
                graph.output == ***id || graph.state_outputs.iter().any(|s| s.id == ***id)
            })
        {
            continue;
        }
        regions.push(Region {
            first_step: first,
            end_step: first + 5,
            layer: window[0].layer.unwrap(),
            operation_ids: ids,
            query_id: query.id.clone(),
            key_id: keys.id.clone(),
            value_id: values.id.clone(),
            positions_id: mask.inputs[1].clone(),
            output_id: value.id.clone(),
            query_heads: *heads,
            kv_heads: *key_heads,
            group_size: group,
            head_dim: *width,
        });
    }
    let attention_count = graph
        .operations
        .iter()
        .filter(|op| op.operator == ModelOperator::AttentionScores)
        .count();
    if regions.is_empty() || regions.len() != attention_count {
        return Err(
            "attention tiling requires exclusive complete float32 dense attention regions".into(),
        );
    }
    Ok(canonical_bytes(&Contract {
        schema: "pllm.client_attention_tiling.v1",
        source_plan_digest: plan.digest(),
        source_schedule_digest: source.digest(),
        composition_digest: source.composition_digest.clone(),
        query_rows,
        kv_heads,
        maximum_query_sequence: graph.query_sequence,
        maximum_key_sequence: plan.decode.maximum_key_sequence,
        regions,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_models::{lower_model_json, DecoderWorkload};

    fn fixture() -> (DecoderPlan, Vec<u8>) {
        let config = json!({"model_type":"qwen2", "hidden_size":32, "intermediate_size":64,
            "num_hidden_layers":2, "num_attention_heads":4, "num_key_value_heads":2,
            "vocab_size":64, "max_position_embeddings":128, "hidden_act":"silu",
            "rms_norm_eps":1e-6, "rope_theta":10000.0, "tie_word_embeddings":true});
        let plan = lower_model_json(
            &canonical_bytes(&config),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 33,
                max_new_tokens: 4,
            },
        )
        .unwrap();
        let composition = canonical_bytes(&json!({"components": {
            "inference": {"component":"pllm/inference", "params":{}},
            "kernels": {"component":"pllm/cpu", "params":{"threads":1}},
            "linear": {"component":"pllm/masked-linear", "params":{}},
            "preparation": {"component":"pllm/model-aware-corrections", "params":{}}
        }, "model":{"source":"fixture"}}));
        (plan, composition)
    }

    #[test]
    fn binds_schedule_geometry_and_optional_mlp_regions() {
        let (plan, composition) = fixture();
        for tiled in [false, true] {
            let mut pipeline: serde_json::Value = serde_json::from_slice(&composition).unwrap();
            if tiled {
                pipeline["components"]["scheduler"] =
                    json!({"component":"pllm/mlp-prefill-tiling/v1", "params":{"tile_rows":7}});
            }
            let bytes = canonical_bytes(&pipeline);
            let raw = attention_tiling_bytes(&canonical_bytes(&plan), &bytes, 16, 2).unwrap();
            let contract: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            assert_eq!(contract["regions"].as_array().unwrap().len(), 2);
            assert_eq!(contract["maximum_key_sequence"], 36);
            assert_eq!(
                contract["source_schedule_digest"],
                json!(lower_decoder_runtime_schedule(&plan, &bytes)
                    .unwrap()
                    .digest())
            );
            assert_ne!(
                raw,
                attention_tiling_bytes(&canonical_bytes(&plan), &bytes, 4, 1).unwrap()
            );
        }
        for (rows, heads) in [(0, 1), (257, 1), (1, 0), (1, 65)] {
            assert!(
                attention_tiling_bytes(&canonical_bytes(&plan), &composition, rows, heads).is_err()
            );
        }
    }

    #[test]
    fn rejects_external_consumers_numeric_changes_and_state_roots() {
        let (plan, composition) = fixture();
        let schedule = lower_decoder_runtime_schedule(&plan, &composition).unwrap();
        let first = plan
            .prefill
            .operations
            .iter()
            .position(|op| op.operator == ModelOperator::AttentionScores)
            .unwrap();
        for mutation in 0..5 {
            let mut changed = plan.clone();
            let interior = changed.prefill.operations[first].id.clone();
            match mutation {
                0 => changed
                    .prefill
                    .operations
                    .last_mut()
                    .unwrap()
                    .inputs
                    .push(interior),
                1 => changed.prefill.output = interior,
                2 => changed.prefill.state_outputs[0].id = interior,
                3 => {
                    changed.prefill.operations[first].attributes["output_dtype"] = json!("bfloat16")
                }
                _ => changed.prefill.operations[first].output_shape[3] += 1,
            }
            assert!(build_contract(&changed, &schedule, 16, 2).is_err());
        }
        let mut changed = schedule.clone();
        changed.retained_state = Some(json!({}));
        assert!(build_contract(&plan, &changed, 16, 2).is_err());
        changed.retained_state = None;
        changed.decode_epochs = Some(json!({}));
        assert!(build_contract(&plan, &changed, 16, 2).is_err());
    }
}
