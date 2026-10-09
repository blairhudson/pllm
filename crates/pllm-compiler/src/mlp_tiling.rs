//! Bounded row-independent MLP regions in an already admitted semantic schedule.
use crate::{lower_decoder_runtime_schedule, DecoderCompositionKind, DecoderRuntimeExecutor};
use pllm_models::{DecoderPlan, ModelOperator};
use pllm_types::{canonical_bytes, Digest};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Serialize)]
struct Region {
    first_step: usize,
    end_step: usize,
    layer: u64,
    input_id: String,
    output_id: String,
    operation_ids: Vec<String>,
    hidden_width: u64,
    intermediate_width: u64,
}

#[derive(Serialize)]
struct Contract {
    schema: &'static str,
    source_plan_digest: Digest,
    source_schedule_digest: Digest,
    composition_digest: Digest,
    tile_rows: u64,
    regions: Vec<Region>,
}

pub fn mlp_tiling_bytes(
    plan: &[u8],
    composition: &[u8],
    tile_rows: u64,
) -> Result<Vec<u8>, String> {
    if plan.len() > 16 << 20 || composition.len() > 2 << 20 || !(1..=256).contains(&tile_rows) {
        return Err("MLP tiling exceeds bounded public admission".into());
    }
    if !matches!(
        crate::classify_decoder_composition(composition)?,
        DecoderCompositionKind::MaskedLinear | DecoderCompositionKind::ClientOnlyLinear
    ) {
        return Err("MLP tiling requires unverified public linear execution".into());
    }
    let plan: DecoderPlan = serde_json::from_slice(plan).map_err(|e| e.to_string())?;
    let source = lower_decoder_runtime_schedule(&plan, composition)?;
    if source.nonlinear.is_some() || source.retained_state.is_some() || plan.prefill.batch != 1 {
        return Err("MLP tiling requires the ordinary dense client numeric contract".into());
    }
    let graph = &plan.prefill;
    let operations: BTreeMap<_, _> = graph
        .operations
        .iter()
        .map(|op| (op.id.as_str(), op))
        .collect();
    let mut regions = Vec::new();
    for (first, window) in source.prefill.steps.windows(4).enumerate() {
        let [gate, silu, product, down] = window else {
            unreachable!()
        };
        if gate.operation_ids.len() != 2
            || gate.operators != [ModelOperator::Linear; 2]
            || silu.operators != [ModelOperator::Silu]
            || product.operators != [ModelOperator::Multiply]
            || down.operators != [ModelOperator::Linear]
            || gate.input_ids.len() != 1
            || silu.operation_ids.len() != 1
            || product.operation_ids.len() != 1
            || down.operation_ids.len() != 1
            || silu.executor != DecoderRuntimeExecutor::ClientLocal
            || product.executor != DecoderRuntimeExecutor::ClientLocal
            || !matches!(
                gate.executor,
                DecoderRuntimeExecutor::RemoteStage | DecoderRuntimeExecutor::ClientLinear
            )
            || down.executor != gate.executor
            || gate.layer.is_none()
            || window
                .iter()
                .any(|step| step.layer != gate.layer || step.terminal_row_only)
        {
            continue;
        }
        let activation = operations[silu.operation_ids[0].as_str()];
        let multiply = operations[product.operation_ids[0].as_str()];
        if activation.inputs.len() != 1
            || !gate.operation_ids.contains(&activation.inputs[0])
            || multiply.inputs.len() != 2
            || down.input_ids != product.operation_ids
        {
            continue;
        }
        let other = gate
            .operation_ids
            .iter()
            .find(|id| **id != activation.inputs[0])
            .unwrap();
        if multiply.inputs.iter().collect::<BTreeSet<_>>()
            != BTreeSet::from([&activation.id, other])
        {
            continue;
        }
        let ids: Vec<_> = window
            .iter()
            .flat_map(|step| step.operation_ids.iter().cloned())
            .collect();
        let output_id = &down.operation_ids[0];
        let interior: BTreeSet<_> = ids.iter().filter(|id| *id != output_id).collect();
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
        let Some(input) = operations.get(gate.input_ids[0].as_str()) else {
            continue;
        };
        let hidden = *input.output_shape.last().ok_or("MLP input width missing")?;
        let width = *activation
            .output_shape
            .last()
            .ok_or("MLP intermediate width missing")?;
        if input.output_shape != [1, graph.query_sequence, hidden]
            || operations[output_id.as_str()].output_shape != input.output_shape
            || ids.iter().any(|id| {
                let op = operations[id.as_str()];
                op.state_kind.is_some()
                    || op
                        .attributes
                        .get("output_dtype")
                        .is_some_and(|v| v != "float32")
                    || (id != output_id && op.output_shape != [1, graph.query_sequence, width])
            })
            || tile_rows.checked_mul(width).is_none_or(|n| n > 4_000_000)
        {
            continue;
        }
        regions.push(Region {
            first_step: first,
            end_step: first + 4,
            layer: gate.layer.unwrap(),
            input_id: gate.input_ids[0].clone(),
            output_id: output_id.clone(),
            operation_ids: ids,
            hidden_width: hidden,
            intermediate_width: width,
        });
    }
    if regions.is_empty() {
        return Err("no exclusive row-independent float32 MLP region".into());
    }
    Ok(canonical_bytes(&Contract {
        schema: "pllm.client_mlp_tiling.v1",
        source_plan_digest: plan.digest(),
        source_schedule_digest: source.digest(),
        composition_digest: source.composition_digest,
        tile_rows,
        regions,
    }))
}
