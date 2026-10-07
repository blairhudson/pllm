//! Model-neutral SiLU/multiply regions and their explicit client numeric boundary.
use crate::decoder_runtime_schedule::{DecoderRuntimeExecutor, DecoderRuntimePhaseSchedule};
use pllm_models::{DecoderGraph, ModelOperator};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const COMPONENT: &str = "pllm/masked-wire-gated-multiply-q12/v1";

pub(crate) fn valid(component: &crate::ExperimentComponent) -> bool {
    component.component == COMPONENT
        && component.params.len() <= 3
        && component
            .params
            .keys()
            .all(|key| matches!(key.as_str(), "layout" | "dispatch" | "material_encoding"))
        && component
            .params
            .get("layout")
            .is_none_or(|value| matches!(value.as_str(), Some("fused" | "unfused")))
        && component
            .params
            .get("dispatch")
            .is_none_or(|value| matches!(value.as_str(), Some("serial" | "overlap")))
        && component
            .params
            .get("material_encoding")
            .is_none_or(|value| matches!(value.as_str(), Some("full" | "seeded")))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MaskedWireNumericContract {
    pub schema: String,
    pub component: String,
    pub layout: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material_encoding: Option<String>,
    pub profile_digest: pllm_types::Digest,
    pub input_boundary: String,
    pub output_boundary: String,
    pub gate_bound: u64,
    pub up_bound: u64,
    pub chunk_lanes: u64,
    pub maximum_tensor_elements: u64,
    pub combined_material_allocation_bytes: u64,
    pub dealer: String,
    pub peer_routing: String,
}

impl MaskedWireNumericContract {
    pub(crate) fn new(component: &crate::ExperimentComponent) -> Result<Self, String> {
        if !valid(component) {
            return Err("unsupported masked-wire numeric component".into());
        }
        Ok(Self {
            schema: "pllm.masked_wire_numeric.v1".into(),
            component: COMPONENT.into(),
            layout: component
                .params
                .get("layout")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unfused")
                .into(),
            dispatch: component
                .params
                .get("dispatch")
                .and_then(serde_json::Value::as_str)
                .filter(|dispatch| *dispatch != "serial")
                .map(str::to_owned),
            material_encoding: component
                .params
                .get("material_encoding")
                .and_then(serde_json::Value::as_str)
                .filter(|encoding| *encoding != "full")
                .map(str::to_owned),
            profile_digest: pllm_types::Digest::from_sha256(
                pllm_core::residual_silu_reference::Profile::new().digest(),
            ),
            input_boundary: "client.float32-to-signed-q24.floor.reject.v1".into(),
            output_boundary: "client.signed-q12-to-float32.v1".into(),
            gate_bound: 48,
            up_bound: 128,
            chunk_lanes: 256,
            maximum_tensor_elements: 4_000_000,
            combined_material_allocation_bytes: 64 << 20,
            dealer: "trusted-client.one-use".into(),
            peer_routing: "authenticated-workers.via-trusted-client".into(),
        })
    }
}

pub(crate) fn bind_regions(
    graph: &DecoderGraph,
    schedule: &mut DecoderRuntimePhaseSchedule,
) -> Result<(), String> {
    let operations: BTreeMap<_, _> = graph
        .operations
        .iter()
        .map(|op| (op.id.as_str(), op))
        .collect();
    let mut users: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for op in &graph.operations {
        for input in &op.inputs {
            users.entry(input).or_default().push(&op.id);
        }
        if op
            .attributes
            .get("output_dtype")
            .is_some_and(|dtype| dtype != "float32" && dtype != "model_native")
            || matches!(
                op.operator,
                ModelOperator::CausalConvolution
                    | ModelOperator::RmsNormGated
                    | ModelOperator::GeluTanh
            )
        {
            return Err(
                "masked-wire regions require explicit dense float32 gated operations".into(),
            );
        }
    }
    let mut pairs = BTreeMap::new();
    let mut removed = BTreeSet::new();
    for silu in graph
        .operations
        .iter()
        .filter(|op| op.operator == ModelOperator::Silu)
    {
        let Some(uses) = users.get(silu.id.as_str()) else {
            return Err("unconsumed SiLU region".into());
        };
        if uses.len() != 1
            || silu.inputs.len() != 1
            || silu.layer.is_none()
            || !silu
                .attributes
                .as_object()
                .is_some_and(|attrs| attrs.is_empty())
        {
            return Err(
                "masked-wire SiLU requires one multiply consumer and no extra attributes".into(),
            );
        }
        let product = operations[uses[0]];
        if product.operator != ModelOperator::Multiply
            || product.inputs.len() != 2
            || product.inputs.iter().filter(|id| **id == silu.id).count() != 1
            || product.layer != silu.layer
            || product.output_shape != silu.output_shape
            || !product
                .attributes
                .as_object()
                .is_some_and(|attrs| attrs.is_empty())
            || silu.id == graph.output
            || graph.state_outputs.iter().any(|state| state.id == silu.id)
        {
            return Err("masked-wire SiLU/multiply region is incomplete".into());
        }
        let up = product
            .inputs
            .iter()
            .find(|id| **id != silu.id)
            .ok_or("missing up input")?;
        let gate_op = operations
            .get(silu.inputs[0].as_str())
            .ok_or("missing gate producer")?;
        let up_op = operations.get(up.as_str()).ok_or("missing up producer")?;
        let valid_shape = matches!(silu.output_shape.as_slice(), [1, rows, width]
            if *rows > 0 && *width > 0 && rows.checked_mul(*width).is_some_and(|n| n <= 4_000_000));
        if !valid_shape
            || gate_op.output_shape != silu.output_shape
            || up_op.output_shape != silu.output_shape
            || graph.state_outputs.iter().any(|state| {
                !matches!(
                    state.kind,
                    pllm_models::StateKind::Key | pllm_models::StateKind::Value
                )
            })
        {
            return Err("masked-wire tensor shape/state exceeds its contract".into());
        }
        removed.insert(silu.id.as_str());
        if pairs
            .insert(product.id.as_str(), (silu, up.clone()))
            .is_some()
        {
            return Err("masked-wire regions overlap".into());
        }
    }
    if pairs.is_empty() {
        return Err("masked-wire composition has no complete gated region".into());
    }
    schedule
        .steps
        .retain(|step| !removed.contains(step.operation_ids[0].as_str()));
    for (order, step) in schedule.steps.iter_mut().enumerate() {
        if let Some((silu, up)) = pairs.get(step.operation_ids[0].as_str()) {
            if step.executor != DecoderRuntimeExecutor::ClientLocal || step.operation_ids.len() != 1
            {
                return Err("masked-wire region overlaps another executor".into());
            }
            step.operation_ids.insert(0, silu.id.clone());
            step.operators.insert(0, ModelOperator::Silu);
            step.input_ids = vec![silu.inputs[0].clone(), up.clone()];
            step.executor = DecoderRuntimeExecutor::RemoteGatedBlock;
        }
        step.order = order as u64;
    }
    Ok(())
}
