//! Bounded GKM decoder composition: source lowering, original fixed-point
//! boundaries, full client-local float32 state, and one final verified window.
//! Capsule reference and separately admitted full-share SDK transport contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use pllm_core::gkm::{self, Client, FloatTable, I16Matrix, Publication, Stage, Step, Worker};
use pllm_models::{DecoderPlan, DecoderWorkload, ModelOperator, StateKind};
use pllm_types::{canonical_bytes, canonical_digest, Digest};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use zeroize::Zeroizing;

pub(crate) mod sdk;
pub use sdk::GkmRuntimeContract;
mod batch;
mod transport;
pub use transport::{GkmTransportClient, GkmTransportWorker};
mod source_import;
pub use source_import::{GkmTransportImport, GKM_SOURCE_CHUNK_BYTES};

use crate::decoder_runtime_schedule::{
    lower_phase, DecoderRuntimeExecutor, DecoderRuntimePhaseSchedule,
};
use crate::component_binding::{
    bind_selected, classify, validate_phase, validate_selected_operation, Implementation,
    InvalidBoundStep, SelectedComponents,
};

const SCHEMA: &str = "pllm.gkm.decoder-reference.v1";
const INPUT_BOUND: u64 = 1 << 31;
const MAX_CONFIG_BYTES: usize = 16 << 10;
const MAX_ELEMENTS: u64 = 1 << 18;
const MAX_SOURCE_BYTES: usize = 8 << 20;

/// Full-share SDK capacity. These public limits are independent of the numeric
/// profile; the historical in-process capsule retains its original limits.
pub const GKM_DECODER_SOURCE_BYTES: usize = gkm::Capacity::Decoder.max_publication_bytes();
pub const GKM_DECODER_EXECUTION_BYTES: usize = gkm::Capacity::Decoder.max_publication_bytes();

#[derive(Serialize)]
struct DecoderResources {
    profile: &'static str,
    source_import_bytes: usize,
    publication_payload_bytes: usize,
    window_payload_bytes: usize,
    client_tensor_bytes: usize,
    accepted_result_bytes: usize,
    source_chunk_bytes: usize,
    publication_frame_bytes: usize,
    operation_frame_bytes: usize,
    client_payload_bytes: usize,
    worker_payload_bytes: usize,
    request_linear_macs_per_worker: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    outer_transport_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_coefficient_copy_bytes: Option<usize>,
}

#[derive(Serialize)]
struct TransportResources {
    profile: &'static str,
    reserved_windows: usize,
    source_payload_bytes: usize,
    import_workspace_bytes: usize,
    publication_payload_bytes: usize,
    publisher_workspace_bytes: usize,
    window_payload_bytes: usize,
    native_admitted_payload_bytes: usize,
    generic_native_admitted_payload_bytes: usize,
}

#[derive(Serialize)]
struct Member {
    weight: String,
    bias: Option<String>,
    rows: usize,
    offset: usize,
}

#[derive(Serialize)]
struct Matrix {
    id: Digest,
    rows: usize,
    columns: usize,
    members: Vec<Member>,
}

/// Constructed only by source lowering and checked against the explicitly
/// selected capacity before any tensor or secret allocation.
#[derive(Serialize)]
pub struct GkmDecoderContract {
    schema_version: &'static str,
    numeric_profile: &'static str,
    local_numeric_profile: &'static str,
    acceptance: &'static str,
    input_bound: u64,
    input_rows: usize,
    decode_steps: usize,
    plan: DecoderPlan,
    prefill: DecoderRuntimePhaseSchedule,
    decode: DecoderRuntimePhaseSchedule,
    matrices: Vec<Matrix>,
    table: String,
    vocabulary: usize,
    width: usize,
    source_shapes: BTreeMap<String, Vec<usize>>,
    local_tensors: BTreeSet<String>,
    source_float32_bytes: usize,
    capsule_payload_allowance_bytes: usize,
    execution_payload_estimate_bytes: usize,
    window_steps: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    decoder_resources: Option<DecoderResources>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transport_resources: Option<TransportResources>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_binding_digest: Option<Digest>,
}

fn reject(message: &str) -> String {
    format!("GKM decoder reference: {message}")
}

fn bind_shape(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    id: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    if id.is_empty() || shapes.get(id).is_some_and(|prior| *prior != shape) {
        return Err(reject("conflicting source shape"));
    }
    shapes.insert(id.to_owned(), shape);
    Ok(())
}

fn tensor_owner_bytes(shape: &[u64]) -> Result<usize, String> {
    shape.iter().try_fold(8usize, |bytes, dimension| {
        usize::try_from(*dimension)
            .ok()
            .and_then(|n| bytes.checked_mul(n))
            .ok_or_else(|| reject("tensor owner overflow"))
    })
}

fn owner_sum(values: impl IntoIterator<Item = usize>) -> Result<usize, String> {
    values
        .into_iter()
        .try_fold(0usize, |sum, n| sum.checked_add(n))
        .ok_or_else(|| reject("owner allowance overflow"))
}

fn owner_product(values: &[usize]) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |product, n| product.checked_mul(*n))
        .ok_or_else(|| reject("owner allowance overflow"))
}

fn decoder_tensor_bytes(
    plan: &DecoderPlan,
    prefill: &DecoderRuntimePhaseSchedule,
    decode: &DecoderRuntimePhaseSchedule,
    vocabulary: usize,
    decode_steps: usize,
) -> Result<usize, String> {
    let mut peak = 0usize;
    for (graph, schedule) in [(&plan.prefill, prefill), (&plan.decode, decode)] {
        // Tokens, absolute positions, attention mask and sequence lengths stay
        // owned independently of graph outputs. Price eight-byte mask elements
        // and retain these public-geometry roots for the complete phase.
        let inputs = owner_product(&[
            graph.batch as usize,
            owner_sum([owner_product(&[3, graph.query_sequence as usize])?, 1])?,
            8,
        ])?;
        let operations: BTreeMap<_, _> = graph
            .operations
            .iter()
            .map(|op| (op.id.as_str(), op))
            .collect();
        let logits = graph
            .operations
            .iter()
            .find(|op| op.operator == ModelOperator::GreedyTokenSelection)
            .and_then(|op| op.inputs.first())
            .ok_or_else(|| reject("missing logits root"))?;
        let mut remaining = BTreeMap::new();
        for key in schedule.steps.iter().flat_map(|step| &step.input_ids) {
            *remaining.entry(key.as_str()).or_insert(0usize) += 1;
        }
        let mut owners = BTreeMap::<&str, BTreeSet<&str>>::new();
        let mut sizes = BTreeMap::<&str, usize>::new();
        let mut live = BTreeSet::new();
        for step in &schedule.steps {
            let combined = step.operation_ids[0].as_str();
            if step.executor != DecoderRuntimeExecutor::ClientLocal {
                // Grouped gate/up members alias one complete reconstructed array.
                let bytes = step.operation_ids.iter().try_fold(0usize, |bytes, id| {
                    owner_sum([
                        bytes,
                        tensor_owner_bytes(&operations[id.as_str()].output_shape)?,
                    ])
                })?;
                sizes.insert(combined, bytes);
            }
            for id in &step.operation_ids {
                let id = id.as_str();
                let op = operations[id];
                let mut roots = BTreeSet::new();
                if step.executor != DecoderRuntimeExecutor::ClientLocal {
                    roots.insert(combined);
                } else {
                    sizes.insert(id, tensor_owner_bytes(&op.output_shape)?);
                    roots.insert(id);
                    if matches!(
                        op.operator,
                        ModelOperator::Reshape
                            | ModelOperator::LastToken
                            | ModelOperator::CacheSuffix
                    ) {
                        // Price both a possible copy and every retained full root.
                        // Narrow views cannot retire their much larger backing owner.
                        for input in &op.inputs {
                            if let Some(prior) = owners.get(input.as_str()) {
                                roots.extend(prior);
                            }
                        }
                    }
                }
                owners.insert(id, roots);
                live.insert(id);
            }
            let roots: BTreeSet<_> = live
                .iter()
                .flat_map(|id| owners[id].iter().copied())
                .collect();
            let bytes = owner_sum(std::iter::once(inputs).chain(roots.iter().map(|id| sizes[id])))?;
            peak = peak.max(bytes);
            for key in &step.input_ids {
                let count = remaining.get_mut(key.as_str()).unwrap();
                *count -= 1;
                if *count == 0 && key != logits {
                    live.remove(key.as_str());
                }
            }
        }
    }
    // Full KV grows geometrically in the common local executor. Price original
    // storage, replacement, final snapshots and byte copies independently of
    // temporary tensor lifetimes. Eight-byte elements also cover numeric scratch.
    let key_rows = plan.decode.maximum_key_sequence as usize;
    let capacity_rows = key_rows
        .max(64)
        .checked_next_power_of_two()
        .ok_or_else(|| reject("KV capacity overflow"))?;
    let state = plan
        .decode
        .state_outputs
        .iter()
        .try_fold(0usize, |total, state| {
            let per_row = tensor_owner_bytes(&state.shape)?
                .checked_div(key_rows)
                .ok_or_else(|| reject("KV capacity overflow"))?;
            owner_sum([total, owner_product(&[per_row, capacity_rows])?])
        })?;
    owner_sum([
        owner_product(&[peak, 6])?,
        owner_product(&[state, 6])?,
        owner_product(&[vocabulary, 4, owner_sum([decode_steps, 3])?])?,
    ])
}

impl GkmDecoderContract {
    pub fn compile(config: &[u8], input_rows: usize, decode_steps: usize) -> Result<Self, String> {
        Self::compile_with_capacity(config, input_rows, decode_steps, gkm::Capacity::Capsule)
    }

    pub fn compile_with_capacity(
        config: &[u8],
        input_rows: usize,
        decode_steps: usize,
        capacity: gkm::Capacity,
    ) -> Result<Self, String> {
        Self::compile_owned(
            config, input_rows, decode_steps, capacity, false, SelectedComponents::Legacy,
        )
    }

    pub fn compile_streamed_with_capacity(
        config: &[u8],
        input_rows: usize,
        decode_steps: usize,
        capacity: gkm::Capacity,
    ) -> Result<Self, String> {
        Self::compile_owned(
            config, input_rows, decode_steps, capacity, true, SelectedComponents::Legacy,
        )
    }

    fn compile_owned(
        config: &[u8],
        input_rows: usize,
        decode_steps: usize,
        capacity: gkm::Capacity,
        streamed: bool,
        selected: SelectedComponents<'_>,
    ) -> Result<Self, String> {
        let expanded = capacity == gkm::Capacity::Decoder;
        let max_elements = if expanded { 1 << 24 } else { MAX_ELEMENTS };
        if config.is_empty()
            || config.len() > MAX_CONFIG_BYTES
            || !(1..=if expanded { 512 } else { 32 }).contains(&input_rows)
            || decode_steps > if expanded { 64 } else { 8 }
        {
            return Err(reject("configuration or workload limit"));
        }
        let plan = pllm_models::lower_model_json_with_budget(
            config,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: input_rows as u64,
                max_new_tokens: decode_steps as u64 + 1,
            },
            pllm_models::DecoderExpansionBudget {
                max_layers: if expanded { 64 } else { gkm::MAX_STAGES as u64 },
                max_feature_width: capacity.max_columns() as u64,
                max_attention_heads: if expanded {
                    128
                } else {
                    gkm::MAX_DIMENSION as u64
                },
                // The complete head uses the same matrix-row limit as the body.
                max_vocabulary: capacity.max_rows() as u64,
                max_tensor_elements: if expanded {
                    capacity.max_matrix_elements() as u64
                } else {
                    MAX_ELEMENTS
                },
            },
        )
        .map_err(|error| error.to_string())?;
        // Reuse the metadata authority, including cross-phase matrix geometry.
        crate::lower_decoder_matrix_inventory(&plan)?;
        plan.validate().map_err(|error| error.to_string())?;
        let mut source_shapes = BTreeMap::new();
        let mut local_tensors = BTreeSet::new();
        for graph in [&plan.prefill, &plan.decode] {
            for op in &graph.operations {
                let implementation = bind_selected(op.operator, selected)
                    .map_err(|_| reject("unsupported operator, representation, or tensor geometry"))?;
                if matches!(selected, SelectedComponents::Explicit(_)) {
                    validate_selected_operation(graph, op, implementation)
                        .map_err(|error| reject(&error.to_string()))?;
                }
                if op
                    .output_shape
                    .iter()
                    .try_fold(1u64, |n, d| n.checked_mul(*d))
                    .is_none_or(|n| n > max_elements)
                    || ["output_dtype", "compute_dtype"].iter().any(|key| {
                        op.attributes.get(*key).is_some_and(|dtype| {
                            op.operator != ModelOperator::OutputHead
                                || dtype.as_str() != Some("model_native")
                        })
                    })
                    || op.attributes.get("frequency_scaling").is_some()
                    || op.attributes.get("attention_domain").is_some_and(|domain| {
                        domain["layout"].as_str() != Some("batch_kv_heads_sequence_feature")
                    })
                    || (op.operator == ModelOperator::CacheSuffix
                        && op.attributes["semantics"].as_str() != Some("visible_valid_prefix"))
                {
                    return Err(reject(
                        "unsupported operator, representation, or tensor geometry",
                    ));
                }
                if implementation == Implementation::RmsNorm {
                    let weight = op.attributes["weight"]
                        .as_str()
                        .ok_or_else(|| reject("normalization weight"))?;
                    bind_shape(
                        &mut source_shapes,
                        weight,
                        vec![*op.output_shape.last().unwrap() as usize],
                    )?;
                    local_tensors.insert(weight.to_owned());
                }
            }
            if graph
                .state_outputs
                .iter()
                .any(|s| !matches!(s.kind, StateKind::Key | StateKind::Value))
            {
                return Err(reject("requires ordinary dense full KV"));
            }
        }
        let lower = |graph| {
            lower_phase(
                graph,
                DecoderRuntimeExecutor::RemoteStage,
                0,
                &BTreeSet::new(),
                &BTreeSet::new(),
            )
        };
        let prefill = lower(&plan.prefill)?;
        let decode = lower(&plan.decode)?;
        validate_phase(&plan.prefill, &prefill).map_err(|e| reject(&e.to_string()))?;
        validate_phase(&plan.decode, &decode).map_err(|e| reject(&e.to_string()))?;
        let operations: BTreeMap<_, _> = plan
            .prefill
            .operations
            .iter()
            .map(|op| (op.id.as_str(), op))
            .collect();
        let heads: Vec<_> = plan
            .prefill
            .operations
            .iter()
            .filter(|op| op.operator == ModelOperator::OutputHead)
            .collect();
        if heads.len() != 1 {
            return Err(reject("requires one full output head"));
        }
        let vocabulary = *heads[0].output_shape.last().unwrap() as usize;
        let mut table = None;
        let mut width = 0;
        let mut matrices = Vec::new();
        let mut calls = 0usize;
        let mut linear_macs = 0u64;
        for step in &prefill.steps {
            if step.executor == DecoderRuntimeExecutor::ClientLocal {
                continue;
            }
            let first = operations[step.operation_ids[0].as_str()];
            let implementation = bind_selected(first.operator, selected).map_err(|e| reject(&e.to_string()))?;
            if implementation == Implementation::Lookup {
                if table.is_some() || step.operation_ids.len() != 1 {
                    return Err(reject("requires one original token table"));
                }
                width = step.outputs[0].stage_width as usize;
                table = Some(step.weight_ids[0].clone());
                bind_shape(
                    &mut source_shapes,
                    &step.weight_ids[0],
                    vec![vocabulary, width],
                )?;
                calls += input_rows;
                continue;
            }
            if !implementation.matrix() {
                return Err(reject("unsupported matrix implementation"));
            }
            let columns = *operations[first.inputs[0].as_str()]
                .output_shape
                .last()
                .unwrap() as usize;
            let mut rows = 0;
            let mut members = Vec::new();
            for (id, out) in step.operation_ids.iter().zip(&step.outputs) {
                let op = operations[id.as_str()];
                if !bind_selected(op.operator, selected).is_ok_and(|code| code.matrix()) {
                    return Err(reject("unsupported matrix member implementation"));
                }
                let weight = op.attributes["weight"]
                    .as_str()
                    .ok_or_else(|| reject("matrix weight"))?
                    .to_owned();
                if matches!(selected, SelectedComponents::Explicit(_))
                    && op.operator == ModelOperator::OutputHead
                    && op.attributes["tied"].as_bool() == Some(true)
                    && table.as_deref() != Some(weight.as_str())
                {
                    return Err(reject("tied head differs from original token table"));
                }
                let bias = op
                    .attributes
                    .get("bias")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
                let count = out.stage_width as usize;
                bind_shape(&mut source_shapes, &weight, vec![count, columns])?;
                if let Some(bias) = &bias {
                    bind_shape(&mut source_shapes, bias, vec![count])?;
                    local_tensors.insert(bias.clone());
                }
                members.push(Member {
                    weight,
                    bias,
                    rows: count,
                    offset: rows,
                });
                rows += count;
            }
            let id = canonical_digest(
                "pllm.gkm.decoder-stage.v1",
                &(plan.digest(), matrices.len(), columns, rows, &members),
            );
            matrices.push(Matrix {
                id,
                rows,
                columns,
                members,
            });
            let input = operations[first.inputs[0].as_str()];
            let row_count: u64 = input.output_shape[..input.output_shape.len() - 1]
                .iter()
                .product();
            // Source lowering must select only the last prefill row for the head.
            if (first.operator == ModelOperator::OutputHead && row_count != 1)
                || (first.operator == ModelOperator::Linear && row_count != input_rows as u64)
            {
                return Err(reject("unsupported linear row schedule"));
            }
            calls += row_count as usize;
            linear_macs = linear_macs
                .checked_add((rows as u64) * (columns as u64) * (row_count + decode_steps as u64))
                .ok_or_else(|| reject("linear work overflow"))?;
        }
        let table = table.ok_or_else(|| reject("missing token table"))?;
        let window_steps = calls + decode_steps * (matrices.len() + 1);
        if window_steps > capacity.max_window_steps() {
            return Err(reject("complete window exceeds capsule capacity"));
        }
        if expanded && linear_macs > 1u64 << 37 {
            return Err(reject("complete request exceeds linear work capacity"));
        }
        let shapes: Vec<_> = matrices.iter().map(|m| (m.rows, m.columns)).collect();
        let generic_resources = gkm::PublicationResources::for_shapes_with_capacity(
            &shapes, vocabulary, width, capacity,
        )
        .map_err(|e| e.to_string())?;
        let specialized = if streamed {
            gkm::PublicationResources::for_streamed_single_window(
                &shapes, vocabulary, width, capacity,
            )
            .map_err(|e| e.to_string())?
        } else {
            generic_resources
        };
        let resources = generic_resources;
        let transport_resources = if streamed {
            Some(TransportResources {
                profile: "pllm.gkm.streamed-single-window.v1",
                reserved_windows: 1,
                source_payload_bytes: owner_sum([
                    specialized.source_weight_bytes,
                    specialized.source_bound_bytes,
                    specialized.source_table_bytes,
                ])?,
                import_workspace_bytes: specialized.max_import_workspace_bytes,
                publication_payload_bytes: specialized.native_publication_payload_bytes,
                publisher_workspace_bytes: specialized.owner_scalar_workspace_bytes,
                window_payload_bytes: specialized.max_window_payload_bytes,
                native_admitted_payload_bytes: specialized.admitted_payload_bytes,
                generic_native_admitted_payload_bytes: generic_resources.admitted_payload_bytes,
            })
        } else {
            None
        };
        let source_float32_bytes = source_shapes
            .values()
            .try_fold(0usize, |sum, shape| {
                shape
                    .iter()
                    .try_fold(4usize, |n, d| n.checked_mul(*d))
                    .and_then(|n| sum.checked_add(n))
            })
            .ok_or_else(|| reject("source byte overflow"))?;
        if source_float32_bytes
            > if expanded {
                GKM_DECODER_SOURCE_BYTES
            } else {
                MAX_SOURCE_BYTES
            }
        {
            return Err(reject("source payload limit"));
        }
        let (execution_payload_estimate_bytes, decoder_resources) = if expanded {
            let local_bytes = local_tensors.iter().try_fold(0usize, |sum, name| {
                owner_sum([
                    sum,
                    owner_product(&[4, owner_product(&source_shapes[name])?])?,
                ])
            })?;
            let source_import_bytes = owner_sum([
                resources.source_weight_bytes,
                resources.source_bound_bytes,
                resources.source_table_bytes,
                local_bytes,
            ])?;
            let tensor_bytes =
                decoder_tensor_bytes(&plan, &prefill, &decode, vocabulary, decode_steps)?;
            // Returned immutable float32 logits and active full KV are already
            // covered by the tensor overlap allowance. Expose their exact bound
            // (plus a bounded acceptance receipt) before any SDK execution.
            let accepted_result_bytes = owner_sum([
                16 << 10,
                owner_product(&[4, vocabulary, owner_sum([decode_steps, 1])?])?,
                plan.decode
                    .state_outputs
                    .iter()
                    .try_fold(0usize, |sum, state| {
                        owner_sum([sum, tensor_owner_bytes(&state.shape)? / 2])
                    })?,
            ])?;
            // The streamed importer transfers its final owners. It has no whole
            // float32 dictionary or duplicate encoded weights. Core admission
            // includes source, publication and publisher scratch. Streamed SDK
            // contracts reserve one complete window and bounded import scratch;
            // general reference contracts retain eager/eight-window pricing.
            // Outer frames coexist with native export/import plus HTTP/msgpack
            // copies. Tokenizer metadata/indexes and graph metadata are separate.
            let outer = 4 * capacity.max_publication_wire_bytes()
                + 8 * capacity.max_wire_frame_bytes()
                + (384 << 20);
            let worker_bytes = owner_sum([
                resources.admitted_payload_bytes,
                outer,
                owner_product(&[2, local_bytes])?,
            ])?;
            let client_bytes = owner_sum([worker_bytes, tensor_bytes])?;
            (
                client_bytes,
                Some(DecoderResources {
                    profile: "pllm.gkm.decoder-capacity.v1",
                    source_import_bytes,
                    publication_payload_bytes: resources.native_publication_payload_bytes,
                    window_payload_bytes: resources.max_window_payload_bytes,
                    client_tensor_bytes: tensor_bytes,
                    accepted_result_bytes,
                    source_chunk_bytes: GKM_SOURCE_CHUNK_BYTES,
                    publication_frame_bytes: capacity.max_publication_wire_bytes(),
                    operation_frame_bytes: capacity.max_wire_frame_bytes(),
                    client_payload_bytes: client_bytes,
                    worker_payload_bytes: worker_bytes,
                    request_linear_macs_per_worker: linear_macs,
                    outer_transport_bytes: None,
                    local_coefficient_copy_bytes: None,
                }),
            )
        } else {
            // Preserve the original Capsule estimate and digest, without ever
            // constructing discarded eager-copy estimates for Decoder owners.
            let mut graph_bytes = 0;
            for graph in [&plan.prefill, &plan.decode] {
                let bytes = graph.operations.iter().try_fold(0usize, |sum, op| {
                    owner_sum([sum, tensor_owner_bytes(&op.output_shape)?])
                })?;
                graph_bytes = graph_bytes.max(bytes);
            }
            (
                owner_sum([
                    resources.admitted_payload_bytes,
                    owner_product(&[3, source_float32_bytes])?,
                    owner_product(&[6, graph_bytes])?,
                    16 << 20,
                ])?,
                None,
            )
        };
        let specialize_total = |bytes: usize| {
            bytes
                .checked_sub(generic_resources.admitted_payload_bytes)
                .and_then(|n| n.checked_add(specialized.admitted_payload_bytes))
                .ok_or_else(|| reject("streamed owner allowance overflow"))
        };
        let admitted_execution_bytes = specialize_total(execution_payload_estimate_bytes)?;
        if admitted_execution_bytes
            > if expanded {
                GKM_DECODER_EXECUTION_BYTES
            } else {
                64 << 20
            }
        {
            return Err(reject(if expanded {
                "complete decoder payload estimate exceeds 4 GiB"
            } else {
                "complete reference payload estimate exceeds 64 MiB"
            }));
        }
        // Construct the unchanged general source-binding contract first. Streamed
        // SDK ownership is substituted below; caller/Python copies remain in the
        // outer envelope, never a claim of a process-RSS bound.
        let mut result = Self {
            schema_version: SCHEMA,
            numeric_profile: gkm::NUMERIC_PROFILE,
            local_numeric_profile: "semantic.float32.dense-full-kv.client-bias-after-div22.v1",
            acceptance: "fixed-length-greedy.complete-window.v1",
            input_bound: INPUT_BOUND,
            input_rows,
            decode_steps,
            plan,
            prefill,
            decode,
            matrices,
            table,
            vocabulary,
            width,
            source_shapes,
            local_tensors,
            source_float32_bytes,
            capsule_payload_allowance_bytes: resources.admitted_payload_bytes,
            execution_payload_estimate_bytes,
            window_steps,
            decoder_resources,
            transport_resources: None,
            source_binding_digest: None,
        };
        result.validate_bindings().map_err(|e| reject(&e.to_string()))?;
        if streamed {
            // Source/artifact identity remains bound to the unchanged general
            // numeric decoder contract. Ownership admission is additionally bound
            // by the SDK contract/request context, not folded into source bytes.
            result.source_binding_digest = Some(result.digest());
            result.capsule_payload_allowance_bytes = specialized.admitted_payload_bytes;
            result.execution_payload_estimate_bytes = admitted_execution_bytes;
            if let Some(resources) = &mut result.decoder_resources {
                let outer = owner_sum([
                    owner_product(&[4, capacity.max_publication_wire_bytes()])?,
                    owner_product(&[8, capacity.max_wire_frame_bytes()])?,
                    384 << 20,
                ])?;
                let local_copies = resources
                    .worker_payload_bytes
                    .checked_sub(generic_resources.admitted_payload_bytes)
                    .and_then(|n| n.checked_sub(outer))
                    .ok_or_else(|| reject("streamed local allowance overflow"))?;
                resources.source_import_bytes = owner_sum([
                    resources.source_import_bytes,
                    specialized.max_import_workspace_bytes,
                ])?;
                resources.worker_payload_bytes = specialize_total(resources.worker_payload_bytes)?;
                resources.client_payload_bytes = specialize_total(resources.client_payload_bytes)?;
                resources.outer_transport_bytes = Some(outer);
                resources.local_coefficient_copy_bytes = Some(local_copies);
            }
            result.transport_resources = transport_resources;
        }
        Ok(result)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        canonical_bytes(self)
    }
    pub fn digest(&self) -> Digest {
        canonical_digest(SCHEMA, self)
    }
    /// Source identity excludes this SDK's reduced ownership overlap. The
    /// serialized digest pins the unchanged general numeric decoder contract;
    /// the SDK/request digest independently binds transport admission metadata.
    pub fn source_binding_digest(&self) -> Digest {
        self.source_binding_digest
            .clone()
            .unwrap_or_else(|| self.digest())
    }
    pub fn source_shapes(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.source_shapes
    }

    pub fn capacity(&self) -> gkm::Capacity {
        if self.decoder_resources.is_some() {
            gkm::Capacity::Decoder
        } else {
            gkm::Capacity::Capsule
        }
    }

    pub fn native_publication_resources(&self) -> Result<gkm::PublicationResources, String> {
        let shapes: Vec<_> = self
            .matrices
            .iter()
            .map(|matrix| (matrix.rows, matrix.columns))
            .collect();
        if self.transport_resources.is_some() {
            gkm::PublicationResources::for_streamed_single_window(
                &shapes,
                self.vocabulary,
                self.width,
                self.capacity(),
            )
        } else {
            gkm::PublicationResources::for_shapes_with_capacity(
                &shapes,
                self.vocabulary,
                self.width,
                self.capacity(),
            )
        }
        .map_err(|error| error.to_string())
    }

    fn window_plan(&self) -> Result<Vec<Step>, InvalidBoundStep> {
        self.validate_bindings()?;
        let mut result = Vec::with_capacity(self.window_steps);
        for (phase, repeats) in [(&self.prefill, 1), (&self.decode, self.decode_steps)] {
            for _ in 0..repeats {
                let mut index = 0;
                for step in &phase.steps {
                    if step.executor == DecoderRuntimeExecutor::ClientLocal {
                        continue;
                    }
                    let implementation = classify(step.operators[0])?;
                    if implementation == Implementation::Lookup {
                        result.extend(std::iter::repeat_n(
                            Step::Lookup,
                            phase.query_sequence as usize,
                        ));
                    } else if implementation.matrix() {
                        let rows = if implementation == Implementation::OutputHead {
                            1
                        } else {
                            phase.query_sequence as usize
                        };
                        result.extend(std::iter::repeat_n(
                            Step::Linear(stage_id(&self.matrices[index])),
                            rows,
                        ));
                        index += 1;
                    } else {
                        return Err(InvalidBoundStep);
                    }
                }
            }
        }
        Ok(result)
    }

    fn validate_bindings(&self) -> Result<(), InvalidBoundStep> {
        if self.numeric_profile != gkm::NUMERIC_PROFILE
            || self.input_bound != INPUT_BOUND
            || self.local_numeric_profile != "semantic.float32.dense-full-kv.client-bias-after-div22.v1"
            || self.acceptance != "fixed-length-greedy.complete-window.v1"
            || !self.plan.token_feedback
        {
            return Err(InvalidBoundStep);
        }
        for (graph, schedule) in [(&self.plan.prefill, &self.prefill), (&self.plan.decode, &self.decode)] {
            validate_phase(graph, schedule)?;
            if graph.state_inputs.iter().chain(&graph.state_outputs)
                .any(|state| !matches!(state.kind, StateKind::Key | StateKind::Value))
            {
                return Err(InvalidBoundStep);
            }
            let mut matrix_index = 0;
            let mut lookup_count = 0;
            for step in &schedule.steps {
                let first = graph.operations.iter().find(|op| op.id == step.operation_ids[0])
                    .ok_or(InvalidBoundStep)?;
                let code = classify(first.operator)?;
                if code == Implementation::Lookup {
                    lookup_count += 1;
                    if first.inputs.len() != 1 || first.inputs[0] != "input.tokens"
                        || step.weight_ids.len() != 1 || step.weight_ids[0] != self.table
                        || first.attributes["weight"].as_str() != Some(self.table.as_str())
                        || step.outputs[0].stage_offset != 0
                        || step.outputs[0].stage_width != self.width as u64
                        || self.source_shapes.get(&self.table).map(Vec::as_slice)
                            != Some([self.vocabulary, self.width].as_slice())
                    {
                        return Err(InvalidBoundStep);
                    }
                } else if code.matrix() {
                    let matrix = self.matrices.get(matrix_index).ok_or(InvalidBoundStep)?;
                    matrix_index += 1;
                    if matrix.members.len() != step.operation_ids.len()
                        || step.weight_ids.len() != matrix.members.len()
                    {
                        return Err(InvalidBoundStep);
                    }
                    let mut offset = 0;
                    for (position, id) in step.operation_ids.iter().enumerate() {
                        let op = graph.operations.iter().find(|op| op.id == *id).ok_or(InvalidBoundStep)?;
                        let member = &matrix.members[position];
                        let output = &step.outputs[position];
                        let input_id = op.inputs.first().ok_or(InvalidBoundStep)?;
                        let input = graph.operations.iter().find(|op| op.id == *input_id).ok_or(InvalidBoundStep)?;
                        if !classify(op.operator)?.matrix() || op.inputs != first.inputs
                            || input.output_shape.last().copied() != Some(matrix.columns as u64)
                            || op.attributes["weight"].as_str() != Some(member.weight.as_str())
                            || op.attributes.get("bias").and_then(serde_json::Value::as_str) != member.bias.as_deref()
                            || step.weight_ids[position] != member.weight
                            || member.offset != offset || output.stage_offset != offset as u64
                            || output.stage_width != member.rows as u64
                            || op.output_shape.last().copied() != Some(member.rows as u64)
                            || self.source_shapes.get(&member.weight).map(Vec::as_slice)
                                != Some([member.rows, matrix.columns].as_slice())
                            || member.bias.as_ref().is_some_and(|bias|
                                !self.local_tensors.contains(bias)
                                || self.source_shapes.get(bias).map(Vec::as_slice) != Some([member.rows].as_slice()))
                        {
                            return Err(InvalidBoundStep);
                        }
                        offset = offset.checked_add(member.rows).ok_or(InvalidBoundStep)?;
                    }
                    if offset != matrix.rows { return Err(InvalidBoundStep); }
                } else if code == Implementation::RmsNorm {
                    let weight = first.attributes["weight"].as_str().ok_or(InvalidBoundStep)?;
                    if !self.local_tensors.contains(weight) {
                        return Err(InvalidBoundStep);
                    }
                } else if code == Implementation::GreedyTokenSelection {
                    let input = first.inputs.first().ok_or(InvalidBoundStep)?;
                    if first.inputs.len() != 1 || !graph.operations.iter().any(|op|
                        op.id == *input && op.operator == ModelOperator::OutputHead)
                    {
                        return Err(InvalidBoundStep);
                    }
                } else if code == Implementation::TokenFeedback {
                    let input = first.inputs.first().ok_or(InvalidBoundStep)?;
                    if first.inputs.len() != 1 || graph.output != first.id
                        || !graph.operations.iter().any(|op|
                            op.id == *input && op.operator == ModelOperator::GreedyTokenSelection)
                    {
                        return Err(InvalidBoundStep);
                    }
                }
            }
            if lookup_count != 1 || matrix_index != self.matrices.len()
                || graph.operations.iter().filter(|op| op.operator == ModelOperator::GreedyTokenSelection).count() != 1
                || graph.operations.iter().filter(|op| op.operator == ModelOperator::TokenFeedback).count() != 1
            {
                return Err(InvalidBoundStep);
            }
        }
        Ok(())
    }

    /// Read-only implementation identity, not a matrix ordinal or completion.
    fn bound_step(&self, phase: &str, operation_id: &str) -> Result<u8, InvalidBoundStep> {
        let (graph, schedule) = match phase {
            "prefill" => (&self.plan.prefill, &self.prefill),
            "decode" => (&self.plan.decode, &self.decode),
            _ => return Err(InvalidBoundStep),
        };
        let op = graph.operations.iter().find(|op| op.id == operation_id).ok_or(InvalidBoundStep)?;
        if schedule.steps.iter().flat_map(|step| &step.operation_ids)
            .filter(|id| id.as_str() == operation_id).count() != 1
        {
            return Err(InvalidBoundStep);
        }
        Ok(classify(op.operator)? as u8)
    }
}

fn stage_id(matrix: &Matrix) -> [u8; 32] {
    Sha256::digest(matrix.id.as_str().as_bytes()).into()
}

struct BoundSource {
    matrices: Vec<I16Matrix>,
    table: FloatTable,
    local: BTreeMap<String, Vec<u8>>,
    digest: [u8; 32],
}

fn bind_source(
    contract: &GkmDecoderContract,
    tensors: &BTreeMap<String, &[u8]>,
) -> Result<BoundSource, String> {
    if contract.capacity() != gkm::Capacity::Capsule {
        return Err(reject(
            "decoder capacity requires bounded streaming source import",
        ));
    }
    if tensors.len() != contract.source_shapes.len() {
        return Err(reject("source tensor set mismatch"));
    }
    let mut hash = Sha256::new();
    hash.update(b"pllm.gkm.decoder-source.v1");
    hash.update(contract.source_binding_digest().as_str().as_bytes());
    for (name, shape) in &contract.source_shapes {
        let bytes = tensors
            .get(name)
            .ok_or_else(|| reject("missing source tensor"))?;
        if bytes.len() != shape.iter().product::<usize>() * 4
            || bytes
                .chunks_exact(4)
                .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        {
            return Err(reject("source float32 shape or finite-value mismatch"));
        }
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    let digest = hash.finalize().into();
    let mut matrices = Vec::new();
    matrices
        .try_reserve_exact(contract.matrices.len())
        .map_err(|_| reject("allocation"))?;
    for matrix in &contract.matrices {
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(matrix.rows * matrix.columns)
            .map_err(|_| reject("allocation"))?;
        for member in &matrix.members {
            for b in tensors[&member.weight].chunks_exact(4) {
                encoded.push(
                    gkm::encode_weight(f64::from(f32::from_le_bytes(b.try_into().unwrap())))
                        .map_err(|e| e.to_string())?,
                );
            }
        }
        matrices.push(
            I16Matrix::from_encoded_owned(
                matrix.rows,
                matrix.columns,
                encoded,
                vec![INPUT_BOUND; matrix.columns],
            )
            .map_err(|e| e.to_string())?,
        );
    }
    let table = FloatTable::from_le_bytes(
        contract.vocabulary,
        contract.width,
        tensors[&contract.table],
    )
    .map_err(|e| e.to_string())?;
    let mut local = BTreeMap::new();
    for name in &contract.local_tensors {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(tensors[name].len())
            .map_err(|_| reject("allocation"))?;
        bytes.extend_from_slice(tensors[name]);
        local.insert(name.clone(), bytes);
    }
    Ok(BoundSource {
        matrices,
        table,
        local,
        digest,
    })
}

/// Single research request owner. Each logical worker evaluates only its own
/// view. They share this process and immutable public weights; no non-collusion
/// or remote transport claim follows. Only finish() returns an acceptance receipt.
pub struct GkmDecoderSession {
    contract: GkmDecoderContract,
    matrices: Vec<I16Matrix>,
    table: Option<FloatTable>,
    local: BTreeMap<String, Vec<u8>>,
    window: Option<gkm::Window>,
    seeded: Option<gkm::SeededWorker>,
    calls: Vec<Step>,
    ordinal: usize,
    source: [u8; 32],
}

impl GkmDecoderSession {
    pub fn new(
        contract: GkmDecoderContract,
        tensors: &BTreeMap<String, &[u8]>,
        seeded: bool,
    ) -> Result<Self, String> {
        let BoundSource {
            matrices,
            table,
            local,
            digest: source,
        } = bind_source(&contract, tensors)?;
        let stages: Vec<_> = contract
            .matrices
            .iter()
            .zip(&matrices)
            .map(|(spec, matrix)| Stage {
                id: stage_id(spec),
                matrix,
            })
            .collect();
        let publication =
            Arc::new(Publication::publish(source, &stages, &table).map_err(|e| e.to_string())?);
        // Trusted in-process source publisher owns this pin. This is not import
        // of an untrusted remotely supplied owner commitment.
        let client = Client::new(Arc::clone(&publication), publication.digest())
            .map_err(|e| e.to_string())?;
        let calls = contract.window_plan().map_err(|e| reject(&e.to_string()))?;
        if calls.len() != contract.window_steps {
            return Err(reject("window count mismatch"));
        }
        let (window, seeded) = if seeded {
            let (window, worker) = client
                .begin_seeded(source, &calls)
                .map_err(|e| e.to_string())?;
            (window, Some(worker))
        } else {
            (
                client.begin(source, &calls).map_err(|e| e.to_string())?,
                None,
            )
        };
        Ok(Self {
            contract,
            matrices,
            table: Some(table),
            local,
            window: Some(window),
            seeded,
            calls,
            ordinal: 0,
            source,
        })
    }

    pub fn contract_bytes(&self) -> Vec<u8> {
        self.contract.canonical_bytes()
    }
    pub fn bound_step(&self, phase: &str, operation_id: &str) -> Result<u8, InvalidBoundStep> {
        if self.window.is_none() { return Err(InvalidBoundStep); }
        self.contract.bound_step(phase, operation_id)
    }
    pub fn local_tensors(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.local
    }
    pub fn abort(&mut self) {
        self.matrices.clear();
        self.table = None;
        self.local.clear();
        self.window = None;
        self.seeded = None;
    }

    pub fn lookup(&mut self, token: usize) -> Result<Vec<u8>, String> {
        let result = (|| {
            let window = self.window.take().ok_or_else(|| reject("retired"))?;
            if self.calls.get(self.ordinal) != Some(&Step::Lookup) {
                return Err(reject("lookup order"));
            }
            let pending = window.lookup(token).map_err(|e| e.to_string())?;
            let table = self.table.as_ref().ok_or_else(|| reject("retired"))?;
            let left = table.evaluate(&pending.share(Worker::Zero));
            let half = pending.freeze_first(left).map_err(|e| e.to_string())?;
            let right = match &mut self.seeded {
                Some(worker) => worker.evaluate_lookup(table, &half.remaining_share()),
                None => table.evaluate(&half.remaining_share()),
            };
            let window = half.freeze_second(right).map_err(|e| e.to_string())?;
            let result = float_bytes(window.tentative_values())?;
            self.window = Some(window);
            self.ordinal += 1;
            Ok(result)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    // Reference-oracle grouping only: this in-process research owner retains its
    // original arithmetic and one final verifier. The SDK batch transport never
    // calls these methods and cannot fall back to a scalar wire execution.
    fn reference_batch_rows(&self, expected: Step) -> Result<usize, String> {
        let plan = self.contract.batch_plan()?;
        plan.descriptors()
            .iter()
            .find(|descriptor| {
                descriptor.first_claim as usize == self.ordinal && descriptor.step == expected
            })
            .map(|descriptor| descriptor.rows as usize)
            .ok_or_else(|| reject("reference batch source boundary mismatch"))
    }

    /// Non-mutating trusted row count before an outer-language token tuple parse.
    pub fn lookup_batch_rows(&self) -> Result<usize, String> {
        if self.window.is_none() {
            return Err(reject("retired"));
        }
        self.reference_batch_rows(Step::Lookup)
    }

    /// Execute one complete source lookup in the local scalar arithmetic oracle.
    pub fn lookup_batch(&mut self, tokens: &[usize]) -> Result<Vec<u8>, String> {
        let result = (|| {
            let rows = self.lookup_batch_rows()?;
            if tokens.len() != rows {
                return Err(reject("reference batch lookup shape"));
            }
            let mut result = Vec::new();
            result
                .try_reserve_exact(rows * self.contract.width * 4)
                .map_err(|_| reject("allocation"))?;
            for token in tokens {
                result.extend(self.lookup(*token)?);
            }
            Ok(result)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    /// Return row-major source-stage output from the unchanged local arithmetic
    /// oracle, with no seed restart and no intermediate externally returned rows.
    pub fn linear_batch(&mut self, stage: usize, bytes: &[u8]) -> Result<Vec<u8>, String> {
        let result = (|| {
            let spec = self
                .contract
                .matrices
                .get(stage)
                .ok_or_else(|| reject("stage"))?;
            let rows = self.reference_batch_rows(Step::Linear(stage_id(spec)))?;
            let stride = spec.columns * 4;
            if bytes.len() != rows * stride {
                return Err(reject("reference batch linear shape"));
            }
            let mut result = Vec::new();
            result
                .try_reserve_exact(rows * spec.rows * 4)
                .map_err(|_| reject("allocation"))?;
            for row in bytes.chunks_exact(stride) {
                result.extend(self.linear(stage, row)?);
            }
            Ok(result)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    pub fn linear(&mut self, stage: usize, bytes: &[u8]) -> Result<Vec<u8>, String> {
        let result = (|| {
            let window = self.window.take().ok_or_else(|| reject("retired"))?;
            let spec = self
                .contract
                .matrices
                .get(stage)
                .ok_or_else(|| reject("stage"))?;
            if self.calls.get(self.ordinal) != Some(&Step::Linear(stage_id(spec)))
                || bytes.len() != spec.columns * 4
            {
                return Err(reject("linear order or input shape"));
            }
            let mut input = Zeroizing::new(Vec::new());
            input
                .try_reserve_exact(spec.columns)
                .map_err(|_| reject("allocation"))?;
            for b in bytes.chunks_exact(4) {
                input.push(
                    gkm::encode_input(f64::from(f32::from_le_bytes(b.try_into().unwrap())))
                        .map_err(|e| e.to_string())?,
                );
            }
            let matrix = &self.matrices[stage];
            let pending = window.linear(&input).map_err(|e| e.to_string())?;
            let left = matrix.evaluate(&pending.share(Worker::Zero));
            let half = pending.freeze_first(left).map_err(|e| e.to_string())?;
            let right = match &mut self.seeded {
                Some(worker) => worker.evaluate_linear(matrix, &half.remaining_share()),
                None => matrix.evaluate(&half.remaining_share()),
            };
            let window = half.freeze_second(right).map_err(|e| e.to_string())?;
            let result = float_bytes(window.tentative_values())?;
            self.window = Some(window);
            self.ordinal += 1;
            Ok(result)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    pub fn finish(&mut self) -> Result<Vec<u8>, String> {
        let result = (|| {
            let window = self.window.take().ok_or_else(|| reject("retired"))?;
            let accepted = window.finish().map_err(|e| e.to_string())?;
            Ok(canonical_bytes(&serde_json::json!({
                "schema_version": "pllm.gkm.decoder-acceptance.v1", "contract": self.contract.digest(),
                "source": Digest::from_sha256(self.source), "publication": Digest::from_sha256(accepted.publication()),
                "request": Digest::from_sha256(accepted.request_id()), "window": Digest::from_sha256(accepted.plan_digest()),
                "steps": accepted.steps(), "seeded": self.seeded.is_some(),
            })))
        })();
        self.abort();
        result
    }
}

fn float_bytes(values: &[f32]) -> Result<Vec<u8>, String> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len() * 4)
        .map_err(|_| reject("allocation"))?;
    for value in values {
        result.extend_from_slice(&value.to_le_bytes());
    }
    Ok(result)
}

#[cfg(test)]
mod binding_tests {
    use super::*;

    fn contract(streamed: bool) -> GkmDecoderContract {
        let config = canonical_bytes(&serde_json::json!({
            "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "vocab_size": 32, "max_position_embeddings": 16, "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": true
        }));
        if streamed {
            GkmDecoderContract::compile_streamed_with_capacity(&config, 2, 1, gkm::Capacity::Capsule).unwrap()
        } else {
            GkmDecoderContract::compile(&config, 2, 1).unwrap()
        }
    }

    #[test]
    fn both_compile_routes_cover_operations_once_without_retained_bindings() {
        for streamed in [false, true] {
            let contract = contract(streamed);
            let before = contract.canonical_bytes();
            assert!(contract.validate_bindings().is_ok());
            for (phase, graph) in [("prefill", &contract.plan.prefill), ("decode", &contract.plan.decode)] {
                for op in &graph.operations {
                    assert_eq!(contract.bound_step(phase, &op.id), Ok(classify(op.operator).unwrap() as u8));
                }
            }
            assert_eq!(before, contract.canonical_bytes());
        }
    }

    #[test]
    fn missing_overlap_unknown_executor_and_cross_phase_members_reject() {
        let mut c = contract(false);
        c.prefill.steps.remove(0);
        assert!(c.validate_bindings().is_err());
        let mut c = contract(false);
        c.decode.steps.push(c.decode.steps[0].clone());
        assert!(c.validate_bindings().is_err());
        let mut c = contract(false);
        c.plan.decode.operations[0].operator = ModelOperator::SecureGather;
        assert!(c.validate_bindings().is_err());
        let mut c = contract(false);
        c.decode.steps[0].executor = DecoderRuntimeExecutor::ClientLocal;
        assert!(c.validate_bindings().is_err());
        for phase in ["prefill", "decode"] {
            for mutation in 0..5 {
                let mut c = contract(false);
                let schedule = if phase == "prefill" { &mut c.prefill } else { &mut c.decode };
                let step = schedule.steps.iter_mut().find(|step|
                    classify(step.operators[0]).unwrap().matrix()).unwrap();
                match mutation {
                    0 => step.weight_ids[0].push_str(".wrong"),
                    1 => step.outputs[0].stage_offset += 1,
                    2 => step.outputs[0].stage_width += 1,
                    3 => { step.operation_ids.pop(); },
                    _ => step.input_ids.clear(),
                }
                assert!(c.validate_bindings().is_err());
                assert!(c.window_plan().is_err());
            }
        }
        for mutation in 0..4 {
            let mut c = contract(false);
            match mutation {
                0 => c.matrices[0].members[0].bias = Some("wrong-bias".into()),
                1 => c.matrices[0].columns += 1,
                2 => c.matrices[0].members[0].offset += 1,
                _ => c.matrices[0].members[0].rows += 1,
            }
            assert!(c.validate_bindings().is_err());
        }
    }
}
