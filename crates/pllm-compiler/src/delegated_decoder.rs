//! Bounded GKM decoder composition: source lowering, original fixed-point
//! boundaries, full client-local float32 state, and one final verified window.
//! This is an in-process research owner, not an installed Pipeline or transport.

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
mod transport;
pub use transport::{GkmTransportClient, GkmTransportWorker};
mod source_import;
pub use source_import::{GkmTransportImport, GKM_SOURCE_CHUNK_BYTES};

use crate::decoder_runtime_schedule::{
    lower_phase, DecoderRuntimeExecutor, DecoderRuntimePhaseSchedule,
};

const SCHEMA: &str = "pllm.gkm.decoder-reference.v1";
const INPUT_BOUND: u64 = 1 << 31;
const MAX_CONFIG_BYTES: usize = 16 << 10;
const MAX_ELEMENTS: u64 = 1 << 18;
const MAX_SOURCE_BYTES: usize = 8 << 20;

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

/// Constructed only by source lowering and checked against the capsule's
/// existing geometry/resource limits before any tensor or secret allocation.
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

impl GkmDecoderContract {
    pub fn compile(config: &[u8], input_rows: usize, decode_steps: usize) -> Result<Self, String> {
        if config.is_empty()
            || config.len() > MAX_CONFIG_BYTES
            || !(1..=32).contains(&input_rows)
            || decode_steps > 8
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
                max_layers: gkm::MAX_STAGES as u64,
                max_feature_width: gkm::MAX_DIMENSION as u64,
                max_attention_heads: gkm::MAX_DIMENSION as u64,
                // The complete head uses the same matrix-row limit as the body.
                max_vocabulary: gkm::MAX_DIMENSION as u64,
                max_tensor_elements: MAX_ELEMENTS,
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
                if !matches!(
                    op.operator,
                    ModelOperator::TokenLookup
                        | ModelOperator::Linear
                        | ModelOperator::OutputHead
                        | ModelOperator::Reshape
                        | ModelOperator::RmsNorm
                        | ModelOperator::RotaryEmbedding
                        | ModelOperator::KvCacheAppend
                        | ModelOperator::CacheSuffix
                        | ModelOperator::AttentionScores
                        | ModelOperator::AttentionScale
                        | ModelOperator::CausalMask
                        | ModelOperator::Softmax
                        | ModelOperator::AttentionValues
                        | ModelOperator::ResidualAdd
                        | ModelOperator::Silu
                        | ModelOperator::Multiply
                        | ModelOperator::LastToken
                        | ModelOperator::GreedyTokenSelection
                        | ModelOperator::TokenFeedback
                ) || op
                    .output_shape
                    .iter()
                    .try_fold(1u64, |n, d| n.checked_mul(*d))
                    .is_none_or(|n| n > MAX_ELEMENTS)
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
                if op.operator == ModelOperator::RmsNorm {
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
        for step in &prefill.steps {
            if step.executor == DecoderRuntimeExecutor::ClientLocal {
                continue;
            }
            let first = operations[step.operation_ids[0].as_str()];
            if first.operator == ModelOperator::TokenLookup {
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
            let columns = *operations[first.inputs[0].as_str()]
                .output_shape
                .last()
                .unwrap() as usize;
            let mut rows = 0;
            let mut members = Vec::new();
            for (id, out) in step.operation_ids.iter().zip(&step.outputs) {
                let op = operations[id.as_str()];
                let weight = op.attributes["weight"]
                    .as_str()
                    .ok_or_else(|| reject("matrix weight"))?
                    .to_owned();
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
        }
        let table = table.ok_or_else(|| reject("missing token table"))?;
        let window_steps = calls + decode_steps * (matrices.len() + 1);
        if window_steps > gkm::MAX_WINDOW_STEPS {
            return Err(reject("complete window exceeds capsule capacity"));
        }
        let shapes: Vec<_> = matrices.iter().map(|m| (m.rows, m.columns)).collect();
        let resources = gkm::PublicationResources::for_shapes(&shapes, vocabulary, width)
            .map_err(|e| e.to_string())?;
        let source_float32_bytes = source_shapes
            .values()
            .try_fold(0usize, |sum, shape| {
                shape
                    .iter()
                    .try_fold(4usize, |n, d| n.checked_mul(*d))
                    .and_then(|n| sum.checked_add(n))
            })
            .ok_or_else(|| reject("source byte overflow"))?;
        if source_float32_bytes > MAX_SOURCE_BYTES {
            return Err(reject("source payload limit"));
        }
        let graph_bytes = [&plan.prefill, &plan.decode]
            .iter()
            .map(|graph| {
                graph
                    .operations
                    .iter()
                    .map(|op| op.output_shape.iter().product::<u64>() * 8)
                    .sum::<u64>()
            })
            .max()
            .unwrap_or(0) as usize;
        // Retain an intentionally conservative estimate for all simultaneous
        // source copies, six graph/scratch/state/snapshot owners and metadata.
        // Interpreter/library overhead and caller source loading are not priced.
        let execution_payload_estimate_bytes = resources.admitted_payload_bytes
            + 3 * source_float32_bytes
            + 6 * graph_bytes
            + (16 << 20);
        if execution_payload_estimate_bytes > 64 << 20 {
            return Err(reject("complete reference payload estimate exceeds 64 MiB"));
        }
        // Native capsule already reserves source owners, eight live windows,
        // commitment work and import scratch. Source float buffers, Python state,
        // metadata and conversions are additional owners, never a process-RSS cap.
        Ok(Self {
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
        })
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        canonical_bytes(self)
    }
    pub fn digest(&self) -> Digest {
        canonical_digest(SCHEMA, self)
    }
    pub fn source_shapes(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.source_shapes
    }

    fn window_plan(&self) -> Vec<Step> {
        let mut result = Vec::with_capacity(self.window_steps);
        for (phase, repeats) in [(&self.prefill, 1), (&self.decode, self.decode_steps)] {
            for _ in 0..repeats {
                let mut index = 0;
                for step in &phase.steps {
                    if step.executor == DecoderRuntimeExecutor::ClientLocal {
                        continue;
                    }
                    if step.operators[0] == ModelOperator::TokenLookup {
                        result.extend(std::iter::repeat_n(
                            Step::Lookup,
                            phase.query_sequence as usize,
                        ));
                    } else {
                        let rows = if step.operators[0] == ModelOperator::OutputHead {
                            1
                        } else {
                            phase.query_sequence as usize
                        };
                        result.extend(std::iter::repeat_n(
                            Step::Linear(stage_id(&self.matrices[index])),
                            rows,
                        ));
                        index += 1;
                    }
                }
            }
        }
        result
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
    if tensors.len() != contract.source_shapes.len() {
        return Err(reject("source tensor set mismatch"));
    }
    let mut hash = Sha256::new();
    hash.update(b"pllm.gkm.decoder-source.v1");
    hash.update(contract.digest().as_str().as_bytes());
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
        let calls = contract.window_plan();
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
