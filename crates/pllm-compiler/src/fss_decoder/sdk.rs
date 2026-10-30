//! Immutable SDK composition and source geometry admission for D13 / opt-in D14.
//!
//! No checkpoint values are read here. Pins bind caller-verified artifacts;
//! numeric domains, range proofs, fresh material, live owners and complete work
//! admission remain the responsibility of Source/Program and the role runtime.
use super::{transport::MaterialDelivery, Capacity, Limits, MAX_NODES, MAX_VALUES};
use crate::{ExperimentDeployment, ExperimentDocument, ExperimentPipeline};
use pllm_models::{
    DecoderExpansionBudget, DecoderPlan, DecoderWorkload, ModelOperation, ModelOperator as Op,
};
use pllm_types::canonical_bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{self, Write};

pub const PROTOCOL: &str = "pllm/fusefss-source-decoder/v1";
pub const TOPOLOGY: &str = "pllm/trusted-dealer-two-share/v1";
pub const STATE: &str = "pllm/fixed-count-private-prefix/v1";
pub const GPT2_NUMERIC: &str = "fusefss.gpt2-complete-ring50-q12.v1";
pub const DENSE_ROPE_NUMERIC: &str = "fusefss.dense-rope-ring50-q12.v1";
pub const DENSE_ROPE_CHECKED_NUMERIC: &str = "fusefss.dense-rope-checked-ranges-ring50-q12.v1";
pub const MAX_CONFIG_BYTES: usize = 16 << 10;
const MAX_EXPERIMENT_BYTES: usize = 32 << 10;
const MAX_REQUESTS: u64 = 64;
pub const MAX_CONTRACT_BYTES: usize = 1 << 20;

struct Count {
    length: usize,
    limit: usize,
    lexical: Lexical,
}
impl Write for Count {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.length = self
            .length
            .checked_add(bytes.len())
            .filter(|&n| n <= self.limit)
            .ok_or_else(|| io::Error::other("FuseFSS bounded serialization exceeded"))?;
        self.lexical.accept(bytes).map_err(io::Error::other)?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub fn encoded_length<T: Serialize>(value: &T, limit: usize) -> Result<usize, String> {
    encoded_length_with_capacity(value, limit, Capacity::Capsule)
}
pub fn encoded_length_with_capacity<T: Serialize>(
    value: &T,
    limit: usize,
    capacity: Capacity,
) -> Result<usize, String> {
    let mut count = Count {
        length: 0,
        limit,
        lexical: Lexical {
            capacity,
            ..Lexical::default()
        },
    };
    serde_json::to_writer(&mut count, value)
        .map_err(|_| reject("serialized metadata exceeds bound"))?;
    Ok(count.length)
}
fn canonical_value(value: &Value, out: &mut Vec<u8>) -> Result<(), String> {
    match value {
        Value::Array(a) => {
            out.push(b'[');
            for (i, v) in a.iter().enumerate() {
                if i != 0 {
                    out.push(b',');
                }
                canonical_value(v, out)?;
            }
            out.push(b']');
        }
        Value::Object(m) => {
            let mut entries = Vec::new();
            entries
                .try_reserve_exact(m.len())
                .map_err(|_| reject("canonical index allocation"))?;
            entries.extend(m.iter());
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            out.push(b'{');
            for (i, (k, v)) in entries.into_iter().enumerate() {
                if i != 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, k).map_err(|_| reject("invalid canonical key"))?;
                out.push(b':');
                canonical_value(v, out)?;
            }
            out.push(b'}');
        }
        _ => serde_json::to_writer(out, value).map_err(|_| reject("invalid canonical value"))?,
    }
    Ok(())
}
/// Two passes: counting sink before Value/canonical output materialization. The
/// bounded Value copy and borrowed-key sorting scratch are separately priced.
pub fn bounded_canonical<T: Serialize>(value: &T, limit: usize) -> Result<Vec<u8>, String> {
    bounded_canonical_with_capacity(value, limit, Capacity::Capsule)
}
pub fn bounded_canonical_with_capacity<T: Serialize>(
    value: &T,
    limit: usize,
    capacity: Capacity,
) -> Result<Vec<u8>, String> {
    encoded_length_with_capacity(value, limit, capacity)?;
    let value = serde_json::to_value(value).map_err(|_| reject("invalid canonical metadata"))?;
    bounded_value_canonical_with_capacity(&value, limit, capacity)
}
pub fn bounded_value_canonical(value: &Value, limit: usize) -> Result<Vec<u8>, String> {
    bounded_value_canonical_with_capacity(value, limit, Capacity::Capsule)
}
pub fn bounded_value_canonical_with_capacity(
    value: &Value,
    limit: usize,
    capacity: Capacity,
) -> Result<Vec<u8>, String> {
    let length = encoded_length_with_capacity(value, limit, capacity)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| reject("canonical allocation"))?;
    canonical_value(value, &mut output)?;
    if output.len() != length || output.capacity() != length {
        return Err(reject("canonical capacity differs"));
    }
    Ok(output)
}

/// Allocation-free lexical envelope before serde owns strings/maps. This is not
/// a JSON parser: serde still checks syntax, and check_metadata checks decoded
/// byte lengths. At most 6 encoded bytes/decoded byte are needed for JSON escapes.
#[derive(Default)]
struct Lexical {
    capacity: Capacity,
    quoted: bool,
    escaped: bool,
    run: usize,
    depth: usize,
    structural: usize,
}
impl Lexical {
    fn accept(&mut self, bytes: &[u8]) -> Result<(), String> {
        for &byte in bytes {
            if self.quoted {
                if !self.escaped && byte == b'"' {
                    self.quoted = false;
                    continue;
                }
                self.run += 1;
                if self.run > 6 * 1024 {
                    return Err(reject("retained JSON string exceeds lexical bound"));
                }
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                }
            } else {
                match byte {
                    b'"' => {
                        self.quoted = true;
                        self.run = 0;
                    }
                    b'{' | b'[' => {
                        self.depth += 1;
                        self.structural += 1;
                    }
                    b'}' | b']' => {
                        self.depth = self.depth.saturating_sub(1);
                    }
                    b':' | b',' => {
                        self.structural += 1;
                    }
                    _ => {}
                }
                let structural_limit = if self.capacity == Capacity::Decoder {
                    1 << 17
                } else {
                    32768
                };
                if self.depth > 16 || self.structural > structural_limit {
                    return Err(reject("JSON structural admission exceeded"));
                }
            }
        }
        Ok(())
    }
}
fn bounded_json_value(bytes: &[u8]) -> Result<Value, String> {
    bounded_json_value_with_capacity(bytes, Capacity::Capsule)
}
fn bounded_json_value_with_capacity(bytes: &[u8], capacity: Capacity) -> Result<Value, String> {
    Lexical {
        capacity,
        ..Lexical::default()
    }
    .accept(bytes)?;
    let value = serde_json::from_slice(bytes).map_err(|_| reject("invalid bounded JSON"))?;
    let mut nodes = 0;
    let mut strings = 0;
    check_metadata_with_capacity(&value, 0, &mut nodes, &mut strings, capacity)?;
    Ok(value)
}

fn check_metadata(
    value: &Value,
    depth: usize,
    nodes: &mut usize,
    strings: &mut usize,
) -> Result<(), String> {
    check_metadata_with_capacity(value, depth, nodes, strings, Capacity::Capsule)
}
fn check_metadata_with_capacity(
    value: &Value,
    depth: usize,
    nodes: &mut usize,
    strings: &mut usize,
    capacity: Capacity,
) -> Result<(), String> {
    *nodes = nodes
        .checked_add(1)
        .ok_or_else(|| reject("metadata overflow"))?;
    let node_limit = if capacity == Capacity::Decoder {
        1 << 16
    } else {
        32768
    };
    if depth > 16 || *nodes > node_limit {
        return Err(reject("metadata nodes/depth exceed native bound"));
    }
    match value {
        Value::String(s) => {
            if s.capacity() > 1024 {
                return Err(reject("retained string exceeds native 1024 bytes"));
            }
            *strings += s.capacity();
        }
        Value::Array(a) => {
            if a.capacity() > 32768 {
                return Err(reject("metadata array exceeds bound"));
            }
            for v in a {
                check_metadata_with_capacity(v, depth + 1, nodes, strings, capacity)?;
            }
        }
        Value::Object(m) => {
            for (k, v) in m {
                if k.len() > 1024 {
                    return Err(reject("metadata key exceeds bound"));
                }
                *strings += k.len();
                check_metadata_with_capacity(v, depth + 1, nodes, strings, capacity)?;
            }
        }
        _ => {}
    }
    if *strings > 1 << 20 {
        return Err(reject("metadata strings exceed native bound"));
    }
    Ok(())
}

fn reject(message: &str) -> String {
    format!("FuseFSS SDK: {message}")
}
fn numeric(profile: &str) -> Result<&'static str, String> {
    match profile {
        GPT2_NUMERIC => Ok(GPT2_NUMERIC),
        DENSE_ROPE_NUMERIC => Ok(DENSE_ROPE_NUMERIC),
        DENSE_ROPE_CHECKED_NUMERIC => Ok(DENSE_ROPE_CHECKED_NUMERIC),
        _ => Err(reject(
            "explicit D13 complete-decoder numeric_profile required",
        )),
    }
}

pub(crate) fn selected(pipeline: &ExperimentPipeline) -> bool {
    pipeline
        .components
        .values()
        .any(|v| matches!(v.component.as_str(), PROTOCOL | TOPOLOGY | STATE))
}

fn profile(document: &ExperimentDocument) -> Result<&'static str, String> {
    numeric(
        document
            .pipeline
            .components
            .get("linear")
            .and_then(|c| c.params.get("numeric_profile"))
            .and_then(Value::as_str)
            .ok_or_else(|| reject("missing numeric_profile"))?,
    )
}

fn capacity(document: &ExperimentDocument) -> Result<Capacity, String> {
    match document
        .pipeline
        .components
        .get("linear")
        .and_then(|c| c.params.get("capacity"))
    {
        None => Ok(Capacity::Capsule),
        Some(Value::String(s)) if s == "decoder" => Ok(Capacity::Decoder),
        _ => Err(reject(
            "capacity must be omitted for capsule or explicitly decoder",
        )),
    }
}

fn delivery_field(value: Option<&Value>) -> Result<MaterialDelivery, String> {
    match value {
        None => Ok(MaterialDelivery::Single),
        Some(Value::String(s)) if s == "batch-v1" => Ok(MaterialDelivery::BatchV1),
        _ => Err(reject(
            "material_delivery must be omitted for single or explicitly batch-v1",
        )),
    }
}

fn online_field(value: Option<&Value>) -> Result<super::OnlineSchedule, String> {
    match value {
        None => Ok(super::OnlineSchedule::Serial),
        Some(Value::String(s)) if s == "wavefront-v1" => Ok(super::OnlineSchedule::WavefrontV1),
        Some(Value::String(s)) if s == "tensor-v1" => Ok(super::OnlineSchedule::TensorV1),
        _ => Err(reject(
            "online_schedule must be omitted for serial or explicitly wavefront-v1 or tensor-v1",
        )),
    }
}
fn online_schedule(document: &ExperimentDocument) -> Result<super::OnlineSchedule, String> {
    online_field(
        document
            .pipeline
            .components
            .get("linear")
            .and_then(|c| c.params.get("online_schedule")),
    )
}
pub fn contract_online_schedule(contract: &[u8]) -> Result<super::OnlineSchedule, String> {
    if contract.is_empty() || contract.len() > MAX_CONTRACT_BYTES {
        return Err(reject("contract byte bound"));
    }
    let value = bounded_json_value_with_capacity(contract, Capacity::Decoder)?;
    online_field(
        value
            .as_object()
            .ok_or_else(|| reject("contract must be an object"))?
            .get("online_schedule"),
    )
}

fn material_delivery(document: &ExperimentDocument) -> Result<MaterialDelivery, String> {
    delivery_field(
        document
            .pipeline
            .components
            .get("linear")
            .and_then(|c| c.params.get("material_delivery")),
    )
}

/// Read the validated transport selector after checked_contract_digest has
/// authenticated the complete typed source contract. This bounded selector
/// reader alone does NOT certify a source/execution binding or caller pin.
pub fn contract_material_delivery(contract: &[u8]) -> Result<MaterialDelivery, String> {
    if contract.is_empty() || contract.len() > MAX_CONTRACT_BYTES {
        return Err(reject("contract byte bound"));
    }
    let value = bounded_json_value_with_capacity(contract, Capacity::Decoder)?;
    let object = value
        .as_object()
        .ok_or_else(|| reject("contract must be an object"))?;
    delivery_field(object.get("material_delivery"))
}

pub(crate) fn validate(document: &ExperimentDocument) -> Result<(), String> {
    // Generic checks only: never route this composition through client-KV or
    // masked/int8 composition validation. The top-level resolver selects us.
    crate::validate_experiment(document)?;
    crate::validate_experiment_model(&document.pipeline.model)?;
    let chosen = profile(document)?;
    let capacity = capacity(document)?;
    let delivery = material_delivery(document)?;
    let online = online_schedule(document)?;
    let limits = Limits::default();
    sequence(
        usize::try_from(document.budget.max_input_tokens)
            .map_err(|_| reject("request row overflow"))?,
        usize::try_from(document.budget.max_new_tokens)
            .map_err(|_| reject("selection count overflow"))?,
    )?;
    if !matches!(
        document.pipeline.model.kind.as_str(),
        "huggingface" | "safetensors"
    ) || !matches!(document.deployment, ExperimentDeployment::Local { .. })
        || !(1..=MAX_REQUESTS).contains(&document.budget.requests)
        || document.budget.max_input_tokens > limits.max_sequence as u64
        || document.budget.max_new_tokens > limits.max_selections as u64
    {
        return Err(reject("unsupported source, deployment or request envelope"));
    }
    let mut linear = json!({"numeric_profile": chosen});
    if capacity == Capacity::Decoder {
        linear["capacity"] = json!("decoder");
    }
    if delivery == MaterialDelivery::BatchV1 {
        linear["material_delivery"] = json!(delivery.name());
    }
    if online.uses_wavefront() {
        linear["online_schedule"] = json!(online.name());
    }
    let expected = [
        ("linear", PROTOCOL, linear),
        ("topology", TOPOLOGY, json!({})),
        ("state", STATE, json!({})),
        ("kernels", "pllm/cpu", json!({"threads":1})),
    ];
    if document.pipeline.components.len() != expected.len() {
        return Err(reject(
            "exactly linear/topology/state/kernels slots required",
        ));
    }
    for (slot, identity, params) in expected {
        let c = document
            .pipeline
            .components
            .get(slot)
            .ok_or_else(|| reject("incomplete composition"))?;
        if c.component != identity
            || serde_json::to_value(&c.params).map_err(|_| reject("invalid component params"))?
                != params
        {
            return Err(reject(
                "incompatible role, public-weight, state, numeric or CPU choice",
            ));
        }
    }
    Ok(())
}

fn sequence(rows: usize, outputs: usize) -> Result<pllm_models::DecoderPhaseGeometry, String> {
    let limits = Limits::default();
    if rows == 0 || outputs == 0 || outputs > limits.max_selections {
        return Err(reject("request rows/selections exceed native Limits"));
    }
    DecoderPlan::validated_phase_geometry(rows, outputs)
        .filter(|geometry| geometry.padded_rows() <= limits.max_sequence)
        .ok_or_else(|| reject("complete-prefix sequence exceeds native Limits"))
}

/// Bounded metadata returned before a SourceBuilder opens any tensor payload.
/// Shapes are checkpoint storage shapes, not silently transposed kernel shapes.
/// Every byte count below is a payload count, NOT an actual allocator reservation.
pub struct SourcePreflight {
    pub capacity: Capacity,
    pub source_config_sha256: String,
    pub plan: DecoderPlan,
    pub source_shapes: BTreeMap<String, Vec<usize>>,
    pub source_layouts: BTreeMap<String, String>,
    pub source_float32_bytes: usize,
    pub source_q12_value_bytes: usize,
    /// File payload only. Independent of native resident admission.
    pub source_disk_bytes: u64,
    /// Conservative logical native source owner plus one bounded read scratch.
    /// Not process RSS, page-cache residency, or complete execution admission.
    pub source_resident_bound_bytes: usize,
    /// Peak optional producer certificate payload, including column scratch once.
    pub source_certificate_peak_bytes: usize,
    pub rotary_shapes: BTreeMap<String, Vec<usize>>,
    pub rotary_float32_bytes: usize,
    pub source_minimum_owner_bytes: usize,
    pub vocabulary: usize,
    pub width: usize,
    pub maximum_sequence: usize,
    pub numeric_profile: &'static str,
    pub phase_admission: SourcePhaseAdmission,
}

/// Conservative logical owner ledger, not a global allocator/RSS reservation.
/// BTree storage prices one maximal sparse node (11 key/value slots, 12 edges)
/// per entry at 1024 B; strings and vector backing capacities are additional.
#[derive(Clone, Copy, Debug, Default)]
pub struct SourcePhaseAdmission {
    pub plan_owned_bound: usize,
    pub inventory_owned_bound: usize,
    pub serialization_peak_bound: usize,
    pub ingestion_peak_bound: usize,
    pub compile_external_bytes: usize,
    pub compile_prepare_peak_bound: usize,
}
fn add_bytes(a: usize, b: usize) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| reject("phase ownership overflow"))
}
fn json_owned(value: &Value) -> Result<usize, String> {
    match value {
        Value::String(s) => Ok(s.capacity()),
        Value::Array(a) => a
            .iter()
            .try_fold(a.capacity() * std::mem::size_of::<Value>(), |n, v| {
                add_bytes(n, json_owned(v)?)
            }),
        Value::Object(m) => m.iter().try_fold(m.len() * 1024, |n, (k, v)| {
            add_bytes(add_bytes(n, k.capacity())?, json_owned(v)?)
        }),
        _ => Ok(0),
    }
}
fn plan_owned(plan: &DecoderPlan) -> Result<usize, String> {
    let mut bytes = std::mem::size_of::<DecoderPlan>()
        + plan.schema_version.capacity()
        + plan.model_family.capacity()
        + plan.adapter.capacity()
        + 128;
    for g in [&plan.prefill, &plan.decode] {
        bytes = add_bytes(
            bytes,
            g.output.capacity() + g.operations.capacity() * std::mem::size_of::<ModelOperation>(),
        )?;
        for states in [&g.state_inputs, &g.state_outputs] {
            bytes = add_bytes(
                bytes,
                states.capacity() * std::mem::size_of::<pllm_models::StateTensor>(),
            )?;
            for state in states {
                bytes = add_bytes(bytes, state.id.capacity() + state.shape.capacity() * 8)?;
            }
        }
        for op in &g.operations {
            bytes = add_bytes(
                bytes,
                op.id.capacity()
                    + op.inputs.capacity() * std::mem::size_of::<String>()
                    + op.output_shape.capacity() * 8,
            )?;
            for input in &op.inputs {
                bytes = add_bytes(bytes, input.capacity())?;
            }
            bytes = add_bytes(bytes, json_owned(&op.attributes)?)?;
        }
    }
    Ok(bytes)
}
fn phase_admission(a: &SourcePreflight) -> Result<SourcePhaseAdmission, String> {
    let plan = plan_owned(&a.plan)?;
    let mut inventory = a.source_config_sha256.capacity()
        + a.source_shapes.len() * 1024
        + a.source_layouts.len() * 1024
        + a.rotary_shapes.len() * 1024;
    for (k, v) in a.source_shapes.iter().chain(a.rotary_shapes.iter()) {
        inventory = add_bytes(
            inventory,
            k.capacity() + v.capacity() * std::mem::size_of::<usize>(),
        )?;
    }
    for (k, v) in &a.source_layouts {
        inventory = add_bytes(inventory, k.capacity() + v.capacity())?;
    }
    // At most two graph/inventory trees: borrowed original plus canonical Value.
    // Output is <=1 MiB. Sorting/index scratch and both parsed configs are priced.
    let scratch = MAX_NODES * 512 + MAX_CONFIG_BYTES * 3 + MAX_CONTRACT_BYTES;
    let serialization = add_bytes(add_bytes(plan * 2, inventory * 2)?, scratch)?;
    // Final source owners move without copying. The 8-MiB resident allowance
    // excludes separately checked immutable files. Rotary staging is <=8*n;
    // borrowed pair frames add 128 KiB. The additional MiB covers two bounded
    // Q12/file write buffers, source hashing scratch, names and pending owners.
    let ingestion = add_bytes(
        add_bytes(plan, inventory)?,
        super::MAX_BYTES
            + MAX_VALUES * 8
            + 2 * super::import::MAX_FRAME_BYTES
            + (1 << 20)
            + scratch
            + a.source_certificate_peak_bytes,
    )?;
    let external = add_bytes(
        plan,
        4096 + 2 * std::mem::size_of::<SourceExecutionBinding>(),
    )?;
    let native_nodes =
        a.plan.prefill.operations.len() * (std::mem::size_of::<super::program::Node>() + 4 * 8);
    let proof_bytes =
        if a.capacity == Capacity::Decoder || a.numeric_profile == DENSE_ROPE_CHECKED_NUMERIC {
            super::energy::reservation_bound(a.plan.prefill.operations.len())
                .map_err(|_| reject("row proof owner overflow"))?
        } else {
            0
        };
    let mut preparation = add_bytes(
        plan * 3,
        super::MAX_BYTES + native_nodes + proof_bytes + scratch + external - plan,
    )?;
    if proof_bytes != 0 {
        // Public ordered admission builds bounded native Plans/cache; weighted
        // certificate scans share the same remaining construction allowance.
        // Program limits that combined allowance to at most one third of 64 MiB.
        preparation = add_bytes(
            preparation,
            pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES / 3,
        )?;
    }
    let cap = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES;
    if serialization > cap || ingestion > cap || preparation > cap {
        return Err(reject(
            "source phase ownership exceeds native 64 MiB envelope",
        ));
    }
    Ok(SourcePhaseAdmission {
        plan_owned_bound: plan,
        inventory_owned_bound: inventory,
        serialization_peak_bound: serialization,
        ingestion_peak_bound: ingestion,
        compile_external_bytes: external,
        compile_prepare_peak_bound: preparation,
    })
}

/// Lower bound of the existing Program metadata credit, before its node vectors
/// exist. Keep aligned with program.rs::Program::compile: source + fixed header +
/// 4096/op + 128/recursive attr node + twice retained attr/key string bytes.
/// Native node/shape backing is intentionally omitted (it only increases credit).
/// Import tests compare this floor with actual compiled resources.
pub(super) fn compile_credit_floor(
    plan: &DecoderPlan,
    source_bytes: usize,
) -> Result<usize, String> {
    let mut nodes = 0;
    let mut strings = 0;
    for g in [&plan.prefill, &plan.decode] {
        for op in &g.operations {
            check_metadata(&op.attributes, 0, &mut nodes, &mut strings)?;
        }
    }
    add_bytes(
        source_bytes + std::mem::size_of::<super::Program>(),
        nodes * 128
            + strings * 2
            + (plan.prefill.operations.len() + plan.decode.operations.len()) * 4096,
    )
}

fn count(shape: &[u64]) -> Result<usize, String> {
    count_limit(shape, MAX_VALUES)
}
fn count_limit(shape: &[u64], limit: usize) -> Result<usize, String> {
    if shape.is_empty() || shape.len() > 4 || shape.contains(&0) {
        return Err(reject("invalid source/graph tensor shape"));
    }
    shape.iter().try_fold(1usize, |p, &d| {
        p.checked_mul(usize::try_from(d).map_err(|_| reject("shape overflow"))?)
            .filter(|&v| v <= limit)
            .ok_or_else(|| reject("tensor exceeds native MAX_VALUES"))
    })
}

fn config_attributes(document: &Value, rotary: bool) -> Result<(), String> {
    let object = document
        .as_object()
        .ok_or_else(|| reject("source config must be an object"))?;
    // Learned-position sources have their own strict parser. The dense readers
    // historically tolerate extra keys: admit only inert metadata or fields whose
    // effects are represented in this plan, without branching on family identity.
    if rotary {
        let allowed = [
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "hidden_act",
            "rms_norm_eps",
            "rope_theta",
            "tie_word_embeddings",
            "sliding_window",
            "use_sliding_window",
            "head_dim",
            "attention_bias",
            "attention_dropout",
            "rope_scaling",
            "use_cache",
            "max_window_layers",
            "layer_types",
            "mlp_bias",
            "pretraining_tp",
            "architectures",
            "torch_dtype",
            "dtype",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "initializer_range",
            "transformers_version",
            "_name_or_path",
        ];
        if object.keys().any(|k| !allowed.contains(&k.as_str())) {
            return Err(reject("unapproved dense source attribute"));
        }
        for key in ["rope_scaling"] {
            if object.get(key).is_some_and(|v| !v.is_null()) {
                return Err(reject("scaled/partial rotary is not approved"));
            }
        }
        if object.get("use_cache").is_some_and(|v| v != true)
            || object.get("use_sliding_window").is_some_and(|v| v != false)
            || object.get("mlp_bias").is_some_and(|v| v != false)
            || object
                .get("pretraining_tp")
                .is_some_and(|v| v.as_u64() != Some(1))
            || object
                .get("attention_dropout")
                .is_some_and(|v| v.as_f64() != Some(0.0))
        {
            return Err(reject("unapproved dense source execution attribute"));
        }
    }
    for key in ["torch_dtype", "dtype"] {
        if object.get(key).is_some_and(|v| {
            !v.is_null() && !matches!(v.as_str(), Some("float32" | "float16" | "bfloat16"))
        }) {
            return Err(reject("unsupported external source dtype"));
        }
    }
    Ok(())
}

fn allowed_attributes(op: &ModelOperation) -> Result<(), String> {
    let keys: &[&str] = match op.operator {
        Op::TokenLookup => &[
            "weight",
            "weight_shape",
            "compute_dtype",
            "semantic_config",
            "index_domain",
            "position_policy",
            "out_of_bounds",
        ],
        Op::Linear => &[
            "weight",
            "bias",
            "weight_shape",
            "bias_shape",
            "weight_layout",
            "formula",
            "compute_dtype",
        ],
        Op::OutputHead => &[
            "weight",
            "bias",
            "tied",
            "weight_shape",
            "bias_shape",
            "weight_layout",
            "compute_dtype",
            "output_dtype",
            "input_layout",
            "output_layout",
        ],
        Op::LayerNorm => &[
            "epsilon",
            "weight",
            "bias",
            "weight_shape",
            "bias_shape",
            "axis",
            "variance",
            "epsilon_placement",
            "affine_order",
            "compute_dtype",
        ],
        Op::RmsNorm => &["epsilon", "weight", "weight_offset"],
        Op::Reshape => &["layout", "input_layout"],
        Op::Slice => &["axis", "start", "end", "squeeze"],
        Op::RotaryEmbedding => &[
            "theta",
            "rotary_dimensions",
            "pairing",
            "position_policy",
            "coefficient_profile",
            "input_layout",
            "output_layout",
            "tail_policy",
            "frequency_scaling",
        ],
        Op::KvCacheAppend => &[
            "state",
            "mode",
            "state_capacity",
            "state_layout",
            "absolute_write_positions_input",
            "padding_mask_input",
            "valid_lengths_input",
            "attention_domain",
        ],
        Op::CacheSuffix => &[
            "axis",
            "maximum_sequence",
            "semantics",
            "absolute_write_positions_input",
            "padding_mask_input",
            "valid_lengths_input",
        ],
        Op::AttentionScores | Op::AttentionValues => &["group_size"],
        Op::AttentionScale => &["head_dim"],
        Op::CausalMask => &[
            "padding_mask_input",
            "valid_lengths_input",
            "cache_validity",
            "masked_value",
        ],
        Op::Softmax => &["axis", "compute_dtype"],
        Op::ResidualAdd => &["dropout"],
        Op::GeluTanh => &["formula", "compute_dtype"],
        Op::Silu | Op::Multiply => &[],
        Op::LastToken => &["axis", "selection", "valid_lengths_input"],
        Op::GreedyTokenSelection => &["policy", "source", "tie_break"],
        Op::TokenFeedback => &["policy", "source"],
        _ => return Err(reject("unsupported semantic operator")),
    };
    let attrs = op
        .attributes
        .as_object()
        .ok_or_else(|| reject("nonobject operator attributes"))?;
    if attrs.keys().any(|k| !keys.contains(&k.as_str())) {
        return Err(reject("unapproved semantic operator attribute"));
    }
    for key in ["compute_dtype", "output_dtype"] {
        if attrs
            .get(key)
            .is_some_and(|v| !matches!(v.as_str(), Some("float32" | "model_native")))
        {
            return Err(reject("unsupported semantic operator dtype"));
        }
    }
    if op.operator == Op::RotaryEmbedding {
        let width = *op
            .output_shape
            .last()
            .ok_or_else(|| reject("missing rotary width"))?;
        if width % 2 != 0
            || width > 4096
            || op.attributes["rotary_dimensions"] != width
            || op.attributes["pairing"] != "split_half"
            || op.attributes["position_policy"] != "sequential_absolute"
            || op.attributes["input_layout"] != "batch_heads_sequence_feature"
            || op.attributes["output_layout"] != "batch_heads_sequence_feature"
            || op.attributes["tail_policy"] != "unchanged"
            || op.attributes["coefficient_profile"] != "pllm.numeric.rope.q30.libm.v1"
            || !op.attributes["frequency_scaling"].is_null()
        {
            return Err(reject("unapproved rotary semantics"));
        }
    }
    Ok(())
}

fn declared(op: &ModelOperation, key: &str, expected: &[u64]) -> Result<(), String> {
    if op
        .attributes
        .get(key)
        .is_some_and(|v| v != &json!(expected))
    {
        return Err(reject(
            "declared checkpoint shape differs from semantic operand",
        ));
    }
    Ok(())
}

fn bind(
    capacity: Capacity,
    shapes: &mut BTreeMap<String, Vec<usize>>,
    layouts: &mut BTreeMap<String, String>,
    name: &str,
    shape: &[u64],
    layout: &str,
) -> Result<(), String> {
    count_limit(shape, capacity.max_tensor_values())?;
    if name.is_empty() || name.len() > 1024 || name.bytes().any(|v| v == 0) {
        return Err(reject("invalid source tensor identifier"));
    }
    let dims = shape
        .iter()
        .map(|&n| usize::try_from(n).map_err(|_| reject("source dimension overflow")))
        .collect::<Result<Vec<_>, _>>()?;
    if shapes.get(name).is_some_and(|v| v != &dims)
        || layouts.get(name).is_some_and(|v| v != layout)
    {
        return Err(reject(
            "inconsistent shared/tied source tensor shape or orientation",
        ));
    }
    shapes.insert(name.into(), dims);
    layouts.insert(name.into(), layout.into());
    if shapes.len() > MAX_NODES {
        return Err(reject("too many source tensors"));
    }
    Ok(())
}

fn artifact<'a>(op: &'a ModelOperation, key: &str) -> Result<&'a str, String> {
    op.attributes[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| reject("missing source tensor artifact"))
}

// Internal inspection only: never accept a caller-supplied plan instead of
// lowering the byte-bounded source config through preflight_source.
#[cfg(test)]
fn preflight_plan(
    plan: DecoderPlan,
    outputs: usize,
    numeric_profile: &str,
) -> Result<SourcePreflight, String> {
    preflight_plan_with_capacity(plan, outputs, numeric_profile, Capacity::Capsule)
}

fn preflight_plan_with_capacity(
    plan: DecoderPlan,
    outputs: usize,
    numeric_profile: &str,
    capacity: Capacity,
) -> Result<SourcePreflight, String> {
    let chosen = numeric(numeric_profile)?;
    let rows = usize::try_from(plan.prefill.query_sequence).map_err(|_| reject("row overflow"))?;
    let geometry = sequence(rows, outputs)?;
    let maximum_sequence = geometry.padded_rows();
    if !plan.transformations.is_empty()
        || plan.prefill.batch != 1
        || plan.decode.batch != 1
        || plan.decode.maximum_key_sequence != maximum_sequence as u64
    {
        return Err(reject("full-prefix batch/sequence/state contract differs"));
    }
    // Bound every view before invoking the more detailed semantic validator.
    for g in [&plan.prefill, &plan.decode] {
        if g.operations.len() > MAX_NODES
            || g.state_inputs.len() > MAX_NODES
            || g.state_outputs.len() > MAX_NODES
            || g.maximum_key_sequence > Limits::default().max_sequence as u64
        {
            return Err(reject("graph exceeds native descriptor limits"));
        }
        for s in g.state_inputs.iter().chain(&g.state_outputs) {
            count_limit(&s.shape, capacity.max_graph_values())?;
        }
        for op in &g.operations {
            count_limit(&op.output_shape, capacity.max_graph_values())?;
            allowed_attributes(op)?;
            // Every selection executes this full padded extent. Active causal
            // prefixes do not discount shape/work pricing to a triangular sum.
            if g.mode == pllm_models::DecoderMode::Prefill {
                let mut expanded = op.output_shape.clone();
                if expanded.len() == 3 {
                    expanded[1] = maximum_sequence as u64;
                }
                if expanded.len() == 4 {
                    expanded[2] = maximum_sequence as u64;
                    if matches!(
                        op.operator,
                        Op::AttentionScores | Op::AttentionScale | Op::CausalMask | Op::Softmax
                    ) {
                        expanded[3] = maximum_sequence as u64;
                    }
                }
                count_limit(&expanded, capacity.max_graph_values())?;
            }
        }
    }
    plan.validate().map_err(|e| reject(&e.to_string()))?;
    let ops = &plan.prefill.operations;
    let rotary = ops.iter().any(|v| v.operator == Op::RotaryEmbedding);
    let positions = ops
        .iter()
        .any(|v| v.operator == Op::TokenLookup && v.inputs == ["input.positions"]);
    let layer_norm = ops.iter().any(|v| v.operator == Op::LayerNorm);
    if (chosen == GPT2_NUMERIC
        && (!positions
            || !layer_norm
            || rotary
            || ops
                .iter()
                .any(|v| matches!(v.operator, Op::Silu | Op::RmsNorm | Op::Multiply))))
        || (matches!(chosen, DENSE_ROPE_NUMERIC | DENSE_ROPE_CHECKED_NUMERIC)
            && (!rotary
                || positions
                || layer_norm
                || ops.iter().any(|v| v.operator == Op::GeluTanh)))
    {
        return Err(reject(
            "numeric_profile does not match semantic capabilities",
        ));
    }
    // Existing matrix inventory covers dense output-input linears and ties. It
    // intentionally does not support learned-position lookups or Conv1D storage.
    if rotary {
        crate::lower_decoder_matrix_inventory(&plan)?;
    }
    let head = ops
        .iter()
        .find(|op| op.operator == Op::OutputHead)
        .ok_or_else(|| reject("missing complete head"))?;
    if head.output_shape.len() != 2 {
        return Err(reject("invalid vocabulary head"));
    }
    let vocabulary =
        usize::try_from(head.output_shape[1]).map_err(|_| reject("vocabulary overflow"))?;
    let token = ops
        .iter()
        .find(|v| v.operator == Op::TokenLookup && v.inputs == ["input.tokens"])
        .ok_or_else(|| reject("missing private token lookup"))?;
    let width = *token
        .output_shape
        .last()
        .ok_or_else(|| reject("missing embedding width"))? as usize;
    let mut source_shapes = BTreeMap::new();
    let mut source_layouts = BTreeMap::new();
    for graph in [&plan.prefill, &plan.decode] {
        let mut shapes: BTreeMap<&str, &[u64]> = BTreeMap::new();
        for op in &graph.operations {
            let input = op
                .inputs
                .first()
                .and_then(|s| shapes.get(s.as_str()))
                .copied();
            match op.operator {
                Op::TokenLookup => {
                    let table_rows = if op.inputs == ["input.positions"] {
                        op.attributes["weight_shape"]
                            .get(0)
                            .and_then(Value::as_u64)
                            .filter(|&n| n >= maximum_sequence as u64)
                            .ok_or_else(|| reject("incomplete learned position table"))?
                    } else if op.inputs == ["input.tokens"] {
                        vocabulary as u64
                    } else {
                        return Err(reject("unsupported lookup index domain"));
                    };
                    let shape = [
                        table_rows,
                        *op.output_shape
                            .last()
                            .ok_or_else(|| reject("lookup width"))?,
                    ];
                    declared(op, "weight_shape", &shape)?;
                    bind(
                        capacity,
                        &mut source_shapes,
                        &mut source_layouts,
                        artifact(op, "weight")?,
                        &shape,
                        "rows_features",
                    )?;
                }
                Op::Linear | Op::OutputHead => {
                    let inner = *input
                        .and_then(|s| s.last())
                        .ok_or_else(|| reject("missing linear input width"))?;
                    let columns = *op
                        .output_shape
                        .last()
                        .ok_or_else(|| reject("missing linear output width"))?;
                    let layout = op.attributes["weight_layout"]
                        .as_str()
                        .unwrap_or("output_input");
                    let shape = match layout {
                        "input_output" => [inner, columns],
                        "output_input" => [columns, inner],
                        _ => return Err(reject("unsupported source matrix orientation")),
                    };
                    declared(op, "weight_shape", &shape)?;
                    // Tied vocabulary weights are the very same rows/features
                    // bytes, not a separately transposed checkpoint tensor.
                    let name = artifact(op, "weight")?;
                    let storage =
                        if op.operator == Op::OutputHead && name == artifact(token, "weight")? {
                            "rows_features"
                        } else {
                            layout
                        };
                    bind(
                        capacity,
                        &mut source_shapes,
                        &mut source_layouts,
                        name,
                        &shape,
                        storage,
                    )?;
                    if !op.attributes["bias"].is_null() {
                        declared(op, "bias_shape", &[columns])?;
                        bind(
                            capacity,
                            &mut source_shapes,
                            &mut source_layouts,
                            artifact(op, "bias")?,
                            &[columns],
                            "features",
                        )?;
                    }
                }
                Op::LayerNorm | Op::RmsNorm => {
                    let width = *op
                        .output_shape
                        .last()
                        .ok_or_else(|| reject("missing norm width"))?;
                    if width > 4096 || op.attributes.get("weight_offset").is_some_and(|v| v != 0) {
                        return Err(reject("unsupported norm width/offset"));
                    }
                    declared(op, "weight_shape", &[width])?;
                    bind(
                        capacity,
                        &mut source_shapes,
                        &mut source_layouts,
                        artifact(op, "weight")?,
                        &[width],
                        "features",
                    )?;
                    if !op.attributes["bias"].is_null() {
                        declared(op, "bias_shape", &[width])?;
                        bind(
                            capacity,
                            &mut source_shapes,
                            &mut source_layouts,
                            artifact(op, "bias")?,
                            &[width],
                            "features",
                        )?;
                    }
                }
                _ => {}
            }
            shapes.insert(&op.id, &op.output_shape);
        }
    }
    let values = source_shapes.values().try_fold(0usize, |sum, shape| {
        let n = shape
            .iter()
            .try_fold(1usize, |p, &n| p.checked_mul(n))
            .ok_or_else(|| reject("source count overflow"))?;
        sum.checked_add(n)
            .ok_or_else(|| reject("total source values overflow"))
    })?;
    let source_q12_value_bytes = values
        .checked_mul(8)
        .filter(|&n| n as u64 <= capacity.max_source_bytes())
        .ok_or_else(|| reject("source Q12 payload exceeds native MAX_BYTES"))?;
    let source_float32_bytes = values
        .checked_mul(4)
        .ok_or_else(|| reject("source FP32 payload overflow"))?;
    let mut rotary_shapes = BTreeMap::new();
    let mut rotary_float32_bytes = 0usize;
    let mut source_disk_bytes = 0u64;
    let mut source_certificate_peak_bytes = 0usize;
    let mut source_minimum_owner_bytes = std::mem::size_of::<super::Source>()
        + source_shapes.len() * std::mem::size_of::<super::Tensor>();
    for (name, shape) in &source_shapes {
        let bytes = shape.iter().product::<usize>() * 8;
        let resident = if capacity.file_backed(shape) {
            let layout = super::source::certified_file_layout(shape, capacity)
                .map_err(|_| reject("source certificate geometry exceeds bounds"))?;
            let resources = layout.resources();
            source_certificate_peak_bytes =
                source_certificate_peak_bytes.max(resources.certificate_payload_bytes);
            source_disk_bytes = source_disk_bytes
                .checked_add(resources.disk_bytes)
                .and_then(|n| n.checked_add(resources.row_statistics_disk_bytes))
                .ok_or_else(|| reject("source disk accounting overflow"))?;
            super::capacity::FILE_OWNER_BYTES
        } else {
            bytes
        };
        source_minimum_owner_bytes = source_minimum_owner_bytes
            .checked_add(name.len() + shape.len() * std::mem::size_of::<usize>() + resident)
            .ok_or_else(|| reject("source owner size overflow"))?;
    }
    if source_disk_bytes != 0 {
        source_minimum_owner_bytes = add_bytes(
            source_minimum_owner_bytes,
            super::capacity::SOURCE_READ_BYTES,
        )?;
    }
    if source_disk_bytes > capacity.max_source_bytes() {
        return Err(reject(
            "source files and statistics exceed aggregate disk allowance",
        ));
    }
    for op in ops.iter().filter(|op| op.operator == Op::RotaryEmbedding) {
        let width = *op
            .output_shape
            .last()
            .ok_or_else(|| reject("missing rotary width"))?;
        let n = count(&[maximum_sequence as u64, width])?;
        rotary_float32_bytes = rotary_float32_bytes
            .checked_add(n * 2 * 4)
            .ok_or_else(|| reject("rotary payload overflow"))?;
        source_minimum_owner_bytes = source_minimum_owner_bytes
            .checked_add(std::mem::size_of::<super::RotaryCoefficients>() + op.id.len() + n * 2 * 8)
            .ok_or_else(|| reject("rotary owner overflow"))?;
        rotary_shapes.insert(op.id.clone(), vec![maximum_sequence, width as usize]);
    }
    if source_minimum_owner_bytes > capacity.max_resident_bytes() {
        return Err(reject("source owner minimum exceeds native MAX_BYTES"));
    }
    let source_resident_bound_bytes = source_minimum_owner_bytes;
    if capacity == Capacity::Capsule {
        source_minimum_owner_bytes -=
            std::mem::size_of::<super::Source>() - super::source::legacy_source_header_bytes();
        source_minimum_owner_bytes -= source_shapes.len()
            * (std::mem::size_of::<super::Tensor>() - super::source::legacy_tensor_header_bytes());
    } else {
        let files = source_shapes
            .values()
            .filter(|s| capacity.file_backed(s))
            .count();
        source_minimum_owner_bytes -= files * super::capacity::FILE_OWNER_BYTES
            + usize::from(files != 0) * super::capacity::SOURCE_READ_BYTES;
    }
    let mut admitted = SourcePreflight {
        capacity,
        source_config_sha256: String::new(),
        plan,
        source_shapes,
        source_layouts,
        source_float32_bytes,
        source_q12_value_bytes,
        source_disk_bytes,
        source_resident_bound_bytes,
        source_certificate_peak_bytes,
        rotary_shapes,
        rotary_float32_bytes,
        source_minimum_owner_bytes,
        vocabulary,
        width,
        maximum_sequence,
        numeric_profile: chosen,
        phase_admission: SourcePhaseAdmission::default(),
    };
    admitted.phase_admission = phase_admission(&admitted)?;
    Ok(admitted)
}

/// Public pre-read gate for SourceBuilder. No tensor file is opened or allocated.
pub fn preflight_source(
    config: &[u8],
    rows: usize,
    outputs: usize,
    numeric_profile: &str,
) -> Result<SourcePreflight, String> {
    preflight_source_with_capacity(config, rows, outputs, numeric_profile, Capacity::Capsule)
}

pub fn preflight_source_with_capacity(
    config: &[u8],
    rows: usize,
    outputs: usize,
    numeric_profile: &str,
    capacity: Capacity,
) -> Result<SourcePreflight, String> {
    numeric(numeric_profile)?;
    sequence(rows, outputs)?;
    if config.is_empty() || config.len() > MAX_CONFIG_BYTES {
        return Err(reject("source config byte bound"));
    }
    let document = bounded_json_value(config)?;
    // Approval-policy filtering precedes expansion as well. Final compatibility
    // still follows actual semantic operators, never a model-family branch.
    let chosen = match numeric_profile {
        GPT2_NUMERIC => GPT2_NUMERIC,
        DENSE_ROPE_NUMERIC => DENSE_ROPE_NUMERIC,
        DENSE_ROPE_CHECKED_NUMERIC => DENSE_ROPE_CHECKED_NUMERIC,
        _ => return Err(reject("unapproved numeric profile")),
    };
    if document.get("rope_scaling").is_some_and(|v| !v.is_null()) {
        return Err(reject("scaled/partial rotary is not approved"));
    }
    config_attributes(&document, chosen != GPT2_NUMERIC)?;
    let workload = DecoderWorkload {
        batch: 1,
        max_input_tokens: rows as u64,
        max_new_tokens: outputs as u64,
    };
    let budget = DecoderExpansionBudget {
        max_layers: if capacity == Capacity::Decoder {
            32
        } else {
            MAX_NODES as u64
        },
        max_feature_width: capacity.max_feature_width() as u64,
        max_attention_heads: if capacity == Capacity::Decoder {
            128
        } else {
            MAX_VALUES as u64
        },
        max_vocabulary: capacity.max_vocabulary() as u64,
        max_tensor_elements: capacity.max_graph_values() as u64,
    };
    let expansion = budget
        .preflight_metadata(&document, workload, MAX_NODES, 1024, 32768, 1 << 20)
        .map_err(|e| reject(&e.to_string()))?;
    // This additional allowance precedes the existing builder and its validation
    // copies. It does not change any old lowering caller or identity.
    if expansion
        .retained_metadata_bytes
        .checked_mul(3)
        .is_none_or(|n| n > 64 << 20)
    {
        return Err(reject(
            "pre-expansion metadata phase exceeds native envelope",
        ));
    }
    let plan = pllm_models::lower_model_json_with_budget(config, workload, budget)
        .map_err(|e| reject(&e.to_string()))?;
    let mut nodes = 0;
    let mut strings = 0;
    for graph in [&plan.prefill, &plan.decode] {
        if graph.operations.len() > expansion.operations_per_phase {
            return Err(reject("mapped operation expansion contract changed"));
        }
        for op in &graph.operations {
            check_metadata(&op.attributes, 0, &mut nodes, &mut strings)?;
        }
    }
    // A known incompatible canonical plan must reject before accepting values.
    encoded_length_with_capacity(&plan, MAX_CONTRACT_BYTES, capacity)?;
    config_attributes(
        &document,
        plan.prefill
            .operations
            .iter()
            .any(|v| v.operator == Op::RotaryEmbedding),
    )?;
    if plan
        .prefill
        .operations
        .iter()
        .any(|v| v.operator == Op::RotaryEmbedding)
    {
        if let Some(bias) = document.get("attention_bias") {
            let qkv: Vec<_> = plan
                .prefill
                .operations
                .iter()
                .filter(|op| {
                    op.operator == Op::Linear
                        && plan.prefill.operations.iter().any(|head| {
                            head.operator == Op::Reshape
                                && head.output_shape.len() == 4
                                && head.inputs.first() == Some(&op.id)
                        })
                })
                .collect();
            if !bias.is_boolean()
                || qkv.is_empty()
                || qkv
                    .iter()
                    .any(|op| op.attributes["bias"].is_null() == bias.as_bool().unwrap_or(false))
            {
                return Err(reject(
                    "declared attention bias differs from source semantic projections",
                ));
            }
        }
        if let Some(dim) = document.get("head_dim") {
            if plan
                .prefill
                .operations
                .iter()
                .filter(|op| op.operator == Op::RotaryEmbedding)
                .any(|op| op.output_shape.last().copied() != dim.as_u64())
            {
                return Err(reject(
                    "declared head width differs from semantic rotary width",
                ));
            }
        }
        if let Some(layers) = document.get("layer_types") {
            if layers.as_array().is_none_or(|v| {
                v.len() as u64 != document["num_hidden_layers"].as_u64().unwrap_or(0)
                    || v.iter().any(|v| v != "full_attention")
            }) {
                return Err(reject("unapproved source layer schedule"));
            }
        }
    }
    let mut admitted = preflight_plan_with_capacity(plan, outputs, numeric_profile, capacity)?;
    admitted.source_config_sha256 = format!("{:x}", Sha256::digest(config));
    admitted.phase_admission = phase_admission(&admitted)?;
    // Count the complete SDK document, including inventories/policy, before any
    // values or serialized output. Fixed-width identity slots here are sizing
    // placeholders only: never emitted, authenticated or accepted as pins.
    let sizing = ContractView {
        admitted: &admitted,
        material_delivery: MaterialDelivery::Single,
        online_schedule: super::OnlineSchedule::Serial,
        rows,
        outputs,
        configuration: "0".repeat(64),
        composition: "0".repeat(64),
        sourcepin: "0000000000000000000000000000000000000000000000000000000000000000",
        tokenizerpin: "0000000000000000000000000000000000000000000000000000000000000000",
    };
    encoded_length_with_capacity(&sizing, MAX_CONTRACT_BYTES, capacity)?;
    Ok(admitted)
}

fn pin(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        && value.bytes().any(|b| b != b'0')
}

fn admission_record(minimum: usize) -> Value {
    json!({"geometry":"bounded","numeric":"not_performed","source_pin":"caller_verified_required",
        "tokenizer_pin":"caller_verified_required","source_minimum_owner_bytes":minimum,
        "source_owner_reservation_bytes":null,"rotary_f32_payloads":null,"range_obligations":null,
        "domain_admission":null,"program_resources":null,"transport_resources":null,
        "complete_live_accounting":null,"serving_qualified":false})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionFields {
    #[serde(default)]
    capacity: Capacity,
    source_config_sha256: String,
    plan: DecoderPlan,
    source_shapes: BTreeMap<String, Vec<usize>>,
    source_layouts: BTreeMap<String, String>,
    rotary_shapes: BTreeMap<String, Vec<usize>>,
    source_float32_bytes: usize,
    source_q12_value_bytes: usize,
    rotary_float32_bytes: usize,
    numeric_profile: String,
    input_rows: usize,
    output_tokens: usize,
    vocabulary: usize,
    width: usize,
    maximum_sequence: usize,
}
const EXECUTION_KEYS: &[&str] = &[
    "source_config_sha256",
    "plan",
    "source_shapes",
    "source_layouts",
    "rotary_shapes",
    "source_float32_bytes",
    "source_q12_value_bytes",
    "rotary_float32_bytes",
    "numeric_profile",
    "input_rows",
    "output_tokens",
    "vocabulary",
    "width",
    "maximum_sequence",
];

struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[allow(clippy::too_many_arguments)]
fn execution_identity(
    plan: &DecoderPlan,
    config_sha256: &str,
    shapes: &BTreeMap<String, Vec<usize>>,
    layouts: &BTreeMap<String, String>,
    rotary: &BTreeMap<String, Vec<usize>>,
    float_bytes: usize,
    code_bytes: usize,
    rotary_bytes: usize,
    profile: &str,
    rows: usize,
    outputs: usize,
    vocabulary: usize,
    width: usize,
    sequence: usize,
    capacity: Capacity,
) -> Result<[u8; 32], String> {
    let mut writer = HashWriter(Sha256::new());
    writer.0.update(b"pllm.fss.source-execution-binding.v1\0");
    if capacity == Capacity::Decoder {
        writer.0.update(b"explicit-source-capacity:decoder\0");
    }
    serde_json::to_writer(
        &mut writer,
        &(
            plan,
            config_sha256,
            shapes,
            layouts,
            rotary,
            float_bytes,
            code_bytes,
            rotary_bytes,
            profile,
            rows,
            outputs,
            vocabulary,
            width,
            sequence,
        ),
    )
    .map_err(|_| reject("execution binding serialization"))?;
    Ok(writer.0.finalize().into())
}

/// Immutable importer-owned commitment to the exact admitted semantic execution.
/// File/tokenizer and external Experiment identities remain authenticated caller
/// claims: an IEEE tensor stream does not prove whole-file/publisher identities.
pub struct SourceExecutionBinding {
    capacity: Capacity,
    source_disk_bytes: u64,
    source_resident_bound_bytes: usize,
    identity: [u8; 32],
    rows: usize,
    outputs: usize,
    minimum_source_bytes: usize,
}

struct ContractView<'a> {
    admitted: &'a SourcePreflight,
    material_delivery: MaterialDelivery,
    online_schedule: super::OnlineSchedule,
    configuration: String,
    composition: String,
    sourcepin: &'a str,
    tokenizerpin: &'a str,
    rows: usize,
    outputs: usize,
}
impl Serialize for ContractView<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        let a = self.admitted;
        macro_rules! entry {
            ($key:expr, $value:expr) => {
                map.serialize_entry($key, &$value)?;
            };
        }
        entry!("schema_version", "pllm.fss.sdk-contract.v1");
        entry!("configuration", self.configuration);
        entry!("composition", self.composition);
        entry!("source_lock", self.sourcepin);
        entry!("source_config_sha256", a.source_config_sha256);
        entry!("tokenizer", self.tokenizerpin);
        entry!("weights", "public");
        entry!("numeric_profile", a.numeric_profile);
        if self.material_delivery == MaterialDelivery::BatchV1 {
            entry!("material_delivery", self.material_delivery.name());
        }
        if self.online_schedule.uses_wavefront() {
            entry!("online_schedule", self.online_schedule.name());
        }
        if a.capacity == Capacity::Decoder {
            entry!("capacity", a.capacity);
            entry!("source_disk_bytes", a.source_disk_bytes);
            entry!("source_resident_bound_bytes", a.source_resident_bound_bytes);
        }
        entry!("plan", a.plan);
        entry!("source_shapes", a.source_shapes);
        entry!("source_layouts", a.source_layouts);
        entry!("source_float32_bytes", a.source_float32_bytes);
        entry!("source_q12_value_bytes", a.source_q12_value_bytes);
        entry!("rotary_shapes", a.rotary_shapes);
        entry!("rotary_float32_bytes", a.rotary_float32_bytes);
        entry!("vocabulary", a.vocabulary);
        entry!("width", a.width);
        entry!("input_rows", self.rows);
        entry!("output_tokens", self.outputs);
        entry!("decode_steps", self.outputs - 1);
        let prefixes = [self.rows, self.rows + 1, self.rows + 2, self.rows + 3];
        entry!("selection_prefix_rows", &prefixes[..self.outputs]);
        entry!("maximum_sequence", a.maximum_sequence);
        entry!("runtime_kind", "fss");
        entry!("client_runtime", "compiled_fss_v1");
        entry!("ring_bits", 50);
        entry!("fractional_bits", 12);
        entry!("setup_owner", "dealer");
        entry!("generation_policy", "fixed_length_greedy");
        entry!("state_policy", "complete_prefix_recomputation");
        entry!("disclosure", "final_client_only_token_ids");
        entry!("persistent_kv", false);
        entry!("verification", Value::Null);
        entry!("admission", admission_record(a.source_minimum_owner_bytes));
        map.end()
    }
}
impl SourceExecutionBinding {
    pub fn capacity(&self) -> Capacity {
        self.capacity
    }
    pub fn new(admitted: &SourcePreflight, outputs: usize) -> Result<Self, String> {
        if !pin(&admitted.source_config_sha256) {
            return Err(reject("missing admitted config byte identity"));
        }
        let rows = admitted.plan.prefill.query_sequence as usize;
        Ok(Self {
            capacity: admitted.capacity,
            source_disk_bytes: admitted.source_disk_bytes,
            source_resident_bound_bytes: admitted.source_resident_bound_bytes,
            identity: execution_identity(
                &admitted.plan,
                &admitted.source_config_sha256,
                &admitted.source_shapes,
                &admitted.source_layouts,
                &admitted.rotary_shapes,
                admitted.source_float32_bytes,
                admitted.source_q12_value_bytes,
                admitted.rotary_float32_bytes,
                admitted.numeric_profile,
                rows,
                outputs,
                admitted.vocabulary,
                admitted.width,
                admitted.maximum_sequence,
                admitted.capacity,
            )?,
            rows,
            outputs,
            minimum_source_bytes: admitted.source_minimum_owner_bytes,
        })
    }
    pub fn checked_contract_digest(&self, contract: &[u8]) -> Result<[u8; 32], String> {
        if contract.is_empty() || contract.len() > MAX_CONTRACT_BYTES {
            return Err(reject("contract byte bound"));
        }
        let mut record = bounded_json_value_with_capacity(contract, self.capacity)?;
        if bounded_value_canonical_with_capacity(&record, MAX_CONTRACT_BYTES, self.capacity)?
            != contract
        {
            return Err(reject("contract is not canonical JSON"));
        }
        let mut fixed = json!({"schema_version":"pllm.fss.sdk-contract.v1","weights":"public",
            "runtime_kind":"fss","client_runtime":"compiled_fss_v1","ring_bits":50,"fractional_bits":12,
            "setup_owner":"dealer","generation_policy":"fixed_length_greedy","state_policy":"complete_prefix_recomputation",
            "disclosure":"final_client_only_token_ids","persistent_kv":false,"verification":null,
            "decode_steps":self.outputs-1,"selection_prefix_rows":(self.rows..self.rows+self.outputs).collect::<Vec<_>>(),
            "admission":admission_record(self.minimum_source_bytes)});
        if self.capacity == Capacity::Decoder {
            fixed["source_disk_bytes"] = json!(self.source_disk_bytes);
            fixed["source_resident_bound_bytes"] = json!(self.source_resident_bound_bytes);
        }
        let object = record
            .as_object_mut()
            .ok_or_else(|| reject("contract must be an object"))?;
        // Transport selection is checked policy, not a tensor-derived numeric
        // claim. It is bound by the complete returned contract digest (and hence
        // Manifest/Request), while the source execution identity stays unchanged.
        delivery_field(object.remove("material_delivery").as_ref())?;
        online_field(object.remove("online_schedule").as_ref())?;
        for (key, expected) in fixed.as_object().unwrap() {
            if object.remove(key).as_ref() != Some(expected) {
                return Err(reject("contract schema/policy differs"));
            }
        }
        for key in ["source_lock", "tokenizer", "configuration", "composition"] {
            if !object
                .remove(key)
                .as_ref()
                .and_then(Value::as_str)
                .is_some_and(pin)
            {
                return Err(reject(
                    "contract authenticated-caller identity claim is malformed",
                ));
            }
        }
        let capacity_fields = usize::from(self.capacity == Capacity::Decoder);
        if object.len() != EXECUTION_KEYS.len() + capacity_fields
            || EXECUTION_KEYS.iter().any(|k| !object.contains_key(*k))
            || (self.capacity == Capacity::Decoder
                && object.get("capacity") != Some(&json!("decoder")))
        {
            return Err(reject("contract execution fields differ"));
        }
        let fields: ExecutionFields = serde_json::from_value(record)
            .map_err(|_| reject("invalid typed execution contract"))?;
        let actual = execution_identity(
            &fields.plan,
            &fields.source_config_sha256,
            &fields.source_shapes,
            &fields.source_layouts,
            &fields.rotary_shapes,
            fields.source_float32_bytes,
            fields.source_q12_value_bytes,
            fields.rotary_float32_bytes,
            &fields.numeric_profile,
            fields.input_rows,
            fields.output_tokens,
            fields.vocabulary,
            fields.width,
            fields.maximum_sequence,
            fields.capacity,
        )?;
        if actual != self.identity {
            return Err(reject(
                "contract differs from immutable source execution binding",
            ));
        }
        Ok(Sha256::digest(contract).into())
    }
}

pub fn source_contract(
    experiment: &[u8],
    config: &[u8],
    rows: usize,
    outputs: usize,
    sourcepin: &str,
    tokenizerpin: &str,
) -> Result<Vec<u8>, String> {
    if experiment.is_empty() || experiment.len() > MAX_EXPERIMENT_BYTES {
        return Err(reject("Experiment byte bound"));
    }
    let document: ExperimentDocument =
        serde_json::from_slice(experiment).map_err(|_| reject("invalid Experiment"))?;
    if canonical_bytes(&document) != experiment {
        return Err(reject("Experiment must use canonical compact sorted JSON"));
    }
    validate(&document)?;
    if !pin(sourcepin) || !pin(tokenizerpin) {
        return Err(reject(
            "source/tokenizer pins must be nonzero lowercase SHA256",
        ));
    }
    if rows == 0
        || rows as u64 > document.budget.max_input_tokens
        || outputs == 0
        || outputs as u64 > document.budget.max_new_tokens
    {
        return Err(reject("request exceeds saved Experiment"));
    }
    // Do not let a small actual query conceal an unsupported saved envelope.
    let profile = profile(&document)?;
    let capacity = capacity(&document)?;
    let delivery = material_delivery(&document)?;
    let envelope = preflight_source_with_capacity(
        config,
        document.budget.max_input_tokens as usize,
        document.budget.max_new_tokens as usize,
        profile,
        capacity,
    )?;
    let admitted = if rows as u64 == document.budget.max_input_tokens
        && outputs as u64 == document.budget.max_new_tokens
    {
        envelope
    } else {
        drop(envelope);
        preflight_source_with_capacity(config, rows, outputs, profile, capacity)?
    };
    // Top-level native dispatch is mandatory, including when invoked via PyO3.
    // Parent integration owns that registration; metadata cannot bypass it.
    crate::resolve_experiment(experiment)?;
    let view = ContractView {
        admitted: &admitted,
        material_delivery: delivery,
        online_schedule: online_schedule(&document)?,
        rows,
        outputs,
        configuration: crate::configuration_digest_bytes(experiment).to_string(),
        composition: crate::pipeline_digest(&document.pipeline).to_string(),
        sourcepin,
        tokenizerpin,
    };
    bounded_canonical_with_capacity(&view, MAX_CONTRACT_BYTES, capacity)
}

#[cfg(test)]
#[path = "sdk_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "capacity_tests.rs"]
mod capacity_tests;
