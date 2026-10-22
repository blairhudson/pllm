//! Model-neutral, metadata-only inventory of declared dense matrices and tables.
//!
//! This accepts untransformed decoder metadata (normally from validated model
//! lowering). Admission below checks structural references and source geometry;
//! it is not model-family validation, numeric lowering, or an execution schedule.
//! In particular it does not interpret operation or artifact names. Full model
//! validation remains the caller's responsibility; the existing family validator
//! also imposes checkpoint naming conventions that are irrelevant to this API.
//! The config digest is the plan's declaration, not a source-verified identity.
//!
//! Supported geometry: batch one, sequences in 1..=2^20; rank-three token
//! lookups on the local `input.tokens` contract; rank-two/three dense linears
//! preserving every input prefix dimension; one unlayered rank-two output head.
//! Linear/head inputs must be operation-produced, not state-only inputs. Existing
//! grouping authority requires a nonempty contiguous set of layer IDs from zero.
//! Source matrices are [output, input], tables [vocabulary, feature], biases
//! [output]. Vocabulary comes only from the declared head output. Dimensions and
//! concatenated widths are <=2^20, individual shape products <=2^40, and summed
//! source-use elements <=2^48. Other geometry and matrix attributes fail closed.
//!
//! Metadata limits (both phases combined unless stated otherwise): 8 MiB input
//! and canonical output; 4096 operations/phase; 1024 states/direction/phase;
//! 16 inputs/operation; rank <=5; 1024 UTF-8 bytes/string; 1 MiB total string
//! bytes (keys and values); 131072 whole-document JSON value nodes; JSON depth
//! <=24 with the document root at zero; 64 object fields. Both entry points use
//! these same counts, including schema fields, enum strings, and null options.
//! Conservative output-size admission runs before grouping or serialization.
//! No tensor storage is allocated or read. Table and matrix artifact ownership
//! is keyed by (artifact_kind, weight_id), even when their source IDs are tied.

use crate::decoder_runtime_schedule::classify_remote_groups;
use pllm_models::{DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator};
use pllm_types::{canonical_bytes, digest_bytes, Digest};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{self, Write};

pub const DECODER_MATRIX_INVENTORY_SCHEMA_VERSION: &str = "pllm.decoder_matrix_inventory.v1";
pub const DECODER_MATRIX_INVENTORY_MAX_PLAN_BYTES: usize = 8 << 20;
const MAX_OUTPUT_BYTES: usize = 8 << 20;
const MAX_OPERATIONS: usize = 4096;
const MAX_STATES: usize = 1024;
const MAX_STRING_BYTES: usize = 1024;
const MAX_TOTAL_STRING_BYTES: usize = 1 << 20;
const MAX_NODES: usize = 131072;
const MAX_DEPTH: usize = 24;
const MAX_DIMENSION: u64 = 1 << 20;
const MAX_ELEMENTS: u64 = 1 << 40;
const MAX_TOTAL_ELEMENTS: u64 = 1 << 48;

/// Immutable canonical metadata snapshot; no deserializer or mutation API.
#[derive(Debug)]
pub struct DecoderMatrixInventory {
    bytes: Vec<u8>,
    digest: Digest,
    // Constructor-only binding; not new serialized fields or digest inputs.
    semantic_plan_identity: [u8; 32],
    config_identity: [u8; 32],
}

impl DecoderMatrixInventory {
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    /// Identities captured only after structural/geometry validation succeeds.
    /// These bind a later immutable plan borrow; they grant no execution authority.
    pub fn semantic_plan_identity(&self) -> [u8; 32] {
        self.semantic_plan_identity
    }

    pub fn config_identity(&self) -> [u8; 32] {
        self.config_identity
    }

    /// Retained logical payload only; prior lowering/construction is excluded.
    /// No allocator-overhead, peak-scratch or RSS claim follows from this quote.
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.bytes.capacity())?
            .checked_add(self.digest.backing_capacity_bytes())
    }
}

#[derive(Serialize)]
struct Inventory {
    schema_version: &'static str,
    model_plan_digest: Digest,
    model_config_digest: Digest,
    prefill: Phase,
    decode: Phase,
}

#[derive(Serialize)]
struct Phase {
    mode: DecoderMode,
    batch: u64,
    query_sequence: u64,
    maximum_key_sequence: u64,
    groups: Vec<Group>,
}

#[derive(Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ArtifactKind {
    TokenTable,
    Matrix,
}

#[derive(Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Orientation {
    VocabularyFeature,
    OutputInput,
}

#[derive(Serialize)]
struct Group {
    /// Dense inventory order, following the first member's graph position.
    order: u64,
    operator: ModelOperator,
    layer: Option<u64>,
    input_id: String,
    input_shape: Vec<u64>,
    artifact_kind: ArtifactKind,
    orientation: Orientation,
    /// Logical concatenation only; each member retains its own source artifact.
    logical_shape: [u64; 2],
    members: Vec<Member>,
}

#[derive(Serialize)]
struct Member {
    operation_id: String,
    weight_id: String,
    bias_id: Option<String>,
    source_shape: [u64; 2],
    bias_shape: Option<[u64; 1]>,
    output_shape: Vec<u64>,
    output_offset: u64,
    output_width: u64,
}

#[derive(Default)]
struct MetadataBudget {
    nodes: usize,
    strings: usize,
}

impl MetadataBudget {
    fn node(&mut self, depth: usize) -> Result<(), String> {
        self.nodes = self.nodes.checked_add(1).ok_or("metadata node overflow")?;
        if depth > MAX_DEPTH || self.nodes > MAX_NODES {
            return Err("matrix inventory metadata depth or node limit exceeded".into());
        }
        Ok(())
    }

    fn string(&mut self, value: &str) -> Result<(), String> {
        self.strings = self
            .strings
            .checked_add(value.len())
            .ok_or("metadata string overflow")?;
        if value.len() > MAX_STRING_BYTES || self.strings > MAX_TOTAL_STRING_BYTES {
            return Err("matrix inventory metadata string limit exceeded".into());
        }
        Ok(())
    }

    fn string_value(&mut self, value: &str, depth: usize) -> Result<(), String> {
        self.node(depth)?;
        self.string(value)
    }

    fn object(&mut self, fields: &[&str], depth: usize) -> Result<(), String> {
        self.node(depth)?;
        for key in fields {
            self.string(key)?;
        }
        Ok(())
    }

    fn dimensions(&mut self, shape: &[u64], depth: usize) -> Result<(), String> {
        self.node(depth)?;
        for _ in shape {
            self.node(depth + 1)?;
        }
        Ok(())
    }

    fn value(&mut self, value: &Value, depth: usize) -> Result<(), String> {
        self.node(depth)?;
        match value {
            Value::String(value) => self.string(value)?,
            Value::Array(values) => {
                if values.len() > MAX_OPERATIONS {
                    return Err("matrix inventory metadata array limit exceeded".into());
                }
                for value in values {
                    self.value(value, depth + 1)?;
                }
            }
            Value::Object(fields) => {
                if fields.len() > 64 {
                    return Err("matrix inventory metadata object limit exceeded".into());
                }
                for (key, value) in fields {
                    self.string(key)?;
                    self.value(value, depth + 1)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

// Streaming admission precedes construction of DecoderPlan/Value. Besides
// resource bounds this rejects duplicate JSON keys, including attribute keys.
struct JsonPreflight<'a> {
    budget: &'a mut MetadataBudget,
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for JsonPreflight<'_> {
    type Value = ();

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        self.budget
            .node(self.depth)
            .map_err(serde::de::Error::custom)?;
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for JsonPreflight<'_> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bounded decoder metadata")
    }

    fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<(), E> {
        Ok(())
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<(), E> {
        self.budget.string(value).map_err(E::custom)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        let mut count = 0;
        while seq
            .next_element_seed(JsonPreflight {
                budget: self.budget,
                depth: self.depth + 1,
            })?
            .is_some()
        {
            count += 1;
            if count > MAX_OPERATIONS {
                return Err(serde::de::Error::custom(
                    "matrix inventory metadata array limit exceeded",
                ));
            }
        }
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut keys = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            self.budget.string(&key).map_err(serde::de::Error::custom)?;
            if keys.len() == 64 || !keys.insert(key) {
                return Err(serde::de::Error::custom(
                    "matrix inventory duplicate JSON key or object limit exceeded",
                ));
            }
            map.next_value_seed(JsonPreflight {
                budget: self.budget,
                depth: self.depth + 1,
            })?;
        }
        Ok(())
    }
}

struct ByteBudget(usize);

impl Write for ByteBudget {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_sub(bytes.len())
            .ok_or_else(|| io::Error::other("metadata byte limit exceeded"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialized_bound(value: &impl Serialize, maximum: usize) -> Result<(), String> {
    serde_json::to_writer(ByteBudget(maximum), value).map_err(|error| error.to_string())
}

/// Bounded JSON entry point, shared with the internal Python binding.
pub fn decoder_matrix_inventory_bytes(bytes: &[u8]) -> Result<DecoderMatrixInventory, String> {
    if bytes.is_empty() || bytes.len() > DECODER_MATRIX_INVENTORY_MAX_PLAN_BYTES {
        return Err("matrix inventory plan byte limit exceeded".into());
    }
    // Bound serde's scratch allocation for escaped strings before parsing.
    let (mut quoted, mut escaped, mut length) = (false, false, 0_usize);
    for byte in bytes {
        if quoted {
            length += 1;
            if length > MAX_STRING_BYTES * 6 + 1 {
                return Err("matrix inventory encoded string limit exceeded".into());
            }
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                quoted = false;
            }
        } else if *byte == b'"' {
            quoted = true;
            length = 0;
        }
    }
    let mut parser = serde_json::Deserializer::from_slice(bytes);
    JsonPreflight {
        budget: &mut MetadataBudget::default(),
        depth: 0,
    }
    .deserialize(&mut parser)
    .map_err(|error| format!("invalid decoder inventory metadata: {error}"))?;
    parser.end().map_err(|error| error.to_string())?;
    let plan =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid decoder plan: {error}"))?;
    lower_decoder_matrix_inventory(&plan)
}

fn shape_elements(shape: &[u64]) -> Result<u64, String> {
    if shape.is_empty() || shape.len() > 5 {
        return Err("matrix inventory unsupported shape rank".into());
    }
    let elements = shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or("matrix inventory shape product overflow")?;
    if shape
        .iter()
        .any(|width| !(1..=MAX_DIMENSION).contains(width))
        || elements > MAX_ELEMENTS
    {
        return Err("matrix inventory shape exceeds supported geometry".into());
    }
    Ok(elements)
}

fn remote(operator: ModelOperator) -> bool {
    matches!(
        operator,
        ModelOperator::TokenLookup | ModelOperator::Linear | ModelOperator::OutputHead
    )
}

fn preflight(plan: &DecoderPlan) -> Result<(), String> {
    if plan.schema_version != pllm_models::DECODER_PLAN_SCHEMA_VERSION
        || !plan.transformations.is_empty()
    {
        return Err(
            "matrix inventory requires an untransformed decoder plan of the supported schema"
                .into(),
        );
    }
    if plan.prefill.mode != DecoderMode::Prefill
        || plan.decode.mode != DecoderMode::Decode
        || !plan.token_feedback
    {
        return Err("matrix inventory invalid decoder phase or feedback contract".into());
    }
    let mut budget = MetadataBudget::default();
    // Walk the serialized schema in place: root 0, phase 1, operations 2,
    // operation 3, attributes 4. Count keys and every value node just as the
    // streaming JSON visitor does, without serializing/allocating the plan.
    // Only fixed-size enums below are converted to their serde string/null form.
    budget.object(
        &[
            "schema_version",
            "model_family",
            "adapter",
            "config_digest",
            "transformations",
            "prefill",
            "decode",
            "token_feedback",
        ],
        0,
    )?;
    for value in [
        plan.schema_version.as_str(),
        &plan.model_family,
        &plan.adapter,
        plan.config_digest.as_str(),
    ] {
        budget.string_value(value, 1)?;
        if value.is_empty() {
            return Err("matrix inventory empty plan identity".into());
        }
    }
    budget.node(1)?; // Empty transformations array (checked above).
    budget.node(1)?; // token_feedback
    let mut output_bound = 4096_usize;
    for graph in [&plan.prefill, &plan.decode] {
        if graph.operations.is_empty()
            || graph.operations.len() > MAX_OPERATIONS
            || graph.state_inputs.len() > MAX_STATES
            || graph.state_outputs.len() > MAX_STATES
        {
            return Err("matrix inventory operation or state limit exceeded".into());
        }
        if graph.batch != 1
            || !(1..=MAX_DIMENSION).contains(&graph.query_sequence)
            || !(graph.query_sequence..=MAX_DIMENSION).contains(&graph.maximum_key_sequence)
            || (graph.mode == DecoderMode::Decode && graph.query_sequence != 1)
        {
            return Err("matrix inventory unsupported batch or sequence geometry".into());
        }
        budget.object(
            &[
                "mode",
                "batch",
                "query_sequence",
                "maximum_key_sequence",
                "operations",
                "state_inputs",
                "state_outputs",
                "output",
            ],
            1,
        )?;
        budget.value(
            &serde_json::to_value(graph.mode).map_err(|error| error.to_string())?,
            2,
        )?;
        for _ in 0..3 {
            budget.node(2)?;
        } // batch and sequence bounds
        budget.string_value(&graph.output, 2)?;
        for states in [&graph.state_inputs, &graph.state_outputs] {
            budget.node(2)?;
            for state in states {
                budget.object(&["id", "layer", "kind", "shape", "maximum_sequence"], 3)?;
                budget.string_value(&state.id, 4)?;
                budget.node(4)?; // layer (integer or null)
                budget.value(
                    &serde_json::to_value(state.kind).map_err(|error| error.to_string())?,
                    4,
                )?;
                budget.node(4)?; // maximum_sequence
                shape_elements(&state.shape)?;
                budget.dimensions(&state.shape, 4)?;
                if state.shape[0] != graph.batch
                    || state.maximum_sequence == 0
                    || state.maximum_sequence > MAX_DIMENSION
                {
                    return Err("matrix inventory unsupported state geometry".into());
                }
            }
        }
        budget.node(2)?; // operations array
        for op in &graph.operations {
            budget.object(
                &[
                    "id",
                    "operator",
                    "layer",
                    "state_kind",
                    "inputs",
                    "output_shape",
                    "attributes",
                ],
                3,
            )?;
            budget.string_value(&op.id, 4)?;
            budget.value(
                &serde_json::to_value(op.operator).map_err(|error| error.to_string())?,
                4,
            )?;
            budget.node(4)?; // layer (integer or null)
            budget.value(
                &serde_json::to_value(op.state_kind).map_err(|error| error.to_string())?,
                4,
            )?;
            if op.inputs.len() > 16 {
                return Err("matrix inventory input count limit exceeded".into());
            }
            budget.node(4)?; // inputs array
            for input in &op.inputs {
                budget.string_value(input, 5)?;
            }
            shape_elements(&op.output_shape)?;
            budget.dimensions(&op.output_shape, 4)?;
            if !op.attributes.is_object() {
                return Err("matrix inventory attributes must be an object".into());
            }
            budget.value(&op.attributes, 4)?;
            if remote(op.operator) {
                // More than the fixed fields, five-dimensional shapes, and all
                // possible JSON-escaped IDs emitted for a member and its group.
                let id_bytes = op.id.len()
                    + op.inputs.iter().map(String::len).sum::<usize>()
                    + ["weight", "bias"]
                        .iter()
                        .filter_map(|key| op.attributes.get(key).and_then(Value::as_str))
                        .map(str::len)
                        .sum::<usize>();
                output_bound = output_bound
                    .checked_add(2048 + 6 * id_bytes)
                    .ok_or("matrix inventory output size overflow")?;
                if output_bound > MAX_OUTPUT_BYTES {
                    return Err("matrix inventory output byte bound exceeded".into());
                }
            }
        }
    }
    serialized_bound(plan, DECODER_MATRIX_INVENTORY_MAX_PLAN_BYTES)
}

const PUBLIC_INPUTS: [&str; 4] = [
    "input.tokens",
    "input.positions",
    "input.attention_mask",
    "input.sequence_lengths",
];

/// Validate reference uniqueness/topological binding without family name rules.
fn validate_references(graph: &DecoderGraph) -> Result<(), String> {
    let mut available: BTreeMap<&str, Option<&[u64]>> =
        PUBLIC_INPUTS.into_iter().map(|id| (id, None)).collect();
    let mut input_slots = BTreeSet::new();
    for state in &graph.state_inputs {
        if state.id.is_empty()
            || available.insert(&state.id, Some(&state.shape)).is_some()
            || !input_slots.insert((state.layer, state.kind))
        {
            return Err("matrix inventory duplicate or empty state input ID/slot".into());
        }
    }
    for op in &graph.operations {
        if op.id.is_empty() || available.contains_key(op.id.as_str()) {
            return Err("matrix inventory duplicate or empty operation ID".into());
        }
        if op
            .inputs
            .iter()
            .any(|id| !available.contains_key(id.as_str()))
        {
            return Err("matrix inventory unbound operation input ID".into());
        }
        available.insert(&op.id, Some(&op.output_shape));
    }
    let mut output_slots = BTreeSet::new();
    let mut output_ids = BTreeSet::new();
    for state in &graph.state_outputs {
        let producer = graph.operations.iter().find(|op| op.id == state.id);
        if !output_slots.insert((state.layer, state.kind))
            || !output_ids.insert(&state.id)
            || producer.is_none_or(|op| {
                op.output_shape != state.shape
                    || op.layer != state.layer
                    || op.state_kind != Some(state.kind)
            })
        {
            return Err("matrix inventory duplicate or unbound state output metadata".into());
        }
    }
    if graph
        .operations
        .last()
        .is_none_or(|op| op.id != graph.output || op.operator != ModelOperator::TokenFeedback)
    {
        return Err("matrix inventory unbound graph output or token feedback".into());
    }
    Ok(())
}

fn artifact_id<'a>(
    op: &'a ModelOperation,
    field: &str,
    optional: bool,
) -> Result<Option<&'a str>, String> {
    match op.attributes.get(field) {
        None | Some(Value::Null) if optional => Ok(None),
        Some(Value::String(value)) if !value.is_empty() => Ok(Some(value)),
        _ => Err(format!("matrix inventory malformed {field} artifact ID")),
    }
}

fn declared_shape(
    op: &ModelOperation,
    field: &str,
    expected: Option<&[u64]>,
) -> Result<(), String> {
    if let Some(value) = op.attributes.get(field) {
        if value
            .as_array()
            .zip(expected)
            .is_none_or(|(actual, expected)| {
                actual.len() != expected.len()
                    || actual
                        .iter()
                        .zip(expected)
                        .any(|(v, e)| v.as_u64() != Some(*e))
            })
        {
            return Err(format!("matrix inventory malformed or mismatched {field}"));
        }
    }
    Ok(())
}

fn validate_attributes(op: &ModelOperation) -> Result<(), String> {
    for (key, value) in op.attributes.as_object().expect("preflight checked object") {
        let valid = match key.as_str() {
            "weight" | "bias" | "weight_shape" | "bias_shape" => true, // Checked against derived geometry below.
            "tied" if op.operator == ModelOperator::OutputHead => value.is_boolean(),
            "input_dtype" | "output_dtype" | "compute_dtype" => matches!(
                value.as_str(),
                Some("model_native" | "float32" | "float16" | "bfloat16")
            ),
            "input_layout" if op.operator == ModelOperator::OutputHead => {
                value.as_str() == Some("batch_hidden")
            }
            "output_layout" if op.operator == ModelOperator::OutputHead => {
                value.as_str() == Some("batch_vocabulary")
            }
            "checkpoint_layout" | "adapter_reference_revision"
                if op.operator == ModelOperator::TokenLookup =>
            {
                value.as_str().is_some_and(|v| !v.is_empty())
            }
            _ => false,
        };
        if !valid {
            return Err(format!(
                "matrix inventory unsupported or malformed matrix attribute {key}"
            ));
        }
    }
    Ok(())
}

#[derive(Default)]
struct Sources<'a> {
    shapes: BTreeMap<&'a str, Vec<u64>>,
    elements: u64,
}

impl<'a> Sources<'a> {
    fn bind(&mut self, id: &'a str, shape: &[u64], graph: &DecoderGraph) -> Result<(), String> {
        if PUBLIC_INPUTS.contains(&id)
            || graph.operations.iter().any(|op| op.id == id)
            || graph
                .state_inputs
                .iter()
                .chain(&graph.state_outputs)
                .any(|state| state.id == id)
        {
            return Err("matrix inventory source artifact ID aliases a graph value".into());
        }
        self.elements = self
            .elements
            .checked_add(shape_elements(shape)?)
            .ok_or("matrix inventory source size sum overflow")?;
        if self.elements > MAX_TOTAL_ELEMENTS {
            return Err("matrix inventory source size sum exceeds limit".into());
        }
        if let Some(previous) = self.shapes.get(id) {
            if previous != shape {
                return Err("matrix inventory inconsistent source artifact shape".into());
            }
        } else {
            self.shapes.insert(id, shape.to_vec());
        }
        Ok(())
    }
}

fn lower_phase<'a>(graph: &'a DecoderGraph, sources: &mut Sources<'a>) -> Result<Phase, String> {
    validate_references(graph)?;
    let heads: Vec<_> = graph
        .operations
        .iter()
        .filter(|op| op.operator == ModelOperator::OutputHead)
        .collect();
    if heads.len() != 1
        || heads[0].output_shape.len() != 2
        || heads[0].output_shape[0] != graph.batch
    {
        return Err("matrix inventory requires one rank-two output head".into());
    }
    let vocabulary = heads[0].output_shape[1];
    if let Some(tied) = heads[0].attributes.get("tied").and_then(Value::as_bool) {
        let shares_table = graph.operations.iter().any(|op| {
            op.operator == ModelOperator::TokenLookup
                && op.attributes.get("weight") == heads[0].attributes.get("weight")
        });
        if tied != shares_table {
            return Err(
                "matrix inventory head tied declaration differs from source ownership".into(),
            );
        }
    }
    let shapes: BTreeMap<_, _> = graph
        .operations
        .iter()
        .map(|op| (op.id.as_str(), op.output_shape.as_slice()))
        .collect();
    let leaders = classify_remote_groups(graph)?;
    let mut groups = Vec::with_capacity(leaders.len());
    for op in &graph.operations {
        let Some(group) = leaders.get(op.id.as_str()) else {
            continue;
        };
        let token = op.operator == ModelOperator::TokenLookup;
        if op.inputs.len() != 1
            || op.state_kind.is_some()
            || (op.operator != ModelOperator::Linear && op.layer.is_some())
        {
            return Err("matrix inventory unsupported matrix input or layer contract".into());
        }
        let input_shape = if token {
            if op.inputs[0] != "input.tokens" {
                return Err("matrix inventory token lookup requires local input.tokens".into());
            }
            vec![graph.batch, graph.query_sequence]
        } else {
            shapes
                .get(op.inputs[0].as_str())
                .ok_or("matrix inventory linear input has no operation source shape")?
                .to_vec()
        };
        let mut offset = 0_u64;
        let mut members = Vec::with_capacity(group.operations.len());
        for member in group.operations.iter() {
            validate_attributes(member)?;
            if member.inputs != op.inputs
                || member.operator != op.operator
                || member.layer != op.layer
                || member.state_kind.is_some()
            {
                return Err("matrix inventory group semantic inputs/layer/operator differ".into());
            }
            let output = &member.output_shape;
            let width = *output
                .last()
                .ok_or("matrix inventory missing output width")?;
            let source_shape = if token {
                if output.len() != 3 || output[..2] != input_shape {
                    return Err("matrix inventory token lookup prefix dimensions differ".into());
                }
                [vocabulary, width]
            } else {
                if !matches!(input_shape.len(), 2 | 3)
                    || input_shape[0] != graph.batch
                    || output.len() != input_shape.len()
                    || output[..output.len() - 1] != input_shape[..input_shape.len() - 1]
                    || (input_shape.len() == 3 && input_shape[1] != graph.query_sequence)
                    || (op.operator == ModelOperator::OutputHead && input_shape.len() != 2)
                {
                    return Err(
                        "matrix inventory linear input/output prefix dimensions differ".into(),
                    );
                }
                [width, *input_shape.last().expect("rank checked")]
            };
            let weight_id = artifact_id(member, "weight", false)?.expect("required artifact");
            let bias_id = artifact_id(member, "bias", true)?;
            if token && bias_id.is_some() {
                return Err("matrix inventory token table cannot have bias".into());
            }
            declared_shape(member, "weight_shape", Some(&source_shape))?;
            let bias_shape = bias_id.map(|_| [width]);
            declared_shape(
                member,
                "bias_shape",
                bias_shape.as_ref().map(|shape| shape.as_slice()),
            )?;
            sources.bind(weight_id, &source_shape, graph)?;
            if let Some(bias_id) = bias_id {
                sources.bind(bias_id, &[width], graph)?;
            }
            members.push(Member {
                operation_id: member.id.clone(),
                weight_id: weight_id.to_owned(),
                bias_id: bias_id.map(str::to_owned),
                source_shape,
                bias_shape,
                output_shape: output.clone(),
                output_offset: offset,
                output_width: width,
            });
            offset = offset
                .checked_add(width)
                .ok_or("matrix inventory group width sum overflow")?;
            if offset > MAX_DIMENSION {
                return Err("matrix inventory grouped width exceeds supported geometry".into());
            }
        }
        let logical_shape = if token {
            [vocabulary, offset]
        } else {
            [offset, input_shape[input_shape.len() - 1]]
        };
        shape_elements(&logical_shape)?;
        groups.push(Group {
            order: groups.len() as u64,
            operator: op.operator,
            layer: op.layer,
            input_id: op.inputs[0].clone(),
            input_shape,
            artifact_kind: if token {
                ArtifactKind::TokenTable
            } else {
                ArtifactKind::Matrix
            },
            orientation: if token {
                Orientation::VocabularyFeature
            } else {
                Orientation::OutputInput
            },
            logical_shape,
            members,
        });
    }
    Ok(Phase {
        mode: graph.mode,
        batch: graph.batch,
        query_sequence: graph.query_sequence,
        maximum_key_sequence: graph.maximum_key_sequence,
        groups,
    })
}

fn compatible(prefill: &Phase, decode: &Phase, plan: &DecoderPlan) -> Result<(), String> {
    if prefill.groups.len() != decode.groups.len() {
        return Err("matrix inventory incompatible phase groups".into());
    }
    let attrs: BTreeMap<_, _> = plan
        .decode
        .operations
        .iter()
        .filter(|op| remote(op.operator))
        .map(|op| (op.id.as_str(), &op.attributes))
        .collect();
    if plan
        .prefill
        .operations
        .iter()
        .filter(|op| remote(op.operator))
        .any(|op| attrs.get(op.id.as_str()) != Some(&&op.attributes))
    {
        return Err("matrix inventory incompatible phase matrix attributes".into());
    }
    for (a, b) in prefill.groups.iter().zip(&decode.groups) {
        if a.operator != b.operator
            || a.layer != b.layer
            || a.input_id != b.input_id
            || a.artifact_kind != b.artifact_kind
            || a.orientation != b.orientation
            || a.logical_shape != b.logical_shape
            || a.input_shape.len() != b.input_shape.len()
            || a.members.len() != b.members.len()
        {
            return Err("matrix inventory incompatible phase groups".into());
        }
        for (a, b) in a.members.iter().zip(&b.members) {
            if a.operation_id != b.operation_id
                || a.weight_id != b.weight_id
                || a.bias_id != b.bias_id
                || a.source_shape != b.source_shape
                || a.bias_shape != b.bias_shape
                || a.output_offset != b.output_offset
                || a.output_width != b.output_width
                || a.output_shape.len() != b.output_shape.len()
            {
                return Err("matrix inventory incompatible phase group members".into());
            }
        }
    }
    Ok(())
}

/// Derive a canonical, plan/config-bound snapshot without an executable composition.
/// Only this constructor can create an inventory; all validation precedes hashing.
pub fn lower_decoder_matrix_inventory(
    plan: &DecoderPlan,
) -> Result<DecoderMatrixInventory, String> {
    preflight(plan)?;
    let mut sources = Sources::default();
    let prefill = lower_phase(&plan.prefill, &mut sources)?;
    let decode = lower_phase(&plan.decode, &mut sources)?;
    compatible(&prefill, &decode, plan)?;
    let inventory = Inventory {
        schema_version: DECODER_MATRIX_INVENTORY_SCHEMA_VERSION,
        model_plan_digest: plan.digest(),
        model_config_digest: plan.config_digest.clone(),
        prefill,
        decode,
    };
    serialized_bound(&inventory, MAX_OUTPUT_BYTES)?;
    let bytes = canonical_bytes(&inventory);
    let digest = digest_bytes(DECODER_MATRIX_INVENTORY_SCHEMA_VERSION, &bytes);
    Ok(DecoderMatrixInventory {
        bytes,
        digest,
        semantic_plan_identity: inventory.model_plan_digest.as_sha256(),
        config_identity: inventory.model_config_digest.as_sha256(),
    })
}
