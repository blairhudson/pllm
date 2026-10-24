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

use crate::decoder_runtime_schedule::{RemoteGroupingError, RemoteGroupingQuote, RemoteGroups};
use pllm_models::bounded_canonical as model_canonical;
use pllm_models::{DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator};
use pllm_types::bounded_canonical::{DomainHash, Session, StreamError, StreamFrame, StreamLimits};
#[cfg(test)]
use pllm_types::digest_bytes;
use pllm_types::Digest;
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fmt;
use std::io::{self, Write};
use std::mem::{size_of, MaybeUninit};

#[path = "decoder_matrix_inventory_construction.rs"]
#[allow(
    dead_code,
    reason = "guarded constructor stays private pending independent admission audit"
)]
mod construction;

// Compiler-only typed owning-input funnel; no raw import/SDK registration.
pub(crate) use construction::{
    admit_inventory_for_original_owner, admit_inventory_for_plan_owner,
    owned_inventory_bootstrap_bytes,
};

pub const DECODER_MATRIX_INVENTORY_SCHEMA_VERSION: &str = "pllm.decoder_matrix_inventory.v1";
pub const DECODER_MATRIX_INVENTORY_MAX_PLAN_BYTES: usize = 8 << 20;
const MAX_OUTPUT_BYTES: usize = 8 << 20;
const MAX_OPERATIONS: usize = 4096;
const MAX_STATES: usize = 1024;
const MAX_INPUTS: usize = 16;
const MAX_OBJECT_FIELDS: usize = 64;
const MAX_STRING_BYTES: usize = 1024;
const MAX_TOTAL_STRING_BYTES: usize = 1 << 20;
const MAX_NODES: usize = 131072;
const MAX_DEPTH: usize = 24;
const MAX_DIMENSION: u64 = 1 << 20;
const MAX_ELEMENTS: u64 = 1 << 40;
const MAX_TOTAL_ELEMENTS: u64 = 1 << 48;
const CANONICAL_SLOTS: usize = MAX_DEPTH + 1;

/// Inventory diagnostics use static/borrowed strings, formatted only at the
/// legacy String boundary. Shared canonical/stream errors remain unchanged; their
/// potential I/O owner/result slots still belong in the aggregate error proof.
#[derive(Debug)]
enum InventoryFailure<'a> {
    Static(&'static str),
    Artifact(&'static str),
    DeclaredShape(&'static str),
    Attribute(&'a str),
    Grouping(RemoteGroupingError<'a>),
    Canonical(model_canonical::Error),
    Stream(StreamError),
    Index(crate::bounded_index::Error),
    Json(pllm_types::bounded_canonical::borrowed::Error),
    Census(pllm_models::owned_backing::bounded::Error),
    Native(pllm_garble::shared_allocation::Error),
}

impl From<&'static str> for InventoryFailure<'_> {
    fn from(value: &'static str) -> Self {
        Self::Static(value)
    }
}

impl fmt::Display for InventoryFailure<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static(error) => f.write_str(error),
            Self::Artifact(field) => write!(f, "matrix inventory malformed {field} artifact ID"),
            Self::DeclaredShape(field) => {
                write!(f, "matrix inventory malformed or mismatched {field}")
            }
            Self::Attribute(key) => write!(
                f,
                "matrix inventory unsupported or malformed matrix attribute {key}"
            ),
            Self::Grouping(error) => fmt::Display::fmt(error, f),
            Self::Canonical(error) => fmt::Display::fmt(error, f),
            Self::Stream(error) => fmt::Display::fmt(error, f),
            Self::Index(error) => write!(f, "matrix inventory bounded index: {error:?}"),
            Self::Json(error) => fmt::Display::fmt(error, f),
            Self::Census(error) => fmt::Display::fmt(error, f),
            Self::Native(error) => write!(f, "matrix inventory original reservation: {error:?}"),
        }
    }
}

impl std::error::Error for InventoryFailure<'_> {}
impl serde::ser::Error for InventoryFailure<'static> {
    fn custom<T: fmt::Display>(_: T) -> Self {
        Self::Static("matrix inventory unsupported metadata enum serialization")
    }
}

#[cfg(test)]
thread_local! {
    static METADATA_FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn plan_canonical_limits(max_work: usize) -> model_canonical::Limits {
    model_canonical::Limits {
        canonical: StreamLimits {
            max_nodes: MAX_NODES,
            max_depth: CANONICAL_SLOTS,
            max_bytes: DECODER_MATRIX_INVENTORY_MAX_PLAN_BYTES,
            max_string_bytes: MAX_STRING_BYTES,
            max_key_bytes: MAX_STRING_BYTES,
            max_total_string_bytes: MAX_TOTAL_STRING_BYTES,
            max_work,
            max_stack_bytes: std::mem::size_of::<[StreamFrame<'_>; CANONICAL_SLOTS]>(),
        },
        max_operations: MAX_OPERATIONS * 2,
        max_states: MAX_STATES * 4,
        max_inputs: MAX_OPERATIONS * 2 * MAX_INPUTS,
        max_rank: 5,
        max_transformations: 0,
        max_domain_bytes: pllm_models::DECODER_PLAN_SCHEMA_VERSION.len(),
    }
}

fn checked_inventory_sum(parts: &[usize]) -> Result<usize, &'static str> {
    parts.iter().try_fold(0usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or("matrix inventory resource byte overflow")
    })
}

/// GROUPING ONLY: this is not a quote for `lower_decoder_matrix_inventory`.
/// Existing plan ownership is included (no caller-supplied baseline). Vec/String
/// capacities are actual; opaque attribute objects use pllm-models' documented
/// shared conservative owner policy, not an observed B-tree layout. Group keys
/// and strings are borrowed; index buffers have exact module-owned layouts.
/// Native/dependency call stacks, allocator headers and RSS are not quoted.
/// Workspace fields cover grouping/identity replay, not a complete census of the
/// allocation-free quote evaluator's own bookkeeping or other enclosing owners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderMatrixInventoryGroupingResources {
    pub plan_owner_bound_bytes: usize,
    pub grouping_workspace_bytes: usize,
    pub canonical_workspace_bytes: usize,
    pub admission_slots_bytes: usize,
    pub grouping_heap_bytes: usize,
    /// Index construction only, not canonical hashing or other enclosing work.
    pub grouping_work_bound: u128,
    pub peak_bytes: usize,
}

/// Immutable original-plan borrow prevents stale identity/capacity quotes in
/// safe Rust. This partial seam deliberately cannot construct an inventory.
///
/// ```compile_fail
/// use pllm_compiler::DecoderMatrixInventory;
/// use pllm_models::DecoderPlan;
/// fn stale_quote(plan: &mut DecoderPlan) {
///     let quote = DecoderMatrixInventory::checked_grouping_quote(plan).unwrap();
///     plan.prefill.output.push_str("changed");
///     let _ = quote.resources();
/// }
/// ```
///
/// Spare capacities are frozen by the same original-owner borrow:
/// ```compile_fail
/// use pllm_compiler::DecoderMatrixInventory;
/// use pllm_models::DecoderPlan;
/// fn stale_capacity(plan: &mut DecoderPlan) {
///     let quote = DecoderMatrixInventory::checked_grouping_quote(plan).unwrap();
///     plan.prefill.operations[0].id.reserve(128);
///     let _ = quote.resources();
/// }
/// ```
#[derive(Debug)]
pub struct DecoderMatrixInventoryGroupingQuote<'a> {
    plan: &'a DecoderPlan,
    identity: [u8; 32],
    grouping: [RemoteGroupingQuote<'a>; 2],
    resources: DecoderMatrixInventoryGroupingResources,
}

#[derive(Debug)]
enum GroupingQuoteFailure<'a> {
    Static(&'static str),
    Owner(pllm_models::owned_backing::Error),
    Canonical(model_canonical::Error),
    Grouping(RemoteGroupingError<'a>),
}

#[derive(Debug)]
pub struct DecoderMatrixInventoryGroupingError<'a>(GroupingQuoteFailure<'a>);

impl fmt::Display for DecoderMatrixInventoryGroupingError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            GroupingQuoteFailure::Static(error) => f.write_str(error),
            GroupingQuoteFailure::Owner(error) => fmt::Display::fmt(error, f),
            GroupingQuoteFailure::Canonical(error) => fmt::Display::fmt(error, f),
            GroupingQuoteFailure::Grouping(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for DecoderMatrixInventoryGroupingError<'_> {}

/// A grouping view, not an inventory or an execution/trust authority.
#[derive(Debug)]
pub struct DecoderMatrixInventoryGrouping<'a> {
    plan: &'a DecoderPlan,
    identity: [u8; 32],
    resources: DecoderMatrixInventoryGroupingResources,
    views: [RemoteGroups<'a>; 2],
}

impl<'a> DecoderMatrixInventoryGroupingQuote<'a> {
    pub fn resources(&self) -> DecoderMatrixInventoryGroupingResources {
        self.resources
    }

    /// The caller MUST hold the original enclosing reservation across this call
    /// and returned view/errors. A numeric cap is NOT a Budget guard or authority.
    /// No downstream inventory construction is covered by this partial quote.
    pub fn admit(
        self,
        reserved_capacity: usize,
    ) -> Result<DecoderMatrixInventoryGrouping<'a>, DecoderMatrixInventoryGroupingError<'a>> {
        let work = self.resources.grouping_work_bound;
        self.admit_with_work_limit(reserved_capacity, work)
    }

    /// Same grouping-only construction with an explicit combined work ceiling.
    /// Caller MUST retain the original real reservation through returned lifetime;
    /// neither numeric argument creates Budget authority.
    pub fn admit_with_work_limit(
        self,
        reserved_capacity: usize,
        max_grouping_work: u128,
    ) -> Result<DecoderMatrixInventoryGrouping<'a>, DecoderMatrixInventoryGroupingError<'a>> {
        if self.resources.peak_bytes > reserved_capacity {
            return Err(DecoderMatrixInventoryGroupingError(
                GroupingQuoteFailure::Static("matrix inventory grouping capacity limit exceeded"),
            ));
        }
        if max_grouping_work < self.resources.grouping_work_bound {
            return Err(DecoderMatrixInventoryGroupingError(
                GroupingQuoteFailure::Static("matrix inventory grouping work limit exceeded"),
            ));
        }
        let Self {
            plan,
            identity,
            grouping: [prefill, decode],
            resources,
        } = self;
        let p = prefill.resources();
        let prefill = prefill
            .construct(p.workspace_bytes, p.work_bound)
            .map_err(|error| {
                DecoderMatrixInventoryGroupingError(GroupingQuoteFailure::Grouping(error))
            })?;
        let d = decode.resources();
        let decode = decode
            .construct(d.workspace_bytes, d.work_bound)
            .map_err(|error| {
                DecoderMatrixInventoryGroupingError(GroupingQuoteFailure::Grouping(error))
            })?;
        Ok(DecoderMatrixInventoryGrouping {
            plan,
            identity,
            resources,
            views: [prefill, decode],
        })
    }
}

impl<'a> DecoderMatrixInventoryGrouping<'a> {
    pub fn plan(&self) -> &'a DecoderPlan {
        self.plan
    }
    pub fn semantic_plan_identity(&self) -> [u8; 32] {
        self.identity
    }
    pub fn group_counts(&self) -> [usize; 2] {
        [self.views[0].len(), self.views[1].len()]
    }
    /// Members remain in original graph order and borrow the original owner.
    /// `None` means the ID is not a remote group leader in this phase.
    pub fn group_members(
        &self,
        mode: DecoderMode,
        leader_id: &str,
    ) -> Option<impl Iterator<Item = &'a ModelOperation> + '_> {
        self.views[match mode {
            DecoderMode::Prefill => 0,
            DecoderMode::Decode => 1,
        }]
        .get(leader_id)
        .map(|group| group.operations.iter())
    }
    pub fn resources(&self) -> DecoderMatrixInventoryGroupingResources {
        self.resources
    }
}

/// Immutable canonical metadata snapshot; no deserializer or mutation API.
/// Guarded construction remains compiler-private, not an SDK admission route:
/// This checks privacy only, not the original-owner lifetime proof.
/// ```compile_fail
/// use pllm_compiler::decoder_matrix_inventory::construction::ConstructionQuote;
/// ```
/// The original charged input exposes no mutable payload/capacity replacement:
/// ```compile_fail
/// use pllm_garble::shared_allocation::OwnedCharged;
/// use pllm_models::DecoderPlan;
/// fn replace_capacity(input: &OwnedCharged<DecoderPlan>) {
///     input.payload().prefill.operations.reserve(1);
/// }
/// ```
#[derive(Debug)]
pub struct DecoderMatrixInventory {
    bytes: Vec<u8>,
    digest: Digest,
    // Constructor-only binding; not new serialized fields or digest inputs.
    semantic_plan_identity: [u8; 32],
    config_identity: [u8; 32],
}

impl DecoderMatrixInventory {
    /// Allocation-free partial quote over the genuine original owner. Bounded
    /// identity has an explicit work cap; the shared owner census rejects JSON
    /// attribute depth >16. These are new-path limits, not legacy behavior changes.
    /// This quotes resources; indexed grouping validation runs during admit.
    /// Grouping validity is NOT structural/geometry/source inventory validation.
    pub fn checked_grouping_quote(
        plan: &DecoderPlan,
    ) -> Result<DecoderMatrixInventoryGroupingQuote<'_>, DecoderMatrixInventoryGroupingError<'_>>
    {
        let static_error =
            |message| DecoderMatrixInventoryGroupingError(GroupingQuoteFailure::Static(message));
        for graph in [&plan.prefill, &plan.decode] {
            if graph.operations.len() > MAX_OPERATIONS
                || graph.state_inputs.len() > MAX_STATES
                || graph.state_outputs.len() > MAX_STATES
            {
                return Err(static_error(
                    "matrix inventory operation or state limit exceeded",
                ));
            }
            for op in &graph.operations {
                if op.inputs.len() > MAX_INPUTS || op.output_shape.len() > 5 {
                    return Err(static_error(
                        "matrix inventory input count or rank limit exceeded",
                    ));
                }
            }
        }
        let mut frames = [StreamFrame::empty(); CANONICAL_SLOTS];
        let identity =
            model_canonical::bounded_plan_digest(plan, plan_canonical_limits(1 << 28), &mut frames)
                .map_err(|error| {
                    DecoderMatrixInventoryGroupingError(GroupingQuoteFailure::Canonical(error))
                })?;
        let backing = plan
            .checked_owned_backing_bytes(usize::MAX)
            .map_err(|error| {
                DecoderMatrixInventoryGroupingError(GroupingQuoteFailure::Owner(error))
            })?;
        let prefill = RemoteGroupingQuote::checked(&plan.prefill).map_err(|error| {
            DecoderMatrixInventoryGroupingError(GroupingQuoteFailure::Grouping(error))
        })?;
        let decode = RemoteGroupingQuote::checked(&plan.decode).map_err(|error| {
            DecoderMatrixInventoryGroupingError(GroupingQuoteFailure::Grouping(error))
        })?;
        let canonical =
            model_canonical::resources::<DomainHash>(CANONICAL_SLOTS).map_err(|error| {
                DecoderMatrixInventoryGroupingError(GroupingQuoteFailure::Canonical(error))
            })?;
        let sum = |parts: &[usize]| checked_inventory_sum(parts).map_err(static_error);
        let plan_owner_bound_bytes = sum(&[std::mem::size_of::<DecoderPlan>(), backing])?;
        let admission_slots_bytes = sum(&[
            std::mem::size_of::<DecoderMatrixInventoryGroupingQuote<'_>>(),
            std::mem::size_of::<
                Result<
                    DecoderMatrixInventoryGroupingQuote<'_>,
                    DecoderMatrixInventoryGroupingError<'_>,
                >,
            >(),
            std::mem::size_of::<
                Result<DecoderMatrixInventoryGrouping<'_>, DecoderMatrixInventoryGroupingError<'_>>,
            >(),
            std::mem::size_of::<DecoderMatrixInventoryGroupingResources>(),
            std::mem::size_of::<[u8; 32]>(),
        ])?;
        // Frames are present for the whole quote call, including grouping. The
        // model quote already includes them; grouping needs them added explicitly.
        let p = prefill.resources();
        let d = decode.resources();
        let grouping_heap_bytes = sum(&[p.heap_bytes, d.heap_bytes])?;
        let grouping_work_bound = p
            .work_bound
            .checked_add(d.work_bound)
            .ok_or_else(|| static_error("matrix inventory grouping work overflow"))?;
        // Conservative sum retains BOTH indexed owners and all their concrete
        // construction slots; no missing prefill-owner/decode-factory overlap.
        let grouping_workspace_bytes = sum(&[
            std::mem::size_of_val(&frames),
            p.workspace_bytes,
            d.workspace_bytes,
        ])?;
        let resources = DecoderMatrixInventoryGroupingResources {
            plan_owner_bound_bytes,
            grouping_workspace_bytes,
            canonical_workspace_bytes: canonical.hash_workspace_bytes,
            admission_slots_bytes,
            grouping_heap_bytes,
            grouping_work_bound,
            peak_bytes: sum(&[
                plan_owner_bound_bytes,
                admission_slots_bytes,
                grouping_workspace_bytes.max(canonical.hash_workspace_bytes),
            ])?,
        };
        Ok(DecoderMatrixInventoryGroupingQuote {
            plan,
            identity,
            grouping: [prefill, decode],
            resources,
        })
    }

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
struct Inventory<'a> {
    schema_version: &'static str,
    model_plan_digest: HexDigest,
    model_config_digest: &'a Digest,
    prefill: Phase<'a>,
    decode: Phase<'a>,
}

/// Fixed-size encoding of the shared canonical SHA-256 result, not a second
/// hashing algorithm. Its length and lowercase ASCII invariant are structural.
struct HexDigest {
    raw: [u8; 32],
    text: [u8; 64],
}

impl HexDigest {
    fn new(raw: [u8; 32]) -> Self {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut text = [0; 64];
        for (i, byte) in raw.into_iter().enumerate() {
            text[2 * i] = HEX[(byte >> 4) as usize];
            text[2 * i + 1] = HEX[(byte & 15) as usize];
        }
        Self { raw, text }
    }
    fn as_str(&self) -> &str {
        // All constructors select ASCII exclusively from HEX above.
        std::str::from_utf8(&self.text).expect("fixed lowercase hexadecimal")
    }
    fn into_digest(self) -> Digest {
        let text: Box<[u8]> = Box::new(self.text);
        let text = String::from_utf8(text.into_vec()).expect("fixed lowercase hexadecimal");
        // StringDeserializer transfers this exact 64-byte owner through the
        // existing Digest validator. No parse scratch or clone/normalization.
        serde::Deserialize::deserialize(serde::de::value::StringDeserializer::<
            serde::de::value::Error,
        >::new(text))
        .expect("fixed validated SHA-256 encoding")
    }
}

impl Serialize for HexDigest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[derive(Serialize)]
struct Phase<'a> {
    mode: DecoderMode,
    batch: u64,
    query_sequence: u64,
    maximum_key_sequence: u64,
    groups: Box<[Group<'a>]>,
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
struct Group<'a> {
    /// Dense inventory order, following the first member's graph position.
    order: u64,
    operator: ModelOperator,
    layer: Option<u64>,
    input_id: &'a str,
    input_shape: InputShape<'a>,
    artifact_kind: ArtifactKind,
    orientation: Orientation,
    /// Logical concatenation only; each member retains its own source artifact.
    logical_shape: [u64; 2],
    members: Box<[Member<'a>]>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(untagged)]
enum InputShape<'a> {
    Tokens([u64; 2]),
    Produced(&'a [u64]),
}

impl std::ops::Deref for InputShape<'_> {
    type Target = [u64];
    fn deref(&self) -> &[u64] {
        match self {
            Self::Tokens(shape) => shape,
            Self::Produced(shape) => shape,
        }
    }
}

#[derive(Serialize)]
struct Member<'a> {
    operation_id: &'a str,
    weight_id: &'a str,
    bias_id: Option<&'a str>,
    source_shape: [u64; 2],
    bias_shape: Option<[u64; 1]>,
    output_shape: &'a [u64],
    output_offset: u64,
    output_width: u64,
}

struct MetadataBudget {
    nodes: usize,
    strings: usize,
    work: u128,
    work_limit: u128,
    settled_work: u128,
}

impl Default for MetadataBudget {
    fn default() -> Self {
        Self {
            nodes: 0,
            strings: 0,
            work: 0,
            work_limit: u128::MAX,
            settled_work: 0,
        }
    }
}

impl MetadataBudget {
    fn admit_work(&mut self, units: u128) -> Result<(), InventoryFailure<'static>> {
        let next = self
            .work
            .checked_add(units)
            .ok_or("metadata work overflow")?;
        if next > self.work_limit {
            return Err("matrix inventory aggregate work limit exceeded".into());
        }
        self.work = next;
        Ok(())
    }

    fn node(&mut self, depth: usize) -> Result<(), InventoryFailure<'static>> {
        self.admit_work(1)?;
        self.nodes = self.nodes.checked_add(1).ok_or("metadata node overflow")?;
        if depth > MAX_DEPTH || self.nodes > MAX_NODES {
            return Err("matrix inventory metadata depth or node limit exceeded".into());
        }
        Ok(())
    }

    fn string(&mut self, value: &str) -> Result<(), InventoryFailure<'static>> {
        self.admit_work(value.len() as u128 + 1)?;
        self.strings = self
            .strings
            .checked_add(value.len())
            .ok_or("metadata string overflow")?;
        if value.len() > MAX_STRING_BYTES || self.strings > MAX_TOTAL_STRING_BYTES {
            return Err("matrix inventory metadata string limit exceeded".into());
        }
        Ok(())
    }

    fn string_value(&mut self, value: &str, depth: usize) -> Result<(), InventoryFailure<'static>> {
        self.node(depth)?;
        self.string(value)
    }

    fn object(&mut self, fields: &[&str], depth: usize) -> Result<(), InventoryFailure<'static>> {
        self.node(depth)?;
        for key in fields {
            self.string(key)?;
        }
        Ok(())
    }

    fn dimensions(&mut self, shape: &[u64], depth: usize) -> Result<(), InventoryFailure<'static>> {
        self.node(depth)?;
        for _ in shape {
            self.node(depth + 1)?;
        }
        Ok(())
    }

    fn checked_shape(&mut self, shape: &[u64]) -> Result<u64, InventoryFailure<'static>> {
        // shape_elements first checks rank, then makes product/range passes.
        if !shape.is_empty() && shape.len() <= 5 {
            self.admit_work(shape.len() as u128 + shape.len() as u128)?;
        }
        shape_elements(shape)
    }

    fn value(&mut self, value: &Value, depth: usize) -> Result<(), InventoryFailure<'static>> {
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
                if fields.len() > MAX_OBJECT_FIELDS {
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

    fn enumeration<T: Serialize>(
        &mut self,
        value: &T,
        depth: usize,
    ) -> Result<(), InventoryFailure<'static>> {
        value.serialize(MetadataEnum {
            budget: self,
            depth,
        })
    }
}

/// Count serde's *actual* enum spelling/null contract without a temporary Value,
/// String, formatting buffer, or a second handwritten enum-name table.
struct MetadataEnum<'a> {
    budget: &'a mut MetadataBudget,
    depth: usize,
}

macro_rules! unsupported_enum_scalar {
    ($($method:ident($argument:ty)),* $(,)?) => {$(
        fn $method(self, _: $argument) -> Result<(), Self::Error> {
            Err("matrix inventory unsupported metadata enum serialization".into())
        }
    )*};
}

impl serde::Serializer for MetadataEnum<'_> {
    type Ok = ();
    type Error = InventoryFailure<'static>;
    type SerializeSeq = serde::ser::Impossible<(), Self::Error>;
    type SerializeTuple = serde::ser::Impossible<(), Self::Error>;
    type SerializeTupleStruct = serde::ser::Impossible<(), Self::Error>;
    type SerializeTupleVariant = serde::ser::Impossible<(), Self::Error>;
    type SerializeMap = serde::ser::Impossible<(), Self::Error>;
    type SerializeStruct = serde::ser::Impossible<(), Self::Error>;
    type SerializeStructVariant = serde::ser::Impossible<(), Self::Error>;
    fn serialize_str(self, value: &str) -> Result<(), Self::Error> {
        self.budget.string_value(value, self.depth)
    }
    fn serialize_none(self) -> Result<(), Self::Error> {
        self.budget.node(self.depth)
    }
    fn serialize_unit(self) -> Result<(), Self::Error> {
        self.budget.node(self.depth)
    }
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), Self::Error> {
        value.serialize(self)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<(), Self::Error> {
        self.serialize_str(variant)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Self::Error> {
        self.serialize_unit()
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: &T,
    ) -> Result<(), Self::Error> {
        Err("matrix inventory unsupported metadata enum serialization".into())
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: &T,
    ) -> Result<(), Self::Error> {
        Err("matrix inventory unsupported metadata enum serialization".into())
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Err("matrix inventory unsupported metadata enum serialization".into())
    }
    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Err("matrix inventory unsupported metadata enum serialization".into())
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Err("matrix inventory unsupported metadata enum serialization".into())
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Err("matrix inventory unsupported metadata enum serialization".into())
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Err("matrix inventory unsupported metadata enum serialization".into())
    }
    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Err("matrix inventory unsupported metadata enum serialization".into())
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Err("matrix inventory unsupported metadata enum serialization".into())
    }
    fn collect_str<T: ?Sized + fmt::Display>(self, _: &T) -> Result<(), Self::Error> {
        Err("matrix inventory unsupported metadata enum serialization".into())
    }
    unsupported_enum_scalar! { serialize_bool(bool), serialize_i8(i8), serialize_i16(i16), serialize_i32(i32), serialize_i64(i64), serialize_u8(u8), serialize_u16(u16), serialize_u32(u32), serialize_u64(u64), serialize_f32(f32), serialize_f64(f64), serialize_char(char), serialize_bytes(&[u8]) }
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
            if keys.len() == MAX_OBJECT_FIELDS || !keys.insert(key) {
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

fn shape_elements(shape: &[u64]) -> Result<u64, InventoryFailure<'static>> {
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

fn preflight(plan: &DecoderPlan) -> Result<(), InventoryFailure<'static>> {
    preflight_with_work(plan, usize::MAX)
}

fn preflight_with_work(
    plan: &DecoderPlan,
    max_canonical_work: usize,
) -> Result<(), InventoryFailure<'static>> {
    preflight_inner(plan, max_canonical_work, None)
}

fn preflight_inner<'a>(
    plan: &'a DecoderPlan,
    max_canonical_work: usize,
    mut context: Option<&mut construction::Context<'a>>,
) -> Result<(), InventoryFailure<'static>> {
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
    if let Some(context) = context.as_deref_mut() {
        budget.work_limit = context.remaining();
    }
    // Walk the serialized schema in place: root 0, phase 1, operations 2,
    // operation 3, attributes 4. Count keys and every value node just as the
    // streaming JSON visitor does, without serializing/allocating the plan.
    // Fixed-size enums are counted through their serde string/null events.
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
        budget.enumeration(&graph.mode, 2)?;
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
                budget.enumeration(&state.kind, 4)?;
                budget.node(4)?; // maximum_sequence
                budget.checked_shape(&state.shape)?;
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
            budget.enumeration(&op.operator, 4)?;
            budget.node(4)?; // layer (integer or null)
            budget.enumeration(&op.state_kind, 4)?;
            if op.inputs.len() > MAX_INPUTS {
                return Err("matrix inventory input count limit exceeded".into());
            }
            budget.node(4)?; // inputs array
            for input in &op.inputs {
                budget.string_value(input, 5)?;
            }
            budget.checked_shape(&op.output_shape)?;
            budget.dimensions(&op.output_shape, 4)?;
            if !op.attributes.is_object() {
                return Err("matrix inventory attributes must be an object".into());
            }
            if let Some(context) = context.as_deref_mut() {
                context.metadata(&mut budget, &op.attributes, 4)?;
            } else {
                budget.value(&op.attributes, 4)?;
            }
            if remote(op.operator) {
                budget.admit_work(op.inputs.len() as u128 + 2)?;
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
    // Preflight above already applied the exact legacy schema/node/string caps.
    // Canonical and serde JSON have identical encoded lengths. Do not involve
    // serde_json's allocating Error/Io error construction in this typed path.
    if let Some(context) = context {
        context.settle_schema(&mut budget)?;
        return context.preflight_bytes(plan);
    }
    let mut frames = [StreamFrame::empty(); CANONICAL_SLOTS];
    let mut limits = plan_canonical_limits(max_canonical_work);
    limits.canonical.max_nodes = usize::MAX;
    limits.canonical.max_total_string_bytes = usize::MAX;
    model_canonical::bounded_plan_write(plan, limits, &mut frames, &mut io::sink())
        .map(|_| ())
        .map_err(|error| match error {
            model_canonical::Error::Canonical(StreamError::ByteLimit) => {
                InventoryFailure::Static("metadata byte limit exceeded")
            }
            other => InventoryFailure::Canonical(other),
        })
}

const PUBLIC_INPUTS: [&str; 4] = [
    "input.tokens",
    "input.positions",
    "input.attention_mask",
    "input.sequence_lengths",
];

/// Validate reference uniqueness/topological binding without family name rules.
#[cfg(test)]
fn validate_references(graph: &DecoderGraph) -> Result<(), InventoryFailure<'static>> {
    for (index, state) in graph.state_inputs.iter().enumerate() {
        if state.id.is_empty()
            || PUBLIC_INPUTS.contains(&state.id.as_str())
            || graph.state_inputs[..index].iter().any(|previous| {
                previous.id == state.id
                    || (previous.layer, previous.kind) == (state.layer, state.kind)
            })
        {
            return Err("matrix inventory duplicate or empty state input ID/slot".into());
        }
    }
    for (index, op) in graph.operations.iter().enumerate() {
        let available = |id: &str| {
            PUBLIC_INPUTS.contains(&id)
                || graph.state_inputs.iter().any(|state| state.id == id)
                || graph.operations[..index]
                    .iter()
                    .any(|previous| previous.id == id)
        };
        if op.id.is_empty() || available(&op.id) {
            return Err("matrix inventory duplicate or empty operation ID".into());
        }
        if op.inputs.iter().any(|id| !available(id)) {
            return Err("matrix inventory unbound operation input ID".into());
        }
    }
    for (index, state) in graph.state_outputs.iter().enumerate() {
        let producer = graph.operations.iter().find(|op| op.id == state.id);
        if graph.state_outputs[..index].iter().any(|previous| {
            (previous.layer, previous.kind) == (state.layer, state.kind) || previous.id == state.id
        }) || producer.is_none_or(|op| {
            op.output_shape != state.shape
                || op.layer != state.layer
                || op.state_kind != Some(state.kind)
        }) {
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
    field: &'static str,
    optional: bool,
) -> Result<Option<&'a str>, InventoryFailure<'a>> {
    match op.attributes.get(field) {
        None | Some(Value::Null) if optional => Ok(None),
        Some(Value::String(value)) if !value.is_empty() => Ok(Some(value)),
        _ => Err(InventoryFailure::Artifact(field)),
    }
}

fn declared_shape(
    op: &ModelOperation,
    field: &'static str,
    expected: Option<&[u64]>,
) -> Result<(), InventoryFailure<'static>> {
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
            return Err(InventoryFailure::DeclaredShape(field));
        }
    }
    Ok(())
}

fn validate_attributes(op: &ModelOperation) -> Result<(), InventoryFailure<'_>> {
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
            return Err(InventoryFailure::Attribute(key));
        }
    }
    Ok(())
}

fn lower_phase<'a>(
    graph: &'a DecoderGraph,
    sources: &mut impl construction::Bindings<'a>,
    lookup: &construction::PhaseLookup<'_, 'a>,
    context: &mut construction::Context<'a>,
) -> Result<Phase<'a>, InventoryFailure<'a>> {
    lookup.validate(graph, context)?;
    context.charge(graph.operations.len() as u128)?;
    let mut head = None;
    let mut head_count = 0;
    for operation in &graph.operations {
        if operation.operator == ModelOperator::OutputHead {
            head_count += 1;
            if head.is_none() {
                head = Some(operation);
            }
        }
    }
    if head.is_none_or(|head| head.output_shape.len() != 2 || head.output_shape[0] != graph.batch)
        || head_count != 1
    {
        return Err("matrix inventory requires one rank-two output head".into());
    }
    let head = head.expect("checked exactly one head");
    let vocabulary = head.output_shape[1];
    if let Some(tied) = head.attributes.get("tied").and_then(Value::as_bool) {
        let mut shares_table = false;
        for op in &graph.operations {
            context.step()?;
            if op.operator == ModelOperator::TokenLookup
                && context.equal(op.attributes.get("weight"), head.attributes.get("weight"))?
            {
                shares_table = true;
                break;
            }
        }
        if tied != shares_table {
            return Err(
                "matrix inventory head tied declaration differs from source ownership".into(),
            );
        }
    }
    context.charge(graph.operations.len() as u128)?;
    for operation in &graph.operations {
        context.charge(operation.inputs.len() as u128)?;
    }
    let quote = RemoteGroupingQuote::checked(graph).map_err(InventoryFailure::Grouping)?;
    let resources = quote.resources();
    context.charge(resources.work_bound)?;
    context.factory()?;
    let leaders = quote
        .construct(resources.workspace_bytes, resources.work_bound)
        .map_err(InventoryFailure::Grouping)?;
    let mut groups = context.buffer(leaders.len())?;
    let mut group_count = 0;
    for (operation_index, op) in graph.operations.iter().enumerate() {
        context.step()?;
        let Some(group) = leaders.at(operation_index) else {
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
            InputShape::Tokens([graph.batch, graph.query_sequence])
        } else {
            InputShape::Produced(
                &lookup
                    .operation(&op.inputs[0], false, context)?
                    .ok_or("matrix inventory linear input has no operation source shape")?
                    .output_shape,
            )
        };
        let mut offset = 0_u64;
        let mut members = context.buffer(group.operations.len())?;
        for member in group.operations.iter() {
            context.step()?;
            context.charge(
                member
                    .attributes
                    .as_object()
                    .map_or(0, |fields| fields.len()) as u128,
            )?;
            if let Some(fields) = member.attributes.as_object() {
                for (key, value) in fields {
                    context
                        .charge(key.len() as u128 + value.as_str().map_or(0, str::len) as u128)?;
                }
            }
            for (a, b) in member.inputs.iter().zip(&op.inputs) {
                context.charge((a.len().min(b.len()) + 1) as u128)?;
            }
            validate_attributes(member)?;
            if member.inputs != op.inputs
                || member.operator != op.operator
                || member.layer != op.layer
                || member.state_kind.is_some()
            {
                return Err("matrix inventory group semantic inputs/layer/operator differ".into());
            }
            let output = &member.output_shape;
            context.charge((input_shape.len() + output.len()) as u128)?;
            let width = *output
                .last()
                .ok_or("matrix inventory missing output width")?;
            let source_shape = if token {
                if output.len() != 3 || output[..2] != *input_shape {
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
            context.charge(source_shape.len() as u128)?;
            declared_shape(member, "weight_shape", Some(&source_shape))?;
            let bias_shape = bias_id.map(|_| [width]);
            context.charge(bias_shape.map_or(0, |shape| shape.len()) as u128)?;
            declared_shape(
                member,
                "bias_shape",
                bias_shape.as_ref().map(|shape| shape.as_slice()),
            )?;
            construction::Bindings::bind(
                sources,
                weight_id,
                &source_shape,
                graph,
                lookup,
                context,
            )?;
            if let Some(bias_id) = bias_id {
                construction::Bindings::bind(sources, bias_id, &[width], graph, lookup, context)?;
            }
            members
                .push(Member {
                    operation_id: &member.id,
                    weight_id,
                    bias_id,
                    source_shape,
                    bias_shape,
                    output_shape: output,
                    output_offset: offset,
                    output_width: width,
                })
                .map_err(InventoryFailure::Index)?;
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
        context.charge(logical_shape.len() as u128 + logical_shape.len() as u128)?;
        shape_elements(&logical_shape)?;
        groups
            .push(Group {
                order: group_count,
                operator: op.operator,
                layer: op.layer,
                input_id: &op.inputs[0],
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
                members: members.finish().map_err(InventoryFailure::Index)?,
            })
            .map_err(InventoryFailure::Index)?;
        group_count += 1;
    }
    Ok(Phase {
        mode: graph.mode,
        batch: graph.batch,
        query_sequence: graph.query_sequence,
        maximum_key_sequence: graph.maximum_key_sequence,
        groups: groups.finish().map_err(InventoryFailure::Index)?,
    })
}

fn compatible<'a>(
    prefill: &Phase<'_>,
    decode: &Phase<'_>,
    plan: &'a DecoderPlan,
    lookup: &construction::PhaseLookup<'_, 'a>,
    context: &mut construction::Context<'a>,
) -> Result<(), InventoryFailure<'a>> {
    if prefill.groups.len() != decode.groups.len() {
        return Err("matrix inventory incompatible phase groups".into());
    }
    for op in &plan.prefill.operations {
        context.step()?;
        if remote(op.operator) {
            let other = lookup.operation(&op.id, true, context)?;
            if other.is_none()
                || !context.equal(Some(&op.attributes), other.map(|other| &other.attributes))?
            {
                return Err("matrix inventory incompatible phase matrix attributes".into());
            }
        }
    }
    for (a, b) in prefill.groups.iter().zip(decode.groups.iter()) {
        context.step()?;
        context.charge(a.logical_shape.len() as u128)?;
        context.charge((a.input_id.len().min(b.input_id.len()) + 1) as u128)?;
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
        for (a, b) in a.members.iter().zip(b.members.iter()) {
            context.step()?;
            context.charge(
                (a.source_shape.len() + a.bias_shape.map_or(0, |shape| shape.len())) as u128,
            )?;
            context.charge(
                (a.operation_id.len().min(b.operation_id.len())
                    + a.weight_id.len().min(b.weight_id.len())
                    + 2) as u128,
            )?;
            if let (Some(a), Some(b)) = (a.bias_id, b.bias_id) {
                context.charge((a.len().min(b.len()) + 1) as u128)?;
            }
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

fn lower_inventory(plan: &DecoderPlan) -> Result<Inventory<'_>, InventoryFailure<'_>> {
    preflight(plan)?;
    construction::metadata(plan, &mut construction::Context::new(false, u128::MAX)?)
}

fn inventory_stream_limits() -> StreamLimits {
    StreamLimits {
        // Legacy byte admission remains authoritative. Every value/string is
        // at least one encoded byte; the inventory schema has fixed depth.
        max_nodes: MAX_OUTPUT_BYTES,
        max_depth: CANONICAL_SLOTS,
        max_bytes: MAX_OUTPUT_BYTES,
        max_string_bytes: MAX_OUTPUT_BYTES,
        max_key_bytes: MAX_OUTPUT_BYTES,
        max_total_string_bytes: MAX_OUTPUT_BYTES,
        max_work: usize::MAX,
        max_stack_bytes: std::mem::size_of::<[StreamFrame<'_>; CANONICAL_SLOTS]>(),
    }
}

struct InventoryCursors<'a> {
    groups: std::slice::Iter<'a, Group<'a>>,
    members: std::slice::Iter<'a, Member<'a>>,
    dimensions: std::slice::Iter<'a, u64>,
}

impl InventoryCursors<'_> {
    fn empty() -> Self {
        Self {
            groups: [].iter(),
            members: [].iter(),
            dimensions: [].iter(),
        }
    }
}

fn emit_shape<'a, W: Write>(
    s: &mut Session<'_, '_, W>,
    shape: &'a [u64],
    cursors: &mut InventoryCursors<'a>,
) -> Result<(), StreamError> {
    s.begin_array()?;
    cursors.dimensions = shape.iter();
    for dimension in cursors.dimensions.by_ref() {
        s.u64(*dimension)?;
    }
    s.end_array()
}

fn emit_layer<W: Write>(s: &mut Session<'_, '_, W>, layer: Option<u64>) -> Result<(), StreamError> {
    match layer {
        Some(value) => s.u64(value),
        None => s.null(),
    }
}

fn emit_inventory_phase<'a, W: Write>(
    s: &mut Session<'_, '_, W>,
    phase: &'a Phase,
    cursors: &mut InventoryCursors<'a>,
) -> Result<(), StreamError> {
    s.begin_object()?;
    s.key("batch")?;
    s.u64(phase.batch)?;
    s.key("groups")?;
    s.begin_array()?;
    cursors.groups = phase.groups.iter();
    while let Some(group) = cursors.groups.next() {
        s.begin_object()?;
        s.key("artifact_kind")?;
        s.string(match group.artifact_kind {
            ArtifactKind::TokenTable => "token_table",
            ArtifactKind::Matrix => "matrix",
        })?;
        s.key("input_id")?;
        s.string(group.input_id)?;
        s.key("input_shape")?;
        emit_shape(s, &group.input_shape, cursors)?;
        s.key("layer")?;
        emit_layer(s, group.layer)?;
        s.key("logical_shape")?;
        emit_shape(s, &group.logical_shape, cursors)?;
        s.key("members")?;
        s.begin_array()?;
        cursors.members = group.members.iter();
        while let Some(member) = cursors.members.next() {
            s.begin_object()?;
            s.key("bias_id")?;
            match &member.bias_id {
                Some(id) => s.string(id)?,
                None => s.null()?,
            }
            s.key("bias_shape")?;
            match &member.bias_shape {
                Some(shape) => emit_shape(s, shape, cursors)?,
                None => s.null()?,
            }
            s.key("operation_id")?;
            s.string(member.operation_id)?;
            s.key("output_offset")?;
            s.u64(member.output_offset)?;
            s.key("output_shape")?;
            emit_shape(s, member.output_shape, cursors)?;
            s.key("output_width")?;
            s.u64(member.output_width)?;
            s.key("source_shape")?;
            emit_shape(s, &member.source_shape, cursors)?;
            s.key("weight_id")?;
            s.string(member.weight_id)?;
            s.end_object()?;
        }
        s.end_array()?;
        s.key("operator")?;
        s.string(match group.operator {
            ModelOperator::TokenLookup => "token_lookup",
            ModelOperator::Linear => "linear",
            ModelOperator::OutputHead => "output_head",
            _ => unreachable!("validated remote operator"),
        })?;
        s.key("order")?;
        s.u64(group.order)?;
        s.key("orientation")?;
        s.string(match group.orientation {
            Orientation::VocabularyFeature => "vocabulary_feature",
            Orientation::OutputInput => "output_input",
        })?;
        s.end_object()?;
    }
    s.end_array()?;
    s.key("maximum_key_sequence")?;
    s.u64(phase.maximum_key_sequence)?;
    s.key("mode")?;
    s.string(match phase.mode {
        DecoderMode::Prefill => "prefill",
        DecoderMode::Decode => "decode",
    })?;
    s.key("query_sequence")?;
    s.u64(phase.query_sequence)?;
    s.end_object()
}

fn emit_inventory<'a, W: Write>(
    s: &mut Session<'_, '_, W>,
    inventory: &'a Inventory,
    cursors: &mut InventoryCursors<'a>,
) -> Result<(), StreamError> {
    s.begin_object()?;
    s.key("decode")?;
    emit_inventory_phase(s, &inventory.decode, cursors)?;
    s.key("model_config_digest")?;
    s.string(inventory.model_config_digest.as_str())?;
    s.key("model_plan_digest")?;
    s.string(inventory.model_plan_digest.as_str())?;
    s.key("prefill")?;
    emit_inventory_phase(s, &inventory.prefill, cursors)?;
    s.key("schema_version")?;
    s.string(inventory.schema_version)?;
    s.end_object()
}

fn inventory_pass<W: Write>(
    inventory: &Inventory,
    frames: &mut [StreamFrame<'_>],
    output: &mut W,
) -> Result<pllm_types::bounded_canonical::StreamUsage, StreamError> {
    inventory_pass_with_limits(inventory, frames, output, inventory_stream_limits())
}

fn inventory_pass_with_limits<W: Write>(
    inventory: &Inventory,
    frames: &mut [StreamFrame<'_>],
    output: &mut W,
    limits: StreamLimits,
) -> Result<pllm_types::bounded_canonical::StreamUsage, StreamError> {
    let mut session = Session::new(output, frames, limits)?;
    let mut cursors = InventoryCursors::empty();
    emit_inventory(&mut session, inventory, &mut cursors)?;
    session.finish()
}

/// Exact-capacity final backing; sink admission precedes the only output factory.
/// No temporary JSON tree, key-sort vector, escaped String or growing writer.
fn inventory_canonical_bytes(
    inventory: &Inventory<'_>,
) -> Result<Vec<u8>, InventoryFailure<'static>> {
    let mut frames = [StreamFrame::empty(); CANONICAL_SLOTS];
    let usage =
        inventory_pass(inventory, &mut frames, &mut io::sink()).map_err(|error| match error {
            StreamError::ByteLimit => InventoryFailure::Static("metadata byte limit exceeded"),
            other => InventoryFailure::Stream(other),
        })?;
    // A boxed slice has an exact logical length, independent of allocator
    // rounding or Vec growth policy. Initialize in place; no temporary Vec.
    let mut storage = Box::<[u8]>::new_uninit_slice(usage.bytes);
    for byte in storage.iter_mut() {
        byte.write(0);
    }
    // SAFETY: every slot above was initialized, including the empty-slice case.
    let mut bytes = unsafe { storage.assume_init() };
    let mut output = io::Cursor::new(bytes.as_mut());
    inventory_pass(inventory, &mut frames, &mut output).map_err(InventoryFailure::Stream)?;
    if output.position() != usage.bytes as u64 {
        return Err("matrix inventory canonical replay length mismatch".into());
    }
    // Transfers the exact-length backing; no copy/reallocation.
    Ok(bytes.into_vec())
}

/// Derive a canonical, plan/config-bound snapshot without an executable composition.
/// Only this constructor can create an inventory; all validation precedes hashing.
pub fn lower_decoder_matrix_inventory(
    plan: &DecoderPlan,
) -> Result<DecoderMatrixInventory, String> {
    let inventory = lower_inventory(plan).map_err(|error| error.to_string())?;
    let bytes = inventory_canonical_bytes(&inventory).map_err(|error| error.to_string())?;
    let mut hash = DomainHash::new(DECODER_MATRIX_INVENTORY_SCHEMA_VERSION);
    hash.write_all(&bytes).expect("infallible shared hash sink");
    let digest = HexDigest::new(hash.finish()).into_digest();
    Ok(DecoderMatrixInventory {
        bytes,
        digest,
        semantic_plan_identity: inventory.model_plan_digest.raw,
        config_identity: inventory.model_config_digest.as_sha256(),
    })
}

#[cfg(test)]
#[path = "decoder_matrix_inventory_preconstruction_tests.rs"]
mod preconstruction_tests;
