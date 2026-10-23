//! Bounded D22 source-byte import and conditional interval proof.
//!
//! Reuses model-neutral semantic IR/inventory, edge contracts and the unchanged
//! D20 activation declaration. This module does not resolve checkpoints or
//! compile complete decoders. Actual source imports check bounded bytes
//! against an independently trusted publisher pin, not a user callback. A pin
//! is a publisher trust assumption, not cryptographic publisher authentication.
//!
//! Complete selected fixture inner dimensions are supported. Any coordinate
//! subset is explicitly labeled a checkpoint-derived submatrix, not execution
//! of the original operation. Selected coordinates remain bound to real IR.
//! Current semantic IR has no exact GELU operator: `GeluTanh` is rejected, not
//! silently substituted. The fixed D20 GELU implementation itself is unchanged.

use crate::decoder_matrix_inventory::{lower_decoder_matrix_inventory, DecoderMatrixInventory};
use crate::verified_source::{sum_bytes, ZeroizingFixed};
use pllm_core::edge_contract::{
    ConsumerUse, EdgeContractWorkspace, EdgeDescriptor, Interpretation, Numeric, Representation,
    SourceIdentity, WeightVisibility,
};
use pllm_garble::masked_wire_reference::typed::{
    ActivationKind, ActivationMetadataWorkspace, ActivationSpec,
};
use pllm_models::bounded_canonical as model_canonical;
use pllm_models::{DecoderMode, DecoderPlan, ModelOperation, ModelOperator};
use pllm_types::bounded_canonical::{DomainHash, Session, StreamFrame, StreamLimits};
use pllm_types::canonical_bytes;
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::io::{self, Write};
use std::mem::size_of;

mod source;
// Bounded local execution retains its checked source and original accounting
// guard. This is not decoder admission or stronger native provenance.
#[path = "sigma_linear_activation/execution.rs"]
mod execution;
pub use execution::{
    execute_owned_sigma_linear_activation, execute_sigma_linear_activation,
    owned_sigma_execution_compiler_preflight, owned_sigma_execution_role_split_resources,
    sigma_execution_compiler_preflight, sigma_execution_role_split_resources,
    OwnedSigmaExecutionFrame, OwnedSigmaExecutionMask, OwnedSigmaExecutionOutput,
    OwnedSigmaExecutionRole, OwnedSigmaExecutionRoleCoordinator, OwnedSigmaExecutionRoleFrame,
    OwnedSigmaExecutionRoleOutput, OwnedSigmaExecutionRoleStep, OwnedSigmaExecutionRun,
    OwnedSigmaExecutionStep, SigmaExecutionContext, SigmaExecutionError, SigmaExecutionFrame,
    SigmaExecutionFrameFor, SigmaExecutionMask, SigmaExecutionMaskFor, SigmaExecutionOutput,
    SigmaExecutionOutputFor, SigmaExecutionProvenance, SigmaExecutionRole,
    SigmaExecutionRoleCoordinator, SigmaExecutionRoleCoordinatorFor, SigmaExecutionRoleFor,
    SigmaExecutionRoleFrame, SigmaExecutionRoleFrameFor, SigmaExecutionRoleOutput,
    SigmaExecutionRoleOutputFor, SigmaExecutionRoleSplitResources, SigmaExecutionRoleStep,
    SigmaExecutionRoleStepFor, SigmaExecutionRun, SigmaExecutionRunFor, SigmaExecutionStep,
    SigmaExecutionStepFor,
};
pub use source::{
    certify_source_checked_sigma_linear_activation, import_sigma_source_bytes, CheckedSigmaSource,
    ExpectedSigmaSourcePin, SigmaSourceBytes, SigmaSourceConstructionResources,
    SigmaSourceResources, SourceCheckedSigmaLinearActivation, SourceDtype, SourceEndian,
};

pub const SIGMA_LINEAR_ACTIVATION_SCHEMA: &str = "pllm.sigma.linear-activation-certificate.v1";
pub const SIGMA_LINEAR_ACTIVATION_ENCODING: &str = "signed-ring50-q12-floor-f64.v1";
const MAX_ELEMENTS: usize = 256;
const MAX_MACS: usize = 1 << 20;
const RING_HALF: i128 = 1 << 49;
const SUM_BOUND: i128 = 1 << 48;
const RESCALED_BOUND: i128 = 1 << 36;

/// Inclusive integer endpoints. Products and sums use checked i128 arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct IntegerInterval {
    minimum: i128,
    maximum: i128,
}

impl IntegerInterval {
    /// Half-open public/native contract, converted once to inclusive endpoints.
    pub fn from_half_open(lower: i128, upper: i128) -> Result<Self, String> {
        if lower >= upper {
            return Err("empty or reversed half-open interval".into());
        }
        Self::new(
            lower,
            upper.checked_sub(1).ok_or("interval endpoint overflow")?,
        )
    }
    pub fn half_open_endpoints(self) -> Result<(i128, i128), String> {
        self.checked_half_open_endpoints().map_err(String::from)
    }

    fn checked_half_open_endpoints(self) -> Result<(i128, i128), &'static str> {
        Ok((
            self.minimum,
            self.maximum
                .checked_add(1)
                .ok_or("interval endpoint overflow")?,
        ))
    }
    pub fn new(minimum: i128, maximum: i128) -> Result<Self, String> {
        Self::checked_new(minimum, maximum).map_err(String::from)
    }

    fn checked_new(minimum: i128, maximum: i128) -> Result<Self, &'static str> {
        if minimum > maximum {
            return Err("reversed integer interval");
        }
        Ok(Self { minimum, maximum })
    }

    pub fn endpoints(self) -> (i128, i128) {
        (self.minimum, self.maximum)
    }

    fn signed_ring50(self) -> Result<(), &'static str> {
        if self.minimum < -RING_HALF || self.maximum >= RING_HALF {
            return Err("operand interval outside signed ring50");
        }
        Ok(())
    }
}

/// Exact checkpoint declaration. Source-checked construction compares all fields
/// against the independently trusted expected pin; a declaration hash alone is
/// not evidence of byte provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CheckpointTensorSelection {
    pub checkpoint: String,
    pub immutable_revision: String,
    pub content_sha256: [u8; 32],
    pub tensor: String,
    /// Inventory convention: original weights are [output, input], row-major.
    pub original_weight_shape: [usize; 2],
    pub weight_layout: String,
    /// Ordered output rows. Every selected inner coordinate must be contracted.
    pub output_coordinates: Vec<usize>,
    pub inner_coordinates: Vec<usize>,
    pub input_tensor: String,
    /// Flattened prefix rows and full inner dimension of the semantic input.
    pub original_input_shape: [usize; 2],
    pub input_layout: String,
    pub input_row_coordinates: Vec<usize>,
    pub encoding: String,
}

impl CheckpointTensorSelection {
    pub fn metadata_backing_bytes(&self) -> Result<usize, String> {
        self.checked_metadata_backing_bytes().map_err(String::from)
    }

    #[doc(hidden)]
    pub fn checked_metadata_backing_bytes(&self) -> Result<usize, &'static str> {
        let coordinate_capacity = self
            .output_coordinates
            .capacity()
            .checked_add(self.inner_coordinates.capacity())
            .and_then(|n| n.checked_add(self.input_row_coordinates.capacity()))
            .and_then(|n| n.checked_mul(size_of::<usize>()))
            .ok_or("selection coordinate capacity overflow")?;
        [
            self.checkpoint.capacity(),
            self.immutable_revision.capacity(),
            self.tensor.capacity(),
            self.weight_layout.capacity(),
            self.input_tensor.capacity(),
            self.input_layout.capacity(),
            self.encoding.capacity(),
            coordinate_capacity,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or("selection metadata capacity overflow")
        })
    }
}

/// One fixed logical allocation; source-checked construction moves this owner
/// into the certificate instead of duplicating a coefficient Vec.
struct SigmaCoefficients {
    values: ZeroizingFixed<i64, MAX_ELEMENTS>,
    len: usize,
}
impl SigmaCoefficients {
    fn new(values: &[i64]) -> Result<Self, &'static str> {
        if values.is_empty() || values.len() > MAX_ELEMENTS {
            return Err("coefficient capacity exceeded");
        }
        let mut result = Self {
            values: ZeroizingFixed::new_zeroed(),
            len: values.len(),
        };
        result.values[..values.len()].copy_from_slice(values);
        Ok(result)
    }
    fn as_slice(&self) -> &[i64] {
        &self.values[..self.len]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixtureScope {
    CompleteOperation,
    CheckpointDerivedSubmatrix,
}

/// Declaration-only path cannot produce a checked source owner.
pub enum SourceImport {
    DeclarationOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceImportStatus {
    UnauthenticatedDeclaration,
    PinnedSourceBytesChecked,
}

/// Complete input contract. Edge operation identities use
/// [`sigma_semantic_operation_identity`]; edge plan identities use
/// [`sigma_semantic_plan_identity`]. First tensor indices are zero: arbitrary
/// coordinate selection is bound separately, never disguised as contiguous data.
pub struct SigmaLinearActivationRequest<'a> {
    pub plan: &'a DecoderPlan,
    pub mode: DecoderMode,
    pub linear_operation: &'a str,
    pub activation_operation: &'a str,
    pub next_operation: &'a str,
    pub selection: CheckpointTensorSelection,
    pub source: SourceIdentity,
    pub input: EdgeDescriptor,
    pub private_weights: EdgeDescriptor,
    pub complete_sum: EdgeDescriptor,
    pub rescaled: EdgeDescriptor,
    pub activation_output: EdgeDescriptor,
    pub rescale_operation: [u8; 32],
    pub next_use: ConsumerUse,
    pub input_domain: IntegerInterval,
    pub source_coefficients: &'a [f64],
    pub encoded_private_coefficients: &'a [i64],
}

/// Non-deserializable, immutable conditional proof. No Clone/Debug to casually
/// duplicate or print private coefficients. Coefficients are erased on drop.
/// This object is not an owner of native matrix material or an admission lease.
pub struct SigmaLinearActivationCertificate {
    digest: [u8; 32],
    selection: CheckpointTensorSelection,
    scope: FixtureScope,
    source_status: SourceImportStatus,
    input_domain: IntegerInterval,
    weight_domain: IntegerInterval,
    sum_domain: IntegerInterval,
    rescaled_domain: IntegerInterval,
    coefficients: SigmaCoefficients,
    activation: ActivationSpec,
    next_use: ConsumerUse,
    macs: usize,
    edges: [EdgeDescriptor; 5],
    mode: DecoderMode,
}

impl SigmaLinearActivationCertificate {
    pub fn input_edge(&self) -> EdgeDescriptor {
        self.edges[0]
    }
    pub fn private_weight_edge(&self) -> EdgeDescriptor {
        self.edges[1]
    }
    pub fn complete_sum_edge(&self) -> EdgeDescriptor {
        self.edges[2]
    }
    pub fn rescaled_edge(&self) -> EdgeDescriptor {
        self.edges[3]
    }
    pub fn activation_output_edge(&self) -> EdgeDescriptor {
        self.edges[4]
    }
    pub fn mode(&self) -> DecoderMode {
        self.mode
    }
    /// Confidential commitment: low-entropy private weights may be guessable.
    /// Do not publish this digest as an opaque hiding commitment.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn selection(&self) -> &CheckpointTensorSelection {
        &self.selection
    }
    pub fn fixture_scope(&self) -> FixtureScope {
        self.scope
    }
    pub fn source_import_status(&self) -> SourceImportStatus {
        self.source_status
    }
    pub fn input_domain(&self) -> IntegerInterval {
        self.input_domain
    }
    pub fn weight_domain(&self) -> IntegerInterval {
        self.weight_domain
    }
    pub fn sum_domain(&self) -> IntegerInterval {
        self.sum_domain
    }
    pub fn rescaled_domain(&self) -> IntegerInterval {
        self.rescaled_domain
    }
    pub fn encoded_private_coefficients(&self) -> &[i64] {
        self.coefficients.as_slice()
    }
    pub fn activation_spec(&self) -> ActivationSpec {
        self.activation
    }
    pub fn next_use(&self) -> ConsumerUse {
        self.next_use
    }
    pub fn multiply_accumulates(&self) -> usize {
        self.macs
    }

    /// Exact retained logical shape/capacities; no allocator headers or RSS.
    pub fn retained_resource_bytes(&self) -> Result<usize, String> {
        sum_bytes([
            size_of::<Self>(),
            self.selection.metadata_backing_bytes()?,
            size_of::<[i64; MAX_ELEMENTS]>(),
        ])
    }

    /// No existing API authenticates source imports or transfers/reserves the
    /// original native owners. Metadata must not mint those capabilities.
    pub fn native_elevation(&self) -> Result<(), &'static str> {
        Err(
            "blocked: consuming original-owner admission is not wired; declaration metadata grants no native capabilities",
        )
    }
}

pub fn sigma_semantic_operation_identity(operation: &ModelOperation) -> [u8; 32] {
    hash(b"pllm.semantic-operation.v1\0", &canonical_bytes(operation))
}

pub fn sigma_semantic_plan_identity(plan: &DecoderPlan, mode: DecoderMode) -> [u8; 32] {
    hash(
        b"pllm.semantic-plan-reference.v1\0",
        &canonical_bytes(&(plan.digest(), mode)),
    )
}

/// Fixed synthetic stage, not an arbitrary truncation operation supplied by a caller.
pub fn sigma_gapars12_operation_identity(linear: &ModelOperation) -> [u8; 32] {
    hash(
        b"pllm.complete-dot-gapars12-floor-once.v1\0",
        &sigma_semantic_operation_identity(linear),
    )
}

fn hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(bytes);
    h.finalize().into()
}

// These are traversal rejection limits, NOT an aggregate compiler storage quote.
// Construction of the borrowed plan/inventory is outside the minimal boundary.
const SIGMA_STREAM_DEPTH: usize = 32;

/// Real caller-owned frame storage, shared by sink/replay within each bounded
/// hash. A resource inventory can price this object without inventing a stack
/// allowance. Session/model admission and enclosing proof state remain separate.
struct SigmaStreamWorkspace<'a> {
    frames: [StreamFrame<'a>; SIGMA_STREAM_DEPTH],
}
impl SigmaStreamWorkspace<'_> {
    fn new() -> Self {
        Self {
            frames: [StreamFrame::empty(); SIGMA_STREAM_DEPTH],
        }
    }
}
fn sigma_stream_limits() -> StreamLimits {
    StreamLimits {
        max_nodes: 131072,
        max_depth: SIGMA_STREAM_DEPTH,
        max_bytes: 8 << 20,
        max_string_bytes: 1024,
        max_key_bytes: 1024,
        max_total_string_bytes: 1 << 20,
        max_work: 1 << 28,
        max_stack_bytes: size_of::<[StreamFrame<'static>; SIGMA_STREAM_DEPTH]>(),
    }
}
fn sigma_model_limits() -> model_canonical::Limits {
    model_canonical::Limits {
        canonical: sigma_stream_limits(),
        max_operations: 4096,
        max_states: 1024,
        max_inputs: 131072,
        max_rank: 8,
        max_transformations: 4096,
        max_domain_bytes: 128,
    }
}

/// One immutable inventory and the SAME borrowed plan for whole certification.
/// This is pairing validation, not resource admission or source authentication.
struct SigmaCertificationContext<'a> {
    plan: &'a DecoderPlan,
    inventory: &'a DecoderMatrixInventory,
    plan_identity: [u8; 32],
    plan_hex: [u8; 64],
}
impl<'a> SigmaCertificationContext<'a> {
    fn new(
        plan: &'a DecoderPlan,
        inventory: &'a DecoderMatrixInventory,
    ) -> Result<Self, &'static str> {
        let mut workspace = SigmaStreamWorkspace::new();
        let plan_identity =
            model_canonical::bounded_plan_digest(plan, sigma_model_limits(), &mut workspace.frames)
                .map_err(|_| "bounded semantic plan hashing rejected")?;
        if inventory.semantic_plan_identity() != plan_identity
            || inventory.config_identity() != plan.config_digest.as_sha256()
        {
            return Err("preexisting inventory differs from borrowed semantic plan/config");
        }
        let mut plan_hex = [0; 64];
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for (i, byte) in plan_identity.iter().copied().enumerate() {
            plan_hex[2 * i] = HEX[usize::from(byte >> 4)];
            plan_hex[2 * i + 1] = HEX[usize::from(byte & 15)];
        }
        Ok(Self {
            plan,
            inventory,
            plan_identity,
            plan_hex,
        })
    }

    fn phase_identity(&self, mode: DecoderMode) -> Result<[u8; 32], &'static str> {
        let hex = std::str::from_utf8(&self.plan_hex).map_err(|_| "invalid digest hex")?;
        let mut hash = DomainHash::new("pllm.semantic-plan-reference.v1");
        bounded_schema_write(SigmaSchema::PlanReference(hex, mode), &mut hash)?;
        Ok(hash.finish())
    }
}

fn bounded_operation_identity(operation: &ModelOperation) -> Result<[u8; 32], &'static str> {
    let mut workspace = SigmaStreamWorkspace::new();
    model_canonical::bounded_model_operation_digest(
        operation,
        "pllm.semantic-operation.v1",
        sigma_model_limits(),
        &mut workspace.frames,
    )
    .map_err(|_| "bounded semantic operation hashing rejected")
}

fn bounded_selection_identity(
    selection: &CheckpointTensorSelection,
) -> Result<[u8; 32], &'static str> {
    let mut hash = DomainHash::new("pllm.source-weight-selection.v1");
    bounded_schema_write(SigmaSchema::Selection(selection), &mut hash)?;
    Ok(hash.finish())
}

struct SigmaHashWriter<'a>(&'a mut Sha256);
impl Write for SigmaHashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum SigmaSchema<'a> {
    Selection(&'a CheckpointTensorSelection),
    PlanReference(&'a str, DecoderMode),
    Mode(DecoderMode),
    Intervals(&'a [IntegerInterval; 4]),
}

fn mode_name(mode: DecoderMode) -> &'static str {
    match mode {
        DecoderMode::Prefill => "prefill",
        DecoderMode::Decode => "decode",
    }
}

fn emit_usizes<W: Write>(
    stream: &mut Session<'_, '_, W>,
    values: &[usize],
) -> Result<(), pllm_types::bounded_canonical::StreamError> {
    use pllm_types::bounded_canonical::StreamError;
    stream.begin_array()?;
    for &value in values {
        stream.u64(u64::try_from(value).map_err(|_| StreamError::Structure)?)?;
    }
    stream.end_array()
}

fn emit_sigma_schema<'a, W: Write>(
    schema: SigmaSchema<'a>,
    stream: &mut Session<'a, '_, W>,
) -> Result<(), pllm_types::bounded_canonical::StreamError> {
    use pllm_types::bounded_canonical::StreamError;
    match schema {
        SigmaSchema::Mode(mode) => stream.string(mode_name(mode)),
        SigmaSchema::PlanReference(hex, mode) => {
            stream.begin_array()?;
            stream.string(hex)?;
            stream.string(mode_name(mode))?;
            stream.end_array()
        }
        SigmaSchema::Intervals(intervals) => {
            stream.begin_array()?;
            for interval in intervals {
                // Validated ring50 operands and bounded sums fit i64 exactly.
                // Legacy i128 JSON emits these same decimal integer bytes.
                stream.begin_object()?;
                stream.key("maximum")?;
                stream.i64(i64::try_from(interval.maximum).map_err(|_| StreamError::Structure)?)?;
                stream.key("minimum")?;
                stream.i64(i64::try_from(interval.minimum).map_err(|_| StreamError::Structure)?)?;
                stream.end_object()?;
            }
            stream.end_array()
        }
        SigmaSchema::Selection(s) => {
            stream.begin_object()?;
            stream.key("checkpoint")?;
            stream.string(&s.checkpoint)?;
            stream.key("content_sha256")?;
            stream.begin_array()?;
            for &byte in &s.content_sha256 {
                stream.u64(u64::from(byte))?;
            }
            stream.end_array()?;
            stream.key("encoding")?;
            stream.string(&s.encoding)?;
            stream.key("immutable_revision")?;
            stream.string(&s.immutable_revision)?;
            stream.key("inner_coordinates")?;
            emit_usizes(stream, &s.inner_coordinates)?;
            stream.key("input_layout")?;
            stream.string(&s.input_layout)?;
            stream.key("input_row_coordinates")?;
            emit_usizes(stream, &s.input_row_coordinates)?;
            stream.key("input_tensor")?;
            stream.string(&s.input_tensor)?;
            stream.key("original_input_shape")?;
            emit_usizes(stream, &s.original_input_shape)?;
            stream.key("original_weight_shape")?;
            emit_usizes(stream, &s.original_weight_shape)?;
            stream.key("output_coordinates")?;
            emit_usizes(stream, &s.output_coordinates)?;
            stream.key("tensor")?;
            stream.string(&s.tensor)?;
            stream.key("weight_layout")?;
            stream.string(&s.weight_layout)?;
            stream.end_object()
        }
    }
}

fn bounded_schema_write<W: Write>(
    schema: SigmaSchema<'_>,
    writer: &mut W,
) -> Result<(), &'static str> {
    // No descriptor tree, Vec, serde serializer or output allocation. Validate
    // immutable schema against sink before replaying to the digest writer.
    let mut workspace = SigmaStreamWorkspace::new();
    let mut sink = io::sink();
    let mut stream = Session::new(&mut sink, &mut workspace.frames, sigma_stream_limits())
        .map_err(|_| "bounded schema workspace rejected")?;
    emit_sigma_schema(schema, &mut stream).map_err(|_| "bounded schema admission rejected")?;
    stream
        .finish()
        .map_err(|_| "bounded schema admission incomplete")?;
    let mut stream = Session::new(writer, &mut workspace.frames, sigma_stream_limits())
        .map_err(|_| "bounded schema workspace rejected")?;
    emit_sigma_schema(schema, &mut stream).map_err(|_| "bounded schema replay rejected")?;
    stream
        .finish()
        .map_err(|_| "bounded schema replay incomplete")?;
    Ok(())
}

/// Exact binary64 source encoding: multiplication by 2^12 is exact whenever the
/// admitted result is representable. Reject before casting (no saturation).
pub fn sigma_floor_encode_q12(value: f64) -> Result<i64, String> {
    checked_floor_encode_q12(value).map_err(String::from)
}

fn checked_floor_encode_q12(value: f64) -> Result<i64, &'static str> {
    let scaled = value * 4096.0;
    if !value.is_finite()
        || !scaled.is_finite()
        || scaled < -(RING_HALF as f64)
        || scaled >= RING_HALF as f64
    {
        return Err("nonfinite or unsupported signed-ring50 floor encoding");
    }
    Ok(scaled.floor() as i64)
}

/// Endpoint products multiplied by the *complete* nonzero inner dimension.
/// No per-term floor, sampled extrema, partial sum or modular-wrap inference.
#[derive(Default)]
struct SigmaDotWorkspace {
    products: [i128; 4],
}

#[cfg(test)]
fn complete_dot_interval(
    input: IntegerInterval,
    weights: IntegerInterval,
    inner: usize,
) -> Result<(IntegerInterval, IntegerInterval), &'static str> {
    complete_dot_interval_with_workspace(input, weights, inner, &mut SigmaDotWorkspace::default())
}

fn complete_dot_interval_with_workspace(
    input: IntegerInterval,
    weights: IntegerInterval,
    inner: usize,
    workspace: &mut SigmaDotWorkspace,
) -> Result<(IntegerInterval, IntegerInterval), &'static str> {
    input.signed_ring50()?;
    weights.signed_ring50()?;
    if inner == 0 {
        return Err("empty complete inner dimension");
    }
    // No parallel owned endpoint-pair array or copied product vector.
    workspace.products[0] = input
        .minimum
        .checked_mul(weights.minimum)
        .ok_or("endpoint product overflow")?;
    workspace.products[1] = input
        .minimum
        .checked_mul(weights.maximum)
        .ok_or("endpoint product overflow")?;
    workspace.products[2] = input
        .maximum
        .checked_mul(weights.minimum)
        .ok_or("endpoint product overflow")?;
    workspace.products[3] = input
        .maximum
        .checked_mul(weights.maximum)
        .ok_or("endpoint product overflow")?;
    let inner = i128::try_from(inner).map_err(|_| "inner dimension overflow")?;
    let minimum = workspace
        .products
        .iter()
        .min()
        .unwrap()
        .checked_mul(inner)
        .ok_or("complete dot overflow")?;
    let maximum = workspace
        .products
        .iter()
        .max()
        .unwrap()
        .checked_mul(inner)
        .ok_or("complete dot overflow")?;
    let sum = IntegerInterval::checked_new(minimum, maximum)?;
    if minimum < -SUM_BOUND || maximum >= SUM_BOUND {
        return Err("complete sum outside [-2^48,2^48)");
    }
    let rescaled =
        IntegerInterval::checked_new(minimum.div_euclid(4096), maximum.div_euclid(4096))?;
    if rescaled.minimum < -RESCALED_BOUND || rescaled.maximum >= RESCALED_BOUND {
        return Err("rescaled interval outside [-2^36,2^36)");
    }
    Ok((sum, rescaled))
}

fn dimensions(shape: &[u64]) -> Result<[usize; 2], &'static str> {
    if !matches!(shape.len(), 2 | 3) {
        return Err("unsupported semantic matrix rank");
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1_u64, |n, d| n.checked_mul(*d))
        .ok_or("semantic prefix geometry overflow")?;
    let result = [
        usize::try_from(rows).map_err(|_| "row conversion overflow")?,
        usize::try_from(*shape.last().unwrap()).map_err(|_| "inner conversion overflow")?,
    ];
    if result.contains(&0) {
        return Err("empty semantic matrix");
    }
    result[0]
        .checked_mul(result[1])
        .ok_or("semantic geometry overflow")?;
    Ok(result)
}

fn coordinates(values: &[usize], extent: usize) -> Result<(), &'static str> {
    if values.is_empty()
        || values.len() > MAX_ELEMENTS
        || extent == 0
        || values
            .iter()
            .enumerate()
            .any(|(i, v)| *v >= extent || values[..i].contains(v))
    {
        return Err("invalid, duplicate or unbounded selected coordinates");
    }
    Ok(())
}

fn geometry(
    selection: &CheckpointTensorSelection,
) -> Result<([usize; 3], FixtureScope), &'static str> {
    if selection.checked_metadata_backing_bytes()? > 64 * 1024 {
        return Err("selected tensor metadata capacity limit exceeded");
    }
    for text in [
        &selection.checkpoint,
        &selection.immutable_revision,
        &selection.tensor,
        &selection.input_tensor,
        &selection.weight_layout,
        &selection.input_layout,
        &selection.encoding,
    ] {
        if text.is_empty() || text.len() > 1024 {
            return Err("invalid source identity string");
        }
    }
    if selection.weight_layout != "output_input_row_major"
        || selection.input_layout != "row_inner_row_major"
        || selection.encoding != SIGMA_LINEAR_ACTIVATION_ENCODING
    {
        return Err("unsupported source layout or floor encoding");
    }
    let [outputs, original_inner] = selection.original_weight_shape;
    let [rows, input_inner] = selection.original_input_shape;
    outputs
        .checked_mul(original_inner)
        .ok_or("original weight geometry overflow")?;
    rows.checked_mul(input_inner)
        .ok_or("original input geometry overflow")?;
    coordinates(&selection.output_coordinates, outputs)?;
    coordinates(&selection.input_row_coordinates, rows)?;
    coordinates(&selection.inner_coordinates, original_inner)?;
    if input_inner != original_inner {
        return Err("original input/weight inner dimensions differ");
    }
    let inner = selection.inner_coordinates.len();
    let m = selection.input_row_coordinates.len();
    let n = selection.output_coordinates.len();
    for elements in [m.checked_mul(inner), n.checked_mul(inner), m.checked_mul(n)] {
        if !matches!(elements, Some(1..=MAX_ELEMENTS)) {
            return Err("matrix exceeds 256 elements");
        }
    }
    let full = selection.output_coordinates.iter().copied().eq(0..outputs)
        && selection.input_row_coordinates.iter().copied().eq(0..rows)
        && selection
            .inner_coordinates
            .iter()
            .copied()
            .eq(0..original_inner);
    Ok((
        [m, inner, n],
        if full {
            FixtureScope::CompleteOperation
        } else {
            FixtureScope::CheckpointDerivedSubmatrix
        },
    ))
}

fn check_edge(
    edge: EdgeDescriptor,
    source: SourceIdentity,
    plan: [u8; 32],
    producer: [u8; 32],
    shape: [usize; 2],
    fraction: u8,
    private_weight: bool,
) -> Result<(), &'static str> {
    if edge.identity().source() != source
        || edge.identity().plan() != plan
        || edge.identity().producer() != producer
        || edge.identity().first_tensor_index() != 0
        || edge.shape().dimensions() != shape
        || edge.numeric()
            != Numeric::new(50, fraction, Interpretation::Signed).map_err(|_| "numeric domain")?
        || edge.representation() != Representation::CommonAdditiveMasked
        || edge.weight_visibility() != private_weight.then_some(WeightVisibility::Private)
    {
        return Err("edge source/plan/producer/geometry/numeric/private-layout mismatch");
    }
    Ok(())
}

/// Checks declarations, independent encoding and complete-dot numeric proof.
/// Caller-supplied source values alone do not establish checkpoint provenance.
pub fn certify_sigma_linear_activation(
    request: SigmaLinearActivationRequest<'_>,
    import: SourceImport,
) -> Result<SigmaLinearActivationCertificate, String> {
    // Legacy declaration construction includes raw lowering. It is outside the
    // proposed admitted execution boundary, which borrows an existing inventory.
    let inventory = lower_decoder_matrix_inventory(request.plan)?;
    let context = SigmaCertificationContext::new(request.plan, &inventory)?;
    certify_with_coefficients(request, import, None, &context).map_err(String::from)
}

/// Actual retained proof state used by the certifier, not a stack allowance.
/// The request and returned certificate have separate inline move lifetimes;
/// this size alone is deliberately not an aggregate admission quote.
struct SigmaProofWorkspace {
    identities: [[u8; 32]; 6],
    edges: [EdgeDescriptor; 5],
    intervals: [IntegerInterval; 4],
    activation: Option<ActivationSpec>,
    dot: SigmaDotWorkspace,
    hash: Sha256,
    edge_hash: EdgeContractWorkspace,
    activation_metadata: ActivationMetadataWorkspace,
}
impl SigmaProofWorkspace {
    fn new(request: &SigmaLinearActivationRequest<'_>) -> Self {
        Self {
            identities: [[0; 32]; 6],
            edges: [
                request.input,
                request.private_weights,
                request.complete_sum,
                request.rescaled,
                request.activation_output,
            ],
            intervals: [request.input_domain; 4],
            activation: None,
            dot: SigmaDotWorkspace::default(),
            hash: Sha256::new(),
            edge_hash: EdgeContractWorkspace::new(),
            activation_metadata: ActivationMetadataWorkspace::new(&request.rescaled),
        }
    }
}

fn certify_with_coefficients(
    request: SigmaLinearActivationRequest<'_>,
    import: SourceImport,
    checked_coefficients: Option<SigmaCoefficients>,
    context: &SigmaCertificationContext<'_>,
) -> Result<SigmaLinearActivationCertificate, &'static str> {
    let mut workspace = SigmaProofWorkspace::new(&request);
    certify_with_workspace(
        request,
        import,
        checked_coefficients,
        context,
        &mut workspace,
    )
}

fn certify_with_workspace(
    request: SigmaLinearActivationRequest<'_>,
    import: SourceImport,
    checked_coefficients: Option<SigmaCoefficients>,
    context: &SigmaCertificationContext<'_>,
    workspace: &mut SigmaProofWorkspace,
) -> Result<SigmaLinearActivationCertificate, &'static str> {
    let r = &request;
    // Workspace may be reused, never reused as proof material. Retained source
    // and input domain always come from this request; all hash state resets.
    workspace.identities.fill([0; 32]);
    workspace.edges = [
        r.input,
        r.private_weights,
        r.complete_sum,
        r.rescaled,
        r.activation_output,
    ];
    workspace.intervals = [r.input_domain; 4];
    workspace.activation = None;
    workspace.hash.reset();
    if !std::ptr::eq(r.plan, context.plan) {
        return Err("certificate context belongs to a different plan borrow");
    }
    let ([m, inner, n], scope) = geometry(&r.selection)?;
    // Structural/geometry admission belongs to the preexisting immutable
    // inventory. Context checks its stored config and whole-plan bindings.
    let graph = match r.mode {
        DecoderMode::Prefill => &r.plan.prefill,
        DecoderMode::Decode => &r.plan.decode,
    };
    let find = |id: &str| {
        graph
            .operations
            .iter()
            .find(|op| op.id == id)
            .ok_or("semantic operation missing")
    };
    let linear = find(r.linear_operation)?;
    let activation = find(r.activation_operation)?;
    let next = find(r.next_operation)?;
    if linear.operator != ModelOperator::Linear
        || linear.inputs.len() != 1
        || linear.state_kind.is_some()
        || activation.state_kind.is_some()
        || linear.attributes.get("bias").is_some_and(|v| !v.is_null())
        || linear.attributes.get("bias_shape").is_some()
    {
        return Err("unsupported or biased semantic linear operation");
    }
    if activation.operator != ModelOperator::Silu
        || !activation
            .attributes
            .as_object()
            .is_some_and(|a| a.is_empty())
        || activation.inputs.len() != 1
        || activation.inputs[0] != linear.id
        || activation.output_shape != linear.output_shape
        || activation.layer != linear.layer
    {
        return Err("unsupported activation attributes/operator (GELU-tanh is not exact GELU)");
    }
    let input_op = find(&linear.inputs[0])?;
    let original_input = dimensions(&input_op.output_shape)?;
    let original_output = dimensions(&linear.output_shape)?;
    if r.selection.original_input_shape != original_input
        || r.selection.original_weight_shape != [original_output[1], original_input[1]]
        || original_input[0] != original_output[0]
        || r.selection.tensor
            != linear
                .attributes
                .get("weight")
                .and_then(|v| v.as_str())
                .unwrap_or("")
        || r.selection.input_tensor != linear.inputs[0]
        || r.source.checkpoint() != r.selection.content_sha256
    {
        return Err("source tensor/checkpoint/original semantic geometry mismatch");
    }
    workspace.identities[0] = context.phase_identity(r.mode)?;
    workspace.identities[1] = bounded_operation_identity(linear)?;
    workspace.identities[2] = bounded_operation_identity(activation)?;
    workspace.identities[3] = bounded_operation_identity(next)?;
    workspace.identities[4] = bounded_operation_identity(input_op)?;
    workspace.identities[5] = bounded_selection_identity(&r.selection)?;
    if r.rescale_operation
        != hash(
            b"pllm.complete-dot-gapars12-floor-once.v1\0",
            &workspace.identities[1],
        )
    {
        return Err("wrong complete-dot floor-once GapARS12 stage identity");
    }
    check_edge(
        r.input,
        r.source,
        workspace.identities[0],
        workspace.identities[4],
        [m, inner],
        12,
        false,
    )?;
    // Source artifact producer identity binds its exact declaration and selected coordinates.
    check_edge(
        r.private_weights,
        r.source,
        workspace.identities[0],
        workspace.identities[5],
        [n, inner],
        12,
        true,
    )?;
    check_edge(
        r.complete_sum,
        r.source,
        workspace.identities[0],
        workspace.identities[1],
        [m, n],
        24,
        false,
    )?;
    check_edge(
        r.rescaled,
        r.source,
        workspace.identities[0],
        r.rescale_operation,
        [m, n],
        12,
        false,
    )?;
    check_edge(
        r.activation_output,
        r.source,
        workspace.identities[0],
        workspace.identities[2],
        [m, n],
        12,
        false,
    )?;
    if workspace.edges.iter().enumerate().any(|(i, e)| {
        workspace.edges[..i]
            .iter()
            .any(|old| old.identity().wire() == e.identity().wire())
    }) {
        return Err("aliased composition wires");
    }
    let position = usize::from(r.next_use.operand_position());
    if r.next_use.edge() != r.activation_output
        || r.next_use.consumer() != workspace.identities[3]
        || next.inputs.get(position) != Some(&activation.id)
    {
        return Err("wrong next consumer/operand binding");
    }
    workspace.activation = Some(
        ActivationSpec::new_with_workspace(
            ActivationKind::Silu,
            workspace.identities[2],
            r.rescaled,
            r.activation_output,
            workspace.identities[3],
            &mut workspace.activation_metadata,
        )
        .map_err(|_| "unchanged D20 declaration rejected")?,
    );
    let count = n
        .checked_mul(inner)
        .ok_or("coefficient geometry overflow")?;
    if r.source_coefficients.len() != count || r.encoded_private_coefficients.len() != count {
        return Err("actual private coefficient count differs from complete matrix");
    }
    for (&source, &encoded) in r
        .source_coefficients
        .iter()
        .zip(r.encoded_private_coefficients)
    {
        if checked_floor_encode_q12(source)? != encoded {
            return Err("actual private coefficient floor encoding mismatch");
        }
    }
    workspace.intervals[1] = IntegerInterval::checked_new(
        i128::from(*r.encoded_private_coefficients.iter().min().unwrap()),
        i128::from(*r.encoded_private_coefficients.iter().max().unwrap()),
    )?;
    (workspace.intervals[2], workspace.intervals[3]) = complete_dot_interval_with_workspace(
        workspace.intervals[0],
        workspace.intervals[1],
        inner,
        &mut workspace.dot,
    )?;
    let macs = m
        .checked_mul(n)
        .and_then(|v| v.checked_mul(inner))
        .ok_or("MAC work overflow")?;
    if macs > MAX_MACS {
        return Err("existing MAC ceiling exceeded");
    }
    let source_status = match import {
        SourceImport::DeclarationOnly => SourceImportStatus::UnauthenticatedDeclaration,
    };
    let h = &mut workspace.hash;
    h.update(SIGMA_LINEAR_ACTIVATION_SCHEMA.as_bytes());
    bounded_schema_write(
        SigmaSchema::Selection(&r.selection),
        &mut SigmaHashWriter(h),
    )?;
    h.update(context.plan_hex);
    bounded_schema_write(SigmaSchema::Mode(r.mode), &mut SigmaHashWriter(h))?;
    h.update(context.inventory.digest().as_str().as_bytes());
    for edge in &workspace.edges {
        h.update(edge.digest_with_workspace(&mut workspace.edge_hash));
    }
    h.update(r.rescale_operation);
    h.update(
        workspace
            .activation
            .as_ref()
            .ok_or("missing activation proof")?
            .digest_with_workspace(&mut workspace.activation_metadata),
    ); // Includes unchanged D20/table identity.
    h.update(r.next_use.digest_with_workspace(&mut workspace.edge_hash));
    bounded_schema_write(
        SigmaSchema::Intervals(&workspace.intervals),
        &mut SigmaHashWriter(h),
    )?;
    for value in r.source_coefficients {
        h.update(value.to_bits().to_le_bytes());
    }
    for value in r.encoded_private_coefficients {
        h.update(value.to_le_bytes());
    }
    match source_status {
        SourceImportStatus::UnauthenticatedDeclaration => h.update([0]),
        SourceImportStatus::PinnedSourceBytesChecked => h.update([1]),
    }
    Ok(SigmaLinearActivationCertificate {
        digest: h.finalize_reset().into(),
        selection: request.selection,
        scope,
        source_status,
        input_domain: workspace.intervals[0],
        weight_domain: workspace.intervals[1],
        sum_domain: workspace.intervals[2],
        rescaled_domain: workspace.intervals[3],
        coefficients: match checked_coefficients {
            Some(values) if values.as_slice() == request.encoded_private_coefficients => values,
            Some(_) => return Err("checked coefficient ownership mismatch"),
            None => SigmaCoefficients::new(request.encoded_private_coefficients)?,
        },
        activation: workspace
            .activation
            .take()
            .ok_or("missing activation proof")?,
        next_use: request.next_use,
        macs,
        edges: workspace.edges,
        mode: request.mode,
    })
}

/// Weight producer tag shared by declaration construction and checked admission.
pub fn sigma_weight_selection_identity(selection: &CheckpointTensorSelection) -> [u8; 32] {
    hash(
        b"pllm.source-weight-selection.v1\0",
        &canonical_bytes(selection),
    )
}

// Compiler-only typed owning inputs; no SDK/Python registration or raw import.
#[path = "sigma_linear_activation/input_owner.rs"]
pub(crate) mod input_owner;

// Opaque typed Rust handoff only; no SDK/component or Python registration.
#[doc(hidden)]
pub use input_owner::{
    bootstrap_bytes as owned_sigma_input_source_bootstrap,
    inventory_bootstrap_bytes as owned_sigma_input_inventory_bootstrap,
    prepare as prepare_owned_sigma_input,
    prepare_from_original_source as prepare_owned_sigma_input_from_original_source,
    prepare_from_plan_bytes as prepare_owned_sigma_input_from_plan_bytes,
    InputError as SigmaInputError, Operation as OwnedSigmaOperation, OwnedSigmaInput,
    OwnedSigmaInputRetention, PlanInputLimits as OwnedSigmaPlanInputLimits,
    PreparationWork as OwnedSigmaPreparationWork,
    PublisherDeclaration as SigmaPublisherDeclaration,
    SourceInputLimits as OwnedSigmaSourceInputLimits,
};

#[cfg(test)]
#[path = "sigma_linear_activation/tests.rs"]
mod tests;
