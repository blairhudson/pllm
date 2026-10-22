//! Metadata-only publication binding and bounded native admission.
//!
//! No tensor/code reads, preprocessing, secret issuance, checkpoint authentication,
//! executable decoder, or security qualification occurs here. Source identities
//! are caller declarations. In particular, a digest does not authenticate a
//! checkpoint. The unresolved source FP8/activation/BF16 profile never becomes
//! W8A8 by passing geometry admission.

use crate::decoder_matrix_inventory::DecoderMatrixInventory;
use pllm_core::maverick_batch::{self as native, AllocationPlan};
use pllm_models::{DecoderMode, DecoderPlan, ModelOperator};
use pllm_types::{canonical_bytes, digest_bytes, Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DELEGATION_PUBLICATION_INVENTORY_SCHEMA_VERSION: &str =
    "pllm.delegation_publication_inventory.v1";
const MAX_ARTIFACTS: usize = 16384;
const MAX_METADATA_BYTES: usize = 32 << 20;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationArtifactKind {
    Matrix,
    TokenTable,
    Bias,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationSourceLayout {
    OutputInput,
    VocabularyFeature,
    OutputVector,
}

/// Public source pin declaration, not an authenticated source/import owner.
#[derive(Clone, Debug, Serialize)]
pub struct PublicationSourceIdentity {
    pub repository: String,
    pub revision: String,
    pub checkpoint_manifest_digest: Digest,
    pub tokenizer_digest: Digest,
    pub config_digest: Digest,
}

#[derive(Clone, Debug, Serialize)]
pub struct PublicationArtifactIdentity {
    pub kind: PublicationArtifactKind,
    pub weight_id: String,
    pub source_artifact_digest: Digest,
    pub publication_artifact_digest: Digest,
    pub source_shape: Vec<u64>,
    pub source_layout: PublicationSourceLayout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationNumericProfile {
    MaverickSourceFp8Scale256PostLinearBf16Unresolved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationCodeFamily {
    ExplicitRaa8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationOwnership {
    /// Client owns P, privacy codes and verifier state; separate server owns M/Q.
    /// Full intended-source P/Q/Qaux validation is required before role split.
    ClientPrivateCorrectionAndVerifierSeparateServerMatrixAndQ,
}

/// Explicit selected algebra/ownership; not a numerical implementation approval.
#[derive(Clone, Debug, Serialize)]
pub struct DelegationPublicationContract {
    pub field_modulus: u32,
    pub field_element_bytes: u64,
    pub code_family: PublicationCodeFamily,
    pub numeric_profile: PublicationNumericProfile,
    pub ownership: PublicationOwnership,
}

impl DelegationPublicationContract {
    pub fn source_derived_maverick() -> Self {
        Self {
            field_modulus: native::FIELD,
            field_element_bytes: 4,
            code_family: PublicationCodeFamily::ExplicitRaa8,
            numeric_profile:
                PublicationNumericProfile::MaverickSourceFp8Scale256PostLinearBf16Unresolved,
            ownership:
                PublicationOwnership::ClientPrivateCorrectionAndVerifierSeparateServerMatrixAndQ,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct InventoryDocument {
    schema_version: String,
    model_plan_digest: Digest,
    model_config_digest: Digest,
    prefill: Phase,
    decode: Phase,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Phase {
    mode: DecoderMode,
    batch: u64,
    query_sequence: u64,
    maximum_key_sequence: u64,
    groups: Vec<Group>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Group {
    order: u64,
    operator: ModelOperator,
    layer: Option<u64>,
    input_id: String,
    input_shape: Vec<u64>,
    artifact_kind: PublicationArtifactKind,
    orientation: PublicationSourceLayout,
    logical_shape: [u64; 2],
    members: Vec<Member>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Serialize)]
struct MatrixUse {
    mode: DecoderMode,
    /// Dense semantic operation position, not a parsed operation name.
    operation_position: u64,
    group_order: u64,
    member_position: u64,
    operation_id: String,
    artifact_position: u64,
    bias_artifact_position: Option<u64>,
    bias_coverage: Option<BiasCoverage>,
    /// Every token row in this use is retained; no terminal-row optimization.
    claim_row_start: u64,
    claim_row_count: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum BiasCoverage {
    BiasAdditionNumericContractBlocked,
}

#[derive(Debug, Serialize)]
struct SemanticCoverage {
    mode: DecoderMode,
    operation_position: u64,
    operation_id: String,
    operator: ModelOperator,
    coverage: Coverage,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Coverage {
    #[serde(rename = "declared_matrix_numeric_profile_blocked")]
    DeclaredMatrixNumericProfile,
    #[serde(rename = "token_lookup_ownership_and_precision_blocked")]
    TokenLookupOwnershipAndPrecision,
    #[serde(rename = "local_semantic_numeric_and_state_contract_blocked")]
    LocalSemanticNumericAndStateContract,
}

#[derive(Debug, Serialize)]
struct Manifest {
    schema_version: &'static str,
    matrix_inventory_digest: Digest,
    source: PublicationSourceIdentity,
    source_identity_authenticated: bool,
    contract: DelegationPublicationContract,
    inventory: InventoryDocument,
    artifacts: Vec<PublicationArtifactIdentity>,
    uses: Vec<MatrixUse>,
    semantic_coverage: Vec<SemanticCoverage>,
    status: PublicationAdmissionStatus,
}

/// No Deserialize/mutation API: admission consumes only compiler-validated metadata.
#[derive(Debug)]
pub struct DelegationPublicationInventory {
    manifest: Manifest,
    bytes: Vec<u8>,
    digest: Digest,
}

impl DelegationPublicationInventory {
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    pub fn status(&self) -> PublicationAdmissionStatus {
        self.manifest.status
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "status", content = "gate", rename_all = "snake_case")]
pub enum PublicationAdmissionStatus {
    Blocked(PublicationBlockedGate),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum PublicationBlockedGate {
    /// Missing exact FP8/scales, activation encoding, rounding, ranges, BF16,
    /// biases/local semantics, embedding/head precision and state contract.
    #[serde(rename = "M-N")]
    MaverickSourceNumericProfile,
}

fn bounded_text(value: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > 1024 {
        return Err("publication identity must contain 1..=1024 UTF-8 bytes".into());
    }
    Ok(())
}

fn checked_product(shape: &[u64]) -> Result<u64, String> {
    if shape.is_empty() || shape.iter().any(|&n| n == 0) {
        return Err("publication source has empty or zero geometry".into());
    }
    shape.iter().try_fold(1_u64, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| "publication source shape overflow".into())
    })
}

fn bounded_canonical_bytes(value: &impl Serialize) -> Result<Vec<u8>, String> {
    struct Budget(usize);
    impl std::io::Write for Budget {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_sub(bytes.len()).ok_or_else(|| {
                std::io::Error::other("publication metadata byte ceiling exceeded")
            })?;
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(Budget(MAX_METADATA_BYTES), value).map_err(|e| e.to_string())?;
    Ok(canonical_bytes(value))
}

/// Complete ordered matrices, full head, table, biases and semantic coverage.
/// Reuses the immutable inventory, including its model-neutral grouping and true
/// source shapes. No operation-name parsing or W8A8 runtime schedule is used.
pub fn lower_delegation_publication_inventory(
    plan: &DecoderPlan,
    inventory: &DecoderMatrixInventory,
    source: PublicationSourceIdentity,
    artifacts: Vec<PublicationArtifactIdentity>,
    contract: DelegationPublicationContract,
) -> Result<DelegationPublicationInventory, String> {
    if artifacts.is_empty() || artifacts.len() > MAX_ARTIFACTS {
        return Err("publication artifact count exceeds bounded metadata".into());
    }
    bounded_text(&source.repository)?;
    bounded_text(&source.revision)?;
    for identity in [
        &source.checkpoint_manifest_digest,
        &source.tokenizer_digest,
        &source.config_digest,
    ] {
        bounded_text(identity.as_str())?;
    }
    if contract.field_modulus != native::FIELD || contract.field_element_bytes != 4 {
        return Err("publication requires explicit BabyBear canonical-u32 algebra".into());
    }
    // The caller can supply a different, publicly mutable plan. The existing
    // inventory only bounds the plan from which it was originally constructed.
    // Reuse the shared lowering entry point: its in-place metadata preflight
    // checks strings, node/count/depth limits and shapes before serialization or
    // hashing. Do not call plan.digest() to reject a replacement plan first.
    let supplied_inventory = crate::decoder_matrix_inventory::lower_decoder_matrix_inventory(plan)?;
    if supplied_inventory.canonical_bytes() != inventory.canonical_bytes() {
        return Err("publication source/plan/inventory identity mismatch".into());
    }
    let document: InventoryDocument = serde_json::from_slice(inventory.canonical_bytes())
        .map_err(|e| format!("unsupported immutable matrix inventory: {e}"))?;
    if document.schema_version
        != crate::decoder_matrix_inventory::DECODER_MATRIX_INVENTORY_SCHEMA_VERSION
        || document.model_config_digest != source.config_digest
        || document.model_config_digest != plan.config_digest
    {
        return Err("publication source/plan/inventory identity mismatch".into());
    }
    let mut declared = BTreeMap::new();
    for (position, artifact) in artifacts.iter().enumerate() {
        bounded_text(&artifact.weight_id)?;
        bounded_text(artifact.source_artifact_digest.as_str())?;
        bounded_text(artifact.publication_artifact_digest.as_str())?;
        if artifact.source_shape.len() > 2 {
            return Err("publication source rank unsupported".into());
        }
        checked_product(&artifact.source_shape)?;
        if declared
            .insert(
                (artifact.kind, artifact.weight_id.as_str()),
                position as u64,
            )
            .is_some()
        {
            return Err("duplicate publication artifact identity".into());
        }
    }
    let mut required = BTreeSet::new();
    let mut uses = Vec::new();
    let mut semantic_coverage = Vec::new();
    for (phase, graph) in [
        (&document.prefill, &plan.prefill),
        (&document.decode, &plan.decode),
    ] {
        let positions: BTreeMap<_, _> = graph
            .operations
            .iter()
            .enumerate()
            .map(|(i, op)| (op.id.as_str(), i as u64))
            .collect();
        for (position, op) in graph.operations.iter().enumerate() {
            semantic_coverage.push(SemanticCoverage {
                mode: phase.mode,
                operation_position: position as u64,
                operation_id: op.id.clone(),
                operator: op.operator,
                coverage: match op.operator {
                    ModelOperator::Linear | ModelOperator::OutputHead => {
                        Coverage::DeclaredMatrixNumericProfile
                    }
                    ModelOperator::TokenLookup => Coverage::TokenLookupOwnershipAndPrecision,
                    _ => Coverage::LocalSemanticNumericAndStateContract,
                },
            });
        }
        for group in &phase.groups {
            for (member_position, member) in group.members.iter().enumerate() {
                let mut bind = |kind, id: &str, shape: &[u64], layout| -> Result<u64, String> {
                    let key = (kind, id);
                    let position = *declared.get(&key).ok_or("missing publication artifact")?;
                    let artifact = &artifacts[position as usize];
                    if artifact.source_shape != shape || artifact.source_layout != layout {
                        return Err("publication source layout or shape mismatch".into());
                    }
                    required.insert(position);
                    Ok(position)
                };
                let artifact_position = bind(
                    group.artifact_kind,
                    &member.weight_id,
                    &member.source_shape,
                    group.orientation,
                )?;
                let bias_artifact_position = member
                    .bias_id
                    .as_ref()
                    .map(|id| {
                        bind(
                            PublicationArtifactKind::Bias,
                            id,
                            member.bias_shape.as_ref().ok_or("missing bias shape")?,
                            PublicationSourceLayout::OutputVector,
                        )
                    })
                    .transpose()?;
                let rows = checked_product(&member.output_shape[..member.output_shape.len() - 1])?;
                uses.push(MatrixUse {
                    mode: phase.mode,
                    operation_position: *positions
                        .get(member.operation_id.as_str())
                        .ok_or("matrix use has no semantic operation")?,
                    group_order: group.order,
                    member_position: member_position as u64,
                    operation_id: member.operation_id.clone(),
                    artifact_position,
                    bias_artifact_position,
                    bias_coverage: bias_artifact_position
                        .map(|_| BiasCoverage::BiasAdditionNumericContractBlocked),
                    claim_row_start: 0,
                    claim_row_count: rows,
                });
            }
        }
    }
    if required.len() != artifacts.len() {
        return Err("extraneous publication artifact; exact coverage required".into());
    }
    let manifest = Manifest {
        schema_version: DELEGATION_PUBLICATION_INVENTORY_SCHEMA_VERSION,
        matrix_inventory_digest: inventory.digest().clone(),
        source,
        source_identity_authenticated: false,
        contract,
        inventory: document,
        artifacts,
        uses,
        semantic_coverage,
        status: PublicationAdmissionStatus::Blocked(
            PublicationBlockedGate::MaverickSourceNumericProfile,
        ),
    };
    let bytes = bounded_canonical_bytes(&manifest)?;
    let digest = digest_bytes(DELEGATION_PUBLICATION_INVENTORY_SCHEMA_VERSION, &bytes);
    Ok(DelegationPublicationInventory {
        manifest,
        bytes,
        digest,
    })
}

/// Distinct historical identity. Not a selectable fallback for the source track.
pub const HISTORICAL_MAVERICK_SCALAR_IDENTITY: &str =
    "MaverickDelegatedLinear/seeded-raa/signed-i8/BabyBear/ty41/256MiB";

/// Explicit ordered native matrix slots. Each slot refers to one logical matrix
/// group (possibly concatenated source members), never to an inferred family op.
#[derive(Clone, Debug, Serialize)]
pub struct MaverickBatchPublication {
    pub group_orders: [u64; 2],
    /// Native CodeDescription::digest identities; declarations only. Actual code
    /// arrays and every P/Q/Qaux entry must still pass native source validation.
    pub privacy_code_digests: [[u8; 32]; 2],
    pub primary_code_digest: [u8; 32],
    pub secondary_code_digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize)]
pub struct MaverickBatchAdmissionRequest {
    pub rows: usize,
    pub columns: usize,
    pub batch_size: usize,
    pub capsule_ceiling_bytes: usize,
    /// All capsules remain resident. No implicit streaming/offload/sharing.
    pub aggregate_ceiling_bytes: u64,
    pub live_windows: u32,
    pub attempts_per_client: u32,
    pub publications: Vec<MaverickBatchPublication>,
}

#[derive(Debug, Serialize)]
pub struct NativePublicationAllocation {
    pub public_payload_bytes: u64,
    pub source_and_incoming_validation_overlap_bytes: u64,
    pub separate_server_matrix_and_q_bytes: u64,
    pub transient_transcript_and_validation_allowance_bytes: u64,
    pub control_stack_allocator_allowance_bytes: u64,
    pub retained_output_bytes: u64,
    pub planned_peak_bytes: u64,
    /// Payload decomposition of the native fixed geometry, checked against its
    /// public/server payload getters. Not an arbitrary-shape native estimator.
    pub p_bytes: u64,
    pub q_bytes: u64,
    pub qaux_bytes: u64,
    pub matrix_bytes: u64,
    pub code_bytes: u64,
}

#[derive(Debug, Serialize)]
struct Admission {
    schema_version: &'static str,
    publication_inventory_digest: Digest,
    profile: &'static str,
    native_parameters: NativeParameters,
    request: MaverickBatchAdmissionRequest,
    native_allocation_per_capsule: NativePublicationAllocation,
    aggregate_planned_peak_bytes: u64,
    resource_scope: &'static str,
    unpriced_decoder_resources: [&'static str; 4],
    status: PublicationAdmissionStatus,
    executable: bool,
    checkpoint_authenticated: bool,
    native_publication_validation_completed: bool,
    code_security_qualified: bool,
}

#[derive(Debug, Serialize)]
struct NativeParameters {
    field_modulus: u32,
    field_element_bytes: u64,
    privacy_code_dimension: usize,
    privacy_code_length: usize,
    privacy_exact_weight: usize,
    primary_code_dimension: usize,
    primary_code_length: usize,
    primary_exact_weight: usize,
    primary_checks: usize,
    secondary_code_dimension: usize,
    secondary_code_length: usize,
    secondary_exact_weight: usize,
    secondary_checks: usize,
}

#[derive(Debug)]
pub struct MaverickBatchPublicationAdmission {
    bytes: Vec<u8>,
    digest: Digest,
}

impl MaverickBatchPublicationAdmission {
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    pub fn status(&self) -> PublicationAdmissionStatus {
        PublicationAdmissionStatus::Blocked(PublicationBlockedGate::MaverickSourceNumericProfile)
    }
}

/// Calls actual immutable native AllocationPlan before any code/tensor reads or
/// secret issuance. Geometry/cap admission does not clear M-N or decoder gates.
pub fn admit_maverick_batch_publication(
    inventory: &DelegationPublicationInventory,
    request: MaverickBatchAdmissionRequest,
) -> Result<MaverickBatchPublicationAdmission, String> {
    let plan = AllocationPlan::for_shape(
        request.rows,
        request.columns,
        request.batch_size,
        request.capsule_ceiling_bytes,
    )
    .map_err(|e| e.to_string())?;
    if request.live_windows != 1 || request.attempts_per_client != native::MAX_ATTEMPTS {
        return Err("native publication requires one live window and 16 attempts".into());
    }
    if request.publications.is_empty() || request.publications.len() > 4096 {
        return Err("native publication count outside bounded metadata".into());
    }
    let groups = &inventory.manifest.inventory.prefill.groups;
    let expected: BTreeSet<_> = groups
        .iter()
        .filter(|g| g.artifact_kind == PublicationArtifactKind::Matrix)
        .map(|g| g.order)
        .collect();
    let mut covered = BTreeSet::new();
    for publication in &request.publications {
        for order in publication.group_orders {
            let group = groups
                .iter()
                .find(|g| g.order == order && g.artifact_kind == PublicationArtifactKind::Matrix)
                .ok_or("native publication references unknown matrix group")?;
            if !covered.insert(order) {
                return Err("duplicate native matrix slot".into());
            }
            // Do not tile, pad, truncate the vocabulary or replace the full head.
            if group.logical_shape != [request.rows as u64, request.columns as u64] {
                return Err(
                    "matrix/full-head geometry unsupported by native AllocationPlan".into(),
                );
            }
        }
    }
    if covered != expected {
        return Err("native publication omits matrix/full-head groups".into());
    }
    let aggregate = (plan.planned_peak_bytes() as u64)
        .checked_mul(request.publications.len() as u64)
        .ok_or("aggregate native allocation overflow")?;
    if aggregate > request.aggregate_ceiling_bytes {
        return Err("aggregate native publication ceiling exceeded".into());
    }
    // Only after for_shape has admitted the fixed native geometry. Native has
    // no P/Q/Qaux getters; decompose its exact fixed payload and cross-check.
    let word = 4_u64;
    let m = native::DIMENSION as u64;
    let n = native::DIMENSION as u64;
    let b = native::BATCH_SIZE as u64;
    let length = native::LENGTH as u64;
    let secondary = native::SECONDARY_LENGTH as u64;
    let matrix_bytes = b * m * n * word;
    let p_bytes = b * m * length * word;
    let q_bytes = b * length * n * word;
    let qaux_bytes = length * secondary * word;
    let code_bytes = (b * length + length + secondary) * 16;
    if matrix_bytes + p_bytes + q_bytes + qaux_bytes + code_bytes
        != plan.public_payload_bytes() as u64
        || matrix_bytes + q_bytes != plan.server_payload_bytes() as u64
    {
        return Err("native allocation payload contract changed".into());
    }
    let allocation = NativePublicationAllocation {
        public_payload_bytes: plan.public_payload_bytes() as u64,
        source_and_incoming_validation_overlap_bytes: plan.external_payload_bytes() as u64,
        separate_server_matrix_and_q_bytes: plan.server_payload_bytes() as u64,
        transient_transcript_and_validation_allowance_bytes: plan.transient_allowance_bytes()
            as u64,
        control_stack_allocator_allowance_bytes: plan.control_allowance_bytes() as u64,
        retained_output_bytes: plan.retained_output_bytes() as u64,
        planned_peak_bytes: plan.planned_peak_bytes() as u64,
        p_bytes,
        q_bytes,
        qaux_bytes,
        matrix_bytes,
        code_bytes,
    };
    const DOMAIN: &str = "pllm.maverick_batch_publication_admission.v1";
    let admission = Admission {
        schema_version: DOMAIN,
        publication_inventory_digest: inventory.digest.clone(),
        profile: "maverick.source-derived-explicit-raa8-babybear-64x64-b2.v1",
        native_parameters: NativeParameters {
            field_modulus: native::FIELD,
            field_element_bytes: 4,
            privacy_code_dimension: native::DIMENSION,
            privacy_code_length: native::LENGTH,
            privacy_exact_weight: native::PRIVACY_WEIGHT,
            primary_code_dimension: native::DIMENSION,
            primary_code_length: native::LENGTH,
            primary_exact_weight: native::CHECK_WEIGHT,
            primary_checks: native::REPETITIONS,
            secondary_code_dimension: native::SECONDARY_DIMENSION,
            secondary_code_length: native::SECONDARY_LENGTH,
            secondary_exact_weight: native::CHECK_WEIGHT,
            secondary_checks: native::REPETITIONS,
        },
        request,
        native_allocation_per_capsule: allocation,
        aggregate_planned_peak_bytes: aggregate,
        resource_scope: "resident native capsules only; not complete decoder memory or throughput",
        unpriced_decoder_resources: [
            "token table and lookup precision/ownership",
            "bias conversion/addition and local nonlinear operators",
            "KV/state, logits, tokenizer, model import and transport owners",
            "caller-created copies, process RSS and allocator residency",
        ],
        status: inventory.status(),
        executable: false,
        checkpoint_authenticated: false,
        native_publication_validation_completed: false,
        code_security_qualified: false,
    };
    let bytes = bounded_canonical_bytes(&admission)?;
    let digest = digest_bytes(DOMAIN, &bytes);
    Ok(MaverickBatchPublicationAdmission { bytes, digest })
}

#[cfg(test)]
#[path = "delegation_publication_inventory/tests.rs"]
mod tests;
