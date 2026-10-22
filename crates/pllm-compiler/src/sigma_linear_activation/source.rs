//! D22 floor-Q12 policy over shared concrete VerifiedSelectedSource ownership.
//! Native issuance requires the admitted checked-source facade; no callback can
//! mint source authority.
use super::*;
use crate::verified_source::{
    checked_digest_identity, checked_sum_bytes, checked_verify_pinned_source_bytes_with_workspace,
    ExpectedSelectedSource, SelectedSourceTuple, SelectedSourceWorkspace, VerifiedSelectedSource,
    SELECTED_SOURCE_BYTE_CAPACITY,
};
pub use crate::verified_source::{SourceDtype, SourceEndian};
use pllm_types::{Digest, NamedDigest};

/// Concrete reusable source verification/decoding and certificate-commitment
/// state. Construct only after caller reservation in a bounded operation. Both
/// import and certification accept the SAME workspace; no source owner escapes.
pub(super) struct SigmaSourceWorkspace {
    selected: SelectedSourceWorkspace,
    certification_hash: Sha256,
}
impl SigmaSourceWorkspace {
    pub(super) fn new() -> Self {
        Self {
            selected: SelectedSourceWorkspace::new(),
            certification_hash: Sha256::new(),
        }
    }
}

/// Frozen independently expected publisher/config/semantic-plan tuple. Raw
/// selected-byte digest, not whole-container digest. Physical tensor offset is
/// a trusted publisher mapping, not discovered from a safetensors header.
/// Expected semantic identity is the publisher's `DecoderPlan::digest()`, not
/// the phase-specific edge tag. Config bytes are not loaded/authenticated here.
pub struct ExpectedSigmaSourcePin {
    selected: ExpectedSelectedSource,
    selection: CheckpointTensorSelection,
}
impl ExpectedSigmaSourcePin {
    #[allow(clippy::too_many_arguments)]
    pub fn from_trusted_publisher(
        artifact: NamedDigest,
        selected_bytes: usize,
        selection: CheckpointTensorSelection,
        dtype: SourceDtype,
        endian: SourceEndian,
        tensor_offset: u64,
        expected_config: &Digest,
        expected_semantic_plan: &Digest,
    ) -> Result<Self, String> {
        Self::checked_from_trusted_publisher(
            artifact,
            selected_bytes,
            selection,
            dtype,
            endian,
            tensor_offset,
            expected_config,
            expected_semantic_plan,
        )
        .map_err(String::from)
    }

    // Expected metadata construction predates the admitted boundary. It clones
    // declaration metadata, never source bytes; failures still use static text.
    #[allow(clippy::too_many_arguments)]
    fn checked_from_trusted_publisher(
        artifact: NamedDigest,
        selected_bytes: usize,
        selection: CheckpointTensorSelection,
        dtype: SourceDtype,
        endian: SourceEndian,
        tensor_offset: u64,
        expected_config: &Digest,
        expected_semantic_plan: &Digest,
    ) -> Result<Self, &'static str> {
        geometry(&selection)?;
        if checked_digest_identity(&artifact.digest)? != selection.content_sha256 {
            return Err("inconsistent expected source digest");
        }
        let tuple = SelectedSourceTuple {
            checkpoint: selection.checkpoint.clone(),
            immutable_revision: selection.immutable_revision.clone(),
            tensor: selection.tensor.clone(),
            original_shape: selection.original_weight_shape,
            output_coordinates: selection.output_coordinates.clone(),
            inner_coordinates: selection.inner_coordinates.clone(),
            output_input_row_major: selection.weight_layout == "output_input_row_major",
            dtype,
            endian,
            tensor_offset,
            expected_config: checked_digest_identity(expected_config)?,
            expected_semantic_plan: checked_digest_identity(expected_semantic_plan)?,
        };
        let result = Self {
            selected: ExpectedSelectedSource::checked_from_trusted_publisher(
                artifact,
                tuple,
                selected_bytes,
            )?,
            selection,
        };
        result.checked_import_quote()?;
        Ok(result)
    }
    pub fn selection(&self) -> &CheckpointTensorSelection {
        &self.selection
    }
    #[cfg(test)]
    fn offsets(&self) -> Result<Vec<u64>, String> {
        (0..self.selected.selected_bytes() / self.selected.tuple().dtype.bytes())
            .map(|i| self.selected.scalar_artifact_offset(i))
            .collect()
    }
    /// Exact owned shapes, calculable before any actual source bytes are copied.
    /// Expected metadata is already caller-owned; its real capacities are priced.
    pub fn import_quote(&self) -> Result<SigmaSourceResources, String> {
        self.checked_import_quote().map_err(String::from)
    }

    pub(super) fn checked_import_quote(&self) -> Result<SigmaSourceResources, &'static str> {
        resources(
            self.selection.checked_metadata_backing_bytes()?,
            self.selected.checked_metadata_backing_bytes()?,
            true,
        )
    }
    /// Final owner quote available BEFORE copying source bytes. The supplied
    /// request selection will move into the certificate without capacity changes.
    /// Reserve the maximum of import overlap and this retained quote, plus
    /// separately priced native/borrowed/compiler-scratch inventories.
    pub fn certification_retained_quote(
        &self,
        request: &SigmaLinearActivationRequest<'_>,
    ) -> Result<usize, String> {
        self.checked_certification_retained_quote(request)
            .map_err(String::from)
    }

    pub(super) fn checked_certification_retained_quote(
        &self,
        request: &SigmaLinearActivationRequest<'_>,
    ) -> Result<usize, &'static str> {
        final_retained_quote(
            self.selection.checked_metadata_backing_bytes()?,
            self.selected.checked_metadata_backing_bytes()?,
            request.selection.checked_metadata_backing_bytes()?,
        )
    }
}

pub struct SigmaSourceBytes<'a> {
    pub artifact_name: &'a str,
    pub selection: &'a CheckpointTensorSelection,
    pub dtype: SourceDtype,
    pub endian: SourceEndian,
    pub tensor_offset: u64,
    pub scalar_artifact_offsets: &'a [u64],
    pub bytes: &'a [u8],
}

/// Exact retained logical capacities plus partial import/encoding state categories.
/// No owned f64 buffer, offsets Vec or Arc exists. No allocator headers/RSS claim.
/// Borrowed caller plan/source/issuance buffers are NOT charged here. Existing
/// inventory lowering is outside the proposed boundary. These categories are
/// NOT an aggregate compiler/import overlap. The internal path uses static
/// errors; public String-returning methods are compatibility wrappers only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SigmaSourceResources {
    pub owner_metadata_bytes: usize,
    pub metadata_backing_bytes: usize,
    pub source_byte_capacity: usize,
    pub decoded_f64_capacity_bytes: usize,
    pub coefficient_capacity_bytes: usize,
    pub arc_backing_bytes: usize,
    pub retained_bytes: usize,
    pub import_encoding_scratch_bytes: usize,
    pub import_encoding_overlap_bytes: usize,
}

/// Source-only lifetime audit. Backings/one destination CheckedSigmaSource live
/// in retained_bytes. These are the actual additional origin/return slots at
/// each call boundary, not an allowance per operation/node. Model/schema state,
/// the enclosing request/context/proof, and native transfer are NOT covered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SigmaSourcePhaseStorage {
    workspace_bytes: usize,
    source_initialization_bytes: usize,
    coefficient_initialization_bytes: usize,
    hash_result_move_bytes: usize,
    verification_move_bytes: usize,
    encoding_move_bytes: usize,
    return_move_bytes: usize,
}
impl SigmaSourcePhaseStorage {
    fn inventory() -> Result<Self, &'static str> {
        Ok(Self {
            workspace_bytes: size_of::<SigmaSourceWorkspace>(),
            source_initialization_bytes: size_of::<[u8; SELECTED_SOURCE_BYTE_CAPACITY]>(),
            coefficient_initialization_bytes: size_of::<[i64; MAX_ELEMENTS]>(),
            hash_result_move_bytes: checked_sum_bytes([
                size_of::<sha2::digest::Output<Sha256>>(),
                size_of::<[u8; 32]>(),
            ])?,
            // Incoming pin -> shared verifier argument -> shared return owner
            // -> caller's verified local. One destination verified is already
            // contained in retained CheckedSigmaSource; charge the other slot.
            verification_move_bytes: checked_sum_bytes([
                size_of::<ExpectedSigmaSourcePin>(),
                size_of::<ExpectedSelectedSource>(),
                size_of::<VerifiedSelectedSource>(),
                size_of::<ZeroizingFixed<u8, SELECTED_SOURCE_BYTE_CAPACITY>>(),
            ])?,
            // Original pin's unmoved selection, verified local, coefficient
            // constructor result; destination Checked already counted retained.
            encoding_move_bytes: checked_sum_bytes([
                size_of::<ExpectedSigmaSourcePin>(),
                size_of::<VerifiedSelectedSource>(),
                size_of::<SigmaCoefficients>(),
            ])?,
            // Moving result into caller while original inline slot still exists.
            return_move_bytes: size_of::<CheckedSigmaSource>(),
        })
    }

    fn scratch_bytes(self) -> Result<usize, &'static str> {
        let verification = checked_sum_bytes([
            // Original-byte hash, Box initialization, and binding hash run
            // sequentially. Count their max, not simultaneous hash/init scratch.
            self.source_initialization_bytes
                .max(self.hash_result_move_bytes),
            self.verification_move_bytes,
        ])?;
        let encoding = checked_sum_bytes([
            self.coefficient_initialization_bytes,
            self.encoding_move_bytes,
        ])?;
        // Source Box initialization finishes before coefficient initialization;
        // never add both initialization arrays as simultaneously live scratch.
        checked_sum_bytes([
            self.workspace_bytes,
            verification.max(encoding).max(self.return_move_bytes),
        ])
    }
}

fn resources(
    sigma_metadata: usize,
    source_metadata: usize,
    coefficients: bool,
) -> Result<SigmaSourceResources, &'static str> {
    let owner_metadata_bytes = size_of::<CheckedSigmaSource>();
    let metadata_backing_bytes = checked_source_metadata_bytes(sigma_metadata, source_metadata)?;
    let coefficient_capacity_bytes = if coefficients {
        size_of::<[i64; MAX_ELEMENTS]>()
    } else {
        0
    };
    let retained_bytes = checked_sum_bytes([
        owner_metadata_bytes,
        metadata_backing_bytes,
        SELECTED_SOURCE_BYTE_CAPACITY,
        coefficient_capacity_bytes,
    ])?;
    let import_encoding_scratch_bytes = SigmaSourcePhaseStorage::inventory()?.scratch_bytes()?;
    Ok(SigmaSourceResources {
        owner_metadata_bytes,
        metadata_backing_bytes,
        source_byte_capacity: SELECTED_SOURCE_BYTE_CAPACITY,
        decoded_f64_capacity_bytes: 0,
        coefficient_capacity_bytes,
        arc_backing_bytes: 0,
        retained_bytes,
        import_encoding_scratch_bytes,
        import_encoding_overlap_bytes: checked_sum_bytes([
            retained_bytes,
            import_encoding_scratch_bytes,
        ])?,
    })
}

fn checked_source_metadata_bytes(
    sigma_metadata: usize,
    source_metadata: usize,
) -> Result<usize, &'static str> {
    let bytes = checked_sum_bytes([sigma_metadata, source_metadata])?;
    if bytes > 64 * 1024 {
        return Err("source metadata capacity limit exceeded");
    }
    Ok(bytes)
}

fn final_retained_quote(
    sigma_metadata: usize,
    source_metadata: usize,
    certificate_metadata: usize,
) -> Result<usize, &'static str> {
    checked_sum_bytes([
        size_of::<SourceCheckedSigmaLinearActivation>(),
        sigma_metadata,
        source_metadata,
        certificate_metadata,
        SELECTED_SOURCE_BYTE_CAPACITY,
        size_of::<[i64; MAX_ELEMENTS]>(),
    ])
}

/// Owns verified original bytes and one canonical coefficient backing. That
/// backing moves into the certificate at certification; it is never cloned.
pub struct CheckedSigmaSource {
    selection: CheckpointTensorSelection,
    verified: VerifiedSelectedSource,
    coefficients: Option<SigmaCoefficients>,
    // Frozen concrete backing capacity from the immutable verified owner. Avoid
    // rerunning source resource calculations during native transfer.
    source_metadata_bytes: usize,
}
impl CheckedSigmaSource {
    pub fn selection(&self) -> &CheckpointTensorSelection {
        &self.selection
    }
    pub fn encoded_private_coefficients(&self) -> &[i64] {
        self.coefficients
            .as_ref()
            .map_or(&[], SigmaCoefficients::as_slice)
    }
    pub fn source_value(&self, index: usize) -> Result<f64, String> {
        self.verified.finite_value(index)
    }
    pub fn source_binding(&self) -> [u8; 32] {
        self.verified.source_binding()
    }
    pub fn source_byte_count(&self) -> usize {
        self.verified.bytes().len()
    }
    pub fn resources(&self) -> Result<SigmaSourceResources, String> {
        self.checked_resources().map_err(String::from)
    }

    fn checked_resources(&self) -> Result<SigmaSourceResources, &'static str> {
        resources(
            self.selection.checked_metadata_backing_bytes()?,
            self.source_metadata_bytes,
            self.coefficients.is_some(),
        )
    }
    /// Pre-transfer retained quote. Includes actual incoming certificate
    /// selection capacities but no second coefficient allocation or source copy.
    /// This is not a bound on existing inventory/serde compiler scratch.
    pub fn certification_retained_quote(
        &self,
        request: &SigmaLinearActivationRequest<'_>,
    ) -> Result<usize, String> {
        self.checked_certification_retained_quote(request)
            .map_err(String::from)
    }

    fn checked_certification_retained_quote(
        &self,
        request: &SigmaLinearActivationRequest<'_>,
    ) -> Result<usize, &'static str> {
        final_retained_quote(
            self.selection.checked_metadata_backing_bytes()?,
            self.source_metadata_bytes,
            request.selection.checked_metadata_backing_bytes()?,
        )
    }
}

pub fn import_sigma_source_bytes(
    expected: ExpectedSigmaSourcePin,
    supplied: SigmaSourceBytes<'_>,
    canonical_issuance_coefficients: &[i64],
) -> Result<CheckedSigmaSource, String> {
    checked_import_sigma_source_bytes(expected, supplied, canonical_issuance_coefficients)
        .map_err(String::from)
}

/// Sole D22 import/encoding path. Reserve the still-unproved aggregate before
/// calling; static failures themselves never allocate error metadata.
pub(super) fn checked_import_sigma_source_bytes(
    expected: ExpectedSigmaSourcePin,
    supplied: SigmaSourceBytes<'_>,
    canonical_issuance_coefficients: &[i64],
) -> Result<CheckedSigmaSource, &'static str> {
    checked_import_sigma_source_bytes_with_workspace(
        expected,
        supplied,
        canonical_issuance_coefficients,
        &mut SigmaSourceWorkspace::new(),
    )
}

pub(super) fn checked_import_sigma_source_bytes_with_workspace(
    expected: ExpectedSigmaSourcePin,
    supplied: SigmaSourceBytes<'_>,
    canonical_issuance_coefficients: &[i64],
    workspace: &mut SigmaSourceWorkspace,
) -> Result<CheckedSigmaSource, &'static str> {
    // Quote records need not be constructed inside the admitted import. Reuse
    // exactly the same capacity policy with a scalar-only checked metadata sum.
    let source_metadata_bytes = expected.selected.checked_metadata_backing_bytes()?;
    checked_source_metadata_bytes(
        expected.selection.checked_metadata_backing_bytes()?,
        source_metadata_bytes,
    )?;
    let tuple = expected.selected.tuple();
    if supplied.selection != &expected.selection
        || supplied.dtype != tuple.dtype
        || supplied.endian != tuple.endian
        || supplied.tensor_offset != tuple.tensor_offset
        || canonical_issuance_coefficients.len() != tuple.checked_count()?
    {
        return Err("source revision/selection/dtype/endian/offset/complete-term mismatch");
    }
    let verified = checked_verify_pinned_source_bytes_with_workspace(
        expected.selected,
        supplied.artifact_name,
        supplied.scalar_artifact_offsets,
        supplied.bytes,
        &mut workspace.selected,
    )?;
    // Shared owner only decodes finite source; floor/ring50 policy stays here.
    for (i, &issued) in canonical_issuance_coefficients.iter().enumerate() {
        if checked_floor_encode_q12(
            verified.checked_finite_value_with_workspace(i, &mut workspace.selected.decode)?,
        )? != issued
        {
            return Err("canonical issuance coefficient differs from pinned source floor-Q12");
        }
    }
    let result = CheckedSigmaSource {
        selection: expected.selection,
        verified,
        coefficients: Some(SigmaCoefficients::new(canonical_issuance_coefficients)?),
        source_metadata_bytes,
    };
    Ok(result)
}

/// Proposed consuming public execution facade input; no native permit or
/// LocalAssumption relabeling is exposed. Keep source guard until this drops.
pub struct SourceCheckedSigmaLinearActivation {
    certificate: SigmaLinearActivationCertificate,
    source: CheckedSigmaSource,
}
impl SourceCheckedSigmaLinearActivation {
    #[cfg(test)]
    pub(super) fn source_byte_pointer(&self) -> *const u8 {
        self.source.verified.bytes().as_ptr()
    }
    pub fn certificate(&self) -> &SigmaLinearActivationCertificate {
        &self.certificate
    }
    pub fn source_binding(&self) -> [u8; 32] {
        self.source.source_binding()
    }
    pub fn native_elevation(&self) -> Result<(), &'static str> {
        self.certificate.native_elevation()
    }
    pub fn retained_resource_bytes(&self) -> Result<usize, String> {
        self.checked_retained_resource_bytes().map_err(String::from)
    }

    pub(super) fn checked_retained_resource_bytes(&self) -> Result<usize, &'static str> {
        [
            size_of::<Self>(),
            self.certificate
                .selection
                .checked_metadata_backing_bytes()?,
            self.source.selection.checked_metadata_backing_bytes()?,
            self.source.source_metadata_bytes,
            SELECTED_SOURCE_BYTE_CAPACITY,
            size_of::<[i64; MAX_ELEMENTS]>(),
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or("retained source capacity overflow")
        })
    }
}
pub fn certify_source_checked_sigma_linear_activation(
    request: SigmaLinearActivationRequest<'_>,
    source: CheckedSigmaSource,
) -> Result<SourceCheckedSigmaLinearActivation, String> {
    // Compatibility construction, NOT an admitted execution entry point. Raw
    // lowering and preexisting borrowed input construction are outside boundary.
    let inventory = lower_decoder_matrix_inventory(request.plan)?;
    certify_source_checked_with_inventory(request, source, &inventory).map_err(String::from)
}

/// Reuse one preexisting inventory and immutable plan borrow through the entire
/// certification. No lowering, allocating plan digest or canonical serializer.
pub(super) fn certify_source_checked_with_inventory(
    request: SigmaLinearActivationRequest<'_>,
    source: CheckedSigmaSource,
    inventory: &DecoderMatrixInventory,
) -> Result<SourceCheckedSigmaLinearActivation, &'static str> {
    certify_source_checked_with_inventory_and_workspace(
        request,
        source,
        inventory,
        &mut SigmaSourceWorkspace::new(),
    )
}

pub(super) fn certify_source_checked_with_inventory_and_workspace(
    request: SigmaLinearActivationRequest<'_>,
    source: CheckedSigmaSource,
    inventory: &DecoderMatrixInventory,
    workspace: &mut SigmaSourceWorkspace,
) -> Result<SourceCheckedSigmaLinearActivation, &'static str> {
    let context = SigmaCertificationContext::new(request.plan, inventory)?;
    certify_source_checked_with_context_and_workspace(request, source, &context, workspace)
}

fn certify_source_checked_with_context_and_workspace(
    request: SigmaLinearActivationRequest<'_>,
    mut source: CheckedSigmaSource,
    context: &SigmaCertificationContext<'_>,
    workspace: &mut SigmaSourceWorkspace,
) -> Result<SourceCheckedSigmaLinearActivation, &'static str> {
    let tuple = source.verified.expected_tuple();
    if checked_digest_identity(&request.plan.config_digest)? != tuple.expected_config
        || context.plan_identity != tuple.expected_semantic_plan
    {
        return Err(
            "semantic plan/config differs from independently expected frozen publisher tuple",
        );
    }
    if request.selection != source.selection
        || request.encoded_private_coefficients != source.encoded_private_coefficients()
        || request.source_coefficients.len() != source.verified.scalar_count()
    {
        return Err("checked source owner differs from certificate/issuance material");
    }
    for (i, value) in request.source_coefficients.iter().enumerate() {
        if value.to_bits()
            != source
                .verified
                .checked_finite_value_with_workspace(i, &mut workspace.selected.decode)?
                .to_bits()
        {
            return Err("checked source values differ");
        }
    }
    let quote = source.checked_certification_retained_quote(&request)?;
    let coefficients = source
        .coefficients
        .take()
        .ok_or("checked coefficient owner consumed")?;
    let mut certificate = certify_with_coefficients(
        request,
        SourceImport::DeclarationOnly,
        Some(coefficients),
        context,
    )?;
    certificate.source_status = SourceImportStatus::PinnedSourceBytesChecked;
    let h = &mut workspace.certification_hash;
    h.reset();
    h.update(b"pllm.sigma.source-checked-linear-activation.v1\0");
    h.update(certificate.digest);
    h.update(source.source_binding());
    certificate.digest = h.finalize_reset().into();
    let result = SourceCheckedSigmaLinearActivation {
        certificate,
        source,
    };
    if result.checked_retained_resource_bytes()? != quote {
        return Err("retained source/certificate capacity differs from quote");
    }
    Ok(result)
}

#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;
