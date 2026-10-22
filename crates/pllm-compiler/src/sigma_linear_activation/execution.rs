//! Source-owned D22/local execution bridge.
//!
//! Incremental logical owner accounting, not borrowed-baseline or machine stack
//! accounting. Import/compiler admission precedes source hashing/copying. Native
//! admission grows the SAME guard before issuance; transfer never refunds it.
//! Native provenance remains LocalAssumption; no decoder or elevation claim.

use super::*;
use pllm_garble::masked_wire_reference::{matrix, typed};
use pllm_garble::shared_allocation::{
    AllocationContext, Budget, ExternalCharge, OwnedBudget, OwnedBudgetContext, OwnedExternalCharge,
};
use std::alloc::Layout;
use std::sync::{atomic::AtomicUsize, Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigmaExecutionError {
    BudgetIdentity,
    Resource,
    Source(&'static str),
    Native(typed::Error),
}

fn execution_sum(parts: impl IntoIterator<Item = usize>) -> Result<usize, SigmaExecutionError> {
    parts.into_iter().try_fold(0usize, |sum, bytes| {
        sum.checked_add(bytes).ok_or(SigmaExecutionError::Resource)
    })
}

mod storage_seal {
    pub trait Sealed {}
    impl Sealed for &super::Budget {}
    impl Sealed for super::OwnedBudgetContext {}
}

/// Storage specialization only. No source, issuance, numeric or transport authority.
/// Sealed to the original borrowed counter and the native owning context.
#[doc(hidden)]
pub trait SigmaExecutionContext: AllocationContext + storage_seal::Sealed {
    const OWNED: bool;
    type Root;
    type Charge;
    type Step;
    type RoleStep;
    fn frames_step(frames: [SigmaExecutionFrameFor<Self>; 2]) -> Self::Step;
    fn output_step(outputs: [SigmaExecutionOutputFor<Self>; 2]) -> Self::Step;
    fn role_frame_step(frame: roles::SigmaExecutionRoleFrameFor<Self>) -> Self::RoleStep;
    fn role_ready_step() -> Self::RoleStep;
    fn root_budget(root: &Self::Root) -> &Budget;
    fn reserve(root: &Self::Root, bytes: usize) -> Result<Self::Charge, typed::Error>;
    fn charge_is_for(charge: &Self::Charge, budget: &Budget) -> bool;
    fn payload(charge: &Self::Charge) -> usize;
    fn grow(charge: &mut Self::Charge, bytes: usize) -> Result<(), typed::Error>;
    fn issue(
        dealer: &mut typed::LocalDealerFor<Self>,
        spec: typed::LocalLinearActivationSpec,
        input: &[i64],
        coefficients: &[i64],
        charge: &mut Self::Charge,
    ) -> Result<
        (
            typed::DealerMaskFor<Self, typed::Ring>,
            typed::LocalLinearActivationRunFor<Self>,
        ),
        typed::Error,
    >;
    fn split(
        run: typed::LocalLinearActivationRunFor<Self>,
        charge: &mut Self::Charge,
    ) -> Result<roles::NativeRolesFor<Self>, typed::Error>;
}

impl<'b> SigmaExecutionContext for &'b Budget {
    const OWNED: bool = false;
    type Root = &'b Budget;
    type Charge = ExternalCharge<'b>;
    type Step = SigmaExecutionStep<'b>;
    type RoleStep = roles::SigmaExecutionRoleStep<'b>;
    fn frames_step(frames: [SigmaExecutionFrameFor<Self>; 2]) -> Self::Step {
        SigmaExecutionStep::Frames(frames)
    }
    fn output_step(outputs: [SigmaExecutionOutputFor<Self>; 2]) -> Self::Step {
        SigmaExecutionStep::Output(outputs)
    }
    fn role_frame_step(frame: roles::SigmaExecutionRoleFrameFor<Self>) -> Self::RoleStep {
        roles::SigmaExecutionRoleStep::Frame(frame)
    }
    fn role_ready_step() -> Self::RoleStep {
        roles::SigmaExecutionRoleStep::Ready
    }
    fn root_budget(root: &Self::Root) -> &Budget {
        root
    }
    fn reserve(root: &Self::Root, bytes: usize) -> Result<Self::Charge, typed::Error> {
        (*root).charge_external(bytes)
    }
    fn charge_is_for(charge: &Self::Charge, budget: &Budget) -> bool {
        charge.is_for(budget)
    }
    fn payload(charge: &Self::Charge) -> usize {
        charge.payload_bytes()
    }
    fn grow(charge: &mut Self::Charge, bytes: usize) -> Result<(), typed::Error> {
        charge.resize_payload(bytes)
    }
    fn issue(
        dealer: &mut typed::LocalDealerFor<Self>,
        spec: typed::LocalLinearActivationSpec,
        input: &[i64],
        coefficients: &[i64],
        charge: &mut Self::Charge,
    ) -> Result<
        (
            typed::DealerMaskFor<Self, typed::Ring>,
            typed::LocalLinearActivationRunFor<Self>,
        ),
        typed::Error,
    > {
        dealer.local_linear_activation_from_reserved_values(spec, input, coefficients, charge)
    }
    fn split(
        run: typed::LocalLinearActivationRunFor<Self>,
        charge: &mut Self::Charge,
    ) -> Result<roles::NativeRolesFor<Self>, typed::Error> {
        run.into_roles_from_reserved(charge)
    }
}

impl SigmaExecutionContext for OwnedBudgetContext {
    const OWNED: bool = true;
    type Root = OwnedBudget;
    type Charge = OwnedExternalCharge;
    type Step = OwnedSigmaExecutionStep;
    type RoleStep = roles::OwnedSigmaExecutionRoleStep;
    fn frames_step(frames: [SigmaExecutionFrameFor<Self>; 2]) -> Self::Step {
        SigmaExecutionStepFor::Frames(frames)
    }
    fn output_step(outputs: [SigmaExecutionOutputFor<Self>; 2]) -> Self::Step {
        SigmaExecutionStepFor::Output(outputs)
    }
    fn role_frame_step(frame: roles::SigmaExecutionRoleFrameFor<Self>) -> Self::RoleStep {
        roles::SigmaExecutionRoleStepFor::Frame(frame)
    }
    fn role_ready_step() -> Self::RoleStep {
        roles::SigmaExecutionRoleStepFor::Ready
    }
    fn root_budget(root: &Self::Root) -> &Budget {
        root.budget()
    }
    fn reserve(root: &Self::Root, bytes: usize) -> Result<Self::Charge, typed::Error> {
        root.charge_external(bytes)
    }
    fn charge_is_for(charge: &Self::Charge, budget: &Budget) -> bool {
        charge.is_for(budget)
    }
    fn payload(charge: &Self::Charge) -> usize {
        charge.payload_bytes()
    }
    fn grow(charge: &mut Self::Charge, bytes: usize) -> Result<(), typed::Error> {
        charge.resize_payload(bytes)
    }
    fn issue(
        dealer: &mut typed::LocalDealerFor<Self>,
        spec: typed::LocalLinearActivationSpec,
        input: &[i64],
        coefficients: &[i64],
        charge: &mut Self::Charge,
    ) -> Result<
        (
            typed::DealerMaskFor<Self, typed::Ring>,
            typed::LocalLinearActivationRunFor<Self>,
        ),
        typed::Error,
    > {
        dealer.local_linear_activation_from_reserved_values(spec, input, coefficients, charge)
    }
    fn split(
        run: typed::LocalLinearActivationRunFor<Self>,
        charge: &mut Self::Charge,
    ) -> Result<roles::NativeRolesFor<Self>, typed::Error> {
        run.into_roles_from_reserved(charge)
    }
}

/// Original transferred inputs drop before workspace and guard on every error.
/// No plan/inventory/borrowed buffer is cloned or retained by the returned owner.
struct SigmaCompilerAdmissionFor<'a, C: SigmaExecutionContext> {
    request: Option<SigmaLinearActivationRequest<'a>>,
    expected: Option<ExpectedSigmaSourcePin>,
    workspace: source::SigmaSourceWorkspace,
    charge: Option<C::Charge>,
}

/// Full concrete inline populations and nested owner quotes. Sequential phases
/// use max; resident caller copies are added, never substituted by callee quotes.
/// Whole shared quotes intentionally overcount resident workspace/spec fields
/// also embedded in SigmaProofWorkspace. This is conservative logical storage,
/// not a guessed allowance per node, native stack or dependency-private scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SigmaCompilerPhases {
    boundary_bytes: usize,
    source_commitment_bytes: usize,
    pairing_bytes: usize,
    proof_bytes: usize,
    certificate_return_bytes: usize,
    native_metadata_bytes: usize,
}

impl SigmaCompilerPhases {
    fn checked() -> Result<Self, SigmaExecutionError> {
        Self::checked_for::<&Budget>()
    }

    fn checked_for<C: SigmaExecutionContext>() -> Result<Self, SigmaExecutionError> {
        use pllm_core::edge_contract::EdgeContractResourceQuote;
        use pllm_types::bounded_canonical::{StreamError, StreamUsage};
        let model = model_canonical::resources::<DomainHash>(SIGMA_STREAM_DEPTH)
            .map_err(|_| SigmaExecutionError::Resource)?;
        let core =
            EdgeContractResourceQuote::checked().map_err(|_| SigmaExecutionError::Resource)?;
        let activation =
            ActivationSpec::checked_logical_resources().map_err(SigmaExecutionError::Native)?;
        let constructor = typed::LocalLinearActivationSpec::checked_logical_resources()
            .map_err(SigmaExecutionError::Native)?;
        let calculation =
            typed::LocalLinearActivationSpec::checked_resource_calculation_resources()
                .map_err(SigmaExecutionError::Native)?;
        // Quote records are concrete owned records, not the reservation values
        // described by those records. Count their entire retained population.
        let quote_records = execution_sum([
            size_of::<Self>(),
            size_of::<model_canonical::ResourceQuote>(),
            size_of::<EdgeContractResourceQuote>(),
            size_of::<typed::ActivationMetadataResources>(),
            size_of::<typed::LocalLinearActivationMetadataResources>(),
            size_of::<typed::LocalLinearActivationResourceCalculationResources>(),
            size_of::<typed::ActivationResourceCalculationResources>(),
            size_of::<SigmaSourceResources>(),
            size_of::<SigmaRetainedResourceInventory>(),
        ])?;
        // Sink admission and writer replay share frames but not live sessions.
        // Conservatively price both complete session/constructor/finish slots.
        // Emitter and formatting buffers are transient but coexist with Session.
        let stream = &model.stream;
        let schema = execution_sum([
            size_of::<SigmaStreamWorkspace<'static>>(),
            size_of::<DomainHash>(),
            size_of::<SigmaHashWriter<'static>>(),
            size_of::<SigmaSchema<'static>>(), // caller's retained Copy schema
            size_of::<SigmaSchema<'static>>(), // emitter argument
            size_of::<Session<'static, 'static, io::Sink>>(),
            size_of::<Result<Session<'static, 'static, io::Sink>, StreamError>>(),
            size_of::<Session<'static, 'static, DomainHash>>(),
            size_of::<Result<Session<'static, 'static, DomainHash>, StreamError>>(),
            size_of::<Result<StreamUsage, StreamError>>(),
            stream.emitter_bytes,
            stream.escape_buffer_bytes.max(execution_sum([
                stream.format_arguments_bytes,
                stream.compact_formatter_bytes,
            ])?),
            size_of::<[u8; 32]>(),
        ])?;
        let boundary_bytes = execution_sum([
            size_of::<SigmaCompilerAdmissionFor<'static, C>>(),
            size_of::<SigmaSourceBytes<'static>>(),
            // Taken request and pin move into helpers while their Option slots
            // remain in admission; no backing duplicates at these moves.
            size_of::<SigmaLinearActivationRequest<'static>>(),
            size_of::<ExpectedSigmaSourcePin>(),
            quote_records,
        ])?;
        let boundary_bytes = if C::OWNED {
            execution_sum([
                boundary_bytes,
                // Owned specialization additionally retains every guard constructor,
                // mapped Result and native issuance return slot. All are concrete
                // types, admitted before import, Arc allocation or issuance. Borrowed
                // compatibility keeps its original logical inventory unchanged.
                execution_sum([
                    size_of::<C::Charge>(),
                    size_of::<Result<C::Charge, typed::Error>>(),
                    size_of::<Result<C::Charge, SigmaExecutionError>>(),
                    size_of::<(SourceCheckedSigmaLinearActivation, C::Charge)>(),
                    size_of::<Result<(), typed::Error>>(),
                    size_of::<Result<(), SigmaExecutionError>>(),
                    size_of::<
                        Result<
                            (
                                typed::DealerMaskFor<C, typed::Ring>,
                                typed::LocalLinearActivationRunFor<C>,
                            ),
                            typed::Error,
                        >,
                    >(),
                    size_of::<
                        Result<
                            (
                                typed::DealerMaskFor<C, typed::Ring>,
                                typed::LocalLinearActivationRunFor<C>,
                            ),
                            SigmaExecutionError,
                        >,
                    >(),
                    size_of::<
                        Result<
                            (SigmaExecutionMaskFor<C>, SigmaExecutionRunFor<C>),
                            SigmaExecutionError,
                        >,
                    >(),
                    size_of::<Result<[typed::LocalLinearActivationFrameFor<C>; 2], typed::Error>>(),
                    size_of::<
                        Result<[typed::LocalLinearActivationFrameFor<C>; 2], SigmaExecutionError>,
                    >(),
                    size_of::<Result<[SigmaExecutionFrameFor<C>; 2], SigmaExecutionError>>(),
                    size_of::<Result<typed::LocalLinearActivationStepFor<C>, typed::Error>>(),
                    size_of::<Result<typed::LocalLinearActivationStepFor<C>, SigmaExecutionError>>(
                    ),
                    size_of::<Result<C::Step, SigmaExecutionError>>(),
                    size_of::<std::array::IntoIter<usize, 16>>(),
                ])?,
                size_of::<std::array::IntoIter<usize, 3>>(),
            ])?
        } else {
            boundary_bytes
        };
        // Shared source commitment owns these explicit reference-array iterators
        // (not their borrowed pointees). They are sequential; summing them is a
        // conservative addition to the source owner's initialization/move quote.
        let source_commitment_bytes = execution_sum([
            // checked_count's identity/coordinate tables and checked metadata
            // sums are also owned arrays; iterators over borrowed slices are not.
            size_of::<std::array::IntoIter<&'static String, 3>>(),
            size_of::<std::array::IntoIter<(&'static Vec<usize>, usize), 2>>(),
            size_of::<std::array::IntoIter<usize, 5>>(),
            size_of::<std::array::IntoIter<usize, 8>>(),
            size_of::<std::array::IntoIter<&'static String, 4>>(),
            size_of::<std::array::IntoIter<&'static Vec<usize>, 2>>(),
            size_of::<std::array::IntoIter<usize, 2>>(),
            size_of::<[u8; 8]>(),
            size_of::<[u8; 3]>(),
            size_of::<sha2::digest::Output<Sha256>>(),
            size_of::<Result<CheckedSigmaSource, &'static str>>(),
        ])?;
        let pairing_bytes = execution_sum([
            size_of::<SigmaCertificationContext<'static>>(),
            size_of::<Result<SigmaCertificationContext<'static>, &'static str>>(),
            size_of::<[u8; 64]>(), // constructor's plan_hex before context return
            size_of::<[u8; 32]>(), // named plan_identity
            model.hash_workspace_bytes,
            size_of::<Result<[u8; 32], &'static str>>(),
        ])?;
        let proof_bytes = execution_sum([
            size_of::<SigmaCertificationContext<'static>>(),
            size_of::<SigmaProofWorkspace>(), // original edges/spec + both workspaces
            model
                .hash_workspace_bytes
                .max(schema)
                .max(activation.peak_bytes)
                .max(core.descriptor_digest_bytes)
                .max(core.use_digest_bytes)
                .max(core.descriptor_construction_peak_bytes)
                .max(execution_sum([
                    size_of::<Sha256>(), // rescale stage hash helper
                    size_of::<sha2::digest::Output<Sha256>>(),
                    size_of::<[u8; 32]>(),
                ])?),
            // check_edge retains its Copy arguments while numeric/shape
            // constructors run; original descriptors stay in request/workspace.
            size_of::<EdgeDescriptor>(),
            size_of::<SourceIdentity>(),
            size_of::<[[u8; 32]; 2]>(),
            size_of::<[usize; 2]>(),
            // Numeric helper arguments and returned domains coexist with proof.
            size_of::<[IntegerInterval; 2]>(),
            size_of::<Result<(IntegerInterval, IntegerInterval), &'static str>>(),
            size_of::<[usize; 3]>(), // selected geometry
            size_of::<[usize; 2]>(), // original input
            size_of::<[usize; 2]>(), // original output
            size_of::<Option<SigmaCoefficients>>(),
            size_of::<Result<ActivationSpec, typed::Error>>(),
            // geometry's retained reference table, checked element-products
            // table, metadata-capacity sum and returned selected dimensions.
            size_of::<std::array::IntoIter<&'static String, 7>>(),
            size_of::<std::array::IntoIter<Option<usize>, 3>>(),
            size_of::<std::array::IntoIter<usize, 8>>(),
            size_of::<Result<([usize; 3], FixtureScope), &'static str>>(),
        ])?;
        let certificate_return_bytes = execution_sum([
            size_of::<SigmaCertificationContext<'static>>(),
            size_of::<SigmaProofWorkspace>(),
            size_of::<CheckedSigmaSource>(), // source local before final owner move
            size_of::<SigmaCoefficients>(),  // taken canonical backing owner
            size_of::<SigmaLinearActivationCertificate>(), // mutable commitment origin
            size_of::<Result<SigmaLinearActivationCertificate, &'static str>>(),
            size_of::<Result<SourceCheckedSigmaLinearActivation, &'static str>>(),
            size_of::<sha2::digest::Output<Sha256>>(),
            size_of::<[u8; 32]>(), // source-binding return
            size_of::<[u8; 32]>(), // commitment return
        ])?;
        let native_metadata_bytes = execution_sum([
            // Retained facade inventory separately includes original native spec,
            // resource record, Arc and every native/wrapper transition slot.
            constructor.peak_bytes.max(calculation.peak_bytes),
            // Native dealer preflight retains its caller's spec/cost while
            // resource calculation and descriptor/root validation run. Its edge
            // iterator OWNS five copied descriptors; root table owns references,
            // not input backings. These are additional to calculation's input.
            size_of::<typed::LocalLinearActivationSpec>(),
            size_of::<typed::LocalLinearActivationResources>(),
            size_of::<std::array::IntoIter<EdgeDescriptor, 5>>(),
            size_of::<std::iter::Enumerate<std::array::IntoIter<&'static [i64], 2>>>(),
            size_of::<EdgeDescriptor>(), // current validate argument
            size_of::<Numeric>(),        // current validate's named numeric
            size_of::<Result<typed::LocalLinearActivationSpec, SigmaExecutionError>>(),
            size_of::<Result<typed::LocalLinearActivationResources, typed::Error>>(),
            // Facade retained-quote sums: wrapper population (5), transition
            // population (10), payload (4), retained source (6), growth (2).
            size_of::<std::array::IntoIter<usize, 5>>(),
            size_of::<std::array::IntoIter<usize, 10>>(),
            size_of::<std::array::IntoIter<usize, 4>>(),
            size_of::<std::array::IntoIter<usize, 6>>(),
            size_of::<std::array::IntoIter<usize, 2>>(),
            size_of::<
                Result<(SigmaExecutionMaskFor<C>, SigmaExecutionRunFor<C>), SigmaExecutionError>,
            >(),
        ])?;
        Ok(Self {
            boundary_bytes,
            source_commitment_bytes,
            pairing_bytes,
            proof_bytes,
            certificate_return_bytes,
            native_metadata_bytes,
        })
    }

    fn peak_bytes(self) -> Result<usize, SigmaExecutionError> {
        execution_sum([
            self.boundary_bytes,
            self.source_commitment_bytes
                .max(self.pairing_bytes)
                .max(self.proof_bytes)
                .max(self.certificate_return_bytes)
                .max(self.native_metadata_bytes),
        ])
    }
}

/// Compiler-only incremental overhead. This pure size query neither hashes nor
/// authenticates inputs and is NOT standalone admission. The facade adds source
/// import/encoding scratch and all transferred/retained backings before reserving.
/// Preexisting borrowed plan/inventory/source/input/issuance slices and static
/// backing are excluded, consistent with shared model/core quote contracts.
pub fn sigma_execution_compiler_preflight(
    _plan: &DecoderPlan,
    _inventory: &DecoderMatrixInventory,
) -> Result<usize, SigmaExecutionError> {
    SigmaCompilerPhases::checked()?.peak_bytes()
}

/// Owned storage census; same compiler algorithm and borrowed caller baseline.
pub fn owned_sigma_execution_compiler_preflight(
    _plan: &DecoderPlan,
    _inventory: &DecoderMatrixInventory,
) -> Result<usize, SigmaExecutionError> {
    SigmaCompilerPhases::checked_for::<OwnedBudgetContext>()?.peak_bytes()
}

#[cfg(test)]
fn execution_payload_quote(
    expected: &ExpectedSigmaSourcePin,
    request: &SigmaLinearActivationRequest<'_>,
) -> Result<usize, SigmaExecutionError> {
    execution_payload_quote_for::<&Budget>(expected, request)
}

fn execution_payload_quote_for<C: SigmaExecutionContext>(
    expected: &ExpectedSigmaSourcePin,
    request: &SigmaLinearActivationRequest<'_>,
) -> Result<usize, SigmaExecutionError> {
    let import = expected
        .checked_import_quote()
        .map_err(SigmaExecutionError::Source)?;
    let source = expected
        .checked_certification_retained_quote(request)
        .map_err(SigmaExecutionError::Source)?;
    let retained =
        retained_resource_inventory_from_source_bytes_for::<C>(source)?.payload_bytes()?;
    execution_sum([
        // Final retained backings include pin and request selection capacities;
        // import owns only a subset, with no cloned source/coefficient backing.
        retained.max(import.retained_bytes),
        import.import_encoding_scratch_bytes,
        SigmaCompilerPhases::checked_for::<C>()?.peak_bytes()?,
    ])
}

/// Consuming pinned-source facade. Existing dealer construction entropy predates
/// this call. There is no compiler-injected entropy observer or zero-draw claim.
/// No caller spec, replacement coefficients, decoded handles or digest override.
#[allow(clippy::too_many_arguments)]
pub fn execute_sigma_linear_activation<'b>(
    expected: ExpectedSigmaSourcePin,
    supplied: SigmaSourceBytes<'_>,
    request: SigmaLinearActivationRequest<'_>,
    inventory: &DecoderMatrixInventory,
    budget: &'b Budget,
    dealer: &mut typed::LocalDealer<'b>,
    input: &[i64],
) -> Result<(SigmaExecutionMask<'b>, SigmaExecutionRun<'b>), SigmaExecutionError> {
    execute_with_context::<&'b Budget>(
        expected, supplied, request, inventory, &budget, dealer, input,
    )
}

/// Owned specialization of the same admitted pinned-source facade. No borrowed
/// owner is upgraded; every native owner starts in the original owning context.
#[allow(clippy::too_many_arguments)]
pub fn execute_owned_sigma_linear_activation(
    expected: ExpectedSigmaSourcePin,
    supplied: SigmaSourceBytes<'_>,
    request: SigmaLinearActivationRequest<'_>,
    inventory: &DecoderMatrixInventory,
    budget: &OwnedBudget,
    dealer: &mut typed::OwnedLocalDealer,
    input: &[i64],
) -> Result<(OwnedSigmaExecutionMask, OwnedSigmaExecutionRun), SigmaExecutionError> {
    execute_with_context::<OwnedBudgetContext>(
        expected, supplied, request, inventory, budget, dealer, input,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_with_context<C: SigmaExecutionContext>(
    expected: ExpectedSigmaSourcePin,
    supplied: SigmaSourceBytes<'_>,
    request: SigmaLinearActivationRequest<'_>,
    inventory: &DecoderMatrixInventory,
    root: &C::Root,
    dealer: &mut typed::LocalDealerFor<C>,
    input: &[i64],
) -> Result<(SigmaExecutionMaskFor<C>, SigmaExecutionRunFor<C>), SigmaExecutionError> {
    let budget = C::root_budget(root);
    if !dealer.is_budget(budget) {
        return Err(SigmaExecutionError::BudgetIdentity);
    }
    let payload = execution_payload_quote_for::<C>(&expected, &request)?;
    let charge = C::reserve(root, payload).map_err(SigmaExecutionError::Native)?;
    let (checked, charge) = {
        let mut admitted = SigmaCompilerAdmissionFor::<C> {
            request: Some(request),
            expected: Some(expected),
            workspace: source::SigmaSourceWorkspace::new(),
            charge: Some(charge),
        };
        let source = source::checked_import_sigma_source_bytes_with_workspace(
            admitted
                .expected
                .take()
                .ok_or(SigmaExecutionError::Resource)?,
            supplied,
            admitted
                .request
                .as_ref()
                .ok_or(SigmaExecutionError::Resource)?
                .encoded_private_coefficients,
            &mut admitted.workspace,
        )
        .map_err(SigmaExecutionError::Source)?;
        let checked = source::certify_source_checked_with_inventory_and_workspace(
            admitted
                .request
                .take()
                .ok_or(SigmaExecutionError::Resource)?,
            source,
            inventory,
            &mut admitted.workspace,
        )
        .map_err(SigmaExecutionError::Source)?;
        // Constructor and cost-record computation are admitted by the compiler
        // phase quote, not by the native reservation value they calculate.
        let native = {
            let spec = local_spec(&checked)?;
            typed::local_linear_activation_resources_for::<C>(spec)
                .map_err(SigmaExecutionError::Native)?
                .reservation_bytes()
        };
        let charge = admitted
            .charge
            .as_mut()
            .ok_or(SigmaExecutionError::Resource)?;
        // Growth is atomic and only positive. Failure leaves original guard
        // intact; checked/source drop before it. No source or scratch refund.
        C::grow(charge, execution_sum([payload, native])?).map_err(SigmaExecutionError::Native)?;
        (
            checked,
            admitted
                .charge
                .take()
                .ok_or(SigmaExecutionError::Resource)?,
        )
        // Admission's workspace and vacated input fields retire before transfer.
    };
    consume_checked_local_from_reserved_for::<C>(checked, budget, dealer, input, charge)
}

/// Source and coefficients erase before the last accounting guard refunds.
/// No public Arc/Weak, source owner, guard, native owner or lease escape exists.
struct RetainedSourceFor<C: SigmaExecutionContext> {
    checked: SourceCheckedSigmaLinearActivation,
    _charge: C::Charge,
}
#[cfg(test)]
type RetainedSource<'b> = RetainedSourceFor<&'b Budget>;

/// Explicitly local conditional execution, even when source bytes were checked.
/// This describes the bridge's provenance, not a native proof relabeling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigmaExecutionProvenance {
    PinnedSourceBytesWithLocalConditionalExecution,
}

/// One consuming local execution; its source and reservation cannot be cloned.
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::SigmaExecutionRun;
/// fn duplicate(run: SigmaExecutionRun<'_>) { let _ = run.clone(); }
/// ```
pub struct SigmaExecutionRunFor<C: SigmaExecutionContext> {
    native: Option<typed::LocalLinearActivationRunFor<C>>,
    retained: Arc<RetainedSourceFor<C>>,
}

#[path = "execution_roles.rs"]
mod roles;
pub use roles::{
    owned_sigma_execution_role_split_resources, sigma_execution_role_split_resources,
    OwnedSigmaExecutionRole, OwnedSigmaExecutionRoleCoordinator, OwnedSigmaExecutionRoleFrame,
    OwnedSigmaExecutionRoleOutput, OwnedSigmaExecutionRoleStep, SigmaExecutionRole,
    SigmaExecutionRoleCoordinator, SigmaExecutionRoleCoordinatorFor, SigmaExecutionRoleFor,
    SigmaExecutionRoleFrame, SigmaExecutionRoleFrameFor, SigmaExecutionRoleOutput,
    SigmaExecutionRoleOutputFor, SigmaExecutionRoleSplitResources, SigmaExecutionRoleStep,
    SigmaExecutionRoleStepFor,
};

pub struct SigmaExecutionFrameFor<C: SigmaExecutionContext> {
    native: typed::LocalLinearActivationFrameFor<C>,
    _retained: Arc<RetainedSourceFor<C>>,
}

// Keep issued owners inline; boxing would introduce an unpriced allocation.
#[allow(clippy::large_enum_variant)]
pub enum SigmaExecutionStepFor<C: SigmaExecutionContext> {
    Frames([SigmaExecutionFrameFor<C>; 2]),
    Output([SigmaExecutionOutputFor<C>; 2]),
}

// Preserve the borrowed enum namespace, including variant imports. Storage
// dispatch constructs wrappers directly, without an intermediate step owner.
#[allow(clippy::large_enum_variant)]
pub enum SigmaExecutionStep<'b> {
    Frames([SigmaExecutionFrame<'b>; 2]),
    Output([SigmaExecutionOutput<'b>; 2]),
}

/// Final masked capability; deliberately no extraction or decoding method.
/// Opaque native result retaining source ownership, not a relabeling authority.
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::SigmaExecutionOutput;
/// fn extract_native(output: SigmaExecutionOutput<'_>) { let _ = output.native; }
/// ```
pub struct SigmaExecutionOutputFor<C: SigmaExecutionContext> {
    native: typed::CommonMaskedFor<C, typed::Ring>,
    _retained: Arc<RetainedSourceFor<C>>,
}

/// Dealer output mask also retains the original source owner until retirement.
/// Separate mask owner; public certificate metadata cannot replace its backing.
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::SigmaExecutionMask;
/// fn duplicate(mask: SigmaExecutionMask<'_>) { let _ = mask.clone(); }
/// ```
pub struct SigmaExecutionMaskFor<C: SigmaExecutionContext> {
    _native: typed::DealerMaskFor<C, typed::Ring>,
    _retained: Arc<RetainedSourceFor<C>>,
}

pub type SigmaExecutionRun<'b> = SigmaExecutionRunFor<&'b Budget>;
pub type SigmaExecutionFrame<'b> = SigmaExecutionFrameFor<&'b Budget>;
pub type SigmaExecutionOutput<'b> = SigmaExecutionOutputFor<&'b Budget>;
pub type SigmaExecutionMask<'b> = SigmaExecutionMaskFor<&'b Budget>;
/// New owned execution only; borrowed native or compiler owners cannot upgrade.
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::{SigmaExecutionRun, OwnedSigmaExecutionRun};
/// fn upgrade(run: SigmaExecutionRun<'_>) -> OwnedSigmaExecutionRun { run }
/// ```
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRun;
/// fn extract(run: OwnedSigmaExecutionRun) { let _ = run.native; }
/// ```
///
/// The original source reservation is private too; no guard can be detached.
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRun;
/// fn extract_guard(run: OwnedSigmaExecutionRun) { let _ = run.retained._charge; }
/// ```
///
/// ```compile_fail,E0599
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRun;
/// fn duplicate(run: OwnedSigmaExecutionRun) { let _ = run.clone(); }
/// ```
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRun;
/// fn needs_send<T: Send>() {}
/// needs_send::<OwnedSigmaExecutionRun>();
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRun;
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<OwnedSigmaExecutionRun>();
/// ```
pub type OwnedSigmaExecutionRun = SigmaExecutionRunFor<OwnedBudgetContext>;
/// Source-retaining owned frame, neither cloneable nor transferable across threads.
///
/// ```compile_fail,E0599
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionFrame;
/// fn duplicate(frame: OwnedSigmaExecutionFrame) { let _ = frame.clone(); }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionFrame;
/// fn extract(frame: OwnedSigmaExecutionFrame) { let _ = frame.native; }
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionFrame;
/// fn needs_send<T: Send>() {}
/// needs_send::<OwnedSigmaExecutionFrame>();
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionFrame;
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<OwnedSigmaExecutionFrame>();
/// ```
///
/// ```compile_fail,E0308
/// use pllm_compiler::sigma_linear_activation::{SigmaExecutionFrame, OwnedSigmaExecutionFrame};
/// fn upgrade(frame: SigmaExecutionFrame<'_>) -> OwnedSigmaExecutionFrame { frame }
/// ```
pub type OwnedSigmaExecutionFrame = SigmaExecutionFrameFor<OwnedBudgetContext>;
pub type OwnedSigmaExecutionStep = SigmaExecutionStepFor<OwnedBudgetContext>;
/// Opaque output retains its original source, coefficients and counter.
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionOutput;
/// fn extract(output: OwnedSigmaExecutionOutput) { let _ = output.native; }
/// ```
///
/// ```compile_fail,E0599
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionOutput;
/// fn duplicate(output: OwnedSigmaExecutionOutput) { let _ = output.clone(); }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionOutput;
/// fn extract_guard(output: OwnedSigmaExecutionOutput) { let _ = output._retained._charge; }
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionOutput;
/// fn needs_send<T: Send>() {}
/// needs_send::<OwnedSigmaExecutionOutput>();
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionOutput;
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<OwnedSigmaExecutionOutput>();
/// ```
///
/// ```compile_fail,E0308
/// use pllm_compiler::sigma_linear_activation::{SigmaExecutionOutput, OwnedSigmaExecutionOutput};
/// fn upgrade(output: SigmaExecutionOutput<'_>) -> OwnedSigmaExecutionOutput { output }
/// ```
pub type OwnedSigmaExecutionOutput = SigmaExecutionOutputFor<OwnedBudgetContext>;
/// Single owning dealer mask; neither native mask nor source guard is extractable.
///
/// ```compile_fail,E0599
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionMask;
/// fn duplicate(mask: OwnedSigmaExecutionMask) { let _ = mask.clone(); }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionMask;
/// fn extract(mask: OwnedSigmaExecutionMask) { let _ = mask._native; }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionMask;
/// fn extract_guard(mask: OwnedSigmaExecutionMask) { let _ = mask._retained._charge; }
/// ```
///
/// Dealer-only masks retain their native auto-traits: unlike execution handles,
/// they contain no matrix singleton or party continuation. The private local
/// binding independently enforces process/thread affinity on its envelope.
///
/// ```
/// use pllm_compiler::sigma_linear_activation::{OwnedSigmaExecutionMask, SigmaExecutionMask};
/// fn needs_thread_safe<T: Send + Sync>() {}
/// needs_thread_safe::<OwnedSigmaExecutionMask>();
/// fn borrowed_mask<'b>() { needs_thread_safe::<SigmaExecutionMask<'b>>(); }
/// ```
///
/// ```compile_fail,E0308
/// use pllm_compiler::sigma_linear_activation::{SigmaExecutionMask, OwnedSigmaExecutionMask};
/// fn upgrade(mask: SigmaExecutionMask<'_>) -> OwnedSigmaExecutionMask { mask }
/// ```
pub type OwnedSigmaExecutionMask = SigmaExecutionMaskFor<OwnedBudgetContext>;

impl<C: SigmaExecutionContext> SigmaExecutionFrameFor<C> {
    pub fn as_bytes(&self) -> &[u8] {
        self.native.as_bytes()
    }
}

impl<C: SigmaExecutionContext> SigmaExecutionOutputFor<C> {
    pub fn descriptor(&self) -> EdgeDescriptor {
        self.native.descriptor()
    }
    pub fn provenance(&self) -> SigmaExecutionProvenance {
        SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution
    }
}

impl<C: SigmaExecutionContext> SigmaExecutionRunFor<C> {
    pub fn provenance(&self) -> SigmaExecutionProvenance {
        SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution
    }

    /// Cancelling burns both peers, but keeps source/metadata charge until drop.
    pub fn cancel(&mut self) {
        if let Some(native) = &mut self.native {
            native.cancel();
        }
    }

    pub fn begin(
        &mut self,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<[SigmaExecutionFrameFor<C>; 2], SigmaExecutionError> {
        let [left, right] = self
            .native
            .as_mut()
            .ok_or(SigmaExecutionError::Native(typed::Error::Consumed))?
            .begin(cancelled)
            .map_err(SigmaExecutionError::Native)?;
        Ok([self.frame(left), self.frame(right)])
    }

    /// Peer order is [party1, party0]. Opaque bytes remain unauthenticated local
    /// frames, not proof objects. Any error burns both native peers.
    pub fn advance(
        &mut self,
        peers: [&[u8]; 2],
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<C::Step, SigmaExecutionError> {
        let step = self
            .native
            .as_mut()
            .ok_or(SigmaExecutionError::Native(typed::Error::Consumed))?
            .advance(peers, cancelled)
            .map_err(SigmaExecutionError::Native)?;
        match step.into_frames() {
            Ok([left, right]) => Ok(C::frames_step([self.frame(left), self.frame(right)])),
            Err(step) => match step.into_output() {
                Ok([left, right]) => Ok(C::output_step([
                    SigmaExecutionOutputFor {
                        native: left,
                        _retained: Arc::clone(&self.retained),
                    },
                    SigmaExecutionOutputFor {
                        native: right,
                        _retained: Arc::clone(&self.retained),
                    },
                ])),
                Err(_) => {
                    self.cancel();
                    Err(SigmaExecutionError::Native(typed::Error::Context))
                }
            },
        }
    }

    fn frame(&self, native: typed::LocalLinearActivationFrameFor<C>) -> SigmaExecutionFrameFor<C> {
        SigmaExecutionFrameFor {
            native,
            _retained: Arc::clone(&self.retained),
        }
    }
}

impl<C: SigmaExecutionContext> Drop for SigmaExecutionRunFor<C> {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn local_spec(
    checked: &SourceCheckedSigmaLinearActivation,
) -> Result<typed::LocalLinearActivationSpec, SigmaExecutionError> {
    let c = checked.certificate();
    let &[rows, inner] = c.input_edge().shape().dimensions() else {
        return Err(SigmaExecutionError::Resource);
    };
    let &[columns, weight_inner] = c.private_weight_edge().shape().dimensions() else {
        return Err(SigmaExecutionError::Resource);
    };
    if weight_inner != inner
        || c.source_import_status() != SourceImportStatus::PinnedSourceBytesChecked
    {
        return Err(SigmaExecutionError::Resource);
    }
    typed::LocalLinearActivationSpec::new(
        matrix::Spec {
            bits: 50,
            rows,
            inner,
            columns,
            layout: matrix::MatrixLayout::OutputInput,
        },
        [
            c.input_edge(),
            c.private_weight_edge(),
            c.complete_sum_edge(),
            c.rescaled_edge(),
            c.activation_output_edge(),
        ],
        [
            c.complete_sum_edge().identity().producer(),
            c.rescaled_edge().identity().producer(),
            c.activation_output_edge().identity().producer(),
            c.next_use().consumer(),
        ],
        c.activation_spec().kind(),
        [
            c.input_domain()
                .checked_half_open_endpoints()
                .map_err(|_| SigmaExecutionError::Resource)?,
            c.weight_domain()
                .checked_half_open_endpoints()
                .map_err(|_| SigmaExecutionError::Resource)?,
        ],
    )
    .map_err(SigmaExecutionError::Native)
}

/// Logical Arc backing including both reference counters and alignment padding.
fn arc_backing_bytes_for<C: SigmaExecutionContext>() -> Result<usize, SigmaExecutionError> {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<RetainedSourceFor<C>>())
        .map(|(layout, _)| layout.pad_to_align().size())
        .map_err(|_| SigmaExecutionError::Resource)
}

/// Fixed facade inventory: one run/mask, 22 outgoing frames (11 exchanges),
/// 11 returned steps and two outputs. Charge full wrapper sizes conservatively,
/// including their native inline fields already covered by the native quote.
/// Named transition values below cover the single issuance/advance overlap.
/// This is NOT an allowance for inventory lowering, serde or borrowed buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SigmaRetainedResourceInventory {
    source_owner_and_backing_bytes: usize,
    arc_control_and_padding_bytes: usize,
    wrapper_population_bytes: usize,
    issuance_transition_bytes: usize,
}

impl SigmaRetainedResourceInventory {
    fn payload_bytes(self) -> Result<usize, SigmaExecutionError> {
        [
            self.source_owner_and_backing_bytes,
            self.arc_control_and_padding_bytes,
            self.wrapper_population_bytes,
            self.issuance_transition_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(SigmaExecutionError::Resource)
        })
    }
}

#[cfg(test)]
fn retained_resource_inventory(
    checked: &SourceCheckedSigmaLinearActivation,
) -> Result<SigmaRetainedResourceInventory, SigmaExecutionError> {
    retained_resource_inventory_for::<&Budget>(checked)
}

fn retained_resource_inventory_for<C: SigmaExecutionContext>(
    checked: &SourceCheckedSigmaLinearActivation,
) -> Result<SigmaRetainedResourceInventory, SigmaExecutionError> {
    let source = checked
        .checked_retained_resource_bytes()
        .map_err(|_| SigmaExecutionError::Resource)?;
    retained_resource_inventory_from_source_bytes_for::<C>(source)
}

/// Same conservative retained inventory from the source owner's PRE-import
/// certification-retained quote. No source/spec construction is needed. This
/// helper alone does not price compiler phases or native quote computation and
/// cannot authorize/reserve pre-import execution.
#[cfg(test)]
fn retained_resource_inventory_from_source_bytes(
    source: usize,
) -> Result<SigmaRetainedResourceInventory, SigmaExecutionError> {
    retained_resource_inventory_from_source_bytes_for::<&Budget>(source)
}

fn retained_resource_inventory_from_source_bytes_for<C: SigmaExecutionContext>(
    source: usize,
) -> Result<SigmaRetainedResourceInventory, SigmaExecutionError> {
    let arc = arc_backing_bytes_for::<C>()?;
    let wrapper_population_bytes = [
        size_of::<SigmaExecutionRunFor<C>>(),
        size_of::<SigmaExecutionMaskFor<C>>(),
        22 * size_of::<SigmaExecutionFrameFor<C>>(),
        11 * size_of::<C::Step>(),
        2 * size_of::<SigmaExecutionOutputFor<C>>(),
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or(SigmaExecutionError::Resource)
    })?;
    let issuance_transition_bytes = [
        // Stack owner while moving into Arc backing (counted separately above).
        size_of::<RetainedSourceFor<C>>(),
        size_of::<typed::LocalLinearActivationSpec>(),
        size_of::<typed::LocalLinearActivationResources>(),
        size_of::<(
            typed::DealerMaskFor<C, typed::Ring>,
            typed::LocalLinearActivationRunFor<C>,
        )>(),
        size_of::<typed::LocalLinearActivationStepFor<C>>(),
        size_of::<
            Result<
                [typed::LocalLinearActivationFrameFor<C>; 2],
                typed::LocalLinearActivationStepFor<C>,
            >,
        >(),
        size_of::<
            Result<
                [typed::CommonMaskedFor<C, typed::Ring>; 2],
                typed::LocalLinearActivationStepFor<C>,
            >,
        >(),
        2 * size_of::<typed::LocalLinearActivationFrameFor<C>>(),
        2 * size_of::<typed::CommonMaskedFor<C, typed::Ring>>(),
        3 * size_of::<Arc<RetainedSourceFor<C>>>(),
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or(SigmaExecutionError::Resource)
    })?;
    // Source quote already counts the inline checked owner. ExternalCharge
    // charges its own metadata; add only Arc counters/alignment, not it twice.
    Ok(SigmaRetainedResourceInventory {
        source_owner_and_backing_bytes: source,
        arc_control_and_padding_bytes: arc
            .checked_sub(size_of::<SourceCheckedSigmaLinearActivation>() + size_of::<C::Charge>())
            .ok_or(SigmaExecutionError::Resource)?,
        wrapper_population_bytes,
        issuance_transition_bytes,
    })
}

#[cfg(test)]
fn retained_payload_bytes(
    checked: &SourceCheckedSigmaLinearActivation,
) -> Result<usize, SigmaExecutionError> {
    retained_resource_inventory(checked)?.payload_bytes()
}

/// Real consuming bridge, intentionally PRIVATE. Its incoming certificate was
/// already allocated by its caller; this only admits its retained lifetime and
/// subsequent Arc/native allocations. It cannot establish pre-import admission.
/// Test-only transfer harness; not an alternative raw-source admission entry.
#[cfg(test)]
fn consume_checked_local<'b>(
    checked: SourceCheckedSigmaLinearActivation,
    budget: &'b Budget,
    dealer: &mut typed::LocalDealer<'b>,
    input: &[i64],
) -> Result<(SigmaExecutionMask<'b>, SigmaExecutionRun<'b>), SigmaExecutionError> {
    if !dealer.is_budget(budget) {
        return Err(SigmaExecutionError::BudgetIdentity);
    }
    let spec = local_spec(&checked)?;
    let payload = retained_payload_bytes(&checked)?;
    let native_bytes = typed::local_linear_activation_resources(spec)
        .map_err(SigmaExecutionError::Native)?
        .reservation_bytes();
    let overlap = payload
        .checked_add(native_bytes)
        .ok_or(SigmaExecutionError::Resource)?;
    let charge = budget
        .charge_external(overlap)
        .map_err(SigmaExecutionError::Native)?;
    consume_checked_local_from_reserved(checked, budget, dealer, input, charge)
}

/// Continuation of the ORIGINAL source/compiler reservation. No shrink, refund
/// or reacquire occurs: native transfers its slice, source retains all remainder
/// (including conservatively retained compiler scratch capacity) until final drop.
/// Raw facade supplies its compiler/import charge plus exact native reservation.
#[cfg(test)]
fn consume_checked_local_from_reserved<'b>(
    checked: SourceCheckedSigmaLinearActivation,
    budget: &'b Budget,
    dealer: &mut typed::LocalDealer<'b>,
    input: &[i64],
    charge: ExternalCharge<'b>,
) -> Result<(SigmaExecutionMask<'b>, SigmaExecutionRun<'b>), SigmaExecutionError> {
    consume_checked_local_from_reserved_for::<&'b Budget>(checked, budget, dealer, input, charge)
}

fn consume_checked_local_from_reserved_for<C: SigmaExecutionContext>(
    checked: SourceCheckedSigmaLinearActivation,
    budget: &Budget,
    dealer: &mut typed::LocalDealerFor<C>,
    input: &[i64],
    charge: C::Charge,
) -> Result<(SigmaExecutionMaskFor<C>, SigmaExecutionRunFor<C>), SigmaExecutionError> {
    // A stack owner enforces source-erasure-before-refund on EVERY error path,
    // including wrong-budget/shape rejection before allocating Arc backing.
    let owner = RetainedSourceFor::<C> {
        checked,
        _charge: charge,
    };
    if !dealer.is_budget(budget) || !C::charge_is_for(&owner._charge, budget) {
        return Err(SigmaExecutionError::BudgetIdentity);
    }
    let spec = local_spec(&owner.checked)?;
    let payload = retained_resource_inventory_for::<C>(&owner.checked)?.payload_bytes()?;
    let native_bytes = typed::local_linear_activation_resources_for::<C>(spec)
        .map_err(SigmaExecutionError::Native)?
        .reservation_bytes();
    if payload
        .checked_add(native_bytes)
        .ok_or(SigmaExecutionError::Resource)?
        > C::payload(&owner._charge)
    {
        return Err(SigmaExecutionError::Resource);
    }
    // Reject geometry/domain before Arc creation and before entering the dealer.
    // Preexisting borrowed input construction/backing is outside this boundary.
    let c = owner.checked.certificate();
    let expected_elements = c
        .input_edge()
        .shape()
        .dimensions()
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .ok_or(SigmaExecutionError::Resource)?;
    let (lower, upper) = c.input_domain().endpoints();
    if input.len() != expected_elements
        || input
            .iter()
            .any(|&v| i128::from(v) < lower || i128::from(v) > upper)
    {
        return Err(SigmaExecutionError::Resource);
    }
    // Reserve before Arc allocation. No source/coefficients are cloned.
    let mut retained = Arc::new(owner);
    // Transfer the exact native slice from the original aggregate charge. No
    // decrement/reacquire window permits another operation to steal admission.
    let owned = Arc::get_mut(&mut retained).ok_or(SigmaExecutionError::Resource)?;
    let (mask, native) = C::issue(
        dealer,
        spec,
        input,
        owned.checked.certificate().encoded_private_coefficients(),
        &mut owned._charge,
    )
    .map_err(SigmaExecutionError::Native)?;
    Ok((
        SigmaExecutionMaskFor {
            _native: mask,
            _retained: Arc::clone(&retained),
        },
        SigmaExecutionRunFor {
            native: Some(native),
            retained,
        },
    ))
}

#[cfg(test)]
static TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
#[path = "execution_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "execution_owned_tests.rs"]
mod owned_tests;

#[cfg(test)]
#[path = "execution_owned_audit_tests.rs"]
mod owned_audit_tests;
