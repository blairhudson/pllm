//! Independent owned-storage acceptance census. Mount as a child of execution.
//!
//! Dependency-owned quotes are authoritative only for their own scratch. Every
//! compiler-owned slot below is enumerated from its actual concrete type; no
//! compiler total, borrowed size, or guessed spare allowance supplies a term.
use super::super::tests::plan;
use super::owned_tests::{dealer, issue};
use super::tests::{checked, fixture_request, pin};
use super::*;
use pllm_core::edge_contract::EdgeContractResourceQuote;
use pllm_types::bounded_canonical::{StreamError, StreamUsage};

type C = OwnedBudgetContext;
type SourceOwner = RetainedSourceFor<C>;
type NativeIssue = (
    typed::DealerMaskFor<C, typed::Ring>,
    typed::LocalLinearActivationRunFor<C>,
);
type CompilerIssue = (OwnedSigmaExecutionMask, OwnedSigmaExecutionRun);
type CompilerRoles = (
    OwnedSigmaExecutionRoleCoordinator,
    [OwnedSigmaExecutionRole; 2],
);

pub(super) fn total<const N: usize>(slots: [usize; N]) -> usize {
    slots.iter().fold(0usize, |sum, bytes| {
        sum.checked_add(*bytes).expect("audit census overflow")
    })
}

fn owned_boundary_extra() -> usize {
    total([
        size_of::<OwnedExternalCharge>(),
        size_of::<Result<OwnedExternalCharge, typed::Error>>(),
        size_of::<Result<OwnedExternalCharge, SigmaExecutionError>>(),
        size_of::<(SourceCheckedSigmaLinearActivation, OwnedExternalCharge)>(),
        size_of::<Result<(), typed::Error>>(),
        size_of::<Result<(), SigmaExecutionError>>(),
        size_of::<Result<NativeIssue, typed::Error>>(),
        size_of::<Result<NativeIssue, SigmaExecutionError>>(),
        size_of::<Result<CompilerIssue, SigmaExecutionError>>(),
        size_of::<Result<[typed::LocalLinearActivationFrameFor<C>; 2], typed::Error>>(),
        size_of::<Result<[typed::LocalLinearActivationFrameFor<C>; 2], SigmaExecutionError>>(),
        size_of::<Result<[OwnedSigmaExecutionFrame; 2], SigmaExecutionError>>(),
        size_of::<Result<typed::LocalLinearActivationStepFor<C>, typed::Error>>(),
        size_of::<Result<typed::LocalLinearActivationStepFor<C>, SigmaExecutionError>>(),
        size_of::<Result<OwnedSigmaExecutionStep, SigmaExecutionError>>(),
        size_of::<std::array::IntoIter<usize, 16>>(),
        size_of::<std::array::IntoIter<usize, 3>>(),
    ])
}

fn independently_enumerated_phases() -> SigmaCompilerPhases {
    let model = model_canonical::resources::<DomainHash>(SIGMA_STREAM_DEPTH).unwrap();
    let core = EdgeContractResourceQuote::checked().unwrap();
    let activation = ActivationSpec::checked_logical_resources().unwrap();
    let constructor = typed::LocalLinearActivationSpec::checked_logical_resources().unwrap();
    let calculation =
        typed::LocalLinearActivationSpec::checked_resource_calculation_resources().unwrap();

    let quote_records = total([
        size_of::<SigmaCompilerPhases>(),
        size_of::<model_canonical::ResourceQuote>(),
        size_of::<EdgeContractResourceQuote>(),
        size_of::<typed::ActivationMetadataResources>(),
        size_of::<typed::LocalLinearActivationMetadataResources>(),
        size_of::<typed::LocalLinearActivationResourceCalculationResources>(),
        size_of::<typed::ActivationResourceCalculationResources>(),
        size_of::<SigmaSourceResources>(),
        size_of::<SigmaRetainedResourceInventory>(),
    ]);
    // Sink preflight and hash replay do not share session/constructor slots.
    // Retained schema and emitter argument are two separate Copy values.
    let schema = total([
        size_of::<SigmaStreamWorkspace<'static>>(),
        size_of::<DomainHash>(),
        size_of::<SigmaHashWriter<'static>>(),
        size_of::<SigmaSchema<'static>>(),
        size_of::<SigmaSchema<'static>>(),
        size_of::<Session<'static, 'static, io::Sink>>(),
        size_of::<Result<Session<'static, 'static, io::Sink>, StreamError>>(),
        size_of::<Session<'static, 'static, DomainHash>>(),
        size_of::<Result<Session<'static, 'static, DomainHash>, StreamError>>(),
        size_of::<Result<StreamUsage, StreamError>>(),
        model.stream.emitter_bytes,
        model.stream.escape_buffer_bytes.max(total([
            model.stream.format_arguments_bytes,
            model.stream.compact_formatter_bytes,
        ])),
        size_of::<[u8; 32]>(),
    ]);
    let boundary_bytes = total([
        size_of::<SigmaCompilerAdmissionFor<'static, C>>(),
        size_of::<SigmaSourceBytes<'static>>(),
        size_of::<SigmaLinearActivationRequest<'static>>(),
        size_of::<ExpectedSigmaSourcePin>(),
        quote_records,
        owned_boundary_extra(),
    ]);
    let source_commitment_bytes = total([
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
    ]);
    let pairing_bytes = total([
        size_of::<SigmaCertificationContext<'static>>(),
        size_of::<Result<SigmaCertificationContext<'static>, &'static str>>(),
        size_of::<[u8; 64]>(),
        size_of::<[u8; 32]>(),
        model.hash_workspace_bytes,
        size_of::<Result<[u8; 32], &'static str>>(),
    ]);
    let rescale_hash = total([
        size_of::<Sha256>(),
        size_of::<sha2::digest::Output<Sha256>>(),
        size_of::<[u8; 32]>(),
    ]);
    let nested_proof_peak = [
        model.hash_workspace_bytes,
        schema,
        activation.peak_bytes,
        core.descriptor_digest_bytes,
        core.use_digest_bytes,
        core.descriptor_construction_peak_bytes,
        rescale_hash,
    ]
    .into_iter()
    .max()
    .unwrap();
    let proof_bytes = total([
        size_of::<SigmaCertificationContext<'static>>(),
        size_of::<SigmaProofWorkspace>(),
        nested_proof_peak,
        size_of::<EdgeDescriptor>(),
        size_of::<SourceIdentity>(),
        size_of::<[[u8; 32]; 2]>(),
        size_of::<[usize; 2]>(),
        size_of::<[IntegerInterval; 2]>(),
        size_of::<Result<(IntegerInterval, IntegerInterval), &'static str>>(),
        size_of::<[usize; 3]>(),
        size_of::<[usize; 2]>(),
        size_of::<[usize; 2]>(),
        size_of::<Option<SigmaCoefficients>>(),
        size_of::<Result<ActivationSpec, typed::Error>>(),
        size_of::<std::array::IntoIter<&'static String, 7>>(),
        size_of::<std::array::IntoIter<Option<usize>, 3>>(),
        size_of::<std::array::IntoIter<usize, 8>>(),
        size_of::<Result<([usize; 3], FixtureScope), &'static str>>(),
    ]);
    let certificate_return_bytes = total([
        size_of::<SigmaCertificationContext<'static>>(),
        size_of::<SigmaProofWorkspace>(),
        size_of::<CheckedSigmaSource>(),
        size_of::<SigmaCoefficients>(),
        size_of::<SigmaLinearActivationCertificate>(),
        size_of::<Result<SigmaLinearActivationCertificate, &'static str>>(),
        size_of::<Result<SourceCheckedSigmaLinearActivation, &'static str>>(),
        size_of::<sha2::digest::Output<Sha256>>(),
        size_of::<[u8; 32]>(),
        size_of::<[u8; 32]>(),
    ]);
    let native_metadata_bytes = total([
        constructor.peak_bytes.max(calculation.peak_bytes),
        size_of::<typed::LocalLinearActivationSpec>(),
        size_of::<typed::LocalLinearActivationResources>(),
        size_of::<std::array::IntoIter<EdgeDescriptor, 5>>(),
        size_of::<std::iter::Enumerate<std::array::IntoIter<&'static [i64], 2>>>(),
        size_of::<EdgeDescriptor>(),
        size_of::<Numeric>(),
        size_of::<Result<typed::LocalLinearActivationSpec, SigmaExecutionError>>(),
        size_of::<Result<typed::LocalLinearActivationResources, typed::Error>>(),
        size_of::<std::array::IntoIter<usize, 5>>(),
        size_of::<std::array::IntoIter<usize, 10>>(),
        size_of::<std::array::IntoIter<usize, 4>>(),
        size_of::<std::array::IntoIter<usize, 6>>(),
        size_of::<std::array::IntoIter<usize, 2>>(),
        size_of::<Result<CompilerIssue, SigmaExecutionError>>(),
    ]);
    SigmaCompilerPhases {
        boundary_bytes,
        source_commitment_bytes,
        pairing_bytes,
        proof_bytes,
        certificate_return_bytes,
        native_metadata_bytes,
    }
}

fn independent_peak(phases: SigmaCompilerPhases) -> usize {
    total([
        phases.boundary_bytes,
        [
            phases.source_commitment_bytes,
            phases.pairing_bytes,
            phases.proof_bytes,
            phases.certificate_return_bytes,
            phases.native_metadata_bytes,
        ]
        .into_iter()
        .max()
        .unwrap(),
    ])
}

#[test]
fn owned_audit_every_compiler_phase_has_its_concrete_population() {
    let expected = independently_enumerated_phases();
    let actual = SigmaCompilerPhases::checked_for::<C>().unwrap();
    assert_eq!(actual.boundary_bytes, expected.boundary_bytes, "boundary");
    assert_eq!(
        actual.source_commitment_bytes, expected.source_commitment_bytes,
        "source commitment"
    );
    assert_eq!(actual.pairing_bytes, expected.pairing_bytes, "pairing");
    assert_eq!(actual.proof_bytes, expected.proof_bytes, "proof");
    assert_eq!(
        actual.certificate_return_bytes, expected.certificate_return_bytes,
        "certificate return"
    );
    assert_eq!(
        actual.native_metadata_bytes, expected.native_metadata_bytes,
        "native metadata"
    );
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    assert_eq!(actual.peak_bytes().unwrap(), independent_peak(expected));
    assert_eq!(
        owned_sigma_execution_compiler_preflight(&p, &inventory).unwrap(),
        independent_peak(expected)
    );

    // Storage specialization must not silently alter the immutable borrowed
    // algorithm's source/numeric/hash inventories. Only concrete storage slots
    // and the explicitly enumerated owned constructor/Result population differ.
    let borrowed = SigmaCompilerPhases::checked().unwrap();
    assert_eq!(
        actual.source_commitment_bytes,
        borrowed.source_commitment_bytes
    );
    assert_eq!(actual.pairing_bytes, borrowed.pairing_bytes);
    assert_eq!(actual.proof_bytes, borrowed.proof_bytes);
    assert_eq!(
        actual.certificate_return_bytes,
        borrowed.certificate_return_bytes
    );
    assert_eq!(
        actual.boundary_bytes + size_of::<SigmaCompilerAdmissionFor<'static, &'static Budget>>(),
        borrowed.boundary_bytes
            + size_of::<SigmaCompilerAdmissionFor<'static, C>>()
            + owned_boundary_extra()
    );
    assert_eq!(
        actual.native_metadata_bytes
            + size_of::<
                Result<
                    (SigmaExecutionMask<'static>, SigmaExecutionRun<'static>),
                    SigmaExecutionError,
                >,
            >(),
        borrowed.native_metadata_bytes + size_of::<Result<CompilerIssue, SigmaExecutionError>>()
    );
}

fn independent_retained_inventory(source: usize) -> SigmaRetainedResourceInventory {
    let backing = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<SourceOwner>())
        .unwrap()
        .0
        .pad_to_align()
        .size();
    SigmaRetainedResourceInventory {
        source_owner_and_backing_bytes: source,
        // Source owner quote and external guard already price their own inline
        // metadata. Only genuine Arc counters/alignment are added here.
        arc_control_and_padding_bytes: backing
            - size_of::<SourceCheckedSigmaLinearActivation>()
            - size_of::<OwnedExternalCharge>(),
        wrapper_population_bytes: total([
            size_of::<OwnedSigmaExecutionRun>(),
            size_of::<OwnedSigmaExecutionMask>(),
            22 * size_of::<OwnedSigmaExecutionFrame>(),
            11 * size_of::<OwnedSigmaExecutionStep>(),
            2 * size_of::<OwnedSigmaExecutionOutput>(),
        ]),
        issuance_transition_bytes: total([
            size_of::<SourceOwner>(),
            size_of::<typed::LocalLinearActivationSpec>(),
            size_of::<typed::LocalLinearActivationResources>(),
            size_of::<NativeIssue>(),
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
            3 * size_of::<Arc<SourceOwner>>(),
        ]),
    }
}

fn independent_retained_payload(source: usize) -> usize {
    let census = independent_retained_inventory(source);
    total([
        census.source_owner_and_backing_bytes,
        census.arc_control_and_padding_bytes,
        census.wrapper_population_bytes,
        census.issuance_transition_bytes,
    ])
}

pub(super) fn independent_execution_payload(
    expected: &ExpectedSigmaSourcePin,
    request: &SigmaLinearActivationRequest<'_>,
) -> usize {
    let source = expected
        .checked_certification_retained_quote(request)
        .unwrap();
    let imported = expected.checked_import_quote().unwrap();
    total([
        independent_retained_payload(source).max(imported.retained_bytes),
        imported.import_encoding_scratch_bytes,
        independent_peak(independently_enumerated_phases()),
    ])
}

#[test]
fn owned_audit_retained_arc_and_every_native_facade_transition_use_original_source_quote() {
    let p = plan();
    let request = fixture_request(&p, &[], &[]);
    let expected_pin = pin();
    let source_quote = expected_pin
        .checked_certification_retained_quote(&request)
        .unwrap();
    let c = checked();
    assert_eq!(source_quote, c.checked_retained_resource_bytes().unwrap());
    let expected = independent_retained_inventory(source_quote);
    assert_eq!(
        retained_resource_inventory_from_source_bytes_for::<C>(source_quote).unwrap(),
        expected
    );
    assert_eq!(retained_resource_inventory_for::<C>(&c).unwrap(), expected);
    assert_eq!(
        retained_resource_inventory_for::<C>(&c)
            .unwrap()
            .payload_bytes()
            .unwrap(),
        independent_retained_payload(source_quote)
    );
    assert_eq!(
        execution_payload_quote_for::<C>(&expected_pin, &request).unwrap(),
        independent_execution_payload(&expected_pin, &request)
    );
    assert!(
        retained_resource_inventory_from_source_bytes_for::<C>(usize::MAX)
            .unwrap()
            .payload_bytes()
            .is_err()
    );
}

#[test]
fn owned_audit_borrowed_backing_excluded_but_transferred_request_capacity_is_not() {
    let p = plan();
    let expected_pin = pin();
    let mut request = fixture_request(&p, &[], &[]);
    let before_source = expected_pin
        .checked_certification_retained_quote(&request)
        .unwrap();
    let before_payload = execution_payload_quote_for::<C>(&expected_pin, &request).unwrap();
    let metadata = request.selection.checked_metadata_backing_bytes().unwrap();
    let unchanged_selection = request.selection.clone();
    request.selection.input_row_coordinates.reserve(37);
    assert_eq!(request.selection, unchanged_selection);
    let capacity_delta = request.selection.checked_metadata_backing_bytes().unwrap() - metadata;
    assert!(capacity_delta > 0);
    let after_source = expected_pin
        .checked_certification_retained_quote(&request)
        .unwrap();
    assert_eq!(after_source, before_source + capacity_delta);
    assert_eq!(
        execution_payload_quote_for::<C>(&expected_pin, &request).unwrap(),
        before_payload + capacity_delta
    );
    assert_eq!(
        independent_retained_payload(after_source),
        independent_retained_payload(before_source) + capacity_delta
    );
    let mut larger_borrowed_plan = p.clone();
    larger_borrowed_plan.model_family.reserve(1 << 20);
    assert_eq!(larger_borrowed_plan, p);
    assert_eq!(
        execution_payload_quote_for::<C>(
            &expected_pin,
            &fixture_request(&larger_borrowed_plan, &[], &[])
        )
        .unwrap(),
        before_payload
    );
}

// The roles-child audit adds private admission/retention values, four concrete
// Arc handles and the separately allocated retained-role backing to this census.
pub(super) fn accessible_role_transitions() -> usize {
    total([
        size_of::<SigmaExecutionRoleSplitResources>(),
        size_of::<Result<SigmaExecutionRoleSplitResources, SigmaExecutionError>>(),
        size_of::<typed::LocalLinearActivationRoleSplitResources>(),
        size_of::<OwnedSigmaExecutionRun>(),
        size_of::<typed::LocalLinearActivationRunFor<C>>(),
        size_of::<roles::NativeRolesFor<C>>(),
        size_of::<Result<roles::NativeRolesFor<C>, typed::Error>>(),
        size_of::<CompilerRoles>(),
        size_of::<Result<CompilerRoles, SigmaExecutionError>>(),
        size_of::<Result<typed::LocalLinearActivationRoleFrameFor<C>, typed::Error>>(),
        size_of::<Result<typed::LocalLinearActivationRoleStepFor<C>, typed::Error>>(),
        size_of::<Result<typed::LocalLinearActivationRoleOutputFor<C>, typed::Error>>(),
        size_of::<Result<OwnedSigmaExecutionRoleFrame, SigmaExecutionError>>(),
        size_of::<Result<OwnedSigmaExecutionRoleStep, SigmaExecutionError>>(),
        size_of::<Result<OwnedSigmaExecutionRoleOutput, SigmaExecutionError>>(),
        size_of::<Result<EdgeDescriptor, SigmaExecutionError>>(),
        size_of::<Arc<SourceOwner>>(),
        size_of::<std::array::IntoIter<usize, 5>>(),
        size_of::<std::array::IntoIter<usize, 22>>(),
        size_of::<OwnedExternalCharge>(),
        size_of::<Result<OwnedExternalCharge, typed::Error>>(),
        size_of::<Result<OwnedExternalCharge, SigmaExecutionError>>(),
        size_of::<std::array::IntoIter<usize, 4>>(),
        size_of::<std::array::IntoIter<usize, 3>>(),
    ])
}

pub(super) fn independent_role_wrapper_population() -> usize {
    total([
        size_of::<OwnedSigmaExecutionRoleCoordinator>(),
        2 * size_of::<OwnedSigmaExecutionRole>(),
        22 * size_of::<OwnedSigmaExecutionRoleFrame>(),
        22 * size_of::<OwnedSigmaExecutionRoleStep>(),
        2 * size_of::<OwnedSigmaExecutionRoleOutput>(),
    ])
}

#[test]
fn owned_audit_role_split_accessible_concrete_census_and_native_owner_quote() {
    let quote = owned_sigma_execution_role_split_resources().unwrap();
    let native = typed::local_linear_activation_role_split_resources_for::<C>().unwrap();
    let native_wrappers = total([
        size_of::<typed::LocalLinearActivationRoleCoordinatorFor<C>>(),
        2 * size_of::<typed::LocalLinearActivationRoleFor<C>>(),
        22 * size_of::<typed::LocalLinearActivationRoleFrameFor<C>>(),
        22 * size_of::<typed::LocalLinearActivationRoleStepFor<C>>(),
        2 * size_of::<typed::LocalLinearActivationRoleOutputFor<C>>(),
    ]);
    assert_eq!(native.wrapper_population_bytes, native_wrappers);
    assert_eq!(quote.native_reservation_bytes, native.reservation_bytes);
    assert_eq!(
        quote.wrapper_population_bytes,
        independent_role_wrapper_population()
    );
    // Full transition/backing/payload equality is checked inside roles, where
    // the real private types are visible. No lower-bound spare allowance.
}

#[test]
fn owned_audit_pinned_source_callback_panic_revokes_both_roles_and_retains_source_until_last_frame()
{
    let _serial = TEST_SERIAL.lock().unwrap();
    let (
        coordinator,
        mut left,
        mut right,
        a,
        b,
        source,
        identity,
        payload,
        source_pointer,
        coefficients,
        binding,
    ) = {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let identity = budget.budget() as *const Budget;
        let mut d = dealer(&budget);
        // Reuse the genuine pinned-byte factory, not the private checked-owner
        // transfer bridge. No source owner, coefficients or guard is fabricated.
        let (mask, run) = issue(&budget, &mut d).unwrap();
        let source = Arc::downgrade(&run.retained);
        let payload = run.retained._charge.payload_bytes();
        let source_pointer = run.retained.checked.source_byte_pointer();
        let coefficients = run
            .retained
            .checked
            .certificate()
            .encoded_private_coefficients()
            .as_ptr();
        let binding = run.retained.checked.source_binding();
        let (coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
        let a = left.begin(&mut || false).unwrap();
        let b = right.begin(&mut || false).unwrap();
        drop(mask);
        (
            coordinator,
            left,
            right,
            a,
            b,
            source,
            identity,
            payload,
            source_pointer,
            coefficients,
            binding,
        )
    }; // All root/dealer/source-packet/plan/inventory/caller-value owners retired.

    let assert_original_source = || {
        let retained = source.upgrade().expect("original source released early");
        assert_eq!(retained._charge.budget() as *const Budget, identity);
        assert_eq!(retained._charge.payload_bytes(), payload);
        assert_eq!(retained.checked.source_byte_pointer(), source_pointer);
        assert_eq!(retained.checked.source_binding(), binding);
        assert_eq!(
            retained
                .checked
                .certificate()
                .encoded_private_coefficients(),
            &[1, 4095]
        );
        assert_eq!(
            retained
                .checked
                .certificate()
                .encoded_private_coefficients()
                .as_ptr(),
            coefficients
        );
        assert!(
            retained._charge.budget().used_bytes() >= payload + size_of::<OwnedExternalCharge>()
        );
    };
    assert_original_source();
    // Observe after native unwind guards have run, before catch_unwind returns.
    // Record rather than assert inside Drop: a regression must not double-panic.
    struct DuringUnwind<F: Fn()>(F);
    impl<F: Fn()> Drop for DuringUnwind<F> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                (self.0)();
            }
        }
    }
    let retained_and_revoked_during_unwind = std::cell::Cell::new(false);
    let mut polled = false;
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _witness = DuringUnwind(|| {
            retained_and_revoked_during_unwind.set(
                source.upgrade().is_some_and(|retained| {
                    retained._charge.budget() as *const Budget == identity
                        && retained._charge.payload_bytes() == payload
                        && retained._charge.budget().used_bytes()
                            >= payload + size_of::<OwnedExternalCharge>()
                        && retained.checked.source_byte_pointer() == source_pointer
                        && retained
                            .checked
                            .certificate()
                            .encoded_private_coefficients()
                            .as_ptr()
                            == coefficients
                }) && coordinator.is_aborted()
                    && a.as_bytes().is_err()
                    && b.as_bytes().is_err(),
            );
        });
        let _ = left.advance(b.as_bytes().unwrap(), &mut || {
            polled = true;
            assert!(!coordinator.is_aborted());
            assert_original_source();
            panic!("owned compiler acceptance callback unwind");
        });
    }));
    assert!(polled, "failure must originate in the caller callback");
    let panic = panic.expect_err("callback panic must propagate");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"owned compiler acceptance callback unwind")
    );
    assert!(retained_and_revoked_during_unwind.get());
    assert!(coordinator.is_aborted());
    assert_original_source();
    assert!(matches!(
        left.begin(&mut || false),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    assert!(matches!(
        right.advance(&[], &mut || false),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    assert!(matches!(
        right.finish(),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    assert_eq!(
        a.as_bytes(),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    );
    assert_eq!(
        b.as_bytes(),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    );
    drop((left, coordinator));
    assert_original_source(); // Only revoked compiler frames retain source now.
    drop(a);
    assert_original_source(); // One revoked frame alone keeps original charge.
    drop(b);
    assert!(source.upgrade().is_none());
}
