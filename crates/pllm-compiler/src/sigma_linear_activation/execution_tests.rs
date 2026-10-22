use super::super::tests::{plan, request};
use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use pllm_types::{Digest, NamedDigest};

// Generated original-source fixture, not a trained-checkpoint execution claim.
// Independent publisher pin is fixed before any supplied bytes or plan changes.
fn raw() -> [u8; 8] {
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&(1.0_f32 / 4096.0).to_le_bytes());
    bytes[4..].copy_from_slice(&(4095.0_f32 / 4096.0).to_le_bytes());
    bytes
}

fn fixture_request<'a>(
    p: &'a DecoderPlan,
    values: &'a [f64],
    encoded: &'a [i64],
) -> SigmaLinearActivationRequest<'a> {
    let mut r = request(p, values, encoded);
    r.selection.immutable_revision = "0123456789abcdef0123456789abcdef01234567".into();
    r.selection.content_sha256 = Sha256::digest(raw()).into();
    r.selection.output_coordinates = vec![0];
    r.selection.inner_coordinates = vec![0, 1];
    let source = SourceIdentity::new(
        r.selection.content_sha256,
        r.source.numeric_profile(),
        r.source.implementation(),
    );
    let replace = |edge: EdgeDescriptor, shape: [usize; 2], producer| {
        let id = edge.identity();
        EdgeDescriptor::new(
            EdgeIdentity::new(source, id.plan(), producer, id.wire(), 0),
            Shape::new(&shape).unwrap(),
            edge.numeric(),
            edge.representation(),
            edge.weight_visibility(),
        )
        .unwrap()
    };
    r.source = source;
    r.input = replace(r.input, [1, 2], r.input.identity().producer());
    r.private_weights = replace(
        r.private_weights,
        [1, 2],
        sigma_weight_selection_identity(&r.selection),
    );
    r.complete_sum = replace(r.complete_sum, [1, 1], r.complete_sum.identity().producer());
    r.rescaled = replace(r.rescaled, [1, 1], r.rescaled.identity().producer());
    r.activation_output = replace(
        r.activation_output,
        [1, 1],
        r.activation_output.identity().producer(),
    );
    r.next_use = ConsumerUse::new(
        r.activation_output,
        r.next_use.consumer(),
        r.next_use.operand_position(),
        r.next_use.ordinal(),
    );
    r
}

fn pin() -> ExpectedSigmaSourcePin {
    pin_with_selection_capacity(0)
}

fn pin_with_selection_capacity(extra: usize) -> ExpectedSigmaSourcePin {
    let p = plan();
    let mut r = fixture_request(&p, &[], &[]);
    r.selection.checkpoint.reserve(extra);
    ExpectedSigmaSourcePin::from_trusted_publisher(
        NamedDigest {
            id: "generated-f32-original".into(),
            digest: Digest::from_sha256(r.selection.content_sha256),
        },
        8,
        r.selection,
        SourceDtype::F32,
        SourceEndian::Little,
        128,
        &p.config_digest,
        &p.digest(),
    )
    .unwrap()
}

fn packet<'a>(selection: &'a CheckpointTensorSelection, bytes: &'a [u8]) -> SigmaSourceBytes<'a> {
    SigmaSourceBytes {
        artifact_name: "generated-f32-original",
        selection,
        dtype: SourceDtype::F32,
        endian: SourceEndian::Little,
        tensor_offset: 128,
        scalar_artifact_offsets: &[128, 132],
        bytes,
    }
}

fn checked() -> SourceCheckedSigmaLinearActivation {
    let p = plan();
    let values = [1.0 / 4096.0, 4095.0 / 4096.0];
    let encoded = [1, 4095];
    let r = fixture_request(&p, &values, &encoded);
    let source = import_sigma_source_bytes(pin(), packet(&r.selection, &raw()), &encoded).unwrap();
    certify_source_checked_sigma_linear_activation(r, source).unwrap()
}

fn dealer<'b>(
    c: &SourceCheckedSigmaLinearActivation,
    budget: &'b Budget,
) -> typed::LocalDealer<'b> {
    let edge = c.certificate().input_edge();
    typed::LocalDealer::new(
        edge.identity().source(),
        edge.identity().plan(),
        [72; 32],
        budget,
    )
    .unwrap()
}

#[test]
fn raw_facade_preimport_one_byte_short_and_budget_identity_reject() {
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    assert!(sigma_execution_compiler_preflight(&p, &inventory).unwrap() > 0);
    let c = checked();
    let sizing = Budget::new(64 << 20).unwrap();
    let d = dealer(&c, &sizing);
    let dealer_bytes = sizing.used_bytes();
    drop(d);
    let values = [1.0 / 4096.0, 4095.0 / 4096.0];
    let encoded = [1, 4095];
    let r = fixture_request(&p, &values, &encoded);
    let payload = execution_payload_quote(&pin(), &r).unwrap();
    let budget =
        Budget::new(dealer_bytes + payload + size_of::<ExternalCharge<'static>>() - 1).unwrap();
    let other = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&c, &budget);
    let baseline = budget.used_bytes();
    let selection = r.selection.clone();
    assert!(matches!(
        execute_sigma_linear_activation(
            pin(),
            packet(&selection, &raw()),
            r,
            &inventory,
            &budget,
            &mut d,
            &[-1, -1]
        ),
        Err(SigmaExecutionError::Native(typed::Error::Resource))
    ));
    assert_eq!(budget.used_bytes(), baseline);
    let r = fixture_request(&p, &[], &[]);
    assert!(matches!(
        execute_sigma_linear_activation(
            pin(),
            packet(&selection, &raw()),
            r,
            &inventory,
            &other,
            &mut d,
            &[]
        ),
        Err(SigmaExecutionError::BudgetIdentity)
    ));
    assert_eq!(other.used_bytes(), 0);
}

#[test]
fn aggregate_phase_quote_is_checked_and_excludes_only_borrowed_baseline() {
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let phases = SigmaCompilerPhases::checked().unwrap();
    assert_eq!(
        phases.peak_bytes().unwrap(),
        phases.boundary_bytes
            + phases
                .source_commitment_bytes
                .max(phases.pairing_bytes)
                .max(phases.proof_bytes)
                .max(phases.certificate_return_bytes)
                .max(phases.native_metadata_bytes)
    );
    assert_eq!(
        sigma_execution_compiler_preflight(&p, &inventory).unwrap(),
        phases.peak_bytes().unwrap()
    );
    let r = fixture_request(&p, &[], &[]);
    let expected = pin();
    let import = expected.checked_import_quote().unwrap();
    let retained = retained_resource_inventory_from_source_bytes(
        expected.checked_certification_retained_quote(&r).unwrap(),
    )
    .unwrap()
    .payload_bytes()
    .unwrap();
    let payload = execution_payload_quote(&expected, &r).unwrap();
    assert_eq!(
        payload,
        retained.max(import.retained_bytes)
            + import.import_encoding_scratch_bytes
            + phases.peak_bytes().unwrap()
    );
    let mut other = p.clone();
    other.model_family.reserve(1 << 20);
    assert_eq!(other, p);
    assert_eq!(
        execution_payload_quote(&expected, &fixture_request(&other, &[], &[])).unwrap(),
        payload
    );
    // Checked arithmetic, including phase and aggregate sums, has no wraparound.
    assert_eq!(
        execution_sum([usize::MAX, 1]),
        Err(SigmaExecutionError::Resource)
    );
    assert_eq!(
        SigmaCompilerPhases {
            boundary_bytes: usize::MAX,
            ..phases
        }
        .peak_bytes(),
        Err(SigmaExecutionError::Resource)
    );
    assert_eq!(
        SigmaCompilerPhases {
            proof_bytes: usize::MAX,
            ..phases
        }
        .peak_bytes(),
        Err(SigmaExecutionError::Resource)
    );
    let budget = Budget::new(1024).unwrap();
    assert!(budget.charge_external(usize::MAX).is_err());
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn raw_native_growth_one_byte_short_preserves_original_guard_until_source_retirement() {
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let values = [1.0 / 4096.0, 4095.0 / 4096.0];
    let encoded = [1, 4095];
    let r = fixture_request(&p, &values, &encoded);
    let payload = execution_payload_quote(&pin(), &r).unwrap();
    let c = checked();
    let native = typed::local_linear_activation_resources(local_spec(&c).unwrap())
        .unwrap()
        .reservation_bytes();
    let sizing = Budget::new(64 << 20).unwrap();
    let d = dealer(&c, &sizing);
    let dealer_bytes = sizing.used_bytes();
    drop(d);
    let budget =
        Budget::new(dealer_bytes + payload + native + size_of::<ExternalCharge<'static>>() - 1)
            .unwrap();
    let mut d = dealer(&c, &budget);
    let selection = r.selection.clone();
    assert!(matches!(
        execute_sigma_linear_activation(
            pin(),
            packet(&selection, &raw()),
            r,
            &inventory,
            &budget,
            &mut d,
            &[-1, -1]
        ),
        Err(SigmaExecutionError::Native(typed::Error::Resource))
    ));
    assert_eq!(budget.used_bytes(), dealer_bytes);
    drop(d);
    assert_eq!(budget.used_bytes(), 0);
    // No compiler entropy observer or destruction-order hook exists. These
    // assertions observe accounting restoration, not entropy or erased bytes.
}

#[test]
fn raw_source_stale_config_plan_issuance_and_input_errors_restore_admission() {
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let c = checked();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&c, &budget);
    let baseline = budget.used_bytes();
    let values = [1.0 / 4096.0, 4095.0 / 4096.0];
    let encoded = [1, 4095];
    for case in 0..8 {
        let mut changed_plan = p.clone();
        if case == 1 {
            changed_plan.model_family.push_str("-stale");
        }
        if case == 2 {
            changed_plan.config_digest = Digest::from_sha256([93; 32]);
        }
        // Cases 3/4 pair a fresh inventory with substituted plan/config, to
        // exercise independent frozen publisher tuple checks after pairing.
        if case == 3 {
            changed_plan.model_family.push_str("-substituted");
        }
        if case == 4 {
            changed_plan.config_digest = Digest::from_sha256([94; 32]);
        }
        let fresh = if matches!(case, 3 | 4) {
            Some(lower_decoder_matrix_inventory(&changed_plan).unwrap())
        } else {
            None
        };
        let changed_encoded = [1, 4094];
        let issued = if case == 5 {
            &changed_encoded
        } else {
            &encoded
        };
        let r = fixture_request(&changed_plan, &values, issued);
        let selection = r.selection.clone();
        let mut bytes = raw();
        if case == 0 {
            bytes[0] ^= 1;
        }
        let input: &[i64] = match case {
            6 => &[],
            7 => &[i64::MAX, i64::MAX],
            _ => &[-1, -1],
        };
        let result = execute_sigma_linear_activation(
            pin(),
            packet(&selection, &bytes),
            r,
            fresh.as_ref().unwrap_or(&inventory),
            &budget,
            &mut d,
            input,
        );
        if case < 6 {
            assert!(matches!(result, Err(SigmaExecutionError::Source(_))));
        } else {
            assert!(matches!(result, Err(SigmaExecutionError::Resource)));
        }
        assert_eq!(budget.used_bytes(), baseline, "case {case}");
    }
    drop(d);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn private_bridge_derives_spec_and_weights_only_from_immutable_checked_owner() {
    let c = checked();
    assert_eq!(c.certificate().encoded_private_coefficients(), &[1, 4095]);
    assert_eq!(
        c.certificate().fixture_scope(),
        FixtureScope::CheckpointDerivedSubmatrix
    );
    assert!(c.native_elevation().is_err());
    let spec = local_spec(&c).unwrap();
    assert!(
        typed::local_linear_activation_resources(spec)
            .unwrap()
            .reservation_bytes()
            > 0
    );
    assert!(retained_payload_bytes(&c).unwrap() > c.retained_resource_bytes().unwrap());
    let inventory = retained_resource_inventory(&c).unwrap();
    assert_eq!(
        inventory.source_owner_and_backing_bytes,
        c.retained_resource_bytes().unwrap()
    );
    let arc_layout = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<RetainedSource<'static>>())
        .unwrap()
        .0
        .pad_to_align();
    assert_eq!(
        inventory.arc_control_and_padding_bytes,
        arc_layout.size()
            - size_of::<SourceCheckedSigmaLinearActivation>()
            - size_of::<ExternalCharge<'static>>()
    );
    assert_eq!(
        inventory.wrapper_population_bytes,
        size_of::<SigmaExecutionRun<'static>>()
            + size_of::<SigmaExecutionMask<'static>>()
            + 22 * size_of::<SigmaExecutionFrame<'static>>()
            + 11 * size_of::<SigmaExecutionStep<'static>>()
            + 2 * size_of::<SigmaExecutionOutput<'static>>()
    );
    assert_eq!(
        inventory.issuance_transition_bytes,
        size_of::<RetainedSource<'static>>()
            + size_of::<typed::LocalLinearActivationSpec>()
            + size_of::<typed::LocalLinearActivationResources>()
            + size_of::<(
                typed::DealerMask<'static, typed::Ring>,
                typed::LocalLinearActivationRun<'static>
            )>()
            + size_of::<typed::LocalLinearActivationStep<'static>>()
            + 2 * size_of::<typed::LocalLinearActivationFrame<'static>>()
            + size_of::<
                Result<
                    [typed::LocalLinearActivationFrame<'static>; 2],
                    typed::LocalLinearActivationStep<'static>,
                >,
            >()
            + size_of::<
                Result<
                    [typed::CommonMasked<'static, typed::Ring>; 2],
                    typed::LocalLinearActivationStep<'static>,
                >,
            >()
            + 2 * size_of::<typed::CommonMasked<'static, typed::Ring>>()
            + 3 * size_of::<Arc<RetainedSource<'static>>>()
    );
}

#[test]
fn preimport_retained_inventory_reuses_source_quote_without_constructing_spec() {
    let p = plan();
    let r = fixture_request(&p, &[], &[]);
    let expected = pin();
    let source_bytes = expected.checked_certification_retained_quote(&r).unwrap();
    let inventory = retained_resource_inventory_from_source_bytes(source_bytes).unwrap();
    assert_eq!(inventory.source_owner_and_backing_bytes, source_bytes);
    let actual = checked();
    assert_eq!(inventory, retained_resource_inventory(&actual).unwrap());
    assert_eq!(
        inventory.payload_bytes().unwrap(),
        inventory.source_owner_and_backing_bytes
            + inventory.arc_control_and_padding_bytes
            + inventory.wrapper_population_bytes
            + inventory.issuance_transition_bytes
    );
    assert!(retained_resource_inventory_from_source_bytes(usize::MAX)
        .unwrap()
        .payload_bytes()
        .is_err());
    // Retained source alone is not the compiler/import admission price.
    let lowered = lower_decoder_matrix_inventory(&p).unwrap();
    assert!(sigma_execution_compiler_preflight(&p, &lowered).unwrap() > 0);
    assert!(execution_payload_quote(&expected, &r).unwrap() > inventory.payload_bytes().unwrap());
}

#[test]
fn preimport_retained_inventory_tracks_expected_and_request_owned_capacities() {
    let p = plan();
    let mut r = fixture_request(&p, &[], &[]);
    let expected = pin();
    let larger_expected = pin_with_selection_capacity(2048);
    assert_eq!(expected.selection(), larger_expected.selection());
    let expected_delta = larger_expected
        .selection()
        .checked_metadata_backing_bytes()
        .unwrap()
        - expected
            .selection()
            .checked_metadata_backing_bytes()
            .unwrap();
    assert!(expected_delta > 0);
    let quote = |pin: &ExpectedSigmaSourcePin, request: &SigmaLinearActivationRequest<'_>| {
        retained_resource_inventory_from_source_bytes(
            pin.checked_certification_retained_quote(request).unwrap(),
        )
        .unwrap()
        .payload_bytes()
        .unwrap()
    };
    let baseline = quote(&expected, &r);
    let raw_baseline = execution_payload_quote(&expected, &r).unwrap();
    assert_eq!(quote(&larger_expected, &r) - baseline, expected_delta);
    assert_eq!(
        execution_payload_quote(&larger_expected, &r).unwrap() - raw_baseline,
        expected_delta
    );
    let old_request_capacity = r.selection.checked_metadata_backing_bytes().unwrap();
    r.selection.input_row_coordinates.reserve(37);
    let request_delta =
        r.selection.checked_metadata_backing_bytes().unwrap() - old_request_capacity;
    assert!(request_delta > 0);
    assert_eq!(quote(&expected, &r) - baseline, request_delta);
    assert_eq!(
        execution_payload_quote(&expected, &r).unwrap() - raw_baseline,
        request_delta
    );
    assert_eq!(
        quote(&larger_expected, &r) - baseline,
        expected_delta + request_delta
    );
    assert_eq!(
        execution_payload_quote(&larger_expected, &r).unwrap() - raw_baseline,
        expected_delta + request_delta
    );
}

#[test]
fn private_bridge_one_byte_short_admission_retires_every_charge() {
    let c = checked();
    let payload = retained_payload_bytes(&c).unwrap();
    let native = typed::local_linear_activation_resources(local_spec(&c).unwrap())
        .unwrap()
        .reservation_bytes();
    let large = Budget::new(64 << 20).unwrap();
    let d = dealer(&c, &large);
    let dealer_bytes = large.used_bytes();
    drop(d);
    let total = dealer_bytes + payload + native + size_of::<ExternalCharge<'static>>();
    let budget = Budget::new(total - 1).unwrap();
    let mut d = dealer(&c, &budget);
    assert!(matches!(
        consume_checked_local(c, &budget, &mut d, &[-1, -1]),
        Err(SigmaExecutionError::Native(typed::Error::Resource))
    ));
    assert_eq!(budget.used_bytes(), dealer_bytes);
    drop(d);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn private_bridge_budget_pairing_and_invalid_input_reject_without_retained_charge() {
    let budget = Budget::new(64 << 20).unwrap();
    let other = Budget::new(64 << 20).unwrap();
    let c = checked();
    let mut d = dealer(&c, &budget);
    let baseline = budget.used_bytes();
    assert!(matches!(
        consume_checked_local(c, &other, &mut d, &[-1, -1]),
        Err(SigmaExecutionError::BudgetIdentity)
    ));
    assert_eq!(other.used_bytes(), 0);
    for input in [&[][..], &[-1][..], &[i64::MAX, i64::MAX][..]] {
        // Native validation precedes singleton admission, transfer and entropy.
        // Compiler has no injected entropy API; native tests own draw counting.
        assert!(consume_checked_local(checked(), &budget, &mut d, input).is_err());
        assert_eq!(budget.used_bytes(), baseline);
    }
    drop(d);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn original_reservation_rejects_short_payload_wrong_budget_and_input_before_native_call() {
    let budget = Budget::new(64 << 20).unwrap();
    let other = Budget::new(64 << 20).unwrap();
    let c = checked();
    let mut d = dealer(&c, &budget);
    let baseline = budget.used_bytes();
    let native = typed::local_linear_activation_resources(local_spec(&c).unwrap())
        .unwrap()
        .reservation_bytes();
    let payload = retained_payload_bytes(&c).unwrap();
    let guard = other.charge_external(payload + native).unwrap();
    assert!(matches!(
        consume_checked_local_from_reserved(c, &budget, &mut d, &[-1, -1], guard),
        Err(SigmaExecutionError::BudgetIdentity)
    ));
    assert_eq!(other.used_bytes(), 0);
    let guard = budget.charge_external(payload + native - 1).unwrap();
    assert!(matches!(
        consume_checked_local_from_reserved(checked(), &budget, &mut d, &[-1, -1], guard),
        Err(SigmaExecutionError::Resource)
    ));
    assert_eq!(budget.used_bytes(), baseline);
    for input in [&[][..], &[-1][..], &[i64::MAX, i64::MAX][..]] {
        let guard = budget.charge_external(payload + native).unwrap();
        assert!(matches!(
            consume_checked_local_from_reserved(checked(), &budget, &mut d, input, guard),
            Err(SigmaExecutionError::Resource)
        ));
        assert_eq!(budget.used_bytes(), baseline);
    }
    drop(d);
    assert_eq!(budget.used_bytes(), 0);
}

// One native test owns the process-wide matrix singleton at a time. This module
// has only this single successful native issuance test; no parallel successes.
#[test]
fn raw_exact_admission_and_private_bridge_frames_outputs_keep_source_guard() {
    let c = checked();
    let payload = {
        let p = plan();
        execution_payload_quote(&pin(), &fixture_request(&p, &[], &[])).unwrap()
    };
    let native = typed::local_linear_activation_resources(local_spec(&c).unwrap())
        .unwrap()
        .reservation_bytes();
    let sizing = Budget::new(64 << 20).unwrap();
    let sizing_dealer = dealer(&c, &sizing);
    let dealer_bytes = sizing.used_bytes();
    drop(sizing_dealer);
    let budget =
        Budget::new(dealer_bytes + payload + native + size_of::<ExternalCharge<'static>>())
            .unwrap();
    let mut d = dealer(&c, &budget);
    let baseline = budget.used_bytes();
    let binding = c.source_binding();
    drop(c);
    // Returned capabilities borrow budget only: plan, inventory, source packet
    // and caller coefficients all retire here, before any native exchanges.
    let (mask, mut run) = {
        let p = plan();
        let inventory = lower_decoder_matrix_inventory(&p).unwrap();
        let values = [1.0 / 4096.0, 4095.0 / 4096.0];
        let encoded = [1, 4095];
        let r = fixture_request(&p, &values, &encoded);
        let selection = r.selection.clone();
        execute_sigma_linear_activation(
            pin(),
            packet(&selection, &raw()),
            r,
            &inventory,
            &budget,
            &mut d,
            &[-1, -1],
        )
        .unwrap()
    };
    assert_eq!(run.retained.checked.source_binding(), binding);
    assert_eq!(
        run.provenance(),
        SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution
    );
    assert_eq!(
        run.retained
            .checked
            .certificate()
            .encoded_private_coefficients(),
        &[1, 4095]
    );
    assert_eq!(run.retained._charge.payload_bytes(), payload);
    let source_pointer = run.retained.checked.source_byte_pointer();
    let coefficient_pointer = run
        .retained
        .checked
        .certificate()
        .encoded_private_coefficients()
        .as_ptr();
    let mut frames = run.begin(&mut || false).unwrap();
    let mut retained_frames = Vec::new();
    let outputs = loop {
        let step = run
            .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
            .unwrap();
        retained_frames.push(frames);
        match step {
            SigmaExecutionStep::Frames(next) => frames = next,
            SigmaExecutionStep::Output(outputs) => break outputs,
        }
    };
    assert_eq!(retained_frames.len(), 11);
    assert_eq!(outputs[0].descriptor(), outputs[1].descriptor());
    assert_eq!(
        outputs[0]._retained.checked.source_byte_pointer(),
        source_pointer
    );
    assert_eq!(
        outputs[0]
            ._retained
            .checked
            .certificate()
            .encoded_private_coefficients()
            .as_ptr(),
        coefficient_pointer
    );
    assert_eq!(outputs[0]._retained.checked.source_binding(), binding);
    for pair in &retained_frames {
        for frame in pair {
            assert!(Arc::ptr_eq(&frame._retained, &outputs[0]._retained));
        }
    }
    assert_eq!(
        outputs[0].provenance(),
        SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution
    );
    run.cancel();
    assert_eq!(run.retained._charge.payload_bytes(), payload);
    drop(run);
    drop(mask);
    assert!(budget.used_bytes() >= baseline + payload + size_of::<ExternalCharge<'static>>());
    drop(retained_frames);
    assert!(budget.used_bytes() >= baseline + payload + size_of::<ExternalCharge<'static>>());
    let [left, right] = outputs;
    drop(left);
    assert!(budget.used_bytes() >= baseline + payload + size_of::<ExternalCharge<'static>>());
    drop(right);
    assert_eq!(budget.used_bytes(), baseline);
    drop(d);
    assert_eq!(budget.used_bytes(), 0);

    // Exercise failure through the same raw source facade, not only the private
    // checked-owner bridge. Caller source/plan owners retire before cancellation
    // or malformed peer bytes; escaped frames keep the original source charge.
    for cancel in [false, true] {
        let c = checked();
        let mut d = dealer(&c, &budget);
        drop(c);
        let baseline = budget.used_bytes();
        let (mask, mut run) = {
            let p = plan();
            let inventory = lower_decoder_matrix_inventory(&p).unwrap();
            let values = [1.0 / 4096.0, 4095.0 / 4096.0];
            let encoded = [1, 4095];
            let r = fixture_request(&p, &values, &encoded);
            let selection = r.selection.clone();
            execute_sigma_linear_activation(
                pin(),
                packet(&selection, &raw()),
                r,
                &inventory,
                &budget,
                &mut d,
                &[-1, -1],
            )
            .unwrap()
        };
        let source_pointer = run.retained.checked.source_byte_pointer();
        let coefficient_pointer = run
            .retained
            .checked
            .certificate()
            .encoded_private_coefficients()
            .as_ptr();
        let frames = run.begin(&mut || false).unwrap();
        assert!(run.advance([&[], &[]], &mut || cancel).is_err());
        run.cancel();
        run.cancel();
        assert!(run.begin(&mut || false).is_err());
        assert_eq!(run.retained._charge.payload_bytes(), payload);
        for frame in &frames {
            assert!(Arc::ptr_eq(&frame._retained, &run.retained));
            assert_eq!(
                frame._retained.checked.source_byte_pointer(),
                source_pointer
            );
            assert_eq!(
                frame
                    ._retained
                    .checked
                    .certificate()
                    .encoded_private_coefficients()
                    .as_ptr(),
                coefficient_pointer
            );
        }
        drop(run);
        drop(mask);
        assert!(budget.used_bytes() >= baseline + payload + size_of::<ExternalCharge<'static>>());
        drop(frames);
        assert_eq!(budget.used_bytes(), baseline);
        drop(d);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn source_and_plan_substitution_never_reach_private_issuance() {
    let p = plan();
    let r = fixture_request(&p, &[], &[]);
    let mut changed = raw();
    changed[0] ^= 1;
    assert!(import_sigma_source_bytes(pin(), packet(&r.selection, &changed), &[1, 4095]).is_err());
    for config in [false, true] {
        let expected = pin();
        let mut p = plan();
        if config {
            p.config_digest = Digest::from_sha256([93; 32]);
        } else {
            p.prefill.operations[0].id.push_str("-substituted");
        }
        let values = [1.0 / 4096.0, 4095.0 / 4096.0];
        let r = fixture_request(&p, &values, &[1, 4095]);
        let source =
            import_sigma_source_bytes(expected, packet(&r.selection, &raw()), &[1, 4095]).unwrap();
        assert!(certify_source_checked_sigma_linear_activation(r, source).is_err());
    }
}
