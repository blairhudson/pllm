use super::super::tests::plan;
use super::owned_audit_tests::independent_execution_payload;
use super::tests::{checked, fixture_request, packet, pin, raw};
use super::*;

pub(super) fn dealer(budget: &OwnedBudget) -> typed::OwnedLocalDealer {
    let c = checked();
    let edge = c.certificate().input_edge();
    typed::OwnedLocalDealer::new(
        edge.identity().source(),
        edge.identity().plan(),
        [72; 32],
        budget,
    )
    .unwrap()
}

pub(super) fn issue(
    budget: &OwnedBudget,
    dealer: &mut typed::OwnedLocalDealer,
) -> Result<(OwnedSigmaExecutionMask, OwnedSigmaExecutionRun), SigmaExecutionError> {
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let values = [1.0 / 4096.0, 4095.0 / 4096.0];
    let encoded = [1, 4095];
    let request = fixture_request(&p, &values, &encoded);
    let selection = request.selection.clone();
    execute_owned_sigma_linear_activation(
        pin(),
        packet(&selection, &raw()),
        request,
        &inventory,
        budget,
        dealer,
        &[-1, -1],
    )
}

fn payload_and_native() -> (usize, usize) {
    let p = plan();
    let expected = pin();
    let request = fixture_request(&p, &[], &[]);
    let payload = independent_execution_payload(&expected, &request);
    assert_eq!(
        execution_payload_quote_for::<OwnedBudgetContext>(&expected, &request).unwrap(),
        payload
    );
    let native = typed::local_linear_activation_resources_for::<OwnedBudgetContext>(
        local_spec(&checked()).unwrap(),
    )
    .unwrap()
    .reservation_bytes();
    (payload, native)
}

#[test]
fn owned_quotes_price_concrete_source_arc_wrappers_and_transition_results() {
    // Keep the original borrowed enum namespaces, not only alias-qualified
    // patterns. Existing clients may import these variants directly.
    use super::SigmaExecutionRoleStep::{Frame, Ready};
    use super::SigmaExecutionStep::{Frames, Output};
    let _frames: fn([SigmaExecutionFrame<'static>; 2]) -> SigmaExecutionStep<'static> = Frames;
    let _outputs: fn([SigmaExecutionOutput<'static>; 2]) -> SigmaExecutionStep<'static> = Output;
    let _frame: fn(SigmaExecutionRoleFrame<'static>) -> SigmaExecutionRoleStep<'static> = Frame;
    let _ready: SigmaExecutionRoleStep<'static> = Ready;
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let phases = SigmaCompilerPhases::checked_for::<OwnedBudgetContext>().unwrap();
    assert_eq!(
        owned_sigma_execution_compiler_preflight(&p, &inventory).unwrap(),
        phases.peak_bytes().unwrap()
    );
    assert!(phases.boundary_bytes > SigmaCompilerPhases::checked().unwrap().boundary_bytes);
    let c = checked();
    let q = retained_resource_inventory_for::<OwnedBudgetContext>(&c).unwrap();
    assert_eq!(
        q.source_owner_and_backing_bytes,
        c.checked_retained_resource_bytes().unwrap()
    );
    let arc = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<RetainedSourceFor<OwnedBudgetContext>>())
        .unwrap()
        .0
        .pad_to_align()
        .size();
    assert_eq!(
        q.arc_control_and_padding_bytes,
        arc - size_of::<SourceCheckedSigmaLinearActivation>() - size_of::<OwnedExternalCharge>()
    );
    assert_eq!(
        q.wrapper_population_bytes,
        size_of::<OwnedSigmaExecutionRun>()
            + size_of::<OwnedSigmaExecutionMask>()
            + 22 * size_of::<OwnedSigmaExecutionFrame>()
            + 11 * size_of::<OwnedSigmaExecutionStep>()
            + 2 * size_of::<OwnedSigmaExecutionOutput>()
    );
    assert!(
        q.issuance_transition_bytes
            >= size_of::<RetainedSourceFor<OwnedBudgetContext>>()
                + size_of::<
                    Result<
                        [typed::LocalLinearActivationFrameFor<OwnedBudgetContext>; 2],
                        typed::LocalLinearActivationStepFor<OwnedBudgetContext>,
                    >,
                >()
                + size_of::<
                    Result<
                        [typed::OwnedCommonMasked<typed::Ring>; 2],
                        typed::LocalLinearActivationStepFor<OwnedBudgetContext>,
                    >,
                >()
    );
    assert!(
        retained_resource_inventory_from_source_bytes_for::<OwnedBudgetContext>(usize::MAX)
            .unwrap()
            .payload_bytes()
            .is_err()
    );
}

#[test]
fn owned_preimport_and_native_growth_exact_short_foreign_and_bad_source() {
    let _serial = TEST_SERIAL.lock().unwrap();
    let (payload, native) = payload_and_native();
    let sizing = OwnedBudget::new(64 << 20).unwrap();
    let d = dealer(&sizing);
    let baseline = sizing.budget().used_bytes();
    drop((d, sizing));
    // First reservation one byte short; then original guard's native growth one
    // byte short. Both fail without leaving source/compiler/native admission.
    for bytes in [payload, payload + native] {
        let budget =
            OwnedBudget::new(baseline + bytes + size_of::<OwnedExternalCharge>() - 1).unwrap();
        let mut d = dealer(&budget);
        assert!(matches!(
            issue(&budget, &mut d),
            Err(SigmaExecutionError::Native(typed::Error::Resource))
        ));
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
    let budget =
        OwnedBudget::new(baseline + payload + native + size_of::<OwnedExternalCharge>()).unwrap();
    let mut d = dealer(&budget);
    let foreign = OwnedBudget::new(64 << 20).unwrap();
    let foreign_baseline = foreign.budget().used_bytes();
    assert!(matches!(
        issue(&foreign, &mut d),
        Err(SigmaExecutionError::BudgetIdentity)
    ));
    assert_eq!(foreign.budget().used_bytes(), foreign_baseline);
    let (mask, run) = issue(&budget, &mut d).unwrap();
    // Admission is a peak, not a promise that all native issuance temporaries
    // remain live on return. Native D20 retires its intermediate masks and
    // workspace before returning; their own leases legitimately refund. The
    // compiler/source/import payload (including compiler scratch) MUST remain
    // whole in the original guard. Do not extend native lifetimes to fit a test.
    let compiler_charge = payload + size_of::<OwnedExternalCharge>();
    let native_live = budget
        .budget()
        .used_bytes()
        .checked_sub(baseline + compiler_charge)
        .expect("original compiler/source charge refunded during issuance");
    let native_inline = size_of::<typed::DealerMaskFor<OwnedBudgetContext, typed::Ring>>()
        + size_of::<typed::LocalLinearActivationRunFor<OwnedBudgetContext>>();
    assert!(
        native_live >= native_inline,
        "returned native owners uncharged"
    );
    assert!(
        native_live < native,
        "retired native issuance storage not refunded"
    );
    assert_eq!(run.retained._charge.charged_bytes(), compiler_charge);
    assert_eq!(run.retained._charge.payload_bytes(), payload);
    assert!(run.retained._charge.is_for(budget.budget()));
    assert!(run.native.as_ref().unwrap().is_budget(budget.budget()));
    assert!(Arc::ptr_eq(&mask._retained, &run.retained));
    assert_eq!(
        run.retained
            .checked
            .certificate()
            .encoded_private_coefficients(),
        &[1, 4095]
    );
    let source = Arc::downgrade(&run.retained);
    let source_pointer = run.retained.checked.source_byte_pointer();
    let coefficient_pointer = run
        .retained
        .checked
        .certificate()
        .encoded_private_coefficients()
        .as_ptr();
    drop(run);
    // Mask alone retains the SAME complete source/compiler guard and original
    // coefficient backing after the native run and its remaining slices retire.
    assert!(source.upgrade().is_some());
    assert_eq!(mask._retained._charge.payload_bytes(), payload);
    assert_eq!(mask._retained._charge.charged_bytes(), compiler_charge);
    assert!(mask._retained._charge.is_for(budget.budget()));
    assert_eq!(mask._retained.checked.source_byte_pointer(), source_pointer);
    assert_eq!(
        mask._retained
            .checked
            .certificate()
            .encoded_private_coefficients()
            .as_ptr(),
        coefficient_pointer
    );
    assert!(
        budget.budget().used_bytes()
            >= baseline
                + compiler_charge
                + size_of::<typed::DealerMaskFor<OwnedBudgetContext, typed::Ring>>()
    );
    drop(mask);
    assert!(source.upgrade().is_none());
    assert_eq!(budget.budget().used_bytes(), baseline);

    // Original source bytes and canonical issuance values remain mandatory.
    for change_values in [false, true] {
        let p = plan();
        let inventory = lower_decoder_matrix_inventory(&p).unwrap();
        let values = [1.0 / 4096.0, 4095.0 / 4096.0];
        let encoded = if change_values { [2, 4095] } else { [1, 4095] };
        let r = fixture_request(&p, &values, &encoded);
        let selection = r.selection.clone();
        let mut bytes = raw();
        if !change_values {
            bytes[0] ^= 1;
        }
        assert!(matches!(
            execute_owned_sigma_linear_activation(
                pin(),
                packet(&selection, &bytes),
                r,
                &inventory,
                &budget,
                &mut d,
                &[-1, -1],
            ),
            Err(SigmaExecutionError::Source(_))
        ));
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn owned_transfer_rejects_foreign_original_guard_without_moving_source_authority() {
    let budget = OwnedBudget::new(64 << 20).unwrap();
    let foreign = OwnedBudget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.budget().used_bytes();
    let foreign_baseline = foreign.budget().used_bytes();
    let (payload, native) = payload_and_native();
    let charge = foreign.charge_external(payload + native).unwrap();
    assert!(matches!(
        consume_checked_local_from_reserved_for::<OwnedBudgetContext>(
            checked(),
            budget.budget(),
            &mut d,
            &[-1, -1],
            charge,
        ),
        Err(SigmaExecutionError::BudgetIdentity)
    ));
    assert_eq!(budget.budget().used_bytes(), baseline);
    assert_eq!(foreign.budget().used_bytes(), foreign_baseline);
}

#[test]
fn owned_frames_outputs_and_original_material_outlive_factory_and_all_roots() {
    let _serial = TEST_SERIAL.lock().unwrap();
    let (mask, mut run, identity, payload) = {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let identity = std::ptr::from_ref(budget.budget());
        let mut d = dealer(&budget);
        let (mask, run) = issue(&budget, &mut d).unwrap();
        let payload = run.retained._charge.payload_bytes();
        (mask, run, identity, payload)
    }; // Dealer, root, plan, inventory, packet and caller coefficients all retired.
    let source = Arc::downgrade(&run.retained);
    let source_pointer = run.retained.checked.source_byte_pointer();
    let coefficient_pointer = run
        .retained
        .checked
        .certificate()
        .encoded_private_coefficients()
        .as_ptr();
    let mut frames = run.begin(&mut || false).unwrap();
    let mut history = Vec::new();
    let outputs = loop {
        let next = run
            .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
            .unwrap();
        history.push(frames);
        match next {
            OwnedSigmaExecutionStep::Frames(next) => frames = next,
            OwnedSigmaExecutionStep::Output(outputs) => break outputs,
        }
    };
    assert_eq!(history.len(), 11);
    for output in &outputs {
        assert_eq!(
            std::ptr::from_ref(output._retained._charge.budget()),
            identity
        );
        assert_eq!(output._retained._charge.payload_bytes(), payload);
        assert_eq!(
            output._retained.checked.source_byte_pointer(),
            source_pointer
        );
        assert_eq!(
            output
                ._retained
                .checked
                .certificate()
                .encoded_private_coefficients()
                .as_ptr(),
            coefficient_pointer
        );
        assert_eq!(output.descriptor(), outputs[0].descriptor());
    }
    for pair in &history {
        for frame in pair {
            assert!(Arc::ptr_eq(&frame._retained, &outputs[0]._retained));
        }
    }
    drop((run, mask));
    // A single original matrix frame retains actual singleton material, not only
    // certificate metadata or a newly created budget with the same limit.
    let first = history.remove(0);
    drop(history);
    let [frame, sibling] = first;
    drop(sibling);
    let retry = OwnedBudget::new(64 << 20).unwrap();
    let mut d = dealer(&retry);
    assert!(matches!(
        issue(&retry, &mut d),
        Err(SigmaExecutionError::Native(typed::Error::Resource))
    ));
    drop(outputs);
    assert!(source.upgrade().is_some());
    assert_eq!(
        std::ptr::from_ref(frame._retained._charge.budget()),
        identity
    );
    assert!(
        frame._retained._charge.budget().used_bytes() >= payload + size_of::<OwnedExternalCharge>()
    );
    drop(frame);
    assert!(source.upgrade().is_none());
    drop(issue(&retry, &mut d).unwrap());
}

#[test]
fn owned_output_alone_keeps_source_coefficients_and_original_budget() {
    let _serial = TEST_SERIAL.lock().unwrap();
    let (output, source, identity) = {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let identity = std::ptr::from_ref(budget.budget());
        let mut d = dealer(&budget);
        let (mask, mut run) = issue(&budget, &mut d).unwrap();
        let source = Arc::downgrade(&run.retained);
        let mut frames = run.begin(&mut || false).unwrap();
        let output = loop {
            match run
                .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
                .unwrap()
            {
                OwnedSigmaExecutionStep::Frames(next) => frames = next,
                OwnedSigmaExecutionStep::Output([left, right]) => {
                    drop(right);
                    break left;
                }
            }
        };
        drop((mask, run, frames));
        (output, source, identity)
    };
    assert_eq!(
        std::ptr::from_ref(output._retained._charge.budget()),
        identity
    );
    assert_eq!(
        output
            ._retained
            .checked
            .certificate()
            .encoded_private_coefficients(),
        &[1, 4095]
    );
    assert_eq!(
        output.descriptor(),
        output
            ._retained
            .checked
            .certificate()
            .activation_output_edge()
    );
    assert!(source.upgrade().is_some());
    drop(output);
    assert!(source.upgrade().is_none());
}
