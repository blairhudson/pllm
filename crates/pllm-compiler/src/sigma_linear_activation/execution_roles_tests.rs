use super::super::super::tests::plan;
use super::super::tests::{checked, dealer, fixture_request, packet, pin, raw};
use super::*;

fn issue<'b>(
    budget: &'b Budget,
    d: &mut typed::LocalDealer<'b>,
) -> (SigmaExecutionMask<'b>, SigmaExecutionRun<'b>) {
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let values = [1.0 / 4096.0, 4095.0 / 4096.0];
    let encoded = [1, 4095];
    let request = fixture_request(&p, &values, &encoded);
    let selection = request.selection.clone();
    execute_sigma_linear_activation(
        pin(),
        packet(&selection, &raw()),
        request,
        &inventory,
        budget,
        d,
        &[-1, -1],
    )
    .unwrap()
}

fn complete<'b>(
    [left, right]: &mut [SigmaExecutionRole<'b>; 2],
) -> Vec<[SigmaExecutionRoleFrame<'b>; 2]> {
    let mut frames = [
        left.begin(&mut || false).unwrap(),
        right.begin(&mut || false).unwrap(),
    ];
    let mut retained = Vec::new();
    for round in 0..11 {
        let a = left
            .advance(frames[1].as_bytes().unwrap(), &mut || false)
            .unwrap();
        let b = right
            .advance(frames[0].as_bytes().unwrap(), &mut || false)
            .unwrap();
        retained.push(frames);
        match (a, b) {
            (SigmaExecutionRoleStep::Frame(a), SigmaExecutionRoleStep::Frame(b)) => {
                frames = [a, b];
            }
            (SigmaExecutionRoleStep::Ready, SigmaExecutionRoleStep::Ready) => {
                assert_eq!(round, 10);
                return retained;
            }
            _ => panic!("role schedules diverged"),
        }
    }
    panic!("missing ready boundary")
}

#[test]
fn d27_compiler_preconstruction_quote_prices_native_and_concrete_wrapper_population() {
    let q = sigma_execution_role_split_resources().unwrap();
    assert_eq!(
        q.native_reservation_bytes,
        typed::local_linear_activation_role_split_resources()
            .unwrap()
            .reservation_bytes
    );
    assert_eq!(
        q.additional_payload_bytes,
        q.native_reservation_bytes
            + q.retention_backing_bytes
            + q.wrapper_population_bytes
            + q.transition_bytes
    );
    assert_eq!(
        q.wrapper_population_bytes,
        size_of::<SigmaExecutionRoleCoordinator<'static>>()
            + 2 * size_of::<SigmaExecutionRole<'static>>()
            + 22 * size_of::<SigmaExecutionRoleFrame<'static>>()
            + 22 * size_of::<SigmaExecutionRoleStep<'static>>()
            + 2 * size_of::<SigmaExecutionRoleOutput<'static>>()
    );
    assert_eq!(
        execution_sum([usize::MAX, q.additional_payload_bytes]),
        Err(SigmaExecutionError::Resource)
    );
}

#[test]
fn d27_compiler_transition_census_explicitly_includes_quote_return_slot() {
    // Independently enumerate the named logical slots; do not infer spare
    // capacity from the total quote or substitute an arbitrary reserve.
    let expected = size_of::<SigmaExecutionRoleSplitResources>()
        + size_of::<Result<SigmaExecutionRoleSplitResources, SigmaExecutionError>>()
        + size_of::<typed::LocalLinearActivationRoleSplitResources>()
        + size_of::<RoleAdmission<'static>>()
        + size_of::<SigmaExecutionRun<'static>>()
        + size_of::<RetainedRoles<'static>>()
        + size_of::<typed::LocalLinearActivationRun<'static>>()
        + size_of::<NativeRoles<'static>>()
        + size_of::<Result<NativeRoles<'static>, typed::Error>>()
        + size_of::<CompilerRoles<'static>>()
        + size_of::<Result<CompilerRoles<'static>, SigmaExecutionError>>()
        + size_of::<Result<typed::LocalLinearActivationRoleFrame<'static>, typed::Error>>()
        + size_of::<Result<typed::LocalLinearActivationRoleStep<'static>, typed::Error>>()
        + size_of::<Result<typed::LocalLinearActivationRoleOutput<'static>, typed::Error>>()
        + size_of::<Result<SigmaExecutionRoleFrame<'static>, SigmaExecutionError>>()
        + size_of::<Result<SigmaExecutionRoleStep<'static>, SigmaExecutionError>>()
        + size_of::<Result<SigmaExecutionRoleOutput<'static>, SigmaExecutionError>>()
        + size_of::<Result<EdgeDescriptor, SigmaExecutionError>>()
        + 4 * size_of::<Arc<RetainedRoles<'static>>>()
        + size_of::<Arc<RetainedSource<'static>>>()
        + size_of::<std::array::IntoIter<usize, 5>>()
        + size_of::<std::array::IntoIter<usize, 22>>();
    assert_eq!(
        sigma_execution_role_split_resources()
            .unwrap()
            .transition_bytes,
        expected
    );
}

#[test]
fn d27_compiler_callback_reentrancy_cancels_both_without_detaching_source_guard() {
    let _serial = super::super::TEST_SERIAL.lock().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let budget = Budget::new(64 << 20).unwrap();
        let c = checked();
        let mut d = dealer(&c, &budget);
        drop(c);
        let baseline = budget.used_bytes();
        let (mask, run) = issue(&budget, &mut d);
        let (mut coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
        let a = left.begin(&mut || coordinator.is_aborted()).unwrap();
        let b = right.begin(&mut || coordinator.is_aborted()).unwrap();
        assert!(left
            .advance(b.as_bytes().unwrap(), &mut || {
                assert!(!coordinator.is_aborted());
                coordinator.cancel();
                false
            })
            .is_err());
        assert!(right.advance(&[], &mut || false).is_err());
        assert!(coordinator.is_aborted());
        drop((left, right, coordinator, mask));
        assert!(budget.used_bytes() > baseline);
        assert!(a.as_bytes().is_err());
        assert!(b.as_bytes().is_err());
        drop((a, b));
        assert_eq!(budget.used_bytes(), baseline);
        drop(d);
        assert_eq!(budget.used_bytes(), 0);
        send.send(()).unwrap();
    });
    receive
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("compiler role callback deadlocked or panicked");
    worker.join().unwrap();
}

// Shares the local compiler binary's matrix singleton with the coupled facade
// test. Native's independent binary owns its own serialization lock.
#[test]
fn d27_compiler_roles_execute_retain_original_source_and_fail_closed() {
    let _serial = super::super::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let c = checked();
    let mut d = dealer(&c, &budget);
    drop(c);
    let baseline = budget.used_bytes();
    let provenance = SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution;
    let (mask, run) = issue(&budget, &mut d);
    let source = Arc::clone(&run.retained);
    assert!(Arc::ptr_eq(&mask._retained, &source));
    let source_pointer = source.checked.source_byte_pointer();
    let coefficients = source
        .checked
        .certificate()
        .encoded_private_coefficients()
        .as_ptr();
    let original_charge = source._charge.payload_bytes();
    let expected_output = source.checked.certificate().activation_output_edge();
    let q = sigma_execution_role_split_resources().unwrap();
    let before_split = budget.used_bytes();
    let (mut coordinator, mut roles) = run.into_roles(&budget).unwrap();
    assert_eq!(
        budget.used_bytes(),
        before_split + q.additional_payload_bytes + size_of::<ExternalCharge<'static>>()
    );
    assert_eq!(roles[0].role(), 0);
    assert_eq!(roles[1].role(), 1);
    assert_eq!(roles[0].provenance(), provenance);
    assert_eq!(coordinator.provenance(), provenance);
    for role in &roles {
        assert!(Arc::ptr_eq(&role.retained._source, &source));
        assert_eq!(
            role.retained._source._charge.payload_bytes(),
            original_charge
        );
        assert_eq!(
            role.retained._source.checked.source_byte_pointer(),
            source_pointer
        );
        assert_eq!(
            role.retained
                ._source
                .checked
                .certificate()
                .encoded_private_coefficients()
                .as_ptr(),
            coefficients
        );
        assert_eq!(
            role.retained._charge.payload_bytes(),
            q.additional_payload_bytes - q.native_reservation_bytes
        );
    }
    let retained = complete(&mut roles);
    assert_eq!(retained.len(), 11);
    let [left, right] = roles;
    let outputs = [left.finish().unwrap(), right.finish().unwrap()];
    for output in &outputs {
        assert_eq!(output.descriptor().unwrap(), expected_output);
        assert_eq!(output.provenance(), provenance);
        assert!(Arc::ptr_eq(&output._retained._source, &source));
    }
    for pair in &retained {
        for frame in pair {
            assert!(Arc::ptr_eq(&frame._retained, &outputs[0]._retained));
        }
    }
    coordinator.cancel();
    assert!(coordinator.is_aborted());
    for output in &outputs {
        assert_eq!(
            output.descriptor(),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        );
    }
    assert!(retained[0][0].as_bytes().is_err());
    drop((mask, source, coordinator));
    assert!(budget.used_bytes() > baseline);
    drop(outputs);
    assert!(budget.used_bytes() > baseline);
    drop(retained);
    assert_eq!(budget.used_bytes(), baseline);

    // First error revokes the sibling, even before it has ever been polled.
    // The same pinned-source facade is used; no source owner is reconstructed.
    for case in 0..5 {
        let (mask, run) = issue(&budget, &mut d);
        let (coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
        let a = left.begin(&mut || false).unwrap();
        if case == 0 {
            drop(left);
        } else if case == 1 {
            assert!(left.advance(&[], &mut || true).is_err());
            drop(left);
        } else if case == 2 {
            assert!(left.finish().is_err());
        } else if case == 3 {
            let b = right.begin(&mut || false).unwrap();
            assert!(left.advance(a.as_bytes().unwrap(), &mut || false).is_err());
            assert!(b.as_bytes().is_err());
            drop(left);
        } else {
            drop(coordinator);
            assert!(left.advance(&[], &mut || false).is_err());
            assert!(right.begin(&mut || false).is_err());
            drop((left, right, mask));
            assert!(budget.used_bytes() > baseline);
            drop(a);
            assert_eq!(budget.used_bytes(), baseline);
            continue;
        }
        assert!(coordinator.is_aborted());
        assert!(right.begin(&mut || false).is_err());
        assert!(right.finish().is_err());
        assert!(a.as_bytes().is_err());
        drop((coordinator, mask));
        assert!(budget.used_bytes() > baseline);
        drop(a);
        assert_eq!(budget.used_bytes(), baseline);
    }
    drop(d);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn d27_compiler_exact_short_wrong_budget_and_nonfresh_admission() {
    let _serial = super::super::TEST_SERIAL.lock().unwrap();
    let q = sigma_execution_role_split_resources().unwrap();
    for short in [true, false] {
        let budget = Budget::new(64 << 20).unwrap();
        let c = checked();
        let mut d = dealer(&c, &budget);
        drop(c);
        let (mask, run) = issue(&budget, &mut d);
        let remaining = (64 << 20)
            - budget.used_bytes()
            - q.additional_payload_bytes
            - size_of::<ExternalCharge<'static>>()
            + usize::from(short);
        // Public guard metadata also belongs to the filler, not its payload.
        let filler = budget
            .charge_external(remaining - size_of::<ExternalCharge<'static>>())
            .unwrap();
        let result = run.into_roles(&budget);
        if short {
            assert!(matches!(
                result,
                Err(SigmaExecutionError::Native(typed::Error::Resource))
            ));
        } else {
            assert_eq!(budget.used_bytes(), 64 << 20);
            drop(result.unwrap());
        }
        drop((mask, filler, d));
        assert_eq!(budget.used_bytes(), 0);
    }
    let budget = Budget::new(64 << 20).unwrap();
    let other = Budget::new(64 << 20).unwrap();
    let c = checked();
    let mut d = dealer(&c, &budget);
    drop(c);
    let baseline = budget.used_bytes();
    let (mask, run) = issue(&budget, &mut d);
    assert!(matches!(
        run.into_roles(&other),
        Err(SigmaExecutionError::BudgetIdentity)
    ));
    assert_eq!(other.used_bytes(), 0);
    drop(mask);
    assert_eq!(budget.used_bytes(), baseline);
    let (mask, mut run) = issue(&budget, &mut d);
    let frames = run.begin(&mut || false).unwrap();
    assert!(matches!(
        run.into_roles(&budget),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    drop(mask);
    assert!(budget.used_bytes() > baseline);
    drop(frames);
    assert_eq!(budget.used_bytes(), baseline);
}
