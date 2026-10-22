use super::super::owned_audit_tests::{
    accessible_role_transitions, independent_role_wrapper_population, total,
};
use super::super::owned_tests::{dealer, issue};
use super::*;

fn independently_enumerated_role_split_resources() -> SigmaExecutionRoleSplitResources {
    type C = OwnedBudgetContext;
    // These private concrete slots are additional to the accessible census.
    // Arc handle values and Arc allocation backing are distinct populations.
    let transition_bytes = total([
        accessible_role_transitions(),
        size_of::<RoleAdmissionFor<C>>(),
        size_of::<RetainedRolesFor<C>>(),
        size_of::<Arc<RetainedRolesFor<C>>>(), // Arc::new return/local
        size_of::<Arc<RetainedRolesFor<C>>>(), // coordinator clone
        size_of::<Arc<RetainedRolesFor<C>>>(), // left clone
        size_of::<Arc<RetainedRolesFor<C>>>(), // right move
    ]);
    let retention_backing_bytes = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<RetainedRolesFor<C>>())
        .unwrap()
        .0
        .pad_to_align()
        .size();
    let native_reservation_bytes = typed::local_linear_activation_role_split_resources_for::<C>()
        .unwrap()
        .reservation_bytes;
    let wrapper_population_bytes = independent_role_wrapper_population();
    SigmaExecutionRoleSplitResources {
        native_reservation_bytes,
        retention_backing_bytes,
        wrapper_population_bytes,
        transition_bytes,
        additional_payload_bytes: total([
            native_reservation_bytes,
            retention_backing_bytes,
            wrapper_population_bytes,
            transition_bytes,
        ]),
    }
}

#[test]
fn owned_role_split_private_slots_complete_independent_transition_census() {
    // No compiler quote field supplies any term of the expected payload.
    assert_eq!(
        owned_sigma_execution_role_split_resources().unwrap(),
        independently_enumerated_role_split_resources()
    );
}

fn complete(
    [left, right]: &mut [OwnedSigmaExecutionRole; 2],
) -> Vec<[OwnedSigmaExecutionRoleFrame; 2]> {
    let mut frames = [
        left.begin(&mut || false).unwrap(),
        right.begin(&mut || false).unwrap(),
    ];
    let mut history = Vec::new();
    for round in 0..11 {
        let a = left
            .advance(frames[1].as_bytes().unwrap(), &mut || false)
            .unwrap();
        let b = right
            .advance(frames[0].as_bytes().unwrap(), &mut || false)
            .unwrap();
        history.push(frames);
        match (a, b) {
            (OwnedSigmaExecutionRoleStep::Frame(a), OwnedSigmaExecutionRoleStep::Frame(b)) => {
                frames = [a, b]
            }
            (OwnedSigmaExecutionRoleStep::Ready, OwnedSigmaExecutionRoleStep::Ready) => {
                assert_eq!(round, 10);
                return history;
            }
            _ => panic!("role schedules diverged"),
        }
    }
    panic!("missing ready boundary")
}

#[test]
fn owned_role_split_exact_short_foreign_and_continuous_original_source_charge() {
    let _serial = TEST_SERIAL.lock().unwrap();
    // Exact/one-byte-short admission uses the independent census, not the
    // production total under test (which could be consistently underpriced).
    let q = independently_enumerated_role_split_resources();
    assert_eq!(owned_sigma_execution_role_split_resources().unwrap(), q);
    for short in [true, false] {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.budget().used_bytes();
        let (mask, run) = issue(&budget, &mut d).unwrap();
        let original = Arc::clone(&run.retained);
        let charge = original._charge.payload_bytes();
        let pointer = original.checked.source_byte_pointer();
        let coefficients = original
            .checked
            .certificate()
            .encoded_private_coefficients()
            .as_ptr();
        let before = budget.budget().used_bytes();
        let available =
            q.additional_payload_bytes + size_of::<OwnedExternalCharge>() - usize::from(short);
        let fill = budget
            .charge_external((64 << 20) - before - available - size_of::<OwnedExternalCharge>())
            .unwrap();
        let before = budget.budget().used_bytes();
        let result = run.into_roles(&budget);
        if short {
            assert!(matches!(
                result,
                Err(SigmaExecutionError::Native(typed::Error::Resource))
            ));
        } else {
            let (coordinator, roles) = result.unwrap();
            assert_eq!(
                budget.budget().used_bytes(),
                before + q.additional_payload_bytes + size_of::<OwnedExternalCharge>()
            );
            for role in &roles {
                assert!(Arc::ptr_eq(&role.retained, &coordinator._retained));
                assert!(Arc::ptr_eq(&role.retained._source, &original));
                assert!(role.retained._charge.is_for(budget.budget()));
                assert!(role.retained._source._charge.is_for(budget.budget()));
                assert_eq!(role.retained._source._charge.payload_bytes(), charge);
                assert_eq!(role.retained._source.checked.source_byte_pointer(), pointer);
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
            drop((coordinator, roles));
        }
        drop((mask, original, fill));
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
    let budget = OwnedBudget::new(64 << 20).unwrap();
    let foreign = OwnedBudget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.budget().used_bytes();
    let foreign_baseline = foreign.budget().used_bytes();
    let (mask, run) = issue(&budget, &mut d).unwrap();
    assert!(matches!(
        run.into_roles(&foreign),
        Err(SigmaExecutionError::BudgetIdentity)
    ));
    drop(mask);
    assert_eq!(budget.budget().used_bytes(), baseline);
    assert_eq!(foreign.budget().used_bytes(), foreign_baseline);
}

#[test]
fn owned_nonfresh_role_split_consumes_run_without_releasing_original_source_charge() {
    let _serial = TEST_SERIAL.lock().unwrap();
    for case in 0..4 {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.budget().used_bytes();
        let (mask, mut run) = issue(&budget, &mut d).unwrap();
        let source = Arc::downgrade(&run.retained);
        let charge = run.retained._charge.charged_bytes();
        let pointer = run.retained.checked.source_byte_pointer();
        let mut history = Vec::new();
        match case {
            0 | 1 => {
                let frames = run.begin(&mut || false).unwrap();
                if case == 1 {
                    let next = run
                        .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
                        .unwrap();
                    let OwnedSigmaExecutionStep::Frames(next) = next else {
                        panic!("premature output");
                    };
                    history.push(next);
                }
                history.push(frames);
            }
            2 => run.cancel(),
            _ => assert!(matches!(
                run.begin(&mut || true),
                Err(SigmaExecutionError::Native(typed::Error::Consumed))
            )),
        }
        assert!(matches!(
            run.into_roles(&budget),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        ));
        // A rejected split must not replace, shrink or detach the original
        // pinned-source/compiler guard. Mask remains the last source owner.
        assert_eq!(mask._retained._charge.charged_bytes(), charge);
        assert_eq!(mask._retained.checked.source_byte_pointer(), pointer);
        assert!(budget.budget().used_bytes() >= baseline + charge);
        drop(history);
        assert_eq!(mask._retained._charge.charged_bytes(), charge);
        assert!(source.upgrade().is_some());
        assert!(budget.budget().used_bytes() >= baseline + charge);
        drop(mask);
        assert!(source.upgrade().is_none());
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn owned_coordinator_drop_burns_started_and_unpolled_siblings() {
    let _serial = TEST_SERIAL.lock().unwrap();
    let budget = OwnedBudget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.budget().used_bytes();
    let (mask, run) = issue(&budget, &mut d).unwrap();
    let source = Arc::downgrade(&run.retained);
    let source_charge = run.retained._charge.charged_bytes();
    let (coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
    let frame = left.begin(&mut || false).unwrap();
    let role_charge = frame._retained._charge.charged_bytes();
    assert!(frame.as_bytes().is_ok());
    drop(coordinator);
    assert!(matches!(
        left.advance(&[], &mut || false),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    assert!(matches!(
        right.begin(&mut || false),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    assert!(matches!(
        left.finish(),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    assert!(matches!(
        right.finish(),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    assert!(matches!(
        frame.as_bytes(),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    drop(mask);
    assert_eq!(
        frame._retained._source._charge.charged_bytes(),
        source_charge
    );
    assert_eq!(frame._retained._charge.charged_bytes(), role_charge);
    assert!(budget.budget().used_bytes() >= baseline + source_charge + role_charge);
    assert!(source.upgrade().is_some());
    drop(frame);
    assert!(source.upgrade().is_none());
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn owned_coordinator_drop_revokes_finished_outputs_and_preserves_last_owner_charges() {
    let _serial = TEST_SERIAL.lock().unwrap();
    for output_last in [false, true] {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.budget().used_bytes();
        let (mask, run) = issue(&budget, &mut d).unwrap();
        let source = Arc::downgrade(&run.retained);
        let source_charge = run.retained._charge.charged_bytes();
        let pointer = run.retained.checked.source_byte_pointer();
        let (coordinator, mut roles) = run.into_roles(&budget).unwrap();
        let mut history = complete(&mut roles);
        let [left, right] = roles;
        let outputs = [left.finish().unwrap(), right.finish().unwrap()];
        let role_charge = outputs[0]._retained._charge.charged_bytes();
        for output in &outputs {
            assert!(output.descriptor().is_ok());
        }
        drop(coordinator);
        for output in &outputs {
            assert_eq!(
                output.descriptor(),
                Err(SigmaExecutionError::Native(typed::Error::Consumed))
            );
        }
        for pair in &history {
            for frame in pair {
                assert!(matches!(
                    frame.as_bytes(),
                    Err(SigmaExecutionError::Native(typed::Error::Consumed))
                ));
            }
        }
        drop(mask);
        if output_last {
            drop(history);
            let [output, sibling] = outputs;
            drop(sibling);
            assert_eq!(
                output._retained._source.checked.source_byte_pointer(),
                pointer
            );
            assert_eq!(
                output._retained._source._charge.charged_bytes(),
                source_charge
            );
            assert_eq!(output._retained._charge.charged_bytes(), role_charge);
            assert!(budget.budget().used_bytes() >= baseline + source_charge + role_charge);
            assert!(source.upgrade().is_some());
            drop(output);
        } else {
            let [frame, sibling] = history.pop().unwrap();
            drop((outputs, history, sibling));
            assert_eq!(
                frame._retained._source.checked.source_byte_pointer(),
                pointer
            );
            assert_eq!(
                frame._retained._source._charge.charged_bytes(),
                source_charge
            );
            assert_eq!(frame._retained._charge.charged_bytes(), role_charge);
            assert!(budget.budget().used_bytes() >= baseline + source_charge + role_charge);
            assert!(source.upgrade().is_some());
            drop(frame);
        }
        assert!(source.upgrade().is_none());
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn owned_roles_frames_outputs_outlive_factory_and_remain_revocable() {
    let _serial = TEST_SERIAL.lock().unwrap();
    let (mask, mut coordinator, mut roles, source, identity) = {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let identity = std::ptr::from_ref(budget.budget());
        let mut d = dealer(&budget);
        let (mask, run) = issue(&budget, &mut d).unwrap();
        let source = Arc::downgrade(&run.retained);
        let (coordinator, roles) = run.into_roles(&budget).unwrap();
        (mask, coordinator, roles, source, identity)
    };
    let pointer = roles[0].retained._source.checked.source_byte_pointer();
    let coefficients = roles[0]
        .retained
        ._source
        .checked
        .certificate()
        .encoded_private_coefficients()
        .as_ptr();
    let history = complete(&mut roles);
    assert_eq!(history.len(), 11);
    let [left, right] = roles;
    let outputs = [left.finish().unwrap(), right.finish().unwrap()];
    for output in &outputs {
        assert_eq!(
            std::ptr::from_ref(output._retained._source._charge.budget()),
            identity
        );
        assert_eq!(
            output._retained._source.checked.source_byte_pointer(),
            pointer
        );
        assert_eq!(
            output
                ._retained
                ._source
                .checked
                .certificate()
                .encoded_private_coefficients()
                .as_ptr(),
            coefficients
        );
        assert_eq!(
            output.descriptor().unwrap(),
            output
                ._retained
                ._source
                .checked
                .certificate()
                .activation_output_edge()
        );
    }
    coordinator.cancel();
    for output in &outputs {
        assert!(output.descriptor().is_err());
    }
    for pair in &history {
        for frame in pair {
            assert!(frame.as_bytes().is_err());
            assert!(Arc::ptr_eq(&frame._retained, &outputs[0]._retained));
        }
    }
    drop((coordinator, mask, history));
    assert!(source.upgrade().is_some()); // Revoked output alone still owns source.
    drop(outputs);
    assert!(source.upgrade().is_none());
}

#[test]
fn owned_d27_early_finish_error_drop_and_reentrant_callback_burn_both() {
    let _serial = TEST_SERIAL.lock().unwrap();
    for case in 0..4 {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.budget().used_bytes();
        let (mask, run) = issue(&budget, &mut d).unwrap();
        let (mut coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
        let a = left.begin(&mut || coordinator.is_aborted()).unwrap();
        let b = right.begin(&mut || coordinator.is_aborted()).unwrap();
        match case {
            0 => {
                assert!(left.finish().is_err());
            }
            1 => {
                assert!(left.advance(&[], &mut || false).is_err());
                drop(left);
            }
            2 => {
                drop(left);
            }
            _ => {
                assert!(left
                    .advance(b.as_bytes().unwrap(), &mut || {
                        assert!(!coordinator.is_aborted());
                        coordinator.cancel();
                        false
                    })
                    .is_err());
                drop(left);
            }
        }
        assert!(coordinator.is_aborted());
        assert!(right.advance(&[], &mut || false).is_err());
        assert!(a.as_bytes().is_err());
        assert!(b.as_bytes().is_err());
        drop((coordinator, right, mask));
        assert!(budget.budget().used_bytes() > baseline);
        drop((a, b));
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn owned_d27_one_ready_is_not_finish_authority_and_frame_alone_retains_source() {
    let _serial = TEST_SERIAL.lock().unwrap();
    let (frame, source, identity) = {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let identity = std::ptr::from_ref(budget.budget());
        let mut d = dealer(&budget);
        let (mask, run) = issue(&budget, &mut d).unwrap();
        let source = Arc::downgrade(&run.retained);
        let (coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
        let mut frames = [
            left.begin(&mut || false).unwrap(),
            right.begin(&mut || false).unwrap(),
        ];
        for _ in 0..10 {
            let a = left
                .advance(frames[1].as_bytes().unwrap(), &mut || false)
                .unwrap();
            let b = right
                .advance(frames[0].as_bytes().unwrap(), &mut || false)
                .unwrap();
            let (OwnedSigmaExecutionRoleStep::Frame(a), OwnedSigmaExecutionRoleStep::Frame(b)) =
                (a, b)
            else {
                panic!("premature ready boundary");
            };
            frames = [a, b];
        }
        assert!(matches!(
            left.advance(frames[1].as_bytes().unwrap(), &mut || false)
                .unwrap(),
            OwnedSigmaExecutionRoleStep::Ready
        ));
        // One Ready is not authority to release even one public output.
        assert!(matches!(
            left.finish(),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        )); // Right has not reached Ready.
        assert!(coordinator.is_aborted());
        assert!(matches!(
            right.advance(&[], &mut || false),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        ));
        assert!(matches!(
            right.finish(),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        ));
        let [frame, sibling] = frames;
        drop((sibling, coordinator, mask));
        (frame, source, identity)
    }; // No root, dealer, run, role, coordinator, mask or output survives.
    assert!(frame.as_bytes().is_err());
    assert_eq!(
        std::ptr::from_ref(frame._retained._charge.budget()),
        identity
    );
    assert_eq!(
        std::ptr::from_ref(frame._retained._source._charge.budget()),
        identity
    );
    assert!(source.upgrade().is_some());
    drop(frame);
    assert!(source.upgrade().is_none());
}

#[test]
fn owned_role_cancellation_and_repeated_begin_burn_both_before_sibling_progress() {
    let _serial = TEST_SERIAL.lock().unwrap();
    for case in 0..4 {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.budget().used_bytes();
        let (mask, run) = issue(&budget, &mut d).unwrap();
        let source = Arc::downgrade(&run.retained);
        let source_charge = run.retained._charge.charged_bytes();
        let pointer = run.retained.checked.source_byte_pointer();
        let coefficients = run
            .retained
            .checked
            .certificate()
            .encoded_private_coefficients()
            .as_ptr();
        let (coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
        let frame = left.begin(&mut || false).unwrap();
        let retained = Arc::downgrade(&frame._retained);
        let role_charge = frame._retained._charge.charged_bytes();
        let mut polled = false;
        let mut cancelled = || {
            polled = true;
            assert!(!coordinator.is_aborted());
            assert!(frame.as_bytes().is_ok());
            assert!(budget.budget().used_bytes() >= baseline + source_charge + role_charge);
            true
        };
        match case {
            0 => assert!(matches!(
                right.begin(&mut cancelled),
                Err(SigmaExecutionError::Native(typed::Error::Consumed))
            )),
            1 => assert!(matches!(
                left.advance(&[], &mut cancelled),
                Err(SigmaExecutionError::Native(typed::Error::Consumed))
            )),
            2 => left.cancel(),
            _ => assert!(left.begin(&mut || false).is_err()),
        }
        assert_eq!(polled, case < 2);
        assert!(coordinator.is_aborted());
        let mut post_abort_polls = 0;
        let mut after_abort = || {
            post_abort_polls += 1;
            false
        };
        assert!(matches!(
            left.begin(&mut after_abort),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        ));
        assert!(matches!(
            right.begin(&mut after_abort),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        ));
        assert_eq!(post_abort_polls, 0, "aborted peers must not call user code");
        assert!(matches!(
            left.finish(),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        ));
        assert!(matches!(
            right.finish(),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        ));
        assert_eq!(
            frame.as_bytes(),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        );
        drop((mask, coordinator));
        assert_eq!(
            frame._retained._source.checked.source_byte_pointer(),
            pointer
        );
        assert_eq!(
            frame
                ._retained
                ._source
                .checked
                .certificate()
                .encoded_private_coefficients()
                .as_ptr(),
            coefficients
        );
        assert!(frame._retained._source._charge.is_for(budget.budget()));
        assert!(frame._retained._charge.is_for(budget.budget()));
        assert_eq!(
            frame._retained._source._charge.charged_bytes(),
            source_charge
        );
        assert_eq!(frame._retained._charge.charged_bytes(), role_charge);
        assert!(budget.budget().used_bytes() >= baseline + source_charge + role_charge);
        drop(frame);
        assert!(retained.upgrade().is_none());
        assert!(source.upgrade().is_none());
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn owned_role_begin_callback_unwind_revokes_started_peer_without_detaching_charges() {
    let _serial = TEST_SERIAL.lock().unwrap();
    let budget = OwnedBudget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.budget().used_bytes();
    let (mask, run) = issue(&budget, &mut d).unwrap();
    let source = Arc::downgrade(&run.retained);
    let source_charge = run.retained._charge.charged_bytes();
    let (coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
    let frame = left.begin(&mut || false).unwrap();
    let retained = Arc::downgrade(&frame._retained);
    let role_charge = frame._retained._charge.charged_bytes();
    let before_poll = budget.budget().used_bytes();
    let mut polled = false;
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = right.begin(&mut || {
            polled = true;
            assert!(!coordinator.is_aborted());
            assert!(frame.as_bytes().is_ok());
            assert_eq!(budget.budget().used_bytes(), before_poll);
            panic!("owned role begin callback unwind");
        });
    }));
    assert!(polled);
    let panic = panic.expect_err("callback panic must propagate");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"owned role begin callback unwind")
    );
    assert!(coordinator.is_aborted());
    assert!(matches!(
        left.advance(&[], &mut || panic!("aborted callback was polled")),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    assert!(matches!(
        right.begin(&mut || panic!("aborted callback was polled")),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    assert_eq!(
        frame.as_bytes(),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    );
    drop((left, right, coordinator, mask));
    assert_eq!(
        frame._retained._source._charge.charged_bytes(),
        source_charge
    );
    assert_eq!(frame._retained._charge.charged_bytes(), role_charge);
    assert!(frame._retained._source._charge.is_for(budget.budget()));
    assert!(frame._retained._charge.is_for(budget.budget()));
    assert!(budget.budget().used_bytes() >= baseline + source_charge + role_charge);
    drop(frame);
    assert!(retained.upgrade().is_none());
    assert!(source.upgrade().is_none());
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn owned_ready_sibling_drop_revokes_already_finished_peer_output() {
    let _serial = TEST_SERIAL.lock().unwrap();
    let budget = OwnedBudget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.budget().used_bytes();
    let (mask, run) = issue(&budget, &mut d).unwrap();
    let source = Arc::downgrade(&run.retained);
    let pointer = run.retained.checked.source_byte_pointer();
    let source_charge = run.retained._charge.charged_bytes();
    let (coordinator, mut roles) = run.into_roles(&budget).unwrap();
    let history = complete(&mut roles);
    assert_eq!(history.len(), 11);
    let [left, right] = roles;
    let output = left.finish().unwrap();
    let retained = Arc::downgrade(&output._retained);
    let role_charge = output._retained._charge.charged_bytes();
    assert_eq!(output.role(), 0);
    assert_eq!(right.role(), 1);
    assert_eq!(
        output.descriptor().unwrap(),
        output
            ._retained
            ._source
            .checked
            .certificate()
            .activation_output_edge()
    );
    assert!(!coordinator.is_aborted());
    drop(right); // Ready alone is not a successful consuming handoff.
    assert!(coordinator.is_aborted());
    assert_eq!(
        output.descriptor(),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    );
    for frames in &history {
        for frame in frames {
            assert_eq!(
                frame.as_bytes(),
                Err(SigmaExecutionError::Native(typed::Error::Consumed))
            );
        }
    }
    drop((mask, coordinator, history));
    assert_eq!(
        output._retained._source.checked.source_byte_pointer(),
        pointer
    );
    assert_eq!(
        output._retained._source._charge.charged_bytes(),
        source_charge
    );
    assert_eq!(output._retained._charge.charged_bytes(), role_charge);
    assert!(output._retained._source._charge.is_for(budget.budget()));
    assert!(output._retained._charge.is_for(budget.budget()));
    assert!(budget.budget().used_bytes() >= baseline + source_charge + role_charge);
    drop(output);
    assert!(retained.upgrade().is_none());
    assert!(source.upgrade().is_none());
    assert_eq!(budget.budget().used_bytes(), baseline);
}
