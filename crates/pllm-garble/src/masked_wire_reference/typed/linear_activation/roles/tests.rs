use super::super::tests::{dealer, oracle, spec, unmask};
use super::*;
use std::sync::atomic::Ordering;

thread_local! {
    static LOCKED_OUTPUT_PANIC_ROLE: std::cell::Cell<Option<u8>> = const {
        std::cell::Cell::new(None)
    };
}

// Armed only by the locked-work regression on its own worker thread. This runs
// before unwind, never from a caller callback or a Drop implementation.
pub(super) fn panic_after_locked_output(role: u8, state: &RoleState<'_>, control: &SplitControl) {
    LOCKED_OUTPUT_PANIC_ROLE.with(|armed| {
        if armed.get() == Some(role) {
            armed.set(None);
            assert!(state.output.is_some());
            assert_eq!(control.ready, [true; 2]);
            panic!("injected locked output work panic");
        }
    });
}

fn issue<'b>(
    d: &mut LocalDealer<'b>,
    kind: ActivationKind,
) -> (DealerMask<'b, Ring>, LinearActivationRun<'b>) {
    d.linear_activation_from_values_with(spec(1, 2, 1, kind), &[-1, -1], &[1, 4095], &mut |bytes| {
        bytes.fill(173);
        Ok(())
    })
    .unwrap()
}

fn assert_burned(shared: &Coordination<'_>) {
    assert!(shared.control().aborted);
    for slot in &shared.slots {
        let state = slot.state.lock().unwrap_or_else(|p| p.into_inner());
        assert!(state.party.phase == Phase::Dead);
        assert_eq!(state.party.ledger.next, USES as u8);
        assert!(state.party.roots.is_none());
        assert!(state.party.dot.is_none());
        assert!(state.output.is_none());
    }
}

fn complete<'b>(
    [left, right]: &mut [LinearActivationRole<'b>; 2],
) -> Vec<[LinearActivationRoleFrame<'b>; 2]> {
    let mut frames = [
        left.begin(&mut || false).unwrap(),
        right.begin(&mut || false).unwrap(),
    ];
    let mut retained = Vec::new();
    for round in 0..EXCHANGES {
        let a = left
            .advance(frames[1].as_bytes().unwrap(), &mut || false)
            .unwrap();
        let b = right
            .advance(frames[0].as_bytes().unwrap(), &mut || false)
            .unwrap();
        retained.push(frames);
        match (a, b) {
            (LinearActivationRoleStep::Frame(a), LinearActivationRoleStep::Frame(b)) => {
                assert!(round < EXCHANGES - 1);
                frames = [a, b];
            }
            (LinearActivationRoleStep::Ready, LinearActivationRoleStep::Ready) => {
                assert_eq!(round, EXCHANGES - 1);
                return retained;
            }
            _ => panic!("role schedules diverged"),
        }
    }
    panic!("missing ready boundary")
}

#[test]
fn d27_quote_is_concrete_checked_and_preconstruction() {
    let q = role_split_resources().unwrap();
    assert_eq!(
        q.role_backing_bytes,
        arc_bytes::<RoleSlot<'static>>().unwrap()
    );
    assert_eq!(
        q.coordination_backing_bytes,
        arc_bytes::<Coordination<'static>>().unwrap()
    );
    assert_eq!(
        q.reservation_bytes,
        2 * q.role_backing_bytes
            + q.coordination_backing_bytes
            + q.wrapper_population_bytes
            + q.transition_bytes
    );
    assert_eq!(metadata_sum(&[usize::MAX, 1]), Err(Error::Resource));
    assert!(q.transition_bytes >= size_of::<Result<RoleSplitResult<'static>, Error>>());
}

#[test]
fn d27_reentrant_callback_inspect_cancel_and_panic_do_not_hold_either_lock() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    // Detached on timeout, so a regression fails this test rather than joining
    // a deadlocked scoped thread forever. Only join after completion was sent.
    let worker = std::thread::spawn(move || {
        for case in 0..6 {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let (mask, run) = issue(&mut d, ActivationKind::Silu);
            let (mut coordinator, [mut left, mut right]) = run.into_roles().unwrap();
            if case == 0 {
                let a = left.begin(&mut || coordinator.is_aborted()).unwrap();
                let b = right.begin(&mut || coordinator.is_aborted()).unwrap();
                assert!(left
                    .advance(b.as_bytes().unwrap(), &mut || coordinator.is_aborted())
                    .is_ok());
                assert!(right
                    .advance(a.as_bytes().unwrap(), &mut || coordinator.is_aborted())
                    .is_ok());
            } else if case == 1 {
                assert!(left
                    .begin(&mut || {
                        coordinator.cancel();
                        false
                    })
                    .is_err());
                assert_burned(&coordinator.shared);
            } else if case == 2 {
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = left.begin(&mut || {
                        assert!(!coordinator.is_aborted());
                        panic!("poll panic");
                    });
                }))
                .is_err());
                assert_burned(&coordinator.shared);
            } else {
                let _a = left.begin(&mut || coordinator.is_aborted()).unwrap();
                let b = right.begin(&mut || coordinator.is_aborted()).unwrap();
                if case == 3 {
                    assert!(left
                        .advance(b.as_bytes().unwrap(), &mut || {
                            coordinator.cancel();
                            false
                        })
                        .is_err());
                } else if case == 4 {
                    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = left.advance(b.as_bytes().unwrap(), &mut || {
                            assert!(!coordinator.is_aborted());
                            panic!("advance poll panic");
                        });
                    }))
                    .is_err());
                } else {
                    // Inspection of own role-slot metadata is test-private; it
                    // proves the callback also runs outside the role-slot lock.
                    assert!(left
                        .advance(b.as_bytes().unwrap(), &mut || {
                            assert!(!coordinator.is_aborted());
                            assert!(coordinator.shared.slots[0].state.try_lock().is_ok());
                            coordinator.cancel();
                            true
                        })
                        .is_err());
                }
                assert_burned(&coordinator.shared);
            }
            drop((left, right, coordinator, mask, d));
            assert_eq!(budget.used_bytes(), 0);
        }
        send.send(()).unwrap();
    });
    receive
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("reentrant callback deadlocked or panicked");
    worker.join().unwrap();
}

#[test]
fn d27_locked_output_unwind_burns_both_slots_and_poison_recovery_is_nonblocking() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    // Do not join a deadlocked worker on timeout. Its budget and all borrowed
    // owners live inside the thread, as in the reentrant-callback regression.
    let worker = std::thread::spawn(move || {
        for faulting_role in 0..2 {
            for coordinator_first in [false, true] {
                let budget = Budget::new(64 << 20).unwrap();
                let mut d = dealer(&budget);
                let baseline = budget.used_bytes();
                let (mask, run) = issue(&mut d, ActivationKind::Silu);
                let (mut coordinator, mut roles) = run.into_roles().unwrap();
                let mut frames = [
                    roles[0].begin(&mut || false).unwrap(),
                    roles[1].begin(&mut || false).unwrap(),
                ];
                let mut retained = Vec::new();
                for _ in 0..EXCHANGES - 1 {
                    let a = roles[0]
                        .advance(frames[1].as_bytes().unwrap(), &mut || false)
                        .unwrap();
                    let b = roles[1]
                        .advance(frames[0].as_bytes().unwrap(), &mut || false)
                        .unwrap();
                    retained.push(frames);
                    frames = match (a, b) {
                        (
                            LinearActivationRoleStep::Frame(a),
                            LinearActivationRoleStep::Frame(b),
                        ) => [a, b],
                        _ => panic!("early ready"),
                    };
                }
                let sibling = 1 - faulting_role;
                assert!(matches!(
                    roles[sibling]
                        .advance(frames[faulting_role].as_bytes().unwrap(), &mut || false)
                        .unwrap(),
                    LinearActivationRoleStep::Ready
                ));
                assert!(coordinator.shared.slots[sibling]
                    .state
                    .lock()
                    .unwrap()
                    .output
                    .is_some());
                let peer = frames[sibling].as_bytes().unwrap();
                LOCKED_OUTPUT_PANIC_ROLE.with(|armed| armed.set(Some(faulting_role as u8)));
                let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = roles[faulting_role].advance(peer, &mut || false);
                }))
                .expect_err("locked-work injection did not panic");
                assert_eq!(
                    panic.downcast_ref::<&str>(),
                    Some(&"injected locked output work panic")
                );
                LOCKED_OUTPUT_PANIC_ROLE.with(|armed| assert_eq!(armed.get(), None));

                // Do NOT call control()/assert_burned first: poison recovery
                // would itself abort both and hide a broken Operation::drop.
                let control = match coordinator.shared.control.try_lock() {
                    Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
                    _ => panic!("locked unwind did not release and poison control"),
                };
                assert!(control.aborted);
                drop(control);
                for (role, slot) in coordinator.shared.slots.iter().enumerate() {
                    assert_eq!(slot.state.is_poisoned(), role == faulting_role);
                    let state = match slot.state.try_lock() {
                        Ok(state) => state,
                        Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
                        Err(std::sync::TryLockError::WouldBlock) => {
                            panic!("role slot still locked")
                        }
                    };
                    assert!(state.party.phase == Phase::Dead);
                    assert_eq!(state.party.ledger.next, USES as u8);
                    assert!(state.party.roots.is_none());
                    assert!(state.party.dot.is_none());
                    assert!(state.output.is_none());
                }

                // Exercise public poison recovery and repeated abort with no
                // callback under either lock, including the untouched sibling.
                assert!(coordinator.is_aborted());
                coordinator.cancel();
                for role in &mut roles {
                    role.cancel();
                    assert!(role.begin(&mut || false).is_err());
                    assert!(role.advance(&[], &mut || false).is_err());
                }
                for frame in retained.iter().chain(std::iter::once(&frames)).flatten() {
                    assert_eq!(frame.as_bytes(), Err(Error::Consumed));
                }
                assert_eq!(retained.len() + 1, EXCHANGES);
                assert_burned(&coordinator.shared);
                if coordinator_first {
                    drop(coordinator);
                    for role in roles {
                        assert!(role.finish().is_err());
                    }
                } else {
                    for role in roles {
                        assert!(role.finish().is_err());
                    }
                    drop(coordinator);
                }
                drop((frames, mask));
                // Earlier original frames keep their native singleton/leases;
                // poison recovery must not refund those retained owners early.
                assert!(budget.used_bytes() > baseline);
                assert!(matches!(
                    matrix::party::IssuanceTicket::acquire(),
                    Err(Error::Resource)
                ));
                drop(retained);
                assert_eq!(budget.used_bytes(), baseline);
                drop(matrix::party::IssuanceTicket::acquire().unwrap());
                drop(d);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
        send.send(()).unwrap();
    });
    receive
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("locked-work poison recovery or retirement deadlocked or panicked");
    worker.join().unwrap();
}

#[test]
fn d27_original_owners_and_eleven_frames_match_coupled_mod_fresh_matrix_issuance() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        // Test-only fresh deterministic setup. Never clone/reset a live run.
        let budget = Budget::new(64 << 20).unwrap();
        let expected = oracle(spec(1, 2, 1, kind), &[-1, -1], &[1, 4095]);
        let coupled_frames = {
            let mut d = dealer(&budget);
            d.namespace = [79; 32];
            let (mask, mut run) = issue(&mut d, kind);
            let mut frames = run.begin(&mut || false).unwrap();
            let mut transcript = Vec::new();
            for round in 0..EXCHANGES {
                transcript.push([frames[0].as_bytes().to_vec(), frames[1].as_bytes().to_vec()]);
                match run
                    .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
                    .unwrap()
                    .body
                {
                    StepBody::Frames(next, _) => frames = next,
                    StepBody::Output(outputs, _) => {
                        assert_eq!(round, EXCHANGES - 1);
                        for output in &outputs {
                            assert_eq!(unmask(&mask, output), expected);
                        }
                        break;
                    }
                }
            }
            transcript
        };
        assert_eq!(budget.used_bytes(), 0);
        let mut d = dealer(&budget);
        d.namespace = [79; 32];
        let baseline = budget.used_bytes();
        let (mask, run) = issue(&mut d, kind);
        let roots = std::array::from_fn::<_, 2, _>(|role| {
            run.parties[role]
                .roots
                .as_ref()
                .unwrap()
                .each_ref()
                .map(|x| x.backing.values().as_ptr())
        });
        let bindings = run.parties[0].ledger.bindings;
        let counter = d.counter;
        let mut draws = 0;
        let q = role_split_resources().unwrap();
        let mut charge = budget.charge_external(q.reservation_bytes).unwrap();
        let used = budget.used_bytes();
        let (mut coordinator, mut roles) = run.into_roles_from_reserved(&mut charge).unwrap();
        // Exact original reservation transfer: no entropy or increased/decreased
        // accounting even though three new Arc backings now exist.
        assert_eq!(budget.used_bytes(), used);
        assert_eq!(charge.payload_bytes(), 0);
        assert_eq!(d.counter, counter);
        for (role, handle) in roles.iter().enumerate() {
            assert_eq!(handle.role(), role as u8);
            let state = handle.slot.state.lock().unwrap();
            assert_eq!(state.party.ledger.bindings, bindings);
            assert_eq!(
                state
                    .party
                    .roots
                    .as_ref()
                    .unwrap()
                    .each_ref()
                    .map(|x| x.backing.values().as_ptr()),
                roots[role]
            );
        }
        let retained = complete(&mut roles);
        assert_eq!(retained.len(), EXCHANGES);
        for (round, (pair, expected_pair)) in retained.iter().zip(&coupled_frames).enumerate() {
            for role in 0..2 {
                let actual = pair[role].as_bytes().unwrap();
                let expected = &expected_pair[role];
                if round == 0 {
                    // Matrix binds a process-monotonic native issuance ticket.
                    // Two FRESH setups must differ here; do not reset a live
                    // counter or weaken its identity to manufacture byte parity.
                    assert_eq!(&actual[..8], b"PLLMMX01");
                    assert_ne!(&actual[8..40], &expected[8..40]);
                    assert_eq!(&actual[..8], &expected[..8]);
                    assert_eq!(&actual[40..], &expected[40..]);
                } else {
                    assert_eq!(actual, expected);
                }
            }
        }
        let [left, right] = roles;
        let outputs = [left.finish().unwrap(), right.finish().unwrap()];
        for output in &outputs {
            assert_eq!(output.descriptor().unwrap(), spec(1, 2, 1, kind).edges[4]);
            let state = output.slot.state.lock().unwrap();
            assert_eq!(unmask(&mask, state.output.as_ref().unwrap()), expected);
        }
        // Retained original matrix frame, not unrelated final output, owns its
        // singleton. Another issuance rejects before the injected root draw.
        assert!(matches!(
            d.linear_activation_from_values_with(
                spec(1, 2, 1, kind),
                &[-1, -1],
                &[1, 4095],
                &mut |_| {
                    draws += 1;
                    Ok(())
                }
            ),
            Err(Error::Resource)
        ));
        assert_eq!(draws, 0);
        drop(retained);
        let second = issue(&mut d, kind);
        drop(second);
        coordinator.cancel();
        assert_burned(&coordinator.shared);
        for output in &outputs {
            assert_eq!(output.descriptor(), Err(Error::Consumed));
        }
        drop((outputs, coordinator, mask, charge));
        assert_eq!(budget.used_bytes(), baseline);
        drop(d);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn d27_first_failure_cancels_unpolled_sibling_and_revokes_retained_frames() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    for case in 0..6 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let (mask, run) = issue(&mut d, ActivationKind::Silu);
        let (coordinator, [mut left, mut right]) = run.into_roles().unwrap();
        let a = left.begin(&mut || false).unwrap();
        let b = right.begin(&mut || false).unwrap();
        match case {
            0 => {
                assert!(left.advance(a.as_bytes().unwrap(), &mut || false).is_err());
            } // wrong role
            1 => {
                let mut foreign_session = b.as_bytes().unwrap().to_vec();
                foreign_session[8] ^= 1;
                assert!(left.advance(&foreign_session, &mut || false).is_err());
            }
            2 => {
                assert!(left.advance(b.as_bytes().unwrap(), &mut || false).is_ok());
                assert!(left.advance(b.as_bytes().unwrap(), &mut || false).is_err());
                // replay
            }
            3 => {
                assert!(left.advance(b.as_bytes().unwrap(), &mut || true).is_err());
            }
            4 => {
                assert!(left.begin(&mut || false).is_err());
            }
            5 => {
                // Panicking caller callback also burns both immediately, without
                // waiting for either evaluator to poll a shared abort bit.
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = left.advance(b.as_bytes().unwrap(), &mut || panic!("cancel fault"));
                }))
                .is_err());
            }
            _ => unreachable!(),
        }
        assert_burned(&coordinator.shared);
        assert_eq!(a.as_bytes(), Err(Error::Consumed));
        assert_eq!(b.as_bytes(), Err(Error::Consumed));
        assert!(right.advance(&[], &mut || false).is_err());
        assert!(right.finish().is_err());
        drop((left, coordinator, mask));
        assert!(budget.used_bytes() > baseline);
        assert!(matches!(
            matrix::party::IssuanceTicket::acquire(),
            Err(Error::Resource)
        ));
        drop(a);
        assert!(matches!(
            matrix::party::IssuanceTicket::acquire(),
            Err(Error::Resource)
        ));
        drop(b);
        assert_eq!(budget.used_bytes(), baseline);
        drop(matrix::party::IssuanceTicket::acquire().unwrap());
    }
}

#[test]
fn d27_ready_is_not_output_and_drop_or_late_failure_erases_both() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    for case in 0..4 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, run) = issue(&mut d, ActivationKind::Silu);
        let (coordinator, [mut left, mut right]) = run.into_roles().unwrap();
        if case == 0 {
            assert!(left.finish().is_err());
            assert_burned(&coordinator.shared);
            assert!(right.begin(&mut || false).is_err());
        } else if case == 1 {
            drop(left);
            assert_burned(&coordinator.shared);
            assert!(right.begin(&mut || false).is_err());
        } else {
            let mut a = left.begin(&mut || false).unwrap();
            let mut b = right.begin(&mut || false).unwrap();
            for round in 0..EXCHANGES {
                let step = left.advance(b.as_bytes().unwrap(), &mut || false).unwrap();
                if round == EXCHANGES - 1 {
                    assert!(matches!(step, LinearActivationRoleStep::Ready));
                    assert!(left.slot.state.lock().unwrap().output.is_some());
                    if case == 2 {
                        assert!(left.finish().is_err());
                    } else {
                        assert!(right.advance(&[], &mut || false).is_err());
                    }
                    assert_burned(&coordinator.shared);
                    break;
                }
                let other = right.advance(a.as_bytes().unwrap(), &mut || false).unwrap();
                match (step, other) {
                    (
                        LinearActivationRoleStep::Frame(next_a),
                        LinearActivationRoleStep::Frame(next_b),
                    ) => {
                        a = next_a;
                        b = next_b;
                    }
                    _ => panic!("early ready"),
                }
            }
        }
        drop((right, coordinator, mask));
    }
}

#[test]
fn d27_escaped_output_revoked_by_sibling_replay_and_output_first_drop_is_safe() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    for replay in [false, true] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let (mask, run) = issue(&mut d, ActivationKind::Silu);
        let (coordinator, mut roles) = run.into_roles().unwrap();
        let frames = complete(&mut roles);
        let [left, mut right] = roles;
        let left = left.finish().unwrap();
        assert!(left.descriptor().is_ok());
        if replay {
            assert!(right.advance(&[], &mut || false).is_err());
            assert_burned(&coordinator.shared);
            assert_eq!(left.descriptor(), Err(Error::Consumed));
            assert!(right.finish().is_err());
            drop(left);
        } else {
            // Once BOTH calculations are ready, retiring one consumed output
            // does not destroy the other role's independent final capability.
            drop(left);
            let right = right.finish().unwrap();
            assert_eq!(right.role(), 1);
            assert!(right.descriptor().is_ok());
            drop(right);
        }
        drop((coordinator, mask));
        assert!(budget.used_bytes() > baseline);
        drop(frames);
        assert_eq!(budget.used_bytes(), baseline);
    }
}

#[test]
fn d27_exact_one_short_fresh_only_budget_and_allocation_faults() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let q = role_split_resources().unwrap();
    for short in [true, false] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, run) = issue(&mut d, ActivationKind::Silu);
        let filler = budget
            .reserve((64 << 20) - budget.used_bytes() - q.reservation_bytes + usize::from(short))
            .unwrap();
        let result = run.into_roles();
        if short {
            assert!(matches!(result, Err(Error::Resource)));
        } else {
            assert_eq!(budget.used_bytes(), 64 << 20);
            drop(result.unwrap());
        }
        drop((mask, filler, d));
        assert_eq!(budget.used_bytes(), 0);
    }
    for fault in 0..3 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let (mask, run) = issue(&mut d, ActivationKind::Silu);
        budget.fail_after.store(fault, Ordering::SeqCst);
        assert!(matches!(run.into_roles(), Err(Error::Allocation)));
        budget.fail_after.store(-1, Ordering::SeqCst);
        drop(mask);
        assert_eq!(budget.used_bytes(), baseline);
        drop(issue(&mut d, ActivationKind::Silu)); // original singleton retired
    }
    let budget = Budget::new(64 << 20).unwrap();
    let other = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    for foreign in [true, false] {
        let (mask, run) = issue(&mut d, ActivationKind::Silu);
        let target = if foreign { &other } else { &budget };
        let mut charge = target
            .charge_external(q.reservation_bytes - usize::from(!foreign))
            .unwrap();
        let before = charge.payload_bytes();
        assert!(matches!(
            run.into_roles_from_reserved(&mut charge),
            Err(Error::Resource)
        ));
        assert_eq!(charge.payload_bytes(), before);
        drop((charge, mask));
    }
    let (mask, mut run) = issue(&mut d, ActivationKind::Silu);
    let frames = run.begin(&mut || false).unwrap();
    assert!(matches!(run.into_roles(), Err(Error::Consumed)));
    drop((frames, mask));
}
