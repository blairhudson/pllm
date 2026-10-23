//! Actual D29 lifecycle cases; UNRUN until the parent grants a native test slot.
//! Fresh production entropy/owners, not replayed seeded keys or a second machine.
//! Raw snapshots ALWAYS precede any recovery-capable API after the first fault.
use super::*;

type OwnedRole = Role<OwnedBudgetContext>;
type OwnedFrame = Frame<OwnedBudgetContext>;
type OwnedOutput = Output<OwnedBudgetContext>;

struct Fixture {
    root: Option<OwnedBudget>,
    mask: Option<Mask<OwnedBudgetContext>>,
    coordinator: Option<Coordinator<OwnedBudgetContext>>,
    roles: [Option<OwnedRole>; 2],
    frames: [Vec<OwnedFrame>; 2],
    outputs: [Option<OwnedOutput>; 2],
}

impl Fixture {
    fn fresh() -> Self {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let source = source();
        let s = spec(source, 2);
        let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
        let input = owner(&root, &[0, 65535], 5);
        let (_, working, original) = owned_working(&dealer, &root, &s, &input);
        // Use the existing OS-entropy constructor: every case has its own original
        // counter, namespace, root masks and complete native child material.
        let (mask, coordinator, [left, right]) = construct(&mut dealer, s, input, working).unwrap();
        drop((original, dealer));
        Self {
            root: Some(root),
            mask: Some(mask),
            coordinator: Some(coordinator),
            roles: [Some(left), Some(right)],
            frames: [Vec::new(), Vec::new()],
            outputs: [None, None],
        }
    }

    fn shared(&self) -> &Arc<Shared<OwnedBudgetContext>> {
        if let Some(coordinator) = &self.coordinator {
            &coordinator.shared
        } else {
            &self.mask.as_ref().unwrap().shared
        }
    }

    fn budget(&self) -> &Budget {
        self.root.as_ref().unwrap().budget()
    }

    fn begin(&mut self, role: usize) -> Result<(), Error> {
        let frame = self.roles[role].as_mut().unwrap().begin(&mut || false)?;
        self.frames[role].push(frame);
        Ok(())
    }

    fn advance(&mut self, role: usize, stage: usize) -> Result<bool, Error> {
        // Keep EVERY earlier frame, including a unilateral frame before a later
        // sibling failure. Fetch the old peer stage, never the just-produced one.
        let peer = self.frames[1 - role][stage - 1].bytes()?;
        match self.roles[role]
            .as_mut()
            .unwrap()
            .advance(peer, &mut || false)?
        {
            Step::Frame(frame) => {
                assert!(stage < ROUNDS);
                self.frames[role].push(frame);
                Ok(false)
            }
            Step::Ready => {
                assert_eq!(stage, ROUNDS);
                Ok(true)
            }
        }
    }

    fn at_stage(stage: usize) -> Self {
        assert!(stage <= ROUNDS);
        let mut case = Self::fresh();
        if stage != 0 {
            case.begin(0).unwrap();
            case.begin(1).unwrap();
            for current in 1..stage {
                assert!(!case.advance(0, current).unwrap());
                assert!(!case.advance(1, current).unwrap());
            }
        }
        case
    }

    fn peer_copy(&self, role: usize, stage: usize) -> Vec<u8> {
        self.frames[1 - role][stage - 1].bytes().unwrap().to_vec()
    }

    fn drive(&mut self) -> Result<(), Error> {
        self.begin(0)?;
        self.begin(1)?;
        for stage in 1..=ROUNDS {
            assert_eq!(self.advance(0, stage)?, stage == ROUNDS);
            assert_eq!(self.advance(1, stage)?, stage == ROUNDS);
        }
        for role in 0..2 {
            self.outputs[role] = Some(self.roles[role].take().unwrap().finish()?);
        }
        Ok(())
    }
}

fn raw_retired(case: &Fixture) {
    // This helper does not call control(), operation(), abort(), frame.bytes()
    // or descriptor(). Poison recovery must not manufacture the assertion.
    immediate_abort(case.shared());
    for slot in &case.shared().slots {
        let state = match slot.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::Poisoned(poison)) => poison.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => panic!("state retained after fault"),
        };
        assert!(state.drelu.is_none() && state.select.is_none() && state.tr.is_none());
        assert!(state.h.is_none() && state.v.is_none() && state.j1.is_none());
        assert!(std::ptr::eq(
            state.tables.input.budget(),
            state.lease.budget.budget()
        ));
        assert!(state.lease.bytes >= 5 * 8 * state.tables.input.payload().len());
        // The source is STILL owned, not refunded/cleared prematurely on abort.
        // Safe raw snapshots prove work-buffer erasure under live original leases.
        assert_eq!(state.tables.input.payload().as_slice(), &[0, 65535]);
        assert!(state.tables.input.charged_bytes() >= 7 * 8 + size_of::<Input>());
    }
}

fn never_restored(case: &mut Fixture) {
    raw_retired(case); // MUST be first: APIs below can recover poisoned locks.
    for _ in 0..2 {
        for role in case.roles.iter_mut().flatten() {
            assert!(matches!(role.begin(&mut || false), Err(Error::Consumed)));
            assert!(matches!(
                role.advance(&[], &mut || false),
                Err(Error::Consumed)
            ));
        }
        assert!(case
            .frames
            .iter()
            .flatten()
            .all(|frame| frame.bytes().is_err()));
        assert!(case
            .outputs
            .iter()
            .flatten()
            .all(|output| output.descriptor().is_err()));
        raw_retired(case);
    }
}

fn assert_callback_unlocked(shared: &Shared<OwnedBudgetContext>) {
    let control = shared
        .control
        .try_lock()
        .expect("callback retained metadata lock");
    assert!(!control.aborted);
    drop(control);
    for slot in &shared.slots {
        drop(
            slot.state
                .try_lock()
                .expect("callback retained native state lock"),
        );
    }
}

#[test]
fn true_cancellation_and_coordinator_reentry_burn_every_prefix_for_both_roles() {
    for stage in 0..=ROUNDS {
        for role in 0..2 {
            for cancel_inside in [false, true] {
                let mut case = Fixture::at_stage(stage);
                let peer = if stage == 0 {
                    Vec::new()
                } else {
                    case.peer_copy(role, stage)
                };
                let coordinator = case.coordinator.as_mut().unwrap();
                let cancelled = &mut || {
                    assert_callback_unlocked(&coordinator.shared);
                    if cancel_inside {
                        coordinator.cancel();
                        false
                    } else {
                        true
                    }
                };
                if stage == 0 {
                    assert!(matches!(
                        case.roles[role].as_mut().unwrap().begin(cancelled),
                        Err(Error::Consumed)
                    ));
                } else {
                    assert!(matches!(
                        case.roles[role].as_mut().unwrap().advance(&peer, cancelled),
                        Err(Error::Consumed)
                    ));
                }
                never_restored(&mut case);
            }
        }
    }
}

#[test]
fn callback_and_locked_work_unwind_synchronously_erase_every_actual_stage_both_roles() {
    for stage in 1..=ROUNDS {
        for role in 0..2 {
            for locked in [false, true] {
                let mut case = Fixture::at_stage(stage);
                let peer = case.peer_copy(role, stage);
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if locked {
                        // The existing private native-operation fault seam. No
                        // new production callback/allocator/panic hook is added.
                        let _: Result<(), Error> = case.shared().operation(|_| {
                            let mut state = case.shared().slots[role].state.lock().unwrap();
                            assert_eq!(usize::from(state.phase), stage);
                            state.work[0][0] = 0x5a;
                            panic!("D29 locked native work at stage {stage}, role {role}")
                        });
                    } else {
                        let shared = &case.coordinator.as_ref().unwrap().shared;
                        let _ = case.roles[role].as_mut().unwrap().advance(&peer, &mut || {
                            assert_callback_unlocked(shared);
                            panic!("D29 unlocked callback at stage {stage}, role {role}")
                        });
                    }
                }))
                .is_err());
                never_restored(&mut case);
            }
        }
    }
}

#[test]
fn first_wrong_role_peer_at_every_stage_is_irreversible_for_both_parties() {
    for stage in 1..=ROUNDS {
        for role in 0..2 {
            let mut case = Fixture::at_stage(stage);
            let honest = case.peer_copy(role, stage);
            let mut wrong = honest.clone();
            wrong[72] = role as u8;
            assert!(case.roles[role]
                .as_mut()
                .unwrap()
                .advance(&wrong, &mut || false)
                .is_err());
            raw_retired(&case);
            assert!(matches!(
                case.roles[role]
                    .as_mut()
                    .unwrap()
                    .advance(&honest, &mut || false),
                Err(Error::Consumed)
            ));
            never_restored(&mut case);
        }
    }
}

#[test]
fn fresh_foreign_invocation_frames_reject_at_every_stage_without_burning_their_owner() {
    for stage in 1..=ROUNDS {
        for role in 0..2 {
            let mut case = Fixture::at_stage(stage);
            let mut foreign = Fixture::at_stage(stage);
            assert!(!std::ptr::eq(case.budget(), foreign.budget()));
            assert_ne!(
                case.shared().slots[0].state.try_lock().unwrap().bindings[0].namespace,
                foreign.shared().slots[0].state.try_lock().unwrap().bindings[0].namespace
            );
            // Same public wire/source/session labels and geometry; DIFFERENT
            // genuine namespace, invocation, root masks and native child keys.
            let peer = foreign.peer_copy(role, stage);
            assert!(case.roles[role]
                .as_mut()
                .unwrap()
                .advance(&peer, &mut || false)
                .is_err());
            never_restored(&mut case);
            assert!(!foreign.shared().control.try_lock().unwrap().aborted);
            for current in stage..=ROUNDS {
                assert_eq!(foreign.advance(0, current).unwrap(), current == ROUNDS);
                assert_eq!(foreign.advance(1, current).unwrap(), current == ROUNDS);
            }
            for other in 0..2 {
                foreign.outputs[other] =
                    Some(foreign.roles[other].take().unwrap().finish().unwrap());
            }
            assert!(foreign
                .outputs
                .iter()
                .flatten()
                .all(|output| output.descriptor().is_ok()));
        }
    }
}

#[test]
fn repeated_begin_and_ready_advance_burn_instead_of_reusing_material() {
    for stage in 1..=ROUNDS {
        for role in 0..2 {
            let mut case = Fixture::at_stage(stage);
            assert!(matches!(
                case.roles[role].as_mut().unwrap().begin(&mut || false),
                Err(Error::Consumed)
            ));
            never_restored(&mut case);
        }
    }
    for role in 0..2 {
        let mut case = Fixture::at_stage(ROUNDS);
        assert!(case.advance(role, ROUNDS).unwrap());
        let peer = case.peer_copy(role, ROUNDS);
        assert!(matches!(
            case.roles[role]
                .as_mut()
                .unwrap()
                .advance(&peer, &mut || false),
            Err(Error::Consumed)
        ));
        never_restored(&mut case);
    }
}

#[test]
fn finish_is_closed_at_every_unready_prefix_and_for_either_single_ready_party() {
    for stage in 0..=ROUNDS {
        for role in 0..2 {
            let mut case = Fixture::at_stage(stage);
            if stage == ROUNDS {
                assert!(case.advance(role, stage).unwrap());
                let mut expected = [false; 2];
                expected[role] = true;
                assert_eq!(case.shared().control.try_lock().unwrap().ready, expected);
                assert!(case.shared().slots[role]
                    .state
                    .try_lock()
                    .unwrap()
                    .output
                    .is_some());
            }
            assert!(matches!(
                case.roles[role].take().unwrap().finish(),
                Err(Error::Consumed)
            ));
            never_restored(&mut case);
        }
    }
}

#[test]
fn role_coordinator_or_mask_drop_at_every_prefix_revokes_every_retained_frame() {
    for stage in 0..=ROUNDS {
        for dropped in 0..4 {
            let mut case = Fixture::at_stage(stage);
            match dropped {
                0 | 1 => drop(case.roles[dropped].take()),
                2 => drop(case.coordinator.take()),
                3 => drop(case.mask.take()),
                _ => unreachable!(),
            }
            never_restored(&mut case);
        }
    }
}

#[test]
fn either_party_drop_or_finish_with_one_ready_revokes_the_real_provisional_output() {
    for ready in 0..2 {
        for action in 0..6 {
            let mut case = Fixture::at_stage(ROUNDS);
            assert!(case.advance(ready, ROUNDS).unwrap());
            assert!(case.shared().slots[ready]
                .state
                .try_lock()
                .unwrap()
                .output
                .is_some());
            match action {
                0 | 1 => assert!(matches!(
                    case.roles[action].take().unwrap().finish(),
                    Err(Error::Consumed)
                )),
                2 | 3 => drop(case.roles[action - 2].take()),
                4 => drop(case.coordinator.take()),
                5 => drop(case.mask.take()),
                _ => unreachable!(),
            }
            never_restored(&mut case);
        }
    }
}

#[test]
fn either_final_output_drop_revokes_both_handoff_orders_and_all_old_frames() {
    for first in 0..2 {
        for dropped in 0..2 {
            let mut case = Fixture::at_stage(ROUNDS);
            assert!(case.advance(first, ROUNDS).unwrap());
            assert!(case.advance(1 - first, ROUNDS).unwrap());
            assert_eq!(case.shared().control.try_lock().unwrap().ready, [true; 2]);
            for role in [first, 1 - first] {
                case.outputs[role] = Some(case.roles[role].take().unwrap().finish().unwrap());
            }
            assert!(case
                .outputs
                .iter()
                .flatten()
                .all(|output| output.descriptor().is_ok()));
            drop(case.outputs[dropped].take());
            never_restored(&mut case);
        }
    }
}

#[test]
fn unwind_after_one_ready_erases_the_real_provisional_output_without_recovery() {
    for ready_role in 0..2 {
        for locked in [false, true] {
            let mut case = Fixture::at_stage(ROUNDS);
            assert!(case.advance(ready_role, ROUNDS).unwrap());
            assert!(case.shared().slots[ready_role]
                .state
                .try_lock()
                .unwrap()
                .output
                .is_some());
            let other = 1 - ready_role;
            let peer = case.peer_copy(other, ROUNDS);
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if locked {
                    let _: Result<(), Error> = case.shared().operation(|_| {
                        let _state = case.shared().slots[other].state.lock().unwrap();
                        panic!("native unwind after sibling Ready")
                    });
                } else {
                    let _ = case.roles[other]
                        .as_mut()
                        .unwrap()
                        .advance(&peer, &mut || panic!("callback unwind after sibling Ready"));
                }
            }))
            .is_err());
            never_restored(&mut case);
        }
    }
}

#[test]
fn each_real_runtime_factory_failure_burns_first_and_never_recharges_or_restores() {
    // Discover only actual existing Budget::allocation hook entries. This does
    // not claim injection into backend-private allocators or PyO3/RSS coverage.
    let count = {
        let mut case = Fixture::fresh();
        case.budget().fail_after.store(100000, Ordering::SeqCst);
        case.drive().unwrap();
        let count = 100000 - case.budget().fail_after.load(Ordering::SeqCst);
        case.budget().fail_after.store(-1, Ordering::SeqCst);
        assert!(count > 0);
        count
    };
    for site in 0..count {
        let mut case = Fixture::fresh();
        case.budget().fail_after.store(site, Ordering::SeqCst);
        assert!(
            matches!(case.drive(), Err(Error::Allocation)),
            "runtime factory {site}"
        );
        assert_eq!(case.budget().fail_after.load(Ordering::SeqCst), 0);
        // Resetting the hook neither accesses a coordinator nor heals poison.
        case.budget().fail_after.store(-1, Ordering::SeqCst);
        never_restored(&mut case);
        let Fixture {
            root,
            mask,
            coordinator,
            roles,
            frames,
            outputs,
        } = case;
        let root = root.unwrap();
        drop((mask, coordinator, roles, frames, outputs));
        assert_eq!(
            root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    }
}

#[test]
fn last_frame_alone_retains_original_source_counter_after_root_and_all_other_handles_drop() {
    let mut case = Fixture::at_stage(ROUNDS);
    let watch = case.root.as_ref().unwrap().try_clone().unwrap();
    let (source_weak, original_ptr, original_charge) = {
        let state = case.shared().slots[0].state.try_lock().unwrap();
        (
            Arc::downgrade(&state.tables),
            state.tables.input.payload().as_ptr(),
            state.tables.input.charged_bytes(),
        )
    };
    let last = case.frames[1].pop().unwrap();
    let Fixture {
        root,
        mask,
        coordinator,
        roles,
        frames,
        outputs,
    } = case;
    drop((root, mask, coordinator, roles, frames, outputs));
    assert_eq!(Arc::strong_count(&last.shared), 1);
    immediate_abort(&last.shared); // raw snapshots, not recovering APIs
    {
        let state = last.shared.slots[0].state.try_lock().unwrap();
        assert_eq!(state.tables.input.payload().as_ptr(), original_ptr);
        assert_eq!(state.tables.input.charged_bytes(), original_charge);
        assert!(std::ptr::eq(state.tables.input.budget(), watch.budget()));
        assert_eq!(state.tables.input.payload().as_slice(), &[0, 65535]);
        assert!(state
            .work
            .iter()
            .all(|words| words.iter().all(|&word| word == 0)));
        assert!(state.lease.bytes >= 5 * 8 * 2);
    }
    assert!(source_weak.upgrade().is_some()); // temporary immediately retires
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    assert!(watch.budget().used_bytes() >= baseline + original_charge);
    assert!(last.bytes().is_err());
    drop(last);
    assert!(source_weak.upgrade().is_none());
    assert_eq!(watch.budget().used_bytes(), baseline);
    // No access through original_ptr after destruction: there is no existing
    // Vec-erasure observer hook. This asserts live-state zeroization, retention
    // and refunds; it does NOT invent a safe post-free source-erasure oracle.
}
