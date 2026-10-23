//! Independent frozen pre-extraction D27 layouts and lifecycle expectations.
//! These are layout-only mirrors, not a second evaluator/coordinator.
use super::super::tests::{dealer, spec};
use super::*;
use std::alloc::Layout;
use std::sync::atomic::AtomicUsize;

struct OldState<B: AllocationContext> {
    _party: LinearPartyFor<B>,
    _output: Option<CommonMaskedFor<B, Ring>>,
}
struct OldSlot<B: AllocationContext> {
    _state: Mutex<OldState<B>>,
    _lease: LeaseFor<B>,
}
struct OldControl {
    _aborted: bool,
    _ready: [bool; 2],
}
struct OldCoordination<B: AllocationContext> {
    _control: Mutex<OldControl>,
    _slots: [Arc<OldSlot<B>>; 2],
    _original_run: Option<LeaseFor<B>>,
    _additional: LeaseFor<B>,
}
struct OldCoordinator<B: AllocationContext> {
    _shared: Arc<OldCoordination<B>>,
}
struct OldRole<B: AllocationContext> {
    _slot: Arc<OldSlot<B>>,
    _shared: Arc<OldCoordination<B>>,
    _role: u8,
    _handed_off: bool,
}
struct OldFrame<B: AllocationContext> {
    _native: LinearActivationFrameFor<B>,
    _shared: Arc<OldCoordination<B>>,
}
#[allow(dead_code, clippy::large_enum_variant)] // Frozen enum layout only.
enum OldStep<B: AllocationContext> {
    Frame(OldFrame<B>),
    Ready,
}
struct OldOutput<B: AllocationContext> {
    _slot: Arc<OldSlot<B>>,
    _shared: Arc<OldCoordination<B>>,
    _role: u8,
}
struct OldOperation<'a, B: AllocationContext> {
    _coordination: &'a OldCoordination<B>,
    _control: MutexGuard<'a, OldControl>,
    _armed: bool,
}
struct OldPoll<'a, B: AllocationContext> {
    _coordination: &'a OldCoordination<B>,
    _armed: bool,
}
type OldSplit<B> = (OldCoordinator<B>, [OldRole<B>; 2]);

fn old_arc<T>() -> usize {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .unwrap()
        .0
        .pad_to_align()
        .size()
}

// Independent original formula: do not replace mirror terms with current aliases.
fn original_quote<B: AllocationContext>() -> LinearActivationRoleSplitResources {
    let role_backing_bytes = old_arc::<OldSlot<B>>();
    let coordination_backing_bytes = old_arc::<OldCoordination<B>>();
    let wrapper_population_bytes = metadata_sum(&[
        size_of::<OldCoordinator<B>>(),
        2 * size_of::<OldRole<B>>(),
        2 * EXCHANGES * size_of::<OldFrame<B>>(),
        2 * EXCHANGES * size_of::<OldStep<B>>(),
        2 * size_of::<OldOutput<B>>(),
    ])
    .unwrap();
    let transition_bytes = metadata_sum(&[
        size_of::<LinearActivationRoleSplitResources>(),
        size_of::<Result<LinearActivationRoleSplitResources, Error>>(),
        size_of::<LinearActivationRunFor<B>>(),
        2 * size_of::<OldSlot<B>>(),
        size_of::<OldCoordination<B>>(),
        size_of::<OldControl>(),
        size_of::<[Arc<OldSlot<B>>; 2]>(),
        size_of::<[OldRole<B>; 2]>(),
        size_of::<OldSplit<B>>(),
        size_of::<Result<OldSplit<B>, Error>>(),
        size_of::<OldOperation<'_, B>>(),
        size_of::<OldPoll<'_, B>>(),
        size_of::<MutexGuard<'_, OldState<B>>>(),
        size_of::<PartyStepFor<B>>(),
        size_of::<Result<PartyStepFor<B>, Error>>(),
        size_of::<Result<LinearActivationFrameFor<B>, Error>>(),
        size_of::<Result<OldFrame<B>, Error>>(),
        size_of::<Result<OldStep<B>, Error>>(),
        size_of::<Result<OldOutput<B>, Error>>(),
        size_of::<Result<EdgeDescriptor, Error>>(),
        size_of::<Result<&'static [u8], Error>>(),
        size_of::<Result<(), Error>>(),
        4 * size_of::<Arc<OldCoordination<B>>>(),
        3 * size_of::<LeaseFor<B>>(),
        size_of::<Option<LeaseFor<B>>>(),
        2 * B::TRANSIENT_LINK_BYTES,
    ])
    .unwrap();
    let reservation_bytes = metadata_sum(&[
        role_backing_bytes,
        role_backing_bytes,
        coordination_backing_bytes,
        wrapper_population_bytes,
        transition_bytes,
    ])
    .unwrap();
    LinearActivationRoleSplitResources {
        role_backing_bytes,
        coordination_backing_bytes,
        wrapper_population_bytes,
        transition_bytes,
        reservation_bytes,
    }
}

#[test]
fn original_d27_borrowed_and_owned_layout_quotes_remain_identical() {
    assert_eq!(
        role_split_resources_for::<&Budget>().unwrap(),
        original_quote::<&Budget>()
    );
    assert_eq!(
        role_split_resources_for::<OwnedBudgetContext>().unwrap(),
        original_quote::<OwnedBudgetContext>()
    );
}

#[derive(Default)]
struct OriginalLifecycle {
    ready: [bool; 2],
    aborted: bool,
}
impl OriginalLifecycle {
    fn ready(&mut self, role: usize) {
        assert!(!self.aborted);
        self.ready[role] = true;
    }
    fn finish(&mut self) -> bool {
        if self.aborted || self.ready != [true; 2] {
            self.aborted = true;
            false
        } else {
            true
        }
    }
    fn abort(&mut self) {
        self.aborted = true;
    }
    fn compare(&self, shared: &Coordination<'_>) {
        let actual = shared.control();
        assert_eq!(actual.aborted, self.aborted);
        assert_eq!(actual.ready, self.ready);
        if self.aborted {
            for slot in &shared.slots {
                let state = slot.state.lock().unwrap_or_else(|p| p.into_inner());
                assert!(state.output.is_none());
                assert!(state.party.phase == Phase::Dead);
                assert_eq!(state.party.ledger.next, USES as u8);
                assert!(state.party.roots.is_none());
            }
        }
    }
}

fn before_final<'b>(
    roles: &mut [LinearActivationRole<'b>; 2],
) -> [LinearActivationRoleFrame<'b>; 2] {
    let mut frames = [
        roles[0].begin(&mut || false).unwrap(),
        roles[1].begin(&mut || false).unwrap(),
    ];
    for _ in 0..EXCHANGES - 1 {
        let LinearActivationRoleStep::Frame(left) = roles[0]
            .advance(frames[1].as_bytes().unwrap(), &mut || false)
            .unwrap()
        else {
            panic!("early Ready")
        };
        let LinearActivationRoleStep::Frame(right) = roles[1]
            .advance(frames[0].as_bytes().unwrap(), &mut || false)
            .unwrap()
        else {
            panic!("early Ready")
        };
        frames = [left, right];
    }
    frames
}

#[test]
fn original_d27_ready_finish_and_root_drop_lifecycle_matches_independent_model() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    for early in [true, false] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut dealer = dealer(&budget);
        let (_mask, run) = dealer
            .linear_activation_from_values_with(
                spec(1, 2, 1, ActivationKind::Silu),
                &[-1, -1],
                &[1, 4095],
                &mut |bytes| {
                    bytes.fill(173);
                    Ok(())
                },
            )
            .unwrap();
        let (coordinator, mut roles) = run.into_roles().unwrap();
        let frames = before_final(&mut roles);
        let mut original = OriginalLifecycle::default();
        assert!(matches!(
            roles[0].advance(frames[1].as_bytes().unwrap(), &mut || false),
            Ok(LinearActivationRoleStep::Ready)
        ));
        original.ready(0);
        original.compare(&coordinator.shared);
        if early {
            let [left, right] = roles;
            assert_eq!(left.finish().is_ok(), original.finish());
            original.compare(&coordinator.shared);
            assert!(right.finish().is_err());
        } else {
            assert!(matches!(
                roles[1].advance(frames[0].as_bytes().unwrap(), &mut || false),
                Ok(LinearActivationRoleStep::Ready)
            ));
            original.ready(1);
            let [left, right] = roles;
            let outputs = [left.finish().unwrap(), right.finish().unwrap()];
            assert!(original.finish());
            original.compare(&coordinator.shared);
            assert!(outputs.iter().all(|output| output.descriptor().is_ok()));
            let shared = Arc::clone(&coordinator.shared);
            drop(coordinator);
            original.abort();
            original.compare(&shared);
            assert!(outputs.iter().all(|output| output.descriptor().is_err()));
        }
        assert!(frames.iter().all(|frame| frame.as_bytes().is_err()));
    }
}

#[test]
fn original_d27_callback_reentry_and_unwind_model_preserves_unlocked_callbacks() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    for panic in [false, true] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut dealer = dealer(&budget);
        let (_mask, run) = dealer
            .linear_activation_from_values_with(
                spec(1, 2, 1, ActivationKind::Gelu),
                &[-1, -1],
                &[1, 4095],
                &mut |bytes| {
                    bytes.fill(173);
                    Ok(())
                },
            )
            .unwrap();
        let (mut coordinator, [mut left, right]) = run.into_roles().unwrap();
        let mut original = OriginalLifecycle::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            left.begin(&mut || {
                assert!(coordinator.shared.control.try_lock().is_ok());
                assert!(coordinator
                    .shared
                    .slots
                    .iter()
                    .all(|slot| slot.state.try_lock().is_ok()));
                if panic {
                    panic!("old D27 callback unwind oracle")
                }
                coordinator.cancel();
                false
            })
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        original.abort();
        original.compare(&coordinator.shared);
        assert!(!coordinator.shared.control.is_poisoned());
        assert!(right.finish().is_err());
    }
}

#[test]
fn borrowed_and_owned_d27_handles_remain_non_send_and_non_sync() {
    macro_rules! not_trait {
        ($ty:ty,$bound:path) => {{
            trait Ambiguous<A> {
                fn check() {}
            }
            impl<T> Ambiguous<()> for T {}
            struct Implemented;
            impl<T: $bound> Ambiguous<Implemented> for T {}
            let _ = <$ty as Ambiguous<_>>::check;
        }};
    }
    macro_rules! local {
        ($ty:ty) => {{
            not_trait!($ty, Send);
            not_trait!($ty, Sync);
        }};
    }
    local!(LinearActivationRoleCoordinator<'static>);
    local!(LinearActivationRole<'static>);
    local!(LinearActivationRoleFrame<'static>);
    local!(LinearActivationRoleOutput<'static>);
    local!(LinearActivationRoleCoordinatorFor<OwnedBudgetContext>);
    local!(LinearActivationRoleFor<OwnedBudgetContext>);
    local!(LinearActivationRoleFrameFor<OwnedBudgetContext>);
    local!(LinearActivationRoleOutputFor<OwnedBudgetContext>);
}
