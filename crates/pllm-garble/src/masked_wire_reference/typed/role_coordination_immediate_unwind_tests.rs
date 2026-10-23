//! Mount as a child of role_coordination_nexp_tests, using its actual terminal
//! fixture. No full nExp/source-table/privacy qualification is asserted here.
use super::*;
use std::sync::TryLockError;

#[test]
fn nexp_terminal_locked_panic_burns_before_any_poison_recovery_access() {
    let (send, receive) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        for faulting in 0..2 {
            let budget = Budget::new(64 << 20).unwrap();
            let baseline = budget.used_bytes();
            let (root, mask) = fixture(&&budget, None).unwrap();
            let frames = first_round(&root);
            let sibling = 1 - faulting;
            assert!(
                advance(&root, sibling, frames[3 - sibling].bytes().unwrap(), false)
                    .unwrap()
                    .is_none()
            );
            let peer = frames[3 - faulting].bytes().unwrap();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = advance(&root, faulting, peer, true);
            }));
            assert!(
                result.is_err(),
                "native terminal panic injection did not fire"
            );

            // Do not call control(), assert_burned(), frame.bytes(), output.live()
            // or any shared operation first. Poison recovery itself aborts both
            // and would hide a missing/broken OperationFor::drop retirement.
            let control = match root.shared.control.try_lock() {
                Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                Err(TryLockError::WouldBlock) => panic!("unwind retained coordination mutex"),
                Ok(_) => panic!("locked-work unwind did not poison coordination mutex"),
            };
            assert!(
                control.aborted,
                "retirement was deferred until later poison recovery"
            );
            assert_eq!(control.ready, [true, true]);
            drop(control);
            for (role, slot) in root.shared.slots.iter().enumerate() {
                assert_eq!(slot.state.is_poisoned(), role == faulting);
                let mut state = match slot.state.try_lock() {
                    Ok(state) => state,
                    Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                    Err(TryLockError::WouldBlock) => panic!("unwind retained native-state mutex"),
                };
                assert!(
                    state.input.is_none(),
                    "original product input survived unwind"
                );
                assert!(
                    state.output.is_none(),
                    "already-ready output survived unwind"
                );
                assert!(matches!(state.party.begin(&[0]), Err(Error::Consumed)));
            }

            // Only now exercise public recovery/revocation and retained leases.
            assert!(frames.iter().all(|frame| frame.bytes().is_err()));
            drop((root, frames, mask));
            assert_eq!(budget.used_bytes(), baseline);
        }
        send.send(()).unwrap();
    });
    receive
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("immediate terminal unwind retirement deadlocked or panicked");
    worker.join().unwrap();
}
