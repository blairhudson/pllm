//! Actual consuming tensor restoration, including authenticated source custody.
use super::*;
use crate::replicated_decoder::body_tensor_integration::restoration_tests as hooks;
use std::{cell::Cell, panic::AssertUnwindSafe, rc::Rc};

#[path = "body_restore_owner_observation_tests.rs"]
mod owner_observation;

fn completed_tensor(replay_unwind: bool) -> CoordinatorDrained<Invocation> {
    let (mut program, prepared, input, output) =
        crate::replicated_decoder::program::prepared_rescale18_worker_fixture(129, domain())
            .unwrap();
    if replay_unwind {
        // Test-only corruption reaches the ORIGINAL ledger after all exports
        // copy. Native body execution and compiler binding are unchanged.
        Arc::get_mut(&mut program).unwrap().retire[output].push(usize::MAX);
    }
    let compiled =
        Compiled::prepare_candidate(program.clone(), WeightMode::Public, input, output, prepared)
            .unwrap();
    let mut roles = fresh_workers(program.clone());
    reach(&mut roles, output, Some(&compiled));
    let mut bodies = begin_after_reach(&compiled, roles);
    let (_, held) = drive(&mut bodies);
    let [a, b, c] = TestCoordinator {
        roles: bodies,
        queue: Vec::new(),
    }
    .certify_end()
    .unwrap();
    drop((b, c, compiled, program, held));
    a
}

#[test]
fn real_tensor_restore_owned_cleanup_and_pre_extraction_exact_one_short() {
    run_large_stack(|| {
        for case in 0..6 {
            let drained = completed_tensor(case == 4);
            let end = drained.owner.compiled.binding.end;
            let exact = drained.owner.compiled.phases.required_bytes().unwrap();
            let owners = hooks::owners(&drained.owner);
            let observed = owners.clone();
            let erased = Rc::new(Cell::new(false));
            let did_erase = erased.clone();
            let observation = Rc::new(Cell::new(None));
            let record = observation.clone();
            let fault = match case {
                2 => hooks::Fault::ReadError,
                3 => hooks::Fault::ReadPanic,
                _ => hooks::Fault::None,
            };
            let _session = hooks::install(exact - usize::from(case == 1), fault, move || {
                did_erase.set(true);
                record.set(Some((observed.counts_after_erasure(), hooks::counts())));
                if case == 5 {
                    panic!("injected caller output-erasure unwind");
                }
            });
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| drained.restore()));
            match case {
                0 => {
                    let worker = result.unwrap().unwrap();
                    assert!(erased.get());
                    let (step, started, offset, empty, idle, _) =
                        body::integration::restored_state(&worker);
                    assert_eq!((step, offset), (end, 0));
                    assert!(started && empty && idle);
                    drop(worker);
                }
                1 => assert!(matches!(result, Ok(Err(Error::Bounds)))),
                2 => assert!(matches!(result, Ok(Err(Error::Binding)))),
                _ => assert!(result.is_err()),
            }
            let (extractions, reads) = hooks::counts();
            if case == 1 {
                assert_eq!((extractions, reads), (0, 0));
                assert!(!erased.get());
                assert!(observation.get().is_none());
            } else {
                assert_eq!(extractions, 1);
                assert!(erased.get());
                assert_eq!(observation.get(), Some(([1, 1, 2], (extractions, reads))));
                assert_eq!(reads, if matches!(case, 2 | 3) { 2 } else { 129 });
            }
            owners.assert_released();
        }
    });
}

#[test]
fn tensor_phase_ceiling_adds_owned_restoration_without_discounting_other_peaks() {
    run_large_stack(|| {
        let drained = completed_tensor(false);
        let compiled = &drained.owner.compiled;
        let phases = &compiled.phases;
        let payload = compiled.binding.lanes * 16;
        let old_restoration = phases.execution
            + payload
            + size_of::<Result<(crate::replicated_decoder::multiply::PartyRuntime, Values), Error>>(
            )
            + size_of::<&Values>();
        let increment = owned_restoration_controls().unwrap();
        assert_eq!(phases.restoration, old_restoration + increment);
        let old_ceiling = phases.baseline
            + phases
                .construction
                .max(phases.copies)
                .max(phases.execution)
                .max(old_restoration);
        let exact = phases.required_bytes().unwrap();
        assert_eq!(exact, old_ceiling.max(phases.baseline + phases.restoration));
        assert_eq!(phases.admit_with_limit(exact), Ok(()));
        assert_eq!(phases.admit_with_limit(exact - 1), Err(Error::Bounds));
        // Construction remains a separate additive peak even if it dominates.
        let dominated = Phases {
            baseline: 137,
            construction: 9999,
            copies: 1,
            execution: 2,
            restoration: 3,
        };
        assert_eq!(dominated.required_bytes().unwrap(), 137 + 9999);
        let overflow = Phases {
            baseline: usize::MAX,
            ..dominated
        };
        assert_eq!(overflow.required_bytes(), Err(Error::Bounds));
        drop(drained);
    });
}

#[test]
fn real_source_tensor_restore_keeps_original_dispatch_until_values_erase() {
    run_large_stack(|| {
        for fault in [hooks::Fault::ReadError, hooks::Fault::ReadPanic] {
            let (manifest, compiled, public, mut roles) = source_backed_candidate();
            let registry = SourceRegistry::from_manifest(&manifest).unwrap();
            reach(&mut roles, compiled.binding.output, Some(&compiled));
            let [a, b, c] = TestCoordinator {
                roles,
                queue: Vec::new(),
            }
            .certify_start(&compiled)
            .unwrap();
            let mut bodies = [
                compiled.clone().begin_with_registry(a, &registry).unwrap(),
                compiled.clone().begin_with_registry(b, &registry).unwrap(),
                compiled.clone().begin_with_registry(c, &registry).unwrap(),
            ];
            #[allow(clippy::drop_non_drop)] // End the registry borrow before original owners.
            drop(registry);
            drop(manifest);
            let (_, held) = drive(&mut bodies);
            let [a, b, c] = TestCoordinator {
                roles: bodies,
                queue: Vec::new(),
            }
            .certify_end()
            .unwrap();
            let source = Arc::downgrade(&public);
            let observed_source = source.clone();
            let owners = hooks::owners(&a.owner);
            let observed = owners.clone();
            let exact = compiled.phases.required_bytes().unwrap();
            drop((b, c, compiled, public, held));
            let erased = Rc::new(Cell::new(false));
            let did_erase = erased.clone();
            let observation = Rc::new(Cell::new(None));
            let record = observation.clone();
            let _session = hooks::install(exact, fault, move || {
                // Executor has finished; ONLY the original Compiled dispatch
                // holds this Arc. Worker independently owns its by-value source.
                record.set(Some((
                    observed.counts_after_erasure(),
                    observed_source.strong_count(),
                    hooks::counts(),
                )));
                did_erase.set(true);
            });
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| a.restore()));
            match fault {
                hooks::Fault::ReadError => assert!(matches!(result, Ok(Err(Error::Binding)))),
                hooks::Fault::ReadPanic => assert!(result.is_err()),
                hooks::Fault::None => unreachable!(),
            }
            assert!(erased.get());
            assert_eq!(hooks::counts(), (1, 2));
            assert_eq!(observation.get(), Some(([1, 1, 2], 1, (1, 2))));
            owners.assert_released();
            assert_eq!(source.strong_count(), 0);
            assert!(matches!(
                worker_parity_gate(),
                Err(PreparationError::UnqualifiedOriginalWorkerParity)
            ));
        }
    });
}
