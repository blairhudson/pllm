//! Actual binary Invocation::finish failures, not a substitute guard fixture.
use super::*;
use crate::replicated_decoder::runtime::body::integration::restoration_tests as hooks;
use std::{cell::Cell, panic::AssertUnwindSafe, rc::Rc};

fn completed_binary(replay_unwind: bool) -> (Invocation, usize, usize) {
    let (mut program, end) = fixture(false);
    if replay_unwind {
        // Deliberately corrupt only a replay ledger entry in this TEST fixture.
        // Native body execution does not read it; the real restore ledger does.
        // No production Program, execution identity or proof issuer is changed.
        Arc::get_mut(&mut program).unwrap().retire[2].push(usize::MAX);
    }
    let compiled = Compiled::new(program.clone(), WeightMode::Public, 2, end).unwrap();
    let exact = compiled.admitted_bytes();
    let clear = BTreeMap::from([("x".into(), vec![0; 3]), ("y".into(), vec![ONE; 3])]);
    let mut bodies =
        workers(program.clone(), &clear).map(|worker| compiled.clone().begin(worker).unwrap());
    let mut queue = Vec::new();
    for _ in 0..1000 {
        if queue.is_empty() && bodies.iter().all(Invocation::complete) {
            break;
        }
        if queue.is_empty() {
            for body in &mut bodies {
                let from = body.party();
                queue.extend(
                    body.advance()
                        .unwrap()
                        .into_iter()
                        .map(|(to, raw)| (from, to, raw)),
                );
            }
        } else {
            for (from, to, raw) in std::mem::take(&mut queue) {
                queue.extend(
                    bodies[to as usize]
                        .receive(raw.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, raw)| (to, next, raw)),
                );
            }
        }
    }
    assert!(queue.is_empty() && bodies.iter().all(Invocation::complete));
    let [a, b, c] = bodies;
    drop((b, c, compiled, program));
    (a, exact, end)
}

#[test]
fn real_binary_finish_owned_cleanup_and_pre_extraction_exact_one_short() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            // Exact success, one-short, partial-reader error/unwind, original
            // ledger replay unwind, and output-erasure unwind respectively.
            for case in 0..6 {
                let (body, exact, end) = completed_binary(case == 4);
                let owners = hooks::owners(&body);
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
                    // Reentrant observation takes the TLS RefCell anew: no
                    // reader/extraction hook may retain a borrow across drop.
                    record.set(Some((observed.counts_after_erasure(), hooks::counts())));
                    if case == 5 {
                        panic!("injected caller output-erasure unwind");
                    }
                });
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| body.finish()));
                match case {
                    0 => {
                        let worker = result.unwrap().unwrap();
                        assert!(erased.get()); // Before returning the Worker.
                        let (step, started, offset, empty, idle, _) = restored_state(&worker);
                        assert_eq!((step, offset), (end, 0));
                        assert!(started && empty && idle);
                        // The returned ORIGINAL Worker, not the erased Values
                        // or released Compiled/Plan, retains the Program now.
                        assert_eq!(owners.counts_after_erasure(), [0, 0, 1]);
                        assert!(!worker.retired());
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
                    if matches!(case, 2 | 3) {
                        assert_eq!(reads, 2); // Failure AFTER a successful lane.
                    } else {
                        assert!(reads >= 3);
                    }
                }
                owners.assert_released();
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
