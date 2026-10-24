//! Production constructor regression; original expanded compiler is oracle ONLY.
use super::super::schedule;
use super::*;
use crate::replicated_decoder::{
    puma::{Context, Party},
    PeerBytes,
};

#[path = "rotary_creation_tests.rs"]
mod rotary_creation_tests;
#[path = "compact_request_storage_tests.rs"]
mod compact_request_storage_tests;
#[path = "source_bias_tests.rs"]
mod source_bias_tests;

fn complete<'s>(case: &mut Case<'s>) -> LogicalComplete<'s> {
    let pending = case.run();
    case.builder.final_acceptance_owned(pending).unwrap()
}

fn finalize(case: &mut Case<'_>, done: LogicalComplete<'_>) -> Arc<Program> {
    let mut guard = case.root.charge_external(0).unwrap();
    let quoted = schedule::quote(&case.builder, &done, &guard).unwrap();
    guard.resize_payload(quoted.phase_payload_bytes).unwrap();
    assert_eq!(
        schedule::quote(&case.builder, &done, &guard).unwrap(),
        quoted
    );
    schedule::finish(
        std::mem::take(&mut case.builder),
        done,
        guard,
        quoted.peak_bytes,
    )
    .unwrap()
}

fn original(fixture: &Fixture) -> Arc<Program> {
    Arc::new(
        Program::compile_with_norms(
            &fixture.manifest.config,
            4,
            2,
            &fixture.manifest.tensors,
            WeightMode::Public,
            fixture.manifest.execution,
            &fixture.manifest.norms,
        )
        .unwrap(),
    )
}

fn fresh(program: Arc<Program>, public: &Arc<PublicSource>, sealed: bool) -> [Worker; 3] {
    let session = [92; 32];
    let context = Context::new(session, program.digest(), 0).unwrap();
    let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
        context,
        program.work_budget().limits().unwrap(),
    )
    .unwrap();
    let mut workers = runtimes.map(|runtime| {
        if sealed {
            program.source_request_worker(runtime, session).unwrap()
        } else {
            Worker::new_with_shared_source(
                Arc::clone(&program),
                runtime,
                session,
                Arc::clone(public),
            )
            .unwrap()
        }
    });
    let tokens = TOKENS.map(u64::from);
    let a = [0x13u64, 0x47, 0x39, 0x81];
    let b = [0x71u64, 0x53, 0x92, 0x31];
    let c: [u64; 4] = std::array::from_fn(|i| tokens[i].wrapping_sub(a[i]).wrapping_sub(b[i]));
    let shares = [a, b, c];
    for (i, worker) in workers.iter_mut().enumerate() {
        worker
            .import("input.tokens", &shares[i], &shares[(i + 1) % 3])
            .unwrap();
    }
    workers
}

type Routed = (Party, Party, PeerBytes);
fn same_state(a: &Worker, b: &Worker) {
    match (a.test_boundary_state(), b.test_boundary_state()) {
        (Ok((at, ac, av)), Ok((bt, bc, bv))) => {
            assert_eq!((at, ac), (bt, bc));
            assert_eq!(av.collect::<Vec<_>>(), bv.collect::<Vec<_>>());
        }
        (Err(Error::Phase), Err(Error::Phase)) => (),
        _ => panic!("original/candidate local boundary mismatch"),
    }
}
fn frames(
    from: Party,
    original: Vec<(Party, PeerBytes)>,
    candidate: Vec<(Party, PeerBytes)>,
    qa: &mut Vec<Routed>,
    qb: &mut Vec<Routed>,
) -> usize {
    assert_eq!(original.len(), candidate.len());
    let mut bytes = 0;
    for ((to, a), (other, b)) in original.into_iter().zip(candidate) {
        assert_eq!(to, other);
        assert_eq!(a.as_bytes(), b.as_bytes());
        bytes += a.as_bytes().len();
        qa.push((from, to, a));
        qb.push((from, to, b));
    }
    bytes
}

/// Test routing only: both sides run the ordinary Worker directly. No Qualified,
/// frozen loan, alternate executor or callback assertion supplies authority.
fn paired(mut a: [Worker; 3], mut b: [Worker; 3], valid: (usize, usize)) {
    let (mut qa, mut qb) = (Vec::new(), Vec::new());
    let mut wire = 0;
    for round in 0..100_000 {
        if a.iter().all(Worker::complete) {
            assert!(b.iter().all(Worker::complete));
            assert!(qa.is_empty() && qb.is_empty());
            assert!(wire > 0);
            let mut accepted = 0u64;
            for role in &a {
                let (owner, lane) = valid;
                let (_, _, mut values) = role.test_boundary_state().unwrap();
                let (first, _) = values.nth(owner).unwrap().unwrap();
                accepted = accepted.wrapping_add(first[lane]);
            }
            assert_eq!(accepted, 1, "genuine original checked epilogue must accept");
            for (a, b) in a.into_iter().zip(b) {
                same_state(&a, &b);
                let x = a.finish_frame().unwrap();
                let y = b.finish_frame().unwrap();
                // Includes both replicated components and original frame header.
                assert_eq!(x.as_slice(), y.as_slice());
            }
            return;
        }
        assert!(round + 1 < 100_000, "bounded test routing stalled");
        if qa.is_empty() {
            for role in 0..3 {
                assert_eq!(a[role].complete(), b[role].complete());
                if !a[role].complete() {
                    let from = a[role].party();
                    let x = a[role].advance().unwrap();
                    let y = b[role].advance().unwrap();
                    wire += frames(from, x, y, &mut qa, &mut qb);
                }
            }
        } else {
            assert_eq!(qa.len(), qb.len());
            for ((from, to, x), (other_from, other_to, y)) in std::mem::take(&mut qa)
                .into_iter()
                .zip(std::mem::take(&mut qb))
            {
                assert_eq!((from, to), (other_from, other_to));
                let x = a[to as usize].receive(x.as_bytes(), from).unwrap();
                let y = b[to as usize].receive(y.as_bytes(), from).unwrap();
                wire += frames(to, x, y, &mut qa, &mut qb);
            }
        }
        for role in 0..3 {
            same_state(&a[role], &b[role]);
            assert_eq!(a[role].loop_test_state(), b[role].loop_test_state());
        }
    }
    unreachable!()
}

#[test]
fn genuine_source_four_plus_two_constructs_original_program_and_full_wire_without_oracle_permit() {
    with_case(|mut case| {
        let done = complete(&mut case);
        assert_eq!(done.input(0).unwrap().rows(), 4);
        assert_eq!(done.input(1).unwrap().rows(), 5);
        let root_source = Arc::downgrade(&done.pending.history.epochs[0]);
        let native: Vec<_> = case
            .builder
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| match &node.code {
                Code::Tensor { program, .. } | Code::CheckedRange { program, .. } => {
                    Some((id, Arc::clone(program)))
                }
                _ => None,
            })
            .collect();
        let expected = original(case.fixture);
        let loops: Vec<_> = case
            .builder
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| {
                if let Code::TensorLoop { program, tail, .. } = &node.code {
                    Some((id, Arc::clone(program), tail.as_ref().map(Arc::clone)))
                } else {
                    None
                }
            })
            .collect();
        let program = finalize(&mut case, done);
        assert_eq!(program.digest(), expected.digest());
        assert_eq!(program.work_budget(), expected.work_budget());
        assert_eq!(program.outputs, expected.outputs);
        assert_eq!(program.retire, expected.retire);
        assert_eq!(program.source_binding, expected.source_binding);
        assert_eq!(program.payload_bytes(), expected.payload_bytes());
        for (id, owner) in native {
            assert!(matches!(&program.nodes[id].code,
                Code::Tensor {program,..} | Code::CheckedRange {program,..} if Arc::ptr_eq(program,&owner)));
        }
        for (id, full, tail) in loops {
            let Code::TensorLoop {
                program: actual,
                tail: actual_tail,
                ..
            } = &program.nodes[id].code
            else {
                panic!("original loop replaced")
            };
            assert!(Arc::ptr_eq(actual, &full));
            match (tail, actual_tail) {
                (None, None) => (),
                (Some(a), Some(b)) => assert!(Arc::ptr_eq(&a, b)),
                _ => panic!("original tail replaced"),
            }
        }
        assert!(root_source.upgrade().is_some());
        paired(
            fresh(expected, &case.fixture.public, false),
            fresh(Arc::clone(&program), &case.fixture.public, true),
            *program.outputs.last().unwrap(),
        );
        drop(program);
        assert!(root_source.upgrade().is_none());
        assert_eq!(
            case.root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    });
}

#[test]
fn consuming_production_entry_runs_shared_driver_and_epilogue_without_expanded_program_argument() {
    with_case(|mut case| {
        let permit = case.permit.take().unwrap();
        let controller = case
            .root
            .charge_external(permit.lowering_control_bytes(limits()).unwrap())
            .unwrap();
        let schedule_guard = case.root.charge_external(0).unwrap();
        let actual = schedule::compile(
            std::mem::take(&mut case.builder),
            permit,
            &case.inputs,
            controller,
            limits(),
            schedule_guard,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
        let expected = original(case.fixture);
        assert_eq!(actual.digest(), expected.digest());
        assert_eq!(actual.work_budget(), expected.work_budget());
        assert_eq!(actual.retire, expected.retire);
        assert_eq!(actual.outputs.len(), 3); // two masked IDs + original sticky bit
        drop(actual);
        assert_eq!(
            case.root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    });
}

#[test]
fn finalizer_one_short_guard_or_cap_burns_custody_before_schedule_allocation() {
    for short_guard in [true, false] {
        with_case(|mut case| {
            let done = complete(&mut case);
            let origin = Arc::clone(&done.pending.history.origin);
            let mut guard = case.root.charge_external(0).unwrap();
            let q = schedule::quote(&case.builder, &done, &guard).unwrap();
            guard
                .resize_payload(q.phase_payload_bytes - usize::from(short_guard))
                .unwrap();
            let cap = q.peak_bytes - usize::from(!short_guard);
            assert!(matches!(
                schedule::finish(std::mem::take(&mut case.builder), done, guard, cap),
                Err(Error::Bounds)
            ));
            assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
            drop(origin);
            assert_eq!(
                case.root.budget().used_bytes(),
                OwnedBudget::resources().unwrap().initial_reserved_bytes
            );
        });
    }
}

#[test]
fn foreign_counter_and_post_acceptance_output_substitution_never_issue_program() {
    for foreign in [false, true] {
        with_case(|mut case| {
            let done = complete(&mut case);
            let origin = Arc::clone(&done.pending.history.origin);
            if foreign {
                let other = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
                let guard = other.charge_external(0).unwrap();
                assert!(matches!(
                    schedule::finish(
                        std::mem::take(&mut case.builder),
                        done,
                        guard,
                        MAX_PROGRAM_BYTES
                    ),
                    Err(Error::Binding)
                ));
                assert_eq!(
                    other.budget().used_bytes(),
                    OwnedBudget::resources().unwrap().initial_reserved_bytes
                );
            } else {
                let masked = done.outputs[0].0;
                case.builder.nodes[masked].bounds = Interval::new(0, 0).unwrap();
                let guard = case.root.charge_external(0).unwrap();
                assert!(matches!(
                    schedule::finish(
                        std::mem::take(&mut case.builder),
                        done,
                        guard,
                        MAX_PROGRAM_BYTES
                    ),
                    Err(Error::Binding)
                ));
            }
            assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
            drop(origin);
            assert_eq!(
                case.root.budget().used_bytes(),
                OwnedBudget::resources().unwrap().initial_reserved_bytes
            );
        });
    }
    with_case(|mut case| {
        let done = complete(&mut case);
        let program = finalize(&mut case, done);
        let foreign = Fixture::new();
        assert!(!Arc::ptr_eq(&foreign.public, &case.fixture.public));
        // Genuine separately authenticated bytes pass the old binding check;
        // they still cannot replace this request's retained creator allocation.
        foreign.public.check(&program).unwrap();
        let session = [92; 32];
        let context = Context::new(session, program.digest(), 0).unwrap();
        let [runtime, sibling_one, sibling_two] =
            puma::prss::insecure_test_support::fresh_compound_parties(
                context,
                program.work_budget().limits().unwrap(),
            )
            .unwrap();
        assert!(runtime.status() == multiply::Status::Ready);
        assert!(runtime.matches_context(context));
        assert!(matches!(
            Worker::new_with_shared_source(
                Arc::clone(&program),
                runtime,
                session,
                Arc::clone(&foreign.public),
            ),
            Err(Error::Binding)
        ));
        drop((sibling_one, sibling_two));
        drop(program);
        assert_eq!(
            case.root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    });
}

#[test]
fn ordinary_worker_cancel_or_error_retains_source_until_last_original_program_owner_drops() {
    for malformed in [false, true] {
        with_case(|mut case| {
            let done = complete(&mut case);
            let original = Arc::downgrade(&done.pending.history.epochs[0]);
            let program = finalize(&mut case, done);
            let mut roles = fresh(Arc::clone(&program), &case.fixture.public, true);
            drop(program);
            let mut held = Vec::new();
            for role in &mut roles {
                held.extend(role.advance().unwrap());
            }
            assert!(!held.is_empty());
            for role in &mut roles {
                if malformed {
                    assert!(role.receive(b"bad", role.party().next()).is_err());
                } else {
                    role.cancel();
                }
                assert!(matches!(role.advance(), Err(Error::Retired)));
            }
            assert!(original.upgrade().is_some());
            let snapshot: Vec<_> = held
                .iter()
                .map(|(_, frame)| frame.as_bytes().to_vec())
                .collect();
            drop(roles);
            assert!(original.upgrade().is_none());
            assert_eq!(
                case.root.budget().used_bytes(),
                OwnedBudget::resources().unwrap().initial_reserved_bytes
            );
            for ((_, frame), snapshot) in held.iter().zip(snapshot) {
                assert_eq!(frame.as_bytes(), snapshot);
            }
        });
    }
}
