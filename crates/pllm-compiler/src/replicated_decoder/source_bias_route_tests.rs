//! Actual three-role/router qualification, not a local drain permit constructor.
use super::*;
use puma::Party;

type Frame = (Party, Party, Vec<u8>);
type Pending = (Party, Party, PeerBytes);
fn enqueue(
    from: Party,
    frames: Vec<(Party, PeerBytes)>,
    queue: &mut Vec<Pending>,
    trace: &mut Vec<Frame>,
) {
    for (to, frame) in frames {
        trace.push((from, to, frame.as_bytes().to_vec()));
        queue.push((from, to, frame));
    }
}
fn boundary(workers: &mut [Worker; 3], end: usize, trace: &mut Vec<Frame>) {
    drive(workers, Some(end), trace)
}
fn drive(workers: &mut [Worker; 3], end: Option<usize>, trace: &mut Vec<Frame>) {
    let mut queue = Vec::new();
    let mut reached = [false; 3];
    for _ in 0..100_000 {
        if queue.is_empty() {
            if reached.iter().all(|v| *v) {
                return;
            }
            for (role, worker) in workers.iter_mut().enumerate() {
                if !reached[role] {
                    let progress = match end {
                        Some(end) => worker.advance_before(end).unwrap(),
                        None if worker.complete() => BoundaryProgress::Reached,
                        None => BoundaryProgress::Frames(worker.advance().unwrap()),
                    };
                    match progress {
                        BoundaryProgress::Reached => reached[role] = true,
                        BoundaryProgress::Frames(frames) => {
                            enqueue(worker.party(), frames, &mut queue, trace)
                        }
                    }
                }
            }
        } else {
            for (from, to, frame) in std::mem::take(&mut queue) {
                let frames = workers[to as usize]
                    .receive(frame.as_bytes(), from)
                    .unwrap();
                enqueue(to, frames, &mut queue, trace);
            }
        }
    }
    panic!("original three-role router failed to reach boundary");
}
fn same(a: &[Worker; 3], b: &[Worker; 3]) {
    for (a, b) in a.iter().zip(b) {
        let (step_a, cursor_a, values_a) = a.test_boundary_state().unwrap();
        let (step_b, cursor_b, values_b) = b.test_boundary_state().unwrap();
        assert_eq!((step_a, cursor_a), (step_b, cursor_b));
        assert_eq!(values_a.collect::<Vec<_>>(), values_b.collect::<Vec<_>>());
        assert_eq!(a.loop_test_state(), b.loop_test_state());
    }
}
fn reconstruct(workers: &[Worker; 3], program: &Program, owner: usize, lane: usize) -> i64 {
    let mut value = 0u64;
    for worker in workers {
        let (_, _, mut values) = worker.test_boundary_state().unwrap();
        let (first, _) = values.nth(program.value_slot(owner)).unwrap().unwrap();
        value = value.wrapping_add(first[lane]);
    }
    value as i64
}

/// Both Programs already exist. Candidate authority was minted BEFORE the
/// independent original oracle was compiled; neither trace nor oracle is input
/// to creation/publication. The sole test drain issuances occur inside this
/// owning three-role loop, after every role and delivery queue is drained.
pub(in crate::replicated_decoder) fn qualify(
    compiled: Arc<Compiled>,
    mut original: [Worker; 3],
    mut candidate: [Worker; 3],
    registry: &SourceRegistry<'_>,
    weights: &[i64],
    bias: &[i64],
) {
    let d = compiled.publication.owner.description();
    let [rows, inner, columns] = d.geometry;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    boundary(&mut original, d.first + 1, &mut a);
    boundary(&mut candidate, d.first + 1, &mut b);
    assert_eq!(a, b);
    same(&original, &candidate);
    let inputs: Vec<_> = (0..rows * inner)
        .map(|i| reconstruct(&candidate, &compiled.program, d.first, i))
        .collect();
    let mut negative_fraction = false;
    let raw_expected: Vec<_> = (0..rows * columns)
        .map(|i| {
            let row = i / columns;
            let col = i % columns;
            let raw: i128 = (0..inner)
                .map(|k| inputs[row * inner + k] as i128 * weights[col * inner + k] as i128)
                .sum();
            negative_fraction |= raw < 0 && raw % (1 << 18) != 0;
            i64::try_from(raw).unwrap()
        })
        .collect();
    assert!(
        negative_fraction,
        "signed negative floor boundary must be exercised"
    );
    boundary(&mut original, d.first + 2, &mut a);
    boundary(&mut candidate, d.first + 2, &mut b);
    assert_eq!(a, b);
    same(&original, &candidate);
    for (i, &raw) in raw_expected.iter().enumerate() {
        assert_eq!(
            reconstruct(&original, &compiled.program, d.first + 1, i),
            raw
        );
    }
    boundary(&mut original, d.bias_start(), &mut a);
    boundary(&mut candidate, d.bias_start(), &mut b);
    assert_eq!(a, b);
    same(&original, &candidate);
    // The unchanged native protocol stochastically selects floor or ceil (see
    // tensor_numeric's rescale contracts). Pin the exact seeded original result,
    // not a false deterministic-floor replacement or an output tolerance.
    let expected: Vec<_> = raw_expected
        .iter()
        .enumerate()
        .map(|(i, &raw)| {
            let scaled = reconstruct(&original, &compiled.program, d.first + 2, i);
            let floor = raw >> 18;
            let ceil = floor + i64::from(raw % (1 << 18) != 0);
            assert!(scaled == floor || scaled == ceil);
            scaled + bias[i % columns]
        })
        .collect();
    // All three original workers and router now drained at the SAME boundary.
    let mut invocations = candidate.map(|worker| {
        Arc::clone(&compiled)
            .begin(CoordinatorDrained { owner: worker }, registry)
            .unwrap()
    });
    let mut queue = Vec::new();
    for _ in 0..1000 {
        if queue.is_empty() && invocations.iter().all(Invocation::complete) {
            break;
        }
        if queue.is_empty() {
            for invocation in &mut invocations {
                if !invocation.complete() {
                    let frames = invocation.advance().unwrap();
                    enqueue(invocation.party(), frames, &mut queue, &mut b);
                }
            }
        } else {
            for (from, to, frame) in std::mem::take(&mut queue) {
                let frames = invocations[to as usize]
                    .receive(frame.as_bytes(), from)
                    .unwrap();
                enqueue(to, frames, &mut queue, &mut b);
            }
        }
    }
    assert!(queue.is_empty() && invocations.iter().all(Invocation::complete));
    // No primitive identity/work loan: invocation custody moves original runtime.
    candidate = invocations.map(|owner| CoordinatorDrained { owner }.restore().unwrap());
    boundary(&mut original, d.end, &mut a);
    assert_eq!(a, b);
    same(&original, &candidate);
    for (i, &expected) in expected.iter().enumerate() {
        let (owner, lane) = d.output_word(i);
        assert_eq!(
            reconstruct(&candidate, &compiled.program, owner, lane),
            expected
        );
    }
    // Original q/k/v creation precedes their zero-node reshape aliases. Resume
    // those SAME operators, then reach the first genuine consumer of q's alias.
    let consumer_start = compiled.publication.owner.consumer_start().unwrap();
    let consumer = (consumer_start..compiled.program.logical_len())
        .find(|&id| {
            (d.bias_start()..d.end)
                .step_by(3)
                .any(|first| body::integration::uses(compiled.program.recipe(id), first + 2))
        })
        .unwrap();
    boundary(&mut original, consumer + 1, &mut a);
    boundary(&mut candidate, consumer + 1, &mut b);
    assert_eq!(a, b);
    same(&original, &candidate);
    drive(&mut original, None, &mut a);
    drive(&mut candidate, None, &mut b);
    assert_eq!(a, b);
    assert!(!a.is_empty());
    same(&original, &candidate);
    for (a, b) in original.into_iter().zip(candidate) {
        assert_eq!(
            a.finish_frame().unwrap().as_slice(),
            b.finish_frame().unwrap().as_slice()
        );
    }
}

pub(in crate::replicated_decoder) fn pending_frames_and_cancellation_retire_original_attempt(
    compiled: Arc<Compiled>,
    mut workers: [Worker; 3],
) {
    let mut pending = Vec::new();
    for _ in 0..1000 {
        for worker in &mut workers {
            match compiled.advance_to_start(worker).unwrap() {
                BoundaryProgress::Frames(frames) => {
                    if !frames.is_empty() {
                        assert!(
                            SuspendedWorker::check_start(
                                worker,
                                &compiled.program,
                                compiled.publication.owner.description().bias_start()
                            )
                            .is_err()
                        );
                        pending.extend(frames);
                    }
                }
                BoundaryProgress::Reached => {
                    panic!("fixture must cross original protocol before bias")
                }
            }
        }
        if !pending.is_empty() {
            break;
        }
    }
    assert!(!pending.is_empty());
    // Undelivered frames never become coordinator custody. No test drain value
    // is minted on this path; cancellation keeps original Worker unrecoverable.
    for worker in &mut workers {
        worker.cancel();
        assert!(matches!(worker.advance(), Err(Error::Retired)));
    }
    drop(pending);
}

/// Fault injection reuses existing restoration hooks; custody still comes only
/// from actual roles/router. Neither fault nor local completion issues a drain.
pub(in crate::replicated_decoder) fn failed_restore_erases_before_owner_release(
    compiled: Arc<Compiled>,
    mut workers: [Worker; 3],
    registry: &SourceRegistry<'_>,
    panic: bool,
) {
    use body::integration::restoration_tests as probes;
    use std::{cell::Cell, rc::Rc};
    let mut trace = Vec::new();
    boundary(
        &mut workers,
        compiled.publication.owner.description().bias_start(),
        &mut trace,
    );
    let mut invocations = workers.map(|worker| {
        Arc::clone(&compiled)
            .begin(CoordinatorDrained { owner: worker }, registry)
            .unwrap()
    });
    let mut queue = Vec::new();
    for _ in 0..1000 {
        if queue.is_empty() && invocations.iter().all(Invocation::complete) {
            break;
        }
        if queue.is_empty() {
            for invocation in &mut invocations {
                if !invocation.complete() {
                    let frames = invocation.advance().unwrap();
                    enqueue(invocation.party(), frames, &mut queue, &mut trace);
                }
            }
        } else {
            for (from, to, frame) in std::mem::take(&mut queue) {
                let frames = invocations[to as usize]
                    .receive(frame.as_bytes(), from)
                    .unwrap();
                enqueue(to, frames, &mut queue, &mut trace);
            }
        }
    }
    assert!(queue.is_empty() && invocations.iter().all(Invocation::complete));
    let [invocation, other, next] = invocations;
    drop(other);
    drop(next);
    let owners = (
        Arc::downgrade(&compiled),
        Arc::downgrade(&compiled.plan),
        Arc::downgrade(&compiled.program),
    );
    let erased = Rc::new(Cell::new(false));
    let observed = Rc::clone(&erased);
    let session = probes::install(
        MAX_PROGRAM_BYTES,
        if panic {
            probes::Fault::ReadPanic
        } else {
            probes::Fault::ReadError
        },
        move || {
            assert!(
                owners.0.strong_count() > 0
                    && owners.1.strong_count() > 0
                    && owners.2.strong_count() >= 2
            );
            observed.set(true);
        },
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CoordinatorDrained { owner: invocation }.restore()
    }));
    if panic {
        assert!(result.is_err());
    } else {
        assert!(matches!(result, Ok(Err(Error::Binding))));
    }
    assert!(erased.get());
    assert_eq!(probes::counts(), (1, 2));
    drop(session);
}
