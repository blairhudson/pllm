//! Reuse original three-role/router assertions from bias qualification; no new
//! transport, fake drain constructor, oracle permit, or receipt framework.
use super::*;
use crate::replicated_decoder::program::request_origin::source_bias::handoff::route_tests::{
    boundary, drive, enqueue, reconstruct, same,
};

fn execute(
    compiled: &Arc<Compiled>,
    workers: [Worker; 3],
    registry: &SourceRegistry<'_>,
    trace: &mut Vec<(puma::Party, puma::Party, Vec<u8>)>,
) -> [Invocation; 3] {
    // Caller just drained all original roles and their actual router queue.
    let mut invocations = workers.map(|worker| {
        Arc::clone(compiled)
            .begin(CoordinatorDrained { owner: worker }, registry)
            .unwrap()
    });
    let mut queue = Vec::new();
    for _ in 0..100_000 {
        if queue.is_empty() && invocations.iter().all(Invocation::complete) {
            return invocations;
        }
        if queue.is_empty() {
            for invocation in &mut invocations {
                if !invocation.complete() {
                    let frames = invocation.advance().unwrap();
                    enqueue(invocation.party(), frames, &mut queue, trace)
                }
            }
        } else {
            for (from, to, frame) in std::mem::take(&mut queue) {
                let frames = invocations[to as usize]
                    .receive(frame.as_bytes(), from)
                    .unwrap();
                enqueue(to, frames, &mut queue, trace)
            }
        }
    }
    panic!("original rotary BODY roles/router failed to drain")
}

pub(in crate::replicated_decoder) fn qualify(
    compiled: Arc<Compiled>,
    mut original: [Worker; 3],
    mut candidate: [Worker; 3],
    registry: &SourceRegistry<'_>,
) {
    let owner = &compiled.publication.owner;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    boundary(&mut original, owner.first, &mut a);
    boundary(&mut candidate, owner.first, &mut b);
    assert_eq!(a, b);
    same(&original, &candidate);
    let input: Vec<_> = owner
        .input
        .iter()
        .map(|(id, lane)| reconstruct(&original, &compiled.program, id, lane))
        .collect();
    let width = owner.shape()[3];
    let half = width / 2;
    let mut negative_fraction = false;
    let floors: Vec<_> = owner
        .rows
        .iter()
        .enumerate()
        .flat_map(|(r, row)| {
            let input = &input;
            (0..width).map(move |i| {
                let cosine = input[r * width + i] as i128 * row.cosine[i % half] as i128;
                let sine = input[r * width + (i + half) % width] as i128
                    * row.sine[i % half] as i128
                    * if i < half { -1 } else { 1 };
                (cosine, sine)
            })
        })
        .map(|(cosine, sine)| {
            negative_fraction |=
                (cosine < 0 && cosine % ONE as i128 != 0) || (sine < 0 && sine % ONE as i128 != 0);
            (cosine >> 18) + (sine >> 18)
        })
        .collect();
    assert!(
        negative_fraction,
        "signed per-term negative rounding boundary must be exercised"
    );
    let invocations = execute(&compiled, candidate, registry, &mut b);
    // Actual three-role BODY router drained; complete() alone never issues this.
    candidate = invocations.map(|owner| CoordinatorDrained { owner }.restore().unwrap());
    boundary(&mut original, owner.end, &mut a);
    assert_eq!(a, b);
    same(&original, &candidate);
    // Original native truncation independently selects floor/ceil for EACH
    // product, then adds. Exact component/frame equality above is stronger than
    // this independent clear interval; no deterministic-floor oracle assumed.
    for (r, row) in owner.rows.iter().enumerate() {
        for lane in 0..width {
            let actual = reconstruct(&original, &compiled.program, row.output, lane) as i128;
            assert!((floors[r * width + lane]..=floors[r * width + lane] + 2).contains(&actual));
        }
    }
    // KV initialize is a genuine zero-node alias. Continue to its first actual
    // attention read (or q's directly bound attention consumer), not a dummy op.
    let consumer = owner.consumer.as_ref().unwrap();
    let after = (consumer.end..compiled.program.logical_len())
        .find(|&id| {
            owner
                .rows
                .iter()
                .any(|row| body::integration::uses(compiled.program.recipe(id), row.output))
        })
        .unwrap_or(consumer.end);
    let target = after + 1;
    assert!(target < compiled.program.logical_len());
    boundary(&mut original, target, &mut a);
    boundary(&mut candidate, target, &mut b);
    assert_eq!(a, b);
    same(&original, &candidate);
    drive(&mut original, None, &mut a);
    drive(&mut candidate, None, &mut b);
    assert_eq!(a, b);
    assert!(!a.is_empty());
    same(&original, &candidate);
    for (original, candidate) in original.into_iter().zip(candidate) {
        assert_eq!(
            original.finish_frame().unwrap().as_slice(),
            candidate.finish_frame().unwrap().as_slice()
        );
    }
}

pub(in crate::replicated_decoder) fn pending_cancel(
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
                                compiled.publication.owner.first
                            )
                            .is_err()
                        );
                        pending.extend(frames);
                    }
                }
                BoundaryProgress::Reached => {
                    panic!("genuine projection must cross original protocol before rotary")
                }
            }
        }
        if !pending.is_empty() {
            break;
        }
    }
    assert!(!pending.is_empty());
    // No drain token constructed here. Undelivered source frames cancel/burn
    // this original attempt rather than resetting or replacing any Worker.
    for worker in &mut workers {
        worker.cancel();
        assert!(matches!(worker.advance(), Err(Error::Retired)))
    }
    drop(pending)
}

pub(in crate::replicated_decoder) fn failed_restore(
    compiled: Arc<Compiled>,
    mut workers: [Worker; 3],
    registry: &SourceRegistry<'_>,
    panic: bool,
) {
    use body::integration::restoration_tests as probes;
    use std::{cell::Cell, rc::Rc};
    let mut trace = Vec::new();
    boundary(&mut workers, compiled.publication.owner.first, &mut trace);
    let [invocation, other, next] = execute(&compiled, workers, registry, &mut trace);
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
            observed.set(true)
        },
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CoordinatorDrained { owner: invocation }.restore()
    }));
    if panic {
        assert!(result.is_err())
    } else {
        assert!(matches!(result, Ok(Err(Error::Binding))))
    }
    assert!(erased.get());
    assert_eq!(probes::counts(), (1, 2));
    drop(session)
}
