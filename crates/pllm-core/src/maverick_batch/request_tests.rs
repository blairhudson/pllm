use super::super::super::{
    testing::{self, FixationEvent as Event, Fixture},
    FIELD, MAX_ATTEMPTS,
};
use super::*;

fn inputs() -> [[Vec<u32>; 2]; 2] {
    std::array::from_fn(|batch| {
        std::array::from_fn(|slot| {
            (0..DIMENSION)
                .map(|i| ((i + 1) * (7919 + batch * 103 + slot * 37)) as u32)
                .collect()
        })
    })
}
fn views(x: &[[Vec<u32>; 2]; 2]) -> [[&[u32]; 2]; 2] {
    [[&x[0][0], &x[0][1]], [&x[1][0], &x[1][1]]]
}
fn primary(memory: &Memory, length: usize, event: Event) -> Result<[Sparse; REPETITIONS]> {
    testing::record(event);
    Ok([
        Sparse::from_pairs(
            memory,
            length,
            CHECK_WEIGHT,
            &testing::challenge(length, CHECK_WEIGHT, 0),
        )?,
        Sparse::from_pairs(
            memory,
            length,
            CHECK_WEIGHT,
            &testing::challenge(length, CHECK_WEIGHT, 1),
        )?,
    ])
}
fn fixed_issue(c: &mut Client, x: &[[Vec<u32>; 2]; 2]) -> (Protocol2Request, [[Request; 2]; 2]) {
    let mut draws = 0;
    let out = c
        .issue_request_with(views(x), |m| {
            draws += 1;
            Sparse::from_pairs(
                m,
                LENGTH,
                PRIVACY_WEIGHT,
                &testing::challenge(LENGTH, PRIVACY_WEIGHT, draws),
            )
        })
        .unwrap();
    assert_eq!(draws, 4);
    out
}
fn originals(s: &mut Server, r: &[[Request; 2]; 2]) -> [Reply; 4] {
    [
        s.evaluate(r[0][0].view()).unwrap(),
        s.evaluate(r[0][1].view()).unwrap(),
        s.evaluate(r[1][0].view()).unwrap(),
        s.evaluate(r[1][1].view()).unwrap(),
    ]
}
fn begin(
    c: &mut Client,
    r: Protocol2Request,
    replies: &[ReplyView<'_>],
) -> Result<(ChallengedRequest, [PrimaryChallenge; 2])> {
    c.begin_request_with(r, replies, |m| primary(m, LENGTH, Event::PrimaryDraw))
}
fn finish(
    c: &mut Client,
    r: ChallengedRequest,
    aux: &[AuxiliaryReplyView<'_>],
) -> Result<AcceptedRequest> {
    c.finish_request_with(r, aux, |m| {
        primary(m, SECONDARY_LENGTH, Event::SecondaryDraw)
    })
}

#[test]
fn four_generated_nonzero_products_atomic_acceptance_and_global_events() {
    let f = Fixture::new();
    let (mut c, mut s) = f.roles().unwrap();
    let x = inputs();
    let (r, frames) = fixed_issue(&mut c, &x);
    assert_eq!(c.attempts_issued(), 2);
    assert_eq!(frames[1][0].header.attempt, frames[0][0].header.attempt + 1);
    assert!(s.evaluate(frames[0][0].view()).is_ok());
    // This first frame was spent by the preceding call; build a fresh request.
    drop(r);
    drop(frames);
    let (r, frames) = fixed_issue(&mut c, &x);
    let replies = originals(&mut s, &frames);
    testing::begin_fixation_trace();
    let (r, p) = begin(
        &mut c,
        r,
        &replies.iter().map(Reply::view).collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        testing::take_fixation_trace(),
        vec![
            Event::OriginalsFixed,
            Event::OriginalsFixed,
            Event::PrimaryDraw,
            Event::PrimaryDraw
        ]
    );
    let aux = [
        s.project(p[0].view()).unwrap(),
        s.project(p[1].view()).unwrap(),
    ];
    assert!(matches!(s.project(p[0].view()), Err(Error::Replay)));
    testing::begin_fixation_trace();
    let accepted = finish(&mut c, r, &[aux[0].view(), aux[1].view()]).unwrap();
    assert_eq!(
        testing::take_fixation_trace(),
        vec![
            Event::AuxiliaryFixed,
            Event::AuxiliaryFixed,
            Event::SecondaryDraw,
            Event::SecondaryDraw
        ]
    );
    for (batch, accepted) in accepted.batches().iter().enumerate() {
        for (slot, out) in accepted.outputs().iter().enumerate() {
            let expected: Vec<u32> = f.matrices[slot]
                .chunks_exact(DIMENSION)
                .map(|row| {
                    (row.iter()
                        .zip(&x[batch][slot])
                        .map(|(&a, &b)| u128::from(a) * u128::from(b))
                        .sum::<u128>()
                        % u128::from(FIELD)) as u32
                })
                .collect();
            assert_eq!(out.values(), expected);
        }
    }
    assert!(!c.has_live_window());
    assert!(c.memory_stats().charged_peak_bytes <= super::super::super::ALLOCATION_CEILING);
}

#[test]
fn tampering_either_original_batch_releases_no_outputs() {
    for bad in [0, 2] {
        let f = Fixture::new();
        let (mut c, mut s) = f.roles().unwrap();
        let (r, frames) = fixed_issue(&mut c, &inputs());
        let replies = originals(&mut s, &frames);
        let mut changed = replies[bad].view().values.to_vec();
        changed[0] = (changed[0] + 1) % FIELD;
        let mut rv: Vec<_> = replies.iter().map(Reply::view).collect();
        rv[bad].values = &changed;
        let (r, p) = begin(&mut c, r, &rv).unwrap();
        let a = [
            s.project(p[0].view()).unwrap(),
            s.project(p[1].view()).unwrap(),
        ];
        let result = finish(&mut c, r, &[a[0].view(), a[1].view()]);
        assert!(matches!(result, Err(Error::Rejected)));
        assert!(!c.has_live_window());
    }
}

#[test]
fn tampering_either_auxiliary_batch_releases_no_outputs() {
    for bad in 0..2 {
        let f = Fixture::new();
        let (mut c, mut s) = f.roles().unwrap();
        let (r, frames) = fixed_issue(&mut c, &inputs());
        let replies = originals(&mut s, &frames);
        let (r, p) = begin(
            &mut c,
            r,
            &replies.iter().map(Reply::view).collect::<Vec<_>>(),
        )
        .unwrap();
        let a = [
            s.project(p[0].view()).unwrap(),
            s.project(p[1].view()).unwrap(),
        ];
        let mut changed = a[bad].view().values.to_vec();
        changed[0] = (changed[0] + 1) % FIELD;
        let mut av = [a[0].view(), a[1].view()];
        av[bad].values = &changed;
        assert!(matches!(finish(&mut c, r, &av), Err(Error::Rejected)));
        assert!(!c.has_live_window());
    }
}

#[test]
fn wrong_order_session_omission_and_duplicate_retire_without_primary_entropy() {
    for bad in 0..4 {
        let f = Fixture::new();
        let (mut c, mut s) = f.roles().unwrap();
        let (r, frames) = fixed_issue(&mut c, &inputs());
        let replies = originals(&mut s, &frames);
        let mut rv: Vec<_> = replies.iter().map(Reply::view).collect();
        match bad {
            0 => rv.swap(0, 2),
            1 => rv[0].header.session[0] ^= 1,
            2 => {
                rv.pop();
            }
            _ => rv[3] = rv[0],
        }
        assert!(c
            .begin_request_with(r, &rv, |_| panic!(
                "primary before valid complete originals"
            ))
            .is_err());
        assert!(!c.has_live_window());
        assert_eq!(c.attempts_issued(), 2);
    }
}

#[test]
fn wrong_auxiliary_order_refuses_before_secondary_entropy() {
    let f = Fixture::new();
    let (mut c, mut s) = f.roles().unwrap();
    let (r, frames) = fixed_issue(&mut c, &inputs());
    let replies = originals(&mut s, &frames);
    let (r, p) = begin(
        &mut c,
        r,
        &replies.iter().map(Reply::view).collect::<Vec<_>>(),
    )
    .unwrap();
    let a = [
        s.project(p[0].view()).unwrap(),
        s.project(p[1].view()).unwrap(),
    ];
    assert!(matches!(
        c.finish_request_with(r, &[a[1].view(), a[0].view()], |_| panic!(
            "secondary before complete valid auxiliaries"
        )),
        Err(Error::Binding)
    ));
    assert!(!c.has_live_window());
}

#[test]
fn drop_busy_entropy_failure_and_attempt_limit_preserve_single_counter() {
    let f = Fixture::new();
    let (mut c, _s) = f.roles().unwrap();
    let x = inputs();
    let (r, _) = fixed_issue(&mut c, &x);
    assert!(matches!(
        c.issue_request_with(views(&x), |_| panic!("busy entropy")),
        Err(Error::Busy)
    ));
    assert_eq!(c.attempts_issued(), 4);
    assert!(c.has_live_window());
    drop(r);
    assert!(!c.has_live_window());
    assert!(matches!(
        c.issue_request_with(views(&x), |_| Err(Error::Randomness)),
        Err(Error::Randomness)
    ));
    assert_eq!(c.attempts_issued(), 6);
    assert!(!c.has_live_window());
    c.control.issued.store(MAX_ATTEMPTS - 1, Ordering::Release);
    assert!(matches!(
        c.issue_request_with(views(&x), |_| panic!("attempt limit entropy")),
        Err(Error::Attempts)
    ));
    let (q, _) = c.issue_query(0, &x[0][0]).unwrap();
    drop(q);
    assert_eq!(c.attempts_issued(), MAX_ATTEMPTS);
}

#[test]
fn complete_quote_exact_fit_one_short_and_no_entropy_before_refusal() {
    for short in [false, true] {
        let f = Fixture::new();
        let (mut c, _s) = f.roles().unwrap();
        let memory = c.material.memory.clone();
        let plan = c.request_allocation_plan().unwrap();
        let stats = c.memory_stats();
        let blocker = memory
            .reserve(
                stats.ceiling_bytes - stats.charged_live_bytes - plan.additional_bytes()
                    + usize::from(short),
            )
            .unwrap();
        let mut entropy = false;
        let result = c.issue_request_with(views(&inputs()), |m| {
            entropy = true;
            Sparse::from_pairs(
                m,
                LENGTH,
                PRIVACY_WEIGHT,
                &testing::challenge(LENGTH, PRIVACY_WEIGHT, 1),
            )
        });
        if short {
            assert!(matches!(result, Err(Error::Allocation)));
            assert!(!entropy);
        } else {
            assert!(result.is_ok());
            assert!(entropy);
        }
        drop(blocker);
        assert_eq!(c.attempts_issued(), 2);
    }
}

#[test]
fn original_copy_allocation_failure_consumes_whole_request_without_challenges() {
    let f = Fixture::new();
    let (mut c, mut s) = f.roles().unwrap();
    let (r, frames) = fixed_issue(&mut c, &inputs());
    let replies = originals(&mut s, &frames);
    let memory = c.material.memory.clone();
    let stats = c.memory_stats();
    let blocker = memory
        .reserve(stats.ceiling_bytes - stats.charged_live_bytes)
        .unwrap();
    assert!(matches!(
        c.begin_request_with(
            r,
            &replies.iter().map(Reply::view).collect::<Vec<_>>(),
            |_| panic!("copy failure sampled primary")
        ),
        Err(Error::Allocation)
    ));
    assert!(!c.has_live_window());
    drop(blocker);
}

#[test]
fn dropped_challenged_request_does_not_restore_ordinals() {
    let f = Fixture::new();
    let (mut c, mut s) = f.roles().unwrap();
    let (r, frames) = fixed_issue(&mut c, &inputs());
    let replies = originals(&mut s, &frames);
    let (r, _) = begin(
        &mut c,
        r,
        &replies.iter().map(Reply::view).collect::<Vec<_>>(),
    )
    .unwrap();
    drop(r);
    assert!(!c.has_live_window());
    assert_eq!(c.attempts_issued(), 2);
}

#[test]
fn every_partial_noise_failure_cleans_before_releasing_the_window() {
    for fail_at in 0..4 {
        let f = Fixture::new();
        let (mut c, _s) = f.roles().unwrap();
        let before = c.memory_stats().charged_live_bytes;
        let mut draws = 0;
        let result = c.issue_request_with(views(&inputs()), |memory| {
            if draws == fail_at {
                return Err(Error::Randomness);
            }
            draws += 1;
            Sparse::from_pairs(
                memory,
                LENGTH,
                PRIVACY_WEIGHT,
                &testing::challenge(LENGTH, PRIVACY_WEIGHT, draws),
            )
        });
        assert!(matches!(result, Err(Error::Randomness)));
        assert_eq!(draws, fail_at);
        assert!(!c.has_live_window());
        assert_eq!(c.attempts_issued(), 2);
        assert_eq!(c.memory_stats().charged_live_bytes, before);
    }
}

#[test]
fn invalid_input_burns_two_without_entropy_and_cannot_change_shape_or_field() {
    for bad in 0..2 {
        let f = Fixture::new();
        let (mut c, _s) = f.roles().unwrap();
        let mut x = inputs();
        if bad == 0 {
            x[1][1].pop();
        } else {
            x[1][1][0] = FIELD;
        }
        let result = c.issue_request_with(views(&x), |_| panic!("invalid input reached entropy"));
        assert!(matches!(result, Err(Error::Shape) | Err(Error::Field)));
        assert_eq!(c.attempts_issued(), 2);
        assert!(!c.has_live_window());
    }
}

#[test]
fn auxiliary_copy_allocation_failure_is_atomic_and_precedes_secondary() {
    let f = Fixture::new();
    let (mut c, mut s) = f.roles().unwrap();
    let (r, frames) = fixed_issue(&mut c, &inputs());
    let replies = originals(&mut s, &frames);
    let (r, p) = begin(
        &mut c,
        r,
        &replies.iter().map(Reply::view).collect::<Vec<_>>(),
    )
    .unwrap();
    let a = [
        s.project(p[0].view()).unwrap(),
        s.project(p[1].view()).unwrap(),
    ];
    let memory = c.material.memory.clone();
    let stats = c.memory_stats();
    let blocker = memory
        .reserve(stats.ceiling_bytes - stats.charged_live_bytes)
        .unwrap();
    let result = c.finish_request_with(r, &[a[0].view(), a[1].view()], |_| {
        panic!("copy failure sampled secondary")
    });
    assert!(matches!(result, Err(Error::Allocation)));
    assert!(!c.has_live_window());
    assert_eq!(c.attempts_issued(), 2);
    drop(blocker);
}

#[test]
fn secondary_entropy_failure_never_yields_a_first_accepted_batch() {
    for fail_at in 0..2 {
        let f = Fixture::new();
        let (mut c, mut s) = f.roles().unwrap();
        let (r, frames) = fixed_issue(&mut c, &inputs());
        let replies = originals(&mut s, &frames);
        let (r, p) = begin(
            &mut c,
            r,
            &replies.iter().map(Reply::view).collect::<Vec<_>>(),
        )
        .unwrap();
        let a = [
            s.project(p[0].view()).unwrap(),
            s.project(p[1].view()).unwrap(),
        ];
        let mut draws = 0;
        let result = c.finish_request_with(r, &[a[0].view(), a[1].view()], |m| {
            if draws == fail_at {
                return Err(Error::Randomness);
            }
            draws += 1;
            primary(m, SECONDARY_LENGTH, Event::SecondaryDraw)
        });
        assert!(matches!(result, Err(Error::Randomness)));
        assert!(!c.has_live_window());
        assert_eq!(c.attempts_issued(), 2);
    }
}

#[test]
fn either_primary_entropy_failure_retires_both_batches() {
    for fail_at in 0..2 {
        let f = Fixture::new();
        let (mut c, mut s) = f.roles().unwrap();
        let (r, frames) = fixed_issue(&mut c, &inputs());
        let replies = originals(&mut s, &frames);
        let mut draws = 0;
        let result = c.begin_request_with(
            r,
            &replies.iter().map(Reply::view).collect::<Vec<_>>(),
            |m| {
                if draws == fail_at {
                    return Err(Error::Randomness);
                }
                draws += 1;
                primary(m, LENGTH, Event::PrimaryDraw)
            },
        );
        assert!(matches!(result, Err(Error::Randomness)));
        assert!(!c.has_live_window());
        assert_eq!(c.attempts_issued(), 2);
    }
}

#[test]
fn whole_lifecycle_at_minimum_admissible_publication_cap_without_padding() {
    use super::super::super::{preprocess, AllocationPlan};
    let minimum = AllocationPlan::bounded().unwrap().planned_peak_bytes();
    assert!(matches!(
        AllocationPlan::for_shape(DIMENSION, DIMENSION, BATCH_SIZE, minimum - 1),
        Err(Error::Allocation)
    ));
    let plan = AllocationPlan::for_shape(DIMENSION, DIMENSION, BATCH_SIZE, minimum).unwrap();
    let f = Fixture::new();
    let (mut c, mut s) = preprocess(plan, f.source())
        .unwrap()
        .validate(f.source())
        .unwrap()
        .split()
        .unwrap();
    let x = inputs();
    let (r, frames) = fixed_issue(&mut c, &x);
    let replies = originals(&mut s, &frames);
    let (r, p) = begin(
        &mut c,
        r,
        &replies.iter().map(Reply::view).collect::<Vec<_>>(),
    )
    .unwrap();
    let a = [
        s.project(p[0].view()).unwrap(),
        s.project(p[1].view()).unwrap(),
    ];
    let accepted = finish(&mut c, r, &[a[0].view(), a[1].view()]).unwrap();
    assert_eq!(
        accepted
            .batches()
            .iter()
            .map(|b| b.outputs().len())
            .sum::<usize>(),
        4
    );
    assert!(!c.has_live_window());
    assert_eq!(c.attempts_issued(), 2);
    let stats = c.memory_stats();
    assert_eq!(stats.ceiling_bytes, minimum);
    assert!(stats.charged_peak_bytes <= minimum);
    println!("D45 full lifecycle: minimum publication cap {minimum} bytes; charged peak {} bytes; charged live with all four accepted outputs retained {} bytes",stats.charged_peak_bytes,stats.charged_live_bytes);
    // This checks the existing publication floor, NOT an assertion that the
    // conservative request quote is an irrevocable or tight peak reservation.
}
