//! Bounded exact-field dependency, not model numerics or a security certificate.
#[allow(dead_code, unused_imports)]
#[path = "../src/maverick_batch/mod.rs"]
mod maverick_batch;

use maverick_batch::testing::{self, FixationEvent, Fixture};
use maverick_batch::*;

fn expect_error<T>(result: Result<T>, expected: Error) {
    assert_eq!(result.err(), Some(expected));
}

// Independent wide-integer oracle: all 64 products fit together in u128. No
// native arithmetic, masking, encoding, verifier or matrix kernel is used here.
fn matvec(matrix: &[u32], input: &[u32]) -> Vec<u32> {
    matrix
        .chunks_exact(DIMENSION)
        .map(|row| {
            (row.iter()
                .zip(input)
                .map(|(&a, &b)| u128::from(a) * u128::from(b))
                .sum::<u128>()
                % u128::from(FIELD)) as u32
        })
        .collect()
}

fn primary(f: &Fixture) -> [&[u32]; 2] {
    [&f.primary_challenges[0], &f.primary_challenges[1]]
}

fn secondary(f: &Fixture) -> [&[u32]; 2] {
    [&f.secondary_challenges[0], &f.secondary_challenges[1]]
}

#[test]
fn dependent_integer_oracle_matches_coupled_transcript_with_immutable_inputs() {
    let f = Fixture::new();
    let first = matvec(&f.matrices[0], &f.inputs[0]);
    let second = matvec(&f.matrices[1], &first);
    let (mut client, mut server) = f.roles().unwrap();
    let mut input = f.inputs[0].clone();
    let (owner, r0) = testing::issue_dependent_fixed(&mut client, &input, &f.noise[0]).unwrap();
    input.fill(FIELD); // Issuance owns its input, not this mutable caller buffer.
    let y0 = server.evaluate(r0.view()).unwrap();
    let mut transport0 = y0.view().values.to_vec();
    testing::begin_fixation_trace();
    let (owner, r1) = testing::continue_dependent_fixed(
        &mut client,
        owner,
        ReplyView {
            header: y0.view().header,
            values: &transport0,
        },
        &f.noise[1],
    )
    .unwrap();
    transport0.fill(FIELD); // Cannot replace the original driving the dependency.
    let y1 = server.evaluate(r1.view()).unwrap();
    let mut transport1 = y1.view().values.to_vec();
    let (fixed, challenge) = testing::begin_dependent_fixed(
        &mut client,
        owner,
        ReplyView {
            header: y1.view().header,
            values: &transport1,
        },
        primary(&f),
    )
    .unwrap();
    transport1.fill(FIELD);
    let auxiliary = server.project(challenge.view()).unwrap();
    let (accepted, secondary_trace, original_trace) =
        testing::finish_batch_fixed(&mut client, fixed, auxiliary.view(), secondary(&f)).unwrap();
    assert_eq!(accepted.outputs()[0].values(), first);
    assert_eq!(accepted.outputs()[1].values(), second);
    assert_eq!(
        testing::take_fixation_trace(),
        [
            FixationEvent::FirstOriginalFixed,
            FixationEvent::DependentNoise,
            FixationEvent::OriginalsFixed,
            FixationEvent::PrimaryDraw,
            FixationEvent::AuxiliaryFixed,
            FixationEvent::SecondaryDraw,
        ]
    );
    assert!(!client.has_live_window());
    assert_eq!(client.attempts_issued(), 1);

    // Same exact algorithm, original matrix order, noise and challenges as the
    // existing coupled API, with its second input supplied by the independent
    // oracle. The historical frozen-vector test remains untouched.
    let (mut coupled, mut evaluator) = f.roles().unwrap();
    let (batch, requests) = testing::issue_batch_fixed(
        &mut coupled,
        [&f.inputs[0], &first],
        [&f.noise[0], &f.noise[1]],
    )
    .unwrap();
    assert_eq!(requests[0].view().values, r0.view().values);
    assert_eq!(requests[1].view().values, r1.view().values);
    let replies = [
        evaluator.evaluate(requests[0].view()).unwrap(),
        evaluator.evaluate(requests[1].view()).unwrap(),
    ];
    assert_eq!(replies[0].view().values, y0.view().values);
    assert_eq!(replies[1].view().values, y1.view().values);
    let (fixed, coupled_challenge) = testing::begin_fixed(
        &mut coupled,
        batch,
        &[replies[0].view(), replies[1].view()],
        primary(&f),
    )
    .unwrap();
    assert_eq!(challenge.view().entries, coupled_challenge.view().entries);
    let coupled_auxiliary = evaluator.project(coupled_challenge.view()).unwrap();
    assert_eq!(auxiliary.view().values, coupled_auxiliary.view().values);
    let (coupled_output, s, o) =
        testing::finish_batch_fixed(&mut coupled, fixed, coupled_auxiliary.view(), secondary(&f))
            .unwrap();
    assert_eq!(s, secondary_trace);
    assert_eq!(o, original_trace);
    for slot in 0..2 {
        assert_eq!(
            accepted.outputs()[slot].values(),
            coupled_output.outputs()[slot].values()
        );
    }
}

#[test]
fn false_first_original_drives_tentative_descendant_but_neither_is_accepted() {
    let f = Fixture::new();
    let (mut client, mut server) = f.roles().unwrap();
    for corrupt in 0..3 {
        let (owner, r0) =
            testing::issue_dependent_fixed(&mut client, &f.inputs[0], &f.noise[0]).unwrap();
        let y0 = server.evaluate(r0.view()).unwrap();
        let mut first = y0.view().values.to_vec();
        if corrupt == 0 {
            first[0] = (first[0] + 1) % FIELD;
        }
        let (owner, r1) = testing::continue_dependent_fixed(
            &mut client,
            owner,
            ReplyView {
                header: y0.view().header,
                values: &first,
            },
            &f.noise[1],
        )
        .unwrap();
        let y1 = server.evaluate(r1.view()).unwrap();
        let mut second = y1.view().values.to_vec();
        if corrupt == 1 {
            second[63] = (second[63] + 1) % FIELD;
        }
        let (fixed, challenge) = testing::begin_dependent_fixed(
            &mut client,
            owner,
            ReplyView {
                header: y1.view().header,
                values: &second,
            },
            primary(&f),
        )
        .unwrap();
        let auxiliary = server.project(challenge.view()).unwrap();
        let mut bad_auxiliary = auxiliary.view().values.to_vec();
        if corrupt == 2 {
            // Orthogonal to xhat0: all original equations still match. Only the
            // mandatory secondary relation detects this auxiliary lie.
            let x = r0.view().values;
            bad_auxiliary[0] =
                ((u64::from(bad_auxiliary[0]) + u64::from(x[1])) % u64::from(FIELD)) as u32;
            bad_auxiliary[1] = ((u64::from(bad_auxiliary[1]) + u64::from(FIELD) - u64::from(x[0]))
                % u64::from(FIELD)) as u32;
            assert_ne!(bad_auxiliary, auxiliary.view().values);
            let dot = |a: &[u32]| {
                (a.iter()
                    .zip(x)
                    .map(|(&a, &b)| u128::from(a) * u128::from(b))
                    .sum::<u128>()
                    % u128::from(FIELD)) as u32
            };
            assert_eq!(
                dot(&bad_auxiliary[..64]),
                dot(&auxiliary.view().values[..64])
            );
        }
        expect_error(
            testing::finish_batch_fixed(
                &mut client,
                fixed,
                AuxiliaryReplyView {
                    header: auxiliary.view().header,
                    values: &bad_auxiliary,
                },
                secondary(&f),
            ),
            Error::Rejected,
        );
        assert!(!client.has_live_window());
    }
}

#[test]
fn dependent_stage_binding_malformed_and_replay_fail_closed() {
    let f = Fixture::new();
    let (mut client, mut server) = f.roles().unwrap();
    for stage in 0..2 {
        for bad in 0..5 {
            let (owner, r0) = client.issue_dependent(&f.inputs[0]).unwrap();
            let y0 = server.evaluate(r0.view()).unwrap();
            expect_error(server.evaluate(r0.view()), Error::Replay);
            let mut header = y0.view().header;
            header.slot = u8::MAX;
            expect_error(
                server.project(PrimaryChallengeView {
                    header,
                    entries: primary(&f),
                }),
                Error::Replay,
            ); // One evaluated original is never enough.
            let change = |view: ReplyView<'_>| {
                let mut header = view.header;
                let mut values = view.values.to_vec();
                let error = match bad {
                    0 => {
                        values.pop();
                        Error::Shape
                    }
                    1 => {
                        values[0] = FIELD;
                        Error::Field
                    }
                    2 => {
                        header.publication[0] ^= 1;
                        Error::Binding
                    }
                    3 => {
                        header.session[0] ^= 1;
                        Error::Binding
                    }
                    _ => {
                        header.slot ^= 1;
                        Error::Binding
                    }
                };
                (header, values, error)
            };
            testing::begin_fixation_trace();
            if stage == 0 {
                let (header, values, error) = change(y0.view());
                expect_error(
                    client.continue_dependent(
                        owner,
                        ReplyView {
                            header,
                            values: &values,
                        },
                    ),
                    error,
                );
            } else {
                let (owner, r1) = client.continue_dependent(owner, y0.view()).unwrap();
                let y1 = server.evaluate(r1.view()).unwrap();
                let (header, values, error) = change(y1.view());
                expect_error(
                    client.begin_dependent_verification(
                        owner,
                        ReplyView {
                            header,
                            values: &values,
                        },
                    ),
                    error,
                );
            }
            // Invalid originals never get as far as either challenge boundary.
            assert!(!testing::take_fixation_trace().contains(&FixationEvent::OriginalsFixed));
            assert!(!client.has_live_window());
        }
    }
    // A reply from a prior attempt cannot stand in for the current original.
    let (old, request) = client.issue_dependent(&f.inputs[0]).unwrap();
    let old_reply = server.evaluate(request.view()).unwrap();
    drop(old);
    let (new, _) = client.issue_dependent(&f.inputs[0]).unwrap();
    expect_error(
        client.continue_dependent(new, old_reply.view()),
        Error::Binding,
    );
}

#[test]
fn drops_busy_and_all_entropy_boundaries_preserve_one_window_and_attempts() {
    let f = Fixture::new();
    let (mut client, mut server) = f.roles().unwrap();
    let idle = client.memory_stats().charged_live_bytes;
    expect_error(
        testing::issue_dependent_rng_failure(&mut client, &f.inputs[0]),
        Error::Randomness,
    );
    for stage in 0..5 {
        let (owner, r0) = client.issue_dependent(&f.inputs[0]).unwrap();
        expect_error(client.issue_dependent(&f.inputs[0]), Error::Busy);
        assert!(client.has_live_window());
        let y0 = server.evaluate(r0.view()).unwrap();
        if stage == 0 {
            expect_error(
                testing::continue_dependent_rng_failure(&mut client, owner, y0.view()),
                Error::Randomness,
            );
        } else {
            let (owner, r1) = client.continue_dependent(owner, y0.view()).unwrap();
            let y1 = server.evaluate(r1.view()).unwrap();
            if stage == 1 {
                expect_error(
                    testing::begin_dependent_rng_failure(&mut client, owner, y1.view()),
                    Error::Randomness,
                );
            } else if stage == 2 {
                drop(owner); // Retires first fixed original AND both private claims.
            } else {
                let (fixed, primary) = client
                    .begin_dependent_verification(owner, y1.view())
                    .unwrap();
                if stage == 3 {
                    let auxiliary = server.project(primary.view()).unwrap();
                    expect_error(
                        testing::finish_batch_rng_failure(&mut client, fixed, auxiliary.view()),
                        Error::Randomness,
                    );
                } else {
                    drop(fixed); // Cancellation after primary reveal retires both claims.
                }
            }
        }
        assert!(!client.has_live_window());
    }
    let (owner, request) = client.issue_dependent(&f.inputs[0]).unwrap();
    drop(owner);
    drop(request);
    assert!(!client.has_live_window());
    assert_eq!(client.memory_stats().charged_live_bytes, idle);
    while client.attempts_issued() < MAX_ATTEMPTS {
        expect_error(client.issue_dependent(&[]), Error::Shape);
    }
    expect_error(client.issue_dependent(&f.inputs[0]), Error::Attempts);
    assert!(!client.has_live_window());
}

#[test]
fn foreign_client_cannot_consume_either_dependent_stage() {
    let f = Fixture::new();
    let (mut first, mut server) = f.roles().unwrap();
    let (mut foreign, _) = f.roles().unwrap();
    let (owner, r0) = first.issue_dependent(&f.inputs[0]).unwrap();
    let y0 = server.evaluate(r0.view()).unwrap();
    expect_error(foreign.continue_dependent(owner, y0.view()), Error::Binding);
    assert!(!first.has_live_window());
    let (owner, r0) = first.issue_dependent(&f.inputs[0]).unwrap();
    let y0 = server.evaluate(r0.view()).unwrap();
    let (owner, r1) = first.continue_dependent(owner, y0.view()).unwrap();
    let y1 = server.evaluate(r1.view()).unwrap();
    expect_error(
        foreign.begin_dependent_verification(owner, y1.view()),
        Error::Binding,
    );
    assert!(!first.has_live_window());
    assert!(!foreign.has_live_window());
}
