//! Targeted integration until the parent exports the new module from lib.rs.
#[allow(dead_code, unused_imports)]
#[path = "../src/maverick_batch/mod.rs"]
mod maverick_batch;

use std::collections::{BTreeMap, BTreeSet};

use maverick_batch::testing::{self, Fixture, FixtureCode};
use maverick_batch::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn golden() -> Value {
    serde_json::from_str(include_str!(
        "../../../paper/reproduction-contracts/maverick/vectors.json"
    ))
    .unwrap()
}

fn digest(value: &impl serde::Serialize) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn matrix_rows(flat: &[u32], columns: usize) -> Vec<&[u32]> {
    flat.chunks_exact(columns).collect()
}

fn expect_error<T>(result: Result<T>, expected: Error) {
    assert_eq!(result.err(), Some(expected));
}

fn raw_dot(a: &[u32], b: &[u32]) -> u32 {
    (a.iter()
        .zip(b)
        .map(|(&x, &y)| u128::from(x) * u128::from(y))
        .sum::<u128>()
        % u128::from(FIELD)) as u32
}

#[test]
fn frozen_protocol3_and_complete_protocol2_vectors_match() {
    let fixture = Fixture::new();
    let golden = golden();
    let expected = &golden["source64"];
    let batch_expected = &golden["source64_batch"];
    let draft = preprocess(AllocationPlan::bounded().unwrap(), fixture.source()).unwrap();
    let view = draft.view();
    let stack: Vec<Vec<u32>> = (0..2)
        .flat_map(|b| {
            (0..64).map(move |j| (0..512).map(|i| view.derived.q[b][i * 64 + j]).collect())
        })
        .collect();
    assert_eq!(
        digest(&stack),
        batch_expected["Qstack_sha256"].as_str().unwrap()
    );
    assert_eq!(
        digest(&matrix_rows(view.derived.q_auxiliary, 1024)),
        batch_expected["Qaux_sha256"].as_str().unwrap()
    );
    let (mut client, mut server) = draft.validate(fixture.source()).unwrap().split().unwrap();
    let (query, request) =
        testing::issue_fixed(&mut client, 0, &fixture.inputs[0], &fixture.noise[0]).unwrap();
    let (masked, correction) = testing::query_snapshot(&query);
    let mask: Vec<u32> = masked
        .iter()
        .zip(&fixture.inputs[0])
        .map(|(&a, &b)| {
            ((u64::from(a) + u64::from(FIELD) - u64::from(b)) % u64::from(FIELD)) as u32
        })
        .collect();
    assert_eq!(digest(&mask), expected["mask_sha256"].as_str().unwrap());
    assert_eq!(
        digest(&masked),
        expected["masked_x_sha256"].as_str().unwrap()
    );
    assert_eq!(
        digest(&correction),
        expected["correction_sha256"].as_str().unwrap()
    );
    let reply = server.evaluate(request.view()).unwrap();
    assert_eq!(
        digest(&reply.view().values),
        expected["masked_y_sha256"].as_str().unwrap()
    );
    let (accepted, checks) = testing::finish_query_fixed(
        &mut client,
        query,
        reply.view(),
        [
            &fixture.primary_challenges[0],
            &fixture.primary_challenges[1],
        ],
    )
    .unwrap();
    assert_eq!(json!(accepted.values()), expected["y"]);
    assert_eq!(accepted.values(), fixture.direct(0));
    assert_eq!(json!(checks), expected["checks"]);
    assert!(!client.has_live_window());

    let (batch, requests) = testing::issue_batch_fixed(
        &mut client,
        [&fixture.inputs[0], &fixture.inputs[1]],
        [&fixture.noise[0], &fixture.noise[1]],
    )
    .unwrap();
    assert_eq!(
        digest(&[requests[0].view().values, requests[1].view().values]),
        batch_expected["masked_inputs_sha256"].as_str().unwrap()
    );
    let replies = [
        server.evaluate(requests[0].view()).unwrap(),
        server.evaluate(requests[1].view()).unwrap(),
    ];
    assert_eq!(
        digest(&[replies[0].view().values, replies[1].view().values]),
        batch_expected["masked_outputs_sha256"].as_str().unwrap()
    );
    let (fixed, challenge) = testing::begin_fixed(
        &mut client,
        batch,
        &[replies[0].view(), replies[1].view()],
        [
            &fixture.primary_challenges[0],
            &fixture.primary_challenges[1],
        ],
    )
    .unwrap();
    let auxiliary = server.project(challenge.view()).unwrap();
    assert_eq!(
        digest(&matrix_rows(auxiliary.view().values, 128)),
        batch_expected["auxiliary_responses_sha256"]
            .as_str()
            .unwrap()
    );
    let (accepted, secondary, original) = testing::finish_batch_fixed(
        &mut client,
        fixed,
        auxiliary.view(),
        [
            &fixture.secondary_challenges[0],
            &fixture.secondary_challenges[1],
        ],
    )
    .unwrap();
    assert_eq!(json!(secondary), batch_expected["secondary_checks"]);
    assert_eq!(json!(original), batch_expected["original_checks"]);
    for b in 0..2 {
        assert_eq!(accepted.outputs()[b].values(), fixture.direct(b));
    }
    assert!(!client.has_live_window());
    assert!(client.memory_stats().charged_peak_bytes <= ALLOCATION_CEILING);
}

#[test]
fn literal_raa_matrix_oracle_checks_every_basis_in_both_directions() {
    let c = FixtureCode::new(3, 1);
    let n = c.dimension;
    let length = n * 8;
    // Direct expansion of R Pi1 V1 A Pi2 V2 A. No native encoder calls.
    let mut g = vec![vec![0_u32; length]; n];
    for (i, row) in g.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            let mut integer = 0_u128;
            for k in 0..=j {
                for h in 0..=c.pi2[k] as usize {
                    if c.pi1[h] as usize / 8 == i {
                        integer += u128::from(c.v1[h]) * u128::from(c.v2[k]);
                    }
                }
            }
            *value = (integer % u128::from(FIELD)) as u32;
        }
    }
    for i in 0..n {
        let mut basis = vec![0; n];
        basis[i] = 1;
        assert_eq!(testing::encode(c.description(), &basis).unwrap(), g[i]);
    }
    for j in 0..length {
        assert_eq!(
            testing::syndrome(c.description(), &[j as u32, 1]).unwrap(),
            (0..n).map(|i| g[i][j]).collect::<Vec<_>>()
        );
    }
    let input = [FIELD - 1, 257, FIELD / 2];
    let expected: Vec<u32> = (0..length)
        .map(|j| raw_dot(&[g[0][j], g[1][j], g[2][j]], &input))
        .collect();
    assert_eq!(testing::encode(c.description(), &input).unwrap(), expected);
    let mut bad = c.description();
    bad.field = 2_147_483_659;
    expect_error(testing::encode(bad, &input), Error::Field);
}

#[test]
fn complete_publication_validation_rejects_last_entry_tampering_and_wrong_identity() {
    let fixture = Fixture::new();
    let draft = preprocess(AllocationPlan::bounded().unwrap(), fixture.source()).unwrap();
    let view = draft.view();
    for kind in 0..3 {
        let original = match kind {
            0 => view.derived.p[1],
            1 => view.derived.q[1],
            _ => view.derived.q_auxiliary,
        };
        let mut changed = original.to_vec();
        let last = changed.len() - 1;
        changed[last] = (changed[last] + 1) % FIELD;
        let mut parts = view.derived;
        match kind {
            0 => parts.p[1] = &changed,
            1 => parts.q[1] = &changed,
            _ => parts.q_auxiliary = &changed,
        }
        let imported = UnvalidatedPublication::import(
            AllocationPlan::bounded().unwrap(),
            fixture.source(),
            parts,
        )
        .unwrap();
        expect_error(imported.validate(fixture.source()), Error::Publication);
    }
    let mut changed = Fixture::new();
    changed.matrices[0][0] += 1;
    expect_error(draft.validate(changed.source()), Error::Binding);
    let draft = preprocess(AllocationPlan::bounded().unwrap(), fixture.source()).unwrap();
    changed = Fixture::new();
    changed.primary.pi1.swap(0, 1);
    expect_error(draft.validate(changed.source()), Error::Binding);
    changed.privacy[0].pi1[0] = changed.privacy[0].pi1[1];
    expect_error(
        preprocess(AllocationPlan::bounded().unwrap(), changed.source()),
        Error::Code,
    );
}

#[test]
fn secure_sampling_roundtrips_full_field_inputs_and_does_not_reuse_masks() {
    let fixture = Fixture::new();
    let (mut client, mut server) = fixture.roles().unwrap();
    let mut masks = BTreeSet::new();
    for slot in [0, 0, 1, 1] {
        let (query, request) = client.issue_query(slot, &fixture.inputs[slot]).unwrap();
        assert!(masks.insert(request.view().values.to_vec()));
        let reply = server.evaluate(request.view()).unwrap();
        assert_eq!(
            client.finish_query(query, reply.view()).unwrap().values(),
            fixture.direct(slot)
        );
    }
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let replies = [
        server.evaluate(requests[0].view()).unwrap(),
        server.evaluate(requests[1].view()).unwrap(),
    ];
    let (fixed, primary) = client
        .begin_verification(batch, &[replies[0].view(), replies[1].view()])
        .unwrap();
    let auxiliary = server.project(primary.view()).unwrap();
    let accepted = client.finish_batch(fixed, auxiliary.view()).unwrap();
    for b in 0..2 {
        assert_eq!(accepted.outputs()[b].values(), fixture.direct(b));
    }
}

#[test]
fn exact_noise_support_sampler_and_rejection_domain_are_checked() {
    let mut supports = BTreeMap::new();
    for first in 0..4 {
        for second in 0..3 {
            // draw support, coefficient, support, coefficient
            let sampled = testing::sampled_pairs(4, 2, Some(&[first, 0, second, 0])).unwrap();
            let mut support = [sampled[0], sampled[2]];
            support.sort();
            *supports.entry(support).or_insert(0) += 1;
            assert_eq!([sampled[1], sampled[3]], [1, 1]);
        }
    }
    assert_eq!(supports.len(), 6);
    assert!(supports.values().all(|&count| count == 2));
    assert_eq!(
        testing::bounded_draw(5, &[u32::MAX, u32::MAX - 1]).unwrap(),
        4
    );
    expect_error(testing::bounded_draw(5, &[u32::MAX]), Error::Randomness);
    for _ in 0..8 {
        let pairs = testing::sampled_pairs(512, 165, None).unwrap();
        let positions: BTreeSet<_> = pairs.chunks_exact(2).map(|p| p[0]).collect();
        assert_eq!(positions.len(), 165);
        assert!(pairs
            .chunks_exact(2)
            .all(|p| p[0] < 512 && p[1] > 0 && p[1] < FIELD));
    }
    expect_error(testing::sampled_pairs(16, 17, None), Error::Shape);
}

#[test]
fn forged_original_outputs_fail_in_protocol3_and_protocol2() {
    let fixture = Fixture::new();
    let (mut client, mut server) = fixture.roles().unwrap();
    for position in [0, 31, 63] {
        let (query, request) = client.issue_query(0, &fixture.inputs[0]).unwrap();
        let reply = server.evaluate(request.view()).unwrap();
        let mut bad = reply.view().values.to_vec();
        bad[position] = (bad[position] + 1) % FIELD;
        expect_error(
            testing::finish_query_fixed(
                &mut client,
                query,
                ReplyView {
                    header: reply.view().header,
                    values: &bad,
                },
                [
                    &fixture.primary_challenges[0],
                    &fixture.primary_challenges[1],
                ],
            ),
            Error::Rejected,
        );
        assert!(!client.has_live_window());
    }
    for (slot, position) in [(0, 0), (0, 63), (1, 31), (1, 63)] {
        let (batch, requests) = client
            .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
            .unwrap();
        let replies = [
            server.evaluate(requests[0].view()).unwrap(),
            server.evaluate(requests[1].view()).unwrap(),
        ];
        let mut bad = replies[slot].view().values.to_vec();
        bad[position] = (bad[position] + 1) % FIELD;
        let mut views = [replies[0].view(), replies[1].view()];
        views[slot].values = &bad;
        let (fixed, primary) = testing::begin_fixed(
            &mut client,
            batch,
            &views,
            [
                &fixture.primary_challenges[0],
                &fixture.primary_challenges[1],
            ],
        )
        .unwrap();
        let auxiliary = server.project(primary.view()).unwrap();
        expect_error(
            testing::finish_batch_fixed(
                &mut client,
                fixed,
                auxiliary.view(),
                [
                    &fixture.secondary_challenges[0],
                    &fixture.secondary_challenges[1],
                ],
            ),
            Error::Rejected,
        );
        assert!(!client.has_live_window());
    }
}

#[test]
fn auxiliary_lie_orthogonal_to_original_input_requires_secondary_check() {
    let fixture = Fixture::new();
    let (mut client, mut server) = fixture.roles().unwrap();
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let x = requests[0].view().values;
    let replies = [
        server.evaluate(requests[0].view()).unwrap(),
        server.evaluate(requests[1].view()).unwrap(),
    ];
    let (fixed, primary) = testing::begin_fixed(
        &mut client,
        batch,
        &[replies[0].view(), replies[1].view()],
        [
            &fixture.primary_challenges[0],
            &fixture.primary_challenges[1],
        ],
    )
    .unwrap();
    let auxiliary = server.project(primary.view()).unwrap();
    let mut bad = auxiliary.view().values.to_vec();
    bad[0] = ((u64::from(bad[0]) + u64::from(x[1])) % u64::from(FIELD)) as u32;
    bad[1] = ((u64::from(bad[1]) + u64::from(FIELD) - u64::from(x[0])) % u64::from(FIELD)) as u32;
    assert_ne!(bad, auxiliary.view().values);
    assert_eq!(
        raw_dot(&bad[..64], x),
        raw_dot(&auxiliary.view().values[..64], x)
    );
    expect_error(
        testing::finish_batch_fixed(
            &mut client,
            fixed,
            AuxiliaryReplyView {
                header: auxiliary.view().header,
                values: &bad,
            },
            [
                &fixture.secondary_challenges[0],
                &fixture.secondary_challenges[1],
            ],
        ),
        Error::Rejected,
    );
    assert!(!client.has_live_window());
}

#[test]
fn fixation_owns_original_bytes_and_server_rejects_replays_and_early_projection() {
    let fixture = Fixture::new();
    let (mut client, mut server) = fixture.roles().unwrap();
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let mut header = requests[0].view().header;
    header.slot = u8::MAX;
    expect_error(
        server.project(PrimaryChallengeView {
            header,
            entries: [
                &fixture.primary_challenges[0],
                &fixture.primary_challenges[1],
            ],
        }),
        Error::Replay,
    );
    let r0 = server.evaluate(requests[0].view()).unwrap();
    expect_error(server.evaluate(requests[0].view()), Error::Replay);
    let r1 = server.evaluate(requests[1].view()).unwrap();
    let mut transport0 = r0.view().values.to_vec();
    let (fixed, primary) = client
        .begin_verification(
            batch,
            &[
                ReplyView {
                    header: r0.view().header,
                    values: &transport0,
                },
                r1.view(),
            ],
        )
        .unwrap();
    transport0.fill(FIELD); // No alias to the client's frozen, canonical copy.
    let auxiliary = server.project(primary.view()).unwrap();
    expect_error(server.project(primary.view()), Error::Replay);
    let accepted = client.finish_batch(fixed, auxiliary.view()).unwrap();
    for b in 0..2 {
        assert_eq!(accepted.outputs()[b].values(), fixture.direct(b));
    }

    // A burned malformed request is not a successfully evaluated original.
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let bad = [FIELD; 64];
    expect_error(
        server.evaluate(RequestView {
            header: requests[0].view().header,
            values: &bad,
        }),
        Error::Field,
    );
    let _second = server.evaluate(requests[1].view()).unwrap();
    let mut header = requests[0].view().header;
    header.slot = u8::MAX;
    expect_error(
        server.project(PrimaryChallengeView {
            header,
            entries: [
                &fixture.primary_challenges[0],
                &fixture.primary_challenges[1],
            ],
        }),
        Error::Replay,
    );
    drop(batch);
    assert!(!client.has_live_window());
}

#[test]
fn malformed_wrong_binding_and_omitted_replies_consume_private_owners() {
    let fixture = Fixture::new();
    let (mut client, mut server) = fixture.roles().unwrap();
    for bad_case in 0..4 {
        let (query, request) = client.issue_query(0, &fixture.inputs[0]).unwrap();
        let reply = server.evaluate(request.view()).unwrap();
        let mut header = reply.view().header;
        let mut values = reply.view().values.to_vec();
        let error = match bad_case {
            0 => {
                values.clear();
                Error::Shape
            }
            1 => {
                values.push(0);
                Error::Shape
            }
            2 => {
                values[0] = FIELD;
                Error::Field
            }
            _ => {
                header.publication[0] ^= 1;
                Error::Binding
            }
        };
        expect_error(
            client.finish_query(
                query,
                ReplyView {
                    header,
                    values: &values,
                },
            ),
            error,
        );
        assert!(!client.has_live_window());
    }
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let reply = server.evaluate(requests[0].view()).unwrap();
    expect_error(
        client.begin_verification(batch, &[reply.view()]),
        Error::Shape,
    );
    assert!(!client.has_live_window());
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let reply = server.evaluate(requests[0].view()).unwrap();
    expect_error(
        client.begin_verification(batch, &[reply.view(), reply.view()]),
        Error::Binding,
    );
    assert!(!client.has_live_window());
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let replies = [
        server.evaluate(requests[0].view()).unwrap(),
        server.evaluate(requests[1].view()).unwrap(),
    ];
    let (fixed, primary) = client
        .begin_verification(batch, &[replies[0].view(), replies[1].view()])
        .unwrap();
    let auxiliary = server.project(primary.view()).unwrap();
    expect_error(
        client.finish_batch(
            fixed,
            AuxiliaryReplyView {
                header: auxiliary.view().header,
                values: &[],
            },
        ),
        Error::Shape,
    );
    assert!(!client.has_live_window());
}

#[test]
fn cancellation_pressure_and_attempt_budget_do_not_recycle_private_material() {
    let fixture = Fixture::new();
    let (mut client, mut server) = fixture.roles().unwrap();
    let (query, request) = client.issue_query(0, &fixture.inputs[0]).unwrap();
    expect_error(client.issue_query(0, &fixture.inputs[0]), Error::Busy);
    assert!(client.has_live_window());
    drop(query);
    assert!(!client.has_live_window());
    drop(request);
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let replies = [
        server.evaluate(requests[0].view()).unwrap(),
        server.evaluate(requests[1].view()).unwrap(),
    ];
    let (fixed, _challenge) = client
        .begin_verification(batch, &[replies[0].view(), replies[1].view()])
        .unwrap();
    drop(fixed);
    assert!(!client.has_live_window());
    while client.attempts_issued() < MAX_ATTEMPTS {
        expect_error(client.issue_query(0, &[]), Error::Shape);
    }
    expect_error(client.issue_query(0, &fixture.inputs[0]), Error::Attempts);
    assert!(!client.has_live_window());
}

#[test]
fn identical_algebra_does_not_allow_cross_session_query_or_response_transfer() {
    let fixture = Fixture::new();
    let (mut first, mut server_first) = fixture.roles().unwrap();
    let (mut second, mut server_second) = fixture.roles().unwrap();
    // Identical deterministic test masks make both field relations identical.
    // Only explicit session binding can reject cross-session substitution here.
    let (q1, r1) =
        testing::issue_fixed(&mut first, 0, &fixture.inputs[0], &fixture.noise[0]).unwrap();
    let (q2, r2) =
        testing::issue_fixed(&mut second, 0, &fixture.inputs[0], &fixture.noise[0]).unwrap();
    assert_eq!(r1.view().header.publication, r2.view().header.publication);
    assert_eq!(r1.view().values, r2.view().values);
    assert_ne!(r1.view().header.session, r2.view().header.session);
    expect_error(server_second.evaluate(r1.view()), Error::Binding);
    let y1 = server_first.evaluate(r1.view()).unwrap();
    let y2 = server_second.evaluate(r2.view()).unwrap();
    assert_eq!(y1.view().values, y2.view().values);
    expect_error(first.finish_query(q1, y2.view()), Error::Binding);
    expect_error(second.finish_query(q2, y1.view()), Error::Binding);
    assert!(!first.has_live_window());
    assert!(!second.has_live_window());
}

#[test]
fn entropy_failure_at_all_secret_sampling_phases_retires_the_owner() {
    expect_error(testing::partial_entropy_failure(), Error::Randomness);
    let fixture = Fixture::new();
    let (mut client, mut server) = fixture.roles().unwrap();
    expect_error(
        testing::issue_rng_failure(&mut client, &fixture.inputs[0]),
        Error::Randomness,
    );
    assert!(!client.has_live_window());
    let (query, request) = client.issue_query(0, &fixture.inputs[0]).unwrap();
    let reply = server.evaluate(request.view()).unwrap();
    expect_error(
        testing::finish_query_rng_failure(&mut client, query, reply.view()),
        Error::Randomness,
    );
    assert!(!client.has_live_window());
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let replies = [
        server.evaluate(requests[0].view()).unwrap(),
        server.evaluate(requests[1].view()).unwrap(),
    ];
    expect_error(
        testing::begin_rng_failure(&mut client, batch, &[replies[0].view(), replies[1].view()]),
        Error::Randomness,
    );
    assert!(!client.has_live_window());
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let replies = [
        server.evaluate(requests[0].view()).unwrap(),
        server.evaluate(requests[1].view()).unwrap(),
    ];
    let (fixed, primary) = client
        .begin_verification(batch, &[replies[0].view(), replies[1].view()])
        .unwrap();
    let auxiliary = server.project(primary.view()).unwrap();
    expect_error(
        testing::finish_batch_rng_failure(&mut client, fixed, auxiliary.view()),
        Error::Randomness,
    );
    assert!(!client.has_live_window());
}
