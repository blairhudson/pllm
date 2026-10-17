use num_bigint::BigInt;

use super::*;
use crate::gkm::{
    encode_input,
    tests::{field_bytes, integer_oracle, scalar_order, ScriptEntropy},
    Stage,
};

const A: [u8; 32] = [1; 32];
const B: [u8; 32] = [2; 32];
const CONTEXT: [u8; 32] = [7; 32];

pub(super) fn artifacts() -> (I16Matrix, I16Matrix, FloatTable) {
    let a = I16Matrix::from_encoded(
        3,
        3,
        &[257, -129, 2048, -300, 17, 2, 0, 0, 0],
        &[100_000; 3],
    )
    .unwrap();
    let b = I16Matrix::from_encoded(2, 3, &[-2048, 256, 1, 400, -20, 8192], &[100_000; 3]).unwrap();
    let bytes = (0..33)
        .flat_map(|i| {
            let value = if i == 0 {
                -0.0_f32
            } else {
                (i as f32 - 17.0) / 16.0
            };
            value.to_le_bytes()
        })
        .collect::<Vec<_>>();
    (a, b, FloatTable::from_le_bytes(11, 3, &bytes).unwrap())
}

pub(super) fn publication(a: &I16Matrix, b: &I16Matrix, table: &FloatTable) -> Publication {
    Publication::publish_with(
        [42; 32],
        &[Stage { id: A, matrix: a }, Stage { id: B, matrix: b }],
        table,
        &mut ScriptEntropy::new(&[2, 3, 5]),
    )
    .unwrap()
}

pub(super) fn client(publication: Publication) -> Client {
    let digest = publication.digest();
    Client::new(Arc::new(publication), digest).unwrap()
}

pub(super) fn run_linear(window: Window, matrix: &I16Matrix, input: &[i64]) -> Window {
    let pending = window.linear(input).unwrap();
    let left = matrix.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(left).unwrap();
    let right = matrix.evaluate(&half.remaining_share());
    half.freeze_second(right).unwrap()
}

pub(super) fn run_lookup(window: Window, table: &FloatTable, token: usize) -> Window {
    let pending = window.lookup(token).unwrap();
    let left = table.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(left).unwrap();
    let right = table.evaluate(&half.remaining_share());
    half.freeze_second(right).unwrap()
}

fn error<T>(result: Result<T, Error>, expected: Error) {
    assert_eq!(result.err(), Some(expected));
}

#[test]
fn lookup_repeated_matrix_and_different_width_stage_accept_only_complete_window() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let plan = [
        Step::Lookup,
        Step::Linear(A),
        Step::Linear(A),
        Step::Linear(B),
    ];
    let mut prior_request = None;
    for token in [0, 10, 4, 0] {
        let mut window = run_lookup(client.begin(CONTEXT, &plan).unwrap(), &table, token);
        let request = window.request_id();
        assert_ne!(prior_request, Some(request));
        prior_request = Some(request);
        let bytes = window
            .tentative_values()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(bytes, table.bytes[token * 12..(token + 1) * 12]);
        let mut oracle_values = table.bytes[token * 12..(token + 1) * 12]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for matrix in [&a, &a, &b] {
            // Small, explicit trusted-client local operation; no model inference.
            let inputs = window
                .tentative_values()
                .iter()
                .map(|&v| encode_input(f64::from(v * 2.0 + 0.5)).unwrap())
                .collect::<Vec<_>>();
            // Oracle encoding uses exact dyadic fixture values; the input is
            // comfortably away from problematic integer range boundaries.
            let oracle_input = oracle_values
                .iter()
                .map(|&v| (f64::from(v * 2.0 + 0.5) * 2048.0).round_ties_even() as i64)
                .collect::<Vec<_>>();
            expected = integer_oracle(&matrix.weights, &oracle_input);
            oracle_values = expected
                .iter()
                .map(|&v| (v as f32) * 2.0_f32.powi(-22))
                .collect();
            window = run_linear(window, matrix, &inputs);
            assert_eq!(window.tentative_signed(), expected);
            assert_eq!(window.tentative_values(), oracle_values);
        }
        let accepted = window.finish().unwrap();
        assert_eq!(accepted.request_id(), request);
        assert_eq!(accepted.steps(), plan.len());
        assert_eq!(accepted.context(), CONTEXT);
        assert_eq!(accepted.publication(), client.publication.digest());
        assert_eq!(accepted.signed_linear_output(), expected);
        assert_eq!(accepted.values(), oracle_values);
        assert_eq!(client.budget.live.load(Ordering::Acquire), 1);
        drop(accepted);
        assert_eq!(client.budget.live.load(Ordering::Acquire), 0);
    }
}

#[test]
fn individual_wrapped_worker_relations_fail_but_reconstructed_signed_check_passes() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let window = client.begin(CONTEXT, &[Step::Linear(A)]).unwrap();
    let pending = window
        .linear_with(&[-7, 11, -13], &mut ScriptEntropy::new(&[]))
        .unwrap();
    let first = a.evaluate(&pending.share(Worker::Zero)).unwrap();
    let published = &client.publication.matrices[0];
    let naive_left = published
        .s
        .iter()
        .zip(pending.shares[0].iter())
        .fold(ProjectivePoint::IDENTITY, |sum, (s, &x)| {
            sum + *s * signed_scalar(x as i64)
        });
    let naive_right = client
        .publication
        .r
        .iter()
        .zip(first.values.iter())
        .fold(ProjectivePoint::IDENTITY, |sum, (r, &y)| {
            sum + *r * signed_scalar(y as i64)
        });
    assert_ne!(naive_left, naive_right);
    let half = pending.freeze_first(Ok(first)).unwrap();
    let second = a.evaluate(&half.remaining_share());
    let result = half.freeze_second(second).unwrap().finish().unwrap();
    assert_eq!(
        result.signed_linear_output(),
        integer_oracle(&a.weights, &[-7, 11, -13])
    );
}

#[test]
fn every_output_coordinate_and_either_or_both_workers_reject_corruption() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    for (id, matrix) in [(A, &a), (B, &b)] {
        for row in 0..matrix.rows {
            for worker_set in 1..=3 {
                let pending = client
                    .begin(CONTEXT, &[Step::Linear(id)])
                    .unwrap()
                    .linear(&[3, -7, 11])
                    .unwrap();
                let mut left = matrix.evaluate(&pending.share(Worker::Zero)).unwrap();
                let mut right = matrix.evaluate(&pending.share(Worker::One)).unwrap();
                if worker_set & 1 != 0 {
                    left.values[row] = left.values[row].wrapping_add(1);
                }
                if worker_set & 2 != 0 {
                    right.values[row] = right.values[row].wrapping_add(2);
                }
                let result = pending
                    .freeze_first(Ok(left))
                    .unwrap()
                    .freeze_second(Ok(right));
                match result {
                    // The zero row's public bound rejects before batching.
                    Err(e) => assert_eq!(e, Error::Bound),
                    Ok(window) => error(window.finish(), Error::Verification),
                }
                assert_eq!(client.budget.live.load(Ordering::Acquire), 0);
            }
        }
    }
    // A later honest request cannot observe or accept failed tentative state.
    assert!(run_linear(
        client.begin(CONTEXT, &[Step::Linear(A)]).unwrap(),
        &a,
        &[3, -7, 11]
    )
    .finish()
    .is_ok());
}

#[test]
fn fresh_per_claim_coefficients_catch_coefficient_one_cancellation() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let mut window = client
        .begin(CONTEXT, &[Step::Linear(A), Step::Linear(A)])
        .unwrap();
    let mut rng = ScriptEntropy::new(&[3, 5, 7]);
    for delta in [1i64, -1] {
        let pending = window.linear(&[1, 2, 3]).unwrap();
        let mut left = a.evaluate(&pending.share(Worker::Zero)).unwrap();
        let right = a.evaluate(&pending.share(Worker::One));
        left.values[0] = left.values[0].wrapping_add(delta as u64);
        window = pending
            .freeze_first(Ok(left))
            .unwrap()
            .freeze_second_with(right, &mut rng)
            .unwrap();
    }
    assert_eq!(1 + (-1), 0); // fixed coefficient-one batching loses the error
    assert_ne!(3 - 5, 0);
    error(window.finish_with(&mut rng), Error::Verification);
    assert_eq!(
        rng.calls,
        [
            Purpose::ClaimChallenge,
            Purpose::ClaimChallenge,
            Purpose::MatrixChallenge
        ]
    );
}

#[test]
fn final_fresh_coefficients_catch_errors_cancelling_across_different_matrices() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let mut window = client
        .begin(CONTEXT, &[Step::Linear(A), Step::Linear(B)])
        .unwrap();
    // Equal inner coefficients deliberately create cancellation between groups;
    // independent final coefficients must still bind both fixed relations.
    let mut rng = ScriptEntropy::new(&[1, 1, 3, 5]);
    for (matrix, delta) in [(&a, 1i64), (&b, -1)] {
        let pending = window.linear(&[1, 2, 3]).unwrap();
        let mut left = matrix.evaluate(&pending.share(Worker::Zero)).unwrap();
        left.values[0] = left.values[0].wrapping_add(delta as u64);
        let half = pending.freeze_first(Ok(left)).unwrap();
        let right = matrix.evaluate(&half.remaining_share());
        window = half.freeze_second_with(right, &mut rng).unwrap();
    }
    error(window.finish_with(&mut rng), Error::Verification);
    assert_eq!(
        rng.calls,
        [
            Purpose::ClaimChallenge,
            Purpose::ClaimChallenge,
            Purpose::MatrixChallenge,
            Purpose::MatrixChallenge
        ]
    );
}

#[test]
fn accumulation_is_mod_scalar_order_and_matches_independent_big_integer_oracle() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let mut window = client
        .begin(CONTEXT, &[Step::Linear(A), Step::Linear(A)])
        .unwrap();
    let minus_one: [u8; 32] = (scalar_order() - 1u8).to_bytes_be().try_into().unwrap();
    let mut rng = ScriptEntropy::new(&[]);
    rng.scalars.extend([minus_one, minus_one, minus_one]);
    let inputs = [[-7, 11, -13], [5, -17, 3]];
    let mut expected_x = vec![BigInt::from(0); 3];
    let mut expected_y = vec![BigInt::from(0); 3];
    for input in inputs {
        let pending = window.linear(&input).unwrap();
        let first = a.evaluate(&pending.share(Worker::Zero));
        let half = pending.freeze_first(first).unwrap();
        let second = a.evaluate(&half.remaining_share());
        window = half.freeze_second_with(second, &mut rng).unwrap();
        for (sum, x) in expected_x.iter_mut().zip(input) {
            *sum -= BigInt::from(x);
        }
        for (sum, y) in expected_y
            .iter_mut()
            .zip(integer_oracle(&a.weights, &input))
        {
            *sum -= BigInt::from(y);
        }
        for (actual, expected) in window.accumulators[0].x.iter().zip(&expected_x) {
            assert_eq!(actual.to_bytes().as_slice(), field_bytes(expected.clone()));
        }
        for (actual, expected) in window.accumulators[0].y.iter().zip(&expected_y) {
            assert_eq!(actual.to_bytes().as_slice(), field_bytes(expected.clone()));
        }
    }
    assert!(window.finish_with(&mut rng).is_ok());
}

#[test]
fn malformed_role_shape_context_stage_count_and_replay_retire_the_owner() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let plan = [Step::Linear(A)];
    let old = client
        .begin(CONTEXT, &plan)
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let replay = a.evaluate(&old.share(Worker::Zero));
    drop(old);
    let new = client
        .begin(CONTEXT, &plan)
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    error(new.freeze_first(replay), Error::Binding);
    for tamper in 0..10 {
        let pending = client
            .begin(CONTEXT, &plan)
            .unwrap()
            .linear(&[1, 2, 3])
            .unwrap();
        let mut reply = a.evaluate(&pending.share(Worker::Zero)).unwrap();
        match tamper {
            0 => reply.binding.context[0] ^= 1,
            1 => reply.binding.publication[0] ^= 1,
            2 => reply.binding.session[0] ^= 1,
            3 => reply.binding.plan[0] ^= 1,
            4 => reply.binding.ordinal += 1,
            5 => reply.binding.total += 1,
            6 => reply.binding.step = Step::Linear(B),
            7 => reply.binding.artifact[0] ^= 1,
            8 => reply.binding.worker = Worker::One,
            9 => reply.binding.sequence += 1,
            _ => unreachable!(),
        }
        error(pending.freeze_first(Ok(reply)), Error::Binding);
    }
    let pending = client
        .begin(CONTEXT, &plan)
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let duplicate = a.evaluate(&pending.share(Worker::Zero));
    let first = a.evaluate(&pending.share(Worker::Zero));
    error(first, Error::Retired);
    error(pending.freeze_first(duplicate), Error::Retired);
    let pending = client
        .begin(CONTEXT, &plan)
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let share = pending.share(Worker::Zero);
    let wrong = LinearResponse::from_untrusted(&share, share.binding(), &[]);
    error(pending.freeze_first(wrong), Error::Shape);
    let pending = client
        .begin(CONTEXT, &plan)
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let huge = LinearResponse::from_untrusted(
        &pending.share(Worker::Zero),
        pending.share(Worker::Zero).binding(),
        &[0; MAX_DIMENSION + 1],
    );
    error(pending.freeze_first(huge), Error::Limit);
    let pending = client
        .begin(CONTEXT, &plan)
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let wrong_artifact = b.evaluate(&pending.share(Worker::Zero));
    error(pending.freeze_first(wrong_artifact), Error::Binding);
    assert_eq!(client.budget.live.load(Ordering::Acquire), 0);
}

#[test]
fn cross_session_and_same_shape_wrong_matrix_artifacts_fail_closed() {
    let (a, b, table) = artifacts();
    let publication = Arc::new(publication(&a, &b, &table));
    let one = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
    let two = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
    let pending = one
        .begin(CONTEXT, &[Step::Linear(A)])
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let reply = a.evaluate(&pending.share(Worker::Zero));
    drop(pending);
    let pending = two
        .begin(CONTEXT, &[Step::Linear(A)])
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    error(pending.freeze_first(reply), Error::Binding);
    let mut weights = a.weights.to_vec();
    weights[0] += 1;
    let wrong = I16Matrix::from_encoded(3, 3, &weights, &[100_000; 3]).unwrap();
    let pending = one
        .begin(CONTEXT, &[Step::Linear(A)])
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let reply = wrong.evaluate(&pending.share(Worker::Zero));
    error(pending.freeze_first(reply), Error::Binding);
}

#[test]
fn lookup_checks_all_row_bytes_before_use_and_retires_partial_or_bad_responses() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    for offset in 0..12 {
        for worker in [Worker::Zero, Worker::One] {
            let pending = client
                .begin(CONTEXT, &[Step::Lookup])
                .unwrap()
                .lookup(10)
                .unwrap();
            let mut left = table.evaluate(&pending.share(Worker::Zero)).unwrap();
            let mut right = table.evaluate(&pending.share(Worker::One)).unwrap();
            if worker == Worker::Zero {
                left.bytes[offset] ^= 1;
            } else {
                right.bytes[offset] ^= 1;
            }
            error(
                pending
                    .freeze_first(Ok(left))
                    .unwrap()
                    .freeze_second(Ok(right)),
                Error::Verification,
            );
        }
    }
    let pending = client
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    let wrong_role = table.evaluate(&pending.share(Worker::One));
    error(pending.freeze_first(wrong_role), Error::Binding);
    let pending = client
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    let share = pending.share(Worker::Zero);
    let malformed = LookupResponse::from_untrusted(&share, share.binding(), &[0; 3]);
    error(pending.freeze_first(malformed), Error::Shape);
    let pending = client
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    let reply = table.evaluate(&pending.share(Worker::Zero));
    drop(pending.freeze_first(reply).unwrap());
    let pending = client
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    let stale = table.evaluate(&pending.share(Worker::Zero));
    drop(pending);
    let pending = client
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    error(pending.freeze_first(stale), Error::Binding);
    let pending = client
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    let reply = table.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(reply).unwrap();
    let oversized = LookupResponse::from_untrusted(
        &half.remaining_share(),
        half.remaining_share().binding(),
        &[0; 4 * MAX_DIMENSION + 1],
    );
    error(half.freeze_second(oversized), Error::Limit);
    assert_eq!(client.budget.live.load(Ordering::Acquire), 0);
    let values = run_lookup(client.begin(CONTEXT, &[Step::Lookup]).unwrap(), &table, 0)
        .finish()
        .unwrap();
    assert_eq!(values.values()[0].to_bits(), (-0.0_f32).to_bits());
}

#[test]
fn publication_pins_cover_points_row_hashes_context_numeric_matrix_and_bounds() {
    let (a, b, table) = artifacts();
    for tamper in 0..7 {
        let mut publication = publication(&a, &b, &table);
        let pin = publication.digest();
        match tamper {
            0 => publication.r[0] += ProjectivePoint::GENERATOR,
            1 => publication.matrices[0].s[0] += ProjectivePoint::GENERATOR,
            2 => publication.row_hashes[0][0] ^= 1,
            3 => publication.context[0] ^= 1,
            4 => publication.matrices[0].digest[0] ^= 1,
            5 => publication.matrices[0].input_bounds[0] += 1,
            6 => publication.table_digest[0] ^= 1,
            _ => unreachable!(),
        }
        assert_ne!(publication.compute_digest(), pin);
        error(Client::new(Arc::new(publication), pin), Error::Binding);
    }
    // Even a locally trusted but faulty publication must not turn a mismatched
    // row or group equation into an accepted output.
    let mut wrong = publication(&a, &b, &table);
    wrong.row_hashes[0][0] ^= 1;
    wrong.digest = wrong.compute_digest();
    let owner = client(wrong);
    let pending = owner
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    let left = table.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(left).unwrap();
    let right = table.evaluate(&half.remaining_share());
    error(half.freeze_second(right), Error::Verification);
    let mut wrong = publication(&a, &b, &table);
    wrong.matrices[0].s[0] += ProjectivePoint::GENERATOR;
    wrong.digest = wrong.compute_digest();
    let owner = client(wrong);
    error(
        run_linear(
            owner.begin(CONTEXT, &[Step::Linear(A)]).unwrap(),
            &a,
            &[1, 2, 3],
        )
        .finish(),
        Error::Verification,
    );
}

#[test]
fn freeze_barrier_precedes_entropy_and_all_entropy_failure_phases_retire() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let pending = client
        .begin(CONTEXT, &[Step::Linear(A)])
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let left = a.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(left).unwrap();
    let share = half.remaining_share();
    let malformed = LinearResponse::from_untrusted(&share, share.binding(), &[]);
    let mut rng = ScriptEntropy::new(&[]);
    error(half.freeze_second_with(malformed, &mut rng), Error::Shape);
    assert!(rng.calls.is_empty());
    for purpose in [
        Purpose::LinearMask,
        Purpose::LookupMask,
        Purpose::ClaimChallenge,
        Purpose::MatrixChallenge,
    ] {
        let mut rng = ScriptEntropy::new(&[]);
        rng.fail = Some(purpose);
        match purpose {
            Purpose::LinearMask => error(
                client
                    .begin(CONTEXT, &[Step::Linear(A)])
                    .unwrap()
                    .linear_with(&[1, 2, 3], &mut rng),
                Error::Entropy,
            ),
            Purpose::LookupMask => error(
                client
                    .begin(CONTEXT, &[Step::Lookup])
                    .unwrap()
                    .lookup_with(0, &mut rng),
                Error::Entropy,
            ),
            Purpose::ClaimChallenge => {
                let pending = client
                    .begin(CONTEXT, &[Step::Linear(A)])
                    .unwrap()
                    .linear(&[1, 2, 3])
                    .unwrap();
                let left = a.evaluate(&pending.share(Worker::Zero));
                let half = pending.freeze_first(left).unwrap();
                let right = a.evaluate(&half.remaining_share());
                error(half.freeze_second_with(right, &mut rng), Error::Entropy);
            }
            Purpose::MatrixChallenge => {
                let window = run_linear(
                    client.begin(CONTEXT, &[Step::Linear(A)]).unwrap(),
                    &a,
                    &[1, 2, 3],
                );
                error(window.finish_with(&mut rng), Error::Entropy);
            }
            _ => unreachable!(),
        }
        assert_eq!(client.budget.live.load(Ordering::Acquire), 0);
    }
    let mut rng = ScriptEntropy::new(&[]);
    rng.fail = Some(Purpose::Session);
    error(
        Client::new_with(
            Arc::clone(&client.publication),
            client.publication.digest(),
            &mut rng,
        ),
        Error::Entropy,
    );
    let mut rng = ScriptEntropy::new(&[]);
    rng.fail = Some(Purpose::Owner);
    error(
        Publication::publish_with([0; 32], &[Stage { id: A, matrix: &a }], &table, &mut rng),
        Error::Entropy,
    );
}

#[test]
fn schedules_omitted_claims_input_bounds_and_cancellation_fail_without_receipt() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let plan = [Step::Lookup, Step::Linear(A), Step::Linear(B)];
    error(
        client.begin(CONTEXT, &plan).unwrap().finish(),
        Error::Incomplete,
    );
    error(
        client.begin(CONTEXT, &plan).unwrap().linear(&[0; 3]),
        Error::Schedule,
    );
    error(
        client.begin(CONTEXT, &plan).unwrap().lookup(11),
        Error::Bound,
    );
    error(
        client
            .begin(CONTEXT, &[Step::Linear(A)])
            .unwrap()
            .linear(&[0; 2]),
        Error::Shape,
    );
    error(
        client
            .begin(CONTEXT, &[Step::Linear(A)])
            .unwrap()
            .linear(&[100_001, 0, 0]),
        Error::Bound,
    );
    error(
        client
            .begin(CONTEXT, &[Step::Linear(A)])
            .unwrap()
            .linear(&[i64::MIN, 0, 0]),
        Error::Bound,
    );
    let mut window = run_linear(
        client.begin(CONTEXT, &[Step::Linear(A)]).unwrap(),
        &a,
        &[1, 2, 3],
    );
    window.accumulators[0].calls = 0; // emulate an internal dropped verifier job
    error(window.finish(), Error::Incomplete);
    let window = run_lookup(client.begin(CONTEXT, &plan).unwrap(), &table, 0);
    error(window.finish(), Error::Incomplete);
    let window = run_linear(
        client.begin(CONTEXT, &[Step::Linear(A)]).unwrap(),
        &a,
        &[1, 2, 3],
    );
    error(window.linear(&[1, 2, 3]), Error::Schedule);
    drop(client.begin(CONTEXT, &plan).unwrap());
    drop(
        client
            .begin(CONTEXT, &[Step::Linear(A)])
            .unwrap()
            .linear(&[1, 2, 3])
            .unwrap(),
    );
    let pending = client
        .begin(CONTEXT, &[Step::Linear(A)])
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let left = a.evaluate(&pending.share(Worker::Zero));
    drop(pending.freeze_first(left).unwrap());
    drop(run_linear(
        client.begin(CONTEXT, &[Step::Linear(A)]).unwrap(),
        &a,
        &[1, 2, 3],
    ));
    assert_eq!(client.budget.live.load(Ordering::Acquire), 0);
}

#[test]
fn fresh_private_views_and_live_cumulative_caps_survive_drop_and_bad_attempts() {
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let plan = [Step::Linear(A)];
    error(
        client.begin(CONTEXT, &[Step::Lookup; MAX_WINDOW_STEPS + 1]),
        Error::Limit,
    );
    error(
        client.begin(CONTEXT, &[Step::Linear([99; 32])]),
        Error::Schedule,
    );
    let one = client
        .begin(CONTEXT, &plan)
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let two = client
        .begin(CONTEXT, &plan)
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    for worker in [Worker::Zero, Worker::One] {
        assert_ne!(one.share(worker).values(), two.share(worker).values());
    }
    drop(one);
    drop(two);
    let one = client
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    let two = client
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    // The tiny V=11 domain can legitimately repeat. Check exact one-hot sharing,
    // canonical padding and RNG purpose, not a flaky no-collision assertion.
    for pending in [&one, &two] {
        let a = pending.share(Worker::Zero);
        let b = pending.share(Worker::One);
        assert_eq!(a.selector()[0] ^ b.selector()[0], 1);
        assert_eq!(a.selector()[1] ^ b.selector()[1], 0);
        assert_eq!(a.selector()[1] & 0xf8, 0);
    }
    drop(one);
    drop(two);
    let mut live = (0..MAX_LIVE_WINDOWS)
        .map(|_| client.begin(CONTEXT, &plan).unwrap())
        .collect::<Vec<_>>();
    error(client.begin(CONTEXT, &plan), Error::Limit);
    drop(live.pop());
    drop(client.begin(CONTEXT, &plan).unwrap());
    drop(live);
    assert_eq!(client.budget.live.load(Ordering::Acquire), 0);
    // Reservations are burned for the entire window even if it is cancelled.
    let before = client.budget.steps.load(Ordering::Acquire);
    drop(client.begin(CONTEXT, &[Step::Linear(A); 4]).unwrap());
    assert_eq!(client.budget.steps.load(Ordering::Acquire), before + 4);
    client
        .budget
        .steps
        .store(MAX_RESERVED_STEPS - 1, Ordering::Release);
    drop(client.begin(CONTEXT, &plan).unwrap());
    error(client.begin(CONTEXT, &plan), Error::Limit);
    client
        .budget
        .windows
        .store(MAX_WINDOWS - 1, Ordering::Release);
    error(client.begin(CONTEXT, &[]), Error::Limit);
    error(client.begin(CONTEXT, &plan), Error::Limit);
    let counter = AtomicU64::new(u64::MAX);
    error(reserve(&counter, 1, u64::MAX), Error::Limit);
    assert_eq!(counter.load(Ordering::Acquire), u64::MAX);
    let process_before = PROCESS_WINDOWS.load(Ordering::Acquire);
    let another_client =
        Client::new(Arc::clone(&client.publication), client.publication.digest()).unwrap();
    drop(another_client.begin(CONTEXT, &plan).unwrap());
    assert!(PROCESS_WINDOWS.load(Ordering::Acquire) > process_before);
}
