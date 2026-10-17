use std::sync::Barrier;

use super::tests::{artifacts, client, publication, run_linear, run_lookup};
use super::*;
use crate::gkm::{publication::AdmissionPool, tests::ScriptEntropy, Stage};

const A: [u8; 32] = [1; 32];
const CONTEXT: [u8; 32] = [7; 32];

fn injected<T>(after: usize, action: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
    let _guard = payload::failure::after(after);
    action()
}

/// Fail every successive payload allocation, stopping only when the complete
/// operation succeeds. No assumed count of implementation allocations is used.
fn fail_each<T>(mut operation: impl FnMut(usize) -> Result<T, Error>, retired: impl Fn()) {
    for after in 0..64 {
        match operation(after) {
            Err(Error::Allocation) => retired(),
            Ok(value) => {
                drop(value);
                retired();
                assert!(after > 0);
                return;
            }
            Err(other) => panic!("unexpected failure at payload {after}: {other:?}"),
        }
    }
    panic!("operation did not reach success within bounded allocation sweep");
}

#[test]
fn all_clients_share_publication_window_admission_including_concurrent_attempts() {
    let (a, b, table) = artifacts();
    let publication = Arc::new(publication(&a, &b, &table));
    let clients = (0..9)
        .map(|_| Client::new(Arc::clone(&publication), publication.digest()).unwrap())
        .collect::<Vec<_>>();
    let mut held = clients[..7]
        .iter()
        .map(|c| c.begin(CONTEXT, &[Step::Lookup]).unwrap())
        .collect::<Vec<_>>();
    let gate = Barrier::new(3);
    let results = std::thread::scope(|scope| {
        let one = scope.spawn(|| {
            gate.wait();
            clients[7].begin(CONTEXT, &[Step::Lookup])
        });
        let two = scope.spawn(|| {
            gate.wait();
            clients[8].begin(CONTEXT, &[Step::Lookup])
        });
        gate.wait();
        [one.join().unwrap(), two.join().unwrap()]
    });
    let mut rejected = 0;
    for result in results {
        match result {
            Ok(window) => held.push(window),
            Err(Error::Limit) => rejected += 1,
            Err(e) => panic!("{e:?}"),
        }
    }
    assert_eq!(rejected, 1);
    assert_eq!(held.len(), MAX_LIVE_WINDOWS as usize);
    assert_eq!(publication.live_windows(), MAX_LIVE_WINDOWS);
    for c in &clients {
        assert!(matches!(
            c.begin(CONTEXT, &[Step::Lookup]),
            Err(Error::Limit)
        ));
    }
    drop(held.pop());
    drop(clients[8].begin(CONTEXT, &[Step::Lookup]).unwrap());
    drop(held);
    assert_eq!(publication.live_windows(), 0);
    assert!(clients
        .iter()
        .all(|c| c.budget.live.load(Ordering::Acquire) == 0));

    let clients = (0..MAX_CLIENTS_PER_PUBLICATION - clients.len() as u32)
        .map(|_| Client::new(Arc::clone(&publication), publication.digest()).unwrap())
        .chain(clients)
        .collect::<Vec<_>>();
    assert!(matches!(
        Client::new(Arc::clone(&publication), publication.digest()),
        Err(Error::Limit)
    ));
    drop(clients);
    assert_eq!(publication.clients.load(Ordering::Acquire), 0);
}

#[test]
fn publication_envelopes_share_aggregate_admission_and_release_after_last_response() {
    static POOL: std::sync::OnceLock<AdmissionPool> = std::sync::OnceLock::new();
    let (a, b, table) = artifacts();
    // Exactly two complete envelopes, including target-specific AES key state.
    // Production's shared 16 MiB cap is unchanged; this seam tests contention.
    let cap = 2 * crate::gkm::PublicationResources::for_shapes(
        &[a.shape(), b.shape()],
        table.shape().0,
        table.shape().1,
    )
    .unwrap()
    .admitted_payload_bytes;
    let pool = POOL.get_or_init(|| AdmissionPool::new(cap));
    assert_eq!(pool.bytes.load(Ordering::Acquire), 0);
    let stages = [
        Stage { id: A, matrix: &a },
        Stage {
            id: [2; 32],
            matrix: &b,
        },
    ];
    let make = || {
        Publication::publish_in_pool(
            [42; 32],
            &stages,
            &table,
            &mut ScriptEntropy::new(&[2, 3, 5]),
            pool,
        )
    };
    let first = Arc::new(make().unwrap());
    let bytes = first.resources().admitted_payload_bytes;
    assert!(2 * bytes == cap && 3 * bytes > cap);
    let gate = Barrier::new(3);
    let results = std::thread::scope(|scope| {
        let one = scope.spawn(|| {
            gate.wait();
            make()
        });
        let two = scope.spawn(|| {
            gate.wait();
            make()
        });
        gate.wait();
        [one.join().unwrap(), two.join().unwrap()]
    });
    let mut winner = None;
    let mut rejected = 0;
    for result in results {
        match result {
            Ok(value) => winner = Some(value),
            Err(Error::Limit) => rejected += 1,
            Err(e) => panic!("{e:?}"),
        }
    }
    assert_eq!(rejected, 1);
    assert_eq!(pool.bytes.load(Ordering::Acquire), 2 * bytes);
    let mut rng = ScriptEntropy::new(&[]);
    assert!(matches!(
        Publication::publish_in_pool([42; 32], &stages, &table, &mut rng, pool),
        Err(Error::Limit)
    ));
    assert!(rng.calls.is_empty()); // rejected before entropy or payload allocation
    drop(winner);
    let owner = Client::new(Arc::clone(&first), first.digest()).unwrap();
    let pending = owner
        .begin(CONTEXT, &[Step::Linear(A)])
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let response = a.evaluate(&pending.share(Worker::Zero)).unwrap();
    drop(pending);
    drop(owner);
    drop(first);
    assert_eq!(pool.bytes.load(Ordering::Acquire), bytes);
    drop(response);
    assert_eq!(pool.bytes.load(Ordering::Acquire), 0);
}

#[test]
fn pending_role_slots_bound_response_handles_and_cancellation_keeps_their_leases() {
    let weights = [0; MAX_DIMENSION];
    let matrix = I16Matrix::from_encoded(MAX_DIMENSION, 1, &weights, &[1]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &[0; 4]).unwrap();
    let publication = Publication::publish_with(
        [0; 32],
        &[Stage {
            id: A,
            matrix: &matrix,
        }],
        &table,
        &mut ScriptEntropy::new(&[2; MAX_DIMENSION]),
    )
    .unwrap();
    let owner = client(publication);
    let pending = owner
        .begin(CONTEXT, &[Step::Linear(A)])
        .unwrap()
        .linear(&[0])
        .unwrap();
    let share = pending.share(Worker::Zero);
    let response =
        LinearResponse::from_untrusted(&share, share.binding(), &[0; MAX_DIMENSION]).unwrap();
    assert_eq!(response.values.capacity(), MAX_DIMENSION);
    for _ in 0..600 {
        assert!(matches!(
            LinearResponse::from_untrusted(&share, share.binding(), &[0; MAX_DIMENSION]),
            Err(Error::Retired)
        ));
        assert!(matches!(matrix.evaluate(&share), Err(Error::Retired)));
    }
    drop(pending);
    assert_eq!(owner.publication.live_windows(), 1);
    let held = (1..MAX_LIVE_WINDOWS)
        .map(|_| owner.begin(CONTEXT, &[Step::Lookup]).unwrap())
        .collect::<Vec<_>>();
    assert!(matches!(
        owner.begin(CONTEXT, &[Step::Lookup]),
        Err(Error::Limit)
    ));
    drop(response);
    drop(owner.begin(CONTEXT, &[Step::Lookup]).unwrap());
    drop(held);
    assert_eq!(owner.publication.live_windows(), 0);

    let pending = owner
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    let first = table.evaluate(&pending.share(Worker::Zero)).unwrap();
    let second = table.evaluate(&pending.share(Worker::One)).unwrap();
    drop(pending);
    assert_eq!(owner.publication.live_windows(), 1);
    drop(first);
    assert_eq!(owner.publication.live_windows(), 1);
    drop(second);
    assert_eq!(owner.publication.live_windows(), 0);

    let accepted = run_lookup(owner.begin(CONTEXT, &[Step::Lookup]).unwrap(), &table, 0)
        .finish()
        .unwrap();
    assert_eq!(owner.publication.live_windows(), 1);
    drop(accepted);
    assert_eq!(owner.publication.live_windows(), 0);
}

#[test]
fn ignored_constructor_or_evaluator_errors_cannot_resurrect_transactions() {
    let (a, b, table) = artifacts();
    let owner = client(publication(&a, &b, &table));
    for kind in 0..4 {
        let pending = owner
            .begin(CONTEXT, &[Step::Linear(A)])
            .unwrap()
            .linear(&[1, 2, 3])
            .unwrap();
        let share = pending.share(Worker::Zero);
        let failure = match kind {
            0 => LinearResponse::from_untrusted(&share, share.binding(), &[0; MAX_DIMENSION + 1]),
            1 => LinearResponse::from_untrusted(&share, share.binding(), &[]),
            2 => b.evaluate(&share),
            3 => injected(0, || a.evaluate(&share)),
            _ => unreachable!(),
        };
        assert!(failure.is_err()); // deliberately ignore the decoder/evaluator error
        assert!(matches!(
            a.evaluate(&pending.share(Worker::One)),
            Err(Error::Retired)
        ));
        let retry = a.evaluate(&pending.share(Worker::Zero));
        assert!(matches!(pending.freeze_first(retry), Err(Error::Retired)));
        assert_eq!(owner.publication.live_windows(), 0);
    }
    let pending = owner
        .begin(CONTEXT, &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    let share = pending.share(Worker::Zero);
    assert!(LookupResponse::from_untrusted(&share, share.binding(), &[]).is_err());
    let retry = table.evaluate(&share);
    assert!(matches!(pending.freeze_first(retry), Err(Error::Retired)));
    assert_eq!(owner.publication.live_windows(), 0);
}

#[test]
fn simultaneous_evaluations_of_one_role_get_one_buffer_and_poison_acceptance() {
    let (a, b, table) = artifacts();
    let owner = client(publication(&a, &b, &table));
    let pending = owner
        .begin(CONTEXT, &[Step::Linear(A)])
        .unwrap()
        .linear(&[1, 2, 3])
        .unwrap();
    let share = pending.share(Worker::Zero);
    let gate = Barrier::new(3);
    let responses = std::thread::scope(|scope| {
        let one = scope.spawn(|| {
            gate.wait();
            a.evaluate(&share)
        });
        let two = scope.spawn(|| {
            gate.wait();
            a.evaluate(&share)
        });
        gate.wait();
        [one.join().unwrap(), two.join().unwrap()]
    });
    let mut only_response = None;
    let mut rejected = 0;
    for result in responses {
        match result {
            Ok(response) => {
                assert!(only_response.is_none());
                only_response = Some(response);
            }
            Err(Error::Retired) => rejected += 1,
            Err(other) => panic!("unexpected response failure: {other:?}"),
        }
    }
    assert_eq!(rejected, 1);
    assert!(matches!(
        pending.freeze_first(Ok(only_response.unwrap())),
        Err(Error::Retired)
    ));
    assert_eq!(owner.publication.live_windows(), 0);
}

#[test]
fn every_import_publication_and_kernel_payload_allocation_is_fallible() {
    fail_each(
        |n| injected(n, || I16Matrix::from_encoded(2, 3, &[1; 6], &[2; 3])),
        || {},
    );
    fail_each(
        |n| injected(n, || I16Matrix::from_f64(2, 3, &[0.5; 6], &[2; 3])),
        || {},
    );
    fail_each(
        |n| injected(n, || FloatTable::from_le_bytes(1, 1, &[0; 4])),
        || {},
    );
    let (a, b, table) = artifacts();
    fail_each(|n| injected(n, || a.multiply_share(&[1, 2, 3])), || {});
    fail_each(|n| injected(n, || table.xor_rows(&[1, 0])), || {});
    static POOL: AdmissionPool = AdmissionPool::new(1 << 20);
    fail_each(
        |n| {
            let mut rng = ScriptEntropy::new(&[2, 3, 5]);
            injected(n, || {
                Publication::publish_in_pool(
                    [42; 32],
                    &[
                        Stage { id: A, matrix: &a },
                        Stage {
                            id: [2; 32],
                            matrix: &b,
                        },
                    ],
                    &table,
                    &mut rng,
                    &POOL,
                )
            })
        },
        || assert_eq!(POOL.bytes.load(Ordering::Acquire), 0),
    );
}

#[test]
fn every_window_payload_failure_retires_and_retained_response_capacity_is_not_refunded() {
    let (a, b, table) = artifacts();
    let owner = client(publication(&a, &b, &table));
    let retired = || assert_eq!(owner.publication.live_windows(), 0);
    fail_each(
        |n| injected(n, || owner.begin(CONTEXT, &[Step::Linear(A)])),
        retired,
    );
    fail_each(
        |n| {
            let window = owner.begin(CONTEXT, &[Step::Linear(A)]).unwrap();
            injected(n, || window.linear(&[1, 2, 3]))
        },
        retired,
    );
    fail_each(
        |n| {
            let window = owner.begin(CONTEXT, &[Step::Lookup]).unwrap();
            injected(n, || window.lookup(0))
        },
        retired,
    );
    fail_each(
        |n| {
            let pending = owner
                .begin(CONTEXT, &[Step::Linear(A)])
                .unwrap()
                .linear(&[1, 2, 3])
                .unwrap();
            let first = a.evaluate(&pending.share(Worker::Zero));
            let half = pending.freeze_first(first).unwrap();
            let second = a.evaluate(&half.remaining_share());
            injected(n, || half.freeze_second(second))
        },
        retired,
    );
    fail_each(
        |n| {
            let pending = owner
                .begin(CONTEXT, &[Step::Lookup])
                .unwrap()
                .lookup(0)
                .unwrap();
            let first = table.evaluate(&pending.share(Worker::Zero));
            let half = pending.freeze_first(first).unwrap();
            let second = table.evaluate(&half.remaining_share());
            injected(n, || half.freeze_second(second))
        },
        retired,
    );
    fail_each(
        |n| {
            let window = run_linear(
                owner.begin(CONTEXT, &[Step::Linear(A)]).unwrap(),
                &a,
                &[1, 2, 3],
            );
            injected(n, || window.finish())
        },
        retired,
    );

    for decode in [false, true] {
        let pending = owner
            .begin(CONTEXT, &[Step::Linear(A)])
            .unwrap()
            .linear(&[1, 2, 3])
            .unwrap();
        let held = a.evaluate(&pending.share(Worker::Zero)).unwrap();
        let share = pending.share(Worker::One);
        let error = injected(0, || {
            if decode {
                LinearResponse::from_untrusted(&share, share.binding(), &[0; 3])
            } else {
                a.evaluate(&share)
            }
        });
        assert!(matches!(error, Err(Error::Allocation)));
        drop(pending);
        assert_eq!(owner.publication.live_windows(), 1);
        drop(held);
        retired();
        let pending = owner
            .begin(CONTEXT, &[Step::Lookup])
            .unwrap()
            .lookup(0)
            .unwrap();
        let share = pending.share(Worker::Zero);
        let error = injected(0, || {
            if decode {
                LookupResponse::from_untrusted(&share, share.binding(), &[0; 12])
            } else {
                table.evaluate(&share)
            }
        });
        assert!(matches!(error, Err(Error::Allocation)));
        drop(pending);
        retired();
    }
}
