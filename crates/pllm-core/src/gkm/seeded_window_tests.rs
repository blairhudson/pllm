use super::*;
use crate::gkm::{
    seeded::tests::{linear_fixture, lookup_fixture, LINEAR_KEY, LOOKUP_KEY},
    tests::{integer_oracle, ScriptEntropy},
    window::tests::{artifacts, client, publication},
};

const A: [u8; 32] = [1; 32];
const B: [u8; 32] = [2; 32];
const CONTEXT: [u8; 32] = [7; 32];

#[derive(Default)]
struct Seeds {
    calls: Vec<Purpose>,
    fail: Option<Purpose>,
}
impl Entropy for Seeds {
    fn fill(&mut self, purpose: Purpose, out: &mut [u8]) -> Result<(), Error> {
        self.calls.push(purpose);
        if self.fail == Some(purpose) {
            return Err(Error::Entropy);
        }
        out.copy_from_slice(match purpose {
            Purpose::LinearSeed => &LINEAR_KEY,
            Purpose::LookupSeed => &LOOKUP_KEY,
            _ => panic!("seed source must never serve masks or verification"),
        });
        Ok(())
    }
}

struct Tape<'a>(Purpose, &'a [u8]);
impl Entropy for Tape<'_> {
    fn fill(&mut self, purpose: Purpose, out: &mut [u8]) -> Result<(), Error> {
        assert_eq!(purpose, self.0);
        out.copy_from_slice(self.1);
        self.1 = &[];
        Ok(())
    }
}

fn linear(window: Window, worker: &mut SeededWorker, matrix: &I16Matrix) -> Window {
    let pending = window.linear(&[-7, 11, -13]).unwrap();
    let left = matrix.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(left).unwrap();
    let right = worker.evaluate_linear(matrix, &half.remaining_share());
    half.freeze_second(right).unwrap()
}

fn lookup(window: Window, worker: &mut SeededWorker, table: &FloatTable) -> Window {
    let pending = window.lookup(10).unwrap();
    let left = table.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(left).unwrap();
    let right = worker.evaluate_lookup(table, &half.remaining_share());
    half.freeze_second(right).unwrap()
}

#[test]
fn matched_full_and_seeded_transcripts_match_independent_aes_integer_and_byte_oracles() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let plan = [
        Step::Lookup,
        Step::Linear(A),
        Step::Lookup,
        Step::Linear(B),
        Step::Linear(A),
        Step::Lookup,
    ];
    let mut seeds = Seeds::default();
    let (mut seeded, mut worker) = client
        .begin_seeded_with(CONTEXT, &plan, &mut seeds)
        .unwrap();
    let mut full = client.begin(CONTEXT, &plan).unwrap();
    assert!(Arc::ptr_eq(&seeded.plan, &worker.plan)); // one full-window schedule allocation
    assert_eq!(seeded.plan.capacity(), plan.len());
    assert_ne!(seeded.plan_digest, full.plan_digest);
    // The default's pre-existing plan identity is byte-for-byte unchanged.
    let mut hash = Sha256::new();
    hash.update(b"pllm.gkm.window-plan.v1");
    hash.update(client.publication.digest());
    hash.update(CONTEXT);
    hash.update((plan.len() as u64).to_le_bytes());
    for step in plan {
        match step {
            Step::Lookup => hash.update([0]),
            Step::Linear(id) => {
                hash.update([1]);
                hash.update(id);
            }
        }
    }
    assert_eq!(full.plan_digest.as_slice(), hash.finalize().as_slice());

    let (linear_bytes, lookup_bytes) = (linear_fixture(), lookup_fixture());
    let (mut l, mut q) = (0, 0);
    let mut seeded_checks = ScriptEntropy::new(&[3, 5, 7, 11, 13]);
    let mut full_checks = ScriptEntropy::new(&[3, 5, 7, 11, 13]);
    for (ordinal, step) in plan.iter().enumerate() {
        match step {
            Step::Lookup => {
                let token = [10, 0, 9][q / 2];
                let mut selector = lookup_bytes[q..q + 2].to_vec();
                q += 2;
                selector[1] &= 7;
                let mut complement = selector.clone();
                complement[token / 8] ^= 1 << (token % 8);
                let s = seeded.lookup_with(token, &mut Seeds::default()).unwrap();
                let f = full
                    .lookup_with(token, &mut Tape(Purpose::LookupMask, &complement))
                    .unwrap();
                assert_eq!(
                    s.share(Worker::Zero).selector(),
                    f.share(Worker::Zero).selector()
                );
                assert_eq!(f.share(Worker::One).selector().unwrap(), selector);
                assert!(s.share(Worker::One).selector().is_none());
                assert!(s.selectors[1].is_empty());
                assert_eq!(s.share(Worker::Zero).selector().unwrap()[1] & 0xf8, 0);
                let s0 = table.evaluate(&s.share(Worker::Zero)).unwrap();
                let f0 = table.evaluate(&f.share(Worker::Zero)).unwrap();
                let s1 = worker
                    .evaluate_lookup(&table, &s.share(Worker::One))
                    .unwrap();
                let f1 = table.evaluate(&f.share(Worker::One)).unwrap();
                assert_eq!(*s0.bytes, *f0.bytes);
                assert_eq!(*s1.bytes, *f1.bytes);
                // Separate byte oracle; no FloatTable selection helper used.
                let mut oracle = [0u8; 12];
                for row in 0..11 {
                    if selector[row / 8] & (1 << (row % 8)) != 0 {
                        for (dst, src) in oracle
                            .iter_mut()
                            .zip(&table.bytes[row * 12..(row + 1) * 12])
                        {
                            *dst ^= src;
                        }
                    }
                }
                assert_eq!(*s1.bytes, oracle);
                seeded = s
                    .freeze_first(Ok(s0))
                    .unwrap()
                    .freeze_second(Ok(s1))
                    .unwrap();
                full = f
                    .freeze_first(Ok(f0))
                    .unwrap()
                    .freeze_second(Ok(f1))
                    .unwrap();
                let bytes: Vec<_> = seeded
                    .tentative_values()
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                assert_eq!(bytes, table.bytes[token * 12..(token + 1) * 12]);
            }
            Step::Linear(id) => {
                let matrix = if *id == A { &a } else { &b };
                let input = [-(ordinal as i64 + 1), 42, -17];
                let mask: Vec<_> = linear_bytes[l..l + 24]
                    .chunks_exact(8)
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                l += 24;
                let complement: Vec<_> = input
                    .iter()
                    .zip(&mask)
                    .flat_map(|(&x, &r)| (x as u64).wrapping_sub(r).to_le_bytes())
                    .collect();
                let s = seeded.linear_with(&input, &mut Seeds::default()).unwrap();
                let f = full
                    .linear_with(&input, &mut Tape(Purpose::LinearMask, &complement))
                    .unwrap();
                assert_eq!(
                    s.share(Worker::Zero).values(),
                    f.share(Worker::Zero).values()
                );
                assert_eq!(f.share(Worker::One).values().unwrap(), mask);
                assert!(s.share(Worker::One).values().is_none());
                assert!(s.shares[1].is_empty());
                let s0 = matrix.evaluate(&s.share(Worker::Zero)).unwrap();
                let f0 = matrix.evaluate(&f.share(Worker::Zero)).unwrap();
                let s1 = worker
                    .evaluate_linear(matrix, &s.share(Worker::One))
                    .unwrap();
                let f1 = matrix.evaluate(&f.share(Worker::One)).unwrap();
                assert_eq!(*s0.values, *f0.values);
                assert_eq!(*s1.values, *f1.values);
                seeded = s
                    .freeze_first(Ok(s0))
                    .unwrap()
                    .freeze_second_with(Ok(s1), &mut seeded_checks)
                    .unwrap();
                full = f
                    .freeze_first(Ok(f0))
                    .unwrap()
                    .freeze_second_with(Ok(f1), &mut full_checks)
                    .unwrap();
                assert_eq!(
                    seeded.tentative_signed(),
                    integer_oracle(&matrix.weights, &input)
                );
            }
        }
        assert_eq!(seeded.tentative_signed(), full.tentative_signed());
        assert_eq!(seeded.tentative_values(), full.tentative_values());
        assert_eq!(seeds.calls, [Purpose::LinearSeed, Purpose::LookupSeed]);
    }
    assert!(worker.streams.complete());
    let s = seeded.finish_with(&mut seeded_checks).unwrap();
    let f = full.finish_with(&mut full_checks).unwrap();
    assert_eq!(s.values(), f.values());
    assert_eq!(s.sharing_profile(), SharingProfile::Aes128WindowV1);
    assert_eq!(f.sharing_profile(), SharingProfile::FullShares);
    assert_eq!(seeded_checks.calls, full_checks.calls);
    assert_eq!(
        seeded_checks.calls,
        [
            Purpose::ClaimChallenge,
            Purpose::ClaimChallenge,
            Purpose::ClaimChallenge,
            Purpose::MatrixChallenge,
            Purpose::MatrixChallenge
        ]
    );
    drop((s, f));
    assert_eq!(client.publication.live_windows(), 1); // designated owner still retains keys/plan
    drop(worker);
    assert_eq!(client.publication.live_windows(), 0);
}

fn binding_mutations() -> [fn(&mut Binding); 11] {
    [
        |b| b.publication[0] ^= 1,
        |b| b.session[0] ^= 1,
        |b| b.sequence += 1,
        |b| b.context[0] ^= 1,
        |b| b.plan[0] ^= 1,
        |b| b.total += 1,
        |b| b.ordinal += 1,
        |b| {
            b.step = match b.step {
                Step::Lookup => Step::Linear(A),
                Step::Linear(_) => Step::Lookup,
            }
        },
        |b| b.artifact[0] ^= 1,
        |b| b.worker = Worker::Zero,
        |b| b.profile = SharingProfile::FullShares,
    ]
}

#[test]
fn every_binding_field_role_artifact_and_profile_is_enforced_before_expansion() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    for mutate in binding_mutations() {
        let (window, mut worker) = client.begin_seeded(CONTEXT, &[Step::Linear(A)]).unwrap();
        let pending = window.linear(&[1, 2, 3]).unwrap();
        let mut view = pending.share(Worker::One);
        mutate(&mut view.binding);
        assert!(matches!(
            worker.evaluate_linear(&a, &view),
            Err(Error::Binding)
        ));
        assert_eq!(worker.ordinal, 0);
        assert!(matches!(
            worker.evaluate_linear(&a, &pending.share(Worker::One)),
            Err(Error::Retired)
        ));
        assert!(matches!(
            a.evaluate(&pending.share(Worker::Zero)),
            Err(Error::Retired)
        ));
    }
    let (window, mut worker) = client.begin_seeded(CONTEXT, &[Step::Linear(A)]).unwrap();
    let pending = window.linear(&[1, 2, 3]).unwrap();
    assert!(matches!(
        worker.evaluate_linear(&b, &pending.share(Worker::One)),
        Err(Error::Binding)
    ));
    assert_eq!(worker.ordinal, 0);
    drop((pending, worker));
    let (window, worker) = client.begin_seeded(CONTEXT, &[Step::Linear(A)]).unwrap();
    let pending = window.linear(&[1, 2, 3]).unwrap();
    assert!(matches!(
        a.evaluate(&pending.share(Worker::One)),
        Err(Error::Binding)
    ));
    assert!(matches!(
        a.evaluate(&pending.share(Worker::Zero)),
        Err(Error::Retired)
    ));
    drop((pending, worker));
}

#[test]
fn lookup_binding_roles_same_shape_table_substitution_and_none_evaluator_retire() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    for mutate in binding_mutations() {
        let mut seeds = Seeds::default();
        let (window, mut worker) = client
            .begin_seeded_with(CONTEXT, &[Step::Lookup], &mut seeds)
            .unwrap();
        let pending = window.lookup(10).unwrap();
        let mut view = pending.share(Worker::One);
        assert!(view.selector().is_none());
        mutate(&mut view.binding);
        assert!(matches!(
            worker.evaluate_lookup(&table, &view),
            Err(Error::Binding)
        ));
        assert_eq!(worker.ordinal, 0);
        assert!(matches!(
            worker.evaluate_lookup(&table, &pending.share(Worker::One)),
            Err(Error::Retired)
        ));
        assert!(matches!(
            table.evaluate(&pending.share(Worker::Zero)),
            Err(Error::Retired)
        ));
        assert_eq!(seeds.calls, [Purpose::LinearSeed, Purpose::LookupSeed]);
    }

    let mut substituted_bytes = table.bytes.clone();
    substituted_bytes[..4].copy_from_slice(&1.0_f32.to_le_bytes());
    let substituted =
        FloatTable::from_le_bytes(table.shape().0, table.shape().1, &substituted_bytes).unwrap();
    assert_eq!(table.shape(), substituted.shape());
    assert_ne!(table.digest(), substituted.digest());
    let (window, mut worker) = client.begin_seeded(CONTEXT, &[Step::Lookup]).unwrap();
    let pending = window.lookup(10).unwrap();
    assert!(matches!(
        worker.evaluate_lookup(&substituted, &pending.share(Worker::One)),
        Err(Error::Binding)
    ));
    assert_eq!(worker.ordinal, 0);
    assert!(matches!(
        worker.evaluate_lookup(&table, &pending.share(Worker::One)),
        Err(Error::Retired)
    ));
    drop((pending, worker));

    // The real complementary view, not just a mutated role tag, is rejected.
    let (window, mut worker) = client.begin_seeded(CONTEXT, &[Step::Lookup]).unwrap();
    let pending = window.lookup(10).unwrap();
    assert!(pending.share(Worker::Zero).selector().is_some());
    assert!(matches!(
        worker.evaluate_lookup(&table, &pending.share(Worker::Zero)),
        Err(Error::Binding)
    ));
    assert_eq!(worker.ordinal, 0);
    assert!(matches!(
        worker.evaluate_lookup(&table, &pending.share(Worker::One)),
        Err(Error::Retired)
    ));
    drop((pending, worker));

    // FloatTable's ordinary evaluator must not treat None as an empty selector.
    let (window, mut worker) = client.begin_seeded(CONTEXT, &[Step::Lookup]).unwrap();
    let pending = window.lookup(10).unwrap();
    assert!(pending.share(Worker::One).selector().is_none());
    assert!(matches!(
        table.evaluate(&pending.share(Worker::One)),
        Err(Error::Binding)
    ));
    assert!(matches!(
        worker.evaluate_lookup(&table, &pending.share(Worker::One)),
        Err(Error::Retired)
    ));
    let reply = table.evaluate(&pending.share(Worker::Zero));
    assert!(matches!(pending.freeze_first(reply), Err(Error::Retired)));
    drop(worker);
    assert_eq!(client.publication.live_windows(), 0);
}

#[test]
fn lookup_replay_cross_window_worker_and_response_substitution_keep_sticky_leases() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let plan = [Step::Lookup];
    let (w1, mut worker1) = client.begin_seeded(CONTEXT, &plan).unwrap();
    let (w2, mut worker2) = client.begin_seeded(CONTEXT, &plan).unwrap();
    let p1 = w1.lookup(10).unwrap();
    let p2 = w2.lookup(10).unwrap();
    assert!(matches!(
        worker1.evaluate_lookup(&table, &p2.share(Worker::One)),
        Err(Error::Binding)
    ));
    assert!(matches!(
        worker1.evaluate_lookup(&table, &p1.share(Worker::One)),
        Err(Error::Retired)
    ));
    assert!(matches!(
        worker2.evaluate_lookup(&table, &p2.share(Worker::One)),
        Err(Error::Retired)
    ));
    assert!(matches!(
        table.evaluate(&p1.share(Worker::Zero)),
        Err(Error::Retired)
    ));
    assert!(matches!(
        table.evaluate(&p2.share(Worker::Zero)),
        Err(Error::Retired)
    ));
    drop((p1, p2, worker1, worker2));

    let (w1, mut worker1) = client.begin_seeded(CONTEXT, &plan).unwrap();
    let (w2, worker2) = client.begin_seeded(CONTEXT, &plan).unwrap();
    let p1 = w1.lookup(10).unwrap();
    let response1 = worker1
        .evaluate_lookup(&table, &p1.share(Worker::One))
        .unwrap();
    let p2 = w2.lookup(10).unwrap();
    let first2 = table.evaluate(&p2.share(Worker::Zero));
    let half2 = p2.freeze_first(first2).unwrap();
    assert!(matches!(
        half2.freeze_second(Ok(response1)),
        Err(Error::Binding)
    ));
    assert_eq!(worker2._lease.ensure_open(), Err(Error::Retired));
    drop((p1, worker1, worker2));

    let (window, mut worker) = client.begin_seeded(CONTEXT, &plan).unwrap();
    let pending = window.lookup(10).unwrap();
    let response = worker
        .evaluate_lookup(&table, &pending.share(Worker::One))
        .unwrap();
    for _ in 0..600 {
        assert!(matches!(
            worker.evaluate_lookup(&table, &pending.share(Worker::One)),
            Err(Error::Retired)
        ));
    }
    assert!(matches!(
        table.evaluate(&pending.share(Worker::Zero)),
        Err(Error::Retired)
    ));
    drop((pending, worker));
    assert_eq!(client.publication.live_windows(), 1);
    drop(response);
    assert_eq!(client.publication.live_windows(), 0);

    let (window, mut worker) = client.begin_seeded(CONTEXT, &plan).unwrap();
    let pending = window.lookup(10).unwrap();
    let view = pending.share(Worker::One);
    let mut wrong = view.binding();
    wrong.context[0] ^= 1;
    assert!(matches!(
        LookupResponse::from_untrusted(&view, wrong, &[0; 12]),
        Err(Error::Binding)
    ));
    // Discarding the constructor error cannot reopen the correct expansion path.
    assert!(matches!(
        worker.evaluate_lookup(&table, &view),
        Err(Error::Retired)
    ));
    drop((pending, worker));
    assert_eq!(client.publication.live_windows(), 0);
}

#[test]
fn authentic_unexpanded_lookup_response_still_cannot_mint_an_accepted_window() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let (window, worker) = client
        .begin_seeded_with(CONTEXT, &[Step::Lookup], &mut Seeds::default())
        .unwrap();
    let pending = window.lookup(10).unwrap();
    let first = table.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(first).unwrap();

    // Independent OpenSSL stream + explicit byte oracle. Do not invoke the
    // seeded worker or FloatTable::xor_rows to construct this valid response.
    let fixture = lookup_fixture();
    let selector = [fixture[0], fixture[1] & 7];
    let mut bytes = [0u8; 12];
    for row in 0..11 {
        if selector[row / 8] & (1 << (row % 8)) != 0 {
            for (dst, src) in bytes.iter_mut().zip(&table.bytes[row * 12..(row + 1) * 12]) {
                *dst ^= src;
            }
        }
    }
    let view = half.remaining_share();
    assert!(view.selector().is_none());
    let raw = LookupResponse::from_untrusted(&view, view.binding(), &bytes);
    let window = half.freeze_second(raw).unwrap(); // row authentication really succeeds
    let actual: Vec<_> = window
        .tentative_values()
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(actual, table.bytes[10 * 12..11 * 12]);
    assert_eq!(worker.ordinal, 0);
    let mut challenges = ScriptEntropy::new(&[]);
    assert!(matches!(
        window.finish_with(&mut challenges),
        Err(Error::Incomplete)
    ));
    assert!(challenges.calls.is_empty());
    assert_eq!(worker._lease.ensure_open(), Err(Error::Retired));
    assert_eq!(client.publication.live_windows(), 1);
    drop(worker);
    assert_eq!(client.publication.live_windows(), 0);
}

#[test]
fn replay_cross_window_substitution_cancellation_and_ignored_errors_retire_all_owners() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let plan = [Step::Linear(A)];
    let (w1, mut worker1) = client.begin_seeded(CONTEXT, &plan).unwrap();
    let (w2, worker2) = client.begin_seeded(CONTEXT, &plan).unwrap();
    let p1 = w1.linear(&[1, 2, 3]).unwrap();
    let p2 = w2.linear(&[1, 2, 3]).unwrap();
    assert_ne!(
        p1.share(Worker::Zero).values(),
        p2.share(Worker::Zero).values()
    );
    assert!(matches!(
        worker1.evaluate_linear(&a, &p2.share(Worker::One)),
        Err(Error::Binding)
    ));
    assert!(matches!(
        a.evaluate(&p1.share(Worker::Zero)),
        Err(Error::Retired)
    ));
    assert!(matches!(
        a.evaluate(&p2.share(Worker::Zero)),
        Err(Error::Retired)
    ));
    drop((p1, p2, worker1, worker2));

    let (w, mut worker) = client.begin_seeded(CONTEXT, &plan).unwrap();
    let p = w.linear(&[1, 2, 3]).unwrap();
    let first = worker.evaluate_linear(&a, &p.share(Worker::One)).unwrap();
    for _ in 0..600 {
        assert!(matches!(
            worker.evaluate_linear(&a, &p.share(Worker::One)),
            Err(Error::Retired)
        ));
    }
    drop(p);
    drop(worker);
    assert_eq!(client.publication.live_windows(), 1); // response lease survives cancellation
    drop(first);
    assert_eq!(client.publication.live_windows(), 0);

    let (w, mut worker) = client.begin_seeded(CONTEXT, &plan).unwrap();
    let p = w.linear(&[1, 2, 3]).unwrap();
    let view = p.share(Worker::One);
    let mut binding = view.binding();
    binding.context[0] ^= 1;
    let _ignored = LinearResponse::from_untrusted(&view, binding, &[0; 3]);
    assert!(matches!(
        worker.evaluate_linear(&a, &view),
        Err(Error::Retired)
    ));
    drop((p, worker));

    let (w, worker) = client.begin_seeded(CONTEXT, &plan).unwrap();
    drop(w);
    assert_eq!(worker._lease.ensure_open(), Err(Error::Retired));
    assert_eq!(client.publication.live_windows(), 1);
    drop(worker);
    let (w, worker) = client.begin_seeded(CONTEXT, &plan).unwrap();
    drop(worker);
    assert!(matches!(w.linear(&[1, 2, 3]), Err(Error::Retired)));
    assert_eq!(client.publication.live_windows(), 0);
}

#[test]
fn unexpanded_or_incomplete_seeded_schedule_cannot_mint_accepted_output() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    let (window, worker) = client
        .begin_seeded_with(CONTEXT, &[Step::Linear(A)], &mut Seeds::default())
        .unwrap();
    let pending = window.linear(&[1, 2, 3]).unwrap();
    let left = a.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(left).unwrap();
    let fixture = linear_fixture();
    let mask: Vec<_> = fixture[..24]
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let values = a.multiply_share(&mask).unwrap();
    let view = half.remaining_share();
    let raw = LinearResponse::from_untrusted(&view, view.binding(), &values);
    let mut challenges = ScriptEntropy::new(&[3]);
    let window = half.freeze_second_with(raw, &mut challenges).unwrap();
    assert!(matches!(
        window.finish_with(&mut challenges),
        Err(Error::Incomplete)
    ));
    assert_eq!(challenges.calls, [Purpose::ClaimChallenge]);
    drop(worker);
    let (window, mut worker) = client
        .begin_seeded(CONTEXT, &[Step::Lookup, Step::Linear(A)])
        .unwrap();
    let window = lookup(window, &mut worker, &table);
    assert!(matches!(window.finish(), Err(Error::Incomplete)));
    drop(worker);
    assert_eq!(client.publication.live_windows(), 0);
}

#[test]
fn seeded_corruption_preserves_row_authentication_and_final_window_checks() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    for role in 0..2 {
        let (window, mut worker) = client.begin_seeded(CONTEXT, &[Step::Linear(A)]).unwrap();
        let pending = window.linear(&[1, 2, 3]).unwrap();
        let mut left = a.evaluate(&pending.share(Worker::Zero)).unwrap();
        let mut right = worker
            .evaluate_linear(&a, &pending.share(Worker::One))
            .unwrap();
        let bad = if role == 0 { &mut left } else { &mut right };
        bad.values[0] = bad.values[0].wrapping_add(1);
        let window = pending
            .freeze_first(Ok(left))
            .unwrap()
            .freeze_second(Ok(right))
            .unwrap();
        assert!(matches!(window.finish(), Err(Error::Verification)));
    }
    for byte in 0..12 {
        let (window, mut worker) = client.begin_seeded(CONTEXT, &[Step::Lookup]).unwrap();
        let pending = window.lookup(10).unwrap();
        let left = table.evaluate(&pending.share(Worker::Zero)).unwrap();
        let mut right = worker
            .evaluate_lookup(&table, &pending.share(Worker::One))
            .unwrap();
        right.bytes[byte] ^= 1;
        assert!(matches!(
            pending
                .freeze_first(Ok(left))
                .unwrap()
                .freeze_second(Ok(right)),
            Err(Error::Verification)
        ));
    }
    assert_eq!(client.publication.live_windows(), 0);
}

#[test]
fn seed_entropy_and_every_seeded_payload_failure_roll_back_without_resampling() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    for purpose in [Purpose::LinearSeed, Purpose::LookupSeed] {
        let mut seeds = Seeds {
            fail: Some(purpose),
            ..Seeds::default()
        };
        assert!(matches!(
            client.begin_seeded_with(CONTEXT, &[Step::Lookup], &mut seeds),
            Err(Error::Entropy)
        ));
        assert_eq!(seeds.calls.iter().filter(|&&p| p == purpose).count(), 1);
        assert_eq!(client.publication.live_windows(), 0);
    }
    let plan = [Step::Lookup, Step::Linear(A), Step::Linear(B), Step::Lookup];
    let mut reached = false;
    for after in 0..64 {
        let mut seeds = Seeds::default();
        let mut challenges = ScriptEntropy::new(&[3, 5, 7, 11]);
        let result = {
            let _guard = payload::failure::after(after);
            (|| {
                let (mut w, mut worker) = client.begin_seeded_with(CONTEXT, &plan, &mut seeds)?;
                for step in plan {
                    w = match step {
                        Step::Lookup => {
                            let p = w.lookup(10)?;
                            let r = table.evaluate(&p.share(Worker::Zero));
                            let h = p.freeze_first(r)?;
                            let r = worker.evaluate_lookup(&table, &h.remaining_share());
                            h.freeze_second(r)?
                        }
                        Step::Linear(id) => {
                            let matrix = if id == A { &a } else { &b };
                            let p = w.linear(&[-7, 11, -13])?;
                            let r = matrix.evaluate(&p.share(Worker::Zero));
                            let h = p.freeze_first(r)?;
                            let r = worker.evaluate_linear(matrix, &h.remaining_share());
                            h.freeze_second_with(r, &mut challenges)?
                        }
                    };
                }
                w.finish_with(&mut challenges)
            })()
        };
        match result {
            Err(Error::Allocation) => (),
            Ok(receipt) => {
                drop(receipt);
                reached = true;
            }
            Err(other) => panic!("payload {after}: {other:?}"),
        }
        assert_eq!(client.publication.live_windows(), 0);
        assert!(
            seeds.calls.is_empty() || seeds.calls == [Purpose::LinearSeed, Purpose::LookupSeed]
        );
        if reached {
            break;
        }
    }
    assert!(reached);
    // Failure after local expansion but before allocating its result is sticky.
    let (w, mut worker) = client.begin_seeded(CONTEXT, &[Step::Linear(A)]).unwrap();
    let p = w.linear(&[1, 2, 3]).unwrap();
    let error = {
        let _guard = payload::failure::after(1);
        worker.evaluate_linear(&a, &p.share(Worker::One))
    };
    assert!(matches!(error, Err(Error::Allocation)));
    assert!(matches!(
        worker.evaluate_linear(&a, &p.share(Worker::One)),
        Err(Error::Retired)
    ));
    drop(p);
    assert_eq!(client.publication.live_windows(), 1);
    drop(worker);
    assert_eq!(client.publication.live_windows(), 0);
}

#[test]
fn seeded_workers_share_admission_with_all_clients_and_keep_full_plan_reservations() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let publication = Arc::new(publication(&a, &b, &table));
    let resources = publication.resources();
    assert_eq!(
        resources.seeded_window_extra_bytes,
        crate::gkm::seeded::WINDOW_EXTRA_BYTES
    );
    assert!(resources.seeded_window_extra_bytes >= 2 * size_of::<Streams>() + MAX_DIMENSION * 8);
    let clients: Vec<_> = (0..9)
        .map(|_| Client::new(Arc::clone(&publication), publication.digest()).unwrap())
        .collect();
    let mut workers = Vec::new();
    for c in &clients[..8] {
        let (window, worker) = c
            .begin_seeded(CONTEXT, &vec![Step::Lookup; MAX_WINDOW_STEPS])
            .unwrap();
        assert_eq!(worker.plan.capacity(), MAX_WINDOW_STEPS);
        assert!(Arc::ptr_eq(&worker.plan, &window.plan));
        drop(window); // only the designated owner remains, with keys and full schedule
        workers.push(worker);
    }
    assert_eq!(publication.live_windows(), MAX_LIVE_WINDOWS);
    assert!(matches!(
        clients[8].begin(CONTEXT, &[Step::Lookup]),
        Err(Error::Limit)
    ));
    assert!(matches!(
        clients[8].begin_seeded(CONTEXT, &[Step::Lookup]),
        Err(Error::Limit)
    ));
    drop(workers.pop());
    drop(clients[8].begin_seeded(CONTEXT, &[Step::Lookup]).unwrap());
    drop(workers);
    assert_eq!(publication.live_windows(), 0);
    let (w, mut worker) = clients[0]
        .begin_seeded(CONTEXT, &[Step::Linear(A)])
        .unwrap();
    let accepted = linear(w, &mut worker, &a).finish().unwrap();
    drop(worker);
    assert_eq!(publication.live_windows(), 1);
    drop(accepted);
    assert_eq!(publication.live_windows(), 0);
}

#[test]
fn seeded_verifier_entropy_stays_post_freeze_and_never_falls_back_to_shared_streams() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (a, b, table) = artifacts();
    let client = client(publication(&a, &b, &table));
    for failure in [Purpose::ClaimChallenge, Purpose::MatrixChallenge] {
        let mut seeds = Seeds::default();
        let (window, mut worker) = client
            .begin_seeded_with(CONTEXT, &[Step::Linear(A)], &mut seeds)
            .unwrap();
        let pending = window
            .linear_with(&[1, 2, 3], &mut Seeds::default())
            .unwrap();
        let mut checks = ScriptEntropy::new(&[3]);
        checks.fail = Some(failure);
        let left = a.evaluate(&pending.share(Worker::Zero));
        let half = pending.freeze_first(left).unwrap();
        assert!(checks.calls.is_empty());
        let right = worker.evaluate_linear(&a, &half.remaining_share());
        assert!(checks.calls.is_empty());
        assert_eq!(seeds.calls, [Purpose::LinearSeed, Purpose::LookupSeed]);
        let result = half.freeze_second_with(right, &mut checks);
        if failure == Purpose::ClaimChallenge {
            assert!(matches!(result, Err(Error::Entropy)));
            assert_eq!(checks.calls, [Purpose::ClaimChallenge]);
        } else {
            assert!(matches!(
                result.unwrap().finish_with(&mut checks),
                Err(Error::Entropy)
            ));
            assert_eq!(
                checks.calls,
                [Purpose::ClaimChallenge, Purpose::MatrixChallenge]
            );
        }
        assert_eq!(worker._lease.ensure_open(), Err(Error::Retired));
        drop(worker);
        assert_eq!(client.publication.live_windows(), 0);
    }
}
