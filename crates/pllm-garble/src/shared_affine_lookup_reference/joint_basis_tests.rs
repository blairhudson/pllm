//! Authored source-only tests: NOT RUN. Exhaustive dimensions are named below;
//! arbitrary partitions, random tapes and all share splits are not exhaustive.
use super::*;
use crate::interval_fss::RingComparison;
use std::sync::atomic::Ordering;
use std::{cell::RefCell, panic::AssertUnwindSafe};

// Same thread-local/RAII pattern as ring_bulk_tests, restricted to this
// module's actual 48-byte OS draw in issue(). No public production hook.
// This legacy controller still covers only the local draw. The separate bounded
// downstream controller below is mutually exclusive with it and the triple hook.
thread_local! {
    static SOURCE: RefCell<Option<Source>> = const { RefCell::new(None) };
}
struct Source {
    calls: Vec<usize>,
    fail_prefix: Option<usize>,
    unwind: bool,
}
struct SourceGuard;
impl SourceGuard {
    fn install(fail_prefix: Option<usize>, unwind: bool) -> Self {
        assert!(!super::super::joint_basis_test_hooks::active());
        assert!(!product::test_source_active());
        SOURCE.with(|source| {
            assert!(source.borrow().is_none());
            *source.borrow_mut() = Some(Source { calls: Vec::new(), fail_prefix, unwind });
        });
        Self
    }
    fn calls(&self) -> Vec<usize> {
        SOURCE.with(|source| source.borrow().as_ref().unwrap().calls.clone())
    }
}
impl Drop for SourceGuard {
    fn drop(&mut self) { SOURCE.with(|source| *source.borrow_mut() = None); }
}
pub(super) fn source_active() -> bool {
    SOURCE.with(|source| source.try_borrow().map_or(true, |source| source.is_some()))
}
pub(super) fn fill_override(bytes: &mut [u8]) -> Option<Result<(), Error>> {
    SOURCE.with(|source| {
        let mut source = source.borrow_mut();
        let source = source.as_mut()?;
        source.calls.push(bytes.len());
        assert_eq!(bytes.len(), 48);
        assert!(bytes.iter().all(|&byte| byte == 0));
        if let Some(prefix) = source.fail_prefix {
            bytes[..prefix].fill(0xa5);
            assert!(!source.unwind, "injected joint issuance partial-fill unwind");
            Some(Err(Error::Randomness))
        } else {
            bytes.fill(0x39);
            Some(Ok(()))
        }
    })
}

fn context() -> Context {
    Context { plan: [1; 32], session: [2; 32], operation: [3; 32], first_tensor_index: 7 }
}
fn budget() -> Budget { Budget::new(MAX_ALLOCATION_BYTES).unwrap() }
fn basis<'a>(bits: u8, budget: &'a Budget) -> JointBasis<'a> {
    let n = 1u64 << bits;
    JointBasis::new(bits, &[0, 1, n / 2, n],
        &[[n - 1, 1], [2, n - 1], [0, 0]], [4; 32], [5; 32], budget).unwrap()
}
fn scan(basis: &JointBasis<'_>, x: u64) -> [u64; 2] {
    basis.table.boundaries().windows(2).zip(basis.table.coefficients())
        .find_map(|(w, row)| (w[0] <= x && x < w[1]).then_some(*row)).unwrap()
}

#[test]
fn exhaustive_comparison_identity_widths_2_through_6_every_mask_input_boundary() {
    // sum N^2*(N+1) = 305040 triples; one DCF evaluation per holder/triple
    // plus reusable eval(0). No partition/share/product cross product here.
    // Actual finite count is computed, not a sampled claim.
    let mut checked = 0u64;
    let mut expected = 0u64;
    for bits in 2..=6 {
        let n = 1u64 << bits;
        expected += n * n * (n + 1);
        for r in 0..n {
            let cost = RingComparison::resources(bits, bits).unwrap();
            let [a, b] = RingComparison::issue(bits, bits, r, cost.combined_allocation_bytes).unwrap();
            for x in 0..n {
                let u = (x + r) & (n - 1);
                let zero = a.eval(0, u, 0).unwrap().wrapping_add(b.eval(1, u, 0).unwrap());
                for p in 0..=n {
                    let sum = a.eval(0, u, p).unwrap()
                        .wrapping_add(b.eval(1, u, p).unwrap()).wrapping_sub(zero)
                        .wrapping_add(RingComparison::lt_correction(bits, bits, r, p).unwrap())
                        & (n - 1);
                    assert_eq!(sum, u64::from(x < p), "bits={bits} r={r} x={x} p={p}");
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, expected);
}

fn run(bits: u8, r: u64, x: u64, split: u64, affine: bool) {
    let budget = budget();
    let basis = basis(bits, &budget);
    let mask = (1u64 << bits) - 1;
    let predicted = scan(&basis, x);
    let [mut a, mut b] = basis.issue_with(context(), 9, |bytes| {
        // Deterministic injection restricted to this private unit-test seam.
        // Ring DCF/base and triple issuance still use ordinary OS randomness.
        bytes.fill(11);
        bytes[..8].copy_from_slice(&r.to_le_bytes());
        bytes[8..16].copy_from_slice(&mask.to_le_bytes());
        Ok(())
    }).unwrap();
    let fa = a.start(split).unwrap();
    let fb = b.start(x.wrapping_sub(split) & mask).unwrap();
    if affine {
        let pa = a.advance_affine(fb.as_bytes()).unwrap();
        let pb = b.advance_affine(fa.as_bytes()).unwrap();
        let ya = a.finish_affine(pb.as_bytes()).unwrap();
        let yb = b.finish_affine(pa.as_bytes()).unwrap();
        assert_eq!(ya.share().wrapping_add(yb.share()) & mask,
            predicted[0].wrapping_add(predicted[1].wrapping_mul(x)) & mask);
        assert!(matches!(a.finish_affine(pb.as_bytes()), Err(Error::Consumed)));
    } else {
        let ya = a.finish(fb.as_bytes()).unwrap();
        let yb = b.finish(fa.as_bytes()).unwrap();
        for channel in 0..2 {
            assert_eq!(ya.shares()[channel].wrapping_add(yb.shares()[channel]) & mask,
                predicted[channel]);
        }
        assert!(matches!(a.finish(fb.as_bytes()), Err(Error::Consumed)));
    }
}

#[test]
fn exhaustive_small_masks_inputs_fixed_three_interval_two_channel_affine_table() {
    // 5456 consuming products, not exhaustive partitions, entropy or share splits.
    for bits in 2..=6 {
        let n = 1u64 << bits;
        for r in 0..n {
            for x in 0..n { run(bits, r, x, n - 1, true); }
        }
    }
}

#[test]
fn channels_and_dummy_splits_fullwidth_f19_boundary_neighborhoods() {
    let n = 1u64 << 50;
    // Numeric neighborhoods only: this is NOT the SiLU table/F19 error test.
    let signed = [-65i64, -64, -1, 0, 63, 64, -65473, -65472, 65535, 65536,
        -(1i64 << 37), (1i64 << 37) - 1];
    for r in [0, 1, n / 2, n - 1] {
        for value in signed {
            let x = (value as u64) & (n - 1);
            for split in [0, x, n - 1] {
                for affine in [false, true] { run(50, r, x, split, affine); }
            }
        }
    }
}

#[test]
fn constant_dummy_rows_and_arithmetic_bit_parity() {
    let budget = budget();
    for (boundaries, rows) in [
        (vec![0, 64], vec![[63, 1]]),
        (vec![0, 1, 32, 64], vec![[63, 1]; 3]),
        (vec![0, 16, 32, 64], vec![[1, 1], [1, 0], [0, 0]]),
    ] {
        let basis = JointBasis::new(6, &boundaries, &rows,
            [4; 32], [5; 32], &budget).unwrap();
        assert_eq!(basis.resources().unwrap().ring.dcf_evaluations_per_party, rows.len());
        for x in 0..64 {
            let [mut a, mut b] = basis.issue(context(), 0).unwrap();
            let fa = a.start(0).unwrap();
            let fb = b.start(x).unwrap();
            let ya = a.finish(fb.as_bytes()).unwrap();
            let yb = b.finish(fa.as_bytes()).unwrap();
            let expected = scan(&basis, x);
            for c in 0..2 {
                assert_eq!(ya.shares()[c].wrapping_add(yb.shares()[c]) & 63, expected[c]);
                if expected[c] <= 1 {
                    // For an actual arithmetic BIT sharing, parity is XOR
                    // because N is even. Not a membership proof for arbitrary v.
                    assert_eq!((ya.shares()[c] & 1) ^ (yb.shares()[c] & 1), expected[c]);
                }
            }
        }
    }
}

#[test]
fn cancellation_invalid_words_phase_role_nonce_context_and_replay_burn() {
    let budget = budget();
    let basis = basis(6, &budget);
    for offset in [0, 8, 40, 72, 73, 74, 75, 76] {
        let [mut a, mut b] = basis.issue(context(), 0).unwrap();
        let _own = a.start(0).unwrap();
        let peer = b.start(0).unwrap();
        let mut bad = peer.as_bytes().to_vec();
        bad[offset] ^= 1;
        assert!(a.advance_affine(&bad).is_err());
        assert!(matches!(a.advance_affine(peer.as_bytes()), Err(Error::Consumed)));
    }
    for phase in 0..3 {
        let [mut a, mut b] = basis.issue(context(), 0).unwrap();
        match phase {
            0 => {},
            1 => { let _ = a.start(0).unwrap(); },
            _ => {
                let fa = a.start(0).unwrap();
                let fb = b.start(0).unwrap();
                let _ = a.advance_affine(fb.as_bytes()).unwrap();
                let _ = b.advance_affine(fa.as_bytes()).unwrap();
            }
        }
        a.cancel();
        assert!(matches!(a.start(0), Err(Error::Consumed)));
    }
    let [mut a, _] = basis.issue(context(), 0).unwrap();
    assert!(matches!(a.start(64), Err(Error::Domain)));
    assert!(matches!(a.start(0), Err(Error::Consumed)));
    let [mut a, mut b] = basis.issue(context(), 0).unwrap();
    let _ = a.start(0).unwrap();
    let peer = b.start(0).unwrap();
    let mut bad = peer.as_bytes().to_vec();
    bad[HEADER..HEADER + 8].copy_from_slice(&64u64.to_le_bytes());
    assert!(matches!(a.finish(&bad), Err(Error::Domain)));
    assert!(matches!(a.finish(peer.as_bytes()), Err(Error::Consumed)));
    let [mut a, mut b] = basis.issue(context(), 0).unwrap();
    let _ = a.start(0).unwrap();
    let fb = b.start(0).unwrap();
    assert!(matches!(a.finish_affine(fb.as_bytes()), Err(Error::Context)));
    assert!(matches!(a.finish(fb.as_bytes()), Err(Error::Consumed)));
    // Different issuance, context, source/descriptor/table, lane or epoch all
    // affect peer validation. No import codec means persistent replay is absent.
    for variant in 0..6 {
        let [mut a, _] = basis.issue(context(), 0).unwrap();
        let _ = a.start(0).unwrap();
        let mut c = context();
        match variant { 1 => c.plan[0] ^= 1, 2 => c.session[0] ^= 1,
            3 => c.operation[0] ^= 1, 4 => c.first_tensor_index += 1, _ => {} }
        let [_, mut other] = basis.issue(c, u64::from(variant == 5)).unwrap();
        if variant != 0 { assert_ne!(a.binding, other.binding); }
        let f = other.start(0).unwrap();
        assert!(matches!(a.finish(f.as_bytes()), Err(Error::Context)));
    }
    for variant in 0..3 {
        let source = if variant == 0 { [6; 32] } else { [4; 32] };
        let descriptor = if variant == 1 { [6; 32] } else { [5; 32] };
        let rows = if variant == 2 { [[1, 1], [2, 63], [0, 0]] }
            else { [[63, 1], [2, 63], [0, 0]] };
        let changed = JointBasis::new(6, &[0, 1, 32, 64], &rows,
            source, descriptor, &budget).unwrap();
        assert_ne!(basis.digest, changed.digest);
    }
    // A noncanonical PRODUCT opening burns the same way as masked input.
    let [mut a, mut b] = basis.issue(context(), 0).unwrap();
    let fa = a.start(0).unwrap();
    let fb = b.start(0).unwrap();
    let _ = a.advance_affine(fb.as_bytes()).unwrap();
    let pb = b.advance_affine(fa.as_bytes()).unwrap();
    let mut bad = pb.as_bytes().to_vec();
    bad[HEADER..HEADER + 8].copy_from_slice(&64u64.to_le_bytes());
    assert!(matches!(a.finish_affine(&bad), Err(Error::Domain)));
    assert!(matches!(a.finish_affine(pb.as_bytes()), Err(Error::Consumed)));
}

#[test]
fn preadmission_refusal_allocation_failure_and_unchanged_interval_ceiling() {
    let budget = budget();
    let basis = basis(6, &budget);
    let cost = basis.resources().unwrap();
    assert_eq!(cost.dcf_path_levels_per_party, 18);
    assert_eq!(cost.triples_per_party, 1);
    assert_eq!(cost.output_bytes_per_party, 16);
    assert_eq!(cost.frame_backing_bytes_per_party, 2 * FRAME);
    assert_eq!(cost.combined_reservation_bytes,
        cost.source_reservation_bytes + cost.issuance_reservation_bytes);
    assert!(cost.private_payload_bytes_per_party > product::TRIPLE_BYTES);
    let blocker = budget.reserve(MAX_ALLOCATION_BYTES - budget.used_bytes()).unwrap();
    assert!(matches!(basis.issue_with(context(), 0, |_| panic!("RNG before admission")),
        Err(Error::Resource)));
    drop(blocker);
    budget.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(basis.issue_with(context(), 0, |_| panic!("RNG before allocation check")),
        Err(Error::Allocation)));
    budget.fail_after.store(-1, Ordering::SeqCst);
    let mut c = context(); c.first_tensor_index = u64::MAX;
    assert!(matches!(basis.issue(c, 0), Err(Error::Domain)));
    let baseline = budget.used_bytes();
    let [mut a, b] = basis.issue(context(), 0).unwrap();
    let frame = a.start(0).unwrap();
    drop(a); drop(b);
    assert!(budget.used_bytes() > baseline); // exported frame retains charge
    drop(frame);
    assert_eq!(budget.used_bytes(), baseline);
    let n = 1u64 << 50;
    let boundaries: Vec<_> = (0..=2049).map(|x| if x == 2049 { n } else { x }).collect();
    assert!(matches!(JointBasis::new(50, &boundaries, &vec![[0, 0]; 2049],
        [1; 32], [2; 32], &budget), Err(Error::Resource)));
    for (bits, boundaries, rows) in [
        (0, vec![0, 1], vec![[0, 0]]),
        (6, vec![0, 0, 64], vec![[0, 0]; 2]),
        (6, vec![0, 64], vec![[64, 0]]),
    ] {
        assert!(JointBasis::new(bits, &boundaries, &rows, [1; 32], [2; 32], &budget).is_err());
    }
}

#[test]
fn r2_exact_513_accepted_514_and_2049_refused_before_issue_entropy_or_charge() {
    let budget = budget();
    let source = SourceGuard::install(None, false);
    // N=4096 >= all tested interval counts. floor(i*N/count) gives strictly
    // ordered, full-ring endpoints, including count=2049 (not a width bug).
    let bits = 12;
    let n = 1u64 << bits;
    for count in [514usize, 2049] {
        let boundaries: Vec<u64> = (0..=count)
            .map(|i| (i as u64 * n) / count as u64).collect();
        let rows: Vec<[u64; 2]> = (0..count)
            .map(|i| [(17 * i as u64 + 3) % n, (5 * i as u64 + 1) % n]).collect();
        assert_eq!(boundaries.first(), Some(&0));
        assert_eq!(boundaries.last(), Some(&n));
        assert!(boundaries.windows(2).all(|w| w[0] < w[1]));
        assert!(rows.iter().flatten().all(|&v| v < n));
        let before = (budget.used_bytes(), budget.admitted_peak_bytes());
        budget.fail_after.store(0, Ordering::SeqCst);
        assert!(matches!(JointBasis::new(bits, &boundaries, &rows,
            [4; 32], [5; 32], &budget), Err(Error::Resource)));
        budget.fail_after.store(-1, Ordering::SeqCst);
        assert_eq!((budget.used_bytes(), budget.admitted_peak_bytes()), before);
        assert!(source.calls().is_empty()); // no owner exists to issue
    }
    let count = 513usize;
    let boundaries: Vec<u64> = (0..=count)
        .map(|i| (i as u64 * n) / count as u64).collect();
    let rows: Vec<[u64; 2]> = (0..count)
        .map(|i| [(17 * i as u64 + 3) % n, (5 * i as u64 + 1) % n]).collect();
    let basis = JointBasis::new(bits, &boundaries, &rows,
        [4; 32], [5; 32], &budget).unwrap();
    let resources = basis.resources().unwrap();
    assert_eq!(resources.ring.dcf_evaluations_per_party, count);
    assert_eq!(resources.dcf_path_levels_per_party, count * usize::from(bits));
    let baseline = budget.used_bytes();
    let [mut a, mut b] = basis.issue(context(), 0).unwrap();
    assert_eq!(source.calls(), [48]); // downstream draws not counted by this hook
    let fa = a.start(n - 1).unwrap();
    let fb = b.start(0).unwrap();
    let pa = a.advance_affine(fb.as_bytes()).unwrap();
    let pb = b.advance_affine(fa.as_bytes()).unwrap();
    let ya = a.finish_affine(pb.as_bytes()).unwrap();
    let yb = b.finish_affine(pa.as_bytes()).unwrap();
    let i = (count - 1) as u128;
    let modulus = u128::from(n);
    let expected = ((17 * i + 3) % modulus
        + ((5 * i + 1) % modulus) * u128::from(n - 1)) % modulus;
    assert_eq!((u128::from(ya.share()) + u128::from(yb.share())) % modulus, expected);
    drop((a, b, fa, fb, pa, pb, ya, yb));
    assert_eq!(budget.used_bytes(), baseline);
    drop(basis);
    assert_eq!(budget.used_bytes(), 0);
}

// A DEFINED SYNTHETIC fixture, NOT a paper SiLU table or an approximation-error
// claim. Public ring cuts contain actual F19 helper cuts N-65472 and 65536.
// Coefficients frozen here; direct oracle below never reads a compiled RingTable.
const SYNTHETIC_ROWS: [(i64, i64); 10] = [
    (-7, 1), (11, 0), (-29, 3), (4096, -1), (-33, 2),
    (17, 1), (-55, -2), (127, 0), (-4096, 1), (5, -3),
];
fn synthetic_basis(budget: &Budget) -> JointBasis<'_> {
    let n = 1u64 << 50;
    let boundaries = [0, 1, 64, 65536, 1 << 37, n / 2,
        n - (1 << 37), n - 65472, n - 64, n - 1, n];
    let rows = SYNTHETIC_ROWS.map(|(c, d)| [
        (i128::from(c).rem_euclid(i128::from(n))) as u64,
        (i128::from(d).rem_euclid(i128::from(n))) as u64,
    ]);
    JointBasis::new(50, &boundaries, &rows, [4; 32], [5; 32], budget).unwrap()
}
fn synthetic_direct_affine(signed: i64) -> u64 {
    // Explicit signed intervals, independent of unsigned cut lookup/eval/DCF.
    let index = if signed >= 0 {
        if signed == 0 { 0 } else if signed < 64 { 1 }
        else if signed < 65536 { 2 } else if signed < (1i64 << 37) { 3 } else { 4 }
    } else if signed < -(1i64 << 37) { 5 }
    else if signed < -65472 { 6 } else if signed < -64 { 7 }
    else if signed < -1 { 8 } else { 9 };
    let (c, d) = SYNTHETIC_ROWS[index];
    (i128::from(c) + i128::from(d) * i128::from(signed))
        .rem_euclid(1i128 << 50) as u64
}
fn synthetic_run(basis: &JointBasis<'_>, signed: i64, r: u64, split: u64)
    -> [(usize, u8); 4]
{
    let n = 1u64 << 50;
    let x = i128::from(signed).rem_euclid(i128::from(n)) as u64;
    let [mut a, mut b] = basis.issue_with(context(), 0, |bytes| {
        bytes.fill(0x39);
        bytes[..8].copy_from_slice(&r.to_le_bytes());
        Ok(())
    }).unwrap();
    let fa = a.start(split).unwrap();
    let fb = b.start((x + n - split) % n).unwrap();
    let pa = a.advance_affine(fb.as_bytes()).unwrap();
    let pb = b.advance_affine(fa.as_bytes()).unwrap();
    let shapes = [&fa, &fb, &pa, &pb].map(|f| (f.as_bytes().len(), f.as_bytes()[73]));
    let ya = a.finish_affine(pb.as_bytes()).unwrap();
    let yb = b.finish_affine(pa.as_bytes()).unwrap();
    assert_eq!((ya.share() + yb.share()) % n, synthetic_direct_affine(signed));
    assert!(matches!(a.finish_affine(pb.as_bytes()), Err(Error::Consumed)));
    shapes
}

#[test]
fn r2_synthetic_actual_f19_helper_cuts_negative_floor_and_wrap_direct_affine() {
    let budget = budget();
    let basis = synthetic_basis(&budget);
    let n = 1u64 << 50;
    assert!(basis.table.boundaries().contains(&(n - 65472)));
    assert!(basis.table.boundaries().contains(&65536));
    assert_eq!(basis.resources().unwrap().private_payload_bytes_per_party, 1372);
    // These are raw payload LOWER BOUNDS, not live-capacity or fit quotes.
    assert_eq!(1372u64 * 7_120_896, 9_769_869_312);
    assert_eq!(9_769_869_312u64 + 25_601_495_040, 35_371_364_352);
    assert!(35_371_364_352u64 > 32 * (1u64 << 30));
    for (x, floor) in [(-129i64, -3), (-128, -2), (-65, -2), (-64, -1),
        (-1, -1), (0, 0), (63, 0), (64, 1)] {
        assert_eq!(x.div_euclid(64), floor);
    }
    let inputs = [-(1i64 << 49), -(1i64 << 37) - 1, -(1i64 << 37),
        -65536, -65473, -65472, -65471, -129, -128, -127, -65, -64, -63,
        -2, -1, 0, 1, 63, 64, 65, 65535, 65536, 65537,
        (1i64 << 37) - 1, 1i64 << 37, (1i64 << 49) - 1];
    let baseline = budget.used_bytes();
    // 26 inputs * 2 masks * 3 splits = 156 consuming affine invocations.
    for signed in inputs {
        let x = i128::from(signed).rem_euclid(i128::from(n)) as u64;
        for r in [0, n - 1] {
            for split in [0, x, n - 1] {
                assert_eq!(synthetic_run(&basis, signed, r, split),
                    [(88, 0), (88, 0), (96, 1), (96, 1)]);
                assert_eq!(budget.used_bytes(), baseline);
            }
        }
    }
}

#[test]
fn r2_outside_signed38_envelope_same_public_schedule_not_membership_enforcement() {
    let budget = budget();
    let basis = synthetic_basis(&budget);
    let n = 1u64 << 50;
    // Adjacent inside/outside signed38 endpoints, all canonical ring words.
    // Success/fixed shapes demonstrate local schedule behavior ONLY, not D14
    // safe substitution, sticky validity, a membership proof or security.
    for signed in [-(1i64 << 37) - 1, -(1i64 << 37),
        (1i64 << 37) - 1, 1i64 << 37] {
        let x = i128::from(signed).rem_euclid(i128::from(n)) as u64;
        for split in [0, x, n - 1] {
            assert_eq!(synthetic_run(&basis, signed, n - 1, split),
                [(88, 0), (88, 0), (96, 1), (96, 1)]);
        }
    }
}

#[test]
fn r2_cross_owner_source_descriptor_table_rejection_with_matched_nonce() {
    let budget = budget();
    let original = basis(6, &budget);
    for variant in 0..3 {
        let source = if variant == 0 { [6; 32] } else { [4; 32] };
        let descriptor = if variant == 1 { [6; 32] } else { [5; 32] };
        let rows = if variant == 2 { [[1, 1], [2, 63], [0, 0]] }
            else { [[63, 1], [2, 63], [0, 0]] };
        let changed = JointBasis::new(6, &[0, 1, 32, 64], &rows,
            source, descriptor, &budget).unwrap();
        for affine in [false, true] {
            // TEST-ONLY deliberate nonce reuse isolates binding rejection. Not
            // an allowed production issuance policy or joint simulation proof.
            let fill = |bytes: &mut [u8]| { bytes.fill(0x39); Ok(()) };
            let [mut a, _] = original.issue_with(context(), 0, fill).unwrap();
            let [_, mut other] = changed.issue_with(context(), 0, fill).unwrap();
            assert_eq!(a.nonce, other.nonce);
            assert_ne!(a.binding, other.binding);
            let _own = a.start(0).unwrap();
            let peer = other.start(0).unwrap();
            if affine {
                assert!(matches!(a.advance_affine(peer.as_bytes()), Err(Error::Context)));
            } else {
                assert!(matches!(a.finish(peer.as_bytes()), Err(Error::Context)));
            }
            assert!(a.state.is_none());
            assert!(matches!(a.finish(peer.as_bytes()), Err(Error::Consumed)));
        }
    }
}

#[test]
fn r2_channel_and_affine_output_leases_remain_until_each_output_drop() {
    let budget = budget();
    let basis = basis(6, &budget);
    let baseline = budget.used_bytes();
    let [mut a, mut b] = basis.issue(context(), 0).unwrap();
    let fa = a.start(0).unwrap();
    let fb = b.start(1).unwrap();
    let ya = a.finish(fb.as_bytes()).unwrap();
    let yb = b.finish(fa.as_bytes()).unwrap();
    drop((a, b, fa, fb));
    let channel_bytes = size_of::<JointOutput<'_>>();
    assert_eq!(budget.used_bytes(), baseline + 2 * channel_bytes);
    drop(ya);
    assert_eq!(budget.used_bytes(), baseline + channel_bytes);
    drop(yb);
    assert_eq!(budget.used_bytes(), baseline);
    let [mut a, mut b] = basis.issue(context(), 0).unwrap();
    let fa = a.start(0).unwrap();
    let fb = b.start(1).unwrap();
    let pa = a.advance_affine(fb.as_bytes()).unwrap();
    let pb = b.advance_affine(fa.as_bytes()).unwrap();
    let ya = a.finish_affine(pb.as_bytes()).unwrap();
    let yb = b.finish_affine(pa.as_bytes()).unwrap();
    drop((a, b, fa, fb, pa, pb));
    let affine_bytes = size_of::<JointAffineOutput<'_>>();
    assert_eq!(budget.used_bytes(), baseline + 2 * affine_bytes);
    drop(ya);
    assert_eq!(budget.used_bytes(), baseline + affine_bytes);
    drop(yb);
    assert_eq!(budget.used_bytes(), baseline);
}

#[test]
fn r2_malformed_lengths_product_second_word_and_wrong_phase_burn_then_drop_charge() {
    let budget = budget();
    let basis = basis(6, &budget);
    let baseline = budget.used_bytes();
    for case in 0..12 {
        let [mut a, mut b] = basis.issue(context(), 0).unwrap();
        let fa = a.start(0).unwrap();
        let fb = b.start(1).unwrap();
        let mut pa = None;
        let mut pb = None;
        if case >= 4 {
            pa = Some(a.advance_affine(fb.as_bytes()).unwrap());
            pb = Some(b.advance_affine(fa.as_bytes()).unwrap());
        }
        if case < 4 {
            let mut bad = fb.as_bytes().to_vec();
            bad.resize([0, HEADER - 1, HEADER + 7, HEADER + 9][case], 0);
            assert!(matches!(a.advance_affine(&bad), Err(Error::Context)));
        } else {
            let peer = pb.as_ref().unwrap().as_bytes();
            let mut bad = peer.to_vec();
            match case {
                4..=7 => {
                    bad.resize([0, HEADER - 1, FRAME - 1, FRAME + 1][case - 4], 0);
                    assert!(matches!(a.finish_affine(&bad), Err(Error::Context)));
                },
                8 => {
                    bad[HEADER + 8..HEADER + 16].copy_from_slice(&64u64.to_le_bytes());
                    assert!(matches!(a.finish_affine(&bad), Err(Error::Domain)));
                },
                9 => {
                    bad[73] = 0;
                    assert!(matches!(a.finish_affine(&bad), Err(Error::Context)));
                },
                10 => { assert!(matches!(a.finish(peer), Err(Error::Context))); },
                _ => { assert!(matches!(a.start(0), Err(Error::Context))); },
            }
        }
        // state.take() destroys private key/triple owners on failure. The
        // conservative reservation stays with the empty party until Drop;
        // external frames retain their own split charge independently.
        assert!(a.state.is_none());
        assert!(matches!(a.finish(fb.as_bytes()), Err(Error::Consumed)));
        let retained = a.lease.bytes;
        let before_drop = budget.used_bytes();
        drop(a);
        assert_eq!(budget.used_bytes(), before_drop - retained);
        drop((b, fa, fb, pa, pb));
        assert_eq!(budget.used_bytes(), baseline);
    }
}

#[test]
fn r2_actual_issue_local_entropy_partial_failure_unwind_and_allocation_cleanup() {
    let budget = budget();
    let basis = basis(6, &budget);
    let baseline = budget.used_bytes();
    // Actual issue() call path, not merely the helper used by algebra fixtures.
    // 6 partial/full prefix lengths * Result/unwind = 12 failures.
    for prefix in [0, 1, 8, 16, 47, 48] {
        for unwind in [false, true] {
            let source = SourceGuard::install(Some(prefix), unwind);
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| basis.issue(context(), 0)));
            if unwind { assert!(result.is_err()); }
            else { assert!(matches!(result, Ok(Err(Error::Randomness)))); }
            assert_eq!(source.calls(), [48]);
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
    let source = SourceGuard::install(None, false);
    budget.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(basis.issue(context(), 0), Err(Error::Allocation)));
    budget.fail_after.store(-1, Ordering::SeqCst);
    assert!(source.calls().is_empty());
    assert_eq!(budget.used_bytes(), baseline);
    let parties = basis.issue(context(), 0).unwrap();
    assert_eq!(source.calls(), [48]);
    drop(parties);
    assert_eq!(budget.used_bytes(), baseline);
    // Downstream regressions below use a separate bounded controller. This
    // legacy test itself remains evidence only for the local 48-byte draw.
}

// SOURCE-ONLY: all following regressions are UNCOMPILED/UNRUN. These fixtures
// borrow the genuine OwnedBudget's original counter; the controller keeps Weak
// only. Expected charges include the complete conservative issuance envelope,
// not merely private output payloads or an arbitrary positive used-byte count.
use super::super::joint_basis_test_hooks as hooks;
use crate::shared_allocation::OwnedBudget;
use std::sync::Weak;

const ISSUE_EVENTS: [hooks::Kind; 17] = [
    hooks::Kind::JointEntropy,
    hooks::Kind::BoundaryReserve, hooks::Kind::BoundaryCapacity,
    hooks::Kind::CoefficientReserve, hooks::Kind::CoefficientCapacity,
    hooks::Kind::BoundaryReserve, hooks::Kind::BoundaryCapacity,
    hooks::Kind::CoefficientReserve, hooks::Kind::CoefficientCapacity,
    hooks::Kind::DcfReserve, hooks::Kind::DcfCapacity,
    hooks::Kind::DcfReserve, hooks::Kind::DcfCapacity,
    hooks::Kind::DcfEntropy, hooks::Kind::DcfEntropy,
    hooks::Kind::RingEntropy, hooks::Kind::TripleEntropy,
];

fn hook_config(owned: &OwnedBudget, expected_charge: usize, fault: Option<hooks::Fault>) -> hooks::Config {
    hooks::Config { counter: owned.test_counter_weak(),
        expected_identity: owned.budget().test_snapshot().0, expected_charge,
        // Before issue: one real reserve. Later phase installs: no new reserve.
        expected_updates: owned.budget().test_accounting_updates()
            + usize::from(expected_charge > owned.budget().used_bytes()), fault }
    // Counter continuity is not a proof of each private lease's ancestry.
    // No substitute source authority or caller work/material ledger is supplied.
}
fn checked_observation(guard: &hooks::Guard, owned: &OwnedBudget) -> hooks::Observation {
    let snapshot = guard.snapshot();
    hooks::assert_valid(&snapshot);
    assert!(snapshot.len > 0);
    assert!(snapshot.records[snapshot.len..].iter().all(Option::is_none));
    for record in &snapshot.records[..snapshot.len] {
        let record = record.as_ref().expect("missing event");
        assert_eq!(record.identity, owned.budget().test_snapshot().0);
        assert_eq!(record.used, record.expected);
        assert_eq!(record.updates, record.expected_updates);
    }
    snapshot
}
fn kind_count(snapshot: &hooks::Observation, kind: hooks::Kind) -> usize {
    snapshot.records[..snapshot.len].iter().flatten()
        .filter(|record| record.event.kind == kind).count()
}
fn assert_issue_prefix(snapshot: &hooks::Observation, count: usize) {
    assert!(snapshot.len >= count);
    for (record, expected) in snapshot.records[..count].iter().zip(&ISSUE_EVENTS[..count]) {
        assert_eq!(record.as_ref().unwrap().event.kind, *expected);
    }
}
fn assert_consumed(snapshot: &hooks::Observation, stop: usize, prefix: Option<usize>) {
    let mut expected = [0; 5];
    let mut dcf = 0;
    for (position, kind) in ISSUE_EVENTS[..stop].iter().enumerate() {
        let (index, len) = match kind {
            hooks::Kind::JointEntropy => (0, 48),
            hooks::Kind::DcfEntropy => { dcf += 1; (dcf, 16) },
            hooks::Kind::RingEntropy => (3, 16), hooks::Kind::TripleEntropy => (4, 40),
            _ => continue,
        };
        let record = snapshot.records[position].unwrap();
        assert_eq!(record.event.amount, len);
        expected[index] += if position + 1 == stop { prefix.unwrap_or(len) } else { len };
    }
    hooks::assert_consumption(snapshot.consumed, expected);
}

fn assert_private_entries(snapshot: &hooks::Observation, keys: usize, triples: usize) {
    assert_eq!(kind_count(snapshot, hooks::Kind::KeyDrop), keys);
    assert_eq!(kind_count(snapshot, hooks::Kind::CorrectionDrop), keys * 6);
    assert_eq!(kind_count(snapshot, hooks::Kind::TripleDrop), triples);
    for record in snapshot.records[..snapshot.len].iter().flatten() {
        match record.event.kind {
            hooks::Kind::KeyDrop => assert_eq!(record.event.amount, 6),
            hooks::Kind::TripleDrop => assert_eq!(record.event.amount, product::TRIPLE_BYTES),
            _ => {},
        }
    }
}
fn finish_owned_fixture(owned: OwnedBudget, weak: &Weak<Budget>, bootstrap: usize) {
    assert_eq!(owned.budget().used_bytes(), bootstrap);
    let quote = OwnedBudget::resources().unwrap();
    assert_eq!(bootstrap, quote.initial_reserved_bytes);
    let fixed = bootstrap.checked_sub(quote.owner_metadata_bytes).unwrap();
    assert_eq!(fixed, quote.arc_backing_bytes.checked_add(quote.bootstrap_transition_bytes).unwrap());
    // This temporary observer keeps the counter, NOT a root or a refund guard.
    let observer = weak.upgrade().unwrap();
    assert!(std::ptr::eq(&*observer, owned.budget()));
    drop(owned);
    assert_eq!(observer.used_bytes(), fixed); // fixed backing is deliberately live
    drop(observer);
    assert!(weak.upgrade().is_none()); // not a physical-deallocation/erasure proof
}

#[test]
fn r3_downstream_reserve_errors_and_unwinds_keep_original_full_reservation() {
    for kind in [hooks::Kind::BoundaryReserve, hooks::Kind::CoefficientReserve, hooks::Kind::DcfReserve] {
        for ordinal in 0..2 {
            for unwind in [false, true] {
                let owned = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
                let bootstrap = owned.budget().used_bytes();
                let weak = owned.test_counter_weak();
                let basis = basis(6, owned.budget());
                let cost = basis.resources().unwrap();
                let source = bootstrap.checked_add(cost.source_reservation_bytes).unwrap();
                assert_eq!(owned.budget().used_bytes(), source);
                let full = source.checked_add(cost.issuance_reservation_bytes).unwrap();
                let fault = hooks::Fault { kind, ordinal, prefix: 0, unwind };
                let expected_updates_after = owned.budget().test_accounting_updates() + 2;
                let guard = hooks::install(hook_config(&owned, full, Some(fault)));
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| basis.issue(context(), 0)));
                if unwind { hooks::assert_injected(&result, fault); }
                else { assert!(matches!(&result, Ok(Err(Error::Resource)))); }
                drop(result);
                assert_eq!(owned.budget().used_bytes(), source);
                assert_eq!(owned.budget().test_accounting_updates(), expected_updates_after);
                let snapshot = checked_observation(&guard, &owned);
                let stop = ISSUE_EVENTS.iter().enumerate().filter(|(_, event)| **event == kind)
                    .nth(ordinal).unwrap().0 + 1;
                assert_issue_prefix(&snapshot, stop);
                assert_eq!(snapshot.len, stop);
                assert_consumed(&snapshot, stop, None);
                assert_private_entries(&snapshot, 0, 0);
                if kind == hooks::Kind::DcfReserve && ordinal == 1 {
                    // The first Vec is EMPTY: no Correction::drop can observe its
                    // backing retirement. Only this POST-reserve capacity event
                    // proves its successful allocation; final baseline proves
                    // reservation release, not independently observed Vec Drop.
                    assert_eq!(kind_count(&snapshot, hooks::Kind::DcfCapacity), 1);
                    let capacity = snapshot.records[..snapshot.len].iter().flatten()
                        .find(|record| record.event.kind == hooks::Kind::DcfCapacity).unwrap();
                    let reserve = snapshot.records[..snapshot.len].iter().flatten()
                        .find(|record| record.event.kind == hooks::Kind::DcfReserve).unwrap();
                    assert!(capacity.event.amount >= reserve.event.amount);
                    assert!(capacity.event.amount > 0);
                }
                drop(basis);
                finish_owned_fixture(owned, &weak, bootstrap);
                // Guard still exists: its Weak and fixed records retained NO root.
                assert!(weak.upgrade().is_none());
            }
        }
    }
}

#[test]
fn r3_downstream_entropy_partial_errors_and_unwinds_keep_original_full_reservation() {
    for (kind, ordinal, len) in [
        (hooks::Kind::JointEntropy, 0, 48),
        (hooks::Kind::DcfEntropy, 0, 16), (hooks::Kind::DcfEntropy, 1, 16),
        (hooks::Kind::RingEntropy, 0, 16), (hooks::Kind::TripleEntropy, 0, 40),
    ] {
        // Four named prefixes, not exhaustive entropy tapes or prefix lengths.
        for prefix in [0, 1, len - 1, len] {
            for unwind in [false, true] {
                let owned = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
                let bootstrap = owned.budget().used_bytes();
                let weak = owned.test_counter_weak();
                let basis = basis(6, owned.budget());
                let cost = basis.resources().unwrap();
                let source = bootstrap.checked_add(cost.source_reservation_bytes).unwrap();
                assert_eq!(owned.budget().used_bytes(), source);
                let full = source.checked_add(cost.issuance_reservation_bytes).unwrap();
                let fault = hooks::Fault { kind, ordinal, prefix, unwind };
                let expected_updates_after = owned.budget().test_accounting_updates() + 2;
                let guard = hooks::install(hook_config(&owned, full, Some(fault)));
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| basis.issue(context(), 0)));
                if unwind { hooks::assert_injected(&result, fault); }
                else { assert!(matches!(&result, Ok(Err(Error::Randomness)))); }
                drop(result);
                assert_eq!(owned.budget().used_bytes(), source);
                assert_eq!(owned.budget().test_accounting_updates(), expected_updates_after);
                let snapshot = checked_observation(&guard, &owned);
                let stop = ISSUE_EVENTS.iter().enumerate().filter(|(_, event)| **event == kind)
                    .nth(ordinal).unwrap().0 + 1;
                assert_issue_prefix(&snapshot, stop);
                assert_consumed(&snapshot, stop, Some(prefix));
                // Ring/base and triple failures occur AFTER both DCF keys exist.
                let keys = if matches!(kind, hooks::Kind::RingEntropy | hooks::Kind::TripleEntropy) {
                    2
                } else { 0 };
                assert_private_entries(&snapshot, keys, 0);
                assert_eq!(snapshot.len, stop + keys * 7);
                assert_eq!(snapshot.records[stop - 1].unwrap().event.amount, len);
                drop(basis);
                finish_owned_fixture(owned, &weak, bootstrap);
            }
        }
    }
}

#[test]
fn r3_cancel_and_malformed_phases_observe_private_entries_before_lease_release() {
    for phase in 0..4 {
        let owned = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let bootstrap = owned.budget().used_bytes();
        let weak = owned.test_counter_weak();
        let basis = basis(6, owned.budget());
        let cost = basis.resources().unwrap();
        let source = bootstrap.checked_add(cost.source_reservation_bytes).unwrap();
        assert_eq!(owned.budget().used_bytes(), source);
        let full = source.checked_add(cost.issuance_reservation_bytes).unwrap();
        let guard = hooks::install(hook_config(&owned, full, None));
        let [mut a, mut b] = basis.issue(context(), 0).unwrap();
        guard.issuance_complete();
        let mut frames: [Option<JointFrame<'_>>; 4] = [None, None, None, None];
        match phase {
            0 => { a.cancel(); b.cancel(); },
            1 => {
                assert!(matches!(a.start(64), Err(Error::Domain)));
                b.cancel();
            },
            _ => {
                frames[0] = Some(a.start(0).unwrap());
                frames[1] = Some(b.start(1).unwrap());
                if phase == 2 {
                    assert!(matches!(a.advance_affine(&[]), Err(Error::Context)));
                } else {
                    frames[2] = Some(a.advance_affine(frames[1].as_ref().unwrap().as_bytes()).unwrap());
                    frames[3] = Some(b.advance_affine(frames[0].as_ref().unwrap().as_bytes()).unwrap());
                    assert!(matches!(a.finish_affine(&[]), Err(Error::Context)));
                }
                b.cancel();
            },
        }
        assert!(a.state.is_none() && b.state.is_none());
        assert!(matches!(a.start(0), Err(Error::Consumed)));
        assert!(matches!(b.start(0), Err(Error::Consumed)));
        let snapshot = checked_observation(&guard, &owned);
        assert_issue_prefix(&snapshot, ISSUE_EVENTS.len());
        assert_consumed(&snapshot, ISSUE_EVENTS.len(), None);
        assert_private_entries(&snapshot, 2, 2);
        assert_eq!(snapshot.len, ISSUE_EVENTS.len() + 16);
        assert_eq!(owned.budget().used_bytes(), full);
        let count = frames.iter().flatten().count();
        drop((a, b));
        assert_eq!(owned.budget().used_bytes(), source + count * size_of::<JointFrame<'_>>());
        for (remaining, frame) in frames.into_iter().flatten().enumerate() {
            drop(frame);
            assert_eq!(owned.budget().used_bytes(),
                source + (count - remaining - 1) * size_of::<JointFrame<'_>>());
        }
        drop(basis);
        finish_owned_fixture(owned, &weak, bootstrap);
    }
}

#[test]
fn r3_channel_drop_entries_adjust_for_explicit_frame_and_output_retirement() {
    let owned = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let bootstrap = owned.budget().used_bytes();
    let weak = owned.test_counter_weak();
    let basis = basis(6, owned.budget());
    let cost = basis.resources().unwrap();
    let source = bootstrap.checked_add(cost.source_reservation_bytes).unwrap();
    assert_eq!(owned.budget().used_bytes(), source);
    let full = source.checked_add(cost.issuance_reservation_bytes).unwrap();
    let guard = hooks::install(hook_config(&owned, full, None));
    let [mut a, mut b] = basis.issue(context(), 0).unwrap();
    guard.issuance_complete();
    let fa = a.start(0).unwrap();
    let fb = b.start(1).unwrap();
    let mut a_public = [0; FRAME];
    a_public[..fa.as_bytes().len()].copy_from_slice(fa.as_bytes());
    let ya = a.finish(fb.as_bytes()).unwrap();
    let snapshot = checked_observation(&guard, &owned);
    assert_issue_prefix(&snapshot, ISSUE_EVENTS.len());
    assert_consumed(&snapshot, ISSUE_EVENTS.len(), None);
    assert_private_entries(&snapshot, 1, 1);
    assert_eq!(snapshot.len, ISSUE_EVENTS.len() + 8);
    drop(guard);
    drop(fa);
    drop(ya);
    let adjusted = full.checked_sub(size_of::<JointFrame<'_>>()).unwrap()
        .checked_sub(size_of::<JointOutput<'_>>()).unwrap();
    assert_eq!(owned.budget().used_bytes(), adjusted);
    let guard = hooks::install(hook_config(&owned, adjusted, None));
    // Party B's key/corrections/triple enter destruction under EXACT remaining
    // original charge, not the issuance total before those explicit drops.
    let yb = b.finish(&a_public[..HEADER + 8]).unwrap();
    let snapshot = checked_observation(&guard, &owned);
    assert_private_entries(&snapshot, 1, 1);
    assert_eq!(snapshot.len, 8);
    drop((a, b, fb, yb));
    assert_eq!(owned.budget().used_bytes(), source);
    drop(basis);
    finish_owned_fixture(owned, &weak, bootstrap);
}

#[test]
fn r3_affine_drop_entries_adjust_for_explicit_frames_and_output_retirement() {
    let owned = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let bootstrap = owned.budget().used_bytes();
    let weak = owned.test_counter_weak();
    let basis = basis(6, owned.budget());
    let cost = basis.resources().unwrap();
    let source = bootstrap.checked_add(cost.source_reservation_bytes).unwrap();
    assert_eq!(owned.budget().used_bytes(), source);
    let full = source.checked_add(cost.issuance_reservation_bytes).unwrap();
    let guard = hooks::install(hook_config(&owned, full, None));
    let [mut a, mut b] = basis.issue(context(), 0).unwrap();
    guard.issuance_complete();
    let fa = a.start(0).unwrap();
    let fb = b.start(1).unwrap();
    let pa = a.advance_affine(fb.as_bytes()).unwrap();
    let pb = b.advance_affine(fa.as_bytes()).unwrap();
    let ya = a.finish_affine(pb.as_bytes()).unwrap();
    let snapshot = checked_observation(&guard, &owned);
    assert_issue_prefix(&snapshot, ISSUE_EVENTS.len());
    assert_consumed(&snapshot, ISSUE_EVENTS.len(), None);
    assert_private_entries(&snapshot, 2, 1);
    assert_eq!(snapshot.len, ISSUE_EVENTS.len() + 15);
    drop(guard);
    drop((fa, fb, ya));
    let adjusted = full.checked_sub(2 * size_of::<JointFrame<'_>>()).unwrap()
        .checked_sub(size_of::<JointAffineOutput<'_>>()).unwrap();
    assert_eq!(owned.budget().used_bytes(), adjusted);
    let guard = hooks::install(hook_config(&owned, adjusted, None));
    let yb = b.finish_affine(pa.as_bytes()).unwrap();
    let snapshot = checked_observation(&guard, &owned);
    assert_private_entries(&snapshot, 0, 1);
    assert_eq!(snapshot.len, 1);
    drop((a, b, pa, pb, yb));
    assert_eq!(owned.budget().used_bytes(), source);
    drop(basis);
    finish_owned_fixture(owned, &weak, bootstrap);
}

#[test]
fn r3_legacy_joint_controller_exclusion_does_not_modify_either_controller() {
    let owned = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let bootstrap = owned.budget().used_bytes();
    let weak = owned.test_counter_weak();
    {
        let legacy = SourceGuard::install(None, false);
        hooks::assert_install_conflict(std::panic::catch_unwind(AssertUnwindSafe(|| {
            hooks::install(hook_config(&owned, bootstrap, None))
        })), "assertion failed: !super::joint_basis::test_source_active()");
        assert!(!hooks::active());
        assert!(legacy.calls().is_empty());
    }
    {
        let guard = hooks::install(hook_config(&owned, bootstrap, None));
        hooks::assert_install_conflict(std::panic::catch_unwind(|| SourceGuard::install(None, false)),
            "assertion failed: !super::super::joint_basis_test_hooks::active()");
        assert!(!source_active());
        assert!(hooks::active());
        assert!(!guard.snapshot().failed);
        assert_eq!(guard.snapshot().len, 0);
        hooks::assert_install_conflict(std::panic::catch_unwind(AssertUnwindSafe(|| {
            hooks::install(hook_config(&owned, bootstrap, None))
        })), "assertion failed: !active()");
        assert!(!guard.snapshot().failed);
        assert_eq!(guard.snapshot().len, 0);
    }
    finish_owned_fixture(owned, &weak, bootstrap);
}
