use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use zeroize::Zeroize;

thread_local! {
    static WIPES: std::cell::RefCell<[usize; 6]> = const { std::cell::RefCell::new([0; 6]) };
    static PANIC_ROW: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}
pub(super) fn observe_wipe(kind: usize, len: usize, all_zero: bool) {
    assert!(all_zero);
    assert!(len > 0); // inspect live storage, not a cleared empty Vec
    WIPES.with(|counts| counts.borrow_mut()[kind] += 1);
}
pub(super) fn construction_checkpoint(row: usize) {
    PANIC_ROW.with(|at| { if at.get() == Some(row) { panic!("test-only partial issuance unwind"); } });
}
fn reset_wipes() { WIPES.with(|counts| *counts.borrow_mut() = [0; 6]); }
fn wipes() -> [usize; 6] { WIPES.with(|counts| *counts.borrow()) }

struct Fixed(u8);
impl Entropy for Fixed {
    fn fill(&mut self, bytes: &mut [u8]) -> Result<(), Error> {
        // Test-only nonuniform masks exercise both signs and wrapping, never
        // exported through a production configuration or runtime injection API.
        for (i, b) in bytes.iter_mut().enumerate() { *b = (i as u8).wrapping_mul(73).wrapping_add(self.0); }
        Ok(())
    }
}
struct Fail { calls: usize }
impl Entropy for Fail {
    fn fill(&mut self, bytes: &mut [u8]) -> Result<(), Error> {
        self.calls += 1;
        bytes[..101].fill(0xff);
        Err(Error::Entropy)
    }
}
fn session(seed: u8) -> Session { issue(&mut Fixed(seed), MAX_LOGICAL_BYTES, u64::MAX).unwrap() }

#[test]
fn residual_half_ties_endpoints_and_overflow_wrap() {
    let h = 1u64 << 31;
    let q = 1u128 << 32;
    for (a, expected) in [(0, 0), (h - 1, h as i128 - 1), (h, -(h as i128)),
        (h + 1, -(h as i128) + 1), (u64::MAX, -1), (u64::MAX - h + 1, -(h as i128))] {
        let (k, r) = quotient_residual(a);
        assert_eq!(r, expected);
        assert!((-(h as i128)..h as i128).contains(&r));
        assert_eq!(((k * q) % (1u128 << 64)) as u64, a.wrapping_add(h) & !((1u64 << 32) - 1));
    }
}

#[test]
fn all_messages_both_noise_signs_independent_phase_and_decode_counts() {
    for seed in [0, 1, 127, 255] {
        let s = session(seed);
        assert!(s.oracle.errors.iter().any(|&e| e == -1));
        assert!(s.oracle.errors.iter().any(|&e| e == 1));
        let aggregate = s.finish(0).unwrap();
        assert_eq!(aggregate.decoded_matches, 4);
        assert_eq!(aggregate.phase_equalities, 4);
        assert!(aggregate.maximum_centered_error <= ERROR_BOUND);
    }
    assert!(ERROR_BOUND < 1u64 << 61);
    for m in 0..4 { for e in [-(ERROR_BOUND as i64), -1, 1, ERROR_BOUND as i64] {
        let phase = ((m as u64) << 62).wrapping_add(e as u64);
        assert_eq!(decode(phase), m as u64);
        assert_eq!(centered_error(phase, m), e.unsigned_abs());
    } }
}

#[test]
fn zero_digits_skip_rows_and_noise_including_rounded_wrap() {
    let mut s = session(0);
    for mask in [0, u64::MAX] {
        let mut ct = Ciphertext { id: Arc::clone(&s.input.id), role: Role::Input,
            words: Zeroizing::new(vec![mask; 9]) };
        ct.words[8] = 17;
        let result = s.evaluator.switch(&ct).unwrap();
        assert_eq!(&result.words[..4], &[0; 4]);
        assert_eq!(result.words[4], 17);
    }
}

#[test]
fn complete_selected_row_once_not_digit_scaled() {
    let mut s = session(1);
    let mut ct = Ciphertext { id: Arc::clone(&s.input.id), role: Role::Input,
        words: Zeroizing::new(vec![0; 9]) };
    ct.words[0] = 15u64 << 60;
    ct.words[8] = 123;
    let row = &s.evaluator.table[14 * 5..15 * 5];
    let expected: Vec<u64> = (0..5).map(|k| (if k == 4 { 123u64 } else { 0 }).wrapping_sub(row[k])).collect();
    let result = s.evaluator.switch(&ct).unwrap();
    assert_eq!(&*result.words, &expected);
}

#[test]
fn cross_session_and_role_refuse_before_arithmetic() {
    let mut a = session(0);
    let b = session(0); // identical test bytes do not forge local identity
    let before = a.evaluator.table.to_vec();
    assert!(matches!(a.evaluator.switch(&b.ciphertexts[0]), Err(Error::Binding)));
    assert_eq!(&*a.evaluator.table, &before);
    assert!(matches!(a.evaluator.switch(&a.ciphertexts[0]), Err(Error::Terminal)));
    let mut a = session(0);
    let output = a.evaluator.switch(&a.ciphertexts[0]).unwrap();
    assert!(matches!(a.evaluator.switch(&output), Err(Error::Binding)));
    assert_eq!(a.output.phase(&b.ciphertexts[0]), Err(Error::Binding));
    assert_eq!(a.output.phase(&output), Err(Error::Terminal));
    let a = session(0);
    assert_eq!(a.output.phase(&a.ciphertexts[0]), Err(Error::Binding));
    assert_eq!(a.oracle.expected(&b.input, &a.ciphertexts[0], 0), Err(Error::Binding));
    assert_eq!(a.oracle.expected(&a.input, &a.ciphertexts[0], 0), Err(Error::Terminal));
}

#[test]
fn checked_inventory_exact_and_one_short_refuse_before_rng() {
    let q = preflight(MAX_LOGICAL_BYTES, u64::MAX).unwrap();
    assert_eq!(REQUESTED_OS_BYTES, 31952);
    assert_eq!(TABLE_BYTES, 38400);
    assert_eq!(PAYLOAD_BYTES, 448);
    assert_eq!(TABLE_BYTES + ROWS + 12 + PAYLOAD_BYTES + REQUESTED_OS_BYTES, 71772);
    assert_eq!(preflight(q.logical_bytes, q.work_units), Ok(q));
    let exact = issue(&mut Fixed(9), q.logical_bytes, q.work_units).unwrap().finish(0).unwrap();
    assert_eq!((exact.decoded_matches, exact.phase_equalities), (4, 4));
    let mut rng = Fail { calls: 0 };
    assert!(matches!(issue(&mut rng, q.logical_bytes - 1, q.work_units), Err(Error::Budget)));
    assert!(matches!(issue(&mut rng, q.logical_bytes, q.work_units - 1), Err(Error::Budget)));
    assert_eq!(rng.calls, 0);
    assert_eq!(inventory(usize::MAX), Err(Error::Overflow));
    assert!(matches!(issue(&mut rng, q.logical_bytes, q.work_units), Err(Error::Entropy)));
    assert_eq!(rng.calls, 1); // no retry/refund
}

#[test]
fn terminal_early_drop_partial_fill_and_bounded_unwind_wipes() {
    let mut s = session(2);
    s.evaluator.terminal = true;
    assert!(matches!(s.evaluator.switch(&s.ciphertexts[0]), Err(Error::Terminal)));
    s.terminal = true;
    assert!(matches!(s.finish(0), Err(Error::Terminal)));
    let early = session(3);
    reset_wipes();
    drop(early); // observe actual early Drop paths, not synthetic freed-memory reads
    let counts = wipes();
    assert_eq!(&counts[..4], &[1, 1, 1, 1]);
    assert_eq!(counts[4], 4);
    reset_wipes();
    PANIC_ROW.with(|at| at.set(Some(17)));
    let partial = catch_unwind(AssertUnwindSafe(|| { let _ = session(5); }));
    PANIC_ROW.with(|at| at.set(None));
    assert!(partial.is_err());
    let counts = wipes();
    assert_eq!(&counts[..4], &[1, 1, 1, 1]);
    assert!(counts[4] >= 18); // initialized row owners, including in-flight row
    reset_wipes();
    assert!(matches!(issue(&mut Fail { calls: 0 }, MAX_LOGICAL_BYTES, u64::MAX), Err(Error::Entropy)));
    assert_eq!(wipes()[5], 1);
    // Observe the same zeroize/wipe mechanisms on still-live backing, rather
    // than reading freed memory (UB). Partial initialized tails included.
    let mut backing = Zeroizing::new(vec![0u64; ROWS * 5]);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let guard = WipeGuard::new(&mut backing);
        guard.words[..17].fill(u64::MAX);
        panic!("partial table unwind");
    }));
    assert!(result.is_err());
    assert!(backing.iter().all(|&v| v == 0));
    let mut bytes = Zeroizing::new(vec![0u8; REQUESTED_OS_BYTES]);
    assert_eq!(Fail { calls: 0 }.fill(&mut bytes), Err(Error::Entropy));
    bytes.as_mut_slice().zeroize();
    assert!(bytes.iter().all(|&v| v == 0));
}

#[test]
fn mismatch_never_returns_partial_aggregate() {
    let mut s = session(4);
    s.oracle.input_errors[3] = 100;
    assert!(matches!(s.finish(0), Err(Error::Verification)));
}

#[test]
fn exact_maximum_bound_all_selected_rows_negative_half_ties() {
    for m in 0..4 {
        let mut s = session(6);
        s.input.secret.fill(1);
        s.oracle.errors.fill(1);
        s.oracle.input_errors.fill(-1);
        // Independently construct valid rows with zero masks and +1 noise.
        // Test-only fixture edits; no production entropy/noise injection API.
        for i in 0..8 { for j in 0..8 { for d in 1..16 {
            let r = (i * 8 + j) * 15 + d - 1;
            s.evaluator.table[r * 5..r * 5 + 4].fill(0);
            s.evaluator.table[r * 5 + 4] = ((d as u64) << (60 - 4 * j)).wrapping_add(1);
        } } }
        let a = 0xffff_fffe_8000_0000u64;
        let mut words = Zeroizing::new(vec![a; 9]);
        words[8] = ((m as u64) << 62).wrapping_sub(1).wrapping_add(a.wrapping_mul(8));
        let ct = Ciphertext { id: Arc::clone(&s.input.id), role: Role::Input, words };
        let expected = s.oracle.expected(&s.input, &ct, m).unwrap();
        assert_eq!(expected.1, ERROR_BOUND);
        let output = s.evaluator.switch(&ct).unwrap();
        let actual = s.output.phase(&output).unwrap();
        assert_eq!(actual, expected.0);
        assert_eq!(decode(actual), m as u64);
        assert_eq!(centered_error(actual, m), ERROR_BOUND);
    }
}
