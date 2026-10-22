use pllm_garble::shared_allocation::{Budget, Error, ExternalCharge, MAX_ALLOCATION_BYTES};
use std::mem::size_of;
use std::sync::Barrier;

const METADATA: usize = size_of::<ExternalCharge<'static>>();

#[test]
fn exact_limit_and_one_byte_short_include_guard_metadata() {
    let payload = 31;
    let exact = Budget::new(payload + METADATA).unwrap();
    let guard = exact.charge_external(payload).unwrap();
    assert_eq!(guard.payload_bytes(), payload);
    assert_eq!(guard.charged_bytes(), payload + METADATA);
    assert_eq!(exact.used_bytes(), payload + METADATA);
    let short = Budget::new(payload + METADATA - 1).unwrap();
    assert!(matches!(
        short.charge_external(payload),
        Err(Error::Resource)
    ));
    assert_eq!(short.used_bytes(), 0);
    drop(guard);
    assert_eq!(exact.used_bytes(), 0);
}

#[test]
fn zero_payload_still_requires_metadata_until_drop() {
    let budget = Budget::new(METADATA).unwrap();
    let mut guard = budget.charge_external(0).unwrap();
    guard.resize_payload(0).unwrap();
    assert_eq!(guard.payload_bytes(), 0);
    assert_eq!(guard.charged_bytes(), METADATA);
    assert!(matches!(budget.charge_external(0), Err(Error::Resource)));
    drop(guard);
    assert_eq!(budget.used_bytes(), 0);
    let short = Budget::new(METADATA - 1).unwrap();
    assert!(matches!(short.charge_external(0), Err(Error::Resource)));
}

#[test]
fn maximum_budget_includes_metadata_and_rejects_overflow() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for payload in [usize::MAX, usize::MAX - METADATA + 1, MAX_ALLOCATION_BYTES] {
        assert!(matches!(
            budget.charge_external(payload),
            Err(Error::Resource)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
    let guard = budget
        .charge_external(MAX_ALLOCATION_BYTES - METADATA)
        .unwrap();
    assert_eq!(guard.charged_bytes(), MAX_ALLOCATION_BYTES);
    assert_eq!(budget.used_bytes(), MAX_ALLOCATION_BYTES);
    drop(guard);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn failed_growth_and_overflow_preserve_original_charge() {
    let budget = Budget::new(METADATA + 20).unwrap();
    let mut guard = budget.charge_external(10).unwrap();
    for payload in [21, usize::MAX, usize::MAX - METADATA + 1] {
        assert_eq!(guard.resize_payload(payload), Err(Error::Resource));
        assert_eq!(guard.payload_bytes(), 10);
        assert_eq!(guard.charged_bytes(), METADATA + 10);
        assert_eq!(budget.used_bytes(), METADATA + 10);
    }
    guard.resize_payload(20).unwrap();
    assert_eq!(guard.payload_bytes(), 20);
    assert_eq!(budget.used_bytes(), METADATA + 20);
    guard.resize_payload(20).unwrap();
    assert_eq!(budget.used_bytes(), METADATA + 20);
    guard.resize_payload(0).unwrap();
    assert_eq!(guard.charged_bytes(), METADATA);
    assert_eq!(budget.used_bytes(), METADATA);
    drop(guard);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn guards_release_independently_and_compare_budget_identity() {
    let budget = Budget::new(2 * METADATA + 30).unwrap();
    let other = Budget::new(2 * METADATA + 30).unwrap();
    let mut first = budget.charge_external(10).unwrap();
    let second = budget.charge_external(20).unwrap();
    assert!(first.is_for(&budget));
    assert!(second.is_for(&budget));
    assert!(!first.is_for(&other));
    assert_eq!(first.resize_payload(11), Err(Error::Resource));
    assert_eq!(first.payload_bytes(), 10);
    drop(second);
    assert_eq!(budget.used_bytes(), METADATA + 10);
    first.resize_payload(30).unwrap();
    assert_eq!(budget.used_bytes(), METADATA + 30);
    assert_eq!(other.used_bytes(), 0);
    drop(first);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn moving_guard_preserves_exact_budget_identity_and_metadata() {
    fn move_guard(guard: ExternalCharge<'_>) -> ExternalCharge<'_> {
        guard
    }
    let budget = Budget::new(METADATA + 1).unwrap();
    let foreign = Budget::new(METADATA + 1).unwrap();
    let mut guard = move_guard(budget.charge_external(1).unwrap());
    assert!(guard.is_for(&budget));
    assert!(!guard.is_for(&foreign));
    guard.resize_payload(0).unwrap();
    assert_eq!(guard.charged_bytes(), METADATA);
    assert_eq!(budget.used_bytes(), METADATA);
    assert_eq!(foreign.used_bytes(), 0);
    drop(guard);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn cancelled_work_does_not_release_retained_owner_charge() {
    struct Owner<'a> {
        // Rust drops fields in declaration order: payload before accounting.
        payload: Vec<u8>,
        charge: ExternalCharge<'a>,
    }
    let budget = Budget::new(METADATA + 16).unwrap();
    let charge = budget.charge_external(16).unwrap();
    let owner = Owner {
        payload: vec![7; 16],
        charge,
    };
    let work: Box<dyn FnOnce() -> usize + '_> = Box::new(|| owner.payload.len());
    // Abandon the work while retaining the actual owner.
    drop(work);
    assert_eq!(owner.charge.payload_bytes(), 16);
    assert_eq!(budget.used_bytes(), METADATA + 16);
    assert!(matches!(budget.charge_external(0), Err(Error::Resource)));
    drop(owner);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn concurrent_growth_admits_only_one_increment_and_preserves_loser() {
    fn grow<'a>(
        mut guard: ExternalCharge<'a>,
        start: &Barrier,
        finish: &Barrier,
    ) -> (ExternalCharge<'a>, Result<(), Error>) {
        start.wait();
        let result = guard.resize_payload(11);
        finish.wait();
        (guard, result)
    }

    let budget = Budget::new(2 * METADATA + 12).unwrap();
    let first = budget.charge_external(1).unwrap();
    let second = budget.charge_external(1).unwrap();
    let start = Barrier::new(3);
    let finish = Barrier::new(3);
    let start = &start;
    let finish = &finish;
    std::thread::scope(|scope| {
        let first = scope.spawn(move || grow(first, start, finish));
        let second = scope.spawn(move || grow(second, start, finish));
        start.wait();
        finish.wait();
        assert_eq!(budget.used_bytes(), 2 * METADATA + 12);
        let (first, first_result) = first.join().unwrap();
        let (second, second_result) = second.join().unwrap();
        assert_eq!(
            usize::from(first_result.is_ok()) + usize::from(second_result.is_ok()),
            1
        );
        for (guard, result) in [(&first, first_result), (&second, second_result)] {
            match result {
                Ok(()) => assert_eq!(guard.payload_bytes(), 11),
                Err(Error::Resource) => assert_eq!(guard.payload_bytes(), 1),
                result => panic!("unexpected growth result: {result:?}"),
            }
        }
        drop(first);
        drop(second);
    });
    assert_eq!(budget.used_bytes(), 0);
}
