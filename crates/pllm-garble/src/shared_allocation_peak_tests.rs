use super::*;

#[test]
fn reservation_growth_shrink_and_drop_retain_peak() {
    let budget = Budget::new(256).unwrap();
    assert_eq!(budget.admitted_peak_bytes(), 0);
    let metadata = size_of::<ExternalCharge<'_>>();
    let mut charge = budget.charge_external(64).unwrap();
    let extra = budget.reserve(7).unwrap();
    assert_eq!(budget.admitted_peak_bytes(), metadata + 71);
    drop(extra);
    charge.resize_payload(96).unwrap();
    assert_eq!(budget.admitted_peak_bytes(), metadata + 96);
    charge.resize_payload(0).unwrap();
    assert_eq!(budget.used_bytes(), metadata);
    drop(charge);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(budget.admitted_peak_bytes(), metadata + 96);
}

#[test]
fn phase_admission_is_not_reservation_and_survives_same_counter_transfers() {
    let budget = Budget::new(256).unwrap();
    budget.admit_peak(192).unwrap();
    assert_eq!(budget.used_bytes(), 0);
    let mut charge = budget.charge_external(64).unwrap();
    let used = budget.used_bytes();
    let child = charge.split_payload(&budget, 31).unwrap();
    assert!(std::ptr::eq(child.budget, &budget));
    assert_eq!(budget.used_bytes(), used);
    budget.admit_peak(0).unwrap();
    budget.admit_peak(100).unwrap();
    drop(child);
    drop(charge);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(budget.admitted_peak_bytes(), 192);

    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let owned = OwnedBudget::new(baseline + 256).unwrap();
    owned.budget().admit_peak(baseline + 224).unwrap();
    let mut parent = owned.charge_external(64).unwrap();
    let child = parent.split_owned(8).unwrap();
    assert!(child.is_for(owned.budget()));
    let native = owned.reserve_native(1).unwrap();
    assert!(std::ptr::eq(native.budget.budget(), owned.budget()));
    drop((native, child, parent));
    assert_eq!(owned.budget().used_bytes(), baseline);
    assert_eq!(owned.budget().admitted_peak_bytes(), baseline + 224);
}

#[test]
fn rejected_limit_and_checked_add_overflow_preserve_both_counters() {
    let budget = Budget::new(64).unwrap();
    let lease = budget.reserve(1).unwrap();
    budget.admit_peak(32).unwrap();
    for bytes in [65, usize::MAX] {
        assert_eq!(budget.admit_peak(bytes), Err(Error::Resource));
        assert!(matches!(budget.reserve(bytes), Err(Error::Resource)));
        assert!(matches!(
            budget.charge_external(bytes),
            Err(Error::Resource)
        ));
        assert_eq!(budget.used_bytes(), 1);
        assert_eq!(budget.admitted_peak_bytes(), 32);
    }
    budget.admit_peak(64).unwrap();
    assert_eq!(budget.used_bytes(), 1);
    drop(lease);
    assert_eq!(budget.admitted_peak_bytes(), 64);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn bootstrap_uses_actual_budget_layout_and_refuses_before_allocation() {
    // Independent concrete slots, not a sum reconstructed from quote fields.
    let backing = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<Budget>())
        .unwrap()
        .0
        .pad_to_align()
        .size();
    let transitions = size_of::<Budget>()
        + size_of::<Result<Budget, Error>>()
        + size_of::<OwnedBudgetResources>()
        + size_of::<Result<OwnedBudgetResources, Error>>()
        + size_of::<Result<OwnedBudget, Error>>()
        + size_of::<Arc<Budget>>()
        + size_of::<Result<(), Error>>();
    let exact = backing + transitions + size_of::<OwnedBudget>();
    assert_eq!(
        OwnedBudget::resources().unwrap(),
        OwnedBudgetResources {
            arc_backing_bytes: backing,
            bootstrap_transition_bytes: transitions,
            owner_metadata_bytes: size_of::<OwnedBudget>(),
            initial_reserved_bytes: exact,
        }
    );
    for (limit, expected) in [(exact - 1, Error::Resource), (exact, Error::Allocation)] {
        let budget = Budget::new(limit).unwrap();
        budget.fail_after.store(0, Ordering::SeqCst);
        assert!(matches!(OwnedBudget::from_fresh_budget(budget), Err(e) if e == expected));
    }
    let owned = OwnedBudget::new(exact).unwrap();
    assert_eq!(owned.budget().used_bytes(), exact);
    assert_eq!(owned.budget().admitted_peak_bytes(), exact);
    assert!(owned.try_clone().is_err());
    assert_eq!(owned.budget().admitted_peak_bytes(), exact);
}
