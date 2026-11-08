use super::*;

#[test]
fn reserve_update_checks_overflow_limit_and_exact_fit() {
    assert_eq!(reserve_update(1, usize::MAX)(usize::MAX), None);
    assert_eq!(reserve_update(2, 10)(9), None);
    assert_eq!(reserve_update(2, 10)(8), Some(10));
}

#[test]
fn exact_limit_reservation_updates_once_and_drop_only_refunds_usage() {
    let budget = Budget::new(10).unwrap();
    let first = budget.reserve(2).unwrap();
    let updates = budget.test_accounting_updates();
    let exact = budget.reserve(8).unwrap();
    assert_eq!(budget.used_bytes(), 10);
    assert_eq!(budget.admitted_peak_bytes(), 10);
    assert_eq!(budget.test_accounting_updates(), updates + 1);
    drop(exact);
    assert_eq!(budget.used_bytes(), 2);
    assert_eq!(budget.admitted_peak_bytes(), 10);
    assert_eq!(budget.test_accounting_updates(), updates + 2);
    drop(first);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(budget.admitted_peak_bytes(), 10);
}

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
    let updates = budget.test_accounting_updates();
    for bytes in [65, usize::MAX] {
        assert_eq!(budget.admit_peak(bytes), Err(Error::Resource));
        assert!(matches!(budget.reserve(bytes), Err(Error::Resource)));
        assert!(matches!(
            budget.charge_external(bytes),
            Err(Error::Resource)
        ));
        assert_eq!(budget.used_bytes(), 1);
        assert_eq!(budget.admitted_peak_bytes(), 32);
        assert_eq!(budget.test_accounting_updates(), updates);
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

#[test]
fn bootstrap_extension_reserves_once_without_changing_resources() {
    let resources = OwnedBudget::resources().unwrap();
    let baseline = resources.initial_reserved_bytes;
    let extra = 37;
    let owned = OwnedBudget::new(baseline + extra).unwrap();
    let identity = owned.budget() as *const Budget;
    assert_eq!(owned.budget().used_bytes(), baseline);
    let updates = owned.budget().test_accounting_updates();

    owned.reserve_bootstrap(extra).unwrap();

    assert_eq!(owned.budget() as *const Budget, identity);
    assert_eq!(owned.budget().used_bytes(), baseline + extra);
    assert_eq!(owned.budget().admitted_peak_bytes(), baseline + extra);
    assert_eq!(owned.budget().test_accounting_updates(), updates + 1);
    assert_eq!(OwnedBudget::resources().unwrap(), resources);
}

#[test]
fn bootstrap_extension_rejections_are_atomic_and_exact_remaining_fits() {
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let remaining = 23;
    let limit = baseline + remaining;
    let owned = OwnedBudget::new(limit).unwrap();
    assert!(owned.budget().used_bytes() > 0);
    let used = owned.budget().used_bytes();
    let peak = owned.budget().admitted_peak_bytes();
    let updates = owned.budget().test_accounting_updates();
    for bytes in [usize::MAX, remaining + 1] {
        assert_eq!(owned.reserve_bootstrap(bytes), Err(Error::Resource));
        assert_eq!(owned.budget().used_bytes(), used);
        assert_eq!(owned.budget().admitted_peak_bytes(), peak);
        assert_eq!(owned.budget().test_accounting_updates(), updates);
        assert_eq!(owned.budget().limit, limit);
    }

    owned.reserve_bootstrap(remaining).unwrap();
    assert_eq!(owned.budget().used_bytes(), limit);
    assert_eq!(owned.budget().admitted_peak_bytes(), limit);
    assert_eq!(owned.budget().test_accounting_updates(), updates + 1);
    assert_eq!(owned.budget().limit, limit);
    assert_eq!(owned.reserve_bootstrap(1), Err(Error::Resource));
    assert_eq!(owned.budget().used_bytes(), limit);
    assert_eq!(owned.budget().admitted_peak_bytes(), limit);
    assert_eq!(owned.budget().test_accounting_updates(), updates + 1);
}

#[test]
fn bootstrap_extensions_accumulate_and_ordinary_owners_refund_only_their_bytes() {
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let root_metadata = size_of::<OwnedBudget>();
    let charge_metadata = size_of::<OwnedExternalCharge>();
    let extra = 11 + 19;
    let charged = 2 * charge_metadata + 7;
    let limit = baseline + extra + root_metadata + charged;
    let owned = OwnedBudget::new(limit).unwrap();
    let identity = owned.budget() as *const Budget;
    let updates = owned.budget().test_accounting_updates();
    owned.reserve_bootstrap(11).unwrap();
    owned.reserve_bootstrap(19).unwrap();
    assert_eq!(owned.budget().used_bytes(), baseline + extra);
    assert_eq!(owned.budget().test_accounting_updates(), updates + 2);

    let sibling = owned.try_clone().unwrap();
    let mut parent = owned.charge_external(charge_metadata + 7).unwrap();
    assert_eq!(owned.budget().used_bytes(), limit);
    let updates = owned.budget().test_accounting_updates();
    let child = parent.split_owned(7).unwrap();
    assert!(child.is_for(owned.budget()));
    assert_eq!(owned.budget().used_bytes(), limit);
    assert_eq!(owned.budget().test_accounting_updates(), updates);
    drop(parent);
    assert_eq!(owned.budget().used_bytes(), limit - charge_metadata);
    drop(sibling);
    assert_eq!(
        owned.budget().used_bytes(),
        baseline + extra + charge_metadata + 7
    );
    drop(child);
    assert_eq!(owned.budget().used_bytes(), baseline + extra);

    let survivor = owned.reserve_native(0).unwrap();
    drop(owned);
    assert_eq!(survivor.budget.budget() as *const Budget, identity);
    assert_eq!(
        survivor.budget.budget().used_bytes(),
        baseline - root_metadata + extra
    );
    assert_eq!(survivor.budget.budget().admitted_peak_bytes(), limit);
    drop(survivor);
}

#[test]
fn bootstrap_extension_survives_roots_until_final_native_or_owned_guard_drops() {
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let root_metadata = size_of::<OwnedBudget>();
    let charge_metadata = size_of::<OwnedExternalCharge>();
    let extra = 29;
    let native_bytes = 7;
    let charge_bytes = charge_metadata + 5;
    let limit = baseline + extra + native_bytes + charge_bytes;
    for native_last in [false, true] {
        let owned = OwnedBudget::new(limit).unwrap();
        let weak = owned.test_counter_weak();
        let identity = owned.budget() as *const Budget;
        owned.reserve_bootstrap(extra).unwrap();
        let native = owned.reserve_native(native_bytes).unwrap();
        let charge = owned.charge_external(5).unwrap();
        drop(owned);
        let fixed = baseline - root_metadata + extra;
        assert_eq!(
            charge.budget().used_bytes(),
            fixed + native_bytes + charge_bytes
        );
        if native_last {
            drop(charge);
            assert_eq!(native.budget.budget() as *const Budget, identity);
            assert_eq!(native.budget.budget().used_bytes(), fixed + native_bytes);
            assert_eq!(weak.strong_count(), 1);
            drop(native);
        } else {
            drop(native);
            assert_eq!(charge.budget() as *const Budget, identity);
            assert_eq!(charge.budget().used_bytes(), fixed + charge_bytes);
            assert_eq!(weak.strong_count(), 1);
            drop(charge);
        }
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn bootstrap_extension_survives_failed_builder_and_cancelled_guard_scope() {
    // Ordinary synchronous error/scope drops only: no async cancellation or
    // native execution is exercised or established by this unit test.
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let extra = 31;
    let charge_bytes = size_of::<OwnedExternalCharge>() + 13;
    let owned = OwnedBudget::new(baseline + extra + charge_bytes).unwrap();
    owned.reserve_bootstrap(extra).unwrap();
    let identity = owned.budget() as *const Budget;
    let charge = owned.charge_external(13).unwrap();
    let failed: Result<OwnedCharged<()>, Error> = charge.try_build(|budget| {
        assert_eq!(budget as *const Budget, identity);
        assert_eq!(budget.used_bytes(), baseline + extra + charge_bytes);
        Err(Error::Allocation)
    });
    assert!(matches!(failed, Err(Error::Allocation)));
    assert_eq!(owned.budget().used_bytes(), baseline + extra);
    {
        let operation_guard = owned.charge_external(13).unwrap();
        assert!(operation_guard.is_for(owned.budget()));
        assert_eq!(owned.budget().used_bytes(), baseline + extra + charge_bytes);
        // Abandon the local operation; ordinary guard destruction refunds it.
    }
    assert_eq!(owned.budget().used_bytes(), baseline + extra);
    assert_eq!(
        owned.budget().admitted_peak_bytes(),
        baseline + extra + charge_bytes
    );
}
