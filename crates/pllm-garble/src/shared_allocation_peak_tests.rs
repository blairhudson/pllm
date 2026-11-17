use super::*;

// Independent Layout-based oracle, not the const implementation's helper.
fn measured_type<T>() -> CallableTypeLayout {
    let layout = Layout::new::<T>();
    CallableTypeLayout { size: layout.size(), alignment: layout.align() }
}

fn measured_mapper<R, A, B, M, F>(_: &F) -> ReservationMapperLayouts
where
    F: FnOnce(A) -> B,
{
    ReservationMapperLayouts {
        callable: measured_type::<F>(),
        shared_reference: measured_type::<&F>(),
        mutable_reference: measured_type::<&mut F>(),
        receiver: measured_type::<R>(),
        adapter_operands: measured_type::<(R, F)>(),
        owned_call: measured_type::<(F, (A,))>(),
        arguments: measured_type::<(A,)>(),
        output: measured_type::<B>(),
        mapped_output: measured_type::<M>(),
        mutable_adapter_reference: measured_type::<&mut (R, F)>(),
        nominal_captures: measured_type::<()>(),
    }
}

// F is inferred from a constructed value here, independently of the const
// factory-output inference. Neither this helper nor its caller invokes F.
fn measured_proposal<F>(_: &F, factory: ReservationFactoryLayouts) -> ReservationProposalLayouts
where
    F: FnMut(usize) -> Option<usize>,
{
    ReservationProposalLayouts {
        callable: measured_type::<F>(),
        shared_reference: measured_type::<&F>(),
        mutable_reference: measured_type::<&mut F>(),
        receiver: measured_type::<(&AtomicUsize, Ordering, Ordering)>(),
        adapter_operands: measured_type::<((&AtomicUsize, Ordering, Ordering), F)>(),
        mutable_adapter_reference:
            measured_type::<&mut ((&AtomicUsize, Ordering, Ordering), F)>(),
        fetch_update_operands: measured_type::<(&AtomicUsize, Ordering, Ordering, F)>(),
        mutable_call: measured_type::<(&mut F, (usize,))>(),
        owned_call: measured_type::<(F, (usize,))>(),
        arguments: measured_type::<(usize,)>(),
        output: measured_type::<Option<usize>>(),
        fetch_update_result: measured_type::<Result<usize, usize>>(),
        nominal_captures: measured_type::<(usize, usize)>(),
        factory,
    }
}

fn measured_factory<Factory, F>(_: &Factory) -> ReservationFactoryLayouts
where
    Factory: FnOnce(usize, usize) -> F,
{
    ReservationFactoryLayouts {
        callable: measured_type::<Factory>(),
        shared_reference: measured_type::<&Factory>(),
        mutable_reference: measured_type::<&mut Factory>(),
        arguments: measured_type::<(usize, usize)>(),
        owned_call: measured_type::<(Factory, (usize, usize))>(),
        output: measured_type::<F>(),
        nominal_captures: measured_type::<()>(),
    }
}

#[test]
fn reservation_callable_const_matches_actual_definition_unit_test_layouts() {
    // This is garble's cfg(test) unit. The same public associated const must
    // separately compile in the compiler's normal garble dependency unit;
    // this test does not certify that unit's cfg-dependent Budget layout.
    const LAYOUTS: ReservationCallableLayouts = OwnedBudgetResources::RESERVATION_CALLABLE_LAYOUTS;
    let proposal = reserve_update(3, 7); // Construct only; never invoke proposal.
    let expected_proposal = measured_proposal(&proposal, measured_factory(&reserve_update));
    assert_eq!(LAYOUTS.proposal, expected_proposal);
    let mapper = resource_error::<usize>; // Exact item, not a function pointer.
    assert_eq!(
        LAYOUTS.failure_mapper,
        measured_mapper::<Result<usize, usize>, usize, Error, Result<usize, Error>, _>(&mapper),
    );
    // Nominal source captures are checked separately from measured opaque F.
    assert_eq!(LAYOUTS.proposal.nominal_captures, measured_type::<(usize, usize)>());
    assert_eq!(LAYOUTS.proposal.factory.nominal_captures, measured_type::<()>());
    assert_eq!(LAYOUTS.failure_mapper.nominal_captures, measured_type::<()>());
}

#[test]
fn reservation_callable_layout_plumbing_preserves_overaligned_whole_tuples() {
    #[repr(align(64))]
    struct ProposalCapture { _bytes: [u8; 3] }
    #[repr(align(128))]
    struct FactoryCapture { _bytes: [u8; 5] }
    #[repr(align(256))]
    struct MapperCapture { _bytes: [u8; 7] }

    let proposal_capture = ProposalCapture { _bytes: [0; 3] };
    let factory_capture = FactoryCapture { _bytes: [0; 5] };
    let factory = move |_: usize, _: usize| {
        let _whole_factory_capture = &factory_capture;
        move |used: usize| {
            let _whole_proposal_capture = &proposal_capture;
            Some(used)
        }
    };
    let layouts = reservation_proposal_layouts(&factory);
    let expected_factory = measured_factory(&factory);
    let proposal = factory(0, 0); // Synthetic construction only, not invocation of F.
    let expected = measured_proposal(&proposal, expected_factory);
    assert_eq!(layouts, expected);
    assert_eq!(layouts.callable.alignment, 64);
    assert_eq!(layouts.factory.callable.alignment, 128);
    assert!(layouts.owned_call.size > layouts.callable.size + layouts.arguments.size);
    assert!(layouts.factory.owned_call.size
        > layouts.factory.callable.size + layouts.factory.arguments.size);

    let mapper_capture = MapperCapture { _bytes: [0; 7] };
    let mapper = move |_: usize| {
        let _whole_mapper_capture = &mapper_capture;
        Error::Resource
    };
    let mapper_layouts = reservation_mapper_layouts::<
        Result<usize, usize>, usize, Error, Result<usize, Error>, _,
    >(&mapper);
    assert_eq!(mapper_layouts,
        measured_mapper::<Result<usize, usize>, usize, Error, Result<usize, Error>, _>(&mapper));
    assert_eq!(mapper_layouts.callable.alignment, 256);
    assert!(mapper_layouts.adapter_operands.size
        > mapper_layouts.receiver.size + mapper_layouts.callable.size);
    assert!(mapper_layouts.owned_call.size
        > mapper_layouts.callable.size + mapper_layouts.arguments.size);
    // Synthetic capture records test padding/wiring only. The nominal fields
    // remain the native source contracts, not claims about these captures.
}

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
