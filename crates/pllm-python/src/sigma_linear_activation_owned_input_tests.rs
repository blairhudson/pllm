//! Genuine charged-byte compiler admission; Python exporter gate remains closed.
use super::*;
use crate::sigma_linear_activation_admission::{
    check_original_data, require_python_allocation_contract,
};

assert_not_trait!(PendingOwners, Send);
assert_not_trait!(PendingOwners, Sync);

#[test]
fn charged_bytes_to_roles_keep_global_prepaid_work_and_original_counter_after_root_drop() {
    let _serial = SERIAL.lock().unwrap();
    let root = OwnedBudget::new(LIMIT).unwrap();
    let identity = std::ptr::from_ref(root.budget());
    let mut dealer = dealer(&root);
    let prepared = prepare_raw_changed(&root, 0).unwrap();
    let source_floor = root.budget().used_bytes() - prepared.baseline;
    let mut pending = from_charged_plan_bytes(
        prepared.raw,
        prepared.declaration,
        prepared.data,
        &root,
        &mut dealer,
        raw_input_limits(),
    )
    .unwrap();
    let receipt = pending.retained._original.preparation_work().unwrap();
    assert!(receipt.parse_prepaid() > 0);
    assert!(receipt.metadata_prepaid() > 0);
    assert_eq!(
        receipt.inventory_prepaid(),
        raw_input_limits().inventory_work
    );
    assert_eq!(
        receipt.total_prepaid(),
        receipt.parse_prepaid() + receipt.metadata_prepaid() + receipt.inventory_prepaid()
    );
    assert!(receipt.total_prepaid() <= raw_input_limits().max_work);
    let (coordinator, roles) = pending.take_owners().unwrap();
    pending.commit().unwrap();
    assert_eq!(
        coordinator.retained._original.preparation_work(),
        Some(receipt)
    );
    drop((dealer, root));
    let frames = ready(&roles); // original facade, eleven exchanges + Ready
    let left = finish(&roles[0]);
    let right = finish(&roles[1]);
    assert_eq!(left.retained._original.preparation_work(), Some(receipt));
    assert_retention(&left.retained, identity, source_floor);
    drop((roles, frames));
    assert_eq!(left.descriptor(), Ok(prepared.descriptor));
    drop(coordinator);
    assert!(left.descriptor().is_err());
    assert!(right.descriptor().is_err());
    assert_retention(&right.retained, identity, source_floor);
}

#[test]
fn creation_error_or_unwind_after_take_revokes_escaped_roles_and_keeps_original_owners() {
    let _serial = SERIAL.lock().unwrap();
    for panic in [false, true] {
        let root = OwnedBudget::new(LIMIT).unwrap();
        let mut dealer = dealer(&root);
        let baseline = root.budget().used_bytes();
        let prepared = prepare_changed(&root, 0).unwrap();
        let mut escaped = None;
        let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), BridgeError> {
            let mut pending =
                from_admitted_rust_inputs(prepared.original, prepared.data, &root, &mut dealer)?;
            escaped = Some(pending.take_owners()?);
            if panic {
                panic!("creation packaging after take");
            }
            Err(BridgeError::Resource)
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert_eq!(result.unwrap(), Err(BridgeError::Resource));
        }
        let (coordinator, roles) = escaped.take().unwrap();
        assert_eq!(coordinator.is_aborted(), Ok(true));
        assert!(roles.iter().all(|r| r.begin(&mut || false).is_err()));
        assert_retention(
            &coordinator.retained,
            std::ptr::from_ref(root.budget()),
            prepared.source_floor,
        );
        drop((coordinator, roles));
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn creation_commit_requires_one_take_and_live_coordinator() {
    let _serial = SERIAL.lock().unwrap();
    for case in 0..3 {
        let root = OwnedBudget::new(LIMIT).unwrap();
        let mut dealer = dealer(&root);
        let baseline = root.budget().used_bytes();
        let p = prepare_changed(&root, 0).unwrap();
        let mut pending =
            from_admitted_rust_inputs(p.original, p.data, &root, &mut dealer).unwrap();
        let owners = if case == 0 {
            None
        } else {
            Some(pending.take_owners().unwrap())
        };
        if case == 1 {
            assert!(matches!(pending.take_owners(), Err(BridgeError::Consumed)));
        }
        if case == 2 {
            owners.as_ref().unwrap().0.cancel().unwrap();
        }
        assert_eq!(pending.commit(), Err(BridgeError::Consumed));
        if let Some((coordinator, roles)) = &owners {
            assert_eq!(coordinator.is_aborted(), Ok(true));
            assert!(roles.iter().all(|role| role.begin(&mut || false).is_err()));
        }
        drop(owners);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn pressure_after_compiler_admission_refuses_bridge_before_native_factory() {
    let _serial = SERIAL.lock().unwrap();
    let root = OwnedBudget::new(LIMIT).unwrap();
    let mut dealer = dealer(&root);
    let baseline = root.budget().used_bytes();
    let p = prepare_changed(&root, 0).unwrap();
    let quote = BridgeResources::checked().unwrap();
    let cost = quote.payload_bytes + size_of::<OwnedExternalCharge>();
    let available = LIMIT - root.budget().used_bytes();
    let pressure = root
        .charge_external(available - cost + 1 - size_of::<OwnedExternalCharge>())
        .unwrap();
    assert!(matches!(
        from_admitted_rust_inputs(p.original, p.data, &root, &mut dealer),
        Err(BridgeError::Allocation(AllocationError::Resource))
    ));
    assert_eq!(
        root.budget().used_bytes(),
        baseline + pressure.charged_bytes()
    );
    drop(pressure);
    assert_eq!(root.budget().used_bytes(), baseline);
    drop(issue(&root, &mut dealer).unwrap());
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn actual_buffer_capacity_and_original_counter_cannot_be_replaced_by_numeric_allowance() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let prepared = prepare_changed(&root, 0).unwrap();
    let mut data = prepared.data.payload().clone();
    data.linear.reserve_exact(4096);
    let underfunded = root
        .charge_external(0)
        .unwrap()
        .try_build(|_| Ok(data))
        .unwrap();
    assert_eq!(
        check_original_data(&underfunded, &root),
        Err(BridgeError::UnadmittedInput)
    );
    let foreign = OwnedBudget::new(LIMIT).unwrap();
    assert_eq!(
        check_original_data(&prepared.data, &foreign),
        Err(BridgeError::ForeignInput)
    );
    assert_eq!(check_original_data(&prepared.data, &root), Ok(()));
}

#[test]
fn python_parser_exporter_and_packaging_contract_remain_closed_without_allocating() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let before = root.budget().used_bytes();
    assert_eq!(
        require_python_allocation_contract(),
        Err(BridgeError::UnadmittedPython)
    );
    assert_eq!(root.budget().used_bytes(), before);
}

#[test]
fn actual_python_packaging_gate_burns_creation_before_any_object_allocation() {
    let _serial = SERIAL.lock().unwrap();
    let root = OwnedBudget::new(LIMIT).unwrap();
    let mut dealer = dealer(&root);
    let baseline = root.budget().used_bytes();
    let p = prepare_changed(&root, 0).unwrap();
    let mut pending = from_admitted_rust_inputs(p.original, p.data, &root, &mut dealer).unwrap();
    let before = root.budget().used_bytes();
    assert!(matches!(
        pending.take_python_owners(),
        Err(BridgeError::UnadmittedPython)
    ));
    assert!(root.budget().used_bytes() <= before);
    let owners = pending.take_owners().unwrap();
    assert_eq!(owners.0.is_aborted(), Ok(true));
    assert_eq!(pending.commit(), Err(BridgeError::Consumed));
    assert!(owners
        .1
        .iter()
        .all(|role| role.begin(&mut || false).is_err()));
    drop(owners);
    assert_eq!(root.budget().used_bytes(), baseline);
}
