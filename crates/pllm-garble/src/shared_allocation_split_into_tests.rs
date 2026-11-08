//! Capacity/accounting regressions, not whole-payload retention or custody proof.
use super::*;
use std::cell::Cell;

fn counter_state(budget: &Budget) -> (usize, usize, usize, usize) {
    (
        budget as *const Budget as usize,
        budget.used_bytes(),
        budget.admitted_peak_bytes(),
        budget.test_accounting_updates(),
    )
}

fn owned_state(charge: &OwnedExternalCharge) -> (usize, usize, usize, usize) {
    (
        charge.budget() as *const Budget as usize,
        charge.charged_bytes(),
        charge.payload_bytes(),
        Arc::strong_count(&charge.budget),
    )
}

#[test]
fn capacity_accounting_owned_occupied_even_same_counter_is_not_reusable() {
    for same_counter in [false, true] {
        let root = OwnedBudget::new(4096).unwrap();
        let foreign = OwnedBudget::new(4096).unwrap();
        let mut parent = root.charge_external(128).unwrap();
        let mut dst = Some(if same_counter {
            root.charge_external(7).unwrap()
        } else {
            foreign.charge_external(7).unwrap()
        });
        let parent_before = owned_state(&parent);
        let dst_before = owned_state(dst.as_ref().unwrap());
        let root_before = counter_state(root.budget());
        let foreign_before = counter_state(foreign.budget());
        // Occupancy wins even when the numeric request also overflows or is valid.
        for payload in [usize::MAX, 0, 1] {
            assert_eq!(
                parent.split_owned_into(payload, &mut dst),
                Err(Error::Resource)
            );
            assert_eq!(owned_state(&parent), parent_before);
            assert_eq!(owned_state(dst.as_ref().unwrap()), dst_before);
            assert_eq!(counter_state(root.budget()), root_before);
            assert_eq!(counter_state(foreign.budget()), foreign_before);
        }
    }
}

#[test]
fn capacity_accounting_owned_overflow_and_insufficient_payload_are_atomic() {
    let metadata = size_of::<OwnedExternalCharge>();
    for available in [0, metadata - 1, metadata + 7] {
        let root = OwnedBudget::new(4096).unwrap();
        let mut parent = root.charge_external(available).unwrap();
        let before = owned_state(&parent);
        let counter = counter_state(root.budget());
        let mut dst = None;
        for payload in [usize::MAX, usize::MAX - metadata + 1, available + 1] {
            assert_eq!(
                parent.split_owned_into(payload, &mut dst),
                Err(Error::Resource)
            );
            assert!(dst.is_none());
            assert_eq!(owned_state(&parent), before);
            assert_eq!(counter_state(root.budget()), counter);
        }
        if available < metadata {
            assert_eq!(parent.split_owned_into(0, &mut dst), Err(Error::Resource));
            assert!(dst.is_none());
            assert_eq!(owned_state(&parent), before);
            assert_eq!(counter_state(root.budget()), counter);
        }
    }
}

#[test]
fn capacity_accounting_owned_exact_fit_zero_and_partial_keep_original_counter() {
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let metadata = size_of::<OwnedExternalCharge>();
    for payload in [0, 1, 31] {
        for residual in [0, 9] {
            let limit = baseline + 2 * metadata + payload + residual;
            let root = OwnedBudget::new(limit).unwrap();
            let mut parent = root
                .charge_external(metadata + payload + residual)
                .unwrap();
            let counter = counter_state(root.budget());
            let mut dst = None;
            parent.split_owned_into(payload, &mut dst).unwrap();
            let child = dst.as_ref().unwrap();
            assert_eq!(child.payload_bytes(), payload);
            assert_eq!(child.charged_bytes(), metadata + payload);
            assert!(child.is_for(root.budget()));
            assert_eq!(parent.payload_bytes(), residual);
            assert_eq!(parent.charged_bytes(), metadata + residual);
            assert_eq!(counter_state(root.budget()), counter);
            assert!(matches!(root.budget().reserve(1), Err(Error::Resource)));
            assert_eq!(counter_state(root.budget()), counter);
            drop(dst);
            assert_eq!(root.budget().used_bytes(), baseline + metadata + residual);
            drop(parent);
            assert_eq!(root.budget().used_bytes(), baseline);
        }
    }
}

#[test]
fn capacity_accounting_borrowed_occupied_even_same_counter_is_not_reusable() {
    for same_counter in [false, true] {
        for payload in [usize::MAX, 0, 1] {
            let root = OwnedBudget::new(4096).unwrap();
            let foreign = OwnedBudget::new(4096).unwrap();
            let mut parent = root.charge_external(128).unwrap();
            let mut dst = Some(if same_counter {
                root.budget().charge_external(7).unwrap()
            } else {
                foreign.budget().charge_external(7).unwrap()
            });
            let parent_before = owned_state(&parent);
            let child_before = {
                let child = dst.as_ref().unwrap();
                (child.lease.budget as *const Budget, child.lease.bytes)
            };
            let root_before = counter_state(root.budget());
            let foreign_before = counter_state(foreign.budget());
            assert_eq!(
                parent.split_external_into(payload, &mut dst),
                Err(Error::Resource)
            );
            let child = dst.as_ref().unwrap();
            assert_eq!(
                (child.lease.budget as *const Budget, child.lease.bytes),
                child_before
            );
            assert_eq!(child.payload_bytes(), 7);
            assert_eq!(counter_state(root.budget()), root_before);
            assert_eq!(counter_state(foreign.budget()), foreign_before);
            // End the conservative mutable-parent borrow before inspecting it.
            drop(dst);
            assert_eq!(owned_state(&parent), parent_before);
        }
    }
}

#[test]
fn capacity_accounting_borrowed_overflow_and_insufficient_payload_are_atomic() {
    let metadata = size_of::<ExternalCharge<'_>>();
    for available in [0, metadata - 1, metadata + 7] {
        for payload in [usize::MAX, usize::MAX - metadata + 1, available + 1] {
            let root = OwnedBudget::new(4096).unwrap();
            let mut parent = root.charge_external(available).unwrap();
            let before = owned_state(&parent);
            let counter = counter_state(root.budget());
            let mut dst = None;
            assert_eq!(
                parent.split_external_into(payload, &mut dst),
                Err(Error::Resource)
            );
            assert!(dst.is_none());
            assert_eq!(counter_state(root.budget()), counter);
            drop(dst);
            assert_eq!(owned_state(&parent), before);
            assert_eq!(counter_state(root.budget()), counter);
        }
    }
    for available in [0, metadata - 1] {
        let root = OwnedBudget::new(4096).unwrap();
        let mut parent = root.charge_external(available).unwrap();
        let before = owned_state(&parent);
        let counter = counter_state(root.budget());
        let mut dst = None;
        assert_eq!(parent.split_external_into(0, &mut dst), Err(Error::Resource));
        assert!(dst.is_none());
        assert_eq!(counter_state(root.budget()), counter);
        drop(dst);
        assert_eq!(owned_state(&parent), before);
        assert_eq!(counter_state(root.budget()), counter);
    }
}

#[test]
fn capacity_accounting_borrowed_exact_fit_zero_and_partial_keep_original_counter() {
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let owned_metadata = size_of::<OwnedExternalCharge>();
    let metadata = size_of::<ExternalCharge<'_>>();
    for payload in [0, 1, 31] {
        for residual in [0, 9] {
            let limit = baseline + owned_metadata + metadata + payload + residual;
            let root = OwnedBudget::new(limit).unwrap();
            let mut parent = root
                .charge_external(metadata + payload + residual)
                .unwrap();
            let counter = counter_state(root.budget());
            let mut dst = None;
            parent.split_external_into(payload, &mut dst).unwrap();
            let child = dst.as_ref().unwrap();
            assert_eq!(child.payload_bytes(), payload);
            assert_eq!(child.charged_bytes(), metadata + payload);
            assert!(child.is_for(root.budget()));
            assert_eq!(counter_state(root.budget()), counter);
            assert!(matches!(root.budget().reserve(1), Err(Error::Resource)));
            assert_eq!(counter_state(root.budget()), counter);
            drop(dst);
            assert_eq!(parent.payload_bytes(), residual);
            assert_eq!(parent.charged_bytes(), owned_metadata + residual);
            assert_eq!(
                root.budget().used_bytes(),
                baseline + owned_metadata + residual
            );
            drop(parent);
            assert_eq!(root.budget().used_bytes(), baseline);
        }
    }
}

struct BackingDropObserver<'a> {
    backing: Vec<u8>,
    budget: &'a Budget,
    observed: &'a Cell<usize>,
}

impl Drop for BackingDropObserver<'_> {
    fn drop(&mut self) {
        assert_eq!(self.backing.len(), 31);
        self.observed.set(self.budget.used_bytes());
        // The Vec field retires after this observer, still before the caller
        // drops its separately resident child charge.
    }
}

#[test]
fn capacity_accounting_owned_resident_slot_survives_error_and_caught_panic() {
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let metadata = size_of::<OwnedExternalCharge>();
    for unwind in [false, true] {
        let root = OwnedBudget::new(4096).unwrap();
        let parent = root.charge_external(metadata + 31).unwrap();
        let observed = Cell::new(0);
        // Both backing and destination already reside OUTSIDE the operation.
        // This caller ordering, not the accounting API, establishes retention.
        let mut dst = None;
        let backing = BackingDropObserver {
            backing: vec![0; 31],
            budget: root.budget(),
            observed: &observed,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut parent = parent;
            parent.split_owned_into(31, &mut dst)?;
            if unwind {
                panic!("caller operation after owned split-into");
            }
            Err::<(), _>(Error::Allocation)
        }));
        if unwind {
            assert!(result.is_err());
        } else {
            assert_eq!(result.unwrap(), Err(Error::Allocation));
        }
        let child = dst.as_ref().unwrap();
        assert!(child.is_for(root.budget()));
        assert_eq!(child.payload_bytes(), 31);
        let held = baseline + metadata + 31;
        assert_eq!(root.budget().used_bytes(), held);
        drop(backing);
        assert_eq!(observed.get(), held);
        assert_eq!(root.budget().used_bytes(), held);
        drop(dst);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn capacity_accounting_borrowed_resident_slot_survives_error_and_caught_panic() {
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let owned_metadata = size_of::<OwnedExternalCharge>();
    let metadata = size_of::<ExternalCharge<'_>>();
    for unwind in [false, true] {
        let root = OwnedBudget::new(4096).unwrap();
        let mut parent = root.charge_external(metadata + 31).unwrap();
        let observed = Cell::new(0);
        let mut dst = None;
        let backing = BackingDropObserver {
            backing: vec![0; 31],
            budget: root.budget(),
            observed: &observed,
        };
        // Explicit references move into FnOnce; no child borrows a closure local.
        let parent_ref = &mut parent;
        let dst_ref = &mut dst;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let parent_ref = parent_ref;
            parent_ref.split_external_into(31, dst_ref)?;
            if unwind {
                panic!("caller operation after borrowed split-into");
            }
            Err::<(), _>(Error::Allocation)
        }));
        if unwind {
            assert!(result.is_err());
        } else {
            assert_eq!(result.unwrap(), Err(Error::Allocation));
        }
        let child = dst.as_ref().unwrap();
        assert!(child.is_for(root.budget()));
        assert_eq!(child.payload_bytes(), 31);
        let held = baseline + owned_metadata + metadata + 31;
        assert_eq!(root.budget().used_bytes(), held);
        drop(backing);
        assert_eq!(observed.get(), held);
        assert_eq!(root.budget().used_bytes(), held);
        drop(dst);
        assert_eq!(parent.payload_bytes(), 0);
        assert_eq!(root.budget().used_bytes(), baseline + owned_metadata);
        drop(parent);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn capacity_accounting_owned_child_outlives_root_and_parent_on_original_counter() {
    let quote = OwnedBudget::resources().unwrap();
    let root = OwnedBudget::new(4096).unwrap();
    let weak = root.test_counter_weak();
    let original = root.budget() as *const Budget;
    let mut parent = root
        .charge_external(size_of::<OwnedExternalCharge>() + 31)
        .unwrap();
    let mut dst = None;
    parent.split_owned_into(31, &mut dst).unwrap();
    drop(root);
    drop(parent);
    let child = dst.as_ref().unwrap();
    assert_eq!(child.budget() as *const Budget, original);
    assert_eq!(
        child.budget().used_bytes(),
        quote.initial_reserved_bytes - quote.owner_metadata_bytes + child.charged_bytes()
    );
    assert!(weak.upgrade().is_some());
    drop(dst);
    assert!(weak.upgrade().is_none());
}

#[test]
fn capacity_accounting_borrowed_derived_lease_keeps_original_counter_until_retired() {
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let root = OwnedBudget::new(4096).unwrap();
    let mut parent = root
        .charge_external(size_of::<ExternalCharge<'_>>() + 31)
        .unwrap();
    let before = counter_state(root.budget());
    let mut dst = None;
    parent.split_external_into(31, &mut dst).unwrap();
    let lease = dst
        .as_mut()
        .unwrap()
        .split_payload(root.budget(), 31)
        .unwrap();
    assert!(std::ptr::eq(lease.budget, root.budget()));
    assert_eq!(lease.bytes, 31);
    assert_eq!(counter_state(root.budget()), before);
    drop(dst);
    // The derived lease still carries the parent's borrow, even without child.
    assert_eq!(
        lease.budget.used_bytes(),
        before.1 - size_of::<ExternalCharge<'_>>()
    );
    drop(lease);
    assert_eq!(parent.payload_bytes(), 0);
    assert_eq!(
        root.budget().used_bytes(),
        baseline + size_of::<OwnedExternalCharge>()
    );
    drop(parent);
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn capacity_accounting_positive_payload_fits_alone_but_child_metadata_does_not() {
    let payload = 7;
    for borrowed in [false, true] {
        let metadata = if borrowed {
            size_of::<ExternalCharge<'_>>()
        } else {
            size_of::<OwnedExternalCharge>()
        };
        let available = payload + metadata - 1;
        assert!(payload > 0);
        assert!(available >= payload);
        assert!(available < payload + metadata);
        let root = OwnedBudget::new(4096).unwrap();
        let mut parent = root.charge_external(available).unwrap();
        let parent_before = owned_state(&parent);
        let counter_before = counter_state(root.budget());
        if borrowed {
            let mut dst = None;
            assert_eq!(
                parent.split_external_into(payload, &mut dst),
                Err(Error::Resource)
            );
            assert!(dst.is_none());
            assert_eq!(counter_state(root.budget()), counter_before);
            drop(dst);
        } else {
            let mut dst = None;
            assert_eq!(
                parent.split_owned_into(payload, &mut dst),
                Err(Error::Resource)
            );
            assert!(dst.is_none());
            assert_eq!(counter_state(root.budget()), counter_before);
        }
        assert_eq!(owned_state(&parent), parent_before);
        assert_eq!(counter_state(root.budget()), counter_before);
    }
}
