use super::*;

fn context() -> Context {
    Context {
        region: [1; 32],
        stage: [2; 32],
        inputs: [[3; 32], [4; 32]],
        output: [5; 32],
        first: 7,
    }
}
fn issue<'a>(
    budget: &'a Budget,
    first: &[u64],
    second: &[u64],
) -> (MasksFor<&'a Budget>, [PartyFor<&'a Budget>; 2]) {
    let cost = resources_for::<&Budget>(first.len()).unwrap();
    let lease = LeaseFor::reserve(&budget, cost.combined_bytes).unwrap();
    let mut sample = 0u64;
    issue_reserved(context(), first, second, lease, |bytes| {
        for word in bytes.chunks_mut(8) {
            sample = sample.wrapping_add(0x1234_5678_9abc);
            word.copy_from_slice(&sample.to_le_bytes()[..word.len()]);
        }
        Ok(())
    })
    .unwrap()
}

#[test]
fn genuine_negative_mask_algebra_both_roles_at_lane_and_mask_endpoints() {
    let ring = mask(50);
    for lanes in [1, 256] {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let rx: Vec<_> = (0..lanes).map(|i| [0, 1, 1 << 49, ring][i % 4]).collect();
        let ry: Vec<_> = rx.iter().map(|v| v ^ ring).collect();
        let x: Vec<_> = (0..lanes).map(|i| [0u64, 1, 4095, 4096][i % 4]).collect();
        let y: Vec<_> = x.iter().rev().copied().collect();
        let a: Vec<_> = x.iter().zip(&rx).map(|(v, r)| (v + r) & ring).collect();
        let b: Vec<_> = y.iter().zip(&ry).map(|(v, r)| (v + r) & ring).collect();
        let (masks, [mut left, mut right]) = issue(&budget, &rx, &ry);
        let lf = left.begin(&a, &b).unwrap();
        let rf = right.begin(&a, &b).unwrap();
        assert_eq!(lf.as_bytes().len(), 76 + 8 * lanes);
        assert_eq!(&lf.as_bytes()[..8], b"PLLMSP01");
        let lo = left.finish(rf.as_bytes()).unwrap();
        let ro = right.finish(lf.as_bytes()).unwrap();
        for i in 0..lanes {
            let oracle =
                ((x[i] as u128 * y[i] as u128) + masks.values()[i] as u128) % (1u128 << 50);
            assert_eq!(lo.values()[i], oracle as u64);
            assert_eq!(ro.values()[i], oracle as u64);
        }
        assert!(matches!(left.finish(rf.as_bytes()), Err(Error::Consumed)));
        drop((left, right, masks));
        assert!(budget.used_bytes() > 0); // escaped frame/output owners still live
        drop((lf, rf, lo, ro));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn exact_budget_and_one_short_refuse_before_entropy_or_factories() {
    let cost = resources_for::<&Budget>(1).unwrap();
    for short in [false, true] {
        let budget = Budget::new(cost.combined_bytes - usize::from(short)).unwrap();
        let reservation = LeaseFor::reserve(&&budget, cost.combined_bytes);
        if short {
            assert!(matches!(reservation, Err(Error::Resource)));
        } else {
            let result = issue_reserved(context(), &[0], &[1], reservation.unwrap(), |_| Ok(()));
            assert!(result.is_ok());
            drop(result);
        }
        assert_eq!(budget.used_bytes(), 0);
    }
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut calls = 0;
    let short = LeaseFor::reserve(&&budget, cost.combined_bytes - 1).unwrap();
    let result = issue_reserved(context(), &[0], &[0], short, |_| {
        calls += 1;
        Ok(())
    });
    assert!(matches!(result, Err(Error::Resource)));
    assert_eq!(calls, 0);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn issuance_has_separate_nonce_and_four_independent_full_ring_requests_per_lane() {
    for lanes in [1, 256] {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let cost = resources_for::<&Budget>(lanes).unwrap();
        let masks = vec![0; lanes];
        let mut sizes = Vec::new();
        let lease = LeaseFor::reserve(&&budget, cost.combined_bytes).unwrap();
        let owners = issue_reserved(context(), &masks, &masks, lease, |bytes| {
            sizes.push(bytes.len());
            bytes.fill(sizes.len() as u8);
            Ok(())
        })
        .unwrap();
        assert_eq!(sizes[0], 32);
        assert_eq!(sizes.len(), 1 + 4 * lanes);
        assert!(sizes[1..].iter().all(|n| *n == 8));
        assert_eq!(sizes.iter().sum::<usize>(), cost.random_bytes);
        drop(owners);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn all_header_fields_payloads_and_both_ordered_input_digests_are_checked_and_burn() {
    // Canonical contribution tampering is NOT authenticated by this codec.
    for position in [0, 8, 40, 72, 73, 74, 75, 83] {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let (_, [mut left, mut right]) = issue(&budget, &[1], &[2]);
        let _lf = left.begin(&[4], &[6]).unwrap();
        let rf = right.begin(&[4], &[6]).unwrap();
        let mut invalid = rf.as_bytes().to_vec();
        if position == 83 {
            invalid[position] |= 0x80;
        } else {
            invalid[position] ^= 1;
        }
        assert!(left.finish(&invalid).is_err());
        assert!(matches!(left.finish(rf.as_bytes()), Err(Error::Consumed)));
    }
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (_, [mut left, mut right]) = issue(&budget, &[1], &[2]);
    let lf = left.begin(&[4], &[6]).unwrap();
    let rf = right.begin(&[6], &[4]).unwrap();
    assert!(matches!(left.finish(rf.as_bytes()), Err(Error::Context)));
    assert!(matches!(right.finish(lf.as_bytes()), Err(Error::Context)));
}

#[test]
fn malformed_inputs_and_contexts_precede_entropy_and_every_rng_failure_rolls_back() {
    let cost = resources_for::<&Budget>(2).unwrap();
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for fail in 1..=9 {
        let mut calls = 0;
        let lease = LeaseFor::reserve(&&budget, cost.combined_bytes).unwrap();
        let result = issue_reserved(context(), &[0, 0], &[0, 0], lease, |_| {
            calls += 1;
            if calls == fail {
                Err(Error::Randomness)
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(Error::Randomness)));
        assert_eq!(budget.used_bytes(), 0);
    }
    let lease = LeaseFor::reserve(&&budget, cost.combined_bytes).unwrap();
    let mut calls = 0;
    let result = issue_reserved(context(), &[0, 1 << 50], &[0, 0], lease, |_| {
        calls += 1;
        Ok(())
    });
    assert!(matches!(result, Err(Error::Domain)));
    assert_eq!(calls, 0);
    let mut invalid = context();
    invalid.inputs[1] = invalid.inputs[0];
    let lease = LeaseFor::reserve(&&budget, cost.combined_bytes).unwrap();
    assert!(matches!(
        issue_reserved(invalid, &[0, 0], &[0, 0], lease, |_| {
            calls += 1;
            Ok(())
        }),
        Err(Error::Context)
    ));
    assert_eq!(calls, 0);
    assert_eq!(budget.used_bytes(), 0);
    assert!(matches!(resources_for::<&Budget>(0), Err(Error::Domain)));
    assert!(matches!(resources_for::<&Budget>(257), Err(Error::Domain)));
    assert!(matches!(mul(usize::MAX, 2), Err(Error::Resource)));
}

#[test]
fn entropy_unwind_and_begin_failure_keep_payload_before_original_guard_drop() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let cost = resources_for::<&Budget>(1).unwrap();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let lease = LeaseFor::reserve(&&budget, cost.combined_bytes).unwrap();
        let _ = issue_reserved(context(), &[0], &[0], lease, |_| -> Result<(), Error> {
            panic!("RNG")
        });
    }));
    assert!(unwind.is_err());
    assert_eq!(budget.used_bytes(), 0);
    let (masks, [mut left, right]) = issue(&budget, &[0], &[0]);
    assert!(matches!(left.begin(&[1 << 50], &[0]), Err(Error::Domain)));
    assert!(matches!(left.begin(&[0], &[0]), Err(Error::Consumed)));
    drop((masks, left, right));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn every_real_material_factory_fails_before_entropy_and_frame_failure_burns() {
    use std::sync::atomic::Ordering;
    let cost = resources_for::<&Budget>(1).unwrap();
    for fail in 0..5 {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        budget.fail_after.store(fail, Ordering::SeqCst);
        let lease = LeaseFor::reserve(&&budget, cost.combined_bytes).unwrap();
        let mut draws = 0;
        assert!(matches!(
            issue_reserved(context(), &[0], &[0], lease, |_| {
                draws += 1;
                Ok(())
            }),
            Err(Error::Allocation)
        ));
        assert_eq!(draws, 0);
        assert_eq!(budget.used_bytes(), 0);
    }
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (masks, [mut left, right]) = issue(&budget, &[0], &[0]);
    budget.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(left.begin(&[1], &[2]), Err(Error::Allocation)));
    assert!(left.execution.is_none());
    assert!(matches!(left.begin(&[1], &[2]), Err(Error::Consumed)));
    drop((masks, left, right));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn owned_reservation_root_drop_keeps_same_counter_through_frames_and_outputs() {
    use crate::shared_allocation::{OwnedBudget, OwnedBudgetContext};
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let original = std::ptr::from_ref(root.budget());
    let cost = resources_for::<OwnedBudgetContext>(1).unwrap();
    let mut working = root.charge_external(cost.combined_bytes).unwrap();
    let lease = working
        .split_native(root.budget(), cost.combined_bytes)
        .unwrap();
    let (masks, [mut left, mut right]) = issue_reserved(context(), &[0], &[0], lease, |bytes| {
        getrandom::fill(bytes).map_err(|_| Error::Randomness)
    })
    .unwrap();
    drop((working, root));
    let a = left.begin(&[4096], &[4096]).unwrap();
    let b = right.begin(&[4096], &[4096]).unwrap();
    let l = left.finish(b.as_bytes()).unwrap();
    let r = right.finish(a.as_bytes()).unwrap();
    for out in [&l, &r] {
        assert_eq!(std::ptr::from_ref(out._lease.budget.budget()), original);
        assert_eq!(
            out.values()[0].wrapping_sub(masks.values()[0]) & mask(50),
            1 << 24
        );
    }
    drop((left, right, masks, l, r));
    assert_eq!(std::ptr::from_ref(a._lease.budget.budget()), original);
    assert!(a._lease.budget.budget().used_bytes() > 0);
    drop((a, b));
}
