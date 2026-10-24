//! Genuine D36 native-region tests. Test mailboxes/expected codes are not a
//! request-storage, source authentication, ideal-softmax or privacy certificate.
use super::*;
use crate::shared_allocation::{OwnedBudget, OwnedBudgetContext, OwnedExternalCharge};
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::Ordering;
type C = OwnedBudgetContext;

fn prepared(
    values: &[i64],
    active: &[u8],
    spare: usize,
) -> (
    OwnedBudget,
    OwnedLocalDealer,
    CausalSoftmaxSpec,
    CausalSoftmaxInput,
    OwnedExternalCharge,
    CausalSoftmaxResources,
) {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = SourceIdentity::new([1; 32], [2; 32], approved_implementation());
    let dealer = OwnedLocalDealer::new(source, [4; 32], [8; 32], &root).unwrap();
    let edge = |wire, representation| {
        EdgeDescriptor::new(
            EdgeIdentity::new(source, [4; 32], [5; 32], [wire; 32], 0),
            Shape::new(&[values.len()]).unwrap(),
            Numeric::new(50, 12, Interpretation::Signed).unwrap(),
            representation,
            None,
        )
        .unwrap()
    };
    let spec = CausalSoftmaxSpec::checked(
        edge(6, Representation::CommonAdditiveMasked),
        edge(
            7,
            if active.len() == 1 {
                Representation::Public
            } else {
                Representation::CommonAdditiveMasked
            },
        ),
        active,
        [9; 32],
        [10; 32],
    )
    .unwrap();
    let input = root
        .charge_external((values.len() + spare) * size_of::<i64>())
        .unwrap()
        .try_build(|budget| {
            let mut payload = crate::shared_allocation::allocate(values.len() + spare, budget)?;
            payload.extend_from_slice(values);
            Ok(Zeroizing::new(payload))
        })
        .unwrap();
    let mut working = root
        .charge_external(bootstrap_bytes::<C>().unwrap())
        .unwrap();
    let q = quote::resources(&dealer, &spec, &input).unwrap();
    working.resize_payload(q.total_reserved_bytes).unwrap();
    (root, dealer, spec, input, working, q)
}
fn issued(
    values: &[i64],
    active: &[u8],
) -> (
    OwnedBudget,
    OwnedLocalDealer,
    CausalSoftmaxSpec,
    CausalSoftmaxResources,
    Created<C>,
) {
    let (root, mut dealer, s, input, mut working, q) = prepared(values, active, 0);
    let native = working
        .split_native(dealer.budget.budget(), q.total_reserved_bytes)
        .unwrap();
    let created = construct(
        &mut dealer,
        s,
        input,
        native,
        CausalSoftmaxWorkLimit(q.work.total),
    )
    .unwrap();
    (root, dealer, s, q, created)
}
fn frame(step: Step<C>) -> Frame<C> {
    match step {
        Step::Frame(f) => f,
        Step::Ready => panic!("unexpected early Ready"),
    }
}
fn complete(roles: [Role<C>; 2], spec: &CausalSoftmaxSpec) -> ([Output<C>; 2], Vec<Frame<C>>) {
    let [mut left, mut right] = roles;
    if spec.count == 1 {
        assert!(matches!(left.begin(&mut || false).unwrap(), Step::Ready));
        assert!(matches!(right.begin(&mut || false).unwrap(), Step::Ready));
        return (
            [left.finish().unwrap(), right.finish().unwrap()],
            Vec::new(),
        );
    }
    let mut a = frame(left.begin(&mut || false).unwrap());
    let mut b = frame(right.begin(&mut || false).unwrap());
    let mut retained = Vec::new();
    for round in 0..spec.exchanges() {
        let ab = a.bytes().unwrap().to_vec();
        let bb = b.bytes().unwrap().to_vec();
        let nexta = left.advance(&bb, &mut || false).unwrap();
        let nextb = right.advance(&ab, &mut || false).unwrap();
        retained.extend([a, b]);
        if round + 1 == spec.exchanges() {
            assert!(matches!(nexta, Step::Ready));
            assert!(matches!(nextb, Step::Ready));
            return ([left.finish().unwrap(), right.finish().unwrap()], retained);
        }
        a = frame(nexta);
        b = frame(nextb);
    }
    unreachable!()
}
fn codes(mask: &Mask<C>, output: &Output<C>) -> Vec<u64> {
    mask.shared
        .operation(|_| {
            let state = output.slot.state.lock().unwrap();
            let s = state.keepalive.source.declaration;
            if let Some(values) = state.public {
                return Ok(values[..s.input.shape().elements()].to_vec());
            }
            let mut values = vec![0; s.input.shape().elements()];
            let native = state.output.as_ref().unwrap();
            let masks = mask.native.as_ref().unwrap();
            for (lane, &position) in s.active_positions().iter().enumerate() {
                values[usize::from(position)] =
                    native.values()[lane].wrapping_sub(masks.values()[lane]) & ((1u64 << 50) - 1);
            }
            Ok(values)
        })
        .unwrap()
}

#[test]
fn actual_root_matches_literal_integer_cases_and_public_inactive_positions() {
    // Independent selected literal facts: E(1)=4095, E(1279)=2995,
    // E(1280)=2996; NO monotonicity or renormalization assumption.
    for (values, active, expected) in [
        (vec![0, 0], vec![0, 1], vec![2048, 2048]),
        (vec![0, 0, 0], vec![0, 1, 2], vec![1365, 1365, 1365]),
        (vec![0, 1], vec![0, 1], vec![2063, 2064]),
        // TR AFTER SUM: residues63+63 produce a carry. Individual truncations
        // would give Z190, not the prescribed Z191/R1372.
        (vec![0, 0, 1], vec![0, 1, 2], vec![1371, 1371, 1372]),
        (vec![0, 1279], vec![0, 1], vec![(2995 * 2383) / 4096, 2383]),
        (vec![0, 1280], vec![0, 1], vec![(2996 * 2383) / 4096, 2383]),
        (
            vec![0, -(1i64 << 37), (1i64 << 37) - 1],
            vec![0, 1, 2],
            vec![0, 0, 4096],
        ),
        (
            vec![(1i64 << 37) - 1, -(1i64 << 37)],
            vec![0, 1],
            vec![4096, 0],
        ),
        (vec![9, -10, 9], vec![0, 2], vec![2048, 0, 2048]),
        (vec![9, -10, 9], vec![2], vec![0, 0, 4096]),
    ] {
        let (root, dealer, s, q, (mask, coordinator, roles)) = issued(&values, &active);
        let (outputs, frames) = complete(roles, &s);
        assert_eq!(frames.len(), 2 * s.exchanges());
        assert_eq!(
            frames
                .iter()
                .map(|f| f.bytes().unwrap().len())
                .sum::<usize>(),
            q.peer_bytes
        );
        for output in &outputs {
            assert_eq!(codes(&mask, output), expected);
        }
        let mass: u64 = expected.iter().sum();
        assert!(if active.len() == 1 {
            mass == 4096
        } else if active.len() == 2 {
            (4094..=4159).contains(&mass)
        } else {
            (4092..=4159).contains(&mass)
        });
        drop((outputs, frames, mask, coordinator, dealer));
        assert_eq!(
            root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    }
}

#[test]
fn exact_original_capacity_and_disjoint_second_floor_work_are_prepaid() {
    for active in [&[0, 1][..], &[0, 1, 2][..]] {
        let (root, dealer, s, input, working, q) = prepared(&[0, 1, 2], active, 7);
        assert_eq!(input.payload().capacity(), 10);
        assert_eq!(q.incoming_owner_bytes, input.charged_bytes());
        assert_eq!(q.work.outer_exp_floor.entries, 512);
        assert_eq!(
            q.work.child.floor_taylor,
            crate::fixed_exp_tables::quote().unwrap().proof.taylor_steps
        );
        assert_eq!(q.work.child.root_encoding_word_writes, 0);
        assert_eq!(q.work.child.root_mask_generation_words, 0);
        assert_eq!(q.work.child.public_resource_query_passes, 3);
        assert!(q.work.public_hash_bytes >= 4096 + 2048 + 2048);
        assert_eq!(s.outer_uses(), 13 * active.len() + 1);
        drop((working, input, dealer));
        assert_eq!(
            root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    }
}

#[test]
fn zero_and_one_short_guard_or_global_work_refuse_before_first_factory_or_entropy() {
    for (bytes_short, work_short) in [(0, 1), (1, 0)] {
        let (root, mut dealer, s, input, mut working, q) = prepared(&[0, 1], &[0, 1], 0);
        working
            .resize_payload(q.total_reserved_bytes - bytes_short)
            .unwrap();
        let native = working
            .split_native(dealer.budget.budget(), q.total_reserved_bytes - bytes_short)
            .unwrap();
        root.budget().fail_after.store(0, Ordering::SeqCst);
        let mut draws = 0;
        let result = construct_with(
            &mut dealer,
            s,
            input,
            native,
            CausalSoftmaxWorkLimit(q.work.total - work_short),
            |_| {
                draws += 1;
                Ok(())
            },
        );
        assert!(matches!(
            result,
            Err(BuildError(Failure::Native(Error::Resource)))
        ));
        assert_eq!(draws, 0);
        root.budget().fail_after.store(-1, Ordering::SeqCst);
    }
    let (root, mut dealer, s, input, working, _) = prepared(&[0, 1], &[0, 1], 0);
    let native = root.reserve_native(0).unwrap();
    let mut draws = 0;
    assert!(construct_with(
        &mut dealer,
        s,
        input,
        native,
        CausalSoftmaxWorkLimit(0),
        |_| {
            draws += 1;
            Ok(())
        }
    )
    .is_err());
    assert_eq!(draws, 0);
    drop(working);
}

#[test]
fn singleton_skips_all_private_entropy_and_requires_both_ready() {
    let (root, mut dealer, s, input, mut working, q) = prepared(&[7, 8, 9], &[2], 0);
    assert_eq!(
        (q.peer_bytes, q.work.entropy_bytes, s.exchanges()),
        (0, 0, 0)
    );
    let native = working
        .split_native(dealer.budget.budget(), q.total_reserved_bytes)
        .unwrap();
    let (mask, coordinator, [mut left, mut right]) = construct_with(
        &mut dealer,
        s,
        input,
        native,
        CausalSoftmaxWorkLimit(q.work.total),
        |_| panic!("public singleton requested entropy"),
    )
    .unwrap();
    assert!(matches!(left.begin(&mut || false).unwrap(), Step::Ready));
    assert!(matches!(left.finish(), Err(Error::Consumed)));
    assert!(matches!(right.begin(&mut || false), Err(Error::Consumed)));
    assert!(mask.descriptor().is_err());
    drop((mask, coordinator, right, dealer, working));
    assert_eq!(
        root.budget().used_bytes(),
        OwnedBudget::resources().unwrap().initial_reserved_bytes
    );
}

#[test]
fn late_parent_cancel_revokes_child_after_both_provisional_receipts() {
    // Independent H regression is mounted separately; this also tests every
    // outer round and both callback cancellation roles on fresh genuine owners.
    for k in [2, 3] {
        for role in 0..2 {
            for stop in 0..16 + 2 * k {
                let values = if k == 2 { vec![0, 1] } else { vec![0, 1, 2] };
                let active = if k == 2 { vec![0, 1] } else { vec![0, 1, 2] };
                let (_root, _dealer, _s, _q, (mask, mut coordinator, [mut left, mut right])) =
                    issued(&values, &active);
                let mut a = frame(left.begin(&mut || false).unwrap());
                let mut b = frame(right.begin(&mut || false).unwrap());
                for _ in 0..stop {
                    let ab = a.bytes().unwrap().to_vec();
                    let bb = b.bytes().unwrap().to_vec();
                    a = frame(left.advance(&bb, &mut || false).unwrap());
                    b = frame(right.advance(&ab, &mut || false).unwrap());
                }
                let peer = if role == 0 {
                    b.bytes().unwrap().to_vec()
                } else {
                    a.bytes().unwrap().to_vec()
                };
                let result = if role == 0 {
                    left.advance(&peer, &mut || true)
                } else {
                    right.advance(&peer, &mut || true)
                };
                assert!(matches!(result, Err(Error::Consumed)));
                assert!(a.bytes().is_err());
                assert!(b.bytes().is_err());
                assert!(mask.child.as_ref().unwrap().descriptor().is_err());
                coordinator.cancel();
                assert!(left.advance(&peer, &mut || false).is_err());
                assert!(right.advance(&peer, &mut || false).is_err());
            }
        }
    }
}

#[test]
fn callback_unwind_and_coordinator_drop_burn_both_without_suffix_retry() {
    for role in 0..2 {
        let (_root, _dealer, _s, _q, (mask, coordinator, [mut left, mut right])) =
            issued(&[0, 1], &[0, 1]);
        let a = frame(left.begin(&mut || false).unwrap());
        let b = frame(right.begin(&mut || false).unwrap());
        let peer = if role == 0 {
            b.bytes().unwrap().to_vec()
        } else {
            a.bytes().unwrap().to_vec()
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            if role == 0 {
                let _ = left.advance(&peer, &mut || panic!("callback"));
            } else {
                let _ = right.advance(&peer, &mut || panic!("callback"));
            }
        }));
        assert!(result.is_err());
        assert!(a.bytes().is_err());
        assert!(mask.child.as_ref().unwrap().descriptor().is_err());
        drop(coordinator);
        assert!(right.advance(&peer, &mut || false).is_err());
    }
}

#[test]
fn original_counter_source_residency_lasts_through_last_retained_frame() {
    let (root, dealer, s, _q, (mask, coordinator, roles)) = issued(&[0, 1], &[0, 1]);
    let observer = root.try_clone().unwrap();
    let identity = root.budget() as *const Budget;
    let (outputs, mut frames) = complete(roles, &s);
    let last = frames.pop().unwrap();
    {
        let state = outputs[0].slot.state.lock().unwrap();
        assert!(std::ptr::eq(
            state.keepalive.source.input.budget(),
            identity
        ));
        assert_eq!(&state.keepalive.source.input.payload()[..], &[0, 1]);
    }
    drop((outputs, frames, mask, coordinator, dealer, root));
    assert!(
        observer.budget().used_bytes() > OwnedBudget::resources().unwrap().initial_reserved_bytes
    );
    assert!(last.bytes().is_err());
    drop(last);
    assert_eq!(
        observer.budget().used_bytes(),
        OwnedBudget::resources().unwrap().initial_reserved_bytes
    );
}

#[test]
fn runtime_allocation_failure_retires_both_original_children() {
    let (root, _dealer, _s, _q, (mask, _coordinator, [mut left, mut right])) =
        issued(&[0, 1], &[0, 1]);
    let a = frame(left.begin(&mut || false).unwrap());
    let b = frame(right.begin(&mut || false).unwrap());
    let ab = a.bytes().unwrap().to_vec();
    let bb = b.bytes().unwrap().to_vec();
    let a = frame(left.advance(&bb, &mut || false).unwrap());
    let b = frame(right.advance(&ab, &mut || false).unwrap());
    // The actual next native Select output or parent-affine allocation fails;
    // no stand-in allocation or synthetic child receipt is used.
    let peer = b.bytes().unwrap().to_vec();
    root.budget().fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(
        left.advance(&peer, &mut || false),
        Err(Error::Allocation)
    ));
    root.budget().fail_after.store(-1, Ordering::SeqCst);
    assert!(a.bytes().is_err());
    assert!(b.bytes().is_err());
    assert!(mask.child.as_ref().unwrap().descriptor().is_err());
    assert!(right.advance(&peer, &mut || false).is_err());
}

#[test]
fn whole_original_signed38_source_including_inactive_slots_rejects_before_draws() {
    for bad in [vec![0, 1i64 << 37, 1], vec![0, -(1i64 << 37) - 1, 1]] {
        let (root, mut dealer, s, input, mut working, q) = prepared(&[0, 0, 1], &[0, 2], 0);
        drop(input);
        let bad = root
            .charge_external(3 * size_of::<i64>())
            .unwrap()
            .try_build(|budget| {
                let mut v = crate::shared_allocation::allocate(3, budget)?;
                v.extend_from_slice(&bad);
                Ok(Zeroizing::new(v))
            })
            .unwrap();
        let native = working
            .split_native(dealer.budget.budget(), q.total_reserved_bytes)
            .unwrap();
        let mut draws = 0;
        root.budget().fail_after.store(0, Ordering::SeqCst);
        assert!(matches!(
            construct_with(
                &mut dealer,
                s,
                bad,
                native,
                CausalSoftmaxWorkLimit(q.work.total),
                |_| {
                    draws += 1;
                    Ok(())
                }
            ),
            Err(BuildError(Failure::Native(Error::Domain)))
        ));
        assert_eq!(draws, 0);
        root.budget().fail_after.store(-1, Ordering::SeqCst);
    }
}

#[test]
fn public_native_output_cannot_escape_one_ready_or_wrong_invocation() {
    let (_root, _dealer, s, _q, (mask, _coordinator, [mut left, mut right])) =
        issued(&[0, 1], &[0, 1]);
    let (_r2, _d2, _s2, _q2, (_m2, _c2, [mut foreign, _other])) = issued(&[0, 1], &[0, 1]);
    let foreign_frame = frame(foreign.begin(&mut || false).unwrap());
    let old = frame(left.begin(&mut || false).unwrap());
    assert!(left
        .advance(foreign_frame.bytes().unwrap(), &mut || false)
        .is_err());
    assert!(old.bytes().is_err());
    assert!(mask.descriptor().is_err());
    assert!(right.begin(&mut || false).is_err());
    let (_root, _dealer, _s, _q, (mask, _coordinator, [mut left, mut right])) =
        issued(&[0, 1], &[0, 1]);
    let mut a = frame(left.begin(&mut || false).unwrap());
    let mut b = frame(right.begin(&mut || false).unwrap());
    for _ in 0..s.exchanges() - 1 {
        let ab = a.bytes().unwrap().to_vec();
        let bb = b.bytes().unwrap().to_vec();
        a = frame(left.advance(&bb, &mut || false).unwrap());
        b = frame(right.advance(&ab, &mut || false).unwrap());
    }
    let peer = b.bytes().unwrap().to_vec();
    assert!(matches!(
        left.advance(&peer, &mut || false).unwrap(),
        Step::Ready
    ));
    assert!(matches!(left.finish(), Err(Error::Consumed)));
    assert!(right.advance(&peer, &mut || false).is_err());
    assert!(mask.child.as_ref().unwrap().descriptor().is_err());
}

#[test]
fn actual_constructor_entropy_failures_and_written_draw_unwinds_retire_original_guards() {
    let (root, mut dealer, s, input, mut working, q) = prepared(&[0, 1], &[0, 1], 0);
    let native = working
        .split_native(dealer.budget.budget(), q.total_reserved_bytes)
        .unwrap();
    let mut count = 0;
    let first = construct_with(
        &mut dealer,
        s,
        input,
        native,
        CausalSoftmaxWorkLimit(q.work.total),
        |bytes| {
            count += 1;
            getrandom::fill(bytes).map_err(|_| Error::Randomness)
        },
    )
    .unwrap();
    drop((first, dealer, working));
    assert!(count > 0);
    assert_eq!(
        root.budget().used_bytes(),
        OwnedBudget::resources().unwrap().initial_reserved_bytes
    );
    for fail in 0..count {
        let (root, mut dealer, s, input, mut working, q) = prepared(&[0, 1], &[0, 1], 0);
        let native = working
            .split_native(dealer.budget.budget(), q.total_reserved_bytes)
            .unwrap();
        let mut calls = 0;
        let result = construct_with(
            &mut dealer,
            s,
            input,
            native,
            CausalSoftmaxWorkLimit(q.work.total),
            |bytes| {
                let current = calls;
                calls += 1;
                getrandom::fill(bytes).map_err(|_| Error::Randomness)?;
                if current == fail {
                    return Err(Error::Randomness);
                }
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(calls, fail + 1);
        drop((dealer, working));
        assert_eq!(
            root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    }
    for fail in [0, count / 2, count - 1] {
        let (root, mut dealer, s, input, mut working, q) = prepared(&[0, 1], &[0, 1], 0);
        let native = working
            .split_native(dealer.budget.budget(), q.total_reserved_bytes)
            .unwrap();
        let mut calls = 0;
        let result = catch_unwind(AssertUnwindSafe(|| {
            construct_with(
                &mut dealer,
                s,
                input,
                native,
                CausalSoftmaxWorkLimit(q.work.total),
                |bytes| {
                    let current = calls;
                    calls += 1;
                    getrandom::fill(bytes).map_err(|_| Error::Randomness)?;
                    assert_ne!(current, fail, "genuine filled entropy draw unwind");
                    Ok(())
                },
            )
        }));
        assert!(result.is_err());
        assert_eq!(calls, fail + 1);
        drop((dealer, working));
        assert_eq!(
            root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    }
}

#[test]
fn each_real_constructor_allocation_failure_rolls_back_without_counter_replacement() {
    let (root, mut dealer, s, input, mut working, q) = prepared(&[0, 1], &[0, 1], 0);
    let native = working
        .split_native(dealer.budget.budget(), q.total_reserved_bytes)
        .unwrap();
    root.budget().fail_after.store(10000, Ordering::SeqCst);
    let first = construct(
        &mut dealer,
        s,
        input,
        native,
        CausalSoftmaxWorkLimit(q.work.total),
    )
    .unwrap();
    let calls = 10000 - root.budget().fail_after.load(Ordering::SeqCst);
    root.budget().fail_after.store(-1, Ordering::SeqCst);
    drop((first, dealer, working));
    assert!(calls > 0);
    for fail in 0..calls {
        let (root, mut dealer, s, input, mut working, q) = prepared(&[0, 1], &[0, 1], 0);
        let native = working
            .split_native(dealer.budget.budget(), q.total_reserved_bytes)
            .unwrap();
        root.budget().fail_after.store(fail, Ordering::SeqCst);
        let result = construct(
            &mut dealer,
            s,
            input,
            native,
            CausalSoftmaxWorkLimit(q.work.total),
        );
        assert!(result.is_err(), "allocation {fail}");
        root.budget().fail_after.store(-1, Ordering::SeqCst);
        drop((result, dealer, working));
        assert_eq!(
            root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    }
}

#[test]
fn parent_poll_is_outside_all_locks_and_children_never_reenter_callback() {
    let (_root, _dealer, s, _q, (_mask, _coordinator, [mut left, mut right])) =
        issued(&[0, 1], &[0, 1]);
    let shared = Arc::clone(&left.shared);
    let mut polls = 0;
    let mut callback = || {
        assert!(shared.control.try_lock().is_ok());
        for slot in &shared.slots {
            assert!(slot.state.try_lock().is_ok());
        }
        polls += 1;
        false
    };
    let mut a = left.begin(&mut callback).unwrap();
    let mut b = right.begin(&mut callback).unwrap();
    for _ in 0..s.exchanges() {
        let (Step::Frame(af), Step::Frame(bf)) = (a, b) else {
            panic!("early Ready")
        };
        let ab = af.bytes().unwrap().to_vec();
        let bb = bf.bytes().unwrap().to_vec();
        a = left.advance(&bb, &mut callback).unwrap();
        b = right.advance(&ab, &mut callback).unwrap();
    }
    assert!(matches!(a, Step::Ready));
    assert!(matches!(b, Step::Ready));
    assert_eq!(polls, 2 + 2 * s.exchanges());
    let outputs = [left.finish().unwrap(), right.finish().unwrap()];
    drop(outputs);
}
