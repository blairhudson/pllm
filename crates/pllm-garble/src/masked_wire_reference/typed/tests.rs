use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use std::sync::atomic::Ordering;

fn source() -> SourceIdentity {
    SourceIdentity::new([11; 32], [12; 32], approved_implementation())
}
fn dealer(budget: &Budget) -> LocalDealer<'_> {
    LocalDealer::new(source(), [1; 32], [2; 32], budget).unwrap()
}
fn edge(id: u8, shape: &[usize], bits: u8, f: u8, private: bool) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [1; 32], [id.wrapping_add(64); 32], [id; 32], 0),
        Shape::new(shape).unwrap(),
        Numeric::new(bits, f, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        if private {
            Some(WeightVisibility::Private)
        } else {
            None
        },
    )
    .unwrap()
}
fn predicate(id: u8, shape: &[usize]) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [1; 32], [id.wrapping_add(64); 32], [id; 32], 0),
        Shape::new(shape).unwrap(),
        Numeric::new(1, 0, Interpretation::Predicate).unwrap(),
        Representation::CommonXorMasked,
        None,
    )
    .unwrap()
}
fn root<'b>(
    d: &mut LocalDealer<'b>,
    e: EdgeDescriptor,
    range: (i128, i128),
    consumer: u8,
    values: &[i64],
) -> (DealerMask<'b, Ring>, [CommonMasked<'b, Ring>; 2]) {
    d.prepare_root(
        e,
        BoundRange::new(range.0, range.1).unwrap(),
        [consumer; 32],
    )
    .unwrap()
    .encode(values)
    .unwrap()
}
fn run<'b, K>(
    mut parties: [ArithmeticParty<'b, K>; 2],
    inputs: [CommonMasked<'b, Ring>; 2],
) -> [CommonMasked<'b, K>; 2] {
    let [a, b] = inputs;
    let [left, right] = &mut parties;
    let mut l = left.begin(a).unwrap();
    let mut r = right.begin(b).unwrap();
    loop {
        match (
            left.advance(r.as_bytes()).unwrap(),
            right.advance(l.as_bytes()).unwrap(),
        ) {
            (Step::Frame(nl), Step::Frame(nr)) => {
                l = nl;
                r = nr;
            }
            (Step::Output(a), Step::Output(b)) => return [a, b],
            _ => panic!("phase mismatch"),
        }
    }
}
fn unmask_ring(mask: &DealerMask<'_, Ring>, output: &CommonMasked<'_, Ring>) -> Vec<i64> {
    assert_eq!(mask.binding, output.binding);
    let bits = mask.binding.descriptor.numeric().ring_bits();
    output
        .backing
        .values()
        .iter()
        .zip(mask.backing.values())
        .map(|(&x, &r)| super::super::signed(x.wrapping_sub(r) & super::super::mask(bits), bits))
        .collect()
}
fn unmask_predicate(
    mask: &DealerMask<'_, Predicate>,
    output: &CommonMasked<'_, Predicate>,
) -> Vec<u64> {
    assert_eq!(mask.binding, output.binding);
    output
        .backing
        .values()
        .iter()
        .zip(mask.backing.values())
        .map(|(&x, &r)| x ^ r)
        .collect()
}

#[test]
fn root_tr_drelu_exact_floor_endpoints_and_zero_true() {
    for bits in [8, 16, 64] {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let values = [-17, -16, -1, 0, 1, 15, 16, 17];
            let (mask, caps) = root(&mut d, edge(3, &[8], bits, 4, false), (-17, 18), 4, &values);
            let (mask, parties) = d
                .tr([4; 32], mask, 2, edge(5, &[8], bits - 2, 2, false), [6; 32])
                .unwrap();
            let caps = run(parties, caps);
            for cap in &caps {
                assert_eq!(unmask_ring(&mask, cap), values.map(|v| v >> 2));
                assert!(matches!(cap.backing, CommonBacking::Arithmetic(_)));
            }
            let (mask, parties) = d.drelu([6; 32], mask, predicate(7, &[8]), [8; 32]).unwrap();
            let caps = run(parties, caps);
            for cap in &caps {
                assert_eq!(unmask_predicate(&mask, cap), vec![0, 0, 0, 1, 1, 1, 1, 1]);
                assert_eq!(
                    cap.descriptor().representation(),
                    Representation::CommonXorMasked
                );
            }
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn gap_full_half_open_quarter_ring_endpoints() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let values = [-64, -63, -1, 0, 1, 62, 63];
    let (mask, caps) = root(&mut d, edge(3, &[7], 8, 3, false), (-64, 64), 4, &values);
    let (mask, parties) = d
        .gap_ars([4; 32], mask, 2, edge(5, &[7], 8, 1, false), [6; 32])
        .unwrap();
    let caps = run(parties, caps);
    for cap in &caps {
        assert_eq!(unmask_ring(&mask, cap), values.map(|v| v >> 2));
    }
    drop((mask, caps));
    // Values are zero, yet the predeclared upper/lower endpoints disqualify.
    for range in [(-65, 64), (-64, 65), (-128, 128)] {
        let (mask, caps) = root(&mut d, edge(3, &[1], 8, 3, false), range, 4, &[0]);
        assert_eq!(
            d.gap_ars([4; 32], mask, 2, edge(5, &[1], 8, 1, false), [6; 32])
                .err(),
            Some(Error::Domain)
        );
        drop(caps);
    }
    let prepared = d
        .prepare_root(
            edge(3, &[1], 8, 3, false),
            BoundRange::new(-64, 64).unwrap(),
            [4; 32],
        )
        .unwrap();
    assert_eq!(prepared.encode(&[64]).err(), Some(Error::Domain));
}

#[test]
fn full_ring64_tr_retains_signed_floor_at_extremes() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let values = [i64::MIN, i64::MIN + 1, -1, 0, i64::MAX - 1, i64::MAX];
    let (mask, caps) = root(
        &mut d,
        edge(3, &[6], 64, 4, false),
        (-(1i128 << 63), 1i128 << 63),
        4,
        &values,
    );
    let (mask, parties) = d
        .tr([4; 32], mask, 2, edge(5, &[6], 62, 2, false), [6; 32])
        .unwrap();
    let caps = run(parties, caps);
    assert_eq!(unmask_ring(&mask, &caps[0]), values.map(|v| v >> 2));
}

#[test]
fn binding_rejects_wrong_mask_context_source_role_q_bound_and_descriptor() {
    for attack in 0..8 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let e = edge(3, &[1], 8, 2, false);
        let (mask, [mut cap, sibling]) = root(&mut d, e, (-2, 3), 4, &[0]);
        let (_mask, [mut left, mut right]) = d
            .tr([4; 32], mask, 1, edge(5, &[1], 7, 1, false), [6; 32])
            .unwrap();
        match attack {
            0 => cap.binding.mask_id[0] ^= 1,
            1 => cap.binding.session[0] ^= 1,
            2 => cap.binding.namespace[0] ^= 1,
            3 => cap.role = 1,
            4 => cap.binding.consumer[0] ^= 1,
            5 => {
                cap.binding.interval = Some(Interval {
                    lower: -1,
                    upper: 2,
                })
            }
            6 => cap.binding.descriptor = edge(3, &[1], 8, 1, false),
            7 => {
                cap.binding.descriptor = EdgeDescriptor::new(
                    EdgeIdentity::new(
                        SourceIdentity::new([99; 32], [12; 32], approved_implementation()),
                        [1; 32],
                        [67; 32],
                        [3; 32],
                        0,
                    ),
                    e.shape(),
                    e.numeric(),
                    e.representation(),
                    None,
                )
                .unwrap()
            }
            _ => unreachable!(),
        }
        assert_eq!(left.begin(cap).err(), Some(Error::Context));
        assert_eq!(left.advance(&[]).err(), Some(Error::Consumed));
        // Failure is local, not silent sibling cancellation.
        let frame = right.begin(sibling).unwrap();
        right.cancel();
        assert_eq!(right.advance(frame.as_bytes()).err(), Some(Error::Consumed));
    }
    // Real independently issued masks cannot be interchanged, even with exactly
    // equal descriptors and clear values; no test-only mutation involved.
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let (mask, caps) = root(&mut d, edge(3, &[1], 8, 2, false), (-1, 2), 4, &[0]);
    let (other_mask, [other, other_sibling]) =
        root(&mut d, edge(3, &[1], 8, 2, false), (-1, 2), 4, &[0]);
    let (_, [mut party, mut sibling]) = d
        .tr([4; 32], mask, 1, edge(5, &[1], 7, 1, false), [6; 32])
        .unwrap();
    assert_eq!(party.begin(other).err(), Some(Error::Context));
    sibling.cancel();
    drop((other_mask, other_sibling, caps));
}

#[test]
fn malformed_peer_burns_and_retained_frames_outputs_keep_budget() {
    for phase in 0..2 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, [a, b]) = root(&mut d, edge(3, &[1], 8, 2, false), (-1, 2), 4, &[0]);
        let (mask, [mut left, mut right]) = d
            .tr([4; 32], mask, 1, edge(5, &[1], 7, 1, false), [6; 32])
            .unwrap();
        let mut l = left.begin(a).unwrap();
        let mut r = right.begin(b).unwrap();
        if phase == 1 {
            let Step::Frame(nl) = left.advance(r.as_bytes()).unwrap() else {
                panic!()
            };
            let Step::Frame(nr) = right.advance(l.as_bytes()).unwrap() else {
                panic!()
            };
            l = nl;
            r = nr;
        }
        assert!(left.advance(&r.as_bytes()[..3]).is_err());
        assert_eq!(left.advance(r.as_bytes()).err(), Some(Error::Consumed));
        right.cancel();
        drop((d, mask, left, right));
        assert!(budget.used_bytes() > 0);
        drop((l, r));
        assert_eq!(budget.used_bytes(), 0);
    }
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let (mask, caps) = root(&mut d, edge(3, &[1], 8, 2, false), (-1, 2), 4, &[0]);
    let (mask, parties) = d
        .tr([4; 32], mask, 1, edge(5, &[1], 7, 1, false), [6; 32])
        .unwrap();
    let [a, b] = run(parties, caps);
    drop((mask, d, b));
    assert!(budget.used_bytes() > common_charge());
    drop(a);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn public_preflight_rejects_profiles_representations_caps_and_numeric_adaptation() {
    let budget = Budget::new(64 << 20).unwrap();
    assert_eq!(
        LocalDealer::new(
            SourceIdentity::new([11; 32], [12; 32], [0; 32]),
            [1; 32],
            [2; 32],
            &budget
        )
        .err(),
        Some(Error::Unsupported)
    );
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    for representation in [Representation::Public, Representation::AdditiveShare] {
        let e = edge(3, &[1], 8, 2, false);
        let e = EdgeDescriptor::new(e.identity(), e.shape(), e.numeric(), representation, None)
            .unwrap();
        assert_eq!(
            d.prepare_root(e, BoundRange::new(-1, 2).unwrap(), [4; 32])
                .err(),
            Some(Error::Unsupported)
        );
    }
    let e = edge(3, &[1], 8, 2, false);
    let e = EdgeDescriptor::new(
        e.identity(),
        e.shape(),
        e.numeric(),
        e.representation(),
        Some(WeightVisibility::Public),
    )
    .unwrap();
    assert_eq!(
        d.prepare_root(e, BoundRange::new(-1, 2).unwrap(), [4; 32])
            .err(),
        Some(Error::Unsupported)
    );
    assert_eq!(
        d.prepare_root(
            edge(3, &[257], 8, 2, false),
            BoundRange::new(-1, 2).unwrap(),
            [4; 32]
        )
        .err(),
        Some(Error::Resource)
    );
    assert_eq!(budget.used_bytes(), baseline);
    for (f, out_f, expected) in [(0, 0, Error::Unsupported), (2, 2, Error::Domain)] {
        let (mask, caps) = root(&mut d, edge(3, &[1], 8, f, false), (-1, 2), 4, &[0]);
        assert_eq!(
            d.tr([4; 32], mask, 1, edge(5, &[1], 7, out_f, false), [6; 32])
                .err(),
            Some(expected)
        );
        drop(caps);
    }
    assert_eq!(budget.used_bytes(), baseline);
}

#[test]
fn exact_capacity_root_and_failure_unwind_never_refund_live_secret_owners() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    let (mask, [a, b]) = root(&mut d, edge(3, &[8], 8, 2, false), (-1, 2), 4, &[0; 8]);
    let MaskBacking::Fresh(v) = &mask.backing else {
        panic!()
    };
    assert_eq!(v.capacity(), 8);
    for cap in [&a, &b] {
        let CommonBacking::Root(v) = &cap.backing else {
            panic!()
        };
        assert_eq!(v.capacity(), 8);
    }
    drop((mask, b));
    assert_eq!(
        budget.used_bytes(),
        baseline + 8 * 8 + ALLOCATOR + common_charge()
    );
    drop(a);
    assert_eq!(budget.used_bytes(), baseline);
    for fail in 0..3 {
        budget.fail_after.store(fail, Ordering::SeqCst);
        assert_eq!(
            d.prepare_root(
                edge(3, &[1], 8, 2, false),
                BoundRange::new(-1, 2).unwrap(),
                [4; 32]
            )
            .err(),
            Some(Error::Allocation)
        );
        assert_eq!(budget.used_bytes(), baseline);
    }
    budget.fail_after.store(-1, Ordering::SeqCst);
    // Native issuance errors retain no detached secret payload or lease.
    for fail in 0..7 {
        let (mask, caps) = root(&mut d, edge(3, &[1], 8, 2, false), (-1, 2), 4, &[0]);
        budget.fail_after.store(fail, Ordering::SeqCst);
        let result = d.tr([4; 32], mask, 1, edge(5, &[1], 7, 1, false), [6; 32]);
        assert_eq!(result.err(), Some(Error::Allocation));
        budget.fail_after.store(-1, Ordering::SeqCst);
        drop(caps);
        assert_eq!(budget.used_bytes(), baseline);
    }
    drop(d);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn aggregate_budget_admission_prices_root_metadata_and_native_overlap() {
    let big = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&big);
    let baseline = big.used_bytes();
    let (mask, caps) = root(&mut d, edge(3, &[1], 8, 2, false), (-1, 2), 4, &[0]);
    let root_charge = big.used_bytes() - baseline;
    drop((d, mask, caps));
    let tiny = Budget::new(baseline + root_charge - 1).unwrap();
    let mut d = dealer(&tiny);
    assert_eq!(
        d.prepare_root(
            edge(3, &[1], 8, 2, false),
            BoundRange::new(-1, 2).unwrap(),
            [4; 32]
        )
        .err(),
        Some(Error::Resource)
    );
    assert_eq!(tiny.used_bytes(), baseline);
    let native = arithmetic::resources(arithmetic::Spec {
        bits: 8,
        lanes: 1,
        gate: arithmetic::Gate::Tr { shift: 1 },
    })
    .unwrap();
    let wrapper_charge =
        2 * (size_of::<ArithmeticParty<'static, Ring>>() + common_charge()) + mask_charge();
    let tiny = Budget::new(
        baseline + root_charge + native.combined_reservation_bytes + wrapper_charge - 1,
    )
    .unwrap();
    let mut d = dealer(&tiny);
    let (mask, caps) = root(&mut d, edge(3, &[1], 8, 2, false), (-1, 2), 4, &[0]);
    assert_eq!(
        d.tr([4; 32], mask, 1, edge(5, &[1], 7, 1, false), [6; 32])
            .err(),
        Some(Error::Resource)
    );
    drop((caps, d));
    assert_eq!(tiny.used_bytes(), 0);
}

#[test]
fn dealer_first_drop_keeps_prepared_roots_and_each_secret_owner_charged() {
    for mask_last in [false, true] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let dealer_bytes = budget.used_bytes();
        let prepared = d
            .prepare_root(
                edge(3, &[4], 8, 2, false),
                BoundRange::new(-1, 2).unwrap(),
                [4; 32],
            )
            .unwrap();
        let common_bytes = 8 * 4 + ALLOCATOR + common_charge();
        let mask_bytes = 8 * 4 + ALLOCATOR + mask_charge() + size_of::<RootInput<'static>>();
        assert_eq!(
            budget.used_bytes(),
            dealer_bytes + mask_bytes + 2 * common_bytes
        );
        drop(d);
        assert_eq!(budget.used_bytes(), mask_bytes + 2 * common_bytes);
        // Prepared roots outlive LocalDealer without retaining any dealer borrow.
        let (mask, [a, b]) = prepared.encode(&[-1, 0, 1, 0]).unwrap();
        assert_eq!(budget.used_bytes(), mask_bytes + 2 * common_bytes);
        if mask_last {
            drop(a);
            assert_eq!(budget.used_bytes(), mask_bytes + common_bytes);
            drop(b);
            assert_eq!(budget.used_bytes(), mask_bytes);
            drop(mask);
        } else {
            drop(mask);
            assert_eq!(budget.used_bytes(), 2 * common_bytes);
            drop(a);
            assert_eq!(budget.used_bytes(), common_bytes);
            drop(b);
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn retained_matrix_common_and_frames_gate_singleton_until_final_owner_drop() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    fn attempt<'b>(
        d: &mut LocalDealer<'b>,
    ) -> Result<(DealerMask<'b, Ring>, [MatrixParty<'b>; 2]), Error> {
        let (xm, x) = root(d, edge(3, &[1, 1], 8, 0, false), (0, 2), 5, &[1]);
        let (wm, w) = root(d, edge(4, &[1, 1], 8, 0, true), (0, 2), 5, &[1]);
        let result = d.matrix(
            [5; 32],
            matrix::Spec {
                bits: 8,
                rows: 1,
                inner: 1,
                columns: 1,
                layout: matrix::MatrixLayout::InputOutput,
            },
            xm,
            wm,
            edge(6, &[1, 1], 8, 0, false),
            [7; 32],
            MatrixRange::RequireSignedNoWrap,
        );
        drop((x, w));
        result
    }
    for common_last in [false, true] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let (xm, [xa, xb]) = root(&mut d, edge(3, &[1, 1], 8, 0, false), (0, 2), 5, &[1]);
        let (wm, [wa, wb]) = root(&mut d, edge(4, &[1, 1], 8, 0, true), (0, 2), 5, &[1]);
        let (mask, [mut a, mut b]) = d
            .matrix(
                [5; 32],
                matrix::Spec {
                    bits: 8,
                    rows: 1,
                    inner: 1,
                    columns: 1,
                    layout: matrix::MatrixLayout::InputOutput,
                },
                xm,
                wm,
                edge(6, &[1, 1], 8, 0, false),
                [7; 32],
                MatrixRange::RequireSignedNoWrap,
            )
            .unwrap();
        let l = a.begin(xa, wa, &mut || false).unwrap();
        let r = b.begin(xb, wb, &mut || false).unwrap();
        let common = a.finish(r.as_bytes()).unwrap();
        let other = b.finish(l.as_bytes()).unwrap();
        drop((a, b, mask, other));
        assert_eq!(attempt(&mut d).err(), Some(Error::Resource));
        if common_last {
            drop((l, r));
            assert_eq!(attempt(&mut d).err(), Some(Error::Resource));
            assert!(budget.used_bytes() > baseline);
            drop(common);
        } else {
            drop(common);
            assert_eq!(attempt(&mut d).err(), Some(Error::Resource));
            drop(l);
            assert_eq!(attempt(&mut d).err(), Some(Error::Resource));
            assert!(budget.used_bytes() > baseline);
            drop(r);
        }
        assert_eq!(budget.used_bytes(), baseline);
        // Neither native material nor a detached lease survives the last owner.
        let next = attempt(&mut d).unwrap();
        drop(next);
        assert_eq!(budget.used_bytes(), baseline);
        drop(d);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn complete_dot_range_not_only_individual_products_must_fit_signed_ring() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    // Two +64 terms overflow signed ring8. Two +63 terms fit at +126;
    // two -64 terms exactly attain the valid inclusive lower endpoint -128.
    // Unit terms additionally distinguish exact upper endpoint +127 from +128.
    for (xv, wv, inner, expected) in [
        (8i64, 8i64, 2, None),
        (7, 9, 2, Some(126)),
        (8, -8, 2, Some(-128)),
        (1, 1, 127, Some(127)),
        (1, 1, 128, None),
    ] {
        let term = i128::from(xv) * i128::from(wv);
        assert!((-128..128).contains(&term));
        let (xm, [xa, xb]) = root(
            &mut d,
            edge(3, &[1, inner], 8, 0, false),
            (i128::from(xv), i128::from(xv) + 1),
            5,
            &vec![xv; inner],
        );
        let (wm, [wa, wb]) = root(
            &mut d,
            edge(4, &[inner, 1], 8, 0, true),
            (i128::from(wv), i128::from(wv) + 1),
            5,
            &vec![wv; inner],
        );
        if expected.is_none() {
            // Public interval rejection must precede even the first allocation.
            budget.fail_after.store(0, Ordering::SeqCst);
        }
        let result = d.matrix(
            [5; 32],
            matrix::Spec {
                bits: 8,
                rows: 1,
                inner,
                columns: 1,
                layout: matrix::MatrixLayout::InputOutput,
            },
            xm,
            wm,
            edge(6, &[1, 1], 8, 0, false),
            [7; 32],
            MatrixRange::RequireSignedNoWrap,
        );
        budget.fail_after.store(-1, Ordering::SeqCst);
        match expected {
            None => {
                assert_eq!(result.err(), Some(Error::Domain));
                drop((xa, xb, wa, wb));
            }
            Some(expected) => {
                let (mask, [mut a, mut b]) = result.unwrap();
                assert_eq!(
                    mask.binding.interval,
                    Some(Interval {
                        lower: expected,
                        upper: expected + 1
                    })
                );
                let l = a.begin(xa, wa, &mut || false).unwrap();
                let r = b.begin(xb, wb, &mut || false).unwrap();
                let output = a.finish(r.as_bytes()).unwrap();
                let other = b.finish(l.as_bytes()).unwrap();
                assert_eq!(unmask_ring(&mask, &output), [expected as i64]);
                assert_eq!(unmask_ring(&mask, &other), [expected as i64]);
                drop((a, b, l, r, mask, output, other));
            }
        }
        assert_eq!(budget.used_bytes(), baseline);
    }
}

// Share native matrix tests' lock; unrelated unit tests can run in parallel.
#[test]
fn private_matrix_to_tr_or_gap_then_drelu_preserves_native_backing() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    for layout in [
        matrix::MatrixLayout::InputOutput,
        matrix::MatrixLayout::OutputInput,
    ] {
        for gap in [false, true] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let s = matrix::Spec {
                bits: 16,
                rows: 2,
                inner: 2,
                columns: 2,
                layout,
            };
            let x = [-4, 3, 0, 1];
            let w = if layout == matrix::MatrixLayout::InputOutput {
                [2, -1, -3, 4]
            } else {
                [2, -3, -1, 4]
            };
            let (xm, [xa, xb]) = root(&mut d, edge(3, &[2, 2], 16, 2, false), (-4, 4), 5, &x);
            let (wm, [wa, wb]) = root(&mut d, edge(4, &[2, 2], 16, 2, true), (-3, 5), 5, &w);
            let (ym, [mut a, mut b]) = d
                .matrix(
                    [5; 32],
                    s,
                    xm,
                    wm,
                    edge(6, &[2, 2], 16, 4, false),
                    [7; 32],
                    MatrixRange::RequireSignedNoWrap,
                )
                .unwrap();
            let l = a.begin(xa, wa, &mut || false).unwrap();
            let r = b.begin(xb, wb, &mut || false).unwrap();
            let ya = a.finish(r.as_bytes()).unwrap();
            let yb = b.finish(l.as_bytes()).unwrap();
            drop((a, b, l, r));
            assert!(matches!(ya.backing, CommonBacking::Matrix(_)));
            assert_eq!(unmask_ring(&ym, &ya), [-17, 16, -3, 4]);
            let (mask, parties) = if gap {
                d.gap_ars([7; 32], ym, 2, edge(8, &[2, 2], 16, 2, false), [9; 32])
                    .unwrap()
            } else {
                d.tr([7; 32], ym, 2, edge(8, &[2, 2], 14, 2, false), [9; 32])
                    .unwrap()
            };
            let caps = run(parties, [ya, yb]);
            assert_eq!(unmask_ring(&mask, &caps[0]), [-5, 4, -1, 1]);
            let (mask, parties) = d
                .drelu([9; 32], mask, predicate(10, &[2, 2]), [11; 32])
                .unwrap();
            let caps = run(parties, caps);
            assert_eq!(unmask_predicate(&mask, &caps[0]), [0, 1, 0, 1]);
            drop((mask, caps, d));
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn ring_only_matrix_can_tr_but_cannot_mint_gap_certificate() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    for gap in [false, true] {
        let (xm, x) = root(&mut d, edge(3, &[1, 1], 8, 2, false), (-128, 128), 5, &[1]);
        let (wm, w) = root(&mut d, edge(4, &[1, 1], 8, 2, true), (-128, 128), 5, &[1]);
        let s = matrix::Spec {
            bits: 8,
            rows: 1,
            inner: 1,
            columns: 1,
            layout: matrix::MatrixLayout::InputOutput,
        };
        let (mask, parties) = d
            .matrix(
                [5; 32],
                s,
                xm,
                wm,
                edge(6, &[1, 1], 8, 4, false),
                [7; 32],
                MatrixRange::RingOnly,
            )
            .unwrap();
        assert_eq!(mask.binding.interval, None);
        // Dropping unused evaluator parties releases the native singleton.
        drop((parties, x, w));
        if gap {
            assert_eq!(
                d.gap_ars([7; 32], mask, 1, edge(8, &[1, 1], 8, 3, false), [9; 32])
                    .err(),
                Some(Error::Domain)
            );
        } else {
            let (mask, parties) = d
                .tr([7; 32], mask, 1, edge(8, &[1, 1], 7, 3, false), [9; 32])
                .unwrap();
            assert_eq!(mask.binding.interval, None);
            drop(parties);
            assert_eq!(
                d.gap_ars([9; 32], mask, 1, edge(10, &[1, 1], 7, 2, false), [11; 32])
                    .err(),
                Some(Error::Domain)
            );
        }
    }
    assert!(matrix_interval(
        Interval {
            lower: i64::MIN as i128,
            upper: (i64::MAX as i128) + 1
        },
        Interval {
            lower: i64::MIN as i128,
            upper: (i64::MAX as i128) + 1
        },
        256,
        64
    )
    .is_err());
}

#[test]
fn matrix_cancellation_malformed_peer_and_allocation_failure_keep_native_lifetimes() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    let s = matrix::Spec {
        bits: 8,
        rows: 1,
        inner: 1,
        columns: 1,
        layout: matrix::MatrixLayout::InputOutput,
    };
    for failure in 0..9 {
        let (xm, x) = root(&mut d, edge(3, &[1, 1], 8, 2, false), (-1, 2), 5, &[0]);
        let (wm, w) = root(&mut d, edge(4, &[1, 1], 8, 2, true), (-1, 2), 5, &[0]);
        // One fresh output-mask buffer then seven native dealer/material
        // buffers. Failure 8 allows successful issuance and exercises drop.
        budget.fail_after.store(failure, Ordering::SeqCst);
        let result = d.matrix(
            [5; 32],
            s,
            xm,
            wm,
            edge(6, &[1, 1], 8, 4, false),
            [7; 32],
            MatrixRange::RequireSignedNoWrap,
        );
        if failure < 8 {
            assert_eq!(result.err(), Some(Error::Allocation));
        } else {
            drop(result.unwrap());
        }
        budget.fail_after.store(-1, Ordering::SeqCst);
        drop((x, w));
        assert_eq!(budget.used_bytes(), baseline);
    }
    for cancel in [false, true] {
        let (xm, [xa, xb]) = root(&mut d, edge(3, &[1, 1], 8, 2, false), (-1, 2), 5, &[0]);
        let (wm, [wa, wb]) = root(&mut d, edge(4, &[1, 1], 8, 2, true), (-1, 2), 5, &[0]);
        let (mask, [mut a, mut b]) = d
            .matrix(
                [5; 32],
                s,
                xm,
                wm,
                edge(6, &[1, 1], 8, 4, false),
                [7; 32],
                MatrixRange::RequireSignedNoWrap,
            )
            .unwrap();
        if cancel {
            assert_eq!(a.begin(xa, wa, &mut || true).err(), Some(Error::Consumed));
            b.cancel();
            drop((xb, wb));
        } else {
            let l = a.begin(xa, wa, &mut || false).unwrap();
            let r = b.begin(xb, wb, &mut || false).unwrap();
            assert!(a.finish(&r.as_bytes()[..3]).is_err());
            assert_eq!(a.finish(r.as_bytes()).err(), Some(Error::Consumed));
            // Sibling remains usable until explicitly retired; its output keeps
            // the full native output owner and singleton after Party drop.
            let output = b.finish(l.as_bytes()).unwrap();
            drop((a, b, l, r, mask));
            assert!(budget.used_bytes() > baseline + common_charge());
            drop(output);
            assert_eq!(budget.used_bytes(), baseline);
            continue;
        }
        drop((a, b, mask));
        assert_eq!(budget.used_bytes(), baseline);
    }
}
