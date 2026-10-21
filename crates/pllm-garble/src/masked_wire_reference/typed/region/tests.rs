use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use std::sync::atomic::Ordering;

fn source() -> SourceIdentity {
    SourceIdentity::new([11; 32], [12; 32], approved_implementation())
}
fn dealer(budget: &Budget) -> LocalDealer<'_> {
    LocalDealer::new(source(), [1; 32], [2; 32], budget).unwrap()
}
fn edge(id: u8, shape: &[usize], bits: u8, f: u8, predicate: bool) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [1; 32], [id.wrapping_add(64); 32], [id; 32], 0),
        Shape::new(shape).unwrap(),
        Numeric::new(
            bits,
            f,
            if predicate {
                Interpretation::Predicate
            } else {
                Interpretation::Signed
            },
        )
        .unwrap(),
        if predicate {
            Representation::CommonXorMasked
        } else {
            Representation::CommonAdditiveMasked
        },
        None,
    )
    .unwrap()
}
fn spec(bits: u8, shape: &[usize]) -> PrefixSpec {
    PrefixSpec::new(
        [7; 32],
        edge(3, shape, bits, 12, false),
        edge(4, shape, bits - 6, 6, false),
        edge(5, shape, 1, 0, true),
        edge(6, shape, bits - 6, 6, false),
        [8; 32],
    )
    .unwrap()
}
fn root<'b>(
    d: &mut LocalDealer<'b>,
    s: PrefixSpec,
    values: &[i64],
) -> (DealerMask<'b, Ring>, [CommonMasked<'b, Ring>; 2]) {
    let half = 1i128 << (s.input.numeric().ring_bits() - 1);
    d.prepare_root(s.input, BoundRange::new(-half, half).unwrap(), s.operation)
        .unwrap()
        .encode(values)
        .unwrap()
}
fn run<'b>(
    mut parties: [ActivationPrefixParty<'b>; 2],
    inputs: [CommonMasked<'b, Ring>; 2],
) -> [CommonMasked<'b, Ring>; 2] {
    let [a, b] = inputs;
    let [left, right] = &mut parties;
    let mut l = left.begin(a).unwrap();
    let mut r = right.begin(b).unwrap();
    let mut exchanges = 0;
    loop {
        exchanges += 1;
        match (
            left.advance(r.as_bytes()).unwrap(),
            right.advance(l.as_bytes()).unwrap(),
        ) {
            (PrefixStep::Frame(a), PrefixStep::Frame(b)) => {
                l = a;
                r = b;
            }
            (PrefixStep::Output(a), PrefixStep::Output(b)) => {
                assert_eq!(exchanges, 4);
                assert_eq!(left.ledger.next, 4);
                assert_eq!(right.ledger.next, 4);
                assert!(left.x.is_none() && left.y.is_none() && left.d.is_none());
                assert_eq!(left.advance(&[]).err(), Some(Error::Consumed));
                return [a, b];
            }
            _ => panic!("prefix phase mismatch"),
        }
    }
}
fn unmask(mask: &DealerMask<'_, Ring>, common: &CommonMasked<'_, Ring>) -> Vec<i128> {
    assert_eq!(mask.binding, common.binding);
    let bits = mask.binding.descriptor.numeric().ring_bits();
    let modulus = 1i128 << bits;
    common
        .backing
        .values()
        .iter()
        .zip(mask.backing.values())
        .map(|(&x, &r)| {
            let value = i128::from(x.wrapping_sub(r) & super::super::super::mask(bits));
            if value >= modulus / 2 {
                value - modulus
            } else {
                value
            }
        })
        .collect()
}
fn incremental_peak(s: PrefixSpec) -> usize {
    let (tr, d, select) = s.native_specs();
    arithmetic::resources(tr)
        .unwrap()
        .combined_reservation_bytes
        + arithmetic::resources(d).unwrap().combined_reservation_bytes
        + selection::resources(select)
            .unwrap()
            .combined_reservation_bytes
        + 2 * party_wrapper_charge()
        + mask_charge()
        + issuance_workspace_charge()
}

#[test]
fn independent_i128_floor_and_conditional_selection_endpoints() {
    for bits in [12, 13, 16, 32, 64] {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let half = 1i128 << (bits - 1);
            let values = [
                -half,
                -half + 1,
                -129,
                -128,
                -127,
                -65,
                -64,
                -63,
                -1,
                0,
                1,
                63,
                64,
                65,
                half - 1,
            ]
            .map(|v| v as i64);
            let s = spec(bits, &[values.len()]);
            let mut d = dealer(&budget);
            let (mask, inputs) = root(&mut d, s, &values);
            let (mask, parties) = d.activation_prefix(s, mask).unwrap();
            let outputs = run(parties, inputs);
            let expected: Vec<_> = values
                .iter()
                .map(|&x| {
                    // Independent signed floor, not native rescale/signed helpers.
                    let floor = i128::from(x).div_euclid(64);
                    if floor >= 0 {
                        floor
                    } else {
                        0
                    }
                })
                .collect();
            for output in &outputs {
                assert_eq!(output.descriptor(), s.r);
                assert_eq!(unmask(&mask, output), expected);
                assert_eq!(output.binding.consumer, [8; 32]);
            }
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn native_drelu_predicate_is_true_at_zero_and_false_at_negative_floor_boundary() {
    for bits in [12, 16, 64] {
        for mask_bit in [0u8, 1] {
            let budget = Budget::new(64 << 20).unwrap();
            {
                let mut d = dealer(&budget);
                let values: Vec<i64> = (-1..64).collect();
                let s = spec(bits, &[values.len()]);
                let (mask, [x0, x1]) = root(&mut d, s, &values);
                // Pin the ORIGINAL native mask through the existing private
                // issuance entropy hook, not a new getter or a second issuance.
                // Native DReLU draws each output mask as LE u64 & 1, so repeated
                // bytes 0/1 give exactly mask_bit for every predicate lane.
                let (mask, mut parties) = d
                    .activation_prefix_with(s, mask, |bytes| {
                        bytes.fill(mask_bit);
                        Ok(())
                    })
                    .unwrap();
                let bindings = parties.each_ref().map(|party| party.internal[1]);
                assert_eq!(bindings[0], bindings[1]);
                let mut frames = [parties[0].begin(x0).unwrap(), parties[1].begin(x1).unwrap()];
                // Two TR exchanges, then DReLU: D is retained, Select not finished.
                for _ in 0..3 {
                    frames = next_frames(&mut parties, &frames);
                }
                for (role, party) in parties.iter().enumerate() {
                    assert!(party.phase == Phase::Select);
                    assert!(party.drelu.is_none());
                    assert_eq!(party.role, role as u8);
                    assert_eq!(party.ledger.role, role as u8);
                    assert_eq!(party.ledger.next, 4);
                    assert_eq!(party.internal[1], bindings[role]);
                    assert_eq!(party.ledger.bindings[2], bindings[role]);
                    assert_eq!(bindings[role].descriptor, s.d);
                    assert_eq!(bindings[role].consumer, s.digest());
                    assert_eq!(bindings[role].interval, None);
                    let native_d = party.d.as_ref().unwrap().values();
                    assert_eq!(native_d.len(), values.len());
                    // Check actual region-produced native output, not final R:
                    // both D=0 and D=1 would select R=0 when Y=0.
                    assert_eq!(native_d[0] ^ u64::from(mask_bit), 0, "X=-1, Y=-1");
                    for (&x, &masked_d) in values[1..].iter().zip(&native_d[1..]) {
                        assert_eq!(masked_d ^ u64::from(mask_bit), 1, "X={x}, Y=0");
                    }
                }
                assert_eq!(
                    parties[0].d.as_ref().unwrap().values(),
                    parties[1].d.as_ref().unwrap().values()
                );
                for party in &mut parties {
                    party.cancel();
                }
                drop((mask, parties, frames));
            }
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn floor_half_open_hulls_and_absent_provenance_stays_absent() {
    for range in [None, Some((-129, -63)), Some((-65, 65)), Some((64, 129))] {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let s = spec(16, &[1]);
            let value = range.map_or(0, |(lo, _)| lo as i64);
            let (mut mask, mut inputs) = root(&mut d, s, &[value]);
            // White-box absence models a RingOnly matrix output; absence cannot
            // be created through a public range-assertion bypass.
            mask.binding.interval = range.map(|(lower, upper)| Interval { lower, upper });
            for input in &mut inputs {
                input.binding.interval = mask.binding.interval;
            }
            let expected = range.map(|(lo, hi)| Interval {
                lower: lo.div_euclid(64).min(0),
                upper: ((hi - 1).div_euclid(64) + 1).max(1),
            });
            let (mask, parties) = d.activation_prefix(s, mask).unwrap();
            assert_eq!(mask.binding.interval, expected);
            for out in run(parties, inputs) {
                assert_eq!(out.binding.interval, expected);
            }
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn lane_boundaries_and_numeric_contract_not_relaxed() {
    // Existing core metadata disallows Q12 in n<12, even though native gates
    // would support n>=9. Do not silently alter that numeric contract.
    for bits in 9..12 {
        assert!(Numeric::new(bits, 12, Interpretation::Signed).is_err());
        let mut narrow = spec(16, &[1]);
        narrow.input = edge(3, &[1], bits, bits, false);
        assert_eq!(narrow.validate(), Err(Error::Domain));
    }
    for lanes in [1, 256] {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let s = spec(12, &[lanes]);
            let values: Vec<_> = (0..lanes).map(|i| i as i64 - 128).collect();
            let (mask, inputs) = root(&mut d, s, &values);
            let (mask, parties) = d.activation_prefix(s, mask).unwrap();
            for out in run(parties, inputs) {
                assert_eq!(
                    unmask(&mask, &out),
                    values
                        .iter()
                        .map(|&x| i128::from(x).div_euclid(64).max(0))
                        .collect::<Vec<_>>()
                );
            }
        }
        assert_eq!(budget.used_bytes(), 0);
    }
    let mut s = spec(16, &[1]);
    s.input = edge(3, &[257], 16, 12, false);
    assert_eq!(s.validate(), Err(Error::Domain));
    s = spec(16, &[1]);
    s.input = edge(3, &[1], 16, 11, false);
    assert_eq!(s.validate(), Err(Error::Domain));
    s = spec(16, &[1]);
    s.r = edge(6, &[1], 16, 6, false);
    assert_eq!(s.validate(), Err(Error::Domain));
}

#[test]
fn metadata_substitution_rejected_before_allocation() {
    for variant in 0..8 {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let mut s = spec(16, &[1]);
            let (mask, inputs) = root(&mut d, s, &[0]);
            let original = s.y;
            let identity = original.identity();
            match variant {
                0 => {
                    s.y = EdgeDescriptor::new(
                        EdgeIdentity::new(
                            SourceIdentity::new([99; 32], [12; 32], approved_implementation()),
                            [1; 32],
                            identity.producer(),
                            identity.wire(),
                            0,
                        ),
                        original.shape(),
                        original.numeric(),
                        original.representation(),
                        None,
                    )
                    .unwrap()
                }
                1 => {
                    s.y = EdgeDescriptor::new(
                        EdgeIdentity::new(
                            source(),
                            [99; 32],
                            identity.producer(),
                            identity.wire(),
                            0,
                        ),
                        original.shape(),
                        original.numeric(),
                        original.representation(),
                        None,
                    )
                    .unwrap()
                }
                2 => {
                    s.y = EdgeDescriptor::new(
                        EdgeIdentity::new(
                            source(),
                            [1; 32],
                            identity.producer(),
                            identity.wire(),
                            1,
                        ),
                        original.shape(),
                        original.numeric(),
                        original.representation(),
                        None,
                    )
                    .unwrap()
                }
                3 => s.y = s.r,
                4 => s.d = edge(5, &[1], 1, 0, false),
                5 => s.operation = [99; 32],
                6 => d.session = [99; 32],
                7 => s.input = edge(9, &[1], 16, 12, false),
                _ => unreachable!(),
            }
            budget.fail_after.store(0, Ordering::SeqCst);
            assert!(matches!(
                d.activation_prefix(s, mask).err(),
                Some(Error::Context | Error::Domain | Error::Unsupported)
            ));
            assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
            drop(inputs);
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn private_ledger_burns_wrong_order_and_never_accepts_public_redeclaration() {
    let s = spec(16, &[1]);
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let x = d.binding(s.input, s.operation, None).unwrap();
    let y = d.binding(s.y, s.digest(), None).unwrap();
    let predicate = d.binding(s.d, s.digest(), None).unwrap();
    let mut ledger = Ledger {
        uses: s.uses(),
        bindings: [x, y, predicate, y],
        role: 0,
        next: 0,
    };
    let bound_digest = ledger.digest();
    ledger.role = 1;
    assert_ne!(ledger.digest(), bound_digest);
    ledger.role = 0;
    ledger.bindings[1].mask_id[0] ^= 1;
    assert_ne!(ledger.digest(), bound_digest);
    ledger.bindings[1] = y;
    ledger.bindings[1].interval = Some(Interval {
        lower: -1,
        upper: 1,
    });
    assert_ne!(ledger.digest(), bound_digest);
    ledger.bindings[1] = y;
    assert_eq!(ledger.digest(), bound_digest);
    assert_eq!(ledger.burn(1, y, 0), Err(Error::Consumed));
    assert_eq!(ledger.next, 1);
    assert_eq!(ledger.burn(0, x, 0), Err(Error::Consumed));
    assert_eq!(ledger.next, 2);
    let uses = s.uses();
    assert_eq!(uses[0].operand_position(), 0);
    assert_eq!(uses[3].operand_position(), 1);
    assert_ne!(uses[1].digest(), uses[3].digest());
    for (i, use_) in uses.iter().enumerate() {
        assert_eq!(use_.ordinal(), i as u32);
    }
}

#[test]
fn wrong_role_binding_duplicate_begin_and_advance_before_begin_burn_region() {
    for variant in 0..5 {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let s = spec(16, &[1]);
            let (mask, [mut x0, x1]) = root(&mut d, s, &[0]);
            let (mask, [mut a, mut b]) = d.activation_prefix(s, mask).unwrap();
            match variant {
                0 => {
                    x0.role = 1;
                    assert_eq!(a.begin(x0).err(), Some(Error::Context));
                }
                1 => {
                    x0.binding.mask_id[0] ^= 1;
                    assert_eq!(a.begin(x0).err(), Some(Error::Context));
                }
                2 => {
                    x0.binding.session[0] ^= 1;
                    assert_eq!(a.begin(x0).err(), Some(Error::Context));
                }
                3 => {
                    let f = a.begin(x0).unwrap();
                    // Fresh root is a distinct external capability, not a clone.
                    let (other_mask, [other, sibling]) = root(&mut d, s, &[0]);
                    assert_eq!(a.begin(other).err(), Some(Error::Consumed));
                    drop((f, other_mask, sibling));
                }
                4 => {
                    assert_eq!(a.advance(&[]).err(), Some(Error::Consumed));
                    drop(x0);
                }
                _ => unreachable!(),
            }
            assert!(a.x.is_none() && a.y.is_none() && a.d.is_none() && a.lease.is_none());
            assert_eq!(a.advance(&[]).err(), Some(Error::Consumed));
            b.cancel();
            drop((mask, x1));
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn malformed_reordered_and_cross_region_frames_burn_all_retained_state() {
    for variant in 0..3 {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let s = spec(16, &[1]);
            let (mask, [x0, x1]) = root(&mut d, s, &[-1]);
            let (mask, [mut a, mut b]) = d.activation_prefix(s, mask).unwrap();
            let l = a.begin(x0).unwrap();
            let r = b.begin(x1).unwrap();
            match variant {
                0 => {
                    assert!(a.advance(&[]).is_err());
                }
                1 => {
                    assert!(matches!(a.advance(r.as_bytes()), Ok(PrefixStep::Frame(_))));
                    assert!(a.advance(r.as_bytes()).is_err());
                }
                2 => {
                    let (other_mask, [other0, other1]) = root(&mut d, s, &[-1]);
                    let (other_mask, [mut c, mut e]) = d.activation_prefix(s, other_mask).unwrap();
                    let f = c.begin(other0).unwrap();
                    let g = e.begin(other1).unwrap();
                    assert_eq!(a.advance(g.as_bytes()).err(), Some(Error::Context));
                    c.cancel();
                    e.cancel();
                    drop((other_mask, f, g));
                }
                _ => unreachable!(),
            }
            assert!(a.x.is_none() && a.y.is_none() && a.d.is_none());
            b.cancel();
            drop((l, r, mask));
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn exact_preflight_and_one_byte_short_with_no_allocation() {
    let s = spec(16, &[1]);
    let setup = {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let owners = root(&mut d, s, &[0]);
        let used = budget.used_bytes();
        drop(owners);
        used
    };
    for short in [false, true] {
        let budget = Budget::new(setup + incremental_peak(s) - usize::from(short)).unwrap();
        {
            let mut d = dealer(&budget);
            let (mask, inputs) = root(&mut d, s, &[0]);
            assert_eq!(budget.used_bytes(), setup);
            if short {
                budget.fail_after.store(0, Ordering::SeqCst);
                assert_eq!(d.activation_prefix(s, mask).err(), Some(Error::Resource));
                assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
                drop(inputs);
            } else {
                let (mask, parties) = d.activation_prefix(s, mask).unwrap();
                drop((mask, run(parties, inputs)));
            }
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn injected_issue_allocation_failures_refund_only_after_owner_retirement() {
    // All three native issuances, including failure after earlier entropy/key
    // generation. No external input is resurrected after dealer-mask failure.
    for fail in 0..24 {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let s = spec(16, &[2]);
            let (mask, inputs) = root(&mut d, s, &[-1, 64]);
            budget.fail_after.store(fail, Ordering::SeqCst);
            let issued = d.activation_prefix(s, mask);
            if let Err(error) = &issued {
                assert_eq!(*error, Error::Allocation);
            }
            drop(issued);
            assert!(budget.used_bytes() > baseline); // original external commons
            drop(inputs);
            assert_eq!(budget.used_bytes(), baseline);
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn cancellation_and_exported_frames_keep_native_and_wrapper_charges() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    let s = spec(16, &[1]);
    let (mask, [x0, x1]) = root(&mut d, s, &[64]);
    let (mask, [mut a, mut b]) = d.activation_prefix(s, mask).unwrap();
    let l = a.begin(x0).unwrap();
    let r = b.begin(x1).unwrap();
    a.cancel();
    b.cancel();
    drop((a, b, mask));
    let native = arithmetic::resources(s.native_specs().0).unwrap();
    let expected_frame = native.frame_bytes
        + size_of::<arithmetic::Frame<'static>>()
        + ALLOCATOR
        + frame_wrapper_charge();
    assert_eq!(budget.used_bytes(), baseline + 2 * expected_frame);
    drop(l);
    assert_eq!(budget.used_bytes(), baseline + expected_frame);
    drop(r);
    assert_eq!(budget.used_bytes(), baseline);
    drop(d);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn retained_final_output_keeps_continuous_wrapper_and_native_leases() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    let s = spec(16, &[1]);
    let (mask, inputs) = root(&mut d, s, &[64]);
    let (mask, parties) = d.activation_prefix(s, mask).unwrap();
    let [out0, out1] = run(parties, inputs);
    drop((mask, out1));
    let native_output = 8 + size_of::<selection::MaskedOutput<'static>>() + ALLOCATOR;
    let wrapper = party_wrapper_charge() - 4 * frame_wrapper_charge();
    assert_eq!(budget.used_bytes(), baseline + native_output + wrapper);
    drop(out0);
    assert_eq!(budget.used_bytes(), baseline);
    drop(d);
    assert_eq!(budget.used_bytes(), 0);
}

fn matrix_probe(budget: &Budget) -> Result<[matrix::party::Party<'_>; 2], Error> {
    matrix::party::issue_local(
        matrix::Spec {
            bits: 16,
            rows: 1,
            inner: 1,
            columns: 1,
            layout: matrix::MatrixLayout::InputOutput,
        },
        matrix::Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [91; 32],
            input_wire: [92; 32],
            weight_wire: [93; 32],
            output_wire: [94; 32],
        },
        &[0],
        &[0],
        &[0],
        budget,
    )
}

#[test]
fn matrix_original_backing_guard_retained_then_not_inherited_by_r() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    for cancel_after in [Some(0), Some(1), Some(2), Some(3), None] {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let s = spec(16, &[1, 1]);
            let x = edge(21, &[1, 1], 16, 6, false);
            let plain_w = edge(22, &[1, 1], 16, 6, false);
            let w = EdgeDescriptor::new(
                plain_w.identity(),
                plain_w.shape(),
                plain_w.numeric(),
                plain_w.representation(),
                Some(WeightVisibility::Private),
            )
            .unwrap();
            let (xm, [x0, x1]) = d
                .prepare_root(x, BoundRange::new(0, 65).unwrap(), [20; 32])
                .unwrap()
                .encode(&[64])
                .unwrap();
            let (wm, [w0, w1]) = d
                .prepare_root(w, BoundRange::new(0, 65).unwrap(), [20; 32])
                .unwrap()
                .encode(&[64])
                .unwrap();
            let (xm, [mut m0, mut m1]) = d
                .matrix(
                    [20; 32],
                    matrix::Spec {
                        bits: 16,
                        rows: 1,
                        inner: 1,
                        columns: 1,
                        layout: matrix::MatrixLayout::InputOutput,
                    },
                    xm,
                    wm,
                    s.input,
                    s.operation,
                    MatrixRange::RequireSignedNoWrap,
                )
                .unwrap();
            let ml = m0.begin(x0, w0, &mut || false).unwrap();
            let mr = m1.begin(x1, w1, &mut || false).unwrap();
            let x0 = m0.finish(mr.as_bytes()).unwrap();
            let x1 = m1.finish(ml.as_bytes()).unwrap();
            drop((m0, m1));
            let (mask, [mut a, mut b]) = d.activation_prefix(s, xm).unwrap();
            let mut l = a.begin(x0).unwrap();
            let mut r = b.begin(x1).unwrap();
            // Matrix frames independently retain their original singleton guard.
            assert_eq!(matrix_probe(&budget).err(), Some(Error::Resource));
            drop((ml, mr));
            let mut exchanges = 0;
            let mut retained_frames = Vec::new(); // caller-owned test transport
            loop {
                // Region keeps original X owner even after native TR finishes.
                assert_eq!(matrix_probe(&budget).err(), Some(Error::Resource));
                if cancel_after == Some(exchanges) {
                    a.cancel();
                    b.cancel();
                    break;
                }
                exchanges += 1;
                match (
                    a.advance(r.as_bytes()).unwrap(),
                    b.advance(l.as_bytes()).unwrap(),
                ) {
                    (PrefixStep::Frame(nl), PrefixStep::Frame(nr)) => {
                        retained_frames.push((l, r));
                        l = nl;
                        r = nr;
                    }
                    (PrefixStep::Output(out0), PrefixStep::Output(out1)) => {
                        assert_eq!(exchanges, 4);
                        assert_eq!(unmask(&mask, &out0), [64]);
                        // R is arithmetic, not a synthetic matrix owner. All
                        // prefix frames and both final outputs remain alive.
                        drop(matrix_probe(&budget).unwrap());
                        drop((out0, out1));
                        break;
                    }
                    _ => panic!("phase mismatch"),
                }
            }
            drop(matrix_probe(&budget).unwrap());
            drop((l, r, retained_frames, mask, a, b));
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn execution_allocation_failure_after_retaining_y_burns_local_only() {
    let budget = Budget::new(64 << 20).unwrap();
    {
        let mut d = dealer(&budget);
        let s = spec(16, &[1]);
        let (mask, [x0, x1]) = root(&mut d, s, &[-65]);
        let (mask, [mut a, mut b]) = d.activation_prefix(s, mask).unwrap();
        let l = a.begin(x0).unwrap();
        let r = b.begin(x1).unwrap();
        let PrefixStep::Frame(l2) = a.advance(r.as_bytes()).unwrap() else {
            panic!("TR carry")
        };
        let PrefixStep::Frame(r2) = b.advance(l.as_bytes()).unwrap() else {
            panic!("TR carry")
        };
        // TR output moves its original local backing without allocation; the
        // next DReLU begin frame allocation fails after Y has been retained.
        budget.fail_after.store(0, Ordering::SeqCst);
        assert_eq!(a.advance(r2.as_bytes()).err(), Some(Error::Allocation));
        assert!(a.x.is_none() && a.y.is_none() && a.d.is_none() && a.lease.is_none());
        assert!(b.x.is_some()); // coordinator, not local failure, cancels sibling
        b.cancel();
        drop((mask, l, r, l2, r2));
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn every_native_entropy_failure_burns_mask_and_preserves_only_external_inputs() {
    let s = spec(16, &[2]);
    let (tr, dr, select) = s.native_specs();
    let stage_bytes = [
        arithmetic::resources(tr).unwrap().random_bytes,
        arithmetic::resources(dr).unwrap().random_bytes,
        selection::resources(select).unwrap().random_bytes,
    ];
    let mut failed_stages = [false; 3];
    let mut succeeded = false;
    // First non-failing position is the sentinel: every earlier fill failed once.
    for fail in 0..512 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let (mask, [x0, x1]) = root(&mut d, s, &[-65, 64]);
        let external = x0._lease.bytes + x1._lease.bytes;
        let consumed = mask._lease.bytes;
        assert_eq!(budget.used_bytes(), baseline + external + consumed);
        let mut calls = 0;
        let mut bytes_requested = 0;
        let issued = d.activation_prefix_with(s, mask, |bytes| {
            let position = calls;
            calls += 1;
            bytes_requested += bytes.len();
            if position == fail {
                return Err(Error::Randomness);
            }
            bytes.fill((position as u8).wrapping_add(1));
            Ok(())
        });
        match issued {
            Err(error) => {
                assert_eq!(error, Error::Randomness);
                assert_eq!(calls, fail + 1);
                let stage = if bytes_requested <= stage_bytes[0] {
                    0
                } else if bytes_requested <= stage_bytes[0] + stage_bytes[1] {
                    1
                } else {
                    2
                };
                failed_stages[stage] = true;
            }
            Ok((mask, parties)) => {
                assert_eq!(calls, fail);
                assert_eq!(bytes_requested, stage_bytes.iter().sum::<usize>());
                assert_eq!(failed_stages, [true; 3]);
                let outputs = run(parties, [x0, x1]);
                for output in &outputs {
                    assert_eq!(unmask(&mask, output), [0, 1]);
                }
                drop((mask, outputs));
                assert_eq!(budget.used_bytes(), baseline);
                drop(d);
                assert_eq!(budget.used_bytes(), 0);
                succeeded = true;
                break;
            }
        }
        // Consumed mask cannot be retried; only the two original commons live.
        assert_eq!(budget.used_bytes(), baseline + external);
        let last = x1._lease.bytes;
        drop(x0);
        assert_eq!(budget.used_bytes(), baseline + last);
        drop(x1);
        assert_eq!(budget.used_bytes(), baseline);
        drop(d);
        assert_eq!(budget.used_bytes(), 0);
    }
    assert!(succeeded, "entropy sweep must reach successful sentinel");
}

#[test]
fn malformed_late_descriptors_and_short_preflight_draw_no_entropy() {
    let s = spec(16, &[1]);
    let setup = {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let owners = root(&mut d, s, &[0]);
        let used = budget.used_bytes();
        drop(owners);
        used
    };
    for variant in 0..5 {
        let budget = Budget::new(setup + incremental_peak(s) - usize::from(variant == 4)).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let (mask, inputs) = root(&mut d, s, &[0]);
        let external = inputs.iter().map(|x| x._lease.bytes).sum::<usize>();
        let mut malformed = s;
        match variant {
            0 => malformed.d = edge(5, &[2], 1, 0, true),
            1 => malformed.d = edge(5, &[1], 1, 0, false),
            2 => malformed.r = edge(6, &[2], 10, 6, false),
            3 => malformed.r = edge(6, &[1], 11, 6, false),
            4 => {}
            _ => unreachable!(),
        }
        let mut draws = 0;
        budget.fail_after.store(0, Ordering::SeqCst);
        let error = d
            .activation_prefix_with(malformed, mask, |_| {
                draws += 1;
                Err(Error::Randomness)
            })
            .err();
        if variant == 4 {
            assert_eq!(error, Some(Error::Resource));
        } else {
            assert!(matches!(error, Some(Error::Context | Error::Domain)));
        }
        assert_eq!(draws, 0);
        assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
        assert_eq!(budget.used_bytes(), baseline + external);
        drop(inputs);
        assert_eq!(budget.used_bytes(), baseline);
        drop(d);
        assert_eq!(budget.used_bytes(), 0);
    }
}

fn next_frames<'b>(
    parties: &mut [ActivationPrefixParty<'b>; 2],
    frames: &[PrefixFrame<'b>; 2],
) -> [PrefixFrame<'b>; 2] {
    let [a, b] = parties;
    let PrefixStep::Frame(l) = a.advance(frames[1].as_bytes()).unwrap() else {
        panic!("expected frame")
    };
    let PrefixStep::Frame(r) = b.advance(frames[0].as_bytes()).unwrap() else {
        panic!("expected frame")
    };
    [l, r]
}

#[test]
fn late_drelu_and_select_peer_errors_consume_region_and_require_sibling_cancel() {
    for stage in [2, 3] {
        for variant in 0..4 {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let s = spec(16, &[1]);
            let (mask, [x0, x1]) = root(&mut d, s, &[-65]);
            let (mask, mut parties) = d.activation_prefix(s, mask).unwrap();
            let mut frames = [parties[0].begin(x0).unwrap(), parties[1].begin(x1).unwrap()];
            for _ in 0..stage {
                frames = next_frames(&mut parties, &frames);
            }
            assert!(parties[0].y.is_some());
            assert_eq!(parties[0].d.is_some(), stage == 3);
            match variant {
                0 => assert!(parties[0].advance(&[]).is_err()),
                1 => {
                    let (other_mask, [c0, c1]) = root(&mut d, s, &[-65]);
                    let (other_mask, mut others) = d.activation_prefix(s, other_mask).unwrap();
                    let mut other_frames =
                        [others[0].begin(c0).unwrap(), others[1].begin(c1).unwrap()];
                    for _ in 0..stage {
                        other_frames = next_frames(&mut others, &other_frames);
                    }
                    assert_eq!(
                        parties[0].advance(other_frames[1].as_bytes()).err(),
                        Some(Error::Context)
                    );
                    for other in &mut others {
                        other.cancel();
                    }
                    drop((others, other_frames, other_mask));
                }
                2 => assert_eq!(
                    parties[0].advance(frames[0].as_bytes()).err(),
                    Some(Error::Context)
                ),
                3 => parties[0].cancel(),
                _ => unreachable!(),
            }
            let a = &mut parties[0];
            assert!(a.x.is_none() && a.y.is_none() && a.d.is_none());
            assert!(a.tr.is_none() && a.drelu.is_none() && a.select.is_none());
            assert!(a.lease.is_none());
            assert_eq!(a.ledger.next, 4);
            assert_eq!(a.advance(frames[1].as_bytes()).err(), Some(Error::Consumed));
            assert!(parties[1].y.is_some() && parties[1].lease.is_some());
            parties[1].cancel();
            // Cancellation never refunds independently retained transport frames.
            drop((parties, mask));
            assert!(budget.used_bytes() > baseline);
            drop(frames);
            assert_eq!(budget.used_bytes(), baseline);
            drop(d);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn exact_budget_retains_all_eight_frames_with_per_drop_charge_conservation() {
    for lanes in [1, 3] {
        let s = spec(16, &[lanes]);
        let values = vec![64; lanes];
        let setup = {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let owners = root(&mut d, s, &values);
            let used = budget.used_bytes();
            drop(owners);
            used
        };
        let budget = Budget::new(setup + incremental_peak(s)).unwrap();
        let mut d = dealer(&budget);
        let (mask, [x0, x1]) = root(&mut d, s, &values);
        let (mask, mut parties) = d.activation_prefix(s, mask).unwrap();
        let mut retained = Vec::new(); // Caller-owned transport, never region storage.
        let mut frames = [parties[0].begin(x0).unwrap(), parties[1].begin(x1).unwrap()];
        for _ in 0..3 {
            let next = next_frames(&mut parties, &frames);
            retained.extend(frames);
            frames = next;
        }
        let PrefixStep::Output(out0) = parties[0].advance(frames[1].as_bytes()).unwrap() else {
            panic!("expected R")
        };
        let PrefixStep::Output(out1) = parties[1].advance(frames[0].as_bytes()).unwrap() else {
            panic!("expected R")
        };
        assert_eq!(unmask(&mask, &out0), vec![1; lanes]);
        retained.extend(frames);
        assert_eq!(retained.len(), 8);
        assert_eq!(
            retained.iter().map(|f| f.as_bytes().len()).sum::<usize>(),
            8 * (76 + 8 * lanes)
        );
        let arithmetic_charge = 76
            + 8 * lanes
            + size_of::<arithmetic::Frame<'static>>()
            + ALLOCATOR
            + frame_wrapper_charge();
        let selection_charge = 76
            + 8 * lanes
            + size_of::<selection::Frame<'static>>()
            + ALLOCATOR
            + frame_wrapper_charge();
        for (index, frame) in retained.iter().enumerate() {
            assert_eq!(frame.as_bytes().len(), 76 + 8 * lanes);
            assert_eq!(frame._lease.bytes, frame_wrapper_charge());
            assert_eq!(
                matches!(&frame.native, NativeFrame::Selection(_)),
                index >= 6
            );
        }
        drop((mask, out0, out1, parties, d));
        let mut remaining = 6 * arithmetic_charge + 2 * selection_charge;
        assert_eq!(budget.used_bytes(), remaining);
        while let Some(frame) = retained.pop() {
            let charge = match &frame.native {
                NativeFrame::Arithmetic(_) => arithmetic_charge,
                NativeFrame::Selection(_) => selection_charge,
            };
            drop(frame);
            remaining -= charge;
            assert_eq!(budget.used_bytes(), remaining);
        }
        assert_eq!(remaining, 0);
    }
}

#[test]
fn drelu_then_select_borrow_same_original_y_owner_with_binding_and_lease_continuity() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let s = spec(16, &[2]);
    let (mask, [x0, x1]) = root(&mut d, s, &[-65, 64]);
    let (mask, mut parties) = d.activation_prefix(s, mask).unwrap();
    let mut frames = [parties[0].begin(x0).unwrap(), parties[1].begin(x1).unwrap()];
    for _ in 0..2 {
        frames = next_frames(&mut parties, &frames);
    }
    let owners = parties.each_ref().map(|party| {
        assert!(party.phase == Phase::Drelu);
        assert_eq!(party.ledger.next, 2);
        assert_eq!(party.ledger.bindings[1], party.internal[0]);
        assert_eq!(party.ledger.bindings[3], party.internal[0]);
        assert_eq!(party.internal[0].consumer, s.digest());
        assert_eq!(party.internal[0].descriptor, s.y);
        let y = party.y.as_ref().unwrap();
        (
            y as *const arithmetic::MaskedOutput<'_>,
            y.values().as_ptr(),
            party.internal[0],
            party.lease.as_ref().unwrap().bytes,
        )
    });
    frames = next_frames(&mut parties, &frames);
    for (party, (owner, backing, binding, charged)) in parties.iter().zip(owners) {
        assert!(party.phase == Phase::Select);
        assert_eq!(party.ledger.next, 4);
        let y = party.y.as_ref().unwrap();
        assert_eq!(y as *const arithmetic::MaskedOutput<'_>, owner);
        assert_eq!(y.values().as_ptr(), backing);
        assert_eq!(party.internal[0], binding);
        assert_eq!(party.ledger.bindings[1], binding);
        assert_eq!(party.ledger.bindings[3], binding);
        let lease = party.lease.as_ref().unwrap();
        assert!(std::ptr::eq(lease.budget, &budget));
        assert_eq!(lease.bytes, charged - frame_wrapper_charge());
        assert_eq!(party.ledger.role, party.role);
    }
    let PrefixStep::Output(out0) = parties[0].advance(frames[1].as_bytes()).unwrap() else {
        panic!("expected R")
    };
    let PrefixStep::Output(out1) = parties[1].advance(frames[0].as_bytes()).unwrap() else {
        panic!("expected R")
    };
    for party in &parties {
        assert!(party.y.is_none() && party.d.is_none() && party.lease.is_none());
    }
    drop((mask, parties, frames, out0, out1, d));
    assert_eq!(budget.used_bytes(), 0);
}
