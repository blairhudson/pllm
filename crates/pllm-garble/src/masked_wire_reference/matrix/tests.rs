use super::TEST_SERIAL as TEST;
use super::*;

#[test]
fn local_party_exact_admission_failures_and_retained_singleton() {
    let _guard = TEST.lock().unwrap();
    let s = scalar(8);
    let cost = party::resources(s).unwrap();
    let short = Budget::new(cost.reservation_bytes - 1).unwrap();
    let mut draws = 0;
    assert!(matches!(
        party::issue_with(
            s,
            context(),
            [&[2], &[3], &[4]],
            &short,
            &mut |_| {
                draws += 1;
                Ok(())
            },
            &mut || false
        ),
        Err(Error::Resource)
    ));
    assert_eq!(draws, 0);
    assert_eq!(short.used_bytes(), 0);
    // Initial cancellation must not ask for entropy; every failed random draw
    // must unwind material, scratch, singleton and reservation.
    let b = Budget::new(cost.reservation_bytes).unwrap();
    assert!(matches!(
        party::issue_with(
            s,
            context(),
            [&[2], &[3], &[4]],
            &b,
            &mut |_| {
                draws += 1;
                Ok(())
            },
            &mut || true,
        ),
        Err(Error::Consumed)
    ));
    assert_eq!(draws, 0);
    assert_eq!(b.used_bytes(), 0);
    for fail in 0..4 {
        let mut draws = 0;
        assert!(matches!(
            party::issue_with(
                s,
                context(),
                [&[2], &[3], &[4]],
                &b,
                &mut |bytes| {
                    if draws == fail {
                        return Err(Error::Randomness);
                    }
                    draws += 1;
                    bytes.fill(0);
                    Ok(())
                },
                &mut || false,
            ),
            Err(Error::Randomness)
        ));
        assert_eq!(b.used_bytes(), 0);
        assert!(!LIVE.load(Ordering::SeqCst));
    }
    // Every material/scratch allocation failure erases and refunds.
    for fail in 0..7 {
        let b = Budget::new(cost.reservation_bytes).unwrap();
        b.fail_after.store(fail, Ordering::SeqCst);
        assert!(matches!(
            party::issue_local(s, context(), &[2], &[3], &[4], &b),
            Err(Error::Allocation)
        ));
        assert_eq!(b.used_bytes(), 0);
        assert!(!LIVE.load(Ordering::SeqCst));
    }
    // Output/frame allocation failures consume only this party. Coordinator
    // cancels sibling explicitly; no hidden coupled owner exists.
    for fail in 0..2 {
        let b = Budget::new(cost.reservation_bytes).unwrap();
        let [mut left, mut right] = party::issue_local(s, context(), &[2], &[3], &[4], &b).unwrap();
        b.fail_after.store(fail, Ordering::SeqCst);
        assert!(matches!(
            left.begin(&[7], &[10], &mut || false),
            Err(Error::Allocation)
        ));
        assert!(matches!(
            left.begin(&[7], &[10], &mut || false),
            Err(Error::Consumed)
        ));
        right.cancel();
        assert_eq!(b.used_bytes(), 0);
        assert!(!LIVE.load(Ordering::SeqCst));
    }
    let b = Budget::new(cost.reservation_bytes).unwrap();
    let [mut left, mut right] = party::issue_local(s, context(), &[2], &[3], &[4], &b).unwrap();
    let lf = left.begin(&[7], &[10], &mut || false).unwrap();
    let rf = right.begin(&[7], &[10], &mut || false).unwrap();
    let lo = left.finish(rf.as_bytes()).unwrap();
    let ro = right.finish(lf.as_bytes()).unwrap();
    assert_eq!(lo.values(), &[39]); // (7-2)*(10-3)+4
    assert_eq!(ro.values(), lo.values());
    drop(left);
    drop(right);
    drop(ro);
    drop(rf);
    assert!(matches!(
        Dealer::issue(context(), s, &[2], &[3], &[4], &budget()),
        Err(Error::Resource)
    ));
    drop(lo);
    // A retained frame alone still prevents old AND new issuance.
    assert!(b.used_bytes() > 0);
    assert!(matches!(
        party::issue_local(s, context(), &[2], &[3], &[4], &budget()),
        Err(Error::Resource)
    ));
    drop(lf);
    assert_eq!(b.used_bytes(), 0);
    let op = Dealer::issue(context(), s, &[2], &[3], &[4], &b).unwrap();
    assert!(matches!(
        party::issue_local(s, context(), &[2], &[3], &[4], &budget()),
        Err(Error::Resource)
    ));
    drop(op);
}

#[test]
fn local_party_parity_canonical_frames_and_cancellation() {
    let _guard = TEST.lock().unwrap();
    for bits in 1..=64 {
        for layout in [MatrixLayout::InputOutput, MatrixLayout::OutputInput] {
            let s = spec(bits, layout);
            let cost = party::resources(s).unwrap();
            let b = Budget::new(cost.reservation_bytes).unwrap();
            let rx = vec![1; cost.input_elements];
            let rw = vec![1; cost.weight_elements];
            let ry = vec![1; cost.output_elements];
            let x: Vec<_> = (0..rx.len()).map(|i| i as u64 & mask(bits)).collect();
            let w: Vec<_> = (0..rw.len()).map(|i| (i as u64 * 3) & mask(bits)).collect();
            let [mut left, mut right] =
                party::issue_local(s, context(), &rx, &rw, &ry, &b).unwrap();
            let lf = left
                .begin(&masked(&x, &rx, bits), &masked(&w, &rw, bits), &mut || {
                    false
                })
                .unwrap();
            let rf = right
                .begin(&masked(&x, &rx, bits), &masked(&w, &rw, bits), &mut || {
                    false
                })
                .unwrap();
            assert_eq!(lf.as_bytes().len(), cost.frame_bytes);
            let lo = left.finish(rf.as_bytes()).unwrap();
            let ro = right.finish(lf.as_bytes()).unwrap();
            assert_eq!(lo.values(), oracle(s, &x, &w, &ry));
            assert_eq!(ro.values(), lo.values());
            assert!(matches!(left.finish(rf.as_bytes()), Err(Error::Consumed)));
        }
    }
    let s = scalar(3);
    let b = budget();
    let [mut left, mut right] = party::issue_local(s, context(), &[1], &[2], &[3], &b).unwrap();
    assert!(matches!(
        left.begin(&[1], &[2], &mut || true),
        Err(Error::Consumed)
    ));
    // Sibling remains locally usable, until outer coordinator cancels it.
    let rf = right.begin(&[1], &[2], &mut || false).unwrap();
    right.cancel();
    drop(rf);
    assert_eq!(b.used_bytes(), 0);
    let [mut left, mut right] = party::issue_local(s, context(), &[1], &[2], &[3], &b).unwrap();
    let lf = left.begin(&[1], &[2], &mut || false).unwrap();
    let rf = right.begin(&[1], &[2], &mut || false).unwrap();
    let mut bad = rf.as_bytes().to_vec();
    bad[76] |= 8; // Noncanonical residue, not merely a binding mismatch.
    assert!(matches!(left.finish(&bad), Err(Error::Domain)));
    assert!(matches!(left.finish(rf.as_bytes()), Err(Error::Consumed)));
    right.cancel();
    drop(lf);
    drop(rf);
    assert_eq!(b.used_bytes(), 0);
}

#[test]
fn local_party_invalid_preflight_never_draws_or_allocates() {
    let _guard = TEST.lock().unwrap();
    let s = scalar(3);
    let b = budget();
    b.fail_after.store(0, Ordering::SeqCst);
    let mut draws = 0;
    let mut fill = |_: &mut [u8]| {
        draws += 1;
        Err(Error::Randomness)
    };
    let mut reject = |spec, ctx, masks, expected| {
        assert_eq!(
            party::issue_with(spec, ctx, masks, &b, &mut fill, &mut || false).err(),
            Some(expected)
        );
        assert_eq!(b.used_bytes(), 0);
        assert!(!LIVE.load(Ordering::SeqCst));
        assert_eq!(LIVE_HANDLES.load(Ordering::SeqCst), 0);
    };
    for (bad, expected) in [
        (Spec { bits: 0, ..s }, Error::Domain),
        (Spec { bits: 65, ..s }, Error::Domain),
        (Spec { rows: 0, ..s }, Error::Resource),
        (Spec { inner: 0, ..s }, Error::Resource),
        (Spec { columns: 0, ..s }, Error::Resource),
        (
            Spec {
                rows: usize::MAX,
                inner: 2,
                ..s
            },
            Error::Resource,
        ),
        (Spec { rows: 257, ..s }, Error::Resource),
    ] {
        reject(bad, context(), [&[1][..]; 3], expected);
    }
    for bad in [
        Context {
            weight_wire: context().input_wire,
            ..context()
        },
        Context {
            output_wire: context().input_wire,
            ..context()
        },
        Context {
            output_wire: context().weight_wire,
            ..context()
        },
    ] {
        reject(s, bad, [&[1][..]; 3], Error::Context);
    }
    for index in 0..3 {
        for bad in [&[][..], &[1, 2][..], &[8][..]] {
            let mut masks = [&[1][..]; 3];
            masks[index] = bad;
            reject(s, context(), masks, Error::Domain);
        }
    }
    // A valid preflight reaches entropy despite allocation failure injection.
    reject(s, context(), [&[1][..]; 3], Error::Randomness);
    assert_eq!(draws, 1);
    assert_eq!(b.fail_after.load(Ordering::SeqCst), 0);
}

#[test]
fn local_party_invalid_online_operands_consume_only_local_owner() {
    let _guard = TEST.lock().unwrap();
    let s = scalar(3);
    let cost = party::resources(s).unwrap();
    let b = Budget::new(cost.reservation_bytes).unwrap();
    for role in 0..2 {
        for index in 0..2 {
            for bad in [&[][..], &[1, 2][..], &[8][..]] {
                let mut parties = party::issue_local(s, context(), &[1], &[2], &[3], &b).unwrap();
                let [left, right] = &mut parties;
                let (local, sibling) = if role == 0 {
                    (left, right)
                } else {
                    (right, left)
                };
                let mut operands = [&[1][..], &[2][..]];
                operands[index] = bad;
                // Domain validation must precede the first output allocation.
                b.fail_after.store(0, Ordering::SeqCst);
                assert!(matches!(
                    local.begin(operands[0], operands[1], &mut || false),
                    Err(Error::Domain)
                ));
                assert!(matches!(
                    local.begin(&[1], &[2], &mut || false),
                    Err(Error::Consumed)
                ));
                assert_eq!(b.used_bytes(), cost.party_reservation_bytes);
                assert_eq!(LIVE_HANDLES.load(Ordering::SeqCst), 1);
                b.fail_after.store(-1, Ordering::SeqCst);
                let frame = sibling.begin(&[1], &[2], &mut || false).unwrap();
                sibling.cancel();
                assert!(b.used_bytes() > 0);
                assert!(LIVE.load(Ordering::SeqCst));
                assert_eq!(LIVE_HANDLES.load(Ordering::SeqCst), 1);
                drop(frame);
                drop(parties);
                assert_eq!(b.used_bytes(), 0);
                assert!(!LIVE.load(Ordering::SeqCst));
                assert_eq!(LIVE_HANDLES.load(Ordering::SeqCst), 0);
            }
        }
    }
}

#[test]
fn local_party_identical_nonce_reissuance_changes_binding_and_rejects_stale_frame() {
    let _guard = TEST.lock().unwrap();
    let s = scalar(3);
    let b = Budget::new(party::resources(s).unwrap().reservation_bytes).unwrap();
    let mut draws = 0;
    let mut fill = |bytes: &mut [u8]| {
        draws += 1;
        bytes.fill(0x5a);
        Ok(())
    };
    // Copied bytes are a caller-owned replay fixture, not a retained Frame lease.
    let stale = {
        let [mut left, mut right] =
            party::issue_with(s, context(), [&[1], &[2], &[3]], &b, &mut fill, &mut || {
                false
            })
            .unwrap();
        let frame = right.begin(&[1], &[2], &mut || false).unwrap();
        let bytes = frame.as_bytes().to_vec();
        left.cancel();
        right.cancel();
        drop(frame);
        bytes
    };
    assert_eq!(b.used_bytes(), 0);
    assert_eq!(LIVE_HANDLES.load(Ordering::SeqCst), 0);
    let [mut left, mut right] =
        party::issue_with(s, context(), [&[1], &[2], &[3]], &b, &mut fill, &mut || {
            false
        })
        .unwrap();
    let lf = left.begin(&[1], &[2], &mut || false).unwrap();
    let rf = right.begin(&[1], &[2], &mut || false).unwrap();
    assert_eq!(draws, 8); // Two identical nonces and six identical split draws.
    assert_ne!(&stale[8..40], &rf.as_bytes()[8..40]);
    assert_eq!(&stale[..8], &rf.as_bytes()[..8]);
    assert_eq!(&stale[40..], &rf.as_bytes()[40..]);
    assert!(matches!(left.finish(&stale), Err(Error::Context)));
    assert!(matches!(left.finish(rf.as_bytes()), Err(Error::Consumed)));
    let output = right.finish(lf.as_bytes()).unwrap();
    assert_eq!(output.values(), &[3]);
    drop((left, right, lf, rf));
    assert!(b.used_bytes() > 0);
    assert_eq!(LIVE_HANDLES.load(Ordering::SeqCst), 1);
    drop(output);
    assert_eq!(b.used_bytes(), 0);
    assert!(!LIVE.load(Ordering::SeqCst));
    assert_eq!(LIVE_HANDLES.load(Ordering::SeqCst), 0);
}

#[test]
fn local_party_each_frame_field_and_payload_word_failure_consumes_only_local_owner() {
    let _guard = TEST.lock().unwrap();
    let s = spec(3, MatrixLayout::InputOutput);
    let cost = party::resources(s).unwrap();
    let b = Budget::new(cost.reservation_bytes).unwrap();
    let rx = vec![1; cost.input_elements];
    let rw = vec![2; cost.weight_elements];
    let ry = vec![3; cost.output_elements];
    // Mutate every header byte separately: magic [0,8), source/issuance binding
    // [8,40), operand digest [40,72), sender 72, phase 73, count [74,76).
    // Then make each output word noncanonical while all other words stay valid.
    const HEADER: usize = 76;
    for role in 0..2 {
        for mutation in 0..HEADER + cost.output_elements {
            let mut parties = party::issue_local(s, context(), &rx, &rw, &ry, &b).unwrap();
            let [left, right] = &mut parties;
            let (local, sibling) = if role == 0 {
                (left, right)
            } else {
                (right, left)
            };
            let lf = local.begin(&rx, &rw, &mut || false).unwrap();
            let rf = sibling.begin(&rx, &rw, &mut || false).unwrap();
            let mut bad = rf.as_bytes().to_vec();
            let expected = if mutation < HEADER {
                bad[mutation] ^= 1;
                Error::Context
            } else {
                bad[HEADER + 8 * (mutation - HEADER)] |= 8;
                Error::Domain
            };
            b.fail_after.store(0, Ordering::SeqCst);
            assert_eq!(
                local.finish(&bad).err(),
                Some(expected),
                "role {role}, mutation {mutation}"
            );
            assert!(matches!(local.finish(rf.as_bytes()), Err(Error::Consumed)));
            // Sibling can still finish with the original valid frame, without
            // allocating; rejected local frame must not retire sibling material.
            let output = sibling.finish(lf.as_bytes()).unwrap();
            assert_eq!(output.values(), ry.as_slice());
            drop(parties);
            drop(output);
            assert!(b.used_bytes() > 0);
            assert_eq!(LIVE_HANDLES.load(Ordering::SeqCst), 2);
            drop((lf, rf));
            assert_eq!(b.used_bytes(), 0);
            assert!(!LIVE.load(Ordering::SeqCst));
            assert_eq!(LIVE_HANDLES.load(Ordering::SeqCst), 0);
            b.fail_after.store(-1, Ordering::SeqCst);
        }
    }
}

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        input_wire: [4; 32],
        weight_wire: [5; 32],
        output_wire: [6; 32],
    }
}
fn spec(bits: u8, layout: MatrixLayout) -> Spec {
    Spec {
        bits,
        rows: 2,
        inner: 3,
        columns: 4,
        layout,
    }
}
fn scalar(bits: u8) -> Spec {
    Spec {
        bits,
        rows: 1,
        inner: 1,
        columns: 1,
        layout: MatrixLayout::InputOutput,
    }
}
fn budget() -> Budget {
    Budget::new(MAX_ALLOCATION_BYTES).unwrap()
}
fn issue<'b>(s: Spec, rx: &[u64], rw: &[u64], ry: &[u64], b: &'b Budget) -> Operation<'b> {
    let mut counter = 0u64;
    Dealer::issue_with(
        context(),
        s,
        [rx, rw, ry],
        b,
        &mut |bytes| {
            counter = counter.wrapping_add(0x9e3779b97f4a7c15);
            for chunk in bytes.chunks_mut(8) {
                chunk.copy_from_slice(&counter.to_le_bytes()[..chunk.len()]);
            }
            Ok(())
        },
        &mut || false,
    )
    .unwrap()
}
fn masked(values: &[u64], masks: &[u64], bits: u8) -> Vec<u64> {
    values
        .iter()
        .zip(masks)
        .map(|(x, r)| x.wrapping_add(*r) & mask(bits))
        .collect()
}
// Independent row-major u128 oracle: no native geometry/traversal reuse.
fn oracle(s: Spec, x: &[u64], w: &[u64], ry: &[u64]) -> Vec<u64> {
    let mut output = Vec::new();
    for row in 0..s.rows {
        for column in 0..s.columns {
            let mut value = ry[row * s.columns + column] as u128;
            for k in 0..s.inner {
                let index = match s.layout {
                    MatrixLayout::InputOutput => k * s.columns + column,
                    MatrixLayout::OutputInput => column * s.inner + k,
                };
                value = value.wrapping_add((x[row * s.inner + k] as u128) * (w[index] as u128));
            }
            output.push(value as u64 & mask(s.bits));
        }
    }
    output
}

#[test]
fn rectangular_complete_ring_oracle_all_widths_layouts_signed_wrap() {
    let _guard = TEST.lock().unwrap();
    for bits in [1, 37, 40, 48, 50, 51, 64] {
        for layout in [MatrixLayout::InputOutput, MatrixLayout::OutputInput] {
            let s = spec(bits, layout);
            let r = resources(s).unwrap();
            let x: Vec<_> = [u64::MAX, 1 << 63, 0, 1, 7, u64::MAX - 1]
                .map(|x| x & mask(bits))
                .into();
            let row_weights: Vec<_> = (0..12)
                .map(|i| {
                    (if i % 2 == 0 {
                        (i as u64 + 1).wrapping_neg()
                    } else {
                        1u64 << (i + 1)
                    }) & mask(bits)
                })
                .collect();
            let w: Vec<_> = (0..12)
                .map(|i| match layout {
                    MatrixLayout::InputOutput => row_weights[i],
                    MatrixLayout::OutputInput => row_weights[(i % 3) * 4 + i / 3],
                })
                .collect();
            let rx: Vec<_> = (0..6)
                .map(|i| (u64::MAX - i as u64 * 13) & mask(bits))
                .collect();
            let rw: Vec<_> = (0..12).map(|i| (i as u64 * 97 + 3) & mask(bits)).collect();
            let ry: Vec<_> = (0..8).map(|i| (i as u64 * 19 + 7) & mask(bits)).collect();
            let b = budget();
            let op = issue(s, &rx, &rw, &ry, &b);
            let input = op
                .masked_inputs(context(), s, &masked(&x, &rx, bits), &masked(&w, &rw, bits))
                .unwrap();
            let [left, right] = op.parties().unwrap();
            let l = left.evaluate(&input, &mut || false).unwrap();
            let rr = right.evaluate(&input, &mut || false).unwrap();
            let out = reconstruct(rr, l).unwrap();
            assert_eq!(out.values(), oracle(s, &x, &w, &ry));
            assert_eq!(out.context(), context());
            assert_eq!(out.spec(), s);
            assert_eq!(r.party_key_bits, bits as usize * (6 + 12 + 8));
            assert_eq!(r.dealer_macs, 24);
            assert_eq!(r.party_macs, [48, 72]);
            assert_eq!(r.complete_macs, 144);
            assert_eq!(r.overlap_payload_bytes, 8 * (6 * 6 + 6 * 12 + 7 * 8));
            drop(input);
            assert_eq!(b.used_bytes(), r.reservation_bytes);
            drop(out);
            // Stack-owned control deliberately retains whole reservation.
            assert_eq!(b.used_bytes(), r.reservation_bytes);
            drop(op);
            assert_eq!(b.used_bytes(), 0);
        }
    }
}

#[test]
fn scalar_three_n_matches_legacy_adapter_and_exhaustive_tiny_mask_shares() {
    let _guard = TEST.lock().unwrap();
    // Exhaust all canonical masks, additive correlation splits, and operands.
    for bits in 1..=2 {
        let q = 1u64 << bits;
        for rx in 0..q {
            for rw in 0..q {
                for ry in 0..q {
                    for a in 0..q {
                        for b in 0..q {
                            for c in 0..q {
                                let left = [a, b, c];
                                let right = [
                                    product::negative_mask(rx).wrapping_sub(a) & mask(bits),
                                    product::negative_mask(rw).wrapping_sub(b) & mask(bits),
                                    product::correlation(rx, rw, ry).wrapping_sub(c) & mask(bits),
                                ];
                                for x in 0..q {
                                    for w in 0..q {
                                        let xhat = (x + rx) & mask(bits);
                                        let what = (w + rw) & mask(bits);
                                        assert_eq!(
                                            product::evaluate(0, xhat, what, &left).wrapping_add(
                                                product::evaluate(1, xhat, what, &right)
                                            ) & mask(bits),
                                            (x * w + ry) & mask(bits)
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for bits in [1, 37, 48, 50, 51, 64] {
        let b = budget();
        let s = scalar(bits);
        let op = issue(s, &[1], &[1], &[1], &b);
        assert_eq!(resources(s).unwrap().party_key_bits, 3 * bits as usize);
        assert_eq!(resources(s).unwrap().complete_macs, 6);
        let coefficients = op.materials.borrow();
        for party in 0..2 {
            let m = coefficients[party].as_ref().unwrap();
            let legacy = super::super::Multiply {
                coefficients: Zeroizing::new([m.a[0], m.b[0], m.c[0]]),
            };
            for (x, w) in [(0, 1), (1, 2), (u64::MAX, u64::MAX - 1)] {
                assert_eq!(
                    legacy.eval(party as u8, x, w),
                    product::evaluate(party as u8, x, w, &[m.a[0], m.b[0], m.c[0]])
                        & super::super::mask(super::super::N)
                );
            }
        }
        drop(coefficients);
        drop(op);
    }
}

#[test]
fn native_scalar_matches_existing_multiply_adapter_party_by_party() {
    let _guard = TEST.lock().unwrap();
    let b = budget();
    let s = scalar(super::super::N);
    let op = issue(s, &[13], &[29], &[47], &b);
    let x = mask(s.bits) - 17;
    let w = mask(s.bits) - 31;
    let expected = {
        let materials = op.materials.borrow();
        [0, 1].map(|party| {
            let m = materials[party].as_ref().unwrap();
            super::super::Multiply {
                coefficients: Zeroizing::new([m.a[0], m.b[0], m.c[0]]),
            }
            .eval(party as u8, x, w)
        })
    };
    let input = op.masked_inputs(context(), s, &[x], &[w]).unwrap();
    let [left, right] = op.parties().unwrap();
    let left = left.evaluate(&input, &mut || false).unwrap();
    let right = right.evaluate(&input, &mut || false).unwrap();
    assert_eq!(left.values.as_slice(), &[expected[0]]);
    assert_eq!(right.values.as_slice(), &[expected[1]]);
    let output = reconstruct(left, right).unwrap();
    assert_eq!(
        output.values(),
        &[x.wrapping_sub(13)
            .wrapping_mul(w.wrapping_sub(29))
            .wrapping_add(47)
            & mask(s.bits)]
    );
}

#[test]
fn tiny_native_scalar_all_masks_and_inputs() {
    let _guard = TEST.lock().unwrap();
    let b = budget();
    let s = scalar(1);
    for rx in 0..2 {
        for rw in 0..2 {
            for ry in 0..2 {
                for x in 0..2 {
                    for w in 0..2 {
                        let op = issue(s, &[rx], &[rw], &[ry], &b);
                        let input = op
                            .masked_inputs(context(), s, &[(x + rx) & 1], &[(w + rw) & 1])
                            .unwrap();
                        let [l, r] = op.parties().unwrap();
                        let output = reconstruct(
                            l.evaluate(&input, &mut || false).unwrap(),
                            r.evaluate(&input, &mut || false).unwrap(),
                        )
                        .unwrap();
                        assert_eq!(output.values(), &[(x * w + ry) & 1]);
                    }
                }
            }
        }
    }
    assert_eq!(b.used_bytes(), 0);
}

#[test]
fn preflight_caps_overflow_canonical_context_before_rng_or_allocation() {
    let _guard = TEST.lock().unwrap();
    let b = budget();
    b.fail_after.store(0, Ordering::SeqCst);
    let mut calls = 0;
    let mut fill = |_: &mut [u8]| {
        calls += 1;
        Err(Error::Randomness)
    };
    for s in [
        Spec {
            bits: 0,
            ..scalar(1)
        },
        Spec {
            bits: 65,
            ..scalar(1)
        },
        Spec {
            rows: usize::MAX,
            inner: 2,
            ..scalar(1)
        },
        Spec {
            rows: 257,
            ..scalar(1)
        },
        Spec {
            columns: 0,
            ..scalar(1)
        },
    ] {
        assert!(
            Dealer::issue_with(context(), s, [&[0], &[0], &[0]], &b, &mut fill, &mut || {
                false
            })
            .is_err()
        );
    }
    assert_eq!(
        Dealer::issue_with(
            context(),
            scalar(1),
            [&[2], &[0], &[0]],
            &b,
            &mut fill,
            &mut || false
        )
        .err(),
        Some(Error::Domain)
    );
    let bad = Context {
        output_wire: context().input_wire,
        ..context()
    };
    assert_eq!(
        Dealer::issue_with(
            bad,
            scalar(1),
            [&[0], &[0], &[0]],
            &b,
            &mut fill,
            &mut || { false }
        )
        .err(),
        Some(Error::Context)
    );
    let small = Budget::new(1).unwrap();
    assert_eq!(
        Dealer::issue_with(
            context(),
            scalar(1),
            [&[0], &[0], &[0]],
            &small,
            &mut fill,
            &mut || false
        )
        .err(),
        Some(Error::Resource)
    );
    assert_eq!(calls, 0);
    assert_eq!(b.used_bytes(), 0);
    for bits in 1..=64 {
        assert_eq!(
            resources(scalar(bits)).unwrap().party_key_bits,
            3 * bits as usize
        );
    }
    assert!(resources(Spec {
        rows: 16,
        inner: 16,
        columns: 16,
        ..scalar(64)
    })
    .is_ok());
}

#[test]
fn issuance_unique_even_with_identical_context_masks_and_nonce() {
    let _guard = TEST.lock().unwrap();
    let b = budget();
    let first = issue(scalar(1), &[1], &[1], &[1], &b);
    let first_binding = first.binding;
    {
        let materials = first.materials.borrow();
        for material in materials.iter().flatten() {
            assert_eq!(
                material.a.capacity() + material.b.capacity() + material.c.capacity(),
                3
            );
        }
    }
    drop(first);
    let second = issue(scalar(1), &[1], &[1], &[1], &b);
    assert_ne!(first_binding, second.binding);
}

#[test]
fn malformed_context_wire_layout_role_duplicate_and_cancel_burn_pair() {
    let _guard = TEST.lock().unwrap();
    let b = budget();
    let s = scalar(1);
    for failure in 0..8 {
        let op = issue(s, &[1], &[1], &[1], &b);
        let [mut l, r] = op.parties().unwrap();
        match failure {
            0 => {
                assert!(op
                    .masked_inputs(
                        Context {
                            session: [8; 32],
                            ..context()
                        },
                        s,
                        &[0],
                        &[0]
                    )
                    .is_err());
            }
            1 => {
                assert!(op
                    .masked_inputs(
                        Context {
                            output_wire: [8; 32],
                            ..context()
                        },
                        s,
                        &[0],
                        &[0]
                    )
                    .is_err());
            }
            2 => {
                assert!(op
                    .masked_inputs(
                        context(),
                        Spec {
                            layout: MatrixLayout::OutputInput,
                            ..s
                        },
                        &[0],
                        &[0]
                    )
                    .is_err());
            }
            3 => {
                assert!(op.masked_inputs(context(), s, &[2], &[0]).is_err());
            }
            4 => {
                assert!(op.parties().is_err());
            }
            5 => {
                let input = op.masked_inputs(context(), s, &[0], &[0]).unwrap();
                assert!(op.masked_inputs(context(), s, &[0], &[0]).is_err());
                drop(input);
            }
            6 => {
                let input = op.masked_inputs(context(), s, &[0], &[0]).unwrap();
                l.role = 2;
                assert_eq!(
                    l.evaluate(&input, &mut || false).err(),
                    Some(Error::Context)
                );
                assert!(r.evaluate(&input, &mut || false).is_err());
                assert!(op.materials.borrow().iter().all(Option::is_none));
                continue;
            }
            _ => {
                op.cancel();
            }
        }
        assert!(op.materials.borrow().iter().all(Option::is_none));
        drop(l);
        drop(r);
    }
    assert_eq!(b.used_bytes(), 0);
    let op = issue(s, &[1], &[1], &[1], &b);
    assert_eq!(
        Dealer::issue(context(), s, &[1], &[1], &[1], &b).err(),
        Some(Error::Resource)
    );
    let input = op.masked_inputs(context(), s, &[0], &[0]).unwrap();
    let [l, r] = op.parties().unwrap();
    l.cancel();
    assert!(r.evaluate(&input, &mut || false).is_err());
}

#[test]
fn every_binding_field_and_operand_validation_failure_retires_both_materials() {
    let _guard = TEST.lock().unwrap();
    let b = budget();
    let s = scalar(2);
    for context in [
        Context {
            plan: [9; 32],
            ..context()
        },
        Context {
            session: [9; 32],
            ..context()
        },
        Context {
            operation: [9; 32],
            ..context()
        },
        Context {
            input_wire: [9; 32],
            ..context()
        },
        Context {
            weight_wire: [9; 32],
            ..context()
        },
        Context {
            output_wire: [9; 32],
            ..context()
        },
    ] {
        let op = issue(s, &[1], &[1], &[1], &b);
        assert_eq!(
            op.masked_inputs(context, s, &[0], &[0]).err(),
            Some(Error::Context)
        );
        assert!(op.materials.borrow().iter().all(Option::is_none));
    }
    for bad_spec in [
        Spec { bits: 1, ..s },
        Spec { rows: 2, ..s },
        Spec { inner: 2, ..s },
        Spec { columns: 2, ..s },
    ] {
        let op = issue(s, &[1], &[1], &[1], &b);
        assert_eq!(
            op.masked_inputs(context(), bad_spec, &[0], &[0]).err(),
            Some(Error::Context)
        );
        assert!(op.materials.borrow().iter().all(Option::is_none));
    }
    for (x, w) in [
        (&[][..], &[0][..]),
        (&[0][..], &[][..]),
        (&[4][..], &[0][..]),
        (&[0][..], &[4][..]),
    ] {
        let op = issue(s, &[1], &[1], &[1], &b);
        assert_eq!(
            op.masked_inputs(context(), s, x, w).err(),
            Some(Error::Domain)
        );
        assert!(op.materials.borrow().iter().all(Option::is_none));
    }
    assert_eq!(b.used_bytes(), 0);
}

#[test]
fn random_failure_every_dealer_boundary_releases_and_never_returns_partial_pair() {
    let _guard = TEST.lock().unwrap();
    let b = budget();
    for fail in 0..4 {
        let mut draws = 0;
        let result = Dealer::issue_with(
            context(),
            scalar(1),
            [&[1], &[1], &[1]],
            &b,
            &mut |bytes| {
                if draws == fail {
                    return Err(Error::Randomness);
                }
                draws += 1;
                bytes.fill(0);
                Ok(())
            },
            &mut || false,
        );
        assert_eq!(result.err(), Some(Error::Randomness));
        assert_eq!(b.used_bytes(), 0);
        assert!(!LIVE.load(Ordering::SeqCst));
    }
}

#[test]
fn allocation_failure_every_issue_import_evaluate_reconstruct_boundary() {
    let _guard = TEST.lock().unwrap();
    let b = budget();
    for fail in 0..12 {
        b.fail_after.store(fail, Ordering::SeqCst);
        let result = (|| -> Result<(), Error> {
            let op = Dealer::issue(context(), scalar(1), &[1], &[1], &[1], &b)?;
            let input = op.masked_inputs(context(), scalar(1), &[0], &[0])?;
            let [l, r] = op.parties()?;
            let left = l.evaluate(&input, &mut || false)?;
            let right = r.evaluate(&input, &mut || false)?;
            let _out = reconstruct(left, right)?;
            Ok(())
        })();
        assert_eq!(result, Err(Error::Allocation), "allocation {fail}");
        assert_eq!(b.used_bytes(), 0);
        assert!(!LIVE.load(Ordering::SeqCst));
    }
    b.fail_after.store(-1, Ordering::SeqCst);
}

#[test]
fn cancellation_each_dealer_and_worker_check_erases_peer() {
    let _guard = TEST.lock().unwrap();
    let b = budget();
    // Scalar dealer: initial, coefficient, three split boundaries.
    for fail in 0..5 {
        let mut checks = 0;
        let result = Dealer::issue_with(
            context(),
            scalar(1),
            [&[1], &[1], &[1]],
            &b,
            &mut |bytes| {
                bytes.fill(0);
                Ok(())
            },
            &mut || {
                checks += 1;
                checks == fail + 1
            },
        );
        assert_eq!(result.err(), Some(Error::Consumed));
        assert_eq!(b.used_bytes(), 0);
    }
    for role in 0..2 {
        for fail in 0..(3 + role) {
            let op = issue(scalar(1), &[1], &[1], &[1], &b);
            let input = op.masked_inputs(context(), scalar(1), &[0], &[0]).unwrap();
            let [l, r] = op.parties().unwrap();
            let (worker, peer) = if role == 0 { (l, r) } else { (r, l) };
            let mut checks = 0;
            assert_eq!(
                worker
                    .evaluate(&input, &mut || {
                        checks += 1;
                        checks == fail + 1
                    })
                    .err(),
                Some(Error::Consumed)
            );
            assert!(op.materials.borrow().iter().all(Option::is_none));
            assert!(peer.evaluate(&input, &mut || false).is_err());
        }
    }
    assert_eq!(b.used_bytes(), 0);
}

#[test]
fn malformed_reconstruction_and_dropped_worker_retire_pair() {
    let _guard = TEST.lock().unwrap();
    let b = budget();
    for failure in 0..3 {
        let op = issue(scalar(1), &[1], &[1], &[1], &b);
        let input = op.masked_inputs(context(), scalar(1), &[0], &[0]).unwrap();
        let [l, r] = op.parties().unwrap();
        let left = l.evaluate(&input, &mut || false).unwrap();
        let mut right = r.evaluate(&input, &mut || false).unwrap();
        match failure {
            0 => right.role = left.role,
            1 => right.binding[0] ^= 1,
            _ => right.values[0] = 2,
        }
        assert!(reconstruct(left, right).is_err());
        assert!(op.retired.get());
    }
    let op = issue(scalar(1), &[1], &[1], &[1], &b);
    let input = op.masked_inputs(context(), scalar(1), &[0], &[0]).unwrap();
    let [l, r] = op.parties().unwrap();
    drop(l);
    assert!(r.evaluate(&input, &mut || false).is_err());
}
