use super::*;
use crate::shared_allocation::Lease;

fn scalar_signed_scan(bits: &[u8], table: &[u64], rotation: usize, role: u8) -> (u64, i64) {
    let mut sum = 0u64;
    let mut count = 0i64;
    for (index, &bit) in bits.iter().enumerate() {
        sum = sum.wrapping_add(table[(index + rotation) % table.len()].wrapping_mul(u64::from(bit)));
        count += i64::from(bit);
    }
    if role == 1 { (0u64.wrapping_sub(sum), -count) } else { (sum, count) }
}

#[test]
fn prepared_fold_scalar_source_first_frames_and_masked_outputs_both_boundaries() {
    use crate::compact_dpf::gpu127::tests::source_bits;
    for width in [8, 10] {
        let entries = 1usize << width;
        let points = [0u64, 127, 128, (entries - 1) as u64];
        let rin = points.map(|p| 0u64.wrapping_sub(p) & (entries - 1) as u64);
        let indices = [(entries - 1) as u64, 0, 127, 128];
        for boundary in [Boundary::Deferred, Boundary::Reconstructed] {
            for phase in [0u8, 1] {
                for kind in 0..3 {
                    let s = Spec { index_bits: width, output_bits: 64, lanes: 4, boundary };
                    let table: Vec<_> = (0..entries).map(|i| match kind {
                        0 => (i as u64).wrapping_mul(7919).wrapping_add((i as u64).pow(3)),
                        1 => if i == 127 { u64::MAX } else { 0 },
                        _ => u64::MAX,
                    }).collect();
                    let budget = Budget::new(resources(s).unwrap().combined_reservation_bytes).unwrap();
                    let mut random = rng(8731);
                    let (masks, [mut a, mut b]) = issue_with(s, context(), &table, &rin, &budget, |bytes| {
                        random(bytes)?;
                        // GPU issuance's second root control; applies to every
                        // 32-byte draw without changing draw count/order/length.
                        if bytes.len() == 32 { bytes[16] = (bytes[16] & !1) | phase; }
                        Ok(())
                    }).unwrap();
                    let mut expected_frames = Vec::new();
                    for party in [&a, &b] {
                        let mut expected = Vec::new();
                        for (lane, &rotation) in indices.iter().enumerate() {
                            let key = &party.state.as_ref().unwrap().lanes[lane].key;
                            let bits = source_bits(key);
                            let (sum, count) = scalar_signed_scan(&bits, &table, rotation as usize, party.role);
                            let before = gpu127::measured_aes_blocks();
                            let folded = bounded_fold(party, lane, rotation).unwrap();
                            assert_eq!(gpu127::measured_aes_blocks() - before, if width == 8 { 2 } else { 14 });
                            assert_eq!((*folded.sum, *folded.count), (sum, count));
                            let local = party.correction.masks[lane];
                            // Literal source modulo-four equations, not correction().
                            let c = if party.role == 0 {
                                ((count + 1).rem_euclid(4) / 2) as u64
                            } else {
                                (2 - (4 - count.rem_euclid(4)) / 2).rem_euclid(2) as u64
                            };
                            expected.extend_from_slice(&sum.wrapping_add(local[0]).to_le_bytes());
                            expected.extend_from_slice(&(c ^ local[1]).to_le_bytes());
                        }
                        expected_frames.push(expected);
                    }
                    let af = a.begin(&indices).unwrap();
                    let bf = b.begin(&indices).unwrap();
                    assert_eq!(&af.as_bytes()[HEADER..], &expected_frames[0]);
                    assert_eq!(&bf.as_bytes()[HEADER..], &expected_frames[1]);
                    let ao = a.advance(bf.as_bytes()).unwrap();
                    let bo = b.advance(af.as_bytes()).unwrap();
                    let values: Vec<u64> = match (ao, bo) {
                        (StepFor::Deferred(ao), StepFor::Deferred(bo)) => ao.values().iter().zip(bo.values())
                            .map(|(&x, &y)| x.wrapping_add(y)).collect(),
                        (StepFor::Frame(af2), StepFor::Frame(bf2)) => {
                            let ao = must_common(a.advance(bf2.as_bytes()).unwrap());
                            let bo = must_common(b.advance(af2.as_bytes()).unwrap());
                            assert_eq!(ao.values(), bo.values());
                            ao.values().to_vec()
                        }
                        _ => panic!("wrong boundary"),
                    };
                    for lane in 0..s.lanes {
                        let x = indices[lane].wrapping_sub(rin[lane]) & (entries - 1) as u64;
                        assert_eq!(values[lane], table[x as usize].wrapping_add(masks.values()[lane]));
                    }
                    drop((af, bf, a, b, masks));
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}

#[test]
fn prepared_fold_preflight_unwind_erasure_and_positive_unadmitted_layout() {
    for width in [8, 10] {
        let s = Spec { index_bits: width, output_bits: 64, lanes: 1, boundary: Boundary::Deferred };
        let entries = s.entries();
        let table: Vec<_> = (0..entries).map(|i| u64::MAX - i as u64).collect();
        let budget = Budget::new(resources(s).unwrap().combined_reservation_bytes).unwrap();
        let (masks, [mut a, mut b]) = issue_with(s, context(), &table, &[0], &budget, rng(71)).unwrap();
        let before = gpu127::measured_aes_blocks();
        for invalid in [7, 9, 11, 13, 63] {
            a.spec.index_bits = invalid;
            assert!(matches!(bounded_fold(&a, 0, 0), Err(Error::Domain)));
            a.spec = s;
            assert!(fold_layout_bytes::<&Budget>(invalid).is_err());
        }
        a.role = 2;
        assert!(matches!(bounded_fold(&a, 0, 0), Err(Error::Domain)));
        a.role = 0;
        let output_masks = Zeroizing::new(masks.values().to_vec());
        for (party, issued_role) in [(&mut a, 0u8), (&mut b, 1u8)] {
            let state = party.state.as_ref().unwrap();
            assert!(state.lanes[0].key.matches_party(issued_role));
            let sums = Zeroizing::new(state.sums.to_vec());
            let counts = Zeroizing::new(state.counts.to_vec());
            let correction_masks = Zeroizing::new(party.correction.masks.to_vec());
            let used = budget.used_bytes();
            let erased = ERASED_FOLDS.with(|n| n.get());
            party.role = issued_role ^ 1;
            // Existing first-visit trap witnesses no callbacks in ascending
            // traversal; existing AES counter must also remain unchanged.
            PANIC_FOLD_AT.with(|at| at.set(Some(0)));
            assert!(matches!(bounded_fold(party, 0, 0), Err(Error::Domain)));
            PANIC_FOLD_AT.with(|at| assert_eq!(at.replace(None), Some(0)));
            assert_eq!(gpu127::measured_aes_blocks(), before);
            assert_eq!(ERASED_FOLDS.with(|n| n.get()), erased);
            let state = party.state.as_ref().unwrap();
            assert_eq!(state.lanes.len(), 1);
            assert!(state.lanes[0].key.matches_party(issued_role));
            assert_eq!(&state.sums[..], &sums[..]);
            assert_eq!(&state.counts[..], &counts[..]);
            assert_eq!(&party.correction.masks[..], &correction_masks[..]);
            assert_eq!(masks.values(), &output_masks[..]);
            assert_eq!(budget.used_bytes(), used);
            party.role = issued_role;
        }
        assert!(matches!(bounded_fold(&a, 1, 0), Err(Error::Domain)));
        assert!(matches!(bounded_fold(&a, 0, entries as u64), Err(Error::Domain)));
        let last = a.state.as_mut().unwrap().table.pop().unwrap();
        assert!(matches!(bounded_fold(&a, 0, 0), Err(Error::Domain)));
        a.state.as_mut().unwrap().table.push(last);
        assert_eq!(gpu127::measured_aes_blocks(), before);
        a.spec.index_bits = if width == 8 { 10 } else { 8 };
        assert!(matches!(bounded_fold(&a, 0, 0), Err(Error::Domain)));
        a.spec = s;
        assert_eq!(gpu127::measured_aes_blocks(), before);
        let layout = fold_layout_bytes::<&Budget>(width).unwrap();
        assert!(layout > gpu127::visit_layout_bytes(width).unwrap());
        // Ordinary checked proposal retains ALL old scan bytes. Not live quote.
        let q = resources(s).unwrap();
        let proposed_party = q.party_reservation_bytes.checked_add(layout).unwrap();
        let proposed_combined = q.combined_reservation_bytes.checked_add(layout.checked_mul(2).unwrap()).unwrap();
        assert!(proposed_party > q.party_reservation_bytes);
        assert!(proposed_combined > q.combined_reservation_bytes);
        assert_eq!(q.full_expansion_scratch_bytes, entries);
        for at in [0, 127, 128, entries - 1] {
            let erased = ERASED_FOLDS.with(|n| n.get());
            PANIC_FOLD_AT.with(|n| n.set(Some(at)));
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bounded_fold(&a, 0, 0)
            }));
            assert!(outcome.is_err());
            assert_eq!(ERASED_FOLDS.with(|n| n.get()), erased + 1);
        }
        let erased = ERASED_FOLDS.with(|n| n.get());
        drop(bounded_fold(&b, 0, 0).unwrap());
        assert_eq!(ERASED_FOLDS.with(|n| n.get()), erased + 1);
        drop((a, b, masks));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn independent_first_frames_own_output_subtraction_party_mutation_controls() {
    for width in [8, 10] {
        for boundary in [Boundary::Deferred, Boundary::Reconstructed] {
            let s = Spec { index_bits: width, output_bits: 64, lanes: 2, boundary };
            let table: Vec<_> = (0..s.entries()).map(|i| (i as u64).wrapping_mul(7919)).collect();
            let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let mut random = rng(31);
            let mut coins = Zeroizing::new(Vec::new());
            let (masks, [mut a, mut b]) = issue_with(s, context(), &table, &[1, 127], &budget, |bytes| {
                random(bytes)?;
                if bytes.len() == 8 {
                    coins.push(u64::from_le_bytes(bytes.try_into().unwrap()));
                }
                Ok(())
            }).unwrap();
            // Original issuance: four correction draws/lane, then Select's
            // nonce and six full-ring draws/lane (rz, five LEFT shares). Retain
            // coins from THIS owner before begin consumes its material. No
            // duplicate owner or Select evaluation derives expected shares.
            assert_eq!(coins.len(), 10 * s.lanes);
            let mut words = Zeroizing::new(vec![[[0u64; 5]; 2]; s.lanes]);
            for lane in 0..s.lanes {
                let beta = a.correction.masks[lane][1] ^ b.correction.masks[lane][1];
                let rx = a.correction.masks[lane][0].wrapping_add(b.correction.masks[lane][0]);
                let draw = &coins[4 * s.lanes + 6 * lane..4 * s.lanes + 6 * (lane + 1)];
                let correlation = [beta, rx, beta.wrapping_mul(rx).wrapping_add(draw[0]),
                    beta.wrapping_mul(2), 0u64.wrapping_sub(beta.wrapping_mul(2).wrapping_mul(rx))];
                for j in 0..5 {
                    words[lane][0][j] = draw[j + 1];
                    words[lane][1][j] = correlation[j].wrapping_sub(draw[j + 1]);
                }
            }
            let af = a.begin(&[0, 128]).unwrap();
            let bf = b.begin(&[0, 128]).unwrap();
            let opened_r: Vec<_> = (0..2).map(|lane| {
                let offset = HEADER + lane * 16;
                u64::from_le_bytes(af.as_bytes()[offset..offset + 8].try_into().unwrap())
                    .wrapping_add(u64::from_le_bytes(bf.as_bytes()[offset..offset + 8].try_into().unwrap()))
            }).collect();
            let opened_c: Vec<_> = (0..2).map(|lane| {
                let offset = HEADER + lane * 16 + 8;
                u64::from_le_bytes(af.as_bytes()[offset..offset + 8].try_into().unwrap())
                    ^ u64::from_le_bytes(bf.as_bytes()[offset..offset + 8].try_into().unwrap())
            }).collect();
            let mut z = Zeroizing::new(vec![[0u64; 2]; s.lanes]);
            for lane in 0..s.lanes {
                for role in 0..2 {
                    let [a, big_b, c, d1, d2] = words[lane][role];
                    let bhat = opened_c[lane];
                    let xhat = opened_r[lane];
                    // Literal integer source Select equation, no helper calls.
                    z[lane][role] = c.wrapping_sub(a.wrapping_mul(xhat))
                        .wrapping_sub(big_b.wrapping_mul(bhat))
                        .wrapping_add(if bhat == 0 { d1.wrapping_mul(xhat).wrapping_add(d2) } else { 0 })
                        .wrapping_add(if role == 1 { bhat.wrapping_mul(xhat) } else { 0 });
                }
            }
            match boundary {
                Boundary::Deferred => {
                    let ao = must_deferred(a.advance(bf.as_bytes()).unwrap());
                    let bo = must_deferred(b.advance(af.as_bytes()).unwrap());
                    for lane in 0..2 {
                        let r = opened_r[lane];
                        assert_ne!(r, 0, "non-vacuous fixture {width}/{boundary:?}/{lane}");
                        assert_eq!(ao.values()[lane], z[lane][0].wrapping_mul(2).wrapping_sub(r));
                        assert_eq!(bo.values()[lane], z[lane][1].wrapping_mul(2));
                        // Mutation: subtract opened arithmetic from role ONE.
                        assert_ne!(ao.values()[lane], z[lane][0].wrapping_mul(2));
                        assert_ne!(bo.values()[lane], z[lane][1].wrapping_mul(2).wrapping_sub(r));
                    }
                }
                Boundary::Reconstructed => {
                    let af2 = must_frame(a.advance(bf.as_bytes()).unwrap());
                    let bf2 = must_frame(b.advance(af.as_bytes()).unwrap());
                    let ao = must_common(a.advance(bf2.as_bytes()).unwrap());
                    let bo = must_common(b.advance(af2.as_bytes()).unwrap());
                    for lane in 0..2 {
                        let common_z = z[lane][0].wrapping_add(z[lane][1]);
                        assert_ne!(opened_r[lane], 0, "non-vacuous fixture {width}/{boundary:?}/{lane}");
                        let expected = common_z.wrapping_mul(2).wrapping_sub(opened_r[lane]);
                        assert_eq!(ao.values()[lane], expected);
                        assert_eq!(bo.values()[lane], expected);
                        assert_ne!(bo.values()[lane], common_z.wrapping_mul(2));
                    }
                }
            }
            drop((af, bf, a, b, masks));
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn prepared_fold_each_width_party_actual_wrong_algebra_has_own_finite_witness() {
    use crate::compact_dpf::gpu127::tests::source_bits;
    for width in [8, 10] {
        for role in 0..2 {
            for kind in ["reverse-leaves", "skip", "wrong-sign", "wrong-rotation"] {
                let mut witness = None;
                // Finite deterministic PUBLIC TEST fixtures only. No adaptive
                // production traversal, mutation or extra-rotation harness.
                for salt in 0..64 {
                    let s = Spec { index_bits: width, output_bits: 64, lanes: 1, boundary: Boundary::Deferred };
                    let entries = s.entries();
                    let budget = Budget::new(resources(s).unwrap().combined_reservation_bytes).unwrap();
                    let table = vec![0; entries];
                    let mut random = rng(919 + salt);
                    let (masks, mut owners) = issue_with(s, context(), &table, &[0], &budget, |bytes| {
                        random(bytes)?;
                        if bytes.len() == 32 { bytes[16] = (bytes[16] & !1) | role as u8; }
                        Ok(())
                    }).unwrap();
                    let party = &mut owners[role];
                    let bits = source_bits(&party.state.as_ref().unwrap().lanes[0].key);
                    let reversed = |i: usize| (entries / 128 - 1 - i / 128) * 128 + i % 128;
                    let selected = match kind {
                        "reverse-leaves" => (0..entries).find(|&i| bits[i] != bits[reversed(i)]),
                        "wrong-rotation" => (0..entries).find(|&i| bits[i] != bits[(i + entries - 1) % entries]),
                        _ => bits.iter().position(|&b| b == 1),
                    };
                    if let Some(index) = selected {
                        // Explicit finite single-hot counterexample. Only this
                        // test fixture's public table changes; no live issuance
                        // or source declaration acquires authority from it.
                        let mut basis = vec![0; entries];
                        basis[index] = 1;
                        party.state.as_mut().unwrap().table.copy_from_slice(&basis);
                        let expected = scalar_signed_scan(&bits, &basis, 0, role as u8);
                        let correct = bounded_fold(party, 0, 0).unwrap();
                        assert_eq!((*correct.sum, *correct.count), expected);
                        let mut wrong = FoldVisitor {
                            table: &basis,
                            // ACTUAL wrong algebra: T[(i + rotation + 1) mod N].
                            rotation: usize::from(kind == "wrong-rotation"),
                            totals: FoldTotals { sum: Zeroizing::new(0), count: Zeroizing::new(0) },
                        };
                        let key = &party.state.as_ref().unwrap().lanes[0].key;
                        let mut actual_bits = Vec::new();
                        let before = gpu127::measured_aes_blocks();
                        key.preparation_visit(|i, bit| {
                            assert_eq!(i, actual_bits.len());
                            actual_bits.push(*bit.value());
                            if kind != "skip" || i != index {
                                wrong.visit(if kind == "reverse-leaves" { reversed(i) } else { i }, bit);
                            }
                        }).unwrap();
                        assert_eq!(actual_bits, bits); // Full independent scalar-AES vector.
                        assert_eq!(gpu127::measured_aes_blocks() - before, if width == 8 { 2 } else { 14 });
                        if role == 1 {
                            *wrong.totals.sum = 0u64.wrapping_sub(*wrong.totals.sum);
                            *wrong.totals.count = -*wrong.totals.count;
                        }
                        if kind == "wrong-sign" {
                            *wrong.totals.sum = 0u64.wrapping_sub(*wrong.totals.sum);
                            *wrong.totals.count = -*wrong.totals.count;
                        }
                        assert_ne!((*wrong.totals.sum, *wrong.totals.count), expected,
                            "independent witness {width}/{role}/{kind}/{salt}/{index}");
                        witness = Some((salt, index));
                    }
                    drop((owners, masks));
                    assert_eq!(budget.used_bytes(), 0);
                    if witness.is_some() { break; }
                }
                assert!(witness.is_some(), "missing finite counterexample {width}/{role}/{kind}");
            }
        }
    }
}

#[test]
fn private_build_aggregate_splits_and_both_dealer_guards_in_both_contexts() {
    fn check<C: AllocationContext>(s: Spec, budget: &Budget, lease: LeaseFor<C>) {
        let q = resources_for_context::<C>(s).unwrap();
        let child = correction::resources::<C>(CorrectionDescriptor { wire_spec: s }).unwrap();
        let baseline = budget.used_bytes() - q.combined_reservation_bytes;
        let table = vec![1; s.entries()];
        let (masks, parties) =
            issue_reserved_with(s, context(), &table, &[0, 1], lease, rng(81)).unwrap();
        assert_eq!(budget.used_bytes(), baseline + q.combined_reservation_bytes);
        assert_eq!(
            masks._scanner_dealer.bytes,
            q.dealer_reservation_bytes - child.dealer
        );
        assert_eq!(
            masks._correction_dealer.bytes,
            child.dealer - child.select.dealer_reservation_bytes
        );
        for party in &parties {
            assert_eq!(party.lease.bytes, q.party_reservation_bytes - child.party);
            assert_eq!(
                party.correction.lease.bytes,
                child.party - child.select.party_reservation_bytes
            );
            assert!(party.correction.scan_lease.is_none());
        }
        drop(parties);
        assert_eq!(budget.used_bytes(), baseline + q.dealer_reservation_bytes);
        assert_eq!(masks.values().len(), 2);
        drop(masks);
        assert_eq!(budget.used_bytes(), baseline);
    }
    for boundary in [Boundary::Deferred, Boundary::Reconstructed] {
        for index_bits in [8, 10] {
            let s = Spec {
                index_bits,
                output_bits: 50,
                lanes: 2,
                boundary,
            };
            let q = resources(s).unwrap();
            let budget = Budget::new(q.combined_reservation_bytes).unwrap();
            check(
                s,
                &budget,
                Lease::reserve(&&budget, q.combined_reservation_bytes).unwrap(),
            );
            let root = crate::shared_allocation::OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let q =
                resources_for_context::<crate::shared_allocation::OwnedBudgetContext>(s).unwrap();
            let mut work = root.charge_external(q.combined_reservation_bytes).unwrap();
            let lease = work
                .split_native(root.budget(), q.combined_reservation_bytes)
                .unwrap();
            check(s, root.budget(), lease);
            let before = root.budget().used_bytes();
            drop(work);
            assert!(root.budget().used_bytes() < before);
        }
    }
}

#[test]
fn private_build_nonexact_charge_rejects_before_allocation() {
    for boundary in [Boundary::Deferred, Boundary::Reconstructed] {
        let s = spec(boundary);
        let d = CorrectionDescriptor { wire_spec: s };
        let q = correction::resources::<&Budget>(d).unwrap();
        for bytes in [q.combined - 1, q.combined + 1] {
            let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let lease = Lease::reserve(&&budget, bytes).unwrap();
            budget.fail_after.store(0, Ordering::SeqCst);
            assert!(matches!(
                correction::prepare_reserved(d, lease),
                Err(Error::Resource)
            ));
            assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn private_build_rejected_lane_burns_and_cannot_retry_entropy() {
    let s = spec(Boundary::Deferred);
    let d = CorrectionDescriptor { wire_spec: s };
    let q = correction::resources::<&Budget>(d).unwrap();
    let budget = Budget::new(q.combined).unwrap();
    let mut build =
        correction::prepare_reserved(d, Lease::reserve(&&budget, q.combined).unwrap()).unwrap();
    // Negative ticket only: no test manufactures a successful issued-key seal.
    assert!(matches!(
        build.push_after_keypair(IssuedKeyLane { index: 1 }, &mut |_| panic!(
            "bad lane entropy"
        )),
        Err(Error::Context)
    ));
    assert!(matches!(
        build.push_after_keypair(IssuedKeyLane { index: 0 }, &mut |_| panic!(
            "replayed entropy"
        )),
        Err(Error::Consumed)
    ));
    assert_eq!(budget.used_bytes(), q.combined);
    drop(build);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn private_build_partial_issuance_failure_and_panic_each_original_draw() {
    for boundary in [Boundary::Deferred, Boundary::Reconstructed] {
        for index_bits in [8, 10] {
            let s = Spec {
                index_bits,
                output_bits: 50,
                lanes: 2,
                boundary,
            };
            let q = resources(s).unwrap();
            let table = vec![1; s.entries()];
            let budget = Budget::new(q.combined_reservation_bytes).unwrap();
            let mut draws = Vec::new();
            let mut fill = rng(19);
            drop(
                issue_with(s, context(), &table, &[0, 1], &budget, |b| {
                    draws.push(b.len());
                    fill(b)
                })
                .unwrap(),
            );
            for fail in 0..draws.len() {
                for panic in [false, true] {
                    let mut index = 0;
                    let mut fill = rng(19);
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        issue_with(s, context(), &table, &[0, 1], &budget, |b| {
                            assert_eq!(b.len(), draws[index]);
                            let here = index;
                            index += 1;
                            fill(b)?;
                            if here == fail {
                                if panic {
                                    panic!("written original draw")
                                } else {
                                    return Err(Error::Randomness);
                                }
                            }
                            Ok(())
                        })
                    }));
                    if panic {
                        assert!(outcome.is_err());
                    } else {
                        assert!(matches!(outcome, Ok(Err(Error::Randomness))));
                    }
                    assert_eq!(index, fail + 1, "no suffix draws after failure");
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}

#[test]
fn private_build_allocation_failures_retire_partial_arrays_and_original_guards() {
    for boundary in [Boundary::Deferred, Boundary::Reconstructed] {
        for index_bits in [8, 10] {
            let s = Spec {
                index_bits,
                output_bits: 50,
                lanes: 2,
                boundary,
            };
            let table = vec![1; s.entries()];
            let q = resources(s).unwrap();
            let budget = Budget::new(q.combined_reservation_bytes).unwrap();
            budget.fail_after.store(1000, Ordering::SeqCst);
            drop(issue_with(s, context(), &table, &[0, 1], &budget, rng(7)).unwrap());
            let factories = 1000 - budget.fail_after.load(Ordering::SeqCst);
            for fail in 0..factories {
                budget.fail_after.store(fail, Ordering::SeqCst);
                assert!(matches!(
                    issue_with(s, context(), &table, &[0, 1], &budget, rng(7)),
                    Err(Error::Allocation)
                ));
                assert_eq!(budget.used_bytes(), 0);
            }
        }
    }
}
use std::sync::atomic::Ordering;

#[test]
fn extracted_advance_unwind_burns_scanner_and_correction_with_original_leases_retained() {
    for (boundary, second_round) in [
        (Boundary::Deferred, false),
        (Boundary::Reconstructed, false),
        (Boundary::Reconstructed, true),
    ] {
        let spec = spec(boundary);
        let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
        let (masks, [mut a, mut b]) =
            issue_with(spec, context(), &[0; 256], &[0, 1], &budget, rng(53)).unwrap();
        let af = a.begin(&[3, 4]).unwrap();
        let bf = b.begin(&[3, 4]).unwrap();
        let second = if second_round {
            let ao = a.advance(bf.as_bytes()).unwrap();
            let bo = b.advance(af.as_bytes()).unwrap();
            match (ao, bo) {
                (StepFor::Frame(af), StepFor::Frame(bf)) => Some((af, bf)),
                _ => panic!("reconstructed transition must retain both scanners"),
            }
        } else {
            None
        };
        assert!(a.state.is_some() && b.state.is_some());
        let before = budget.used_bytes();
        let peer = second
            .as_ref()
            .map_or(bf.as_bytes(), |(_, bf)| bf.as_bytes());
        correction::PANIC_ADVANCE.with(|armed| armed.set(true));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.advance(peer)));
        assert!(result.is_err());
        assert!(a.state.is_none());
        assert!(a.correction.state.is_none());
        assert!(a.correction.masks.is_empty());
        assert!(a.correction.scan_lease.is_some());
        assert_eq!(budget.used_bytes(), before);
        assert!(matches!(a.advance(peer), Err(Error::Consumed)));
        assert!(matches!(a.begin(&[3, 4]), Err(Error::Consumed)));
        assert!(b.state.is_some()); // no new sibling coordinator or policy
        b.cancel();
        drop((a, b, masks, second));
        assert!(budget.used_bytes() > 0);
        drop((af, bf));
        assert_eq!(budget.used_bytes(), 0);
    }
}

// Captured from unchanged pre-extraction 46dfa0d6 (both issue/material/codec paths).
#[test]
fn frozen_pre_extraction_material_and_complete_transcript_vectors() {
    for bits in [8, 10] {
        for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
            let mut spec = spec(boundary);
            spec.index_bits = bits;
            spec.output_bits = 50;
            let table: Vec<_> = (0..spec.entries())
                .map(|i| (i as u64 * 7919) & mask(50))
                .collect();
            let cost = resources(spec).unwrap().combined_reservation_bytes;
            let mut hash = Sha256::new();
            for material_only in [true, false] {
                let budget = Budget::new(cost).unwrap();
                let mut random = rng(8731);
                let (masks, [mut a, mut b]) =
                    issue_with(spec, context(), &table, &[0, 1], &budget, |bytes| {
                        hash.update((bytes.len() as u64).to_le_bytes());
                        random(bytes)
                    })
                    .unwrap();
                for &word in masks.values() {
                    hash.update(word.to_le_bytes());
                }
                if material_only {
                    for party in [&mut a, &mut b] {
                        for lane in party.state.as_mut().unwrap().lanes.drain(..) {
                            hash.update(&*lane.key.into_material(4096).unwrap());
                        }
                    }
                } else {
                    let indices = [0, (spec.entries() - 1) as u64];
                    let af = a.begin(&indices).unwrap();
                    let bf = b.begin(&indices).unwrap();
                    hash.update(af.as_bytes());
                    hash.update(bf.as_bytes());
                    let (ao, bo) = match (
                        a.advance(bf.as_bytes()).unwrap(),
                        b.advance(af.as_bytes()).unwrap(),
                    ) {
                        (StepFor::Frame(af), StepFor::Frame(bf)) => {
                            hash.update(af.as_bytes());
                            hash.update(bf.as_bytes());
                            (
                                a.advance(bf.as_bytes()).unwrap(),
                                b.advance(af.as_bytes()).unwrap(),
                            )
                        }
                        outputs => outputs,
                    };
                    for output in [ao, bo] {
                        match output {
                            StepFor::Deferred(o) => {
                                for &v in o.values() {
                                    hash.update(v.to_le_bytes());
                                }
                            }
                            StepFor::Reconstructed(o) => {
                                for &v in o.values() {
                                    hash.update(v.to_le_bytes());
                                }
                            }
                            _ => panic!("unexpected nonterminal output"),
                        }
                    }
                }
            }
            let observed: [u8; 32] = hash.finalize().into();
            let expected = match (bits, boundary) {
                (8, Boundary::Reconstructed) => [
                    46, 187, 193, 85, 216, 88, 195, 194, 85, 184, 17, 125, 195, 102, 116, 252, 252,
                    179, 171, 23, 183, 37, 101, 224, 152, 159, 98, 201, 126, 172, 173, 153,
                ],
                (8, Boundary::Deferred) => [
                    102, 112, 156, 179, 237, 61, 161, 93, 13, 203, 248, 189, 141, 204, 209, 65, 97,
                    76, 21, 101, 17, 24, 41, 226, 148, 181, 159, 106, 253, 228, 203, 239,
                ],
                (10, Boundary::Reconstructed) => [
                    207, 117, 231, 167, 25, 38, 40, 231, 104, 84, 212, 2, 72, 164, 8, 137, 100, 28,
                    56, 214, 225, 232, 66, 173, 131, 100, 205, 169, 47, 124, 62, 129,
                ],
                (10, Boundary::Deferred) => [
                    102, 68, 230, 72, 216, 166, 76, 68, 231, 243, 32, 255, 140, 42, 23, 181, 17,
                    169, 206, 101, 57, 57, 6, 67, 184, 155, 195, 185, 35, 68, 149, 199,
                ],
                _ => unreachable!(),
            };
            assert_eq!(observed, expected);
        }
    }
}

#[test]
fn extracted_terminal_outputs_retire_scanner_in_both_modes_with_frames_retained() {
    for boundary in [Boundary::Deferred, Boundary::Reconstructed] {
        let spec = spec(boundary);
        let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
        let (masks, [mut a, mut b]) =
            issue_with(spec, context(), &[0; 256], &[0, 1], &budget, rng(24)).unwrap();
        let af = a.begin(&[3, 4]).unwrap();
        let bf = b.begin(&[3, 4]).unwrap();
        let ao = a.advance(bf.as_bytes()).unwrap();
        let bo = b.advance(af.as_bytes()).unwrap();
        let (aout, bout, second_frames) = match (ao, bo) {
            (StepFor::Frame(af2), StepFor::Frame(bf2)) => {
                assert!(a.state.is_some() && b.state.is_some());
                let aout = a.advance(bf2.as_bytes()).unwrap();
                let bout = b.advance(af2.as_bytes()).unwrap();
                (aout, bout, Some((af2, bf2)))
            }
            (StepFor::Deferred(ao), StepFor::Deferred(bo)) => {
                (StepFor::Deferred(ao), StepFor::Deferred(bo), None)
            }
            _ => panic!("unexpected boundary transition"),
        };
        assert!(a.state.is_none() && b.state.is_none());
        assert!(a.correction.state.is_none() && b.correction.state.is_none());
        assert!(matches!(a.advance(bf.as_bytes()), Err(Error::Consumed)));
        assert!(matches!(b.advance(af.as_bytes()), Err(Error::Consumed)));
        drop((a, b, masks, second_frames));
        assert!(budget.used_bytes() > 0);
        drop((aout, bout));
        assert!(budget.used_bytes() > 0); // actual retained first frames, not Party
        drop((af, bf));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn extracted_correction_begin_unwind_retires_scanner_and_burns_without_parent_refund() {
    for boundary in [Boundary::Deferred, Boundary::Reconstructed] {
        let spec = spec(boundary);
        let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
        let (masks, [mut a, mut b]) =
            issue_with(spec, context(), &[0; 256], &[0, 1], &budget, rng(29)).unwrap();
        let retained = b.begin(&[3, 4]).unwrap();
        let before = budget.used_bytes();
        correction::PANIC_BEGIN.with(|armed| armed.set(true));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.begin(&[3, 4])));
        assert!(result.is_err());
        assert!(a.state.is_none()); // actual local Execution unwound, not restored
        assert!(a.correction.state.is_none());
        assert!(a.correction.masks.is_empty());
        assert!(a.correction.select.as_ref().is_some()); // original lease retained
        assert_eq!(budget.used_bytes(), before);
        assert!(matches!(a.begin(&[3, 4]), Err(Error::Consumed)));
        assert!(matches!(
            a.advance(retained.as_bytes()),
            Err(Error::Consumed)
        ));
        drop((a, b, masks));
        assert!(budget.used_bytes() > 0);
        drop(retained);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn extracted_correction_rejects_foreign_counter_role_spec_and_invocation_and_burns() {
    for fault in 0..4 {
        let spec = spec(Boundary::Reconstructed);
        let cost = resources(spec).unwrap().combined_reservation_bytes;
        let budget = Budget::new(cost).unwrap();
        let foreign = Budget::new(cost).unwrap();
        let (masks, [mut a, b]) =
            issue_with(spec, context(), &[0; 256], &[0, 1], &budget, rng(44)).unwrap();
        match fault {
            0 => a.correction.lease = foreign.reserve(a.correction.lease.bytes).unwrap(),
            1 => a.correction.role ^= 1,
            2 => a.correction.spec.output_bits -= 1,
            3 => a.correction.binding[0] ^= 1,
            _ => unreachable!(),
        }
        assert!(matches!(a.begin(&[3, 4]), Err(Error::Context)));
        assert!(a.state.is_none());
        assert!(a.correction.state.is_none());
        assert!(a.correction.masks.is_empty());
        assert!(matches!(a.begin(&[3, 4]), Err(Error::Consumed)));
        drop((a, b, masks));
        assert_eq!(budget.used_bytes(), 0);
        assert_eq!(foreign.used_bytes(), 0);
    }
}

#[test]
fn extracted_scan_moves_sums_and_keeps_original_parent_material_until_retirement() {
    let spec = spec(Boundary::Reconstructed);
    let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
    let (masks, [mut a, b]) =
        issue_with(spec, context(), &[0; 256], &[0, 1], &budget, rng(19)).unwrap();
    let scanner = a.state.as_ref().unwrap();
    let sums = scanner.sums.as_ptr();
    let keys = scanner.lanes.as_ptr();
    let table = scanner.table.as_ptr();
    let frame = a.begin(&[255, 0]).unwrap();
    assert_eq!(
        a.correction.state.as_ref().unwrap().arithmetic.as_ptr(),
        sums
    );
    assert!(a.correction.masks.is_empty());
    assert_eq!(a.state.as_ref().unwrap().lanes.as_ptr(), keys);
    assert_eq!(a.state.as_ref().unwrap().table.as_ptr(), table);
    a.cancel();
    assert!(a.state.is_none());
    drop((a, b, masks));
    assert!(budget.used_bytes() > 0);
    drop(frame);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn extracted_first_frames_match_original_point_scan_and_count_correction_equations() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for bits in [8, 10] {
            let mut spec = spec(boundary);
            spec.index_bits = bits;
            let table: Vec<_> = (0..spec.entries())
                .map(|i| (i as u64 * 37) & mask(spec.output_bits))
                .collect();
            let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
            let (masks, [mut a, mut b]) =
                issue_with(spec, context(), &table, &[0, 1], &budget, rng(91)).unwrap();
            let indices = [0, (spec.entries() - 1) as u64];
            for party in [&mut a, &mut b] {
                let mut expected = Vec::new();
                for (lane, &index) in indices.iter().enumerate() {
                    let key = &party.state.as_ref().unwrap().lanes[lane].key;
                    let mut sum = 0u64;
                    let mut count = 0i64;
                    for i in 0..spec.entries() {
                        let bit = key.point(i as u64).unwrap();
                        sum = sum.wrapping_add(
                            table[(i + index as usize) & (spec.entries() - 1)]
                                .wrapping_mul(u64::from(bit)),
                        );
                        count += i64::from(bit);
                    }
                    if party.role == 1 {
                        sum = 0u64.wrapping_sub(sum);
                        count = -count;
                    }
                    let local = party.correction.masks[lane];
                    expected.extend_from_slice(
                        &((sum.wrapping_add(local[0])) & mask(spec.output_bits)).to_le_bytes(),
                    );
                    expected.extend_from_slice(
                        &(correction(party.role, count) ^ local[1]).to_le_bytes(),
                    );
                }
                let frame = party.begin(&indices).unwrap();
                assert_eq!(&frame.as_bytes()[HEADER..], expected);
                party.cancel();
            }
            drop((a, b, masks));
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn reserved_deferred_issuance_keeps_admission_draw_order_and_original_backing() {
    let spec = spec(Boundary::Deferred);
    let table: Vec<u64> = (0..256).map(|i| i & mask(spec.output_bits)).collect();
    let cost = resources(spec).unwrap().combined_reservation_bytes;
    let budget = Budget::new(cost + 97).unwrap();
    let guard = budget.reserve(97).unwrap();
    let reservation = budget.reserve(cost).unwrap();
    let mut fill = rng(7919);
    let (masks, [mut a, mut b]) =
        issue_reserved_with(spec, context(), &table, &[0, 1], reservation, |bytes| {
            assert_eq!(budget.used_bytes(), cost + 97);
            fill(bytes)
        })
        .unwrap();
    let reference_budget = Budget::new(cost).unwrap();
    let (reference_masks, [mut ra, mut rb]) = issue_with(
        spec,
        context(),
        &table,
        &[0, 1],
        &reference_budget,
        rng(7919),
    )
    .unwrap();
    assert_eq!(masks.values(), reference_masks.values());
    let binding = a.internal_binding();
    assert_eq!(binding, b.internal_binding());
    let af = a.begin(&[64, 0]).unwrap();
    let bf = b.begin(&[64, 0]).unwrap();
    let raf = ra.begin(&[64, 0]).unwrap();
    let rbf = rb.begin(&[64, 0]).unwrap();
    assert_eq!(af.as_bytes(), raf.as_bytes());
    assert_eq!(bf.as_bytes(), rbf.as_bytes());
    let output = must_deferred(a.advance(bf.as_bytes()).unwrap());
    let reference = must_deferred(ra.advance(rbf.as_bytes()).unwrap());
    assert_eq!(output.values(), reference.values());
    let pointer = output.values().as_ptr();
    let used = budget.used_bytes();
    let moved = output;
    assert_eq!(
        (moved.role(), *moved.binding(), moved.bits(), moved.lanes()),
        (0, binding, spec.output_bits, spec.lanes)
    );
    assert_eq!(pointer, moved.values().as_ptr());
    assert_eq!(budget.used_bytes(), used);
    drop(moved);
    drop((af, bf, masks, a, b, guard));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn reserved_lookup_rejects_nonexact_lease_before_entropy() {
    let spec = spec(Boundary::Deferred);
    let table = [0; 256];
    let cost = resources(spec).unwrap().combined_reservation_bytes;
    for bytes in [cost - 1, cost + 1] {
        let budget = Budget::new(cost + 1).unwrap();
        let result = issue_reserved_with(
            spec,
            context(),
            &table,
            &[0, 1],
            budget.reserve(bytes).unwrap(),
            |_| panic!("nonexact admission must not draw randomness"),
        );
        assert!(matches!(result, Err(Error::Resource)));
        assert_eq!(budget.used_bytes(), 0);
    }
}

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        input_wire: [4; 32],
        output_wire: [5; 32],
        arithmetic_wire: [6; 32],
        correction_wire: [7; 32],
        select_wire: [8; 32],
        numeric_contract: [9; 32],
        first_tensor_index: 17,
    }
}
fn spec(boundary: Boundary) -> Spec {
    Spec {
        index_bits: 8,
        output_bits: 3,
        lanes: 2,
        boundary,
    }
}
fn rng(mut state: u64) -> impl FnMut(&mut [u8]) -> Result<(), Error> {
    move |bytes| {
        for byte in bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        Ok(())
    }
}
fn must_frame(step: Step<'_>) -> Frame<'_> {
    match step {
        Step::Frame(frame) => frame,
        _ => panic!("expected second exchange"),
    }
}
fn must_common(step: Step<'_>) -> MaskedOutput<'_> {
    match step {
        Step::Reconstructed(output) => output,
        _ => panic!("expected common masked output"),
    }
}
fn must_deferred(step: Step<'_>) -> DeferredOutput<'_> {
    match step {
        Step::Deferred(output) => output,
        _ => panic!("expected own additive share"),
    }
}

fn exercise(
    spec: Spec,
    table: &[u64],
    rin: &[u64],
    indices: &[u64],
    budget: &Budget,
    fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(), Error> {
    let (masks, [mut a, mut b]) = issue_with(spec, context(), table, rin, budget, fill)?;
    let af = a.begin(indices)?;
    let bf = b.begin(indices)?;
    assert_eq!(af.as_bytes().len(), 80 + 16 * spec.lanes);
    let ao = a.advance(bf.as_bytes())?;
    let bo = b.advance(af.as_bytes())?;
    match spec.boundary {
        Boundary::Reconstructed => {
            let af2 = must_frame(ao);
            let bf2 = must_frame(bo);
            assert_eq!(af2.as_bytes().len(), 76 + 8 * spec.lanes);
            let ao = must_common(a.advance(bf2.as_bytes())?);
            let bo = must_common(b.advance(af2.as_bytes())?);
            assert_eq!(ao.values(), bo.values());
            assert_eq!(ao.binding(), bo.binding());
            assert_eq!(ao.bits(), spec.output_bits);
            assert_eq!(ao.lanes(), spec.lanes);
            for i in 0..spec.lanes {
                let clear = indices[i].wrapping_sub(rin[i]) & mask(spec.index_bits);
                assert_eq!(
                    ao.values()[i].wrapping_sub(masks.values()[i]) & mask(spec.output_bits),
                    table[clear as usize]
                );
            }
        }
        Boundary::Deferred => {
            let ao = must_deferred(ao);
            let bo = must_deferred(bo);
            assert_eq!((ao.role(), bo.role()), (0, 1));
            assert_eq!(ao.binding(), bo.binding());
            assert_eq!(ao.bits(), spec.output_bits);
            assert_eq!(ao.lanes(), spec.lanes);
            for i in 0..spec.lanes {
                let clear = indices[i].wrapping_sub(rin[i]) & mask(spec.index_bits);
                assert_eq!(
                    ao.values()[i]
                        .wrapping_add(bo.values()[i])
                        .wrapping_sub(masks.values()[i])
                        & mask(spec.output_bits),
                    table[clear as usize]
                );
            }
        }
    }
    assert!(matches!(a.advance(bf.as_bytes()), Err(Error::Consumed)));
    assert!(matches!(b.begin(indices), Err(Error::Consumed)));
    Ok(())
}

#[test]
fn full_signed_counts_not_unit_count_shortcut() {
    for n in [256i64, 1024] {
        for c0 in 0..=n {
            for sigma in [-1, 1] {
                let c1 = sigma - c0;
                if (-n..=0).contains(&c1) {
                    assert_eq!(
                        correction(0, c0) ^ correction(1, c1),
                        ((sigma + 1) / 2) as u64
                    );
                }
            }
        }
    }
}

#[test]
fn numeric_endpoints_tables_and_both_boundaries() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for n in [8, 10] {
            let endpoints = [0, 1, 63, 64, 127, 128, 255, 256, 511, 512, 1023];
            let points: Vec<u64> = endpoints.into_iter().filter(|&i| i < 1 << n).collect();
            for m in [3, 64] {
                let spec = Spec {
                    index_bits: n,
                    output_bits: m,
                    lanes: points.len(),
                    boundary,
                };
                let rin: Vec<u64> = points
                    .iter()
                    .map(|&p| 0u64.wrapping_sub(p) & mask(n))
                    .collect();
                // Opposite endpoints exercise wraparound and nontrivial shifts.
                let indices: Vec<u64> = points.iter().rev().copied().collect();
                for kind in 0..5 {
                    let table: Vec<u64> = (0..1usize << n)
                        .map(|i| match kind {
                            0 => 0,
                            1 => 1,
                            2 => i as u64 & mask(m),
                            3 => {
                                if i & 1 == 0 {
                                    mask(m)
                                } else {
                                    0
                                }
                            }
                            _ => mask(m),
                        })
                        .collect();
                    let budget =
                        Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
                    exercise(spec, &table, &rin, &indices, &budget, rng(7919 + kind)).unwrap();
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}

#[test]
fn both_target_signs_and_complete_ring_mask_endpoints() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for n in [8, 10] {
            for m in [3, 64] {
                let spec = Spec {
                    index_bits: n,
                    output_bits: m,
                    ..spec(boundary)
                };
                let entries = 1usize << n;
                let rin = [0, (entries / 2 + 1) as u64];
                let indices = [(entries - 1) as u64, (entries / 2) as u64];
                let table: Vec<u64> = (0..entries)
                    .map(|i| (i as u64).wrapping_mul(17) & mask(m))
                    .collect();
                let mut signs = [false; 2];
                for seed in 1..=32 {
                    let budget =
                        Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
                    {
                        let (_, [a, b]) =
                            issue_with(spec, context(), &table, &rin, &budget, rng(seed)).unwrap();
                        for (i, alpha) in [0, (entries / 2 - 1) as u64].into_iter().enumerate() {
                            let ba = a.state.as_ref().unwrap().lanes[i].key.point(alpha).unwrap();
                            let bb = b.state.as_ref().unwrap().lanes[i].key.point(alpha).unwrap();
                            assert_ne!(ba, bb);
                            signs[ba as usize] = true;
                        }
                    }
                    exercise(spec, &table, &rin, &indices, &budget, rng(seed)).unwrap();
                    assert_eq!(budget.used_bytes(), 0);
                }
                assert_eq!(signs, [true, true]);
                for value in [0, 1, 1u64 << (m - 1), mask(m)] {
                    // Cover every table output in bounded batches, with alternating
                    // endpoint input masks. No geometry exceeds MAX_LANES.
                    for first in (0..entries).step_by(MAX_LANES) {
                        let spec = Spec {
                            lanes: MAX_LANES,
                            ..spec
                        };
                        let rin: Vec<u64> = (0..spec.lanes)
                            .map(|i| if i & 1 == 0 { 0 } else { (entries - 1) as u64 })
                            .collect();
                        let indices: Vec<u64> = rin
                            .iter()
                            .enumerate()
                            .map(|(i, &r)| (first as u64 + i as u64 + r) & ((entries - 1) as u64))
                            .collect();
                        let budget =
                            Budget::new(resources(spec).unwrap().combined_reservation_bytes)
                                .unwrap();
                        // Every full word draw is this endpoint (Boolean draws use low
                        // bit); GPU seeds use its repeated LE representation as well.
                        exercise(spec, &table, &rin, &indices, &budget, |bytes| {
                            for (i, byte) in bytes.iter_mut().enumerate() {
                                *byte = value.to_le_bytes()[i % 8];
                            }
                            Ok(())
                        })
                        .unwrap();
                        assert_eq!(budget.used_bytes(), 0);
                    }
                }
            }
        }
    }
}

#[test]
fn independently_inventoried_padded_owner_reservations() {
    // These are physical owners and protocol allowances, not compact-key sizes
    // or calls to lookup's frame/output/checked-arithmetic accounting helpers.
    let allocator_allowance = 64usize;
    let stack_allowance = 8 * 1024;
    let frame_owner = size_of::<Frame<'static>>();
    let output_owner = size_of::<MaskedOutput<'static>>().max(size_of::<DeferredOutput<'static>>());
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for n in [8, 10] {
            let entries = 1usize << n;
            let compact_key_bytes = 18 + (129usize * usize::from(n - 7) + 255).div_ceil(8);
            assert!(size_of::<GpuKey>() > compact_key_bytes);
            assert_eq!(size_of::<LookupLane>(), size_of::<GpuKey>());
            assert!(size_of::<LookupLane>() > compact_key_bytes);
            for m in 3..=64 {
                for lanes in 1..=MAX_LANES {
                    let spec = Spec {
                        index_bits: n,
                        output_bits: m,
                        lanes,
                        boundary,
                    };
                    let first_payload = 80 + 16 * lanes;
                    let second_payload = if boundary == Boundary::Reconstructed {
                        76 + 8 * lanes
                    } else {
                        0
                    };
                    // Literal independent decomposition. Scan transfer T appears
                    // ONCE, and the child contains no GPU/table/scanner owner.
                    let transfer =
                        size_of::<ScanResultFor<&Budget>>() + 16 * lanes + 2 * allocator_allowance;
                    let scanner = size_of::<Party<'static>>() - size_of::<CorrectionFor<&Budget>>()
                        + size_of::<LookupLane>() * lanes
                        + 8 * entries
                        + 2 * allocator_allowance
                        + 8 * lanes
                        + entries
                        + size_of::<Zeroizing<Vec<u8>>>()
                        + allocator_allowance
                        + size_of::<ScannerControls<'_, &Budget>>()
                        + stack_allowance;
                    let correction = size_of::<CorrectionFor<&Budget>>()
                        + 24 * lanes
                        + 2 * allocator_allowance
                        + size_of::<correction::Controls<'_, &Budget>>()
                        + first_payload
                        + second_payload
                        + first_payload
                        + frame_owner
                        + allocator_allowance
                        + 8 * lanes
                        + output_owner
                        + allocator_allowance
                        + stack_allowance
                        + if second_payload == 0 { 0 } else { frame_owner };
                    let correction_dealer = size_of::<correction::CorrectionMasksFor<&Budget>>()
                        + 24 * lanes
                        + 3 * allocator_allowance
                        + stack_allowance
                        + size_of::<correction::IssuanceControls<'_, &Budget>>();
                    let scanner_dealer = 8 * lanes
                        + 8 * entries
                        + 2 * size_of::<GpuKey>()
                        + stack_allowance
                        + size_of::<ParentControls<'_, &Budget>>()
                        + size_of::<OutputMasks<'static>>()
                        - size_of::<correction::CorrectionMasksFor<&Budget>>();
                    // Select has its own independently tested inventory. Add it
                    // once only; lookup must not replace inline keys by codecs.
                    let nested =
                        selection::resources_for(selection::Spec { bits: m, lanes }, boundary)
                            .unwrap();
                    let actual = resources(spec).unwrap();
                    let expected_party =
                        scanner + transfer + correction + nested.party_reservation_bytes;
                    let expected_dealer =
                        scanner_dealer + correction_dealer + nested.dealer_reservation_bytes;
                    let expected_combined = expected_dealer + 2 * expected_party;
                    assert_eq!(actual.party_reservation_bytes, expected_party);
                    assert_eq!(actual.dealer_reservation_bytes, expected_dealer);
                    assert_eq!(actual.combined_reservation_bytes, expected_combined);
                    if [3, 64].contains(&m) && [1, MAX_LANES].contains(&lanes) {
                        let table = vec![0; entries];
                        let rin = vec![0; lanes];
                        let short = Budget::new(expected_combined - 1).unwrap();
                        short.fail_after.store(0, Ordering::SeqCst);
                        assert!(matches!(
                            issue_with(spec, context(), &table, &rin, &short, |_| panic!(
                                "entropy before admission"
                            )),
                            Err(Error::Resource)
                        ));
                        assert_eq!(short.used_bytes(), 0);
                        let exact = Budget::new(expected_combined).unwrap();
                        let owners =
                            issue_with(spec, context(), &table, &rin, &exact, rng(7)).unwrap();
                        assert_eq!(exact.used_bytes(), expected_combined);
                        drop(owners);
                        assert_eq!(exact.used_bytes(), 0);
                    }
                }
            }
        }
    }
}

#[test]
fn caller_table_and_input_mask_mutation_cannot_change_issued_lookup_ring64() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for n in [8, 10] {
            let spec = Spec {
                index_bits: n,
                output_bits: 64,
                lanes: 4,
                boundary,
            };
            let entries = 1usize << n;
            let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let mut table: Vec<u64> = (0..entries).map(|i| u64::MAX - i as u64).collect();
            let mut rin = [0, (entries - 1) as u64, (entries / 2) as u64, 1];
            let original_table = table.clone();
            let original_rin = rin;
            let indices = [0, 0, (entries / 2 - 1) as u64, (entries - 1) as u64];
            let (masks, [mut a, mut b]) =
                issue_with(spec, context(), &table, &rin, &budget, rng(37)).unwrap();
            let (_, [mut baseline_a, mut baseline_b]) = issue_with(
                spec,
                context(),
                &original_table,
                &original_rin,
                &budget,
                rng(37),
            )
            .unwrap();
            let original_binding = a.binding;
            // Caller backing remains mutable because issuance retains no slices.
            // Change EVERY entry and mask after issuance, before either begin.
            for value in &mut table {
                *value ^= 1u64 << 63;
            }
            for value in &mut rin {
                *value = (*value + 1) & (entries - 1) as u64;
            }
            assert_ne!(table, original_table);
            assert_ne!(rin, original_rin);
            let af = a.begin(&indices).unwrap();
            let bf = b.begin(&indices).unwrap();
            let baseline_af = baseline_a.begin(&indices).unwrap();
            let baseline_bf = baseline_b.begin(&indices).unwrap();
            assert_eq!(&af.as_bytes()[8..40], &original_binding);
            assert_eq!(af.as_bytes(), baseline_af.as_bytes());
            assert_eq!(bf.as_bytes(), baseline_bf.as_bytes());
            let ao = a.advance(bf.as_bytes()).unwrap();
            let bo = b.advance(af.as_bytes()).unwrap();
            let common: Vec<u64> = match boundary {
                Boundary::Reconstructed => {
                    let af2 = must_frame(ao);
                    let bf2 = must_frame(bo);
                    let ao = must_common(a.advance(bf2.as_bytes()).unwrap());
                    let bo = must_common(b.advance(af2.as_bytes()).unwrap());
                    assert_eq!(ao.values(), bo.values());
                    assert_eq!(ao.binding(), &original_binding);
                    ao.values().to_vec()
                }
                Boundary::Deferred => {
                    let ao = must_deferred(ao);
                    let bo = must_deferred(bo);
                    assert_eq!((ao.role(), bo.role()), (0, 1));
                    assert_eq!(ao.binding(), &original_binding);
                    ao.values()
                        .iter()
                        .zip(bo.values())
                        .map(|(&a, &b)| a.wrapping_add(b))
                        .collect()
                }
            };
            for i in 0..spec.lanes {
                let original_index =
                    indices[i].wrapping_sub(original_rin[i]) & (entries - 1) as u64;
                let changed_index = indices[i].wrapping_sub(rin[i]) & (entries - 1) as u64;
                let clear = common[i].wrapping_sub(masks.values()[i]);
                assert_eq!(clear, original_table[original_index as usize]);
                assert_ne!(clear, table[changed_index as usize]);
            }
        }
    }
}

#[test]
fn measured_aes_issuance_and_lane_serial_expansion_no_select_aes() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for n in [8, 10] {
            for lanes in [1, 17, MAX_LANES] {
                let spec = Spec {
                    index_bits: n,
                    output_bits: 64,
                    lanes,
                    boundary,
                };
                let entries = 1usize << n;
                let table: Vec<u64> = (0..entries).map(|i| u64::MAX - i as u64).collect();
                let rin: Vec<u64> = (0..lanes)
                    .map(|i| i as u64 & (entries - 1) as u64)
                    .collect();
                let indices: Vec<u64> = rin
                    .iter()
                    .map(|&r| r.wrapping_mul(3) & (entries - 1) as u64)
                    .collect();
                let budget =
                    Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
                let d = usize::from(n - 7);
                let expected_issue = 4 * d * lanes;
                let expected_each_scan = 2 * ((1usize << d) - 1) * lanes;
                let before_issue = gpu127::measured_aes_blocks();
                let (_masks, [mut a, mut b]) =
                    issue_with(spec, context(), &table, &rin, &budget, rng(71)).unwrap();
                let after_issue = gpu127::measured_aes_blocks();
                assert_eq!(after_issue - before_issue, expected_issue);
                let af = a.begin(&indices).unwrap();
                let after_a = gpu127::measured_aes_blocks();
                assert_eq!(after_a - after_issue, expected_each_scan);
                let bf = b.begin(&indices).unwrap();
                let after_b = gpu127::measured_aes_blocks();
                assert_eq!(after_b - after_a, expected_each_scan);
                let ao = a.advance(bf.as_bytes()).unwrap();
                assert_eq!(gpu127::measured_aes_blocks(), after_b);
                let bo = b.advance(af.as_bytes()).unwrap();
                assert_eq!(gpu127::measured_aes_blocks(), after_b);
                match boundary {
                    Boundary::Reconstructed => {
                        let af2 = must_frame(ao);
                        let bf2 = must_frame(bo);
                        let _ao = must_common(a.advance(bf2.as_bytes()).unwrap());
                        assert_eq!(gpu127::measured_aes_blocks(), after_b);
                        let _bo = must_common(b.advance(af2.as_bytes()).unwrap());
                        assert_eq!(gpu127::measured_aes_blocks(), after_b);
                    }
                    Boundary::Deferred => {
                        let _ao = must_deferred(ao);
                        let _bo = must_deferred(bo);
                    }
                }
                a.cancel();
                b.cancel();
                assert_eq!(gpu127::measured_aes_blocks(), after_b);
            }
        }
    }
}

#[test]
fn n10_moderate_exhaustive_and_max_lane_transition_allocation_faults() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for lanes in [17, MAX_LANES] {
            let spec = Spec {
                index_bits: 10,
                output_bits: 64,
                lanes,
                boundary,
            };
            let table: Vec<u64> = (0..1024).map(|i| u64::MAX - i as u64).collect();
            let rin: Vec<u64> = (0..lanes).map(|i| (i as u64 * 63) & 1023).collect();
            let indices: Vec<u64> = rin.iter().map(|&r| (r + 511) & 1023).collect();
            let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
            let allowance = 10_000isize;
            budget.fail_after.store(allowance, Ordering::SeqCst);
            exercise(spec, &table, &rin, &indices, &budget, rng(23)).unwrap();
            let total_allocations = allowance - budget.fail_after.load(Ordering::SeqCst);
            assert_eq!(budget.used_bytes(), 0);
            // Moderate lanes sweep every callback. At maximum lanes bound cost
            // by sweeping all issue allocations, both scan edges, and every
            // frame/Select transition, instead of ~500 repeated full scans.
            let failure_points: Vec<isize> = if lanes == 17 {
                (0..total_allocations).collect()
            } else {
                let baseline =
                    Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
                baseline.fail_after.store(allowance, Ordering::SeqCst);
                let owners = issue_with(spec, context(), &table, &rin, &baseline, rng(23)).unwrap();
                let issue_allocations = allowance - baseline.fail_after.load(Ordering::SeqCst);
                drop(owners);
                assert_eq!(baseline.used_bytes(), 0);
                let mut points: Vec<isize> = (0..issue_allocations).collect();
                let l = lanes as isize;
                points.extend([
                    issue_allocations,
                    issue_allocations + l - 1,
                    issue_allocations + l,
                    issue_allocations + l + 1,
                    issue_allocations + 2 * l,
                    issue_allocations + 2 * l + 1,
                ]);
                points.extend(issue_allocations + 2 * l + 2..total_allocations);
                points.sort_unstable();
                points.dedup();
                points
            };
            for fail in failure_points {
                assert!(fail < total_allocations);
                budget.fail_after.store(fail, Ordering::SeqCst);
                let result = exercise(spec, &table, &rin, &indices, &budget, rng(23));
                assert_eq!(
                    result,
                    Err(Error::Allocation),
                    "n10/{lanes}/{boundary:?} allocation {fail}"
                );
                assert_eq!(budget.used_bytes(), 0);
            }
        }
    }
}

#[test]
fn deferred_exact_own_share_equations_and_dependent_output_mask() {
    for m in [3, 64] {
        let spec = Spec {
            output_bits: m,
            ..spec(Boundary::Deferred)
        };
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let table: Vec<u64> = (0u64..256)
            .map(|i| i.wrapping_mul(7919) & mask(m))
            .collect();
        let (masks, [mut a, mut b]) =
            issue_with(spec, context(), &table, &[1, 255], &budget, rng(31)).unwrap();
        let (_, [mut copy_a, mut copy_b]) =
            issue_with(spec, context(), &table, &[1, 255], &budget, rng(31)).unwrap();
        for i in 0..spec.lanes {
            let r = a.correction.masks[i][0].wrapping_add(b.correction.masks[i][0]) & mask(m);
            let u = masks._select_masks.values()[i];
            assert_eq!(
                masks.values()[i],
                u.wrapping_mul(2).wrapping_sub(r) & mask(m)
            );
            assert_eq!(masks.values()[i] & 1, r & 1);
        }
        let af = a.begin(&[0, 128]).unwrap();
        let bf = b.begin(&[0, 128]).unwrap();
        let mut opened_r = [0u64; 2];
        let mut opened_c = [0u64; 2];
        for i in 0..2 {
            let offset = HEADER + 16 * i;
            let ar = u64::from_le_bytes(af.as_bytes()[offset..offset + 8].try_into().unwrap());
            let br = u64::from_le_bytes(bf.as_bytes()[offset..offset + 8].try_into().unwrap());
            let ac = u64::from_le_bytes(af.as_bytes()[offset + 8..offset + 16].try_into().unwrap());
            let bc = u64::from_le_bytes(bf.as_bytes()[offset + 8..offset + 16].try_into().unwrap());
            opened_r[i] = ar.wrapping_add(br) & mask(m);
            opened_c[i] = ac ^ bc;
        }
        let z0 = copy_a
            .correction
            .select
            .take()
            .unwrap()
            .into_deferred(&opened_c, &opened_r)
            .unwrap();
        let z1 = copy_b
            .correction
            .select
            .take()
            .unwrap()
            .into_deferred(&opened_c, &opened_r)
            .unwrap();
        let ao = must_deferred(a.advance(bf.as_bytes()).unwrap());
        let bo = must_deferred(b.advance(af.as_bytes()).unwrap());
        for (i, &r) in opened_r.iter().enumerate() {
            assert_eq!(
                ao.values()[i],
                z0.values()[i].wrapping_mul(2).wrapping_sub(r) & mask(m)
            );
            assert_eq!(bo.values()[i], z1.values()[i].wrapping_mul(2) & mask(m));
        }
    }
}

#[test]
fn formulas_and_every_admission_boundary_before_entropy_or_allocation() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for n in [8, 10] {
            for lanes in [1, MAX_LANES] {
                let spec = Spec {
                    index_bits: n,
                    output_bits: 64,
                    lanes,
                    boundary,
                };
                let r = resources(spec).unwrap();
                assert_eq!(r.frame_bytes, 80 + 16 * lanes);
                assert_eq!(
                    r.combined_reservation_bytes,
                    r.dealer_reservation_bytes + 2 * r.party_reservation_bytes
                );
                assert_eq!(
                    r.frames_per_party,
                    if boundary == Boundary::Reconstructed {
                        2
                    } else {
                        1
                    }
                );
                assert_eq!(
                    r.total_peer_frame_bytes,
                    if boundary == Boundary::Reconstructed {
                        312 + 48 * lanes
                    } else {
                        160 + 32 * lanes
                    }
                );
                assert_eq!(
                    r.paired_issue_aes_blocks,
                    if n == 8 { 4 * lanes } else { 12 * lanes }
                );
                assert_eq!(
                    r.evaluation_aes_blocks_per_party,
                    if n == 8 { 2 * lanes } else { 14 * lanes }
                );
                assert_eq!(
                    r.logical_key_bits_per_party,
                    ((if n == 8 { 384 } else { 642 }) + 6 * 64 + 1) * lanes
                );
                assert_eq!(r.full_expansion_scratch_bytes, 1 << n);
                assert!(r.combined_reservation_bytes <= MAX_ALLOCATION_BYTES);
                let budget = Budget::new(r.combined_reservation_bytes).unwrap();
                let table = vec![0; 1 << n];
                let masks = vec![0; lanes];
                let short = Budget::new(r.combined_reservation_bytes - 1).unwrap();
                short.fail_after.store(0, Ordering::SeqCst);
                assert!(matches!(
                    issue_with(spec, context(), &table, &masks, &short, |_| panic!(
                        "short admission entropy"
                    )),
                    Err(Error::Resource)
                ));
                assert_eq!(short.used_bytes(), 0);
                let owners = issue_with(spec, context(), &table, &masks, &budget, rng(1)).unwrap();
                assert_eq!(budget.used_bytes(), r.combined_reservation_bytes);
                drop(owners);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
        // Exhaust all smaller integer byte limits for minimum geometry. Larger
        // geometries above check exact-fit and one-byte-short admission.
        let spec = Spec {
            lanes: 1,
            ..spec(boundary)
        };
        let r = resources(spec).unwrap();
        for limit in 1..r.combined_reservation_bytes {
            let budget = Budget::new(limit).unwrap();
            budget.fail_after.store(0, Ordering::SeqCst);
            assert!(matches!(
                issue_with(spec, context(), &[0; 256], &[0], &budget, |_| panic!(
                    "pre-admission entropy"
                )),
                Err(Error::Resource)
            ));
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn complete_preflight_precedes_reservation_entropy_and_allocation() {
    let spec = spec(Boundary::Deferred);
    let budget = Budget::new(1).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    let mut bad_table = [0; 256];
    bad_table[255] = 8;
    assert!(matches!(
        issue_with(spec, context(), &bad_table, &[0, 0], &budget, |_| panic!()),
        Err(Error::Domain)
    ));
    assert!(matches!(
        issue_with(spec, context(), &[0; 256], &[0, 256], &budget, |_| panic!()),
        Err(Error::Domain)
    ));
    assert!(matches!(
        issue_with(spec, context(), &[0; 255], &[0, 0], &budget, |_| panic!()),
        Err(Error::Domain)
    ));
    assert!(matches!(
        issue_with(spec, context(), &[0; 256], &[0], &budget, |_| panic!()),
        Err(Error::Domain)
    ));
    let mut bad = context();
    bad.first_tensor_index = u64::MAX - 1;
    assert!(matches!(
        issue_with(spec, bad, &[0; 256], &[0, 0], &budget, |_| panic!()),
        Err(Error::Resource)
    ));
    for i in 0..5 {
        for j in i + 1..5 {
            let mut bad = context();
            let mut wires = [
                bad.input_wire,
                bad.output_wire,
                bad.arithmetic_wire,
                bad.correction_wire,
                bad.select_wire,
            ];
            wires[j] = wires[i];
            [
                bad.input_wire,
                bad.output_wire,
                bad.arithmetic_wire,
                bad.correction_wire,
                bad.select_wire,
            ] = wires;
            assert!(matches!(
                issue_with(spec, bad, &[0; 256], &[0, 0], &budget, |_| panic!()),
                Err(Error::Context)
            ));
        }
    }
    for bad in [
        Spec {
            index_bits: 9,
            ..spec
        },
        Spec {
            output_bits: 2,
            ..spec
        },
        Spec { lanes: 0, ..spec },
        Spec {
            lanes: MAX_LANES + 1,
            ..spec
        },
    ] {
        assert_eq!(resources(bad), Err(Error::Domain));
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn exhaustive_entropy_failure_unwind_and_requested_bytes() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        let spec = spec(boundary);
        let r = resources(spec).unwrap();
        let budget = Budget::new(r.combined_reservation_bytes).unwrap();
        let mut calls = 0;
        let mut bytes = 0;
        {
            let _owners = issue_with(spec, context(), &[0; 256], &[0, 255], &budget, |out| {
                calls += 1;
                bytes += out.len();
                out.fill(0);
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(bytes, r.random_bytes);
        assert_eq!(budget.used_bytes(), 0);
        for fail in 0..calls {
            let mut call = 0;
            let result = issue_with(spec, context(), &[0; 256], &[0, 255], &budget, |out| {
                if call == fail {
                    return Err(Error::Randomness);
                }
                call += 1;
                out.fill(0);
                Ok(())
            });
            assert!(matches!(result, Err(Error::Randomness)));
            drop(result);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn exhaustive_allocation_failure_across_issuance_scans_frames_and_outputs() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        let spec = spec(boundary);
        let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
        let mut succeeded = false;
        for fail in 0..100 {
            budget.fail_after.store(fail, Ordering::SeqCst);
            let result = exercise(spec, &[3; 256], &[0, 255], &[255, 0], &budget, rng(13));
            assert_eq!(budget.used_bytes(), 0, "allocation boundary {fail}");
            match result {
                Ok(()) => {
                    succeeded = true;
                    break;
                }
                Err(Error::Allocation) => {}
                Err(other) => panic!("unexpected failure {other:?} at {fail}"),
            }
        }
        assert!(succeeded);
    }
}

#[test]
fn full_first_frame_validation_and_burning() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        let spec = spec(boundary);
        for offset in 0..80 + 16 * spec.lanes {
            // Payload modifications can be canonical and are unauthenticated.
            // Only corrupt header bytes and deliberately noncanonical words.
            if offset >= HEADER && ![HEADER, HEADER + 8, HEADER + 16, HEADER + 24].contains(&offset)
            {
                continue;
            }
            let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
            {
                let (_, [mut a, mut b]) =
                    issue_with(spec, context(), &[0; 256], &[0, 0], &budget, rng(1)).unwrap();
                let _af = a.begin(&[0, 0]).unwrap();
                let bf = b.begin(&[0, 0]).unwrap();
                let mut bad = bf.as_bytes().to_vec();
                bad[offset] ^= if offset < HEADER { 1 } else { 0x80 };
                assert!(matches!(
                    a.advance(&bad),
                    Err(Error::Context | Error::Domain)
                ));
                assert!(a.state.is_none());
                assert!(matches!(a.advance(bf.as_bytes()), Err(Error::Consumed)));
            }
            assert_eq!(budget.used_bytes(), 0);
        }
        for trim in [true, false] {
            let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
            let (_, [mut a, mut b]) =
                issue_with(spec, context(), &[0; 256], &[0, 0], &budget, rng(1)).unwrap();
            let _af = a.begin(&[0, 0]).unwrap();
            let bf = b.begin(&[0, 0]).unwrap();
            let mut bad = bf.as_bytes().to_vec();
            if trim {
                bad.pop();
            } else {
                bad.push(0);
            }
            assert!(matches!(a.advance(&bad), Err(Error::Context)));
        }
    }
}

#[test]
fn invocation_context_table_role_and_common_input_substitution() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        let spec = spec(boundary);
        for mutation in 0..15 {
            let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let (_, [mut a, mut sibling]) =
                issue_with(spec, context(), &[1; 256], &[0, 0], &budget, rng(1)).unwrap();
            let mut other_context = context();
            let mut table = [1; 256];
            let mut seed = 1;
            let mut other_spec = spec;
            let mut indices = [0, 0];
            match mutation {
                0 => seed = 2, // fresh issuance
                1 => other_context.plan[0] ^= 1,
                2 => other_context.session[0] ^= 1,
                3 => other_context.operation[0] ^= 1,
                4 => other_context.input_wire[0] ^= 1,
                5 => other_context.output_wire[0] ^= 1,
                6 => other_context.arithmetic_wire[0] ^= 1,
                7 => other_context.correction_wire[0] ^= 1,
                8 => other_context.select_wire[0] ^= 1,
                9 => other_context.numeric_contract[0] ^= 1,
                10 => other_context.first_tensor_index += 1,
                11 => table[255] ^= 1,
                12 => {
                    other_spec.boundary = if boundary == Boundary::Deferred {
                        Boundary::Reconstructed
                    } else {
                        Boundary::Deferred
                    }
                }
                13 => indices[1] = 1,
                _ => {} // same-role frame from the same issuance
            }
            let (_, [mut other_a, mut other_b]) = issue_with(
                other_spec,
                other_context,
                &table,
                &[0, 0],
                &budget,
                rng(seed),
            )
            .unwrap();
            let af = a.begin(&[0, 0]).unwrap();
            let peer = if mutation == 14 {
                other_a.begin(&indices).unwrap()
            } else {
                other_b.begin(&indices).unwrap()
            };
            assert!(matches!(a.advance(peer.as_bytes()), Err(Error::Context)));
            assert!(matches!(a.advance(af.as_bytes()), Err(Error::Consumed)));
            sibling.cancel();
        }
    }
}

#[test]
fn retained_frames_outputs_cancel_and_dealer_first_drop() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        let spec = spec(boundary);
        let r = resources(spec).unwrap();
        let budget = Budget::new(r.combined_reservation_bytes).unwrap();
        let (masks, [mut a, mut b]) =
            issue_with(spec, context(), &[1; 256], &[0, 0], &budget, rng(1)).unwrap();
        assert_eq!(budget.used_bytes(), r.combined_reservation_bytes);
        drop(masks);
        assert_eq!(budget.used_bytes(), 2 * r.party_reservation_bytes);
        let af = a.begin(&[0, 255]).unwrap();
        let bf = b.begin(&[0, 255]).unwrap();
        assert_eq!(budget.used_bytes(), 2 * r.party_reservation_bytes);
        let ao = a.advance(bf.as_bytes()).unwrap();
        let bo = b.advance(af.as_bytes()).unwrap();
        if boundary == Boundary::Reconstructed {
            let af2 = must_frame(ao);
            let bf2 = must_frame(bo);
            let ao = must_common(a.advance(bf2.as_bytes()).unwrap());
            let bo = must_common(b.advance(af2.as_bytes()).unwrap());
            let before = budget.used_bytes();
            a.cancel();
            b.cancel();
            assert_eq!(budget.used_bytes(), before);
            drop((a, b));
            assert_eq!(
                budget.used_bytes(),
                2 * (frame_charge(spec) + size_of::<Frame<'static>>() + output_charge(spec))
                    + 2 * (selection::resources(spec.select()).unwrap().frame_bytes
                        + size_of::<selection::Frame<'static>>()
                        + ALLOCATOR)
            );
            drop((af2, bf2));
            assert_eq!(
                budget.used_bytes(),
                2 * (frame_charge(spec) + output_charge(spec))
            );
            drop((ao, bo));
        } else {
            let ao = must_deferred(ao);
            let bo = must_deferred(bo);
            let before = budget.used_bytes();
            a.cancel();
            b.cancel();
            assert_eq!(budget.used_bytes(), before);
            drop((a, b));
            assert_eq!(
                budget.used_bytes(),
                2 * (frame_charge(spec) + output_charge(spec))
            );
            drop((ao, bo));
        }
        assert_eq!(budget.used_bytes(), 2 * frame_charge(spec));
        drop((af, bf));
        assert_eq!(budget.used_bytes(), 0);

        let (masks, [mut a, mut b]) =
            issue_with(spec, context(), &[0; 256], &[0, 0], &budget, rng(2)).unwrap();
        let before = budget.used_bytes();
        a.cancel();
        b.cancel();
        assert_eq!(budget.used_bytes(), before);
        assert!(matches!(a.begin(&[0, 0]), Err(Error::Consumed)));
        drop((masks, a, b));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn wrong_order_repeats_invalid_last_input_and_second_phase_burn() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        let spec = spec(boundary);
        let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
        for mode in 0..5 {
            let (_, [mut a, mut b]) =
                issue_with(spec, context(), &[1; 256], &[0, 0], &budget, rng(1)).unwrap();
            match mode {
                0 => assert!(matches!(a.advance(&[]), Err(Error::Consumed))),
                1 => assert!(matches!(a.begin(&[0, 256]), Err(Error::Domain))),
                2 => assert!(matches!(a.begin(&[0]), Err(Error::Domain))),
                3 => {
                    let _frame = a.begin(&[0, 0]).unwrap();
                    assert!(matches!(a.begin(&[0, 0]), Err(Error::Consumed)));
                }
                _ => {
                    let af = a.begin(&[0, 0]).unwrap();
                    let bf = b.begin(&[0, 0]).unwrap();
                    let _output = a.advance(bf.as_bytes()).unwrap();
                    // A first-phase frame cannot be used as Select's phase two;
                    // deferred is already terminal and cannot exchange again.
                    assert!(a.advance(af.as_bytes()).is_err());
                }
            }
            assert!(a.state.is_none());
            assert!(matches!(a.begin(&[0, 0]), Err(Error::Consumed)));
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}
