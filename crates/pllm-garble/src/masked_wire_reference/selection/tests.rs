use super::*;
use std::sync::atomic::Ordering;

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        predicate_wire: [4; 32],
        input_wire: [5; 32],
        output_wire: [6; 32],
        first_tensor_index: 7,
    }
}
fn spec() -> Spec {
    Spec { bits: 8, lanes: 2 }
}
fn execute<'a>(
    parties: &mut [Party<'a>; 2],
    predicate: &[u64],
    input: &[u64],
) -> [MaskedOutput<'a>; 2] {
    let [a, b] = parties;
    let left = a.begin(predicate, input).unwrap();
    let right = b.begin(predicate, input).unwrap();
    [
        a.finish(right.as_bytes()).unwrap(),
        b.finish(left.as_bytes()).unwrap(),
    ]
}
fn oracle(bits: u8, b: u64, x: u64, rz: u64) -> u64 {
    // Independent wide arithmetic, including signed ring encodings.
    ((u128::from(b) * u128::from(x) + u128::from(rz)) % (1u128 << bits)) as u64
}

#[test]
fn native_width3_all_inputs_masks_output_masks_and_both_predicate_bits_batched() {
    let mut rows = Vec::new();
    for beta in 0..=1u64 {
        for b in 0..=1u64 {
            for rx in 0..8u64 {
                for x in 0..8u64 {
                    for rz in 0..8u64 {
                        rows.push((beta, b, rx, x, rz));
                    }
                }
            }
        }
    }
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for rows in rows.chunks(MAX_LANES) {
        let spec = Spec {
            bits: 3,
            lanes: rows.len(),
        };
        let betas: Vec<_> = rows.iter().map(|r| r.0).collect();
        let masks: Vec<_> = rows.iter().map(|r| r.2).collect();
        let predicates: Vec<_> = rows.iter().map(|r| r.0 ^ r.1).collect();
        let inputs: Vec<_> = rows.iter().map(|r| (r.2 + r.3) % 8).collect();
        let mut call = 0;
        let (out_masks, mut parties) =
            issue_with(spec, context(), &betas, &masks, &budget, |bytes| {
                if call > 0 && (call - 1) % 6 == 0 {
                    bytes.copy_from_slice(&rows[(call - 1) / 6].4.to_le_bytes());
                } else {
                    bytes.fill((call * 37) as u8);
                }
                call += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(call, 1 + 6 * rows.len());
        // Assert actual issuance has exactly the five pinned correlations,
        // independently using wide modular arithmetic, not evaluate().
        for (i, &(beta, _, rx, _, rz)) in rows.iter().enumerate() {
            let expected = [
                beta,
                rx,
                (beta * rx + rz) % 8,
                2 * beta,
                (16 - 2 * beta * rx) % 8,
            ];
            for (j, expected) in expected.into_iter().enumerate() {
                let sum = parties[0].state.as_ref().unwrap().material[i][j]
                    + parties[1].state.as_ref().unwrap().material[i][j];
                assert_eq!(sum % 8, expected);
            }
        }
        let outputs = execute(&mut parties, &predicates, &inputs);
        assert_eq!(outputs[0].values(), outputs[1].values());
        for (i, &(_, b, _, x, rz)) in rows.iter().enumerate() {
            assert_eq!(out_masks.values()[i], rz);
            assert_eq!(outputs[0].values()[i], oracle(3, b, x, rz));
        }
        drop((out_masks, parties, outputs));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn independent_algebra_larger_small_rings_and_ring64_wrapping_edges() {
    // No per-row native allocations. Independent wide-derived correlations and
    // split choices check evaluator against b*x, not a second copy of its formula.
    for bits in 3..=7 {
        let modulus = 1u128 << bits;
        for beta in 0..=1u64 {
            for b in 0..=1u64 {
                for rx in 0..modulus as u64 {
                    for x in 0..modulus as u64 {
                        for rz in [0, 1, modulus as u64 - 1] {
                            let beta128 = u128::from(beta);
                            let rx128 = u128::from(rx);
                            let totals = [
                                beta128,
                                rx128,
                                beta128 * rx128 + u128::from(rz),
                                2 * beta128,
                                2 * modulus - 2 * beta128 * rx128,
                            ];
                            let left =
                                std::array::from_fn(|j| (j as u64 * 13 + x + rx) & mask(bits));
                            let right = std::array::from_fn(|j| {
                                ((totals[j] + modulus - u128::from(left[j])) % modulus) as u64
                            });
                            let bhat = b ^ beta;
                            let xhat = (x + rx) & mask(bits);
                            let result = evaluate(0, bits, bhat, xhat, &left)
                                + evaluate(1, bits, bhat, xhat, &right);
                            assert_eq!(result & mask(bits), oracle(bits, b, x, rz));
                        }
                    }
                }
            }
        }
    }
    let edges = [0, 1, u64::MAX, u64::MAX - 1, 1 << 63, (1 << 63) - 1];
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for bits in [3, 9, 32, 50, 64] {
        for beta in 0..=1 {
            for b in 0..=1 {
                for rx in edges.map(|x| x & mask(bits)) {
                    let spec = Spec {
                        bits,
                        lanes: edges.len(),
                    };
                    let (masks, mut parties) =
                        issue(spec, context(), &[beta; 6], &[rx; 6], &budget).unwrap();
                    let inputs = edges.map(|x| (x & mask(bits)).wrapping_add(rx) & mask(bits));
                    let outputs = execute(&mut parties, &[beta ^ b; 6], &inputs);
                    assert_eq!(outputs[0].values(), outputs[1].values());
                    for (i, x) in edges.into_iter().enumerate() {
                        assert_eq!(
                            outputs[0].values()[i],
                            oracle(bits, b, x & mask(bits), masks.values()[i])
                        );
                    }
                    drop((masks, parties, outputs));
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}

#[test]
fn full_ring_output_distribution_for_each_clear_predicate_and_signed_value() {
    let spec = Spec { bits: 4, lanes: 16 };
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for b in 0..=1 {
        for x in [0, 1, 7, 8, 15] {
            let mut call = 0;
            let (masks, mut parties) =
                issue_with(spec, context(), &[1; 16], &[13; 16], &budget, |bytes| {
                    bytes.fill(0xa5);
                    if call > 0 && (call - 1) % 6 == 0 {
                        bytes.copy_from_slice(&(((call - 1) / 6) as u64).to_le_bytes());
                    }
                    call += 1;
                    Ok(())
                })
                .unwrap();
            let outputs = execute(&mut parties, &[b ^ 1; 16], &[(x + 13) % 16; 16]);
            let mut histogram = [0; 16];
            for (i, &actual) in outputs[0].values().iter().enumerate() {
                assert_eq!(masks.values()[i], i as u64);
                assert_eq!(actual, oracle(4, b, x, i as u64));
                histogram[actual as usize] += 1;
            }
            assert_eq!(histogram, [1; 16]);
        }
    }
}

#[test]
fn exact_resources_rng_requests_256_lanes_and_retained_owner_leases() {
    let spec = Spec {
        bits: 64,
        lanes: MAX_LANES,
    };
    let cost = resources(spec).unwrap();
    assert_eq!(cost.logical_key_bits_per_party, 5 * 64 * MAX_LANES);
    assert_eq!(cost.party_material_payload_bytes, 40 * MAX_LANES);
    assert_eq!(cost.output_bits, 64);
    assert_eq!(cost.output_mask_entropy_bits, 64 * MAX_LANES);
    assert_eq!(cost.paired_issue_aes_blocks, 0);
    assert_eq!(cost.evaluation_aes_blocks_per_party, 0);
    assert_eq!(cost.frames_per_party, 1);
    assert_eq!(cost.frame_bytes, 76 + 8 * MAX_LANES);
    assert_eq!(cost.total_peer_frame_bytes, 2 * cost.frame_bytes);
    assert_eq!(cost.borrowed_input_mask_bytes, 16 * MAX_LANES);
    assert_eq!(cost.borrowed_masked_input_bytes, 32 * MAX_LANES);
    assert_eq!(
        cost.party_reservation_bytes,
        size_of::<Party<'static>>()
            + 64 * MAX_LANES
            + 128
            + frame_charge(spec)
            + output_charge(spec)
            + 8192
    );
    assert_eq!(
        cost.dealer_reservation_bytes,
        size_of::<OutputMasks<'static>>() + 24 * MAX_LANES + 64
    );
    assert_eq!(
        cost.combined_reservation_bytes,
        2 * cost.party_reservation_bytes + cost.dealer_reservation_bytes
    );
    let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
    let mut requests = Vec::new();
    let before = crate::compact_dpf::gpu127::measured_aes_blocks();
    let (masks, [mut a, mut b]) = issue_with(
        spec,
        context(),
        &[1; MAX_LANES],
        &[u64::MAX; MAX_LANES],
        &budget,
        |bytes| {
            requests.push(bytes.len());
            bytes.fill(0xa5);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(requests[0], 32);
    assert_eq!(requests.len(), 1 + 6 * MAX_LANES);
    assert!(requests[1..].iter().all(|&n| n == 8));
    assert_eq!(requests.iter().sum::<usize>(), cost.random_bytes);
    assert_eq!(cost.random_bytes, 32 + 48 * MAX_LANES);
    assert_eq!(budget.used_bytes(), cost.combined_reservation_bytes);
    let x = a
        .begin(&[0; MAX_LANES], &[u64::MAX - 1; MAX_LANES])
        .unwrap();
    let y = b
        .begin(&[0; MAX_LANES], &[u64::MAX - 1; MAX_LANES])
        .unwrap();
    assert_eq!(x.as_bytes().len(), cost.frame_bytes);
    // finish moves backing; must not request an allocation.
    budget.fail_after.store(0, Ordering::SeqCst);
    let left = a.finish(y.as_bytes()).unwrap();
    let right = b.finish(x.as_bytes()).unwrap();
    assert_eq!(crate::compact_dpf::gpu127::measured_aes_blocks(), before);
    assert_eq!(left.values(), right.values());
    for (i, &value) in left.values().iter().enumerate() {
        assert_eq!(value, oracle(64, 1, u64::MAX, masks.values()[i]));
    }
    assert!(matches!(a.finish(y.as_bytes()), Err(Error::Consumed)));
    assert!(matches!(
        b.begin(&[0; MAX_LANES], &[0; MAX_LANES]),
        Err(Error::Consumed)
    ));
    drop((a, b, masks));
    assert_eq!(
        budget.used_bytes(),
        2 * frame_charge(spec) + 2 * output_charge(spec)
    );
    drop((x, y));
    assert_eq!(budget.used_bytes(), 2 * output_charge(spec));
    drop(left);
    assert_eq!(budget.used_bytes(), output_charge(spec));
    drop(right);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn public_preflight_before_allocations_entropy_and_exact_budget_admission() {
    let spec = spec();
    let cost = resources(spec).unwrap();
    let budget = Budget::new(cost.combined_reservation_bytes - 1).unwrap();
    assert!(matches!(
        issue_with(spec, context(), &[0; 2], &[0; 2], &budget, |_| panic!(
            "entropy before admission"
        )),
        Err(Error::Resource)
    ));
    assert_eq!(budget.used_bytes(), 0);
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    for invalid in [
        Spec { bits: 2, ..spec },
        Spec { bits: 65, ..spec },
        Spec { lanes: 0, ..spec },
        Spec { lanes: 257, ..spec },
        Spec {
            lanes: usize::MAX,
            ..spec
        },
    ] {
        assert!(matches!(
            issue_with(invalid, context(), &[], &[], &budget, |_| panic!(
                "entropy before geometry"
            )),
            Err(Error::Domain)
        ));
    }
    for (predicate, input) in [
        (&[][..], &[0; 2][..]),
        (&[0][..], &[0; 2][..]),
        (&[0; 3][..], &[0; 2][..]),
        (&[0; 2][..], &[][..]),
        (&[0; 2][..], &[0][..]),
        (&[0; 2][..], &[0; 3][..]),
        (&[0, 2][..], &[0; 2][..]),
        (&[0; 2][..], &[0, 256][..]),
    ] {
        assert!(matches!(
            issue_with(spec, context(), predicate, input, &budget, |_| panic!(
                "entropy before canonical masks"
            )),
            Err(Error::Domain)
        ));
    }
    for mutation in 0..4 {
        let mut bad = context();
        match mutation {
            0 => bad.output_wire = bad.input_wire,
            1 => bad.output_wire = bad.predicate_wire,
            2 => bad.input_wire = bad.predicate_wire,
            _ => bad.first_tensor_index = u64::MAX,
        }
        let expected = if mutation == 3 {
            Error::Resource
        } else {
            Error::Context
        };
        assert!(
            matches!(issue_with(spec, bad, &[0; 2], &[0; 2], &budget, |_| panic!("entropy before context")), Err(e) if e == expected)
        );
    }
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn every_rng_allocation_failure_and_cancel_refunds_only_after_owners_retire() {
    let spec = spec();
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for allowed in 0..=5 {
        budget.fail_after.store(allowed, Ordering::SeqCst);
        let mut calls = 0;
        let result = issue_with(spec, context(), &[1; 2], &[7; 2], &budget, |bytes| {
            calls += 1;
            bytes.fill(0x55);
            Ok(())
        });
        if allowed < 5 {
            assert!(matches!(result, Err(Error::Allocation)));
            assert_eq!(calls, 0);
        } else {
            assert!(result.is_ok());
            assert_eq!(calls, 13);
        }
        drop(result);
        assert_eq!(budget.used_bytes(), 0);
    }
    budget.fail_after.store(-1, Ordering::SeqCst);
    for failure in 0..=13 {
        let mut calls = 0;
        let result = issue_with(spec, context(), &[1; 2], &[7; 2], &budget, |bytes| {
            let current = calls;
            calls += 1;
            bytes.fill(0x37); // Also exercise partial writes before failure.
            if current == failure {
                Err(Error::Randomness)
            } else {
                Ok(())
            }
        });
        if failure == 13 {
            assert!(result.is_ok());
            assert_eq!(calls, 13);
        } else {
            assert!(matches!(result, Err(Error::Randomness)));
            assert_eq!(calls, failure + 1);
        }
        drop(result);
        assert_eq!(budget.used_bytes(), 0);
    }
    for role in 0..2 {
        for begun in [false, true] {
            budget.fail_after.store(-1, Ordering::SeqCst);
            let (masks, mut parties) = issue(spec, context(), &[1; 2], &[7; 2], &budget).unwrap();
            let retained = if begun {
                Some([
                    parties[0].begin(&[0; 2], &[6; 2]).unwrap(),
                    parties[1].begin(&[0; 2], &[6; 2]).unwrap(),
                ])
            } else {
                None
            };
            budget.fail_after.store(0, Ordering::SeqCst);
            let output = if let Some(frames) = &retained {
                Some(parties[role].finish(frames[1 - role].as_bytes()).unwrap())
            } else {
                assert!(matches!(
                    parties[role].begin(&[0; 2], &[6; 2]),
                    Err(Error::Allocation)
                ));
                None
            };
            assert!(matches!(
                parties[role].begin(&[0; 2], &[6; 2]),
                Err(Error::Consumed)
            ));
            parties[1 - role].cancel();
            parties[1 - role].cancel();
            assert!(matches!(
                parties[1 - role].finish(&[]),
                Err(Error::Consumed)
            ));
            drop((parties, masks));
            assert_eq!(
                budget.used_bytes(),
                if begun {
                    2 * frame_charge(spec) + output_charge(spec)
                } else {
                    0
                }
            );
            drop(retained);
            assert_eq!(
                budget.used_bytes(),
                if begun { output_charge(spec) } else { 0 }
            );
            drop(output);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn malformed_role_phase_binding_canonical_payload_and_replay_burn_both_roles() {
    let spec = spec();
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for role in 0..2 {
        for mutation in 0..15 {
            let (masks, mut parties) = issue(spec, context(), &[1; 2], &[7; 2], &budget).unwrap();
            let frames = [
                parties[0].begin(&[0; 2], &[6; 2]).unwrap(),
                parties[1].begin(&[0; 2], &[6; 2]).unwrap(),
            ];
            let mut peer = frames[1 - role].as_bytes().to_vec();
            let mut expected = Error::Context;
            match mutation {
                0 => peer[0] ^= 1,
                1 => peer[8] ^= 1,
                2 => peer[40] ^= 1,
                3 => peer[72] = role as u8,
                4 => peer[72] = 2,
                5 => peer[73] = 0,
                6 => peer[73] = 2,
                7 => peer[74] ^= 1,
                8 => peer[75] ^= 1,
                9 => {
                    peer.pop();
                }
                10 => peer.push(0),
                11 => peer.clear(),
                12 => peer.truncate(HEADER - 1),
                13 | 14 => {
                    let offset = HEADER + 8 * (mutation - 13);
                    peer[offset..offset + 8].copy_from_slice(&256u64.to_le_bytes());
                    expected = Error::Domain;
                }
                _ => unreachable!(),
            }
            assert!(matches!(parties[role].finish(&peer), Err(e) if e == expected));
            assert!(matches!(
                parties[role].finish(frames[1 - role].as_bytes()),
                Err(Error::Consumed)
            ));
            parties[1 - role].cancel();
            drop((masks, parties, frames));
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn both_complete_input_views_lane_order_and_begin_finish_lifecycle_bind() {
    let spec = spec();
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for role in 0..2 {
        for (predicate, input) in [
            (&[][..], &[6; 2][..]),
            (&[0, 2][..], &[6; 2][..]),
            (&[0; 2][..], &[6][..]),
            (&[0; 2][..], &[6, 256][..]),
        ] {
            let (masks, mut parties) = issue(spec, context(), &[1; 2], &[7; 2], &budget).unwrap();
            assert!(matches!(
                parties[role].begin(predicate, input),
                Err(Error::Domain)
            ));
            assert!(matches!(
                parties[role].begin(&[0; 2], &[6; 2]),
                Err(Error::Consumed)
            ));
            drop((masks, parties));
            assert_eq!(budget.used_bytes(), 0);
        }
        for (predicate, input) in [
            ([0, 0], [6, 7]),
            ([1, 1], [6, 7]),
            ([1, 0], [6, 7]),
            ([0, 1], [5, 7]),
            ([0, 1], [6, 8]),
            ([0, 1], [7, 6]),
        ] {
            let (masks, mut parties) = issue(spec, context(), &[1; 2], &[7; 2], &budget).unwrap();
            let local = parties[role].begin(&[0, 1], &[6, 7]).unwrap();
            let peer = parties[1 - role].begin(&predicate, &input).unwrap();
            assert!(matches!(
                parties[role].finish(peer.as_bytes()),
                Err(Error::Context)
            ));
            assert!(matches!(
                parties[role].finish(peer.as_bytes()),
                Err(Error::Consumed)
            ));
            drop((masks, parties, local, peer));
            assert_eq!(budget.used_bytes(), 0);
        }
        for lifecycle in 0..3 {
            let (masks, mut parties) = issue(spec, context(), &[1; 2], &[7; 2], &budget).unwrap();
            let peer = parties[1 - role].begin(&[0; 2], &[6; 2]).unwrap();
            let local = if lifecycle > 0 {
                Some(parties[role].begin(&[0; 2], &[6; 2]).unwrap())
            } else {
                None
            };
            match lifecycle {
                0 => assert!(matches!(
                    parties[role].finish(peer.as_bytes()),
                    Err(Error::Context)
                )),
                1 => assert!(matches!(
                    parties[role].begin(&[0; 2], &[6; 2]),
                    Err(Error::Domain)
                )),
                _ => {
                    parties[role].cancel();
                }
            }
            assert!(matches!(
                parties[role].finish(peer.as_bytes()),
                Err(Error::Consumed)
            ));
            assert!(matches!(
                parties[role].begin(&[0; 2], &[6; 2]),
                Err(Error::Consumed)
            ));
            drop((masks, parties, peer, local));
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn every_context_geometry_and_nonce_field_binds_independently() {
    let spec = spec();
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for mutation in 0..10 {
        let mut other_context = context();
        let mut other_spec = spec;
        match mutation {
            0 => other_context.plan[0] ^= 1,
            1 => other_context.session[0] ^= 1,
            2 => other_context.operation[0] ^= 1,
            3 => other_context.predicate_wire[0] ^= 1,
            4 => other_context.input_wire[0] ^= 1,
            5 => other_context.output_wire[0] ^= 1,
            6 => other_context.first_tensor_index += 1,
            7 => {} // Nonce alone differs below.
            8 => other_spec.bits = 9,
            _ => other_spec.lanes = 3,
        }
        let (masks, mut parties) =
            issue_with(spec, context(), &[1; 2], &[7; 2], &budget, |bytes| {
                bytes.fill(0x65);
                Ok(())
            })
            .unwrap();
        let mut call = 0;
        let (other_masks, mut other) = issue_with(
            other_spec,
            other_context,
            &vec![1; other_spec.lanes],
            &vec![7; other_spec.lanes],
            &budget,
            |bytes| {
                bytes.fill(if mutation == 7 && call == 0 {
                    0x66
                } else {
                    0x65
                });
                call += 1;
                Ok(())
            },
        )
        .unwrap();
        let local = parties[0].begin(&[0; 2], &[6; 2]).unwrap();
        let foreign = other[1]
            .begin(&vec![0; other_spec.lanes], &vec![6; other_spec.lanes])
            .unwrap();
        assert_ne!(&local.as_bytes()[8..40], &foreign.as_bytes()[8..40]);
        assert!(matches!(
            parties[0].finish(foreign.as_bytes()),
            Err(Error::Context)
        ));
        assert!(matches!(
            parties[0].finish(foreign.as_bytes()),
            Err(Error::Consumed)
        ));
        drop((masks, parties, other_masks, other, local, foreign));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn cancellation_retains_exact_charges_and_failures_preserve_existing_budget_owner() {
    let spec = spec();
    let cost = resources(spec).unwrap();
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = budget.reserve(123).unwrap();
    for begun in [false, true] {
        let (masks, [mut a, mut b]) = issue(spec, context(), &[1; 2], &[7; 2], &budget).unwrap();
        let frames = if begun {
            Some([
                a.begin(&[0; 2], &[6; 2]).unwrap(),
                b.begin(&[0; 2], &[6; 2]).unwrap(),
            ])
        } else {
            None
        };
        a.cancel();
        a.cancel();
        b.cancel();
        assert_eq!(budget.used_bytes(), 123 + cost.combined_reservation_bytes);
        drop((a, b));
        let retained = if begun { 2 * frame_charge(spec) } else { 0 };
        assert_eq!(
            budget.used_bytes(),
            123 + cost.dealer_reservation_bytes + retained
        );
        drop(masks);
        assert_eq!(budget.used_bytes(), 123 + retained);
        drop(frames);
        assert_eq!(budget.used_bytes(), 123);
    }
    budget.fail_after.store(3, Ordering::SeqCst);
    assert!(matches!(
        issue(spec, context(), &[1; 2], &[7; 2], &budget),
        Err(Error::Allocation)
    ));
    assert_eq!(budget.used_bytes(), 123);
    drop(baseline);
    assert_eq!(budget.used_bytes(), 0);
}
