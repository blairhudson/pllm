use super::*;
use std::sync::atomic::Ordering;

#[test]
fn reserved_issuance_matches_standalone_and_moves_output_without_refund() {
    for gate in [
        Gate::Tr { shift: 6 },
        Gate::GapArs { shift: 6 },
        Gate::Drelu,
    ] {
        let spec = Spec {
            bits: 38,
            lanes: 2,
            gate,
        };
        let cost = resources(spec).unwrap().combined_reservation_bytes;
        let budget = Budget::new(cost + 97).unwrap();
        let guard = budget.reserve(97).unwrap();
        let reservation = budget.reserve(cost).unwrap();
        let mut calls = 0usize;
        let (masks, mut parties) =
            issue_reserved_with(spec, context(), &[0, 1], reservation, |bytes| {
                assert_eq!(budget.used_bytes(), cost + 97);
                calls += 1;
                bytes.fill(calls as u8);
                Ok(())
            })
            .unwrap();
        let reference_budget = Budget::new(cost).unwrap();
        let mut reference_calls = 0usize;
        let (reference_masks, mut reference_parties) =
            issue_with(spec, context(), &[0, 1], &reference_budget, |bytes| {
                reference_calls += 1;
                bytes.fill(reference_calls as u8);
                Ok(())
            })
            .unwrap();
        assert_eq!(calls, reference_calls);
        assert_eq!(masks.values(), reference_masks.values());
        let [left, right] = execute(spec, &mut parties, &[64, 0]).unwrap();
        let reference = execute(spec, &mut reference_parties, &[64, 0]).unwrap();
        assert_eq!(left.values(), reference[0].values());
        assert_eq!(right.values(), reference[1].values());
        let pointer = left.values().as_ptr();
        let used = budget.used_bytes();
        let moved = left;
        assert_eq!(moved.values().as_ptr(), pointer);
        assert_eq!(budget.used_bytes(), used);
        drop(moved);
        drop((right, masks, parties, guard));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn reserved_issuance_rejects_nonexact_lease_before_entropy() {
    let spec = Spec {
        bits: 38,
        lanes: 1,
        gate: Gate::Tr { shift: 6 },
    };
    let cost = resources(spec).unwrap().combined_reservation_bytes;
    for bytes in [cost - 1, cost + 1] {
        let budget = Budget::new(cost + 1).unwrap();
        let reservation = budget.reserve(bytes).unwrap();
        let result = issue_reserved_with(spec, context(), &[0], reservation, |_| {
            panic!("nonexact admission must not draw randomness")
        });
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
        first_tensor_index: 7,
    }
}
fn frame(step: Step<'_>) -> Frame<'_> {
    match step {
        Step::Frame(f) => f,
        _ => panic!("expected frame"),
    }
}
fn output(step: Step<'_>) -> MaskedOutput<'_> {
    match step {
        Step::Output(o) => o,
        _ => panic!("expected output"),
    }
}
fn execute<'a>(
    spec: Spec,
    parties: &mut [Party<'a>; 2],
    masked: &[u64],
) -> Result<[MaskedOutput<'a>; 2], Error> {
    let [a, b] = parties;
    let x = a.begin(masked)?;
    let y = b.begin(masked)?;
    let first = a.advance(y.as_bytes())?;
    let second = b.advance(x.as_bytes())?;
    if spec.gate == Gate::Drelu {
        return Ok([output(first), output(second)]);
    }
    let x = frame(first);
    let y = frame(second);
    Ok([
        output(a.advance(y.as_bytes())?),
        output(b.advance(x.as_bytes())?),
    ])
}
fn signed(value: u64, bits: u8) -> i128 {
    let value = i128::from(value);
    if value >= 1i128 << (bits - 1) {
        value - (1i128 << bits)
    } else {
        value
    }
}
fn oracle(spec: Spec, x: u64) -> u64 {
    match spec.gate {
        // Independent signed floor, including i64::MIN. TR keeps only the
        // reduced ring, where unsigned and signed floor have the same encoding.
        Gate::Tr { shift } => {
            signed(x, spec.bits).div_euclid(1i128 << shift) as u64 & mask(spec.bits - shift)
        }
        Gate::GapArs { shift } => {
            signed(x, spec.bits).div_euclid(1i128 << shift) as u64 & mask(spec.bits)
        }
        Gate::Drelu => u64::from(signed(x, spec.bits) >= 0),
    }
}

#[test]
fn exhaustive_small_inputs_every_input_mask_floor_sign_and_public_gap_domain() {
    for bits in 3..=6 {
        let modulus = 1u64 << bits;
        for gate in (1..bits)
            .map(|shift| Gate::Tr { shift })
            .chain((1..bits - 1).map(|shift| Gate::GapArs { shift }))
            .chain([Gate::Drelu])
        {
            let spec = Spec {
                bits,
                gate,
                lanes: 1,
            };
            let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
            for rin in 0..modulus {
                for x in 0..modulus {
                    if matches!(gate, Gate::GapArs { .. })
                        && !(-(1i128 << (bits - 2))..(1i128 << (bits - 2)))
                            .contains(&signed(x, bits))
                    {
                        continue;
                    }
                    let (masks, mut parties) = issue(spec, context(), &[rin], &budget).unwrap();
                    let outputs = execute(spec, &mut parties, &[(x + rin) % modulus]).unwrap();
                    let encoded = if gate == Gate::Drelu {
                        outputs[0].values()[0] ^ masks.values()[0]
                    } else {
                        outputs[0].values()[0].wrapping_sub(masks.values()[0])
                            & mask(resources(spec).unwrap().output_bits)
                    };
                    assert_eq!(outputs[0].values(), outputs[1].values());
                    assert_eq!(encoded, oracle(spec, x), "{spec:?}: x={x} rin={rin}");
                    drop((outputs, masks, parties));
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}

#[test]
fn signed_ring64_edges_all_shifts_nonpower_lanes_and_gap_endpoints() {
    for bits in [8, 9, 16, 32, 50, 64] {
        let max = mask(bits);
        for shift in 1..bits {
            for gate in [Gate::Tr { shift }, Gate::GapArs { shift }, Gate::Drelu] {
                if matches!(gate, Gate::GapArs { .. }) && shift >= bits - 1 {
                    continue;
                }
                let spec = Spec {
                    bits,
                    gate,
                    lanes: 7,
                };
                let quarter = 1u64 << (bits - 2);
                let inputs = if matches!(gate, Gate::GapArs { .. }) {
                    [
                        0,
                        1,
                        quarter - 1,
                        max,
                        max - quarter + 1,
                        max - quarter + 2,
                        2,
                    ]
                } else {
                    [0, 1, max, max - 1, quarter, 2 * quarter, 2 * quarter - 1]
                };
                let rins = [max, 0, 1, quarter, quarter - 1, max - 1, 42 & max];
                let masked =
                    std::array::from_fn::<_, 7, _>(|i| inputs[i].wrapping_add(rins[i]) & max);
                let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
                let (masks, mut parties) = issue(spec, context(), &rins, &budget).unwrap();
                let outputs = execute(spec, &mut parties, &masked).unwrap();
                assert_eq!(outputs[0].values(), outputs[1].values());
                for (i, &input) in inputs.iter().enumerate() {
                    let actual = if gate == Gate::Drelu {
                        outputs[0].values()[i] ^ masks.values()[i]
                    } else {
                        outputs[0].values()[i].wrapping_sub(masks.values()[i])
                            & mask(resources(spec).unwrap().output_bits)
                    };
                    assert_eq!(actual, oracle(spec, input), "{spec:?} input={input}");
                }
            }
        }
    }
}

#[test]
fn full_ring_output_mask_draws_remove_the_specific_one_bit_witness() {
    // Force every possible output-mask draw. This checks actual native issuance
    // and execution, NOT a composed-security proof or a test of RNG quality.
    for gate in [Gate::Tr { shift: 1 }, Gate::GapArs { shift: 1 }] {
        let spec = Spec {
            bits: 5,
            gate,
            lanes: 1,
        };
        let cost = resources(spec).unwrap();
        let modulus = 1usize << cost.output_bits;
        for clear in [0, 4] {
            let mut counts = vec![0; modulus];
            for desired in 0..modulus {
                let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
                let mut draws = 0;
                let (masks, mut parties) = issue_with(spec, context(), &[13], &budget, |bytes| {
                    bytes.fill(0x65);
                    // Draw zero is nonce; draw one is the complete output mask.
                    if draws == 1 {
                        bytes.copy_from_slice(&(desired as u64).to_le_bytes());
                    }
                    draws += 1;
                    Ok(())
                })
                .unwrap();
                assert_eq!(masks.values(), &[desired as u64]);
                let outputs = execute(spec, &mut parties, &[(clear + 13) & 31]).unwrap();
                counts[outputs[0].values()[0] as usize] += 1;
            }
            assert!(counts.iter().all(|&n| n == 1));
        }
    }
}

#[test]
fn exact_resources_rng_aes_frames_maximum_tile_and_retained_owners() {
    for gate in [
        Gate::Tr { shift: 12 },
        Gate::GapArs { shift: 12 },
        Gate::Drelu,
    ] {
        let spec = Spec {
            bits: 64,
            gate,
            lanes: MAX_LANES,
        };
        let cost = resources(spec).unwrap();
        let (comparison, output_bits, frames, scalars, constants) = match gate {
            Gate::Tr { shift } => (shift.max(8), 64 - shift, 2, 4, 2),
            Gate::GapArs { shift } => (shift.max(8), 64, 2, 5, 3),
            Gate::Drelu => (63, 1, 1, 2, 1),
        };
        let depth = usize::from(comparison - 7);
        assert_eq!(cost.output_bits, output_bits);
        assert_eq!(cost.comparison_bits, comparison);
        assert_eq!(
            cost.logical_key_bits_per_party,
            spec.lanes * (384 + 129 * (depth - 1) + constants * usize::from(output_bits))
        );
        assert_eq!(cost.paired_issue_aes_blocks, 4 * depth * spec.lanes);
        assert_eq!(cost.evaluation_aes_blocks_per_party, depth * spec.lanes);
        assert_eq!(cost.frames_per_party, frames);
        assert_eq!(cost.frame_bytes, 76 + 8 * spec.lanes);
        assert_eq!(cost.total_peer_frame_bytes, 2 * frames * cost.frame_bytes);
        assert_eq!(cost.borrowed_input_mask_bytes, 8 * spec.lanes);
        assert_eq!(cost.borrowed_masked_input_bytes, 16 * spec.lanes);
        assert_eq!(
            cost.party_material_payload_bytes,
            spec.lanes * size_of::<Lane>()
        );
        assert_eq!(cost.random_bytes, 32 + spec.lanes * (8 * scalars + 32));
        assert_eq!(
            cost.combined_reservation_bytes,
            2 * cost.party_reservation_bytes + cost.dealer_reservation_bytes
        );
        assert_eq!(
            cost.output_mask_entropy_bits,
            spec.lanes * usize::from(cost.output_bits)
        );
        let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
        let mut random_bytes = 0;
        let mut requests = Vec::new();
        let before_issue = gpu127::measured_aes_blocks();
        let (masks, [mut a, mut b]) =
            issue_with(spec, context(), &vec![17; spec.lanes], &budget, |bytes| {
                random_bytes += bytes.len();
                requests.push(bytes.len());
                bytes.fill(0xa5);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            gpu127::measured_aes_blocks() - before_issue,
            cost.paired_issue_aes_blocks
        );
        assert_eq!(random_bytes, cost.random_bytes);
        let mut expected_requests = vec![32];
        for _ in 0..spec.lanes {
            expected_requests.extend(vec![8; scalars]);
            expected_requests.push(32); // Both DPF roots, not optional randomness.
        }
        assert_eq!(requests, expected_requests);
        assert_eq!(budget.used_bytes(), cost.combined_reservation_bytes);
        let before_begin = gpu127::measured_aes_blocks();
        let x = a.begin(&vec![13; spec.lanes]).unwrap();
        assert_eq!(
            gpu127::measured_aes_blocks() - before_begin,
            cost.evaluation_aes_blocks_per_party
        );
        let before_begin = gpu127::measured_aes_blocks();
        let y = b.begin(&vec![13; spec.lanes]).unwrap();
        assert_eq!(
            gpu127::measured_aes_blocks() - before_begin,
            cost.evaluation_aes_blocks_per_party
        );
        assert_eq!(x.as_bytes().len(), cost.frame_bytes);
        let before_advance = gpu127::measured_aes_blocks();
        let sa = a.advance(y.as_bytes()).unwrap();
        let sb = b.advance(x.as_bytes()).unwrap();
        let mut actual_frames = x.as_bytes().len() + y.as_bytes().len();
        let outputs = if gate == Gate::Drelu {
            [output(sa), output(sb)]
        } else {
            let x = frame(sa);
            let y = frame(sb);
            actual_frames += x.as_bytes().len() + y.as_bytes().len();
            [
                output(a.advance(y.as_bytes()).unwrap()),
                output(b.advance(x.as_bytes()).unwrap()),
            ]
        };
        assert_eq!(actual_frames, cost.total_peer_frame_bytes);
        assert_eq!(
            gpu127::measured_aes_blocks(),
            before_advance,
            "advance performs no AES work"
        );
        assert_eq!(outputs[0].values(), outputs[1].values());
        for i in 0..spec.lanes {
            let clear = 13u64.wrapping_sub(17);
            let decoded = if gate == Gate::Drelu {
                outputs[0].values()[i] ^ masks.values()[i]
            } else {
                outputs[0].values()[i].wrapping_sub(masks.values()[i]) & mask(cost.output_bits)
            };
            assert_eq!(decoded, oracle(spec, clear));
        }
        drop((a, b, masks));
        assert_eq!(
            budget.used_bytes(),
            2 * frame_charge(spec) + 2 * output_charge(spec)
        );
        drop((x, y));
        assert_eq!(budget.used_bytes(), 2 * output_charge(spec));
        drop(outputs);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn public_validation_exact_admission_and_entropy_failure_precede_material() {
    let spec = Spec {
        bits: 32,
        gate: Gate::Tr { shift: 12 },
        lanes: 1,
    };
    let cost = resources(spec).unwrap();
    let budget = Budget::new(cost.combined_reservation_bytes - 1).unwrap();
    assert!(matches!(
        issue_with(spec, context(), &[0], &budget, |_| panic!(
            "entropy before admission"
        )),
        Err(Error::Resource)
    ));
    assert_eq!(budget.used_bytes(), 0);
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for invalid in [
        Spec { bits: 2, ..spec },
        Spec { lanes: 0, ..spec },
        Spec { bits: 65, ..spec },
        Spec {
            lanes: MAX_LANES + 1,
            ..spec
        },
        Spec {
            lanes: usize::MAX,
            ..spec
        },
        Spec {
            gate: Gate::Tr { shift: 0 },
            ..spec
        },
        Spec {
            gate: Gate::GapArs { shift: 0 },
            ..spec
        },
        Spec {
            gate: Gate::Tr { shift: 32 },
            ..spec
        },
        Spec {
            gate: Gate::GapArs { shift: 31 },
            ..spec
        },
    ] {
        assert!(matches!(
            issue_with(invalid, context(), &[], &budget, |_| panic!(
                "entropy before validation"
            )),
            Err(Error::Domain)
        ));
    }
    assert!(matches!(
        issue_with(spec, context(), &[1u64 << 32], &budget, |_| panic!(
            "bad input mask"
        )),
        Err(Error::Domain)
    ));
    for bad_masks in [&[][..], &[0, 0][..]] {
        assert!(matches!(
            issue_with(spec, context(), bad_masks, &budget, |_| panic!(
                "bad input-mask length"
            )),
            Err(Error::Domain)
        ));
    }
    let mut invalid = context();
    invalid.output_wire = invalid.input_wire;
    assert!(matches!(
        issue_with(spec, invalid, &[0], &budget, |_| panic!("bad context")),
        Err(Error::Context)
    ));
    invalid = context();
    invalid.first_tensor_index = u64::MAX;
    assert!(matches!(
        issue_with(spec, invalid, &[0], &budget, |_| panic!("overflow")),
        Err(Error::Resource)
    ));
    for gate in [
        Gate::Tr { shift: 12 },
        Gate::GapArs { shift: 12 },
        Gate::Drelu,
    ] {
        let spec = Spec {
            gate,
            lanes: 3,
            ..spec
        };
        let per_lane_calls = match gate {
            Gate::Tr { .. } => 5,
            Gate::GapArs { .. } => 6,
            Gate::Drelu => 3,
        };
        let calls = 1 + spec.lanes * per_lane_calls;
        for fail in 0..=calls {
            let mut draw = 0;
            let before_issue = gpu127::measured_aes_blocks();
            let result = issue_with(spec, context(), &[1; 3], &budget, |bytes| {
                let current = draw;
                draw += 1;
                if current == fail {
                    return Err(Error::Randomness);
                }
                bytes.fill(0x57);
                Ok(())
            });
            if fail == calls {
                assert!(result.is_ok(), "{gate:?}: all RNG calls accounted for");
                assert_eq!(draw, calls);
            } else {
                assert!(
                    matches!(result, Err(Error::Randomness)),
                    "{gate:?} call={fail}"
                );
                assert_eq!(draw, fail + 1, "no RNG after failure");
            }
            let completed_lanes = if fail == calls {
                spec.lanes
            } else {
                fail.saturating_sub(1) / per_lane_calls
            };
            assert_eq!(
                gpu127::measured_aes_blocks() - before_issue,
                completed_lanes * resources(spec).unwrap().paired_issue_aes_blocks / spec.lanes,
                "no native key work after failed RNG request"
            );
            drop(result);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn allocation_failure_sweeps_burn_parties_without_refunding_retained_frames() {
    let spec = Spec {
        bits: 32,
        gate: Gate::GapArs { shift: 12 },
        lanes: 3,
    };
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut succeeded = false;
    for failure in 0..20 {
        budget.fail_after.store(failure, Ordering::SeqCst);
        match issue(spec, context(), &[1; 3], &budget) {
            Err(Error::Allocation) => {}
            Ok(value) => {
                drop(value);
                succeeded = true;
            }
            Err(error) => panic!("unexpected {error:?}"),
        }
        assert_eq!(budget.used_bytes(), 0);
        if succeeded {
            break;
        }
    }
    assert!(succeeded);
    budget.fail_after.store(-1, Ordering::SeqCst);
    let (masks, [mut a, mut b]) = issue(spec, context(), &[1; 3], &budget).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(a.begin(&[3; 3]), Err(Error::Allocation)));
    assert!(matches!(a.begin(&[3; 3]), Err(Error::Consumed)));
    budget.fail_after.store(-1, Ordering::SeqCst);
    let packet = b.begin(&[3; 3]).unwrap();
    b.cancel();
    drop((a, b, masks));
    assert_eq!(budget.used_bytes(), frame_charge(spec));
    drop(packet);
    assert_eq!(budget.used_bytes(), 0);

    let (masks, [mut a, mut b]) = issue(spec, context(), &[1; 3], &budget).unwrap();
    let x = a.begin(&[3; 3]).unwrap();
    let y = b.begin(&[3; 3]).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(a.advance(y.as_bytes()), Err(Error::Allocation)));
    assert!(matches!(a.advance(y.as_bytes()), Err(Error::Consumed)));
    drop((a, b, masks, x, y));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn role_context_nonce_input_phase_canonical_words_replay_and_cancel_burn() {
    let spec = Spec {
        bits: 32,
        gate: Gate::Tr { shift: 12 },
        lanes: 2,
    };
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for mutation in [0, 8, 40, 72, 73, 74, HEADER, usize::MAX] {
        let (masks, [mut a, mut b]) = issue(spec, context(), &[1; 2], &budget).unwrap();
        let x = a.begin(&[3; 2]).unwrap();
        let y = b.begin(&[3; 2]).unwrap();
        let mut corrupt = y.as_bytes().to_vec();
        if mutation == usize::MAX {
            corrupt.pop();
        } else {
            corrupt[mutation] ^= 2;
        }
        assert!(a.advance(&corrupt).is_err());
        assert!(matches!(a.advance(y.as_bytes()), Err(Error::Consumed)));
        b.cancel();
        assert!(matches!(b.advance(x.as_bytes()), Err(Error::Consumed)));
        drop((masks, a, b, x, y));
        assert_eq!(budget.used_bytes(), 0);
    }
    let (_, [mut a, mut b]) = issue(spec, context(), &[1; 2], &budget).unwrap();
    let (_, [_, mut other]) = issue(spec, context(), &[1; 2], &budget).unwrap();
    let x = a.begin(&[3; 2]).unwrap();
    let y = b.begin(&[3; 2]).unwrap();
    let foreign = other.begin(&[3; 2]).unwrap();
    assert_ne!(foreign.as_bytes(), y.as_bytes(), "fresh nonce/material");
    assert!(matches!(a.advance(foreign.as_bytes()), Err(Error::Context)));
    let step = b.advance(x.as_bytes()).unwrap();
    assert!(
        matches!(b.advance(x.as_bytes()), Err(Error::Context)),
        "carry replay in output phase"
    );
    drop(step);
    let (_, [mut a, mut b]) = issue(spec, context(), &[1; 2], &budget).unwrap();
    let _ = a.begin(&[3; 2]).unwrap();
    let y = b.begin(&[4; 2]).unwrap();
    assert!(matches!(a.advance(y.as_bytes()), Err(Error::Context)));
    let (_, [mut a, _]) = issue(spec, context(), &[1; 2], &budget).unwrap();
    assert!(matches!(a.begin(&[1u64 << 32; 2]), Err(Error::Domain)));
    assert!(matches!(a.begin(&[3; 2]), Err(Error::Consumed)));
}

#[test]
fn allocation_failpoints_every_gate_role_and_phase_exact_counts() {
    for gate in [
        Gate::Tr { shift: 2 },
        Gate::GapArs { shift: 2 },
        Gate::Drelu,
    ] {
        let spec = Spec {
            bits: 8,
            lanes: 3,
            gate,
        };
        let cost = resources(spec).unwrap();
        let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
        // One dealer vector and three vectors per party. Native DPF issuance
        // uses inline keys; no invisible eighth allocation is allowed.
        for allowed in 0..=7 {
            budget.fail_after.store(allowed, Ordering::SeqCst);
            let mut calls = 0;
            let before_issue = gpu127::measured_aes_blocks();
            let result = issue_with(spec, context(), &[17; 3], &budget, |bytes| {
                calls += 1;
                bytes.fill(0x65);
                Ok(())
            });
            if allowed < 7 {
                assert!(matches!(result, Err(Error::Allocation)));
                assert_eq!(calls, 0, "all allocation admission precedes RNG");
                assert_eq!(gpu127::measured_aes_blocks(), before_issue);
            } else {
                assert!(result.is_ok());
                assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
                assert_eq!(
                    gpu127::measured_aes_blocks() - before_issue,
                    cost.paired_issue_aes_blocks
                );
            }
            drop(result);
            assert_eq!(budget.used_bytes(), 0);
        }
        for role in 0..2 {
            for phase in 0..cost.frames_per_party + 1 {
                for allowed in 0..=1 {
                    budget.fail_after.store(-1, Ordering::SeqCst);
                    let (masks, mut parties) = issue(spec, context(), &[17; 3], &budget).unwrap();
                    let mut retained = Vec::new();
                    if phase > 0 {
                        retained.push(parties[0].begin(&[42; 3]).unwrap());
                        retained.push(parties[1].begin(&[42; 3]).unwrap());
                    }
                    if phase > 1 {
                        let a = frame(parties[0].advance(retained[1].as_bytes()).unwrap());
                        let b = frame(parties[1].advance(retained[0].as_bytes()).unwrap());
                        retained.extend([a, b]);
                    }
                    budget.fail_after.store(allowed, Ordering::SeqCst);
                    let before_call = gpu127::measured_aes_blocks();
                    let result = if phase == 0 {
                        parties[role].begin(&[42; 3]).map(Step::Frame)
                    } else {
                        let peer = &retained[2 * (phase - 1) + 1 - role];
                        parties[role].advance(peer.as_bytes())
                    };
                    assert_eq!(
                        gpu127::measured_aes_blocks() - before_call,
                        if phase == 0 {
                            cost.evaluation_aes_blocks_per_party
                        } else {
                            0
                        }
                    );
                    // Output moves its preallocated lane vector; it must work
                    // even when the next allocation is forced to fail.
                    let allocations = usize::from(phase < cost.frames_per_party);
                    if allowed == 0 && allocations == 1 {
                        assert!(
                            matches!(result, Err(Error::Allocation)),
                            "{gate:?} role={role} phase={phase}"
                        );
                    } else {
                        assert!(result.is_ok(), "{gate:?} role={role} phase={phase}");
                    }
                    let consumed = result.is_err() || phase == cost.frames_per_party;
                    assert_eq!(
                        budget.fail_after.load(Ordering::SeqCst),
                        allowed - if allowed > 0 { allocations as isize } else { 0 }
                    );
                    drop(result);
                    if consumed {
                        assert!(matches!(parties[role].advance(&[]), Err(Error::Consumed)));
                        assert!(matches!(
                            parties[role].begin(&[42; 3]),
                            Err(Error::Consumed)
                        ));
                    }
                    // Success at begin/carry leaves a live next phase. Cancel
                    // explicitly; failed calls and completed output stay burnt.
                    parties[role].cancel();
                    assert!(matches!(
                        parties[role].begin(&[42; 3]),
                        Err(Error::Consumed)
                    ));
                    assert!(matches!(parties[role].advance(&[]), Err(Error::Consumed)));
                    drop((parties, masks));
                    assert_eq!(budget.used_bytes(), retained.len() * frame_charge(spec));
                    drop(retained);
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}

#[test]
fn cancel_and_drop_at_every_phase_preserve_exact_exported_leases() {
    for gate in [
        Gate::Tr { shift: 2 },
        Gate::GapArs { shift: 2 },
        Gate::Drelu,
    ] {
        let spec = Spec {
            bits: 8,
            lanes: 3,
            gate,
        };
        let cost = resources(spec).unwrap();
        let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
        for phase in 0..=cost.frames_per_party {
            let (masks, [mut a, mut b]) = issue(spec, context(), &[17; 3], &budget).unwrap();
            let mut retained = Vec::new();
            if phase > 0 {
                retained.push(a.begin(&[42; 3]).unwrap());
                retained.push(b.begin(&[42; 3]).unwrap());
            }
            if phase > 1 {
                let x = frame(a.advance(retained[1].as_bytes()).unwrap());
                let y = frame(b.advance(retained[0].as_bytes()).unwrap());
                retained.extend([x, y]);
            }
            a.cancel();
            a.cancel();
            b.cancel();
            b.cancel();
            assert!(matches!(a.begin(&[42; 3]), Err(Error::Consumed)));
            assert!(matches!(b.advance(&[]), Err(Error::Consumed)));
            assert_eq!(
                budget.used_bytes(),
                cost.combined_reservation_bytes,
                "cancel erases state; leases remain owned until drop"
            );
            drop((a, b));
            assert_eq!(
                budget.used_bytes(),
                cost.dealer_reservation_bytes + retained.len() * frame_charge(spec)
            );
            assert_eq!(masks.values().len(), spec.lanes);
            drop(masks);
            assert_eq!(budget.used_bytes(), retained.len() * frame_charge(spec));
            while let Some(packet) = retained.pop() {
                assert_eq!(packet.as_bytes().len(), cost.frame_bytes);
                drop(packet);
                assert_eq!(budget.used_bytes(), retained.len() * frame_charge(spec));
            }
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

fn phase_frames<'a>(
    spec: Spec,
    parties: &mut [Party<'a>; 2],
    output_phase: bool,
) -> [Frame<'a>; 2] {
    let [a, b] = parties;
    let x = a.begin(&vec![42; spec.lanes]).unwrap();
    let y = b.begin(&vec![42; spec.lanes]).unwrap();
    if output_phase && spec.gate != Gate::Drelu {
        [
            frame(a.advance(y.as_bytes()).unwrap()),
            frame(b.advance(x.as_bytes()).unwrap()),
        ]
    } else {
        [x, y]
    }
}

#[test]
fn malformed_frames_both_roles_every_phase_and_canonical_output_width() {
    for gate in [
        Gate::Tr { shift: 2 },
        Gate::GapArs { shift: 2 },
        Gate::Drelu,
    ] {
        let spec = Spec {
            bits: 8,
            lanes: 2,
            gate,
        };
        let cost = resources(spec).unwrap();
        let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
        for role in 0..2 {
            for output_phase in [false, true] {
                // DReLU has only its output phase.
                if gate == Gate::Drelu && !output_phase {
                    continue;
                }
                for mutation in 0..15 {
                    let (masks, mut parties) = issue(spec, context(), &[17; 2], &budget).unwrap();
                    let frames = phase_frames(spec, &mut parties, output_phase);
                    let mut peer = frames[1 - role].as_bytes().to_vec();
                    let mut expected = Error::Context;
                    match mutation {
                        0 => peer[0] ^= 1,
                        1 => peer[8] ^= 1,  // Full context / nonce binding.
                        2 => peer[40] ^= 1, // Complete masked-input digest.
                        3 => peer[72] = role as u8, // Same-role reflection.
                        4 => peer[72] = 2,
                        5 => peer[73] ^= 1,
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
                            let bits = if output_phase { cost.output_bits } else { 1 };
                            let lane = mutation - 13;
                            peer[HEADER + lane * 8..HEADER + (lane + 1) * 8]
                                .copy_from_slice(&(1u64 << bits).to_le_bytes());
                            expected = Error::Domain;
                        }
                        _ => unreachable!(),
                    }
                    assert!(
                        matches!(parties[role].advance(&peer), Err(e) if e == expected),
                        "{gate:?} role={role} output={output_phase} mutation={mutation}"
                    );
                    assert!(matches!(
                        parties[role].advance(frames[1 - role].as_bytes()),
                        Err(Error::Consumed)
                    ));
                    assert!(matches!(
                        parties[role].begin(&[42; 2]),
                        Err(Error::Consumed)
                    ));
                    drop((parties, masks, frames));
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}

#[test]
fn each_public_context_field_nonce_gate_and_width_bind_independently() {
    let spec = Spec {
        bits: 8,
        lanes: 2,
        gate: Gate::Tr { shift: 2 },
    };
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for mutation in 0..11 {
        let mut other_context = context();
        let mut other_spec = spec;
        match mutation {
            0 => other_context.plan[0] ^= 1,
            1 => other_context.session[0] ^= 1,
            2 => other_context.operation[0] ^= 1,
            3 => other_context.input_wire[0] ^= 1,
            4 => other_context.output_wire[0] ^= 1,
            5 => other_context.first_tensor_index += 1,
            6 => {} // Nonce only; all subsequent RNG requests stay identical.
            7 => other_spec.bits = 9,
            8 => other_spec.gate = Gate::Tr { shift: 3 },
            9 => other_spec.gate = Gate::GapArs { shift: 2 },
            10 => other_spec.lanes = 3,
            _ => unreachable!(),
        }
        let (masks, mut parties) = issue_with(spec, context(), &[17; 2], &budget, |bytes| {
            bytes.fill(0x65);
            Ok(())
        })
        .unwrap();
        let mut calls = 0;
        let (other_masks, mut other) = issue_with(
            other_spec,
            other_context,
            &vec![17; other_spec.lanes],
            &budget,
            |bytes| {
                bytes.fill(if mutation == 6 && calls == 0 {
                    0x66
                } else {
                    0x65
                });
                calls += 1;
                Ok(())
            },
        )
        .unwrap();
        let local = phase_frames(spec, &mut parties, false);
        let foreign = phase_frames(other_spec, &mut other, false);
        assert_ne!(&local[1].as_bytes()[8..40], &foreign[1].as_bytes()[8..40]);
        assert!(
            matches!(
                parties[0].advance(foreign[1].as_bytes()),
                Err(Error::Context)
            ),
            "mutation={mutation}"
        );
        assert!(matches!(
            parties[0].advance(local[1].as_bytes()),
            Err(Error::Consumed)
        ));
        drop((masks, parties, other_masks, other, local, foreign));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn drelu_native_output_masks_cover_both_bits_for_each_clear_sign() {
    // Deterministically enumerate the actual native mask draws. Distribution
    // regression only: neither a randomness-quality nor a security proof.
    let spec = Spec {
        bits: 8,
        lanes: 1,
        gate: Gate::Drelu,
    };
    let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
    for clear in [0u64, 1, 127, 128, 255] {
        let mut counts = [0; 2];
        for desired in 0..=1u64 {
            let mut calls = 0;
            let (masks, mut parties) = issue_with(spec, context(), &[173], &budget, |bytes| {
                bytes.fill(0x65);
                if calls == 1 {
                    bytes.copy_from_slice(&desired.to_le_bytes());
                }
                calls += 1;
                Ok(())
            })
            .unwrap();
            assert_eq!(calls, 4, "nonce, mask bit, split, paired roots");
            assert_eq!(masks.values(), &[desired]);
            let outputs = execute(spec, &mut parties, &[(clear + 173) & 255]).unwrap();
            assert_eq!(outputs[0].values(), outputs[1].values());
            let value = outputs[0].values()[0];
            assert_eq!(value ^ desired, u64::from((clear as i8) >= 0));
            counts[value as usize] += 1;
            drop((masks, parties, outputs));
            assert_eq!(budget.used_bytes(), 0);
        }
        assert_eq!(counts, [1, 1]);
    }
}

#[test]
fn shared_rescale_matches_legacy_shift_and_independent_signed_floor() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for gap in [false, true] {
        for bits in [8, 64] {
            for shift in [1, bits - 2] {
                let gate = if gap {
                    Gate::GapArs { shift }
                } else {
                    Gate::Tr { shift }
                };
                let spec = Spec {
                    bits,
                    lanes: 1,
                    gate,
                };
                let quarter = 1u64 << (bits - 2);
                let inputs = if gap {
                    [0, 1, quarter - 1, mask(bits), mask(bits) - quarter + 1]
                } else {
                    [0, 1, quarter, 2 * quarter, mask(bits)]
                };
                for rin in [0, 1, quarter, mask(bits)] {
                    for clear in inputs {
                        let (masks, mut parties) = issue(spec, context(), &[rin], &budget).unwrap();
                        let rout = masks.values()[0];
                        let keys = super::super::Shift::issue(bits, shift, gap, rin, rout).unwrap();
                        let masked = clear.wrapping_add(rin) & mask(bits);
                        let opened = keys[0].start(0, masked) ^ keys[1].start(1, masked);
                        let legacy = keys[0]
                            .finish(0, masked, opened)
                            .wrapping_add(keys[1].finish(1, masked, opened));
                        let outputs = execute(spec, &mut parties, &[masked]).unwrap();
                        let ring = mask(resources(spec).unwrap().output_bits);
                        assert_eq!(outputs[0].values()[0], legacy & ring);
                        assert_eq!(legacy.wrapping_sub(rout) & ring, oracle(spec, clear));
                        drop((masks, parties, outputs));
                        assert_eq!(budget.used_bytes(), 0);
                    }
                }
            }
        }
    }
}

#[test]
fn wrong_input_lengths_lane_digest_order_phase_and_replay_burn() {
    for gate in [
        Gate::Tr { shift: 2 },
        Gate::GapArs { shift: 2 },
        Gate::Drelu,
    ] {
        let spec = Spec {
            bits: 8,
            lanes: 2,
            gate,
        };
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        for role in 0..2 {
            for malformed in [&[][..], &[42][..], &[42, 42, 42][..], &[42, 256][..]] {
                let (masks, mut parties) = issue(spec, context(), &[17; 2], &budget).unwrap();
                assert!(matches!(parties[role].begin(malformed), Err(Error::Domain)));
                assert!(matches!(
                    parties[role].begin(&[42; 2]),
                    Err(Error::Consumed)
                ));
                assert!(matches!(parties[role].advance(&[]), Err(Error::Consumed)));
                drop((masks, parties));
                assert_eq!(budget.used_bytes(), 0);
            }
            // Valid peer packet is still invalid before this role's begin.
            let (masks, mut parties) = issue(spec, context(), &[17; 2], &budget).unwrap();
            let peer = parties[1 - role].begin(&[42; 2]).unwrap();
            assert!(matches!(
                parties[role].advance(peer.as_bytes()),
                Err(Error::Context)
            ));
            assert!(matches!(
                parties[role].begin(&[42; 2]),
                Err(Error::Consumed)
            ));
            drop((masks, parties, peer));
            assert_eq!(budget.used_bytes(), 0);

            // Begin is single-use even before the peer's first message.
            let (masks, mut parties) = issue(spec, context(), &[17; 2], &budget).unwrap();
            let peer = phase_frames(spec, &mut parties, false);
            assert!(matches!(parties[role].begin(&[42; 2]), Err(Error::Domain)));
            assert!(matches!(
                parties[role].advance(peer[1 - role].as_bytes()),
                Err(Error::Consumed)
            ));
            drop((masks, parties, peer));
            assert_eq!(budget.used_bytes(), 0);

            for peer_input in [[42, 44], [43, 45], [44, 43]] {
                let (masks, mut parties) = issue(spec, context(), &[17; 2], &budget).unwrap();
                let local = parties[role].begin(&[43, 44]).unwrap();
                // Change each lane independently, then swap lane order.
                let peer = parties[1 - role].begin(&peer_input).unwrap();
                assert!(matches!(
                    parties[role].advance(peer.as_bytes()),
                    Err(Error::Context)
                ));
                assert!(matches!(
                    parties[role].advance(peer.as_bytes()),
                    Err(Error::Consumed)
                ));
                drop((masks, parties, local, peer));
                assert_eq!(budget.used_bytes(), 0);
            }

            let (masks, mut parties) = issue(spec, context(), &[17; 2], &budget).unwrap();
            let peer = phase_frames(spec, &mut parties, false);
            let next = parties[role].advance(peer[1 - role].as_bytes()).unwrap();
            let replay = parties[role].advance(peer[1 - role].as_bytes());
            if gate == Gate::Drelu {
                assert!(matches!(replay, Err(Error::Consumed)));
            } else {
                assert!(
                    matches!(replay, Err(Error::Context)),
                    "carry replay in output phase"
                );
                assert!(matches!(
                    parties[role].advance(peer[1 - role].as_bytes()),
                    Err(Error::Consumed)
                ));
            }
            drop((masks, parties, peer, next));
            assert_eq!(budget.used_bytes(), 0);

            let (masks, mut parties) = issue(spec, context(), &[17; 2], &budget).unwrap();
            let peer = phase_frames(spec, &mut parties, true);
            let done = parties[role].advance(peer[1 - role].as_bytes()).unwrap();
            assert!(matches!(
                parties[role].advance(peer[1 - role].as_bytes()),
                Err(Error::Consumed)
            ));
            drop((masks, parties, peer));
            assert_eq!(budget.used_bytes(), output_charge(spec));
            assert_eq!(output(done).values().len(), spec.lanes);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}
