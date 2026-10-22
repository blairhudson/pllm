use super::*;
use pllm_core::edge_contract::Shape;
use std::sync::atomic::Ordering;

fn source() -> SourceIdentity {
    SourceIdentity::new([11; 32], [12; 32], approved_implementation())
}
fn dealer(budget: &Budget) -> LocalDealer<'_> {
    LocalDealer::new(source(), [1; 32], [2; 32], budget).unwrap()
}
fn edge(wire: u8, lanes: usize) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [1; 32], [wire + 64; 32], [wire; 32], 19),
        Shape::new(&[lanes]).unwrap(),
        Numeric::new(50, 12, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        None,
    )
    .unwrap()
}
fn spec(kind: ActivationKind, lanes: usize) -> ActivationSpec {
    ActivationSpec::new(kind, [7; 32], edge(3, lanes), edge(4, lanes), [8; 32]).unwrap()
}
fn oracle(kind: ActivationKind, values: &[i64]) -> Vec<i128> {
    // Independent signed wide-integer floor/absolute/clamp. This does not use
    // native ring helpers or the region's clipping implementation.
    let k = 1i128 << kind.index_bits();
    values
        .iter()
        .map(|&x| {
            let y = i128::from(x).div_euclid(64);
            let index = y.abs().min(k - 1) as usize;
            (if y >= 0 { i128::from(x) } else { 0 }) - i128::from(kind.table()[index])
        })
        .collect()
}
fn unmask(mask: &DealerMask<'_, Ring>, output: &CommonMasked<'_, Ring>) -> Vec<i128> {
    assert_eq!(mask.binding, output.binding);
    output
        .backing
        .values()
        .iter()
        .zip(mask.backing.values())
        .map(|(&x, &r)| {
            let value = i128::from(x.wrapping_sub(r) & super::super::super::mask(50));
            if value >= 1i128 << 49 {
                value - (1i128 << 50)
            } else {
                value
            }
        })
        .collect()
}
fn next<'b>(
    parties: &mut [ActivationParty<'b>; 2],
    frames: &[ActivationFrame<'b>; 2],
) -> [ActivationFrame<'b>; 2] {
    let [a, b] = parties;
    match (
        a.advance(frames[1].as_bytes()).unwrap(),
        b.advance(frames[0].as_bytes()).unwrap(),
    ) {
        (ActivationStep::Frame(a), ActivationStep::Frame(b)) => [a, b],
        _ => panic!("unexpected final output"),
    }
}
fn begin<'b>(
    parties: &mut [ActivationParty<'b>; 2],
    inputs: [CommonMasked<'b, Ring>; 2],
) -> [ActivationFrame<'b>; 2] {
    let [x0, x1] = inputs;
    [parties[0].begin(x0).unwrap(), parties[1].begin(x1).unwrap()]
}
fn run<'b>(
    mut parties: [ActivationParty<'b>; 2],
    inputs: [CommonMasked<'b, Ring>; 2],
) -> [CommonMasked<'b, Ring>; 2] {
    let mut frames = begin(&mut parties, inputs);
    let mut total = 0;
    let n = parties[0].spec.input.shape().elements();
    let expected_magic: [&[u8]; 8] = [
        b"PLLMSA01",
        b"PLLMSA01",
        b"PLLMSA01",
        b"PLLMSL01",
        b"PLLMSA01",
        b"PLLMSL01",
        b"PLLMLU01",
        b"PLLMAC01",
    ];
    for phase in 0..8 {
        for frame in &frames {
            let bytes = frame.as_bytes();
            assert_eq!(&bytes[..8], expected_magic[phase]);
            assert_eq!(
                bytes.len(),
                if phase == 6 { 80 + 16 * n } else { 76 + 8 * n }
            );
            total += bytes.len();
        }
        if phase < 7 {
            frames = next(&mut parties, &frames);
        }
    }
    assert_eq!(total, 1224 + 144 * n);
    for (role, frame) in frames.iter().enumerate() {
        assert_eq!(frame.as_bytes()[72], role as u8);
        assert_eq!(frame.as_bytes()[73], 8);
    }
    let [a, b] = &mut parties;
    let outputs = match (
        a.advance(frames[1].as_bytes()).unwrap(),
        b.advance(frames[0].as_bytes()).unwrap(),
    ) {
        (ActivationStep::Output(a), ActivationStep::Output(b)) => [a, b],
        _ => panic!("missing final reconstruction"),
    };
    for party in &mut parties {
        assert_eq!(party.advance(&[]).err(), Some(Error::Consumed));
        assert!(party.x.is_none() && party.y.is_none() && party.d.is_none());
        assert!(party.local.is_empty());
    }
    outputs
}

#[test]
fn signed_endpoints_negative_bins_and_clip_edges_match_wide_oracle() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let k = 1i64 << kind.index_bits();
            let values = [
                -(1i64 << 37),
                -(1i64 << 37) + 1,
                -64 * (k + 1),
                -64 * k,
                -64 * (k - 1),
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
                64 * (k - 1),
                64 * k,
                64 * (k + 1),
                (1i64 << 37) - 1,
            ];
            let s = spec(kind, values.len());
            let (mask, parties, inputs) = d.activation_from_values(s, &values).unwrap();
            let outputs = run(parties, inputs);
            let expected = oracle(kind, &values);
            for output in &outputs {
                assert_eq!(output.descriptor(), s.output);
                assert_eq!(output.binding.consumer, [8; 32]);
                assert_eq!(unmask(&mask, output), expected);
            }
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn gelu_explicit_floor_examples_and_minimum_unsigned_absolute_residue() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let values = [-1, 63, 64, -(1i64 << 37)];
    let s = spec(ActivationKind::Gelu, values.len());
    let (mask, mut parties, inputs) = d.activation_from_values(s, &values).unwrap();
    let mut frames = begin(&mut parties, inputs);
    // Three deliveries reach prefix Select; fourth computes affine C=A-K.
    for _ in 0..4 {
        frames = next(&mut parties, &frames);
    }
    for party in &parties {
        assert!(party.phase == Phase::ClipDrelu);
        assert_eq!(
            party.wires[A].descriptor.numeric().interpretation(),
            Interpretation::Unsigned
        );
        assert_eq!(party.wires[A].interval.unwrap().upper, (1i128 << 31) + 1);
        assert!(party.x.is_some());
    }
    for _ in 0..3 {
        frames = next(&mut parties, &frames);
    }
    let [a, b] = &mut parties;
    let outputs = match (
        a.advance(frames[1].as_bytes()).unwrap(),
        b.advance(frames[0].as_bytes()).unwrap(),
    ) {
        (ActivationStep::Output(a), ActivationStep::Output(b)) => [a, b],
        _ => panic!("expected output"),
    };
    assert_eq!(&unmask(&mask, &outputs[0])[..3], &[-31, 63, 33]);
    assert_eq!(
        unmask(&mask, &outputs[1]),
        oracle(ActivationKind::Gelu, &values)
    );
}

#[test]
fn full_ring_forced_draws_preserve_affine_and_predicate_masks() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        let mut words = vec![0, 1];
        for width in [38, 32, 8, 10, 50] {
            words.extend([1u64 << (width - 1), (1u64 << width) - 1]);
        }
        for word in words {
            for bit in [0u8, 1] {
                let budget = Budget::new(64 << 20).unwrap();
                let mut d = dealer(&budget);
                let values = [-(1i64 << 37), -65, -1, 0, 63, 64, (1i64 << 37) - 1];
                let s = spec(kind, values.len());
                // u64 draws exercise full-ring masks; other root/nonce draws
                // remain deterministic test data. Alternate bit parity explicitly.
                let (mask, parties, inputs) = d
                    .activation_from_values_with(s, &values, |bytes| {
                        if bytes.len() == 8 {
                            bytes.copy_from_slice(&((word & !1) | u64::from(bit)).to_le_bytes());
                        } else {
                            bytes.fill(bit);
                        }
                        Ok(())
                    })
                    .unwrap();
                let outputs = run(parties, inputs);
                assert_eq!(unmask(&mask, &outputs[0]), oracle(kind, &values));
                assert_eq!(unmask(&mask, &outputs[1]), oracle(kind, &values));
            }
        }
    }
    assert_eq!(affine_mask(u32::MAX as u64, 1), u32::MAX as u64 - 2);
}

#[test]
fn complete_clear_preflight_is_zero_draw_and_zero_allocation() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    let s = spec(ActivationKind::Gelu, 1);
    budget.fail_after.store(0, Ordering::SeqCst);
    for values in [vec![], vec![1i64 << 37], vec![-(1i64 << 37) - 1]] {
        let mut draws = 0;
        let result = d.activation_from_values_with(s, &values, |_| {
            draws += 1;
            Ok(())
        });
        assert_eq!(result.err(), Some(Error::Domain));
        assert_eq!(draws, 0);
        assert_eq!(budget.used_bytes(), baseline);
    }
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    budget.fail_after.store(-1, Ordering::SeqCst);
}

#[test]
fn immutable_hashes_profiles_geometry_and_wire_alias_rejected() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        kind.validate_table().unwrap();
    }
    let s = spec(ActivationKind::Gelu, 1);
    let t = spec(ActivationKind::Silu, 1);
    assert_ne!(s.digest(), t.digest());
    assert_ne!(s.numeric_contract(), t.numeric_contract());
    let wires = s.wires().unwrap();
    for index in [L, F, O] {
        assert_eq!(wires[index].representation(), Representation::AdditiveShare);
    }
    assert!(ActivationSpec::new(s.kind, s.operation, s.input, s.input, s.next_consumer).is_err());
    assert!(
        ActivationSpec::new(s.kind, s.operation, s.input, edge(4, 2), s.next_consumer).is_err()
    );
    assert!(ActivationSpec::new(
        s.kind,
        s.operation,
        edge(3, 257),
        edge(4, 257),
        s.next_consumer
    )
    .is_err());
    let wrong = EdgeDescriptor::new(
        s.input.identity(),
        s.input.shape(),
        Numeric::new(50, 11, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        None,
    )
    .unwrap();
    assert!(ActivationSpec::new(s.kind, s.operation, wrong, s.output, s.next_consumer).is_err());
}

fn root_bytes(lanes: usize) -> usize {
    2 * (8 * lanes + ALLOCATOR + common_charge())
        + 8 * lanes
        + ALLOCATOR
        + mask_charge()
        + size_of::<RootInput<'static>>()
}
#[test]
fn exact_aggregate_budget_and_one_byte_short_for_one_and_256_lanes() {
    for lanes in [1, 256] {
        for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
            let s = spec(kind, lanes);
            let cost = activation_resources(s).unwrap();
            let dealer_bytes = size_of::<LocalDealer<'static>>() + (4 << 10);
            let exact = dealer_bytes + root_bytes(lanes) + cost.combined_reservation_bytes;
            let values = vec![0; lanes];
            let budget = Budget::new(exact).unwrap();
            {
                let mut d = dealer(&budget);
                let (mask, parties, inputs) = d.activation_from_values(s, &values).unwrap();
                let outputs = run(parties, inputs);
                assert_eq!(unmask(&mask, &outputs[0]), vec![0; lanes]);
            }
            assert_eq!(budget.used_bytes(), 0);
            let budget = Budget::new(exact - 1).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let mut draws = 0;
            assert_eq!(
                d.activation_from_values_with(s, &values, |_| {
                    draws += 1;
                    Ok(())
                })
                .err(),
                Some(Error::Resource)
            );
            assert_eq!(draws, 0);
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}

#[test]
fn wrong_external_role_binding_and_session_burn_local_party() {
    for fault in 0..3 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, mut parties, [mut x0, x1]) = d
            .activation_from_values(spec(ActivationKind::Gelu, 1), &[64])
            .unwrap();
        match fault {
            0 => x0.role = 1,
            1 => x0.binding.mask_id[0] ^= 1,
            _ => x0.binding.session[0] ^= 1,
        }
        assert_eq!(parties[0].begin(x0).err(), Some(Error::Context));
        assert_eq!(parties[0].advance(&[]).err(), Some(Error::Consumed));
        assert!(parties[0].x.is_none());
        parties[1].cancel();
        drop((x1, mask, parties));
    }
}

#[test]
fn ordered_slots_replay_and_failure_burn() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let (_, mut parties, inputs) = d
        .activation_from_values(spec(ActivationKind::Gelu, 1), &[1])
        .unwrap();
    let frames = begin(&mut parties, inputs);
    assert_eq!(
        parties[0].advance(frames[0].as_bytes()).err(),
        Some(Error::Context)
    );
    assert_eq!(
        parties[0].advance(frames[1].as_bytes()).err(),
        Some(Error::Consumed)
    );
    parties[1].cancel();
    let s = spec(ActivationKind::Gelu, 1);
    let (_, mut parties, _) = d.activation_from_values(s, &[1]).unwrap();
    assert_eq!(parties[0].burn(1), Err(Error::Consumed));
    assert_eq!(parties[0].burn(0), Err(Error::Consumed));
    parties[0].cancel();
    parties[1].cancel();
}

#[test]
fn final_codec_rejects_complete_header_payload_and_length_before_mutation() {
    // Magic, region, common input, role, phase, lanes, noncanonical LAST word,
    // truncation and suffix. Each fixture is fresh because failures burn state.
    for fault in 0..9 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (_, mut parties, inputs) = d
            .activation_from_values(spec(ActivationKind::Gelu, 2), &[-1, 64])
            .unwrap();
        let mut frames = begin(&mut parties, inputs);
        for _ in 0..7 {
            frames = next(&mut parties, &frames);
        }
        let mut bad = frames[1].as_bytes().to_vec();
        match fault {
            0 => bad[0] ^= 1,
            1 => bad[8] ^= 1,
            2 => bad[40] ^= 1,
            3 => bad[72] ^= 1,
            4 => bad[73] ^= 1,
            5 => bad[74] ^= 1,
            6 => *bad.last_mut().unwrap() |= 0x80,
            7 => {
                bad.pop();
            }
            _ => bad.push(0),
        }
        let before = parties[0].local.to_vec();
        assert_eq!(
            parties[0].validate_final(&bad),
            Err(if fault == 6 {
                Error::Domain
            } else {
                Error::Context
            })
        );
        assert_eq!(&*parties[0].local, &before);
        assert!(parties[0].advance(&bad).is_err());
        assert_eq!(
            parties[0].advance(frames[1].as_bytes()).err(),
            Some(Error::Consumed)
        );
        parties[1].cancel();
    }
}

#[test]
fn deferred_invocations_distinct_and_wrong_issuance_rejected() {
    for fault in [0, 1] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (_, mut parties, inputs) = d
            .activation_from_values(spec(ActivationKind::Silu, 1), &[-65])
            .unwrap();
        for party in &parties {
            assert_ne!(party.lookup_binding, party.select_binding);
        }
        let mut frames = begin(&mut parties, inputs);
        for _ in 0..6 {
            frames = next(&mut parties, &frames);
        }
        assert!(parties[0].x.is_some());
        if fault == 0 {
            parties[0].lookup_binding[0] ^= 1;
        } else {
            parties[0].select_binding[0] ^= 1;
        }
        assert_eq!(
            parties[0].advance(frames[1].as_bytes()).err(),
            Some(Error::Context)
        );
        assert_eq!(parties[0].advance(&[]).err(), Some(Error::Consumed));
        parties[1].cancel();
    }
}

#[test]
fn allocation_and_entropy_fault_sweeps_release_every_partial_owner() {
    for entropy in [false, true] {
        let mut reached_success = false;
        for fail in 0..512 {
            let budget = Budget::new(64 << 20).unwrap();
            {
                let mut d = dealer(&budget);
                let baseline = budget.used_bytes();
                if !entropy {
                    budget.fail_after.store(fail, Ordering::SeqCst);
                }
                let mut draws = 0;
                let result =
                    d.activation_from_values_with(spec(ActivationKind::Gelu, 1), &[-1], |bytes| {
                        if entropy && draws == fail {
                            return Err(Error::Randomness);
                        }
                        draws += 1;
                        bytes.fill(1);
                        Ok(())
                    });
                match result {
                    Ok((mask, mut parties, inputs)) => {
                        parties[0].cancel();
                        parties[1].cancel();
                        drop((parties, mask, inputs));
                        reached_success = true;
                    }
                    Err(error) => assert_eq!(
                        error,
                        if entropy {
                            Error::Randomness
                        } else {
                            Error::Allocation
                        }
                    ),
                }
                assert_eq!(budget.used_bytes(), baseline);
            }
            assert_eq!(budget.used_bytes(), 0);
            if reached_success {
                break;
            }
        }
        assert!(
            reached_success,
            "fault sweep did not reach complete issuance"
        );
    }
}

#[test]
fn retained_sends_outlive_cancelled_regions_without_early_refund() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    let (mask, mut parties, inputs) = d
        .activation_from_values(spec(ActivationKind::Gelu, 1), &[64])
        .unwrap();
    let mut frames = begin(&mut parties, inputs);
    let mut retained = Vec::new();
    for _ in 0..7 {
        let new = next(&mut parties, &frames);
        retained.push(frames);
        frames = new;
    }
    parties[0].cancel();
    parties[1].cancel();
    drop((mask, parties));
    assert!(budget.used_bytes() > baseline);
    // Frames own their original native/wrapper splits; cancellation cannot
    // refund or corrupt any retained frame backing.
    assert_eq!(&frames[0].as_bytes()[..8], MAGIC);
    drop(frames);
    retained.reverse();
    drop(retained);
    assert_eq!(budget.used_bytes(), baseline);
}

#[test]
fn retained_cancelled_and_completed_parties_keep_exact_metadata_until_each_drop() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for complete in [false, true] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let (mask, mut parties, inputs) =
                d.activation_from_values(spec(kind, 1), &[-65]).unwrap();
            if complete {
                let mut frames = begin(&mut parties, inputs);
                for _ in 0..7 {
                    frames = next(&mut parties, &frames);
                }
                let [a, b] = &mut parties;
                let outputs = match (
                    a.advance(frames[1].as_bytes()).unwrap(),
                    b.advance(frames[0].as_bytes()).unwrap(),
                ) {
                    (ActivationStep::Output(a), ActivationStep::Output(b)) => [a, b],
                    _ => panic!("expected completed parties"),
                };
                assert_eq!(unmask(&mask, &outputs[0]), oracle(kind, &[-65]));
                drop((outputs, frames));
            } else {
                // Exercise cancellation after the original input owners move in.
                let frames = begin(&mut parties, inputs);
                for party in &mut parties {
                    party.cancel();
                }
                drop(frames);
            }
            drop(mask);
            let metadata = size_of::<ActivationParty<'static>>();
            assert_eq!(budget.used_bytes(), baseline + 2 * metadata);
            for party in &mut parties {
                assert_eq!(party.lease.as_ref().unwrap().bytes, metadata);
                party.cancel();
                party.cancel();
                assert_eq!(party.lease.as_ref().unwrap().bytes, metadata);
                assert_eq!(party.advance(&[]).err(), Some(Error::Consumed));
                assert_eq!(budget.used_bytes(), baseline + 2 * metadata);
            }
            let [a, b] = parties;
            drop(a);
            assert_eq!(budget.used_bytes(), baseline + metadata);
            drop(b);
            assert_eq!(budget.used_bytes(), baseline);
            drop(d);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn imported_and_clear_issuance_measure_exact_requested_random_bytes() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for lanes in [1, 256] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let s = spec(kind, lanes);
            let cost = activation_resources(s).unwrap();
            let values = vec![0; lanes];
            let mut bytes_requested = 0;
            let clear = d
                .activation_from_values_with(s, &values, |bytes| {
                    bytes_requested += bytes.len();
                    bytes.fill(1);
                    Ok(())
                })
                .unwrap();
            assert_eq!(bytes_requested, cost.clear_input_random_bytes);
            drop(clear);
            assert_eq!(budget.used_bytes(), baseline);
            let (input_mask, inputs) = d
                .prepare_root(
                    s.input,
                    BoundRange::new(-(1i128 << 37), 1i128 << 37).unwrap(),
                    s.operation,
                )
                .unwrap()
                .encode(&values)
                .unwrap();
            let mut bytes_requested = 0;
            let imported = d
                .activation_with(s, input_mask, |bytes| {
                    bytes_requested += bytes.len();
                    bytes.fill(1);
                    Ok(())
                })
                .unwrap();
            assert_eq!(bytes_requested, cost.random_bytes);
            assert_eq!(cost.clear_input_random_bytes - bytes_requested, 8 * lanes);
            drop((imported, inputs));
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}

#[test]
fn imported_missing_or_wider_range_rejected_before_draws_or_admission() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for range in [
            None,
            Some(Interval {
                lower: -(1i128 << 37) - 1,
                upper: 1,
            }),
            Some(Interval {
                lower: 0,
                upper: (1i128 << 37) + 1,
            }),
        ] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let s = spec(kind, 1);
            let (mut input, inputs) = d
                .prepare_root(s.input, BoundRange::new(0, 1).unwrap(), s.operation)
                .unwrap()
                .encode(&[0])
                .unwrap();
            input.binding.interval = range;
            budget.fail_after.store(0, Ordering::SeqCst);
            let mut draws = 0;
            assert_eq!(
                d.activation_with(s, input, |_| {
                    draws += 1;
                    Ok(())
                })
                .err(),
                Some(Error::Domain)
            );
            assert_eq!(draws, 0);
            assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
            budget.fail_after.store(-1, Ordering::SeqCst);
            drop(inputs);
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}

#[test]
fn actual_deferred_select_owner_substitutions_burn_region() {
    for fault in 0..4 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, mut parties, inputs) = d
            .activation_from_values(spec(ActivationKind::Silu, 1), &[-65])
            .unwrap();
        // Wider donor accepts all canonical ring50 inputs, so rejection reaches
        // the deferred capability width check rather than input validation.
        let bits = if fault == 1 { 64 } else { 50 };
        let lanes = if fault == 2 { 2 } else { 1 };
        let context = selection::Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [91; 32],
            predicate_wire: [92; 32],
            input_wire: [93; 32],
            output_wire: [94; 32],
            first_tensor_index: 19,
        };
        let (donor_mask, [left, right]) = selection::issue_deferred(
            selection::Spec { bits, lanes },
            context,
            &vec![0; lanes],
            &vec![0; lanes],
            &budget,
        )
        .unwrap();
        let donor = if fault == 0 {
            drop(left);
            right
        } else {
            drop(right);
            left
        };
        // Match the expected binding for role/width/lane faults to isolate those
        // checks. Cross-invocation fault keeps the original expected binding.
        if fault != 3 {
            parties[0].select_binding = donor.internal_binding();
        }
        parties[0].final_select = Some(donor);
        let mut frames = begin(&mut parties, inputs);
        for _ in 0..6 {
            frames = next(&mut parties, &frames);
        }
        assert_eq!(
            parties[0].advance(frames[1].as_bytes()).err(),
            Some(if fault == 2 {
                Error::Domain
            } else {
                Error::Context
            })
        );
        assert_eq!(parties[0].advance(&[]).err(), Some(Error::Consumed));
        parties[1].cancel();
        drop((parties, frames, mask, donor_mask));
        drop(d);
        assert_eq!(budget.used_bytes(), 0);
    }
}

fn matrix_probe(budget: &Budget) -> Result<[matrix::party::Party<'_>; 2], Error> {
    matrix::party::issue_local(
        matrix::Spec {
            bits: 50,
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
fn original_matrix_guard_survives_projection_through_final_select_or_cancel() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    for cancel_after in [Some(0), Some(1), Some(4), Some(6), None] {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let edge = |wire, fractional, private| {
                EdgeDescriptor::new(
                    EdgeIdentity::new(source(), [1; 32], [wire; 32], [wire; 32], 19),
                    Shape::new(&[1, 1]).unwrap(),
                    Numeric::new(50, fractional, Interpretation::Signed).unwrap(),
                    Representation::CommonAdditiveMasked,
                    if private {
                        Some(WeightVisibility::Private)
                    } else {
                        None
                    },
                )
                .unwrap()
            };
            let s = ActivationSpec::new(
                ActivationKind::Gelu,
                [7; 32],
                edge(3, 12, false),
                edge(4, 12, false),
                [8; 32],
            )
            .unwrap();
            let (xm, [x0, x1]) = d
                .prepare_root(
                    edge(21, 6, false),
                    BoundRange::new(0, 65).unwrap(),
                    [20; 32],
                )
                .unwrap()
                .encode(&[64])
                .unwrap();
            let (wm, [w0, w1]) = d
                .prepare_root(edge(22, 6, true), BoundRange::new(0, 65).unwrap(), [20; 32])
                .unwrap()
                .encode(&[64])
                .unwrap();
            let (xm, [mut m0, mut m1]) = d
                .matrix(
                    [20; 32],
                    matrix::Spec {
                        bits: 50,
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
            let (mask, mut parties) = d.activation(s, xm).unwrap();
            let mut frames = begin(&mut parties, [x0, x1]);
            drop((ml, mr));
            let mut retained = Vec::new();
            for phase in 0..7 {
                // At phase six, lookup exchange is about to trigger final
                // Select against ORIGINAL ring50 X. Guard must still be held.
                assert_eq!(matrix_probe(&budget).err(), Some(Error::Resource));
                assert!(parties[0].x.is_some() && parties[1].x.is_some());
                if cancel_after == Some(phase) {
                    break;
                }
                let new = next(&mut parties, &frames);
                retained.push(frames);
                frames = new;
            }
            if cancel_after.is_none() {
                assert!(parties[0].phase == Phase::Final);
                assert!(parties[0].x.is_none() && parties[1].x.is_none());
                // Neither final output nor any retained activation frame
                // inherits/replaces the original matrix guard.
                drop(matrix_probe(&budget).unwrap());
                let [a, b] = &mut parties;
                let outputs = match (
                    a.advance(frames[1].as_bytes()).unwrap(),
                    b.advance(frames[0].as_bytes()).unwrap(),
                ) {
                    (ActivationStep::Output(a), ActivationStep::Output(b)) => [a, b],
                    _ => panic!("expected output"),
                };
                assert_eq!(
                    unmask(&mask, &outputs[0]),
                    oracle(ActivationKind::Gelu, &[4096])
                );
                drop(matrix_probe(&budget).unwrap());
            } else {
                parties[0].cancel();
                parties[1].cancel();
                drop(matrix_probe(&budget).unwrap());
            }
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn execution_allocation_faults_burn_and_refund_after_retained_frames_drop() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        let mut completed = false;
        for fail in 0..128 {
            let budget = Budget::new(64 << 20).unwrap();
            {
                let mut d = dealer(&budget);
                let baseline = budget.used_bytes();
                let (mask, mut parties, [x0, x1]) =
                    d.activation_from_values(spec(kind, 1), &[-65]).unwrap();
                budget.fail_after.store(fail, Ordering::SeqCst);
                let mut retained = Vec::new();
                let result = (|| -> Result<(), Error> {
                    let l = parties[0].begin(x0)?;
                    let r = parties[1].begin(x1)?;
                    let mut frames = [l, r];
                    for _ in 0..7 {
                        let a = parties[0].advance(frames[1].as_bytes())?;
                        let b = parties[1].advance(frames[0].as_bytes())?;
                        let new = match (a, b) {
                            (ActivationStep::Frame(a), ActivationStep::Frame(b)) => [a, b],
                            _ => return Err(Error::Context),
                        };
                        retained.push(frames);
                        frames = new;
                    }
                    let a = parties[0].advance(frames[1].as_bytes())?;
                    let b = parties[1].advance(frames[0].as_bytes())?;
                    assert!(matches!(a, ActivationStep::Output(_)));
                    assert!(matches!(b, ActivationStep::Output(_)));
                    Ok(())
                })();
                match result {
                    Ok(()) => completed = true,
                    Err(error) => assert_eq!(error, Error::Allocation),
                }
                parties[0].cancel();
                parties[1].cancel();
                drop((mask, parties, retained));
                assert_eq!(budget.used_bytes(), baseline);
            }
            assert_eq!(budget.used_bytes(), 0);
            if completed {
                break;
            }
        }
        assert!(completed);
    }
}

#[test]
fn silu_and_max_lane_imported_issuance_selected_faults_release_all_owners() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for lanes in [1, 256] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let s = spec(kind, lanes);
            let values = vec![0; lanes];
            let (input, inputs) = d
                .prepare_root(s.input, BoundRange::new(0, 1).unwrap(), s.operation)
                .unwrap()
                .encode(&values)
                .unwrap();
            budget.fail_after.store(1_000_000, Ordering::SeqCst);
            let mut draws = 0isize;
            let issued = d
                .activation_with(s, input, |bytes| {
                    draws += 1;
                    bytes.fill(1);
                    Ok(())
                })
                .unwrap();
            let allocations = 1_000_000 - budget.fail_after.load(Ordering::SeqCst);
            budget.fail_after.store(-1, Ordering::SeqCst);
            drop((issued, inputs));
            assert_eq!(budget.used_bytes(), baseline);
            assert!(draws > 2 && allocations > 2);
            for entropy in [false, true] {
                let count = if entropy { draws } else { allocations };
                // Bound max-lane key issuance work: beginning, midpoint, final
                // failure, and exact successful boundary, not an exhaustive sweep.
                for fail in [0, 1, count / 2, count - 1, count] {
                    let (input, inputs) = d
                        .prepare_root(s.input, BoundRange::new(0, 1).unwrap(), s.operation)
                        .unwrap()
                        .encode(&values)
                        .unwrap();
                    if !entropy {
                        budget.fail_after.store(fail, Ordering::SeqCst);
                    }
                    let mut calls = 0;
                    let result = d.activation_with(s, input, |bytes| {
                        if entropy && calls == fail {
                            return Err(Error::Randomness);
                        }
                        calls += 1;
                        bytes.fill(1);
                        Ok(())
                    });
                    budget.fail_after.store(-1, Ordering::SeqCst);
                    if fail == count {
                        drop(result.unwrap());
                    } else {
                        assert_eq!(
                            result.err(),
                            Some(if entropy {
                                Error::Randomness
                            } else {
                                Error::Allocation
                            })
                        );
                    }
                    drop(inputs);
                    assert_eq!(budget.used_bytes(), baseline);
                }
            }
            drop(d);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn max_lane_selected_transition_allocation_faults_burn_and_preserve_retained_sends() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for phase in [0, 3, 5, 6] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let (mask, mut parties, inputs) = d
                .activation_from_values(spec(kind, 256), &vec![-65; 256])
                .unwrap();
            let mut frames = begin(&mut parties, inputs);
            let mut retained = Vec::new();
            for _ in 0..phase {
                let new = next(&mut parties, &frames);
                retained.push(frames);
                frames = new;
            }
            budget.fail_after.store(0, Ordering::SeqCst);
            assert_eq!(
                parties[0].advance(frames[1].as_bytes()).err(),
                Some(Error::Allocation)
            );
            budget.fail_after.store(-1, Ordering::SeqCst);
            assert_eq!(parties[0].advance(&[]).err(), Some(Error::Consumed));
            parties[0].cancel();
            parties[1].cancel();
            drop((parties, mask));
            assert!(budget.used_bytes() > baseline);
            assert_eq!(
                frames[0].as_bytes().len(),
                if phase == 6 {
                    80 + 16 * 256
                } else {
                    76 + 8 * 256
                }
            );
            drop((frames, retained));
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}
