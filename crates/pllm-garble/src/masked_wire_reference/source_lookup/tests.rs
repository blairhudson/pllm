use super::*;
use std::sync::atomic::Ordering;

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
            assert!(size_of::<LookupLane>() >= size_of::<GpuKey>() + size_of::<[u64; 2]>());
            assert!(size_of::<LookupLane>() > compact_key_bytes + 16);
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
                    let party_inventory = [
                        size_of::<Party<'static>>(),     // inline execution and Select metadata
                        size_of::<LookupLane>() * lanes, // padded key + mask vector backing
                        8 * entries,                     // one immutable table backing per party
                        8 * lanes,                       // local arithmetic backing
                        8 * lanes,                       // local correction backing
                        8 * lanes,                       // borrowed masked indices
                        4 * allocator_allowance,         // lane/table/arithmetic/correction vectors
                        entries, // lane-serial full expansion backing, not L*N
                        size_of::<Zeroizing<Vec<u8>>>(), // scratch vector owner
                        allocator_allowance, // scratch allocation
                        first_payload, // borrowed first receive frame
                        first_payload + frame_owner + allocator_allowance, // retained first send owner
                        8 * lanes + output_owner + allocator_allowance, // final output; overlaps Select output
                        second_payload, // borrowed Select receive frame
                        if second_payload == 0 { 0 } else { frame_owner }, // retained Select frame wrapper
                        stack_allowance,
                    ];
                    let dealer_inventory = [
                        size_of::<OutputMasks<'static>>(),
                        8 * lanes,   // derived mask backing
                        8 * lanes,   // borrowed input masks
                        8 * entries, // borrowed table
                        8 * lanes,   // transient predicate masks for Select
                        8 * lanes,   // transient arithmetic masks for Select
                        3 * allocator_allowance,
                        2 * size_of::<GpuKey>(), // transient paired issuance owners
                        stack_allowance,
                    ];
                    let lookup_party: usize = party_inventory.iter().sum();
                    let lookup_dealer: usize = dealer_inventory.iter().sum();
                    // Independent compact cross-check; scratch-owner size stays
                    // ABI-derived rather than assuming a 24-byte Vec layout.
                    assert_eq!(
                        lookup_party,
                        size_of::<Party<'static>>()
                            + size_of::<LookupLane>() * lanes
                            + 9 * entries
                            + 64 * lanes
                            + frame_owner
                            + output_owner
                            + 8800
                            + size_of::<Zeroizing<Vec<u8>>>()
                            + if second_payload == 0 {
                                0
                            } else {
                                76 + 8 * lanes + frame_owner
                            }
                    );
                    assert_eq!(
                        lookup_dealer,
                        size_of::<OutputMasks<'static>>()
                            + 32 * lanes
                            + 8 * entries
                            + 2 * size_of::<GpuKey>()
                            + 8384
                    );
                    // Select has its own independently tested inventory. Add it
                    // once only; lookup must not replace inline keys by codecs.
                    let nested =
                        selection::resources_for(selection::Spec { bits: m, lanes }, boundary)
                            .unwrap();
                    let actual = resources(spec).unwrap();
                    let expected_party = lookup_party + nested.party_reservation_bytes;
                    let expected_dealer = lookup_dealer + nested.dealer_reservation_bytes;
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
            let r = a.state.as_ref().unwrap().lanes[i].masks[0]
                .wrapping_add(b.state.as_ref().unwrap().lanes[i].masks[0])
                & mask(m);
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
            .select
            .take()
            .unwrap()
            .into_deferred(&opened_c, &opened_r)
            .unwrap();
        let z1 = copy_b
            .select
            .take()
            .unwrap()
            .into_deferred(&opened_c, &opened_r)
            .unwrap();
        let ao = must_deferred(a.advance(bf.as_bytes()).unwrap());
        let bo = must_deferred(b.advance(af.as_bytes()).unwrap());
        for i in 0..2 {
            assert_eq!(
                ao.values()[i],
                z0.values()[i].wrapping_mul(2).wrapping_sub(opened_r[i]) & mask(m)
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
