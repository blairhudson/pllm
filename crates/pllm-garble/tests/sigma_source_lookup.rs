//! Independent public-boundary checks for the approved bounded source lookup.
//! Trusted local dealer fixtures only: no secrecy, authentication, malicious
//! security, upstream wire parity, or full activation-composition claim.

use pllm_garble::masked_wire_reference::source_lookup::{
    self as lookup, Boundary, Budget, Context, Error, Frame, Party, Spec, Step,
};

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
        lanes: 1,
        boundary,
    }
}

// All oracle arithmetic uses u128/i128, not production ring/sign helpers.
fn modulus(bits: u8) -> u128 {
    1u128 << bits
}

fn residue(value: i128, bits: u8) -> u64 {
    value.rem_euclid(modulus(bits) as i128) as u64
}

fn expected(table: &[u64], masked: u64, input_mask: u64, rho: u64, s: Spec) -> u64 {
    let index = (i128::from(masked) - i128::from(input_mask))
        .rem_euclid(modulus(s.index_bits) as i128) as usize;
    ((u128::from(table[index]) + u128::from(rho)) % modulus(s.output_bits)) as u64
}

fn word(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[test]
fn deferred_output_has_no_clone_capability() {
    // Compile-time negative assertion, without a subprocess/compiler fixture.
    // Adding Clone makes the marker ambiguous and fails this integration test
    // to compile; a dormant compile_fail doc block in tests/ would not run.
    trait AmbiguousIfClone<Marker> {
        fn check() {}
    }
    impl<T: ?Sized> AmbiguousIfClone<()> for T {}
    impl<T: ?Sized + Clone> AmbiguousIfClone<u8> for T {}
    let _ = <lookup::DeferredOutput<'static> as AmbiguousIfClone<_>>::check;
}

fn first_header(frame: &Frame<'_>, s: Spec, role: u8) {
    let bytes = frame.as_bytes();
    assert_eq!(bytes.len(), 80 + 16 * s.lanes);
    assert_eq!(&bytes[..8], b"PLLMLU01");
    assert_eq!(bytes[72], role);
    assert_eq!(bytes[73], 0);
    assert_eq!(&bytes[74..76], &(s.lanes as u16).to_le_bytes());
    assert_eq!(bytes[76], s.index_bits);
    assert_eq!(bytes[77], s.output_bits);
    assert_eq!(
        bytes[78],
        match s.boundary {
            Boundary::Reconstructed => 0,
            Boundary::Deferred => 1,
        }
    );
    assert_eq!(bytes[79], 0);
    for lane in 0..s.lanes {
        assert!(u128::from(word(bytes, 80 + 16 * lane)) < modulus(s.output_bits));
        assert!(word(bytes, 88 + 16 * lane) <= 1);
    }
}

fn second_header(frame: &Frame<'_>, s: Spec, role: u8) {
    let bytes = frame.as_bytes();
    assert_eq!(bytes.len(), 76 + 8 * s.lanes);
    assert_eq!(&bytes[..8], b"PLLMSL01");
    assert_eq!(bytes[72], role);
    assert_eq!(bytes[73], 1);
    assert_eq!(&bytes[74..76], &(s.lanes as u16).to_le_bytes());
    for lane in 0..s.lanes {
        assert!(u128::from(word(bytes, 76 + 8 * lane)) < modulus(s.output_bits));
    }
}

#[test]
fn exhaustive_compatible_counts_and_independent_select_algebra() {
    // A positive count c0 and negative count c1 come from N Boolean outputs.
    // Exhaust every compatible pair through the largest admitted domain.
    for n in 1..=10 {
        let count_bound = 1i128 << n;
        for sigma in [-1i128, 1] {
            for c0 in 0..=count_bound {
                let c1 = sigma - c0;
                if !(-count_bound..=0).contains(&c1) {
                    continue;
                }
                let h0 = (c0 + 1).rem_euclid(4) / 2;
                let h1 = (2 - (4 - c1.rem_euclid(4)) / 2).rem_euclid(2);
                assert_eq!(h0 ^ h1, (sigma + 1) / 2, "n={n}, {c0}, {c1}");
            }
        }
    }

    // Independently evaluate the five-word polynomial in unbounded-for-these-
    // inputs i128 arithmetic. Full-ring endpoints include m=64 without signed
    // overflow or shifting a u64 by 64. No native Select evaluator is called.
    for bits in [3, 64] {
        let m = modulus(bits) as i128;
        let endpoints = [0, 1, m / 2, m - 1];
        for sigma in [-1i128, 1] {
            let b = (sigma + 1) / 2;
            for q in [0i128, 1] {
                for r in endpoints {
                    for u in endpoints {
                        for t in endpoints {
                            let signed_value = (sigma * t).rem_euclid(m);
                            let opened_r = (signed_value + r).rem_euclid(m);
                            let opened_c = b ^ q;
                            // Correlations: q, r, q*r+u, 2q, -2q*r.
                            let selected = (q * r + u - q * opened_r - r * opened_c
                                + (1 - opened_c) * (2 * q * opened_r - 2 * q * r)
                                + opened_c * opened_r)
                                .rem_euclid(m);
                            assert_eq!(selected, (b * signed_value + u).rem_euclid(m));
                            let rho = (2 * u - r).rem_euclid(m);
                            assert_eq!(rho % 2, r % 2);
                            assert_eq!((2 * selected - opened_r).rem_euclid(m), (t + rho) % m);
                            // Arbitrary additive Select shares: only role zero
                            // subtracts common R. Both or neither is incorrect.
                            for z0 in endpoints {
                                let z1 = (selected - z0).rem_euclid(m);
                                let y0 = (2 * z0 - opened_r).rem_euclid(m);
                                let y1 = (2 * z1).rem_euclid(m);
                                assert_eq!((y0 + y1) % m, (t + rho) % m);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn endpoint_target_index_grid_matches_direct_table_oracle_at_both_boundaries() {
    let points = [0u64, 63, 64, 127, 128, 255, 256, 511, 512, 1023];
    for index_bits in [8, 10] {
        let n = 1usize << index_bits;
        let points: Vec<_> = points.iter().copied().filter(|&v| v < n as u64).collect();
        let mut masks = Vec::new();
        let mut masked = Vec::new();
        for &target in &points {
            for &index in &points {
                // Cartesian DPF-target/common-masked-index endpoints. Clear
                // table index is derived independently as (Xhat-rin) mod N.
                let rin = residue(-i128::from(target), index_bits);
                masks.push(rin);
                masked.push(index);
            }
        }
        for output_bits in [3, 64] {
            let max = (modulus(output_bits) - 1) as u64;
            for pattern in 0..5 {
                let table: Vec<_> = (0..n)
                    .map(|i| match pattern {
                        0 => 0,
                        1 => 3,
                        2 => (i as u128 % modulus(output_bits)) as u64,
                        3 => {
                            if i % 2 == 0 {
                                0
                            } else {
                                max
                            }
                        }
                        _ => max,
                    })
                    .collect();
                for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
                    // One issuance per entire Cartesian grid, not one per lane.
                    let s = Spec {
                        index_bits,
                        output_bits,
                        lanes: masks.len(),
                        boundary,
                    };
                    let cost = lookup::resources(s).unwrap();
                    let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
                    let (rho, [mut p0, mut p1]) =
                        lookup::issue(s, context(), &table, &masks, &budget).unwrap();
                    let f0 = p0.begin(&masked).unwrap();
                    let f1 = p1.begin(&masked).unwrap();
                    first_header(&f0, s, 0);
                    first_header(&f1, s, 1);
                    assert_eq!(&f0.as_bytes()[8..72], &f1.as_bytes()[8..72]);
                    let opened_r: Vec<_> = (0..s.lanes)
                        .map(|i| {
                            residue(
                                i128::from(word(f0.as_bytes(), 80 + 16 * i))
                                    + i128::from(word(f1.as_bytes(), 80 + 16 * i)),
                                output_bits,
                            )
                        })
                        .collect();
                    let expected: Vec<_> = (0..s.lanes)
                        .map(|i| expected(&table, masked[i], masks[i], rho.values()[i], s))
                        .collect();
                    let a = p0.advance(f1.as_bytes()).unwrap();
                    let b = p1.advance(f0.as_bytes()).unwrap();
                    match (boundary, a, b) {
                        (Boundary::Reconstructed, Step::Frame(g0), Step::Frame(g1)) => {
                            second_header(&g0, s, 0);
                            second_header(&g1, s, 1);
                            assert_eq!(cost.frames_per_party, 2);
                            assert_eq!(cost.total_peer_frame_bytes, 312 + 48 * s.lanes);
                            let Step::Reconstructed(y0) = p0.advance(g1.as_bytes()).unwrap() else {
                                panic!("reconstructed lookup must finish after two exchanges")
                            };
                            let Step::Reconstructed(y1) = p1.advance(g0.as_bytes()).unwrap() else {
                                panic!("peer must receive common masked output")
                            };
                            assert_eq!(y0.values(), expected);
                            assert_eq!(y1.values(), expected);
                            // The output equals 2*S-R directly from wire words.
                            for i in 0..s.lanes {
                                let selected = i128::from(word(g0.as_bytes(), 76 + 8 * i))
                                    + i128::from(word(g1.as_bytes(), 76 + 8 * i));
                                assert_eq!(
                                    y0.values()[i],
                                    residue(2 * selected - i128::from(opened_r[i]), output_bits)
                                );
                            }
                            assert!(matches!(p0.advance(g1.as_bytes()), Err(Error::Consumed)));
                            drop((p0, p1, rho));
                            assert!(budget.used_bytes() > 0);
                            drop((f0, f1, g0, g1));
                            assert!(
                                budget.used_bytes() > 0,
                                "outputs retain leases after all frames drop"
                            );
                            drop(y0);
                            assert!(budget.used_bytes() > 0, "peer output keeps its lease");
                            drop(y1);
                        }
                        (Boundary::Deferred, Step::Deferred(y0), Step::Deferred(y1)) => {
                            assert_eq!(cost.frames_per_party, 1);
                            assert_eq!(cost.total_peer_frame_bytes, 160 + 32 * s.lanes);
                            assert_eq!(y0.role(), 0);
                            assert_eq!(y1.role(), 1);
                            for i in 0..s.lanes {
                                let sum = i128::from(y0.values()[i]) + i128::from(y1.values()[i]);
                                assert_eq!(residue(sum, output_bits), expected[i]);
                                // Doubling makes role one's share even; role
                                // zero alone accounts for the parity of -R.
                                assert_eq!(y1.values()[i] % 2, 0);
                                assert_eq!(y0.values()[i] % 2, opened_r[i] % 2);
                            }
                            assert!(matches!(p0.advance(f1.as_bytes()), Err(Error::Consumed)));
                            drop((p0, p1, rho));
                            drop((f0, f1));
                            assert!(budget.used_bytes() > 0, "deferred outputs retain leases");
                            drop(y0);
                            assert!(budget.used_bytes() > 0);
                            drop(y1);
                        }
                        _ => panic!("output capability must match requested boundary"),
                    }
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}

fn assert_burned(party: &mut Party<'_>) {
    assert!(matches!(party.begin(&[0]), Err(Error::Consumed)));
    assert!(matches!(party.advance(&[]), Err(Error::Consumed)));
}

#[test]
fn malformed_first_frames_reject_and_burn_in_both_boundaries() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for mutation in 0..18 {
            let s = spec(boundary);
            let budget = Budget::new(1 << 20).unwrap();
            let (rho, [mut p0, mut p1]) =
                lookup::issue(s, context(), &[3; 256], &[0], &budget).unwrap();
            let own = p0.begin(&[0]).unwrap();
            let peer = p1.begin(&[0]).unwrap();
            let mut bad = peer.as_bytes().to_vec();
            match mutation {
                0 => {
                    bad.pop();
                }
                1 => bad.push(0),
                2 => bad[0] ^= 1,
                3 => bad[8] ^= 1,  // invocation binding
                4 => bad[40] ^= 1, // common input digest
                5 => bad[72] = 0,  // self role
                6 => bad[72] = 2,
                7 => bad[73] = 1,
                8 => bad[74] = 2,
                9 => bad[75] = 1,
                10 => bad[76] = 10,
                11 => bad[77] = 64,
                12 => bad[78] ^= 1,
                13 => bad[79] = 1,
                14 => bad[80..88].copy_from_slice(&8u64.to_le_bytes()),
                15 => bad[88..96].copy_from_slice(&2u64.to_le_bytes()),
                16 => bad[88..96].copy_from_slice(&u64::MAX.to_le_bytes()),
                _ => bad.clear(),
            }
            assert!(p0.advance(&bad).is_err(), "mutation={mutation}");
            assert_burned(&mut p0);
            p1.cancel();
            assert_burned(&mut p1);
            drop((p0, p1, rho));
            assert!(
                budget.used_bytes() > 0,
                "retained frames stay charged after failure"
            );
            drop((own, peer));
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn malformed_second_frames_and_first_phase_replay_burn() {
    for mutation in 0..10 {
        let s = spec(Boundary::Reconstructed);
        let budget = Budget::new(1 << 20).unwrap();
        let (rho, [mut p0, mut p1]) =
            lookup::issue(s, context(), &[3; 256], &[0], &budget).unwrap();
        let f0 = p0.begin(&[0]).unwrap();
        let f1 = p1.begin(&[0]).unwrap();
        let Step::Frame(g0) = p0.advance(f1.as_bytes()).unwrap() else {
            panic!("second frame")
        };
        let Step::Frame(g1) = p1.advance(f0.as_bytes()).unwrap() else {
            panic!("second frame")
        };
        let mut bad = g1.as_bytes().to_vec();
        match mutation {
            0 => bad = f1.as_bytes().to_vec(),
            1 => {
                bad.pop();
            }
            2 => bad.push(0),
            3 => bad[8] ^= 1,
            4 => bad[40] ^= 1,
            5 => bad[72] = 0,
            6 => bad[73] = 0,
            7 => bad[74] = 2,
            8 => bad[76..84].copy_from_slice(&8u64.to_le_bytes()),
            _ => bad[0] ^= 1,
        }
        assert!(p0.advance(&bad).is_err());
        assert_burned(&mut p0);
        p1.cancel();
        drop((p0, p1, rho, f0, f1, g0, g1));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn order_repeated_begin_cancel_and_bad_input_are_consuming() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for action in 0..7 {
            let s = spec(boundary);
            let budget = Budget::new(1 << 20).unwrap();
            let (rho, [mut p0, mut p1]) =
                lookup::issue(s, context(), &[0; 256], &[0], &budget).unwrap();
            let mut kept = None;
            match action {
                0 => assert!(p0.advance(&[]).is_err()),
                1 => {
                    p0.cancel();
                    p0.cancel();
                }
                2 => {
                    kept = Some(p0.begin(&[0]).unwrap());
                    assert!(p0.begin(&[0]).is_err());
                }
                3 => {
                    kept = Some(p0.begin(&[0]).unwrap());
                    p0.cancel();
                }
                4 => assert!(p0.begin(&[]).is_err()),
                5 => assert!(p0.begin(&[0, 0]).is_err()),
                _ => assert!(p0.begin(&[256]).is_err()),
            }
            assert_burned(&mut p0);
            p1.cancel();
            drop((p0, p1, rho));
            if kept.is_some() {
                assert!(budget.used_bytes() > 0);
            }
            drop(kept);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn different_common_inputs_and_other_issuances_are_rejected() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for changed in 0..14 {
            let s = spec(boundary);
            let budget = Budget::new(2 << 20).unwrap();
            let (rho, [mut p0, mut p1]) =
                lookup::issue(s, context(), &[3; 256], &[0], &budget).unwrap();
            let own = p0.begin(&[0]).unwrap();
            if changed == 0 {
                let peer = p1.begin(&[1]).unwrap();
                assert!(p0.advance(peer.as_bytes()).is_err());
                assert_burned(&mut p0);
                p1.cancel();
                drop((own, peer, p0, p1, rho));
            } else {
                let mut c = context();
                let mut table = [3; 256];
                let mut other_spec = s;
                match changed {
                    1 => c.plan[0] ^= 1,
                    2 => c.session[0] ^= 1,
                    3 => c.operation[0] ^= 1,
                    4 => c.input_wire[0] ^= 1,
                    5 => c.output_wire[0] ^= 1,
                    6 => c.arithmetic_wire[0] ^= 1,
                    7 => c.correction_wire[0] ^= 1,
                    8 => c.select_wire[0] ^= 1,
                    9 => c.numeric_contract[0] ^= 1,
                    10 => c.first_tensor_index += 1,
                    11 => table[255] = 4, // even an unselected entry is bound
                    12 => {
                        other_spec.boundary = match boundary {
                            Boundary::Reconstructed => Boundary::Deferred,
                            Boundary::Deferred => Boundary::Reconstructed,
                        }
                    }
                    _ => {} // same public context, distinct fresh issuance
                }
                let (rho2, [mut q0, mut q1]) =
                    lookup::issue(other_spec, c, &table, &[0], &budget).unwrap();
                let peer = q1.begin(&[0]).unwrap();
                // Fresh issuance also differs here: this is a substitution
                // rejection check, not isolation of each hash input's effect.
                assert!(p0.advance(peer.as_bytes()).is_err(), "changed={changed}");
                assert_burned(&mut p0);
                p1.cancel();
                q0.cancel();
                q1.cancel();
                drop((own, peer, p0, p1, q0, q1, rho, rho2));
            }
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn lane_order_is_part_of_common_input_identity() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        let s = Spec {
            lanes: 2,
            ..spec(boundary)
        };
        let budget = Budget::new(1 << 20).unwrap();
        let (rho, [mut p0, mut p1]) =
            lookup::issue(s, context(), &[3; 256], &[0, 0], &budget).unwrap();
        let own = p0.begin(&[0, 1]).unwrap();
        let peer = p1.begin(&[1, 0]).unwrap();
        assert!(p0.advance(peer.as_bytes()).is_err());
        assert_burned(&mut p0);
        p1.cancel();
        drop((rho, p0, p1, own, peer));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn retained_second_phase_frames_keep_budget_after_outputs_drop() {
    let s = spec(Boundary::Reconstructed);
    let budget = Budget::new(1 << 20).unwrap();
    let (rho, [mut p0, mut p1]) = lookup::issue(s, context(), &[3; 256], &[0], &budget).unwrap();
    let f0 = p0.begin(&[0]).unwrap();
    let f1 = p1.begin(&[0]).unwrap();
    let Step::Frame(g0) = p0.advance(f1.as_bytes()).unwrap() else {
        panic!("second frame")
    };
    let Step::Frame(g1) = p1.advance(f0.as_bytes()).unwrap() else {
        panic!("second frame")
    };
    let y0 = p0.advance(g1.as_bytes()).unwrap();
    let y1 = p1.advance(g0.as_bytes()).unwrap();
    drop((p0, p1, rho, f0, f1, y0, y1));
    assert!(budget.used_bytes() > 0);
    drop(g0);
    assert!(
        budget.used_bytes() > 0,
        "each exported Select frame owns a lease"
    );
    drop(g1);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn public_preflight_and_exact_aggregate_budget_boundaries() {
    for boundary in [Boundary::Reconstructed, Boundary::Deferred] {
        for (n, m, lanes) in [(8, 3, 1), (10, 64, 256)] {
            let s = Spec {
                index_bits: n,
                output_bits: m,
                lanes,
                boundary,
            };
            let cost = lookup::resources(s).unwrap();
            let table = vec![0; 1usize << n];
            let masks = vec![0; lanes];
            let short = Budget::new(cost.combined_reservation_bytes - 1).unwrap();
            assert!(matches!(
                lookup::issue(s, context(), &table, &masks, &short),
                Err(Error::Resource)
            ));
            assert_eq!(short.used_bytes(), 0);
            let exact = Budget::new(cost.combined_reservation_bytes).unwrap();
            let owners = lookup::issue(s, context(), &table, &masks, &exact).unwrap();
            assert_eq!(exact.used_bytes(), cost.combined_reservation_bytes);
            assert!(matches!(
                lookup::issue(s, context(), &table, &masks, &exact),
                Err(Error::Resource)
            ));
            assert_eq!(exact.used_bytes(), cost.combined_reservation_bytes);
            drop(owners);
            assert_eq!(exact.used_bytes(), 0);
        }

        let s = spec(boundary);
        let budget = Budget::new(1 << 20).unwrap();
        for bad in [
            Spec { index_bits: 7, ..s },
            Spec { index_bits: 9, ..s },
            Spec {
                index_bits: 11,
                ..s
            },
            Spec {
                output_bits: 2,
                ..s
            },
            Spec {
                output_bits: 65,
                ..s
            },
            Spec { lanes: 0, ..s },
            Spec { lanes: 257, ..s },
            Spec {
                lanes: usize::MAX,
                ..s
            },
        ] {
            assert!(lookup::resources(bad).is_err());
            assert!(lookup::issue(bad, context(), &[0; 256], &[0], &budget).is_err());
            assert_eq!(budget.used_bytes(), 0);
        }
        for length in [0, 255, 257] {
            assert!(lookup::issue(s, context(), &vec![0; length], &[0], &budget).is_err());
            assert_eq!(budget.used_bytes(), 0);
        }
        for input in [vec![], vec![0, 0], vec![256], vec![u64::MAX]] {
            assert!(lookup::issue(s, context(), &[0; 256], &input, &budget).is_err());
            assert_eq!(budget.used_bytes(), 0);
        }
        for index in [0, 255] {
            let mut table = [0; 256];
            table[index] = 8;
            assert!(lookup::issue(s, context(), &table, &[0], &budget).is_err());
            assert_eq!(budget.used_bytes(), 0);
        }
        let mut overflow = context();
        overflow.first_tensor_index = u64::MAX;
        assert!(lookup::issue(s, overflow, &[0; 256], &[0], &budget).is_err());
        assert_eq!(budget.used_bytes(), 0);
    }
}
