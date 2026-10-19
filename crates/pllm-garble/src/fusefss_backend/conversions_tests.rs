use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        first_tensor_index: 9,
    }
}
fn spec(input_bits: u8, output_bits: u8, operation: Operation) -> Spec {
    Spec {
        input_bits,
        output_bits,
        operation,
        lanes: 1,
        source_digest: [4; 32],
        profile_digest: [5; 32],
    }
}
fn signed(x: u64, bits: u8) -> i128 {
    if x & (1 << (bits - 1)) != 0 {
        i128::from(x) - (1i128 << bits)
    } else {
        i128::from(x)
    }
}
fn floor_oracle(x: u64, bits: u8, shift: u8, output: u8) -> u64 {
    let x = signed(x, bits);
    let divisor = 1i128 << shift;
    // Independent integer division (Rust division truncates toward zero).
    let q = x / divisor - i128::from(x < 0 && x % divisor != 0);
    (q & ((1i128 << output) - 1)) as u64
}
fn run(s: Spec, x: u64, share: u64) -> u64 {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let [mut a, mut b] = issue(s, context(), &budget).unwrap();
    let f = a.start(&[share]).unwrap();
    let g = b
        .start(&[x.wrapping_sub(share) & mask(s.input_bits)])
        .unwrap();
    let u = a.finish(g.as_bytes()).unwrap();
    let v = b.finish(f.as_bytes()).unwrap();
    let result = u.shares()[0].wrapping_add(v.shares()[0]) & mask(s.output_bits);
    assert!(a.is_consumed() && b.is_consumed());
    drop((a, b));
    assert!(budget.used_bytes() > 0);
    drop((f, g, u, v));
    assert_eq!(budget.used_bytes(), 0);
    result
}

#[test]
fn exhaustive_small_rings_all_shifts_and_extensions() {
    for bits in 1..=5 {
        let range = SignedRange::full_ring(bits).unwrap();
        for x in 0..=mask(bits) {
            for shift in 0..bits {
                for (output, op) in [
                    (bits - shift, Operation::Tr { shift, range }),
                    (bits, Operation::Ars { shift, range }),
                ] {
                    assert_eq!(
                        run(spec(bits, output, op), x, x.wrapping_mul(13) & mask(bits)),
                        floor_oracle(x, bits, shift, output),
                        "bits={bits} x={x} shift={shift}"
                    );
                }
            }
            assert_eq!(
                run(spec(bits, bits + 2, Operation::ZeroExtend), x, mask(bits)),
                x
            );
            assert_eq!(
                run(
                    spec(bits, bits + 2, Operation::SignExtend { range }),
                    x,
                    mask(bits)
                ),
                (signed(x, bits) & i128::from(mask(bits + 2))) as u64
            );
        }
    }
}

#[test]
fn exhaustive_every_mask_and_opened_input_dcf_decomposition() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for bits in 1..=4 {
        for shift in 0..bits {
            for op in [
                Operation::Tr {
                    shift,
                    range: SignedRange::full_ring(bits).unwrap(),
                },
                Operation::Ars {
                    shift,
                    range: SignedRange::full_ring(bits).unwrap(),
                },
                Operation::SignExtend {
                    range: SignedRange::full_ring(bits).unwrap(),
                },
                Operation::ZeroExtend,
            ] {
                let output = match op {
                    Operation::Tr { .. } => bits - shift,
                    Operation::Ars { .. } => bits,
                    _ => bits + 2,
                };
                let s = spec(bits, output, op);
                for r in 0..=mask(bits) {
                    let [a, b] = issue_inner(s, context(), &budget, || Ok(r)).unwrap();
                    let (Some(State::Ready(ka)), Some(State::Ready(kb))) = (&a.state, &b.state)
                    else {
                        panic!()
                    };
                    for u in 0..=mask(bits) {
                        let mut value = u >> s.shift();
                        for (id, lanes) in [(0, ka), (1, kb)] {
                            let Lane::Convert {
                                full, low, base, ..
                            } = &lanes[0]
                            else {
                                panic!()
                            };
                            value = value.wrapping_add(**base);
                            if let Some(k) = full {
                                value = value.wrapping_add(k.eval(id, u).unwrap()[0]);
                            }
                            if let Some(k) = low {
                                value =
                                    value.wrapping_add(k.eval(id, u & mask(s.shift())).unwrap()[0]);
                            }
                        }
                        let x = u.wrapping_sub(r) & mask(bits);
                        let expected = if matches!(op, Operation::ZeroExtend) {
                            x
                        } else {
                            floor_oracle(x, bits, s.shift(), output)
                        };
                        assert_eq!(
                            value & mask(output),
                            expected,
                            "bits={bits} r={r} u={u} op={op:?}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn ring50_to38_to50_and_full_width_wrap_carry() {
    let range = SignedRange::source_contract(50, 38, [8; 32]).unwrap();
    for raw in [
        -(1i128 << 37),
        -4097,
        -4096,
        -1,
        0,
        1,
        4095,
        4096,
        (1i128 << 37) - 1,
    ] {
        let x = (raw & i128::from(mask(50))) as u64;
        let tr = run(
            spec(50, 38, Operation::Tr { shift: 12, range }),
            x,
            mask(50),
        );
        let extended = run(
            spec(
                38,
                50,
                Operation::SignExtend {
                    range: SignedRange::full_ring(38).unwrap(),
                },
            ),
            tr,
            mask(38),
        );
        assert_eq!(extended, floor_oracle(x, 50, 12, 50));
    }
    for bits in [38, 50] {
        for x in [
            0,
            1,
            (1 << (bits - 1)) - 1,
            1 << (bits - 1),
            (1 << (bits - 1)) + 1,
            mask(bits) - 1,
            mask(bits),
        ] {
            for share in [0, 1, mask(bits), 1 << (bits - 1)] {
                for shift in [0, 1, 12, bits - 1] {
                    let s = spec(
                        bits,
                        bits,
                        Operation::Ars {
                            shift,
                            range: SignedRange::full_ring(bits).unwrap(),
                        },
                    );
                    assert_eq!(run(s, x, share), floor_oracle(x, bits, shift, bits));
                }
            }
        }
    }
}

#[test]
fn b2a_and_public_phi_both_branches_and_addition() {
    for bits in [1, 4, 38, 50] {
        for a in 0..=1 {
            for b in 0..=1 {
                assert_eq!(
                    run(spec(1, bits, Operation::BooleanToArithmetic), a ^ b, a),
                    a ^ b
                );
            }
        }
        for bit in [0u64, 1] {
            let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let s = spec(
                bits,
                bits,
                Operation::PhiSelect {
                    add: mask(bits),
                    bit_contract: [7; 32],
                },
            );
            let [mut a, mut b] = issue(s, context(), &budget).unwrap();
            let (yes, no) = (mask(bits), 1);
            let f = a.start_phi(&[mask(bits)], &[3 & mask(bits)], &[0]).unwrap();
            let g = b
                .start_phi(
                    &[bit.wrapping_sub(mask(bits)) & mask(bits)],
                    &[yes.wrapping_sub(3 & mask(bits)) & mask(bits)],
                    &[no],
                )
                .unwrap();
            let u = a.finish(g.as_bytes()).unwrap();
            let v = b.finish(f.as_bytes()).unwrap();
            assert_eq!(
                u.shares()[0].wrapping_add(v.shares()[0]) & mask(bits),
                (if bit == 1 { yes } else { no }).wrapping_add(mask(bits)) & mask(bits)
            );
        }
    }
}

#[test]
fn invalid_width_geometry_provenance_and_preflight_before_entropy() {
    assert!(SignedRange::full_ring(0).is_err());
    assert!(SignedRange::full_ring(64).is_err());
    assert!(SignedRange::source_contract(50, 51, [1; 32]).is_err());
    assert!(SignedRange::source_contract(50, 38, [0; 32]).is_err());
    let range = SignedRange::full_ring(50).unwrap();
    let valid = spec(50, 38, Operation::Tr { shift: 12, range });
    let mut cases = vec![
        spec(0, 38, Operation::ZeroExtend),
        spec(51, 51, Operation::ZeroExtend),
        spec(50, 50, Operation::Tr { shift: 12, range }),
        spec(50, 38, Operation::Tr { shift: 50, range }),
        spec(38, 50, Operation::SignExtend { range }),
        spec(2, 50, Operation::BooleanToArithmetic),
        spec(
            50,
            50,
            Operation::PhiSelect {
                add: 0,
                bit_contract: [0; 32],
            },
        ),
    ];
    let mut bad = valid;
    bad.lanes = MAX_LANES + 1;
    cases.push(bad);
    bad = valid;
    bad.source_digest = [0; 32];
    cases.push(bad);
    bad = valid;
    bad.profile_digest = [0; 32];
    cases.push(bad);
    let budget = Budget::new(1).unwrap();
    let entropy = AtomicUsize::new(0);
    for s in cases.into_iter().chain([valid]) {
        assert!(issue_inner(s, context(), &budget, || {
            entropy.fetch_add(1, Ordering::SeqCst);
            Ok(0)
        })
        .is_err());
        assert_eq!(budget.used_bytes(), 0);
    }
    assert_eq!(entropy.load(Ordering::SeqCst), 0);
}

#[test]
fn roles_context_source_profile_nonce_replay_order_cancel_fail_closed() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let s = spec(
        8,
        8,
        Operation::Ars {
            shift: 2,
            range: SignedRange::full_ring(8).unwrap(),
        },
    );
    for corruption in 0..8 {
        let [mut a, mut b] = issue(s, context(), &budget).unwrap();
        let f = a.start(&[12]).unwrap();
        let g = b.start(&[17]).unwrap();
        let mut bad = g.as_bytes().to_vec();
        match corruption {
            0 => bad[8] ^= 1,
            1 => bad[40] ^= 1,
            2 => bad[74] ^= 1,
            3 => bad[73] ^= 1,
            4 => bad[76] ^= 1,
            5 => {
                bad.pop();
            }
            6 => bad[HEADER_BYTES + 1] = 1,
            _ => bad = f.as_bytes().to_vec(),
        }
        assert!(a.finish(&bad).is_err());
        assert!(a.is_consumed());
        assert!(matches!(a.finish(g.as_bytes()), Err(Error::Consumed)));
        b.cancel();
        assert!(matches!(b.start(&[0]), Err(Error::Consumed)));
    }
    for change in 0..4 {
        let [mut a, _] = issue(s, context(), &budget).unwrap();
        let mut t = s;
        let mut c = context();
        match change {
            0 => c.operation[0] ^= 1,
            1 => t.source_digest[0] ^= 1,
            2 => t.profile_digest[0] ^= 1,
            _ => {}
        }
        let [_, mut d] = issue(t, c, &budget).unwrap();
        let _f = a.start(&[0]).unwrap();
        let g = d.start(&[0]).unwrap();
        assert!(matches!(a.finish(g.as_bytes()), Err(Error::Context)));
    }
    let [mut a, mut b] = issue(s, context(), &budget).unwrap();
    assert!(a.finish(&[]).is_err());
    assert!(a.is_consumed());
    assert!(b.start(&[256]).is_err());
    assert!(b.is_consumed());
    drop((a, b));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn every_allocation_failpoint_and_entropy_failure_refunds() {
    let s = spec(
        50,
        50,
        Operation::Ars {
            shift: 12,
            range: SignedRange::full_ring(50).unwrap(),
        },
    );
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for n in 0..7 {
        let mut draws = 0;
        assert!(matches!(
            issue_inner(s, context(), &budget, || {
                draws += 1;
                if draws > n {
                    Err(Error::Randomness)
                } else {
                    Ok(9)
                }
            }),
            Err(Error::Randomness)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
    // Issuance, DCF correction allocations, both starts and both output exports.
    let mut succeeded = false;
    for n in 0..40 {
        budget.fail_after.store(n, Ordering::SeqCst);
        let outcome = (|| -> Result<(), Error> {
            let [mut a, mut b] = issue(s, context(), &budget)?;
            let f = a.start(&[1])?;
            let g = b.start(&[2])?;
            let _u = a.finish(g.as_bytes())?;
            let _v = b.finish(f.as_bytes())?;
            Ok(())
        })();
        assert_eq!(budget.used_bytes(), 0, "failpoint {n}");
        match outcome {
            Ok(()) => {
                succeeded = true;
                break;
            }
            Err(e) => assert_eq!(e, Error::Allocation),
        }
    }
    assert!(succeeded);
}

#[test]
fn max_lanes_exact_resources_export_lease_and_replay() {
    let mut s = spec(
        50,
        50,
        Operation::PhiSelect {
            add: 0,
            bit_contract: [1; 32],
        },
    );
    s.lanes = MAX_LANES;
    let cost = resources(s).unwrap();
    assert_eq!(cost.triples, MAX_LANES);
    assert_eq!(cost.dcf_keys_per_party, 0);
    assert_eq!(cost.frame_bytes_per_party, HEADER_BYTES + MAX_LANES * 16);
    let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
    let [mut a, mut b] = issue(s, context(), &budget).unwrap();
    let f = a
        .start_phi(&[0; MAX_LANES], &[0; MAX_LANES], &[0; MAX_LANES])
        .unwrap();
    let g = b
        .start_phi(&[1; MAX_LANES], &[7; MAX_LANES], &[9; MAX_LANES])
        .unwrap();
    let u = a.finish(g.as_bytes()).unwrap();
    let v = b.finish(f.as_bytes()).unwrap();
    assert!(a.finish(g.as_bytes()).is_err());
    drop((a, b));
    assert_eq!(
        budget.used_bytes(),
        2 * (cost.frame_bytes_per_party + cost.output_bytes_per_party)
    );
    drop((f, g));
    assert_eq!(budget.used_bytes(), 2 * cost.output_bytes_per_party);
    drop((u, v));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn product_failpoints_bad_inputs_cancel_and_exported_frames() {
    for s in [
        spec(1, 50, Operation::BooleanToArithmetic),
        spec(
            50,
            50,
            Operation::PhiSelect {
                add: 7,
                bit_contract: [9; 32],
            },
        ),
    ] {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut succeeded = false;
        for n in 0..20 {
            budget.fail_after.store(n, Ordering::SeqCst);
            let outcome = (|| -> Result<(), Error> {
                let [mut a, mut b] = issue(s, context(), &budget)?;
                let (f, g) = if matches!(s.operation, Operation::BooleanToArithmetic) {
                    (a.start(&[1])?, b.start(&[0])?)
                } else {
                    (
                        a.start_phi(&[1], &[2], &[3])?,
                        b.start_phi(&[0], &[5], &[7])?,
                    )
                };
                let _u = a.finish(g.as_bytes())?;
                let _v = b.finish(f.as_bytes())?;
                Ok(())
            })();
            assert_eq!(budget.used_bytes(), 0);
            match outcome {
                Ok(()) => {
                    succeeded = true;
                    break;
                }
                Err(e) => assert_eq!(e, Error::Allocation),
            }
        }
        assert!(succeeded);
        budget.fail_after.store(-1, Ordering::SeqCst);
        let [mut a, mut b] = issue(s, context(), &budget).unwrap();
        let f = if matches!(s.operation, Operation::BooleanToArithmetic) {
            assert!(b.start(&[2]).is_err());
            a.start(&[1]).unwrap()
        } else {
            assert!(b.start_phi(&[0], &[0], &[]).is_err());
            a.start_phi(&[0], &[0], &[0]).unwrap()
        };
        assert!(b.is_consumed());
        // Same-role frame and out-of-order replay burn product owners too.
        assert!(a.finish(f.as_bytes()).is_err());
        assert!(a.is_consumed());
        a.cancel();
        drop((a, b));
        assert_eq!(budget.used_bytes(), f.as_bytes().len());
        drop(f);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn common_geometry_resource_ledger() {
    let range = SignedRange::full_ring(50).unwrap();
    for s in [
        spec(50, 38, Operation::Tr { shift: 12, range }),
        spec(50, 50, Operation::Ars { shift: 12, range }),
        spec(
            38,
            50,
            Operation::SignExtend {
                range: SignedRange::full_ring(38).unwrap(),
            },
        ),
        spec(38, 50, Operation::ZeroExtend),
        spec(1, 50, Operation::BooleanToArithmetic),
        spec(50, 50, Operation::Multiply),
        spec(
            50,
            50,
            Operation::PhiSelect {
                add: 0,
                bit_contract: [1; 32],
            },
        ),
    ] {
        let cost = resources(s).unwrap();
        println!("{:?}: {:?}", s.operation, cost);
        assert_eq!(cost.output_bytes_per_party, 8);
        assert!(cost.combined_reservation_bytes <= MAX_ALLOCATION_BYTES);
    }
}

fn run_product(bits: u8, x: u64, y: u64, x0: u64, y0: u64) -> u64 {
    let s = spec(bits, bits, Operation::Multiply);
    let cost = resources(s).unwrap();
    let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
    let [mut a, mut b] = issue(s, context(), &budget).unwrap();
    let f = a.start_product(&[x0], &[y0]).unwrap();
    let g = b
        .start_product(
            &[x.wrapping_sub(x0) & mask(bits)],
            &[y.wrapping_sub(y0) & mask(bits)],
        )
        .unwrap();
    let u = a.finish(g.as_bytes()).unwrap();
    let v = b.finish(f.as_bytes()).unwrap();
    let result = u.shares()[0].wrapping_add(v.shares()[0]) & mask(bits);
    assert!(a.is_consumed() && b.is_consumed());
    assert!(matches!(a.finish(g.as_bytes()), Err(Error::Consumed)));
    drop((a, b));
    assert_eq!(
        budget.used_bytes(),
        2 * (cost.frame_bytes_per_party + cost.output_bytes_per_party)
    );
    drop((f, g));
    assert_eq!(budget.used_bytes(), 2 * cost.output_bytes_per_party);
    drop((u, v));
    assert_eq!(budget.used_bytes(), 0);
    result
}

#[test]
fn multiply_exhaustive_small_and_wrapping_ring50_independent_oracle() {
    for bits in 1..=4 {
        for x in 0..=mask(bits) {
            for y in 0..=mask(bits) {
                assert_eq!(
                    run_product(
                        bits,
                        x,
                        y,
                        x.wrapping_mul(7) & mask(bits),
                        y.wrapping_add(3) & mask(bits)
                    ),
                    ((u128::from(x) * u128::from(y)) & u128::from(mask(bits))) as u64
                );
            }
        }
    }
    for bits in [38, 50] {
        for x in [
            0,
            1,
            mask(bits),
            mask(bits) - 1,
            1 << (bits - 1),
            (1 << (bits - 1)) - 1,
        ] {
            for y in [
                0,
                1,
                mask(bits),
                mask(bits) - 1,
                1 << (bits - 1),
                (1 << (bits - 1)) + 1,
            ] {
                for (x0, y0) in [(0, mask(bits)), (mask(bits), 0), (1 << (bits - 1), 1)] {
                    assert_eq!(
                        run_product(bits, x, y, x0, y0),
                        ((u128::from(x) * u128::from(y)) & u128::from(mask(bits))) as u64
                    );
                }
            }
        }
    }
}

#[test]
fn multiply_method_order_geometry_and_frame_fail_closed() {
    let s = spec(50, 50, Operation::Multiply);
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    assert!(resources(spec(38, 50, Operation::Multiply)).is_err());
    for method in 0..6 {
        let [mut a, _] = issue(s, context(), &budget).unwrap();
        let result = match method {
            0 => a.start(&[0]),
            1 => a.start_phi(&[0], &[0], &[0]),
            2 => a.start_product(&[], &[0]),
            3 => a.start_product(&[0], &[]),
            4 => a.start_product(&[1 << 50], &[0]),
            _ => a.start_product(&[0], &[1 << 50]),
        };
        assert!(result.is_err());
        assert!(a.is_consumed());
        assert!(matches!(a.start_product(&[0], &[0]), Err(Error::Consumed)));
    }
    for op in [
        Operation::BooleanToArithmetic,
        Operation::PhiSelect {
            add: 0,
            bit_contract: [1; 32],
        },
        Operation::ZeroExtend,
    ] {
        let input = if matches!(op, Operation::BooleanToArithmetic) {
            1
        } else {
            50
        };
        let [mut a, _] = issue(spec(input, 50, op), context(), &budget).unwrap();
        assert!(matches!(a.start_product(&[0], &[0]), Err(Error::Context)));
        assert!(a.is_consumed());
    }
    for mistake in 0..5 {
        let [mut a, mut b] = issue(s, context(), &budget).unwrap();
        let f = a.start_product(&[3], &[4]).unwrap();
        let g = b.start_product(&[5], &[6]).unwrap();
        match mistake {
            0 => {
                assert!(a.start_product(&[7], &[8]).is_err());
            }
            1 => {
                assert!(a.start(&[0]).is_err());
            }
            2 => {
                assert!(a.start_phi(&[0], &[0], &[0]).is_err());
            }
            3 => {
                assert!(a.finish(f.as_bytes()).is_err());
            }
            _ => {
                let mut bad = g.as_bytes().to_vec();
                bad[HEADER_BYTES + 7] = 255;
                assert!(a.finish(&bad).is_err());
            }
        }
        assert!(a.is_consumed());
        assert!(a.finish(g.as_bytes()).is_err());
        b.cancel();
        drop((a, b));
        assert_eq!(
            budget.used_bytes(),
            2 * resources(s).unwrap().frame_bytes_per_party
        );
        drop((f, g));
        assert_eq!(budget.used_bytes(), 0);
    }
    let [mut a, _] = issue(s, context(), &budget).unwrap();
    assert!(a.finish(&[]).is_err());
    assert!(a.is_consumed());
    drop(a);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn multiply_all_allocation_failpoints_and_failure_burns() {
    let mut s = spec(50, 50, Operation::Multiply);
    s.lanes = MAX_LANES;
    let cost = resources(s).unwrap();
    assert_eq!(cost.triples, MAX_LANES);
    assert_eq!(cost.private_material_bytes_per_party, 24 * MAX_LANES);
    assert_eq!(cost.private_material_bytes_total, 48 * MAX_LANES);
    let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
    let mut succeeded = false;
    for n in 0..20 {
        budget.fail_after.store(n, Ordering::SeqCst);
        let outcome = (|| -> Result<(), Error> {
            let [mut a, mut b] = issue(s, context(), &budget)?;
            let f = a.start_product(&[mask(50); MAX_LANES], &[1; MAX_LANES])?;
            let g = b.start_product(&[2; MAX_LANES], &[3; MAX_LANES])?;
            let u = a.finish(g.as_bytes())?;
            let v = b.finish(f.as_bytes())?;
            for i in 0..MAX_LANES {
                assert_eq!(u.shares()[i].wrapping_add(v.shares()[i]) & mask(50), 4);
            }
            Ok(())
        })();
        assert_eq!(budget.used_bytes(), 0, "failpoint {n}");
        match outcome {
            Ok(()) => {
                succeeded = true;
                break;
            }
            Err(e) => assert_eq!(e, Error::Allocation),
        }
    }
    assert!(succeeded);
    for phase in 0..2 {
        budget.fail_after.store(-1, Ordering::SeqCst);
        let [mut a, mut b] = issue(s, context(), &budget).unwrap();
        if phase == 0 {
            budget.fail_after.store(0, Ordering::SeqCst);
            assert!(matches!(
                a.start_product(&[0; MAX_LANES], &[0; MAX_LANES]),
                Err(Error::Allocation)
            ));
        } else {
            let f = a.start_product(&[0; MAX_LANES], &[0; MAX_LANES]).unwrap();
            let g = b.start_product(&[0; MAX_LANES], &[0; MAX_LANES]).unwrap();
            budget.fail_after.store(0, Ordering::SeqCst);
            assert!(matches!(a.finish(g.as_bytes()), Err(Error::Allocation)));
            drop((f, g));
        }
        assert!(a.is_consumed());
        drop((a, b));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn capacity_material_accounting_and_exact_reservation_admission() {
    for operation in [
        Operation::Tr {
            shift: 12,
            range: SignedRange::full_ring(50).unwrap(),
        },
        Operation::Ars {
            shift: 12,
            range: SignedRange::full_ring(50).unwrap(),
        },
        Operation::SignExtend {
            range: SignedRange::full_ring(38).unwrap(),
        },
        Operation::ZeroExtend,
        Operation::Multiply,
        Operation::BooleanToArithmetic,
        Operation::PhiSelect {
            add: 0,
            bit_contract: [1; 32],
        },
    ] {
        let input = match operation {
            Operation::BooleanToArithmetic => 1,
            Operation::SignExtend { .. } | Operation::ZeroExtend => 38,
            _ => 50,
        };
        let output = if matches!(operation, Operation::Tr { .. }) {
            38
        } else {
            50
        };
        let mut s = spec(input, output, operation);
        s.lanes = MAX_LANES;
        let cost = resources(s).unwrap();
        let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
        let [a, b] = issue(s, context(), &budget).unwrap();
        assert_eq!(budget.used_bytes(), 2 * cost.party_reservation_bytes);
        assert_eq!(
            cost.private_material_bytes_total,
            2 * cost.private_material_bytes_per_party
        );
        for party in [&a, &b] {
            assert_eq!(party.lease.bytes, cost.party_reservation_bytes);
            let Some(State::Ready(lanes)) = &party.state else {
                panic!()
            };
            assert_eq!(lanes.capacity(), MAX_LANES);
            let mut encoded = 0;
            for lane in lanes {
                if let Lane::Convert { full, low, .. } = lane {
                    for key in [full, low].into_iter().flatten() {
                        let allocation = key.allocation();
                        encoded += allocation.encoded_key_bytes;
                        assert!(allocation.key_capacity_bytes > 0);
                        assert!(
                            allocation.combined_allocation_bytes <= cost.combined_reservation_bytes
                        );
                    }
                }
            }
            assert_eq!(encoded, cost.dcf_encoded_bytes_per_party);
            assert_eq!(
                cost.private_material_bytes_per_party,
                encoded + MAX_LANES * if s.product() { ring::TRIPLE_BYTES } else { 16 }
            );
        }
        drop((a, b));
        assert_eq!(budget.used_bytes(), 0);
        let small = Budget::new(cost.combined_reservation_bytes - 1).unwrap();
        let mut entropy = false;
        assert!(matches!(
            issue_inner(s, context(), &small, || {
                entropy = true;
                Ok(0)
            }),
            Err(Error::Resource)
        ));
        assert!(!entropy);
        assert_eq!(small.used_bytes(), 0);
    }
}
