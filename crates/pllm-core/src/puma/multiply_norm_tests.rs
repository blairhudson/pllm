use super::*;
use crate::puma::multiply::tests::{finish_truncation, parties};

fn bounds(radius: i64) -> InputBounds {
    InputBounds {
        left: Interval::new(-radius, radius).unwrap(),
        right: Interval::new(-radius, radius).unwrap(),
    }
}

fn proof(inner: usize, radius: i64) -> RowEnergyProof {
    let energy = inner as u128 * (radius as u128).pow(2);
    RowEnergyProof::new(inner, bounds(radius), energy, energy).unwrap()
}

fn shares(plan: Plan, operand: Operand, values: &[i64]) -> [ReplicatedShare; 3] {
    let a: Vec<u64> = (0..values.len()).map(|i| u64::MAX - i as u64).collect();
    let b: Vec<u64> = (0..values.len())
        .map(|i| 0x123456789 + 17 * i as u64)
        .collect();
    let c: Vec<u64> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (v as u64).wrapping_sub(a[i]).wrapping_sub(b[i]))
        .collect();
    let parts = [&a, &b, &c];
    [Party::P0, Party::P1, Party::P2].map(|p| {
        plan.share_operand(
            operand,
            p,
            parts[p as usize],
            parts[p.next() as usize],
            Limits::default(),
        )
        .unwrap()
    })
}

fn messages(owners: &mut [PartyRuntime; 3], plan: Plan, x: &[i64], y: &[i64]) -> Vec<Message> {
    owners
        .iter_mut()
        .zip(shares(plan, Operand::Left, x))
        .zip(shares(plan, Operand::Right, y))
        .map(|((p, x), y)| p.begin(x, y).unwrap())
        .collect()
}

fn rotate(owners: &mut [PartyRuntime; 3], mut messages: Vec<Message>) -> Vec<Message> {
    for p in owners.iter_mut() {
        let i = messages
            .iter()
            .position(|m| m.header().to == p.party() && m.header().phase == MessagePhase::Multiply)
            .unwrap();
        let next = messages.swap_remove(i);
        messages.extend(p.rotate(next).unwrap().into_iter().flatten());
    }
    messages
}

#[test]
fn loose_energy_certificates_cannot_widen_admitted_scalar_bounds() {
    for b in [
        bounds(1024),
        InputBounds {
            left: Interval::new(2, 5).unwrap(),
            right: Interval::new(-7, -3).unwrap(),
        },
        InputBounds {
            left: Interval::new(2, 5).unwrap(),
            right: Interval::new(3, 7).unwrap(),
        },
    ] {
        let p = RowEnergyProof::new(128, b, 1 << 50, 1 << 50).unwrap();
        assert_eq!(p.raw_bounds(), b.dot_product(128).unwrap());
        assert_eq!(p.output_bounds().unwrap(), b.dot_output(128).unwrap());
    }
    // A loose energy hull outside the quarter-gap does not defeat a stronger
    // scalar proof. No random draw or source arithmetic is changed.
    let p = RowEnergyProof::new(128, bounds(1), 1 << 62, 1 << 62).unwrap();
    assert_eq!(p.raw_bounds(), Interval::new(-128, 128).unwrap());
    assert_eq!(p.output_bounds().unwrap(), Interval::new(-1, 1).unwrap());
}

#[test]
fn intersected_bounds_cover_every_admitted_small_two_term_dot() {
    let b = InputBounds {
        left: Interval::new(1, 3).unwrap(),
        right: Interval::new(-3, 2).unwrap(),
    };
    for left_energy in 2..=18 {
        for right_energy in 0..=18 {
            let p = RowEnergyProof::new(2, b, left_energy, right_energy).unwrap();
            let raw = p.raw_bounds();
            for a in 1i64..=3 {
                for c in 1i64..=3 {
                    if (a * a + c * c) as u128 > left_energy {
                        continue;
                    }
                    for d in -3i64..=2 {
                        for e in -3i64..=2 {
                            if (d * d + e * e) as u128 > right_energy {
                                continue;
                            }
                            let actual = a * d + c * e;
                            assert!((raw.lower()..=raw.upper()).contains(&actual));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn integer_square_root_and_overflow_boundaries() {
    for v in 0..10000_u128 {
        let r = ceil_sqrt(v);
        assert!(r * r >= v);
        assert!(r == 0 || (r - 1) * (r - 1) < v);
    }
    assert_eq!(ceil_sqrt(u128::MAX), 1_u128 << 64);
    assert_eq!(ceil_sqrt((1_u128 << 126) - 1), 1_u128 << 63);
    assert_eq!(ceil_sqrt(1_u128 << 126), 1_u128 << 63);
    assert_eq!(
        RowEnergyProof::new(1, bounds(1), u128::MAX, 2),
        Err(Error::Bounds)
    );
    assert_eq!(
        RowEnergyProof::new(1, bounds(1), 1_u128 << 63, 1_u128 << 63),
        Err(Error::Bounds)
    );
    assert_eq!(RowEnergyProof::new(0, bounds(1), 1, 1), Err(Error::Bounds));
    assert_eq!(
        RowEnergyProof::new(crate::puma::MAX_LANES + 1, bounds(1), 1, 1),
        Err(Error::Bounds)
    );
    let asymmetric = RowEnergyProof::new(1, bounds(3), 2, 3).unwrap();
    assert_eq!(asymmetric.raw_bounds(), Interval::new(-3, 3).unwrap());
    assert_eq!(asymmetric.squared_energy_maxima(), [2, 3]);
    assert_eq!(asymmetric.inner(), 1);
    assert_eq!(asymmetric.input_bounds(), bounds(3));
    assert_eq!(
        RowEnergyProof::new(1, bounds(i64::MAX), 0, u128::MAX)
            .unwrap()
            .raw_bounds(),
        Interval::new(0, 0).unwrap()
    );
}

#[test]
fn scalar_obligations_and_quarter_gap_remain_binding() {
    for interval in [Interval::new(2, 3).unwrap(), Interval::new(-3, -2).unwrap()] {
        let b = InputBounds {
            left: interval,
            right: Interval::new(0, 0).unwrap(),
        };
        assert_eq!(RowEnergyProof::new(3, b, 11, 0), Err(Error::Bounds));
        assert!(RowEnergyProof::new(3, b, 12, 0).is_ok());
        let b = InputBounds {
            left: b.right,
            right: b.left,
        };
        assert_eq!(RowEnergyProof::new(3, b, 0, 11), Err(Error::Bounds));
    }
    let radius = INPUT_MAX_EXCLUSIVE as u128;
    let p = RowEnergyProof::new(1, bounds(i64::MAX), radius, radius).unwrap();
    assert_eq!(p.output_bounds(), Err(Error::Bounds));
    let mut owners = parties(PrssLimits::default());
    assert_eq!(
        owners[0].plan_norm_matmul(MatrixShape::new(1, 1, 1).unwrap(), p),
        Err(Error::Bounds)
    );
    assert_eq!(owners[0].cursors(), Cursors::default());
    assert_eq!(owners[0].status(), Status::Retired);
    let mut owners = parties(PrssLimits::default());
    assert!(owners[0]
        .plan_ring_norm_matmul(MatrixShape::new(1, 1, 1).unwrap(), p)
        .is_ok());
}

#[test]
fn signed_dot_oracles_rounding_envelopes_and_unchanged_work() {
    for rescale in [false, true] {
        for [rows, inner, columns] in [[1, 1, 1], [2, 3, 4], [3, 7, 5], [1, 1024, 1]] {
            let mut norm = parties(PrssLimits::default());
            let mut boxed = parties(PrssLimits::default());
            let shape = MatrixShape::new(rows, inner, columns).unwrap();
            let p = proof(inner, (1 << 18) + 7);
            let x: Vec<_> = (0..shape.left_elements())
                .map(|i| ((i * 711337 + 31) % (1 << 19)) as i64 - (1 << 18))
                .collect();
            let y: Vec<_> = (0..shape.right_elements())
                .map(|i| ((i * 7717 + 7) % (1 << 19)) as i64 - (1 << 18))
                .collect();
            let expected: Vec<i128> = (0..rows)
                .flat_map(|r| {
                    let (x, y) = (&x, &y);
                    (0..columns).map(move |c| {
                        (0..inner)
                            .map(|k| i128::from(x[r * inner + k]) * i128::from(y[k * columns + c]))
                            .sum()
                    })
                })
                .collect();
            let before = norm.each_ref().map(|o| o.cursors());
            let plans = norm.each_mut().map(|o| {
                if rescale {
                    o.plan_norm_matmul(shape, p)
                } else {
                    o.plan_ring_norm_matmul(shape, p)
                }
                .unwrap()
            });
            assert_eq!(plans, [plans[0]; 3]);
            assert_eq!(norm.each_ref().map(|o| o.cursors()), before);
            let ordinary = boxed.each_mut().map(|o| {
                if rescale {
                    o.plan_matmul(shape, p.input_bounds())
                } else {
                    o.plan_ring_matmul(shape, p.input_bounds())
                }
                .unwrap()
            });
            assert_ne!(plans[0].digest(), ordinary[0].digest());
            assert_eq!(plans[0].admitted_bytes(), ordinary[0].admitted_bytes());
            let m = messages(&mut norm, plans[0], &x, &y);
            let b = messages(&mut boxed, ordinary[0], &x, &y);
            for (a, b) in m.iter().zip(&b) {
                assert_eq!(a.words(), b.words());
            }
            for (lane, &dot) in expected.iter().enumerate() {
                assert_eq!(
                    m.iter().fold(0_u64, |s, m| s.wrapping_add(m.words()[lane])) as i64 as i128,
                    dot
                );
                assert!((i128::from(p.raw.lower)..=i128::from(p.raw.upper)).contains(&dot));
            }
            let m = rotate(&mut norm, m);
            let b = rotate(&mut boxed, b);
            let (out, reference) = if rescale {
                (
                    finish_truncation(&mut norm, m),
                    finish_truncation(&mut boxed, b),
                )
            } else {
                assert!(m.is_empty() && b.is_empty());
                (
                    norm.each_mut().map(|o| o.take_output().unwrap()),
                    boxed.each_mut().map(|o| o.take_output().unwrap()),
                )
            };
            for (lane, &dot) in expected.iter().enumerate() {
                let actual = out
                    .iter()
                    .fold(0_u64, |s, o| s.wrapping_add(o.first()[lane]))
                    as i64 as i128;
                if rescale {
                    let unit = 1_i128 << FRACTION_BITS;
                    assert!(actual == dot.div_euclid(unit) || actual == -(-dot).div_euclid(unit));
                    let envelope = p.output_bounds().unwrap();
                    assert!(
                        (i128::from(envelope.lower)..=i128::from(envelope.upper)).contains(&actual)
                    );
                } else {
                    assert_eq!(actual, dot);
                }
            }
            let q = blocks(shape.output_elements()).unwrap();
            for i in 0..3 {
                assert_eq!(out[i].first(), reference[i].first());
                assert_eq!(out[i].second(), out[(i + 1) % 3].first());
                assert_eq!(norm[i].cursors(), boxed[i].cursors());
                assert_eq!(
                    norm[i].cursors().pair_blocks,
                    before[i].pair_blocks + if rescale { 3 * q } else { q }
                );
                assert_eq!(
                    norm[i].cursors().public_blocks,
                    before[i].public_blocks + u64::from(rescale)
                );
                if !rescale {
                    assert_eq!(norm[i].cursors().aes_work, before[i].aes_work + 2 * q);
                }
                assert!(norm[i].payload_usage().peak_bytes() <= plans[0].admitted_bytes());
            }
            drop(out);
            drop(reference);
            assert!(norm.iter().all(|o| o.payload_usage().live_bytes() == 0));
        }
    }
}

#[test]
fn whole_row_energy_admits_sparse_dot_rejected_by_independent_boxes() {
    let shape = MatrixShape::new(1, 4, 1).unwrap();
    let radius = 1_i64 << 30;
    let b = bounds(radius);
    assert_eq!(b.dot_product(4), Err(Error::Bounds));
    let p = RowEnergyProof::new(4, b, 1_u128 << 60, 1_u128 << 60).unwrap();
    assert_eq!(p.raw_bounds(), Interval::new(-(1 << 60), 1 << 60).unwrap());
    let mut owners = parties(PrssLimits::default());
    let plans = owners
        .each_mut()
        .map(|o| o.plan_norm_matmul(shape, p).unwrap());
    let m = messages(
        &mut owners,
        plans[0],
        &[radius, 0, 0, 0],
        &[-radius, 0, 0, 0],
    );
    let m = rotate(&mut owners, m);
    let out = finish_truncation(&mut owners, m);
    assert_eq!(
        out.iter().fold(0_u64, |s, o| s.wrapping_add(o.first()[0])) as i64,
        -(1 << 42)
    );
}

#[test]
fn every_stochastic_mask_respects_integer_envelope() {
    let scale = 1_i128 << FRACTION_BITS;
    for radius in [0, 1, 17, (1 << 18) - 1, 1 << 18, (1 << 18) + 1] {
        let p = RowEnergyProof::new(1, bounds(radius), radius as u128, radius as u128).unwrap();
        let envelope = p.output_bounds().unwrap();
        for dot in [-radius, 0, radius] {
            // All residues of the source's low-bit stochastic mask, not samples.
            for mask in 0..scale {
                let rounded = (i128::from(dot) + mask).div_euclid(scale);
                assert!(
                    (i128::from(envelope.lower)..=i128::from(envelope.upper)).contains(&rounded)
                );
            }
        }
    }
}

#[test]
fn width_tampered_proof_geometry_context_and_replay_fail_closed() {
    let p = proof(3, 8);
    for ticket in [
        RowEnergyProof {
            raw: Interval::new(-1, 1).unwrap(),
            ..p
        },
        p,
    ] {
        let mut owners = parties(PrssLimits::default());
        let shape = MatrixShape::new(1, if ticket == p { 2 } else { 3 }, 1).unwrap();
        assert!(owners[0].plan_norm_matmul(shape, ticket).is_err());
        assert_eq!(owners[0].cursors(), Cursors::default());
        assert_eq!(owners[0].status(), Status::Retired);
    }
    let shape = MatrixShape::new(2, 3, 2).unwrap();
    let mut owners = parties(PrssLimits::default());
    let plan = owners[0].plan_norm_matmul(shape, p).unwrap();
    let variants = [
        RowEnergyProof::new(3, bounds(9), 192, 192).unwrap(),
        RowEnergyProof::new(3, bounds(8), 193, 192).unwrap(),
    ];
    for variant in variants {
        let mut other = parties(PrssLimits::default());
        assert_ne!(
            plan.digest(),
            other[0].plan_norm_matmul(shape, variant).unwrap().digest()
        );
    }
    let mut other = parties(PrssLimits::default());
    assert_ne!(
        plan.digest(),
        other[0]
            .plan_norm_matmul(MatrixShape::new(1, 3, 4).unwrap(), p)
            .unwrap()
            .digest()
    );
    let mut x = shares(plan, Operand::Left, &[0; 6])
        .into_iter()
        .next()
        .unwrap();
    x.operation.context.phase += 1;
    let y = shares(plan, Operand::Right, &[0; 6])
        .into_iter()
        .next()
        .unwrap();
    assert!(owners[0].begin(x, y).is_err());
    assert_eq!(owners[0].cursors(), Cursors::default());
    assert_eq!(owners[0].status(), Status::Retired);

    let mut owners = parties(PrssLimits::default());
    let plan = owners[0].plan_ring_norm_matmul(shape, p).unwrap();
    let mut x = shares(plan, Operand::Left, &[0; 6]);
    let mut y = shares(plan, Operand::Right, &[0; 6]);
    let old_x = std::mem::replace(
        &mut x[0],
        shares(plan, Operand::Left, &[0; 6])
            .into_iter()
            .next()
            .unwrap(),
    );
    let old_y = std::mem::replace(
        &mut y[0],
        shares(plan, Operand::Right, &[0; 6])
            .into_iter()
            .next()
            .unwrap(),
    );
    let _frame = owners[0].begin(old_x, old_y).unwrap();
    let replay_x = x.into_iter().next().unwrap();
    let replay_y = y.into_iter().next().unwrap();
    assert!(owners[0].begin(replay_x, replay_y).is_err());
    assert_eq!(owners[0].status(), Status::Retired);
}

#[test]
fn malformed_rotation_cancel_and_resource_limits_preserve_ledger() {
    let shape = MatrixShape::new(1, 3, 1).unwrap();
    let p = proof(3, 8);
    let mut owners = parties(PrssLimits::default());
    let plans = owners
        .each_mut()
        .map(|o| o.plan_norm_matmul(shape, p).unwrap());
    let mut m = messages(&mut owners, plans[0], &[1, -2, 3], &[-4, 5, -6]);
    let i = m.iter().position(|m| m.header.to == Party::P0).unwrap();
    let mut bad = m.swap_remove(i);
    bad.header.binding[0] ^= 1;
    let usage = owners[0].payload_usage();
    assert!(owners[0].rotate(bad).is_err());
    assert_eq!(owners[0].status(), Status::Retired);
    for owner in &mut owners {
        owner.cancel();
    }
    drop(m);
    assert_eq!(usage.live_bytes(), 0);
    assert!(owners.iter().all(|o| o.payload_usage().live_bytes() == 0));
    assert!(owners[0].plan_norm_matmul(shape, p).is_err());

    let mut owners = parties(PrssLimits::new(Limits::default(), 1).unwrap());
    assert_eq!(owners[0].plan_norm_matmul(shape, p), Err(Error::Bounds));
    assert_eq!(owners[0].cursors(), Cursors::default());
    let shape = MatrixShape::new(1, 1024, 1).unwrap();
    let mut owners = parties(PrssLimits::default());
    let usage = owners[0].payload_usage();
    let retained: Vec<_> = (0..15).map(|_| usage.zeros(2048).unwrap()).collect();
    assert_eq!(
        owners[0].plan_ring_norm_matmul(shape, proof(1024, 1)),
        Err(Error::Bounds)
    );
    assert_eq!(owners[0].cursors(), Cursors::default());
    assert_eq!(usage.live_bytes(), 15 * 2048 * 8);
    drop(retained);
    assert_eq!(usage.live_bytes(), 0);
}

#[test]
fn substituted_proof_inputs_malformed_imports_and_narrow_limits_reject() {
    let shape = MatrixShape::new(2, 3, 2).unwrap();
    let p = proof(3, 8);
    for substitution in 0..3 {
        let mut owners = parties(PrssLimits::default());
        let mut other = parties(PrssLimits::default());
        let plan = owners[0].plan_norm_matmul(shape, p).unwrap();
        let foreign = match substitution {
            0 => other[0]
                .plan_norm_matmul(shape, RowEnergyProof::new(3, bounds(8), 193, 192).unwrap()),
            1 => other[0].plan_ring_norm_matmul(shape, p),
            _ => other[0].plan_matmul(shape, p.input_bounds()),
        }
        .unwrap();
        assert_eq!(
            plan.share_operand(
                Operand::Left,
                Party::P0,
                &[0; 5],
                &[0; 6],
                Limits::default()
            )
            .err(),
            Some(Error::Shape)
        );
        let x = shares(foreign, Operand::Left, &[0; 6])
            .into_iter()
            .next()
            .unwrap();
        let y = shares(plan, Operand::Right, &[0; 6])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(owners[0].begin(x, y).err(), Some(Error::Binding));
        assert_eq!(owners[0].cursors(), Cursors::default());
        assert_eq!(owners[0].status(), Status::Retired);
    }
    let core = Limits::new(4, 1, Limits::default().max_bytes).unwrap();
    let mut owners = parties(PrssLimits::new(core, 100).unwrap());
    assert_eq!(owners[0].plan_norm_matmul(shape, p), Err(Error::Bounds));
    assert_eq!(owners[0].cursors(), Cursors::default());

    // Consume exactly the last permitted ordinal, then reject before randomness.
    let core = Limits::new(6, 1, Limits::default().max_bytes).unwrap();
    let mut owners = parties(PrssLimits::new(core, 100).unwrap());
    let plans = owners
        .each_mut()
        .map(|o| o.plan_ring_norm_matmul(shape, p).unwrap());
    let m = messages(&mut owners, plans[0], &[0; 6], &[0; 6]);
    assert!(rotate(&mut owners, m).is_empty());
    drop(owners.each_mut().map(|o| o.take_output().unwrap()));
    let before = owners[0].cursors();
    assert_eq!(
        owners[0].plan_ring_norm_matmul(shape, p),
        Err(Error::Bounds)
    );
    assert_eq!(owners[0].cursors(), before);
    assert_eq!(owners[0].status(), Status::Retired);
}
