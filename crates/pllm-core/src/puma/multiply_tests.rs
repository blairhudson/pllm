use super::*;
use crate::puma::prss::{SeedEntropy, SeedPurpose};
use crate::puma::{Limits, MAX_LANES};
use serde_json::Value;
use zeroize::Zeroizing;

// Public deterministic oracle keys exist only in this test module.
struct FixedSeeds(usize);
impl SeedEntropy for FixedSeeds {
    fn draw(&mut self, purpose: SeedPurpose) -> Result<Zeroizing<[u8; 16]>, Error> {
        let value = match purpose {
            SeedPurpose::Public => self.0 as u128 + 1,
            SeedPurpose::Private => self.0 as u128 + 101,
            SeedPurpose::Pairwise => (self.0 as u128 + 1) * 11,
        };
        Ok(Zeroizing::new(value.to_le_bytes()))
    }
}

fn parties(limits: PrssLimits) -> [PartyRuntime; 3] {
    let context = bound_context(Context::new([31; 32], [43; 32], 7).unwrap());
    let mut setups = [Party::P0, Party::P1, Party::P2].map(|party| {
        SetupParty(
            super::super::SetupParty::with_entropy(
                context,
                party,
                limits,
                &mut FixedSeeds(party as usize),
            )
            .unwrap(),
        )
    });
    let public = setups.each_mut().map(|p| p.contribution().unwrap());
    for (i, owner) in setups.iter_mut().enumerate() {
        for (j, contribution) in public.iter().enumerate() {
            if i != j {
                owner.receive_public(*contribution).unwrap();
            }
        }
    }
    let mut seeds = setups.each_mut().map(|p| Some(p.share_seed().unwrap()));
    for (i, owner) in setups.iter_mut().enumerate() {
        owner
            .receive_seed(seeds[(i + 1) % 3].take().unwrap())
            .unwrap();
    }
    setups.each_mut().map(|p| p.finish().unwrap())
}

fn bound(x: (i64, i64), y: (i64, i64)) -> InputBounds {
    InputBounds {
        left: Interval::new(x.0, x.1).unwrap(),
        right: Interval::new(y.0, y.1).unwrap(),
    }
}

fn shares(op: Plan, x: &[i64], right: bool) -> [ReplicatedShare; 3] {
    let seq = op.sequence();
    let a: Vec<_> = (0..x.len())
        .map(|i| {
            if right {
                0xfedcba9876543210_u64
                    .wrapping_sub(i as u64)
                    .wrapping_add(11 * seq)
            } else {
                0x1020304050607080_u64 + i as u64 + 13 * seq
            }
        })
        .collect();
    let b: Vec<_> = (0..x.len())
        .map(|i| {
            if right {
                0x0102030405060708_u64
                    .wrapping_add(29 * i as u64)
                    .wrapping_sub(23 * seq)
            } else {
                0x8877665544332211_u64
                    .wrapping_sub(17 * i as u64)
                    .wrapping_add(37 * seq)
            }
        })
        .collect();
    let c: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| (v as u64).wrapping_sub(a[i]).wrapping_sub(b[i]))
        .collect();
    let parts = [&a[..], &b[..], &c[..]];
    [Party::P0, Party::P1, Party::P2].map(|p| {
        op.share_operand(
            if right { Operand::Right } else { Operand::Left },
            p,
            parts[p as usize],
            parts[p.next() as usize],
            Limits::default(),
        )
        .unwrap()
    })
}

fn take(messages: &mut Vec<Message>, party: Party, phase: MessagePhase) -> Message {
    let index = messages
        .iter()
        .position(|m| m.header().to == party && m.header().phase == phase)
        .unwrap();
    messages.swap_remove(index)
}

fn begins(owners: &mut [PartyRuntime; 3], op: Plan, x: &[i64], y: &[i64]) -> Vec<Message> {
    owners
        .iter_mut()
        .zip(shares(op, x, false))
        .zip(shares(op, y, true))
        .map(|((p, x), y)| p.begin(x, y).unwrap())
        .collect()
}

fn finish(
    owners: &mut [PartyRuntime; 3],
    plan: Plan,
    mut messages: Vec<Message>,
) -> [ReplicatedShare; 3] {
    for owner in owners.iter_mut() {
        let next = take(&mut messages, owner.party(), MessagePhase::Multiply);
        messages.extend(owner.rotate(next).unwrap().into_iter().flatten());
    }
    let op = owners[0].truncation_operation().unwrap();
    assert_eq!(op.consumer, Some(plan.digest()));
    assert!(owners.iter().all(|p| p.truncation_operation() == Some(op)));
    for p in [op.evaluator_a(), op.evaluator_b()] {
        let peer = take(&mut messages, p, MessagePhase::MaskedInput);
        let correlation = take(&mut messages, p, MessagePhase::Correlation);
        messages.push(owners[p as usize].open(peer, correlation).unwrap());
    }
    for p in [op.evaluator_a(), op.evaluator_b()] {
        owners[p as usize]
            .reshare(take(&mut messages, p, MessagePhase::Reshare))
            .unwrap();
    }
    assert!(messages.is_empty());
    owners.each_mut().map(|p| p.take_output().unwrap())
}

fn words(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u64::from_str_radix(v.as_str().unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn independent_frozen_transcripts_match_products_outputs_and_every_cursor() {
    let vectors: Value = serde_json::from_str(include_str!(
        "../../../../paper/reproduction-contracts/puma/vectors.json"
    ))
    .unwrap();
    let unit = 1_i64 << 18;
    let cases = [
        (
            vec![-9, 1 << 31, unit - 1],
            vec![1, -(1 << 31), unit + 17],
            vec![
                bound((-11, -7), (1, 2)),
                bound((1 << 31, 1 << 31), (-(1 << 31), -(1 << 31))),
                bound((unit - 2, unit), (unit, unit + 17)),
            ],
        ),
        (
            vec![INPUT_MAX_EXCLUSIVE - 1],
            vec![1],
            vec![bound(
                (INPUT_MAX_EXCLUSIVE - 2, INPUT_MAX_EXCLUSIVE - 1),
                (1, 1),
            )],
        ),
        (
            vec![0, unit + 1, -unit - 1, (1 << 31) - 1],
            vec![i64::MAX, unit - 1, unit + 1, (1 << 31) - 1],
            vec![
                bound((0, 0), (i64::MIN, i64::MAX)),
                bound((unit, unit + 2), (unit - 2, unit)),
                bound((-unit - 2, -unit), (unit, unit + 2)),
                bound(
                    (-(1 << 31) + 1, (1 << 31) - 1),
                    (-(1 << 31) + 1, (1 << 31) - 1),
                ),
            ],
        ),
    ];
    let mut owners = parties(PrssLimits::default());
    for (ordinal, (x, y, bounds)) in cases.into_iter().enumerate() {
        let expected = &vectors["cases"][ordinal];
        assert_eq!(expected["plan"]["profile"], PROFILE);
        let before = owners.each_ref().map(|p| p.cursors());
        let ops = owners.each_mut().map(|p| p.plan(&bounds).unwrap());
        assert_eq!(ops, [ops[0]; 3]);
        let op = ops[0];
        assert_eq!(op.sequence(), ordinal as u64);
        assert_eq!(owners.each_ref().map(|p| p.cursors()), before);
        assert!(owners.iter().all(|p| p.truncation_operation().is_none()));
        let messages = begins(&mut owners, op, &x, &y);
        for (i, owner) in owners.iter().enumerate() {
            let q = blocks(x.len()).unwrap();
            assert_eq!(owner.cursors().public_blocks, before[i].public_blocks);
            assert_eq!(owner.cursors().private_blocks, before[i].private_blocks);
            assert_eq!(owner.cursors().pair_blocks, before[i].pair_blocks + q);
            assert_eq!(owner.cursors().aes_work, before[i].aes_work + 2 * q);
            assert!(owner.truncation_operation().is_none());
        }
        for (i, message) in messages.iter().enumerate() {
            assert_eq!(message.words(), words(&expected["mul_components_hex"][i]));
            assert_eq!(message.header().from as usize, i);
            assert_eq!(message.header().to as usize, (i + 2) % 3);
        }
        let output = finish(&mut owners, op, messages);
        for (i, share) in output.iter().enumerate() {
            assert_eq!(share.first(), words(&expected["output_pairs_hex"][i][0]));
            assert_eq!(share.second(), words(&expected["output_pairs_hex"][i][1]));
            assert_eq!(share.operation().consumer, Some(op.digest()));
            assert_eq!(
                share.operation().evaluator_a() as u64,
                expected["roles"]["a"].as_u64().unwrap()
            );
            let cursors = owners[i].cursors();
            assert_eq!(
                cursors.public_blocks,
                expected["after"]["public"].as_u64().unwrap()
            );
            assert_eq!(
                cursors.pair_blocks,
                expected["after"]["pair"].as_u64().unwrap()
            );
            assert_eq!(
                cursors.private_blocks,
                expected["after"]["private"][i].as_u64().unwrap()
            );
            assert_eq!(
                cursors.aes_work,
                expected["after"]["work"][i].as_u64().unwrap()
            );
        }
        for j in 0..x.len() {
            let clear = output
                .iter()
                .fold(0_u64, |s, p| s.wrapping_add(p.first()[j])) as i64;
            assert_eq!(
                clear,
                expected["expected_signed"][j]
                    .as_str()
                    .unwrap()
                    .parse::<i64>()
                    .unwrap()
            );
        }
        drop(output);
        assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
    }
}

#[test]
fn public_hulls_check_unwrapped_corners_and_signed_rounding() {
    assert!(Interval::new(1, 0).is_err());
    for b in [
        bound((1 << 32, 1 << 32), (1 << 32, 1 << 32)), // wraps to zero
        bound((1 << 31, 1 << 31), (1 << 31, 1 << 31)), // excluded positive edge
        bound((i64::MIN, i64::MAX), (-1, 1)),
        bound((-(1 << 31), 0), (-(1 << 31), 0)),
    ] {
        assert_eq!(b.product(), Err(Error::Bounds));
    }
    assert_eq!(
        bound((-(1 << 31), -(1 << 31)), (1 << 31, 1 << 31))
            .product()
            .unwrap()
            .lower(),
        INPUT_MIN
    );
    assert_eq!(
        bound((0, 0), (i64::MIN, i64::MAX)).output(),
        Interval::new(0, 0)
    );
    assert_eq!(bound((-9, -9), (1, 1)).output(), Interval::new(-1, 0));
    for a in -6..=6 {
        for b in a..=6 {
            for c in -6..=6 {
                for d in c..=6 {
                    let values: Vec<i64> =
                        (a..=b).flat_map(|x| (c..=d).map(move |y| x * y)).collect();
                    let hull = bound((a, b), (c, d)).product().unwrap();
                    assert_eq!(hull.lower(), *values.iter().min().unwrap());
                    assert_eq!(hull.upper(), *values.iter().max().unwrap());
                }
            }
        }
    }
}

#[test]
fn all_lanes_and_work_are_admitted_before_prss() {
    for bounds in [
        vec![],
        vec![bound((0, 0), (0, 0)); MAX_LANES + 1],
        vec![
            bound((0, 0), (1, 1)),
            bound((1 << 32, 1 << 32), (1 << 32, 1 << 32)),
        ],
    ] {
        let mut owners = parties(PrssLimits::default());
        assert_eq!(owners[0].plan(&bounds), Err(Error::Bounds));
        assert_eq!(owners[0].cursors(), Cursors::default());
        assert_eq!(owners[0].status(), Status::Retired);
    }
    let mut owners = parties(PrssLimits::new(Limits::default(), 1).unwrap());
    assert_eq!(owners[0].plan(&[bound((1, 1), (1, 1))]), Err(Error::Bounds));
    assert_eq!(owners[0].cursors(), Cursors::default());
    assert!(owners[0].source.seeds.is_none());
}

#[test]
fn every_public_interval_and_compound_namespace_binds_the_operation() {
    let mut first = parties(PrssLimits::default());
    let mut second = parties(PrssLimits::default());
    let a = first[0].plan(&[bound((0, 1), (0, 1))]).unwrap();
    let b = second[0].plan(&[bound((0, 2), (0, 1))]).unwrap();
    assert_ne!(a.digest(), b.digest());
    let mut plain = a.input_binding();
    plain.consumer = None;
    assert_ne!(a.digest(), plain.digest());
    let mut old = super::super::SetupParty::new(
        Context::new([31; 32], [43; 32], 7).unwrap(),
        Party::P1,
        PrssLimits::default(),
    )
    .unwrap();
    let message = old.contribution().unwrap();
    let mut new = SetupParty::new(
        Context::new([31; 32], [43; 32], 7).unwrap(),
        Party::P0,
        PrssLimits::default(),
    )
    .unwrap();
    new.contribution().unwrap();
    assert_eq!(new.receive_public(message), Err(Error::Binding));
    assert_eq!(new.status(), super::super::SetupStatus::Retired);
}

#[test]
fn wrong_phase_role_binding_and_replay_burn_the_entire_owner() {
    let bounds = [bound((-8, 8), (-8, 8))];
    for mutation in 0..5 {
        let mut owners = parties(PrssLimits::default());
        let op = owners[0].plan(&bounds).unwrap();
        owners[1].plan(&bounds).unwrap();
        owners[2].plan(&bounds).unwrap();
        let mut messages = begins(&mut owners, op, &[3], &[5]);
        let mut next = take(&mut messages, Party::P0, MessagePhase::Multiply);
        match mutation {
            0 => next.header.binding[0] ^= 1,
            1 => next.header.from = Party::P2,
            2 => next.header.to = Party::P1,
            3 => next.header.phase = MessagePhase::MaskedInput,
            _ => {
                let duplicate = Message::from_received(
                    next.header(),
                    next.words(),
                    op.input_binding(),
                    Limits::default(),
                )
                .unwrap();
                drop(owners[0].rotate(duplicate).unwrap());
            }
        }
        assert!(owners[0].rotate(next).is_err());
        assert_eq!(owners[0].status(), Status::Retired);
        assert!(owners[0].source.seeds.is_none());
        assert_eq!(owners[0].plan(&bounds), Err(Error::Retired));
        drop(messages);
        drop(owners);
    }
}

#[test]
fn cancellation_at_each_compound_phase_preserves_exported_leases() {
    for phase in 0..7 {
        let mut owners = parties(PrssLimits::default());
        let usages = owners.each_ref().map(|p| p.payload_usage());
        let mut messages = Vec::new();
        let mut outputs = None;
        if phase > 0 {
            let ops = owners
                .each_mut()
                .map(|p| p.plan(&[bound((-8, 8), (-8, 8))]).unwrap());
            if phase > 1 {
                messages = begins(&mut owners, ops[0], &[3], &[5]);
                if phase > 2 {
                    for p in &mut owners {
                        let next = take(&mut messages, p.party(), MessagePhase::Multiply);
                        messages.extend(p.rotate(next).unwrap().into_iter().flatten());
                    }
                    let op = owners[0].truncation_operation().unwrap();
                    if phase > 3 {
                        for p in [op.evaluator_a(), op.evaluator_b()] {
                            let peer = take(&mut messages, p, MessagePhase::MaskedInput);
                            let correlation = take(&mut messages, p, MessagePhase::Correlation);
                            messages.push(owners[p as usize].open(peer, correlation).unwrap());
                        }
                    }
                    if phase > 4 {
                        for p in [op.evaluator_a(), op.evaluator_b()] {
                            owners[p as usize]
                                .reshare(take(&mut messages, p, MessagePhase::Reshare))
                                .unwrap();
                        }
                    }
                    if phase > 5 {
                        outputs = Some(owners.each_mut().map(|p| p.take_output().unwrap()));
                    }
                }
            }
        }
        for owner in &mut owners {
            owner.cancel();
            owner.cancel();
            assert_eq!(owner.status(), Status::Retired);
            assert!(owner.take_output().is_err());
        }
        drop(owners);
        assert_eq!(
            usages.iter().any(|u| u.live_bytes() > 0),
            !messages.is_empty() || outputs.is_some()
        );
        drop(messages);
        drop(outputs);
        assert!(usages.iter().all(|u| u.live_bytes() == 0));
    }
}

#[test]
fn joint_peak_counts_retained_rotation_with_helper_material() {
    assert!(std::mem::size_of::<PartyRuntime>() <= FIXED_RESERVE_BYTES);
    let mut owners = parties(PrssLimits::default());
    let bounds = vec![bound((-8, 8), (-8, 8)); MAX_LANES];
    let ops = owners.each_mut().map(|p| p.plan(&bounds).unwrap());
    let values = vec![1; MAX_LANES];
    let mut messages = begins(&mut owners, ops[0], &values, &values);
    // Fixed public fixture's helper goes first, so its original rotation is
    // still live at the complete helper-material peak.
    let next = take(&mut messages, Party::P1, MessagePhase::Multiply);
    let frames = owners[1].rotate(next).unwrap();
    assert_eq!(
        owners[1].truncation_operation().unwrap().helper(),
        Party::P1
    );
    let usage = owners[1].payload_usage();
    assert_eq!(usage.peak_bytes(), 11 * MAX_LANES * 8);
    assert!(usage.peak_bytes() <= Limits::default().admitted_bytes(MAX_LANES).unwrap());
    drop(owners);
    assert!(usage.live_bytes() > 0);
    drop(frames);
    drop(messages);
    assert_eq!(usage.live_bytes(), 0);
}

#[test]
fn full_shape_budget_exhaustion_precedes_randomness_and_retains_old_outputs() {
    let mut owners = parties(PrssLimits::default());
    let bounds = vec![bound((-8, 8), (-8, 8)); MAX_LANES];
    let values = vec![1; MAX_LANES];
    let usages = owners.each_ref().map(|p| p.payload_usage());
    let mut retained = Vec::new();
    loop {
        let before = owners[0].cursors();
        let op = match owners[0].plan(&bounds) {
            Ok(op) => op,
            Err(Error::Bounds) => {
                assert_eq!(owners[0].cursors(), before);
                break;
            }
            Err(e) => panic!("{e}"),
        };
        for owner in &mut owners[1..] {
            assert_eq!(owner.plan(&bounds), Ok(op));
        }
        let messages = begins(&mut owners, op, &values, &values);
        retained.push(finish(&mut owners, op, messages));
        assert!(retained.len() < 32);
    }
    assert_eq!(retained.len(), 8);
    assert!(usages.iter().all(|u| u.peak_bytes() <= u.limit_bytes()));
    drop(owners);
    assert!(usages
        .iter()
        .all(|u| u.live_bytes() == 8 * 2 * MAX_LANES * 8));
    drop(retained);
    assert!(usages.iter().all(|u| u.live_bytes() == 0));
}

#[test]
fn matrix_products_reconstruct_before_one_truncation_of_the_complete_sum() {
    let mut owners = parties(PrssLimits::default());
    let unit = 1_i64 << FRACTION_BITS;
    for (ordinal, dimensions) in [[1, 3, 1], [2, 3, 4], [3, 7, 5], [1, 1024, 1], [32, 32, 32]]
        .into_iter()
        .enumerate()
    {
        let [rows, inner, columns] = dimensions;
        let shape = MatrixShape::new(rows, inner, columns).unwrap();
        let bounds = bound((-unit, unit), (-unit, unit));
        let before = owners.each_ref().map(|p| p.cursors());
        let plans = owners
            .each_mut()
            .map(|p| p.plan_matmul(shape, bounds).unwrap());
        assert_eq!(plans, [plans[0]; 3]);
        assert_eq!(owners.each_ref().map(|p| p.cursors()), before);
        let plan = plans[0];
        let x: Vec<_> = (0..shape.left_elements())
            .map(|i| {
                if ordinal == 0 {
                    unit - 1
                } else {
                    ((i * 111337 + 31) % (2 * unit as usize)) as i64 - unit
                }
            })
            .collect();
        let y: Vec<_> = (0..shape.right_elements())
            .map(|i| {
                if ordinal == 0 {
                    1
                } else {
                    ((i * 7717 + 7) % (2 * unit as usize)) as i64 - unit
                }
            })
            .collect();
        // Independent signed i128 dot products: neither ring-kernel reuse nor
        // per-term quantization can supply this reference.
        let expected: Vec<i128> = (0..rows)
            .flat_map(|row| {
                let x = &x;
                let y = &y;
                (0..columns).map(move |column| {
                    (0..inner)
                        .map(|k| {
                            i128::from(x[row * inner + k]) * i128::from(y[k * columns + column])
                        })
                        .sum()
                })
            })
            .collect();
        let messages = begins(&mut owners, plan, &x, &y);
        for (lane, value) in expected.iter().enumerate() {
            let raw = messages
                .iter()
                .fold(0_u64, |sum, m| sum.wrapping_add(m.words()[lane]));
            assert_eq!(i128::from(raw as i64), *value);
        }
        let output = finish(&mut owners, plan, messages);
        for (lane, value) in expected.iter().enumerate() {
            let actual = output
                .iter()
                .fold(0_u64, |sum, p| sum.wrapping_add(p.first()[lane]))
                as i64;
            let floor = value.div_euclid(i128::from(unit)) as i64;
            let ceil = -(-value).div_euclid(i128::from(unit)) as i64;
            assert!(actual == floor || actual == ceil);
            if ordinal == 0 {
                assert!((2..=3).contains(&actual));
            }
        }
        let q = (shape.output_elements() as u64).div_ceil(2);
        for (i, owner) in owners.iter().enumerate() {
            assert_eq!(owner.cursors().pair_blocks, before[i].pair_blocks + 3 * q);
            assert_eq!(owner.cursors().public_blocks, before[i].public_blocks + 1);
            assert_eq!(output[i].second(), output[(i + 1) % 3].first());
            assert!(owner.payload_usage().peak_bytes() <= plan.admitted_bytes());
        }
        drop(output);
        assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
    }
}

#[test]
fn matrix_accumulator_ranges_shapes_and_live_inputs_are_admitted_before_prss() {
    for dimensions in [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [usize::MAX, 2, 1],
        [33, 33, 1],
        [1, 33, 33],
        [33, 1, 33],
    ] {
        assert_eq!(
            MatrixShape::new(dimensions[0], dimensions[1], dimensions[2]),
            Err(Error::Bounds)
        );
    }
    let safe_scalar = bound((1 << 30, 1 << 30), (1 << 30, 1 << 30));
    assert!(safe_scalar.product().is_ok());
    assert_eq!(safe_scalar.dot_product(4), Err(Error::Bounds));
    assert_eq!(safe_scalar.dot_product(0), Err(Error::Bounds));
    assert_eq!(safe_scalar.dot_product(MAX_LANES + 1), Err(Error::Bounds));
    assert_eq!(
        bound((-(1 << 30), -(1 << 30)), (1 << 30, 1 << 30))
            .dot_product(4)
            .unwrap()
            .lower(),
        INPUT_MIN
    );
    assert_eq!(bound((-9, -9), (1, 1)).dot_output(3), Interval::new(-1, 0));
    let mut owners = parties(PrssLimits::default());
    assert_eq!(
        owners[0].plan_matmul(MatrixShape::new(1, 4, 1).unwrap(), safe_scalar),
        Err(Error::Bounds)
    );
    assert_eq!(owners[0].cursors(), Cursors::default());
    assert_eq!(owners[0].status(), Status::Retired);

    let shape = MatrixShape::new(1, 1024, 1).unwrap();
    let mut owners = parties(PrssLimits::default());
    let usage = owners[0].payload_usage();
    // Retained payload leaves room for 16 output words but NOT both inputs.
    let retained: Vec<_> = (0..15).map(|_| usage.zeros(2048).unwrap()).collect();
    assert_eq!(
        owners[0].plan_matmul(shape, bound((-1, 1), (-1, 1))),
        Err(Error::Bounds)
    );
    assert_eq!(owners[0].cursors(), Cursors::default());
    assert_eq!(usage.live_bytes(), 15 * 2048 * 8);
    drop(retained);
    assert_eq!(usage.live_bytes(), 0);
}

#[test]
fn matrix_operand_order_and_geometry_cannot_be_substituted() {
    let bounds = bound((-8, 8), (-8, 8));
    let mut first = parties(PrssLimits::default());
    let mut second = parties(PrssLimits::default());
    let a = first[0]
        .plan_matmul(MatrixShape::new(2, 3, 2).unwrap(), bounds)
        .unwrap();
    let b = second[0]
        .plan_matmul(MatrixShape::new(1, 3, 4).unwrap(), bounds)
        .unwrap();
    assert_ne!(a.digest(), b.digest());
    assert_ne!(
        a.operand_binding(Operand::Left),
        a.operand_binding(Operand::Right)
    );
    assert!(a
        .share(Party::P0, &[0; 4], &[0; 4], Limits::default())
        .is_err());
    let left = shares(a, &[0; 6], false).into_iter().next().unwrap();
    let right = shares(a, &[0; 6], true).into_iter().next().unwrap();
    assert!(first[0].begin(right, left).is_err());
    assert_eq!(first[0].cursors(), Cursors::default());
    assert_eq!(first[0].status(), Status::Retired);
}
