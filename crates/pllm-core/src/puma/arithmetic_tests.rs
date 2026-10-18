use super::*;
use crate::puma::multiply::{
    tests::{finish_truncation, parties},
    Status,
};
use crate::puma::prss::PrssLimits;

fn input(owners: &[PartyRuntime; 3], values: &[i64]) -> [ReplicatedShare; 3] {
    let operation = Operation::new(
        owners[0].context(),
        0,
        Party::P0,
        values.len(),
        1,
        Limits::default(),
    )
    .unwrap();
    let first: Vec<_> = (0..values.len())
        .map(|i| 0xfedc_ba98_7654_3210_u64.wrapping_sub(i as u64))
        .collect();
    let second: Vec<_> = (0..values.len())
        .map(|i| 0x8765_4321_0123_4567_u64.wrapping_add(i as u64))
        .collect();
    let third: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (v as u64).wrapping_sub(first[i]).wrapping_sub(second[i]))
        .collect();
    let rows = [&first[..], &second[..], &third[..]];
    [Party::P0, Party::P1, Party::P2].map(|p| {
        ReplicatedShare::from_local(
            operation,
            p,
            rows[p as usize],
            rows[p.next() as usize],
            Limits::default(),
        )
        .unwrap()
    })
}

fn clear(values: &[ReplicatedShare; 3]) -> Vec<i64> {
    for i in 0..3 {
        assert_eq!(values[i].second(), values[(i + 1) % 3].first());
    }
    (0..values[0].first().len())
        .map(|i| {
            values
                .iter()
                .fold(0_u64, |a, p| a.wrapping_add(p.first()[i])) as i64
        })
        .collect()
}

#[test]
fn affine_preserves_exact_integer_values_and_retains_charged_inputs() {
    for n in [1, 3, MAX_LANES] {
        let mut owners = parties(PrssLimits::default());
        let x: Vec<_> = (0..n).map(|i| i as i64 - 100).collect();
        let y: Vec<_> = (0..n).map(|i| 10000 - i as i64).collect();
        let mut left = input(&owners, &x);
        let mut right = input(&owners, &y);
        let coefficients = Affine {
            left: -120,
            right: 27,
            constant: -73,
        };
        let bounds = InputBounds {
            left: Interval::new(-100, 1024).unwrap(),
            right: Interval::new(8977, 10000).unwrap(),
        };
        let output = std::array::from_fn::<_, 3, _>(|i| {
            let before = owners[i].cursors();
            let (out, bound) = owners[i]
                .affine(&mut left[i], Some(&mut right[i]), coefficients, bounds)
                .unwrap();
            assert_eq!(bound, coefficients.output_bounds(bounds).unwrap());
            assert_eq!(before, owners[i].cursors());
            assert_eq!(owners[i].payload_usage().live_bytes(), 6 * n * 8);
            out
        });
        assert_eq!(
            clear(&output),
            x.iter()
                .zip(&y)
                .map(|(&a, &b)| -120 * a + 27 * b - 73)
                .collect::<Vec<_>>()
        );
        assert_eq!(clear(&left), x);
        assert_eq!(clear(&right), y);
        for p in &mut owners {
            p.cancel();
            assert_eq!(p.payload_usage().live_bytes(), 6 * n * 8);
        }
        drop(output);
        drop(left);
        drop(right);
        assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
    }
}

#[test]
fn public_product_rescales_once_without_private_product_randomness() {
    for shift in [1, 5, 18, 62] {
        for n in [1, 3, MAX_LANES] {
            let mut owners = parties(PrssLimits::default());
            let x: Vec<_> = (0..n).map(|i| i as i64 * 519001 - 2000000).collect();
            let mut originals = input(&owners, &x);
            let coefficients = Affine {
                left: -30952,
                right: 0,
                constant: 0,
            };
            let bounds = InputBounds {
                left: Interval::new(-2000000, 540000000).unwrap(),
                right: Interval::new(0, 0).unwrap(),
            };
            let output_bound = coefficients.output_bounds(bounds).unwrap();
            let output = std::array::from_fn::<_, 3, _>(|i| {
                owners[i]
                    .affine(&mut originals[i], None, coefficients, bounds)
                    .unwrap()
                    .0
            });
            let mut messages = Vec::new();
            let mut operations = Vec::new();
            for (owner, value) in owners.iter_mut().zip(output) {
                let (operation, frames) = owner.begin_rescale(value, shift, output_bound).unwrap();
                messages.extend(frames.into_iter().flatten());
                operations.push(operation);
            }
            assert!(operations.iter().all(|&v| v == operations[0]));
            let operation = operations[0];
            assert_eq!(operation.sequence(), 1);
            let output = finish_truncation(&mut owners, messages);
            let expected_bound = rescaled_bounds(output_bound, shift).unwrap();
            for (i, v) in clear(&output).into_iter().enumerate() {
                let raw = i128::from(x[i]) * i128::from(coefficients.left);
                let unit = 1_i128 << shift;
                let floor = raw.div_euclid(unit);
                assert!(
                    i128::from(v) == floor
                        || (raw.rem_euclid(unit) != 0 && i128::from(v) == floor + 1)
                );
                assert!(expected_bound.lower <= v && v <= expected_bound.upper);
            }
            let q = (n as u64).div_ceil(2);
            for p in &owners {
                let helper = p.party() == operation.helper();
                assert_eq!(p.cursors().pair_blocks, 2 * q);
                assert_eq!(p.cursors().public_blocks, 1);
                assert_eq!(
                    p.cursors().private_blocks,
                    if helper { n as u64 } else { 0 }
                );
                assert_eq!(
                    p.cursors().aes_work,
                    1 + if helper { 4 * q + n as u64 } else { 2 * q }
                );
            }
            drop(output);
            drop(originals);
            assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
        }
    }
}

#[test]
fn signed_affine_overflow_and_invalid_rescale_reject_before_draws() {
    for limit in [
        Interval::new(i64::MIN, i64::MAX).unwrap(),
        Interval::new(INPUT_MIN, INPUT_MAX_EXCLUSIVE).unwrap(),
    ] {
        let mut owners = parties(PrssLimits::default());
        let mut values = input(&owners, &[0]);
        let mut rhs = input(&owners, &[0]);
        let before = owners[0].cursors();
        let coefficients = Affine {
            left: -1,
            right: -1,
            constant: 0,
        };
        assert!(owners[0]
            .affine(
                &mut values[0],
                Some(&mut rhs[0]),
                coefficients,
                InputBounds {
                    left: limit,
                    right: limit
                }
            )
            .is_err());
        assert_eq!(owners[0].cursors(), before);
        assert_eq!(owners[0].status(), Status::Retired);
        assert!(coefficients
            .output_bounds(InputBounds {
                left: Interval::new(i64::MIN, i64::MAX).unwrap(),
                right: limit
            })
            .is_err());
    }
    for (shift, range, work) in [
        (0, (0, 1), 1000),
        (18, (INPUT_MIN, INPUT_MAX_EXCLUSIVE), 1000),
        (18, (0, 1), 1),
    ] {
        let mut owners = parties(PrssLimits::new(Limits::default(), work).unwrap());
        let [value, _, _] = input(&owners, &[0]);
        let before = owners[0].cursors();
        assert!(owners[0]
            .begin_rescale(value, shift, Interval::new(range.0, range.1).unwrap())
            .is_err());
        assert_eq!(owners[0].cursors(), before);
        assert_eq!(owners[0].status(), Status::Retired);
        assert_eq!(owners[0].payload_usage().live_bytes(), 0);
    }
}

#[test]
fn local_arithmetic_cannot_cross_an_active_conversion_or_a_session() {
    let mut owners = parties(PrssLimits::default());
    let mut values = input(&owners, &[0]);
    owners[0]
        .plan_conversion(1, crate::puma::multiply::conversion::Kind::Bits)
        .unwrap();
    let before = owners[0].cursors();
    assert!(owners[0]
        .affine(
            &mut values[0],
            None,
            Affine {
                left: 1,
                right: 0,
                constant: 0
            },
            InputBounds {
                left: Interval::new(0, 0).unwrap(),
                right: Interval::new(0, 0).unwrap()
            }
        )
        .is_err());
    assert_eq!(owners[0].cursors(), before);
    let mut owners = parties(PrssLimits::default());
    values[0].operation.context.phase += 1;
    let [value, _, _] = values;
    assert!(owners[0]
        .begin_rescale(value, 18, Interval::new(0, 0).unwrap())
        .is_err());
    assert_eq!(owners[0].cursors(), before);
}

#[test]
fn exact_ring_product_preserves_sign_selection_without_rounding_or_pivot() {
    use crate::puma::multiply::Operand;
    for n in [1, 3, MAX_LANES] {
        let mut owners = parties(PrssLimits::default());
        let values: Vec<_> = (0..n)
            .map(|i| match i % 4 {
                0 => i64::MIN,
                1 => i64::MAX,
                2 => -1,
                _ => 379,
            })
            .collect();
        let predicates: Vec<_> = (0..n).map(|i| (i % 2) as i64).collect();
        let x = input(&owners, &values);
        let p = input(&owners, &predicates);
        let bounds = InputBounds {
            left: Interval::new(i64::MIN, i64::MAX).unwrap(),
            right: Interval::new(0, 1).unwrap(),
        };
        let plans = owners
            .each_mut()
            .map(|p| p.plan_ring_product(&vec![bounds; n]).unwrap());
        assert_eq!(plans, [plans[0]; 3]);
        let mut messages: [Option<Message>; 3] = [None, None, None];
        for (i, (x, p)) in x.into_iter().zip(p).enumerate() {
            let x = owners[i].bind_operand(Operand::Left, x).unwrap();
            let p = owners[i].bind_operand(Operand::Right, p).unwrap();
            messages[i] = Some(owners[i].begin(x, p).unwrap());
        }
        for i in 0..3 {
            assert!(owners[i]
                .rotate(messages[(i + 1) % 3].take().unwrap())
                .unwrap()
                .iter()
                .all(Option::is_none));
            assert_eq!(owners[i].status(), Status::Complete);
        }
        let output = owners.each_mut().map(|p| p.take_output().unwrap());
        assert_eq!(
            clear(&output),
            values
                .iter()
                .zip(&predicates)
                .map(|(&x, &p)| x * p)
                .collect::<Vec<_>>()
        );
        for p in &owners {
            let q = (n as u64).div_ceil(2);
            assert_eq!(p.cursors().pair_blocks, q);
            assert_eq!(p.cursors().aes_work, 2 * q);
            assert_eq!(p.cursors().public_blocks, 0);
            assert_eq!(p.cursors().private_blocks, 0);
        }
        drop(output);
        assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
    }
}

#[test]
fn raw_product_bounds_binding_and_replay_fail_closed() {
    use crate::puma::multiply::Operand;
    let bounds = InputBounds {
        left: Interval::new(0, i64::MAX).unwrap(),
        right: Interval::new(0, 2).unwrap(),
    };
    let mut owners = parties(PrssLimits::default());
    let before = owners[0].cursors();
    assert!(owners[0].plan_ring_product(&[bounds]).is_err());
    assert_eq!(owners[0].cursors(), before);
    let mut owners = parties(PrssLimits::default());
    let [x, _, _] = input(&owners, &[0, 0]);
    owners[0]
        .plan_ring_product(&[InputBounds {
            left: Interval::new(0, 1).unwrap(),
            right: Interval::new(0, 1).unwrap(),
        }])
        .unwrap();
    assert!(owners[0].bind_operand(Operand::Left, x).is_err());
    assert_eq!(owners[0].status(), Status::Retired);
    assert_eq!(owners[0].cursors(), before);
}

#[test]
fn opaque_affine_sign_conversion_and_integer_product_compose_exact_relu() {
    use crate::puma::multiply::{conversion::Kind, Operand};
    for n in [3, MAX_LANES / 2] {
        let mut owners = parties(PrssLimits::default());
        let values: Vec<_> = (0..n)
            .map(|i| match i % 4 {
                0 => -1000000,
                1 => 1000000,
                2 => 0,
                _ => -1,
            })
            .collect();
        let range = Interval::new(-1000000, 1000000).unwrap();
        let unary_bounds = InputBounds {
            left: range,
            right: Interval::new(0, 0).unwrap(),
        };
        let identity = Affine {
            left: 1,
            right: 0,
            constant: 0,
        };
        let mut original = input(&owners, &values);
        let sign_inputs = std::array::from_fn::<_, 3, _>(|i| {
            owners[i]
                .affine(&mut original[i], None, identity, unary_bounds)
                .unwrap()
                .0
        });
        for p in &mut owners {
            p.plan_conversion(n, Kind::MostSignificantBit).unwrap();
        }
        let mut messages = Vec::new();
        for (p, value) in owners.iter_mut().zip(sign_inputs) {
            let value = p.bind_conversion_input(value).unwrap();
            messages.push(p.begin_conversion(value).unwrap());
        }
        for _ in 0..8 {
            let mut next = Vec::new();
            for message in messages.drain(..) {
                let to = message.header().to as usize;
                next.extend(owners[to].advance_conversion(message).unwrap());
            }
            messages = next;
        }
        assert!(messages.is_empty());
        let predicates = owners.each_mut().map(|p| p.take_boolean_output().unwrap());
        let plans = std::array::from_fn::<_, 3, _>(|i| {
            owners[i]
                .plan_boolean_to_arithmetic(&predicates[i])
                .unwrap()
        });
        let plan = plans[0];
        for (p, predicate) in owners.iter_mut().zip(predicates) {
            messages.extend(
                p.begin_boolean_to_arithmetic(predicate)
                    .unwrap()
                    .into_iter()
                    .flatten(),
            );
        }
        let incoming = [
            MessagePhase::BooleanOtMask,
            MessagePhase::BooleanOtZero,
            MessagePhase::BooleanOtOne,
        ]
        .map(|phase| {
            let at = messages
                .iter()
                .position(|m| m.header().phase == phase)
                .unwrap();
            Some(messages.remove(at))
        });
        let reply = owners[plan.receiver() as usize]
            .receive_boolean_to_arithmetic(incoming)
            .unwrap();
        owners[plan.helper() as usize]
            .receive_boolean_to_arithmetic([reply, None, None])
            .unwrap();
        let predicates = owners.each_mut().map(|p| p.take_output().unwrap());
        let copies = std::array::from_fn::<_, 3, _>(|i| {
            owners[i]
                .affine(&mut original[i], None, identity, unary_bounds)
                .unwrap()
                .0
        });
        for p in &mut owners {
            p.plan_ring_product(&vec![
                InputBounds {
                    left: range,
                    right: Interval::new(0, 1).unwrap()
                };
                n
            ])
            .unwrap();
        }
        for ((p, x), predicate) in owners.iter_mut().zip(copies).zip(predicates) {
            let x = p.bind_operand(Operand::Left, x).unwrap();
            let predicate = p.bind_operand(Operand::Right, predicate).unwrap();
            messages.push(p.begin(x, predicate).unwrap());
        }
        for message in messages.drain(..) {
            let to = message.header().to as usize;
            assert!(owners[to]
                .rotate(message)
                .unwrap()
                .iter()
                .all(Option::is_none));
        }
        let mut negative = owners.each_mut().map(|p| p.take_output().unwrap());
        let output = std::array::from_fn::<_, 3, _>(|i| {
            owners[i]
                .affine(
                    &mut original[i],
                    Some(&mut negative[i]),
                    Affine {
                        left: 1,
                        right: -1,
                        constant: 0,
                    },
                    InputBounds {
                        left: range,
                        right: range,
                    },
                )
                .unwrap()
                .0
        });
        assert_eq!(
            clear(&output),
            values.iter().map(|&v| v.max(0)).collect::<Vec<_>>()
        );
        drop(output);
        drop(negative);
        drop(original);
        assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
    }
}

#[test]
fn local_boolean_xor_preserves_source_public_injection_and_random_padding() {
    for bits in [0, 1, 8, 9, 16, 17, 32, 33, 64] {
        let mut owners = parties(PrssLimits::default());
        let mask = if bits == 0 {
            0
        } else {
            u64::MAX >> (64 - bits)
        };
        let operation =
            Operation::new(owners[0].context(), 0, Party::P0, 3, 1, Limits::default()).unwrap();
        let values = [0, mask, mask / 2];
        let shared = |reverse: bool| {
            let a = [0xa5_u64; 3];
            let b = [0x39_u64; 3];
            let c = values.map(|v| (if reverse { v ^ mask } else { v }) ^ 0xa5 ^ 0x39);
            let rows = [&a[..], &b[..], &c[..]];
            [Party::P0, Party::P1, Party::P2].map(|p| boolean::Share {
                value: ReplicatedShare::from_local(
                    operation,
                    p,
                    rows[p as usize],
                    rows[p.next() as usize],
                    Limits::default(),
                )
                .unwrap(),
                bits,
            })
        };
        let mut left = shared(false);
        let mut right = shared(true);
        let public = 1_u64 << bits.min(63);
        let output = std::array::from_fn::<_, 3, _>(|i| {
            owners[i]
                .boolean_xor(&mut left[i], Some(&mut right[i]), public)
                .unwrap()
        });
        for i in 0..3 {
            assert_eq!(output[i].second(), output[(i + 1) % 3].first());
            assert_eq!(output[i].bits(), (bits + 1).min(64));
            for lane in 0..3 {
                assert_eq!(
                    output[i].first()[lane],
                    left[i].first()[lane] ^ right[i].first()[lane] ^ public
                );
            }
            assert_eq!(owners[i].cursors(), Default::default());
        }
        for lane in 0..3 {
            assert_eq!(
                output.iter().fold(0, |v, p| v ^ p.first()[lane]),
                mask ^ public
            );
        }
        drop(output);
        drop(left);
        drop(right);
        assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
    }
}
