//! Production-exported API with OS-generated party-local seeds. Aggregate
//! reconstruction/routing below is test-only, never a runtime/dealer API.
use pllm_core::puma::{
    multiply::*, prss::PrssLimits, Context, Limits, Message, MessagePhase, Party, ReplicatedShare,
    MAX_LANES,
};

fn owners(context: Context) -> [PartyRuntime; 3] {
    let mut setup = [Party::P0, Party::P1, Party::P2]
        .map(|p| SetupParty::new(context, p, PrssLimits::default()).unwrap());
    let public = setup.each_mut().map(|p| p.contribution().unwrap());
    for (i, p) in setup.iter_mut().enumerate() {
        for (j, contribution) in public.iter().enumerate() {
            if i != j {
                p.receive_public(*contribution).unwrap();
            }
        }
    }
    let mut seeds = setup.each_mut().map(|p| Some(p.share_seed().unwrap()));
    for (i, p) in setup.iter_mut().enumerate() {
        p.receive_seed(seeds[(i + 1) % 3].take().unwrap()).unwrap();
    }
    setup.each_mut().map(|p| p.finish().unwrap())
}

fn shares(op: Plan, operand: Operand, x: &[i64]) -> [ReplicatedShare; 3] {
    let a = vec![u64::MAX - 31; x.len()];
    let b = vec![u64::MAX / 2 + 7; x.len()];
    let c: Vec<_> = x
        .iter()
        .map(|&v| (v as u64).wrapping_sub(a[0]).wrapping_sub(b[0]))
        .collect();
    let parts = [&a[..], &b[..], &c[..]];
    [Party::P0, Party::P1, Party::P2].map(|p| {
        op.share_operand(
            operand,
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

#[test]
fn os_seeded_full_shape_products_obey_rounding_law_and_source_work_counts() {
    let context = Context::new([11; 32], [22; 32], 7).unwrap();
    let mut owners = owners(context);
    let usages = owners.each_ref().map(|p| p.payload_usage());
    let mut pair_blocks = 0;
    let mut public_blocks = 0;
    let mut private = [0; 3];
    let mut work = [0; 3];
    let mut admitted = 0;
    let cases = [1, 3, 8, 17, MAX_LANES].map(|n| (n, None));
    let matrices = [[1, 3, 1], [2, 3, 4], [32, 32, 32]].map(|[m, k, n]| {
        let shape = MatrixShape::new(m, k, n).unwrap();
        (shape.output_elements(), Some(shape))
    });
    for (ordinal, (n, shape)) in cases.into_iter().chain(matrices).enumerate() {
        let bounds = vec![
            InputBounds {
                left: Interval::new(-1 << 20, 1 << 20).unwrap(),
                right: Interval::new(-1 << 20, 1 << 20).unwrap(),
            };
            n
        ];
        let x: Vec<_> = (0..shape.map_or(n, MatrixShape::left_elements))
            .map(|i| (i as i64 - 515) * 113)
            .collect();
        let y: Vec<_> = (0..shape.map_or(n, MatrixShape::right_elements))
            .map(|i| (i as i64 - 127) * 79)
            .collect();
        let ops = owners.each_mut().map(|p| match shape {
            Some(shape) => p.plan_matmul(shape, bounds[0]).unwrap(),
            None => p.plan(&bounds).unwrap(),
        });
        assert_eq!(ops, [ops[0]; 3]);
        let op = ops[0];
        assert_eq!(op.sequence(), 5 * ordinal as u64);
        admitted = admitted.max(op.admitted_bytes());
        let mut messages: Vec<_> = owners
            .iter_mut()
            .zip(shares(op, Operand::Left, &x))
            .zip(shares(op, Operand::Right, &y))
            .map(|((p, a), b)| p.begin(a, b).unwrap())
            .collect();
        let mut frame_count = messages.len();
        let mut payload_words: usize = messages.iter().map(|m| m.words().len()).sum();
        for p in &mut owners {
            let next = take(&mut messages, p.party(), MessagePhase::Multiply);
            let replies = p.rotate(next).unwrap();
            for m in replies.into_iter().flatten() {
                frame_count += 1;
                payload_words += m.words().len();
                messages.push(m);
            }
        }
        let op = owners[0].truncation_operation().unwrap();
        assert!(owners.iter().all(|p| p.truncation_operation() == Some(op)));
        for p in [op.evaluator_a(), op.evaluator_b()] {
            let peer = take(&mut messages, p, MessagePhase::MaskedInput);
            let correlation = take(&mut messages, p, MessagePhase::Correlation);
            let reply = owners[p as usize].open(peer, correlation).unwrap();
            frame_count += 1;
            payload_words += reply.words().len();
            messages.push(reply);
        }
        for p in [op.evaluator_a(), op.evaluator_b()] {
            owners[p as usize]
                .reshare(take(&mut messages, p, MessagePhase::Reshare))
                .unwrap();
        }
        assert!(messages.is_empty());
        assert_eq!(frame_count, 9);
        assert_eq!(payload_words, 11 * n);
        let outputs = owners.each_mut().map(|p| p.take_output().unwrap());
        for p in 0..3 {
            assert_eq!(outputs[p].second(), outputs[(p + 1) % 3].first());
        }
        for lane in 0..n {
            let actual = outputs
                .iter()
                .fold(0_u64, |s, p| s.wrapping_add(p.first()[lane]))
                as i64;
            let product = match shape {
                None => i128::from(x[lane]) * i128::from(y[lane]),
                Some(shape) => {
                    let [_, inner, columns] = shape.dimensions();
                    (0..inner)
                        .map(|k| {
                            i128::from(x[lane / columns * inner + k])
                                * i128::from(y[k * columns + lane % columns])
                        })
                        .sum()
                }
            };
            let floor = product.div_euclid(1 << 18) as i64;
            let ceil = -(-product).div_euclid(1 << 18) as i64;
            assert!(actual == floor || actual == ceil);
        }
        let q = (n as u64).div_ceil(2);
        pair_blocks += 3 * q;
        public_blocks += 1;
        private[op.helper() as usize] += n as u64;
        for (i, p) in owners.iter().enumerate() {
            work[i] += if i == op.helper() as usize {
                6 * q + n as u64 + 1
            } else {
                4 * q + 1
            };
            assert_eq!(p.cursors().public_blocks, public_blocks);
            assert_eq!(p.cursors().pair_blocks, pair_blocks);
            assert_eq!(p.cursors().private_blocks, private[i]);
            assert_eq!(p.cursors().aes_work, work[i]);
            assert_eq!(p.status(), Status::Ready);
            assert!(usages[i].peak_bytes() <= admitted);
        }
        drop(outputs);
        assert!(usages.iter().all(|u| u.live_bytes() == 0));

        // A Boolean kernel shares the SAME PRSS stream between arithmetic
        // operations. It consumes narrow typed words and no public/private draw.
        let plans = owners.each_mut().map(|p| p.plan_and(n, 14, 5).unwrap());
        let plan = plans[0];
        assert_eq!(plans, [plan; 3]);
        assert_eq!(plan.sequence(), 5 * ordinal as u64 + 1);
        let inputs = [Operand::Left, Operand::Right].map(|operand| {
            let value = if operand == Operand::Left {
                0x1234
            } else {
                0x17
            };
            let components = [vec![0x81; n], vec![0x23; n], vec![value ^ 0x81 ^ 0x23; n]];
            [Party::P0, Party::P1, Party::P2].map(|p| {
                plan.share_operand(
                    operand,
                    p,
                    &components[p as usize],
                    &components[p.next() as usize],
                    Limits::default(),
                )
                .unwrap()
            })
        });
        let [left, right] = inputs;
        let mut messages: Vec<_> = owners
            .iter_mut()
            .zip(left)
            .zip(right)
            .map(|((p, x), y)| p.begin_and(x, y).unwrap())
            .collect();
        for p in &mut owners {
            let original = take(&mut messages, p.party(), MessagePhase::BooleanAnd);
            let received = plan
                .message_from_received(original.header(), original.words(), Limits::default())
                .unwrap();
            drop(original);
            p.receive_and(received).unwrap();
        }
        let outputs = owners.each_mut().map(|p| p.take_boolean_output().unwrap());
        for i in 0..n {
            assert_eq!(
                outputs[0].first()[i] ^ outputs[1].first()[i] ^ outputs[2].first()[i],
                0x1234 & 0x17
            );
        }
        let q = (2 * n as u64).div_ceil(16);
        pair_blocks += q;
        for (i, p) in owners.iter().enumerate() {
            work[i] += 2 * q;
            assert_eq!(p.cursors().pair_blocks, pair_blocks);
            assert_eq!(p.cursors().aes_work, work[i]);
            assert_eq!(p.cursors().public_blocks, public_blocks);
            assert_eq!(p.cursors().private_blocks, private[i]);
        }
        drop(outputs);
        assert!(usages.iter().all(|u| u.live_bytes() == 0));

        for kind in [conversion::Kind::Bits, conversion::Kind::MostSignificantBit] {
            let lanes = n.min(MAX_LANES / 2);
            let plans = owners
                .each_mut()
                .map(|p| p.plan_conversion(lanes, kind).unwrap());
            let plan = plans[0];
            assert_eq!(plans, [plan; 3]);
            admitted = admitted.max(plan.admitted_bytes());
            let values: Vec<_> = (0..lanes)
                .map(|i| (i as u64).wrapping_mul(0xfedc_ba98_7654_3211))
                .collect();
            let components = [
                vec![u64::MAX; lanes],
                vec![91; lanes],
                values.iter().map(|v| v.wrapping_sub(90)).collect(),
            ];
            let input = [Party::P0, Party::P1, Party::P2].map(|p| {
                plan.share_input(
                    p,
                    &components[p as usize],
                    &components[p.next() as usize],
                    Limits::default(),
                )
                .unwrap()
            });
            let mut messages: Vec<_> = owners
                .iter_mut()
                .zip(input)
                .map(|(p, x)| p.begin_conversion(x).unwrap())
                .collect();
            let mut body_words = 0;
            for round in 0..plan.rotations() {
                body_words += messages.iter().map(|m| m.words().len()).sum::<usize>();
                let mut next = Vec::new();
                for p in &mut owners {
                    let phase = if round == 0 {
                        MessagePhase::ArithmeticToBoolean
                    } else {
                        MessagePhase::BooleanAnd
                    };
                    let original = take(&mut messages, p.party(), phase);
                    let received = plan
                        .message_from_received(
                            round as u8,
                            original.header(),
                            original.words(),
                            Limits::default(),
                        )
                        .unwrap();
                    drop(original);
                    if let Some(message) = p.advance_conversion(received).unwrap() {
                        next.push(message);
                    }
                }
                messages = next;
            }
            assert!(messages.is_empty());
            assert_eq!(body_words, 42 * lanes);
            let output = owners.each_mut().map(|p| p.take_boolean_output().unwrap());
            for (lane, &value) in values.iter().enumerate() {
                assert_eq!(
                    output.iter().fold(0, |v, p| v ^ p.first()[lane]),
                    if kind == conversion::Kind::Bits {
                        value
                    } else {
                        value >> 63
                    }
                );
            }
            pair_blocks += plan.pair_blocks();
            for (i, p) in owners.iter().enumerate() {
                work[i] += 2 * plan.pair_blocks();
                assert_eq!(p.cursors().pair_blocks, pair_blocks);
                assert_eq!(p.cursors().aes_work, work[i]);
                assert_eq!(p.cursors().public_blocks, public_blocks);
                assert_eq!(p.cursors().private_blocks, private[i]);
            }
            if kind == conversion::Kind::MostSignificantBit {
                // Direct opaque-owner conversion; no caller reconstructs or
                // copies a predicate between the two protected operations.
                let plans = std::array::from_fn::<_, 3, _>(|i| {
                    owners[i].plan_boolean_to_arithmetic(&output[i]).unwrap()
                });
                let plan = plans[0];
                assert_eq!(plans, [plan; 3]);
                admitted = admitted.max(plan.admitted_bytes());
                let mut messages: Vec<_> = owners
                    .iter_mut()
                    .zip(output)
                    .flat_map(|(p, x)| {
                        p.begin_boolean_to_arithmetic(x)
                            .unwrap()
                            .into_iter()
                            .flatten()
                    })
                    .collect();
                let incoming = [
                    MessagePhase::BooleanOtMask,
                    MessagePhase::BooleanOtZero,
                    MessagePhase::BooleanOtOne,
                ]
                .map(|phase| {
                    let original = take(&mut messages, plan.receiver(), phase);
                    let received = plan
                        .message_from_received(
                            original.header(),
                            original.words(),
                            Limits::default(),
                        )
                        .unwrap();
                    drop(original);
                    Some(received)
                });
                let reply = owners[plan.receiver() as usize]
                    .receive_boolean_to_arithmetic(incoming)
                    .unwrap()
                    .unwrap();
                let received = plan
                    .message_from_received(reply.header(), reply.words(), Limits::default())
                    .unwrap();
                drop(reply);
                owners[plan.helper() as usize]
                    .receive_boolean_to_arithmetic([Some(received), None, None])
                    .unwrap();
                assert!(messages.is_empty());
                let output = owners.each_mut().map(|p| p.take_output().unwrap());
                for (lane, &value) in values.iter().enumerate() {
                    assert_eq!(
                        output
                            .iter()
                            .fold(0_u64, |v, p| v.wrapping_add(p.first()[lane])),
                        value >> 63
                    );
                }
                let q = (lanes as u64).div_ceil(2);
                pair_blocks += 3 * q;
                public_blocks += 1;
                for (i, p) in owners.iter().enumerate() {
                    work[i] += 1 + if p.party() == plan.receiver() {
                        2 * q
                    } else {
                        4 * q
                    };
                    assert_eq!(p.cursors().pair_blocks, pair_blocks);
                    assert_eq!(p.cursors().public_blocks, public_blocks);
                    assert_eq!(p.cursors().private_blocks, private[i]);
                    assert_eq!(p.cursors().aes_work, work[i]);
                }
                drop(output);
            } else {
                drop(output);
            }
            assert!(usages.iter().all(|u| u.live_bytes() == 0));
        }
    }
}

#[test]
fn fresh_setup_rejects_old_operands_and_cancellation_has_no_restart() {
    let context = Context::new([11; 32], [22; 32], 7).unwrap();
    let bounds = [InputBounds {
        left: Interval::new(-1, 1).unwrap(),
        right: Interval::new(-1, 1).unwrap(),
    }];
    let mut old = owners(context);
    let old_op = old[0].plan(&bounds).unwrap();
    let mut fresh = owners(context);
    let new_op = fresh[0].plan(&bounds).unwrap();
    assert_ne!(old_op.digest(), new_op.digest());
    let a = shares(old_op, Operand::Left, &[1])
        .into_iter()
        .next()
        .unwrap();
    let b = shares(old_op, Operand::Right, &[1])
        .into_iter()
        .next()
        .unwrap();
    assert!(fresh[0].begin(a, b).is_err());
    assert_eq!(fresh[0].status(), Status::Retired);
    for p in &mut old {
        p.cancel();
        assert!(p.plan(&bounds).is_err());
    }
}
