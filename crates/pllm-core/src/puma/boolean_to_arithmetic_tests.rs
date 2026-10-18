use super::*;
use crate::puma::prss::{multiply::tests::parties, PrssLimits};

fn inputs(owners: &[PartyRuntime; 3], values: &[u64], bits: u8) -> [Share; 3] {
    let op = Operation::new(
        owners[0].context(),
        0,
        Party::P0,
        values.len(),
        1,
        Limits::default(),
    )
    .unwrap();
    let a = vec![0xa5; values.len()];
    let b = vec![0x33; values.len()];
    let c: Vec<_> = values.iter().map(|v| v ^ 0xa5 ^ 0x33).collect();
    let parts = [&a[..], &b[..], &c[..]];
    [Party::P0, Party::P1, Party::P2].map(|party| Share {
        value: ReplicatedShare::from_local(
            op,
            party,
            parts[party as usize],
            parts[party.next() as usize],
            Limits::default(),
        )
        .unwrap(),
        bits,
    })
}

fn take(frames: &mut Vec<Message>, phase: MessagePhase) -> Message {
    let at = frames
        .iter()
        .position(|m| m.header().phase == phase)
        .unwrap();
    frames.remove(at)
}

fn begin(owners: &mut [PartyRuntime; 3], input: [Share; 3]) -> (Plan, Vec<Message>) {
    let plans = std::array::from_fn::<_, 3, _>(|i| {
        owners[i].plan_boolean_to_arithmetic(&input[i]).unwrap()
    });
    assert_eq!(plans, [plans[0]; 3]);
    let frames = owners
        .iter_mut()
        .zip(input)
        .flat_map(|(p, x)| {
            p.begin_boolean_to_arithmetic(x)
                .unwrap()
                .into_iter()
                .flatten()
        })
        .collect();
    (plans[0], frames)
}

fn run(
    owners: &mut [PartyRuntime; 3],
    values: &[u64],
    bits: u8,
) -> ([ReplicatedShare; 3], [u8; 32], Plan) {
    let input = inputs(owners, values, bits);
    let (plan, mut frames) = begin(owners, input);
    let mut digest = Sha256::new();
    let mut retained = Vec::new();
    let mut import = |original: Message| {
        for word in original.words() {
            digest.update(word.to_le_bytes());
        }
        let received = plan
            .message_from_received(original.header(), original.words(), Limits::default())
            .unwrap();
        retained.push(original);
        received
    };
    if bits > 0 {
        let messages = [
            MessagePhase::BooleanOtMask,
            MessagePhase::BooleanOtZero,
            MessagePhase::BooleanOtOne,
        ]
        .map(|phase| Some(import(take(&mut frames, phase))));
        let reply = owners[plan.receiver() as usize]
            .receive_boolean_to_arithmetic(messages)
            .unwrap()
            .unwrap();
        let received = import(reply);
        assert!(owners[plan.helper() as usize]
            .receive_boolean_to_arithmetic([Some(received), None, None])
            .unwrap()
            .is_none());
    }
    assert!(frames.is_empty());
    let output = owners.each_mut().map(|p| p.take_output().unwrap());
    for s in &output {
        for v in s.first() {
            digest.update(v.to_le_bytes());
        }
    }
    for i in 0..3 {
        assert_eq!(output[i].second(), output[(i + 1) % 3].first());
    }
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(
            output
                .iter()
                .fold(0_u64, |a, s| a.wrapping_add(s.first()[i])),
            v
        );
    }
    for p in owners.iter() {
        assert!(p.payload_usage().peak_bytes() + FIXED_RESERVE_BYTES <= plan.admitted_bytes());
    }
    drop(retained);
    (output, digest.finalize().into(), plan)
}

#[test]
fn all_ot_widths_and_maximum_tiles_preserve_values_and_source_cursors() {
    for bits in 0..=8 {
        for n in [1, 3, MAX_LANES / usize::from(bits.max(1))] {
            let mut owners = parties(PrssLimits::default());
            for offset in [0, 128] {
                let values: Vec<_> = (0..n)
                    .map(|i| (i as u64 + offset) & ((1 << bits) - 1))
                    .collect();
                let before = owners.each_ref().map(|p| p.cursors());
                let (output, _, plan) = run(&mut owners, &values, bits);
                let q = (n as u64 * u64::from(bits)).div_ceil(2);
                for (i, p) in owners.iter().enumerate() {
                    let after = p.cursors();
                    assert_eq!(after.pair_blocks - before[i].pair_blocks, 3 * q);
                    assert_eq!(
                        after.public_blocks - before[i].public_blocks,
                        u64::from(bits > 0)
                    );
                    let aes = if bits == 0 {
                        0
                    } else {
                        1 + if p.party() == plan.receiver() {
                            2 * q
                        } else {
                            4 * q
                        }
                    };
                    assert_eq!(after.aes_work - before[i].aes_work, aes);
                    assert_eq!(after.private_blocks, before[i].private_blocks);
                    assert_eq!(p.status(), Status::Ready);
                }
                drop(output);
                assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
            }
        }
    }
}

#[test]
fn conversion_rejects_unsupported_width_work_and_geometry_before_public_draw() {
    for (bits, n, work) in [(65, 1, 1000), (8, MAX_LANES / 8 + 1, 10000), (1, 3, 2)] {
        let mut owners = parties(PrssLimits::new(Limits::default(), work).unwrap());
        let x = inputs(&owners, &vec![0; n], bits);
        let before = owners[0].cursors();
        assert!(owners[0].plan_boolean_to_arithmetic(&x[0]).is_err());
        assert_eq!(owners[0].cursors(), before);
        assert_eq!(owners[0].status(), Status::Retired);
    }
}

#[test]
fn wrong_ot_phase_duplicate_frame_and_early_output_retire_without_opening() {
    for mode in 0..4 {
        let mut owners = parties(PrssLimits::default());
        let input = inputs(&owners, &[0, 1, 3], 2);
        let (plan, mut frames) = begin(&mut owners, input);
        let receiver = plan.receiver() as usize;
        let mut messages = [
            MessagePhase::BooleanOtMask,
            MessagePhase::BooleanOtZero,
            MessagePhase::BooleanOtOne,
        ]
        .map(|phase| Some(take(&mut frames, phase)));
        match mode {
            0 => messages[0].as_mut().unwrap().header.binding[0] ^= 1,
            1 => messages.swap(1, 2),
            2 => messages[0].as_mut().unwrap().header.to = plan.helper(),
            _ => {
                assert!(owners[receiver].take_output().is_err());
            }
        }
        assert!(owners[receiver]
            .receive_boolean_to_arithmetic(messages)
            .is_err());
        assert_eq!(owners[receiver].status(), Status::Retired);
        for p in &mut owners {
            p.cancel();
        }
        assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
    }
}

#[test]
fn accepted_ot_output_and_input_bindings_are_one_use() {
    let mut owners = parties(PrssLimits::default());
    let mut input = inputs(&owners, &[0, 1, 3], 2);
    owners[0].plan_boolean_to_arithmetic(&input[0]).unwrap();
    input[0].bits = 1;
    let [bad, _, _] = input;
    assert!(owners[0].begin_boolean_to_arithmetic(bad).is_err());
    assert_eq!(owners[0].cursors().pair_blocks, 0);
    let mut owners = parties(PrssLimits::default());
    drop(run(&mut owners, &[0, 1, 3], 2));
    for p in &mut owners {
        assert!(p.take_output().is_err());
        assert_eq!(p.status(), Status::Retired);
    }
}

#[test]
fn independent_ot_transcripts_match_mask_delivery_and_full_arithmetic_pairs() {
    let mut owners = parties(PrssLimits::default());
    for (bits, helper, end, expected) in [
        (
            0,
            0,
            0,
            "834a709ba2534ebe3ee1397fd4f7bd288b2acc1d20a08d6c862dcd99b6f04400",
        ),
        (
            1,
            2,
            6,
            "c9aace3e1031119812976b935fa7c65afc4946a8c09f2b9c9e9fe7f37c3c98fb",
        ),
        (
            4,
            0,
            24,
            "1b13d3dae62b3f0870479cc4e90c41b39f18befdf5bc1f3f171e486fcbeb3806",
        ),
        (
            8,
            0,
            60,
            "84847a2cda0df6efffa74cdd9c8d47bb3a92179fa8b82fb59c8ae89efc9ac966",
        ),
    ] {
        let maximum = (1_u64 << bits) - 1;
        let (output, digest, plan) = run(&mut owners, &[0, maximum, maximum / 2], bits);
        let actual: String = digest.iter().map(|v| format!("{v:02x}")).collect();
        assert_eq!(actual, expected);
        assert_eq!(plan.helper() as usize, helper);
        assert!(owners.iter().all(|p| p.cursors().pair_blocks == end));
        drop(output);
    }
}

fn run_ppa(
    owners: &mut [PartyRuntime; 3],
    values: &[u64],
    bits: u8,
) -> ([ReplicatedShare; 3], [u8; 32], Plan) {
    let input = inputs(owners, values, bits);
    let (plan, mut messages) = begin(owners, input);
    let mut digest = Sha256::new();
    let mut retained = Vec::new();
    let mut import = |original: Message| {
        for word in original.words() {
            digest.update(word.to_le_bytes());
        }
        let received = plan
            .message_from_received(original.header(), original.words(), Limits::default())
            .unwrap();
        retained.push(original);
        received
    };
    for round in 0..8 {
        let phase = if round == 0 {
            MessagePhase::BooleanPpaMask
        } else {
            MessagePhase::BooleanAnd
        };
        let mut next = Vec::new();
        // Record sender order, then route to the previous role.
        messages.sort_by_key(|m| m.header().from as u8);
        for message in messages.drain(..) {
            assert_eq!(message.header().phase, phase);
            let to = message.header().to as usize;
            next.extend(
                owners[to]
                    .advance_boolean_to_arithmetic(import(message))
                    .unwrap()
                    .into_iter()
                    .flatten(),
            );
        }
        messages = next;
    }
    let reveal = take(&mut messages, MessagePhase::BooleanPpaReveal);
    messages.extend(
        owners[0]
            .advance_boolean_to_arithmetic(import(reveal))
            .unwrap()
            .into_iter()
            .flatten(),
    );
    messages.sort_by_key(|m| m.header().from as u8);
    for message in messages.drain(..) {
        let to = message.header().to as usize;
        assert!(owners[to]
            .advance_boolean_to_arithmetic(import(message))
            .unwrap()
            .iter()
            .all(Option::is_none));
    }
    let output = owners.each_mut().map(|p| p.take_output().unwrap());
    for i in 0..3 {
        assert_eq!(output[i].second(), output[(i + 1) % 3].first());
        for v in output[i].first() {
            digest.update(v.to_le_bytes());
        }
    }
    for (i, &value) in values.iter().enumerate() {
        assert_eq!(
            output
                .iter()
                .fold(0_u64, |x, p| x.wrapping_add(p.first()[i])),
            value
        );
    }
    for p in owners.iter() {
        assert!(p.payload_usage().peak_bytes() + FIXED_RESERVE_BYTES <= plan.admitted_bytes());
    }
    assert_eq!(retained.len(), 28);
    assert_eq!(
        retained.iter().map(|m| m.words().len()).sum::<usize>(),
        46 * values.len()
    );
    drop(retained);
    (output, digest.finalize().into(), plan)
}

#[test]
fn full_width_ppa_preserves_all_bits_with_only_masked_reveal() {
    for bits in [9, 16, 17, 32, 33, 64] {
        for n in [1, 3, 17, MAX_LANES / 2] {
            let mut owners = parties(PrssLimits::default());
            let mask = u64::MAX >> (64 - bits);
            let values: Vec<_> = (0..n)
                .map(|i| match i % 4 {
                    0 => mask,
                    1 => 0,
                    2 => mask / 2,
                    _ => (i as u64).wrapping_mul(0xfeed_cafe_1234_9876) & mask,
                })
                .collect();
            let (output, _, plan) = run_ppa(&mut owners, &values, bits);
            let expected = 3 * (n as u64).div_ceil(2) + 6 * n as u64;
            for p in &owners {
                assert_eq!(p.cursors().pair_blocks, expected);
                assert_eq!(p.cursors().aes_work, 2 * expected);
                assert_eq!(p.cursors().public_blocks, 0);
                assert_eq!(p.cursors().private_blocks, 0);
            }
            assert_eq!(plan.pair_blocks(), expected);
            drop(output);
            assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
        }
    }
}

#[test]
fn ppa_phase_replay_and_incomplete_reveal_retire_at_each_rotation() {
    for stop in 0..9 {
        let mut owners = parties(PrssLimits::default());
        let input = inputs(&owners, &[0, u64::MAX, i64::MIN as u64], 64);
        let (plan, mut messages) = begin(&mut owners, input);
        for _ in 0..stop.min(8) {
            let mut next = Vec::new();
            for message in messages.drain(..) {
                let to = message.header().to as usize;
                next.extend(
                    owners[to]
                        .advance_boolean_to_arithmetic(message)
                        .unwrap()
                        .into_iter()
                        .flatten(),
                );
            }
            messages = next;
        }
        let phase = match stop {
            0 => MessagePhase::BooleanPpaMask,
            1..=7 => MessagePhase::BooleanAnd,
            _ => MessagePhase::BooleanPpaReveal,
        };
        let at = messages
            .iter()
            .position(|m| m.header().phase == phase)
            .unwrap();
        let mut bad = messages.remove(at);
        let role = bad.header().to as usize;
        let original = plan
            .message_from_received(bad.header(), bad.words(), Limits::default())
            .unwrap();
        if stop == 8 {
            let more = owners[role]
                .advance_boolean_to_arithmetic(original)
                .unwrap();
            drop(more);
        } else {
            bad.header.binding[0] ^= 1;
            drop(original);
        }
        assert!(owners[role].advance_boolean_to_arithmetic(bad).is_err());
        for p in &mut owners {
            p.cancel();
        }
        drop(messages);
        assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
    }
}

#[test]
fn independent_ppa_transcripts_match_mask_rotation_carry_reveal_and_output() {
    let mut owners = parties(PrssLimits::default());
    for (bits, end, expected) in [
        (
            9,
            24,
            "b26b6079ce6ab42e04922f9b9143227932eeca07a117892fc05808e5b147b0e8",
        ),
        (
            17,
            48,
            "8befc346b398e639e9162155d5629f977889c10991065ca2731e073a1e977bc4",
        ),
        (
            64,
            72,
            "93dd2200dfd7f948ef4f9f093fb2b0df3e861988cfc0e96090aa309354826ae4",
        ),
    ] {
        let maximum = u64::MAX >> (64 - bits);
        let (output, digest, _) = run_ppa(&mut owners, &[0, maximum, maximum / 2], bits);
        let actual: String = digest.iter().map(|v| format!("{v:02x}")).collect();
        assert_eq!(actual, expected);
        assert!(owners.iter().all(|p| p.cursors().pair_blocks == end));
        drop(output);
    }
}
