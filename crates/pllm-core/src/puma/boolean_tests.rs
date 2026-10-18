use super::*;
use crate::puma::prss::multiply::tests::parties;
use crate::puma::prss::PrssLimits;

fn value_mask(bits: u8) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

fn shares(plan: Plan, operand: Operand) -> ([Share; 3], Vec<u64>) {
    let right = operand == Operand::Right;
    let bits = if right {
        plan.right_bits
    } else {
        plan.left_bits
    };
    let mask = word_mask(word_bytes(bits).unwrap());
    let seq = plan.sequence();
    let values: Vec<_> = (0..plan.lanes())
        .map(|i| {
            (if right {
                i as u64 * 0x10203 + 17
            } else {
                (i as u64 + 1) * 0x9e3779b9
            }) & value_mask(bits)
        })
        .collect();
    let a: Vec<_> = (0..plan.lanes())
        .map(|i| ((if right { 0x12345678 } else { 0x89abcdef }) + i as u64 + 13 * seq) & mask)
        .collect();
    let b: Vec<_> = (0..plan.lanes())
        .map(|i| ((if right { 0x87654321 } else { 0xfedcba98 }) - i as u64 + 7 * seq) & mask)
        .collect();
    let c: Vec<_> = values
        .iter()
        .zip(&a)
        .zip(&b)
        .map(|((&v, &a), &b)| v ^ a ^ b)
        .collect();
    let components = [&a[..], &b[..], &c[..]];
    let shares = [Party::P0, Party::P1, Party::P2].map(|party| {
        plan.share_operand(
            operand,
            party,
            components[party as usize],
            components[party.next() as usize],
            Limits::default(),
        )
        .unwrap()
    });
    (shares, values)
}

fn begin(owners: &mut [PartyRuntime; 3], plan: Plan) -> (Vec<Message>, Vec<u64>) {
    let (left, x) = shares(plan, Operand::Left);
    let (right, y) = shares(plan, Operand::Right);
    let messages = owners
        .iter_mut()
        .zip(left)
        .zip(right)
        .map(|((p, x), y)| p.begin_and(x, y).unwrap())
        .collect();
    (messages, x.iter().zip(y).map(|(&x, y)| x & y).collect())
}

fn finish(owners: &mut [PartyRuntime; 3], mut messages: Vec<Message>, expected: &[u64]) {
    for owner in owners.iter_mut() {
        let i = messages
            .iter()
            .position(|m| m.header().to == owner.party())
            .unwrap();
        owner.receive_and(messages.swap_remove(i)).unwrap();
        assert_eq!(owner.status(), Status::Complete);
    }
    let output = owners.each_mut().map(|p| p.take_boolean_output().unwrap());
    for i in 0..3 {
        assert_eq!(output[i].second(), output[(i + 1) % 3].first());
    }
    for (i, &value) in expected.iter().enumerate() {
        assert_eq!(
            output[0].first()[i] ^ output[1].first()[i] ^ output[2].first()[i],
            value
        );
    }
    assert!(owners.iter().all(|p| p.status() == Status::Ready));
    drop(output);
    assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
}

#[test]
fn independently_frozen_typed_aes_transcripts_match_each_word_width() {
    // Independent Python AES reference + literal XOR equations, seed fixture
    // keys 11/22/33. Hash each sender's complete u64-encoded rotation in order.
    let cases = [
        (
            0,
            1,
            3,
            "d7a29a8832c467364dd0de07f9b4705500b4a3857583d75808e6b705502bcdbd",
        ),
        (
            9,
            8,
            9,
            "89c04bcf6a19de44bbf89222e119d4a939cc541f73d4cc28e2bc2d6be6c7d647",
        ),
        (
            17,
            16,
            5,
            "81d204c6a02f85e84aaa3a590e656eedf82a20c5ceb4554947a35aaac593ec22",
        ),
        (
            33,
            64,
            3,
            "3833690eb4f33adc9543b95e33d79f231cf19e9c5da90156a93c7fbe2e5ceb13",
        ),
    ];
    let mut owners = parties(PrssLimits::default());
    let mut pair_blocks = 0;
    for (ordinal, (left, right, n, expected_hash)) in cases.into_iter().enumerate() {
        let before = owners.each_ref().map(|p| p.cursors());
        let plans = owners
            .each_mut()
            .map(|p| p.plan_and(n, left, right).unwrap());
        assert_eq!(plans, [plans[0]; 3]);
        let plan = plans[0];
        assert_eq!(plan.sequence(), ordinal as u64);
        assert_eq!(owners.each_ref().map(|p| p.cursors()), before);
        let (messages, expected) = begin(&mut owners, plan);
        let mut hash = Sha256::new();
        for (i, message) in messages.iter().enumerate() {
            assert_eq!(message.header().from as usize, i);
            assert_eq!(message.header().to as usize, (i + 2) % 3);
            for word in message.words() {
                hash.update(word.to_le_bytes());
            }
        }
        assert_eq!(format!("{:x}", hash.finalize()), expected_hash);
        pair_blocks += (n as u64 * plan.bytes as u64).div_ceil(16);
        for owner in &owners {
            assert_eq!(owner.cursors().pair_blocks, pair_blocks);
            assert_eq!(owner.cursors().aes_work, 2 * pair_blocks);
            assert_eq!(owner.cursors().public_blocks, 0);
            assert_eq!(owner.cursors().private_blocks, 0);
        }
        finish(&mut owners, messages, &expected);
    }
}

#[test]
fn all_nominal_widths_and_maximum_lanes_preserve_complete_boolean_outputs() {
    let mut owners = parties(PrssLimits::default());
    for bits in 0..=64 {
        let n = [1, 3, 17, MAX_LANES][bits as usize % 4];
        let plans = owners
            .each_mut()
            .map(|p| p.plan_and(n, bits, 64 - bits).unwrap());
        let (messages, expected) = begin(&mut owners, plans[0]);
        finish(&mut owners, messages, &expected);
    }
}

#[test]
fn malformed_or_substituted_boolean_inputs_retire_the_common_owner() {
    for mismatch in 0..7 {
        let mut owners = parties(PrssLimits::default());
        let plans = owners.each_mut().map(|p| p.plan_and(3, 8, 8).unwrap());
        let before = owners[0].cursors();
        let (mut left, _) = shares(plans[0], Operand::Left);
        let (mut right, _) = shares(plans[0], Operand::Right);
        match mismatch {
            0 => {
                left.swap(0, 1);
            }
            1 => {
                std::mem::swap(&mut left[0], &mut right[0]);
            }
            2 => {
                left[0].bits = 7;
            }
            3 => {
                left[0].value.operation.sequence = 1;
            }
            4 => {
                left[0].value.operation.context.session = [77; 32];
            }
            5 => {
                left[0].value.operation.consumer = None;
            }
            6 => {
                left[0].value.operation.context.schedule = [88; 32];
            }
            _ => unreachable!(),
        }
        let [left, _, _] = left;
        let [right, _, _] = right;
        assert!(owners[0].begin_and(left, right).is_err());
        assert_eq!(owners[0].cursors(), before);
        assert_eq!(owners[0].status(), Status::Retired);
        assert!(owners[0].source.seeds.is_none());
    }
}

#[test]
fn malformed_frames_replay_and_wrong_output_extraction_retire() {
    for mismatch in 0..7 {
        let mut owners = parties(PrssLimits::default());
        let plans = owners.each_mut().map(|p| p.plan_and(3, 8, 8).unwrap());
        let (mut messages, _) = begin(&mut owners, plans[0]);
        let mut message = messages.remove(1); // P1 -> P0
        match mismatch {
            0 => {
                message.header.binding = [1; 32];
            }
            1 => {
                message.header.from = Party::P2;
            }
            2 => {
                message.header.to = Party::P1;
            }
            3 => {
                message.header.phase = MessagePhase::Multiply;
            }
            4 => {
                message.words[0] |= 1 << 8;
            }
            5 => {
                let replay = plans[0]
                    .message_from_received(message.header(), message.words(), Limits::default())
                    .unwrap();
                owners[0].receive_and(message).unwrap();
                drop(owners[0].take_boolean_output().unwrap());
                message = replay;
            }
            6 => {
                assert!(owners[0].take_output().is_err());
            }
            _ => unreachable!(),
        }
        assert!(owners[0].receive_and(message).is_err());
        assert_eq!(owners[0].status(), Status::Retired);
        for p in &mut owners {
            p.cancel();
        }
        drop(messages);
        assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
    }
}

#[test]
fn boolean_preflight_and_cancellation_preserve_the_payload_ceiling() {
    for (n, lhs, rhs) in [(0, 1, 1), (MAX_LANES + 1, 1, 1), (1, 65, 1), (1, 1, 65)] {
        let mut owners = parties(PrssLimits::default());
        let before = owners[0].cursors();
        assert!(owners[0].plan_and(n, lhs, rhs).is_err());
        assert_eq!(owners[0].cursors(), before);
        assert_eq!(owners[0].status(), Status::Retired);
    }
    let limits = PrssLimits::new(Limits::default(), 1).unwrap();
    let mut owners = parties(limits);
    assert!(owners[0].plan_and(1, 1, 1).is_err());
    assert_eq!(owners[0].cursors().aes_work, 0);
    for phase in 0..3 {
        let mut owners = parties(PrssLimits::default());
        let plans = owners
            .each_mut()
            .map(|p| p.plan_and(MAX_LANES, 1, 1).unwrap());
        let mut messages = Vec::new();
        if phase > 0 {
            messages = begin(&mut owners, plans[0]).0;
        }
        if phase == 2 {
            for p in &mut owners {
                let i = messages
                    .iter()
                    .position(|m| m.header().to == p.party())
                    .unwrap();
                p.receive_and(messages.swap_remove(i)).unwrap();
            }
        }
        let usage = owners.each_ref().map(|p| p.payload_usage());
        for p in &mut owners {
            p.cancel();
        }
        if phase == 1 {
            assert!(usage.iter().all(|u| u.live_bytes() > 0));
        }
        drop(messages);
        assert!(usage.iter().all(|u| u.live_bytes() == 0));
        assert!(usage
            .iter()
            .all(|u| u.peak_bytes() + FIXED_RESERVE_BYTES <= plans[0].admitted_bytes()));
    }
}
