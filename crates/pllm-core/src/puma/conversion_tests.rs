use super::*;
use crate::puma::prss::{multiply::tests::parties, PrssLimits};

fn inputs(plan: Plan, values: &[u64]) -> [ReplicatedShare; 3] {
    let first: Vec<_> = (0..values.len())
        .map(|i| 0xfedc_ba98_7654_3210_u64.wrapping_add(17 * i as u64))
        .collect();
    let second: Vec<_> = (0..values.len())
        .map(|i| 0x1234_5678_9abc_def0_u64.wrapping_sub(5 * i as u64))
        .collect();
    let third: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &x)| x.wrapping_sub(first[i]).wrapping_sub(second[i]))
        .collect();
    let components = [&first[..], &second[..], &third[..]];
    [Party::P0, Party::P1, Party::P2].map(|p| {
        plan.share_input(
            p,
            components[p as usize],
            components[p.next() as usize],
            Limits::default(),
        )
        .unwrap()
    })
}

fn begin(owners: &mut [PartyRuntime; 3], plan: Plan, values: &[u64]) -> Vec<Message> {
    owners
        .iter_mut()
        .zip(inputs(plan, values))
        .map(|(p, x)| p.begin_conversion(x).unwrap())
        .collect()
}

fn run(
    owners: &mut [PartyRuntime; 3],
    plan: Plan,
    values: &[u64],
    retain: bool,
) -> ([Share; 3], [u8; 32]) {
    let mut messages = begin(owners, plan, values);
    let mut transcript = Sha256::new();
    let mut retained = Vec::new();
    for round in 0..8 {
        // Sender order is deterministic; hash original words, not metadata.
        messages.sort_by_key(|m| m.header().from as u8);
        for m in &messages {
            for v in m.words() {
                transcript.update(v.to_le_bytes());
            }
        }
        let mut next = Vec::new();
        for p in owners.iter_mut() {
            let at = messages
                .iter()
                .position(|m| m.header().to == p.party())
                .unwrap();
            let original = messages.remove(at);
            let received = plan
                .message_from_received(
                    round,
                    original.header(),
                    original.words(),
                    Limits::default(),
                )
                .unwrap();
            if retain {
                retained.push(original);
            } else {
                drop(original);
            }
            if let Some(message) = p.advance_conversion(received).unwrap() {
                next.push(message);
            }
        }
        assert!(messages.is_empty());
        assert_eq!(next.len(), if round == 7 { 0 } else { 3 });
        messages = next;
    }
    let output = owners.each_mut().map(|p| p.take_boolean_output().unwrap());
    for i in 0..3 {
        assert_eq!(output[i].second(), output[(i + 1) % 3].first());
    }
    for (i, &x) in values.iter().enumerate() {
        assert_eq!(
            output.iter().fold(0, |x, s| x ^ s.first()[i]),
            if plan.kind() == Kind::Bits {
                x
            } else {
                x >> 63
            }
        );
    }
    for p in owners.iter() {
        assert_eq!(p.status(), Status::Ready);
        assert!(p.payload_usage().peak_bytes() + FIXED_RESERVE_BYTES <= plan.admitted_bytes());
    }
    drop(retained);
    (output, transcript.finalize().into())
}

#[test]
fn complete_conversion_and_sign_match_clear_ring64_without_reconstruction_in_roles() {
    for kind in [Kind::Bits, Kind::MostSignificantBit] {
        let mut owners = parties(PrssLimits::default());
        for n in [1, 3, 7, 16, MAX_LANES / 2] {
            let plans = owners
                .each_mut()
                .map(|p| p.plan_conversion(n, kind).unwrap());
            assert_eq!(plans, [plans[0]; 3]);
            let plan = plans[0];
            let values: Vec<_> = (0..n)
                .map(|i| match i % 8 {
                    0 => 0,
                    1 => 1,
                    2 => u64::MAX,
                    3 => i64::MAX as u64,
                    4 => i64::MIN as u64,
                    5 => 1 << 18,
                    6 => (1_u64 << 62) + i as u64,
                    _ => (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                })
                .collect();
            let before = owners[0].cursors();
            let (output, _) = run(&mut owners, plan, &values, true);
            for p in &owners {
                let after = p.cursors();
                assert_eq!(after.pair_blocks - before.pair_blocks, plan.pair_blocks());
                assert_eq!(after.aes_work - before.aes_work, 2 * plan.pair_blocks());
                assert_eq!(after.public_blocks, before.public_blocks);
                assert_eq!(after.private_blocks, before.private_blocks);
            }
            drop(output);
            assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
        }
    }
}

#[test]
fn every_circuit_phase_cancels_and_frame_substitution_retires() {
    for kind in [Kind::Bits, Kind::MostSignificantBit] {
        for stop in 0..8 {
            let mut owners = parties(PrssLimits::default());
            let plans = owners
                .each_mut()
                .map(|p| p.plan_conversion(3, kind).unwrap());
            let mut messages = begin(&mut owners, plans[0], &[0, u64::MAX, 1]);
            for _ in 0..stop {
                let mut next = Vec::new();
                for p in &mut owners {
                    let at = messages
                        .iter()
                        .position(|m| m.header().to == p.party())
                        .unwrap();
                    next.push(p.advance_conversion(messages.remove(at)).unwrap().unwrap());
                }
                messages = next;
            }
            let at = messages
                .iter()
                .position(|m| m.header().to == Party::P0)
                .unwrap();
            let mut malformed = messages.remove(at);
            malformed.header.binding[0] ^= 1;
            assert!(owners[0].advance_conversion(malformed).is_err());
            for p in &mut owners {
                p.cancel();
                assert_eq!(p.status(), Status::Retired);
                assert!(p.source.seeds.is_none());
                assert!(p.take_boolean_output().is_err());
            }
            drop(messages);
            assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
        }
    }
}

#[test]
fn all_round_work_and_payload_are_admitted_before_first_random_draw() {
    for kind in [Kind::Bits, Kind::MostSignificantBit] {
        for n in [0, MAX_LANES / 2 + 1, usize::MAX] {
            let mut owners = parties(PrssLimits::default());
            assert!(owners[0].plan_conversion(n, kind).is_err());
            assert_eq!(owners[0].cursors(), Default::default());
        }
        let mut owners = parties(PrssLimits::new(Limits::default(), 2).unwrap());
        assert!(owners[0].plan_conversion(1, kind).is_err());
        assert_eq!(owners[0].cursors(), Default::default());
    }
    assert!(
        std::mem::size_of::<PartyRuntime>() + std::mem::size_of::<Plan>() < FIXED_RESERVE_BYTES
    );
}

#[test]
fn kind_input_binding_and_replay_cannot_cross_conversion_boundaries() {
    let mut owners = parties(PrssLimits::default());
    let plans = owners
        .each_mut()
        .map(|p| p.plan_conversion(3, Kind::Bits).unwrap());
    let mut input = inputs(plans[0], &[0, 1, u64::MAX]);
    input[0].operation.consumer = None;
    let [bad, _, _] = input;
    assert!(owners[0].begin_conversion(bad).is_err());
    assert_eq!(owners[0].cursors(), Default::default());
    let mut owners = parties(PrssLimits::default());
    let plans = owners
        .each_mut()
        .map(|p| p.plan_conversion(3, Kind::Bits).unwrap());
    let mut messages = begin(&mut owners, plans[0], &[0, 1, u64::MAX]);
    let at = messages
        .iter()
        .position(|m| m.header().to == Party::P0)
        .unwrap();
    let old = messages.remove(at);
    let replay = plans[0]
        .message_from_received(0, old.header(), old.words(), Limits::default())
        .unwrap();
    drop(owners[0].advance_conversion(old).unwrap());
    assert!(owners[0].advance_conversion(replay).is_err());
    assert_eq!(owners[0].status(), Status::Retired);
}

#[test]
fn independent_openssl_scalar_circuit_transcripts_match_all_padding_bits() {
    let mut owners = parties(PrssLimits::default());
    for (kind, end, expected) in [
        (
            Kind::Bits,
            22,
            "31af625820c93e1a53a26ae326a80094496df7e8267315978ee8b76471308d53",
        ),
        (
            Kind::MostSignificantBit,
            33,
            "8c54ac188fbaaf4b096ee2bb945fd1c8ca35cc01a6f4fc3086a843531614eeab",
        ),
    ] {
        let plans = owners
            .each_mut()
            .map(|p| p.plan_conversion(3, kind).unwrap());
        let (output, digest) = run(
            &mut owners,
            plans[0],
            &[1 << 63, u64::MAX, i64::MAX as u64],
            true,
        );
        let actual: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(actual, expected);
        assert!(owners.iter().all(|p| p.cursors().pair_blocks == end));
        drop(output);
    }
}
