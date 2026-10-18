//! Public fixtures and test-only correlated-material dealer. Production has no
//! aggregate three-party runtime or dealer. These tests alone reconstruct outputs.
use pllm_core::puma::*;

fn context() -> Context {
    Context::new([41; 32], [73; 32], 12).unwrap()
}
fn parties() -> [Party; 3] {
    [Party::P0, Party::P1, Party::P2]
}

struct Fixture {
    operation: Operation,
    runtimes: [PartyRuntime; 3],
    inputs: [Option<ReplicatedShare>; 3],
    material: [Option<LocalMaterial>; 3],
}

// This factory intentionally knows all source values and correlations. It is
// test-only, and deterministic public numbers must never be production material.
fn fixture(x: &[i64], masks: &[u64], shift: u8, pivot: Party) -> Fixture {
    fixture_at(x, masks, shift, pivot, 0)
}

fn fixture_at(x: &[i64], masks: &[u64], shift: u8, pivot: Party, sequence: u64) -> Fixture {
    let limits = Limits::default();
    let operation = Operation::new(context(), sequence, pivot, x.len(), shift, limits).unwrap();
    let first: Vec<u64> = (0..x.len())
        .map(|i| 0xfeeef000ffff0000_u64.wrapping_add(i as u64))
        .collect();
    let second: Vec<u64> = (0..x.len())
        .map(|i| 0x87654321fedcba98_u64.wrapping_sub(i as u64 * 17))
        .collect();
    let third: Vec<u64> = x
        .iter()
        .enumerate()
        .map(|(i, x)| (*x as u64).wrapping_sub(first[i]).wrapping_sub(second[i]))
        .collect();
    let shares = [&first[..], &second[..], &third[..]];
    let ra: Vec<u64> = (0..x.len())
        .map(|i| 0xaaff12348765bbcc_u64.wrapping_add(i as u64 + sequence * 413))
        .collect();
    let rb: Vec<u64> = masks
        .iter()
        .zip(&ra)
        .map(|(r, a)| r.wrapping_sub(*a))
        .collect();
    let ya: Vec<u64> = (0..x.len())
        .map(|i| 0x9988776600112233_u64.wrapping_add(i as u64 + sequence * 997))
        .collect();
    let yb: Vec<u64> = (0..x.len())
        .map(|i| 0xff123456789abcde_u64.wrapping_sub(i as u64 + sequence * 977))
        .collect();
    Fixture {
        operation,
        runtimes: parties().map(|p| PartyRuntime::new(context(), p, limits)),
        inputs: std::array::from_fn(|i| {
            Some(
                ReplicatedShare::from_local(
                    operation,
                    parties()[i],
                    shares[i],
                    shares[(i + 1) % 3],
                    limits,
                )
                .unwrap(),
            )
        }),
        material: parties().map(|p| {
            Some(if p == operation.helper() {
                LocalMaterial::helper(operation, &ra, &rb, &ya, &yb, limits).unwrap()
            } else if p == operation.evaluator_a() {
                LocalMaterial::evaluator(operation, p, &ra, &ya, limits).unwrap()
            } else {
                LocalMaterial::evaluator(operation, p, &rb, &yb, limits).unwrap()
            })
        }),
    }
}

fn begin(f: &mut Fixture) -> Vec<Message> {
    let mut messages = Vec::new();
    for i in 0..3 {
        let out = f.runtimes[i]
            .begin(
                f.operation,
                f.inputs[i].take().unwrap(),
                f.material[i].take().unwrap(),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        messages.extend(out.into_iter().flatten());
    }
    assert_eq!(messages.len(), 4);
    messages
}

fn take(messages: &mut Vec<Message>, to: Party, phase: MessagePhase) -> Message {
    let i = messages
        .iter()
        .position(|m| m.header().to == to && m.header().phase == phase)
        .unwrap();
    messages.swap_remove(i)
}

fn finish(f: &mut Fixture, mut messages: Vec<Message>) -> [ReplicatedShare; 3] {
    for party in [f.operation.evaluator_a(), f.operation.evaluator_b()] {
        let peer = take(&mut messages, party, MessagePhase::MaskedInput);
        let corr = take(&mut messages, party, MessagePhase::Correlation);
        messages.push(
            f.runtimes[party as usize]
                .open(peer, corr)
                .unwrap_or_else(|e| panic!("{e}")),
        );
    }
    for party in [f.operation.evaluator_b(), f.operation.evaluator_a()] {
        f.runtimes[party as usize]
            .reshare(take(&mut messages, party, MessagePhase::Reshare))
            .unwrap();
    }
    assert!(messages.is_empty());
    std::array::from_fn(|i| {
        f.runtimes[i]
            .take_output()
            .unwrap_or_else(|e| panic!("{e}"))
    })
}

fn reconstructed(output: &[ReplicatedShare; 3]) -> Vec<i64> {
    for i in 0..3 {
        assert_eq!(output[i].second(), output[(i + 1) % 3].first());
    }
    (0..output[0].first().len())
        .map(|i| {
            output
                .iter()
                .fold(0_u64, |v, share| v.wrapping_add(share.first()[i])) as i64
        })
        .collect()
}

#[test]
fn three_actual_party_views_match_independent_oracle_all_pivots_and_wraps() {
    let x = [
        INPUT_MIN,
        INPUT_MIN + 1,
        -262145,
        -262144,
        -9,
        -8,
        -1,
        0,
        1,
        7,
        262143,
        INPUT_MAX_EXCLUSIVE - 1,
    ];
    for pivot in parties() {
        for shift in [1, 3, 18, 31, 61, 62] {
            for high in [0, 1 << 63, u64::MAX] {
                let masks: Vec<u64> = (0..x.len())
                    .map(|i| high.wrapping_sub(i as u64 * 131))
                    .collect();
                let mut f = fixture(&x, &masks, shift, pivot);
                let messages = begin(&mut f);
                // H receives no masked input, nor a reshare/opening of C.
                assert!(messages
                    .iter()
                    .all(|m| m.header().to != f.operation.helper()));
                let output = finish(&mut f, messages);
                let expected: Vec<i64> = x
                    .iter()
                    .zip(&masks)
                    .map(|(&v, &r)| oracle::truncate(v, 64, shift, r).unwrap())
                    .collect();
                assert_eq!(reconstructed(&output), expected);
                assert!(f
                    .runtimes
                    .iter()
                    .all(|r| r.status() == Status::Ready && r.next_sequence() == 1));
            }
        }
    }
}

#[test]
fn exact_stochastic_frequencies_over_all_low_mask_residues() {
    // Condition on fixed high bits and enumerate every low residue. These
    // frequencies are exact; there is no flaky statistical acceptance test.
    for shift in 1..=8 {
        let unit = 1_u64 << shift;
        for x in [-257, -9, -8, -1, 0, 1, 7, 255] {
            let inputs = vec![x; unit as usize];
            let masks: Vec<u64> = (0..unit).map(|low| (u64::MAX << shift) | low).collect();
            for pivot in parties() {
                let mut f = fixture(&inputs, &masks, shift, pivot);
                let messages = begin(&mut f);
                let outputs = reconstructed(&finish(&mut f, messages));
                let law = oracle::distribution(x, 64, shift).unwrap();
                assert_eq!(
                    outputs.iter().filter(|&&v| v > law.floor).count() as u64,
                    law.upper_count
                );
                assert!(outputs.iter().all(|&v| v == law.floor || v == law.upper));
            }
        }
    }
}

#[test]
fn consecutive_operations_keep_local_pairs_and_rotate_helper() {
    let x = [-1001, -1, 0, 777];
    let masks = [0, 1, u64::MAX, 7];
    let mut f = fixture(&x, &masks, 3, Party::P0);
    let messages = begin(&mut f);
    let first_output = finish(&mut f, messages);
    let expected_first: Vec<i64> = x
        .iter()
        .zip(masks)
        .map(|(&v, r)| oracle::truncate(v, 64, 3, r).unwrap())
        .collect();
    let next_masks = [8, 19, 27, u64::MAX - 17];
    let mut next = fixture_at(&expected_first, &next_masks, 2, Party::P1, 1);
    // Reuse only each party's own returned pair, with explicit next-operation
    // binding. Neither the production API nor this chaining step reconstructs it.
    for (i, output) in first_output.iter().enumerate() {
        next.inputs[i] = Some(
            ReplicatedShare::from_local(
                next.operation,
                parties()[i],
                output.first(),
                output.second(),
                Limits::default(),
            )
            .unwrap(),
        );
    }
    std::mem::swap(&mut f.runtimes, &mut next.runtimes);
    let messages = begin(&mut next);
    let second_output = finish(&mut next, messages);
    let expected: Vec<i64> = expected_first
        .iter()
        .zip(next_masks)
        .map(|(&v, r)| oracle::truncate(v, 64, 2, r).unwrap())
        .collect();
    assert_eq!(reconstructed(&second_output), expected);
    assert!(next.runtimes.iter().all(|r| r.next_sequence() == 2));
}

#[test]
fn wrong_input_or_material_operation_retires_without_output() {
    for wrong_input in [false, true] {
        let mut f = fixture(&[9], &[1], 3, Party::P0);
        let mut other = fixture_at(&[9], &[17], 3, Party::P0, 1);
        let input = if wrong_input {
            other.inputs[0].take().unwrap()
        } else {
            f.inputs[0].take().unwrap()
        };
        let material = if wrong_input {
            f.material[0].take().unwrap()
        } else {
            other.material[0].take().unwrap()
        };
        assert!(matches!(
            f.runtimes[0].begin(f.operation, input, material),
            Err(Error::Binding)
        ));
        assert_eq!(f.runtimes[0].status(), Status::Retired);
        assert!(f.runtimes[0].take_output().is_err());
    }
}

#[test]
fn replay_burns_runtime_and_output_is_one_use() {
    let mut f = fixture(&[-9], &[7], 3, Party::P0);
    let messages = begin(&mut f);
    let _output = finish(&mut f, messages);
    let mut replay = fixture(&[-9], &[7], 3, Party::P0);
    assert!(matches!(
        f.runtimes[0].begin(
            f.operation,
            replay.inputs[0].take().unwrap(),
            replay.material[0].take().unwrap()
        ),
        Err(Error::Binding)
    ));
    assert_eq!(f.runtimes[0].status(), Status::Retired);
    assert!(f.runtimes[1].take_output().is_err());
    assert_eq!(f.runtimes[1].status(), Status::Retired);
}

#[test]
fn wrong_binding_role_or_phase_burns_before_using_peer_values() {
    for mode in 0..4 {
        let mut f = fixture(&[123], &[456], 18, Party::P0);
        let mut messages = begin(&mut f);
        let peer = take(&mut messages, Party::P0, MessagePhase::MaskedInput);
        let corr = take(&mut messages, Party::P0, MessagePhase::Correlation);
        let mut header = peer.header();
        let mut op = f.operation;
        match mode {
            0 => {
                op = Operation::new(
                    Context::new([99; 32], [73; 32], 12).unwrap(),
                    0,
                    Party::P0,
                    1,
                    18,
                    Limits::default(),
                )
                .unwrap();
                header.binding = op.digest();
            }
            1 => header.from = Party::P2,
            2 => header.to = Party::P2,
            _ => header.phase = MessagePhase::Reshare,
        }
        let altered = Message::from_received(header, peer.words(), op, Limits::default())
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(f.runtimes[0].open(altered, corr).is_err());
        assert_eq!(f.runtimes[0].status(), Status::Retired);
        assert!(f.runtimes[0].take_output().is_err());
    }
}

#[test]
fn cancellation_at_each_round_and_duplicate_open_are_terminal() {
    for round in 0..=2 {
        let mut f = fixture(&[3], &[7], 3, Party::P0);
        if round > 0 {
            let mut messages = begin(&mut f);
            if round > 1 {
                let peer = take(&mut messages, Party::P0, MessagePhase::MaskedInput);
                let corr = take(&mut messages, Party::P0, MessagePhase::Correlation);
                f.runtimes[0]
                    .open(peer, corr)
                    .unwrap_or_else(|e| panic!("{e}"));
            }
        }
        f.runtimes[0].cancel();
        f.runtimes[0].cancel();
        assert_eq!(f.runtimes[0].status(), Status::Retired);
        assert!(f.runtimes[0].take_output().is_err());
    }
    let mut f = fixture(&[3], &[7], 3, Party::P0);
    let mut messages = begin(&mut f);
    let peer = take(&mut messages, Party::P0, MessagePhase::MaskedInput);
    let corr = take(&mut messages, Party::P0, MessagePhase::Correlation);
    let duplicate_peer =
        Message::from_received(peer.header(), peer.words(), f.operation, Limits::default())
            .unwrap_or_else(|e| panic!("{e}"));
    let duplicate_corr =
        Message::from_received(corr.header(), corr.words(), f.operation, Limits::default())
            .unwrap_or_else(|e| panic!("{e}"));
    f.runtimes[0]
        .open(peer, corr)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(
        f.runtimes[0].open(duplicate_peer, duplicate_corr),
        Err(Error::Phase)
    ));
    assert_eq!(f.runtimes[0].status(), Status::Retired);
}

#[test]
fn limits_invalid_shift_material_shape_and_helper_role_reject() {
    let limits = Limits::default();
    assert_eq!(
        Limits::new(usize::MAX, 1, MAX_ADMITTED_BYTES),
        Err(Error::Bounds)
    );
    assert_eq!(Limits::new(1, 1, 4096), Err(Error::Bounds));
    assert!(Operation::new(context(), 0, Party::P0, MAX_LANES + 1, 18, limits).is_err());
    assert!(Operation::new(context(), 0, Party::P0, 1, 63, limits).is_err());
    let op = Operation::new(context(), 0, Party::P0, 1, 18, limits).unwrap();
    assert!(matches!(
        LocalMaterial::evaluator(op, Party::P2, &[0], &[0], limits),
        Err(Error::Role)
    ));
    assert!(matches!(
        LocalMaterial::helper(op, &[0], &[], &[0], &[0], limits),
        Err(Error::Shape)
    ));
    let mut f = fixture(&[3], &[7], 3, Party::P0);
    begin(&mut f);
    assert_eq!(f.runtimes[2].status(), Status::Complete);
    // Helper has no open phase, even with plausibly sized frames.
    let h = Header {
        binding: f.operation.digest(),
        from: Party::P0,
        to: Party::P2,
        phase: MessagePhase::MaskedInput,
    };
    let p = Message::from_received(h, &[0], f.operation, limits).unwrap_or_else(|e| panic!("{e}"));
    let q = Message::from_received(h, &[0], f.operation, limits).unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(f.runtimes[2].open(p, q), Err(Error::Phase)));
    assert_eq!(f.runtimes[2].status(), Status::Retired);
}
