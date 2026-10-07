use super::*;

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        first_tensor_index: 17,
    }
}

#[test]
fn complete_blocks_preserve_signed_ties_index_boundaries_and_every_charged_frame() {
    for levels in [0, 2, 4, 8] {
        let p = Profile::new(levels).unwrap();
        let step = 1i32 << p.index_shift();
        let inputs = [
            -48 << 18,
            -step - 1,
            -step,
            -step + 1,
            -257,
            -256,
            -1,
            0,
            1,
            256,
            257,
            step - 1,
            step,
            step + 1,
            48 << 18,
        ];
        let ups = [
            -128 << 18,
            -65537,
            -768,
            -257,
            -256,
            -255,
            -1,
            0,
            1,
            255,
            256,
            257,
            768,
            65537,
            128 << 18,
        ];
        let cost = resources(levels, inputs.len()).unwrap();
        let [mut a, mut b] = issue_reference(&p, context(), inputs.len()).unwrap();
        assert_eq!(a.key_payload_bytes(), cost.party_key_payload_bytes);
        assert_eq!(b.key_payload_bytes(), cost.party_key_payload_bytes);
        assert_eq!(
            cost.party_key_bytes_by_operation.iter().sum::<usize>(),
            cost.party_key_payload_bytes
        );
        let ag: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(i, _)| u32::MAX.wrapping_sub(i as u32 * 7654321))
            .collect();
        let au: Vec<_> = ag.iter().map(|x| x.rotate_left(17)).collect();
        let bg: Vec<_> = inputs
            .iter()
            .zip(&ag)
            .map(|(&g, &a)| (g as u32).wrapping_sub(a))
            .collect();
        let bu: Vec<_> = ups
            .iter()
            .zip(&au)
            .map(|(&u, &a)| (u as u32).wrapping_sub(a))
            .collect();
        let mut frames = [a.start(&ag, &au).unwrap(), b.start(&bg, &bu).unwrap()];
        for round in 0..ROUNDS {
            assert_eq!(frames[0].len(), cost.phase_frame_bytes_per_party[round]);
            assert_eq!(frames[1].len(), cost.phase_frame_bytes_per_party[round]);
            match (
                a.advance(&frames[1]).unwrap(),
                b.advance(&frames[0]).unwrap(),
            ) {
                (Progress::Open(aa), Progress::Open(bb)) => {
                    assert!(round < ROUNDS - 1);
                    frames = [aa, bb];
                }
                (Progress::Complete(aa), Progress::Complete(bb)) => {
                    assert_eq!(round, ROUNDS - 1);
                    for (i, (&gate, &up)) in inputs.iter().zip(&ups).enumerate() {
                        let index = ((i64::from(gate) + (64 << 18)) / i64::from(step)) as usize;
                        let uq = (f64::from(up) / 512.0).round_ties_even();
                        let expected =
                            (f64::from(p.table()[index]) * uq / 512.0).round_ties_even() as i32;
                        assert_eq!(aa[i].wrapping_add(bb[i]) as i32, expected);
                    }
                }
                _ => panic!("party phases differ"),
            }
        }
        assert_eq!(a.key_payload_bytes(), 0);
        assert_eq!(b.key_payload_bytes(), 0);
        assert!(a.start(&ag, &au).is_err());
        assert!(b.advance(&frames[0]).is_err());
    }
}

#[test]
fn failures_at_every_round_burn_all_current_and_future_material() {
    let p = Profile::new(4).unwrap();
    for round in 0..ROUNDS {
        for change in 0..10 {
            let [mut a, mut b] = issue_reference(&p, context(), 2).unwrap();
            let mut frames = [
                a.start(&[u32::MAX; 2], &[1; 2]).unwrap(),
                b.start(&[1; 2], &[u32::MAX; 2]).unwrap(),
            ];
            for _ in 0..round {
                let (Progress::Open(aa), Progress::Open(bb)) = (
                    a.advance(&frames[1]).unwrap(),
                    b.advance(&frames[0]).unwrap(),
                ) else {
                    panic!("early completion");
                };
                frames = [aa, bb];
            }
            match change {
                0..=6 => frames[1][[0, 8, 40, 72, 73, 74, HEADER][change]] ^= 128,
                7 => {
                    frames[1].pop();
                }
                8 => frames[1].push(0),
                _ => {
                    frames[1] = frames[0].clone();
                }
            }
            assert!(a.advance(&frames[1]).is_err());
            assert_eq!(a.key_payload_bytes(), 0);
            assert!(a.advance(&frames[1]).is_err());
            b.cancel();
            assert_eq!(b.key_payload_bytes(), 0);
        }
    }
}

#[test]
fn independent_issuance_context_and_lane_order_cannot_be_substituted() {
    let p = Profile::new(2).unwrap();
    for change in 0..6 {
        let [mut a, _] = issue_reference(&p, context(), 2).unwrap();
        let mut other = context();
        match change {
            0 => other.plan[0] ^= 1,
            1 => other.session[0] ^= 1,
            2 => other.operation[0] ^= 1,
            3 => other.first_tensor_index += 1,
            _ => (),
        }
        let q = Profile::new(if change == 4 { 4 } else { 2 }).unwrap();
        let [_, mut b] = issue_reference(&q, other, 2).unwrap();
        a.start(&[0; 2], &[0; 2]).unwrap();
        let frame = b.start(&[0; 2], &[0; 2]).unwrap();
        assert!(a.advance(&frame).is_err());
        assert_eq!(a.key_payload_bytes(), 0);
    }
    let [mut a, mut b] = issue_reference(&p, context(), 2).unwrap();
    let af = a.start(&[0; 2], &[0; 2]).unwrap();
    let bf = b.start(&[0; 2], &[0; 2]).unwrap();
    let Progress::Open(mut lookup) = b.advance(&af).unwrap() else {
        panic!("lookup absent")
    };
    a.advance(&bf).unwrap();
    lookup[HEADER..].rotate_left(lut::OPENING_BYTES);
    assert!(a.advance(&lookup).is_err());
    assert_eq!(a.key_payload_bytes(), 0);
}

#[test]
fn aggregate_preflight_and_invalid_start_are_burn_boundaries() {
    for levels in 0..=MAX_LEVELS {
        let cost = resources(levels, MAX_LANES).unwrap();
        assert!(cost.allocation_estimate_bytes <= MAX_ALLOCATION);
    }
    assert!(resources(0, 0).is_err());
    assert!(resources(0, MAX_LANES + 1).is_err());
    assert!(resources(MAX_LEVELS + 1, 1).is_err());
    let p = Profile::new(0).unwrap();
    let mut ctx = context();
    ctx.first_tensor_index = u64::MAX;
    assert!(issue_reference(&p, ctx, 1).is_err());
    let [mut a, _] = issue_reference(&p, context(), 1).unwrap();
    assert!(a.start(&[], &[0]).is_err());
    assert_eq!(a.key_payload_bytes(), 0);
    let [mut a, _] = issue_reference(&p, context(), 1).unwrap();
    a.start(&[0], &[0]).unwrap();
    assert!(a.start(&[0], &[0]).is_err());
    assert_eq!(a.key_payload_bytes(), 0);
}
