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

fn shares(op: Plan, x: &[i64]) -> [ReplicatedShare; 3] {
    let a = vec![u64::MAX - 31; x.len()];
    let b = vec![u64::MAX / 2 + 7; x.len()];
    let c: Vec<_> = x
        .iter()
        .map(|&v| (v as u64).wrapping_sub(a[0]).wrapping_sub(b[0]))
        .collect();
    let parts = [&a[..], &b[..], &c[..]];
    [Party::P0, Party::P1, Party::P2].map(|p| {
        op.share(
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
    let mut private = [0; 3];
    let mut work = [0; 3];
    for (ordinal, n) in [1, 3, 8, 17, MAX_LANES].into_iter().enumerate() {
        let bounds = vec![
            InputBounds {
                left: Interval::new(-1 << 20, 1 << 20).unwrap(),
                right: Interval::new(-1 << 20, 1 << 20).unwrap(),
            };
            n
        ];
        let x: Vec<_> = (0..n).map(|i| (i as i64 - 515) * 113).collect();
        let y: Vec<_> = (0..n).map(|i| (i as i64 - 127) * 79).collect();
        let ops = owners.each_mut().map(|p| p.plan(&bounds).unwrap());
        assert_eq!(ops, [ops[0]; 3]);
        let op = ops[0];
        let mut messages: Vec<_> = owners
            .iter_mut()
            .zip(shares(op, &x))
            .zip(shares(op, &y))
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
            let product = i128::from(x[lane]) * i128::from(y[lane]);
            let floor = product.div_euclid(1 << 18) as i64;
            let ceil = -(-product).div_euclid(1 << 18) as i64;
            assert!(actual == floor || actual == ceil);
        }
        let q = (n as u64).div_ceil(2);
        pair_blocks += 3 * q;
        private[op.helper() as usize] += n as u64;
        for (i, p) in owners.iter().enumerate() {
            work[i] += if i == op.helper() as usize {
                6 * q + n as u64 + 1
            } else {
                4 * q + 1
            };
            assert_eq!(p.cursors().public_blocks, ordinal as u64 + 1);
            assert_eq!(p.cursors().pair_blocks, pair_blocks);
            assert_eq!(p.cursors().private_blocks, private[i]);
            assert_eq!(p.cursors().aes_work, work[i]);
            assert_eq!(p.status(), Status::Ready);
            assert!(usages[i].peak_bytes() <= Limits::default().admitted_bytes(n).unwrap());
        }
        drop(outputs);
        assert!(usages.iter().all(|u| u.live_bytes() == 0));
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
    let a = shares(old_op, &[1]).into_iter().next().unwrap();
    let b = shares(old_op, &[1]).into_iter().next().unwrap();
    assert!(fresh[0].begin(a, b).is_err());
    assert_eq!(fresh[0].status(), Status::Retired);
    for p in &mut old {
        p.cancel();
        assert!(p.plan(&bounds).is_err());
    }
}
