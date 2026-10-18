use pllm_core::puma::{prss::*, *};

fn setup(limits: PrssLimits) -> [SourceParty; 3] {
    let context = Context::new([81; 32], [93; 32], 5).unwrap();
    let mut owners =
        [Party::P0, Party::P1, Party::P2].map(|p| SetupParty::new(context, p, limits).unwrap());
    let contributions = owners.each_mut().map(|p| p.contribution().unwrap());
    for (i, owner) in owners.iter_mut().enumerate() {
        for (j, contribution) in contributions.iter().enumerate() {
            if i != j {
                owner.receive_public(*contribution).unwrap();
            }
        }
    }
    let mut seeds = owners.each_mut().map(|p| Some(p.share_seed().unwrap()));
    for (i, owner) in owners.iter_mut().enumerate() {
        owner
            .receive_seed(seeds[(i + 1) % 3].take().unwrap())
            .unwrap();
    }
    owners.each_mut().map(|p| p.finish().unwrap())
}

fn inputs(op: Operation, x: &[i64]) -> [ReplicatedShare; 3] {
    let a = vec![u64::MAX - 123; x.len()];
    let b = vec![u64::MAX / 2; x.len()];
    let c: Vec<u64> = x
        .iter()
        .map(|&v| (v as u64).wrapping_sub(a[0]).wrapping_sub(b[0]))
        .collect();
    let parts = [&a[..], &b[..], &c[..]];
    [Party::P0, Party::P1, Party::P2].map(|p| {
        ReplicatedShare::from_local(
            op,
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
fn production_os_setup_drives_party_local_truncation_with_exact_cursors() {
    let mut owners = setup(PrssLimits::default());
    let mut pair_blocks = 0;
    let mut private_blocks = [0; 3];
    for (ordinal, (n, shift)) in [(1, 1), (3, 18), (8, 62), (17, 3)].into_iter().enumerate() {
        let ops = owners.each_mut().map(|p| p.plan(n, shift).unwrap());
        assert_eq!(ops, [ops[0]; 3]);
        let op = ops[0];
        let source = [INPUT_MIN, -9, -1, 0, 1, INPUT_MAX_EXCLUSIVE - 1];
        let x: Vec<i64> = (0..n).map(|i| source[i % source.len()]).collect();
        let mut messages = Vec::new();
        for (owner, input) in owners.iter_mut().zip(inputs(op, &x)) {
            messages.extend(owner.begin(input).unwrap().into_iter().flatten());
        }
        assert!(messages.iter().all(|m| m.header().to != op.helper()));
        for party in [op.evaluator_a(), op.evaluator_b()] {
            let peer = take(&mut messages, party, MessagePhase::MaskedInput);
            let corr = take(&mut messages, party, MessagePhase::Correlation);
            messages.push(owners[party as usize].open(peer, corr).unwrap());
        }
        for party in [op.evaluator_a(), op.evaluator_b()] {
            owners[party as usize]
                .reshare(take(&mut messages, party, MessagePhase::Reshare))
                .unwrap();
        }
        let output = owners.each_mut().map(|p| p.take_output().unwrap());
        for i in 0..3 {
            assert_eq!(output[i].second(), output[(i + 1) % 3].first());
        }
        for (i, &v) in x.iter().enumerate() {
            let result = output
                .iter()
                .fold(0_u64, |s, p| s.wrapping_add(p.first()[i])) as i64;
            let law = oracle::distribution(v, 64, shift).unwrap();
            assert!(result == law.floor || result == law.upper);
        }
        pair_blocks += 2 * (n as u64).div_ceil(2);
        private_blocks[op.helper() as usize] += n as u64;
        for (i, owner) in owners.iter().enumerate() {
            let cursors = owner.cursors();
            assert_eq!(cursors.public_blocks, ordinal as u64 + 1);
            assert_eq!(cursors.pair_blocks, pair_blocks);
            assert_eq!(cursors.private_blocks, private_blocks[i]);
        }
    }
}

#[test]
fn cancellation_and_budget_exhaustion_leave_no_restart_path() {
    let mut owners = setup(PrssLimits::new(Limits::default(), 1).unwrap());
    let ops = owners.each_mut().map(|p| p.plan(1, 18).unwrap());
    for (owner, input) in owners.iter_mut().zip(inputs(ops[0], &[7])) {
        assert!(matches!(owner.begin(input), Err(Error::Bounds)));
        assert_eq!(owner.status(), Status::Retired);
        assert!(matches!(owner.plan(1, 18), Err(Error::Retired)));
    }
    let mut owners = setup(PrssLimits::default());
    owners[0].plan(1, 18).unwrap();
    owners[0].cancel();
    owners[0].cancel();
    assert!(matches!(owners[0].plan(1, 18), Err(Error::Retired)));
    assert!(owners[0].take_output().is_err());
}

#[test]
fn old_session_frames_cannot_be_replayed_into_fresh_os_setup() {
    let mut old = setup(PrssLimits::default());
    let old_op = old[0].plan(1, 3).unwrap();
    let mut fresh = setup(PrssLimits::default());
    let new_op = fresh[0].plan(1, 3).unwrap();
    assert_ne!(old_op.digest(), new_op.digest());
    let old_input =
        ReplicatedShare::from_local(old_op, Party::P0, &[0], &[0], Limits::default()).unwrap();
    assert!(matches!(fresh[0].begin(old_input), Err(Error::Binding)));
    assert_eq!(fresh[0].status(), Status::Retired);
}
