use super::*;

struct FixedSeeds {
    role: usize,
    calls: usize,
}
impl SeedEntropy for FixedSeeds {
    fn draw(&mut self, purpose: SeedPurpose) -> Result<Zeroizing<[u8; 16]>, Error> {
        let expected = [
            SeedPurpose::Public,
            SeedPurpose::Private,
            SeedPurpose::Pairwise,
        ];
        assert_eq!(purpose, expected[self.calls]);
        let value = match purpose {
            SeedPurpose::Public => self.role as u128 + 1,
            SeedPurpose::Private => self.role as u128 + 101,
            SeedPurpose::Pairwise => (self.role as u128 + 1) * 11,
        };
        self.calls += 1;
        Ok(Zeroizing::new(value.to_le_bytes()))
    }
}

fn context() -> Context {
    Context::new([9; 32], [7; 32], 5).unwrap()
}
fn roles() -> [Party; 3] {
    [Party::P0, Party::P1, Party::P2]
}

// Deterministic seed injection and aggregate routing exist ONLY in tests.
fn setups() -> [SetupParty; 3] {
    roles().map(|party| {
        let mut entropy = FixedSeeds {
            role: party as usize,
            calls: 0,
        };
        let setup = SetupParty::with_entropy(context(), party, PrssLimits::default(), &mut entropy)
            .unwrap();
        assert_eq!(entropy.calls, 3);
        setup
    })
}
fn gather(setups: &mut [SetupParty; 3]) {
    let public = std::array::from_fn::<_, 3, _>(|i| setups[i].contribution().unwrap());
    for (i, setup) in setups.iter_mut().enumerate() {
        for (j, message) in public.iter().enumerate() {
            if i != j {
                setup.receive_public(*message).unwrap();
            }
        }
    }
}
fn parties() -> [SourceParty; 3] {
    let mut setups = setups();
    gather(&mut setups);
    let mut messages: [Option<SeedMessage>; 3] =
        std::array::from_fn(|i| Some(setups[i].share_seed().unwrap()));
    for (i, setup) in setups.iter_mut().enumerate() {
        setup
            .receive_seed(messages[(i + 1) % 3].take().unwrap())
            .unwrap();
    }
    std::array::from_fn(|i| setups[i].finish().unwrap())
}

fn inputs(op: Operation, x: &[i64]) -> [ReplicatedShare; 3] {
    let a = vec![0x1020304050607080_u64; x.len()];
    let b = vec![0x8877665544332211_u64; x.len()];
    let c: Vec<u64> = x
        .iter()
        .map(|&v| (v as u64).wrapping_sub(a[0]).wrapping_sub(b[0]))
        .collect();
    let components = [&a[..], &b[..], &c[..]];
    std::array::from_fn(|i| {
        ReplicatedShare::from_local(
            op,
            roles()[i],
            components[i],
            components[(i + 1) % 3],
            Limits::default(),
        )
        .unwrap()
    })
}

fn begin(parties: &mut [SourceParty; 3], op: Operation, x: &[i64]) -> Vec<Message> {
    let mut messages = Vec::new();
    for (party, input) in parties.iter_mut().zip(inputs(op, x)) {
        messages.extend(party.begin(input).unwrap().into_iter().flatten());
    }
    messages
}
fn take(messages: &mut Vec<Message>, recipient: Party, phase: MessagePhase) -> Message {
    let at = messages
        .iter()
        .position(|m| m.header().to == recipient && m.header().phase == phase)
        .unwrap();
    messages.swap_remove(at)
}
fn finish(
    parties: &mut [SourceParty; 3],
    op: Operation,
    mut messages: Vec<Message>,
) -> [ReplicatedShare; 3] {
    for party in [op.evaluator_a(), op.evaluator_b()] {
        let open = take(&mut messages, party, MessagePhase::MaskedInput);
        let corr = take(&mut messages, party, MessagePhase::Correlation);
        messages.push(parties[party as usize].open(open, corr).unwrap());
    }
    for party in [op.evaluator_a(), op.evaluator_b()] {
        parties[party as usize]
            .reshare(take(&mut messages, party, MessagePhase::Reshare))
            .unwrap();
    }
    assert!(messages.is_empty());
    std::array::from_fn(|i| parties[i].take_output().unwrap())
}

#[test]
fn aes_call_reset_vectors_from_independent_openssl() {
    // AES-128-CTR with zero key, IV=start.to_le_bytes(), encrypt zero bytes.
    // Constants computed by system OpenSSL, never by upstream SPU/YACL code.
    let cases: &[(u64, &[u64])] = &[
        (0, &[0x3b2c8aefd44be966]),
        (
            0,
            &[
                0x3b2c8aefd44be966,
                0x2e2b34ca59fa4c88,
                0x61307efacefce258,
                0x5a45e7a4571d7f36,
            ],
        ),
        (1, &[0xf06f1de916187147]),
        (
            2,
            &[0x8580adeaa776f1bc, 0x81a2622436eaaceb, 0x4687b4e2b25fc76e],
        ),
        (
            255,
            &[0x677d9630a51a4fdb, 0x4be20eeb1547ce32, 0x84fa3640153d31df],
        ),
        (
            256,
            &[0x72fef58ef11c32fa, 0x4151941e5c2ad87d, 0xd1edcfc85008f30f],
        ),
    ];
    for &(start, expected) in cases {
        let mut out = vec![0; expected.len()];
        assert_eq!(
            fill(&[0; 16], start, &mut out).unwrap(),
            start + (expected.len() as u64).div_ceil(2)
        );
        assert_eq!(out, expected);
    }
    let first = expand(&[0; 16], 0, 1, None).unwrap();
    let second = expand(&[0; 16], 1, 1, None).unwrap();
    let combined = expand(&[0; 16], 0, 2, None).unwrap();
    assert_eq!(first[0], combined[0]);
    assert_ne!(second[0], combined[1]); // no reuse of a discarded half-block
}

#[test]
fn independent_aes_vector_crosses_within_fill_big_endian_carry() {
    // OpenSSL AES-128-CTR, zero key/IV, 4128 zero input bytes. Offsets 255/256
    // INSIDE one fill cross the low counter byte; new LE cursor starts do not.
    let mut out = [0_u64; 516];
    assert_eq!(fill(&[0; 16], 0, &mut out), Ok(258));
    assert_eq!(
        &out[508..],
        &[
            0x7fc21296b989dea3,
            0x03c89f96bf20cf35,
            0x8825a63bf9de0df7,
            0xe045d6d0670e2a24,
            0xd0b180b609cc56fb,
            0x7cf0299e14525a7c,
            0xad9fe8b95c9db649,
            0x3f43bab94389ffae,
        ]
    );
    let mut hash = Sha256::new();
    for word in out {
        hash.update(word.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", hash.finalize()),
        "b844e6a17d006cd080e7adcdf6d82be099aead7190912154e5358900dc3bdc04"
    );
    for start in [255, 256] {
        let mut reset = [0; 2];
        fill(&[0; 16], start, &mut reset).unwrap();
        assert_ne!(&out[2 * start as usize..2 * start as usize + 2], &reset);
    }
}

#[test]
fn setup_seed_direction_separation_and_source_sum_are_exact() {
    let parties = parties();
    for (i, p) in parties.iter().enumerate() {
        let seeds = p.seeds.as_ref().unwrap();
        assert_eq!(u128::from_le_bytes(*seeds.own), (i as u128 + 1) * 11);
        assert_eq!(
            u128::from_le_bytes(**seeds.next.as_ref().unwrap()),
            (((i + 1) % 3) as u128 + 1) * 11
        );
        assert_eq!(u128::from_le_bytes(*seeds.private), i as u128 + 101);
        assert_eq!(p.public_seed, 6); // sum, NOT XOR (1^2^3 == 0)
        assert_eq!(p.context(), parties[0].context());
        assert_eq!(p.cursors(), Cursors::default());
    }
}

#[test]
fn independent_full_prss_transcript_and_truncation_golden_vector() {
    let mut parties = parties();
    let operations = parties.each_mut().map(|p| p.plan(3, 3).unwrap());
    assert_eq!(operations, [operations[0]; 3]);
    let op = operations[0];
    assert_eq!(op.evaluator_a(), Party::P2);
    let messages = begin(&mut parties, op, &[-9, 1, INPUT_MAX_EXCLUSIVE - 1]);
    // Independent OpenSSL PRSS expansion + Python integer arithmetic only.
    let ca = [
        17150596819922755392,
        4139010590947917390,
        16350437219255618998,
        13681099616973011411,
        7065740685556729496,
        9666272435057114645,
    ];
    let cb = [
        1296147253786796225,
        14307733482761634226,
        2096306854453932619,
        5734752195131898559,
        11462564151222433859,
        9913939687251112775,
    ];
    for frame in &messages {
        if frame.header().phase == MessagePhase::Correlation {
            assert_eq!(
                frame.words(),
                if frame.header().to == Party::P2 {
                    &ca
                } else {
                    &cb
                }
            );
        }
    }
    let outputs = finish(&mut parties, op, messages);
    let ya = [
        12895254768842343935,
        14723401510629772609,
        14776894518607917464,
    ];
    let yb = [
        3130976561542318425,
        8379566212570222057,
        12223716883108595360,
    ];
    let z = [
        2420512743324889255,
        13790520424219108566,
        10469337498006013896,
    ];
    assert_eq!(outputs[2].first(), ya);
    assert_eq!(outputs[2].second(), z);
    assert_eq!(outputs[0].first(), z);
    assert_eq!(outputs[0].second(), yb);
    assert_eq!(outputs[1].first(), yb);
    assert_eq!(outputs[1].second(), ya);
    let masks = [
        16976233944017642647,
        652486104556893918,
        18291116425644182241,
    ];
    for (i, x) in [-9, 1, INPUT_MAX_EXCLUSIVE - 1].into_iter().enumerate() {
        let reconstructed = outputs
            .iter()
            .fold(0_u64, |sum, s| sum.wrapping_add(s.first()[i]))
            as i64;
        assert_eq!(reconstructed, oracle::truncate(x, 64, 3, masks[i]).unwrap());
    }
    assert_eq!(
        parties[0].cursors(),
        Cursors {
            public_blocks: 1,
            pair_blocks: 4,
            private_blocks: 0,
            aes_work: 5
        }
    );
    assert_eq!(
        parties[1].cursors(),
        Cursors {
            public_blocks: 1,
            pair_blocks: 4,
            private_blocks: 3,
            aes_work: 12
        }
    );
    assert_eq!(parties[2].cursors(), parties[0].cursors());
    // Odd -> odd -> even: both stream cursors advance, including unused streams.
    for (step, lanes) in [(1, 1), (2, 4)] {
        let ops = parties.each_mut().map(|p| p.plan(lanes, 3).unwrap());
        assert_eq!(ops, [ops[0]; 3]);
        assert_eq!(ops[0].evaluator_a(), Party::P0);
        let messages = begin(&mut parties, ops[0], &vec![0; lanes]);
        let output = finish(&mut parties, ops[0], messages);
        for i in 0..lanes {
            assert_eq!(
                output
                    .iter()
                    .fold(0_u64, |s, v| s.wrapping_add(v.first()[i])),
                0
            );
        }
        assert!(parties
            .iter()
            .all(|p| p.cursors().pair_blocks == if step == 1 { 6 } else { 10 }));
    }
    assert_eq!(parties[1].cursors().private_blocks, 3);
    assert_eq!(parties[2].cursors().private_blocks, 5);
}

#[test]
fn counter_and_work_exhaustion_retire_every_secret() {
    for mode in 0..4 {
        let mut parties = parties();
        let p = &mut parties[0];
        match mode {
            0 => p.cursors.public_blocks = u64::MAX,
            1 => p.cursors.pair_blocks = u64::MAX - 1,
            2 => p.cursors.aes_work = p.limits.aes_blocks,
            _ => p.cursors.private_blocks = u64::MAX,
        }
        if mode < 3 {
            assert!(p.plan(3, 3).is_err());
        } else {
            // Public first pivot is P2, so role 1 is H; exercise its private cap.
            let p = &mut parties[1];
            p.cursors.private_blocks = u64::MAX;
            let op = p.plan(1, 3).unwrap();
            let input =
                ReplicatedShare::from_local(op, Party::P1, &[0], &[0], Limits::default()).unwrap();
            assert!(matches!(p.begin(input), Err(Error::Bounds)));
            assert!(p.seeds.is_none());
            assert_eq!(p.status(), Status::Retired);
            continue;
        }
        assert!(p.seeds.is_none());
        assert_eq!(p.status(), Status::Retired);
        assert!(matches!(p.plan(1, 3), Err(Error::Retired)));
    }
    assert_eq!(pivot(u64::MAX), Err(Error::Bounds));
    assert_eq!(pivot(u64::MAX - 1), Err(Error::Bounds));
    assert_eq!(pivot(u64::MAX - 2), Ok(Party::P1));
    assert_eq!(fill(&[0; 16], u64::MAX, &mut [99]), Err(Error::Bounds));
}

#[test]
fn replay_role_session_phase_and_fork_mismatch_retire_setup() {
    let mut s = setups();
    let contributions = s.each_mut().map(|p| p.contribution().unwrap());
    s[0].receive_public(contributions[1]).unwrap();
    assert_eq!(s[0].receive_public(contributions[1]), Err(Error::Role));
    assert!(s[0].seeds.is_none());
    let mut foreign = contributions[0];
    foreign.context = Context::new([1; 32], [2; 32], 3).unwrap();
    assert_eq!(s[1].receive_public(foreign), Err(Error::Binding));
    assert!(s[1].seeds.is_none());
    assert!(matches!(s[2].share_seed(), Err(Error::Phase)));
    assert!(s[2].seeds.is_none());
    let mut old = setups();
    gather(&mut old);
    let old_message = old[1].share_seed().unwrap();
    let mut new = setups();
    // A fresh contribution stands for a new OS setup, even under same Context.
    new[0].contributions[0] = Some(1000);
    gather(&mut new);
    let _send = new[0].share_seed().unwrap();
    assert_eq!(new[0].receive_seed(old_message), Err(Error::Binding));
    assert!(new[0].seeds.is_none());
    let mut another = setups();
    gather(&mut another);
    let wrong_peer = another[2].share_seed().unwrap();
    let _outbound = another[0].share_seed().unwrap();
    assert_eq!(another[0].receive_seed(wrong_peer), Err(Error::Role));
    assert!(another[0].seeds.is_none());
    let mut parties = parties();
    parties[0].pid = std::process::id().wrapping_add(1);
    assert!(matches!(parties[0].plan(1, 3), Err(Error::Retired)));
    assert!(parties[0].seeds.is_none());
}

#[test]
fn cursor_and_purpose_are_bound_into_operation_identity() {
    let mut parties = parties();
    let op = parties[0].plan(3, 3).unwrap();
    let source = op.source.unwrap();
    for changed in [
        SourceBinding {
            public_start: source.public_start + 1,
            ..source
        },
        SourceBinding {
            mask_start: source.mask_start + 1,
            ..source
        },
        SourceBinding {
            output_start: source.output_start + 1,
            ..source
        },
        SourceBinding {
            mask_start: source.output_start,
            output_start: source.mask_start,
            ..source
        },
    ] {
        let mut other = op;
        other.source = Some(changed);
        assert_ne!(op.digest(), other.digest());
    }
    // No second public draw or detached plan reuse is available.
    assert!(matches!(parties[0].plan(3, 3), Err(Error::Phase)));
    assert!(parties[0].seeds.is_none());
}

#[test]
fn entropy_failure_has_no_partial_setup_or_fallback() {
    struct FailAt {
        fail: usize,
        calls: usize,
    }
    impl SeedEntropy for FailAt {
        fn draw(&mut self, _: SeedPurpose) -> Result<Zeroizing<[u8; 16]>, Error> {
            self.calls += 1;
            if self.calls == self.fail {
                Err(Error::Entropy)
            } else {
                Ok(Zeroizing::new([42; 16]))
            }
        }
    }
    for fail in 1..=3 {
        let mut entropy = FailAt { fail, calls: 0 };
        assert!(matches!(
            SetupParty::with_entropy(context(), Party::P0, PrssLimits::default(), &mut entropy),
            Err(Error::Entropy)
        ));
        assert_eq!(entropy.calls, fail);
    }
}

#[test]
fn online_rejection_and_cancellation_retire_coupled_prss_keys() {
    let mut owners = parties();
    let ops = owners.each_mut().map(|p| p.plan(1, 3).unwrap());
    let op = ops[0];
    let mut messages = begin(&mut owners, op, &[9]);
    let mut peer = take(&mut messages, op.evaluator_a(), MessagePhase::MaskedInput);
    let corr = take(&mut messages, op.evaluator_a(), MessagePhase::Correlation);
    peer.header.binding = [0; 32];
    let a = &mut owners[op.evaluator_a() as usize];
    assert!(matches!(a.open(peer, corr), Err(Error::Binding)));
    assert!(a.seeds.is_none());
    assert_eq!(a.status(), Status::Retired);
    for owner in &mut owners {
        owner.cancel();
        owner.cancel();
        assert!(owner.seeds.is_none());
        assert!(!owner.has_pending_operation());
        assert!(owner.take_output().is_err());
    }
    let mut ready = parties();
    ready[0].cancel();
    assert!(ready[0].seeds.is_none());
    let mut pending = parties();
    pending[0].plan(1, 3).unwrap();
    pending[0].cancel();
    assert!(pending[0].seeds.is_none());
    assert!(!pending[0].has_pending_operation());
    let mut setup = setups();
    setup[0].cancel();
    assert!(setup[0].seeds.is_none());
    assert!(matches!(setup[0].contribution(), Err(Error::Retired)));
}
