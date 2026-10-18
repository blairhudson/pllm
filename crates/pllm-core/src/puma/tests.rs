use super::*;

#[test]
fn exhaustive_small_ring_equation_and_distribution() {
    // Independent full-mask high-bit reconstruction versus the wide integer
    // low-carry oracle. No party reconstruction code is called by the oracle.
    let mut cases = 0;
    for bits in 4_u8..=8 {
        let n = 1_u64 << bits;
        for shift in 1..=bits - 2 {
            for x in -(n as i64 / 4)..(n as i64 / 4) {
                let law = oracle::distribution(x, bits, shift).unwrap();
                let mut ups = 0;
                for mask in 0..n {
                    let c = (i128::from(x) + i128::from(n / 4) + i128::from(mask))
                        .rem_euclid(i128::from(n)) as u64;
                    let high = (c >> (bits - 1)) ^ (mask >> (bits - 1));
                    let y = i128::from((c & (n / 2 - 1)) >> shift)
                        - i128::from((mask & (n / 2 - 1)) >> shift)
                        + i128::from(high << (bits - 1 - shift))
                        - i128::from(1_u64 << (bits - 2 - shift));
                    assert_eq!(
                        i128::from(oracle::truncate(x, bits, shift, mask).unwrap()),
                        y
                    );
                    ups += u64::from(y > i128::from(law.floor));
                    cases += 1;
                }
                assert_eq!(ups, law.upper_count * n / law.mask_residue_count);
            }
        }
    }
    assert_eq!(cases, 247_552);
}

#[test]
fn ring64_oracle_rejects_outside_gap_and_shift_edges() {
    for x in [INPUT_MIN - 1, INPUT_MAX_EXCLUSIVE, i64::MIN, i64::MAX] {
        assert_eq!(oracle::truncate(x, 64, 18, 0), Err(Error::Bounds));
    }
    for shift in [0, 63, 64, 255] {
        assert_eq!(oracle::truncate(0, 64, shift, 0), Err(Error::Bounds));
    }
    for mask in [0, 1, u64::MAX / 2, u64::MAX] {
        assert_eq!(oracle::truncate(INPUT_MIN, 64, 62, mask), Ok(-1));
        assert_eq!(oracle::truncate(-16, 64, 3, mask), Ok(-2));
    }
    let law = oracle::distribution(-9, 64, 3).unwrap();
    assert_eq!(
        (
            law.floor,
            law.upper,
            law.upper_count,
            law.mask_residue_count
        ),
        (-2, -1, 7, 8)
    );
}

#[test]
fn binding_covers_context_shape_numeric_and_pivot_fields() {
    let limits = Limits::default();
    let c = Context::new([1; 32], [2; 32], 3).unwrap();
    let op = Operation::new(c, 0, Party::P0, 2, 18, limits).unwrap();
    let alternatives = [
        Operation::new(
            Context::new([9; 32], [2; 32], 3).unwrap(),
            0,
            Party::P0,
            2,
            18,
            limits,
        ),
        Operation::new(
            Context::new([1; 32], [9; 32], 3).unwrap(),
            0,
            Party::P0,
            2,
            18,
            limits,
        ),
        Operation::new(
            Context::new([1; 32], [2; 32], 9).unwrap(),
            0,
            Party::P0,
            2,
            18,
            limits,
        ),
        Operation::new(c, 1, Party::P0, 2, 18, limits),
        Operation::new(c, 0, Party::P1, 2, 18, limits),
        Operation::new(c, 0, Party::P0, 1, 18, limits),
        Operation::new(c, 0, Party::P0, 2, 17, limits),
    ];
    for other in alternatives {
        assert_ne!(op.digest(), other.unwrap().digest());
    }
}

#[test]
fn malformed_owned_frame_burns_evaluator() {
    let limits = Limits::default();
    let c = Context::new([1; 32], [2; 32], 3).unwrap();
    let op = Operation::new(c, 0, Party::P0, 2, 18, limits).unwrap();
    let mut runtime = PartyRuntime::new(c, Party::P0, limits);
    let input = ReplicatedShare::from_local(op, Party::P0, &[0; 2], &[0; 2], limits).unwrap();
    let material = LocalMaterial::evaluator(op, Party::P0, &[1; 2], &[2; 2], limits).unwrap();
    runtime
        .begin(op, input, material)
        .unwrap_or_else(|e| panic!("{e}"));
    // Internal construction models a future wire decoder returning a malformed
    // frame. Public constructors already reject this before copying payloads.
    let bad = Message {
        header: Header {
            binding: op.digest(),
            from: Party::P1,
            to: Party::P0,
            phase: MessagePhase::MaskedInput,
        },
        words: zero_words(1).unwrap(),
    };
    let corr = Message {
        header: Header {
            binding: op.digest(),
            from: Party::P2,
            to: Party::P0,
            phase: MessagePhase::Correlation,
        },
        words: zero_words(4).unwrap(),
    };
    assert!(matches!(runtime.open(bad, corr), Err(Error::Shape)));
    assert_eq!(runtime.status(), Status::Retired);
    assert!(matches!(runtime.take_output(), Err(Error::Phase)));
}
