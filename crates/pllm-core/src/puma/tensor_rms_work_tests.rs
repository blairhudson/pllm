use super::rms_work::attempts;

fn old_xor(b: &mut Builder, input: Value, constant: u64) -> Result<Value, Error> {
    b.bit_width(input)?;
    b.push(
        Instruction::BitConstantXor(input, constant),
        Interval::new(0, 0)?,
        true,
    )
}
fn old_outside(b: &mut Builder, bits: Value, bound: i64, below: bool) -> Result<Value, Error> {
    let key = (bound as u64) ^ (1 << 63);
    let ordered = old_xor(b, bits, 1 << 63)?;
    let mut spread = old_xor(b, ordered, key)?;
    for shift in [1, 2, 4, 8, 16, 32] {
        let lower = b.bit_map(spread, BitMap::ShiftRight(shift))?;
        spread = b.bit_or(spread, lower)?;
    }
    let lower = b.bit_map(spread, BitMap::ShiftRight(1))?;
    let highest = b.bit_xor(spread, lower)?;
    b.bit_map(highest, BitMap::Parity(if below { key } else { !key }))
}
// Literal pre-extraction codec; no new RmsSink/checked_recipe/all_valid_recipe.
fn old_checked(
    lanes: usize,
    incoming: Interval,
    accepted: Interval,
    fallback: i64,
) -> Result<Program, Error> {
    let mut b = Builder::new(lanes, incoming)?;
    let input = b.input();
    b.node(input, false)?;
    if fallback < accepted.lower() || fallback > accepted.upper() {
        return Err(Error::Bounds);
    }
    b.0.numeric_profile = CHECKED_RANGE_PROFILE;
    let bits = b.bits(input)?;
    let below = old_outside(&mut b, bits, accepted.lower(), true)?;
    let above = old_outside(&mut b, bits, accepted.upper(), false)?;
    let invalid = b.bit_or(below, above)?;
    let valid_bit = old_xor(&mut b, invalid, 1)?;
    let valid = b.from_bit(valid_bit)?;
    let mut mask = valid_bit;
    for shift in [1, 2, 4, 8, 16, 32] {
        let upper = b.bit_map(mask, BitMap::ShiftLeft(shift))?;
        mask = b.bit_xor(mask, upper)?;
    }
    let delta = old_xor(&mut b, bits, fallback as u64)?;
    let union = b.bit_or(mask, delta)?;
    let xor = b.bit_xor(mask, delta)?;
    let intersection = b.bit_xor(union, xor)?;
    let selected = old_xor(&mut b, intersection, fallback as u64)?;
    let low = b.bit_map(selected, BitMap::LowBits(32))?;
    let low = b.from_boolean(low)?;
    let high = b.bit_map(selected, BitMap::ShiftRight(32))?;
    let high = b.bit_map(high, BitMap::LowBits(31))?;
    let high = b.from_boolean(high)?;
    let sign = b.bit_map(selected, BitMap::ShiftRight(63))?;
    let sign = b.from_bit(sign)?;
    let negative = b.affine(
        sign,
        None,
        Affine {
            left: i64::MIN,
            right: 0,
            constant: 0,
        },
    )?;
    let upper = b.affine(
        high,
        Some(negative),
        Affine {
            left: 1 << 32,
            right: 1,
            constant: 0,
        },
    )?;
    let safe = b.affine(
        upper,
        Some(low),
        Affine {
            left: 1,
            right: 1,
            constant: 0,
        },
    )?;
    b.0.nodes[safe.0 as usize].as_mut().unwrap().bounds = accepted;
    b.finish_checked(safe, valid)
}

#[test]
fn checked_native_original_codec_and_all_valid_vectors_profiles_and_hashes_unchanged() {
    for lanes in [1, 3, 128] {
        for (lo, hi, fallback) in [
            (-32, 32, 0),
            (i64::MIN, i64::MAX, i64::MIN),
            (i64::MIN, i64::MAX, i64::MAX),
        ] {
            let input = Interval::new(i64::MIN, i64::MAX).unwrap();
            let accepted = Interval::new(lo, hi).unwrap();
            let a = Program::checked_range(lanes, input, accepted, fallback).unwrap();
            let e = old_checked(lanes, input, accepted, fallback).unwrap();
            assert_eq!(a.nodes, e.nodes);
            assert_eq!(a.digest(), e.digest());
            assert_eq!(a.bit_widths, e.bit_widths);
            assert_eq!(a.checked_validity, e.checked_validity);
            assert_eq!(a.work_budget(), e.work_budget());
            let mut b = Builder::row(lanes, Interval::new(0, 1).unwrap()).unwrap();
            b.0.numeric_profile = CHECKED_RANGE_PROFILE;
            let sum = b.row_map(b.input(), lanes, RowMap::Sum).unwrap();
            let d = b
                .affine(
                    sum,
                    None,
                    Affine {
                        left: 1,
                        right: 0,
                        constant: -(lanes as i64),
                    },
                )
                .unwrap();
            let failed = b.sign(d).unwrap();
            let failed = b.from_bit(failed).unwrap();
            let v = b
                .affine(
                    failed,
                    None,
                    Affine {
                        left: -1,
                        right: 0,
                        constant: 1,
                    },
                )
                .unwrap();
            let e = b.finish(v).unwrap();
            let a = Program::all_valid(lanes).unwrap();
            assert_eq!(a.nodes, e.nodes);
            assert_eq!(a.digest(), e.digest());
            assert_eq!(a.work_budget(), e.work_budget());
        }
    }
}
use super::*;

#[derive(Clone, Copy, Debug)]
enum Rms {
    Plain,
    Certified,
    InputNorm(u128),
}
impl Rms {
    fn work(
        self,
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        space: &mut RmsWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        match self {
            Self::Plain => Program::rms_norm_work(rows, width, input, gamma, space),
            Self::Certified => Program::rms_norm_certified_work(rows, width, input, gamma, space),
            Self::InputNorm(norm) => {
                Program::rms_norm_with_input_norm_work(rows, width, input, gamma, norm, space)
            }
        }
    }
    fn native(
        self,
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
    ) -> Result<(Program, u128), Error> {
        match self {
            Self::Plain => Program::rms_norm(rows, width, input, gamma).map(|p| (p, 0)),
            Self::Certified => Program::rms_norm_certified(rows, width, input, gamma),
            Self::InputNorm(norm) => {
                Program::rms_norm_with_input_norm(rows, width, input, gamma, norm)
            }
        }
    }
}

// Literal original OUTER composition, not rms_recipe or a work query. The
// normalization proof remains the native owner's shared primitive (not an
// independently reimplemented numeric/certificate recipe).
fn original_outer(
    kind: Rms,
    rows: usize,
    width: usize,
    input: Interval,
    gamma: &[i64],
) -> Result<(Program, u128), Error> {
    let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
    if rows == 0 || width == 0 || gamma.len() != width {
        return Err(Error::Shape);
    }
    let mut b = Builder::row(lanes, input)?;
    b.0.numeric_profile = RMS_NORM_PROFILE;
    let partitioned = !matches!(kind, Rms::Plain);
    let norm = match kind {
        Rms::InputNorm(v) => Some(v),
        _ => None,
    };
    let (normalized, radius) = numeric::rms_normalization(&mut b, width, input, partitioned, norm)?;
    let scales: [Affine; MAX_ROW_LANES] = std::array::from_fn(|i| Affine {
        left: gamma[i % width],
        right: 0,
        constant: 0,
    });
    let scaled = b.affine_lanes(normalized, &scales[..lanes])?;
    let output = b.rescale(scaled, 18)?;
    fn root(v: u128) -> u128 {
        let r = v.isqrt();
        r + u128::from(r * r != v)
    }
    let energy = if partitioned {
        let g = gamma
            .iter()
            .map(|v| v.unsigned_abs())
            .max()
            .ok_or(Error::Shape)?;
        let r = radius
            .checked_mul(u128::from(g))
            .ok_or(Error::Bounds)?
            .div_ceil(1 << 18)
            .checked_add(root(width as u128))
            .ok_or(Error::Bounds)?;
        r.checked_mul(r).ok_or(Error::Bounds)?
    } else {
        0
    };
    Ok((b.finish(output)?, energy))
}

#[test]
fn rms_work_native_metadata_outer_composition_gamma_columns_and_certificates_match() {
    let mut space = RmsWorkWorkspace::empty();
    let mut successes = [0; 3];
    for rows in [1, 2] {
        for width in [1, 3, 7, 128, MAX_ROW_LANES] {
            for input in [
                Interval::new(0, 0).unwrap(),
                Interval::new(-32, 32).unwrap(),
                Interval::new(-(1 << 18), 1 << 18).unwrap(),
            ] {
                let magnitude = u128::from(
                    input
                        .lower()
                        .unsigned_abs()
                        .max(input.upper().unsigned_abs()),
                );
                let obligation = magnitude * magnitude * width as u128;
                let gamma: Vec<_> = (0..width)
                    .map(|i| match i % 3 {
                        0 => 1 << 18,
                        1 => -(1 << 17),
                        _ => 2 << 18,
                    })
                    .collect();
                for (index, kind) in [Rms::Plain, Rms::Certified, Rms::InputNorm(obligation)]
                    .into_iter()
                    .enumerate()
                {
                    space.reset();
                    let before = attempts();
                    let q = kind.work(rows, width, input, &gamma, &mut space);
                    assert_eq!(attempts(), before);
                    let a = kind.native(rows, width, input, &gamma);
                    let e = original_outer(kind, rows, width, input, &gamma);
                    match (q, a, e) {
                        (Ok(q), Ok((a, ae)), Ok((e, ee))) => {
                            successes[index] += 1;
                            space.assert_native_metadata(&a);
                            assert_eq!(q, a.work_budget());
                            assert_eq!(q, e.work_budget());
                            assert_eq!(ae, ee);
                            assert_eq!(a.nodes, e.nodes);
                            assert_eq!(a.digest(), e.digest());
                            assert_eq!(a.numeric_profile(), e.numeric_profile());
                            assert_eq!(a.last_use, e.last_use);
                            assert_eq!(a.slots, e.slots);
                            assert_eq!(a.register_count, e.register_count);
                            assert_eq!(a.public_affines, e.public_affines);
                            assert_eq!(a.public_affine_count, 1);
                            let table = &a.public_affines[0];
                            assert_eq!(table.len(), rows * width);
                            assert_eq!(table.capacity(), table.len());
                            for (i, c) in table.iter().enumerate() {
                                assert_eq!((c.left, c.right, c.constant), (gamma[i % width], 0, 0));
                            }
                        }
                        (Err(q), Err(a), Err(e)) => {
                            assert_eq!(q, a);
                            assert_eq!(a, e);
                            assert!(space.is_poisoned());
                        }
                        _ => panic!(
                            "RMS work/native/outer mismatch {kind:?}, {rows}x{width}, {input:?}"
                        ),
                    }
                }
            }
        }
    }
    assert!(successes.into_iter().all(|n| n > 0));
}

#[test]
fn rms_work_parameter_norm_overflow_and_competing_error_precedence_poison_then_reset() {
    let mut space = RmsWorkWorkspace::empty();
    let cases = [
        (
            usize::MAX,
            2,
            Interval::new(0, 0).unwrap(),
            &[][..],
            Rms::Certified,
            Error::Bounds,
        ),
        (
            MAX_ROW_LANES + 1,
            1,
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            &[][..],
            Rms::InputNorm(u128::MAX),
            Error::Shape,
        ),
        (
            0,
            0,
            Interval::new(0, 0).unwrap(),
            &[][..],
            Rms::Plain,
            Error::Shape,
        ),
        (
            1,
            1,
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            &[1_i64][..],
            Rms::Certified,
            Error::Bounds,
        ),
        (
            1,
            1,
            Interval::new(1 << 18, 1 << 18).unwrap(),
            &[1_i64][..],
            Rms::InputNorm(0),
            Error::Bounds,
        ),
        (
            1,
            1,
            Interval::new(0, 0).unwrap(),
            &[1_i64][..],
            Rms::InputNorm(u128::MAX),
            Error::Bounds,
        ),
        (
            1,
            1,
            Interval::new(1 << 18, 1 << 18).unwrap(),
            &[i64::MAX][..],
            Rms::Plain,
            Error::Bounds,
        ),
    ];
    for (rows, width, input, gamma, kind, error) in cases {
        space.reset();
        let before = attempts();
        assert!(matches!(kind.work(rows,width,input,gamma,&mut space),Err(e) if e==error));
        assert_eq!(attempts(), before);
        assert!(matches!(kind.native(rows,width,input,gamma),Err(e) if e==error));
        assert!(space.is_poisoned());
        let before = attempts();
        for _ in 0..3 {
            assert!(matches!(
                Program::all_valid_work(1, &mut space),
                Err(Error::Phase)
            ));
            assert!(matches!(
                Program::rms_norm_work(usize::MAX, 2, input, &[], &mut space),
                Err(Error::Phase)
            ));
        }
        assert_eq!(attempts(), before);
        space.reset();
        assert!(!space.is_poisoned());
        let mut fresh = RmsWorkWorkspace::empty();
        assert_eq!(
            Program::rms_norm_work(1, 1, Interval::new(0, 0).unwrap(), &[1], &mut space),
            Program::rms_norm_work(1, 1, Interval::new(0, 0).unwrap(), &[1], &mut fresh)
        );
    }
}

#[test]
fn checked_range_and_all_valid_are_distinct_native_factories_with_exact_work_and_caps() {
    let mut space = RmsWorkWorkspace::empty();
    for lanes in [1, 3, 127, 128] {
        for accepted in [
            Interval::new(-32, 32).unwrap(),
            Interval::new(i64::MIN, i64::MAX).unwrap(),
        ] {
            let incoming = Interval::new(i64::MIN, i64::MAX).unwrap();
            let before = attempts();
            let checked =
                Program::checked_range_work(lanes, incoming, accepted, 0, &mut space).unwrap();
            assert_eq!(attempts(), before);
            let program = Program::checked_range(lanes, incoming, accepted, 0).unwrap();
            space.assert_native_metadata(&program);
            assert_eq!(checked, program.work_budget());
            assert_eq!(program.output_bounds(), accepted);
            assert!(program.checked_validity.is_some());
            assert_eq!(program.numeric_profile(), CHECKED_RANGE_PROFILE);
            let before = attempts();
            let reduce = Program::all_valid_work(lanes, &mut space).unwrap();
            assert_eq!(attempts(), before);
            let p = Program::all_valid(lanes).unwrap();
            space.assert_native_metadata(&p);
            assert_eq!(reduce, p.work_budget());
            assert!(p.checked_validity.is_none());
            assert_ne!(checked, reduce);
            let total = checked.checked_add(reduce).unwrap();
            assert!(total
                .admit(
                    0,
                    crate::puma::prss::Cursors::default(),
                    total.operations() - 1,
                    total.cursors().aes_work
                )
                .is_err());
        }
    }
    // Reduction uses a complete row, not Builder::new's tile cap.
    let q = Program::all_valid_work(MAX_ROW_LANES, &mut space).unwrap();
    assert_eq!(q, Program::all_valid(MAX_ROW_LANES).unwrap().work_budget());
    let input = Interval::new(-32, 32).unwrap();
    let before = attempts();
    assert!(matches!(
        Program::checked_range_work(129, input, input, 0, &mut space),
        Err(Error::Bounds)
    ));
    assert_eq!(attempts(), before);
    assert!(matches!(
        Program::all_valid_work(1, &mut space),
        Err(Error::Phase)
    ));
    space.reset();
    assert!(matches!(
        Program::checked_range_work(1, input, input, 33, &mut space),
        Err(Error::Bounds)
    ));
    space.reset();
    assert!(matches!(
        Program::all_valid_work(0, &mut space),
        Err(Error::Bounds)
    ));
    space.reset();
    assert!(matches!(
        Program::all_valid_work(MAX_ROW_LANES + 1, &mut space),
        Err(Error::Bounds)
    ));
}

fn full_tail(space: &mut RmsWorkWorkspace) -> Result<WorkBudget, Error> {
    let input = Interval::new(-32, 32)?;
    Program::rms_norm_certified_work(2, 3, input, &[1 << 18; 3], space)?
        .checked_mul(2)?
        .checked_add(Program::rms_norm_certified_work(
            1,
            3,
            input,
            &[1 << 18; 3],
            space,
        )?)
}
fn admitted(space: &mut RmsWorkWorkspace, ops: u64, aes: u64) -> Result<(), Error> {
    let q = full_tail(space)?;
    q.admit(0, crate::puma::prss::Cursors::default(), ops, aes)?;
    Program::rms_norm_certified(2, 3, Interval::new(-32, 32)?, &[1 << 18; 3])?;
    Program::rms_norm_certified(1, 3, Interval::new(-32, 32)?, &[1 << 18; 3])?;
    Ok(())
}

#[test]
fn rms_full_tail_pre_entry_exact_short_caps_checked_aggregation_and_reset_do_not_refund() {
    let mut space = RmsWorkWorkspace::empty();
    let before = attempts();
    let q = full_tail(&mut space).unwrap();
    assert_eq!(attempts(), before);
    for (ops, aes) in [
        (0, q.cursors().aes_work),
        (q.operations() - 1, q.cursors().aes_work),
        (q.operations(), 0),
        (q.operations(), q.cursors().aes_work - 1),
    ] {
        assert!(matches!(admitted(&mut space, ops, aes), Err(Error::Bounds)));
        assert_eq!(attempts(), before);
    }
    admitted(&mut space, q.operations(), q.cursors().aes_work).unwrap();
    let after = attempts();
    assert_eq!(after[0], before[0] + 2);
    assert!(matches!(q.checked_mul(u64::MAX), Err(Error::Bounds)));
    assert!(matches!(
        WorkBudget::local()
            .checked_mul(u64::MAX)
            .unwrap()
            .checked_add(q),
        Err(Error::Bounds)
    ));
    for c in [
        crate::puma::prss::Cursors {
            public_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            pair_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            private_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            aes_work: u64::MAX,
            ..Default::default()
        },
    ] {
        assert!(q.admit(0, c, u64::MAX, u64::MAX).is_err());
    }
    assert_eq!(attempts(), after);
    let full = Program::rms_norm_certified(2, 3, Interval::new(-32, 32).unwrap(), &[1 << 18; 3])
        .unwrap()
        .0
        .work_budget();
    let tail = Program::rms_norm_certified(1, 3, Interval::new(-32, 32).unwrap(), &[1 << 18; 3])
        .unwrap()
        .0
        .work_budget();
    assert_eq!(q, full.checked_mul(2).unwrap().checked_add(tail).unwrap());
    let merged = Program::rms_norm_certified_work(
        5,
        3,
        Interval::new(-32, 32).unwrap(),
        &[1 << 18; 3],
        &mut space,
    )
    .unwrap();
    assert_ne!(q.cursors(), merged.cursors());
    let prior = q;
    space.reset();
    assert_eq!(
        q.checked_add(full_tail(&mut space).unwrap()).unwrap(),
        prior.checked_mul(2).unwrap()
    );
}

#[test]
fn rms_workspace_quote_before_constructor_is_actual_inline_capacity_not_budget_authority() {
    fn construct(limit: usize, calls: &mut usize) -> Result<RmsWorkWorkspace, Error> {
        if RmsWorkWorkspace::workspace_bytes() > limit {
            return Err(Error::Bounds);
        }
        *calls += 1;
        Ok(RmsWorkWorkspace::empty())
    }
    let quote = RmsWorkWorkspace::workspace_bytes();
    assert_eq!(quote, std::mem::size_of::<RmsWorkWorkspace>());
    assert!(
        quote
            >= SoftmaxWorkWorkspace::workspace_bytes()
                + std::mem::size_of::<usize>()
                + std::mem::size_of::<bool>()
    );
    let mut calls = 0;
    assert!(construct(quote - 1, &mut calls).is_err());
    assert_eq!(calls, 0);
    let mut space = construct(quote, &mut calls).unwrap();
    assert_eq!(calls, 1);
    assert!(
        Program::rms_norm_certified_work(1, 1, Interval::new(0, 0).unwrap(), &[1], &mut space)
            .is_ok()
    );
    // No complete constructor/controller/proof-stack/RSS/input-source quote.
}
