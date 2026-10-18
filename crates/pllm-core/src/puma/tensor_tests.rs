use super::*;
use crate::puma::multiply::tests::parties;
use crate::puma::prss::{fill, PrssLimits};

const ONE: i64 = 1 << 18;

fn normalization_oracle(rsqrt: bool, x: i64, masks: impl IntoIterator<Item = u64>) -> i64 {
    let mut masks = masks.into_iter();
    let mut trunc = |raw: i128| -> i64 {
        let r = masks.next().unwrap() & ((1 << 18) - 1);
        i64::try_from(
            raw.div_euclid(1 << 18)
                + i128::from(raw.rem_euclid(1 << 18) + i128::from(r) >= 1 << 18),
        )
        .unwrap()
    };
    let j = 63 - (x as u64).leading_zeros();
    let result = if rsqrt {
        let length = j + 1;
        let u = trunc(i128::from(x) * (1_i128 << (35 - length)));
        let u2 = trunc(i128::from(u) * i128::from(u));
        let u3 = trunc(i128::from(u2) * i128::from(u));
        let u4 = trunc(i128::from(u3) * i128::from(u));
        let p = trunc(
            -4057974 * i128::from(u) + 10085068 * i128::from(u2) - 13072088 * i128::from(u3)
                + 6823457 * i128::from(u4),
        ) + 1086023;
        let comp = (if length % 2 == 0 { 724_i128 } else { 512 }) << (17 - length / 2);
        trunc(i128::from(p) * comp)
    } else {
        let f = 1_i128 << (35 - j);
        let c = trunc(i128::from(x) * f);
        let w = trunc((763940 - 2 * i128::from(c)) * f);
        let e = ONE - trunc(i128::from(x) * i128::from(w));
        let r = trunc(i128::from(w) * i128::from(ONE + e));
        let e2 = trunc(i128::from(e) * i128::from(e));
        trunc(i128::from(r) * i128::from(ONE + e2))
    };
    assert!(masks.next().is_none());
    result
}

#[test]
fn protected_normalization_matches_independent_rounding_at_every_exponent() {
    for rsqrt in [false, true] {
        let width = if rsqrt { 35 } else { 36 };
        let mut values = vec![];
        for j in 0..width {
            let power = 1_i64 << j;
            values.extend([power, (power + 1).min((1 << width) - 1), 2 * power - 1]);
        }
        let bounds = Interval::new(1, (1 << width) - 1).unwrap();
        let program = if rsqrt {
            Program::rsqrt(values.len(), bounds)
        } else {
            Program::reciprocal(values.len(), bounds)
        }
        .unwrap();
        let mut owners = owners(&program, &values);
        let usage = owners.each_ref().map(Executor::payload_usage);
        let mut tapes = vec![];
        for _ in 0..program.instructions() {
            if let Some(tape) = step(&mut owners).0 {
                tapes.push(tape);
            }
        }
        let output = owners.map(|o| o.finish().unwrap().1);
        let expected: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, &x)| normalization_oracle(rsqrt, x, tapes.iter().map(|t| t[i])))
            .collect();
        assert_eq!(clear(&output), expected, "rsqrt={rsqrt}");
        assert!(usage
            .iter()
            .all(|u| u.peak_bytes() + FIXED_RESERVE_BYTES <= program.admitted_bytes()));
        drop(output);
        assert!(usage.iter().all(|u| u.live_bytes() == 0));
    }
}

#[test]
fn normalization_rejects_unproved_domains_and_retains_source_zero_tape_answers() {
    let input = [65536, 131072, 262144, 524288, 1048576];
    assert_eq!(
        input.map(|x| normalization_oracle(false, x, [0; 6])),
        [1048570, 524285, 262142, 131070, 65535]
    );
    assert_eq!(
        input.map(|x| normalization_oracle(true, x, [0; 6])),
        [524249, 370660, 262124, 185330, 131062]
    );
    for bounds in [
        Interval::new(0, 1).unwrap(),
        Interval::new(-1, 1).unwrap(),
        Interval::new(1, 1 << 36).unwrap(),
    ] {
        assert!(Program::reciprocal(1, bounds).is_err());
        assert!(Program::rsqrt(1, bounds).is_err());
    }
    assert!(Program::rsqrt(1, Interval::new(1, 1 << 35).unwrap()).is_err());
}

fn rounded(raw: i128, mask: u64, shift: u8) -> i64 {
    let unit = 1_i128 << shift;
    (raw.div_euclid(unit)
        + i128::from(raw.rem_euclid(unit) + i128::from(mask & ((1 << shift) - 1)) >= unit))
        as i64
}

#[test]
fn half_rotary_preserves_signed_pairs_and_independent_product_roundings() {
    for (rows, width) in [(1, 2), (3, 6), (2, 64), (1, 128)] {
        let lanes = rows * width;
        let values: Vec<_> = (0..lanes)
            .map(|i| [ONE + 1, -2 * ONE - 3, 0, -1, 1, 3 * ONE][i % 6])
            .collect();
        for identity in [false, true] {
            let cosine: Vec<_> = (0..lanes / 2)
                .map(|i| {
                    if identity {
                        ONE
                    } else {
                        [ONE, 0, -ONE, ONE / 2, 185363][i % 5]
                    }
                })
                .collect();
            let sine: Vec<_> = (0..lanes / 2)
                .map(|i| {
                    if identity {
                        0
                    } else {
                        [0, ONE, 0, -227023, 185363][i % 5]
                    }
                })
                .collect();
            let program = Program::rotary_half(
                rows,
                width,
                Interval::new(-3 * ONE, 3 * ONE).unwrap(),
                &cosine,
                &sine,
            )
            .unwrap();
            let mut owners = owners(&program, &values);
            let usage = owners.each_ref().map(Executor::payload_usage);
            let mut tapes = vec![];
            for _ in 0..program.instructions() {
                if let Some(tape) = step(&mut owners).0 {
                    tapes.push(tape);
                }
            }
            assert_eq!(tapes.len(), 2);
            let mut expected = vec![0; lanes];
            for row in 0..rows {
                for column in 0..width / 2 {
                    let left = row * width + column;
                    let right = left + width / 2;
                    let c = i128::from(cosine[row * (width / 2) + column]);
                    let s = i128::from(sine[row * (width / 2) + column]);
                    let x = i128::from(values[left]);
                    let y = i128::from(values[right]);
                    expected[left] =
                        rounded(x * c, tapes[0][left], 18) + rounded(-y * s, tapes[1][left], 18);
                    expected[right] =
                        rounded(y * c, tapes[0][right], 18) + rounded(x * s, tapes[1][right], 18);
                }
            }
            let output = owners.map(|o| o.finish().unwrap().1);
            assert_eq!(clear(&output), expected);
            if identity {
                assert_eq!(expected, values);
            }
            assert!(usage
                .iter()
                .all(|u| u.peak_bytes() + FIXED_RESERVE_BYTES <= program.admitted_bytes()));
            drop(output);
            assert!(usage.iter().all(|u| u.live_bytes() == 0));
        }
    }
    // Coalescing the two truncations changes this valid low-code result.
    assert_eq!(rounded(1, 0, 18) + rounded(1, (1 << 18) - 1, 18), 1);
    assert_eq!(rounded(2, 0, 18), 0);
}

#[test]
fn half_rotary_binds_geometry_and_coefficients_before_execution() {
    let bounds = Interval::new(-ONE, ONE).unwrap();
    let make = |rows, width, c: &[i64], s: &[i64]| Program::rotary_half(rows, width, bounds, c, s);
    let a = make(1, 4, &[ONE; 2], &[0; 2]).unwrap();
    assert_ne!(a.digest(), make(2, 2, &[ONE; 2], &[0; 2]).unwrap().digest());
    assert_ne!(
        a.digest(),
        make(1, 4, &[ONE - 1; 2], &[0; 2]).unwrap().digest()
    );
    assert_ne!(a.digest(), make(1, 4, &[ONE; 2], &[1; 2]).unwrap().digest());
    for (rows, width) in [(0, 4), (1, 0), (1, 3), (1, 130), (usize::MAX, 4)] {
        assert!(make(rows, width, &[ONE; 2], &[0; 2]).is_err());
    }
    assert!(make(1, 4, &[ONE], &[0; 2]).is_err());
    assert!(make(1, 4, &[ONE; 2], &[0]).is_err());
    assert!(make(1, 4, &[ONE + 1; 2], &[0; 2]).is_err());
    assert!(make(1, 4, &[ONE; 2], &[i64::MIN; 2]).is_err());
    assert!(make(1, 130, &[ONE; 65], &[0; 65]).is_err());
    assert!(Program::rotary_half(
        1,
        2,
        Interval::new(0, INPUT_MAX_EXCLUSIVE / ONE).unwrap(),
        &[ONE],
        &[0]
    )
    .is_err());
    cancel_at_every_event(&a, &[ONE, -ONE, 1, -1]);
}

fn row_oracle(
    layer_norm: bool,
    x: &[i64],
    width: usize,
    gamma: &[i64],
    beta: &[i64],
    tapes: &[Vec<u64>],
) -> Vec<i64> {
    let mut out = Vec::new();
    for (row, input) in x.chunks_exact(width).enumerate() {
        let base = row * width;
        if layer_norm {
            assert_eq!(tapes.len(), 11);
            let coefficient = ((1_f32 / width as f32) * ONE as f32) as i64;
            let mean = rounded(
                input.iter().map(|&v| i128::from(v)).sum::<i128>() * i128::from(coefficient),
                tapes[0][row],
                18,
            );
            let d: Vec<_> = input.iter().map(|&v| v - mean).collect();
            let squares: i128 = d
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    i128::from(rounded(
                        i128::from(v) * i128::from(v),
                        tapes[1][base + i],
                        18,
                    ))
                })
                .sum();
            let variance = rounded(squares * i128::from(coefficient), tapes[2][row], 18) + 2;
            let inverse = normalization_oracle(true, variance, tapes[3..9].iter().map(|t| t[row]));
            for (i, &v) in d.iter().enumerate() {
                let normalized =
                    rounded(i128::from(v) * i128::from(inverse), tapes[9][base + i], 18);
                out.push(
                    rounded(
                        i128::from(normalized) * i128::from(gamma[i]),
                        tapes[10][base + i],
                        18,
                    ) + beta[i],
                );
            }
        } else {
            assert_eq!(tapes.len(), 13);
            let maximum = *input.iter().max().unwrap();
            let exp: Vec<_> = input
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let z = v - maximum;
                    let mut p = ONE + rounded(i128::from(z), tapes[0][base + i], 5);
                    for t in &tapes[1..6] {
                        p = rounded(i128::from(p) * i128::from(p), t[base + i], 18);
                    }
                    if z > -14 * ONE {
                        p
                    } else {
                        0
                    }
                })
                .collect();
            let inverse =
                normalization_oracle(false, exp.iter().sum(), tapes[6..12].iter().map(|t| t[row]));
            for (i, value) in exp.into_iter().enumerate() {
                out.push(rounded(
                    i128::from(value) * i128::from(inverse),
                    tapes[12][base + i],
                    18,
                ));
            }
        }
    }
    out
}

#[test]
fn protected_rows_match_one_reciprocal_or_rsqrt_per_row() {
    for layer_norm in [false, true] {
        for (rows, width) in [
            (2, 1),
            (3, 3),
            (2, 8),
            (1, 128),
            (1, 150),
            (1, 288),
            (1, 512),
            (1, 896),
        ] {
            if !layer_norm && width > MAX_LANES / 2 {
                continue;
            }
            let x: Vec<_> = (0..rows * width)
                .map(|i| {
                    if layer_norm {
                        ((i % 7) as i64 - 3) * ONE + i as i64
                    } else {
                        [
                            0,
                            -14 * ONE - 1,
                            -14 * ONE,
                            -14 * ONE + 1,
                            -ONE,
                            -20 * ONE,
                            -3 * ONE,
                            0,
                        ][i % 8]
                    }
                })
                .collect();
            let gamma: Vec<_> = (0..width)
                .map(|i| if i % 2 == 0 { ONE } else { -ONE / 2 })
                .collect();
            let beta: Vec<_> = (0..width).map(|i| i as i64 - 2).collect();
            let bounds = if layer_norm {
                Interval::new(-4 * ONE, 4 * ONE)
            } else {
                Interval::new(-20 * ONE, 0)
            }
            .unwrap();
            let program = if layer_norm {
                Program::layer_norm(rows, width, bounds, &gamma, &beta)
            } else {
                Program::softmax(rows, width, bounds)
            }
            .unwrap();
            let mut owners = owners(&program, &x);
            let usage = owners.each_ref().map(Executor::payload_usage);
            let mut tapes = vec![];
            for _ in 0..program.instructions() {
                if let Some(tape) = step(&mut owners).0 {
                    tapes.push(tape);
                }
            }
            let output = owners.map(|o| o.finish().unwrap().1);
            assert_eq!(
                clear(&output),
                row_oracle(layer_norm, &x, width, &gamma, &beta, &tapes),
                "layer_norm={layer_norm}, {rows}x{width}"
            );
            assert_eq!(program.output_lanes(), rows * width);
            assert!(usage
                .iter()
                .all(|u| u.peak_bytes() + FIXED_RESERVE_BYTES <= program.admitted_bytes()));
            drop(output);
            assert!(usage.iter().all(|u| u.live_bytes() == 0));
        }
    }
}

#[test]
fn row_programs_bind_geometry_coefficients_and_eager_range_gates() {
    let input = Interval::new(-ONE, ONE).unwrap();
    let a = Program::layer_norm(2, 2, input, &[ONE; 2], &[0; 2]).unwrap();
    let b = Program::layer_norm(2, 2, input, &[ONE, ONE + 1], &[0; 2]).unwrap();
    assert_ne!(a.digest(), b.digest());
    assert_ne!(
        Program::softmax(2, 2, input).unwrap().digest(),
        Program::softmax(1, 4, input).unwrap().digest()
    );
    assert!(Program::softmax(0, 2, input).is_err());
    assert!(Program::softmax(1, 0, input).is_err());
    assert!(Program::softmax(1, MAX_ROW_LANES + 1, input).is_err());
    // Clipped lanes still evaluate their powers; clipping cannot admit them.
    assert!(Program::softmax(1, 4, Interval::new(-128 * ONE, 0).unwrap()).is_err());
    assert!(Program::layer_norm(1, 2, input, &[ONE], &[0; 2]).is_err());
    assert!(Program::layer_norm(1, 2, input, &[i64::MAX; 2], &[0; 2]).is_err());
}

fn rms_norm_oracle(x: &[i64], width: usize, gamma: &[i64], tapes: &[Vec<u64>]) -> Vec<i64> {
    assert_eq!(tapes.len(), 10);
    let mut output = Vec::new();
    for (row, values) in x.chunks_exact(width).enumerate() {
        let base = row * width;
        let squares: i128 = values
            .iter()
            .enumerate()
            .map(|(i, &v)| i128::from(rounded(i128::from(v).pow(2), tapes[0][base + i], 18)))
            .sum();
        let coefficient = ((1_f32 / width as f32) * ONE as f32) as i64;
        let denominator = rounded(squares * i128::from(coefficient), tapes[1][row], 18) + 1;
        let inverse = normalization_oracle(true, denominator, tapes[2..8].iter().map(|t| t[row]));
        for (i, &v) in values.iter().enumerate() {
            let normalized = rounded(i128::from(v) * i128::from(inverse), tapes[8][base + i], 18);
            output.push(rounded(
                i128::from(normalized) * i128::from(gamma[i]),
                tapes[9][base + i],
                18,
            ));
        }
    }
    output
}

#[test]
fn positive_epsilon_rms_norm_preserves_zero_constant_and_signed_rows() {
    let bounds = Interval::new(-4 * ONE, 4 * ONE).unwrap();
    for (rows, width) in [(1, 1), (3, 3), (2, 7), (1, 128), (1, 896), (1, 1024)] {
        let gamma: Vec<_> = (0..width).map(|i| [ONE, -3 * ONE / 2, 0][i % 3]).collect();
        let program = Program::rms_norm(rows, width, bounds, &gamma).unwrap();
        assert!(
            program.admitted_bytes() <= MAX_ADMITTED_BYTES,
            "{} lanes require {} bytes",
            program.lanes(),
            program.admitted_bytes()
        );
        assert_eq!(program.numeric_profile(), RMS_NORM_PROFILE);
        assert_eq!(program.output_lanes(), rows * width);
        for values in [
            vec![0; rows * width],
            vec![2 * ONE; rows * width],
            (0..rows * width)
                .map(|i| [1, -1, -4 * ONE, 4 * ONE, ONE + 3, 0, -ONE / 2][i % 7])
                .collect(),
        ] {
            let mut owners = owners(&program, &values);
            let usage = owners.each_ref().map(Executor::payload_usage);
            let mut tapes = vec![];
            for _ in 0..program.instructions() {
                if let Some(tape) = step(&mut owners).0 {
                    tapes.push(tape);
                }
            }
            let output = owners.map(|o| o.finish().unwrap().1);
            let actual = clear(&output);
            assert_eq!(actual, rms_norm_oracle(&values, width, &gamma, &tapes));
            if values.iter().all(|&x| x == 0) {
                assert!(actual.iter().all(|&x| x == 0));
            } else if values[0] == 2 * ONE {
                // Centered LayerNorm would turn this row into zero.
                assert!(actual[0] > ONE / 2);
            }
            assert!(usage
                .iter()
                .all(|u| u.peak_bytes() + FIXED_RESERVE_BYTES <= program.admitted_bytes()));
            drop(output);
            assert!(usage.iter().all(|u| u.live_bytes() == 0));
        }
    }
}

#[test]
fn rms_norm_identity_and_range_gates_preserve_legacy_program_digests() {
    let bounds = Interval::new(-ONE, ONE).unwrap();
    let make = |numeric_profile| {
        let mut b = Builder::new(3, bounds).unwrap();
        b.0.numeric_profile = numeric_profile;
        let output = b
            .affine(
                b.input(),
                None,
                Affine {
                    left: 1,
                    right: 0,
                    constant: 1,
                },
            )
            .unwrap();
        b.finish(output).unwrap()
    };
    let legacy = make(crate::puma::multiply::NUMERIC_PROFILE);
    // Independently serialized with Python hashlib from the pre-D10 contract.
    assert_eq!(
        legacy.digest(),
        [
            111, 223, 122, 206, 149, 235, 99, 196, 200, 197, 164, 182, 191, 69, 211, 186, 155, 212,
            186, 3, 143, 200, 52, 254, 110, 78, 34, 248, 161, 155, 26, 27,
        ]
    );
    assert_ne!(legacy.digest(), make(RMS_NORM_PROFILE).digest());
    let a = Program::rms_norm(2, 2, bounds, &[ONE; 2]).unwrap();
    assert_ne!(
        a.digest(),
        Program::rms_norm(1, 4, bounds, &[ONE; 4]).unwrap().digest()
    );
    assert_ne!(
        a.digest(),
        Program::rms_norm(2, 2, bounds, &[ONE, ONE + 1])
            .unwrap()
            .digest()
    );
    assert!(Program::rms_norm(0, 2, bounds, &[ONE; 2]).is_err());
    assert!(Program::rms_norm(1, 0, bounds, &[]).is_err());
    assert!(Program::rms_norm(1, 2, bounds, &[ONE]).is_err());
    assert!(Program::rms_norm(1, MAX_ROW_LANES + 1, bounds, &[ONE; MAX_ROW_LANES + 1]).is_err());
    assert!(Program::rms_norm(usize::MAX, 2, bounds, &[ONE; 2]).is_err());
    assert!(Program::rms_norm(1, 2, bounds, &[i64::MAX; 2]).is_err());
    assert!(
        Program::rms_norm(1, 2, Interval::new(-i64::MAX, i64::MAX).unwrap(), &[ONE; 2]).is_err()
    );
    assert!(Program::rms_norm(1, 2, Interval::new(0, 512 * ONE).unwrap(), &[ONE; 2]).is_err());
}

// All-party reconstruction is confined to this test oracle.
fn inputs(owners: &[PartyRuntime; 3], x: &[i64]) -> [ReplicatedShare; 3] {
    let op = Operation::new(
        owners[0].context(),
        0,
        Party::P0,
        x.len(),
        18,
        Limits::default(),
    )
    .unwrap();
    let a: Vec<_> = (0..x.len())
        .map(|i| 0x8765432112345678_u64.wrapping_add(i as u64))
        .collect();
    let b: Vec<_> = (0..x.len())
        .map(|i| 0xfedcba98ffffffff_u64.wrapping_sub(i as u64))
        .collect();
    let c: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, &x)| (x as u64).wrapping_sub(a[i]).wrapping_sub(b[i]))
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

fn owners(program: &Program, values: &[i64]) -> [Executor; 3] {
    let runtimes = parties(PrssLimits::default());
    let mut values = inputs(&runtimes, values).map(Some);
    let mut i = 0;
    runtimes.map(|r| {
        let out = Executor::new(r, program.clone(), values[i].take().unwrap()).unwrap();
        i += 1;
        out
    })
}

fn clear(values: &[ReplicatedShare; 3]) -> Vec<i64> {
    for i in 0..3 {
        assert_eq!(values[i].second(), values[(i + 1) % 3].first());
    }
    (0..values[0].first().len())
        .map(|j| {
            values
                .iter()
                .fold(0_u64, |v, p| v.wrapping_add(p.first()[j])) as i64
        })
        .collect()
}

fn masks(owners: &[Executor; 3]) -> Option<Vec<u64>> {
    let op = owners[0].runtime.as_ref().unwrap().truncation_operation()?;
    let mut a = vec![0; op.lanes()];
    let mut b = vec![0; op.lanes()];
    // Read only deterministic public TEST keys, never production exports.
    let a_key = ((op.evaluator_a() as u128 + 1) * 11).to_le_bytes();
    let h_key = ((op.helper() as u128 + 1) * 11).to_le_bytes();
    let cursor = op.source.unwrap().mask_start;
    fill(&a_key, cursor, &mut a).unwrap();
    fill(&h_key, cursor, &mut b).unwrap();
    Some(
        a.into_iter()
            .zip(b)
            .map(|(x, y)| x.wrapping_add(y))
            .collect(),
    )
}

fn step(owners: &mut [Executor; 3]) -> (Option<Vec<u64>>, usize) {
    let mut frames: Vec<_> = owners
        .iter_mut()
        .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
        .collect();
    let mut tape = masks(owners);
    let mut count = 0;
    // Synchronized public protocol rounds; rotate, open and reshare barriers
    // precede delivery of frames generated in the next round.
    while !frames.is_empty() {
        count += frames.len();
        let mut next = Vec::new();
        for frame in frames {
            let p = frame.recipient() as usize;
            let phase = frame.inner.header().phase;
            let instruction = owners[p].program.nodes[owners[p].step].unwrap().instruction;
            next.extend(
                owners[p]
                    .receive(frame)
                    .unwrap_or_else(|e| panic!("party={p}, {phase:?}, {instruction:?}: {e:?}"))
                    .into_iter()
                    .flatten(),
            );
        }
        if tape.is_none() {
            tape = masks(owners);
        }
        frames = next;
    }
    for o in owners {
        o.finish_step().unwrap();
    }
    (tape, count)
}

// Independent scalar recipe: Euclidean quotient plus stochastic carry. Uses
// actual admitted mask tapes, not the tensor instruction interpreter or bounds.
fn oracle(kind: Activation, x: i64, masks: impl Iterator<Item = u64>) -> i64 {
    let mut masks = masks;
    let mut trunc = |x: i128| {
        let r = (masks.next().unwrap() & ((1 << 18) - 1)) as i128;
        (x.div_euclid(ONE as i128) + i128::from(x.rem_euclid(ONE as i128) + r >= ONE as i128))
            as i64
    };
    let x2 = trunc(x as i128 * x as i128);
    let x3 = if kind == Activation::Gelu {
        trunc(x2 as i128 * x as i128)
    } else {
        0
    };
    let x4 = trunc(x2 as i128 * x2 as i128);
    let x6 = if kind == Activation::Gelu {
        trunc(x3 as i128 * x3 as i128)
    } else {
        trunc(x2 as i128 * x4 as i128)
    };
    let (a, b, pivots) = if kind == Activation::Gelu {
        let a = -132488
            + trunc(-110694 * x as i128)
            + trunc(-30952 * x2 as i128)
            + trunc(-2892 * x3 as i128);
        let b = 2235
            + trunc(131072 * x as i128)
            + trunc(94458 * x2 as i128)
            + trunc(-9879 * x4 as i128)
            + trunc(473 * x6 as i128);
        (a, b, [-1048576, -511180, 786432])
    } else {
        let a = -80413 + trunc(-21489 * x as i128) + trunc(-1453 * x2 as i128);
        let b = 2229
            + trunc(131072 * x as i128)
            + trunc(59806 * x2 as i128)
            + trunc(-2913 * x4 as i128)
            + trunc(71 * x6 as i128);
        (a, b, [-2097152, -1048576, 1048576])
    };
    assert!(masks.next().is_none());
    if x < pivots[0] {
        0
    } else if x < pivots[1] {
        a
    } else if x <= pivots[2] {
        b
    } else {
        x
    }
}

#[test]
fn shared_activations_match_independent_stochastic_recipe_at_all_endpoints() {
    for kind in [Activation::Gelu, Activation::Silu] {
        for n in [1, 3, 31, MAX_TILE_LANES] {
            let pivots = match kind {
                Activation::Gelu => [-1048576, -511180, 786432],
                Activation::Silu => [-2097152, -1048576, 1048576],
            };
            let mut points = vec![-9 * ONE, -1, 0, 1, 9 * ONE];
            for p in pivots {
                points.extend([p - 1, p, p + 1]);
            }
            let x: Vec<_> = (0..n).map(|i| points[i % points.len()]).collect();
            let program =
                Program::activation(kind, n, Interval::new(-9 * ONE, 9 * ONE).unwrap()).unwrap();
            assert!(program.admitted_bytes() <= MAX_ADMITTED_BYTES);
            let mut owners = owners(&program, &x);
            let usage = owners.each_ref().map(Executor::payload_usage);
            let mut tapes = vec![];
            let mut count = 0;
            for _ in 0..program.instructions() {
                let (tape, frames) = step(&mut owners);
                if let Some(tape) = tape {
                    tapes.push(tape);
                }
                count += frames;
            }
            assert!(owners.iter().all(Executor::complete));
            // 9 frames/private Q18 product, 6/public rescale, 24/sign,
            // 4/sign B2A and 3/exact selection product; tails execute all.
            assert_eq!(count, if kind == Activation::Gelu { 171 } else { 156 });
            let output = owners.map(|o| {
                let (mut runtime, output) = o.finish().unwrap();
                runtime.cancel();
                output
            });
            let expected: Vec<_> = x
                .iter()
                .enumerate()
                .map(|(i, &x)| oracle(kind, x, tapes.iter().map(|t| t[i])))
                .collect();
            assert_eq!(clear(&output), expected, "{kind:?}, {n}");
            assert!(usage.iter().all(|u| u.live_bytes() >= 2 * n * 8));
            assert!(usage
                .iter()
                .all(|u| u.peak_bytes() + FIXED_RESERVE_BYTES <= program.admitted_bytes()));
            drop(output);
            assert!(usage.iter().all(|u| u.live_bytes() == 0));
        }
    }
}

#[test]
fn eager_polynomial_range_and_whole_program_limits_precede_randomness() {
    for kind in [Activation::Gelu, Activation::Silu] {
        assert!(Program::activation(kind, 3, Interval::new(-64 * ONE, 64 * ONE).unwrap()).is_err());
        assert!(
            Program::activation(kind, MAX_TILE_LANES + 1, Interval::new(0, 0).unwrap()).is_err()
        );
        let program = Program::activation(kind, 3, Interval::new(-ONE, ONE).unwrap()).unwrap();
        for limits in [
            PrssLimits::new(Limits::new(6, 1, MAX_ADMITTED_BYTES).unwrap(), 100000).unwrap(),
            PrssLimits::new(Limits::default(), 1).unwrap(),
            PrssLimits::new(Limits::new(6, MAX_OPERATIONS, 8192).unwrap(), 100000).unwrap(),
        ] {
            let runtimes = parties(limits);
            let input = inputs(&runtimes, &[0, 1, -1]);
            for (r, input) in runtimes.into_iter().zip(input) {
                let usage = r.payload_usage();
                assert!(Executor::new(r, program.clone(), input).is_err());
                assert_eq!(usage.live_bytes(), 0);
            }
        }
    }
}

#[test]
fn cancelled_exported_frames_keep_reservations_until_erasure() {
    let program =
        Program::activation(Activation::Silu, 3, Interval::new(-ONE, ONE).unwrap()).unwrap();
    let mut owners = owners(&program, &[0, 1, -1]);
    let usage = owners[0].payload_usage();
    let frames = owners[0].start_step().unwrap();
    owners[0].cancel();
    assert!(owners[0].retired());
    assert!(owners[0].start_step().is_err());
    drop(owners);
    assert_eq!(usage.live_bytes(), 3 * 8 + std::mem::size_of::<Frame>());
    drop(frames);
    assert_eq!(usage.live_bytes(), 0);
}

#[test]
fn wrong_program_invocation_step_and_recipient_burn_owner() {
    for variant in 0..4 {
        let program =
            Program::activation(Activation::Silu, 3, Interval::new(-ONE, ONE).unwrap()).unwrap();
        let mut owners = owners(&program, &[0, 1, -1]);
        let mut frames: Vec<_> = owners
            .iter_mut()
            .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
            .collect();
        let mut frame = frames.remove(0);
        let to = frame.recipient() as usize;
        match variant {
            0 => frame.program[0] ^= 1,
            1 => frame.invocation += 1,
            2 => frame.step += 1,
            _ => frame.inner.header.to = frame.inner.header.to.next(),
        }
        assert!(owners[to].receive(frame).is_err());
        assert!(owners[to].retired());
        let usage = owners.each_ref().map(Executor::payload_usage);
        drop(frames);
        drop(owners);
        assert!(usage.iter().all(|u| u.live_bytes() == 0));
    }
}

#[test]
fn program_binding_changes_with_coefficients_geometry_and_public_ranges() {
    let program =
        Program::activation(Activation::Silu, 3, Interval::new(-ONE, ONE).unwrap()).unwrap();
    for other in [
        Program::activation(Activation::Gelu, 3, Interval::new(-ONE, ONE).unwrap()).unwrap(),
        Program::activation(Activation::Silu, 2, Interval::new(-ONE, ONE).unwrap()).unwrap(),
        Program::activation(Activation::Silu, 3, Interval::new(-ONE, ONE + 1).unwrap()).unwrap(),
    ] {
        assert_ne!(program.digest(), other.digest());
    }
}

#[test]
fn every_public_instruction_and_protocol_round_can_be_cancelled_without_leaks() {
    let program = Program::activation(
        Activation::Gelu,
        3,
        Interval::new(-9 * ONE, 9 * ONE).unwrap(),
    )
    .unwrap();
    cancel_at_every_event(&program, &[-9 * ONE, 0, 9 * ONE]);
}

#[test]
fn row_normalization_and_boolean_prefix_rounds_retire_on_cancellation() {
    let input = [-ONE, 0, ONE];
    let range = Interval::new(-ONE, ONE).unwrap();
    for program in [
        Program::softmax(1, 3, range).unwrap(),
        Program::layer_norm(1, 3, range, &[ONE; 3], &[0; 3]).unwrap(),
        Program::rms_norm(1, 3, range, &[ONE; 3]).unwrap(),
    ] {
        cancel_at_every_event(&program, &input);
    }
}

fn cancel_at_every_event(program: &Program, input: &[i64]) {
    let mut saw_complete = false;
    for stop in 0..1000 {
        let mut owners = owners(program, input);
        let usage = owners.each_ref().map(Executor::payload_usage);
        let mut event = 0;
        'execution: while !owners[0].complete() {
            let mut frames: Vec<_> = owners
                .iter_mut()
                .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
                .collect();
            loop {
                if event == stop {
                    break 'execution;
                }
                event += 1;
                if frames.is_empty() {
                    break;
                }
                let mut next = Vec::new();
                for frame in frames {
                    next.extend(
                        owners[frame.recipient() as usize]
                            .receive(frame)
                            .unwrap()
                            .into_iter()
                            .flatten(),
                    );
                }
                frames = next;
            }
            for owner in &mut owners {
                owner.finish_step().unwrap();
            }
        }
        saw_complete |= owners[0].complete();
        for owner in &mut owners {
            owner.cancel();
            assert!(owner.start_step().is_err());
        }
        drop(owners);
        assert!(usage.iter().all(|u| u.live_bytes() == 0));
        if saw_complete {
            break;
        }
    }
    assert!(saw_complete);
}

#[test]
fn public_tensor_builder_reuses_shared_bits_and_rejects_invalid_programs() {
    let mut builder = Builder::new(3, Interval::new(-ONE, ONE).unwrap()).unwrap();
    let input = builder.input();
    let sign = builder.sign(input).unwrap();
    assert!(builder.product(sign, input, false).is_err());
    let a = builder.from_bit(sign).unwrap();
    let b = builder.from_bit(sign).unwrap();
    let c = builder
        .affine(
            a,
            Some(b),
            Affine {
                left: 1,
                right: 1,
                constant: 0,
            },
        )
        .unwrap();
    let program = builder.finish(c).unwrap();
    let mut owners = owners(&program, &[-1, 0, 1]);
    for _ in 0..program.instructions() {
        step(&mut owners);
    }
    assert_eq!(clear(&owners.map(|o| o.finish().unwrap().1)), [2, 0, 0]);
    let mut builder = Builder::new(1, Interval::new(0, 0).unwrap()).unwrap();
    let input = builder.input();
    for _ in 1..MAX_NODES {
        builder.sign(input).unwrap();
    }
    assert!(builder.sign(input).is_err());
    assert!(builder.finish(input).is_err());
}

#[test]
fn reused_register_slots_preserve_live_inputs_across_unused_instructions() {
    let mut b = Builder::new(3, Interval::new(-ONE, ONE).unwrap()).unwrap();
    let input = b.input();
    let saved = b
        .affine(
            input,
            None,
            Affine {
                left: 1,
                right: 0,
                constant: 7,
            },
        )
        .unwrap();
    for _ in 0..40 {
        let unused = b.product(input, input, false).unwrap();
        b.sign(unused).unwrap();
    }
    let output = b
        .affine(
            saved,
            Some(input),
            Affine {
                left: 1,
                right: -1,
                constant: 0,
            },
        )
        .unwrap();
    let program = b.finish(output).unwrap();
    assert!(program.register_count < 6);
    let mut owners = owners(&program, &[-ONE, 0, ONE]);
    let usage = owners.each_ref().map(Executor::payload_usage);
    for _ in 0..program.instructions() {
        step(&mut owners);
    }
    let outputs = owners.map(|o| o.finish().unwrap().1);
    assert_eq!(clear(&outputs), [7, 7, 7]);
    drop(outputs);
    assert!(usage.iter().all(|u| u.live_bytes() == 0));
}

#[test]
fn exact_tensor_allowances_cover_wide_rows_ot_and_consumed_input_continuation() {
    let bounds = Interval::new(-ONE, ONE).unwrap();
    let mut b = Builder::new(128, bounds).unwrap();
    let bits = b.bits(b.input()).unwrap();
    let low = b.bit_map(bits, BitMap::LowBits(8)).unwrap();
    let value = b.from_boolean(low).unwrap();
    let ot = b.finish(value).unwrap();
    for (program, peer_retained_outputs) in [
        (
            Program::rms_norm(1, 1024, bounds, &[ONE; 1024]).unwrap(),
            false,
        ),
        (
            Program::layer_norm(1, 896, bounds, &[ONE; 896], &[0; 896]).unwrap(),
            false,
        ),
        (Program::softmax(1, 512, bounds).unwrap(), false),
        (ot, true),
    ] {
        let bytes = program.admitted_bytes();
        let limits = Limits::new(MAX_LANES, crate::puma::MAX_OPERATIONS, bytes).unwrap();
        let runtimes =
            parties(PrssLimits::new(limits, super::super::super::MAX_AES_BLOCKS).unwrap());
        let mut input = inputs(&runtimes, &vec![0; program.lanes()]).map(Some);
        let mut p = 0;
        let mut owners = runtimes.map(|runtime| {
            let out = Executor::new(runtime, program.clone(), input[p].take().unwrap()).unwrap();
            p += 1;
            out
        });
        let usage = owners.each_ref().map(Executor::payload_usage);
        for run in 0..2 {
            for _ in 0..program.instructions() {
                step(&mut owners);
            }
            let completed = owners.map(|o| o.finish().unwrap());
            if run == 1 {
                drop(completed);
                break;
            }
            if peer_retained_outputs {
                // OT can return an original frame into another party's output.
                // Its sender lease stays charged; only the consumed local input
                // may receive graph credit. It cannot hide that retained frame.
                let mut completed = completed.map(Some);
                let party = completed
                    .iter()
                    .position(|pair| {
                        let (runtime, output) = pair.as_ref().unwrap();
                        let usage = runtime.payload_usage();
                        usage.live_bytes()
                            > output.first.charged_bytes(&usage)
                                + output.second.charged_bytes(&usage)
                    })
                    .unwrap();
                let (runtime, output) = completed[party].take().unwrap();
                assert!(matches!(
                    Executor::new(runtime, program.clone(), output),
                    Err(Error::Bounds)
                ));
                assert!(usage[party].live_bytes() > 0);
                drop(completed);
                break;
            }
            // The original output already owns this runtime's charge. Admission
            // must move it into the next graph, while preserving its PRSS cursor.
            owners = completed
                .map(|(runtime, output)| Executor::new(runtime, program.clone(), output).unwrap());
        }
        assert!(usage
            .iter()
            .all(|u| u.live_bytes() == 0 && u.peak_bytes() + FIXED_RESERVE_BYTES <= bytes));
    }
}

#[test]
fn sequential_tensors_preserve_cursors_and_reject_prior_invocation_frames() {
    let program = Program::activation(
        Activation::Silu,
        3,
        Interval::new(-9 * ONE, 9 * ONE).unwrap(),
    )
    .unwrap();
    let mut owners = owners(&program, &[-9 * ONE, 0, 9 * ONE]);
    let mut stale = None;
    for at in 0..program.instructions() {
        if at == 0 {
            let frames: Vec<_> = owners
                .iter_mut()
                .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
                .collect();
            let frame = &frames[0];
            let usage = owners[0].payload_usage();
            stale = Some(Frame {
                program: frame.program,
                invocation: frame.invocation,
                step: frame.step,
                inner: Message {
                    header: frame.inner.header,
                    words: usage.copy(&frame.inner.words, program.lanes).unwrap(),
                },
                _metadata: usage.reserve(std::mem::size_of::<Frame>()).unwrap(),
            });
            let mut frames = frames;
            while !frames.is_empty() {
                let mut next = Vec::new();
                for frame in frames {
                    next.extend(
                        owners[frame.recipient() as usize]
                            .receive(frame)
                            .unwrap()
                            .into_iter()
                            .flatten(),
                    );
                }
                frames = next;
            }
            for o in &mut owners {
                o.finish_step().unwrap();
            }
        } else {
            step(&mut owners);
        }
    }
    let before = owners.each_ref().map(Executor::cursors);
    let mut outputs = [None, None, None];
    let mut i = 0;
    let runtimes = owners.map(|o| {
        let (runtime, output) = o.finish().unwrap();
        outputs[i] = Some(output);
        i += 1;
        runtime
    });
    let mut fresh_inputs = inputs(&runtimes, &[-9 * ONE, 0, 9 * ONE]).map(Some);
    i = 0;
    let mut owners = runtimes.map(|r| {
        assert_eq!(r.cursors(), before[i]);
        let result = Executor::new(r, program.clone(), fresh_inputs[i].take().unwrap()).unwrap();
        i += 1;
        result
    });
    let frames: Vec<_> = owners
        .iter_mut()
        .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
        .collect();
    let stale = stale.unwrap();
    let to = stale.recipient() as usize;
    assert!(owners[to].receive(stale).is_err());
    assert!(owners[to].retired());
    let usage = owners.each_ref().map(Executor::payload_usage);
    drop(frames);
    drop(owners);
    drop(outputs);
    assert!(usage.iter().all(|u| u.live_bytes() == 0));
}

#[test]
fn incomplete_or_repeated_step_completion_burns_owner() {
    for incomplete in [true, false] {
        let program =
            Program::activation(Activation::Silu, 3, Interval::new(-ONE, ONE).unwrap()).unwrap();
        let mut owners = owners(&program, &[0, 1, -1]);
        if incomplete {
            let frames = owners[0].start_step().unwrap();
            assert!(owners[0].finish_step().is_err());
            drop(frames);
        } else {
            step(&mut owners);
            assert!(owners[0].finish_step().is_err());
        }
        assert!(owners[0].retired());
        let usage = owners.each_ref().map(Executor::payload_usage);
        drop(owners);
        assert!(usage.iter().all(|u| u.live_bytes() == 0));
    }
}
