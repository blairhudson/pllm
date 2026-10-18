use super::*;
use crate::puma::multiply::tests::parties;
use crate::puma::prss::{fill, PrssLimits};

const ONE: i64 = 1 << 18;

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
            next.extend(owners[p].receive(frame).unwrap().into_iter().flatten());
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
    let mut saw_complete = false;
    for stop in 0..250 {
        let mut owners = owners(&program, &[-9 * ONE, 0, 9 * ONE]);
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
