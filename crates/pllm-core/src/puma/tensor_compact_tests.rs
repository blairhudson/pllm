use super::*;

#[path = "tensor_compact_row_tests.rs"]
mod row_tests;

#[test]
fn compact_signed_arbitrary_ring_components_random_carries_and_stable_ties() {
    let mut seed = 0x8db7_3df1_0cb7_825a_u64;
    let mut random = || {
        seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    for width in [8, 16, 32, 64] {
        let shift = 64 - width;
        let values: Vec<_> = (0..128)
            .map(|_| ((random() << shift) as i64) >> shift)
            .collect();
        let half = 1_i128 << (width - 1);
        let bounds = Interval::new((-half) as i64, (half - 1) as i64).unwrap();
        let program = Program::negative_compact(128, bounds).unwrap();
        let runtimes = parties(program.work_budget().limits().unwrap());
        let op = Operation::new(
            runtimes[0].context(),
            0,
            Party::P0,
            128,
            18,
            Limits::default(),
        )
        .unwrap();
        let a: Vec<u64> = (0..128).map(|_| random()).collect();
        let b: Vec<u64> = (0..128).map(|_| random()).collect();
        let c: Vec<u64> = (0..128)
            .map(|i| (values[i] as u64).wrapping_sub(a[i]).wrapping_sub(b[i]))
            .collect();
        let rows = [&a[..], &b[..], &c[..]];
        let mut owners = runtimes.map(|r| {
            let p = r.party();
            let input = ReplicatedShare::from_local(
                op,
                p,
                rows[p as usize],
                rows[p.next() as usize],
                Limits::default(),
            )
            .unwrap();
            Executor::new(r, program.clone(), input).unwrap()
        });
        drive_encoded(&mut owners);
        let output = owners.map(|o| o.finish().unwrap().1);
        assert_eq!(
            clear(&output),
            values.iter().map(|&x| i64::from(x < 0)).collect::<Vec<_>>()
        );
    }
    // Pairwise strict comparison: tied candidates retain the earlier candidate.
    let values = [-7, -7, -2, 1, 5, 3, 0, 0];
    let mut b = Builder::new(8, Interval::new(-7, 7).unwrap()).unwrap();
    let other = b.row_map(b.input(), 2, RowMap::Rotate(1)).unwrap();
    let less = b.less_than_compact(b.input(), other).unwrap();
    let less = b.from_bit(less).unwrap();
    let program = b.finish(less).unwrap();
    let (result, _) = run_encoded(&program, &values);
    assert_eq!(result, [0, 0, 1, 0, 0, 1, 0, 0]);
}

#[derive(Default, Debug)]
struct Traffic {
    directed: [[usize; 3]; 3],
    conversion: [[usize; 3]; 3],
    uncompressed: usize,
    tapes: Vec<Vec<u64>>,
}

// Encoded transport, independently replayed truncation tapes, and actual
// per-stage native work. Arithmetic products/OT must retain every ring bit.
fn drive_encoded(owners: &mut [Executor; 3]) -> Traffic {
    let mut traffic = Traffic::default();
    while !owners[0].complete() {
        let node = owners[0].program.nodes[owners[0].step].unwrap();
        let budget = owners[0].program.instruction_work(node).unwrap();
        let conversion = matches!(
            node.instruction,
            Instruction::Sign(_)
                | Instruction::CompactSign(..)
                | Instruction::Bits(_)
                | Instruction::LowBits(..)
        );
        let before = owners.each_ref().map(|o| {
            (
                o.runtime.as_ref().unwrap().source.runtime.next_sequence(),
                o.cursors(),
            )
        });
        let mut frames: Vec<_> = owners
            .iter_mut()
            .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
            .collect();
        let mut tape = masks(owners);
        while !frames.is_empty() {
            let mut next = Vec::new();
            for frame in frames {
                let from = frame.sender();
                let to = frame.recipient() as usize;
                traffic.uncompressed += 48 + frame.inner.encode_bytes().unwrap().len();
                let compact = frame.compact_bits.is_some();
                if matches!(
                    node.instruction,
                    Instruction::Bits(_) | Instruction::Sign(_)
                ) {
                    assert!(
                        !compact,
                        "full-word guard/legacy conversion must not narrow"
                    );
                }
                if matches!(node.instruction, Instruction::FromBit(_)) {
                    assert!(frame.compact_bits.is_none() || frame.compact_bits == Some(64));
                }
                if !conversion
                    && !matches!(
                        node.instruction,
                        Instruction::BitOrLow(..) | Instruction::FromBit(_)
                    )
                {
                    assert!(!compact, "arithmetic/guard stage unexpectedly compact");
                }
                let encoded = frame.encode().unwrap();
                assert_eq!(
                    &encoded.as_bytes()[..4],
                    if compact { b"PUT2" } else { b"PUT1" }
                );
                traffic.directed[from as usize][to] += encoded.as_bytes().len();
                if conversion {
                    traffic.conversion[from as usize][to] += encoded.as_bytes().len();
                }
                next.extend(
                    owners[to]
                        .receive_encoded(encoded.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .flatten(),
                );
            }
            if tape.is_none() {
                tape = masks(owners);
            }
            frames = next;
        }
        if let Some(tape) = tape {
            traffic.tapes.push(tape);
        }
        let mut aes = 0;
        let mut private = 0;
        for (o, (ordinal, start)) in owners.iter_mut().zip(before) {
            o.finish_step().unwrap();
            let end = o.cursors();
            assert_eq!(
                o.runtime.as_ref().unwrap().source.runtime.next_sequence() - ordinal,
                budget.operations()
            );
            assert_eq!(
                end.pair_blocks - start.pair_blocks,
                budget.cursors().pair_blocks
            );
            assert_eq!(
                end.public_blocks - start.public_blocks,
                budget.cursors().public_blocks
            );
            aes = aes.max(end.aes_work - start.aes_work);
            private = private.max(end.private_blocks - start.private_blocks);
        }
        assert_eq!(aes, budget.cursors().aes_work);
        assert_eq!(private, budget.cursors().private_blocks);
    }
    traffic
}

fn run_encoded(program: &Program, values: &[i64]) -> (Vec<i64>, Traffic) {
    let mut owners = owners(program, values);
    let traffic = drive_encoded(&mut owners);
    eprintln!("compact measurement profile={} lanes={} work={:?} actual={:?} directed={:?} a2b={:?} uncompressed={}",
        program.numeric_profile(), program.lanes(), program.work_budget(), owners.each_ref().map(Executor::cursors), traffic.directed, traffic.conversion, traffic.uncompressed);
    let shares = owners.map(|o| o.finish().unwrap().1);
    (clear(&shares), traffic)
}

#[test]
fn compact_signed_buckets_endpoints_full_ring_shares_and_real_wire_costs() {
    for (width, pairs, edge) in [
        (8, 64, 940),
        (16, 96, 1544),
        (32, 160, 2660),
        (64, 288, 4800),
    ] {
        let half = 1_i128 << (width - 1);
        let lo = i64::try_from(-half).unwrap();
        let hi = i64::try_from(half - 1).unwrap();
        let bounds = Interval::new(lo, hi).unwrap();
        assert_eq!(Program::signed_width(bounds), width);
        let program = Program::negative_compact(128, bounds).unwrap();
        assert_eq!(program.numeric_profile(), COMPACT_NONLINEAR_PROFILE);
        assert_eq!(
            WorkBudget::conversion_signed_compact(128, width)
                .unwrap()
                .cursors()
                .pair_blocks,
            pairs
        );
        let values: Vec<_> = (0..128)
            .map(|i| [lo, lo + 1, -1, 0, 1, hi - 1, hi][i % 7])
            .collect();
        let (result, traffic) = run_encoded(&program, &values);
        assert_eq!(
            result,
            values.iter().map(|&x| i64::from(x < 0)).collect::<Vec<_>>()
        );
        for p in 0..3 {
            assert_eq!(traffic.conversion[p][(p + 2) % 3], edge);
        }
        assert_eq!(program.work_budget().operations(), 2);
        assert_eq!(program.work_budget().cursors().pair_blocks, pairs + 192);
        assert_eq!(program.work_budget().cursors().aes_work, pairs * 2 + 257);
        assert_eq!(
            traffic.directed.iter().flatten().sum::<usize>(),
            3 * edge + 4464
        );
        let legacy = Program::negative(128, bounds).unwrap();
        let (result, traffic) = run_encoded(&legacy, &values);
        assert_eq!(
            result,
            values.iter().map(|&x| i64::from(x < 0)).collect::<Vec<_>>()
        );
        for p in 0..3 {
            assert_eq!(traffic.conversion[p][(p + 2) % 3], 15072);
        }
        assert_eq!(traffic.directed.iter().flatten().sum::<usize>(), 49680);
        assert_eq!(legacy.work_budget().cursors().aes_work, 833);
    }
}

#[test]
fn compact_signed_public_proofs_reject_overflow_and_distinguish_equality_width() {
    assert_eq!(Program::equality_width(Interval::new(-3, 3).unwrap()), 2);
    assert_eq!(Program::signed_width(Interval::new(-3, 3).unwrap()), 8);
    for (lo, hi, width) in [
        (-129, 127, 16),
        (-128, 128, 16),
        (-32769, 32767, 32),
        (-32768, 32768, 32),
        (i32::MIN as i64 - 1, 0, 64),
        (0, i32::MAX as i64 + 1, 64),
    ] {
        assert_eq!(Program::signed_width(Interval::new(lo, hi).unwrap()), width);
    }
    for (left, right) in [
        (i64::MIN, 1),
        (i64::MAX, -1),
        (i64::MIN, i64::MAX),
        (i64::MAX, i64::MIN),
    ] {
        assert_eq!(
            Program::signed_difference_bounds(
                Interval::new(left, left).unwrap(),
                Interval::new(right, right).unwrap()
            ),
            Err(Error::Bounds)
        );
    }
    assert_eq!(
        Program::signed_difference_bounds(
            Interval::new(i64::MIN, i64::MIN).unwrap(),
            Interval::new(-1, -1).unwrap()
        )
        .unwrap(),
        Interval::new(i64::MIN + 1, i64::MIN + 1).unwrap()
    );
    let mut b = Builder::new(1, Interval::new(i64::MIN, i64::MAX).unwrap()).unwrap();
    let min = b
        .affine(
            b.input(),
            None,
            Affine {
                left: 0,
                right: 0,
                constant: i64::MIN,
            },
        )
        .unwrap();
    let before = b.0.len;
    assert_eq!(b.less_than_compact(b.input(), min), Err(Error::Bounds));
    assert_eq!(b.0.len, before);
    for width in [0, 1, 7, 9, 18, 31, 33, 63, 65] {
        let mut runtimes = parties(PrssLimits::default());
        assert!(runtimes[0]
            .plan_conversion(3, Kind::CompactMostSignificantBit(width))
            .is_err());
        assert_eq!(runtimes[0].status(), super::super::super::Status::Retired);
        assert_eq!(runtimes[0].cursors(), Default::default());
        assert!(WorkBudget::conversion_signed_compact(3, width).is_err());
    }
    // Deliberate invalid-domain controls: projection is not membership proof.
    let program = Program::negative_compact(3, Interval::new(-128, 127).unwrap()).unwrap();
    let (result, _) = run_encoded(&program, &[128, -129, 256]);
    assert_eq!(result, [1, 0, 0]);
}

#[test]
fn compact_signed_composes_full_word_guard_before_safe_arithmetic_and_sticky_validity() {
    let values = [i64::MIN, -129, -128, -1, 0, 127, 128, i64::MAX];
    let mut b = Builder::new(values.len(), Interval::new(i64::MIN, i64::MAX).unwrap()).unwrap();
    let guarded = b
        .checked_range(b.input(), Interval::new(-128, 127).unwrap(), 0)
        .unwrap();
    let bit = b.sign_compact(guarded.safe).unwrap();
    let predicate = b.from_bit(bit).unwrap();
    let output = b.product(predicate, guarded.valid, false).unwrap();
    let program = b.finish_checked(output, guarded.valid).unwrap();
    assert_eq!(program.numeric_profile(), COMPACT_NONLINEAR_PROFILE);
    assert!(program
        .nodes
        .iter()
        .flatten()
        .any(|n| matches!(n.instruction, Instruction::Bits(_))));
    let mut owners = owners(&program, &values);
    drive_encoded(&mut owners);
    let finished = owners.map(|o| o.finish_checked().unwrap());
    let outputs = clear_refs(finished.each_ref().map(|v| &v.1));
    let valid = clear_refs(finished.each_ref().map(|v| &v.2));
    assert_eq!(valid, values.map(|x| i64::from((-128..=127).contains(&x))));
    assert_eq!(outputs, [0, 0, 1, 1, 0, 0, 0, 0]);
}

#[test]
fn compact_guarded_activations_match_new_tape_integer_recipes_and_all_cutoffs() {
    for kind in [Activation::Gelu, Activation::Silu] {
        let pivots = if kind == Activation::Gelu {
            [-1048576, -511180, 786432]
        } else {
            [-2097152, -1048576, 1048576]
        };
        let mut points = vec![-256 * ONE, -1, 0, 1, 256 * ONE];
        for p in pivots {
            points.extend([p - 1, p, p + 1]);
        }
        let values: Vec<_> = (0..128).map(|i| points[i % points.len()]).collect();
        let bounds = Interval::new(-256 * ONE, 256 * ONE).unwrap();
        let program = Program::activation_guarded_compact(kind, 128, bounds).unwrap();
        let (result, traffic) = run_encoded(&program, &values);
        assert_eq!(
            traffic.directed.iter().flatten().sum::<usize>(),
            if kind == Activation::Gelu {
                160300
            } else {
                139464
            }
        );
        for p in 0..3 {
            assert_eq!(traffic.conversion[p][(p + 2) % 3], 7980);
        }
        for (i, &x) in values.iter().enumerate() {
            let expected = if x < pivots[0] {
                0
            } else if x > pivots[2] {
                x
            } else {
                oracle(kind, x, traffic.tapes.iter().map(|t| t[i]))
            };
            assert_eq!(result[i], expected, "{kind:?} x={x}");
        }
        let legacy = Program::activation_guarded(kind, 128, bounds).unwrap();
        assert_ne!(program.digest(), legacy.digest());
        assert_eq!(program.instructions(), legacy.instructions());
        assert_eq!(program.output_bounds(), legacy.output_bounds());
        let (result, traffic) = run_encoded(&legacy, &values);
        assert_eq!(
            traffic.directed.iter().flatten().sum::<usize>(),
            if kind == Activation::Gelu {
                272008
            } else {
                251172
            }
        );
        for (i, &x) in values.iter().enumerate() {
            let expected = if x < pivots[0] {
                0
            } else if x > pivots[2] {
                x
            } else {
                oracle(kind, x, traffic.tapes.iter().map(|t| t[i]))
            };
            assert_eq!(result[i], expected);
        }
        assert!(Program::activation_guarded_compact(
            kind,
            1,
            Interval::new(i64::MIN, i64::MAX).unwrap()
        )
        .is_err());
    }
}

#[test]
fn compact_positive_normalization_preserves_every_exponent_and_original_reversal() {
    for rsqrt in [false, true] {
        let width = if rsqrt { 35 } else { 36 };
        let mut points = Vec::new();
        for j in 0..width {
            let power = 1_i64 << j;
            points.extend([power, (power + 1).min((1 << width) - 1), 2 * power - 1]);
        }
        let values: Vec<_> = (0..128).map(|i| points[i % points.len()]).collect();
        let bounds = Interval::new(1, (1 << width) - 1).unwrap();
        let compact = if rsqrt {
            Program::rsqrt_compact(128, bounds)
        } else {
            Program::reciprocal_compact(128, bounds)
        }
        .unwrap();
        let (result, traffic) = run_encoded(&compact, &values);
        assert_eq!(
            traffic.directed.iter().flatten().sum::<usize>(),
            if rsqrt { 213840 } else { 160656 }
        );
        for p in 0..3 {
            assert_eq!(
                traffic.conversion[p][(p + 2) % 3],
                if rsqrt { 8576 } else { 8800 }
            );
        }
        for (i, &x) in values.iter().enumerate() {
            assert_eq!(
                result[i],
                normalization_oracle(rsqrt, x, traffic.tapes.iter().map(|t| t[i]))
            );
        }
        let legacy = if rsqrt {
            Program::rsqrt(128, bounds)
        } else {
            Program::reciprocal(128, bounds)
        }
        .unwrap();
        assert_ne!(compact.digest(), legacy.digest());
        assert_eq!(compact.work_budget(), legacy.work_budget());
        assert_eq!(compact.output_bounds(), legacy.output_bounds());
        assert_eq!(compact.instructions(), legacy.instructions());
        let (result, traffic) = run_encoded(&legacy, &values);
        assert_eq!(
            traffic.directed.iter().flatten().sum::<usize>(),
            if rsqrt { 241680 } else { 187536 }
        );
        for (i, &x) in values.iter().enumerate() {
            assert_eq!(
                result[i],
                normalization_oracle(rsqrt, x, traffic.tapes.iter().map(|t| t[i]))
            );
        }
        for bounds in [
            Interval::new(0, 1).unwrap(),
            Interval::new(-1, 1).unwrap(),
            Interval::new(1, 1 << width).unwrap(),
        ] {
            assert!(if rsqrt {
                Program::rsqrt_compact(1, bounds)
            } else {
                Program::reciprocal_compact(1, bounds)
            }
            .is_err());
        }
    }
}

#[test]
fn compact_signed_width_stage_padding_and_namespace_fail_closed() {
    let program = Program::negative_compact(3, Interval::new(-128, 127).unwrap()).unwrap();
    for variant in 0..11 {
        let mut owners = owners(&program, &[-128, 0, 127]);
        let mut frames: Vec<_> = owners
            .iter_mut()
            .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
            .collect();
        if variant == 10 {
            // Split and initial AND use the SAME width/lane layout. Replaying
            // the accepted split must fail on stage binding, not just shape.
            let from = frames[0].sender();
            let to = frames[0].recipient() as usize;
            let stale = frames.remove(0).encode().unwrap();
            let mut pending = Vec::new();
            pending.extend(
                owners[to]
                    .receive_encoded(stale.as_bytes(), from)
                    .unwrap()
                    .into_iter()
                    .flatten(),
            );
            for frame in frames {
                let p = frame.recipient() as usize;
                pending.extend(owners[p].receive(frame).unwrap().into_iter().flatten());
            }
            assert_eq!(pending[0].compact_bits, Some(8));
            assert!(owners[to].receive_encoded(stale.as_bytes(), from).is_err());
            assert!(owners[to].retired());
            continue;
        }
        if variant == 9 {
            while frames[0].compact_bits != Some(2) {
                let mut next = Vec::new();
                for frame in frames {
                    let p = frame.recipient() as usize;
                    next.extend(owners[p].receive(frame).unwrap().into_iter().flatten());
                }
                frames = next;
            }
        }
        let frame = frames.remove(0);
        let from = frame.sender();
        let to = frame.recipient() as usize;
        let encoded = frame.encode().unwrap();
        let mut raw = encoded.as_bytes().to_vec();
        match variant {
            0 => {
                raw.pop();
            }
            1 => raw.push(0),
            2 => raw[87] = 16,
            3 => raw[87] = 7,
            4 => raw[..4].copy_from_slice(b"PUT1"),
            5 => raw[48..52].copy_from_slice(b"PUM1"),
            6 => raw[44] ^= 1,
            7 => raw[36] ^= 1,
            8 => raw[86] = MessagePhase::BooleanAnd as u8,
            _ => *raw.last_mut().unwrap() |= 0x80,
        }
        assert!(
            owners[to].receive_encoded(&raw, from).is_err(),
            "variant={variant}"
        );
        assert!(owners[to].retired());
        assert!(owners[to]
            .receive_encoded(encoded.as_bytes(), from)
            .is_err());
    }
}

#[test]
fn compact_nonlinear_fresh_repetition_cancellation_and_replay() {
    let program = Program::negative_compact(3, Interval::new(-32768, 32767).unwrap()).unwrap();
    cancel_at_every_event(&program, &[-32768, 0, 32767]);
    let mut owners = owners_for_repeats(&program, &[0, 0, 0], 2);
    drive_encoded(&mut owners);
    let finished = owners.map(|o| o.finish().unwrap());
    let old = finished.each_ref().map(|v| v.1.first().to_vec());
    let mut owners = finished.map(|(r, v)| Executor::new(r, program.clone(), v).unwrap());
    drive_encoded(&mut owners);
    let finished = owners.map(|o| o.finish().unwrap());
    assert_eq!(clear_refs(finished.each_ref().map(|v| &v.1)), [0, 0, 0]);
    assert!(finished.iter().zip(old).any(|(v, old)| v.1.first() != old));
    for (runtime, _) in finished {
        assert_eq!(
            runtime.cursors().pair_blocks,
            2 * program.work_budget().cursors().pair_blocks
        );
    }
    for kind in [Activation::Gelu, Activation::Silu] {
        let program = Program::activation_guarded_compact(
            kind,
            3,
            Interval::new(-256 * ONE, 256 * ONE).unwrap(),
        )
        .unwrap();
        cancel_at_every_event(&program, &[-256 * ONE, 0, 256 * ONE]);
    }
    for program in [
        Program::reciprocal_compact(3, Interval::new(1, (1 << 36) - 1).unwrap()).unwrap(),
        Program::rsqrt_compact(3, Interval::new(1, (1 << 35) - 1).unwrap()).unwrap(),
    ] {
        cancel_at_every_event(&program, &[1, ONE, (1 << 34) - 1]);
    }
}
