use super::*;
use crate::puma::multiply::tests::parties;
use crate::puma::prss::{fill, PrssLimits};

const ONE: i64 = 1 << 18;

#[path = "tensor_compact_tests.rs"]
mod compact_tests;

#[path = "tensor_shared_tests.rs"]
mod shared_tests;

#[path = "activation_work_tests.rs"]
mod activation_work_tests;

#[test]
fn bounded_equality_derives_minimum_public_width_and_excludes_zero_aliases() {
    for (lo, hi, width) in [
        (0, 0, 1),
        (-1, 1, 1),
        (-2, 2, 2),
        (5, 5, 1),
        (-6, -5, 2),
        (-151935, 151935, 18),
        (-262144, 262144, 19),
        (i64::MIN, i64::MAX, 64),
        (i64::MIN, i64::MIN, 64),
        (-i64::MAX, i64::MAX, 63),
        (i64::MAX, i64::MAX, 1),
    ] {
        assert_eq!(
            Program::equality_width(Interval::new(lo, hi).unwrap()),
            width
        );
    }
    for lo in -12..=12 {
        for hi in lo..=12 {
            let expected = (1..=64)
                .find(|&bits| {
                    (lo..=hi).all(|x| x == 0 || i128::from(x).rem_euclid(1_i128 << bits) != 0)
                })
                .unwrap();
            assert_eq!(
                Program::equality_width(Interval::new(lo, hi).unwrap()),
                expected
            );
        }
    }
    // These are concrete false-zero counterexamples to blindly using 18 bits.
    assert_eq!((1_i64 << 18) & ((1 << 18) - 1), 0);
    for value in [-(1_i64 << 18), 1_i64 << 18] {
        let bounds = Interval::new(value.min(0), value.max(0)).unwrap();
        let program = Program::equal_zero_bounded(2, bounds).unwrap();
        let mut owners = owners(&program, &[0, value]);
        for _ in 0..program.instructions() {
            step(&mut owners);
        }
        let output = owners.map(|o| o.finish().unwrap().1);
        assert_eq!(clear(&output), [1, 0]);
    }
}

#[test]
fn bounded_equality_full_ring_shares_cover_widths_signed_endpoints_and_vocabulary() {
    for width in 1..=64 {
        let radius = if width == 64 {
            i64::MAX
        } else {
            1_i64 << (width - 1)
        };
        let lower = if width == 64 { i64::MIN } else { -radius };
        let bounds = Interval::new(lower, radius).unwrap();
        let values = [
            lower,
            lower.saturating_add(1).min(radius),
            (-1_i64).max(lower),
            0,
            1_i64.min(radius),
            radius,
        ];
        let program = Program::equal_zero_bounded(values.len(), bounds).unwrap();
        let mut owners = owners(&program, &values);
        step(&mut owners);
        let slot = program.slots[1] as usize;
        let bits = owners
            .each_ref()
            .map(|owner| match owner.registers[slot].as_ref().unwrap() {
                Register::Bit(value) => value,
                _ => panic!("expected quotient conversion output"),
            });
        let actual_width = Program::equality_width(bounds);
        let mask = u64::MAX >> (64 - actual_width);
        for p in 0..3 {
            assert_eq!(bits[p].bits(), actual_width);
            assert_eq!(bits[p].second(), bits[(p + 1) % 3].first());
            assert!(bits[p]
                .first()
                .iter()
                .chain(bits[p].second())
                .all(|&word| word & !mask == 0));
        }
        for (i, &x) in values.iter().enumerate() {
            assert_eq!(
                bits.iter().fold(0_u64, |acc, share| acc ^ share.first()[i]),
                x as u64 & mask
            );
        }
        for _ in 1..program.instructions() {
            step(&mut owners);
        }
        let output = owners.map(|o| o.finish().unwrap().1);
        assert_eq!(
            clear(&output),
            values.map(|x| i64::from(x == 0)),
            "width={width}"
        );
    }
    let vocabulary = 151936_i64;
    for lanes in [1, 3, 17, 127, 128] {
        for token in [0, vocabulary - 1] {
            let values: Vec<_> = (0..lanes)
                .map(|i| {
                    let row = match i % 4 {
                        0 => 0,
                        1 => vocabulary - 1,
                        2 => token,
                        _ => (i as i64 * 104729) % vocabulary,
                    };
                    token - row
                })
                .collect();
            let program = Program::equal_zero_bounded(
                lanes,
                Interval::new(-vocabulary + 1, vocabulary - 1).unwrap(),
            )
            .unwrap();
            let mut owners = owners(&program, &values);
            for _ in 0..program.instructions() {
                step(&mut owners);
            }
            let output = owners.map(|o| o.finish().unwrap().1);
            assert_eq!(
                clear(&output),
                values
                    .iter()
                    .map(|&x| i64::from(x == 0))
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
// Directed role indices are part of this literal transcript/byte-count oracle.
#[allow(clippy::needless_range_loop)]
fn bounded_equality_costs_and_actual_directional_encoded_frames() {
    let bounds = Interval::new(-151935, 151935).unwrap();
    for width in [0, 18, 64] {
        let bounded = width != 0;
        let program = if bounded {
            Program::equal_zero_bounded(
                128,
                if width == 64 {
                    Interval::new(i64::MIN, i64::MAX).unwrap()
                } else {
                    bounds
                },
            )
        } else {
            Program::equal_zero(128, bounds)
        }
        .unwrap();
        let work = program.work_budget();
        assert_eq!(
            work.operations(),
            match width {
                18 => 23,
                64 => 27,
                _ => 28,
            }
        );
        assert_eq!(
            work.cursors(),
            crate::puma::prss::Cursors {
                public_blocks: 1,
                pair_blocks: match width {
                    18 => 648,
                    64 => 1224,
                    _ => 1472,
                },
                private_blocks: 0,
                aes_work: match width {
                    18 => 1169,
                    64 => 2321,
                    _ => 2817,
                },
            }
        );
        let values: Vec<_> = (0..128)
            .map(|i| [-151935, -1, 0, 1, 151935][i % 5])
            .collect();
        let mut owners = owners(&program, &values);
        let mut bytes = [[0_usize; 3]; 3];
        let mut conversion_bytes = [[0_usize; 3]; 3];
        let mut uncompressed_bytes = 0;
        let mut frames_count = 0;
        while !owners[0].complete() {
            let conversion = matches!(
                program.nodes[owners[0].step].unwrap().instruction,
                Instruction::Bits(_) | Instruction::LowBits(..)
            );
            let mut frames: Vec<_> = owners
                .iter_mut()
                .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
                .collect();
            while !frames.is_empty() {
                let mut next = Vec::new();
                for frame in frames {
                    let from = frame.sender();
                    let to = frame.recipient() as usize;
                    uncompressed_bytes += 48 + frame.inner.encode_bytes().unwrap().len();
                    let encoded = frame.encode().unwrap();
                    assert_eq!(
                        &encoded.as_bytes()[..4],
                        if bounded { b"PUT2" } else { b"PUT1" }
                    );
                    let n = encoded.as_bytes().len();
                    bytes[from as usize][to] += n;
                    if conversion {
                        conversion_bytes[from as usize][to] += n;
                    }
                    frames_count += 1;
                    next.extend(
                        owners[to]
                            .receive_encoded(encoded.as_bytes(), from)
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
        let output = owners.map(|o| o.finish().unwrap().1);
        assert_eq!(
            clear(&output),
            values
                .iter()
                .map(|&x| i64::from(x == 0))
                .collect::<Vec<_>>()
        );
        assert_eq!(frames_count, if width == 18 { 40 } else { 46 });
        assert_eq!(uncompressed_bytes, if width == 18 { 60000 } else { 69768 });
        assert_eq!(
            bytes.iter().flatten().sum::<usize>(),
            match width {
                18 => 19920,
                64 => 57384,
                _ => 69768,
            }
        );
        assert_eq!(
            bytes,
            match width {
                18 => [[0, 0, 6268], [7384, 0, 0], [1116, 5152, 0]],
                64 => [[0, 0, 18756], [19872, 0, 0], [1116, 17640, 0]],
                _ => [[0, 0, 22884], [24000, 0, 0], [1116, 21768, 0]],
            }
        );
        for from in 0..3 {
            assert_eq!(
                conversion_bytes[from][(from + 2) % 3],
                if width == 18 { 4100 } else { 15072 }
            );
        }
        eprintln!("width={width} (0=legacy): work={work:?}, encoded_directed={bytes:?}, a2b_directed={conversion_bytes:?}, uncompressed_total={uncompressed_bytes}");
    }
}

#[test]
fn bounded_equality_fresh_invocation_and_cancellation_preserve_owners() {
    let bounds = Interval::new(-151935, 151935).unwrap();
    let program = Program::equal_zero_bounded(3, bounds).unwrap();
    cancel_at_every_event(&program, &[-151935, 0, 151935]);
    let mut owners = owners_for_repeats(&program, &[-151935, 0, 151935], 2);
    for _ in 0..program.instructions() {
        step(&mut owners);
    }
    let output = owners.map(|o| o.finish().unwrap());
    let old = output.each_ref().map(|r| r.1.first().to_vec());
    let mut owners =
        output.map(|(runtime, input)| Executor::new(runtime, program.clone(), input).unwrap());
    for _ in 0..program.instructions() {
        step(&mut owners);
    }
    let output = owners.map(|o| o.finish().unwrap());
    assert_eq!(clear_refs(output.each_ref().map(|r| &r.1)), [1, 0, 1]);
    assert!(output.iter().zip(old).any(|(r, old)| r.1.first() != old));
    for (runtime, _) in &output {
        assert_eq!(
            runtime.cursors().pair_blocks,
            2 * program.work_budget().cursors().pair_blocks
        );
    }
    assert_eq!(
        Program::equal_zero(3, bounds).unwrap().numeric_profile(),
        super::super::NUMERIC_PROFILE
    );
    assert_eq!(program.numeric_profile(), BOUNDED_EQUALITY_PROFILE);
    let mut owners = self::owners(&program, &[-151935, 0, 151935]);
    let usages = owners.each_ref().map(Executor::payload_usage);
    let mut frames: Vec<_> = owners
        .iter_mut()
        .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
        .collect();
    let mut stale = frames.remove(0);
    stale.invocation += 1;
    let to = stale.recipient() as usize;
    assert!(owners[to].receive(stale).is_err());
    assert!(owners[to].retired());
    drop(frames);
    drop(owners);
    assert!(usages.iter().all(|u| u.live_bytes() == 0));
}

#[test]
fn bounded_equality_compact_layout_padding_roles_and_namespace_fail_closed() {
    let program = Program::equal_zero_bounded(3, Interval::new(-151935, 151935).unwrap()).unwrap();
    for variant in 0..18 {
        let mut owners = owners(&program, &[-151935, 0, 151935]);
        let usages = owners.each_ref().map(Executor::payload_usage);
        let mut frames: Vec<_> = owners
            .iter_mut()
            .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
            .collect();
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
            2 => raw[87] = 17,
            3 => raw[87] = 19,
            4 => raw[87] = 0,
            5 => raw[87] = 65,
            6 => raw[88..92].copy_from_slice(&4_u32.to_le_bytes()),
            7 => *raw.last_mut().unwrap() |= 0x40, // 54 used bits; two padding bits
            8 => raw[..4].copy_from_slice(b"PUT1"),
            9 => raw[48..52].copy_from_slice(b"PUM1"),
            10 => raw[4] ^= 1,
            11 => raw[36] ^= 1,
            12 => raw[44] ^= 1,
            13 => raw[52] ^= 1,
            14 => raw[84] = from.next() as u8,
            15 => raw[86] = MessagePhase::BooleanAnd as u8,
            16 => raw[85] = from as u8,
            _ => raw[88..92].copy_from_slice(&u32::MAX.to_le_bytes()),
        }
        assert!(
            owners[to].receive_encoded(&raw, from).is_err(),
            "variant={variant}"
        );
        assert!(owners[to].retired());
        assert!(matches!(
            owners[to].receive_encoded(encoded.as_bytes(), from),
            Err(Error::Retired)
        ));
        drop(encoded);
        drop(frames);
        drop(owners);
        assert!(usages.iter().all(|u| u.live_bytes() == 0));
    }
    // No encoder may silently discard live high bits under the compact layout.
    let mut owners = owners(&program, &[-151935, 0, 151935]);
    let mut frames: Vec<_> = owners
        .iter_mut()
        .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
        .collect();
    let mut frame = frames.remove(0);
    frame.inner.words[0] |= 1 << 18;
    assert!(matches!(frame.encode(), Err(Error::Bounds)));
}

#[test]
fn bounded_equality_compact_replay_cannot_cross_carry_stage() {
    let program = Program::equal_zero_bounded(3, Interval::new(-151935, 151935).unwrap()).unwrap();
    let mut owners = owners(&program, &[-151935, 0, 151935]);
    let usages = owners.each_ref().map(Executor::payload_usage);
    let mut frames: Vec<_> = owners
        .iter_mut()
        .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
        .collect();
    let frame = frames.remove(0);
    let from = frame.sender();
    let to = frame.recipient() as usize;
    let encoded = frame.encode().unwrap();
    let replies = owners[to]
        .receive_encoded(encoded.as_bytes(), from)
        .unwrap();
    assert!(owners[to]
        .receive_encoded(encoded.as_bytes(), from)
        .is_err());
    assert!(owners[to].retired());
    drop((encoded, replies, frames, owners));
    assert!(usages.iter().all(|u| u.live_bytes() == 0));
}

#[test]
fn checked_range_full_signed_domain_matches_independent_integer_recipe() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    let mut words = vec![i64::MIN, i64::MIN + 1, i64::MAX - 1, i64::MAX, -1, 0, 1];
    // Deterministic full-word coverage, independent of production predicates.
    let mut seed = 0xdef0123456789abc_u64;
    for _ in 0..32 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        words.push(seed as i64);
    }
    for (lower, upper, fallback) in [
        (-7, 11, 3),
        (1, 1, 1),
        (i64::MIN, -1, i64::MIN),
        (0, i64::MAX, i64::MAX),
        (i64::MIN, i64::MAX, 0),
        (i64::MIN, i64::MIN, i64::MIN),
        (i64::MAX, i64::MAX, i64::MAX),
    ] {
        let accepted = Interval::new(lower, upper).unwrap();
        let mut values = words.clone();
        values.extend([lower, upper]);
        if let Some(x) = lower.checked_sub(1) {
            values.push(x);
        }
        if let Some(x) = upper.checked_add(1) {
            values.push(x);
        }
        let program = Program::checked_range(values.len(), full, accepted, fallback).unwrap();
        assert_eq!(program.numeric_profile(), CHECKED_RANGE_PROFILE);
        assert_eq!(program.output_bounds(), accepted);
        assert!(program.admitted_bytes() <= 256 * 1024);
        let mut parties = owners(&program, &values);
        for _ in 0..program.instructions() {
            step(&mut parties);
        }
        let results = parties.map(|p| p.finish_checked().unwrap());
        let safe = clear_refs(results.each_ref().map(|r| &r.1));
        let valid = clear_refs(results.each_ref().map(|r| &r.2));
        for ((&x, &safe), &valid) in values.iter().zip(&safe).zip(&valid) {
            let ok = i128::from(x) >= i128::from(lower) && i128::from(x) <= i128::from(upper);
            assert_eq!(valid, i64::from(ok), "x={x}, [{lower},{upper}]");
            assert_eq!(safe, if ok { x } else { fallback });
        }
        let usages = results.each_ref().map(|r| r.0.payload_usage());
        for (mut runtime, safe, valid) in results {
            let usage = runtime.payload_usage();
            assert!(usage.peak_bytes() <= usage.limit_bytes());
            runtime.cancel();
            drop((runtime, safe, valid));
        }
        assert!(usages.iter().all(|u| u.live_bytes() == 0));
    }
}

#[test]
fn checked_range_precedes_unsafe_square_and_preserves_sticky_failure() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    let mut b = Builder::new(7, full).unwrap();
    assert!(b.square(b.input()).is_err());
    let checked = b
        .checked_range(b.input(), Interval::new(-ONE, ONE).unwrap(), ONE)
        .unwrap();
    let square = b.square(checked.safe).unwrap();
    let second = b
        .checked_range(square, Interval::new(0, ONE).unwrap(), 0)
        .unwrap();
    let sticky = b.product(checked.valid, second.valid, false).unwrap();
    let output = b
        .affine(
            second.safe,
            None,
            Affine {
                left: 1,
                right: 0,
                constant: 0,
            },
        )
        .unwrap();
    let program = b.finish_checked(output, sticky).unwrap();
    let values = [i64::MIN, -ONE - 1, -ONE, 0, ONE, ONE + 1, i64::MAX];
    let mut parties = owners(&program, &values);
    for _ in 0..program.instructions() {
        step(&mut parties);
    }
    let results = parties.map(|p| p.finish_checked().unwrap());
    assert_eq!(
        clear_refs(results.each_ref().map(|r| &r.1)),
        [ONE, ONE, ONE, 0, ONE, ONE, ONE]
    );
    assert_eq!(
        clear_refs(results.each_ref().map(|r| &r.2)),
        [0, 0, 1, 1, 1, 0, 0]
    );
    assert!(Program::checked_range(1, full, Interval::new(1, 2).unwrap(), 0).is_err());
    assert_eq!(
        Program::equal_zero(1, full).unwrap().numeric_profile(),
        super::super::NUMERIC_PROFILE
    );
}

#[test]
fn checked_range_lifecycle_and_fresh_repetition() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    let program = Program::checked_range(3, full, Interval::new(1, 9).unwrap(), 1).unwrap();
    cancel_at_every_event(&program, &[i64::MIN, 4, i64::MAX]);
    let mut parties = owners_for_repeats(&program, &[i64::MIN, 4, i64::MAX], 2);
    for _ in 0..program.instructions() {
        step(&mut parties);
    }
    let mut next = Vec::new();
    let mut first_words = Vec::new();
    for party in parties {
        let (runtime, safe, valid) = party.finish_checked().unwrap();
        first_words.push(safe.first().to_vec());
        drop(valid);
        next.push(Executor::new(runtime, program.clone(), safe).unwrap());
    }
    let mut parties: [Executor; 3] = next.try_into().ok().unwrap();
    for _ in 0..program.instructions() {
        step(&mut parties);
    }
    let results = parties.map(|p| p.finish_checked().unwrap());
    assert_eq!(clear_refs(results.each_ref().map(|r| &r.1)), [1, 4, 1]);
    assert_eq!(clear_refs(results.each_ref().map(|r| &r.2)), [1, 1, 1]);
    assert!(results
        .iter()
        .zip(first_words)
        .any(|(r, old)| r.1.first() != old));
    // Checked owners must not silently discard the acceptance obligation.
    let mut parties = owners(&program, &[1, 4, 9]);
    for _ in 0..program.instructions() {
        step(&mut parties);
    }
    for party in parties {
        assert!(matches!(party.finish(), Err(Error::Binding)));
    }
}

#[test]
fn checked_range_frozen_binding_replay_and_short_budget_fail_closed() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    let range = Interval::new(-9, 9).unwrap();
    let program = Program::checked_range(3, full, range, 0).unwrap();
    for other in [
        Program::checked_range(3, full, range, 1).unwrap(),
        Program::checked_range(3, full, Interval::new(-9, 10).unwrap(), 0).unwrap(),
        Program::checked_range(3, Interval::new(-99, 99).unwrap(), range, 0).unwrap(),
    ] {
        assert_ne!(program.digest(), other.digest());
    }

    for variant in 0..3 {
        let mut owners = owners(&program, &[i64::MIN, 0, i64::MAX]);
        let usage = owners.each_ref().map(Executor::payload_usage);
        let mut frames: Vec<_> = owners
            .iter_mut()
            .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
            .collect();
        let mut frame = frames.remove(0);
        let to = frame.recipient() as usize;
        match variant {
            0 => frame.program[0] ^= 1,
            1 => frame.invocation += 1,
            _ => frame.step += 1,
        }
        assert!(owners[to].receive(frame).is_err());
        assert!(owners[to].retired());
        drop(frames);
        drop(owners);
        assert!(usage.iter().all(|u| u.live_bytes() == 0));
    }
    let work = program.work_budget();
    for limits in [
        PrssLimits::program(work.operations() - 1, work.cursors().aes_work).unwrap(),
        PrssLimits::program(work.operations(), work.cursors().aes_work - 1).unwrap(),
    ] {
        let runtimes = parties(limits);
        let inputs = inputs(&runtimes, &[i64::MIN, 0, i64::MAX]);
        for (runtime, input) in runtimes.into_iter().zip(inputs) {
            let usage = runtime.payload_usage();
            assert!(Executor::new(runtime, program.clone(), input).is_err());
            assert_eq!(usage.live_bytes(), 0);
        }
    }
}

#[test]
fn checked_range_aggregate_validity_and_failure_schedule_are_private() {
    let program = Program::all_valid(3).unwrap();
    assert_eq!(program.output_lanes(), 1);
    for values in [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]] {
        let mut owners = owners(&program, &values);
        for _ in 0..program.instructions() {
            step(&mut owners);
        }
        let results = owners.map(|o| o.finish().unwrap().1);
        assert_eq!(clear(&results), [i64::from(values.iter().all(|&v| v == 1))]);
    }
    // Even values violating incoming metadata are safe to check. That metadata
    // is not consulted by the full-word Boolean predicate/selection circuit.
    let guard = Program::checked_range(
        3,
        Interval::new(-4, 4).unwrap(),
        Interval::new(-2, 2).unwrap(),
        1,
    )
    .unwrap();
    let mut schedules = Vec::new();
    for values in [[-2, 0, 2], [i64::MIN, 0, i64::MAX]] {
        let mut owners = owners(&guard, &values);
        let mut frames = 0;
        for _ in 0..guard.instructions() {
            frames += step(&mut owners).1;
        }
        schedules.push((frames, owners.each_ref().map(Executor::cursors)));
        let results = owners.map(|o| o.finish_checked().unwrap());
        assert_eq!(
            clear_refs(results.each_ref().map(|r| &r.1)),
            if values[0] == -2 {
                [-2, 0, 2]
            } else {
                [1, 0, 1]
            }
        );
    }
    assert_eq!(schedules[0], schedules[1]);
}

#[test]
fn tight_work_budgets_cover_boolean_backing_widths_and_fill_tails() {
    for lanes in [1, 2, 3, 15, 16, 17, 127, 128] {
        let values: Vec<_> = (0..lanes).map(|i| i as i64 - 7).collect();
        let range = Interval::new(-128, 128).unwrap();
        for bits in [0, 1, 7, 8, 9, 16, 17, 32, 33, 62] {
            let mut b = Builder::new(lanes, range).unwrap();
            let input = if bits == 0 {
                b.sign(b.input())
            } else {
                b.bits(b.input())
            }
            .unwrap();
            let masked = b
                .bit_map(
                    input,
                    if bits == 0 {
                        BitMap::ShiftRight(1)
                    } else {
                        BitMap::LowBits(bits)
                    },
                )
                .unwrap();
            let output = b.from_boolean(masked).unwrap();
            let program = b.finish(output).unwrap();
            let mut parties = owners(&program, &values);
            for _ in 0..program.instructions() {
                step(&mut parties);
            }
            let output = parties.map(|p| p.finish().unwrap().1);
            let mask = (1u64 << bits) - 1;
            assert_eq!(
                clear(&output),
                values
                    .iter()
                    .map(|&v| (v as u64 & mask) as i64)
                    .collect::<Vec<_>>()
            );
        }
        let equality = Program::equal_zero(lanes, range).unwrap();
        let mut parties = owners(&equality, &values);
        for _ in 0..equality.instructions() {
            step(&mut parties);
        }
        let result = parties.map(|p| p.finish().unwrap().1);
        assert_eq!(
            clear(&result),
            values
                .iter()
                .map(|&v| i64::from(v == 0))
                .collect::<Vec<_>>()
        );
        if lanes == 3 || lanes == 128 {
            let bound = equality.work_budget();
            assert_eq!(bound.operations(), 28);
            assert_eq!(
                bound.cursors(),
                if lanes == 3 {
                    crate::puma::prss::Cursors {
                        public_blocks: 1,
                        pair_blocks: 40,
                        private_blocks: 0,
                        aes_work: 77,
                    }
                } else {
                    crate::puma::prss::Cursors {
                        public_blocks: 1,
                        pair_blocks: 1472,
                        private_blocks: 0,
                        aes_work: 2817,
                    }
                }
            );
        }
    }
}

#[test]
fn exhausted_exact_schedule_and_short_aes_allowances_retire_before_start() {
    let program = Program::equal_zero(3, Interval::new(-1, 1).unwrap()).unwrap();
    let bound = program.work_budget();
    for limits in [
        PrssLimits::program(bound.operations() - 1, bound.cursors().aes_work).unwrap(),
        PrssLimits::program(bound.operations(), bound.cursors().aes_work - 1).unwrap(),
    ] {
        let runtimes = parties(limits);
        let inputs = inputs(&runtimes, &[-1, 0, 1]);
        for (runtime, input) in runtimes.into_iter().zip(inputs) {
            let usage = runtime.payload_usage();
            assert!(Executor::new(runtime, program.clone(), input).is_err());
            assert_eq!(usage.live_bytes(), 0);
        }
    }
    let mut parties = owners(&program, &[-1, 0, 1]);
    for _ in 0..program.instructions() {
        step(&mut parties);
    }
    for party in parties {
        let (runtime, output) = party.finish().unwrap();
        let usage = runtime.payload_usage();
        assert!(Executor::new(runtime, program.clone(), output).is_err());
        assert_eq!(usage.live_bytes(), 0);
    }
}

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

fn softmax_oracle(x: &[i64], width: usize, tapes: &[Vec<u64>]) -> Vec<i64> {
    // Independent D6 equations. Clipped terms are identically zero, regardless
    // of their discarded powers. Only retained terms need exponent arithmetic.
    assert_eq!(tapes.len(), 13);
    let mut output = Vec::new();
    for (row, values) in x.chunks_exact(width).enumerate() {
        let base = row * width;
        let maximum = *values.iter().max().unwrap();
        let exponentials: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let z = x - maximum;
                if z <= -14 * ONE {
                    return 0;
                }
                let mut v = ONE + rounded(i128::from(z), tapes[0][base + i], 5);
                for tape in &tapes[1..6] {
                    v = rounded(i128::from(v).pow(2), tape[base + i], 18);
                }
                v
            })
            .collect();
        let total = exponentials.iter().sum();
        let inverse = normalization_oracle(false, total, tapes[6..12].iter().map(|t| t[row]));
        output.extend(
            exponentials.iter().enumerate().map(|(i, &v)| {
                rounded(i128::from(v) * i128::from(inverse), tapes[12][base + i], 18)
            }),
        );
    }
    output
}

#[test]
fn guarded_softmax_preserves_strict_cutoff_and_stochastic_source_equations() {
    let bounds = Interval::new(-1_500_000_000_000, 1_500_000_000_000).unwrap();
    assert!(Program::softmax(1, 3, bounds).is_err());
    for (rows, width) in [(1, 1), (3, 3), (2, 7), (1, 128), (1, 512)] {
        let program = Program::softmax_guarded(rows, width, bounds).unwrap();
        let guard = program.nodes[..program.len]
            .iter()
            .find_map(|n| match n.unwrap().instruction {
                Instruction::Rescale { input, shift: 5 } => Some(input),
                _ => None,
            })
            .unwrap();
        let Instruction::Product { right: keep, .. } =
            program.nodes[guard.0 as usize].unwrap().instruction
        else {
            panic!("shift input must be guarded");
        };
        for values in [
            vec![bounds.lower(); rows * width],
            (0..rows * width)
                .map(|i| {
                    let maximum = [bounds.upper(), 0, bounds.lower() + 15 * ONE][i / width % 3];
                    maximum
                        + [
                            0,
                            -14 * ONE - 1,
                            -14 * ONE,
                            -14 * ONE + 1,
                            -ONE - 3,
                            -1,
                            -15 * ONE,
                        ][i % width % 7]
                })
                .collect(),
            (0..rows * width)
                .map(|i| {
                    if i % width == width - 1 {
                        bounds.upper()
                    } else {
                        bounds.lower()
                    }
                })
                .collect(),
        ] {
            let mut owners = owners(&program, &values);
            let usage = owners.each_ref().map(Executor::payload_usage);
            let mut tapes = vec![];
            for index in 1..=program.instructions() {
                if index == guard.0 as usize {
                    // Observe only in this all-party test, before the exact
                    // guard product. Boundary outputs can round to zero even
                    // on the retained side, so check the secret predicate too.
                    let shares = owners.each_ref().map(|o| {
                        let Some(Register::Arithmetic(value)) =
                            &o.registers[program.slots[keep.0 as usize] as usize]
                        else {
                            panic!("keep must remain live");
                        };
                        value
                    });
                    let expected: Vec<_> = values
                        .chunks_exact(width)
                        .flat_map(|row| {
                            let maximum = *row.iter().max().unwrap();
                            row.iter().map(move |&x| i64::from(x - maximum > -14 * ONE))
                        })
                        .collect();
                    assert_eq!(clear_refs(shares), expected);
                }
                if let Some(tape) = step(&mut owners).0 {
                    tapes.push(tape);
                }
            }
            let output = owners.map(|o| o.finish().unwrap().1);
            assert_eq!(clear(&output), softmax_oracle(&values, width, &tapes));
            assert!(usage
                .iter()
                .all(|u| u.peak_bytes() + FIXED_RESERVE_BYTES <= program.admitted_bytes()));
            drop(output);
            assert!(usage.iter().all(|u| u.live_bytes() == 0));
        }
    }
}

#[test]
fn guarded_softmax_binds_added_work_and_retains_range_and_lifetime_gates() {
    let range = Interval::new(-14 * ONE, 0).unwrap();
    let eager = Program::softmax(1, 3, range).unwrap();
    let guarded = Program::softmax_guarded(1, 3, range).unwrap();
    assert_ne!(eager.digest(), guarded.digest());
    assert_eq!(guarded.instructions(), eager.instructions() + 1);
    assert_eq!(
        guarded.work_budget().operations(),
        eager.work_budget().operations() + 1
    );
    assert!(guarded.work_budget().cursors().aes_work > eager.work_budget().cursors().aes_work);
    // Each schedule matches the same independent equations, with its own fresh
    // random tape. Moving operations does not promise identical PRSS cursors.
    let values = [-14 * ONE, -14 * ONE + 1, 0];
    for program in [&eager, &guarded] {
        let mut owners = owners(program, &values);
        let mut tapes = vec![];
        for _ in 0..program.instructions() {
            if let Some(tape) = step(&mut owners).0 {
                tapes.push(tape);
            }
        }
        assert_eq!(
            clear(&owners.map(|o| o.finish().unwrap().1)),
            softmax_oracle(&values, 3, &tapes)
        );
    }
    for (rows, width) in [(0, 1), (1, 0), (usize::MAX, 2), (1, MAX_ROW_LANES + 1)] {
        assert!(Program::softmax_guarded(rows, width, range).is_err());
    }
    assert!(Program::softmax_guarded(1, 3, Interval::new(i64::MIN, i64::MAX).unwrap()).is_err());
    let broad =
        Program::softmax_guarded(1, 3, Interval::new(-1_500_000_000_000, 0).unwrap()).unwrap();
    cancel_at_every_event(&broad, &[-1_500_000_000_000, -14 * ONE + 1, 0]);
}

#[test]
fn certified_softmax_preserves_recipe_and_bounds_complete_row_mass() {
    let bounds = Interval::new(-1_500_000_000_000, 1_500_000_000_000).unwrap();
    for (rows, width) in [(1, 1), (3, 3), (2, 7), (1, 128), (1, 512)] {
        let (program, mass) = Program::softmax_certified(rows, width, bounds).unwrap();
        let old = Program::softmax_guarded(rows, width, bounds).unwrap();
        assert_eq!(program.instructions(), old.instructions());
        assert_eq!(program.work_budget(), old.work_budget());
        assert_eq!(program.admitted_bytes(), old.admitted_bytes());
        assert!(mass < (ONE * 103 / 100 + width as i64) as u64);
        for values in [
            vec![bounds.lower(); rows * width],
            (0..rows * width)
                .map(|i| {
                    [
                        0,
                        -14 * ONE - 1,
                        -14 * ONE,
                        -14 * ONE + 1,
                        -ONE - 3,
                        -1,
                        bounds.lower(),
                    ][i % width % 7]
                })
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
            assert_eq!(actual, softmax_oracle(&values, width, &tapes));
            assert!(actual
                .iter()
                .all(|&v| (0..=program.output_bounds().upper()).contains(&v)));
            for row in actual.chunks_exact(width) {
                assert!(row.iter().map(|&v| v as u64).sum::<u64>() <= mass);
            }
            assert!(usage
                .iter()
                .all(|u| u.peak_bytes() + FIXED_RESERVE_BYTES <= program.admitted_bytes()));
            drop(output);
            assert!(usage.iter().all(|u| u.live_bytes() == 0));
        }
    }
    let (program, _) = Program::softmax_certified(1, 3, bounds).unwrap();
    cancel_at_every_event(&program, &[bounds.lower(), -14 * ONE + 1, 0]);
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
            assert!(actual.iter().all(|&x| {
                program.output_bounds().lower() <= x && x <= program.output_bounds().upper()
            }));
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
fn rms_norm_correlated_denominator_bounds_cover_wide_and_near_zero_rows() {
    for width in [3, 7] {
        let bounds = Interval::new(-256 * ONE, 256 * ONE).unwrap();
        let gamma = vec![ONE; width];
        let program = Program::rms_norm(3, width, bounds, &gamma).unwrap();
        // The public proof uses x_i^2 <= sum(x^2), even with stochastic
        // square/mean rounding. It must not multiply the largest input by the
        // zero-row inverse bound; that hull would exceed 100,000 real units.
        assert!(program.output_bounds().lower() > -128 * ONE);
        assert!(program.output_bounds().upper() < 128 * ONE);
        let mut values = vec![0; 3 * width];
        values[0] = -256 * ONE;
        values[width] = 1;
        values[width + 1] = -1;
        values[2 * width..].fill(256 * ONE);
        let mut owners = owners(&program, &values);
        let mut tapes = vec![];
        for _ in 0..program.instructions() {
            if let Some(tape) = step(&mut owners).0 {
                tapes.push(tape);
            }
        }
        let actual = clear(&owners.map(|o| o.finish().unwrap().1));
        assert_eq!(actual, rms_norm_oracle(&values, width, &gamma, &tapes));
        assert!(actual.iter().all(|&x| {
            program.output_bounds().lower() <= x && x <= program.output_bounds().upper()
        }));
    }
}

#[test]
fn certified_input_energy_preserves_sparse_rms_norm_rounding_and_ownership() {
    let bound = Interval::new(-512 * ONE, 512 * ONE).unwrap();
    let energy = (512 * ONE) as u128 * (512 * ONE) as u128;
    for width in [3, 128, 896] {
        let gamma: Vec<_> = (0..width).map(|i| [ONE, -ONE, 0][i % 3]).collect();
        assert!(Program::rms_norm_certified(1, width, bound, &gamma).is_err());
        let (program, squared) =
            Program::rms_norm_with_input_norm(1, width, bound, &gamma, energy).unwrap();
        let narrow =
            Program::rms_norm_certified(1, width, Interval::new(-ONE, ONE).unwrap(), &gamma)
                .unwrap()
                .0;
        assert_eq!(program.work_budget(), narrow.work_budget());
        assert_eq!(program.instructions(), narrow.instructions());
        assert_eq!(program.admitted_bytes(), narrow.admitted_bytes());
        for sign in [-1, 0, 1] {
            let mut values = vec![0; width];
            values[width / 2] = sign * 512 * ONE;
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
            assert!(
                actual
                    .iter()
                    .map(|v| i128::from(*v).unsigned_abs().pow(2))
                    .sum::<u128>()
                    <= squared
            );
            drop(output);
            assert!(usage.iter().all(|u| u.live_bytes() == 0));
        }
    }
    let program = Program::rms_norm_with_input_norm(1, 3, bound, &[ONE; 3], energy)
        .unwrap()
        .0;
    cancel_at_every_event(&program, &[512 * ONE, 0, 0]);
    assert!(Program::rms_norm_with_input_norm(1, 3, bound, &[ONE; 3], 3 * energy).is_err());
    assert!(Program::rms_norm_with_input_norm(
        1,
        3,
        Interval::new(ONE, ONE).unwrap(),
        &[ONE; 3],
        0
    )
    .is_err());
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

#[test]
fn partitioned_rms_norm_preserves_stochastic_recipe_work_and_payload() {
    for (rows, width) in [(1, 1), (1, 3), (1, 128), (1, 896), (3, 7)] {
        let lanes = rows * width;
        let bounds = Interval::new(-256 * ONE, 256 * ONE).unwrap();
        let gamma: Vec<_> = (0..width).map(|i| [ONE, -2 * ONE, 0][i % 3]).collect();
        let legacy = Program::rms_norm(rows, width, bounds, &gamma).unwrap();
        let (program, squared_norm) =
            Program::rms_norm_certified(rows, width, bounds, &gamma).unwrap();
        assert_eq!(program.work_budget(), legacy.work_budget());
        assert_eq!(program.admitted_bytes(), legacy.admitted_bytes());
        assert_eq!(program.instructions(), legacy.instructions());
        assert_eq!(program.numeric_profile(), legacy.numeric_profile());
        assert_eq!(program.len, legacy.len);
        for (a, b) in program.nodes[..program.len]
            .iter()
            .zip(&legacy.nodes[..legacy.len])
        {
            let a = a.unwrap();
            let b = b.unwrap();
            assert_eq!((a.lanes, a.bit), (b.lanes, b.bit));
            match (a.instruction, b.instruction) {
                (
                    Instruction::Product {
                        left: a_left,
                        right: a_right,
                        rescale: a_rescale,
                        certified: a_certified,
                    },
                    Instruction::Product {
                        left: b_left,
                        right: b_right,
                        rescale: b_rescale,
                        certified: b_certified,
                    },
                ) => {
                    assert_eq!((a_left, a_right, a_rescale), (b_left, b_right, b_rescale));
                    assert_eq!(a_certified.is_some(), b_certified.is_some());
                }
                (a, b) => assert_eq!(a, b),
            }
        }
        assert_ne!(program.digest(), legacy.digest());
        // Partitioned and ordinary legacy enclosures have independent contracts;
        // do not require relative containment or strict ordering. These
        // fixtures have reciprocal-width coefficient >= 292 and |gamma| <= 2Q,
        // so check an absolute bound, not universal relative tightening.
        assert!(program.output_bounds().lower() > -128 * ONE);
        assert!(program.output_bounds().upper() < 128 * ONE);
        if width >= 128 {
            let scalar = program
                .output_bounds()
                .lower()
                .unsigned_abs()
                .max(program.output_bounds().upper().unsigned_abs());
            assert!(squared_norm < u128::from(scalar).pow(2) * width as u128 / 16);
        }
        for values in [
            vec![0; lanes],
            vec![256 * ONE; lanes],
            vec![1; lanes],
            (0..lanes)
                .map(|i| if i == width / 2 { 256 * ONE } else { 0 })
                .collect(),
            (0..lanes).map(|i| [-256 * ONE, 1, -1, 0][i % 4]).collect(),
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
            assert!(actual.chunks_exact(width).all(|row| row
                .iter()
                .map(|v| i128::from(*v).unsigned_abs().pow(2))
                .sum::<u128>()
                <= squared_norm));
            assert!(actual.iter().all(|v| (program.output_bounds().lower()
                ..=program.output_bounds().upper())
                .contains(v)));
            assert!(usage
                .iter()
                .all(|u| u.peak_bytes() + FIXED_RESERVE_BYTES <= program.admitted_bytes()));
            drop(output);
            assert!(usage.iter().all(|u| u.live_bytes() == 0));
        }
    }
}

#[test]
fn partitioned_rms_norm_zero_input_bounds_match_legacy_with_signed_gamma() {
    let bounds = Interval::new(0, 0).unwrap();
    let gamma = [ONE, -ONE];
    let legacy = Program::rms_norm(1, 2, bounds, &gamma).unwrap();
    let (program, _) = Program::rms_norm_certified(1, 2, bounds, &gamma).unwrap();
    assert_eq!(legacy.output_bounds(), bounds);
    assert_eq!(program.output_bounds(), bounds);
    assert_eq!(program.digest(), legacy.digest());
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
    owners_for_repeats(program, values, 1)
}

fn owners_for_repeats(program: &Program, values: &[i64], repeats: u64) -> [Executor; 3] {
    let runtimes = parties(
        program
            .work_budget()
            .checked_mul(repeats)
            .unwrap()
            .limits()
            .unwrap(),
    );
    let mut values = inputs(&runtimes, values).map(Some);
    let mut i = 0;
    runtimes.map(|r| {
        let out = Executor::new(r, program.clone(), values[i].take().unwrap()).unwrap();
        i += 1;
        out
    })
}

fn clear(values: &[ReplicatedShare; 3]) -> Vec<i64> {
    clear_refs(values.each_ref())
}

fn clear_refs(values: [&ReplicatedShare; 3]) -> Vec<i64> {
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
    let before = owners.each_ref().map(|o| {
        (
            o.runtime.as_ref().unwrap().source.runtime.next_sequence(),
            o.cursors(),
        )
    });
    let budget = owners[0]
        .program
        .instruction_work(owners[0].program.nodes[owners[0].step].unwrap())
        .unwrap();
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
    let mut max_aes = 0;
    let mut max_private = 0;
    for (o, (sequence, start)) in owners.iter_mut().zip(before) {
        o.finish_step().unwrap();
        let end = o.cursors();
        assert_eq!(
            o.runtime.as_ref().unwrap().source.runtime.next_sequence() - sequence,
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
        max_aes = max_aes.max(end.aes_work - start.aes_work);
        max_private = max_private.max(end.private_blocks - start.private_blocks);
        if o.complete() {
            assert_eq!(
                o.runtime.as_ref().unwrap().source.runtime.next_sequence() - o.invocation,
                o.program.work_budget().operations()
            );
        }
    }
    assert_eq!(max_aes, budget.cursors().aes_work);
    assert_eq!(max_private, budget.cursors().private_blocks);
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
fn guarded_activations_preserve_source_equations_cutoffs_and_work() {
    for kind in [Activation::Gelu, Activation::Silu] {
        let pivots = match kind {
            Activation::Gelu => [-1048576, -511180, 786432],
            Activation::Silu => [-2097152, -1048576, 1048576],
        };
        for n in [1, 17, MAX_TILE_LANES] {
            let mut points = vec![-256 * ONE, -109 * ONE, -1, 0, 1, 109 * ONE, 256 * ONE];
            for pivot in pivots {
                points.extend([pivot - 1, pivot, pivot + 1]);
            }
            let x: Vec<_> = (0..n).map(|i| points[i % points.len()]).collect();
            let program =
                Program::activation_guarded(kind, n, Interval::new(-256 * ONE, 256 * ONE).unwrap())
                    .unwrap();
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
            assert_eq!(count, if kind == Activation::Gelu { 174 } else { 159 });
            let output = owners.map(|o| {
                let (mut runtime, output) = o.finish().unwrap();
                runtime.cancel();
                output
            });
            let actual = clear(&output);
            for (i, &x) in x.iter().enumerate() {
                let expected = if x < pivots[0] {
                    0
                } else if x > pivots[2] {
                    x
                } else {
                    oracle(kind, x, tapes.iter().map(|t| t[i]))
                };
                assert_eq!(actual[i], expected, "{kind:?}, {x}");
                assert!(
                    (program.output_bounds().lower()..=program.output_bounds().upper())
                        .contains(&actual[i])
                );
            }
            assert!(usage
                .iter()
                .all(|u| u.peak_bytes() + FIXED_RESERVE_BYTES <= program.admitted_bytes()));
            drop(output);
            assert!(usage.iter().all(|u| u.live_bytes() == 0));
        }
        let range = Interval::new(-9 * ONE, 9 * ONE).unwrap();
        let eager = Program::activation(kind, 3, range).unwrap();
        let guarded = Program::activation_guarded(kind, 3, range).unwrap();
        assert_ne!(eager.digest(), guarded.digest());
        assert_eq!(guarded.instructions(), eager.instructions() + 2);
        assert_eq!(
            guarded.work_budget(),
            eager
                .work_budget()
                .checked_add(WorkBudget::local())
                .unwrap()
                .checked_add(WorkBudget::product(3, false).unwrap())
                .unwrap()
        );
        cancel_at_every_event(&guarded, &[-9 * ONE, 0, 9 * ONE]);
        assert!(Program::activation_guarded(kind, 129, range).is_err());
        assert!(
            Program::activation_guarded(kind, 1, Interval::new(i64::MIN, i64::MAX).unwrap())
                .is_err()
        );
    }
}

#[test]
fn guarded_activation_range_proof_covers_independent_rounding_choices() {
    for kind in [Activation::Silu, Activation::Gelu] {
        let (lo, pivot, hi, truncations) = match kind {
            Activation::Silu => (-2097152, -1048576, 1048576, 9),
            Activation::Gelu => (-1048576, -511180, 786432, 11),
        };
        // Different, overlapping public cells from the proof implementation;
        // enumerate every floor/ceil combination in the independent recipe.
        for start in (lo..=hi).step_by(7919).chain([lo, pivot - 1, pivot, hi]) {
            let end = hi.min(start + 8003);
            let bound =
                numeric::activation_bounds(kind, Interval::new(start, end).unwrap()).unwrap();
            for x in [start, start + (end - start) / 2, end] {
                for bits in 0..1_u64 << truncations {
                    let masks = (0..truncations).map(|i| {
                        if (bits >> i) & 1 == 0 {
                            0
                        } else {
                            (ONE - 1) as u64
                        }
                    });
                    let output = oracle(kind, x, masks);
                    assert!(
                        (bound.lower()..=bound.upper()).contains(&output),
                        "{kind:?}, {x}, {bits}, {output}, {bound:?}"
                    );
                }
            }
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
    let mut owners = owners_for_repeats(&program, &[-9 * ONE, 0, 9 * ONE], 2);
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
                compact_bits: frame.compact_bits,
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
