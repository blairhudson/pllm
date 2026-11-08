use super::*;

fn query(
    kind: Activation,
    guarded: bool,
    lanes: usize,
    input: Interval,
    workspace: &mut ActivationWorkWorkspace,
) -> Result<WorkBudget, Error> {
    if guarded {
        Program::activation_guarded_work(kind, lanes, input, workspace)
    } else {
        Program::activation_work(kind, lanes, input, workspace)
    }
}
fn construct(
    kind: Activation,
    guarded: bool,
    lanes: usize,
    input: Interval,
) -> Result<Program, Error> {
    if guarded {
        Program::activation_guarded(kind, lanes, input)
    } else {
        Program::activation(kind, lanes, input)
    }
}

#[test]
fn activation_work_matches_actual_native_metadata_every_legacy_kind_and_width() {
    let mut workspace = ActivationWorkWorkspace::empty();
    for kind in [Activation::Silu, Activation::Gelu] {
        for guarded in [false, true] {
            for lanes in [1, 2, 3, 7, 17, 31, 63, 64, 65, 127, 128] {
                for (lo, hi) in [
                    (-ONE, ONE),
                    (0, 0),
                    (-ONE, -ONE),
                    (ONE, ONE),
                    (-10 * ONE, -9 * ONE),
                    (9 * ONE, 10 * ONE),
                ] {
                    let input = Interval::new(lo, hi).unwrap();
                    let before = activation_work::entries();
                    let work = query(kind, guarded, lanes, input, &mut workspace).unwrap();
                    assert_eq!(activation_work::entries(), before);
                    assert!(!workspace.is_poisoned());
                    let native = construct(kind, guarded, lanes, input).unwrap();
                    assert_eq!(
                        work,
                        native.work_budget(),
                        "{kind:?}/{guarded}/{lanes}/{lo}/{hi}"
                    );
                    workspace.assert_native_metadata(&native);
                }
            }
        }
    }
}

#[test]
fn activation_work_original_error_winners_poison_and_reset_without_factory_entry() {
    let good = Interval::new(-ONE, ONE).unwrap();
    let extreme = Interval::new(i64::MIN, i64::MAX).unwrap();
    let mut workspace = ActivationWorkWorkspace::empty();
    for kind in [Activation::Gelu, Activation::Silu] {
        for guarded in [false, true] {
            for (lanes, input) in [
                (0, good),
                (129, good),
                (usize::MAX, extreme),
                (1, extreme),
                (1, Interval::new(i64::MIN, i64::MIN).unwrap()),
                (1, Interval::new(i64::MAX, i64::MAX).unwrap()),
            ] {
                workspace.reset();
                let before = activation_work::entries();
                let quoted = query(kind, guarded, lanes, input, &mut workspace);
                assert_eq!(activation_work::entries(), before);
                let native = construct(kind, guarded, lanes, input);
                match (quoted, native) {
                    (Err(a), Err(b)) => {
                        assert_eq!(a, b);
                        assert!(workspace.is_poisoned());
                        let before = activation_work::entries();
                        assert_eq!(
                            query(kind, guarded, 1, good, &mut workspace),
                            Err(Error::Phase)
                        );
                        assert_eq!(activation_work::entries(), before);
                        workspace.reset();
                        assert!(query(kind, guarded, 1, good, &mut workspace).is_ok());
                    }
                    (Ok(work), Ok(native)) => assert_eq!(work, native.work_budget()),
                    _ => panic!("native/query winner mismatch {kind:?}/{guarded}/{lanes}"),
                }
            }
        }
    }
}

#[test]
fn activation_workspace_is_quoted_before_creation_without_program_or_request_authority() {
    fn workspace(limit: usize, entries: &mut usize) -> Result<ActivationWorkWorkspace, Error> {
        if ActivationWorkWorkspace::workspace_bytes() > limit {
            return Err(Error::Bounds);
        }
        *entries += 1;
        Ok(ActivationWorkWorkspace::empty())
    }
    let bytes = ActivationWorkWorkspace::workspace_bytes();
    assert_eq!(bytes, std::mem::size_of::<ActivationWorkWorkspace>());
    assert!(bytes > 0);
    let before = activation_work::entries();
    let mut creations = 0;
    assert!(matches!(
        workspace(bytes - 1, &mut creations),
        Err(Error::Bounds)
    ));
    assert_eq!(creations, 0);
    let mut space = workspace(bytes, &mut creations).unwrap();
    assert_eq!(creations, 1);
    let work = Program::activation_guarded_work(
        Activation::Silu,
        3,
        Interval::new(-ONE, ONE).unwrap(),
        &mut space,
    )
    .unwrap();
    assert!(work.operations() > 0);
    assert_eq!(activation_work::entries(), before);
}

fn schedule(space: &mut ActivationWorkWorkspace) -> Result<WorkBudget, Error> {
    let bounds = Interval::new(-ONE, ONE)?;
    Program::activation_guarded_work(Activation::Silu, 128, bounds, space)?
        .checked_mul(2)?
        .checked_add(Program::activation_guarded_work(
            Activation::Silu,
            3,
            bounds,
            space,
        )?)
}

#[test]
fn activation_full_tail_real_invocation_aggregation_exact_short_overflow_and_no_reset_refund() {
    let mut space = ActivationWorkWorkspace::empty();
    let before = activation_work::entries();
    let work = schedule(&mut space).unwrap();
    let prior = WorkBudget::local().checked_mul(19).unwrap();
    let expected = prior.checked_add(work).unwrap();
    let mut total = prior;
    for (operations, aes) in [
        (expected.operations() - 1, expected.cursors().aes_work),
        (expected.operations(), expected.cursors().aes_work - 1),
        (0, expected.cursors().aes_work),
        (expected.operations(), 0),
    ] {
        assert_eq!(
            work.admit(total.operations(), total.cursors(), operations, aes),
            Err(Error::Bounds)
        );
        assert_eq!(activation_work::entries(), before);
        assert_eq!(total, prior);
    }
    work.admit(
        total.operations(),
        total.cursors(),
        expected.operations(),
        expected.cursors().aes_work,
    )
    .unwrap();
    total = total.checked_add(work).unwrap();
    let full =
        Program::activation_guarded(Activation::Silu, 128, Interval::new(-ONE, ONE).unwrap())
            .unwrap();
    let tail = Program::activation_guarded(Activation::Silu, 3, Interval::new(-ONE, ONE).unwrap())
        .unwrap();
    assert_eq!(
        work,
        full.work_budget()
            .checked_mul(2)
            .unwrap()
            .checked_add(tail.work_budget())
            .unwrap()
    );
    let after = activation_work::entries();
    assert_eq!(after[3], before[3] + 2);
    assert_eq!(work.checked_mul(u64::MAX), Err(Error::Bounds));
    assert_eq!(
        WorkBudget::local()
            .checked_mul(u64::MAX)
            .unwrap()
            .checked_add(work),
        Err(Error::Bounds)
    );
    for cursor in [
        prss::Cursors {
            public_blocks: u64::MAX,
            ..Default::default()
        },
        prss::Cursors {
            pair_blocks: u64::MAX,
            ..Default::default()
        },
        prss::Cursors {
            private_blocks: u64::MAX,
            ..Default::default()
        },
        prss::Cursors {
            aes_work: u64::MAX,
            ..Default::default()
        },
    ] {
        assert_eq!(
            work.admit(0, cursor, u64::MAX, u64::MAX),
            Err(Error::Bounds)
        );
    }
    space.reset();
    assert_eq!(total, expected);
    assert_eq!(schedule(&mut space).unwrap(), work);
    assert_eq!(activation_work::entries(), after);
}

#[test]
fn activation_query_leaves_live_prss_untouched_then_actual_native_cursors_and_equations_match() {
    for kind in [Activation::Gelu, Activation::Silu] {
        for guarded in [false, true] {
            for lanes in [1, 17, 128] {
                let bounds = Interval::new(-9 * ONE, 9 * ONE).unwrap();
                let values: Vec<_> = (0..lanes)
                    .map(|i| [-9 * ONE, -ONE, -1, 0, 1, ONE, 9 * ONE][i % 7])
                    .collect();
                let program = construct(kind, guarded, lanes, bounds).unwrap();
                let mut owners = owners(&program, &values);
                let before = owners.each_ref().map(|owner| {
                    (
                        owner
                            .runtime
                            .as_ref()
                            .unwrap()
                            .source
                            .runtime
                            .next_sequence(),
                        owner.cursors(),
                    )
                });
                let factories = activation_work::entries();
                let mut workspace = ActivationWorkWorkspace::empty();
                let work = query(kind, guarded, lanes, bounds, &mut workspace).unwrap();
                assert_eq!(work, program.work_budget());
                assert_eq!(activation_work::entries(), factories);
                assert_eq!(
                    before,
                    owners.each_ref().map(|owner| (
                        owner
                            .runtime
                            .as_ref()
                            .unwrap()
                            .source
                            .runtime
                            .next_sequence(),
                        owner.cursors()
                    ))
                );
                let mut tapes = Vec::new();
                let mut frames = 0;
                for _ in 0..program.instructions() {
                    // Existing primitive helper independently checks actual
                    // per-role ordinals and PRSS envelopes at every barrier.
                    let (tape, count) = step(&mut owners);
                    if let Some(tape) = tape {
                        tapes.push(tape);
                    }
                    frames += count;
                }
                assert_eq!(
                    frames,
                    match (kind, guarded) {
                        (Activation::Gelu, false) => 171,
                        (Activation::Gelu, true) => 174,
                        (Activation::Silu, false) => 156,
                        (Activation::Silu, true) => 159,
                    }
                );
                for (owner, (sequence, start)) in owners.iter().zip(before) {
                    assert_eq!(
                        owner
                            .runtime
                            .as_ref()
                            .unwrap()
                            .source
                            .runtime
                            .next_sequence()
                            - sequence,
                        work.operations()
                    );
                    let end = owner.cursors();
                    assert_eq!(
                        end.pair_blocks - start.pair_blocks,
                        work.cursors().pair_blocks
                    );
                    assert_eq!(
                        end.public_blocks - start.public_blocks,
                        work.cursors().public_blocks
                    );
                    assert!(
                        end.private_blocks - start.private_blocks <= work.cursors().private_blocks
                    );
                    assert!(end.aes_work - start.aes_work <= work.cursors().aes_work);
                }
                let usage = owners.each_ref().map(Executor::payload_usage);
                let output = owners.map(|owner| {
                    let (mut runtime, output) = owner.finish().unwrap();
                    runtime.cancel();
                    output
                });
                let pivots = match kind {
                    Activation::Gelu => [-1048576, -511180, 786432],
                    Activation::Silu => [-2097152, -1048576, 1048576],
                };
                let expected: Vec<_> = values
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| {
                        if guarded && value < pivots[0] {
                            0
                        } else if guarded && value > pivots[2] {
                            value
                        } else {
                            oracle(kind, value, tapes.iter().map(|t| t[i]))
                        }
                    })
                    .collect();
                assert_eq!(clear(&output), expected);
                drop(output);
                assert!(usage.iter().all(|owner| owner.live_bytes() == 0));
            }
        }
    }
}

fn silu_bound_points() -> [i64; 17] {
    [
        -9 * ONE, -8 * ONE - 1, -8 * ONE, -8 * ONE + 1,
        -4 * ONE - 1, -4 * ONE, -4 * ONE + 1,
        4 * ONE - 1, 4 * ONE, 4 * ONE + 1,
        -6 * ONE, -ONE - 1, -1, 0, 1, ONE + 1, 9 * ONE,
    ]
}

// Selected-output checks only: not bounds on eager intermediates, fallback
// integration, source provenance or an exhaustive polynomial-domain proof.
fn assert_selected_silu_bound(input: i64, output: i64) {
    if input < -8 * ONE {
        assert_eq!(output, 0);
    } else if input < -4 * ONE {
        assert!(output.unsigned_abs() <= 345_317, "low: {input}/{output}");
    } else if input <= 4 * ONE {
        assert!(output.unsigned_abs() <= 2_519_957, "high: {input}/{output}");
    } else {
        assert_eq!(output, input);
    }
    let x = i128::from(input);
    let y = i128::from(output);
    assert!(y * y <= x * x + 2_519_957_i128 * 2_519_957);
}

#[test]
fn guarded_silu_selected_bound_matches_native_tape() {
    let values = silu_bound_points();
    for (i, value) in values.iter().enumerate() {
        assert!(!values[..i].contains(value));
    }
    let program = construct(Activation::Silu, true, values.len(),
        Interval::new(-9 * ONE, 9 * ONE).unwrap()).unwrap();
    // Reuse the existing deterministic test-owner contract; no new ledger.
    let mut owners = owners(&program, &values);
    let mut tapes = Vec::new();
    for _ in 0..program.instructions() {
        let (tape, _) = step(&mut owners);
        if let Some(tape) = tape {
            assert_eq!(tape.len(), values.len());
            tapes.push(tape);
        }
    }
    // Native PRSS masks, in eager order: x², x⁴, x⁶, two low
    // coefficient products, then four high coefficient products. All TR18.
    assert_eq!(tapes.len(), 9);
    let usage = owners.each_ref().map(Executor::payload_usage);
    let output = owners.map(|owner| {
        let (mut runtime, output) = owner.finish().unwrap();
        runtime.cancel();
        drop(runtime);
        output
    });
    let actual = clear(&output);
    assert_eq!(actual.len(), values.len());
    for (i, (&input, &selected)) in values.iter().zip(&actual).enumerate() {
        // Guarded tails select the ORIGINAL input or zero. Do not claim that
        // the oracle's unused, unmasked tail intermediates match native ones.
        let expected = if input < -8 * ONE { 0 }
            else if input > 4 * ONE { input }
            else { oracle(Activation::Silu, input, tapes.iter().map(|t| t[i])) };
        assert_eq!(selected, expected, "lane {i}: {input}");
        assert_selected_silu_bound(input, selected);
    }
    drop(output);
    assert!(usage.iter().all(|owner| owner.live_bytes() == 0));
}

#[test]
fn silu_oracle_signed_tape_endpoints() {
    // Independent coefficient/radius arithmetic, not production bound queries.
    assert_eq!(80_413 + 8 * 21_489 + 64 * 1_453, 345_317);
    assert_eq!(2_229 + 4 * 131_072 + 16 * 59_806
        + 256 * 2_913 + 4096 * 71, 2_519_957);
    let last = (ONE - 1) as u64;
    let tapes: [[u64; 9]; 4] = [
        [0; 9], [last; 9],
        std::array::from_fn(|i| if i % 2 == 0 { 0 } else { last }),
        std::array::from_fn(|i| if i % 2 == 0 { last } else { 0 }),
    ];
    let mut exact_signed = [false; 2];
    let mut inexact_signed = [false; 2];
    for input in silu_bound_points() {
        // Existing independent rounded helper: signed Euclidean quotient plus
        // carry, not production evaluator. Exercise exact/nonexact signed
        // coefficient products at floor/ceil mask endpoints explicitly.
        for coefficient in [-21_489_i128, -1_453, 131_072, 59_806, -2_913, 71] {
            let raw = coefficient * i128::from(input);
            let floor = raw.div_euclid(i128::from(ONE));
            let inexact = raw.rem_euclid(i128::from(ONE)) != 0;
            assert_eq!(i128::from(rounded(raw, 0, 18)), floor);
            assert_eq!(i128::from(rounded(raw, last, 18)), floor + i128::from(inexact));
            if raw != 0 {
                let sign = usize::from(raw > 0);
                if inexact { inexact_signed[sign] = true; }
                else { exact_signed[sign] = true; }
            }
        }
        for tape in tapes {
            // Existing oracle independently evaluates all nine eager TR18
            // operations and selects zero/low/high/identity at exact pivots.
            let selected = oracle(Activation::Silu, input, tape.into_iter());
            assert_selected_silu_bound(input, selected);
            // At exact Q-multiple pivots all nine divisions are exact;
            // independently pin low at -8Q and HIGH at both -4Q and +4Q.
            if input == -8 * ONE {
                assert_eq!(selected, -80_413 + 8 * 21_489 - 64 * 1_453);
            } else if input == -4 * ONE {
                assert_eq!(selected, 2_229 - 4 * 131_072 + 16 * 59_806
                    - 256 * 2_913 + 4096 * 71);
            } else if input == 4 * ONE {
                assert_eq!(selected, 2_229 + 4 * 131_072 + 16 * 59_806
                    - 256 * 2_913 + 4096 * 71);
            }
        }
    }
    assert_eq!(exact_signed, [true; 2]);
    assert_eq!(inexact_signed, [true; 2]);
}
