use super::*;

// Independent dense integer oracle: no row-fold/layout/compression calls.
fn oracle(input: &[f64], row: usize) -> i128 {
    let mut a = [0i128; 768];
    let mut z = [0i128; 768];
    let mut s = 0i128;
    let mut squares = 0i128;
    for c in 0..768 {
        a[c] = input[row * 1024 + c] as i128;
        s += a[c];
        squares += a[c] * a[c];
    }
    let mut energy = 0i128;
    for c in 0..768 {
        z[c] = 768 * a[c] - s;
        energy += z[c] * z[c];
    }
    assert_eq!(energy, 768 * 768 * squares - 768 * s * s);
    energy
}

#[test]
fn all_rows_dense_independent_parity_six_full_maps() {
    let g = Geometry::new(32, 1024, 768).unwrap();
    let q = g.quote().unwrap();
    assert_eq!(
        (
            q.operations,
            q.public_vectors,
            q.workspace_cells,
            q.array_work
        ),
        (126, 22, 196608, 17760256)
    );
    let mut ledger = WorkLedger::new(1 << 28, 1 << 20);
    for case in 0..6 {
        let mut input = vec![0.0; q.slots];
        for r in 0..32 {
            for c in 0..1024 {
                input[r * 1024 + c] = if c >= 768 {
                    if case == 4 {
                        8.0
                    } else {
                        0.0
                    }
                } else {
                    match case {
                        0 => {
                            if r % 2 == 0 {
                                0.0
                            } else {
                                (r % 9) as f64
                            }
                        }
                        1 | 4 | 5 => ((r * 13 + c * 7) % 17) as f64 - 8.0,
                        2 => {
                            if c == 0 {
                                (r % 8 + 1) as f64
                            } else if c == 767 {
                                -8.0
                            } else {
                                0.0
                            }
                        }
                        3 => {
                            if r == 15 {
                                1.0
                            } else if r == 16 && c % 2 == 0 {
                                -3.0
                            } else {
                                0.0
                            }
                        }
                        _ => unreachable!(),
                    }
                };
            }
        }
        // Two row summaries plus two oracle row buffers stay within 25,600 bytes.
        let mut expected = [0i128; 32];
        let mut checked = [0i128; 32];
        ledger.debit(1 << 18, 1024).unwrap();
        for (r, v) in expected.iter_mut().enumerate() {
            *v = oracle(&input, r);
        }
        let mut public = vec![99.0; q.public_cells];
        let mut pool = vec![99.0; q.workspace_cells];
        let mut state = State::Fresh;
        {
            let mut invocation =
                Invocation::new(g, &input, &mut public, &mut pool, &mut state, &mut ledger)
                    .unwrap();
            let ops = invocation.instructions();
            assert_eq!(
                ops.iter()
                    .filter(|o| matches!(o.kernel, Kernel::Rotate(_)))
                    .count(),
                40
            );
            assert_eq!(
                ops.iter()
                    .filter(|o| matches!(o.kernel, Kernel::PublicProduct(_)))
                    .count(),
                44
            );
            assert_eq!(
                ops.iter()
                    .filter(|o| matches!(o.kernel, Kernel::Add | Kernel::Subtract))
                    .count(),
                41
            );
            assert_eq!(
                ops.iter()
                    .filter(|o| matches!(o.kernel, Kernel::Square))
                    .count(),
                1
            );
            let mut offsets = ops
                .iter()
                .filter_map(|o| {
                    if let Kernel::Rotate(d) = o.kernel {
                        Some(d)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            offsets.sort_unstable();
            offsets.dedup();
            assert_eq!(offsets.len(), 20);
            assert_eq!(
                ops[0],
                Instruction {
                    kernel: Kernel::PublicProduct(0),
                    left: 0,
                    right: None
                }
            );
            for fold_start in [1, 65] {
                for stage in 0..10 {
                    let start = fold_start + stage * 6;
                    let d = 1isize << stage;
                    assert_eq!(ops[start].kernel, Kernel::Rotate(d));
                    assert_eq!(ops[start + 1].kernel, Kernel::PublicProduct(2 + 2 * stage));
                    assert_eq!(ops[start + 2].kernel, Kernel::Rotate(d - 1024));
                    assert_eq!(ops[start].left, ops[start + 2].left);
                }
            }
            let result = invocation.execute().unwrap();
            for r in 0..32 {
                for c in 0..768 {
                    assert_eq!(
                        result[r * 1024 + c],
                        expected[r] as f64,
                        "case {case} row {r} column {c}"
                    );
                }
                for c in 768..1024 {
                    assert_eq!(result[r * 1024 + c], 0.0);
                }
                checked[r] = result[r * 1024] as i128;
            }
            assert_eq!(expected, checked);
            assert_eq!(invocation.execute(), Err(Error::Poisoned));
        }
        assert_eq!(state, State::Complete);
        assert!(pool.iter().chain(&public).all(|&v| v == 0.0));
    }
    assert_eq!(
        ledger.remaining(),
        (1 << 28) - 6 * (17760256 + (1 << 18) + q.control_work + 1024)
    );
}

#[test]
fn centering_padding_and_boundary_negative_controls() {
    // All-one active row, zero padding: omitting second mask creates false energy.
    assert_eq!(256i128 * 768 * 768, 150994944);
    let sum = 768i128;
    assert_eq!(768 * (768 - sum).pow(2) + 256 * (-sum).pow(2), 150994944);
    let wrong_width_energy = (0..768).map(|_| (1024i128 - sum).pow(2)).sum::<i128>();
    assert_eq!(wrong_width_energy, 50331648);
    assert_ne!(wrong_width_energy, 0); // Center with 768, never padded width 1024.
    let mut two_rows = [0i128; 2048];
    two_rows[..768].fill(1);
    let leaked: i128 = (0..1024).map(|d| two_rows[(767 + d) % 2048]).sum();
    let wrong_sign: i128 = (0..1024).map(|d| two_rows[(767 + 2048 - d) % 2048]).sum();
    assert_eq!(leaked, 1);
    assert_ne!(leaked, 768);
    assert_ne!(wrong_sign, 1);
    // Correct high branch wraps within current row; wrong sign selects row one.
    assert_eq!(two_rows[(767 + 512) % 1024], 1);
    assert_eq!(two_rows[1279], 0); // Global pull +512 crosses into row one.
}

#[test]
fn exact_short_workspace_work_and_poison() {
    let g = Geometry::new(2, 8, 6).unwrap();
    let q = g.quote().unwrap();
    let input = vec![1.0; q.slots];
    let mut public = vec![7.0; q.public_cells];
    let mut pool = vec![7.0; q.workspace_cells];
    let mut state = State::Fresh;
    let mut short = WorkLedger::new(q.array_work + q.control_work - 1, 1 << 20);
    assert!(matches!(
        Invocation::new(g, &input, &mut public, &mut pool, &mut state, &mut short),
        Err(Error::WorkRefused)
    ));
    assert_eq!(state, State::Fresh);
    assert!(public.iter().all(|&x| x == 7.0));
    let mut exact = WorkLedger::new(q.array_work + q.control_work, q.control_work);
    let mut control_short = WorkLedger::new(q.array_work + q.control_work, q.control_work - 1);
    assert!(matches!(
        Invocation::new(
            g,
            &input,
            &mut public,
            &mut pool,
            &mut state,
            &mut control_short
        ),
        Err(Error::ControlWorkExhausted)
    ));
    assert_eq!(control_short.remaining(), q.array_work + q.control_work);
    assert!(matches!(
        Invocation::new(
            g,
            &input,
            &mut public,
            &mut pool[..q.workspace_cells - 1],
            &mut state,
            &mut exact
        ),
        Err(Error::ShortStorage)
    ));
    assert_eq!(exact.remaining(), q.array_work + q.control_work);
    {
        let mut inv =
            Invocation::new(g, &input, &mut public, &mut pool, &mut state, &mut exact).unwrap();
        assert!(inv.execute().unwrap().iter().all(|&x| x == 0.0));
        assert_eq!(inv.live.iter().filter(|&&x| x).count(), 1);
        assert_eq!(inv.pool.len(), 6 * q.slots); // No physical release/refund.
    }
    assert_eq!(exact.remaining(), 0);
    assert!(matches!(
        Invocation::new(g, &input, &mut public, &mut pool, &mut state, &mut exact),
        Err(Error::Poisoned)
    ));
    assert_eq!(Geometry::new(32, 1024, 0), Err(Error::Geometry));
    assert_eq!(Geometry::new(33, 1024, 768), Err(Error::Geometry));
    assert_eq!(Geometry::new(1, 768, 768), Err(Error::Geometry));
    let large = 1usize << (usize::BITS / 2);
    assert_eq!(
        Geometry::new(large, large, 1),
        Err(Error::ArithmeticOverflow)
    );
}

#[test]
fn nonfinite_padding_mutated_between_invocations_overflow_and_unwind() {
    let g = Geometry::new(2, 8, 6).unwrap();
    let q = g.quote().unwrap();
    let mut ledger = WorkLedger::new(1 << 20, 1 << 20);
    let mut input = vec![0.0; q.slots];
    let mut public = vec![0.0; q.public_cells];
    let mut pool = vec![0.0; q.workspace_cells];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        input[7] = bad;
        let mut state = State::Fresh;
        assert!(matches!(
            Invocation::new(g, &input, &mut public, &mut pool, &mut state, &mut ledger),
            Err(Error::NonFinite)
        ));
        assert_eq!(state, State::Poisoned);
        assert!(pool.iter().chain(&public).all(|&x| x == 0.0));
    }
    input.fill(f64::MAX);
    let mut state = State::Fresh;
    {
        let mut inv =
            Invocation::new(g, &input, &mut public, &mut pool, &mut state, &mut ledger).unwrap();
        assert_eq!(inv.execute(), Err(Error::NonFinite));
        assert_eq!(inv.execute(), Err(Error::Poisoned));
    }
    assert_eq!(state, State::Poisoned);
    input.fill(1.0);
    state = State::Fresh;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _inv =
            Invocation::new(g, &input, &mut public, &mut pool, &mut state, &mut ledger).unwrap();
        panic!("public cancellation fixture");
    }));
    assert!(unwind.is_err());
    assert_eq!(state, State::Poisoned);
    assert!(pool.iter().chain(&public).all(|&x| x == 0.0));
}

#[test]
fn combined_clear_oracle_control_exact_boundary_no_refund() {
    let mut work = WorkLedger::new(MAX_CLEAR_WORK, 1 << 20);
    assert_eq!(work.debit(MAX_CLEAR_WORK, 1), Err(Error::WorkRefused));
    assert_eq!(work.remaining(), MAX_CLEAR_WORK);
    assert_eq!(work.control_remaining(), 1 << 20);
    work.debit(MAX_CLEAR_WORK - 1, 1).unwrap();
    assert_eq!(work.remaining(), 0);
    assert_eq!(work.debit(0, 1), Err(Error::WorkRefused));
    assert_eq!(work.remaining(), 0);
}
