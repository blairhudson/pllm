use super::*;

// Independent dense adjacent-pair reduction: no rotations, masks or production
// fold. This binds the row-head binary64 tree under adversarial cancellation.
fn independent_pair_sum(values: &[f64]) -> f64 {
    let mut dense = values.to_vec();
    while dense.len() > 1 {
        dense = dense
            .chunks_exact(2)
            .map(|pair| pair[0] + pair[1])
            .collect();
    }
    dense[0]
}

#[test]
fn common_row_scalar_binary64_cancellation_not_per_lane_means() {
    let g = Geometry::new(1, 8, 6).unwrap();
    let q = g.quote().unwrap();
    let input = [1e16, 1., -1e16, 3., 0.125, -0.25, 99., -99.];
    let gamma = [0.25, -2., 3., -0.5, 1., 0.];
    let beta = [1., -1., 2., 0., 0.25, 8.];
    let sum = independent_pair_sum(&[1e16, 1., -1e16, 3., 0.125, -0.25, 0., 0.]);
    assert_eq!(sum, 3.875);
    // O603/60570 counterexample: the legacy cyclic lane-one tree rounds to 4.0.
    // The approved scalar-row reference broadcasts 3.875, not that lane's sum.
    assert_eq!(
        independent_pair_sum(&[1., -1e16, 3., 0.125, -0.25, 0., 0., 1e16]),
        4.0
    );
    let mut z = [0.; 8];
    let mut square = [0.; 8];
    for c in 0..6 {
        z[c] = 6. * input[c] - sum;
        square[c] = z[c] * z[c];
    }
    let energy = independent_pair_sum(&square);
    let r = 6_f64.sqrt() / energy.sqrt();
    let mut masks = vec![0.; q.public_cells];
    let mut pool = vec![0.; q.workspace_cells];
    let mut state = State::Fresh;
    let mut work = WorkLedger::new(1 << 20, 1 << 20);
    let output = Reference::new(
        g, &input, &gamma, &beta, &mut masks, &mut pool, &mut state, &mut work,
    )
    .unwrap()
    .execute()
    .unwrap();
    assert_eq!(output.energy.centered()[1], 2.125);
    for c in 0..6 {
        let scaled = z[c] * r;
        let weighted = scaled * gamma[c];
        let expected = weighted + beta[c];
        assert_eq!(output.values()[c].to_bits(), expected.to_bits());
    }
    assert_eq!(&output.values()[6..], &[0., 0.]);
}

// Independently dense row arithmetic, not the production fold or energy API.
// Small integer inputs make sums/z/energy exact; affine ordering remains f64.
fn dense(input: &[f64], gamma: &[f64], beta: &[f64], g: Geometry) -> Vec<f64> {
    let mut expected = vec![0.0; g.rows() * g.block()];
    for row in 0..g.rows() {
        let entries = &input[row * g.block()..row * g.block() + g.width()];
        let sum = entries.iter().copied().sum::<f64>();
        let centered = entries
            .iter()
            .map(|a| g.width() as f64 * a - sum)
            .collect::<Vec<_>>();
        let v = centered.iter().map(|z| z * z).sum::<f64>();
        let r = (g.width() as f64).sqrt() / v.sqrt();
        for (col, &z) in centered.iter().enumerate() {
            let scaled = z * r;
            let weighted = scaled * gamma[col];
            expected[row * g.block() + col] = weighted + beta[col];
        }
    }
    expected
}

#[test]
fn whole_clear_region_dense_affine_and_consuming_prefix() {
    let g = Geometry::new(2, 8, 6).unwrap();
    let input = [
        -3., 1., 4., 8., 2., 0., 99., -99., 2., 7., 3., -1., 8., 6., 4., 5.,
    ];
    let gamma = [0.5, -2., 1., 0., 3., -0.25];
    let beta = [1., 2., -3., 4., 0., 0.5];
    let q = g.quote().unwrap();
    let mut state = State::Fresh;
    let (array, control) = total_charge(g).unwrap();
    let prefix_work = q.array_work + q.control_work;
    let mut work = WorkLedger::new(
        array + control + prefix_work + 5120,
        control + q.control_work + 1024,
    );
    work.debit(4096, 1024).unwrap(); // Oracle allocations/loops prepaid separately.
    let mut masks = vec![1.; q.public_cells];
    let mut pool = vec![1.; q.workspace_cells];
    let expected = dense(&input, &gamma, &beta, g);
    {
        let output = Reference::new(
            g, &input, &gamma, &beta, &mut masks, &mut pool, &mut state, &mut work,
        )
        .unwrap()
        .execute()
        .unwrap();
        for (actual, want) in output.values().iter().zip(&expected) {
            assert_eq!(actual.to_bits(), want.to_bits());
        }
    }
    assert_eq!(work.remaining(), prefix_work);
    assert_eq!(state, State::Complete);
    assert!(masks.iter().chain(&pool).all(|&x| x == 0.));
    // New consuming prefix returns the ACTUAL z/energy, never a replay. This
    // tiny geometry is distinct from the six historical full-map campaign.
    state = State::Fresh;
    let centered = Invocation::new(g, &input, &mut masks, &mut pool, &mut state, &mut work)
        .unwrap()
        .into_centered_energy()
        .unwrap();
    assert_eq!(&centered.centered()[..6], &[-30., -6., 12., 36., 0., -12.]);
    assert_eq!(centered.energy()[0], 2520.);
    assert!(centered.centered()[6..8].iter().all(|&x| x == 0.));
    assert_eq!(work.remaining(), 0);
}

#[test]
fn whole_clear_exact_one_short_preentry_and_control_exhaustion() {
    let g = Geometry::new(1, 4, 3).unwrap();
    let q = g.quote().unwrap();
    let input = [1., 3., 8., 5.];
    let gamma = [1.; 3];
    let beta = [0.; 3];
    let mut masks = vec![7.; q.public_cells];
    let mut pool = vec![7.; q.workspace_cells];
    let mut state = State::Fresh;
    let (array, control) = total_charge(g).unwrap();
    let mut work = WorkLedger::new(array + control - 1, control);
    assert!(matches!(
        Reference::new(g, &input, &gamma, &beta, &mut masks, &mut pool, &mut state, &mut work),
        Err(Error::Fold(FoldError::WorkRefused))
    ));
    assert_eq!(work.remaining(), array + control - 1);
    assert_eq!(state, State::Fresh);
    assert!(masks.iter().chain(&pool).all(|&x| x == 7.));
    let mut work = WorkLedger::new(array + control, control - 1);
    assert!(matches!(
        Reference::new(g, &input, &gamma, &beta, &mut masks, &mut pool, &mut state, &mut work),
        Err(Error::Fold(FoldError::ControlWorkExhausted))
    ));
    let mut work = WorkLedger::new(array + control, control);
    assert!(matches!(
        Reference::new(
            g,
            &input,
            &gamma,
            &beta,
            &mut masks,
            &mut pool[..q.workspace_cells - 1],
            &mut state,
            &mut work
        ),
        Err(Error::Fold(FoldError::ShortStorage))
    ));
    assert_eq!(work.remaining(), array + control);
}

#[test]
fn zero_energy_nonfinite_and_affine_overflow_burn_and_cleanup() {
    let g = Geometry::new(1, 4, 3).unwrap();
    let q = g.quote().unwrap();
    let (array, control) = total_charge(g).unwrap();
    for case in 0..7 {
        let mut input = [1., 1., 1., 7.];
        let mut gamma = [1.; 3];
        let beta = [0.; 3];
        match case {
            1 => input[1] = f64::NAN,
            2 => {
                input[1] = 3.;
                gamma[2] = f64::INFINITY;
            }
            3 => {
                input[1] = 3.;
                gamma[0] = f64::MAX;
                gamma[1] = f64::MAX;
            }
            4 => input[1] = f64::MAX,
            5 => {
                input[..3].fill(0.);
                input[1] = f64::MIN_POSITIVE;
            }
            6 => input[3] = f64::NAN,
            _ => {}
        }
        let mut masks = vec![7.; q.public_cells];
        let mut pool = vec![7.; q.workspace_cells];
        let mut state = State::Fresh;
        let mut work = WorkLedger::new(array + control, control);
        let result = Reference::new(
            g, &input, &gamma, &beta, &mut masks, &mut pool, &mut state, &mut work,
        )
        .and_then(Reference::execute);
        if case == 0 || case == 5 {
            assert!(matches!(result, Err(Error::NonPositiveEnergy)));
        } else {
            assert!(result.is_err());
        }
        drop(result);
        assert_eq!(work.remaining(), 0);
        assert_eq!(state, State::Poisoned);
        assert!(masks.iter().chain(&pool).all(|&x| x == 0.));
    }
}

#[test]
fn cancellation_retains_spent_work_and_cleans_whole_pool() {
    let g = Geometry::new(1, 4, 3).unwrap();
    let q = g.quote().unwrap();
    let mut masks = vec![7.; q.public_cells];
    let mut pool = vec![7.; q.workspace_cells];
    let mut state = State::Fresh;
    let (array, control) = total_charge(g).unwrap();
    let mut work = WorkLedger::new(array + control, control);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _reference = Reference::new(
            g,
            &[1., 3., 8., 4.],
            &[1.; 3],
            &[0.; 3],
            &mut masks,
            &mut pool,
            &mut state,
            &mut work,
        )
        .unwrap();
        panic!("public clear-reference cancellation");
    }));
    assert!(result.is_err());
    assert_eq!(state, State::Poisoned);
    assert_eq!(work.remaining(), 0);
    assert!(masks.iter().chain(&pool).all(|&x| x == 0.));
}
