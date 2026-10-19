use super::*;

fn check_energy(actual: &[i64], width: usize, bound: u128) {
    for row in actual.chunks_exact(width) {
        assert!(
            row.iter()
                .map(|&v| i128::from(v).unsigned_abs().pow(2))
                .sum::<u128>()
                <= bound
        );
    }
}

fn compare(legacy: &Program, compact: &Program, values: &[i64]) -> (Vec<i64>, Traffic) {
    assert_ne!(legacy.digest(), compact.digest());
    assert_eq!(compact.numeric_profile(), COMPACT_NONLINEAR_PROFILE);
    assert_eq!(legacy.instructions(), compact.instructions());
    assert_eq!(legacy.output_bounds(), compact.output_bounds());
    let (old, old_traffic) = run_encoded(legacy, values);
    let (actual, traffic) = run_encoded(compact, values);
    assert!(
        traffic.directed.iter().flatten().sum::<usize>()
            < old_traffic.directed.iter().flatten().sum::<usize>()
    );
    eprintln!("ROW width={} legacy_total={} compact_total={} legacy_direction={:?} compact_direction={:?}",
        values.len(), old_traffic.directed.iter().flatten().sum::<usize>(), traffic.directed.iter().flatten().sum::<usize>(), old_traffic.directed, traffic.directed);
    // Reciprocal/rsqrt retain their fixed backing widths and all arithmetic
    // draws. Without sign predicates, compact/legacy tapes and outputs agree.
    if !compact
        .nodes
        .iter()
        .flatten()
        .any(|n| matches!(n.instruction, Instruction::CompactSign(..)))
    {
        assert_eq!(legacy.work_budget(), compact.work_budget());
        assert_eq!(old_traffic.tapes, traffic.tapes);
        assert_eq!(old, actual);
    }
    (actual, traffic)
}

#[test]
fn compact_rows_rms_layer_norm_preserve_actual_tapes_equations_and_energy() {
    for (rows, width) in [(2, 1), (3, 3), (2, 8), (1, 128), (1, 896)] {
        let bounds = Interval::new(-4 * ONE, 4 * ONE).unwrap();
        let gamma: Vec<_> = (0..width).map(|i| [ONE, -ONE / 2, 0][i % 3]).collect();
        let beta: Vec<_> = (0..width).map(|i| i as i64 - 2).collect();
        let values: Vec<_> = (0..rows * width)
            .map(|i| [0, -4 * ONE, 4 * ONE, -1, 1, ONE + 3][i % 6])
            .collect();
        let legacy = Program::rms_norm(rows, width, bounds, &gamma).unwrap();
        let compact = Program::rms_norm_compact(rows, width, bounds, &gamma).unwrap();
        let (actual, traffic) = compare(&legacy, &compact, &values);
        assert_eq!(
            actual,
            rms_norm_oracle(&values, width, &gamma, &traffic.tapes)
        );
        let (legacy, old_energy) =
            Program::rms_norm_certified(rows, width, bounds, &gamma).unwrap();
        let (compact, energy) =
            Program::rms_norm_certified_compact(rows, width, bounds, &gamma).unwrap();
        assert_eq!(old_energy, energy);
        let (actual, traffic) = compare(&legacy, &compact, &values);
        assert_eq!(
            actual,
            rms_norm_oracle(&values, width, &gamma, &traffic.tapes)
        );
        check_energy(&actual, width, energy);
        let legacy = Program::layer_norm(rows, width, bounds, &gamma, &beta).unwrap();
        let compact = Program::layer_norm_compact(rows, width, bounds, &gamma, &beta).unwrap();
        let (actual, traffic) = compare(&legacy, &compact, &values);
        assert_eq!(
            actual,
            row_oracle(true, &values, width, &gamma, &beta, &traffic.tapes)
        );
        let (certified, energy) =
            Program::layer_norm_certified_compact(rows, width, bounds, &gamma, &beta).unwrap();
        check_energy(&actual, width, energy);
        assert_ne!(certified.digest(), compact.digest());
    }
}

#[test]
fn compact_rows_sparse_input_energy_and_exact_obligations_are_bound() {
    let bounds = Interval::new(-512 * ONE, 512 * ONE).unwrap();
    let input_energy = (512 * ONE) as u128 * (512 * ONE) as u128;
    for width in [3, 128, 896] {
        let gamma = vec![ONE; width];
        assert!(Program::rms_norm_certified_compact(1, width, bounds, &gamma).is_err());
        let (legacy, old_energy) =
            Program::rms_norm_with_input_norm(1, width, bounds, &gamma, input_energy).unwrap();
        let (compact, energy) =
            Program::rms_norm_with_input_norm_compact(1, width, bounds, &gamma, input_energy)
                .unwrap();
        assert_eq!(energy, old_energy);
        for value in [-512 * ONE, 0, 512 * ONE] {
            let mut values = vec![0; width];
            values[width / 2] = value;
            let (actual, traffic) = compare(&legacy, &compact, &values);
            assert_eq!(
                actual,
                rms_norm_oracle(&values, width, &gamma, &traffic.tapes)
            );
            check_energy(&actual, width, energy);
        }
    }
    let bounds = Interval::new(-ONE, ONE).unwrap();
    let energy = 3 * (ONE as u128).pow(2);
    let a = Program::rms_norm_with_input_norm_compact(1, 3, bounds, &[ONE; 3], energy)
        .unwrap()
        .0;
    let b = Program::rms_norm_with_input_norm_compact(1, 3, bounds, &[ONE; 3], energy + 1)
        .unwrap()
        .0;
    assert_ne!(a.digest(), b.digest());
    assert_eq!(a.work_budget(), b.work_budget());
    assert!(Program::rms_norm_with_input_norm_compact(
        1,
        3,
        Interval::new(ONE, ONE).unwrap(),
        &[ONE; 3],
        0
    )
    .is_err());
    assert!(Program::rms_norm_with_input_norm_compact(1, 3, bounds, &[ONE; 3], u128::MAX).is_err());
}

#[test]
fn compact_rows_softmax_eager_guarded_and_certified_preserve_strict_cutoff_and_mass() {
    for (rows, width) in [(2, 1), (3, 3), (2, 7), (1, 128)] {
        for guarded in [false, true] {
            let bounds = Interval::new(if guarded { -256 * ONE } else { -20 * ONE }, 0).unwrap();
            let values: Vec<_> = (0..rows * width)
                .map(|i| {
                    [
                        0,
                        -14 * ONE - 1,
                        -14 * ONE,
                        -14 * ONE + 1,
                        -ONE,
                        bounds.lower(),
                        0,
                    ][i % width % 7]
                })
                .collect();
            let legacy = if guarded {
                Program::softmax_guarded(rows, width, bounds)
            } else {
                Program::softmax(rows, width, bounds)
            }
            .unwrap();
            let compact = if guarded {
                Program::softmax_guarded_compact(rows, width, bounds)
            } else {
                Program::softmax_compact(rows, width, bounds)
            }
            .unwrap();
            let (actual, traffic) = compare(&legacy, &compact, &values);
            assert_eq!(actual, softmax_oracle(&values, width, &traffic.tapes));
            if guarded {
                let (legacy, old_mass) = Program::softmax_certified(rows, width, bounds).unwrap();
                let (compact, mass) =
                    Program::softmax_certified_compact(rows, width, bounds).unwrap();
                assert_eq!(old_mass, mass);
                let (actual, traffic) = compare(&legacy, &compact, &values);
                assert_eq!(actual, softmax_oracle(&values, width, &traffic.tapes));
                for row in actual.chunks_exact(width) {
                    assert!(row.iter().all(|&v| v >= 0));
                    assert!(row.iter().map(|&v| v as u64).sum::<u64>() <= mass);
                }
            }
        }
    }
    // Broad signed64 comparison plus safe fallback before eager powers.
    let bounds = Interval::new(-1_500_000_000_000, 1_500_000_000_000).unwrap();
    let (compact, mass) = Program::softmax_certified_compact(1, 3, bounds).unwrap();
    let values = [bounds.lower(), -14 * ONE + 1, 0];
    let (actual, traffic) = run_encoded(&compact, &values);
    assert_eq!(actual, softmax_oracle(&values, 3, &traffic.tapes));
    assert!(actual.iter().map(|&v| v as u64).sum::<u64>() <= mass);
}

#[test]
fn compact_rows_domain_rejection_cancellation_and_certificate_replay() {
    let bounds = Interval::new(-ONE, ONE).unwrap();
    for (rows, width) in [(0, 3), (1, 0), (usize::MAX, 3), (1, MAX_ROW_LANES + 1)] {
        assert!(Program::softmax_certified_compact(rows, width, bounds).is_err());
        assert!(Program::rms_norm_compact(rows, width, bounds, &[ONE; 3]).is_err());
        assert!(Program::layer_norm_compact(rows, width, bounds, &[ONE; 3], &[0; 3]).is_err());
    }
    assert!(Program::softmax_compact(1, 3, Interval::new(-128 * ONE, 0).unwrap()).is_err());
    assert!(
        Program::softmax_guarded_compact(1, 3, Interval::new(i64::MIN, i64::MAX).unwrap()).is_err()
    );
    assert!(Program::layer_norm_compact(1, 3, bounds, &[i64::MAX; 3], &[0; 3]).is_err());
    let energy = 3 * (ONE as u128).pow(2);
    let a = Program::rms_norm_with_input_norm_compact(1, 3, bounds, &[ONE; 3], energy)
        .unwrap()
        .0;
    let b = Program::rms_norm_with_input_norm_compact(1, 3, bounds, &[ONE; 3], energy + 1)
        .unwrap()
        .0;
    let mut old = owners(&a, &[0, ONE, -ONE]);
    let mut current = owners(&b, &[0, ONE, -ONE]);
    let mut stale: Vec<_> = old
        .iter_mut()
        .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
        .collect();
    for o in &mut current {
        o.start_step().unwrap();
    }
    let stale = stale.remove(0);
    let to = stale.recipient() as usize;
    assert!(current[to].receive(stale).is_err());
    assert!(current[to].retired());
    for program in [
        a,
        Program::layer_norm_certified_compact(1, 3, bounds, &[ONE; 3], &[0; 3])
            .unwrap()
            .0,
        Program::softmax_certified_compact(1, 3, bounds).unwrap().0,
    ] {
        cancel_at_every_event(&program, &[0, ONE, -ONE]);
    }
}
