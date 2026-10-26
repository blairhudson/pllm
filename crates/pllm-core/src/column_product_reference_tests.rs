use super::*;
fn limits() -> Limits {
    Limits {
        max_buffer_elements: MAX_BYTES / 8,
        max_work: MAX_VISITS,
    }
}
fn fixture(m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    (
        (0..m * n)
            .map(|x| ((x * 7 + x / n * 3 + 2) % 17) as f64 - 8.0)
            .collect(),
        (0..m * n)
            .map(|x| ((x * 11 + x / n * 5 + 9) % 17) as f64 - 8.0)
            .collect(),
    )
}
// Independent INTEGER row-pair contraction, not column/diagonal traversal.
fn dense(q: &[f64], k: &[f64], m: usize, n: usize) -> Vec<f64> {
    q.chunks_exact(n)
        .flat_map(|row| {
            k.chunks_exact(n).map(move |key| {
                row.iter()
                    .zip(key)
                    .map(|(&a, &b)| (a as i64) * (b as i64))
                    .sum::<i64>() as f64
            })
        })
        .take(m * m)
        .collect()
}
#[test]
fn integer_dense_parity_all_three_asymmetric_fixtures_and_final_rotation() {
    for (m, n) in [(1, 1), (4, 3), (128, 64)] {
        let g = Geometry::new(m, n).unwrap();
        let quote = g.quote(limits()).unwrap();
        let (q, k) = fixture(m, n);
        let expected = dense(&q, &k, m, n);
        let mut workspace = vec![9.0; quote.workspace_cells];
        let mut scores = vec![9.0; quote.output_cells];
        let mut state = State::Fresh;
        let mut work = WorkLedger::new(quote.visits, 0);
        let p = prepay(g, limits(), &mut work, 0).unwrap();
        let output = Reference::new_prepaid(p, &q, &k, &mut workspace, &mut scores, &mut state)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(output.values(), expected);
        for j in 0..n {
            for t in 0..m {
                assert_eq!(output.final_key_columns()[j * m + t], k[t * n + j]);
            }
        }
        drop(output);
        assert_eq!(work.remaining(), 0);
        assert!(workspace.iter().chain(&scores).all(|&x| x == 0.0));
        assert_eq!(state, State::Complete);
    }
}
#[test]
fn wrong_rotation_sign_is_detected() {
    let (q, k) = fixture(4, 3);
    let expected = dense(&q, &k, 4, 3);
    let mut wrong = vec![0.0; 16];
    for i in 0..4 {
        for t in 0..4 {
            for j in 0..3 {
                wrong[t * 4 + (t + i) % 4] += q[t * 3 + j] * k[((t + 4 - i) % 4) * 3 + j];
            }
        }
    }
    assert_ne!(wrong, expected);
}
#[test]
fn scatter_addresses_are_bijective() {
    for m in [1, 4, 128] {
        let g = Geometry::new(m, 3).unwrap();
        let mut seen = vec![false; m * m];
        for i in 0..m {
            for t in 0..m {
                let a = g.scatter_address(i, t).unwrap();
                assert!(!seen[a]);
                seen[a] = true;
            }
        }
        assert!(seen.into_iter().all(|b| b));
        assert!(g.scatter_address(m, 0).is_err());
        assert!(g.scatter_address(0, m).is_err());
    }
}
#[test]
fn wrap_first_and_last_diagonals() {
    let g = Geometry::new(4, 3).unwrap();
    assert_eq!(g.scatter_address(0, 0).unwrap(), 0);
    assert_eq!(g.scatter_address(0, 3).unwrap(), 15);
    assert_eq!(g.scatter_address(3, 0).unwrap(), 3);
    assert_eq!(g.scatter_address(3, 3).unwrap(), 14);
    assert_eq!(g.scatter_address(1, 3).unwrap(), 12);
}
#[test]
fn immutable_original_inputs_survive_cleanup() {
    let g = Geometry::new(4, 3).unwrap();
    let quote = g.quote(limits()).unwrap();
    let (q, k) = fixture(4, 3);
    let before = (q.clone(), k.clone());
    let mut w = vec![0.0; quote.workspace_cells];
    let mut o = vec![0.0; 16];
    let mut s = State::Fresh;
    let mut work = WorkLedger::new(MAX_VISITS, 0);
    let p = prepay(g, limits(), &mut work, 0).unwrap();
    drop(
        Reference::new_prepaid(p, &q, &k, &mut w, &mut o, &mut s)
            .unwrap()
            .execute()
            .unwrap(),
    );
    assert_eq!((q, k), before);
}
#[test]
fn short_exact_storage_and_tighter_caps() {
    let g = Geometry::new(4, 3).unwrap();
    let quote = g.quote(limits()).unwrap();
    let (q, k) = fixture(4, 3);
    for field in 0..4 {
        let mut w = vec![7.0; quote.workspace_cells - usize::from(field == 2)];
        let mut o = vec![7.0; 16 - usize::from(field == 3)];
        let mut s = State::Fresh;
        let mut work = WorkLedger::new(MAX_VISITS, 0);
        let p = prepay(g, limits(), &mut work, 0).unwrap();
        assert!(matches!(
            Reference::new_prepaid(
                p,
                &q[..12 - usize::from(field == 0)],
                &k[..12 - usize::from(field == 1)],
                &mut w,
                &mut o,
                &mut s
            ),
            Err(Error::Layout(LayoutError::Buffer))
        ));
        assert_eq!(s, State::Fresh);
        assert!(w.iter().chain(&o).all(|&x| x == 7.0));
    }
    assert!(g
        .quote(Limits {
            max_buffer_elements: quote.workspace_cells - 1,
            ..limits()
        })
        .is_err());
    assert!(g
        .quote(Limits {
            max_work: quote.visits - 1,
            ..limits()
        })
        .is_err());
    assert!(Geometry::new(usize::MAX, 2).is_err());
    assert!(Geometry::new(0, 3).is_err());
}
#[test]
fn short_exact_cumulative_work_including_prior_use_and_failure() {
    let g = Geometry::new(4, 3).unwrap();
    let quote = g.quote(limits()).unwrap();
    for delta in [0, 1] {
        let mut ledger = WorkLedger::new(quote.visits + 1024 + 99 - delta, 1024);
        ledger.debit(99, 0).unwrap();
        let payment = prepay(g, limits(), &mut ledger, 1024);
        assert_eq!(payment.is_ok(), delta == 0);
        assert_eq!(
            ledger.remaining(),
            if delta == 0 { 0 } else { quote.visits + 1023 }
        );
        assert!(prepay(g, limits(), &mut ledger, 1024).is_err());
    }
}
#[test]
fn partial_failure_poison_cleanup_no_output_and_no_refund() {
    let g = Geometry::new(4, 3).unwrap();
    let quote = g.quote(limits()).unwrap();
    for nonfinite_input in [false, true] {
        let mut q = vec![1.0; 12];
        let k = vec![2.0; 12];
        q[6] = if nonfinite_input { f64::NAN } else { f64::MAX };
        let mut w = vec![7.0; quote.workspace_cells];
        let mut o = vec![7.0; 16];
        let mut s = State::Fresh;
        let mut work = WorkLedger::new(quote.visits, 0);
        let p = prepay(g, limits(), &mut work, 0).unwrap();
        let result =
            Reference::new_prepaid(p, &q, &k, &mut w, &mut o, &mut s).and_then(Reference::execute);
        assert!(matches!(result, Err(Error::Layout(LayoutError::NonFinite))));
        drop(result);
        assert_eq!(s, State::Poisoned);
        assert_eq!(work.remaining(), 0);
        assert!(w.iter().chain(&o).all(|&x| x == 0.0));
    }
}

#[test]
fn rounding_sensitive_nonfused_ascending_binary64_literal_oracles() {
    // (1+2^-27)*(1-2^-27)=1-2^-54 rounds to 1 BEFORE addition.
    // Literal expected zero differs from fused multiply/add's exact -2^-54.
    let a = 1.0000000074505806_f64;
    let b = 0.9999999925494194_f64;
    assert_eq!(a.mul_add(b, -1.0), -5.551115123125783e-17);
    // Ascending ((2^54 + -2^54) + 1)=1; reassociation loses final 1.
    assert_eq!(
        18014398509481984.0_f64 + (-18014398509481984.0_f64 + 1.0),
        0.0
    );
    for (query, key, expected) in [
        (vec![-1.0, a], vec![1.0, b], 0.0_f64),
        (
            vec![18014398509481984.0, -18014398509481984.0, 1.0],
            vec![1.0; 3],
            1.0,
        ),
    ] {
        let g = Geometry::new(1, query.len()).unwrap();
        let quote = g.quote(limits()).unwrap();
        let mut workspace = vec![7.0; quote.workspace_cells];
        let mut scores = [7.0];
        let mut state = State::Fresh;
        let mut work = WorkLedger::new(quote.visits, 0);
        let payment = prepay(g, limits(), &mut work, 0).unwrap();
        let output = Reference::new_prepaid(
            payment,
            &query,
            &key,
            &mut workspace,
            &mut scores,
            &mut state,
        )
        .unwrap()
        .execute()
        .unwrap();
        // Expected is literal, never derived from candidate or a float reducer.
        assert_eq!(output.values()[0].to_bits(), expected.to_bits());
        drop(output);
        assert_eq!(work.remaining(), 0);
    }
}

#[test]
fn poisoned_reentry_consumes_no_additional_work_and_mutates_no_output() {
    let g = Geometry::new(1, 1).unwrap();
    let quote = g.quote(limits()).unwrap();
    let mut work = WorkLedger::new(2 * quote.visits, 0);
    // Reserve both attempts first. Reentry uses its existing opaque payment;
    // prepay itself remains nonrefundable and does not inspect invocation state.
    let first = prepay(g, limits(), &mut work, 0).unwrap();
    let second = prepay(g, limits(), &mut work, 0).unwrap();
    let mut workspace = [7.0; 3];
    let mut scores = [7.0];
    let mut state = State::Fresh;
    assert!(matches!(
        Reference::new_prepaid(
            first,
            &[f64::NAN],
            &[1.0],
            &mut workspace,
            &mut scores,
            &mut state
        ),
        Err(Error::Layout(LayoutError::NonFinite))
    ));
    assert_eq!(state, State::Poisoned);
    workspace.fill(9.0);
    scores.fill(11.0);
    let before = (work.remaining(), work.control_remaining());
    assert!(matches!(
        Reference::new_prepaid(
            second,
            &[1.0],
            &[1.0],
            &mut workspace,
            &mut scores,
            &mut state
        ),
        Err(Error::Poisoned)
    ));
    assert_eq!((work.remaining(), work.control_remaining()), before);
    assert_eq!(workspace, [9.0; 3]);
    assert_eq!(scores, [11.0]);
    assert_eq!(state, State::Poisoned);
}

#[test]
fn oversized_original_backings_refuse_without_prefix_or_suffix_writes() {
    let g = Geometry::new(4, 3).unwrap();
    let quote = g.quote(limits()).unwrap();
    for field in 0..4 {
        let query = vec![1.0; quote.input_cells + usize::from(field == 0)];
        let key = vec![2.0; quote.input_cells + usize::from(field == 1)];
        let mut workspace = vec![7.0; quote.workspace_cells + usize::from(field == 2)];
        let mut scores = vec![11.0; quote.output_cells + usize::from(field == 3)];
        let mut state = State::Fresh;
        let mut work = WorkLedger::new(quote.visits, 0);
        let payment = prepay(g, limits(), &mut work, 0).unwrap();
        assert!(matches!(
            Reference::new_prepaid(
                payment,
                &query,
                &key,
                &mut workspace,
                &mut scores,
                &mut state
            ),
            Err(Error::Layout(LayoutError::Buffer))
        ));
        assert!(workspace.iter().all(|&x| x == 7.0));
        assert!(scores.iter().all(|&x| x == 11.0));
        assert_eq!(state, State::Fresh);
        assert_eq!(work.remaining(), 0);
    }
}
