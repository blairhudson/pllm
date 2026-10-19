use super::*;

fn input(builder: &mut Builder, lanes: usize, bounds: Interval) -> Words {
    let id = builder
        .push(Code::Input("x".into()), lanes, bounds)
        .unwrap();
    (0..lanes).map(|i| (id, i)).collect()
}

// Independent integer semantics for this composition only. The actual Boolean
// predicate/mux and PRSS schedule have separate core conformance tests.
fn recipe(builder: &Builder, input: &[i64]) -> Vec<Vec<i64>> {
    let mut values: Vec<Vec<i64>> = Vec::new();
    for node in &builder.nodes {
        let result = match &node.code {
            Code::Input(_) => input.to_vec(),
            Code::Gather(words) => words.iter().map(|(id, i)| values[id][i]).collect(),
            Code::CheckedRange { input, program } => {
                let bound = program.output_bounds();
                let mut safe = Vec::new();
                let mut valid = Vec::new();
                for &x in &values[*input] {
                    let ok = i128::from(x) >= i128::from(bound.lower())
                        && i128::from(x) <= i128::from(bound.upper());
                    safe.push(if ok { x } else { 0 });
                    valid.push(i64::from(ok));
                }
                assert_eq!(program.validity_lanes(), Some(safe.len()));
                safe.extend(valid);
                safe
            }
            Code::Tensor { input, program } => {
                assert_eq!(program.output_lanes(), 1);
                assert!(values[*input].iter().all(|x| [0, 1].contains(x)));
                vec![i64::from(values[*input].iter().all(|x| *x == 1))]
            }
            Code::Product {
                left,
                right,
                shape: None,
                rescale: false,
                ..
            } => values[*left]
                .iter()
                .zip(&values[*right])
                .map(|(&a, &b)| a * b)
                .collect(),
            _ => panic!("unexpected range-composition instruction"),
        };
        assert_eq!(result.len(), node.lanes);
        assert!(result
            .iter()
            .all(|&x| x >= node.bounds.lower() && x <= node.bounds.upper()));
        values.push(result);
    }
    values
}

#[test]
fn checked_range_compiler_static_proofs_preserve_aliases_and_certificates() {
    let mut b = Builder::default();
    let words = input(&mut b, 3, Interval::new(-9, 9).unwrap());
    b.record_row_energy(&words, 243).unwrap();
    let allocated = b.allocated;
    assert_eq!(b.checked_view(&words).unwrap(), words);
    b.checked_bound = Some(9);
    assert_eq!(b.checked_view(&words).unwrap(), words);
    assert_eq!(b.nodes.len(), 1);
    assert_eq!(b.validity, None);
    assert_eq!(b.allocated, allocated);
    assert_eq!(b.row_norm(&words, 3), Some(243));
}

#[test]
fn checked_range_compiler_tiles_endpoints_tail_and_sticky_integer_recipe() {
    for n in [1, 127, 128, 129, 257] {
        for invalid in [None, Some(0), Some(n - 1)] {
            let mut b = Builder {
                checked_bound: Some(9),
                ..Builder::default()
            };
            let words = input(&mut b, n, Interval::new(i64::MIN, i64::MAX).unwrap());
            let safe = b.checked_view(&words).unwrap();
            let first_valid = b.validity.unwrap();
            assert_eq!(safe.len(), n);
            assert_eq!(b.bounds(&safe).unwrap(), Interval::new(-9, 9).unwrap());
            let guards: Vec<_> = b
                .nodes
                .iter()
                .filter_map(|node| {
                    if let Code::CheckedRange { program, .. } = &node.code {
                        Some((node, program))
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(guards.len(), n.div_ceil(128));
            for (node, program) in guards {
                assert!(program.lanes() <= 128);
                assert_eq!(node.lanes, 2 * program.lanes());
                assert_eq!(program.validity_lanes(), Some(program.lanes()));
            }
            let endpoints: Vec<_> = (0..n).map(|i| [-10, -9, 0, 9, 10][i % 5]).collect();
            let endpoint_values = recipe(&b, &endpoints);
            let actual: Vec<_> = safe.iter().map(|(id, i)| endpoint_values[id][i]).collect();
            assert_eq!(
                actual,
                endpoints
                    .iter()
                    .map(|&x| if (-9..=9).contains(&x) { x } else { 0 })
                    .collect::<Vec<_>>()
            );
            // A later narrower guard succeeds on fallback zero. Earlier failure
            // must nevertheless remain sticky, with fresh exact products.
            b.checked_bound = Some(8);
            let final_safe = b.checked_view(&safe).unwrap();
            assert!(b.validity.unwrap() > first_valid);
            assert!(b.nodes.iter().any(|node| matches!(
                node.code,
                Code::Product {
                    rescale: false,
                    shape: None,
                    ..
                }
            )));
            let mut x: Vec<_> = (0..n).map(|i| [-8, 0, 8][i % 3]).collect();
            if let Some(i) = invalid {
                x[i] = if i % 2 == 0 { i64::MIN } else { i64::MAX };
            }
            let values = recipe(&b, &x);
            assert_eq!(values[b.validity.unwrap()], [i64::from(invalid.is_none())]);
            let actual: Vec<_> = final_safe.iter().map(|(id, i)| values[id][i]).collect();
            let expected: Vec<_> = x
                .iter()
                .map(|&x| if (-8..=8).contains(&x) { x } else { 0 })
                .collect();
            assert_eq!(actual, expected);
            let work = schedule_work(&b.nodes).unwrap();
            let expected_operations: u64 = b
                .nodes
                .iter()
                .map(|node| match &node.code {
                    Code::CheckedRange { program, .. } | Code::Tensor { program, .. } => {
                        program.work_budget().operations()
                    }
                    Code::Product { rescale: false, .. } => {
                        WorkBudget::product(node.lanes, false).unwrap().operations()
                    }
                    _ => 0,
                })
                .sum();
            assert_eq!(work.operations(), expected_operations);
            assert!(b.allocated < MAX_PROGRAM_BYTES);
        }
    }
}

#[test]
fn checked_range_compiler_zero_bound_separates_validity_and_rejects_bad_policy() {
    let mut b = Builder {
        checked_bound: Some(0),
        ..Builder::default()
    };
    let words = input(&mut b, 3, Interval::new(i64::MIN, i64::MAX).unwrap());
    let safe = b.checked_view(&words).unwrap();
    assert_eq!(b.bounds(&safe).unwrap(), Interval::new(0, 0).unwrap());
    let values = recipe(&b, &[i64::MIN, 0, i64::MAX]);
    assert_eq!(values[b.validity.unwrap()], [0]);
    assert!(safe.iter().all(|(id, i)| values[id][i] == 0));
    b.checked_bound = Some(-1);
    assert!(b.checked_view(&words).is_err());
    b.checked_bound = Some(1);
    assert!(b.checked_view(&Words::new()).is_err());
}

#[test]
fn checked_range_compiler_resource_failure_cannot_issue_a_checked_view() {
    let mut b = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let words = input(&mut b, 129, Interval::new(i64::MIN, i64::MAX).unwrap());
    b.allocated = MAX_PROGRAM_BYTES;
    assert!(matches!(b.checked_view(&words), Err(Error::Bounds)));
    assert_eq!(b.validity, None);
    assert_eq!(b.nodes.len(), 1);
}

#[test]
fn checked_range_compiler_native_outputs_match_independent_integer_recipe() {
    use crate::replicated_decoder::tests::{execute, workers};
    let input_values = [i64::MIN, -10, -9, -8, 0, 8, 9, 10, i64::MAX];
    let mut b = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let words = input(
        &mut b,
        input_values.len(),
        Interval::new(i64::MIN, i64::MAX).unwrap(),
    );
    let safe = b.checked_view(&words).unwrap();
    b.checked_bound = Some(8);
    let safe = b.checked_view(&safe).unwrap();
    let mut outputs: Vec<_> = safe.iter().collect();
    outputs.push((b.validity.unwrap(), 0));
    let work = schedule_work(&b.nodes).unwrap();
    let program = Arc::new(Program {
        retire: vec![vec![]; b.nodes.len()],
        nodes: b.nodes,
        inputs: 1,
        outputs,
        digest: [197; 32],
        source_binding: [0; 32],
        payload_bytes: b.allocated,
        work,
    });
    for input_values in [input_values, [-8, -7, -1, 0, 1, 2, 6, 7, 8]] {
        let result = execute(workers(
            program.clone(),
            &BTreeMap::from([("x".into(), input_values.to_vec())]),
        ));
        let mut expected: Vec<_> = input_values
            .iter()
            .map(|&x| if (-8..=8).contains(&x) { x as u64 } else { 0 })
            .collect();
        expected.push(u64::from(input_values.iter().all(|x| (-8..=8).contains(x))));
        assert_eq!(result, expected);
    }
}
