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
fn checked_rows_static_alias_preserves_certificates_resources_and_schedule() {
    for checked_bound in [None, Some(9)] {
        let mut b = Builder {
            checked_bound,
            ..Builder::default()
        };
        let words = input(&mut b, 6, Interval::new(-9, 9).unwrap());
        b.record_row_energy(&words.slice(..3), 150).unwrap();
        b.record_row_energy(&words.slice(3..), 160).unwrap();
        let allocated = b.allocated;
        let node_count = b.nodes.len();
        let certificates = b.row_energies.clone();
        let work = schedule_work(&b.nodes).unwrap();
        let safe = b.checked_rows(&words, 3).unwrap();
        assert_eq!(safe, words);
        assert_eq!(b.row_norm(&safe, 3), Some(160));
        assert_eq!(b.row_energies, certificates);
        assert!(b.row_norms.is_empty());
        assert_eq!(b.allocated, allocated);
        assert_eq!(b.nodes.len(), node_count);
        assert!(b.tensors.is_empty());
        assert_eq!(b.workspace_bytes, 0);
        assert_eq!(b.validity, None);
        assert_eq!(schedule_work(&b.nodes).unwrap(), work);
    }
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

#[test]
fn checked_rows_zero_mux_preserves_energy_even_when_validity_fails() {
    let mut b = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let words = input(&mut b, 3, Interval::new(-12, 12).unwrap());
    b.record_row_energy(&words, 150).unwrap();
    let safe = b.checked_rows(&words, 3).unwrap();
    assert_eq!(b.row_norm(&safe, 3), Some(150));
    // Independent exhaustive integer zero-mux proof. Include rejected rows;
    // success-only reasoning would not justify a pre-acceptance constructor.
    let mut rejected = 0;
    for x in -12_i64..=12 {
        for y in -12_i64..=12 {
            for z in -12_i64..=12 {
                let original = [x, y, z];
                let energy: i128 = original.iter().map(|&v| i128::from(v).pow(2)).sum();
                if energy > 150 {
                    continue;
                }
                let values = recipe(&b, &original);
                let actual: Vec<_> = safe.iter().map(|(id, i)| values[id][i]).collect();
                let expected: Vec<_> = original
                    .iter()
                    .map(|&v| if (-9..=9).contains(&v) { v } else { 0 })
                    .collect();
                assert_eq!(actual, expected);
                let safe_energy: i128 = actual.iter().map(|&v| i128::from(v).pow(2)).sum();
                assert!(safe_energy <= energy.min(3 * 9 * 9));
                let valid = original.iter().all(|v| (-9..=9).contains(v));
                assert_eq!(values[b.validity.unwrap()], [i64::from(valid)]);
                rejected += usize::from(!valid);
            }
        }
    }
    assert!(rejected > 0);
}

#[test]
fn checked_rows_transfer_only_exact_complete_ordered_provenance() {
    for indices in [vec![0, 1], vec![2, 1, 0], vec![0, 0, 2]] {
        let mut b = Builder {
            checked_bound: Some(9),
            ..Builder::default()
        };
        let row = input(&mut b, 3, Interval::new(-12, 12).unwrap());
        b.record_row_energy(&row, 150).unwrap();
        let view: Words = indices.iter().map(|&i| row.at(i)).collect();
        let safe = b.checked_rows(&view, view.len()).unwrap();
        assert_eq!(b.row_norm(&safe, safe.len()), None);
    }
    let mut b = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let row = input(&mut b, 3, Interval::new(-12, 12).unwrap());
    let other = b
        .push(
            Code::Input("other".into()),
            3,
            Interval::new(-12, 12).unwrap(),
        )
        .unwrap();
    b.record_row_energy(&row, 150).unwrap();
    let mixed: Words = [row.at(0), row.at(1), (other, 2)].into_iter().collect();
    let safe = b.checked_rows(&mixed, 3).unwrap();
    assert_eq!(b.row_norm(&safe, 3), None);
    assert!(matches!(b.checked_rows(&row, 2), Err(Error::Shape)));
}

#[test]
fn checked_rows_ordered_fragmented_alias_uses_existing_authority_only() {
    let build = |certified: bool| {
        let mut b = Builder {
            checked_bound: Some(9),
            ..Builder::default()
        };
        let mut row = input(&mut b, 1, Interval::new(-12, 12).unwrap());
        let tail = input(&mut b, 2, Interval::new(-12, 12).unwrap());
        row.extend(tail.iter());
        if certified {
            b.record_row_energy(&row, 150).unwrap();
        }
        let mut alias = row.slice(..1);
        alias.extend(row.slice(1..).iter());
        assert!(alias.iter().eq(row.iter()));
        let safe = b.checked_rows(&alias, 3).unwrap();
        (b, safe)
    };
    let (with_proof, safe) = build(true);
    let (without_proof, uncertified) = build(false);
    assert_eq!(with_proof.row_norm(&safe, 3), Some(150));
    assert_eq!(without_proof.row_norm(&uncertified, 3), None);
    assert_eq!(with_proof.validity, without_proof.validity);
    assert_eq!(with_proof.nodes.len(), without_proof.nodes.len());
    assert_eq!(
        schedule_work(&with_proof.nodes).unwrap(),
        schedule_work(&without_proof.nodes).unwrap()
    );
    // Equal geometry and bounds on another owner cannot acquire this proof.
    let mut b = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let row = input(&mut b, 3, Interval::new(-12, 12).unwrap());
    b.record_row_energy(&row, 150).unwrap();
    let other = input(&mut b, 3, Interval::new(-12, 12).unwrap());
    assert_eq!(b.row_norm(&other, 3), None);
    let safe = b.checked_rows(&other, 3).unwrap();
    assert_eq!(b.row_norm(&safe, 3), None);
}

#[test]
fn checked_rows_strict_energy_benefit_admits_sparse_rms_norm_without_guard_work_changes() {
    let width = 3;
    let cap = 512 * ONE;
    let energy = (cap as u128).pow(2);
    let build = |transfer: bool| {
        let mut b = Builder {
            checked_bound: Some(cap),
            ..Builder::default()
        };
        let row = input(&mut b, width, Interval::new(-2 * cap, 2 * cap).unwrap());
        b.record_row_energy(&row, energy).unwrap();
        let safe = if transfer {
            b.checked_rows(&row, width)
        } else {
            b.checked_view(&row)
        }
        .unwrap();
        (b, safe)
    };
    let (baseline, _) = build(false);
    let (b, safe) = build(true);
    assert_eq!(b.nodes.len(), baseline.nodes.len());
    assert_eq!(b.validity, baseline.validity);
    assert_eq!(
        schedule_work(&b.nodes).unwrap(),
        schedule_work(&baseline.nodes).unwrap()
    );
    for (new, old) in b.nodes.iter().zip(&baseline.nodes) {
        match (&new.code, &old.code) {
            (Code::CheckedRange { program: a, .. }, Code::CheckedRange { program: b, .. })
            | (Code::Tensor { program: a, .. }, Code::Tensor { program: b, .. }) => {
                assert_eq!(a.digest(), b.digest());
            }
            _ => {}
        }
    }
    assert!(b.allocated > baseline.allocated); // certificate charged, not free
    let bounds = b.bounds(&safe).unwrap();
    let gamma = [ONE; 3];
    assert!(tensor::Program::rms_norm_certified(1, width, bounds, &gamma).is_err());
    let norm = b.row_norm(&safe, width).unwrap();
    assert_eq!(norm, energy);
    let admitted = tensor::Program::rms_norm_with_input_norm(1, width, bounds, &gamma, norm)
        .unwrap()
        .0;
    let narrow =
        tensor::Program::rms_norm_certified(1, width, Interval::new(-ONE, ONE).unwrap(), &gamma)
            .unwrap()
            .0;
    // Existing fixed highest-bit/rsqrt circuit uses these proof envelopes only
    // for admission. No different number/order of stochastic instructions.
    assert_eq!(admitted.work_budget(), narrow.work_budget());
    assert_eq!(admitted.instructions(), narrow.instructions());
    assert_eq!(admitted.admitted_bytes(), narrow.admitted_bytes());
}

#[test]
fn admission_ledger_distinguishes_guard_bytes_nodes_and_values() {
    let mut b = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let row = input(&mut b, 3, Interval::new(-12, 12).unwrap());
    b.allocated = MAX_PROGRAM_BYTES;
    assert!(matches!(b.checked_view(&row), Err(Error::Bounds)));
    let ledger = b.admission_ledger();
    assert!(ledger.contains("stage: checked_range_view_metadata"));
    assert!(ledger.contains("limit: Some(\"program_bytes\")"));
    assert!(ledger.contains(&format!(
        "requested_bytes: Some({})",
        MAX_PROGRAM_BYTES + 128
    )));

    let mut b = Builder::default();
    assert!(matches!(
        b.push(
            Code::Input("x".into()),
            MAX_VALUES + 1,
            Interval::new(0, 0).unwrap()
        ),
        Err(Error::Bounds)
    ));
    assert!(b.admission_ledger().contains("limit: Some(\"values\")"));

    let mut b = Builder::default();
    for i in 0..MAX_NODES {
        b.push(
            Code::Input(format!("x{i}")),
            1,
            Interval::new(0, 0).unwrap(),
        )
        .unwrap();
    }
    assert!(matches!(
        b.push(
            Code::Input("overflow".into()),
            1,
            Interval::new(0, 0).unwrap()
        ),
        Err(Error::Bounds)
    ));
    assert!(b.admission_ledger().contains("limit: Some(\"nodes\")"));
}

#[test]
fn checked_rows_exact_multitile_rows_and_nonbeneficial_energy() {
    let mut b = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let width = tensor::MAX_TILE_LANES + 1;
    let words = input(&mut b, width * 2, Interval::new(-12, 12).unwrap());
    b.record_row_energy(&words.slice(..width), 150).unwrap();
    b.record_row_energy(&words.slice(width..), 160).unwrap();
    let safe = b.checked_rows(&words, width).unwrap();
    assert_eq!(b.row_norm(&safe.slice(..width), width), Some(150));
    assert_eq!(b.row_norm(&safe.slice(width..), width), Some(160));
    assert_eq!(b.row_norm(&safe, width), Some(160));

    let mut b = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let words = input(&mut b, 3, Interval::new(-12, 12).unwrap());
    b.row_norms.push((words.at(0).0, 150));
    let safe = b.checked_rows(&words, 3).unwrap();
    assert_eq!(b.row_norm(&safe, 3), Some(150));

    let mut b = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let words = input(&mut b, 3, Interval::new(-12, 12).unwrap());
    b.record_row_energy(&words, 3 * 12 * 12).unwrap();
    let safe = b.checked_rows(&words, 3).unwrap();
    // Scalar cap is already tighter; no duplicate proof storage.
    assert_eq!(b.row_norm(&safe, 3), None);
    assert_eq!(b.row_energies.len(), 1);
}

#[test]
fn row_certificate_metadata_obeys_original_program_byte_cap() {
    let mut b = Builder::default();
    let words = input(&mut b, 3, Interval::new(-12, 12).unwrap());
    b.allocated = MAX_PROGRAM_BYTES;
    b.admission_stage = "checked_range_row_certificate";
    assert!(matches!(
        b.record_row_energy(&words, 150),
        Err(Error::Bounds)
    ));
    assert!(b.row_energies.is_empty());
    assert_eq!(b.allocated, MAX_PROGRAM_BYTES);
    assert!(b
        .admission_ledger()
        .contains("limit: Some(\"program_bytes\")"));
}

#[test]
fn checked_rows_energy_product_overflow_returns_no_certificate() {
    let build = || {
        let mut b = Builder {
            checked_bound: Some(i64::MAX),
            ..Builder::default()
        };
        let row = input(&mut b, 5, Interval::new(i64::MIN, i64::MAX).unwrap());
        b.record_row_energy(&row, 150).unwrap();
        (b, row)
    };
    let (mut guard_only, row) = build();
    guard_only.checked_view(&row).unwrap();
    let (mut b, row) = build();
    assert!((i64::MAX as u128).pow(2).checked_mul(5).is_none());
    assert!(matches!(b.checked_rows(&row, 5), Err(Error::Bounds)));
    assert_eq!(b.row_energies.len(), 1);
    assert_eq!(b.nodes.len(), guard_only.nodes.len());
    assert_eq!(b.allocated, guard_only.allocated);
    assert_eq!(
        schedule_work(&b.nodes).unwrap(),
        schedule_work(&guard_only.nodes).unwrap()
    );
}

#[test]
fn checked_rows_equal_energy_endpoints_and_zero_cap_add_no_certificate() {
    for cap in [0_i64, 9] {
        let mut b = Builder {
            checked_bound: Some(cap),
            ..Builder::default()
        };
        let row = input(&mut b, 3, Interval::new(-12, 12).unwrap());
        b.record_row_energy(&row, 3 * (cap as u128).pow(2)).unwrap();
        let safe = b.checked_rows(&row, 3).unwrap();
        assert_eq!(b.row_energies.len(), 1);
        assert_eq!(b.row_norm(&safe, 3), None);
        let values = recipe(&b, &[-cap, 0, cap]);
        assert_eq!(
            safe.iter().map(|(id, i)| values[id][i]).collect::<Vec<_>>(),
            [-cap, 0, cap]
        );
        assert_eq!(values[b.validity.unwrap()], [1]);
        let values = recipe(&b, &[cap + 1, 0, -cap - 1]);
        assert_eq!(
            safe.iter().map(|(id, i)| values[id][i]).collect::<Vec<_>>(),
            [0, 0, 0]
        );
        assert_eq!(values[b.validity.unwrap()], [0]);
    }
}

#[test]
fn checked_rows_guard_fits_but_certificate_exhausts_whole_admission_budget() {
    let build = || {
        let mut b = Builder {
            checked_bound: Some(9),
            ..Builder::default()
        };
        let row = input(&mut b, 129, Interval::new(-12, 12).unwrap());
        b.record_row_energy(&row, 150).unwrap();
        (b, row)
    };
    let (mut probe, row) = build();
    let initial = probe.allocated;
    let safe = probe.checked_view(&row).unwrap();
    let guard_bytes = probe.allocated - initial;
    let proof_bytes = safe.metadata_bytes() + std::mem::size_of::<(Words, u128)>();
    let (mut b, row) = build();
    b.allocated = MAX_PROGRAM_BYTES - guard_bytes - proof_bytes + 1;
    let admission = b.checked_rows(&row, 129);
    assert!(matches!(admission, Err(Error::Bounds)));
    assert_eq!(b.nodes.len(), probe.nodes.len()); // every guard fit
    assert_eq!(b.validity, probe.validity);
    assert_eq!(b.row_energies.len(), 1); // no output proof was issued
    assert_eq!(b.allocated, MAX_PROGRAM_BYTES - proof_bytes + 1);
    assert_eq!(b.admission_stage, "checked_range_row_certificate");
    assert_eq!(b.admission_limit, Some("program_bytes"));
}

#[test]
fn checked_rows_rms_norm_native_final_acceptance_stochastic_oracle_tail() {
    use crate::replicated_decoder::runtime::oracle_trace::audit_checked_row_energy;
    let width = 129;
    let cap = 512 * ONE;
    let energy = 2 * (cap as u128 + 1).pow(2);
    let mut b = Builder {
        checked_bound: Some(cap),
        public_weights: true,
        ..Builder::default()
    };
    let row = input(&mut b, width, Interval::new(-2 * cap, 2 * cap).unwrap());
    b.record_row_energy(&row, energy).unwrap();
    let safe = b.checked_rows(&row, width).unwrap();
    let first_valid = b.validity.unwrap();
    assert_eq!(b.row_norm(&safe, width), Some(energy));
    assert_eq!(
        b.nodes
            .iter()
            .filter(|n| matches!(&n.code, Code::CheckedRange { .. }))
            .count(),
        2
    );
    let input = b.gather(&safe).unwrap();
    let gamma = vec![ONE; width];
    assert!(tensor::Program::rms_norm_certified(1, width, b.nodes[input].bounds, &gamma).is_err());
    let (norm, _) =
        tensor::Program::rms_norm_with_input_norm(1, width, b.nodes[input].bounds, &gamma, energy)
            .unwrap();
    let rms = b.tensor(input, norm).unwrap();
    // A later, independently successful guard must not clear an earlier failure.
    b.checked_bound = Some(cap / 2);
    let later_safe = b.checked_rows(&safe, width).unwrap();
    let sticky = b.validity.unwrap();
    assert_ne!(first_valid, sticky);
    let logits = b.view(rms, vec![1, width]);
    let selected = b.argmax_tree(&logits).unwrap();
    let selected = b.gather(&selected.words).unwrap();
    let masked = b.product(selected, sticky, None, false).unwrap();
    let work = schedule_work(&b.nodes).unwrap();
    let program = Arc::new(Program {
        retire: vec![vec![]; b.nodes.len()],
        nodes: b.nodes,
        inputs: 1,
        outputs: vec![(masked, 0), (sticky, 0)],
        digest: [198; 32],
        source_binding: [0; 32],
        payload_bytes: b.allocated,
        work,
    });
    let checkpoints = vec![
        ("safe", safe.iter().collect()),
        ("rms", (0..width).map(|i| (rms, i)).collect()),
        ("first_valid", vec![(first_valid, 0)]),
        ("later_safe", later_safe.iter().collect()),
        ("sticky", vec![(sticky, 0)]),
        ("selected", vec![(selected, 0)]),
    ];
    let mut accepted_schedule = None;
    for rejected in [false, true] {
        let mut values = vec![0; width];
        if rejected {
            values[1] = 128 * ONE + 3; // nonzero winning ID before final masking
            values[width - 1] = cap + 1; // failure in the one-lane guard tail
        } else {
            values[0] = -128 * ONE + 7;
            values[width - 1] = 128 * ONE + 3;
        }
        assert!(
            values
                .iter()
                .map(|&x| i128::from(x).unsigned_abs().pow(2))
                .sum::<u128>()
                <= energy
        );
        let schedule =
            audit_checked_row_energy(program.clone(), values, cap, rms, &checkpoints, rejected);
        if let Some(accepted) = &accepted_schedule {
            assert_eq!(&schedule, accepted);
        } else {
            accepted_schedule = Some(schedule);
        }
    }
}
