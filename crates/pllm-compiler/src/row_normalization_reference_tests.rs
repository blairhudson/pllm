use super::*;
use pllm_core::row_fold::WorkLedger;
use serde_json::json;

// Independent dense adjacent-pair reduction. Same selected arithmetic identity,
// different storage/traversal from the production rotation/mask slot program.
fn independent_row_head_tree(values: &mut [f64; 1024]) -> f64 {
    let mut extent = 1024;
    while extent > 1 {
        for pair in 0..extent / 2 {
            values[pair] = values[2 * pair] + values[2 * pair + 1];
        }
        extent /= 2;
    }
    values[0]
}

fn operation() -> ModelOperation {
    ModelOperation {
        id: "clear_eq5_reference".into(),
        operator: ModelOperator::LayerNorm,
        layer: None,
        state_kind: None,
        inputs: vec![
            "original_input".into(),
            "public_gamma".into(),
            "public_beta".into(),
        ],
        output_shape: vec![32, 1024],
        attributes: json!({"reference_policy":CENTERED_ENERGY_REFERENCE_POLICY,"dtype":"float64",
            "layout":"rows32_block1024_active768","axis":-1,"active_width":768,
            "epsilon_policy":"none_reject_zero_energy","variance":"centered_energy",
            "affine_order":"normalize_multiply_add","arithmetic":"ordered_nonfused_binary64_row_head_ascending_tree",
            "sqrt":"clear_sqrt","status":"research_reference"}),
    }
}

#[test]
fn whole_operator_admitted_affine_retained_output_and_constant_row_refusal() {
    // Two NEW whole-region attempts only: finite dense affine and constant-row
    // refusal. Never executes the historical six-full-map prefix test.
    let g = Geometry::new(32, 1024, 768).unwrap();
    let q = g.quote().unwrap();
    let mut work = WorkLedger::new(1 << 28, 1 << 20);
    work.debit(32 * q.slots + 4 * 768, 8192).unwrap();
    let budget = Budget::new(32 << 20).unwrap();
    // Admit original capacities BEFORE each fixture allocation, no visible-slice
    // refund. Policy DTO/stack/oracle owners also prepaid under the same cap.
    let input_guard = budget.charge_external(q.slots * 8).unwrap();
    let mask_guard = budget.charge_external(q.public_cells * 8).unwrap();
    let pool_guard = budget.charge_external(q.workspace_cells * 8).unwrap();
    let control_guard = budget.charge_external(65536).unwrap();
    let gamma_guard = budget.charge_external(768 * 8).unwrap();
    let beta_guard = budget.charge_external(768 * 8).unwrap();
    let _oracle_guard = budget.charge_external(65536).unwrap();
    let operation_guard = budget.charge_external(32768).unwrap();
    let op = operation();
    let mut input = vec![0.; q.slots];
    let mut masks = vec![0.; q.public_cells];
    let mut pool = vec![0.; q.workspace_cells];
    let gamma = (0..768)
        .map(|c| (if c % 2 == 0 { 1. } else { -1. }) * ((c % 7 + 1) as f64) / 4.)
        .collect::<Vec<_>>();
    let beta = (0..768)
        .map(|c| ((c % 11) as f64 - 5.) / 2.)
        .collect::<Vec<_>>();
    let mut control = ControlLedger::new(&mut work);
    for constant in [false, true] {
        control.core_work().debit(q.slots, 1024).unwrap();
        for row in 0..32 {
            for c in 0..1024 {
                input[row * 1024 + c] = if c >= 768 {
                    77.
                } else if constant && row == 17 {
                    3.
                } else {
                    ((row * 11 + c * 5) % 17) as f64 - 5.
                };
            }
        }
        let used = budget.used_bytes();
        let mut state = State::Fresh;
        {
            let reference = ChargedRowFold::normalization_reference(
                &op,
                BorrowedNormalizationBuffers {
                    clear: BorrowedClearBuffers {
                        input: &input,
                        public: &mut masks,
                        workspace: &mut pool,
                        original_guards: [&input_guard, &mask_guard, &pool_guard],
                        control_guard: &control_guard,
                    },
                    gamma: &gamma,
                    beta: &beta,
                    affine_guards: [&gamma_guard, &beta_guard],
                    operation_guard: &operation_guard,
                    operation_original_bytes: Some(32768),
                },
                &mut state,
                &mut control,
                &budget,
            )
            .unwrap();
            let result = reference.execute();
            if constant {
                assert!(matches!(
                    result,
                    Err(NormalizationRefusal::Arithmetic(
                        normalization::Error::NonPositiveEnergy
                    ))
                ));
            } else {
                let output = result.unwrap();
                assert_eq!(output.status(), ReferenceStatus::ResearchReference);
                assert!(std::ptr::eq(output.operation(), &op));
                // The oracle and separate row-invariant checks stay within the
                // existing 2^20 oracle ceiling. Prepay before scratch/loops.
                control.core_work().debit(1 << 20, 1024).unwrap();
                for row in 0..32 {
                    let mut dense = [0.; 1024];
                    dense[..768].copy_from_slice(&input[row * 1024..row * 1024 + 768]);
                    let sum = independent_row_head_tree(&mut dense);
                    let mut sum_squares = 0.;
                    for &a in &input[row * 1024..row * 1024 + 768] {
                        sum_squares += a * a;
                    }
                    let mut square = [0.; 1024];
                    for c in 0..768 {
                        let z = 768. * input[row * 1024 + c] - sum;
                        square[c] = z * z;
                    }
                    let energy = independent_row_head_tree(&mut square);
                    assert_eq!(energy, 768. * 768. * sum_squares - 768. * sum * sum);
                    let r = 768_f64.sqrt() / energy.sqrt();
                    let mut recovered = [0.; 1024];
                    let mut recovered_square = [0.; 1024];
                    for c in 0..768 {
                        let z = 768. * input[row * 1024 + c] - sum;
                        let scaled = z * r;
                        let weighted = scaled * gamma[c];
                        let expected = weighted + beta[c];
                        assert_eq!(
                            output.values()[row * 1024 + c].to_bits(),
                            expected.to_bits()
                        );
                        recovered[c] = (output.values()[row * 1024 + c] - beta[c]) / gamma[c];
                        recovered_square[c] = recovered[c] * recovered[c];
                    }
                    // Separate mathematical checks on ACTUAL output, after
                    // undoing nonzero affine weights. Fixed binary64 roundoff
                    // allowances cover affine recovery + depth-10 tree sums;
                    // never evidence of bit parity with the legacy cyclic path.
                    let mean_sum = independent_row_head_tree(&mut recovered);
                    let norm_square_sum = independent_row_head_tree(&mut recovered_square);
                    assert!(mean_sum.abs() <= 128. * f64::EPSILON * 768.);
                    assert!((norm_square_sum - 768.).abs() <= 256. * f64::EPSILON * 768.);
                    for c in 768..1024 {
                        assert_eq!(output.values()[row * 1024 + c].to_bits(), 0_f64.to_bits());
                    }
                }
                assert_eq!(budget.used_bytes(), used);
            }
        }
        assert_eq!(budget.used_bytes(), used);
        assert_eq!(
            state,
            if constant {
                State::Poisoned
            } else {
                State::Complete
            }
        );
        assert!(masks.iter().chain(&pool).all(|&v| v == 0.));
    }
    // Independent literal census: two complete 18,359,312-unit invocations,
    // original fixture initialization, two ingress fills, and one dense oracle.
    assert_eq!(control.core_work().remaining(), 229_539_808);
    assert_eq!(control.core_work().control_remaining(), 898_016);
}

#[test]
fn operator_dtype_epsilon_and_unknown_policy_refuse_without_buffer_entry() {
    let budget = Budget::new(1 << 20).unwrap();
    let guard = budget.charge_external(65536).unwrap();
    let mut work = WorkLedger::new(1 << 28, 1 << 20);
    let mut control = ControlLedger::new(&mut work);
    for case in 0..7 {
        let mut op = operation();
        match case {
            0 => op.attributes["dtype"] = json!("float32"),
            1 => op.attributes["epsilon"] = json!(0.0),
            2 => op.attributes["epsilon_policy"] = json!("inside_sqrt"),
            3 => op.attributes["arithmetic"] = json!("fused"),
            4 => op.operator = ModelOperator::RmsNorm,
            5 => op.attributes["reference_policy"] = json!("ordinary_layer_norm"),
            _ => op.attributes["active_width"] = json!(1024),
        }
        if case >= 3 {
            control.core_work().debit(0, 1024).unwrap();
            assert_eq!(
                check_normalization_policy(&op),
                Err(NormalizationRefusal::IncompatibleOperatorPolicy)
            );
            continue;
        }
        let mut public = [7.];
        let mut pool = [7.];
        let mut state = State::Fresh;
        let result = ChargedRowFold::normalization_reference(
            &op,
            BorrowedNormalizationBuffers {
                clear: BorrowedClearBuffers {
                    input: &[1.],
                    public: &mut public,
                    workspace: &mut pool,
                    original_guards: [&guard, &guard, &guard],
                    control_guard: &guard,
                },
                gamma: &[1.],
                beta: &[0.],
                affine_guards: [&guard, &guard],
                operation_guard: &guard,
                operation_original_bytes: Some(32768),
            },
            &mut state,
            &mut control,
            &budget,
        );
        assert!(matches!(
            result,
            Err(NormalizationRefusal::IncompatibleOperatorPolicy)
        ));
        drop(result);
        assert_eq!(public, [7.]);
        assert_eq!(pool, [7.]);
        assert_eq!(state, State::Fresh);
    }
}

#[test]
fn complete_compiler_precharge_one_short_before_policy_or_payload() {
    let g = Geometry::new(32, 1024, 768).unwrap();
    let (array, control_units) = normalization::total_charge(g).unwrap();
    let budget = Budget::new(1 << 20).unwrap();
    let guard = budget.charge_external(65536).unwrap();
    let mut op = operation();
    op.attributes["dtype"] = json!("float32");
    let mut public = [7.];
    let mut pool = [7.];
    let mut state = State::Fresh;
    let mut work = WorkLedger::new(array + control_units + 1024 - 1, 1 << 20);
    let mut control = ControlLedger::new(&mut work);
    let result = ChargedRowFold::normalization_reference(
        &op,
        BorrowedNormalizationBuffers {
            clear: BorrowedClearBuffers {
                input: &[1.],
                public: &mut public,
                workspace: &mut pool,
                original_guards: [&guard, &guard, &guard],
                control_guard: &guard,
            },
            gamma: &[1.],
            beta: &[0.],
            affine_guards: [&guard, &guard],
            operation_guard: &guard,
            operation_original_bytes: Some(32768),
        },
        &mut state,
        &mut control,
        &budget,
    );
    assert!(matches!(
        result,
        Err(NormalizationRefusal::Arithmetic(
            normalization::Error::Fold(Error::WorkRefused)
        ))
    ));
    drop(result);
    assert_eq!(public, [7.]);
    assert_eq!(pool, [7.]);
    assert_eq!(state, State::Fresh);
    assert_eq!(
        control.core_work().remaining(),
        array + control_units + 1024 - 1
    );
}

#[test]
fn unknown_original_operator_owner_refuses_before_payload_entry() {
    let budget = Budget::new(1 << 20).unwrap();
    let guard = budget.charge_external(65536).unwrap();
    let op = operation();
    let mut work = WorkLedger::new(1 << 28, 1 << 20);
    let mut control = ControlLedger::new(&mut work);
    let mut public = [7.];
    let mut pool = [7.];
    let mut state = State::Fresh;
    let before = control.core_work().remaining();
    let result = ChargedRowFold::normalization_reference(
        &op,
        BorrowedNormalizationBuffers {
            clear: BorrowedClearBuffers {
                input: &[1.],
                public: &mut public,
                workspace: &mut pool,
                original_guards: [&guard, &guard, &guard],
                control_guard: &guard,
            },
            gamma: &[1.],
            beta: &[0.],
            affine_guards: [&guard, &guard],
            operation_guard: &guard,
            operation_original_bytes: None,
        },
        &mut state,
        &mut control,
        &budget,
    );
    assert!(matches!(
        result,
        Err(NormalizationRefusal::UnknownOriginalCapacity)
    ));
    drop(result);
    assert!(control.core_work().remaining() < before); // Complete payment burns.
    assert_eq!(public, [7.]);
    assert_eq!(pool, [7.]);
    assert_eq!(state, State::Fresh);
}
