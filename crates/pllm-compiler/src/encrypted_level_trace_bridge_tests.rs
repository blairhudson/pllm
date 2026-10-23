use super::*;

fn limits() -> PublicEncryptedLevelTraceCapacity {
    PublicEncryptedLevelTraceCapacity {
        levels: PublicLevelSolverCapacity {
            max_operations: 8,
            max_levels: 4,
            max_table_rows: 128,
            max_vertices: 64,
            max_work: 100_000,
            max_buffer_bytes: 8 << 20,
        },
        trace: PublicCiphertextTraceCapacity {
            max_records: 1024,
            max_work: usize::MAX,
            max_buffer_bytes: 1 << 20,
            max_live_owner_bytes: 1000,
        },
    }
}

fn charge(id: &str, units: u64) -> PublicCost {
    PublicCost {
        identity: id.into(),
        units: Some(units),
    }
}

fn preserve() -> PublicLevelTransition {
    PublicLevelTransition {
        kind: PublicLevelTransitionKind::Preserve,
        from_level: 1,
        to_level: 1,
        cost: charge("preserve-one", 0),
    }
}

fn refresh(from: u32, units: u64) -> PublicLevelTransition {
    PublicLevelTransition {
        kind: PublicLevelTransitionKind::Refresh,
        from_level: from,
        to_level: 2,
        cost: charge(&format!("complete-refresh-from-{from}"), units),
    }
}

fn trip(evaluate: u64) -> PublicHybridRoundTrip {
    PublicHybridRoundTrip {
        identity: "complete-round-trip".into(),
        input_level: 1,
        output_level: 2,
        entry: charge("complete-entry", 1),
        evaluate: charge("hybrid-evaluate", evaluate),
        recover: charge("complete-recovery", 3),
    }
}

/// Independent numeric problem: no graph, capability, trace, or solver output is
/// used to construct it. Synthetic units only, not any paper's numeric model.
fn level_problem(hybrid_wins: bool) -> PublicLevelProblem {
    PublicLevelProblem {
        cost_model_identity: "bridge-synthetic-cost-model".into(),
        cost_unit: "synthetic-unit".into(),
        maximum_level: 2,
        input_level: 2,
        output_level: 2,
        operations: vec![
            PublicLevelOperation {
                identity: "first".into(),
                predecessors: vec![],
                primitive: PublicPrimitive::Arithmetic,
                depth: 1,
                compute: vec![
                    PublicLevelCompute {
                        input_level: 1,
                        cost: charge("first-at-one", 5),
                    },
                    PublicLevelCompute {
                        input_level: 2,
                        cost: charge("first-at-two", 3),
                    },
                ],
                transitions: vec![preserve(), refresh(1, 7)],
                hybrid: vec![],
            },
            PublicLevelOperation {
                identity: "second".into(),
                predecessors: vec!["first".into()],
                primitive: PublicPrimitive::Inv,
                depth: 1,
                compute: vec![
                    PublicLevelCompute {
                        input_level: 1,
                        cost: charge("second-at-one", 6),
                    },
                    PublicLevelCompute {
                        input_level: 2,
                        cost: charge("second-at-two", 4),
                    },
                ],
                transitions: vec![refresh(0, 8), refresh(1, 8)],
                hybrid: vec![trip(if hybrid_wins { 2 } else { 20 })],
            },
        ],
    }
}

fn state(level: u32) -> PublicCiphertextState {
    PublicCiphertextState {
        scheme: "synthetic-carrier-not-crypto".into(),
        key_domain: "external-client-domain".into(),
        layout: "synthetic-layout".into(),
        scale: "synthetic-scale".into(),
        level,
        shape: vec![4],
        error: PublicCiphertextError {
            model_identity: "external-error-contract".into(),
            bound: Some(1),
        },
    }
}

fn value(id: &str, owner: &str, level: u32) -> PublicCiphertextValue {
    PublicCiphertextValue {
        identity: id.into(),
        owner: owner.into(),
        state: state(level),
    }
}

/// Separately authored SSA trace. In particular, this NEVER consumes the level
/// problem or solution to select rows or construct typed executions.
fn execution_graph(hybrid: bool) -> (PublicCiphertextGraph, Vec<PublicCiphertextTraceStep>) {
    let graph = PublicCiphertextGraph {
        cost_model_identity: "bridge-synthetic-cost-model".into(),
        cost_unit: "synthetic-unit".into(),
        maximum_level: 2,
        owners: [("input-backing", 40), ("a-backing", 12), ("b-backing", 14)]
            .into_iter()
            .map(|(id, bytes)| PublicCiphertextOwner {
                identity: id.into(),
                bytes: Some(bytes),
            })
            .collect(),
        persistent_owners: vec![],
        inputs: vec![
            value("input", "input-backing", 2),
            value("parent-alias", "input-backing", 2),
        ],
        operations: vec![
            PublicCiphertextGraphOperation {
                identity: "first".into(),
                kernel_identity: "first-kernel".into(),
                primitive: PublicPrimitive::Arithmetic,
                inputs: vec!["input".into()],
                output: value("a", "a-backing", 1),
            },
            PublicCiphertextGraphOperation {
                identity: "second".into(),
                kernel_identity: "second-kernel".into(),
                primitive: PublicPrimitive::Inv,
                inputs: vec!["a".into()],
                output: value("b", "b-backing", 2),
            },
        ],
        capabilities: vec![
            PublicCiphertextCapability {
                identity: "first-cap".into(),
                kernel_identity: "first-kernel".into(),
                primitive: PublicPrimitive::Arithmetic,
                inputs: vec![state(2)],
                output: state(1),
                join: PublicCiphertextJoin::Unary,
                execution: PublicCiphertextExecution::Ckks {
                    compute: PublicLevelCompute {
                        input_level: 2,
                        cost: charge("first-at-two", 3),
                    },
                    depth: 1,
                    transition: preserve(),
                },
                error_assumption: Some("external-first-error-assumption".into()),
                scratch_bytes: Some(0),
                recovery_overlap_bytes: Some(0),
            },
            PublicCiphertextCapability {
                identity: "second-cap".into(),
                kernel_identity: "second-kernel".into(),
                primitive: PublicPrimitive::Inv,
                inputs: vec![state(1)],
                output: state(2),
                join: PublicCiphertextJoin::Unary,
                execution: if hybrid {
                    PublicCiphertextExecution::Hybrid {
                        round_trip: trip(2),
                    }
                } else {
                    PublicCiphertextExecution::Ckks {
                        compute: PublicLevelCompute {
                            input_level: 1,
                            cost: charge("second-at-one", 6),
                        },
                        depth: 1,
                        transition: refresh(0, 8),
                    }
                },
                error_assumption: Some("external-second-error-assumption".into()),
                scratch_bytes: Some(5),
                recovery_overlap_bytes: Some(7),
            },
        ],
        outputs: vec!["b".into()],
        retained_inputs: vec!["parent-alias".into()],
    };
    let trace = vec![
        PublicCiphertextTraceStep {
            operation: "first".into(),
            capability: "first-cap".into(),
            release: vec!["input".into()],
        },
        PublicCiphertextTraceStep {
            operation: "second".into(),
            capability: "second-cap".into(),
            release: vec!["a".into()],
        },
    ];
    (graph, trace)
}

fn binding() -> PublicEncryptedLevelTraceBinding {
    PublicEncryptedLevelTraceBinding {
        input_value: "input".into(),
        scheme: "synthetic-carrier-not-crypto".into(),
        key_domain: "external-client-domain".into(),
        kernel_identities: vec!["first-kernel".into(), "second-kernel".into()],
    }
}

fn check<'a>(
    p: &'a PublicLevelProblem,
    s: &'a PublicLevelSolution,
    g: &'a PublicCiphertextGraph,
    t: &'a [PublicCiphertextTraceStep],
    b: &'a PublicEncryptedLevelTraceBinding,
) -> Result<CheckedPublicEncryptedLevelTrace<'a>, String> {
    validate_public_encrypted_level_trace(p, s, g, t, b, limits())
}

#[test]
fn level_trace_bridge_independent_compositions_choose_competing_costs_and_borrow_inputs() {
    for hybrid in [false, true] {
        let p = level_problem(hybrid);
        let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
        let (g, t) = execution_graph(hybrid);
        let b = binding();
        let originals = (p.clone(), s.clone(), g.clone(), t.clone(), b.clone());
        let checked = check(&p, &s, &g, &t, &b).unwrap();
        // Direct enumeration: preserve first -> second CKKS costs 3+6+8=17;
        // refresh first -> second CKKS costs 3+7+4+8=22; atomic hybrid costs
        // 3+1+evaluate+3=9 or 27. No hybrid entry exists at level two.
        assert_eq!(
            checked.trace_summary().total_cost,
            if hybrid { 9 } else { 17 }
        );
        assert_eq!(checked.trace_summary().peak_live_backing_bytes, 66);
        assert_eq!(checked.trace_summary().peak_live_owner_bytes, 78);
        assert_eq!(checked.trace_summary().final_live_backing_bytes, 54);
        assert!(std::ptr::eq(checked.problem(), &p));
        assert!(std::ptr::eq(checked.solution(), &s));
        assert!(std::ptr::eq(checked.graph(), &g));
        assert!(std::ptr::eq(checked.trace().as_ptr(), t.as_ptr()));
        assert!(std::ptr::eq(checked.binding(), &b));
        assert_eq!(
            originals,
            (p.clone(), s.clone(), g.clone(), t.clone(), b.clone())
        );
        // Storage order is not execution order; no accidental zip with graph rows.
        let mut reordered = g.clone();
        reordered.operations.reverse();
        reordered.capabilities.reverse();
        assert!(check(&p, &s, &reordered, &t, &b).is_ok());
    }
}

#[test]
fn level_trace_bridge_rejects_forged_ledger_hashes_indices_levels_order_and_charges() {
    let p = level_problem(true);
    let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
    let (g, t) = execution_graph(true);
    let b = binding();
    for field in 0..15 {
        let mut forged = s.clone();
        match field {
            0 => forged.ledger_digest = Digest::from_sha256([0; 32]),
            1 => forged.ledger.problem_digest = Digest::from_sha256([0; 32]),
            2 => forged.ledger.steps[0].operation_identity = "wrong-node".into(),
            3 => forged.ledger.steps[0].input_level = 1,
            4 => forged.ledger.steps[0].output_level = 2,
            5 => forged.ledger.steps.swap(0, 1),
            6 => forged.ledger.steps[0].total_cost += 1,
            7 => forged.ledger.total_cost += 1,
            8 => forged.ledger.steps[0].charges[0].identity = "wrong-charge".into(),
            9 => forged.ledger.steps[0].charges[0].units = None,
            10 => forged.ledger.steps[0].charges.clear(),
            11 => forged.ledger.steps.clear(),
            12 => {
                forged.ledger.steps[0].choice = PublicLevelChoice::Ckks {
                    compute_row: usize::MAX,
                    transition_row: 0,
                }
            }
            13 => {
                forged.ledger.steps[0].choice = PublicLevelChoice::Ckks {
                    compute_row: 1,
                    transition_row: usize::MAX,
                }
            }
            _ => {
                forged.ledger.steps[1].choice = PublicLevelChoice::Hybrid {
                    round_trip_row: usize::MAX,
                }
            }
        }
        // Most forgeries have a self-consistent new hash; hashing is not enough.
        if field > 1 {
            forged.ledger_digest = canonical_digest(ENCRYPTED_LEVEL_LEDGER_DOMAIN, &forged.ledger);
        }
        assert!(
            check(&p, &forged, &g, &t, &b)
                .unwrap_err()
                .contains("solution/problem/digest"),
            "field {field}"
        );
    }
}

#[test]
fn level_trace_bridge_binds_unused_alternatives_problem_model_and_chain() {
    let p = level_problem(true);
    let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
    let (g, t) = execution_graph(true);
    let b = binding();
    for field in 0..7 {
        let mut changed = p.clone();
        match field {
            0 => changed.operations[0].compute[0].cost.units = Some(99),
            1 => changed.operations[1].hybrid[0].identity = "other-round-trip".into(),
            2 => changed.cost_model_identity = "other-cost-model".into(),
            3 => changed.cost_unit = "other-unit".into(),
            4 => changed.operations[1]
                .predecessors
                .push("other-parent".into()),
            5 => changed.operations[0].compute[0].cost.units = None,
            _ => changed.operations[0].compute.reverse(),
        }
        assert!(check(&changed, &s, &g, &t, &b).is_err(), "field {field}");
    }
    let mut wrong_graph = g.clone();
    wrong_graph.maximum_level = 3;
    assert!(check(&p, &s, &wrong_graph, &t, &b).is_err());
}

#[test]
fn level_trace_bridge_exact_selected_cost_rows_not_equal_totals() {
    for hybrid in [false, true] {
        let p = level_problem(hybrid);
        let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
        let (g, t) = execution_graph(hybrid);
        let b = binding();
        for field in 0..5 {
            let mut changed = g.clone();
            match &mut changed.capabilities[1].execution {
                PublicCiphertextExecution::Hybrid { round_trip } => match field {
                    0 => round_trip.identity = "other-trip".into(),
                    1 => round_trip.entry.identity = "other-entry".into(),
                    2 => round_trip.evaluate.identity = "other-evaluate".into(),
                    3 => round_trip.recover.identity = "other-recovery".into(),
                    _ => {
                        round_trip.entry.units = Some(2);
                        round_trip.evaluate.units = Some(1);
                    }
                },
                PublicCiphertextExecution::Ckks {
                    compute,
                    depth,
                    transition,
                } => match field {
                    0 => compute.cost.identity = "other-compute".into(),
                    1 => transition.cost.identity = "other-refresh".into(),
                    2 => {
                        compute.cost.units = Some(7);
                        transition.cost.units = Some(7);
                    }
                    3 => {
                        *depth = 0;
                        transition.from_level = 1;
                    }
                    _ => {
                        compute.cost.units = Some(5);
                        transition.cost.units = Some(9);
                    }
                },
                _ => unreachable!(),
            }
            let independent =
                validate_public_ciphertext_trace(&changed, &t, limits().trace).unwrap();
            assert_eq!(independent.total_cost, s.ledger.total_cost);
            assert!(check(&p, &s, &changed, &t, &b)
                .unwrap_err()
                .contains("selected execution/cost/transition"));
        }
    }
}

#[test]
fn level_trace_bridge_competing_legal_equal_cost_rows_cannot_replace_selected_identity() {
    for hybrid in [false, true] {
        let mut p = level_problem(hybrid);
        let (mut g, t) = execution_graph(hybrid);
        let b = binding();
        // Both rows are explicitly legal in the same problem, not merely forged
        // capabilities. Exact stable tie choice still binds the first identity.
        if hybrid {
            let mut alternate = trip(2);
            alternate.identity = "equally-priced-other-trip".into();
            alternate.recover.identity = "other-complete-recovery".into();
            p.operations[1].hybrid.push(alternate.clone());
            g.capabilities[1].execution = PublicCiphertextExecution::Hybrid {
                round_trip: alternate,
            };
        } else {
            let mut alternate = refresh(0, 8);
            alternate.cost.identity = "equally-priced-other-refresh".into();
            p.operations[1].transitions.push(alternate.clone());
            if let PublicCiphertextExecution::Ckks { transition, .. } =
                &mut g.capabilities[1].execution
            {
                *transition = alternate;
            }
        }
        let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
        assert_eq!(
            validate_public_ciphertext_trace(&g, &t, limits().trace)
                .unwrap()
                .total_cost,
            s.ledger.total_cost
        );
        assert!(check(&p, &s, &g, &t, &b).is_err());
    }
}

#[test]
fn level_trace_bridge_binds_order_operator_kernel_and_rejects_multi_input_planning() {
    let p = level_problem(true);
    let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
    let (g, t) = execution_graph(true);
    let b = binding();
    for field in 0..4 {
        let mut changed = g.clone();
        let mut trace = t.clone();
        match field {
            0 => {
                changed.operations[0].identity = "second".into();
                changed.operations[1].identity = "first".into();
                trace[0].operation = "second".into();
                trace[1].operation = "first".into();
            }
            1 => {
                changed.operations[1].primitive = PublicPrimitive::Exp;
                changed.capabilities[1].primitive = PublicPrimitive::Exp;
            }
            2 => {
                changed.operations[1].kernel_identity = "other-kernel".into();
                changed.capabilities[1].kernel_identity = "other-kernel".into();
            }
            _ => {
                changed.operations[0].inputs.push("parent-alias".into());
                changed.capabilities[0].inputs.push(state(2));
                changed.capabilities[0].join = PublicCiphertextJoin::Aligned;
            }
        }
        assert!(validate_public_ciphertext_trace(&changed, &trace, limits().trace).is_ok());
        assert!(
            check(&p, &s, &changed, &trace, &b).is_err(),
            "field {field}"
        );
    }
}

fn map_states(g: &mut PublicCiphertextGraph, mut f: impl FnMut(&mut PublicCiphertextState)) {
    for v in &mut g.inputs {
        f(&mut v.state);
    }
    for op in &mut g.operations {
        f(&mut op.output.state);
    }
    for cap in &mut g.capabilities {
        for s in &mut cap.inputs {
            f(s);
        }
        f(&mut cap.output);
    }
}

#[test]
fn level_trace_bridge_binds_explicit_carrier_domain_levels_and_endpoint() {
    let p = level_problem(true);
    let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
    let (g, t) = execution_graph(true);
    let b = binding();
    for field in 0..5 {
        let mut changed = g.clone();
        match field {
            0 => map_states(&mut changed, |s| s.scheme = "other-scheme".into()),
            1 => map_states(&mut changed, |s| s.key_domain = "other-domain".into()),
            2 => {
                // Same valid trace endpoints after a different legal first
                // computation, but its input level is not the planner's input.
                changed.inputs[0].state.level = 1;
                changed.inputs[1].state.level = 1;
                changed.capabilities[0].inputs[0].level = 1;
                if let PublicCiphertextExecution::Ckks { compute, depth, .. } =
                    &mut changed.capabilities[0].execution
                {
                    compute.input_level = 1;
                    *depth = 0;
                }
            }
            3 => changed.outputs.push("parent-alias".into()),
            _ => {
                // Existing atomic hybrid gate accepts externally declared
                // recovery endpoints; this bridge also binds the carrier domain.
                changed.operations[1].output.state.key_domain = "other-recovery-domain".into();
                changed.capabilities[1].output.key_domain = "other-recovery-domain".into();
            }
        }
        assert!(validate_public_ciphertext_trace(&changed, &t, limits().trace).is_ok());
        assert!(check(&p, &s, &changed, &t, &b).is_err());
    }
    for field in 0..5 {
        let mut bad = b.clone();
        match field {
            0 => bad.input_value = "parent-alias".into(),
            1 => bad.scheme = "other-scheme".into(),
            2 => bad.key_domain = "other-domain".into(),
            3 => bad.kernel_identities.reverse(),
            _ => bad.kernel_identities.clear(),
        }
        assert!(check(&p, &s, &g, &t, &bad).is_err());
    }
}

#[test]
fn level_trace_bridge_reuses_retained_alias_immutable_owner_and_unknown_capability_checks() {
    let p = level_problem(true);
    let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
    let (g, t) = execution_graph(true);
    let b = binding();
    let mut release = t.clone();
    release[0].release.push("parent-alias".into());
    assert!(check(&p, &s, &g, &release, &b)
        .unwrap_err()
        .contains("retained parent"));
    let mut early = t.clone();
    early[0].release.push("a".into());
    assert!(check(&p, &s, &g, &early, &b)
        .unwrap_err()
        .contains("unavailable parent"));
    for field in 0..9 {
        let mut bad = g.clone();
        let mut trace = t.clone();
        match field {
            0 => {
                bad.operations[0].output.owner = "input-backing".into();
                bad.owners.remove(1);
            }
            1 => bad.inputs[1].state.key_domain = "inconsistent-alias-domain".into(),
            2 => bad.capabilities[1].recovery_overlap_bytes = None,
            3 => bad.capabilities[1].scratch_bytes = None,
            4 => bad.owners[0].bytes = None,
            5 => bad.capabilities[0].error_assumption = None,
            6 => trace[1].capability = "missing-capability".into(),
            7 => bad.inputs[0].state.key_domain.clear(),
            _ => {
                let mut unused = bad.capabilities[0].clone();
                unused.identity = "unused-unknown-capability".into();
                unused.scratch_bytes = None;
                bad.capabilities.push(unused);
            }
        }
        assert!(check(&p, &s, &bad, &trace, &b).is_err());
    }
}

#[test]
fn level_trace_bridge_requires_complete_recovery_and_exact_refresh_execution_type() {
    for hybrid in [false, true] {
        let p = level_problem(hybrid);
        let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
        let (g, t) = execution_graph(hybrid);
        let b = binding();
        let mut wrong_type = g.clone();
        wrong_type.capabilities[1].execution = if hybrid {
            PublicCiphertextExecution::Ckks {
                compute: PublicLevelCompute {
                    input_level: 1,
                    cost: charge("other-compute", 1),
                },
                depth: 1,
                transition: refresh(0, 5),
            }
        } else {
            PublicCiphertextExecution::Hybrid {
                round_trip: trip(10), // 1 + 10 + 3 equals CKKS 6 + 8.
            }
        };
        assert_eq!(
            validate_public_ciphertext_trace(&wrong_type, &t, limits().trace)
                .unwrap()
                .total_cost,
            s.ledger.total_cost
        );
        assert!(check(&p, &s, &wrong_type, &t, &b).is_err());
        let mut missing = g.clone();
        match &mut missing.capabilities[1].execution {
            PublicCiphertextExecution::Hybrid { round_trip } => round_trip.recover.units = None,
            PublicCiphertextExecution::Ckks { transition, .. } => {
                transition.kind = PublicLevelTransitionKind::Drop
            }
            _ => unreachable!(),
        }
        assert!(check(&p, &s, &missing, &t, &b).is_err());
        let mut missing_type = serde_json::to_value(&g.capabilities[1]).unwrap();
        if hybrid {
            missing_type["execution"]["hybrid"]["round_trip"]
                .as_object_mut()
                .unwrap()
                .remove("recover");
        } else {
            missing_type["execution"]["ckks"]["transition"]
                .as_object_mut()
                .unwrap()
                .remove("kind");
        }
        assert!(serde_json::from_value::<PublicCiphertextCapability>(missing_type).is_err());
    }
}

#[test]
fn level_trace_bridge_read_only_alias_is_not_planned_computation() {
    let mut p = level_problem(true);
    p.operations.truncate(1);
    p.operations[0].depth = 0;
    p.operations[0].compute.insert(
        0,
        PublicLevelCompute {
            input_level: 0,
            cost: charge("first-at-zero", 2),
        },
    );
    p.operations[0].transitions = vec![PublicLevelTransition {
        kind: PublicLevelTransitionKind::Preserve,
        from_level: 2,
        to_level: 2,
        cost: charge("preserve-two", 0),
    }];
    let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
    let (mut g, mut t) = execution_graph(true);
    g.operations.truncate(1);
    g.capabilities.truncate(1);
    g.owners.truncate(1);
    g.operations[0].output = value("a", "input-backing", 2);
    g.capabilities[0].output = state(2);
    g.capabilities[0].execution = PublicCiphertextExecution::View {
        cost: charge("first-at-two", 3),
    };
    g.outputs = vec!["a".into()];
    t.truncate(1);
    let mut b = binding();
    b.kernel_identities.truncate(1);
    let view = validate_public_ciphertext_trace(&g, &t, limits().trace).unwrap();
    assert_eq!(view.total_cost, s.ledger.total_cost);
    assert_eq!(view.final_live_backing_bytes, 40);
    assert!(check(&p, &s, &g, &t, &b)
        .unwrap_err()
        .contains("selected execution/cost/transition"));
}

#[test]
fn level_trace_bridge_capacity_boundaries_and_arithmetic_overflow_reject() {
    let p = level_problem(true);
    let s = solve_public_encrypted_levels(&p, limits().levels).unwrap();
    let (g, t) = execution_graph(true);
    let b = binding();
    let summary = check(&p, &s, &g, &t, &b).unwrap().trace_summary();
    let mut exact = limits();
    exact.trace = PublicCiphertextTraceCapacity {
        max_records: (summary.buffer_bytes - 1024) / 256,
        max_work: summary.work_bound,
        max_buffer_bytes: summary.buffer_bytes,
        max_live_owner_bytes: summary.peak_live_owner_bytes,
    };
    assert!(validate_public_encrypted_level_trace(&p, &s, &g, &t, &b, exact).is_ok());
    for field in 0..10 {
        let mut short = exact;
        match field {
            0 => short.trace.max_records -= 1,
            1 => short.trace.max_work -= 1,
            2 => short.trace.max_buffer_bytes -= 1,
            3 => short.trace.max_live_owner_bytes -= 1,
            4 => short.levels.max_operations = 0,
            5 => short.levels.max_levels = 0,
            6 => short.levels.max_table_rows = 0,
            7 => short.levels.max_vertices = 0,
            8 => short.levels.max_work = 0,
            _ => short.levels.max_buffer_bytes = 0,
        }
        assert!(validate_public_encrypted_level_trace(&p, &s, &g, &t, &b, short).is_err());
    }
    let mut overflowing = p.clone();
    overflowing.operations[1].hybrid[0].recover.units = Some(u64::MAX);
    assert!(check(&overflowing, &s, &g, &t, &b)
        .unwrap_err()
        .contains("overflow"));
    let mut overflowing = g.clone();
    overflowing.inputs[0].state.shape = vec![u64::MAX, 2];
    assert!(check(&p, &s, &overflowing, &t, &b)
        .unwrap_err()
        .contains("overflow"));
    let mut overflowing = g.clone();
    overflowing.capabilities[1].scratch_bytes = Some(u64::MAX);
    assert!(check(&p, &s, &overflowing, &t, &b)
        .unwrap_err()
        .contains("overflow"));
}

/// Literal selected path, independent of both production validators and solver.
/// Only the existing canonical identity function computes its two digests.
fn literal_bridge_solution(hybrid: bool) -> PublicLevelSolution {
    let ledger = PublicLevelCostLedger {
        problem_digest: canonical_digest(ENCRYPTED_LEVEL_PROBLEM_DOMAIN, &level_problem(hybrid)),
        steps: vec![
            PublicLevelPathStep {
                operation_identity: "first".into(),
                input_level: 2,
                output_level: 1,
                choice: PublicLevelChoice::Ckks {
                    compute_row: 1,
                    transition_row: 0,
                },
                charges: vec![charge("first-at-two", 3), charge("preserve-one", 0)],
                total_cost: 3,
            },
            PublicLevelPathStep {
                operation_identity: "second".into(),
                input_level: 1,
                output_level: 2,
                choice: if hybrid {
                    PublicLevelChoice::Hybrid { round_trip_row: 0 }
                } else {
                    PublicLevelChoice::Ckks {
                        compute_row: 0,
                        transition_row: 0,
                    }
                },
                charges: if hybrid {
                    vec![
                        charge("complete-entry", 1),
                        charge("hybrid-evaluate", 2),
                        charge("complete-recovery", 3),
                    ]
                } else {
                    vec![
                        charge("second-at-one", 6),
                        charge("complete-refresh-from-0", 8),
                    ]
                },
                total_cost: if hybrid { 6 } else { 14 },
            },
        ],
        total_cost: if hybrid { 9 } else { 17 },
    };
    PublicLevelSolution {
        ledger_digest: canonical_digest(ENCRYPTED_LEVEL_LEDGER_DOMAIN, &ledger),
        ledger,
    }
}

#[test]
fn level_trace_bridge_literal_oracles_cover_entire_ledger_and_manifest_summary() {
    for hybrid in [false, true] {
        let p = level_problem(hybrid);
        let literal = literal_bridge_solution(hybrid);
        let (g, t) = execution_graph(hybrid);
        let b = binding();
        assert_eq!(
            solve_public_encrypted_levels(&p, limits().levels).unwrap(),
            literal
        );
        let expected = PublicCiphertextTraceSummary {
            total_cost: if hybrid { 9 } else { 17 },
            // Declared backing arithmetic, NOT measured physical memory:
            // initial aliases share 40; first overlaps 40+12; second 40+12+14;
            // scratch/recovery add 5+7 before a's release; final retains 40+14.
            peak_live_backing_bytes: 66,
            peak_live_owner_bytes: 78,
            final_live_backing_bytes: 54,
            // Literal preflight oracle: 134 records; 64*134^3 + 256*134;
            // metadata envelope 256*134 + 1024. Neither comes from a summary.
            work_bound: 154_024_960,
            buffer_bytes: 35_328,
        };
        assert_eq!(
            validate_public_ciphertext_trace(&g, &t, limits().trace).unwrap(),
            expected
        );
        assert_eq!(
            check(&p, &literal, &g, &t, &b).unwrap().trace_summary(),
            expected
        );
    }
}

#[test]
fn level_trace_bridge_literal_work_envelopes_accept_exact_reject_zero_and_one_short() {
    let p = level_problem(true);
    let s = literal_bridge_solution(true);
    let (g, t) = execution_graph(true);
    let b = binding();
    let mut exact = limits();
    // Level oracle: 9 vertices + two 2*2 edge scans + one hybrid + 12^2
    // validation scans = 162. These are existing phase limits, not a full Budget.
    exact.levels.max_operations = 2;
    exact.levels.max_levels = 3;
    exact.levels.max_table_rows = 12;
    exact.levels.max_vertices = 9;
    exact.levels.max_work = 162;
    exact.trace.max_records = 134;
    exact.trace.max_work = 154_024_960;
    exact.trace.max_buffer_bytes = 35_328;
    exact.trace.max_live_owner_bytes = 78;
    assert_eq!(solve_public_encrypted_levels(&p, exact.levels).unwrap(), s);
    assert!(validate_public_ciphertext_trace(&g, &t, exact.trace).is_ok());
    assert!(validate_public_encrypted_level_trace(&p, &s, &g, &t, &b, exact).is_ok());
    for field in 0..5 {
        for zero in [false, true] {
            let mut short = exact;
            match field {
                0 => short.levels.max_work = if zero { 0 } else { 161 },
                1 => short.trace.max_work = if zero { 0 } else { 154_024_959 },
                2 => short.trace.max_buffer_bytes = if zero { 0 } else { 35_327 },
                3 => short.trace.max_live_owner_bytes = if zero { 0 } else { 77 },
                _ => short.trace.max_records = if zero { 0 } else { 133 },
            }
            let error =
                validate_public_encrypted_level_trace(&p, &s, &g, &t, &b, short).unwrap_err();
            if field == 0 {
                assert_eq!(
                    solve_public_encrypted_levels(&p, short.levels).unwrap_err(),
                    error
                );
            } else {
                assert_eq!(
                    validate_public_ciphertext_trace(&g, &t, short.trace).unwrap_err(),
                    error
                );
            }
        }
    }
}

#[test]
fn level_trace_bridge_compensating_cross_node_charges_and_rehashed_ledger_still_reject() {
    let p = level_problem(true);
    let s = literal_bridge_solution(true);
    let (mut g, t) = execution_graph(true);
    let b = binding();
    // Transfer two units from one operation to another. Each external capability
    // remains well typed, and the graph total is still the literal nine units.
    if let PublicCiphertextExecution::Ckks { compute, .. } = &mut g.capabilities[0].execution {
        compute.cost.units = Some(1);
    }
    if let PublicCiphertextExecution::Hybrid { round_trip } = &mut g.capabilities[1].execution {
        round_trip.entry.units = Some(3);
    }
    assert_eq!(
        validate_public_ciphertext_trace(&g, &t, limits().trace)
            .unwrap()
            .total_cost,
        9
    );
    assert!(check(&p, &s, &g, &t, &b)
        .unwrap_err()
        .contains("selected execution/cost/transition"));
    let mut forged = s.clone();
    forged.ledger.steps[0].charges[0].units = Some(1);
    forged.ledger.steps[0].total_cost = 1;
    forged.ledger.steps[1].charges[0].units = Some(3);
    forged.ledger.steps[1].total_cost = 8;
    forged.ledger_digest = canonical_digest(ENCRYPTED_LEVEL_LEDGER_DOMAIN, &forged.ledger);
    assert_eq!(forged.ledger.total_cost, 9);
    assert_eq!(
        forged
            .ledger
            .steps
            .iter()
            .map(|step| step.total_cost)
            .sum::<u64>(),
        9
    );
    assert!(check(&p, &forged, &g, &t, &b)
        .unwrap_err()
        .contains("solution/problem/digest"));
}

#[test]
fn level_trace_bridge_hybrid_endpoint_crossings_are_not_free_domain_or_level_changes() {
    let p = level_problem(true);
    let s = literal_bridge_solution(true);
    let (g, t) = execution_graph(true);
    let b = binding();
    for field in 0..3 {
        let mut crossing = g.clone();
        match field {
            0 => {
                crossing.operations[1].output.state.key_domain = "other-domain".into();
                crossing.capabilities[1].output.key_domain = "other-domain".into();
            }
            1 => {
                crossing.operations[1].output.state.scheme = "other-scheme".into();
                crossing.capabilities[1].output.scheme = "other-scheme".into();
            }
            _ => {
                crossing.operations[1].output.state.level = 1;
                crossing.capabilities[1].output.level = 1;
                if let PublicCiphertextExecution::Hybrid { round_trip } =
                    &mut crossing.capabilities[1].execution
                {
                    round_trip.output_level = 1;
                }
            }
        }
        // Shared SSA accepts internally consistent external atomic contracts.
        // The bridge must additionally reject their departure from the planner.
        assert_eq!(
            validate_public_ciphertext_trace(&crossing, &t, limits().trace)
                .unwrap()
                .total_cost,
            9
        );
        assert_eq!(
            check(&p, &s, &crossing, &t, &b).unwrap_err(),
            "level trace carrier level/scheme/key-domain mismatch"
        );
    }
    for field in 0..3 {
        let mut mismatch = g.clone();
        match field {
            0 => mismatch.capabilities[1].inputs[0].key_domain = "other-domain".into(),
            1 => mismatch.capabilities[1].inputs[0].scheme = "other-scheme".into(),
            _ => mismatch.capabilities[1].inputs[0].level = 0,
        }
        let existing = validate_public_ciphertext_trace(&mismatch, &t, limits().trace).unwrap_err();
        assert_eq!(check(&p, &s, &mismatch, &t, &b).unwrap_err(), existing);
    }
}

#[test]
fn level_trace_bridge_forked_parent_shared_ssa_accepts_but_serial_binding_never_flattens() {
    let mut p = level_problem(true);
    p.operations.push(PublicLevelOperation {
        identity: "fork".into(),
        predecessors: vec!["second".into()],
        primitive: PublicPrimitive::Arithmetic,
        depth: 0,
        compute: (0..=2)
            .map(|level| PublicLevelCompute {
                input_level: level,
                cost: charge("fork-compute", 4),
            })
            .collect(),
        transitions: vec![PublicLevelTransition {
            kind: PublicLevelTransitionKind::Preserve,
            from_level: 2,
            to_level: 2,
            cost: charge("fork-preserve", 0),
        }],
        hybrid: vec![],
    });
    let mut s = literal_bridge_solution(true);
    s.ledger.problem_digest = canonical_digest(ENCRYPTED_LEVEL_PROBLEM_DOMAIN, &p);
    s.ledger.steps.push(PublicLevelPathStep {
        operation_identity: "fork".into(),
        input_level: 2,
        output_level: 2,
        choice: PublicLevelChoice::Ckks {
            compute_row: 2,
            transition_row: 0,
        },
        charges: vec![charge("fork-compute", 4), charge("fork-preserve", 0)],
        total_cost: 4,
    });
    s.ledger.total_cost = 13;
    s.ledger_digest = canonical_digest(ENCRYPTED_LEVEL_LEDGER_DOMAIN, &s.ledger);
    assert_eq!(
        solve_public_encrypted_levels(&p, limits().levels).unwrap(),
        s
    );
    let (mut g, mut t) = execution_graph(true);
    let mut b = binding();
    b.kernel_identities.push("fork-kernel".into());
    g.owners.push(PublicCiphertextOwner {
        identity: "fork-backing".into(),
        bytes: Some(9),
    });
    g.operations.push(PublicCiphertextGraphOperation {
        identity: "fork".into(),
        kernel_identity: "fork-kernel".into(),
        primitive: PublicPrimitive::Arithmetic,
        // Actual retained parent fork, not the serial planner's previous b.
        inputs: vec!["parent-alias".into()],
        output: value("fork-value", "fork-backing", 2),
    });
    g.capabilities.push(PublicCiphertextCapability {
        identity: "fork-cap".into(),
        kernel_identity: "fork-kernel".into(),
        primitive: PublicPrimitive::Arithmetic,
        inputs: vec![state(2)],
        output: state(2),
        join: PublicCiphertextJoin::Unary,
        execution: PublicCiphertextExecution::Ckks {
            compute: PublicLevelCompute {
                input_level: 2,
                cost: charge("fork-compute", 4),
            },
            depth: 0,
            transition: PublicLevelTransition {
                kind: PublicLevelTransitionKind::Preserve,
                from_level: 2,
                to_level: 2,
                cost: charge("fork-preserve", 0),
            },
        },
        error_assumption: Some("external-fork-error-assumption".into()),
        scratch_bytes: Some(0),
        recovery_overlap_bytes: Some(0),
    });
    t.push(PublicCiphertextTraceStep {
        operation: "fork".into(),
        capability: "fork-cap".into(),
        release: vec!["b".into()],
    });
    g.outputs = vec!["fork-value".into()];
    let shared = validate_public_ciphertext_trace(&g, &t, limits().trace).unwrap();
    assert_eq!(
        (
            shared.total_cost,
            shared.peak_live_backing_bytes,
            shared.peak_live_owner_bytes,
            shared.final_live_backing_bytes
        ),
        (13, 66, 78, 49)
    );
    assert_eq!(
        check(&p, &s, &g, &t, &b).unwrap_err(),
        "level trace operator/kernel/non-chain mismatch"
    );
    // An actual chain with the same declared types and costs IS accepted. No
    // operator or method-specific numerical policy is needed to distinguish it.
    let mut chain = g.clone();
    chain.operations[2].inputs[0] = "b".into();
    assert_eq!(
        check(&p, &s, &chain, &t, &b)
            .unwrap()
            .trace_summary()
            .total_cost,
        13
    );
    let mut retired = t.clone();
    retired[1].release.push("parent-alias".into());
    assert!(
        validate_public_ciphertext_trace(&g, &retired, limits().trace)
            .unwrap_err()
            .contains("retained parent")
    );
    assert!(check(&p, &s, &chain, &retired, &b)
        .unwrap_err()
        .contains("retained parent"));
    let mut dead_handle = g.clone();
    dead_handle.operations[2].inputs[0] = "input".into();
    assert!(
        validate_public_ciphertext_trace(&dead_handle, &t, limits().trace)
            .unwrap_err()
            .contains("unavailable parent")
    );
    // A view may keep the FULL parent backing without allocating the declared
    // nine-byte output. This remains valid SSA, not a legal serial computation.
    let mut view = g.clone();
    view.owners.pop();
    view.operations[2].output.owner = "input-backing".into();
    view.capabilities[2].execution = PublicCiphertextExecution::View {
        cost: charge("fork-compute", 4),
    };
    assert_eq!(
        validate_public_ciphertext_trace(&view, &t, limits().trace)
            .unwrap()
            .final_live_backing_bytes,
        40
    );
    assert!(check(&p, &s, &view, &t, &b).is_err());
}

#[test]
fn level_trace_bridge_known_zero_exact_u64_cost_and_one_unit_overflow_are_distinct() {
    // No max-cost admission field exists in this API. Test its actual checked
    // integer range, not an invented budget or a physical resource policy.
    for total in [0, u64::MAX] {
        let mut p = level_problem(true);
        p.operations[0].transitions.truncate(1);
        for compute in &mut p.operations[0].compute {
            compute.cost.units = Some(0);
        }
        p.operations[1].depth = 3;
        p.operations[1].compute.clear();
        p.operations[1].transitions.clear();
        p.operations[1].hybrid[0].entry.units = Some(0);
        p.operations[1].hybrid[0].evaluate.units = Some(total);
        p.operations[1].hybrid[0].recover.units = Some(0);
        let (mut g, t) = execution_graph(true);
        let b = binding();
        if let PublicCiphertextExecution::Ckks { compute, .. } = &mut g.capabilities[0].execution {
            compute.cost.units = Some(0);
        }
        if let PublicCiphertextExecution::Hybrid { round_trip } = &mut g.capabilities[1].execution {
            round_trip.entry.units = Some(0);
            round_trip.evaluate.units = Some(total);
            round_trip.recover.units = Some(0);
        }
        let mut literal = literal_bridge_solution(true);
        literal.ledger.problem_digest = canonical_digest(ENCRYPTED_LEVEL_PROBLEM_DOMAIN, &p);
        literal.ledger.steps[0].charges[0].units = Some(0);
        literal.ledger.steps[0].total_cost = 0;
        literal.ledger.steps[1].charges[0].units = Some(0);
        literal.ledger.steps[1].charges[1].units = Some(total);
        literal.ledger.steps[1].charges[2].units = Some(0);
        literal.ledger.steps[1].total_cost = total;
        literal.ledger.total_cost = total;
        literal.ledger_digest = canonical_digest(ENCRYPTED_LEVEL_LEDGER_DOMAIN, &literal.ledger);
        assert_eq!(
            solve_public_encrypted_levels(&p, limits().levels).unwrap(),
            literal
        );
        assert_eq!(
            validate_public_ciphertext_trace(&g, &t, limits().trace)
                .unwrap()
                .total_cost,
            total
        );
        assert_eq!(
            check(&p, &literal, &g, &t, &b)
                .unwrap()
                .trace_summary()
                .total_cost,
            total
        );
        let mut unknown = p.clone();
        unknown.operations[1].hybrid[0].recover.units = None;
        assert!(check(&unknown, &literal, &g, &t, &b)
            .unwrap_err()
            .contains("unknown public cost/capability"));
        if total == u64::MAX {
            // Both atomic-edge and whole-path overflows reject; a smaller quoted
            // ledger cannot mask them. These are literal one-unit boundaries.
            for edge_overflow in [false, true] {
                let mut overflow_p = p.clone();
                let mut overflow_g = g.clone();
                if edge_overflow {
                    overflow_p.operations[1].hybrid[0].recover.units = Some(1);
                    if let PublicCiphertextExecution::Hybrid { round_trip } =
                        &mut overflow_g.capabilities[1].execution
                    {
                        round_trip.recover.units = Some(1);
                    }
                } else {
                    overflow_p.operations[0].compute[1].cost.units = Some(1);
                    if let PublicCiphertextExecution::Ckks { compute, .. } =
                        &mut overflow_g.capabilities[0].execution
                    {
                        compute.cost.units = Some(1);
                    }
                }
                assert_eq!(
                    solve_public_encrypted_levels(&overflow_p, limits().levels).unwrap_err(),
                    "public cost overflow"
                );
                assert_eq!(
                    validate_public_ciphertext_trace(&overflow_g, &t, limits().trace).unwrap_err(),
                    "public cost overflow"
                );
                assert_eq!(
                    check(&overflow_p, &literal, &overflow_g, &t, &b).unwrap_err(),
                    "public cost overflow"
                );
            }
        }
    }
}
