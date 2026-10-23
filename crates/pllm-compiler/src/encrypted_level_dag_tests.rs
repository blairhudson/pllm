use super::*;

fn capacity(owners: u64) -> PublicEncryptedDagCapacity {
    PublicEncryptedDagCapacity {
        max_operations: 8,
        max_states: 128,
        max_frontier: 16,
        max_work: usize::MAX,
        max_buffer_bytes: 16 << 20,
        trace: PublicCiphertextTraceCapacity {
            max_records: 1024,
            max_work: usize::MAX,
            max_buffer_bytes: 1 << 20,
            max_live_owner_bytes: owners,
        },
    }
}

fn cost(id: &str, units: u64) -> PublicCost {
    PublicCost {
        identity: id.into(),
        units: Some(units),
    }
}

fn state() -> PublicCiphertextState {
    PublicCiphertextState {
        scheme: "synthetic-carrier-not-crypto".into(),
        key_domain: "external-client-domain".into(),
        layout: "synthetic-layout".into(),
        scale: "synthetic-scale".into(),
        level: 2,
        shape: vec![4],
        error: PublicCiphertextError {
            model_identity: "external-error-model".into(),
            bound: Some(1),
        },
    }
}

fn value(id: &str, owner: &str) -> PublicCiphertextValue {
    PublicCiphertextValue {
        identity: id.into(),
        state: state(),
        owner: owner.into(),
    }
}

fn preserve() -> PublicLevelTransition {
    PublicLevelTransition {
        kind: PublicLevelTransitionKind::Preserve,
        from_level: 2,
        to_level: 2,
        cost: cost("explicit-preserve", 0),
    }
}

fn cap(
    id: &str,
    kernel: &str,
    arity: usize,
    units: u64,
    scratch: u64,
) -> PublicCiphertextCapability {
    PublicCiphertextCapability {
        identity: id.into(),
        kernel_identity: kernel.into(),
        primitive: PublicPrimitive::Arithmetic,
        inputs: vec![state(); arity],
        output: state(),
        join: if arity == 1 {
            PublicCiphertextJoin::Unary
        } else {
            PublicCiphertextJoin::Aligned
        },
        execution: PublicCiphertextExecution::Ckks {
            compute: PublicLevelCompute {
                input_level: 2,
                cost: cost(id, units),
            },
            depth: 0,
            transition: preserve(),
        },
        error_assumption: Some("external-error-assumption-not-a-proof".into()),
        scratch_bytes: Some(scratch),
        recovery_overlap_bytes: Some(0),
    }
}

/// Synthetic fixed SSA source declarations, not any paper/backend parameters.
fn fork_join(a_bytes: u64, b_bytes: u64, fast_scratch: u64) -> PublicCiphertextGraph {
    let mut join = cap("join-cap", "join-kernel", 2, 5, 2);
    join.recovery_overlap_bytes = Some(3);
    PublicCiphertextGraph {
        cost_model_identity: "synthetic-dag-cost-model".into(),
        cost_unit: "synthetic-unit".into(),
        maximum_level: 2,
        owners: [
            ("x-owner", 40),
            ("a-owner", a_bytes),
            ("b-owner", b_bytes),
            ("out-owner", 11),
            ("persistent-owner", 5),
        ]
        .into_iter()
        .map(|(id, bytes)| PublicCiphertextOwner {
            identity: id.into(),
            bytes: Some(bytes),
        })
        .collect(),
        persistent_owners: vec!["persistent-owner".into()],
        inputs: vec![value("x", "x-owner")],
        operations: vec![
            PublicCiphertextGraphOperation {
                identity: "a-op".into(),
                kernel_identity: "a-kernel".into(),
                primitive: PublicPrimitive::Arithmetic,
                inputs: vec!["x".into()],
                output: value("a", "a-owner"),
            },
            PublicCiphertextGraphOperation {
                identity: "b-op".into(),
                kernel_identity: "b-kernel".into(),
                primitive: PublicPrimitive::Arithmetic,
                inputs: vec!["x".into()],
                output: value("b", "b-owner"),
            },
            PublicCiphertextGraphOperation {
                identity: "join-op".into(),
                kernel_identity: "join-kernel".into(),
                primitive: PublicPrimitive::Arithmetic,
                inputs: vec!["a".into(), "b".into()],
                output: value("out", "out-owner"),
            },
        ],
        capabilities: vec![
            cap("a-fast", "a-kernel", 1, 2, fast_scratch),
            cap("a-small", "a-kernel", 1, 4, 0),
            cap("b-cap", "b-kernel", 1, 3, 0),
            join,
        ],
        outputs: vec!["out".into()],
        retained_inputs: vec!["x".into()],
    }
}

#[derive(Debug, Eq, PartialEq)]
struct LiteralPath {
    cost: u64,
    selections: Vec<(usize, usize)>,
    peak_backing: u64,
    peak_owners: u64,
    final_backing: u64,
}

/// Independent exhaustive integer oracle: enumerate ALL six permutations and
/// both A choices, then apply the literal fork/join dependency matrix. Uses no
/// production planner, readiness, release, cost or trace-validation helpers.
fn enumerate_literal(a: u64, b: u64, scratch: u64, ceiling: u64) -> Option<LiteralPath> {
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut paths = Vec::new();
    for order in permutations {
        for fast in [true, false] {
            let mut executed = 0u8;
            let mut backing = 45; // retained x=40 + declared persistent owner=5
            let mut peak_backing = backing;
            let mut peak_owners = backing;
            let mut units = 0;
            let mut selections = Vec::new();
            for op in order {
                if op == 2 && executed & 3 != 3 {
                    break;
                }
                let (bytes, charge, temporary, cap_index) = match op {
                    0 => (
                        a,
                        if fast { 2 } else { 4 },
                        if fast { scratch } else { 0 },
                        if fast { 0 } else { 1 },
                    ),
                    1 => (b, 3, 0, 2),
                    _ => (11, 5, 5, 3),
                };
                backing += bytes;
                peak_backing = peak_backing.max(backing);
                peak_owners = peak_owners.max(backing + temporary);
                units += charge;
                executed |= 1 << op;
                selections.push((op, cap_index));
                if op == 2 {
                    backing -= a + b;
                }
            }
            if executed == 7 && peak_owners <= ceiling {
                paths.push(LiteralPath {
                    cost: units,
                    selections,
                    peak_backing,
                    peak_owners,
                    final_backing: backing,
                });
            }
        }
    }
    paths
        .into_iter()
        .min_by(|a, b| a.cost.cmp(&b.cost).then(a.selections.cmp(&b.selections)))
}

#[test]
fn encrypted_dag_independent_exhaustive_fork_join_cost_and_owner_oracles() {
    for a_bytes in [1, 30, 60] {
        for b_bytes in [1, 7, 40] {
            for scratch in [0, 20, 60] {
                for ceiling in [55, 68, 97, 98, 128, 135, 142, 256] {
                    let g = fork_join(a_bytes, b_bytes, scratch);
                    match enumerate_literal(a_bytes, b_bytes, scratch, ceiling) {
                        Some(expected) => {
                            let solved = solve_public_encrypted_dag(&g, capacity(ceiling)).unwrap();
                            let actual = LiteralPath {
                                cost: solved.ledger.total_cost,
                                selections: solved
                                    .ledger
                                    .selections
                                    .iter()
                                    .map(|s| (s.operation_index, s.capability_index))
                                    .collect(),
                                peak_backing: solved.trace_summary.peak_live_backing_bytes,
                                peak_owners: solved.trace_summary.peak_live_owner_bytes,
                                final_backing: solved.trace_summary.final_live_backing_bytes,
                            };
                            assert_eq!(
                                actual, expected,
                                "a={a_bytes}, b={b_bytes}, scratch={scratch}, ceiling={ceiling}"
                            );
                        }
                        None => assert!(solve_public_encrypted_dag(&g, capacity(ceiling))
                            .unwrap_err()
                            .contains("no feasible typed DAG trace")),
                    }
                }
            }
        }
    }
}

#[test]
fn encrypted_dag_literal_trace_and_finite_frontier_work_counts() {
    let g = fork_join(30, 7, 60);
    let c = capacity(135);
    let solution = solve_public_encrypted_dag(&g, c).unwrap();
    assert_eq!(
        solution.ledger.selections,
        vec![
            PublicEncryptedDagSelection {
                operation_index: 0,
                capability_index: 0
            },
            PublicEncryptedDagSelection {
                operation_index: 1,
                capability_index: 2
            },
            PublicEncryptedDagSelection {
                operation_index: 2,
                capability_index: 3
            },
        ]
    );
    assert_eq!(
        solution.ledger.trace,
        vec![
            PublicCiphertextTraceStep {
                operation: "a-op".into(),
                capability: "a-fast".into(),
                release: vec![]
            },
            PublicCiphertextTraceStep {
                operation: "b-op".into(),
                capability: "b-cap".into(),
                release: vec![]
            },
            PublicCiphertextTraceStep {
                operation: "join-op".into(),
                capability: "join-cap".into(),
                release: vec!["a".into(), "b".into()]
            },
        ]
    );
    assert_eq!(
        solution.trace_summary,
        PublicCiphertextTraceSummary {
            total_cost: 10,
            peak_live_backing_bytes: 93,
            peak_live_owner_bytes: 135,
            final_live_backing_bytes: 56,
            work_bound: 719_380_480,
            buffer_bytes: 58_368,
        }
    );
    // Root + 3 depth-one + 4 depth-two + 4 terminal prefixes. All candidates
    // remain in the search even after a winner exists; graph/trace records R=226.
    assert_eq!(solution.search_summary.states_generated, 12);
    assert_eq!(solution.search_summary.states_expanded, 12);
    assert_eq!(solution.search_summary.peak_frontier, 3);
    assert_eq!(solution.search_summary.work_quantum, 1_481_229_312);
    assert_eq!(solution.search_summary.work, 14 * 1_481_229_312usize);
    assert_eq!(
        validate_public_ciphertext_trace(&g, &solution.ledger.trace, c.trace).unwrap(),
        solution.trace_summary
    );
    let checked = validate_public_encrypted_dag_solution(&g, &solution, c).unwrap();
    assert!(std::ptr::eq(checked.graph(), &g));
    assert!(std::ptr::eq(checked.solution(), &solution));
}

#[test]
fn encrypted_dag_costlier_capability_and_equal_cost_schedule_owners_remain_distinct() {
    let g = fork_join(30, 7, 60);
    let low = solve_public_encrypted_dag(&g, capacity(98)).unwrap();
    assert_eq!(low.ledger.total_cost, 12);
    assert_eq!(low.ledger.selections[0].capability_index, 1);
    assert_eq!(low.trace_summary.peak_live_owner_bytes, 98);
    let high = solve_public_encrypted_dag(&g, capacity(135)).unwrap();
    assert_eq!(high.ledger.total_cost, 10);
    let mut reversed = high.ledger.trace.clone();
    reversed.swap(0, 1);
    assert!(validate_public_ciphertext_trace(&g, &reversed, capacity(141).trace).is_err());
    let reverse_summary =
        validate_public_ciphertext_trace(&g, &reversed, capacity(142).trace).unwrap();
    assert_eq!(reverse_summary.total_cost, 10);
    assert_eq!(reverse_summary.peak_live_owner_bytes, 142);
    // Even when both orders fit, stable table-order ties bind the first order.
    let original = solve_public_encrypted_dag(&g, capacity(142)).unwrap();
    let mut forged = original.clone();
    forged.ledger.trace = reversed;
    forged.ledger.selections.swap(0, 1);
    forged.trace_summary = reverse_summary;
    forged.ledger_digest = canonical_digest(ENCRYPTED_DAG_LEDGER_DOMAIN, &forged.ledger);
    assert!(validate_public_encrypted_dag_solution(&g, &forged, capacity(142)).is_err());
}

#[test]
fn encrypted_dag_zero_exact_short_state_frontier_work_and_metadata_ceilings() {
    let g = fork_join(30, 7, 60);
    let mut exact = capacity(135);
    exact.max_states = 12;
    exact.max_frontier = 3;
    exact.max_work = 14 * 1_481_229_312usize;
    // Preflight R=226; metadata = 16384*(226+6*3+8) + 128*(3+2)*(3+1).
    exact.max_buffer_bytes = 4_131_328;
    assert!(solve_public_encrypted_dag(&g, exact).is_ok());
    for field in 0..5 {
        for zero in [false, true] {
            let mut short = exact;
            match field {
                0 => short.max_states = if zero { 0 } else { 11 },
                1 => short.max_frontier = if zero { 0 } else { 2 },
                2 => short.max_work = if zero { 0 } else { exact.max_work - 1 },
                3 => short.max_buffer_bytes = if zero { 0 } else { exact.max_buffer_bytes - 1 },
                _ => short.trace.max_live_owner_bytes = if zero { 0 } else { 97 },
            }
            assert!(
                solve_public_encrypted_dag(&g, short).is_err(),
                "field {field}, zero {zero}"
            );
        }
    }
    // A tighter capacity cannot make an already found cost-10 incumbent count as
    // an optimum: 11 states and finalization-one-short both reject the whole run.
    let mut fewer = exact;
    fewer.max_states = 11;
    assert!(solve_public_encrypted_dag(&g, fewer)
        .unwrap_err()
        .contains("optimum not established"));
    let mut overflow = exact;
    overflow.max_frontier = usize::MAX;
    assert!(solve_public_encrypted_dag(&g, overflow)
        .unwrap_err()
        .contains("overflow"));
    assert!(work_quantum(usize::MAX).is_err());
    let mut used = usize::MAX;
    assert!(charge_work(&mut used, 1, usize::MAX)
        .unwrap_err()
        .contains("overflow"));
}

#[test]
fn encrypted_dag_full_parent_backing_lives_through_scalar_view_and_join() {
    let mut g = fork_join(30, 7, 60);
    g.capabilities.remove(1);
    g.owners.remove(1);
    g.retained_inputs.clear();
    g.operations[0].output.owner = "x-owner".into();
    g.operations[0].output.state.shape = vec![1];
    g.capabilities[0].output.shape = vec![1];
    g.capabilities[0].scratch_bytes = Some(0);
    g.capabilities[0].execution = PublicCiphertextExecution::View {
        cost: cost("declared-view", 2),
    };
    g.capabilities[2].inputs[0].shape = vec![1];
    g.capabilities[2].join = PublicCiphertextJoin::ScalarBroadcast { vector_input: 1 };
    let solution = solve_public_encrypted_dag(&g, capacity(68)).unwrap();
    assert_eq!(solution.ledger.total_cost, 10);
    assert_eq!(solution.trace_summary.peak_live_backing_bytes, 63);
    assert_eq!(solution.trace_summary.peak_live_owner_bytes, 68);
    assert_eq!(solution.trace_summary.final_live_backing_bytes, 16);
    assert_eq!(solution.ledger.trace[1].release, vec!["x"]);
    assert!(solve_public_encrypted_dag(&g, capacity(67)).is_err());
    let mut early = solution.ledger.trace.clone();
    early[0].release.push("x".into());
    assert!(
        validate_public_ciphertext_trace(&g, &early, capacity(68).trace)
            .unwrap_err()
            .contains("unavailable parent")
    );
    // Changing the view into an ordinary compute cannot reuse an immutable input
    // backing, even after the original input handle has been retired.
    g.capabilities[0].execution = PublicCiphertextExecution::Ckks {
        compute: PublicLevelCompute {
            input_level: 2,
            cost: cost("declared-copy", 2),
        },
        depth: 0,
        transition: preserve(),
    };
    assert!(solve_public_encrypted_dag(&g, capacity(1000)).is_err());
}

#[test]
fn encrypted_dag_digest_indices_owner_identity_typed_costs_and_order_are_not_totals() {
    let g = fork_join(30, 7, 60);
    let c = capacity(135);
    let s = solve_public_encrypted_dag(&g, c).unwrap();
    for field in 0..8 {
        let mut forged = s.clone();
        match field {
            0 => forged.ledger.problem_digest = Digest::from_sha256([0; 32]),
            1 => forged.ledger.selections[0].operation_index = usize::MAX,
            2 => forged.ledger.selections[0].capability_index = usize::MAX,
            3 => forged.ledger.trace[0].capability = "a-small".into(),
            4 => forged.ledger.trace[0].release.push("x".into()),
            5 => forged.trace_summary.peak_live_owner_bytes -= 1,
            6 => forged.search_summary.states_expanded -= 1,
            _ => forged.ledger.selections.swap(0, 1),
        }
        forged.ledger_digest = canonical_digest(ENCRYPTED_DAG_LEDGER_DOMAIN, &forged.ledger);
        assert!(validate_public_encrypted_dag_solution(&g, &forged, c).is_err());
    }
    let mut renamed = g.clone();
    renamed.owners[1].identity = "different-a-owner".into();
    renamed.operations[0].output.owner = "different-a-owner".into();
    let new = solve_public_encrypted_dag(&renamed, c).unwrap();
    assert_eq!(new.ledger.total_cost, 10);
    assert_ne!(new.ledger.problem_digest, s.ledger.problem_digest);
    assert!(validate_public_encrypted_dag_solution(&renamed, &s, c).is_err());
    let mut compensated = g.clone();
    if let PublicCiphertextExecution::Ckks { compute, .. } =
        &mut compensated.capabilities[0].execution
    {
        compute.cost.units = Some(1);
    }
    if let PublicCiphertextExecution::Ckks { compute, .. } =
        &mut compensated.capabilities[2].execution
    {
        compute.cost.units = Some(4);
    }
    let changed = solve_public_encrypted_dag(&compensated, c).unwrap();
    assert_eq!(changed.ledger.total_cost, 10);
    assert!(validate_public_encrypted_dag_solution(&compensated, &s, c).is_err());
}

#[test]
fn encrypted_dag_unknown_unused_costs_keys_errors_owners_and_cycles_explicitly_fail() {
    let g = fork_join(30, 7, 60);
    for field in 0..12 {
        let mut bad = g.clone();
        match field {
            0 => {
                if let PublicCiphertextExecution::Ckks { compute, .. } =
                    &mut bad.capabilities[1].execution
                {
                    compute.cost.units = None;
                }
            }
            1 => bad.capabilities[1].scratch_bytes = None,
            2 => bad.capabilities[1].error_assumption = None,
            3 => bad.owners[0].bytes = None,
            4 => bad.inputs[0].state.key_domain.clear(),
            5 => bad.capabilities[0].inputs[0].error.bound = None,
            6 => bad.operations[1].inputs = vec!["out".into()],
            7 => bad.operations[2].output.identity = "a".into(),
            8 => bad.operations[2].inputs.push("missing".into()),
            9 => {
                bad.operations[1].output.owner = "a-owner".into();
                bad.owners.remove(2);
            }
            10 => bad.inputs[0].state.shape = vec![u64::MAX, 2],
            _ => bad.owners[0].bytes = Some(u64::MAX),
        }
        assert!(
            solve_public_encrypted_dag(&bad, capacity(1000)).is_err(),
            "field {field}"
        );
    }
    for field in 0..3 {
        let mut mismatch = g.clone();
        for cap in &mut mismatch.capabilities {
            if cap.kernel_identity == "a-kernel" {
                match field {
                    0 => cap.inputs[0].key_domain = "other-domain".into(),
                    1 => cap.inputs[0].scheme = "other-scheme".into(),
                    _ => cap.inputs[0].level = 1,
                }
            }
        }
        assert!(solve_public_encrypted_dag(&mismatch, capacity(1000)).is_err());
    }
}

#[test]
fn encrypted_dag_chain_costs_remain_equal_to_existing_serial_solver_without_flattening_forks() {
    // Independently specified unary chain: compute at two, preserve at two.
    let mut g = fork_join(30, 7, 0);
    g.operations.truncate(2);
    g.operations[1].inputs = vec!["a".into()];
    g.capabilities = vec![g.capabilities[0].clone(), g.capabilities[2].clone()];
    g.outputs = vec!["b".into()];
    g.owners.remove(3);
    let p = PublicLevelProblem {
        cost_model_identity: "synthetic-dag-cost-model".into(),
        cost_unit: "synthetic-unit".into(),
        maximum_level: 2,
        input_level: 2,
        output_level: 2,
        operations: [("a-op", "a-fast", 2), ("b-op", "b-cap", 3)]
            .into_iter()
            .enumerate()
            .map(|(i, (id, charge, units))| PublicLevelOperation {
                identity: id.into(),
                predecessors: if i == 0 { vec![] } else { vec!["a-op".into()] },
                primitive: PublicPrimitive::Arithmetic,
                depth: 0,
                compute: (0..=2)
                    .map(|input_level| PublicLevelCompute {
                        input_level,
                        cost: cost(charge, units),
                    })
                    .collect(),
                transitions: vec![preserve()],
                hybrid: vec![],
            })
            .collect(),
    };
    let serial = solve_public_encrypted_levels(
        &p,
        PublicLevelSolverCapacity {
            max_operations: 2,
            max_levels: 3,
            max_table_rows: 32,
            max_vertices: 9,
            max_work: 10_000,
            max_buffer_bytes: 4 << 20,
        },
    )
    .unwrap();
    let dag = solve_public_encrypted_dag(&g, capacity(1000)).unwrap();
    assert_eq!(serial.ledger.total_cost, 5);
    assert_eq!(dag.ledger.total_cost, 5);
    assert_eq!(dag.ledger.trace[0].release, Vec::<String>::new());
    assert_eq!(dag.ledger.trace[1].release, vec!["a"]);
}

#[test]
fn encrypted_dag_existing_atomic_hybrid_and_refresh_rows_bind_without_new_crypto_types() {
    let mut g = fork_join(30, 7, 60);
    g.operations[0].primitive = PublicPrimitive::Inv;
    g.capabilities[0].primitive = PublicPrimitive::Inv;
    g.capabilities[1].primitive = PublicPrimitive::Inv;
    g.capabilities[0].execution = PublicCiphertextExecution::Hybrid {
        round_trip: PublicHybridRoundTrip {
            identity: "external-atomic-round-trip".into(),
            input_level: 2,
            output_level: 2,
            entry: cost("external-entry", 1),
            evaluate: cost("external-evaluate", 0),
            recover: cost("external-recovery", 1),
        },
    };
    let c = capacity(135);
    let solved = solve_public_encrypted_dag(&g, c).unwrap();
    assert_eq!(solved.ledger.total_cost, 10);
    assert_eq!(solved.ledger.trace[0].capability, "a-fast");
    assert_eq!(
        validate_public_ciphertext_trace(&g, &solved.ledger.trace, c.trace)
            .unwrap()
            .total_cost,
        10
    );
    let mut redistributed = g.clone();
    if let PublicCiphertextExecution::Hybrid { round_trip } =
        &mut redistributed.capabilities[0].execution
    {
        round_trip.entry.units = Some(0);
        round_trip.recover.units = Some(2);
    }
    assert_eq!(
        solve_public_encrypted_dag(&redistributed, c)
            .unwrap()
            .ledger
            .total_cost,
        10
    );
    assert!(validate_public_encrypted_dag_solution(&redistributed, &solved, c).is_err());
    if let PublicCiphertextExecution::Hybrid { round_trip } =
        &mut redistributed.capabilities[0].execution
    {
        round_trip.recover.units = None;
    }
    assert!(solve_public_encrypted_dag(&redistributed, c)
        .unwrap_err()
        .contains("unknown"));
    // Explicit synthetic zero charge is not inferred free refresh: the supplied
    // typed row still must compute before a legal residual-one -> output-two edge.
    let mut refresh_graph = fork_join(30, 7, 60);
    if let PublicCiphertextExecution::Ckks {
        depth, transition, ..
    } = &mut refresh_graph.capabilities[0].execution
    {
        *depth = 1;
        transition.kind = PublicLevelTransitionKind::Refresh;
        transition.from_level = 1;
    }
    assert_eq!(
        solve_public_encrypted_dag(&refresh_graph, c)
            .unwrap()
            .ledger
            .total_cost,
        10
    );
    if let PublicCiphertextExecution::Ckks { transition, .. } =
        &mut refresh_graph.capabilities[0].execution
    {
        transition.kind = PublicLevelTransitionKind::Drop;
    }
    assert!(solve_public_encrypted_dag(&refresh_graph, c).is_err());
}

#[test]
fn encrypted_dag_coherent_domain_and_scheme_relabeling_still_changes_bound_problem() {
    let g = fork_join(30, 7, 60);
    let c = capacity(135);
    let solved = solve_public_encrypted_dag(&g, c).unwrap();
    for key in [false, true] {
        let mut changed = g.clone();
        let change = |state: &mut PublicCiphertextState| {
            if key {
                state.key_domain = "other-external-domain".into();
            } else {
                state.scheme = "other-external-scheme".into();
            }
        };
        for input in &mut changed.inputs {
            change(&mut input.state);
        }
        for op in &mut changed.operations {
            change(&mut op.output.state);
        }
        for cap in &mut changed.capabilities {
            change(&mut cap.output);
            for input in &mut cap.inputs {
                change(input);
            }
        }
        let new = solve_public_encrypted_dag(&changed, c).unwrap();
        assert_eq!(new.ledger.total_cost, 10);
        assert_ne!(new.ledger.problem_digest, solved.ledger.problem_digest);
        assert!(validate_public_encrypted_dag_solution(&changed, &solved, c).is_err());
    }
}

#[test]
fn encrypted_dag_unsorted_storage_and_input_aliases_keep_ssa_identity_and_backing() {
    let mut g = fork_join(30, 7, 60);
    g.operations.swap(0, 2); // join stored first, producer A stored last
    let c = capacity(135);
    let solved = solve_public_encrypted_dag(&g, c).unwrap();
    assert_eq!(solved.ledger.total_cost, 10);
    assert_eq!(
        solved
            .ledger
            .selections
            .iter()
            .map(|s| s.operation_index)
            .collect::<Vec<_>>(),
        vec![2, 1, 0]
    );
    assert_eq!(solved.trace_summary.peak_live_owner_bytes, 135);
    let mut alias = fork_join(30, 7, 60);
    alias.inputs.push(value("unused-input-alias", "x-owner"));
    let with_alias = solve_public_encrypted_dag(&alias, c).unwrap();
    assert_eq!(
        with_alias.ledger.trace[0].release,
        vec!["unused-input-alias"]
    );
    assert_eq!(with_alias.trace_summary.peak_live_owner_bytes, 135);
    assert_eq!(with_alias.trace_summary.final_live_backing_bytes, 56);
    alias.inputs[1].state.key_domain = "incompatible-backing-domain".into();
    assert!(solve_public_encrypted_dag(&alias, c)
        .unwrap_err()
        .contains("aliased inputs"));
}

#[test]
fn encrypted_dag_known_zero_cost_and_any_typed_prefix_cost_overflow_are_distinct() {
    let mut zero = fork_join(30, 7, 60);
    for cap in &mut zero.capabilities {
        if let PublicCiphertextExecution::Ckks { compute, .. } = &mut cap.execution {
            compute.cost.units = Some(0);
        }
    }
    assert_eq!(
        solve_public_encrypted_dag(&zero, capacity(135))
            .unwrap()
            .ledger
            .total_cost,
        0
    );
    let mut overflow = fork_join(30, 7, 60);
    if let PublicCiphertextExecution::Ckks { compute, .. } = &mut overflow.capabilities[0].execution
    {
        compute.cost.units = Some(u64::MAX);
    }
    // The cost-12 small-overlap alternative exists; it cannot excuse arithmetic
    // overflow in another type-compatible path or certify an incomplete optimum.
    assert_eq!(
        solve_public_encrypted_dag(&overflow, capacity(135)).unwrap_err(),
        "public cost overflow"
    );
}
