use super::*;

fn capacity() -> PublicLevelSolverCapacity {
    PublicLevelSolverCapacity {
        max_operations: 32,
        max_levels: 16,
        max_table_rows: 4096,
        max_vertices: 1024,
        max_work: 20_000_000,
        max_buffer_bytes: 128 << 20,
    }
}

fn charge(id: impl Into<String>, units: u64) -> PublicCost {
    PublicCost {
        identity: id.into(),
        units: Some(units),
    }
}

fn transition(from: u32, to: u32, units: u64) -> PublicLevelTransition {
    PublicLevelTransition {
        kind: if from == to {
            PublicLevelTransitionKind::Preserve
        } else if from > to {
            PublicLevelTransitionKind::Drop
        } else {
            PublicLevelTransitionKind::Refresh
        },
        from_level: from,
        to_level: to,
        cost: charge(format!("transition:{from}:{to}"), units),
    }
}

fn op(id: &str, depth: u32, max: u32) -> PublicLevelOperation {
    PublicLevelOperation {
        identity: id.into(),
        predecessors: vec![],
        primitive: PublicPrimitive::Inv,
        depth,
        compute: (depth..=max)
            .map(|x| PublicLevelCompute {
                input_level: x,
                cost: charge(format!("compute:{x}"), u64::from(x) + 1),
            })
            .collect(),
        transitions: (0..=max)
            .flat_map(|x| {
                (0..=max).map(move |y| transition(x, y, if y > x { 10 + u64::from(y) } else { 0 }))
            })
            .collect(),
        hybrid: vec![],
    }
}

fn problem(
    mut operations: Vec<PublicLevelOperation>,
    input: u32,
    output: u32,
    max: u32,
) -> PublicLevelProblem {
    for i in 1..operations.len() {
        operations[i].predecessors = vec![operations[i - 1].identity.clone()];
    }
    PublicLevelProblem {
        cost_model_identity: "synthetic-test-only".into(),
        cost_unit: "synthetic-integer-unit".into(),
        maximum_level: max,
        input_level: input,
        output_level: output,
        operations,
    }
}

fn hybrid(input: u32, output: u32, entry: u64, eval: u64, recovery: u64) -> PublicHybridRoundTrip {
    PublicHybridRoundTrip {
        identity: "explicit-round-trip".into(),
        input_level: input,
        output_level: output,
        entry: charge("entry-all-transforms", entry),
        evaluate: charge("evaluate", eval),
        recover: charge("recover-all-transforms", recovery),
    }
}

#[test]
fn minimal_input_compute_then_refresh_counterexample() {
    let p = problem(vec![op("first", 1, 2), op("internal", 1, 2)], 2, 2, 2);
    let result = solve_public_encrypted_levels(&p, capacity()).unwrap();
    assert_eq!(result.ledger.steps[1].input_level, 1);
    assert_eq!(result.ledger.steps[1].output_level, 2);
    let PublicLevelChoice::Ckks {
        compute_row,
        transition_row,
    } = result.ledger.steps[1].choice
    else {
        panic!("expected CKKS")
    };
    assert_eq!(p.operations[1].compute[compute_row].input_level, 1);
    let t = &p.operations[1].transitions[transition_row];
    assert_eq!((t.from_level, t.to_level), (0, 2));
    assert_eq!(result.ledger.total_cost, 3 + 2 + 12);
}

/// Independent brute-force oracle: enumerate complete level sequences directly
/// from Eq.(1), without production transitions, DP, relax, or cost helpers.
fn enumerate_formula(
    depths: &[u32],
    costs: &[Vec<u64>],
    boots: &[Vec<u64>],
    input: u32,
    output: u32,
    max: u32,
) -> Option<u64> {
    fn visit(
        i: usize,
        x: u32,
        acc: u64,
        d: &[u32],
        c: &[Vec<u64>],
        b: &[Vec<u64>],
        end: u32,
        max: u32,
        paths: &mut Vec<u64>,
    ) {
        if i == d.len() {
            if x == end {
                paths.push(acc);
            }
            return;
        }
        if x < d[i] {
            return;
        }
        for y in 0..=max {
            let refresh = if y > x - d[i] { b[i][y as usize] } else { 0 };
            visit(
                i + 1,
                y,
                acc + c[i][x as usize] + refresh,
                d,
                c,
                b,
                end,
                max,
                paths,
            );
        }
    }
    let mut paths = Vec::new();
    visit(0, input, 0, depths, costs, boots, output, max, &mut paths);
    paths.into_iter().min()
}

#[test]
fn full_dag_matches_independent_exhaustive_enumeration() {
    let max = 2;
    // All 27 three-operation depth patterns, every boundary pair, 8 cost
    // families including zero, ties, and deliberately nonmonotone entries.
    for pattern in 0..27 {
        let depths = [pattern % 3, (pattern / 3) % 3, (pattern / 9) % 3];
        for seed in 0..8_u64 {
            let mut costs = Vec::new();
            let mut boots = Vec::new();
            let mut operations = Vec::new();
            for (i, depth) in depths.iter().enumerate() {
                let c: Vec<_> = (0..=max)
                    .map(|x| (seed * 5 + u64::from(x) * 7 + i as u64) % 11)
                    .collect();
                let b: Vec<_> = (0..=max)
                    .map(|y| (seed * 3 + u64::from(y) * 5 + i as u64) % 13)
                    .collect();
                let mut operation = op(&format!("operation:{i}"), *depth, max);
                for row in &mut operation.compute {
                    row.cost.units = Some(c[row.input_level as usize]);
                }
                for t in &mut operation.transitions {
                    t.cost.units = Some(if t.to_level > t.from_level {
                        b[t.to_level as usize]
                    } else {
                        0
                    });
                }
                operations.push(operation);
                costs.push(c);
                boots.push(b);
            }
            for input in 0..=max {
                for output in 0..=max {
                    let expected = enumerate_formula(&depths, &costs, &boots, input, output, max);
                    let p = problem(operations.clone(), input, output, max);
                    match expected {
                        Some(total) => assert_eq!(
                            solve_public_encrypted_levels(&p, capacity())
                                .unwrap()
                                .ledger
                                .total_cost,
                            total
                        ),
                        None => assert!(solve_public_encrypted_levels(&p, capacity())
                            .unwrap_err()
                            .contains("unreachable")),
                    }
                }
            }
        }
    }
}

#[test]
fn nonmonotone_compute_cost_retains_high_level_path() {
    let mut a = op("a", 0, 2);
    a.compute.iter_mut().for_each(|r| r.cost.units = Some(0));
    let mut b = op("b", 1, 2);
    b.compute[0].cost.units = Some(100);
    b.compute[1].cost.units = Some(1);
    let p = problem(vec![a, b], 2, 1, 2);
    let r = solve_public_encrypted_levels(&p, capacity()).unwrap();
    assert_eq!(r.ledger.total_cost, 1);
    assert_eq!(r.ledger.steps[1].input_level, 2);
}

// Exhaust all concrete paths, not a shortest-path recurrence or a production
// edge generator. Used independently for explicit hybrid legality/cost tables.
fn enumerate_legal_paths(p: &PublicLevelProblem, i: usize, input: u32) -> Vec<(u32, u64)> {
    if i == p.operations.len() {
        return vec![(input, 0)];
    }
    let op = &p.operations[i];
    let mut one_step = Vec::new();
    for compute in &op.compute {
        if compute.input_level != input || input < op.depth {
            continue;
        }
        for t in &op.transitions {
            if input == t.from_level + op.depth {
                one_step.push((
                    t.to_level,
                    compute.cost.units.unwrap() + t.cost.units.unwrap(),
                ));
            }
        }
    }
    for h in &op.hybrid {
        if h.input_level == input {
            one_step.push((
                h.output_level,
                h.entry.units.unwrap() + h.evaluate.units.unwrap() + h.recover.units.unwrap(),
            ));
        }
    }
    let mut all = Vec::new();
    for (next, edge) in one_step {
        for (end, tail) in enumerate_legal_paths(p, i + 1, next) {
            all.push((end, edge + tail));
        }
    }
    all
}

#[test]
fn hybrid_small_dags_match_exhaustive_paths() {
    for seed in 0..16_u64 {
        let mut operations = Vec::new();
        for i in 0..3_u32 {
            let mut a = op(&format!("op:{i}"), 1 + (seed as u32 + i) % 2, 2);
            for row in &mut a.compute {
                row.cost.units = Some((seed + u64::from(row.input_level) * 7) % 17);
            }
            // Distinct explicit legal entries; no implied access from other levels.
            for x in 0..=2 {
                let mut h = hybrid(
                    x,
                    (x + i) % 3,
                    seed % 5,
                    (seed + u64::from(x)) % 7,
                    (seed + 3) % 11,
                );
                h.identity = format!("hybrid:{x}");
                a.hybrid.push(h);
            }
            operations.push(a);
        }
        for input in 0..=2 {
            for output in 0..=2 {
                let p = problem(operations.clone(), input, output, 2);
                let expected = enumerate_legal_paths(&p, 0, input)
                    .into_iter()
                    .filter(|(end, _)| *end == output)
                    .map(|(_, cost)| cost)
                    .min();
                match expected {
                    Some(total) => assert_eq!(
                        solve_public_encrypted_levels(&p, capacity())
                            .unwrap()
                            .ledger
                            .total_cost,
                        total
                    ),
                    None => assert!(solve_public_encrypted_levels(&p, capacity()).is_err()),
                }
            }
        }
    }
}

#[test]
fn no_refresh_no_conversion_and_no_free_restoration() {
    let mut a = op("a", 1, 2);
    a.transitions
        .retain(|t| t.kind != PublicLevelTransitionKind::Refresh);
    let mut p = problem(vec![a], 2, 1, 2);
    assert_eq!(
        solve_public_encrypted_levels(&p, capacity())
            .unwrap()
            .ledger
            .total_cost,
        3
    );
    p.output_level = 2;
    assert!(solve_public_encrypted_levels(&p, capacity())
        .unwrap_err()
        .contains("unreachable"));
    p.operations[0].transitions = vec![transition(1, 2, 7)];
    assert_eq!(
        solve_public_encrypted_levels(&p, capacity())
            .unwrap()
            .ledger
            .total_cost,
        10
    );
    p.operations[0].transitions[0].kind = PublicLevelTransitionKind::Drop;
    assert!(solve_public_encrypted_levels(&p, capacity()).is_err());
    p.input_level = 0;
    p.operations[0].transitions = vec![transition(0, 2, 7)];
    assert!(solve_public_encrypted_levels(&p, capacity())
        .unwrap_err()
        .contains("unreachable"));
}

#[test]
fn hybrid_joint_choice_charges_all_conversions_and_enforces_entry() {
    let mut a = op("a", 2, 2);
    a.hybrid.push(hybrid(1, 2, 3, 4, 5));
    let mut p = problem(vec![a, op("b", 1, 2)], 1, 1, 2);
    let r = solve_public_encrypted_levels(&p, capacity()).unwrap();
    assert_eq!(r.ledger.total_cost, 12 + 3);
    assert!(matches!(
        r.ledger.steps[0].choice,
        PublicLevelChoice::Hybrid { .. }
    ));
    assert_eq!(r.ledger.steps[0].charges.len(), 3);
    p.input_level = 0;
    assert!(solve_public_encrypted_levels(&p, capacity())
        .unwrap_err()
        .contains("unreachable"));
    p.input_level = 2;
    // No entry capability at level 2: compute 3, refresh to 2 costs 12,
    // then downstream computation costs 3 (its residual is requested level 1).
    assert_eq!(
        solve_public_encrypted_levels(&p, capacity())
            .unwrap()
            .ledger
            .total_cost,
        18
    );
}

#[test]
fn explicit_hybrid_only_depth_and_unknown_costs() {
    let mut a = op("a", 3, 2);
    a.hybrid.push(hybrid(1, 2, 1, 2, 3));
    let mut p = problem(vec![a], 1, 2, 2);
    assert_eq!(
        solve_public_encrypted_levels(&p, capacity())
            .unwrap()
            .ledger
            .total_cost,
        6
    );
    p.operations[0].hybrid[0].recover.units = None;
    assert!(solve_public_encrypted_levels(&p, capacity())
        .unwrap_err()
        .contains("unknown"));
    let mut p = problem(vec![op("a", 1, 2)], 1, 0, 2);
    // Unknown unreachable row still rejects: cannot claim optimum over unknowns.
    p.operations[0].compute[1].cost.units = None;
    assert!(solve_public_encrypted_levels(&p, capacity()).is_err());
    p.operations[0].compute[1].cost.units = Some(3);
    p.operations[0].transitions.last_mut().unwrap().cost.units = None;
    assert!(solve_public_encrypted_levels(&p, capacity()).is_err());
}

#[test]
fn malformed_tables_boundaries_dependencies_and_nonlinear_reject() {
    let original = problem(vec![op("a", 1, 2), op("b", 1, 2)], 2, 0, 2);
    let mut p = original.clone();
    p.operations[0].compute.pop();
    assert!(solve_public_encrypted_levels(&p, capacity()).is_err());
    let mut p = original.clone();
    p.operations[0].compute[1].input_level = 1;
    assert!(solve_public_encrypted_levels(&p, capacity()).is_err());
    let mut p = original.clone();
    p.operations[1].predecessors.push("skip".into());
    assert!(solve_public_encrypted_levels(&p, capacity())
        .unwrap_err()
        .contains("branch/join"));
    let mut p = original.clone();
    p.operations[1].predecessors.clear();
    assert!(solve_public_encrypted_levels(&p, capacity()).is_err());
    let mut p = original.clone();
    p.operations[0].primitive = PublicPrimitive::UnknownNonlinear;
    assert!(solve_public_encrypted_levels(&p, capacity())
        .unwrap_err()
        .contains("unknown nonlinear"));
    let mut p = original.clone();
    p.operations[0].primitive = PublicPrimitive::Arithmetic;
    p.operations[0].hybrid.push(hybrid(2, 2, 1, 1, 1));
    assert!(solve_public_encrypted_levels(&p, capacity()).is_err());
    let mut p = original.clone();
    p.output_level = 3;
    assert!(solve_public_encrypted_levels(&p, capacity()).is_err());
    let mut p = original;
    p.cost_model_identity = "not a valid identity".into();
    assert!(solve_public_encrypted_levels(&p, capacity()).is_err());
}

#[test]
fn stable_ties_ledger_conservation_and_digest_binding() {
    let mut a = op("a", 1, 2);
    a.hybrid.push(hybrid(1, 2, 2, 4, 8)); // CKKS = 2 + 12 = 14 too.
    let p = problem(vec![a], 1, 2, 2);
    let result = solve_public_encrypted_levels(&p, capacity()).unwrap();
    assert!(matches!(
        result.ledger.steps[0].choice,
        PublicLevelChoice::Ckks { .. }
    ));
    assert_eq!(
        result,
        solve_public_encrypted_levels(&p, capacity()).unwrap()
    );
    assert_eq!(
        result.ledger.total_cost,
        result
            .ledger
            .steps
            .iter()
            .flat_map(|s| &s.charges)
            .map(|c| c.units.unwrap())
            .sum::<u64>()
    );
    assert_eq!(
        result.ledger_digest,
        canonical_digest(ENCRYPTED_LEVEL_LEDGER_DOMAIN, &result.ledger)
    );
    let mut changed = p.clone();
    changed.cost_model_identity = "different-model".into();
    assert_ne!(
        result.ledger_digest,
        solve_public_encrypted_levels(&changed, capacity())
            .unwrap()
            .ledger_digest
    );
    changed = p;
    changed.operations[0]
        .transitions
        .iter_mut()
        .find(|t| t.from_level == 0 && t.to_level == 2)
        .unwrap()
        .cost
        .identity = "different-legal-transform".into();
    assert_ne!(
        result.ledger_digest,
        solve_public_encrypted_levels(&changed, capacity())
            .unwrap()
            .ledger_digest
    );
}

#[test]
fn equal_cost_legal_transition_ties_follow_table_order_and_bind_digest() {
    let mut a = op("a", 1, 2);
    let mut first = transition(0, 2, 7);
    first.cost.identity = "legal-refresh:first".into();
    let mut second = first.clone();
    second.cost.identity = "legal-refresh:second".into();
    a.transitions = vec![first, second];
    let mut p = problem(vec![a], 1, 2, 2);
    let original = solve_public_encrypted_levels(&p, capacity()).unwrap();
    assert_eq!(original.ledger.total_cost, 9);
    assert_eq!(
        original.ledger.steps[0].charges[1].identity,
        "legal-refresh:first"
    );
    assert!(matches!(
        original.ledger.steps[0].choice,
        PublicLevelChoice::Ckks {
            transition_row: 0,
            ..
        }
    ));

    p.operations[0].transitions.swap(0, 1);
    let swapped = solve_public_encrypted_levels(&p, capacity()).unwrap();
    assert_eq!(swapped.ledger.total_cost, original.ledger.total_cost);
    assert_eq!(
        swapped.ledger.steps[0].charges[1].identity,
        "legal-refresh:second"
    );
    assert!(matches!(
        swapped.ledger.steps[0].choice,
        PublicLevelChoice::Ckks {
            transition_row: 0,
            ..
        }
    ));
    assert_ne!(
        original.ledger.problem_digest,
        swapped.ledger.problem_digest
    );
    assert_ne!(original.ledger_digest, swapped.ledger_digest);
}

#[test]
fn unused_computation_cost_row_still_binds_problem_digest() {
    let mut p = problem(vec![op("a", 1, 2)], 1, 0, 2);
    let original = solve_public_encrypted_levels(&p, capacity()).unwrap();
    // Input level 2 is unreachable in this single-operation problem.
    assert_eq!(p.operations[0].compute[1].input_level, 2);
    p.operations[0].compute[1].cost.units = Some(99);
    let changed = solve_public_encrypted_levels(&p, capacity()).unwrap();
    assert_eq!(original.ledger.steps, changed.ledger.steps);
    assert_eq!(original.ledger.total_cost, changed.ledger.total_cost);
    assert_ne!(
        original.ledger.problem_digest,
        changed.ledger.problem_digest
    );
    assert_ne!(original.ledger_digest, changed.ledger_digest);
}

#[test]
fn unreachable_matching_edge_cost_overflow_rejects_entire_problem() {
    let mut a = op("a", 1, 2);
    a.transitions = vec![transition(0, 0, 0), transition(1, 2, 1)];
    let mut p = problem(vec![a], 1, 0, 2);
    // Reachable requested path costs 2 + 0. The other matching pair starts
    // at unreachable input 2, computes to 1, then refreshes to 2.
    let baseline = solve_public_encrypted_levels(&p, capacity()).unwrap();
    assert_eq!(baseline.ledger.total_cost, 2);
    assert_eq!(p.operations[0].compute[1].input_level, 2);
    assert_eq!(p.operations[0].transitions[1].from_level, 1);
    p.operations[0].compute[1].cost.units = Some(u64::MAX);
    assert_eq!(
        solve_public_encrypted_levels(&p, capacity()).unwrap_err(),
        "public cost overflow"
    );
}

#[test]
fn cost_and_size_overflow_and_each_capacity_reject_without_truncation() {
    let p = problem(vec![op("a", 1, 2)], 1, 2, 2);
    let mut too_expensive = p.clone();
    too_expensive.operations[0].compute[0].cost.units = Some(u64::MAX);
    assert!(solve_public_encrypted_levels(&too_expensive, capacity())
        .unwrap_err()
        .contains("overflow"));
    let mut p2 = problem(vec![op("a", 0, 2), op("b", 0, 2)], 1, 1, 2);
    p2.operations[0]
        .compute
        .iter_mut()
        .for_each(|r| r.cost.units = Some(u64::MAX));
    p2.operations[0]
        .transitions
        .retain(|t| t.kind == PublicLevelTransitionKind::Preserve);
    assert!(solve_public_encrypted_levels(&p2, capacity())
        .unwrap_err()
        .contains("overflow"));
    assert!(add(usize::MAX, 1).is_err());
    assert!(mul(usize::MAX, 2).is_err());
    for field in 0..6 {
        let mut c = capacity();
        match field {
            0 => c.max_operations = 0,
            1 => c.max_levels = 2,
            2 => c.max_table_rows = 1,
            3 => c.max_vertices = 1,
            4 => c.max_work = 1,
            _ => c.max_buffer_bytes = 1,
        }
        assert!(solve_public_encrypted_levels(&p, c)
            .unwrap_err()
            .contains("capacity"));
    }
}
