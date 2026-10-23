use super::*;

fn limits() -> PublicEncryptedDagCapacity {
    PublicEncryptedDagCapacity {
        max_operations: 16,
        max_states: 2_000,
        max_frontier: 128,
        max_work: usize::MAX,
        max_buffer_bytes: 64 << 20,
        trace: PublicCiphertextTraceCapacity {
            max_records: 4096,
            max_work: usize::MAX,
            max_buffer_bytes: 2 << 20,
            max_live_owner_bytes: 10_000,
        },
    }
}

fn memo(objective: PublicEncryptedDagMemoObjective) -> PublicEncryptedDagMemoCapacity {
    PublicEncryptedDagMemoCapacity {
        max_entries: 512,
        objective,
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
        scheme: "synthetic-not-encrypted-execution".into(),
        key_domain: "external-domain".into(),
        layout: "synthetic-layout".into(),
        scale: "external-scale".into(),
        level: 2,
        shape: vec![4],
        error: PublicCiphertextError {
            model_identity: "external-error".into(),
            bound: Some(1),
        },
    }
}

fn value(id: &str, owner: &str) -> PublicCiphertextValue {
    PublicCiphertextValue {
        identity: id.into(),
        owner: owner.into(),
        state: state(),
    }
}

fn capability(
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
            transition: PublicLevelTransition {
                kind: PublicLevelTransitionKind::Preserve,
                from_level: 2,
                to_level: 2,
                cost: cost("provided-preserve", 0),
            },
        },
        error_assumption: Some("external-assumption-not-proof".into()),
        scratch_bytes: Some(scratch),
        recovery_overlap_bytes: Some(0),
    }
}

fn forks(n: usize) -> PublicCiphertextGraph {
    let mut g = PublicCiphertextGraph {
        cost_model_identity: "synthetic-memo-model".into(),
        cost_unit: "provided-unit".into(),
        maximum_level: 2,
        owners: vec![
            PublicCiphertextOwner {
                identity: "x-owner".into(),
                bytes: Some(40),
            },
            PublicCiphertextOwner {
                identity: "persistent-owner".into(),
                bytes: Some(5),
            },
        ],
        persistent_owners: vec!["persistent-owner".into()],
        inputs: vec![value("x", "x-owner")],
        operations: Vec::new(),
        capabilities: Vec::new(),
        outputs: Vec::new(),
        retained_inputs: vec!["x".into()],
    };
    for i in 0..n {
        let id = format!("v-{i}");
        let owner = format!("owner-{i}");
        let kernel = format!("kernel-{i}");
        g.owners.push(PublicCiphertextOwner {
            identity: owner.clone(),
            bytes: Some(1),
        });
        g.operations.push(PublicCiphertextGraphOperation {
            identity: format!("op-{i}"),
            kernel_identity: kernel.clone(),
            primitive: PublicPrimitive::Arithmetic,
            inputs: vec!["x".into()],
            output: value(&id, &owner),
        });
        g.capabilities
            .push(capability(&format!("cap-{i}"), &kernel, 1, 1, 0));
        g.outputs.push(id);
    }
    g
}

fn fork_join(a: u64, b: u64, scratch: u64) -> PublicCiphertextGraph {
    let mut g = forks(2);
    g.owners[2].bytes = Some(a);
    g.owners[3].bytes = Some(b);
    g.capabilities[0] = capability("fast", "kernel-0", 1, 2, scratch);
    g.capabilities[1] = capability("b", "kernel-1", 1, 3, 0);
    g.capabilities
        .insert(1, capability("small", "kernel-0", 1, 4, 0));
    let mut join = capability("join", "join-kernel", 2, 5, 2);
    join.recovery_overlap_bytes = Some(3);
    g.capabilities.push(join);
    g.owners.push(PublicCiphertextOwner {
        identity: "out-owner".into(),
        bytes: Some(11),
    });
    g.operations.push(PublicCiphertextGraphOperation {
        identity: "join-op".into(),
        kernel_identity: "join-kernel".into(),
        primitive: PublicPrimitive::Arithmetic,
        inputs: vec!["v-0".into(), "v-1".into()],
        output: value("out", "out-owner"),
    });
    g.outputs = vec!["out".into()];
    g
}

#[derive(Debug, Eq, PartialEq)]
struct Literal {
    units: u64,
    peak: u64,
    backing_peak: u64,
    selections: Vec<(usize, usize)>,
}

/// Independent exhaustive numeric oracle. No production readiness, memo,
/// transition, release, trace, cost or dominance helper is called.
fn literal(a: u64, b: u64, scratch: u64, ceiling: u64, peak_first: bool) -> Option<Literal> {
    let mut candidates = Vec::new();
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for small in [false, true] {
            let mut finished = 0u8;
            let mut backing = 45u64;
            let mut peak = 45u64;
            let mut backing_peak = 45u64;
            let mut units = 0u64;
            let mut selections = Vec::new();
            for node in order {
                if node == 2 && finished != 3 {
                    break;
                }
                let (bytes, charge, temporary, cap) = match node {
                    0 => (
                        a,
                        if small { 4 } else { 2 },
                        if small { 0 } else { scratch },
                        usize::from(small),
                    ),
                    1 => (b, 3, 0, 2),
                    _ => (11, 5, 5, 3),
                };
                backing += bytes;
                peak = peak.max(backing + temporary);
                backing_peak = backing_peak.max(backing);
                units += charge;
                finished |= 1 << node;
                selections.push((node, cap));
                if node == 2 {
                    backing -= a + b;
                }
            }
            if finished == 7 && peak <= ceiling {
                candidates.push(Literal {
                    units,
                    peak,
                    backing_peak,
                    selections,
                });
            }
        }
    }
    candidates.into_iter().min_by(|a, b| {
        a.units
            .cmp(&b.units)
            .then_with(|| {
                if peak_first {
                    a.peak.cmp(&b.peak)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then(a.selections.cmp(&b.selections))
    })
}

fn observed(s: &PublicEncryptedDagSolution) -> Literal {
    Literal {
        units: s.ledger.total_cost,
        peak: s.trace_summary.peak_live_owner_bytes,
        backing_peak: s.trace_summary.peak_live_backing_bytes,
        selections: s
            .ledger
            .selections
            .iter()
            .map(|p| (p.operation_index, p.capability_index))
            .collect(),
    }
}

#[test]
fn memo_matches_independent_exhaustive_oracle_and_legacy_cost_lex_winner() {
    for a in [1, 30] {
        for b in [1, 7] {
            for scratch in [0, 60] {
                for ceiling in [55, 68, 98, 135, 142, 256] {
                    let g = fork_join(a, b, scratch);
                    let mut c = limits();
                    c.trace.max_live_owner_bytes = ceiling;
                    for objective in [
                        PublicEncryptedDagMemoObjective::CostThenLexicographic,
                        PublicEncryptedDagMemoObjective::CostThenPeakThenLexicographic,
                    ] {
                        let peak_first = objective
                            == PublicEncryptedDagMemoObjective::CostThenPeakThenLexicographic;
                        let actual = solve_public_encrypted_dag_memo(&g, c, memo(objective));
                        match literal(a, b, scratch, ceiling, peak_first) {
                            Some(expected) => {
                                let solved = actual.unwrap();
                                assert_eq!(observed(&solved.solution), expected);
                                if !peak_first {
                                    let old = solve_public_encrypted_dag(&g, c).unwrap();
                                    assert_eq!(solved.solution.ledger, old.ledger);
                                    assert_eq!(solved.solution.trace_summary, old.trace_summary);
                                    assert_eq!(solved.solution.ledger_digest, old.ledger_digest);
                                }
                            }
                            None => {
                                assert!(actual.unwrap_err().contains("no feasible typed DAG trace"))
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn peak_lex_requires_pareto_labels_when_a_later_suffix_equalizes_peaks() {
    let mut g = fork_join(30, 7, 0);
    g.capabilities[2].scratch_bytes = Some(60);
    let c = limits();
    let legacy = solve_public_encrypted_dag(&g, c).unwrap();
    let low_peak = solve_public_encrypted_dag_memo(
        &g,
        c,
        memo(PublicEncryptedDagMemoObjective::CostThenPeakThenLexicographic),
    )
    .unwrap();
    assert_eq!(legacy.ledger.selections[0].operation_index, 0);
    assert_eq!(legacy.trace_summary.peak_live_owner_bytes, 142);
    assert_eq!(low_peak.solution.ledger.selections[0].operation_index, 1);
    assert_eq!(low_peak.solution.trace_summary.peak_live_owner_bytes, 112);
    // Before join: [A,B] peak 142, lex earlier; [B,A] peak 112, lex later.
    // Neither dominates. Join's 193 overlap erases the peak advantage, so A wins.
    g.capabilities[3].scratch_bytes = Some(100);
    g.capabilities[3].recovery_overlap_bytes = Some(0);
    let equalized = solve_public_encrypted_dag_memo(
        &g,
        c,
        memo(PublicEncryptedDagMemoObjective::CostThenPeakThenLexicographic),
    )
    .unwrap();
    assert_eq!(equalized.solution.trace_summary.peak_live_owner_bytes, 193);
    assert_eq!(equalized.solution.ledger.selections[0].operation_index, 0);
    assert_eq!(equalized.solution.ledger.total_cost, 10);
    let earlier = MemoLabel {
        len: 2,
        cost: 5,
        peak: 142,
        active: true,
    };
    let later = MemoLabel {
        peak: 112,
        ..earlier
    };
    let ab = [
        PublicEncryptedDagSelection {
            operation_index: 0,
            capability_index: 0,
        },
        PublicEncryptedDagSelection {
            operation_index: 1,
            capability_index: 2,
        },
    ];
    let ba = [ab[1], ab[0]];
    let objective = PublicEncryptedDagMemoObjective::CostThenPeakThenLexicographic;
    assert!(!memo_dominates(earlier, &ab, later, &ba, objective));
    assert!(!memo_dominates(later, &ba, earlier, &ab, objective));
}

#[test]
fn equal_totals_do_not_merge_different_live_owners_or_typed_domains() {
    let mut g = forks(2);
    let mut y = value("y", "y-owner");
    y.state.key_domain = "other-external-domain".into();
    y.state.scheme = "other-external-scheme".into();
    g.owners.push(PublicCiphertextOwner {
        identity: "y-owner".into(),
        bytes: Some(40),
    });
    g.operations[1].inputs = vec!["y".into()];
    g.operations[1].output.state = y.state.clone();
    g.capabilities[1].inputs = vec![y.state.clone()];
    g.capabilities[1].output = y.state.clone();
    g.inputs.push(y);
    g.retained_inputs.push("y".into()); // Same cost/backing/peak totals after A or B.
    let c = limits();
    let m = memo(PublicEncryptedDagMemoObjective::CostThenLexicographic);
    let actual = solve_public_encrypted_dag_memo(&g, c, m).unwrap();
    assert_eq!(actual.memo.entries, 4); // {}, {A}, {B}, {A,B}, not byte-total keys.
    assert_eq!(actual.solution.search_summary.states_generated, 5);
    assert_eq!(actual.memo.dominated, 1);
    assert_eq!(
        actual.solution.ledger,
        solve_public_encrypted_dag(&g, c).unwrap().ledger
    );
    let checked = validate_public_encrypted_dag_memo_solution(&g, &actual, c, m).unwrap();
    assert!(std::ptr::eq(checked.graph(), &g));
    assert!(std::ptr::eq(checked.solution(), &actual.solution));
    let mut forged = actual.clone();
    forged.memo.probes += 1;
    assert!(validate_public_encrypted_dag_memo_solution(&g, &forged, c, m).is_err());
}

#[test]
fn fork_join_views_keep_full_parent_backing_and_alias_refs_until_last_use() {
    let mut g = fork_join(30, 7, 0);
    g.retained_inputs.clear();
    g.inputs.push(value("unused-alias", "x-owner"));
    g.owners.remove(2);
    g.operations[0].output.owner = "x-owner".into();
    g.operations[0].output.state.shape = vec![1];
    for cap in &mut g.capabilities[..2] {
        cap.output.shape = vec![1];
        cap.execution = PublicCiphertextExecution::View {
            cost: cost("provided-view", 2),
        };
    }
    g.capabilities[3].inputs[0].shape = vec![1];
    g.capabilities[3].join = PublicCiphertextJoin::ScalarBroadcast { vector_input: 1 };
    let mut c = limits();
    c.trace.max_live_owner_bytes = 68;
    let m = memo(PublicEncryptedDagMemoObjective::CostThenLexicographic);
    let actual = solve_public_encrypted_dag_memo(&g, c, m).unwrap();
    let old = solve_public_encrypted_dag(&g, c).unwrap();
    assert_eq!(actual.solution.ledger, old.ledger);
    assert_eq!(actual.solution.trace_summary.peak_live_backing_bytes, 63);
    assert_eq!(actual.solution.trace_summary.peak_live_owner_bytes, 68);
    assert_eq!(actual.solution.trace_summary.final_live_backing_bytes, 16);
    assert_eq!(
        actual.solution.ledger.trace[0].release,
        vec!["unused-alias"]
    );
    assert_eq!(actual.solution.ledger.trace[1].release, vec!["x"]);
    c.trace.max_live_owner_bytes = 67;
    assert!(solve_public_encrypted_dag_memo(&g, c, m).is_err());
    g.inputs[1].state.key_domain = "incompatible-alias".into();
    assert!(solve_public_encrypted_dag_memo(&g, limits(), m)
        .unwrap_err()
        .contains("aliased inputs"));
}

#[test]
fn deterministic_subset_counts_replace_factorial_order_enumeration_not_timing() {
    let c = limits();
    let m = memo(PublicEncryptedDagMemoObjective::CostThenLexicographic);
    for n in 5..=8 {
        let g = forks(n);
        let actual = solve_public_encrypted_dag_memo(&g, c, m).unwrap();
        let states = &actual.solution.search_summary;
        assert_eq!(actual.memo.entries, 1 << n);
        assert_eq!(states.states_expanded, 1 << n);
        assert_eq!(states.states_generated, 1 + n * (1 << (n - 1)));
        assert_eq!(
            actual.memo.dominated,
            states.states_generated - actual.memo.entries
        );
        assert!(actual.memo.probes >= actual.memo.dominated);
        assert_eq!(actual.solution.ledger.total_cost, n as u64);
        assert_eq!(
            actual
                .solution
                .ledger
                .selections
                .iter()
                .map(|p| p.operation_index)
                .collect::<Vec<_>>(),
            (0..n).collect::<Vec<_>>()
        );
        if n <= 6 {
            let old = solve_public_encrypted_dag(&g, c).unwrap();
            assert_eq!(actual.solution.ledger, old.ledger);
            assert!(states.states_generated < old.search_summary.states_generated);
        } else {
            assert!(solve_public_encrypted_dag(&g, c)
                .unwrap_err()
                .contains("state capacity exceeded"));
        }
    }
}

#[test]
fn memo_preflight_zero_short_max_and_overflow_reject_before_exact_factories() {
    let g = fork_join(30, 7, 60);
    let c = limits();
    let m = memo(PublicEncryptedDagMemoObjective::CostThenLexicographic);
    let r = quote_public_encrypted_dag_memo(&g, c, m).unwrap();
    for case in 0..9 {
        let mut short = c;
        let mut bounded = m;
        match case {
            0 => bounded.max_entries = 0,
            1 => bounded.max_entries = usize::MAX,
            2 => short.max_work = 0,
            3 => short.max_work = r.minimum_work - 1,
            4 => short.max_buffer_bytes = 0,
            5 => short.max_buffer_bytes = r.buffer_bytes - 1,
            6 => short.max_frontier = usize::MAX,
            7 => short.max_states = 0,
            _ => short.max_operations = 0,
        }
        MEMO_FACTORIES.with(|n| n.set(0));
        assert!(
            solve_public_encrypted_dag_memo(&g, short, bounded).is_err(),
            "case {case}"
        );
        MEMO_FACTORIES.with(|n| assert_eq!(n.get(), 0, "case {case}"));
    }
}

#[test]
fn memo_exact_and_one_short_search_limits_never_return_an_incumbent() {
    let g = fork_join(30, 7, 60);
    let c = limits();
    let m = memo(PublicEncryptedDagMemoObjective::CostThenLexicographic);
    MEMO_FACTORIES.with(|n| n.set(0));
    let actual = solve_public_encrypted_dag_memo(&g, c, m).unwrap();
    MEMO_FACTORIES.with(|n| assert_eq!(n.get(), 6));
    let mut exact = c;
    exact.max_work = actual.solution.search_summary.work;
    assert_eq!(
        solve_public_encrypted_dag_memo(&g, exact, m).unwrap(),
        actual
    );
    exact.max_work -= 1;
    assert!(solve_public_encrypted_dag_memo(&g, exact, m)
        .unwrap_err()
        .contains("optimum not established"));
    for case in 0..3 {
        let mut short = c;
        let mut bounded = m;
        match case {
            0 => short.max_states = actual.solution.search_summary.states_generated - 1,
            1 => short.max_frontier = actual.solution.search_summary.peak_frontier - 1,
            _ => bounded.max_entries = actual.memo.entries - 1,
        }
        assert!(solve_public_encrypted_dag_memo(&g, short, bounded)
            .unwrap_err()
            .contains("optimum not established"));
    }
}

#[test]
fn provided_zero_max_and_dormant_cost_overflow_remain_distinct() {
    let c = limits();
    let m = memo(PublicEncryptedDagMemoObjective::CostThenLexicographic);
    let mut zero = forks(2);
    for cap in &mut zero.capabilities {
        if let PublicCiphertextExecution::Ckks { compute, .. } = &mut cap.execution {
            compute.cost.units = Some(0);
        }
    }
    assert_eq!(
        solve_public_encrypted_dag_memo(&zero, c, m)
            .unwrap()
            .solution
            .ledger
            .total_cost,
        0
    );
    let mut maximum = forks(1);
    if let PublicCiphertextExecution::Ckks { compute, .. } = &mut maximum.capabilities[0].execution
    {
        compute.cost.units = Some(u64::MAX);
    }
    assert_eq!(
        solve_public_encrypted_dag_memo(&maximum, c, m)
            .unwrap()
            .solution
            .ledger
            .total_cost,
        u64::MAX
    );
    let mut overflow = fork_join(30, 7, 60);
    if let PublicCiphertextExecution::Ckks { compute, .. } = &mut overflow.capabilities[0].execution
    {
        compute.cost.units = Some(u64::MAX);
    }
    MEMO_FACTORIES.with(|n| n.set(0));
    assert_eq!(
        solve_public_encrypted_dag_memo(&overflow, c, m).unwrap_err(),
        "public cost overflow"
    );
    MEMO_FACTORIES.with(|n| assert_eq!(n.get(), 0));
}

#[test]
fn malformed_static_source_cannot_hide_behind_dominance() {
    let source = fork_join(30, 7, 60);
    let c = limits();
    let m = memo(PublicEncryptedDagMemoObjective::CostThenLexicographic);
    for case in 0..7 {
        let mut g = source.clone();
        match case {
            0 => g.operations[1].identity = g.operations[0].identity.clone(),
            1 => g.operations[1].output.identity = "v-0".into(),
            2 => g.operations[1].inputs = vec!["missing".into()],
            3 => g.persistent_owners.push("unknown".into()),
            4 => g.outputs.push("unknown".into()),
            5 => g.capabilities[1].recovery_overlap_bytes = None,
            _ => g.owners[1].bytes = Some(u64::MAX),
        }
        MEMO_FACTORIES.with(|n| n.set(0));
        assert!(
            solve_public_encrypted_dag_memo(&g, c, m).is_err(),
            "case {case}"
        );
        MEMO_FACTORIES.with(|n| assert_eq!(n.get(), 0));
    }
}

#[test]
fn unsorted_operations_and_initial_dead_owners_do_not_change_state_meaning() {
    let mut g = fork_join(30, 7, 60);
    g.owners.push(PublicCiphertextOwner {
        identity: "cold-owner".into(),
        bytes: Some(100),
    });
    g.inputs.push(value("cold", "cold-owner"));
    let mut c = limits();
    c.trace.max_live_owner_bytes = 200;
    let m = memo(PublicEncryptedDagMemoObjective::CostThenLexicographic);
    let result = solve_public_encrypted_dag_memo(&g, c, m).unwrap();
    assert_eq!(
        result.solution.ledger,
        solve_public_encrypted_dag(&g, c).unwrap().ledger
    );
    assert_eq!(result.solution.ledger.total_cost, 10);
    assert_eq!(result.solution.ledger.selections[0].operation_index, 1);
    assert_eq!(result.solution.trace_summary.peak_live_owner_bytes, 152);
    assert_eq!(result.solution.ledger.trace[0].release, vec!["cold"]);
    g.operations.swap(0, 2);
    let unsorted = solve_public_encrypted_dag_memo(&g, c, m).unwrap();
    assert_eq!(
        unsorted.solution.ledger,
        solve_public_encrypted_dag(&g, c).unwrap().ledger
    );
    assert_eq!(unsorted.solution.ledger.total_cost, 10);
    // Even after cold's handle dies, its backing may NEVER become a fresh
    // compute owner. A live-only owner key/check would incorrectly permit this.
    g.operations[2].output.owner = "cold-owner".into();
    g.owners.retain(|o| o.identity != "owner-0");
    assert!(solve_public_encrypted_dag_memo(&g, c, m)
        .unwrap_err()
        .contains("no feasible typed DAG trace"));
    assert!(solve_public_encrypted_dag(&g, c).is_err());
}

#[test]
fn exact_arena_census_measures_actual_slice_layouts_not_the_quote_formula() {
    use std::mem::size_of_val;
    let g = fork_join(30, 7, 60);
    let c = limits();
    let m = memo(PublicEncryptedDagMemoObjective::CostThenLexicographic);
    let r = quote_public_encrypted_dag_memo(&g, c, m).unwrap();
    assert!(c.max_work >= r.minimum_work);
    let a = memo_arena(g.operations.len(), m.max_entries, r).unwrap();
    let arrays = [
        size_of_val(a.labels.as_ref()),
        size_of_val(a.paths.as_ref()),
        size_of_val(a.keys.as_ref()),
        size_of_val(a.frontier.as_ref()),
        size_of_val(a.path.as_ref()),
        size_of_val(a.key.as_ref()),
    ];
    let actual_backing: usize = arrays.iter().sum();
    assert_eq!(a.labels.len(), m.max_entries);
    assert_eq!(a.paths.len(), m.max_entries * g.operations.len());
    assert_eq!(a.keys.len(), m.max_entries * a.key.len());
    assert_eq!(a.frontier.len(), c.max_frontier.min(c.max_states));
    assert_eq!(a.path.len(), g.operations.len());
    assert!(actual_backing + size_of_val(&a) <= r.arena_bytes);
    eprintln!("DAG memo actual logical slice bytes={arrays:?}; backing={actual_backing}; owner_inline={}; quoted_arena={}; metadata={}",
        size_of_val(&a), r.arena_bytes, r.metadata_bytes);
}

#[test]
fn trace_work_buffer_and_record_exhaustion_remain_fatal_not_a_memo_skip() {
    let g = fork_join(30, 7, 60);
    let m = memo(PublicEncryptedDagMemoObjective::CostThenLexicographic);
    for case in 0..3 {
        let mut c = limits();
        match case {
            0 => c.trace.max_work = 0,
            1 => c.trace.max_buffer_bytes = 0,
            _ => c.trace.max_records = 0,
        }
        assert!(solve_public_encrypted_dag_memo(&g, c, m)
            .unwrap_err()
            .contains("composite"));
    }
}
