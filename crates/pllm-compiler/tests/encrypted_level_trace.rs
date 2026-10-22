//! External contract regressions. All sizes, costs and states are synthetic.
//! No ciphertext execution, cryptographic validity, source fidelity or decoder
//! availability follows from these tests. The oracle uses birth/death intervals
//! and owner-set unions, not the implementation's mutable reference counts.

use pllm_compiler::encrypted_level_schedule::*;

const VALUE_IDS: [&str; 4] = ["root", "view", "branch", "joined"];
const OP_IDS: [&str; 3] = ["make-view", "make-branch", "join"];
const ORDERS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

#[derive(Clone, Copy)]
struct Geometry {
    // Root and view share owner zero, irrespective of logical view width.
    owners: [u64; 4],
    scratch: [u64; 3],
    recovery: [u64; 3],
}

const SMALL: Geometry = Geometry {
    owners: [13, 5, 7, 3],
    scratch: [0, 2, 3],
    recovery: [0, 4, 9],
};
const ASYMMETRIC: Geometry = Geometry {
    owners: [29, 11, 17, 7],
    scratch: [0, 5, 13],
    recovery: [0, 8, 21],
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Expected {
    cost: u64,
    peak_backing: u64,
    peak_owners: u64,
    final_backing: u64,
}

/// Four release choices per value: after step 0, 1, 2, or never (3).
/// Exhaustion is exactly 6 orders * 4^4 release assignments per fixture.
fn interval_oracle(
    geometry: Geometry,
    retained_root: bool,
    order: [usize; 3],
    deaths: [usize; 4],
) -> Option<Expected> {
    let mut birth = [-1i32; 4];
    for (time, operation) in order.iter().enumerate() {
        birth[*operation + 1] = time as i32;
    }
    let dependencies: [&[usize]; 3] = [&[0], &[0], &[1, 2]];
    if (retained_root && deaths[0] != 3) || deaths[3] != 3 {
        return None;
    }
    for value in 0..4 {
        if (deaths[value] as i32) < birth[value] {
            return None; // Cannot retire an unborn handle.
        }
    }
    for (time, &operation) in order.iter().enumerate() {
        for &input in dependencies[operation] {
            if birth[input] >= time as i32 || deaths[input] < time {
                return None;
            }
        }
    }
    // Independent occupancy formula: an owner occupies memory iff any handle's
    // lifetime interval contains the allocation/recovery event. Owner 3 persists.
    let owner_for_value = [0, 0, 1, 2];
    let bytes_at = |time: i32, after_releases: bool| {
        let mut owner_set = 1u8 << 3;
        for value in 0..4 {
            let not_dead = if after_releases {
                deaths[value] as i32 > time
            } else {
                deaths[value] as i32 >= time
            };
            if birth[value] <= time && not_dead {
                owner_set |= 1u8 << owner_for_value[value];
            }
        }
        (0..4)
            .filter(|owner| owner_set & (1u8 << *owner) != 0)
            .map(|owner| geometry.owners[owner])
            .sum::<u64>()
    };
    let mut result = Expected {
        cost: 2 + 3 + 5,
        peak_backing: geometry.owners[0] + geometry.owners[3],
        peak_owners: geometry.owners[0] + geometry.owners[3],
        final_backing: bytes_at(2, true),
    };
    for (time, &operation) in order.iter().enumerate() {
        let backing = bytes_at(time as i32, false);
        result.peak_backing = result.peak_backing.max(backing);
        result.peak_owners = result
            .peak_owners
            .max(backing + geometry.scratch[operation] + geometry.recovery[operation]);
    }
    Some(result)
}

fn cost(id: &str, units: u64) -> PublicCost {
    PublicCost {
        identity: id.into(),
        units: Some(units),
    }
}

fn state(width: u64, level: u32) -> PublicCiphertextState {
    PublicCiphertextState {
        scheme: "synthetic-scheme".into(),
        key_domain: "synthetic-client-key".into(),
        layout: "synthetic-packing".into(),
        scale: "synthetic-scale".into(),
        level,
        shape: vec![width],
        error: PublicCiphertextError {
            model_identity: "synthetic-error-unit".into(),
            bound: Some(1),
        },
    }
}

fn value(id: &str, width: u64, level: u32, owner: &str) -> PublicCiphertextValue {
    PublicCiphertextValue {
        identity: id.into(),
        state: state(width, level),
        owner: owner.into(),
    }
}

fn preserve(level: u32) -> PublicLevelTransition {
    PublicLevelTransition {
        kind: PublicLevelTransitionKind::Preserve,
        from_level: level,
        to_level: level,
        cost: cost(&format!("preserve-{level}"), 0),
    }
}

fn compute(units: u64) -> PublicCiphertextExecution {
    PublicCiphertextExecution::Ckks {
        compute: PublicLevelCompute {
            input_level: 2,
            cost: cost(&format!("compute-{units}"), units),
        },
        depth: 0,
        transition: preserve(2),
    }
}

fn capacity(owner_limit: u64) -> PublicCiphertextTraceCapacity {
    PublicCiphertextTraceCapacity {
        max_records: 256,
        max_work: 1_100_000_000,
        max_buffer_bytes: 128 << 10,
        max_live_owner_bytes: owner_limit,
    }
}

fn fixture(geometry: Geometry, retained_root: bool) -> PublicCiphertextGraph {
    let root = value("root", 4, 2, "root-owner");
    let view = value("view", 1, 2, "root-owner");
    let branch = value("branch", 1, 2, "branch-owner");
    let joined = value("joined", 1, 2, "joined-owner");
    let inputs = [
        vec![root.state.clone()],
        vec![root.state.clone()],
        vec![view.state.clone(), branch.state.clone()],
    ];
    let outputs = [view, branch, joined];
    let dependencies: [Vec<String>; 3] = [
        vec!["root".into()],
        vec!["root".into()],
        vec!["view".into(), "branch".into()],
    ];
    let executions = [
        PublicCiphertextExecution::View {
            cost: cost("view-cost", 2),
        },
        compute(3),
        compute(5),
    ];
    PublicCiphertextGraph {
        cost_model_identity: "external-synthetic-cost-model".into(),
        cost_unit: "synthetic-unit".into(),
        maximum_level: 2,
        owners: [
            "root-owner",
            "branch-owner",
            "joined-owner",
            "persistent-owner",
        ]
        .iter()
        .zip(geometry.owners)
        .map(|(id, bytes)| PublicCiphertextOwner {
            identity: (*id).into(),
            bytes: Some(bytes),
        })
        .collect(),
        persistent_owners: vec!["persistent-owner".into()],
        inputs: vec![root],
        operations: (0..3)
            .map(|i| PublicCiphertextGraphOperation {
                identity: OP_IDS[i].into(),
                kernel_identity: OP_IDS[i].into(),
                primitive: PublicPrimitive::Arithmetic,
                inputs: dependencies[i].clone(),
                output: outputs[i].clone(),
            })
            .collect(),
        capabilities: (0..3)
            .map(|i| PublicCiphertextCapability {
                identity: OP_IDS[i].into(),
                kernel_identity: OP_IDS[i].into(),
                primitive: PublicPrimitive::Arithmetic,
                inputs: inputs[i].clone(),
                output: outputs[i].state.clone(),
                join: if i == 2 {
                    PublicCiphertextJoin::Aligned
                } else {
                    PublicCiphertextJoin::Unary
                },
                execution: executions[i].clone(),
                error_assumption: Some("synthetic-external-assumption".into()),
                scratch_bytes: Some(geometry.scratch[i]),
                recovery_overlap_bytes: Some(geometry.recovery[i]),
            })
            .collect(),
        outputs: vec!["joined".into()],
        retained_inputs: if retained_root {
            vec!["root".into()]
        } else {
            vec![]
        },
    }
}

fn trace(order: [usize; 3], deaths: [usize; 4]) -> Vec<PublicCiphertextTraceStep> {
    order
        .iter()
        .enumerate()
        .map(|(time, &operation)| PublicCiphertextTraceStep {
            operation: OP_IDS[operation].into(),
            capability: OP_IDS[operation].into(),
            release: (0..4)
                .filter(|&value| deaths[value] == time)
                .map(|value| VALUE_IDS[value].into())
                .collect(),
        })
        .collect()
}

fn assert_summary(actual: PublicCiphertextTraceSummary, expected: Expected) {
    assert_eq!(actual.total_cost, expected.cost);
    assert_eq!(actual.peak_live_backing_bytes, expected.peak_backing);
    assert_eq!(actual.peak_live_owner_bytes, expected.peak_owners);
    assert_eq!(actual.final_live_backing_bytes, expected.final_backing);
}

#[test]
fn exhaustive_small_orders_and_handle_lifetimes_match_interval_oracle() {
    // At most 6144 validations; three operations, four values and four owners.
    // There is no random geometry, open-ended search or reference to internals.
    for geometry in [SMALL, ASYMMETRIC] {
        for retained in [false, true] {
            let graph = fixture(geometry, retained);
            let source_before = graph.clone();
            let mut accepted = 0;
            for order in ORDERS {
                for code in 0usize..256 {
                    let deaths = [code % 4, (code / 4) % 4, (code / 16) % 4, (code / 64) % 4];
                    let candidate = trace(order, deaths);
                    let expected = interval_oracle(geometry, retained, order, deaths);
                    let result =
                        validate_public_ciphertext_trace(&graph, &candidate, capacity(1000));
                    assert_eq!(
                        result.is_ok(),
                        expected.is_some(),
                        "order={order:?}, deaths={deaths:?}, retained={retained}"
                    );
                    if let Some(expected) = expected {
                        accepted += 1;
                        assert_summary(result.unwrap(), expected);
                    }
                }
            }
            // Two legal orders. View and branch each have two retirement choices;
            // root has three choices unless retained. Final output cannot retire.
            assert_eq!(accepted, if retained { 8 } else { 24 });
            assert_eq!(graph, source_before);
        }
    }
}

#[test]
fn manual_peak_exact_and_one_short_include_alias_backing_and_recovery_overlap() {
    for (geometry, peak_backing, peak_owners, final_without_root) in
        [(SMALL, 28, 40, 10), (ASYMMETRIC, 64, 98, 24)]
    {
        for retained in [false, true] {
            let graph = fixture(geometry, retained);
            let deaths = [if retained { 3 } else { 1 }, 2, 2, 3];
            let candidate = trace([0, 1, 2], deaths);
            let expected = Expected {
                cost: 10,
                peak_backing,
                peak_owners,
                final_backing: final_without_root + if retained { geometry.owners[0] } else { 0 },
            };
            assert_eq!(
                interval_oracle(geometry, retained, [0, 1, 2], deaths),
                Some(expected)
            );
            assert_summary(
                validate_public_ciphertext_trace(&graph, &candidate, capacity(peak_owners))
                    .unwrap(),
                expected,
            );
            let before = (graph.clone(), candidate.clone());
            assert!(validate_public_ciphertext_trace(
                &graph,
                &candidate,
                capacity(peak_owners - 1)
            )
            .is_err());
            assert_eq!((graph, candidate), before);
        }
    }
}

#[test]
fn broadcast_accepts_both_operand_orders_and_rejects_illegal_scalar_shape() {
    for vector_input in [0, 1] {
        let mut graph = fixture(SMALL, true);
        let vector = graph.inputs[0].state.clone();
        let scalar = graph.operations[1].output.state.clone();
        graph.operations[2].inputs = if vector_input == 0 {
            vec!["root".into(), "branch".into()]
        } else {
            vec!["branch".into(), "root".into()]
        };
        graph.operations[2].output.state = vector.clone();
        graph.capabilities[2].inputs = if vector_input == 0 {
            vec![vector.clone(), scalar.clone()]
        } else {
            vec![scalar.clone(), vector.clone()]
        };
        graph.capabilities[2].output = vector;
        graph.capabilities[2].join = PublicCiphertextJoin::ScalarBroadcast { vector_input };
        let candidate = trace([0, 1, 2], [3, 2, 2, 3]);
        // Same owner inventory, hence same manually known peak despite shapes.
        assert_summary(
            validate_public_ciphertext_trace(&graph, &candidate, capacity(40)).unwrap(),
            Expected {
                cost: 10,
                peak_backing: 28,
                peak_owners: 40,
                final_backing: 23,
            },
        );
        let scalar_index = 1 - vector_input;
        let mut bad = graph.clone();
        bad.capabilities[2].inputs[scalar_index].shape = vec![2];
        assert!(validate_public_ciphertext_trace(&bad, &candidate, capacity(40)).is_err());
        let mut bad = graph;
        bad.capabilities[2].inputs[scalar_index].key_domain = "other-key".into();
        assert!(validate_public_ciphertext_trace(&bad, &candidate, capacity(40)).is_err());
    }
}

#[test]
fn aligned_join_rejects_independently_declared_state_disagreements() {
    let graph = fixture(SMALL, true);
    let candidate = trace([0, 1, 2], [3, 2, 2, 3]);
    for field in 0..7 {
        let mut bad = graph.clone();
        let input = &mut bad.capabilities[2].inputs[1];
        match field {
            0 => input.scheme = "other-scheme".into(),
            1 => input.key_domain = "other-key".into(),
            2 => input.layout = "other-layout".into(),
            3 => input.scale = "other-scale".into(),
            4 => input.level = 1,
            5 => input.error.model_identity = "other-error-unit".into(),
            _ => input.shape = vec![2],
        }
        let before = bad.clone();
        assert!(validate_public_ciphertext_trace(&bad, &candidate, capacity(40)).is_err());
        assert_eq!(bad, before);
    }
}

#[test]
fn source_state_binding_and_unknown_assumptions_fail_without_mutation() {
    let graph = fixture(SMALL, true);
    let candidate = trace([0, 1, 2], [3, 2, 2, 3]);
    for field in 0..12 {
        let mut bad = graph.clone();
        match field {
            0 => bad.inputs[0].state.scheme = "source-other-scheme".into(),
            1 => bad.inputs[0].state.key_domain = "source-other-key".into(),
            2 => bad.inputs[0].state.layout = "source-other-layout".into(),
            3 => bad.inputs[0].state.scale = "source-other-scale".into(),
            4 => bad.inputs[0].state.level = 1,
            5 => bad.inputs[0].state.error.model_identity = "source-other-error-unit".into(),
            6 => bad.inputs[0].state.error.bound = Some(2),
            7 => bad.inputs[0].state.shape = vec![3],
            8 => bad.capabilities[1].error_assumption = None,
            9 => bad.capabilities[1].output.error.bound = None,
            10 => bad.capabilities[1].recovery_overlap_bytes = None,
            _ => {
                if let PublicCiphertextExecution::Ckks { compute, .. } =
                    &mut bad.capabilities[1].execution
                {
                    compute.cost.units = None;
                }
            }
        }
        let before = (bad.clone(), candidate.clone());
        assert!(validate_public_ciphertext_trace(&bad, &candidate, capacity(1000)).is_err());
        assert_eq!((bad, candidate.clone()), before);
    }
    // Cyclic producer dependency is not rescued by a complete-looking trace.
    let mut cyclic = graph.clone();
    cyclic.operations[0].inputs = vec!["joined".into()];
    let before = cyclic.clone();
    assert!(validate_public_ciphertext_trace(&cyclic, &candidate, capacity(1000)).is_err());
    assert_eq!(cyclic, before);
    for field in 0..5 {
        let mut bad_output = graph.clone();
        let output = &mut bad_output.operations[2].output.state;
        match field {
            0 => output.key_domain = "output-other-key".into(),
            1 => output.level = 1,
            2 => output.layout = "output-other-layout".into(),
            3 => output.shape = vec![2],
            _ => output.error.bound = Some(2),
        }
        let before = bad_output.clone();
        assert!(validate_public_ciphertext_trace(&bad_output, &candidate, capacity(1000)).is_err());
        assert_eq!(bad_output, before);
    }
    assert_eq!(graph, fixture(SMALL, true));
}

#[test]
fn serial_and_composite_cost_identities_have_manual_known_totals() {
    assert_eq!(
        ENCRYPTED_LEVEL_PROBLEM_DOMAIN,
        "pllm.encrypted_level_problem.v1"
    );
    assert_eq!(
        ENCRYPTED_LEVEL_LEDGER_DOMAIN,
        "pllm.encrypted_level_ledger.v1"
    );
    for prefer_hybrid in [false, true] {
        let round_trip = PublicHybridRoundTrip {
            identity: "external-round-trip".into(),
            input_level: 2,
            output_level: 1,
            entry: cost("entry", if prefer_hybrid { 2 } else { 20 }),
            evaluate: cost("evaluate", if prefer_hybrid { 3 } else { 30 }),
            recover: cost("recover", if prefer_hybrid { 5 } else { 50 }),
        };
        let problem = PublicLevelProblem {
            cost_model_identity: "external-synthetic-cost-model".into(),
            cost_unit: "synthetic-unit".into(),
            maximum_level: 2,
            input_level: 2,
            output_level: 1,
            operations: vec![PublicLevelOperation {
                identity: "single".into(),
                predecessors: vec![],
                primitive: PublicPrimitive::Inv,
                depth: 1,
                compute: vec![
                    PublicLevelCompute {
                        input_level: 1,
                        cost: cost("compute-low", 7),
                    },
                    PublicLevelCompute {
                        input_level: 2,
                        cost: cost("compute-high", 11),
                    },
                ],
                transitions: vec![
                    preserve(0),
                    preserve(1),
                    PublicLevelTransition {
                        kind: PublicLevelTransitionKind::Refresh,
                        from_level: 0,
                        to_level: 1,
                        cost: cost("refresh", 13),
                    },
                ],
                hybrid: vec![round_trip.clone()],
            }],
        };
        let serial = solve_public_encrypted_levels(
            &problem,
            PublicLevelSolverCapacity {
                max_operations: 2,
                max_levels: 3,
                max_table_rows: 32,
                max_vertices: 12,
                max_work: 10_000,
                max_buffer_bytes: 1 << 20,
            },
        )
        .unwrap();
        let (execution, expected_cost, expected_charge_ids) = if prefer_hybrid {
            assert!(matches!(
                serial.ledger.steps[0].choice,
                PublicLevelChoice::Hybrid { .. }
            ));
            (
                PublicCiphertextExecution::Hybrid { round_trip },
                10,
                vec!["entry", "evaluate", "recover"],
            )
        } else {
            assert!(matches!(
                serial.ledger.steps[0].choice,
                PublicLevelChoice::Ckks { .. }
            ));
            (
                PublicCiphertextExecution::Ckks {
                    compute: PublicLevelCompute {
                        input_level: 2,
                        cost: cost("compute-high", 11),
                    },
                    depth: 1,
                    transition: preserve(1),
                },
                11,
                vec!["compute-high", "preserve-1"],
            )
        };
        assert_eq!(serial.ledger.total_cost, expected_cost);
        assert_eq!(
            serial.ledger.steps[0]
                .charges
                .iter()
                .map(|charge| charge.identity.as_str())
                .collect::<Vec<_>>(),
            expected_charge_ids
        );
        let input = value("input", 1, 2, "input-owner");
        let output = value("output", 1, 1, "output-owner");
        let graph = PublicCiphertextGraph {
            cost_model_identity: problem.cost_model_identity.clone(),
            cost_unit: problem.cost_unit.clone(),
            maximum_level: 2,
            owners: vec![
                PublicCiphertextOwner {
                    identity: "input-owner".into(),
                    bytes: Some(13),
                },
                PublicCiphertextOwner {
                    identity: "output-owner".into(),
                    bytes: Some(7),
                },
            ],
            persistent_owners: vec![],
            inputs: vec![input.clone()],
            operations: vec![PublicCiphertextGraphOperation {
                identity: "single".into(),
                kernel_identity: "single".into(),
                primitive: PublicPrimitive::Inv,
                inputs: vec!["input".into()],
                output: output.clone(),
            }],
            capabilities: vec![PublicCiphertextCapability {
                identity: "single".into(),
                kernel_identity: "single".into(),
                primitive: PublicPrimitive::Inv,
                inputs: vec![input.state],
                output: output.state,
                join: PublicCiphertextJoin::Unary,
                execution,
                error_assumption: Some("synthetic-external-assumption".into()),
                scratch_bytes: Some(3),
                recovery_overlap_bytes: Some(9),
            }],
            outputs: vec!["output".into()],
            retained_inputs: vec![],
        };
        let candidate = [PublicCiphertextTraceStep {
            operation: "single".into(),
            capability: "single".into(),
            release: vec!["input".into()],
        }];
        assert_summary(
            validate_public_ciphertext_trace(&graph, &candidate, capacity(32)).unwrap(),
            Expected {
                cost: expected_cost,
                peak_backing: 20,
                peak_owners: 32,
                final_backing: 7,
            },
        );
        assert!(validate_public_ciphertext_trace(&graph, &candidate, capacity(31)).is_err());
    }
}
