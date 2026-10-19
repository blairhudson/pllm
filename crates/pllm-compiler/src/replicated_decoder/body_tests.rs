use super::*;

#[path = "body_layer_tests.rs"]
mod layer_tests;
use crate::replicated_decoder::{runtime::body, tests::workers};

type Waves = Vec<Vec<(String, usize)>>;

fn wave(queue: &[(puma::Party, puma::Party, PeerBytes)]) -> Vec<(String, usize)> {
    let mut signature: Vec<_> = queue
        .iter()
        .map(|(from, to, frame)| {
            let message = puma::Message::decode(frame.as_bytes(), *from, *to).unwrap();
            (
                format!("{:?}", message.header().phase),
                message.words().len(),
            )
        })
        .collect();
    signature.sort();
    signature
}

fn expanded_run(mut owners: [Worker; 3]) -> (Vec<u64>, Waves) {
    let mut queue = Vec::new();
    let mut waves = Vec::new();
    for _ in 0..10000 {
        if owners.iter().all(Worker::complete) {
            break;
        }
        if queue.is_empty() {
            for owner in &mut owners {
                let from = owner.party();
                queue.extend(
                    owner
                        .advance()
                        .unwrap()
                        .into_iter()
                        .map(|(to, frame)| (from, to, frame)),
                );
            }
        } else {
            waves.push(wave(&queue));
            for (from, to, frame) in std::mem::take(&mut queue) {
                queue.extend(
                    owners[to as usize]
                        .receive(frame.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, frame)| (to, next, frame)),
                );
            }
        }
    }
    assert!(queue.is_empty());
    assert!(owners.iter().all(Worker::complete));
    let shares = owners.map(|owner| owner.finish().unwrap());
    let result = (0..shares[0].len())
        .map(|i| {
            shares[0][i]
                .wrapping_add(shares[1][i])
                .wrapping_add(shares[2][i])
        })
        .collect();
    (result, waves)
}

fn expanded(
    inputs: &[body::Register],
    sites: &[body::Site],
    outputs: &[body::View],
) -> Arc<Program> {
    let mut builder = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let mut registers = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let id = builder
            .push(Code::Input(format!("x{i}")), input.lanes, input.bounds)
            .unwrap();
        registers.push(builder.view(id, vec![input.lanes]).words);
    }
    let view = |layout: body::View, registers: &[Words]| -> Words {
        (0..layout.rows)
            .flat_map(|row| {
                (0..layout.columns).map(move |column| {
                    registers[layout.owner]
                        .at(layout.offset + row * layout.row_stride + column * layout.column_stride)
                })
            })
            .collect()
    };
    for site in sites {
        registers.push(
            builder
                .binary(
                    &view(site.left, &registers),
                    &view(site.right, &registers),
                    matches!(site.operation, body::Binary::Multiply),
                )
                .unwrap(),
        );
    }
    let outputs: Vec<_> = outputs
        .iter()
        .flat_map(|v| view(*v, &registers).iter().collect::<Vec<_>>())
        .collect();
    let mut last: Vec<_> = (0..builder.nodes.len()).collect();
    for (step, node) in builder.nodes.iter().enumerate() {
        for id in node.inputs() {
            last[id] = step;
        }
    }
    for &(owner, _) in &outputs {
        last[owner] = builder.nodes.len();
    }
    let mut retire = vec![vec![]; builder.nodes.len()];
    for (id, step) in last.into_iter().enumerate() {
        if step < retire.len() {
            retire[step].push(id);
        }
    }
    let work = schedule_work(&builder.nodes).unwrap();
    Arc::new(Program {
        nodes: builder.nodes,
        inputs: inputs.len(),
        outputs,
        retire,
        digest: [83; 32],
        source_binding: [0; 32],
        payload_bytes: builder.allocated,
        work,
    })
}

fn run(
    plan: Arc<body::Plan>,
    fixture: Arc<Program>,
    clear: &BTreeMap<String, Vec<i64>>,
) -> (Vec<u64>, usize, Waves) {
    let mut owners =
        workers(fixture, clear).map(|worker| body::from_test_worker(plan.clone(), worker));
    let mut queue = Vec::new();
    let mut advances = 0;
    let mut waves = Vec::new();
    for _ in 0..10000 {
        if owners.iter().all(body::Executor::complete) {
            break;
        }
        if queue.is_empty() {
            advances += 1;
            for (from, owner) in owners.iter_mut().enumerate() {
                queue.extend(owner.advance().unwrap().into_iter().map(|(to, frame)| {
                    (
                        [puma::Party::P0, puma::Party::P1, puma::Party::P2][from],
                        to,
                        frame,
                    )
                }));
            }
        } else {
            waves.push(wave(&queue));
            for (from, to, frame) in std::mem::take(&mut queue) {
                queue.extend(
                    owners[to as usize]
                        .receive(frame.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, frame)| (to, next, frame)),
                );
            }
        }
        for owner in &owners {
            assert!(owner.live_bytes() <= plan.peak_live_bytes());
        }
    }
    assert!(queue.is_empty());
    assert!(owners.iter().all(body::Executor::complete));
    let results = owners.map(|owner| owner.finish().unwrap());
    let expected = plan.work_budget().cursors();
    for (runtime, _) in &results {
        let actual = runtime.cursors();
        assert_eq!(actual.public_blocks, expected.public_blocks);
        assert_eq!(actual.pair_blocks, expected.pair_blocks);
        assert!(actual.private_blocks <= expected.private_blocks);
        assert!(actual.aes_work <= expected.aes_work);
    }
    let mut out = Vec::new();
    // Fixture output count matches the retained views. All test fixtures have
    // one full output and a second retained view; bounds are checked by word().
    for output in 0..2 {
        for index in 0..MAX_VALUES {
            let Ok((a, _)) = results[0].1.word(output, index) else {
                break;
            };
            let (b, _) = results[1].1.word(output, index).unwrap();
            let (c, _) = results[2].1.word(output, index).unwrap();
            out.push(a.wrapping_add(b).wrapping_add(c));
        }
    }
    (out, advances, waves)
}

#[test]
fn body_unit_binary_order_tails_and_retained_strided_views_match_expanded() {
    for n in [1, 127, 128, 129, 257] {
        let inputs = [body::Register {
            lanes: n,
            bounds: Interval::new(-2 * ONE, 2 * ONE).unwrap(),
        }; 2];
        let a = body::View::contiguous(0, n);
        let b = body::View::contiguous(1, n);
        let sites = [
            body::Site {
                left: a,
                right: b,
                operation: body::Binary::Multiply,
            },
            body::Site {
                left: body::View::contiguous(2, n),
                right: a,
                operation: body::Binary::Add,
            },
            body::Site {
                left: body::View::contiguous(3, n),
                right: b,
                operation: body::Binary::Multiply,
            },
        ];
        let outputs = [
            body::View::contiguous(4, n),
            body::View {
                owner: 0,
                offset: 0,
                rows: 1,
                columns: n.div_ceil(2),
                row_stride: n,
                column_stride: 2,
            },
        ];
        let plan =
            Arc::new(body::Plan::new(WeightMode::Public, &inputs, &sites, &outputs).unwrap());
        let fixture = expanded(&inputs, &sites, &outputs);
        assert_eq!(plan.work_budget(), fixture.work_budget());
        let clear = BTreeMap::from([
            (
                "x0".into(),
                (0..n).map(|i| (i as i64 % 5 - 2) * ONE).collect(),
            ),
            ("x1".into(), vec![ONE; n]),
        ]);
        let (expected, expanded_waves) = expanded_run(workers(fixture.clone(), &clear));
        let (actual, advances, body_waves) = run(plan.clone(), fixture, &clear);
        assert_eq!(actual, expected);
        // Public phase/width wave order, not PRSS payload equality: setup uses
        // fresh OS entropy in each run. Stronger fixed-PRSS proof needs the
        // separate test-only runtime factory; do not fake it with zero masks.
        assert_eq!(body_waves, expanded_waves);
        assert_eq!(advances as u64, plan.coordinator_advances());
    }
}

#[test]
fn body_unit_compact_liveness_prices_peak_not_cumulative_expansion() {
    let n = 1024;
    let inputs = [body::Register {
        lanes: n,
        bounds: Interval::new(-ONE, ONE).unwrap(),
    }; 2];
    let sites: Vec<_> = (0..100)
        .map(|i| body::Site {
            left: body::View::contiguous(if i == 0 { 0 } else { i + 1 }, n),
            right: body::View::contiguous(1, n),
            operation: body::Binary::Multiply,
        })
        .collect();
    let output = [body::View::contiguous(101, n)];
    let plan = body::Plan::new(WeightMode::Public, &inputs, &sites, &output).unwrap();
    assert_eq!(plan.peak_live_bytes(), 3 * n * 16);
    assert!(plan.metadata_bytes() < 64 << 10);
    assert!(plan.peak_live_bytes() < n * 16 * sites.len());
    let report = crate::replicated_decoder::cost::binary_body_unit(&plan);
    assert!(!report.complete_decoder);
    assert_eq!(report.resources.peak_live_secret_bytes, 49_152);
    assert!(report.resources.admitted_bytes < MAX_PROGRAM_BYTES);
    assert_eq!(report.work.ordinals, plan.work_budget().operations());
    assert!(report.peer_rounds.is_none());
    assert!(body::Plan::new(WeightMode::Private, &inputs, &sites, &output).is_err());
    let large = [body::Register {
        lanes: MAX_VALUES,
        bounds: inputs[0].bounds,
    }; 4];
    assert!(body::Plan::new(
        WeightMode::Public,
        &large,
        &[body::Site {
            left: body::View::contiguous(0, MAX_VALUES),
            right: body::View::contiguous(1, MAX_VALUES),
            operation: body::Binary::Multiply
        }],
        &[body::View::contiguous(4, MAX_VALUES)]
    )
    .is_err());
}

#[test]
fn body_unit_transpose_and_row_broadcast_keep_original_odd_layout() {
    let inputs = [
        body::Register {
            lanes: 6,
            bounds: Interval::new(-3 * ONE, 3 * ONE).unwrap(),
        },
        body::Register {
            lanes: 3,
            bounds: Interval::new(-2 * ONE, 2 * ONE).unwrap(),
        },
    ];
    let sites = [body::Site {
        left: body::View {
            owner: 0,
            offset: 0,
            rows: 3,
            columns: 2,
            row_stride: 1,
            column_stride: 3,
        },
        right: body::View {
            owner: 1,
            offset: 0,
            rows: 3,
            columns: 2,
            row_stride: 1,
            column_stride: 0,
        },
        operation: body::Binary::Add,
    }];
    let outputs = [body::View::contiguous(2, 6), body::View::contiguous(0, 6)];
    let plan = Arc::new(body::Plan::new(WeightMode::Public, &inputs, &sites, &outputs).unwrap());
    let fixture = expanded(&inputs, &sites, &outputs);
    let clear = BTreeMap::from([
        ("x0".into(), vec![-3 * ONE, -2 * ONE, -ONE, 0, ONE, 2 * ONE]),
        ("x1".into(), vec![ONE, -ONE, 2 * ONE]),
    ]);
    let (expected, waves) = expanded_run(workers(fixture.clone(), &clear));
    let (actual, advances, body_waves) = run(plan, fixture, &clear);
    assert_eq!(actual, expected);
    assert_eq!(
        actual[..6],
        [-2 * ONE, ONE, -3 * ONE, 0, ONE, 4 * ONE].map(|x| x as u64)
    );
    assert_eq!(waves, body_waves);
    assert_eq!(advances, 1);
}

#[test]
fn body_unit_layouts_are_checked_and_bound_without_lane_descriptors() {
    let input = [body::Register {
        lanes: 129,
        bounds: Interval::new(-ONE, ONE).unwrap(),
    }];
    let make = |offset| {
        [body::Site {
            left: body::View {
                offset,
                ..body::View::contiguous(0, 128)
            },
            right: body::View::contiguous(0, 128),
            operation: body::Binary::Add,
        }]
    };
    let outputs = [body::View::contiguous(1, 128)];
    let a = body::Plan::new(WeightMode::Public, &input, &make(0), &outputs).unwrap();
    let b = body::Plan::new(WeightMode::Public, &input, &make(1), &outputs).unwrap();
    assert_ne!(a.digest(), b.digest());
    assert!(body::Plan::new(WeightMode::Public, &input, &make(2), &outputs).is_err());
    assert!(body::Plan::new(WeightMode::Public, &input, &make(usize::MAX), &outputs).is_err());
    let mut forged = make(0);
    forged[0].left.owner = 1;
    assert!(body::Plan::new(WeightMode::Public, &input, &forged, &outputs).is_err());
}

#[test]
fn body_unit_protocol_failure_and_cancellation_drop_local_registers() {
    let inputs = [body::Register {
        lanes: 129,
        bounds: Interval::new(-ONE, ONE).unwrap(),
    }; 2];
    let sites = [body::Site {
        left: body::View::contiguous(0, 129),
        right: body::View::contiguous(1, 129),
        operation: body::Binary::Multiply,
    }];
    let outputs = [body::View::contiguous(2, 129)];
    let plan = Arc::new(body::Plan::new(WeightMode::Public, &inputs, &sites, &outputs).unwrap());
    let fixture = expanded(&inputs, &sites, &outputs);
    let clear = BTreeMap::from([("x0".into(), vec![ONE; 129]), ("x1".into(), vec![ONE; 129])]);
    let mut owners =
        workers(fixture, &clear).map(|worker| body::from_test_worker(plan.clone(), worker));
    owners[0].advance().unwrap();
    assert!(owners[0].receive(b"invalid", puma::Party::P1).is_err());
    assert_eq!(owners[0].live_bytes(), 0);
    assert!(matches!(owners[0].advance(), Err(Error::Retired)));
    owners[1].cancel();
    owners[1].cancel();
    assert_eq!(owners[1].live_bytes(), 0);
    assert!(matches!(owners[1].advance(), Err(Error::Retired)));
    assert!(!owners[1].complete());
}
