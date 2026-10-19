use super::*;
use crate::replicated_decoder::tests::{execute, workers, workers_with_programs};

fn selector_program(vocabulary: usize) -> Arc<Program> {
    let mut b = Builder::default();
    let token = b
        .push(
            Code::Input("token".into()),
            1,
            Interval::new(0, vocabulary as i64 - 1).unwrap(),
        )
        .unwrap();
    let bound = Interval::new(-(vocabulary as i64 - 1), vocabulary as i64 - 1).unwrap();
    let output = b
        .tensor_loop(
            LoopInput::TokenOffsets {
                owner: token,
                index: 0,
            },
            vocabulary,
            |n| tensor::Program::equal_zero(n, bound),
        )
        .unwrap();
    let work = schedule_work(&b.nodes).unwrap();
    let mut retire = vec![vec![]; b.nodes.len()];
    retire[output].push(token);
    Arc::new(Program {
        retire,
        nodes: b.nodes,
        inputs: 1,
        outputs: (0..vocabulary).map(|i| (output, i)).collect(),
        digest: [97; 32],
        payload_bytes: b.allocated,
        work,
    })
}

#[test]
fn compact_private_selectors_cover_every_lane_and_tail_under_one_live_loop() {
    for vocabulary in [1, 128, 129, 257, 513] {
        let program = selector_program(vocabulary);
        assert_eq!(program.nodes.len(), 2);
        for token in [0, vocabulary / 2, vocabulary - 1] {
            let actual = execute(workers(
                program.clone(),
                &BTreeMap::from([("token".into(), vec![token as i64])]),
            ));
            assert_eq!(
                actual,
                (0..vocabulary)
                    .map(|i| u64::from(i == token))
                    .collect::<Vec<_>>()
            );
        }
    }
    let full = selector_program(151936);
    assert!(full.payload_bytes() < 4 << 20);
    // One full lookup has 1,187 complete 128-lane invocations, each with
    // 28 top-level ordinals. Nested gate rounds retain their separate AES cost.
    assert_eq!(full.operation_limit(), 1187 * 28);
    assert_eq!(full.work_budget().cursors().aes_work, 1187 * 2817);
    full.setup([43; 32], puma::Party::P0).unwrap().cancel();
    assert!(full
        .work_budget()
        .checked_mul(32)
        .unwrap()
        .limits()
        .is_err());
}

#[test]
fn replay_between_loop_iterations_burns_partial_outputs_and_runtime() {
    let program = selector_program(257);
    let mut owners = workers(program, &BTreeMap::from([("token".into(), vec![128])]));
    let mut queue = Vec::new();
    let mut first = None;
    for _ in 0..10000 {
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
            for (from, to, frame) in std::mem::take(&mut queue) {
                first.get_or_insert_with(|| (from, to, frame.as_bytes().to_vec()));
                queue.extend(
                    owners[to as usize]
                        .receive(frame.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, frame)| (to, next, frame)),
                );
            }
        }
        if owners.iter().all(|w| w.loop_test_state().0 == 128) {
            break;
        }
    }
    assert!(owners.iter().all(|w| w.loop_test_state().0 == 128));
    assert!(owners.iter().all(|w| w.loop_test_state().1 == 2));
    let (from, to, frame) = first.unwrap();
    assert!(owners[to as usize].receive(&frame, from).is_err());
    assert!(owners[to as usize].retired());
    assert_eq!(owners[to as usize].loop_test_state(), (0, 0));
    assert!(owners[to as usize].advance().is_err());
    for owner in &mut owners {
        owner.cancel();
        assert_eq!(owner.loop_test_state(), (0, 0));
    }
}

#[test]
fn batched_private_argmax_preserves_lowest_index_ties_and_odd_tails() {
    for (n, maxima) in [(1, vec![0]), (257, vec![256]), (513, vec![127, 128, 512])] {
        let mut logits: Vec<i64> = (0..n).map(|i| -(i as i64) - 100).collect();
        for &index in &maxima {
            logits[index] = 50;
        }
        let mut builder = Builder::default();
        let input = builder
            .push(Code::Input("logits".into()), n, interval(&logits))
            .unwrap();
        let output = builder.argmax_tree(&builder.view(input, vec![n])).unwrap();
        let work = schedule_work(&builder.nodes).unwrap();
        let program = Arc::new(Program {
            retire: vec![vec![]; builder.nodes.len()],
            nodes: builder.nodes,
            inputs: 1,
            outputs: output.words.iter().collect(),
            digest: [95; 32],
            payload_bytes: builder.allocated,
            work,
        });
        assert_eq!(
            execute(workers(
                program,
                &BTreeMap::from([("logits".into(), logits)])
            )),
            [*maxima.iter().min().unwrap() as u64]
        );
    }
}

#[test]
fn paged_public_matrix_products_cross_disk_chunks_and_truncate_complete_sums() {
    for (inner, columns) in [(1025, 17), (4864, 3), (65, 129)] {
        for transposed in [false, true] {
            let rows = 2;
            let x: Vec<i64> = (0..rows * inner)
                .map(|i| (i as i64 % 37 - 18) * 257)
                .collect();
            let w: Vec<i64> = (0..columns * inner)
                .map(|i| (i as i64 % 29 - 14) * 19)
                .collect();
            for rescale in [false, true] {
                let mut b = Builder::default();
                let digest =
                    Sha256::digest(w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                        .into();
                let source = b
                    .push(
                        Code::PublicInput {
                            key: "w".into(),
                            digest,
                        },
                        w.len(),
                        interval(&w),
                    )
                    .unwrap();
                let input = b
                    .push(Code::Input("x".into()), x.len(), interval(&x))
                    .unwrap();
                let source = SourceView {
                    owner: source,
                    shape: if transposed {
                        vec![inner, columns]
                    } else {
                        vec![columns, inner]
                    },
                };
                let out = b
                    .matrix(
                        &b.view(input, vec![rows, inner]),
                        MatrixView::source(&source, transposed).unwrap(),
                        rescale,
                    )
                    .unwrap();
                assert!(!b
                    .nodes
                    .iter()
                    .any(|n| matches!(n.code, Code::Product { .. })));
                let work = schedule_work(&b.nodes).unwrap();
                let program = Arc::new(Program {
                    retire: vec![vec![]; b.nodes.len()],
                    nodes: b.nodes,
                    inputs: 2,
                    outputs: out.words.iter().collect(),
                    digest: [93; 32],
                    payload_bytes: b.allocated,
                    work,
                });
                let actual = execute(workers(
                    program,
                    &BTreeMap::from([("x".into(), x.clone()), ("w".into(), w.clone())]),
                ));
                for r in 0..rows {
                    for c in 0..columns {
                        let raw: i128 = (0..inner)
                            .map(|k| {
                                i128::from(x[r * inner + k])
                                    * i128::from(
                                        w[if transposed {
                                            k * columns + c
                                        } else {
                                            c * inner + k
                                        }],
                                    )
                            })
                            .sum();
                        let got = i128::from(actual[r * columns + c] as i64);
                        if rescale {
                            let floor = raw.div_euclid(i128::from(ONE));
                            assert!(
                                got == floor
                                    || got
                                        == floor + i128::from(raw.rem_euclid(i128::from(ONE)) != 0)
                            );
                        } else {
                            assert_eq!(got, raw);
                        }
                    }
                }
            }
        }
    }
}

fn compile(
    rows: usize,
    inner: usize,
    columns: usize,
    bounds: InputBounds,
    rescale: bool,
) -> Result<Arc<Program>, Error> {
    let mut b = Builder::default();
    let x = b.push(Code::Input("x".into()), rows * inner, bounds.left)?;
    let w = b.push(Code::Input("w".into()), columns * inner, bounds.right)?;
    let out = b.linear(
        &b.view(x, vec![rows, inner]),
        &b.view(w, vec![columns, inner]),
        rescale,
    )?;
    let work = schedule_work(&b.nodes)?;
    Ok(Arc::new(Program {
        retire: vec![vec![]; b.nodes.len()],
        nodes: b.nodes,
        inputs: 2,
        outputs: out.words.iter().collect(),
        digest: [71; 32],
        payload_bytes: b.allocated,
        work,
    }))
}

fn interval(x: &[i64]) -> Interval {
    Interval::new(*x.iter().min().unwrap(), *x.iter().max().unwrap()).unwrap()
}

#[test]
fn wide_private_dots_preserve_complete_integer_reductions_and_one_rounding() {
    // Real decoder widths and an incomplete final chunk. Both signed raw ring
    // results and the source stochastic Q18 law use an independent i128 oracle.
    for inner in [1025, 4864] {
        let (rows, columns) = (2, 3);
        let x: Vec<i64> = (0..rows * inner)
            .map(|i| (i as i64 % 41 - 20) * 257)
            .collect();
        let w: Vec<i64> = (0..columns * inner)
            .map(|i| (i as i64 % 29 - 14) * 19)
            .collect();
        let raw: Vec<i128> = (0..rows)
            .flat_map(|r| {
                let x = &x;
                let w = &w;
                (0..columns).map(move |c| {
                    (0..inner)
                        .map(|k| i128::from(x[r * inner + k]) * i128::from(w[c * inner + k]))
                        .sum()
                })
            })
            .collect();
        for rescale in [false, true] {
            let program = compile(
                rows,
                inner,
                columns,
                InputBounds {
                    left: interval(&x),
                    right: interval(&w),
                },
                rescale,
            )
            .unwrap();
            assert!(program
                .nodes
                .iter()
                .all(|n| !matches!(n.code, Code::Product { rescale: true, .. })));
            assert_eq!(
                program
                    .nodes
                    .iter()
                    .filter(|n| matches!(n.code, Code::Tensor { .. }))
                    .count(),
                if rescale { rows * columns } else { 0 }
            );
            let clear = BTreeMap::from([("x".into(), x.clone()), ("w".into(), w.clone())]);
            let actual = execute(workers(program, &clear));
            for (out, expected) in actual.iter().zip(&raw) {
                let got = i128::from(*out as i64);
                if rescale {
                    let floor = expected.div_euclid(i128::from(ONE));
                    let ceil = floor + i128::from(expected.rem_euclid(i128::from(ONE)) != 0);
                    assert!(got == floor || got == ceil, "{got} not in {floor}..={ceil}");
                } else {
                    assert_eq!(got, *expected);
                }
            }
        }
    }
}

#[test]
fn half_units_across_chunk_boundary_sum_exactly_before_truncation() {
    let inner = 1025;
    let mut x = vec![0; inner];
    let mut w = vec![0; inner];
    x[0] = 1;
    x[1024] = 1;
    w[0] = ONE / 2;
    w[1024] = ONE / 2;
    let program = compile(
        1,
        inner,
        1,
        InputBounds {
            left: interval(&x),
            right: interval(&w),
        },
        true,
    )
    .unwrap();
    let clear = BTreeMap::from([("x".into(), x), ("w".into(), w)]);
    for _ in 0..8 {
        // Independently rounded half units could yield 0 or 2. One complete
        // Q18 integer must always yield 1, for every fresh PRSS setup.
        assert_eq!(execute(workers(program.clone(), &clear)), [1]);
    }
}

#[test]
fn complete_accumulator_range_rejects_even_when_each_chunk_fits() {
    let bounds = InputBounds {
        left: Interval::new(0, 1 << 20).unwrap(),
        right: Interval::new(0, 1 << 31).unwrap(),
    };
    assert!(compile(1, 1024, 1, bounds, true).is_ok());
    assert!(compile(1, 2048, 1, bounds, true).is_err());
}

#[test]
fn compact_source_tiles_preserve_strided_transpose_and_partial_contractions() {
    for (inner, columns) in [(17, 7), (1025, 3), (4864, 3)] {
        for transposed in [false, true] {
            let rows = 2;
            let x: Vec<i64> = (0..rows * inner).map(|i| i as i64 % 47 - 23).collect();
            let w: Vec<i64> = (0..inner * columns).map(|i| i as i64 % 37 - 18).collect();
            let mut b = Builder::default();
            let input = b
                .push(Code::Input("x".into()), x.len(), interval(&x))
                .unwrap();
            let source = b
                .push(Code::Input("w".into()), w.len(), interval(&w))
                .unwrap();
            let source = SourceView {
                owner: source,
                shape: if transposed {
                    vec![inner, columns]
                } else {
                    vec![columns, inner]
                },
            };
            let output = b
                .matrix(
                    &b.view(input, vec![rows, inner]),
                    MatrixView::source(&source, transposed).unwrap(),
                    false,
                )
                .unwrap();
            assert!(b
                .nodes
                .iter()
                .any(|n| matches!(n.code, Code::SourceGather(_))));
            assert!(
                !b.nodes.iter().any(|n| matches!(&n.code, Code::Gather(items)
                if items.iter().any(|(id, _)| id == source.owner)))
            );
            let work = schedule_work(&b.nodes).unwrap();
            let program = Arc::new(Program {
                retire: vec![vec![]; b.nodes.len()],
                nodes: b.nodes,
                inputs: 2,
                outputs: output.words.iter().collect(),
                digest: [73; 32],
                payload_bytes: b.allocated,
                work,
            });
            // Keep one party on the original explicit-gather representation.
            // Contexts, operand bindings and fresh PRSS remain interoperable.
            let mut expanded = Program {
                nodes: program.nodes.clone(),
                retire: program.retire.clone(),
                inputs: program.inputs,
                outputs: program.outputs.clone(),
                digest: program.digest,
                payload_bytes: program.payload_bytes,
                work: program.work,
            };
            for node in &mut expanded.nodes {
                if let Code::SourceGather(view) = node.code {
                    let items: Words = view.indices().map(|i| (view.owner, i)).collect();
                    expanded.payload_bytes += items.metadata_bytes();
                    node.code = Code::Gather(items);
                }
            }
            assert!(expanded.payload_bytes <= MAX_PROGRAM_BYTES);
            let actual = execute(workers_with_programs(
                [program.clone(), Arc::new(expanded), program],
                &BTreeMap::from([("x".into(), x.clone()), ("w".into(), w.clone())]),
            ));
            for r in 0..rows {
                for c in 0..columns {
                    let expected: i128 = (0..inner)
                        .map(|k| {
                            i128::from(x[r * inner + k])
                                * i128::from(
                                    w[if transposed {
                                        k * columns + c
                                    } else {
                                        c * inner + k
                                    }],
                                )
                        })
                        .sum();
                    assert_eq!(i128::from(actual[r * columns + c] as i64), expected);
                }
            }
        }
    }
}

#[test]
fn compact_source_descriptors_reject_overflow_and_non_source_owners_before_nodes() {
    let mut b = Builder::default();
    let owner = b
        .push(Code::Input("w".into()), 32, Interval::new(-1, 1).unwrap())
        .unwrap();
    let valid = SourceGather {
        owner,
        offset: 1,
        rows: 3,
        columns: 4,
        row_stride: 8,
        column_stride: 2,
    };
    let id = b.source_gather(valid).unwrap();
    let before = (b.nodes.len(), b.allocated);
    for bad in [
        SourceGather { owner: id, ..valid },
        SourceGather {
            owner: usize::MAX,
            ..valid
        },
        SourceGather {
            offset: 32,
            ..valid
        },
        SourceGather {
            offset: usize::MAX,
            ..valid
        },
        SourceGather {
            row_stride: usize::MAX,
            ..valid
        },
        SourceGather {
            column_stride: usize::MAX,
            ..valid
        },
        SourceGather { rows: 0, ..valid },
        SourceGather {
            rows: usize::MAX,
            ..valid
        },
        SourceGather {
            columns: 0,
            ..valid
        },
        SourceGather {
            rows: 1025,
            columns: 1,
            row_stride: 0,
            ..valid
        },
    ] {
        assert!(b.source_gather(bad).is_err());
        assert_eq!((b.nodes.len(), b.allocated), before);
    }
    assert_eq!(
        valid.indices().collect::<Vec<_>>(),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    );
}
