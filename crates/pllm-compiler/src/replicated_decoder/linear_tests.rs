use super::*;
use crate::replicated_decoder::tests::{execute, workers, workers_with_programs};

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
    let work = b
        .nodes
        .iter()
        .map(|n| match &n.code {
            Code::Product { .. } => 1,
            Code::Tensor { program, .. } => program.instructions() as u64 * 128,
            _ => 0,
        })
        .sum();
    Ok(Arc::new(Program {
        retire: vec![vec![]; b.nodes.len()],
        nodes: b.nodes,
        inputs: 2,
        outputs: out.words,
        digest: [71; 32],
        payload_bytes: b.allocated,
        operation_limit: work,
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
                if items.iter().any(|&(id, _)| id == source.owner)))
            );
            let work = b
                .nodes
                .iter()
                .filter(|n| matches!(n.code, Code::Product { .. }))
                .count() as u64;
            let program = Arc::new(Program {
                retire: vec![vec![]; b.nodes.len()],
                nodes: b.nodes,
                inputs: 2,
                outputs: output.words,
                digest: [73; 32],
                payload_bytes: b.allocated,
                operation_limit: work,
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
                operation_limit: program.operation_limit,
            };
            for node in &mut expanded.nodes {
                if let Code::SourceGather(view) = node.code {
                    let items: Vec<_> = view.indices().map(|i| (view.owner, i)).collect();
                    expanded.payload_bytes +=
                        items.capacity() * std::mem::size_of::<(usize, usize)>();
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
