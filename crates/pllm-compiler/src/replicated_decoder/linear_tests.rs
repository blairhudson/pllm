use super::*;
use crate::replicated_decoder::tests::{execute, workers};

fn compile(
    rows: usize,
    inner: usize,
    columns: usize,
    bounds: InputBounds,
    rescale: bool,
) -> Result<Arc<Program>, Error> {
    let mut b = Builder {
        nodes: vec![],
        allocated: 0,
    };
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
