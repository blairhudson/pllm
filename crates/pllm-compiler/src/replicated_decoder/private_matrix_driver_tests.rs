//! Ordinary Builder delegate versus independent PRE-DELEGATE loop.
//! Primitive controls only: no Source12, compact publication or runtime issuer.
use super::*;

fn fixture(
    rows: usize,
    inner: usize,
    columns: usize,
    mass: bool,
    proof: bool,
) -> (Builder, View, View) {
    let mut builder = Builder::default();
    let mut words = Words::new();
    for row in 0..rows {
        let radius = i64::try_from(row + 1).unwrap();
        let id = builder
            .push(
                Code::Input(format!("left.{row}")),
                inner,
                Interval::new(-radius, radius).unwrap(),
            )
            .unwrap();
        let mut view = builder.view(id, vec![inner]);
        if mass {
            let input = builder.gather(&view.words).unwrap();
            let (native, actual_mass) =
                tensor::Program::softmax_certified(1, inner, builder.nodes[input].bounds).unwrap();
            let output = builder.tensor(input, native).unwrap();
            builder.row_masses.push((output, actual_mass));
            view = builder.view(output, vec![inner]);
        }
        if proof {
            // Derived from ACTUAL original node intervals, not an asserted
            // checkpoint/producer certificate or general->attention promotion.
            let energy = builder.scalar_row_energy(&view.words).unwrap();
            builder
                .record_attention_energy(&view.words, energy)
                .unwrap();
        }
        words.extend(view.words.iter());
    }
    let left = View {
        words,
        shape: vec![1, rows, inner],
    };
    let id = builder
        .push(
            Code::Input("right".into()),
            columns * inner,
            Interval::new(-2, 3).unwrap(),
        )
        .unwrap();
    let right = builder.view(id, vec![columns, inner]);
    if proof {
        for row in right.words.chunks_exact(inner) {
            let energy = builder.scalar_row_energy(&row).unwrap();
            builder.record_attention_energy(&row, energy).unwrap();
        }
    }
    (builder, left, right)
}

/// Literal original arm. Does NOT call driver, with_mass, or delegated matrix.
fn before_delegate(
    builder: &mut Builder,
    input: &View,
    weight: &MatrixView<'_>,
    inner: usize,
    columns: usize,
    rescale: bool,
    attention: bool,
) -> Result<View, Error> {
    let rows = input.words.len() / inner;
    let tile = (puma::MAX_LANES / inner).min(128);
    let mut words = Words::new();
    for row in input.words.chunks_exact(inner) {
        let mass = builder.row_certificate(&row, inner, &builder.row_masses);
        let left = builder.gather(&row)?;
        for start in (0..columns).step_by(tile) {
            let n = tile.min(columns - start);
            let right = weight.gather(builder, start, n, 0, inner)?;
            let norm = if attention && !builder.attention_energies.is_empty() {
                builder.attention_dot(&row, weight, start, n, inner, left, right)?
            } else {
                None
            };
            let output = builder.product_with_norm(
                left,
                right,
                Some(MatrixShape::new(1, inner, n)?),
                rescale,
                norm,
            )?;
            if let Some(mass) = mass {
                let values = builder.nodes[right].bounds;
                let bound = |v: i64| {
                    i64::try_from(i128::from(v) * i128::from(mass)).map_err(|_| Error::Bounds)
                };
                let mut range =
                    Interval::new(bound(values.lower().min(0))?, bound(values.upper().max(0))?)?;
                if rescale {
                    range = Interval::new(
                        range.lower().div_euclid(ONE),
                        -(-range.upper()).div_euclid(ONE),
                    )?;
                }
                let ordinary = builder.nodes[output].bounds;
                builder.nodes[output].bounds = Interval::new(
                    ordinary.lower().max(range.lower()),
                    ordinary.upper().min(range.upper()),
                )?;
            }
            words.extend((0..n).map(|index| (output, index)));
        }
    }
    Ok(View {
        words,
        shape: vec![1, rows, columns],
    })
}

fn compare(actual: &Builder, expected: &Builder) {
    assert_eq!(actual.nodes.len(), expected.nodes.len());
    assert_eq!(actual.allocated, expected.allocated);
    assert_eq!(actual.row_masses, expected.row_masses);
    assert_eq!(actual.attention_energies, expected.attention_energies);
    assert_eq!(actual.row_energies, expected.row_energies);
    assert_eq!(
        schedule_work(&actual.nodes).unwrap(),
        schedule_work(&expected.nodes).unwrap()
    );
    for (a, b) in actual.nodes.iter().zip(&expected.nodes) {
        assert_eq!((a.lanes, a.bounds), (b.lanes, b.bounds));
        match (&a.code, &b.code) {
            (Code::Input(a), Code::Input(b)) => assert_eq!(a, b),
            (Code::Gather(a), Code::Gather(b)) => assert!(a.iter().eq(b.iter())),
            (
                Code::Product {
                    left: al,
                    right: ar,
                    shape: ash,
                    rescale: asc,
                    norm: an,
                },
                Code::Product {
                    left: bl,
                    right: br,
                    shape: bsh,
                    rescale: bsc,
                    norm: bn,
                },
            ) => {
                assert_eq!((al, ar, ash, asc), (bl, br, bsh, bsc));
                assert_eq!(an.as_deref(), bn.as_deref());
            }
            (
                Code::Tensor {
                    input: a,
                    program: ap,
                },
                Code::Tensor {
                    input: b,
                    program: bp,
                },
            ) => {
                assert_eq!(a, b);
                assert_eq!(ap.digest(), bp.digest());
            }
            _ => panic!("unexpected original primitive recipe"),
        }
    }
}

#[test]
fn full8_tail3_preserves_original_proofs_shapes_order_and_49_work() {
    let inner = puma::MAX_LANES / 8;
    assert_eq!((puma::MAX_LANES / inner).min(128), 8);
    for rescale in [false, true] {
        for proof in [false, true] {
            let (mut actual, a, b) = fixture(2, inner, 11, false, proof);
            let (mut expected, x, y) = fixture(2, inner, 11, false, proof);
            let first = actual.nodes.len();
            let output = actual
                .matrix_with_attention(&a, MatrixView::Values(&b), rescale, true)
                .unwrap();
            let oracle = before_delegate(
                &mut expected,
                &x,
                &MatrixView::Values(&y),
                inner,
                11,
                rescale,
                true,
            )
            .unwrap();
            compare(&actual, &expected);
            assert_eq!(output.shape, oracle.shape);
            assert!(output.words.iter().eq(oracle.words.iter()));
            assert_eq!(actual.nodes.len() - first, 10);
            let mut row_proofs = Vec::new();
            for row in 0..2 {
                for (tile, lanes) in [(0, 8), (1, 3)] {
                    let ordinal = first + row * 5 + 2 + tile * 2;
                    let Code::Product {
                        left,
                        right,
                        shape,
                        norm,
                        rescale: scaled,
                    } = &actual.nodes[ordinal].code
                    else {
                        panic!()
                    };
                    assert_eq!((*left, *right), (first + row * 5, ordinal - 1));
                    assert_eq!(*shape, Some(MatrixShape::new(1, inner, lanes).unwrap()));
                    assert_eq!(*scaled, rescale);
                    assert_eq!(
                        norm.is_some(),
                        proof,
                        "actual per-call attention proof cannot disappear or be invented"
                    );
                    if tile == 0 {
                        row_proofs.push(norm.as_deref().map(|proof| proof.digest()));
                    }
                }
            }
            if proof {
                assert_ne!(
                    row_proofs[0], row_proofs[1],
                    "different actual original row bounds remain distinct"
                );
            }
            let per_row = WorkBudget::product(8, rescale)
                .unwrap()
                .checked_add(WorkBudget::product(3, rescale).unwrap())
                .unwrap();
            assert_eq!(
                schedule_work(&actual.nodes).unwrap(),
                per_row.checked_mul(2).unwrap()
            );
            if rescale {
                assert_eq!(
                    per_row.cursors().aes_work,
                    49,
                    "full8 + tail3 are separate calls, not merged11"
                );
            }
        }
    }
}

#[test]
fn actual_factory_mass_matches_original_intersection_both_scalings() {
    for rescale in [false, true] {
        let (mut actual, a, b) = fixture(2, 3, 11, true, false);
        let (mut expected, x, y) = fixture(2, 3, 11, true, false);
        let masses = actual.row_masses.clone();
        let output = actual
            .matrix_with_attention(&a, MatrixView::Values(&b), rescale, false)
            .unwrap();
        let oracle = before_delegate(
            &mut expected,
            &x,
            &MatrixView::Values(&y),
            3,
            11,
            rescale,
            false,
        )
        .unwrap();
        compare(&actual, &expected);
        assert!(output.words.iter().eq(oracle.words.iter()));
        assert_eq!(actual.row_masses, masses);
        assert!(actual.row_energies.is_empty() && actual.attention_energies.is_empty());
    }
}

#[test]
fn actual_private_source_underflow_preserves_original_error_and_partial_gather() {
    let (mut actual, a, b) = fixture(1, 16, 3, false, false);
    let (mut expected, x, y) = fixture(1, 16, 3, false, false);
    let source = |owner| MatrixView::Source {
        owner,
        columns: 11,
        inner: 16,
        column_stride: 16,
        inner_stride: 1,
    };
    let error = actual
        .matrix_with_attention(&a, source(b.words.at(0).0), false, false)
        .err()
        .unwrap();
    let original = before_delegate(
        &mut expected,
        &x,
        &source(y.words.at(0).0),
        16,
        11,
        false,
        false,
    )
    .err()
    .unwrap();
    assert_eq!(error, original);
    assert_eq!(
        actual.nodes.len(),
        3,
        "left gather persists; right source validation must fail before Product"
    );
    compare(&actual, &expected);
}
