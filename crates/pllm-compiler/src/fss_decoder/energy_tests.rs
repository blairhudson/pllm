use super::*;
use crate::fss_decoder::source::TensorImport;

#[test]
fn provenance_rejects_foreign_source_shape_rewire_and_forged_energy() {
    let mut ledger = Ledger::new([7; 32], "semantic-plan", 2).unwrap();
    let mut node = program::Node {
        code: program::Code::Embedding {
            weight: 0,
            positions: false,
        },
        shape: vec![1, 2, 2],
        range: Range::new(-3, 3).unwrap(),
        last_use: 1,
    };
    node.range = ledger
        .record(node.code, &node.shape, node.range, Some(8), None, None)
        .unwrap();
    assert_eq!(ledger.row([7; 32], 0, &node).unwrap(), 8);
    assert!(ledger.row([8; 32], 0, &node).is_err());
    node.shape = vec![1, 1, 4];
    assert!(ledger.row([7; 32], 0, &node).is_err());
    node.shape = vec![1, 2, 2];
    node.code = program::Code::Embedding {
        weight: 1,
        positions: false,
    };
    assert!(ledger.row([7; 32], 0, &node).is_err());
    node.code = program::Code::Embedding {
        weight: 0,
        positions: false,
    };
    ledger.entries[0].squared = 1;
    assert!(ledger.row([7; 32], 0, &node).is_err());
    assert!(ledger
        .record(
            node.code,
            &[1, 2],
            Range::new(2, 3).unwrap(),
            Some(0),
            None,
            None
        )
        .is_err());
}

#[test]
fn source_certificate_comes_from_every_decoded_row() {
    let values = [1.0f32, -2.0, 0.5, 3.0, -0.25, 0.0];
    let mut import = TensorImport::new("matrix", &[2, 3], Capacity::Decoder).unwrap();
    // Deliberately cross row boundaries between independently bounded frames.
    for chunk in values.chunks(2) {
        let bytes: Vec<_> = chunk
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        import.append(WeightDtype::F32, &bytes).unwrap();
    }
    let mut tensor = import.finish().unwrap();
    let certificate = tensor.matrix_certificate().unwrap().unwrap();
    let a = integer::row_energy(&[4096, -8192, 2048]).unwrap();
    let b = integer::row_energy(&[12288, -1024, 0]).unwrap();
    assert_eq!(tensor.maximum_row_energy().unwrap(), Some(a.max(b)));
    assert_eq!(certificate.frobenius_squared(), a + b);
    assert_eq!(certificate.maximum_row_squared_energy(), a.max(b));
    tensor.shape = vec![3, 2];
    assert!(tensor.maximum_row_energy().is_err());
    assert!(tensor.matrix_certificate().is_err());
    let layout =
        source::certified_file_layout(&[integer::MAX_ROWS + 1, 3], Capacity::Decoder).unwrap();
    assert_eq!(
        layout.certificate_mode(),
        pllm_core::fixed_file::CertificateMode::RowEnergy
    );
    assert_eq!(layout.resources().certificate_column_bytes, 0);
    assert!(source::certified_file_layout(&[2, 3], Capacity::Capsule).is_err());
}

#[test]
fn integer_linear_bias_affine_and_transpose_bounds_enclose_rows() {
    let weights = Tensor::from_q12("w", &[2, 2], &[2500, -1500, 2000, 500]).unwrap();
    let gamma = Tensor::from_q12("g", &[2], &[7000, -3000]).unwrap();
    let beta = Tensor::from_q12("b", &[2], &[2, -1]).unwrap();
    for x in -5i64..=5 {
        for y in -5i64..=5 {
            let row = [x, y];
            let input = integer::row_energy(&row).unwrap();
            for transposed in [false, true] {
                let (a, b) = if transposed {
                    (2000, -1500)
                } else {
                    (-1500, 2000)
                };
                let output = [
                    (2500 * x + a * y).div_euclid(4096) + 2,
                    (b * x + 500 * y).div_euclid(4096) - 1,
                ];
                assert!(
                    integer::row_energy(&output).unwrap()
                        <= linear(input, &weights, transposed, Some(&beta))
                            .unwrap()
                            .unwrap()
                );
            }
            let affine_output = [
                (7000 * x).div_euclid(4096) + 2,
                (-3000 * y).div_euclid(4096) - 1,
            ];
            assert!(
                integer::row_energy(&affine_output).unwrap()
                    <= affine(input, &gamma, Some(&beta)).unwrap().unwrap()
            );
        }
    }
    assert!(linear(1, &weights, false, Some(&gamma)).is_ok());
    let wrong_bias = Tensor::from_q12("bad", &[1], &[0]).unwrap();
    assert!(linear(1, &weights, false, Some(&wrong_bias)).is_err());
}

#[test]
fn cauchy_admits_exact_dot_without_scalar_product_sum_overflow() {
    let range = Range::new(-(1i128 << 24), 1i128 << 24).unwrap();
    assert!(range.product(range).and_then(|term| term.scale(2)).is_err());
    let (raw, proof) = dot_bounds(range, range, 2, Some((1 << 48, 1 << 48))).unwrap();
    assert_eq!(raw, Range::new(-(1i128 << 48), 1i128 << 48).unwrap());
    assert_eq!(proof.unwrap().inner, 2);
    assert!(dot_bounds(Range::new(4, 5).unwrap(), range, 2, Some((1, 1 << 48))).is_err());
    // Two different positions each contain a unit row. Their gathered column has
    // energy two, even though every original row has energy one.
    assert_eq!(
        gathered_column(1, Range::new(-1, 1).unwrap(), 2).unwrap(),
        2
    );
    let output = [2i64];
    assert!(
        integer::row_energy(&output).unwrap()
            <= values_output(2 * 4096u128.pow(2), 1, 2, 1).unwrap()
    );
}

#[test]
fn two_floor_rotary_preserves_signed_floor_witness() {
    let descriptor = RotaryCoefficients::from_f32(
        "rope",
        10000,
        [2, 2],
        &[1.0, 1.0, 3000.0 / 4096.0, 3000.0 / 4096.0],
        &[0.0, 0.0, 1200.0 / 4096.0, 1200.0 / 4096.0],
    )
    .unwrap();
    for x in -8i64..=8 {
        for y in -8i64..=8 {
            let output = [
                (x * 3000).div_euclid(4096) + (-y * 1200).div_euclid(4096),
                (y * 3000).div_euclid(4096) + (x * 1200).div_euclid(4096),
            ];
            let input = integer::row_energy(&[x, y]).unwrap();
            assert!(
                integer::row_energy(&output).unwrap()
                    <= rotary(input, &descriptor, 2).unwrap().unwrap()
            );
        }
    }
}

#[test]
fn pending_qwen_norm_input_fails_before_any_conditional_output_proof() {
    use pllm_garble::fusefss_backend::{composition::cache, Budget};
    let budget = Budget::new(8 << 20).unwrap();
    let tables = cache::SourceTableCache::new(
        [91; 32],
        cache::Limits {
            max_entries: cache::MAX_ENTRIES,
            max_retained_bytes: cache::MAX_RETAINED_BYTES,
        },
        &budget,
    )
    .unwrap();
    let gamma = Tensor::from_q12("gamma", &[896], &vec![4096; 896]).unwrap();
    for (lo, hi, energy) in [
        (-824, 680, 10_202_097u128),
        (-1_590_281, 1_591_012, 235_198_324_975_329),
    ] {
        admit_norm(
            [91; 32],
            Range::new(lo, hi).unwrap(),
            energy,
            &gamma,
            None,
            false,
            &tables,
        )
        .unwrap();
    }
    // The log's first not-yet-admitted RMS input is node 29. Its individual
    // square already fails, before gamma/beta or conditional output bounds.
    let range = Range::new(-2_648_483_429, 2_648_484_225).unwrap();
    let energy = 1_678_161_750_782_077_346_724;
    assert_eq!(i128::from(range.hi).pow(2), 7_014_468_690_073_850_625);
    for gamma in [
        gamma,
        Tensor::from_q12("zero", &[896], &vec![0; 896]).unwrap(),
    ] {
        assert!(matches!(
            admit_norm([91; 32], range, energy, &gamma, None, false, &tables),
            Err(Error::Domain)
        ));
    }
    assert!(pllm_garble::fusefss_backend::composition::norm::bounds::source_bounds(896).is_ok());
}

#[test]
fn coordinate_product_energy_intersects_scalar_and_complete_row_constraints() {
    let range = Range::new(-4097, 4097).unwrap();
    for a in [-4097i64, -1, 0, 1, 4097] {
        for b in [-4097i64, -1, 0, 1, 4097] {
            let left = [a, b];
            let right = [b, a];
            let output = [(a * b).div_euclid(4096); 2];
            let bound = product_energy(
                range,
                integer::row_energy(&left).unwrap(),
                range,
                integer::row_energy(&right).unwrap(),
                2,
            )
            .unwrap()
            .unwrap();
            assert!(integer::row_energy(&output).unwrap() <= bound);
        }
    }
    let bound = product_energy(
        Range::new(-65536, 314025).unwrap(),
        268060458992704,
        Range::new(-202042, 202041).unwrap(),
        87395349708304,
        4864,
    )
    .unwrap()
    .unwrap();
    assert_eq!(bound, 513685092886158400);
    assert!(bound < 1167039262674553600);
}

#[test]
fn weighted_source_affine_bounds_preserve_both_signed_floors_and_beta() {
    let source = Source::new(vec![
        Tensor::from_q12("beta", &[2], &[2, -3]).unwrap(),
        Tensor::from_q12("gamma", &[2], &[1, -4095]).unwrap(),
        Tensor::from_q12("w", &[3, 2], &[8192, -8000, 1, 4096, -12288, 3]).unwrap(),
    ])
    .unwrap();
    let node = program::Node {
        code: program::Code::Norm {
            input: 0,
            gamma: 1,
            beta: Some(0),
            layer: false,
        },
        shape: vec![1, 1, 2],
        range: Range::new(-16384, 16384).unwrap(),
        last_use: 4,
    };
    let proof = AffineLinear::derive(&source, 3, &node, 2, 8 << 20)
        .unwrap()
        .unwrap();
    assert!(proof.matches(3, &node));
    assert!(!proof.matches(4, &node));
    let bias = Tensor::from_q12("bias", &[3], &[-3, 4, -5]).unwrap();
    let bound = proof.output_energy(Some(&bias)).unwrap().unwrap();
    for a in [-4097i64, -64, -1, 0, 1, 2048, 4095, 4097] {
        for b in [-4097i64, -64, -1, 0, 1, 2048, 4095, 4097] {
            let q = [a, b];
            assert!(integer::row_energy(&q).unwrap() <= proof.normalized);
            let x = [a.div_euclid(4096) + 2, (-4095 * b).div_euclid(4096) - 3];
            let mut output = Vec::new();
            for row in 0..3 {
                let mut weights = [0; 2];
                source.tensors[2].read_row(row, 0, &mut weights).unwrap();
                let stats = source.tensors[2].row_statistics(row).unwrap();
                let (lo, hi) = proof.raw_bounds(stats, true).unwrap();
                let mut raw = 0i128;
                for (&w, &v) in weights.iter().zip(&x) {
                    let term = i128::from(w) * i128::from(v);
                    assert!((lo..=hi).contains(&term));
                    raw += term;
                    assert!((lo..=hi).contains(&raw));
                }
                let range = public_raw_with_affine(node.range, stats, None, Some(proof)).unwrap();
                assert!((i128::from(range.lo)..=i128::from(range.hi)).contains(&raw));
                output.push((raw.div_euclid(4096) + [-3, 4, -5][row]) as i64);
            }
            assert!(integer::row_energy(&output).unwrap() <= bound);
        }
    }
    // Genuine fusion counterexamples: source execution must keep both floors.
    for q in [2048i128, -2048] {
        let separate = (q.div_euclid(4096) * 8192).div_euclid(4096);
        let fused = (q * 8192).div_euclid(4096 * 4096);
        assert_ne!(separate, fused);
    }
}

#[test]
fn weighted_certificate_avoids_unrelated_gamma_outlier_and_rejects_shape() {
    let source = Source::new(vec![
        Tensor::from_q12("g", &[2], &[40960, 1]).unwrap(),
        Tensor::from_q12("w", &[2, 2], &[0, 4096, 0, -4096]).unwrap(),
    ])
    .unwrap();
    let node = program::Node {
        code: program::Code::Norm {
            input: 0,
            gamma: 0,
            beta: None,
            layer: false,
        },
        shape: vec![1, 1, 2],
        range: Range::new(-60000, 60000).unwrap(),
        last_use: 1,
    };
    let proof = AffineLinear::derive(&source, 0, &node, 1, 8 << 20)
        .unwrap()
        .unwrap();
    let old = affine(proof.normalized, &source.tensors[0], None)
        .unwrap()
        .unwrap();
    let old = linear(old, &source.tensors[1], false, None)
        .unwrap()
        .unwrap();
    assert!(proof.output_energy(None).unwrap().unwrap() * 1_000_000 < old);
    let bad = Tensor::from_q12("bad", &[1], &[4096]).unwrap();
    assert!(source.tensors[1]
        .weighted_columns(&bad, None, 8 << 20)
        .is_err());
}
