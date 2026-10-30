use super::*;

fn plan(n: usize, q: u64) -> CoefficientCompression {
    CoefficientCompression::new(CoefficientRing::new(n, q).unwrap(), MAX_COMPRESSION_WORK).unwrap()
}

fn reconstruct(p: CoefficientCompression, input: &[u64]) {
    let n = input.len();
    let mut output = vec![u64::MAX; p.output_words()];
    let mut scratch = vec![u64::MAX; p.scratch_words()];
    p.decompress(input, &mut output, &mut scratch).unwrap();
    // Independent oracle: coefficient projection, not another trace tree.
    for s in 0..n {
        for i in 0..n {
            assert_eq!(output[s * n + i], if i == 0 { input[s] } else { 0 });
        }
    }
}

#[test]
fn exhaustive_tiny_rings_include_composite_moduli_and_wraps() {
    for (n, q) in [(1, 3), (2, 9), (4, 3)] {
        let p = plan(n, q);
        for mut code in 0..q.pow(n as u32) {
            let mut values = vec![0; n];
            for v in &mut values {
                *v = code % q;
                code /= q;
            }
            reconstruct(p, &values);
        }
    }
}

#[test]
fn basis_order_signs_padding_and_large_limb() {
    for n in [2, 8, 32] {
        for q in [15, (1 << 61) - 1] {
            let p = plan(n, q);
            for at in 0..n {
                let mut basis = vec![0; n];
                basis[at] = q - 1;
                reconstruct(p, &basis);
            }
            let mut packed = vec![99; n];
            p.compress(&[q - 1], &mut packed).unwrap();
            assert_eq!(packed[0], q - 1);
            assert!(packed[1..].iter().all(|&v| v == 0));
            reconstruct(p, &packed);
        }
    }
}

#[test]
fn mixed_canonical_polynomials_check_every_output_coefficient() {
    for n in [64, 128] {
        for q in [15, (1 << 61) - 1] {
            let p = plan(n, q);
            let input: Vec<u64> = (0..n)
                .map(|i| match i % 5 {
                    0 => 0,
                    1 => q - 1,
                    2 => q / 2,
                    3 => 1,
                    _ => ((i as u128 * i as u128 * 177 + 31) % u128::from(q)) as u64,
                })
                .collect();
            // Mixed residues exercise simultaneous trace cancellation, signed
            // wrap and coefficient ordering. The oracle checks all N^2 words.
            reconstruct(p, &input);
        }
    }
}

#[test]
fn maximum_degree_constructor_checks_work_and_storage_without_allocation() {
    let n = 4096_usize;
    let ring = CoefficientRing::new(n, (1 << 61) - 1).unwrap();
    let output_words = n.checked_mul(n).unwrap();
    let scratch_words = n.checked_mul(2).unwrap();
    let required_work = output_words
        .checked_mul(10)
        .unwrap()
        .checked_add(n.checked_mul(4).unwrap())
        .unwrap();
    assert_eq!(output_words, 16_777_216);
    assert_eq!(scratch_words, 8192);
    assert_eq!(scratch_words.checked_mul(8), Some(65_536));
    assert_eq!(required_work, 167_788_544);
    assert!(required_work <= MAX_COMPRESSION_WORK);
    for cap in [0, required_work - 1] {
        assert_eq!(
            CoefficientCompression::new(ring, cap),
            Err(LayoutError::ResourceLimit)
        );
    }
    let exact = CoefficientCompression::new(ring, required_work).unwrap();
    assert_eq!(exact.output_words(), output_words);
    assert_eq!(exact.scratch_words(), scratch_words);
    assert_eq!(exact.coefficient_work(), required_work);
    for cap in [required_work + 1, MAX_COMPRESSION_WORK, usize::MAX] {
        assert_eq!(CoefficientCompression::new(ring, cap), Ok(exact));
    }
    // Existing ring admission remains the execution-degree boundary.
    assert_eq!(
        CoefficientRing::new(n.checked_mul(2).unwrap(), 65537),
        Err(LayoutError::InvalidDegree)
    );
}

#[test]
fn public_expanded_ring_is_refused_before_quadratic_cost_or_work_admission() {
    use crate::coefficient_layout::DegreeBound;
    let ring = CoefficientRing::bounded(8192, 65537, DegreeBound::Public8192).unwrap();
    for cap in [0, MAX_COMPRESSION_WORK, usize::MAX] {
        assert_eq!(CoefficientCompression::new(ring, cap), Err(LayoutError::InvalidDegree));
    }
}

#[test]
fn source_geometry_is_address_algebra_not_backend_admission() {
    // Source §VI ring degree and 768 x 768 weights. Caller supplies scalar order;
    // no claim about an unresolved matrix-to-slot conversion chain.
    let g = CompressionGeometry::new(65536, 768 * 768).unwrap();
    assert_eq!(g.polynomial_count(), 9);
    assert_eq!(g.packed_words(), 768 * 768);
    for i in [0, 65535, 65536, 768 * 768 - 1] {
        let (p, c) = g.address(i).unwrap();
        assert_eq!(p * 65536 + c, i);
        assert_eq!(g.valid_coefficients(p), Ok(65536));
    }
    let edge = CompressionGeometry::new(4096, 4097).unwrap();
    assert_eq!(edge.polynomial_count(), 2);
    assert_eq!(edge.valid_coefficients(1), Ok(1));
    assert_eq!(edge.address(4097), Err(LayoutError::Index));
    assert_eq!(edge.valid_coefficients(2), Err(LayoutError::Index));
    assert!(CoefficientRing::new(65536, 65537).is_err());
}

#[test]
fn malformed_overflow_work_and_atomic_rejection() {
    for (n, count) in [(0, 1), (3, 1), (4, 0)] {
        assert_eq!(
            CompressionGeometry::new(n, count),
            Err(LayoutError::InvalidGeometry)
        );
    }
    assert_eq!(
        CompressionGeometry::new(2, usize::MAX),
        Err(LayoutError::Overflow)
    );
    assert_eq!(
        CompressionGeometry::new(1, usize::MAX),
        Err(LayoutError::Overflow)
    );
    for q in [0, 2, 4, 1 << 61, u64::MAX] {
        assert!(CoefficientRing::new(4, q).is_err());
    }
    let p = plan(4, 9);
    assert_eq!(p.coefficient_work(), 176);
    assert_eq!(
        CoefficientCompression::new(p.ring, 175),
        Err(LayoutError::ResourceLimit)
    );
    assert!(CoefficientCompression::new(p.ring, 176).is_ok());
    for (input, out_len, scratch_len, error) in [
        (vec![0, 1, 2, 9], 16, 8, LayoutError::NonCanonical),
        (vec![0, 1, 2, u64::MAX], 16, 8, LayoutError::NonCanonical),
        (vec![0; 3], 16, 8, LayoutError::BufferLength),
        (vec![0; 4], 15, 8, LayoutError::BufferLength),
        (vec![0; 4], 16, 7, LayoutError::BufferLength),
    ] {
        let mut output = vec![77; out_len];
        let mut scratch = vec![88; scratch_len];
        assert_eq!(p.decompress(&input, &mut output, &mut scratch), Err(error));
        assert_eq!(output, vec![77; out_len]);
        assert_eq!(scratch, vec![88; scratch_len]);
    }
    for values in [vec![], vec![0; 5], vec![9], vec![u64::MAX]] {
        let mut output = [77; 4];
        assert!(p.compress(&values, &mut output).is_err());
        assert_eq!(output, [77; 4]);
    }
    let mut short = [77; 3];
    assert_eq!(p.compress(&[1], &mut short), Err(LayoutError::BufferLength));
    assert_eq!(short, [77; 3]);
}
