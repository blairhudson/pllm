use super::*;

// Deliberately independent dense modular/negacyclic oracles, not descriptor
// address calculations or production trace/merge helpers.
fn dense_product(a: &[u64], b: &[u64], k: usize, m: usize, n: usize, q: u64) -> Vec<u64> {
    let mut out = vec![0; k * n];
    for r in 0..k {
        for c in 0..n {
            let mut sum = 0_i128;
            for h in 0..m {
                sum = (sum + i128::from(a[r * m + h]) * i128::from(b[h * n + c]))
                    .rem_euclid(q.into());
            }
            out[r * n + c] = sum as u64;
        }
    }
    out
}

fn convolution(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let mut out = vec![0_i128; a.len()];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            let term = i128::from(x) * i128::from(y);
            let sign = if i + j < a.len() { 1 } else { -1 };
            let index = (i + j) % a.len();
            out[index] = (out[index] + sign * term).rem_euclid(q.into());
        }
    }
    out.into_iter().map(|v| v as u64).collect()
}

fn values(count: usize, seed: u64, q: u64) -> Vec<u64> {
    (0..count)
        .map(|i| ((i as u128 * 31 + u128::from(seed)) % u128::from(q)) as u64)
        .collect()
}

// Keep the independent lane/coefficient coordinate oracle and accumulation order.
#[allow(clippy::needless_range_loop)]
fn evaluate(layout: CoefficientLayout, a: &[u64], b: &[u64]) -> Vec<u64> {
    let degree = layout.ring().degree();
    let q = layout.ring().modulus();
    let [kb, mb, nb] = layout.blocks();
    let mw = layout.window()[1];
    let mut encoded_a = vec![0; degree];
    let mut encoded_b = vec![0; degree];
    let mut group = vec![0; layout.storage().group_words];
    let mut packed = vec![0; layout.storage().packed_words];
    for g in 0..layout.group_count() {
        group.fill(0);
        let count = layout.group_inputs(g).unwrap();
        for lane in 0..count {
            // Global block ordering is constructed here independently of parse.
            let block = g * mw + lane;
            let alpha = block / nb;
            let gamma = block - alpha * nb;
            assert!(alpha < kb);
            for beta in 0..mb {
                layout.pack_lhs(a, alpha, beta, &mut encoded_a).unwrap();
                layout.pack_rhs(b, beta, gamma, &mut encoded_b).unwrap();
                let product = convolution(&encoded_a, &encoded_b, q);
                for c in 0..degree {
                    let at = lane * degree + c;
                    group[at] =
                        ((u128::from(group[at]) + u128::from(product[c])) % u128::from(q)) as u64;
                }
            }
        }
        let mut scratch = vec![0; layout.ring().interleave_scratch_words(mw, count).unwrap()];
        layout
            .ring()
            .interleave(
                &group[..count * degree],
                mw,
                count,
                &mut packed[g * degree..(g + 1) * degree],
                &mut scratch,
            )
            .unwrap();
        // The final incomplete group is zero-padded, not rotated or repeated.
        for lane in count..mw {
            for coefficient in (lane..degree).step_by(mw) {
                assert_eq!(packed[g * degree + coefficient], 0);
            }
        }
    }
    let mut output = vec![0; layout.storage().output_matrix_words];
    layout.parse(&packed, &mut output).unwrap();
    output
}

#[test]
fn rectangular_and_global_partial_windows_match_dense_products() {
    for (shape, window, degree) in [
        ([2, 2, 3], [2, 2, 3], 16),
        ([3, 2, 7], [2, 2, 3], 16),
        ([5, 5, 7], [2, 2, 3], 16),
        ([3, 3, 2], [2, 4, 1], 16),
        ([2, 1, 4], [1, 1, 2], 8),
        ([1, 1, 1], [1, 1, 1], 1),
    ] {
        for q in [9, 17, 257, (1 << 61) - 1] {
            let ring = CoefficientRing::new(degree, q).unwrap();
            let layout = CoefficientLayout::new(ring, shape, window, usize::MAX).unwrap();
            let [k, m, n] = shape;
            let a = values(k * m, 7, q);
            let b = values(m * n, 19, q);
            assert_eq!(evaluate(layout, &a, &b), dense_product(&a, &b, k, m, n, q));
        }
    }
}

#[test]
fn transposed_direction_and_two_cross_terms_reconstruct_shared_product() {
    let q = 257;
    let ring = CoefficientRing::new(16, q).unwrap();
    let forward = CoefficientLayout::new(ring, [3, 2, 7], [2, 2, 3], usize::MAX).unwrap();
    let reverse = CoefficientLayout::new(ring, [7, 2, 3], [3, 2, 2], usize::MAX).unwrap();
    let a = [values(6, 11, q), values(6, 5, q)];
    let b = [values(14, 29, q), values(14, 3, q)];
    let transpose = |input: &[u64], rows: usize, cols: usize| {
        (0..cols * rows)
            .map(|i| input[(i % rows) * cols + i / rows])
            .collect::<Vec<_>>()
    };
    let mut sum = vec![0; 21];
    for (lhs, rhs) in TwoPartyProductPlan::LOCAL_TERMS
        .into_iter()
        .chain(TwoPartyProductPlan::CROSS_TERMS)
    {
        let term = evaluate(forward, &a[lhs as usize], &b[rhs as usize]);
        let flipped = evaluate(
            reverse,
            &transpose(&b[rhs as usize], 2, 7),
            &transpose(&a[lhs as usize], 3, 2),
        );
        assert_eq!(term, transpose(&flipped, 7, 3));
        for (v, x) in sum.iter_mut().zip(term) {
            *v = (*v + x) % q;
        }
    }
    let reconstruct = |parts: &[Vec<u64>; 2]| {
        parts[0]
            .iter()
            .zip(&parts[1])
            .map(|(&x, &y)| (x + y) % q)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        sum,
        dense_product(&reconstruct(&a), &reconstruct(&b), 3, 2, 7, q)
    );
}

#[test]
fn corrected_row_stride_negative_exponents_and_global_parse_addresses() {
    let ring = CoefficientRing::new(16, 257).unwrap();
    let layout = CoefficientLayout::new(ring, [3, 2, 7], [2, 2, 3], usize::MAX).unwrap();
    let mut encoded = [99; 16];
    layout
        .pack_lhs(&[1, 2, 3, 4, 5, 6], 0, 0, &mut encoded)
        .unwrap();
    let mut expected = [0; 16];
    expected[0] = 1;
    expected[15] = 255; // 2*X^-1 = -2*X^15
    expected[6] = 3; // nw*mw, not kw*mw (=4)
    expected[5] = 4;
    assert_eq!(encoded, expected);
    assert_eq!(layout.blocks(), [2, 1, 3]);
    assert_eq!(layout.group_count(), 3);
    assert_eq!(layout.packed_address(0, 6), Ok((1, 0))); // global t=2
    assert_eq!(layout.packed_address(2, 0), Ok((1, 1))); // global t=3
    assert_eq!(layout.packed_address(2, 6), Ok((2, 1))); // global t=5
    assert_eq!(layout.packed_address(3, 0), Err(LayoutError::Index));
}

#[test]
fn optimized_interleave_matches_direct_lane_oracle_including_single_input() {
    for degree in [1, 2, 4, 8, 16, 64] {
        for q in [3, 9, 257, (1 << 61) - 1] {
            let ring = CoefficientRing::new(degree, q).unwrap();
            let mut stride = 1;
            while stride <= degree {
                for count in [1, stride.min(3), stride] {
                    let input = values(count * degree, 7, q);
                    let mut output = vec![q - 1; degree];
                    let mut scratch =
                        vec![q - 1; ring.interleave_scratch_words(stride, count).unwrap()];
                    ring.interleave(&input, stride, count, &mut output, &mut scratch)
                        .unwrap();
                    for i in 0..degree {
                        let lane = i % stride;
                        let expected = if lane < count {
                            input[lane * degree + i - lane]
                        } else {
                            0
                        };
                        assert_eq!(
                            output[i], expected,
                            "N={degree}, stride={stride}, count={count}, i={i}"
                        );
                    }
                }
                stride *= 2;
            }
        }
    }
}

#[test]
fn maximum_degree_and_large_stride_partial_groups_preserve_exact_lanes() {
    let degree = 4096;
    for q in [9, (1 << 61) - 1] {
        let ring = CoefficientRing::new(degree, q).unwrap();
        for (stride, count) in [(4096, 256), (4096, 255), (4096, 1), (512, 129)] {
            // Dense irrelevant coefficients must cancel, including coefficients
            // near the modulus. Partial counts exercise padded merge inputs;
            // stride > count also exercises the preceding trace stages.
            let input = values(count * degree, 7, q)
                .into_iter()
                .map(|v| q - 1 - v)
                .collect::<Vec<_>>();
            let mut expected = vec![0; degree];
            // Independent direct selection: no automorphism, modular inverse,
            // trace, merge, or production address helper participates here.
            for (lane, polynomial) in input.chunks_exact(degree).enumerate() {
                for (block, &coefficient) in polynomial.iter().step_by(stride).enumerate() {
                    expected[block * stride + lane] = coefficient;
                }
            }
            let mut output = vec![q - 1; degree];
            let mut scratch = vec![q - 1; ring.interleave_scratch_words(stride, count).unwrap()];
            ring.interleave(&input, stride, count, &mut output, &mut scratch)
                .unwrap();
            // Equality checks every retained coefficient and every absent lane,
            // rather than only a checksum or the nonzero prefix.
            assert_eq!(output, expected, "stride={stride}, count={count}, q={q}");
        }
    }
}

#[test]
fn inverse_matches_existing_repacking_and_works_for_odd_composites() {
    for q in [3, 9, 15, 65537, (1 << 61) - 1] {
        let ring = CoefficientRing::new(MAX_DEGREE, q).unwrap();
        for divisor in [1, 2, 64, MAX_DEGREE] {
            let inverse = ring.inverse_power_two(divisor).unwrap();
            assert_eq!(u128::from(inverse) * divisor as u128 % u128::from(q), 1);
            let mut old = vec![0; 2 * MAX_DEGREE];
            old[0] = 1;
            crate::coefficient_repacking::divide_power_two(&mut old, MAX_DEGREE, &[q], divisor)
                .unwrap();
            assert_eq!(old[0], inverse);
        }
        for bad in [0, 3, 8192, usize::MAX] {
            assert_eq!(
                ring.inverse_power_two(bad),
                Err(LayoutError::InvalidGeometry)
            );
        }
    }
}

#[test]
fn monomial_handles_negative_wraps_and_extreme_exponents() {
    let ring = CoefficientRing::new(16, 257).unwrap();
    let input = values(16, 13, 257);
    for exponent in [i64::MIN, -33, -17, -16, -1, 0, 1, 15, 16, 17, 33, i64::MAX] {
        let mut out = [0; 16];
        ring.monomial(&input, exponent, &mut out).unwrap();
        let mut oracle = [0; 16];
        for (i, &v) in input.iter().enumerate() {
            let power = i as i128 + i128::from(exponent);
            let sign = if power.div_euclid(16).rem_euclid(2) == 0 {
                1
            } else {
                -1
            };
            oracle[power.rem_euclid(16) as usize] = (sign * i128::from(v)).rem_euclid(257) as u64;
        }
        assert_eq!(out, oracle);
    }
}

#[test]
fn storage_admission_counts_disjoint_live_buffers_once_and_owns_no_heap() {
    let ring = CoefficientRing::new(16, 257).unwrap();
    let layout = CoefficientLayout::new(ring, [3, 2, 7], [2, 2, 3], usize::MAX).unwrap();
    let storage = layout.storage();
    assert_eq!(storage.input_matrix_words, 6 + 14);
    assert_eq!(storage.output_matrix_words, 21);
    assert_eq!(storage.encoding_words, 32);
    assert_eq!(storage.group_words, 32);
    assert_eq!(storage.interleave_scratch_words, 48); // two inputs + temp
    assert_eq!(storage.packed_words, 48); // three destination polynomials
    assert_eq!(storage.caller_bytes, (20 + 21 + 32 + 32 + 48 + 48) * 8);
    assert_eq!(
        storage.descriptor_bytes,
        std::mem::size_of::<CoefficientLayout>()
    );
    assert!(!std::mem::needs_drop::<CoefficientLayout>());
    assert_eq!(
        CoefficientLayout::new(ring, [3, 2, 7], [2, 2, 3], storage.caller_bytes),
        Ok(layout)
    );
    assert_eq!(
        CoefficientLayout::new(ring, [3, 2, 7], [2, 2, 3], storage.caller_bytes - 1),
        Err(LayoutError::ResourceLimit)
    );
    // No extra output allocation may alias group or scratch to reduce this bill.
    // Output is a slice of packed storage; Rust enforces the other disjoint borrows.
    let partial = CoefficientLayout::new(ring, [1, 1, 1], [1, 8, 1], usize::MAX).unwrap();
    assert_eq!(partial.storage().interleave_scratch_words, 32);
    assert_eq!(partial.group_inputs(0), Ok(1));
    assert_eq!(partial.group_inputs(1), Err(LayoutError::Index));
}

#[test]
fn invalid_domains_overflow_and_resource_bounds_are_rejected() {
    for degree in [0, 3, 8192, usize::MAX] {
        assert_eq!(
            CoefficientRing::new(degree, 17),
            Err(LayoutError::InvalidDegree)
        );
    }
    for q in [0, 1, 2, 4, 1 << 61, u64::MAX] {
        assert_eq!(
            CoefficientRing::new(16, q),
            Err(LayoutError::InvalidModulus)
        );
    }
    let ring = CoefficientRing::new(16, 257).unwrap();
    for window in [[0, 2, 3], [2, 0, 3], [2, 3, 2], [2, 2, 5], [1, 512, 1]] {
        assert_eq!(
            CoefficientLayout::new(ring, [3, 2, 7], window, usize::MAX),
            Err(LayoutError::InvalidGeometry)
        );
    }
    assert_eq!(
        CoefficientLayout::new(ring, [0, 1, 1], [1; 3], usize::MAX),
        Err(LayoutError::InvalidGeometry)
    );
    assert_eq!(
        CoefficientLayout::new(ring, [1; 3], [usize::MAX, 2, 2], usize::MAX),
        Err(LayoutError::Overflow)
    );
    assert_eq!(
        CoefficientLayout::new(ring, [usize::MAX, 2, 1], [1; 3], usize::MAX),
        Err(LayoutError::Overflow)
    );
    assert_eq!(
        CoefficientLayout::new(ring, [1, usize::MAX / 32, 1], [1; 3], usize::MAX),
        Err(LayoutError::ResourceLimit)
    );
    for (stride, count) in [(0, 1), (3, 1), (32, 1), (2, 0), (2, 3)] {
        assert_eq!(
            ring.interleave_scratch_words(stride, count),
            Err(LayoutError::InvalidGeometry)
        );
    }
    let largest = CoefficientRing::new(4096, 17).unwrap();
    assert_eq!(
        largest.interleave_scratch_words(4096, 257),
        Err(LayoutError::InvalidGeometry)
    );
    assert!(largest.interleave_scratch_words(4096, 256).is_ok());
}

#[test]
fn explicit_public_bound_preserves_legacy_constructor_and_method_gates() {
    assert_eq!(MAX_DEGREE, 4096);
    assert_eq!(DegreeBound::Legacy4096.maximum(), 4096);
    assert_eq!(DegreeBound::Public8192.maximum(), 8192);
    for degree in [1, 2, 16, 4096] {
        for q in [3, 15, 65537, (1_u64 << 61) - 1] {
            let legacy = CoefficientRing::new(degree, q);
            assert_eq!(CoefficientRing::bounded(degree, q, DegreeBound::Legacy4096), legacy);
            assert_eq!(CoefficientRing::bounded(degree, q, DegreeBound::Public8192), legacy);
        }
    }
    // Degree validation still precedes modulus validation for mixed failures.
    for degree in [0, 3, 8193, 16384, usize::MAX] {
        for q in [0, 2, 65537, u64::MAX] {
            assert_eq!(CoefficientRing::new(degree, q), Err(LayoutError::InvalidDegree));
            for bound in [DegreeBound::Legacy4096, DegreeBound::Public8192] {
                assert_eq!(CoefficientRing::bounded(degree, q, bound), Err(LayoutError::InvalidDegree));
            }
        }
    }
    for q in [0, 1, 2, 4, 1_u64 << 61, u64::MAX] {
        assert_eq!(CoefficientRing::new(8192, q), Err(LayoutError::InvalidDegree));
        assert_eq!(CoefficientRing::bounded(8192, q, DegreeBound::Public8192), Err(LayoutError::InvalidModulus));
    }
    assert_eq!(CoefficientRing::new(8192, 65537), Err(LayoutError::InvalidDegree));
    assert_eq!(CoefficientRing::bounded(8192, 65537, DegreeBound::Legacy4096), Err(LayoutError::InvalidDegree));
    let ring = CoefficientRing::bounded(8192, 65537, DegreeBound::Public8192).unwrap();
    assert_eq!((ring.degree(), ring.modulus()), (8192, 65537));
    for divisor in [0, 1, 3, 4096, 8192, 16384] {
        assert_eq!(ring.inverse_power_two(divisor), Err(LayoutError::InvalidDegree));
    }
    for input in [vec![0; 8192], vec![65537; 8192], vec![u64::MAX; 8191]] {
        let original = input.clone();
        for length in [0, 8191, 8192] {
            let mut out = vec![77; length];
            let mut scratch = vec![88; 16384];
            assert_eq!(ring.monomial(&input, i64::MIN, &mut out), Err(LayoutError::InvalidDegree));
            assert_eq!(ring.interleave(&input, 1, 1, &mut out, &mut scratch), Err(LayoutError::InvalidDegree));
            assert_eq!(ring.interleave(&input, 0, 0, &mut out, &mut scratch), Err(LayoutError::InvalidDegree));
            assert_eq!(out, vec![77; length]);
            assert_eq!(scratch, vec![88; 16384]);
            assert_eq!(input, original);
        }
    }
    for (stride, count) in [(1, 1), (0, 0), (8192, 256)] {
        assert_eq!(ring.interleave_scratch_words(stride, count), Err(LayoutError::InvalidDegree));
    }
    for (shape, window, cap) in [([1; 3], [1; 3], usize::MAX), ([0; 3], [0; 3], 0), ([usize::MAX; 3], [usize::MAX; 3], 0)] {
        assert_eq!(CoefficientLayout::new(ring, shape, window, cap), Err(LayoutError::InvalidDegree));
    }
    // Existing errors within the legacy domain retain their old order.
    let legacy = CoefficientRing::new(16, 257).unwrap();
    assert_eq!(legacy.inverse_power_two(3), Err(LayoutError::InvalidGeometry));
    let mut out = [77; 16];
    assert_eq!(legacy.monomial(&[257; 15], 0, &mut out), Err(LayoutError::BufferLength));
    assert_eq!(legacy.monomial(&[257; 16], 0, &mut out), Err(LayoutError::NonCanonical));
    assert_eq!(out, [77; 16]);
    assert_eq!(CoefficientLayout::new(legacy, [usize::MAX; 3], [0; 3], 0), Err(LayoutError::InvalidGeometry));
}

#[test]
fn invalid_buffers_are_atomic_including_noncanonical_padding() {
    let ring = CoefficientRing::new(16, 257).unwrap();
    let layout = CoefficientLayout::new(ring, [3, 2, 7], [2, 2, 3], usize::MAX).unwrap();
    let mut output = [77; 16];
    assert_eq!(
        layout.pack_lhs(&[0; 5], 0, 0, &mut output),
        Err(LayoutError::BufferLength)
    );
    assert_eq!(
        layout.pack_lhs(&[0; 6], 2, 0, &mut output),
        Err(LayoutError::Index)
    );
    assert_eq!(
        layout.pack_lhs(&[0, 0, 0, 0, 0, 257], 0, 0, &mut output),
        Err(LayoutError::NonCanonical)
    );
    assert_eq!(
        layout.pack_rhs(&[0; 14], 1, 0, &mut output),
        Err(LayoutError::Index)
    );
    assert_eq!(
        layout.pack_rhs(&[257; 14], 0, 0, &mut output),
        Err(LayoutError::NonCanonical)
    );
    assert_eq!(
        ring.monomial(&[257; 16], -1, &mut output),
        Err(LayoutError::NonCanonical)
    );
    assert_eq!(output, [77; 16]);
    let mut scratch = [88; 48];
    assert_eq!(
        ring.interleave(&[0; 31], 2, 2, &mut output, &mut scratch),
        Err(LayoutError::BufferLength)
    );
    assert_eq!(
        ring.interleave(&[257; 32], 2, 2, &mut output, &mut scratch),
        Err(LayoutError::NonCanonical)
    );
    assert_eq!(
        ring.interleave(&[0; 32], 2, 2, &mut output, &mut scratch[..47]),
        Err(LayoutError::BufferLength)
    );
    assert_eq!(output, [77; 16]);
    assert_eq!(scratch, [88; 48]);
    let mut parsed = [99; 21];
    let mut packed = [0; 48];
    packed[47] = 257; // validate even unused/cropped coefficients
    assert_eq!(
        layout.parse(&packed, &mut parsed),
        Err(LayoutError::NonCanonical)
    );
    assert_eq!(
        layout.parse(&[0; 47], &mut parsed),
        Err(LayoutError::BufferLength)
    );
    assert_eq!(parsed, [99; 21]);
}

#[test]
fn schedule_tiny_all_partial_groups_match_independent_signed_selector() {
    // For output a, lane=a%stride and source coefficient=a-lane. The lane's
    // polynomial starts at lane*N (not lane*stride); absent lanes select zero.
    for degree in [1, 2, 4, 8, 16, 32] {
        for q in [9, 15, 257, (1 << 61) - 1] {
            let ring = CoefficientRing::new(degree, q).unwrap();
            for stride in (0..=degree.trailing_zeros()).map(|bit| 1 << bit) {
                for count in 1..=stride {
                    // Near-modulus values force signed negacyclic wrap during
                    // traces/merges; irrelevant coefficients must cancel too.
                    let input: Vec<_> = (0..count * degree)
                        .map(|i| (-(i as i128 * 17 + 1)).rem_euclid(q.into()) as u64)
                        .collect();
                    let mut output = vec![q - 1; degree];
                    let mut scratch =
                        vec![q - 2; ring.interleave_scratch_words(stride, count).unwrap()];
                    ring.interleave(&input, stride, count, &mut output, &mut scratch)
                        .unwrap();
                    let expected: Vec<_> = (0..degree)
                        .map(|a| {
                            let lane = a % stride;
                            if lane < count {
                                input[lane * degree + a - lane]
                            } else {
                                0
                            }
                        })
                        .collect();
                    assert_eq!(
                        output, expected,
                        "N={degree}, stride={stride}, count={count}, q={q}"
                    );
                }
            }
        }
    }
}

#[test]
fn schedule_tiny_basis_inputs_remove_every_irrelevant_coefficient() {
    // Every input coordinate independently, including those not retained by
    // the selector. No production coordinates, inverses or automorphism oracle.
    for degree in [1, 2, 4, 8] {
        let ring = CoefficientRing::new(degree, 15).unwrap();
        for stride in (0..=degree.trailing_zeros()).map(|bit| 1 << bit) {
            for count in 1..=stride {
                let mut input = vec![0; count * degree];
                let mut output = vec![0; degree];
                let mut scratch = vec![0; ring.interleave_scratch_words(stride, count).unwrap()];
                for source in 0..input.len() {
                    input[source] = 14; // -1 in an odd composite ring
                    ring.interleave(&input, stride, count, &mut output, &mut scratch)
                        .unwrap();
                    for (a, &actual) in output.iter().enumerate() {
                        let lane = a % stride;
                        let expected = if lane < count && source == lane * degree + a - lane {
                            14
                        } else {
                            0
                        };
                        assert_eq!(
                            actual, expected,
                            "N={degree}, stride={stride}, count={count}, source={source}, a={a}"
                        );
                    }
                    input[source] = 0;
                }
            }
        }
    }
}

#[test]
fn schedule_tiny_matches_pre_extraction_output_and_scratch_bit_for_bit() {
    // Frozen pre-extraction loop order, intentionally not schedule-driven. This
    // checks intermediate scratch/temporary contents as well as final output;
    // independent selector/basis tests above check the mathematical contract.
    fn legacy(
        n: usize,
        q: u64,
        input: &[u64],
        stride: usize,
        count: usize,
    ) -> (Vec<u64>, Vec<u64>) {
        fn multiply(a: u64, b: u64, q: u64) -> u64 {
            (u128::from(a) * u128::from(b) % u128::from(q)) as u64
        }
        fn subtract(a: u64, b: u64, q: u64) -> u64 {
            if a >= b {
                a - b
            } else {
                q - (b - a)
            }
        }
        fn automorphism(n: usize, q: u64, output: &mut [u64], input: &[u64], element: usize) {
            for (i, &v) in input.iter().enumerate() {
                let at = i * element % (2 * n);
                let v = if at < n { v } else { subtract(0, v, q) };
                output[at % n] = (output[at % n] + v) % q;
            }
        }
        let padded = count.next_power_of_two();
        let mut scratch = vec![q - 2; (padded + 1) * n];
        let (values, temp) = scratch.split_at_mut(padded * n);
        let mut inverse = 1;
        for _ in 0..stride.trailing_zeros() {
            inverse = multiply(inverse, q.div_ceil(2), q);
        }
        values.fill(0);
        for (v, &a) in values.iter_mut().zip(input) {
            *v = multiply(a, inverse, q);
        }
        for value in values.chunks_exact_mut(n) {
            let mut h = padded;
            while h < stride {
                temp.copy_from_slice(value);
                automorphism(n, q, value, temp, n / h + 1);
                h *= 2;
            }
        }
        let mut h = padded / 2;
        while h != 0 {
            let (left, right) = values.split_at_mut(h * n);
            for (a, b) in left.chunks_exact_mut(n).zip(right.chunks_exact(n)) {
                for i in 0..n {
                    let shifted = if i >= h {
                        b[i - h]
                    } else {
                        subtract(0, b[n + i - h], q)
                    };
                    temp[i] = subtract(a[i], shifted, q);
                    a[i] = (a[i] + shifted) % q;
                }
                automorphism(n, q, a, temp, n / h + 1);
            }
            h /= 2;
        }
        let output = values[..n].to_vec();
        (output, scratch)
    }
    for degree in [1, 2, 4, 8, 16] {
        for q in [9, 15, (1 << 61) - 1] {
            let ring = CoefficientRing::new(degree, q).unwrap();
            for stride in (0..=degree.trailing_zeros()).map(|bit| 1 << bit) {
                for count in 1..=stride {
                    let input: Vec<_> = values(count * degree, 7, q)
                        .into_iter()
                        .map(|v| q - 1 - v)
                        .collect();
                    let expected = legacy(degree, q, &input, stride, count);
                    let mut output = vec![77; degree];
                    let mut scratch =
                        vec![q - 2; ring.interleave_scratch_words(stride, count).unwrap()];
                    ring.interleave(&input, stride, count, &mut output, &mut scratch)
                        .unwrap();
                    assert_eq!(
                        (output, scratch),
                        expected,
                        "N={degree}, stride={stride}, count={count}, q={q}"
                    );
                }
            }
        }
    }
}

#[test]
fn schedule_8192_exact_order_padding_and_read_before_overwrite() {
    use schedule::{InterleaveSchedule, InterleaveStep};
    use InterleaveStep::*;
    let schedule = InterleaveSchedule::bounded(8192, 64, 6, 8192).unwrap();
    assert_eq!(schedule.padded(), 8);
    assert_eq!(schedule.scratch_words(), 9 * 8192);
    assert!(!std::mem::needs_drop::<InterleaveSchedule>());
    assert!(!std::mem::needs_drop::<schedule::InterleaveSteps>());
    assert!(
        std::mem::size_of::<InterleaveSchedule>()
            + std::mem::size_of::<schedule::InterleaveSteps>()
            < 1024
    );
    let mut steps = schedule.steps();
    assert_eq!(
        steps.next(),
        Some(Normalize {
            count: 6,
            padded: 8,
            divisor: 64
        })
    );
    // No coefficient execution at this degree. Track source dependencies and
    // structural zeros symbolically; every padded lane still gets all traces.
    let mut dependencies = [1_u16, 2, 4, 8, 16, 32, 0, 0];
    let mut automorphisms = 0;
    for lane in 0..8 {
        for element in [1025, 513, 257] {
            assert_eq!(steps.next(), Some(Trace { lane, element }));
            automorphisms += 1;
        }
    }
    for (shift, element) in [(4, 2049), (2, 4097), (1, 8193)] {
        for left in 0..shift {
            let right = left + shift;
            assert_eq!(
                steps.next(),
                Some(Merge {
                    left,
                    right,
                    shift,
                    element
                })
            );
            // Right is outside this level's destinations, remains readable,
            // and contains no dependency already consumed by left.
            assert!(right >= shift);
            let before = dependencies[right];
            assert_eq!(dependencies[left] & before, 0);
            dependencies[left] |= before;
            assert_eq!(dependencies[right], before);
            automorphisms += 1;
        }
    }
    assert_eq!(dependencies[0], 0b11_1111);
    assert_eq!(automorphisms, 31);
    assert_eq!(steps.next(), Some(Finish));
    assert_eq!(steps.next(), None);
    assert_eq!(steps.next(), None);
}

#[test]
fn schedule_invalid_geometry_and_kernel_errors_leave_both_buffers_unchanged() {
    use schedule::InterleaveSchedule;
    for (degree, stride, count, maximum, error) in [
        (0, 1, 1, 8192, LayoutError::InvalidDegree),
        (3, 1, 1, 8192, LayoutError::InvalidDegree),
        (16384, 1, 1, 16384, LayoutError::InvalidDegree),
        (8192, 64, 6, 4096, LayoutError::InvalidDegree),
        (16, 0, 1, 16, LayoutError::InvalidGeometry),
        (16, 3, 1, 16, LayoutError::InvalidGeometry),
        (16, 32, 1, 16, LayoutError::InvalidGeometry),
        (16, 2, 0, 16, LayoutError::InvalidGeometry),
        (16, 2, 3, 16, LayoutError::InvalidGeometry),
        (8192, 512, 257, 8192, LayoutError::InvalidGeometry),
        (16, usize::MAX, usize::MAX, 16, LayoutError::InvalidGeometry),
    ] {
        assert_eq!(
            InterleaveSchedule::bounded(degree, stride, count, maximum),
            Err(error)
        );
        let mut output = [77; 16];
        let mut scratch = [88; 48];
        assert_eq!(
            trace::interleave(
                degree,
                15,
                maximum,
                &[0; 32],
                stride,
                count,
                &mut output,
                &mut scratch
            ),
            Err(error)
        );
        assert_eq!(output, [77; 16]);
        assert_eq!(scratch, [88; 48]);
    }
    // Preserve legacy error precedence and validate every input before writes.
    for (modulus, input, expected) in [
        (4, [0; 32], LayoutError::InvalidModulus),
        (15, [15; 32], LayoutError::NonCanonical),
    ] {
        let mut output = [77; 16];
        let mut scratch = [88; 48];
        assert_eq!(
            trace::interleave(16, modulus, 16, &input, 2, 2, &mut output, &mut scratch),
            Err(expected)
        );
        assert_eq!(output, [77; 16]);
        assert_eq!(scratch, [88; 48]);
    }
}
