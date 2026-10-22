use super::*;

const LIMITS: Limits = Limits {
    max_buffer_elements: 2_000_000,
    max_work: 100_000_000,
};
fn layout(r: usize, c: usize, h: usize, p: usize, d: Diagonal) -> Layout {
    Layout::new(r, c, r, c, h, p, d, LIMITS).unwrap()
}
fn data(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 7 % 13) as i32 - 6) as f64).collect()
}
fn packed(l: Layout, a: &[f64]) -> Vec<f64> {
    let mut v = vec![777.0; l.elements()];
    l.pack(a, &mut v, LIMITS).unwrap();
    v
}
fn unpacked(l: Layout, a: &[f64]) -> Vec<f64> {
    let mut v = vec![777.0; l.dense_elements()];
    l.unpack(a, &mut v, LIMITS).unwrap();
    v
}
// Independent row-major contraction, never calls diagonal/address helpers.
fn dense_mm(a: &[f64], b: &[f64], m: usize, n: usize, h: usize) -> Vec<f64> {
    let mut c = vec![0.0; h * m * n];
    for z in 0..h {
        for i in 0..m {
            for j in 0..n {
                for k in 0..n {
                    c[(z * m + i) * n + j] += a[(z * m + i) * n + k] * b[(z * n + k) * n + j];
                }
            }
        }
    }
    c
}

// Independent dense coordinates only: no layout, diagonal, rotation, pack or
// unpack helpers. Pair inner coordinates k and k+m/2 in the cyclic column order
// starting at j. Im(X) = (A[i,k]*B[k+m/2,j]-A[i,k+m/2]*B[k,j])/2.
fn dense_paired_oracle(
    a: &[f64],
    b: &[f64],
    n: usize,
    m: usize,
    heads: usize,
    vr: usize,
    vk: usize,
    vc: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut real = vec![0.0; heads * n * n];
    let mut imaginary = real.clone();
    for h in 0..heads {
        for i in 0..vr {
            for j in 0..vc {
                let av = |k| if k < vk { a[(h * n + i) * m + k] } else { 0.0 };
                let bv = |k| if k < vk { b[(h * m + k) * n + j] } else { 0.0 };
                for k in 0..m {
                    real[(h * n + i) * n + j] += av(k) * bv(k);
                }
                for l in 0..m / 2 {
                    let k = (j + l) % m;
                    let paired = (k + m / 2) % m;
                    imaginary[(h * n + i) * n + j] +=
                        (av(k) * bv(paired) - av(paired) * bv(k)) * 0.5;
                }
            }
        }
    }
    (real, imaginary)
}

const SENTINEL: f64 = 12345.0;
fn assert_padding_and_suffix(l: Layout, values: &[f64]) {
    for i in 0..l.elements() {
        if l.lane(i).unwrap().is_none() {
            assert_eq!(values[i], 0.0, "padding lane {i}");
        }
    }
    assert!(values[l.elements()..].iter().all(|v| *v == SENTINEL));
}
fn poison_padding(l: Layout, values: &mut [f64]) {
    for i in 0..l.elements() {
        if l.lane(i).unwrap().is_none() {
            values[i] = f64::NAN;
        }
    }
}

#[test]
fn all_addresses_heads_diagonals_and_partial_frames_are_bijective() {
    for (r, c) in [(1, 1), (1, 8), (2, 8), (8, 2), (4, 4), (8, 8)] {
        for h in [1, 3, 12] {
            for p in 1..=r.min(c) {
                for d in [Diagonal::Upper, Diagonal::Lower] {
                    let l = layout(r, c, h, p, d);
                    let a = data(l.dense_elements());
                    let v = packed(l, &a);
                    let mut seen = vec![false; l.elements()];
                    for z in 0..h {
                        for i in 0..r {
                            for j in 0..c {
                                let x = l.address(z, i, j).unwrap();
                                assert!(!seen[x]);
                                seen[x] = true;
                                assert_eq!(l.lane(x), Ok(Some((z, i, j))));
                                assert_eq!(v[x], a[(z * r + i) * c + j]);
                            }
                        }
                    }
                    for x in 0..l.elements() {
                        assert_eq!(seen[x], l.lane(x).unwrap().is_some());
                    }
                    assert_eq!(unpacked(l, &v), a);
                }
            }
        }
    }
}

#[test]
fn every_basis_vector_transposes_across_rectangles_and_orientations() {
    for (r, c) in [(2, 8), (8, 2), (4, 4)] {
        for d in [Diagonal::Upper, Diagonal::Lower] {
            let a = layout(r, c, 3, 2, d);
            let b = layout(c, r, 3, 1, d);
            for basis in 0..a.dense_elements() {
                let mut dense = vec![0.0; a.dense_elements()];
                dense[basis] = 1.0;
                let mut out = vec![9.0; b.elements()];
                transpose(a, &packed(a, &dense), b, &mut out, LIMITS).unwrap();
                let mut oracle = vec![0.0; dense.len()];
                for h in 0..3 {
                    for i in 0..r {
                        for j in 0..c {
                            oracle[(h * c + j) * r + i] = dense[(h * r + i) * c + j];
                        }
                    }
                }
                assert_eq!(unpacked(b, &out), oracle);
            }
        }
    }
}

#[test]
fn replication_and_masked_rotations_preserve_heads_and_padding() {
    for d in [Diagonal::Upper, Diagonal::Lower] {
        let l = Layout::new(4, 8, 3, 5, 3, 3, d, LIMITS).unwrap();
        let dense = data(l.dense_elements());
        let mut a = packed(l, &dense);
        for i in 0..a.len() {
            if l.lane(i).unwrap().is_none() {
                a[i] = f64::NAN;
            }
        }
        let mut replicated = vec![9.0; l.diagonals * l.frame_slots];
        replicate(l, &a, &mut replicated, LIMITS).unwrap();
        for diag in 0..l.diagonals {
            for copy in 0..l.capacity {
                for t in 0..l.length {
                    for h in 0..l.heads {
                        let (r, c) = l.coordinates(diag, t);
                        let expected = if r < 3 && c < 5 {
                            dense[(h * 4 + r) * 8 + c]
                        } else {
                            0.0
                        };
                        assert_eq!(
                            replicated[diag * l.frame_slots + (copy * l.length + t) * l.heads + h],
                            expected
                        );
                    }
                }
            }
        }
        for shift in [0, 1, 7, 8, 9, usize::MAX] {
            let mut out = vec![9.0; l.elements()];
            internal_rotate(l, &a, shift, &mut out, LIMITS).unwrap();
            for i in 0..out.len() {
                if let Some((h, _, _)) = l.lane(i).unwrap() {
                    let diag = (i / l.frame_slots) * l.capacity
                        + (i % l.frame_slots) / (l.length * l.heads);
                    let t = (i / l.heads) % l.length;
                    let (r, c) = l.coordinates(diag, (t + shift % l.length) % l.length);
                    assert_eq!(
                        out[i],
                        if r < 3 && c < 5 {
                            dense[(h * 4 + r) * 8 + c]
                        } else {
                            0.0
                        }
                    );
                } else {
                    assert_eq!(out[i], 0.0);
                }
            }
        }
    }
}

#[test]
fn lower_products_match_dense_oracle_for_all_small_geometry_and_valid_rectangles() {
    for m in [1, 2, 4] {
        for n in [m, 2 * m] {
            for h in [1, 3] {
                for p in 1..=m {
                    for (vr, vk, vc) in [(m, n, n), (m - 1, n - 1, n - 1), (1, 1, 1), (0, 0, 0)] {
                        let a = Layout::new(m, n, vr, vk, h, p, Diagonal::Lower, LIMITS).unwrap();
                        let b = Layout::new(n, n, vk, vc, h, p, Diagonal::Lower, LIMITS).unwrap();
                        let c = Layout::new(m, n, vr, vc, h, p, Diagonal::Lower, LIMITS).unwrap();
                        let pa = packed(a, &data(a.dense_elements()));
                        let pb = packed(b, &data(b.dense_elements()));
                        let da = unpacked(a, &pa);
                        let db = unpacked(b, &pb);
                        let expected = dense_mm(&da, &db, m, n, h);
                        let mut out = vec![999.0; c.elements()];
                        lower_mm(a, &pa, b, &pb, c, &mut out, LIMITS).unwrap();
                        assert_eq!(unpacked(c, &out), expected);
                        if p == m && a == c {
                            let mut scratch = vec![999.0; 2 * c.elements()];
                            let mut optimized = vec![999.0; c.elements()];
                            optimized_lower_mm(
                                a,
                                &pa,
                                b,
                                &pb,
                                c,
                                &mut scratch,
                                &mut optimized,
                                LIMITS,
                            )
                            .unwrap();
                            assert_eq!(optimized, out);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_lower_basis_pair_checks_all_head_and_diagonal_boundaries() {
    let a = layout(2, 4, 2, 1, Diagonal::Lower);
    let optimized_a = layout(2, 4, 2, 2, Diagonal::Lower);
    let b = layout(4, 4, 2, 2, Diagonal::Lower);
    for i in 0..a.dense_elements() {
        for j in 0..b.dense_elements() {
            let mut da = vec![0.0; a.dense_elements()];
            da[i] = 1.0;
            let mut db = vec![0.0; b.dense_elements()];
            db[j] = 1.0;
            let mut out = vec![9.0; a.elements()];
            lower_mm(a, &packed(a, &da), b, &packed(b, &db), a, &mut out, LIMITS).unwrap();
            assert_eq!(unpacked(a, &out), dense_mm(&da, &db, 2, 4, 2));
            let mut optimized = vec![9.0; optimized_a.elements()];
            let mut scratch = vec![9.0; 2 * optimized_a.elements()];
            optimized_lower_mm(
                optimized_a,
                &packed(optimized_a, &da),
                b,
                &packed(b, &db),
                optimized_a,
                &mut scratch,
                &mut optimized,
                LIMITS,
            )
            .unwrap();
            assert_eq!(
                unpacked(optimized_a, &optimized),
                dense_mm(&da, &db, 2, 4, 2)
            );
        }
    }
}

#[test]
fn block_pc_full_cross_block_weights_match_independent_dense_oracle() {
    for n in [1, 2, 4] {
        for h in [1, 2, 3] {
            for p in 1..=n {
                let l = layout(n, n, h, p, Diagonal::Lower);
                // Algorithm 1 requires capacity dividing n, not arbitrary final frames.
                if n % p != 0 {
                    continue;
                }
                let d = n * h;
                let weights = data(d * d);
                let b = data(l.dense_elements());
                let mut out = vec![9.0; l.elements()];
                block_pc_mm(l, &weights, &packed(l, &b), &mut out, LIMITS).unwrap();
                let mut expected = vec![0.0; b.len()];
                for i in 0..d {
                    for t in 0..n {
                        for k in 0..d {
                            expected[i * n + t] += weights[i * d + k] * b[k * n + t];
                        }
                    }
                }
                assert_eq!(unpacked(l, &out), expected);
            }
        }
    }
}

#[test]
fn source_shapes_enumerate_without_he_objects_and_counts_have_separate_stages() {
    let l = layout(64, 128, 12, 16, Diagonal::Lower);
    assert_eq!(
        (l.frames(), l.frame_slots(), l.elements()),
        (4, 24576, 98304)
    );
    let mut seen = vec![false; l.elements()];
    for h in 0..12 {
        for r in 0..64 {
            for c in 0..128 {
                let i = l.address(h, r, c).unwrap();
                assert!(!seen[i]);
                seen[i] = true;
            }
        }
    }
    assert!(seen.iter().all(|x| *x));
    let s = lower_schedule(l).unwrap();
    assert_eq!(s.tensor_products, 512);
    assert_eq!(s.plaintext_products, 4 * 4 * 127);
    assert_eq!(s.input_rotations, 2 * 4 * 127);
    assert_eq!(s.relinearize_after_product_sums, 4);
    assert_eq!(s.conjugations, 0);
    assert_eq!(
        optimized_lower_schedule(l),
        Err(Error::OptimizedMultiBlockIndexGate)
    );
    let pc = block_pc_schedule(layout(128, 128, 6, 32, Diagonal::Lower)).unwrap();
    assert_eq!(pc.plaintext_products, 6 * 4 * 128 + 6 * 4);
    assert_eq!(pc.input_rotations, 4 * 31);
    assert_eq!(pc.output_rotations, 2 * 5 * 4);
    assert_eq!(pc.tensor_products, 0);
    assert_eq!(pc.relinearize_after_product_sums, 0);
    assert_eq!(pc.rescale_after_product_sums, 24);
}

#[test]
fn printed_multiblock_rotation_has_explicit_counterexample() {
    // m=n=4,c=2,j=0,l=2,r=t=0. Eq.(10) requires L_2(A)[2]
    // = A[0,2]. Algorithm 2 line 7 rotates frame 0 by +2 and exposes
    // L_0(A)[2] = A[2,2]; its j-1 branch is empty for l mod c=0.
    let l = layout(4, 4, 1, 2, Diagonal::Lower);
    let a: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let slots = packed(l, &a);
    assert_eq!(slots[2], 10.0);
    assert_eq!(slots[l.address(0, 0, 2).unwrap()], 2.0);
    assert_eq!(
        optimized_lower_schedule(l),
        Err(Error::OptimizedMultiBlockIndexGate)
    );
}

#[test]
fn specified_complexification_has_nonzero_imaginary_control_and_dense_real_result() {
    let left = layout(8, 4, 2, 1, Diagonal::Upper);
    let right = layout(4, 8, 2, 1, Diagonal::Lower);
    let out = layout(8, 8, 2, 1, Diagonal::Lower);
    let a = data(left.dense_elements());
    let b = data(right.dense_elements());
    let mut c = vec![99.0; out.elements()];
    let mut im = c.clone();
    paired_upper_lower_mm(
        left,
        &packed(left, &a),
        right,
        &packed(right, &b),
        out,
        &mut im,
        &mut c,
        LIMITS,
    )
    .unwrap();
    let (expected, expected_imaginary) = dense_paired_oracle(&a, &b, 8, 4, 2, 8, 4, 8);
    assert_eq!(unpacked(out, &c), expected);
    assert_eq!(unpacked(out, &im), expected_imaginary);
    assert!(im.iter().any(|x| *x != 0.0));
    let counts = paired_upper_lower_schedule(left).unwrap();
    assert_eq!(counts.tensor_products, 16);
    assert_eq!(counts.conjugations, 8);
    assert_eq!(counts.input_rotations, 14);
}

#[test]
fn overflow_short_buffers_invalid_inputs_and_work_reject_before_mutation() {
    assert_eq!(
        Layout::new(usize::MAX, usize::MAX, 0, 0, 2, 1, Diagonal::Lower, LIMITS),
        Err(Error::Overflow)
    );
    assert_eq!(
        Layout::new(3, 4, 3, 4, 1, 1, Diagonal::Lower, LIMITS),
        Err(Error::Geometry)
    );
    let l = layout(2, 2, 1, 2, Diagonal::Lower);
    let a = packed(l, &[1.0, 2.0, 3.0, 4.0]);
    let mut out = vec![999.0; 5];
    let sentinel = out.clone();
    assert_eq!(l.pack(&[1.0], &mut out, LIMITS), Err(Error::Buffer));
    assert_eq!(out, sentinel);
    assert_eq!(
        l.pack(&[1.0, 2.0, 3.0, f64::NAN], &mut out, LIMITS),
        Err(Error::NonFinite)
    );
    assert_eq!(out, sentinel);
    let tiny = Limits {
        max_work: 0,
        ..LIMITS
    };
    assert_eq!(
        lower_mm(l, &a, l, &a, l, &mut out, tiny),
        Err(Error::WorkLimit)
    );
    assert_eq!(out, sentinel);
    assert_eq!(
        lower_mm(l, &a[..3], l, &a, l, &mut out, LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(out, sentinel);
    assert_eq!(replicate(l, &a, &mut out, LIMITS), Err(Error::Buffer));
    assert_eq!(out, sentinel);
    assert_eq!(
        transpose(l, &a, l, &mut out[..3], LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(out, sentinel);
    let mut scratch = vec![888.0; 7];
    let before = scratch.clone();
    assert_eq!(
        optimized_lower_mm(l, &a, l, &a, l, &mut scratch, &mut out, LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(out, sentinel);
    assert_eq!(scratch, before);
    let mut bad = a.clone();
    bad[3] = f64::INFINITY;
    assert_eq!(
        internal_rotate(l, &bad, 1, &mut out, LIMITS),
        Err(Error::NonFinite)
    );
    assert_eq!(out, sentinel);
    let mut with_tail = vec![999.0; 5];
    l.pack(&[1.0; 4], &mut with_tail, LIMITS).unwrap();
    assert_eq!(with_tail[4], 999.0);
    assert_eq!(l.address(1, 0, 0), Err(Error::Geometry));
    assert_eq!(l.lane(l.elements()), Err(Error::Buffer));
}

#[test]
fn all_buffer_apis_check_extents_and_work_before_late_nan_scans() {
    // Construct with generous limits, then deny work at call time. Last active
    // element is NaN: budget must win over the otherwise late NonFinite error.
    let l = layout(4, 4, 2, 4, Diagonal::Lower);
    let upper = layout(4, 4, 2, 1, Diagonal::Upper);
    let single = layout(4, 4, 2, 1, Diagonal::Lower);
    let mut bad = vec![1.0; l.elements()];
    *bad.last_mut().unwrap() = f64::NAN;
    let good = vec![1.0; l.elements()];
    let mut weights = vec![1.0; 64];
    *weights.last_mut().unwrap() = f64::NAN;
    let mut output = vec![SENTINEL; l.diagonals * l.frame_slots + 3];
    let mut scratch = vec![SENTINEL; 2 * l.elements() + 3];
    let before = output.clone();
    let scratch_before = scratch.clone();
    let denied = Limits {
        max_work: 0,
        ..LIMITS
    };
    assert_eq!(l.pack(&bad, &mut output, denied), Err(Error::WorkLimit));
    assert_eq!(l.unpack(&bad, &mut output, denied), Err(Error::WorkLimit));
    assert_eq!(
        transpose(l, &bad, l, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        internal_rotate(l, &bad, 1, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        replicate(l, &bad, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        lower_mm(l, &bad, l, &bad, l, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        block_pc_mm(l, &weights, &bad, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        optimized_lower_mm(l, &bad, l, &bad, l, &mut scratch, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        paired_upper_lower_mm(
            upper,
            &bad,
            single,
            &bad,
            single,
            &mut scratch,
            &mut output,
            denied
        ),
        Err(Error::WorkLimit)
    );
    assert_eq!(output, before);
    assert_eq!(scratch, scratch_before);

    // Once work is admitted, the same late active NaN must be diagnosed without
    // changing the destination (the budget rejection did not mask a bad fixture).
    assert_eq!(l.unpack(&bad, &mut output, LIMITS), Err(Error::NonFinite));
    assert_eq!(
        transpose(l, &bad, l, &mut output, LIMITS),
        Err(Error::NonFinite)
    );
    assert_eq!(
        internal_rotate(l, &bad, 1, &mut output, LIMITS),
        Err(Error::NonFinite)
    );
    assert_eq!(
        replicate(l, &bad, &mut output, LIMITS),
        Err(Error::NonFinite)
    );
    assert_eq!(output, before);

    // Extent checks must also precede *all* scans, including earlier operands.
    let short = l.elements() - 1;
    assert_eq!(
        l.unpack(&bad, &mut output[..short], LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        transpose(l, &bad, l, &mut output[..short], LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        internal_rotate(l, &bad, 1, &mut output[..short], LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        replicate(l, &bad, &mut output[..short], LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        lower_mm(l, &bad, l, &good[..short], l, &mut output, LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        block_pc_mm(l, &weights, &good[..short], &mut output, LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        optimized_lower_mm(
            l,
            &bad,
            l,
            &good[..short],
            l,
            &mut scratch,
            &mut output,
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        paired_upper_lower_mm(
            upper,
            &bad,
            single,
            &good[..short],
            single,
            &mut scratch,
            &mut output,
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(output, before);
    assert_eq!(scratch, scratch_before);
}

#[test]
fn optimized_and_complex_errors_preserve_both_outputs_including_suffixes() {
    let l = layout(4, 4, 2, 4, Diagonal::Lower);
    let upper = layout(4, 4, 2, 1, Diagonal::Upper);
    let single = layout(4, 4, 2, 1, Diagonal::Lower);
    let good = vec![1.0; l.elements()];
    let mut bad = good.clone();
    *bad.last_mut().unwrap() = f64::NAN;
    let mut output = vec![SENTINEL; l.elements() + 3];
    let mut scratch = vec![SENTINEL; 2 * l.elements() + 3];
    let before = output.clone();
    let scratch_before = scratch.clone();
    for (a, b) in [(&bad, &good), (&good, &bad)] {
        assert_eq!(
            optimized_lower_mm(l, a, l, b, l, &mut scratch, &mut output, LIMITS),
            Err(Error::NonFinite)
        );
        assert_eq!(
            paired_upper_lower_mm(
                upper,
                a,
                single,
                b,
                single,
                &mut scratch,
                &mut output,
                LIMITS
            ),
            Err(Error::NonFinite)
        );
        assert_eq!(output, before);
        assert_eq!(scratch, scratch_before);
    }
    let short = l.elements() - 1;
    assert_eq!(
        optimized_lower_mm(
            l,
            &good,
            l,
            &good,
            l,
            &mut scratch[..2 * l.elements() - 1],
            &mut output,
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        optimized_lower_mm(
            l,
            &good,
            l,
            &good,
            l,
            &mut scratch,
            &mut output[..short],
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        paired_upper_lower_mm(
            upper,
            &good,
            single,
            &good,
            single,
            &mut scratch[..short],
            &mut output,
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        paired_upper_lower_mm(
            upper,
            &good,
            single,
            &good,
            single,
            &mut scratch,
            &mut output[..short],
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        optimized_lower_mm(
            l,
            &good,
            l,
            &good,
            single,
            &mut scratch,
            &mut output,
            LIMITS
        ),
        Err(Error::Geometry)
    );
    assert_eq!(
        paired_upper_lower_mm(
            l,
            &good,
            single,
            &good,
            single,
            &mut scratch,
            &mut output,
            LIMITS
        ),
        Err(Error::Geometry)
    );
    assert_eq!(output, before);
    assert_eq!(scratch, scratch_before);
}

#[test]
fn transforms_zero_padding_ignore_poison_and_preserve_output_suffixes() {
    for d in [Diagonal::Upper, Diagonal::Lower] {
        let l = Layout::new(4, 8, 3, 5, 2, 3, d, LIMITS).unwrap();
        let target = Layout::new(8, 4, 5, 3, 2, 3, d, LIMITS).unwrap();
        let dense = data(l.dense_elements());
        let mut slots = vec![SENTINEL; l.elements() + 3];
        l.pack(&dense, &mut slots, LIMITS).unwrap();
        assert_padding_and_suffix(l, &slots);
        poison_padding(l, &mut slots);
        let mut restored = vec![SENTINEL; l.dense_elements() + 3];
        l.unpack(&slots, &mut restored, LIMITS).unwrap();
        for h in 0..2 {
            for i in 0..4 {
                for j in 0..8 {
                    assert_eq!(
                        restored[(h * 4 + i) * 8 + j],
                        if i < 3 && j < 5 {
                            dense[(h * 4 + i) * 8 + j]
                        } else {
                            0.0
                        }
                    );
                }
            }
        }
        assert!(
            restored[l.dense_elements()..]
                .iter()
                .all(|v| *v == SENTINEL)
        );
        let mut transposed = vec![SENTINEL; target.elements() + 3];
        transpose(l, &slots, target, &mut transposed, LIMITS).unwrap();
        assert_padding_and_suffix(target, &transposed);
        for h in 0..2 {
            for i in 0..3 {
                for j in 0..5 {
                    assert_eq!(
                        transposed[target.address(h, j, i).unwrap()],
                        dense[(h * 4 + i) * 8 + j]
                    );
                }
            }
        }
        let mut rotated = vec![SENTINEL; l.elements() + 3];
        internal_rotate(l, &slots, 1, &mut rotated, LIMITS).unwrap();
        assert_padding_and_suffix(l, &rotated);
        let required = l.diagonals * l.frame_slots;
        let mut replicated = vec![SENTINEL; required + 3];
        replicate(l, &slots, &mut replicated, LIMITS).unwrap();
        for diag in 0..l.diagonals {
            for copy in 0..l.capacity {
                for t in 0..l.length {
                    for h in 0..l.heads {
                        let (r, c) = l.coordinates(diag, t);
                        assert_eq!(
                            replicated[diag * l.frame_slots + (copy * l.length + t) * l.heads + h],
                            if r < 3 && c < 5 {
                                dense[(h * 4 + r) * 8 + c]
                            } else {
                                0.0
                            }
                        );
                    }
                }
            }
        }
        assert!(replicated[required..].iter().all(|v| *v == SENTINEL));
    }
}

#[test]
fn lower_and_pc_products_zero_padding_and_preserve_suffixes() {
    // Nondividing capacity also exercises padded final frame in the basic path.
    let left = Layout::new(4, 8, 3, 5, 2, 3, Diagonal::Lower, LIMITS).unwrap();
    let right = Layout::new(8, 8, 5, 6, 2, 3, Diagonal::Lower, LIMITS).unwrap();
    let out = Layout::new(4, 8, 3, 6, 2, 3, Diagonal::Lower, LIMITS).unwrap();
    let mut a = packed(left, &data(left.dense_elements()));
    let mut b = packed(right, &data(right.dense_elements()));
    let expected = dense_mm(&unpacked(left, &a), &unpacked(right, &b), 4, 8, 2);
    poison_padding(left, &mut a);
    poison_padding(right, &mut b);
    let mut c = vec![SENTINEL; out.elements() + 3];
    lower_mm(left, &a, right, &b, out, &mut c, LIMITS).unwrap();
    assert_eq!(unpacked(out, &c), expected);
    assert_padding_and_suffix(out, &c);

    let pc = Layout::new(4, 4, 4, 3, 2, 2, Diagonal::Lower, LIMITS).unwrap();
    let raw = data(pc.dense_elements());
    let weights = data(64);
    let mut b = packed(pc, &raw);
    poison_padding(pc, &mut b);
    let mut c = vec![SENTINEL; pc.elements() + 3];
    block_pc_mm(pc, &weights, &b, &mut c, LIMITS).unwrap();
    assert_padding_and_suffix(pc, &c);
    for i in 0..8 {
        for j in 0..3 {
            let expected: f64 = (0..8).map(|k| weights[i * 8 + k] * raw[k * 4 + j]).sum();
            assert_eq!(c[pc.address(i / 4, i % 4, j).unwrap()], expected);
        }
    }
}

#[test]
fn optimized_partial_rectangle_zeroes_output_padding_and_preserves_scratch_suffix() {
    let left = Layout::new(2, 4, 1, 3, 2, 2, Diagonal::Lower, LIMITS).unwrap();
    let right = Layout::new(4, 4, 3, 3, 2, 2, Diagonal::Lower, LIMITS).unwrap();
    let mut a = packed(left, &data(left.dense_elements()));
    let mut b = packed(right, &data(right.dense_elements()));
    let expected = dense_mm(&unpacked(left, &a), &unpacked(right, &b), 2, 4, 2);
    poison_padding(left, &mut a);
    poison_padding(right, &mut b);
    let mut c = vec![SENTINEL; left.elements() + 3];
    let mut scratch = vec![SENTINEL; 2 * left.elements() + 3];
    optimized_lower_mm(left, &a, right, &b, left, &mut scratch, &mut c, LIMITS).unwrap();
    assert_eq!(unpacked(left, &c), expected);
    assert_padding_and_suffix(left, &c);
    // Scratch holds unmasked intermediate partial sums, not a matrix layout.
    assert!(
        scratch[2 * left.elements()..]
            .iter()
            .all(|v| *v == SENTINEL)
    );
}

#[test]
fn complex_basis_pairs_and_partial_rectangles_match_both_dense_oracles() {
    for (n, m) in [(4, 2), (4, 4)] {
        for (vr, vk, vc) in [(n, m, n), (n - 1, m - 1, n - 1)] {
            let left = Layout::new(n, m, vr, vk, 2, 1, Diagonal::Upper, LIMITS).unwrap();
            let right = Layout::new(m, n, vk, vc, 2, 1, Diagonal::Lower, LIMITS).unwrap();
            let out = Layout::new(n, n, vr, vc, 2, 1, Diagonal::Lower, LIMITS).unwrap();
            // Includes basis entries outside valid rectangles: these must never
            // contribute to either component, even with poisoned packed padding.
            for ai in 0..left.dense_elements() {
                for bi in 0..right.dense_elements() {
                    let mut da = vec![0.0; left.dense_elements()];
                    da[ai] = 1.0;
                    let mut db = vec![0.0; right.dense_elements()];
                    db[bi] = 1.0;
                    let (expected, expected_im) =
                        dense_paired_oracle(&da, &db, n, m, 2, vr, vk, vc);
                    let mut a = packed(left, &da);
                    let mut b = packed(right, &db);
                    poison_padding(left, &mut a);
                    poison_padding(right, &mut b);
                    let mut c = vec![SENTINEL; out.elements() + 3];
                    let mut im = c.clone();
                    paired_upper_lower_mm(left, &a, right, &b, out, &mut im, &mut c, LIMITS)
                        .unwrap();
                    assert_eq!(unpacked(out, &c), expected);
                    assert_eq!(unpacked(out, &im), expected_im);
                    assert_padding_and_suffix(out, &c);
                    assert_padding_and_suffix(out, &im);
                }
            }
        }
    }
}
