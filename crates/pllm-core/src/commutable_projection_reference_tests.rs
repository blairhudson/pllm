use super::*;
use crate::coefficient_compression::CoefficientCompression;

// Independent signed full convolution followed by X^N=-1 reduction.
// No candidate arithmetic, monomial or compression helper constructs expectations.
fn convolution(a: &[u64], b: &[u64], q: i128) -> Vec<i128> {
    let n = a.len();
    let mut full = vec![0_i128; 2 * n - 1];
    for i in 0..n {
        for j in 0..n {
            full[i + j] += i128::from(a[i]) * i128::from(b[j]);
        }
    }
    for i in n..full.len() {
        let high = full[i];
        full[i - n] -= high;
    }
    full[..n].iter().map(|x| x.rem_euclid(q)).collect()
}
fn dense(a: &[u64], w: &[u64], [rows, inner, cols]: [usize; 3], q: i128) -> Vec<u64> {
    (0..rows * cols)
        .map(|i| {
            (0..inner)
                .map(|h| i128::from(a[(i / cols) * inner + h]) * i128::from(w[h * cols + i % cols]))
                .sum::<i128>()
                .rem_euclid(q) as u64
        })
        .collect()
}
fn ledger(work: usize) -> WorkLedger {
    WorkLedger::new(work, 0)
}
fn price(ring: CoefficientRing, op: Operation) -> Quote {
    quote(ring, op, usize::MAX, MAX_INVOCATION_WORK).unwrap()
}

#[test]
fn expanded_public_descriptor_refuses_before_pricing_buffers_or_debit() {
    use crate::coefficient_layout::DegreeBound;
    let ring = CoefficientRing::bounded(8192, 65537, DegreeBound::Public8192).unwrap();
    let error = Err(Error::Layout(LayoutError::InvalidDegree));
    for op in [
        Operation::AddClientLayer,
        Operation::RemoveServerLayer,
        Operation::ComposeProjection([1; 3]),
        Operation::ComposeProjection([0; 3]),
        Operation::ComposeProjection([usize::MAX; 3]),
    ] {
        for (bytes, work) in [(0, 0), (usize::MAX, usize::MAX)] {
            assert_eq!(quote(ring, op, bytes, work), Err(Error::Layout(LayoutError::InvalidDegree)));
        }
    }
    // Full-sized canonical/noncanonical and malformed inputs: degree refusal
    // wins, and the caller's original ledger and buffers remain unchanged.
    let mut work = ledger(123);
    for (length, value, cap) in [(1, u64::MAX, 0), (8192, 1, usize::MAX), (8192, 65537, usize::MAX)] {
        let input = vec![value; length];
        let noise = vec![0; length];
        let mut out = vec![77; length];
        let mut scratch = vec![88; length];
        assert_eq!(add_client_layer(ring, &input, &input, &input, &noise, &mut out, &mut scratch, cap, cap, &mut work), error);
        assert_eq!(remove_server_layer(ring, &input, &input, &input, &mut out, &mut scratch, cap, cap, &mut work), error);
        for shape in [[1; 3], [0; 3], [usize::MAX; 3]] {
            assert_eq!(compose_projection(ring, shape, &input, &input, &input, &mut out, cap, cap, &mut work), error);
        }
        assert_eq!(work.remaining(), 123);
        assert_eq!(work.control_remaining(), 0);
        assert_eq!(out, vec![77; length]);
        assert_eq!(scratch, vec![88; length]);
        assert_eq!(input, vec![value; length]);
        assert_eq!(noise, vec![0; length]);
    }
}

#[test]
fn appendix_b_both_signed_noises_wrap_and_projection_are_distinct() {
    for q in [17_u64, 97, (1 << 61) - 1] {
        let ring = CoefficientRing::new(4, q).unwrap();
        let a = [q - 1, 3, 4, 7];
        let server = [2, 0, q - 1, 3];
        let client = [1, q - 2, 3, 1];
        let noise = [-2_i64, 1, -1, 2];
        let prime = [1_i64, -3, 2, -1];
        let matrix = [2, 5, 7, 3];
        let u = [9, 1, 4, 8];
        let weights = [3, 2, 1, 4];
        let shape = [2, 2, 2];
        let uw = dense(&u, &weights, shape, i128::from(q));
        let aw = dense(&matrix, &weights, shape, i128::from(q));
        let compression = CoefficientCompression::new(ring, MAX_INVOCATION_WORK).unwrap();
        let mut message = [0; 4];
        compression.compress(&uw, &mut message).unwrap();
        let mut constants = [0; 16];
        let mut compression_scratch = [0; 8];
        compression
            .decompress(&message, &mut constants, &mut compression_scratch)
            .unwrap();
        for i in 0..4 {
            assert_eq!(&constants[4 * i..4 * i + 4], &[uw[i], 0, 0, 0]);
        }
        let server_product = convolution(&a, &server, i128::from(q));
        let client_product = convolution(&a, &client, i128::from(q));
        let b: Vec<_> = (0..4)
            .map(|i| {
                (server_product[i] + i128::from(uw[i]) + i128::from(noise[i]))
                    .rem_euclid(i128::from(q)) as u64
            })
            .collect();
        let p_add = price(ring, Operation::AddClientLayer);
        let p_remove = price(ring, Operation::RemoveServerLayer);
        let p_projection = price(ring, Operation::ComposeProjection(shape));
        let mut work = ledger(p_add.work + p_remove.work + p_projection.work);
        let mut layered = [99; 4];
        let mut scratch = [88; 4];
        add_client_layer(
            ring,
            &a,
            &b,
            &client,
            &prime,
            &mut layered,
            &mut scratch,
            p_add.caller_bytes,
            p_add.work,
            &mut work,
        )
        .unwrap();
        let expected_layered: Vec<_> = (0..4)
            .map(|i| {
                (i128::from(b[i]) + client_product[i] + i128::from(prime[i]))
                    .rem_euclid(i128::from(q)) as u64
            })
            .collect();
        assert_eq!(layered.as_slice(), expected_layered);
        let mut removed = [99; 4];
        remove_server_layer(
            ring,
            &a,
            &layered,
            &server,
            &mut removed,
            &mut scratch,
            p_remove.caller_bytes,
            p_remove.work,
            &mut work,
        )
        .unwrap();
        for i in 0..4 {
            let phase = (i128::from(removed[i]) - client_product[i]).rem_euclid(i128::from(q));
            assert_eq!(
                phase,
                (i128::from(uw[i]) + i128::from(noise[i]) + i128::from(prime[i]))
                    .rem_euclid(i128::from(q))
            );
            assert_ne!(phase, i128::from(uw[i]));
            // Online correction retains noise: the resulting diagnostic phase
            // is AW+e+e', never the noise-free matrix identity.
            let online = (i128::from(aw[i]) - i128::from(uw[i])).rem_euclid(i128::from(q));
            assert_eq!(
                (online + phase).rem_euclid(i128::from(q)),
                (i128::from(aw[i]) + i128::from(noise[i]) + i128::from(prime[i]))
                    .rem_euclid(i128::from(q))
            );
        }
        let mut projection = [99; 4];
        compose_projection(
            ring,
            shape,
            &matrix,
            &u,
            &weights,
            &mut projection,
            p_projection.caller_bytes,
            p_projection.work,
            &mut work,
        )
        .unwrap();
        assert_eq!(projection.as_slice(), aw);
        assert_eq!(work.remaining(), 0);
    }

    // Rectangular success distinguishes the weight stride cols=4 from inner=3.
    // Manual unreduced AW rows: [38,44,50,56], [83,98,113,128].
    let ring = CoefficientRing::new(4, 17).unwrap();
    let shape = [2, 3, 4];
    let matrix = [1, 2, 3, 4, 5, 6];
    let u = [6, 5, 4, 3, 2, 1];
    let weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let expected = [4, 10, 16, 5, 15, 13, 11, 9];
    let aw = dense(&matrix, &weights, shape, 17);
    assert_eq!(aw.as_slice(), expected);
    let masked: Vec<u64> = matrix
        .iter()
        .zip(u.iter())
        .map(|(&a, &mask)| (i128::from(a) - i128::from(mask)).rem_euclid(17) as u64)
        .collect();
    assert_eq!(masked, [12, 14, 16, 1, 3, 5]);
    let online = dense(&masked, &weights, shape, 17);
    let offline = dense(&u, &weights, shape, 17);
    let reconstructed: Vec<u64> = online
        .iter()
        .zip(offline.iter())
        .map(|(&a, &b)| (a + b) % 17)
        .collect();
    assert_eq!(reconstructed.as_slice(), expected);
    let p = price(ring, Operation::ComposeProjection(shape));
    let mut work = ledger(p.work);
    let mut projection = [99; 8];
    compose_projection(
        ring,
        shape,
        &matrix,
        &u,
        &weights,
        &mut projection,
        p.caller_bytes,
        p.work,
        &mut work,
    )
    .unwrap();
    assert_eq!(projection, expected);
    assert_eq!(projection.as_slice(), reconstructed);
    assert_eq!(work.remaining(), 0);
}

#[test]
fn layer_refusals_preserve_output_and_scratch() {
    let ring = CoefficientRing::new(4, 17).unwrap();
    for op in [Operation::AddClientLayer, Operation::RemoveServerLayer] {
        let p = price(ring, op);
        // Every input position, output/scratch shape, literal byte/work cap,
        // cumulative work, and late noncanonical input gets its own refusal.
        for case in 0..14 {
            let mut a = vec![1; 4];
            let mut b = vec![2; 4];
            let mut factor = vec![3; 4];
            let mut noise = vec![-1; 4];
            let mut out = vec![
                55;
                match case {
                    4 => 3,
                    13 => 5,
                    _ => 4,
                }
            ];
            let mut scratch = vec![
                66;
                match case {
                    5 => 5,
                    12 => 3,
                    _ => 4,
                }
            ];
            match case {
                0 => {
                    a.pop();
                }
                1 => {
                    b.push(0);
                }
                2 => {
                    factor.pop();
                }
                3 if op == Operation::AddClientLayer => {
                    noise.pop();
                }
                3 => {
                    a.push(0);
                }
                9 => a[3] = 17,
                10 => b[3] = 17,
                11 => factor[3] = 17,
                _ => (),
            }
            let bytes = p.caller_bytes - usize::from(case == 6);
            let cap = p.work - usize::from(case == 7);
            let mut work = ledger(p.work - usize::from(case == 8));
            let result = if op == Operation::AddClientLayer {
                add_client_layer(
                    ring,
                    &a,
                    &b,
                    &factor,
                    &noise,
                    &mut out,
                    &mut scratch,
                    bytes,
                    cap,
                    &mut work,
                )
            } else {
                remove_server_layer(
                    ring,
                    &a,
                    &b,
                    &factor,
                    &mut out,
                    &mut scratch,
                    bytes,
                    cap,
                    &mut work,
                )
            };
            assert!(result.is_err(), "{op:?}, case {case}");
            assert!(out.iter().all(|&x| x == 55));
            assert!(scratch.iter().all(|&x| x == 66));
        }
    }
}

#[test]
fn projection_refusals_preserve_output() {
    let ring = CoefficientRing::new(4, 17).unwrap();
    let p = price(ring, Operation::ComposeProjection([2, 3, 2]));
    for case in 0..13 {
        let mut a = vec![1; 6];
        let mut u = vec![2; 6];
        let mut w = vec![3; 6];
        let mut out = vec![
            55;
            match case {
                3 => 5,
                12 => 3,
                _ => 4,
            }
        ];
        match case {
            0 => {
                a.pop();
            }
            1 => {
                u.push(0);
            }
            2 => {
                w.pop();
            }
            7 => a[5] = 17,
            8 => u[5] = 17,
            9 => w[5] = 17,
            _ => (),
        }
        let shape = match case {
            10 => [0, 3, 2],
            11 => [usize::MAX, 3, 2],
            _ => [2, 3, 2],
        };
        let result = compose_projection(
            ring,
            shape,
            &a,
            &u,
            &w,
            &mut out,
            p.caller_bytes - usize::from(case == 4),
            p.work - usize::from(case == 5),
            &mut ledger(p.work - usize::from(case == 6)),
        );
        assert!(result.is_err(), "case {case}");
        assert!(out.iter().all(|&x| x == 55));
    }
}

#[test]
fn projection_noncanonical_retry_preserves_original_ledger_residue() {
    let ring = CoefficientRing::new(4, 17).unwrap();
    let shape = [2, 3, 2];
    let a = [1; 6];
    let u = [2; 6];
    let mut w = [3, 3, 3, 3, 3, 17];
    let sentinel = [55, 66, 77, 88];
    let mut out = sentinel;
    let p = price(ring, Operation::ComposeProjection(shape));
    let initial = 2 * p.work - 1;
    let residue = p.work - 1;
    let mut work = ledger(initial);
    assert_eq!(work.remaining(), initial);
    assert_eq!(work.control_remaining(), 0);

    assert_eq!(
        compose_projection(
            ring,
            shape,
            &a,
            &u,
            &w,
            &mut out,
            p.caller_bytes,
            p.work,
            &mut work,
        ),
        Err(Error::Layout(LayoutError::NonCanonical))
    );
    assert_eq!(work.remaining(), residue);
    assert_eq!(out, sentinel);
    assert_eq!(work.control_remaining(), 0);

    // Retry unchanged first, then repair the input without replacing the ledger.
    for repair in [false, true] {
        if repair {
            w[5] = 3;
        }
        assert_eq!(
            compose_projection(
                ring,
                shape,
                &a,
                &u,
                &w,
                &mut out,
                p.caller_bytes,
                p.work,
                &mut work,
            ),
            Err(Error::Work(crate::row_fold::Error::WorkRefused))
        );
        assert_eq!(work.remaining(), residue);
        assert_eq!(out, sentinel);
        assert_eq!(work.control_remaining(), 0);
    }
}

#[test]
fn quote_boundaries_and_cumulative_debit_cannot_reset() {
    let ring = CoefficientRing::new(4, 17).unwrap();
    for op in [
        Operation::AddClientLayer,
        Operation::RemoveServerLayer,
        Operation::ComposeProjection([2, 3, 2]),
    ] {
        let p = price(ring, op);
        assert_eq!(quote(ring, op, p.caller_bytes, p.work).unwrap(), p);
        assert!(quote(ring, op, p.caller_bytes - 1, p.work).is_err());
        assert!(quote(ring, op, p.caller_bytes, p.work - 1).is_err());
    }
    assert!(
        quote(
            ring,
            Operation::ComposeProjection([usize::MAX, 2, 2]),
            usize::MAX,
            usize::MAX
        )
        .is_err()
    );
    assert!(
        quote(
            ring,
            Operation::ComposeProjection([1, 1, MAX_INVOCATION_WORK]),
            usize::MAX,
            usize::MAX
        )
        .is_err()
    );
    let p = price(ring, Operation::AddClientLayer);
    let mut work = ledger(p.work);
    let mut out = [55; 4];
    let mut scratch = [66; 4];
    assert!(
        add_client_layer(
            ring,
            &[1; 4],
            &[1, 1, 1, 17],
            &[1; 4],
            &[i64::MIN; 4],
            &mut out,
            &mut scratch,
            p.caller_bytes,
            p.work,
            &mut work
        )
        .is_err()
    );
    assert_eq!(work.remaining(), 0);
    assert!(
        add_client_layer(
            ring,
            &[1; 4],
            &[1; 4],
            &[1; 4],
            &[i64::MAX; 4],
            &mut out,
            &mut scratch,
            p.caller_bytes,
            p.work,
            &mut work
        )
        .is_err()
    );
    assert_eq!(out, [55; 4]);
    assert_eq!(scratch, [66; 4]);
}

#[test]
fn noisy_post_removal_decompression_composes_rectangular_projection_with_tail() {
    use crate::coefficient_compression::CompressionGeometry;

    // Caller-owned clear test arrays: neither factors nor noise carry key authority.
    let q = 97_i128;
    let ring = CoefficientRing::new(4, q as u64).unwrap();
    let shape = [2, 2, 3];
    let matrix = [1, 4, 6, 2];
    let u = [5, 3, 2, 7];
    let weights = [2, 5, 1, 7, 3, 4];
    // Independent row-major dot products, never compression-derived oracles.
    let uw = dense(&u, &weights, shape, q);
    let aw = dense(&matrix, &weights, shape, q);
    assert_eq!(uw, [31, 34, 17, 53, 31, 30]);
    assert_eq!(aw, [30, 17, 17, 26, 36, 14]);
    let masked: Vec<_> = matrix
        .iter()
        .zip(u.iter())
        .map(|(&a, &mask)| (i128::from(a) - i128::from(mask)).rem_euclid(q) as u64)
        .collect();
    let online = dense(&masked, &weights, shape, q);
    assert_eq!(online, [96, 80, 0, 70, 5, 81]);

    let geometry = CompressionGeometry::new(ring.degree(), aw.len()).unwrap();
    assert_eq!(geometry.polynomial_count(), 2);
    assert_eq!(geometry.packed_words(), 8);
    assert_eq!(geometry.valid_coefficients(0).unwrap(), 4);
    assert_eq!(geometry.valid_coefficients(1).unwrap(), 2);
    for scalar in 0..aw.len() {
        assert_eq!(geometry.address(scalar).unwrap(), (scalar / 4, scalar % 4));
    }
    let public = [[96, 3, 4, 7], [2, 95, 5, 9]];
    let server = [[2, 0, 96, 3], [4, 1, 3, 96]];
    let client = [[1, 95, 3, 1], [3, 2, 96, 4]];
    let noise = [[-2_i64, 1, -3, 4], [5, -6, 7, -8]];
    let prime = [[1_i64, -3, 2, 1], [-2, 1, -4, 2]];
    let compression = CoefficientCompression::new(ring, MAX_INVOCATION_WORK).unwrap();
    assert_eq!(compression.output_words(), 16);
    assert_eq!(compression.scratch_words(), 8);
    let p_add = price(ring, Operation::AddClientLayer);
    let p_remove = price(ring, Operation::RemoveServerLayer);
    // Compression has no ledger argument: debit its conservative envelope before
    // each call. One shared-operation ledger includes a test bookkeeping
    // allowance, not whole-resource pricing or admission, with no reset.
    let auxiliary_work = 128;
    let per_polynomial = compression
        .coefficient_work()
        .checked_mul(2)
        .and_then(|work| work.checked_add(p_add.work))
        .and_then(|work| work.checked_add(p_remove.work))
        .unwrap();
    let total_work = geometry
        .polynomial_count()
        .checked_mul(per_polynomial)
        .and_then(|work| work.checked_add(auxiliary_work))
        .unwrap();
    assert!(total_work <= MAX_INVOCATION_WORK);
    let mut work = ledger(total_work);
    work.debit(auxiliary_work, 0).unwrap();
    let mut packed = [99; 8];
    let mut decompressed = [99; 32];
    let mut wrong_factor_phase = [0; 8];
    let mut scratch = [88; 4];
    let mut compression_scratch = [88; 8];
    for polynomial in 0..geometry.polynomial_count() {
        let start = polynomial * 4;
        let valid = geometry.valid_coefficients(polynomial).unwrap();
        work.debit(compression.coefficient_work(), 0).unwrap();
        compression
            .compress(&uw[start..start + valid], &mut packed[start..start + 4])
            .unwrap();
        for coefficient in valid..4 {
            assert_eq!(packed[start + coefficient], 0);
        }
        // CoefficientRing::monomial uses negacyclic indexing. The independent
        // full convolution reduces high powers with X^4=-1, not cyclic wrap.
        let server_product = convolution(&public[polynomial], &server[polynomial], q);
        let client_product = convolution(&public[polynomial], &client[polynomial], q);
        let b: Vec<_> = (0..4)
            .map(|i| {
                (server_product[i]
                    + i128::from(packed[start + i])
                    + i128::from(noise[polynomial][i]))
                .rem_euclid(q) as u64
            })
            .collect();
        let mut layered = [99; 4];
        add_client_layer(
            ring,
            &public[polynomial],
            &b,
            &client[polynomial],
            &prime[polynomial],
            &mut layered,
            &mut scratch,
            p_add.caller_bytes,
            p_add.work,
            &mut work,
        )
        .unwrap();
        let mut removed = [99; 4];
        remove_server_layer(
            ring,
            &public[polynomial],
            &layered,
            &server[polynomial],
            &mut removed,
            &mut scratch,
            p_remove.caller_bytes,
            p_remove.work,
            &mut work,
        )
        .unwrap();
        let mut phase = [0; 4];
        for i in 0..4 {
            phase[i] = (i128::from(removed[i]) - client_product[i]).rem_euclid(q) as u64;
            wrong_factor_phase[start + i] =
                (i128::from(removed[i]) - server_product[i]).rem_euclid(q) as u64;
            // Padding has zero message, not zero noise. Derive independently
            // from UW and signed fixtures, never from the packed candidate.
            let message = if i < valid { uw[start + i] } else { 0 };
            let expected = (i128::from(message)
                + i128::from(noise[polynomial][i])
                + i128::from(prime[polynomial][i]))
            .rem_euclid(q) as u64;
            assert_eq!(phase[i], expected);
        }
        assert_ne!(phase.as_slice(), &wrong_factor_phase[start..start + 4]);
        work.debit(compression.coefficient_work(), 0).unwrap();
        compression
            .decompress(
                &phase,
                &mut decompressed[polynomial * 16..(polynomial + 1) * 16],
                &mut compression_scratch,
            )
            .unwrap();
    }
    let constants: Vec<_> = decompressed.chunks_exact(4).map(|poly| poly[0]).collect();
    assert_eq!(constants, [30, 32, 16, 58, 34, 25, 3, 91]);
    for (scalar, poly) in decompressed.chunks_exact(4).enumerate() {
        assert_eq!(poly, &[constants[scalar], 0, 0, 0]);
    }
    // Explicit tail calculation: 7 + (-4) = 3; -8 + 2 = -6 = 91 mod 97.
    assert_eq!(&decompressed[24..28], &[3, 0, 0, 0]);
    assert_eq!(&decompressed[28..32], &[91, 0, 0, 0]);

    let expected: Vec<_> = (0..aw.len())
        .map(|i| {
            (i128::from(aw[i])
                + i128::from(noise[i / 4][i % 4])
                + i128::from(prime[i / 4][i % 4]))
            .rem_euclid(q) as u64
        })
        .collect();
    assert_eq!(expected, [29, 15, 16, 31, 39, 9]);
    let correct = |values: &[u64]| -> Vec<u64> {
        (0..aw.len())
            .map(|i| (i128::from(values[i]) + i128::from(online[i])).rem_euclid(q) as u64)
            .collect()
    };
    assert_eq!(correct(&constants), expected);
    for i in 0..aw.len() {
        decompressed[i * 4] =
            (i128::from(decompressed[i * 4]) + i128::from(online[i])).rem_euclid(q) as u64;
        assert_eq!(&decompressed[i * 4..i * 4 + 4], &[expected[i], 0, 0, 0]);
    }
    let swapped_polynomials: Vec<_> = (0..8).map(|i| constants[(i + 4) % 8]).collect();
    let reversed_coefficients: Vec<_> = (0..8)
        .map(|i| constants[(i / 4) * 4 + 3 - i % 4])
        .collect();
    // Each control reaches a well-shaped algebraic result; none relies on panic
    // or refusal to distinguish wrong ordering, missing noise or wrong factors.
    assert_ne!(correct(&swapped_polynomials), expected);
    assert_ne!(correct(&reversed_coefficients), expected);
    assert_eq!(correct(&uw), aw);
    assert_ne!(correct(&uw), expected);
    assert_ne!(correct(&wrong_factor_phase), expected);
    assert_eq!(work.remaining(), 0);
}
