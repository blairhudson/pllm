//! AUTHOR-ONLY SYNTHETIC fixtures: N=8 is NOT a security parameter.
//! Deterministic arithmetic data, no cryptographic RNG or production encryption.
//! Decryption lives only in this client-side test module, never in the kernel.
use num_bigint::BigInt;
use pllm_core::torus_cmux::{
    cmux_into, quote, Error, Geometry, Limits, COMPONENTS, DIGITS, MAX_DEGREE,
};

const N: usize = 8;
const G: Geometry = Geometry { rank: 1, degree: N };

fn modulus() -> BigInt {
    BigInt::from(1_u8) << 64
}

fn residue(x: BigInt) -> u64 {
    let m = modulus();
    let r = ((x % &m) + &m) % &m;
    u64::try_from(r).unwrap()
}

/// Independent integer algebra: form the unreduced ordinary polynomial first,
/// then fold its high half with a minus sign; never accumulate wrapping words.
fn polynomial(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    assert_eq!(a.len(), N);
    assert_eq!(b.len(), N);
    let mut ordinary = vec![BigInt::from(0); 2 * N - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            ordinary[i + j] += x * y;
        }
    }
    (0..N)
        .map(|k| {
            let mut x = ordinary[k].clone();
            if k + N < ordinary.len() {
                x -= &ordinary[k + N];
            }
            x
        })
        .collect()
}

fn integers(a: &[u64]) -> Vec<BigInt> {
    a.iter().copied().map(BigInt::from).collect()
}

/// Oracle uses integer subtraction/modulo and division/remainder digits, not
/// kernel shifts/subtraction or its inner wrapping negacyclic accumulation.
fn oracle(f: &[u64], t: &[u64], selector: &[u64]) -> Vec<u64> {
    let mut result = integers(f);
    for c in 0..2 {
        let difference: Vec<_> = (0..N)
            .map(|k| BigInt::from(residue(BigInt::from(t[c * N + k]) - f[c * N + k])))
            .collect();
        let mut power = BigInt::from(1);
        for d in 0..8 {
            let digits: Vec<_> = difference.iter().map(|x| (x / &power) % 256_u32).collect();
            for out in 0..2 {
                let start = (c * 8 + d) * 2 * N + out * N;
                let product = polynomial(&digits, &integers(&selector[start..start + N]));
                for k in 0..N {
                    result[out * N + k] += &product[k];
                }
            }
            power *= 256_u32;
        }
    }
    result.into_iter().map(residue).collect()
}

fn limits() -> Limits {
    let q = quote(G).unwrap();
    Limits { max_work_events: q.work.total_events, max_caller_bytes: q.caller_bytes }
}

fn evaluate(f: &[u64], t: &[u64], selector: &[u64]) -> Vec<u64> {
    let q = quote(G).unwrap();
    let mut output = vec![0xdead_beef; q.glwe_words];
    let mut scratch = vec![u64::MAX; q.scratch_words];
    cmux_into(G, f, t, selector, &mut output, &mut scratch, limits()).unwrap();
    output
}

#[test]
fn exact_quotes_are_local_event_and_slice_counts() {
    let q = quote(G).unwrap();
    assert_eq!((COMPONENTS, DIGITS), (2, 8));
    assert_eq!(q.glwe_words, 2 * N);
    assert_eq!(q.ggsw_words, 32 * N);
    assert_eq!(q.scratch_words, 3 * N);
    assert_eq!(q.caller_bytes, 41 * N * 8);
    assert_eq!(q.work.difference_words, 2 * N);
    assert_eq!(q.work.digit_words, 16 * N);
    assert_eq!(q.work.output_copy_words, 2 * N);
    assert_eq!(q.work.negacyclic_macs, 32 * N * N);
    assert_eq!(q.work.total_events, 20 * N + 32 * N * N);
    // Repeated quotation/limits do not constitute cumulative owner admission.
    assert_eq!(quote(G).unwrap(), q);
}

#[test]
fn all_bits_digits_extrema_and_identity_diagonal_reconstruct() {
    let values = [0, 1, 255, 256, 1_u64 << 63, u64::MAX, u64::MAX - 1, 0x0123_4567_89ab_cdef];
    for x in values {
        let mut quotient = BigInt::from(x);
        let mut reconstructed = BigInt::from(0);
        let mut power = BigInt::from(1);
        for _ in 0..8 {
            let digit = &quotient % 256_u32;
            reconstructed += &digit * &power;
            quotient /= 256_u32;
            power *= 256_u32;
        }
        assert_eq!(reconstructed, BigInt::from(x));
        assert_eq!(quotient, BigInt::from(0));
    }
    let f: Vec<_> = (0..2 * N).map(|i| values[i % N]).collect();
    let t: Vec<_> = (0..2 * N).map(|i| values[(i + 3) % N]).collect();
    // Algebra-only identity GGSW, NOT an encrypted security fixture. Exercises
    // every digit and both component diagonals; catches top-digit truncation.
    let mut diagonal = vec![0; 32 * N];
    for c in 0..2 {
        for d in 0..8 {
            diagonal[((c * 8 + d) * 2 + c) * N] = 1_u64 << (8 * d);
        }
    }
    assert_eq!(evaluate(&f, &t, &diagonal), t);
    assert_eq!(oracle(&f, &t, &diagonal), t);
    let zero = vec![0; 32 * N];
    assert_eq!(evaluate(&f, &t, &zero), f);
}

#[test]
fn independent_external_product_oracle_extreme_words_and_negacyclic_wrap() {
    let f: Vec<_> = (0..2 * N).map(|i| if i % 2 == 0 { u64::MAX } else { 1 }).collect();
    let t: Vec<_> = (0..2 * N).map(|i| (i as u64).wrapping_mul(0xfedc_ba98_7654_3211)).collect();
    let rows: Vec<_> = (0..32 * N)
        .map(|i| u64::MAX.wrapping_sub((i as u64).wrapping_mul(0x1234_5678_9abc_def1)))
        .collect();
    assert_eq!(evaluate(&f, &t, &rows), oracle(&f, &t, &rows));
    // X^(N-1) times X is -1 in the quotient, not +1 or a cyclic shift.
    let mut delta = vec![0; 2 * N];
    delta[N - 1] = 255;
    let mut sparse = vec![0; 32 * N];
    sparse[1] = u64::MAX;
    let output = evaluate(&vec![0; 2 * N], &delta, &sparse);
    assert_eq!(output[0], 255);
    assert!(output[1..].iter().all(|&x| x == 0));
    assert_eq!(output, oracle(&vec![0; 2 * N], &delta, &sparse));
}

// Client-only deterministic synthetic encryption. Fixed tiny binary secret;
// masks are nonzero arithmetic patterns, not uniform cryptographic samples.
fn client_encrypt(message: &[u64], mask: &[u64], secret: &[u64], noise: &[i64]) -> Vec<u64> {
    let product = polynomial(&integers(mask), &integers(secret));
    let mut out: Vec<_> = (0..N)
        .map(|k| residue(BigInt::from(message[k]) + noise[k] - &product[k]))
        .collect();
    out.extend_from_slice(mask);
    out
}

fn client_decrypt(ciphertext: &[u64], secret: &[u64]) -> Vec<u64> {
    let product = polynomial(&integers(&ciphertext[N..]), &integers(secret));
    (0..N).map(|k| residue(BigInt::from(ciphertext[k]) + &product[k])).collect()
}

fn synthetic_selector(bit: u64, secret: &[u64]) -> (Vec<u64>, Vec<Vec<i64>>) {
    let mut rows = Vec::new();
    let mut errors = Vec::new();
    for c in 0..2 {
        for d in 0..8 {
            let row_number = c * 8 + d;
            let mask: Vec<_> = (0..N)
                .map(|k| ((row_number * N + k + 1) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                .collect();
            let noise: Vec<_> = (0..N).map(|k| if (k + row_number) % 2 == 0 { 1 } else { -1 }).collect();
            let mut row = client_encrypt(&vec![0; N], &mask, secret, &noise);
            row[c * N] = row[c * N].wrapping_add(bit.wrapping_mul(1_u64 << (8 * d)));
            rows.extend(row);
            errors.push(noise);
        }
    }
    (rows, errors)
}

#[test]
fn encrypted_both_bits_nonzero_masks_noise_client_final_phase_bound() {
    let secret = [1, 0, 1, 1, 0, 0, 1, 0];
    let m0: Vec<_> = (0..N).map(|k| (k as u64 + 1) << 48).collect();
    let m1: Vec<_> = (0..N).map(|k| u64::MAX - ((k as u64 + 3) << 49)).collect();
    let a0: Vec<_> = (0..N).map(|k| (k as u64 + 1).wrapping_mul(0xfedc_ba98_7654_3211)).collect();
    let a1: Vec<_> = (0..N).map(|k| (k as u64 + 2).wrapping_mul(0x1234_5678_9abc_def1)).collect();
    let e0 = [1, -1, 1, -1, 1, -1, 1, -1];
    let e1 = [-2, 2, -2, 2, -2, 2, -2, 2];
    let f = client_encrypt(&m0, &a0, &secret, &e0);
    let t = client_encrypt(&m1, &a1, &secret, &e1);
    for bit in [0, 1] {
        let (rows, errors) = synthetic_selector(bit, &secret);
        assert!(rows.chunks_exact(2 * N).all(|row| row[N..].iter().all(|&x| x != 0)));
        let output = evaluate(&f, &t, &rows);
        assert_eq!(output, oracle(&f, &t, &rows));
        let phase = client_decrypt(&output, &secret);
        // Exact integer noise prediction, independent division-based digits.
        let mut predicted = vec![BigInt::from(0); N];
        for c in 0..2 {
            for d in 0..8 {
                let power = BigInt::from(1_u8) << (8 * d);
                let digits: Vec<_> = (0..N)
                    .map(|k| {
                        let delta = residue(BigInt::from(t[c * N + k]) - f[c * N + k]);
                        (BigInt::from(delta) / &power) % 256_u32
                    })
                    .collect();
                let error: Vec<_> = errors[c * 8 + d].iter().copied().map(BigInt::from).collect();
                let term = polynomial(&digits, &error);
                for k in 0..N {
                    predicted[k] += &term[k];
                }
            }
        }
        let (message, input_noise) = if bit == 0 { (&m0, &e0) } else { (&m1, &e1) };
        let bound = BigInt::from(16 * N * 255 + 2);
        for k in 0..N {
            let error = &predicted[k] + input_noise[k];
            assert!(error >= -&bound && error <= bound);
            assert_eq!(phase[k], residue(BigInt::from(message[k]) + error));
            let circular = residue(BigInt::from(phase[k]) - message[k]);
            let distance = BigInt::from(circular).min(modulus() - circular);
            assert!(distance <= bound);
        }
    }
}

#[test]
fn every_refusal_is_atomic_for_output_and_scratch() {
    let q = quote(G).unwrap();
    let f = vec![0; q.glwe_words];
    let t = vec![u64::MAX; q.glwe_words];
    let rows = vec![1; q.ggsw_words];
    // Each exact slice and one-short/one-long slice, plus local limit edges.
    for which in 0..5 {
        for extra in [false, true] {
            let mut lengths = [q.glwe_words, q.glwe_words, q.ggsw_words, q.glwe_words, q.scratch_words];
            lengths[which] = if extra { lengths[which] + 1 } else { lengths[which] - 1 };
            let inputs = [vec![1; lengths[0]], vec![2; lengths[1]], vec![3; lengths[2]]];
            let mut output = vec![0xabcd; lengths[3]];
            let mut scratch = vec![0xdcba; lengths[4]];
            let before = (output.clone(), scratch.clone());
            assert_eq!(cmux_into(G, &inputs[0], &inputs[1], &inputs[2], &mut output, &mut scratch, limits()), Err(Error::BufferLength));
            assert_eq!((output, scratch), before);
        }
    }
    for cap in [
        Limits { max_work_events: q.work.total_events - 1, ..limits() },
        Limits { max_caller_bytes: q.caller_bytes - 1, ..limits() },
    ] {
        let mut output = vec![17; q.glwe_words];
        let mut scratch = vec![19; q.scratch_words];
        let before = (output.clone(), scratch.clone());
        assert_eq!(cmux_into(G, &f, &t, &rows, &mut output, &mut scratch, cap), Err(Error::ResourceLimit));
        assert_eq!((output, scratch), before);
    }
    for (geometry, error) in [
        (Geometry { rank: 0, degree: N }, Error::InvalidGeometry),
        (Geometry { rank: 2, degree: N }, Error::InvalidGeometry),
        (Geometry { rank: 1, degree: 0 }, Error::InvalidDegree),
        (Geometry { rank: 1, degree: 3 }, Error::InvalidDegree),
        (Geometry { rank: 1, degree: MAX_DEGREE + 1 }, Error::InvalidDegree),
        (Geometry { rank: usize::MAX, degree: N }, Error::Overflow),
        (Geometry { rank: 1, degree: usize::MAX }, Error::Overflow),
        (Geometry { rank: 1, degree: usize::MAX / 64 }, Error::Overflow),
    ] {
        let mut output = vec![17; q.glwe_words];
        let mut scratch = vec![19; q.scratch_words];
        let before = (output.clone(), scratch.clone());
        assert_eq!(quote(geometry), Err(error));
        assert_eq!(cmux_into(geometry, &f, &t, &rows, &mut output, &mut scratch, limits()), Err(error));
        assert_eq!((output, scratch), before);
    }
    // Exact work/storage/scratch accepted, including shared immutable inputs.
    assert_eq!(evaluate(&f, &f, &rows), f);
    assert_eq!(evaluate(&f, &t, &rows), oracle(&f, &t, &rows));
}
