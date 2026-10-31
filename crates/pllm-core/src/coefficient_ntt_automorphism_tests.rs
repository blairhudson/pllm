//! Source-only tests: public algebra fixtures, not keys or backend qualification.
//! Oracles use test-local arithmetic and coefficient scatter, never NTT gather.
use super::{Error, LogicalOperations, NegacyclicNtt, Operation};
use crate::coefficient_layout::{CoefficientRing, DegreeBound, LayoutError};

fn power(mut base: u64, mut exponent: u64, q: u64) -> u64 {
    let mut value = 1;
    while exponent != 0 {
        if exponent % 2 == 1 {
            value = (u128::from(value) * u128::from(base) % u128::from(q)) as u64;
        }
        base = (u128::from(base) * u128::from(base) % u128::from(q)) as u64;
        exponent /= 2;
    }
    value
}

fn plan(n: usize, q: u64, psi: u64) -> NegacyclicNtt {
    assert_eq!(power(psi, n as u64, q), q - 1);
    assert_eq!(power(psi, 2 * n as u64, q), 1);
    NegacyclicNtt::new(CoefficientRing::new(n, q).unwrap(), psi).unwrap()
}

fn residue(value: i128, q: u64) -> u64 {
    value.rem_euclid(i128::from(q)) as u64
}

// Schoolbook f(X^g): scatter each coefficient by exponent modulo 2N,
// folding X^N=-1. No spectral index permutation or production modular helpers.
fn substitute(input: &[u64], g: u64, q: u64) -> Vec<u64> {
    let n = input.len();
    let mut out = vec![0_i128; n];
    for (j, &value) in input.iter().enumerate() {
        let exponent = (j as u128 * u128::from(g)) % (2 * n) as u128;
        let signed = if exponent < n as u128 { i128::from(value) } else { -i128::from(value) };
        out[exponent as usize % n] += signed;
    }
    out.into_iter().map(|value| residue(value, q)).collect()
}

// Independent O(N^2) natural-order evaluation, no production transform/gather.
fn direct(input: &[u64], psi: u64, q: u64) -> Vec<u64> {
    (0..input.len()).map(|k| {
        let mut sum = 0_u128;
        for (j, &value) in input.iter().enumerate() {
            sum += u128::from(value) * u128::from(power(psi, ((2 * k + 1) * j) as u64, q));
        }
        (sum % u128::from(q)) as u64
    }).collect()
}

// Signed schoolbook product; only small public fixtures call this oracle.
fn product(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    let mut out = vec![0_i128; n];
    for (i, &a) in a.iter().enumerate() {
        for (j, &b) in b.iter().enumerate() {
            let value = i128::from(a) * i128::from(b);
            out[(i + j) % n] += if i + j < n { value } else { -value };
        }
    }
    out.into_iter().map(|value| residue(value, q)).collect()
}

fn combine(a: &[u64], b: &[u64], q: u64, subtract: bool) -> Vec<u64> {
    a.iter().zip(b).map(|(&a, &b)| {
        residue(i128::from(a) + if subtract { -i128::from(b) } else { i128::from(b) }, q)
    }).collect()
}

fn apply(ntt: &NegacyclicNtt, input: &[u64], g: u64) -> Vec<u64> {
    let mut out = vec![u64::MAX; input.len()];
    ntt.automorphism_into(input, g, &mut out).unwrap();
    out
}

#[test]
fn all_small_ring_odd_elements_and_basis_match_substitution_direct_dft() {
    for (n, q, psi) in [(1, 3, 2), (2, 5, 2), (4, 17, 2), (8, 17, 3), (16, 97, 28)] {
        let ntt = plan(n, q, psi);
        let mut cases = vec![vec![0; n], vec![q - 1; n]];
        cases.push((0..n).map(|j| (7 * j as u64 + 1) % q).collect());
        // Exhaust every basis position, including negative coefficients.
        for j in 0..n {
            for c in [1, q - 1] {
                let mut basis = vec![0; n];
                basis[j] = c;
                cases.push(basis);
            }
        }
        for input in cases {
            let original = input.clone();
            let spectrum = direct(&input, psi, q);
            let before = spectrum.clone();
            for g in (1..2 * n as u64).step_by(2) {
                let expected = substitute(&input, g, q);
                let actual = apply(&ntt, &spectrum, g);
                assert_eq!(actual, direct(&expected, psi, q), "N={n}, g={g}");
                let mut recovered = vec![0; n];
                ntt.inverse_into(&actual, &mut recovered).unwrap();
                assert_eq!(recovered, expected);
                assert_eq!(spectrum, before);
                assert_eq!(input, original);
            }
        }
    }
}

#[test]
fn exhaustive_tiny_polynomials_all_odd_elements() {
    for (n, q, psi) in [(1, 3_u64, 2), (2, 5, 2)] {
        let ntt = plan(n, q, psi);
        for mut encoded in 0..q.pow(n as u32) {
            let mut coefficients = vec![0; n];
            for value in &mut coefficients {
                *value = encoded % q;
                encoded /= q;
            }
            let spectrum = direct(&coefficients, psi, q);
            for g in (1..2 * n as u64).step_by(2) {
                assert_eq!(apply(&ntt, &spectrum, g), direct(&substitute(&coefficients, g, q), psi, q));
            }
        }
    }
}

#[test]
fn identity_conjugation_composition_and_inverse_on_arbitrary_spectra() {
    for (n, q, psi) in [(1, 17, 16), (2, 17, 4), (4, 17, 2), (8, 17, 3), (16, 97, 28)] {
        let ntt = plan(n, q, psi);
        let input: Vec<_> = (0..n).map(|k| (3 * k as u64 + 2) % q).collect();
        let two_n = 2 * n as u64;
        assert_eq!(apply(&ntt, &input, 1), input);
        // Conjugation has a separate closed form: reverse natural-order slots.
        assert_eq!(apply(&ntt, &input, two_n - 1), input.iter().rev().copied().collect::<Vec<_>>());
        for g in (1..two_n).step_by(2) {
            let inverse = (1..two_n).step_by(2).find(|&h| g * h % two_n == 1).unwrap();
            let first = apply(&ntt, &input, g);
            assert_eq!(apply(&ntt, &first, inverse), input);
            for h in (1..two_n).step_by(2) {
                assert_eq!(apply(&ntt, &first, h), apply(&ntt, &input, g * h % two_n));
            }
        }
    }
}

#[test]
fn automorphism_refusals_precede_all_stores_with_explicit_error_order() {
    let ntt = plan(8, 17, 3);
    let good = [0, 1, 2, 3, 4, 5, 15, 16];
    let sentinel: Vec<_> = (0..8).map(|i| u64::MAX - i).collect();
    for g in [0, 2, 4, 14, 16, 17, u64::MAX - 1, u64::MAX] {
        let mut out = sentinel.clone();
        assert_eq!(ntt.automorphism_into(&good, g, &mut out), Err(Error::InvalidAutomorphism));
        assert_eq!(out, sentinel);
        // Invalid element precedes bad lengths and residues, too.
        assert_eq!(ntt.automorphism_into(&[17], g, &mut out), Err(Error::InvalidAutomorphism));
        assert_eq!(out, sentinel);
    }
    for length in [0, 7, 9] {
        let malformed = vec![17; length];
        let mut out = sentinel.clone();
        assert_eq!(ntt.automorphism_into(&malformed, 3, &mut out), Err(Error::BufferLength));
        assert_eq!(out, sentinel);
        let mut wrong_out = vec![u64::MAX; length];
        let before = wrong_out.clone();
        assert_eq!(ntt.automorphism_into(&[17; 8], 3, &mut wrong_out), Err(Error::BufferLength));
        assert_eq!(wrong_out, before);
    }
    for at in 0..8 {
        for value in [17, u64::MAX] {
            let mut bad = good;
            bad[at] = value;
            let original = bad;
            let mut out = sentinel.clone();
            assert_eq!(ntt.automorphism_into(&bad, 3, &mut out), Err(Error::NonCanonical));
            assert_eq!(out, sentinel);
            assert_eq!(bad, original);
        }
    }
    assert_eq!(good, [0, 1, 2, 3, 4, 5, 15, 16]);
}

#[test]
fn public_8192_support_index_quotient_bounds_and_logical_quote_only() {
    let n = 8192;
    let q = 65537;
    assert_eq!(CoefficientRing::new(n, q), Err(LayoutError::InvalidDegree));
    let ring = CoefficientRing::bounded(n, q, DegreeBound::Public8192).unwrap();
    assert_eq!(power(81, n as u64, q), q - 1);
    assert_eq!(power(81, 2 * n as u64, q), 1);
    let ntt = NegacyclicNtt::new(ring, 81).unwrap();
    let last = (2 * n - 1) as u64;
    let max_product = last.checked_mul(last).unwrap();
    assert_eq!(max_product, 268_402_689);
    assert!(max_product < 1 << 28);
    assert!(max_product <= u64::from(u32::MAX));
    assert_eq!(max_product / (2 * n) as u64, 2 * n as u64 - 2);
    // No large DFT: arbitrary spectra, reversal oracle, and permutation inverse.
    let input: Vec<_> = (0..n).map(|k| k as u64).collect();
    assert_eq!(apply(&ntt, &input, last), input.iter().rev().copied().collect::<Vec<_>>());
    // NEXUS form g=N/h+1 for public h=2, and its elementary odd-group inverse.
    let g = n as u64 / 2 + 1;
    let inverse = (1..2 * n as u64).step_by(2).find(|&h| g * h % (2 * n as u64) == 1).unwrap();
    assert_eq!(apply(&ntt, &apply(&ntt, &input, g), inverse), input);
    for plan in [plan(1, 17, 16), plan(8, 17, 3), ntt] {
        let words = plan.ring().degree() as u64;
        assert_eq!(plan.operations(Operation::Automorphism), LogicalOperations {
            checks: words + 5,
            residue_checks: words,
            moves: words,
            ..LogicalOperations::default()
        });
    }
    assert_eq!(CoefficientRing::bounded(16384, q, DegreeBound::Public8192), Err(LayoutError::InvalidDegree));
}

// Exercise the public kernel through actual transforms; expected coefficients
// still come from independent signed substitution, not this round trip.
fn transform_coefficients(ntt: &NegacyclicNtt, input: &[u64], g: u64) -> Vec<u64> {
    let n = input.len();
    let mut spectrum = vec![0; n];
    ntt.forward_into(input, &mut spectrum).unwrap();
    let transformed = apply(ntt, &spectrum, g);
    let mut coefficients = vec![0; n];
    ntt.inverse_into(&transformed, &mut coefficients).unwrap();
    assert_eq!(coefficients, substitute(input, g, ntt.ring().modulus()));
    coefficients
}

#[test]
fn schoolbook_single_and_layered_phase_change_key_domain_not_key_switch() {
    let ntt = plan(8, 17, 3);
    let q = 17;
    let g = 5; // N/h+1 with h=2.
    // Public deterministic polynomial fixtures, never secret keys or encryption.
    let a = [1, 2, 0, 0, 0, 0, 0, 0];
    let server_factor = [0, 1, 0, 0, 0, 0, 0, 0];
    let client_factor = [1, 0, 1, 0, 0, 0, 0, 0];
    let message = [2, 4, 0, 1, 0, 0, 0, 3];
    let error = [1, 0, 16, 0, 0, 0, 0, 0];
    let phase = combine(&message, &error, q, false);
    let b = combine(&phase, &product(&a, &server_factor, q), q, false);
    let sigma_a = transform_coefficients(&ntt, &a, g);
    let sigma_b = transform_coefficients(&ntt, &b, g);
    let sigma_server = transform_coefficients(&ntt, &server_factor, g);
    let sigma_phase = substitute(&phase, g, q);
    assert_eq!(combine(&sigma_b, &product(&sigma_a, &sigma_server, q), q, true), sigma_phase);
    assert_ne!(combine(&sigma_b, &product(&sigma_a, &server_factor, q), q, true), sigma_phase);
    // Appendix B adds A*sC+e' to b. Remove TRANSFORMED server factor first;
    // supplied client error survives alongside the original message/error.
    let client_error = [0, 1, 0, 0, 16, 0, 0, 0];
    let layered_phase = combine(&phase, &client_error, q, false);
    let sigma_layered_phase = substitute(&layered_phase, g, q);
    let layered_b = combine(&combine(&b, &product(&a, &client_factor, q), q, false), &client_error, q, false);
    let sigma_layered = transform_coefficients(&ntt, &layered_b, g);
    let remaining = combine(&sigma_layered, &product(&sigma_a, &sigma_server, q), q, true);
    let sigma_client = transform_coefficients(&ntt, &client_factor, g);
    assert_eq!(remaining, combine(&sigma_layered_phase, &product(&sigma_a, &sigma_client, q), q, false));
    assert_eq!(combine(&remaining, &product(&sigma_a, &sigma_client, q), q, true), sigma_layered_phase);
    assert_ne!(combine(&remaining, &product(&sigma_a, &client_factor, q), q, true), sigma_layered_phase);
    // This proves a public polynomial identity, not evaluation-key correctness,
    // secret-factor execution, key switching, or encrypted NEXUS/Qwen execution.
}

#[test]
fn permuted_sources_feed_existing_ordered_rns_embedding_contract() {
    use crate::coefficient_rns_contraction::{self as rns, Evaluation, RnsContraction, SourceSpectra};

    let n = 4;
    // Deliberately unsorted moduli. Root search is test-local, bounded, public.
    let moduli = [41, 17, 73, 97];
    let roots = moduli.map(|q| {
        (2..q).find(|&root| power(root, n as u64, q) == q - 1
            && power(root, 2 * n as u64, q) == 1).unwrap()
    });
    let targets: [NegacyclicNtt; 4] = std::array::from_fn(|i| plan(n, moduli[i], roots[i]));
    let sources = [targets[0].clone(), targets[1].clone(), targets[2].clone()];
    let contraction = RnsContraction::new(&sources, &targets).unwrap();
    let g = 3;
    let mut spectra = vec![0; 3 * n];
    let mut digits = Vec::new();
    for (i, source) in sources.iter().enumerate() {
        let q = source.ring().modulus();
        let coefficients = [q - 1, 1 + i as u64, 0, q - 2];
        let sigma = substitute(&coefficients, g, q);
        let before = direct(&coefficients, roots[i], q);
        source.automorphism_into(&before, g, &mut spectra[i * n..(i + 1) * n]).unwrap();
        assert_eq!(&spectra[i * n..(i + 1) * n], direct(&sigma, roots[i], q).as_slice());
        digits.push(sigma);
    }
    let storage = contraction.storage(Evaluation::Embed).unwrap();
    let ceiling = contraction.work(Evaluation::Embed).unwrap();
    let mut output = vec![u64::MAX; storage.output_words];
    let mut scratch = vec![u64::MAX; storage.scratch_words];
    contraction.embed_into(SourceSpectra { contexts: &sources, values: &spectra },
        &mut output, &mut scratch, ceiling).unwrap();
    for (i, digit) in digits.iter().enumerate() {
        for limb in 0..4 {
            let lifted: Vec<_> = digit.iter().map(|&v| v % moduli[limb]).collect();
            let start = (i * 4 + limb) * n;
            assert_eq!(&output[start..start + n], direct(&lifted, roots[limb], moduli[limb]).as_slice());
        }
    }
    // Identical ring but different root remains a context mismatch. Applying
    // sigma changes data only; it does not authorize relabeling root contexts.
    let mut wrong_contexts = sources.clone();
    wrong_contexts[0] = plan(n, moduli[0], power(roots[0], 3, moduli[0]));
    let before_output = output.clone();
    let before_scratch = scratch.clone();
    assert_eq!(contraction.embed_into(SourceSpectra { contexts: &wrong_contexts, values: &spectra },
        &mut output, &mut scratch, ceiling), Err(rns::Error::ContextMismatch));
    assert_eq!(output, before_output);
    assert_eq!(scratch, before_scratch);
}
