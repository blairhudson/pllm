//! Independent BigInt oracles; source-only until ordinary test execution is allowed.
use super::*;
use crate::coefficient_layout::LayoutError;
use num_bigint::BigInt;

fn residue(value: BigInt, q: u64) -> u64 {
    let modulus = BigInt::from(q);
    let value = ((value % &modulus) + &modulus) % &modulus;
    let (_, digits) = value.to_u64_digits();
    digits.first().copied().unwrap_or(0)
}

fn power(a: u64, exponent: u64, q: u64) -> u64 {
    residue(BigInt::from(a).modpow(&BigInt::from(exponent), &BigInt::from(q)), q)
}

// Natural-order evaluation, no shared modular helper or NTT scheduling.
fn direct(input: &[u64], psi: u64, q: u64, inverse: bool) -> Vec<u64> {
    let n = input.len();
    let modulus = BigInt::from(q);
    (0..n).map(|at| {
        let mut sum = BigInt::from(0);
        for (from, &value) in input.iter().enumerate() {
            let exponent = if inverse { (2 * from + 1) * at } else { (2 * at + 1) * from };
            let root = if inverse { power(psi, q - 2, q) } else { psi };
            sum += BigInt::from(value)
                * BigInt::from(root).modpow(&BigInt::from(exponent), &modulus);
        }
        if inverse {
            sum *= BigInt::from(power(n as u64, q - 2, q));
        }
        residue(sum, q)
    }).collect()
}

// Signed O(N^2) convolution in Z[X]/(X^N+1), reduced only at the end.
fn convolution(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    let mut out = vec![BigInt::from(0); n];
    for (i, &a) in a.iter().enumerate() {
        for (j, &b) in b.iter().enumerate() {
            let product = BigInt::from(a) * BigInt::from(b);
            if i + j >= n {
                out[(i + j) % n] -= product;
            } else {
                out[i + j] += product;
            }
        }
    }
    out.into_iter().map(|v| residue(v, q)).collect()
}

fn plan(n: usize, q: u64, psi: u64) -> NegacyclicNtt {
    assert_eq!(power(psi, n as u64, q), q - 1);
    assert_eq!(power(psi, 2 * n as u64, q), 1);
    NegacyclicNtt::new(CoefficientRing::new(n, q).unwrap(), psi).unwrap()
}

fn product(ntt: &NegacyclicNtt, a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut af = vec![0; a.len()];
    let mut bf = vec![0; a.len()];
    let mut cf = vec![0; a.len()];
    let mut out = vec![0; a.len()];
    ntt.forward_into(a, &mut af).unwrap();
    ntt.forward_into(b, &mut bf).unwrap();
    ntt.pointwise_mul_into(&af, &bf, &mut cf).unwrap();
    ntt.inverse_into(&cf, &mut out).unwrap();
    out
}

#[test]
fn natural_order_and_independent_inverse_and_signed_convolution() {
    for (n, q, psi) in [(1, 17, 16), (2, 17, 4), (4, 17, 2), (8, 17, 3), (16, 97, 28)] {
        let ntt = plan(n, q, psi);
        let mut sparse = vec![0; n];
        sparse[n - 1] = q - 1;
        let cases = [
            vec![0; n],
            vec![q - 1; n],
            sparse,
            (0..n).map(|i| (i as u64 * 7 + 1) % q).collect(),
            (0..n).map(|i| if i & 1 == 0 { q - 1 } else { q - 2 }).collect(),
        ];
        for a in &cases {
            let original = a.clone();
            let mut out = vec![u64::MAX; n];
            ntt.forward_into(a, &mut out).unwrap();
            assert_eq!(out, direct(a, psi, q, false));
            // Arbitrary input spectrum, not merely forward output.
            ntt.inverse_into(a, &mut out).unwrap();
            assert_eq!(out, direct(a, psi, q, true));
            assert_eq!(*a, original);
            for b in &cases {
                let b_original = b.clone();
                assert_eq!(product(&ntt, a, b), convolution(a, b, q));
                ntt.pointwise_add_into(a, b, &mut out).unwrap();
                let expected: Vec<_> = a.iter().zip(b).map(|(&a, &b)| {
                    residue(BigInt::from(a) + BigInt::from(b), q)
                }).collect();
                assert_eq!(out, expected);
                assert_eq!(*b, b_original);
            }
            // Same immutable input in both positions is explicitly supported.
            ntt.pointwise_mul_into(a, a, &mut out).unwrap();
            let expected: Vec<_> = a.iter().map(|&v| residue(BigInt::from(v) * BigInt::from(v), q)).collect();
            assert_eq!(out, expected);
            ntt.pointwise_add_into(a, a, &mut out).unwrap();
            let expected: Vec<_> = a.iter().map(|&v| residue(BigInt::from(v) * 2, q)).collect();
            assert_eq!(out, expected);
            assert_eq!(*a, original);
        }
        if n > 1 {
            let mut x = vec![0; n];
            x[1] = 1;
            let mut spectrum = vec![0; n];
            ntt.forward_into(&x, &mut spectrum).unwrap();
            let expected: Vec<_> = (0..n).map(|k| power(psi, (2 * k + 1) as u64, q)).collect();
            assert_eq!(spectrum, expected);
        }
    }
}

#[test]
fn degree_one_and_known_mersenne_prime_product_boundary() {
    // Known Mersenne prime M_61; no near-61-bit N>1 root is asserted.
    let q = (1_u64 << 61) - 1;
    // Independent Lucas–Lehmer certificate for this mathematical fixture.
    let modulus = BigInt::from(q);
    let mut s = BigInt::from(4);
    for _ in 0..59 {
        s = (&s * &s - 2) % &modulus;
    }
    assert_eq!(s, BigInt::from(0));
    let ntt = plan(1, q, q - 1);
    assert_eq!(ntt.n_inv, 1);
    assert_eq!(ntt.forward_stage_roots, [0; 13]);
    assert_eq!(ntt.inverse_stage_roots, [0; 13]);
    let mut out = [0];
    ntt.forward_into(&[q - 1], &mut out).unwrap();
    assert_eq!(out, [q - 1]);
    ntt.inverse_into(&[q - 1], &mut out).unwrap();
    assert_eq!(out, [q - 1]);
    ntt.pointwise_mul_into(&[q - 1], &[q - 1], &mut out).unwrap();
    assert_eq!(out[0], residue(BigInt::from(q - 1) * BigInt::from(q - 1), q));
    assert_eq!(out, [1]);
    ntt.pointwise_add_into(&[q - 1], &[q - 1], &mut out).unwrap();
    assert_eq!(out, [q - 2]);
}

#[test]
fn max_degree_roundtrip_only_not_quadratic_qualification() {
    // 65537 prime; psi=3^8=6561. BigInt independently checks exact root order.
    let ntt = plan(4096, 65537, 6561);
    assert_eq!(ntt.ring().degree(), crate::coefficient_layout::MAX_DEGREE);
    let input: Vec<_> = (0..4096).map(|i| if i & 1 == 0 { 65536 } else { i as u64 }).collect();
    let original = input.clone();
    let mut transformed = vec![0; 4096];
    let mut recovered = vec![0; 4096];
    ntt.forward_into(&input, &mut transformed).unwrap();
    assert!(transformed.iter().all(|&v| v < 65537));
    ntt.inverse_into(&transformed, &mut recovered).unwrap();
    assert_eq!(recovered, input);
    assert_eq!(input, original);
}

// Public mathematical fixture only, not a production/security parameter set.
// Pocklington: F=q-1=2^16>sqrt(q); the sole prime factor 2 is certified
// trivially. BigInt witnesses and gcd are independent of production MR/pow.
fn public_fermat_plan(n: usize) -> NegacyclicNtt {
    use crate::coefficient_layout::DegreeBound;
    let q = 65537_u64;
    assert_eq!(q - 1, 2_u64.pow(16));
    assert!(BigInt::from(q - 1).pow(2) > BigInt::from(q));
    assert_eq!(power(3, q - 1, q), 1);
    assert_eq!(power(3, (q - 1) / 2, q), q - 1);
    assert_eq!(absolute_gcd(BigInt::from(q - 2), BigInt::from(q)), BigInt::from(1));
    assert!([4096, 8192].contains(&n));
    let psi = power(3, (q - 1) / (2 * n as u64), q);
    assert_eq!(psi, if n == 4096 { 6561 } else { 81 });
    // Since 2N is a power of two, these establish exact order 2N.
    assert_eq!(power(psi, n as u64, q), q - 1);
    assert_eq!(power(psi, 2 * n as u64, q), 1);
    let ring = CoefficientRing::bounded(n, q, DegreeBound::Public8192).unwrap();
    let ntt = NegacyclicNtt::new(ring, psi).unwrap();
    assert_eq!(ntt.ring(), ring);
    if n == 4096 {
        assert_eq!(ntt, plan(n, q, psi));
    } else {
        assert_eq!(CoefficientRing::new(n, q), Err(LayoutError::InvalidDegree));
    }
    ntt
}

#[test]
fn large_forward_basis_and_direct_inverse_spectrum_basis_independent_bigint() {
    for n in [4096, 8192] {
        let ntt = public_fermat_plan(n);
        let q = ntt.ring().modulus();
        let psi = power(3, (q - 1) / (2 * n as u64), q);
        let inverse_root = power(psi, q - 2, q);
        let inverse_n = power(n as u64, q - 2, q);
        assert_eq!(residue(BigInt::from(psi) * BigInt::from(inverse_root), q), 1);
        assert_eq!(residue(BigInt::from(n) * BigInt::from(inverse_n), q), 1);
        for at in [1, n / 2, n - 1] {
            for c in [1, q - 1] {
                let mut basis = vec![0; n];
                basis[at] = c;
                let original = basis.clone();
                let mut out = vec![u64::MAX; n];
                let forward: Vec<_> = (0..n).map(|k| {
                    residue(BigInt::from(c) * BigInt::from(power(psi, ((2 * k + 1) * at) as u64, q)), q)
                }).collect();
                ntt.forward_into(&basis, &mut out).unwrap();
                assert_eq!(out, forward);
                assert!(out[n / 2..].iter().all(|&value| value != 0));
                assert_ne!(out[n - 1], 0);
                // Input is a directly constructed spectrum basis, never forward
                // output. This checks inverse ordering, scale and inverse twist.
                let inverse: Vec<_> = (0..n).map(|j| {
                    residue(BigInt::from(c) * BigInt::from(inverse_n)
                        * BigInt::from(power(inverse_root, ((2 * at + 1) * j) as u64, q)), q)
                }).collect();
                ntt.inverse_into(&basis, &mut out).unwrap();
                assert_eq!(out, inverse);
                assert!(out[n / 2..].iter().all(|&value| value != 0));
                assert_ne!(out[n - 1], 0);
                assert_eq!(basis, original);
            }
        }
    }
}

#[test]
fn large_dense_times_monomial_signed_wrap_has_linear_bigint_oracle() {
    for n in [4096, 8192] {
        let ntt = public_fermat_plan(n);
        let q = ntt.ring().modulus();
        let dense: Vec<_> = (0..n).map(|j| 1 + ((17 * j as u64 + 5) % (q - 1))).collect();
        let original = dense.clone();
        for shift in [1, n / 2, n - 1] {
            for c in [1, q - 1] {
                let mut monomial = vec![0; n];
                monomial[shift] = c;
                let before = monomial.clone();
                // Coefficient formula in Z[X]/(X^N+1), without production
                // monomial, modular helpers, NTT scheduling or a roundtrip oracle.
                let expected: Vec<_> = (0..n).map(|k| {
                    let source = if k >= shift { k - shift } else { k + n - shift };
                    let value = BigInt::from(c) * BigInt::from(dense[source]);
                    residue(if k >= shift { value } else { -value }, q)
                }).collect();
                assert_eq!(product(&ntt, &dense, &monomial), expected);
                assert!(expected[n / 2..].iter().all(|&value| value != 0));
                assert_ne!(expected[n - 1], 0);
                assert_eq!(dense, original);
                assert_eq!(monomial, before);
            }
        }
    }
}

#[test]
fn expanded_descriptor_root_refusals_and_all_operation_errors_are_atomic() {
    let n = 8192;
    let ntt = public_fermat_plan(n);
    let ring = ntt.ring();
    let q = ring.modulus();
    for psi in [0, 1, power(81, 2, q)] {
        assert_eq!(NegacyclicNtt::new(ring, psi), Err(Error::InvalidRoot));
    }
    for psi in [q, u64::MAX] {
        assert_eq!(NegacyclicNtt::new(ring, psi), Err(Error::NonCanonical));
    }
    // Modulus rejection still precedes root canonicality when both are bad.
    for modulus in [17, 65535] {
        let incompatible = CoefficientRing::bounded(n, modulus, crate::coefficient_layout::DegreeBound::Public8192).unwrap();
        assert_eq!(NegacyclicNtt::new(incompatible, u64::MAX), Err(Error::InvalidModulus));
    }
    let good: Vec<_> = (0..n).map(|i| i as u64).collect();
    let original = good.clone();
    let unary: [Unary; 2] = [NegacyclicNtt::forward_into, NegacyclicNtt::inverse_into];
    let binary: [Binary; 2] = [NegacyclicNtt::pointwise_add_into, NegacyclicNtt::pointwise_mul_into];
    for call in unary {
        for length in [0, n - 1, n + 1] {
            let bad = vec![q; length];
            let mut out = sentinels(n);
            let before = out.clone();
            assert_eq!(call(&ntt, &bad, &mut out), Err(Error::BufferLength));
            assert_eq!(out, before);
            assert_eq!(bad, vec![q; length]);
            let mut short = sentinels(length);
            let before = short.clone();
            assert_eq!(call(&ntt, &vec![q; n], &mut short), Err(Error::BufferLength));
            assert_eq!(short, before);
        }
        for at in [0, n / 2, n - 1] {
            for value in [q, u64::MAX] {
                let mut bad = good.clone();
                bad[at] = value;
                let snapshot = bad.clone();
                let mut out = sentinels(n);
                let before = out.clone();
                assert_eq!(call(&ntt, &bad, &mut out), Err(Error::NonCanonical));
                assert_eq!(out, before);
                assert_eq!(bad, snapshot);
            }
        }
    }
    for call in binary {
        for length in [0, n - 1, n + 1] {
            let bad = vec![q; length];
            let mut out = sentinels(n);
            let before = out.clone();
            assert_eq!(call(&ntt, &bad, &good, &mut out), Err(Error::BufferLength));
            assert_eq!(out, before);
            assert_eq!(call(&ntt, &good, &bad, &mut out), Err(Error::BufferLength));
            assert_eq!(out, before);
            assert_eq!(bad, vec![q; length]);
            let mut short = sentinels(length);
            let before = short.clone();
            let noncanonical = vec![q; n];
            assert_eq!(call(&ntt, &noncanonical, &good, &mut short), Err(Error::BufferLength));
            assert_eq!(short, before);
            assert_eq!(call(&ntt, &good, &noncanonical, &mut short), Err(Error::BufferLength));
            assert_eq!(short, before);
        }
        for at in [0, n / 2, n - 1] {
            for value in [q, u64::MAX] {
                let mut bad = good.clone();
                bad[at] = value;
                let snapshot = bad.clone();
                let mut out = sentinels(n);
                let before = out.clone();
                assert_eq!(call(&ntt, &bad, &good, &mut out), Err(Error::NonCanonical));
                assert_eq!(out, before);
                assert_eq!(call(&ntt, &good, &bad, &mut out), Err(Error::NonCanonical));
                assert_eq!(out, before);
                assert_eq!(bad, snapshot);
            }
        }
    }
    assert_eq!(good, original);
}

fn absolute_gcd(mut a: BigInt, mut b: BigInt) -> BigInt {
    if a < BigInt::from(0) {
        a = -a;
    }
    if b < BigInt::from(0) {
        b = -b;
    }
    while b != BigInt::from(0) {
        let remainder = a % &b;
        a = b;
        b = remainder;
    }
    a
}

#[test]
fn near_limit_candidate_pocklington_n8_independent_oracles() {
    // Candidate only until this ordinary test passes. No repository certificate
    // or production parameter adoption is claimed. Do not use production MR
    // to establish this fixture: certify it before constructing any descriptor.
    let q = 0x1fffffffffe00001_u64;
    assert_eq!(q, (1_u64 << 61) - (1_u64 << 21) + 1);
    let modulus = BigInt::from(q);
    let factors = [(2_u64, 21_u32), (3, 1), (5, 2), (11, 1), (17, 1), (31, 1), (41, 1), (61681, 1)];
    let mut factor_product = BigInt::from(1);
    assert!(249_u64 * 249 > 61681);
    for &(p, multiplicity) in &factors {
        assert!((2..=61681).contains(&p));
        for divisor in 2_u64..=248 {
            if divisor * divisor > p {
                break;
            }
            assert_ne!(p % divisor, 0, "factor {p} is composite");
        }
        factor_product *= BigInt::from(p).pow(multiplicity);
    }
    assert_eq!(factor_product, BigInt::from(q - 1));
    assert!(&factor_product * &factor_product > modulus);
    // Pocklington with F=q-1: every distinct prime divisor requires its own
    // witness. Bounded search uses at most 8*63*2=1008 BigInt modular powers;
    // missing witness is a failure, never a skip, fallback, or assumed prime.
    for &(p, _) in &factors {
        let witness = (2_u64..=64).find(|&a| {
            power(a, q - 1, q) == 1
                && absolute_gcd(
                    BigInt::from(power(a, (q - 1) / p, q)) - BigInt::from(1),
                    modulus.clone(),
                ) == BigInt::from(1)
        });
        assert!(witness.is_some(), "no bounded Pocklington witness for factor {p}");
    }
    // Independent BigInt root search, also fail-closed and before constructor.
    let psi = (2_u64..=64).find_map(|a| {
        let candidate = power(a, (q - 1) / 16, q);
        if power(candidate, 8, q) == q - 1 && power(candidate, 16, q) == 1 {
            Some(candidate)
        } else {
            None
        }
    }).expect("no bounded order-16 root witness for candidate fixture");
    let ntt = NegacyclicNtt::new(CoefficientRing::new(8, q).unwrap(), psi).unwrap();
    let cases = [
        [q - 1; 8],
        [q - 2; 8],
        [q - 1, 1, q - 2, 0, q / 2, 7, q - 7, 2],
    ];
    for a in &cases {
        let original = *a;
        let mut out = vec![0; 8];
        ntt.forward_into(a, &mut out).unwrap();
        assert_eq!(out, direct(a, psi, q, false));
        // An arbitrary spectrum, not a roundtrip-only inverse check.
        ntt.inverse_into(a, &mut out).unwrap();
        assert_eq!(out, direct(a, psi, q, true));
        for b in &cases {
            let b_original = *b;
            ntt.pointwise_add_into(a, b, &mut out).unwrap();
            let expected: Vec<_> = a.iter().zip(b).map(|(&a, &b)| {
                residue(BigInt::from(a) + BigInt::from(b), q)
            }).collect();
            assert_eq!(out, expected);
            ntt.pointwise_mul_into(a, b, &mut out).unwrap();
            let expected: Vec<_> = a.iter().zip(b).map(|(&a, &b)| {
                residue(BigInt::from(a) * BigInt::from(b), q)
            }).collect();
            assert_eq!(out, expected);
            // Nine signed O(8^2) convolutions, only 576 oracle products.
            assert_eq!(product(&ntt, a, b), convolution(a, b, q));
            assert_eq!(*b, b_original);
            assert_eq!(*a, original);
        }
    }
}

#[test]
fn prime_and_root_refusals() {
    for q in [9, 15, 25, 49, 561, 1105, 1729, 3215031751, 2152302898747, 341550071728321] {
        let ring = CoefficientRing::new(1, q).unwrap();
        assert_eq!(NegacyclicNtt::new(ring, q - 1), Err(Error::InvalidModulus));
    }
    // Full-u64 deterministic helper, including strong pseudoprime and large prime.
    for q in [0, 1, 4, 341550071728321, 3825123056546413051, u64::MAX] {
        assert!(!prime(q), "{q}");
    }
    for q in [2, 3, 5, 17, 73, 193, (1_u64 << 61) - 1, 18446744073709551557] {
        assert!(prime(q), "{q}");
    }
    // 73 divides MR base 28178: base%q==0 must not reject a prime.
    plan(1, 73, 72);
    plan(1, 3, 2);
    let ring = CoefficientRing::new(8, 17).unwrap();
    for psi in [0, 1, 2, 4, 16] {
        assert_eq!(NegacyclicNtt::new(ring, psi), Err(Error::InvalidRoot));
    }
    for psi in [17, 18, u64::MAX] {
        assert_eq!(NegacyclicNtt::new(ring, psi), Err(Error::NonCanonical));
    }
    assert_eq!(NegacyclicNtt::new(CoefficientRing::new(16, 17).unwrap(), 3), Err(Error::InvalidModulus));
    assert_eq!(NegacyclicNtt::new(CoefficientRing::new(1, 17).unwrap(), 1), Err(Error::InvalidRoot));
    assert_eq!(CoefficientRing::new(1, 1_u64 << 61), Err(LayoutError::InvalidModulus));
    assert_eq!(CoefficientRing::new(1, 2), Err(LayoutError::InvalidModulus));
    assert_eq!(CoefficientRing::new(1, (1_u64 << 61) + 1), Err(LayoutError::InvalidModulus));
    assert_eq!(CoefficientRing::new(0, 17), Err(LayoutError::InvalidDegree));
    assert_eq!(CoefficientRing::new(3, 17), Err(LayoutError::InvalidDegree));
    assert_eq!(CoefficientRing::new(8192, 65537), Err(LayoutError::InvalidDegree));
}

type Unary = fn(&NegacyclicNtt, &[u64], &mut [u64]) -> Result<(), Error>;
type Binary = fn(&NegacyclicNtt, &[u64], &[u64], &mut [u64]) -> Result<(), Error>;

fn sentinels(length: usize) -> Vec<u64> {
    (0..length).map(|i| u64::MAX - 37 * i as u64).collect()
}

#[test]
fn every_operation_refuses_atomically_and_preserves_inputs() {
    let ntt = plan(8, 17, 3);
    let good = [0, 1, 2, 3, 4, 5, 15, 16];
    let good_original = good;
    let unary: [Unary; 2] = [NegacyclicNtt::forward_into, NegacyclicNtt::inverse_into];
    let binary: [Binary; 2] = [NegacyclicNtt::pointwise_add_into, NegacyclicNtt::pointwise_mul_into];
    for call in unary {
        for length in [0, 7, 9] {
            for value in [0, 17, u64::MAX] {
                // Length gate precedes residue rejection, including combined
                // invalid length + noncanonical residue (nonempty cases).
                let malformed = vec![value; length];
                let snapshot = malformed.clone();
                let mut out = sentinels(8);
                let before = out.clone();
                assert_eq!(call(&ntt, &malformed, &mut out), Err(Error::BufferLength));
                assert_eq!(out, before);
                assert_eq!(malformed, snapshot);
            }
            let mut wrong_out = sentinels(length);
            let before = wrong_out.clone();
            assert_eq!(call(&ntt, &good, &mut wrong_out), Err(Error::BufferLength));
            assert_eq!(wrong_out, before);
            let noncanonical = [17; 8];
            let snapshot = noncanonical;
            assert_eq!(call(&ntt, &noncanonical, &mut wrong_out), Err(Error::BufferLength));
            assert_eq!(wrong_out, before);
            assert_eq!(noncanonical, snapshot);
            assert_eq!(good, good_original);
        }
        for at in 0..8 {
            for bad in [17, u64::MAX] {
                let mut input = good;
                input[at] = bad;
                let original = input;
                let mut out = sentinels(8);
                let before = out.clone();
                assert_eq!(call(&ntt, &input, &mut out), Err(Error::NonCanonical));
                assert_eq!(out, before);
                assert_eq!(input, original);
            }
        }
    }
    for call in binary {
        for length in [0, 7, 9] {
            for value in [0, 17, u64::MAX] {
                let malformed = vec![value; length];
                let snapshot = malformed.clone();
                let mut out = sentinels(8);
                let before = out.clone();
                assert_eq!(call(&ntt, &malformed, &good, &mut out), Err(Error::BufferLength));
                assert_eq!(out, before);
                assert_eq!(call(&ntt, &good, &malformed, &mut out), Err(Error::BufferLength));
                assert_eq!(out, before);
                assert_eq!(malformed, snapshot);
                assert_eq!(good, good_original);
            }
            let mut wrong_out = sentinels(length);
            let before = wrong_out.clone();
            assert_eq!(call(&ntt, &good, &good, &mut wrong_out), Err(Error::BufferLength));
            assert_eq!(wrong_out, before);
            let noncanonical = [17; 8];
            let snapshot = noncanonical;
            assert_eq!(call(&ntt, &noncanonical, &good, &mut wrong_out), Err(Error::BufferLength));
            assert_eq!(wrong_out, before);
            assert_eq!(call(&ntt, &good, &noncanonical, &mut wrong_out), Err(Error::BufferLength));
            assert_eq!(wrong_out, before);
            assert_eq!(noncanonical, snapshot);
        }
        for at in 0..8 {
            for bad in [17, u64::MAX] {
                let mut input = good;
                input[at] = bad;
                let original = input;
                let mut out = sentinels(8);
                let before = out.clone();
                assert_eq!(call(&ntt, &input, &good, &mut out), Err(Error::NonCanonical));
                assert_eq!(out, before);
                assert_eq!(call(&ntt, &good, &input, &mut out), Err(Error::NonCanonical));
                assert_eq!(out, before);
                assert_eq!(input, original);
                assert_eq!(good, good_original);
            }
        }
    }
    assert_eq!(good, good_original);
}

#[test]
fn clear_phase_trace_algebra_only_no_phase_encoding_substitution() {
    let ntt = plan(8, 17, 3);
    let ring = ntt.ring();
    let a = [16, 1, 0, 3, 0, 16, 2, 4];
    let b = [1, 0, 16, 0, 4, 2, 0, 1];
    let actual = product(&ntt, &a, &b);
    let expected = convolution(&a, &b, 17);
    assert_eq!(actual, expected);
    let mut shifted = [0; 8];
    ring.monomial(&a, -1, &mut shifted).unwrap();
    let mut monomial = [0; 8];
    monomial[7] = 16; // X^-1 = -X^7 in this ring.
    assert_eq!(product(&ntt, &a, &monomial), shifted);
    // Distinct lanes: NTT forward results are the actual trace inputs. Expected
    // coefficients come only from independent BigInt natural-order evaluation
    // and the semantic lane formula, never either trace wrapper or its kernel.
    let mut lane0 = [0; 8];
    let mut lane1 = [0; 8];
    ntt.forward_into(&a, &mut lane0).unwrap();
    ntt.forward_into(&b, &mut lane1).unwrap();
    assert_ne!(lane0, lane1);
    let input = [lane0.as_slice(), lane1.as_slice()].concat();
    let forward_oracle = [direct(&a, 3, 17, false), direct(&b, 3, 17, false)].concat();
    let mut expected_trace = [0; 8];
    for k in 0..4 {
        for j in 0..2 {
            expected_trace[2 * k + j] = forward_oracle[j * 8 + 2 * k];
        }
    }
    let mut clear_trace = [0; 8];
    let mut phase_trace = [0; 8];
    let mut scratch = vec![0; ring.interleave_scratch_words(2, 2).unwrap()];
    ring.interleave(&input, 2, 2, &mut clear_trace, &mut scratch).unwrap();
    assert_eq!(clear_trace, expected_trace);
    crate::coefficient_phase_reference::trace_interleave(
        8, 17, &input, 2, 2, &mut phase_trace, &mut scratch,
    ).unwrap();
    assert_eq!(phase_trace, expected_trace);
}

#[test]
fn inventory_is_bounded_nested_events_not_tariff() {
    for (n, q, psi) in [(1, 17, 16), (8, 17, 3), (4096, 65537, 6561)] {
        let ntt = plan(n, q, psi);
        let setup = construction_operations(ntt.ring());
        assert!(setup.exponentiations <= 35);
        assert_eq!(setup.exponent_steps, 64 * setup.exponentiations);
        assert_eq!(setup.modular_multiplications, 128 * setup.exponentiations + 448);
        assert_eq!(setup.primality_rounds, 7);
        for operation in [Operation::Forward, Operation::Inverse, Operation::PointwiseAdd, Operation::PointwiseMul] {
            let events = ntt.operations(operation);
            assert!(events.residue_checks >= n as u64);
            assert!(events.residue_checks <= 2 * n as u64);
            assert!(events.moves <= 14 * n as u64);
            assert!(events.modular_multiplications <= 15 * n as u64);
            assert_eq!(events.exponentiations, 0);
            assert_eq!(events.exponent_steps, 0);
        }
    }
}

#[test]
fn expanded_inventory_recounts_thirteen_stages_without_funded_runtime_claim() {
    assert_eq!(std::mem::size_of::<[u64; 13]>() * 2 - std::mem::size_of::<[u64; 12]>() * 2, 16);
    for (n, powers, steps, products) in [(4096, 35, 2240, 4928), (8192, 37, 2368, 5184)] {
        let ntt = public_fermat_plan(n);
        let setup = construction_operations(ntt.ring());
        assert_eq!(setup, LogicalOperations {
            checks: 4096,
            residue_checks: 1,
            moves: 128,
            modular_multiplications: products,
            exponentiations: powers,
            exponent_steps: steps,
            primality_rounds: 7,
            ..LogicalOperations::default()
        });
        let words = n as u64;
        let butterflies = words / 2 * u64::from(n.trailing_zeros());
        for (operation, extra_products) in [(Operation::Forward, 2), (Operation::Inverse, 3)] {
            let events = ntt.operations(operation);
            assert_eq!(events, LogicalOperations {
                checks: words + 2,
                residue_checks: words,
                moves: 2 * words + 2 * butterflies,
                swaps: words / 2,
                modular_additions: butterflies,
                modular_subtractions: butterflies,
                modular_multiplications: 2 * butterflies + extra_products * words,
                ..LogicalOperations::default()
            });
            if n == 8192 {
                assert_eq!(butterflies, 53_248);
                assert_eq!(events.moves, 122_880);
                assert_eq!(events.swaps, 4096); // conservative; 4032 actual bit-reversal swaps
                assert_eq!(events.modular_multiplications, if operation == Operation::Forward { 122_880 } else { 131_072 });
            }
        }
        for operation in [Operation::PointwiseAdd, Operation::PointwiseMul] {
            assert_eq!(ntt.operations(operation), LogicalOperations {
                checks: 3 + 2 * words,
                residue_checks: 2 * words,
                moves: words,
                modular_additions: if operation == Operation::PointwiseAdd { words } else { 0 },
                modular_multiplications: if operation == Operation::PointwiseMul { words } else { 0 },
                ..LogicalOperations::default()
            });
        }
    }
}
