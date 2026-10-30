//! Independent BigInt CRT/rounded-quotient oracles. Source-only, UNRUN.
use super::*;
use crate::coefficient_layout::DegreeBound;
use num_bigint::BigInt;

fn positive_mod(value: BigInt, modulus: &BigInt) -> BigInt {
    ((value % modulus) + modulus) % modulus
}

fn word(value: BigInt) -> u64 {
    let (_, digits) = value.to_u64_digits();
    assert!(digits.len() <= 1);
    digits.first().copied().unwrap_or(0)
}

// Independent global CRT, then integer rounded division. No production inverse,
// modular arithmetic, centered remainder or Wide/Rns3 reconstruction is used.
fn oracle(aq: u64, ap: u64, q: u64, p: u64) -> u64 {
    let qb = BigInt::from(q);
    let pb = BigInt::from(p);
    let digit = positive_mod(
        (BigInt::from(ap) - BigInt::from(aq)) * qb.modinv(&pb).unwrap(),
        &pb,
    );
    let x = BigInt::from(aq) + &qb * digit;
    assert!(x >= BigInt::from(0) && x < &qb * &pb);
    word(((x + BigInt::from((p - 1) / 2)) / &pb) % &qb)
}

fn plan(degree: usize, q: u64, p: u64) -> RoundedLimbDrop {
    RoundedLimbDrop::new(
        CoefficientRing::new(degree, q).unwrap(),
        CoefficientRing::new(degree, p).unwrap(),
    ).unwrap()
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[test]
fn exhaustive_tiny_odd_coprime_rings_include_composites() {
    for q in (3..=25).step_by(2) {
        for p in (3..=25).step_by(2) {
            if gcd(q, p) != 1 {
                continue;
            }
            let drop = plan(1, q, p);
            for x in 0..q * p {
                let mut out = [u64::MAX];
                drop.apply_into(&[x % q], &[x % p], &mut out).unwrap();
                assert_eq!(out[0], oracle(x % q, x % p, q, p), "q={q}, p={p}, x={x}");
                assert_eq!(out[0], ((x + (p - 1) / 2) / p) % q);
            }
        }
    }
}

#[test]
fn small_retained_modulus_center_midpoint_and_wrap() {
    let drop = plan(4, 3, 7);
    let x = [3, 4, 20, 0];
    let mut out = [u64::MAX; 4];
    drop.apply_into(&x.map(|v| v % 3), &x.map(|v| v % 7), &mut out).unwrap();
    assert_eq!(out, [0, 1, 0, 0]);
}

#[test]
fn limb_boundaries_both_modulus_orders_and_negative_centers() {
    let high = (1_u64 << 61) - 1;
    for (q, p) in [(high, high - 2), (high - 2, high), (3, high), (high, 3), (9, 25), (25, 9)] {
        let drop = plan(1, q, p);
        for aq in [0, 1, q / 2, q - 1] {
            for ap in [0, 1, (p - 1) / 2, (p - 1) / 2 + 1, p - 1] {
                let mut out = [u64::MAX];
                drop.apply_into(&[aq], &[ap], &mut out).unwrap();
                assert_eq!(out[0], oracle(aq, ap, q, p), "q={q}, p={p}, aq={aq}, ap={ap}");
            }
        }
    }
}

#[test]
fn constructor_and_ring_domain_errors_keep_existing_bounds() {
    for degree in [0, 3, 16384] {
        assert_eq!(CoefficientRing::new(degree, 9), Err(LayoutError::InvalidDegree));
    }
    for modulus in [0, 1, 2, 4, 1_u64 << 61, (1_u64 << 61) + 1, u64::MAX] {
        assert_eq!(CoefficientRing::new(1, modulus), Err(LayoutError::InvalidModulus));
    }
    assert_eq!(
        RoundedLimbDrop::new(CoefficientRing::new(1, 9).unwrap(), CoefficientRing::new(2, 25).unwrap()),
        Err(LayoutError::InvalidDegree),
    );
    for (q, p) in [(9, 9), (9, 15), (15, 9)] {
        assert_eq!(
            RoundedLimbDrop::new(CoefficientRing::new(1, q).unwrap(), CoefficientRing::new(1, p).unwrap()),
            Err(LayoutError::InvalidModulus),
        );
    }
    assert!(CoefficientRing::new(4096, 9).is_ok());
    assert_eq!(CoefficientRing::new(8192, 9), Err(LayoutError::InvalidDegree));
    assert_eq!(
        CoefficientRing::bounded(16384, 9, DegreeBound::Public8192),
        Err(LayoutError::InvalidDegree),
    );
}

#[test]
fn explicit_public_8192_uses_only_coefficientwise_linear_work() {
    let q = CoefficientRing::bounded(8192, 9, DegreeBound::Public8192).unwrap();
    let p = CoefficientRing::bounded(8192, 25, DegreeBound::Public8192).unwrap();
    let drop = RoundedLimbDrop::new(q, p).unwrap();
    let mut vq = vec![0; 8192];
    let mut vp = vec![0; 8192];
    let mut expected = vec![0; 8192];
    // Sparse boundary/basis values; no transform or dense quadratic oracle.
    for (at, x) in [(0, 12_u64), (1, 13), (4096, 25), (8191, 224)] {
        vq[at] = x % 9;
        vp[at] = x % 25;
        expected[at] = oracle(vq[at], vp[at], 9, 25);
    }
    let mut out = vec![u64::MAX; 8192];
    drop.apply_into(&vq, &vp, &mut out).unwrap();
    assert_eq!(out, expected);
    // Larger public descriptors do not relax these legacy consumers.
    assert_eq!(q.inverse_power_two(1), Err(LayoutError::InvalidDegree));
    assert_eq!(q.monomial(&vq, 0, &mut out), Err(LayoutError::InvalidDegree));
    assert_eq!(q.interleave_scratch_words(1, 1), Err(LayoutError::InvalidDegree));
}

#[test]
fn every_length_and_late_residue_error_preserves_all_output_words() {
    let drop = plan(4, 9, 25);
    for qlen in [0, 3, 4, 5] {
        for plen in [0, 3, 4, 5] {
            for olen in [0, 3, 4, 5] {
                if (qlen, plen, olen) == (4, 4, 4) {
                    continue;
                }
                let mut out = vec![u64::MAX; olen];
                assert_eq!(drop.apply_into(&vec![0; qlen], &vec![0; plen], &mut out), Err(LayoutError::BufferLength));
                assert_eq!(out, vec![u64::MAX; olen]);
            }
        }
    }
    for (vq, vp) in [([1, 2, 3, 9], [1, 2, 3, 4]), ([1, 2, 3, 4], [1, 2, 3, 25]), ([1, 2, 3, u64::MAX], [1, 2, 3, u64::MAX])] {
        let mut out = [u64::MAX; 4];
        assert_eq!(drop.apply_into(&vq, &vp, &mut out), Err(LayoutError::NonCanonical));
        assert_eq!(out, [u64::MAX; 4]);
    }
}

#[test]
fn extracted_inverse_preserves_legacy_rns3_errors_and_crt_down_cases() {
    use crate::coefficient_phase_reference::{Error, LIMBS, Rns3, Wide};
    assert_eq!(Rns3::new([LIMBS[0], LIMBS[0], LIMBS[2]]), Err(Error::Unsupported));
    assert_eq!(Rns3::new([9, 15, 17]), Err(Error::Capacity));
    assert_eq!(Rns3::new([2, LIMBS[1], LIMBS[2]]), Err(Error::Unsupported));
    let rns = Rns3::new(LIMBS).unwrap();
    for x in [Wide::from(0), Wide::from(1), rns.modulus().half(), rns.modulus().checked_sub(Wide::from(1)).unwrap()] {
        let residues = LIMBS.map(|p| x.rem_word(p));
        assert_eq!(rns.reconstruct(residues).unwrap(), x);
    }
    for m in [0, 1, 2, 1 << 63, u64::MAX - 1, u64::MAX] {
        assert_eq!(rns.down(rns.encode(m).unwrap()).unwrap(), m);
    }
    assert_eq!(rns.down(rns.modulus().checked_sub(Wide::from(1)).unwrap()).unwrap(), 0);
    assert_eq!(rns.down(rns.modulus()), Err(Error::NonCanonical));
    assert_eq!(rns.reconstruct([LIMBS[0], 0, 0]), Err(Error::NonCanonical));
}
