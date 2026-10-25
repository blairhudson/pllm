use super::*;
use crate::coefficient_layout::CoefficientRing;

#[test]
fn actual_generated_boundary_coefficients_join_small_ring_selection() {
    let mut table = storage();
    let q = resources(&table, 0, 32).unwrap();
    let mut workspace = Workspace::prepare(&mut table, 0, 32, q.aggregate_bytes, q.work).unwrap();
    let ring = CoefficientRing::new(8, 65537).unwrap();
    let modulus = 65537i128;
    for segment in 0..4 {
        let indices = [0usize, 1, 127, 128, 1023, 1024, 2046, 2047];
        let coefficients: Vec<_> = indices
            .iter()
            .map(|&i| workspace.pairs().unwrap()[segment * 2048 + i])
            .collect();
        for (lane, &local) in indices.iter().enumerate() {
            let x = (GRID << segment) + (local as i64) * (1 << (9 + segment));
            let result = evaluate_reference(x, &mut workspace).unwrap();
            assert_eq!((result.segment, result.local_index), (segment, local));
            // Analysis reduction only: a_int*X+b_int is the EXACT quantized
            // affine numerator at denominator2^44, not a small-ring approximation.
            let polynomial: Vec<_> = coefficients
                .iter()
                .map(|c| {
                    (i128::from(c.slope) * i128::from(x) + i128::from(c.intercept))
                        .rem_euclid(modulus) as u64
                })
                .collect();
            let mut shifted = [0u64; 8];
            ring.monomial(&polynomial, -(lane as i64), &mut shifted)
                .unwrap();
            assert_eq!(i128::from(shifted[0]), result.numerator.rem_euclid(modulus));
            let c = coefficients[lane];
            let a = i128::from(c.slope).rem_euclid(modulus);
            let b = i128::from(c.intercept).rem_euclid(modulus);
            assert_eq!(
                i128::from(shifted[0]),
                (a * i128::from(x) + b).rem_euclid(modulus)
            );
        }
    }
}

fn storage() -> Vec<Coefficient> {
    vec![Coefficient::default(); PAIRS]
}

// Independent signed rational nearest-even, using Euclidean floor (including
// negative half ties), not production's positive-magnitude RNE.
fn signed_round(n: i128, d: i128) -> i128 {
    let q = n.div_euclid(d);
    let r = n.rem_euclid(d);
    q + i128::from(r > d - r || (r == d - r && q.rem_euclid(2) == 1))
}

#[test]
fn all_pairs_match_direct_rational_secants_and_ties_even() {
    let mut table = storage();
    let q = resources(&table, 0, 0).unwrap();
    let w = Workspace::prepare(&mut table, 0, 0, q.aggregate_bytes, q.work).unwrap();
    for s in 0..4 {
        for i in 0..2048 {
            // Endpoints represented over2048; compute original a=-1/(l*u),
            // b=1/l+1/u before independently rounding each scale.
            let l = (2048 + i) as i128 * (1 << s);
            let u = (2049 + i) as i128 * (1 << s);
            let den = l * u;
            let c = w.pairs().unwrap()[s * 2048 + i];
            assert_eq!(
                i128::from(c.slope),
                signed_round(-(1i128 << 24) * 2048 * 2048, den)
            );
            assert_eq!(
                i128::from(c.intercept),
                signed_round((1i128 << 44) * 2048 * (l + u), den)
            );
            // Original rational secant evaluated at either endpoint equals reciprocal.
            assert_eq!(-l + (l + u), u);
            assert_eq!(-u + (l + u), l);
        }
    }
    for (n, expected) in [(1, 0), (3, 2), (5, 2), (7, 4)] {
        assert_eq!(rne(n, 2).unwrap() as i128, expected);
        assert_eq!(signed_round(-(n as i128), 2), -expected);
    }
}

#[test]
fn every_local_and_segment_boundary_and_neighbors_exact_affine() {
    let mut table = storage();
    let q = resources(&table, 0, MAX_CASES).unwrap();
    let mut w = Workspace::prepare(&mut table, 0, MAX_CASES, q.aggregate_bytes, q.work).unwrap();
    for s in 0..4 {
        for i in 0..2048 {
            let at = (GRID << s) + (i as i64) * (1 << (9 + s));
            for x in [at - 1, at, at + 1] {
                if x < GRID {
                    continue;
                }
                let e = evaluate_reference(x, &mut w).unwrap();
                let segment = if x < 2 * GRID {
                    0
                } else if x < 4 * GRID {
                    1
                } else if x < 8 * GRID {
                    2
                } else {
                    3
                };
                let width = 512i64 * (1 << segment);
                let local = ((x - GRID * (1 << segment)) / width) as usize;
                assert_eq!((e.segment, e.local_index), (segment, local));
                let l = (2048 + local) as i128 * (1 << segment);
                let u = (2049 + local) as i128 * (1 << segment);
                let a = signed_round(-(1i128 << 46), l * u);
                let b = signed_round((1i128 << 55) * (l + u), l * u);
                assert_eq!(e.numerator, a * i128::from(x) + b);
            }
        }
    }
    let last = evaluate_reference(16 * GRID - 1, &mut w).unwrap();
    assert_eq!((last.segment, last.local_index), (3, 2047));
}

#[test]
fn invalid_grid_nonfinite_and_unrepresentable_inputs_refuse() {
    for x in [-1, 0, GRID - 1, 16 * GRID, i64::MAX] {
        assert!(index(x).is_err());
    }
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0, 16.0] {
        assert!(float_grid_input(x).is_err());
    }
    assert_eq!(float_grid_input(1.0 + 2f64.powi(-21)), Err(Error::OffGrid));
    assert_eq!(float_grid_input(1.0), Ok(GRID));
    assert_eq!(grid_input(3, 2), Ok(3 * GRID / 2));
    assert_eq!(grid_input(4, 3), Err(Error::OffGrid));
    assert_eq!(grid_input(i128::MAX, 1), Err(Error::Overflow));
    assert_eq!(grid_input(1, 0), Err(Error::Domain));
    assert_eq!(grid_input(-1, 1), Err(Error::Domain));
    assert_eq!(rne(u128::MAX, u128::MAX), Ok(1));
    assert_eq!(rne(1, 0), Err(Error::Domain));
}

#[test]
fn exact_short_resources_no_write_poison_and_retirement() {
    let sentinel = Coefficient {
        slope: 17,
        intercept: 19,
    };
    let mut table = vec![sentinel; PAIRS];
    let q = resources(&table, 123, 1).unwrap();
    for (bytes, work) in [
        (q.aggregate_bytes - 1, q.work),
        (q.aggregate_bytes, q.work - 1),
        (0, 0),
    ] {
        assert!(Workspace::prepare(&mut table, 123, 1, bytes, work).is_err());
        assert!(table.iter().all(|p| *p == sentinel));
    }
    assert_eq!(resources(&table, usize::MAX, 1), Err(Error::Overflow));
    assert_eq!(resources(&table, MAX_OWNER_BYTES, 1), Err(Error::Resource));
    let mut short = vec![sentinel; PAIRS - 1];
    assert!(Workspace::prepare(&mut short, 0, 1, usize::MAX, u128::MAX).is_err());
    let mut excess = Vec::with_capacity(PAIRS + 1);
    excess.resize(PAIRS, sentinel);
    assert!(Workspace::prepare(&mut excess, 0, 1, usize::MAX, u128::MAX).is_err());
    {
        let mut w = Workspace::prepare(&mut table, 123, 1, q.aggregate_bytes, q.work).unwrap();
        assert_eq!(evaluate_reference(16 * GRID, &mut w), Err(Error::Domain));
        assert_eq!(evaluate_reference(GRID, &mut w), Err(Error::Poisoned));
        assert_eq!(w.pairs(), Err(Error::Poisoned));
    }
    assert!(table.iter().all(|p| *p == Coefficient::default()));
    {
        let mut w = Workspace::prepare(&mut table, 123, 1, q.aggregate_bytes, q.work).unwrap();
        evaluate_reference(GRID, &mut w).unwrap();
        assert_eq!(evaluate_reference(GRID, &mut w), Err(Error::Resource));
        assert_eq!(evaluate_reference(GRID, &mut w), Err(Error::Poisoned));
    }
}

#[test]
fn independent_small_odd_ring_encrypted_accumulator_selection_equation_only() {
    let ring = CoefficientRing::new(8, 65537).unwrap();
    let inverse = ring.inverse_power_two(8).unwrap();
    assert_eq!(inverse * 8 % 65537, 1);
    // Small clear odd-ring control, NOT q0/source primes or ciphertexts. Each
    // coefficient still includes x: P_i=a_i*x+b_i, never public-LUT substitution.
    let a = [-3i64, 2, -7, 11, 5, -1, 0, 9];
    let b = [19i64, 17, 13, 23, 31, 41, 43, 47];
    for x in [1i64, 3, 15] {
        let p: Vec<u64> = (0..8)
            .map(|i| (a[i] * x + b[i]).rem_euclid(65537) as u64)
            .collect();
        for j in 0..8 {
            let mut shifted = [0u64; 8];
            ring.monomial(&p, -(j as i64), &mut shifted).unwrap();
            assert_eq!(shifted[0], (a[j] * x + b[j]).rem_euclid(65537) as u64);
            for (k, &actual) in shifted.iter().enumerate() {
                let source = (k + j) % 8;
                let expected = if k + j >= 8 {
                    (65537 - p[source]) % 65537
                } else {
                    p[source]
                };
                assert_eq!(actual, expected);
            }
        }
    }
    assert!(CoefficientRing::new(8, 4096).is_err());
}
