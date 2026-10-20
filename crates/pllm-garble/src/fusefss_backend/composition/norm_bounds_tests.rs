use super::*;
use crate::fusefss_backend::composition::{cache, norm, source_tables};
use crate::fusefss_backend::{Budget, MAX_ALLOCATION_BYTES};

// Independent source oracle: binary-search sqrt and signed Euclidean division.
// It does not use the production table, extraction, bound, or sqrt helper.
fn root(n: u128) -> u128 {
    let (mut lo, mut hi) = (0u128, 1u128 << 30);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if mid * mid <= n {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

fn oracle(row: &[i64]) -> (u128, u128, Vec<i128>, Vec<i128>) {
    let s = 1 + row
        .iter()
        .map(|&v| i128::from(v).unsigned_abs().pow(2))
        .sum::<u128>();
    assert!((1..1u128 << 50).contains(&s));
    let e = (0..50).rev().find(|&e| (1u128 << e) <= s).unwrap();
    let j = 128 * s / (1u128 << e);
    let r = root(((1u128 << 55) * row.len() as u128) / (j * (1u128 << e)));
    let products: Vec<_> = row.iter().map(|&v| i128::from(v) * r as i128).collect();
    let normalized = products.iter().map(|v| v.div_euclid(4096)).collect();
    (s, r, products, normalized)
}

fn check_row(row: &[i64]) {
    let bound = source_bounds(row.len()).unwrap();
    let (_, _, products, normalized) = oracle(row);
    let (lo, hi) = bound.normalized_interval();
    assert!(products
        .iter()
        .all(|p| p.unsigned_abs() <= bound.raw_product_abs_max() as u128));
    assert!(
        products
            .iter()
            .map(|p| p.unsigned_abs().pow(2))
            .sum::<u128>()
            <= bound.raw_row_energy()
    );
    assert!(normalized
        .iter()
        .all(|&v| i128::from(lo) <= v && v <= i128::from(hi)));
    assert!(
        normalized
            .iter()
            .map(|p| p.unsigned_abs().pow(2))
            .sum::<u128>()
            <= bound.normalized_row_energy()
    );
}

#[test]
fn source_width_and_checked_integer_boundaries() {
    for k in [0, 4097, usize::MAX] {
        assert_eq!(source_bounds(k), Err(Error::Domain));
    }
    for (n, expected) in [
        (0, 0),
        (1, 1),
        (2, 2),
        (4, 2),
        (5, 3),
        (u128::MAX, 1u128 << 64),
    ] {
        assert_eq!(ceil_root(n).unwrap(), expected);
    }
    assert!(floor_energy(u128::MAX, 4096).is_err());
    for k in 1..=4096 {
        let b = source_bounds(k).unwrap();
        assert_eq!(b.width(), k);
        assert_eq!(b.table_kind(), cache::Kind::Rsqrt { k });
        assert!(i128::from(b.raw_product_abs_max()) < norm::HALF);
    }
    let b = source_bounds(896).unwrap();
    assert_eq!(b.raw_product_abs_max(), 504_154_642);
    assert_eq!(b.normalized_interval(), (-123_085, 123_084));
    assert_eq!(b.normalized_row_energy(), 15_157_193_792);
}

#[test]
fn every_reachable_source_bin_and_all_table_tails_obey_the_bound() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for k in [1, 4, 64, 768, 896, 4096] {
        let bound = source_bounds(k).unwrap();
        let table = source_tables::rsqrt(k, &budget).unwrap();
        assert_eq!(table.len(), 8192);
        for (index, &r) in table.iter().enumerate() {
            let j = 128 + (index >> 6) as u128;
            let e = index & 63;
            let r = r as u128;
            let n = ((1u128 << 55) * k as u128) / (j << e);
            assert!(r * r <= n && (r + 1) * (r + 1) > n);
            if e >= 50 {
                continue; // Formula-defined, unreachable entries stay intact.
            }
            let lower = (j << e).div_ceil(128);
            let upper = (((j + 1) << e).div_ceil(128) - 1).min((1u128 << 50) - 1);
            if lower > upper {
                continue;
            }
            for s in [lower, upper] {
                assert!((1u128 << e) <= s && s < (1u128 << (e + 1)));
                assert_eq!(128 * s / (1u128 << e), j);
            }
            // R is constant in the bin; energy increases with S. This checks
            // every S in the bin, including the unsigned50 top-bit region.
            assert!(
                r * r * (upper - 1) <= bound.raw_row_energy(),
                "k={k}, index={index}"
            );
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn exact_signed_rows_preserve_bin_approximation_and_both_floors() {
    let choices = [-17, -4, -2, -1, 0, 1, 2, 4, 17];
    for k in 1..=4u32 {
        for mut code in 0..choices.len().pow(k) {
            let mut row = vec![0; k as usize];
            for v in &mut row {
                *v = choices[code % choices.len()];
                code /= choices.len();
            }
            check_row(&row);
        }
    }
    for k in [64, 768, 896, 4096] {
        let limit = ((norm::HALF as u128 - 2) / k as u128).isqrt() as i64;
        for x in [-limit, -1, 0, 1, limit] {
            check_row(&vec![x; k]);
        }
        for e in 7..49 {
            let x = ((129u128 << (e - 7)) - 2).isqrt() as i64;
            for sign in [-1, 1] {
                let mut row = vec![0; k];
                row[0] = sign * x;
                check_row(&row);
            }
        }
    }
    assert_eq!(oracle(&[23]).3, [4099]); // Ideal RMS bound 4096 is false.
    assert_eq!(oracle(&[1]).3, [2896]);
    assert_eq!(oracle(&[-1]).3, [-2897]);
    let (_, _, raw, q) = oracle(&[2]);
    assert_eq!((q[0] * 8192).div_euclid(4096), 7326);
    assert_eq!((raw[0] * 8192).div_euclid(4096 * 4096), 7327);
}

#[test]
fn correlated_affine_admission_keeps_input_obligations_separate() {
    let gamma = [8192; 896];
    let beta = [0; 896];
    let inverse_max = ((1u128 << 48) * 896).isqrt() as u64;
    let spec = norm::Spec {
        kind: norm::Kind::Rms,
        input_min: 0,
        input_max: 600_000,
        gamma: &gamma,
        beta: &beta,
    };
    let old_product = 600_000i128 * inverse_max as i128;
    assert!(old_product < norm::HALF);
    assert!((old_product.div_euclid(4096) * 8192) >= norm::HALF);
    let proof = norm::preflight(&spec, inverse_max).unwrap();
    assert_eq!(proof.s.hi, 322_560_000_000_001);
    assert_eq!(proof.mean.lo, 0);
    assert_eq!(proof.centered.hi, 600_000);
    check_row(&[600_000; 896]);
    check_row(&[-600_000; 896]);
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    drop(norm::compile([5; 32], &spec, &budget).unwrap());
    assert_eq!(budget.used_bytes(), 0);

    // Same conditional output bound cannot admit an unproved S, even though
    // some sparse rows inside the rejected interval would be safe.
    let rejected = norm::Spec {
        input_max: 800_000,
        ..spec
    };
    assert!(matches!(
        norm::preflight(&rejected, inverse_max),
        Err(Error::Domain)
    ));
    assert!(matches!(
        norm::compile([5; 32], &rejected, &budget),
        Err(Error::Domain)
    ));
    let wrapped = norm::Spec {
        kind: norm::Kind::Rms,
        input_min: 1 << 25,
        input_max: 1 << 25,
        gamma: &[4096],
        beta: &[0],
    };
    assert!(source_bounds(1).is_ok());
    assert!(matches!(
        norm::compile([5; 32], &wrapped, &budget),
        Err(Error::Domain)
    ));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn legacy_norm_plan_identities_and_tighter_zero_intervals() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let cache = cache::SourceTableCache::new(
        [7; 32],
        cache::Limits {
            max_entries: 4,
            max_retained_bytes: 1024 * 1024,
        },
        &budget,
    )
    .unwrap();
    // Captured from the unchanged preflight BEFORE enabling correlated bounds.
    // The digest commits the existing contract, tables, and instruction stream.
    let cases = [
        (
            norm::Kind::Layer,
            4,
            4096,
            4096,
            4096,
            0,
            "97ce12ae139db41f6998608ce6910b9e615ce24268ea57d102aad9c9c1394e89",
        ),
        (
            norm::Kind::Layer,
            768,
            4096,
            4096,
            4096,
            0,
            "55bcbd61ee1b94e71ae2da84493ac68e795dda87ac98818251771dab8d177718",
        ),
        (
            norm::Kind::Rms,
            896,
            -4096,
            4096,
            4096,
            0,
            "36a770a37912c9afd17d2306cb614d39ac8effdbf295c020dbddf5818a21a80f",
        ),
        (
            norm::Kind::Rms,
            4,
            0,
            0,
            norm::HALF as i64 - 1,
            norm::HALF as i64 - 1,
            "a1e3ba1ba75729e4e8267984c66d5e300fae94d2cd4fc5328ea6f8b32b7c12d4",
        ),
    ];
    for (kind, width, lo, hi, g, b, expected) in cases {
        let gamma = vec![g; width];
        let beta = vec![b; width];
        let spec = norm::Spec {
            kind,
            input_min: lo,
            input_max: hi,
            gamma: &gamma,
            beta: &beta,
        };
        let plan = norm::compile([5; 32], &spec, &budget).unwrap();
        let expected: [u8; 32] =
            std::array::from_fn(|i| u8::from_str_radix(&expected[2 * i..2 * i + 2], 16).unwrap());
        assert_eq!(plan.digest(), expected);
        let cached = norm::compile_cached([5; 32], &spec, &cache).unwrap();
        assert_eq!(cached.digest(), plan.digest());
        assert_eq!(
            cached.resources().cumulative_work,
            plan.resources().cumulative_work
        );
        assert_eq!(
            cached.resources().private_material_bytes_total,
            plan.resources().private_material_bytes_total
        );
        assert_eq!(
            cached.resources().frames_total,
            plan.resources().frames_total
        );
        let identity = cache
            .identity(source_bounds(width).unwrap().table_kind(), [5; 32])
            .unwrap();
        assert_eq!(identity.numeric_revision, source_tables::source_digest());
        assert_eq!(identity.kind, cache::Kind::Rsqrt { k: width });
        assert_eq!((identity.bits, identity.entries), (50, 8192));
        cache.validate(identity).unwrap();
    }
    drop(cache);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn complete_input_energy_intersects_each_square_and_every_partial_sum() {
    let choices = [-17i64, -2, -1, 0, 1, 2, 17];
    for k in 1..=4u32 {
        let kind = cache::Kind::Rsqrt { k: k as usize };
        let scalar_square_max = (1i128 << 25).pow(2);
        let gamma = vec![8193; k as usize];
        let beta = vec![-7; k as usize];
        let spec = norm::Spec {
            kind: norm::Kind::Rms,
            input_min: -(1 << 25),
            input_max: 1 << 25,
            gamma: &gamma,
            beta: &beta,
        };
        let inverse_max = root((1u128 << 48) * u128::from(k)) as u64;
        assert!(norm::preflight(&spec, inverse_max).is_err());
        for mut code in 0..choices.len().pow(k) {
            let mut row = vec![0; k as usize];
            for x in &mut row {
                *x = choices[code % choices.len()];
                code /= choices.len();
            }
            let energy = row
                .iter()
                .map(|&x| i128::from(x).unsigned_abs().pow(2))
                .sum::<u128>();
            let (square, sum) =
                rms_input_energy_bounds(kind, 0, scalar_square_max, energy).unwrap();
            let proof = norm::preflight_inner(&spec, inverse_max, Some(energy)).unwrap();
            let mut partial = 0i128;
            for &x in &row {
                let exact = i128::from(x).pow(2);
                assert!((square.lo..=square.hi).contains(&exact));
                partial += exact;
                assert!((0..=sum.hi).contains(&partial));
            }
            assert!(sum.hi <= scalar_square_max * i128::from(k));
            assert!(sum.hi <= energy as i128);
            assert!((sum.lo..=sum.hi).contains(&partial));
            assert!((proof.s.lo..=proof.s.hi).contains(&(partial + 1)));
            let (_, _, raw, q) = oracle(&row);
            let bound = source_bounds(k as usize).unwrap();
            for (&p, &v) in raw.iter().zip(&q) {
                assert!(p.unsigned_abs() <= bound.raw_product_abs_max() as u128);
                let affine = v * 8193;
                let y = affine.div_euclid(4096) - 7;
                assert!((-norm::HALF..norm::HALF).contains(&affine));
                assert!((-norm::HALF..norm::HALF).contains(&y));
            }
        }
    }
    // A loose E never widens the original per-coordinate or row bounds.
    let (square, sum) =
        rms_input_energy_bounds(cache::Kind::Rsqrt { k: 4 }, 4, 9, 1_000_000).unwrap();
    assert_eq!(square, norm::Interval { lo: 4, hi: 9 });
    assert_eq!(sum, norm::Interval { lo: 16, hi: 36 });
}

#[test]
fn input_energy_rejects_wrong_formula_contradiction_and_checked_overflow() {
    for kind in [
        cache::Kind::Silu,
        cache::Kind::Exp0,
        cache::Kind::Exp1,
        cache::Kind::Reciprocal { k: 4 },
        cache::Kind::Rsqrt { k: 0 },
        cache::Kind::Rsqrt { k: 4097 },
        cache::Kind::Rsqrt { k: usize::MAX },
    ] {
        assert!(rms_input_energy_bounds(kind, 0, 1, 4).is_err());
    }
    let kind = cache::Kind::Rsqrt { k: 4 };
    for (lo, hi, energy) in [
        (-1, 1, 4),
        (2, 1, 4),
        (4, 9, 15), // Four coordinates each contribute at least four.
        (0, 1, u128::MAX),
        (0, i128::MAX, 1), // Legacy scalar sum arithmetic must remain checked.
        (i128::MAX, i128::MAX, 1),
        (norm::HALF, norm::HALF, (norm::HALF * 4) as u128),
    ] {
        assert!(rms_input_energy_bounds(kind, lo, hi, energy).is_err());
    }
}
