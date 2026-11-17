use super::{public_dot_operand_cap, Error};

// Independent exact ring64 quarter-gap specification, not the implementation's
// source constant or quotient. All oracle arithmetic precedes ring conversion.
const QUARTER: i128 = 1_i128 << 62;
const DOMAIN: u128 = (1_u128 << 62) - 1;

#[test]
fn negative_requests_reject_even_for_zero_rows() {
    for requested in [i64::MIN, -2, -1] {
        for row_l1 in [0, 1, u64::MAX] {
            assert_eq!(public_dot_operand_cap(requested, row_l1), Err(Error::Bounds));
        }
    }
}

#[test]
fn zero_requests_and_zero_rows_preserve_exact_endpoints() {
    for row_l1 in [0, 1, u64::MAX] {
        assert_eq!(public_dot_operand_cap(0, row_l1), Ok(0));
    }
    // A zero-row premise says nothing about authenticated weights or later bias.
    for requested in [0, 1, 1 << 26, i64::MAX] {
        assert_eq!(public_dot_operand_cap(requested, 0), Ok(requested));
    }
}

#[test]
fn synthetic_boundary_excludes_the_positive_quarter_gap_endpoint() {
    // Synthetic scalar obstruction only, not measured Qwen row statistics.
    let requested = 1_i64 << 26;
    let row_l1 = 1_u64 << 36;
    let cap = public_dot_operand_cap(requested, row_l1).unwrap();
    assert_eq!(cap, (1 << 26) - 1);
    let safe = i128::from(cap) * i128::from(row_l1);
    assert_eq!(safe, QUARTER - (1_i128 << 36));
    assert!((-QUARTER..QUARTER).contains(&safe));
    assert!((-QUARTER..QUARTER).contains(&-safe));

    // A single positive coefficient realizes the C+1 counterexample exactly.
    let unsafe_product = i128::from(cap + 1) * i128::from(row_l1);
    assert_eq!(cap + 1, requested);
    assert_eq!(unsafe_product, QUARTER);
    assert!(!(-QUARTER..QUARTER).contains(&unsafe_product));
    // Native domain is asymmetric; the returned operand domain is not.
    assert!((-QUARTER..QUARTER).contains(&-unsafe_product));
}

#[test]
fn extreme_rows_allow_zero_and_use_the_symmetric_native_domain() {
    let domain = i64::try_from(DOMAIN).unwrap();
    assert_eq!(public_dot_operand_cap(i64::MAX, 1), Ok(domain));
    assert_eq!(public_dot_operand_cap(2, DOMAIN as u64), Ok(1));
    assert_eq!(public_dot_operand_cap(i64::MAX, DOMAIN as u64), Ok(1));
    assert_eq!(public_dot_operand_cap(i64::MAX, (DOMAIN + 1) as u64), Ok(0));
    assert_eq!(public_dot_operand_cap(i64::MAX, u64::MAX), Ok(0));

    let cap = i128::from(public_dot_operand_cap(i64::MAX, 1).unwrap());
    assert_eq!(cap, QUARTER - 1);
    assert_eq!(-cap, -QUARTER + 1);
    assert!((-QUARTER..QUARTER).contains(&cap));
    assert!((-QUARTER..QUARTER).contains(&-cap));
    assert!(!(-QUARTER..QUARTER).contains(&(cap + 1)));
    assert!((-QUARTER..QUARTER).contains(&(-cap - 1)));
    assert_eq!(pllm_core::puma::INPUT_MIN, -(1_i64 << 62));
    assert_eq!(pllm_core::puma::INPUT_MAX_EXCLUSIVE, 1_i64 << 62);
}

fn requested_samples() -> [i64; 12] {
    [
        0,
        1,
        2,
        3,
        7,
        (1 << 26) - 1,
        1 << 26,
        (1 << 26) + 1,
        (1 << 62) - 2,
        (1 << 62) - 1,
        1 << 62,
        i64::MAX,
    ]
}

fn row_l1_samples() -> Vec<u64> {
    let mut rows = vec![0, 1, 2, 3, 7, 1 << 36, u64::MAX];
    for k in 1..=64_u128 {
        // Quotients locate interesting test inputs only. Expected caps below
        // come from multiplication inequalities or exhaustive candidate search.
        let boundary = u64::try_from(DOMAIN / k).unwrap();
        rows.extend([boundary - 1, boundary, boundary + 1]);
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

#[test]
fn independent_inequality_and_maximality_hold_across_extrema() {
    for requested in requested_samples() {
        for row_l1 in row_l1_samples() {
            let cap = public_dot_operand_cap(requested, row_l1).unwrap();
            assert!(cap >= 0);
            assert!(cap <= requested);
            let cap = u128::try_from(cap).unwrap();
            let row_l1 = u128::from(row_l1);
            assert!(cap * row_l1 <= DOMAIN);
            if cap < requested as u128 {
                assert!(row_l1 > 0);
                assert!((cap + 1) * row_l1 > DOMAIN);
            }
        }
    }
}

#[test]
fn finite_enumeration_selects_largest_satisfying_candidate() {
    for requested in 0..=64_i64 {
        for row_l1 in row_l1_samples() {
            let expected = (0..=requested)
                .rev()
                .find(|&candidate| candidate as u128 * u128::from(row_l1) <= DOMAIN)
                .unwrap();
            assert_eq!(public_dot_operand_cap(requested, row_l1), Ok(expected));
        }
    }
}

#[test]
fn caps_are_monotone_in_requested_and_antitone_in_row_l1() {
    let rows = row_l1_samples();
    for &row_l1 in &rows {
        let mut previous = 0;
        for requested in requested_samples() {
            let cap = public_dot_operand_cap(requested, row_l1).unwrap();
            assert!(cap >= previous);
            previous = cap;
        }
    }
    for requested in requested_samples() {
        let mut previous = requested;
        for &row_l1 in &rows {
            let cap = public_dot_operand_cap(requested, row_l1).unwrap();
            assert!(cap <= previous);
            previous = cap;
        }
    }
}

fn assert_signed_products_and_prefixes(coefficients: [i64; 3], requested: i64) {
    let row_l1 = coefficients.iter().map(|x| x.unsigned_abs()).sum();
    let cap = public_dot_operand_cap(requested, row_l1).unwrap();
    // Every vertex/zero choice of this small symmetric input box, independently
    // accumulated in i128. Mixed signs include cancellation to a small final dot.
    for x0 in [-cap, 0, cap] {
        for x1 in [-cap, 0, cap] {
            for x2 in [-cap, 0, cap] {
                let mut prefix = 0_i128;
                for (x, coefficient) in [x0, x1, x2].into_iter().zip(coefficients) {
                    let product = i128::from(x) * i128::from(coefficient);
                    assert!((-QUARTER..QUARTER).contains(&product));
                    assert!(product.unsigned_abs() <= DOMAIN);
                    prefix += product;
                    assert!((-QUARTER..QUARTER).contains(&prefix));
                    assert!(prefix.unsigned_abs() <= DOMAIN);
                    assert!(i64::try_from(prefix).is_ok());
                }
            }
        }
    }
}

#[test]
fn signed_products_and_every_prefix_remain_inside_raw_domain() {
    for a in -2..=2_i64 {
        for b in -2..=2_i64 {
            for c in -2..=2_i64 {
                assert_signed_products_and_prefixes([a, b, c], i64::MAX);
            }
        }
    }
    for coefficients in [
        [1 << 35, -(1 << 35), 0],
        [-(1 << 35), 0, 1 << 35],
        [1 << 34, -(1 << 35), 1 << 34],
    ] {
        assert_signed_products_and_prefixes(coefficients, 1 << 26);
    }
}
