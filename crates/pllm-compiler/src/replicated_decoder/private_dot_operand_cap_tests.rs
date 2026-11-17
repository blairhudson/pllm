use super::{private_dot_operand_caps, Error};

// Independent literal native-domain specification. No root-based expected caps.
const DOMAIN: u128 = (1_u128 << 62) - 1;
const QUARTER: i128 = 1_i128 << 62;

fn raw_bound(inner: usize, cap: u128) -> Option<u128> {
    cap.checked_mul(cap)
        .and_then(|square| square.checked_mul(u128::try_from(inner).unwrap()))
}

fn safe_candidate(inner: usize, cap: u128) -> bool {
    // Overflow of this nonnegative exact product implies it exceeds DOMAIN.
    matches!(raw_bound(inner, cap), Some(bound) if bound <= DOMAIN)
}

fn requested_samples() -> [i64; 13] {
    [
        0,
        1,
        2,
        3,
        7,
        32768,
        (1 << 26) - 1,
        1 << 26,
        (1 << 26) + 1,
        (1 << 28) - 1,
        1 << 28,
        1 << 31,
        i64::MAX,
    ]
}

fn width_samples() -> Vec<usize> {
    let mut widths = vec![
        1, 2, 3, 4, 63, 64, 65, 128, 896, 1024, 1025, usize::MAX - 1, usize::MAX,
    ];
    for bit in 0..usize::BITS {
        let power = 1_u128 << bit;
        for value in [power - 1, power, power + 1] {
            if let Ok(width) = usize::try_from(value) {
                if width > 0 {
                    widths.push(width);
                }
            }
        }
    }
    for value in [
        DOMAIN - 1,
        DOMAIN,
        DOMAIN + 1,
        u128::try_from(i64::MAX).unwrap(),
    ] {
        if let Ok(width) = usize::try_from(value) {
            widths.push(width);
        }
    }
    for k in 1..=64_u128 {
        // Quotients locate test inputs only; multiplication decides safety.
        let boundary = DOMAIN / (k * k);
        for value in [boundary - 1, boundary, boundary + 1] {
            if let Ok(width) = usize::try_from(value) {
                if width > 0 {
                    widths.push(width);
                }
            }
        }
    }
    widths.sort_unstable();
    widths.dedup();
    widths
}

#[test]
fn negative_requests_and_zero_width_refuse_bounds() {
    for requested in [i64::MIN, -2, -1] {
        for inner in [0, 1, 64, usize::MAX] {
            assert_eq!(private_dot_operand_caps(requested, inner), Err(Error::Bounds));
        }
    }
    for requested in [0, 1, 1 << 26, i64::MAX] {
        assert_eq!(private_dot_operand_caps(requested, 0), Err(Error::Bounds));
    }
}

#[test]
fn zero_request_is_the_singleton_zero_box() {
    for inner in width_samples() {
        assert_eq!(private_dot_operand_caps(0, inner), Ok((0, 0)));
    }
}

#[test]
fn literal_equal_cap_endpoints_and_h64_raw_units() {
    assert_eq!(pllm_core::puma::INPUT_MIN, -(1_i64 << 62));
    assert_eq!(pllm_core::puma::INPUT_MAX_EXCLUSIVE, 1_i64 << 62);
    for (inner, expected) in [
        (1, (1 << 31) - 1),
        (4, (1 << 30) - 1),
        (64, (1 << 28) - 1),
    ] {
        assert_eq!(
            private_dot_operand_caps(i64::MAX, inner),
            Ok((expected, expected))
        );
    }
    let requested = 256 * 262144;
    assert_eq!(
        private_dot_operand_caps(requested, 64),
        Ok((requested, requested))
    );
    assert_eq!(
        raw_bound(64, u128::try_from(requested).unwrap()),
        Some(1_u128 << 58)
    );
    // Equal-cap maximality is not asymmetric optimality or input authority.
    let (q_cap, k_cap) = (DOMAIN, 1_u128);
    assert_eq!(q_cap.checked_mul(k_cap), Some(DOMAIN));
    assert_ne!(q_cap, k_cap);
}

#[test]
fn exact_positive_boundary_is_excluded_despite_legal_negative_endpoint() {
    let requested = 1_i64 << 28;
    let (cap, other) = private_dot_operand_caps(requested, 64).unwrap();
    assert_eq!((cap, other), (requested - 1, requested - 1));
    let safe = 64_i128 * i128::from(cap) * i128::from(cap);
    assert_eq!(safe, QUARTER - (1_i128 << 35) + 64);
    assert!((-QUARTER..QUARTER).contains(&safe));
    assert!((-QUARTER..QUARTER).contains(&-safe));
    let rejected = 64_i128 * i128::from(cap + 1) * i128::from(cap + 1);
    assert_eq!(rejected, QUARTER);
    assert!(!(-QUARTER..QUARTER).contains(&rejected));
    assert!((-QUARTER..QUARTER).contains(&-rejected));
    assert!(rejected.unsigned_abs() > DOMAIN);
    assert!((-rejected).unsigned_abs() > DOMAIN);
}

#[test]
fn extreme_abstract_widths_preserve_honest_zero_caps() {
    if let Ok(inner) = usize::try_from(DOMAIN) {
        for requested in [2, i64::MAX] {
            assert_eq!(private_dot_operand_caps(requested, inner), Ok((1, 1)));
        }
    }
    if let Ok(inner) = usize::try_from(DOMAIN + 1) {
        assert_eq!(private_dot_operand_caps(i64::MAX, inner), Ok((0, 0)));
    }
    if usize::BITS == 64 {
        for requested in requested_samples() {
            assert_eq!(private_dot_operand_caps(requested, usize::MAX), Ok((0, 0)));
        }
    } else if usize::BITS == 32 {
        assert_eq!(
            private_dot_operand_caps(i64::MAX, usize::MAX),
            Ok((32768, 32768))
        );
    }
}

#[test]
fn independent_inequality_and_maximality_hold_across_extrema() {
    for requested in requested_samples() {
        for inner in width_samples() {
            let (cap, other) = private_dot_operand_caps(requested, inner).unwrap();
            assert_eq!(cap, other);
            assert!(cap >= 0 && cap <= requested);
            let cap = u128::try_from(cap).unwrap();
            assert!(
                safe_candidate(inner, cap),
                "requested={requested}, inner={inner}"
            );
            if cap < u128::try_from(requested).unwrap() {
                assert!(
                    !safe_candidate(inner, cap + 1),
                    "requested={requested}, inner={inner}"
                );
            }
        }
    }
}

#[test]
fn overflowing_rejected_candidate_is_unsafe_not_an_oracle_panic() {
    let candidate = u128::try_from(i64::MAX).unwrap();
    assert_eq!(raw_bound(usize::MAX, candidate), None);
    assert!(!safe_candidate(usize::MAX, candidate));
    // A representable product above DOMAIN must also fail the same predicate.
    assert_eq!(raw_bound(64, 1_u128 << 28), Some(1_u128 << 62));
    assert!(!safe_candidate(64, 1_u128 << 28));
}

#[test]
fn finite_brute_force_selects_greatest_safe_candidate() {
    for inner in width_samples() {
        for requested in 0..=64_i64 {
            let expected = (0..=requested)
                .filter(|&candidate| safe_candidate(inner, u128::try_from(candidate).unwrap()))
                .max()
                .unwrap();
            assert_eq!(
                private_dot_operand_caps(requested, inner),
                Ok((expected, expected))
            );
        }
    }
}

#[test]
fn caps_are_monotone_in_request_and_antitone_in_positive_width() {
    let widths = width_samples();
    for &inner in &widths {
        let mut previous = 0;
        for requested in requested_samples() {
            let (cap, other) = private_dot_operand_caps(requested, inner).unwrap();
            assert_eq!(cap, other);
            assert!(cap >= previous);
            previous = cap;
        }
    }
    for requested in requested_samples() {
        let mut previous = requested;
        for &inner in &widths {
            let (cap, _) = private_dot_operand_caps(requested, inner).unwrap();
            assert!(cap <= previous);
            previous = cap;
        }
    }
}

fn assert_signed_value(value: i128) {
    assert!(value.unsigned_abs() <= DOMAIN);
    assert!((-QUARTER..QUARTER).contains(&value));
    assert!(i64::try_from(value).is_ok());
}

fn assert_signed_dot(left: &[i64], right: &[i64]) -> i128 {
    assert_eq!(left.len(), right.len());
    let mut prefix = 0_i128;
    assert_signed_value(prefix);
    for (&left, &right) in left.iter().zip(right) {
        let product = i128::from(left) * i128::from(right);
        assert_signed_value(product);
        prefix += product;
        assert_signed_value(prefix);
    }
    prefix
}

fn box_vector(mut code: usize, inner: usize, cap: i64) -> Vec<i64> {
    (0..inner)
        .map(|_| {
            let value = [-cap, 0, cap][code % 3];
            code /= 3;
            value
        })
        .collect()
}

#[test]
fn signed_products_and_every_prefix_stay_in_raw_domain() {
    // Vertices, zero and mixed signs; the conditional triangle inequality in
    // the helper's contract also covers every interior integer box point.
    for inner in [1, 2, 3] {
        let (cap, _) = private_dot_operand_caps(i64::MAX, inner).unwrap();
        let count = 3_usize.pow(u32::try_from(inner).unwrap());
        for left in 0..count {
            for right in 0..count {
                assert_signed_dot(
                    &box_vector(left, inner, cap),
                    &box_vector(right, inner, cap),
                );
            }
        }
    }
    let (cap, _) = private_dot_operand_caps(i64::MAX, 64).unwrap();
    let positive = [cap; 64];
    let endpoint = assert_signed_dot(&positive, &positive);
    assert_eq!(endpoint, QUARTER - (1_i128 << 35) + 64);
    assert_eq!(assert_signed_dot(&positive, &[-cap; 64]), -endpoint);
    let alternating: Vec<_> = (0..64)
        .map(|i| if i % 2 == 0 { cap } else { -cap })
        .collect();
    assert_eq!(assert_signed_dot(&positive, &alternating), 0);
    let delayed: Vec<_> = (0..64)
        .map(|i| if i < 32 { cap } else { -cap })
        .collect();
    assert_eq!(assert_signed_dot(&positive, &delayed), 0);
}
