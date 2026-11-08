use super::{reciprocal_bound, reciprocal_ceil, Error, ReciprocalBoundSpec, ReciprocalStepError};

const ZERO: ReciprocalStepError = ReciprocalStepError {
    delta_d: 0,
    delta_n: 0,
};

fn spec(fraction_bits: u8, iterations: u8) -> ReciprocalBoundSpec {
    ReciprocalBoundSpec {
        fraction_bits,
        iterations,
        xmin: 1,
        xmax: 1,
        e0: 0,
        w0: 0,
        steps: [ZERO; 8],
    }
}

#[test]
fn reference_endpoints_and_zero_one_eight_iterations() {
    for f in [0, 127] {
        for m in [0, 1, 8] {
            let input = spec(f, m);
            let bound = reciprocal_bound(&input).unwrap();
            assert_eq!(bound.code(), 0);
            assert_eq!(bound.fraction_bits(), f);
        }
    }
    let mut input = spec(0, 8);
    input.e0 = 1;
    input.w0 = 1;
    // E stays 1, W doubles eight times, B=1+256.
    assert_eq!(reciprocal_bound(&input).unwrap().code(), 257);
    input.iterations = 1;
    assert_eq!(reciprocal_bound(&input).unwrap().code(), 3);
    input.iterations = 0;
    assert_eq!(reciprocal_bound(&input).unwrap().code(), 2);
    input = spec(127, 0);
    input.e0 = 1;
    assert_eq!(reciprocal_bound(&input).unwrap().code(), 1u128 << 127);
}

#[test]
fn invalid_domains_and_every_inactive_field_are_rejected() {
    let mut invalid = spec(0, 0);
    invalid.fraction_bits = 128;
    assert_eq!(reciprocal_bound(&invalid), Err(Error::Bounds));
    invalid = spec(0, 9);
    assert_eq!(reciprocal_bound(&invalid), Err(Error::Bounds));
    invalid = spec(0, 0);
    invalid.xmin = 0;
    assert_eq!(reciprocal_bound(&invalid), Err(Error::Bounds));
    invalid.xmin = 2;
    assert_eq!(reciprocal_bound(&invalid), Err(Error::Bounds));
    for m in 0..=7 {
        for index in usize::from(m)..8 {
            for field in 0..2 {
                let mut input = spec(0, m);
                if field == 0 {
                    input.steps[index].delta_d = 1;
                } else {
                    input.steps[index].delta_n = 1;
                }
                assert_eq!(reciprocal_bound(&input), Err(Error::Bounds));
            }
        }
    }
}

#[test]
fn ceiling_uses_quotient_and_remainder_without_additive_overflow() {
    for (n, d, expected) in [
        (0, 7, 0),
        (12, 4, 3),
        (13, 4, 4),
        (u128::MAX, 1, u128::MAX),
        (u128::MAX, 2, 1u128 << 127),
        (u128::MAX, u128::MAX, 1),
        (u128::MAX - 1, u128::MAX, 1),
    ] {
        assert_eq!(reciprocal_ceil(n, d), Ok(expected));
    }
}

#[test]
fn reachable_arithmetic_overflows_return_only_error_and_preserve_input() {
    // Every witness has valid positive xmin/xmax and zero inactive records.
    let mut cases = [spec(0, 1); 8];
    cases[0].e0 = 1u128 << 64; // E squared.
    cases[1] = spec(1, 1);
    cases[1].w0 = u128::MAX; // (S+old E)*old W, reduced value would fit.
    cases[2].xmax = u128::MAX;
    cases[2].steps[0].delta_n = 2; // xmax*delta_n.
    cases[3].e0 = 1;
    cases[3].steps[0].delta_d = u128::MAX; // next E addition.
    cases[4].w0 = 1;
    cases[4].steps[0].delta_d = u128::MAX; // first W addition; next E fits.
    cases[5].w0 = 1;
    cases[5].xmax = u128::MAX;
    cases[5].steps[0].delta_n = 1; // second W addition.
    cases[6] = spec(0, 0);
    cases[6].e0 = u128::MAX;
    cases[6].w0 = 1; // final E+W.
    cases[7] = spec(1, 0);
    cases[7].e0 = 1u128 << 127; // final (E+W)*S.
    for input in cases {
        let original = input;
        assert_eq!(reciprocal_bound(&input), Err(Error::Bounds));
        assert_eq!(input, original);
        assert_eq!(reciprocal_bound(&input), Err(Error::Bounds));
    }
    // One complete local step precedes this second-step square failure.
    let mut late = spec(0, 2);
    late.steps[0].delta_d = 1u128 << 64;
    let original = late;
    assert_eq!(reciprocal_bound(&late), Err(Error::Bounds));
    assert_eq!(late, original);
    // Not falsely claimed as isolated failures: S+E is unreachable after E^2
    // succeeds (E<=2^64-1, S<=2^127); ceiling increment cannot overflow for d>0;
    // checked shift cannot fail after f<=127. Proofs accompany production code.
}

// Independent small checked-i128 oracle: never calls production ceiling helper.
fn add(a: i128, b: i128) -> i128 {
    a.checked_add(b).unwrap()
}

fn sub(a: i128, b: i128) -> i128 {
    a.checked_sub(b).unwrap()
}

fn mul(a: i128, b: i128) -> i128 {
    a.checked_mul(b).unwrap()
}

fn abs(a: i128) -> i128 {
    a.checked_abs().unwrap()
}

fn ceil(n: i128, d: i128) -> i128 {
    assert!(n >= 0 && d > 0);
    // Safe in this tiny oracle, checked independently; unlike production q/r.
    add(n, sub(d, 1)).checked_div(d).unwrap()
}

#[test]
fn independent_signed_old_state_oracle_covers_2268_cases() {
    let s = 4i128;
    let s2 = mul(s, s);
    let s3 = mul(s2, s);
    let mut cases = 0;
    for x in 1..=4i128 {
        for n in 0..=8i128 {
            for d in 0..=6i128 {
                for ed in -1..=1i128 {
                    for en in -1..=1i128 {
                        let e0 = abs(sub(s, d));
                        let w0 = ceil(abs(sub(mul(d, s), mul(x, n))), s);
                        let e1 = add(ceil(mul(e0, e0), s), 1);
                        let w1 = add(add(ceil(mul(add(s, e0), w0), s), 1), ceil(x, s));
                        let b = ceil(mul(add(e1, w1), s), x);
                        let mut input = spec(2, 1);
                        input.xmin = x as u128;
                        input.xmax = x as u128;
                        input.e0 = e0 as u128;
                        input.w0 = w0 as u128;
                        input.steps[0] = ReciprocalStepError {
                            delta_d: 1,
                            delta_n: 1,
                        };
                        let bound = reciprocal_bound(&input).unwrap();
                        assert_eq!(bound.code(), b as u128);
                        assert_eq!(bound.fraction_bits(), 2);
                        // BOTH updates use old D, never updated d1.
                        let factor = sub(mul(2, s), d);
                        let d1_num = add(mul(d, factor), mul(ed, s));
                        let n1_num = add(mul(n, factor), mul(en, s));
                        assert!(abs(sub(s2, d1_num)) <= mul(e1, s));
                        assert!(abs(sub(mul(d1_num, s), mul(x, n1_num))) <= mul(w1, s2));
                        assert!(abs(sub(mul(n1_num, x), s3)) <= mul(mul(b, s), x));
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 2268);
}

#[test]
fn lying_seed_premises_are_not_detectable_by_this_api() {
    let input = spec(0, 0); // Claims E0=W0=0, x=1, but supplies no actual state.
    assert_eq!(reciprocal_bound(&input).unwrap().code(), 0);
    // False E premise: d0=0,n0=0 gives W=0 but |1-d0|=1, actual error=1.
    let (x, d0, n0) = (1i128, 0i128, 0i128);
    assert_eq!(abs(sub(d0, mul(x, n0))), 0);
    assert_eq!(abs(sub(1, d0)), 1);
    assert_eq!(abs(sub(n0, 1)), 1);
    // False W premise: d0=1,n0=0 gives E=0 but |d0-x*n0|=1.
    let d0 = 1i128;
    assert_eq!(abs(sub(1, d0)), 0);
    assert_eq!(abs(sub(d0, mul(x, n0))), 1);
    assert_eq!(abs(sub(n0, 1)), 1);
    // Success is conditional on justified premises, not certification of lies.
}

#[test]
fn monotonicity_holds_only_for_successful_pairs_at_fixed_f_and_m() {
    let mut pairs = 0;
    for f in [0, 2] {
        for m in [0, 1, 8] {
            let mut base = spec(f, m);
            base.xmin = 2;
            base.xmax = 3;
            base.w0 = 1;
            let lower = reciprocal_bound(&base).unwrap().code();
            // Increase E0, W0, xmax, or decrease positive xmin.
            for field in 0..4 {
                let mut higher = base;
                match field {
                    0 => higher.e0 += 1,
                    1 => higher.w0 += 1,
                    2 => higher.xmax += 1,
                    _ => higher.xmin -= 1,
                }
                assert!(reciprocal_bound(&higher).unwrap().code() >= lower);
                pairs += 1;
            }
            // Increase each active error separately, preserving inactive zeros.
            for index in 0..usize::from(m) {
                for field in 0..2 {
                    let mut higher = base;
                    if field == 0 {
                        higher.steps[index].delta_d = 1;
                    } else {
                        higher.steps[index].delta_n = 1;
                    }
                    assert!(reciprocal_bound(&higher).unwrap().code() >= lower);
                    pairs += 1;
                }
            }
        }
    }
    assert_eq!(pairs, 60);
    // No assertion across f/m changes or overflow; no convergence claim.
}
