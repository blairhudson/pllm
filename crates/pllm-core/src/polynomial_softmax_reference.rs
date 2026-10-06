//! Bounded plaintext softmax-approximation gate, independently transcribed from
//! NEXUS Eq. (4) / §IV and THOR Algorithm 3 / Appendix C.
//! Exact f64 normalization is an optimistic oracle, not protected reciprocal,
//! CKKS arithmetic, a ciphertext schedule, or a whole-paper implementation.

pub const MAX_ROWS: usize = 512;
pub const PUBLIC_SCORE_BOUND: f32 = 16.0;

#[derive(Clone, Copy)]
pub enum Method {
    RepeatedSquaring,
    NormalizedSquaring,
}

const THOR_EXP: [f64; 16] = [
    9.9999e-1, 9.9999e-1, 5.0002e-1, 1.6667e-1, 4.1653e-2, 8.3316e-3, 1.3920e-3, 1.9871e-4,
    2.4471e-5, 2.7286e-6, 2.9466e-7, 2.6438e-8, 1.4896e-9, 1.2104e-10, 2.0777e-11, 1.3376e-12,
];

fn normalize(values: &mut [f64]) -> Result<(), &'static str> {
    let sum: f64 = values.iter().sum();
    if !sum.is_finite() || sum <= 0.0 || values.iter().any(|&x| !x.is_finite() || x < 0.0) {
        return Err("invalid approximate exponential");
    }
    for value in values {
        *value /= sum;
    }
    Ok(())
}

/// Caller removes only publicly invalid causal slots before evaluation. Public
/// bounds and approximation parameters stay fixed across all private rows.
pub fn evaluate(scores: &[f32], method: Method) -> Result<Vec<f32>, &'static str> {
    if scores.is_empty() || scores.len() > MAX_ROWS {
        return Err("softmax row exceeds reference capacity");
    }
    if scores
        .iter()
        .any(|&x| !x.is_finite() || x.abs() > PUBLIC_SCORE_BOUND)
    {
        return Err("softmax score exceeds fixed public domain");
    }
    let mut values: Vec<f64> = scores
        .iter()
        .map(|&x| match method {
            Method::RepeatedSquaring => {
                // Fixed public maximum, never a maximum computed from private data.
                let mut y = 1.0 + (f64::from(x) - f64::from(PUBLIC_SCORE_BOUND)) / 256.0;
                for _ in 0..8 {
                    y *= y;
                }
                y
            }
            Method::NormalizedSquaring => {
                // Public delta1=1, delta2=32; evaluate on [-0.5,0.5], then five
                // square-and-normalize steps. Parameters differ from BERT tuning.
                let y = f64::from(x) / 32.0;
                THOR_EXP
                    .iter()
                    .rev()
                    .fold(0.0, |acc, coefficient| acc * y + coefficient)
            }
        })
        .collect();
    normalize(&mut values)?;
    if matches!(method, Method::NormalizedSquaring) {
        for _ in 0..5 {
            for value in &mut values {
                *value *= *value;
            }
            normalize(&mut values)?;
        }
    }
    Ok(values.into_iter().map(|x| x as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_approximations_preserve_probability_and_order() {
        for method in [Method::RepeatedSquaring, Method::NormalizedSquaring] {
            for row in [vec![0.], vec![16.; 512], vec![-16., -2., 0., 1., 16.]] {
                let output = evaluate(&row, method).unwrap();
                assert!((output.iter().map(|&x| f64::from(x)).sum::<f64>() - 1.0).abs() < 1e-6);
                assert!(output.windows(2).all(|p| p[0] <= p[1]));
                assert!(output.iter().all(|&x| x >= 0.0));
            }
        }
    }

    #[test]
    fn published_exponential_is_not_exact_softmax_or_shift_invariant() {
        let row = [-3., -1., 0., 2.];
        let exact: Vec<_> = row.iter().map(|&x| f64::from(x).exp()).collect();
        let sum: f64 = exact.iter().sum();
        let mut errors = Vec::new();
        for method in [Method::RepeatedSquaring, Method::NormalizedSquaring] {
            let actual = evaluate(&row, method).unwrap();
            let error = actual
                .iter()
                .zip(&exact)
                .map(|(&a, &b)| (f64::from(a) - b / sum).abs())
                .fold(0.0, f64::max);
            assert!(error > 0.0);
            errors.push(error);
        }
        assert!(errors[0] > 0.01);
        assert!(errors[1] < 0.001);
        let shifted = row.map(|x| x + 5.0);
        assert_ne!(
            evaluate(&row, Method::RepeatedSquaring).unwrap(),
            evaluate(&shifted, Method::RepeatedSquaring).unwrap()
        );
    }

    #[test]
    fn reject_nonfinite_outside_domain_and_oversized_rows() {
        for row in [
            vec![],
            vec![16.01],
            vec![-16.01],
            vec![f32::NAN],
            vec![f32::NEG_INFINITY],
            vec![0.; 513],
        ] {
            assert!(evaluate(&row, Method::RepeatedSquaring).is_err());
        }
    }
}
