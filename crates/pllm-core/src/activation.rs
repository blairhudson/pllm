use std::{error::Error, fmt};

use crate::fixed_point::div_round_ties_even;

pub const SILU_QUADRATIC_Q7_PROFILE: &str = "pllm.numeric.silu.quadratic_q7.v1";
pub const SILU_QUADRATIC_Q7_SCALE: i16 = 128;
pub const SILU_QUADRATIC_Q7_MIN: i16 = -128;
pub const SILU_QUADRATIC_Q7_MAX: i16 = 128;
pub const SILU_QUADRATIC_Q7_MAX_ABS_ERROR: f64 = 0.02285;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActivationRangeError {
    pub value: i16,
}

impl fmt::Display for ActivationRangeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Q7 SiLU input {} is outside [{}, {}]",
            self.value, SILU_QUADRATIC_Q7_MIN, SILU_QUADRATIC_Q7_MAX
        )
    }
}

impl Error for ActivationRangeError {}

/// Evaluate q(x) = x/2 + x^2/4 on Q7 input with deterministic ties-to-even rounding.
pub fn silu_quadratic_q7(value: i16) -> Result<i16, ActivationRangeError> {
    if !(SILU_QUADRATIC_Q7_MIN..=SILU_QUADRATIC_Q7_MAX).contains(&value) {
        return Err(ActivationRangeError { value });
    }
    let value = i64::from(value);
    let numerator = value * value + 256 * value;
    Ok(div_round_ties_even(numerator, 512) as i16)
}

pub fn silu_quadratic_q7_tensor(input: &[i16]) -> Result<Vec<i16>, ActivationRangeError> {
    input.iter().copied().map(silu_quadratic_q7).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_locked_q7_points() {
        assert_eq!(silu_quadratic_q7(-128), Ok(-32));
        assert_eq!(silu_quadratic_q7(-64), Ok(-24));
        assert_eq!(silu_quadratic_q7(0), Ok(0));
        assert_eq!(silu_quadratic_q7(64), Ok(40));
        assert_eq!(silu_quadratic_q7(128), Ok(96));
    }

    #[test]
    fn rounds_halfway_values_to_even() {
        for (input, expected) in [
            (-112, -32),
            (-80, -28),
            (-48, -20),
            (-16, -8),
            (16, 8),
            (48, 28),
            (80, 52),
            (112, 80),
        ] {
            assert_eq!(silu_quadratic_q7(input), Ok(expected));
        }
        assert_eq!(div_round_ties_even(3, 2), 2);
        assert_eq!(div_round_ties_even(5, 2), 2);
        assert_eq!(div_round_ties_even(-3, 2), -2);
        assert_eq!(div_round_ties_even(-5, 2), -2);
    }

    #[test]
    fn exhaustive_outputs_match_independent_exact_oracle() {
        for encoded in SILU_QUADRATIC_Q7_MIN..=SILU_QUADRATIC_Q7_MAX {
            let value = i64::from(encoded);
            let expected = ((value * value + 256 * value) as f64 / 512.0).round_ties_even();
            assert_eq!(silu_quadratic_q7(encoded), Ok(expected as i16));
        }
    }

    #[test]
    fn rejects_values_outside_the_locked_domain() {
        assert_eq!(
            silu_quadratic_q7(-129),
            Err(ActivationRangeError { value: -129 })
        );
        assert_eq!(
            silu_quadratic_q7(129),
            Err(ActivationRangeError { value: 129 })
        );
    }

    #[test]
    fn approximation_meets_the_declared_error_bound() {
        for encoded in SILU_QUADRATIC_Q7_MIN..=SILU_QUADRATIC_Q7_MAX {
            let x = f64::from(encoded) / f64::from(SILU_QUADRATIC_Q7_SCALE);
            let expected = x / (1.0 + (-x).exp());
            let actual =
                f64::from(silu_quadratic_q7(encoded).unwrap()) / f64::from(SILU_QUADRATIC_Q7_SCALE);
            assert!((actual - expected).abs() <= SILU_QUADRATIC_Q7_MAX_ABS_ERROR);
        }
    }

    #[test]
    fn tensor_path_matches_scalar_oracle() {
        let input = [-128, -1, 0, 1, 128];
        assert_eq!(
            silu_quadratic_q7_tensor(&input).unwrap(),
            input
                .into_iter()
                .map(|value| silu_quadratic_q7(value).unwrap())
                .collect::<Vec<_>>()
        );
    }
}
