use std::{error::Error, fmt};

pub const Q14_TO_Q7_PROFILE: &str = "pllm.numeric.rescale.q14_to_q7.v1";
pub const Q7_MULTIPLY_PROFILE: &str = "pllm.numeric.multiply.q7.v1";
pub const GATED_MULTIPLY_Q7_PROFILE: &str = "pllm.numeric.gated_multiply.q7.v1";
pub const SIGNED_Q7_SCALE: i16 = 128;
pub const SIGNED_Q7_MIN: i16 = -128;
pub const SIGNED_Q7_MAX: i16 = 128;
pub const SIGNED_Q14_MIN: i32 = -16_384;
pub const SIGNED_Q14_MAX: i32 = 16_384;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedPointError {
    Q14InputOutOfRange { value: i32 },
    Q7InputOutOfRange { value: i16 },
    LengthMismatch { left: usize, right: usize },
}

impl fmt::Display for FixedPointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Q14InputOutOfRange { value } => write!(
                formatter,
                "signed Q14 input {value} is outside [{SIGNED_Q14_MIN}, {SIGNED_Q14_MAX}]"
            ),
            Self::Q7InputOutOfRange { value } => write!(
                formatter,
                "signed Q7 input {value} is outside [{SIGNED_Q7_MIN}, {SIGNED_Q7_MAX}]"
            ),
            Self::LengthMismatch { left, right } => write!(
                formatter,
                "Q7 multiplication requires equal lengths, received {left} and {right}"
            ),
        }
    }
}

impl Error for FixedPointError {}

/// Convert signed Q14 in the bounded [-1, 1] domain to signed Q7.
pub fn rescale_q14_to_q7(value: i32) -> Result<i16, FixedPointError> {
    if !(SIGNED_Q14_MIN..=SIGNED_Q14_MAX).contains(&value) {
        return Err(FixedPointError::Q14InputOutOfRange { value });
    }
    Ok(div_round_ties_even(i64::from(value), i64::from(SIGNED_Q7_SCALE)) as i16)
}

pub fn rescale_q14_to_q7_tensor(input: &[i32]) -> Result<Vec<i16>, FixedPointError> {
    input.iter().copied().map(rescale_q14_to_q7).collect()
}

/// Multiply signed Q7 values and return signed Q7 with ties-to-even rescaling.
pub fn multiply_q7(left: i16, right: i16) -> Result<i16, FixedPointError> {
    validate_q7(left)?;
    validate_q7(right)?;
    Ok(div_round_ties_even(
        i64::from(left) * i64::from(right),
        i64::from(SIGNED_Q7_SCALE),
    ) as i16)
}

/// Exact bounded composition used by the protected gated-MLP reference slice.
pub fn gated_multiply_q7(gate: i16, up: i16) -> Result<i16, FixedPointError> {
    validate_q7(gate)?;
    validate_q7(up)?;
    let activated = crate::activation::silu_quadratic_q7(gate)
        .map_err(|_| FixedPointError::Q7InputOutOfRange { value: gate })?;
    multiply_q7(activated, up)
}

pub fn multiply_q7_tensor(left: &[i16], right: &[i16]) -> Result<Vec<i16>, FixedPointError> {
    if left.len() != right.len() {
        return Err(FixedPointError::LengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| multiply_q7(*left, *right))
        .collect()
}

fn validate_q7(value: i16) -> Result<(), FixedPointError> {
    if !(SIGNED_Q7_MIN..=SIGNED_Q7_MAX).contains(&value) {
        return Err(FixedPointError::Q7InputOutOfRange { value });
    }
    Ok(())
}

pub(crate) fn div_round_ties_even(numerator: i64, denominator: i64) -> i64 {
    debug_assert!(denominator > 0);
    let quotient = numerator.div_euclid(denominator);
    let remainder = numerator.rem_euclid(denominator);
    match (remainder * 2).cmp(&denominator) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal if quotient % 2 == 0 => quotient,
        std::cmp::Ordering::Equal => quotient + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescales_q14_with_signed_ties_to_even() {
        for (input, expected) in [
            (-16_384, -128),
            (-320, -2),
            (-192, -2),
            (-64, 0),
            (64, 0),
            (192, 2),
            (320, 2),
            (16_384, 128),
        ] {
            assert_eq!(rescale_q14_to_q7(input), Ok(expected));
        }
    }

    #[test]
    fn rejects_q14_values_outside_the_bounded_domain() {
        assert_eq!(
            rescale_q14_to_q7(-16_385),
            Err(FixedPointError::Q14InputOutOfRange { value: -16_385 })
        );
        assert_eq!(
            rescale_q14_to_q7(16_385),
            Err(FixedPointError::Q14InputOutOfRange { value: 16_385 })
        );
    }

    #[test]
    fn multiplication_matches_an_independent_exhaustive_oracle() {
        for left in SIGNED_Q7_MIN..=SIGNED_Q7_MAX {
            for right in SIGNED_Q7_MIN..=SIGNED_Q7_MAX {
                let expected = (f64::from(left) * f64::from(right) / f64::from(SIGNED_Q7_SCALE))
                    .round_ties_even() as i16;
                assert_eq!(multiply_q7(left, right), Ok(expected));
            }
        }
    }

    #[test]
    fn gated_multiplication_matches_explicit_composition_exhaustively() {
        for gate in SIGNED_Q7_MIN..=SIGNED_Q7_MAX {
            for up in SIGNED_Q7_MIN..=SIGNED_Q7_MAX {
                assert_eq!(
                    gated_multiply_q7(gate, up),
                    multiply_q7(crate::activation::silu_quadratic_q7(gate).unwrap(), up)
                );
            }
        }
    }

    #[test]
    fn multiplication_rejects_bad_inputs_and_lengths() {
        assert_eq!(
            multiply_q7(-129, 0),
            Err(FixedPointError::Q7InputOutOfRange { value: -129 })
        );
        assert_eq!(
            multiply_q7_tensor(&[1], &[1, 2]),
            Err(FixedPointError::LengthMismatch { left: 1, right: 2 })
        );
    }
}
