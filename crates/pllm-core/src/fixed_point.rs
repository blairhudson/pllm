use std::{error::Error, fmt};

pub const Q14_TO_Q7_PROFILE: &str = "pllm.numeric.rescale.q14_to_q7.v1";
pub const Q14_TO_Q10_PROFILE: &str = "pllm.numeric.rescale.q14_to_q10.centered_i32_u32.v1";
pub const Q7_MULTIPLY_PROFILE: &str = "pllm.numeric.multiply.q7.v1";
pub const GATED_MULTIPLY_Q7_PROFILE: &str = "pllm.numeric.gated_multiply.q7.v1";
pub const SIGNED_Q7_SCALE: i16 = 128;
pub const SIGNED_Q7_MIN: i16 = -128;
pub const SIGNED_Q7_MAX: i16 = 128;
pub const SIGNED_Q14_MIN: i32 = -16_384;
pub const SIGNED_Q14_MAX: i32 = 16_384;
/// Exact declared Q14 input domain: the signed-Q10 i16 endpoints scaled by 16.
pub const Q14_TO_Q10_INPUT_MIN: i32 = i16::MIN as i32 * 16;
pub const Q14_TO_Q10_INPUT_MAX: i32 = i16::MAX as i32 * 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedPointError {
    Q14InputOutOfRange { value: i32 },
    Q14ToQ10InputOutOfRange { value: i32 },
    Q10OutputOutOfRange { value: i64 },
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
            Self::Q14ToQ10InputOutOfRange { value } => write!(
                formatter,
                "signed Q14 input {value} is outside [{Q14_TO_Q10_INPUT_MIN}, {Q14_TO_Q10_INPUT_MAX}]"
            ),
            Self::Q10OutputOutOfRange { value } => {
                write!(formatter, "rescaled Q10 output {value} is outside signed i16")
            }
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

/// Convert centered signed Q14 to signed Q10 by dividing by 16, ties-to-even.
///
/// The accepted domain is exactly [`Q14_TO_Q10_INPUT_MIN`] through
/// [`Q14_TO_Q10_INPUT_MAX`]. No saturation or wrapping is performed.
pub fn rescale_q14_to_q10(value: i32) -> Result<i16, FixedPointError> {
    if !(Q14_TO_Q10_INPUT_MIN..=Q14_TO_Q10_INPUT_MAX).contains(&value) {
        return Err(FixedPointError::Q14ToQ10InputOutOfRange { value });
    }
    let rounded = div_round_ties_even(i64::from(value), 16);
    i16::try_from(rounded).map_err(|_| FixedPointError::Q10OutputOutOfRange { value: rounded })
}

/// Decode a centered two's-complement `u32` bit pattern, then rescale Q14 to Q10.
pub fn rescale_q14_to_q10_centered_u32(value: u32) -> Result<i16, FixedPointError> {
    rescale_q14_to_q10(i32::from_ne_bytes(value.to_ne_bytes()))
}

pub fn rescale_q14_to_q7_tensor(input: &[i32]) -> Result<Vec<i16>, FixedPointError> {
    input.iter().copied().map(rescale_q14_to_q7).collect()
}

pub fn rescale_q14_to_q10_centered_u32_tensor(input: &[u32]) -> Result<Vec<i16>, FixedPointError> {
    input
        .iter()
        .copied()
        .map(rescale_q14_to_q10_centered_u32)
        .collect()
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
    i64::try_from(div_round_ties_even_i128(
        i128::from(numerator),
        i128::from(denominator),
    ))
    .expect("an i64 numerator divided by a positive i64 denominator fits i64")
}

pub(crate) fn div_round_ties_even_i128(numerator: i128, denominator: i128) -> i128 {
    debug_assert!(denominator > 0);
    let quotient = numerator.div_euclid(denominator);
    let remainder = numerator.rem_euclid(denominator);
    match remainder.cmp(&(denominator - remainder)) {
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
    fn q14_to_q10_matches_exhaustive_signed_oracle() {
        for input in Q14_TO_Q10_INPUT_MIN..=Q14_TO_Q10_INPUT_MAX {
            let expected = (f64::from(input) / 16.0).round_ties_even() as i16;
            assert_eq!(rescale_q14_to_q10(input), Ok(expected), "input={input}");
        }
    }

    #[test]
    fn q14_to_q10_locks_boundaries_ties_signs_and_centered_u32() {
        assert_eq!(
            Q14_TO_Q10_PROFILE,
            "pllm.numeric.rescale.q14_to_q10.centered_i32_u32.v1"
        );
        assert_eq!(Q14_TO_Q10_INPUT_MIN, -524_288);
        assert_eq!(Q14_TO_Q10_INPUT_MAX, 524_272);
        for (input, expected) in [
            (Q14_TO_Q10_INPUT_MIN, i16::MIN),
            (-40, -2),
            (-24, -2),
            (-8, 0),
            (8, 0),
            (24, 2),
            (40, 2),
            (Q14_TO_Q10_INPUT_MAX, i16::MAX),
        ] {
            assert_eq!(rescale_q14_to_q10(input), Ok(expected));
            assert_eq!(
                rescale_q14_to_q10_centered_u32(u32::from_ne_bytes(input.to_ne_bytes())),
                Ok(expected)
            );
        }
        for input in [Q14_TO_Q10_INPUT_MIN - 1, Q14_TO_Q10_INPUT_MAX + 1] {
            assert_eq!(
                rescale_q14_to_q10(input),
                Err(FixedPointError::Q14ToQ10InputOutOfRange { value: input })
            );
        }
    }

    #[test]
    fn q14_to_q10_centered_u32_tensor_preserves_order_and_bounds() {
        let centered = |value: i32| u32::from_ne_bytes(value.to_ne_bytes());
        let input = [
            centered(0),
            centered(8),
            centered(-8),
            centered(24),
            centered(-24),
            centered(Q14_TO_Q10_INPUT_MIN),
            centered(Q14_TO_Q10_INPUT_MAX),
        ];
        assert_eq!(
            rescale_q14_to_q10_centered_u32_tensor(&input),
            Ok(vec![0, 0, 0, 2, -2, i16::MIN, i16::MAX])
        );
        assert_eq!(
            rescale_q14_to_q10_centered_u32_tensor(&[centered(40), centered(-40)]),
            Ok(vec![2, -2])
        );
        for value in [Q14_TO_Q10_INPUT_MIN - 1, Q14_TO_Q10_INPUT_MAX + 1, i32::MAX] {
            assert_eq!(
                rescale_q14_to_q10_centered_u32_tensor(&[centered(value)]),
                Err(FixedPointError::Q14ToQ10InputOutOfRange { value })
            );
        }
        assert_eq!(
            rescale_q14_to_q10_centered_u32_tensor(&[centered(0), centered(i32::MAX)]),
            Err(FixedPointError::Q14ToQ10InputOutOfRange { value: i32::MAX })
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
