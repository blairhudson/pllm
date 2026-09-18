use std::{error::Error, fmt};

use crate::fixed_point::div_round_ties_even;

pub const RMS_NORM_Q10_DIRECT_PROFILE: &str = "pllm.numeric.rms_norm.q10_direct.q30_rsqrt.v1";
pub const RMS_NORM_Q10_SCALE: i32 = 1 << 10;
pub const RMS_NORM_Q30_RECIPROCAL_SCALE: u32 = 1 << 30;
pub const RMS_NORM_EPSILON_NUMERATOR: u64 = 1;
pub const RMS_NORM_EPSILON_DENOMINATOR: u64 = 1_000_000;
pub const RMS_NORM_Q10_MAX_WIDTH: usize = 8192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmsNormQ10Error {
    EmptyInput,
    EmptyWeight,
    WidthTooLarge { width: usize, maximum: usize },
    LengthMismatch { input: usize, width: usize },
    SumSquaresOutOfRange,
    ArithmeticOverflow,
    AllocationFailed,
}

impl fmt::Display for RmsNormQ10Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(formatter, "RMSNorm input must not be empty"),
            Self::EmptyWeight => write!(formatter, "RMSNorm weight must not be empty"),
            Self::WidthTooLarge { width, maximum } => {
                write!(formatter, "RMSNorm width {width} exceeds maximum {maximum}")
            }
            Self::LengthMismatch { input, width } => write!(
                formatter,
                "RMSNorm input length {input} is not divisible by width {width}"
            ),
            Self::SumSquaresOutOfRange => {
                write!(
                    formatter,
                    "RMSNorm sum of squares is outside the Q10 domain"
                )
            }
            Self::ArithmeticOverflow => write!(formatter, "RMSNorm arithmetic overflowed"),
            Self::AllocationFailed => write!(formatter, "RMSNorm output allocation failed"),
        }
    }
}

impl Error for RmsNormQ10Error {}

pub const RMS_NORM_F32_DIRECT_PROFILE: &str = "pllm.numeric.rms_norm.f32_direct.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmsNormError {
    EmptyInput,
    EmptyWeight,
    LengthMismatch { input: usize, width: usize },
    InvalidEpsilon,
    NonFiniteInput { index: usize },
    NonFiniteWeight { index: usize },
    NonFiniteIntermediate { row: usize },
}

impl fmt::Display for RmsNormError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(formatter, "RMSNorm input must not be empty"),
            Self::EmptyWeight => write!(formatter, "RMSNorm weight must not be empty"),
            Self::LengthMismatch { input, width } => write!(
                formatter,
                "RMSNorm input length {input} is not divisible by width {width}"
            ),
            Self::InvalidEpsilon => {
                write!(formatter, "RMSNorm epsilon must be positive and finite")
            }
            Self::NonFiniteInput { index } => {
                write!(formatter, "RMSNorm input {index} is not finite")
            }
            Self::NonFiniteWeight { index } => {
                write!(formatter, "RMSNorm weight {index} is not finite")
            }
            Self::NonFiniteIntermediate { row } => write!(
                formatter,
                "RMSNorm row {row} produced a non-finite intermediate"
            ),
        }
    }
}

impl Error for RmsNormError {}

/// Exact integer Q10 direct-weight RMSNorm with decimal epsilon 1/1,000,000.
///
/// The reciprocal normalizer is rounded to Q30, then each Q10 output is rounded
/// ties-to-even. Against exact RMSNorm of the encoded inputs and weights, the
/// decoded absolute arithmetic error is at most one Q10 unit (2^-10).
pub fn rms_norm_q10_direct(input: &[i16], weight: &[i16]) -> Result<Vec<i32>, RmsNormQ10Error> {
    if input.is_empty() {
        return Err(RmsNormQ10Error::EmptyInput);
    }
    validate_q10_width(weight.len())?;
    if input.len() % weight.len() != 0 {
        return Err(RmsNormQ10Error::LengthMismatch {
            input: input.len(),
            width: weight.len(),
        });
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| RmsNormQ10Error::AllocationFailed)?;
    for row in input.chunks_exact(weight.len()) {
        let sum_squares = row.iter().try_fold(0_u64, |sum, value| {
            let value = i64::from(*value);
            let square =
                u64::try_from(value * value).map_err(|_| RmsNormQ10Error::ArithmeticOverflow)?;
            sum.checked_add(square)
                .ok_or(RmsNormQ10Error::ArithmeticOverflow)
        })?;
        let reciprocal = i64::from(rms_normalizer_q30(sum_squares, weight.len())?);
        for (value, weight) in row.iter().zip(weight) {
            let product = i64::from(*value)
                .checked_mul(i64::from(*weight))
                .and_then(|product| product.checked_mul(reciprocal))
                .ok_or(RmsNormQ10Error::ArithmeticOverflow)?;
            let rounded = div_round_ties_even(product, 1_i64 << 30);
            output.push(i32::try_from(rounded).map_err(|_| RmsNormQ10Error::ArithmeticOverflow)?);
        }
    }
    Ok(output)
}

/// Return ties-to-even Q30 encoding of sqrt(width * 1,000,000 / statistic).
pub fn rms_normalizer_q30(sum_squares: u64, width: usize) -> Result<u32, RmsNormQ10Error> {
    validate_q10_width(width)?;
    let width = u64::try_from(width).map_err(|_| RmsNormQ10Error::ArithmeticOverflow)?;
    let maximum_sum = width
        .checked_mul(1_u64 << 30)
        .ok_or(RmsNormQ10Error::ArithmeticOverflow)?;
    if sum_squares > maximum_sum {
        return Err(RmsNormQ10Error::SumSquaresOutOfRange);
    }

    // A / (width * 1024^2 * 1,000,000) is mean(x^2) + exact decimal epsilon.
    let statistic = sum_squares
        .checked_mul(RMS_NORM_EPSILON_DENOMINATOR)
        .and_then(|value| value.checked_add(width * (1_u64 << 20)))
        .ok_or(RmsNormQ10Error::ArithmeticOverflow)?;
    let numerator = width
        .checked_mul(RMS_NORM_EPSILON_DENOMINATOR)
        .ok_or(RmsNormQ10Error::ArithmeticOverflow)?;
    let target = u128::from(numerator) << 60;

    // Fixed-round binary search for floor(2^30 * sqrt(numerator / statistic)).
    let mut low = 0_u64;
    let mut high = u64::from(RMS_NORM_Q30_RECIPROCAL_SCALE);
    for _ in 0..31 {
        let midpoint = (low + high + 1) / 2;
        let product = u128::from(midpoint) * u128::from(midpoint) * u128::from(statistic);
        if product <= target {
            low = midpoint;
        } else {
            high = midpoint - 1;
        }
    }

    // Compare the exact square root against low + 1/2 without floating point.
    let twice_low_plus_one = u128::from(low) * 2 + 1;
    let midpoint_left = twice_low_plus_one * twice_low_plus_one * u128::from(statistic);
    let midpoint_right = target * 4;
    let rounded = match midpoint_left.cmp(&midpoint_right) {
        std::cmp::Ordering::Less => low + 1,
        std::cmp::Ordering::Greater => low,
        std::cmp::Ordering::Equal if low % 2 == 0 => low,
        std::cmp::Ordering::Equal => low + 1,
    };
    u32::try_from(rounded).map_err(|_| RmsNormQ10Error::ArithmeticOverflow)
}

fn validate_q10_width(width: usize) -> Result<(), RmsNormQ10Error> {
    if width == 0 {
        return Err(RmsNormQ10Error::EmptyWeight);
    }
    if width > RMS_NORM_Q10_MAX_WIDTH {
        return Err(RmsNormQ10Error::WidthTooLarge {
            width,
            maximum: RMS_NORM_Q10_MAX_WIDTH,
        });
    }
    Ok(())
}

/// RMSNorm over contiguous last-axis rows using direct FP32 weights.
///
/// Each row evaluates `x / sqrt(mean(x * x) + epsilon) * weight`. The scalar
/// reduction order is left-to-right. This profile has no bias, weight offset,
/// clipping, quantization, or saturation.
pub fn rms_norm_f32_direct(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, RmsNormError> {
    if input.is_empty() {
        return Err(RmsNormError::EmptyInput);
    }
    if weight.is_empty() {
        return Err(RmsNormError::EmptyWeight);
    }
    if input.len() % weight.len() != 0 {
        return Err(RmsNormError::LengthMismatch {
            input: input.len(),
            width: weight.len(),
        });
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(RmsNormError::InvalidEpsilon);
    }
    if let Some(index) = input.iter().position(|value| !value.is_finite()) {
        return Err(RmsNormError::NonFiniteInput { index });
    }
    if let Some(index) = weight.iter().position(|value| !value.is_finite()) {
        return Err(RmsNormError::NonFiniteWeight { index });
    }

    let width = weight.len();
    let mut output = Vec::with_capacity(input.len());
    for (row_index, row) in input.chunks_exact(width).enumerate() {
        let mut sum = 0.0f32;
        for value in row {
            sum += value * value;
            if !sum.is_finite() {
                return Err(RmsNormError::NonFiniteIntermediate { row: row_index });
            }
        }
        let denominator = (sum / width as f32 + epsilon).sqrt();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(RmsNormError::NonFiniteIntermediate { row: row_index });
        }
        for (value, scale) in row.iter().zip(weight) {
            let normalized = *value / denominator;
            let result = normalized * scale;
            if !result.is_finite() {
                return Err(RmsNormError::NonFiniteIntermediate { row: row_index });
            }
            output.push(result);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q10_profile_uses_exact_decimal_epsilon_and_ties_to_even() {
        assert_eq!(
            rms_normalizer_q30(0, 1),
            Ok(1_048_576_000),
            "sqrt(1_000_000 / 2^20) is exactly 125/128"
        );
        assert_eq!(
            rms_norm_q10_direct(&[1024, 1024], &[1024, 1024]),
            Ok(vec![1024, 1024])
        );
        assert_eq!(
            rms_norm_q10_direct(&[1024, 0], &[1024, 1024]),
            Ok(vec![1448, 0])
        );
    }

    #[test]
    fn q10_profile_stays_within_one_raw_unit_of_exact_encoded_rms_norm() {
        for width in [1_usize, 2, 128, 896, 2560, 8192] {
            let input = (0..width)
                .map(|index| ((index * 7919 % 65536) as i32 - 32768) as i16)
                .collect::<Vec<_>>();
            let weight = (0..width)
                .map(|index| ((index * 3571 % 65536) as i32 - 32768) as i16)
                .collect::<Vec<_>>();
            let output = rms_norm_q10_direct(&input, &weight).unwrap();
            let sum_squares = input
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>();
            let normalizer = ((width as f64 * RMS_NORM_EPSILON_DENOMINATOR as f64)
                / (sum_squares * RMS_NORM_EPSILON_DENOMINATOR as f64
                    + width as f64 * f64::from(RMS_NORM_Q10_SCALE).powi(2)))
            .sqrt();
            for ((value, weight), actual) in input.iter().zip(&weight).zip(output) {
                let exact = f64::from(*value) * f64::from(*weight) * normalizer;
                assert!(
                    (f64::from(actual) - exact).abs() <= 1.000_001,
                    "width={width} actual={actual} exact={exact}"
                );
            }
        }
    }

    #[test]
    fn q10_profile_accepts_proven_extremes_and_rejects_out_of_contract_inputs() {
        let input = vec![i16::MIN; RMS_NORM_Q10_MAX_WIDTH];
        let weight = vec![i16::MIN; RMS_NORM_Q10_MAX_WIDTH];
        let output = rms_norm_q10_direct(&input, &weight).unwrap();
        assert_eq!(output.len(), RMS_NORM_Q10_MAX_WIDTH);
        assert!(output.iter().all(|value| *value > 0));

        assert_eq!(
            rms_norm_q10_direct(&[], &[1]),
            Err(RmsNormQ10Error::EmptyInput)
        );
        assert_eq!(
            rms_norm_q10_direct(&[1], &[]),
            Err(RmsNormQ10Error::EmptyWeight)
        );
        assert_eq!(
            rms_norm_q10_direct(&[1, 2, 3], &[1, 1]),
            Err(RmsNormQ10Error::LengthMismatch { input: 3, width: 2 })
        );
        assert_eq!(
            rms_normalizer_q30(0, RMS_NORM_Q10_MAX_WIDTH + 1),
            Err(RmsNormQ10Error::WidthTooLarge {
                width: RMS_NORM_Q10_MAX_WIDTH + 1,
                maximum: RMS_NORM_Q10_MAX_WIDTH,
            })
        );
        assert_eq!(
            rms_normalizer_q30((1_u64 << 30) + 1, 1),
            Err(RmsNormQ10Error::SumSquaresOutOfRange)
        );
    }

    #[test]
    fn evaluates_locked_rows_with_direct_weights() {
        assert_eq!(
            rms_norm_f32_direct(&[1.0, -1.0, 0.0, 0.0], &[2.0, 4.0], 3.0),
            Ok(vec![1.0, -2.0, 0.0, 0.0])
        );
        assert_eq!(
            rms_norm_f32_direct(&[3.0, 4.0], &[0.0, 0.0], 1.0),
            Ok(vec![0.0, 0.0])
        );
    }

    #[test]
    fn normalizes_each_last_axis_row_independently() {
        let output = rms_norm_f32_direct(&[1.0, 1.0, 3.0, 4.0], &[1.0, 2.0], 1.0).unwrap();
        let expected = [
            1.0 / 2.0f32.sqrt(),
            2.0 / 2.0f32.sqrt(),
            3.0 / 13.5f32.sqrt(),
            8.0 / 13.5f32.sqrt(),
        ];
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() <= f32::EPSILON);
        }
    }

    #[test]
    fn epsilon_is_inside_the_square_root() {
        let output = rms_norm_f32_direct(&[0.001], &[1.0], 0.000_001).unwrap();
        assert!((output[0] - (0.5f32).sqrt()).abs() <= f32::EPSILON);
    }

    #[test]
    fn supports_representative_hidden_and_head_widths() {
        for width in [128, 896, 2560] {
            let input = (0..width)
                .map(|index| (index as f32 % 17.0 - 8.0) / 8.0)
                .collect::<Vec<_>>();
            let weight = (0..width)
                .map(|index| 0.5 + (index as f32 % 11.0) / 20.0)
                .collect::<Vec<_>>();
            let output = rms_norm_f32_direct(&input, &weight, 0.000_001).unwrap();
            assert_eq!(output.len(), width);
            for (value, scale) in output.iter().zip(&weight) {
                assert!(value.abs() <= (width as f32).sqrt() * scale.abs() + 0.000_01);
            }
        }
    }

    #[test]
    fn rejects_invalid_shapes_values_and_intermediates() {
        assert_eq!(
            rms_norm_f32_direct(&[], &[1.0], 0.000_001),
            Err(RmsNormError::EmptyInput)
        );
        assert_eq!(
            rms_norm_f32_direct(&[1.0], &[], 0.000_001),
            Err(RmsNormError::EmptyWeight)
        );
        assert_eq!(
            rms_norm_f32_direct(&[1.0, 2.0, 3.0], &[1.0, 1.0], 0.000_001),
            Err(RmsNormError::LengthMismatch { input: 3, width: 2 })
        );
        for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                rms_norm_f32_direct(&[1.0], &[1.0], epsilon),
                Err(RmsNormError::InvalidEpsilon)
            );
        }
        assert_eq!(
            rms_norm_f32_direct(&[f32::NAN], &[1.0], 0.000_001),
            Err(RmsNormError::NonFiniteInput { index: 0 })
        );
        assert_eq!(
            rms_norm_f32_direct(&[1.0], &[f32::INFINITY], 0.000_001),
            Err(RmsNormError::NonFiniteWeight { index: 0 })
        );
        assert_eq!(
            rms_norm_f32_direct(&[f32::MAX], &[1.0], 0.000_001),
            Err(RmsNormError::NonFiniteIntermediate { row: 0 })
        );
    }
}
