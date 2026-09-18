use std::{error::Error, fmt};

use crate::fixed_point::div_round_ties_even;

pub const ROPE_Q30_COEFFICIENT_PROFILE: &str = "pllm.numeric.rope.q30.libm.v1";
/// Full Q10 execution profile using [`ROPE_Q30_COEFFICIENT_PROFILE`] coefficients.
pub const ROPE_Q10_PROFILE: &str = "pllm.numeric.rope.q10_q30.libm_split_half.v1";
pub const ROPE_Q10_COEFFICIENT_PROFILE: &str = ROPE_Q30_COEFFICIENT_PROFILE;
pub const ROPE_Q10_LAYOUT: &str = "[batch,heads,sequence,head_dim]";
pub const ROPE_Q10_PAIRING: &str = "split_half[j,j+rotary_dimensions/2]";
pub const ROPE_Q10_ROUNDING: &str = "q30_coefficients_and_q10_outputs_ties_to_even";
pub const ROPE_Q30_SCALE: i64 = 1_i64 << 30;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeQ10Config {
    pub theta: f64,
    pub rotary_dimensions: usize,
    /// Exclusive upper bound for every absolute position.
    pub maximum_position: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RopeQ30Coefficient {
    pub cosine: i32,
    pub sine: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RopeError {
    InvalidTheta,
    InvalidRotaryDimensions {
        rotary_dimensions: usize,
    },
    RotaryDimensionsExceedHead {
        rotary_dimensions: usize,
        head_dim: usize,
    },
    InvalidMaximumPosition,
    InvalidShape {
        dimension: &'static str,
    },
    ShapeOverflow,
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    PositionOutOfRange {
        index: usize,
        position: u32,
        maximum_position: u32,
    },
    NonFiniteCoefficient {
        pair: usize,
    },
    ArithmeticOverflow,
    OutputOutOfRange {
        index: usize,
        value: i64,
    },
    AllocationFailed,
}

impl fmt::Display for RopeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTheta => write!(formatter, "RoPE theta must be positive and finite"),
            Self::InvalidRotaryDimensions { rotary_dimensions } => write!(
                formatter,
                "RoPE rotary dimensions must be positive and even, received {rotary_dimensions}"
            ),
            Self::RotaryDimensionsExceedHead {
                rotary_dimensions,
                head_dim,
            } => write!(
                formatter,
                "RoPE rotary dimensions {rotary_dimensions} exceed head dimension {head_dim}"
            ),
            Self::InvalidMaximumPosition => {
                write!(formatter, "RoPE maximum position must be positive")
            }
            Self::InvalidShape { dimension } => {
                write!(formatter, "RoPE {dimension} dimension must be positive")
            }
            Self::ShapeOverflow => write!(formatter, "RoPE shape arithmetic overflowed"),
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "RoPE {tensor} length must be {expected}, received {actual}"
            ),
            Self::PositionOutOfRange {
                index,
                position,
                maximum_position,
            } => write!(
                formatter,
                "RoPE position {position} at index {index} is not below {maximum_position}"
            ),
            Self::NonFiniteCoefficient { pair } => {
                write!(
                    formatter,
                    "RoPE pair {pair} produced a non-finite coefficient"
                )
            }
            Self::ArithmeticOverflow => write!(formatter, "RoPE arithmetic overflowed"),
            Self::OutputOutOfRange { index, value } => write!(
                formatter,
                "RoPE output {value} at index {index} is outside signed i16"
            ),
            Self::AllocationFailed => write!(formatter, "RoPE allocation failed"),
        }
    }
}

impl Error for RopeError {}

/// Return one split-half coefficient row as signed Q30 values.
pub fn rope_q30_coefficients(
    position: u32,
    config: RopeQ10Config,
) -> Result<Vec<RopeQ30Coefficient>, RopeError> {
    validate_config(config)?;
    validate_position(position, 0, config.maximum_position)?;

    let pairs = config.rotary_dimensions / 2;
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(pairs)
        .map_err(|_| RopeError::AllocationFailed)?;
    for pair in 0..pairs {
        let exponent = -2.0 * pair as f64 / config.rotary_dimensions as f64;
        let frequency = libm::pow(config.theta, exponent);
        let angle = f64::from(position) * frequency;
        let cosine = libm::cos(angle);
        let sine = libm::sin(angle);
        if !frequency.is_finite() || !angle.is_finite() || !cosine.is_finite() || !sine.is_finite()
        {
            return Err(RopeError::NonFiniteCoefficient { pair });
        }
        coefficients.push(RopeQ30Coefficient {
            cosine: quantize_q30(cosine, pair)?,
            sine: quantize_q30(sine, pair)?,
        });
    }
    Ok(coefficients)
}

/// Apply deterministic split-half RoPE to signed Q10 rank-four input.
pub fn rope_q10(
    input: &[i16],
    positions: &[u32],
    shape: [usize; 4],
    config: RopeQ10Config,
) -> Result<Vec<i16>, RopeError> {
    validate_config(config)?;
    let [batch, heads, sequence, head_dim] = shape;
    for (dimension, value) in [
        ("batch", batch),
        ("heads", heads),
        ("sequence", sequence),
        ("head_dim", head_dim),
    ] {
        if value == 0 {
            return Err(RopeError::InvalidShape { dimension });
        }
    }
    if config.rotary_dimensions > head_dim {
        return Err(RopeError::RotaryDimensionsExceedHead {
            rotary_dimensions: config.rotary_dimensions,
            head_dim,
        });
    }

    let expected_input = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(sequence))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or(RopeError::ShapeOverflow)?;
    let expected_positions = batch
        .checked_mul(sequence)
        .ok_or(RopeError::ShapeOverflow)?;
    validate_length("input", expected_input, input.len())?;
    validate_length("positions", expected_positions, positions.len())?;
    for (index, position) in positions.iter().copied().enumerate() {
        validate_position(position, index, config.maximum_position)?;
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| RopeError::AllocationFailed)?;
    output.extend_from_slice(input);

    let pairs = config.rotary_dimensions / 2;
    for batch_index in 0..batch {
        for sequence_index in 0..sequence {
            let position_index = batch_index
                .checked_mul(sequence)
                .and_then(|value| value.checked_add(sequence_index))
                .ok_or(RopeError::ShapeOverflow)?;
            let coefficients = rope_q30_coefficients(positions[position_index], config)?;
            for head_index in 0..heads {
                let base = batch_index
                    .checked_mul(heads)
                    .and_then(|value| value.checked_add(head_index))
                    .and_then(|value| value.checked_mul(sequence))
                    .and_then(|value| value.checked_add(sequence_index))
                    .and_then(|value| value.checked_mul(head_dim))
                    .ok_or(RopeError::ShapeOverflow)?;
                for (pair, coefficient) in coefficients.iter().enumerate() {
                    let first_index = base.checked_add(pair).ok_or(RopeError::ShapeOverflow)?;
                    let second_index = base
                        .checked_add(pair + pairs)
                        .ok_or(RopeError::ShapeOverflow)?;
                    let first = i64::from(input[first_index]);
                    let second = i64::from(input[second_index]);
                    let cosine = i64::from(coefficient.cosine);
                    let sine = i64::from(coefficient.sine);
                    let rotated_first = first
                        .checked_mul(cosine)
                        .and_then(|value| {
                            second
                                .checked_mul(sine)
                                .and_then(|term| value.checked_sub(term))
                        })
                        .ok_or(RopeError::ArithmeticOverflow)?;
                    let rotated_second = second
                        .checked_mul(cosine)
                        .and_then(|value| {
                            first
                                .checked_mul(sine)
                                .and_then(|term| value.checked_add(term))
                        })
                        .ok_or(RopeError::ArithmeticOverflow)?;
                    output[first_index] = checked_q10_output(rotated_first, first_index)?;
                    output[second_index] = checked_q10_output(rotated_second, second_index)?;
                }
            }
        }
    }
    Ok(output)
}

fn validate_config(config: RopeQ10Config) -> Result<(), RopeError> {
    if !config.theta.is_finite() || config.theta <= 0.0 {
        return Err(RopeError::InvalidTheta);
    }
    if config.rotary_dimensions == 0 || config.rotary_dimensions % 2 != 0 {
        return Err(RopeError::InvalidRotaryDimensions {
            rotary_dimensions: config.rotary_dimensions,
        });
    }
    if config.maximum_position == 0 {
        return Err(RopeError::InvalidMaximumPosition);
    }
    Ok(())
}

fn validate_position(position: u32, index: usize, maximum_position: u32) -> Result<(), RopeError> {
    if position >= maximum_position {
        return Err(RopeError::PositionOutOfRange {
            index,
            position,
            maximum_position,
        });
    }
    Ok(())
}

fn validate_length(tensor: &'static str, expected: usize, actual: usize) -> Result<(), RopeError> {
    if expected != actual {
        return Err(RopeError::LengthMismatch {
            tensor,
            expected,
            actual,
        });
    }
    Ok(())
}

fn quantize_q30(value: f64, pair: usize) -> Result<i32, RopeError> {
    let scaled = value * ROPE_Q30_SCALE as f64;
    if !scaled.is_finite() {
        return Err(RopeError::NonFiniteCoefficient { pair });
    }
    i32::try_from(scaled.round_ties_even() as i64)
        .map_err(|_| RopeError::NonFiniteCoefficient { pair })
}

fn checked_q10_output(value: i64, index: usize) -> Result<i16, RopeError> {
    let rounded = div_round_ties_even(value, ROPE_Q30_SCALE);
    i16::try_from(rounded).map_err(|_| RopeError::OutputOutOfRange {
        index,
        value: rounded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(rotary_dimensions: usize) -> RopeQ10Config {
        RopeQ10Config {
            theta: 10_000.0,
            rotary_dimensions,
            maximum_position: 128,
        }
    }

    #[test]
    fn execution_profile_binds_semantic_ir_coefficient_profile() {
        assert_eq!(
            ROPE_Q30_COEFFICIENT_PROFILE,
            "pllm.numeric.rope.q30.libm.v1"
        );
        assert_eq!(ROPE_Q10_COEFFICIENT_PROFILE, ROPE_Q30_COEFFICIENT_PROFILE);
        assert_ne!(ROPE_Q10_PROFILE, ROPE_Q30_COEFFICIENT_PROFILE);
    }

    #[test]
    fn position_zero_is_identity() {
        let input = [-32_768, -1000, 1000, 32_767];
        assert_eq!(
            rope_q10(&input, &[0], [1, 1, 1, 4], config(4)),
            Ok(input.to_vec())
        );
        assert_eq!(
            rope_q30_coefficients(0, config(4)),
            Ok(vec![
                RopeQ30Coefficient {
                    cosine: 1 << 30,
                    sine: 0,
                },
                RopeQ30Coefficient {
                    cosine: 1 << 30,
                    sine: 0,
                },
            ])
        );
    }

    #[test]
    fn matches_locked_independent_nonzero_vector() {
        let input = [1024, -2048, 3072, -4096];
        assert_eq!(
            rope_q30_coefficients(3, config(4)),
            Ok(vec![
                RopeQ30Coefficient {
                    cosine: -1_062_996_349,
                    sine: 151_526_455,
                },
                RopeQ30Coefficient {
                    cosine: 1_073_258_676,
                    sine: 32_207_423,
                },
            ])
        );
        assert_eq!(
            rope_q10(&input, &[3], [1, 1, 1, 4], config(4)),
            Ok(vec![-1447, -1924, -2897, -4156])
        );
    }

    #[test]
    fn leaves_non_rotary_tail_unchanged() {
        let input = [100, 200, 300, 400, 12_345, -23_456];
        let output = rope_q10(&input, &[1], [1, 1, 1, 6], config(4)).unwrap();
        assert_eq!(&output[4..], &input[4..]);
        assert_ne!(&output[..4], &input[..4]);
    }

    #[test]
    fn indexes_batches_heads_and_sequences_independently() {
        let mut input = Vec::new();
        for index in 0..32 {
            input.push(index * 37 - 500);
        }
        let positions = [0, 1, 2, 3];
        let output = rope_q10(&input, &positions, [2, 2, 2, 4], config(4)).unwrap();
        for batch in 0..2 {
            for head in 0..2 {
                for sequence in 0..2 {
                    let base = ((batch * 2 + head) * 2 + sequence) * 4;
                    let expected = rope_q10(
                        &input[base..base + 4],
                        &[positions[batch * 2 + sequence]],
                        [1, 1, 1, 4],
                        config(4),
                    )
                    .unwrap();
                    assert_eq!(&output[base..base + 4], expected);
                }
            }
        }
    }

    #[test]
    fn rejects_odd_and_oversized_rotary_dimensions() {
        assert_eq!(
            rope_q10(&[0; 4], &[0], [1, 1, 1, 4], config(3)),
            Err(RopeError::InvalidRotaryDimensions {
                rotary_dimensions: 3
            })
        );
        assert_eq!(
            rope_q10(&[0; 4], &[0], [1, 1, 1, 4], config(6)),
            Err(RopeError::RotaryDimensionsExceedHead {
                rotary_dimensions: 6,
                head_dim: 4,
            })
        );
    }

    #[test]
    fn rejects_invalid_theta_and_maximum_position() {
        for theta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut invalid = config(2);
            invalid.theta = theta;
            assert_eq!(
                rope_q10(&[0; 2], &[0], [1, 1, 1, 2], invalid),
                Err(RopeError::InvalidTheta)
            );
        }
        let mut invalid = config(2);
        invalid.maximum_position = 0;
        assert_eq!(
            rope_q10(&[0; 2], &[0], [1, 1, 1, 2], invalid),
            Err(RopeError::InvalidMaximumPosition)
        );
    }

    #[test]
    fn rejects_position_at_exclusive_bound() {
        assert_eq!(
            rope_q10(&[0; 2], &[128], [1, 1, 1, 2], config(2)),
            Err(RopeError::PositionOutOfRange {
                index: 0,
                position: 128,
                maximum_position: 128,
            })
        );
    }

    #[test]
    fn rejects_tensor_and_position_length_mismatches() {
        assert_eq!(
            rope_q10(&[0; 3], &[0], [1, 1, 1, 4], config(4)),
            Err(RopeError::LengthMismatch {
                tensor: "input",
                expected: 4,
                actual: 3,
            })
        );
        assert_eq!(
            rope_q10(&[0; 4], &[], [1, 1, 1, 4], config(4)),
            Err(RopeError::LengthMismatch {
                tensor: "positions",
                expected: 1,
                actual: 0,
            })
        );
    }

    #[test]
    fn rejects_output_outside_i16_without_saturation() {
        assert!(matches!(
            rope_q10(&[i16::MAX, 0, i16::MIN, 0], &[1], [1, 1, 1, 4], config(4)),
            Err(RopeError::OutputOutOfRange { .. })
        ));
    }

    #[test]
    fn repeated_execution_is_identical() {
        let input = [-3000, 2000, 1000, -4000, 17, 23, -31, 47];
        let first = rope_q10(&input, &[7, 11], [1, 1, 2, 4], config(4)).unwrap();
        for _ in 0..16 {
            assert_eq!(
                rope_q10(&input, &[7, 11], [1, 1, 2, 4], config(4)),
                Ok(first.clone())
            );
        }
    }
}
