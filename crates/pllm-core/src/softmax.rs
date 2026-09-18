use std::{error::Error, fmt};

use zeroize::{Zeroize, Zeroizing};

use crate::fixed_point::div_round_ties_even_i128;

pub const SOFTMAX_Q20_TO_Q30_PROFILE: &str =
    "pllm.numeric.softmax.q20_to_q30.libm_0_2_16_exact_sum.v1";
pub const SOFTMAX_Q30_ONE: i32 = 1_i32 << 30;
pub const SOFTMAX_Q30_MAX_ROW_LENGTH: usize = 32_768;
pub const SOFTMAX_Q30_MAX_ELEMENTS: usize = 16 * 1024 * 1024;
pub const SOFTMAX_Q30_MAX_DECODED_ERROR: f64 =
    (SOFTMAX_Q30_MAX_ROW_LENGTH as f64 + 2.0) / SOFTMAX_Q30_ONE as f64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SoftmaxQ30Policy {
    max_elements: usize,
    max_row_length: usize,
}

impl SoftmaxQ30Policy {
    pub fn new(max_elements: usize, max_row_length: usize) -> Result<Self, SoftmaxQ30Error> {
        if max_elements == 0 {
            return Err(SoftmaxQ30Error::InvalidPolicyLimit {
                limit: "max_elements",
            });
        }
        if max_row_length == 0 {
            return Err(SoftmaxQ30Error::InvalidPolicyLimit {
                limit: "max_row_length",
            });
        }
        if max_elements > SOFTMAX_Q30_MAX_ELEMENTS {
            return Err(SoftmaxQ30Error::PolicyLimitExceedsHardCap {
                limit: "max_elements",
                value: max_elements,
                maximum: SOFTMAX_Q30_MAX_ELEMENTS,
            });
        }
        if max_row_length > SOFTMAX_Q30_MAX_ROW_LENGTH {
            return Err(SoftmaxQ30Error::PolicyLimitExceedsHardCap {
                limit: "max_row_length",
                value: max_row_length,
                maximum: SOFTMAX_Q30_MAX_ROW_LENGTH,
            });
        }
        Ok(Self {
            max_elements,
            max_row_length,
        })
    }

    pub const fn max_elements(self) -> usize {
        self.max_elements
    }

    pub const fn max_row_length(self) -> usize {
        self.max_row_length
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoftmaxQ30Error {
    InvalidDimension {
        axis: &'static str,
    },
    ShapeOverflow,
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidPolicyLimit {
        limit: &'static str,
    },
    PolicyLimitExceedsHardCap {
        limit: &'static str,
        value: usize,
        maximum: usize,
    },
    ResourceLimitExceeded {
        elements: usize,
        maximum: usize,
    },
    RowLengthLimitExceeded {
        row_length: usize,
        maximum: usize,
    },
    NonFiniteExponential,
    ArithmeticOverflow,
    AllocationFailed,
}

impl fmt::Display for SoftmaxQ30Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { axis } => {
                write!(formatter, "softmax {axis} dimension must be positive")
            }
            Self::ShapeOverflow => write!(formatter, "softmax shape arithmetic overflowed"),
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "softmax {tensor} length must be {expected}, received {actual}"
            ),
            Self::InvalidPolicyLimit { limit } => {
                write!(formatter, "softmax policy {limit} must be positive")
            }
            Self::PolicyLimitExceedsHardCap {
                limit,
                value,
                maximum,
            } => write!(
                formatter,
                "softmax policy {limit} {value} exceeds hard cap {maximum}"
            ),
            Self::ResourceLimitExceeded { elements, maximum } => write!(
                formatter,
                "softmax elements {elements} exceed maximum {maximum}"
            ),
            Self::RowLengthLimitExceeded {
                row_length,
                maximum,
            } => write!(
                formatter,
                "softmax row length {row_length} exceeds maximum {maximum}"
            ),
            Self::NonFiniteExponential => {
                write!(formatter, "softmax exponential is not finite")
            }
            Self::ArithmeticOverflow => write!(formatter, "softmax arithmetic overflowed"),
            Self::AllocationFailed => write!(formatter, "softmax output allocation failed"),
        }
    }
}

impl Error for SoftmaxQ30Error {}

pub struct SoftmaxProbabilitiesQ30 {
    probabilities: Vec<i32>,
    shape: [usize; 4],
}

impl SoftmaxProbabilitiesQ30 {
    pub fn probabilities(&self) -> &[i32] {
        &self.probabilities
    }

    pub const fn shape(&self) -> [usize; 4] {
        self.shape
    }
}

impl Drop for SoftmaxProbabilitiesQ30 {
    fn drop(&mut self) {
        self.probabilities.zeroize();
    }
}

pub fn softmax_q20_to_q30(
    scores: &[i64],
    allowed: &[bool],
    shape: [usize; 4],
    policy: SoftmaxQ30Policy,
) -> Result<SoftmaxProbabilitiesQ30, SoftmaxQ30Error> {
    const AXIS_NAMES: [&str; 4] = ["batch", "heads", "query", "key"];
    for (index, &dimension) in shape.iter().enumerate() {
        if dimension == 0 {
            return Err(SoftmaxQ30Error::InvalidDimension {
                axis: AXIS_NAMES[index],
            });
        }
    }
    let elements = shape
        .iter()
        .try_fold(1_usize, |accumulator, &dimension| {
            accumulator.checked_mul(dimension)
        })
        .ok_or(SoftmaxQ30Error::ShapeOverflow)?;
    if scores.len() != elements {
        return Err(SoftmaxQ30Error::LengthMismatch {
            tensor: "scores",
            expected: elements,
            actual: scores.len(),
        });
    }
    if allowed.len() != elements {
        return Err(SoftmaxQ30Error::LengthMismatch {
            tensor: "allowed",
            expected: elements,
            actual: allowed.len(),
        });
    }
    if elements > SOFTMAX_Q30_MAX_ELEMENTS || elements > policy.max_elements() {
        return Err(SoftmaxQ30Error::ResourceLimitExceeded {
            elements,
            maximum: SOFTMAX_Q30_MAX_ELEMENTS.min(policy.max_elements()),
        });
    }
    let row_length = shape[3];
    if row_length > SOFTMAX_Q30_MAX_ROW_LENGTH || row_length > policy.max_row_length() {
        return Err(SoftmaxQ30Error::RowLengthLimitExceeded {
            row_length,
            maximum: SOFTMAX_Q30_MAX_ROW_LENGTH.min(policy.max_row_length()),
        });
    }
    let mut probabilities = Vec::new();
    probabilities
        .try_reserve_exact(elements)
        .map_err(|_| SoftmaxQ30Error::AllocationFailed)?;
    probabilities.resize(elements, 0_i32);
    let mut output = SoftmaxProbabilitiesQ30 {
        probabilities,
        shape,
    };
    let q20_scale = (1_u64 << 20) as f64;
    let q30_scale = (1_u64 << 30) as f64;
    let row_count = elements / row_length;
    for row in 0..row_count {
        let base = row * row_length;
        let row_scores = &scores[base..base + row_length];
        let row_allowed = &allowed[base..base + row_length];
        let mut maximum_index: Option<usize> = None;
        let mut maximum_score = 0_i64;
        for (index, &is_allowed) in row_allowed.iter().enumerate() {
            if !is_allowed {
                continue;
            }
            let score = row_scores[index];
            if maximum_index.is_none() || score > maximum_score {
                maximum_index = Some(index);
                maximum_score = score;
            }
        }
        let Some(maximum_index) = maximum_index else {
            continue;
        };
        let mut weights = Zeroizing::new(Vec::<u64>::new());
        weights
            .try_reserve_exact(row_length)
            .map_err(|_| SoftmaxQ30Error::AllocationFailed)?;
        weights.resize(row_length, 0_u64);
        let mut weight_sum = 0_u128;
        for (index, &is_allowed) in row_allowed.iter().enumerate() {
            if !is_allowed {
                continue;
            }
            let delta = i128::from(row_scores[index]) - i128::from(maximum_score);
            let decoded = delta as f64 / q20_scale;
            let exponential = libm::exp(decoded);
            if !exponential.is_finite() || exponential < 0.0 {
                return Err(SoftmaxQ30Error::NonFiniteExponential);
            }
            let scaled = exponential * q30_scale;
            if !scaled.is_finite() {
                return Err(SoftmaxQ30Error::NonFiniteExponential);
            }
            let weight = scaled.round_ties_even() as u64;
            weights[index] = weight;
            weight_sum = weight_sum
                .checked_add(u128::from(weight))
                .ok_or(SoftmaxQ30Error::ArithmeticOverflow)?;
        }
        let denominator =
            i128::try_from(weight_sum).map_err(|_| SoftmaxQ30Error::ArithmeticOverflow)?;
        let mut probability_sum = 0_i64;
        for (index, &weight) in weights.iter().enumerate() {
            if weight == 0 {
                continue;
            }
            let raw = div_round_ties_even_i128(i128::from(weight) * (1_i128 << 30), denominator);
            let probability =
                i32::try_from(raw).map_err(|_| SoftmaxQ30Error::ArithmeticOverflow)?;
            output.probabilities[base + index] = probability;
            probability_sum = probability_sum
                .checked_add(i64::from(probability))
                .ok_or(SoftmaxQ30Error::ArithmeticOverflow)?;
        }
        let residual = i64::from(SOFTMAX_Q30_ONE) - probability_sum;
        let corrected = i64::from(output.probabilities[base + maximum_index])
            .checked_add(residual)
            .ok_or(SoftmaxQ30Error::ArithmeticOverflow)?;
        if !(0..=i64::from(SOFTMAX_Q30_ONE)).contains(&corrected) {
            return Err(SoftmaxQ30Error::ArithmeticOverflow);
        }
        output.probabilities[base + maximum_index] =
            i32::try_from(corrected).map_err(|_| SoftmaxQ30Error::ArithmeticOverflow)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SoftmaxQ30Policy {
        SoftmaxQ30Policy::new(SOFTMAX_Q30_MAX_ELEMENTS, SOFTMAX_Q30_MAX_ROW_LENGTH).unwrap()
    }

    #[test]
    fn splits_two_equal_scores_into_exact_halves() {
        let output = softmax_q20_to_q30(&[0, 0], &[true, true], [1, 1, 1, 2], policy()).unwrap();
        assert_eq!(output.shape(), [1, 1, 1, 2]);
        assert_eq!(
            output.probabilities(),
            &[SOFTMAX_Q30_ONE / 2, SOFTMAX_Q30_ONE / 2]
        );
        assert_eq!(
            output
                .probabilities()
                .iter()
                .copied()
                .map(i64::from)
                .sum::<i64>(),
            i64::from(SOFTMAX_Q30_ONE)
        );
    }

    #[test]
    fn masks_extreme_disallowed_scores_without_reading_them() {
        let scores = [i64::MIN, 0, i64::MAX];
        let allowed = [false, true, false];
        let output = softmax_q20_to_q30(&scores, &allowed, [1, 1, 1, 3], policy()).unwrap();
        assert_eq!(output.probabilities(), &[0, SOFTMAX_Q30_ONE, 0]);
        let different =
            softmax_q20_to_q30(&[i64::MAX, 0, i64::MIN], &allowed, [1, 1, 1, 3], policy()).unwrap();
        assert_eq!(different.probabilities(), output.probabilities());
    }

    #[test]
    fn inactive_rows_stay_zero_while_active_rows_sum_exactly() {
        let shape = [1, 3, 1, 4];
        let mut scores = vec![0_i64; 12];
        scores[..4].copy_from_slice(&[1 << 20, 0, -(1 << 20), 1 << 19]);
        scores[8..].copy_from_slice(&[5 << 20, 4 << 20, 3 << 20, 2 << 20]);
        let mut allowed = vec![false; 12];
        allowed[..3].copy_from_slice(&[true, true, true]);
        allowed[8..].copy_from_slice(&[true, true, true, false]);
        let output = softmax_q20_to_q30(&scores, &allowed, shape, policy()).unwrap();
        let probabilities = output.probabilities();
        assert_eq!(
            probabilities[..4]
                .iter()
                .copied()
                .map(i64::from)
                .sum::<i64>(),
            i64::from(SOFTMAX_Q30_ONE)
        );
        assert_eq!(&probabilities[4..8], &[0, 0, 0, 0]);
        assert_eq!(
            probabilities[8..]
                .iter()
                .copied()
                .map(i64::from)
                .sum::<i64>(),
            i64::from(SOFTMAX_Q30_ONE)
        );
        assert_eq!(probabilities[3], 0);
        assert_eq!(probabilities[11], 0);
        assert!(probabilities.iter().all(|&value| value >= 0));
    }

    #[test]
    fn matches_reference_f64_softmax_within_the_declared_bound() {
        let mut maximum_error = 0.0_f64;
        for row_length in [1_usize, 2, 3, 16, 257, 1024] {
            let shape = [1, 1, 1, row_length];
            let scores: Vec<i64> = (0..row_length)
                .map(|index| {
                    let mixed = (index as i64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(row_length as i64);
                    (mixed % (97 << 20)) - (49 << 20)
                })
                .collect();
            let allowed = vec![true; row_length];
            let output = softmax_q20_to_q30(&scores, &allowed, shape, policy()).unwrap();
            let probabilities = output.probabilities();
            assert_eq!(
                probabilities.iter().copied().map(i64::from).sum::<i64>(),
                i64::from(SOFTMAX_Q30_ONE)
            );
            assert!(probabilities.iter().all(|&value| value >= 0));
            let decoded: Vec<f64> = scores
                .iter()
                .map(|&score| score as f64 / (1_u64 << 20) as f64)
                .collect();
            let reference_max = decoded.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let reference_weights: Vec<f64> = decoded
                .iter()
                .map(|value| (value - reference_max).exp())
                .collect();
            let reference_sum: f64 = reference_weights.iter().sum();
            let bound = (row_length as f64 + 2.0) / (1_u64 << 30) as f64;
            for (index, &probability) in probabilities.iter().enumerate() {
                let expected = reference_weights[index] / reference_sum;
                let actual = f64::from(probability) / (1_u64 << 30) as f64;
                maximum_error = maximum_error.max((actual - expected).abs());
                assert!((actual - expected).abs() <= bound);
            }
        }
        assert!(maximum_error <= SOFTMAX_Q30_MAX_DECODED_ERROR);
    }

    #[test]
    fn randomized_masked_rows_match_an_independent_softmax_bound() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for row_length in [2_usize, 3, 7, 31, 257, 1024] {
            for _ in 0..64 {
                let mut scores = Vec::with_capacity(row_length);
                let mut allowed = Vec::with_capacity(row_length);
                for _ in 0..row_length {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    scores.push(((state >> 16) % (64 << 20)) as i64 - (32 << 20));
                    allowed.push(state & 3 != 0);
                }
                allowed[0] = true;
                let output =
                    softmax_q20_to_q30(&scores, &allowed, [1, 1, 1, row_length], policy()).unwrap();
                let probabilities = output.probabilities();
                assert_eq!(
                    probabilities.iter().copied().map(i64::from).sum::<i64>(),
                    i64::from(SOFTMAX_Q30_ONE)
                );
                let maximum = scores
                    .iter()
                    .zip(&allowed)
                    .filter_map(|(&score, &is_allowed)| is_allowed.then_some(score))
                    .max()
                    .unwrap() as f64
                    / (1_u64 << 20) as f64;
                let weights: Vec<f64> = scores
                    .iter()
                    .zip(&allowed)
                    .map(|(&score, &is_allowed)| {
                        is_allowed
                            .then(|| (score as f64 / (1_u64 << 20) as f64 - maximum).exp())
                            .unwrap_or(0.0)
                    })
                    .collect();
                let denominator: f64 = weights.iter().sum();
                let bound = (row_length as f64 + 2.0) / (1_u64 << 30) as f64;
                for ((&probability, &weight), &is_allowed) in
                    probabilities.iter().zip(&weights).zip(&allowed)
                {
                    if is_allowed {
                        let actual = f64::from(probability) / (1_u64 << 30) as f64;
                        assert!((actual - weight / denominator).abs() <= bound);
                    } else {
                        assert_eq!(probability, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn handles_extreme_allowed_scores_without_overflow() {
        let output =
            softmax_q20_to_q30(&[i64::MIN, i64::MAX], &[true, true], [1, 1, 1, 2], policy())
                .unwrap();
        assert_eq!(
            output
                .probabilities()
                .iter()
                .copied()
                .map(i64::from)
                .sum::<i64>(),
            i64::from(SOFTMAX_Q30_ONE)
        );
        assert_eq!(output.probabilities()[1], SOFTMAX_Q30_ONE);
    }

    #[test]
    fn rejects_invalid_dimensions_lengths_and_limits() {
        let scores = [0_i64; 4];
        let allowed = [true; 4];
        for (index, shape) in [[0, 1, 1, 4], [1, 0, 1, 4], [1, 1, 0, 4], [1, 1, 1, 0]]
            .iter()
            .enumerate()
        {
            assert!(matches!(
                softmax_q20_to_q30(&scores, &allowed, *shape, policy()),
                Err(SoftmaxQ30Error::InvalidDimension { axis })
                    if axis == ["batch", "heads", "query", "key"][index]
            ));
        }
        assert!(matches!(
            softmax_q20_to_q30(&scores[..3], &allowed, [1, 1, 1, 4], policy()),
            Err(SoftmaxQ30Error::LengthMismatch {
                tensor: "scores",
                ..
            })
        ));
        assert!(matches!(
            softmax_q20_to_q30(&scores, &allowed[..3], [1, 1, 1, 4], policy()),
            Err(SoftmaxQ30Error::LengthMismatch {
                tensor: "allowed",
                ..
            })
        ));
        assert!(softmax_q20_to_q30(
            &[usize::MAX as i64; 0],
            &[],
            [usize::MAX, 1, 1, usize::MAX],
            policy()
        )
        .is_err());
        assert_eq!(
            SoftmaxQ30Policy::new(0, 4),
            Err(SoftmaxQ30Error::InvalidPolicyLimit {
                limit: "max_elements"
            })
        );
        assert_eq!(
            SoftmaxQ30Policy::new(4, 0),
            Err(SoftmaxQ30Error::InvalidPolicyLimit {
                limit: "max_row_length"
            })
        );
        assert!(matches!(
            SoftmaxQ30Policy::new(SOFTMAX_Q30_MAX_ELEMENTS + 1, 4),
            Err(SoftmaxQ30Error::PolicyLimitExceedsHardCap {
                limit: "max_elements",
                ..
            })
        ));
        assert!(matches!(
            SoftmaxQ30Policy::new(4, SOFTMAX_Q30_MAX_ROW_LENGTH + 1),
            Err(SoftmaxQ30Error::PolicyLimitExceedsHardCap {
                limit: "max_row_length",
                ..
            })
        ));
        let tight_elements = SoftmaxQ30Policy::new(3, 4).unwrap();
        assert!(matches!(
            softmax_q20_to_q30(&scores, &allowed, [1, 1, 1, 4], tight_elements),
            Err(SoftmaxQ30Error::ResourceLimitExceeded { elements: 4, .. })
        ));
        let tight_row = SoftmaxQ30Policy::new(16, 3).unwrap();
        assert!(matches!(
            softmax_q20_to_q30(&scores, &allowed, [1, 1, 1, 4], tight_row),
            Err(SoftmaxQ30Error::RowLengthLimitExceeded { row_length: 4, .. })
        ));
    }

    #[test]
    fn error_paths_return_no_partial_output() {
        assert!(softmax_q20_to_q30(&[0], &[true], [1, 1, 1, 2], policy()).is_err());
        assert!(softmax_q20_to_q30(&[0, 0], &[true], [1, 1, 1, 2], policy()).is_err());
    }
}
