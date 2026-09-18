use std::{error::Error, fmt};

use serde::Serialize;
use zeroize::Zeroize;

use crate::{fixed_point::div_round_ties_even_i128, SOFTMAX_Q30_ONE};

pub const ATTENTION_VALUE_Q30_Q10_PROFILE: &str =
    "pllm.numeric.attention_value.q30_probability_q10_value_q10_output.v1";
pub const ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS: usize = 16 * 1024 * 1024;
pub const ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES: u64 = 16 * 1024 * 1024 * 1024;
pub const ATTENTION_VALUE_Q10_MAX_ROUNDING_ERROR_RAW: f64 = 0.5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionValueQ10Policy {
    max_output_elements: usize,
    max_multiply_accumulates: u64,
}

impl AttentionValueQ10Policy {
    pub fn new(
        max_output_elements: usize,
        max_multiply_accumulates: u64,
    ) -> Result<Self, AttentionValueQ10Error> {
        if max_output_elements == 0 {
            return Err(AttentionValueQ10Error::InvalidPolicyLimit {
                limit: "max_output_elements",
            });
        }
        if max_multiply_accumulates == 0 {
            return Err(AttentionValueQ10Error::InvalidPolicyLimit {
                limit: "max_multiply_accumulates",
            });
        }
        if max_output_elements > ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS {
            return Err(AttentionValueQ10Error::PolicyLimitExceedsHardCap {
                limit: "max_output_elements",
                value: max_output_elements as u64,
                maximum: ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS as u64,
            });
        }
        if max_multiply_accumulates > ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES {
            return Err(AttentionValueQ10Error::PolicyLimitExceedsHardCap {
                limit: "max_multiply_accumulates",
                value: max_multiply_accumulates,
                maximum: ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES,
            });
        }
        Ok(Self {
            max_output_elements,
            max_multiply_accumulates,
        })
    }

    pub const fn max_output_elements(self) -> usize {
        self.max_output_elements
    }

    pub const fn max_multiply_accumulates(self) -> u64 {
        self.max_multiply_accumulates
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionValueQ10Layout {
    GroupedQuery,
    PerQueryHeadWindow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionValueQ10Error {
    InvalidDimension {
        tensor: &'static str,
        axis: &'static str,
    },
    InvalidValueRank {
        rank: usize,
    },
    ValueShapeMismatch {
        axis: &'static str,
        expected: usize,
        actual: usize,
    },
    QueryHeadsNotDivisible {
        query_heads: usize,
        value_heads: usize,
    },
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    ProbabilityOutOfRange {
        index: usize,
        value: i32,
    },
    ProbabilityRowSumInvalid {
        row: usize,
        sum: i64,
    },
    InvalidPolicyLimit {
        limit: &'static str,
    },
    PolicyLimitExceedsHardCap {
        limit: &'static str,
        value: u64,
        maximum: u64,
    },
    OutputElementsLimitExceeded {
        elements: usize,
        maximum: usize,
    },
    MultiplyAccumulatesLimitExceeded {
        multiply_accumulates: u64,
        maximum: u64,
    },
    ShapeOverflow,
    ArithmeticOverflow,
    AllocationFailed,
}

impl fmt::Display for AttentionValueQ10Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { tensor, axis } => {
                write!(formatter, "attention-value {tensor} {axis} dimension must be positive")
            }
            Self::InvalidValueRank { rank } => write!(
                formatter,
                "attention-value values must have rank four or five, received {rank}"
            ),
            Self::ValueShapeMismatch {
                axis,
                expected,
                actual,
            } => write!(
                formatter,
                "attention-value values {axis} dimension must be {expected}, received {actual}"
            ),
            Self::QueryHeadsNotDivisible {
                query_heads,
                value_heads,
            } => write!(
                formatter,
                "attention-value query heads {query_heads} are not divisible by value heads {value_heads}"
            ),
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "attention-value {tensor} length must be {expected}, received {actual}"
            ),
            Self::ProbabilityOutOfRange { index, value } => write!(
                formatter,
                "attention-value probability {value} at index {index} is outside Q30 range"
            ),
            Self::ProbabilityRowSumInvalid { row, sum } => write!(
                formatter,
                "attention-value probability row {row} sums to {sum}, not zero or Q30"
            ),
            Self::InvalidPolicyLimit { limit } => {
                write!(formatter, "attention-value policy {limit} must be positive")
            }
            Self::PolicyLimitExceedsHardCap {
                limit,
                value,
                maximum,
            } => write!(
                formatter,
                "attention-value policy {limit} {value} exceeds hard cap {maximum}"
            ),
            Self::OutputElementsLimitExceeded { elements, maximum } => write!(
                formatter,
                "attention-value output elements {elements} exceed maximum {maximum}"
            ),
            Self::MultiplyAccumulatesLimitExceeded {
                multiply_accumulates,
                maximum,
            } => write!(
                formatter,
                "attention-value multiply-accumulates {multiply_accumulates} exceed maximum {maximum}"
            ),
            Self::ShapeOverflow => write!(formatter, "attention-value shape arithmetic overflowed"),
            Self::ArithmeticOverflow => {
                write!(formatter, "attention-value arithmetic overflowed")
            }
            Self::AllocationFailed => {
                write!(formatter, "attention-value output allocation failed")
            }
        }
    }
}

impl Error for AttentionValueQ10Error {}

pub struct AttentionValuesQ10 {
    values: Vec<i16>,
    shape: [usize; 4],
    layout: AttentionValueQ10Layout,
}

impl AttentionValuesQ10 {
    pub fn values(&self) -> &[i16] {
        &self.values
    }

    pub const fn shape(&self) -> [usize; 4] {
        self.shape
    }

    pub const fn layout(&self) -> AttentionValueQ10Layout {
        self.layout
    }
}

impl Drop for AttentionValuesQ10 {
    fn drop(&mut self) {
        self.values.zeroize();
    }
}

fn checked_product(shape: &[usize]) -> Result<usize, AttentionValueQ10Error> {
    shape
        .iter()
        .try_fold(1_usize, |accumulator, &dimension| {
            accumulator.checked_mul(dimension)
        })
        .ok_or(AttentionValueQ10Error::ShapeOverflow)
}

pub fn attention_values_q10(
    probabilities: &[i32],
    probability_shape: [usize; 4],
    values: &[i16],
    value_shape: &[usize],
    policy: AttentionValueQ10Policy,
) -> Result<AttentionValuesQ10, AttentionValueQ10Error> {
    const PROBABILITY_AXES: [&str; 4] = ["batch", "heads", "query", "key"];
    const VALUE_AXES: [&str; 5] = ["batch", "heads", "query", "key", "depth"];
    for (index, &dimension) in probability_shape.iter().enumerate() {
        if dimension == 0 {
            return Err(AttentionValueQ10Error::InvalidDimension {
                tensor: "probabilities",
                axis: PROBABILITY_AXES[index],
            });
        }
    }
    if !(4..=5).contains(&value_shape.len()) {
        return Err(AttentionValueQ10Error::InvalidValueRank {
            rank: value_shape.len(),
        });
    }
    for (index, &dimension) in value_shape.iter().enumerate() {
        if dimension == 0 {
            return Err(AttentionValueQ10Error::InvalidDimension {
                tensor: "values",
                axis: VALUE_AXES[index],
            });
        }
    }
    let (batch, query_heads, queries, keys) = (
        probability_shape[0],
        probability_shape[1],
        probability_shape[2],
        probability_shape[3],
    );
    let (layout, value_heads, depth) = if value_shape.len() == 4 {
        let value_heads = value_shape[1];
        if value_shape[0] != batch {
            return Err(AttentionValueQ10Error::ValueShapeMismatch {
                axis: "batch",
                expected: batch,
                actual: value_shape[0],
            });
        }
        if value_shape[2] != keys {
            return Err(AttentionValueQ10Error::ValueShapeMismatch {
                axis: "key",
                expected: keys,
                actual: value_shape[2],
            });
        }
        if query_heads % value_heads != 0 {
            return Err(AttentionValueQ10Error::QueryHeadsNotDivisible {
                query_heads,
                value_heads,
            });
        }
        (
            AttentionValueQ10Layout::GroupedQuery,
            value_heads,
            value_shape[3],
        )
    } else {
        for (index, &dimension) in value_shape[..4].iter().enumerate() {
            if dimension != probability_shape[index] {
                return Err(AttentionValueQ10Error::ValueShapeMismatch {
                    axis: VALUE_AXES[index],
                    expected: probability_shape[index],
                    actual: dimension,
                });
            }
        }
        (
            AttentionValueQ10Layout::PerQueryHeadWindow,
            query_heads,
            value_shape[4],
        )
    };
    let probability_elements = checked_product(&probability_shape)?;
    if probabilities.len() != probability_elements {
        return Err(AttentionValueQ10Error::LengthMismatch {
            tensor: "probabilities",
            expected: probability_elements,
            actual: probabilities.len(),
        });
    }
    let value_elements = checked_product(value_shape)?;
    if values.len() != value_elements {
        return Err(AttentionValueQ10Error::LengthMismatch {
            tensor: "values",
            expected: value_elements,
            actual: values.len(),
        });
    }
    let output_elements = batch
        .checked_mul(query_heads)
        .and_then(|product| product.checked_mul(queries))
        .and_then(|product| product.checked_mul(depth))
        .ok_or(AttentionValueQ10Error::ShapeOverflow)?;
    let output_elements_u64 =
        u64::try_from(output_elements).map_err(|_| AttentionValueQ10Error::ShapeOverflow)?;
    let keys_u64 = u64::try_from(keys).map_err(|_| AttentionValueQ10Error::ShapeOverflow)?;
    let multiply_accumulates = output_elements_u64
        .checked_mul(keys_u64)
        .ok_or(AttentionValueQ10Error::ShapeOverflow)?;
    if output_elements > policy.max_output_elements() {
        return Err(AttentionValueQ10Error::OutputElementsLimitExceeded {
            elements: output_elements,
            maximum: policy.max_output_elements(),
        });
    }
    if output_elements > ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS {
        return Err(AttentionValueQ10Error::OutputElementsLimitExceeded {
            elements: output_elements,
            maximum: ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS,
        });
    }
    if multiply_accumulates > policy.max_multiply_accumulates() {
        return Err(AttentionValueQ10Error::MultiplyAccumulatesLimitExceeded {
            multiply_accumulates,
            maximum: policy.max_multiply_accumulates(),
        });
    }
    if multiply_accumulates > ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES {
        return Err(AttentionValueQ10Error::MultiplyAccumulatesLimitExceeded {
            multiply_accumulates,
            maximum: ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES,
        });
    }
    let row_count = batch
        .checked_mul(query_heads)
        .and_then(|product| product.checked_mul(queries))
        .ok_or(AttentionValueQ10Error::ShapeOverflow)?;
    for row in 0..row_count {
        let base = row * keys;
        let mut row_sum = 0_i64;
        for index in 0..keys {
            let probability = probabilities[base + index];
            if !(0..=SOFTMAX_Q30_ONE).contains(&probability) {
                return Err(AttentionValueQ10Error::ProbabilityOutOfRange {
                    index: base + index,
                    value: probability,
                });
            }
            row_sum = row_sum
                .checked_add(i64::from(probability))
                .ok_or(AttentionValueQ10Error::ArithmeticOverflow)?;
        }
        if row_sum != 0 && row_sum != i64::from(SOFTMAX_Q30_ONE) {
            return Err(AttentionValueQ10Error::ProbabilityRowSumInvalid { row, sum: row_sum });
        }
    }
    let mut output_values = Vec::new();
    output_values
        .try_reserve_exact(output_elements)
        .map_err(|_| AttentionValueQ10Error::AllocationFailed)?;
    output_values.resize(output_elements, 0_i16);
    let mut output = AttentionValuesQ10 {
        values: output_values,
        shape: [batch, query_heads, queries, depth],
        layout,
    };
    let group_size = query_heads / value_heads;
    for row in 0..row_count {
        let probability_base = row * keys;
        if probabilities[probability_base..probability_base + keys]
            .iter()
            .all(|&probability| probability == 0)
        {
            continue;
        }
        let batch_index = row / (query_heads * queries);
        let value_head = (row / queries) % query_heads / group_size;
        for feature in 0..depth {
            let mut accumulate = 0_i128;
            for key in 0..keys {
                let probability = probabilities[probability_base + key];
                if probability == 0 {
                    continue;
                }
                let value_index = match layout {
                    AttentionValueQ10Layout::GroupedQuery => {
                        ((batch_index * value_heads + value_head) * keys + key) * depth + feature
                    }
                    AttentionValueQ10Layout::PerQueryHeadWindow => {
                        (row * keys + key) * depth + feature
                    }
                };
                accumulate += i128::from(probability) * i128::from(values[value_index]);
            }
            let quantized = div_round_ties_even_i128(accumulate, 1_i128 << 30);
            output.values[row * depth + feature] =
                i16::try_from(quantized).map_err(|_| AttentionValueQ10Error::ArithmeticOverflow)?;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::softmax_q20_to_q30;

    fn policy() -> AttentionValueQ10Policy {
        AttentionValueQ10Policy::new(
            ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS,
            ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES,
        )
        .unwrap()
    }

    fn half_q30() -> i32 {
        SOFTMAX_Q30_ONE / 2
    }

    #[test]
    fn contracts_grouped_query_rank4_values() {
        let probability_shape = [1, 2, 1, 2];
        let probabilities = [half_q30(), half_q30(), 0, SOFTMAX_Q30_ONE];
        let value_shape = [1, 1, 2, 2];
        let values = [4_i16, -2, 6, 8];
        let output = attention_values_q10(
            &probabilities,
            probability_shape,
            &values,
            &value_shape,
            policy(),
        )
        .unwrap();
        assert_eq!(output.layout(), AttentionValueQ10Layout::GroupedQuery);
        assert_eq!(output.shape(), [1, 2, 1, 2]);
        assert_eq!(output.values(), &[5, 3, 6, 8]);
    }

    #[test]
    fn contracts_per_query_head_rank5_windows() {
        let probability_shape = [1, 1, 2, 2];
        let probabilities = [SOFTMAX_Q30_ONE, 0, half_q30(), half_q30()];
        let value_shape = [1, 1, 2, 2, 2];
        let values = [10_i16, 20, 30, 40, -4, -8, 2, 6];
        let output = attention_values_q10(
            &probabilities,
            probability_shape,
            &values,
            &value_shape,
            policy(),
        )
        .unwrap();
        assert_eq!(output.layout(), AttentionValueQ10Layout::PerQueryHeadWindow);
        assert_eq!(output.shape(), [1, 1, 2, 2]);
        assert_eq!(output.values(), &[10, 20, -1, -1]);
    }

    #[test]
    fn inactive_rows_output_zero_without_reading_extreme_values() {
        let probability_shape = [1, 1, 2, 2];
        let probabilities = [0, 0, SOFTMAX_Q30_ONE, 0];
        let value_shape = [1, 1, 2, 2];
        let values = [i16::MIN, i16::MAX, i16::MAX, i16::MIN];
        let output = attention_values_q10(
            &probabilities,
            probability_shape,
            &values,
            &value_shape,
            policy(),
        )
        .unwrap();
        assert_eq!(&output.values()[..2], &[0, 0]);
        assert_eq!(&output.values()[2..], &[i16::MIN, i16::MAX]);
    }

    #[test]
    fn handles_extreme_values_with_exact_ties_to_even() {
        let probability_shape = [1, 1, 1, 2];
        let value_shape = [1, 1, 2, 1];
        let one_hot = attention_values_q10(
            &[SOFTMAX_Q30_ONE, 0],
            probability_shape,
            &[i16::MIN, i16::MAX],
            &value_shape,
            policy(),
        )
        .unwrap();
        assert_eq!(one_hot.values(), &[i16::MIN]);
        let half = SOFTMAX_Q30_ONE / 2;
        let convex = attention_values_q10(
            &[half, half],
            probability_shape,
            &[i16::MAX, i16::MIN],
            &value_shape,
            policy(),
        )
        .unwrap();
        assert_eq!(convex.values(), &[0]);
        let third = SOFTMAX_Q30_ONE / 3;
        let weights = [third, third, SOFTMAX_Q30_ONE - 2 * third];
        let mixed = attention_values_q10(
            &weights,
            [1, 1, 1, 3],
            &[1_i16, 2, 3],
            &[1, 1, 3, 1],
            policy(),
        )
        .unwrap();
        let numerator =
            i128::from(weights[0]) + 2 * i128::from(weights[1]) + 3 * i128::from(weights[2]);
        let expected = div_round_ties_even_i128(numerator, 1_i128 << 30) as i16;
        assert_eq!(mixed.values(), &[expected]);
    }

    #[test]
    fn rejects_invalid_probabilities_shapes_and_limits() {
        let values = [0_i16; 2];
        let value_shape = [1, 1, 2, 1];
        assert!(matches!(
            attention_values_q10(
                &[-1, SOFTMAX_Q30_ONE + 1],
                [1, 1, 1, 2],
                &values,
                &value_shape,
                policy()
            ),
            Err(AttentionValueQ10Error::ProbabilityOutOfRange { .. })
        ));
        assert!(matches!(
            attention_values_q10(&[1, 1], [1, 1, 1, 2], &values, &value_shape, policy()),
            Err(AttentionValueQ10Error::ProbabilityRowSumInvalid { .. })
        ));
        assert!(matches!(
            attention_values_q10(
                &[0, SOFTMAX_Q30_ONE],
                [1, 1, 1, 2],
                &values,
                &[1, 1, 2, 1, 1, 1],
                policy()
            ),
            Err(AttentionValueQ10Error::InvalidValueRank { rank: 6 })
        ));
        assert!(matches!(
            attention_values_q10(
                &[0, SOFTMAX_Q30_ONE],
                [1, 1, 1, 2],
                &values,
                &[0, 1, 2, 1],
                policy()
            ),
            Err(AttentionValueQ10Error::InvalidDimension {
                tensor: "values",
                ..
            })
        ));
        assert!(matches!(
            attention_values_q10(
                &[0, SOFTMAX_Q30_ONE],
                [0, 1, 1, 2],
                &values,
                &value_shape,
                policy()
            ),
            Err(AttentionValueQ10Error::InvalidDimension {
                tensor: "probabilities",
                ..
            })
        ));
        assert!(matches!(
            attention_values_q10(
                &[0, SOFTMAX_Q30_ONE],
                [1, 3, 1, 2],
                &[0_i16; 6],
                &[1, 2, 2, 1],
                policy()
            ),
            Err(AttentionValueQ10Error::QueryHeadsNotDivisible { .. })
        ));
        assert!(matches!(
            attention_values_q10(
                &[0, SOFTMAX_Q30_ONE],
                [1, 1, 1, 2],
                &values,
                &[2, 1, 2, 1],
                policy()
            ),
            Err(AttentionValueQ10Error::ValueShapeMismatch { axis: "batch", .. })
        ));
        assert!(matches!(
            attention_values_q10(
                &[0, SOFTMAX_Q30_ONE],
                [1, 1, 1, 2],
                &values[..1],
                &value_shape,
                policy()
            ),
            Err(AttentionValueQ10Error::LengthMismatch {
                tensor: "values",
                ..
            })
        ));
        assert!(matches!(
            attention_values_q10(
                &[0, SOFTMAX_Q30_ONE],
                [1, 1, 1, 2],
                &[0_i16; 3],
                &[1, 1, 1, 3, 1],
                policy()
            ),
            Err(AttentionValueQ10Error::ValueShapeMismatch { axis: "key", .. })
        ));
        assert_eq!(
            AttentionValueQ10Policy::new(0, 4),
            Err(AttentionValueQ10Error::InvalidPolicyLimit {
                limit: "max_output_elements"
            })
        );
        assert_eq!(
            AttentionValueQ10Policy::new(4, 0),
            Err(AttentionValueQ10Error::InvalidPolicyLimit {
                limit: "max_multiply_accumulates"
            })
        );
        assert!(matches!(
            AttentionValueQ10Policy::new(ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS + 1, 4),
            Err(AttentionValueQ10Error::PolicyLimitExceedsHardCap {
                limit: "max_output_elements",
                ..
            })
        ));
        assert!(matches!(
            AttentionValueQ10Policy::new(4, ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES + 1),
            Err(AttentionValueQ10Error::PolicyLimitExceedsHardCap {
                limit: "max_multiply_accumulates",
                ..
            })
        ));
        let tight = AttentionValueQ10Policy::new(1, 16).unwrap();
        assert!(matches!(
            attention_values_q10(
                &[0, SOFTMAX_Q30_ONE],
                [1, 1, 1, 2],
                &[0_i16; 4],
                &[1, 1, 2, 2],
                tight
            ),
            Err(AttentionValueQ10Error::OutputElementsLimitExceeded { .. })
        ));
        let tight_work = AttentionValueQ10Policy::new(16, 1).unwrap();
        assert!(matches!(
            attention_values_q10(
                &[0, SOFTMAX_Q30_ONE],
                [1, 1, 1, 2],
                &values,
                &value_shape,
                tight_work
            ),
            Err(AttentionValueQ10Error::MultiplyAccumulatesLimitExceeded { .. })
        ));
    }

    fn reference_ties_even(numerator: i128, denominator: i128) -> i128 {
        let quotient = numerator.div_euclid(denominator);
        let remainder = numerator.rem_euclid(denominator);
        match remainder.cmp(&(denominator - remainder)) {
            std::cmp::Ordering::Less => quotient,
            std::cmp::Ordering::Greater => quotient + 1,
            std::cmp::Ordering::Equal if quotient % 2 == 0 => quotient,
            std::cmp::Ordering::Equal => quotient + 1,
        }
    }

    #[test]
    fn randomized_grouped_and_window_layouts_match_independent_indexing() {
        let mut state = 0xd1b5_4a32_d192_ed03_u64;
        for keys in [1_usize, 2, 7, 31] {
            let probability_shape = [2, 4, 3, keys];
            let probability_elements: usize = probability_shape.iter().product();
            let mut scores = Vec::with_capacity(probability_elements);
            let mut allowed = Vec::with_capacity(probability_elements);
            for row in 0..24 {
                for key in 0..keys {
                    state = state
                        .wrapping_mul(2_862_933_555_777_941_757)
                        .wrapping_add(3_037_000_493);
                    scores.push(((state >> 20) % (24 << 20)) as i64 - (12 << 20));
                    allowed.push(key == 0 || (state.wrapping_add(row) & 3 != 0));
                }
            }
            let probabilities = crate::softmax_q20_to_q30(
                &scores,
                &allowed,
                probability_shape,
                crate::SoftmaxQ30Policy::new(
                    crate::SOFTMAX_Q30_MAX_ELEMENTS,
                    crate::SOFTMAX_Q30_MAX_ROW_LENGTH,
                )
                .unwrap(),
            )
            .unwrap();
            for layout in [
                AttentionValueQ10Layout::GroupedQuery,
                AttentionValueQ10Layout::PerQueryHeadWindow,
            ] {
                let value_shape = match layout {
                    AttentionValueQ10Layout::GroupedQuery => vec![2, 2, keys, 5],
                    AttentionValueQ10Layout::PerQueryHeadWindow => vec![2, 4, 3, keys, 5],
                };
                let value_elements: usize = value_shape.iter().product();
                let mut values = Vec::with_capacity(value_elements);
                for _ in 0..value_elements {
                    state = state
                        .wrapping_mul(2_862_933_555_777_941_757)
                        .wrapping_add(3_037_000_493);
                    values.push(((state >> 32) as i16).wrapping_rem(2048));
                }
                let output = attention_values_q10(
                    probabilities.probabilities(),
                    probability_shape,
                    &values,
                    &value_shape,
                    policy(),
                )
                .unwrap();
                assert_eq!(output.layout(), layout);
                for row in 0..24 {
                    let batch = row / 12;
                    let head = row / 3 % 4;
                    for feature in 0..5 {
                        let mut numerator = 0_i128;
                        for key in 0..keys {
                            let probability = probabilities.probabilities()[row * keys + key];
                            let value_index = match layout {
                                AttentionValueQ10Layout::GroupedQuery => {
                                    ((batch * 2 + head / 2) * keys + key) * 5 + feature
                                }
                                AttentionValueQ10Layout::PerQueryHeadWindow => {
                                    (row * keys + key) * 5 + feature
                                }
                            };
                            numerator += i128::from(probability) * i128::from(values[value_index]);
                        }
                        let expected = reference_ties_even(numerator, 1_i128 << 30) as i16;
                        assert_eq!(output.values()[row * 5 + feature], expected);
                    }
                }
            }
        }
    }

    #[test]
    fn reports_output_shape_and_is_deterministic() {
        let probabilities = [half_q30(), half_q30()];
        let output = attention_values_q10(
            &probabilities,
            [1, 1, 1, 2],
            &[3_i16, 5],
            &[1, 1, 2, 1],
            policy(),
        )
        .unwrap();
        assert_eq!(output.shape(), [1, 1, 1, 1]);
        let again = attention_values_q10(
            &probabilities,
            [1, 1, 1, 2],
            &[3_i16, 5],
            &[1, 1, 2, 1],
            policy(),
        )
        .unwrap();
        assert_eq!(output.values(), again.values());
        assert_eq!(output.values(), &[4]);
    }

    #[test]
    fn composes_with_softmax_q30_outputs() {
        let softmax = softmax_q20_to_q30(
            &[1 << 20, 0, -(1 << 20)],
            &[true, true, true],
            [1, 1, 1, 3],
            crate::SoftmaxQ30Policy::new(1024, 1024).unwrap(),
        )
        .unwrap();
        let output = attention_values_q10(
            softmax.probabilities(),
            softmax.shape(),
            &[4_i16, -4, 12],
            &[1, 1, 3, 1],
            policy(),
        )
        .unwrap();
        assert_eq!(output.shape(), [1, 1, 1, 1]);
        let expected = div_round_ties_even_i128(
            softmax
                .probabilities()
                .iter()
                .zip([4_i64, -4, 12])
                .map(|(&probability, value)| i128::from(probability) * i128::from(value))
                .sum(),
            1_i128 << 30,
        ) as i16;
        assert_eq!(output.values(), &[expected]);
    }
}
