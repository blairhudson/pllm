use std::{error::Error, fmt};

use zeroize::Zeroize;

pub const TOKEN_LOOKUP_Q10_PROFILE: &str = "pllm.numeric.token_lookup.q10_embedding.v1";
pub const TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS: usize = 16 * 1024 * 1024;
pub const TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS: usize = 1_073_741_824;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenLookupQ10Policy {
    max_output_elements: usize,
    max_weight_elements: usize,
}

impl TokenLookupQ10Policy {
    pub fn new(
        max_output_elements: usize,
        max_weight_elements: usize,
    ) -> Result<Self, TokenLookupQ10Error> {
        if max_output_elements == 0 {
            return Err(TokenLookupQ10Error::InvalidPolicyLimit {
                limit: "max_output_elements",
            });
        }
        if max_weight_elements == 0 {
            return Err(TokenLookupQ10Error::InvalidPolicyLimit {
                limit: "max_weight_elements",
            });
        }
        if max_output_elements > TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS {
            return Err(TokenLookupQ10Error::PolicyLimitExceedsHardCap {
                limit: "max_output_elements",
                value: max_output_elements,
                maximum: TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS,
            });
        }
        if max_weight_elements > TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS {
            return Err(TokenLookupQ10Error::PolicyLimitExceedsHardCap {
                limit: "max_weight_elements",
                value: max_weight_elements,
                maximum: TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS,
            });
        }
        Ok(Self {
            max_output_elements,
            max_weight_elements,
        })
    }

    pub const fn max_output_elements(self) -> usize {
        self.max_output_elements
    }

    pub const fn max_weight_elements(self) -> usize {
        self.max_weight_elements
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenLookupQ10Error {
    InvalidDimension {
        tensor: &'static str,
        axis: &'static str,
    },
    ShapeOverflow,
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    TokenOutOfRange {
        index: usize,
        token: u32,
        vocabulary: usize,
    },
    InvalidPolicyLimit {
        limit: &'static str,
    },
    PolicyLimitExceedsHardCap {
        limit: &'static str,
        value: usize,
        maximum: usize,
    },
    OutputElementsLimitExceeded {
        elements: usize,
        maximum: usize,
    },
    WeightElementsLimitExceeded {
        elements: usize,
        maximum: usize,
    },
    AllocationFailed,
}

impl fmt::Display for TokenLookupQ10Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { tensor, axis } => {
                write!(
                    formatter,
                    "token-lookup {tensor} {axis} dimension must be positive"
                )
            }
            Self::ShapeOverflow => write!(formatter, "token-lookup shape arithmetic overflowed"),
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "token-lookup {tensor} length must be {expected}, received {actual}"
            ),
            Self::TokenOutOfRange {
                index,
                token,
                vocabulary,
            } => write!(
                formatter,
                "token-lookup token {token} at index {index} is outside vocabulary {vocabulary}"
            ),
            Self::InvalidPolicyLimit { limit } => {
                write!(formatter, "token-lookup policy {limit} must be positive")
            }
            Self::PolicyLimitExceedsHardCap {
                limit,
                value,
                maximum,
            } => write!(
                formatter,
                "token-lookup policy {limit} {value} exceeds hard cap {maximum}"
            ),
            Self::OutputElementsLimitExceeded { elements, maximum } => write!(
                formatter,
                "token-lookup output elements {elements} exceed maximum {maximum}"
            ),
            Self::WeightElementsLimitExceeded { elements, maximum } => write!(
                formatter,
                "token-lookup weight elements {elements} exceed maximum {maximum}"
            ),
            Self::AllocationFailed => {
                write!(formatter, "token-lookup output allocation failed")
            }
        }
    }
}

impl Error for TokenLookupQ10Error {}

pub struct TokenEmbeddingsQ10 {
    values: Vec<i16>,
    shape: [usize; 3],
}

impl TokenEmbeddingsQ10 {
    pub fn values(&self) -> &[i16] {
        &self.values
    }

    pub const fn shape(&self) -> [usize; 3] {
        self.shape
    }
}

impl Drop for TokenEmbeddingsQ10 {
    fn drop(&mut self) {
        self.values.zeroize();
    }
}

fn checked_product(shape: &[usize]) -> Result<usize, TokenLookupQ10Error> {
    shape
        .iter()
        .try_fold(1_usize, |accumulator, &dimension| {
            accumulator.checked_mul(dimension)
        })
        .ok_or(TokenLookupQ10Error::ShapeOverflow)
}

pub fn token_lookup_q10(
    token_ids: &[u32],
    token_shape: [usize; 2],
    weights: &[i16],
    weight_shape: [usize; 2],
    policy: TokenLookupQ10Policy,
) -> Result<TokenEmbeddingsQ10, TokenLookupQ10Error> {
    const TOKEN_AXES: [&str; 2] = ["batch", "query"];
    const WEIGHT_AXES: [&str; 2] = ["vocabulary", "hidden"];
    for (index, &dimension) in token_shape.iter().enumerate() {
        if dimension == 0 {
            return Err(TokenLookupQ10Error::InvalidDimension {
                tensor: "tokens",
                axis: TOKEN_AXES[index],
            });
        }
    }
    for (index, &dimension) in weight_shape.iter().enumerate() {
        if dimension == 0 {
            return Err(TokenLookupQ10Error::InvalidDimension {
                tensor: "weights",
                axis: WEIGHT_AXES[index],
            });
        }
    }
    let (batch, queries) = (token_shape[0], token_shape[1]);
    let (vocabulary, hidden) = (weight_shape[0], weight_shape[1]);
    let token_elements = checked_product(&token_shape)?;
    if token_ids.len() != token_elements {
        return Err(TokenLookupQ10Error::LengthMismatch {
            tensor: "tokens",
            expected: token_elements,
            actual: token_ids.len(),
        });
    }
    let weight_elements = checked_product(&weight_shape)?;
    if weights.len() != weight_elements {
        return Err(TokenLookupQ10Error::LengthMismatch {
            tensor: "weights",
            expected: weight_elements,
            actual: weights.len(),
        });
    }
    let output_elements = batch
        .checked_mul(queries)
        .and_then(|product| product.checked_mul(hidden))
        .ok_or(TokenLookupQ10Error::ShapeOverflow)?;
    if output_elements > policy.max_output_elements() {
        return Err(TokenLookupQ10Error::OutputElementsLimitExceeded {
            elements: output_elements,
            maximum: policy.max_output_elements(),
        });
    }
    if output_elements > TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS {
        return Err(TokenLookupQ10Error::OutputElementsLimitExceeded {
            elements: output_elements,
            maximum: TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS,
        });
    }
    if weight_elements > policy.max_weight_elements() {
        return Err(TokenLookupQ10Error::WeightElementsLimitExceeded {
            elements: weight_elements,
            maximum: policy.max_weight_elements(),
        });
    }
    if weight_elements > TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS {
        return Err(TokenLookupQ10Error::WeightElementsLimitExceeded {
            elements: weight_elements,
            maximum: TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS,
        });
    }
    let vocabulary_u64 =
        u64::try_from(vocabulary).map_err(|_| TokenLookupQ10Error::ShapeOverflow)?;
    for (index, &token) in token_ids.iter().enumerate() {
        if u64::from(token) >= vocabulary_u64 {
            return Err(TokenLookupQ10Error::TokenOutOfRange {
                index,
                token,
                vocabulary,
            });
        }
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_elements)
        .map_err(|_| TokenLookupQ10Error::AllocationFailed)?;
    values.resize(output_elements, 0_i16);
    let mut output = TokenEmbeddingsQ10 {
        values,
        shape: [batch, queries, hidden],
    };
    for (row, &token) in token_ids.iter().enumerate() {
        let token_index = usize::try_from(token).map_err(|_| TokenLookupQ10Error::ShapeOverflow)?;
        let source = token_index
            .checked_mul(hidden)
            .ok_or(TokenLookupQ10Error::ShapeOverflow)?;
        let destination = row
            .checked_mul(hidden)
            .ok_or(TokenLookupQ10Error::ShapeOverflow)?;
        output.values[destination..destination + hidden]
            .copy_from_slice(&weights[source..source + hidden]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> TokenLookupQ10Policy {
        TokenLookupQ10Policy::new(
            TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS,
            TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS,
        )
        .unwrap()
    }

    #[test]
    fn gathers_distinct_rows_for_repeated_tokens() {
        let weights: Vec<i16> = (0..5 * 4)
            .map(|index| (index * 97 % 257 - 128) as i16)
            .collect();
        let token_ids = [4_u32, 0, 2, 2, 1, 4];
        let output = token_lookup_q10(&token_ids, [2, 3], &weights, [5, 4], policy()).unwrap();
        assert_eq!(output.shape(), [2, 3, 4]);
        let expected: Vec<i16> = [4_usize, 0, 2, 2, 1, 4]
            .iter()
            .flat_map(|&token| weights[token * 4..token * 4 + 4].iter().copied())
            .collect();
        assert_eq!(output.values(), expected.as_slice());
    }

    #[test]
    fn boundary_tokens_and_out_of_range_rejection() {
        let weights: Vec<i16> = (0..5 * 4).map(|index| index as i16 - 20).collect();
        let output = token_lookup_q10(&[0_u32, 4], [1, 2], &weights, [5, 4], policy()).unwrap();
        assert_eq!(output.values()[..4], weights[..4]);
        assert_eq!(output.values()[4..], weights[16..20]);
        let error = token_lookup_q10(&[5_u32, 0], [1, 2], &weights, [5, 4], policy())
            .err()
            .unwrap();
        assert_eq!(
            error,
            TokenLookupQ10Error::TokenOutOfRange {
                index: 0,
                token: 5,
                vocabulary: 5
            }
        );
    }

    #[test]
    fn rejects_bad_dimensions_lengths_and_policies() {
        let weights = vec![0_i16; 20];
        let tokens = vec![0_u32; 6];
        assert_eq!(
            token_lookup_q10(&tokens, [0, 3], &weights, [5, 4], policy()).err(),
            Some(TokenLookupQ10Error::InvalidDimension {
                tensor: "tokens",
                axis: "batch"
            })
        );
        assert_eq!(
            token_lookup_q10(&tokens, [2, 0], &weights, [5, 4], policy()).err(),
            Some(TokenLookupQ10Error::InvalidDimension {
                tensor: "tokens",
                axis: "query"
            })
        );
        assert_eq!(
            token_lookup_q10(&tokens, [2, 3], &weights, [0, 4], policy()).err(),
            Some(TokenLookupQ10Error::InvalidDimension {
                tensor: "weights",
                axis: "vocabulary"
            })
        );
        assert_eq!(
            token_lookup_q10(&tokens, [2, 3], &weights, [5, 0], policy()).err(),
            Some(TokenLookupQ10Error::InvalidDimension {
                tensor: "weights",
                axis: "hidden"
            })
        );
        assert_eq!(
            token_lookup_q10(&tokens[..5], [2, 3], &weights, [5, 4], policy()).err(),
            Some(TokenLookupQ10Error::LengthMismatch {
                tensor: "tokens",
                expected: 6,
                actual: 5
            })
        );
        assert_eq!(
            token_lookup_q10(&tokens, [2, 3], &weights[..19], [5, 4], policy()).err(),
            Some(TokenLookupQ10Error::LengthMismatch {
                tensor: "weights",
                expected: 20,
                actual: 19
            })
        );
        let tight_output =
            TokenLookupQ10Policy::new(8, TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS).unwrap();
        assert_eq!(
            token_lookup_q10(&tokens, [2, 3], &weights, [5, 4], tight_output).err(),
            Some(TokenLookupQ10Error::OutputElementsLimitExceeded {
                elements: 24,
                maximum: 8
            })
        );
        let tight_weights =
            TokenLookupQ10Policy::new(TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS, 16).unwrap();
        assert_eq!(
            token_lookup_q10(&tokens, [2, 3], &weights, [5, 4], tight_weights).err(),
            Some(TokenLookupQ10Error::WeightElementsLimitExceeded {
                elements: 20,
                maximum: 16
            })
        );
        assert_eq!(
            TokenLookupQ10Policy::new(0, 1).err(),
            Some(TokenLookupQ10Error::InvalidPolicyLimit {
                limit: "max_output_elements"
            })
        );
        assert_eq!(
            TokenLookupQ10Policy::new(1, 0).err(),
            Some(TokenLookupQ10Error::InvalidPolicyLimit {
                limit: "max_weight_elements"
            })
        );
        assert_eq!(
            TokenLookupQ10Policy::new(TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS + 1, 1).err(),
            Some(TokenLookupQ10Error::PolicyLimitExceedsHardCap {
                limit: "max_output_elements",
                value: TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS + 1,
                maximum: TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS
            })
        );
        assert_eq!(
            TokenLookupQ10Policy::new(1, TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS + 1).err(),
            Some(TokenLookupQ10Error::PolicyLimitExceedsHardCap {
                limit: "max_weight_elements",
                value: TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS + 1,
                maximum: TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS
            })
        );
    }

    #[test]
    fn output_is_deterministic_and_opaque() {
        let weights: Vec<i16> = (0..4 * 3).map(|index| index as i16 * 11 - 30).collect();
        let tokens = [3_u32, 1, 0];
        let first = token_lookup_q10(&tokens, [1, 3], &weights, [4, 3], policy()).unwrap();
        let second = token_lookup_q10(&tokens, [1, 3], &weights, [4, 3], policy()).unwrap();
        assert_eq!(first.values(), second.values());
        assert_eq!(first.shape(), [1, 3, 3]);
        let mut output = first;
        output.values.zeroize();
        assert!(output.values().iter().all(|&value| value == 0));
    }

    #[test]
    fn matches_oracle_across_shapes_and_ids() {
        for (batch, queries, vocabulary, hidden) in [
            (1_usize, 1, 2, 1),
            (2, 3, 5, 4),
            (3, 2, 7, 5),
            (1, 4, 11, 2),
        ] {
            let weight_elements = vocabulary * hidden;
            let weights: Vec<i16> = (0..weight_elements)
                .map(|index| ((index * 31 + vocabulary) % 197) as i16 - 98)
                .collect();
            let token_elements = batch * queries;
            let token_ids: Vec<u32> = (0..token_elements)
                .map(|index| ((index * 5 + batch) % vocabulary) as u32)
                .collect();
            let output = token_lookup_q10(
                &token_ids,
                [batch, queries],
                &weights,
                [vocabulary, hidden],
                policy(),
            )
            .unwrap();
            assert_eq!(output.shape(), [batch, queries, hidden]);
            for (row, &token) in token_ids.iter().enumerate() {
                let token = token as usize;
                assert_eq!(
                    &output.values()[row * hidden..row * hidden + hidden],
                    &weights[token * hidden..token * hidden + hidden]
                );
            }
        }
    }
}
