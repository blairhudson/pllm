use std::{error::Error, fmt};

use serde::Serialize;
use zeroize::Zeroize;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use crate::fixed_point::div_round_ties_even_i128;

pub const ATTENTION_SCORE_Q20_PROFILE: &str =
    "pllm.numeric.attention_score.q10_dot_q20.q30_libm_0_2_16.v1";
pub const ATTENTION_SCORE_Q20_LAYOUT: &str = "[batch,query_heads,query,key_capacity]";
pub const ATTENTION_SCALE_Q30: i64 = 1_i64 << 30;
pub const ATTENTION_Q20_MAX_HEAD_DIM: usize = 8192;
pub const ATTENTION_Q20_MAX_SCORE_ELEMENTS: usize = 16 * 1024 * 1024;
pub const ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES: u64 = 16 * 1024 * 1024 * 1024;
/// Q30 coefficient error against pinned-libm `1 / sqrt(head_dim)`, in raw Q30 units.
pub const ATTENTION_Q30_COEFFICIENT_MAX_ERROR_RAW: f64 = 0.5;
/// Output error against the exact pinned-libm coefficient, in raw Q20 units.
pub const ATTENTION_Q20_MAX_ENCODED_ERROR_RAW: f64 = ATTENTION_Q20_MAX_HEAD_DIM as f64 / 2.0 + 0.5;
pub const ATTENTION_Q20_MAX_DECODED_ERROR: f64 =
    ATTENTION_Q20_MAX_ENCODED_ERROR_RAW / (1_u64 << 20) as f64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionScoreQ20Policy {
    max_score_elements: usize,
    max_multiply_accumulates: u64,
}

impl AttentionScoreQ20Policy {
    pub fn new(
        max_score_elements: usize,
        max_multiply_accumulates: u64,
    ) -> Result<Self, AttentionScoreQ20Error> {
        validate_policy_limit("max_score_elements", max_score_elements as u64)?;
        validate_policy_limit("max_multiply_accumulates", max_multiply_accumulates)?;
        if max_score_elements > ATTENTION_Q20_MAX_SCORE_ELEMENTS {
            return Err(AttentionScoreQ20Error::PolicyLimitExceedsHardCap {
                limit: "max_score_elements",
                value: max_score_elements as u64,
                maximum: ATTENTION_Q20_MAX_SCORE_ELEMENTS as u64,
            });
        }
        if max_multiply_accumulates > ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES {
            return Err(AttentionScoreQ20Error::PolicyLimitExceedsHardCap {
                limit: "max_multiply_accumulates",
                value: max_multiply_accumulates,
                maximum: ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES,
            });
        }
        Ok(Self {
            max_score_elements,
            max_multiply_accumulates,
        })
    }

    pub const fn max_score_elements(self) -> usize {
        self.max_score_elements
    }

    pub const fn max_multiply_accumulates(self) -> u64 {
        self.max_multiply_accumulates
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionScoreQ20Layout {
    GroupedQueryCache,
    PerQueryHeadWindow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionScoreQ20Error {
    InvalidDimension {
        tensor: &'static str,
        axis: &'static str,
    },
    BatchMismatch {
        query: usize,
        key: usize,
    },
    HeadDimensionMismatch {
        query: usize,
        key: usize,
    },
    QueryHeadsNotDivisible {
        query_heads: usize,
        key_value_heads: usize,
    },
    HeadDimensionTooLarge {
        head_dim: usize,
        maximum: usize,
    },
    ShapeOverflow,
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    ValidLengthOutOfRange {
        batch: usize,
        valid_length: usize,
        capacity: usize,
    },
    PositionOutOfRange {
        index: usize,
        position: u32,
        capacity: usize,
    },
    NonCanonicalQueryMask {
        index: usize,
        value: u8,
    },
    ActiveQueryAfterPadding {
        batch: usize,
        query: usize,
    },
    InvalidWindowRank,
    SelectedPositionAfterPadding {
        row: usize,
        index: usize,
    },
    SelectedPositionsNotSorted {
        row: usize,
        index: usize,
        previous: u32,
        current: u32,
    },
    InvalidPolicyLimit {
        limit: &'static str,
    },
    PolicyLimitExceedsHardCap {
        limit: &'static str,
        value: u64,
        maximum: u64,
    },
    ResourceLimitExceeded {
        score_elements: usize,
        maximum: usize,
    },
    MultiplyAccumulatesLimitExceeded {
        multiply_accumulates: u64,
        maximum: u64,
    },
    NonFiniteCoefficient,
    ArithmeticOverflow,
    AllocationFailed,
}

impl fmt::Display for AttentionScoreQ20Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { tensor, axis } => {
                write!(formatter, "attention {tensor} {axis} dimension must be positive")
            }
            Self::BatchMismatch { query, key } => {
                write!(formatter, "attention query batch {query} differs from key batch {key}")
            }
            Self::HeadDimensionMismatch { query, key } => write!(
                formatter,
                "attention query head dimension {query} differs from key head dimension {key}"
            ),
            Self::QueryHeadsNotDivisible {
                query_heads,
                key_value_heads,
            } => write!(
                formatter,
                "attention query heads {query_heads} are not divisible by key/value heads {key_value_heads}"
            ),
            Self::HeadDimensionTooLarge { head_dim, maximum } => write!(
                formatter,
                "attention head dimension {head_dim} exceeds maximum {maximum}"
            ),
            Self::ShapeOverflow => write!(formatter, "attention shape arithmetic overflowed"),
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "attention {tensor} length must be {expected}, received {actual}"
            ),
            Self::ValidLengthOutOfRange {
                batch,
                valid_length,
                capacity,
            } => write!(
                formatter,
                "attention batch {batch} valid length {valid_length} exceeds key capacity {capacity}"
            ),
            Self::PositionOutOfRange {
                index,
                position,
                capacity,
            } => write!(
                formatter,
                "attention position {position} at index {index} is outside key capacity {capacity}"
            ),
            Self::NonCanonicalQueryMask { index, value } => write!(
                formatter,
                "attention query mask value {value} at index {index} is not canonical 0/1"
            ),
            Self::ActiveQueryAfterPadding { batch, query } => write!(
                formatter,
                "attention batch {batch} query {query} is active after padding"
            ),
            Self::InvalidWindowRank => write!(
                formatter,
                "attention key window must align with the query batch, heads, and queries"
            ),
            Self::SelectedPositionAfterPadding { row, index } => write!(
                formatter,
                "attention selected position at row {row} index {index} follows padding"
            ),
            Self::SelectedPositionsNotSorted {
                row,
                index,
                previous,
                current,
            } => write!(
                formatter,
                "attention selected position {current} at row {row} index {index} is below {previous}"
            ),
            Self::InvalidPolicyLimit { limit } => {
                write!(formatter, "attention policy {limit} must be positive")
            }
            Self::PolicyLimitExceedsHardCap {
                limit,
                value,
                maximum,
            } => write!(
                formatter,
                "attention policy {limit} {value} exceeds hard cap {maximum}"
            ),
            Self::ResourceLimitExceeded {
                score_elements,
                maximum,
            } => write!(
                formatter,
                "attention score elements {score_elements} exceed maximum {maximum}"
            ),
            Self::MultiplyAccumulatesLimitExceeded {
                multiply_accumulates,
                maximum,
            } => write!(
                formatter,
                "attention multiply-accumulates {multiply_accumulates} exceed maximum {maximum}"
            ),
            Self::NonFiniteCoefficient => write!(formatter, "attention scale coefficient is not finite"),
            Self::ArithmeticOverflow => write!(formatter, "attention arithmetic overflowed"),
            Self::AllocationFailed => write!(formatter, "attention output allocation failed"),
        }
    }
}

impl Error for AttentionScoreQ20Error {}

/// Plaintext score and mask buffers. Both buffers are zeroized on drop.
/// Scores at mask-false positions are unspecified and carry no sentinel meaning.
///
/// Output intentionally implements neither `Debug` nor `Clone`:
///
/// ```compile_fail
/// use pllm_core::AttentionScoresQ20;
/// fn requires_debug<T: std::fmt::Debug>() {}
/// requires_debug::<AttentionScoresQ20>();
/// ```
///
/// ```compile_fail
/// use pllm_core::AttentionScoresQ20;
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<AttentionScoresQ20>();
/// ```
pub struct AttentionScoresQ20 {
    scores: Vec<i64>,
    allowed: Vec<bool>,
    shape: [usize; 4],
}

impl AttentionScoresQ20 {
    pub fn scores(&self) -> &[i64] {
        &self.scores
    }

    pub fn allowed(&self) -> &[bool] {
        &self.allowed
    }

    pub const fn shape(&self) -> [usize; 4] {
        self.shape
    }

    fn zeroize_contents(&mut self) {
        self.scores.zeroize();
        self.allowed.zeroize();
    }
}

impl Drop for AttentionScoresQ20 {
    fn drop(&mut self) {
        self.zeroize_contents();
        #[cfg(test)]
        {
            assert!(self.scores.is_empty());
            assert!(self.allowed.is_empty());
            ATTENTION_OUTPUT_DROPS.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }
}

#[cfg(test)]
static ATTENTION_OUTPUT_DROPS: AtomicUsize = AtomicUsize::new(0);

/// Return ties-to-even Q30 encoding of pinned-libm `1 / sqrt(head_dim)`.
pub fn attention_scale_q30(head_dim: usize) -> Result<i64, AttentionScoreQ20Error> {
    validate_head_dim(head_dim)?;
    let coefficient = 1.0 / libm::sqrt(head_dim as f64);
    let scaled = coefficient * ATTENTION_SCALE_Q30 as f64;
    if !scaled.is_finite() {
        return Err(AttentionScoreQ20Error::NonFiniteCoefficient);
    }
    Ok(scaled.round_ties_even() as i64)
}

/// Exact signed-Q10 grouped-query attention scores, returned as signed Q20.
///
/// `query_shape` is `[B,Hq,Q,D]`; `key_shape` is `[B,Hkv,K_capacity,D]`.
/// Key storage must contain full capacity, but only keys satisfying both
/// `key_index < valid_lengths[b]` and `key_index <= positions[b,q]` are read.
/// `query_mask` is canonical byte 0/1 `[B,Q]`; active entries must form a
/// prefix per batch. Inactive query values and positions are never read.
#[allow(clippy::too_many_arguments)]
pub fn attention_scores_q20(
    query: &[i16],
    query_shape: [usize; 4],
    key: &[i16],
    key_shape: [usize; 4],
    positions: &[u32],
    query_mask: &[u8],
    valid_lengths: &[usize],
    policy: AttentionScoreQ20Policy,
) -> Result<AttentionScoresQ20, AttentionScoreQ20Error> {
    attention_scores_q20_inner(
        query,
        query_shape,
        key,
        key_shape,
        positions,
        query_mask,
        valid_lengths,
        policy,
        key_shape[2],
        None,
    )
}

/// Cache-facing score path. `key_shape[2]` is visible output capacity while
/// `key_storage_capacity` remains physical stride of opaque cache storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_scores_q20_from_cache_storage(
    query: &[i16],
    query_shape: [usize; 4],
    key: &[i16],
    key_shape: [usize; 4],
    positions: &[u32],
    query_mask: &[u8],
    valid_lengths: &[usize],
    policy: AttentionScoreQ20Policy,
    key_storage_capacity: usize,
) -> Result<AttentionScoresQ20, AttentionScoreQ20Error> {
    attention_scores_q20_inner(
        query,
        query_shape,
        key,
        key_shape,
        positions,
        query_mask,
        valid_lengths,
        policy,
        key_storage_capacity,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn attention_scores_q20_inner(
    query: &[i16],
    query_shape: [usize; 4],
    key: &[i16],
    key_shape: [usize; 4],
    positions: &[u32],
    query_mask: &[u8],
    valid_lengths: &[usize],
    policy: AttentionScoreQ20Policy,
    key_storage_capacity: usize,
    _fault_after_writes: Option<usize>,
) -> Result<AttentionScoresQ20, AttentionScoreQ20Error> {
    let [batch, query_heads, query_length, head_dim] = query_shape;
    let [key_batch, key_value_heads, key_capacity, key_head_dim] = key_shape;
    validate_shape(query_shape, "query")?;
    validate_shape(key_shape, "key")?;
    if key_storage_capacity < key_capacity {
        return Err(AttentionScoreQ20Error::ValidLengthOutOfRange {
            batch: 0,
            valid_length: key_capacity,
            capacity: key_storage_capacity,
        });
    }
    if batch != key_batch {
        return Err(AttentionScoreQ20Error::BatchMismatch {
            query: batch,
            key: key_batch,
        });
    }
    if head_dim != key_head_dim {
        return Err(AttentionScoreQ20Error::HeadDimensionMismatch {
            query: head_dim,
            key: key_head_dim,
        });
    }
    validate_head_dim(head_dim)?;
    if query_heads % key_value_heads != 0 {
        return Err(AttentionScoreQ20Error::QueryHeadsNotDivisible {
            query_heads,
            key_value_heads,
        });
    }

    let score_elements = batch
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(query_length))
        .and_then(|value| value.checked_mul(key_capacity))
        .ok_or(AttentionScoreQ20Error::ShapeOverflow)?;
    if score_elements > policy.max_score_elements {
        return Err(AttentionScoreQ20Error::ResourceLimitExceeded {
            score_elements,
            maximum: policy.max_score_elements,
        });
    }

    let query_elements = checked_product(&query_shape)?;
    let key_elements = checked_product(&[
        key_batch,
        key_value_heads,
        key_storage_capacity,
        key_head_dim,
    ])?;
    let position_elements = batch
        .checked_mul(query_length)
        .ok_or(AttentionScoreQ20Error::ShapeOverflow)?;
    validate_length("positions", position_elements, positions.len())?;
    validate_length("query_mask", position_elements, query_mask.len())?;
    validate_length("valid_lengths", batch, valid_lengths.len())?;
    for (batch_index, &valid_length) in valid_lengths.iter().enumerate() {
        if valid_length > key_capacity {
            return Err(AttentionScoreQ20Error::ValidLengthOutOfRange {
                batch: batch_index,
                valid_length,
                capacity: key_capacity,
            });
        }
    }

    let mut multiply_accumulates = 0_u64;
    for (batch_index, &valid_length) in valid_lengths.iter().enumerate() {
        let mut saw_padding = false;
        for query_index in 0..query_length {
            let index = batch_index * query_length + query_index;
            match query_mask[index] {
                0 => saw_padding = true,
                1 if saw_padding => {
                    return Err(AttentionScoreQ20Error::ActiveQueryAfterPadding {
                        batch: batch_index,
                        query: query_index,
                    });
                }
                1 => {
                    let position = positions[index];
                    if u64::from(position) >= key_capacity as u64 {
                        return Err(AttentionScoreQ20Error::PositionOutOfRange {
                            index,
                            position,
                            capacity: key_capacity,
                        });
                    }
                    let allowed_keys = valid_length.min(position as usize + 1);
                    let work = u64::try_from(query_heads)
                        .ok()
                        .and_then(|value| value.checked_mul(allowed_keys as u64))
                        .and_then(|value| value.checked_mul(head_dim as u64))
                        .ok_or(AttentionScoreQ20Error::ShapeOverflow)?;
                    multiply_accumulates = multiply_accumulates
                        .checked_add(work)
                        .ok_or(AttentionScoreQ20Error::ShapeOverflow)?;
                }
                value => {
                    return Err(AttentionScoreQ20Error::NonCanonicalQueryMask { index, value });
                }
            }
        }
    }
    if multiply_accumulates > policy.max_multiply_accumulates {
        return Err(AttentionScoreQ20Error::MultiplyAccumulatesLimitExceeded {
            multiply_accumulates,
            maximum: policy.max_multiply_accumulates,
        });
    }

    validate_length("query", query_elements, query.len())?;
    validate_length("key", key_elements, key.len())?;

    let scale = i128::from(attention_scale_q30(head_dim)?);

    let mut output = AttentionScoresQ20 {
        scores: try_zeroed(score_elements)?,
        allowed: Vec::new(),
        shape: [batch, query_heads, query_length, key_capacity],
    };
    output
        .allowed
        .try_reserve_exact(score_elements)
        .map_err(|_| AttentionScoreQ20Error::AllocationFailed)?;
    output.allowed.resize(score_elements, false);
    let group_size = query_heads / key_value_heads;
    #[cfg(test)]
    let mut writes = 0_usize;
    for (batch_index, &valid_length) in valid_lengths.iter().enumerate() {
        for query_head in 0..query_heads {
            let key_head = query_head / group_size;
            for query_index in 0..query_length {
                let query_metadata_index = batch_index * query_length + query_index;
                if query_mask[query_metadata_index] == 0 {
                    continue;
                }
                let position = positions[query_metadata_index] as usize;
                let allowed_keys = valid_length.min(position + 1);
                let query_base = ((batch_index * query_heads + query_head) * query_length
                    + query_index)
                    * head_dim;
                let output_base = ((batch_index * query_heads + query_head) * query_length
                    + query_index)
                    * key_capacity;
                for key_index in 0..allowed_keys {
                    let key_base = ((batch_index * key_value_heads + key_head)
                        * key_storage_capacity
                        + key_index)
                        * head_dim;
                    let dot = query[query_base..query_base + head_dim]
                        .iter()
                        .zip(&key[key_base..key_base + head_dim])
                        .try_fold(0_i64, |sum, (&query_value, &key_value)| {
                            sum.checked_add(i64::from(query_value) * i64::from(key_value))
                                .ok_or(AttentionScoreQ20Error::ArithmeticOverflow)
                        })?;
                    let scaled = i128::from(dot)
                        .checked_mul(scale)
                        .ok_or(AttentionScoreQ20Error::ArithmeticOverflow)?;
                    let score = div_round_ties_even_i128(scaled, i128::from(ATTENTION_SCALE_Q30));
                    output.scores[output_base + key_index] = i64::try_from(score)
                        .map_err(|_| AttentionScoreQ20Error::ArithmeticOverflow)?;
                    output.allowed[output_base + key_index] = true;
                    #[cfg(test)]
                    {
                        writes += 1;
                        if _fault_after_writes == Some(writes) {
                            return Err(AttentionScoreQ20Error::ArithmeticOverflow);
                        }
                    }
                }
            }
        }
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn attention_scores_window_q20(
    query: &[i16],
    query_shape: [usize; 4],
    key: &[i16],
    key_shape: [usize; 5],
    selected_positions: &[u32],
    positions: &[u32],
    query_mask: &[u8],
    policy: AttentionScoreQ20Policy,
) -> Result<AttentionScoresQ20, AttentionScoreQ20Error> {
    let [batch, query_heads, query_length, head_dim] = query_shape;
    let [key_batch, key_heads, key_queries, window, key_head_dim] = key_shape;
    validate_shape(query_shape, "query")?;
    for (axis, value) in ["batch", "heads", "query", "sequence", "head_dim"]
        .into_iter()
        .zip(key_shape)
    {
        if value == 0 {
            return Err(AttentionScoreQ20Error::InvalidDimension {
                tensor: "key",
                axis,
            });
        }
    }
    if key_batch != batch {
        return Err(AttentionScoreQ20Error::BatchMismatch {
            query: batch,
            key: key_batch,
        });
    }
    if key_heads != query_heads || key_queries != query_length {
        return Err(AttentionScoreQ20Error::InvalidWindowRank);
    }
    if head_dim != key_head_dim {
        return Err(AttentionScoreQ20Error::HeadDimensionMismatch {
            query: head_dim,
            key: key_head_dim,
        });
    }
    validate_head_dim(head_dim)?;

    let score_elements = batch
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(query_length))
        .and_then(|value| value.checked_mul(window))
        .ok_or(AttentionScoreQ20Error::ShapeOverflow)?;
    if score_elements > policy.max_score_elements {
        return Err(AttentionScoreQ20Error::ResourceLimitExceeded {
            score_elements,
            maximum: policy.max_score_elements,
        });
    }

    let query_elements = checked_product(&query_shape)?;
    let key_elements = checked_product(&key_shape)?;
    let position_elements = batch
        .checked_mul(query_length)
        .ok_or(AttentionScoreQ20Error::ShapeOverflow)?;
    validate_length("positions", position_elements, positions.len())?;
    validate_length("query_mask", position_elements, query_mask.len())?;
    validate_length(
        "selected_positions",
        score_elements,
        selected_positions.len(),
    )?;

    let head_dim_u64 =
        u64::try_from(head_dim).map_err(|_| AttentionScoreQ20Error::ShapeOverflow)?;
    let mut multiply_accumulates = 0_u64;
    for batch_index in 0..batch {
        let mut saw_padding = false;
        for query_index in 0..query_length {
            let index = batch_index * query_length + query_index;
            match query_mask[index] {
                0 => saw_padding = true,
                1 if saw_padding => {
                    return Err(AttentionScoreQ20Error::ActiveQueryAfterPadding {
                        batch: batch_index,
                        query: query_index,
                    });
                }
                1 => {
                    let position = positions[index];
                    for head in 0..query_heads {
                        let row = (batch_index * query_heads + head) * query_length + query_index;
                        let row_base = row * window;
                        let mut padded = false;
                        let mut previous = None;
                        let mut allowed_keys = 0_u64;
                        for (key_index, &selected) in selected_positions
                            [row_base..row_base + window]
                            .iter()
                            .enumerate()
                        {
                            if selected == u32::MAX {
                                padded = true;
                                continue;
                            }
                            if padded {
                                return Err(AttentionScoreQ20Error::SelectedPositionAfterPadding {
                                    row,
                                    index: key_index,
                                });
                            }
                            if let Some(previous) = previous {
                                if selected < previous {
                                    return Err(
                                        AttentionScoreQ20Error::SelectedPositionsNotSorted {
                                            row,
                                            index: key_index,
                                            previous,
                                            current: selected,
                                        },
                                    );
                                }
                            }
                            previous = Some(selected);
                            if selected <= position {
                                allowed_keys += 1;
                            }
                        }
                        multiply_accumulates = multiply_accumulates
                            .checked_add(
                                allowed_keys
                                    .checked_mul(head_dim_u64)
                                    .ok_or(AttentionScoreQ20Error::ShapeOverflow)?,
                            )
                            .ok_or(AttentionScoreQ20Error::ShapeOverflow)?;
                    }
                }
                value => {
                    return Err(AttentionScoreQ20Error::NonCanonicalQueryMask { index, value });
                }
            }
        }
    }
    if multiply_accumulates > policy.max_multiply_accumulates {
        return Err(AttentionScoreQ20Error::MultiplyAccumulatesLimitExceeded {
            multiply_accumulates,
            maximum: policy.max_multiply_accumulates,
        });
    }

    validate_length("query", query_elements, query.len())?;
    validate_length("key", key_elements, key.len())?;

    let scale = i128::from(attention_scale_q30(head_dim)?);

    let mut output = AttentionScoresQ20 {
        scores: try_zeroed(score_elements)?,
        allowed: Vec::new(),
        shape: [batch, query_heads, query_length, window],
    };
    output
        .allowed
        .try_reserve_exact(score_elements)
        .map_err(|_| AttentionScoreQ20Error::AllocationFailed)?;
    output.allowed.resize(score_elements, false);
    for batch_index in 0..batch {
        for query_index in 0..query_length {
            let metadata_index = batch_index * query_length + query_index;
            if query_mask[metadata_index] == 0 {
                continue;
            }
            let position = positions[metadata_index];
            for query_head in 0..query_heads {
                let row = (batch_index * query_heads + query_head) * query_length + query_index;
                let row_base = row * window;
                let query_base = row * head_dim;
                for (key_index, &selected) in selected_positions[row_base..row_base + window]
                    .iter()
                    .enumerate()
                {
                    if selected == u32::MAX || selected > position {
                        continue;
                    }
                    let key_base = (row_base + key_index) * head_dim;
                    let dot = query[query_base..query_base + head_dim]
                        .iter()
                        .zip(&key[key_base..key_base + head_dim])
                        .try_fold(0_i64, |sum, (&query_value, &key_value)| {
                            sum.checked_add(i64::from(query_value) * i64::from(key_value))
                                .ok_or(AttentionScoreQ20Error::ArithmeticOverflow)
                        })?;
                    let scaled = i128::from(dot)
                        .checked_mul(scale)
                        .ok_or(AttentionScoreQ20Error::ArithmeticOverflow)?;
                    let score = div_round_ties_even_i128(scaled, i128::from(ATTENTION_SCALE_Q30));
                    output.scores[row_base + key_index] = i64::try_from(score)
                        .map_err(|_| AttentionScoreQ20Error::ArithmeticOverflow)?;
                    output.allowed[row_base + key_index] = true;
                }
            }
        }
    }

    Ok(output)
}

fn validate_shape(shape: [usize; 4], tensor: &'static str) -> Result<(), AttentionScoreQ20Error> {
    for (axis, value) in ["batch", "heads", "sequence", "head_dim"]
        .into_iter()
        .zip(shape)
    {
        if value == 0 {
            return Err(AttentionScoreQ20Error::InvalidDimension { tensor, axis });
        }
    }
    Ok(())
}

fn validate_head_dim(head_dim: usize) -> Result<(), AttentionScoreQ20Error> {
    if head_dim == 0 {
        return Err(AttentionScoreQ20Error::InvalidDimension {
            tensor: "query",
            axis: "head_dim",
        });
    }
    if head_dim > ATTENTION_Q20_MAX_HEAD_DIM {
        return Err(AttentionScoreQ20Error::HeadDimensionTooLarge {
            head_dim,
            maximum: ATTENTION_Q20_MAX_HEAD_DIM,
        });
    }
    // i16::MIN squared is 2^30; this proves every accepted dot fits i64.
    let _maximum_dot = (head_dim as i64)
        .checked_mul(1_i64 << 30)
        .ok_or(AttentionScoreQ20Error::ArithmeticOverflow)?;
    Ok(())
}

fn validate_policy_limit(limit: &'static str, value: u64) -> Result<(), AttentionScoreQ20Error> {
    if value == 0 {
        return Err(AttentionScoreQ20Error::InvalidPolicyLimit { limit });
    }
    Ok(())
}

fn checked_product(shape: &[usize]) -> Result<usize, AttentionScoreQ20Error> {
    shape.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(AttentionScoreQ20Error::ShapeOverflow)
    })
}

fn validate_length(
    tensor: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), AttentionScoreQ20Error> {
    if expected != actual {
        return Err(AttentionScoreQ20Error::LengthMismatch {
            tensor,
            expected,
            actual,
        });
    }
    Ok(())
}

fn try_zeroed<T: Default + Clone>(length: usize) -> Result<Vec<T>, AttentionScoreQ20Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| AttentionScoreQ20Error::AllocationFailed)?;
    values.resize(length, T::default());
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> AttentionScoreQ20Policy {
        AttentionScoreQ20Policy::new(
            ATTENTION_Q20_MAX_SCORE_ELEMENTS,
            ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES,
        )
        .unwrap()
    }

    fn score_index(shape: [usize; 4], b: usize, h: usize, q: usize, k: usize) -> usize {
        (((b * shape[1] + h) * shape[2] + q) * shape[3]) + k
    }

    #[test]
    fn coefficient_profile_and_locked_vectors() {
        assert_eq!(
            ATTENTION_SCORE_Q20_PROFILE,
            "pllm.numeric.attention_score.q10_dot_q20.q30_libm_0_2_16.v1"
        );
        assert_eq!(
            ATTENTION_SCORE_Q20_LAYOUT,
            "[batch,query_heads,query,key_capacity]"
        );
        assert_eq!(ATTENTION_SCALE_Q30, 1_073_741_824);
        assert_eq!(ATTENTION_Q20_MAX_HEAD_DIM, 8192);
        assert_eq!(ATTENTION_Q20_MAX_SCORE_ELEMENTS, 16_777_216);
        assert_eq!(ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES, 17_179_869_184);
        assert_eq!(ATTENTION_Q30_COEFFICIENT_MAX_ERROR_RAW, 0.5);
        assert_eq!(ATTENTION_Q20_MAX_ENCODED_ERROR_RAW, 4096.5);
        assert_eq!(ATTENTION_Q20_MAX_DECODED_ERROR, 0.003_906_726_837_158_203);
        for (head_dim, expected) in [
            (1, 1_073_741_824),
            (2, 759_250_125),
            (64, 134_217_728),
            (128, 94_906_266),
            (8192, 11_863_283),
        ] {
            assert_eq!(attention_scale_q30(head_dim), Ok(expected));
            let reference = 1.0 / libm::sqrt(head_dim as f64) * ATTENTION_SCALE_Q30 as f64;
            assert!((expected as f64 - reference).abs() <= ATTENTION_Q30_COEFFICIENT_MAX_ERROR_RAW);
        }
    }

    #[test]
    fn scalar_reference_and_gqa_mapping() {
        let query = [1, 2, 10, 20, 100, 200, 1000, 2000];
        let key = [3, 4, 5, 6, 30, 40, 50, 60];
        let output = attention_scores_q20(
            &query,
            [1, 4, 1, 2],
            &key,
            [1, 2, 2, 2],
            &[1],
            &[1],
            &[2],
            policy(),
        )
        .unwrap();
        assert_eq!(output.shape(), [1, 4, 1, 2]);
        let scale = attention_scale_q30(2).unwrap() as i128;
        let expected_dots = [
            [11_i64, 17],
            [110, 170],
            [11_000, 17_000],
            [110_000, 170_000],
        ];
        for (head, dots) in expected_dots.into_iter().enumerate() {
            for (key_index, dot) in dots.into_iter().enumerate() {
                let index = score_index(output.shape(), 0, head, 0, key_index);
                let expected = div_round_ties_even_i128(
                    i128::from(dot) * scale,
                    i128::from(ATTENTION_SCALE_Q30),
                ) as i64;
                assert_eq!(output.scores()[index], expected);
                assert!(output.allowed()[index]);
            }
        }
    }

    #[test]
    fn causal_prefill_decode_and_variable_batch_lengths() {
        let query = vec![1_i16; 6];
        let key = [10, 20, 30, 99, 40, 50, 99, 99];
        let output = attention_scores_q20(
            &query,
            [2, 1, 3, 1],
            &key,
            [2, 1, 4, 1],
            &[0, 1, 2, 0, 1, 3],
            &[1, 1, 1, 1, 1, 1],
            &[3, 2],
            policy(),
        )
        .unwrap();
        for batch in 0..2 {
            for query_index in 0..3 {
                for key_index in 0..4 {
                    let index = score_index(output.shape(), batch, 0, query_index, key_index);
                    let expected_allowed = key_index < [3, 2][batch]
                        && key_index <= [0, 1, 2, 0, 1, 3][batch * 3 + query_index];
                    assert_eq!(output.allowed()[index], expected_allowed);
                }
            }
        }
        assert_eq!(output.scores()[score_index(output.shape(), 0, 0, 2, 2)], 30);
        assert_eq!(output.scores()[score_index(output.shape(), 1, 0, 2, 1)], 50);

        let decoded = attention_scores_q20(
            &[2],
            [1, 1, 1, 1],
            &[7, 8, 9],
            [1, 1, 3, 1],
            &[2],
            &[1],
            &[3],
            policy(),
        )
        .unwrap();
        assert_eq!(decoded.scores(), &[14, 16, 18]);
        assert_eq!(decoded.allowed(), &[true, true, true]);
    }

    #[test]
    fn scaling_rounds_ties_to_even_for_both_signs() {
        let mut query = vec![0_i16; 4 * 64];
        for (row, value) in [4, 12, -4, -12].into_iter().enumerate() {
            query[row * 64] = value;
        }
        let mut key = vec![0_i16; 64];
        key[0] = 1;
        let output = attention_scores_q20(
            &query,
            [1, 4, 1, 64],
            &key,
            [1, 1, 1, 64],
            &[0],
            &[1],
            &[1],
            policy(),
        )
        .unwrap();
        assert_eq!(output.scores(), &[0, 2, 0, -2]);
    }

    #[test]
    fn exact_extreme_dot_fits_proven_i64_domain() {
        let query = vec![i16::MIN; ATTENTION_Q20_MAX_HEAD_DIM];
        let key = vec![i16::MIN; ATTENTION_Q20_MAX_HEAD_DIM];
        let output = attention_scores_q20(
            &query,
            [1, 1, 1, ATTENTION_Q20_MAX_HEAD_DIM],
            &key,
            [1, 1, 1, ATTENTION_Q20_MAX_HEAD_DIM],
            &[0],
            &[1],
            &[1],
            policy(),
        )
        .unwrap();
        let dot = (ATTENTION_Q20_MAX_HEAD_DIM as i64) * (1_i64 << 30);
        let expected = div_round_ties_even_i128(
            i128::from(dot) * i128::from(attention_scale_q30(ATTENTION_Q20_MAX_HEAD_DIM).unwrap()),
            i128::from(ATTENTION_SCALE_Q30),
        ) as i64;
        assert_eq!(output.scores(), &[expected]);
    }

    #[test]
    fn poison_tail_never_affects_disallowed_scores() {
        let clean = attention_scores_q20(
            &[3],
            [1, 1, 1, 1],
            &[5, 0, 0],
            [1, 1, 3, 1],
            &[0],
            &[1],
            &[1],
            policy(),
        )
        .unwrap();
        let poison = attention_scores_q20(
            &[3],
            [1, 1, 1, 1],
            &[5, i16::MIN, i16::MAX],
            [1, 1, 3, 1],
            &[0],
            &[1],
            &[1],
            policy(),
        )
        .unwrap();
        assert_eq!(clean.scores(), poison.scores());
        assert_eq!(poison.allowed(), &[true, false, false]);
    }

    #[test]
    fn padded_batches_ignore_sentinel_positions_and_poison_query_and_key_tails() {
        let query_clean = [2, 3, 0, 4, 0, 0];
        let query_poison = [2, 3, i16::MIN, 4, i16::MAX, i16::MIN];
        let key_clean = [5, 6, 0, 7, 0, 0];
        let key_poison = [5, 6, i16::MIN, 7, i16::MAX, i16::MIN];
        let positions = [0, 1, u32::MAX, 0, u32::MAX, u32::MAX];
        let query_mask = [1, 1, 0, 1, 0, 0];
        let exact_policy = AttentionScoreQ20Policy::new(18, 4).unwrap();
        let clean = attention_scores_q20(
            &query_clean,
            [2, 1, 3, 1],
            &key_clean,
            [2, 1, 3, 1],
            &positions,
            &query_mask,
            &[2, 1],
            exact_policy,
        )
        .unwrap();
        let poison = attention_scores_q20(
            &query_poison,
            [2, 1, 3, 1],
            &key_poison,
            [2, 1, 3, 1],
            &positions,
            &query_mask,
            &[2, 1],
            exact_policy,
        )
        .unwrap();
        assert_eq!(clean.scores(), poison.scores());
        assert_eq!(clean.allowed(), poison.allowed());
        for (batch, first_inactive) in [(0, 2), (1, 1)] {
            for query_index in first_inactive..3 {
                let row = score_index(poison.shape(), batch, 0, query_index, 0);
                assert_eq!(&poison.allowed()[row..row + 3], &[false; 3]);
            }
        }
    }

    #[test]
    fn rejects_noncanonical_nonprefix_and_wrong_length_query_masks() {
        for (mask, expected) in [
            (
                &[2_u8, 0][..],
                AttentionScoreQ20Error::NonCanonicalQueryMask { index: 0, value: 2 },
            ),
            (
                &[0, 1][..],
                AttentionScoreQ20Error::ActiveQueryAfterPadding { batch: 0, query: 1 },
            ),
        ] {
            assert_eq!(
                attention_scores_q20(
                    &[0; 2],
                    [1, 1, 2, 1],
                    &[0],
                    [1, 1, 1, 1],
                    &[0, 0],
                    mask,
                    &[1],
                    policy(),
                )
                .err(),
                Some(expected)
            );
        }
        assert_eq!(
            attention_scores_q20(
                &[0; 2],
                [1, 1, 2, 1],
                &[0],
                [1, 1, 1, 1],
                &[0, 0],
                &[1],
                &[1],
                policy(),
            )
            .err(),
            Some(AttentionScoreQ20Error::LengthMismatch {
                tensor: "query_mask",
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn policy_is_positive_bounded_and_rejects_work_before_secret_buffers() {
        assert_eq!(
            AttentionScoreQ20Policy::new(0, 1),
            Err(AttentionScoreQ20Error::InvalidPolicyLimit {
                limit: "max_score_elements"
            })
        );
        assert_eq!(
            AttentionScoreQ20Policy::new(1, 0),
            Err(AttentionScoreQ20Error::InvalidPolicyLimit {
                limit: "max_multiply_accumulates"
            })
        );
        assert!(matches!(
            AttentionScoreQ20Policy::new(ATTENTION_Q20_MAX_SCORE_ELEMENTS + 1, 1),
            Err(AttentionScoreQ20Error::PolicyLimitExceedsHardCap {
                limit: "max_score_elements",
                ..
            })
        ));
        assert!(matches!(
            AttentionScoreQ20Policy::new(1, ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES + 1),
            Err(AttentionScoreQ20Error::PolicyLimitExceedsHardCap {
                limit: "max_multiply_accumulates",
                ..
            })
        ));

        let restrictive = AttentionScoreQ20Policy::new(8192, 1_000_000).unwrap();
        assert_eq!(
            attention_scores_q20(
                &[],
                [1, 8192, 1, 8192],
                &[],
                [1, 1, 1, 8192],
                &[0],
                &[1],
                &[1],
                restrictive,
            )
            .err(),
            Some(AttentionScoreQ20Error::MultiplyAccumulatesLimitExceeded {
                multiply_accumulates: 67_108_864,
                maximum: 1_000_000,
            })
        );
    }

    #[test]
    fn rejects_shapes_ranges_and_resources_before_allocation() {
        assert!(matches!(
            attention_scores_q20(
                &[],
                [1, 1, 1, 0],
                &[],
                [1, 1, 1, 0],
                &[0],
                &[1],
                &[1],
                policy()
            ),
            Err(AttentionScoreQ20Error::InvalidDimension { .. })
        ));
        assert!(matches!(
            attention_scores_q20(
                &[0; 3],
                [1, 3, 1, 1],
                &[0; 2],
                [1, 2, 1, 1],
                &[0],
                &[1],
                &[1],
                policy()
            ),
            Err(AttentionScoreQ20Error::QueryHeadsNotDivisible { .. })
        ));
        assert!(matches!(
            attention_scores_q20(
                &[0],
                [1, 1, 1, 1],
                &[0],
                [1, 1, 1, 1],
                &[0],
                &[1],
                &[2],
                policy()
            ),
            Err(AttentionScoreQ20Error::ValidLengthOutOfRange { .. })
        ));
        assert!(matches!(
            attention_scores_q20(
                &[0],
                [1, 1, 1, 1],
                &[0],
                [1, 1, 1, 1],
                &[1],
                &[1],
                &[1],
                policy()
            ),
            Err(AttentionScoreQ20Error::PositionOutOfRange { .. })
        ));

        let capacity = ATTENTION_Q20_MAX_SCORE_ELEMENTS + 1;
        assert!(matches!(
            attention_scores_q20(
                &[0], [1, 1, 1, 1], &[], [1, 1, capacity, 1], &[0], &[1], &[0],
                policy()
            ),
            Err(AttentionScoreQ20Error::ResourceLimitExceeded {
                score_elements,
                maximum: ATTENTION_Q20_MAX_SCORE_ELEMENTS,
            }) if score_elements == capacity
        ));
        assert_eq!(
            attention_scale_q30(ATTENTION_Q20_MAX_HEAD_DIM + 1),
            Err(AttentionScoreQ20Error::HeadDimensionTooLarge {
                head_dim: ATTENTION_Q20_MAX_HEAD_DIM + 1,
                maximum: ATTENTION_Q20_MAX_HEAD_DIM,
            })
        );
    }

    #[test]
    fn output_error_bound_holds_against_pinned_reference() {
        let dot = (ATTENTION_Q20_MAX_HEAD_DIM as i64) * (1_i64 << 30);
        let encoded = div_round_ties_even_i128(
            i128::from(dot) * i128::from(attention_scale_q30(ATTENTION_Q20_MAX_HEAD_DIM).unwrap()),
            i128::from(ATTENTION_SCALE_Q30),
        ) as i64;
        let reference = dot as f64 / libm::sqrt(ATTENTION_Q20_MAX_HEAD_DIM as f64);
        assert!((encoded as f64 - reference).abs() <= ATTENTION_Q20_MAX_ENCODED_ERROR_RAW);
    }

    #[test]
    fn zeroizes_plaintext_output_buffers() {
        let mut output = attention_scores_q20(
            &[2],
            [1, 1, 1, 1],
            &[3],
            [1, 1, 1, 1],
            &[0],
            &[1],
            &[1],
            policy(),
        )
        .unwrap();
        output.zeroize_contents();
        assert!(output.scores().is_empty());
        assert!(output.allowed().is_empty());
    }

    #[test]
    fn partial_output_error_runs_zeroizing_drop() {
        let drops_before = ATTENTION_OUTPUT_DROPS.load(AtomicOrdering::SeqCst);
        let result = attention_scores_q20_inner(
            &[2],
            [1, 1, 1, 1],
            &[3],
            [1, 1, 1, 1],
            &[0],
            &[1],
            &[1],
            policy(),
            1,
            Some(1),
        );
        assert!(matches!(
            result,
            Err(AttentionScoreQ20Error::ArithmeticOverflow)
        ));
        assert!(ATTENTION_OUTPUT_DROPS.load(AtomicOrdering::SeqCst) > drops_before);
    }

    fn scaled_score(dot: i64, head_dim: usize) -> i64 {
        div_round_ties_even_i128(
            i128::from(dot) * i128::from(attention_scale_q30(head_dim).unwrap()),
            i128::from(ATTENTION_SCALE_Q30),
        ) as i64
    }

    #[test]
    fn window_matches_base_for_equivalent_contiguous_selection() {
        let query = [1, 2, 10, 20, 100, 200, 1000, 2000];
        let key = [3, 4, 5, 6, 30, 40, 50, 60];
        let base = attention_scores_q20(
            &query,
            [1, 4, 1, 2],
            &key,
            [1, 2, 2, 2],
            &[1],
            &[1],
            &[2],
            policy(),
        )
        .unwrap();
        let mut window_key = [0_i16; 16];
        for head in 0..4 {
            window_key[head * 4..head * 4 + 2]
                .copy_from_slice(&key[(head / 2) * 4..(head / 2) * 4 + 2]);
            window_key[head * 4 + 2..head * 4 + 4]
                .copy_from_slice(&key[(head / 2) * 4 + 2..(head / 2) * 4 + 4]);
        }
        let window = attention_scores_window_q20(
            &query,
            [1, 4, 1, 2],
            &window_key,
            [1, 4, 1, 2, 2],
            &[0, 1, 0, 1, 0, 1, 0, 1],
            &[1],
            &[1],
            policy(),
        )
        .unwrap();
        assert_eq!(window.shape(), base.shape());
        assert_eq!(window.scores(), base.scores());
        assert_eq!(window.allowed(), base.allowed());
    }

    #[test]
    fn window_scores_use_exact_strides_and_scaling() {
        let query: Vec<i16> = (0..16).map(|index| (index % 5) as i16 - 2).collect();
        let key: Vec<i16> = (0..48).map(|index| (index % 7) as i16 - 3).collect();
        let selected = [
            0,
            u32::MAX,
            u32::MAX,
            0,
            1,
            2,
            0,
            u32::MAX,
            u32::MAX,
            0,
            1,
            2,
            0,
            1,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            0,
            1,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        ];
        let positions = [0, 2, 1, 0];
        let query_mask = [1, 1, 1, 0];
        let output = attention_scores_window_q20(
            &query,
            [2, 2, 2, 2],
            &key,
            [2, 2, 2, 3, 2],
            &selected,
            &positions,
            &query_mask,
            policy(),
        )
        .unwrap();
        assert_eq!(output.shape(), [2, 2, 2, 3]);
        let expected_allowed = [
            [true, false, false],
            [true, true, true],
            [true, false, false],
            [true, true, true],
            [true, true, false],
            [false, false, false],
            [true, true, false],
            [false, false, false],
        ];
        for row in 0..8 {
            for key_index in 0..3 {
                let index = row * 3 + key_index;
                assert_eq!(output.allowed()[index], expected_allowed[row][key_index]);
                if expected_allowed[row][key_index] {
                    let dot: i64 = (0..2)
                        .map(|dimension| {
                            i64::from(query[row * 2 + dimension])
                                * i64::from(key[index * 2 + dimension])
                        })
                        .sum();
                    assert_eq!(output.scores()[index], scaled_score(dot, 2));
                } else {
                    assert_eq!(output.scores()[index], 0);
                }
            }
        }
    }

    #[test]
    fn window_ignores_disallowed_and_inactive_values() {
        let selected = [0, u32::MAX, u32::MAX, 0, 1, 2];
        let positions = [0, 2];
        let query_mask = [1, 1];
        let clean = attention_scores_window_q20(
            &[4, 0, 0, 4],
            [1, 1, 2, 2],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 1, 2, 3, 2],
            &selected,
            &positions,
            &query_mask,
            policy(),
        )
        .unwrap();
        let poison = attention_scores_window_q20(
            &[4, 0, 0, 4],
            [1, 1, 2, 2],
            &[
                1,
                2,
                i16::MIN,
                i16::MAX,
                i16::MIN,
                i16::MAX,
                7,
                8,
                9,
                10,
                11,
                12,
            ],
            [1, 1, 2, 3, 2],
            &selected,
            &positions,
            &query_mask,
            policy(),
        )
        .unwrap();
        assert_eq!(clean.scores(), poison.scores());
        assert_eq!(clean.allowed(), poison.allowed());
        assert_eq!(poison.allowed(), &[true, false, false, true, true, true]);
        assert_eq!(poison.scores()[0], scaled_score(4, 2));
        assert_eq!(poison.scores()[3], scaled_score(32, 2));
    }

    #[test]
    fn window_inactive_rows_skip_metadata_and_values() {
        let selected = [0, 1, 7, 3, u32::MAX, 0];
        let output = attention_scores_window_q20(
            &[5, 6, i16::MIN, i16::MAX],
            [1, 1, 2, 2],
            &[
                1,
                2,
                3,
                4,
                5,
                6,
                i16::MIN,
                i16::MAX,
                i16::MAX,
                i16::MIN,
                11,
                12,
            ],
            [1, 1, 2, 3, 2],
            &selected,
            &[1, u32::MAX],
            &[1, 0],
            policy(),
        )
        .unwrap();
        assert_eq!(output.allowed()[3..6], [false, false, false]);
        assert_eq!(output.scores()[3..6], [0, 0, 0]);
        assert_eq!(output.allowed(), &[true, true, false, false, false, false]);
    }

    #[test]
    fn window_rejects_unsorted_and_post_padding_selection() {
        let valid = attention_scores_window_q20(
            &[1, 2],
            [1, 1, 1, 2],
            &[3, 4, 5, 6],
            [1, 1, 1, 2, 2],
            &[0, 1],
            &[1],
            &[1],
            policy(),
        );
        assert!(valid.is_ok());
        assert_eq!(
            attention_scores_window_q20(
                &[1, 2],
                [1, 1, 1, 2],
                &[3, 4, 5, 6],
                [1, 1, 1, 2, 2],
                &[1, 0],
                &[1],
                &[1],
                policy(),
            )
            .err(),
            Some(AttentionScoreQ20Error::SelectedPositionsNotSorted {
                row: 0,
                index: 1,
                previous: 1,
                current: 0,
            })
        );
        assert!(attention_scores_window_q20(
            &[1, 2],
            [1, 1, 1, 2],
            &[3, 4, 5, 6],
            [1, 1, 1, 2, 2],
            &[0, u32::MAX],
            &[1],
            &[1],
            policy(),
        )
        .is_ok());
        assert_eq!(
            attention_scores_window_q20(
                &[1, 2],
                [1, 1, 1, 2],
                &[3, 4, 5, 6],
                [1, 1, 1, 2, 2],
                &[u32::MAX, 0],
                &[1],
                &[1],
                policy(),
            )
            .err(),
            Some(AttentionScoreQ20Error::SelectedPositionAfterPadding { row: 0, index: 1 })
        );
        assert_eq!(
            attention_scores_window_q20(
                &[1, 2],
                [1, 1, 1, 2],
                &[3, 4, 5, 6, 7, 8],
                [1, 1, 1, 3, 2],
                &[0, u32::MAX, 1],
                &[1],
                &[1],
                policy(),
            )
            .err(),
            Some(AttentionScoreQ20Error::SelectedPositionAfterPadding { row: 0, index: 2 })
        );
    }

    #[test]
    fn window_rejects_bad_shapes_lengths_masks_and_resources() {
        let base_args = |selected: &[u32], positions: &[u32], mask: &[u8]| {
            attention_scores_window_q20(
                &[1, 2],
                [1, 1, 1, 2],
                &[3, 4, 5, 6],
                [1, 1, 1, 2, 2],
                selected,
                positions,
                mask,
                policy(),
            )
        };
        assert!(matches!(
            attention_scores_window_q20(
                &[1, 2],
                [1, 1, 1, 2],
                &[3, 4, 5, 6],
                [1, 2, 1, 2, 2],
                &[0, 1],
                &[1],
                &[1],
                policy(),
            ),
            Err(AttentionScoreQ20Error::InvalidWindowRank)
        ));
        assert!(matches!(
            attention_scores_window_q20(
                &[1, 2],
                [1, 1, 1, 2],
                &[3, 4, 5, 6],
                [2, 1, 1, 2, 2],
                &[0, 1],
                &[1],
                &[1],
                policy(),
            ),
            Err(AttentionScoreQ20Error::BatchMismatch { .. })
        ));
        assert!(matches!(
            attention_scores_window_q20(
                &[1, 2],
                [1, 1, 1, 2],
                &[3, 4, 5, 6],
                [1, 1, 1, 2, 3],
                &[0, 1],
                &[1],
                &[1],
                policy(),
            ),
            Err(AttentionScoreQ20Error::HeadDimensionMismatch { .. })
        ));
        assert!(matches!(
            attention_scores_window_q20(
                &[1],
                [1, 1, 1, 2],
                &[3, 4, 5, 6],
                [1, 1, 1, 2, 2],
                &[0, 1],
                &[1],
                &[1],
                policy(),
            ),
            Err(AttentionScoreQ20Error::LengthMismatch {
                tensor: "query",
                ..
            })
        ));
        assert!(matches!(
            attention_scores_window_q20(
                &[1, 2],
                [1, 1, 1, 2],
                &[3, 4, 5],
                [1, 1, 1, 2, 2],
                &[0, 1],
                &[1],
                &[1],
                policy(),
            ),
            Err(AttentionScoreQ20Error::LengthMismatch { tensor: "key", .. })
        ));
        assert!(matches!(
            base_args(&[0, 1], &[1, 1], &[1]),
            Err(AttentionScoreQ20Error::LengthMismatch {
                tensor: "positions",
                ..
            })
        ));
        assert!(matches!(
            base_args(&[0, 1], &[1], &[1, 1]),
            Err(AttentionScoreQ20Error::LengthMismatch {
                tensor: "query_mask",
                ..
            })
        ));
        assert!(matches!(
            base_args(&[0], &[1], &[1]),
            Err(AttentionScoreQ20Error::LengthMismatch {
                tensor: "selected_positions",
                ..
            })
        ));
        assert_eq!(
            base_args(&[0, 1], &[1], &[2]).err(),
            Some(AttentionScoreQ20Error::NonCanonicalQueryMask { index: 0, value: 2 })
        );
        assert_eq!(
            attention_scores_window_q20(
                &[1, 2, 3, 4],
                [1, 1, 2, 2],
                &[0; 12],
                [1, 1, 2, 3, 2],
                &[0, 1, 2, 0, 1, 2],
                &[0, 1],
                &[0, 1],
                policy(),
            )
            .err(),
            Some(AttentionScoreQ20Error::ActiveQueryAfterPadding { batch: 0, query: 1 })
        );
        assert!(matches!(
            attention_scores_window_q20(
                &[0; 4],
                [1, 1, 1, 4],
                &[0; 8],
                [1, 1, 1, 2, 4],
                &[0, 1],
                &[1],
                &[1],
                AttentionScoreQ20Policy::new(1, ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES).unwrap(),
            ),
            Err(AttentionScoreQ20Error::ResourceLimitExceeded { .. })
        ));
        assert!(matches!(
            attention_scores_window_q20(
                &[1, 2],
                [1, 1, 1, 2],
                &[3, 4, 5, 6],
                [1, 1, 1, 2, 2],
                &[0, 1],
                &[1],
                &[1],
                AttentionScoreQ20Policy::new(ATTENTION_Q20_MAX_SCORE_ELEMENTS, 1).unwrap(),
            ),
            Err(AttentionScoreQ20Error::MultiplyAccumulatesLimitExceeded { .. })
        ));
    }
}
