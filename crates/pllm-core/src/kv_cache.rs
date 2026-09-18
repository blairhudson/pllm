use std::{error::Error, fmt};

use zeroize::Zeroize;

use crate::attention::{
    attention_scores_q20_from_cache_storage, AttentionScoreQ20Error, AttentionScoreQ20Policy,
    AttentionScoresQ20,
};

pub const Q10_KV_CACHE_PROFILE: &str = "pllm.state.kv_cache.q10_bhcd.sequential_zero_padding.v1";
pub const Q10_KV_CACHE_LAYOUT: &str = "[batch,kv_heads,capacity,head_dim]";
pub const Q10_KV_CURRENT_LAYOUT: &str = "[batch,kv_heads,query,head_dim]";
pub const Q10_KV_ATTENTION_MASK_LAYOUT: &str = "[batch,current_sequence]";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheError {
    InvalidDimension {
        dimension: &'static str,
    },
    CapacityExceedsPositionDomain {
        capacity: usize,
    },
    ShapeOverflow,
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    ValidLengthRewind {
        batch: usize,
        old: usize,
        new: usize,
    },
    CapacityExceeded {
        batch: usize,
        valid_length: usize,
        capacity: usize,
    },
    DeltaExceedsQuery {
        batch: usize,
        delta: usize,
        query: usize,
    },
    PositionOverwrite {
        batch: usize,
        query: usize,
        expected: u32,
        actual: u32,
    },
    PositionGap {
        batch: usize,
        query: usize,
        expected: u32,
        actual: u32,
    },
    MissingActiveMask {
        batch: usize,
        query: usize,
    },
    UnexpectedActiveMask {
        batch: usize,
        query: usize,
    },
    NonZeroPadding {
        batch: usize,
        head: usize,
        query: usize,
        dimension: usize,
    },
    BatchOutOfRange {
        batch: usize,
    },
    HeadOutOfRange {
        head: usize,
    },
    VisibleLimitExceeded {
        batch: usize,
        valid_length: usize,
        maximum_visible: usize,
    },
    AllocationFailed,
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { dimension } => {
                write!(formatter, "KV cache {dimension} dimension must be positive")
            }
            Self::CapacityExceedsPositionDomain { capacity } => write!(
                formatter,
                "KV cache capacity {capacity} exceeds u32 position domain"
            ),
            Self::ShapeOverflow => write!(formatter, "KV cache shape arithmetic overflowed"),
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "KV cache {tensor} length must be {expected}, received {actual}"
            ),
            Self::ValidLengthRewind { batch, old, new } => write!(
                formatter,
                "KV cache batch {batch} valid length rewinds from {old} to {new}"
            ),
            Self::CapacityExceeded {
                batch,
                valid_length,
                capacity,
            } => write!(
                formatter,
                "KV cache batch {batch} valid length {valid_length} exceeds capacity {capacity}"
            ),
            Self::DeltaExceedsQuery {
                batch,
                delta,
                query,
            } => write!(
                formatter,
                "KV cache batch {batch} append delta {delta} exceeds query length {query}"
            ),
            Self::PositionOverwrite {
                batch,
                query,
                expected,
                actual,
            } => write!(
                formatter,
                "KV cache position {actual} at batch {batch}, query {query} would overwrite before {expected}"
            ),
            Self::PositionGap {
                batch,
                query,
                expected,
                actual,
            } => write!(
                formatter,
                "KV cache position {actual} at batch {batch}, query {query} leaves gap before {expected}"
            ),
            Self::MissingActiveMask { batch, query } => write!(
                formatter,
                "KV cache active row at batch {batch}, query {query} has a false attention mask"
            ),
            Self::UnexpectedActiveMask { batch, query } => write!(
                formatter,
                "KV cache padding row at batch {batch}, query {query} has a true attention mask"
            ),
            Self::NonZeroPadding {
                batch,
                head,
                query,
                dimension,
            } => write!(
                formatter,
                "KV cache invalid padding at batch {batch}, head {head}, query {query}, dimension {dimension} is nonzero"
            ),
            Self::BatchOutOfRange { batch } => {
                write!(formatter, "KV cache batch index {batch} is out of range")
            }
            Self::HeadOutOfRange { head } => {
                write!(formatter, "KV cache head index {head} is out of range")
            }
            Self::VisibleLimitExceeded {
                batch,
                valid_length,
                maximum_visible,
            } => write!(
                formatter,
                "KV cache batch {batch} visible length {valid_length} exceeds bound {maximum_visible}"
            ),
            Self::AllocationFailed => write!(formatter, "KV cache allocation failed"),
        }
    }
}

impl Error for KvCacheError {}

/// Bounded signed-Q10 cache. Storage contents are redacted from `Debug` and zeroized on drop.
pub struct Q10KvCache {
    storage: Vec<i16>,
    valid_lengths: Vec<usize>,
    batch: usize,
    kv_heads: usize,
    capacity: usize,
    head_dim: usize,
}

/// Explicit bounded-cache name used by provenance-bound compiler execution.
pub type BoundedKvCacheQ10 = Q10KvCache;

impl fmt::Debug for Q10KvCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Q10KvCache")
            .field("shape", &self.shape())
            .field("valid_lengths", &self.valid_lengths)
            .field("storage", &"<redacted>")
            .finish()
    }
}

impl Drop for Q10KvCache {
    fn drop(&mut self) {
        self.zeroize_contents();
    }
}

impl Q10KvCache {
    /// Build full-capacity storage and apply initial current rows to zero-length batches.
    /// Mask rows must be an exact `true` prefix matching each valid length. Rows after that
    /// prefix must have a false mask and contain only zero padding.
    #[allow(clippy::too_many_arguments)]
    pub fn initialize(
        batch: usize,
        kv_heads: usize,
        capacity: usize,
        head_dim: usize,
        current: &[i16],
        query: usize,
        positions: &[u32],
        attention_mask: &[bool],
        valid_lengths: &[usize],
    ) -> Result<Self, KvCacheError> {
        validate_dimension("batch", batch)?;
        validate_dimension("kv_heads", kv_heads)?;
        validate_dimension("capacity", capacity)?;
        validate_dimension("head_dim", head_dim)?;
        validate_dimension("query", query)?;
        if capacity > u32::MAX as usize {
            return Err(KvCacheError::CapacityExceedsPositionDomain { capacity });
        }
        validate_apply_inputs(
            batch,
            kv_heads,
            capacity,
            head_dim,
            None,
            current,
            query,
            positions,
            attention_mask,
            valid_lengths,
        )?;
        let elements = batch
            .checked_mul(kv_heads)
            .and_then(|value| value.checked_mul(capacity))
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or(KvCacheError::ShapeOverflow)?;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(elements)
            .map_err(|_| KvCacheError::AllocationFailed)?;
        storage.resize(elements, 0);
        let mut old_lengths = Vec::new();
        old_lengths
            .try_reserve_exact(batch)
            .map_err(|_| KvCacheError::AllocationFailed)?;
        old_lengths.resize(batch, 0);
        let mut cache = Self {
            storage,
            valid_lengths: old_lengths,
            batch,
            kv_heads,
            capacity,
            head_dim,
        };
        cache.write_validated(current, query, valid_lengths);
        Ok(cache)
    }

    /// Append current rows using new cumulative valid lengths. Validation is transactional.
    /// Mask rows must be an exact `true` prefix matching each append delta. Every later row must
    /// have a false mask and contain only zero padding.
    pub fn append(
        &mut self,
        current: &[i16],
        query: usize,
        positions: &[u32],
        attention_mask: &[bool],
        new_valid_lengths: &[usize],
    ) -> Result<(), KvCacheError> {
        validate_dimension("query", query)?;
        validate_apply_inputs(
            self.batch,
            self.kv_heads,
            self.capacity,
            self.head_dim,
            Some(&self.valid_lengths),
            current,
            query,
            positions,
            attention_mask,
            new_valid_lengths,
        )?;
        self.write_validated(current, query, new_valid_lengths);
        Ok(())
    }

    pub const fn shape(&self) -> [usize; 4] {
        [self.batch, self.kv_heads, self.capacity, self.head_dim]
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn valid_lengths(&self) -> &[usize] {
        &self.valid_lengths
    }

    /// Evaluate Q20 attention scores against opaque key storage.
    ///
    /// Output key extent is `visible_key_capacity`; physical reads retain full
    /// cache-capacity stride and are limited to each state's active prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_scores_q20(
        &self,
        query: &[i16],
        query_shape: [usize; 4],
        visible_key_capacity: usize,
        positions: &[u32],
        query_mask: &[u8],
        policy: AttentionScoreQ20Policy,
    ) -> Result<AttentionScoresQ20, AttentionScoreQ20Error> {
        attention_scores_q20_from_cache_storage(
            query,
            query_shape,
            &self.storage,
            [self.batch, self.kv_heads, visible_key_capacity, self.head_dim],
            positions,
            query_mask,
            &self.valid_lengths,
            policy,
            self.capacity,
        )
    }

    /// Borrow one `[visible_sequence, head_dim]` prefix without exposing unwritten slots.
    pub fn visible_prefix(
        &self,
        batch: usize,
        head: usize,
        maximum_visible: usize,
    ) -> Result<&[i16], KvCacheError> {
        if batch >= self.batch {
            return Err(KvCacheError::BatchOutOfRange { batch });
        }
        if head >= self.kv_heads {
            return Err(KvCacheError::HeadOutOfRange { head });
        }
        let visible = self.valid_lengths[batch];
        if visible > maximum_visible {
            return Err(KvCacheError::VisibleLimitExceeded {
                batch,
                valid_length: visible,
                maximum_visible,
            });
        }
        let start = ((batch * self.kv_heads + head) * self.capacity) * self.head_dim;
        let end = start + visible * self.head_dim;
        Ok(&self.storage[start..end])
    }

    fn write_validated(&mut self, current: &[i16], query: usize, new_valid_lengths: &[usize]) {
        for (batch, (&old, &new)) in self.valid_lengths.iter().zip(new_valid_lengths).enumerate() {
            let delta = new - old;
            for head in 0..self.kv_heads {
                for query_index in 0..delta {
                    let source = current_index(
                        batch,
                        head,
                        query_index,
                        0,
                        self.kv_heads,
                        query,
                        self.head_dim,
                    );
                    let destination =
                        ((batch * self.kv_heads + head) * self.capacity + old + query_index)
                            * self.head_dim;
                    self.storage[destination..destination + self.head_dim]
                        .copy_from_slice(&current[source..source + self.head_dim]);
                }
            }
        }
        self.valid_lengths.copy_from_slice(new_valid_lengths);
    }

    fn zeroize_contents(&mut self) {
        self.storage.zeroize();
        self.valid_lengths.zeroize();
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_apply_inputs(
    batch_count: usize,
    kv_heads: usize,
    capacity: usize,
    head_dim: usize,
    old_lengths: Option<&[usize]>,
    current: &[i16],
    query: usize,
    positions: &[u32],
    attention_mask: &[bool],
    new_valid_lengths: &[usize],
) -> Result<(), KvCacheError> {
    let expected_current = batch_count
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(query))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or(KvCacheError::ShapeOverflow)?;
    let expected_positions = batch_count
        .checked_mul(query)
        .ok_or(KvCacheError::ShapeOverflow)?;
    validate_length("current", expected_current, current.len())?;
    validate_length("positions", expected_positions, positions.len())?;
    validate_length("attention_mask", expected_positions, attention_mask.len())?;
    validate_length("valid_lengths", batch_count, new_valid_lengths.len())?;
    if let Some(old_lengths) = old_lengths {
        validate_length("old_valid_lengths", batch_count, old_lengths.len())?;
    }

    for (batch, &new) in new_valid_lengths.iter().enumerate() {
        let old = old_lengths.map_or(0, |lengths| lengths[batch]);
        if new < old {
            return Err(KvCacheError::ValidLengthRewind { batch, old, new });
        }
        if new > capacity {
            return Err(KvCacheError::CapacityExceeded {
                batch,
                valid_length: new,
                capacity,
            });
        }
        let delta = new - old;
        if delta > query {
            return Err(KvCacheError::DeltaExceedsQuery {
                batch,
                delta,
                query,
            });
        }
        for query_index in 0..query {
            let active = attention_mask[batch * query + query_index];
            if query_index < delta && !active {
                return Err(KvCacheError::MissingActiveMask {
                    batch,
                    query: query_index,
                });
            }
            if query_index >= delta && active {
                return Err(KvCacheError::UnexpectedActiveMask {
                    batch,
                    query: query_index,
                });
            }
        }
        for query_index in 0..delta {
            let expected =
                u32::try_from(old + query_index).map_err(|_| KvCacheError::ShapeOverflow)?;
            let actual = positions[batch * query + query_index];
            if actual < expected {
                return Err(KvCacheError::PositionOverwrite {
                    batch,
                    query: query_index,
                    expected,
                    actual,
                });
            }
            if actual > expected {
                return Err(KvCacheError::PositionGap {
                    batch,
                    query: query_index,
                    expected,
                    actual,
                });
            }
        }
        for head in 0..kv_heads {
            for query_index in delta..query {
                for dimension in 0..head_dim {
                    let index = current_index(
                        batch,
                        head,
                        query_index,
                        dimension,
                        kv_heads,
                        query,
                        head_dim,
                    );
                    if current[index] != 0 {
                        return Err(KvCacheError::NonZeroPadding {
                            batch,
                            head,
                            query: query_index,
                            dimension,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_dimension(dimension: &'static str, value: usize) -> Result<(), KvCacheError> {
    if value == 0 {
        return Err(KvCacheError::InvalidDimension { dimension });
    }
    Ok(())
}

fn validate_length(
    tensor: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), KvCacheError> {
    if expected != actual {
        return Err(KvCacheError::LengthMismatch {
            tensor,
            expected,
            actual,
        });
    }
    Ok(())
}

fn current_index(
    batch: usize,
    head: usize,
    query: usize,
    dimension: usize,
    kv_heads: usize,
    query_length: usize,
    head_dim: usize,
) -> usize {
    (((batch * kv_heads + head) * query_length + query) * head_dim) + dimension
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializes_prefill_and_exposes_only_visible_prefix() {
        let cache = Q10KvCache::initialize(
            1,
            2,
            4,
            2,
            &[1, 2, 3, 4, 0, 0, 11, 12, 13, 14, 0, 0],
            3,
            &[0, 1, 99],
            &[true, true, false],
            &[2],
        )
        .unwrap();
        assert_eq!(cache.shape(), [1, 2, 4, 2]);
        assert_eq!(cache.capacity(), 4);
        assert_eq!(cache.valid_lengths(), &[2]);
        assert_eq!(cache.visible_prefix(0, 0, 2), Ok(&[1, 2, 3, 4][..]));
        assert_eq!(cache.visible_prefix(0, 1, 2), Ok(&[11, 12, 13, 14][..]));
        assert_eq!(
            format!("{cache:?}"),
            "Q10KvCache { shape: [1, 2, 4, 2], valid_lengths: [2], storage: \"<redacted>\" }"
        );
    }

    #[test]
    fn appends_decode_rows() {
        let mut cache =
            Q10KvCache::initialize(1, 1, 4, 2, &[1, 2, 3, 4], 2, &[0, 1], &[true, true], &[2])
                .unwrap();
        cache.append(&[5, 6], 1, &[2], &[true], &[3]).unwrap();
        assert_eq!(cache.valid_lengths(), &[3]);
        assert_eq!(cache.visible_prefix(0, 0, 3), Ok(&[1, 2, 3, 4, 5, 6][..]));
    }

    #[test]
    fn supports_multi_batch_differing_lengths() {
        let mut cache = Q10KvCache::initialize(
            2,
            1,
            5,
            1,
            &[10, 11, 0, 20, 0, 0],
            3,
            &[0, 1, 77, 0, 88, 99],
            &[true, true, false, true, false, false],
            &[2, 1],
        )
        .unwrap();
        cache
            .append(
                &[12, 0, 21, 22],
                2,
                &[2, 90, 1, 2],
                &[true, false, true, true],
                &[3, 3],
            )
            .unwrap();
        assert_eq!(cache.valid_lengths(), &[3, 3]);
        assert_eq!(cache.visible_prefix(0, 0, 3), Ok(&[10, 11, 12][..]));
        assert_eq!(cache.visible_prefix(1, 0, 3), Ok(&[20, 21, 22][..]));
    }

    #[test]
    fn failed_batch_is_atomic() {
        let mut cache =
            Q10KvCache::initialize(2, 1, 4, 1, &[10, 20], 1, &[0, 0], &[true, true], &[1, 1])
                .unwrap();
        let error = cache.append(&[11, 21], 1, &[1, 2], &[true, true], &[2, 2]);
        assert!(matches!(
            error,
            Err(KvCacheError::PositionGap { batch: 1, .. })
        ));
        assert_eq!(cache.valid_lengths(), &[1, 1]);
        assert_eq!(cache.visible_prefix(0, 0, 1), Ok(&[10][..]));
        assert_eq!(cache.visible_prefix(1, 0, 1), Ok(&[20][..]));
    }

    #[test]
    fn mask_failure_in_later_batch_is_atomic() {
        let mut cache =
            Q10KvCache::initialize(2, 1, 3, 1, &[10, 20], 1, &[0, 0], &[true, true], &[1, 1])
                .unwrap();
        assert_eq!(
            cache.append(&[11, 21], 1, &[1, 1], &[true, false], &[2, 2]),
            Err(KvCacheError::MissingActiveMask { batch: 1, query: 0 })
        );
        assert_eq!(cache.valid_lengths(), &[1, 1]);
        assert_eq!(cache.visible_prefix(0, 0, 1), Ok(&[10][..]));
        assert_eq!(cache.visible_prefix(1, 0, 1), Ok(&[20][..]));
    }

    #[test]
    fn true_mask_preserves_numerically_zero_valid_rows() {
        let mut cache =
            Q10KvCache::initialize(1, 1, 3, 1, &[0, 0], 2, &[0, 99], &[true, false], &[1]).unwrap();
        assert_eq!(cache.visible_prefix(0, 0, 1), Ok(&[0][..]));
        cache.append(&[0], 1, &[1], &[true], &[2]).unwrap();
        assert_eq!(cache.valid_lengths(), &[2]);
        assert_eq!(cache.visible_prefix(0, 0, 2), Ok(&[0, 0][..]));
    }

    #[test]
    fn rejects_mask_length_holes_extra_bits_and_length_disagreement() {
        assert_eq!(
            Q10KvCache::initialize(1, 1, 4, 1, &[1, 2], 2, &[0, 1], &[true], &[2],).unwrap_err(),
            KvCacheError::LengthMismatch {
                tensor: "attention_mask",
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            Q10KvCache::initialize(
                1,
                1,
                4,
                1,
                &[1, 2, 3],
                3,
                &[0, 1, 2],
                &[true, false, true],
                &[3],
            )
            .unwrap_err(),
            KvCacheError::MissingActiveMask { batch: 0, query: 1 }
        );
        assert_eq!(
            Q10KvCache::initialize(1, 1, 4, 1, &[1, 0], 2, &[0, 99], &[true, true], &[1],)
                .unwrap_err(),
            KvCacheError::UnexpectedActiveMask { batch: 0, query: 1 }
        );

        let mut cache = Q10KvCache::initialize(1, 1, 4, 1, &[1], 1, &[0], &[true], &[1]).unwrap();
        assert_eq!(
            cache.append(&[2], 1, &[1], &[false], &[2]),
            Err(KvCacheError::MissingActiveMask { batch: 0, query: 0 })
        );
        assert_eq!(
            cache.append(&[0], 1, &[99], &[true], &[1]),
            Err(KvCacheError::UnexpectedActiveMask { batch: 0, query: 0 })
        );
        assert_eq!(cache.valid_lengths(), &[1]);
        assert_eq!(cache.visible_prefix(0, 0, 1), Ok(&[1][..]));
    }

    #[test]
    fn rejects_capacity_gap_rewind_overwrite_and_padding() {
        let mut cache = Q10KvCache::initialize(1, 1, 2, 1, &[7], 1, &[0], &[true], &[1]).unwrap();
        assert!(matches!(
            cache.append(&[8, 9], 2, &[1, 2], &[true, true], &[3]),
            Err(KvCacheError::CapacityExceeded { .. })
        ));
        assert!(matches!(
            cache.append(&[8], 1, &[2], &[true], &[2]),
            Err(KvCacheError::PositionGap { .. })
        ));
        assert!(matches!(
            cache.append(&[0], 1, &[0], &[false], &[0]),
            Err(KvCacheError::ValidLengthRewind { .. })
        ));
        assert!(matches!(
            cache.append(&[8], 1, &[0], &[true], &[2]),
            Err(KvCacheError::PositionOverwrite { .. })
        ));
        assert!(matches!(
            cache.append(&[8, 9], 2, &[1, 99], &[true, false], &[2]),
            Err(KvCacheError::NonZeroPadding { query: 1, .. })
        ));
        assert_eq!(cache.valid_lengths(), &[1]);
        assert_eq!(cache.visible_prefix(0, 0, 1), Ok(&[7][..]));
    }

    #[test]
    fn rejects_initial_padding_and_too_large_prefill() {
        assert!(matches!(
            Q10KvCache::initialize(1, 1, 4, 1, &[1, 2], 2, &[0, 9], &[true, false], &[1]),
            Err(KvCacheError::NonZeroPadding { .. })
        ));
        assert!(matches!(
            Q10KvCache::initialize(1, 1, 1, 1, &[1, 2], 2, &[0, 1], &[true, true], &[2]),
            Err(KvCacheError::CapacityExceeded { .. })
        ));
        assert!(matches!(
            Q10KvCache::initialize(1, 1, 4, 1, &[1], 1, &[0], &[true], &[2]),
            Err(KvCacheError::DeltaExceedsQuery { .. })
        ));
    }

    #[test]
    fn rejects_cheap_invalid_inputs_before_full_capacity_allocation() {
        let capacity = 1_000_000_000;
        assert_eq!(
            Q10KvCache::initialize(1, 1, capacity, 1, &[], 1, &[0], &[true], &[1],).unwrap_err(),
            KvCacheError::LengthMismatch {
                tensor: "current",
                expected: 1,
                actual: 0,
            }
        );
        assert_eq!(
            Q10KvCache::initialize(1, 1, capacity, 1, &[7], 1, &[0], &[false], &[1],).unwrap_err(),
            KvCacheError::MissingActiveMask { batch: 0, query: 0 }
        );
    }

    #[test]
    fn visible_prefix_enforces_explicit_bound_and_indices() {
        let cache =
            Q10KvCache::initialize(1, 1, 4, 1, &[1, 2], 2, &[0, 1], &[true, true], &[2]).unwrap();
        assert_eq!(
            cache.visible_prefix(0, 0, 1),
            Err(KvCacheError::VisibleLimitExceeded {
                batch: 0,
                valid_length: 2,
                maximum_visible: 1,
            })
        );
        assert_eq!(
            cache.visible_prefix(1, 0, 4),
            Err(KvCacheError::BatchOutOfRange { batch: 1 })
        );
        assert_eq!(
            cache.visible_prefix(0, 1, 4),
            Err(KvCacheError::HeadOutOfRange { head: 1 })
        );
    }

    #[test]
    fn scores_use_visible_extent_with_full_storage_stride() {
        let cache = Q10KvCache::initialize(
            2,
            1,
            4,
            1,
            &[10, 20, 0, 30, 0, 0],
            3,
            &[0, 1, 99, 0, 99, 99],
            &[true, true, false, true, false, false],
            &[2, 1],
        )
        .unwrap();
        let output = cache
            .attention_scores_q20(
                &[2, 3],
                [2, 1, 1, 1],
                3,
                &[1, 0],
                &[1, 1],
                AttentionScoreQ20Policy::new(6, 3).unwrap(),
            )
            .unwrap();
        assert_eq!(output.shape(), [2, 1, 1, 3]);
        assert_eq!(output.scores(), &[20, 40, 0, 90, 0, 0]);
        assert_eq!(output.allowed(), &[true, true, false, true, false, false]);
    }

    #[test]
    fn scores_reject_state_longer_than_visible_view() {
        let cache = Q10KvCache::initialize(
            1,
            1,
            4,
            1,
            &[1, 2],
            2,
            &[0, 1],
            &[true, true],
            &[2],
        )
        .unwrap();
        assert!(matches!(
            cache.attention_scores_q20(
                &[1],
                [1, 1, 1, 1],
                1,
                &[0],
                &[1],
                AttentionScoreQ20Policy::new(1, 1).unwrap(),
            ),
            Err(AttentionScoreQ20Error::ValidLengthOutOfRange {
                batch: 0,
                valid_length: 2,
                capacity: 1,
            })
        ));
    }

    #[test]
    fn drop_path_zeroizes_plaintext_and_lengths() {
        let mut cache =
            Q10KvCache::initialize(1, 1, 2, 2, &[1, -2], 1, &[0], &[true], &[1]).unwrap();
        cache.zeroize_contents();
        assert!(cache.storage.iter().all(|value| *value == 0));
        assert!(cache.valid_lengths.iter().all(|value| *value == 0));
    }
}
