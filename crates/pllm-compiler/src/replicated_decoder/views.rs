//! Public run-length/strided activation references. These contain no shares.
//! Contiguous slices and broadcasts retain one descriptor per run, including
//! slices across primitive tiles. Logical iteration preserves the original order.
use std::ops::{Bound, RangeBounds};

pub(super) type Word = (usize, usize);

#[derive(Clone, Debug, PartialEq, Eq)]
struct Span {
    owner: usize,
    start: usize,
    stride: usize,
    end: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Words {
    spans: Vec<Span>,
    len: usize,
}

impl Words {
    /// Opt-in preallocated descriptor pool; legacy creation remains unchanged.
    pub(super) fn try_with_spans(capacity: usize) -> Result<Self, pllm_core::puma::Error> {
        let mut spans = Vec::new();
        spans
            .try_reserve_exact(capacity)
            .map_err(|_| pllm_core::puma::Error::Allocation)?;
        if spans.capacity() != capacity {
            return Err(pllm_core::puma::Error::Allocation);
        }
        Ok(Self { spans, len: 0 })
    }

    pub(super) fn span_pool_bytes(capacity: usize) -> Option<usize> {
        capacity.checked_mul(std::mem::size_of::<Span>())
    }

    pub(super) fn clear_preserving_capacity(&mut self) {
        self.spans.clear();
        self.len = 0;
    }

    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn metadata_bytes(&self) -> usize {
        self.spans.capacity() * std::mem::size_of::<Span>()
    }

    pub(super) fn owners(&self) -> impl Iterator<Item = usize> + '_ {
        self.spans.iter().map(|s| s.owner)
    }

    pub(super) fn push(&mut self, (owner, index): Word) {
        let base = self
            .spans
            .len()
            .checked_sub(2)
            .map_or(0, |i| self.spans[i].end);
        if let Some(last) = self.spans.last_mut() {
            let n = last.end - base;
            if last.owner == owner {
                if n == 1 && index >= last.start {
                    last.stride = index - last.start;
                    last.end += 1;
                    self.len += 1;
                    return;
                }
                if n.checked_mul(last.stride)
                    .and_then(|v| last.start.checked_add(v))
                    == Some(index)
                {
                    last.end += 1;
                    self.len += 1;
                    return;
                }
            }
        }
        self.len += 1;
        if self.spans.len() == self.spans.capacity() {
            self.spans.reserve_exact(self.spans.len().max(1));
        }
        self.spans.push(Span {
            owner,
            start: index,
            stride: 0,
            end: self.len,
        });
    }

    pub(super) fn at(&self, index: usize) -> Word {
        assert!(index < self.len);
        let i = self.spans.partition_point(|s| s.end <= index);
        let base = i.checked_sub(1).map_or(0, |j| self.spans[j].end);
        let s = &self.spans[i];
        (s.owner, s.start + (index - base) * s.stride)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = Word> + '_ {
        self.spans
            .iter()
            .scan(0, |base, s| {
                let n = s.end - *base;
                *base = s.end;
                Some((0..n).map(move |i| (s.owner, s.start + i * s.stride)))
            })
            .flatten()
    }

    pub(super) fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let start = match range.start_bound() {
            Bound::Included(n) => *n,
            Bound::Excluded(n) => n.checked_add(1).expect("checked public view"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(n) => n.checked_add(1).expect("checked public view"),
            Bound::Excluded(n) => *n,
            Bound::Unbounded => self.len,
        };
        assert!(start <= end && end <= self.len);
        let mut spans = Vec::new();
        let mut base = 0;
        for s in &self.spans {
            let lo = start.max(base);
            let hi = end.min(s.end);
            if lo < hi {
                if spans.len() == spans.capacity() {
                    spans.reserve_exact(spans.len().max(1));
                }
                spans.push(Span {
                    owner: s.owner,
                    start: s.start + (lo - base) * s.stride,
                    stride: s.stride,
                    end: hi - start,
                });
            }
            base = s.end;
            if base >= end {
                break;
            }
        }
        Self {
            spans,
            len: end - start,
        }
    }

    pub(super) fn chunks(&self, width: usize) -> impl Iterator<Item = Self> + '_ {
        assert!(width > 0);
        (0..self.len)
            .step_by(width)
            .map(move |start| self.slice(start..(start + width).min(self.len)))
    }

    pub(super) fn chunks_exact(&self, width: usize) -> impl Iterator<Item = Self> + '_ {
        assert!(width > 0 && self.len % width == 0);
        self.chunks(width)
    }

    pub(super) fn extend(&mut self, values: impl IntoIterator<Item = Word>) {
        for value in values {
            self.push(value);
        }
    }
}

impl FromIterator<Word> for Words {
    fn from_iter<T: IntoIterator<Item = Word>>(values: T) -> Self {
        let mut result = Self::new();
        result.extend(values);
        result
    }
}

impl PartialEq<Vec<Word>> for Words {
    fn eq(&self, other: &Vec<Word>) -> bool {
        self.len == other.len() && self.iter().eq(other.iter().copied())
    }
}

impl IntoIterator for Words {
    type Item = Word;
    type IntoIter = OwnedIter;
    fn into_iter(self) -> Self::IntoIter {
        OwnedIter {
            words: self,
            index: 0,
        }
    }
}

pub(super) struct OwnedIter {
    words: Words,
    index: usize,
}
impl Iterator for OwnedIter {
    type Item = Word;
    fn next(&mut self) -> Option<Word> {
        if self.index == self.words.len {
            return None;
        }
        let word = self.words.at(self.index);
        self.index += 1;
        Some(word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_broadcasts_permutations_and_partial_tiles_keep_exact_order() {
        let dense: Vec<_> = (0..3)
            .flat_map(|h| {
                (0..5).flat_map(move |r| (0..7).map(move |d| (r / 2, (r % 2 * 3 + h) * 7 + d)))
            })
            .collect();
        let repeated: Vec<_> = std::iter::repeat_n((9, 31), 261).collect();
        for values in [dense, repeated, (0..1031).map(|i| (1, i * 3 + 7)).collect()] {
            let words: Words = values.iter().copied().collect();
            assert_eq!(words, values);
            for (i, expected) in values.iter().enumerate() {
                assert_eq!(words.at(i), *expected);
            }
            for width in [1, 6, 128, 256] {
                for (actual, expected) in words.chunks(width).zip(values.chunks(width)) {
                    assert!(actual.iter().eq(expected.iter().copied()));
                }
            }
            for start in [0, 1, 6, values.len() - 1, values.len()] {
                for end in [start, values.len()] {
                    assert!(words
                        .slice(start..end)
                        .iter()
                        .eq(values[start..end].iter().copied()));
                }
            }
        }
        let words: Words = (0..151936).map(|i| (3, i)).collect();
        assert!(words.metadata_bytes() <= 128);
        assert!(words.slice(7..151935).metadata_bytes() <= 128);
    }

    #[test]
    fn maximal_indices_and_owner_changes_do_not_create_overflowed_spans() {
        let reference = vec![
            (1, 0),
            (1, usize::MAX),
            (1, usize::MAX),
            (0, 1),
            (0, 0),
            (0, 0),
        ];
        let words: Words = reference.iter().copied().collect();
        assert_eq!(words, reference);
        for (i, expected) in reference.into_iter().enumerate() {
            assert_eq!(words.at(i), expected);
        }
    }
}
