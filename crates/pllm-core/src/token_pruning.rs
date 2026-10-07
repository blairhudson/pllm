//! Bounded attention-column scoring and stable complete-record compaction.
//!
//! Scores retain the exact Q16 numerator. Comparing against `threshold * H * N`
//! avoids division of additive shares (which is not a local ring operation).
//! This is a numeric reference, not a trained pruning policy or decoder pass.

pub const SCALE: u32 = 1 << 16;
pub const MAX_ATTENTION_ELEMENTS: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parameters {
    pub tokens: usize,
    pub heads: usize,
    pub features: usize,
    pub threshold_q16: u32,
    pub reduction_threshold_q16: u32,
}

impl Parameters {
    pub fn validate(self) -> Result<(), String> {
        if !(1..=256).contains(&self.tokens)
            || !(1..=32).contains(&self.heads)
            || !(1..=8192).contains(&self.features)
            || self.threshold_q16 >= self.reduction_threshold_q16
            || self.reduction_threshold_q16 > SCALE
        {
            return Err("invalid token-pruning dimensions or Q16 thresholds".into());
        }
        if self.attention_elements() > MAX_ATTENTION_ELEMENTS {
            return Err("token-pruning attention exceeds its bounded input size".into());
        }
        Ok(())
    }

    pub fn attention_elements(self) -> usize {
        self.heads * self.tokens * self.tokens
    }

    pub fn score_threshold(self, reduction: bool) -> u32 {
        (self.heads * self.tokens) as u32
            * if reduction {
                self.reduction_threshold_q16
            } else {
                self.threshold_q16
            }
    }

    pub fn score_sums(self, attention: &[u32]) -> Result<Vec<u32>, String> {
        self.validate()?;
        if attention.len() != self.attention_elements() || attention.iter().any(|&v| v > SCALE) {
            return Err("attention must contain bounded nonnegative Q16 entries".into());
        }
        let mut sums = vec![0_u32; self.tokens];
        for row in attention.chunks_exact(self.tokens) {
            for (sum, &value) in sums.iter_mut().zip(row) {
                *sum += value;
            }
        }
        Ok(sums)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Output {
    pub words: Vec<u32>,
    pub retained: usize,
    /// Absent when no pruning occurred, as required by the source reduction step.
    pub high_degree: Option<Vec<bool>>,
}

/// Independent clear filter, preserving every retained 32-bit payload word.
pub fn reference(p: Parameters, attention: &[u32], words: &[u32]) -> Result<Output, String> {
    let scores = p.score_sums(attention)?;
    if words.len() != p.tokens * p.features {
        return Err("token words differ from the declared matrix shape".into());
    }
    let mut selected = Vec::new();
    let mut high_degree = Vec::new();
    for (row, &score) in scores.iter().enumerate() {
        if score > p.score_threshold(false) {
            selected.extend_from_slice(&words[row * p.features..(row + 1) * p.features]);
            high_degree.push(score > p.score_threshold(true));
        }
    }
    let retained = high_degree.len();
    Ok(Output {
        words: selected,
        retained,
        high_degree: (retained < p.tokens).then_some(high_degree),
    })
}

/// Figure 14's shrinking adjacent-swap schedule, without allocating records.
pub fn swaps(tokens: usize, retained: usize) -> Result<usize, String> {
    if tokens == 0 || tokens > 256 || retained > tokens {
        return Err("invalid pruning-count projection".into());
    }
    let removed = tokens - retained;
    Ok(removed * (2 * tokens - removed - 1) / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_thresholds_preserve_order_complete_words_and_empty_results() {
        let p = Parameters {
            tokens: 4,
            heads: 2,
            features: 2,
            threshold_q16: 10,
            reduction_threshold_q16: 20,
        };
        let attention = [10, 11, 20, 21].repeat(8);
        let words = [0, u32::MAX, 0x8000_0000, 3, 4, 5, 6, 7];
        assert_eq!(
            reference(p, &attention, &words).unwrap(),
            Output {
                words: words[2..].to_vec(),
                retained: 3,
                high_degree: Some(vec![false, false, true]),
            }
        );
        assert_eq!(reference(p, &[0; 32], &words).unwrap().retained, 0);
        assert_eq!(reference(p, &[21; 32], &words).unwrap().high_degree, None);
        assert!(reference(p, &[SCALE + 1; 32], &words).is_err());
        assert!(reference(p, &attention, &words[..7]).is_err());
    }

    #[test]
    fn exact_numerator_avoids_sharewise_division_and_count_is_explicit_leakage() {
        // Uniform ring shares can wrap; independently dividing them is wrong.
        let left = u32::MAX;
        let right = 4_u32;
        assert_ne!(
            (left / 2).wrapping_add(right / 2),
            left.wrapping_add(right) / 2
        );
        // With one row, the public retained count reveals the comparison bit.
        let p = Parameters {
            tokens: 1,
            heads: 1,
            features: 1,
            threshold_q16: 10,
            reduction_threshold_q16: 20,
        };
        assert_eq!(reference(p, &[10], &[123]).unwrap().retained, 0);
        assert_eq!(reference(p, &[11], &[123]).unwrap().retained, 1);
        // A marker cannot be inserted into an occupied payload bit.
        assert_eq!(0x8000_0000_u32.wrapping_add(1 << 31), 0);
        assert_eq!(swaps(8, 6).unwrap(), 13);
        assert_eq!(swaps(8, 0).unwrap(), 28);
        assert!(swaps(usize::MAX, 0).is_err());
    }
}
