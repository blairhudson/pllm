//! Exact greedy bounds for signed-i8 head bitplanes. The oracle retains all
//! weights: it does not implement private remote refinement or reduce residency.
use crate::kernels::dot_prefix;

pub struct ProgressiveHead {
    weights: Vec<i8>,
    scales: Vec<f32>,
    columns: usize,
}
pub struct HeadQuery {
    pub winner: usize,
    pub rounds: Vec<(u8, usize, usize)>,
}

impl ProgressiveHead {
    pub fn new(weights: &[u8], scales: &[f32], columns: usize) -> Result<Self, String> {
        if !(1..=16384).contains(&columns)
            || scales.is_empty()
            || scales.len() > 262144
            || scales.len().checked_mul(columns) != Some(weights.len())
            || weights.len() > 256 << 20
            || scales
                .iter()
                .any(|s| !s.is_finite() || *s <= 0.0 || *s > 1e6)
        {
            return Err("head weights/scales exceed finite bounded domain".into());
        }
        Ok(Self {
            weights: weights.iter().map(|&w| w as i8).collect(),
            scales: scales.to_vec(),
            columns,
        })
    }

    pub fn query(
        &self,
        input: &[i8],
        activation_scale: f32,
        first_bits: u8,
    ) -> Result<HeadQuery, String> {
        if input.len() != self.columns
            || !(1..=8).contains(&first_bits)
            || !activation_scale.is_finite()
            || !(0.0..=1e6).contains(&activation_scale)
        {
            return Err("head query differs from finite bounded contract".into());
        }
        let negative: i64 = input.iter().map(|&x| i64::from(x).min(0)).sum();
        let positive: i64 = input.iter().map(|&x| i64::from(x).max(0)).sum();
        let mut candidates: Vec<usize> = (0..self.scales.len()).collect();
        let mut rounds = Vec::new();
        for bits in first_bits..=8 {
            let shift = 8 - bits;
            let mask = (1_i64 << shift) - 1;
            let mut best = (f32::NEG_INFINITY, usize::MAX);
            let mut intervals = Vec::with_capacity(candidates.len());
            for &row in &candidates {
                let w = &self.weights[row * self.columns..(row + 1) * self.columns];
                let dot = dot_prefix(w, input, shift) << shift;
                // Same monotone f32 cast/multiplication order as dequantization.
                let lower = ((dot + mask * negative) as f32 * activation_scale) * self.scales[row];
                let upper = ((dot + mask * positive) as f32 * activation_scale) * self.scales[row];
                if lower > best.0 || (lower == best.0 && row < best.1) {
                    best = (lower, row);
                }
                intervals.push((row, upper));
            }
            let scanned = candidates.len();
            candidates = intervals
                .into_iter()
                .filter_map(|(row, upper)| {
                    (upper > best.0 || (upper == best.0 && row <= best.1)).then_some(row)
                })
                .collect();
            rounds.push((bits, scanned, candidates.len()));
            if candidates.len() == 1 {
                return Ok(HeadQuery {
                    winner: candidates[0],
                    rounds,
                });
            }
        }
        Err("full-precision greedy ordering did not resolve".into())
    }

    /// Independently addressable public residual records for a PIR cost probe.
    pub fn residual_table(&self, prefix_bits: u8) -> Result<Vec<u8>, String> {
        if !(1..=7).contains(&prefix_bits) {
            return Err("residual prefix must be in [1,7]".into());
        }
        let bits = 8 - usize::from(prefix_bits);
        let bytes = (bits * self.columns).div_ceil(8);
        let mask = (1_u16 << bits) - 1;
        let mut result = vec![0; self.scales.len() * bytes];
        for (row, weights) in self.weights.chunks_exact(self.columns).enumerate() {
            for (column, &value) in weights.iter().enumerate() {
                let offset = column * bits;
                let part = (u16::from(value as u8) & mask) << (offset % 8);
                result[row * bytes + offset / 8] |= part as u8;
                if offset % 8 + bits > 8 {
                    result[row * bytes + offset / 8 + 1] |= (part >> 8) as u8;
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_precision_preserves_signed_float_greedy_and_ties() {
        let w: Vec<u8> = (0..37 * 49).map(|i| ((i * 71 + 19) % 256) as u8).collect();
        let scales: Vec<f32> = (0..37).map(|i| (i + 1) as f32 / 97.0).collect();
        let head = ProgressiveHead::new(&w, &scales, 49).unwrap();
        for seed in 0..17 {
            let x: Vec<i8> = (0..49)
                .map(|i| ((i * 31 + seed * 19) % 256) as u8 as i8)
                .collect();
            let scores: Vec<f32> = w
                .chunks_exact(49)
                .zip(&scales)
                .map(|(row, scale)| {
                    let dot: i32 = row
                        .iter()
                        .zip(&x)
                        .map(|(&a, &b)| i32::from(a as i8) * i32::from(b))
                        .sum();
                    (dot as f32 * 0.013) * scale
                })
                .collect();
            let expected = (0..37)
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a)))
                .unwrap();
            for bits in 1..=8 {
                assert_eq!(head.query(&x, 0.013, bits).unwrap().winner, expected);
            }
            for shift in 0..8 {
                for row in head.weights.chunks_exact(49) {
                    let scalar: i64 = row
                        .iter()
                        .zip(&x)
                        .map(|(&a, &b)| i64::from(a >> shift) * i64::from(b))
                        .sum();
                    assert_eq!(dot_prefix(row, &x, shift), scalar);
                }
            }
        }
        assert_eq!(head.query(&[0; 49], 1.0, 1).unwrap().winner, 0);
        assert!(head.query(&[0; 49], f32::NAN, 1).is_err());
        assert_eq!(head.residual_table(5).unwrap().len(), 37 * 19);
        let tied = ProgressiveHead::new(&[7, 8], &[8.0, 7.0], 1).unwrap();
        assert_eq!(tied.query(&[1], 1.0, 4).unwrap().winner, 0);
    }
}
