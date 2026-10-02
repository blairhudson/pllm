//! Bounded client-side shortlist scoring from a public compressed head index.
//! Residual radii are supplied by the trusted offline source-weight fitter.
use crate::kernels::{Executor, Matrix};
use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

pub struct Index {
    matrix: Matrix,
    executor: Executor,
    projection: Vec<f64>,
    profiles: Vec<[f64; 4]>,
    features: usize,
    projection_norm: f64,
}
#[derive(Clone, Copy)]
struct Entry {
    score: f64,
    upper: f64,
    id: usize,
}
impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.id == other.id
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl Index {
    /// Profiles contain (index row scale, upper coefficient-error norm, upper source norm).
    pub fn new(
        rows: usize,
        features: usize,
        rank: usize,
        weights: &[u8],
        projection: &[u8],
        profiles: &[u8],
    ) -> Result<Self, String> {
        if !(1..=262144).contains(&rows)
            || !(1..=4096).contains(&features)
            || !(1..=256).contains(&rank)
            || rank > features
            || weights.len() != rows * rank
            || projection.len() != rank * features * 8
            || profiles.len() != rows * 24
        {
            return Err("head index exceeds bounded dimensions".into());
        }
        let values = |bytes: &[u8]| -> Result<Vec<f64>, String> {
            let out: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
                .collect();
            if out.iter().any(|x| !x.is_finite() || x.abs() > 1e16) {
                return Err("head index contains an invalid coefficient".into());
            }
            Ok(out)
        };
        let projection = values(projection)?;
        let projection_norm = projection.iter().map(|v| v * v).sum::<f64>().sqrt() * (1.0 + 1e-10);
        let mut checked = Vec::with_capacity(rows);
        for (row, p) in values(profiles)?.chunks_exact(3).enumerate() {
            if p[0] <= 0.0 || p[1] < 0.0 || p[2] < 0.0 {
                return Err("head index profile must be positive/bounded".into());
            }
            let sum: u64 = weights[row * rank..(row + 1) * rank]
                .iter()
                .map(|&x| {
                    let v = x as i8 as i64;
                    (v * v) as u64
                })
                .sum();
            let norm = (sum as f64).sqrt() * p[0] * (1.0 + 1e-10);
            checked.push([p[0], p[1], p[2], norm]);
        }
        Ok(Self {
            matrix: Matrix::new(weights, rows, rank)?,
            executor: Executor::new(1, true)?,
            projection,
            profiles: checked,
            features,
            projection_norm,
        })
    }
    pub fn payload_bytes(&self) -> usize {
        self.matrix.weight_bytes() + self.projection.len() * 8 + self.profiles.len() * 32
    }
    /// Return fixed-count candidate IDs and a conservative upper bound for all
    /// excluded source rows, including projection/quantization and FP32 rounding.
    pub fn candidates(
        &self,
        input: &[u8],
        scale: f64,
        count: usize,
    ) -> Result<(Vec<u32>, f64), String> {
        if input.len() != self.features
            || !scale.is_finite()
            || !(0.0..=1e12).contains(&scale)
            || scale == 0.0
            || count == 0
            || count > self.profiles.len().min(4096)
        {
            return Err("head query differs from bounded index contract".into());
        }
        let x: Vec<f64> = input.iter().map(|&v| v as i8 as f64 * scale).collect();
        let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt() * (1.0 + 1e-10);
        let projected: Vec<f64> = self
            .projection
            .chunks_exact(self.features)
            .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
            .collect();
        // x = U(U^T x) + residual, for any supplied U. This avoids treating
        // the entire query norm as missing information when calibration spans it.
        let mut residual = x.clone();
        for (row, &coordinate) in self.projection.chunks_exact(self.features).zip(&projected) {
            for (value, &coefficient) in residual.iter_mut().zip(row) {
                *value -= coefficient * coordinate;
            }
        }
        let residual_norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt() * (1.0 + 1e-10)
            + 1e-10 * norm_x * (1.0 + self.projection_norm * self.projection_norm);
        let projected_norm = projected.iter().map(|v| v * v).sum::<f64>().sqrt() * (1.0 + 1e-10)
            + 1e-10 * self.projection_norm * norm_x;
        let maximum = projected.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let query_scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
        let q: Vec<i8> = projected
            .iter()
            .map(|v| (v / query_scale).round_ties_even().clamp(-127.0, 127.0) as i8)
            .collect();
        let error = projected
            .iter()
            .zip(&q)
            .map(|(a, &b)| (a - b as f64 * query_scale).powi(2))
            .sum::<f64>()
            .sqrt()
            + 1e-10 * self.projection_norm * norm_x;
        let dots = self.matrix.clear(&self.executor, &q, 1)?;
        let mut heap: BinaryHeap<Reverse<Entry>> = BinaryHeap::with_capacity(count);
        let mut outside = f64::NEG_INFINITY;
        for (id, (&dot, p)) in dots.iter().zip(&self.profiles).enumerate() {
            let score = dot as f64 * query_scale * p[0];
            let correction = p[2] * residual_norm + p[1] * projected_norm + p[3] * error;
            let upper = score
                + correction
                + 8.0 * f32::EPSILON as f64 * p[2] * norm_x
                + 1e-10 * (score.abs() + correction + 1.0);
            if !upper.is_finite() || !score.is_finite() {
                return Err("head index query overflow".into());
            }
            let entry = Entry { score, upper, id };
            if heap.len() < count {
                heap.push(Reverse(entry));
            } else if entry > heap.peek().unwrap().0 {
                outside = outside.max(heap.pop().unwrap().0.upper);
                heap.push(Reverse(entry));
            } else {
                outside = outside.max(upper);
            }
        }
        let ids = heap
            .into_sorted_vec()
            .into_iter()
            .map(|v| v.0.id as u32)
            .collect();
        Ok((ids, outside))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bytes(v: &[f64]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    #[test]
    fn identity_source_certifies_large_margin_and_breaks_ties_by_id() {
        let index = Index::new(
            4,
            2,
            2,
            &[127, 0, 0, 127, 129, 0, 0, 129],
            &bytes(&[1.0, 0.0, 0.0, 1.0]),
            &bytes(&[1.0, 0.0, 127.0].repeat(4)),
        )
        .unwrap();
        let (ids, bound) = index.candidates(&[127, 244], 1.0, 1).unwrap();
        assert_eq!(ids, [0]);
        assert!(bound < 16129.0);
        assert_eq!(index.candidates(&[0, 0], 1.0, 1).unwrap().0, [0]);
        assert!(index.candidates(&[0], 1.0, 1).is_err());
        assert!(index.candidates(&[0, 0], f64::NAN, 1).is_err());
        assert!(index.candidates(&[0, 0], 1.0, 5).is_err());
    }
}
