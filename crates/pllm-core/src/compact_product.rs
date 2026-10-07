//! Bounded Compact-inspired complete SiLU × up numeric profile.
//! Public density-weighted dyadic refinement replaces the paper's annealing;
//! Q12/Horner rounding and the ±8 tails are explicit PLLM adaptations.
use crate::{compact::COMPACT_SOURCE_SHA256, fixed_point::div_round_ties_even};
use sha2::{Digest, Sha256};

pub const CALIBRATION_BINS: usize = 1024;
pub const COORDINATE_SCALE: i64 = 1 << 20;
pub const OUTPUT_SCALE: i64 = 1 << 12;
pub const TAIL_Q12: i32 = 8 << 12;
pub const PROFILE: &str = "pllm.numeric.compact_product_q18_q12.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub lower: i32,
    pub upper: i32,
    /// Normalized-coordinate power coefficients, encoded Q12 × 2^20.
    pub coefficients: [i64; 4],
}

impl Piece {
    fn evaluate(&self, gate_q12: i32, degree: u8) -> i64 {
        let t = i64::from(2 * gate_q12 - self.lower - self.upper) * COORDINATE_SCALE
            / i64::from(self.upper - self.lower);
        let mut value = self.coefficients[usize::from(degree)];
        for index in (0..usize::from(degree)).rev() {
            value = div_round_ties_even(value * t, COORDINATE_SCALE) + self.coefficients[index];
        }
        div_round_ties_even(value, COORDINATE_SCALE)
    }
}

#[derive(Clone, Debug)]
pub struct Profile {
    pieces: Box<[Piece]>,
    degree: u8,
    digest: [u8; 32],
}

fn fit_piece(lower: i32, upper: i32, degree: u8) -> Result<Piece, String> {
    let nodes = usize::from(degree) + 1;
    let center = f64::from(lower + upper) / 2.0;
    let radius = f64::from(upper - lower) / 2.0;
    let mut chebyshev = [0.0; 4];
    for node in 0..nodes {
        let theta = (2 * node + 1) as f64 * std::f64::consts::PI / (2 * nodes) as f64;
        let x = (center + radius * theta.cos()) / OUTPUT_SCALE as f64;
        let target = OUTPUT_SCALE as f64 * x / (1.0 + (-x).exp());
        for (index, coefficient) in chebyshev.iter_mut().enumerate().take(nodes) {
            *coefficient +=
                target * (index as f64 * theta).cos() * if index == 0 { 1.0 } else { 2.0 }
                    / nodes as f64;
        }
    }
    // T2=2t²−1, T3=4t³−3t. Conversion precedes coefficient quantization.
    let [c0, c1, c2, c3] = chebyshev;
    let coefficients = [c0 - c2, c1 - 3.0 * c3, 2.0 * c2, 4.0 * c3]
        .map(|x| (x * COORDINATE_SCALE as f64).round_ties_even() as i64);
    // For |t|≤1, every Horner intermediate is bounded by this sum plus
    // three rounding units. Multiplication by Q20 then fits signed i64.
    if coefficients.iter().map(|x| x.unsigned_abs()).sum::<u64>()
        + 4
        + (COORDINATE_SCALE as u64 / 2)
        >= 1 << 38
    {
        return Err("Compact coefficient bound exceeds the signed arithmetic contract".into());
    }
    Ok(Piece {
        lower,
        upper,
        coefficients,
    })
}

fn weighted_error(piece: &Piece, degree: u8, counts: &[u32]) -> f64 {
    // Fixed public bin centers; no checkpoint/private values enter this fit.
    counts
        .iter()
        .enumerate()
        .filter_map(|(index, &count)| {
            let gate = -TAIL_Q12 + 32 + 64 * index as i32;
            if gate < piece.lower || gate >= piece.upper {
                return None;
            }
            let x = f64::from(gate) / OUTPUT_SCALE as f64;
            let target = OUTPUT_SCALE as f64 * x / (1.0 + (-x).exp());
            Some((piece.evaluate(gate, degree) as f64 - target).abs() * (f64::from(count) + 1.0))
        })
        .sum()
}

impl Profile {
    pub fn fit(counts: &[u32], max_pieces: u8, degree: u8) -> Result<Self, String> {
        if counts.len() != CALIBRATION_BINS
            || counts.iter().any(|&x| x > 1_000_000)
            || counts.iter().map(|&x| u64::from(x)).sum::<u64>() > 100_000_000
            || !(1..=16).contains(&max_pieces)
            || !(2..=3).contains(&degree)
        {
            return Err(
                "Compact requires bounded public counts, 1–16 pieces and degree 2 or 3".into(),
            );
        }
        let mut pieces = vec![fit_piece(-TAIL_Q12, TAIL_Q12, degree)?];
        while pieces.len() < usize::from(max_pieces) {
            let mut best: Option<(f64, usize, Piece, Piece)> = None;
            for (index, piece) in pieces.iter().enumerate() {
                if piece.upper - piece.lower < 128 {
                    continue;
                }
                let midpoint = (piece.lower + piece.upper) / 2;
                let left = fit_piece(piece.lower, midpoint, degree)?;
                let right = fit_piece(midpoint, piece.upper, degree)?;
                let gain = weighted_error(piece, degree, counts)
                    - weighted_error(&left, degree, counts)
                    - weighted_error(&right, degree, counts);
                if gain > 0.0 && best.as_ref().is_none_or(|(old, ..)| gain > *old) {
                    best = Some((gain, index, left, right));
                }
            }
            let Some((_, index, left, right)) = best else {
                break;
            };
            pieces.splice(index..=index, [left, right]);
        }
        let mut hash = Sha256::new();
        hash.update(PROFILE.as_bytes());
        hash.update(COMPACT_SOURCE_SHA256.as_bytes());
        hash.update([max_pieces, degree, pieces.len() as u8]);
        for count in counts {
            hash.update(count.to_le_bytes());
        }
        for piece in &pieces {
            hash.update(piece.lower.to_le_bytes());
            hash.update(piece.upper.to_le_bytes());
            for coefficient in piece.coefficients {
                hash.update(coefficient.to_le_bytes());
            }
        }
        Ok(Self {
            pieces: pieces.into_boxed_slice(),
            degree,
            digest: hash.finalize().into(),
        })
    }

    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }
    pub fn degree(&self) -> u8 {
        self.degree
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn validate_inputs(gate: i32, up: i32) -> Result<(), String> {
        if !(-(48 << 18)..=48 << 18).contains(&gate) || !(-(128 << 18)..=128 << 18).contains(&up) {
            return Err("Compact Q18 gate/up inputs must fit public ±48/±128 ranges".into());
        }
        Ok(())
    }

    pub fn evaluate(&self, gate: i32, up: i32) -> Result<i32, String> {
        Self::validate_inputs(gate, up)?;
        let gate = div_round_ties_even(i64::from(gate), 64) as i32;
        let activated = if gate <= -TAIL_Q12 {
            0
        } else if gate >= TAIL_Q12 {
            i64::from(gate)
        } else {
            self.pieces
                .iter()
                .find(|p| gate >= p.lower && gate < p.upper)
                .expect("complete dyadic partition")
                .evaluate(gate, self.degree)
        };
        let up = div_round_ties_even(i64::from(up), 64);
        Ok(div_round_ties_even(activated * up, OUTPUT_SCALE) as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_profiles_cover_domain_tails_and_signed_ties_without_clipping() {
        for degree in [2, 3] {
            let p = Profile::fit(&[0; CALIBRATION_BINS], 16, degree).unwrap();
            assert_eq!(p.pieces[0].lower, -TAIL_Q12);
            assert_eq!(p.pieces.last().unwrap().upper, TAIL_Q12);
            for pair in p.pieces.windows(2) {
                assert_eq!(pair[0].upper, pair[1].lower);
            }
            for piece in p.pieces() {
                assert!((piece.upper - piece.lower).unsigned_abs().is_power_of_two());
            }
            for gate in (-TAIL_Q12..=TAIL_Q12).step_by(7) {
                let x = f64::from(gate) / 4096.0;
                let actual = p.evaluate(gate << 6, 1 << 18).unwrap();
                assert!((f64::from(actual) / 4096.0 - x / (1.0 + (-x).exp())).abs() < 0.008);
            }
            assert_eq!(p.evaluate(-48 << 18, 128 << 18).unwrap(), 0);
            assert_eq!(p.evaluate(48 << 18, -128 << 18).unwrap(), -6144 << 12);
            assert!(p.evaluate((48 << 18) + 1, 0).is_err());
            assert!(p.evaluate(0, (128 << 18) + 1).is_err());
        }
    }

    #[test]
    fn public_calibration_and_geometry_bind_identity() {
        let p = Profile::fit(&[0; CALIBRATION_BINS], 8, 3).unwrap();
        let mut counts = [0; CALIBRATION_BINS];
        counts[512] = 10_000;
        assert_ne!(p.digest(), Profile::fit(&counts, 8, 3).unwrap().digest());
        assert_ne!(p.digest(), Profile::fit(&counts, 8, 2).unwrap().digest());
        assert!(Profile::fit(&counts[..1023], 8, 3).is_err());
        assert!(Profile::fit(&counts, 17, 3).is_err());
        counts.fill(1_000_000);
        assert!(Profile::fit(&counts, 8, 3).is_err());
    }
}
