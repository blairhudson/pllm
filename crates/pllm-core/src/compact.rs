//! Bounded numeric reference for Compact-inspired, density-aware Q7 SiLU fits.
//!
//! Fitting requires public offline calibration. Evaluation selects a piece in
//! plaintext and is **not** a private MPC interval-selection protocol.

use std::{error::Error, f64::consts::PI, fmt};

use sha2::{Digest, Sha256};

use crate::{
    activation::{SILU_QUADRATIC_Q7_MAX, SILU_QUADRATIC_Q7_MIN, SILU_QUADRATIC_Q7_SCALE},
    fixed_point::div_round_ties_even,
};

pub const COMPACT_SILU_Q7_PROFILE: &str = "pllm.numeric.compact_silu_q7.reference.v1";
pub const COMPACT_SOURCE_SHA256: &str =
    "e4497a81055e60cca1c462dac5ae04b4504c5f3fc76e392e170cc32bf8fee86c";
pub const COMPACT_Q7_POINTS: usize = 257;
pub const COMPACT_MAX_PIECES: u8 = 8;
pub const COMPACT_MAX_CALIBRATION_COUNT: u32 = 1_000_000;
pub const COMPACT_MAX_CALIBRATION_TOTAL: u64 = 100_000_000;
const COEFFICIENT_SCALE: i64 = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactQ7Error {
    CalibrationLength { actual: usize },
    CalibrationCount { index: usize, count: u32 },
    CalibrationTotal { total: u64 },
    PieceBudget { budget: u8 },
    InputOutOfRange { value: i16 },
}

impl fmt::Display for CompactQ7Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for CompactQ7Error {}

/// A public Chebyshev polynomial on one inclusive interval of encoded Q7.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactQ7Piece {
    lower: i16,
    upper: i16,
    /// T0, T1, T2 coefficients; each is an encoded-Q7 output scaled by 2^20.
    coefficients_q20: [i64; 3],
}

impl CompactQ7Piece {
    pub const fn lower(&self) -> i16 {
        self.lower
    }

    pub const fn upper(&self) -> i16 {
        self.upper
    }

    pub const fn coefficients_q20(&self) -> [i64; 3] {
        self.coefficients_q20
    }

    fn evaluate(&self, value: i16) -> i16 {
        let span = i64::from(self.upper - self.lower);
        let centered = i64::from(2 * value - self.lower - self.upper);
        let t = div_round_ties_even(centered * COEFFICIENT_SCALE, span);
        let t_squared = div_round_ties_even(t * t, COEFFICIENT_SCALE);
        let [constant, linear, quadratic] = self.coefficients_q20;
        let result = constant
            + div_round_ties_even(linear * t, COEFFICIENT_SCALE)
            + div_round_ties_even(
                quadratic * (2 * t_squared - COEFFICIENT_SCALE),
                COEFFICIENT_SCALE,
            );
        // Each fitted coefficient comes from the continuous SiLU of [-1, 1].
        // The bounded polynomial's intermediate and final values fit i64/i16.
        div_round_ties_even(result, COEFFICIENT_SCALE) as i16
    }
}

/// Immutable, source- and calibration-bound reference profile. Not executable
/// through the protected compiler or a `ComponentRef`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactQ7Profile {
    pieces: Box<[CompactQ7Piece]>,
    requested_pieces: u8,
    digest: [u8; 32],
    calibration_digest: [u8; 32],
    weighted_error_numerator: u64,
    weighted_error_denominator: u64,
    maximum_encoded_error: u16,
}

impl CompactQ7Profile {
    pub fn pieces(&self) -> &[CompactQ7Piece] {
        &self.pieces
    }

    pub const fn requested_pieces(&self) -> u8 {
        self.requested_pieces
    }

    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub const fn calibration_digest(&self) -> [u8; 32] {
        self.calibration_digest
    }

    pub const fn source_sha256(&self) -> &'static str {
        COMPACT_SOURCE_SHA256
    }

    /// Weighted absolute encoded-Q7 error, including one unit per input.
    pub const fn weighted_error(&self) -> (u64, u64) {
        (
            self.weighted_error_numerator,
            self.weighted_error_denominator,
        )
    }

    pub const fn maximum_encoded_error(&self) -> u16 {
        self.maximum_encoded_error
    }

    /// Public reference selection; must never be used on provider-private input.
    pub fn evaluate(&self, value: i16) -> Result<i16, CompactQ7Error> {
        if !(SILU_QUADRATIC_Q7_MIN..=SILU_QUADRATIC_Q7_MAX).contains(&value) {
            return Err(CompactQ7Error::InputOutOfRange { value });
        }
        let piece = self
            .pieces
            .iter()
            .find(|piece| value >= piece.lower && value <= piece.upper)
            .expect("fitter covers the complete signed-Q7 domain");
        Ok(piece.evaluate(value))
    }
}

fn exact_silu_encoded(value: f64) -> f64 {
    let x = value / f64::from(SILU_QUADRATIC_Q7_SCALE);
    f64::from(SILU_QUADRATIC_Q7_SCALE) * x / (1.0 + (-x).exp())
}

fn fit_piece(lower: i16, upper: i16) -> CompactQ7Piece {
    let center = (f64::from(lower) + f64::from(upper)) / 2.0;
    let radius = (f64::from(upper) - f64::from(lower)) / 2.0;
    let mut coefficient = [0.0; 3];
    for node in 0..3 {
        let t = ((2 * node + 1) as f64 * PI / 6.0).cos();
        let target = exact_silu_encoded(center + radius * t);
        coefficient[0] += target / 3.0;
        coefficient[1] += 2.0 * target * t / 3.0;
        coefficient[2] += 2.0 * target * (2.0 * t * t - 1.0) / 3.0;
    }
    CompactQ7Piece {
        lower,
        upper,
        coefficients_q20: coefficient
            .map(|value| (value * COEFFICIENT_SCALE as f64).round_ties_even() as i64),
    }
}

fn piece_error(piece: CompactQ7Piece, counts: &[u32]) -> u64 {
    (piece.lower..=piece.upper)
        .map(|encoded| {
            let index = (encoded - SILU_QUADRATIC_Q7_MIN) as usize;
            let oracle = exact_silu_encoded(f64::from(encoded)).round_ties_even() as i16;
            u64::from(piece.evaluate(encoded).abs_diff(oracle)) * (u64::from(counts[index]) + 1)
        })
        .sum()
}

/// Fit at most eight degree-two Chebyshev pieces on signed Q7 [-1, 1].
///
/// The paper's input-density weighting is adapted to explicit, *public*
/// offline Q7 counts rather than assuming Qwen RMSNorm inputs are Gaussian.
/// Each encoded value gets one pseudocount, making the measured tails visible.
/// Exhaustive integer-error reduction chooses splits, with leftmost ties.
pub fn fit_compact_silu_q7(
    public_calibration_counts: &[u32],
    max_pieces: u8,
) -> Result<CompactQ7Profile, CompactQ7Error> {
    if public_calibration_counts.len() != COMPACT_Q7_POINTS {
        return Err(CompactQ7Error::CalibrationLength {
            actual: public_calibration_counts.len(),
        });
    }
    if !(1..=COMPACT_MAX_PIECES).contains(&max_pieces) {
        return Err(CompactQ7Error::PieceBudget { budget: max_pieces });
    }
    let mut total = 0u64;
    for (index, &count) in public_calibration_counts.iter().enumerate() {
        if count > COMPACT_MAX_CALIBRATION_COUNT {
            return Err(CompactQ7Error::CalibrationCount { index, count });
        }
        total += u64::from(count);
        if total > COMPACT_MAX_CALIBRATION_TOTAL {
            return Err(CompactQ7Error::CalibrationTotal { total });
        }
    }

    let mut pieces = vec![fit_piece(SILU_QUADRATIC_Q7_MIN, SILU_QUADRATIC_Q7_MAX)];
    while pieces.len() < usize::from(max_pieces) {
        let mut best: Option<(u64, usize, CompactQ7Piece, CompactQ7Piece)> = None;
        for (position, piece) in pieces.iter().copied().enumerate() {
            if piece.upper - piece.lower < 3 {
                continue;
            }
            let previous = piece_error(piece, public_calibration_counts);
            for split in (piece.lower + 1)..=(piece.upper - 2) {
                let left = fit_piece(piece.lower, split);
                let right = fit_piece(split + 1, piece.upper);
                let new_error = piece_error(left, public_calibration_counts)
                    + piece_error(right, public_calibration_counts);
                if new_error < previous {
                    let improvement = previous - new_error;
                    if best
                        .as_ref()
                        .is_none_or(|(value, _, _, _)| improvement > *value)
                    {
                        best = Some((improvement, position, left, right));
                    }
                }
            }
        }
        let Some((_, position, left, right)) = best else {
            break;
        };
        pieces.splice(position..=position, [left, right]);
    }

    let mut calibration_hasher = Sha256::new();
    calibration_hasher.update(b"pllm.compact.q7.public-calibration.v1\0");
    for count in public_calibration_counts {
        calibration_hasher.update(count.to_le_bytes());
    }
    let calibration_digest: [u8; 32] = calibration_hasher.finalize().into();
    let mut profile_hasher = Sha256::new();
    profile_hasher.update(COMPACT_SILU_Q7_PROFILE.as_bytes());
    profile_hasher.update([0, max_pieces]);
    profile_hasher.update(COMPACT_SOURCE_SHA256.as_bytes());
    profile_hasher.update(calibration_digest);
    profile_hasher.update([pieces.len() as u8]);
    for piece in &pieces {
        profile_hasher.update(piece.lower.to_le_bytes());
        profile_hasher.update(piece.upper.to_le_bytes());
        for coefficient in piece.coefficients_q20 {
            profile_hasher.update(coefficient.to_le_bytes());
        }
    }
    let digest = profile_hasher.finalize().into();
    let weighted_error_numerator = pieces
        .iter()
        .copied()
        .map(|piece| piece_error(piece, public_calibration_counts))
        .sum();
    let maximum_encoded_error = pieces
        .iter()
        .flat_map(|piece| piece.lower..=piece.upper)
        .map(|encoded| {
            let oracle = exact_silu_encoded(f64::from(encoded)).round_ties_even() as i16;
            let piece = pieces
                .iter()
                .find(|piece| (piece.lower..=piece.upper).contains(&encoded))
                .expect("fitter covers the complete signed-Q7 domain");
            piece.evaluate(encoded).abs_diff(oracle)
        })
        .max()
        .unwrap_or(0);
    Ok(CompactQ7Profile {
        pieces: pieces.into_boxed_slice(),
        requested_pieces: max_pieces,
        digest,
        calibration_digest,
        weighted_error_numerator,
        weighted_error_denominator: total + COMPACT_Q7_POINTS as u64,
        maximum_encoded_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::silu_quadratic_q7;

    fn uncalibrated() -> [u32; COMPACT_Q7_POINTS] {
        [0; COMPACT_Q7_POINTS]
    }

    #[test]
    fn fitted_profile_covers_the_entire_encoded_domain_and_beats_quadratic_reference() {
        let profile = fit_compact_silu_q7(&uncalibrated(), 4).unwrap();
        assert!(!profile.pieces().is_empty());
        assert!(profile.pieces().len() <= 4);
        assert_eq!(profile.pieces()[0].lower(), -128);
        assert_eq!(profile.pieces().last().unwrap().upper(), 128);
        for window in profile.pieces().windows(2) {
            assert_eq!(window[0].upper() + 1, window[1].lower());
        }
        let mut quadratic_error = 0u64;
        assert!(profile.evaluate(-128).unwrap().abs_diff(-34) <= 1);
        assert_eq!(profile.evaluate(0), Ok(0));
        assert!(profile.evaluate(128).unwrap().abs_diff(94) <= 1);
        for encoded in -128..=128 {
            let oracle = exact_silu_encoded(f64::from(encoded)).round_ties_even() as i16;
            quadratic_error += u64::from(silu_quadratic_q7(encoded).unwrap().abs_diff(oracle));
            let result = profile.evaluate(encoded).unwrap();
            assert!(result.abs_diff(oracle) <= profile.maximum_encoded_error());
            assert!((-128..=128).contains(&result));
        }
        assert!(profile.weighted_error().0 < quadratic_error);
        assert!(profile.maximum_encoded_error() <= 2);
        assert_eq!(profile.weighted_error().1, 257);
    }

    #[test]
    fn public_density_affects_the_fit_and_all_inputs_remain_measured() {
        let uniform = fit_compact_silu_q7(&uncalibrated(), 2).unwrap();
        let mut skewed_counts = uncalibrated();
        for encoded in -100..=-80 {
            skewed_counts[(encoded + 128) as usize] = 1_000_000;
        }
        let adapted = fit_compact_silu_q7(&skewed_counts, 2).unwrap();
        let score = |profile: &CompactQ7Profile| {
            (-128..=128)
                .map(|encoded| {
                    let oracle = exact_silu_encoded(f64::from(encoded)).round_ties_even() as i16;
                    u64::from(profile.evaluate(encoded).unwrap().abs_diff(oracle))
                        * (u64::from(skewed_counts[(encoded + 128) as usize]) + 1)
                })
                .sum::<u64>()
        };
        assert!(score(&adapted) < score(&uniform));
        assert_ne!(adapted.pieces(), uniform.pieces());
        assert_eq!(adapted.weighted_error().0, score(&adapted));
        assert_eq!(adapted.weighted_error().1, 21_000_257);
        assert_ne!(uniform.calibration_digest(), adapted.calibration_digest());
        assert_ne!(uniform.digest(), adapted.digest());
        assert_eq!(adapted, fit_compact_silu_q7(&skewed_counts, 2).unwrap());
    }

    #[test]
    fn admission_and_tail_policy_fail_closed() {
        assert_eq!(
            fit_compact_silu_q7(&[], 1),
            Err(CompactQ7Error::CalibrationLength { actual: 0 })
        );
        assert_eq!(
            fit_compact_silu_q7(&uncalibrated(), 0),
            Err(CompactQ7Error::PieceBudget { budget: 0 })
        );
        assert_eq!(
            fit_compact_silu_q7(&uncalibrated(), 9),
            Err(CompactQ7Error::PieceBudget { budget: 9 })
        );
        let mut excess = uncalibrated();
        excess[0] = COMPACT_MAX_CALIBRATION_COUNT + 1;
        assert_eq!(
            fit_compact_silu_q7(&excess, 1),
            Err(CompactQ7Error::CalibrationCount {
                index: 0,
                count: 1_000_001,
            })
        );
        excess.fill(COMPACT_MAX_CALIBRATION_COUNT);
        assert!(matches!(
            fit_compact_silu_q7(&excess, 1),
            Err(CompactQ7Error::CalibrationTotal { total: 101_000_000 })
        ));
        let profile = fit_compact_silu_q7(&uncalibrated(), 2).unwrap();
        assert_eq!(
            profile.evaluate(-129),
            Err(CompactQ7Error::InputOutOfRange { value: -129 })
        );
        assert_eq!(
            profile.evaluate(129),
            Err(CompactQ7Error::InputOutOfRange { value: 129 })
        );
    }
}
