//! Public, immutable SiLU affine-spline profiles for bounded numeric research.
//! Fitting uses only the mathematical function on a fixed uniform grid. No
//! checkpoint, calibration trace or private prompt chooses a range or piece.
use crate::fixed_point::div_round_ties_even;
use sha2::{Digest, Sha256};

pub const INPUT_FRACTION: u8 = 16;
pub const COEFFICIENT_FRACTION: u8 = 14;
pub const GATE_BOUND: i32 = 48;
pub const UP_BOUND: i32 = 128;
pub const TAIL: i32 = 8;
pub const ID: &str = "pllm.numeric.silu_affine_gated.reference.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Profile,
    Domain,
    Overflow,
}

#[derive(Clone, Debug)]
pub struct Profile {
    fraction: u8,
    pieces: u16,
    // Canonical unsigned starts partition all of Z/(2^32). Values are signed
    // affine coefficients (slope Q14, intercept Q(fraction+14)). The public
    // accepted numeric domain is narrower; secret range enforcement is absent.
    intervals: Vec<(u32, [u32; 2])>,
    digest: [u8; 32],
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

impl Profile {
    pub fn new(fraction: u8, pieces: u16) -> Result<Self, Error> {
        if ![8, 9].contains(&fraction) || ![16, 64].contains(&pieces) {
            return Err(Error::Profile);
        }
        let scale = 1i32 << fraction;
        let step = 2 * TAIL * scale / i32::from(pieces);
        let mut intervals = vec![
            ((TAIL * scale) as u32, [1 << COEFFICIENT_FRACTION, 0]),
            (1 << 31, [0, 0]),
        ];
        for i in 0..i32::from(pieces) {
            let lo = -TAIL * scale + i * step;
            let x = f64::from(lo) / f64::from(scale);
            let y = f64::from(lo + step) / f64::from(scale);
            let a = ((silu(y) - silu(x)) / (y - x) * f64::from(1 << COEFFICIENT_FRACTION))
                .round_ties_even() as i32;
            let b = ((silu(x) - f64::from(a) * x / f64::from(1 << COEFFICIENT_FRACTION))
                * f64::from(scale << COEFFICIENT_FRACTION))
            .round_ties_even() as i32;
            intervals.push((lo as u32, [a as u32, b as u32]));
        }
        intervals.sort_unstable_by_key(|(start, _)| *start);
        let mut hash = Sha256::new();
        hash.update(ID);
        hash.update([INPUT_FRACTION, COEFFICIENT_FRACTION, fraction]);
        hash.update(pieces.to_le_bytes());
        for bound in [GATE_BOUND, UP_BOUND, TAIL] {
            hash.update(bound.to_le_bytes());
        }
        for (start, payload) in &intervals {
            hash.update(start.to_le_bytes());
            for value in payload {
                hash.update(value.to_le_bytes());
            }
        }
        Ok(Self {
            fraction,
            pieces,
            intervals,
            digest: hash.finalize().into(),
        })
    }
    pub fn fraction(&self) -> u8 {
        self.fraction
    }
    pub fn pieces(&self) -> u16 {
        self.pieces
    }
    pub fn intervals(&self) -> &[(u32, [u32; 2])] {
        &self.intervals
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn encode(value: f32, bound: i32) -> Result<i32, Error> {
        if ![GATE_BOUND, UP_BOUND].contains(&bound)
            || !value.is_finite()
            || f64::from(value).abs() > f64::from(bound)
        {
            return Err(Error::Domain);
        }
        Ok((f64::from(value) * f64::from(1 << INPUT_FRACTION)).round_ties_even() as i32)
    }

    pub fn coefficients(&self, encoded: u32) -> [u32; 2] {
        let index = self
            .intervals
            .partition_point(|(start, _)| *start <= encoded)
            - 1;
        self.intervals[index].1
    }

    pub fn evaluate(&self, gate_q16: i32, up_q16: i32) -> Result<i32, Error> {
        if gate_q16.unsigned_abs() > (GATE_BOUND << INPUT_FRACTION) as u32
            || up_q16.unsigned_abs() > (UP_BOUND << INPUT_FRACTION) as u32
        {
            return Err(Error::Domain);
        }
        let divisor = 1i64 << (INPUT_FRACTION - self.fraction);
        let g = div_round_ties_even(i64::from(gate_q16), divisor);
        let u = div_round_ties_even(i64::from(up_q16), divisor);
        let [a, b] = self.coefficients(g as u32).map(|v| i64::from(v as i32));
        let linear = i32::try_from(a * g + b).map_err(|_| Error::Overflow)?;
        let activation = div_round_ties_even(i64::from(linear), 1 << COEFFICIENT_FRACTION);
        let product = i32::try_from(activation * u).map_err(|_| Error::Overflow)?;
        i32::try_from(div_round_ties_even(i64::from(product), 1 << self.fraction))
            .map_err(|_| Error::Overflow)
    }

    pub fn evaluate_f32(&self, gate: f32, up: f32) -> Result<f32, Error> {
        let output = self.evaluate(Self::encode(gate, GATE_BOUND)?, Self::encode(up, UP_BOUND)?)?;
        Ok(output as f32 / (1u32 << self.fraction) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_encoded_domain_has_no_overflow_and_explicit_tails() {
        for fraction in [8, 9] {
            for pieces in [16, 64] {
                let p = Profile::new(fraction, pieces).unwrap();
                let scale = 1 << fraction;
                assert_eq!(p.intervals.len(), usize::from(pieces) + 2);
                assert_eq!(p.intervals[0].0, 0);
                for g in -GATE_BOUND * scale..=GATE_BOUND * scale {
                    for up in [-UP_BOUND, -1, 0, 1, UP_BOUND] {
                        let result = p.evaluate(g << (16 - fraction), up << 16).unwrap();
                        if g >= TAIL * scale {
                            assert_eq!(result, g * up);
                        }
                        if g < -TAIL * scale {
                            assert_eq!(result, 0);
                        }
                    }
                    let got = f64::from(p.evaluate(g << (16 - fraction), 1 << 16).unwrap())
                        / f64::from(scale);
                    // |SiLU''| < 1; uniform interpolation error <= h²/8.
                    // Coefficient/input/output rounding and tail error extra.
                    let bound =
                        (16.0 / f64::from(pieces)).powi(2) / 8.0 + 0.003 + 1.0 / f64::from(scale);
                    assert!((got - silu(f64::from(g) / f64::from(scale))).abs() < bound);
                }
            }
        }
    }
    #[test]
    fn rejects_ranges_nonfinite_and_unknown_profiles_without_clipping() {
        let p = Profile::new(9, 64).unwrap();
        for (g, u) in [
            (48.01, 0.0),
            (-48.01, 0.0),
            (0.0, 128.01),
            (f32::NAN, 1.0),
            (1.0, f32::INFINITY),
        ] {
            assert_eq!(p.evaluate_f32(g, u), Err(Error::Domain));
        }
        assert_eq!(p.evaluate(i32::MIN, 0), Err(Error::Domain));
        assert!(Profile::new(10, 64).is_err());
        assert_ne!(p.digest(), Profile::new(8, 64).unwrap().digest());
        assert_ne!(p.digest(), Profile::new(9, 16).unwrap().digest());
    }
}
