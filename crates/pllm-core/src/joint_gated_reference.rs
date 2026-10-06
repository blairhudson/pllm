//! Public numeric gate for a joint SiLU-times-up function with one final rounding.
//! No sharing, FSS keys, secret range checks or executable decoder component.
use sha2::{Digest, Sha256};

pub const ID: &str = "pllm.numeric.joint_silu_gated.reference.v1";
pub const COEFFICIENT_FRACTION: u8 = 28;
pub const GATE_BOUND: i32 = 48;
pub const UP_BOUND: i32 = 128;
pub const TAIL: i32 = 16;

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
    intervals: Vec<(i32, [i64; 2])>,
    digest: [u8; 32],
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn round_even(value: i128, shift: u8) -> i128 {
    let divisor = 1i128 << shift;
    let q = value.div_euclid(divisor);
    let r = value.rem_euclid(divisor);
    q + i128::from(2 * r > divisor || (2 * r == divisor && q & 1 != 0))
}

impl Profile {
    pub fn new(fraction: u8, pieces: u16) -> Result<Self, Error> {
        if ![12, 16].contains(&fraction) || ![512, 2048].contains(&pieces) {
            return Err(Error::Profile);
        }
        let scale = 1i32 << fraction;
        let coefficient_scale = (1u64 << COEFFICIENT_FRACTION) as f64;
        let step = 2 * TAIL * scale / i32::from(pieces);
        let mut intervals = vec![(-GATE_BOUND * scale, [0, 0])];
        for i in 0..i32::from(pieces) {
            let lo = -TAIL * scale + i * step;
            let x = f64::from(lo) / f64::from(scale);
            let y = f64::from(lo + step) / f64::from(scale);
            let a = ((silu(y) - silu(x)) / (y - x) * coefficient_scale).round_ties_even() as i64;
            let b = ((silu(x) - a as f64 * x / coefficient_scale)
                * coefficient_scale
                * f64::from(scale))
            .round_ties_even() as i64;
            intervals.push((lo, [a, b]));
        }
        intervals.push((TAIL * scale, [1 << COEFFICIENT_FRACTION, 0]));
        let mut hash = Sha256::new();
        hash.update(ID);
        hash.update([fraction, COEFFICIENT_FRACTION]);
        hash.update(pieces.to_le_bytes());
        for bound in [GATE_BOUND, UP_BOUND, TAIL] {
            hash.update(bound.to_le_bytes());
        }
        for (start, coefficients) in &intervals {
            hash.update(start.to_le_bytes());
            for coefficient in coefficients {
                hash.update(coefficient.to_le_bytes());
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
    pub fn intervals(&self) -> &[(i32, [i64; 2])] {
        &self.intervals
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    fn coefficients(&self, gate: i32) -> [i64; 2] {
        self.intervals[self.intervals.partition_point(|(start, _)| *start <= gate) - 1].1
    }

    fn numerator(&self, gate: i32, up: i32) -> Result<i128, Error> {
        if gate.unsigned_abs() > (GATE_BOUND << self.fraction) as u32
            || up.unsigned_abs() > (UP_BOUND << self.fraction) as u32
        {
            return Err(Error::Domain);
        }
        let [a, b] = self.coefficients(gate).map(i128::from);
        // Admitted public bounds fit i128; checked arithmetic also guards changes.
        a.checked_mul(i128::from(gate))
            .and_then(|v| v.checked_add(b))
            .and_then(|v| v.checked_mul(i128::from(up)))
            .ok_or(Error::Overflow)
    }

    pub fn evaluate(&self, gate: i32, up: i32) -> Result<i32, Error> {
        i32::try_from(round_even(
            self.numerator(gate, up)?,
            COEFFICIENT_FRACTION + self.fraction,
        ))
        .map_err(|_| Error::Overflow)
    }

    pub fn evaluate_f32(&self, gate: f32, up: f32) -> Result<f32, Error> {
        let encode = |x: f32, bound| {
            if !x.is_finite() || f64::from(x).abs() > f64::from(bound) {
                return Err(Error::Domain);
            }
            Ok((f64::from(x) * f64::from(1 << self.fraction)).round_ties_even() as i32)
        };
        let value = self.evaluate(encode(gate, GATE_BOUND)?, encode(up, UP_BOUND)?)?;
        Ok(value as f32 / (1u32 << self.fraction) as f32)
    }

    /// Bilinear on each public interval: absolute extrema occur at corners.
    pub fn maximum_numerator_absolute(&self) -> i128 {
        self.intervals
            .iter()
            .enumerate()
            .flat_map(|(i, (start, _))| {
                let end = self
                    .intervals
                    .get(i + 1)
                    .map_or(GATE_BOUND << self.fraction, |next| next.0 - 1);
                [*start, end]
            })
            .map(|g| self.numerator(g, UP_BOUND << self.fraction).unwrap().abs())
            .max()
            .unwrap()
    }
}

#[derive(Debug)]
pub struct AlgebraAudit {
    pub checked_masked_pairs: usize,
    pub missing_wrap_wrong_outputs: usize,
    pub sharewise_rounding: (i128, i128),
    pub recovered_gate_up: (u32, u32),
}

/// Tiny exhaustive algebra and privacy counterexamples, not a protocol simulator.
pub fn algebra_audit() -> AlgebraAudit {
    let mut checked = 0;
    let mut wrong = 0;
    for g in -4i128..4 {
        let (a, b) = if g < 0 { (3, -2) } else { (5, 1) };
        for u in -4i128..4 {
            for rg in 0..8 {
                for ru in 0..8 {
                    let z = (g + rg).rem_euclid(8);
                    let w = (u + ru).rem_euclid(8);
                    let cg = (g + rg - z) / 8;
                    let cu = (u + ru - w) / 8;
                    let naive = (a * z + b - a * rg) * (w - ru);
                    let corrected = naive
                        + 8 * (a * cg * (w - ru) + cu * (a * z + b - a * rg))
                        + 64 * a * cg * cu;
                    let exact = (a * g + b) * u;
                    assert_eq!(corrected, exact);
                    assert_eq!(round_even(corrected, 2), round_even(exact, 2));
                    wrong += usize::from(round_even(naive, 2) != round_even(exact, 2));
                    checked += 1;
                }
            }
        }
    }
    // Identity-tail numerator: (z-rg)(w-ru). Public linear coefficients expose
    // both masks even over the exact wrapping ring, independently of FSS details.
    let (rg, ru) = (17u32, 29u32);
    let (z, w) = (3u32.wrapping_add(rg), 5u32.wrapping_add(ru));
    let (coefficient_z, coefficient_w) = (ru.wrapping_neg(), rg.wrapping_neg());
    AlgebraAudit {
        checked_masked_pairs: checked,
        missing_wrap_wrong_outputs: wrong,
        sharewise_rounding: (round_even(3, 2) + round_even(-1, 2), round_even(2, 2)),
        recovered_gate_up: (z.wrapping_add(coefficient_w), w.wrapping_add(coefficient_z)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signs_ties_and_single_round_are_distinct_from_separate_rounding() {
        assert_eq!(
            (-7..=7).map(|n| round_even(n, 1)).collect::<Vec<_>>(),
            [-4, -3, -2, -2, -2, -1, 0, 0, 0, 1, 2, 2, 2, 3, 4]
        );
        let p = Profile::new(12, 512).unwrap();
        assert_eq!(p.evaluate(1, 2 << 12), Ok(1));
        let [a, b] = p.coefficients(1).map(i128::from);
        assert_eq!(round_even(a + b, 28) * 2, 2);
    }

    #[test]
    fn every_piece_boundary_tail_and_signed_extremum_is_bounded() {
        for f in [12, 16] {
            for pieces in [512, 2048] {
                let p = Profile::new(f, pieces).unwrap();
                assert!(p.maximum_numerator_absolute() > i128::from(i64::MAX));
                assert!(p.maximum_numerator_absolute() < 1i128 << 80);
                let limit = GATE_BOUND << f;
                for &(start, _) in p.intervals() {
                    for g in [start - 1, start, start + 1, -limit, limit] {
                        if g.abs() > limit {
                            continue;
                        }
                        for u in [-UP_BOUND << f, -1, 0, 1, UP_BOUND << f] {
                            let got = p.evaluate(g, u).unwrap();
                            let expected = silu(f64::from(g) / f64::from(1 << f)) * f64::from(u)
                                / f64::from(1 << f);
                            // |SiLU''| < 1, h²/8 interpolation; public tail and
                            // coefficient rounding bounds; one encoded output ulp.
                            let error = ((32.0 / f64::from(pieces)).powi(2) / 8.0
                                + 0.000002
                                + 32.0 / 268435456.0)
                                * 128.0
                                + 1.0 / f64::from(1 << f);
                            assert!((f64::from(got) / f64::from(1 << f) - expected).abs() < error);
                            if g < -TAIL << f {
                                assert_eq!(got, 0);
                            }
                            if g >= TAIL << f {
                                assert_eq!(
                                    i128::from(got),
                                    round_even(i128::from(g) * i128::from(u), f)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_profiles_inputs_and_domain_do_not_clip() {
        let p = Profile::new(16, 2048).unwrap();
        for (g, u) in [
            (48.01, 1.0),
            (-48.01, 1.0),
            (1.0, 128.01),
            (f32::NAN, 0.0),
            (0.0, f32::INFINITY),
        ] {
            assert_eq!(p.evaluate_f32(g, u), Err(Error::Domain));
        }
        assert_eq!(p.evaluate(i32::MIN, 0), Err(Error::Domain));
        assert!(Profile::new(9, 2048).is_err());
        assert!(Profile::new(16, 1024).is_err());
        assert_ne!(p.digest(), Profile::new(12, 2048).unwrap().digest());
        assert_ne!(p.digest(), Profile::new(16, 512).unwrap().digest());
    }

    #[test]
    fn wrap_corrections_rounding_and_hidden_coefficients_are_required() {
        let a = algebra_audit();
        assert_eq!(a.checked_masked_pairs, 4096);
        assert!(a.missing_wrap_wrong_outputs > 0);
        assert_eq!(a.sharewise_rounding, (1, 0));
        assert_eq!(a.recovered_gate_up, (3, 5));
    }
}
