//! SIGMA §5.1 / Appendix F, Figure 7 order, instantiated for SiLU.
//! Fixed public Q12 profile; truncation is floor, not PLLM ties-to-even.
//! The residual uses abs(floor(x * 64)), clipped to 1023, then a public LUT.
//! Clear numeric research only, not an executable decoder component.

use sha2::{Digest, Sha256};

pub const FRACTION: u8 = 12;
pub const RING_BITS: u8 = 40;
pub const EFFECTIVE_GATE_BITS: u8 = 19;
pub const GATE_BOUND: i64 = 48;
pub const UP_BOUND: i64 = 128;

#[derive(Debug, PartialEq, Eq)]
pub struct Domain;

pub struct Profile {
    table: [i64; 1024],
    digest: [u8; 32],
}

impl Default for Profile {
    fn default() -> Self {
        Self::new()
    }
}
impl Profile {
    pub fn new() -> Self {
        let table = std::array::from_fn(|i| {
            let x = i as f64 / 64.0;
            (4096.0 * x / (1.0 + x.exp())).floor() as i64
        });
        let mut h = Sha256::new();
        h.update(b"pllm.residual_silu_reference.sigma_gpu_order.q12.v1\0");
        h.update([FRACTION, RING_BITS, EFFECTIVE_GATE_BITS]);
        h.update(GATE_BOUND.to_le_bytes());
        h.update(UP_BOUND.to_le_bytes());
        for x in table {
            h.update(x.to_le_bytes());
        }
        Self {
            table,
            digest: h.finalize().into(),
        }
    }
    pub fn table(&self) -> &[i64; 1024] {
        &self.table
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Both inputs Q24 as at the output of a fixed-point linear product.
    /// Public bounds prove both input GapARS and final product GapARS valid.
    pub fn evaluate(&self, gate_q24: i64, up_q24: i64) -> Result<i64, Domain> {
        if gate_q24.unsigned_abs() > (GATE_BOUND as u64) << 24
            || up_q24.unsigned_abs() > (UP_BOUND as u64) << 24
        {
            return Err(Domain);
        }
        let g = gate_q24.div_euclid(4096);
        let u = up_q24.div_euclid(4096);
        let index = g.div_euclid(64).unsigned_abs().min(1023) as usize;
        let activation = g.max(0) - self.table[index];
        Ok((activation * u).div_euclid(4096))
    }
    pub fn evaluate_f32(&self, gate: f32, up: f32) -> Result<f32, Domain> {
        if !gate.is_finite()
            || !up.is_finite()
            || gate.abs() > GATE_BOUND as f32
            || up.abs() > UP_BOUND as f32
        {
            return Err(Domain);
        }
        // This trusted clear bridge has no protected IEEE-to-Q conversion or
        // claimed network cost. Floor follows SIGMA's integer representation.
        let g = (f64::from(gate) * 16777216.0).floor() as i64;
        let u = (f64::from(up) * 16777216.0).floor() as i64;
        Ok((self.evaluate(g, u)? as f64 / 4096.0) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_gap_bounds_and_residual_table() {
        let p = Profile::new();
        assert_eq!(p.table[0], 0);
        assert_eq!(p.table[1023], 0);
        assert!(p.table.iter().all(|&x| (0..=1141).contains(&x)));
        // Exhaust every admitted Q12 gate, both extremal ups. Includes the
        // asymmetry introduced by moving TR ahead of absolute value.
        for gate in -(GATE_BOUND << 12)..=GATE_BOUND << 12 {
            let index = gate.div_euclid(64).unsigned_abs().min(1023) as usize;
            let a = gate.max(0) - p.table[index];
            assert!((a * (UP_BOUND << 12)).unsigned_abs() < 1 << (RING_BITS - 2));
            assert!(a.unsigned_abs() <= (GATE_BOUND << 12) as u64);
            for up in [-UP_BOUND << 24, UP_BOUND << 24] {
                assert!(p.evaluate(gate << 12, up).is_ok());
            }
        }
        assert!(p.evaluate((GATE_BOUND << 24) + 1, 0).is_err());
        assert!(p.evaluate_f32(f32::NAN, 0.0).is_err());
        assert_ne!(p.evaluate(-1, 1 << 24).unwrap(), 0);
    }
}
