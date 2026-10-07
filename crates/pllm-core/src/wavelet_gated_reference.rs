//! Bounded public Haar SiLU table and complete Q18 -> Q9 gated numeric contract.
//! Mathematical samples and ranges are fixed independently of checkpoint data.
//! This is an exact-rounding PLLM adaptation, not Curl's probabilistic truncation.
use crate::fixed_point::div_round_ties_even;
use sha2::{Digest, Sha256};

pub const INPUT_FRACTION: u8 = 18;
pub const OUTPUT_FRACTION: u8 = 9;
pub const INDEX_FRACTION: u8 = 5;
pub const TABLE_ROWS: usize = 4096;
pub const TABLE_OFFSET: i32 = 64;
pub const GATE_BOUND: i32 = 48;
pub const UP_BOUND: i32 = 128;
pub const MAX_LEVELS: u8 = 8;
pub const ID: &str = "pllm.numeric.haar_silu_gated.q18_q9.v1";

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Profile,
    Domain,
    Overflow,
}

/// Immutable Haar low-pass coefficients. Each discarded-detail block has one
/// ties-even rounded integer mean; rounding is not repeated between DWT levels.
pub struct Profile {
    levels: u8,
    table: Vec<i32>,
    digest: [u8; 32],
}

impl Profile {
    pub fn new(levels: u8) -> Result<Self, Error> {
        if levels > MAX_LEVELS {
            return Err(Error::Profile);
        }
        let original: Vec<i64> = (0..TABLE_ROWS)
            .map(|i| {
                let x = i as f64 / f64::from(1 << INDEX_FRACTION) - f64::from(TABLE_OFFSET);
                (x / (1.0 + (-x).exp()) * f64::from(1 << OUTPUT_FRACTION)).round_ties_even() as i64
            })
            .collect();
        let table: Vec<i32> = original
            .chunks_exact(1 << levels)
            .map(|group| div_round_ties_even(group.iter().sum(), group.len() as i64) as i32)
            .collect();
        // Even the complete table, not merely its admitted gate subset, proves
        // that activation_Q9 * up_Q9 stays in the signed 32-bit domain.
        if table.iter().any(|&x| {
            i64::from(x).unsigned_abs() * ((UP_BOUND as u64) << OUTPUT_FRACTION) >= 1 << 31
        }) {
            return Err(Error::Overflow);
        }
        let mut hash = Sha256::new();
        hash.update(ID);
        hash.update([INPUT_FRACTION, OUTPUT_FRACTION, INDEX_FRACTION, levels]);
        for value in [TABLE_OFFSET, GATE_BOUND, UP_BOUND, TABLE_ROWS as i32] {
            hash.update(value.to_le_bytes());
        }
        for value in &table {
            hash.update(value.to_le_bytes());
        }
        Ok(Self {
            levels,
            table,
            digest: hash.finalize().into(),
        })
    }

    pub fn levels(&self) -> u8 {
        self.levels
    }
    pub fn table(&self) -> &[i32] {
        &self.table
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn index_shift(&self) -> u8 {
        INPUT_FRACTION - INDEX_FRACTION + self.levels
    }

    pub fn validate_inputs(gate_q18: i32, up_q18: i32) -> Result<(), Error> {
        if gate_q18.unsigned_abs() > (GATE_BOUND << INPUT_FRACTION) as u32
            || up_q18.unsigned_abs() > (UP_BOUND << INPUT_FRACTION) as u32
        {
            return Err(Error::Domain);
        }
        Ok(())
    }

    pub fn encode(value: f32, bound: i32) -> Result<i32, Error> {
        if ![GATE_BOUND, UP_BOUND].contains(&bound)
            || !value.is_finite()
            || value.abs() > bound as f32
        {
            return Err(Error::Domain);
        }
        Ok((f64::from(value) * f64::from(1 << INPUT_FRACTION)).round_ties_even() as i32)
    }

    pub fn evaluate(&self, gate_q18: i32, up_q18: i32) -> Result<i32, Error> {
        Self::validate_inputs(gate_q18, up_q18)?;
        let index = ((gate_q18 + (TABLE_OFFSET << INPUT_FRACTION)) >> self.index_shift()) as usize;
        let up = div_round_ties_even(i64::from(up_q18), 1 << (INPUT_FRACTION - OUTPUT_FRACTION));
        let product =
            i32::try_from(i64::from(self.table[index]) * up).map_err(|_| Error::Overflow)?;
        i32::try_from(div_round_ties_even(
            i64::from(product),
            1 << OUTPUT_FRACTION,
        ))
        .map_err(|_| Error::Overflow)
    }

    pub fn evaluate_f32(&self, gate: f32, up: f32) -> Result<f32, Error> {
        let result = self.evaluate(Self::encode(gate, GATE_BOUND)?, Self::encode(up, UP_BOUND)?)?;
        Ok(result as f32 / (1 << OUTPUT_FRACTION) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_public_table_and_extreme_products_are_bounded() {
        let original = Profile::new(0).unwrap();
        for levels in 0..=MAX_LEVELS {
            let p = Profile::new(levels).unwrap();
            assert_eq!(p.table().len(), TABLE_ROWS >> levels);
            for (i, group) in original.table().chunks_exact(1 << levels).enumerate() {
                let mean = group.iter().map(|&x| f64::from(x)).sum::<f64>() / group.len() as f64;
                assert_eq!(p.table()[i], mean.round_ties_even() as i32);
            }
            for g in -(GATE_BOUND << INDEX_FRACTION)..=GATE_BOUND << INDEX_FRACTION {
                for delta in [-1, 0, 1] {
                    let gate = (g << (INPUT_FRACTION - INDEX_FRACTION)) + delta;
                    if gate.unsigned_abs() > (GATE_BOUND << INPUT_FRACTION) as u32 {
                        continue;
                    }
                    for up in [
                        -UP_BOUND << INPUT_FRACTION,
                        -257,
                        -256,
                        256,
                        257,
                        UP_BOUND << INPUT_FRACTION,
                    ] {
                        let index = ((gate + (TABLE_OFFSET << INPUT_FRACTION)) as usize)
                            / (1 << p.index_shift());
                        let u = (f64::from(up) / 512.0).round_ties_even();
                        let expected =
                            (f64::from(p.table()[index]) * u / 512.0).round_ties_even() as i32;
                        assert_eq!(p.evaluate(gate, up).unwrap(), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_out_of_range_and_binds_compression_identity() {
        let p = Profile::new(0).unwrap();
        assert_ne!(p.digest(), Profile::new(2).unwrap().digest());
        assert!(Profile::new(MAX_LEVELS + 1).is_err());
        for (g, u) in [
            (i32::MIN, 0),
            ((GATE_BOUND << 18) + 1, 0),
            (0, (UP_BOUND << 18) + 1),
        ] {
            assert_eq!(p.evaluate(g, u), Err(Error::Domain));
        }
        for value in [f32::NAN, f32::INFINITY, 48.01, -48.01] {
            assert_eq!(p.evaluate_f32(value, 0.0), Err(Error::Domain));
        }
    }
}
