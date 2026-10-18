//! Deterministic mathematical oracle *conditioned on supplied random mask*.
//! Not deterministic exact truncation, not a production protocol or RNG.
//! Uses wide Euclidean arithmetic, independently of the protocol's high-bit
//! correction equation and wrapping share computations.

use super::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Distribution {
    pub floor: i64,
    pub upper: i64,
    pub upper_count: u64,
    pub mask_residue_count: u64,
}

pub fn distribution(x: i64, ring_bits: u8, shift: u8) -> Result<Distribution, Error> {
    if !(3..=64).contains(&ring_bits) || shift == 0 || shift > ring_bits - 2 {
        return Err(Error::Bounds);
    }
    let quarter = 1_i128 << (ring_bits - 2);
    if i128::from(x) < -quarter || i128::from(x) >= quarter {
        return Err(Error::Bounds);
    }
    let unit = 1_i128 << shift;
    let floor = i128::from(x).div_euclid(unit) as i64;
    let remainder = i128::from(x).rem_euclid(unit) as u64;
    Ok(Distribution {
        floor,
        upper: floor + i64::from(remainder != 0),
        upper_count: remainder,
        mask_residue_count: unit as u64,
    })
}

/// Output for the given full ring mask. Exact multiples return floor for every
/// mask; otherwise upper occurs for exactly `x mod 2^shift` low mask residues.
pub fn truncate(x: i64, ring_bits: u8, shift: u8, mask: u64) -> Result<i64, Error> {
    let law = distribution(x, ring_bits, shift)?;
    let low = mask & (law.mask_residue_count - 1);
    let carry = u128::from(low) + u128::from(law.upper_count) >= u128::from(law.mask_residue_count);
    Ok(law.floor + i64::from(carry))
}
