//! Leakage witness for an exposed, unshared shifted polynomial.
//!
//! Deliberately weakened design, not opaque party-local correlations.
//! F(G,U)=((G-a)^2+L(G-a))(U-b) exposes c_GU=L-2a and c_G2=-b.
use serde::Serialize;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PolynomialShiftWitness {
    pub ring_bits: u8,
    pub gate_residue_bits: u8,
    pub gate_residue: u64,
    pub up_value: u64,
}

pub fn evaluate(
    ring_bits: u8,
    linear: u64,
    masked_gate: u64,
    masked_up: u64,
    coefficient_gu: u64,
    coefficient_g2: u64,
) -> Result<PolynomialShiftWitness, String> {
    if !matches!(ring_bits, 8 | 16 | 24 | 32 | 64) {
        return Err("polynomial shift witness requires a bounded power-of-two ring".into());
    }
    let mask = u64::MAX >> (64 - ring_bits);
    if [
        linear,
        masked_gate,
        masked_up,
        coefficient_gu,
        coefficient_g2,
    ]
    .iter()
    .any(|v| *v > mask)
    {
        return Err("polynomial shift witness contains an invalid residue".into());
    }
    let twice_mask = linear.wrapping_sub(coefficient_gu) & mask;
    if twice_mask & 1 != 0 {
        return Err("coefficients are inconsistent with the stated shifted polynomial".into());
    }
    Ok(PolynomialShiftWitness {
        ring_bits,
        gate_residue_bits: ring_bits - 1,
        gate_residue: masked_gate.wrapping_sub(twice_mask / 2) & (mask >> 1),
        up_value: masked_up.wrapping_add(coefficient_g2) & mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exhaustive_small_ring_exposes_up_and_all_but_one_gate_bit() {
        for x in 0u64..256 {
            for a in 0u64..256 {
                let y = (x * 79 + 7) & 255;
                let b = (a * 31 + x) & 255;
                let witness = evaluate(
                    8,
                    0,
                    (x + a) & 255,
                    (y + b) & 255,
                    0u64.wrapping_sub(2 * a) & 255,
                    0u64.wrapping_sub(b) & 255,
                )
                .unwrap();
                assert_eq!(witness.gate_residue, x & 127);
                assert_eq!(witness.up_value, y);
            }
        }
    }
    #[test]
    fn wrap64_and_inconsistent_claims() {
        let (x, a, y, b) = (u64::MAX, 11u64, u64::MAX - 1, u64::MAX);
        let witness = evaluate(
            64,
            256,
            x.wrapping_add(a),
            y.wrapping_add(b),
            256u64.wrapping_sub(2 * a),
            0u64.wrapping_sub(b),
        )
        .unwrap();
        assert_eq!(witness.up_value, y);
        assert_eq!(witness.gate_residue, x & (u64::MAX >> 1));
        assert!(evaluate(24, 256, 18, 12, 233, 0).is_err());
        assert!(evaluate(24, 256, 1 << 24, 12, 234, 0).is_err());
    }
}
