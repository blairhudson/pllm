//! Model-neutral Q18 complete-row inequalities. INACTIVE: numbers here confer
//! no source, accepted-row, ordered-coordinate or compiler-work authority.
//! Production integration requires authentic post-guard energies (including
//! fallback rows), imported original matrix statistics and final output bounds.
use super::{ceil_sqrt, residual_energy, Error};

const Q: u128 = 262_144;
// Bounds SELECTED guarded-SiLU output only (low branch: 345_317);
// never bounds eager polynomial intermediates or low-input powers.
const SELECTED_SILU_C: u128 = 2_519_957;

pub(super) struct Projection {
    pub width: u128,
    pub input_energy: Option<u128>,
    pub input_magnitude: u128,
    pub row_squared_norm: Option<u128>,
    pub row_l1_squared_sum: Option<u128>,
    pub bias_magnitude: u128,
    pub final_magnitude: u128,
}

pub(super) struct Activation {
    pub width: u128,
    pub gate_energy: Option<u128>,
    pub final_magnitude: u128,
}

pub(super) struct Product {
    pub width: u128,
    pub activation_energy: Option<u128>,
    pub up_energy: Option<u128>,
    pub activation_magnitude: u128,
    pub up_magnitude: u128,
    pub final_magnitude: u128,
}

/// One named fixed frame. Distinct phase fields intentionally do not claim
/// sequential reuse across contraction, publication or caller frames.
#[derive(Default)]
pub(super) struct RowEnergyWorkspace {
    scalar: u128,
    projection_f: u128,
    projection_d: u128,
    projection_radius: u128,
    projection_cap: u128,
    activation: u128,
    product_k: u128,
    product_radius: u128,
    product_cap: u128,
    residual: u128,
}

fn mul(a: u128, b: u128) -> Result<u128, Error> {
    a.checked_mul(b).ok_or(Error::Bounds)
}
fn add(a: u128, b: u128) -> Result<u128, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn ceil_q(value: u128) -> Result<u128, Error> {
    // No value + Q - 1 intermediate overflow, including u128::MAX.
    add(value / Q, u128::from(value % Q != 0))
}
fn nonempty(width: u128) -> Result<(), Error> {
    if width == 0 { Err(Error::Bounds) } else { Ok(()) }
}

impl RowEnergyWorkspace {
    /// Complete raw dot -> ONE signed stochastic rescale -> additive bias.
    /// |rounding error| < 1 implies a whole-row radius increment sqrt(m).
    /// No transpose is justified by this inequality's imported R/S contract.
    /// F=min(mR,S), D=min(M²S,BF); E=min(mY²,[ceil(sqrt(D)/Q)
    /// +ceil_sqrt(m)+ceil_sqrt(mJ²)]²). Roots are rounded UP before ceil/Q.
    pub(super) fn projection(&mut self, p: &Projection) -> Result<Option<u128>, Error> {
        nonempty(p.width)?;
        let (Some(b), Some(r), Some(s)) =
            (p.input_energy, p.row_squared_norm, p.row_l1_squared_sum)
        else { return Ok(None) };
        self.projection_f = mul(p.width, r)?.min(s);
        self.scalar = mul(p.input_magnitude, p.input_magnitude)?;
        self.projection_d = mul(self.scalar, s)?.min(mul(b, self.projection_f)?);
        self.projection_radius = add(ceil_q(ceil_sqrt(self.projection_d))?, ceil_sqrt(p.width))?;
        self.scalar = mul(p.bias_magnitude, p.bias_magnitude)?;
        self.projection_radius = add(self.projection_radius, ceil_sqrt(mul(p.width, self.scalar)?))?;
        self.scalar = mul(p.final_magnitude, p.final_magnitude)?;
        self.projection_cap = mul(p.width, self.scalar)?;
        Ok(Some(self.projection_cap.min(mul(self.projection_radius, self.projection_radius)?)))
    }

    /// Only the EXISTING selected branch: 0 below -8Q; low [-8Q,-4Q),
    /// high [-4Q,4Q], identity above 4Q. C does not bound eager low-input powers.
    pub(super) fn activation(&mut self, a: &Activation) -> Result<Option<u128>, Error> {
        nonempty(a.width)?;
        let Some(eg) = a.gate_energy else { return Ok(None) };
        self.scalar = mul(SELECTED_SILU_C, SELECTED_SILU_C)?;
        self.activation = add(eg, mul(a.width, self.scalar)?)?;
        self.scalar = mul(a.final_magnitude, a.final_magnitude)?;
        self.activation = self.activation.min(mul(a.width, self.scalar)?);
        Ok(Some(self.activation))
    }

    /// Same ordered lane pairing, with no independence/spreading assumption.
    /// K=min(A²Eu,U²Ea,EaEu); E=min(mH²,[ceil(ceil_sqrt(K)/Q)
    /// +ceil_sqrt(m)]²). In particular, no division by width in K.
    pub(super) fn product(&mut self, p: &Product) -> Result<Option<u128>, Error> {
        nonempty(p.width)?;
        let (Some(ea), Some(eu)) = (p.activation_energy, p.up_energy)
        else { return Ok(None) };
        self.scalar = mul(p.activation_magnitude, p.activation_magnitude)?;
        self.product_k = mul(self.scalar, eu)?;
        self.scalar = mul(p.up_magnitude, p.up_magnitude)?;
        self.product_k = self.product_k.min(mul(self.scalar, ea)?).min(mul(ea, eu)?);
        self.product_radius = add(ceil_q(ceil_sqrt(self.product_k))?, ceil_sqrt(p.width))?;
        self.scalar = mul(p.final_magnitude, p.final_magnitude)?;
        self.product_cap = mul(p.width, self.scalar)?;
        Ok(Some(self.product_cap.min(mul(self.product_radius, self.product_radius)?)))
    }

    pub(super) fn residual(
        &mut self, existing: u128, left: Option<u128>, right: Option<u128>,
    ) -> Result<Option<u128>, Error> {
        let (Some(left), Some(right)) = (left, right) else { return Ok(None) };
        self.residual = existing.min(residual_energy(left, right)?);
        Ok(Some(self.residual))
    }
}

/// Explicit arithmetic census of the full successful path, not CPU fuel. Each
/// checked operation's refusal branch must be priced separately by the caller.
/// Roots use the existing fixed-width ceil_sqrt/isqrt logical primitive policy.
/// ceil_q adds one checked add, /, %, comparison and bool conversion.
pub(super) struct Arithmetic {
    pub mul: usize, pub add: usize, pub min: usize, pub roots: usize,
    pub div: usize, pub rem: usize, pub comparisons: usize, pub conversions: usize,
    pub checked_refusals: usize, pub proof_options: usize,
}
pub(super) const PROJECTION: Arithmetic = Arithmetic {
    mul: 9, add: 3, min: 3, roots: 3, div: 1, rem: 1,
    comparisons: 2, conversions: 1, checked_refusals: 12, proof_options: 3,
};
pub(super) const ACTIVATION: Arithmetic = Arithmetic {
    mul: 4, add: 1, min: 1, roots: 0, div: 0, rem: 0,
    comparisons: 1, conversions: 0, checked_refusals: 5, proof_options: 1,
};
pub(super) const PRODUCT: Arithmetic = Arithmetic {
    mul: 8, add: 2, min: 3, roots: 2, div: 1, rem: 1,
    comparisons: 2, conversions: 1, checked_refusals: 10, proof_options: 2,
};
pub(super) const RESIDUAL: Arithmetic = Arithmetic {
    mul: 1, add: 1, min: 1, roots: 2, div: 0, rem: 0,
    comparisons: 0, conversions: 0, checked_refusals: 2, proof_options: 2,
};

#[cfg(test)]
#[path = "row_energy_tests.rs"]
mod tests;
