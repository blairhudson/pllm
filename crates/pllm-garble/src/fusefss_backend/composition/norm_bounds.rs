//! Conditional bounds for the closed D13 source normalization formula.
//!
//! SourceBounds do NOT certify an input row, its mean/centering, or its sum of
//! squares. The caller must first prove that all those operations, including
//! `S = 1 + sum(z_i^2)`, satisfy the normalization kernel's no-wrap obligations.
//! In particular, output energy cannot be used to justify that same input S.
//!
//! The only table covered here is `source_tables::rsqrt(k)`, also owned by the
//! closed `cache::Kind::Rsqrt { k }` factory. Its existing source-lock revision,
//! table digest and norm contract remain binding; an arbitrary table with the
//! same maximum is not covered. No caller-supplied table or certificate enters
//! this API. The bounds themselves are derived public constants, not private
//! membership proofs or permission to change either signed ARS12 floor.
//! The separate RMS-only input-energy intersection consumes an independently
//! established complete-row premise; it never derives that premise from this
//! same row's conditional output bound.
use super::super::super::Error;
use super::super::cache;

const Q: u128 = 4096;

/// Width-specific consequences of the frozen source formula, conditional on an
/// independently admitted, unwrapped row. Fields cannot be caller-authored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceBounds {
    width: usize,
    raw_product_abs_max: i64,
    normalized_min: i64,
    normalized_max: i64,
    raw_row_energy: u128,
    normalized_row_energy: u128,
}

impl SourceBounds {
    pub fn width(self) -> usize {
        self.width
    }

    /// Closed native table formula to which this conditional proof applies.
    pub fn table_kind(self) -> cache::Kind {
        cache::Kind::Rsqrt { k: self.width }
    }

    /// Inclusive absolute bound on each Q24 product z_i * R, BEFORE ARS12.
    pub fn raw_product_abs_max(self) -> i64 {
        self.raw_product_abs_max
    }

    /// Inclusive Q12 interval AFTER the first floor and BEFORE gamma/beta.
    /// Negative floors require an asymmetric interval.
    pub fn normalized_interval(self) -> (i64, i64) {
        (self.normalized_min, self.normalized_max)
    }

    /// Complete-row sum of squared Q24 products, not a per-coordinate bound.
    pub fn raw_row_energy(self) -> u128 {
        self.raw_row_energy
    }

    /// Complete-row sum of squared Q12 codes after the first floor only.
    /// Gamma, its separate floor, and beta still need their own proofs.
    pub fn normalized_row_energy(self) -> u128 {
        self.normalized_row_energy
    }
}

fn ceil_root(n: u128) -> Result<u128, Error> {
    let root = n.isqrt();
    let square = root.checked_mul(root).ok_or(Error::Domain)?;
    root.checked_add(u128::from(square != n))
        .ok_or(Error::Domain)
}

fn floor_energy(raw: u128, width: usize) -> Result<u128, Error> {
    // Q*q_i = p_i-r_i, with integer 0<=r_i<=Q-1 for BOTH signs of p_i.
    // Cauchy--Schwarz bounds the cross term; ceil_root keeps it integral.
    let rounding = (width as u128)
        .checked_mul((Q - 1) * (Q - 1))
        .ok_or(Error::Domain)?;
    let cross = ceil_root(raw.checked_mul(rounding).ok_or(Error::Domain)?)?;
    raw.checked_add(rounding)
        .and_then(|n| n.checked_add(cross.checked_mul(2)?))
        .map(|n| n / (Q * Q))
        .ok_or(Error::Domain)
}

/// Derive bounds without allocating a table or accepting an input certificate.
///
/// For an independently proved 1<=S<2^50, let e=floor(log2(S)),
/// j=floor(128*S/2^e), and R=isqrt(floor((2^55*k)/(j*2^e))). Then
/// 128<=j<=255 and j*2^e > (128^2/129)*S. Consequently
///
/// sum((z_i*R)^2) = (S-1)*R^2 < 129*k*2^41.
///
/// The strict inequality gives the integer energy N=129*k*2^41-1. Its floor
/// square root bounds each raw product. For q_i=floor(z_i*R/4096), writing
/// 4096*q_i=z_i*R-r_i with 0<=r_i<=4095 proves the returned row energy:
/// floor((N + k*4095^2 + 2*ceil_sqrt(N*k*4095^2))/4096^2).
///
/// This implication covers the full unsigned index domain, but does not relax
/// the current kernel's stricter signed50 admission of squares and S.
pub fn source_bounds(width: usize) -> Result<SourceBounds, Error> {
    if !(1..=4096).contains(&width) {
        return Err(Error::Domain);
    }
    let raw_row_energy = (1u128 << 41)
        .checked_mul(129)
        .and_then(|n| n.checked_mul(width as u128))
        .and_then(|n| n.checked_sub(1))
        .ok_or(Error::Domain)?;
    let raw_product_abs_max = i64::try_from(raw_row_energy.isqrt()).map_err(|_| Error::Domain)?;
    let normalized_min = (-raw_product_abs_max).div_euclid(Q as i64);
    let normalized_max = raw_product_abs_max.div_euclid(Q as i64);
    Ok(SourceBounds {
        width,
        raw_product_abs_max,
        normalized_min,
        normalized_max,
        raw_row_energy,
        normalized_row_energy: floor_energy(raw_row_energy, width)?,
    })
}

/// Conditional input proof for RMS only. `square_min/max` are the original
/// scalar interval's exact squared extrema, before signed50 admission. The
/// caller has already proved its input/mean/centering interval obligations.
///
/// If E bounds the ENTIRE row, every square and every partial sum of these
/// nonnegative terms is <=E. Intersect each with its independent scalar bound;
/// never multiply E by k, drop the interval lower bound, or widen an interval.
/// The caller must still prove +1 and the separate product/affine operations.
pub(super) fn rms_input_energy_bounds(
    table: cache::Kind,
    square_min: i128,
    square_max: i128,
    input_energy: u128,
) -> Result<(super::Interval, super::Interval), Error> {
    let k = match table {
        cache::Kind::Rsqrt { k } if (1..=4096).contains(&k) => k,
        _ => return Err(Error::Domain),
    };
    if square_min < 0 || square_min > square_max {
        return Err(Error::Domain);
    }
    let energy = i128::try_from(input_energy).map_err(|_| Error::Domain)?;
    let sum_min = square_min.checked_mul(k as i128).ok_or(Error::Domain)?;
    let sum_max = square_max.checked_mul(k as i128).ok_or(Error::Domain)?;
    if energy < sum_min {
        return Err(Error::Domain);
    }
    Ok((
        super::Interval::checked(square_min, square_max.min(energy))?,
        super::Interval::checked(sum_min, sum_max.min(energy))?,
    ))
}

#[cfg(test)]
#[path = "norm_bounds_tests.rs"]
mod tests;
