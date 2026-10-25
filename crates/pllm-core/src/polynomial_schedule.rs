//! Bounded clear binary64 Stockmeyer schedule. Not an HE/precision certificate.
//! Operations are separate multiply/add instructions (no mul_add/fast math).
use std::mem::size_of;
use zeroize::{Zeroize, Zeroizing};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Shape,
    NonFinite,
}

pub struct Workspace {
    babies: [f64; 8],
    first: [f64; 4],
    second: [f64; 2],
    powers: [f64; 3],
}
impl Workspace {
    pub const fn empty() -> Self {
        Self {
            babies: [0.; 8],
            first: [0.; 4],
            second: [0.; 2],
            powers: [0.; 3],
        }
    }
}
impl Zeroize for Workspace {
    fn zeroize(&mut self) {
        self.babies.zeroize();
        self.first.zeroize();
        self.second.zeroize();
        self.powers.zeroize();
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Resources {
    pub workspace_bytes: usize,
    pub control_bytes: usize,
    pub logical_peak_bytes: usize,
}
pub fn resources() -> Resources {
    let workspace_bytes = size_of::<Zeroizing<Workspace>>();
    // Two slice/reference arguments, input, three loop indices and their
    // iterator/Option positions; coefficient group + arithmetic/Result copies.
    let control_bytes = size_of::<(&[f64], f64, &mut Workspace)>()
        + 3 * size_of::<(usize, std::ops::Range<usize>, Option<usize>)>()
        + size_of::<[f64; 4]>()
        + 8 * size_of::<f64>()
        + 4 * size_of::<Result<f64, Error>>()
        + size_of::<Result<usize, Error>>()
        + size_of::<std::slice::Iter<'_, f64>>()
        + size_of::<Option<&f64>>();
    Resources {
        workspace_bytes,
        control_bytes,
        logical_peak_bytes: workspace_bytes + control_bytes,
    }
}
pub fn arithmetic_work(coefficients: usize) -> Result<usize, Error> {
    Ok(match coefficients {
        16 => 37,
        28 => 65,
        32 => 74,
        _ => return Err(Error::Shape),
    })
}
// Keep finite checks at every original arithmetic boundary, never clip.
fn finite(x: f64) -> Result<f64, Error> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(Error::NonFinite)
    }
}
fn mul(a: f64, b: f64) -> Result<f64, Error> {
    finite(a * b)
}
fn add(a: f64, b: f64) -> Result<f64, Error> {
    finite(a + b)
}

/// Increasing-order complete tuples ONLY. Caller controls coefficient reversal,
/// input scaling and outer transforms; none are silently normalized here.
pub fn stockmeyer(coefficients: &[f64], x: f64, w: &mut Workspace) -> Result<f64, Error> {
    arithmetic_work(coefficients.len())?;
    finite(x)?;
    for &c in coefficients {
        finite(c)?;
    }
    w.powers[0] = mul(x, x)?;
    w.powers[1] = mul(w.powers[0], w.powers[0])?;
    w.powers[2] = mul(w.powers[1], w.powers[1])?;
    let count = coefficients.len() / 4;
    for i in 0..count {
        let [c0, c1, c2, c3] = [
            coefficients[4 * i],
            coefficients[4 * i + 1],
            coefficients[4 * i + 2],
            coefficients[4 * i + 3],
        ];
        let value = mul(c1, x)?;
        let value = add(value, mul(c2, w.powers[0])?)?;
        // Deliberately x2*(c3*x), NOT c3*(x*x2).
        let value = add(value, mul(w.powers[0], mul(c3, x)?)?)?;
        w.babies[i] = add(value, c0)?;
    }
    for i in 0..count.div_ceil(2) {
        w.first[i] = if 2 * i + 1 < count {
            add(w.babies[2 * i], mul(w.babies[2 * i + 1], w.powers[1])?)?
        } else {
            w.babies[2 * i]
        };
    }
    if coefficients.len() == 16 {
        return add(w.first[0], mul(w.first[1], w.powers[2])?);
    }
    for i in 0..2 {
        w.second[i] = add(w.first[2 * i], mul(w.first[2 * i + 1], w.powers[2])?)?;
    }
    add(
        w.second[0],
        mul(w.second[1], mul(w.powers[2], w.powers[2])?)?,
    )
}

#[cfg(test)]
#[path = "polynomial_schedule_tests.rs"]
mod tests;
