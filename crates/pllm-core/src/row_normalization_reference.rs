//! Ordered binary64, no-epsilon centered-energy normalization reference.
//! Ordinary clear square roots/division only: no inverse polynomial, HE profile,
//! decoder-f32 equivalence, parameter authority, or source authentication.
//! One common s per row is its row-head ascending Fold(A) result; v is the
//! row-head Fold(z*z) result. Their binary64 tree order is part of this clear
//! policy, not a claim that cyclic floating reductions agree at every lane.
//! This is a separate scalar-row reference, NOT bitwise continuation of the
//! legacy per-lane cyclic prefix `Invocation::execute`.
use crate::row_fold::{
    CenteredEnergy, Error as FoldError, Geometry, Invocation, State, WorkLedger,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Fold(FoldError),
    NonPositiveEnergy,
    NonFinite,
    AffineShape,
}
impl From<FoldError> for Error {
    fn from(e: FoldError) -> Self {
        Self::Fold(e)
    }
}

/// Conservative deterministic charge includes affine validation, all active
/// lane reads/writes/checks, padding writes, row sqrt/division and transitions.
/// Prefix charge already includes cleanup of the ENTIRE six-buffer pool/public
/// storage on success, failure and cancellation. No extra vector allocation.
pub fn continuation_charge(g: Geometry) -> Result<(usize, usize), Error> {
    Ok((
        16 * g.quote()?.slots + 4 * g.width() + 32 * g.rows() + 1024,
        4096,
    ))
}
pub fn total_charge(g: Geometry) -> Result<(usize, usize), Error> {
    let q = g.quote()?;
    let (work, control) = continuation_charge(g)?;
    Ok((
        q.array_work
            .checked_add(work)
            .ok_or(FoldError::ArithmeticOverflow)?,
        q.control_work
            .checked_add(control)
            .ok_or(FoldError::ArithmeticOverflow)?,
    ))
}
pub struct Reference<'a> {
    prefix: Invocation<'a>,
    gamma: &'a [f64],
    beta: &'a [f64],
}
/// Opaque single-use COMPLETE-work prepayment. Compiler policy/control work can
/// be included before any policy traversal; no admission field is caller mutable.
pub struct Prepayment {
    prefix: crate::row_fold::PrefixPrepayment,
    geometry: Geometry,
}
pub fn prepay(
    g: Geometry,
    work: &mut WorkLedger,
    additional_control: usize,
) -> Result<Prepayment, Error> {
    let (extra, control) = continuation_charge(g)?;
    let control = control
        .checked_add(additional_control)
        .ok_or(FoldError::ArithmeticOverflow)?;
    Ok(Prepayment {
        prefix: Invocation::prepay_continuation(g, work, extra, control)?,
        geometry: g,
    })
}
pub fn control_bytes(g: Geometry) -> Result<usize, Error> {
    let bytes = g.quote()?.control_bytes
        + std::mem::size_of::<Reference<'static>>()
        + std::mem::size_of::<Output<'static>>()
        + std::mem::size_of::<Prepayment>();
    if bytes > crate::row_fold::CONTROL_BYTES {
        return Err(FoldError::ShortStorage.into());
    }
    Ok(bytes)
}
/// Final output owns the continuation's original invocation borrows. Dropping it
/// clears the whole pool/public region, without releasing caller-owned charges.
///
/// ```compile_fail
/// use pllm_core::{row_fold::*, row_normalization_reference::Reference};
/// let g=Geometry::new(1,2,2).unwrap();
/// let input=[1.0,2.0]; let mut masks=[0.0;8]; let mut pool=[0.0;12];
/// let mut gamma=[1.0;2]; let beta=[0.0;2]; let mut state=State::Fresh;
/// let mut work=WorkLedger::new(1000000,1000000);
/// let output=Reference::new(g,&input,&gamma,&beta,&mut masks,&mut pool,&mut state,&mut work).unwrap().execute().unwrap();
/// gamma[0]=2.0;
/// assert!(output.values()[0].is_finite());
/// ```
///
/// ```compile_fail
/// use pllm_core::{row_fold::*, row_normalization_reference::Reference};
/// let output;
/// {
///     let input=[1.0,2.0]; let gamma=[1.0;2]; let beta=[0.0;2];
///     let mut masks=[0.0;8]; let mut pool=[0.0;12]; let mut state=State::Fresh;
///     let mut work=WorkLedger::new(1000000,1000000);
///     output=Reference::new(Geometry::new(1,2,2).unwrap(),&input,&gamma,&beta,
///         &mut masks,&mut pool,&mut state,&mut work).unwrap().execute().unwrap();
/// }
/// assert!(output.values()[0].is_finite()); // Cannot escape original pool/owners.
/// ```
pub struct Output<'a> {
    energy: CenteredEnergy<'a>,
    _gamma: &'a [f64],
    _beta: &'a [f64],
}
impl Output<'_> {
    pub fn values(&self) -> &[f64] {
        self.energy.normalized()
    }
}
impl<'a> Reference<'a> {
    #[allow(clippy::too_many_arguments)] // Explicit bounded borrowed regions, no hidden owner.
    pub fn new(
        g: Geometry,
        input: &'a [f64],
        gamma: &'a [f64],
        beta: &'a [f64],
        public: &'a mut [f64],
        pool: &'a mut [f64],
        state: &'a mut State,
        work: &mut WorkLedger,
    ) -> Result<Self, Error> {
        let q = g.quote()?;
        if *state != State::Fresh {
            return Err(FoldError::Poisoned.into());
        }
        if gamma.len() != g.width() || beta.len() != g.width() {
            return Err(Error::AffineShape);
        }
        if input.len() != q.slots
            || public.len() != q.public_cells
            || pool.len() != q.workspace_cells
        {
            return Err(FoldError::ShortStorage.into());
        }
        control_bytes(g)?;
        // Debit COMPLETE work before input/mask/affine loops or initialization.
        let payment = prepay(g, work, 0)?;
        Self::new_prepaid(payment, input, gamma, beta, public, pool, state)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new_prepaid(
        payment: Prepayment,
        input: &'a [f64],
        gamma: &'a [f64],
        beta: &'a [f64],
        public: &'a mut [f64],
        pool: &'a mut [f64],
        state: &'a mut State,
    ) -> Result<Self, Error> {
        control_bytes(payment.geometry)?;
        if gamma.len() != payment.geometry.width() || beta.len() != payment.geometry.width() {
            return Err(Error::AffineShape);
        }
        let prefix = Invocation::new_prepaid(payment.prefix, input, public, pool, state)?;
        // A failure here drops the prefix, retaining poison and burning prepayment.
        if gamma.iter().chain(beta).any(|x| !x.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(Self {
            prefix,
            gamma,
            beta,
        })
    }
    pub fn execute(self) -> Result<Output<'a>, Error> {
        let mut energy = self.prefix.into_centered_energy()?;
        let g = energy.geometry();
        {
            let (z, v, y) = energy.continuation_parts()?;
            let numerator = (g.width() as f64).sqrt();
            if !numerator.is_finite() {
                return Err(Error::NonFinite);
            }
            for row in 0..g.rows() {
                let base = row * g.block();
                let variance = v[base];
                if !variance.is_finite() {
                    return Err(Error::NonFinite);
                }
                if variance <= 0.0 {
                    return Err(Error::NonPositiveEnergy);
                }
                let root = variance.sqrt();
                let r = numerator / root;
                if !root.is_finite() || root <= 0.0 || !r.is_finite() {
                    return Err(Error::NonFinite);
                }
                for c in 0..g.width() {
                    // Separate binary64 operations, deliberately no mul_add.
                    let normalized = z[base + c] * r;
                    let weighted = normalized * self.gamma[c];
                    let value = weighted + self.beta[c];
                    if !normalized.is_finite() || !weighted.is_finite() || !value.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    y[base + c] = value;
                }
                y[base + g.width()..base + g.block()].fill(0.0);
            }
        }
        energy.finish_normalized();
        Ok(Output {
            energy,
            _gamma: self.gamma,
            _beta: self.beta,
        })
    }
}

#[cfg(test)]
#[path = "row_normalization_reference_tests.rs"]
mod tests;
