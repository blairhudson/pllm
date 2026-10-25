//! Original-counter ownership adapter for the clear coefficient phase kernel.
//! No native/private Share capability, encryption, source adoption or transport.
use crate::shared_allocation::{self, Budget, Error, OwnedBudget, OwnedCharged};
use phase::{Admitted, Buffers, Inputs, Quote, Work};
use pllm_core::coefficient_phase_reference as phase;
use std::mem::size_of;

/// Original clear vector owners, not arbitrary views into unknown larger backing.
/// Capacity must equal the admitted length; retained borrows prevent replacement
/// or growth. Construction of these supplied records is the caller's own phase.
pub struct Supplied<'a> {
    pub lhs: &'a [Vec<u64>; 2],
    pub rhs: &'a [Vec<u64>; 2],
    pub masks: &'a [Vec<phase::Wide>; 2],
    pub crypto_error: phase::CryptoError,
}
impl<'a> Supplied<'a> {
    fn input(&self) -> Inputs<'a> {
        Inputs {
            lhs: [&self.lhs[0], &self.lhs[1]],
            rhs: [&self.rhs[0], &self.rhs[1]],
            masks: [&self.masks[0], &self.masks[1]],
            crypto_error: self.crypto_error,
        }
    }
    fn validate(&self) -> Result<(), Error> {
        self.input().validate_lengths().map_err(map)?;
        if self
            .lhs
            .iter()
            .chain(self.rhs.iter())
            .any(|v| v.capacity() != v.len())
            || self.masks.iter().any(|v| v.capacity() != v.len())
        {
            return Err(Error::Resource);
        }
        Ok(())
    }
}

/// Payload-first drop is enforced by existing OwnedCharged<T>. Original source
/// borrows and every scratch/output backing remain charged until this owner drops.
/// Keeping scratch is conservative: no refund/reacquire at output transition.
pub struct ClearRegion<'a> {
    input: Inputs<'a>,
    _original: Supplied<'a>,
    scratch: Vec<u64>,
    phases: Vec<u64>,
    converted: Vec<u64>,
    matrices: Vec<u64>,
}
impl ClearRegion<'_> {
    pub fn phases(&self) -> &[u64] {
        &self.phases
    }
    pub fn converted(&self) -> &[u64] {
        &self.converted
    }
    pub fn matrices(&self) -> &[u64] {
        &self.matrices
    }
    pub fn inputs(&self) -> &Inputs<'_> {
        &self.input
    }
    pub fn scratch_words(&self) -> usize {
        self.scratch.len()
    }
}
fn map(_: phase::Error) -> Error {
    Error::Resource
}

/// Exact logical reservation before all native payload factories. Inline
/// OwnedCharged<T> storage is accounted by its existing try_build transition.
pub fn reservation_bytes() -> Result<usize, Error> {
    let q = Quote::qualification().map_err(map)?;
    let extra = size_of::<OwnedCharged<ClearRegion<'_>>>()
        .checked_sub(size_of::<shared_allocation::OwnedExternalCharge>())
        .ok_or(Error::Resource)?;
    q.total_bytes()
        .checked_add(extra)
        .and_then(|n| n.checked_add(size_of::<shared_allocation::OwnedExternalCharge>()))
        .filter(|&n| n <= phase::MAX_BYTES)
        .ok_or(Error::Resource)
}

fn zeroed(words: usize, budget: &Budget) -> Result<Vec<u64>, Error> {
    let mut v = shared_allocation::allocate(words, budget)?;
    v.resize(words, 0);
    Ok(v)
}

/// Both directions execute under ONE consumed work quote and ONE original
/// reservation. No callback can obtain output before the complete region returns.
pub fn evaluate<'a>(
    budget: &OwnedBudget,
    work: &mut Work,
    supplied: Supplied<'a>,
) -> Result<OwnedCharged<ClearRegion<'a>>, Error> {
    supplied.validate()?;
    let input = supplied.input();
    let q = Quote::qualification().map_err(map)?;
    let _ = reservation_bytes()?;
    // Concrete named constructor/control inventory, inside the kernel envelope.
    let inline = size_of::<ClearRegion<'_>>()
        + size_of::<Inputs<'_>>()
        + size_of::<Quote>()
        + size_of::<Admitted>()
        + size_of::<Buffers<'_>>()
        + size_of::<Result<OwnedCharged<ClearRegion<'_>>, Error>>()
        + size_of::<Result<Admitted, phase::Error>>()
        + size_of::<Result<Vec<u64>, Error>>()
        + size_of::<Work>()
        + 2 * size_of::<[Vec<u64>; 2]>()
        + size_of::<[Vec<phase::Wide>; 2]>()
        + 32 * size_of::<phase::Wide>();
    if inline > q.control_bytes() {
        return Err(Error::Resource);
    }
    let charge = budget.charge_external(q.total_bytes())?;
    charge.try_build(move |original| {
        let admitted =
            Admitted::qualification(q.total_bytes(), work, input.crypto_error).map_err(map)?;
        admitted.validate_inputs(&input).map_err(map)?;
        // All construction, failure and unwind paths retain the original guard.
        let mut result = ClearRegion {
            input,
            _original: supplied,
            scratch: zeroed(phase::SCRATCH_WORDS, original)?,
            phases: zeroed(phase::PHASE_WORDS, original)?,
            converted: zeroed(phase::CONVERTED_WORDS, original)?,
            matrices: zeroed(phase::MATRIX_WORDS, original)?,
        };
        admitted
            .evaluate(
                &result.input,
                Buffers {
                    scratch: &mut result.scratch,
                    phases: &mut result.phases,
                    converted: &mut result.converted,
                    matrices: &mut result.matrices,
                },
            )
            .map_err(map)?;
        Ok(result)
    })
}

#[cfg(test)]
#[path = "coefficient_phase_reference/tests.rs"]
mod tests;
