//! Owning D43 parent implementation. Native interval ingress is kept closed
//! here: neither a raw scan nor a caller Binding/interval constructs a loan.
use super::*;

#[cfg(test)]
#[path = "parent/interval_tests.rs"]
mod interval_tests;

const ENERGY_SLOT: usize = 7;
const ENERGY_USE: u8 = 7;

// Only local linear paper maps. Nonlinear stages NEVER enter this function:
// square/products, signed floors, unsigned TR, interval and inverse remain
// actual native children. Destination storage is preissued by the parent.
#[allow(dead_code)] // Fixed linear parent stages staged until actual nonlinear producer chain is installed.
enum LocalMap {
    ScaledMean,
    Center,
    Energy,
    Mantissa,
    Index,
}
#[allow(dead_code)] // No native activation or whole-row substitute; only preissued-buffer local maps.
impl LocalMap {
    fn apply(
        &self,
        left: &[u64],
        right: &[u64],
        out: &mut [u64],
        dealer_mask: bool,
    ) -> Result<(), Error> {
        let canonical = |v: &[u64], lanes: usize, bits: u8| -> Result<(), Error> {
            if v.len() != lanes || v.iter().any(|x| x & !super::super::super::mask(bits) != 0) {
                return Err(Error::Domain);
            }
            Ok(())
        };
        let ring = super::super::super::mask(50);
        match self {
            Self::ScaledMean => {
                canonical(left, 4, 50)?;
                canonical(right, 0, 50)?;
                if out.len() != 1 {
                    return Err(Error::Domain);
                }
                out[0] = left
                    .iter()
                    .fold(0u64, |a, b| a.wrapping_add(*b))
                    .wrapping_mul(1 << 10)
                    & ring;
            }
            Self::Center => {
                canonical(left, 4, 50)?;
                canonical(right, 1, 50)?;
                if out.len() != 4 {
                    return Err(Error::Domain);
                }
                for (out, x) in out.iter_mut().zip(left) {
                    *out = x.wrapping_sub(right[0]) & ring;
                }
            }
            Self::Energy => {
                canonical(left, 4, 50)?;
                canonical(right, 0, 50)?;
                if out.len() != 1 {
                    return Err(Error::Domain);
                }
                // +1 in the Q24 CODE energy, not a public Q24 unit. Dealer
                // mask remains the linear sum; no secret value is opened here.
                out[0] = left
                    .iter()
                    .fold(u64::from(!dealer_mask), |a, b| a.wrapping_add(*b))
                    & ring;
            }
            Self::Mantissa => {
                canonical(left, 1, 8)?;
                canonical(right, 0, 8)?;
                if out.len() != 1 {
                    return Err(Error::Domain);
                }
                out[0] = left[0] & 127;
            }
            Self::Index => {
                canonical(left, 1, 7)?;
                canonical(right, 1, 13)?;
                if out.len() != 1 {
                    return Err(Error::Domain);
                }
                out[0] = (left[0].wrapping_mul(64).wrapping_add(right[0])) & 8191;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::masked_wire_reference) struct EnergyBinding(Binding);

pub(in crate::masked_wire_reference) struct EnergyMaskLoan<'a, C: AllocationContext> {
    source: Arc<SourceFor<C>>,
    original: &'a ProducedMaskFor<C>,
}
#[cfg_attr(not(test), allow(dead_code))] // Native component ingress; production energy producer remains absent.
impl<'a, C: AllocationContext> EnergyMaskLoan<'a, C> {
    // Called only from the fixed parent ledger after its actual square->energy
    // producer is installed. No original-root/unsigned slot can substitute.
    fn take(
        source: &Arc<SourceFor<C>>,
        ledger: &mut internal_use::Ledger<USES>,
        masks: &'a [Option<ProducedMaskFor<C>>; SLOTS],
    ) -> Result<Self, Error> {
        let Some(original) = masks[ENERGY_SLOT].as_ref() else {
            ledger.next = ledger.next.saturating_add(1);
            return Err(Error::Consumed);
        };
        ledger.burn(ENERGY_USE, original.binding, ledger.role)?;
        // Reserve the two fixed scan->correction uses in this same consuming
        // attempt. Output-specific receipts are still minted ONLY after actual
        // key creation/scan; precommit cannot turn a raw result into authority.
        ledger.burn(ENERGY_USE + 1, original.binding, ledger.role)?;
        ledger.burn(ENERGY_USE + 2, original.binding, ledger.role)?;
        if !Arc::ptr_eq(source, &original.source)
            || source.descriptors[ENERGY_SLOT] != original.binding.descriptor
            || (usize::from(ENERGY_USE)..=usize::from(ENERGY_USE + 2))
                .any(|i| source.uses[i] != ledger.uses[i])
            || source.kinds[ENERGY_SLOT]
                != (ValueKind::SignedFixed {
                    bits: 50,
                    fraction: 24,
                })
            || !std::ptr::eq(source.lease.budget.budget(), original.lease.budget.budget())
            || original.backing.values().len() != 1
            || original.backing.values()[0] >= 1 << 50
        {
            return Err(Error::Context);
        }
        Ok(Self {
            source: Arc::clone(source),
            original,
        })
    }
    pub(in crate::masked_wire_reference) fn source(&self) -> &Arc<SourceFor<C>> {
        &self.source
    }
    pub(in crate::masked_wire_reference) fn binding(&self) -> EnergyBinding {
        EnergyBinding(self.original.binding)
    }
    pub(in crate::masked_wire_reference) fn mask(&self) -> u64 {
        self.original.backing.values()[0]
    }
    pub(in crate::masked_wire_reference) fn budget(&self) -> &Budget {
        self.original.lease.budget.budget()
    }
    pub(in crate::masked_wire_reference) fn interval_entries(&self) -> [[u64; 4]; 51] {
        // Same canonical checked fixed payload definition as the certified table
        // module; this is metadata, never a second membership evaluator.
        table::intervals().map(|entry| [entry.start, entry.end, entry.exponent, entry.multiplier])
    }
    pub(in crate::masked_wire_reference) fn context(
        &self,
        output: usize,
    ) -> super::super::super::source_lookup::Context {
        interval_context(&self.source, output)
    }
}

pub(in crate::masked_wire_reference) struct EnergyValueLoan<'a, C: AllocationContext> {
    source: Arc<SourceFor<C>>,
    original: &'a ProducedValueFor<C>,
}
#[cfg_attr(not(test), allow(dead_code))] // Ordered original-role loan; no caller scalar-to-energy authority.
impl<'a, C: AllocationContext> EnergyValueLoan<'a, C> {
    fn take(
        source: &Arc<SourceFor<C>>,
        ledger: &mut internal_use::Ledger<USES>,
        values: &'a [Option<ProducedValueFor<C>>; SLOTS],
        role: u8,
    ) -> Result<Self, Error> {
        let Some(original) = values[ENERGY_SLOT].as_ref() else {
            ledger.next = ledger.next.saturating_add(1);
            return Err(Error::Consumed);
        };
        ledger.burn(ENERGY_USE, original.binding, role)?;
        ledger.burn(ENERGY_USE + 1, original.binding, role)?;
        ledger.burn(ENERGY_USE + 2, original.binding, role)?;
        if !Arc::ptr_eq(source, &original.source)
            || original.role != role
            || source.descriptors[ENERGY_SLOT] != original.binding.descriptor
            || (usize::from(ENERGY_USE)..=usize::from(ENERGY_USE + 2))
                .any(|i| source.uses[i] != ledger.uses[i])
            || source.kinds[ENERGY_SLOT]
                != (ValueKind::SignedFixed {
                    bits: 50,
                    fraction: 24,
                })
            || !std::ptr::eq(source.lease.budget.budget(), original.lease.budget.budget())
            || original.backing.values().len() != 1
            || original.backing.values()[0] >= 1 << 50
        {
            return Err(Error::Context);
        }
        Ok(Self {
            source: Arc::clone(source),
            original,
        })
    }
    pub(in crate::masked_wire_reference) fn source(&self) -> &Arc<SourceFor<C>> {
        &self.source
    }
    pub(in crate::masked_wire_reference) fn binding(&self) -> EnergyBinding {
        EnergyBinding(self.original.binding)
    }
    pub(in crate::masked_wire_reference) fn role(&self) -> u8 {
        self.original.role
    }
    pub(in crate::masked_wire_reference) fn masked(&self) -> u64 {
        self.original.backing.values()[0]
    }
}

fn interval_context<C: AllocationContext>(
    source: &SourceFor<C>,
    output: usize,
) -> super::super::super::source_lookup::Context {
    // These are actual owning source descriptors. Hashes separate transcripts;
    // they do NOT mint source authority or assert S>=1 from a masked number.
    let operation = digest(&[
        b"pllm.sigma.interval51.gpu127.r50.e13-u50.v1",
        &source.profile,
        &source.descriptors[ENERGY_SLOT].digest(),
    ]);
    let wire = |domain: &[u8]| digest(&[&operation, domain, &source.descriptors[output].digest()]);
    super::super::super::source_lookup::Context {
        plan: source.descriptors[ENERGY_SLOT].identity().plan(),
        session: source.session,
        operation,
        input_wire: source.descriptors[ENERGY_SLOT].digest(),
        output_wire: source.descriptors[output].digest(),
        arithmetic_wire: wire(b"arithmetic"),
        correction_wire: wire(b"correction"),
        select_wire: wire(b"select"),
        numeric_contract: source.profile,
        first_tensor_index: source.descriptors[ENERGY_SLOT]
            .identity()
            .first_tensor_index(),
    }
}
