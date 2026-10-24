//! Private D43 certified inverse subrecipe and exact owning-source loans.
//! Whole-parent issuance/activation remains absent. Unfinished parent stages
//! are preserved in target/d43-pre-narrowing-parent-work-hold-88679dd8.patch.
use super::*;
use pllm_core::edge_contract::ConsumerUse;
use std::sync::Arc;

#[cfg(test)]
#[path = "layernorm4/norm13_tests.rs"]
mod norm13_tests;
pub(in crate::masked_wire_reference) mod parent;
pub(in crate::masked_wire_reference) mod table;

const SLOTS: usize = 18;
const USES: usize = 24;
const INVERSE_USE: u8 = 16;
const INDEX_SLOT: usize = 13;
const OUTPUT_SLOT: usize = 14;

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Private parent kinds; no public profile or caller kind adoption.
enum ValueKind {
    SignedFixed { bits: u8, fraction: u8 },
    UnsignedResidue { bits: u8 },
}

/// Immutable original source scope; no masks, clear inputs or cancellation
/// backlink. No public constructor, digest-to-source upgrade or parent success.
#[allow(dead_code)] // Real closed native scope, not activated by a production parent yet.
pub(in crate::masked_wire_reference) struct SourceFor<C: AllocationContext> {
    profile: [u8; 32],
    session: [u8; 32],
    operation: [u8; 32],
    descriptors: [EdgeDescriptor; SLOTS],
    kinds: [ValueKind; SLOTS],
    uses: [ConsumerUse; USES],
    table: Arc<table::CertifiedTableFor<C>>,
    lease: LeaseFor<C>,
}

// Only original encoded owners or the parent's private homomorphic local map.
// No external constructor adopts caller vectors or bindings.
#[allow(dead_code)] // Remaining original-owner variant awaits the complete parent.
#[allow(clippy::large_enum_variant)] // Actual inline native owner layout is priced, not boxed.
enum MaskBacking<C: AllocationContext> {
    Original(DealerMaskFor<C, Ring>),
    Local(Zeroizing<Vec<u64>>),
}
impl<C: AllocationContext> MaskBacking<C> {
    fn values(&self) -> &[u64] {
        match self {
            Self::Original(v) => v.backing.values(),
            Self::Local(v) => v,
        }
    }
}
#[allow(dead_code)] // Remaining original-owner variant awaits the complete parent.
#[allow(clippy::large_enum_variant)] // Actual inline native owner layout is priced, not boxed.
enum ValueBacking<C: AllocationContext> {
    Original(CommonMaskedFor<C, Ring>),
    Local(Zeroizing<Vec<u64>>),
}
impl<C: AllocationContext> ValueBacking<C> {
    fn values(&self) -> &[u64] {
        match self {
            Self::Original(v) => v.backing.values(),
            Self::Local(v) => v,
        }
    }
}
struct ProducedMaskFor<C: AllocationContext> {
    backing: MaskBacking<C>,
    binding: Binding,
    source: Arc<SourceFor<C>>,
    lease: LeaseFor<C>,
}
struct ProducedValueFor<C: AllocationContext> {
    backing: ValueBacking<C>,
    binding: Binding,
    role: u8,
    source: Arc<SourceFor<C>>,
    lease: LeaseFor<C>,
}

#[derive(Clone, Copy)]
pub(in crate::masked_wire_reference) struct Norm13Binding(Binding);

/// Consumes exactly inverse use16 against the actual slot, before validation.
#[allow(dead_code)] // Real sealed loan, only native qualification until parent issuance.
pub(in crate::masked_wire_reference) struct Norm13MaskLoan<'a, C: AllocationContext> {
    source: Arc<SourceFor<C>>,
    original: &'a ProducedMaskFor<C>,
}
#[allow(dead_code)] // No caller constructor or production activation.
impl<'a, C: AllocationContext> Norm13MaskLoan<'a, C> {
    fn take(
        source: &Arc<SourceFor<C>>,
        ledger: &mut internal_use::Ledger<USES>,
        masks: &'a [Option<ProducedMaskFor<C>>; SLOTS],
    ) -> Result<Self, Error> {
        let Some(original) = masks[INDEX_SLOT].as_ref() else {
            ledger.next = ledger.next.saturating_add(1);
            return Err(Error::Consumed);
        };
        ledger.burn(INVERSE_USE, original.binding, ledger.role)?;
        if !Arc::ptr_eq(source, &original.source)
            || source.descriptors[INDEX_SLOT] != original.binding.descriptor
            || source.uses[usize::from(INVERSE_USE)] != ledger.uses[usize::from(INVERSE_USE)]
            || source.kinds[INDEX_SLOT] != (ValueKind::UnsignedResidue { bits: 13 })
            || !std::ptr::eq(source.lease.budget.budget(), original.lease.budget.budget())
            || !std::ptr::eq(source.lease.budget.budget(), source.table.budget())
            || original.backing.values().len() != 1
            || original.backing.values()[0] >= 8192
        {
            return Err(Error::Context);
        }
        Ok(Self {
            source: Arc::clone(source),
            original,
        })
    }
    pub(in crate::masked_wire_reference) fn values(&self) -> &[u64] {
        self.original.backing.values()
    }
    pub(in crate::masked_wire_reference) fn input_binding(&self) -> Norm13Binding {
        Norm13Binding(self.original.binding)
    }
    pub(in crate::masked_wire_reference) fn table(&self) -> &table::CertifiedTableFor<C> {
        &self.source.table
    }
    pub(in crate::masked_wire_reference) fn source(&self) -> &Arc<SourceFor<C>> {
        &self.source
    }
    pub(in crate::masked_wire_reference) fn context(&self) -> super::super::source_lookup::Context {
        let s = &self.source;
        let wire = |domain: &[u8]| {
            digest(&[
                b"pllm.sigma.layernorm4.inverse13.v1",
                domain,
                &s.profile,
                &s.operation,
            ])
        };
        super::super::source_lookup::Context {
            plan: s.descriptors[INDEX_SLOT].identity().plan(),
            session: s.session,
            operation: s.operation,
            input_wire: s.descriptors[INDEX_SLOT].digest(),
            output_wire: s.descriptors[OUTPUT_SLOT].digest(),
            arithmetic_wire: wire(b"arithmetic"),
            correction_wire: wire(b"correction"),
            select_wire: wire(b"select"),
            numeric_contract: s.profile,
            first_tensor_index: s.descriptors[INDEX_SLOT].identity().first_tensor_index(),
        }
    }
}

#[allow(dead_code)] // Actual sealed role loan; complete owning parent still required.
pub(in crate::masked_wire_reference) struct Norm13ValueLoan<'a, C: AllocationContext> {
    source: Arc<SourceFor<C>>,
    original: &'a ProducedValueFor<C>,
}
#[allow(dead_code)] // No caller binding/interval adoption or public production ingress.
impl<'a, C: AllocationContext> Norm13ValueLoan<'a, C> {
    fn take(
        source: &Arc<SourceFor<C>>,
        ledger: &mut internal_use::Ledger<USES>,
        values: &'a [Option<ProducedValueFor<C>>; SLOTS],
        role: u8,
    ) -> Result<Self, Error> {
        let Some(original) = values[INDEX_SLOT].as_ref() else {
            ledger.next = ledger.next.saturating_add(1);
            return Err(Error::Consumed);
        };
        ledger.burn(INVERSE_USE, original.binding, role)?;
        if !Arc::ptr_eq(source, &original.source)
            || original.role != role
            || source.descriptors[INDEX_SLOT] != original.binding.descriptor
            || source.uses[usize::from(INVERSE_USE)] != ledger.uses[usize::from(INVERSE_USE)]
            || source.kinds[INDEX_SLOT] != (ValueKind::UnsignedResidue { bits: 13 })
            || !std::ptr::eq(source.lease.budget.budget(), original.lease.budget.budget())
            || original.backing.values().len() != 1
            || original.backing.values()[0] >= 8192
        {
            return Err(Error::Context);
        }
        Ok(Self {
            source: Arc::clone(source),
            original,
        })
    }
    pub(in crate::masked_wire_reference) fn values(&self) -> &[u64] {
        self.original.backing.values()
    }
    pub(in crate::masked_wire_reference) fn is_for(&self, input: Norm13Binding) -> bool {
        self.original.binding == input.0
    }
    pub(in crate::masked_wire_reference) fn role(&self) -> u8 {
        self.original.role
    }
    pub(in crate::masked_wire_reference) fn source(&self) -> &Arc<SourceFor<C>> {
        &self.source
    }
}
