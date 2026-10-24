//! D36 one-row owner. Clear scores are original charged inputs, not inherited
//! attention certificates. Native child/coordinate gates remain private.
//! No transport, checkpoint authentication, SDK or joint-privacy qualification.
use super::*;
use crate::shared_allocation::{OwnedCharged, MAX_ALLOCATION_BYTES};
use std::sync::Arc;

#[path = "causal_softmax_clear.rs"]
mod clear;
#[path = "causal_softmax_ledger.rs"]
mod ledger;
#[path = "causal_softmax_max.rs"]
mod maximum;
#[path = "causal_softmax_quote.rs"]
mod quote;
#[path = "causal_softmax_records.rs"]
mod records;
#[path = "causal_softmax_region.rs"]
mod region;
#[path = "causal_softmax_spec.rs"]
mod spec;
pub use clear::{
    causal_softmax_bootstrap_bytes, causal_softmax_from_reserved, causal_softmax_resources,
    CausalSoftmaxCoordinatorFor, CausalSoftmaxCreatedFor, CausalSoftmaxError,
    CausalSoftmaxFrameFor, CausalSoftmaxMaskFor, CausalSoftmaxOutputFor, CausalSoftmaxRoleFor,
    CausalSoftmaxStepFor,
};
pub use quote::{
    CausalSoftmaxResources, CausalSoftmaxWork, PublicFloorWork as CausalSoftmaxPublicFloorWork,
};
pub use spec::{CausalSoftmaxSpec, CausalSoftmaxWorkLimit};

pub type CausalSoftmaxInput = OwnedCharged<Zeroizing<Vec<i64>>>;

/// Original clear source outlives every derived mask, party, frame and output.
/// Never points back to the coordinator: ownership graph stays acyclic.
struct SourceFor<C: AllocationContext> {
    input: CausalSoftmaxInput,
    declaration: CausalSoftmaxSpec,
    lease: LeaseFor<C>,
}

/// Private max-derived difference vector. Only this module can construct it;
/// values are ORIGINAL affine masks/common-masked words, never recovered scores.
pub(super) struct DifferenceMaskFor<C: AllocationContext> {
    binding: Binding,
    words: Zeroizing<Box<[u64]>>,
    source: Arc<SourceFor<C>>,
    lease: LeaseFor<C>,
}
pub(super) struct DifferenceValueFor<C: AllocationContext> {
    binding: Binding,
    role: u8,
    words: Zeroizing<Box<[u64]>>,
    source: Arc<SourceFor<C>>,
    lease: LeaseFor<C>,
}

/// Private sum of the genuine inherited E owner. The constructor establishes
/// the maximum relation (some E=4096), not a caller-supplied interval assertion.
pub(super) struct DenominatorMaskFor<C: AllocationContext> {
    binding: Binding,
    words: Zeroizing<Box<[u64]>>,
    #[allow(dead_code)] // Original source custody, never detached from native TR.
    source: Arc<SourceFor<C>>,
    lease: LeaseFor<C>,
}
pub(super) struct DenominatorValueFor<C: AllocationContext> {
    binding: Binding,
    role: u8,
    words: Zeroizing<Box<[u64]>>,
    #[allow(dead_code)] // Original source custody, never detached from native TR.
    source: Arc<SourceFor<C>>,
    lease: LeaseFor<C>,
}
macro_rules! private_views {
    ($name:ident) => {
        impl<C: AllocationContext> $name<C> {
            pub(super) fn binding(&self) -> Binding {
                self.binding
            }
            pub(super) fn values(&self) -> &[u64] {
                &self.words
            }
            pub(super) fn budget(&self) -> &Budget {
                self.lease.budget.budget()
            }
        }
    };
}
private_views!(DifferenceMaskFor);
private_views!(DifferenceValueFor);
private_views!(DenominatorMaskFor);
private_views!(DenominatorValueFor);
impl<C: AllocationContext> DifferenceValueFor<C> {
    pub(super) fn role(&self) -> u8 {
        self.role
    }
    pub(super) fn is_for(&self, mask: &DifferenceMaskFor<C>) -> bool {
        self.role < 2
            && self.binding == mask.binding
            && Arc::ptr_eq(&self.source, &mask.source)
            && std::ptr::eq(self.budget(), mask.budget())
    }
    pub(super) fn retained_bytes(&self) -> Result<usize, Error> {
        sum(&[
            self.lease.bytes,
            self.source.input.charged_bytes(),
            self.source.lease.bytes,
        ])
    }
    #[allow(dead_code)] // Sealed source-owner inspection; not a separate authority.
    pub(super) fn original_source_bytes(&self) -> usize {
        self.source.input.charged_bytes()
    }
}
impl<C: AllocationContext> DifferenceMaskFor<C> {
    pub(super) fn retained_bytes(&self) -> Result<usize, Error> {
        sum(&[
            self.lease.bytes,
            self.source.input.charged_bytes(),
            self.source.lease.bytes,
        ])
    }
    pub(super) fn original_source_bytes(&self) -> usize {
        self.source.input.charged_bytes()
    }
}
impl<C: AllocationContext> DenominatorValueFor<C> {
    pub(super) fn role(&self) -> u8 {
        self.role
    }
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Resource)
}
fn sum(parts: &[usize]) -> Result<usize, Error> {
    parts.iter().try_fold(0, |a, b| add(a, *b))
}
fn exact_words(n: usize, budget: &Budget) -> Result<Zeroizing<Box<[u64]>>, Error> {
    budget.allocation()?;
    Ok(Zeroizing::new(vec![0; n].into_boxed_slice()))
}
