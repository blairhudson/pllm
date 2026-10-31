//! Exact slot placement, not an encrypted kernel or capability issuer.
//!
//! Figures 5(d), 6(b,d) and §5.3 of the governing CacheMir PDF give:
//! K tile n/t, lane r=n%t; V group n/G and c=(floor((n%G)/t)+p-f)%p.
//! Here G=S/H=p*t. A coordinate starts at Geometry::output_slot(h,f).
//! Mask BEFORE rotating: all other source slots may contain arbitrary poison.
//! Rotation is out[j]=in[(j+s) mod S], hence s=-r moves a coordinate to
//! output_slot(h,f)+r without crossing a head/feature block.
//!
//! Binary masks and permutations have ideal infinity-norm gain <=1. Retained
//! old + contribution has error bounded by the SUM of operand errors. These
//! exact statements supply no encoded-mask, key-switch, rescale, level, scale,
//! noise or cost certificate. Such bounds/capabilities remain external and
//! mandatory. Mask fusion is algebraically possible, not a fused-depth claim.

use super::{AppendDescriptor, Geometry, LayoutError};

mod payload_reference;
#[cfg(test)]
mod payload_reference_tests;

/// Logical invocation census, not funded work, CPU/AES/backend cost or a tariff.
/// Counts API calls/predicates, not individual arithmetic instructions. Quotes
/// allocate nothing and do not debit a ledger. Every repetition costs again;
/// exhausting a fused iterator grants no free subsequent `next` calls.
/// Getters, size_hint/len, iterator/quote construction and quote arithmetic are
/// excluded, not CPU-free: this census names only the four event categories.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlacementWork {
    /// At most six descriptor predicates; exactly six for a successful new().
    pub constructor_checks: usize,
    pub recipe_next_calls: usize,
    pub mask_next_calls: usize,
    /// One selected(slot) invocation, including range rejection, is one event.
    pub mask_predicate_calls: usize,
}

impl PlacementWork {
    pub fn checked_add(self, other: Self) -> Result<Self, LayoutError> {
        Ok(Self {
            constructor_checks: self.constructor_checks.checked_add(other.constructor_checks).ok_or(LayoutError::Overflow)?,
            recipe_next_calls: self.recipe_next_calls.checked_add(other.recipe_next_calls).ok_or(LayoutError::Overflow)?,
            mask_next_calls: self.mask_next_calls.checked_add(other.mask_next_calls).ok_or(LayoutError::Overflow)?,
            mask_predicate_calls: self.mask_predicate_calls.checked_add(other.mask_predicate_calls).ok_or(LayoutError::Overflow)?,
        })
    }

    pub fn checked_repeated(self, calls: usize) -> Result<Self, LayoutError> {
        Ok(Self {
            constructor_checks: self.constructor_checks.checked_mul(calls).ok_or(LayoutError::Overflow)?,
            recipe_next_calls: self.recipe_next_calls.checked_mul(calls).ok_or(LayoutError::Overflow)?,
            mask_next_calls: self.mask_next_calls.checked_mul(calls).ok_or(LayoutError::Overflow)?,
            mask_predicate_calls: self.mask_predicate_calls.checked_mul(calls).ok_or(LayoutError::Overflow)?,
        })
    }
}

/// Hypothetical full-ring interpreter events, NOT work done by this recipe.
/// No payload buffers, execution, inactive fill, retirement or custody exist here.
/// An interpreter that separately initializes/copies buffers must quote that too.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementPayloadWork {
    pub source_mask_products: usize,
    pub rotation_slots: usize,
    pub destination_mask_products: usize,
    pub parent_additions: usize,
    pub inactive_initializations: usize,
}

impl PlacementPayloadWork {
    pub fn checked_repeated(self, calls: usize) -> Result<Self, LayoutError> {
        Ok(Self {
            source_mask_products: self.source_mask_products.checked_mul(calls).ok_or(LayoutError::Overflow)?,
            rotation_slots: self.rotation_slots.checked_mul(calls).ok_or(LayoutError::Overflow)?,
            destination_mask_products: self.destination_mask_products.checked_mul(calls).ok_or(LayoutError::Overflow)?,
            parent_additions: self.parent_additions.checked_mul(calls).ok_or(LayoutError::Overflow)?,
            inactive_initializations: self.inactive_initializations.checked_mul(calls).ok_or(LayoutError::Overflow)?,
        })
    }
}

// Instrument actual predicate/recipe entry points, not quote calculations.
#[cfg(test)]
thread_local! {
    pub(super) static CENSUS: std::cell::Cell<PlacementWork> = const {
        std::cell::Cell::new(PlacementWork {
            constructor_checks: 0, recipe_next_calls: 0,
            mask_next_calls: 0, mask_predicate_calls: 0,
        })
    };
}

fn descriptor_check(condition: bool) -> bool {
    #[cfg(test)]
    CENSUS.with(|c| { let mut w = c.get(); w.constructor_checks += 1; c.set(w); });
    condition
}

/// Ordered destinations are K first, then absolute V ciphertexts in increasing
/// c within the current group. K and V are separate storage families.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementDestination {
    Key { tile: usize },
    Value { group: usize, ciphertext: usize },
}

/// Allocation-free explicit binary mask. Iteration emits every slot, including
/// zeros; `selected` rejects out-of-range indices rather than wrapping them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementMask {
    geometry: Geometry,
    lane: usize,
    feature: Option<usize>,
}

struct MaskBits {
    mask: PlacementMask,
    slots: core::ops::Range<usize>,
}

impl Iterator for MaskBits {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        #[cfg(test)]
        CENSUS.with(|c| { let mut w = c.get(); w.mask_next_calls += 1; c.set(w); });
        self.slots.next().map(|slot| u8::from(self.mask.selected(slot).unwrap()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) { self.slots.size_hint() }
}

impl ExactSizeIterator for MaskBits {}
impl core::iter::FusedIterator for MaskBits {}

impl PlacementMask {
    pub fn query_work(self) -> PlacementWork {
        PlacementWork { mask_predicate_calls: 1, ..PlacementWork::default() }
    }

    /// One COMPLETE bits() scan, including zeros and final None. No scan occurs
    /// when quoting or merely constructing bits(). Re-scan requires another quote.
    /// Extra next() calls after None each add one mask_next_calls event, but do
    /// not invoke selected(). Direct selected() calls use query_work() instead.
    pub fn scan_work(self) -> Result<PlacementWork, LayoutError> {
        Ok(PlacementWork {
            mask_next_calls: self.geometry.slots().checked_add(1).ok_or(LayoutError::Overflow)?,
            mask_predicate_calls: self.geometry.slots(),
            ..PlacementWork::default()
        })
    }

    pub fn selected(self, slot: usize) -> Result<bool, LayoutError> {
        #[cfg(test)]
        CENSUS.with(|c| { let mut w = c.get(); w.mask_predicate_calls += 1; c.set(w); });
        if slot >= self.geometry.slots() {
            return Err(LayoutError::Index);
        }
        let block = slot / self.geometry.tokens_per_key_tile();
        Ok(slot % self.geometry.tokens_per_key_tile() == self.lane
            && self.feature.is_none_or(|f| block / self.geometry.heads() == f))
    }

    pub fn bits(self) -> impl ExactSizeIterator<Item = u8> + core::iter::FusedIterator {
        MaskBits { mask: self, slots: 0..self.geometry.slots() }
    }
}

/// Execute source mask, signed slot rotation, destination mask, in that order.
/// If `retain_parent`, add the old destination into a FRESH output. Otherwise
/// the contribution is a fresh boundary destination; no old/zero owner is needed.
/// This record carries no ciphertext ownership or authorization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementContribution {
    pub destination: PlacementDestination,
    pub source_mask: PlacementMask,
    pub rotation: i32,
    pub destination_mask: PlacementMask,
    pub retain_parent: bool,
}

impl PlacementContribution {
    /// Scenario: full S-slot source mask, full ring rotation (even shift zero),
    /// full S-slot destination mask, then S additions iff retaining a parent.
    /// Separate from source_mask.scan_work()/destination_mask.scan_work(): those
    /// quote actual mask API traversal; this quotes only future payload events.
    pub fn ideal_payload_work(self) -> PlacementPayloadWork {
        let slots = self.source_mask.geometry.slots();
        PlacementPayloadWork {
            source_mask_products: slots,
            rotation_slots: slots,
            destination_mask_products: slots,
            parent_additions: if self.retain_parent { slots } else { 0 },
            inactive_initializations: 0,
        }
    }
}

/// Finite, allocation-free placement recipe; exactly 1+head_dim contributions.
/// Descriptor validation is constant-time and precedes any expansion. Consumers
/// must debit their EXISTING cumulative ledgers before expanding/building SSA.
#[derive(Clone, Debug)]
pub struct AppendPlacement {
    geometry: Geometry,
    append: AppendDescriptor,
    next: usize,
}

impl AppendPlacement {
    /// Upper bound on constructor predicates, independent of descriptor validity.
    /// Short-circuit refusal may use fewer; successful construction uses all six.
    pub fn construction_work() -> PlacementWork {
        PlacementWork { constructor_checks: 6, ..PlacementWork::default() }
    }

    /// Exactly one next() call, including every call after exhaustion.
    pub fn next_work() -> PlacementWork {
        PlacementWork { recipe_next_calls: 1, ..PlacementWork::default() }
    }

    /// Remaining contributions plus ONE final None call. Does not scan masks,
    /// rotate slots, initialize storage or mutate payload. Clones/replays pay again.
    pub fn traversal_work(&self) -> Result<PlacementWork, LayoutError> {
        let calls = self.len().checked_add(1).ok_or(LayoutError::Overflow)?;
        Self::next_work().checked_repeated(calls)
    }

    pub fn new(geometry: Geometry, append: AppendDescriptor) -> Result<Self, LayoutError> {
        let t = geometry.tokens_per_key_tile();
        let group = geometry.tokens_per_value_group();
        if descriptor_check(append.token >= geometry.capacity()) {
            return Err(LayoutError::Capacity);
        }
        if descriptor_check(append.key_tile != append.token / t)
            || descriptor_check(append.key_lane != append.token % t)
            || descriptor_check(append.value_group != append.token / group)
            || descriptor_check(append.new_key_ciphertexts != usize::from(append.token % t == 0))
            || descriptor_check(append.new_value_ciphertexts
                != if append.token % group == 0 { geometry.head_dim() } else { 0 })
        {
            return Err(LayoutError::Geometry);
        }
        Ok(Self { geometry, append, next: 0 })
    }
}

impl Iterator for AppendPlacement {
    type Item = PlacementContribution;

    fn next(&mut self) -> Option<Self::Item> {
        #[cfg(test)]
        CENSUS.with(|c| { let mut w = c.get(); w.recipe_next_calls += 1; c.set(w); });
        let g = self.geometry;
        if self.next > g.head_dim() {
            return None;
        }
        let a = self.append;
        let (destination, feature, retain_parent) = if self.next == 0 {
            (PlacementDestination::Key { tile: a.key_tile }, None, a.new_key_ciphertexts == 0)
        } else {
            let c = self.next - 1;
            let q = (a.token % g.tokens_per_value_group()) / g.tokens_per_key_tile();
            let f = (q + g.head_dim() - c) % g.head_dim();
            (
                PlacementDestination::Value {
                    group: a.value_group,
                    ciphertext: a.value_group * g.head_dim() + c,
                },
                Some(f),
                a.new_value_ciphertexts == 0,
            )
        };
        self.next += 1;
        Some(PlacementContribution {
            destination,
            source_mask: PlacementMask { geometry: g, lane: 0, feature },
            // Geometry caps S at 32768, so the cast and negation are exact.
            rotation: -(a.key_lane as i32),
            destination_mask: PlacementMask { geometry: g, lane: a.key_lane, feature },
            retain_parent,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.geometry.head_dim() + 1 - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for AppendPlacement {}
impl core::iter::FusedIterator for AppendPlacement {}
