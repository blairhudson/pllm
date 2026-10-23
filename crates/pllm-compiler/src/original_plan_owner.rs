//! Closed compiler custody adapters; neither variant grants checkpoint trust.
//! The source variant owns the actual config-factory result, not a cloned or
//! serialized plan. Shared consumers borrow only while that custody remains live.

use crate::original_source::OriginalSourcePlan;
use pllm_garble::shared_allocation::{Budget, OwnedCharged};
use pllm_models::DecoderPlan;
use std::mem::size_of;

// Inline incoming owner is already charged; boxing adds an unrelated factory.
#[allow(clippy::large_enum_variant)]
pub(crate) enum OriginalPlanOwner {
    Charged(OwnedCharged<DecoderPlan>),
    Source(OriginalSourcePlan),
}

#[derive(Clone, Copy)]
pub(crate) enum OriginalPlanRef<'a> {
    Charged(&'a OwnedCharged<DecoderPlan>),
    Source(&'a OriginalSourcePlan),
}

impl OriginalPlanOwner {
    pub(crate) fn borrow(&self) -> OriginalPlanRef<'_> {
        match self {
            Self::Charged(plan) => OriginalPlanRef::Charged(plan),
            Self::Source(plan) => OriginalPlanRef::Source(plan),
        }
    }
}

impl<'a> OriginalPlanRef<'a> {
    pub(crate) fn plan(self) -> &'a DecoderPlan {
        match self {
            Self::Charged(plan) => plan.payload(),
            Self::Source(plan) => plan.plan(),
        }
    }

    pub(crate) fn budget(self) -> &'a Budget {
        match self {
            Self::Charged(plan) => plan.budget(),
            Self::Source(plan) => plan.budget(),
        }
    }

    pub(crate) fn backing_limit(self) -> usize {
        match self {
            Self::Charged(plan) => plan.charged_bytes(),
            // Captured by the genuine factory quote, admitted before that
            // factory, rechecked by its post-census and immutable thereafter.
            Self::Source(plan) => plan.resources().model_retained_bytes,
        }
    }

    pub(crate) fn reserved_bytes(self) -> usize {
        match self {
            Self::Charged(plan) => plan.charged_bytes(),
            // The original retained charge, NOT permission to release/reacquire
            // config/root/raw/Value reservations or lend their excess.
            Self::Source(plan) => plan.resources().working_charge_bytes,
        }
    }

    pub(crate) fn inline_bytes(self) -> usize {
        match self {
            Self::Charged(_) => size_of::<OwnedCharged<DecoderPlan>>(),
            Self::Source(_) => size_of::<OriginalSourcePlan>(),
        }
    }
}
