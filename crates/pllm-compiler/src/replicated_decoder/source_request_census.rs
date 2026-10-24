//! Borrowed owner census at the already-lowered source-validation boundary.
//!
//! No caller-labelled byte baseline is accepted. All retained charges come from
//! these actual owners, including both DecoderPlan graphs and spare capacities.
//! The graph must be the exact selected plan field; the exact shape map is kept
//! borrowed for the original descriptor validator. Neither structural equality
//! nor this resource census confers source/semantic validation authority.
//!
//! Quotes cover only the subsequent descriptor-validation or registry step.
//! They do NOT preadmit config parsing, lower_with_weights, source-shape-map
//! construction, or aggregate request construction. Other live owner categories
//! are outside this deliberately closed inventory, not implicitly zero bytes.

use super::{program, source_registry, Error, Manifest, PublicSource, SourceTensor};
use pllm_models::{owned_backing, DecoderGraph, DecoderPlan};
use std::{collections::BTreeMap, mem::size_of, sync::Arc};

/// Current retained observations, not bounds for constructing these owners.
/// Config bytes belong to Manifest; an optional separately retained parsed JSON
/// document is charged as another owner. No alias/sharing credits are taken.
pub(super) struct SourceRequestCensus<'a> {
    pub manifest: &'a Manifest,
    pub plan: &'a DecoderPlan,
    pub graph: &'a DecoderGraph,
    pub shapes: &'a BTreeMap<String, Vec<usize>>,
    pub incoming: Option<&'a Vec<u8>>,
    pub parsed_config: Option<&'a serde_json::Value>,
    pub public: Option<&'a Arc<PublicSource>>,
}

fn add(total: usize, bytes: usize, cap: usize) -> Result<usize, Error> {
    let total = source_registry::add_bytes(total, bytes)?;
    if total > cap.min(super::MAX_PROGRAM_BYTES) {
        return Err(Error::Bounds);
    }
    Ok(total)
}

/// Same full-node policy as shared model backing, including a possibly retained
/// empty root. Strings/vectors use original capacity, never logical shape size.
fn shape_map_bytes(shapes: &BTreeMap<String, Vec<usize>>, cap: usize) -> Result<usize, Error> {
    let mut bytes = add(0, size_of::<BTreeMap<String, Vec<usize>>>(), cap)?;
    if shapes.is_empty() {
        bytes = add(
            bytes,
            source_registry::tree_entry_bytes::<(String, Vec<usize>)>()?,
            cap,
        )?;
    }
    for (key, shape) in shapes {
        bytes = add(
            bytes,
            source_registry::tree_entry_bytes::<(String, Vec<usize>)>()?,
            cap,
        )?;
        bytes = add(bytes, key.capacity(), cap)?;
        bytes = add(
            bytes,
            source_registry::array_bytes::<usize>(shape.capacity())?,
            cap,
        )?;
    }
    Ok(bytes)
}

impl<'a> SourceRequestCensus<'a> {
    fn check_graph_identity(&self) -> Result<(), Error> {
        if !std::ptr::eq(self.graph, &self.plan.prefill)
            && !std::ptr::eq(self.graph, &self.plan.decode)
        {
            return Err(Error::Binding);
        }
        Ok(())
    }

    fn common_bytes(&self, mut bytes: usize, cap: usize) -> Result<usize, Error> {
        bytes = add(bytes, size_of::<Self>(), cap)?;
        if let Some(config) = self.parsed_config {
            bytes = add(bytes, size_of::<serde_json::Value>(), cap)?;
            let remaining = cap
                .min(super::MAX_PROGRAM_BYTES)
                .checked_sub(bytes)
                .ok_or(Error::Bounds)?;
            bytes = add(
                bytes,
                owned_backing::checked_json_backing_bytes(config, remaining)
                    .map_err(|_| Error::Bounds)?,
                cap,
            )?;
        }
        // The older registry census walks entries only. Preserve a conservative
        // allowance for each empty tree's potentially retained root as well.
        if self.manifest.tensors.is_empty() {
            bytes = add(
                bytes,
                source_registry::tree_entry_bytes::<(String, SourceTensor)>()?,
                cap,
            )?;
        }
        if self.manifest.norms.is_empty() {
            bytes = add(
                bytes,
                source_registry::tree_entry_bytes::<(String, super::BlockNorms)>()?,
                cap,
            )?;
        }
        if self.public.is_some() {
            // resident_metadata_bytes prices actual entries/key capacities and
            // file Arc owners, but exposes no empty-root state. Add one node
            // unconditionally: conservative overlap for nonempty maps, coverage
            // for an empty retained root. No private-file or alias inference.
            bytes = add(
                bytes,
                source_registry::tree_entry_bytes::<(String, pllm_core::public_file::PublicFile)>(
                )?,
                cap,
            )?;
        }
        Ok(bytes)
    }

    fn descriptor_owners(
        &self,
        cap: usize,
    ) -> Result<source_registry::SourceConstructionOwners<'a>, Error> {
        self.check_graph_identity()?;
        // Existing descriptor quote counts this exact graph (inline + backing).
        // Subtract only that field, after checking the COMPLETE plan's cap.
        let remainder =
            source_registry::decoder_plan_metadata_outside_graph_bytes(self.plan, self.graph, cap)?;
        let mut baseline = self.common_bytes(remainder, cap)?;
        if self.shapes.is_empty() {
            baseline = add(
                baseline,
                source_registry::tree_entry_bytes::<(String, Vec<usize>)>()?,
                cap,
            )?;
        }
        Ok(source_registry::SourceConstructionOwners {
            manifest: self.manifest,
            incoming: self.incoming,
            public: self.public,
            retained_baseline_bytes: baseline,
        })
    }

    fn registry_owners(
        &self,
        cap: usize,
    ) -> Result<source_registry::SourceConstructionOwners<'a>, Error> {
        self.check_graph_identity()?;
        // Registry construction counts neither graph nor shapes itself. Retain
        // the complete original plan plus this exact still-live shape map.
        let plan = source_registry::decoder_plan_metadata_bytes(self.plan, cap)?;
        let baseline = add(
            self.common_bytes(plan, cap)?,
            shape_map_bytes(self.shapes, cap)?,
            cap,
        )?;
        Ok(source_registry::SourceConstructionOwners {
            manifest: self.manifest,
            incoming: self.incoming,
            public: self.public,
            retained_baseline_bytes: baseline,
        })
    }

    /// Allocation-free preflight immediately before original record validation.
    /// The returned new-owned peak is prospective; its retained baseline is an
    /// observation of owners already built, NOT permission to have built them.
    pub(super) fn descriptor_validation_quote(
        &self,
        cap: usize,
    ) -> Result<source_registry::SourceConstructionQuote, Error> {
        source_registry::descriptor_validation_quote(
            self.graph,
            self.shapes,
            &self.descriptor_owners(cap)?,
            cap,
        )
    }

    /// Subsequent borrowed registry/hash step while all census owners remain live.
    pub(super) fn registry_construction_quote(
        &self,
        cap: usize,
    ) -> Result<source_registry::SourceConstructionQuote, Error> {
        source_registry::SourceRegistry::construction_quote_with_owners(
            &self.registry_owners(cap)?,
            cap,
        )
    }

    /// Consume only original compiler validation authority. Existing checked
    /// constructor enforces pointer identity of BOTH Manifest maps; this census
    /// never mints a token, clones a source, or replaces file authentication.
    pub(super) fn registry_from_validated_checked(
        &self,
        validated: program::ValidatedSourceDescriptors<'a>,
        cap: usize,
    ) -> Result<source_registry::SourceRegistry<'a>, Error> {
        source_registry::SourceRegistry::from_validated_with_owners_checked(
            validated,
            &self.registry_owners(cap)?,
            cap,
        )
    }

    /// Explicit unavailable gate. Positive stage quotes do not close upstream
    /// parsing/lowering/shape-generation peaks or unenumerated outside owners.
    pub(super) fn aggregate_constructor_completion_bound(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
#[path = "source_request_census_tests.rs"]
mod tests;
