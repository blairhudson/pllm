//! Preconstruction input binding using original creation seals and shared quotes.
use super::super::original_import::OriginalImportedView;
use super::compact_admission::{self, ConstructionPreflight, RequestOwners, RequestWork};
use super::*;
use crate::replicated_decoder::source_registry::{self, SourceRegistry};
use pllm_models::DecoderPlan;
use std::mem::size_of;

/// Borrow the original owners for the entire compact graph lifetime. Binding
/// vector spare capacity is charged; Arc backing belongs to the original Builder
/// and is already charged there. No source/template sharing by digest is inferred.
pub(super) struct InputOwner<'a> {
    _bindings: &'a Vec<OriginalImportedView<'a>>,
    _owners: &'a RequestOwners<'a>,
    pub(super) borrowed_bytes: usize,
    pub(super) metadata_bytes: usize,
    /// Enclosing outside inventory at original construction, excluding owned
    /// input-binding/plan metadata. Internal ledger value, never an import seal.
    pub(super) external_bytes: usize,
    pub(super) work: WorkBudget,
}

impl<'a> InputOwner<'a> {
    pub(super) fn original_bindings(&self) -> &[OriginalImportedView<'a>] {
        self._bindings
    }
}

struct BindingQuote {
    construction: ConstructionPreflight,
    outside_bytes: usize,
    borrowed_bytes: usize,
    work: WorkBudget,
    metadata_bytes: usize,
}

#[allow(clippy::too_many_arguments)]
fn quote(
    imports: &Builder,
    registry: &SourceRegistry<'_>,
    plan: &DecoderPlan,
    semantic: &DecoderGraph,
    start: usize,
    bindings: &Vec<OriginalImportedView<'_>>,
    prior: &RequestWork,
    owners: &RequestOwners<'_>,
    cap: usize,
) -> Result<BindingQuote, Error> {
    let (outside, work, _) = compact_admission::input_construction_baseline(prior, owners)?;
    if bindings.is_empty() || start >= semantic.operations.len() {
        return Err(Error::Binding);
    }
    let plan_remainder =
        source_registry::decoder_plan_metadata_outside_graph_bytes(plan, semantic, cap)?;
    for (index, input) in bindings.iter().enumerate() {
        input.check(imports, registry, plan, semantic)?;
        if !std::ptr::eq(input.registry(), registry)
            || bindings[..index]
                .iter()
                .any(|prior| prior.semantic() == input.semantic() || prior.owner() == input.owner())
            || semantic.operations[start..]
                .iter()
                .any(|op| op.id == input.semantic())
            || !semantic.operations[start..]
                .iter()
                .any(|op| op.inputs.iter().any(|id| id == input.semantic()))
        {
            return Err(Error::Binding);
        }
    }
    // All secret imports, including token/state inputs not used by a particular
    // operation, remain original owners. No omitted input or arithmetic prefix
    // can acquire aggregate authority from a late bind.
    for (owner, node) in imports.nodes.iter().enumerate() {
        if matches!(node.code, Code::Input(_))
            && !bindings.iter().any(|input| input.owner() == owner)
        {
            return Err(Error::Binding);
        }
    }
    for op in &semantic.operations[start..] {
        for id in &op.inputs {
            if semantic.operations[..start]
                .iter()
                .any(|producer| producer.id == *id)
                && !semantic.operations[start..]
                    .iter()
                    .any(|producer| producer.id == *id)
            {
                // Incoming prefix exports need a different genuine producer seal.
                return Err(Error::Binding);
            }
        }
    }
    let mut extra = plan_remainder;
    for bytes in [
        size_of::<Vec<OriginalImportedView<'_>>>(),
        array_bytes::<OriginalImportedView<'_>>(bindings.capacity())?,
        size_of::<SourceRegistry<'_>>(),
        size_of::<BindingQuote>(),
        size_of::<Result<BindingQuote, Error>>(),
        size_of::<InputOwner<'_>>(),
        2 * size_of::<PublishedView>(),
        source_registry::binding_workspace_bytes()?,
    ] {
        extra = add_bytes(extra, bytes)?;
    }
    let construction = compact_admission::preflight_bound_inputs(
        imports,
        semantic,
        start,
        registry.source_records(),
        registry.source_norms(),
        prior,
        owners,
        extra,
        cap,
    )?;
    let borrowed_bytes = add_bytes(
        construction.borrowed_source_bytes,
        construction.borrowed_semantic_bytes,
    )?;
    Ok(BindingQuote {
        construction,
        outside_bytes: add_bytes(outside, extra)?,
        borrowed_bytes,
        work,
        metadata_bytes: extra,
    })
}

impl<'a> CompactGraph<'a, 'a> {
    /// Original immutable input seals and all owner/work checks precede compact
    /// container factories. This is not an executable Plan or full-request gate.
    /// Borrowed source records are not cloned or reimported by this constructor.
    /// Outside inventory excludes imports, bindings, registry, complete plan and
    /// source maps. Those actual owners are added here, not caller byte estimates.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder::program) fn with_bound_inputs(
        imports: &'a Builder,
        validated: ValidatedSourceDescriptors<'a>,
        registry: &'a SourceRegistry<'a>,
        plan: &'a DecoderPlan,
        semantic: &'a DecoderGraph,
        start: usize,
        bindings: &'a Vec<OriginalImportedView<'a>>,
        prior: &RequestWork,
        owners: &'a RequestOwners<'a>,
        cap: usize,
    ) -> Result<Self, Error> {
        if !std::ptr::eq(validated.sources, registry.source_records())
            || !std::ptr::eq(validated.norms, registry.source_norms())
        {
            return Err(Error::Binding);
        }
        let quote = quote(
            imports, registry, plan, semantic, start, bindings, prior, owners, cap,
        )?;
        let mut graph = Self::with_limit_and_owner(
            imports,
            validated,
            semantic,
            start,
            cap,
            quote.outside_bytes,
        )?;
        graph.prebound_inputs = Some(InputOwner {
            _bindings: bindings,
            _owners: owners,
            borrowed_bytes: quote.borrowed_bytes,
            work: quote.work,
            metadata_bytes: quote.metadata_bytes,
            external_bytes: quote
                .outside_bytes
                .checked_sub(quote.metadata_bytes)
                .ok_or(Error::Binding)?,
        });
        // Historical normalization is the SAME existing freeze contract: exclude
        // source/semantic backing from each arm, then freeze adds both once.
        graph
            .construction_peak
            .set(quote.construction.construction_bytes);
        graph.construction_peak_without_source.set(
            quote
                .construction
                .construction_bytes
                .checked_sub(quote.borrowed_bytes)
                .ok_or(Error::Binding)?,
        );
        for input in bindings {
            if graph.bound_imports.len() == graph.bound_imports.capacity() {
                return Err(Error::Bounds);
            }
            let logical = input.logical();
            let view = PublishedView {
                origin: graph.origin.clone(),
                slot: None,
                mapping: Mapping::Dense(input.owner()),
                start: 0,
                lanes: logical.lanes()?,
                shape: logical.envelope()?,
                rank: logical.rank,
                logical,
                order: LaneOrder::Linear,
                import_slot: Some(graph.bound_imports.len()),
                bounds: input.bounds(),
            };
            graph.bound_imports.push(BoundImport {
                semantic: input.semantic(),
                view: view.clone(),
            });
            graph.validate_view(&view)?;
            graph.frontier.bind(input.semantic().to_owned(), view);
        }
        Ok(graph)
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static CONTAINER_FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
#[path = "compact_input_owner_tests.rs"]
mod tests;
