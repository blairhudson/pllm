//! Additive enclosing-owner admission for the current compact publication path.
//!
//! Declare this module INSIDE `compact_lowering`, so the census reads the actual
//! private CompactGraph owners rather than a caller's descriptor-size estimate.
//! This is a quote, not an execution/source/certificate capability. The current
//! graph exposes original physical layouts but no request-wide retirement map:
//! body liveness comes from an actual layer::Plan, while outside owners remain
//! charged independently. Early preflight below precedes compact containers and
//! source-record clones, not the upstream model graph/validator construction.
//! Earlier graph/source-validation construction must be preadmitted by its owner;
//! observing a finished CompactGraph cannot retroactively admit that allocation.
use super::{
    frozen_region::FrozenRegion, Code, CompactGraph, DecoderGraph, DescriptorKind, Error,
    PhysicalLayout, PublishedOp, PublishedView, WorkBudget, MAX_NODES, MAX_PROGRAM_BYTES,
};
use crate::replicated_decoder::{
    multiply::PartyRuntime,
    runtime::body::{
        handoff::TRANSITION_INLINE_BYTES,
        layer::{self, Layout, LayoutQuote},
        Register,
    },
    source_norms,
    source_registry::{SourceConstructionQuote, SourceRegistry},
    tensor, PublicSourceImport, Secret, SourceNorms, SourceTensor, Worker,
};
use std::{collections::BTreeMap, mem::size_of};

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn array<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
}

fn tree_entry<T>() -> Result<usize, Error> {
    crate::replicated_decoder::source_registry::tree_entry_bytes::<T>()
}

/// Actual borrowed source capacities, plus the simultaneous new record-clone
/// destinations used by source import. No authenticated file is cloned or given
/// authority here. Opaque importer/file buffers remain owned_source_bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SourceOwnerCensus {
    pub borrowed_bytes: usize,
    pub record_clone_bytes: usize,
}

impl SourceOwnerCensus {
    fn read(sources: &BTreeMap<String, SourceTensor>, norms: &SourceNorms) -> Result<Self, Error> {
        let borrowed =
            crate::replicated_decoder::source_registry::source_metadata_bytes(sources, norms)?;
        let mut clones = add(
            add(
                size_of::<Vec<(String, SourceTensor)>>(),
                size_of::<BTreeMap<String, SourceTensor>>(),
            )?,
            array::<(String, SourceTensor)>(sources.len())?,
        )?;
        for (key, source) in sources {
            let mut backing = add(key.capacity(), array::<usize>(source.shape.capacity())?)?;
            if let Some(values) = &source.public_q18 {
                backing = add(backing, array::<i64>(values.capacity())?)?;
            }
            let record = add(backing, tree_entry::<(String, SourceTensor)>()?)?;
            // Count both map and Vec destinations conservatively, never infer
            // sharing from equal source or program digests.
            clones = add(clones, record)?;
        }
        let norm = source_norms::import_envelope(norms)?;
        // Empty maps still have inline destination owners. Nonempty norm import
        // retains its original four-set overlap policy; no payload is refunded.
        clones = add(clones, norm.total_bytes.max(4 * size_of::<SourceNorms>()))?;
        Ok(Self {
            borrowed_bytes: borrowed,
            record_clone_bytes: clones,
        })
    }
}

/// Existing semantic graph backing is borrowed, not newly allocated by the
/// compact preflight. Reuse source admission's bounded capacity census so its
/// nested-attribute and owner contracts cannot drift between request phases.
fn semantic_bytes(graph: &DecoderGraph) -> Result<usize, Error> {
    crate::replicated_decoder::source_registry::semantic_metadata_bytes(graph)
}

/// Read-only physical census, NOT a runtime allocation/liveness authorization.
/// Output slots follow actual Dense/BinaryTiles layouts, never virtual node IDs.
/// Until a retirement/export mapping exists, retain every published output; do
/// not assert that this conservative envelope rescues the target decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CompactOwnerCensus {
    pub metadata_bytes: usize,
    pub borrowed_source_bytes: usize,
    pub borrowed_semantic_bytes: usize,
    pub prepared_program_bytes: usize,
    pub output_slots: usize,
    pub output_slot_bytes: usize,
    pub retained_output_bytes: usize,
    pub stored_descriptors: usize,
    /// None when no original-kernel body publication mapping exists yet. This
    /// does not prevent a metadata census; it DOES prevent body construction.
    pub body_descriptors: Option<usize>,
    pub remaining_operations: usize,
    pub work: WorkBudget,
}

impl CompactOwnerCensus {
    fn read(graph: &CompactGraph<'_, '_>) -> Result<Self, Error> {
        let containers = graph.container_census()?;
        let mut slots = 0usize;
        let mut payload = 0usize;
        let mut body = Some(0usize);
        for descriptor in graph.descriptor_census() {
            let alias = matches!(
                &descriptor.kind,
                DescriptorKind::LastToken
                    | DescriptorKind::TokenFeedback
                    | DescriptorKind::Reshape
                    | DescriptorKind::CausalMask
            );
            let count = match descriptor.output.layout {
                PhysicalLayout::CausalTensorRows { rows } => add(rows, 1)?,
                PhysicalLayout::CausalProducts { products } => add(products, 1)?,
                PhysicalLayout::Dense { .. } => 1,
                PhysicalLayout::BinaryTiles { tile_lanes, .. } if tile_lanes > 0 => {
                    descriptor.output.lanes.div_ceil(tile_lanes)
                }
                PhysicalLayout::TensorRows { row_width, .. } if row_width > 0 => add(
                    descriptor.output.offset % row_width,
                    descriptor.output.lanes,
                )?
                .div_ceil(row_width),
                _ => return Err(Error::Binding),
            };
            // LastToken, TokenFeedback and Reshape retain original view owners,
            // not another Secret/buffer allocation. Their descriptor/mapping and
            // frontier backing remain charged by graph.baseline(); no prior
            // value owner, request work or certificate ledger is refunded here.
            if !alias {
                slots = add(slots, count)?;
                payload = add(
                    payload,
                    descriptor
                        .output
                        .lanes
                        .checked_mul(16)
                        .ok_or(Error::Bounds)?,
                )?;
            }
            #[allow(unreachable_patterns)]
            let operations = match descriptor.kind {
                DescriptorKind::Projection { bias, .. } => Some(2 + usize::from(bias.is_some())), // raw matrix + original floor + optional BODY bias
                DescriptorKind::GuardedSilu | DescriptorKind::Binary { .. } => Some(1),
                DescriptorKind::LastToken
                | DescriptorKind::TokenFeedback
                | DescriptorKind::Reshape => Some(0),
                // New emitter families need their actual body mapping, not an
                // assumed one-op replacement or a silently incomplete quote.
                _ => None,
            };
            body = match (body, operations) {
                (Some(total), Some(count)) => Some(add(total, count)?),
                _ => None,
            };
        }
        let mut prepared = 0;
        for program in graph.owned_prepared_programs() {
            prepared = add(
                prepared,
                add(
                    program.resources()?.immutable_template_bytes,
                    2 * size_of::<usize>(),
                )?,
            )?;
        }
        let sources = SourceOwnerCensus::read(
            graph.borrowed_source_records(),
            graph.borrowed_source_norms(),
        )?;
        Ok(Self {
            metadata_bytes: graph.retained_metadata_bytes()?,
            borrowed_source_bytes: sources.borrowed_bytes,
            borrowed_semantic_bytes: semantic_bytes(graph.frontier.semantic_graph())?,
            prepared_program_bytes: prepared, // diagnostic subset; baseline already charges it
            output_slots: slots,
            output_slot_bytes: array::<Option<Secret>>(slots)?,
            retained_output_bytes: payload,
            stored_descriptors: add(
                containers.imported_descriptors,
                containers.stored_operations,
            )?,
            body_descriptors: body,
            remaining_operations: graph
                .frontier
                .operations()
                .len()
                .checked_sub(graph.cursor)
                .ok_or(Error::Binding)?,
            work: containers.partial_work,
        })
    }
}

/// One request's accumulated original work and still-retained graph metadata.
/// Repeated selections/regions are charged repeatedly, even with equal digests.
/// No reset, refund or completion API. The caller retains every recorded graph or
/// frozen region; this snapshot confers no execution or certificate authority.
#[derive(Debug, Default)]
pub(in crate::replicated_decoder::program) struct RequestWork {
    work: WorkBudget,
    stored_descriptors: usize,
    retained_graph_bytes: usize,
    regions: usize,
}

impl RequestWork {
    pub(super) fn record_graph(&mut self, graph: &CompactGraph<'_, '_>) -> Result<(), Error> {
        if graph.failed {
            return Err(Error::Retired);
        }
        let containers = graph.container_census()?;
        self.record_region(
            containers.partial_work,
            || {
                add(
                    containers.imported_descriptors,
                    containers.stored_operations,
                )
            },
            || {
                let sources = SourceOwnerCensus::read(
                    graph.borrowed_source_records(),
                    graph.borrowed_source_norms(),
                )?;
                add(
                    add(graph.retained_metadata_bytes()?, sources.borrowed_bytes)?,
                    semantic_bytes(graph.frontier.semantic_graph())?,
                )
            },
        )
    }

    /// Record the actual consuming freeze result, without its former semantic
    /// graph. This snapshot does not retain the region or admit a request: the
    /// enclosing owner keeps it and its original source/import borrows alive.
    /// Retained resources include their existing outside-owner allowance; no
    /// sharing credit, historical-peak refund or completion authority is inferred.
    pub(super) fn record_frozen(&mut self, region: &FrozenRegion<'_>) -> Result<(), Error> {
        self.record_region(
            region.work_budget(),
            || {
                // All original imports count, including public source recipes
                // which need not have a bound semantic input. import_count in
                // FreezeResources counts only bindings, NOT those descriptors.
                let mut imports = 0;
                while region.imported_recipe(imports).is_some() {
                    imports = add(imports, 1)?;
                }
                region
                    .descriptor_census()
                    .try_fold(imports, |count, _| add(count, 1))
            },
            || Ok(region.resources().retained_bytes),
        )
    }

    /// Shared atomic policy; only the compiler-owned census paths above supply
    /// these allocation-free closures. Keep work/descriptor/retained-owner check
    /// order unchanged for the borrowed graph path.
    fn record_region(
        &mut self,
        region_work: WorkBudget,
        region_descriptors: impl FnOnce() -> Result<usize, Error>,
        region_bytes: impl FnOnce() -> Result<usize, Error>,
    ) -> Result<(), Error> {
        let work = self.work.checked_add(region_work)?;
        work.limits()?;
        let descriptors = add(self.stored_descriptors, region_descriptors()?)?;
        if descriptors > MAX_NODES {
            return Err(Error::Bounds);
        }
        let bytes = add(self.retained_graph_bytes, region_bytes()?)?;
        if bytes > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let regions = add(self.regions, 1)?;
        // Commit only after every overflow/ceiling check, preserving earlier
        // request work and owners on failure.
        self.work = work;
        self.stored_descriptors = descriptors;
        self.retained_graph_bytes = bytes;
        self.regions = regions;
        Ok(())
    }
    pub(super) fn work(&self) -> WorkBudget {
        self.work
    }

    /// Register an actual retained body Plan in addition to its compact graph.
    /// The graph already supplies invocation work; do not count it a second time.
    /// Independent ordinary invocation work still belongs in record_work. This
    /// snapshot neither retains the Plan nor authorizes execution: the enclosing
    /// owner must keep the original Arc and register each still-live Plan once.
    pub(super) fn record_retained_plan(
        &mut self,
        plan: &std::sync::Arc<layer::Plan>,
    ) -> Result<(), Error> {
        let descriptors = add(self.stored_descriptors, plan.semantic_sites())?;
        let bytes = add(
            self.retained_graph_bytes,
            add(
                add(
                    plan.resources().immutable_metadata_bytes,
                    // No separate retained/allocation-only breakdown exists:
                    // preserve the original whole owner/Arc allowance rather
                    // than guessing which part may be refunded after finish.
                    plan.resources().owner_and_arc_allowance_bytes,
                )?,
                2 * size_of::<usize>(),
            )?,
        )?;
        if descriptors > MAX_NODES || bytes > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        self.stored_descriptors = descriptors;
        self.retained_graph_bytes = bytes;
        Ok(())
    }
    /// Original compiler-owned ordinary schedule work outside compact regions.
    /// This appends to the SAME request; failure never replaces its prior budget.
    pub(super) fn record_work(&mut self, work: WorkBudget) -> Result<(), Error> {
        let total = self.work.checked_add(work)?;
        total.limits()?;
        self.work = total;
        Ok(())
    }
}

/// Early all-owner container/source-record construction peak. This deliberately
/// reports unlowered operations, never a whole-decoder completion/admission bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ConstructionPreflight {
    pub borrowed_source_bytes: usize,
    pub borrowed_semantic_bytes: usize,
    pub new_source_record_bytes: usize,
    /// Present only for the actual registry-backed importer path. Recomputed
    /// from original owners, never accepted as a caller-supplied certificate.
    /// Its complete new-owned peak replaces the generic record-clone estimate.
    pub source_import_quote: Option<SourceConstructionQuote>,
    pub graph_container_bytes: usize,
    /// Current live construction arm, before maxing with an earlier request peak.
    pub active_construction_bytes: usize,
    pub construction_bytes: usize,
    pub accumulated_work: WorkBudget,
    pub reserved_descriptors: usize,
    pub unlowered_operations: usize,
}

/// Call BEFORE CompactGraph::new/frontier/descriptor allocation and BEFORE source
/// record cloning. `semantic` and source records already exist and are borrowed.
/// Upstream lowering/validation/Pending/PublicFile construction needs its own
/// owner quote; this cannot preadmit those opaque allocations retroactively.
pub(super) fn preflight_graph_construction(
    imports: &super::Builder,
    semantic: &DecoderGraph,
    start: usize,
    sources: &BTreeMap<String, SourceTensor>,
    norms: &SourceNorms,
    prior: &RequestWork,
    owners: &RequestOwners<'_>,
) -> Result<ConstructionPreflight, Error> {
    preflight_graph_with_limit(
        imports,
        semantic,
        start,
        sources,
        norms,
        prior,
        owners,
        MAX_PROGRAM_BYTES,
    )
}

/// Preferred entry when the original borrowing registry already exists. Call
/// before CompactGraph construction AND PublicSourceImport::new_checked. Counts
/// actual Pending/file/scan/frame/control layouts through that constructor's
/// shared quote, not a guessed source-clone allowance. Registry minting and the
/// upstream model/source validator still need their original early preflights.
pub(super) fn preflight_graph_and_source_import(
    imports: &super::Builder,
    semantic: &DecoderGraph,
    start: usize,
    registry: &SourceRegistry<'_>,
    prior: &RequestWork,
    owners: &RequestOwners<'_>,
) -> Result<ConstructionPreflight, Error> {
    graph_source_preflight(
        imports,
        semantic,
        start,
        registry.source_records(),
        registry.source_norms(),
        prior,
        owners,
        Some(registry),
        MAX_PROGRAM_BYTES,
    )
}

#[allow(clippy::too_many_arguments)]
fn preflight_graph_with_limit(
    imports: &super::Builder,
    semantic: &DecoderGraph,
    start: usize,
    sources: &BTreeMap<String, SourceTensor>,
    norms: &SourceNorms,
    prior: &RequestWork,
    owners: &RequestOwners<'_>,
    limit: usize,
) -> Result<ConstructionPreflight, Error> {
    graph_source_preflight(
        imports, semantic, start, sources, norms, prior, owners, None, limit,
    )
}

#[allow(clippy::too_many_arguments)]
fn graph_source_preflight(
    imports: &super::Builder,
    semantic: &DecoderGraph,
    start: usize,
    sources: &BTreeMap<String, SourceTensor>,
    norms: &SourceNorms,
    prior: &RequestWork,
    owners: &RequestOwners<'_>,
    registry: Option<&SourceRegistry<'_>>,
    limit: usize,
) -> Result<ConstructionPreflight, Error> {
    graph_input_preflight(
        imports, semantic, start, sources, norms, prior, owners, registry, limit, 0, false,
    )
}

/// Borrow-only source records for the sealed, prebound constructor. It allocates
/// no source-record clone/importer, and charges actual binding/plan owners once.
#[allow(clippy::too_many_arguments)]
pub(super) fn preflight_bound_inputs(
    imports: &super::Builder,
    semantic: &DecoderGraph,
    start: usize,
    sources: &BTreeMap<String, SourceTensor>,
    norms: &SourceNorms,
    prior: &RequestWork,
    owners: &RequestOwners<'_>,
    extra_owners: usize,
    limit: usize,
) -> Result<ConstructionPreflight, Error> {
    // Pending original full/tail contraction work is known without constructing
    // kernels. Price every occurrence, including repeated equal-digest recipes.
    let pending = super::pending_contraction_work(imports, semantic, start, sources)?;
    let aggregate = prior
        .work
        .checked_add(owners.other_work)?
        .checked_add(pending)?;
    aggregate.limits()?;
    let mut quote = graph_input_preflight(
        imports,
        semantic,
        start,
        sources,
        norms,
        prior,
        owners,
        None,
        limit,
        extra_owners,
        true,
    )?;
    let (norm_work, work_peak) = super::compact_validity::pending_input_work(
        imports,
        semantic,
        start,
        sources,
        quote.active_construction_bytes,
        limit,
    )?;
    quote.accumulated_work = aggregate.checked_add(norm_work)?;
    quote.accumulated_work.limits()?;
    quote.active_construction_bytes = quote.active_construction_bytes.max(work_peak);
    quote.construction_bytes = quote.construction_bytes.max(work_peak);
    Ok(quote)
}

#[allow(clippy::too_many_arguments)]
fn graph_input_preflight(
    imports: &super::Builder,
    semantic: &DecoderGraph,
    start: usize,
    sources: &BTreeMap<String, SourceTensor>,
    norms: &SourceNorms,
    prior: &RequestWork,
    owners: &RequestOwners<'_>,
    registry: Option<&SourceRegistry<'_>>,
    limit: usize,
    extra_owners: usize,
    borrow_only: bool,
) -> Result<ConstructionPreflight, Error> {
    let operations = semantic
        .operations
        .len()
        .checked_sub(start)
        .filter(|n| *n != 0)
        .ok_or(Error::Bounds)?;
    let reserved = add(
        add(prior.stored_descriptors, owners.other_stored_descriptors)?,
        add(imports.nodes.len(), operations)?,
    )?;
    if reserved > MAX_NODES {
        return Err(Error::Bounds);
    }
    let work = prior.work.checked_add(owners.other_work)?;
    work.limits()?;
    let source = SourceOwnerCensus::read(sources, norms)?;
    let semantic_bytes = semantic_bytes(semantic)?;
    let mut graph_bytes = CompactGraph::imports_bytes(imports)?;
    for bytes in [
        size_of::<CompactGraph<'_, '_>>(),
        array::<PublishedOp<'_>>(operations)?,
        super::graph::OrderedGraph::<PublishedView>::storage_bound(semantic)?,
        // The constructor reserves all semantic input references, not merely
        // the imports bound so far or the suffix's distinct input names.
        CompactGraph::import_binding_storage_bound(semantic)?,
        2 * size_of::<Result<CompactGraph<'_, '_>, Error>>(),
        2 * size_of::<usize>(),
        4 * size_of::<PublishedOp<'_>>(),
    ] {
        graph_bytes = add(graph_bytes, bytes)?;
    }
    let mut total = add(add(graph_bytes, outside_bytes(owners)?)?, extra_owners)?;
    for bytes in [
        semantic_bytes,
        prior.retained_graph_bytes,
        size_of::<RequestWork>(),
        size_of::<ConstructionPreflight>(),
        size_of::<Result<ConstructionPreflight, Error>>(),
        size_of::<SourceOwnerCensus>(),
        size_of::<Result<SourceOwnerCensus, Error>>(),
    ] {
        total = add(total, bytes)?;
    }
    let source_import_quote = if let Some(registry) = registry {
        if !std::ptr::eq(sources, registry.source_records())
            || !std::ptr::eq(norms, registry.source_norms())
        {
            return Err(Error::Binding);
        }
        // Source maps are a separate disjoint owner set in the native quote.
        // Compute the outside baseline directly; never subtract/refund a live
        // owner or hide a newly allocated backing in the borrowed category.
        let quote = PublicSourceImport::construction_quote(
            registry,
            add(total, size_of::<SourceRegistry<'_>>())?,
        )?;
        quote.admit(limit)?;
        total = quote.peak_bytes;
        Some(quote)
    } else {
        total = add(total, source.borrowed_bytes)?;
        if !borrow_only {
            total = add(total, source.record_clone_bytes)?;
        }
        None
    };
    let active_construction_bytes = total;
    total = total.max(owners.prior_request_peak_bytes);
    if total > limit.min(MAX_PROGRAM_BYTES) {
        return Err(Error::Bounds);
    }
    Ok(ConstructionPreflight {
        borrowed_source_bytes: source.borrowed_bytes,
        borrowed_semantic_bytes: semantic_bytes,
        new_source_record_bytes: if source_import_quote.is_some() || borrow_only {
            0
        } else {
            source.record_clone_bytes
        },
        source_import_quote,
        graph_container_bytes: graph_bytes,
        active_construction_bytes,
        construction_bytes: total,
        accumulated_work: work,
        reserved_descriptors: reserved,
        unlowered_operations: operations,
    })
}

pub(in crate::replicated_decoder::program) fn outside_bytes(
    owners: &RequestOwners<'_>,
) -> Result<usize, Error> {
    let mut bytes = 0;
    for n in [
        owners.borrowed_external_bytes,
        owners.owned_source_bytes,
        owners.other_owned_metadata_bytes,
        retained_values_bytes(owners.retained_values)?,
        owners.external_frame_bytes,
        owners.caller_control_bytes,
        size_of::<Worker>(),
        TRANSITION_INLINE_BYTES,
        size_of::<RequestOwners<'_>>(),
        size_of::<RequestQuote<'_, '_, '_>>(),
        size_of::<Result<RequestQuote<'_, '_, '_>, Error>>(),
        size_of::<RequestPeak>(),
        size_of::<Result<RequestPeak, Error>>(),
    ] {
        bytes = add(bytes, n)?;
    }
    Ok(bytes)
}

/// Explicit OUTSIDE owners, borrowed for the entire quote's lifetime. None of
/// these byte counts confers source, row-proof, runtime or transport authority.
/// No field has a default: the integrating compiler must inventory every owner.
pub(in crate::replicated_decoder::program) struct RequestOwners<'a> {
    /// Already existing manifest/file-handle resident baseline OUTSIDE the source
    /// records/norms and semantic graph automatically inventoried by this module
    /// during both early construction and later request/body phases.
    /// Borrowed does not mean free. Do not charge paged checkpoint file length as
    /// resident tensor backing; do include retained metadata and actual buffers.
    pub borrowed_external_bytes: usize,
    /// Newly owned importer/source backing, including active import buffers.
    /// Do not place new validation maps in the borrowed category.
    pub owned_source_bytes: usize,
    /// Every retained descriptor/template/slot owner outside CompactGraph.
    /// Include prior regions and the rest of this same complete request. There
    /// is no segment reset, digest-sharing credit or retirement refund here.
    pub other_owned_metadata_bytes: usize,
    pub other_stored_descriptors: usize,
    /// All work OUTSIDE this CompactGraph, including every selection, ordinary
    /// prefix/suffix, lookup, head, acceptance and feedback operation. Body work
    /// replacing this graph must not be added again. Never pass a per-segment
    /// budget in place of the complete outside request inventory.
    pub other_work: WorkBudget,
    /// Actual retained physical slots/shares outside the body Plan. Capacity,
    /// not length, is charged, including both components. The Vec is borrowed;
    /// no share or PRSS owner is cloned, moved, retired or manufactured here.
    pub retained_values: &'a Vec<Option<Secret>>,
    /// Router/held-frame backing not covered by the original primitive leases.
    /// Existing leases remain charged through Plan resources; never subtract
    /// them merely because a frame left the executor or crossed a boundary.
    pub external_frame_bytes: usize,
    /// Enclosing invocation/result slots, layouts, control containers and any
    /// other simultaneously retained backing not already included above.
    /// Worker and this module's own quote/result slots are added separately.
    pub caller_control_bytes: usize,
    /// Earlier ALL-OWNER peak, not a per-owner or per-region maximum. This keeps
    /// earlier preadmitted construction pressure in the final phase maximum;
    /// still-live earlier backing must ALSO remain in the fields above.
    pub prior_request_peak_bytes: usize,
}

/// Shared outside-owner ledger for original input declaration and prebound graph
/// construction. Source maps, the original plan and imports are separate owners.
pub(in crate::replicated_decoder::program) fn input_construction_baseline(
    prior: &RequestWork,
    owners: &RequestOwners<'_>,
) -> Result<(usize, WorkBudget, usize), Error> {
    let work = prior.work.checked_add(owners.other_work)?;
    work.limits()?;
    let descriptors = add(prior.stored_descriptors, owners.other_stored_descriptors)?;
    if descriptors > MAX_NODES {
        return Err(Error::Bounds);
    }
    let bytes = add(
        add(outside_bytes(owners)?, prior.retained_graph_bytes)?,
        size_of::<RequestWork>(),
    )?;
    Ok((bytes, work, descriptors))
}

/// Allocation-free physical slot census. Repeated ownership is conservatively
/// charged; there is no alias or digest deduplication and no payload refund.
fn retained_values_bytes(values: &Vec<Option<Secret>>) -> Result<usize, Error> {
    let mut bytes = add(
        size_of::<Vec<Option<Secret>>>(),
        array::<Option<Secret>>(values.capacity())?,
    )?;
    for value in values.iter().flatten() {
        bytes = add(bytes, array::<u64>(value.first.capacity())?)?;
        bytes = add(bytes, array::<u64>(value.second.capacity())?)?;
    }
    Ok(bytes)
}

/// Named simultaneous phase totals. These are ordinary diagnostics, not an
/// admission ticket. Recompute a quote after mutating any enclosing owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RequestPeak {
    pub construction_bytes: usize,
    pub execution_bytes: usize,
    pub restoration_bytes: usize,
    pub peak_bytes: usize,
}

/// Borrows the actual compact metadata and physical retained values. Consequently
/// neither can be changed through this handle while its snapshot is in use.
/// Source/transport byte inputs remain the enclosing owners' audited inventories.
pub(super) struct RequestQuote<'q, 'g, 'a> {
    graph: &'q CompactGraph<'g, 'a>,
    owners: RequestOwners<'q>,
    census: CompactOwnerCensus,
    baseline: usize,
    work: WorkBudget,
    descriptors: usize,
    limit: usize,
}

impl<'q, 'g, 'a> RequestQuote<'q, 'g, 'a> {
    pub(super) fn new(
        graph: &'q CompactGraph<'g, 'a>,
        owners: RequestOwners<'q>,
    ) -> Result<Self, Error> {
        Self::with_limit(graph, owners, MAX_PROGRAM_BYTES)
    }

    /// Include all previously frozen regions/selections in the same request.
    /// Do not also put that inventory in owners.other_*; ordinary outstanding
    /// work/metadata can still be supplied there. Register retained body Plans in
    /// prior.record_retained_plan. Shared semantic/source/template backing is
    /// overcounted, not deduplicated or released while it remains live.
    pub(super) fn with_request_work(
        graph: &'q CompactGraph<'g, 'a>,
        mut owners: RequestOwners<'q>,
        prior: &RequestWork,
    ) -> Result<Self, Error> {
        owners.other_work = prior.work.checked_add(owners.other_work)?;
        owners.other_stored_descriptors =
            add(prior.stored_descriptors, owners.other_stored_descriptors)?;
        owners.other_owned_metadata_bytes = add(
            add(
                prior.retained_graph_bytes,
                owners.other_owned_metadata_bytes,
            )?,
            size_of::<RequestWork>(),
        )?;
        Self::new(graph, owners)
    }

    fn with_limit(
        graph: &'q CompactGraph<'g, 'a>,
        owners: RequestOwners<'q>,
        limit: usize,
    ) -> Result<Self, Error> {
        if graph.failed {
            return Err(Error::Retired);
        }
        let census = CompactOwnerCensus::read(graph)?;
        let descriptors = add(
            add(census.stored_descriptors, owners.other_stored_descriptors)?,
            census.body_descriptors.unwrap_or(0),
        )?;
        if descriptors > MAX_NODES {
            return Err(Error::Bounds);
        }
        let work = owners.other_work.checked_add(census.work)?;
        work.limits()?;
        // The actual compact baseline includes its imports Builder, frontier,
        // descriptor capacity, owned prepared full/tail backing and candidate
        // construction/result overlap. It does NOT include the outside fields.
        let mut baseline = add(census.metadata_bytes, outside_bytes(&owners)?)?;
        baseline = add(baseline, census.borrowed_source_bytes)?;
        baseline = add(baseline, census.borrowed_semantic_bytes)?;
        baseline = add(baseline, size_of::<CompactOwnerCensus>())?;
        baseline = add(baseline, size_of::<Result<CompactOwnerCensus, Error>>())?;
        let quote = Self {
            graph,
            owners,
            census,
            baseline,
            work,
            descriptors,
            limit: limit.min(MAX_PROGRAM_BYTES),
        };
        quote.admit_total(baseline)?;
        Ok(quote)
    }

    pub(super) fn baseline_bytes(&self) -> usize {
        self.baseline
    }
    pub(super) fn complete_work(&self) -> WorkBudget {
        self.work
    }
    pub(super) fn stored_descriptors(&self) -> usize {
        self.descriptors
    }
    pub(super) fn owner_census(&self) -> CompactOwnerCensus {
        self.census
    }

    /// Use as PreparedTensorMap's EXISTING admit callback with baseline_bytes()
    /// as caller_baseline. The factory supplies an all-inclusive total; adding
    /// baseline again here would double charge it. Every callback checks prior
    /// peak too. This does not wrap or replace the native constructor.
    pub(super) fn admit_total(&self, total: usize) -> Result<(), Error> {
        if total < self.baseline {
            return Err(Error::Binding);
        }
        if total.max(self.owners.prior_request_peak_bytes) > self.limit {
            Err(Error::Bounds)
        } else {
            Ok(())
        }
    }

    fn construction(&self, extra: usize) -> Result<RequestPeak, Error> {
        let total = add(self.baseline, extra)?;
        self.admit_total(total)?;
        Ok(RequestPeak {
            construction_bytes: total,
            execution_bytes: 0,
            restoration_bytes: 0,
            peak_bytes: total.max(self.owners.prior_request_peak_bytes),
        })
    }

    /// Current region imports are full original Input owners. Discover them
    /// through the emitter's real physical descriptor layouts, without a set,
    /// lane-reference array or virtual-node-indexed slot allocation. Register
    /// order is original import order. Future sliced/external layouts need an
    /// explicit emitter mapping, not a smaller caller-selected input geometry.
    fn validate_body_inputs(&self, inputs: &[Register]) -> Result<(), Error> {
        let mut index = 0;
        for (owner, node) in self.graph.imports.nodes.iter().enumerate() {
            if !matches!(node.code, Code::Input(_)) {
                continue;
            }
            let used = self.graph.descriptor_census().any(|descriptor| {
                descriptor.inputs.iter().flatten().any(|input| {
                    matches!(input.layout, PhysicalLayout::Dense { owner: id } if id == owner)
                })
            });
            if used {
                let input = inputs.get(index).ok_or(Error::Binding)?;
                if input.lanes != node.lanes || input.bounds != node.bounds {
                    return Err(Error::Binding);
                }
                index += 1;
            }
        }
        if index != inputs.len() {
            return Err(Error::Binding);
        }
        Ok(())
    }

    /// Before Builder::new AND the input Layout allocation. LayoutQuote comes
    /// from the actual Layout constructor's quote; it includes retained backing,
    /// destination and span-buffer overlap. Caller arrays/results belong in
    /// caller_control_bytes. No Layout or Builder is allocated by this method.
    pub(super) fn preflight_body_inputs(
        &self,
        inputs: &[Register],
        layouts: &[LayoutQuote],
    ) -> Result<RequestPeak, Error> {
        self.census.body_descriptors.ok_or(Error::Binding)?;
        self.validate_body_inputs(inputs)?;
        let mut extra = layer::Builder::input_container_quote(inputs)?;
        for layout in layouts {
            extra = add(extra, layout.construction_peak_bytes)?;
        }
        extra = add(extra, size_of::<Result<layer::Builder, Error>>())?;
        extra = add(extra, size_of::<Result<Layout, Error>>())?;
        self.construction(extra)
    }

    /// Existing native logical construction quote, BEFORE calling its factory.
    /// Native scratch/return/Arc publication overlap remains unchanged. Full
    /// borrowed caller tables/capacities must already be in the outer baseline.
    pub(super) fn preflight_native(
        &self,
        construction: tensor::Construction,
    ) -> Result<RequestPeak, Error> {
        let native = construction.quote().map_err(|error| match error {
            tensor::ConstructionQuoteError::Geometry(error) => error,
            tensor::ConstructionQuoteError::UncheckedRotaryCoefficientCapacity => Error::Binding,
        })?;
        let mut extra = native.new_peak_bytes;
        for bytes in [
            size_of::<tensor::Construction>(),
            size_of::<tensor::ConstructionQuote>(),
            size_of::<Result<tensor::ConstructionQuote, tensor::ConstructionQuoteError>>(),
        ] {
            extra = add(extra, bytes)?;
        }
        self.construction(extra)
    }

    /// Existing capacity-aware Builder/finish quote, BEFORE Plan::finish. The
    /// quote already includes Builder metadata; add it once, not once per owner.
    /// Earlier staging/native peaks must be retained separately by the caller.
    pub(super) fn preflight_body_finish(
        &self,
        builder: &layer::Builder,
        outputs: &[Layout],
    ) -> Result<RequestPeak, Error> {
        self.census.body_descriptors.ok_or(Error::Binding)?;
        let total = builder.proof_finish_quote(
            outputs,
            add(
                self.baseline,
                size_of::<Result<std::sync::Arc<layer::Plan>, Error>>(),
            )?,
        )?;
        self.admit_total(total)?;
        Ok(RequestPeak {
            construction_bytes: total,
            execution_bytes: 0,
            restoration_bytes: 0,
            peak_bytes: total.max(self.owners.prior_request_peak_bytes),
        })
    }

    /// Compose actual Plan liveness/resources with ALL retained outside owners,
    /// input-copy backing and simultaneous exported restoration copies. The Plan
    /// must replace this graph's work, not a prefix or a regenerated substitute.
    /// Work equality is only an accounting check: original IDs/layout/proof/kernel
    /// mapping and all-role router drain remain the emitter/dispatcher's gates.
    /// This does not authorize executing the current partial CompactGraph.
    pub(super) fn body_phases(
        &self,
        plan: &layer::Plan,
        construction: &[RequestPeak],
    ) -> Result<RequestPeak, Error> {
        if plan.work_budget() != self.graph.work_budget() {
            return Err(Error::Binding);
        }
        if Some(plan.statistics().semantic_sites) != self.census.body_descriptors {
            return Err(Error::Binding);
        }
        self.validate_body_inputs(plan.input_specs())?;
        let mut copies = add(
            size_of::<Vec<Secret>>(),
            array::<Secret>(plan.input_count())?,
        )?;
        for input in plan.input_specs() {
            copies = add(copies, input.lanes.checked_mul(16).ok_or(Error::Bounds)?)?;
        }
        // Existing Plan resources include metadata, physical slots, peak live
        // shares, gathers and original primitive ledgers. No per-region refund
        // and no native lease/template sharing credit against retained graph.
        let execution = add(plan.resources().admitted_bytes, copies)?;
        let mut restoration = execution;
        for output in 0..plan.output_count() {
            restoration = add(
                restoration,
                plan.output_layout(output)?
                    .len()
                    .checked_mul(16)
                    .ok_or(Error::Bounds)?,
            )?;
        }
        for bytes in [
            size_of::<Secret>(),
            size_of::<Result<Secret, Error>>(),
            size_of::<Result<(PartyRuntime, layer::Values), Error>>(),
            size_of::<&layer::Values>(),
            size_of::<Result<Worker, Error>>(),
        ] {
            restoration = add(restoration, bytes)?;
        }
        let execution_bytes = add(self.baseline, execution)?;
        let restoration_bytes = add(self.baseline, restoration)?;
        let mut construction_bytes = self.baseline;
        let mut peak_bytes = self.owners.prior_request_peak_bytes;
        for phase in construction {
            construction_bytes = construction_bytes.max(phase.construction_bytes);
            peak_bytes = peak_bytes.max(phase.peak_bytes);
        }
        peak_bytes = peak_bytes
            .max(construction_bytes)
            .max(execution_bytes)
            .max(restoration_bytes);
        self.admit_total(peak_bytes)?;
        Ok(RequestPeak {
            construction_bytes,
            execution_bytes,
            restoration_bytes,
            peak_bytes,
        })
    }
}

#[cfg(test)]
#[path = "compact_admission_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "compact_request_path_tests.rs"]
mod request_path_tests;
