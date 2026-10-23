//! Consuming compiler export/import obligations. NOT executable input authority.
//!
//! The actual frozen producer and original destination creation seals move here.
//! Coordinates are read from those owners, never accepted as an edge issuer.
//! Coordinator custody can be retained alongside this owner, but does not prove
//! that its runtime executed this frozen producer or copied ANY destination.
use super::*;
use crate::replicated_decoder::body_tensor_integration::{CoordinatorDrained, Invocation};
use crate::replicated_decoder::program::original_import::OriginalImportedView;
use crate::replicated_decoder::source_registry::SourceRegistry;
use compact_admission::{RequestOwners, RequestWork};
use pllm_models::{DecoderPlan, StateKind};

/// Ordered source/destination coordinates; not an executable copy capability.
type CopyCoordinates = ((usize, usize), (usize, usize));

/// No SelectedId -> token import coercion. Only the original TokenFeedback op
/// may feed a token declaration. KV/cache/control and general data stay distinct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) enum EdgeKind {
    TokenFeedback,
    Key,
    Value,
    CacheIndices,
    Recurrent,
    Convolution,
}

impl From<StateKind> for EdgeKind {
    fn from(kind: StateKind) -> Self {
        match kind {
            StateKind::Key => Self::Key,
            StateKind::Value => Self::Value,
            StateKind::CacheIndices => Self::CacheIndices,
            StateKind::Recurrent => Self::Recurrent,
            StateKind::Convolution => Self::Convolution,
        }
    }
}

/// Bounded mapping metadata: one entry per original input, not per lane/row.
struct Edge {
    export: usize,
    input: usize,
    kind: EdgeKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) struct PrefixResources {
    pub retained_bytes: usize,
    pub construction_bytes: usize,
    pub peak_bytes: usize,
    pub mapping_capacity: usize,
    pub mapped_lanes: usize,
    pub accumulated_work: WorkBudget,
}

impl PrefixResources {
    fn admit(&self, cap: usize) -> Result<(), Error> {
        if self.peak_bytes > cap.min(MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        self.accumulated_work.limits().map(|_| ())
    }
}

/// Single consuming batch. No Clone, export splitting, detached row certificate,
/// copied-share extraction or executable bind_import. All sibling mappings and
/// destination seals burn together on error/cancel/unwind. Existing source row
/// ledgers stay in the original frozen recipes; no attention/general coercion.
pub(in crate::replicated_decoder) struct PendingPrefix<'a> {
    // Drop original destination seals/mappings before releasing producer custody.
    inputs: Vec<OriginalImportedView<'a>>,
    edges: Vec<Edge>,
    destination: &'a Builder,
    source: FrozenRegion<'a>,
    quote: PrefixResources,
    next_epoch: usize,
    #[cfg(test)]
    runtime_attempted: std::cell::Cell<bool>,
}

/// Borrow of the ORIGINAL destination creation seal, not a coordinate key.
#[cfg(test)]
pub(in crate::replicated_decoder) struct DestinationKey<'p, 'a> {
    pending: &'p PendingPrefix<'a>,
    edge: usize,
}

#[cfg(test)]
impl DestinationKey<'_, '_> {
    pub(in crate::replicated_decoder) fn matches(
        &self,
        pending: &PendingPrefix<'_>,
        edge: usize,
    ) -> bool {
        let Some(a) = self.pending.edges.get(self.edge) else {
            return false;
        };
        let Some(b) = pending.edges.get(edge) else {
            return false;
        };
        self.pending.inputs[a.input].same_declaration(&pending.inputs[b.input])
    }
}

/// This ordered guard owns an EXISTING consuming coordinator capability. It does NOT
/// certify that capability's invocation corresponds to PendingPrefix.source.
/// No import/acceptance API accepts this guard. E must join actual owned frozen
/// execution + physical copies, not bless this compiler recipe after drain.
pub(in crate::replicated_decoder) struct CoordinatorHeld<'a> {
    // Cancel/erase actual runtime shares BEFORE releasing frozen native/source
    // owners. No field extraction/restore escape leaves sibling mappings live.
    drained: CoordinatorDrained<Invocation>,
    pending: PendingPrefix<'a>,
}

impl CoordinatorHeld<'_> {
    pub(in crate::replicated_decoder) fn cancel(self) {
        drop(self);
    }
}

fn original_source<'s, 'a>(
    source: &'s FrozenRegion<'a>,
) -> Result<&'s OriginalImportedView<'a>, Error> {
    source
        ._prebound_inputs
        .as_ref()
        .ok_or(Error::Binding)?
        .original_bindings()
        .first()
        .ok_or(Error::Binding)
}

fn relation(
    source: &FrozenRegion<'_>,
    input: &OriginalImportedView<'_>,
) -> Result<(usize, EdgeKind), Error> {
    let original = original_source(source)?;
    // Same ACTUAL original plan and registry, not a same-digest plan/record set.
    // Growing-prefill plans need the request owner's genuine plan-to-plan edge;
    // this endpoint does not infer one from config_digest or a numeric epoch.
    if !std::ptr::eq(original.plan(), input.plan())
        || !std::ptr::eq(original.registry(), input.registry())
        || std::ptr::eq(original.graph(), input.graph())
        || !std::ptr::eq(input.graph(), &input.plan().decode)
    {
        return Err(Error::Binding);
    }
    let graph = original.graph();
    let (name, kind) = if input.semantic() == "input.tokens" {
        if !original.plan().token_feedback {
            return Err(Error::Binding);
        }
        let op = graph
            .operations
            .iter()
            .find(|op| op.id == graph.output)
            .ok_or(Error::Binding)?;
        if op.operator != Op::TokenFeedback {
            return Err(Error::Binding);
        }
        (op.id.as_str(), EdgeKind::TokenFeedback)
    } else {
        let state = input
            .graph()
            .state_inputs
            .iter()
            .find(|s| s.id == input.semantic())
            .ok_or(Error::Binding)?;
        let mut outputs = graph
            .state_outputs
            .iter()
            .filter(|s| s.layer == state.layer && s.kind == state.kind);
        let output = outputs.next().ok_or(Error::Binding)?;
        if outputs.next().is_some()
            || output.shape != state.shape
            || output.maximum_sequence != state.maximum_sequence
        {
            return Err(Error::Binding);
        }
        let op = graph
            .operations
            .iter()
            .find(|op| op.id == output.id)
            .ok_or(Error::Binding)?;
        if op.state_kind != Some(state.kind)
            || op.layer != state.layer
            || op.output_shape != output.shape
        {
            return Err(Error::Binding);
        }
        (output.id.as_str(), state.kind.into())
    };
    let slot = source
        .recipe_names
        .iter()
        .position(|candidate| candidate == name)
        .ok_or(Error::Binding)?;
    let view = &source.ops.get(slot).ok_or(Error::Binding)?.output;
    source.resolve(view)?;
    // Find the original publication in the OWNED export list. Coordinate/Word
    // comparison never establishes this association.
    let export = source
        .exports
        .iter()
        .position(|candidate| same_view(candidate, view))
        .ok_or(Error::Binding)?;
    if kind == EdgeKind::TokenFeedback {
        let feedback = source
            .ops
            .get(view.slot.ok_or(Error::Binding)?)
            .ok_or(Error::Binding)?;
        if !matches!(feedback.kind, PublishedKind::TokenFeedback) {
            return Err(Error::Binding);
        }
        let selected = source
            .ops
            .get(feedback.input.slot.ok_or(Error::Binding)?)
            .ok_or(Error::Binding)?;
        let PublishedKind::Argmax(selection) = &selected.kind else {
            return Err(Error::Binding);
        };
        let vocabulary = input
            .graph()
            .operations
            .iter()
            .find(|op| op.operator == Op::OutputHead)
            .and_then(|op| op.output_shape.last())
            .copied()
            .ok_or(Error::Binding)?;
        if u64::try_from(selection.geometry.lanes).map_err(|_| Error::Bounds)? != vocabulary {
            return Err(Error::Binding);
        }
        // Original TokenFeedback clones selected's rank-one View, while its
        // semantic declaration and next token input are [batch, sequence]. This
        // is the ORIGINAL one-word zero-storage alias, never arithmetic or a
        // general row-width change. Do not infer growing token history here.
        let declared = graph
            .operations
            .iter()
            .find(|op| op.id == name)
            .ok_or(Error::Binding)?;
        if ViewShape::semantic(&declared.output_shape)? != input.logical()
            || view.lanes != 1
            || input.logical().lanes()? != 1
            || view.logical.row_width() != input.logical().row_width()
        {
            return Err(Error::Shape);
        }
    } else if view.logical != input.logical() {
        return Err(Error::Shape);
    }
    Ok((export, kind))
}

#[allow(clippy::too_many_arguments)]
fn quote(
    source: &FrozenRegion<'_>,
    destination: &Builder,
    inputs: &Vec<OriginalImportedView<'_>>,
    registry: &SourceRegistry<'_>,
    plan: &DecoderPlan,
    graph: &DecoderGraph,
    prior: &RequestWork,
    owners: &RequestOwners<'_>,
    cap: usize,
) -> Result<PrefixResources, Error> {
    source
        .validity
        .check_frozen_owner(source.imports_owner, &source.origin)?;
    let original = original_source(source)?;
    original.check(
        source.imports_owner,
        original.registry(),
        original.plan(),
        original.graph(),
    )?;
    if inputs.is_empty()
        || inputs.len() > MAX_NODES
        || std::ptr::eq(source.imports_owner, destination)
    {
        return Err(Error::Binding);
    }
    let mut lanes = 0;
    for (index, input) in inputs.iter().enumerate() {
        input.check(destination, registry, plan, graph)?;
        if !std::ptr::eq(input.registry(), registry) {
            return Err(Error::Binding);
        }
        if inputs[..index]
            .iter()
            .any(|prior| prior.same_declaration(input) || prior.semantic() == input.semantic())
        {
            return Err(Error::Binding);
        }
        relation(source, input)?;
        lanes = add_bytes(lanes, input.logical().lanes()?)?;
    }
    // Destination imports cannot be omitted from this all-or-nothing batch.
    // No caller-selected subset obtains sibling executable import authority.
    for (owner, node) in destination.nodes.iter().enumerate() {
        if matches!(node.code, Code::Input(_)) && !inputs.iter().any(|input| input.owner() == owner)
        {
            return Err(Error::Binding);
        }
    }
    let (outside, work, _) = compact_admission::input_construction_baseline(prior, owners)?;
    let destination_work = destination
        .nodes
        .iter()
        .try_fold(WorkBudget::default(), |work, node| {
            work.checked_add(node.work_budget()?)
        })?;
    let accumulated_work = work
        .checked_add(source.work)?
        .checked_add(destination_work)?;
    accumulated_work.limits()?;
    // Replace the old OUTSIDE inventory with the current enclosing inventory.
    // The excluded quantity comes from source's original checked owner ledger,
    // not caller numeric credit. Keep actual binding/plan/source backing intact.
    let old_external = source
        ._prebound_inputs
        .as_ref()
        .ok_or(Error::Binding)?
        .external_bytes;
    let own_source = source
        .resources
        .retained_bytes
        .checked_sub(old_external)
        .ok_or(Error::Binding)?;
    let mut retained = add_bytes(outside, own_source)?;
    for bytes in [
        CompactGraph::imports_bytes(destination)?,
        array_bytes::<OriginalImportedView<'_>>(inputs.capacity())?,
        array_bytes::<Edge>(inputs.len())?,
        size_of::<PendingPrefix<'_>>(),
    ] {
        retained = add_bytes(retained, bytes)?;
    }
    // Complete plan/registry/source backing already belongs to source's sealed
    // retained inventory. Input checks above require those exact SAME owners.
    let construction = [
        size_of::<Vec<OriginalImportedView<'_>>>(),
        2 * size_of::<Result<PendingPrefix<'_>, Error>>(),
        2 * size_of::<PrefixResources>(),
        size_of::<Result<PrefixResources, Error>>(),
        size_of::<Result<(usize, EdgeKind), Error>>(),
        size_of::<Edge>(),
    ]
    .into_iter()
    .try_fold(retained, add_bytes)?;
    let result = PrefixResources {
        retained_bytes: retained,
        construction_bytes: construction,
        peak_bytes: construction
            .max(source.resources.peak_bytes)
            .max(owners.prior_request_peak_bytes),
        mapping_capacity: inputs.len(),
        mapped_lanes: lanes,
        accumulated_work,
    };
    result.admit(cap)?;
    Ok(result)
}

impl<'a> FrozenRegion<'a> {
    /// Current prior/owners inventory excludes this consumed source, its actual
    /// plan/registry and the destination Builder/input-seal vector. Those owners
    /// are read above, not accepted as a caller's numeric source baseline.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder::program) fn prefix_quote(
        &self,
        destination: &Builder,
        inputs: &Vec<OriginalImportedView<'_>>,
        registry: &SourceRegistry<'_>,
        plan: &DecoderPlan,
        graph: &DecoderGraph,
        prior: &RequestWork,
        owners: &RequestOwners<'_>,
        cap: usize,
    ) -> Result<PrefixResources, Error> {
        quote(
            self,
            destination,
            inputs,
            registry,
            plan,
            graph,
            prior,
            owners,
            cap,
        )
    }

    /// Consumes BOTH original producer and destination creation seals, including
    /// on validation/allocation failure. Never reclassifies this as completed copy.
    /// This implements same-original-plan prefill->decode endpoint binding only.
    /// Fresh growing-prefill plans require D's existing request owner to seal
    /// their plan-to-plan relationship; matching config/hash/epoch is insufficient.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder::program) fn into_prefix(
        self,
        destination: &'a Builder,
        inputs: Vec<OriginalImportedView<'a>>,
        registry: &SourceRegistry<'a>,
        plan: &DecoderPlan,
        graph: &DecoderGraph,
        prior: &RequestWork,
        owners: &RequestOwners<'_>,
        cap: usize,
    ) -> Result<PendingPrefix<'a>, Error> {
        let quote = quote(
            &self,
            destination,
            &inputs,
            registry,
            plan,
            graph,
            prior,
            owners,
            cap,
        )?;
        let next_epoch = self.validity.epoch().checked_add(1).ok_or(Error::Bounds)?;
        #[cfg(test)]
        MAPPING_FACTORIES.with(|n| n.set(n.get() + 1));
        let mut edges = Vec::new();
        edges
            .try_reserve_exact(inputs.len())
            .map_err(|_| Error::Allocation)?;
        if edges.capacity() != quote.mapping_capacity {
            return Err(Error::Allocation);
        }
        for (index, input) in inputs.iter().enumerate() {
            let (export, kind) = relation(&self, input)?;
            edges.push(Edge {
                export,
                input: index,
                kind,
            });
        }
        Ok(PendingPrefix {
            inputs,
            edges,
            destination,
            source: self,
            quote,
            next_epoch,
            #[cfg(test)]
            runtime_attempted: std::cell::Cell::new(false),
        })
    }
}

impl<'a> PendingPrefix<'a> {
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn begin_runtime_attempt(&self) -> Result<(), Error> {
        if self.runtime_attempted.replace(true) {
            Err(Error::Retired)
        } else {
            Ok(())
        }
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn destination_key(
        &self,
        edge: usize,
    ) -> Result<DestinationKey<'_, 'a>, Error> {
        self.edges.get(edge).ok_or(Error::Binding)?;
        Ok(DestinationKey {
            pending: self,
            edge,
        })
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn destination_count(&self) -> usize {
        self.edges.len()
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn destination_lanes(
        &self,
        edge: usize,
    ) -> Result<usize, Error> {
        self.inputs[self.edges.get(edge).ok_or(Error::Binding)?.input]
            .logical()
            .lanes()
    }
    pub(in crate::replicated_decoder) fn resources(&self) -> PrefixResources {
        self.quote
    }
    pub(in crate::replicated_decoder) fn source(&self) -> &FrozenRegion<'a> {
        &self.source
    }
    pub(in crate::replicated_decoder) fn epoch(&self) -> usize {
        self.source.validity.epoch()
    }
    /// Proposed epoch only: no runtime-copy/continuation claim.
    pub(in crate::replicated_decoder) fn next_epoch(&self) -> usize {
        self.next_epoch
    }
    pub(in crate::replicated_decoder) fn sticky_source(&self) -> Option<usize> {
        self.source.validity.current()
    }
    pub(in crate::replicated_decoder) fn kind(&self, edge: usize) -> Result<EdgeKind, Error> {
        Ok(self.edges.get(edge).ok_or(Error::Binding)?.kind)
    }
    pub(in crate::replicated_decoder::program) fn check_destination(
        &self,
        candidate: &Builder,
        input: &OriginalImportedView<'_>,
        edge: usize,
    ) -> Result<(), Error> {
        let edge = self.edges.get(edge).ok_or(Error::Binding)?;
        if !std::ptr::eq(candidate, self.destination)
            || !self.inputs[edge.input].same_declaration(input)
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
    /// Ordered read-only copy RECIPE. This returns no Secret, row certificate or
    /// import capability. Native/source owners remain in self throughout replay.
    pub(in crate::replicated_decoder) fn mapping(
        &self,
        edge: usize,
        lane: usize,
    ) -> Result<CopyCoordinates, Error> {
        let edge = self.edges.get(edge).ok_or(Error::Binding)?;
        let input = &self.inputs[edge.input];
        if lane >= input.logical().lanes()? {
            return Err(Error::Shape);
        }
        let source = self.source.word(&self.source.exports[edge.export], lane)?;
        Ok((source, (input.owner(), lane)))
    }
    pub(in crate::replicated_decoder) fn cancel(self) {
        drop(self);
    }

    /// Minimal E endpoint: retain the EXISTING consuming capability unchanged.
    /// This is custody ONLY, not source-owner qualification, completed copy or an
    /// issuer for imports. In particular no generic map(T)->U recertifies drain.
    /// Its actual invocation/runtime must stay live until pending mappings erase.
    pub(in crate::replicated_decoder) fn hold_for_coordinator(
        self,
        drained: CoordinatorDrained<Invocation>,
        cap: usize,
    ) -> Result<CoordinatorHeld<'a>, Error> {
        // All fallible work precedes the owning transition. On refusal the
        // consumed drained argument cancels before self's frozen owners erase.
        let (invocation_bytes, invocation_work) = drained.prefix_custody_inventory()?;
        self.quote
            .accumulated_work
            .checked_add(invocation_work)?
            .limits()?;
        let controls = [
            size_of::<CoordinatorHeld<'_>>(),
            2 * size_of::<Result<CoordinatorHeld<'_>, Error>>(),
            size_of::<CoordinatorDrained<Invocation>>(),
        ]
        .into_iter()
        .try_fold(
            add_bytes(self.quote.retained_bytes, invocation_bytes)?,
            add_bytes,
        )?;
        self.quote.admit(cap)?;
        if controls > cap.min(MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        Ok(CoordinatorHeld {
            drained,
            pending: self,
        })
    }
}

#[cfg(test)]
thread_local! { static MAPPING_FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
#[path = "compact_prefix_custody_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "frozen_prefix_handoff_fixture.rs"]
pub(in crate::replicated_decoder) mod execution_fixture;
