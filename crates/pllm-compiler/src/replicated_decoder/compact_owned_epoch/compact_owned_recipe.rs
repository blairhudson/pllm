//! Inspection only: loans original owned records to the canonical vocabulary.
//! No Storage, Worker, LogicalComplete, epoch issuer or alternate interpreter.
use super::super::super::compact_attention_softmax::owning::InvocationLoan;
use super::super::recipe::{check_loop, guard_event_node, replay_argmax_block};
use super::*;
use crate::replicated_decoder::program::schedule_finalization::{
    self, BorrowedLaneRefs, BorrowedNode, BorrowedOrderedNodes, BorrowedRecipe,
    GeneratedPublicValues,
};
use crate::replicated_decoder::program::source_entry::compact_pending::sealed::{
    CustodyOwner, FinalizedRootFor, LegacyRoot,
};
use pllm_garble::shared_allocation::ExternalCharge;

enum Coordinates<'s> {
    View(&'s PublishedView, usize),
    Dense(usize, usize),
    Periodic(usize, usize, usize),
    Checked(&'s compact_validity::OwnedCheckedRows, usize),
    Softmax(InvocationLoan<'s>),
    Argmax(
        &'s PublishedView,
        &'s CompactArgmax,
        usize,
        ArgmaxField,
        usize,
        usize,
    ),
}
struct Gather<'s> {
    ordinal: usize,
    lanes: usize,
    bounds: Interval,
    coordinates: Coordinates<'s>,
}
impl BorrowedLaneRefs for Gather<'_> {
    fn len(&self) -> usize {
        self.lanes
    }
    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        if index >= self.lanes {
            return Err(Error::Shape);
        }
        match &self.coordinates {
            Coordinates::View(view, start) => view.word(add_bytes(*start, index)?),
            Coordinates::Dense(owner, start) => Ok((*owner, add_bytes(*start, index)?)),
            Coordinates::Periodic(owner, width, start) if *width > 0 => {
                Ok((*owner, add_bytes(*start, index)? % width))
            }
            Coordinates::Periodic(..) => Err(Error::Shape),
            Coordinates::Checked(rows, start) => rows.word(add_bytes(*start, index)?),
            Coordinates::Softmax(row) => row.input_word(index),
            Coordinates::Argmax(input, plan, round, field, start, parity) => {
                let lane = index
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(*start))
                    .and_then(|n| n.checked_add(*parity))
                    .ok_or(Error::Bounds)?;
                plan.geometry.word(input, plan.first, *round, *field, lane)
            }
        }
    }
}
enum Description<'s> {
    Node(BorrowedNode<'s>),
    Gather(Gather<'s>),
}

/// Pure borrowed traversal. Its references are captured ONLY from a checked
/// loan; there is no independent imports/epoch/Manifest constructor. No OPEN,
/// COMPLETE, Attempt or source authority test is embedded in recipe traversal.
/// A future SEALED entry is exclusively E's genuine finalized-loan integration,
/// not a broadened inspection check or an execution permit supplied here.
pub(super) struct RecipeBody<'s, C: CustodyOwner = LegacyRoot> {
    imports: &'s Builder,
    epoch: &'s OwnedEpoch,
    sources: SourceLenses<'s, C>,
    gathers: Vec<Gather<'s>>,
    work: WorkBudget,
}
#[derive(Clone, Copy)]
struct SourceLens<'s> {
    sources: RecipeSources<'s>,
    iota: Option<GeneratedPublicValues<'s>>,
    bias_bounds: Option<Interval>,
    primary_import: Option<ImportAssociation<'s>>,
    bias_import: Option<ImportAssociation<'s>>,
}
#[derive(Clone, Copy)]
struct ImportAssociation<'s> {
    owner: usize,
    key: &'s str,
    source: &'s SourceTensor,
    lanes: usize,
    bounds: Interval,
}
enum SourceLenses<'s, C: CustodyOwner = LegacyRoot> {
    Inspection(Vec<SourceLens<'s>>),
    Retained(&'s RetainedEpochFor<C>),
}
impl<'s, C: CustodyOwner> SourceLenses<'s, C> {
    fn get(&self, index: usize) -> Result<SourceLens<'s>, Error> {
        match self {
            Self::Inspection(v) => v.get(index).copied().ok_or(Error::Binding),
            Self::Retained(epoch) => RetainedEpochFor::<C>::lens(epoch, index),
        }
    }
}

/// This one accessor is an explicit E dependency, NOT supplied by this child:
/// InspectionLoan::checked_import_source(owner) -> (&str, &SourceTensor).
/// E must resolve the import key in this SAME custody's original Manifest and
/// charge its finite key search. No digest or independently supplied map seals it.
fn bind_import<'s, C: HistoryInspection>(
    loan: &InspectionLoan<'s, C>,
    owner: usize,
    expected: &'s SourceTensor,
) -> Result<ImportAssociation<'s>, Error> {
    let (key, actual) = loan.checked_import_source(owner)?;
    let node = loan.imports().nodes.get(owner).ok_or(Error::Binding)?;
    let Code::PublicInput {
        key: original_key, ..
    } = &node.code
    else {
        return Err(Error::Binding);
    };
    if !std::ptr::eq(original_key.as_str(), key)
        || !std::ptr::eq(actual, expected)
        || actual
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or(Error::Bounds)?
            != node.lanes
    {
        return Err(Error::Binding);
    }
    Ok(ImportAssociation {
        owner,
        key,
        source: actual,
        lanes: node.lanes,
        bounds: node.bounds,
    })
}

#[derive(Default)]
struct Census {
    rows: usize,
    tiles: usize,
    guards: usize,
    invocations: usize,
    argmax_programs: usize,
}
impl Census {
    // Only O(ops) vector lengths/checked arithmetic, prepaid by entry's seed.
    fn lengths(epoch: &OwnedEpoch) -> Result<Self, Error> {
        let mut c = Self::default();
        let (lo, hi) = epoch.invocations.programs().size_hint();
        if hi != Some(lo) {
            return Err(Error::Binding);
        }
        c.invocations = lo;
        for op in &epoch.ops {
            let (rows, tiles, guards, programs) = match &op.kind {
                Kind::Scores(v) => (v.rows.len(), v.tiles.len(), 0, 0),
                Kind::Values(v) => (v.rows.len(), v.tiles.len(), 0, 0),
                Kind::Rotary(v) => (v.native.len(), 0, 0, 0),
                Kind::Scale(v) => (v.native.len(), 0, 0, 0),
                Kind::Rms { checked, .. } => (0, 0, checked.as_ref().map_or(0, |v| v.nodes()), 0),
                Kind::Argmax(v) => (0, 0, 0, v.programs.len()),
                _ => (0, 0, 0, 0),
            };
            c.rows = add_bytes(c.rows, rows)?;
            c.tiles = add_bytes(c.tiles, tiles)?;
            c.guards = add_bytes(c.guards, guards)?;
            c.argmax_programs = add_bytes(c.argmax_programs, programs)?;
        }
        Ok(c)
    }
    fn scans(&self, epoch: &OwnedEpoch, imports: &Builder) -> Result<usize, Error> {
        let mut n = add_bytes(epoch.ops.len(), imports.nodes.len())?;
        for count in [
            self.rows,
            self.tiles,
            self.guards,
            self.invocations,
            self.argmax_programs,
            1,
        ] {
            n = add_bytes(n, count)?;
        }
        Ok(n)
    }
    // Enter ONLY after scans()*128 is prepaid. Count original retained gather
    // references, not an estimate inferred from already-spent work. No Vec/Op
    // cloning, native arithmetic, payload reads or expanded coordinate arrays.
    fn envelope(&self, epoch: &OwnedEpoch, imports: &Builder) -> Result<u64, Error> {
        let mut refs = 0usize;
        for node in &imports.nodes {
            if matches!(&node.code, Code::Gather(_) | Code::SourceGather(_)) {
                refs = add_bytes(refs, node.lanes)?;
            }
        }
        for native in epoch.invocations.programs() {
            refs = add_bytes(refs, native.lanes())?;
        }
        for op in &epoch.ops {
            let count = match &op.kind {
                Kind::Lookup { plan, .. } => op
                    .input
                    .lanes
                    .checked_mul(plan.vocabulary)
                    .ok_or(Error::Bounds)?,
                Kind::Projection { bias, .. } => add_bytes(
                    op.input.lanes,
                    if bias.is_some() {
                        op.output.lanes.checked_mul(2).ok_or(Error::Bounds)?
                    } else {
                        0
                    },
                )?,
                Kind::Scores(v) => {
                    let mut n = 0usize;
                    for row in &v.rows {
                        n = add_bytes(n, row.gather.len())?;
                    }
                    for tile in &v.tiles {
                        n = add_bytes(n, tile.gather.len())?;
                    }
                    n
                }
                Kind::Values(v) => {
                    let mut n = 0usize;
                    for row in &v.rows {
                        n = add_bytes(n, row.gather.len())?;
                    }
                    for tile in &v.tiles {
                        n = add_bytes(n, tile.gather.len())?;
                    }
                    n
                }
                Kind::Scale(v) => v
                    .native
                    .iter()
                    .try_fold(0usize, |n, r| add_bytes(n, r.lanes))?,
                Kind::Rms { .. } => op.input.lanes.checked_mul(4).ok_or(Error::Bounds)?, // rows + 3 guarded gathers
                Kind::Rotary(_) | Kind::Silu => op.input.lanes,
                Kind::Binary { .. } => op.output.lanes.checked_mul(2).ok_or(Error::Bounds)?,
                Kind::Argmax(_) => op.input.lanes.checked_mul(16).ok_or(Error::Bounds)?, // sum of all reducing pair gathers <16*V
                _ => 0,
            };
            refs = add_bytes(refs, count)?;
        }
        let scans = self.scans(epoch, imports)?;
        // 2 descriptor passes + canonical quote, its dependency/source-node
        // reads, validation and coordinate checks: <=8*(N+refs+scans+1)
        // calls. Each includes <=4 full descriptor/row/tile/guard/invocation
        // scans plus 32 fixed record/control visits. This includes EACH lane's
        // linear Invocation lookup, repeated full Guard replay, Previous-map
        // chains, Argmax kernel searches and quote construction/visitation.
        let calls = add_bytes(add_bytes(epoch.next, refs)?, add_bytes(scans, 1)?)?
            .checked_mul(8)
            .ok_or(Error::Bounds)?;
        let per_call = add_bytes(scans.checked_mul(4).ok_or(Error::Bounds)?, 32)?;
        u64::try_from(calls.checked_mul(per_call).ok_or(Error::Bounds)?).map_err(|_| Error::Bounds)
    }
}
impl<'s, C: CustodyOwner> RecipeBody<'s, C> {
    fn imports(&self) -> &'s Builder {
        self.imports
    }
    fn epoch(&self) -> &'s OwnedEpoch {
        self.epoch
    }
    fn sources(&self, index: usize) -> Result<RecipeSources<'s>, Error> {
        Ok(self.sources.get(index)?.sources)
    }
}

/// Inspection-only entry. Stable spans/lenses borrow original owners; no Op or
/// native Arc is copied per read. Payload drops before every covering charge.
pub(in crate::replicated_decoder::program) struct OwnedSchedule<'s, C: CustodyOwner = LegacyRoot> {
    body: RecipeBody<'s, C>,
    loan: InspectionLoan<'s, C>,
    _gathers: ExternalCharge<'s>,
    _sources: ExternalCharge<'s>,
    _controls: ExternalCharge<'s>,
}

impl<'s, C: CustodyOwner> OwnedSchedule<'s, C> {
    pub(in crate::replicated_decoder::program) fn new(
        loan: InspectionLoan<'s, C>,
    ) -> Result<Self, Error> {
        Self::prepare(loan, false)
    }
    fn prepare(loan: InspectionLoan<'s, C>, history_segment: bool) -> Result<Self, Error> {
        // Recheck the checked inspection entry; commit immediately. Structural
        // recipe traversal does not burn/complete the original history. Shared
        // charge admission retains ITS existing owner failure policy unchanged.
        let mut check = loan.attempt()?;
        check.commit();
        drop(check);
        let epoch = loan.epoch();
        let segment_start = if history_segment && epoch.association.epoch() > 0 {
            epoch.read_start(loan.imports())?.0
        } else {
            0
        };
        let seed = add_bytes(add_bytes(epoch.ops.len(), loan.imports().nodes.len())?, 1)?
            .checked_mul(128)
            .ok_or(Error::Bounds)?;
        let mut controls = add_bytes(
            2 * size_of::<Result<Self, Error>>(),
            add_bytes(
                2 * size_of::<Description<'s>>(),
                2 * size_of::<Gather<'s>>(),
            )?,
        )?;
        for bytes in [
            array_bytes::<SourceLens<'s>>(2)?,
            array_bytes::<ImportAssociation<'s>>(2)?,
            array_bytes::<Census>(4)?,
            array_bytes::<Result<Census, Error>>(4)?,
        ] {
            controls = add_bytes(controls, bytes)?;
        }
        let control_charge =
            loan.charge(controls, u64::try_from(seed).map_err(|_| Error::Bounds)?)?;
        let census = Census::lengths(epoch)?;
        let census_visits = census
            .scans(epoch, loan.imports())?
            .checked_mul(128)
            .ok_or(Error::Bounds)?;
        let _census = loan.charge(0, u64::try_from(census_visits).map_err(|_| Error::Bounds)?)?;
        let visits = census.envelope(epoch, loan.imports())?;
        let sources_charge =
            loan.charge(array_bytes::<SourceLens<'s>>(epoch.ops.len())?, visits)?;
        let mut sources = if epoch.ops.is_empty() {
            Vec::new()
        } else {
            reserve(epoch.ops.len())?
        };
        if sources.capacity() != epoch.ops.len() {
            return Err(Error::Bounds);
        }
        for (index, op) in epoch.ops.iter().enumerate() {
            let resolved = loan.sources(index)?;
            let primary_import =
                match &op.kind {
                    Kind::Lookup { weight, .. } | Kind::Projection { weight, .. } => Some(
                        bind_import(&loan, *weight, resolved.primary().ok_or(Error::Binding)?)?,
                    ),
                    _ => None,
                };
            let bias_import = match &op.kind {
                Kind::Projection {
                    bias: Some(bias), ..
                } => Some(bind_import(
                    &loan,
                    bias.owner,
                    resolved.bias().ok_or(Error::Binding)?,
                )?),
                _ => None,
            };
            let bias_bounds = if let Some(bias) = resolved.bias() {
                // Exact borrowed ProjectionBias::bounds contract: file-backed
                // admitted bounds, not inline gamma or a clear payload scan.
                Some(public_bounds(bias)?)
            } else {
                None
            };
            sources.push(SourceLens {
                sources: resolved,
                iota: if matches!(&op.kind, Kind::Argmax(_)) {
                    Some(GeneratedPublicValues::owned_argmax_indices(&loan, index)?)
                } else {
                    None
                },
                bias_bounds,
                primary_import,
                bias_import,
            });
        }
        let work = schedule_work(&loan.imports().nodes)?.checked_add(epoch.work)?;
        let mut body = RecipeBody::<C> {
            imports: loan.imports(),
            epoch,
            sources: SourceLenses::Inspection(sources),
            gathers: Vec::new(),
            work,
        };
        validate(&body)?;
        let mut count = 0usize;
        for ordinal in segment_start..epoch.next {
            if matches!(describe(&body, ordinal)?, Description::Gather(_)) {
                count = add_bytes(count, 1)?;
            }
        }
        let gather_charge = loan.charge(
            array_bytes::<Gather<'s>>(count)?,
            0, // Entire construction/quote envelope was prepaid before sources.
        )?;
        let mut gathers = if count == 0 {
            Vec::new()
        } else {
            reserve(count)?
        };
        if gathers.capacity() != count {
            return Err(Error::Bounds);
        }
        for ordinal in segment_start..epoch.next {
            if let Description::Gather(gather) = describe(&body, ordinal)? {
                gathers.push(gather);
            }
        }
        if gathers.len() != count {
            return Err(Error::Binding);
        }
        body.gathers = gathers;
        let schedule = Self {
            body,
            loan,
            _gathers: gather_charge,
            _sources: sources_charge,
            _controls: control_charge,
        };
        // SAME dependency/work visitor used by the frozen reader and encoder.
        // This quote allocates nothing and grants no finalization capability.
        if !history_segment && schedule_finalization::quote(&schedule, &[], 0)?.work != work {
            return Err(Error::Binding);
        }
        Ok(schedule)
    }

    pub(in crate::replicated_decoder::program) fn work(&self) -> WorkBudget {
        self.body.work
    }
}
impl<C: CustodyOwner> BorrowedOrderedNodes for OwnedSchedule<'_, C> {
    fn node_count(&self) -> usize {
        self.body.node_count()
    }
    fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
        self.body.node(ordinal)
    }
}

/// One source-owned two-epoch traversal. It only dispatches ordinal ranges to
/// the SAME pure RecipeBody and actual acceptance records; no new node engine.
/// Still OPEN inspection, never an executable or a runtime completion entry.
pub(in crate::replicated_decoder::program) struct HistorySchedule<'s, C: CustodyOwner = LegacyRoot> {
    epochs: Vec<OwnedSchedule<'s, C>>,
    custody: &'s C,
    _controls: ExternalCharge<'s>,
}

// Owned per-OP metadata only, not per-node recipes or a reconstructed graph.
// Source handles/native programs remain in the ORIGINAL owned epochs.
struct RetainedLens {
    primary: Option<usize>,
    bias: Option<usize>,
    bias_bounds: Option<Interval>,
    iota: bool,
}
pub(in crate::replicated_decoder::program) struct RetainedEpochFor<C: CustodyOwner> {
    root: Arc<FinalizedRootFor<C>>,
    index: usize,
    sources: Vec<RetainedLens>,
}
impl<C: CustodyOwner> RetainedEpochFor<C> {
    pub(in crate::replicated_decoder::program) fn epoch(&self) -> &OwnedEpoch {
        &self.root.custody().epochs()[self.index]
    }
    fn association<'a>(
        &'a self,
        owner: usize,
        source: &'a SourceTensor,
    ) -> Result<ImportAssociation<'a>, Error> {
        let node = self
            .root
            .imports()?
            .nodes
            .get(owner)
            .ok_or(Error::Binding)?;
        let Code::PublicInput { key, .. } = &node.code else {
            return Err(Error::Binding);
        };
        Ok(ImportAssociation {
            owner,
            key,
            source,
            lanes: node.lanes,
            bounds: node.bounds,
        })
    }
    fn lens(&self, index: usize) -> Result<SourceLens<'_>, Error> {
        let op = self.epoch().ops.get(index).ok_or(Error::Binding)?;
        let metadata = self.sources.get(index).ok_or(Error::Binding)?;
        let mut sources = RecipeSources {
            primary: None,
            norms: None,
            bias: None,
            bias_norms: None,
        };
        match &op.kind {
            Kind::Lookup { source, .. } => sources.primary = Some(self.root.tensor(source)?),
            Kind::Projection {
                source,
                norms,
                bias,
                ..
            } => {
                sources.primary = Some(self.root.tensor(source)?);
                sources.norms = norms.as_ref().map(|n| self.root.norm(n)).transpose()?;
                if let Some(bias) = bias {
                    sources.bias = Some(self.root.tensor(&bias.source)?);
                    sources.bias_norms =
                        bias.norms.as_ref().map(|n| self.root.norm(n)).transpose()?;
                }
            }
            Kind::Rms { gamma, .. } => sources.primary = Some(self.root.tensor(gamma)?),
            _ => {}
        }
        let primary_import = metadata
            .primary
            .map(|owner| self.association(owner, sources.primary.ok_or(Error::Binding)?))
            .transpose()?;
        let bias_import = metadata
            .bias
            .map(|owner| self.association(owner, sources.bias.ok_or(Error::Binding)?))
            .transpose()?;
        let iota = if metadata.iota {
            Some(GeneratedPublicValues::retained_argmax_indices(self, index)?)
        } else {
            None
        };
        Ok(SourceLens {
            sources,
            primary_import,
            bias_import,
            iota,
            bias_bounds: metadata.bias_bounds,
        })
    }
    fn body(&self) -> Result<RecipeBody<'_, C>, Error> {
        Ok(RecipeBody::<C> {
            imports: self.root.imports()?,
            epoch: self.epoch(),
            sources: SourceLenses::Retained(self),
            gathers: Vec::new(),
            work: self.epoch().work,
        })
    }
}
struct RetainedGatherFor<C: CustodyOwner> {
    epoch: Arc<RetainedEpochFor<C>>,
    ordinal: usize,
    lanes: usize,
    bounds: Interval,
}
impl<C: CustodyOwner> BorrowedLaneRefs for RetainedGatherFor<C> {
    fn len(&self) -> usize {
        self.lanes
    }
    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        if index >= self.lanes {
            return Err(Error::Shape);
        }
        let body = self.epoch.body()?;
        match describe(&body, self.ordinal)? {
            Description::Gather(g) if g.lanes == self.lanes && g.bounds == self.bounds => {
                g.lane_ref(index)
            }
            _ => Err(Error::Binding),
        }
    }
}
/// Borrow-free retained metadata. Auto traits follow the concrete whole owner C;
/// this contract grants no Send/Sync authority. Reads use the same describe() and
/// coordinate helpers; stable Gather backing is allocated once during sealing.
pub(in crate::replicated_decoder::program) struct RetainedRecipesFor<C: CustodyOwner> {
    gathers: Vec<RetainedGatherFor<C>>,
    epochs: Vec<Arc<RetainedEpochFor<C>>>,
    root: Arc<FinalizedRootFor<C>>,
}
impl<C: CustodyOwner> BorrowedOrderedNodes for RetainedRecipesFor<C> {
    fn node_count(&self) -> usize {
        self.root.custody().logical_len()
    }
    fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
        if ordinal >= self.epochs[1].epoch().next {
            return self
                .root
                .custody()
                .acceptance_node(ordinal);
        }
        let index = usize::from(ordinal >= self.epochs[0].epoch().next);
        let body = self.epochs[index].body()?;
        match describe(&body, ordinal)? {
            Description::Node(node) => Ok(node),
            Description::Gather(g) => {
                let index = self
                    .gathers
                    .binary_search_by_key(&ordinal, |g| g.ordinal)
                    .map_err(|_| Error::Binding)?;
                let retained = &self.gathers[index];
                if retained.lanes != g.lanes || retained.bounds != g.bounds {
                    return Err(Error::Binding);
                }
                Ok(BorrowedNode {
                    lanes: g.lanes,
                    bounds: g.bounds,
                    recipe: BorrowedRecipe::Gather(retained),
                })
            }
        }
    }
}
impl<'s, C: CustodyOwner> HistorySchedule<'s, C> {
    pub(in crate::replicated_decoder::program) fn retained_bytes(&self) -> Result<usize, Error> {
        // These concrete C-bearing descriptors retain aliases to the one
        // FinalizedRootFor<C>. Root payload/Arc backing is covered by the seal,
        // never billed again for each retained Arc clone.
        let mut bytes = add_bytes(
            2 * size_of::<Result<RetainedRecipesFor<C>, Error>>(),
            array_bytes::<Arc<RetainedEpochFor<C>>>(self.epochs.len())?,
        )?;
        for epoch in &self.epochs {
            let (layout, _) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
                .extend(std::alloc::Layout::new::<RetainedEpochFor<C>>())
                .map_err(|_| Error::Bounds)?;
            bytes = add_bytes(bytes, layout.pad_to_align().size())?;
            bytes = add_bytes(
                bytes,
                array_bytes::<RetainedLens>(epoch.body.epoch.ops.len())?,
            )?;
            bytes = add_bytes(
                bytes,
                array_bytes::<RetainedGatherFor<C>>(epoch.body.gathers.len())?,
            )?;
        }
        Ok(bytes)
    }
    pub(in crate::replicated_decoder::program) fn into_retained(
        self,
        root: Arc<FinalizedRootFor<C>>,
    ) -> Result<RetainedRecipesFor<C>, Error> {
        if !std::ptr::eq(self.custody, root.custody()) {
            return Err(Error::Binding);
        }
        let mut epochs = reserve(self.epochs.len())?;
        let count = self
            .epochs
            .iter()
            .try_fold(0usize, |n, e| add_bytes(n, e.body.gathers.len()))?;
        let mut gathers = reserve(count)?;
        for (index, original) in self.epochs.iter().enumerate() {
            let SourceLenses::Inspection(lenses) = &original.body.sources else {
                return Err(Error::Binding);
            };
            let mut sources = reserve(lenses.len())?;
            for lens in lenses {
                sources.push(RetainedLens {
                    primary: lens.primary_import.map(|a| a.owner),
                    bias: lens.bias_import.map(|a| a.owner),
                    bias_bounds: lens.bias_bounds,
                    iota: lens.iota.is_some(),
                });
            }
            let epoch = Arc::new(RetainedEpochFor::<C> {
                root: Arc::clone(&root),
                index,
                sources,
            });
            for (slot, original) in lenses.iter().enumerate() {
                let resolved = epoch.lens(slot)?;
                for (a, b) in [
                    (original.sources.primary, resolved.sources.primary),
                    (original.sources.bias, resolved.sources.bias),
                ] {
                    match (a, b) {
                        (Some(a), Some(b)) if std::ptr::eq(a, b) => {}
                        (None, None) => {}
                        _ => return Err(Error::Binding),
                    }
                }
                for (a, b) in [
                    (original.primary_import, resolved.primary_import),
                    (original.bias_import, resolved.bias_import),
                ] {
                    match (a, b) {
                        (Some(a), Some(b))
                            if a.owner == b.owner
                                && std::ptr::eq(a.key, b.key)
                                && std::ptr::eq(a.source, b.source) => {}
                        (None, None) => {}
                        _ => return Err(Error::Binding),
                    }
                }
            }
            for gather in &original.body.gathers {
                gathers.push(RetainedGatherFor::<C> {
                    epoch: Arc::clone(&epoch),
                    ordinal: gather.ordinal,
                    lanes: gather.lanes,
                    bounds: gather.bounds,
                });
            }
            epochs.push(epoch);
        }
        Ok(RetainedRecipesFor::<C> {
            gathers,
            epochs,
            root,
        })
    }
    pub(in crate::replicated_decoder::program) fn prepay_finish(&self) -> Result<(), Error> {
        for reader in &self.epochs {
            let epoch = reader.loan.epoch();
            let imports = reader.loan.imports();
            let seed = add_bytes(add_bytes(epoch.ops.len(), imports.nodes.len())?, 1)?
                .checked_mul(128)
                .ok_or(Error::Bounds)?;
            let _seed = reader
                .loan
                .charge(0, u64::try_from(seed).map_err(|_| Error::Bounds)?)?;
            let census = Census::lengths(epoch)?;
            let scans = census
                .scans(epoch, imports)?
                .checked_mul(128)
                .ok_or(Error::Bounds)?;
            let _scan = reader
                .loan
                .charge(0, u64::try_from(scans).map_err(|_| Error::Bounds)?)?;
            let _finish = reader.loan.charge(0, census.envelope(epoch, imports)?)?;
        }
        // Actual three acceptance recipes: two 2-word gathers and Product2.
        let _acceptance = self.epochs[1].loan.charge(0, 128)?;
        Ok(())
    }
    pub(in crate::replicated_decoder::program) fn new(
        custody: &'s C,
    ) -> Result<Self, Error> {
        let first = custody.inspect_history_epoch(0)?;
        let mut attempt = first.attempt()?;
        if custody.epochs().len() != 2 {
            return Err(Error::Binding);
        }
        let bytes = add_bytes(
            array_bytes::<OwnedSchedule<'s, C>>(2)?,
            2 * size_of::<Result<Self, Error>>(),
        )?;
        let controls = first.charge(bytes, 16)?;
        let mut epochs = reserve(2)?;
        if epochs.capacity() != 2 {
            return Err(Error::Bounds);
        }
        epochs.push(OwnedSchedule::<C>::prepare(first, true)?);
        let second = custody.inspect_history_epoch(1)?;
        if second.epoch().read_start(second.imports())?.0 != epochs[0].body.epoch.next {
            return Err(Error::Binding);
        }
        epochs.push(OwnedSchedule::<C>::prepare(second, true)?);
        let result = Self {
            epochs,
            custody,
            _controls: controls,
        };
        if schedule_finalization::quote(&result, &custody.outputs(), 0)?.work
            != custody.work()
        {
            return Err(Error::Binding);
        }
        attempt.commit();
        Ok(result)
    }
}
impl<C: CustodyOwner> BorrowedOrderedNodes for HistorySchedule<'_, C> {
    fn node_count(&self) -> usize {
        self.custody.logical_len()
    }
    fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
        if ordinal < self.epochs[0].body.epoch.next {
            self.epochs[0].node(ordinal)
        } else if ordinal < self.epochs[1].body.epoch.next {
            self.epochs[1].node(ordinal)
        } else {
            self.custody.acceptance_node(ordinal)
        }
    }
}
impl<C: CustodyOwner> BorrowedOrderedNodes for RecipeBody<'_, C> {
    fn node_count(&self) -> usize {
        self.epoch.next
    }
    fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
        let result = match describe(self, ordinal)? {
            Description::Node(node) => node,
            Description::Gather(_) => {
                let index = self
                    .gathers
                    .binary_search_by_key(&ordinal, |g| g.ordinal)
                    .map_err(|_| Error::Binding)?;
                let span = &self.gathers[index];
                BorrowedNode {
                    lanes: span.lanes,
                    bounds: span.bounds,
                    recipe: BorrowedRecipe::Gather(span),
                }
            }
        };
        Ok(result)
    }
}

fn gather(
    ordinal: usize,
    lanes: usize,
    bounds: Interval,
    coordinates: Coordinates<'_>,
) -> Description<'_> {
    Description::Gather(Gather {
        ordinal,
        lanes,
        bounds,
        coordinates,
    })
}
fn prepared_map(epoch: &OwnedEpoch, slot: usize) -> Result<&PreparedTensorMap, Error> {
    let op = epoch.ops.get(slot).ok_or(Error::Binding)?;
    let mut descriptor = slot;
    let map = loop {
        let owner = epoch
            .ops
            .get(descriptor)
            .and_then(|op| op.prepared.as_ref())
            .ok_or(Error::Binding)?;
        match &owner.storage {
            MapStorage::Owned(map) => break map,
            MapStorage::Previous(previous) if *previous < descriptor => descriptor = *previous,
            _ => return Err(Error::Binding),
        }
    };
    map.validate()?;
    if map.lanes() != op.output.lanes {
        return Err(Error::Shape);
    }
    Ok(map)
}
fn prepared(epoch: &OwnedEpoch, slot: usize, input: usize) -> Result<BorrowedRecipe<'_>, Error> {
    let map = prepared_map(epoch, slot)?;
    let full = map.program_at(0)?;
    let remainder = map.lanes() % full.lanes();
    let tail = if remainder == 0 {
        None
    } else {
        Some(map.program_at(map.lanes() - remainder)?)
    };
    check_loop(map.lanes(), full, tail)?;
    Ok(BorrowedRecipe::TensorLoop {
        input: LoopInput::Slice(input),
        program: full,
        tail,
    })
}
fn public_bounds(source: &SourceTensor) -> Result<Interval, Error> {
    Interval::new(source.lower, source.upper)
}

fn describe<'s, C: CustodyOwner>(loan: &RecipeBody<'s, C>, ordinal: usize) -> Result<Description<'s>, Error> {
    let epoch = loan.epoch();
    if let Some(node) = loan.imports().nodes.get(ordinal) {
        return Ok(Description::Node(BorrowedNode::from(node)));
    }
    if let Some(prelude) = &epoch.request_prelude {
        if ordinal >= prelude.first() && ordinal < add_bytes(prelude.first(), 2)? {
            return Ok(Description::Node(prelude.node(ordinal)?));
        }
    }
    if ordinal >= epoch.next {
        return Err(Error::Binding);
    }
    let slot = epoch
        .ops
        .partition_point(|op| op.first <= ordinal)
        .checked_sub(1)
        .ok_or(Error::Binding)?;
    let op = &epoch.ops[slot];
    let offset = ordinal.checked_sub(op.first).ok_or(Error::Binding)?;
    let (lanes, bounds, recipe) = match &op.kind {
        Kind::Values(values) => {
            if let Some(row) = values.rows.iter().find(|r| r.gather_offset == offset) {
                (
                    row.inner,
                    row.input_bounds,
                    BorrowedRecipe::Gather(&row.gather),
                )
            } else {
                let tile = values
                    .tiles
                    .iter()
                    .find(|t| {
                        t.right_offset == offset || t.right_offset.checked_add(1) == Some(offset)
                    })
                    .ok_or(Error::Binding)?;
                if tile.right_offset == offset {
                    (
                        tile.gather.len(),
                        tile.inputs.right,
                        BorrowedRecipe::Gather(&tile.gather),
                    )
                } else {
                    (
                        tile.count,
                        tile.bounds,
                        BorrowedRecipe::Product {
                            left: op.first + tile.left_offset,
                            right: op.first + tile.right_offset,
                            shape: Some(tile.shape),
                            rescale: tile.rescale,
                            norm: tile.proof.as_ref(),
                        },
                    )
                }
            }
        }
        Kind::Softmax => {
            let row = epoch.invocations.row(slot, offset / 2)?;
            if offset % 2 == 0 {
                return Ok(gather(
                    ordinal,
                    row.native_width(),
                    row.input_bounds(),
                    Coordinates::Softmax(row),
                ));
            }
            (
                row.native_width(),
                row.output_bounds(),
                BorrowedRecipe::Tensor {
                    input: row.gather_ordinal(),
                    program: row.program(),
                },
            )
        }
        Kind::Scale(rows) => {
            let chunk = rows.native.get(offset / 2).ok_or(Error::Binding)?;
            if offset % 2 == 0 {
                return Ok(gather(
                    ordinal,
                    chunk.lanes,
                    chunk.incoming,
                    Coordinates::View(&op.input, chunk.start),
                ));
            }
            (
                chunk.lanes,
                chunk.program.output_bounds(),
                BorrowedRecipe::Tensor {
                    input: ordinal - 1,
                    program: &chunk.program,
                },
            )
        }
        Kind::Scores(scores) => {
            if let Some(row) = scores.rows.iter().find(|r| r.node_offset == offset) {
                (
                    scores.width,
                    row.query_bounds,
                    BorrowedRecipe::Gather(&row.gather),
                )
            } else {
                let tile = scores
                    .tiles
                    .iter()
                    .find(|t| {
                        t.right_offset == offset || t.right_offset.checked_add(1) == Some(offset)
                    })
                    .ok_or(Error::Binding)?;
                if tile.right_offset == offset {
                    (
                        tile.gather.len(),
                        tile.inputs.right,
                        BorrowedRecipe::Gather(&tile.gather),
                    )
                } else {
                    (
                        tile.count,
                        tile.bounds,
                        BorrowedRecipe::Product {
                            left: op.first + tile.left_offset,
                            right: op.first + tile.right_offset,
                            shape: Some(tile.shape),
                            rescale: tile.rescale,
                            norm: tile.proof.as_ref(),
                        },
                    )
                }
            }
        }
        Kind::Lookup { weight, plan, .. } => {
            let first = ordinal - offset % 3;
            match offset % 3 {
                0 => {
                    let (owner, index) = op.input.word(offset / 3)?;
                    (
                        plan.vocabulary,
                        plan.selector_bounds,
                        BorrowedRecipe::TensorLoop {
                            input: LoopInput::TokenOffsets { owner, index },
                            program: &plan.full.program,
                            tail: plan.tail.as_ref().map(|p| &p.program),
                        },
                    )
                }
                1 => {
                    return Ok(gather(
                        ordinal,
                        plan.vocabulary,
                        plan.selector_bounds,
                        Coordinates::Dense(first, 0),
                    ))
                }
                _ => (
                    op.output.shape[2],
                    op.output.bounds,
                    BorrowedRecipe::PublicProduct {
                        input: first + 1,
                        weight: *weight,
                        rows: 1,
                        inner: plan.vocabulary,
                        columns: op.output.shape[2],
                        transposed: true,
                    },
                ),
            }
        }
        Kind::Projection {
            weight,
            geometry: [rows, inner, columns],
            raw_bounds,
            bias,
            ..
        } => match offset {
            0 => (
                op.input.lanes,
                op.input.bounds,
                BorrowedRecipe::Gather(&op.input),
            ),
            1 => (
                op.output.lanes,
                *raw_bounds,
                BorrowedRecipe::PublicProduct {
                    input: op.first,
                    weight: *weight,
                    rows: *rows,
                    inner: *inner,
                    columns: *columns,
                    transposed: false,
                },
            ),
            2 => (
                op.output.lanes,
                bias.as_ref().map_or(op.output.bounds, |b| b.floor_bounds),
                prepared(epoch, slot, op.first + 1)?,
            ),
            _ => {
                let bias = bias.as_ref().ok_or(Error::Binding)?;
                let start = 128 * ((offset - 3) / 3);
                let lanes = op
                    .output
                    .lanes
                    .checked_sub(start)
                    .ok_or(Error::Shape)?
                    .min(128);
                let sources = loan.sources(slot)?;
                let source = sources.bias().ok_or(Error::Binding)?;
                match (offset - 3) % 3 {
                    0 => {
                        return Ok(gather(
                            ordinal,
                            lanes,
                            bias.floor_bounds,
                            Coordinates::Dense(op.first + 2, start),
                        ))
                    }
                    1 => {
                        return Ok(gather(
                            ordinal,
                            lanes,
                            loan.sources.get(slot)?.bias_bounds.ok_or(Error::Binding)?,
                            Coordinates::Periodic(bias.owner, source.shape[0], start),
                        ))
                    }
                    _ => (
                        lanes,
                        op.output.bounds,
                        BorrowedRecipe::Affine {
                            left: ordinal - 2,
                            right: Some(ordinal - 1),
                            coefficients: bias.coefficients,
                        },
                    ),
                }
            }
        },
        Kind::Silu => match offset {
            0 => (
                op.input.lanes,
                op.input.bounds,
                BorrowedRecipe::Gather(&op.input),
            ),
            1 => (
                op.output.lanes,
                op.output.bounds,
                prepared(epoch, slot, op.first)?,
            ),
            _ => return Err(Error::Binding),
        },
        Kind::Rotary(rows) => {
            let row = rows.native.get(offset / 2).ok_or(Error::Binding)?;
            if offset % 2 == 0 {
                return Ok(gather(
                    ordinal,
                    rows.width,
                    row.incoming,
                    Coordinates::View(&op.input, rows.width * (offset / 2)),
                ));
            }
            (
                rows.width,
                row.bounds,
                BorrowedRecipe::Tensor {
                    input: ordinal - 1,
                    program: &row.program,
                },
            )
        }
        Kind::Rms {
            program,
            checked,
            row_bounds,
            ..
        } => {
            let first = checked.as_ref().map_or(op.first, |r| r.rows_first());
            if ordinal < first {
                let mut found = None;
                checked
                    .as_ref()
                    .ok_or(Error::Binding)?
                    .visit(&mut |event| {
                        if event.owner == ordinal {
                            found = Some(match event.code {
                                compact_validity::GuardCode::Gather(mapping) => gather(
                                    event.owner,
                                    event.lanes,
                                    event.bounds,
                                    match mapping {
                                        compact_validity::GuardGather::Original { start } => {
                                            Coordinates::View(&op.input, start)
                                        }
                                        compact_validity::GuardGather::Native { owner, start } => {
                                            Coordinates::Dense(owner, start)
                                        }
                                    },
                                ),
                                _ => Description::Node(guard_event_node(event, |_, _| {
                                    Err(Error::Binding)
                                })?),
                            });
                        }
                        Ok(())
                    })?;
                return found.ok_or(Error::Binding);
            }
            let offset = ordinal - first;
            let width = op.output.shape[2];
            if offset % 2 == 0 {
                let start = width * (offset / 2);
                return Ok(gather(
                    ordinal,
                    width,
                    *row_bounds,
                    match checked {
                        Some(rows) if rows.guarded() => Coordinates::Checked(rows, start),
                        _ => Coordinates::View(&op.input, start),
                    },
                ));
            }
            (
                width,
                program.output_bounds(),
                BorrowedRecipe::Tensor {
                    input: ordinal - 1,
                    program,
                },
            )
        }
        Kind::Binary { right, product, .. } => {
            let start = 128 * (offset / 3);
            let lanes = op
                .output
                .lanes
                .checked_sub(start)
                .ok_or(Error::Shape)?
                .min(128);
            match offset % 3 {
                0 => {
                    return Ok(gather(
                        ordinal,
                        lanes,
                        op.input.bounds,
                        Coordinates::View(&op.input, start),
                    ))
                }
                1 => {
                    return Ok(gather(
                        ordinal,
                        lanes,
                        right.bounds,
                        Coordinates::View(right, start),
                    ))
                }
                _ => (
                    lanes,
                    op.output.bounds,
                    if *product {
                        BorrowedRecipe::Product {
                            left: ordinal - 2,
                            right: ordinal - 1,
                            shape: None,
                            rescale: true,
                            norm: None,
                        }
                    } else {
                        BorrowedRecipe::Affine {
                            left: ordinal - 2,
                            right: Some(ordinal - 1),
                            coefficients: affine(1, 1, 0),
                        }
                    },
                ),
            }
        }
        Kind::Argmax(plan) => {
            if offset == 0 {
                return Ok(Description::Node(BorrowedNode {
                    lanes: op.input.lanes,
                    bounds: Interval::new(0, op.input.lanes as i64 - 1)?,
                    recipe: BorrowedRecipe::GeneratedPublicValues(
                        loan.sources.get(slot)?.iota.ok_or(Error::Binding)?,
                    ),
                }));
            }
            let mut found = None;
            replay_argmax_block(plan, &op.input, (offset - 1) / 12, &mut |event| {
                if event.owner != ordinal {
                    return Ok(());
                }
                found = Some(match event.code {
                    ArgmaxCode::Gather(g) => {
                        let layout = g.layout();
                        gather(
                            ordinal,
                            event.lanes,
                            event.bounds,
                            Coordinates::Argmax(
                                &op.input,
                                plan,
                                layout.round,
                                layout.field,
                                layout.start,
                                layout.parity,
                            ),
                        )
                    }
                    ArgmaxCode::PublicIndices => return Err(Error::Binding),
                    ArgmaxCode::Affine {
                        left,
                        right,
                        coefficients,
                    } => Description::Node(BorrowedNode {
                        lanes: event.lanes,
                        bounds: event.bounds,
                        recipe: BorrowedRecipe::Affine {
                            left,
                            right: Some(right),
                            coefficients,
                        },
                    }),
                    ArgmaxCode::Product { left, right } => Description::Node(BorrowedNode {
                        lanes: event.lanes,
                        bounds: event.bounds,
                        recipe: BorrowedRecipe::Product {
                            left,
                            right,
                            shape: None,
                            rescale: false,
                            norm: None,
                        },
                    }),
                    ArgmaxCode::Tensor { input, program } => {
                        let native = plan
                            .programs
                            .iter()
                            .find(|p| std::ptr::eq(&p.program, program))
                            .ok_or(Error::Binding)?;
                        Description::Node(BorrowedNode {
                            lanes: event.lanes,
                            bounds: event.bounds,
                            recipe: BorrowedRecipe::Tensor {
                                input,
                                program: &native.program,
                            },
                        })
                    }
                });
                Ok(())
            })?;
            return found.ok_or(Error::Binding);
        }
        // Zero-node aliases have NO new ordinals.
        Kind::CausalMask
        | Kind::CacheSuffix
        | Kind::KvInitialize { .. }
        | Kind::LastToken { .. }
        | Kind::Feedback
        | Kind::Reshape => return Err(Error::Binding),
    };
    Ok(Description::Node(BorrowedNode {
        lanes,
        bounds,
        recipe,
    }))
}

fn validate<C: CustodyOwner>(loan: &RecipeBody<'_, C>) -> Result<(), Error> {
    let epoch = loan.epoch();
    let imports = loan.imports();
    let (mut next, mut sticky) = epoch.read_start(imports)?;
    if let Some(prelude) = &epoch.request_prelude {
        if prelude.first() != next {
            return Err(Error::Binding);
        }
        next = add_bytes(next, 2)?;
    }
    if epoch.next > MAX_NODES {
        return Err(Error::Bounds);
    }
    for (slot, op) in epoch.ops.iter().enumerate() {
        if op.first != next
            || op.input.lanes == 0
            || op.output.lanes == 0
            || !Arc::ptr_eq(&op.input.origin, &epoch.origin)
            || !Arc::ptr_eq(&op.output.origin, &epoch.origin)
        {
            return Err(Error::Binding);
        }
        let sources = loan.sources(slot)?;
        let count = match &op.kind {
            Kind::Lookup { weight, plan, .. } => {
                let source = sources.primary().ok_or(Error::Binding)?;
                source_import(loan, slot, *weight, source, false)?;
                if source.shape != [plan.vocabulary, op.output.shape[2]]
                    || op.input.lanes.checked_mul(op.output.shape[2]) != Some(op.output.lanes)
                {
                    return Err(Error::Shape);
                }
                check_loop(
                    plan.vocabulary,
                    &plan.full.program,
                    plan.tail.as_ref().map(|p| &p.program),
                )?;
                op.input.lanes.checked_mul(3).ok_or(Error::Bounds)?
            }
            Kind::Projection {
                weight,
                geometry: [rows, inner, columns],
                bias,
                ..
            } => {
                source_import(
                    loan,
                    slot,
                    *weight,
                    sources.primary().ok_or(Error::Binding)?,
                    false,
                )?;
                if rows.checked_mul(*inner) != Some(op.input.lanes)
                    || rows.checked_mul(*columns) != Some(op.output.lanes)
                {
                    return Err(Error::Shape);
                }
                prepared(epoch, slot, op.first + 1)?;
                if let Some(bias) = bias {
                    let source = sources.bias().ok_or(Error::Binding)?;
                    source_import(loan, slot, bias.owner, source, true)?;
                    let original = imports.nodes.get(bias.owner).ok_or(Error::Binding)?;
                    let bounds = loan.sources.get(slot)?.bias_bounds.ok_or(Error::Binding)?;
                    if source.shape != [*columns]
                        || original.lanes != *columns
                        || original.bounds != bounds
                        || bias.floor_bounds != prepared_map(epoch, slot)?.output_bounds()?
                        || bias.coefficients.output_bounds(InputBounds {
                            left: bias.floor_bounds,
                            right: bounds,
                        })? != op.output.bounds
                        || op.output.mapping != Mapping::Products(op.first + 3)
                    {
                        return Err(Error::Binding);
                    }
                    add_bytes(3, ProjectionBias::nodes(op.output.lanes)?)?
                } else {
                    3
                }
            }
            Kind::Silu => {
                prepared(epoch, slot, op.first)?;
                2
            }
            Kind::Rms {
                checked,
                program,
                row_bounds,
                ..
            } => {
                let source = sources.primary().ok_or(Error::Binding)?;
                let [_, rows, width] = op.output.shape;
                if source.shape != [width]
                    || source.public_q18.is_none()
                    || width != program.lanes()
                    || width != program.output_lanes()
                    || rows.checked_mul(width) != Some(op.input.lanes)
                {
                    return Err(Error::Shape);
                }
                let guards = if let Some(checked) = checked {
                    checked.validate_owned(&op.input, op.first, width, sticky)?;
                    if checked.accepted != *row_bounds {
                        return Err(Error::Binding);
                    }
                    sticky = checked.outgoing;
                    checked.nodes()
                } else {
                    if imports.checked_bound.is_some() {
                        return Err(Error::Binding);
                    }
                    0
                };
                if op.output.mapping
                    != (Mapping::Rows {
                        first: op.first + guards,
                        width,
                    })
                {
                    return Err(Error::Binding);
                }
                add_bytes(guards, rows.checked_mul(2).ok_or(Error::Bounds)?)?
            }
            Kind::Rotary(rows) => {
                if rows.width == 0
                    || rows.native.len().checked_mul(rows.width) != Some(op.input.lanes)
                {
                    return Err(Error::Shape);
                }
                for row in &rows.native {
                    if row.program.lanes() != rows.width
                        || row.program.output_lanes() != rows.width
                        || row.bounds != row.program.output_bounds()
                    {
                        return Err(Error::Binding);
                    }
                }
                rows.native.len().checked_mul(2).ok_or(Error::Bounds)?
            }
            Kind::Scale(rows) => rows.virtual_nodes()?,
            Kind::Scores(scores) => {
                if op.output.mapping
                    != (Mapping::AttentionScores {
                        first: op.first,
                        layout: scores.layout()?,
                        products: scores.tiles.len(),
                    })
                {
                    return Err(Error::Binding);
                }
                for tile in &scores.tiles {
                    check_tile(tile)?;
                }
                scores.nodes
            }
            Kind::Values(values) => {
                for row in &values.rows {
                    {
                        // Source-bound native request values must retain their
                        // producing invocation association; None is no fallback.
                        let index = row.probability.as_ref().ok_or(Error::Binding)?;
                        let (producer, number, start, lanes, reversed) = index.coordinates();
                        let native = epoch.invocations.row(producer, number)?;
                        if producer >= slot
                            || reversed
                            || start != 0
                            || lanes != row.inner
                            || native.native_width() != row.inner
                        {
                            return Err(Error::Binding);
                        }
                        for i in 0..row.inner {
                            if epoch.invocations.word(index, i)? != (native.tensor_ordinal(), i)
                                || row.gather.lane_ref(i)? != (native.tensor_ordinal(), i)
                            {
                                return Err(Error::Binding);
                            }
                        }
                        for tile in values
                            .tiles
                            .get(row.tile_start..add_bytes(row.tile_start, row.tile_count)?)
                            .ok_or(Error::Binding)?
                        {
                            let bound =
                                crate::replicated_decoder::program::private_matrix_driver::with_mass(
                                    tile.ordinary_bounds,
                                    tile.inputs.right,
                                    native.mass_bound(),
                                    tile.rescale,
                                )?;
                            if tile.bounds != bound {
                                return Err(Error::Binding);
                            }
                        }
                    }
                }
                for tile in &values.tiles {
                    check_tile(tile)?;
                }
                values.nodes
            }
            Kind::Softmax => {
                let mut rows = 0usize;
                // The output is variable-causal; use genuine invocation keys,
                // never infer a uniform native width from a tensor shape.
                let end = epoch
                    .ops
                    .get(slot + 1)
                    .map_or(epoch.next, |next| next.first);
                if end < op.first || (end - op.first) % 2 != 0 {
                    return Err(Error::Binding);
                }
                while add_bytes(op.first, rows.checked_mul(2).ok_or(Error::Bounds)?)? < end {
                    let row = epoch.invocations.row(slot, rows)?;
                    if row.gather_ordinal() != op.first + 2 * rows
                        || row.tensor_ordinal() != row.gather_ordinal() + 1
                    {
                        return Err(Error::Binding);
                    }
                    rows = add_bytes(rows, 1)?;
                }
                rows.checked_mul(2).ok_or(Error::Bounds)?
            }
            Kind::Binary { right, .. } => {
                if op.input.lanes != op.output.lanes
                    || right.lanes != op.output.lanes
                    || !Arc::ptr_eq(&right.origin, &epoch.origin)
                {
                    return Err(Error::Shape);
                }
                op.output
                    .lanes
                    .div_ceil(128)
                    .checked_mul(3)
                    .ok_or(Error::Bounds)?
            }
            Kind::Argmax(plan) => {
                if plan.first != op.first || plan.geometry.lanes != op.input.lanes {
                    return Err(Error::Binding);
                }
                plan.visit(&op.input, &mut |_| Ok(()))?;
                plan.geometry.nodes()?
            }
            Kind::CausalMask
            | Kind::CacheSuffix
            | Kind::KvInitialize { .. }
            | Kind::LastToken { .. }
            | Kind::Feedback
            | Kind::Reshape => 0,
        };
        next = add_bytes(next, count)?;
    }
    if next != epoch.next || sticky != epoch.sticky {
        return Err(Error::Binding);
    }
    Ok(())
}
fn source_import<C: CustodyOwner>(
    body: &RecipeBody<'_, C>,
    slot: usize,
    owner: usize,
    source: &SourceTensor,
    bias: bool,
) -> Result<(), Error> {
    let lens = body.sources.get(slot)?;
    let association = if bias {
        &lens.bias_import
    } else {
        &lens.primary_import
    }
    .as_ref()
    .ok_or(Error::Binding)?;
    let node = body.imports.nodes.get(owner).ok_or(Error::Binding)?;
    if association.owner != owner
        || !std::ptr::eq(association.source, source)
        || node.lanes != association.lanes
        || node.bounds != association.bounds
        || !matches!(&node.code, Code::PublicInput { key, .. } if std::ptr::eq(key.as_str(), association.key))
    {
        return Err(Error::Binding);
    }
    Ok(())
}
fn check_tile(
    tile: &super::super::super::compact_attention_scores::ScoreTile,
) -> Result<(), Error> {
    if let Some(proof) = tile.proof {
        if proof.inner() != tile.shape.dimensions()[1] || proof.input_bounds() != tile.inputs {
            return Err(Error::Binding);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "compact_owned_recipe_tests.rs"]
mod tests;
