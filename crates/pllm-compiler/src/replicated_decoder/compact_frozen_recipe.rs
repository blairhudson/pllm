//! Frozen original owners lent to the shared canonical schedule vocabulary.
//!
//! Only gather *spans* need stable control backing: no Node, Words, expanded
//! coordinate array, native program, source record or Arc is copied. Whole-view
//! gathers borrow the publication directly. The original import prefix is lent
//! unchanged, including source IDs; zero-node aliases never get new ordinals.
//! Argmax borrows a sealed compiler-derived public iota recipe and its original
//! numeric/event generators. This is not an execution/completion capability.
use super::super::super::schedule_finalization::{
    BorrowedLaneRefs, BorrowedNode, BorrowedOrderedNodes, BorrowedRecipe, GeneratedPublicValues,
};
use super::*;

impl BorrowedLaneRefs for PublishedView {
    fn len(&self) -> usize {
        self.lanes
    }

    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        self.word(index)
    }
}

enum Coordinates<'r> {
    View {
        view: &'r PublishedView,
        start: usize,
    },
    Dense(usize),
    DenseOffset {
        owner: usize,
        start: usize,
    },
    Periodic {
        owner: usize,
        width: usize,
        start: usize,
    },
    SafeRows {
        rows: &'r compact_validity::CheckedRows<'r>,
        start: usize,
    },
    Argmax {
        input: &'r PublishedView,
        plan: &'r CompactArgmax,
        round: usize,
        field: ArgmaxField,
        start: usize,
        parity: usize,
    },
}

/// Stable borrowed slice metadata, not an expanded node/recipe table. Its size
/// is independent of the number of lanes in the gather.
struct GatherSpan<'r> {
    ordinal: usize,
    coordinates: Coordinates<'r>,
    lanes: usize,
}

impl BorrowedLaneRefs for GatherSpan<'_> {
    fn len(&self) -> usize {
        self.lanes
    }

    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        if index >= self.lanes {
            return Err(Error::Shape);
        }
        match self.coordinates {
            Coordinates::View { view, start } => {
                view.word(start.checked_add(index).ok_or(Error::Bounds)?)
            }
            Coordinates::Dense(owner) => Ok((owner, index)),
            Coordinates::DenseOffset { owner, start } => {
                Ok((owner, start.checked_add(index).ok_or(Error::Bounds)?))
            }
            Coordinates::Periodic {
                owner,
                width,
                start,
            } if width != 0 => Ok((
                owner,
                start.checked_add(index).ok_or(Error::Bounds)? % width,
            )),
            Coordinates::Periodic { .. } => Err(Error::Shape),
            Coordinates::SafeRows { rows, start } => {
                rows.word(start.checked_add(index).ok_or(Error::Bounds)?)
            }
            Coordinates::Argmax {
                input,
                plan,
                round,
                field,
                start,
                parity,
            } => {
                let lane = index
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(start))
                    .and_then(|n| n.checked_add(parity))
                    .ok_or(Error::Bounds)?;
                plan.geometry.word(input, plan.first, round, field, lane)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) struct RecipeResources {
    pub gather_spans: usize,
    pub gather_bytes: usize,
    pub retained_bytes: usize,
    pub transition_bytes: usize,
    pub peak_bytes: usize,
    pub replay_bytes: usize,
    /// Compiler-owned diagnostic ledger, never caller-supplied work authority.
    /// Shared finalization independently recomputes every emitted primitive.
    pub work: WorkBudget,
}

impl RecipeResources {
    /// Add indexed original-sink replay's named transient slots when lending to
    /// shared finalization. They are not retained heap backing or native kernels.
    pub(in crate::replicated_decoder) fn finalization_baseline_bytes(self) -> Result<usize, Error> {
        add_bytes(self.retained_bytes, self.replay_bytes)
    }

    pub(in crate::replicated_decoder) fn admit(self, cap: usize) -> Result<(), Error> {
        if self.peak_bytes > cap.min(MAX_PROGRAM_BYTES) {
            Err(Error::Bounds)
        } else {
            Ok(())
        }
    }
}

/// Borrows frozen owner, its original native Arc handles, source/import records
/// and provenance. No semantic graph borrow survives. Caller must separately
/// inventory any still-live semantic graph or other subsequent-phase owners.
pub(in crate::replicated_decoder) struct FrozenSchedule<'r, 'a> {
    owner: &'r FrozenRegion<'a>,
    gathers: Vec<GatherSpan<'r>>,
    resources: RecipeResources,
}

impl<'a> FrozenRegion<'a> {
    /// Quote only this new control owner/transition, above frozen retained bytes.
    /// Earlier frozen construction/transition peaks remain charged, not credited.
    pub(in crate::replicated_decoder) fn schedule_recipe_quote(
        &self,
    ) -> Result<RecipeResources, Error> {
        self.validate_recipe_geometry()?;
        let work = schedule_work(&self.imports_owner.nodes)?.checked_add(self.work)?;
        work.limits()?;
        let mut gather_spans = 0;
        self.visit_gather_spans(&mut |_| {
            gather_spans = add_bytes(gather_spans, 1)?;
            Ok(())
        })?;
        let gather_bytes = array_bytes::<GatherSpan<'_>>(gather_spans)?;
        let retained_bytes = add_bytes(
            self.resources.retained_bytes,
            add_bytes(size_of::<FrozenSchedule<'_, 'a>>(), gather_bytes)?,
        )?;
        let mut replay_bytes = if self
            .ops
            .iter()
            .any(|op| matches!(op.kind, PublishedKind::Argmax(_)))
        {
            let mut bytes = 0;
            for slot in [
                size_of::<ArgmaxBlockReplay<'_, '_>>(),
                size_of::<ArgmaxEvent<'_>>(),
                4 * size_of::<Value>(),
                2 * size_of::<BorrowedNode<'_>>(),
                size_of::<Result<BorrowedNode<'_>, Error>>(),
                size_of::<GeneratedPublicValues<'_>>(),
            ] {
                bytes = add_bytes(bytes, slot)?;
            }
            bytes
        } else {
            0
        };
        if self.ops.iter().any(|op| {
            matches!(
                op.kind,
                PublishedKind::RmsNorm {
                    checked: Some(_),
                    ..
                }
            )
        }) {
            let mut checked = 0;
            for slot in [
                size_of::<compact_validity::GuardEvent<'_>>(),
                size_of::<compact_validity::GuardGather>(),
                size_of::<Option<BorrowedNode<'_>>>(),
                2 * size_of::<Result<BorrowedNode<'_>, Error>>(),
                size_of::<Result<(), Error>>(),
                size_of::<Option<usize>>(),
            ] {
                checked = add_bytes(checked, slot)?;
            }
            // A canonical node is either argmax replay or checked-row replay.
            replay_bytes = replay_bytes.max(checked);
        }
        let mut transition_bytes = add_bytes(retained_bytes, replay_bytes)?;
        for bytes in [
            size_of::<Result<FrozenSchedule<'_, 'a>, Error>>(),
            2 * size_of::<RecipeResources>(),
            size_of::<Result<RecipeResources, Error>>(),
            size_of::<Result<Vec<GatherSpan<'_>>, Error>>(),
            size_of::<Vec<GatherSpan<'_>>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<Result<(), Error>>(),
            2 * size_of::<GatherSpan<'_>>(),
            size_of::<BorrowedNode<'_>>(),
            size_of::<Result<BorrowedNode<'_>, Error>>(),
            size_of::<tensor::Resources>(),
            size_of::<Result<tensor::Resources, Error>>(),
            size_of::<Result<WorkBudget, Error>>(),
        ] {
            transition_bytes = add_bytes(transition_bytes, bytes)?;
        }
        Ok(RecipeResources {
            gather_spans,
            gather_bytes,
            retained_bytes,
            transition_bytes,
            peak_bytes: transition_bytes.max(self.resources.peak_bytes),
            replay_bytes,
            work,
        })
    }

    /// Recompute and admit BEFORE reserving even control metadata. Failure does
    /// not mutate frozen owner; no kernel factory or Arc clone is involved.
    pub(in crate::replicated_decoder) fn borrow_schedule(
        &self,
        cap: usize,
    ) -> Result<FrozenSchedule<'_, 'a>, Error> {
        let resources = self.schedule_recipe_quote()?;
        resources.admit(cap)?;
        let mut gathers = if resources.gather_spans == 0 {
            Vec::new()
        } else {
            reserve(resources.gather_spans)?
        };
        self.visit_gather_spans(&mut |span| {
            gathers.push(span);
            Ok(())
        })?;
        Ok(FrozenSchedule {
            owner: self,
            gathers,
            resources,
        })
    }

    fn visit_gather_spans<'r>(
        &'r self,
        visit: &mut impl FnMut(GatherSpan<'r>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        for op in &self.ops {
            match &op.kind {
                PublishedKind::AttentionScale(rows) => {
                    for (index, chunk) in rows.native.iter().enumerate() {
                        visit(GatherSpan {
                            ordinal: op.first + 2 * index,
                            lanes: chunk.lanes,
                            coordinates: Coordinates::View {
                                view: &op.input,
                                start: chunk.start,
                            },
                        })?;
                    }
                }
                PublishedKind::TokenLookup { plan, .. } => {
                    for token in 0..op.input.lanes {
                        let first = op.first + 3 * token;
                        visit(GatherSpan {
                            ordinal: first + 1,
                            coordinates: Coordinates::Dense(first),
                            lanes: plan.vocabulary,
                        })?;
                    }
                }
                PublishedKind::Projection {
                    bias: Some(bias), ..
                } => {
                    for start in (0..op.output.lanes).step_by(128) {
                        let first = op.first + 3 + 3 * (start / 128);
                        let lanes = (op.output.lanes - start).min(128);
                        visit(GatherSpan {
                            ordinal: first,
                            lanes,
                            coordinates: Coordinates::DenseOffset {
                                owner: op.first + 2,
                                start,
                            },
                        })?;
                        visit(GatherSpan {
                            ordinal: first + 1,
                            lanes,
                            coordinates: Coordinates::Periodic {
                                owner: bias.owner,
                                width: bias.source.shape[0],
                                start,
                            },
                        })?;
                    }
                }
                PublishedKind::Rotary(rows) => {
                    for row in 0..rows.native.len() {
                        if row == 0 && rows.width == op.input.lanes {
                            continue;
                        }
                        visit(GatherSpan {
                            ordinal: op.first + 2 * row,
                            coordinates: Coordinates::View {
                                view: &op.input,
                                start: row * rows.width,
                            },
                            lanes: rows.width,
                        })?;
                    }
                }
                PublishedKind::RmsNorm { checked, .. } => {
                    let width = op.output.shape[2];
                    if let Some(rows) = checked {
                        rows.visit(&mut |event| {
                            use compact_validity::{GuardCode, GuardGather};
                            let GuardCode::Gather(gather) = event.code else {
                                return Ok(());
                            };
                            let coordinates = match gather {
                                GuardGather::Original { start } => {
                                    if start == 0 && event.lanes == op.input.lanes {
                                        return Ok(());
                                    }
                                    Coordinates::View {
                                        view: &op.input,
                                        start,
                                    }
                                }
                                GuardGather::Native { owner, start } => {
                                    Coordinates::DenseOffset { owner, start }
                                }
                            };
                            visit(GatherSpan {
                                ordinal: event.owner,
                                lanes: event.lanes,
                                coordinates,
                            })
                        })?;
                    }
                    let guarded = checked.as_ref().is_some_and(|rows| rows.guarded());
                    let first = checked.as_ref().map_or(op.first, |rows| rows.rows_first());
                    for row in 0..op.output.shape[1] {
                        if !guarded && row == 0 && width == op.input.lanes {
                            continue;
                        }
                        visit(GatherSpan {
                            ordinal: first + 2 * row,
                            coordinates: if guarded {
                                Coordinates::SafeRows {
                                    rows: checked.as_ref().ok_or(Error::Binding)?,
                                    start: row * width,
                                }
                            } else {
                                Coordinates::View {
                                    view: &op.input,
                                    start: row * width,
                                }
                            },
                            lanes: width,
                        })?;
                    }
                }
                PublishedKind::Binary { right, .. } => {
                    for start in (0..op.output.lanes).step_by(128) {
                        for (offset, view) in [&op.input, right].into_iter().enumerate() {
                            let lanes = (op.output.lanes - start).min(128);
                            if start == 0 && lanes == view.lanes {
                                continue;
                            }
                            visit(GatherSpan {
                                ordinal: op.first + 3 * (start / 128) + offset,
                                coordinates: Coordinates::View { view, start },
                                lanes,
                            })?;
                        }
                    }
                }
                PublishedKind::Argmax(plan) => {
                    plan.visit(&op.input, &mut |event| {
                        if let ArgmaxCode::Gather(gather) = event.code {
                            let layout = gather.layout();
                            visit(GatherSpan {
                                ordinal: event.owner,
                                lanes: event.lanes,
                                coordinates: Coordinates::Argmax {
                                    input: &op.input,
                                    plan,
                                    round: layout.round,
                                    field: layout.field,
                                    start: layout.start,
                                    parity: layout.parity,
                                },
                            })?;
                        }
                        Ok(())
                    })?;
                }
                _ => (),
            }
        }
        Ok(())
    }

    // Validate native geometry and immutable descriptor extents before control
    // allocation. The shared finalizer still validates every dependency/lane and
    // derives all work itself; this is not a replacement numeric/source verifier.
    fn validate_recipe_geometry(&self) -> Result<(), Error> {
        let mut next = self.imports_owner.nodes.len();
        if let Some(prelude) = &self.request_prelude {
            if prelude.first() != next {
                return Err(Error::Binding);
            }
            next = add_bytes(next, 2)?;
        }
        let mut sticky = self.imports_owner.validity;
        if self.next > MAX_NODES {
            return Err(Error::Bounds);
        }
        for (slot, op) in self.ops.iter().enumerate() {
            if op.first != next || op.input.lanes == 0 || op.output.lanes == 0 {
                return Err(Error::Binding);
            }
            let count = match &op.kind {
                PublishedKind::AttentionValues(values) => {
                    if op.output.mapping
                        != (Mapping::AttentionValues {
                            first: op.first,
                            layout: values.layout()?,
                            products: values.tiles.len(),
                        })
                    {
                        return Err(Error::Binding);
                    }
                    values.nodes
                }
                PublishedKind::AttentionSoftmax(rows) => {
                    if rows.first() != op.first {
                        return Err(Error::Binding);
                    }
                    rows.len().checked_mul(2).ok_or(Error::Bounds)?
                }
                PublishedKind::AttentionScale(rows) => {
                    if op.output.mapping
                        != (Mapping::Rows {
                            first: op.first,
                            width: 128,
                        })
                        || rows.native.iter().enumerate().any(|(i, c)| {
                            c.start != 128 * i
                                || c.lanes != (op.input.lanes - c.start).min(128)
                                || c.program.lanes() != c.lanes
                                || c.program.output_lanes() != c.lanes
                        })
                    {
                        return Err(Error::Binding);
                    }
                    rows.virtual_nodes()?
                }
                PublishedKind::AttentionScores(scores) => {
                    let layout = scores.layout()?;
                    if op.output.mapping
                        != (Mapping::AttentionScores {
                            first: op.first,
                            layout,
                            products: scores.tiles.len(),
                        })
                        || scores.work
                            != scores
                                .tiles
                                .iter()
                                .try_fold(WorkBudget::default(), |w, t| {
                                    w.checked_add(WorkBudget::product(t.count, true)?)
                                })?
                    {
                        return Err(Error::Binding);
                    }
                    scores.nodes
                }
                PublishedKind::TokenLookup {
                    weight,
                    source,
                    plan,
                } => {
                    if source.shape.len() != 2
                        || source.shape[0] != plan.vocabulary
                        || source.shape[1] != op.output.shape[2]
                        || self.imported_recipe(*weight).is_none()
                        || op.input.lanes.checked_mul(source.shape[1]) != Some(op.output.lanes)
                    {
                        return Err(Error::Shape);
                    }
                    check_loop(
                        plan.vocabulary,
                        &plan.full.program,
                        plan.tail.as_ref().map(|tail| &tail.program),
                    )?;
                    op.input.lanes.checked_mul(3).ok_or(Error::Bounds)?
                }
                PublishedKind::Projection {
                    weight,
                    geometry: [rows, inner, columns],
                    bias,
                    ..
                } => {
                    if rows.checked_mul(*inner) != Some(op.input.lanes)
                        || rows.checked_mul(*columns) != Some(op.output.lanes)
                        || self.imported_recipe(*weight).is_none()
                    {
                        return Err(Error::Shape);
                    }
                    self.check_prepared(slot, op.output.lanes)?;
                    if let Some(bias) = bias {
                        let original = self
                            .imports_owner
                            .nodes
                            .get(bias.owner)
                            .ok_or(Error::Binding)?;
                        let Code::PublicInput { key, .. } = &original.code else {
                            return Err(Error::Binding);
                        };
                        if !self
                            .sources
                            .get(key)
                            .is_some_and(|source| std::ptr::eq(source, bias.source))
                            || bias.source.shape != [*columns]
                            || original.lanes != *columns
                            || original.bounds != bias.bounds()?
                            || op.output.mapping != Mapping::Products(op.first + 3)
                            || bias.floor_bounds != self.prepared(slot)?.output_bounds()?
                            || bias.coefficients.output_bounds(InputBounds {
                                left: bias.floor_bounds,
                                right: bias.bounds()?,
                            })? != op.output.bounds
                        {
                            return Err(Error::Binding);
                        }
                        add_bytes(3, ProjectionBias::nodes(op.output.lanes)?)?
                    } else {
                        3
                    }
                }
                PublishedKind::Silu => {
                    if op.input.lanes != op.output.lanes {
                        return Err(Error::Shape);
                    }
                    self.check_prepared(slot, op.output.lanes)?;
                    2
                }
                PublishedKind::Rotary(rows) => {
                    if rows.width == 0
                        || rows.native.len().checked_mul(rows.width) != Some(op.output.lanes)
                        || op.input.lanes != op.output.lanes
                        || op.output.logical.rank != 4
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
                PublishedKind::RmsNorm {
                    program,
                    checked,
                    row_bounds,
                    ..
                } => {
                    let [_, rows, width] = op.output.shape;
                    if rows.checked_mul(width) != Some(op.output.lanes)
                        || op.input.lanes != op.output.lanes
                        || width != program.lanes()
                        || width != program.output_lanes()
                    {
                        return Err(Error::Shape);
                    }
                    let guards = if let Some(checked) = checked {
                        if self._prebound_inputs.is_none() {
                            let context = self.source_context.ok_or(Error::Binding)?;
                            context.check_imports(self.imports_owner)?;
                            self.validity
                                .check_frozen_owner(self.imports_owner, &self.origin)?;
                        }
                        if self.imports_owner.checked_bound.is_none() {
                            return Err(Error::Binding);
                        }
                        checked.validate(self.imports_owner, &op.input, op.first, width, sticky)?;
                        if checked.accepted != *row_bounds {
                            return Err(Error::Binding);
                        }
                        sticky = checked.outgoing;
                        checked.nodes()
                    } else {
                        if self.imports_owner.checked_bound.is_some() {
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
                PublishedKind::Binary { right, .. } => {
                    if op.input.lanes != op.output.lanes || right.lanes != op.output.lanes {
                        return Err(Error::Shape);
                    }
                    op.output
                        .lanes
                        .div_ceil(128)
                        .checked_mul(3)
                        .ok_or(Error::Bounds)?
                }
                PublishedKind::LastToken { .. }
                | PublishedKind::TokenFeedback
                | PublishedKind::Reshape
                | PublishedKind::KvInitialize { .. }
                | PublishedKind::CacheSuffix
                | PublishedKind::CausalMask => 0,
                PublishedKind::Argmax(plan) => {
                    let expected = ArgmaxGeometry::new(op.input.lanes)?;
                    if plan.first != op.first
                        || plan.geometry.lanes != expected.lanes
                        || plan.geometry.rounds != expected.rounds
                        || plan.geometry.blocks != expected.blocks
                        || plan.geometry.kernels != expected.kernels
                        || plan.programs.len() != expected.kernels
                        || op.output.lanes != 1
                        || op.output.word(0)? != plan.output
                        || op.output.bounds != Interval::new(0, op.input.lanes as i64 - 1)?
                    {
                        return Err(Error::Shape);
                    }
                    for kernel in &plan.programs {
                        if kernel.lanes == 0
                            || kernel.lanes > puma::MAX_LANES
                            || kernel.program.lanes() != kernel.lanes
                            || kernel.program.output_lanes() != kernel.lanes
                            || kernel.program.validity_lanes().is_some()
                            || kernel.program.output_bounds() != Interval::new(0, 1)?
                        {
                            return Err(Error::Shape);
                        }
                    }
                    // Original generator checks rounds, carries, dependencies,
                    // final output and native work without rebuilding kernels.
                    plan.visit(&op.input, &mut |_| Ok(()))?;
                    expected.nodes()?
                }
            };
            next = add_bytes(next, count)?;
        }
        if next != self.next || sticky != self.validity.current() {
            return Err(Error::Binding);
        }
        Ok(())
    }

    fn check_prepared(&self, slot: usize, lanes: usize) -> Result<(), Error> {
        let prepared = self.prepared(slot)?;
        prepared.validate()?;
        if prepared.lanes() != lanes {
            return Err(Error::Shape);
        }
        Ok(())
    }
}

fn check_loop(
    lanes: usize,
    full: &Arc<tensor::Program>,
    tail: Option<&Arc<tensor::Program>>,
) -> Result<(), Error> {
    if lanes == 0 || full.lanes() == 0 || full.output_lanes() != full.lanes() {
        return Err(Error::Shape);
    }
    match (lanes % full.lanes(), tail) {
        (0, None) => Ok(()),
        (n, Some(tail)) if n != 0 && tail.lanes() == n && tail.output_lanes() == n => Ok(()),
        _ => Err(Error::Shape),
    }
}

impl FrozenSchedule<'_, '_> {
    pub(in crate::replicated_decoder) fn resources(&self) -> RecipeResources {
        self.resources
    }

    fn gather(&self, ordinal: usize) -> Result<BorrowedRecipe<'_>, Error> {
        let index = self
            .gathers
            .binary_search_by_key(&ordinal, |span| span.ordinal)
            .map_err(|_| Error::Binding)?;
        Ok(BorrowedRecipe::Gather(&self.gathers[index]))
    }

    fn view_gather<'s>(
        &'s self,
        ordinal: usize,
        view: &'s PublishedView,
        start: usize,
        lanes: usize,
    ) -> Result<BorrowedRecipe<'s>, Error> {
        if start == 0 && lanes == view.lanes {
            Ok(BorrowedRecipe::Gather(view))
        } else {
            self.gather(ordinal)
        }
    }

    fn prepared_loop(&self, slot: usize, input: usize) -> Result<BorrowedRecipe<'_>, Error> {
        let prepared = self.owner.prepared(slot)?;
        let full = prepared.program_at(0)?;
        let remainder = prepared.lanes() % full.lanes();
        let tail = if remainder == 0 {
            None
        } else {
            Some(prepared.program_at(prepared.lanes() - remainder)?)
        };
        Ok(BorrowedRecipe::TensorLoop {
            input: LoopInput::Slice(input),
            program: full,
            tail,
        })
    }

    fn checked_node<'s>(
        &'s self,
        op: &'s PublishedOp<'_>,
        rows: &'s compact_validity::CheckedRows<'_>,
        ordinal: usize,
    ) -> Result<BorrowedNode<'s>, Error> {
        let mut found = None;
        rows.visit(&mut |event| {
            use compact_validity::{GuardCode, GuardGather};
            if event.owner != ordinal {
                return Ok(());
            }
            let recipe = match event.code {
                GuardCode::Gather(GuardGather::Original { start }) => {
                    self.view_gather(ordinal, &op.input, start, event.lanes)?
                }
                GuardCode::Gather(GuardGather::Native { .. }) => self.gather(ordinal)?,
                GuardCode::Checked { input, program } => {
                    BorrowedRecipe::CheckedRange { input, program }
                }
                GuardCode::AllValid { input, program } => BorrowedRecipe::Tensor { input, program },
                GuardCode::Product { left, right } => BorrowedRecipe::Product {
                    left,
                    right,
                    shape: None,
                    rescale: false,
                    norm: None,
                },
            };
            found = Some(BorrowedNode {
                lanes: event.lanes,
                bounds: event.bounds,
                recipe,
            });
            Ok(())
        })?;
        found.ok_or(Error::Binding)
    }

    fn argmax_node(&self, slot: usize, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
        let op = &self.owner.ops[slot];
        let PublishedKind::Argmax(plan) = &op.kind else {
            return Err(Error::Binding);
        };
        if ordinal == op.first {
            return Ok(BorrowedNode {
                lanes: op.input.lanes,
                bounds: Interval::new(0, op.input.lanes as i64 - 1)?,
                recipe: BorrowedRecipe::GeneratedPublicValues(
                    GeneratedPublicValues::argmax_indices(self.owner, slot)?,
                ),
            });
        }
        let block = (ordinal - op.first - 1) / 12;
        let mut found = None;
        replay_argmax_block(plan, &op.input, block, &mut |event| {
            if event.owner != ordinal {
                return Ok(());
            }
            let recipe = match event.code {
                ArgmaxCode::PublicIndices => return Err(Error::Binding),
                ArgmaxCode::Gather(_) => self.gather(ordinal)?,
                ArgmaxCode::Affine {
                    left,
                    right,
                    coefficients,
                } => BorrowedRecipe::Affine {
                    left,
                    right: Some(right),
                    coefficients,
                },
                ArgmaxCode::Product { left, right } => BorrowedRecipe::Product {
                    left,
                    right,
                    shape: None,
                    rescale: false,
                    norm: None,
                },
                ArgmaxCode::Tensor { input, program } => {
                    // Recover the stable ORIGINAL handle from its owner, never
                    // extend an event loan, clone an Arc or substitute a digest.
                    let kernel = plan
                        .programs
                        .iter()
                        .find(|kernel| std::ptr::eq(&kernel.program, program))
                        .ok_or(Error::Binding)?;
                    BorrowedRecipe::Tensor {
                        input,
                        program: &kernel.program,
                    }
                }
            };
            found = Some(BorrowedNode {
                lanes: event.lanes,
                bounds: event.bounds,
                recipe,
            });
            Ok(())
        })?;
        found.ok_or(Error::Binding)
    }
}

/// Replay one original 12-node compare/select block through the SAME traversal
/// and numeric sink. Indexing skips earlier blocks, not their ordinals/carries.
/// No per-node recipe table or quadratic whole-tree replay is needed.
struct ArgmaxBlockReplay<'s, 'v> {
    sink: ArgmaxNumeric<'s, 'v>,
    round: usize,
    start: usize,
    pairs: usize,
    stopped: bool,
}

impl ArgmaxSink for ArgmaxBlockReplay<'_, '_> {
    type Owner = Value;
    type Output = ();
    fn indices(&mut self, lanes: usize) -> Result<(), Error> {
        if lanes != 2 * self.pairs {
            return Err(Error::Shape);
        }
        Ok(()) // Original iota has its own genuine generated-public recipe.
    }
    fn begin_round(&mut self, round: usize, _: usize) -> Result<(), Error> {
        if round != 0 {
            return Err(Error::Binding);
        }
        let length = self.sink.geometry.level(self.round)?.0;
        self.sink.begin_round(self.round, length)
    }
    fn gather_pairs(
        &mut self,
        field: ArgmaxField,
        start: usize,
        pairs: usize,
        parity: usize,
    ) -> Result<Value, Error> {
        if start != 0 || pairs != self.pairs {
            return Err(Error::Binding);
        }
        self.sink.gather_pairs(field, self.start, pairs, parity)
    }
    fn difference(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        self.sink.difference(left, right)
    }
    fn negative(&mut self, difference: Value) -> Result<Value, Error> {
        self.sink.negative(difference)
    }
    fn select(&mut self, bit: Value, yes: Value, no: Value) -> Result<Value, Error> {
        ArgmaxSink::select(&mut self.sink, bit, yes, no)
    }
    fn publish_pairs(&mut self, best: Value, index: Value, lanes: usize) -> Result<(), Error> {
        self.sink.publish_pairs(best, index, lanes)?;
        self.stopped = true;
        Err(Error::Retired) // Private, checked early-stop; never caller authority.
    }
    fn carry(&mut self, _: usize) -> Result<(), Error> {
        Err(Error::Binding)
    }
    fn end_round(&mut self) -> Result<(), Error> {
        Err(Error::Binding)
    }
    fn finish(&mut self) -> Result<(), Error> {
        Err(Error::Binding)
    }
}

fn replay_argmax_block(
    plan: &CompactArgmax,
    input: &PublishedView,
    block: usize,
    visitor: &mut dyn for<'e> FnMut(ArgmaxEvent<'e>) -> Result<(), Error>,
) -> Result<(), Error> {
    let mut position = None;
    for round in 0..plan.geometry.rounds {
        let (length, first) = plan.geometry.level(round)?;
        let end = plan.geometry.level(round + 1)?.1;
        if block >= first && block < end {
            let start = (block - first).checked_mul(256).ok_or(Error::Bounds)?;
            position = Some((round, start, (length - start).min(256) / 2));
            break;
        }
    }
    let (round, start, pairs) = position.ok_or(Error::Binding)?;
    let mut sink = ArgmaxNumeric::new(
        input,
        plan.first,
        plan.geometry,
        NegativePrograms::Borrowed(&plan.programs),
        Some(visitor),
    )?;
    sink.next = plan
        .first
        .checked_add(1)
        .and_then(|n| n.checked_add(block.checked_mul(12)?))
        .ok_or(Error::Bounds)?;
    let mut replay = ArgmaxBlockReplay {
        sink,
        round,
        start,
        pairs,
        stopped: false,
    };
    match argmax_tree(&mut replay, 2 * pairs) {
        Err(Error::Retired) if replay.stopped => Ok(()),
        Err(error) => Err(error),
        Ok(()) => Err(Error::Binding),
    }
}

impl BorrowedLaneRefs for compact_attention_softmax::OwnedRow<'_> {
    fn len(&self) -> usize {
        self.native_width()
    }
    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        self.input_word(index)
    }
}

impl BorrowedOrderedNodes for FrozenSchedule<'_, '_> {
    fn node_count(&self) -> usize {
        self.owner.next
    }

    fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
        if let Some(node) = self.owner.imported_recipe(ordinal) {
            return Ok(BorrowedNode::from(node));
        }
        if let Some(prelude) = &self.owner.request_prelude {
            if ordinal >= prelude.first() && ordinal < add_bytes(prelude.first(), 2)? {
                return prelude.node(ordinal);
            }
        }
        if ordinal >= self.owner.next {
            return Err(Error::Binding);
        }
        // Zero-node aliases can share the following owner's start. Use the last
        // descriptor starting no later than ordinal, never invent alias nodes.
        let slot = self
            .owner
            .ops
            .partition_point(|op| op.first <= ordinal)
            .checked_sub(1)
            .ok_or(Error::Binding)?;
        let op = &self.owner.ops[slot];
        if matches!(op.kind, PublishedKind::Argmax(_)) {
            return self.argmax_node(slot, ordinal);
        }
        let offset = ordinal - op.first;
        let (lanes, bounds, recipe) = match &op.kind {
            PublishedKind::AttentionValues(values) => {
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
                            t.right_offset == offset
                                || t.right_offset.checked_add(1) == Some(offset)
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
            PublishedKind::AttentionSoftmax(rows) => {
                let row = rows.row(offset / 2)?;
                if offset % 2 == 0 {
                    (
                        row.native_width(),
                        row.input_bounds(),
                        BorrowedRecipe::Gather(row),
                    )
                } else {
                    (
                        row.native_width(),
                        row.output_bounds(),
                        BorrowedRecipe::Tensor {
                            input: row.gather_ordinal(),
                            program: row.program(),
                        },
                    )
                }
            }
            PublishedKind::AttentionScale(rows) => {
                let chunk = rows.native.get(offset / 2).ok_or(Error::Binding)?;
                if offset % 2 == 0 {
                    (chunk.lanes, chunk.incoming, self.gather(ordinal)?)
                } else {
                    (
                        chunk.lanes,
                        chunk.program.output_bounds(),
                        BorrowedRecipe::Tensor {
                            input: ordinal - 1,
                            program: &chunk.program,
                        },
                    )
                }
            }
            PublishedKind::AttentionScores(scores) => {
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
                            t.right_offset == offset
                                || t.right_offset.checked_add(1) == Some(offset)
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
            PublishedKind::TokenLookup { weight, plan, .. } => {
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
                                tail: plan.tail.as_ref().map(|tail| &tail.program),
                            },
                        )
                    }
                    1 => (plan.vocabulary, plan.selector_bounds, self.gather(ordinal)?),
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
            PublishedKind::Projection {
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
                    bias.as_ref()
                        .map_or(op.output.bounds, |bias| bias.floor_bounds),
                    self.prepared_loop(slot, op.first + 1)?,
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
                    match (offset - 3) % 3 {
                        0 => (lanes, bias.floor_bounds, self.gather(ordinal)?),
                        1 => (lanes, bias.bounds()?, self.gather(ordinal)?),
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
            PublishedKind::Silu => match offset {
                0 => (
                    op.input.lanes,
                    op.input.bounds,
                    BorrowedRecipe::Gather(&op.input),
                ),
                1 => (
                    op.output.lanes,
                    op.output.bounds,
                    self.prepared_loop(slot, op.first)?,
                ),
                _ => return Err(Error::Binding),
            },
            PublishedKind::Rotary(rows) => {
                let row = rows.native.get(offset / 2).ok_or(Error::Binding)?;
                if offset % 2 == 0 {
                    (
                        rows.width,
                        row.incoming,
                        self.view_gather(
                            ordinal,
                            &op.input,
                            rows.width * (offset / 2),
                            rows.width,
                        )?,
                    )
                } else {
                    (
                        rows.width,
                        row.bounds,
                        BorrowedRecipe::Tensor {
                            input: ordinal - 1,
                            program: &row.program,
                        },
                    )
                }
            }
            PublishedKind::RmsNorm {
                program,
                checked,
                row_bounds,
                ..
            } => {
                let width = op.output.shape[2];
                let first = checked.as_ref().map_or(op.first, |rows| rows.rows_first());
                if ordinal < first {
                    return self.checked_node(op, checked.as_ref().ok_or(Error::Binding)?, ordinal);
                }
                let offset = ordinal - first;
                if offset % 2 == 0 {
                    (
                        width,
                        *row_bounds,
                        if checked.as_ref().is_some_and(|rows| rows.guarded()) {
                            self.gather(ordinal)?
                        } else {
                            self.view_gather(ordinal, &op.input, width * (offset / 2), width)?
                        },
                    )
                } else {
                    (
                        width,
                        program.output_bounds(),
                        BorrowedRecipe::Tensor {
                            input: ordinal - 1,
                            program,
                        },
                    )
                }
            }
            PublishedKind::Binary { right, product, .. } => {
                let start = 128 * (offset / 3);
                let lanes = (op.output.lanes - start).min(128);
                match offset % 3 {
                    0 => (
                        lanes,
                        op.input.bounds,
                        self.view_gather(ordinal, &op.input, start, lanes)?,
                    ),
                    1 => (
                        lanes,
                        right.bounds,
                        self.view_gather(ordinal, right, start, lanes)?,
                    ),
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
            _ => return Err(Error::Binding),
        };
        Ok(BorrowedNode {
            lanes,
            bounds,
            recipe,
        })
    }
}

#[cfg(test)]
#[path = "compact_frozen_recipe_tests.rs"]
mod tests;
