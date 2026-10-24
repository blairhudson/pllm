//! Compiler-local checked-row custody. No prefix-export or request acceptance issuer.
use super::*;
#[path = "compact_rms_admission.rs"]
mod request_rms;

/// The original immutable import owner and its one continuing sticky scalar.
/// Never initialized from a caller's numeric ID or a purported bit interval.
pub(super) struct RequestValidity<'a> {
    original: &'a Builder,
    origin: Arc<()>,
    current: Option<usize>,
    // Compiler continuation epoch, not a runtime drain/copy assertion.
    epoch: usize,
}

impl<'a> RequestValidity<'a> {
    pub(super) fn original(original: &'a Builder, origin: Arc<()>) -> Self {
        Self {
            original,
            origin,
            current: original.validity,
            epoch: 0,
        }
    }

    pub(super) fn current(&self) -> Option<usize> {
        self.current
    }

    pub(super) fn epoch(&self) -> usize {
        self.epoch
    }

    pub(super) fn check_frozen_owner(
        &self,
        original: &Builder,
        origin: &Arc<()>,
    ) -> Result<(), Error> {
        if !std::ptr::eq(self.original, original) || !Arc::ptr_eq(&self.origin, origin) {
            return Err(Error::Binding);
        }
        Ok(())
    }

    pub(super) fn check(&self, graph: &CompactGraph<'_, 'a>) -> Result<(), Error> {
        if !std::ptr::eq(self.original, graph.imports)
            || !Arc::ptr_eq(&self.origin, &graph.origin)
            || self.original.validity.is_some()
            || graph.cursor.checked_sub(graph.ops.len()) != Some(0)
        {
            // There is no genuine incoming prefix-export/sticky issuer yet.
            return Err(Error::Binding);
        }
        if graph.prebound_inputs.is_none() {
            // Genuine consumed source request is the other ORIGINAL input
            // issuer. It binds this live import allocation and semantic plan;
            // an arbitrary View/coordinate/digest still cannot enter custody.
            let context = graph.request_admission.ok_or(Error::Binding)?;
            context.check_imports(self.original)?;
            if !std::ptr::eq(
                &context.plan().plan().prefill,
                graph.frontier.semantic_graph(),
            ) {
                return Err(Error::Binding);
            }
        }
        Ok(())
    }

    pub(super) fn commit(&mut self, rows: &CheckedRows<'a>) -> Result<(), Error> {
        if !std::ptr::eq(self.original, rows.original)
            || !Arc::ptr_eq(&self.origin, &rows.input.origin)
            || self.current != rows.incoming
        {
            return Err(Error::Binding);
        }
        self.current = rows.outgoing;
        Ok(())
    }
}

/// One contiguous run of exactly identical ORIGINAL guard factories. Full/tail
/// and heterogeneous input bounds stay distinct; no tile/lane reference array.
pub(super) struct GuardRun {
    first_tile: usize,
    tiles: usize,
    pub(super) lanes: usize,
    pub(super) incoming: Interval,
    pub(super) checked: Arc<tensor::Program>,
    owns_checked: bool,
    pub(super) all_valid: Arc<tensor::Program>,
    owns_valid: bool,
}

pub(super) struct CheckedRows<'a> {
    original: &'a Builder,
    input: PublishedView,
    width: usize,
    pub(super) accepted: Interval,
    first: usize,
    pub(super) incoming: Option<usize>,
    pub(super) outgoing: Option<usize>,
    runs: Vec<GuardRun>,
    pub(super) work: WorkBudget,
}

/// Sealed ordered LOCAL row binding, not runtime acceptance or an energy proof.
/// Guarded case binds genuine factory safe outputs. Static alias preserves the
/// original owner/domain obligation exactly; it does not prove that declaration.
/// Neither a caller interval nor attention energy can construct this binding.
pub(super) struct SafeRow<'s, 'a> {
    rows: &'s CheckedRows<'a>,
    start: usize,
}

impl SafeRow<'_, '_> {
    pub(super) fn bounds(&self) -> Interval {
        self.rows.accepted
    }
    pub(super) fn word(&self, lane: usize) -> Result<(usize, usize), Error> {
        if lane >= self.rows.width {
            return Err(Error::Shape);
        }
        self.rows.word(self.start + lane)
    }
}

#[derive(Clone, Copy)]
pub(super) enum GuardGather {
    Original { start: usize },
    Native { owner: usize, start: usize },
}

pub(super) enum GuardCode<'s> {
    Gather(GuardGather),
    Checked {
        input: usize,
        program: &'s Arc<tensor::Program>,
    },
    AllValid {
        input: usize,
        program: &'s Arc<tensor::Program>,
    },
    Product {
        left: usize,
        right: usize,
    },
}

pub(super) struct GuardEvent<'s> {
    pub(super) owner: usize,
    pub(super) lanes: usize,
    pub(super) bounds: Interval,
    pub(super) code: GuardCode<'s>,
}

pub(super) type PreparedTransition<'a> = (
    Option<CheckedRows<'a>>,
    Interval,
    Option<u128>,
    WorkBudget,
    Option<tensor::AdmittedFactory<'a>>,
);

fn control_slots() -> Result<usize, Error> {
    use std::mem::size_of;
    let mut bytes = 0;
    for slot in [
        2 * size_of::<Result<PreparedTransition<'_>, Error>>(),
        size_of::<CheckedRows<'_>>(),
        size_of::<GuardRun>(),
        size_of::<SafeRow<'_, '_>>(),
        4 * size_of::<tensor::Construction>(), // work_frames' concrete array
        2 * size_of::<tensor::ConstructionQuote>(),
        size_of::<Result<tensor::ConstructionQuote, tensor::ConstructionQuoteError>>(),
        4 * size_of::<Result<WorkBudget, Error>>(),
        2 * size_of::<WorkBudget>(),
    ] {
        bytes = add_bytes(bytes, slot)?;
    }
    Ok(bytes)
}

fn same_view(a: &PublishedView, b: &PublishedView) -> bool {
    Arc::ptr_eq(&a.origin, &b.origin)
        && a.slot == b.slot
        && a.mapping == b.mapping
        && a.start == b.start
        && a.lanes == b.lanes
        && a.logical == b.logical
        && a.order == b.order
        && a.import_slot == b.import_slot
        && a.bounds == b.bounds
        && a.shape == b.shape
        && a.rank == b.rank
}

impl<'a> CheckedRows<'a> {
    #[cfg(test)]
    pub(super) fn guard_tiles(&self) -> usize {
        self.runs.iter().map(|run| run.tiles).sum()
    }
    pub(super) fn nodes(&self) -> usize {
        if self.runs.is_empty() {
            0
        } else {
            6 * self.input.lanes.div_ceil(128) - usize::from(self.incoming.is_none())
        }
    }

    pub(super) fn rows_first(&self) -> usize {
        self.first + self.nodes()
    }
    pub(super) fn guarded(&self) -> bool {
        !self.runs.is_empty()
    }

    fn tile_first(&self, tile: usize) -> usize {
        self.first + 6 * tile - usize::from(tile > 0 && self.incoming.is_none())
    }

    pub(super) fn row<'s>(
        &'s self,
        original: &Builder,
        input: &PublishedView,
        start: usize,
        width: usize,
    ) -> Result<SafeRow<'s, 'a>, Error> {
        if !std::ptr::eq(self.original, original)
            || !same_view(&self.input, input)
            || width != self.width
            || start % width != 0
            || start.checked_add(width).is_none_or(|end| end > input.lanes)
        {
            return Err(Error::Binding);
        }
        Ok(SafeRow { rows: self, start })
    }

    pub(super) fn word(&self, lane: usize) -> Result<(usize, usize), Error> {
        if lane >= self.input.lanes {
            return Err(Error::Shape);
        }
        if self.runs.is_empty() {
            return self.input.word(lane);
        }
        Ok((self.tile_first(lane / 128) + 2, lane % 128))
    }

    pub(super) fn programs(&self) -> impl Iterator<Item = &Arc<tensor::Program>> {
        self.runs.iter().flat_map(|run| {
            [
                run.owns_checked.then_some(&run.checked),
                run.owns_valid.then_some(&run.all_valid),
            ]
            .into_iter()
            .flatten()
        })
    }

    pub(super) fn metadata_bytes(&self) -> Result<usize, Error> {
        array_bytes::<GuardRun>(self.runs.capacity())
    }

    pub(super) fn baseline(&self) -> Result<usize, Error> {
        self.programs()
            .try_fold(self.metadata_bytes()?, |bytes, program| {
                add_bytes(bytes, program.resources()?.immutable_template_bytes)
            })
    }

    pub(super) fn validate(
        &self,
        original: &Builder,
        input: &PublishedView,
        first: usize,
        width: usize,
        incoming: Option<usize>,
    ) -> Result<(), Error> {
        self.row(original, input, 0, width)?;
        if first != self.first || incoming != self.incoming {
            return Err(Error::Binding);
        }
        let mut tiles = 0;
        for run in &self.runs {
            if run.first_tile != tiles
                || run.tiles == 0
                || run.lanes
                    != input
                        .lanes
                        .checked_sub(128usize.checked_mul(tiles).ok_or(Error::Bounds)?)
                        .ok_or(Error::Binding)?
                        .min(128)
                || (run.lanes != 128 && run.tiles != 1)
                || run.checked.lanes() != run.lanes
                || run.checked.output_lanes() != run.lanes
                || run.checked.validity_lanes() != Some(run.lanes)
                || run.checked.output_bounds() != self.accepted
                || run.all_valid.lanes() != run.lanes
                || run.all_valid.output_lanes() != 1
                || run.all_valid.validity_lanes().is_some()
            {
                return Err(Error::Binding);
            }
            tiles = add_bytes(tiles, run.tiles)?;
        }
        let outgoing = if tiles == 0 {
            incoming
        } else {
            if tiles != input.lanes.div_ceil(128) {
                return Err(Error::Binding);
            }
            Some(self.first + self.nodes() - 1)
        };
        if self.outgoing != outgoing {
            return Err(Error::Binding);
        }
        let mut actual = WorkBudget::default();
        for run in &self.runs {
            actual = actual.checked_add(
                run.checked
                    .work_budget()
                    .checked_add(run.all_valid.work_budget())?
                    .checked_mul(run.tiles as u64)?,
            )?;
        }
        if tiles > 0 {
            actual = actual.checked_add(
                WorkBudget::product(1, false)?
                    .checked_mul((tiles - usize::from(incoming.is_none())) as u64)?,
            )?;
        }
        if actual != self.work {
            return Err(Error::Binding);
        }
        Ok(())
    }

    /// One original traversal shared by hashing and frozen schedule publication.
    /// All these events precede every RMS row event, including heterogeneous runs.
    pub(super) fn visit<'s>(
        &'s self,
        visit: &mut impl FnMut(GuardEvent<'s>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let bits = Interval::new(0, 1)?;
        let owner_bounds = Interval::new(self.accepted.lower(), self.accepted.upper().max(1))?;
        let mut previous = self.incoming;
        for run in &self.runs {
            for tile in run.first_tile..run.first_tile + run.tiles {
                let first = self.tile_first(tile);
                let n = run.lanes;
                visit(GuardEvent {
                    owner: first,
                    lanes: n,
                    bounds: run.incoming,
                    code: GuardCode::Gather(GuardGather::Original { start: tile * 128 }),
                })?;
                visit(GuardEvent {
                    owner: first + 1,
                    lanes: 2 * n,
                    bounds: owner_bounds,
                    code: GuardCode::Checked {
                        input: first,
                        program: &run.checked,
                    },
                })?;
                visit(GuardEvent {
                    owner: first + 2,
                    lanes: n,
                    bounds: self.accepted,
                    code: GuardCode::Gather(GuardGather::Native {
                        owner: first + 1,
                        start: 0,
                    }),
                })?;
                visit(GuardEvent {
                    owner: first + 3,
                    lanes: n,
                    bounds: bits,
                    code: GuardCode::Gather(GuardGather::Native {
                        owner: first + 1,
                        start: n,
                    }),
                })?;
                visit(GuardEvent {
                    owner: first + 4,
                    lanes: 1,
                    bounds: bits,
                    code: GuardCode::AllValid {
                        input: first + 3,
                        program: &run.all_valid,
                    },
                })?;
                previous = Some(if let Some(left) = previous {
                    visit(GuardEvent {
                        owner: first + 5,
                        lanes: 1,
                        bounds: bits,
                        code: GuardCode::Product {
                            left,
                            right: first + 4,
                        },
                    })?;
                    first + 5
                } else {
                    first + 4
                });
            }
        }
        if previous != self.outgoing {
            return Err(Error::Binding);
        }
        Ok(())
    }
}

/// Exact original gather hull, from original nodes or compiler-owned output
/// publications. This is numeric metadata, NOT a validity/row certificate issuer.
pub(super) fn input_bounds(
    graph: &CompactGraph<'_, '_>,
    view: &PublishedView,
    start: usize,
    lanes: usize,
) -> Result<Interval, Error> {
    graph.validate_view(view)?;
    if lanes == 0 || start.checked_add(lanes).is_none_or(|end| end > view.lanes) {
        return Err(Error::Shape);
    }
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    for lane in start..start + lanes {
        let (owner, _) = view.word(lane)?;
        let bounds = if let Some(node) = graph.imports.nodes.get(owner) {
            node.bounds
        } else {
            let op = graph
                .ops
                .iter()
                .rev()
                .find(|op| {
                    op.first <= owner
                        && !matches!(
                            op.kind,
                            PublishedKind::Reshape
                                | PublishedKind::LastToken { .. }
                                | PublishedKind::TokenFeedback
                        )
                })
                .ok_or(Error::Binding)?;
            match &op.kind {
                PublishedKind::RmsNorm {
                    program, checked, ..
                } => {
                    let first = checked.as_ref().map_or(op.first, |rows| rows.rows_first());
                    if owner < first || (owner - first) % 2 != 1 {
                        return Err(Error::Binding);
                    }
                    program.output_bounds()
                }
                PublishedKind::Projection { bias, .. } => {
                    let slot = op.output.slot.ok_or(Error::Binding)?;
                    let floor = graph.kernel(slot)?.output_bounds()?;
                    let bounds = if let Some(bias) = bias {
                        if owner < op.first + 3 || (owner - op.first - 3) % 3 != 2 {
                            return Err(Error::Binding);
                        }
                        bias.coefficients.output_bounds(InputBounds {
                            left: floor,
                            right: graph
                                .imports
                                .nodes
                                .get(bias.owner)
                                .ok_or(Error::Binding)?
                                .bounds,
                        })?
                    } else {
                        if owner != op.first + 2 {
                            return Err(Error::Binding);
                        }
                        floor
                    };
                    if bounds != op.output.bounds {
                        return Err(Error::Binding);
                    }
                    bounds
                }
                PublishedKind::Silu => {
                    if owner != op.first + 1 {
                        return Err(Error::Binding);
                    }
                    graph
                        .kernel(op.output.slot.ok_or(Error::Binding)?)?
                        .output_bounds()?
                }
                PublishedKind::Binary { right, product, .. } => {
                    let offset = owner - op.first;
                    if offset % 3 != 2 {
                        return Err(Error::Binding);
                    }
                    let start = 128 * (offset / 3);
                    let n = op
                        .output
                        .lanes
                        .checked_sub(start)
                        .ok_or(Error::Binding)?
                        .min(128);
                    let bounds = InputBounds {
                        left: input_bounds(graph, &op.input, start, n)?,
                        right: input_bounds(graph, right, start, n)?,
                    };
                    let exact = if *product {
                        product_bounds(bounds, 1, false, true, None)?
                    } else {
                        affine(1, 1, 0).output_bounds(bounds)?
                    };
                    // Existing binary publication is uniform. Do not silently
                    // accept a newly heterogeneous predecessor it cannot encode.
                    if exact != op.output.bounds {
                        return Err(Error::Binding);
                    }
                    exact
                }
                PublishedKind::TokenLookup { weight, .. } => {
                    if (owner - op.first) % 3 != 2 {
                        return Err(Error::Binding);
                    }
                    graph
                        .imports
                        .nodes
                        .get(*weight)
                        .ok_or(Error::Binding)?
                        .bounds
                }
                _ => return Err(Error::Binding),
            }
        };
        lo = lo.min(bounds.lower());
        hi = hi.max(bounds.upper());
    }
    Interval::new(lo, hi)
}

/// Exact ordered GENERAL row lineage only. A whole-row permutation may move a
/// row, but cannot change its original full width or supply attention energy.
pub(super) fn row_energy(
    graph: &CompactGraph<'_, '_>,
    input: &PublishedView,
    row: usize,
) -> Result<Option<u128>, Error> {
    let width = input.logical.row_width();
    if row >= input.lanes / width {
        return Err(Error::Shape);
    }
    let Some(slot) = input.slot else {
        return Ok(None);
    };
    match &graph.ops[slot].kind {
        PublishedKind::Reshape | PublishedKind::LastToken { .. } | PublishedKind::TokenFeedback => {
            let Some((original, start)) = graph.alias_range(input, row * width, width)? else {
                return Ok(None);
            };
            if original.logical.row_width() != width || start % width != 0 {
                return Ok(None);
            }
            row_energy(graph, original, start / width)
        }
        PublishedKind::RmsNorm { output_energy, .. } => Ok(Some(*output_energy)),
        PublishedKind::Binary { row_energy, .. } => Ok(*row_energy),
        _ => Ok(graph.complete_row(input, row)?.map(|proof| proof.energy)),
    }
}

pub(super) fn guarded_energy(
    graph: &CompactGraph<'_, '_>,
    input: &PublishedView,
    row: usize,
    bound: Option<i64>,
) -> Result<Option<u128>, Error> {
    let energy = row_energy(graph, input, row)?;
    let Some(bound) = bound else {
        return Ok(energy);
    };
    let cap = crate::replicated_decoder::checked_row_energy::cap_energy(
        input.logical.row_width(),
        bound,
    )?;
    Ok(energy.and_then(|energy| {
        crate::replicated_decoder::checked_row_energy::strict_improvement(energy, cap)
    }))
}

fn previous_runs<'s>(graph: &'s CompactGraph<'_, '_>) -> impl Iterator<Item = &'s GuardRun> {
    graph
        .ops
        .iter()
        .filter_map(|op| match &op.kind {
            PublishedKind::RmsNorm {
                checked: Some(rows),
                ..
            } => Some(rows),
            _ => None,
        })
        .flat_map(|rows| rows.runs.iter())
}

/// Shared native recipe frames use their existing construction envelopes, in
/// addition to (not instead of) the fixed work workspace.
fn work_frames(width: usize, gamma_len: usize, lanes: usize) -> Result<usize, Error> {
    let mut bytes = 0;
    for construction in [
        tensor::Construction::RmsNormCertified {
            rows: 1,
            width,
            gamma_len,
        },
        tensor::Construction::RmsNormWithInputNorm {
            rows: 1,
            width,
            gamma_len,
        },
        tensor::Construction::CheckedRange {
            lanes: lanes.min(128),
        },
        tensor::Construction::AllValid {
            lanes: lanes.min(128),
        },
    ] {
        bytes = bytes.max(native_construction_quote(construction)?.new_peak_bytes);
    }
    Ok(bytes)
}

/// Known original work before compact container factories. Only direct original
/// state declarations at region start qualify. No unpublished producer proof or
/// numeric recipe is reconstructed from a declared future output envelope.
pub(super) fn pending_input_work(
    imports: &Builder,
    semantic: &DecoderGraph,
    start: usize,
    sources: &BTreeMap<String, SourceTensor>,
    baseline: usize,
    cap: usize,
) -> Result<(WorkBudget, usize), Error> {
    let mut work = WorkBudget::default();
    let mut peak = baseline;
    let mut sticky = imports.validity.is_some();
    if start != 0 || !imports.public_weights {
        return Ok((work, peak));
    }
    for op in &semantic.operations {
        if op.operator != Op::RmsNorm
            || op.inputs.len() != 1
            || op.attributes.get("weight_offset").and_then(|v| v.as_i64()) != Some(0)
        {
            break;
        }
        let Some(state) = semantic
            .state_inputs
            .iter()
            .find(|state| state.id == op.inputs[0])
        else {
            break;
        };
        let Some(node) = imports
            .nodes
            .iter()
            .find(|node| matches!(&node.code, Code::Input(key) if key == &state.id))
        else {
            break;
        };
        let Some(gamma) = op
            .attributes
            .get("weight")
            .and_then(|v| v.as_str())
            .and_then(|key| sources.get(key))
        else {
            break;
        };
        let Some(coefficients) = gamma.public_q18.as_ref() else {
            break;
        };
        let [1, rows, width] = state.shape.as_slice() else {
            break;
        };
        let (rows, width) = (
            usize::try_from(*rows).map_err(|_| Error::Bounds)?,
            usize::try_from(*width).map_err(|_| Error::Bounds)?,
        );
        if rows == 0
            || width == 0
            || rows.checked_mul(width) != Some(node.lanes)
            || state.shape != op.output_shape
            || gamma.shape != [width]
        {
            break;
        }
        let extra = add_bytes(
            compact_norm::driver_slots()?,
            add_bytes(
                tensor::RmsWorkWorkspace::workspace_bytes(),
                work_frames(width, coefficients.len(), node.lanes)?,
            )?,
        )?;
        peak = peak.max(add_bytes(baseline, extra)?);
        if peak > cap.min(MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        let mut workspace = tensor::RmsWorkWorkspace::empty();
        let mut bounds = node.bounds;
        if let Some(bound) = imports.checked_bound {
            if bound < 0 {
                return Err(Error::Bounds);
            }
            let accepted = Interval::new(-bound, bound)?;
            if bounds.lower() < accepted.lower() || bounds.upper() > accepted.upper() {
                for start in (0..node.lanes).step_by(128) {
                    let n = (node.lanes - start).min(128);
                    work = work
                        .checked_add(tensor::Program::checked_range_work(
                            n,
                            bounds,
                            accepted,
                            0,
                            &mut workspace,
                        )?)?
                        .checked_add(tensor::Program::all_valid_work(n, &mut workspace)?)?;
                    if sticky {
                        work = work.checked_add(WorkBudget::product(1, false)?)?;
                    }
                    sticky = true;
                }
                bounds = accepted;
            }
        }
        work = work.checked_add(
            tensor::Program::rms_norm_certified_work(
                1,
                width,
                bounds,
                coefficients,
                &mut workspace,
            )?
            .checked_mul(rows as u64)?,
        )?;
    }
    Ok((work, peak))
}

/// Quote native work and ALL pending native/control backing before Vec, Program
/// or Arc construction. Workspace is native-owned fixed metadata, not a recipe.
pub(super) fn prepare<'a>(
    graph: &CompactGraph<'_, 'a>,
    input: &PublishedView,
    width: usize,
    gamma: &'a SourceTensor,
) -> Result<PreparedTransition<'a>, Error> {
    use std::mem::size_of;
    let checked = graph.imports.checked_bound;
    let incoming = input_bounds(graph, input, 0, input.lanes)?;
    let accepted = match checked {
        Some(bound) if bound >= 0 => Interval::new(-bound, bound)?,
        Some(_) => return Err(Error::Bounds),
        None => incoming,
    };
    if checked.is_some() {
        graph.validity.check(graph)?;
    }
    let guarded = checked.is_some()
        && (incoming.lower() < accepted.lower() || incoming.upper() > accepted.upper());
    let tiles = if guarded {
        input.lanes.div_ceil(128)
    } else {
        0
    };
    let nodes = if tiles == 0 {
        0
    } else {
        tiles
            .checked_mul(6)
            .and_then(|n| n.checked_sub(usize::from(graph.validity.current().is_none())))
            .ok_or(Error::Bounds)?
    };
    let end = add_bytes(
        add_bytes(graph.next, nodes)?,
        (input.lanes / width).checked_mul(2).ok_or(Error::Bounds)?,
    )?;
    if end > MAX_NODES {
        return Err(Error::Bounds);
    }
    let storage = array_bytes::<GuardRun>(tiles)?;
    let gamma_bytes = add_bytes(
        size_of::<SourceTensor>(),
        add_bytes(
            array_bytes::<usize>(gamma.shape.capacity())?,
            array_bytes::<i64>(gamma.public_q18.as_ref().map_or(0, |v| v.capacity()))?,
        )?,
    )?;
    let ticket_slots = if graph.request_admission.is_some() {
        request_rms::slots(tiles)?
    } else {
        0
    };
    let controls = add_bytes(add_bytes(storage, control_slots()?)?, ticket_slots)?;
    let baseline = add_bytes(add_bytes(graph.baseline()?, gamma_bytes)?, controls)?;
    let admit = |extra| {
        graph.admit_total_with_source(
            add_bytes(graph.outside_owner_bytes, add_bytes(baseline, extra)?)?,
            gamma_bytes,
        )?;
        if let Some(context) = graph.request_admission {
            context.admit_phase(
                add_bytes(
                    add_bytes(controls, graph.contraction_control_bytes.get())?,
                    extra,
                )?,
                0,
            )?;
        }
        Ok::<(), Error>(())
    };
    admit(add_bytes(
        compact_norm::driver_slots()?,
        add_bytes(
            tensor::RmsWorkWorkspace::workspace_bytes(),
            work_frames(
                width,
                gamma.public_q18.as_ref().map_or(0, |v| v.len()),
                input.lanes,
            )?,
        )?,
    )?)?;
    // BEFORE fixed-workspace initialization, even when work will reject.
    let mut workspace = tensor::RmsWorkWorkspace::empty();
    let mut guard_work = WorkBudget::default();
    let mut retained = 0;
    let mut peak = 0;
    let mut last = None;
    for tile in 0..tiles {
        let n = (input.lanes - tile * 128).min(128);
        let bounds = input_bounds(graph, input, tile * 128, n)?;
        let (range_work, valid_work) = if let Some((old_n, old_bounds, range, valid)) = last {
            if old_n == n && old_bounds == bounds {
                (range, valid)
            } else {
                (
                    tensor::Program::checked_range_work(n, bounds, accepted, 0, &mut workspace)?,
                    tensor::Program::all_valid_work(n, &mut workspace)?,
                )
            }
        } else {
            (
                tensor::Program::checked_range_work(n, bounds, accepted, 0, &mut workspace)?,
                tensor::Program::all_valid_work(n, &mut workspace)?,
            )
        };
        last = Some((n, bounds, range_work, valid_work));
        guard_work = guard_work
            .checked_add(range_work)?
            .checked_add(valid_work)?;
        if tile > 0 || graph.validity.current().is_some() {
            guard_work = guard_work.checked_add(WorkBudget::product(1, false)?)?;
        }
        // Exact source-bound key, not bounds-only proof or Program-digest authority.
        let seen_range = previous_runs(graph).any(|run| {
            run.lanes == n && run.incoming == bounds && run.checked.output_bounds() == accepted
        }) || (0..tile).any(|prior| {
            (input.lanes - prior * 128).min(128) == n
                && input_bounds(graph, input, prior * 128, n).is_ok_and(|old| old == bounds)
        });
        let seen_valid = previous_runs(graph).any(|run| run.lanes == n)
            || (0..tile).any(|prior| (input.lanes - prior * 128).min(128) == n);
        for construction in [
            (!seen_range).then_some(tensor::Construction::CheckedRange { lanes: n }),
            (!seen_valid).then_some(tensor::Construction::AllValid { lanes: n }),
        ]
        .into_iter()
        .flatten()
        {
            let quote = native_construction_quote(construction)?;
            peak = peak.max(add_bytes(retained, quote.new_peak_bytes)?);
            retained = add_bytes(retained, quote.immutable_template_bytes)?;
        }
    }
    // Native guard validation/work always wins over any later RMS recipe error.
    let coefficients = gamma.public_q18.as_ref().ok_or(Error::Binding)?;
    let bounds = if guarded { accepted } else { incoming };
    let energy = guarded_energy(graph, input, 0, if guarded { checked } else { None })?;
    if !guarded && input_bounds(graph, input, 0, width)? != bounds {
        return Err(Error::Binding);
    }
    for row in 1..input.lanes / width {
        if !guarded && input_bounds(graph, input, row * width, width)? != bounds {
            // No common original RMS template for these row domains.
            return Err(Error::Binding);
        }
        if guarded_energy(graph, input, row, if guarded { checked } else { None })? != energy {
            // Current compact RMS publication has one row template. Never widen
            // heterogeneous proof obligations into a different original factory.
            return Err(Error::Binding);
        }
    }
    let row_work = if let Some(energy) = energy {
        tensor::Program::rms_norm_with_input_norm_work(
            1,
            width,
            bounds,
            coefficients,
            energy,
            &mut workspace,
        )?
    } else {
        tensor::Program::rms_norm_certified_work(1, width, bounds, coefficients, &mut workspace)?
    };
    let work = guard_work.checked_add(row_work.checked_mul((input.lanes / width) as u64)?)?;
    let aggregate = graph.work.checked_add(work)?;
    aggregate.limits()?;
    if let Some(owner) = &graph.prebound_inputs {
        owner.work.checked_add(aggregate)?.limits()?;
    }
    let construction = if energy.is_some() {
        tensor::Construction::RmsNormWithInputNorm {
            rows: 1,
            width,
            gamma_len: coefficients.len(),
        }
    } else {
        tensor::Construction::RmsNormCertified {
            rows: 1,
            width,
            gamma_len: coefficients.len(),
        }
    };
    let quote = native_construction_quote(construction)?;
    // The fixed workspace remains in THIS callframe through guard capture.
    // RMS construction is later, after this frame returns to the shared driver.
    let guard_peak = add_bytes(
        peak,
        add_bytes(
            tensor::RmsWorkWorkspace::workspace_bytes(),
            compact_norm::driver_slots()?,
        )?,
    )?;
    let rms_peak = add_bytes(
        add_bytes(retained, quote.new_peak_bytes)?,
        compact_norm::factory_slots()?,
    )?;
    admit(guard_peak.max(rms_peak))?;
    // Original guard/work validation above still precedes RMS binding. Capture
    // and debit ALL guards, scalar sticky uses and RMS row invocations now,
    // before GuardRun allocation or the FIRST native constructor below.
    let mut tickets = request_rms::prepare(
        graph,
        input,
        request_rms::RequestNorm {
            width,
            gamma,
            accepted,
            bounds,
            energy,
            tiles,
            controls,
        },
    )?;
    let rms_part = tickets.as_mut().and_then(|tickets| tickets.rms.take());
    if checked.is_none() {
        return Ok((None, bounds, energy, work, rms_part));
    }
    let mut rows = CheckedRows {
        original: graph.imports,
        input: input.clone(),
        width,
        accepted: bounds,
        first: graph.next,
        incoming: graph.validity.current(),
        outgoing: graph.validity.current(),
        runs: Vec::new(),
        work: guard_work,
    };
    if tiles != 0 {
        #[cfg(test)]
        STORAGE.with(|n| n.set(n.get() + 1));
        rows.runs
            .try_reserve_exact(tiles)
            .map_err(|_| Error::Allocation)?;
        if rows.runs.capacity() != tiles {
            return Err(Error::Allocation);
        }
    }
    for tile in 0..tiles {
        let n = (input.lanes - tile * 128).min(128);
        let incoming = input_bounds(graph, input, tile * 128, n)?;
        if let Some(run) = rows
            .runs
            .last_mut()
            .filter(|run| run.lanes == n && run.incoming == incoming)
        {
            run.tiles += 1;
            continue;
        }
        let old = rows.runs.iter().chain(previous_runs(graph)).find(|run| {
            run.lanes == n && run.incoming == incoming && run.checked.output_bounds() == accepted
        });
        let (program, owns_checked) = if let Some(old) = old {
            (old.checked.clone(), false)
        } else {
            #[cfg(test)]
            FACTORIES.with(|n| n.set(n.get() + 1));
            let program = if let Some(tickets) = tickets.as_mut() {
                let (program, certificate) = tickets.guards[tile]
                    .range
                    .take()
                    .ok_or(Error::Binding)?
                    .build()?
                    .into_parts();
                if certificate != tensor::ProducedCertificate::None {
                    return Err(Error::Binding);
                }
                program
            } else {
                tensor::Program::checked_range(n, incoming, accepted, 0)?
            };
            (Arc::new(program), true)
        };
        let old = rows
            .runs
            .iter()
            .chain(previous_runs(graph))
            .find(|run| run.lanes == n);
        let (all_valid, owns_valid) = if let Some(old) = old {
            (old.all_valid.clone(), false)
        } else {
            #[cfg(test)]
            FACTORIES.with(|n| n.set(n.get() + 1));
            let program = if let Some(tickets) = tickets.as_mut() {
                let (program, certificate) = tickets.guards[tile]
                    .valid
                    .take()
                    .ok_or(Error::Binding)?
                    .build()?
                    .into_parts();
                if certificate != tensor::ProducedCertificate::None {
                    return Err(Error::Binding);
                }
                program
            } else {
                tensor::Program::all_valid(n)?
            };
            (Arc::new(program), true)
        };
        rows.runs.push(GuardRun {
            first_tile: tile,
            tiles: 1,
            lanes: n,
            incoming,
            checked: program,
            owns_checked,
            all_valid,
            owns_valid,
        });
    }
    if tiles != 0 {
        rows.outgoing = Some(graph.next + rows.nodes() - 1);
    }
    rows.validate(
        graph.imports,
        input,
        graph.next,
        width,
        graph.validity.current(),
    )?;
    if let Some(context) = graph.request_admission {
        let retained = rows
            .programs()
            .try_fold(rows.metadata_bytes()?, |bytes, program| {
                add_bytes(
                    bytes,
                    add_bytes(
                        program.resources()?.immutable_template_bytes,
                        2 * size_of::<usize>(),
                    )?,
                )
            })?;
        context.retain(retained)?;
    }
    Ok((Some(rows), bounds, energy, work, rms_part))
}

#[cfg(test)]
thread_local! {
    pub(super) static FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static STORAGE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
#[path = "compact_validity_tests.rs"]
mod tests;
