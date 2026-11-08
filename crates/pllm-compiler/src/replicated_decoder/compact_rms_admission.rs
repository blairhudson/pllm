//! Request-only capture of EXISTING checked-row/RMS native factory tickets.
//! No GENERAL premise is invented: None selects the ordinary certified factory.
use super::*;
use super::ControllerAccounting;
use std::mem::size_of;

/// Existing compiler-metadata debit, on the authentic request's original ledger.
/// Called only after the RMS frame (including lookup-proof slots) is admitted.
/// Cover complete original owner extents and source-key comparison bytes before
/// the new walks; this is not native work or a new source/proof issuer.
pub(super) fn prepay_lookup_proof<G: ControllerAccounting>(graph: &CompactGraph<'_, '_, G>, key_bytes: usize) -> Result<(), Error> {
    let context = graph.request_admission.ok_or(Error::Binding)?;
    let visits = [
        graph.sources.len(), graph.imports.nodes.len(), graph.imports.original_inputs.len(), graph.bound_imports.len(),
        graph.ops.len(), key_bytes, 1,
    ].into_iter().try_fold(0usize, add_bytes)?;
    context.prepay_metadata(u64::try_from(visits).map_err(|_| Error::Bounds)?)
}

pub(super) struct GuardTicket<'a> {
    pub(super) range: tensor::ResidentNativeSchedule<'a>,
    pub(super) valid: tensor::ResidentNativeSchedule<'a>,
    pub(super) range_output: Option<tensor::FactoryOutput>,
    pub(super) range_tail: Option<tensor::FactoryOutput>,
    pub(super) valid_output: Option<tensor::FactoryOutput>,
    pub(super) valid_tail: Option<tensor::FactoryOutput>,
    pub(super) checked: Option<Arc<tensor::Program>>,
    pub(super) all_valid: Option<Arc<tensor::Program>>,
}
pub(crate) struct Tickets<'a> {
    pub(super) guards: tensor::ExactStorage<Option<GuardTicket<'a>>>,
    pub(super) rms: tensor::ResidentNativeSchedule<'a>,
    pub(super) rms_output: Option<tensor::FactoryOutput>,
    pub(super) rms_tail: Option<tensor::FactoryOutput>,
    pub(super) rms_program: Option<Arc<tensor::Program>>,
    pub(super) rows: Option<OwnedCheckedRows>,
}
impl<'a> Tickets<'a> {
    pub(in crate::replicated_decoder::program::compact_lowering) fn ticket_extent(&self) -> usize { self.guards.extent() }
    fn empty() -> Self {
        Self { guards: tensor::ExactStorage::empty(), rms: tensor::WorkPreflightWorkspace::resident_schedule(),
            rms_output: None, rms_tail: None, rms_program: None, rows: None }
    }
    pub(crate) fn backing_bytes(&self) -> Result<usize, Error> {
        add_bytes(array_bytes::<Option<GuardTicket<'a>>>(self.guards.extent())?,
            self.rows.as_ref().map_or(Ok(0), |rows| rows.metadata_bytes())?)
    }
    pub(crate) fn rows(&self) -> Option<&OwnedCheckedRows> { self.rows.as_ref() }
    pub(crate) fn programs(&self) -> impl Iterator<Item = &tensor::Program> {
        let guards = match &self.guards { tensor::ExactStorage::Ready(v) => &v[..], _ => &[] };
        guards.iter().filter_map(Option::as_ref).flat_map(|g| [
            g.checked.as_deref(), g.all_valid.as_deref(),
            g.range_output.as_ref().map(tensor::FactoryOutput::program),
            g.range_tail.as_ref().map(tensor::FactoryOutput::program),
            g.valid_output.as_ref().map(tensor::FactoryOutput::program),
            g.valid_tail.as_ref().map(tensor::FactoryOutput::program),
        ].into_iter().flatten()).chain([
            self.rms_program.as_deref(), self.rms_output.as_ref().map(tensor::FactoryOutput::program),
            self.rms_tail.as_ref().map(tensor::FactoryOutput::program),
        ].into_iter().flatten())
    }
}
impl GuardTicket<'_> {
    fn empty() -> Self {
        Self { range: tensor::WorkPreflightWorkspace::resident_schedule(), valid: tensor::WorkPreflightWorkspace::resident_schedule(),
            range_output: None, range_tail: None, valid_output: None, valid_tail: None,
            checked: None, all_valid: None }
    }
}
/// Compiler owners only. Native tables/proofs/Builder are separately priced.
pub(in crate::replicated_decoder::program::compact_lowering) fn compiler_events(tiles: usize, width: usize) -> Result<tensor::LogicalEvents, Error> {
    let t = u64::try_from(tiles).map_err(|_| Error::Bounds)?;
    let w = u64::try_from(width).map_err(|_| Error::Bounds)?;
    let parts = t.checked_mul(2).and_then(|n| n.checked_add(1)).ok_or(Error::Bounds)?;
    let schedules = crate::replicated_decoder::program::request_origin::compact_storage::resident_schedule_events()
        .checked_mul(parts)?;
    // Exact ticket+run None/prefix writes(4), ticket assembly/overwrite(2),
    // run/live overwrite(2); coalescing's one tiles write is exclusive/smaller.
    // Each of two native outputs has six transfer and four Arc-control writes.
    let tile = tensor::LogicalEvents {
        writes: 4 + 2 + 2 + 2 * 6 + 2 * 4,
        // Six local shape/immutable-header/metadata 64-event blocks; NOT full
        // resources(), whose independently priced J calls are added in prepare.
        validation: 6 * 64,
        transitions: 2 + 1 + 1 + 4 + 2,
        // Ticket/run dispatch(2), four Arc refs(4), two Arc-control releases(2).
        // Program destruction belongs to native, not a scalar-drop loop here.
        cleanup_dispatches: 2 + 4 + 2,
        ..tensor::LogicalEvents::default()
    }.checked_mul(t)?;
    let fixed = tensor::LogicalEvents {
        // RMS transfer/Arc(6+4), row refs(4), root writes(4).
        writes: 6 + 4 + 4 + 4,
        validation: w.checked_mul(3).and_then(|n| n.checked_add(4 * 64)).ok_or(Error::Bounds)?,
        transitions: 2 + 2 + 2,
        // 2/tile Program Arcs + RMS Arc, ticket backing, run backing, run Arc.
        // A null attempt counts. Arc OOM abort is NOT recoverable.
        allocation_attempts: t.checked_mul(2).and_then(|n| n.checked_add(4)).ok_or(Error::Bounds)?,
        // RMS refs/control(3), run refs/control(3), two containers' owner/release
        // pairs(4), root/origin refs(3). Retained backing is charged only once.
        cleanup_dispatches: 3 + 3 + 4 + 3,
    };
    schedules.checked_add(tile)?.checked_add(fixed)
}

pub(super) fn slots(tiles: usize) -> Result<usize, Error> {
    let r = tensor::WorkPreflightWorkspace::resources();
    let mut total = add_bytes(array_bytes::<Option<GuardTicket<'_>>>(tiles)?, pending_walk_control_bytes()?)?;
    for n in [
        // Tickets itself is embedded in the graph, not a second inline owner.
        2 * size_of::<Result<bool, Error>>(),
        2 * size_of::<Option<tensor::ResidentFactorySource<'_>>>(),
        2 * size_of::<Result<tensor::ResidentFactorySource<'_>, tensor::WorkPreflightError>>(),
        r.summary_bytes,
        r.ledger_bytes,
        r.build_result_bytes,
    ] {
        total = add_bytes(total, n)?;
    }
    Ok(total)
}

pub(super) struct RequestNorm<'a> {
    pub(super) width: usize,
    pub(super) gamma: &'a SourceTensor,
    pub(super) accepted: Interval,
    pub(super) bounds: Interval,
    pub(super) energy: Option<u128>,
    pub(super) tiles: usize,
    pub(super) controls: usize,
}
pub(super) fn prepare<'a, G: ControllerAccounting>(
    graph: &CompactGraph<'_, 'a, G>,
    input: &PublishedView,
    request: RequestNorm<'a>,
) -> Result<bool, Error> {
    let RequestNorm {
        width,
        gamma,
        accepted,
        bounds,
        energy,
        tiles,
        controls,
    } = request;
    let Some(context) = graph.request_admission else {
        return Ok(false);
    };
    context.check_final_work(graph.work)?;
    let r = tensor::WorkPreflightWorkspace::resources();
    let live = add_bytes(
        add_bytes(controls, graph.contraction_control_bytes.get())?,
        add_bytes(
            tensor::RmsWorkWorkspace::workspace_bytes(),
            r.workspace_bytes,
        )?,
    )?;
    context.admit_phase(live, 0)?;
    let visits = tiles
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(tensor::MAX_NODES * 36))
        .ok_or(Error::Bounds)?;
    context.prepay_metadata(u64::try_from(visits).map_err(|_| Error::Bounds)?)?;
    // New pending census pays H/B on EACH baseline invocation instead.
    let rms_construction = if energy.is_some() {
        tensor::Construction::RmsNormWithInputNorm { rows: 1, width, gamma_len: width }
    } else { tensor::Construction::RmsNormCertified { rows: 1, width, gamma_len: width } };
    // RMS and every checked/all-valid output each have a resources postcheck.
    // Sequential to the old proof debit; reuse prepays the skipped calls too.
    let parts = u64::try_from(tiles).map_err(|_| Error::Bounds)?.checked_mul(2)
        .and_then(|n| n.checked_add(1)).ok_or(Error::Bounds)?;
    let mut events = rms_construction.logical_events()?.checked_add(tensor::LogicalEvents {
        validation: tensor::Program::full_resource_validation_visits()?.checked_mul(parts).ok_or(Error::Bounds)?,
        ..tensor::LogicalEvents::default()
    })?;
    for tile in 0..tiles {
        let lanes = (input.lanes - tile * 128).min(128);
        for factory in [tensor::Construction::CheckedRange { lanes }, tensor::Construction::AllValid { lanes }] {
            events = events.checked_add(factory.logical_events()?)?;
        }
    }
    events = events.checked_add(compiler_events(tiles, width)?)?;
    let events = events.total()?;
    context.prepay_primitive(events)?;
    // Ticket storage and BOTH workspaces were admitted before either allocation.
    let mut pending = graph.rms_pending.borrow_mut();
    if pending.is_some() { return Err(Error::Binding); }
    *pending = Some(Tickets::empty());
    let pending = pending.as_mut().ok_or(Error::Binding)?;
    pending.rows = Some(OwnedCheckedRows {
            input: input.clone(), width, accepted: bounds, first: graph.next,
            incoming: graph.validity.current(), outgoing: graph.validity.current(),
            runs: GuardRuns::new(), work: WorkBudget::default(),
    });
    pending.rows.as_mut().ok_or(Error::Binding)?.runs.allocate(tiles)?;
    pending.guards.acquire_slots(tiles)?;
    let mut workspace = tensor::WorkPreflightWorkspace::empty();
    for tile in 0..tiles {
        let lanes = (input.lanes - tile * 128).min(128);
        let incoming = input_bounds(graph, input, tile * 128, lanes)?;
        let slot = pending.guards.ready_mut()?.get_mut(tile).ok_or(Error::Binding)?;
        *slot = Some(GuardTicket::empty());
        let guard = slot.as_mut().ok_or(Error::Binding)?;
        let mut range = Some(tensor::Construction::CheckedRange { lanes }
            .resident_parameters(tensor::FactoryParameters::CheckedRange {
                incoming,
                accepted,
                fallback: 0,
            }));
        guard.range.install(&mut range, None, 1).map_err(|_| Error::Bounds)?;
        guard.range.bind().and_then(|_| guard.range.preflight(&mut workspace)).map_err(|_| Error::Bounds)?;
        context.admit_resident(&mut guard.range)?;
        let mut valid = Some(tensor::Construction::AllValid { lanes }
            .resident_parameters(tensor::FactoryParameters::AllValid));
        guard.valid.install(&mut valid, None, 1).map_err(|_| Error::Bounds)?;
        guard.valid.bind().and_then(|_| guard.valid.preflight(&mut workspace)).map_err(|_| Error::Bounds)?;
        context.admit_resident(&mut guard.valid)?;
    }
    let coefficients = gamma.public_q18.as_deref().ok_or(Error::Binding)?;
    let (construction, parameters) = if let Some(squared_norm) = energy {
        (
            tensor::Construction::RmsNormWithInputNorm {
                rows: 1,
                width,
                gamma_len: coefficients.len(),
            },
            tensor::FactoryParameters::RmsNormWithInputNorm {
                input: bounds,
                gamma: coefficients,
                squared_norm,
            },
        )
    } else {
        (
            tensor::Construction::RmsNormCertified {
                rows: 1,
                width,
                gamma_len: coefficients.len(),
            },
            tensor::FactoryParameters::RmsNormCertified {
                input: bounds,
                gamma: coefficients,
            },
        )
    };
    let rows = u64::try_from(input.lanes / width).map_err(|_| Error::Bounds)?;
    let mut rms = Some(construction.resident_parameters(parameters));
    pending.rms.install(&mut rms, None, rows).map_err(|_| Error::Bounds)?;
    pending.rms.bind().and_then(|_| pending.rms.preflight(&mut workspace)).map_err(|_| Error::Bounds)?;
    context.admit_resident(&mut pending.rms)?;
    let products = if tiles == 0 {
        0
    } else {
        tiles - usize::from(graph.validity.current().is_none())
    };
    context.admit_products(products)?;
    Ok(true)
}

/// Borrow-check before taking the original output. Arc allocation may abort;
/// there is no claimed recoverable gap between Program and its installed Arc.
pub(super) fn build_guard(guard: &mut GuardTicket<'_>, valid: bool) -> Result<Arc<tensor::Program>, Error> {
    let (schedule, full, tail, arc) = if valid {
        (&mut guard.valid, &mut guard.valid_output, &mut guard.valid_tail, &mut guard.all_valid)
    } else { (&mut guard.range, &mut guard.range_output, &mut guard.range_tail, &mut guard.checked) };
    schedule.build_into(full, tail).map_err(|_| Error::Bounds)?;
    let output = full.as_ref().ok_or(Error::Binding)?;
    if output.certificate() != tensor::ProducedCertificate::None || tail.is_some() { return Err(Error::Binding); }
    let (program, _) = full.take().ok_or(Error::Binding)?.into_parts();
    *arc = Some(Arc::new(program));
    Ok(arc.as_ref().ok_or(Error::Binding)?.clone())
}

pub(crate) fn build_rms(pending: &mut Tickets<'_>, width: usize, quote: tensor::ConstructionQuote)
    -> Result<(Arc<tensor::Program>, u128), Error> {
    pending.rms.build_into(&mut pending.rms_output, &mut pending.rms_tail).map_err(|_| Error::Bounds)?;
    let output = pending.rms_output.as_ref().ok_or(Error::Binding)?;
    let tensor::ProducedCertificate::GeneralOutputRowSquared(energy) = output.certificate() else { return Err(Error::Binding); };
    let program = output.program();
    let resources = program.resources()?;
    if pending.rms_tail.is_some() || program.output_lanes() != width || program.validity_lanes().is_some()
        || resources.immutable_template_bytes > quote.immutable_template_bytes
        || resources.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES { return Err(Error::Bounds); }
    let (program, _) = pending.rms_output.take().ok_or(Error::Binding)?.into_parts();
    pending.rms_program = Some(Arc::new(program));
    Ok((pending.rms_program.as_ref().ok_or(Error::Binding)?.clone(), energy))
}
