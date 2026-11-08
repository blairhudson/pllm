//! Closed request work/factory bridge for ORIGINAL ArgmaxNumeric/argmax_tree.
//! Parent count observer/routing hooks are an UNAPPLIED E proposal. No alternate
//! selection arithmetic, per-lane table, native source issuer or phase retirement.
use super::*;
use crate::replicated_decoder::program::request_origin::compact_storage::ControllerAccounting;
use std::mem::{size_of, size_of_val};

struct Requirement {
    lanes: usize,
    bounds: Interval,
    uses: u64,
    work: WorkBudget,
    admitted: Option<tensor::AdmittedFactory<'static>>,
}
pub(super) struct RequestCount {
    requirements: Vec<Requirement>,
    native: WorkBudget,
    products: WorkBudget,
}
impl RequestCount {
    /// Called ONLY by original ArgmaxNumeric::negative after its actual native
    /// work query. Counts repeated invocations even if a Program will be reused.
    pub(super) fn negative(
        &mut self,
        lanes: usize,
        bounds: Interval,
        work: WorkBudget,
    ) -> Result<(), Error> {
        if lanes == 0 || lanes > 128 {
            return Err(Error::Shape);
        }
        let next = self.native.checked_add(work)?;
        if let Some(record) = self
            .requirements
            .iter_mut()
            .find(|r| r.lanes == lanes && r.bounds == bounds)
        {
            if record.work != work {
                return Err(Error::Binding);
            }
            record.uses = record.uses.checked_add(1).ok_or(Error::Bounds)?;
        } else {
            if self.requirements.len() == self.requirements.capacity() {
                return Err(Error::Bounds);
            }
            self.requirements.push(Requirement {
                lanes,
                bounds,
                uses: 1,
                work,
                admitted: None,
            });
        }
        self.native = next;
        Ok(())
    }
    /// Called by original SelectSink::product with the work it actually emits.
    /// No reconstructed 'two selects' formula or subtraction from opaque work.
    pub(super) fn product(&mut self, work: WorkBudget) -> Result<(), Error> {
        self.products = self.products.checked_add(work)?;
        Ok(())
    }
}
fn arc_header() -> Result<usize, Error> {
    let a = std::mem::align_of::<tensor::Program>();
    Ok(add_bytes(2 * size_of::<usize>(), a - 1)? / a * a)
}
fn controls(attempt: usize) -> Result<usize, Error> {
    let r = tensor::WorkPreflightWorkspace::resources();
    let mut bytes = 0;
    macro_rules! pair {
        ($n:expr) => {
            bytes = add_bytes(bytes, ($n).checked_mul(2).ok_or(Error::Bounds)?)?;
        };
    }
    pair!(r.workspace_bytes);
    pair!(r.factory_bytes);
    pair!(r.ticket_bytes);
    pair!(r.admitted_schedule_bytes);
    pair!(r.admitted_part_bytes);
    pair!(r.summary_bytes);
    pair!(r.ledger_bytes);
    pair!(r.ticket_result_bytes);
    pair!(r.build_result_bytes);
    pair!(size_of::<RequestCount>());
    pair!(size_of::<Requirement>());
    pair!(size_of::<Option<tensor::AdmittedFactory<'static>>>());
    pair!(size_of::<Result<CompactArgmax, Error>>());
    pair!(size_of::<CompactArgmax>());
    pair!(size_of::<tensor::FactoryOutput>());
    pair!(size_of::<Result<tensor::FactoryOutput, Error>>());
    pair!(size_of::<(tensor::Program, tensor::ProducedCertificate)>());
    pair!(size_of::<NegativeProgram>());
    pair!(size_of::<Vec<NegativeProgram>>());
    pair!(size_of::<Vec<Requirement>>());
    pair!(size_of::<tensor::ConstructionQuote>());
    pair!(size_of::<tensor::Construction>());
    pair!(size_of::<tensor::FactoryParameters<'static>>());
    pair!(size_of::<WorkBudget>());
    pair!(size_of::<(usize, usize)>());
    pair!(size_of::<Result<(usize, usize), Error>>());
    bytes = add_bytes(bytes, attempt.checked_mul(4).ok_or(Error::Bounds)?)?;
    Ok(bytes)
}
fn peak<G: ControllerAccounting>(
    graph: &CompactGraph<'_, '_, G>,
    lanes: usize,
    attempt: usize,
) -> Result<(ArgmaxGeometry, usize), Error> {
    let mut baseline = add_bytes(graph.baseline()?, graph.outside_owner_bytes)?;
    // Preserve earlier request Argmax Arc headers concurrently with this phase;
    // never use a currently retained ledger value as spare allocation credit.
    for op in &graph.ops {
        if let PublishedKind::Argmax(old) = &op.kind {
            baseline = add_bytes(
                baseline,
                arc_header()?
                    .checked_mul(old.programs.iter().filter(|p| p.owns).count())
                    .ok_or(Error::Bounds)?,
            )?;
        }
    }
    let (geometry, old) = CompactArgmax::quote(lanes, baseline)?;
    let mut peak = add_bytes(old, array_bytes::<Requirement>(geometry.kernels)?)?;
    peak = add_bytes(peak, controls(attempt)?)?;
    // Existing quote already covers native Program immutable payload. New Arc
    // control headers receive no alias/free credit, even when actual reuse wins.
    peak = add_bytes(
        peak,
        arc_header()?
            .checked_mul(geometry.kernels)
            .ok_or(Error::Bounds)?,
    )?;
    peak = add_bytes(peak, graph.contraction_control_bytes.get())?;
    Ok((geometry, peak))
}
fn same_input(a: &PublishedView, b: &PublishedView) -> bool {
    Arc::ptr_eq(&a.origin, &b.origin)
        && a.slot == b.slot
        && a.mapping == b.mapping
        && a.order == b.order
        && a.start == b.start
        && a.lanes == b.lanes
        && a.rank == b.rank
        && a.logical == b.logical
        && a.bounds == b.bounds
}
fn prior<'a, G: ControllerAccounting>(
    graph: &'a CompactGraph<'_, '_, G>,
    lanes: usize,
    bounds: Interval,
) -> Option<&'a NegativeProgram> {
    graph.ops.iter().find_map(|op| match &op.kind {
        PublishedKind::Argmax(p) => p
            .programs
            .iter()
            .find(|p| p.lanes == lanes && p.bounds == bounds),
        _ => None,
    })
}
fn validate(
    program: &tensor::Program,
    record: &Requirement,
    quote: tensor::ConstructionQuote,
) -> Result<(), Error> {
    let r = program.resources()?;
    if program.lanes() != record.lanes
        || program.output_lanes() != record.lanes
        || program.output_bounds() != Interval::new(0, 1)?
        || program.validity_lanes().is_some()
        || program.work_budget() != record.work
        || r.immutable_template_bytes > quote.immutable_template_bytes
        || r.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES
    {
        return Err(Error::Binding);
    }
    Ok(())
}
/// Actual request-selected semantic input only. Expanded/PerToken source policy
/// stays with original graph; no accepted-token/feedback authority is issued.
pub(super) fn prepare_request_argmax<G: ControllerAccounting>(
    graph: &CompactGraph<'_, '_, G>,
    input: &PublishedView,
    first: usize,
) -> Result<CompactArgmax, Error> {
    let context = graph.request_admission.ok_or(Error::Binding)?;
    let mut attempt = context.attempt();
    context.check_open()?;
    context.check_final_work(graph.work)?;
    graph.validate_view(input)?;
    let op = graph
        .frontier
        .operations()
        .nth(graph.cursor)
        .ok_or(Error::Binding)?
        .1;
    let (bound, other) = graph.frontier.inputs(op);
    if op.operator != Op::GreedyTokenSelection
        || op.output_shape != [1]
        || other.is_some()
        || !same_input(input, bound.ok_or(Error::Binding)?)
        || first != graph.next
        || graph.imports.selection != SelectionPolicy::Expanded
        || graph.imports.lookup == LookupPolicy::Trie
    {
        return Err(Error::Binding);
    }
    let (geometry, peak) = peak(graph, input.lanes, size_of_val(&attempt))?;
    graph.admit_total(peak)?;
    context.admit_phase(peak, 0)?;
    let visits = geometry
        .nodes()?
        .checked_mul(add_bytes(
            add_bytes(
                graph.ops.len(),
                graph.previous_epoch.map_or(0, |e| e.recipe_count()),
            )?,
            1,
        )?)
        .and_then(|n| n.checked_mul(32))
        .ok_or(Error::Bounds)?;
    context.prepay_metadata(u64::try_from(visits).map_err(|_| Error::Bounds)?)?;
    let mut requirements = Vec::new();
    requirements
        .try_reserve_exact(geometry.kernels)
        .map_err(|_| Error::Allocation)?;
    if requirements.capacity() != geometry.kernels {
        return Err(Error::Allocation);
    }
    let mut count = RequestCount {
        requirements,
        native: WorkBudget::default(),
        products: WorkBudget::default(),
    };
    let pending = {
        let mut sink = ArgmaxNumeric::new(
            input,
            first,
            geometry,
            NegativePrograms::RequestCount(&mut count),
            None,
        )?;
        let output = argmax_tree(&mut sink, input.lanes)?;
        if output != geometry.word(input, first, geometry.rounds, ArgmaxField::Indices, 0)?
            || sink.next.checked_sub(first) != Some(geometry.nodes()?)
        {
            return Err(Error::Binding);
        }
        sink.work
    };
    if pending != count.native.checked_add(count.products)?
        || count.requirements.len() != geometry.kernels
    {
        return Err(Error::Binding);
    }
    graph.admit_pending_work(pending)?;
    let mut workspace = tensor::WorkPreflightWorkspace::empty();
    // ALL native invocation tickets admitted before first factory, including
    // every repeated use of a native Arc; no ticket reuse/debit/refund loop.
    for record in &mut count.requirements {
        let bound = tensor::Construction::Negative {
            lanes: record.lanes,
        }
        .bind(tensor::FactoryParameters::Negative {
            input: record.bounds,
        })
        .map_err(|_| Error::Bounds)?;
        let preflight = bound
            .preflight(record.uses, &mut workspace)
            .map_err(|_| Error::Bounds)?;
        let (full, tail) = context.admit_work(preflight)?.into_parts();
        if tail.is_some() {
            return Err(Error::Binding);
        }
        record.admitted = Some(full);
    }
    context.admit_protocol_work(count.products)?;
    context.check_final_work(graph.work.checked_add(pending)?)?;
    #[cfg(test)]
    if FAIL_AFTER_DEBIT.with(|f| f.replace(false)) {
        return Err(Error::Allocation);
    }
    let mut programs = Vec::new();
    programs
        .try_reserve_exact(geometry.kernels)
        .map_err(|_| Error::Allocation)?;
    if programs.capacity() != geometry.kernels {
        return Err(Error::Allocation);
    }
    for record in &mut count.requirements {
        let part = record.admitted.take().ok_or(Error::Binding)?;
        let quote = native_construction_quote(tensor::Construction::Negative {
            lanes: record.lanes,
        })?;
        let existing = prior(graph, record.lanes, record.bounds)
            .map(|p| &p.program)
            .or_else(|| {
                graph
                    .previous_epoch
                    .and_then(|epoch| epoch.previous_negative(record.lanes, record.bounds))
            });
        if let Some(existing) = existing {
            validate(existing, record, quote)?;
            // Construction permission discarded, persistent invocation debt held.
            let _permission = part;
            programs.push(NegativeProgram {
                lanes: record.lanes,
                bounds: record.bounds,
                program: Arc::clone(existing),
                owns: false,
            });
        } else {
            #[cfg(test)]
            NATIVE_FACTORIES.with(|n| n.set(n.get() + 1));
            let (program, certificate) = part.build()?.into_parts();
            if certificate != tensor::ProducedCertificate::None {
                return Err(Error::Binding);
            }
            validate(&program, record, quote)?;
            programs.push(NegativeProgram {
                lanes: record.lanes,
                bounds: record.bounds,
                program: Arc::new(program),
                owns: true,
            });
            #[cfg(test)]
            PANIC_AFTER_FACTORY
                .with(|f| assert!(!f.replace(false), "request argmax factory unwind"));
        }
    }
    // No Build/get direct factory. Replay SAME original math with the existing
    // typed Borrowed cache and check original output/counter; never debit twice.
    let (output, work) = {
        let mut sink = ArgmaxNumeric::new(
            input,
            first,
            geometry,
            NegativePrograms::Borrowed(&programs),
            None,
        )?;
        let out = argmax_tree(&mut sink, input.lanes)?;
        if sink.work != pending || sink.next.checked_sub(first) != Some(geometry.nodes()?) {
            return Err(Error::Binding);
        }
        (out, sink.work)
    };
    let result = CompactArgmax {
        first,
        geometry,
        programs,
        output,
        work,
        peak,
    };
    let headers = arc_header()?
        .checked_mul(result.programs.iter().filter(|p| p.owns).count())
        .ok_or(Error::Bounds)?;
    context.retain(add_bytes(result.baseline()?, headers)?)?;
    #[cfg(test)]
    if FAIL_PUBLICATION.with(|f| f.replace(false)) {
        return Err(Error::Allocation);
    }
    attempt.commit();
    Ok(result)
}
#[cfg(test)]
thread_local! {
    static NATIVE_FACTORIES:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};
    static FAIL_AFTER_DEBIT:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};
    static FAIL_PUBLICATION:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};
    static PANIC_AFTER_FACTORY:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};
}
#[cfg(test)]
#[path = "compact_request_argmax_tests.rs"]
mod tests;
