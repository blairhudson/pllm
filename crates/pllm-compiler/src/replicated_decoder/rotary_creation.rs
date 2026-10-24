//! Original semantic creation custody. No constructor accepts a published View,
//! Program digest, detached recipe, native template, or caller completion flag.
use super::*;
use crate::replicated_decoder::runtime::body::layer;

#[path = "rotary_handoff.rs"]
pub(in crate::replicated_decoder) mod handoff;

/// One request-owned creation roster, not a second paper-specific owner forest.
/// Its actual enum size is charged by declaration before roster allocation.
pub(in crate::replicated_decoder) enum CreatedOperator {
    Bias(Box<source_bias::Creation>),
    Produced(Box<Produced>),
    Rotary(Box<Creation>),
}
pub(super) fn roster_owner_bytes() -> usize {
    size_of::<source_bias::Creation>()
        .max(size_of::<Produced>())
        .max(size_of::<Creation>())
}

struct Binding {
    origin: Arc<OwnedCharged<Origin>>,
    plan: Arc<OriginalSourcePlan>,
    epoch: usize,
    operation: usize,
    sticky: Option<usize>,
    records: usize,
    registry: source_registry::Instance,
}
impl Binding {
    fn capture(active: &Active, op: &pllm_models::ModelOperation) -> Result<Self, Error> {
        let operation = active
            .plan
            .plan()
            .prefill
            .operations
            .iter()
            .position(|v| std::ptr::eq(v, op))
            .ok_or(Error::Binding)?;
        if active.origin.payload().state.load(Ordering::Acquire) != OPEN {
            return Err(Error::Binding);
        }
        Ok(Self {
            origin: Arc::clone(&active.origin),
            plan: Arc::clone(&active.plan),
            epoch: active.epoch,
            operation,
            sticky: active.sticky_in,
            records: active.source_records,
            registry: active.registry_identity,
        })
    }
    fn same(&self, active: &Active) -> bool {
        Arc::ptr_eq(&self.origin, &active.origin)
            && Arc::ptr_eq(&self.plan, &active.plan)
            && self.epoch == active.epoch
            && self.sticky == active.sticky_in
    }
    fn check(&self, program: &Program, registry: &SourceRegistry<'_>) -> Result<(), Error> {
        let custody = program.request_custody.as_ref().ok_or(Error::Binding)?;
        if !Arc::ptr_eq(&self.origin, &custody.origin)
            || self.origin.payload().state.load(Ordering::Acquire) != COMPLETE
            || !Arc::ptr_eq(
                &self.plan,
                custody.epochs.get(self.epoch).ok_or(Error::Binding)?,
            )
            || self.sticky
                != custody
                    .fragments
                    .get(self.epoch)
                    .ok_or(Error::Binding)?
                    .sticky_in
            || self.registry != registry.instance()
            || self.records != std::ptr::from_ref(registry.source_records()) as usize
        {
            return Err(Error::Binding);
        }
        self.origin.payload().public.check_registry(registry)?;
        self.origin.payload().public.check(program)
    }
}

/// Produced only by successful ORIGINAL projection or reshape emission. The
/// output snapshot and parent slot refer to that same live OrderedGraph event.
pub(in crate::replicated_decoder) struct Produced {
    binding: Binding,
    parent: Option<usize>,
    words: Words,
    shape: Vec<usize>,
    first: usize,
    end: usize,
    kind: ProducedKind,
    // Drop snapshots before their original, continuously held reservation.
    _charge: OwnedExternalCharge,
}
enum ProducedKind {
    Projection {
        weight: usize,
        full: Arc<tensor::Program>,
        tail: Option<Arc<tensor::Program>>,
    },
    Reshape,
}

fn capture_charge(
    plan: &OriginalSourcePlan,
    controller: &mut Option<OwnedExternalCharge>,
    payload: usize,
    visits: usize,
) -> Result<OwnedExternalCharge, Error> {
    let payload = add(
        payload,
        OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
    )?;
    let controller = controller.as_mut().ok_or(Error::Binding)?;
    if !controller.is_for(plan.budget()) {
        return Err(Error::Binding);
    }
    controller
        .resize_payload(add(
            controller.payload_bytes(),
            add(payload, size_of::<OwnedExternalCharge>())?,
        )?)
        .map_err(|_| Error::Bounds)?;
    let charge = controller.split_owned(payload).map_err(|_| Error::Bounds)?;
    plan.prepay_compiler_metadata(&charge, u64::try_from(visits).map_err(|_| Error::Bounds)?)
        .map_err(|_| Error::Bounds)?;
    Ok(charge)
}

fn producer_slot(active: &Active, op: &pllm_models::ModelOperation) -> Option<usize> {
    if op.inputs.is_empty() {
        return None;
    }
    active.projections.iter().position(|entry| {
        matches!(entry,
        Some(CreatedOperator::Produced(p)) if p.binding.same(active)
            && p.binding.plan.plan().prefill.operations[p.binding.operation].id == op.inputs[0])
    })
}

/// Reservation only. It cannot mint an owner or validate arbitrary input words.
/// Paid at the original graph's entry, before ANY projection/native emission.
pub(in crate::replicated_decoder::program) fn preflight(
    builder: &mut Builder,
    graph: &DecoderGraph,
    op: &pllm_models::ModelOperation,
    shape: &[usize],
) -> Result<(), Error> {
    let Some(active) = &mut builder.request_emission else {
        return Ok(());
    };
    if !matches!(op.operator, Op::Linear | Op::Reshape) {
        return Ok(());
    }
    if op.operator == Op::Reshape && producer_slot(active, op).is_none() {
        return Ok(());
    }
    if active.produced_charge.is_some()
        || !std::ptr::eq(graph, &active.plan.plan().prefill)
        || active.projections.len() == active.projections.capacity()
    {
        return Err(Error::Binding);
    }
    let lanes = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d).ok_or(Error::Bounds))?;
    let payload = add(
        bytes::<Produced>(2)?,
        add(
            Words::span_pool_bytes(lanes).ok_or(Error::Bounds)?,
            bytes::<usize>(shape.len())?,
        )?,
    )?;
    let visits = add(
        lanes,
        add(
            active.projections.len(),
            graph.operations.len().checked_mul(2).ok_or(Error::Bounds)?,
        )?,
    )?;
    let charge = capture_charge(&active.plan, &mut active.creation_charge, payload, visits)?;
    builder
        .request_emission
        .as_mut()
        .ok_or(Error::Binding)?
        .produced_charge = Some(charge);
    Ok(())
}

/// Invoked only with graph's private post-creation event, before frontier finish.
pub(in crate::replicated_decoder::program) fn produced(
    builder: &mut Builder,
    event: graph::ProducedEvent<'_>,
) -> Result<(), Error> {
    let Some(active) = &mut builder.request_emission else {
        return Ok(());
    };
    let op = event.operation();
    if !matches!(op.operator, Op::Linear | Op::Reshape) {
        return Ok(());
    }
    if !std::ptr::eq(event.graph(), &active.plan.plan().prefill)
        || active.projections.len() == active.projections.capacity()
    {
        return Err(Error::Binding);
    }
    let parent = if op.operator == Op::Reshape {
        let Some(slot) = producer_slot(active, op) else {
            return Ok(());
        };
        let Some(CreatedOperator::Produced(p)) = &active.projections[slot] else {
            return Err(Error::Binding);
        };
        if p.words != event.input().ok_or(Error::Binding)?.words {
            return Err(Error::Binding);
        }
        Some(slot)
    } else {
        None
    };
    let output = event.output();
    let binding = Binding::capture(active, op)?;
    let kind = if op.operator == Op::Linear {
        let first = event.start();
        let Code::PublicProduct {
            input,
            weight,
            transposed: false,
            ..
        } = builder.nodes.get(first + 1).ok_or(Error::Binding)?.code
        else {
            return Err(Error::Binding);
        };
        let Code::TensorLoop {
            input: LoopInput::Slice(raw),
            program,
            tail,
        } = &builder.nodes.get(first + 2).ok_or(Error::Binding)?.code
        else {
            return Err(Error::Binding);
        };
        if input != first || *raw != first + 1 {
            return Err(Error::Binding);
        }
        ProducedKind::Projection {
            weight,
            full: Arc::clone(program),
            tail: tail.as_ref().map(Arc::clone),
        }
    } else {
        ProducedKind::Reshape
    };
    let produced = Produced {
        binding,
        parent,
        words: output.words.clone(),
        shape: output.shape.clone(),
        first: event.start(),
        end: builder.nodes.len(),
        kind,
        _charge: builder
            .request_emission
            .as_mut()
            .ok_or(Error::Binding)?
            .produced_charge
            .take()
            .ok_or(Error::Binding)?,
    };
    builder
        .request_emission
        .as_mut()
        .ok_or(Error::Binding)?
        .projections
        .push(Some(CreatedOperator::Produced(Box::new(produced))));
    Ok(())
}

pub(super) struct Pending {
    binding: Binding,
    producer: usize,
    shape: [usize; 4],
    position: usize,
    sequence: usize,
    theta: f32,
    first: usize,
    input: Words,
    rows: Vec<Row>,
    work: WorkBudget,
    projection: ProjectionPrograms,
    quote: CreationQuote,
    charge: OwnedExternalCharge,
}
pub(super) enum Emission {
    OriginalOnly { operation: usize },
    Capture(Box<Pending>),
}

// Eligibility reads semantic declarations, never the presence/absence of a
// creation record. Losing an eligible record MUST NOT turn into fallback.
fn capture_eligible(graph: &DecoderGraph, op: &pllm_models::ModelOperation) -> Result<bool, Error> {
    let input = op.inputs.first().ok_or(Error::Binding)?;
    let parent = graph
        .operations
        .iter()
        .find(|p| &p.id == input)
        .ok_or(Error::Binding)?;
    if parent.operator != Op::Reshape {
        return Ok(false);
    }
    let input = parent.inputs.first().ok_or(Error::Binding)?;
    let producer = graph
        .operations
        .iter()
        .find(|p| &p.id == input)
        .ok_or(Error::Binding)?;
    Ok(producer.operator == Op::Linear)
}
#[cfg(test)]
thread_local! {static DROP_NEXT_PRODUCER:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};}
#[cfg(test)]
pub(in crate::replicated_decoder) fn drop_next_producer_for_test() {
    DROP_NEXT_PRODUCER.set(true)
}
struct ProjectionPrograms {
    full: Arc<tensor::Program>,
    tail: Option<Arc<tensor::Program>>,
    raw_bounds: Interval,
}
#[derive(Clone, Copy, Debug)]
pub(in crate::replicated_decoder) struct CreationQuote {
    pub control_bytes: usize,
    pub row_handles_bytes: usize,
    pub public_coefficient_bytes: usize,
    pub mapping_bytes: usize,
    pub work_preflight_bytes: usize,
    pub native_construction_bytes: usize,
    pub phase_bytes: usize,
}
pub(in crate::replicated_decoder) struct Row {
    pub input: usize,
    pub output: usize,
    pub bounds: Interval,
    pub program: Arc<tensor::Program>,
    cosine: Vec<i64>,
    sine: Vec<i64>,
    position: u32,
}
pub(in crate::replicated_decoder) struct Creation {
    binding: Binding,
    producer: usize,
    shape: [usize; 4],
    position: usize,
    sequence: usize,
    theta: f32,
    pub first: usize,
    pub end: usize,
    pub input: Words,
    pub rows: Vec<Row>,
    consumer: Option<Consumer>,
    armed: bool,
    projection: ProjectionPrograms,
    quote: CreationQuote,
    _charge: OwnedExternalCharge,
    // LAST, installed only when removed from original Program custody. Keeps
    // all source/plan/history/native charges alive until every snapshot erases.
    execution: Option<Arc<Program>>,
}
struct Consumer {
    operation: usize,
    first: usize,
    end: usize,
    words: Words,
    shape: Vec<usize>,
    _charge: OwnedExternalCharge,
}
impl Drop for Creation {
    fn drop(&mut self) {
        if self.armed {
            self.binding
                .origin
                .payload()
                .state
                .store(BURNED, Ordering::Release)
        }
    }
}

/// Called from original shared rotary driver AFTER its original attribute
/// guards, BEFORE coefficient generation, gathers, native factory or row arrays.
pub(in crate::replicated_decoder::program) fn begin(
    builder: &mut Builder,
    event: graph::RotaryEvent<'_>,
    theta: f32,
) -> Result<(), Error> {
    let Some(active) = &mut builder.request_emission else {
        return Ok(());
    };
    let op = event.operation();
    if active.rotary.is_some() || !std::ptr::eq(event.graph(), &active.plan.plan().prefill) {
        return Err(Error::Binding);
    }
    if !capture_eligible(event.graph(), op)? {
        let binding = Binding::capture(active, op)?;
        active.rotary = Some(Emission::OriginalOnly {
            operation: binding.operation,
        });
        return Ok(());
    }
    #[cfg(test)]
    if DROP_NEXT_PRODUCER.replace(false) {
        let slot = producer_slot(active, op).ok_or(Error::Binding)?;
        active.projections[slot] = None;
    }
    let producer = producer_slot(active, op).ok_or(Error::Binding)?;
    let Some(CreatedOperator::Produced(p)) = &active.projections[producer] else {
        return Err(Error::Binding);
    };
    // Only source projection -> genuine original reshape, not an arbitrary
    // same-coordinate View or a shape inferred by parsing model/source names.
    if p.parent.is_none()
        || p.words != event.input().words
        || p.shape != event.shape()
        || !std::ptr::eq(event.graph(), &active.plan.plan().prefill)
        || active.rotary.is_some()
        || active.projections.len() == active.projections.capacity()
    {
        return Err(Error::Binding);
    }
    let shape: [usize; 4] = event.shape().try_into().map_err(|_| Error::Shape)?;
    let Some(CreatedOperator::Produced(projection)) = active
        .projections
        .get(p.parent.ok_or(Error::Binding)?)
        .and_then(Option::as_ref)
    else {
        return Err(Error::Binding);
    };
    let ProducedKind::Projection { full, tail, .. } = &projection.kind else {
        return Err(Error::Binding);
    };
    let width = shape[3];
    if shape[0] != 1
        || width == 0
        || width % 2 != 0
        || width > 128
        || event.sequence() == 0
        || shape[2] != event.sequence()
        || event.position() != 0
        || event.graph().query_sequence != event.sequence() as u64
        || event.graph().mode != pllm_models::DecoderMode::Prefill
        || event.input().words.len() != shape.iter().product::<usize>()
    {
        return Err(Error::Shape);
    }
    let count = event.input().words.len() / width;
    let native = tensor::Construction::RotaryHalf {
        rows: 1,
        width,
        cosine_len: width / 2,
        sine_len: width / 2,
    }
    .quote()
    .map_err(|_| Error::Bounds)?;
    let work = tensor::WorkPreflightWorkspace::resources();
    let control_bytes = [
        bytes::<Pending>(2)?,
        bytes::<Creation>(2)?,
        bytes::<usize>(4)?,
        size_of::<Consumer>(),
        size_of::<graph::RotaryEvent<'_>>(),
        size_of::<graph::ProducedEvent<'_>>(),
        bytes::<tensor::WorkPreflight<'_>>(2)?,
        size_of::<tensor::AdmittedWorkSchedule<'_>>(),
        size_of::<tensor::FactoryOutput>(),
    ]
    .into_iter()
    .try_fold(0, add)?;
    let work_preflight_bytes = [
        work.workspace_bytes,
        work.factory_bytes,
        work.ticket_bytes,
        work.admitted_schedule_bytes,
        work.admitted_part_bytes,
        work.summary_bytes,
        work.ledger_bytes,
        work.ticket_result_bytes,
        work.build_result_bytes,
        tensor::Program::rotary_half_work_resources().known_peak_bytes,
    ]
    .into_iter()
    .try_fold(0, add)?;
    let mut quote = CreationQuote {
        control_bytes,
        row_handles_bytes: bytes::<Row>(count)?,
        public_coefficient_bytes: bytes::<i64>(count.checked_mul(width).ok_or(Error::Bounds)?)?,
        mapping_bytes: Words::span_pool_bytes(event.input().words.len()).ok_or(Error::Bounds)?,
        work_preflight_bytes,
        native_construction_bytes: native.new_peak_bytes,
        phase_bytes: 0,
    };
    quote.phase_bytes = [
        quote.control_bytes,
        quote.row_handles_bytes,
        quote.public_coefficient_bytes,
        quote.mapping_bytes,
        quote
            .work_preflight_bytes
            .max(quote.native_construction_bytes),
    ]
    .into_iter()
    .try_fold(0, add)?;
    let charge = capture_charge(
        &active.plan,
        &mut active.creation_charge,
        quote.phase_bytes,
        add(
            builder.nodes.len(),
            add(
                event.input().words.len(),
                add(active.projections.len(), event.graph().operations.len())?,
            )?,
        )?,
    )?;
    let outside = active.plan.budget().used_bytes();
    if add(outside, builder.allocated)? > MAX_PROGRAM_BYTES {
        return Err(Error::Bounds);
    }
    let prior = builder
        .nodes
        .iter()
        .try_fold(WorkBudget::default(), |w, n| {
            w.checked_add(schedule_finalization::BorrowedNode::from(n).work_budget()?)
        })?;
    prior.limits()?;
    let pending = Pending {
        binding: Binding::capture(active, op)?,
        producer,
        shape,
        position: event.position(),
        sequence: event.sequence(),
        theta,
        first: builder.nodes.len(),
        input: event.input().words.clone(),
        rows: reserve(count)?,
        work: prior,
        charge,
        projection: ProjectionPrograms {
            full: Arc::clone(full),
            tail: tail.as_ref().map(Arc::clone),
            raw_bounds: builder.nodes[projection.first + 1].bounds,
        },
        quote,
    };
    builder
        .request_emission
        .as_mut()
        .ok_or(Error::Binding)?
        .rotary = Some(Emission::Capture(Box::new(pending)));
    Ok(())
}

/// Native parameters are borrowed only from original coefficient generation.
/// No caller substitutes equal bytes after this closed ticket is admitted.
pub(in crate::replicated_decoder::program) fn native_row<'a>(
    builder: &mut Builder,
    row: &Words,
    width: usize,
    cosine: &'a [i64],
    sine: &'a [i64],
) -> Result<Option<tensor::AdmittedFactory<'a>>, Error> {
    let Some(active) = &builder.request_emission else {
        return Ok(None);
    };
    let pending = match &active.rotary {
        Some(Emission::OriginalOnly { .. }) => return Ok(None),
        Some(Emission::Capture(pending)) => pending,
        None => return Err(Error::Binding),
    };
    if pending.rows.len() == pending.rows.capacity()
        || pending.shape[3] != width
        || pending
            .input
            .iter()
            .skip(pending.rows.len() * width)
            .take(width)
            .ne(row.iter())
    {
        return Err(Error::Binding);
    }
    // The continuously held phase reservation ALREADY contains this factory's
    // native peak. Add only the still-live original Builder prefix, not that
    // same native allowance a second time.
    if add(active.plan.budget().used_bytes(), builder.allocated)? > MAX_PROGRAM_BYTES {
        return Err(Error::Bounds);
    }
    let bounds = builder.bounds(row)?;
    let mut workspace = tensor::WorkPreflightWorkspace::empty();
    let ticket = tensor::Construction::RotaryHalf {
        rows: 1,
        width,
        cosine_len: cosine.len(),
        sine_len: sine.len(),
    }
    .bind(tensor::FactoryParameters::RotaryHalf {
        input: bounds,
        cosine,
        sine,
    })
    .map_err(|_| Error::Binding)?
    .preflight(1, &mut workspace)
    .map_err(|_| Error::Bounds)?;
    let mut work = pending.work;
    // Original full-program ceilings, not a smaller cap or a fresh local ledger.
    work.checked_add(ticket.summary().work)?.limits()?;
    let schedule = tensor::CallerWorkLedger::new(
        &mut work,
        puma::MAX_PROGRAM_OPERATIONS,
        puma::prss::MAX_PROGRAM_AES_BLOCKS,
    )
    .map_err(|_| Error::Bounds)?
    .admit(ticket)
    .map_err(|_| Error::Bounds)?;
    let Some(Emission::Capture(pending)) = builder
        .request_emission
        .as_mut()
        .ok_or(Error::Binding)?
        .rotary
        .as_mut()
    else {
        return Err(Error::Binding);
    };
    pending.work = work;
    let (factory, tail) = schedule.into_parts();
    if tail.is_some() {
        return Err(Error::Binding);
    }
    Ok(Some(factory))
}

pub(in crate::replicated_decoder::program) fn row_created(
    builder: &mut Builder,
    input: usize,
    output: usize,
    cosine: Vec<i64>,
    sine: Vec<i64>,
    position: u32,
) -> Result<(), Error> {
    let Some(active) = &mut builder.request_emission else {
        return Ok(());
    };
    let pending = match active.rotary.as_mut() {
        Some(Emission::OriginalOnly { .. }) => return Ok(()),
        Some(Emission::Capture(pending)) => pending,
        None => return Err(Error::Binding),
    };
    let Code::Tensor {
        input: actual,
        program,
    } = &builder.nodes.get(output).ok_or(Error::Binding)?.code
    else {
        return Err(Error::Binding);
    };
    if *actual != input
        || output != input + 1
        || input != pending.first + pending.rows.len() * 2
        || cosine.capacity() != pending.shape[3] / 2
        || sine.capacity() != pending.shape[3] / 2
        || position
            != u32::try_from(pending.position + pending.rows.len() % pending.sequence)
                .map_err(|_| Error::Bounds)?
    {
        return Err(Error::Binding);
    }
    pending.rows.push(Row {
        input,
        output,
        bounds: builder.nodes[input].bounds,
        program: Arc::clone(program),
        cosine,
        sine,
        position,
    });
    Ok(())
}

pub(in crate::replicated_decoder::program) fn completed(
    builder: &mut Builder,
    event: graph::ProducedEvent<'_>,
) -> Result<(), Error> {
    let Some(active) = &mut builder.request_emission else {
        return Ok(());
    };
    let pending = match active.rotary.take().ok_or(Error::Binding)? {
        Emission::OriginalOnly { operation } => {
            if !std::ptr::eq(
                event.operation(),
                active
                    .plan
                    .plan()
                    .prefill
                    .operations
                    .get(operation)
                    .ok_or(Error::Binding)?,
            ) {
                return Err(Error::Binding);
            }
            return Ok(());
        }
        Emission::Capture(pending) => *pending,
    };
    if !std::ptr::eq(
        event.operation(),
        &pending.binding.plan.plan().prefill.operations[pending.binding.operation],
    ) || pending.first != event.start()
        || builder.nodes.len() != pending.first + pending.rows.len() * 2
        || event.output().words.iter().ne(pending
            .rows
            .iter()
            .flat_map(|r| (0..pending.shape[3]).map(move |i| (r.output, i))))
    {
        return Err(Error::Binding);
    }
    let creation = Creation {
        binding: pending.binding,
        producer: pending.producer,
        shape: pending.shape,
        position: pending.position,
        sequence: pending.sequence,
        theta: pending.theta,
        first: pending.first,
        end: builder.nodes.len(),
        input: pending.input,
        rows: pending.rows,
        consumer: None,
        armed: false,
        projection: pending.projection,
        quote: pending.quote,
        _charge: pending.charge,
        execution: None,
    };
    active
        .projections
        .push(Some(CreatedOperator::Rotary(Box::new(creation))));
    Ok(())
}

pub(in crate::replicated_decoder::program) fn consumed(
    builder: &mut Builder,
    event: graph::ProducedEvent<'_>,
) -> Result<(), Error> {
    let Some(active) = &mut builder.request_emission else {
        return Ok(());
    };
    if !matches!(
        event.operation().operator,
        Op::KvCacheAppend | Op::AttentionScores
    ) {
        return Ok(());
    }
    let operation = event
        .graph()
        .operations
        .iter()
        .position(|o| std::ptr::eq(o, event.operation()))
        .ok_or(Error::Binding)?;
    let slot = active.projections.iter().position(|entry| match entry {
        Some(CreatedOperator::Rotary(owner)) => {
            owner.consumer.is_none()
                && event.operation().inputs.iter().any(|v| {
                    v == &owner.binding.plan.plan().prefill.operations[owner.binding.operation].id
                })
        }
        _ => false,
    });
    let Some(slot) = slot else { return Ok(()) };
    let charge = capture_charge(
        &active.plan,
        &mut active.creation_charge,
        add(
            bytes::<Consumer>(2)?,
            add(
                Words::span_pool_bytes(event.output().words.len()).ok_or(Error::Bounds)?,
                bytes::<usize>(event.output().shape.len())?,
            )?,
        )?,
        event.output().words.len(),
    )?;
    {
        let Some(CreatedOperator::Rotary(owner)) = &mut active.projections[slot] else {
            return Err(Error::Binding);
        };
        let id = &owner.binding.plan.plan().prefill.operations[owner.binding.operation].id;
        if owner.consumer.is_some() || !event.operation().inputs.iter().any(|v| v == id) {
            return Err(Error::Binding);
        }
        if event.input().is_none_or(|v| {
            v.words.iter().ne(owner
                .rows
                .iter()
                .flat_map(|r| (0..owner.shape[3]).map(move |i| (r.output, i))))
        }) {
            return Err(Error::Binding);
        }
        owner.consumer = Some(Consumer {
            operation,
            first: event.start(),
            end: builder.nodes.len(),
            words: event.output().words.clone(),
            shape: event.output().shape.clone(),
            _charge: charge,
        });
    }
    Ok(())
}

impl Creation {
    pub(in crate::replicated_decoder) fn creation_quote(&self) -> CreationQuote {
        self.quote
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn assert_foreign_bindings_rejected(
        &mut self,
        program: &Program,
        registry: &SourceRegistry<'_>,
    ) {
        self.check(program, registry).unwrap();
        let epoch = self.binding.epoch;
        self.binding.epoch += 1;
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.binding.epoch = epoch;
        let operation = self.binding.operation;
        self.binding.operation = self.producer;
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.binding.operation = operation;
        let shape = self.shape;
        self.shape[3] += 2;
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.shape = shape;
        let position = self.position;
        self.position += 1;
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.position = position;
        let input = self.input.clone();
        self.input = self.input.iter().map(|(id, lane)| (id, lane + 1)).collect();
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.input = input;
        let original = Arc::clone(&self.rows[0].program);
        let row = &self.rows[0];
        let foreign = Arc::new(
            tensor::Program::rotary_half(1, self.shape[3], row.bounds, &row.cosine, &row.sine)
                .unwrap(),
        );
        assert_eq!(original.digest(), foreign.digest());
        assert!(!Arc::ptr_eq(&original, &foreign));
        self.rows[0].program = foreign;
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.rows[0].program = original;
        let replace = |original: &Arc<tensor::Program>| {
            let mut b = tensor::Builder::new(original.lanes(), self.projection.raw_bounds).unwrap();
            let output = b.rescale(b.input(), 18).unwrap();
            let foreign = Arc::new(b.finish(output).unwrap());
            assert_eq!(original.digest(), foreign.digest());
            foreign
        };
        let full = Arc::clone(&self.projection.full);
        self.projection.full = replace(&full);
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.projection.full = full;
        if let Some(tail) = self.projection.tail.as_ref().map(Arc::clone) {
            self.projection.tail = Some(replace(&tail));
            assert!(matches!(self.check(program, registry), Err(Error::Binding)));
            self.projection.tail = Some(tail);
        }
        self.check(program, registry).unwrap();
    }
    pub(in crate::replicated_decoder) fn shape(&self) -> [usize; 4] {
        self.shape
    }
    pub(in crate::replicated_decoder) fn public(&self) -> &Arc<PublicSource> {
        &self.binding.origin.payload().public
    }
    pub(in crate::replicated_decoder) fn published(&mut self) {
        self.armed = false
    }
    pub(in crate::replicated_decoder) fn consumer_end(&self) -> Result<usize, Error> {
        Ok(self.consumer.as_ref().ok_or(Error::Binding)?.end)
    }
    pub(in crate::replicated_decoder) fn check(
        &self,
        program: &Program,
        registry: &SourceRegistry<'_>,
    ) -> Result<(), Error> {
        if self
            .execution
            .as_ref()
            .is_none_or(|p| !std::ptr::eq(p.as_ref(), program))
        {
            return Err(Error::Binding);
        }
        self.binding.check(program, registry)?;
        let custody = program.request_custody.as_ref().ok_or(Error::Binding)?;
        let Some(CreatedOperator::Produced(reshape)) = custody
            .projections
            .get(self.producer)
            .and_then(Option::as_ref)
        else {
            return Err(Error::Binding);
        };
        let Some(CreatedOperator::Produced(projection)) = custody
            .projections
            .get(reshape.parent.ok_or(Error::Binding)?)
            .and_then(Option::as_ref)
        else {
            return Err(Error::Binding);
        };
        let graph = &self.binding.plan.plan().prefill;
        let op = graph
            .operations
            .get(self.binding.operation)
            .ok_or(Error::Binding)?;
        let parent = graph
            .operations
            .get(reshape.binding.operation)
            .ok_or(Error::Binding)?;
        let producer = graph
            .operations
            .get(projection.binding.operation)
            .ok_or(Error::Binding)?;
        if !Arc::ptr_eq(&reshape.binding.plan, &self.binding.plan)
            || !Arc::ptr_eq(&projection.binding.plan, &self.binding.plan)
            || reshape.binding.epoch != self.binding.epoch
            || projection.binding.epoch != self.binding.epoch
            || !Arc::ptr_eq(&reshape.binding.origin, &self.binding.origin)
            || !Arc::ptr_eq(&projection.binding.origin, &self.binding.origin)
            || op.operator != Op::RotaryEmbedding
            || parent.operator != Op::Reshape
            || producer.operator != Op::Linear
            || op.inputs.len() != 2
            || op.inputs[0] != parent.id
            || parent.inputs.len() != 1
            || parent.inputs[0] != producer.id
            || self.input != reshape.words
            || self.shape.as_slice() != reshape.shape.as_slice()
            || reshape.first != reshape.end
            || projection.end > reshape.first
            || self.shape[2] != self.sequence
            || self.position != 0
            || graph.query_sequence != self.sequence as u64
            || op
                .attributes
                .get("theta")
                .and_then(|v| v.as_u64())
                .map(|v| v as f32)
                != Some(self.theta)
        {
            return Err(Error::Binding);
        }
        let ProducedKind::Projection { weight, full, tail } = &projection.kind else {
            return Err(Error::Binding);
        };
        if !matches!(reshape.kind, ProducedKind::Reshape) {
            return Err(Error::Binding);
        }
        let Code::PublicInput { key, digest, .. } =
            &program.recipe_at(*weight).ok_or(Error::Binding)?.code
        else {
            return Err(Error::Binding);
        };
        let node = program.recipe(*weight);
        if Some(key.as_str()) != producer.attributes.get("weight").and_then(|v| v.as_str()) {
            return Err(Error::Binding);
        }
        registry.check_descriptor(*weight, key, *digest, node.lanes, node.bounds)?;
        let Code::TensorLoop {
            program: actual,
            tail: actual_tail,
            ..
        } = &program
            .recipe_at(projection.first + 2)
            .ok_or(Error::Binding)?
            .code
        else {
            return Err(Error::Binding);
        };
        if !Arc::ptr_eq(&self.projection.full, actual)
            || match (&self.projection.tail, actual_tail) {
                (None, None) => false,
                (Some(a), Some(b)) => !Arc::ptr_eq(a, b),
                _ => true,
            }
            || program.recipe(projection.first + 1).bounds != self.projection.raw_bounds
            || !Arc::ptr_eq(full, actual)
            || match (tail, actual_tail) {
                (None, None) => false,
                (Some(a), Some(b)) => !Arc::ptr_eq(a, b),
                _ => true,
            }
        {
            return Err(Error::Binding);
        }
        for (index, row) in self.rows.iter().enumerate() {
            let gather = program.recipe_at(row.input).ok_or(Error::Binding)?;
            let native = program.recipe_at(row.output).ok_or(Error::Binding)?;
            if row.input != self.first + index * 2
                || row.output != row.input + 1
                || gather.bounds != row.bounds
                || !matches!(&gather.code,Code::Gather(words) if words.iter().eq(self.input.iter().skip(index*self.shape[3]).take(self.shape[3])))
                || !matches!(&native.code,Code::Tensor{input,program} if *input==row.input && Arc::ptr_eq(program,&row.program))
                || row.position
                    != u32::try_from(self.position + index % self.sequence)
                        .map_err(|_| Error::Bounds)?
                || row.cosine.len() != self.shape[3] / 2
                || row.sine.len() != self.shape[3] / 2
            {
                return Err(Error::Binding);
            }
        }
        let consumer = self.consumer.as_ref().ok_or(Error::Binding)?;
        let use_op = graph
            .operations
            .get(consumer.operation)
            .ok_or(Error::Binding)?;
        if !use_op.inputs.iter().any(|id| id == &op.id)
            || consumer.first < self.end
            || !matches!(use_op.operator, Op::KvCacheAppend | Op::AttentionScores)
            || consumer.words.len() != consumer.shape.iter().product::<usize>()
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
}

impl Program {
    pub(in crate::replicated_decoder) fn take_rotary(
        self: &mut Arc<Self>,
        index: usize,
    ) -> Result<Creation, Error> {
        // Wrong-family lookup does not consume or burn another genuine owner.
        let custody = self.request_custody.as_ref().ok_or(Error::Binding)?;
        if !matches!(
            custody.projections.get(index),
            Some(Some(CreatedOperator::Rotary(_)))
        ) {
            return Err(Error::Binding);
        }
        let program = Arc::get_mut(self).ok_or(Error::Binding)?;
        let Some(CreatedOperator::Rotary(owner)) = program
            .request_custody
            .as_mut()
            .ok_or(Error::Binding)?
            .projections[index]
            .take()
        else {
            return Err(Error::Binding);
        };
        let mut owner = *owner;
        owner.armed = true;
        owner.execution = Some(Arc::clone(self));
        Ok(owner)
    }
}
