//! Private construction machinery. No SDK registration or numeric-cap authority.
//! Bounds cover known logical Rust owners under the model census's existing
//! opaque-map policy, NOT allocator overhead, private dependency/compiled stacks
//! or RSS. Input creation/import/drop and diagnostic formatting are separate
//! enclosing-owner gates; this routine never retroactively admits those owners.
use super::*;
use crate::bounded_index::{
    self as bounded,
    keyed::{Entry, Index, KeyStats, Quote, Scalar},
};
use pllm_models::{StateKind, StateTensor};
use pllm_types::bounded_canonical::borrowed as json;
use std::cell::Cell;

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
enum Namespace {
    Available,
    InputId,
    InputSlot,
    OutputId,
    OutputSlot,
    Operation,
    Remote,
    Alias,
}
impl bounded::keyed::ScalarValue for Namespace {}
impl bounded::keyed::ScalarValue for StateKind {}
type PhaseKey<'p> = (
    Scalar<Namespace>,
    (Option<Scalar<u64>>, Option<Scalar<StateKind>>),
    &'p str,
);
type PhaseEntry<'p> = Entry<PhaseKey<'p>, Option<&'p ModelOperation>>;

fn id_key(namespace: Namespace, id: &str) -> PhaseKey<'_> {
    (Scalar(namespace), (None, None), id)
}
fn slot_key(namespace: Namespace, state: &StateTensor) -> PhaseKey<'static> {
    (
        Scalar(namespace),
        (state.layer.map(Scalar), Some(Scalar(state.kind))),
        "",
    )
}

/// Single row generator for key-only quotation AND actual exact row construction.
/// Namespace ordering is internal only; no enum discriminants enter serialization.
fn phase_entries<'p>(
    graph: &'p DecoderGraph,
    mut put: impl FnMut(PhaseEntry<'p>) -> Result<(), InventoryFailure<'p>>,
) -> Result<(), InventoryFailure<'p>> {
    for id in PUBLIC_INPUTS {
        for namespace in [Namespace::Available, Namespace::InputId, Namespace::Alias] {
            put(Entry {
                key: id_key(namespace, id),
                position: 0,
                value: None,
            })?;
        }
    }
    for (position, state) in graph.state_inputs.iter().enumerate() {
        for namespace in [Namespace::Available, Namespace::InputId, Namespace::Alias] {
            put(Entry {
                key: id_key(namespace, &state.id),
                position: if namespace == Namespace::InputId {
                    position + 1
                } else {
                    0
                },
                value: None,
            })?;
        }
        put(Entry {
            key: slot_key(Namespace::InputSlot, state),
            position,
            value: None,
        })?;
    }
    for (position, state) in graph.state_outputs.iter().enumerate() {
        for namespace in [Namespace::OutputId, Namespace::Alias] {
            put(Entry {
                key: id_key(namespace, &state.id),
                position,
                value: None,
            })?;
        }
        put(Entry {
            key: slot_key(Namespace::OutputSlot, state),
            position,
            value: None,
        })?;
    }
    for (position, operation) in graph.operations.iter().enumerate() {
        for namespace in [Namespace::Operation, Namespace::Available, Namespace::Alias] {
            put(Entry {
                key: id_key(namespace, &operation.id),
                position: position + 1,
                value: Some(operation),
            })?;
        }
        if remote(operation.operator) {
            put(Entry {
                key: id_key(Namespace::Remote, &operation.id),
                position: position + 1,
                value: Some(operation),
            })?;
        }
    }
    Ok(())
}

fn index_error(error: bounded::Error) -> InventoryFailure<'static> {
    InventoryFailure::Index(error)
}

pub(super) struct PhaseRows<'p> {
    rows: Box<[PhaseEntry<'p>]>,
}
pub(super) struct PhaseLookup<'r, 'p> {
    index: Index<'r, PhaseKey<'p>, Option<&'p ModelOperation>>,
}

fn phase_stats<'p>(
    graph: &'p DecoderGraph,
    context: &mut Context<'p>,
) -> Result<KeyStats<PhaseKey<'p>, Option<&'p ModelOperation>>, InventoryFailure<'p>> {
    let mut stats = KeyStats::new();
    phase_entries(graph, |entry| {
        // KeyStats preadmits this row visit AND its key projection. Do not run
        // a second, overlapping Context debit while its local usage is pending.
        stats
            .push(entry.key, context.remaining())
            .map_err(index_error)
    })?;
    context.charge(stats.resources().map_err(index_error)?.quote_work)?;
    Ok(stats)
}

impl<'p> PhaseRows<'p> {
    fn build(
        graph: &'p DecoderGraph,
        context: &mut Context<'p>,
    ) -> Result<Self, InventoryFailure<'p>> {
        let stats = phase_stats(graph, context)?;
        let mut rows = context.buffer(stats.count())?;
        phase_entries(graph, |entry| {
            context.step()?;
            rows.push(entry).map_err(index_error)
        })?;
        Ok(Self {
            rows: rows.finish().map_err(index_error)?,
        })
    }
    fn index<'r>(
        &'r self,
        context: &mut Context<'p>,
    ) -> Result<PhaseLookup<'r, 'p>, InventoryFailure<'p>> {
        let quote = Quote::checked(&self.rows, context.remaining()).map_err(index_error)?;
        let resources = quote.resources();
        context.charge(resources.quote_work)?;
        context.charge(resources.construction_work)?;
        context.factory()?;
        Ok(PhaseLookup {
            index: quote
                .construct(resources.workspace_bytes, resources.construction_work)
                .map_err(index_error)?,
        })
    }
}

impl<'p> PhaseLookup<'_, 'p> {
    fn first(
        &self,
        key: PhaseKey<'p>,
        context: &mut Context<'p>,
    ) -> Result<Option<&PhaseEntry<'p>>, InventoryFailure<'p>> {
        let bound = self.index.resources().lookup_work;
        context.charge(bound)?;
        let found = self
            .index
            .matches(&key, &mut bounded::Usage::default(), bound)
            .map_err(index_error)?;
        Ok(found.first())
    }
    pub(super) fn operation(
        &self,
        id: &'p str,
        remote_only: bool,
        context: &mut Context<'p>,
    ) -> Result<Option<&'p ModelOperation>, InventoryFailure<'p>> {
        Ok(self
            .first(
                id_key(
                    if remote_only {
                        Namespace::Remote
                    } else {
                        Namespace::Operation
                    },
                    id,
                ),
                context,
            )?
            .and_then(|entry| entry.value))
    }
    fn available(
        &self,
        id: &'p str,
        before: usize,
        context: &mut Context<'p>,
    ) -> Result<bool, InventoryFailure<'p>> {
        Ok(self
            .first(id_key(Namespace::Available, id), context)?
            .is_some_and(|entry| entry.position < before))
    }
    fn prior(
        &self,
        key: PhaseKey<'p>,
        position: usize,
        context: &mut Context<'p>,
    ) -> Result<bool, InventoryFailure<'p>> {
        Ok(self
            .first(key, context)?
            .is_some_and(|entry| entry.position < position))
    }
    pub(super) fn alias(
        &self,
        id: &'p str,
        context: &mut Context<'p>,
    ) -> Result<bool, InventoryFailure<'p>> {
        Ok(self.first(id_key(Namespace::Alias, id), context)?.is_some())
    }
    pub(super) fn validate(
        &self,
        graph: &'p DecoderGraph,
        context: &mut Context<'p>,
    ) -> Result<(), InventoryFailure<'p>> {
        for (position, state) in graph.state_inputs.iter().enumerate() {
            context.step()?;
            if state.id.is_empty()
                || self.prior(id_key(Namespace::InputId, &state.id), position + 1, context)?
                || self.prior(slot_key(Namespace::InputSlot, state), position, context)?
            {
                return Err("matrix inventory duplicate or empty state input ID/slot".into());
            }
        }
        for (position, operation) in graph.operations.iter().enumerate() {
            context.step()?;
            if operation.id.is_empty() || self.available(&operation.id, position + 1, context)? {
                return Err("matrix inventory duplicate or empty operation ID".into());
            }
            for input in &operation.inputs {
                context.step()?;
                if !self.available(input, position + 1, context)? {
                    return Err("matrix inventory unbound operation input ID".into());
                }
            }
        }
        for (position, state) in graph.state_outputs.iter().enumerate() {
            context.step()?;
            if self.prior(id_key(Namespace::OutputId, &state.id), position, context)?
                || self.prior(slot_key(Namespace::OutputSlot, state), position, context)?
            {
                return Err("matrix inventory duplicate or unbound state output metadata".into());
            }
            let producer = self.operation(&state.id, false, context)?;
            context.charge(state.shape.len() as u128)?;
            if producer.is_none_or(|op| {
                op.output_shape != state.shape
                    || op.layer != state.layer
                    || op.state_kind != Some(state.kind)
            }) {
                return Err("matrix inventory duplicate or unbound state output metadata".into());
            }
        }
        if let Some(last) = graph.operations.last() {
            context.charge((last.id.len().min(graph.output.len()) + 1) as u128)?;
        }
        if graph
            .operations
            .last()
            .is_none_or(|op| op.id != graph.output || op.operator != ModelOperator::TokenFeedback)
        {
            return Err("matrix inventory unbound graph output or token feedback".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Shape {
    dimensions: [u64; 2],
    rank: usize,
}
type SourceEntry<'p> = Entry<&'p str, Cell<Option<Shape>>>;
const SOURCE_FIELDS: [&str; 2] = ["weight", "bias"];

fn prepay_source_scan(
    plan: &DecoderPlan,
    context: &mut Context<'_>,
) -> Result<(), InventoryFailure<'static>> {
    let operations = (plan.prefill.operations.len() as u128)
        .checked_add(plan.decode.operations.len() as u128)
        .ok_or("matrix inventory source scan work overflow")?;
    // Outer operation visit and each concrete optional field visit, including
    // missing bias (no emitted row). Burn unused local-op field reservations.
    context.charge(
        operations
            .checked_mul(1 + SOURCE_FIELDS.len() as u128)
            .ok_or("matrix inventory source scan work overflow")?,
    )
}

fn source_entries<'p>(
    plan: &'p DecoderPlan,
    mut put: impl FnMut(SourceEntry<'p>) -> Result<(), InventoryFailure<'p>>,
) -> Result<(), InventoryFailure<'p>> {
    let mut position = 0;
    for graph in [&plan.prefill, &plan.decode] {
        for op in &graph.operations {
            if remote(op.operator) {
                for field in SOURCE_FIELDS {
                    if let Some(id) = op.attributes.get(field).and_then(Value::as_str) {
                        put(Entry {
                            key: id,
                            position,
                            value: Cell::new(None),
                        })?;
                        position += 1;
                    }
                }
            }
        }
    }
    Ok(())
}

fn source_stats<'p>(
    plan: &'p DecoderPlan,
    context: &mut Context<'p>,
) -> Result<KeyStats<&'p str, Cell<Option<Shape>>>, InventoryFailure<'p>> {
    prepay_source_scan(plan, context)?;
    let mut stats = KeyStats::new();
    source_entries(plan, |entry| {
        stats
            .push(entry.key, context.remaining())
            .map_err(index_error)
    })?;
    context.charge(stats.resources().map_err(index_error)?.quote_work)?;
    Ok(stats)
}

pub(super) struct SourceLedger<'r, 'p> {
    index: Index<'r, &'p str, Cell<Option<Shape>>>,
    elements: u64,
}

pub(super) trait Bindings<'p> {
    fn bind(
        &mut self,
        id: &'p str,
        shape: &[u64],
        graph: &'p DecoderGraph,
        lookup: &PhaseLookup<'_, 'p>,
        context: &mut Context<'p>,
    ) -> Result<(), InventoryFailure<'p>>;
}

impl<'p> Bindings<'p> for SourceLedger<'_, 'p> {
    fn bind(
        &mut self,
        id: &'p str,
        shape: &[u64],
        _graph: &'p DecoderGraph,
        lookup: &PhaseLookup<'_, 'p>,
        context: &mut Context<'p>,
    ) -> Result<(), InventoryFailure<'p>> {
        context.step()?;
        if lookup.alias(id, context)? {
            return Err("matrix inventory source artifact ID aliases a graph value".into());
        }
        context.charge((shape.len() + shape.len() + shape.len()) as u128)?;
        self.elements = self
            .elements
            .checked_add(shape_elements(shape)?)
            .ok_or("matrix inventory source size sum overflow")?;
        if self.elements > MAX_TOTAL_ELEMENTS {
            return Err("matrix inventory source size sum exceeds limit".into());
        }
        let bound = self.index.resources().lookup_work;
        context.charge(bound)?;
        let found = self
            .index
            .matches(&id, &mut bounded::Usage::default(), bound)
            .map_err(index_error)?;
        let cell = &found
            .first()
            .ok_or("matrix inventory missing indexed source owner")?
            .value;
        if let Some(previous) = cell.get() {
            if &previous.dimensions[..previous.rank] != shape {
                return Err("matrix inventory inconsistent source artifact shape".into());
            }
        } else {
            if shape.len() > 2 {
                return Err("matrix inventory unsupported source binding rank".into());
            }
            let mut dimensions = [0; 2];
            dimensions[..shape.len()].copy_from_slice(shape);
            cell.set(Some(Shape {
                dimensions,
                rank: shape.len(),
            }));
        }
        Ok(())
    }
}

/// All loops in the bounded core consume this single cumulative allowance.
/// Units: explicit visits, compared spans, shared index/group/JSON/stream work,
/// initialized/hashed/retired bytes or elements. Not CPU instructions or time.
pub(super) struct Context<'p> {
    bounded: bool,
    pub(super) work: u128,
    pub(super) limit: u128,
    native: [json::NativeFrame<'p>; CANONICAL_SLOTS],
    left: [json::Frame<'p>; CANONICAL_SLOTS],
    right: [json::Frame<'p>; CANONICAL_SLOTS],
    pub(super) stream: [StreamFrame<'p>; CANONICAL_SLOTS],
    factories: usize,
    plan_bytes: usize,
    json_schema_credit: u128,
    #[cfg(test)]
    first_metadata_work: Option<u128>,
    #[cfg(test)]
    metadata_events: usize,
    #[cfg(test)]
    drop_observer: Option<DropObserver<'p>>,
    #[cfg(test)]
    fail_after: Option<usize>,
    #[cfg(test)]
    panic_after: Option<usize>,
}

impl<'p> Context<'p> {
    pub(super) fn pool_work() -> Result<u128, InventoryFailure<'static>> {
        let n = CANONICAL_SLOTS as u128;
        // Four independently initialized AND explicitly retired pools. Enumerate
        // actual loops; no zero-work assumption for allocation-free state.
        n.checked_add(n)
            .and_then(|v| v.checked_add(n))
            .and_then(|v| v.checked_add(n))
            .and_then(|initialization| initialization.checked_add(initialization))
            .ok_or_else(|| "matrix inventory pool work overflow".into())
    }

    pub(super) fn new(bounded: bool, limit: u128) -> Result<Self, InventoryFailure<'static>> {
        let work = if bounded { Self::pool_work()? } else { 0 };
        if work > limit {
            return Err("matrix inventory aggregate work limit exceeded".into());
        }
        Ok(Self {
            bounded,
            work,
            limit,
            native: [const { json::NativeFrame::empty() }; CANONICAL_SLOTS],
            left: [json::Frame::empty(); CANONICAL_SLOTS],
            right: [json::Frame::empty(); CANONICAL_SLOTS],
            stream: [StreamFrame::empty(); CANONICAL_SLOTS],
            factories: 0,
            plan_bytes: 0,
            json_schema_credit: 0,
            #[cfg(test)]
            first_metadata_work: None,
            #[cfg(test)]
            metadata_events: 0,
            #[cfg(test)]
            drop_observer: None,
            #[cfg(test)]
            fail_after: None,
            #[cfg(test)]
            panic_after: None,
        })
    }
    pub(super) fn remaining(&self) -> u128 {
        if self.bounded {
            self.limit.saturating_sub(self.work)
        } else {
            u128::MAX
        }
    }
    pub(super) fn remaining_usize(&self) -> usize {
        self.remaining().min(usize::MAX as u128) as usize
    }
    pub(super) fn charge(&mut self, units: u128) -> Result<(), InventoryFailure<'static>> {
        if !self.bounded {
            return Ok(());
        }
        let next = self
            .work
            .checked_add(units)
            .ok_or("matrix inventory work arithmetic overflow")?;
        if next > self.limit {
            return Err("matrix inventory aggregate work limit exceeded".into());
        }
        self.work = next;
        Ok(())
    }
    pub(super) fn step(&mut self) -> Result<(), InventoryFailure<'static>> {
        self.charge(1)
    }

    fn prepay_json_schema(&mut self, usage: json::Usage) -> Result<(), InventoryFailure<'static>> {
        // Genuine native census: one node charge per Value, at most one String
        // value per node, exactly one key per native object candidate. Raw bytes
        // include key/string contents (and harmlessly overquote number text).
        let nodes = usage.nodes as u128;
        let credit = nodes
            .checked_add(nodes)
            .and_then(|v| v.checked_add(usage.object_candidates as u128))
            .and_then(|v| v.checked_add(usage.bytes as u128))
            .ok_or("matrix inventory schema work overflow")?;
        self.charge(credit)?;
        self.json_schema_credit = self
            .json_schema_credit
            .checked_add(credit)
            .ok_or("matrix inventory schema work overflow")?;
        // Unused prepayment is burned, never lent to the native walker or refunded.
        Ok(())
    }

    pub(super) fn settle_schema(
        &mut self,
        budget: &mut MetadataBudget,
    ) -> Result<(), InventoryFailure<'static>> {
        let pending = budget
            .work
            .checked_sub(budget.settled_work)
            .ok_or("matrix inventory schema work overflow")?;
        self.charge(pending)?;
        budget.settled_work = budget.work;
        budget.work_limit = budget
            .work
            .checked_add(self.remaining())
            .ok_or("matrix inventory schema work overflow")?;
        Ok(())
    }
    pub(super) fn factory(&mut self) -> Result<(), InventoryFailure<'static>> {
        #[cfg(test)]
        {
            if self.panic_after == Some(self.factories) {
                panic!("inventory construction factory panic");
            }
            if self.fail_after == Some(self.factories) {
                return Err("matrix inventory injected factory failure".into());
            }
        }
        self.factories += 1;
        #[cfg(test)]
        METADATA_FACTORIES.with(|count| count.set(count.get() + 1));
        Ok(())
    }
    pub(super) fn buffer<T>(
        &mut self,
        count: usize,
    ) -> Result<bounded::ExactBuffer<T>, InventoryFailure<'static>> {
        self.charge(count as u128)?; // initialization; cleanup is separately prepaid
        self.charge(count as u128)?;
        self.factory()?;
        bounded::ExactBuffer::new(count).map_err(index_error)
    }
    fn json_limits(&self) -> json::Limits {
        json::Limits {
            max_nodes: usize::MAX,
            max_depth: CANONICAL_SLOTS,
            max_bytes: usize::MAX,
            max_string_bytes: usize::MAX,
            max_key_bytes: usize::MAX,
            max_work: self.remaining_usize(),
            max_workspace_bytes: usize::MAX,
        }
    }
    pub(super) fn equal(
        &mut self,
        left: Option<&'p Value>,
        right: Option<&'p Value>,
    ) -> Result<bool, InventoryFailure<'p>> {
        self.step()?;
        if !self.bounded {
            return Ok(left == right);
        }
        let (Some(left), Some(right)) = (left, right) else {
            return Ok(left.is_none() && right.is_none());
        };
        // Two Cursor::new root assignments and two replay root resets; all
        // descendant resets occur inside the primitive's metered next loop.
        self.charge(4)?;
        let limits = self.json_limits();
        let result = json::structural_eq(left, right, &mut self.left, &mut self.right, limits);
        match result {
            Ok(result) => {
                self.charge(result.usage.work as u128)?;
                Ok(result.equal)
            }
            Err(error) => {
                self.work = self.limit;
                Err(InventoryFailure::Json(error))
            }
        }
    }
    pub(super) fn metadata(
        &mut self,
        budget: &mut MetadataBudget,
        value: &'p Value,
        root_depth: usize,
    ) -> Result<(), InventoryFailure<'static>> {
        if !self.bounded {
            return budget.value(value, root_depth);
        }
        self.settle_schema(budget)?;
        if self.json_schema_credit == 0 {
            return Err("matrix inventory schema credit exhausted".into());
        }
        self.step()?; // Cursor::new resets one supplied native root slot.
        #[cfg(test)]
        if self.first_metadata_work.is_none() {
            self.first_metadata_work = Some(self.work);
        }
        let before = budget.work;
        budget.work_limit = before
            .checked_add(self.json_schema_credit)
            .ok_or("matrix inventory schema work overflow")?;
        let mut limits = self.json_limits();
        limits.max_depth = MAX_DEPTH - root_depth + 1;
        #[cfg(test)]
        let mut events = 0;
        let (result, work) = {
            let mut traversal = json::Traversal::new_native(value, &mut self.native, limits)
                .map_err(InventoryFailure::Json)?;
            let result = (|| {
                while let Some(event) = traversal.next_event().map_err(InventoryFailure::Json)? {
                    #[cfg(test)]
                    {
                        events += 1;
                    }
                    match event {
                        json::Event::Key { key, .. } => budget.string(key)?,
                        json::Event::Value { value, depth, .. } => {
                            budget.node(root_depth + depth - 1)?;
                            match value {
                                Value::String(text) => budget.string(text)?,
                                Value::Array(values) if values.len() > MAX_OPERATIONS => {
                                    return Err(
                                        "matrix inventory metadata array limit exceeded".into()
                                    )
                                }
                                Value::Object(values) if values.len() > MAX_OBJECT_FIELDS => {
                                    return Err(
                                        "matrix inventory metadata object limit exceeded".into()
                                    )
                                }
                                _ => (),
                            }
                        }
                    }
                }
                Ok(())
            })();
            (result, traversal.usage().work)
        };
        #[cfg(test)]
        {
            self.metadata_events += events;
        }
        let spent = budget
            .work
            .checked_sub(before)
            .ok_or("matrix inventory schema work overflow")?;
        self.json_schema_credit = self
            .json_schema_credit
            .checked_sub(spent)
            .ok_or("matrix inventory schema credit exhausted")?;
        budget.settled_work = budget.work; // already covered by disjoint prepayment
        self.charge(work as u128)?;
        budget.work_limit = budget
            .work
            .checked_add(self.remaining())
            .ok_or("matrix inventory schema work overflow")?;
        result
    }

    pub(super) fn preflight_bytes(
        &mut self,
        plan: &'p DecoderPlan,
    ) -> Result<(), InventoryFailure<'static>> {
        let allowance = self.remaining_usize() / 2;
        let mut limits = plan_canonical_limits(allowance);
        limits.canonical.max_nodes = usize::MAX;
        limits.canonical.max_total_string_bytes = usize::MAX;
        match model_canonical::bounded_plan_write(plan, limits, &mut self.stream, &mut io::sink()) {
            Ok(usage) => {
                self.plan_bytes = usage.bytes;
                self.charge(2 * usage.work as u128)
            }
            Err(model_canonical::Error::Canonical(StreamError::ByteLimit)) => {
                Err("metadata byte limit exceeded".into())
            }
            Err(error) => Err(InventoryFailure::Canonical(error)),
        }
    }
}

#[cfg(test)]
struct DropObserver<'a> {
    budget: &'a pllm_garble::shared_allocation::Budget,
    observation: &'a Cell<Option<(usize, usize)>>,
}

impl Drop for Context<'_> {
    fn drop(&mut self) {
        // All four retirement loops were prepaid before any pool initialized.
        // Erase even inactive slots: they can retain native iterator references.
        #[cfg(test)]
        let mut cleared = 0;
        for slot in &mut self.native {
            *slot = json::NativeFrame::empty();
            #[cfg(test)]
            {
                cleared += 1;
            }
        }
        for slot in &mut self.left {
            *slot = json::Frame::empty();
            #[cfg(test)]
            {
                cleared += 1;
            }
        }
        for slot in &mut self.right {
            *slot = json::Frame::empty();
            #[cfg(test)]
            {
                cleared += 1;
            }
        }
        for slot in &mut self.stream {
            *slot = StreamFrame::empty();
            #[cfg(test)]
            {
                cleared += 1;
            }
        }
        #[cfg(test)]
        if let Some(observer) = &self.drop_observer {
            observer
                .observation
                .set(Some((observer.budget.used_bytes(), cleared)));
        }
    }
}

/// Both callers use exactly this indexed lowering/geometry/canonical identity path.
pub(super) fn metadata<'p>(
    plan: &'p DecoderPlan,
    context: &mut Context<'p>,
) -> Result<Inventory<'p>, InventoryFailure<'p>> {
    let prefill_rows = PhaseRows::build(&plan.prefill, context)?;
    let prefill_index = prefill_rows.index(context)?;
    let source_stats = source_stats(plan, context)?;
    let mut source_rows = context.buffer(source_stats.count())?;
    prepay_source_scan(plan, context)?;
    context.charge(plan.prefill.operations.len() as u128 + plan.decode.operations.len() as u128)?;
    source_entries(plan, |entry| {
        context.step()?;
        source_rows.push(entry).map_err(index_error)
    })?;
    let source_rows = source_rows.finish().map_err(index_error)?;
    let source_quote = Quote::checked(&source_rows, context.remaining()).map_err(index_error)?;
    let resources = source_quote.resources();
    context.charge(resources.quote_work)?;
    context.charge(resources.construction_work)?;
    context.factory()?;
    let mut sources = SourceLedger {
        index: source_quote
            .construct(resources.workspace_bytes, resources.construction_work)
            .map_err(index_error)?,
        elements: 0,
    };
    let prefill = lower_phase(&plan.prefill, &mut sources, &prefill_index, context)?;
    let decode_rows = PhaseRows::build(&plan.decode, context)?;
    let decode_index = decode_rows.index(context)?;
    let decode = lower_phase(&plan.decode, &mut sources, &decode_index, context)?;
    compatible(&prefill, &decode, plan, &decode_index, context)?;
    let model_plan_digest = plan_identity(plan, context)?;
    context.charge(64)?; // fixed plan identity hexadecimal encoding
    Ok(Inventory {
        schema_version: DECODER_MATRIX_INVENTORY_SCHEMA_VERSION,
        model_plan_digest: HexDigest::new(model_plan_digest),
        model_config_digest: &plan.config_digest,
        prefill,
        decode,
    })
}

fn plan_identity<'p>(
    plan: &'p DecoderPlan,
    context: &mut Context<'p>,
) -> Result<[u8; 32], InventoryFailure<'p>> {
    context.charge((pllm_models::DECODER_PLAN_SCHEMA_VERSION.len() + 1) as u128)?;
    // The preflight sink measured the same immutable plan's canonical bytes.
    // Precharge hash transport separately, then bound BOTH model-write passes.
    context.charge(context.plan_bytes as u128)?;
    let allowance = if context.bounded {
        context.remaining_usize() / 2
    } else {
        usize::MAX
    };
    let mut hash = DomainHash::new(pllm_models::DECODER_PLAN_SCHEMA_VERSION);
    let result = model_canonical::bounded_plan_write(
        plan,
        plan_canonical_limits(allowance),
        &mut context.stream,
        &mut hash,
    );
    match result {
        Ok(usage) => {
            context.charge(2 * usage.work as u128)?;
            if context.bounded && usage.bytes != context.plan_bytes {
                return Err("matrix inventory plan replay length mismatch".into());
            }
            context.step()?;
            Ok(hash.finish())
        }
        Err(error) => {
            if context.bounded {
                context.work = context.limit;
            }
            Err(InventoryFailure::Canonical(error))
        }
    }
}

use pllm_garble::shared_allocation::{Error as NativeError, OwnedCharged, OwnedExternalCharge};
use pllm_models::owned_backing::bounded as census;

/// Opt-in cumulative work ceiling. Existing metadata/geometry/output limits are
/// unchanged. This is not a Budget, reservation, source or execution authority.
#[derive(Clone, Copy)]
struct Limits {
    max_work: u128,
}

#[derive(Clone, Copy, Debug)]
struct Resources {
    input_owner_bytes: usize,
    input_reserved_bytes: usize,
    bootstrap_bytes: usize,
    row_backing_bytes: usize,
    index_workspace_bytes: usize,
    grouping_workspace_bytes: usize,
    metadata_backing_bytes: usize,
    output_and_digest_bytes: usize,
    construction_payload_bytes: usize,
    outcome_inline_bytes: usize,
    final_working_payload_bytes: usize,
    additional_reservation_bytes: usize,
    total_original_counter_bytes: usize,
    max_work: u128,
}

/// The borrow is of the actual charged owner, not merely its payload. No detach,
/// mutable replacement, diagnostic extraction or reservation migration exists.
/// All construction reservation is deliberately retained through either result.
/// Lifetime proof: the actual T in OwnedCharged<T> contains the original owner
/// borrow. OwnedCharged<T>'s non-may_dangle Drop destroys T before its charge;
/// drop checking therefore requires that owner through destruction, not merely
/// the last payload read. There is no lifetime-erasing/detaching accessor. The
/// separate public compile-fail import checks privacy ONLY, not this property.
struct Outcome<'p> {
    input: &'p OwnedCharged<DecoderPlan>,
    resources: Option<Resources>,
    work: u128,
    factories: usize,
    result: Result<DecoderMatrixInventory, InventoryFailure<'p>>,
}

struct ConstructionQuote<'p> {
    input: &'p OwnedCharged<DecoderPlan>,
    resources: Resources,
    context: Context<'p>,
    // Declaration order retires fixed frames/control before their real charge.
    working: OwnedExternalCharge,
}

// Scalar checked accumulation: unlike a quotation array this creates no owned
// bookkeeping collection before the bootstrap guard is checked.
macro_rules! sum {
    (&[$($part:expr),* $(,)?]) => {{
        let mut total = 0usize;
        $(total = total.checked_add($part).ok_or(InventoryFailure::Static("matrix inventory resource arithmetic overflow"))?;)*
        Ok::<usize, InventoryFailure<'static>>(total)
    }};
}

fn quote_control_bytes() -> Result<usize, InventoryFailure<'static>> {
    sum!(&[
        size_of::<census::Resources>(),
        size_of::<Result<census::Resources, census::Error>>(),
        size_of::<json::Resources>(),
        size_of::<Result<json::Resources, json::Error>>(),
        size_of::<model_canonical::ResourceQuote>(),
        size_of::<Result<model_canonical::ResourceQuote, model_canonical::Error>>(),
        size_of::<pllm_types::bounded_canonical::Resources>(),
        size_of::<Result<pllm_types::bounded_canonical::Resources, StreamError>>(),
        size_of::<Result<usize, InventoryFailure<'_>>>(),
        size_of::<KeyStats<PhaseKey<'_>, Option<&ModelOperation>>>(),
        size_of::<KeyStats<&str, Cell<Option<Shape>>>>(),
        size_of::<bounded::keyed::Resources>(),
        size_of::<bounded::keyed::Resources>(),
        size_of::<Result<bounded::keyed::Resources, bounded::Error>>(),
        // Exact concrete accounting-array policy in KeyStats::resources (24
        // named terms), including its IntoIter. No private tree-layout estimate.
        size_of::<[usize; 24]>(),
        size_of::<std::array::IntoIter<usize, 24>>(),
    ])
}

/// Supplied frame capacity plus every concrete inventory stream pass slot.
/// Writer backing is priced separately; moves transfer logical slots. Native
/// stack/formatter-private buffers, allocator overhead and RSS are NOT included.
fn pass_resources<W: Write>() -> Result<usize, InventoryFailure<'static>> {
    let stream =
        Session::<W>::resources_for_slots(CANONICAL_SLOTS).map_err(InventoryFailure::Stream)?;
    sum!(&[
        stream.stack_bytes,
        size_of::<Result<Session<'_, '_, W>, StreamError>>(),
        size_of::<InventoryCursors<'_>>(),
        size_of::<Result<pllm_types::bounded_canonical::StreamUsage, StreamError>>(),
        stream.emitter_bytes,
        stream.escape_buffer_bytes,
        stream.format_arguments_bytes,
        stream.compact_formatter_bytes,
        size_of::<W>(),
    ])
}

/// Allocation-free numeric bootstrap, evaluated before creating any fixed frame
/// arrays/control records. The caller ALREADY owns a real working charge. It
/// covers this quote's logical arithmetic/return records as part of handoff.
fn bootstrap_bytes() -> Result<usize, InventoryFailure<'static>> {
    let native = census::resources_for_slots(CANONICAL_SLOTS).map_err(InventoryFailure::Census)?;
    let equality = json::equality_resources_for_slots(CANONICAL_SLOTS, CANONICAL_SLOTS)
        .map_err(InventoryFailure::Json)?;
    let canonical = model_canonical::resources::<DomainHash>(CANONICAL_SLOTS)
        .map_err(InventoryFailure::Canonical)?;
    // Empty-key prequotes price the SAME concrete quote/index/cursor machinery
    // before derived rows exist. Nonempty permutation backing is added later.
    let phase_index = KeyStats::<PhaseKey<'_>, Option<&ModelOperation>>::new()
        .resources()
        .map_err(index_error)?;
    let source_index = KeyStats::<&str, Cell<Option<Shape>>>::new()
        .resources()
        .map_err(index_error)?;
    // Deliberately sum sequential phases too. No overlap credit for retired
    // frames, earlier Results, quote records or partially initialized metadata.
    sum!(&[
        quote_control_bytes()?,
        size_of::<Context<'_>>(),
        size_of::<Result<Context<'_>, InventoryFailure<'_>>>(),
        size_of::<Result<u128, InventoryFailure<'_>>>(),
        size_of::<ConstructionQuote<'_>>(),
        size_of::<Result<ConstructionQuote<'_>, (InventoryFailure<'_>, OwnedExternalCharge)>>(),
        size_of::<Resources>(),
        size_of::<Result<Resources, InventoryFailure<'_>>>(),
        size_of::<Outcome<'_>>(),
        size_of::<Result<OwnedCharged<Outcome<'_>>, NativeError>>(),
        size_of::<OwnedExternalCharge>(),
        size_of::<Limits>(),
        size_of::<MetadataBudget>(),
        size_of::<MetadataEnum<'_>>(),
        size_of::<RemoteGroupingQuote<'_>>(),
        size_of::<crate::decoder_runtime_schedule::RemoteGroupingResources>(),
        size_of::<Result<RemoteGroupingQuote<'_>, RemoteGroupingError<'_>>>(),
        size_of::<Inventory<'_>>(),
        size_of::<Phase<'_>>(),
        size_of::<Phase<'_>>(),
        size_of::<Group<'_>>(),
        size_of::<Member<'_>>(),
        size_of::<InputShape<'_>>(),
        size_of::<HexDigest>(),
        size_of::<Result<Inventory<'_>, InventoryFailure<'_>>>(),
        size_of::<Result<Phase<'_>, InventoryFailure<'_>>>(),
        size_of::<Result<DecoderMatrixInventory, InventoryFailure<'_>>>(),
        size_of::<Result<(), InventoryFailure<'_>>>(),
        size_of::<Result<u64, InventoryFailure<'_>>>(),
        size_of::<Result<usize, InventoryFailure<'_>>>(),
        size_of::<Result<Vec<u8>, InventoryFailure<'_>>>(),
        size_of::<DomainHash>(),
        size_of::<Result<[u8; 32], model_canonical::Error>>(),
        size_of::<Box<[MaybeUninit<u8>]>>(),
        size_of::<Box<[u8]>>(),
        size_of::<std::io::Cursor<&mut [u8]>>(),
        size_of::<String>(),
        size_of::<Digest>(),
        size_of::<Result<String, std::string::FromUtf8Error>>(),
        size_of::<Result<Digest, serde::de::value::Error>>(),
        size_of::<serde::de::value::StringDeserializer<serde::de::value::Error>>(),
        size_of::<bounded::ExactBuffer<Group<'_>>>(),
        size_of::<bounded::ExactBuffer<Member<'_>>>(),
        size_of::<Result<Box<[Group<'_>]>, bounded::Error>>(),
        size_of::<Result<Box<[Member<'_>]>, bounded::Error>>(),
        size_of::<bounded::ExactBuffer<PhaseEntry<'_>>>(),
        size_of::<bounded::ExactBuffer<SourceEntry<'_>>>(),
        size_of::<Result<bounded::ExactBuffer<Group<'_>>, InventoryFailure<'_>>>(),
        size_of::<Result<bounded::ExactBuffer<Member<'_>>, InventoryFailure<'_>>>(),
        size_of::<Result<bounded::ExactBuffer<PhaseEntry<'_>>, InventoryFailure<'_>>>(),
        size_of::<Result<bounded::ExactBuffer<SourceEntry<'_>>, InventoryFailure<'_>>>(),
        size_of::<Result<bounded::ExactBuffer<Group<'_>>, bounded::Error>>(),
        size_of::<Result<bounded::ExactBuffer<Member<'_>>, bounded::Error>>(),
        size_of::<Result<bounded::ExactBuffer<PhaseEntry<'_>>, bounded::Error>>(),
        size_of::<Result<bounded::ExactBuffer<SourceEntry<'_>>, bounded::Error>>(),
        size_of::<PhaseRows<'_>>(),
        size_of::<PhaseRows<'_>>(),
        size_of::<SourceLedger<'_, '_>>(),
        // Actual iterator/next-result types of row generation, geometry and
        // validation. String/JSON backing remains in original owner census.
        size_of::<std::iter::Enumerate<std::slice::Iter<'_, ModelOperation>>>(),
        size_of::<std::slice::Iter<'_, ModelOperation>>(),
        size_of::<Option<&ModelOperation>>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_, StateTensor>>>(),
        size_of::<std::slice::Iter<'_, String>>(),
        size_of::<std::slice::Iter<'_, u64>>(),
        size_of::<std::slice::Iter<'_, Group<'_>>>(),
        size_of::<std::slice::Iter<'_, Member<'_>>>(),
        size_of::<std::iter::Zip<std::slice::Iter<'_, String>, std::slice::Iter<'_, String>>>(),
        size_of::<std::iter::Zip<std::slice::Iter<'_, Group<'_>>, std::slice::Iter<'_, Group<'_>>>>(
        ),
        size_of::<std::iter::Zip<std::slice::Iter<'_, Member<'_>>, std::slice::Iter<'_, Member<'_>>>>(
        ),
        size_of::<serde_json::map::Iter<'_>>(),
        size_of::<Result<Option<json::Event<'_>>, json::Error>>(),
        size_of::<Option<(usize, &ModelOperation)>>(),
        size_of::<Option<(usize, &StateTensor)>>(),
        size_of::<Result<PhaseRows<'_>, InventoryFailure<'_>>>(),
        size_of::<Result<PhaseLookup<'_, '_>, InventoryFailure<'_>>>(),
        size_of::<Result<Box<[PhaseEntry<'_>]>, bounded::Error>>(),
        size_of::<Result<Box<[SourceEntry<'_>]>, bounded::Error>>(),
        size_of::<Result<Box<[u8]>, InventoryFailure<'_>>>(),
        size_of::<[Namespace; 3]>(),
        size_of::<std::array::IntoIter<Namespace, 3>>(),
        size_of::<[Namespace; 2]>(),
        size_of::<std::array::IntoIter<Namespace, 2>>(),
        size_of::<[&DecoderGraph; 2]>(),
        size_of::<std::array::IntoIter<&DecoderGraph, 2>>(),
        size_of::<[&str; 2]>(),
        size_of::<std::array::IntoIter<&str, 2>>(),
        size_of::<[&str; 4]>(),
        size_of::<std::array::IntoIter<&str, 4>>(),
        size_of::<Result<pllm_types::bounded_canonical::StreamUsage, model_canonical::Error>>(),
        native.logical_workspace_bytes,
        equality.logical_workspace_bytes,
        canonical.write_workspace_bytes,
        canonical.hash_workspace_bytes,
        phase_index.workspace_bytes,
        source_index.workspace_bytes,
        pass_resources::<io::Sink>()?,
        pass_resources::<io::Cursor<&mut [u8]>>()?,
        pass_resources::<DomainHash>()?,
    ])
}

impl<'p> ConstructionQuote<'p> {
    fn checked(
        input: &'p OwnedCharged<DecoderPlan>,
        working: OwnedExternalCharge,
        limits: Limits,
    ) -> Result<Self, (InventoryFailure<'p>, OwnedExternalCharge)> {
        match Self::parts(input, &working, limits) {
            Ok((resources, context)) => Ok(Self {
                input,
                resources,
                context,
                working,
            }),
            Err(error) => Err((error, working)),
        }
    }

    fn parts(
        input: &'p OwnedCharged<DecoderPlan>,
        working: &OwnedExternalCharge,
        limits: Limits,
    ) -> Result<(Resources, Context<'p>), InventoryFailure<'p>> {
        if working.payload_bytes() < quote_control_bytes()? {
            return Err("matrix inventory bootstrap reservation too short".into());
        }
        let bootstrap = bootstrap_bytes()?;
        if !working.is_for(input.budget()) {
            return Err("matrix inventory original Budget identity mismatch".into());
        }
        if working.payload_bytes() < bootstrap {
            return Err("matrix inventory bootstrap reservation too short".into());
        }
        let plan = input.payload();
        // Census invokes exactly one native Cursor::new per operation's attrs.
        // Its DTO/JSON visits have their own shared meter; root-slot assignment
        // is prepaid independently, including roots skipped by early failure.
        let census_roots = (plan.prefill.operations.len() as u128)
            .checked_add(plan.decode.operations.len() as u128)
            .ok_or("matrix inventory census root work overflow")?;
        // Genuine original capacities; no legacy recursive fallback or caller
        // baseline. Effective census depth remains its documented <=16 policy.
        let native_limits = census::Limits {
            max_backing_bytes: input.charged_bytes(),
            max_work: limits
                .max_work
                .checked_sub(Context::pool_work()?)
                .and_then(|work| work.checked_sub(census_roots))
                .ok_or("matrix inventory aggregate work limit exceeded")?
                .min(usize::MAX as u128) as usize,
            max_workspace_bytes: bootstrap,
            max_json_nodes: MAX_NODES,
            max_json_payload_bytes: usize::MAX,
            max_json_string_bytes: usize::MAX,
            max_json_key_bytes: usize::MAX,
            max_json_depth: 16,
        };
        census::resources_for_slots(CANONICAL_SLOTS)
            .map_err(InventoryFailure::Census)?
            .check(native_limits)
            .map_err(InventoryFailure::Census)?;
        let mut context = Context::new(true, limits.max_work)?;
        context.charge(census_roots)?;
        let (report, work) = {
            let mut owner = census::Census::new(&mut context.native, native_limits)
                .map_err(InventoryFailure::Census)?;
            let report = owner.plan(plan).map_err(InventoryFailure::Census)?;
            (report, owner.usage().work)
        };
        context.charge(work as u128)?;
        let input_owner_bytes = sum!(&[
            size_of::<OwnedCharged<DecoderPlan>>(),
            report.backing_bytes()
        ])?;
        if input_owner_bytes > input.charged_bytes() {
            return Err("matrix inventory original plan reservation too short".into());
        }
        context.prepay_json_schema(report.usage().json)?;
        preflight_inner(plan, context.remaining_usize(), Some(&mut context))?;
        let mut row_backing_bytes = 0;
        let mut index_workspace_bytes = 0;
        let mut grouping_workspace_bytes = 0;
        let mut metadata_backing_bytes = 0;
        for graph in [&plan.prefill, &plan.decode] {
            context.step()?;
            let stats = phase_stats(graph, &mut context)?;
            row_backing_bytes = sum!(&[
                row_backing_bytes,
                bounded::layout_bytes::<PhaseEntry<'_>>(stats.count()).map_err(index_error)?
            ])?;
            index_workspace_bytes = sum!(&[
                index_workspace_bytes,
                stats.resources().map_err(index_error)?.workspace_bytes
            ])?;
            context.charge(graph.operations.len() as u128)?;
            for operation in &graph.operations {
                context.charge(operation.inputs.len() as u128)?;
            }
            let group = RemoteGroupingQuote::checked(graph)
                .map_err(InventoryFailure::Grouping)?
                .resources();
            grouping_workspace_bytes = sum!(&[grouping_workspace_bytes, group.workspace_bytes])?;
            metadata_backing_bytes = sum!(&[
                metadata_backing_bytes,
                bounded::layout_bytes::<Group<'_>>(graph.operations.len()).map_err(index_error)?,
                bounded::layout_bytes::<Member<'_>>(graph.operations.len()).map_err(index_error)?
            ])?;
        }
        let stats = source_stats(plan, &mut context)?;
        row_backing_bytes = sum!(&[
            row_backing_bytes,
            bounded::layout_bytes::<SourceEntry<'_>>(stats.count()).map_err(index_error)?
        ])?;
        index_workspace_bytes = sum!(&[
            index_workspace_bytes,
            stats.resources().map_err(index_error)?.workspace_bytes
        ])?;
        // Pre-admit the entire existing output ceiling BEFORE discovering length.
        let output_and_digest_bytes = sum!(&[MAX_OUTPUT_BYTES, 64])?;
        let construction_payload_bytes = sum!(&[
            bootstrap,
            row_backing_bytes,
            index_workspace_bytes,
            grouping_workspace_bytes,
            metadata_backing_bytes,
            output_and_digest_bytes
        ])?
        .max(working.payload_bytes());
        let outcome_inline_bytes = size_of::<OwnedCharged<Outcome<'_>>>()
            .checked_sub(size_of::<OwnedExternalCharge>())
            .ok_or("matrix inventory outcome layout overflow")?;
        let final_working_payload_bytes =
            sum!(&[construction_payload_bytes, outcome_inline_bytes])?;
        let additional_reservation_bytes = final_working_payload_bytes
            .checked_sub(working.payload_bytes())
            .ok_or("matrix inventory reservation arithmetic overflow")?;
        let total_original_counter_bytes =
            sum!(&[input.budget().used_bytes(), additional_reservation_bytes])?;
        Ok((
            Resources {
                input_owner_bytes,
                input_reserved_bytes: input.charged_bytes(),
                bootstrap_bytes: bootstrap,
                row_backing_bytes,
                index_workspace_bytes,
                grouping_workspace_bytes,
                metadata_backing_bytes,
                output_and_digest_bytes,
                construction_payload_bytes,
                outcome_inline_bytes,
                final_working_payload_bytes,
                additional_reservation_bytes,
                total_original_counter_bytes,
                max_work: limits.max_work,
            },
            context,
        ))
    }

    fn construct(self) -> Result<OwnedCharged<Outcome<'p>>, NativeError> {
        let Self {
            input,
            resources,
            context,
            mut working,
        } = self;
        if let Err(error) = working.resize_payload(resources.construction_payload_bytes) {
            drop(context);
            return working.try_build(|_| {
                Ok(Outcome {
                    input,
                    resources: Some(resources),
                    work: resources.max_work,
                    factories: 0,
                    result: Err(InventoryFailure::Native(error)),
                })
            });
        }
        working.try_build(move |_| {
            let mut context = context;
            let result = finish(input.payload(), &mut context);
            if result.is_err() {
                context.work = context.limit;
            }
            let work = context.work;
            let factories = context.factories;
            drop(context);
            Ok(Outcome {
                input,
                resources: Some(resources),
                work,
                factories,
                result,
            })
        })
    }
}

/// Private opt-in entry. Bootstrap/Native reservation failures may return only a
/// fixed NativeError (never an allocated/borrowed diagnostic). Inventory failures
/// are payloads inside the original same-counter charge, just like success.
fn construct(
    input: &OwnedCharged<DecoderPlan>,
    working: OwnedExternalCharge,
    limits: Limits,
) -> Result<OwnedCharged<Outcome<'_>>, NativeError> {
    if !working.is_for(input.budget()) {
        return Err(NativeError::Context);
    }
    if working.payload_bytes() < quote_control_bytes().map_err(|_| NativeError::Resource)? {
        return Err(NativeError::Resource);
    }
    let quote = match ConstructionQuote::checked(input, working, limits) {
        Ok(quote) => quote,
        Err((error, working)) => {
            return working.try_build(|_| {
                Ok(Outcome {
                    input,
                    resources: None,
                    work: limits.max_work,
                    factories: 0,
                    result: Err(error),
                })
            })
        }
    };
    quote.construct()
}

fn controlled_pass<'p, W: Write>(
    inventory: &Inventory<'p>,
    context: &mut Context<'p>,
    output: &mut W,
) -> Result<pllm_types::bounded_canonical::StreamUsage, InventoryFailure<'p>> {
    let mut limits = inventory_stream_limits();
    limits.max_work = context.remaining_usize();
    match inventory_pass_with_limits(inventory, &mut context.stream, output, limits) {
        Ok(usage) => {
            context.charge(usage.work as u128)?;
            Ok(usage)
        }
        Err(error) => {
            context.work = context.limit;
            Err(InventoryFailure::Stream(error))
        }
    }
}

fn finish<'p>(
    plan: &'p DecoderPlan,
    context: &mut Context<'p>,
) -> Result<DecoderMatrixInventory, InventoryFailure<'p>> {
    let inventory = metadata(plan, context)?;
    let usage = controlled_pass(&inventory, context, &mut io::sink())?;
    context.charge(usage.bytes as u128)?; // initialize exact final backing
    context.charge(usage.bytes as u128)?; // final output hash
    context.charge(usage.bytes as u128)?; // retirement; no cleanup work refund
    context.factory()?;
    let mut bytes = Box::<[u8]>::new_uninit_slice(usage.bytes);
    for byte in bytes.iter_mut() {
        byte.write(0);
    }
    // SAFETY: every element was initialized; no growth/shrink or source copy.
    let mut bytes = unsafe { bytes.assume_init() };
    let mut output = io::Cursor::new(bytes.as_mut());
    let replay = controlled_pass(&inventory, context, &mut output)?;
    if output.position() != usage.bytes as u64 || replay.bytes != usage.bytes {
        return Err("matrix inventory canonical replay length mismatch".into());
    }
    let semantic_plan_identity = inventory.model_plan_digest.raw;
    context.charge(64)?; // decode validated configuration hexadecimal identity
    let config_identity = inventory.model_config_digest.as_sha256();
    // No borrowed metadata/index/source owner survives final encoding.
    drop(inventory);
    context.charge((DECODER_MATRIX_INVENTORY_SCHEMA_VERSION.len() + 1) as u128)?;
    let mut hash = DomainHash::new(DECODER_MATRIX_INVENTORY_SCHEMA_VERSION);
    hash.write_all(&bytes).expect("infallible shared hash sink");
    context.charge(64)?; // fixed hex encoding
    context.charge(64)?; // Digest's owned-string validation (no copy/growth)
    context.charge(64)?; // fixed digest backing retirement
    context.step()?; // hash finalization (native/dependency instructions excluded)
    context.factory()?;
    let digest = HexDigest::new(hash.finish()).into_digest();
    Ok(DecoderMatrixInventory {
        bytes: bytes.into_vec(),
        digest,
        semantic_plan_identity,
        config_identity,
    })
}

/// Concrete owning-return slots, added to the unchanged checked construction
/// envelope. Input backing is still censused from the same genuine owner.
fn owned_inventory_control_bytes() -> Result<usize, NativeError> {
    let mut bytes = size_of::<OwnedCharged<DecoderMatrixInventory>>();
    bytes = bytes
        .checked_add(size_of::<
            Result<OwnedCharged<DecoderMatrixInventory>, NativeError>,
        >())
        .ok_or(NativeError::Resource)?;
    bytes
        .checked_add(size_of::<Result<DecoderMatrixInventory, NativeError>>())
        .ok_or(NativeError::Resource)
}

pub(crate) fn owned_inventory_bootstrap_bytes() -> Result<usize, NativeError> {
    bootstrap_bytes()
        .map_err(|_| NativeError::Resource)?
        .checked_add(owned_inventory_control_bytes()?)
        .ok_or(NativeError::Resource)
}

/// Independently owning inventory, constructed from the original charged plan.
/// No OwnedCharged extraction, borrowed-owner upgrade, cloned plan or refund/
/// reacquire. Calling bundle keeps the original owner through inventory lifetime;
/// this result alone grants no source/execution authority. Legacy diagnostics
/// stay unchanged; this private consuming funnel returns fixed rejection only.
pub(crate) fn admit_inventory_for_original_owner(
    input: &OwnedCharged<DecoderPlan>,
    working: OwnedExternalCharge,
    max_work: u128,
) -> Result<OwnedCharged<DecoderMatrixInventory>, NativeError> {
    if !working.is_for(input.budget()) {
        return Err(NativeError::Context);
    }
    if working.payload_bytes() < owned_inventory_bootstrap_bytes()? {
        return Err(NativeError::Resource);
    }
    let extra = owned_inventory_control_bytes()?;
    let quote = match ConstructionQuote::checked(input, working, Limits { max_work }) {
        Ok(quote) => quote,
        Err((error, working)) => {
            // Retire the borrowed diagnostic while the original guard is live.
            {
                let _diagnostic = error;
            }
            drop(working);
            return Err(NativeError::Resource);
        }
    };
    // Do all fallible arithmetic before destructuring the quote, whose field
    // order already retires Context before its charge on early return/unwind.
    let payload = quote
        .resources
        .construction_payload_bytes
        .checked_add(extra)
        .ok_or(NativeError::Resource)?;
    let ConstructionQuote {
        context,
        mut working,
        ..
    } = quote;
    let payload = payload.max(working.payload_bytes()); // never shrink incoming guard
    if let Err(error) = working.resize_payload(payload) {
        drop(context);
        return Err(error);
    }
    let plan = input.payload();
    working.try_build(move |_| {
        let mut context = context;
        let result = finish(plan, &mut context);
        if result.is_err() {
            context.work = context.limit;
        }
        result.map_err(|_| NativeError::Resource)
        // Captured Context erases prepaid pools before try_build's guard can
        // refund on error/panic. Success retains the entire original reservation.
    })
}

#[cfg(test)]
#[path = "decoder_matrix_inventory_construction_tests.rs"]
mod tests;
