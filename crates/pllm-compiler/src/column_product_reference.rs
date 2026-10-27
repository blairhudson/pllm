//! Algorithm 5 CLEAR reference adapter over actual retained clear-input custody.
//! Not CKKS packing, checkpoint authentication, or full NEXUS adoption.
//! No caller ledger, lifecycle state, guard, DTO, or buffer adoption route.
use crate::clear_input_owner::{self as owner, BorrowedOutput, ClearInputs, Phase, Prepared};
use crate::clear_value_binding::{BoundValue, RetainedProducer, ValueEdge};
use pllm_core::{column_product_reference as core, row_fold::State};
use pllm_models::{ModelOperation, ModelOperator};
use std::{cell::Ref, mem::size_of};
use zeroize::Zeroize;

const CONTROL: usize = 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Owner(owner::Error),
    Arithmetic(core::Error),
    OperationPolicy,
    Resource,
}
impl From<owner::Error> for Refusal {
    fn from(error: owner::Error) -> Self { Self::Owner(error) }
}
impl From<core::Error> for Refusal {
    fn from(error: core::Error) -> Self { Self::Arithmetic(error) }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status { ClearReference }

// Nested capacities reserved by the phase BEFORE allocation. Separate retained
// values are necessary: dropping core Output wipes its borrowed score storage.
struct Scores {
    workspace: Vec<f64>,
    scratch_scores: Vec<f64>,
    values: Vec<f64>,
}
impl Zeroize for Scores {
    fn zeroize(&mut self) {
        self.workspace.as_mut_slice().zeroize();
        self.scratch_scores.as_mut_slice().zeroize();
        self.values.as_mut_slice().zeroize();
    }
}
impl Drop for Scores {
    fn drop(&mut self) {
        self.zeroize();
        #[cfg(test)]
        observe::retired(true, self.workspace.iter().chain(&self.scratch_scores).chain(&self.values),
            self.workspace.len() + self.scratch_scores.len() + self.values.len(),
            self.workspace.capacity() + self.scratch_scores.capacity() + self.values.capacity());
    }
}
fn initialize(values: &mut Vec<f64>, length: usize) -> Result<(), Refusal> {
    #[cfg(test)]
    observe::factory();
    values.try_reserve_exact(length).map_err(|_| Refusal::Resource)?;
    if values.capacity() != length { return Err(Refusal::Resource); }
    values.resize(length, 0.0);
    Ok(())
}

/// Read-only result retains the actual input, metadata, ledger and budget root.
/// Positive API control: no missing-import success for the negative examples.
/// ```
/// use pllm_compiler::{clear_input_owner::ClearInputs, column_product_reference::{execute, RetainedOutput}};
/// use pllm_core::column_product_reference::{Geometry, Limits};
/// fn run(input: &ClearInputs, g: Geometry, limits: Limits) -> RetainedOutput<'_> {
///     execute(input, 0, 0, 1, g, limits).unwrap()
/// }
/// ```
/// Original DTO/backing owner cannot drop while an actual result remains live.
/// ```compile_fail,E0505
/// use pllm_compiler::{clear_input_owner::ClearInputs, column_product_reference::execute};
/// use pllm_core::column_product_reference::{Geometry, Limits};
/// fn retained_output_borrows_original_dto(input: ClearInputs, g: Geometry, limits: Limits) {
///     let output = execute(&input, 0, 0, 1, g, limits).unwrap();
///     drop(input);
///     std::hint::black_box(output.operation());
/// }
/// ```
/// Owner replacement cannot release its original reservation behind output.
/// ```compile_fail,E0506
/// use pllm_compiler::{clear_input_owner::ClearInputs, column_product_reference::execute};
/// use pllm_core::column_product_reference::{Geometry, Limits};
/// fn retained_output_borrows_original_guard(input: &mut ClearInputs, replacement: ClearInputs, g: Geometry, limits: Limits) {
///     let output = execute(input, 0, 0, 1, g, limits).unwrap();
///     *input = replacement;
///     std::hint::black_box(output.values());
/// }
/// ```
/// Numeric views are immutable, not caller-owned mutable aliases.
/// ```compile_fail,E0594
/// use pllm_compiler::{clear_input_owner::ClearInputs, column_product_reference::execute};
/// use pllm_core::column_product_reference::{Geometry, Limits};
/// fn retained_output_borrows_original_input(input: &ClearInputs, g: Geometry, limits: Limits) {
///     let output = execute(input, 0, 0, 1, g, limits).unwrap();
///     let mut query = output.original().f64_tensor(0).unwrap();
///     query[0] = 4.0;
///     std::hint::black_box(output.values());
/// }
/// ```
pub struct RetainedOutput<'a> {
    inner: BorrowedOutput<'a, Scores>,
    operation: usize,
}
impl RetainedOutput<'_> {
    pub fn values(&self) -> &[f64] { &self.inner.values().values }
    pub fn original(&self) -> &ClearInputs { self.inner.original() }
    pub fn operation(&self) -> Ref<'_, ModelOperation> {
        Ref::map(self.original().operations(), |ops| &ops[self.operation])
    }
    pub fn status(&self) -> Status { Status::ClearReference }
}
impl RetainedProducer for RetainedOutput<'_> {
    fn original(&self) -> &ClearInputs { self.inner.original() }
    fn operation_index(&self) -> usize { self.operation }
}

/// Query/key IDs select immutable ORIGINAL tensors, never separate borrowed data.
/// Ordered names and shapes must match the ORIGINAL bounded AttentionScores DTO.
/// All attempts, including failures and outputs already dropped, share its ledger.
pub fn execute(
    input: &ClearInputs,
    operation: usize,
    query: usize,
    key: usize,
    geometry: core::Geometry,
    limits: core::Limits,
) -> Result<RetainedOutput<'_>, Refusal> {
    let ticket = input.begin_attempt()?;
    let ((payment, quote), prepared) = prepay(ticket, geometry, limits)?;
    let phase = prepared.reserve::<Scores>(producer_bytes(quote))?;
    validate(input, operation, query, key, geometry)?;
    run(input, operation, query, key, payment, quote, phase)
}

fn prepay(
    ticket: owner::Attempt<'_>,
    geometry: core::Geometry,
    limits: core::Limits,
) -> Result<((core::Prepayment, core::Quote), Prepared<'_>), Refusal> {
    let ((payment, quote), prepared) = ticket.with_ledger(|ledger| {
        // Pay bounded metadata checks even if the subsequent quote refuses.
        ledger.debit(CONTROL, 0).map_err(|_| owner::Error::Work)?;
        let quote = geometry.quote(limits).map_err(|_| owner::Error::Work)?;
        // Extra copy, initialization, and redundant zeroization on success/error
        // and final result drop. Core quote still covers its own complete region.
        let cells = quote.workspace_cells + 2 * quote.output_cells;
        ledger.debit(4 * cells, 0).map_err(|_| owner::Error::Work)?;
        let payment = core::prepay(geometry, limits, ledger, CONTROL)
            .map_err(|_| owner::Error::Work)?;
        Ok((payment, quote))
    })?;
    Ok(((payment, quote), prepared))
}

fn producer_bytes(quote: core::Quote) -> usize {
    quote.bytes + quote.output_cells * 8 + size_of::<Scores>()
}

fn validate(
    input: &ClearInputs,
    operation: usize,
    query: usize,
    key: usize,
    geometry: core::Geometry,
) -> Result<(), Refusal> {
    // with_ledger's RefMut has ENDED. Metadata checks allocate nothing; all
    // payload factories follow complete work payment and phase reservations.
    let operations = input.operations();
    let op = operations.get(operation).ok_or(Refusal::OperationPolicy)?;
    let tensors = input.tensors();
    let q = tensors.get(query).ok_or(Refusal::OperationPolicy)?;
    let k = tensors.get(key).ok_or(Refusal::OperationPolicy)?;
    let shape = [1, 1, geometry.rows(), geometry.width()];
    if query == key
        || q.shape() != shape || k.shape() != shape
        || op.operator != ModelOperator::AttentionScores
        || op.inputs.len() != 2
        || op.inputs[0] != q.name() || op.inputs[1] != k.name()
        || op.output_shape.as_slice() != [1, 1, geometry.rows() as u64, geometry.rows() as u64]
        || op.state_kind.is_some()
        || op.attributes.as_object().is_none_or(|attrs| {
            attrs.len() != 1 || attrs.get("group_size").and_then(|v| v.as_u64()) != Some(1)
        })
    { return Err(Refusal::OperationPolicy); }
    Ok(())
}

fn run<'a>(
    input: &'a ClearInputs,
    operation: usize,
    query: usize,
    key: usize,
    payment: core::Prepayment,
    quote: core::Quote,
    phase: Phase<'a, Scores>,
) -> Result<RetainedOutput<'a>, Refusal> {
    let query_values = input.f64_tensor(query)?;
    let key_values = input.f64_tensor(key)?;
    if query_values.len() != quote.input_cells || key_values.len() != quote.input_cells {
        return Err(Refusal::OperationPolicy);
    }
    // Install the cleanup owner before any later allocation can fail.
    let mut scores = Scores { workspace: Vec::new(), scratch_scores: Vec::new(), values: Vec::new() };
    initialize(&mut scores.workspace, quote.workspace_cells)?;
    initialize(&mut scores.scratch_scores, quote.output_cells)?;
    initialize(&mut scores.values, quote.output_cells)?;
    let mut state = State::Fresh; // PRIVATE invocation state, never replaces work.
    let output = core::Reference::new_prepaid(payment, &query_values, &key_values,
        &mut scores.workspace, &mut scores.scratch_scores, &mut state)?.execute()?;
    scores.values.copy_from_slice(output.values());
    drop(output); // wipes core workspace/output BEFORE phase can release charge
    Ok(RetainedOutput { inner: phase.finish(scores), operation })
}

/// Explicit CLEAR binary64 multiplication, not a protected scale policy or an
/// inferred `1/sqrt(head_dim)` profile. The original consumer must contain only
/// the public `factor` attribute with exactly these binary64 bits.
#[derive(Clone, Copy)]
pub struct Binary64PublicScale {
    pub factor: f64,
}

#[derive(Clone, Copy)]
pub struct ScoreScaleOperations {
    pub scores: usize,
    pub scale: usize,
    pub query: usize,
    pub key: usize,
}

// Cross-namespace comparisons8192 + edge256 + schema1024 + binding/control768.
// Builder bounds tensor count16, names256 and attributes8 before this traversal.
const SCALE_CONTROL: usize = 8192 + 256 + 1024 + 768;
// Conservatively covers both quoted/retained control wrappers, inventoried below.
const SCALE_METADATA: usize = 1024;

// O(1) metadata quote, before the scan: each of the at-most-eight operation IDs
// is compared to Q and K. One length check plus at most the longer source name's
// bytes per comparison. Invalid IDs use the original builder name cap instead.
fn source_namespace_work(input: &ClearInputs, operations: ScoreScaleOperations) -> usize {
    let tensors = input.tensors();
    let longest = [operations.query, operations.key].into_iter()
        .map(|id| tensors.get(id).map_or(256, |t| t.name().len())).max().unwrap_or(256);
    2 * input.operations().len() * (1 + longest)
}

// All operator/schema/factor/shape/ordered-edge policy stays in this adapter.
// Called only after the original ledger paid both SCALE_CONTROL and the Q/K
// namespace scan, and before either numeric factory or phase construction.
fn validate_scale_edge(
    input: &ClearInputs,
    operations: ScoreScaleOperations,
    shape: [u64; 4],
    factor: f64,
) -> Result<ValueEdge<'_>, Refusal> {
    let ops = input.operations();
    let p = ops.get(operations.scores).ok_or(Refusal::OperationPolicy)?;
    let c = ops.get(operations.scale).ok_or(Refusal::OperationPolicy)?;
    let tensors = input.tensors();
    let q = tensors.get(operations.query).ok_or(Refusal::OperationPolicy)?;
    let k = tensors.get(operations.key).ok_or(Refusal::OperationPolicy)?;
    if !factor.is_finite()
        || operations.scores >= operations.scale
        || p.operator != ModelOperator::AttentionScores
        || c.operator != ModelOperator::AttentionScale
        || c.inputs.len() != 1 || c.inputs[0] != p.id
        || p.output_shape.as_slice() != shape || c.output_shape.as_slice() != shape
        || c.layer != p.layer || p.state_kind.is_some() || c.state_kind.is_some()
        || c.attributes.as_object().is_none_or(|attrs| {
            attrs.len() != 1 || attrs.get("factor").and_then(|v| v.as_f64())
                .is_none_or(|n| n.to_bits() != factor.to_bits())
        })
        || tensors.iter().any(|t| t.name() == p.id || t.name() == c.id)
        // A source tensor name must not also designate ANY original operation,
        // including unrelated/later graph operations with matching Q/K names.
        || ops.iter().any(|op| op.id == q.name() || op.id == k.name())
    { return Err(Refusal::OperationPolicy); }
    ValueEdge::new(input, operations.scores, operations.scale).ok_or(Refusal::OperationPolicy)
}

struct ScaledValues(Vec<f64>);
impl Zeroize for ScaledValues {
    fn zeroize(&mut self) {
        #[cfg(test)]
        observe::partial(self.0.iter().any(|&v| v != 0.0));
        self.0.as_mut_slice().zeroize();
    }
}
impl Drop for ScaledValues {
    fn drop(&mut self) {
        self.zeroize();
        #[cfg(test)]
        observe::retired(false, self.0.iter(), self.0.len(), self.0.capacity());
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompositionQuote {
    pub original_bytes: usize,
    /// Both simultaneously retained phase charges, including guards/metadata.
    pub phase_bytes: usize,
    /// Two burned attempts plus all producer/consumer work, even on cancellation.
    pub additional_work: usize,
    /// Original construction, builder writes, prior attempts and this composition.
    pub cumulative_work: usize,
}

/// Non-cloneable, one-use admission. Dropping it refunds bytes, never work.
/// No score/consumer buffers have been allocated yet. Neither phase can escape.
pub struct QuotedScaleComposition<'a> {
    edge: ValueEdge<'a>,
    factor: f64,
    operations: ScoreScaleOperations,
    payment: core::Prepayment,
    kernel: core::Quote,
    resources: CompositionQuote,
    producer: Phase<'a, Scores>,
    consumer: Phase<'a, ScaledValues>,
}
impl QuotedScaleComposition<'_> {
    pub fn resources(&self) -> CompositionQuote { self.resources }
}

fn charged_phase<T: Zeroize>(bytes: usize) -> Result<usize, Refusal> {
    bytes.checked_add(size_of::<BorrowedOutput<'_, T>>())
        .and_then(|n| n.checked_add(size_of::<Prepared<'_>>()))
        .and_then(|n| n.checked_add(size_of::<pllm_garble::shared_allocation::ExternalCharge<'_>>()))
        .ok_or(Refusal::Resource)
}

/// Quote the WHOLE original Q/K -> ordered Algorithm 5 -> retained scores ->
/// explicit public binary64 multiply slice before either numeric factory.
/// Original MAX_WORK=65536, owner caps and per-phase caps remain unchanged.
/// No softmax, causal mask, SDK routing, HE, source/import or security authority.
pub fn quote_scaled(
    input: &ClearInputs,
    operations: ScoreScaleOperations,
    geometry: core::Geometry,
    limits: core::Limits,
    policy: Binary64PublicScale,
) -> Result<QuotedScaleComposition<'_>, Refusal> {
    let before = input.consumed_work();
    let producer_ticket = input.begin_attempt()?;
    let consumer_ticket = input.begin_attempt()?;
    let ((payment, kernel), producer) = prepay(producer_ticket, geometry, limits)?;
    let namespace_work = source_namespace_work(input, operations);
    let (_, consumer) = consumer_ticket.with_ledger(|ledger| {
        // Eight visits per result cover initialize, read/finite, multiplication,
        // product finite check and all success/error/cancellation retirement.
        let work = kernel.output_cells.checked_mul(8)
            .and_then(|n| n.checked_add(SCALE_CONTROL))
            .and_then(|n| n.checked_add(namespace_work)).ok_or(owner::Error::Work)?;
        ledger.debit(work, 0).map_err(|_| owner::Error::Work)
    })?;
    validate(input, operations.scores, operations.query, operations.key, geometry)?;
    let edge = validate_scale_edge(input, operations,
        [1, 1, geometry.rows() as u64, geometry.rows() as u64], policy.factor)?;
    if size_of::<QuotedScaleComposition<'_>>() + size_of::<RetainedScaledOutput<'_>>()
        + size_of::<ValueEdge<'_>>() > SCALE_METADATA {
        return Err(Refusal::Resource);
    }
    let consumer_bytes = kernel.output_cells.checked_mul(8)
        .and_then(|n| n.checked_add(size_of::<ScaledValues>() + SCALE_METADATA))
        .ok_or(Refusal::Resource)?;
    let phase_bytes = charged_phase::<Scores>(producer_bytes(kernel))?
        .checked_add(charged_phase::<ScaledValues>(consumer_bytes)?)
        .ok_or(Refusal::Resource)?;
    let resources = CompositionQuote {
        original_bytes: input.charged_bytes(), phase_bytes,
        additional_work: input.consumed_work() - before,
        cumulative_work: input.consumed_work(),
    };
    // Both reservations precede either factory. Failed second reservation drops
    // the first; all attempt/quote work stays burned on the original ledger.
    let producer = producer.reserve(producer_bytes(kernel))?;
    #[cfg(test)]
    observe::reservation();
    let consumer = consumer.reserve(consumer_bytes)?;
    #[cfg(test)]
    observe::reservation();
    Ok(QuotedScaleComposition { edge, factor: policy.factor, operations, payment, kernel, resources, producer, consumer })
}

/// Actual producer allocation remains owned until this output drops. No copied
/// ClearInputs, caller-provided derived values, mutable alias or replay token.
pub struct RetainedScaledOutput<'a> {
    // Drop scaled storage/charge before retiring the actual upstream allocation.
    scaled: BorrowedOutput<'a, ScaledValues>,
    binding: BoundValue<'a, RetainedOutput<'a>>,
}
impl<'a> RetainedScaledOutput<'a> {
    pub fn values(&self) -> &[f64] { &self.scaled.values().0 }
    pub fn scores(&self) -> &RetainedOutput<'a> { self.binding.producer() }
    pub fn original(&self) -> &ClearInputs { self.scaled.original() }
    pub fn operation(&self) -> Ref<'_, ModelOperation> {
        Ref::map(self.original().operations(), |ops| &ops[self.binding.consumer()])
    }
    pub fn status(&self) -> Status { Status::ClearReference }
}

/// Consumes the admitted slice once; never publishes a tentative scaled prefix.
/// ```compile_fail,E0382
/// use pllm_compiler::column_product_reference::{execute_scaled, QuotedScaleComposition};
/// fn replay(quote: QuotedScaleComposition<'_>) {
///     let _ = execute_scaled(quote);
///     let _ = execute_scaled(quote);
/// }
/// ```
/// Original owner cannot retire behind either retained allocation.
/// ```compile_fail,E0505
/// use pllm_compiler::{clear_input_owner::ClearInputs, column_product_reference::*};
/// use pllm_core::column_product_reference::{Geometry, Limits};
/// fn retire(input: ClearInputs, ops: ScoreScaleOperations, g: Geometry, limits: Limits) {
///     let quote = quote_scaled(&input, ops, g, limits, Binary64PublicScale { factor: 0.5 }).unwrap();
///     let output = execute_scaled(quote).unwrap();
///     drop(input);
///     std::hint::black_box(output.values());
/// }
/// ```
/// Derived scores never expose mutable aliases.
/// ```compile_fail,E0594
/// use pllm_compiler::column_product_reference::RetainedScaledOutput;
/// fn mutate(output: &RetainedScaledOutput<'_>) {
///     output.scores().values()[0] = 9.0;
/// }
/// ```
pub fn execute_scaled(quote: QuotedScaleComposition<'_>) -> Result<RetainedScaledOutput<'_>, Refusal> {
    let QuotedScaleComposition { edge, factor, operations, payment, kernel, producer, consumer, .. } = quote;
    let input = edge.original();
    let scores = run(input, operations.scores, operations.query, operations.key, payment, kernel, producer)?;
    let binding = edge.bind(scores).map_err(|_| Refusal::OperationPolicy)?;
    // Cleanup owner installed before the first fallible consumer allocation.
    let mut scaled = ScaledValues(Vec::new());
    initialize(&mut scaled.0, kernel.output_cells)?;
    for (out, &score) in scaled.0.iter_mut().zip(binding.producer().values()) {
        let product = score * factor; // separate binary64 multiply
        if !score.is_finite() || !product.is_finite() {
            return Err(core::Error::Layout(core::LayoutError::NonFinite).into());
        }
        *out = product;
    }
    Ok(RetainedScaledOutput { scaled: consumer.finish(scaled), binding })
}

// Fixed-size, thread-local observation only; never an admission/authority input,
// never a heap event log. Hooks run on the real allocation/Drop/failure paths.
#[cfg(test)]
mod observe {
    use std::cell::Cell;
    #[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
    pub(super) struct Counts {
        pub factories: usize,
        pub reservations: usize,
        pub scores_retired: usize,
        pub scaled_retired: usize,
        pub retired_cells: usize,
        pub dirty_retirements: usize,
        pub partial_scaled_wipes: usize,
    }
    thread_local! { static COUNTS: Cell<Counts> = Cell::new(Counts::default()); }
    pub(super) fn snapshot() -> Counts { COUNTS.with(Cell::get) }
    fn update(f: impl FnOnce(&mut Counts)) {
        COUNTS.with(|counts| { let mut value = counts.get(); f(&mut value); counts.set(value); });
    }
    pub(super) fn factory() { update(|c| c.factories += 1); }
    pub(super) fn reservation() { update(|c| c.reservations += 1); }
    pub(super) fn partial(nonzero: bool) {
        if nonzero { update(|c| c.partial_scaled_wipes += 1); }
    }
    pub(super) fn retired<'a>(scores: bool, mut values: impl Iterator<Item = &'a f64>, len: usize, capacity: usize) {
        let dirty = len != capacity || values.any(|&v| v != 0.0);
        update(|c| {
            if scores { c.scores_retired += 1; } else { c.scaled_retired += 1; }
            c.retired_cells += len;
            c.dirty_retirements += usize::from(dirty);
        });
    }
}

#[cfg(test)]
#[path = "column_product_reference_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "column_product_composition_tests.rs"]
mod composition_tests;
