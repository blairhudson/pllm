//! Algorithm 5 CLEAR reference adapter over actual retained clear-input custody.
//! Not CKKS packing, checkpoint authentication, or full NEXUS adoption.
//! No caller ledger, lifecycle state, guard, DTO, or buffer adoption route.
use crate::clear_input_owner::{self as owner, BorrowedOutput, ClearInputs};
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
    fn drop(&mut self) { self.zeroize(); }
}
fn initialize(values: &mut Vec<f64>, length: usize) -> Result<(), Refusal> {
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
    // with_ledger's RefMut has ENDED. No payload is borrowed or allocated inside
    // that callback. Reservation precedes metadata validation and every factory.
    let phase_bytes = quote.bytes + quote.output_cells * 8 + size_of::<Scores>();
    let phase = prepared.reserve::<Scores>(phase_bytes)?;
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

#[cfg(test)]
#[path = "column_product_reference_tests.rs"]
mod tests;
