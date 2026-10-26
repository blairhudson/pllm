use super::*;
use crate::clear_input_owner::{Attribute, Caps, ClearInputBuilder, MAX_WORK};
use pllm_garble::shared_allocation::OwnedBudget;

fn geometry() -> core::Geometry { core::Geometry::new(4, 3).unwrap() }
fn limits() -> core::Limits {
    core::Limits { max_buffer_elements: 1 << 20, max_work: core::MAX_VISITS }
}
fn caps() -> Caps { Caps { f64_values: 32, u64_values: 0, tensors: 2, operations: 1 } }
fn fixture(work: usize, budget: usize, overflow: bool, reversed: bool) -> ClearInputs {
    let mut builder = ClearInputBuilder::new(OwnedBudget::new(budget).unwrap(), caps(), work).unwrap();
    builder.add_f64_tensor("query", &[1,1,4,3], &[1.,2.,3.,4.,5.,6.,-1.,-2.,-3.,7.,8.,-8.]).unwrap();
    let mut key = [2.,-1.,4.,3.,0.,-2.,1.,8.,-7.,-3.,5.,6.];
    if overflow { key[6] = f64::MAX; }
    builder.add_f64_tensor("key", &[1,1,4,3], &key).unwrap();
    let names = if reversed { ["key","query"] } else { ["query","key"] };
    let op = builder.add_operation("scores", ModelOperator::AttentionScores, None, &names, &[1,1,4,4]).unwrap();
    builder.set_attribute(op, "group_size", Attribute::Unsigned(1)).unwrap();
    builder.finish().unwrap()
}
fn generous(overflow: bool, reversed: bool) -> ClearInputs {
    fixture(MAX_WORK, 1 << 20, overflow, reversed)
}
fn attempt_work() -> usize {
    let q = geometry().quote(limits()).unwrap();
    32 + CONTROL + 4 * (q.workspace_cells + 2 * q.output_cells) + q.visits + CONTROL
}
fn phase_bytes() -> usize {
    let q = geometry().quote(limits()).unwrap();
    q.bytes + q.output_cells * 8 + size_of::<Scores>()
        + size_of::<BorrowedOutput<'_, Scores>>() + size_of::<owner::Prepared<'_>>()
        + size_of::<pllm_garble::shared_allocation::ExternalCharge<'_>>()
}
#[test]
fn original_owned_operation_order_shape_custody_and_integer_parity() {
    let input = generous(false, false);
    let original_bytes = input.charged_bytes();
    let qp = input.f64_tensor(0).unwrap().as_ptr();
    let kp = input.f64_tensor(1).unwrap().as_ptr();
    let op_ptr = &input.operations()[0] as *const ModelOperation;
    let before = input.remaining_work();
    let output = execute(&input, 0, 0, 1, geometry(), limits()).unwrap();
    assert!(std::ptr::eq(output.original(), &input));
    assert!(std::ptr::eq(&*output.operation(), op_ptr));
    assert_eq!(output.operation().inputs, ["query", "key"]);
    assert_eq!(output.operation().output_shape, [1,1,4,4]);
    assert_eq!(output.status(), Status::ClearReference);
    let query = input.f64_tensor(0).unwrap();
    let key = input.f64_tensor(1).unwrap();
    for row in 0..4 {
        for col in 0..4 {
            // Independent integer oracle, not float-reduction-order claims.
            let expected: i64 = (0..3).map(|j| query[row*3+j] as i64 * key[col*3+j] as i64).sum();
            assert_eq!(output.values()[row*4+col], expected as f64);
        }
    }
    assert_eq!(input.remaining_work(), before - attempt_work());
    assert_eq!(input.charged_bytes(), original_bytes);
    assert_eq!(input.f64_tensor(0).unwrap().as_ptr(), qp);
    assert_eq!(input.f64_tensor(1).unwrap().as_ptr(), kp);
    assert!(output.inner.values().workspace.iter().chain(&output.inner.values().scratch_scores).all(|&x| x == 0.));
    assert_eq!(output.inner.values().values.capacity(), 16);
    assert_eq!(output.inner.phase_bytes(), phase_bytes());
}
#[test]
fn owner_construction_plus_one_attempt_exact_and_one_short_work() {
    let probe = generous(false, false);
    let construction = probe.consumed_work();
    assert!(construction > ClearInputBuilder::resources(caps()).unwrap().construction_work);
    for short in [0, 1] {
        let input = fixture(construction + attempt_work() - short, 1 << 20, false, false);
        let result = execute(&input, 0, 0, 1, geometry(), limits());
        if short == 0 {
            let output = result.unwrap();
            assert_eq!(input.remaining_work(), 0);
            drop(output);
            assert_eq!(input.remaining_work(), 0);
        } else {
            assert!(matches!(result, Err(Refusal::Owner(owner::Error::Work))));
            assert!(input.consumed_work() > construction);
        }
    }
}
#[test]
fn full_owner_and_phase_storage_exact_one_short_no_unpaid_factory() {
    // new() consumes/drops the transient OwnedBudget root handle after transfer.
    // Its inline charge is released; it must not create slack in the exact gate.
    let minimum = ClearInputBuilder::resources(caps()).unwrap().minimum_fresh_budget_bytes
        - size_of::<OwnedBudget>();
    for short in [0, 1] {
        let input = fixture(MAX_WORK, minimum + phase_bytes() - short, false, false);
        let before = input.remaining_work();
        let result = execute(&input, 0, 0, 1, geometry(), limits());
        if short == 0 { drop(result.unwrap()); }
        else { assert!(matches!(result, Err(Refusal::Owner(owner::Error::Resource)))); }
        assert_eq!(input.remaining_work(), before - attempt_work());
        assert_eq!(&*input.f64_tensor(0).unwrap(), &[1.,2.,3.,4.,5.,6.,-1.,-2.,-3.,7.,8.,-8.]);
    }
}
#[test]
fn sequential_attempts_and_failed_validation_burn_original_work() {
    let input = generous(false, false);
    let before = input.remaining_work();
    for iteration in 1..=3 {
        drop(execute(&input, 0, 0, 1, geometry(), limits()).unwrap());
        assert_eq!(input.remaining_work(), before - iteration * attempt_work());
    }
    let failed = input.remaining_work();
    assert!(matches!(execute(&input, 0, 1, 0, geometry(), limits()), Err(Refusal::OperationPolicy)));
    assert_eq!(input.remaining_work(), failed - attempt_work());
    drop(execute(&input, 0, 0, 1, geometry(), limits()).unwrap());
    assert_eq!(input.remaining_work(), failed - 2 * attempt_work());
    let reversed = generous(false, true);
    assert!(matches!(execute(&reversed, 0, 0, 1, geometry(), limits()), Err(Refusal::OperationPolicy)));
}
#[test]
fn fresh_external_counters_cannot_bypass_retained_construction_and_attempts() {
    let probe = generous(false, false);
    let work = probe.consumed_work() + attempt_work();
    let input = fixture(work, 1 << 20, false, false);
    drop(execute(&input, 0, 0, 1, geometry(), limits()).unwrap());
    for _ in 0..3 {
        // Neither fresh value can enter the adapter signature.
        let fresh = pllm_core::row_fold::WorkLedger::new(MAX_WORK, MAX_WORK);
        let unrelated = OwnedBudget::new(1 << 20).unwrap();
        assert_eq!(fresh.remaining(), MAX_WORK);
        std::hint::black_box(unrelated);
        assert!(matches!(execute(&input, 0, 0, 1, geometry(), limits()), Err(Refusal::Owner(owner::Error::Work))));
        assert_eq!(input.remaining_work(), 0);
    }
}
#[test]
fn failed_arithmetic_retains_original_backing_and_never_refunds_attempts() {
    let input = generous(true, false);
    let pointer = input.f64_tensor(1).unwrap().as_ptr();
    let bytes = input.charged_bytes();
    let before = input.remaining_work();
    for iteration in 1..=2 {
        assert!(matches!(execute(&input, 0, 0, 1, geometry(), limits()), Err(Refusal::Arithmetic(core::Error::Layout(core::LayoutError::NonFinite)))));
        assert_eq!(input.remaining_work(), before - iteration * attempt_work());
        assert_eq!(input.f64_tensor(1).unwrap().as_ptr(), pointer);
        assert_eq!(input.f64_tensor(1).unwrap()[6], f64::MAX);
        assert_eq!(input.charged_bytes(), bytes);
    }
}
#[test]
fn phase_payload_zeroizes_entire_initialized_capacity_without_releasing_shape() {
    let mut scores = Scores { workspace: vec![7.; 28], scratch_scores: vec![9.; 16], values: vec![11.; 16] };
    scores.zeroize(); // same routine used on failure Drop and owned output Drop
    assert_eq!(scores.workspace.len(), scores.workspace.capacity());
    assert_eq!(scores.scratch_scores.len(), scores.scratch_scores.capacity());
    assert_eq!(scores.values.len(), scores.values.capacity());
    assert!(scores.workspace.iter().chain(&scores.scratch_scores).chain(&scores.values).all(|&x| x == 0.));
}
#[test]
fn large_head_refuses_tighter_original_owner_work_cap_without_backing_mutation() {
    let input = generous(false, false);
    let before = input.remaining_work();
    let g = core::Geometry::new(128, 64).unwrap();
    assert!(matches!(execute(&input, 0, 0, 1, g, limits()), Err(Refusal::Owner(owner::Error::Work))));
    assert!(input.remaining_work() < before);
    assert_eq!(input.f64_tensor(0).unwrap()[0], 1.0);
}

#[test]
fn owned_literal_rounding_sensitive_nonfused_and_ascending_results() {
    for (query, key, expected) in [
        (vec![-1.0, 1.0000000074505806], vec![1.0, 0.9999999925494194], 0.0_f64),
        (vec![18014398509481984.0, -18014398509481984.0, 1.0], vec![1.0; 3], 1.0_f64),
    ] {
        let width = query.len();
        let mut builder = ClearInputBuilder::new(OwnedBudget::new(1 << 20).unwrap(), caps(), MAX_WORK).unwrap();
        builder.add_f64_tensor("query", &[1,1,1,width], &query).unwrap();
        builder.add_f64_tensor("key", &[1,1,1,width], &key).unwrap();
        let op = builder.add_operation("scores", ModelOperator::AttentionScores, None, &["query","key"], &[1,1,1,1]).unwrap();
        builder.set_attribute(op, "group_size", Attribute::Unsigned(1)).unwrap();
        let input = builder.finish().unwrap();
        let output = execute(&input, op, 0, 1, core::Geometry::new(1,width).unwrap(), limits()).unwrap();
        // Independent literals: neither candidate output nor a float reducer is
        // used to construct the oracle. Core tests distinguish FMA/reassociation.
        assert_eq!(output.values()[0].to_bits(), expected.to_bits());
    }
}

#[test]
fn retained_output_charge_blocks_overlap_then_releases_bytes_not_work() {
    let minimum = ClearInputBuilder::resources(caps()).unwrap().minimum_fresh_budget_bytes
        - size_of::<OwnedBudget>();
    let input = fixture(MAX_WORK, minimum + phase_bytes(), false, false);
    let before = input.remaining_work();
    let first = execute(&input, 0, 0, 1, geometry(), limits()).unwrap();
    assert!(matches!(execute(&input, 0, 0, 1, geometry(), limits()), Err(Refusal::Owner(owner::Error::Resource))));
    assert_eq!(input.remaining_work(), before - 2 * attempt_work());
    assert_eq!(first.values()[0], 12.0);
    drop(first);
    drop(execute(&input, 0, 0, 1, geometry(), limits()).unwrap());
    assert_eq!(input.remaining_work(), before - 3 * attempt_work());
}

#[test]
fn bounded_original_descriptor_rejects_wrong_operator_shape_or_attributes() {
    for invalid in 0..4 {
        let mut builder = ClearInputBuilder::new(OwnedBudget::new(1 << 20).unwrap(), caps(), MAX_WORK).unwrap();
        builder.add_f64_tensor("query", &[1,1,4,3], &[1.0;12]).unwrap();
        builder.add_f64_tensor("key", &[1,1,4,3], &[2.0;12]).unwrap();
        let operator = if invalid == 0 { ModelOperator::Linear } else { ModelOperator::AttentionScores };
        let shape = if invalid == 1 { [1,1,4,3] } else { [1,1,4,4] };
        let op = builder.add_operation("scores", operator, None, &["query","key"], &shape).unwrap();
        builder.set_attribute(op, "group_size", Attribute::Unsigned(if invalid == 2 { 2 } else { 1 })).unwrap();
        if invalid == 3 { builder.set_attribute(op, "extra", Attribute::Bool(true)).unwrap(); }
        let input = builder.finish().unwrap();
        let before = input.remaining_work();
        assert!(matches!(execute(&input, op, 0, 1, geometry(), limits()), Err(Refusal::OperationPolicy)));
        assert_eq!(input.remaining_work(), before - attempt_work());
    }
}
