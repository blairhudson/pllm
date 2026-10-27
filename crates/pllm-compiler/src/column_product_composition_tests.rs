//! Independent dense integer row-pair oracle, never Algorithm 5 replay.
use super::*;
use crate::clear_input_owner::{Attribute, Caps, ClearInputBuilder, MAX_WORK};
use pllm_garble::shared_allocation::OwnedBudget;

const QUERY: [f64; 4] = [1., -2., 3., 4.];
const KEY: [f64; 4] = [5., 6., -7., 8.];
fn geometry() -> core::Geometry { core::Geometry::new(2, 2).unwrap() }
fn limits() -> core::Limits {
    core::Limits { max_buffer_elements: 1024, max_work: core::MAX_VISITS }
}
fn ops() -> ScoreScaleOperations {
    ScoreScaleOperations { scores: 0, scale: 1, query: 0, key: 1 }
}
fn caps() -> Caps { Caps { f64_values: 8, u64_values: 0, tensors: 2, operations: 3 } }

// Cases change original descriptors, never mutate/adopt a derived buffer.
fn fixture(work: usize, budget: usize, case: usize, factor: f64) -> ClearInputs {
    let mut b = ClearInputBuilder::new(OwnedBudget::new(budget).unwrap(), caps(), work).unwrap();
    let mut q = QUERY;
    let mut k = KEY;
    if case == 10 { q[0] = f64::MAX; } // producer product overflow
    if case == 11 { q = [1., 0., 1., 0.]; k = [1., 0., f64::MAX, 0.]; }
    let qname = if case == 9 { "scores" } else { "query" };
    b.add_f64_tensor(qname, &[1,1,2,2], &q).unwrap();
    b.add_f64_tensor("key", &[1,1,2,2], &k).unwrap();
    let names = if case == 8 { ["key", qname] } else { [qname, "key"] };
    let score = b.add_operation("scores", ModelOperator::AttentionScores, Some(0), &names, &[1,1,2,2]).unwrap();
    b.set_attribute(score, "group_size", Attribute::Unsigned(1)).unwrap();
    let operator = if case == 1 { ModelOperator::Scale } else { ModelOperator::AttentionScale };
    let edge = if case == 2 { "other_scores" } else { "scores" };
    let shape = if case == 3 { [1,1,2,1] } else { [1,1,2,2] };
    let layer = if case == 4 { Some(1) } else { Some(0) };
    let scale = b.add_operation("scaled", operator, layer, &[edge], &shape).unwrap();
    if case == 5 {
        b.set_attribute(scale, "head_dim", Attribute::Unsigned(2)).unwrap();
    } else {
        b.set_attribute(scale, "factor", Attribute::Real(factor)).unwrap();
    }
    if case == 6 { b.set_attribute(scale, "extra", Attribute::Bool(true)).unwrap(); }
    let other_name = match case { 12 => "query", 13 => "key", _ => "other_scores" };
    let other = b.add_operation(other_name, ModelOperator::AttentionScores, Some(0), &[qname,"key"], &[1,1,2,2]).unwrap();
    b.set_attribute(other, "group_size", Attribute::Unsigned(1)).unwrap();
    b.finish().unwrap()
}
fn fresh(case: usize, factor: f64) -> ClearInputs { fixture(MAX_WORK, 1 << 20, case, factor) }
fn quote(input: &ClearInputs, factor: f64) -> Result<QuotedScaleComposition<'_>, Refusal> {
    quote_scaled(input, ops(), geometry(), limits(), Binary64PublicScale { factor })
}
fn work() -> usize {
    let q = geometry().quote(limits()).unwrap();
    64 + 2 * CONTROL + 4 * (q.workspace_cells + 2 * q.output_cells)
        + q.visits + SCALE_CONTROL + 8 * q.output_cells
        // Independent source-scan oracle: 3 original ops, two names, maximum
        // source-name length5, plus one length check per comparison.
        + 3 * 2 * (1 + 5)
}
fn construction_work(case: usize) -> usize {
    let qname = if case == 9 { "scores" } else { "query" };
    let other_name = match case { 12 => "query", 13 => "key", _ => "other_scores" };
    // Original construction quote plus independently enumerated builder writes.
    // No output/candidate consumed-work counter constructs this oracle.
    ClearInputBuilder::resources(caps()).unwrap().construction_work
        + (512 + 2 * 4 + qname.len() * 2)
        + (512 + 2 * 4 + "key".len() * 2)
        + (1536 + "scores".len() * 4 + qname.len() + "key".len())
        + (1536 + "scaled".len() * 4 + "scores".len())
        + (1536 + other_name.len() * 4 + qname.len() + "key".len())
        + 3 * 1024 + if case == 6 { 1024 } else { 0 }
}
fn exact_bytes() -> usize {
    let probe = fresh(0, 0.5);
    let phases = quote(&probe, 0.5).unwrap().resources().phase_bytes;
    ClearInputBuilder::resources(caps()).unwrap().minimum_fresh_budget_bytes
        - size_of::<OwnedBudget>() + phases
}
fn nonfinite(result: Result<RetainedScaledOutput<'_>, Refusal>) {
    assert!(matches!(result, Err(Refusal::Arithmetic(core::Error::Layout(core::LayoutError::NonFinite)))));
}

#[test]
fn dense_all_outputs_dyadic_multiply_and_original_allocation_identity() {
    for factor in [0.5, -0.25, 2.0, 0.0] {
        let input = fresh(0, factor);
        let before = input.remaining_work();
        let qp = input.f64_tensor(0).unwrap().as_ptr();
        let kp = input.f64_tensor(1).unwrap().as_ptr();
        let score_op = &input.operations()[0] as *const ModelOperation;
        let scale_op = &input.operations()[1] as *const ModelOperation;
        let quoted = quote(&input, factor).unwrap();
        let resources = quoted.resources();
        assert_eq!(resources.additional_work, work());
        assert_eq!(resources.cumulative_work, input.consumed_work());
        assert!(resources.cumulative_work <= MAX_WORK);
        assert_eq!(resources.original_bytes, input.charged_bytes());
        assert_eq!(input.remaining_work(), before - work());
        let output = execute_scaled(quoted).unwrap();
        assert!(std::ptr::eq(output.original(), &input));
        assert!(std::ptr::eq(output.scores().original(), &input));
        assert!(std::ptr::eq(&*output.scores().operation(), score_op));
        assert!(std::ptr::eq(&*output.operation(), scale_op));
        assert_eq!(output.status(), Status::ClearReference);
        assert_ne!(output.scores().values().as_ptr(), output.values().as_ptr());
        for row in 0..2 {
            for col in 0..2 {
                let dense: i64 = (0..2).map(|j| QUERY[2*row+j] as i64 * KEY[2*col+j] as i64).sum();
                assert_eq!(output.scores().values()[2*row+col], dense as f64);
                assert_eq!(output.values()[2*row+col].to_bits(), ((dense as f64) * factor).to_bits());
            }
        }
        assert_eq!(output.scores().inner.phase_bytes() + output.scaled.phase_bytes(), resources.phase_bytes);
        assert!(output.scores().inner.values().workspace.iter()
            .chain(&output.scores().inner.values().scratch_scores).all(|&x| x == 0.));
        assert_eq!(&*input.f64_tensor(0).unwrap(), &QUERY);
        assert_eq!(&*input.f64_tensor(1).unwrap(), &KEY);
        assert_eq!(input.f64_tensor(0).unwrap().as_ptr(), qp);
        assert_eq!(input.f64_tensor(1).unwrap().as_ptr(), kp);
        assert_eq!(input.remaining_work(), before - work());
        drop(output);
        assert_eq!(input.remaining_work(), before - work());
    }
}

#[test]
fn private_binding_rejects_same_shaped_cross_owner_and_other_graph_operation() {
    let input = fresh(0, 0.5);
    let foreign = fresh(0, 0.5);
    let edge = validate_scale_edge(&input, ops(), [1,1,2,2], 0.5).unwrap();
    let wrong_owner = execute(&foreign, 0, 0, 1, geometry(), limits()).unwrap();
    assert!(matches!(edge.bind(wrong_owner), Err(())));
    let edge = validate_scale_edge(&input, ops(), [1,1,2,2], 0.5).unwrap();
    let wrong_operation = execute(&input, 2, 0, 1, geometry(), limits()).unwrap();
    assert!(matches!(edge.bind(wrong_operation), Err(())));
    // Positive control: ownership transfer preserves the producer allocation.
    let edge = validate_scale_edge(&input, ops(), [1,1,2,2], 0.5).unwrap();
    let producer = execute(&input, 0, 0, 1, geometry(), limits()).unwrap();
    let pointer = producer.values().as_ptr();
    let binding = edge.bind(producer).unwrap();
    assert_eq!(binding.producer().values().as_ptr(), pointer);
}

#[test]
fn wrong_edge_shape_layer_operator_namespace_attributes_and_policy_refuse() {
    for case in [1,2,3,4,5,6,8,9] {
        let input = fresh(case, 0.5);
        let before = input.remaining_work();
        assert!(matches!(quote(&input, 0.5), Err(Refusal::OperationPolicy)));
        // Case9's original source name is six bytes ("scores"), not five.
        let namespace_extra = if case == 9 { 3 * 2 } else { 0 };
        assert_eq!(input.remaining_work(), before - work() - namespace_extra);
    }
    for factor in [0.25, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let input = fresh(0, 0.5);
        let before = input.remaining_work();
        assert!(matches!(quote(&input, factor), Err(Refusal::OperationPolicy)));
        assert_eq!(input.remaining_work(), before - work());
    }
    let input = fresh(0, 0.0);
    assert!(matches!(quote(&input, -0.0), Err(Refusal::OperationPolicy)));
    let input = fresh(0, 0.5);
    let reversed = ScoreScaleOperations { scores: 2, ..ops() };
    assert!(matches!(quote_scaled(&input, reversed, geometry(), limits(), Binary64PublicScale { factor: 0.5 }), Err(Refusal::OperationPolicy)));
}

#[test]
fn whole_construction_and_composition_work_exact_one_short_and_no_replay_refill() {
    let construction = construction_work(0);
    assert_eq!(fresh(0, 0.5).consumed_work(), construction);
    assert!(construction > ClearInputBuilder::resources(caps()).unwrap().construction_work);
    for short in [0, 1] {
        let input = fixture(construction + work() - short, 1 << 20, 0, 0.5);
        let result = quote(&input, 0.5);
        if short == 0 {
            let admitted = result.unwrap();
            assert_eq!(input.remaining_work(), 0);
            drop(execute_scaled(admitted).unwrap());
            assert_eq!(input.remaining_work(), 0);
        } else {
            assert!(matches!(result, Err(Refusal::Owner(owner::Error::Work))));
            assert!(input.consumed_work() > construction);
        }
        let before = input.remaining_work();
        assert!(matches!(quote(&input, 0.5), Err(Refusal::Owner(owner::Error::Work))));
        assert!(input.remaining_work() <= before);
    }
}

#[test]
fn both_backings_exact_one_short_overlap_and_cancel_release_bytes_not_work() {
    let probe = fresh(0, 0.5);
    let bytes = quote(&probe, 0.5).unwrap().resources().phase_bytes;
    let minimum = ClearInputBuilder::resources(caps()).unwrap().minimum_fresh_budget_bytes
        - size_of::<OwnedBudget>();
    for short in [0, 1] {
        let input = fixture(MAX_WORK, minimum + bytes - short, 0, 0.5);
        let before = input.remaining_work();
        let result = quote(&input, 0.5);
        if short == 0 {
            let admitted = result.unwrap();
            // Neither producer nor consumer factory has run. Both charges already
            // block overlap; cancellation releases them together, not work.
            assert!(matches!(quote(&input, 0.5), Err(Refusal::Owner(owner::Error::Resource))));
            assert_eq!(input.remaining_work(), before - 2 * work());
            drop(admitted);
            let output = execute_scaled(quote(&input, 0.5).unwrap()).unwrap();
            assert_eq!(output.scores().values(), &[-7., -23., 39., 11.]);
            assert_eq!(input.remaining_work(), before - 3 * work());
            drop(output);
            assert_eq!(input.remaining_work(), before - 3 * work());
        } else {
            assert!(matches!(result, Err(Refusal::Owner(owner::Error::Resource))));
            assert_eq!(input.remaining_work(), before - work());
        }
        assert_eq!(&*input.f64_tensor(0).unwrap(), &QUERY);
    }
}

#[test]
fn producer_and_consumer_overflow_never_publish_partial_scaled_output() {
    let live_owner = fresh(0, 0.5);
    let live_output = execute_scaled(quote(&live_owner, 0.5).unwrap()).unwrap();
    let live_scores = live_output.scores().values().as_ptr();
    let live_scaled = live_output.values().as_ptr();
    let bytes = exact_bytes();
    for (case, factor) in [(10, 0.5), (11, 2.0)] {
        let construction = construction_work(case);
        // Exact original construction/writes + two fully spent failures. There
        // is one composition's storage, not two; retry proves actual release.
        assert_eq!(work(), 13740);
        assert_eq!(construction, 20226);
        let input = fixture(construction + 2 * work(), bytes, case, factor);
        assert_eq!(input.consumed_work(), construction);
        let pointer = input.f64_tensor(0).unwrap().as_ptr();
        let expected_q = if case == 10 { [f64::MAX, -2., 3., 4.] } else { [1., 0., 1., 0.] };
        let expected_k = if case == 10 { KEY } else { [1., 0., f64::MAX, 0.] };
        for iteration in 1..=2 {
            let before = observe::snapshot();
            let admitted = quote(&input, factor).unwrap();
            assert_eq!(input.remaining_work(), (2 - iteration) * work());
            nonfinite(execute_scaled(admitted));
            let after = observe::snapshot();
            assert_eq!(after.reservations - before.reservations, 2);
            assert_eq!(after.factories - before.factories, if case == 10 { 3 } else { 4 });
            assert_eq!(after.scores_retired - before.scores_retired, 1);
            assert_eq!(after.scaled_retired - before.scaled_retired, usize::from(case == 11));
            assert_eq!(after.retired_cells - before.retired_cells, if case == 10 { 18 } else { 22 });
            assert_eq!(after.partial_scaled_wipes - before.partial_scaled_wipes, usize::from(case == 11));
            assert_eq!(after.dirty_retirements, before.dirty_retirements);
            assert_eq!(input.consumed_work(), construction + iteration * work());
            assert_eq!(input.f64_tensor(0).unwrap().as_ptr(), pointer);
            assert_eq!(&*input.f64_tensor(0).unwrap(), &expected_q);
            assert_eq!(&*input.f64_tensor(1).unwrap(), &expected_k);
            // An independently live owner/output must never be wiped by this
            // owner's partial multiplication or upstream arithmetic retirement.
            assert_eq!(&*live_owner.f64_tensor(0).unwrap(), &QUERY);
            assert_eq!(&*live_owner.f64_tensor(1).unwrap(), &KEY);
            assert_eq!(live_output.scores().values(), &[-7., -23., 39., 11.]);
            assert_eq!(live_output.values(), &[-3.5, -11.5, 19.5, 5.5]);
            assert_eq!(live_output.scores().values().as_ptr(), live_scores);
            assert_eq!(live_output.values().as_ptr(), live_scaled);
        }
        assert_eq!(input.remaining_work(), 0);
        let before = observe::snapshot();
        assert!(matches!(quote(&input, factor), Err(Refusal::Owner(owner::Error::Work))));
        assert_eq!(observe::snapshot(), before);
        assert_eq!(input.consumed_work(), construction + 2 * work());
    }
}

#[test]
fn retained_output_keeps_both_phase_charges_until_drop() {
    let probe = fresh(0, 0.5);
    let bytes = quote(&probe, 0.5).unwrap().resources().phase_bytes;
    let minimum = ClearInputBuilder::resources(caps()).unwrap().minimum_fresh_budget_bytes
        - size_of::<OwnedBudget>();
    let input = fixture(MAX_WORK, minimum + bytes, 0, 0.5);
    let before = input.remaining_work();
    let first = execute_scaled(quote(&input, 0.5).unwrap()).unwrap();
    let producer_pointer = first.scores().values().as_ptr();
    assert!(matches!(quote(&input, 0.5), Err(Refusal::Owner(owner::Error::Resource))));
    assert_eq!(first.scores().values().as_ptr(), producer_pointer);
    assert_eq!(first.scores().values(), &[-7., -23., 39., 11.]);
    assert_eq!(first.values(), &[-3.5, -11.5, 19.5, 5.5]);
    drop(first);
    drop(execute_scaled(quote(&input, 0.5).unwrap()).unwrap());
    assert_eq!(input.remaining_work(), before - 3 * work());
}

#[test]
fn cancellation_and_error_cleanup_wipes_full_consumer_capacity() {
    let input = fixture(construction_work(0) + 2 * work(), exact_bytes(), 0, 0.5);
    let before = observe::snapshot();
    let cancelled = quote(&input, 0.5).unwrap();
    drop(cancelled); // actual admitted cancellation, no numeric factory ran
    let cancelled = observe::snapshot();
    assert_eq!(cancelled.reservations - before.reservations, 2);
    assert_eq!(cancelled.factories, before.factories);
    assert_eq!(cancelled.retired_cells, before.retired_cells);
    assert_eq!(input.remaining_work(), work());
    // Exact storage can be reserved again; no refunded original work. Dropping
    // this real retained output runs both actual production retirement paths.
    let output = execute_scaled(quote(&input, 0.5).unwrap()).unwrap();
    assert_eq!(output.values(), &[-3.5, -11.5, 19.5, 5.5]);
    assert_eq!(output.scaled.values().0.len(), output.scaled.values().0.capacity());
    drop(output);
    let after = observe::snapshot();
    assert_eq!(after.reservations - before.reservations, 4);
    assert_eq!(after.factories - before.factories, 4);
    assert_eq!(after.scores_retired - before.scores_retired, 1);
    assert_eq!(after.scaled_retired - before.scaled_retired, 1);
    assert_eq!(after.retired_cells - before.retired_cells, 22);
    assert_eq!(after.partial_scaled_wipes - before.partial_scaled_wipes, 1);
    assert_eq!(after.dirty_retirements, before.dirty_retirements);
    assert_eq!(input.remaining_work(), 0);
}

#[test]
fn qk_names_colliding_with_unrelated_operation_ids_refuse_before_factories() {
    for (case, collision) in [(12, "query"), (13, "key")] {
        let input = fresh(case, 0.5);
        assert_eq!(input.operations()[2].id, collision);
        assert_eq!(input.operations()[0].inputs, ["query", "key"]);
        assert_eq!(input.tensors()[usize::from(case == 13)].name(), collision);
        let before_work = input.consumed_work();
        let before = observe::snapshot();
        assert!(matches!(quote(&input, 0.5), Err(Refusal::OperationPolicy)));
        assert_eq!(input.consumed_work(), before_work + work());
        assert_eq!(observe::snapshot(), before); // no reservations OR factories
        assert_eq!(&*input.f64_tensor(0).unwrap(), &QUERY);
        assert_eq!(&*input.f64_tensor(1).unwrap(), &KEY);
    }
}
