//! Mount inside original_import for the owner audit; intentionally unrun here.
//! No imported identity, proof, native kernel or resource authority is fabricated.
use super::super::compact_lowering::CompactGraph;
use super::compact_admission::{RequestOwners, RequestWork};
use super::tests::{Fixture, owners};
use super::*;
use crate::replicated_decoder::prepared_tensor_map::native_attempts;

fn fixture() -> Fixture {
    let mut fixture = Fixture::new(1);
    let mut silu = fixture
        .plan
        .prefill
        .operations
        .iter()
        .find(|op| op.operator == Op::Silu)
        .unwrap()
        .clone();
    silu.id = "audit.silu".into();
    silu.inputs = vec!["audit.state".into()];
    silu.state_kind = Some(pllm_models::StateKind::Recurrent);
    let state = pllm_models::StateTensor {
        id: "audit.state".into(),
        layer: silu.layer,
        kind: pllm_models::StateKind::Recurrent,
        shape: silu.output_shape.clone(),
        maximum_sequence: 1,
    };
    fixture.plan.prefill.state_inputs = vec![state.clone()];
    fixture.plan.prefill.state_outputs = vec![pllm_models::StateTensor {
        id: silu.id.clone(),
        ..state
    }];
    fixture.plan.prefill.output = silu.id.clone();
    fixture.plan.prefill.operations = vec![silu];
    fixture.sources.clear();
    fixture
}

fn bounds() -> Interval {
    Interval::new(-ONE / 4, ONE / 4).unwrap()
}

#[test]
fn actual_original_record_spare_capacity_remains_charged_at_construction_and_freeze() {
    let fixture = fixture();
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut builder = fixture.builder();
    let bindings = vec![
        builder
            .declare_original_input(
                &registry,
                &fixture.plan,
                &fixture.plan.prefill,
                "audit.state",
                bounds(),
                &prior,
                &outside,
                MAX_PROGRAM_BYTES,
            )
            .unwrap(),
    ];
    let before = {
        let graph = CompactGraph::with_bound_inputs(
            &builder,
            fixture.validated(),
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &bindings,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
        let exports = Vec::new();
        let frozen = graph.freeze_quote(&exports).unwrap();
        (
            graph.construction_peak_bytes(),
            frozen.borrowed_owner_bytes,
            frozen.retained_bytes,
        )
    };
    let old_capacity = builder.original_inputs.capacity();
    let charged = builder.allocated;
    // Legitimate spare backing, not an inserted/relabelled input declaration.
    // The original seal and immutable record allocation remain exactly the same.
    builder.original_inputs.try_reserve_exact(8192).unwrap();
    let extra = (builder.original_inputs.capacity() - old_capacity)
        .checked_mul(size_of::<Arc<InputRecord>>())
        .unwrap();
    assert!(extra > 0);
    assert_eq!(builder.allocated, charged);
    bindings[0]
        .check(&builder, &registry, &fixture.plan, &fixture.plan.prefill)
        .unwrap();
    let graph = CompactGraph::with_bound_inputs(
        &builder,
        fixture.validated(),
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        0,
        &bindings,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let exports = Vec::new();
    let frozen = graph.freeze_quote(&exports).unwrap();
    // Independent expected delta: actual Vec capacity * literal owned slot.
    // Neither imports_bytes nor another quote field supplies this expectation.
    assert!(graph.construction_peak_bytes() >= before.0 + extra);
    assert!(frozen.borrowed_owner_bytes >= before.1 + extra);
    assert!(frozen.retained_bytes >= before.2 + extra);
}

#[test]
fn bound_input_silu_refuses_exhausted_original_work_before_native_entry() {
    let fixture = fixture();
    let registry = fixture.registry();
    let values = Vec::new();
    let mut outside: RequestOwners<'_> = owners(&values);
    // Ordinary request work uses the existing outside-owner inventory. Keep
    // private compact-region ledger mutation private; both enter the same gate.
    outside.other_work = WorkBudget::local()
        .checked_mul(puma::MAX_PROGRAM_OPERATIONS)
        .unwrap();
    let prior = RequestWork::default();
    let mut builder = fixture.builder();
    let bindings = vec![
        builder
            .declare_original_input(
                &registry,
                &fixture.plan,
                &fixture.plan.prefill,
                "audit.state",
                bounds(),
                &prior,
                &outside,
                MAX_PROGRAM_BYTES,
            )
            .unwrap(),
    ];
    let before = native_attempts();
    let pending = CompactGraph::with_bound_inputs(
        &builder,
        fixture.validated(),
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        0,
        &bindings,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    );
    match pending {
        Err(Error::Bounds) => {}
        Err(error) => panic!("unexpected preflight error: {error:?}"),
        Ok(mut graph) => assert!(matches!(graph.lower_next(), Err(Error::Bounds))),
    }
    assert_eq!(
        native_attempts(),
        before,
        "aggregate work refusal must precede even the first native constructor"
    );
}
