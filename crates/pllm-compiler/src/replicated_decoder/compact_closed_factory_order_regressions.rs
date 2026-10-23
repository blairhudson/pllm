//! Mount inside compact_lowering. Source-only definition; not executed here.
//! Genuine state declarations, original native attempt observer, no work oracle
//! made from the compact quote or an independently fabricated native Program.
use super::super::original_import::tests::{Fixture, owners};
use super::compact_admission::RequestWork;
use super::*;

fn selection_fixture() -> Fixture {
    let mut fixture = Fixture::new(1);
    let mut selection = fixture
        .plan
        .prefill
        .operations
        .iter()
        .find(|op| op.operator == Op::GreedyTokenSelection)
        .unwrap()
        .clone();
    selection.id = "audit.argmax".into();
    selection.inputs = vec!["audit.logits".into()];
    let state = pllm_models::StateTensor {
        id: "audit.logits".into(),
        layer: selection.layer,
        kind: pllm_models::StateKind::Recurrent,
        shape: vec![1, 3],
        maximum_sequence: 1,
    };
    fixture.plan.prefill.state_inputs = vec![state];
    fixture.plan.prefill.state_outputs.clear();
    fixture.plan.prefill.output = selection.id.clone();
    fixture.plan.prefill.operations = vec![selection];
    fixture.sources.clear();
    fixture
}

#[test]
fn original_argmax_exhausted_request_refuses_before_any_negative_factory() {
    // First prove the independently selected state/argmax fixture reaches the
    // real native factory with room. Then change ONLY outside request work.
    for exhausted in [false, true] {
        let fixture = selection_fixture();
        let registry = fixture.registry();
        let values = Vec::new();
        let outside = owners(&values);
        let mut prior = RequestWork::default();
        if exhausted {
            prior
                .record_work(
                    WorkBudget::local()
                        .checked_mul(puma::MAX_PROGRAM_OPERATIONS)
                        .unwrap(),
                )
                .unwrap();
        }
        let mut builder = fixture.builder();
        let bindings = vec![
            builder
                .declare_original_input(
                    &registry,
                    &fixture.plan,
                    &fixture.plan.prefill,
                    "audit.logits",
                    Interval::new(-ONE / 4, ONE / 4).unwrap(),
                    &prior,
                    &outside,
                    MAX_PROGRAM_BYTES,
                )
                .unwrap(),
        ];
        let before = ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get());
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
        if exhausted {
            match pending {
                Err(Error::Bounds) => {}
                Err(error) => panic!("unexpected preflight error: {error:?}"),
                Ok(mut graph) => assert!(matches!(graph.lower_next(), Err(Error::Bounds))),
            }
            assert_eq!(
                ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()),
                before,
                "an exhausted original request must refuse before native negative construction"
            );
        } else {
            pending.unwrap().lower_next().unwrap();
            assert!(ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()) > before);
        }
    }
}

fn literal_original_work(lanes: usize) -> WorkBudget {
    let mut original = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let input = original
        .push(
            Code::Input("audit.logits".into()),
            lanes,
            Interval::new(-ONE / 4, ONE / 4).unwrap(),
        )
        .unwrap();
    let view = original.view(input, vec![1, lanes]);
    let start = original.nodes.len();
    original.argmax_tree(&view).unwrap();
    // Actual independent original Builder records and native owners, not a
    // compiler work quote or a hand-written per-instruction recipe.
    schedule_work(&original.nodes[start..]).unwrap()
}

#[test]
fn whole_tree_one_short_and_exact_preserve_original_work_and_factory_order() {
    for lanes in [3, 129, 257, 513] {
        let expected = literal_original_work(lanes);
        assert!(expected.operations() > 1);
        for short in [true, false] {
            let mut fixture = selection_fixture();
            fixture.plan.prefill.state_inputs[0].shape = vec![1, lanes as u64];
            let registry = fixture.registry();
            let values = Vec::new();
            let outside = owners(&values);
            let mut prior = RequestWork::default();
            prior
                .record_work(
                    WorkBudget::local()
                        .checked_mul(
                            puma::MAX_PROGRAM_OPERATIONS
                                .checked_sub(expected.operations())
                                .unwrap()
                                .checked_add(u64::from(short))
                                .unwrap(),
                        )
                        .unwrap(),
                )
                .unwrap();
            let mut builder = fixture.builder();
            let bindings = vec![
                builder
                    .declare_original_input(
                        &registry,
                        &fixture.plan,
                        &fixture.plan.prefill,
                        "audit.logits",
                        Interval::new(-ONE / 4, ONE / 4).unwrap(),
                        &prior,
                        &outside,
                        MAX_PROGRAM_BYTES,
                    )
                    .unwrap(),
            ];
            let before = ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get());
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
            if short {
                match pending {
                    Err(Error::Bounds) => {}
                    Err(error) => panic!("unexpected preflight error: {error:?}"),
                    Ok(mut graph) => {
                        assert!(matches!(graph.lower_next(), Err(Error::Bounds)));
                        assert_eq!(graph.work_budget(), WorkBudget::default());
                    }
                }
                assert_eq!(ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()), before);
            } else {
                let mut graph = pending.unwrap();
                graph.lower_next().unwrap();
                assert_eq!(graph.work_budget(), expected);
                assert!(ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()) > before);
            }
        }
    }
}

#[test]
fn published_graph_work_and_request_work_gate_second_tree_before_reconstructed_interned_kernels() {
    let expected = literal_original_work(3);
    let mut fixture = selection_fixture();
    let mut second = fixture.plan.prefill.operations[0].clone();
    second.id = "audit.argmax.second".into();
    fixture.plan.prefill.output = second.id.clone();
    fixture.plan.prefill.operations.push(second);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let mut prior = RequestWork::default();
    prior
        .record_work(
            WorkBudget::local()
                .checked_mul(
                    puma::MAX_PROGRAM_OPERATIONS
                        .checked_sub(expected.checked_mul(2).unwrap().operations())
                        .unwrap()
                        .checked_add(1)
                        .unwrap(),
                )
                .unwrap(),
        )
        .unwrap();
    let mut builder = fixture.builder();
    let bindings = vec![
        builder
            .declare_original_input(
                &registry,
                &fixture.plan,
                &fixture.plan.prefill,
                "audit.logits",
                Interval::new(-ONE / 4, ONE / 4).unwrap(),
                &prior,
                &outside,
                MAX_PROGRAM_BYTES,
            )
            .unwrap(),
    ];
    let mut graph = CompactGraph::with_bound_inputs(
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
    graph.lower_next().unwrap();
    assert_eq!(graph.work_budget(), expected);
    let before = ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get());
    assert!(matches!(graph.lower_next(), Err(Error::Bounds)));
    assert_eq!(ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()), before);
    assert_eq!(graph.work_budget(), expected);
}
