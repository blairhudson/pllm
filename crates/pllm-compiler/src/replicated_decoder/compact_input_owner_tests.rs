use super::*;
use crate::replicated_decoder::program::original_import::tests::{Fixture, owners};

#[test]
fn constructor_checks_exact_quote_and_one_short_before_any_compact_container() {
    for rows in [1, 2, 17, 39] {
        let fixture = Fixture::new(rows);
        let registry = fixture.registry();
        let values = Vec::new();
        let outside = owners(&values);
        let prior = RequestWork::default();
        let mut builder = fixture.builder();
        let input = builder
            .declare_original_input(
                &registry,
                &fixture.plan,
                &fixture.plan.prefill,
                "input.tokens",
                Interval::new(0, 2).unwrap(),
                &prior,
                &outside,
                MAX_PROGRAM_BYTES,
            )
            .unwrap();
        let bindings = vec![input];
        let quoted = quote(
            &builder,
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
        assert_eq!(quoted.construction.new_source_record_bytes, 0);
        assert!(quoted.construction.source_import_quote.is_none());
        let exact = quoted.construction.construction_bytes;
        let attempts = CONTAINER_FACTORIES.with(|n| n.get());
        assert!(matches!(
            CompactGraph::with_bound_inputs(
                &builder,
                fixture.validated(),
                &registry,
                &fixture.plan,
                &fixture.plan.prefill,
                0,
                &bindings,
                &prior,
                &outside,
                exact - 1
            ),
            Err(Error::Bounds)
        ));
        assert_eq!(CONTAINER_FACTORIES.with(|n| n.get()), attempts);
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
            exact,
        )
        .unwrap();
        assert_eq!(CONTAINER_FACTORIES.with(|n| n.get()), attempts + 1);
        assert_eq!(graph.bound_imports.len(), 1);
        let input = graph.frontier.value("input.tokens").unwrap().clone();
        assert_eq!(input.logical.dimensions(), &[1, rows]);
        assert_eq!(
            input.word(rows - 1).unwrap(),
            (bindings[0].owner(), rows - 1)
        );
        assert_eq!(graph.next, builder.nodes.len());
        assert_eq!(graph.work_budget(), WorkBudget::default());
        assert_eq!(graph.construction_peak_bytes(), exact);
        assert_eq!(
            graph.construction_peak_without_source.get() + quoted.borrowed_bytes,
            exact
        );
        let legacy = builder.view(bindings[0].owner(), vec![1, rows]);
        assert!(matches!(
            graph.bind_import("input.tokens", &legacy),
            Err(Error::Binding)
        ));
    }
}

#[test]
fn constructor_rejects_foreign_input_source_and_original_plan_before_factories() {
    let fixture = Fixture::new(3);
    let foreign_fixture = Fixture::new(3);
    let registry = fixture.registry();
    let foreign_registry = foreign_fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut builder = fixture.builder();
    let mut foreign_builder = fixture.builder();
    let input = builder
        .declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            "input.tokens",
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    let foreign = foreign_builder
        .declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            "input.tokens",
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    assert_eq!(input.owner(), foreign.owner());
    let bindings = vec![input];
    let foreign_bindings = vec![foreign];
    let attempts = CONTAINER_FACTORIES.with(|n| n.get());
    assert!(matches!(
        CompactGraph::with_bound_inputs(
            &builder,
            fixture.validated(),
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &foreign_bindings,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert!(matches!(
        CompactGraph::with_bound_inputs(
            &builder,
            foreign_fixture.validated(),
            &foreign_registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &bindings,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    let clone = fixture.plan.clone();
    assert!(matches!(
        CompactGraph::with_bound_inputs(
            &builder,
            fixture.validated(),
            &registry,
            &clone,
            &clone.prefill,
            0,
            &bindings,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert_eq!(CONTAINER_FACTORIES.with(|n| n.get()), attempts);
}

#[test]
fn unsealed_secret_import_cannot_hide_outside_prebound_input_inventory() {
    let fixture = Fixture::new(1);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut builder = fixture.builder();
    let input = builder
        .declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            "input.tokens",
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    let bindings = vec![input];
    builder
        .push(
            Code::Input("unsealed".into()),
            1,
            Interval::new(0, 0).unwrap(),
        )
        .unwrap();
    let attempts = CONTAINER_FACTORIES.with(|n| n.get());
    assert!(matches!(
        CompactGraph::with_bound_inputs(
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
        ),
        Err(Error::Binding)
    ));
    assert_eq!(CONTAINER_FACTORIES.with(|n| n.get()), attempts);
}

#[test]
fn actual_binding_capacity_and_retained_value_capacity_change_constructor_census() {
    let fixture = Fixture::new(3);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut builder = fixture.builder();
    let input = builder
        .declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            "input.tokens",
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    let mut bindings = vec![input];
    let before = quote(
        &builder,
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
    let capacity = bindings.capacity();
    bindings.reserve_exact(17);
    let after = quote(
        &builder,
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
    assert_eq!(
        after.construction.construction_bytes - before.construction.construction_bytes,
        (bindings.capacity() - capacity) * size_of::<OriginalImportedView<'_>>()
    );
    let mut retained = Vec::with_capacity(7);
    retained.push(Some(Secret::new(3).unwrap()));
    let held = owners(&retained);
    let held_quote = quote(
        &builder,
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        0,
        &bindings,
        &prior,
        &held,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let expected = compact_admission::outside_bytes(&held).unwrap()
        - compact_admission::outside_bytes(&outside).unwrap();
    assert_eq!(
        held_quote.construction.construction_bytes - after.construction.construction_bytes,
        expected
    );
    assert!(expected >= retained.capacity() * size_of::<Option<Secret>>() + 3 * 16);
}

#[test]
fn existing_aggregate_work_and_prior_peak_refuse_before_input_binding_factories() {
    let fixture = Fixture::new(1);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut builder = fixture.builder();
    let input = builder
        .declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            "input.tokens",
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    let bindings = vec![input];
    let mut excessive = owners(&values);
    excessive.other_work = WorkBudget::local().checked_mul(u64::MAX).unwrap();
    let attempts = CONTAINER_FACTORIES.with(|n| n.get());
    assert!(matches!(
        CompactGraph::with_bound_inputs(
            &builder,
            fixture.validated(),
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &bindings,
            &prior,
            &excessive,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Bounds)
    ));
    let mut historical = owners(&values);
    historical.prior_request_peak_bytes = MAX_PROGRAM_BYTES + 1;
    assert!(matches!(
        CompactGraph::with_bound_inputs(
            &builder,
            fixture.validated(),
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &bindings,
            &prior,
            &historical,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(CONTAINER_FACTORIES.with(|n| n.get()), attempts);
}

#[test]
fn consuming_freeze_retains_original_input_and_plan_without_new_numeric_storage() {
    let fixture = Fixture::new(3);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let mut prior = RequestWork::default();
    let mut builder = fixture.builder();
    let input = builder
        .declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            "input.tokens",
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    let bindings = vec![input];
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
    let census = compact_admission::RequestQuote::new(&graph, owners(&values))
        .unwrap()
        .owner_census();
    assert_eq!(census.output_slots, 0);
    assert_eq!(census.retained_output_bytes, 0);
    assert_eq!(
        census.metadata_bytes,
        graph.retained_metadata_bytes().unwrap()
    );
    let original = graph.frontier.value("input.tokens").unwrap().clone();
    let exports = vec![original.clone()];
    let quoted = graph.freeze_quote(&exports).unwrap();
    let semantic = source_registry::semantic_metadata_bytes(&fixture.plan.prefill).unwrap();
    let expected_retained = quoted.borrowed_owner_bytes
        + semantic
        + size_of::<frozen_region::FrozenRegion<'_>>()
        + quoted.recipe_bytes
        + quoted.native_owner_bytes
        + quoted.export_bytes
        + quoted.new_backing_bytes
        + 2 * size_of::<usize>();
    assert_eq!(quoted.retained_bytes, expected_retained);
    let frozen = graph.freeze(exports, MAX_PROGRAM_BYTES).unwrap();
    assert_eq!(frozen.work_budget(), WorkBudget::default());
    assert!(std::ptr::eq(
        frozen.imported_recipe(bindings[0].owner()).unwrap(),
        &builder.nodes[bindings[0].owner()]
    ));
    assert!(std::ptr::eq(
        frozen.borrowed_source_records(),
        &fixture.sources
    ));
    assert_eq!(frozen.word(&original, 2).unwrap(), (bindings[0].owner(), 2));
    // Snapshot can append after consuming freeze; no borrow of the entire
    // mutable request ledger is retained by the input-owner contract.
    prior.record_frozen(&frozen).unwrap();
    assert_eq!(prior.work(), WorkBudget::default());
}
