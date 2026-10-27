use super::*;
use pllm_garble::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};
use pllm_types::{canonical_bytes, digest_bytes};

fn model() -> DecoderPlan {
    super::super::preconstruction_tests::plan("qwen2", true)
}
fn root() -> OwnedBudget {
    OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap()
}
fn limits() -> Limits {
    Limits { max_work: 1 << 28 }
}

// Upstream fixture construction is deliberately outside this constructor's gate.
// This helper establishes a genuine original charged owner; production parse /
// import must separately preadmit its own factories and cannot use this shortcut.
fn input(root: &OwnedBudget, plan: DecoderPlan) -> OwnedCharged<DecoderPlan> {
    let backing = plan.checked_owned_backing_bytes(usize::MAX).unwrap();
    root.charge_external(backing)
        .unwrap()
        .try_build(|_| Ok(plan))
        .unwrap()
}
fn prepare(
    input: &OwnedCharged<DecoderPlan>,
    working: OwnedExternalCharge,
) -> ConstructionQuote<'_> {
    ConstructionQuote::checked(input, working, limits())
        .unwrap_or_else(|(error, _working)| panic!("fixture quote: {error}"))
}
fn work(root: &OwnedBudget) -> OwnedExternalCharge {
    root.charge_external(bootstrap_bytes().unwrap()).unwrap()
}

#[test]
fn guarded_original_legacy_bytes_digest_and_owner_binding() {
    for family in ["qwen2", "qwen3", "llama"] {
        for tied in [false, true] {
            let root = root();
            let source = input(
                &root,
                super::super::preconstruction_tests::plan(family, tied),
            );
            let expected = canonical_bytes(
                &super::super::preconstruction_tests::legacy_inventory_json(source.payload()),
            );
            let quote = prepare(&source, work(&root));
            assert!(std::ptr::eq(quote.input, &source));
            assert!(quote.context.factories == 0);
            let minimum = size_of::<OwnedCharged<DecoderPlan>>()
                + source
                    .payload()
                    .checked_owned_backing_bytes(usize::MAX)
                    .unwrap();
            assert_eq!(quote.resources.input_owner_bytes, minimum);
            let resources = quote.resources;
            let outcome = quote.construct().unwrap();
            assert!(std::ptr::eq(outcome.payload().input, &source));
            assert!(std::ptr::eq(outcome.budget(), source.budget()));
            let actual = outcome.payload().result.as_ref().unwrap();
            assert_eq!(actual.canonical_bytes(), expected);
            assert_eq!(
                actual.digest(),
                &digest_bytes(DECODER_MATRIX_INVENTORY_SCHEMA_VERSION, &expected)
            );
            assert_eq!(
                actual.semantic_plan_identity(),
                source.payload().digest().as_sha256()
            );
            assert_eq!(actual.digest().backing_capacity_bytes(), 64);
            assert_eq!(actual.bytes.capacity(), actual.bytes.len());
            assert!(outcome.payload().work <= limits().max_work);
            assert_eq!(
                outcome.charged_bytes(),
                resources.final_working_payload_bytes + size_of::<OwnedExternalCharge>()
            );
        }
    }
}

#[test]
fn guarded_quote_prices_actual_spare_capacities_not_lengths_or_clones() {
    let root = root();
    let mut plan = model();
    let before = {
        let source = input(&root, model());
        let resources = prepare(&source, work(&root)).resources;
        resources
    };
    // Compare actual original backing before/after, not a clone's capacities.
    let old_backing = plan.checked_owned_backing_bytes(usize::MAX).unwrap();
    let old = plan.prefill.operations.capacity();
    plan.prefill.operations.reserve(31);
    let mut delta = (plan.prefill.operations.capacity() - old) * size_of::<ModelOperation>();
    let old = plan.prefill.operations[0].id.capacity();
    plan.prefill.operations[0].id.reserve(8192);
    delta += plan.prefill.operations[0].id.capacity() - old;
    let old = plan.prefill.operations[0].output_shape.capacity();
    plan.prefill.operations[0].output_shape.reserve(17);
    delta += (plan.prefill.operations[0].output_shape.capacity() - old) * size_of::<u64>();
    let source = input(&root, plan);
    let quote = prepare(&source, work(&root));
    assert_eq!(
        quote.resources.input_owner_bytes,
        size_of::<OwnedCharged<DecoderPlan>>()
            + source
                .payload()
                .checked_owned_backing_bytes(usize::MAX)
                .unwrap()
    );
    assert_eq!(quote.resources.input_reserved_bytes, source.charged_bytes());
    assert_eq!(quote.resources.input_owner_bytes,
        size_of::<OwnedCharged<DecoderPlan>>() + old_backing + delta);
    assert_eq!(quote.resources.row_backing_bytes, before.row_backing_bytes);
    assert_eq!(quote.resources.index_workspace_bytes, before.index_workspace_bytes);
    assert_eq!(quote.resources.grouping_workspace_bytes, before.grouping_workspace_bytes);
    assert_eq!(quote.resources.metadata_backing_bytes, before.metadata_backing_bytes);
    assert!(std::ptr::eq(quote.input.payload(), source.payload()));
    assert!(quote.resources.output_and_digest_bytes >= MAX_OUTPUT_BYTES + 64);
}

#[test]
fn guarded_independent_row_census_all_three_indices_and_both_groupings() {
    let root = root();
    let source = input(&root, model());
    let plan = source.payload();
    let mut context = Context::new(true, limits().max_work).unwrap();
    let mut row_bytes = 0;
    let mut index_bytes = 0;
    let mut grouping_bytes = 0;
    let mut metadata_bytes = 0;
    for graph in [&plan.prefill, &plan.decode] {
        // Independent namespace census; never read candidate quote counts.
        let remote_count = graph.operations.iter().filter(|op| remote(op.operator)).count();
        let count = 3 * PUBLIC_INPUTS.len() + 4 * graph.state_inputs.len()
            + 3 * graph.state_outputs.len() + 3 * graph.operations.len() + remote_count;
        row_bytes += std::alloc::Layout::array::<PhaseEntry<'_>>(count).unwrap().size();
        let rows = PhaseRows::build(graph, &mut context).unwrap();
        assert_eq!(rows.rows.len(), count);
        let index = rows.index(&mut context).unwrap();
        let resources = index.index.resources();
        assert_eq!(resources.heap_bytes, count * size_of::<usize>());
        index_bytes += resources.workspace_bytes;
        grouping_bytes += RemoteGroupingQuote::checked(graph).unwrap().resources().workspace_bytes;
        metadata_bytes += std::alloc::Layout::array::<Group<'_>>(graph.operations.len()).unwrap().size()
            + std::alloc::Layout::array::<Member<'_>>(graph.operations.len()).unwrap().size();
    }
    let source_count = [&plan.prefill, &plan.decode].iter().flat_map(|graph| &graph.operations)
        .filter(|op| remote(op.operator))
        .map(|op| ["weight", "bias"].iter()
            .filter(|field| op.attributes.get(**field).and_then(Value::as_str).is_some()).count())
        .sum::<usize>();
    row_bytes += std::alloc::Layout::array::<SourceEntry<'_>>(source_count).unwrap().size();
    let stats = source_stats(plan, &mut context).unwrap();
    assert_eq!(stats.count(), source_count);
    let resources = stats.resources().unwrap();
    assert_eq!(resources.heap_bytes, source_count * size_of::<usize>());
    index_bytes += resources.workspace_bytes;
    let quote = prepare(&source, work(&root));
    assert_eq!(quote.context.factories, 0);
    assert_eq!(quote.resources.row_backing_bytes, row_bytes);
    assert_eq!(quote.resources.index_workspace_bytes, index_bytes);
    assert_eq!(quote.resources.grouping_workspace_bytes, grouping_bytes);
    assert_eq!(quote.resources.metadata_backing_bytes, metadata_bytes);
    assert!(quote.construct().unwrap().payload().result.is_ok());
}

#[test]
fn guarded_original_metadata_backing_is_inspected_before_serialization() {
    let root = root();
    let source = input(&root, super::super::preconstruction_tests::plan("llama", true));
    let mut quote = prepare(&source, work(&root));
    quote.context.inspect_metadata = true;
    let outcome = quote.construct().unwrap();
    assert!(outcome.payload().result.is_ok());
    assert!(std::ptr::eq(outcome.payload().input, &source));
    assert!(std::ptr::eq(outcome.budget(), source.budget()));
}

#[test]
fn source_scan_one_short_refuses_before_any_field_visit_or_factory() {
    let plan = model();
    let scan = 3 * (plan.prefill.operations.len() + plan.decode.operations.len()) as u128;
    let pool = Context::pool_work().unwrap();
    let mut context = Context::new(true, pool + scan - 1).unwrap();
    SOURCE_VISITS.with(|visits| visits.set(0));
    assert!(source_stats(&plan, &mut context).is_err());
    SOURCE_VISITS.with(|visits| assert_eq!(visits.get(), 0));
    assert_eq!(context.factories, 0);
    let mut context = Context::new(true, limits().max_work).unwrap();
    let stats = source_stats(&plan, &mut context).unwrap();
    SOURCE_VISITS.with(|visits| assert!(visits.get() > 0));
    assert!(stats.count() > 0);
    // Row emission has a separate scan prepayment, even after a valid census.
    let mut context = Context::new(true, pool + scan - 1).unwrap();
    SOURCE_VISITS.with(|visits| visits.set(0));
    assert!(prepay_source_scan(&plan, &mut context).is_err());
    SOURCE_VISITS.with(|visits| assert_eq!(visits.get(), 0));
    assert_eq!(context.factories, 0);
}

#[test]
fn existing_inventory_caps_remain_fixed_and_enforced() {
    assert_eq!((MAX_OPERATIONS, MAX_STATES, MAX_INPUTS), (4096, 1024, 16));
    assert_eq!((MAX_OBJECT_FIELDS, MAX_STRING_BYTES), (64, 1024));
    assert_eq!((MAX_TOTAL_STRING_BYTES, MAX_NODES, MAX_DEPTH), (1 << 20, 131072, 24));
    assert_eq!(DECODER_MATRIX_INVENTORY_MAX_PLAN_BYTES, 8 << 20);
    assert_eq!(MAX_OUTPUT_BYTES, 8 << 20);
    assert_eq!(plan_canonical_limits(1 << 28).canonical.max_bytes, 8 << 20);
    assert_eq!(inventory_stream_limits().max_bytes, 8 << 20);
    for case in 0..4 {
        let mut plan = model();
        match case {
            0 => plan.prefill.operations.resize(MAX_OPERATIONS + 1, plan.prefill.operations[0].clone()),
            1 => plan.decode.state_inputs.resize(MAX_STATES + 1, plan.decode.state_inputs[0].clone()),
            2 => plan.decode.state_outputs.resize(MAX_STATES + 1, plan.decode.state_outputs[0].clone()),
            _ => plan.prefill.operations[0].inputs.resize(MAX_INPUTS + 1, "input.tokens".into()),
        }
        assert!(preflight_with_work(&plan, limits().max_work as usize).is_err());
    }
    let mut budget = MetadataBudget {
        nodes: MAX_NODES - 1,
        work_limit: limits().max_work,
        ..Default::default()
    };
    assert!(budget.node(MAX_DEPTH).is_ok());
    assert!(budget.node(MAX_DEPTH).is_err());
    let mut budget = MetadataBudget { work_limit: limits().max_work, ..Default::default() };
    assert!(budget.node(MAX_DEPTH + 1).is_err());
    let text = "x".repeat(MAX_STRING_BYTES);
    let mut budget = MetadataBudget {
        strings: MAX_TOTAL_STRING_BYTES - MAX_STRING_BYTES,
        work_limit: limits().max_work,
        ..Default::default()
    };
    assert!(budget.string(&text).is_ok());
    assert!(budget.string("x").is_err());
    let mut budget = MetadataBudget { work_limit: limits().max_work, ..Default::default() };
    assert!(budget.string(&(text + "x")).is_err());
    let mut fields = serde_json::Map::new();
    for index in 0..MAX_OBJECT_FIELDS { fields.insert(format!("key.{index}"), Value::Null); }
    let mut value = Value::Object(fields);
    let mut budget = MetadataBudget { work_limit: limits().max_work, ..Default::default() };
    assert!(budget.value(&value, 0).is_ok());
    value.as_object_mut().unwrap().insert("one.more".into(), Value::Null);
    assert!(budget.value(&value, 0).is_err());
}

#[test]
fn shared_exact_buffer_error_and_finish_paths_drop_initialized_owners_once() {
    use std::rc::Rc;
    struct Probe(Rc<Cell<usize>>);
    impl Drop for Probe {
        fn drop(&mut self) { self.0.set(self.0.get() + 1); }
    }
    let drops = Rc::new(Cell::new(0));
    let mut partial = bounded::ExactBuffer::new(2).unwrap();
    partial.push(Probe(drops.clone())).unwrap();
    assert!(partial.finish().is_err());
    assert_eq!(drops.get(), 1);
    let mut full = bounded::ExactBuffer::new(1).unwrap();
    full.push(Probe(drops.clone())).unwrap();
    assert!(full.push(Probe(drops.clone())).is_err());
    assert_eq!(drops.get(), 2);
    let owners = full.finish().unwrap();
    assert_eq!(drops.get(), 2);
    drop(owners);
    assert_eq!(drops.get(), 3);
}

#[test]
fn shared_layout_overflow_and_charged_borrowed_errors_keep_original_backing() {
    assert!(bounded::ExactBuffer::<[u64; 2]>::new(usize::MAX).is_err());
    assert!(bounded::layout_bytes::<SourceEntry<'_>>(usize::MAX).is_err());
    assert!(bounded::checked_add(u128::MAX, 1).is_err());
    assert!(bounded::checked_mul(u128::MAX, 2).is_err());
    for unbound in [false, true] {
        let root = root();
        let mut plan = model();
        plan.prefill.operations.iter_mut().find(|op| remote(op.operator))
            .unwrap().attributes["unsupported-λ"] = Value::Null;
        if unbound { plan.prefill.operations[0].inputs[0] = "not.bound".into(); }
        let source = input(&root, plan);
        let quote = prepare(&source, work(&root));
        assert_eq!(quote.context.factories, 0);
        let outcome = quote.construct().unwrap();
        let failure = outcome.payload().result.as_ref().unwrap_err();
        if unbound {
            assert_eq!(failure.to_string(), "matrix inventory unbound operation input ID");
        } else {
            match failure {
                InventoryFailure::Attribute(key) => {
                    let original = source.payload().prefill.operations.iter()
                        .find(|op| remote(op.operator)).unwrap().attributes.as_object().unwrap()
                        .keys().find(|key| *key == "unsupported-λ").unwrap();
                    assert_eq!(key.as_ptr(), original.as_ptr());
                }
                _ => panic!("expected original borrowed attribute key"),
            }
        }
        assert!(std::ptr::eq(outcome.budget(), source.budget()));
        assert!(outcome.charged_bytes() > 0);
    }
}

#[test]
fn guarded_rejects_undercharged_original_owner_and_wrong_counter() {
    let root = root();
    let source = root
        .charge_external(0)
        .unwrap()
        .try_build(|_| Ok(model()))
        .unwrap();
    let outcome = construct(&source, work(&root), limits()).unwrap();
    assert!(outcome.payload().result.is_err());
    assert_eq!(outcome.payload().factories, 0);
    assert!(std::ptr::eq(outcome.budget(), source.budget()));
    let other = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    assert!(matches!(
        construct(&source, work(&other), limits()),
        Err(NativeError::Context)
    ));
}

#[test]
fn guarded_bootstrap_zero_one_short_and_work_refusal_before_factories() {
    let root = root();
    let source = input(&root, model());
    assert!(matches!(
        construct(&source, root.charge_external(0).unwrap(), limits()),
        Err(NativeError::Resource)
    ));
    let short = root
        .charge_external(bootstrap_bytes().unwrap() - 1)
        .unwrap();
    let outcome = construct(&source, short, limits()).unwrap();
    assert!(outcome.payload().result.is_err());
    assert_eq!(outcome.payload().factories, 0);
    let outcome = construct(&source, work(&root), Limits { max_work: 0 }).unwrap();
    assert!(outcome.payload().result.is_err());
    assert_eq!(outcome.payload().factories, 0);
}

#[test]
fn guarded_exact_and_one_short_counter_boundaries_precede_factories() {
    let probe = root();
    let source = input(&probe, model());
    let quote = prepare(&source, work(&probe));
    let required = quote.resources.total_original_counter_bytes;
    drop(quote);
    drop(source);
    drop(probe);
    for short in [false, true] {
        let root = OwnedBudget::new(required - usize::from(short)).unwrap();
        let source = input(&root, model());
        let quote = prepare(&source, work(&root));
        assert_eq!(quote.resources.total_original_counter_bytes, required);
        assert!(std::ptr::eq(quote.input, &source));
        assert_eq!(quote.context.factories, 0);
        let result = quote.construct();
        if short {
            match result {
                Err(NativeError::Resource) => (), // final try_build inline gate
                Ok(outcome) => {
                    assert!(outcome.payload().result.is_err());
                    assert_eq!(outcome.payload().factories, 0);
                    assert!(std::ptr::eq(outcome.budget(), source.budget()));
                    assert!(outcome.charged_bytes() > 0);
                }
                _ => panic!("wrong short admission result"),
            }
        } else {
            let outcome = result.unwrap();
            assert!(outcome.payload().result.is_ok());
            assert!(std::ptr::eq(outcome.payload().input, &source));
            assert!(std::ptr::eq(outcome.budget(), source.budget()));
            assert_eq!(outcome.budget().used_bytes(), required);
        }
    }
}

#[test]
fn guarded_factory_errors_and_panics_retire_payload_before_original_charge() {
    let root = root();
    let source = input(&root, model());
    let baseline = source.budget().used_bytes();
    let successful = prepare(&source, work(&root)).construct().unwrap();
    let factories = successful.payload().factories;
    assert!(factories > 8);
    drop(successful);
    assert_eq!(source.budget().used_bytes(), baseline);
    for ordinal in 0..factories {
        let mut quote = prepare(&source, work(&root));
        quote.context.fail_after = Some(ordinal);
        let outcome = quote.construct().unwrap();
        assert!(outcome.payload().result.is_err(), "factory {ordinal}");
        assert_eq!(outcome.payload().work, limits().max_work);
        assert_eq!(outcome.payload().factories, ordinal);
        assert!(outcome.budget().used_bytes() > baseline);
        drop(outcome);
        assert_eq!(source.budget().used_bytes(), baseline);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut quote = prepare(&source, work(&root));
            quote.context.panic_after = Some(ordinal);
            let _ = quote.construct();
        }));
        assert!(result.is_err());
        assert_eq!(source.budget().used_bytes(), baseline);
    }
}

#[test]
fn guarded_success_and_borrowed_failure_retain_counter_after_root_drop() {
    for fail in [false, true] {
        let root = root();
        let mut plan = model();
        if fail {
            plan.prefill.operations[0].inputs[0] = "future.input".into();
        }
        let source = input(&root, plan);
        let outcome = construct(&source, work(&root), limits()).unwrap();
        assert_eq!(outcome.payload().result.is_err(), fail);
        let before = source.budget().used_bytes();
        let address = source.budget() as *const _;
        drop(root);
        assert_eq!(outcome.budget() as *const _, address);
        assert!(outcome.budget().used_bytes() < before);
        assert!(outcome.budget().used_bytes() >= source.charged_bytes() + outcome.charged_bytes());
        drop(outcome);
        assert!(source.budget().used_bytes() >= source.charged_bytes());
    }
}

#[test]
fn indexed_references_match_original_forward_duplicate_and_directional_errors() {
    for case in 0..8 {
        let mut plan = model();
        // Prefill intentionally has no cache inputs; decode owns both directions.
        let graph = &mut plan.decode;
        match case {
            0 => graph.operations[0].inputs[0] = graph.operations[1].id.clone(),
            1 => graph.operations[1].id = graph.operations[0].id.clone(),
            2 => graph.state_inputs[0].id = "input.tokens".into(),
            3 => graph.state_inputs.push(graph.state_inputs[0].clone()),
            4 => graph.state_outputs.push(graph.state_outputs[0].clone()),
            5 => graph.state_outputs[0].shape[0] += 1,
            6 => graph.output = "not.a.producer".into(),
            _ => (),
        }
        let expected = validate_references(graph).map_err(|error| error.to_string());
        let mut context = Context::new(true, limits().max_work).unwrap();
        let rows = PhaseRows::build(graph, &mut context).unwrap();
        let lookup = rows.index(&mut context).unwrap();
        let actual = lookup
            .validate(graph, &mut context)
            .map_err(|error| error.to_string());
        assert_eq!(actual, expected, "case {case}");
    }
}

#[test]
fn guarded_matches_legacy_malformed_errors_and_tied_precedence() {
    for case in 0..9 {
        let mut plan = model();
        match case {
            0 => plan.prefill.operations[0].attributes["weight"] = serde_json::json!([1, 2]),
            1 => {
                let head = plan
                    .prefill
                    .operations
                    .iter_mut()
                    .find(|op| op.operator == ModelOperator::OutputHead)
                    .unwrap();
                head.attributes["tied"] = Value::Bool(false);
                plan.prefill.operations[0].attributes["weight"] = Value::Null;
            }
            2 => {
                plan.prefill.operations[0].attributes["weight"] =
                    Value::String("input.tokens".into())
            }
            3 => plan.prefill.operations[0].attributes["bias"] = Value::String("table.bias".into()),
            4 => {
                plan.decode.operations[0].attributes["checkpoint_layout"] =
                    Value::String("changed".into());
            }
            5 => plan.prefill.operations[0].inputs[0] = "unknown".into(),
            6 => plan.prefill.operations[0].attributes["unsupported"] = Value::Bool(true),
            7 | 8 => {
                plan.prefill.operations[0].attributes["weight"] = if case == 7 {
                    serde_json::json!(-0.0)
                } else {
                    serde_json::json!(1)
                };
                let head = plan
                    .prefill
                    .operations
                    .iter_mut()
                    .find(|op| op.operator == ModelOperator::OutputHead)
                    .unwrap();
                head.attributes["weight"] = if case == 7 {
                    serde_json::json!(0.0)
                } else {
                    serde_json::json!(1.0)
                };
            }
            _ => unreachable!(),
        }
        // Frozen HEAD 17e55e9e error branches, NOT the newly shared public
        // lowering as oracle. Only the original operation ID is interpolated.
        let expected =
            match case {
                0 | 2 | 8 => "matrix inventory head tied declaration differs from source ownership"
                    .to_owned(),
                1 | 7 => format!(
                    "remote operation {} requires one declared weight artifact",
                    plan.prefill.operations[0].id
                ),
                3 => "matrix inventory token table cannot have bias".to_owned(),
                4 => "matrix inventory incompatible phase matrix attributes".to_owned(),
                5 => "matrix inventory unbound operation input ID".to_owned(),
                6 => "matrix inventory unsupported or malformed matrix attribute unsupported"
                    .to_owned(),
                _ => unreachable!(),
            };
        assert_eq!(lower_decoder_matrix_inventory(&plan).unwrap_err(), expected);
        let root = root();
        let source = input(&root, plan);
        let outcome = construct(&source, work(&root), limits()).unwrap();
        let actual = outcome.payload().result.as_ref().unwrap_err().to_string();
        assert_eq!(actual, expected, "case {case}");
    }
}

#[test]
fn indexed_source_ledger_alias_per_use_sum_and_conflict_precedence() {
    let plan = model();
    // Actual source keys, not an invented factory/authority token.
    let id = plan
        .prefill
        .operations
        .iter()
        .find_map(|op| op.attributes.get("weight").and_then(Value::as_str))
        .unwrap();
    let graph = &plan.prefill;
    let mut context = Context::new(true, limits().max_work).unwrap();
    let rows = PhaseRows::build(graph, &mut context).unwrap();
    let lookup = rows.index(&mut context).unwrap();
    let stats = source_stats(&plan, &mut context).unwrap();
    let mut entries = context.buffer(stats.count()).unwrap();
    prepay_source_scan(&plan, &mut context).unwrap();
    source_entries(&plan, |entry| {
        context.step()?;
        entries.push(entry).map_err(index_error)
    }).unwrap();
    let entries = entries.finish().unwrap();
    let quote = Quote::checked(&entries, context.remaining()).unwrap();
    let resources = quote.resources();
    context.charge(resources.quote_work).unwrap();
    context.charge(resources.construction_work).unwrap();
    context.factory().unwrap();
    let mut ledger = SourceLedger {
        index: quote
            .construct(resources.workspace_bytes, resources.construction_work)
            .unwrap(),
        elements: 0,
    };
    ledger
        .bind(id, &[4, 8], graph, &lookup, &mut context)
        .unwrap();
    let bound = ledger.index.resources().lookup_work;
    context.charge(bound).unwrap();
    let found = ledger.index.matches(&id, &mut bounded::Usage::default(), bound).unwrap();
    let cell = &found.first().unwrap().value;
    let original_cell = cell as *const _;
    assert_eq!(cell.get().unwrap().dimensions, [4, 8]);
    ledger
        .bind(id, &[4, 8], graph, &lookup, &mut context)
        .unwrap();
    context.charge(bound).unwrap();
    let found = ledger.index.matches(&id, &mut bounded::Usage::default(), bound).unwrap();
    assert_eq!(&found.first().unwrap().value as *const _, original_cell);
    assert_eq!(ledger.elements, 64);
    assert_eq!(
        ledger
            .bind(id, &[8, 4], graph, &lookup, &mut context)
            .unwrap_err()
            .to_string(),
        "matrix inventory inconsistent source artifact shape"
    );
    ledger.elements = u64::MAX;
    assert_eq!(
        ledger
            .bind("input.tokens", &[8, 4], graph, &lookup, &mut context)
            .unwrap_err()
            .to_string(),
        "matrix inventory source artifact ID aliases a graph value"
    );
    assert_eq!(
        ledger
            .bind(id, &[8, 4], graph, &lookup, &mut context)
            .unwrap_err()
            .to_string(),
        "matrix inventory source size sum overflow"
    );
    ledger.elements = MAX_TOTAL_ELEMENTS;
    assert_eq!(
        ledger
            .bind(id, &[8, 4], graph, &lookup, &mut context)
            .unwrap_err()
            .to_string(),
        "matrix inventory source size sum exceeds limit"
    );
}

#[test]
fn guarded_depth_work_and_layout_overflow_never_fall_back_to_legacy_census() {
    let root = root();
    let mut plan = model();
    let mut nested = Value::Null;
    for _ in 0..17 {
        nested = serde_json::json!([nested]);
    }
    plan.prefill.operations.iter_mut().find(|op| !remote(op.operator))
        .unwrap().attributes["extra"] = nested;
    assert!(lower_decoder_matrix_inventory(&plan).is_ok());
    // A deliberately generous upstream reservation permits testing bounded depth
    // rejection without invoking the legacy depth-16 census in fixture setup.
    let source = root
        .charge_external(1 << 20)
        .unwrap()
        .try_build(|_| Ok(plan))
        .unwrap();
    let outcome = construct(&source, work(&root), limits()).unwrap();
    assert!(matches!(
        outcome.payload().result.as_ref(),
        Err(InventoryFailure::Census(_))
    ));
    assert_eq!(outcome.payload().factories, 0);
    assert!(bounded::layout_bytes::<SourceEntry<'_>>(usize::MAX).is_err());
    let pool = Context::pool_work().unwrap();
    let mut context = Context::new(true, pool + 1).unwrap();
    assert!(context.charge(u128::MAX).is_err());
    assert_eq!(context.work, pool);
}

#[test]
fn inventory_index_build_and_producer_lookup_work_scale_n_log_n() {
    let mut previous = None;
    for count in [32, 64, 128, 256, 512, 1024] {
        let mut graph = model().prefill;
        let mut operation = graph.operations[0].clone();
        operation.operator = ModelOperator::TokenFeedback;
        operation.inputs.clear();
        graph.operations = (0..count)
            .map(|index| {
                let mut operation = operation.clone();
                operation.id = format!("value.{index:08}");
                operation
            })
            .collect();
        let mut context = Context::new(true, 1 << 48).unwrap();
        let rows = PhaseRows::build(&graph, &mut context).unwrap();
        let lookup = rows.index(&mut context).unwrap();
        for operation in &graph.operations {
            let actual = lookup
                .operation(&operation.id, false, &mut context)
                .unwrap()
                .unwrap();
            assert!(std::ptr::eq(actual, operation));
        }
        if let Some(previous) = previous {
            assert!(context.work < 3 * previous);
        }
        previous = Some(context.work);
    }
}

#[test]
fn guarded_exact_work_and_one_short_late_transition_keep_full_reservation() {
    let probe = root();
    let source = input(&probe, model());
    let successful = construct(&source, work(&probe), limits()).unwrap();
    let exact_work = successful.payload().work;
    let factories = successful.payload().factories;
    drop(successful);
    for short in [false, true] {
        let allowance = exact_work - u128::from(short);
        let outcome = construct(
            &source,
            work(&probe),
            Limits {
                max_work: allowance,
            },
        )
        .unwrap();
        assert_eq!(outcome.payload().result.is_err(), short);
        assert_eq!(outcome.payload().work, allowance);
        if short {
            assert!(outcome.payload().factories < factories);
        }
        let resources = outcome
            .payload()
            .resources
            .expect("complete quote before construction");
        assert_eq!(
            outcome.charged_bytes(),
            resources.final_working_payload_bytes + size_of::<OwnedExternalCharge>()
        );
        assert!(std::ptr::eq(outcome.budget(), source.budget()));
    }
}

#[test]
fn pending_schema_cannot_overlap_native_allowance_minimal_regression() {
    let value = serde_json::json!({});
    let pool = Context::pool_work().unwrap();
    // Reviewer's state, normalized by the independently prepaid pool work:
    // remaining=2, schema limit=2, already pending schema work=1.
    let mut context = Context::new(true, pool + 2).unwrap();
    let mut budget = MetadataBudget {
        work_limit: 2,
        ..Default::default()
    };
    budget.node(4).unwrap();
    assert!(context.metadata(&mut budget, &value, 4).is_err());
    assert_eq!(context.work, pool + 1);
    assert_eq!(budget.work, 1);
    assert_eq!(context.metadata_events, 0);

    // Real disjoint schema prepayment: empty object needs one Value event. The
    // source-derived credit also conservatively reserves one possible String.
    for free in [5, 6] {
        let mut context = Context::new(true, pool + free).unwrap();
        context
            .prepay_json_schema(json::Usage {
                nodes: 1,
                ..Default::default()
            })
            .unwrap();
        let mut budget = MetadataBudget {
            work_limit: context.remaining(),
            ..Default::default()
        };
        budget.node(4).unwrap();
        let result = context.metadata(&mut budget, &value, 4);
        assert_eq!(result.is_ok(), free == 6);
        assert!(context.work <= context.limit);
        assert_eq!(context.metadata_events, 1);
        assert_eq!(budget.work, budget.settled_work);
    }
}

#[test]
fn fixed_pool_loops_are_admitted_before_initialization_and_retirement() {
    let n = CANONICAL_SLOTS as u128;
    let initialization = n + n + n + n; // native, left, right, stream
    let retirement = n + n + n + n; // same four explicit Drop loops
    let exact = initialization + retirement;
    assert_eq!(Context::pool_work().unwrap(), exact);
    assert!(Context::new(true, exact - 1).is_err());
    let context = Context::new(true, exact).unwrap();
    assert_eq!(context.work, exact);
    assert_eq!(context.remaining(), 0);
    drop(context);
}

#[test]
fn full_admission_refuses_during_metadata_not_at_late_settlement() {
    let root = root();
    let source = input(&root, model());
    let baseline = prepare(&source, work(&root));
    let first = baseline.context.first_metadata_work.unwrap();
    assert!(baseline.context.metadata_events > 1);
    drop(baseline);
    // First attribute Value event fits; the following native transition does
    // not. Native census and schema prepayment have already been admitted.
    let outcome = construct(
        &source,
        work(&root),
        Limits {
            max_work: first + 1,
        },
    )
    .unwrap();
    assert!(matches!(
        outcome.payload().result.as_ref(),
        Err(InventoryFailure::Json(json::Error::WorkLimit))
    ));
    assert_eq!(outcome.payload().factories, 0);
    assert_eq!(outcome.payload().work, first + 1);
}

#[test]
fn frozen_head_competing_phase_errors_and_cross_phase_source_conflicts() {
    for case in 0..5 {
        let mut plan = model();
        let expected = match case {
            0 => {
                plan.prefill.operations[0].inputs[0] = "unbound.prefill".into();
                plan.decode.operations[0].attributes["weight"] = Value::Null;
                "matrix inventory unbound operation input ID"
            }
            1 => {
                plan.prefill.operations[0].attributes["weight"] = Value::Null;
                plan.decode.operations[0].inputs[0] = "unbound.decode".into();
                "matrix inventory head tied declaration differs from source ownership"
            }
            2 => {
                plan.prefill.operations[0].attributes["weight"] =
                    Value::String("input.tokens".into());
                let head = plan
                    .prefill
                    .operations
                    .iter_mut()
                    .find(|op| op.operator == ModelOperator::OutputHead)
                    .unwrap();
                head.attributes["weight"] = Value::String("input.tokens".into());
                plan.decode.operations[0].inputs[0] = "unbound.decode".into();
                "matrix inventory source artifact ID aliases a graph value"
            }
            3 => {
                let head = plan
                    .decode
                    .operations
                    .iter_mut()
                    .find(|op| op.operator == ModelOperator::OutputHead)
                    .unwrap();
                head.output_shape[1] += 1;
                let linear = plan
                    .decode
                    .operations
                    .iter_mut()
                    .find(|op| op.operator == ModelOperator::Linear)
                    .unwrap();
                linear.attributes["unsupported"] = Value::Bool(true);
                "matrix inventory inconsistent source artifact shape"
            }
            4 => {
                *plan.prefill.state_outputs[0].shape.last_mut().unwrap() += 1;
                plan.decode.operations[0].attributes = Value::Null;
                "matrix inventory attributes must be an object"
            }
            _ => unreachable!(),
        };
        // Frozen HEAD order: whole-plan preflight; prefill refs/head/group/source;
        // decode equivalents; only then compatibility. Never use new-vs-new as
        // the expected error oracle.
        assert_eq!(
            lower_decoder_matrix_inventory(&plan).unwrap_err(),
            expected,
            "public case {case}"
        );
        let root = root();
        let source = input(&root, plan);
        let outcome = construct(&source, work(&root), limits()).unwrap();
        assert_eq!(
            outcome.payload().result.as_ref().unwrap_err().to_string(),
            expected,
            "guarded case {case}"
        );
    }
}

#[test]
fn pressure_after_quote_refuses_before_factories_and_erases_before_refund() {
    for inline_room in [false, true] {
        let observation = Cell::new(None);
        let root = root();
        let source = input(&root, model());
        let mut quote = prepare(&source, work(&root));
        quote.context.drop_observer = Some(DropObserver {
            budget: source.budget(),
            observation: &observation,
        });
        let working_charge = size_of::<OwnedExternalCharge>() + quote.working.payload_bytes();
        let leave = if inline_room {
            quote.resources.outcome_inline_bytes
        } else {
            0
        };
        let available = MAX_ALLOCATION_BYTES - source.budget().used_bytes();
        let pressure = root
            .charge_external(available - leave - size_of::<OwnedExternalCharge>())
            .unwrap();
        let before = source.budget().used_bytes();
        let result = quote.construct();
        let (at_retirement, cleared) = observation.get().expect("actual Context drop observed");
        assert_eq!(at_retirement, before);
        assert_eq!(cleared, 4 * CANONICAL_SLOTS);
        if inline_room {
            let outcome = result.unwrap();
            assert!(matches!(
                outcome.payload().result.as_ref(),
                Err(InventoryFailure::Native(NativeError::Resource))
            ));
            assert_eq!(outcome.payload().factories, 0);
            assert_eq!(outcome.budget().used_bytes(), before + leave);
            drop(outcome);
        } else {
            assert!(matches!(result, Err(NativeError::Resource)));
        }
        assert_eq!(source.budget().used_bytes(), before - working_charge);
        drop(pressure);
    }
}

#[test]
fn pool_erasure_observed_with_full_guard_on_success_error_and_unwind() {
    for kind in 0..3 {
        let observation = Cell::new(None);
        let root = root();
        let source = input(&root, model());
        let baseline = source.budget().used_bytes();
        let mut quote = prepare(&source, work(&root));
        let expected = quote.resources.total_original_counter_bytes;
        quote.context.drop_observer = Some(DropObserver {
            budget: source.budget(),
            observation: &observation,
        });
        if kind == 1 {
            quote.context.fail_after = Some(3);
        }
        if kind == 2 {
            quote.context.panic_after = Some(3);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| quote.construct()));
        let (used, cleared) = observation.get().expect("actual pool erasure hook");
        assert_eq!(used, expected);
        assert_eq!(cleared, 4 * CANONICAL_SLOTS);
        if kind == 2 {
            assert!(result.is_err());
        } else {
            let outcome = result.unwrap().unwrap();
            assert_eq!(outcome.payload().result.is_err(), kind == 1);
            drop(outcome);
        }
        assert_eq!(source.budget().used_bytes(), baseline);
    }
}
