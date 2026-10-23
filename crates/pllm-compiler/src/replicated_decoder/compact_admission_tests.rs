//! Concrete compact publication + existing body resource/constructor quotes.
//! No full-decoder execution or qualification claim. Parent serializes execution.
use super::*;
use crate::replicated_decoder::{ExecutionPolicy, Interval, SourceNorms, SourceTensor, WeightMode};
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeMap, sync::Arc};

fn with_compact(lanes: usize, run: impl FnOnce(&CompactGraph<'_, '_>)) {
    with_compact_parts(lanes, |compact, _| run(compact));
}

fn with_compact_parts(lanes: usize, run: impl FnOnce(&CompactGraph<'_, '_>, &DecoderGraph)) {
    with_operation(lanes, pllm_models::ModelOperator::Multiply, run);
}

fn with_operation(
    lanes: usize,
    operator: pllm_models::ModelOperator,
    run: impl FnOnce(&CompactGraph<'_, '_>, &DecoderGraph),
) {
    with_operation_prefix(lanes, operator, false, |compact, semantic| {
        run(compact, semantic)
    });
}

// Optional identity suffix shares the SAME admitted descriptor/frontier backing
// before and after publication. Source validation still uses the original graph.
fn with_operation_prefix(
    lanes: usize,
    operator: pllm_models::ModelOperator,
    feedback: bool,
    run: impl FnOnce(&mut CompactGraph<'_, '_>, &DecoderGraph),
) {
    use super::super::super::{
        lower_with_weights, validate_source_records, Builder, Code, Op, Program,
    };
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type":"llama", "hidden_size":2, "intermediate_size":lanes,
        "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
        "head_dim":2, "vocab_size":3, "tie_word_embeddings":false,
        "hidden_act":"silu", "attention_bias":false, "max_position_embeddings":64,
        "rope_theta":10000, "rms_norm_eps":0.000001, "attention_dropout":0.0,
        "use_sliding_window":false, "use_cache":true, "max_window_layers":1
    }))
    .unwrap();
    let shapes = Program::source_shapes_with_weights(&config, 1, 1, WeightMode::Public).unwrap();
    let workload = lower_with_weights(&config, 1, 1, WeightMode::Public).unwrap();
    let semantic = &workload.prefill;
    let coefficients: Vec<_> = semantic
        .operations
        .iter()
        .filter(|op| op.operator == Op::RmsNorm)
        .map(|op| op.attributes["weight"].as_str().unwrap())
        .collect();
    let sources: BTreeMap<_, _> = shapes
        .iter()
        .map(|(key, shape)| {
            let n = shape.iter().product::<usize>();
            (
                key.clone(),
                SourceTensor {
                    shape: shape.clone(),
                    lower: 0,
                    upper: 0,
                    public_q18: coefficients.contains(&key.as_str()).then(|| vec![0; n]),
                    digest: [7; 32],
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
            )
        })
        .collect();
    // Public coefficients retain their actual encoded commitment. No fake
    // source/certificate capability is supplied to CompactGraph.
    let mut sources = sources;
    for source in sources.values_mut() {
        if let Some(values) = &source.public_q18 {
            let mut hash = Sha256::new();
            for value in values {
                hash.update(value.to_le_bytes());
            }
            source.digest = hash.finalize().into();
        }
    }
    let norms = SourceNorms::new();
    let validated = validate_source_records(
        semantic,
        &shapes,
        &sources,
        WeightMode::Public,
        ExecutionPolicy::default(),
        &norms,
    )
    .unwrap();
    let mut imports = Builder {
        public_weights: true,
        ..Builder::default()
    };
    for (key, source) in &sources {
        if source.public_q18.is_some() {
            continue;
        }
        imports
            .push(
                Code::PublicInput {
                    key: key.clone(),
                    digest: source.digest,
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
                source.shape.iter().product(),
                Interval::new(0, 0).unwrap(),
            )
            .unwrap();
    }
    let (start, operation) = semantic
        .operations
        .iter()
        .enumerate()
        .rev()
        .find(|(_, op)| op.operator == operator)
        .unwrap();
    let input_shape: Vec<_> = semantic
        .operations
        .iter()
        .find(|op| op.id == operation.inputs[0])
        .unwrap()
        .output_shape
        .iter()
        .map(|n| *n as usize)
        .collect();
    let input_lanes = input_shape.iter().product();
    let left = imports
        .push(Code::Input("left".into()), input_lanes, domain())
        .unwrap();
    let right = imports
        .push(Code::Input("right".into()), input_lanes, domain())
        .unwrap();
    let left = imports.view(left, input_shape.clone());
    let right = imports.view(right, input_shape);
    let region = feedback.then(|| {
        let mut region = semantic.clone();
        let mut identity = semantic
            .operations
            .iter()
            .find(|op| op.operator == Op::TokenFeedback)
            .unwrap()
            .clone();
        identity.inputs = vec![operation.id.clone()];
        identity.output_shape = operation.output_shape.clone();
        region.output = identity.id.clone();
        region.operations = vec![operation.clone(), identity];
        region
    });
    let (semantic, start) = region
        .as_ref()
        .map_or((semantic, start), |region| (region, 0));
    let mut compact = CompactGraph::new(&imports, validated, semantic, start).unwrap();
    compact.bind_import(&operation.inputs[0], &left).unwrap();
    if operation.inputs.len() == 2 {
        compact.bind_import(&operation.inputs[1], &right).unwrap();
    }
    compact.lower_next().unwrap();
    run(&mut compact, semantic);
}

fn domain() -> Interval {
    Interval::new(-1, 1).unwrap()
}

fn owners(values: &Vec<Option<Secret>>) -> RequestOwners<'_> {
    RequestOwners {
        borrowed_external_bytes: 101,
        owned_source_bytes: 103,
        other_owned_metadata_bytes: 107,
        other_stored_descriptors: 3,
        other_work: WorkBudget::local(),
        retained_values: values,
        external_frame_bytes: 109,
        caller_control_bytes: 113,
        prior_request_peak_bytes: 0,
    }
}

fn inputs(lanes: usize) -> [Register; 2] {
    [Register {
        lanes,
        bounds: domain(),
    }; 2]
}

fn body(lanes: usize) -> (layer::Builder, Layout) {
    let mut builder = layer::Builder::new(WeightMode::Public, &inputs(lanes)).unwrap();
    let left = builder.input(0).unwrap();
    let right = builder.input(1).unwrap();
    let output = builder.binary(&left, &right, true).unwrap();
    (builder, output)
}

// Rebuild from the fixture's actual original imports and validated sources so
// the consuming freeze can outlive a local semantic graph. No capability or
// allocation baseline is supplied by the test.
fn replay_single_operation<'g, 'a>(
    fixture: &CompactGraph<'_, 'a>,
    semantic: &'g DecoderGraph,
) -> CompactGraph<'g, 'a> {
    use super::super::super::validate_source_records;

    let sources = fixture.borrowed_source_records();
    let norms = fixture.borrowed_source_norms();
    let shapes = sources
        .iter()
        .map(|(key, source)| (key.clone(), source.shape.clone()))
        .collect();
    let validated = validate_source_records(
        semantic,
        &shapes,
        sources,
        WeightMode::Public,
        ExecutionPolicy::default(),
        norms,
    )
    .unwrap();
    let start = fixture.cursor - fixture.ops.len();
    assert_eq!(semantic.operations[start].inputs.len(), 1);
    let original = &fixture.ops[0].input;
    let super::super::Mapping::Dense(owner) = original.mapping else {
        panic!("original dense import required");
    };
    let input = fixture
        .imports
        .view(owner, original.logical.dimensions().to_vec());
    let mut compact = CompactGraph::new(fixture.imports, validated, semantic, start).unwrap();
    compact
        .bind_import(&semantic.operations[start].inputs[0], &input)
        .unwrap();
    compact.lower_next().unwrap();
    compact
}

fn request_snapshot(request: &RequestWork) -> (WorkBudget, usize, usize, usize) {
    (
        request.work(),
        request.stored_descriptors,
        request.retained_graph_bytes,
        request.regions,
    )
}

// Compose public schedule quotes, never forge WorkBudget's private counters.
// The wide conversion keeps ordinal usage below its independent complete-request
// cap when targeting the AES cap. Two- and five-block schedules fill its tail.
fn schedule_with_exact_aes(aes: u64) -> WorkBudget {
    use crate::replicated_decoder::puma::MAX_LANES;

    // Carry gates double the input width; their native lane cap is unchanged.
    let wide = WorkBudget::conversion_bounded(MAX_LANES / 2, 64).unwrap();
    let width = wide.cursors().aes_work;
    let mut count = aes / width;
    let mut tail = aes % width;
    if tail == 1 || tail == 3 {
        count = count.checked_sub(1).unwrap();
        tail += width;
    }
    let mut work = wide.checked_mul(count).unwrap();
    if tail % 2 == 1 {
        let odd = WorkBudget::from_boolean(1, 1).unwrap();
        assert_eq!(odd.cursors().aes_work, 5);
        work = work.checked_add(odd).unwrap();
        tail -= 5;
    }
    let pair = WorkBudget::product(1, false).unwrap();
    assert_eq!(pair.cursors().aes_work, 2);
    work = work
        .checked_add(pair.checked_mul(tail / 2).unwrap())
        .unwrap();
    assert_eq!(work.cursors().aes_work, aes);
    work
}

#[test]
fn actual_compact_metadata_and_physical_retained_capacities_add_without_refund() {
    with_compact(129, |compact| {
        let empty = Vec::new();
        let base = RequestQuote::new(compact, owners(&empty)).unwrap();
        assert!(base.baseline_bytes() > compact.baseline().unwrap() + 101 + 103 + 107 + 109 + 113);
        assert_eq!(base.stored_descriptors(), compact.stored_descriptors() + 4);
        assert_eq!(
            base.complete_work(),
            WorkBudget::local()
                .checked_add(compact.work_budget())
                .unwrap()
        );
        let mut retained = Vec::with_capacity(9);
        retained.push(Some(Secret::new(17).unwrap()));
        retained.push(None);
        let expected =
            retained_values_bytes(&retained).unwrap() - retained_values_bytes(&empty).unwrap();
        let held = RequestQuote::new(compact, owners(&retained)).unwrap();
        assert_eq!(held.baseline_bytes() - base.baseline_bytes(), expected);
        assert_eq!(retained[0].as_ref().unwrap().first.len(), 17);
    });
}

#[test]
fn request_baseline_exact_fit_one_short_overflow_and_descriptor_ceiling() {
    with_compact(129, |compact| {
        let values = Vec::new();
        let exact = RequestQuote::new(compact, owners(&values))
            .unwrap()
            .baseline_bytes();
        assert!(RequestQuote::with_limit(compact, owners(&values), exact).is_ok());
        assert!(matches!(
            RequestQuote::with_limit(compact, owners(&values), exact - 1),
            Err(Error::Bounds)
        ));
        let mut overflow = owners(&values);
        overflow.owned_source_bytes = usize::MAX;
        assert!(matches!(
            RequestQuote::new(compact, overflow),
            Err(Error::Bounds)
        ));
        let mut fits = owners(&values);
        fits.other_stored_descriptors = MAX_NODES - compact.stored_descriptors() - 1;
        assert_eq!(
            RequestQuote::new(compact, fits)
                .unwrap()
                .stored_descriptors(),
            MAX_NODES
        );
        let mut too_many = owners(&values);
        too_many.other_stored_descriptors = MAX_NODES - compact.stored_descriptors();
        assert!(matches!(
            RequestQuote::new(compact, too_many),
            Err(Error::Bounds)
        ));
    });
}

#[test]
fn whole_request_work_and_earlier_peaks_cannot_reset_at_region_boundary() {
    with_compact(129, |compact| {
        let values = Vec::new();
        let mut excessive = owners(&values);
        excessive.other_work = WorkBudget::local()
            .checked_mul(crate::replicated_decoder::puma::MAX_PROGRAM_OPERATIONS)
            .unwrap();
        assert!(matches!(
            RequestQuote::new(compact, excessive),
            Err(Error::Bounds)
        ));
        let mut prior = owners(&values);
        prior.prior_request_peak_bytes = MAX_PROGRAM_BYTES + 1;
        assert!(matches!(
            RequestQuote::new(compact, prior),
            Err(Error::Bounds)
        ));
        let quote = RequestQuote::new(compact, owners(&values)).unwrap();
        assert!(matches!(
            quote.admit_total(quote.baseline_bytes() - 1),
            Err(Error::Binding)
        ));
        assert!(quote.admit_total(MAX_PROGRAM_BYTES).is_ok());
        assert!(matches!(
            quote.admit_total(MAX_PROGRAM_BYTES + 1),
            Err(Error::Bounds)
        ));
        let mut held_peak = owners(&values);
        held_peak.prior_request_peak_bytes = MAX_PROGRAM_BYTES;
        let held_peak = RequestQuote::new(compact, held_peak).unwrap();
        let (builder, output) = body(129);
        let plan = builder.finish(&[output]).unwrap();
        assert_eq!(
            held_peak.body_phases(&plan, &[]).unwrap().peak_bytes,
            MAX_PROGRAM_BYTES
        );
    });
}

#[test]
fn input_and_native_preflights_use_existing_quotes_before_constructor_entry() {
    with_compact(129, |compact| {
        let values = Vec::new();
        let layouts = [
            Layout::quote_contiguous(0, 129).unwrap(),
            Layout::quote_contiguous(1, 129).unwrap(),
        ];
        let quote = RequestQuote::new(compact, owners(&values)).unwrap();
        let input_peak = quote.preflight_body_inputs(&inputs(129), &layouts).unwrap();
        let exact = input_peak.peak_bytes;
        let fitting = RequestQuote::with_limit(compact, owners(&values), exact).unwrap();
        assert_eq!(
            fitting
                .preflight_body_inputs(&inputs(129), &layouts)
                .unwrap(),
            input_peak
        );
        let short = RequestQuote::with_limit(compact, owners(&values), exact - 1).unwrap();
        assert!(matches!(
            short.preflight_body_inputs(&inputs(129), &layouts),
            Err(Error::Bounds)
        ));
        let attempts = crate::replicated_decoder::prepared_tensor_map::native_attempts();
        let native = quote
            .preflight_native(tensor::Construction::GuardedSilu { lanes: 128 })
            .unwrap();
        assert!(native.peak_bytes > quote.baseline_bytes());
        assert_eq!(
            attempts,
            crate::replicated_decoder::prepared_tensor_map::native_attempts()
        );
        let fitting =
            RequestQuote::with_limit(compact, owners(&values), native.peak_bytes).unwrap();
        assert_eq!(
            fitting
                .preflight_native(tensor::Construction::GuardedSilu { lanes: 128 })
                .unwrap(),
            native
        );
        let short =
            RequestQuote::with_limit(compact, owners(&values), native.peak_bytes - 1).unwrap();
        assert!(matches!(
            short.preflight_native(tensor::Construction::GuardedSilu { lanes: 128 }),
            Err(Error::Bounds)
        ));
        // Physical import geometry comes from CompactGraph, not an arbitrary
        // smaller Plan/input specification with a convenient resource quote.
        assert!(matches!(
            quote.preflight_body_inputs(&inputs(128), &layouts),
            Err(Error::Binding)
        ));
        let mut malformed = layouts;
        malformed[0].construction_peak_bytes = usize::MAX;
        assert!(matches!(
            quote.preflight_body_inputs(&inputs(129), &malformed),
            Err(Error::Bounds)
        ));
    });
}

#[test]
#[allow(clippy::vec_init_then_push)] // Preserve actual spare-capacity inventory.
fn actual_body_finish_and_liveness_add_retained_owners_and_restoration_overlap() {
    for lanes in [1, 128, 129, 257] {
        with_compact(lanes, |compact| {
            let mut values = Vec::new();
            values.push(Some(Secret::new(3).unwrap()));
            let quote = RequestQuote::new(compact, owners(&values)).unwrap();
            let (builder, output) = body(lanes);
            let construction = quote
                .preflight_body_finish(&builder, std::slice::from_ref(&output))
                .unwrap();
            let expected = builder
                .proof_finish_quote(
                    std::slice::from_ref(&output),
                    quote.baseline_bytes() + size_of::<Result<Arc<layer::Plan>, Error>>(),
                )
                .unwrap();
            assert_eq!(construction.construction_bytes, expected);
            let fitting = RequestQuote::with_limit(compact, owners(&values), expected).unwrap();
            assert!(fitting
                .preflight_body_finish(&builder, std::slice::from_ref(&output))
                .is_ok());
            let short = RequestQuote::with_limit(compact, owners(&values), expected - 1).unwrap();
            assert!(matches!(
                short.preflight_body_finish(&builder, std::slice::from_ref(&output)),
                Err(Error::Bounds)
            ));
            let plan = builder.finish(&[output]).unwrap();
            let (mut wrong_builder, wrong_output) = body(lanes);
            let wrong_output = wrong_builder
                .binary(&wrong_output, &wrong_output, false)
                .unwrap();
            let wrong_plan = wrong_builder.finish(&[wrong_output]).unwrap();
            assert!(matches!(
                quote.body_phases(&wrong_plan, &[]),
                Err(Error::Binding)
            ));
            let phases = quote.body_phases(&plan, &[construction]).unwrap();
            assert!(
                phases.execution_bytes
                    >= quote.baseline_bytes() + plan.resources().admitted_bytes + 2 * lanes * 16
            );
            assert!(phases.restoration_bytes >= phases.execution_bytes + lanes * 16);
            assert_eq!(
                phases.peak_bytes,
                phases
                    .construction_bytes
                    .max(phases.execution_bytes)
                    .max(phases.restoration_bytes)
            );
            let exact =
                RequestQuote::with_limit(compact, owners(&values), phases.peak_bytes).unwrap();
            assert_eq!(exact.body_phases(&plan, &[construction]).unwrap(), phases);
            let short =
                RequestQuote::with_limit(compact, owners(&values), phases.peak_bytes - 1).unwrap();
            assert!(matches!(
                short.body_phases(&plan, &[construction]),
                Err(Error::Bounds)
            ));
            let mut held = owners(&values);
            held.other_owned_metadata_bytes += 4093;
            let retained = RequestQuote::new(compact, held)
                .unwrap()
                .body_phases(&plan, &[])
                .unwrap();
            let plain = quote.body_phases(&plan, &[]).unwrap();
            assert_eq!(retained.execution_bytes - plain.execution_bytes, 4093);
            assert_eq!(retained.restoration_bytes - plain.restoration_bytes, 4093);
        });
    }
}

#[test]
fn physical_census_uses_actual_output_tiles_not_virtual_node_arrays() {
    for lanes in [1, 128, 129, 257] {
        with_compact(lanes, |compact| {
            let values = Vec::new();
            let quote = RequestQuote::new(compact, owners(&values)).unwrap();
            let census = quote.owner_census();
            let containers = compact.container_census().unwrap();
            assert_eq!(census.output_slots, lanes.div_ceil(128));
            assert_eq!(
                census.output_slot_bytes,
                lanes.div_ceil(128) * size_of::<Option<Secret>>()
            );
            assert_eq!(census.retained_output_bytes, lanes * 16);
            assert_eq!(
                census.stored_descriptors,
                containers.imported_descriptors + containers.stored_operations
            );
            assert_eq!(census.body_descriptors, Some(1));
            assert_eq!(census.prepared_program_bytes, 0);
            assert_eq!(census.metadata_bytes, compact.baseline().unwrap());
            assert!(census.remaining_operations > 0); // No full-decoder claim.
            assert!(census.output_slots < containers.expanded_virtual_nodes);
        });
    }
}

#[test]
fn early_preflight_matches_constructor_inventory_exact_one_short_overflow_and_prior_pressure() {
    with_compact_parts(129, |compact, semantic| {
        let values = Vec::new();
        let outside = owners(&values);
        let prior = RequestWork::default();
        let start = compact.cursor - compact.ops.len();
        let containers = compact.container_census().unwrap();
        let references: usize = semantic.operations.iter().map(|op| op.inputs.len()).sum();
        // Reconcile the constructor's actual reservation, including unused
        // capacity, against every semantic input reference (not unique names).
        assert_eq!(containers.import_binding_capacity, references);
        assert!(containers.import_binding_capacity > compact.bound_imports.len());
        assert_eq!(
            containers.import_binding_bytes,
            references * size_of::<super::super::BoundImport<'_>>()
        );
        assert_eq!(
            CompactGraph::import_binding_storage_bound(semantic).unwrap(),
            containers.import_binding_bytes
        );
        assert_eq!(
            containers.descriptor_capacity,
            semantic.operations.len() - start
        );
        // Independent named inventory: use actual constructor capacities rather
        // than the preflight result or a second invocation of its calculation.
        // Lowered native/work owners are not part of this initial container quote.
        let graph_bytes: usize = [
            CompactGraph::imports_bytes(compact.imports).unwrap(),
            containers.inline_bytes,
            containers.descriptor_bytes,
            containers.frontier_storage_bound,
            containers.import_binding_bytes,
            2 * size_of::<Result<CompactGraph<'_, '_>, Error>>(),
            containers.origin_arc_bytes,
            4 * size_of::<PublishedOp<'_>>(),
        ]
        .into_iter()
        .sum();
        let source = SourceOwnerCensus::read(
            compact.borrowed_source_records(),
            compact.borrowed_source_norms(),
        )
        .unwrap();
        let exact: usize = [
            graph_bytes,
            outside_bytes(&outside).unwrap(),
            source.borrowed_bytes,
            source.record_clone_bytes,
            semantic_bytes(semantic).unwrap(),
            prior.retained_graph_bytes,
            size_of::<RequestWork>(),
            size_of::<ConstructionPreflight>(),
            size_of::<Result<ConstructionPreflight, Error>>(),
            size_of::<SourceOwnerCensus>(),
            size_of::<Result<SourceOwnerCensus, Error>>(),
        ]
        .into_iter()
        .sum();
        let quote = preflight_graph_construction(
            compact.imports,
            semantic,
            start,
            compact.borrowed_source_records(),
            compact.borrowed_source_norms(),
            &prior,
            &outside,
        )
        .unwrap();
        assert_eq!(
            quote.unlowered_operations,
            semantic.operations.len() - start
        );
        assert_eq!(quote.accumulated_work, outside.other_work);
        assert!(quote.borrowed_source_bytes > 0 && quote.borrowed_semantic_bytes > 0);
        assert!(quote.new_source_record_bytes > 0 && quote.graph_container_bytes > 0);
        assert_eq!(quote.graph_container_bytes, graph_bytes);
        assert_eq!(quote.construction_bytes, exact);
        assert!(exact <= MAX_PROGRAM_BYTES);
        let fitting = preflight_graph_with_limit(
            compact.imports,
            semantic,
            start,
            compact.borrowed_source_records(),
            compact.borrowed_source_norms(),
            &prior,
            &outside,
            exact,
        )
        .unwrap();
        assert_eq!(fitting, quote);
        assert!(matches!(
            preflight_graph_with_limit(
                compact.imports,
                semantic,
                start,
                compact.borrowed_source_records(),
                compact.borrowed_source_norms(),
                &prior,
                &outside,
                exact - 1,
            ),
            Err(Error::Bounds)
        ));
        let mut overflow = owners(&values);
        overflow.owned_source_bytes = usize::MAX;
        assert!(matches!(
            preflight_graph_construction(
                compact.imports,
                semantic,
                start,
                compact.borrowed_source_records(),
                compact.borrowed_source_norms(),
                &prior,
                &overflow,
            ),
            Err(Error::Bounds)
        ));
        let mut held = owners(&values);
        held.other_owned_metadata_bytes += 8179;
        let held = preflight_graph_construction(
            compact.imports,
            semantic,
            start,
            compact.borrowed_source_records(),
            compact.borrowed_source_norms(),
            &prior,
            &held,
        )
        .unwrap();
        assert_eq!(held.construction_bytes - quote.construction_bytes, 8179);
    });
}

#[test]
fn actual_source_and_semantic_capacities_change_early_quote_without_allocating_clones() {
    with_compact_parts(129, |compact, semantic| {
        let sources = compact.borrowed_source_records();
        let norms = compact.borrowed_source_norms();
        let mut reserved = sources.clone();
        let before = SourceOwnerCensus::read(&reserved, norms).unwrap();
        let first = reserved.values_mut().next().unwrap();
        let old = first.shape.capacity();
        first.shape.reserve_exact(61);
        let actual = (first.shape.capacity() - old) * size_of::<usize>();
        let after = SourceOwnerCensus::read(&reserved, norms).unwrap();
        assert_eq!(after.borrowed_bytes - before.borrowed_bytes, actual);
        assert_eq!(after.record_clone_bytes - before.record_clone_bytes, actual);
        let before = semantic_bytes(semantic).unwrap();
        // A clone may trim other original capacities: compare the same clone
        // immediately before/after changing its operation buffer instead.
        let mut original_clone = semantic.clone();
        let clone_bytes = semantic_bytes(&original_clone).unwrap();
        let old = original_clone.operations.capacity();
        original_clone.operations.reserve_exact(37);
        let delta =
            (original_clone.operations.capacity() - old) * size_of::<pllm_models::ModelOperation>();
        assert_eq!(
            semantic_bytes(&original_clone).unwrap() - clone_bytes,
            delta
        );
        assert!(before > 0 && delta > 0);
    });
}

#[test]
fn empty_source_maps_keep_original_and_clone_control_owners() {
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let census = SourceOwnerCensus::read(&sources, &norms).unwrap();
    assert_eq!(
        census.borrowed_bytes,
        size_of::<BTreeMap<String, SourceTensor>>() + size_of::<SourceNorms>()
    );
    assert_eq!(
        census.record_clone_bytes,
        size_of::<Vec<(String, SourceTensor)>>()
            + size_of::<BTreeMap<String, SourceTensor>>()
            + 4 * size_of::<SourceNorms>()
    );
}

#[test]
fn early_original_importer_quote_replaces_record_estimate_and_keeps_all_prior_owners() {
    use super::super::super::validate_source_records;
    with_compact_parts(129, |compact, semantic| {
        let sources = compact.borrowed_source_records();
        let norms = compact.borrowed_source_norms();
        let shapes = sources
            .iter()
            .map(|(key, source)| (key.clone(), source.shape.clone()))
            .collect();
        let validated = validate_source_records(
            semantic,
            &shapes,
            sources,
            WeightMode::Public,
            ExecutionPolicy::default(),
            norms,
        )
        .unwrap();
        let registry = SourceRegistry::from_validated(validated).unwrap();
        let values = Vec::new();
        let outside = owners(&values);
        let mut prior = RequestWork::default();
        prior.record_graph(compact).unwrap();
        prior.record_graph(compact).unwrap();
        let start = compact.cursor - compact.ops.len();
        let quote = preflight_graph_and_source_import(
            compact.imports,
            semantic,
            start,
            &registry,
            &prior,
            &outside,
        )
        .unwrap();
        let import = quote.source_import_quote.unwrap();
        assert_eq!(quote.new_source_record_bytes, 0);
        assert_eq!(import.borrowed_metadata_bytes, quote.borrowed_source_bytes);
        assert_eq!(quote.construction_bytes, import.peak_bytes);
        assert_eq!(
            import.peak_bytes,
            import.retained_baseline_bytes
                + import.borrowed_metadata_bytes
                + import.new_owned_peak_bytes
        );
        assert_eq!(
            quote.accumulated_work,
            compact
                .work_budget()
                .checked_mul(2)
                .unwrap()
                .checked_add(outside.other_work)
                .unwrap()
        );
        // PublicSourceImport's real quote includes Pending, file handles/scans,
        // bounded frame and return slots absent from a record-only estimate.
        assert!(import.new_owned_peak_bytes > 2 * 16384 + size_of::<PublicSourceImport>());
        let exact = quote.construction_bytes;
        assert_eq!(
            graph_source_preflight(
                compact.imports,
                semantic,
                start,
                sources,
                norms,
                &prior,
                &outside,
                Some(&registry),
                exact
            )
            .unwrap(),
            quote
        );
        assert!(matches!(
            graph_source_preflight(
                compact.imports,
                semantic,
                start,
                sources,
                norms,
                &prior,
                &outside,
                Some(&registry),
                exact - 1
            ),
            Err(Error::Bounds)
        ));
        let mut held = owners(&values);
        held.owned_source_bytes += 4111;
        let held = preflight_graph_and_source_import(
            compact.imports,
            semantic,
            start,
            &registry,
            &prior,
            &held,
        )
        .unwrap();
        assert_eq!(held.construction_bytes - quote.construction_bytes, 4111);
        let mut overflow = owners(&values);
        overflow.owned_source_bytes = usize::MAX;
        assert!(matches!(
            preflight_graph_and_source_import(
                compact.imports,
                semantic,
                start,
                &registry,
                &prior,
                &overflow
            ),
            Err(Error::Bounds)
        ));
        let mut over_cap = owners(&values);
        over_cap.owned_source_bytes = MAX_PROGRAM_BYTES;
        assert!(matches!(
            graph_source_preflight(
                compact.imports,
                semantic,
                start,
                sources,
                norms,
                &prior,
                &over_cap,
                Some(&registry),
                usize::MAX
            ),
            Err(Error::Bounds)
        ));
    });
}

#[test]
fn repeated_selection_inventory_accumulates_original_work_and_retains_prior_graphs() {
    with_compact(129, |compact| {
        let values = Vec::new();
        let mut prior = RequestWork::default();
        prior.record_graph(compact).unwrap();
        let once_bytes = prior.retained_graph_bytes;
        assert_eq!(
            once_bytes,
            compact.baseline().unwrap()
                + SourceOwnerCensus::read(
                    compact.borrowed_source_records(),
                    compact.borrowed_source_norms(),
                )
                .unwrap()
                .borrowed_bytes
                + semantic_bytes(compact.frontier.semantic_graph()).unwrap()
        );
        prior.record_graph(compact).unwrap(); // Equal digest is NOT a work/owner refund.
        assert_eq!(prior.work(), compact.work_budget().checked_mul(2).unwrap());
        assert_eq!(prior.regions, 2);
        assert_eq!(prior.retained_graph_bytes, once_bytes * 2);
        let quote = RequestQuote::with_request_work(compact, owners(&values), &prior).unwrap();
        assert_eq!(
            quote.complete_work(),
            WorkBudget::local()
                .checked_add(compact.work_budget().checked_mul(3).unwrap())
                .unwrap()
        );
        let plain = RequestQuote::new(compact, owners(&values)).unwrap();
        assert_eq!(
            quote.baseline_bytes() - plain.baseline_bytes(),
            prior.retained_graph_bytes + size_of::<RequestWork>()
        );
        let before = (
            prior.work,
            prior.stored_descriptors,
            prior.retained_graph_bytes,
            prior.regions,
        );
        assert!(matches!(
            prior.record_work(
                WorkBudget::local()
                    .checked_mul(crate::replicated_decoder::puma::MAX_PROGRAM_OPERATIONS)
                    .unwrap()
            ),
            Err(Error::Bounds)
        ));
        assert_eq!(
            (
                prior.work,
                prior.stored_descriptors,
                prior.retained_graph_bytes,
                prior.regions
            ),
            before
        );
        prior.record_work(WorkBudget::local()).unwrap();
        assert_eq!(
            prior.work(),
            before.0.checked_add(WorkBudget::local()).unwrap()
        );
    });
}

#[test]
fn frozen_request_inventory_outlives_semantic_graph_repeats_work_and_rejects_atomically() {
    use super::super::super::validate_source_records;

    with_operation(129, pllm_models::ModelOperator::Silu, |fixture, full| {
        let imports = fixture.imports;
        let sources = fixture.borrowed_source_records();
        let norms = fixture.borrowed_source_norms();
        let super::super::Mapping::Dense(owner) = fixture.ops[0].input.mapping else {
            panic!("original dense import required");
        };
        let input = imports.view(owner, fixture.ops[0].input.logical.dimensions().to_vec());
        let (frozen, native_owners, work, descriptors) = {
            // Only this local clone supplies the semantic graph to the new
            // CompactGraph. Recording below cannot borrow it after scope exit.
            let semantic = full.clone();
            let shapes = sources
                .iter()
                .map(|(key, source)| (key.clone(), source.shape.clone()))
                .collect();
            let validated = validate_source_records(
                &semantic,
                &shapes,
                sources,
                WeightMode::Public,
                ExecutionPolicy::default(),
                norms,
            )
            .unwrap();
            let start = fixture.cursor - fixture.ops.len();
            let mut compact = CompactGraph::new(imports, validated, &semantic, start).unwrap();
            compact
                .bind_import(&semantic.operations[start].inputs[0], &input)
                .unwrap();
            let output = compact.lower_next().unwrap();
            assert!(compact.cursor < semantic.operations.len());
            let native_owners: Vec<_> = compact
                .owned_native_programs()
                .map(|program| {
                    (
                        Arc::as_ptr(program),
                        Arc::downgrade(program),
                        Arc::strong_count(program),
                    )
                })
                .collect();
            assert!(!native_owners.is_empty());
            let work = compact.work_budget();
            let descriptors = compact.stored_descriptors();
            let exports = vec![output];
            let quote = compact.freeze_quote(&exports).unwrap();
            let frozen = compact.freeze(exports, quote.peak_bytes).unwrap();
            (frozen, native_owners, work, descriptors)
            // semantic drops here. Imports, sources and norms still belong to
            // their original outer owners; freeze does not extend their lives.
        };
        let resources = frozen.resources();
        assert!(resources.retained_bytes > 0 && resources.retained_bytes < MAX_PROGRAM_BYTES);
        assert!(work.operations() > 0);
        assert_eq!(frozen.work_budget(), work);
        assert_eq!(descriptors, imports.nodes.len() + 1);
        assert!(resources.import_count < imports.nodes.len());
        assert_eq!(frozen.descriptor_census().count(), 1);
        assert!(
            frozen.output().is_none(),
            "partial region is not completion"
        );
        assert_eq!(frozen.source_binding(), fixture.binding);
        for (owner, recipe) in imports.nodes.iter().enumerate() {
            assert!(std::ptr::eq(frozen.imported_recipe(owner).unwrap(), recipe));
        }
        assert!(frozen.imported_recipe(imports.nodes.len()).is_none());

        let mut prior = RequestWork::default();
        let ordinary = WorkBudget::local();
        prior.record_work(ordinary).unwrap();
        for repetitions in 1..=2 {
            prior.record_frozen(&frozen).unwrap();
            assert_eq!(
                prior.work(),
                ordinary
                    .checked_add(work.checked_mul(repetitions).unwrap())
                    .unwrap()
            );
            assert_eq!(prior.stored_descriptors, descriptors * repetitions as usize);
            assert_eq!(
                prior.retained_graph_bytes,
                resources.retained_bytes * repetitions as usize
            );
            assert_eq!(prior.regions, repetitions as usize);
            let programs = frozen.prepared(0).unwrap();
            assert_eq!(programs.programs().count(), native_owners.len());
            for (program, (pointer, weak, strong)) in programs.programs().zip(&native_owners) {
                assert_eq!(Arc::as_ptr(program), *pointer);
                assert_eq!(weak.as_ptr(), *pointer);
                assert_eq!(Arc::strong_count(program), *strong);
            }
        }

        // Exact retained bytes and descriptors both reach their ORIGINAL caps.
        let mut exact = RequestWork {
            work: ordinary,
            stored_descriptors: MAX_NODES - descriptors,
            retained_graph_bytes: MAX_PROGRAM_BYTES - resources.retained_bytes,
            regions: 3,
        };
        exact.record_frozen(&frozen).unwrap();
        assert_eq!(exact.work(), ordinary.checked_add(work).unwrap());
        assert_eq!(exact.stored_descriptors, MAX_NODES);
        assert_eq!(exact.retained_graph_bytes, MAX_PROGRAM_BYTES);
        assert_eq!(exact.regions, 4);

        let max_work = ordinary
            .checked_mul(crate::replicated_decoder::puma::MAX_PROGRAM_OPERATIONS)
            .unwrap();
        for (work, descriptors_before, bytes_before, regions_before) in [
            (ordinary, MAX_NODES - descriptors + 1, 211, 3),
            (ordinary, usize::MAX, 211, 3),
            (
                ordinary,
                7,
                MAX_PROGRAM_BYTES - resources.retained_bytes + 1,
                3,
            ),
            (ordinary, 7, usize::MAX, 3),
            (ordinary, 7, 211, usize::MAX),
            (max_work, 7, 211, 3),
        ] {
            let mut held = RequestWork {
                work,
                stored_descriptors: descriptors_before,
                retained_graph_bytes: bytes_before,
                regions: regions_before,
            };
            assert!(matches!(held.record_frozen(&frozen), Err(Error::Bounds)));
            assert_eq!(held.work(), work);
            assert_eq!(held.stored_descriptors, descriptors_before);
            assert_eq!(held.retained_graph_bytes, bytes_before);
            assert_eq!(held.regions, regions_before);
        }
        for (pointer, weak, strong) in &native_owners {
            assert_eq!(weak.as_ptr(), *pointer);
            assert_eq!(weak.strong_count(), *strong);
        }
    });
}

#[test]
fn frozen_alias_only_regions_retain_recipes_and_backing_without_numeric_or_body_work() {
    use pllm_models::ModelOperator::{Reshape, TokenFeedback};

    for operator in [TokenFeedback, Reshape] {
        with_operation(129, operator, |fixture, full| {
            let (frozen, descriptors, words) = {
                let semantic = full.clone();
                let compact = replay_single_operation(fixture, &semantic);
                let census = CompactOwnerCensus::read(&compact).unwrap();
                assert_eq!(census.work, WorkBudget::default());
                assert_eq!(census.body_descriptors, Some(0));
                assert_eq!(census.output_slots, 0);
                assert_eq!(census.output_slot_bytes, 0);
                assert_eq!(census.retained_output_bytes, 0);
                assert_eq!(compact.owned_native_programs().count(), 0);
                let output = compact.ops[0].output.clone();
                let words = output.words().collect::<Vec<_>>();
                let descriptors = compact.stored_descriptors();
                let exports = vec![output];
                let quote = compact.freeze_quote(&exports).unwrap();
                assert_eq!(quote.native_owner_bytes, 0);
                assert!(quote.recipe_capacity > 0 && quote.recipe_bytes > 0);
                assert!(quote.borrowed_owner_bytes > 0 && quote.new_backing_bytes > 0);
                assert!(quote.retained_bytes >= quote.recipe_bytes + quote.borrowed_owner_bytes);
                quote.admit(quote.peak_bytes).unwrap();
                assert_eq!(quote.admit(quote.peak_bytes - 1), Err(Error::Bounds));
                let frozen = compact.freeze(exports, quote.peak_bytes).unwrap();
                assert_eq!(frozen.resources(), quote);
                (frozen, descriptors, words)
            };
            let resources = frozen.resources();
            assert_eq!(frozen.work_budget(), WorkBudget::default());
            let descriptor = frozen.descriptor_census().next().unwrap();
            assert!(matches!(
                descriptor.kind,
                DescriptorKind::TokenFeedback | DescriptorKind::Reshape
            ));
            assert_eq!(descriptor.virtual_start, descriptor.virtual_end);
            assert!(matches!(frozen.prepared(0), Err(Error::Binding)));
            assert!(frozen.native_row_program(0).is_none());
            for (lane, expected) in words.into_iter().enumerate() {
                assert_eq!(frozen.word(&frozen.exports()[0], lane).unwrap(), expected);
            }
            for (owner, recipe) in fixture.imports.nodes.iter().enumerate() {
                assert!(std::ptr::eq(frozen.imported_recipe(owner).unwrap(), recipe));
            }
            assert_eq!(descriptors, fixture.imports.nodes.len() + 1);
            assert!(resources.import_count < fixture.imports.nodes.len());

            // An alias-only prefix cannot refund earlier ordinary work, nor can
            // repeating the same frozen recipe refund retained backing.
            let ordinary = WorkBudget::rescale(1).unwrap();
            let mut prior = RequestWork::default();
            prior.record_work(ordinary).unwrap();
            for repetitions in 1..=2 {
                prior.record_frozen(&frozen).unwrap();
                assert_eq!(prior.work(), ordinary);
                assert_eq!(prior.regions, repetitions);
                assert_eq!(prior.stored_descriptors, descriptors * repetitions);
                assert_eq!(
                    prior.retained_graph_bytes,
                    resources.retained_bytes * repetitions
                );
            }

            let mut exact = RequestWork {
                work: ordinary,
                stored_descriptors: MAX_NODES - descriptors,
                retained_graph_bytes: MAX_PROGRAM_BYTES - resources.retained_bytes,
                regions: prior.regions,
            };
            exact.record_frozen(&frozen).unwrap();
            assert_eq!(exact.work(), ordinary);
            assert_eq!(exact.stored_descriptors, MAX_NODES);
            assert_eq!(exact.retained_graph_bytes, MAX_PROGRAM_BYTES);
            for (stored_descriptors, retained_graph_bytes) in [
                (MAX_NODES - descriptors + 1, prior.retained_graph_bytes),
                (
                    prior.stored_descriptors,
                    MAX_PROGRAM_BYTES - resources.retained_bytes + 1,
                ),
            ] {
                let mut short = RequestWork {
                    work: prior.work(),
                    stored_descriptors,
                    retained_graph_bytes,
                    regions: prior.regions,
                };
                let before = request_snapshot(&short);
                assert!(matches!(short.record_frozen(&frozen), Err(Error::Bounds)));
                assert_eq!(request_snapshot(&short), before);
            }
        });
    }
}

#[test]
fn frozen_rmsnorm_and_argmax_keep_actual_native_arcs_and_resource_charges() {
    use super::super::ArgmaxCode;
    use pllm_models::ModelOperator::{GreedyTokenSelection, RmsNorm};

    for operator in [RmsNorm, GreedyTokenSelection] {
        with_operation(129, operator, |fixture, full| {
            let (frozen, native, work, descriptors, kernel_metadata) = {
                let semantic = full.clone();
                let compact = replay_single_operation(fixture, &semantic);
                let native: Vec<_> = compact
                    .owned_native_programs()
                    .map(|program| {
                        (
                            Arc::as_ptr(program),
                            Arc::downgrade(program),
                            Arc::strong_count(program),
                            program.resources().unwrap().immutable_template_bytes,
                        )
                    })
                    .collect();
                assert!(!native.is_empty());
                let kernel_metadata: usize = compact
                    .argmax_census()
                    .map(|census| census.unwrap().kernel_metadata_bytes)
                    .sum();
                assert_eq!(kernel_metadata > 0, operator == GreedyTokenSelection);
                let expected_native: usize = native
                    .iter()
                    .map(|(_, _, _, bytes)| bytes + 2 * size_of::<usize>())
                    .sum::<usize>()
                    + kernel_metadata;
                let work = compact.work_budget();
                assert!(work.operations() > 0 && work.cursors().aes_work > 0);
                let descriptors = compact.stored_descriptors();
                let exports = vec![compact.ops[0].output.clone()];
                let quote = compact.freeze_quote(&exports).unwrap();
                assert_eq!(quote.native_owner_bytes, expected_native);
                quote.admit(quote.peak_bytes).unwrap();
                assert_eq!(quote.admit(quote.peak_bytes - 1), Err(Error::Bounds));
                let frozen = compact.freeze(exports, quote.peak_bytes).unwrap();
                assert_eq!(frozen.resources(), quote);
                (frozen, native, work, descriptors, kernel_metadata)
            };
            assert_eq!(frozen.work_budget(), work);
            if operator == RmsNorm {
                let program = frozen.native_row_program(0).unwrap();
                assert_eq!(native.len(), 1);
                assert_eq!(Arc::as_ptr(program), native[0].0);
                assert_eq!(program.work_budget(), work); // exactly one original row
                assert_eq!(kernel_metadata, 0);
            } else {
                let mut tensor_events = 0;
                frozen
                    .visit_argmax(0, &mut |event| {
                        if let ArgmaxCode::Tensor { program, .. } = event.code {
                            tensor_events += 1;
                            let owner = native
                                .iter()
                                .find(|(pointer, _, _, _)| *pointer == Arc::as_ptr(program))
                                .expect("argmax must retain its original native Arc");
                            assert_eq!(
                                program.resources().unwrap().immutable_template_bytes,
                                owner.3
                            );
                        }
                        Ok(())
                    })
                    .unwrap();
                assert!(tensor_events > 0);
            }

            let mut prior = RequestWork::default();
            let ordinary = WorkBudget::local();
            prior.record_work(ordinary).unwrap();
            for repetitions in 1..=2 {
                prior.record_frozen(&frozen).unwrap();
                assert_eq!(
                    prior.work(),
                    ordinary
                        .checked_add(work.checked_mul(repetitions).unwrap())
                        .unwrap()
                );
                assert_eq!(prior.stored_descriptors, descriptors * repetitions as usize);
                assert_eq!(
                    prior.retained_graph_bytes,
                    frozen.resources().retained_bytes * repetitions as usize
                );
                assert_eq!(prior.regions, repetitions as usize);
                for (pointer, weak, strong, _) in &native {
                    assert_eq!(weak.as_ptr(), *pointer);
                    assert_eq!(weak.strong_count(), *strong);
                }
            }
            let retained = frozen.resources().retained_bytes;
            let mut exact = RequestWork {
                work: prior.work(),
                stored_descriptors: MAX_NODES - descriptors,
                retained_graph_bytes: MAX_PROGRAM_BYTES - retained,
                regions: prior.regions,
            };
            exact.record_frozen(&frozen).unwrap();
            assert_eq!(exact.work(), prior.work().checked_add(work).unwrap());
            assert_eq!(exact.stored_descriptors, MAX_NODES);
            assert_eq!(exact.retained_graph_bytes, MAX_PROGRAM_BYTES);
            for (stored_descriptors, retained_graph_bytes) in [
                (MAX_NODES - descriptors + 1, prior.retained_graph_bytes),
                (prior.stored_descriptors, MAX_PROGRAM_BYTES - retained + 1),
            ] {
                let mut short = RequestWork {
                    work: prior.work(),
                    stored_descriptors,
                    retained_graph_bytes,
                    regions: prior.regions,
                };
                let before = request_snapshot(&short);
                assert!(matches!(short.record_frozen(&frozen), Err(Error::Bounds)));
                assert_eq!(request_snapshot(&short), before);
                for (_, weak, strong, _) in &native {
                    assert_eq!(weak.strong_count(), *strong);
                }
            }
            drop(frozen);
            assert!(native
                .iter()
                .all(|(_, weak, _, _)| weak.upgrade().is_none()));
        });
    }
}

#[test]
fn frozen_request_aes_and_ordinal_caps_and_accumulator_overflows_reject_atomically() {
    use crate::replicated_decoder::puma::{prss::MAX_PROGRAM_AES_BLOCKS, MAX_PROGRAM_OPERATIONS};

    with_operation(129, pllm_models::ModelOperator::Silu, |fixture, full| {
        let frozen = {
            let semantic = full.clone();
            let compact = replay_single_operation(fixture, &semantic);
            let exports = vec![compact.ops[0].output.clone()];
            let quote = compact.freeze_quote(&exports).unwrap();
            compact.freeze(exports, quote.peak_bytes).unwrap()
        };
        let work = frozen.work_budget();
        assert!(work.operations() > 0 && work.cursors().aes_work > 0);
        let aes_before = MAX_PROGRAM_AES_BLOCKS - work.cursors().aes_work;
        let ordinal_before = MAX_PROGRAM_OPERATIONS - work.operations();
        for (prefix, aes_cap) in [
            (schedule_with_exact_aes(aes_before), true),
            (
                WorkBudget::local().checked_mul(ordinal_before).unwrap(),
                false,
            ),
        ] {
            prefix.limits().unwrap();
            let mut prior = RequestWork::default();
            prior.record_work(prefix).unwrap();
            prior.record_frozen(&frozen).unwrap();
            assert_eq!(prior.work(), prefix.checked_add(work).unwrap());
            if aes_cap {
                assert_eq!(prior.work().cursors().aes_work, MAX_PROGRAM_AES_BLOCKS);
                assert!(prior.work().operations() < MAX_PROGRAM_OPERATIONS);
            } else {
                assert_eq!(prior.work().operations(), MAX_PROGRAM_OPERATIONS);
                assert!(prior.work().cursors().aes_work < MAX_PROGRAM_AES_BLOCKS);
            }
            let before = request_snapshot(&prior);
            assert!(matches!(prior.record_frozen(&frozen), Err(Error::Bounds)));
            assert_eq!(request_snapshot(&prior), before);
        }

        // One-short complete-request allowances: each prefix independently fits,
        // but cannot admit this same actual frozen region. Other cap still fits.
        for prefix in [
            schedule_with_exact_aes(aes_before + 1),
            WorkBudget::local().checked_mul(ordinal_before + 1).unwrap(),
        ] {
            prefix.limits().unwrap();
            let mut prior = RequestWork::default();
            prior.record_frozen(&frozen).unwrap();
            let before = request_snapshot(&prior);
            assert!(matches!(prior.record_work(prefix), Err(Error::Bounds)));
            assert_eq!(request_snapshot(&prior), before);
            let mut short = RequestWork {
                work: prefix,
                stored_descriptors: prior.stored_descriptors,
                retained_graph_bytes: prior.retained_graph_bytes,
                regions: prior.regions,
            };
            let before = request_snapshot(&short);
            assert!(matches!(short.record_frozen(&frozen), Err(Error::Bounds)));
            assert_eq!(request_snapshot(&short), before);
        }

        // Exercise checked_add overflow, not just limits(): construct counters
        // through real schedule arithmetic, leaving retained-owner counters live.
        let aes_overflow = schedule_with_exact_aes(u64::MAX - work.cursors().aes_work + 1);
        assert!(aes_overflow
            .operations()
            .checked_add(work.operations())
            .is_some());
        assert!(aes_overflow
            .cursors()
            .aes_work
            .checked_add(work.cursors().aes_work)
            .is_none());
        for prefix in [
            aes_overflow,
            WorkBudget::local().checked_mul(u64::MAX).unwrap(),
        ] {
            assert!(matches!(prefix.checked_add(work), Err(Error::Bounds)));
            let mut prior = RequestWork::default();
            prior.record_frozen(&frozen).unwrap();
            prior.work = prefix;
            let before = request_snapshot(&prior);
            assert!(matches!(prior.record_frozen(&frozen), Err(Error::Bounds)));
            assert_eq!(request_snapshot(&prior), before);
            assert!(matches!(prior.record_work(work), Err(Error::Bounds)));
            assert_eq!(request_snapshot(&prior), before);
        }
    });
}

#[test]
fn semantic_backing_survives_early_preflight_into_body_and_retained_request_quotes() {
    with_compact_parts(129, |compact, semantic| {
        let values = Vec::new();
        let quote = RequestQuote::new(compact, owners(&values)).unwrap();
        let semantic_bytes = semantic_bytes(semantic).unwrap();
        assert_eq!(quote.owner_census().borrowed_semantic_bytes, semantic_bytes);
        let without_semantic = quote.baseline_bytes() - semantic_bytes;
        assert!(matches!(
            RequestQuote::with_limit(compact, owners(&values), without_semantic),
            Err(Error::Bounds)
        ));
        let (builder, output) = body(129);
        let plan = builder.finish(&[output]).unwrap();
        let plain = quote.body_phases(&plan, &[]).unwrap();
        let mut prior = RequestWork::default();
        prior.record_graph(compact).unwrap();
        let carried = RequestQuote::with_request_work(compact, owners(&values), &prior).unwrap();
        let held = carried.body_phases(&plan, &[]).unwrap();
        let delta = prior.retained_graph_bytes + size_of::<RequestWork>();
        assert_eq!(held.execution_bytes - plain.execution_bytes, delta);
        assert_eq!(held.restoration_bytes - plain.restoration_bytes, delta);
    });
}

#[test]
fn actual_retained_body_plan_adds_owners_without_recounting_invocation_work() {
    with_compact(129, |compact| {
        let values = Vec::new();
        let mut prior = RequestWork::default();
        prior.record_graph(compact).unwrap();
        let before_work = prior.work();
        let before_bytes = prior.retained_graph_bytes;
        let before_descriptors = prior.stored_descriptors;
        let (builder, output) = body(129);
        let plan = builder.finish(&[output]).unwrap();
        let expected = plan.resources().immutable_metadata_bytes
            + plan.resources().owner_and_arc_allowance_bytes
            + 2 * size_of::<usize>();
        prior.record_retained_plan(&plan).unwrap();
        assert_eq!(Arc::strong_count(&plan), 1); // quote does not take ownership
        assert_eq!(prior.work(), before_work);
        assert_eq!(prior.retained_graph_bytes - before_bytes, expected);
        assert_eq!(
            prior.stored_descriptors - before_descriptors,
            plan.statistics().semantic_sites
        );
        let quote = RequestQuote::with_request_work(compact, owners(&values), &prior).unwrap();
        assert!(quote.baseline_bytes() >= prior.retained_graph_bytes);

        // Ceiling/overflow failures leave all prior accounting intact.
        prior.retained_graph_bytes = MAX_PROGRAM_BYTES;
        let before = (
            prior.retained_graph_bytes,
            prior.stored_descriptors,
            prior.work(),
        );
        assert!(matches!(
            prior.record_retained_plan(&plan),
            Err(Error::Bounds)
        ));
        assert_eq!(
            (
                prior.retained_graph_bytes,
                prior.stored_descriptors,
                prior.work()
            ),
            before
        );
        prior.retained_graph_bytes = usize::MAX;
        assert!(matches!(
            prior.record_retained_plan(&plan),
            Err(Error::Bounds)
        ));
        assert_eq!(prior.retained_graph_bytes, usize::MAX);
        assert_eq!(prior.stored_descriptors, before.1);
    });
}

#[test]
fn retained_plan_descriptor_exact_fit_one_over_and_overflows_preserve_request() {
    let (builder, output) = body(129);
    let plan = builder.finish(&[output]).unwrap();
    let sites = plan.semantic_sites();
    assert_eq!(sites, 1);
    assert_eq!(sites, plan.statistics().semantic_sites);
    let retained_bytes = plan.resources().immutable_metadata_bytes
        + plan.resources().owner_and_arc_allowance_bytes
        + 2 * size_of::<usize>();
    let work = WorkBudget::local();
    let mut fitting = RequestWork {
        work,
        stored_descriptors: MAX_NODES - sites,
        retained_graph_bytes: MAX_PROGRAM_BYTES - retained_bytes,
        regions: 3,
    };
    fitting.record_retained_plan(&plan).unwrap();
    assert_eq!(fitting.stored_descriptors, MAX_NODES);
    assert_eq!(fitting.retained_graph_bytes, MAX_PROGRAM_BYTES);
    assert_eq!(fitting.work(), work);
    assert_eq!(fitting.regions, 3);

    // Descriptor ceiling, descriptor addition overflow, byte ceiling and byte
    // addition overflow each reject independently, without partial mutation.
    for (descriptors, bytes) in [
        (MAX_NODES - sites + 1, 211),
        (usize::MAX, 211),
        (7, MAX_PROGRAM_BYTES - retained_bytes + 1),
        (7, usize::MAX),
    ] {
        let mut prior = RequestWork {
            work,
            stored_descriptors: descriptors,
            retained_graph_bytes: bytes,
            regions: 3,
        };
        assert!(matches!(
            prior.record_retained_plan(&plan),
            Err(Error::Bounds),
        ));
        assert_eq!(prior.stored_descriptors, descriptors);
        assert_eq!(prior.retained_graph_bytes, bytes);
        assert_eq!(prior.work(), work);
        assert_eq!(prior.regions, 3);
        assert_eq!(Arc::strong_count(&plan), 1);
    }
}

#[test]
fn final_norm_row_owners_are_censused_without_inventing_body_proof_mapping() {
    with_operation(129, pllm_models::ModelOperator::RmsNorm, |compact, _| {
        let values = Vec::new();
        let quote = RequestQuote::new(compact, owners(&values)).unwrap();
        let census = quote.owner_census();
        assert_eq!(census.output_slots, 1);
        assert_eq!(census.retained_output_bytes, 2 * 16);
        assert!(census.prepared_program_bytes > 0);
        assert_eq!(census.body_descriptors, None);
        assert!(census.remaining_operations > 0);
        let layout = Layout::quote_contiguous(0, 2).unwrap();
        assert!(matches!(
            quote.preflight_body_inputs(
                &[Register {
                    lanes: 2,
                    bounds: domain()
                }],
                &[layout]
            ),
            Err(Error::Binding)
        ));
    });
}

#[test]
fn last_token_alias_does_not_manufacture_new_physical_value_owners() {
    with_operation(129, pllm_models::ModelOperator::LastToken, |compact, _| {
        let values = Vec::new();
        let quote = RequestQuote::new(compact, owners(&values)).unwrap();
        let census = quote.owner_census();
        assert_eq!(census.output_slots, 0);
        assert_eq!(census.output_slot_bytes, 0);
        assert_eq!(census.retained_output_bytes, 0);
        assert_eq!(census.body_descriptors, Some(0));
        assert!(census.remaining_operations > 0);
    });
}

#[test]
fn reshape_inverse_aliases_keep_exact_import_order_without_phantom_value_owners() {
    use super::super::super::{validate_source_records, Builder, Code, Op};

    with_compact_parts(129, |fixture, full| {
        let sources = fixture.borrowed_source_records();
        let norms = fixture.borrowed_source_norms();
        let shapes = sources
            .iter()
            .map(|(key, source)| (key.clone(), source.shape.clone()))
            .collect();
        let validated = validate_source_records(
            full,
            &shapes,
            sources,
            WeightMode::Public,
            ExecutionPolicy::default(),
            norms,
        )
        .unwrap();
        let mut sequence = full
            .operations
            .iter()
            .find(|op| {
                op.operator == Op::Reshape && op.attributes["layout"] == "batch_sequence_hidden"
            })
            .unwrap()
            .clone();
        let mut heads = full
            .operations
            .iter()
            .find(|op| {
                op.operator == Op::Reshape
                    && op.attributes["layout"] == "batch_heads_sequence_feature"
            })
            .unwrap()
            .clone();
        let entry = "reshape.input";
        sequence.inputs = vec![entry.into()];
        sequence.output_shape = vec![1, 3, 4];
        heads.inputs = vec![sequence.id.clone()];
        heads.output_shape = vec![1, 2, 3, 2];
        let mut region = full.clone();
        region.output = heads.id.clone();
        region.operations = vec![sequence, heads];

        // Import-only original owners; no arithmetic-prefix, native program or
        // executable handoff/proof capability is manufactured by this fixture.
        let mut imports = Builder {
            public_weights: true,
            ..Builder::default()
        };
        for (key, source) in sources {
            if source.public_q18.is_none() {
                imports
                    .push(
                        Code::PublicInput {
                            key: key.clone(),
                            digest: source.digest,
                            row_l1: source.row_l1,
                            row_squared_norm: source.row_squared_norm,
                            row_l1_squared_sum: source.row_l1_squared_sum,
                        },
                        source.shape.iter().product(),
                        Interval::new(source.lower, source.upper).unwrap(),
                    )
                    .unwrap();
            }
        }
        let owner = imports
            .push(Code::Input(entry.into()), 12, domain())
            .unwrap();
        let input = imports.view(owner, vec![1, 2, 3, 2]);
        let mut compact = CompactGraph::new(&imports, validated, &region, 0).unwrap();
        let bound = compact.bind_import(entry, &input).unwrap();
        let before = CompactOwnerCensus::read(&compact).unwrap();
        let containers = compact.container_census().unwrap();
        let mut canonical = Sha256::new();
        canonical.update(b"reshape-alias-census");
        compact.hash_expanded_nodes(&mut canonical).unwrap();
        let digest: [u8; 32] = canonical.finalize().into();
        let values = Vec::new();

        // Independent exact lane oracle, not the lowering's index function.
        // Heads [1,2,3,2] -> sequence [1,3,4] -> original heads ordering.
        for (step, expected) in [
            [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ]
        .into_iter()
        .enumerate()
        {
            let output = compact.lower_next().unwrap();
            assert!(output
                .words()
                .eq(expected.into_iter().map(|lane| (owner, lane))));
            assert!(input.words.iter().eq((0..12).map(|lane| (owner, lane))));
            assert_eq!(output.mapping, bound.mapping);
            assert_eq!(output.bounds, bound.bounds);
            assert!(Arc::ptr_eq(&output.origin, &bound.origin));
            assert_eq!(
                output.logical.dimensions(),
                region.operations[step]
                    .output_shape
                    .iter()
                    .map(|dimension| *dimension as usize)
                    .collect::<Vec<_>>()
            );
            let quote = RequestQuote::new(&compact, owners(&values)).unwrap();
            let after = quote.owner_census();
            // Before the fix each alias invented one slot and 12*16 bytes:
            // the inverse pair falsely added two slots and 384 payload bytes.
            assert_eq!(after.output_slots, 0);
            assert_eq!(after.output_slots, before.output_slots);
            assert_eq!(after.output_slot_bytes, 0);
            assert_eq!(after.output_slot_bytes, before.output_slot_bytes);
            assert_eq!(after.retained_output_bytes, 0);
            assert_eq!(after.retained_output_bytes, before.retained_output_bytes);
            assert_eq!(after.body_descriptors, Some(0));
            assert_eq!(after.work, WorkBudget::default());
            assert_eq!(after.work, before.work);
            assert_eq!(compact.next, containers.expanded_virtual_nodes);
            assert_eq!(compact.owned_native_programs().count(), 0);
            assert_eq!(
                after.stored_descriptors,
                before.stored_descriptors + step + 1
            );
            assert_eq!(after.metadata_bytes, before.metadata_bytes);
            assert_eq!(after.metadata_bytes, compact.baseline().unwrap());
            let current = compact.container_census().unwrap();
            assert_eq!(current.descriptor_capacity, containers.descriptor_capacity);
            assert_eq!(current.descriptor_bytes, containers.descriptor_bytes);
            assert_eq!(
                current.import_binding_bytes,
                containers.import_binding_bytes
            );
            let mut canonical = Sha256::new();
            canonical.update(b"reshape-alias-census");
            compact.hash_expanded_nodes(&mut canonical).unwrap();
            assert_eq!(<[u8; 32]>::from(canonical.finalize()), digest);

            // Zero payload is not zero descriptor/metadata cost or a new cap.
            let mut fitting = owners(&values);
            fitting.other_stored_descriptors = MAX_NODES - after.stored_descriptors;
            assert_eq!(
                RequestQuote::new(&compact, fitting)
                    .unwrap()
                    .stored_descriptors(),
                MAX_NODES
            );
            let mut excessive = owners(&values);
            excessive.other_stored_descriptors = MAX_NODES - after.stored_descriptors + 1;
            assert!(matches!(
                RequestQuote::new(&compact, excessive),
                Err(Error::Bounds)
            ));
        }
        assert!(compact.ops[1].output.words().eq(input.words.iter()));
        assert_eq!(compact.ops[1].output.order, super::super::LaneOrder::Linear);
        assert!(compact.descriptor_census().all(|descriptor| {
            matches!(descriptor.kind, DescriptorKind::Reshape)
                && descriptor.virtual_start == descriptor.virtual_end
        }));
    });
}

#[test]
fn feedback_alias_keeps_predecessor_owners_work_and_all_owner_peaks() {
    for lanes in [1, 128, 129, 257] {
        with_operation_prefix(
            lanes,
            pllm_models::ModelOperator::Multiply,
            true,
            |compact, semantic| {
                let mut values = Vec::with_capacity(5);
                values.push(Some(Secret::new(3).unwrap()));
                let predecessor = compact.ops[0].output.clone();
                let next = compact.next;
                let construction_peak = compact.construction_peak_bytes();
                let imports = (
                    compact.imports.nodes.len(),
                    compact.imports.allocated,
                    compact.imports.validity,
                );
                let layouts = [
                    Layout::quote_contiguous(0, lanes).unwrap(),
                    Layout::quote_contiguous(1, lanes).unwrap(),
                ];
                let (builder, output) = body(lanes);
                let plan = builder.finish(&[output]).unwrap();
                let (before, baseline, work, descriptors, input_peak, phases) = {
                    let quote = RequestQuote::new(compact, owners(&values)).unwrap();
                    (
                        quote.owner_census(),
                        quote.baseline_bytes(),
                        quote.complete_work(),
                        quote.stored_descriptors(),
                        quote
                            .preflight_body_inputs(&inputs(lanes), &layouts)
                            .unwrap(),
                        quote.body_phases(&plan, &[]).unwrap(),
                    )
                };
                let feedback = compact.lower_next().unwrap();
                assert!(feedback.words().eq(predecessor.words()));
                assert_eq!(feedback.mapping, predecessor.mapping);
                assert_eq!(feedback.start, predecessor.start);
                assert_eq!(feedback.bounds, predecessor.bounds);
                assert_eq!(feedback.rank, predecessor.rank);
                assert_eq!(compact.next, next);
                assert_eq!(compact.construction_peak_bytes(), construction_peak);
                assert_eq!(
                    imports,
                    (
                        compact.imports.nodes.len(),
                        compact.imports.allocated,
                        compact.imports.validity
                    )
                );
                assert!(compact.frontier.value(&semantic.operations[0].id).is_none());
                assert!(compact.frontier.value(&semantic.output).is_some());
                let quote = RequestQuote::new(compact, owners(&values)).unwrap();
                let after = quote.owner_census();
                assert_eq!(after.output_slots, before.output_slots);
                assert_eq!(after.output_slot_bytes, before.output_slot_bytes);
                assert_eq!(after.retained_output_bytes, before.retained_output_bytes);
                assert_eq!(after.body_descriptors, Some(1));
                assert_eq!(after.body_descriptors, before.body_descriptors);
                assert_eq!(after.stored_descriptors, before.stored_descriptors + 1);
                assert_eq!(after.metadata_bytes, before.metadata_bytes); // capacity already admitted
                assert_eq!(after.metadata_bytes, compact.baseline().unwrap());
                assert_eq!(after.borrowed_source_bytes, before.borrowed_source_bytes);
                assert_eq!(
                    after.borrowed_semantic_bytes,
                    before.borrowed_semantic_bytes
                );
                assert_eq!(after.work, before.work);
                assert_eq!(after.remaining_operations, 0);
                assert_eq!(quote.complete_work(), work);
                assert_eq!(quote.stored_descriptors(), descriptors + 1);
                assert_eq!(quote.baseline_bytes(), baseline);
                let mut excessive = owners(&values);
                excessive.other_work = WorkBudget::local()
                    .checked_mul(crate::replicated_decoder::puma::MAX_PROGRAM_OPERATIONS)
                    .unwrap();
                assert!(matches!(
                    RequestQuote::new(compact, excessive),
                    Err(Error::Bounds)
                ));
                assert_eq!(
                    quote
                        .preflight_body_inputs(&inputs(lanes), &layouts)
                        .unwrap(),
                    input_peak
                );
                assert_eq!(quote.body_phases(&plan, &[]).unwrap(), phases);

                // The alias has no new value/body site, but its history record
                // still consumes the original stored-descriptor ceiling.
                let mut fitting = owners(&values);
                fitting.other_stored_descriptors = MAX_NODES - after.stored_descriptors - 1;
                assert_eq!(
                    RequestQuote::new(compact, fitting)
                        .unwrap()
                        .stored_descriptors(),
                    MAX_NODES
                );
                let mut over = owners(&values);
                over.other_stored_descriptors = MAX_NODES - after.stored_descriptors;
                assert!(matches!(
                    RequestQuote::new(compact, over),
                    Err(Error::Bounds)
                ));

                assert!(RequestQuote::with_limit(compact, owners(&values), baseline).is_ok());
                assert!(matches!(
                    RequestQuote::with_limit(compact, owners(&values), baseline - 1),
                    Err(Error::Bounds)
                ));
                let exact =
                    RequestQuote::with_limit(compact, owners(&values), input_peak.peak_bytes)
                        .unwrap();
                assert_eq!(
                    exact
                        .preflight_body_inputs(&inputs(lanes), &layouts)
                        .unwrap(),
                    input_peak
                );
                let short =
                    RequestQuote::with_limit(compact, owners(&values), input_peak.peak_bytes - 1)
                        .unwrap();
                assert!(matches!(
                    short.preflight_body_inputs(&inputs(lanes), &layouts),
                    Err(Error::Bounds)
                ));
                let exact =
                    RequestQuote::with_limit(compact, owners(&values), phases.peak_bytes).unwrap();
                assert_eq!(exact.body_phases(&plan, &[]).unwrap(), phases);
                let short =
                    RequestQuote::with_limit(compact, owners(&values), phases.peak_bytes - 1)
                        .unwrap();
                assert!(matches!(short.body_phases(&plan, &[]), Err(Error::Bounds)));

                let mut prior = RequestWork::default();
                prior.record_graph(compact).unwrap();
                assert_eq!(prior.work(), before.work);
                assert_eq!(prior.stored_descriptors, after.stored_descriptors);
                assert_eq!(
                    prior.retained_graph_bytes,
                    after.metadata_bytes
                        + after.borrowed_source_bytes
                        + after.borrowed_semantic_bytes
                );
                prior.record_retained_plan(&plan).unwrap();
                assert_eq!(prior.work(), before.work);
                let carried =
                    RequestQuote::with_request_work(compact, owners(&values), &prior).unwrap();
                assert_eq!(
                    carried.complete_work(),
                    work.checked_add(before.work).unwrap()
                );
                assert_eq!(
                    carried.baseline_bytes() - baseline,
                    prior.retained_graph_bytes + size_of::<RequestWork>()
                );
            },
        );
    }
}

#[test]
fn feedback_after_last_token_remains_zero_body_and_zero_new_value_slots() {
    with_operation_prefix(
        129,
        pllm_models::ModelOperator::LastToken,
        true,
        |compact, _| {
            let before = CompactOwnerCensus::read(compact).unwrap();
            let output = compact.lower_next().unwrap();
            let values = Vec::new();
            let quote = RequestQuote::new(compact, owners(&values)).unwrap();
            let after = quote.owner_census();
            assert_eq!(after.body_descriptors, Some(0));
            assert_eq!(after.output_slots, 0);
            assert_eq!(after.output_slot_bytes, 0);
            assert_eq!(after.retained_output_bytes, 0);
            assert_eq!(after.stored_descriptors, before.stored_descriptors + 1);
            assert_eq!(after.metadata_bytes, before.metadata_bytes);
            assert_eq!(after.work, WorkBudget::default());
            assert_eq!(quote.complete_work(), WorkBudget::local());
            assert!(compact.complete_row(&output, 0).unwrap().is_none());
            let specifications = [Register {
                lanes: 2,
                bounds: domain(),
            }];
            let mut builder = layer::Builder::new(WeightMode::Public, &specifications).unwrap();
            let input = builder.input(0).unwrap();
            assert!(quote
                .preflight_body_inputs(&specifications, &[Layout::quote_contiguous(0, 2).unwrap()],)
                .is_ok());
            // Existing Plan rejects empty bodies. Census is not execution
            // authority: enclosing emitter must retain this alias directly,
            // never add arithmetic merely to manufacture an executable Plan.
            assert!(matches!(builder.finish(&[input]), Err(Error::Shape)));
        },
    );
}

#[test]
fn feedback_alias_does_not_admit_malformed_body_imports_with_matching_work() {
    with_operation_prefix(
        129,
        pllm_models::ModelOperator::Multiply,
        true,
        |compact, _| {
            compact.lower_next().unwrap();
            let values = Vec::new();
            let quote = RequestQuote::new(compact, owners(&values)).unwrap();
            let layouts = [
                Layout::quote_contiguous(0, 129).unwrap(),
                Layout::quote_contiguous(1, 129).unwrap(),
            ];
            let mut foreign = inputs(129);
            foreign[0].bounds = Interval::new(-2, 2).unwrap();
            assert!(matches!(
                quote.preflight_body_inputs(&foreign, &layouts),
                Err(Error::Binding)
            ));
            assert!(matches!(
                quote.preflight_body_inputs(&inputs(128), &layouts),
                Err(Error::Binding)
            ));
            assert!(matches!(
                quote.preflight_body_inputs(&inputs(129)[..1], &layouts),
                Err(Error::Binding)
            ));
            let extra = [inputs(129)[0]; 3];
            assert!(matches!(
                quote.preflight_body_inputs(&extra, &layouts),
                Err(Error::Binding)
            ));
            let mut builder = layer::Builder::new(WeightMode::Public, &foreign).unwrap();
            let left = builder.input(0).unwrap();
            let right = builder.input(1).unwrap();
            let output = builder.binary(&left, &right, true).unwrap();
            let plan = builder.finish(&[output]).unwrap();
            assert_eq!(plan.work_budget(), compact.work_budget());
            assert!(matches!(quote.body_phases(&plan, &[]), Err(Error::Binding)));
            // Shape/work checks are necessary only; equal-shaped foreign Plans do
            // not acquire original execution/proof/import authority from this quote.
        },
    );
}

#[test]
fn feedback_foreign_or_forged_frontier_import_retires_quote_without_request_refund() {
    for forged_offset in [false, true] {
        with_operation_prefix(
            129,
            pllm_models::ModelOperator::Multiply,
            true,
            |compact, semantic| {
                let values = Vec::new();
                let mut prior = RequestWork::default();
                prior.record_work(WorkBudget::local()).unwrap();
                let before = (
                    prior.work(),
                    prior.stored_descriptors,
                    prior.retained_graph_bytes,
                    prior.regions,
                );
                let next = compact.next;
                let stored = compact.stored_descriptors();
                with_compact(129, |foreign| {
                    let mut input = if forged_offset {
                        compact.ops[0].output.clone()
                    } else {
                        foreign.ops[0].output.clone()
                    };
                    if forged_offset {
                        input.start += 1;
                    }
                    compact
                        .frontier
                        .bind(semantic.operations[1].inputs[0].clone(), input);
                    assert!(matches!(compact.lower_next(), Err(Error::Binding)));
                });
                assert_eq!(compact.next, next);
                assert_eq!(compact.stored_descriptors(), stored);
                assert!(matches!(
                    RequestQuote::new(compact, owners(&values)),
                    Err(Error::Retired)
                ));
                assert!(matches!(prior.record_graph(compact), Err(Error::Retired)));
                assert_eq!(
                    (
                        prior.work(),
                        prior.stored_descriptors,
                        prior.retained_graph_bytes,
                        prior.regions
                    ),
                    before
                );
                assert!(matches!(compact.lower_next(), Err(Error::Retired)));
            },
        );
    }
}
