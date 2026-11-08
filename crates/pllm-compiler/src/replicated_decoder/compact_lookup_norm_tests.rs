//! Genuine publisher -> manifest -> consumed source request only. No digest/B
//! fixture is promoted to authority; all tests are authoring-only until run.
use super::*;
use request_origin::compact_storage::{EmissionContext, EmissionScope};

fn with_embedding<T>(
    value: i64,
    run: impl for<'a> FnOnce(&'a Builder, &'a EmissionContext<'a>) -> Result<T, Error>,
) -> Result<T, Error> {
    with_embedding_row(&[value; 2], 1, false, run)
}

// Omission removes an optional premise through the real DTO/import path; it
// never installs a replacement statistic. Other fixtures keep the original cap.
fn with_embedding_row<T>(
    row: &[i64],
    activation_cap: u32,
    omit_statistic: bool,
    run: impl for<'a> FnOnce(&'a Builder, &'a EmissionContext<'a>) -> Result<T, Error>,
) -> Result<T, Error> {
    with_embedding_geometry(row, activation_cap, omit_statistic, None, run)
}

fn with_embedding_geometry<T>(
    row: &[i64],
    activation_cap: u32,
    omit_statistic: bool,
    hidden_width: Option<usize>,
    run: impl for<'a> FnOnce(&'a Builder, &'a EmissionContext<'a>) -> Result<T, Error>,
) -> Result<T, Error> {
    assert!(!row.is_empty());
    let (mut config, _, clear) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    if let Some(width) = hidden_width {
        let mut geometry: serde_json::Value = serde_json::from_slice(&config).unwrap();
        let heads = usize::try_from(geometry["num_attention_heads"].as_u64().unwrap()).unwrap();
        assert_eq!(width % heads, 0);
        geometry["hidden_size"] = serde_json::json!(width);
        geometry["head_dim"] = serde_json::json!(width / heads);
        config = serde_json::to_vec(&geometry).unwrap();
    }
    let mut publisher = Publisher::new_with_policy(
        &config, 4, 2, [1; 32], [2; 32], WeightMode::Public,
        ExecutionPolicy {
            range_policy: RangePolicy::Checked, max_abs_activation: activation_cap,
            ..ExecutionPolicy::default()
        },
    ).unwrap();
    let shapes = publisher.shapes().iter()
        .map(|(key, shape)| (key.clone(), shape.clone())).collect::<Vec<_>>();
    let mut payloads = BTreeMap::new();
    for (key, shape) in shapes {
        let values = &clear[&key];
        let bytes = (0..shape.iter().product()).flat_map(|i| {
            let encoded = if key == "model.embed_tokens.weight" {
                row[i % row.len()]
            } else if hidden_width.is_some() {
                // Shapes come from this Qwen2 publisher, not the width-2 data.
                // The widened fixture has unit gamma/norms and zero other weights.
                if key.contains("norm") { ONE } else { 0 }
            } else {
                values[i % values.len()]
            };
            (encoded as f32 / ONE as f32).to_le_bytes()
        }).collect::<Vec<_>>();
        publisher.append(&key, 0, &bytes).unwrap();
        payloads.insert(key, bytes);
    }
    publisher.seal(&[1, 2, 1, 2]).unwrap();
    let mut manifest: Manifest = serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
    manifest.program = [0; 32];
    if omit_statistic {
        manifest.tensors.get_mut("model.embed_tokens.weight").unwrap().row_squared_norm = None;
    }
    let mut importer = source_entry::SourceRequestImport::new(&serde_json::to_vec(&manifest).unwrap())?;
    for (key, bytes) in payloads { importer.append(&key, 0, &bytes)?; }
    importer.with_created(|created, guard| {
        let mut scope = EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
        let (builder, context, _) = scope.parts();
        run(builder, context)
    })
}

fn admit_frame(graph: &CompactGraph<'_, '_>) -> Result<(), Error> {
    let controls = control_slots()?;
    graph.admit_local(add_bytes(graph.baseline()?, controls)?)?;
    graph.request_admission.ok_or(Error::Binding)?.admit_phase(controls, 0)
}

fn original_observation_limits() -> crate::original_source::Limits {
    crate::original_source::Limits {
        expansion: DecoderExpansionBudget {
            max_layers: 8, max_feature_width: 128, max_attention_heads: 8,
            max_vocabulary: 128, max_tensor_elements: 32768,
        },
        factory: pllm_models::ConfigLowererLimits {
            max_operations_per_phase: 256, max_string_bytes: 1024,
            max_attribute_nodes: 32768, max_retained_strings: 1 << 20,
            max_factory_bytes: pllm_garble::shared_allocation::MAX_ALLOCATION_BYTES,
        },
        max_work: 1_000_000_000_000,
        max_working_payload: pllm_garble::shared_allocation::MAX_ALLOCATION_BYTES,
    }
}

// Read-only observation of ORIGINAL private Ledger.used, not a new ledger or
// historical Resources.prepaid_work. At cap<=used, lower_bootstrap_bytes returns
// Work; at cap=used+1 it reaches the subsequent bootstrap query (possibly Quote).
// The source's original 10^12 cap is never raised or changed. No lowering occurs.
fn original_spent(plan: &crate::original_source::OriginalSourcePlan) -> u64 {
    let mut limits = original_observation_limits();
    let (mut lo, mut hi) = (0, limits.max_work);
    assert!(matches!(plan.lower_bootstrap_bytes(limits), Ok(_) | Err(crate::original_source::Error::Quote)));
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        limits.max_work = mid;
        match plan.lower_bootstrap_bytes(limits) {
            Err(crate::original_source::Error::Work) => lo = mid,
            Ok(_) | Err(crate::original_source::Error::Quote) => hi = mid,
            other => panic!("unexpected original-ledger observation: {other:?}"),
        }
    }
    limits.max_work = lo;
    assert!(matches!(plan.lower_bootstrap_bytes(limits), Err(crate::original_source::Error::Work)));
    lo
}

// Independent fixture census, not prepay_lookup_proof or its quote. One actual
// token Input, one bound token view and one emitted lookup accompany the file
// declarations. Inline gamma records do not create PublicInput declarations.
fn lookup_visits(graph: &CompactGraph<'_, '_>) -> u64 {
    let manifest = graph.request_admission.unwrap().manifest();
    let records = manifest.tensors.len();
    let files = manifest.tensors.values().filter(|record| record.public_q18.is_none()).count();
    assert_eq!(graph.sources.len(), records);
    assert_eq!(graph.imports.nodes.len(), files + 1);
    assert_eq!(graph.imports.original_inputs.len(), 1);
    assert_eq!(graph.bound_imports.len(), 1);
    assert_eq!(graph.ops.len(), 1);
    (records + files + "model.embed_tokens.weight".len() + 5) as u64
}

#[test]
fn lookup_isolated_import_producer_row_offset_and_width_controls() {
    with_embedding(ONE / 64, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let view = graph.lower_next()?;
        admit_frame(&graph)?;
        assert!(LookupRow::original(&graph, &view, 1)?.is_some());
        // Same declarations, same original InputRecord Arcs, different physical
        // import allocation: only import-owner identity changes.
        let copied = Builder {
            nodes: builder.nodes.clone(), original_inputs: builder.original_inputs.clone(),
            ..Builder::default()
        };
        assert!(matches!(context.check_imports(&copied), Err(Error::Binding)));

        let mut wrong_slot = view.clone();
        wrong_slot.slot = Some(graph.ops.len());
        assert!(matches!(LookupRow::original(&graph, &wrong_slot, 1), Err(Error::Binding)));
        let mut wrong_offset = view.clone();
        wrong_offset.start += view.logical.row_width();
        assert!(matches!(LookupRow::original(&graph, &wrong_offset, 1), Err(Error::Binding)));
        let mut wrong_width = view.clone();
        wrong_width.logical = ViewShape::new(&[1, 2, 4])?;
        assert!(LookupRow::original(&graph, &wrong_width, 1).is_err());

        // A retained token Input is not a public-file representation. Keep the
        // genuine source descriptor, plan, row and request unchanged.
        let slot = view.slot.unwrap();
        let original_weight = match &graph.ops[slot].kind {
            PublishedKind::TokenLookup { weight, .. } => *weight,
            _ => unreachable!(),
        };
        let token_owner = match graph.ops[slot].input.mapping {
            Mapping::Dense(owner) => owner,
            _ => unreachable!(),
        };
        if let PublishedKind::TokenLookup { weight, .. } = &mut graph.ops[slot].kind { *weight = token_owner; }
        assert!(matches!(LookupRow::original(&graph, &view, 1), Err(Error::Binding)));
        if let PublishedKind::TokenLookup { weight, .. } = &mut graph.ops[slot].kind { *weight = original_weight; }

        graph.lower_next()?;
        admit_frame(&graph)?;
        // Corrupt one private consumed-proof field at a time, AFTER real mint.
        // Negative fixtures only; no fabricated source statistic is accepted.
        let mut proof = LookupRow::original(&graph, &view, 1)?.unwrap();
        proof.producer = &graph.ops[1];
        assert!(matches!(proof.energy(&graph, &view, 1), Err(Error::Binding)));
        let mut proof = LookupRow::original(&graph, &view, 1)?.unwrap();
        proof.start += 1;
        assert!(matches!(proof.energy(&graph, &view, 1), Err(Error::Binding)));
        let mut proof = LookupRow::original(&graph, &view, 1)?.unwrap();
        proof.width += 1;
        assert!(matches!(proof.energy(&graph, &view, 1), Err(Error::Binding)));
        Ok(())
    }).unwrap();
}

#[test]
fn copied_import_allocation_refuses_actual_constructor_without_lookup_debit() {
    with_embedding(0, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let view = graph.lower_next()?;
        admit_frame(&graph)?;
        assert!(LookupRow::original(&graph, &view, 0)?.is_some());
        let copied = Builder {
            nodes: builder.nodes.clone(), original_inputs: builder.original_inputs.clone(),
            ..Builder::default()
        };
        let spent = original_spent(context.plan());
        let attempts = test_lookup_attempts();
        assert!(matches!(CompactGraph::from_source_request(&copied, context, MAX_PROGRAM_BYTES), Err(Error::Binding)));
        assert_eq!(original_spent(context.plan()), spent);
        assert_eq!(test_lookup_attempts(), attempts);
        // Real constructor's failed attempt retires this request. Do not mint
        // another context or retry a positive assertion after that refusal.
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    }).unwrap();
}

#[test]
fn lookup_missing_statistic_gains_no_proof_through_actual_import() {
    with_embedding_row(&[0; 2], 1, true, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let view = graph.lower_next()?;
        admit_frame(&graph)?;
        assert!(context.manifest().tensors["model.embed_tokens.weight"].row_squared_norm.is_none());
        assert!(LookupRow::original(&graph, &view, 0)?.is_none());
        assert_eq!(guarded_energy(&graph, &view, 0, Some(ONE))?, None);
        Ok(())
    }).unwrap();
}

#[test]
fn lookup_success_and_spent_binding_failure_debit_exact_original_census() {
    with_embedding(0, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let view = graph.lower_next()?;
        admit_frame(&graph)?;
        let visits = lookup_visits(&graph);
        let before = original_spent(context.plan());
        assert_eq!(LookupRow::original(&graph, &view, 0)?.unwrap().energy(&graph, &view, 0)?, 0);
        assert_eq!(original_spent(context.plan()), before + visits);
        let mut foreign = view.clone();
        foreign.origin = Arc::new(());
        for attempt in 1..=2 {
            assert!(matches!(LookupRow::original(&graph, &foreign, 0), Err(Error::Binding)));
            assert_eq!(original_spent(context.plan()), before + (attempt + 1) * visits);
        }
        Ok(())
    }).unwrap();
}

#[test]
fn lookup_exhausted_original_ledger_wins_before_new_validation_and_retry() {
    with_embedding(0, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let mut view = graph.lower_next()?;
        admit_frame(&graph)?;
        let visits = lookup_visits(&graph);
        let before = original_spent(context.plan());
        let target = 1_000_000_000_000 - visits + 1;
        context.prepay_metadata(target - before).unwrap();
        assert_eq!(original_spent(context.plan()), target);
        let bytes = context.plan().budget().used_bytes();
        let attempts = test_lookup_attempts();
        // If new validation/walk runs before prepayment this sentinel produces
        // Binding, not Bounds. Failed cumulative debit must burn same owner.
        view.origin = Arc::new(());
        for _ in 0..2 {
            assert!(matches!(LookupRow::original(&graph, &view, 0), Err(Error::Bounds)));
            assert!(matches!(context.prepay_metadata(0), Err(Error::Bounds)));
            assert!(matches!(context.plan().lower_bootstrap_bytes(original_observation_limits()),
                Err(crate::original_source::Error::Burned)));
            assert_eq!(context.plan().budget().used_bytes(), bytes);
            assert_eq!(test_lookup_attempts(), attempts);
        }
        Ok(())
    }).unwrap();
}

#[test]
fn lookup_proof_control_slots_one_byte_short_precedes_work_and_payload_growth() {
    use std::mem::size_of;
    with_embedding(0, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let view = graph.lower_next()?;
        let gamma = &graph.sources["model.layers.0.input_layernorm.weight"];
        // Named concrete logical slots, independently counted. Not an allocator,
        // compiled call-frame or native-factory counter claim.
        let controls = 2 * size_of::<Result<PreparedTransition<'_>, Error>>()
            + size_of::<CheckedRows<'_>>() + size_of::<GuardRun>() + size_of::<SafeRow<'_, '_>>()
            + size_of::<LookupRow<'_, '_, '_>>()
            + 2 * size_of::<Result<Option<LookupRow<'_, '_, '_>>, Error>>()
            + size_of::<[usize; 7]>()
            + size_of::<std::collections::btree_map::Iter<'_, String, SourceTensor>>()
            + size_of::<std::slice::Iter<'_, BoundImport<'_>>>()
            + 2 * size_of::<Option<(&String, &SourceTensor)>>()
            + 4 * size_of::<tensor::Construction>() + 2 * size_of::<tensor::ConstructionQuote>()
            + size_of::<Result<tensor::ConstructionQuote, tensor::ConstructionQuoteError>>()
            + 4 * size_of::<Result<WorkBudget, Error>>() + 2 * size_of::<WorkBudget>();
        assert_eq!(control_slots()?, controls);
        let width = view.logical.row_width();
        let gamma_bytes = size_of::<SourceTensor>() + gamma.shape.capacity() * size_of::<usize>()
            + gamma.public_q18.as_ref().unwrap().capacity() * size_of::<i64>();
        let first_phase = graph.outside_owner_bytes + graph.baseline()? + gamma_bytes + controls
            + request_rms::slots(0)? + compact_norm::driver_slots()?
            + tensor::RmsWorkWorkspace::workspace_bytes()
            + work_frames(width, width, view.lanes)?;
        assert!(first_phase <= graph.limit);
        graph.limit = first_phase - 1;
        let spent = original_spent(context.plan());
        let bytes = context.plan().budget().used_bytes();
        let attempts = test_lookup_attempts();
        let ops = graph.ops.len();
        assert!(matches!(prepare(&graph, &view, width, gamma), Err(Error::Bounds)));
        assert_eq!(original_spent(context.plan()), spent);
        assert_eq!(context.plan().budget().used_bytes(), bytes);
        assert_eq!(test_lookup_attempts(), attempts);
        assert_eq!(graph.ops.len(), ops);
        Ok(())
    }).unwrap();
}

#[test]
fn authentic_weak_strict_lookup_bound_does_not_imply_coordinate_improvement() {
    let c = 1_i64 << 26;
    with_embedding_row(&[5 * c / 4, 0], 256, false, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let view = graph.lower_next()?;
        admit_frame(&graph)?;
        let bound = 25 * (1_u128 << 52) / 16;
        assert_eq!(view.logical.row_width(), 2);
        assert_eq!(LookupRow::original(&graph, &view, 0)?.unwrap().energy(&graph, &view, 0)?, bound);
        assert_eq!(guarded_energy(&graph, &view, 0, Some(c))?, Some(bound));
        assert!((1_u128 << 52) <= bound && bound < 2 * (1_u128 << 52));
        // Actual retained-source extraction, not a manufactured cap-1 premise.
        // A weak row bound cannot promise that every coordinate is <C.
        Ok(())
    }).unwrap();
}

#[test]
fn authentic_guarded_lookup_cap_equality_keeps_ordinary_lowering() {
    let c = 1_i64 << 26;
    assert_eq!(ExecutionPolicy::default().max_abs_activation, 256);
    // Real F32 rows [C,C], [5C/4,0], [0,0] have max B=2C^2. The second row
    // widens the retained source interval beyond C, forcing the genuine checked
    // safe-zero guard at the existing 256-Q18 production profile cap.
    with_embedding_row(&[c, c, 5 * c / 4, 0, 0, 0], 256, false, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let view = graph.lower_next()?;
        admit_frame(&graph)?;
        assert_eq!(view.logical.row_width(), 2);
        assert_eq!(LookupRow::original(&graph, &view, 0)?.unwrap().energy(&graph, &view, 0)?,
            2_u128 * (1_u128 << 52));
        assert_eq!(guarded_energy(&graph, &view, 0, Some(c))?, None);
        graph.lower_next()?;
        assert!(matches!(&graph.ops.last().unwrap().kind,
            PublishedKind::RmsNorm { input_energy: None, checked: Some(_), .. }));
        Ok(())
    }).unwrap();
    // Width-2 fixture constraint: 2*C^2-1 == 7 (mod 8), impossible for
    // a sum of TWO integer squares, even before F32 representability matters.
    // The publisher's max of actual width-2 row statistics cannot equal it.
    let target = 2_u128 * (1_u128 << 52) - 1;
    assert_eq!(target % 8, 7);
    for x in 0_u128..8 {
        for y in 0_u128..8 { assert_ne!((x * x + y * y) % 8, target % 8); }
    }
    // We do NOT substitute cap-1 into a source/proof or claim positive authority
    // from a corrupt DTO. Width-6 controls below supply the authentic boundary.
}

const THRESHOLD_C: i64 = 67_108_864;
const THRESHOLD_R6: [i64; 6] = [150_059_968, 64_347, 391, 46, 7, THRESHOLD_C];

fn integer_row_squared_sum(row: &[i64]) -> u128 {
    row.iter().map(|value| u128::from(value.unsigned_abs()).pow(2)).sum()
}

#[test]
fn lookup_width6_threshold_integer_sums_f32_encoding_and_rms_bounds() {
    let q = 262_144_u128;
    let cap = 6_u128 * (1_u128 << 52);
    assert_eq!(ONE, 262_144);
    assert_eq!(THRESHOLD_C as u128 * THRESHOLD_C as u128, 1_u128 << 52);
    assert_eq!(integer_row_squared_sum(&THRESHOLD_R6[..5]), 22_517_998_136_852_479);
    assert_eq!(integer_row_squared_sum(&THRESHOLD_R6), 27_021_597_764_222_975);
    assert_eq!(integer_row_squared_sum(&THRESHOLD_R6), cap - 1);
    assert_eq!(integer_row_squared_sum(&[THRESHOLD_C; 6]), cap);
    assert_eq!(THRESHOLD_R6[0], 9_378_748 * 16);
    assert!((1 << 23..1 << 24).contains(&9_378_748));
    for value in THRESHOLD_R6.into_iter().chain([0]) {
        let encoded = value as f32;
        assert_eq!(f64::from(encoded), value as f64);
        let source = encoded / ONE as f32;
        assert_eq!(f64::from(source) * q as f64, value as f64);
        // Exercise the existing toward-zero encoder, not a rounded replacement.
        assert_eq!(crate::replicated_decoder::source::encode_q18(&source.to_le_bytes()).unwrap(), value);
    }

    // Source rms_normalization: square/TR -> ordered sum -> energy intersection
    // -> raw mean with the original F32 reciprocal coefficient -> TR -> epsilon.
    // Stochastic truncation of each exact endpoint C^2/Q has no endpoint slack.
    let square_upper = (1_u128 << 52) / q;
    assert_eq!(square_upper, 17_179_869_184);
    let sum_upper = 6 * square_upper;
    assert_eq!(sum_upper, 103_079_215_104);
    for bound in [cap - 1, cap] {
        let energy_sum_upper = (bound + 6 * (q - 1)) / q;
        assert!(energy_sum_upper >= sum_upper);
        // B>=C^2: intersecting with floor_sqrt(B) cannot narrow [-C,C].
        assert!(bound >= 1_u128 << 52);
    }
    let coefficient = q / 6;
    assert_eq!(coefficient, 43_690);
    assert_eq!(((1.0_f32 / 6.0_f32) * ONE as f32) as u128, coefficient);
    let raw_mean_upper = sum_upper * coefficient;
    assert_eq!(raw_mean_upper, (1_u128 << 52) - (1_u128 << 36));
    assert!(raw_mean_upper <= i64::MAX as u128);
    assert_eq!(raw_mean_upper % q, 0);
    let mean_upper = raw_mean_upper / q;
    assert_eq!(mean_upper, 17_179_607_040);
    assert_eq!(tensor::RMS_NORM_EPSILON_Q18, 1);
    assert_eq!(mean_upper + 1, 17_179_607_041);
    assert!(mean_upper + 1 < 1_u128 << 34);
    // Width 5's coefficient differs, but 5*floor(Q/5)=6*floor(Q/6)=Q-4:
    // both cap-domain raw mean uppers coincide. No reciprocal/profile change.
    assert_eq!(q / 5, 52_428);
    assert_eq!(5 * (q / 5), 6 * coefficient);
}

fn assert_authentic_width6_threshold(equality: bool) {
    let mut embedding = THRESHOLD_R6.to_vec();
    embedding.extend_from_slice(if equality { &[THRESHOLD_C; 6] } else { &THRESHOLD_R6 });
    embedding.extend_from_slice(&[0; 6]);
    let source_bound = embedding.chunks_exact(6).map(integer_row_squared_sum).max().unwrap();
    let cap = 6_u128 * (1_u128 << 52);
    assert_eq!(source_bound, if equality { cap } else { cap - 1 });
    let expected_energy = if equality { None } else { Some(source_bound) };
    assert_eq!(ExecutionPolicy::default().max_abs_activation, 256);
    with_embedding_geometry(&embedding, 256, false, Some(6), |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let view = graph.lower_next()?;
        let slot = view.slot.unwrap();
        let PublishedKind::TokenLookup { source, .. } = &graph.ops[slot].kind else {
            panic!("original lookup publication required")
        };
        assert_eq!(source.shape.as_slice(), &[3, 6]);
        assert_eq!(source.shape.iter().product::<usize>(), embedding.len());
        assert!(std::ptr::eq(*source, &graph.sources["model.embed_tokens.weight"]));
        assert_eq!(source.row_squared_norm, Some(source_bound));
        assert_eq!(context.manifest().tensors["model.embed_tokens.weight"].row_squared_norm, Some(source_bound));
        let broad = Interval::new(0, THRESHOLD_R6[0])?;
        let accepted = Interval::new(-THRESHOLD_C, THRESHOLD_C)?;
        assert_eq!(Interval::new(source.lower, source.upper)?, broad);
        assert_eq!(input_bounds(&graph, &view, 0, view.lanes)?, broad);
        assert_eq!(graph.imports.checked_bound, Some(THRESHOLD_C));
        assert!(broad.upper() > accepted.upper());
        assert_eq!(view.logical.row_width(), 6);
        admit_frame(&graph)?;
        for row in 0..view.lanes / 6 {
            assert_eq!(LookupRow::original(&graph, &view, row)?.unwrap().energy(&graph, &view, row)?, source_bound);
            assert_eq!(guarded_energy(&graph, &view, row, Some(THRESHOLD_C))?, expected_energy);
        }

        // Actual checked-range admission/construction MUST precede RMS. Any
        // geometry, range, ticket or downstream refusal fails this control; no
        // early error is accepted as threshold coverage and no broad RMS call
        // is substituted. No secret/runtime acceptance is asserted here.
        let incoming_validity = graph.validity.current();
        graph.lower_next()?;
        let op = graph.ops.last().unwrap();
        let PublishedKind::RmsNorm {
            gamma, input_energy, checked: Some(checked), row_bounds, program, ..
        } = &op.kind else {
            panic!("genuine checked RMS publication required")
        };
        assert!(same_view(&op.input, &view));
        assert_eq!(*input_energy, expected_energy);
        assert_eq!(*row_bounds, accepted);
        assert!(std::ptr::eq(*gamma, &graph.sources["model.layers.0.input_layernorm.weight"]));
        assert_eq!(gamma.public_q18.as_deref(), Some(&[ONE; 6][..]));
        assert_eq!(program.lanes(), 6);
        assert_eq!(program.output_lanes(), 6);
        assert_eq!(program.numeric_profile(), tensor::RMS_NORM_PROFILE);
        checked.validate(builder, &op.input, op.first, 6, incoming_validity)?;
        assert!(std::ptr::eq(checked.original, builder));
        assert_eq!(checked.accepted, accepted);
        assert!(checked.guarded());
        assert_eq!(checked.guard_tiles(), view.lanes.div_ceil(128));
        assert_eq!(checked.outgoing, graph.validity.current());
        assert!(checked.outgoing.is_some());
        let mut guard_count = 0;
        checked.visit(&mut |event| {
            assert!(event.owner < checked.rows_first());
            if let GuardCode::Checked { program, .. } = event.code {
                guard_count += 1;
                assert_eq!(program.numeric_profile(), tensor::CHECKED_RANGE_PROFILE);
                assert_eq!(program.output_bounds(), accepted);
                assert_eq!(program.validity_lanes(), Some(event.lanes / 2));
            }
            Ok(())
        })?;
        assert_eq!(guard_count, view.lanes.div_ceil(128));
        for row in 0..view.lanes / 6 {
            let safe = checked.row(builder, &op.input, row * 6, 6)?;
            assert_eq!(safe.bounds(), accepted);
            assert_eq!(safe.word(0)?, checked.word(row * 6)?);
            assert_ne!(safe.word(0)?, view.word(row * 6)?);
        }
        // R6 exceeds C: actual checked execution can sticky-fail. This proves
        // compiler custody/routing only, not successful numeric runtime. Both
        // bounds exceed C^2, so no coordinate narrowing is claimed.
        Ok(())
    }).unwrap();
}

#[test]
fn lookup_authentic_width6_strict_threshold_survives_guarded_rms_custody() {
    assert_authentic_width6_threshold(false);
}

#[test]
fn lookup_authentic_width6_cap_equality_retains_ordinary_guarded_rms_custody() {
    assert_authentic_width6_threshold(true);
}

#[test]
fn lookup_general_rejects_source_row_request_mismatch() {
    with_embedding(ONE / 64, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let view = graph.lower_next()?;
        assert!(matches!(graph.ops[view.slot.unwrap()].kind, PublishedKind::TokenLookup { .. }));
        admit_frame(&graph)?;
        let proof = LookupRow::original(&graph, &view, 0)?.unwrap();
        assert!(matches!(proof.energy(&graph, &view, 1), Err(Error::Binding)));
        assert!(matches!(LookupRow::original(&graph, &view, view.lanes / view.logical.row_width()), Err(Error::Shape)));

        let mut foreign = view.clone();
        foreign.origin = Arc::new(());
        assert!(matches!(LookupRow::original(&graph, &foreign, 0), Err(Error::Binding)));
        let proof = LookupRow::original(&graph, &view, 0)?.unwrap();
        let other = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        // Even a fresh graph using exactly the same source, imports and budget
        // cannot receive this original producer/row proof.
        assert!(matches!(proof.energy(&other, &view, 0), Err(Error::Binding)));
        drop(other);
        admit_frame(&graph)?;

        graph.validity.epoch = 1;
        assert!(matches!(LookupRow::original(&graph, &view, 0), Err(Error::Binding)));
        graph.validity.epoch = 0;
        let slot = view.slot.unwrap();
        let original = match &graph.ops[slot].kind {
            PublishedKind::TokenLookup { source, .. } => *source,
            _ => unreachable!(),
        };
        let wrong = graph.sources.values().find(|record| !std::ptr::eq(*record, original)).unwrap();
        if let PublishedKind::TokenLookup { source, .. } = &mut graph.ops[slot].kind { *source = wrong; }
        assert!(matches!(LookupRow::original(&graph, &view, 0), Err(Error::Binding)));
        Ok(())
    }).unwrap();
}

#[test]
fn lookup_zero_energy_uses_original_encoded_integer_statistic() {
    for value in [0, ONE / 64] {
        with_embedding(value, |builder, context| {
            let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
            let view = graph.lower_next()?;
            admit_frame(&graph)?;
            let width = view.logical.row_width();
            let expected = width as u128 * u128::from(value.unsigned_abs()).pow(2);
            for row in 0..view.lanes / width {
                assert_eq!(LookupRow::original(&graph, &view, row)?.unwrap().energy(&graph, &view, row)?, expected);
            }
            // Other GENERAL callers still receive no lookup/projection proof.
            assert_eq!(row_energy(&graph, &view, 0)?, None);
            assert!(graph.complete_row(&view, 0)?.is_none());
            assert_eq!(guarded_energy(&graph, &view, 0, Some(1 << 26))?, Some(expected));
            // The ordinary next RMS lowering, not a hand-selected producer or
            // factory, must carry the same original bound.
            graph.lower_next()?;
            assert!(matches!(&graph.ops.last().unwrap().kind,
                PublishedKind::RmsNorm { input_energy: Some(bound), .. } if *bound == expected));
            Ok(())
        }).unwrap();
    }
}

#[test]
fn authenticated_rms_output_seed_matches_retained_native_recipe() {
    // SOURCE precedent only, UNRUN: successful genuine outer publication must
    // retain certificate/recipe identity and complete ordered row geometry.
    // No worker outputs, full FFN/numerical fit or all-fallback proof is claimed.
    // Native canonical slot remains HOLD. Original residual traversal is a
    // separate unproven seam: intervening attention factories/funding are unknown.
    with_embedding(ONE / 64, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let lookup = graph.lower_next()?;
        let normalized = graph.lower_next()?;
        context.check_open()?;
        graph.validate_view(&normalized)?;

        // Independent encoded fixture scalar: two coordinates, each 4096.
        let expected = 2_u128.checked_mul(4096_u128.checked_pow(2).unwrap()).unwrap();
        assert_eq!(expected, 33_554_432);
        let slot = normalized.slot.unwrap();
        let op = &graph.ops[slot];
        let PublishedKind::RmsNorm {
            gamma, input_energy, output_energy, row_bounds, program,
            checked: Some(checked), ..
        } = &op.kind else {
            panic!("genuine checked RMS publication required")
        };
        assert!(same_view(&op.input, &lookup));
        assert_eq!(*input_energy, Some(expected));
        assert!(std::ptr::eq(*gamma, &graph.sources["model.layers.0.input_layernorm.weight"]));
        assert_eq!(gamma.public_q18.as_deref(), Some(&[ONE; 2][..]));
        assert_eq!(*row_bounds, checked.accepted);
        assert_eq!(program.output_bounds(), normalized.bounds);
        assert_eq!(program.lanes(), 2);
        assert_eq!(program.output_lanes(), 2);

        // SourceRequest's factory requires GeneralOutputRowSquared before this
        // retained publication. Rebuilding the recipe is NOT an authenticator.
        let (recipe, recipe_energy) = rms_norm_program(
            true, 2, *row_bounds, gamma.public_q18.as_ref().unwrap(), Some(expected),
        )?;
        assert_eq!(recipe.digest(), program.digest());
        assert_eq!(recipe_energy, Some(*output_energy));
        assert_eq!(recipe.output_bounds(), normalized.bounds);

        let width = normalized.logical.row_width();
        assert_eq!(width, 2);
        assert_eq!(normalized.lanes, 4 * width);
        let rows = graph.native_row_census()
            .filter(|row| row.descriptor == slot).collect::<Vec<_>>();
        assert_eq!(rows.len(), 4);
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row.row, index);
            assert_eq!(row.width, width);
            assert_eq!(row.offset, 0);
            assert_eq!(row.owner, checked.rows_first() + 2 * index + 1);
            assert!(Arc::ptr_eq(row.program, program));
            assert!(std::ptr::eq(row.gamma, *gamma));
            assert_eq!(row.energy, *output_energy);
            for lane in 0..width {
                assert_eq!(normalized.word(index * width + lane)?, (row.owner, lane));
            }
            // row_energy reads retained fields; it does not authenticate context,
            // producer or order. Outer publication and geometry above bind those.
            assert_eq!(row_energy(&graph, &normalized, index)?, Some(*output_energy));
        }
        let bx = *output_energy;
        let mx = u128::from(normalized.bounds.lower().unsigned_abs())
            .max(u128::from(normalized.bounds.upper().unsigned_abs()));
        assert_eq!(Some(bx), recipe_energy);
        assert_eq!(mx, u128::from(program.output_bounds().lower().unsigned_abs())
            .max(u128::from(program.output_bounds().upper().unsigned_abs())));
        // Mx uses the actual published output interval, not the input guard cap;
        // neither inequality against that cap nor exact output energy is assumed.
        Ok(())
    }).unwrap();
}

#[test]
fn lookup_stat_fractional_and_negative_refused_by_real_metadata_interface() {
    with_embedding(0, |_, context| {
        for malformed in [serde_json::json!(0.5), serde_json::json!(-1)] {
            let mut manifest = serde_json::to_value(context.manifest()).unwrap();
            manifest["tensors"]["model.embed_tokens.weight"]["row_squared_norm"] = malformed;
            assert!(source_entry::SourceRequestImport::new(&serde_json::to_vec(&manifest).unwrap()).is_err());
        }
        Ok(())
    }).unwrap();
}

#[test]
fn cap_equal_is_not_strict_energy_improvement() {
    let width = 4;
    let cap = crate::replicated_decoder::checked_row_energy::cap_energy(width, 1 << 26).unwrap();
    assert_eq!(crate::replicated_decoder::checked_row_energy::strict_improvement(cap, cap), None);
    assert_eq!(crate::replicated_decoder::checked_row_energy::strict_improvement(cap - 1, cap), Some(cap - 1));
}

#[test]
fn lookup_original_ledger_retry_never_resets_prior_debit_or_burn() {
    with_embedding(0, |builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let view = graph.lower_next()?;
        admit_frame(&graph)?;
        assert_eq!(LookupRow::original(&graph, &view, 0)?.unwrap().energy(&graph, &view, 0)?, 0);
        let reservation = context.reservation();
        let charged = reservation.payload_bytes();
        assert!(context.prepay_metadata(u64::MAX).is_err());
        for _ in 0..2 {
            assert!(matches!(LookupRow::original(&graph, &view, 0), Err(Error::Bounds)));
            assert!(context.prepay_metadata(0).is_err());
            assert_eq!(reservation.payload_bytes(), charged);
        }
        Ok(())
    }).unwrap();
}
