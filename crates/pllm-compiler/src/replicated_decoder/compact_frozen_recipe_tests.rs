//! Bounded schedule metadata parity, not full-request or execution qualification.
use super::super::super::super::schedule_finalization as shared;
use super::*;

fn fixture(rows: usize, vocabulary: usize) -> (DecoderGraph, BTreeMap<String, SourceTensor>) {
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type": "llama", "hidden_size": 2, "intermediate_size": 129,
        "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
        "head_dim": 2, "vocab_size": vocabulary, "tie_word_embeddings": false,
        "hidden_act": "silu", "attention_bias": false, "max_position_embeddings": 64,
        "rope_theta": 10000, "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
        "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
    }))
    .unwrap();
    let graph = lower_with_weights(&config, rows, 1, WeightMode::Public)
        .unwrap()
        .prefill;
    let sources = Program::source_shapes_with_weights(&config, rows, 1, WeightMode::Public)
        .unwrap()
        .into_iter()
        .map(|(key, shape)| {
            let public = key.contains("norm");
            let n = shape.iter().product();
            let value = if public { ONE } else { ONE / 64 };
            let mut hash = Sha256::new();
            for _ in 0..n {
                hash.update(value.to_le_bytes());
            }
            let row = (shape.len() == 2).then(|| shape[1] as u64 * (ONE / 64) as u64);
            (
                key,
                SourceTensor {
                    lower: if public { ONE } else { -ONE / 64 },
                    upper: value,
                    row_l1: row,
                    row_squared_norm: row
                        .map(|_| shape[1] as u128 * (ONE / 64) as u128 * (ONE / 64) as u128),
                    row_l1_squared_sum: row.map(|row| shape[0] as u128 * u128::from(row).pow(2)),
                    shape,
                    public_q18: public.then(|| vec![ONE; n]),
                    digest: hash.finalize().into(),
                },
            )
        })
        .collect();
    (graph, sources)
}

fn validated<'a>(
    graph: &DecoderGraph,
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
) -> ValidatedSourceDescriptors<'a> {
    let shapes = sources
        .iter()
        .map(|(key, source)| (key.clone(), source.shape.clone()))
        .collect();
    validate_source_records(
        graph,
        &shapes,
        sources,
        WeightMode::Public,
        ExecutionPolicy::default(),
        norms,
    )
    .unwrap()
}

fn imports(sources: &BTreeMap<String, SourceTensor>, rows: usize, tokens: bool) -> (Builder, View) {
    let mut builder = Builder {
        public_weights: true,
        ..Builder::default()
    };
    for (key, source) in sources
        .iter()
        .filter(|(_, source)| source.public_q18.is_none())
    {
        builder
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
    let (key, lanes, shape, bounds) = if tokens {
        let vocabulary = sources["model.embed_tokens.weight"].shape[0];
        (
            "input.tokens",
            rows,
            vec![1, rows],
            Interval::new(0, vocabulary as i64 - 1).unwrap(),
        )
    } else {
        (
            "hidden",
            2 * rows,
            vec![1, rows, 2],
            Interval::new(-ONE / 4, ONE / 4).unwrap(),
        )
    };
    let owner = builder
        .push(Code::Input(key.into()), lanes, bounds)
        .unwrap();
    let input = builder.view(owner, shape);
    (builder, input)
}

// Independent original numeric constructors, not reconstruction from adapter
// recipes. The expanded Builder remains the expected schedule/work authority.
fn expanded_op(
    builder: &mut Builder,
    sources: &BTreeMap<String, SourceTensor>,
    op: &pllm_models::ModelOperation,
    input: &View,
    right: Option<&View>,
) -> View {
    let shape = op.output_shape.iter().map(|n| *n as usize).collect();
    match op.operator {
        Op::Linear | Op::OutputHead => {
            let key = op.attributes["weight"].as_str().unwrap();
            let owner = sources
                .iter()
                .filter(|(_, s)| s.public_q18.is_none())
                .position(|(name, _)| name == key)
                .unwrap();
            let source = SourceView {
                owner,
                shape: sources[key].shape.clone(),
            };
            let mut output = builder
                .matrix(input, MatrixView::source(&source, false).unwrap(), true)
                .unwrap();
            output.shape = shape;
            builder
                .projection_energies(input, &output, &source, None)
                .unwrap();
            output
        }
        Op::Silu => {
            let gathered = builder.gather(&input.words).unwrap();
            let bounds = builder.nodes[gathered].bounds;
            let owner = builder
                .tensor_loop(LoopInput::Slice(gathered), input.words.len(), |n| {
                    tensor::Program::activation_guarded(tensor::Activation::Silu, n, bounds)
                })
                .unwrap();
            builder.view(owner, shape)
        }
        Op::Multiply => View {
            words: builder
                .binary(&input.words, &right.unwrap().words, true)
                .unwrap(),
            shape,
        },
        Op::ResidualAdd => View {
            words: builder.residual(input, right.unwrap()).unwrap(),
            shape,
        },
        Op::RmsNorm => {
            let gamma = &sources[op.attributes["weight"].as_str().unwrap()];
            let width = *input.shape.last().unwrap();
            let mut words = Words::new();
            for row in input.words.chunks_exact(width) {
                let owner = builder.rms_norm_row(&row, width, gamma).unwrap();
                words.extend((0..width).map(|i| (owner, i)));
            }
            View { words, shape }
        }
        Op::LastToken => {
            let width = *input.shape.last().unwrap();
            View {
                words: input.words.slice(input.words.len() - width..),
                shape,
            }
        }
        _ => panic!("unsupported original fixture operation"),
    }
}

fn assert_parity(
    schedule: &FrozenSchedule<'_, '_>,
    expanded: &Builder,
    selected: &[(usize, usize)],
) {
    let baseline = schedule.resources().finalization_baseline_bytes().unwrap();
    let actual_quote = shared::quote(schedule, selected, baseline).unwrap();
    let expected_quote = shared::quote(expanded.nodes.as_slice(), selected, 0).unwrap();
    assert_eq!(actual_quote.node_count, expanded.nodes.len());
    assert_eq!(actual_quote.work, expected_quote.work);
    assert_eq!(actual_quote.work, schedule_work(&expanded.nodes).unwrap());
    assert_eq!(actual_quote.work, schedule.resources().work);
    // Empty public prefix scopes this to schedule canonical parity, not source
    // identity serialization/admission or authenticated request publication.
    let prefix = Sha256::new();
    let actual = shared::finalize(
        schedule,
        selected,
        &prefix,
        baseline,
        actual_quote.peak_bytes,
    )
    .unwrap();
    let expected = shared::finalize(
        expanded.nodes.as_slice(),
        selected,
        &prefix,
        0,
        expected_quote.peak_bytes,
    )
    .unwrap();
    assert_eq!(actual.digest, expected.digest);
    assert_eq!(actual.work, expected.work);
    assert_eq!(actual.retire, expected.retire);
    assert!(matches!(
        shared::finalize(
            schedule,
            selected,
            &prefix,
            baseline,
            actual_quote.peak_bytes - 1
        ),
        Err(Error::Bounds)
    ));
}

#[test]
fn original_projection_silu_binary_rms_and_last_alias_survive_semantic_scope_exit() {
    for rows in [1, 3] {
        let (full, sources) = fixture(rows, 3);
        let norms = SourceNorms::new();
        let (imported, input) = imports(&sources, rows, false);
        let (mut expanded, reference) = imports(&sources, rows, false);
        let (frozen, selected, kernels) = {
            let semantic = full.clone();
            let start = semantic
                .operations
                .iter()
                .position(|op| op.id.ends_with(".gate_proj"))
                .unwrap();
            let entry = semantic.operations[start].inputs[0].clone();
            let residual = semantic.operations[start + 5]
                .inputs
                .iter()
                .find(|id| **id != semantic.operations[start + 4].id)
                .unwrap()
                .clone();
            let mut compact = CompactGraph::new(
                &imported,
                validated(&full, &sources, &norms),
                &semantic,
                start,
            )
            .unwrap();
            compact.bind_import(&entry, &input).unwrap();
            compact.bind_import(&residual, &input).unwrap();
            let mut values = BTreeMap::from([(entry, reference.clone()), (residual, reference)]);
            let mut exports = Vec::new();
            for op in &semantic.operations[start..start + 9] {
                let left = values[&op.inputs[0]].clone();
                let right = op.inputs.get(1).and_then(|id| values.get(id));
                let expected = expanded_op(&mut expanded, &sources, op, &left, right);
                let published = compact.lower_next().unwrap();
                assert!(published.words().eq(expected.words.iter()));
                values.insert(op.id.clone(), expected);
                exports.push(published);
            }
            let selected = exports.last().unwrap().words().collect::<Vec<_>>();
            let kernels = compact
                .owned_native_programs()
                .map(Arc::downgrade)
                .collect::<Vec<_>>();
            let quote = compact.freeze_quote(&exports).unwrap();
            (
                compact.freeze(exports, quote.peak_bytes).unwrap(),
                selected,
                kernels,
            )
        };
        let attempts = super::super::super::super::super::prepared_tensor_map::native_attempts();
        let quote = frozen.schedule_recipe_quote().unwrap();
        let schedule = frozen.borrow_schedule(quote.peak_bytes).unwrap();
        assert_eq!(schedule.resources(), quote);
        assert_eq!(
            quote.gather_bytes,
            schedule.gathers.capacity() * size_of::<GatherSpan<'_>>()
        );
        assert!(std::ptr::eq(frozen.borrowed_source_records(), &sources));
        for ordinal in 0..imported.nodes.len() {
            assert!(std::ptr::eq(
                frozen.imported_recipe(ordinal).unwrap(),
                &imported.nodes[ordinal]
            ));
        }
        for (slot, op) in frozen.ops.iter().enumerate() {
            if matches!(
                op.kind,
                PublishedKind::Projection { .. } | PublishedKind::Silu
            ) {
                let last = op.first
                    + if matches!(op.kind, PublishedKind::Silu) {
                        1
                    } else {
                        2
                    };
                let BorrowedRecipe::TensorLoop { program, tail, .. } =
                    schedule.node(last).unwrap().recipe
                else {
                    panic!()
                };
                let original = frozen.prepared(slot).unwrap();
                assert!(std::ptr::eq(program, original.program_at(0).unwrap()));
                if let Some(tail) = tail {
                    assert!(std::ptr::eq(
                        tail,
                        original
                            .program_at(original.lanes() - tail.lanes())
                            .unwrap()
                    ));
                }
            }
        }
        assert_parity(&schedule, &expanded, &selected);
        assert_eq!(
            attempts,
            super::super::super::super::super::prepared_tensor_map::native_attempts()
        );
        drop(schedule);
        assert!(kernels.iter().all(|weak| weak.upgrade().is_some()));
        drop(frozen);
        assert!(kernels.iter().all(|weak| weak.upgrade().is_none()));
    }
}

#[test]
fn per_token_full_tail_gather_order_and_source_owners_match_expanded() {
    for vocabulary in [1, 128, 129, 257] {
        let (full, sources) = fixture(3, vocabulary);
        let norms = SourceNorms::new();
        let (imported, input) = imports(&sources, 3, true);
        let (mut expanded, reference) = imports(&sources, 3, true);
        let frozen = {
            let mut semantic = full.clone();
            semantic
                .operations
                .retain(|op| op.operator == Op::TokenLookup);
            semantic.output = semantic.operations[0].id.clone();
            let mut compact =
                CompactGraph::new(&imported, validated(&full, &sources, &norms), &semantic, 0)
                    .unwrap();
            compact.bind_import("input.tokens", &input).unwrap();
            let output = compact.lower_next().unwrap();
            let quote = compact.freeze_quote(&vec![output.clone()]).unwrap();
            compact.freeze(vec![output], quote.peak_bytes).unwrap()
        };
        let census = frozen.token_lookup_census().next().unwrap();
        let weight = census.source_owner;
        let table = SourceView {
            owner: weight,
            shape: vec![vocabulary, 2],
        };
        let table_bounds = expanded.nodes[weight].bounds;
        for (owner, index) in reference.words.iter() {
            let original = expanded.nodes[owner].bounds;
            let bounds =
                Interval::new(original.lower() - (vocabulary - 1) as i64, original.upper())
                    .unwrap();
            let equal = expanded
                .tensor_loop(LoopInput::TokenOffsets { owner, index }, vocabulary, |n| {
                    tensor::Program::equal_zero(n, bounds)
                })
                .unwrap();
            let selectors = expanded.view(equal, vec![1, vocabulary]);
            let row = expanded
                .matrix(&selectors, MatrixView::source(&table, true).unwrap(), false)
                .unwrap();
            for (owner, _) in row.words.iter() {
                expanded.nodes[owner].bounds = table_bounds;
            }
        }
        let quote = frozen.schedule_recipe_quote().unwrap();
        let schedule = frozen.borrow_schedule(quote.peak_bytes).unwrap();
        for token in 0..3 {
            let first = imported.nodes.len() + 3 * token;
            let BorrowedRecipe::TensorLoop { program, tail, .. } =
                schedule.node(first).unwrap().recipe
            else {
                panic!()
            };
            assert!(std::ptr::eq(program, census.program));
            assert_eq!(tail.map(Arc::as_ptr), census.tail.map(Arc::as_ptr));
            let BorrowedRecipe::Gather(gather) = schedule.node(first + 1).unwrap().recipe else {
                panic!()
            };
            assert_eq!(gather.len(), vocabulary);
            for lane in 0..vocabulary {
                assert_eq!(gather.lane_ref(lane).unwrap(), (first, lane));
            }
            assert!(gather.lane_ref(vocabulary).is_err());
        }
        let selected = frozen.exports()[0].words().collect::<Vec<_>>();
        assert_parity(&schedule, &expanded, &selected);
    }
}

#[test]
fn alias_only_region_has_no_control_backing_or_new_logical_owners() {
    let (full, sources) = fixture(3, 3);
    let norms = SourceNorms::new();
    let (imported, input) = imports(&sources, 3, false);
    let frozen = {
        let mut semantic = full.clone();
        semantic
            .operations
            .retain(|op| op.operator == Op::LastToken);
        semantic.output = semantic.operations[0].id.clone();
        let mut compact =
            CompactGraph::new(&imported, validated(&full, &sources, &norms), &semantic, 0).unwrap();
        compact
            .bind_import(&semantic.operations[0].inputs[0], &input)
            .unwrap();
        let output = compact.lower_next().unwrap();
        compact.freeze(vec![output], MAX_PROGRAM_BYTES).unwrap()
    };
    let quote = frozen.schedule_recipe_quote().unwrap();
    assert_eq!(quote.gather_spans, 0);
    assert_eq!(quote.gather_bytes, 0);
    FREEZE_FAIL_AFTER.with(|n| n.set(Some(0)));
    let schedule = frozen.borrow_schedule(quote.peak_bytes).unwrap();
    assert_eq!(FREEZE_FAIL_AFTER.with(|n| n.get()), Some(0));
    FREEZE_FAIL_AFTER.with(|n| n.set(None));
    assert_eq!(schedule.node_count(), imported.nodes.len());
    assert_eq!(
        frozen.exports()[0].word(0).unwrap(),
        (imported.nodes.len() - 1, 4)
    );
    assert!(
        matches!(
            shared::quote(&schedule, &[], quote.retained_bytes),
            Err(Error::Bounds)
        ),
        "zero work does not become complete schedule authority"
    );
}

#[test]
fn exact_short_and_failed_control_reservation_do_not_clone_or_rebuild_owners() {
    let (full, sources) = fixture(1, 3);
    let norms = SourceNorms::new();
    let (imported, input) = imports(&sources, 1, false);
    let start = full
        .operations
        .iter()
        .position(|op| op.id.ends_with(".gate_proj"))
        .unwrap();
    let mut compact =
        CompactGraph::new(&imported, validated(&full, &sources, &norms), &full, start).unwrap();
    compact
        .bind_import(&full.operations[start].inputs[0], &input)
        .unwrap();
    let mut exports = Vec::new();
    for _ in 0..4 {
        exports.push(compact.lower_next().unwrap());
    }
    let frozen = compact.freeze(exports, MAX_PROGRAM_BYTES).unwrap();
    let quote = frozen.schedule_recipe_quote().unwrap();
    assert_eq!(quote.gather_spans, 4); // Two 128/1 tiles, two gathers per tile.
    quote.admit(quote.peak_bytes).unwrap();
    assert_eq!(quote.admit(quote.peak_bytes - 1), Err(Error::Bounds));
    FREEZE_FAIL_AFTER.with(|n| n.set(Some(0)));
    assert!(matches!(
        frozen.borrow_schedule(quote.peak_bytes - 1),
        Err(Error::Bounds)
    ));
    assert_eq!(
        FREEZE_FAIL_AFTER.with(|n| n.get()),
        Some(0),
        "short cap must not attempt reservation"
    );
    assert!(matches!(
        frozen.borrow_schedule(quote.peak_bytes),
        Err(Error::Allocation)
    ));
    FREEZE_FAIL_AFTER.with(|n| n.set(None));
    let schedule = frozen.borrow_schedule(quote.peak_bytes).unwrap();
    assert_eq!(schedule.resources(), quote);
    assert_eq!(
        frozen.work_budget(),
        shared::quote(&schedule, &[], quote.retained_bytes)
            .unwrap()
            .work
    );
}

#[test]
fn wrong_projection_and_argmax_geometry_fail_closed() {
    let (full, sources) = fixture(1, 3);
    let norms = SourceNorms::new();
    let (imported, input) = imports(&sources, 1, false);
    let start = full
        .operations
        .iter()
        .position(|op| op.id.ends_with(".gate_proj"))
        .unwrap();
    for count in [1, 10] {
        let mut compact =
            CompactGraph::new(&imported, validated(&full, &sources, &norms), &full, start).unwrap();
        compact
            .bind_import(&full.operations[start].inputs[0], &input)
            .unwrap();
        let residual = full.operations[start + 5]
            .inputs
            .iter()
            .find(|id| **id != full.operations[start + 4].id)
            .unwrap();
        compact.bind_import(residual, &input).unwrap();
        let mut exports = Vec::new();
        for _ in 0..count {
            exports.push(compact.lower_next().unwrap());
        }
        let mut frozen = compact.freeze(exports, MAX_PROGRAM_BYTES).unwrap();
        if count == 1 {
            let PublishedKind::Projection { geometry, .. } = &mut frozen.ops[0].kind else {
                panic!()
            };
            geometry[2] += 1;
            assert!(matches!(frozen.schedule_recipe_quote(), Err(Error::Shape)));
        } else {
            let PublishedKind::Argmax(plan) = &mut frozen.ops[9].kind else {
                panic!()
            };
            plan.geometry.lanes += 1;
            assert!(matches!(frozen.schedule_recipe_quote(), Err(Error::Shape)));
        }
        FREEZE_FAIL_AFTER.with(|n| n.set(Some(0)));
        assert!(frozen.borrow_schedule(MAX_PROGRAM_BYTES).is_err());
        assert_eq!(FREEZE_FAIL_AFTER.with(|n| n.get()), Some(0));
        FREEZE_FAIL_AFTER.with(|n| n.set(None));
    }
}

#[test]
fn lazy_head_permutation_and_feedback_keep_original_alias_coordinates() {
    let (full, sources) = fixture(3, 3);
    let norms = SourceNorms::new();
    let (imported, input) = imports(&sources, 3, false);
    let frozen = {
        let mut semantic = full.clone();
        let mut reshape = semantic.operations[0].clone();
        reshape.id = "local.heads".into();
        reshape.operator = Op::Reshape;
        reshape.inputs = vec!["local.hidden".into()];
        reshape.output_shape = vec![1, 2, 3, 1];
        reshape.attributes = serde_json::json!({"layout": "batch_heads_sequence_feature"});
        let mut feedback = reshape.clone();
        feedback.id = "local.history".into();
        feedback.operator = Op::TokenFeedback;
        feedback.inputs = vec![reshape.id.clone()];
        feedback.output_shape = vec![6];
        feedback.attributes = serde_json::json!({});
        semantic.output = feedback.id.clone();
        semantic.operations = vec![reshape, feedback];
        semantic.state_inputs.clear();
        semantic.state_outputs.clear();
        let mut compact =
            CompactGraph::new(&imported, validated(&full, &sources, &norms), &semantic, 0).unwrap();
        compact.bind_import("local.hidden", &input).unwrap();
        let heads = compact.lower_next().unwrap();
        let feedback = compact.lower_next().unwrap();
        assert_eq!(
            feedback.logical, heads.logical,
            "feedback retains input rank/shape"
        );
        compact
            .freeze(vec![heads, feedback], MAX_PROGRAM_BYTES)
            .unwrap()
    };
    let quote = frozen.schedule_recipe_quote().unwrap();
    assert_eq!(quote.gather_spans, 0);
    let schedule = frozen.borrow_schedule(quote.peak_bytes).unwrap();
    assert_eq!(schedule.node_count(), imported.nodes.len());
    let owner = input.words.at(0).0;
    for view in frozen.exports() {
        assert_eq!(
            view.words().collect::<Vec<_>>(),
            [0, 2, 4, 1, 3, 5].map(|index| (owner, index))
        );
        for lane in 0..view.lanes {
            assert_eq!(frozen.word(view, lane).unwrap(), view.word(lane).unwrap());
        }
    }
    assert!(schedule.node(schedule.node_count()).is_err());
}

#[test]
fn absent_original_lookup_tail_rejects_before_control_allocation() {
    let (full, sources) = fixture(3, 129);
    let norms = SourceNorms::new();
    let (imported, input) = imports(&sources, 3, true);
    let mut semantic = full.clone();
    semantic
        .operations
        .retain(|op| op.operator == Op::TokenLookup);
    semantic.output = semantic.operations[0].id.clone();
    let mut compact =
        CompactGraph::new(&imported, validated(&full, &sources, &norms), &semantic, 0).unwrap();
    compact.bind_import("input.tokens", &input).unwrap();
    let output = compact.lower_next().unwrap();
    let mut frozen = compact.freeze(vec![output], MAX_PROGRAM_BYTES).unwrap();
    let PublishedKind::TokenLookup { plan, .. } = &mut frozen.ops[0].kind else {
        panic!()
    };
    assert!(plan.tail.take().is_some());
    FREEZE_FAIL_AFTER.with(|n| n.set(Some(0)));
    assert!(matches!(frozen.schedule_recipe_quote(), Err(Error::Shape)));
    assert!(matches!(
        frozen.borrow_schedule(MAX_PROGRAM_BYTES),
        Err(Error::Shape)
    ));
    assert_eq!(FREEZE_FAIL_AFTER.with(|n| n.get()), Some(0));
    FREEZE_FAIL_AFTER.with(|n| n.set(None));
}

fn argmax_inputs(sources: &BTreeMap<String, SourceTensor>, lanes: usize) -> (Builder, View) {
    let (mut builder, _) = imports(sources, 1, false);
    let owner = builder
        .push(
            Code::Input("local.logits".into()),
            lanes,
            Interval::new(-32, 32).unwrap(),
        )
        .unwrap();
    // Bind the genuine whole Input. Offset slices are intentionally not import
    // declarations; arithmetic/gather offsets remain covered by the oracle.
    let input = builder.view(owner, vec![1, 1, lanes]);
    (builder, input)
}

fn argmax_semantic(full: &DecoderGraph, lanes: usize) -> DecoderGraph {
    let mut semantic = full.clone();
    let mut product = full
        .operations
        .iter()
        .find(|op| op.operator == Op::Multiply)
        .unwrap()
        .clone();
    product.id = "local.product".into();
    product.inputs = vec!["local.logits".into(), "local.logits".into()];
    product.output_shape = vec![1, 1, lanes as u64];
    let mut selection = full
        .operations
        .iter()
        .find(|op| op.operator == Op::GreedyTokenSelection)
        .unwrap()
        .clone();
    selection.id = "local.argmax".into();
    selection.inputs = vec![product.id.clone()];
    let mut feedback = selection.clone();
    feedback.id = "local.history".into();
    feedback.operator = Op::TokenFeedback;
    feedback.inputs = vec![selection.id.clone()];
    feedback.output_shape = vec![1, 1];
    feedback.attributes = serde_json::json!({});
    semantic.output = feedback.id.clone();
    semantic.operations = vec![product, selection, feedback];
    semantic.state_inputs.clear();
    semantic.state_outputs.clear();
    semantic
}

fn freeze_argmax<'a>(
    full: &DecoderGraph,
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
    imported: &'a Builder,
    input: &View,
    lanes: usize,
) -> FrozenRegion<'a> {
    let semantic = argmax_semantic(full, lanes);
    let mut compact =
        CompactGraph::new(imported, validated(full, sources, norms), &semantic, 0).unwrap();
    compact.bind_import("local.logits", input).unwrap();
    let mut exports = Vec::new();
    for _ in 0..3 {
        exports.push(compact.lower_next().unwrap());
    }
    compact.freeze(exports, MAX_PROGRAM_BYTES).unwrap()
    // The semantic graph is destroyed before any schedule loan is constructed.
}

#[test]
fn generated_argmax_iota_tails_carries_and_feedback_match_original_expanded_schedule() {
    let (full, sources) = fixture(1, 3);
    let norms = SourceNorms::new();
    for lanes in [1, 2, 3, 127, 128, 129, 255, 256, 257, 258, 511, 513, 1025] {
        let (imported, input) = argmax_inputs(&sources, lanes);
        let (mut expanded, reference) = argmax_inputs(&sources, lanes);
        let product = View {
            words: expanded
                .binary(&reference.words, &reference.words, true)
                .unwrap(),
            shape: vec![1, 1, lanes],
        };
        let expected_iota = expanded.nodes.len();
        let expected = expanded.argmax_tree(&product).unwrap();
        let frozen = freeze_argmax(&full, &sources, &norms, &imported, &input, lanes);
        let recipes = frozen.ops.as_ptr();
        let exports = frozen.exports.as_ptr();
        let origin = Arc::downgrade(&frozen.origin);
        let PublishedKind::Argmax(plan) = &frozen.ops[1].kind else {
            panic!()
        };
        let kernels = plan
            .programs
            .iter()
            .map(|kernel| {
                (
                    &kernel.program as *const Arc<tensor::Program>,
                    Arc::downgrade(&kernel.program),
                    Arc::strong_count(&kernel.program),
                )
            })
            .collect::<Vec<_>>();
        let attempts = ARGMAX_NATIVE_ATTEMPTS.with(|n| n.get());
        let quote = frozen.schedule_recipe_quote().unwrap();
        assert!(quote.replay_bytes > 0);
        assert_eq!(
            quote.gather_spans,
            (if lanes > 128 {
                2 * lanes.div_ceil(128)
            } else {
                0
            }) + 4 * plan.geometry.blocks
        );
        assert_eq!(
            quote.gather_bytes,
            quote.gather_spans * size_of::<GatherSpan<'_>>()
        );
        assert_eq!(plan.first, expected_iota);
        assert!(matches!(
            GeneratedPublicValues::argmax_indices(&frozen, 0),
            Err(Error::Binding)
        ));
        let schedule = frozen.borrow_schedule(quote.peak_bytes).unwrap();
        assert_eq!(schedule.node_count(), expanded.nodes.len());
        assert!(matches!(
            schedule.node(expected_iota).unwrap().recipe,
            BorrowedRecipe::GeneratedPublicValues(_)
        ));
        let mut tensor_nodes = 0;
        for ordinal in expected_iota..schedule.node_count() {
            let node = schedule.node(ordinal).unwrap();
            assert_eq!(node.lanes, expanded.nodes[ordinal].lanes);
            assert_eq!(node.bounds, expanded.nodes[ordinal].bounds);
            match node.recipe {
                BorrowedRecipe::Gather(gather) => {
                    let Code::Gather(words) = &expanded.nodes[ordinal].code else {
                        panic!()
                    };
                    assert_eq!(gather.len(), words.len());
                    for (index, word) in words.iter().enumerate() {
                        assert_eq!(gather.lane_ref(index).unwrap(), word);
                    }
                }
                BorrowedRecipe::Tensor { program, .. } => {
                    tensor_nodes += 1;
                    assert!(
                        kernels
                            .iter()
                            .any(|(handle, _, _)| std::ptr::eq(program, *handle))
                    );
                }
                _ => (),
            }
        }
        assert_eq!(tensor_nodes, plan.geometry.blocks);
        for (view, expected) in frozen.exports()[1..].iter().zip([&expected, &expected]) {
            assert_eq!(view.logical.dimensions(), expected.shape);
            assert!(view.words().eq(expected.words.iter()));
        }
        let mut selected = expected.words.iter().collect::<Vec<_>>();
        selected.extend([(expected_iota, lanes - 1), (expected_iota, 0)]);
        assert_parity(&schedule, &expanded, &selected);
        assert_eq!(frozen.ops.as_ptr(), recipes);
        assert_eq!(frozen.exports.as_ptr(), exports);
        assert!(std::ptr::eq(frozen.borrowed_source_records(), &sources));
        assert!(std::ptr::eq(
            frozen.imported_recipe(0).unwrap(),
            &imported.nodes[0]
        ));
        assert_eq!(ARGMAX_NATIVE_ATTEMPTS.with(|n| n.get()), attempts);
        for (kernel, (_, weak, count)) in plan.programs.iter().zip(&kernels) {
            assert_eq!(Arc::as_ptr(&kernel.program), weak.as_ptr());
            assert_eq!(Arc::strong_count(&kernel.program), *count);
        }
        drop(schedule);
        assert!(kernels.iter().all(|(_, weak, _)| weak.upgrade().is_some()));
        drop(frozen);
        assert!(origin.upgrade().is_none());
        assert!(kernels.iter().all(|(_, weak, _)| weak.upgrade().is_none()));
    }
}

#[test]
fn argmax_recipe_exact_short_overflow_and_geometry_reject_before_control_reservation() {
    let (full, sources) = fixture(1, 3);
    let norms = SourceNorms::new();
    let (imported, input) = argmax_inputs(&sources, 258);
    let frozen = freeze_argmax(&full, &sources, &norms, &imported, &input, 258);
    let quote = frozen.schedule_recipe_quote().unwrap();
    let attempts = ARGMAX_NATIVE_ATTEMPTS.with(|n| n.get());
    FREEZE_FAIL_AFTER.with(|n| n.set(Some(0)));
    assert!(matches!(
        frozen.borrow_schedule(quote.peak_bytes - 1),
        Err(Error::Bounds)
    ));
    assert_eq!(FREEZE_FAIL_AFTER.with(|n| n.get()), Some(0));
    assert!(matches!(
        frozen.borrow_schedule(quote.peak_bytes),
        Err(Error::Allocation)
    ));
    FREEZE_FAIL_AFTER.with(|n| n.set(None));
    let schedule = frozen.borrow_schedule(quote.peak_bytes).unwrap();
    assert_eq!(schedule.resources(), quote);
    assert_eq!(ARGMAX_NATIVE_ATTEMPTS.with(|n| n.get()), attempts);
    drop(schedule);
    drop(frozen);

    for change in 0..6 {
        let mut frozen = freeze_argmax(&full, &sources, &norms, &imported, &input, 258);
        let PublishedKind::Argmax(plan) = &mut frozen.ops[1].kind else {
            panic!()
        };
        match change {
            0 => plan.geometry.lanes = 0,
            1 => plan.geometry.blocks = usize::MAX,
            2 => plan.geometry.rounds += 1,
            3 => {
                let (first, rest) = plan.programs.split_at_mut(1);
                assert_ne!(first[0].lanes, rest[0].lanes);
                std::mem::swap(&mut first[0].program, &mut rest[0].program);
            }
            4 => plan.work = WorkBudget::default(),
            _ => frozen.resources.retained_bytes = usize::MAX,
        }
        FREEZE_FAIL_AFTER.with(|n| n.set(Some(0)));
        assert!(frozen.schedule_recipe_quote().is_err(), "change={change}");
        assert!(
            frozen.borrow_schedule(MAX_PROGRAM_BYTES).is_err(),
            "change={change}"
        );
        assert_eq!(FREEZE_FAIL_AFTER.with(|n| n.get()), Some(0));
        FREEZE_FAIL_AFTER.with(|n| n.set(None));
    }
}
