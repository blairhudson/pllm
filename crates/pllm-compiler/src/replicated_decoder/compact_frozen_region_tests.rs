//! Recipe retention controls, not compact request/execution qualification.
use super::*;

fn fixture() -> (
    DecoderGraph,
    BTreeMap<String, SourceTensor>,
    SourceNorms,
    Builder,
    View,
) {
    let config = br#"{"model_type":"llama","hidden_size":2,"intermediate_size":129,
        "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
        "head_dim":2,"vocab_size":3,"tie_word_embeddings":false,"hidden_act":"silu",
        "attention_bias":false,"max_position_embeddings":64,"rope_theta":10000,
        "rms_norm_eps":0.000001,"attention_dropout":0.0,"use_cache":true}"#;
    let graph = lower_with_weights(config, 1, 1, WeightMode::Public)
        .unwrap()
        .prefill;
    let sources: BTreeMap<_, _> =
        Program::source_shapes_with_weights(config, 1, 1, WeightMode::Public)
            .unwrap()
            .into_iter()
            .map(|(key, shape)| {
                let public = key.contains("norm");
                let n = shape.iter().product::<usize>();
                let value = if public { ONE } else { ONE / 64 };
                let mut hash = Sha256::new();
                for _ in 0..n {
                    hash.update(value.to_le_bytes());
                }
                let row = (shape.len() == 2).then(|| shape[1] as u64 * (ONE / 64) as u64);
                let source = SourceTensor {
                    lower: if public { ONE } else { -ONE / 64 },
                    upper: value,
                    row_l1: row,
                    row_squared_norm: row
                        .map(|_| shape[1] as u128 * (ONE / 64) as u128 * (ONE / 64) as u128),
                    row_l1_squared_sum: row.map(|row| shape[0] as u128 * u128::from(row).pow(2)),
                    shape,
                    public_q18: public.then(|| vec![ONE; n]),
                    digest: hash.finalize().into(),
                };
                (key, source)
            })
            .collect();
    let norms = SourceNorms::new();
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
                    row_l1: source.row_l1,
                    row_squared_norm: source.row_squared_norm,
                    row_l1_squared_sum: source.row_l1_squared_sum,
                },
                source.shape.iter().product(),
                Interval::new(source.lower, source.upper).unwrap(),
            )
            .unwrap();
    }
    let owner = imports
        .push(
            Code::Input("hidden".into()),
            2,
            Interval::new(-ONE / 4, ONE / 4).unwrap(),
        )
        .unwrap();
    let input = imports.view(owner, vec![1, 1, 2]);
    (graph, sources, norms, imports, input)
}

fn region<'g, 'a>(
    graph: &'g DecoderGraph,
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
    imports: &'a Builder,
    input: &View,
    count: usize,
) -> (CompactGraph<'g, 'a>, Vec<PublishedView>) {
    let shapes = sources
        .iter()
        .map(|(key, source)| (key.clone(), source.shape.clone()))
        .collect();
    let validated = validate_source_records(
        graph,
        &shapes,
        sources,
        WeightMode::Public,
        ExecutionPolicy::default(),
        norms,
    )
    .unwrap();
    let start = graph
        .operations
        .iter()
        .position(|op| op.id.ends_with(".gate_proj"))
        .unwrap();
    let mut compact = CompactGraph::new(imports, validated, graph, start).unwrap();
    compact
        .bind_import(&graph.operations[start].inputs[0], input)
        .unwrap();
    let residual = graph.operations[start + 5]
        .inputs
        .iter()
        .find(|name| **name != graph.operations[start + 4].id)
        .unwrap();
    compact.bind_import(residual, input).unwrap();
    let mut exports = Vec::new();
    for _ in 0..count {
        exports.push(compact.lower_next().unwrap());
    }
    (compact, exports)
}

fn attempts() -> (usize, usize, usize, usize) {
    (
        FREEZE_ALLOCATIONS.with(|count| count.get()),
        super::super::super::super::prepared_tensor_map::native_attempts(),
        RMS_NATIVE_ATTEMPTS.with(|count| count.get()),
        ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()),
    )
}

#[test]
fn freeze_outlives_semantic_graph_moves_original_recipes_kernels_names_and_views() {
    let (graph, sources, norms, imports, input) = fixture();
    let (frozen, pointers, output_words, work, next, names, output_name, recipe_ptr, export_ptr) = {
        let semantic = graph;
        let (compact, exports) = region(&semantic, &sources, &norms, &imports, &input, 11);
        let pointers: Vec<_> = compact
            .owned_native_programs()
            .map(|program| (Arc::as_ptr(program), Arc::downgrade(program)))
            .collect();
        let output_words: Vec<_> = exports
            .iter()
            .map(|view| view.words().collect::<Vec<_>>())
            .collect();
        let start = compact.cursor - compact.ops.len();
        let names: Vec<_> = semantic.operations[start..compact.cursor]
            .iter()
            .map(|op| op.id.clone())
            .collect();
        let output_name = semantic.output.clone();
        let work = compact.work_budget();
        let next = compact.next;
        let recipe_ptr = compact.ops.as_ptr();
        let export_ptr = exports.as_ptr();
        let native_before = attempts();
        let quote = compact.freeze_quote(&exports).unwrap();
        let frozen = compact.freeze(exports, quote.peak_bytes).unwrap();
        let native_after = attempts();
        assert_eq!(
            (native_before.1, native_before.2, native_before.3),
            (native_after.1, native_after.2, native_after.3)
        );
        assert_eq!(frozen.resources(), quote);
        (
            frozen,
            pointers,
            output_words,
            work,
            next,
            names,
            output_name,
            recipe_ptr,
            export_ptr,
        )
        // semantic drops here; FrozenRegion's public type has only lifetime 'a.
    };
    assert_eq!(frozen.ops.as_ptr(), recipe_ptr);
    assert_eq!(frozen.exports.as_ptr(), export_ptr);
    assert_eq!(frozen.work_budget(), work);
    assert_eq!(frozen.next_virtual_owner(), next);
    assert_eq!(frozen.descriptor_census().count(), names.len());
    let actual_name_bytes = frozen
        .imports
        .iter()
        .map(|entry| entry.name.capacity())
        .sum::<usize>()
        + frozen
            .recipe_names
            .iter()
            .map(String::capacity)
            .sum::<usize>();
    let actual_new_backing = actual_name_bytes
        + frozen.imports.capacity() * size_of::<Import>()
        + frozen.recipe_names.capacity() * size_of::<String>()
        + frozen.frontier.capacity() * size_of::<Frontier>();
    assert_eq!(frozen.resources().name_bytes, actual_name_bytes);
    assert_eq!(frozen.resources().new_backing_bytes, actual_new_backing);
    assert!(std::ptr::eq(frozen.imports_owner, &imports));
    for (owner, original) in imports.nodes.iter().enumerate() {
        assert!(std::ptr::eq(
            frozen.imported_recipe(owner).unwrap(),
            original
        ));
    }
    assert!(frozen.imported_recipe(imports.nodes.len()).is_none());
    assert!(std::ptr::eq(frozen.borrowed_source_records(), &sources));
    assert!(std::ptr::eq(frozen.borrowed_source_norms(), &norms));
    assert_eq!(
        frozen.source_binding(),
        source_binding_with_norms(&sources, &norms)
    );
    for (index, name) in names.iter().enumerate() {
        assert_eq!(frozen.recipe_name(index), Some(name.as_str()));
    }
    for (view, words) in frozen.exports().iter().zip(&output_words) {
        frozen.resolve(view).unwrap();
        assert_eq!(
            (0..words.len())
                .map(|i| frozen.word(view, i).unwrap())
                .collect::<Vec<_>>(),
            *words
        );
    }
    assert!(same_view(
        frozen.output().unwrap(),
        frozen.named_view(&output_name).unwrap()
    ));
    assert!(
        frozen.named_view(&names[3]).is_none(),
        "retired frontier must not resurrect"
    );
    for (pointer, weak) in &pointers {
        assert_eq!(Arc::as_ptr(&weak.upgrade().unwrap()), *pointer);
    }
    for (left, right) in frozen
        .prepared(0)
        .unwrap()
        .programs()
        .zip(frozen.prepared(1).unwrap().programs())
    {
        assert!(Arc::ptr_eq(left, right), "original prepared alias lost");
    }
    assert!(frozen.native_row_program(6).is_some());
    let mut events = 0;
    frozen
        .visit_argmax(9, &mut |_| {
            events += 1;
            Ok(())
        })
        .unwrap();
    assert!(events > 0);
    drop(frozen);
    assert!(pointers.iter().all(|(_, weak)| weak.upgrade().is_none()));
}

#[test]
fn freeze_exact_fit_one_short_and_spare_capacities_use_recomputed_quote() {
    let (graph, sources, norms, imports, input) = fixture();
    let (compact, mut exports) = region(&graph, &sources, &norms, &imports, &input, 2);
    let before = compact.freeze_quote(&exports).unwrap();
    exports.reserve_exact(7);
    let after = compact.freeze_quote(&exports).unwrap();
    assert_eq!(after.recipe_capacity, compact.ops.capacity());
    assert_eq!(after.export_capacity, exports.capacity());
    assert_eq!(
        after.export_bytes - before.export_bytes,
        (after.export_capacity - before.export_capacity) * size_of::<PublishedView>()
    );
    assert_eq!(
        after.retained_bytes - before.retained_bytes,
        after.export_bytes - before.export_bytes
    );
    after.admit(after.peak_bytes).unwrap();
    assert_eq!(after.admit(after.peak_bytes - 1), Err(Error::Bounds));
    let native = compact
        .owned_native_programs()
        .map(Arc::downgrade)
        .collect::<Vec<_>>();
    let count = attempts();
    assert!(matches!(
        compact.freeze(exports, after.peak_bytes - 1),
        Err(Error::Bounds)
    ));
    assert_eq!(
        attempts(),
        count,
        "one-short must reject before new backing/factories"
    );
    assert!(
        native.iter().all(|weak| weak.upgrade().is_none()),
        "failed consuming transition must burn owners"
    );

    let (exact, exports) = region(&graph, &sources, &norms, &imports, &input, 2);
    let quote = exact.freeze_quote(&exports).unwrap();
    let count = attempts();
    let frozen = exact.freeze(exports, quote.peak_bytes).unwrap();
    assert_eq!(attempts().0, count.0 + 1);
    assert_eq!(frozen.resources(), quote);
    assert!(
        frozen.output().is_none(),
        "partial region is not complete semantic graph"
    );
}

#[test]
fn foreign_same_source_owner_and_substituted_export_are_rejected_and_burned() {
    let (graph, sources, norms, imports, input) = fixture();
    let (other, other_exports) = region(&graph, &sources, &norms, &imports, &input, 1);
    for change in 0..7 {
        let (compact, mut exports) = region(&graph, &sources, &norms, &imports, &input, 1);
        let weak = Arc::downgrade(compact.owned_native_programs().next().unwrap());
        match change {
            0 => exports[0] = other_exports[0].clone(),
            1 => exports[0].start += 1,
            2 => exports[0].mapping = Mapping::Dense(0),
            3 => exports[0].bounds = Interval::new(0, 0).unwrap(),
            4 => exports[0].slot = None,
            5 => exports[0].import_slot = Some(0),
            6 => exports[0].logical = ViewShape::new(&[1, 129]).unwrap(),
            _ => unreachable!(),
        }
        assert!(compact.freeze_quote(&exports).is_err());
        let count = attempts();
        assert!(compact.freeze(exports, MAX_PROGRAM_BYTES).is_err());
        assert_eq!(attempts(), count);
        assert!(weak.upgrade().is_none());
    }
    let (own, exports) = region(&graph, &sources, &norms, &imports, &input, 1);
    let frozen = own.freeze(exports, MAX_PROGRAM_BYTES).unwrap();
    assert!(matches!(
        frozen.resolve(&other_exports[0]),
        Err(Error::Binding)
    ));
    assert_eq!(other.work_budget(), frozen.work_budget());
}

#[test]
fn lowering_failure_cannot_be_frozen_or_retried() {
    let (mut graph, sources, norms, imports, input) = fixture();
    let start = graph
        .operations
        .iter()
        .position(|op| op.id.ends_with(".gate_proj"))
        .unwrap();
    graph.operations[start + 1].operator = Op::TokenLookup;
    let (mut compact, exports) = region(&graph, &sources, &norms, &imports, &input, 1);
    let weak = Arc::downgrade(compact.owned_native_programs().next().unwrap());
    assert!(compact.lower_next().is_err());
    assert_eq!(compact.lower_next().err(), Some(Error::Retired));
    assert!(matches!(
        compact.freeze_quote(&exports),
        Err(Error::Retired)
    ));
    let count = attempts();
    assert!(matches!(
        compact.freeze(exports, MAX_PROGRAM_BYTES),
        Err(Error::Retired)
    ));
    assert_eq!(attempts(), count);
    assert!(weak.upgrade().is_none());
}

#[test]
fn admitted_midtransition_allocation_failure_drops_original_and_partial_owners() {
    let (graph, sources, norms, imports, input) = fixture();
    for fail_after in [0, 2, 3, 5] {
        let (compact, exports) = region(&graph, &sources, &norms, &imports, &input, 2);
        let native = compact
            .owned_native_programs()
            .map(Arc::downgrade)
            .collect::<Vec<_>>();
        let origin = Arc::downgrade(&compact.origin);
        let before = attempts();
        FREEZE_FAIL_AFTER.with(|count| count.set(Some(fail_after)));
        let result = compact.freeze(exports, MAX_PROGRAM_BYTES);
        FREEZE_FAIL_AFTER.with(|count| count.set(None));
        assert!(matches!(result, Err(Error::Allocation)));
        assert_eq!(attempts().0, before.0 + 1);
        assert_eq!(
            (attempts().1, attempts().2, attempts().3),
            (before.1, before.2, before.3)
        );
        assert!(origin.upgrade().is_none());
        assert!(native.iter().all(|weak| weak.upgrade().is_none()));
    }
}

const HISTORY_OUTSIDE_BYTES: usize = 17_000;

fn rms_semantic(full: &DecoderGraph) -> DecoderGraph {
    let mut semantic = full.clone();
    semantic.operations.retain(|op| op.id == "final_norm");
    assert_eq!(semantic.operations.len(), 1);
    semantic.state_inputs.clear();
    semantic.state_outputs.clear();
    semantic.output = semantic.operations[0].id.clone();
    semantic
}

fn rms_region<'g, 'a>(
    full: &DecoderGraph,
    semantic: &'g DecoderGraph,
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
    imports: &'a Builder,
    input: &View,
) -> (CompactGraph<'g, 'a>, Vec<PublishedView>, usize) {
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
    let mut compact =
        CompactGraph::with_owner_baseline(imports, validated, semantic, 0, HISTORY_OUTSIDE_BYTES)
            .unwrap();
    compact
        .bind_import(&semantic.operations[0].inputs[0], input)
        .unwrap();
    let before_native = compact.construction_peak_bytes();
    let output = compact.lower_next().unwrap();
    (compact, vec![output], before_native)
}

/// Independent single-RMS arm oracle. All earlier arms exclude gamma; both RMS
/// admissions include EXACTLY this original source subowner. Prove native wins
/// before removing gamma, rather than trusting freeze's normalized accumulator.
fn rms_historical_oracle(compact: &CompactGraph<'_, '_>, before_native: usize) -> usize {
    let semantic = compact.frontier.semantic_graph();
    let gamma = &compact.sources[semantic.operations[0].attributes["weight"]
        .as_str()
        .unwrap()];
    let gamma_bytes = size_of::<SourceTensor>()
        + gamma.shape.capacity() * size_of::<usize>()
        + gamma.public_q18.as_ref().unwrap().capacity() * size_of::<i64>();
    let raw = compact.construction_peak_bytes();
    assert!(
        raw > before_native + gamma_bytes,
        "fixture must have a genuinely dominant RMS native-construction arm"
    );
    let sources = super::super::super::super::source_registry::source_metadata_bytes(
        compact.sources,
        compact.norms,
    )
    .unwrap();
    let semantic =
        super::super::super::super::source_registry::semantic_metadata_bytes(semantic).unwrap();
    raw.checked_sub(gamma_bytes)
        .unwrap()
        .checked_add(sources)
        .unwrap()
        .checked_add(semantic)
        .unwrap()
}

fn recipe_digest(compact: &CompactGraph<'_, '_>) -> [u8; 32] {
    let mut hash = Sha256::new();
    compact.hash_expanded_nodes(&mut hash).unwrap();
    hash.finalize().into()
}

#[test]
fn construction_dominant_freeze_adds_actual_semantic_string_spare_capacity() {
    let (full, sources, norms, imports, input) = fixture();
    let mut semantic = rms_semantic(&full);
    let (base, exports, before_native) =
        rms_region(&full, &semantic, &sources, &norms, &imports, &input);
    let before_attempts = attempts();
    let before = base.freeze_quote(&exports).unwrap();
    assert_eq!(
        attempts(),
        before_attempts,
        "quote must not enter a factory"
    );
    let expected_before = rms_historical_oracle(&base, before_native);
    let raw_before = base.construction_peak_bytes();
    let digest_before = recipe_digest(&base);
    let frontier_before = base.frontier_bytes;
    let work_before = base.work_budget();
    let last_word_before = exports[0].word(exports[0].lanes - 1).unwrap();
    assert_eq!(before.historical_construction_bytes, expected_before);
    assert_eq!(before.peak_bytes, expected_before);
    drop(base);
    drop(exports);

    // Change ONLY an attribute String's spare backing, not its value/length or
    // IDs/inputs used by OrderedGraph's frontier bound or native numeric recipes.
    let weight = semantic.operations[0]
        .attributes
        .get("weight")
        .unwrap()
        .as_str()
        .unwrap()
        .to_owned();
    let serde_json::Value::String(name) =
        semantic.operations[0].attributes.get_mut("weight").unwrap()
    else {
        unreachable!()
    };
    let old_capacity = name.capacity();
    name.reserve_exact(old_capacity - name.len() + 64);
    let extra = name.capacity() - old_capacity;
    assert!(extra >= 64);
    assert_eq!(name.as_str(), weight.as_str());

    let (grown, exports, before_native) =
        rms_region(&full, &semantic, &sources, &norms, &imports, &input);
    let count = attempts();
    let after = grown.freeze_quote(&exports).unwrap();
    assert_eq!(attempts(), count);
    assert_eq!(grown.construction_peak_bytes(), raw_before);
    assert_eq!(grown.frontier_bytes, frontier_before);
    assert_eq!(recipe_digest(&grown), digest_before);
    assert_eq!(grown.work_budget(), work_before);
    assert_eq!(
        exports[0].word(exports[0].lanes - 1).unwrap(),
        last_word_before
    );
    assert_eq!(after.native_owner_bytes, before.native_owner_bytes);
    assert_eq!(after.recipe_capacity, before.recipe_capacity);
    assert_eq!(after.new_backing_bytes, before.new_backing_bytes);
    assert_eq!(after.borrowed_owner_bytes, before.borrowed_owner_bytes);
    assert_eq!(
        after.borrowed_semantic_bytes - before.borrowed_semantic_bytes,
        extra
    );
    assert!(
        expected_before > before.transition_bytes + extra,
        "historical native arm must dominate beyond the entire spare-capacity delta"
    );
    assert!(after.historical_construction_bytes > after.transition_bytes);
    assert_eq!(
        after.historical_construction_bytes,
        rms_historical_oracle(&grown, before_native)
    );
    assert_eq!(
        after.peak_bytes,
        before.peak_bytes + extra,
        "present transition max must not hide omitted historical borrowed backing"
    );
    after.admit(after.peak_bytes).unwrap();
    assert_eq!(after.admit(after.peak_bytes - 1), Err(Error::Bounds));
    let weak = Arc::downgrade(grown.owned_native_programs().next().unwrap());
    let count = attempts();
    assert!(matches!(
        grown.freeze(exports, after.peak_bytes - 1),
        Err(Error::Bounds)
    ));
    assert_eq!(attempts(), count);
    assert!(weak.upgrade().is_none());

    let (exact, exports, _) = rms_region(&full, &semantic, &sources, &norms, &imports, &input);
    let count = attempts();
    let frozen = exact.freeze(exports, after.peak_bytes).unwrap();
    let after_freeze = attempts();
    assert_eq!(
        (after_freeze.1, after_freeze.2, after_freeze.3),
        (count.1, count.2, count.3)
    );
    assert_eq!(frozen.resources(), after);
}

#[test]
fn historical_normalization_preserves_each_arm_before_taking_maximum() {
    let (graph, sources, norms, imports, input) = fixture();
    let (compact, exports) = region(&graph, &sources, &norms, &imports, &input, 0);
    // Resource-recorder arithmetic control only: no fabricated numeric recipe,
    // native execution or request-completion claim. A lower raw arm can become
    // the greater source-excluded arm; normalizing only the winner is unsound.
    let initial = compact.construction_peak_bytes();
    let first = initial + 100_000;
    let lower_raw = initial + 95_000;
    compact.admit_total_with_source(first, 10_000).unwrap();
    compact.admit_total(lower_raw).unwrap();
    assert_eq!(compact.construction_peak_bytes(), first);
    assert_eq!(compact.construction_peak_without_source.get(), lower_raw);
    let source_bytes =
        super::super::super::super::source_registry::source_metadata_bytes(&sources, &norms)
            .unwrap();
    let semantic_bytes =
        super::super::super::super::source_registry::semantic_metadata_bytes(&graph).unwrap();
    let quote = compact.freeze_quote(&exports).unwrap();
    assert_eq!(
        quote.historical_construction_bytes,
        lower_raw + source_bytes + semantic_bytes
    );
}

#[test]
fn historical_borrowed_overlap_overflow_rejects_before_transition_allocations() {
    let (graph, sources, norms, imports, input) = fixture();
    let source_bytes =
        super::super::super::super::source_registry::source_metadata_bytes(&sources, &norms)
            .unwrap();
    for historical in [usize::MAX, usize::MAX - source_bytes] {
        let (compact, exports) = region(&graph, &sources, &norms, &imports, &input, 1);
        let weak = Arc::downgrade(compact.owned_native_programs().next().unwrap());
        // Overflow-only corruption of a diagnostic counter, not a forged native
        // construction or completion capability. Both additions must be checked.
        compact.construction_peak_without_source.set(historical);
        let count = attempts();
        assert!(matches!(compact.freeze_quote(&exports), Err(Error::Bounds)));
        assert!(matches!(
            compact.freeze(exports, MAX_PROGRAM_BYTES),
            Err(Error::Bounds)
        ));
        assert_eq!(attempts(), count);
        assert!(weak.upgrade().is_none());
    }
}
