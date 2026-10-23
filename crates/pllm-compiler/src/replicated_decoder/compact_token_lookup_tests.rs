//! Original public PerToken operator qualification only, not decoder execution.
use super::*;

fn fixture(vocabulary: usize, tokens: usize) -> (DecoderGraph, BTreeMap<String, SourceTensor>) {
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type": "llama", "hidden_size": 2, "intermediate_size": 2,
        "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
        "head_dim": 2, "vocab_size": vocabulary, "tie_word_embeddings": false,
        "hidden_act": "silu", "attention_bias": false, "max_position_embeddings": 64,
        "rope_theta": 10000, "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
        "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
    }))
    .unwrap();
    let graph = lower_with_weights(&config, tokens, 1, WeightMode::Public)
        .unwrap()
        .prefill;
    let sources = Program::source_shapes_with_weights(&config, tokens, 1, WeightMode::Public)
        .unwrap()
        .into_iter()
        .map(|(key, shape)| {
            let public_q18 = key
                .contains("norm")
                .then(|| vec![ONE; shape.iter().product()]);
            let mut hash = Sha256::new();
            for value in std::iter::repeat_n(
                if public_q18.is_some() { ONE } else { ONE / 64 },
                shape.iter().product(),
            ) {
                hash.update(value.to_le_bytes());
            }
            let public = public_q18.is_some();
            (
                key,
                SourceTensor {
                    shape,
                    lower: if public { ONE } else { -ONE / 64 },
                    upper: if public { ONE } else { ONE / 64 },
                    public_q18,
                    digest: hash.finalize().into(),
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
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
    execution: ExecutionPolicy,
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
        execution,
        norms,
    )
    .unwrap()
}

fn imports(
    sources: &BTreeMap<String, SourceTensor>,
    vocabulary: usize,
    tokens: usize,
) -> (Builder, View) {
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
    let owner = builder
        .push(
            Code::Input("input.tokens".into()),
            tokens,
            Interval::new(0, vocabulary as i64 - 1).unwrap(),
        )
        .unwrap();
    let input = builder.view(owner, vec![1, tokens]);
    (builder, input)
}

fn lookup_region(full: &DecoderGraph, repeat: bool) -> DecoderGraph {
    let mut region = full.clone();
    let lookup = full
        .operations
        .iter()
        .find(|op| op.operator == Op::TokenLookup)
        .unwrap()
        .clone();
    region.operations = vec![lookup.clone()];
    region.output = lookup.id.clone();
    if repeat {
        let mut second = lookup;
        second.id.push_str(".again");
        region.output = second.id.clone();
        region.operations.push(second);
    }
    region
}

// Original graph.rs public PerToken arm through its original Builder owners.
// No numerical oracle, independent matrix equation or rescale substitute.
fn expanded_lookup(
    builder: &mut Builder,
    input: &View,
    weight: usize,
    vocabulary: usize,
    width: usize,
) -> View {
    let table_bound = builder.nodes[weight].bounds;
    let table = SourceView {
        owner: weight,
        shape: vec![vocabulary, width],
    };
    let mut words = Words::new();
    for token in input.words.iter() {
        let original = builder.nodes[token.0].bounds;
        let bounds = Interval::new(
            original
                .lower()
                .checked_sub((vocabulary - 1) as i64)
                .unwrap(),
            original.upper(),
        )
        .unwrap();
        let equal = builder
            .tensor_loop(
                LoopInput::TokenOffsets {
                    owner: token.0,
                    index: token.1,
                },
                vocabulary,
                |n| tensor::Program::equal_zero(n, bounds),
            )
            .unwrap();
        let selectors = builder.view(equal, vec![1, vocabulary]);
        let row = builder
            .matrix(&selectors, MatrixView::source(&table, true).unwrap(), false)
            .unwrap();
        for (owner, _) in row.words.iter() {
            builder.nodes[owner].bounds = table_bound;
        }
        words.extend(row.words);
    }
    View {
        words,
        shape: vec![1, input.words.len(), width],
    }
}

fn hash_nodes(nodes: &[Node]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for node in nodes {
        hash_node_header(&mut hash, node.lanes, node.bounds);
        match &node.code {
            Code::TensorLoop {
                input,
                program,
                tail,
            } => {
                hash_tensor_loop(&mut hash, *input, program, tail.as_ref());
            }
            Code::Gather(words) => hash_gather(&mut hash, words.iter()),
            Code::PublicValues(values) => {
                hash.update([7]);
                for value in values {
                    hash.update(value.to_le_bytes());
                }
            }
            Code::Affine {
                left,
                right,
                coefficients,
            } => {
                hash_affine(&mut hash, *left, *right, *coefficients);
            }
            Code::Product {
                left,
                right,
                shape,
                rescale,
                norm,
            } => {
                hash_product(&mut hash, *left, *right, *shape, *rescale, norm.as_deref());
            }
            Code::Tensor { input, program } => hash_tensor(&mut hash, *input, program),
            Code::PublicProduct {
                input,
                weight,
                rows,
                inner,
                columns,
                transposed,
            } => {
                hash_public_product(
                    &mut hash,
                    *input,
                    *weight,
                    *rows,
                    *inner,
                    *columns,
                    *transposed,
                );
            }
            _ => panic!("not an original public PerToken node"),
        }
    }
    hash.finalize().into()
}

#[test]
fn public_per_token_matches_original_nodes_ids_bounds_work_and_sources() {
    for vocabulary in [1, 127, 128, 129, 257] {
        for tokens in [1, 3] {
            let (full, sources) = fixture(vocabulary, tokens);
            let region = lookup_region(&full, false);
            let norms = SourceNorms::new();
            let (borrowed, input) = imports(&sources, vocabulary, tokens);
            let original = (
                borrowed.nodes.len(),
                borrowed.allocated,
                borrowed.tensors.len(),
            );
            let mut compact = CompactGraph::new(
                &borrowed,
                validated(&full, &sources, &norms, ExecutionPolicy::default()),
                &region,
                0,
            )
            .unwrap();
            compact.bind_import("input.tokens", &input).unwrap();
            let attempts = LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get());
            let output = compact.lower_next().unwrap();
            let expected_kernels = 1 + usize::from(vocabulary > 128 && vocabulary % 128 != 0);
            assert_eq!(
                LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()) - attempts,
                expected_kernels
            );
            let census = compact.token_lookup_census().next().unwrap();
            let key = region.operations[0].attributes["weight"].as_str().unwrap();
            assert!(std::ptr::eq(census.source, &sources[key]));
            assert_eq!(census.descriptor, 0);
            assert_eq!(census.first, borrowed.nodes.len());
            assert_eq!(census.geometry, [tokens, vocabulary, 2]);
            assert_eq!(
                census.source_binding,
                source_binding_with_norms(&sources, &norms)
            );
            assert_eq!(compact.owned_native_programs().count(), expected_kernels);
            assert_eq!(compact.owned_prepared_programs().count(), expected_kernels);
            assert!(compact.native_row_census().next().is_none());
            assert!(compact.complete_row(&output, 0).unwrap().is_none());
            assert!(compact.row_norm(&output).unwrap().is_none());
            assert!(compact.attention_energy(&output, 0, 2).unwrap().is_none());
            assert!(compact.ops[0].prepared.is_none());

            let (mut expanded, reference) = imports(&sources, vocabulary, tokens);
            let first = expanded.nodes.len();
            let expected = expanded_lookup(
                &mut expanded,
                &reference,
                census.source_owner,
                vocabulary,
                2,
            );
            assert_eq!(compact.next, expanded.nodes.len());
            assert_eq!(output.logical.dimensions(), expected.shape);
            assert!(output.words().eq(expected.words.iter()));
            assert_eq!(output.bounds, expanded.bounds(&expected.words).unwrap());
            assert_eq!(
                compact.work_budget(),
                schedule_work(&expanded.nodes[first..]).unwrap()
            );
            let mut hash = Sha256::new();
            compact.hash_expanded_nodes(&mut hash).unwrap();
            assert_eq!(
                <[u8; 32]>::from(hash.finalize()),
                hash_nodes(&expanded.nodes[first..])
            );
            let descriptor = compact.descriptor_census().next().unwrap();
            assert!(
                matches!(descriptor.kind, DescriptorKind::TokenLookup { geometry, .. }
                if geometry == [tokens, vocabulary, 2])
            );
            assert!(
                matches!(descriptor.output.layout, PhysicalLayout::TensorRows {
                first: owner, row_width: 2, node_stride: 3, output_node: 2,
            } if owner == first)
            );
            assert_eq!(
                descriptor.virtual_end - descriptor.virtual_start,
                tokens * 3
            );
            assert_eq!(
                original,
                (
                    borrowed.nodes.len(),
                    borrowed.allocated,
                    borrowed.tensors.len()
                )
            );
        }
    }
}

#[test]
fn repeated_lookup_keeps_original_full_tail_arcs_and_charges_every_token() {
    let (full, sources) = fixture(129, 3);
    let region = lookup_region(&full, true);
    let norms = SourceNorms::new();
    let (borrowed, input) = imports(&sources, 129, 3);
    let mut compact = CompactGraph::new(
        &borrowed,
        validated(&full, &sources, &norms, ExecutionPolicy::default()),
        &region,
        0,
    )
    .unwrap();
    compact.bind_import("input.tokens", &input).unwrap();
    let first = compact.lower_next().unwrap();
    let work = compact.work_budget();
    let mut census = compact.token_lookup_census();
    let first_census = census.next().unwrap();
    let full_arc = first_census.program.clone();
    let tail_arc = first_census.tail.unwrap().clone();
    drop(census);
    let attempts = LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get());
    let second = compact.lower_next().unwrap();
    assert_eq!(attempts, LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()));
    let second_census = compact.token_lookup_census().nth(1).unwrap();
    assert!(Arc::ptr_eq(&full_arc, second_census.program));
    assert!(Arc::ptr_eq(&tail_arc, second_census.tail.unwrap()));
    assert_eq!(compact.owned_native_programs().count(), 2);
    assert_eq!(compact.work_budget(), work.checked_mul(2).unwrap());
    assert_eq!(second.word(0).unwrap().0 - first.word(0).unwrap().0, 9);
    let native_bytes = compact
        .owned_native_programs()
        .map(|program| {
            program.resources().unwrap().immutable_template_bytes + 2 * std::mem::size_of::<usize>()
        })
        .sum::<usize>();
    let exports = vec![first.clone(), second.clone()];
    let quote = compact.freeze_quote(&exports).unwrap();
    assert_eq!(quote.native_owner_bytes, native_bytes);
    let frozen = compact.freeze(exports, MAX_PROGRAM_BYTES).unwrap();
    assert_eq!(frozen.resources(), quote);
    assert_eq!(frozen.word(&first, 5).unwrap(), first.word(5).unwrap());
    assert_eq!(frozen.word(&second, 5).unwrap(), second.word(5).unwrap());
    assert!(std::ptr::eq(frozen.borrowed_source_records(), &sources));
}

#[test]
fn lookup_construction_limit_rejects_before_any_native_kernel_or_publication() {
    let (full, sources) = fixture(129, 3);
    let region = lookup_region(&full, false);
    let norms = SourceNorms::new();
    let (borrowed, input) = imports(&sources, 129, 3);
    let make = || {
        CompactGraph::new(
            &borrowed,
            validated(&full, &sources, &norms, ExecutionPolicy::default()),
            &region,
            0,
        )
        .unwrap()
    };
    let mut priced = make();
    priced.bind_import("input.tokens", &input).unwrap();
    priced.lower_next().unwrap();
    let peak = priced.construction_peak_bytes();
    let mut rejected = make();
    rejected.bind_import("input.tokens", &input).unwrap();
    rejected.limit = peak - 1;
    let first = rejected.next;
    let attempts = LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get());
    assert!(matches!(rejected.lower_next(), Err(Error::Bounds)));
    assert_eq!(attempts, LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()));
    assert!(rejected.ops.is_empty());
    assert_eq!(rejected.next, first);
    assert_eq!(rejected.work_budget(), WorkBudget::default());
    assert!(matches!(rejected.lower_next(), Err(Error::Retired)));
    let mut exact = make();
    exact.bind_import("input.tokens", &input).unwrap();
    exact.limit = peak;
    exact.lower_next().unwrap();
    assert_eq!(exact.construction_peak_bytes(), peak);
}

#[test]
fn lookup_rejects_alternative_policies_shapes_and_original_raw_matrix_overflow() {
    let (full, sources) = fixture(129, 1);
    let region = lookup_region(&full, false);
    let norms = SourceNorms::new();
    for lookup in [LookupPolicy::Tiled, LookupPolicy::Trie] {
        let execution = ExecutionPolicy {
            lookup,
            ..ExecutionPolicy::default()
        };
        let (mut borrowed, input) = imports(&sources, 129, 1);
        borrowed.lookup = lookup;
        let mut compact = CompactGraph::new(
            &borrowed,
            validated(&full, &sources, &norms, execution),
            &region,
            0,
        )
        .unwrap();
        compact.bind_import("input.tokens", &input).unwrap();
        let attempts = LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get());
        assert!(matches!(compact.lower_next(), Err(Error::Binding)));
        assert_eq!(attempts, LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()));
    }
    let mut malformed = region.clone();
    malformed.operations[0].output_shape[2] = 3;
    let (borrowed, input) = imports(&sources, 129, 1);
    let mut compact = CompactGraph::new(
        &borrowed,
        validated(&full, &sources, &norms, ExecutionPolicy::default()),
        &malformed,
        0,
    )
    .unwrap();
    compact.bind_import("input.tokens", &input).unwrap();
    let attempts = LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get());
    assert!(matches!(compact.lower_next(), Err(Error::Shape)));
    assert_eq!(attempts, LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()));

    let key = region.operations[0].attributes["weight"].as_str().unwrap();
    let mut sources = sources.clone();
    sources.get_mut(key).unwrap().lower = -(1_i64 << 60);
    sources.get_mut(key).unwrap().upper = 1_i64 << 60;
    let (borrowed, input) = imports(&sources, 129, 1);
    let mut compact = CompactGraph::new(
        &borrowed,
        validated(&full, &sources, &norms, ExecutionPolicy::default()),
        &region,
        0,
    )
    .unwrap();
    compact.bind_import("input.tokens", &input).unwrap();
    assert!(matches!(compact.lower_next(), Err(Error::Bounds)));
    assert!(compact.ops.is_empty());
    assert_eq!(compact.work_budget(), WorkBudget::default());
}

#[test]
fn lookup_rows_feed_original_last_token_and_projection_without_lane_arrays() {
    let (full, sources) = fixture(129, 3);
    let mut region = lookup_region(&full, false);
    let mut last = full
        .operations
        .iter()
        .find(|op| op.operator == Op::LastToken)
        .unwrap()
        .clone();
    last.inputs = vec![region.output.clone()];
    let mut head = full
        .operations
        .iter()
        .find(|op| op.operator == Op::OutputHead)
        .unwrap()
        .clone();
    head.inputs = vec![last.id.clone()];
    region.output = head.id.clone();
    region.operations.extend([last, head]);
    let norms = SourceNorms::new();
    let (borrowed, input) = imports(&sources, 129, 3);
    let mut compact = CompactGraph::new(
        &borrowed,
        validated(&full, &sources, &norms, ExecutionPolicy::default()),
        &region,
        0,
    )
    .unwrap();
    compact.bind_import("input.tokens", &input).unwrap();
    let embedding = compact.lower_next().unwrap();
    let last = compact.lower_next().unwrap();
    assert_eq!(last.mapping, embedding.mapping);
    assert_eq!(last.start, 4);
    assert_eq!(last.word(0).unwrap(), embedding.word(4).unwrap());
    assert_eq!(last.word(1).unwrap(), embedding.word(5).unwrap());
    let output = compact.lower_next().unwrap();
    let (mut expanded, reference) = imports(&sources, 129, 3);
    let first = expanded.nodes.len();
    let lookup_owner = compact.token_lookup_census().next().unwrap().source_owner;
    let embedding = expanded_lookup(&mut expanded, &reference, lookup_owner, 129, 2);
    let last = View {
        words: embedding.words.slice(4..6),
        shape: vec![1, 2],
    };
    let key = region.operations[2].attributes["weight"].as_str().unwrap();
    let weight = sources
        .iter()
        .filter(|(_, source)| source.public_q18.is_none())
        .position(|(name, _)| name == key)
        .unwrap();
    let source = SourceView {
        owner: weight,
        shape: vec![129, 2],
    };
    let expected = expanded
        .matrix(&last, MatrixView::source(&source, false).unwrap(), true)
        .unwrap();
    assert!(output.words().eq(expected.words.iter()));
    assert_eq!(
        compact.work_budget(),
        schedule_work(&expanded.nodes[first..]).unwrap()
    );
    let mut hash = Sha256::new();
    compact.hash_expanded_nodes(&mut hash).unwrap();
    assert_eq!(
        <[u8; 32]>::from(hash.finalize()),
        hash_nodes(&expanded.nodes[first..])
    );
}

#[test]
fn lookup_census_prices_real_rows_and_fails_closed_for_missing_body_mapping() {
    use compact_admission::{RequestOwners, RequestQuote};
    let (full, sources) = fixture(129, 3);
    let region = lookup_region(&full, false);
    let norms = SourceNorms::new();
    let (borrowed, input) = imports(&sources, 129, 3);
    let mut compact = CompactGraph::new(
        &borrowed,
        validated(&full, &sources, &norms, ExecutionPolicy::default()),
        &region,
        0,
    )
    .unwrap();
    compact.bind_import("input.tokens", &input).unwrap();
    compact.lower_next().unwrap();
    let values = Vec::new();
    let quote = RequestQuote::new(
        &compact,
        RequestOwners {
            borrowed_external_bytes: 0,
            owned_source_bytes: 0,
            other_owned_metadata_bytes: 0,
            other_stored_descriptors: 0,
            other_work: WorkBudget::default(),
            retained_values: &values,
            external_frame_bytes: 0,
            caller_control_bytes: 0,
            prior_request_peak_bytes: 0,
        },
    )
    .unwrap();
    let census = quote.owner_census();
    assert_eq!(census.output_slots, 3);
    assert_eq!(census.retained_output_bytes, 3 * 2 * 16);
    assert_eq!(census.body_descriptors, None);
    assert_eq!(census.remaining_operations, 0);
    assert_eq!(census.work, compact.work_budget());
    assert_eq!(
        census.prepared_program_bytes,
        compact
            .owned_native_programs()
            .map(|program| {
                program.resources().unwrap().immutable_template_bytes
                    + 2 * std::mem::size_of::<usize>()
            })
            .sum::<usize>()
    );
    assert!(matches!(
        quote.preflight_body_inputs(&[], &[]),
        Err(Error::Binding)
    ));
}

#[test]
fn feedback_tokens_keep_protected_argmax_owner_and_original_next_lookup_recipe() {
    let (full, sources) = fixture(3, 1);
    let mut region = lookup_region(&full, false);
    let mut selection = full
        .operations
        .iter()
        .find(|op| op.operator == Op::GreedyTokenSelection)
        .unwrap()
        .clone();
    selection.inputs = vec!["scores".into()];
    let mut feedback = full
        .operations
        .iter()
        .find(|op| op.operator == Op::TokenFeedback)
        .unwrap()
        .clone();
    feedback.inputs = vec![selection.id.clone()];
    let mut lookup = region.operations.remove(0);
    lookup.inputs = vec![feedback.id.clone()];
    region.operations = vec![selection, feedback, lookup];
    let norms = SourceNorms::new();
    let (mut borrowed, _) = imports(&sources, 3, 3);
    let owner = borrowed.nodes.len() - 1;
    borrowed.nodes[owner].code = Code::Input("scores".into());
    borrowed.nodes[owner].bounds = Interval::new(-ONE, ONE).unwrap();
    let scores = borrowed.view(owner, vec![1, 3]);
    let mut compact = CompactGraph::new(
        &borrowed,
        validated(&full, &sources, &norms, ExecutionPolicy::default()),
        &region,
        0,
    )
    .unwrap();
    compact.bind_import("scores", &scores).unwrap();
    let selected = compact.lower_next().unwrap();
    let feedback = compact.lower_next().unwrap();
    assert_eq!(feedback.word(0).unwrap(), selected.word(0).unwrap());
    assert_eq!(feedback.logical.dimensions(), [1]); // not semantic [1, 1]
    let output = compact.lower_next().unwrap();
    let (mut expanded, _) = imports(&sources, 3, 3);
    expanded.nodes[owner].code = Code::Input("scores".into());
    expanded.nodes[owner].bounds = Interval::new(-ONE, ONE).unwrap();
    let scores = expanded.view(owner, vec![1, 3]);
    let first = expanded.nodes.len();
    let selected = expanded.argmax_tree(&scores).unwrap();
    assert_eq!(selected.words.at(0), feedback.word(0).unwrap());
    let weight = compact.token_lookup_census().next().unwrap().source_owner;
    let expected = expanded_lookup(&mut expanded, &selected, weight, 3, 2);
    assert!(output.words().eq(expected.words.iter()));
    assert_eq!(
        compact.work_budget(),
        schedule_work(&expanded.nodes[first..]).unwrap()
    );
    let mut hash = Sha256::new();
    compact.hash_expanded_nodes(&mut hash).unwrap();
    assert_eq!(
        <[u8; 32]>::from(hash.finalize()),
        hash_nodes(&expanded.nodes[first..])
    );
}

#[test]
fn interval_only_import_cannot_supply_original_token_producer_lineage() {
    let (full, sources) = fixture(129, 1);
    let mut region = lookup_region(&full, false);
    region.operations[0].inputs = vec!["token.copy".into()];
    let norms = SourceNorms::new();
    let (mut borrowed, _) = imports(&sources, 129, 1);
    let owner = borrowed.nodes.len() - 1;
    borrowed.nodes[owner].code = Code::Input("token.copy".into());
    let input = borrowed.view(owner, vec![1, 1]);
    let mut compact = CompactGraph::new(
        &borrowed,
        validated(&full, &sources, &norms, ExecutionPolicy::default()),
        &region,
        0,
    )
    .unwrap();
    compact.bind_import("token.copy", &input).unwrap();
    let attempts = LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get());
    assert!(matches!(compact.lower_next(), Err(Error::Binding)));
    assert_eq!(attempts, LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()));
    assert!(compact.ops.is_empty());
}
