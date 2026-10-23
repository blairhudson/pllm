//! Numeric-sink qualification only. No complete-request admission/runtime claim.
use super::*;

fn graph(lanes: usize) -> DecoderGraph {
    graph_with_rows(lanes, 1)
}

fn graph_with_rows(lanes: usize, rows: usize) -> DecoderGraph {
    lower_with_weights(&config(lanes), rows, 1, WeightMode::Public)
        .unwrap()
        .prefill
}

fn config(lanes: usize) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "model_type": "llama", "hidden_size": 2, "intermediate_size": lanes,
        "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
        "head_dim": 2, "vocab_size": 3, "tie_word_embeddings": false,
        "hidden_act": "silu", "attention_bias": false,
        "max_position_embeddings": 64, "rope_theta": 10000,
        "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
        "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
    }))
    .unwrap()
}

fn inputs(lanes: usize, offset: usize) -> (Builder, View, View) {
    let mut builder = Builder::default();
    let left = builder
        .push(
            Code::Input("left".into()),
            lanes + offset,
            Interval::new(-ONE - 1, ONE / 2 + 1).unwrap(),
        )
        .unwrap();
    let right = builder
        .push(
            Code::Input("right".into()),
            lanes + offset,
            Interval::new(-ONE / 3, ONE / 5).unwrap(),
        )
        .unwrap();
    let view = |owner| View {
        words: (offset..offset + lanes).map(|i| (owner, i)).collect(),
        shape: vec![1, 1, lanes],
    };
    (builder, view(left), view(right))
}

fn hash_expanded_binary_nodes(nodes: &[Node]) -> [u8; 32] {
    let mut h = Sha256::new();
    for node in nodes {
        hash_node_header(&mut h, node.lanes, node.bounds);
        match &node.code {
            Code::Gather(words) => hash_gather(&mut h, words.iter()),
            Code::Product {
                left,
                right,
                shape,
                rescale,
                norm,
            } => hash_product(&mut h, *left, *right, *shape, *rescale, norm.as_deref()),
            Code::PublicProduct {
                input,
                weight,
                rows,
                inner,
                columns,
                transposed,
            } => hash_public_product(
                &mut h,
                *input,
                *weight,
                *rows,
                *inner,
                *columns,
                *transposed,
            ),
            Code::TensorLoop {
                input,
                program,
                tail,
            } => hash_tensor_loop(&mut h, *input, program, tail.as_ref()),
            Code::Tensor { input, program } => hash_tensor(&mut h, *input, program),
            Code::PublicValues(values) => {
                h.update([7]);
                for value in values {
                    h.update(value.to_le_bytes());
                }
            }
            Code::Affine {
                left,
                right,
                coefficients,
            } => hash_affine(&mut h, *left, *right, *coefficients),
            _ => panic!("unexpected original binary node"),
        }
    }
    h.finalize().into()
}

#[test]
fn compact_multiply_reuses_semantic_frontier_and_exact_original_numeric_sink() {
    for lanes in [1, 128, 129, 257, 1025] {
        for offset in [0, 3] {
            let graph = graph(lanes);
            let op = graph
                .operations
                .iter()
                .find(|op| op.operator == Op::Multiply)
                .unwrap();
            let (borrowed, a, b) = inputs(lanes, offset);
            let mut traversal = super::super::graph::OrderedGraph::new(&graph);
            traversal.bind(op.inputs[0].clone(), a);
            traversal.bind(op.inputs[1].clone(), b);
            let before = (
                borrowed.nodes.len(),
                borrowed.allocated,
                borrowed.tensors.len(),
            );
            let compact = traversal.compact_binary_candidate(&borrowed, op).unwrap();
            assert_eq!(compact.stored_descriptors(), 1);
            assert_eq!(compact.expanded_nodes(), 3 * lanes.div_ceil(128));
            assert_eq!(
                before,
                (
                    borrowed.nodes.len(),
                    borrowed.allocated,
                    borrowed.tensors.len()
                )
            );

            let (mut expanded, a, b) = inputs(lanes, offset);
            let first = expanded.nodes.len();
            let output = expanded.binary(&a.words, &b.words, true).unwrap();
            assert_eq!(expanded.nodes.len() - first, compact.expanded_nodes());
            assert_eq!(
                schedule_work(&expanded.nodes[first..]).unwrap(),
                compact.work_budget()
            );
            for (index, word) in output.iter().enumerate() {
                assert_eq!(compact.output_word(index).unwrap(), word);
            }
            assert!(compact.output_word(lanes).is_err());
            assert_eq!(
                compact.summary.bounds.unwrap(),
                expanded.bounds(&output).unwrap()
            );
            let mut h = Sha256::new();
            compact.hash_expanded_nodes(&mut h);
            assert_eq!(
                <[u8; 32]>::from(h.finalize()),
                hash_expanded_binary_nodes(&expanded.nodes[first..])
            );
        }
    }
}

#[test]
fn large_candidate_retains_one_inline_descriptor_not_virtual_nodes_or_lane_arrays() {
    let graph = graph_with_rows(8192, 32);
    let op = graph
        .operations
        .iter()
        .find(|op| op.operator == Op::Multiply)
        .unwrap();
    let lanes = 32 * 8192;
    let (builder, a, b) = inputs(lanes, 0);
    let original = (builder.nodes.capacity(), builder.allocated);
    let candidate = CompactBinary::lower_unadmitted(&builder, op, &a, &b).unwrap();
    assert_eq!(candidate.stored_descriptors(), 1);
    assert_eq!(candidate.expanded_nodes(), 6144);
    assert_eq!(candidate.output_word(lanes - 1).unwrap(), (6145, 127));
    assert_eq!(original, (builder.nodes.capacity(), builder.allocated));
    assert_eq!(builder.nodes.len(), 2);
    assert!(!std::mem::needs_drop::<CompactBinary<'_>>());
    assert_eq!(
        std::mem::size_of_val(&candidate),
        std::mem::size_of::<CompactBinary<'_>>()
    );
    // This is NOT a whole-owner peak quote or permission to run the candidate.
    // In particular original import/value/slot owners remain borrowed and live.
}

#[test]
fn candidate_can_borrow_real_public_gate_up_silu_owners_without_new_kernels() {
    let lanes = 129;
    let graph = graph(lanes);
    let op = graph
        .operations
        .iter()
        .find(|op| op.operator == Op::Multiply)
        .unwrap();
    let mut builder = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let input = builder
        .push(
            Code::Input("hidden".into()),
            2,
            Interval::new(-ONE / 4, ONE / 4).unwrap(),
        )
        .unwrap();
    let input = builder.view(input, vec![1, 1, 2]);
    let mut project = |key: &str| {
        let weight = builder
            .push(
                Code::PublicInput {
                    key: key.into(),
                    digest: [7; 32],
                    row_l1: Some((ONE / 32) as u64),
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
                2 * lanes,
                Interval::new(-ONE / 64, ONE / 64).unwrap(),
            )
            .unwrap();
        let source = SourceView {
            owner: weight,
            shape: vec![lanes, 2],
        };
        builder
            .matrix(&input, MatrixView::source(&source, false).unwrap(), true)
            .unwrap()
    };
    let gate = project("gate_proj.weight");
    let up = project("up_proj.weight");
    let input = builder.gather(&gate.words).unwrap();
    let bounds = builder.nodes[input].bounds;
    let silu = builder
        .tensor_loop(LoopInput::Slice(input), lanes, |n| {
            tensor::Program::activation_guarded(tensor::Activation::Silu, n, bounds)
        })
        .unwrap();
    let gate = builder.view(silu, vec![1, 1, lanes]);
    let original_templates: Vec<_> = builder.tensors.values().map(Arc::as_ptr).collect();
    let template_references: Vec<_> = builder.tensors.values().map(Arc::strong_count).collect();
    let native_attempts = super::super::super::prepared_tensor_map::native_attempts();
    let candidate = CompactBinary::lower_unadmitted(&builder, op, &gate, &up).unwrap();
    assert_eq!(candidate.expanded_nodes(), 6);
    assert_eq!(candidate.left.owner, silu);
    assert_eq!(candidate.right.owner, up.words.at(0).0);
    assert_eq!(
        original_templates,
        builder
            .tensors
            .values()
            .map(Arc::as_ptr)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        template_references,
        builder
            .tensors
            .values()
            .map(Arc::strong_count)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        native_attempts,
        super::super::super::prepared_tensor_map::native_attempts()
    );
    assert!(matches!(candidate.left.node.code, Code::TensorLoop { .. }));
}

#[test]
fn candidate_rejects_unsupported_ops_fragmentation_shapes_and_file_owners() {
    let graph = graph(129);
    let op = graph
        .operations
        .iter()
        .find(|op| op.operator == Op::Multiply)
        .unwrap();
    let unsupported = graph
        .operations
        .iter()
        .find(|op| op.operator == Op::Silu)
        .unwrap();
    let (mut builder, a, b) = inputs(129, 0);
    assert!(CompactBinary::lower_unadmitted(&builder, unsupported, &a, &b).is_err());
    let mut fragmented = a.clone();
    fragmented.words = (0..129).map(|i| (i % 2, i)).collect();
    assert!(CompactBinary::lower_unadmitted(&builder, op, &fragmented, &b).is_err());
    let mut permuted = a.clone();
    permuted.words = (0..129).rev().map(|i| (0, i)).collect();
    assert!(CompactBinary::lower_unadmitted(&builder, op, &permuted, &b).is_err());
    let mut short = a.clone();
    short.words = short.words.slice(..128);
    assert!(CompactBinary::lower_unadmitted(&builder, op, &short, &b).is_err());
    let file = builder
        .push(
            Code::PublicInput {
                key: "weight".into(),
                digest: [9; 32],
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
            129,
            Interval::new(-1, 1).unwrap(),
        )
        .unwrap();
    let file = builder.view(file, vec![1, 1, 129]);
    assert!(CompactBinary::lower_unadmitted(&builder, op, &file, &b).is_err());
}

#[test]
fn compact_sink_keeps_original_quarter_ring_rejection_and_floor_envelope() {
    let graph = graph(129);
    let op = graph
        .operations
        .iter()
        .find(|op| op.operator == Op::Multiply)
        .unwrap();
    let (mut borrowed, a, b) = inputs(129, 0);
    let unsafe_bounds = Interval::new(-(1 << 32), 1 << 32).unwrap();
    borrowed.nodes[0].bounds = unsafe_bounds;
    borrowed.nodes[1].bounds = unsafe_bounds;
    assert!(CompactBinary::lower_unadmitted(&borrowed, op, &a, &b).is_err());
    assert_eq!(borrowed.nodes.len(), 2);
    assert!(borrowed.binary(&a.words, &b.words, true).is_err());
    // Preserve original fail order: both original gathers precede arithmetic.
    assert_eq!(borrowed.nodes.len(), 4);

    let bounds = InputBounds {
        left: Interval::new(-ONE - 1, ONE + 1).unwrap(),
        right: Interval::new(-1, 1).unwrap(),
    };
    let range = product_bounds(bounds, 1, false, true, None).unwrap();
    assert_eq!(range, bounds.dot_output(1).unwrap());
    for a in [-ONE - 1, -ONE, -1, 0, 1, ONE, ONE + 1] {
        for b in [-1, 0, 1] {
            let floor = (a * b).div_euclid(ONE);
            assert!(range.lower() <= floor && floor <= range.upper());
        }
    }
}

#[test]
fn extracted_product_contract_preserves_original_certificate_checks() {
    use puma::multiply::norm::RowEnergyProof;

    let bounds = InputBounds {
        left: Interval::new(-3, 3).unwrap(),
        right: Interval::new(-5, 5).unwrap(),
    };
    let proof = RowEnergyProof::new(2, bounds, 4, 9).unwrap();
    assert_eq!(
        product_bounds(bounds, 2, true, false, Some(proof)).unwrap(),
        proof.raw_bounds()
    );
    assert_eq!(
        product_bounds(bounds, 2, true, true, Some(proof)).unwrap(),
        proof.output_bounds().unwrap()
    );
    assert!(product_bounds(bounds, 2, false, true, Some(proof)).is_err());
    assert!(product_bounds(bounds, 1, true, true, Some(proof)).is_err());
    let changed = InputBounds {
        left: Interval::new(-2, 2).unwrap(),
        ..bounds
    };
    assert!(product_bounds(changed, 2, true, true, Some(proof)).is_err());
}

#[test]
fn expanded_add_and_fragmented_binary_still_follow_original_gathers() {
    let (mut builder, a, b) = inputs(257, 0);
    let a: Words = a
        .words
        .iter()
        .zip(b.words.iter())
        .flat_map(|(a, b)| [a, b])
        .collect();
    let first = builder.nodes.len();
    let output = builder.binary(&a, &a, false).unwrap();
    assert_eq!(output.len(), 514);
    for (tile, nodes) in builder.nodes[first..].chunks_exact(3).enumerate() {
        let expected = a.slice(tile * 128..(tile * 128 + 128).min(a.len()));
        assert!(matches!(&nodes[0].code, Code::Gather(words) if words == &expected));
        assert!(matches!(&nodes[1].code, Code::Gather(words) if words == &expected));
        assert!(matches!(nodes[2].code, Code::Affine { .. }));
    }
    assert_eq!(
        schedule_work(&builder.nodes[first..]).unwrap(),
        WorkBudget::default()
    );
}

fn region_sources(lanes: usize) -> BTreeMap<String, SourceTensor> {
    Program::source_shapes_with_weights(&config(lanes), 1, 1, WeightMode::Public)
        .unwrap()
        .into_iter()
        .map(|(key, shape)| {
            let public = key.contains("norm");
            let n = shape.iter().product();
            let coefficients = public.then(|| vec![ONE; n]);
            let mut hash = Sha256::new();
            for value in std::iter::repeat_n(if public { ONE } else { ONE / 64 }, n) {
                hash.update(value.to_le_bytes());
            }
            let digest = hash.finalize().into();
            let row = (shape.len() == 2).then(|| shape[1] as u64 * (ONE / 64) as u64);
            let source = SourceTensor {
                lower: if public { ONE } else { -ONE / 64 },
                upper: if public { ONE } else { ONE / 64 },
                row_l1: row,
                row_squared_norm: row
                    .map(|_| shape[1] as u128 * (ONE / 64) as u128 * (ONE / 64) as u128),
                row_l1_squared_sum: row.map(|row| shape[0] as u128 * u128::from(row).pow(2)),
                shape,
                public_q18: coefficients,
                digest,
            };
            (key, source)
        })
        .collect()
}

fn region_execution(norms: &SourceNorms) -> ExecutionPolicy {
    if norms.is_empty() {
        ExecutionPolicy::default()
    } else {
        ExecutionPolicy {
            range_policy: RangePolicy::Checked,
            max_abs_activation: 1,
            ..ExecutionPolicy::default()
        }
    }
}

fn region_validated<'a>(
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
        region_execution(norms),
        norms,
    )
    .unwrap()
}

fn region_imports(
    sources: &BTreeMap<String, SourceTensor>,
    norms: &SourceNorms,
) -> (Builder, View) {
    region_import_rows(sources, norms, 1)
}

fn region_import_rows(
    sources: &BTreeMap<String, SourceTensor>,
    norms: &SourceNorms,
    rows: usize,
) -> (Builder, View) {
    let execution = region_execution(norms);
    let mut builder = Builder {
        public_weights: true,
        checked_bound: execution.checked_bound(),
        allocated: source_norms::import_envelope(norms).unwrap().total_bytes,
        ..Builder::default()
    };
    for (key, source) in sources {
        if source.public_q18.is_some() {
            continue;
        }
        let owner = builder
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
        if let Some(record) = norms.get(key) {
            builder.public_norms.insert(owner, record.clone());
        }
    }
    let owner = builder
        .push(
            Code::Input("hidden".into()),
            2 * rows,
            Interval::new(-ONE / 4, ONE / 4).unwrap(),
        )
        .unwrap();
    let input = builder.view(owner, vec![1, rows, 2]);
    (builder, input)
}

fn expanded_region_op(
    builder: &mut Builder,
    sources: &BTreeMap<String, SourceTensor>,
    op: &pllm_models::ModelOperation,
    input: &View,
    right: Option<&View>,
) -> View {
    let shape: Vec<usize> = op.output_shape.iter().map(|n| *n as usize).collect();
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
            assert!(
                builder.checked_bound.is_none(),
                "checked fixture needs original checked_rows before ANY row gather"
            );
            let gamma = &sources[op.attributes["weight"].as_str().unwrap()];
            let width = *input.shape.last().unwrap();
            let mut words = Words::new();
            for row in input.words.chunks_exact(width) {
                let output = builder.rms_norm_row(&row, width, gamma).unwrap();
                words.extend((0..width).map(|i| (output, i)));
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
        Op::GreedyTokenSelection => builder.argmax_tree(input).unwrap(),
        Op::Reshape => {
            // Original graph recipe, independent of compact index geometry.
            let words = match op.attributes["layout"].as_str().unwrap() {
                "batch_heads_sequence_feature" => (0..shape[1])
                    .flat_map(|h| {
                        let (rows, width, heads) = (shape[2], shape[3], shape[1]);
                        (0..rows).flat_map(move |r| {
                            (0..width).map(move |d| input.words.at((r * heads + h) * width + d))
                        })
                    })
                    .collect(),
                "batch_sequence_hidden" => {
                    let (heads, rows, width) = (input.shape[1], input.shape[2], input.shape[3]);
                    (0..rows)
                        .flat_map(|r| {
                            (0..heads).flat_map(move |h| {
                                (0..width).map(move |d| input.words.at((h * rows + r) * width + d))
                            })
                        })
                        .collect()
                }
                _ => panic!("unsupported original reshape layout"),
            };
            View { words, shape }
        }
        // Same original graph lowering: preserve the input View, not the
        // operation's declared [batch, 1] shape.
        Op::TokenFeedback => input.clone(),
        _ => panic!("unsupported original region fixture"),
    }
}

#[test]
fn compact_frontier_publishes_gate_up_through_native_final_norm_head_and_selection() {
    for lanes in [2, 129, 257] {
        let graph = graph(lanes);
        let sources = region_sources(lanes);
        let norms = SourceNorms::new();
        let (imports, input) = region_imports(&sources, &norms);
        let start = graph
            .operations
            .iter()
            .position(|op| op.id.ends_with(".gate_proj"))
            .unwrap();
        let entry = graph.operations[start].inputs[0].clone();
        let residual_entry = graph.operations[start + 5]
            .inputs
            .iter()
            .find(|id| **id != graph.operations[start + 4].id)
            .unwrap()
            .clone();
        let mut compact = CompactGraph::new(
            &imports,
            region_validated(&graph, &sources, &norms),
            &graph,
            start,
        )
        .unwrap();
        compact.bind_import(&entry, &input).unwrap();
        compact.bind_import(&residual_entry, &input).unwrap();
        let (mut expanded, input) = region_imports(&sources, &norms);
        let first = expanded.nodes.len();
        let mut values = BTreeMap::from([(entry.clone(), input.clone()), (residual_entry, input)]);
        let mut last = None;
        for (index, op) in graph.operations[start..start + 11].iter().enumerate() {
            let a = values[&op.inputs[0]].clone();
            let b = op.inputs.get(1).and_then(|id| values.get(id));
            let output = expanded_region_op(&mut expanded, &sources, op, &a, b);
            let published = compact.lower_next().unwrap();
            assert!(published.words().eq(output.words.iter()));
            assert_eq!(published.bounds, expanded.bounds(&output.words).unwrap());
            assert_eq!(
                compact.work_budget(),
                schedule_work(&expanded.nodes[first..]).unwrap()
            );
            if index < 2 {
                // Original source row-energy proof, not a supplied bound.
                let proof = compact.complete_row(&published, 0).unwrap().unwrap();
                assert_eq!(Some(proof.energy), expanded.row_norm(&output.words, lanes));
                assert!(std::ptr::eq(
                    proof.source,
                    &sources[op.attributes["weight"].as_str().unwrap()]
                ));
                assert_eq!(
                    proof.source_binding,
                    source_binding_with_norms(&sources, &norms)
                );
                assert_eq!(
                    (proof.owner, proof.offset, proof.width),
                    (output.words.at(0).0, 0, lanes)
                );
            }
            if index == 3 {
                assert!(matches!(published.mapping, Mapping::Products(_)));
                assert!(compact.complete_row(&published, 0).unwrap().is_none());
                assert!(compact.frontier.value(&op.id).is_some());
            }
            if index == 5 {
                assert_eq!(
                    compact.row_norm(&published).unwrap(),
                    expanded.row_norm(&output.words, 2)
                );
                assert!(matches!(
                    compact.ops[5].kind,
                    PublishedKind::Binary { product: false, .. }
                ));
            }
            if index == 6 {
                assert!(matches!(published.mapping, Mapping::Rows { .. }));
                assert_eq!(
                    compact.row_norm(&published).unwrap(),
                    expanded.row_norm(&output.words, 2)
                );
                assert!(compact
                    .attention_energy(&published, 0, 2)
                    .unwrap()
                    .is_none());
                let row = compact.native_row_census().next().unwrap();
                assert_eq!(
                    (row.owner, row.offset, row.width),
                    (output.words.at(0).0, 0, 2)
                );
                assert_eq!(Some(row.energy), expanded.row_norm(&output.words, 2));
                assert!(std::ptr::eq(
                    row.gamma,
                    &sources[op.attributes["weight"].as_str().unwrap()]
                ));
                assert_eq!(
                    row.source_binding,
                    source_binding_with_norms(&sources, &norms)
                );
                assert!(matches!(
                    compact.ops[6].kind,
                    PublishedKind::RmsNorm {
                        input_energy: Some(_),
                        ..
                    }
                ));
            }
            if index == 7 {
                assert_eq!(published.rank, 2);
                assert_eq!(
                    compact.row_norm(&published).unwrap(),
                    expanded.row_norm(&output.words, 2)
                );
                assert_eq!(compact.ops[7].first, compact.next); // view-only, no PRSS/node work
            }
            values.insert(op.id.clone(), output);
            last = Some(published);
        }
        let last = last.unwrap();
        assert_eq!(last.shape, [1, 1, 1]);
        assert_eq!(last.rank, 1);
        assert_eq!(compact.ops[8].input.rank, 2);
        assert!(matches!(compact.ops[8].input.mapping, Mapping::Rows { .. }));
        assert!(matches!(compact.ops[4].input.mapping, Mapping::Products(_)));
        assert!(compact
            .frontier
            .value(&graph.operations[start + 3].id)
            .is_none());
        assert!(compact.frontier.value(&entry).is_none());
        let mut hash = Sha256::new();
        compact.hash_expanded_nodes(&mut hash).unwrap();
        assert_eq!(
            <[u8; 32]>::from(hash.finalize()),
            hash_expanded_binary_nodes(&expanded.nodes[first..])
        );
        assert_eq!(compact.stored_descriptors(), imports.nodes.len() + 11);
        assert_eq!(compact.next, expanded.nodes.len());
        assert!(imports
            .nodes
            .iter()
            .all(|n| matches!(n.code, Code::Input(_) | Code::PublicInput { .. })));
        // Equal closed factories share ORIGINAL capability owners, not rebuilt
        // equal-digest kernels. Publication never wraps an expanded Program.
        for (a, b) in compact
            .kernel(0)
            .unwrap()
            .programs()
            .zip(compact.kernel(1).unwrap().programs())
        {
            assert!(Arc::ptr_eq(a, b));
        }
        let census = compact.container_census().unwrap();
        assert_eq!(census.stored_operations, 11);
        assert_eq!(census.expanded_virtual_nodes, expanded.nodes.len());
        let physical: Vec<_> = compact.descriptor_census().collect();
        assert!(matches!(
            physical[4].inputs[0].unwrap().layout,
            PhysicalLayout::BinaryTiles { .. }
        ));
        assert!(matches!(
            physical[5].kind,
            DescriptorKind::Binary { multiply: false }
        ));
        assert!(matches!(
            physical[6].kind,
            DescriptorKind::RmsNorm {
                input_energy: Some(_),
                ..
            }
        ));
        assert!(matches!(
            physical[6].output.layout,
            PhysicalLayout::TensorRows {
                row_width: 2,
                node_stride: 2,
                output_node: 1,
                ..
            }
        ));
        assert_eq!(physical[7].virtual_start, physical[7].virtual_end);
        assert_eq!(physical[7].output.semantic_rank, 2);
        // Both semantic history entries retain the SAME original token owner.
        // Feedback does not synthesize final acceptance or change the View rank.
        assert!(matches!(physical[10].kind, DescriptorKind::TokenFeedback));
        assert_eq!(physical[10].virtual_start, physical[10].virtual_end);
        assert_eq!(physical[10].output.semantic_rank, 1);
        assert!(compact.ops[9]
            .output
            .words()
            .eq(compact.ops[10].output.words()));
        assert!(compact
            .frontier
            .value(&graph.operations[start + 9].id)
            .is_none());
        assert!(compact
            .frontier
            .value(&graph.output)
            .unwrap()
            .words()
            .eq(last.words()));
        assert_eq!(compact.cursor, graph.operations.len());
        assert!(!compact.failed);
        assert!(compact.failure_site().is_none());
    }
}

#[test]
fn compact_owner_and_source_certificate_mismatches_reject_without_rebinding() {
    let graph = graph(129);
    let sources = region_sources(129);
    let norms = SourceNorms::new();
    let (mut imports, input) = region_imports(&sources, &norms);
    let start = graph
        .operations
        .iter()
        .position(|op| op.id.ends_with(".gate_proj"))
        .unwrap();
    let entry = &graph.operations[start].inputs[0];
    let mut a = CompactGraph::new(
        &imports,
        region_validated(&graph, &sources, &norms),
        &graph,
        start,
    )
    .unwrap();
    let mut b = CompactGraph::new(
        &imports,
        region_validated(&graph, &sources, &norms),
        &graph,
        start,
    )
    .unwrap();
    let foreign = b.bind_import(entry, &input).unwrap();
    assert!(a.validate_view(&foreign).is_err());
    a.bind_import(entry, &input).unwrap();
    let output = a.lower_next().unwrap();
    let mut fragment = output.clone();
    fragment.start = 1;
    assert!(a.complete_row(&fragment, 0).is_err());
    let mut wrong_owner = output.clone();
    wrong_owner.mapping = Mapping::Dense(0);
    assert!(a.complete_row(&wrong_owner, 0).is_err());
    assert!(a.complete_row(&output, 1).is_err());
    drop(a);
    drop(b);
    let original = imports.nodes[0].bounds;
    imports.nodes[0].bounds = Interval::new(-1, 1).unwrap();
    assert!(CompactGraph::new(
        &imports,
        region_validated(&graph, &sources, &norms),
        &graph,
        start
    )
    .is_err());
    imports.nodes[0].bounds = original;
    if let Code::PublicInput { digest, .. } = &mut imports.nodes[0].code {
        *digest = [0; 32];
    }
    assert!(CompactGraph::new(
        &imports,
        region_validated(&graph, &sources, &norms),
        &graph,
        start
    )
    .is_err());
    if let Code::PublicInput { digest, .. } = &mut imports.nodes[0].code {
        *digest = sources
            .values()
            .find(|source| source.public_q18.is_none())
            .unwrap()
            .digest;
    }
    // No silently discarded original CompleteRow/attention/sticky ledger: such
    // prefix imports need a qualified producer/export map, not numeric labels.
    imports.row_norms.push((input.words.at(0).0, 4));
    assert!(CompactGraph::new(
        &imports,
        region_validated(&graph, &sources, &norms),
        &graph,
        start
    )
    .is_err());
}

#[test]
fn compact_multirow_final_norm_preserves_native_owners_and_last_token_alias() {
    for rows in [2, 3] {
        let graph = graph_with_rows(129, rows);
        let sources = region_sources(129);
        let norms = SourceNorms::new();
        let (imports, input) = region_import_rows(&sources, &norms, rows);
        let start = graph
            .operations
            .iter()
            .position(|op| op.id.ends_with(".gate_proj"))
            .unwrap();
        let entry = graph.operations[start].inputs[0].clone();
        let residual_entry = graph.operations[start + 5]
            .inputs
            .iter()
            .find(|id| **id != graph.operations[start + 4].id)
            .unwrap()
            .clone();
        let mut compact = CompactGraph::new(
            &imports,
            region_validated(&graph, &sources, &norms),
            &graph,
            start,
        )
        .unwrap();
        compact.bind_import(&entry, &input).unwrap();
        compact.bind_import(&residual_entry, &input).unwrap();
        let (mut expanded, input) = region_import_rows(&sources, &norms, rows);
        let first = expanded.nodes.len();
        let mut values = BTreeMap::from([(entry, input.clone()), (residual_entry, input)]);
        for (index, op) in graph.operations[start..start + 11].iter().enumerate() {
            let output = expanded_region_op(
                &mut expanded,
                &sources,
                op,
                &values[&op.inputs[0]],
                op.inputs.get(1).and_then(|id| values.get(id)),
            );
            let published = compact.lower_next().unwrap();
            assert!(published.words().eq(output.words.iter()));
            assert_eq!(published.bounds, expanded.bounds(&output.words).unwrap());
            assert_eq!(compact.next, expanded.nodes.len());
            assert_eq!(
                compact.work_budget(),
                schedule_work(&expanded.nodes[first..]).unwrap()
            );
            if index == 6 {
                let owners: Vec<_> = compact.native_row_census().collect();
                assert_eq!(owners.len(), rows);
                for (row, owner) in owners.iter().enumerate() {
                    assert_eq!(
                        (
                            owner.descriptor,
                            owner.row,
                            owner.owner,
                            owner.offset,
                            owner.width
                        ),
                        (6, row, output.words.at(2 * row).0, 0, 2)
                    );
                    assert_eq!(
                        Some(owner.energy),
                        expanded.row_norm(&output.words.slice(2 * row..2 * row + 2), 2)
                    );
                    let Code::Tensor { program, .. } = &expanded.nodes[owner.owner].code else {
                        panic!("RMS must remain an original Tensor per row");
                    };
                    assert_eq!(program.digest(), owner.program.digest());
                    assert!(Arc::ptr_eq(owner.program, owners[0].program));
                }
                let mut forged = published.clone();
                forged.start += 1;
                assert!(compact.validate_view(&forged).is_err());
                forged = published.clone();
                forged.mapping = Mapping::Rows {
                    first: compact.ops[6].first + 1,
                    width: 2,
                };
                assert!(compact.validate_view(&forged).is_err());
                // Native row proof is not relabelled as source-projection proof.
                assert!(compact.complete_row(&published, 0).unwrap().is_none());
            }
            if index == 7 {
                assert_eq!(published.start, 2 * (rows - 1));
                assert_eq!(published.rank, 2);
                assert_eq!(
                    compact.row_norm(&published).unwrap(),
                    expanded.row_norm(&output.words, 2)
                );
            }
            if index == 8 {
                let physical = compact.descriptor_census().last().unwrap();
                assert_eq!(physical.inputs[0].unwrap().offset, 2 * (rows - 1));
                assert_eq!(physical.inputs[0].unwrap().semantic_rank, 2);
            }
            values.insert(op.id.clone(), output);
        }
        let mut hash = Sha256::new();
        compact.hash_expanded_nodes(&mut hash).unwrap();
        assert_eq!(
            <[u8; 32]>::from(hash.finalize()),
            hash_expanded_binary_nodes(&expanded.nodes[first..])
        );
        assert_eq!(compact.cursor, graph.operations.len());
        assert!(matches!(compact.ops[10].kind, PublishedKind::TokenFeedback));
        assert_eq!(compact.ops[10].output.rank, 1);
        assert!(!compact.failed);
    }
}

#[test]
fn compact_feedback_history_keeps_original_dense_and_tiled_aliases_without_arithmetic() {
    let full = graph(2);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    for rows in [3, 65] {
        for tiled in [false, true] {
            let mut region = full.clone();
            let mut last = full
                .operations
                .iter()
                .find(|op| op.operator == Op::LastToken)
                .unwrap()
                .clone();
            let entry = "local.alias.input".to_owned();
            last.inputs = vec![entry.clone()];
            region.operations.clear();
            if tiled {
                let mut residual = full
                    .operations
                    .iter()
                    .find(|op| op.operator == Op::ResidualAdd)
                    .unwrap()
                    .clone();
                residual.inputs = vec![entry.clone(), entry.clone()];
                residual.output_shape = vec![1, rows as u64, 2];
                last.inputs = vec![residual.id.clone()];
                region.operations.push(residual);
            }
            let mut feedback = full
                .operations
                .iter()
                .find(|op| op.operator == Op::TokenFeedback)
                .unwrap()
                .clone();
            feedback.inputs = vec![last.id.clone()];
            feedback.output_shape = vec![1, 2];
            let mut repeated = feedback.clone();
            repeated.id.push_str(".history");
            repeated.inputs = vec![feedback.id.clone()];
            region.output = repeated.id.clone();
            region.operations.extend([last, feedback, repeated]);
            let (imports, input) = region_import_rows(&sources, &norms, rows);
            let before = (
                imports.nodes.len(),
                imports.allocated,
                imports.tensors.len(),
                imports.validity,
            );
            let mut compact = CompactGraph::new(
                &imports,
                region_validated(&full, &sources, &norms),
                &region,
                0,
            )
            .unwrap();
            compact.bind_import(&entry, &input).unwrap();
            let (mut expanded, input) = region_import_rows(&sources, &norms, rows);
            let first = expanded.nodes.len();
            let mut values = BTreeMap::from([(entry, input)]);
            let attempts = (
                super::super::super::prepared_tensor_map::native_attempts(),
                RMS_NATIVE_ATTEMPTS.with(|n| n.get()),
                ARGMAX_NATIVE_ATTEMPTS.with(|n| n.get()),
            );
            for op in &region.operations {
                let original = expanded_region_op(
                    &mut expanded,
                    &sources,
                    op,
                    &values[&op.inputs[0]],
                    op.inputs.get(1).and_then(|id| values.get(id)),
                );
                let next = compact.next;
                let work = compact.work_budget();
                let published = compact.lower_next().unwrap();
                assert!(published.words().eq(original.words.iter()));
                assert_eq!(published.rank as usize, original.shape.len());
                assert_eq!(compact.next, expanded.nodes.len());
                if op.operator == Op::TokenFeedback {
                    let stored = compact.ops.last().unwrap();
                    assert_eq!(published.mapping, stored.input.mapping);
                    assert_eq!(published.start, 2 * (rows - 1));
                    assert_eq!(published.start, stored.input.start);
                    assert_eq!(published.shape, stored.input.shape);
                    assert_eq!(published.rank, stored.input.rank);
                    assert_eq!(published.bounds, stored.input.bounds);
                    assert_eq!(compact.next, next);
                    assert_eq!(compact.work_budget(), work);
                    assert_eq!(
                        compact.row_norm(&published).unwrap(),
                        compact.row_norm(&stored.input).unwrap()
                    );
                    let census = compact.descriptor_census().last().unwrap();
                    assert!(matches!(census.kind, DescriptorKind::TokenFeedback));
                    assert_eq!(census.virtual_start, census.virtual_end);
                    assert_eq!(census.output.offset, 2 * (rows - 1));
                    let mut forged = published.clone();
                    forged.start -= 1;
                    assert!(compact.validate_view(&forged).is_err());
                }
                values.insert(op.id.clone(), original);
            }
            assert_eq!(
                before,
                (
                    imports.nodes.len(),
                    imports.allocated,
                    imports.tensors.len(),
                    imports.validity
                )
            );
            assert!(compact.owned_native_programs().next().is_none());
            assert_eq!(
                attempts,
                (
                    super::super::super::prepared_tensor_map::native_attempts(),
                    RMS_NATIVE_ATTEMPTS.with(|n| n.get()),
                    ARGMAX_NATIVE_ATTEMPTS.with(|n| n.get()),
                )
            );
            let mut hash = Sha256::new();
            compact.hash_expanded_nodes(&mut hash).unwrap();
            assert_eq!(
                <[u8; 32]>::from(hash.finalize()),
                hash_expanded_binary_nodes(&expanded.nodes[first..])
            );
            // Retired semantic inputs still have their immutable descriptor maps;
            // only the final identity remains in the shared frontier.
            for op in &region.operations[..region.operations.len() - 1] {
                assert!(compact.frontier.value(&op.id).is_none());
            }
            assert!(compact
                .frontier
                .value(&region.output)
                .unwrap()
                .words()
                .eq(values[&region.output].words.iter()));
        }
    }
}

#[test]
fn compact_feedback_rejects_lane_mismatch_and_extra_inputs_with_sticky_failure() {
    let full = graph(2);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    for extra_input in [false, true] {
        let mut region = full.clone();
        let mut feedback = full
            .operations
            .iter()
            .find(|op| op.operator == Op::TokenFeedback)
            .unwrap()
            .clone();
        if extra_input {
            feedback.inputs.push(feedback.inputs[0].clone());
            feedback.output_shape = vec![1, 2];
        } // Otherwise original [1, 1] declares too few lanes for this input.
        region.output = feedback.id.clone();
        region.operations = vec![feedback];
        let (imports, input) = region_imports(&sources, &norms);
        let mut compact = CompactGraph::new(
            &imports,
            region_validated(&full, &sources, &norms),
            &region,
            0,
        )
        .unwrap();
        compact
            .bind_import(&region.operations[0].inputs[0], &input)
            .unwrap();
        assert!(compact.lower_next().is_err());
        assert!(compact.failed);
        assert!(compact.ops.is_empty());
        assert_eq!(compact.next, imports.nodes.len());
        assert_eq!(compact.work_budget(), WorkBudget::default());
        assert_eq!(compact.failure_site().unwrap().operator, Op::TokenFeedback);
        assert!(compact.lower_next().is_err());
    }
}

#[test]
fn compact_native_norm_reuses_original_arc_without_relabelling_row_certificates() {
    let full = graph(2);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    let start = full
        .operations
        .iter()
        .position(|op| op.id.ends_with(".gate_proj"))
        .unwrap();
    let mut region = full.clone();
    region.operations = full.operations[start..start + 7].to_vec();
    let mut second = region.operations[6].clone();
    second.id.push_str(".same_input");
    region.output = second.id.clone();
    region.operations.push(second);
    let entry = region.operations[0].inputs[0].clone();
    let residual_entry = region.operations[5]
        .inputs
        .iter()
        .find(|id| **id != region.operations[4].id)
        .unwrap()
        .clone();
    let (imports, input) = region_imports(&sources, &norms);
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    compact.bind_import(&entry, &input).unwrap();
    compact.bind_import(&residual_entry, &input).unwrap();
    let (mut expanded, input) = region_imports(&sources, &norms);
    let first = expanded.nodes.len();
    let mut values = BTreeMap::from([(entry, input.clone()), (residual_entry, input)]);
    for op in &region.operations {
        let original = expanded_region_op(
            &mut expanded,
            &sources,
            op,
            &values[&op.inputs[0]],
            op.inputs.get(1).and_then(|id| values.get(id)),
        );
        let output = compact.lower_next().unwrap();
        assert!(output.words().eq(original.words.iter()));
        values.insert(op.id.clone(), original);
    }
    let rows: Vec<_> = compact.native_row_census().collect();
    assert_eq!(rows.len(), 2);
    assert_ne!(rows[0].owner, rows[1].owner);
    assert!(Arc::ptr_eq(rows[0].program, rows[1].program));
    assert_eq!(rows[0].energy, rows[1].energy);
    assert!(matches!(
        compact.ops[6].kind,
        PublishedKind::RmsNorm {
            owns_program: true,
            ..
        }
    ));
    assert!(matches!(
        compact.ops[7].kind,
        PublishedKind::RmsNorm {
            owns_program: false,
            ..
        }
    ));
    assert_eq!(
        compact
            .owned_native_programs()
            .filter(|program| program.digest() == rows[0].program.digest())
            .count(),
        1
    );
    let mut hash = Sha256::new();
    compact.hash_expanded_nodes(&mut hash).unwrap();
    assert_eq!(
        <[u8; 32]>::from(hash.finalize()),
        hash_expanded_binary_nodes(&expanded.nodes[first..])
    );
    assert_eq!(
        compact.work_budget(),
        schedule_work(&expanded.nodes[first..]).unwrap()
    );
}

#[test]
fn compact_native_norm_preadmits_factory_and_rejects_checked_or_offset_substitutes() {
    let full = graph(2);
    let sources = region_sources(2);
    let mut region = full.clone();
    region.operations = vec![full
        .operations
        .iter()
        .find(|op| op.id == "final_norm")
        .unwrap()
        .clone()];
    region.output = region.operations[0].id.clone();
    let norms = SourceNorms::new();
    let (imports, input) = region_imports(&sources, &norms);
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    compact
        .bind_import(&region.operations[0].inputs[0], &input)
        .unwrap();
    compact.limit = compact.baseline().unwrap();
    let attempts = RMS_NATIVE_ATTEMPTS.with(|count| count.get());
    assert!(compact.lower_next().is_err());
    assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
    assert_eq!(compact.next, imports.nodes.len());
    assert!(compact.ops.is_empty());
    assert_eq!(compact.failure_site().unwrap().operation, "final_norm");
    assert!(compact.lower_next().is_err());
    drop(compact);
    region.operations[0].attributes["weight_offset"] = serde_json::json!(1);
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    compact
        .bind_import(&region.operations[0].inputs[0], &input)
        .unwrap();
    assert!(compact.lower_next().is_err());
    assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
    drop(compact);
    region.operations[0].attributes["weight_offset"] = serde_json::json!(0);
    let mut checked_norms = SourceNorms::new();
    for (key, (rows, columns, width)) in source_norms::requests(&full).unwrap() {
        let mut scan = source_norms::BlockScan::new(rows, columns, width).unwrap();
        scan.append(0, &vec![ONE / 64; rows * columns]).unwrap();
        checked_norms.insert(key, scan.finish().unwrap());
    }
    let (imports, input) = region_imports(&sources, &checked_norms);
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &checked_norms),
        &region,
        0,
    )
    .unwrap();
    compact
        .bind_import(&region.operations[0].inputs[0], &input)
        .unwrap();
    assert!(imports.checked_bound.is_some());
    assert!(compact.lower_next().is_err());
    assert!(compact.native_row_census().next().is_none());
    assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
}

#[test]
fn compact_native_norm_preserves_original_numeric_and_gamma_binding_rejections() {
    let full = graph(2);
    let mut sources = region_sources(2);
    let norms = SourceNorms::new();
    let mut region = full.clone();
    region.operations = vec![full
        .operations
        .iter()
        .find(|op| op.id == "final_norm")
        .unwrap()
        .clone()];
    region.output = region.operations[0].id.clone();
    let (mut imports, input) = region_imports(&sources, &norms);
    let owner = input.words.at(0).0;
    imports.nodes[owner].bounds = Interval::new(-(1_i64 << 40), 1_i64 << 40).unwrap();
    let gamma_key = region.operations[0].attributes["weight"].as_str().unwrap();
    assert!(rms_norm_program(
        true,
        2,
        imports.nodes[owner].bounds,
        sources[gamma_key].public_q18.as_ref().unwrap(),
        None
    )
    .is_err());
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    compact
        .bind_import(&region.operations[0].inputs[0], &input)
        .unwrap();
    assert!(compact.lower_next().is_err());
    assert_eq!(compact.next, imports.nodes.len());
    assert!(compact.ops.is_empty());
    drop(compact);
    // Encoded gamma source bytes cannot change beneath the original commitment.
    sources
        .get_mut(gamma_key)
        .unwrap()
        .public_q18
        .as_mut()
        .unwrap()[0] += 1;
    let shapes = sources
        .iter()
        .map(|(key, source)| (key.clone(), source.shape.clone()))
        .collect();
    assert!(validate_source_records(
        &full,
        &shapes,
        &sources,
        WeightMode::Public,
        region_execution(&norms),
        &norms
    )
    .is_err());
}

#[test]
fn compact_source_attention_certificate_stays_bound_to_exact_projection_block() {
    let full = graph(2);
    let sources = region_sources(2);
    let mut norms = SourceNorms::new();
    for (key, (rows, columns, width)) in source_norms::requests(&full).unwrap() {
        let mut scan = source_norms::BlockScan::new(rows, columns, width).unwrap();
        scan.append(0, &vec![ONE / 64; rows * columns]).unwrap();
        norms.insert(key, scan.finish().unwrap());
    }
    // A generic two-projection region uses ORIGINAL Q source/block ownership.
    // First source-statistics projection supplies the exact complete-row energy
    // consumed by the second, just as original projection_energies does.
    let gate = full
        .operations
        .iter()
        .find(|op| op.id.ends_with(".gate_proj"))
        .unwrap()
        .clone();
    let mut query = full
        .operations
        .iter()
        .find(|op| {
            op.operator == Op::Linear
                && op.attributes.get("weight").and_then(|value| value.as_str())
                    == Some("model.layers.0.self_attn.q_proj.weight")
        })
        .unwrap()
        .clone();
    assert!(norms.contains_key(query.attributes["weight"].as_str().unwrap()));
    query.inputs = vec![gate.id.clone()];
    let mut region = full.clone();
    region.operations = vec![gate.clone(), query.clone()];
    region
        .operations
        .extend(original_reshape_pair(&full, &query.id, 2, 1, 1, false));
    region.output = region.operations.last().unwrap().id.clone();
    let (imports, input) = region_imports(&sources, &norms);
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    compact.bind_import(&gate.inputs[0], &input).unwrap();
    let first = compact.lower_next().unwrap();
    assert!(compact.complete_row(&first, 0).unwrap().is_some());
    let second = compact.lower_next().unwrap();
    let (mut expanded, input) = region_imports(&sources, &norms);
    let first = expanded_region_op(&mut expanded, &sources, &gate, &input, None);
    let second_original = expanded_region_op(&mut expanded, &sources, &query, &first, None);
    assert_eq!(
        compact.attention_energy(&second, 0, 2).unwrap(),
        expanded.attention_energy(&second_original.words, 2)
    );
    assert!(compact.attention_energy(&second, 0, 2).unwrap().is_some());
    assert!(compact.attention_energy(&second, 1, 1).unwrap().is_none());
    assert!(compact.attention_energy(&second, 1, 2).is_err());
    let slot = second.slot.unwrap();
    if let PublishedKind::Projection {
        norms: Some(actual),
        source,
        weight,
        ..
    } = &compact.ops[slot].kind
    {
        let key = query.attributes["weight"].as_str().unwrap();
        assert!(std::ptr::eq(*actual, &norms[key]));
        assert!(std::ptr::eq(*source, &sources[key]));
        assert_eq!(
            *weight,
            compact
                .complete_row(&second, 0)
                .unwrap()
                .unwrap()
                .source_owner
        );
    } else {
        panic!("lost original source/block provenance");
    }
    let attention = compact.attention_energy(&second, 0, 2).unwrap();
    let energy = compact.complete_row(&second, 0).unwrap().unwrap().energy;
    let owners: Vec<_> = compact.owned_native_programs().map(Arc::as_ptr).collect();
    let split = compact.lower_next().unwrap();
    assert_eq!(compact.attention_energy(&split, 0, 2).unwrap(), attention);
    assert!(compact.complete_row(&split, 0).unwrap().is_none());
    assert!(compact.row_norm(&split).unwrap().is_none());
    let restored = compact.lower_next().unwrap();
    assert_eq!(
        compact.attention_energy(&restored, 0, 2).unwrap(),
        attention
    );
    assert_eq!(
        compact.complete_row(&restored, 0).unwrap().unwrap().energy,
        energy
    );
    assert_eq!(
        compact
            .owned_native_programs()
            .map(Arc::as_ptr)
            .collect::<Vec<_>>(),
        owners
    );
}

#[test]
fn compact_frontier_and_native_factory_are_preadmitted_and_failure_retires_region() {
    let graph = graph(129);
    let sources = region_sources(129);
    let norms = SourceNorms::new();
    let (imports, input) = region_imports(&sources, &norms);
    let start = graph
        .operations
        .iter()
        .position(|op| op.id.ends_with(".gate_proj"))
        .unwrap();
    let attempts = super::super::super::prepared_tensor_map::native_attempts();
    assert!(CompactGraph::with_limit(
        &imports,
        region_validated(&graph, &sources, &norms),
        &graph,
        start,
        0
    )
    .is_err());
    assert_eq!(
        attempts,
        super::super::super::prepared_tensor_map::native_attempts()
    );
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&graph, &sources, &norms),
        &graph,
        start,
    )
    .unwrap();
    compact
        .bind_import(&graph.operations[start].inputs[0], &input)
        .unwrap();
    compact.limit = compact.baseline().unwrap();
    assert!(compact.lower_next().is_err());
    assert!(compact.ops.is_empty());
    assert_eq!(compact.next, imports.nodes.len());
    let failure = compact.failure_site().unwrap();
    assert_eq!(failure.operation, graph.operations[start].id);
    assert_eq!(failure.operator, Op::Linear);
    assert_eq!(failure.input_owners[0], Some(input.words.at(0).0));
    assert_eq!(
        attempts,
        super::super::super::prepared_tensor_map::native_attempts()
    );
    assert!(compact.lower_next().is_err());
}

fn selection_region(full: &DecoderGraph) -> DecoderGraph {
    let mut region = full.clone();
    region.operations = vec![full
        .operations
        .iter()
        .find(|op| op.operator == Op::GreedyTokenSelection)
        .unwrap()
        .clone()];
    region.output = region.operations[0].id.clone();
    region
}

fn selection_imports(
    sources: &BTreeMap<String, SourceTensor>,
    norms: &SourceNorms,
    lanes: usize,
) -> (Builder, View) {
    let (mut builder, _) = region_imports(sources, norms);
    let owner = builder
        .push(
            Code::Input("local.logits".into()),
            lanes,
            Interval::new(-32, 32).unwrap(),
        )
        .unwrap();
    let view = builder.view(owner, vec![1, 1, lanes]);
    (builder, view)
}

// Test-only point semantics for this already-qualified native sign constructor.
// This is NOT a share/protocol/PRSS parity claim or a production selector.
fn selection_point_affine(left: &[i64], right: &[i64], coefficients: Affine) -> Vec<i64> {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            i64::try_from(
                i128::from(*left) * i128::from(coefficients.left)
                    + i128::from(*right) * i128::from(coefficients.right)
                    + i128::from(coefficients.constant),
            )
            .unwrap()
        })
        .collect()
}

fn selection_expanded_points(
    builder: &Builder,
    first: usize,
    input_owner: usize,
    input: &[i64],
) -> BTreeMap<usize, Vec<i64>> {
    let mut values = BTreeMap::from([(input_owner, input.to_vec())]);
    for (offset, node) in builder.nodes[first..].iter().enumerate() {
        let output = match &node.code {
            Code::PublicValues(values) => values.clone(),
            Code::Gather(words) => words
                .iter()
                .map(|(owner, lane)| values[&owner][lane])
                .collect(),
            Code::Affine {
                left,
                right: Some(right),
                coefficients,
            } => selection_point_affine(&values[left], &values[right], *coefficients),
            Code::Product {
                left,
                right,
                shape: None,
                rescale: false,
                norm: None,
            } => values[left]
                .iter()
                .zip(&values[right])
                .map(|(a, b)| a.checked_mul(*b).unwrap())
                .collect(),
            Code::Tensor { input, program } => {
                assert_eq!(program.output_bounds(), Interval::new(0, 1).unwrap());
                values[input]
                    .iter()
                    .map(|value| i64::from(*value < 0))
                    .collect()
            }
            _ => panic!("unexpected original selection primitive"),
        };
        values.insert(first + offset, output);
    }
    values
}

#[test]
fn compact_argmax_original_primitives_identity_work_and_independent_tie_oracle() {
    let full = graph(2);
    let region = selection_region(&full);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    for lanes in [
        1, 2, 3, 127, 128, 129, 255, 256, 257, 258, 511, 512, 513, 1025,
    ] {
        let (imports, input) = selection_imports(&sources, &norms, lanes);
        let first = imports.nodes.len();
        let mut compact = CompactGraph::new(
            &imports,
            region_validated(&full, &sources, &norms),
            &region,
            0,
        )
        .unwrap();
        compact
            .bind_import(&region.operations[0].inputs[0], &input)
            .unwrap();
        let output = compact.lower_next().unwrap();
        let (mut expanded, original) = selection_imports(&sources, &norms, lanes);
        let reference = expanded.argmax_tree(&original).unwrap();
        assert!(output.words().eq(reference.words.iter()));
        assert_eq!(compact.next, expanded.nodes.len());
        assert_eq!(
            compact.work_budget(),
            schedule_work(&expanded.nodes[first..]).unwrap()
        );
        let mut hash = Sha256::new();
        compact.hash_expanded_nodes(&mut hash).unwrap();
        assert_eq!(
            <[u8; 32]>::from(hash.finalize()),
            hash_expanded_binary_nodes(&expanded.nodes[first..])
        );
        let census = compact.argmax_census().next().unwrap().unwrap();
        assert_eq!(census.expanded_nodes, expanded.nodes.len() - first);
        assert_eq!(census.expanded_nodes, 1 + 12 * census.compare_blocks);
        assert!(census.kernel_count <= census.rounds + 1);
        assert_eq!(census.kernel_count, census.kernel_capacity);
        assert_eq!(compact.stored_descriptors(), imports.nodes.len() + 1);
        let mut last_high = (0..lanes).map(|i| i as i64 % 23 - 11).collect::<Vec<_>>();
        last_high[lanes - 1] = 31;
        let mut tied = last_high.clone();
        tied[0] = 31;
        for data in [vec![0; lanes], last_high, tied] {
            let maximum = *data.iter().max().unwrap();
            let expected = data.iter().position(|value| *value == maximum).unwrap() as i64;
            let expected_values =
                selection_expanded_points(&expanded, first, input.words.at(0).0, &data);
            let mut actual = BTreeMap::from([(input.words.at(0).0, data)]);
            let (mut events, mut products, mut compare_lanes) = (0usize, 0usize, 0usize);
            let mut owners = BTreeMap::new();
            let attempts = ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get());
            compact.visit_argmax(0, &mut |event| {
                let original = &expanded.nodes[event.owner];
                assert_eq!(event.owner, first + events);
                assert_eq!((event.lanes, event.bounds), (original.lanes, original.bounds));
                events += 1;
                let value: Vec<i64> = match event.code {
                    ArgmaxCode::PublicIndices => {
                        let Code::PublicValues(values) = &original.code else { panic!("iota position changed"); };
                        assert!(values.iter().copied().eq(0..lanes as i64));
                        (0..lanes as i64).collect()
                    }
                    ArgmaxCode::Gather(gather) => {
                        let Code::Gather(words) = &original.code else { panic!("gather position changed"); };
                        assert!(gather.words().eq(words.iter()));
                        gather.words().map(|(owner, lane)| actual[&owner][lane]).collect()
                    }
                    ArgmaxCode::Affine { left, right, coefficients } => {
                        let Code::Affine { left: old_left, right: Some(old_right), coefficients: old } = &original.code else { panic!("mask affine position changed"); };
                        assert_eq!((left, right, coefficients.left, coefficients.right, coefficients.constant), (*old_left, *old_right, old.left, old.right, old.constant));
                        selection_point_affine(&actual[&left], &actual[&right], coefficients)
                    }
                    ArgmaxCode::Product { left, right } => {
                        assert!(matches!(&original.code, Code::Product { left: a, right: b, shape: None, rescale: false, norm: None } if (*a, *b) == (left, right)));
                        products += event.lanes;
                        actual[&left].iter().zip(&actual[&right]).map(|(a, b)| a.checked_mul(*b).unwrap()).collect()
                    }
                    ArgmaxCode::Tensor { input, program } => {
                        let Code::Tensor { input: old, program: old_program } = &original.code else { panic!("negative position changed"); };
                        assert_eq!(input, *old);
                        assert_eq!(program.digest(), old_program.digest());
                        assert_eq!(*owners.entry(event.lanes).or_insert(Arc::as_ptr(program)), Arc::as_ptr(program));
                        compare_lanes += event.lanes;
                        actual[&input].iter().map(|value| i64::from(*value < 0)).collect()
                    }
                };
                assert_eq!(value.len(), event.lanes);
                actual.insert(event.owner, value);
                Ok(())
            }).unwrap();
            assert_eq!(attempts, ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get())); // no rebuild during visitation
            assert_eq!(actual, expected_values);
            assert_eq!(events, census.expanded_nodes);
            assert_eq!(compare_lanes, lanes - 1);
            assert_eq!(products, 2 * (lanes - 1)); // preserve BOTH secret select masks
            let (owner, lane) = output.word(0).unwrap();
            assert_eq!(actual[&owner][lane], expected);
        }
    }
}

#[test]
fn compact_argmax_three_logits_keeps_frozen_pair_order_and_original_odd_carry() {
    let full = graph(2);
    let region = selection_region(&full);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    let (imports, input) = selection_imports(&sources, &norms, 3);
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    compact
        .bind_import(&region.operations[0].inputs[0], &input)
        .unwrap();
    let first = imports.nodes.len();
    let root = input.words.at(0).0;
    let output = compact.lower_next().unwrap();
    assert_eq!(output.word(0).unwrap(), (first + 24, 0));
    let mut gathers = BTreeMap::new();
    compact
        .visit_argmax(0, &mut |event| {
            if let ArgmaxCode::Gather(gather) = event.code {
                gathers.insert(event.owner - first, gather.words().collect::<Vec<_>>());
            }
            Ok(())
        })
        .unwrap();
    // Independently fixed original trace: no padding, no candidate reorder, no
    // comparison against a fictitious minimum, and no rewritten tie circuit.
    assert_eq!(
        gathers,
        BTreeMap::from([
            (1, vec![(root, 0)]),
            (2, vec![(root, 1)]),
            (8, vec![(first, 0)]),
            (9, vec![(first, 1)]),
            (13, vec![(first + 7, 0)]),
            (14, vec![(root, 2)]),
            (20, vec![(first + 12, 0)]),
            (21, vec![(first, 2)]),
        ])
    );
}

#[test]
fn compact_argmax_enclosing_exact_fit_and_one_short_reject_before_any_factory() {
    let full = graph(2);
    let region = selection_region(&full);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    let (imports, input) = selection_imports(&sources, &norms, 257);
    let mut measured = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    measured
        .bind_import(&region.operations[0].inputs[0], &input)
        .unwrap();
    measured.lower_next().unwrap();
    let local = measured.construction_peak_bytes();
    drop(measured);
    let outside = MAX_PROGRAM_BYTES.checked_sub(local).unwrap();
    let mut exact = CompactGraph::with_owner_baseline(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
        outside,
    )
    .unwrap();
    exact
        .bind_import(&region.operations[0].inputs[0], &input)
        .unwrap();
    exact.lower_next().unwrap();
    assert_eq!(exact.construction_peak_bytes(), MAX_PROGRAM_BYTES);
    assert_eq!(exact.outside_owner_bytes(), outside);
    drop(exact);
    let attempts = ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get());
    let mut short = CompactGraph::with_owner_baseline(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
        outside + 1,
    )
    .unwrap();
    short
        .bind_import(&region.operations[0].inputs[0], &input)
        .unwrap();
    assert!(short.lower_next().is_err());
    assert_eq!(attempts, ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()));
    assert!(short.ops.is_empty());
    assert_eq!(short.next, imports.nodes.len());
    assert!(short.failed);
    assert_eq!(
        short.failure_site().unwrap().operator,
        Op::GreedyTokenSelection
    );
    assert!(short.lower_next().is_err());
}

#[test]
fn compact_existing_prepared_and_rms_factories_include_outside_exact_and_short() {
    let full = graph(2);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    for operation in [
        full.operations
            .iter()
            .find(|op| op.id.ends_with(".gate_proj"))
            .unwrap(),
        full.operations
            .iter()
            .find(|op| op.id == "final_norm")
            .unwrap(),
    ] {
        let mut region = full.clone();
        region.operations = vec![operation.clone()];
        region.output = operation.id.clone();
        let (imports, input) = region_imports(&sources, &norms);
        let mut measured = CompactGraph::new(
            &imports,
            region_validated(&full, &sources, &norms),
            &region,
            0,
        )
        .unwrap();
        measured.bind_import(&operation.inputs[0], &input).unwrap();
        measured.lower_next().unwrap();
        let outside = MAX_PROGRAM_BYTES - measured.construction_peak_bytes();
        drop(measured);
        let mut exact = CompactGraph::with_owner_baseline(
            &imports,
            region_validated(&full, &sources, &norms),
            &region,
            0,
            outside,
        )
        .unwrap();
        exact.bind_import(&operation.inputs[0], &input).unwrap();
        exact.lower_next().unwrap();
        assert_eq!(exact.construction_peak_bytes(), MAX_PROGRAM_BYTES);
        drop(exact);
        let attempts = (
            super::super::super::prepared_tensor_map::native_attempts(),
            RMS_NATIVE_ATTEMPTS.with(|count| count.get()),
        );
        let mut short = CompactGraph::with_owner_baseline(
            &imports,
            region_validated(&full, &sources, &norms),
            &region,
            0,
            outside + 1,
        )
        .unwrap();
        short.bind_import(&operation.inputs[0], &input).unwrap();
        assert!(short.lower_next().is_err());
        assert_eq!(
            attempts,
            (
                super::super::super::prepared_tensor_map::native_attempts(),
                RMS_NATIVE_ATTEMPTS.with(|count| count.get())
            )
        );
        assert!(short.ops.is_empty());
        assert_eq!(short.next, imports.nodes.len());
    }
    let (imports, _) = region_imports(&sources, &norms);
    assert!(CompactGraph::with_owner_baseline(
        &imports,
        region_validated(&full, &sources, &norms),
        &full,
        0,
        usize::MAX
    )
    .is_err());
}

#[test]
fn compact_argmax_rejects_alternative_policy_and_original_numeric_overflow() {
    let full = graph(2);
    let region = selection_region(&full);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    let shapes = sources
        .iter()
        .map(|(key, source)| (key.clone(), source.shape.clone()))
        .collect();
    for policy in [
        ExecutionPolicy {
            selection: SelectionPolicy::Tiled,
            ..ExecutionPolicy::default()
        },
        ExecutionPolicy {
            lookup: LookupPolicy::Trie,
            ..ExecutionPolicy::default()
        },
    ] {
        let (mut imports, input) = selection_imports(&sources, &norms, 3);
        imports.selection = policy.selection;
        imports.lookup = policy.lookup;
        let validated =
            validate_source_records(&full, &shapes, &sources, WeightMode::Public, policy, &norms)
                .unwrap();
        let mut compact = CompactGraph::new(&imports, validated, &region, 0).unwrap();
        compact
            .bind_import(&region.operations[0].inputs[0], &input)
            .unwrap();
        let attempts = ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get());
        assert!(compact.lower_next().is_err());
        assert_eq!(attempts, ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()));
        assert!(compact.ops.is_empty());
    }
    let (mut imports, input) = selection_imports(&sources, &norms, 3);
    imports.nodes[input.words.at(0).0].bounds = Interval::new(-(1_i64 << 62), 1_i64 << 62).unwrap();
    let (mut expanded, reference) = selection_imports(&sources, &norms, 3);
    expanded.nodes[reference.words.at(0).0].bounds = imports.nodes[input.words.at(0).0].bounds;
    assert!(expanded.argmax_tree(&reference).is_err());
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    compact
        .bind_import(&region.operations[0].inputs[0], &input)
        .unwrap();
    let attempts = ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get());
    assert!(compact.lower_next().is_err());
    assert_eq!(attempts, ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()));
    assert!(compact.ops.is_empty());
}

#[test]
fn compact_argmax_geometry_is_bounded_not_a_source_request_admission() {
    for lanes in [1, 3, 257, 151_936, MAX_VALUES] {
        let (geometry, peak) = CompactArgmax::quote(lanes, 0).unwrap();
        assert_eq!(geometry.level(geometry.rounds).unwrap().0, 1);
        assert!(geometry.kernels <= geometry.rounds + 1);
        assert!(geometry.kernels <= 128);
        assert_eq!(geometry.nodes().unwrap(), 1 + 12 * geometry.blocks);
        assert!(peak <= MAX_PROGRAM_BYTES);
        if lanes >= 257 {
            assert!(array_bytes::<NegativeProgram>(geometry.kernels).unwrap() < lanes * 8);
        }
    }
    assert!(ArgmaxGeometry::new(0).is_err());
    assert!(ArgmaxGeometry::new(MAX_VALUES + 1).is_err());
    assert!(CompactArgmax::quote(257, usize::MAX).is_err());
}

fn original_reshape_pair(
    full: &DecoderGraph,
    entry: &str,
    heads: usize,
    rows: usize,
    width: usize,
    from_heads: bool,
) -> Vec<pllm_models::ModelOperation> {
    let mut heads_op = full
        .operations
        .iter()
        .find(|op| {
            op.operator == Op::Reshape && op.attributes["layout"] == "batch_heads_sequence_feature"
        })
        .unwrap()
        .clone();
    let mut sequence_op = full
        .operations
        .iter()
        .find(|op| op.operator == Op::Reshape && op.attributes["layout"] == "batch_sequence_hidden")
        .unwrap()
        .clone();
    heads_op.output_shape = vec![1, heads as u64, rows as u64, width as u64];
    sequence_op.output_shape = vec![1, rows as u64, (heads * width) as u64];
    let (mut first, mut second) = if from_heads {
        (sequence_op, heads_op)
    } else {
        (heads_op, sequence_op)
    };
    first.inputs = vec![entry.into()];
    second.inputs = vec![first.id.clone()];
    vec![first, second]
}

#[test]
fn compact_rank_two_original_token_import_is_sealed_and_feedback_identity() {
    let full = graph(2);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    let mut region = full.clone();
    let mut feedback = full
        .operations
        .iter()
        .find(|op| op.operator == Op::TokenFeedback)
        .unwrap()
        .clone();
    feedback.inputs = vec!["input.tokens".into()];
    feedback.output_shape = vec![1, 5];
    region.query_sequence = 5;
    region.output = feedback.id.clone();
    region.operations = vec![feedback];
    let (mut imports, _) = region_imports(&sources, &norms);
    // Same original compiler token input name/shape; no new feedback input node.
    let owner = imports
        .push(
            Code::Input("input.tokens".into()),
            5,
            Interval::new(0, 2).unwrap(),
        )
        .unwrap();
    let input = imports.view(owner, vec![1, 5]);
    let first = imports.nodes.len();
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    let bound = compact.bind_import("input.tokens", &input).unwrap();
    assert_eq!(bound.logical.dimensions(), input.shape);
    assert_eq!(bound.rank, 2);
    let mut forged = bound.clone();
    forged.logical = ViewShape::new(&[1, 1, 5]).unwrap();
    forged.shape = forged.logical.envelope().unwrap();
    forged.rank = 3;
    assert!(compact.validate_view(&forged).is_err());
    forged = bound.clone();
    forged.import_slot = None;
    assert!(compact.validate_view(&forged).is_err());
    assert!(compact.bind_import(&region.output, &input).is_err());
    let output = compact.lower_next().unwrap();
    assert_eq!(output.logical.dimensions(), input.shape);
    assert!(output.words().eq(input.words.iter()));
    assert_eq!(output.mapping, bound.mapping);
    assert!(Arc::ptr_eq(&output.origin, &bound.origin));
    assert_eq!(compact.next, first);
    assert_eq!(compact.work_budget(), WorkBudget::default());
    assert!(compact.complete_row(&output, 0).unwrap().is_none());
    assert!(compact.bind_import("input.tokens", &input).is_err());
}

#[test]
fn compact_original_rank_four_reshape_roundtrip_has_exact_order_and_no_nodes() {
    let full = lower_with_weights(&config(2), 1, 1, WeightMode::Public)
        .unwrap()
        .decode;
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    for (heads, rows, width) in [(1, 3, 2), (3, 1, 2), (2, 3, 2), (3, 4, 1)] {
        let mut region = full.clone();
        let state = region.state_inputs.first_mut().unwrap();
        let entry = state.id.clone();
        state.shape = vec![1, heads as u64, rows as u64, width as u64];
        region.operations = original_reshape_pair(&full, &entry, heads, rows, width, true);
        region.output = region.operations[1].id.clone();
        // Import-only original Builder fixture, NOT an arithmetic KV-prefix
        // handoff or proof qualification. Production never synthesizes this node.
        let (mut imports, _) = region_imports(&sources, &norms);
        let owner = imports
            .push(
                Code::Input(entry.clone()),
                heads * rows * width,
                Interval::new(-32, 32).unwrap(),
            )
            .unwrap();
        let input = imports.view(owner, vec![1, heads, rows, width]);
        let first = imports.nodes.len();
        let mut compact = CompactGraph::new(
            &imports,
            region_validated(&full, &sources, &norms),
            &region,
            0,
        )
        .unwrap();
        let bound = compact.bind_import(&entry, &input).unwrap();
        let before = compact.container_census().unwrap();
        assert_eq!(
            before.import_binding_bytes,
            CompactGraph::import_binding_storage_bound(&region).unwrap()
        );
        let mut expected = input.clone();
        let mut expanded = Builder::default(); // Reshape recipe does not touch it.
        for op in &region.operations {
            expected = expanded_region_op(&mut expanded, &sources, op, &expected, None);
            let output = compact.lower_next().unwrap();
            assert_eq!(output.logical.dimensions(), expected.shape);
            assert!(output.words().eq(expected.words.iter()));
            assert_eq!(output.bounds, bound.bounds);
            assert_eq!(output.mapping, bound.mapping);
            assert!(Arc::ptr_eq(&output.origin, &bound.origin));
            assert!(compact.row_norm(&output).unwrap().is_none());
            assert!(compact.complete_row(&output, 0).unwrap().is_none());
        }
        assert!(compact.ops[1].output.words().eq(input.words.iter()));
        assert_eq!(compact.ops[1].output.order, LaneOrder::Linear);
        assert_eq!(compact.next, first);
        assert_eq!(compact.work_budget(), WorkBudget::default());
        assert_eq!(compact.owned_native_programs().count(), 0);
        let after = compact.container_census().unwrap();
        assert_eq!(before.descriptor_capacity, after.descriptor_capacity);
        assert_eq!(before.descriptor_bytes, after.descriptor_bytes);
        assert_eq!(before.import_binding_bytes, after.import_binding_bytes);
        assert!(expanded.nodes.is_empty());
        let mut hash = Sha256::new();
        compact.hash_expanded_nodes(&mut hash).unwrap();
        assert_eq!(
            <[u8; 32]>::from(hash.finalize()),
            hash_expanded_binary_nodes(&[])
        );
        for census in compact.descriptor_census() {
            assert!(matches!(census.kind, DescriptorKind::Reshape));
            assert_eq!(census.virtual_start, census.virtual_end);
            assert_eq!(
                census.output.logical_shape.dimensions().len(),
                census.output.semantic_rank as usize
            );
        }
    }
}

#[test]
fn compact_original_rank_two_head_import_retains_original_owner_and_shape() {
    let full = graph(2);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    let region = selection_region(&full);
    let (imports, mut input) = selection_imports(&sources, &norms, 7);
    input.shape = vec![1, 7];
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    let output = compact
        .bind_import(&region.operations[0].inputs[0], &input)
        .unwrap();
    assert_eq!(output.logical.dimensions(), input.shape.as_slice());
    assert!(output.words().eq(input.words.iter()));
    assert_eq!(output.rank, 2);
    assert_eq!(output.bounds, imports.bounds(&input.words).unwrap());
    assert!(compact.complete_row(&output, 0).unwrap().is_none());
    let mut forged = output.clone();
    forged.logical = ViewShape::new(&[1, 1, 7]).unwrap();
    forged.rank = 3;
    assert!(compact.validate_view(&forged).is_err());
    assert!(compact.ops.is_empty());
    assert_eq!(compact.next, imports.nodes.len());
}

#[test]
fn compact_reshaped_rank_four_input_feeds_original_shared_residual_sink() {
    let full = lower_with_weights(&config(2), 1, 1, WeightMode::Public)
        .unwrap()
        .decode;
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    let mut region = full.clone();
    let state = region.state_inputs.first_mut().unwrap();
    let entry = state.id.clone();
    state.shape = vec![1, 2, 3, 2];
    let first = original_reshape_pair(&full, &entry, 2, 3, 2, true).remove(0);
    let mut residual = full
        .operations
        .iter()
        .find(|op| op.operator == Op::ResidualAdd)
        .unwrap()
        .clone();
    residual.inputs = vec![first.id.clone(), first.id.clone()];
    residual.output_shape = vec![1, 3, 4];
    region.output = residual.id.clone();
    region.operations = vec![first, residual];
    let make = || {
        let (mut builder, _) = region_imports(&sources, &norms);
        let owner = builder
            .push(
                Code::Input(entry.clone()),
                12,
                Interval::new(-32, 32).unwrap(),
            )
            .unwrap();
        let view = builder.view(owner, vec![1, 2, 3, 2]);
        (builder, view)
    };
    let (imports, input) = make();
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    compact.bind_import(&entry, &input).unwrap();
    let permuted = compact.lower_next().unwrap();
    assert_ne!(permuted.word(2).unwrap(), input.words.at(2));
    let output = compact.lower_next().unwrap();
    let (mut expanded, input) = make();
    let first = expanded.nodes.len();
    let permuted = expanded_region_op(&mut expanded, &sources, &region.operations[0], &input, None);
    let original = expanded_region_op(
        &mut expanded,
        &sources,
        &region.operations[1],
        &permuted,
        Some(&permuted),
    );
    assert!(output.words().eq(original.words.iter()));
    assert_eq!(
        compact.row_norm(&output).unwrap(),
        expanded.row_norm(&original.words, 4)
    );
    let mut hash = Sha256::new();
    compact.hash_expanded_nodes(&mut hash).unwrap();
    assert_eq!(
        <[u8; 32]>::from(hash.finalize()),
        hash_expanded_binary_nodes(&expanded.nodes[first..])
    );
    assert_eq!(compact.next, expanded.nodes.len());
    assert_eq!(
        compact.work_budget(),
        schedule_work(&expanded.nodes[first..]).unwrap()
    );
    assert_eq!(compact.owned_native_programs().count(), 0);
}

#[test]
fn compact_reshape_does_not_promote_split_rows_and_restores_original_certificate_arc() {
    let full = graph(2);
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    let mut region = full.clone();
    let mut gate = full
        .operations
        .iter()
        .find(|op| op.id.ends_with(".gate_proj"))
        .unwrap()
        .clone();
    gate.output_shape = vec![1, 3, 2];
    let entry = gate.inputs[0].clone();
    let pair = original_reshape_pair(&full, &gate.id, 2, 3, 1, false);
    region.output = pair[1].id.clone();
    region.operations = std::iter::once(gate).chain(pair).collect();
    let (imports, input) = region_import_rows(&sources, &norms, 3);
    let mut compact = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
    )
    .unwrap();
    compact.bind_import(&entry, &input).unwrap();
    let projected = compact.lower_next().unwrap();
    let original = compact.complete_row(&projected, 1).unwrap().unwrap();
    let program = Arc::as_ptr(compact.kernel(0).unwrap().program_at(0).unwrap());
    let work = compact.work_budget();
    let next = compact.next;
    let split = compact.lower_next().unwrap();
    assert_ne!(split.word(1).unwrap(), projected.word(1).unwrap());
    assert!(compact.complete_row(&split, 0).unwrap().is_none());
    assert!(compact.row_norm(&split).unwrap().is_none());
    let restored = compact.lower_next().unwrap();
    let proof = compact.complete_row(&restored, 1).unwrap().unwrap();
    assert_eq!(proof.owner, original.owner);
    assert_eq!(proof.offset, original.offset);
    assert_eq!(proof.width, original.width);
    assert_eq!(proof.energy, original.energy);
    assert_eq!(proof.source_binding, original.source_binding);
    assert!(std::ptr::eq(proof.source, original.source));
    assert!(restored.words().eq(projected.words()));
    assert_eq!(compact.next, next);
    assert_eq!(compact.work_budget(), work);
    assert_eq!(
        Arc::as_ptr(compact.kernel(0).unwrap().program_at(0).unwrap()),
        program
    );
}

#[test]
fn compact_ranked_import_rejects_malformed_reordered_unbound_and_foreign_views() {
    for dimensions in [
        vec![],
        vec![1, 1, 1, 1, 1],
        vec![1, 0],
        vec![usize::MAX, 2],
        vec![MAX_VALUES + 1],
    ] {
        assert!(ViewShape::new(&dimensions).is_err());
    }
    assert!(ViewShape::semantic(&[u64::MAX, 2]).is_err());
    let full = lower_with_weights(&config(2), 1, 1, WeightMode::Public)
        .unwrap()
        .decode;
    let sources = region_sources(2);
    let norms = SourceNorms::new();
    let mut region = full.clone();
    let state = region.state_inputs.first_mut().unwrap();
    let entry = state.id.clone();
    state.shape = vec![1, 2, 3, 2];
    region.operations = original_reshape_pair(&full, &entry, 2, 3, 2, true);
    let (mut imports, _) = region_imports(&sources, &norms);
    let owner = imports
        .push(
            Code::Input(entry.clone()),
            12,
            Interval::new(-32, 32).unwrap(),
        )
        .unwrap();
    let input = imports.view(owner, vec![1, 2, 3, 2]);
    let make = || {
        CompactGraph::new(
            &imports,
            region_validated(&full, &sources, &norms),
            &region,
            0,
        )
        .unwrap()
    };
    let mut compact = make();
    assert!(compact
        .bind_import("not.an.original.semantic.input", &input)
        .is_err());
    let mut bad = input.clone();
    bad.shape = vec![1, 3, 2, 2]; // Same length, wrong declared axes.
    assert!(compact.bind_import(&entry, &bad).is_err());
    bad = input.clone();
    bad.words = (0..input.words.len())
        .rev()
        .map(|i| input.words.at(i))
        .collect();
    assert!(compact.bind_import(&entry, &bad).is_err());
    bad = input.clone();
    bad.shape = vec![1, 2, 2, 2];
    assert!(compact.bind_import(&entry, &bad).is_err());
    assert!(compact.bound_imports.is_empty());
    let bound = compact.bind_import(&entry, &input).unwrap();
    let mut forged = bound.clone();
    forged.import_slot = Some(usize::MAX);
    assert!(compact.validate_view(&forged).is_err());
    forged = bound.clone();
    forged.order = LaneOrder::HeadsSequence {
        heads: 2,
        rows: 3,
        width: 2,
        to_heads: false,
        base_start: 0,
    };
    assert!(compact.validate_view(&forged).is_err());
    let mut other = make();
    let foreign = other.bind_import(&entry, &input).unwrap();
    assert!(compact.validate_view(&foreign).is_err());
    let mut malformed = region.clone();
    malformed.operations[0].output_shape = vec![1, 2, 6]; // Wrong row/hidden axes despite same length.
    let mut rejected = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &malformed,
        0,
    )
    .unwrap();
    rejected.bind_import(&entry, &input).unwrap();
    assert!(rejected.lower_next().is_err());
    assert_eq!(rejected.next, imports.nodes.len());
    assert!(rejected.ops.is_empty());
    let mut numeric = region.clone();
    numeric.operations[0].operator = Op::Silu;
    numeric.operations[0].output_shape = vec![1, 2, 3, 2];
    let mut rejected = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &numeric,
        0,
    )
    .unwrap();
    rejected.bind_import(&entry, &input).unwrap();
    assert!(rejected.lower_next().is_err());
    assert_eq!(rejected.owned_native_programs().count(), 0);
    let mut unknown = region.clone();
    unknown.operations[0].attributes["layout"] = serde_json::json!("flat_identity");
    let mut rejected = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &unknown,
        0,
    )
    .unwrap();
    rejected.bind_import(&entry, &input).unwrap();
    assert!(rejected.lower_next().is_err());
    // Individually legal re-partitioning requires a second non-inverse indexed
    // map. Current bounded contract rejects rather than silently flattening it.
    let mut chained = region.clone();
    chained.operations[1].output_shape = vec![1, 4, 3, 1];
    let mut rejected = CompactGraph::new(
        &imports,
        region_validated(&full, &sources, &norms),
        &chained,
        0,
    )
    .unwrap();
    rejected.bind_import(&entry, &input).unwrap();
    rejected.lower_next().unwrap();
    assert!(rejected.lower_next().is_err());
    assert_eq!(rejected.ops.len(), 1);
    // Initial binding storage is gated before reserving its backing, including
    // the explicit outside owner. This remains a local construction test only.
    let peak = make().construction_peak_bytes();
    let outside = MAX_PROGRAM_BYTES - peak;
    let exact = CompactGraph::with_limit_and_owner(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
        MAX_PROGRAM_BYTES,
        outside,
    )
    .unwrap();
    assert_eq!(exact.construction_peak_bytes(), MAX_PROGRAM_BYTES);
    assert!(CompactGraph::with_limit_and_owner(
        &imports,
        region_validated(&full, &sources, &norms),
        &region,
        0,
        MAX_PROGRAM_BYTES,
        outside + 1
    )
    .is_err());
}

#[test]
fn shared_projection_extraction_keeps_independent_frozen_expanded_identity() {
    // Independently frozen before numeric-sink extractions. Never update to
    // bless changed behavior; the full graph snapshot suite remains unchanged.
    let config = config(2);
    let shapes = Program::source_shapes(&config, 2, 2).unwrap();
    let mut sources = BTreeMap::new();
    for (key, shape) in shapes {
        let values: Vec<i64> = if key == "model.embed_tokens.weight" {
            vec![ONE, ONE, ONE, 0, -ONE, 0]
        } else if key == "lm_head.weight" {
            vec![ONE, 0, -ONE, 0, 0, 0]
        } else if key.contains("norm") {
            vec![ONE; shape.iter().product()]
        } else {
            vec![0; shape.iter().product()]
        };
        let mut hash = Sha256::new();
        for value in &values {
            hash.update(value.to_le_bytes());
        }
        sources.insert(
            key.clone(),
            SourceTensor {
                shape,
                lower: *values.iter().min().unwrap(),
                upper: *values.iter().max().unwrap(),
                public_q18: key.contains("norm").then_some(values),
                digest: hash.finalize().into(),
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
        );
    }
    let mut program = Program::compile(&config, 2, 2, &sources).unwrap();
    program.bind([7; 32], [9; 32]);
    use std::fmt::Write as _;
    let mut digest = String::new();
    for byte in program.digest() {
        write!(&mut digest, "{byte:02x}").unwrap();
    }
    assert_eq!(
        digest,
        "f9b5f1dcc3c06ac87013adca91751f7acae25e5993fae0137b0b90ca98c08eca"
    );
}
