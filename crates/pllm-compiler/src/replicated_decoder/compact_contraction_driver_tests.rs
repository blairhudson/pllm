//! Literal original Builder records versus the shared driver's compact adapter.
//! No executable request, file authentication, row issuer or drain is fabricated.
use super::super::original_import::OriginalImportedView;
use super::*;
use crate::replicated_decoder::source_registry::SourceRegistry;
use compact_admission::{RequestOwners, RequestWork};

struct Fixture {
    plan: pllm_models::DecoderPlan,
    sources: BTreeMap<String, SourceTensor>,
    norms: SourceNorms,
    operations: usize,
    vocabulary: usize,
}

impl Fixture {
    fn new(vocabulary: usize, tokens: usize, repeat: bool) -> Self {
        let config = serde_json::to_vec(&serde_json::json!({
            "model_type":"llama", "hidden_size":2, "intermediate_size":2,
            "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
            "head_dim":2, "vocab_size":vocabulary, "tie_word_embeddings":false,
            "hidden_act":"silu", "attention_bias":false, "max_position_embeddings":64,
            "rope_theta":10000, "rms_norm_eps":0.000001, "attention_dropout":0.0,
            "use_sliding_window":false, "use_cache":true, "max_window_layers":1
        }))
        .unwrap();
        let mut plan = lower_with_weights(&config, tokens, 1, WeightMode::Public).unwrap();
        let shapes =
            Program::source_shapes_with_weights(&config, tokens, 1, WeightMode::Public).unwrap();
        let sources = shapes
            .into_iter()
            .map(|(key, shape)| {
                let public = plan.prefill.operations.iter().any(|op| {
                    op.operator == Op::RmsNorm
                        && op.attributes.get("weight").and_then(|value| value.as_str())
                            == Some(key.as_str())
                });
                let values = public.then(|| vec![0i64; shape.iter().product()]);
                let digest = if let Some(values) = &values {
                    let mut hash = Sha256::new();
                    for value in values {
                        hash.update(value.to_le_bytes());
                    }
                    hash.finalize().into()
                } else {
                    [7; 32]
                };
                (
                    key,
                    SourceTensor {
                        shape,
                        lower: 0,
                        upper: 0,
                        public_q18: values,
                        digest,
                        row_l1: None,
                        row_squared_norm: None,
                        row_l1_squared_sum: None,
                    },
                )
            })
            .collect();
        // A contraction-only qualification prefix, retaining original source
        // validation obligations afterward. This is not a complete decoder.
        let lookup = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::TokenLookup)
            .unwrap()
            .clone();
        let mut linear = plan
            .prefill
            .operations
            .iter()
            .find(|op| {
                op.operator == Op::Linear
                    && op.attributes["weight"]
                        .as_str()
                        .unwrap()
                        .ends_with("q_proj.weight")
            })
            .unwrap()
            .clone();
        let mut last = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::LastToken)
            .unwrap()
            .clone();
        let mut head = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::OutputHead)
            .unwrap()
            .clone();
        let mut prefix = vec![lookup.clone()];
        if repeat {
            let mut again = lookup;
            again.id.push_str(".again");
            prefix.push(again);
        }
        linear.inputs = vec![prefix.last().unwrap().id.clone()];
        prefix.push(linear.clone());
        if repeat {
            let mut again = linear;
            again.id.push_str(".again");
            prefix.push(again);
        }
        last.inputs = vec![prefix.last().unwrap().id.clone()];
        head.inputs = vec![last.id.clone()];
        prefix.push(last);
        prefix.push(head.clone());
        let operations = prefix.len();
        prefix.extend(
            plan.prefill
                .operations
                .iter()
                .filter(|op| op.operator == Op::RmsNorm)
                .cloned(),
        );
        plan.prefill.operations = prefix;
        plan.prefill.output = head.id;
        Self {
            plan,
            sources,
            norms: SourceNorms::new(),
            operations,
            vocabulary,
        }
    }

    fn validated(&self) -> ValidatedSourceDescriptors<'_> {
        let shapes = self
            .sources
            .iter()
            .map(|(key, source)| (key.clone(), source.shape.clone()))
            .collect();
        validate_source_records(
            &self.plan.prefill,
            &shapes,
            &self.sources,
            WeightMode::Public,
            ExecutionPolicy::default(),
            &self.norms,
        )
        .unwrap()
    }
    fn registry(&self) -> SourceRegistry<'_> {
        SourceRegistry::from_validated(self.validated()).unwrap()
    }
    fn imports(&self) -> Builder {
        let mut builder = Builder {
            public_weights: true,
            ..Builder::default()
        };
        for (key, source) in self
            .sources
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
        builder
    }
    fn declare<'a>(
        &'a self,
        builder: &mut Builder,
        registry: &'a SourceRegistry<'a>,
        prior: &RequestWork,
        owners: &RequestOwners<'_>,
    ) -> OriginalImportedView<'a> {
        builder
            .declare_original_input(
                registry,
                &self.plan,
                &self.plan.prefill,
                "input.tokens",
                Interval::new(0, self.vocabulary as i64 - 1).unwrap(),
                prior,
                owners,
                MAX_PROGRAM_BYTES,
            )
            .unwrap()
    }
}

fn owners(values: &Vec<Option<Secret>>) -> RequestOwners<'_> {
    crate::replicated_decoder::program::original_import::tests::owners(values)
}

/// Literal original arms: no call through ContractionSink or compact helpers.
fn original(
    builder: &mut Builder,
    op: &pllm_models::ModelOperation,
    input: Option<&View>,
    tokens: &View,
    sources: &BTreeMap<String, SourceView>,
    vocabulary: usize,
) -> View {
    let shape = op
        .output_shape
        .iter()
        .map(|dimension| *dimension as usize)
        .collect::<Vec<_>>();
    if op.operator == Op::LastToken {
        let input = input.unwrap();
        let width = *shape.last().unwrap();
        return View {
            words: input.words.slice(input.words.len() - width..),
            shape,
        };
    }
    let table = &sources[op.attributes["weight"].as_str().unwrap()];
    if op.operator != Op::TokenLookup {
        let mut output = builder
            .matrix(
                input.unwrap(),
                MatrixView::source(table, false).unwrap(),
                true,
            )
            .unwrap();
        output.shape = shape;
        builder
            .projection_energies(input.unwrap(), &output, table, None)
            .unwrap();
        return output;
    }
    let mut words = Words::new();
    let table_bounds = builder.nodes[table.owner].bounds;
    for (owner, index) in tokens.words.iter() {
        let bounds = builder.nodes[owner].bounds;
        let offsets = Interval::new(
            bounds.lower().checked_sub((vocabulary - 1) as i64).unwrap(),
            bounds.upper(),
        )
        .unwrap();
        let equal = builder
            .tensor_loop(
                LoopInput::TokenOffsets { owner, index },
                vocabulary,
                |lanes| tensor::Program::equal_zero(lanes, offsets),
            )
            .unwrap();
        let selectors = builder.view(equal, vec![1, vocabulary]);
        let row = builder
            .matrix(&selectors, MatrixView::source(table, true).unwrap(), false)
            .unwrap();
        for (owner, _) in row.words.iter() {
            builder.nodes[owner].bounds = table_bounds;
        }
        words.extend(row.words);
    }
    View { words, shape }
}

fn hash_original(nodes: &[Node]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for node in nodes {
        hash_node_header(&mut hash, node.lanes, node.bounds);
        match &node.code {
            Code::TensorLoop {
                input,
                program,
                tail,
            } => hash_tensor_loop(&mut hash, *input, program, tail.as_ref()),
            Code::Gather(words) => hash_gather(&mut hash, words.iter()),
            Code::PublicProduct {
                input,
                weight,
                rows,
                inner,
                columns,
                transposed,
            } => hash_public_product(
                &mut hash,
                *input,
                *weight,
                *rows,
                *inner,
                *columns,
                *transposed,
            ),
            _ => panic!("unexpected literal contraction record"),
        }
    }
    hash.finalize().into()
}

#[test]
fn sealed_shared_adapter_matches_literal_full_tail_ids_hash_work_source_and_zero_aliases() {
    for vocabulary in [1, 127, 128, 129, 257] {
        for tokens in [1, 3] {
            let fixture = Fixture::new(vocabulary, tokens, false);
            let registry = fixture.registry();
            let values = Vec::new();
            let outside = owners(&values);
            let prior = RequestWork::default();
            let mut imports = fixture.imports();
            let bindings = vec![fixture.declare(&mut imports, &registry, &prior, &outside)];
            let mut compact = CompactGraph::with_bound_inputs(
                &imports,
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
            let mut expanded = fixture.imports();
            let token_owner = expanded
                .push(
                    Code::Input("input.tokens".into()),
                    tokens,
                    Interval::new(0, vocabulary as i64 - 1).unwrap(),
                )
                .unwrap();
            assert_eq!(token_owner, bindings[0].owner());
            let tokens_view = expanded.view(token_owner, vec![1, tokens]);
            let source_views = fixture
                .sources
                .iter()
                .filter(|(_, source)| source.public_q18.is_none())
                .enumerate()
                .map(|(owner, (key, source))| {
                    (
                        key.clone(),
                        SourceView {
                            owner,
                            shape: source.shape.clone(),
                        },
                    )
                })
                .collect();
            let first = expanded.nodes.len();
            let mut frontier = BTreeMap::from([("input.tokens".to_owned(), tokens_view.clone())]);
            for (index, op) in fixture.plan.prefill.operations[..fixture.operations]
                .iter()
                .enumerate()
            {
                let old_nodes = expanded.nodes.len();
                let old_work = compact.work_budget();
                let output = original(
                    &mut expanded,
                    op,
                    frontier.get(&op.inputs[0]),
                    &tokens_view,
                    &source_views,
                    vocabulary,
                );
                let published = compact.lower_next().unwrap();
                assert!(published.words().eq(output.words.iter()));
                assert_eq!(published.bounds, expanded.bounds(&output.words).unwrap());
                assert_eq!(published.logical.dimensions(), output.shape.as_slice());
                assert_eq!(compact.next, expanded.nodes.len());
                assert_eq!(
                    compact.work_budget(),
                    schedule_work(&expanded.nodes[first..]).unwrap()
                );
                let mut hash = Sha256::new();
                compact.hash_expanded_nodes(&mut hash).unwrap();
                assert_eq!(
                    <[u8; 32]>::from(hash.finalize()),
                    hash_original(&expanded.nodes[first..])
                );
                if op.operator == Op::LastToken {
                    assert_eq!(old_nodes, expanded.nodes.len());
                    assert_eq!(old_work, compact.work_budget());
                    assert!(matches!(
                        compact.ops[index].kind,
                        PublishedKind::LastToken { .. }
                    ));
                }
                if matches!(op.operator, Op::TokenLookup | Op::Linear | Op::OutputHead) {
                    let key = op.attributes["weight"].as_str().unwrap();
                    let (owner, source) = match &compact.ops[index].kind {
                        PublishedKind::TokenLookup { weight, source, .. }
                        | PublishedKind::Projection { weight, source, .. } => (*weight, *source),
                        _ => panic!("shared contraction publication"),
                    };
                    assert!(std::ptr::eq(source, &fixture.sources[key]));
                    assert_eq!(owner, source_views[key].owner);
                    assert!(
                        matches!(&imports.nodes[owner].code, Code::PublicInput { key: original, .. } if original == key)
                    );
                }
                frontier.insert(op.id.clone(), output);
            }
            assert_eq!(compact.ops.len(), fixture.operations);
            assert!(compact.ops.len() < expanded.nodes.len() - first);
            assert!(
                imports
                    .nodes
                    .iter()
                    .all(|node| matches!(node.code, Code::Input(_) | Code::PublicInput { .. }))
            );
        }
    }
}

#[test]
fn repeated_shared_calls_charge_work_again_but_keep_original_closed_arcs() {
    let fixture = Fixture::new(129, 3, true);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imports = fixture.imports();
    let bindings = vec![fixture.declare(&mut imports, &registry, &prior, &outside)];
    let mut graph = CompactGraph::with_bound_inputs(
        &imports,
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
    let first_work = graph.work_budget();
    let attempts = LOOKUP_NATIVE_ATTEMPTS.with(|n| n.get());
    graph.lower_next().unwrap();
    assert_eq!(graph.work_budget(), first_work.checked_mul(2).unwrap());
    assert_eq!(LOOKUP_NATIVE_ATTEMPTS.with(|n| n.get()), attempts);
    let (
        PublishedKind::TokenLookup { plan: first, .. },
        PublishedKind::TokenLookup { plan: second, .. },
    ) = (&graph.ops[0].kind, &graph.ops[1].kind)
    else {
        panic!("lookup owners")
    };
    assert!(Arc::ptr_eq(&first.full.program, &second.full.program));
    assert!(Arc::ptr_eq(
        &first.tail.as_ref().unwrap().program,
        &second.tail.as_ref().unwrap().program
    ));
    graph.lower_next().unwrap();
    let attempts = super::super::super::prepared_tensor_map::native_attempts();
    graph.lower_next().unwrap();
    assert_eq!(
        super::super::super::prepared_tensor_map::native_attempts(),
        attempts
    );
    assert!(matches!(
        graph.ops[3].prepared.as_ref().unwrap().storage,
        MapStorage::Previous(2)
    ));
    assert!(std::ptr::eq(
        graph.kernel(2).unwrap(),
        graph.kernel(3).unwrap()
    ));
    assert_eq!(graph.contraction_control_bytes.get(), 0);
}

#[test]
fn pending_native_work_rejects_individually_fitting_outside_ledger_before_containers() {
    let fixture = Fixture::new(129, 3, false);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imports = fixture.imports();
    let bindings = vec![fixture.declare(&mut imports, &registry, &prior, &outside)];
    let offsets = Interval::new(-128, 128).unwrap();
    let lookup = tensor::Program::equal_zero_work(128, offsets)
        .unwrap()
        .checked_add(tensor::Program::equal_zero_work(1, offsets).unwrap())
        .unwrap()
        .checked_mul(3)
        .unwrap();
    let floor = WorkBudget::rescale(6)
        .unwrap()
        .checked_add(WorkBudget::rescale(128).unwrap())
        .unwrap()
        .checked_add(WorkBudget::rescale(1).unwrap())
        .unwrap();
    let pending = lookup.checked_add(floor).unwrap();
    assert_eq!(
        pending_contraction_work(&imports, &fixture.plan.prefill, 0, &fixture.sources).unwrap(),
        pending
    );
    let mut pressure = owners(&values);
    pressure.other_work = WorkBudget::local()
        .checked_mul(puma::MAX_PROGRAM_OPERATIONS - pending.operations() + 1)
        .unwrap();
    pressure.other_work.limits().unwrap();
    assert!(
        pressure
            .other_work
            .checked_add(pending)
            .unwrap()
            .limits()
            .is_err()
    );
    // Equality alone fits: the independently approved floor contracts are what
    // push this whole retained request over its ordinal ceiling.
    pressure
        .other_work
        .checked_add(lookup)
        .unwrap()
        .limits()
        .unwrap();
    let containers = input_owner::CONTAINER_FACTORIES.with(|n| n.get());
    let native = LOOKUP_NATIVE_ATTEMPTS.with(|n| n.get());
    let maps = super::super::super::prepared_tensor_map::native_attempts();
    assert!(matches!(
        CompactGraph::with_bound_inputs(
            &imports,
            fixture.validated(),
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &bindings,
            &prior,
            &pressure,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(
        input_owner::CONTAINER_FACTORIES.with(|n| n.get()),
        containers
    );
    assert_eq!(LOOKUP_NATIVE_ATTEMPTS.with(|n| n.get()), native);
    assert_eq!(
        super::super::super::prepared_tensor_map::native_attempts(),
        maps
    );
}

#[test]
fn full_and_tail_construction_peak_refuses_one_short_before_first_native_factory() {
    let fixture = Fixture::new(129, 3, false);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imports = fixture.imports();
    let bindings = vec![fixture.declare(&mut imports, &registry, &prior, &outside)];
    let construct = |cap| {
        CompactGraph::with_bound_inputs(
            &imports,
            fixture.validated(),
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &bindings,
            &prior,
            &outside,
            cap,
        )
    };
    let mut observed = construct(MAX_PROGRAM_BYTES).unwrap();
    let initial = observed.construction_peak_bytes();
    observed.lower_next().unwrap();
    let peak = observed.construction_peak_bytes();
    assert!(peak > initial);
    drop(observed);
    let mut short = construct(peak - 1).unwrap();
    let native = LOOKUP_NATIVE_ATTEMPTS.with(|n| n.get());
    assert!(matches!(short.lower_next(), Err(Error::Bounds)));
    assert_eq!(LOOKUP_NATIVE_ATTEMPTS.with(|n| n.get()), native);
    assert_eq!(short.contraction_control_bytes.get(), 0);
    assert!(short.ops.is_empty());
    assert!(matches!(short.lower_next(), Err(Error::Retired)));
}

#[test]
fn cloned_same_digest_source_is_not_a_compact_contraction_owner() {
    let fixture = Fixture::new(3, 1, false);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imports = fixture.imports();
    let bindings = vec![fixture.declare(&mut imports, &registry, &prior, &outside)];
    let graph = CompactGraph::with_bound_inputs(
        &imports,
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
    let input = &graph.bound_imports[0].view;
    let sink = CompactContractionSink {
        graph: &graph,
        input,
        declared: input.logical,
        lookup_source: std::cell::Cell::new(None),
        lookup: None,
        projection: None,
    };
    let key = fixture.plan.prefill.operations[0].attributes["weight"]
        .as_str()
        .unwrap();
    let cloned = fixture.sources[key].clone();
    assert_eq!(cloned.digest, fixture.sources[key].digest);
    assert!(matches!(sink.source(&cloned), Err(Error::Binding)));
    assert!(sink.source(&fixture.sources[key]).is_ok());
}

#[test]
fn shared_lookup_uses_genuine_global_tokens_not_first_semantic_input() {
    let mut fixture = Fixture::new(3, 2, true);
    fixture.plan.prefill.operations[0].inputs = vec!["ignored.semantic.input".into()];
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imports = fixture.imports();
    let bindings = vec![fixture.declare(&mut imports, &registry, &prior, &outside)];
    let mut graph = CompactGraph::with_bound_inputs(
        &imports,
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
    assert_eq!(
        graph.ops[0].input.word(0).unwrap(),
        (bindings[0].owner(), 0)
    );
    assert_eq!(
        graph.ops[0].input.word(1).unwrap(),
        (bindings[0].owner(), 1)
    );
    assert!(graph.frontier.value("ignored.semantic.input").is_none());
}

#[test]
fn declared_projection_geometry_refuses_before_rescale_factory_or_publication() {
    let mut fixture = Fixture::new(3, 2, false);
    fixture.plan.prefill.operations[1].output_shape = vec![1, 2, 3];
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imports = fixture.imports();
    let bindings = vec![fixture.declare(&mut imports, &registry, &prior, &outside)];
    let mut graph = CompactGraph::with_bound_inputs(
        &imports,
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
    let attempts = super::super::super::prepared_tensor_map::native_attempts();
    let before = (graph.ops.len(), graph.next, graph.work_budget());
    assert!(matches!(graph.lower_next(), Err(Error::Shape)));
    assert_eq!(
        super::super::super::prepared_tensor_map::native_attempts(),
        attempts
    );
    assert_eq!((graph.ops.len(), graph.next, graph.work_budget()), before);
    assert_eq!(graph.contraction_control_bytes.get(), 0);
}
