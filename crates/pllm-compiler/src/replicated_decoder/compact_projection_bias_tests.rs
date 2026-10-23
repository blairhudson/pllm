//! Original Qwen projection/source contracts, not a complete request or executor.
use super::super::original_import::OriginalImportedView;
use super::*;
use crate::replicated_decoder::{
    program::schedule_finalization as shared, source_registry::SourceRegistry,
};
use compact_admission::{RequestOwners, RequestWork};

struct Fixture {
    config: Vec<u8>,
    plan: pllm_models::DecoderPlan,
    sources: BTreeMap<String, SourceTensor>,
    norms: SourceNorms,
    width: usize,
}

fn values(key: &str, lanes: usize) -> Vec<i64> {
    if key.ends_with(".bias") {
        (0..lanes)
            .map(|i| [ONE / 32, -ONE / 16, 7][i % 3])
            .collect()
    } else if key.contains("norm") {
        vec![ONE; lanes]
    } else {
        vec![ONE / 64; lanes]
    }
}

impl Fixture {
    fn new(rows: usize, width: usize, certificates: bool) -> Self {
        let config = serde_json::to_vec(&serde_json::json!({
            "model_type":"qwen2", "hidden_size":width, "intermediate_size":width,
            "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
            "head_dim":width, "vocab_size":3, "tie_word_embeddings":false,
            "hidden_act":"silu", "attention_bias":true, "max_position_embeddings":64,
            "rope_theta":10000, "rms_norm_eps":0.000001, "attention_dropout":0.0,
            "use_sliding_window":false, "use_cache":true, "max_window_layers":1
        }))
        .unwrap();
        let mut plan = lower_with_weights(&config, rows, 1, WeightMode::Public).unwrap();
        let lookup = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::TokenLookup)
            .unwrap()
            .clone();
        let mut gate = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.id.ends_with(".gate_proj"))
            .unwrap()
            .clone();
        gate.inputs = vec![lookup.id.clone()];
        let mut prefix = vec![lookup, gate.clone()];
        for suffix in ["q_proj.weight", "k_proj.weight", "v_proj.weight"] {
            let mut op = plan
                .prefill
                .operations
                .iter()
                .find(|op| {
                    op.operator == Op::Linear
                        && op.attributes["weight"].as_str().unwrap().ends_with(suffix)
                })
                .unwrap()
                .clone();
            assert!(op.attributes["bias"].as_str().unwrap().ends_with(".bias"));
            op.inputs = vec![gate.id.clone()];
            prefix.push(op);
        }
        // Keep original source/certificate obligations and model metadata. Only
        // the first five genuine operators are qualified/executed by these tests.
        prefix.extend(
            plan.prefill
                .operations
                .iter()
                .filter(|op| !prefix_id(op, &gate.id))
                .cloned(),
        );
        plan.prefill.operations = prefix;
        plan.prefill.output = plan.prefill.operations[4].id.clone();
        let sources: BTreeMap<_, _> =
            Program::source_shapes_with_weights(&config, rows, 1, WeightMode::Public)
                .unwrap()
                .into_iter()
                .map(|(key, shape)| {
                    let data = values(&key, shape.iter().product());
                    let public = key.contains("norm");
                    let mut hash = Sha256::new();
                    for value in &data {
                        hash.update(value.to_le_bytes());
                    }
                    let matrix = shape.len() == 2;
                    let row_l1 = matrix.then(|| shape[1] as u64 * (ONE / 64) as u64);
                    (
                        key,
                        SourceTensor {
                            lower: *data.iter().min().unwrap(),
                            upper: *data.iter().max().unwrap(),
                            public_q18: public.then_some(data),
                            digest: hash.finalize().into(),
                            row_l1,
                            row_squared_norm: matrix.then(|| {
                                shape[1] as u128 * (ONE / 64) as u128 * (ONE / 64) as u128
                            }),
                            row_l1_squared_sum: row_l1
                                .map(|row| shape[0] as u128 * u128::from(row).pow(2)),
                            shape,
                        },
                    )
                })
                .collect();
        let mut norms = SourceNorms::new();
        if certificates {
            for (key, (rows, columns, width)) in source_norms::requests(&plan.prefill).unwrap() {
                let mut scan = source_norms::BlockScan::new(rows, columns, width).unwrap();
                scan.append(0, &values(&key, rows * columns)).unwrap();
                norms.insert(key, scan.finish().unwrap());
            }
        }
        Self {
            config,
            plan,
            sources,
            norms,
            width,
        }
    }
    fn execution(&self) -> ExecutionPolicy {
        ExecutionPolicy {
            range_policy: if self.norms.is_empty() {
                RangePolicy::Static
            } else {
                RangePolicy::Checked
            },
            ..ExecutionPolicy::default()
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
            self.execution(),
            &self.norms,
        )
        .unwrap()
    }
    fn imports(&self) -> Builder {
        let mut builder = Builder {
            public_weights: true,
            checked_bound: self.execution().checked_bound(),
            allocated: source_norms::import_envelope(&self.norms)
                .unwrap()
                .total_bytes,
            ..Builder::default()
        };
        for (key, source) in self
            .sources
            .iter()
            .filter(|(_, source)| source.public_q18.is_none())
        {
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
            if let Some(norms) = self.norms.get(key) {
                builder.public_norms.insert(owner, norms.clone());
            }
        }
        builder
    }
    fn declare<'a>(
        &'a self,
        imports: &mut Builder,
        registry: &'a SourceRegistry<'a>,
        prior: &RequestWork,
        owners: &RequestOwners<'_>,
    ) -> OriginalImportedView<'a> {
        imports
            .declare_original_input(
                registry,
                &self.plan,
                &self.plan.prefill,
                "input.tokens",
                Interval::new(0, 2).unwrap(),
                prior,
                owners,
                MAX_PROGRAM_BYTES,
            )
            .unwrap()
    }
}

fn prefix_id(op: &pllm_models::ModelOperation, gate: &str) -> bool {
    op.operator == Op::TokenLookup
        || op.id == gate
        || (op.operator == Op::Linear
            && ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
                .iter()
                .any(|suffix| op.attributes["weight"].as_str().unwrap().ends_with(suffix)))
}

fn owners(values: &Vec<Option<Secret>>) -> RequestOwners<'_> {
    crate::replicated_decoder::program::original_import::tests::owners(values)
}

fn source_owner(builder: &Builder, key: &str) -> usize {
    builder
        .nodes
        .iter()
        .position(
            |node| matches!(&node.code, Code::PublicInput { key: original, .. } if original == key),
        )
        .unwrap()
}

// Literal original arm composition; not the shared driver or compact recipe.
fn original(
    builder: &mut Builder,
    fixture: &Fixture,
    op: &pllm_models::ModelOperation,
    input: &View,
) -> View {
    let key = op.attributes["weight"].as_str().unwrap();
    let table = SourceView {
        owner: source_owner(builder, key),
        shape: fixture.sources[key].shape.clone(),
    };
    let shape = op
        .output_shape
        .iter()
        .map(|dimension| *dimension as usize)
        .collect();
    if op.operator == Op::TokenLookup {
        let mut words = Words::new();
        for (owner, index) in input.words.iter() {
            let offsets = Interval::new(
                builder.nodes[owner].bounds.lower() - 2,
                builder.nodes[owner].bounds.upper(),
            )
            .unwrap();
            let equal = builder
                .tensor_loop(LoopInput::TokenOffsets { owner, index }, 3, |lanes| {
                    tensor::Program::equal_zero(lanes, offsets)
                })
                .unwrap();
            let selectors = builder.view(equal, vec![1, 3]);
            let row = builder
                .matrix(&selectors, MatrixView::source(&table, true).unwrap(), false)
                .unwrap();
            for (owner, _) in row.words.iter() {
                builder.nodes[owner].bounds = builder.nodes[table.owner].bounds;
            }
            words.extend(row.words);
        }
        return View { words, shape };
    }
    let mut output = builder
        .matrix(input, MatrixView::source(&table, false).unwrap(), true)
        .unwrap();
    let bias = op
        .attributes
        .get("bias")
        .and_then(|value| value.as_str())
        .map(|key| SourceView {
            owner: source_owner(builder, key),
            shape: fixture.sources[key].shape.clone(),
        });
    if let Some(bias) = &bias {
        assert_eq!(bias.shape, [fixture.width]);
        let repeated = (0..output.words.len())
            .map(|index| (bias.owner, index % bias.shape[0]))
            .collect();
        output.words = builder.binary(&output.words, &repeated, false).unwrap();
    }
    output.shape = shape;
    builder
        .projection_energies(input, &output, &table, bias.as_ref())
        .unwrap();
    output
}

fn original_hash(nodes: &[Node]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for node in nodes {
        hash_node_header(&mut hash, node.lanes, node.bounds);
        match &node.code {
            Code::Gather(words) => hash_gather(&mut hash, words.iter()),
            Code::TensorLoop {
                input,
                program,
                tail,
            } => hash_tensor_loop(&mut hash, *input, program, tail.as_ref()),
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
            Code::Affine {
                left,
                right,
                coefficients,
            } => hash_affine(&mut hash, *left, *right, *coefficients),
            _ => panic!("unexpected original projection record"),
        }
    }
    hash.finalize().into()
}

#[test]
fn qwen_qkv_signed_bias_matches_original_records_rows_tile_crossings_and_floor_work() {
    for width in [2, 130] {
        for rows in [1, 3] {
            let fixture = Fixture::new(rows, width, false);
            let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
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
            let mut expanded = fixture.imports();
            let token = expanded
                .push(
                    Code::Input("input.tokens".into()),
                    rows,
                    Interval::new(0, 2).unwrap(),
                )
                .unwrap();
            let token_view = expanded.view(token, vec![1, rows]);
            let first = expanded.nodes.len();
            let mut frontier = BTreeMap::from([("input.tokens".into(), token_view)]);
            for (slot, op) in fixture.plan.prefill.operations[..5].iter().enumerate() {
                let reference = original(&mut expanded, &fixture, op, &frontier[&op.inputs[0]]);
                let actual = graph.lower_next().unwrap();
                assert!(actual.words().eq(reference.words.iter()));
                assert_eq!(actual.bounds, expanded.bounds(&reference.words).unwrap());
                assert_eq!(actual.logical.dimensions(), reference.shape.as_slice());
                assert_eq!(graph.next, expanded.nodes.len());
                assert_eq!(
                    graph.work_budget(),
                    schedule_work(&expanded.nodes[first..]).unwrap()
                );
                let mut hash = Sha256::new();
                graph.hash_expanded_nodes(&mut hash).unwrap();
                assert_eq!(
                    <[u8; 32]>::from(hash.finalize()),
                    original_hash(&expanded.nodes[first..])
                );
                if slot >= 2 {
                    let PublishedKind::Projection {
                        bias: Some(bias),
                        row_energy,
                        ..
                    } = &graph.ops[slot].kind
                    else {
                        panic!("real bias recipe")
                    };
                    let key = op.attributes["bias"].as_str().unwrap();
                    assert!(std::ptr::eq(bias.source, &fixture.sources[key]));
                    assert_eq!(bias.owner, source_owner(&expanded, key));
                    assert_eq!(actual.mapping, Mapping::Products(graph.ops[slot].first + 3));
                    assert!(row_energy.is_none());
                    assert!(graph.row_norm(&actual).unwrap().is_none());
                    let mut body = crate::replicated_decoder::runtime::body::layer::Builder::new(
                        WeightMode::Public,
                        &[crate::replicated_decoder::runtime::body::Register {
                            lanes: rows * width,
                            bounds: bias.floor_bounds,
                        }],
                    )
                    .unwrap();
                    let input = body.input(0).unwrap();
                    let source = crate::replicated_decoder::runtime::body::layer::Source::new(
                        bias.owner,
                        key.to_owned(),
                        bias.source.digest,
                        width,
                        bias.bounds().unwrap(),
                    )
                    .unwrap();
                    let output = body.bias(&input, source, width).unwrap();
                    let body = body.finish(&[output]).unwrap();
                    assert_eq!(body.work_budget(), WorkBudget::default());
                    assert_eq!(body.statistics().public_bias_vector_reads, 1);
                }
                frontier.insert(op.id.clone(), reference);
            }
        }
    }
}

#[test]
fn bias_keeps_attention_source_certificate_separate_from_raw_general_row() {
    let fixture = Fixture::new(3, 2, true);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
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
    let mut expanded = fixture.imports();
    let token = expanded
        .push(
            Code::Input("input.tokens".into()),
            3,
            Interval::new(0, 2).unwrap(),
        )
        .unwrap();
    let mut reference = expanded.view(token, vec![1, 3]);
    for op in &fixture.plan.prefill.operations[..3] {
        reference = original(&mut expanded, &fixture, op, &reference);
    }
    graph.lower_next().unwrap();
    let gate = graph.lower_next().unwrap();
    assert!(graph.complete_row(&gate, 0).unwrap().is_some());
    let query = graph.lower_next().unwrap();
    assert!(expanded.row_norm(&reference.words, 2).is_none());
    assert!(graph.complete_row(&query, 0).unwrap().is_none());
    assert_eq!(
        graph.attention_energy(&query, 0, 2).unwrap(),
        expanded.attention_energy(&reference.words, 2)
    );
    assert!(graph.attention_energy(&query, 0, 2).unwrap().is_some());
}

#[test]
fn frozen_bias_retains_original_sources_arcs_and_exact_canonical_recipe() {
    let fixture = Fixture::new(3, 130, false);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
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
    let mut expanded = fixture.imports();
    let token = expanded
        .push(
            Code::Input("input.tokens".into()),
            3,
            Interval::new(0, 2).unwrap(),
        )
        .unwrap();
    let mut frontier = BTreeMap::from([("input.tokens".into(), expanded.view(token, vec![1, 3]))]);
    let mut exports = Vec::new();
    let mut selected = Vec::new();
    for (slot, op) in fixture.plan.prefill.operations[..5].iter().enumerate() {
        let output = original(&mut expanded, &fixture, op, &frontier[&op.inputs[0]]);
        let actual = graph.lower_next().unwrap();
        if slot >= 2 {
            exports.push(actual);
            selected.extend(output.words.iter());
        }
        frontier.insert(op.id.clone(), output);
    }
    let weak = Arc::downgrade(graph.kernel(2).unwrap().program_at(0).unwrap());
    let frozen = graph.freeze(exports, MAX_PROGRAM_BYTES).unwrap();
    for descriptor in frozen.descriptor_census().skip(2) {
        let DescriptorKind::Projection {
            bias: Some(bias), ..
        } = descriptor.kind
        else {
            panic!("frozen bias")
        };
        let Code::PublicInput { key, .. } =
            &frozen.imported_recipe(bias.source_owner).unwrap().code
        else {
            panic!("original bias import")
        };
        assert!(std::ptr::eq(
            &frozen.borrowed_source_records()[key],
            &fixture.sources[key]
        ));
    }
    let quote = frozen.schedule_recipe_quote().unwrap();
    assert!(matches!(
        frozen.borrow_schedule(quote.peak_bytes - 1),
        Err(Error::Bounds)
    ));
    let recipe = frozen.borrow_schedule(quote.peak_bytes).unwrap();
    let baseline = recipe.resources().finalization_baseline_bytes().unwrap();
    let actual = shared::quote(&recipe, &selected, baseline).unwrap();
    let expected = shared::quote(expanded.nodes.as_slice(), &selected, 0).unwrap();
    let actual = shared::finalize(
        &recipe,
        &selected,
        &Sha256::new(),
        baseline,
        actual.peak_bytes,
    )
    .unwrap();
    let expected = shared::finalize(
        expanded.nodes.as_slice(),
        &selected,
        &Sha256::new(),
        0,
        expected.peak_bytes,
    )
    .unwrap();
    assert_eq!(actual.digest, expected.digest);
    assert_eq!(actual.work, expected.work);
    assert_eq!(actual.retire, expected.retire);
    assert!(weak.upgrade().is_some());
    drop(recipe);
    drop(frozen);
    assert!(weak.upgrade().is_none());
}

#[test]
fn same_digest_foreign_bias_source_refuses_frozen_emission_before_control_allocation() {
    let fixture = Fixture::new(1, 2, false);
    let key = fixture.plan.prefill.operations[2].attributes["bias"]
        .as_str()
        .unwrap();
    let foreign = fixture.sources[key].clone();
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
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
    graph.lower_next().unwrap();
    let output = graph.lower_next().unwrap();
    let PublishedKind::Projection {
        bias: Some(bias), ..
    } = &mut graph.ops[2].kind
    else {
        panic!("bias")
    };
    assert_eq!(bias.source.digest, foreign.digest);
    assert_eq!(bias.source.shape, foreign.shape);
    bias.source = &foreign;
    let frozen = graph.freeze(vec![output], MAX_PROGRAM_BYTES).unwrap();
    assert!(matches!(
        frozen.borrow_schedule(MAX_PROGRAM_BYTES),
        Err(Error::Binding)
    ));
}

#[test]
fn bias_error_order_is_raw_factory_then_binding_shape_then_addition() {
    for shape_error in [false, true] {
        let mut fixture = Fixture::new(1, 2, false);
        if shape_error {
            fixture.plan.prefill.operations[2].output_shape = vec![1, 1, 3];
        } else {
            fixture.plan.prefill.operations[2]
                .attributes
                .as_object_mut()
                .unwrap()
                .insert(
                    "bias".into(),
                    serde_json::Value::String("missing.bias".into()),
                );
        }
        let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
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
        graph.lower_next().unwrap();
        let native = super::super::super::prepared_tensor_map::native_attempts();
        let bias = BIAS_RECIPE_ATTEMPTS.with(|count| count.get());
        let result = graph.lower_next();
        if shape_error {
            assert!(matches!(result, Err(Error::Shape)));
        } else {
            assert!(matches!(result, Err(Error::Binding)));
        }
        assert!(super::super::super::prepared_tensor_map::native_attempts() > native);
        assert_eq!(BIAS_RECIPE_ATTEMPTS.with(|count| count.get()), bias);
        assert_eq!(graph.ops.len(), 2);
        assert_eq!(graph.contraction_control_bytes.get(), 0);
    }
}

#[test]
fn projection_bias_exact_cap_and_one_short_gate_before_failing_factory_or_recipe() {
    let fixture = Fixture::new(3, 130, false);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
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
    for _ in 0..5 {
        observed.lower_next().unwrap();
    }
    let peak = observed.construction_peak_bytes();
    drop(observed);
    let mut exact = construct(peak).unwrap();
    for _ in 0..5 {
        exact.lower_next().unwrap();
    }
    drop(exact);
    match construct(peak - 1) {
        Err(Error::Bounds) => (),
        Ok(mut short) => {
            let mut refused = false;
            for _ in 0..5 {
                let before = (
                    LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()),
                    super::super::super::prepared_tensor_map::native_attempts(),
                    BIAS_RECIPE_ATTEMPTS.with(|count| count.get()),
                );
                match short.lower_next() {
                    Err(Error::Bounds) => {
                        assert_eq!(
                            (
                                LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()),
                                super::super::super::prepared_tensor_map::native_attempts(),
                                BIAS_RECIPE_ATTEMPTS.with(|count| count.get())
                            ),
                            before
                        );
                        refused = true;
                        break;
                    }
                    Ok(_) => (),
                    Err(_) => panic!("wrong refusal"),
                }
            }
            assert!(refused);
        }
        Err(_) => panic!("wrong construction refusal"),
    }
}

#[test]
fn signed_bias_source_uses_original_float_import_quantization_and_commitment() {
    let fixture = Fixture::new(1, 2, false);
    let manifest = Manifest {
        schema: "pllm.replicated-decoder.manifest.v1".into(),
        weights: WeightMode::Public,
        execution: fixture.execution(),
        config: fixture.config.clone(),
        rows: 1,
        outputs: 1,
        source: [1; 32],
        composition: [2; 32],
        session: [3; 32],
        tensors: fixture.sources.clone(),
        norms: fixture.norms.clone(),
        program: [0; 32],
    };
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let mut import = PublicSourceImport::new(&manifest).unwrap();
    for (key, source) in &fixture.sources {
        let bytes = values(key, source.shape.iter().product())
            .into_iter()
            .flat_map(|value| (value as f32 / ONE as f32).to_le_bytes())
            .collect::<Vec<_>>();
        import.append(key, 0, &bytes).unwrap();
    }
    import.finish().unwrap().check_registry(&registry).unwrap();
}

#[test]
fn biased_raw_full_tail_work_rejects_aggregate_before_containers_and_all_factories() {
    let mut fixture = Fixture::new(3, 130, false);
    let prefix = fixture.plan.prefill.operations[..5]
        .iter()
        .map(|op| op.id.clone())
        .collect::<Vec<_>>();
    fixture.plan.prefill.operations.retain(|op| {
        prefix.contains(&op.id) || matches!(op.operator, Op::RmsNorm | Op::OutputHead)
    });
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let mut reference = fixture.imports();
    let token = reference
        .push(
            Code::Input("input.tokens".into()),
            3,
            Interval::new(0, 2).unwrap(),
        )
        .unwrap();
    let first = reference.nodes.len();
    let mut frontier = BTreeMap::from([("input.tokens".into(), reference.view(token, vec![1, 3]))]);
    for op in &fixture.plan.prefill.operations[..5] {
        let output = original(&mut reference, &fixture, op, &frontier[&op.inputs[0]]);
        frontier.insert(op.id.clone(), output);
    }
    // The real head declaration is needed for original token geometry and its
    // future floor work belongs in the constructor's complete pending census.
    // Measure that work through the independent original composition too, using
    // a genuine last-row alias of the already computed hidden-width lookup.
    // This remains a selected-prefix work oracle, not full-graph qualification.
    let head = fixture
        .plan
        .prefill
        .operations
        .iter()
        .find(|op| op.operator == Op::OutputHead)
        .unwrap();
    let head_input = View {
        words: frontier[&prefix[0]]
            .words
            .chunks_exact(fixture.width)
            .last()
            .unwrap(),
        shape: vec![1, 1, fixture.width],
    };
    original(&mut reference, &fixture, head, &head_input);
    let native = schedule_work(&reference.nodes[first..]).unwrap();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imports = fixture.imports();
    let bindings = vec![fixture.declare(&mut imports, &registry, &prior, &outside)];
    assert_eq!(
        pending_contraction_work(&imports, &fixture.plan.prefill, 0, &fixture.sources).unwrap(),
        native
    );
    let mut pressure = owners(&values);
    pressure.other_work = WorkBudget::local()
        .checked_mul(puma::MAX_PROGRAM_OPERATIONS - native.operations() + 1)
        .unwrap();
    pressure.other_work.limits().unwrap();
    let before = (
        input_owner::CONTAINER_FACTORIES.with(|count| count.get()),
        LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()),
        super::super::super::prepared_tensor_map::native_attempts(),
        BIAS_RECIPE_ATTEMPTS.with(|count| count.get()),
    );
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
        (
            input_owner::CONTAINER_FACTORIES.with(|count| count.get()),
            LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()),
            super::super::super::prepared_tensor_map::native_attempts(),
            BIAS_RECIPE_ATTEMPTS.with(|count| count.get())
        ),
        before
    );
}
