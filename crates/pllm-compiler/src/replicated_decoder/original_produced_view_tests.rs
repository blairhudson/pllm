use super::*;
use crate::replicated_decoder::{PublicSourceImport, SourceNorms, SourceTensor};
use pllm_models::DecoderPlan;

// Genuine model lowering + free source validation + authenticated source files.
// No Manifest::compile/Program finalizer, synthetic producer ID, incoming row
// certificate or retroactively wrapped View participates in seal creation.
struct Fixture {
    plan: DecoderPlan,
    sources: BTreeMap<String, SourceTensor>,
    norms: SourceNorms,
}
impl Fixture {
    fn new(rows: usize) -> Self {
        let config = serde_json::to_vec(&serde_json::json!({
            "model_type":"llama", "hidden_size":2, "intermediate_size":2,
            "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
            "head_dim":2, "vocab_size":3, "tie_word_embeddings":false,
            "hidden_act":"silu", "attention_bias":false, "max_position_embeddings":1024,
            "rope_theta":10000, "rms_norm_eps":0.000001, "attention_dropout":0.0,
            "use_sliding_window":false, "use_cache":true, "max_window_layers":1
        }))
        .unwrap();
        let plan = lower_with_weights(&config, rows, 2, WeightMode::Public).unwrap();
        let shapes =
            Program::source_shapes_with_weights(&config, rows, 2, WeightMode::Public).unwrap();
        let sources = shapes
            .into_iter()
            .map(|(key, shape)| {
                let gamma = plan.prefill.operations.iter().any(|op| {
                    op.operator == Op::RmsNorm
                        && op.attributes.get("weight").and_then(|v| v.as_str())
                            == Some(key.as_str())
                });
                let values = vec![if gamma { ONE / 2 } else { ONE / 4 }; shape.iter().product()];
                let mut hash = Sha256::new();
                for value in &values {
                    hash.update(value.to_le_bytes());
                }
                (
                    key,
                    SourceTensor {
                        shape,
                        lower: values[0],
                        upper: values[0],
                        public_q18: gamma.then_some(values),
                        digest: hash.finalize().into(),
                        row_l1: None,
                        row_squared_norm: None,
                        row_l1_squared_sum: None,
                    },
                )
            })
            .collect();
        Self {
            plan,
            sources,
            norms: SourceNorms::new(),
        }
    }
    fn validated(&self) -> ValidatedSourceDescriptors<'_> {
        let shapes = self
            .sources
            .iter()
            .map(|(key, value)| (key.clone(), value.shape.clone()))
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
    fn source(&self, registry: &SourceRegistry<'_>) -> Arc<PublicSource> {
        let mut import = PublicSourceImport::new_checked(registry, 0, MAX_PROGRAM_BYTES).unwrap();
        for (key, record) in registry.source_records() {
            let value = if record.public_q18.is_some() { 0.5_f32 } else { 0.25_f32 };
            let bytes: Vec<_> = std::iter::repeat_n(value, record.shape.iter().product())
                .flat_map(f32::to_le_bytes)
                .collect();
            import.append(key, 0, &bytes).unwrap();
        }
        Arc::new(import.finish().unwrap())
    }
    fn builder(&self) -> Builder {
        let mut builder = Builder {
            public_weights: true,
            lookup: policy::LookupPolicy::PerToken,
            ..Builder::default()
        };
        for (key, source) in &self.sources {
            if source.public_q18.is_none() {
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
        }
        builder
    }
    fn input<'s>(
        &'s self,
        registry: &'s SourceRegistry<'s>,
        builder: &mut Builder,
        outside: &compact_admission::RequestOwners<'_>,
    ) -> OriginalImportedView<'s> {
        builder
            .declare_original_input(
                registry,
                &self.plan,
                &self.plan.prefill,
                "input.tokens",
                Interval::new(0, 2).unwrap(),
                &compact_admission::RequestWork::default(),
                outside,
                MAX_PROGRAM_BYTES,
            )
            .unwrap()
    }
    fn lookup(&self) -> &ModelOperation {
        self.plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::TokenLookup)
            .unwrap()
    }
    fn norm(&self) -> &ModelOperation {
        self.plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::RmsNorm && op.inputs.first() == Some(&self.lookup().id))
            .unwrap()
    }
    fn projection(&self) -> &ModelOperation {
        self.plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::Linear && op.inputs.first() == Some(&self.norm().id))
            .unwrap()
    }
}
fn owners(values: &Vec<Option<Secret>>) -> compact_admission::RequestOwners<'_> {
    compact_admission::RequestOwners {
        borrowed_external_bytes: 0,
        owned_source_bytes: 0,
        other_owned_metadata_bytes: 0,
        other_stored_descriptors: 0,
        other_work: WorkBudget::default(),
        retained_values: values,
        external_frame_bytes: 0,
        caller_control_bytes: size_of::<Fixture>(),
        prior_request_peak_bytes: 0,
    }
}
fn entries() -> usize {
    FACTORY_ENTRIES.with(Cell::get)
        + crate::replicated_decoder::prepared_tensor_map::native_attempts()
}

#[test]
fn genuine_lookup_rms_projection_chain_retains_original_factories_without_expanded_nodes() {
    let fixture = Fixture::new(65);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&registry, &mut builder, &outside);
    let before = (
        builder.nodes.len(),
        builder.nodes.capacity(),
        builder.allocated,
    );
    let lookup = OriginalProducedView::lookup(
        &builder,
        &input,
        &source,
        fixture.lookup(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let norm = OriginalProducedView::rms_norm(
        &builder,
        &lookup,
        &source,
        fixture.norm(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let projection = SourceTensorPublication::prepare_binding(
        &builder,
        &norm,
        &source,
        fixture.projection(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    assert_eq!(
        before,
        (
            builder.nodes.len(),
            builder.nodes.capacity(),
            builder.allocated
        )
    );
    assert_eq!(projection.geometry().dimensions(), &[1, 65, 2]);
    assert!(projection.preparation_peak() >= norm.preparation_peak());
    assert!(norm.preparation_peak() >= lookup.preparation_peak());
    assert!(Arc::ptr_eq(projection.source(), &source));
    let RecipeRef::Lookup {
        full,
        tail,
        vocabulary,
    } = lookup.recipe()
    else {
        panic!("lookup recipe")
    };
    assert_eq!(vocabulary, 3);
    assert!(tail.is_none());
    assert_eq!(
        full.digest(),
        tensor::Program::equal_zero(3, Interval::new(-2, 2).unwrap())
            .unwrap()
            .digest()
    );
    let RecipeRef::Rms { program, width } = norm.recipe() else {
        panic!("RMS recipe")
    };
    assert_eq!(width, 2);
    let gamma = fixture.sources[fixture.norm().attributes["weight"].as_str().unwrap()]
        .public_q18
        .as_ref()
        .unwrap();
    let (original, _) = rms_norm_program(true, 2, lookup.record.bounds, gamma, None).unwrap();
    assert_eq!(program.digest(), original.digest());
    let RecipeRef::Projection {
        map,
        inner,
        columns,
        raw,
    } = projection.recipe()
    else {
        panic!("projection recipe")
    };
    assert_eq!((inner, columns), (2, 2));
    assert_eq!(map.program_at(0).unwrap().lanes(), 128);
    assert_eq!(map.program_at(128).unwrap().lanes(), 2);
    for offset in [0, 128] {
        let native = map.program_at(offset).unwrap();
        let mut old = tensor::Builder::new(native.lanes(), raw).unwrap();
        let output = old.rescale(old.input(), 18).unwrap();
        assert_eq!(native.digest(), old.finish(output).unwrap().digest());
    }
    assert!(builder.row_norms.is_empty() && builder.row_energies.is_empty());
    assert!(builder.validity.is_none()); // creation identity did not issue proofs
    assert!(std::ptr::eq(
        projection.record.operation,
        fixture.projection()
    ));
    let pending = projection.unresolved_requirements();
    assert_eq!(pending.original_input, input.owner());
    assert_eq!(pending.declared_input_domain, input.bounds());
    assert_eq!(pending.complete_row_width, 2);
}

#[test]
fn lazy_rank_aliases_preserve_exact_owner_and_order_but_not_semantic_relabelling() {
    let fixture = Fixture::new(65);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&registry, &mut builder, &outside);
    let lookup = OriginalProducedView::lookup(
        &builder,
        &input,
        &source,
        fixture.lookup(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let flat = lookup.reshape(&[130], MAX_PROGRAM_BYTES).unwrap();
    for index in 0..130 {
        assert!(flat
            .coordinate(index)
            .unwrap()
            .same(&lookup.coordinate(index).unwrap()));
    }
    let transposed = lookup.permute(&[0, 2, 1], MAX_PROGRAM_BYTES).unwrap();
    assert_eq!(transposed.coordinate(1).unwrap().lane(), 2);
    assert_eq!(transposed.coordinate(65).unwrap().lane(), 1);
    let restored = transposed.permute(&[0, 2, 1], MAX_PROGRAM_BYTES).unwrap();
    for index in 0..130 {
        assert!(restored
            .coordinate(index)
            .unwrap()
            .same(&lookup.coordinate(index).unwrap()));
    }
    let before = entries();
    assert!(matches!(
        OriginalProducedView::rms_norm(
            &builder,
            &transposed,
            &source,
            fixture.norm(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Shape)
    ));
    assert_eq!(entries(), before);
    assert!(matches!(
        lookup.reshape(&[129], MAX_PROGRAM_BYTES),
        Err(Error::Shape)
    ));
    assert!(matches!(
        lookup.permute(&[0, 1, 1], MAX_PROGRAM_BYTES),
        Err(Error::Shape)
    ));
    assert!(matches!(lookup.coordinate(130), Err(Error::Shape)));
    assert!(!input
        .ordered_coordinate(0)
        .unwrap()
        .same(&lookup.coordinate(0).unwrap()));
}

#[test]
fn foreign_same_ids_digest_plan_and_source_cannot_mint_continuation() {
    let fixture = Fixture::new(1);
    let cloned = fixture.plan.clone();
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let foreign_source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&registry, &mut builder, &outside);
    let lookup = OriginalProducedView::lookup(
        &builder,
        &input,
        &source,
        fixture.lookup(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let independent = OriginalProducedView::lookup(
        &builder,
        &input,
        &source,
        fixture.lookup(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let (RecipeRef::Lookup { full: first, .. }, RecipeRef::Lookup { full: second, .. }) =
        (lookup.recipe(), independent.recipe())
    else {
        panic!("lookup")
    };
    assert_eq!(first.digest(), second.digest());
    assert!(!Arc::ptr_eq(first, second));
    assert!(!lookup
        .coordinate(0)
        .unwrap()
        .same(&independent.coordinate(0).unwrap()));
    let cloned_norm = cloned
        .prefill
        .operations
        .iter()
        .find(|op| op.id == fixture.norm().id)
        .unwrap();
    let before = entries();
    assert!(matches!(
        OriginalProducedView::rms_norm(
            &builder,
            &lookup,
            &source,
            cloned_norm,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert!(matches!(
        OriginalProducedView::rms_norm(
            &builder,
            &lookup,
            &foreign_source,
            fixture.norm(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    let other = fixture.builder();
    assert!(matches!(
        OriginalProducedView::rms_norm(
            &other,
            &lookup,
            &source,
            fixture.norm(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert_eq!(entries(), before);
}

#[test]
fn exact_and_one_short_lookup_rms_projection_quotes_refuse_before_factories() {
    let fixture = Fixture::new(65);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&registry, &mut builder, &outside);
    let probe = OriginalProducedView::lookup(
        &builder,
        &input,
        &source,
        fixture.lookup(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let exact = probe.preparation_peak();
    drop(probe);
    let before = entries();
    assert!(matches!(
        OriginalProducedView::lookup(
            &builder,
            &input,
            &source,
            fixture.lookup(),
            &prior,
            &outside,
            exact - 1
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(entries(), before);
    let lookup = OriginalProducedView::lookup(
        &builder,
        &input,
        &source,
        fixture.lookup(),
        &prior,
        &outside,
        exact,
    )
    .unwrap();
    let probe = OriginalProducedView::rms_norm(
        &builder,
        &lookup,
        &source,
        fixture.norm(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let exact = probe.preparation_peak();
    drop(probe);
    let before = entries();
    assert!(matches!(
        OriginalProducedView::rms_norm(
            &builder,
            &lookup,
            &source,
            fixture.norm(),
            &prior,
            &outside,
            exact - 1
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(entries(), before);
    let norm = OriginalProducedView::rms_norm(
        &builder,
        &lookup,
        &source,
        fixture.norm(),
        &prior,
        &outside,
        exact,
    )
    .unwrap();
    let probe = SourceTensorPublication::prepare_binding(
        &builder,
        &norm,
        &source,
        fixture.projection(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let exact = probe.preparation_peak();
    drop(probe);
    let before = entries();
    assert!(matches!(
        SourceTensorPublication::prepare_binding(
            &builder,
            &norm,
            &source,
            fixture.projection(),
            &prior,
            &outside,
            exact - 1
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(entries(), before);
    let accepted = SourceTensorPublication::prepare_binding(
        &builder,
        &norm,
        &source,
        fixture.projection(),
        &prior,
        &outside,
        exact,
    )
    .unwrap();
    assert_eq!(accepted.preparation_peak(), exact);
}

#[test]
fn derived_owner_keeps_native_source_and_import_lifetimes_after_callers_drop() {
    let fixture = Fixture::new(65);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let source_weak = Arc::downgrade(&source);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&registry, &mut builder, &outside);
    let import_weak = Arc::downgrade(builder.original_inputs.iter()
        .find(|record| record.owner() == input.owner()).unwrap());
    let lookup = OriginalProducedView::lookup(
        &builder,
        &input,
        &source,
        fixture.lookup(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let norm = OriginalProducedView::rms_norm(
        &builder,
        &lookup,
        &source,
        fixture.norm(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let RecipeRef::Rms { program, .. } = norm.recipe() else {
        panic!("RMS")
    };
    let rms_weak = Arc::downgrade(program);
    let projection = SourceTensorPublication::prepare_binding(
        &builder,
        &norm,
        &source,
        fixture.projection(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let RecipeRef::Projection { map, .. } = projection.recipe() else {
        panic!("projection")
    };
    let full = Arc::downgrade(map.program_at(0).unwrap());
    let tail = Arc::downgrade(map.program_at(128).unwrap());
    drop(norm);
    drop(lookup);
    drop(input);
    drop(builder);
    drop(source);
    assert!(source_weak.upgrade().is_some() && import_weak.upgrade().is_some());
    assert!(rms_weak.upgrade().is_some() && full.upgrade().is_some() && tail.upgrade().is_some());
    drop(projection);
    assert!(source_weak.upgrade().is_none() && import_weak.upgrade().is_none());
    assert!(rms_weak.upgrade().is_none() && full.upgrade().is_none() && tail.upgrade().is_none());
}

#[test]
fn real_silu_site_cannot_relabel_an_unrelated_projection_as_its_gate_input() {
    let fixture = Fixture::new(1);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&registry, &mut builder, &outside);
    let lookup = OriginalProducedView::lookup(
        &builder,
        &input,
        &source,
        fixture.lookup(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let norm = OriginalProducedView::rms_norm(
        &builder,
        &lookup,
        &source,
        fixture.norm(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let projection = SourceTensorPublication::prepare_binding(
        &builder,
        &norm,
        &source,
        fixture.projection(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let silu = fixture
        .plan
        .prefill
        .operations
        .iter()
        .find(|op| op.operator == Op::Silu)
        .unwrap();
    let before = entries();
    assert!(matches!(
        OriginalProducedView::activation(
            &builder,
            &projection,
            &source,
            silu,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert_eq!(entries(), before);
}

#[test]
fn authentic_source_statistics_do_not_discharge_missing_general_row_contract() {
    let mut fixture = Fixture::new(1);
    let key = fixture.projection().attributes["weight"]
        .as_str()
        .unwrap()
        .to_owned();
    // Genuine statistic over the exact imported nonzero coefficients, not a
    // forged certificate. Only the missing INPUT row authority is the gate.
    fixture.sources.get_mut(&key).unwrap().row_squared_norm = Some(2 * (ONE as u128 / 4).pow(2));
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&registry, &mut builder, &outside);
    let lookup = OriginalProducedView::lookup(
        &builder,
        &input,
        &source,
        fixture.lookup(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let norm = OriginalProducedView::rms_norm(
        &builder,
        &lookup,
        &source,
        fixture.norm(),
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let before = entries();
    assert!(matches!(
        SourceTensorPublication::prepare_binding(
            &builder,
            &norm,
            &source,
            fixture.projection(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert_eq!(entries(), before);
    assert!(builder.row_norms.is_empty() && builder.row_energies.is_empty());
}
