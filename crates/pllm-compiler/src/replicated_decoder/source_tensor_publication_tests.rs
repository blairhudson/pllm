use super::*;
use crate::replicated_decoder::{PublicSourceImport, SourceNorms, SourceTensor};
use pllm_models::DecoderPlan;

// Descriptor validation and authenticated PublicFile import, NOT Manifest::compile
// or a captured/reconstructed native program. All commitments cover these actual
// nonzero encoded bytes. No external row certificate is minted by this fixture.
struct Fixture {
    plan: DecoderPlan,
    sources: BTreeMap<String, SourceTensor>,
    norms: SourceNorms,
}

impl Fixture {
    fn new(rows: usize) -> Self {
        let config = serde_json::to_vec(&serde_json::json!({
            "model_type":"llama", "hidden_size":10, "intermediate_size":2,
            "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
            "head_dim":10, "vocab_size":3, "tie_word_embeddings":false,
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
                let coefficient = plan.prefill.operations.iter().any(|op| {
                    op.operator == Op::RmsNorm
                        && op.attributes.get("weight").and_then(|v| v.as_str())
                            == Some(key.as_str())
                });
                let values =
                    vec![if coefficient { ONE / 2 } else { ONE / 4 }; shape.iter().product()];
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
                        public_q18: coefficient.then_some(values),
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

    fn source(&self, registry: &SourceRegistry<'_>) -> Arc<PublicSource> {
        let mut import = PublicSourceImport::new_checked(registry, 0, MAX_PROGRAM_BYTES).unwrap();
        // Import affine coefficients too, in the importer's complete record order.
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

    fn input<'a>(
        &'a self,
        builder: &mut Builder,
        registry: &'a SourceRegistry<'a>,
        owners: &compact_admission::RequestOwners<'_>,
    ) -> OriginalImportedView<'a> {
        let graph = &self.plan.decode;
        let semantic = graph
            .state_inputs
            .iter()
            .find(|state| state.shape.last() == Some(&10))
            .unwrap();
        builder
            .declare_original_input(
                registry,
                &self.plan,
                graph,
                &semantic.id,
                Interval::new(-ONE / 2, ONE / 2).unwrap(),
                &compact_admission::RequestWork::default(),
                owners,
                MAX_PROGRAM_BYTES,
            )
            .unwrap()
    }

    fn raw(&self, builder: &mut Builder, input: &OriginalImportedView<'_>) -> usize {
        let weight = builder.nodes.iter().position(|node|
            matches!(&node.code, Code::PublicInput { key, .. } if key.ends_with("q_proj.weight"))).unwrap();
        let view = builder.view(input.owner(), input.logical().dimensions().to_vec());
        // Genuine original numeric implementation, before native rescale and
        // before any finalizer. Not an artificial Code::PublicProduct wrapper.
        builder
            .public_matrix(&view, weight, 10, false, false)
            .unwrap()
            .words
            .at(0)
            .0
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

#[test]
fn authenticated_original_state_product_prepares_full_tail_without_expanded_finalization() {
    let fixture = Fixture::new(13);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&mut builder, &registry, &outside);
    let raw = fixture.raw(&mut builder, &input);
    assert_eq!(builder.nodes[raw].lanes, 130);
    let publication = SourceTensorPublication::prepare(
        &mut builder,
        fixture.validated(),
        &input,
        &source,
        raw,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    assert_eq!(publication.prepared().program_at(0).unwrap().lanes(), 128);
    assert_eq!(publication.prepared().program_at(128).unwrap().lanes(), 2);
    assert_eq!(publication.native_work(), rescale_work(130).unwrap());
    for offset in [0, 128] {
        let actual = publication.prepared().program_at(offset).unwrap();
        let mut original = tensor::Builder::new(actual.lanes(), builder.nodes[raw].bounds).unwrap();
        let output = original.rescale(original.input(), 18).unwrap();
        let original = original.finish(output).unwrap();
        assert_eq!(actual.digest(), original.digest());
        assert_eq!(actual.work_budget(), original.work_budget());
        assert_eq!(actual.output_bounds(), original.output_bounds());
    }
    assert_eq!(publication.required_checks().input_owner, input.owner());
    assert_eq!(publication.required_checks().input_bounds, input.bounds());
    assert_eq!(
        (
            publication.required_checks().rows,
            publication.required_checks().input_width,
            publication.required_checks().output_width
        ),
        (13, 10, 10)
    );
    assert!(builder.row_norms.is_empty() && builder.row_energies.is_empty());
    assert!(builder.validity.is_none()); // no range/sticky authority was promoted
    let full = Arc::downgrade(publication.prepared().program_at(0).unwrap());
    let tail = Arc::downgrade(publication.prepared().program_at(128).unwrap());
    let original_source = Arc::downgrade(&source);
    let output = publication
        .publish_original(
            &mut builder,
            &input,
            &source,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    let Code::TensorLoop {
        program,
        tail: Some(rest),
        ..
    } = &builder.nodes[output].code
    else {
        panic!("original loop")
    };
    assert!(Arc::ptr_eq(
        program,
        publication.prepared().program_at(0).unwrap()
    ));
    assert!(Arc::ptr_eq(
        rest,
        publication.prepared().program_at(128).unwrap()
    ));
    drop(source);
    assert!(original_source.upgrade().is_some());
    drop(publication);
    assert!(original_source.upgrade().is_none());
    assert!(full.upgrade().is_some() && tail.upgrade().is_some());
    drop(builder);
    assert!(full.upgrade().is_none() && tail.upgrade().is_none());
}

#[test]
fn equal_digest_foreign_file_and_native_owners_never_replace_bound_owners() {
    let fixture = Fixture::new(13);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let foreign_source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&mut builder, &registry, &outside);
    let raw = fixture.raw(&mut builder, &input);
    let publication = SourceTensorPublication::prepare(
        &mut builder,
        fixture.validated(),
        &input,
        &source,
        raw,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    assert!(matches!(
        publication.check(&builder, &input, &foreign_source, publication.prepared()),
        Err(Error::Binding)
    ));
    let foreign =
        PreparedTensorMap::rescale(130, builder.nodes[raw].bounds, 18, 0, |_| Ok(())).unwrap();
    for (original, candidate) in publication.prepared().programs().zip(foreign.programs()) {
        assert_eq!(original.digest(), candidate.digest());
        assert!(!Arc::ptr_eq(original, candidate));
    }
    assert!(matches!(
        publication.check(&builder, &input, &source, &foreign),
        Err(Error::Binding)
    ));
    assert!(publication
        .check(&builder, &input, &source, publication.prepared())
        .is_ok());
}

#[test]
fn exact_preparation_peak_and_one_short_refusal_precede_native_factories() {
    let fixture = Fixture::new(13);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&mut builder, &registry, &outside);
    let raw = fixture.raw(&mut builder, &input);
    let first = SourceTensorPublication::prepare(
        &mut builder,
        fixture.validated(),
        &input,
        &source,
        raw,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let exact = first.preparation_peak();
    drop(first);
    let attempts = crate::replicated_decoder::prepared_tensor_map::native_attempts();
    let source_count = Arc::strong_count(&source);
    assert!(matches!(
        SourceTensorPublication::prepare(
            &mut builder,
            fixture.validated(),
            &input,
            &source,
            raw,
            &prior,
            &outside,
            exact - 1
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(
        crate::replicated_decoder::prepared_tensor_map::native_attempts(),
        attempts
    );
    assert_eq!(Arc::strong_count(&source), source_count);
    let accepted = SourceTensorPublication::prepare(
        &mut builder,
        fixture.validated(),
        &input,
        &source,
        raw,
        &prior,
        &outside,
        exact,
    )
    .unwrap();
    assert_eq!(accepted.preparation_peak(), exact);
    assert_eq!(
        crate::replicated_decoder::prepared_tensor_map::native_attempts(),
        attempts + 2
    );
}

#[test]
fn original_input_and_registry_identity_are_checked_before_factories() {
    let fixture = Fixture::new(1);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let other_registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&mut builder, &registry, &outside);
    let raw = fixture.raw(&mut builder, &input);
    let mut other = fixture.builder();
    let foreign = fixture.input(&mut other, &other_registry, &outside);
    assert_eq!(input.owner(), foreign.owner());
    assert_eq!(registry.binding(), other_registry.binding());
    let attempts = crate::replicated_decoder::prepared_tensor_map::native_attempts();
    assert!(matches!(
        SourceTensorPublication::prepare(
            &mut builder,
            fixture.validated(),
            &foreign,
            &source,
            raw,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert_eq!(
        crate::replicated_decoder::prepared_tensor_map::native_attempts(),
        attempts
    );
}

#[test]
fn aggregate_work_and_failed_publication_leave_original_owners_recoverable() {
    let fixture = Fixture::new(1);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let values = Vec::new();
    let mut outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&mut builder, &registry, &outside);
    let raw = fixture.raw(&mut builder, &input);
    outside.other_work = WorkBudget::rescale(1)
        .unwrap()
        .checked_mul(1 << 40)
        .unwrap();
    let attempts = crate::replicated_decoder::prepared_tensor_map::native_attempts();
    assert!(SourceTensorPublication::prepare(
        &mut builder,
        fixture.validated(),
        &input,
        &source,
        raw,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES
    )
    .is_err());
    assert_eq!(
        crate::replicated_decoder::prepared_tensor_map::native_attempts(),
        attempts
    );
    outside.other_work = WorkBudget::default();
    let publication = SourceTensorPublication::prepare(
        &mut builder,
        fixture.validated(),
        &input,
        &source,
        raw,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let before = (
        builder.nodes.len(),
        builder.nodes.capacity(),
        builder.allocated,
    );
    assert!(matches!(
        publication.publish_original(&mut builder, &input, &source, &prior, &outside, 1),
        Err(Error::Bounds)
    ));
    assert_eq!(
        before,
        (
            builder.nodes.len(),
            builder.nodes.capacity(),
            builder.allocated
        )
    );
    publication
        .publish_original(
            &mut builder,
            &input,
            &source,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
}

#[test]
fn unsupported_original_transposed_product_is_rejected_before_native_entry() {
    let fixture = Fixture::new(1);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let source = fixture.source(&registry);
    let values = Vec::new();
    let outside = owners(&values);
    let prior = compact_admission::RequestWork::default();
    let mut builder = fixture.builder();
    let input = fixture.input(&mut builder, &registry, &outside);
    let weight = builder.nodes.iter().position(|node|
        matches!(&node.code, Code::PublicInput { key, .. } if key.ends_with("q_proj.weight"))).unwrap();
    let view = builder.view(input.owner(), input.logical().dimensions().to_vec());
    let raw = builder
        .public_matrix(&view, weight, 10, true, false)
        .unwrap()
        .words
        .at(0)
        .0;
    let attempts = crate::replicated_decoder::prepared_tensor_map::native_attempts();
    assert!(matches!(
        SourceTensorPublication::prepare(
            &mut builder,
            fixture.validated(),
            &input,
            &source,
            raw,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert_eq!(
        crate::replicated_decoder::prepared_tensor_map::native_attempts(),
        attempts
    );
}

#[test]
fn f32_import_commits_independent_i64_le_q18_output_not_input_bytes() {
    let mut fixture = Fixture::new(1);
    for record in fixture.sources.values_mut().filter(|r| r.public_q18.is_none()) {
        let lanes: usize = record.shape.iter().product();
        let expected: Vec<i64> = (0..lanes).map(|i| [0, 262144, -131072][i % 3]).collect();
        let encoded: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
        record.lower = -131072;
        record.upper = 262144;
        record.digest = Sha256::digest(&encoded).into();
    }
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let mut import = PublicSourceImport::new_checked(&registry, 0, MAX_PROGRAM_BYTES).unwrap();
    for (key, record) in registry.source_records() {
        let lanes: usize = record.shape.iter().product();
        let bytes: Vec<u8> = (0..lanes).flat_map(|i| {
            let f = if record.public_q18.is_some() { 0.5_f32 } else { [0.0_f32, 1.0, -0.5][i % 3] };
            f.to_le_bytes()
        }).collect();
        import.append(key, 0, &bytes).unwrap();
    }
    let source = import.finish().unwrap();
    for descriptor in registry.files() {
        let expected: Vec<u8> = (0..descriptor.lanes)
            .flat_map(|i| [0_i64, 262144, -131072][i % 3].to_le_bytes()).collect();
        let mut actual = vec![0; expected.len()];
        source.file(descriptor.key).unwrap().read(0, &mut actual).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(<[u8; 32]>::from(Sha256::digest(&actual)), descriptor.digest);
    }
}

#[test]
fn wrong_i64_frame_can_be_finite_f32_but_content_mismatch_retires_importer() {
    let mut fixture = Fixture::new(1);
    for record in fixture.sources.values_mut().filter(|r| r.public_q18.is_none()) {
        let lanes: usize = record.shape.iter().product();
        let expected: Vec<u8> = std::iter::repeat_n(262144_i64, lanes)
            .flat_map(i64::to_le_bytes).collect();
        record.lower = 0;
        record.upper = 262144;
        record.digest = Sha256::digest(&expected).into();
    }
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let mut import = PublicSourceImport::new_checked(&registry, 0, MAX_PROGRAM_BYTES).unwrap();
    for (key, record) in registry.source_records() {
        let lanes: usize = record.shape.iter().product();
        if record.public_q18.is_some() {
            import.append(key, 0, &std::iter::repeat_n(0.5_f32, lanes)
                .flat_map(f32::to_le_bytes).collect::<Vec<_>>()).unwrap();
            continue;
        }
        // Both halves of 262144_i64 are finite F32; syntax rejection is NOT
        // the contract. Keep the expected frame length, then compare CONTENT.
        let wrong: Vec<u8> = std::iter::repeat_n(262144_i64, lanes)
            .flat_map(i64::to_le_bytes).take(4 * lanes).collect();
        assert!(wrong.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
        assert_eq!(import.append(key, 0, &wrong).unwrap_err(), "public source digest differs");
        assert!(import.append(key, 0, &wrong).is_err());
        assert!(import.finish().is_err());
        return;
    }
    panic!("fixture needs a stored Q18 file");
}

#[test]
fn checked_declaration_retention_preserves_identity_not_foreign_equal_content() {
    use super::custody::OriginalInputBinding;
    let fixture = Fixture::new(1);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let values = Vec::new();
    let outside = owners(&values);
    let mut builder = fixture.builder();
    let input = fixture.input(&mut builder, &registry, &outside);
    let record = builder.original_inputs.iter().find(|r| r.owner() == input.owner()).unwrap();
    let weak = Arc::downgrade(record);
    let count = Arc::strong_count(record);
    let retained = input.retain_checked(&builder).unwrap();
    assert_eq!(Arc::strong_count(record), count + 1);
    assert!(input.same_declaration(&retained));
    assert!(input.ordered_coordinate(0).unwrap().same(&retained.ordered_coordinate(0).unwrap()));
    let mut other_builder = fixture.builder();
    let foreign = fixture.input(&mut other_builder, &registry, &outside);
    assert_eq!(input.owner(), foreign.owner());
    assert!(!retained.same_declaration(&foreign));
    assert!(!retained.ordered_coordinate(0).unwrap().same(&foreign.ordered_coordinate(0).unwrap()));
    assert!(input.retain_checked(&other_builder).is_err());
    assert_eq!(Arc::strong_count(record), count + 1);
    retained.check(&builder, &registry, &fixture.plan, input.graph()).unwrap();
    drop(input);
    drop(builder);
    assert!(weak.upgrade().is_some());
    drop(retained);
    assert!(weak.upgrade().is_none());
}
