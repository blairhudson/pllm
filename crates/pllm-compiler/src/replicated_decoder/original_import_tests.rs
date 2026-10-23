use super::compact_admission::{RequestOwners, RequestWork};
use super::*;

pub(in crate::replicated_decoder::program) struct Fixture {
    pub plan: pllm_models::DecoderPlan,
    pub sources: BTreeMap<String, SourceTensor>,
    pub norms: SourceNorms,
}

impl Fixture {
    pub fn new(rows: usize) -> Self {
        let config = serde_json::to_vec(&serde_json::json!({
            "model_type":"llama", "hidden_size":2, "intermediate_size":2,
            "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
            "head_dim":2, "vocab_size":3, "tie_word_embeddings":false,
            "hidden_act":"silu", "attention_bias":false, "max_position_embeddings":1024,
            "rope_theta":10000, "rms_norm_eps":0.000001, "attention_dropout":0.0,
            "use_sliding_window":false, "use_cache":true, "max_window_layers":1
        }))
        .unwrap();
        let plan = lower_with_weights(&config, rows, 1, WeightMode::Public).unwrap();
        let shapes =
            Program::source_shapes_with_weights(&config, rows, 1, WeightMode::Public).unwrap();
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
        Self {
            plan,
            sources,
            norms: SourceNorms::new(),
        }
    }

    pub fn validated(&self) -> ValidatedSourceDescriptors<'_> {
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

    pub fn registry(&self) -> SourceRegistry<'_> {
        SourceRegistry::from_validated(self.validated()).unwrap()
    }

    pub fn builder(&self) -> Builder {
        let mut builder = Builder {
            public_weights: true,
            ..Builder::default()
        };
        for (key, source) in &self.sources {
            if source.public_q18.is_some() {
                continue;
            }
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
}

pub(in crate::replicated_decoder::program) fn owners(
    values: &Vec<Option<Secret>>,
) -> RequestOwners<'_> {
    RequestOwners {
        borrowed_external_bytes: 0,
        owned_source_bytes: 0,
        other_owned_metadata_bytes: 0,
        other_stored_descriptors: 0,
        other_work: WorkBudget::default(),
        retained_values: values,
        external_frame_bytes: 0,
        caller_control_bytes: 0,
        prior_request_peak_bytes: 0,
    }
}

#[test]
fn original_declaration_retains_real_record_not_builder_borrow_or_numeric_identity() {
    let fixture = Fixture::new(3);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut first = fixture.builder();
    let mut second = fixture.builder();
    let declare = |builder: &mut Builder| {
        builder
            .declare_original_input(
                &registry,
                &fixture.plan,
                &fixture.plan.prefill,
                "input.tokens",
                Interval::new(0, 2).unwrap(),
                &prior,
                &outside,
                MAX_PROGRAM_BYTES,
            )
            .unwrap()
    };
    let input = declare(&mut first);
    let foreign = declare(&mut second);
    assert_eq!(input.owner(), foreign.owner());
    assert_eq!(input.logical(), foreign.logical());
    assert_eq!(input.registry().binding(), foreign.registry().binding());
    assert!(
        input
            .check(&first, &registry, &fixture.plan, &fixture.plan.prefill)
            .is_ok()
    );
    assert!(matches!(
        foreign.check(&first, &registry, &fixture.plan, &fixture.plan.prefill),
        Err(Error::Binding)
    ));
    // Keeping the seal does not freeze the entire mutable numeric Builder.
    first.affine(input.owner(), None, affine(1, 0, 0)).unwrap();
    let weak = Arc::downgrade(&input.record);
    assert_eq!(Arc::strong_count(&input.record), 2);
    drop(first);
    assert_eq!(weak.strong_count(), 1);
    drop(input);
    assert!(weak.upgrade().is_none());
}

#[test]
fn equal_source_digest_and_cloned_plan_cannot_replace_original_borrowed_owners() {
    let fixture = Fixture::new(1);
    let foreign_fixture = Fixture::new(1);
    let registry = fixture.registry();
    let foreign_registry = foreign_fixture.registry();
    assert_eq!(registry.binding(), foreign_registry.binding());
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut builder = fixture.builder();
    let input = builder
        .declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            "input.tokens",
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    assert!(matches!(
        input.check(
            &builder,
            &foreign_registry,
            &fixture.plan,
            &fixture.plan.prefill
        ),
        Err(Error::Binding)
    ));
    let clone = fixture.plan.clone();
    assert!(matches!(
        input.check(&builder, &registry, &clone, &clone.prefill),
        Err(Error::Binding)
    ));
    assert!(std::ptr::eq(input.plan(), &fixture.plan));
    assert!(std::ptr::eq(
        input.registry().source_records(),
        &fixture.sources
    ));
}

#[test]
fn refusal_precedes_original_input_record_and_destination_factories() {
    let fixture = Fixture::new(1);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut builder = fixture.builder();
    let before = (
        builder.nodes.len(),
        builder.nodes.capacity(),
        builder.allocated,
        builder.original_inputs.capacity(),
    );
    let attempts = INPUT_FACTORIES.with(|n| n.get());
    assert!(matches!(
        builder.declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            "input.tokens",
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            1
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(INPUT_FACTORIES.with(|n| n.get()), attempts);
    assert_eq!(
        before,
        (
            builder.nodes.len(),
            builder.nodes.capacity(),
            builder.allocated,
            builder.original_inputs.capacity()
        )
    );
    let prefix = fixture.plan.prefill.operations[0].id.as_str();
    assert!(matches!(
        builder.declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            prefix,
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert_eq!(INPUT_FACTORIES.with(|n| n.get()), attempts);
}

#[test]
fn opt_in_declaration_keeps_legacy_import_ordinals_canonical_bytes_and_zero_work() {
    let fixture = Fixture::new(39);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut original = fixture.builder();
    let mut legacy = fixture.builder();
    let input = original
        .declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            "input.tokens",
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    let owner = legacy
        .push(
            Code::Input("input.tokens".into()),
            39,
            Interval::new(0, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(owner, input.owner());
    let hash = |node: &Node| {
        let mut hash = Sha256::new();
        hash_node_header(&mut hash, node.lanes, node.bounds);
        let Code::Input(key) = &node.code else {
            panic!("original input")
        };
        // Original Input encoding from Builder::finish.
        hash.update([0]);
        hash.update((key.len() as u64).to_le_bytes());
        hash.update(key);
        <[u8; 32]>::from(hash.finalize())
    };
    assert_eq!(hash(&original.nodes[owner]), hash(&legacy.nodes[owner]));
    assert_eq!(
        schedule_work(&original.nodes).unwrap(),
        schedule_work(&legacy.nodes).unwrap()
    );
    assert!(legacy.original_inputs.is_empty());
    assert_eq!(legacy.original_inputs.capacity(), 0);
    assert!(original.allocated > legacy.allocated);
}

#[test]
fn finished_builder_cannot_reuse_original_input_ordinal_as_new_origin() {
    let fixture = Fixture::new(1);
    let registry = fixture.registry();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut builder = fixture.builder();
    let input = builder
        .declare_original_input(
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            "input.tokens",
            Interval::new(0, 2).unwrap(),
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    let imports = builder.nodes.len();
    // Finalization requires genuine interactive work, not a zero-work alias.
    // One original product preserves the output ordinal and exercises consuming
    // finalization without relaxing its positive-work gate.
    let output = builder
        .product(input.owner(), input.owner(), None, false)
        .unwrap();
    let _program = builder
        .finish(
            imports,
            vec![(output, 0)],
            &mut AdmissionSite::default(),
            |hash| {
                hash.update(b"original-input-lifecycle-test");
                Ok(())
            },
            || registry.binding(),
        )
        .unwrap();
    assert!(builder.nodes.is_empty());
    assert!(matches!(
        input.check(&builder, &registry, &fixture.plan, &fixture.plan.prefill),
        Err(Error::Binding)
    ));
    // Recreate ordinary source declarations through their genuine push path,
    // reaching the old token ordinal after consuming original finalization.
    for (key, source) in &fixture.sources {
        if source.public_q18.is_some() {
            continue;
        }
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
    assert_eq!(builder.nodes.len(), input.owner());
    assert!(matches!(
        builder.push(
            Code::Input("input.tokens".into()),
            1,
            Interval::new(0, 2).unwrap()
        ),
        Err(Error::Binding)
    ));
}
