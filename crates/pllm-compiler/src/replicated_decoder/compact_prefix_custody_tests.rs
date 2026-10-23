//! Genuine creation/frozen owners. No test import/drain capability constructor.
use super::*;
use crate::replicated_decoder::program::original_import;

struct Fixture {
    plan: DecoderPlan,
    sources: BTreeMap<String, SourceTensor>,
    norms: SourceNorms,
}

impl Fixture {
    fn new() -> Self {
        let config = serde_json::to_vec(&serde_json::json!({
            "model_type":"qwen2", "hidden_size":130, "intermediate_size":130,
            "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
            "head_dim":130, "vocab_size":3, "tie_word_embeddings":false,
            "hidden_act":"silu", "attention_bias":false, "max_position_embeddings":64,
            "rope_theta":10000, "rms_norm_eps":0.000001, "attention_dropout":0.0,
            "use_sliding_window":false, "use_cache":true, "max_window_layers":1
        }))
        .unwrap();
        let mut plan = lower_with_weights(&config, 3, 1, WeightMode::Public).unwrap();
        let mut norm = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::RmsNorm)
            .unwrap()
            .clone();
        norm.inputs = vec!["state.local".into()];
        norm.output_shape = vec![1, 3, 130];
        norm.state_kind = Some(StateKind::Recurrent);
        let id = norm.id.clone();
        plan.prefill.operations.retain(|op| op.id != id);
        plan.prefill.operations.insert(0, norm);
        let state = pllm_models::StateTensor {
            id: "state.local".into(),
            layer: plan.prefill.operations[0].layer,
            kind: StateKind::Recurrent,
            shape: vec![1, 3, 130],
            maximum_sequence: 3,
        };
        plan.prefill.state_inputs.push(state.clone());
        plan.prefill.state_outputs = vec![pllm_models::StateTensor {
            id,
            ..state.clone()
        }];
        plan.decode.state_inputs = vec![pllm_models::StateTensor {
            id: "next.state".into(),
            ..state
        }];
        let sources = Program::source_shapes_with_weights(&config, 3, 1, WeightMode::Public)
            .unwrap()
            .into_iter()
            .map(|(key, shape)| {
                let public = key.contains("norm");
                let data = vec![if public { ONE / 2 } else { 0i64 }; shape.iter().product()];
                let mut hash = Sha256::new();
                for value in &data {
                    hash.update(value.to_le_bytes());
                }
                let source = SourceTensor {
                    shape,
                    lower: data[0],
                    upper: data[0],
                    digest: hash.finalize().into(),
                    public_q18: public.then_some(data),
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                };
                (key, source)
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
            ExecutionPolicy {
                range_policy: RangePolicy::Checked,
                max_abs_activation: 1,
                ..ExecutionPolicy::default()
            },
            &self.norms,
        )
        .unwrap()
    }
    fn builder(&self) -> Builder {
        let mut b = Builder {
            public_weights: true,
            checked_bound: Some(ONE),
            ..Builder::default()
        };
        for (key, source) in self
            .sources
            .iter()
            .filter(|(_, source)| source.public_q18.is_none())
        {
            b.push(
                Code::PublicInput {
                    key: key.clone(),
                    digest: source.digest,
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
                source.shape.iter().product(),
                Interval::new(source.lower, source.upper).unwrap(),
            )
            .unwrap();
        }
        b
    }
    fn declare<'a>(
        &'a self,
        builder: &mut Builder,
        registry: &'a SourceRegistry<'a>,
        source: bool,
        prior: &RequestWork,
        owners: &RequestOwners<'_>,
    ) -> OriginalImportedView<'a> {
        let (graph, name) = if source {
            (&self.plan.prefill, "state.local")
        } else {
            (&self.plan.decode, "next.state")
        };
        builder
            .declare_original_input(
                registry,
                &self.plan,
                graph,
                name,
                Interval::new(i64::MIN, i64::MAX).unwrap(),
                prior,
                owners,
                MAX_PROGRAM_BYTES,
            )
            .unwrap()
    }
    fn frozen<'a>(
        &'a self,
        source: &'a Builder,
        bindings: &'a Vec<OriginalImportedView<'a>>,
        registry: &'a SourceRegistry<'a>,
        prior: &RequestWork,
        owners: &'a RequestOwners<'a>,
    ) -> FrozenRegion<'a> {
        let mut graph = CompactGraph::with_bound_inputs(
            source,
            self.validated(),
            registry,
            &self.plan,
            &self.plan.prefill,
            0,
            bindings,
            prior,
            owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
        let output = graph.lower_next().unwrap();
        graph.freeze(vec![output], MAX_PROGRAM_BYTES).unwrap()
    }
}

fn owners(values: &Vec<Option<Secret>>) -> RequestOwners<'_> {
    original_import::tests::owners(values)
}

#[test]
fn ordered_native_export_to_creation_seal_matches_independent_original_builder() {
    let f = Fixture::new();
    let registry = SourceRegistry::from_validated(f.validated()).unwrap();
    let values = Vec::new();
    let owners = owners(&values);
    let prior = RequestWork::default();
    let mut original = f.builder();
    let bindings = vec![f.declare(&mut original, &registry, true, &prior, &owners)];
    let frozen = f.frozen(&original, &bindings, &registry, &prior, &owners);
    let mut oracle = f.builder();
    let owner = oracle
        .push(
            Code::Input("state.local".into()),
            390,
            Interval::new(i64::MIN, i64::MAX).unwrap(),
        )
        .unwrap();
    let words: Words = (0..390).map(|lane| (owner, lane)).collect();
    let safe = oracle.checked_rows(&words, 130).unwrap();
    let gamma = &f.sources[f.plan.prefill.operations[0].attributes["weight"]
        .as_str()
        .unwrap()];
    let mut expected = Words::new();
    for row in safe.chunks_exact(130) {
        let owner = oracle.rms_norm_row(&row, 130, gamma).unwrap();
        expected.extend((0..130).map(|lane| (owner, lane)));
    }
    assert_eq!(frozen.local_validity_owner(), oracle.validity);
    let native = frozen.native_row_program(0).unwrap().clone();
    let mut destination = f.builder();
    let incoming = vec![f.declare(&mut destination, &registry, false, &prior, &owners)];
    let input_owner = incoming[0].owner();
    let pending = frozen
        .into_prefix(
            &destination,
            incoming,
            &registry,
            &f.plan,
            &f.plan.decode,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    assert_eq!(pending.kind(0).unwrap(), EdgeKind::Recurrent);
    assert_eq!(pending.epoch(), 0);
    assert_eq!(pending.next_epoch(), 1);
    assert_eq!(pending.sticky_source(), oracle.validity);
    for (lane, word) in expected.iter().enumerate() {
        assert_eq!(
            pending.mapping(0, lane).unwrap(),
            (word, (input_owner, lane))
        );
    }
    assert!(pending.mapping(0, 390).is_err());
    assert!(Arc::ptr_eq(
        pending.source().native_row_program(0).unwrap(),
        &native
    ));
    // No destination complete-row ledger, attention certificate or executable
    // bind_import can be obtained from this pending read-only recipe.
    pending
        .check_destination(&destination, &pending.inputs[0], 0)
        .unwrap();
}

#[test]
fn wrong_destination_same_ids_and_hash_cannot_replace_creation_record() {
    let f = Fixture::new();
    let registry = SourceRegistry::from_validated(f.validated()).unwrap();
    let values = Vec::new();
    let owners = owners(&values);
    let prior = RequestWork::default();
    let mut source = f.builder();
    let bindings = vec![f.declare(&mut source, &registry, true, &prior, &owners)];
    let frozen = f.frozen(&source, &bindings, &registry, &prior, &owners);
    let mut destination = f.builder();
    let incoming = vec![f.declare(&mut destination, &registry, false, &prior, &owners)];
    let mut foreign = f.builder();
    let impostor = f.declare(&mut foreign, &registry, false, &prior, &owners);
    assert_eq!(impostor.owner(), incoming[0].owner());
    assert_eq!(impostor.bounds(), incoming[0].bounds());
    let pending = frozen
        .into_prefix(
            &destination,
            incoming,
            &registry,
            &f.plan,
            &f.plan.decode,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    assert!(matches!(
        pending.check_destination(&foreign, &impostor, 0),
        Err(Error::Binding)
    ));
    assert!(matches!(
        pending.check_destination(&destination, &impostor, 0),
        Err(Error::Binding)
    ));
    assert!(matches!(
        pending.check_destination(&foreign, &pending.inputs[0], 0),
        Err(Error::Binding)
    ));
}

#[test]
fn foreign_equal_plan_and_registry_owners_refused_before_mapping_factory() {
    let f = Fixture::new();
    let other = Fixture::new();
    assert_eq!(f.plan, other.plan);
    let registry = SourceRegistry::from_validated(f.validated()).unwrap();
    let foreign_registry = SourceRegistry::from_validated(other.validated()).unwrap();
    let values = Vec::new();
    let owners = owners(&values);
    let prior = RequestWork::default();
    let mut source = f.builder();
    let bindings = vec![f.declare(&mut source, &registry, true, &prior, &owners)];
    let frozen = f.frozen(&source, &bindings, &registry, &prior, &owners);
    let weak = Arc::downgrade(frozen.native_row_program(0).unwrap());
    let mut destination = other.builder();
    let incoming = vec![other.declare(&mut destination, &foreign_registry, false, &prior, &owners)];
    MAPPING_FACTORIES.with(|n| n.set(0));
    assert!(matches!(
        frozen.into_prefix(
            &destination,
            incoming,
            &foreign_registry,
            &other.plan,
            &other.plan.decode,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    MAPPING_FACTORIES.with(|n| assert_eq!(n.get(), 0));
    assert!(
        weak.upgrade().is_none(),
        "consumed failure burns frozen native owner"
    );
}

#[test]
fn exact_and_one_short_gate_before_new_mapping_backing() {
    for short in [false, true] {
        let f = Fixture::new();
        let registry = SourceRegistry::from_validated(f.validated()).unwrap();
        let values = Vec::new();
        let owners = owners(&values);
        let prior = RequestWork::default();
        let mut source = f.builder();
        let bindings = vec![f.declare(&mut source, &registry, true, &prior, &owners)];
        let frozen = f.frozen(&source, &bindings, &registry, &prior, &owners);
        let mut destination = f.builder();
        let mut incoming = Vec::with_capacity(5);
        incoming.push(f.declare(&mut destination, &registry, false, &prior, &owners));
        let quote = frozen
            .prefix_quote(
                &destination,
                &incoming,
                &registry,
                &f.plan,
                &f.plan.decode,
                &prior,
                &owners,
                MAX_PROGRAM_BYTES,
            )
            .unwrap();
        assert_eq!(quote.mapping_capacity, 1);
        assert_eq!(quote.mapped_lanes, 390);
        MAPPING_FACTORIES.with(|n| n.set(0));
        let result = frozen.into_prefix(
            &destination,
            incoming,
            &registry,
            &f.plan,
            &f.plan.decode,
            &prior,
            &owners,
            quote.peak_bytes - usize::from(short),
        );
        MAPPING_FACTORIES.with(|n| assert_eq!(n.get(), usize::from(!short)));
        if short {
            assert!(matches!(result, Err(Error::Bounds)));
        } else {
            assert!(result.is_ok());
        }
    }
}

#[test]
fn no_redeclaration_or_numeric_interval_can_replay_consumed_destination_seal() {
    let f = Fixture::new();
    let registry = SourceRegistry::from_validated(f.validated()).unwrap();
    let values = Vec::new();
    let owners = owners(&values);
    let prior = RequestWork::default();
    let mut source = f.builder();
    let bindings = vec![f.declare(&mut source, &registry, true, &prior, &owners)];
    let frozen = f.frozen(&source, &bindings, &registry, &prior, &owners);
    let mut destination = f.builder();
    let incoming = vec![f.declare(&mut destination, &registry, false, &prior, &owners)];
    let pending = frozen
        .into_prefix(
            &destination,
            incoming,
            &registry,
            &f.plan,
            &f.plan.decode,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    pending.cancel();
    assert!(matches!(
        destination.declare_original_input(
            &registry,
            &f.plan,
            &f.plan.decode,
            "next.state",
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            &prior,
            &owners,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
}

#[test]
fn swapped_rows_cannot_replace_original_named_native_export() {
    let f = Fixture::new();
    let registry = SourceRegistry::from_validated(f.validated()).unwrap();
    let values = Vec::new();
    let owners = owners(&values);
    let prior = RequestWork::default();
    let mut source = f.builder();
    let bindings = vec![f.declare(&mut source, &registry, true, &prior, &owners)];
    let mut frozen = f.frozen(&source, &bindings, &registry, &prior, &owners);
    // Real producer retained, caller attempts substituted ordering, not a new
    // sealed producer. Source.resolve must reject before a map can be allocated.
    let original = frozen.exports[0].word(130).unwrap();
    frozen.exports[0].order = LaneOrder::HeadsSequence {
        heads: 2,
        rows: 2,
        width: 130,
        to_heads: true,
        base_start: 0,
    };
    assert_ne!(frozen.exports[0].word(130).unwrap(), original);
    let mut destination = f.builder();
    let incoming = vec![f.declare(&mut destination, &registry, false, &prior, &owners)];
    MAPPING_FACTORIES.with(|n| n.set(0));
    assert!(matches!(
        frozen.into_prefix(
            &destination,
            incoming,
            &registry,
            &f.plan,
            &f.plan.decode,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    MAPPING_FACTORIES.with(|n| assert_eq!(n.get(), 0));
}

#[test]
fn failed_mapping_replay_and_copy_guard_unwind_burn_whole_batch() {
    for unwind in [false, true] {
        let f = Fixture::new();
        let registry = SourceRegistry::from_validated(f.validated()).unwrap();
        let values = Vec::new();
        let owners = owners(&values);
        let prior = RequestWork::default();
        let mut source = f.builder();
        let bindings = vec![f.declare(&mut source, &registry, true, &prior, &owners)];
        let frozen = f.frozen(&source, &bindings, &registry, &prior, &owners);
        let native = Arc::downgrade(frozen.native_row_program(0).unwrap());
        let mut destination = f.builder();
        let incoming = vec![f.declare(&mut destination, &registry, false, &prior, &owners)];
        let pending = frozen
            .into_prefix(
                &destination,
                incoming,
                &registry,
                &f.plan,
                &f.plan.decode,
                &prior,
                &owners,
                MAX_PROGRAM_BYTES,
            )
            .unwrap();
        let erased = std::cell::Cell::new(false);
        struct BeforeProducer<'s> {
            native: &'s std::sync::Weak<tensor::Program>,
            erased: &'s std::cell::Cell<bool>,
        }
        impl Drop for BeforeProducer<'_> {
            fn drop(&mut self) {
                assert!(self.native.upgrade().is_some());
                self.erased.set(true);
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::replicated_decoder::runtime::body::handoff::test_restore_owner_order(
                BeforeProducer {
                    native: &native,
                    erased: &erased,
                },
                (),
                pending,
                |_, held| {
                    held.mapping(0, 0)?;
                    held.mapping(0, 129)?;
                    if unwind {
                        panic!("copy failure after first spans");
                    }
                    Err(Error::Allocation)
                },
            )
        }));
        if unwind {
            assert!(result.is_err());
        } else {
            assert!(matches!(result, Ok(Err(Error::Allocation))));
        }
        assert!(erased.get());
        assert!(native.upgrade().is_none());
    }
}

#[test]
fn different_kv_kind_never_coerces_to_recurrent_data() {
    assert_ne!(EdgeKind::Key, EdgeKind::Value);
    assert_ne!(EdgeKind::CacheIndices, EdgeKind::TokenFeedback);
    let mut f = Fixture::new();
    f.plan.decode.state_inputs[0].kind = StateKind::Key;
    let registry = SourceRegistry::from_validated(f.validated()).unwrap();
    let values = Vec::new();
    let owners = owners(&values);
    let prior = RequestWork::default();
    let mut source = f.builder();
    let bindings = vec![f.declare(&mut source, &registry, true, &prior, &owners)];
    let frozen = f.frozen(&source, &bindings, &registry, &prior, &owners);
    let mut destination = f.builder();
    let incoming = vec![f.declare(&mut destination, &registry, false, &prior, &owners)];
    assert!(matches!(
        frozen.into_prefix(
            &destination,
            incoming,
            &registry,
            &f.plan,
            &f.plan.decode,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    // Actual TokenFeedback/selected-ID execution awaits a frozen runtime owner;
    // this module deliberately defines no public/test executable import minter.
}

#[test]
fn genuine_feedback_alias_maps_but_raw_selected_id_is_not_a_token_import() {
    for raw_selection in [false, true] {
        let mut f = Fixture::new();
        let mut selection = f
            .plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::GreedyTokenSelection)
            .unwrap()
            .clone();
        let mut feedback = f
            .plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::TokenFeedback)
            .unwrap()
            .clone();
        selection.inputs = vec!["state.logits".into()];
        feedback.inputs = vec![selection.id.clone()];
        feedback.output_shape = vec![1, 1];
        let selected_id = selection.id.clone();
        let feedback_id = feedback.id.clone();
        f.plan
            .prefill
            .operations
            .retain(|op| op.id != selected_id && op.id != feedback_id);
        f.plan.prefill.operations.insert(0, feedback);
        f.plan.prefill.operations.insert(0, selection);
        f.plan.prefill.output = if raw_selection {
            selected_id
        } else {
            feedback_id
        };
        f.plan.prefill.state_inputs = vec![pllm_models::StateTensor {
            id: "state.logits".into(),
            layer: None,
            kind: StateKind::Recurrent,
            shape: vec![1, 3],
            maximum_sequence: 3,
        }];
        f.plan.prefill.state_outputs.clear();
        let registry = SourceRegistry::from_validated(f.validated()).unwrap();
        let values = Vec::new();
        let owners = owners(&values);
        let prior = RequestWork::default();
        let mut source = f.builder();
        let bindings = vec![
            source
                .declare_original_input(
                    &registry,
                    &f.plan,
                    &f.plan.prefill,
                    "state.logits",
                    Interval::new(-ONE, ONE).unwrap(),
                    &prior,
                    &owners,
                    MAX_PROGRAM_BYTES,
                )
                .unwrap(),
        ];
        let mut graph = CompactGraph::with_bound_inputs(
            &source,
            f.validated(),
            &registry,
            &f.plan,
            &f.plan.prefill,
            0,
            &bindings,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
        let selected = graph.lower_next().unwrap();
        let exported = if raw_selection {
            selected
        } else {
            graph.lower_next().unwrap()
        };
        let frozen = graph.freeze(vec![exported], MAX_PROGRAM_BYTES).unwrap();
        let mut destination = f.builder();
        let incoming = vec![
            destination
                .declare_original_input(
                    &registry,
                    &f.plan,
                    &f.plan.decode,
                    "input.tokens",
                    Interval::new(0, 2).unwrap(),
                    &prior,
                    &owners,
                    MAX_PROGRAM_BYTES,
                )
                .unwrap(),
        ];
        MAPPING_FACTORIES.with(|n| n.set(0));
        let result = frozen.into_prefix(
            &destination,
            incoming,
            &registry,
            &f.plan,
            &f.plan.decode,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES,
        );
        if raw_selection {
            assert!(matches!(result, Err(Error::Binding)));
            MAPPING_FACTORIES.with(|n| assert_eq!(n.get(), 0));
        } else {
            let pending = result.unwrap();
            assert_eq!(pending.kind(0).unwrap(), EdgeKind::TokenFeedback);
            assert_eq!(pending.mapping(0, 0).unwrap().1.1, 0);
            assert!(pending.mapping(0, 1).is_err());
        }
    }
}

#[test]
fn accumulated_original_work_rejects_before_mapping_allocation() {
    let f = Fixture::new();
    let registry = SourceRegistry::from_validated(f.validated()).unwrap();
    let values = Vec::new();
    let owners = owners(&values);
    let prior = RequestWork::default();
    let mut source = f.builder();
    let bindings = vec![f.declare(&mut source, &registry, true, &prior, &owners)];
    let frozen = f.frozen(&source, &bindings, &registry, &prior, &owners);
    let mut destination = f.builder();
    let incoming = vec![f.declare(&mut destination, &registry, false, &prior, &owners)];
    // Saturate the existing work ledger using actual original native recipe
    // work. These work numbers never mint source/copy/row authority.
    let work = frozen.work_budget();
    let mut low = 0u64;
    let mut high = u64::MAX / work.operations().max(1);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if work
            .checked_mul(mid)
            .is_ok_and(|work| work.limits().is_ok())
        {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    let mut saturated = RequestWork::default();
    saturated
        .record_work(work.checked_mul(low).unwrap())
        .unwrap();
    MAPPING_FACTORIES.with(|n| n.set(0));
    assert!(matches!(
        frozen.into_prefix(
            &destination,
            incoming,
            &registry,
            &f.plan,
            &f.plan.decode,
            &saturated,
            &owners,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Bounds)
    ));
    MAPPING_FACTORIES.with(|n| assert_eq!(n.get(), 0));
}

#[test]
fn omitted_sibling_burns_both_native_exports_and_all_destination_seals() {
    let mut f = Fixture::new();
    let mut sibling = f
        .plan
        .prefill
        .operations
        .iter()
        .skip(1)
        .find(|op| op.operator == Op::RmsNorm)
        .unwrap()
        .clone();
    let sibling_id = sibling.id.clone();
    sibling.inputs = vec!["state.sibling".into()];
    sibling.output_shape = vec![1, 3, 130];
    sibling.state_kind = Some(StateKind::Convolution);
    let state = pllm_models::StateTensor {
        id: "state.sibling".into(),
        layer: sibling.layer,
        kind: StateKind::Convolution,
        shape: vec![1, 3, 130],
        maximum_sequence: 3,
    };
    f.plan.prefill.state_inputs.push(state.clone());
    f.plan.prefill.state_outputs.push(pllm_models::StateTensor {
        id: sibling_id.clone(),
        ..state.clone()
    });
    f.plan.decode.state_inputs.push(pllm_models::StateTensor {
        id: "next.sibling".into(),
        ..state
    });
    f.plan.prefill.operations.retain(|op| op.id != sibling_id);
    f.plan.prefill.operations.insert(1, sibling);
    let registry = SourceRegistry::from_validated(f.validated()).unwrap();
    let values = Vec::new();
    let owners = owners(&values);
    let prior = RequestWork::default();
    let mut source = f.builder();
    let first = f.declare(&mut source, &registry, true, &prior, &owners);
    let second = source
        .declare_original_input(
            &registry,
            &f.plan,
            &f.plan.prefill,
            "state.sibling",
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            &prior,
            &owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    let bindings = vec![first, second];
    let mut graph = CompactGraph::with_bound_inputs(
        &source,
        f.validated(),
        &registry,
        &f.plan,
        &f.plan.prefill,
        0,
        &bindings,
        &prior,
        &owners,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let first = graph.lower_next().unwrap();
    let second = graph.lower_next().unwrap();
    let frozen = graph
        .freeze(vec![first, second], MAX_PROGRAM_BYTES)
        .unwrap();
    let a = Arc::downgrade(frozen.native_row_program(0).unwrap());
    let b = Arc::downgrade(frozen.native_row_program(1).unwrap());
    let mut destination = f.builder();
    let first = f.declare(&mut destination, &registry, false, &prior, &owners);
    let second = destination
        .declare_original_input(
            &registry,
            &f.plan,
            &f.plan.decode,
            "next.sibling",
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            &prior,
            &owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    drop(second); // A descriptor omission cannot authorize a partial batch.
    MAPPING_FACTORIES.with(|n| n.set(0));
    assert!(matches!(
        frozen.into_prefix(
            &destination,
            vec![first],
            &registry,
            &f.plan,
            &f.plan.decode,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    MAPPING_FACTORIES.with(|n| assert_eq!(n.get(), 0));
    assert!(a.upgrade().is_none());
    assert!(b.upgrade().is_none());
    assert!(matches!(
        destination.declare_original_input(
            &registry,
            &f.plan,
            &f.plan.decode,
            "next.sibling",
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            &prior,
            &owners,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
}
