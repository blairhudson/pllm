//! Actual guarded source factory, authenticated files and original graph only.
use super::super::request_driver::drive_owned as drive;
use super::*;

#[path = "request_schedule_tests.rs"]
mod schedule_tests;
use crate::plan_input::json;
use pllm_garble::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};

const TOKENS: [u32; 4] = [1, 2, 1, 2];

fn limits() -> original_source::Limits {
    original_source::Limits {
        expansion: pllm_models::DecoderExpansionBudget {
            max_layers: 8,
            max_feature_width: 128,
            max_attention_heads: 8,
            max_vocabulary: 128,
            max_tensor_elements: 32768,
        },
        factory: pllm_models::ConfigLowererLimits {
            max_operations_per_phase: 256,
            max_string_bytes: 1024,
            max_attribute_nodes: 32768,
            max_retained_strings: 1 << 20,
            max_factory_bytes: 64 << 20,
        },
        max_work: 1_000_000_000_000,
        max_working_payload: 128 << 20,
    }
}

fn prepared(
    root: &OwnedBudget,
    config: &[u8],
    policy: original_source::Limits,
) -> original_source::SourceQuote {
    // The byte and JSON factories themselves run under this original counter.
    let raw = root
        .charge_external(config.len())
        .unwrap()
        .try_build(|_| Ok(config.to_vec()))
        .unwrap();
    let imported = json::quote(
        raw,
        root.charge_external(json::bootstrap_bytes().unwrap())
            .unwrap(),
        json::Limits {
            max_source_bytes: 32768,
            max_nodes: 512,
            max_depth: 16,
            max_string_bytes: 16384,
            max_number_bytes: 64,
            max_number_work: 1_000_000,
            max_work: 100_000_000,
            max_payload_bytes: 4 << 20,
        },
    )
    .unwrap()
    .construct()
    .unwrap();
    let bytes = original_source::bootstrap_bytes(&imported, policy).unwrap();
    original_source::quote(
        imported,
        root.charge_external(bytes).unwrap(),
        pllm_models::DecoderWorkload {
            batch: 1,
            max_input_tokens: 4,
            max_new_tokens: 2,
        },
        policy,
    )
    .unwrap()
}

struct Fixture {
    manifest: Manifest,
    public: Arc<PublicSource>,
}
impl Fixture {
    fn new() -> Self {
        let (config, _, mut clear) =
            crate::replicated_decoder::tests::fixture_sources("qwen2", false);
        // Real authenticated unused row zero forces the unchanged checked
        // profile to emit a guard. Prompt and first feedback select safe rows;
        // no post-compilation interval or certificate is substituted.
        clear.get_mut("model.embed_tokens.weight").unwrap()[..2].fill(3 * ONE / 2);
        let execution = ExecutionPolicy {
            range_policy: RangePolicy::Checked,
            max_abs_activation: 1,
            ..ExecutionPolicy::default()
        };
        // Genuine existing Publisher/codec computes the original source stats.
        // Its compiled identity is TEST-ONLY oracle material, never permit input.
        let mut publisher = Publisher::new_with_policy(
            &config,
            4,
            2,
            [1; 32],
            [2; 32],
            WeightMode::Public,
            execution,
        )
        .unwrap();
        // Release the shapes borrow before mutable append; don't clone extents.
        let keys: Vec<_> = publisher
            .shapes()
            .iter()
            .map(|(key, _)| key.clone())
            .collect();
        for key in keys {
            let bytes: Vec<_> = clear[&key]
                .iter()
                .flat_map(|v| (*v as f32 / ONE as f32).to_le_bytes())
                .collect();
            publisher.append(&key, 0, &bytes).unwrap();
        }
        publisher.seal(&TOKENS).unwrap();
        let mut manifest: Manifest =
            serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
        // Explicitly prove no expanded Program identity is request authority.
        manifest.program = [0; 32];
        let mut importer = PublicSourceImport::new(&manifest).unwrap();
        for key in manifest.tensors.keys() {
            let bytes: Vec<_> = clear[key]
                .iter()
                .flat_map(|v| (*v as f32 / ONE as f32).to_le_bytes())
                .collect();
            importer.append(key, 0, &bytes).unwrap();
        }
        Self {
            manifest,
            public: Arc::new(importer.finish().unwrap()),
        }
    }
    fn sources<'s>(&'s self, registry: &'s SourceRegistry<'s>) -> RequestSources<'s> {
        RequestSources::new(&self.manifest, registry, Arc::clone(&self.public)).unwrap()
    }
}

fn imports(manifest: &Manifest) -> (Builder, BTreeMap<String, SourceView>) {
    let mut builder = Builder {
        public_weights: true,
        checked_bound: manifest.execution.checked_bound(),
        lookup: manifest.execution.lookup,
        selection: manifest.execution.selection,
        ..Builder::default()
    };
    builder.allocated = source_norms::import_envelope(&manifest.norms)
        .unwrap()
        .total_bytes;
    let mut inputs = BTreeMap::new();
    for (key, source) in &manifest.tensors {
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
        inputs.insert(
            key.clone(),
            SourceView {
                owner,
                shape: source.shape.clone(),
            },
        );
        if let Some(norm) = manifest.norms.get(key) {
            builder.public_norms.insert(owner, norm.clone());
        }
    }
    (builder, inputs)
}

struct Case<'s> {
    fixture: &'s Fixture,
    registry: &'s SourceRegistry<'s>,
    inputs: BTreeMap<String, SourceView>,
    builder: Builder,
    permit: Option<RequestInputPermit<'s>>,
    count: usize,
    root: OwnedBudget,
}
impl<'s> Case<'s> {
    fn new(fixture: &'s Fixture, registry: &'s SourceRegistry<'s>) -> Self {
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let original = prepared(&root, &fixture.manifest.config, limits())
            .construct()
            .unwrap();
        let (mut builder, inputs) = imports(&fixture.manifest);
        let context = fixture.sources(registry);
        let required = quote(&builder, &original, &context).unwrap();
        let permit = declare(
            &mut builder,
            original,
            context,
            root.charge_external(required.payload_bytes).unwrap(),
        )
        .unwrap();
        let count = builder.nodes.len();
        Self {
            fixture,
            registry,
            inputs,
            builder,
            permit: Some(permit),
            count,
            root,
        }
    }
    fn run(&mut self) -> PendingRequest<'s> {
        let permit = self.permit.take().unwrap();
        let controller = self
            .root
            .charge_external(permit.lowering_control_bytes(limits()).unwrap())
            .unwrap();
        drive(
            permit,
            &mut self.builder,
            &self.inputs,
            &self.fixture.manifest.tensors,
            controller,
            limits(),
            &mut AdmissionSite::default(),
        )
        .unwrap()
    }
}
fn with_case(run: impl FnOnce(Case<'_>)) {
    let fixture = Fixture::new();
    let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
    run(Case::new(&fixture, &registry));
}
fn program(
    builder: &mut Builder,
    count: usize,
    selected: Vec<(usize, usize)>,
    binding: [u8; 32],
) -> Program {
    builder
        .finish(
            count,
            selected,
            &mut AdmissionSite::default(),
            |hash| {
                hash.update(b"original-request-history-unit-oracle");
                Ok(())
            },
            || binding,
        )
        .unwrap()
}

#[test]
fn four_original_plus_real_feedback_makes_exact_five_token_creation_alias_seal() {
    with_case(|mut case| {
        let pending = case.run();
        let first = pending.input(0).unwrap();
        let second = pending.input(1).unwrap();
        assert_eq!((first.rows(), second.rows()), (4, 5));
        assert!(second.follows(&first));
        assert!(!first.follows(&second));
        assert_eq!(first.plan().prefill.query_sequence, 4);
        assert_eq!(second.plan().prefill.query_sequence, 5);
        assert!(!std::ptr::eq(first.plan(), second.plan()));
        for lane in 0..4 {
            assert_eq!(first.word(lane).unwrap(), second.word(lane).unwrap());
        }
        assert_eq!(
            second.word(4).unwrap(),
            pending.history.fragments[0].selection.word
        );
        assert!(first.word(4).is_err());
        assert!(second.word(5).is_err());
        assert_eq!(
            pending.history.fragments[1].sticky_in,
            pending.history.fragments[0].sticky_out
        );
        assert!(pending.history.fragments[0].sticky_out.is_some());
        assert_eq!(pending.history.epochs.capacity(), 2);
        assert_eq!(pending.history.fragments.capacity(), 2);
        assert_eq!(pending.selected.capacity(), 3);
        assert!(Arc::ptr_eq(
            &pending.history.origin.payload().public,
            &case.fixture.public
        ));
        assert_eq!(case.fixture.manifest.program, [0; 32]);
    });
}

#[test]
fn owned_driver_and_acceptance_preserve_independent_original_builder_ids_hash_and_work() {
    with_case(|mut case| {
        let pending = case.run();
        let completed = case.builder.final_acceptance_owned(pending).unwrap();
        let selected = completed.outputs().to_vec();
        let binding = case.registry.binding();
        let own = program(&mut case.builder, case.count, selected.clone(), binding);
        let (mut oracle, inputs) = imports(&case.fixture.manifest);
        let input = oracle
            .push(
                Code::Input("input.tokens".into()),
                4,
                Interval::new(0, 2).unwrap(),
            )
            .unwrap();
        let count = oracle.nodes.len();
        let mut tokens = oracle.view(input, vec![1, 4]);
        let mut expected = Vec::new();
        // Independent original eager loop, NOT the shared request driver.
        for selection in 0..2 {
            let plan = lower_with_weights(
                &case.fixture.manifest.config,
                4 + selection,
                2 - selection,
                WeightMode::Public,
            )
            .unwrap();
            let output = oracle
                .graph(
                    &plan.prefill,
                    &tokens,
                    &inputs,
                    &case.fixture.manifest.tensors,
                    3,
                    0,
                    &mut BTreeMap::new(),
                    &mut AdmissionSite::default(),
                )
                .unwrap();
            assert_eq!(output.words.len(), 1);
            expected.extend(output.words.iter());
            tokens.words.extend(output.words.iter());
            tokens.shape = vec![1, tokens.words.len()];
        }
        oracle.final_acceptance(input, 2, &mut expected).unwrap();
        assert_eq!(selected, expected);
        let reference = program(&mut oracle, count, expected, binding);
        assert_eq!(own.digest(), reference.digest());
        assert_eq!(own.work_budget(), reference.work_budget());
        assert_eq!(own.nodes.len(), reference.nodes.len());
        assert!(completed
            .input(1)
            .unwrap()
            .follows(&completed.input(0).unwrap()));
    });
}

#[test]
fn foreign_same_shape_same_source_hash_builder_is_not_original_input_owner() {
    with_case(|mut a| {
        with_case(|mut b| {
            let permit = a.permit.take().unwrap();
            let origin = Arc::clone(&permit.history.origin);
            let controller = a
                .root
                .charge_external(permit.lowering_control_bytes(limits()).unwrap())
                .unwrap();
            let before = b.builder.nodes.len();
            assert_eq!(
                origin.payload().source.plan(),
                b.permit
                    .as_ref()
                    .unwrap()
                    .history
                    .origin
                    .payload()
                    .source
                    .plan()
            );
            assert!(!origin
                .payload()
                .source
                .same_origin(&b.permit.as_ref().unwrap().history.origin.payload().source));
            assert!(matches!(
                drive(
                    permit,
                    &mut b.builder,
                    &b.inputs,
                    &b.fixture.manifest.tensors,
                    controller,
                    limits(),
                    &mut AdmissionSite::default()
                ),
                Err(Error::Binding)
            ));
            assert_eq!(b.builder.nodes.len(), before);
            assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
        })
    });
}

#[test]
fn reordered_root_tokens_and_swapped_actual_epochs_reject_before_epilogue_factory() {
    for swap in [false, true] {
        with_case(|mut case| {
            let mut pending = case.run();
            let origin = Arc::clone(&pending.history.origin);
            if swap {
                pending.history.epochs.swap(0, 1);
            } else {
                let mut words: Vec<_> = pending.tokens.words.iter().collect();
                words.swap(0, 1);
                pending.tokens.words = words.into_iter().collect();
            }
            let before = case.builder.nodes.len();
            assert!(matches!(
                case.builder.final_acceptance_owned(pending),
                Err(Error::Binding)
            ));
            assert_eq!(case.builder.nodes.len(), before);
            assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
        });
    }
}

#[test]
fn foreign_request_history_links_do_not_follow_even_same_original_config() {
    with_case(|mut a| {
        with_case(|mut b| {
            let a = a.run();
            let b = b.run();
            assert_eq!(a.input(0).unwrap().plan(), b.input(0).unwrap().plan());
            assert!(!a.input(1).unwrap().follows(&b.input(0).unwrap()));
        })
    });
}

#[test]
fn genuine_foreign_same_origin_same_workload_epoch_is_not_the_feedback_creator() {
    with_case(|mut case| {
        let mut pending = case.run();
        let origin = Arc::clone(&pending.history.origin);
        let source = &origin.payload().source;
        let bytes = source.lower_bootstrap_bytes(limits()).unwrap();
        let foreign = source
            .lower(
                case.root.charge_external(bytes).unwrap(),
                pllm_models::DecoderWorkload {
                    batch: 1,
                    max_input_tokens: 5,
                    max_new_tokens: 1,
                },
                limits(),
            )
            .unwrap();
        assert!(foreign.same_origin(&pending.history.epochs[1]));
        assert_eq!(foreign.plan(), pending.history.epochs[1].plan());
        assert!(!foreign.same_plan(&pending.history.epochs[1]));
        pending.history.epochs[1] = Arc::new(foreign);
        let before = case.builder.nodes.len();
        assert!(matches!(
            case.builder.final_acceptance_owned(pending),
            Err(Error::Binding)
        ));
        assert_eq!(case.builder.nodes.len(), before);
        assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
    });
}

#[test]
fn exact_metadata_capacity_admits_one_short_refuses_before_input_creation() {
    let fixture = Fixture::new();
    let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
    for short in [true, false] {
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let source = prepared(&root, &fixture.manifest.config, limits())
            .construct()
            .unwrap();
        let (mut builder, _) = imports(&fixture.manifest);
        let context = fixture.sources(&registry);
        let resource = quote(&builder, &source, &context).unwrap();
        let before = builder.nodes.len();
        let mints = MINTS.with(|n| n.get());
        let result = declare(
            &mut builder,
            source,
            context,
            root.charge_external(resource.payload_bytes - usize::from(short))
                .unwrap(),
        );
        if short {
            assert!(matches!(result, Err(Error::Bounds)));
            assert_eq!(builder.nodes.len(), before);
            assert_eq!(MINTS.with(|n| n.get()), mints);
        } else {
            let permit = result.unwrap();
            assert_eq!(builder.nodes.len(), before + 1);
            assert_eq!(
                permit.history.charge.payload_bytes()
                    + size_of::<OwnedExternalCharge>()
                    + size_of::<Origin>()
                    + 2 * size_of::<usize>(),
                resource.payload_bytes
            );
        }
    }
}

#[test]
fn exact_actual_counter_transition_and_one_short_inline_growth_precede_input_factory() {
    let fixture = Fixture::new();
    let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
    const CAP: usize = MAX_ALLOCATION_BYTES;
    for short in [false, true] {
        let root = OwnedBudget::new(CAP).unwrap();
        let source = prepared(&root, &fixture.manifest.config, limits())
            .construct()
            .unwrap();
        let (mut builder, _) = imports(&fixture.manifest);
        let context = fixture.sources(&registry);
        let resource = quote(&builder, &source, &context).unwrap();
        // Independent literal API contract: charge_external adds its guard;
        // try_build adds exactly inline/padding BEFORE invoking the Input hook.
        let literal = resource.payload_bytes
            + size_of::<OwnedExternalCharge>()
            + size_of::<OwnedCharged<Origin>>()
            - size_of::<OwnedExternalCharge>();
        assert_eq!(resource.required_counter_bytes, literal);
        // Real held counter-pressure reservation, not a physical-buffer claim.
        let _pressure = root
            .charge_external(
                CAP - root.budget().used_bytes()
                    - resource.required_counter_bytes
                    - size_of::<OwnedExternalCharge>()
                    + usize::from(short),
            )
            .unwrap();
        let before = builder.nodes.len();
        let mints = MINTS.with(|n| n.get());
        let result = declare(
            &mut builder,
            source,
            context,
            root.charge_external(resource.payload_bytes).unwrap(),
        );
        if short {
            assert!(matches!(result, Err(Error::Bounds)));
            assert_eq!(builder.nodes.len(), before);
            assert_eq!(MINTS.with(|n| n.get()), mints);
        } else {
            let _permit = result.unwrap();
            assert_eq!(root.budget().used_bytes(), CAP);
            assert_eq!(builder.nodes.len(), before + 1);
        }
    }
}

#[test]
fn borrowed_source_spare_capacity_is_literal_retained_baseline_not_alias_credit() {
    let mut fixture = Fixture::new();
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = prepared(&root, &fixture.manifest.config, limits())
        .construct()
        .unwrap();
    let (builder, _) = imports(&fixture.manifest);
    let first = {
        let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
        quote(&builder, &source, &fixture.sources(&registry)).unwrap()
    };
    let before = fixture.manifest.config.capacity();
    fixture.manifest.config.reserve_exact(before + 113);
    let delta = fixture.manifest.config.capacity() - before;
    assert!(delta > 0);
    let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
    let second = quote(&builder, &source, &fixture.sources(&registry)).unwrap();
    assert_eq!(second.payload_bytes - first.payload_bytes, delta);
    assert_eq!(
        second.required_counter_bytes - first.required_counter_bytes,
        delta
    );
}

#[test]
fn one_short_cumulative_source_ledger_work_refuses_before_token_input_factory() {
    let fixture = Fixture::new();
    let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let initial = prepared(&root, &fixture.manifest.config, limits());
    let prior = initial.resources().prepaid_work;
    let source = initial.construct().unwrap();
    let (builder, _) = imports(&fixture.manifest);
    let metadata = quote(&builder, &source, &fixture.sources(&registry))
        .unwrap()
        .metadata_work;
    drop(source);
    let mut tight = limits();
    tight.max_work = prior + metadata - 1;
    let source = prepared(&root, &fixture.manifest.config, tight)
        .construct()
        .unwrap();
    let (mut builder, _) = imports(&fixture.manifest);
    let context = fixture.sources(&registry);
    let quote = quote(&builder, &source, &context).unwrap();
    let before = builder.nodes.len();
    assert!(matches!(
        declare(
            &mut builder,
            source,
            context,
            root.charge_external(quote.payload_bytes).unwrap()
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(builder.nodes.len(), before);
}

#[test]
fn foreign_controller_and_missing_future_control_burn_without_native_prefill() {
    for foreign in [false, true] {
        with_case(|mut case| {
            let other = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let permit = case.permit.take().unwrap();
            let origin = Arc::clone(&permit.history.origin);
            let required = permit.lowering_control_bytes(limits()).unwrap();
            let controller = if foreign {
                other.charge_external(required).unwrap()
            } else {
                case.root.charge_external(required - 1).unwrap()
            };
            let before = case.builder.nodes.len();
            assert!(drive(
                permit,
                &mut case.builder,
                &case.inputs,
                &case.fixture.manifest.tensors,
                controller,
                limits(),
                &mut AdmissionSite::default()
            )
            .is_err());
            assert_eq!(case.builder.nodes.len(), before);
            assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
        });
    }
}

#[test]
fn genuine_source_ledger_addition_overflow_refuses_before_input_factory() {
    let fixture = Fixture::new();
    let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut unlimited = limits();
    unlimited.max_work = u64::MAX;
    let source = prepared(&root, &fixture.manifest.config, unlimited)
        .construct()
        .unwrap();
    let (mut builder, _) = imports(&fixture.manifest);
    let context = fixture.sources(&registry);
    let required = quote(&builder, &source, &context).unwrap();
    let guard = root
        .charge_external(OriginalSourcePlan::compiler_metadata_control_bytes().unwrap())
        .unwrap();
    source
        .prepay_compiler_metadata(&guard, u64::MAX - source.resources().prepaid_work)
        .unwrap();
    let before = builder.nodes.len();
    assert!(matches!(
        declare(
            &mut builder,
            source,
            context,
            root.charge_external(required.payload_bytes).unwrap()
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(builder.nodes.len(), before);
}

// Adversarial TEST-ONLY reassembly, never a production permit constructor.
fn stale_for_test<'s>(pending: &PendingRequest<'s>, root: &OwnedBudget) -> RequestInputPermit<'s> {
    let origin = Arc::clone(&pending.history.origin);
    let rows = pending.history.resources.rows;
    let input = origin.payload().input.owner();
    RequestInputPermit {
        tokens: View {
            words: (0..rows).map(|lane| (input, lane)).collect(),
            shape: vec![1, rows],
        },
        history: History {
            epochs: vec![Arc::clone(&origin.payload().source)],
            fragments: Vec::new(),
            scratch: Words::try_with_spans(pending.history.resources.outputs + 1).unwrap(),
            sources: RequestSources {
                manifest: pending.history.sources.manifest,
                registry: pending.history.sources.registry,
                public: Arc::clone(&pending.history.sources.public),
            },
            origin: HistoryOrigin(origin),
            resources: pending.history.resources,
            sticky: None,
            charge: root
                .charge_external(pending.history.resources.payload_bytes)
                .unwrap(),
        },
    }
}

#[test]
fn cancellation_and_completed_history_burn_replay_without_new_emission() {
    for accept in [false, true] {
        with_case(|mut case| {
            let pending = case.run();
            let stale = stale_for_test(&pending, &case.root);
            let origin = Arc::clone(&pending.history.origin);
            if accept {
                let _completed = case.builder.final_acceptance_owned(pending).unwrap();
            } else {
                pending.cancel();
            }
            assert_ne!(origin.payload().state.load(Ordering::Acquire), OPEN);
            let controller = case
                .root
                .charge_external(stale.lowering_control_bytes(limits()).unwrap())
                .unwrap();
            let before = case.builder.nodes.len();
            assert!(matches!(
                drive(
                    stale,
                    &mut case.builder,
                    &case.inputs,
                    &case.fixture.manifest.tensors,
                    controller,
                    limits(),
                    &mut AdmissionSite::default()
                ),
                Err(Error::Binding)
            ));
            assert_eq!(case.builder.nodes.len(), before);
        });
    }
}

#[test]
fn unwind_burns_request_but_retains_genuine_source_until_last_record_owner_drops() {
    with_case(|mut case| {
        let pending = case.run();
        let origin = Arc::clone(&pending.history.origin);
        let before = case.root.budget().used_bytes();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _pending = pending;
            panic!("request compiler cancellation");
        }))
        .is_err());
        assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
        assert_eq!(origin.payload().source.plan().prefill.query_sequence, 4);
        assert!(case.root.budget().used_bytes() < before);
        assert_eq!(bytes::<Fragment>(usize::MAX), Err(Error::Bounds));
        assert_eq!(add(usize::MAX, 1), Err(Error::Bounds));
    });
}

#[test]
fn foreign_descriptor_maps_and_substituted_config_refuse_before_input_creation() {
    let fixture = Fixture::new();
    let foreign = Fixture::new();
    let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
    assert!(matches!(
        RequestSources::new(&foreign.manifest, &registry, Arc::clone(&fixture.public)),
        Err(Error::Binding)
    ));
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut changed = fixture.manifest.config.clone();
    changed.push(b' ');
    let source = prepared(&root, &changed, limits()).construct().unwrap();
    let (mut builder, _) = imports(&fixture.manifest);
    let context = fixture.sources(&registry);
    let resource = quote(&builder, &source, &context).unwrap();
    let before = builder.nodes.len();
    assert!(matches!(
        declare(
            &mut builder,
            source,
            context,
            root.charge_external(resource.payload_bytes).unwrap()
        ),
        Err(Error::Binding)
    ));
    assert_eq!(builder.nodes.len(), before);
}

#[test]
fn an_existing_original_token_input_cannot_be_retroactively_minted_again() {
    with_case(|mut case| {
        let source = prepared(&case.root, &case.fixture.manifest.config, limits())
            .construct()
            .unwrap();
        let context = case.fixture.sources(case.registry);
        let required = quote(&case.builder, &source, &context).unwrap();
        let before = case.builder.nodes.len();
        let mints = MINTS.with(|n| n.get());
        assert!(matches!(
            declare(
                &mut case.builder,
                source,
                context,
                case.root.charge_external(required.payload_bytes).unwrap()
            ),
            Err(Error::Binding)
        ));
        assert_eq!(case.builder.nodes.len(), before);
        assert_eq!(MINTS.with(|n| n.get()), mints);
        // Different failed declaration did not consume the existing input's
        // linear permit. It still must produce real events before any history.
        let pending = case.run();
        assert_eq!(pending.input(1).unwrap().rows(), 5);
    });
}

#[test]
fn changed_selected_record_and_removed_sticky_owner_cannot_complete_history() {
    for sticky in [false, true] {
        with_case(|mut case| {
            let pending = case.run();
            let origin = Arc::clone(&pending.history.origin);
            if sticky {
                case.builder.validity = None;
            } else {
                let selected = pending.history.fragments[0].selection.word.0;
                case.builder.nodes[selected].bounds = Interval::new(-1, 1).unwrap();
            }
            let before = case.builder.nodes.len();
            assert!(matches!(
                case.builder.final_acceptance_owned(pending),
                Err(Error::Binding)
            ));
            assert_eq!(case.builder.nodes.len(), before);
            assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
        });
    }
}

#[test]
fn emission_failure_or_unwind_clears_builder_and_burns_all_pending_custody() {
    for unwind in [false, true] {
        with_case(|mut case| {
            let permit = case.permit.take().unwrap();
            let origin = Arc::clone(&permit.history.origin);
            let controller = case
                .root
                .charge_external(permit.lowering_control_bytes(limits()).unwrap())
                .unwrap();
            SELECTION_FAULT.with(|slot| slot.set(if unwind { 2 } else { 1 }));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                drive(
                    permit,
                    &mut case.builder,
                    &case.inputs,
                    &case.fixture.manifest.tensors,
                    controller,
                    limits(),
                    &mut AdmissionSite::default(),
                )
            }));
            SELECTION_FAULT.with(|slot| slot.set(0));
            if unwind {
                assert!(result.is_err());
            } else {
                assert!(matches!(result, Ok(Err(Error::Binding))));
            }
            assert!(case.builder.request_emission.is_none());
            assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
        });
    }
}

#[test]
fn arbitrary_scalar_alias_cannot_replace_genuine_token_feedback_publication() {
    with_case(|mut case| {
        let RequestInputPermit {
            mut history,
            mut tokens,
        } = case.permit.take().unwrap();
        let origin = Arc::clone(&history.origin);
        let before: Vec<_> = tokens.words.iter().collect();
        let scalar = View {
            words: tokens.words.slice(0..1),
            shape: vec![1],
        };
        // Adversarial TEST-ONLY assembly. The production active constructor is
        // closed inside BoundExpanded::prefill; no event or producer is forged.
        case.builder.request_emission = Some(Active {
            origin: Arc::clone(&origin),
            plan: Arc::clone(&history.epochs[0]),
            epoch: 0,
            sticky_in: None,
            selected: None,
            feedback: None,
            input_words: tokens.words.clone(),
            committed: false,
        });
        let mut sink = BoundExpanded {
            builder: &mut case.builder,
            inputs: &case.inputs,
            sources: &case.fixture.manifest.tensors,
            history: &mut history,
        };
        let mut selected = reserve(history_output_capacity(sink.history)).unwrap();
        use request_driver::RequestSink;
        assert!(matches!(
            sink.append_feedback(&mut tokens, &scalar, &mut selected),
            Err(Error::Binding)
        ));
        assert!(selected.is_empty());
        assert_eq!(tokens.words, before);
        assert!(sink.builder.request_emission.is_none());
        assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
    });
}

fn history_output_capacity(history: &History<'_>) -> usize {
    history.resources.outputs + 1
}

#[test]
fn foreign_public_input_mapping_refuses_before_any_native_graph_entry() {
    with_case(|mut case| {
        let keys: Vec<_> = case.inputs.keys().cloned().collect();
        assert!(keys.len() >= 2);
        let a = case.inputs[&keys[0]].owner;
        let b = case.inputs[&keys[1]].owner;
        case.inputs.get_mut(&keys[0]).unwrap().owner = b;
        case.inputs.get_mut(&keys[1]).unwrap().owner = a;
        let permit = case.permit.take().unwrap();
        let origin = Arc::clone(&permit.history.origin);
        let controller = case
            .root
            .charge_external(permit.lowering_control_bytes(limits()).unwrap())
            .unwrap();
        let before = case.builder.nodes.len();
        assert!(matches!(
            drive(
                permit,
                &mut case.builder,
                &case.inputs,
                &case.fixture.manifest.tensors,
                controller,
                limits(),
                &mut AdmissionSite::default()
            ),
            Err(Error::Binding)
        ));
        assert_eq!(case.builder.nodes.len(), before);
        assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
    });
}
