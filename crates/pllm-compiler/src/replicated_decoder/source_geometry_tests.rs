//! Ordinary source-owner regressions; no numeric backend or publication proof.
use super::*;

const CONFIG: &[u8] = br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}"#;

fn experiment(storage: &str) -> Vec<u8> {
    experiment_with_workload(storage, 4, 2)
}

fn experiment_with_workload(storage: &str, inputs: usize, outputs: usize) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "schema": "pllm.experiment.v1", "name": "geometry-owner",
        "pipeline": {"model": {"source": "original-tiny-source"}, "components": {
            "linear": {"component": crate::replicated_decoder::sdk::PROTOCOL,
                "params": {"weights": "public", "range_policy": "checked",
                    "max_abs_activation": 1, "storage": storage}}}},
        "deployment": {"kind": "local", "root": "."},
        "budget": {"requests": 1, "max_input_tokens": inputs, "max_new_tokens": outputs}
    })).unwrap()
}

fn prepare() -> GeometryOwner {
    GeometryOwner::prepare(&experiment("compact"), CONFIG, 4, 2).unwrap()
}

fn assert_policy_fee(geometry: &GeometryOwner, raw_config: &[u8]) -> u128 {
    assert_eq!(geometry.original().raw_configuration(), raw_config);
    // Same no-manifest fee as source_policy::policy_work, not a new work cap.
    let prior = geometry.experiment.payload().resources().work.total
        .checked_mul(2).unwrap()
        .checked_add(u128::try_from(raw_config.len()).unwrap()).unwrap();
    let summary = u128::from(geometry.original().workload().max_new_tokens)
        .checked_mul(u128::from(pllm_models::SOURCE_SCALAR_SUMMARY_WORK)).unwrap();
    let policy_fee = prior.checked_add(summary).unwrap();
    assert_eq!(u128::from(geometry.limits.max_work).checked_add(policy_fee).unwrap(),
        u128::from(source_policy::SOURCE_WORK));
    policy_fee
}

#[test]
fn qwen_05b_checkpoint_requirement_quote_fit() {
    use pllm_models::{checkpoint_requirements as requirements, owned_backing};
    use pllm_types::bounded_canonical::borrowed::{NativeFrame, Traversal};

    // Public Qwen2.5-0.5B metadata, through the ordinary SourceConfigData factory.
    // No checkpoint, family-specific builder, graph census or numeric backend.
    const QWEN: &[u8] = br#"{"model_type":"qwen2","hidden_size":896,"intermediate_size":4864,"num_hidden_layers":24,"num_attention_heads":14,"num_key_value_heads":2,"vocab_size":151936,"max_position_embeddings":32768,"rope_theta":1000000.0,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":true,"attention_bias":true}"#;
    let headers = HEADER_FACTORIES.with(|n| n.get());
    let directories = snapshot_directory::OwnedSnapshotDirectory::factories_for_tests();
    let geometry = GeometryOwner::prepare(
        &experiment_with_workload("compact", 32, 2), QWEN, 32, 2,
    ).unwrap();
    let original = geometry.original();
    assert_eq!(original.raw_configuration(), QWEN);
    assert_eq!(original.workload().max_input_tokens, 32);
    assert_eq!(original.workload().max_new_tokens, 2);
    assert_eq!(original.workload().batch, 1);
    assert!(std::ptr::eq(original.budget(), geometry.root.budget()));
    assert!(std::ptr::eq(geometry.experiment.budget(), original.budget()));
    assert!(geometry.controller.is_for(original.budget()));
    assert_eq!(HEADER_FACTORIES.with(|n| n.get()), headers);
    assert_eq!(snapshot_directory::OwnedSnapshotDirectory::factories_for_tests(), directories);
    let policy_fee = assert_policy_fee(&geometry, QWEN);
    assert_eq!(geometry.limits.max_working_payload, MAX_ALLOCATION_BYTES);

    let header = original.header();
    let resources = original.resources();
    assert_eq!(header.graphs(), 2);
    assert!(header.operations() > 0);
    assert!(header.input_refs() > 0);
    assert!(header.dimensions() >= header.operations());
    assert!(header.json_values() >= header.operations());
    // Fresh root ONLY: no sibling/clone/lowering/metadata debit has occurred.
    // Historical resources are not a remaining-ledger query after any such use.
    let remaining = geometry.limits.max_work.checked_sub(resources.prepaid_work).unwrap();
    assert_eq!(policy_fee.checked_add(u128::from(resources.prepaid_work)).unwrap()
        .checked_add(u128::from(remaining)).unwrap(), u128::from(source_policy::SOURCE_WORK));
    let retained = original.budget().used_bytes();
    assert!(retained <= MAX_PROGRAM_BYTES);
    assert!(original.budget().admitted_peak_bytes() >= retained);
    assert!(resources.model_retained_bytes <= retained);

    // Header totals dominate every individual rank/string/key and include both
    // phases. Native walker bytes count raw text/keys plus formatted numbers.
    // Pinned ordinary serde_json i64/u64/f64 encodings fit 32 bytes per number;
    // charging every Value that amount also conservatively covers other leaves.
    let text = header.text_bytes().checked_add(header.key_bytes()).unwrap();
    let attribute_bytes = text.checked_add(header.json_values().checked_mul(32).unwrap()).unwrap();
    // Bounded-tree DepthConvention: root depth 0, maximum 16 inclusive;
    // requirements' native walker uses root depth 1, hence 17 frame slots.
    let depth = owned_backing::MAX_JSON_DEPTH.checked_add(1).unwrap();
    let limits = requirements::Limits {
        max_operations: header.operations(),
        max_inputs: header.input_refs(),
        max_dimensions: header.dimensions(),
        max_rank: header.dimensions(),
        max_string_bytes: text,
        max_attribute_nodes: header.json_values(),
        max_attribute_bytes: attribute_bytes,
        max_attribute_depth: depth,
        max_work: usize::try_from(remaining).unwrap(),
        max_scratch_bytes: MAX_PROGRAM_BYTES,
    };
    let quote = requirements::quote(limits).unwrap();
    assert_eq!(quote.operation_slots, header.operations());
    assert_eq!(quote.requirement_slots, header.operations()
        .checked_mul(requirements::MAX_REQUIREMENTS_PER_OPERATION).unwrap());
    assert_eq!(quote.attribute_frame_slots, depth);
    let operation_bytes = owned_backing::checked_array_bytes::<requirements::OperationSlot<'_>>(
        quote.operation_slots,
    ).unwrap();
    let requirement_bytes = owned_backing::checked_array_bytes::<requirements::Requirement<'_>>(
        quote.requirement_slots,
    ).unwrap();
    let frame_bytes = owned_backing::checked_array_bytes::<NativeFrame<'_>>(depth).unwrap();
    assert_eq!(quote.scratch_bytes,
        operation_bytes.checked_add(requirement_bytes).unwrap().checked_add(frame_bytes).unwrap());
    let walker = Traversal::resources_for_native_slots(depth).unwrap();
    assert_eq!(walker.frame_bytes, frame_bytes);
    assert_eq!(quote.attribute_workspace_bytes, walker.logical_workspace_bytes);
    // Scratch already includes frame backing; add walker controls only once.
    let walker_controls = quote.attribute_workspace_bytes.checked_sub(frame_bytes).unwrap();

    // Prospective logical subtotal: all initialized pools coexist with original
    // root and descriptors for EVERY possible (not merely unique) requirement.
    // Tuple backing includes String/Shape inline slots; name backing is separate.
    // No allocator, String clone or actual pool construction occurs here.
    let copy_descriptors = owned_backing::checked_array_bytes::<(String, requirements::Shape)>(
        quote.requirement_slots,
    ).unwrap();
    assert_eq!(copy_descriptors,
        quote.requirement_slots.checked_mul(size_of::<(String, requirements::Shape)>()).unwrap());
    // Deduplicated artifact names borrow attribute Value strings. Their aggregate
    // copied backing is bounded by all header text, without copying JSON keys.
    let copy_names = header.text_bytes();
    let controls = [
        OriginalSourcePlan::compiler_metadata_control_bytes().unwrap(),
        size_of::<requirements::Limits>(),
        size_of::<requirements::Quote>(),
        size_of::<Result<requirements::Quote, requirements::Error>>(),
        size_of::<requirements::Scratch<'_, '_>>(),
        size_of::<Result<requirements::RequirementsView<'_, '_>, requirements::Error>>(),
        size_of::<Vec<requirements::OperationSlot<'_>>>(),
        size_of::<Vec<requirements::Requirement<'_>>>(),
        size_of::<Vec<NativeFrame<'_>>>(),
        size_of::<Vec<(String, requirements::Shape)>>(),
    ].into_iter().try_fold(0usize, usize::checked_add).unwrap();
    let phase_subtotal = [retained, quote.scratch_bytes, walker_controls,
        controls, copy_descriptors, copy_names]
        .into_iter().try_fold(0usize, usize::checked_add).unwrap();
    let work_fits = quote.work_ceiling <= limits.max_work;
    let scratch_fits = quote.scratch_bytes <= limits.max_scratch_bytes;
    let subtotal_fits = phase_subtotal <= MAX_PROGRAM_BYTES;
    // No API quotes private checker/copy controls or dependency stack/buffers.
    // A subtotal fitting is NOT proof of full 64 MiB phase admission.
    let full_phase_bytes: Option<usize> = None;
    let fits = if !work_fits || !scratch_fits || !subtotal_fits {
        Some(false)
    } else {
        full_phase_bytes.map(|bytes| bytes <= MAX_PROGRAM_BYTES)
    };
    let classification = if !work_fits { "Work" }
        else if !scratch_fits { "Scratch" }
        else if !subtotal_fits { "ProspectiveSubtotal" }
        else { "UnknownFullPhase" };
    assert_eq!(fits, if work_fits && scratch_fits && subtotal_fits { None } else { Some(false) });
    assert_eq!(classification, match (work_fits, scratch_fits, subtotal_fits) {
        (false, _, _) => "Work",
        (true, false, _) => "Scratch",
        (true, true, false) => "ProspectiveSubtotal",
        (true, true, true) => "UnknownFullPhase",
    });
    // Verify reported work comparison on the authentic counter, using only the
    // actual quote (never an arbitrary overflow debit). No check/model walk.
    assert!(geometry.controller.payload_bytes() >= OriginalSourcePlan::compiler_metadata_control_bytes().unwrap());
    let admission = original.prepay_compiler_metadata(
        &geometry.controller, u64::try_from(quote.work_ceiling).unwrap(),
    );
    if work_fits {
        assert!(admission.is_ok());
    } else {
        assert_eq!(classification, "Work");
        assert!(matches!(admission, Err(original_source::Error::Work)));
    }
    assert_eq!(original.budget().used_bytes(), retained);
    eprintln!("Qwen2.5-0.5B public geometry: counts={header:?}; limits={limits:?}; \
        checker_work_ceiling={}; scratch_bytes={}; attribute_workspace_bytes={}; \
        source_work_cap={}; policy_fee={policy_fee}; source_policy_work_cap={}; \
        source_prepaid_work={}; root_retained_bytes={retained}; \
        remaining_allowance={remaining}; phase_cap={MAX_PROGRAM_BYTES}; \
        initialized_pool_bytes={}; walker_controls={walker_controls}; typed_controls={controls}; \
        owned_copy_descriptors={copy_descriptors}; bounded_name_backing={copy_names}; \
        prospective_phase_subtotal={phase_subtotal}; full_phase_bytes={full_phase_bytes:?}; \
        work_fits={work_fits}; scratch_fits={scratch_fits}; subtotal_fits={subtotal_fits}; \
        fits={fits:?}; classification={classification}; \
        full_checker/copy_controls_and_dependency_buffers=unknown; Qwen_qualification=NOT_ESTABLISHED",
        quote.work_ceiling, quote.scratch_bytes, quote.attribute_workspace_bytes,
        geometry.limits.max_work, source_policy::SOURCE_WORK,
        resources.prepaid_work, quote.scratch_bytes);
}

#[test]
fn geometry_moves_same_owned_plan_counter_and_configuration() {
    // Exercise the parent-module query alias too, without exposing a public API.
    let geometry: crate::replicated_decoder::source_geometry::GeometryOwner = prepare();
    let plan = geometry.plan() as *const DecoderPlan;
    let configuration = geometry.original().raw_configuration().as_ptr();
    let counter = geometry.original().budget() as *const _;
    assert!(std::ptr::eq(geometry.original().budget(), geometry.root.budget()));
    assert!(std::ptr::eq(geometry.experiment.budget(), geometry.root.budget()));
    assert!(geometry.controller.is_for(geometry.root.budget()));
    assert_eq!(geometry.original().raw_configuration(), CONFIG);
    let (original, experiment, _limits, controller, root) = geometry.into_ingress();
    assert_eq!(original.plan() as *const DecoderPlan, plan);
    assert_eq!(original.raw_configuration().as_ptr(), configuration);
    assert_eq!(root.budget() as *const _, counter);
    assert!(controller.is_for(original.budget()));
    // Explicitly preserve the same retirement order as the owner/tuple fields.
    drop(original);
    drop(experiment);
    drop(controller);
    drop(root);
}

#[test]
fn geometry_never_enters_header_or_directory_factory() {
    let headers = HEADER_FACTORIES.with(|n| n.get());
    let directories = snapshot_directory::OwnedSnapshotDirectory::factories_for_tests();
    let geometry = prepare();
    assert_eq!(HEADER_FACTORIES.with(|n| n.get()), headers);
    assert_eq!(snapshot_directory::OwnedSnapshotDirectory::factories_for_tests(), directories);
    drop(geometry);
    assert_eq!(snapshot_directory::OwnedSnapshotDirectory::factories_for_tests(), directories);
}

#[test]
fn header_identity_and_byte_guards_stay_before_factory_but_json_stays_after() {
    let before = HEADER_FACTORIES.with(|n| n.get());
    for (session, source, composition) in [
        ([0; 32], [2; 32], [3; 32]),
        ([1; 32], [0; 32], [3; 32]),
        ([1; 32], [2; 32], [0; 32]),
    ] {
        assert!(matches!(SourceHeaderEntry::begin(
            b"x", b"x", 4, 2, session, source, composition,
        ), Err(Error::Binding)));
    }
    let oversized = vec![b'x'; (32 << 10) + 1];
    let oversized_config = vec![b'x'; source_policy::CONFIG_BYTES + 1];
    for (experiment, config) in [
        (&b""[..], CONFIG), (oversized.as_slice(), CONFIG),
        (&b"x"[..], &b""[..]), (&b"x"[..], oversized_config.as_slice()),
    ] {
        assert!(matches!(GeometryOwner::prepare(experiment, config, 4, 2), Err(Error::Binding)));
        assert!(matches!(SourceHeaderEntry::begin(
            experiment, config, 4, 2, [1; 32], [2; 32], [3; 32],
        ), Err(Error::Binding)));
    }
    assert_eq!(HEADER_FACTORIES.with(|n| n.get()), before);
    assert!(SourceHeaderEntry::begin(b"x", CONFIG, 4, 2, [1; 32], [2; 32], [3; 32]).is_err());
    assert_eq!(HEADER_FACTORIES.with(|n| n.get()), before + 1);
}

#[test]
fn geometry_keeps_compact_only_and_existing_workload_policy() {
    assert!(matches!(GeometryOwner::prepare(&experiment("expanded"), CONFIG, 4, 2), Err(Error::Binding)));
    // Omitted selector is the real legacy Expanded choice, not an invalid tag.
    let mut legacy: serde_json::Value = serde_json::from_slice(&experiment("compact")).unwrap();
    legacy["pipeline"]["components"]["linear"]["params"].as_object_mut().unwrap().remove("storage");
    let legacy = serde_json::to_vec(&legacy).unwrap();
    assert!(matches!(GeometryOwner::prepare(&legacy, CONFIG, 4, 2), Err(Error::Binding)));
    assert!(matches!(GeometryOwner::prepare(&experiment("compact"), CONFIG, 4, 1), Err(Error::Binding)));
    assert!(GeometryOwner::prepare(&experiment("compact"), CONFIG, 0, 2).is_err());
    assert!(GeometryOwner::prepare(&experiment("compact"), CONFIG, 32, 2).is_err());
}

#[test]
fn aggregate_replaces_only_bootstrap_and_checks_all_arithmetic() {
    assert_eq!(aggregate_phase(100, 40, 70).unwrap(), 130);
    assert_eq!(aggregate_phase(MAX_PROGRAM_BYTES, 50, 50).unwrap(), MAX_PROGRAM_BYTES);
    assert!(matches!(aggregate_phase(MAX_PROGRAM_BYTES, 50, 51), Err(Error::Bounds)));
    assert!(matches!(aggregate_phase(10, 11, 0), Err(Error::Bounds)));
    assert!(matches!(aggregate_phase(usize::MAX, 0, 1), Err(Error::Bounds)));
}

#[test]
fn geometry_reserves_floor_plus_typed_controls_and_admits_peak() {
    let geometry = prepare();
    assert_eq!(geometry.controller.payload_bytes(),
        geometry_controller_floor().unwrap() + additional_controller_bytes().unwrap());
    assert!(geometry.root.budget().used_bytes() <= MAX_PROGRAM_BYTES);
    assert!(geometry.root.budget().admitted_peak_bytes() >= geometry.root.budget().used_bytes());
    assert!(additional_controller_bytes().unwrap() >= 3 * size_of::<GeometryOwner>());
}

#[test]
fn bootstrap_exact_and_one_short_keep_original_quote_contract() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let config = import_json(&root, CONFIG).unwrap();
    let server = import_json(&root, &experiment("compact")).unwrap();
    let workload = DecoderWorkload { batch: 1, max_input_tokens: 4, max_new_tokens: 2 };
    let (limits, _) = source_policy::admitted_choice(&server, &config, None, workload).unwrap();
    let bootstrap = original_source::bootstrap_bytes(&config, limits).unwrap();
    assert!(bootstrap > 0);
    assert!(original_source::quote(
        config, root.charge_external(bootstrap - 1).unwrap(), workload, limits,
    ).is_err());
    let config = import_json(&root, CONFIG).unwrap();
    let quote = original_source::quote(
        config, root.charge_external(bootstrap).unwrap(), workload, limits,
    ).unwrap();
    let phase = aggregate_phase(root.budget().used_bytes(), bootstrap,
        quote.resources().working_charge_bytes).unwrap();
    root.budget().admit_peak(phase).unwrap();
    let geometry = prepare();
    assert_eq!(quote.construct().unwrap().plan().digest(), geometry.plan().digest());
}

#[test]
fn foreign_counter_refusal_does_not_burn_but_work_failure_is_sticky() {
    let geometry = prepare();
    let foreign = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let foreign_controller = foreign.charge_external(geometry.controller.payload_bytes()).unwrap();
    assert!(matches!(geometry.original().prepay_compiler_metadata(&foreign_controller, 1),
        Err(original_source::Error::Counter)));
    geometry.original().prepay_compiler_metadata(&geometry.controller, 1).unwrap();
    assert!(matches!(geometry.original().prepay_compiler_metadata(&geometry.controller, u64::MAX),
        Err(original_source::Error::Work)));
    assert!(matches!(geometry.original().prepay_compiler_metadata(&geometry.controller, 0),
        Err(original_source::Error::Burned)));
}
