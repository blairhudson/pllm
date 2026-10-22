//! Opt-in source construction only. Fixtures already own their Manifest/semantic
//! metadata; no test claims to preadmit parsing or the allocating legacy validator.
use super::*;
use crate::replicated_decoder::source_registry::REGISTRY_HASH_FACTORIES;
use crate::replicated_decoder::source_registry::{
    array_bytes, decoder_plan_metadata_bytes, decoder_plan_metadata_outside_graph_bytes,
    descriptor_validation_quote, semantic_metadata_bytes, tree_entry_bytes,
    SourceConstructionOwners, REGISTRY_RECORD_VALIDATIONS,
};

const CONFIG: &[u8] = br#"{"model_type":"llama","hidden_size":2,"intermediate_size":2,
    "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
    "vocab_size":3,"max_position_embeddings":64,"rope_theta":10000,
    "rms_norm_eps":0.000001,"tie_word_embeddings":false,"hidden_act":"silu",
    "attention_bias":false,"head_dim":2,"attention_dropout":0.0,"use_cache":true}"#;

fn manifest(with_norms: bool) -> Manifest {
    let execution = ExecutionPolicy {
        range_policy: if with_norms {
            RangePolicy::Checked
        } else {
            RangePolicy::Static
        },
        ..ExecutionPolicy::default()
    };
    let shapes =
        Program::source_shapes_with_policy(CONFIG, 1, 1, WeightMode::Public, execution).unwrap();
    let mut tensors = BTreeMap::new();
    for (key, shape) in shapes {
        let lanes = shape.iter().product::<usize>();
        let matrix = shape.len() == 2;
        tensors.insert(
            key.clone(),
            SourceTensor {
                shape,
                lower: 0,
                upper: 0,
                digest: Sha256::digest(vec![0u8; lanes * 8]).into(),
                public_q18: key.contains("norm").then(|| vec![0; lanes]),
                row_l1: matrix.then_some(0),
                row_squared_norm: matrix.then_some(0),
                row_l1_squared_sum: matrix.then_some(0),
            },
        );
    }
    let mut norms = SourceNorms::new();
    if with_norms {
        let plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
        for (key, (rows, columns, width)) in source_norms::requests(&plan.prefill).unwrap() {
            let mut scan = source_norms::BlockScan::new(rows, columns, width).unwrap();
            scan.append(0, &vec![0; rows * columns]).unwrap();
            norms.insert(key, scan.finish().unwrap());
        }
    }
    Manifest {
        schema: "pllm.replicated-decoder.manifest.v1".into(),
        weights: WeightMode::Public,
        execution,
        config: CONFIG.to_vec(),
        rows: 1,
        outputs: 1,
        source: [1; 32],
        composition: [2; 32],
        session: [3; 32],
        tensors,
        norms,
        program: [0; 32], // source authority is NOT manifest program authentication
    }
}

fn registry(manifest: &Manifest) -> SourceRegistry<'_> {
    SourceRegistry::from_validated(Program::validated_source_descriptors(manifest).unwrap())
        .unwrap()
}

fn finish_import(mut importer: PublicSourceImport, manifest: &Manifest) -> PublicSource {
    for (key, tensor) in &manifest.tensors {
        let values = vec![0u8; tensor.shape.iter().product::<usize>() * 4];
        for (index, frame) in values.chunks(12).enumerate() {
            importer.append(key, index * 3, frame).unwrap();
        }
    }
    importer.finish().unwrap()
}

#[test]
fn checked_registry_exact_fit_one_short_and_overflow_precede_hash_factory() {
    for with_norms in [false, true] {
        let manifest = manifest(with_norms);
        let baseline = 137;
        let quote =
            SourceRegistry::construction_quote(&manifest.tensors, &manifest.norms, baseline)
                .unwrap();
        REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
        let validated = Program::validated_source_descriptors(&manifest).unwrap();
        assert!(matches!(
            SourceRegistry::from_validated_checked(validated, baseline, quote.peak_bytes - 1,),
            Err(Error::Bounds)
        ));
        assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
        let validated = Program::validated_source_descriptors(&manifest).unwrap();
        let checked =
            SourceRegistry::from_validated_checked(validated, baseline, quote.peak_bytes).unwrap();
        assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 1);
        assert!(std::ptr::eq(checked.source_records(), &manifest.tensors));
        assert!(std::ptr::eq(checked.source_norms(), &manifest.norms));
        assert_eq!(checked.owned_backing_bytes(), 0);
        assert!(matches!(
            SourceRegistry::construction_quote(&manifest.tensors, &manifest.norms, usize::MAX,),
            Err(Error::Bounds)
        ));
        REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
        let validated = Program::validated_source_descriptors(&manifest).unwrap();
        assert!(matches!(
            SourceRegistry::from_validated_checked(validated, usize::MAX, usize::MAX,),
            Err(Error::Bounds)
        ));
        assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
    }
}

#[test]
fn checked_import_exact_fit_one_short_overflow_and_fixed_ceiling_before_copy_factory() {
    for with_norms in [false, true] {
        let manifest = manifest(with_norms);
        let registry = registry(&manifest);
        let baseline = 293;
        let quote = PublicSourceImport::construction_quote(&registry, baseline).unwrap();
        CHECKED_IMPORT_FACTORIES.with(|n| n.set(0));
        assert!(matches!(
            PublicSourceImport::new_checked(&registry, baseline, quote.peak_bytes - 1,),
            Err(Error::Bounds)
        ));
        assert_eq!(CHECKED_IMPORT_FACTORIES.with(|n| n.get()), 0);
        assert!(matches!(
            PublicSourceImport::new_checked(&registry, usize::MAX, usize::MAX,),
            Err(Error::Bounds)
        ));
        assert!(matches!(
            PublicSourceImport::new_checked(&registry, MAX_PROGRAM_BYTES, usize::MAX,),
            Err(Error::Bounds)
        ));
        assert_eq!(CHECKED_IMPORT_FACTORIES.with(|n| n.get()), 0);
        let importer =
            PublicSourceImport::new_checked(&registry, baseline, quote.peak_bytes).unwrap();
        assert_eq!(CHECKED_IMPORT_FACTORIES.with(|n| n.get()), 1);
        let source = finish_import(importer, &manifest);
        source.check_registry(&registry).unwrap();
        assert!(source.resident_metadata_bytes().unwrap() < quote.new_owned_peak_bytes);
    }
}

#[test]
fn quote_counts_actual_borrowed_capacities_and_never_serializes_binding() {
    let mut manifest = manifest(false);
    let before = SourceRegistry::construction_quote(&manifest.tensors, &manifest.norms, 0).unwrap();
    let import_before = PublicSourceImport::construction_quote(&registry(&manifest), 0).unwrap();
    let tensor = manifest
        .tensors
        .values_mut()
        .find(|t| t.public_q18.is_some())
        .unwrap();
    let old_shape = tensor.shape.capacity();
    let values = tensor.public_q18.as_mut().unwrap();
    let old_values = values.capacity();
    tensor.shape.reserve_exact(17);
    values.reserve_exact(31);
    let extra = (tensor.shape.capacity() - old_shape) * size_of::<usize>()
        + (values.capacity() - old_values) * size_of::<i64>();
    let after = SourceRegistry::construction_quote(&manifest.tensors, &manifest.norms, 0).unwrap();
    assert_eq!(
        after.borrowed_metadata_bytes - before.borrowed_metadata_bytes,
        extra
    );
    assert_eq!(after.new_owned_peak_bytes, before.new_owned_peak_bytes);
    let registry = registry(&manifest);
    let import_after = PublicSourceImport::construction_quote(&registry, 0).unwrap();
    assert_eq!(
        import_after.borrowed_metadata_bytes - import_before.borrowed_metadata_bytes,
        extra
    );
    assert_eq!(
        import_after.new_owned_peak_bytes - import_before.new_owned_peak_bytes,
        extra
    );
    let mut legacy = PublicSourceImport::new(&manifest).unwrap();
    let mut checked =
        PublicSourceImport::new_checked(&registry, 0, import_after.peak_bytes).unwrap();
    assert_eq!(
        legacy.0.as_ref().unwrap().binding,
        checked.0.as_ref().unwrap().binding
    );
    let a = legacy.0.take().unwrap();
    let b = checked.0.take().unwrap();
    assert_eq!(a.tensors.len(), b.tensors.len());
    for ((ka, a), (kb, b)) in a.tensors.iter().zip(&b.tensors) {
        assert_eq!(ka, kb);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.public_q18, b.public_q18);
        assert_eq!(a.digest, b.digest);
    }
}

#[test]
fn checked_source_mismatch_still_rejects_and_burns_importer() {
    let manifest = manifest(true);
    let registry = registry(&manifest);
    let quote = PublicSourceImport::construction_quote(&registry, 0).unwrap();
    let mut importer = PublicSourceImport::new_checked(&registry, 0, quote.peak_bytes).unwrap();
    let key = manifest.tensors.keys().next().unwrap();
    assert!(importer.append(key, 0, &1.0f32.to_le_bytes()).is_err());
    assert!(importer.append(key, 0, &0.0f32.to_le_bytes()).is_err());
    assert!(importer.finish().is_err());

    let mut source = finish_import(
        PublicSourceImport::new_checked(&registry, 0, quote.peak_bytes).unwrap(),
        &manifest,
    );
    source.binding[0] ^= 1;
    assert!(matches!(
        source.check_registry(&registry),
        Err(Error::Binding)
    ));
}

#[test]
fn invalid_descriptor_validation_cannot_enter_checked_factories() {
    let mut manifest = manifest(false);
    manifest
        .tensors
        .values_mut()
        .find(|t| t.public_q18.is_some())
        .unwrap()
        .digest = [7; 32];
    REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
    CHECKED_IMPORT_FACTORIES.with(|n| n.set(0));
    assert!(Program::validated_source_descriptors(&manifest).is_err());
    assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
    assert_eq!(CHECKED_IMPORT_FACTORIES.with(|n| n.get()), 0);
}

#[test]
fn file_census_prices_handles_not_authenticated_disk_extent() {
    fn source(bytes: usize) -> PublicSource {
        let data = vec![0u8; bytes];
        let mut writer = PublicFileWriter::new(bytes as u64, Sha256::digest(&data).into()).unwrap();
        writer.write(&data).unwrap();
        PublicSource {
            files: BTreeMap::from([("weight".into(), writer.finish().unwrap())]),
            binding: [1; 32],
        }
    }
    let small = source(8);
    let large = source(65536);
    assert_ne!(small.files["weight"].size(), large.files["weight"].size());
    assert_eq!(
        small.resident_metadata_bytes().unwrap(),
        large.resident_metadata_bytes().unwrap()
    );
}

#[test]
fn descriptor_collection_quote_keeps_growth_old_allocation_and_sort_scratch() {
    let manifest = manifest(false);
    let shapes = Program::source_shapes_with_policy(
        &manifest.config,
        manifest.rows,
        manifest.outputs,
        manifest.weights,
        manifest.execution,
    )
    .unwrap();
    let mut plan = program::lower_with_weights(
        &manifest.config,
        manifest.rows,
        manifest.outputs,
        manifest.weights,
    )
    .unwrap();
    let owners = SourceConstructionOwners {
        manifest: &manifest,
        incoming: None,
        public: None,
        retained_baseline_bytes: 0,
    };
    let before =
        descriptor_validation_quote(&plan.prefill, &shapes, &owners, MAX_PROGRAM_BYTES).unwrap();
    let rms = plan
        .prefill
        .operations
        .iter()
        .find(|op| op.operator == pllm_models::ModelOperator::RmsNorm)
        .unwrap()
        .clone();
    let count = plan
        .prefill
        .operations
        .iter()
        .filter(|op| op.operator == pllm_models::ModelOperator::RmsNorm)
        .count();
    assert!(count >= 2); // beyond std Vec's minimum nonempty staging capacity
                         // Syntax-only quote probe. Duplicating an operation is NOT a new validated
                         // semantic plan; it increases the conservative coefficient insertion census.
    plan.prefill.operations.push(rms);
    let after =
        descriptor_validation_quote(&plan.prefill, &shapes, &owners, MAX_PROGRAM_BYTES).unwrap();
    assert_eq!(
        after.new_owned_peak_bytes - before.new_owned_peak_bytes,
        tree_entry_bytes::<&str>().unwrap() + 4 * size_of::<&str>(),
    );
    REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
    REGISTRY_RECORD_VALIDATIONS.with(|n| n.set(0));
    assert!(descriptor_validation_quote(&plan.prefill, &shapes, &owners, after.peak_bytes).is_ok());
    assert!(matches!(
        descriptor_validation_quote(&plan.prefill, &shapes, &owners, after.peak_bytes - 1),
        Err(Error::Bounds),
    ));
    let impossible = SourceConstructionOwners {
        retained_baseline_bytes: usize::MAX,
        ..owners
    };
    assert!(matches!(
        descriptor_validation_quote(&plan.prefill, &shapes, &impossible, usize::MAX),
        Err(Error::Bounds),
    ));
    assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
    assert_eq!(REGISTRY_RECORD_VALIDATIONS.with(|n| n.get()), 0);
}

#[test]
fn deep_owned_attributes_reject_before_record_hash_or_source_copy_factory() {
    let manifest = manifest(true);
    let source = Arc::new(finish_import(
        PublicSourceImport::new(&manifest).unwrap(),
        &manifest,
    ));
    let shapes = Program::source_shapes_with_policy(
        &manifest.config,
        manifest.rows,
        manifest.outputs,
        manifest.weights,
        manifest.execution,
    )
    .unwrap();
    let mut plan = program::lower_with_weights(
        &manifest.config,
        manifest.rows,
        manifest.outputs,
        manifest.weights,
    )
    .unwrap();
    // Already-owned input, not a claim about preadmitting the allocations below.
    let mut attributes = serde_json::Value::Null;
    for _ in 0..18 {
        attributes = serde_json::Value::Array(vec![attributes]);
    }
    plan.prefill.operations[0].attributes = attributes;
    let owners = SourceConstructionOwners {
        manifest: &manifest,
        incoming: None,
        public: Some(&source),
        retained_baseline_bytes: 0,
    };
    REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
    REGISTRY_RECORD_VALIDATIONS.with(|n| n.set(0));
    CHECKED_IMPORT_FACTORIES.with(|n| n.set(0));
    assert!(matches!(
        descriptor_validation_quote(&plan.prefill, &shapes, &owners, MAX_PROGRAM_BYTES),
        Err(Error::Bounds),
    ));
    assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
    assert_eq!(REGISTRY_RECORD_VALIDATIONS.with(|n| n.get()), 0);
    assert_eq!(CHECKED_IMPORT_FACTORIES.with(|n| n.get()), 0);
    assert_eq!(Arc::strong_count(&source), 1);
}

#[test]
fn original_source_registry_owner_quote_leaves_manifest_identity_guard_intact() {
    let mut manifest = manifest(false);
    let mut program = Program::compile_with_norms(
        &manifest.config,
        manifest.rows,
        manifest.outputs,
        &manifest.tensors,
        manifest.weights,
        manifest.execution,
        &manifest.norms,
    )
    .unwrap();
    program.bind(manifest.source, manifest.composition);
    manifest.program = program.digest();
    drop(program); // no unpriced expanded Program owner in the following phase
    assert!(manifest.compile().is_ok());
    let public = Arc::new(finish_import(
        PublicSourceImport::new(&manifest).unwrap(),
        &manifest,
    ));
    // End registry borrows only; retain the incoming owner through the final check.
    let _incoming = {
        let original_registry = registry(&manifest);
        let mut incoming = Vec::with_capacity(129);
        incoming.push(0);
        let owners = SourceConstructionOwners {
            manifest: &manifest,
            incoming: Some(&incoming),
            public: Some(&public),
            retained_baseline_bytes: std::mem::size_of_val(&original_registry),
        };
        let quote =
            SourceRegistry::construction_quote_with_owners(&owners, MAX_PROGRAM_BYTES).unwrap();
        let validated = Program::validated_source_descriptors(&manifest).unwrap();
        REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
        REGISTRY_RECORD_VALIDATIONS.with(|n| n.set(0));
        assert!(matches!(
            SourceRegistry::from_validated_with_owners_checked(
                validated,
                &owners,
                quote.peak_bytes - 1,
            ),
            Err(Error::Bounds)
        ));
        assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
        assert_eq!(REGISTRY_RECORD_VALIDATIONS.with(|n| n.get()), 0);
        let validated = Program::validated_source_descriptors(&manifest).unwrap();
        let registry = SourceRegistry::from_validated_with_owners_checked(
            validated,
            &owners,
            quote.peak_bytes,
        )
        .unwrap();
        public.check_registry(&registry).unwrap();
        assert!(std::ptr::eq(
            registry.source_records(),
            original_registry.source_records()
        ));
        assert_eq!(Arc::strong_count(&public), 1);
        incoming
    };
    // Quotes and the registry's descriptor/source binding do not authenticate
    // Manifest.program or source/composition/session. Ordinary compile still must.
    manifest.program[0] ^= 1;
    assert_eq!(
        manifest.compile().err().as_deref(),
        Some("replicated manifest program differs")
    );
}

#[test]
fn census_wrappers_match_shared_helpers_and_preserve_bounds_mapping() {
    use pllm_models::owned_backing;
    use serde_json::Value;

    for capacity in [0, 1, 17] {
        assert_eq!(
            array_bytes::<u64>(capacity).unwrap(),
            owned_backing::checked_array_bytes::<u64>(capacity).unwrap(),
        );
    }
    assert!(matches!(array_bytes::<u64>(usize::MAX), Err(Error::Bounds)));
    assert_eq!(
        tree_entry_bytes::<(String, Value)>().unwrap(),
        owned_backing::checked_tree_node_bytes::<(String, Value)>().unwrap(),
    );
    assert_eq!(
        tree_entry_bytes::<&str>().unwrap(),
        owned_backing::checked_tree_node_bytes::<&str>().unwrap(),
    );
}

#[test]
fn semantic_wrapper_matches_shared_backing_plus_one_inline_graph_and_quote_caps() {
    use pllm_models::{owned_backing, DecoderGraph};
    use serde_json::{Map, Value};

    let manifest = manifest(false);
    let shapes = Program::source_shapes_with_policy(
        &manifest.config,
        manifest.rows,
        manifest.outputs,
        manifest.weights,
        manifest.execution,
    )
    .unwrap();
    let mut plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
    for graph in [&mut plan.prefill, &mut plan.decode] {
        // Already-owned census probes, not source/semantic validation authority.
        graph.operations.reserve_exact(7);
        graph.output.reserve_exact(19);
        graph.operations[0].output_shape.reserve_exact(11);
        let mut values = Vec::with_capacity(5);
        values.push(Value::Object(Map::new()));
        graph.operations[0].attributes["census_probe"] = Value::Array(values);
        let backing = graph.checked_owned_backing_bytes(usize::MAX).unwrap();
        assert_eq!(graph.checked_owned_backing_bytes(backing).unwrap(), backing);
        assert!(matches!(
            graph.checked_owned_backing_bytes(backing - 1),
            Err(owned_backing::Error::ByteLimit),
        ));
        assert_eq!(
            semantic_metadata_bytes(graph).unwrap(),
            size_of::<DecoderGraph>() + backing,
        );
    }
    let owners = SourceConstructionOwners {
        manifest: &manifest,
        incoming: None,
        public: None,
        retained_baseline_bytes: 137,
    };
    let quote =
        descriptor_validation_quote(&plan.prefill, &shapes, &owners, MAX_PROGRAM_BYTES).unwrap();
    REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
    REGISTRY_RECORD_VALIDATIONS.with(|n| n.set(0));
    assert_eq!(
        descriptor_validation_quote(&plan.prefill, &shapes, &owners, quote.peak_bytes).unwrap(),
        quote,
    );
    assert!(matches!(
        descriptor_validation_quote(&plan.prefill, &shapes, &owners, quote.peak_bytes - 1),
        Err(Error::Bounds),
    ));
    assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
    assert_eq!(REGISTRY_RECORD_VALIDATIONS.with(|n| n.get()), 0);
}

#[test]
fn semantic_census_charges_fresh_and_move_only_emptied_objects_and_array_capacity() {
    use serde_json::{Map, Value};

    let mut plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
    let graph = &mut plan.prefill;
    let operation = graph
        .operations
        .iter()
        .position(|op| op.operator == pllm_models::ModelOperator::Multiply)
        .unwrap();
    // Probe already-owned attributes only, not semantic/source validation or
    // admission of the allocations used to construct this fixture.
    graph.operations[operation].attributes["census_probe"] = Value::Null;
    let baseline = semantic_metadata_bytes(graph).unwrap();
    let object_bytes = tree_entry_bytes::<(String, Value)>().unwrap();
    let emptied_object = || {
        let mut object = Map::new();
        object.insert("temporary".into(), Value::Null);
        assert_eq!(object.remove("temporary"), Some(Value::Null));
        assert!(object.is_empty());
        // Move, never clone: cloning may discard the retained empty root.
        Value::Object(object)
    };

    graph.operations[operation].attributes["census_probe"] = Value::Object(Map::new());
    let fresh = semantic_metadata_bytes(graph).unwrap();
    assert_eq!(fresh - baseline, object_bytes);
    graph.operations[operation].attributes["census_probe"] = emptied_object();
    assert_eq!(semantic_metadata_bytes(graph).unwrap(), fresh);

    for count in [1, 9, 33] {
        let mut objects = Vec::with_capacity(count + 7);
        for _ in 0..count {
            objects.push(emptied_object());
        }
        assert!(objects.capacity() > objects.len());
        let array_bytes = objects.capacity() * size_of::<Value>();
        graph.operations[operation].attributes["census_probe"] = Value::Array(objects);
        assert_eq!(
            semantic_metadata_bytes(graph).unwrap() - baseline,
            array_bytes + count * object_bytes,
        );
    }
}

#[test]
fn semantic_census_accepts_depth_sixteen_and_rejects_seventeen() {
    use serde_json::{Map, Value};

    let mut plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
    let graph = &mut plan.prefill;
    // Attribute object is depth zero; its probe value starts at depth one.
    graph.operations[0].attributes["census_probe"] = Value::Null;
    let baseline = semantic_metadata_bytes(graph).unwrap();
    let mut nested = Value::Object(Map::new());
    let mut array_bytes = 0;
    for _ in 1..16 {
        let values = vec![nested];
        array_bytes += values.capacity() * size_of::<Value>();
        nested = Value::Array(values);
    }
    graph.operations[0].attributes["census_probe"] = nested;
    assert_eq!(
        semantic_metadata_bytes(graph).unwrap() - baseline,
        array_bytes + tree_entry_bytes::<(String, Value)>().unwrap(),
    );

    let boundary = graph.operations[0].attributes["census_probe"].take();
    graph.operations[0].attributes["census_probe"] = Value::Array(vec![boundary]);
    assert!(matches!(semantic_metadata_bytes(graph), Err(Error::Bounds)));
}

#[test]
fn complete_plan_census_matches_shared_backing_plus_one_inline_owner_and_caps() {
    use pllm_models::DecoderPlan;

    let plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
    let expected = size_of::<DecoderPlan>() + plan.checked_owned_backing_bytes(usize::MAX).unwrap();
    REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
    REGISTRY_RECORD_VALIDATIONS.with(|n| n.set(0));
    assert_eq!(
        decoder_plan_metadata_bytes(&plan, expected).unwrap(),
        expected
    );
    assert_eq!(
        decoder_plan_metadata_bytes(&plan, usize::MAX).unwrap(),
        expected
    );
    for cap in [0, size_of::<DecoderPlan>() - 1, expected - 1] {
        assert!(matches!(
            decoder_plan_metadata_bytes(&plan, cap),
            Err(Error::Bounds)
        ));
    }
    for graph in [&plan.prefill, &plan.decode] {
        let graph_bytes = semantic_metadata_bytes(graph).unwrap();
        assert!(expected > graph_bytes);
        let remainder = decoder_plan_metadata_outside_graph_bytes(&plan, graph, expected).unwrap();
        assert_eq!(remainder + graph_bytes, expected);
        assert!(remainder < expected - 1);
        // A fitting remainder cannot hide an over-cap complete owner.
        assert!(matches!(
            decoder_plan_metadata_outside_graph_bytes(&plan, graph, expected - 1),
            Err(Error::Bounds),
        ));
    }
    assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
    assert_eq!(REGISTRY_RECORD_VALIDATIONS.with(|n| n.get()), 0);
}

#[test]
fn complete_plan_census_cannot_bypass_program_cap_with_usize_max() {
    use pllm_models::AppliedMethod;

    let mut plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
    // Reserve only: no initialized methods or copied source payload. Spare
    // backing alone exceeds the unchanged whole-program ceiling.
    let count = MAX_PROGRAM_BYTES / size_of::<AppliedMethod>() + 1;
    plan.transformations.try_reserve_exact(count).unwrap();
    assert!(plan.transformations.is_empty());
    assert!(plan.checked_owned_backing_bytes(usize::MAX).unwrap() > MAX_PROGRAM_BYTES);
    assert!(semantic_metadata_bytes(&plan.prefill).unwrap() < MAX_PROGRAM_BYTES);
    assert!(matches!(
        decoder_plan_metadata_bytes(&plan, usize::MAX),
        Err(Error::Bounds)
    ));
    for graph in [&plan.prefill, &plan.decode] {
        assert!(matches!(
            decoder_plan_metadata_outside_graph_bytes(&plan, graph, usize::MAX),
            Err(Error::Bounds)
        ));
    }
}

#[test]
fn complete_plan_subset_credit_requires_original_field_identity_not_equal_digests() {
    let plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
    let independent = plan.clone();
    let cloned_graph = plan.prefill.clone();
    assert_eq!(plan.digest(), independent.digest());
    assert_eq!(plan.prefill, cloned_graph);
    let complete = decoder_plan_metadata_bytes(&plan, MAX_PROGRAM_BYTES).unwrap();
    for graph in [&independent.prefill, &independent.decode, &cloned_graph] {
        assert!(matches!(
            decoder_plan_metadata_outside_graph_bytes(&plan, graph, MAX_PROGRAM_BYTES),
            Err(Error::Binding),
        ));
        // No foreign-owner credit even when the supplied cap also rejects.
        assert!(matches!(
            decoder_plan_metadata_outside_graph_bytes(&plan, graph, 0),
            Err(Error::Binding),
        ));
    }
    let independent_bytes = decoder_plan_metadata_bytes(&independent, MAX_PROGRAM_BYTES).unwrap();
    let original_graph = semantic_metadata_bytes(&plan.prefill).unwrap();
    let remainder =
        decoder_plan_metadata_outside_graph_bytes(&plan, &plan.prefill, MAX_PROGRAM_BYTES).unwrap();
    // Independently owned equal records remain charged in full, never deduped.
    assert_eq!(
        remainder + original_graph + independent_bytes,
        complete + independent_bytes
    );
}

#[test]
fn complete_plan_counts_decode_header_and_transform_spare_capacity_missing_from_prefill() {
    use pllm_models::AppliedMethod;

    let mut plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
    let complete = decoder_plan_metadata_bytes(&plan, MAX_PROGRAM_BYTES).unwrap();
    let prefill = semantic_metadata_bytes(&plan.prefill).unwrap();
    let outside =
        decoder_plan_metadata_outside_graph_bytes(&plan, &plan.prefill, MAX_PROGRAM_BYTES).unwrap();
    let adapter_capacity = plan.adapter.capacity();
    let decode_capacity = plan.decode.output.capacity();
    let transform_capacity = plan.transformations.capacity();
    plan.adapter.reserve_exact(257);
    plan.decode.output.reserve_exact(263);
    plan.transformations.reserve_exact(3);
    let extra = plan.adapter.capacity() - adapter_capacity + plan.decode.output.capacity()
        - decode_capacity
        + (plan.transformations.capacity() - transform_capacity) * size_of::<AppliedMethod>();
    assert!(extra > 0);
    assert_eq!(semantic_metadata_bytes(&plan.prefill).unwrap(), prefill);
    assert_eq!(
        decoder_plan_metadata_bytes(&plan, complete + extra).unwrap(),
        complete + extra
    );
    assert_eq!(
        decoder_plan_metadata_outside_graph_bytes(&plan, &plan.prefill, complete + extra).unwrap(),
        outside + extra,
    );
    assert!(matches!(
        decoder_plan_metadata_bytes(&plan, complete + extra - 1),
        Err(Error::Bounds),
    ));
}

#[test]
fn complete_plan_census_keeps_fresh_and_retained_empty_object_roots_in_both_graphs() {
    use serde_json::{Map, Value};

    for decode in [false, true] {
        let mut plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
        let graph = if decode {
            &mut plan.decode
        } else {
            &mut plan.prefill
        };
        graph.operations[0].attributes["owner_probe"] = Value::Null;
        let baseline = decoder_plan_metadata_bytes(&plan, MAX_PROGRAM_BYTES).unwrap();
        let node = tree_entry_bytes::<(String, Value)>().unwrap();
        let mut retained = Map::new();
        retained.insert("removed".into(), Value::Null);
        retained.remove("removed");
        for object in [Map::new(), retained] {
            let graph = if decode {
                &mut plan.decode
            } else {
                &mut plan.prefill
            };
            graph.operations[0].attributes["owner_probe"] = Value::Object(object);
            let complete = baseline + node;
            assert_eq!(
                decoder_plan_metadata_bytes(&plan, complete).unwrap(),
                complete
            );
            assert!(matches!(
                decoder_plan_metadata_bytes(&plan, complete - 1),
                Err(Error::Bounds),
            ));
            let graph = if decode { &plan.decode } else { &plan.prefill };
            let remainder =
                decoder_plan_metadata_outside_graph_bytes(&plan, graph, complete).unwrap();
            assert_eq!(
                remainder + semantic_metadata_bytes(graph).unwrap(),
                complete
            );
        }
    }
}

#[test]
fn complete_plan_census_checks_depth_16_17_even_outside_borrowed_graph() {
    use serde_json::{Map, Value};

    for decode in [false, true] {
        let mut plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
        let mut attributes = Value::Object(Map::new());
        for _ in 0..16 {
            attributes = Value::Array(vec![attributes]);
        }
        let graph = if decode {
            &mut plan.decode
        } else {
            &mut plan.prefill
        };
        graph.operations[0].attributes = attributes;
        let complete = decoder_plan_metadata_bytes(&plan, MAX_PROGRAM_BYTES).unwrap();
        assert_eq!(
            decoder_plan_metadata_bytes(&plan, complete).unwrap(),
            complete
        );
        for graph in [&plan.prefill, &plan.decode] {
            let remainder =
                decoder_plan_metadata_outside_graph_bytes(&plan, graph, complete).unwrap();
            assert_eq!(
                remainder + semantic_metadata_bytes(graph).unwrap(),
                complete
            );
        }
        let graph = if decode {
            &mut plan.decode
        } else {
            &mut plan.prefill
        };
        let boundary = graph.operations[0].attributes.take();
        graph.operations[0].attributes = Value::Array(vec![boundary]);
        assert!(matches!(
            decoder_plan_metadata_bytes(&plan, MAX_PROGRAM_BYTES),
            Err(Error::Bounds),
        ));
        let clean_graph = if decode { &plan.prefill } else { &plan.decode };
        assert!(semantic_metadata_bytes(clean_graph).is_ok());
        assert!(matches!(
            decoder_plan_metadata_outside_graph_bytes(&plan, clean_graph, MAX_PROGRAM_BYTES),
            Err(Error::Bounds),
        ));
    }
}
