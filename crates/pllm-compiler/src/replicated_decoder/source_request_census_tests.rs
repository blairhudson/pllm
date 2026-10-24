//! Fixtures are already constructed. These tests do not preadmit their parsing,
//! lowering or source-shape generation, and confer no file/source authority.
use super::*;
use crate::replicated_decoder::{
    ExecutionPolicy, Program, SourceNorms, WeightMode, MAX_PROGRAM_BYTES,
};
use sha2::{Digest as _, Sha256};

const CONFIG: &[u8] = br#"{"model_type":"llama","hidden_size":2,"intermediate_size":2,
    "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
    "vocab_size":3,"max_position_embeddings":64,"rope_theta":10000,
    "rms_norm_eps":0.000001,"tie_word_embeddings":false,"hidden_act":"silu",
    "attention_bias":false,"head_dim":2,"attention_dropout":0.0,"use_cache":true}"#;

fn fixture() -> (Manifest, DecoderPlan, BTreeMap<String, Vec<usize>>) {
    let execution = ExecutionPolicy::default();
    let shapes =
        Program::source_shapes_with_policy(CONFIG, 1, 1, WeightMode::Public, execution).unwrap();
    let tensors = shapes
        .iter()
        .map(|(key, shape)| {
            let lanes = shape.iter().product::<usize>();
            let matrix = shape.len() == 2;
            (
                key.clone(),
                SourceTensor {
                    shape: shape.clone(),
                    lower: 0,
                    upper: 0,
                    digest: Sha256::digest(vec![0u8; lanes * 8]).into(),
                    public_q18: key.contains("norm").then(|| vec![0; lanes]),
                    row_l1: matrix.then_some(0),
                    row_squared_norm: matrix.then_some(0),
                    row_l1_squared_sum: matrix.then_some(0),
                },
            )
        })
        .collect();
    let manifest = Manifest {
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
        norms: SourceNorms::new(),
        program: [0; 32],
    };
    let plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
    (manifest, plan, shapes)
}

fn census<'a>(
    manifest: &'a Manifest,
    plan: &'a DecoderPlan,
    shapes: &'a BTreeMap<String, Vec<usize>>,
) -> SourceRequestCensus<'a> {
    SourceRequestCensus {
        manifest,
        plan,
        graph: &plan.prefill,
        shapes,
        incoming: None,
        parsed_config: None,
        public: None,
    }
}

#[test]
fn stage_quotes_exact_fit_one_short_and_unavailable_aggregate() {
    assert_eq!(MAX_PROGRAM_BYTES, 64 << 20);
    let (manifest, plan, shapes) = fixture();
    let request = census(&manifest, &plan, &shapes);
    let descriptor = request
        .descriptor_validation_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    assert_eq!(
        request
            .descriptor_validation_quote(descriptor.peak_bytes)
            .unwrap(),
        descriptor,
    );
    assert!(matches!(
        request.descriptor_validation_quote(descriptor.peak_bytes - 1),
        Err(Error::Bounds),
    ));
    let registry = request
        .registry_construction_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    assert_eq!(
        request
            .registry_construction_quote(registry.peak_bytes)
            .unwrap(),
        registry,
    );
    assert!(matches!(
        request.registry_construction_quote(registry.peak_bytes - 1),
        Err(Error::Bounds),
    ));
    assert_eq!(request.aggregate_constructor_completion_bound(), None);
    assert!(matches!(
        request.registry_construction_quote(0),
        Err(Error::Bounds)
    ));
}

#[test]
fn composition_counts_complete_plan_once_and_keeps_exact_borrows() {
    let (manifest, plan, shapes) = fixture();
    let request = census(&manifest, &plan, &shapes);
    let descriptor = request.descriptor_owners(MAX_PROGRAM_BYTES).unwrap();
    let registry = request.registry_owners(MAX_PROGRAM_BYTES).unwrap();
    assert!(std::ptr::eq(descriptor.manifest, &manifest));
    assert!(std::ptr::eq(request.shapes, &shapes));
    assert_eq!(
        registry.retained_baseline_bytes - descriptor.retained_baseline_bytes,
        source_registry::semantic_metadata_bytes(&plan.prefill).unwrap()
            + shape_map_bytes(&shapes, MAX_PROGRAM_BYTES).unwrap(),
    );
    let shared = source_registry::decoder_plan_metadata_bytes(&plan, MAX_PROGRAM_BYTES).unwrap();
    assert_eq!(
        shared,
        size_of::<DecoderPlan>() + plan.checked_owned_backing_bytes(MAX_PROGRAM_BYTES).unwrap(),
    );
    // The complete plan must fit before any selected-graph subtraction.
    assert!(matches!(
        request.descriptor_owners(shared - 1),
        Err(Error::Bounds)
    ));
    // Decode gets exactly its own inline/backing subtraction too, not prefill's.
    let decode = SourceRequestCensus {
        graph: &plan.decode,
        ..census(&manifest, &plan, &shapes)
    };
    assert_eq!(
        decode
            .registry_owners(MAX_PROGRAM_BYTES)
            .unwrap()
            .retained_baseline_bytes
            - decode
                .descriptor_owners(MAX_PROGRAM_BYTES)
                .unwrap()
                .retained_baseline_bytes,
        source_registry::semantic_metadata_bytes(&plan.decode).unwrap()
            + shape_map_bytes(&shapes, MAX_PROGRAM_BYTES).unwrap(),
    );
}

#[test]
fn complete_plan_owner_keeps_non_graph_strings_and_spare_transform_slots() {
    let (manifest, mut plan, shapes) = fixture();
    let before = census(&manifest, &plan, &shapes)
        .descriptor_validation_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    let old =
        plan.schema_version.capacity() + plan.model_family.capacity() + plan.adapter.capacity();
    let old_methods = plan.transformations.capacity();
    plan.schema_version.reserve_exact(23);
    plan.model_family.reserve_exact(29);
    plan.adapter.reserve_exact(31);
    plan.transformations.reserve_exact(7);
    let after = census(&manifest, &plan, &shapes)
        .descriptor_validation_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    assert_eq!(
        after.peak_bytes - before.peak_bytes,
        plan.schema_version.capacity() + plan.model_family.capacity() + plan.adapter.capacity()
            - old
            + (plan.transformations.capacity() - old_methods)
                * size_of::<pllm_models::AppliedMethod>(),
    );
    assert_eq!(after.new_owned_peak_bytes, before.new_owned_peak_bytes);
}

#[test]
fn structurally_equal_foreign_graph_receives_no_identity_credit() {
    let (manifest, plan, shapes) = fixture();
    let foreign = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
    let request = SourceRequestCensus {
        graph: &foreign.prefill,
        ..census(&manifest, &plan, &shapes)
    };
    assert!(matches!(
        request.descriptor_validation_quote(MAX_PROGRAM_BYTES),
        Err(Error::Binding),
    ));
    assert!(matches!(
        request.registry_construction_quote(MAX_PROGRAM_BYTES),
        Err(Error::Binding),
    ));
    let decode = SourceRequestCensus {
        graph: &plan.decode,
        ..census(&manifest, &plan, &shapes)
    };
    assert!(decode
        .descriptor_validation_quote(MAX_PROGRAM_BYTES)
        .is_ok());
}

#[test]
fn spare_capacity_in_nonselected_graph_incoming_config_and_shapes_stays_charged() {
    let (mut manifest, mut plan, mut shapes) = fixture();
    let mut incoming = Vec::with_capacity(31);
    incoming.push(0);
    let quote = |manifest: &Manifest,
                 plan: &DecoderPlan,
                 shapes: &BTreeMap<String, Vec<usize>>,
                 incoming: &Vec<u8>| {
        SourceRequestCensus {
            incoming: Some(incoming),
            ..census(manifest, plan, shapes)
        }
        .descriptor_validation_quote(MAX_PROGRAM_BYTES)
        .unwrap()
    };
    let before = quote(&manifest, &plan, &shapes, &incoming);
    let old_graph = plan.decode.operations.capacity();
    let old_incoming = incoming.capacity();
    let old_config = manifest.config.capacity();
    let old_shape = shapes.values().next().unwrap().capacity();
    plan.decode.operations.reserve_exact(11);
    incoming.reserve_exact(79);
    manifest.config.reserve_exact(97);
    shapes.values_mut().next().unwrap().reserve_exact(13);
    let after = quote(&manifest, &plan, &shapes, &incoming);
    assert_eq!(
        after.peak_bytes - before.peak_bytes,
        (plan.decode.operations.capacity() - old_graph) * size_of::<pllm_models::ModelOperation>()
            + incoming.capacity()
            - old_incoming
            + manifest.config.capacity()
            - old_config
            + (shapes.values().next().unwrap().capacity() - old_shape) * size_of::<usize>(),
    );
    assert_eq!(after.new_owned_peak_bytes, before.new_owned_peak_bytes);
}

#[test]
fn parsed_config_uses_shared_full_json_census_including_spare_capacity() {
    let (manifest, plan, shapes) = fixture();
    let mut entries = Vec::with_capacity(17);
    entries.push(serde_json::json!({"extra": "retained"}));
    let parsed = serde_json::Value::Array(entries);
    let before = census(&manifest, &plan, &shapes)
        .descriptor_validation_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    let request = SourceRequestCensus {
        parsed_config: Some(&parsed),
        ..census(&manifest, &plan, &shapes)
    };
    let after = request
        .descriptor_validation_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    assert_eq!(
        after.peak_bytes - before.peak_bytes,
        size_of::<serde_json::Value>()
            + owned_backing::checked_json_backing_bytes(&parsed, MAX_PROGRAM_BYTES).unwrap(),
    );
    assert!(matches!(add(usize::MAX, 1, usize::MAX), Err(Error::Bounds)));
    assert!(matches!(
        add(MAX_PROGRAM_BYTES, 1, usize::MAX),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        source_registry::array_bytes::<usize>(usize::MAX),
        Err(Error::Bounds)
    ));
}

#[test]
fn source_capacity_growth_is_observed_without_validating_or_hashing_records() {
    let (mut manifest, plan, shapes) = fixture();
    source_registry::REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
    source_registry::REGISTRY_RECORD_VALIDATIONS.with(|n| n.set(0));
    let before = census(&manifest, &plan, &shapes)
        .descriptor_validation_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    let tensor = manifest
        .tensors
        .values_mut()
        .find(|v| v.public_q18.is_some())
        .unwrap();
    let old_shape = tensor.shape.capacity();
    let old_values = tensor.public_q18.as_ref().unwrap().capacity();
    tensor.shape.reserve_exact(11);
    tensor.public_q18.as_mut().unwrap().reserve_exact(19);
    let delta = (tensor.shape.capacity() - old_shape) * size_of::<usize>()
        + (tensor.public_q18.as_ref().unwrap().capacity() - old_values) * size_of::<i64>();
    let after = census(&manifest, &plan, &shapes)
        .descriptor_validation_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    assert_eq!(after.peak_bytes - before.peak_bytes, delta);
    assert_eq!(
        source_registry::REGISTRY_HASH_FACTORIES.with(|n| n.get()),
        0
    );
    assert_eq!(
        source_registry::REGISTRY_RECORD_VALIDATIONS.with(|n| n.get()),
        0
    );
    // Census does not authenticate coefficients, even with a correct shape map.
    *manifest
        .tensors
        .values_mut()
        .find_map(|v| v.public_q18.as_mut())
        .unwrap()
        .first_mut()
        .unwrap() = 1;
    assert!(census(&manifest, &plan, &shapes)
        .descriptor_validation_quote(MAX_PROGRAM_BYTES)
        .is_ok());
    assert!(Program::validated_source_descriptors(&manifest).is_err());
}

#[test]
fn parsed_config_depth_limit_remains_checked_without_recursive_allocation() {
    let (manifest, plan, shapes) = fixture();
    let mut parsed = serde_json::Value::Null;
    for _ in 0..=owned_backing::MAX_JSON_DEPTH {
        parsed = serde_json::Value::Array(vec![parsed]);
    }
    let request = SourceRequestCensus {
        parsed_config: Some(&parsed),
        ..census(&manifest, &plan, &shapes)
    };
    assert!(matches!(
        request.descriptor_validation_quote(MAX_PROGRAM_BYTES),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        request.registry_construction_quote(MAX_PROGRAM_BYTES),
        Err(Error::Bounds)
    ));
}

#[test]
fn original_validation_and_both_map_identities_still_required() {
    let (manifest, plan, shapes) = fixture();
    let request = census(&manifest, &plan, &shapes);
    let quote = request
        .registry_construction_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    let validated = Program::validated_source_descriptors(&manifest).unwrap();
    assert!(matches!(
        request.registry_from_validated_checked(validated, quote.peak_bytes - 1),
        Err(Error::Bounds),
    ));
    let validated = Program::validated_source_descriptors(&manifest).unwrap();
    let registry = {
        // Returned registry borrows actual Manifest maps, never this temporary
        // census or its temporary SourceConstructionOwners adapter.
        let stage = census(&manifest, &plan, &shapes);
        stage
            .registry_from_validated_checked(validated, quote.peak_bytes)
            .unwrap()
    };
    assert!(std::ptr::eq(registry.source_records(), &manifest.tensors));
    assert!(std::ptr::eq(registry.source_norms(), &manifest.norms));
    let (foreign, _, _) = fixture();
    let validated = Program::validated_source_descriptors(&foreign).unwrap();
    assert!(matches!(
        request.registry_from_validated_checked(validated, MAX_PROGRAM_BYTES),
        Err(Error::Binding),
    ));
}

#[test]
fn empty_shape_map_retains_possible_tree_root_without_validation_authority() {
    let mut shapes = BTreeMap::new();
    shapes.insert("removed".into(), vec![1]);
    shapes.remove("removed");
    let bytes = size_of::<BTreeMap<String, Vec<usize>>>()
        + source_registry::tree_entry_bytes::<(String, Vec<usize>)>().unwrap();
    assert_eq!(shape_map_bytes(&shapes, bytes).unwrap(), bytes);
    assert!(matches!(
        shape_map_bytes(&shapes, bytes - 1),
        Err(Error::Bounds)
    ));
}

#[test]
fn actual_public_source_metadata_arc_and_possible_empty_root_remain_charged() {
    let (manifest, plan, shapes) = fixture();
    // Original import API constructs the fixture; this test is not permission
    // to construct it or proof of aggregate admission. No fake source handle.
    let mut importer = crate::replicated_decoder::PublicSourceImport::new(&manifest).unwrap();
    for (key, tensor) in &manifest.tensors {
        let lanes = tensor.shape.iter().product::<usize>();
        importer.append(key, 0, &vec![0u8; lanes * 4]).unwrap();
    }
    let public = Arc::new(importer.finish().unwrap());
    let before = census(&manifest, &plan, &shapes)
        .registry_construction_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    let request = SourceRequestCensus {
        public: Some(&public),
        ..census(&manifest, &plan, &shapes)
    };
    let after = request
        .registry_construction_quote(MAX_PROGRAM_BYTES)
        .unwrap();
    let expected = size_of::<Arc<PublicSource>>()
        + source_registry::array_bytes::<usize>(2).unwrap()
        + public.resident_metadata_bytes().unwrap()
        + source_registry::tree_entry_bytes::<(String, pllm_core::public_file::PublicFile)>()
            .unwrap();
    assert_eq!(after.peak_bytes - before.peak_bytes, expected);
    assert_eq!(after.new_owned_peak_bytes, before.new_owned_peak_bytes);
    assert_eq!(
        request
            .registry_construction_quote(after.peak_bytes)
            .unwrap(),
        after
    );
    assert!(matches!(
        request.registry_construction_quote(after.peak_bytes - 1),
        Err(Error::Bounds)
    ));
}
