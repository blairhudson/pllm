//! Stage-one checks only; no complete compact-decoder admission assertion.
use super::*;
use crate::replicated_decoder::runtime::body::{self, layer};
use crate::replicated_decoder::source_registry::SourceRegistry;
use crate::replicated_decoder::source_registry::{
    descriptor_validation_quote, SourceConstructionOwners, REGISTRY_HASH_FACTORIES,
    REGISTRY_RECORD_VALIDATIONS,
};

#[path = "source_registry_execution_tests.rs"]
mod execution_tests;

const CONFIG: &[u8] = br#"{"model_type":"llama","hidden_size":2,"intermediate_size":2,
    "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
    "vocab_size":3,"max_position_embeddings":64,"rope_theta":10000,
    "rms_norm_eps":0.000001,"tie_word_embeddings":false,"hidden_act":"silu",
    "attention_bias":false,"head_dim":2,"attention_dropout":0.0,"use_cache":true}"#;

fn manifest(with_norms: bool) -> Manifest {
    manifest_for_config(CONFIG, with_norms)
}

fn manifest_for_config(config: &[u8], with_norms: bool) -> Manifest {
    let execution = ExecutionPolicy {
        range_policy: if with_norms {
            RangePolicy::Checked
        } else {
            RangePolicy::Static
        },
        ..ExecutionPolicy::default()
    };
    let shapes =
        Program::source_shapes_with_policy(config, 1, 1, WeightMode::Public, execution).unwrap();
    let plan = program::lower_with_weights(config, 1, 1, WeightMode::Public).unwrap();
    let coefficients: std::collections::BTreeSet<_> = plan
        .prefill
        .operations
        .iter()
        .filter(|op| op.operator == pllm_models::ModelOperator::RmsNorm)
        .map(|op| op.attributes["weight"].as_str().unwrap())
        .collect();
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
                    public_q18: coefficients.contains(key.as_str()).then(|| vec![0; lanes]),
                    digest: Sha256::digest(vec![0; lanes * 8]).into(),
                    row_l1: matrix.then_some(0),
                    row_squared_norm: matrix.then_some(0),
                    row_l1_squared_sum: matrix.then_some(0),
                },
            )
        })
        .collect();
    let mut norms = SourceNorms::new();
    if with_norms {
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
        config: config.to_vec(),
        rows: 1,
        outputs: 1,
        source: [1; 32],
        composition: [2; 32],
        session: [3; 32],
        tensors,
        norms,
        program: [0; 32], // No successful expanded compilation needed by fixture.
    }
}

fn import(manifest: &Manifest) -> PublicSource {
    let mut importer = PublicSourceImport::new(manifest).unwrap();
    for (key, tensor) in &manifest.tensors {
        let bytes: Vec<_> = (0..tensor.shape.iter().product::<usize>())
            .flat_map(|_| (tensor.lower as f32 / ONE as f32).to_le_bytes())
            .collect();
        for (frame, bytes) in bytes.chunks(12).enumerate() {
            importer.append(key, frame * 3, bytes).unwrap();
        }
    }
    importer.finish().unwrap()
}

#[test]
fn registry_borrows_ordered_statistics_and_norms_without_new_backing() {
    let manifest = manifest(true);
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let source = import(&manifest);
    source.check_registry(&registry).unwrap();
    assert_eq!(registry.owned_backing_bytes(), 0);
    assert_eq!(
        registry.binding(),
        program::source_binding_with_norms(&manifest.tensors, &manifest.norms)
    );
    for (owner, key, tensor, norms) in registry.records() {
        assert!(std::ptr::eq(tensor, &manifest.tensors[key]));
        assert_eq!(tensor.row_l1, Some(0));
        assert_eq!(tensor.row_squared_norm, Some(0));
        assert_eq!(tensor.row_l1_squared_sum, Some(0));
        if let Some(norms) = norms {
            assert!(std::ptr::eq(norms, &manifest.norms[key]));
        }
        let lanes = tensor.shape.iter().product();
        let bounds = Interval::new(tensor.lower, tensor.upper).unwrap();
        registry
            .check_descriptor(owner, key, tensor.digest, lanes, bounds)
            .unwrap();
        assert!(registry
            .check_descriptor(owner, "other", tensor.digest, lanes, bounds)
            .is_err());
        assert!(registry
            .check_descriptor(owner, key, [9; 32], lanes, bounds)
            .is_err());
        assert!(registry
            .check_descriptor(owner, key, tensor.digest, lanes + 1, bounds)
            .is_err());
        assert!(registry
            .check_descriptor(
                owner,
                key,
                tensor.digest,
                lanes,
                Interval::new(0, 1).unwrap()
            )
            .is_err());
    }
    assert!(registry
        .check_descriptor(usize::MAX, "x", [1; 32], 1, Interval::new(0, 0).unwrap())
        .is_err());
}

#[test]
fn original_manifest_and_import_owners_exact_one_short_overflow_before_registry_hash() {
    for with_norms in [false, true] {
        let manifest = manifest(with_norms);
        let public = Arc::new(import(&manifest));
        let mut incoming = Vec::with_capacity(4093);
        incoming.push(0);
        let owners = SourceConstructionOwners {
            manifest: &manifest,
            incoming: Some(&incoming),
            public: Some(&public),
            retained_baseline_bytes: 137,
        };
        let quote =
            SourceRegistry::construction_quote_with_owners(&owners, MAX_PROGRAM_BYTES).unwrap();
        assert!(
            quote.retained_baseline_bytes
                >= incoming.capacity()
                    + manifest.config.capacity()
                    + public.resident_metadata_bytes().unwrap()
                    + 137
        );
        let without_public = SourceConstructionOwners {
            public: None,
            ..owners
        };
        let without_public =
            SourceRegistry::construction_quote_with_owners(&without_public, MAX_PROGRAM_BYTES)
                .unwrap();
        assert_eq!(
            quote.peak_bytes - without_public.peak_bytes,
            public.resident_metadata_bytes().unwrap()
                + std::mem::size_of::<Arc<PublicSource>>()
                + 2 * std::mem::size_of::<usize>()
        );
        REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
        REGISTRY_RECORD_VALIDATIONS.with(|n| n.set(0));
        let validated = Program::validated_source_descriptors(&manifest).unwrap();
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
        assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 1);
        assert_eq!(REGISTRY_RECORD_VALIDATIONS.with(|n| n.get()), 1);
        public.check_registry(&registry).unwrap();
        assert_eq!(Arc::strong_count(&public), 1);
        for baseline in [usize::MAX, MAX_PROGRAM_BYTES] {
            let impossible = SourceConstructionOwners {
                retained_baseline_bytes: baseline,
                ..owners
            };
            assert!(matches!(
                SourceRegistry::construction_quote_with_owners(&impossible, usize::MAX),
                Err(Error::Bounds)
            ));
        }
    }
}

#[test]
fn owner_quote_counts_spare_incoming_config_and_source_vec_capacity() {
    let mut manifest = manifest(false);
    let mut incoming = Vec::with_capacity(31);
    incoming.push(0);
    let quote = |manifest: &Manifest, incoming: &Vec<u8>| {
        SourceRegistry::construction_quote_with_owners(
            &SourceConstructionOwners {
                manifest,
                incoming: Some(incoming),
                public: None,
                retained_baseline_bytes: 0,
            },
            MAX_PROGRAM_BYTES,
        )
        .unwrap()
    };
    let before = quote(&manifest, &incoming);
    let old_incoming = incoming.capacity();
    let old_config = manifest.config.capacity();
    incoming.reserve_exact(4093);
    manifest.config.reserve_exact(4093);
    let after = quote(&manifest, &incoming);
    assert_eq!(
        after.peak_bytes - before.peak_bytes,
        incoming.capacity() - old_incoming + manifest.config.capacity() - old_config
    );
    assert_eq!(after.new_owned_peak_bytes, before.new_owned_peak_bytes);
    let tensor = manifest
        .tensors
        .values_mut()
        .find(|v| v.public_q18.is_some())
        .unwrap();
    let values = tensor.public_q18.as_mut().unwrap();
    let old_values = values.capacity();
    values.reserve_exact(127);
    let extra = (values.capacity() - old_values) * std::mem::size_of::<i64>();
    let grown = quote(&manifest, &incoming);
    assert_eq!(
        grown.borrowed_metadata_bytes - after.borrowed_metadata_bytes,
        extra
    );
    assert_eq!(grown.new_owned_peak_bytes, after.new_owned_peak_bytes);
}

#[test]
fn foreign_equal_binding_validation_token_cannot_substitute_quoted_manifest_owner() {
    let manifest = manifest(true);
    let foreign = self::manifest(true);
    let owners = SourceConstructionOwners {
        manifest: &manifest,
        incoming: None,
        public: None,
        retained_baseline_bytes: 0,
    };
    assert_eq!(
        program::source_binding_with_norms(&manifest.tensors, &manifest.norms),
        program::source_binding_with_norms(&foreign.tensors, &foreign.norms)
    );
    let validated = Program::validated_source_descriptors(&foreign).unwrap();
    REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
    assert!(matches!(
        SourceRegistry::from_validated_with_owners_checked(validated, &owners, MAX_PROGRAM_BYTES,),
        Err(Error::Binding)
    ));
    assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
}

#[test]
fn reachable_descriptor_validation_quote_exact_one_short_and_actual_graph_capacity() {
    // Existing graph/shapes are baseline. These tests do not preadmit fixture
    // parsing/lowering, and never replace the original descriptor validator.
    for with_norms in [false, true] {
        let manifest = manifest(with_norms);
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
            retained_baseline_bytes: 293,
        };
        REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
        let quote = descriptor_validation_quote(&plan.prefill, &shapes, &owners, MAX_PROGRAM_BYTES)
            .unwrap();
        assert_eq!(
            descriptor_validation_quote(&plan.prefill, &shapes, &owners, quote.peak_bytes).unwrap(),
            quote
        );
        assert!(matches!(
            descriptor_validation_quote(&plan.prefill, &shapes, &owners, quote.peak_bytes - 1),
            Err(Error::Bounds)
        ));
        assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
        let old_capacity = plan.prefill.operations.capacity();
        plan.prefill.operations.reserve_exact(127);
        let grown = descriptor_validation_quote(&plan.prefill, &shapes, &owners, MAX_PROGRAM_BYTES)
            .unwrap();
        assert_eq!(
            grown.peak_bytes - quote.peak_bytes,
            (plan.prefill.operations.capacity() - old_capacity)
                * std::mem::size_of::<pllm_models::ModelOperation>()
        );
        assert_eq!(grown.new_owned_peak_bytes, quote.new_owned_peak_bytes);
        let impossible = SourceConstructionOwners {
            retained_baseline_bytes: usize::MAX,
            ..owners
        };
        assert!(matches!(
            descriptor_validation_quote(&plan.prefill, &shapes, &impossible, usize::MAX),
            Err(Error::Bounds)
        ));
        // A successful quote conveys no validation authority. Compiler still
        // performs all original source-shape/numeric/coefficient checks.
        SourceRegistry::from_validated(Program::validated_source_descriptors(&manifest).unwrap())
            .unwrap();
    }
}

#[test]
fn capacity_only_descriptor_quote_does_not_replace_original_coefficient_validation() {
    let mut manifest = manifest(false);
    let shapes = Program::source_shapes_with_policy(
        &manifest.config,
        manifest.rows,
        manifest.outputs,
        manifest.weights,
        manifest.execution,
    )
    .unwrap();
    let plan = program::lower_with_weights(
        &manifest.config,
        manifest.rows,
        manifest.outputs,
        manifest.weights,
    )
    .unwrap();
    let values = manifest
        .tensors
        .values_mut()
        .find_map(|source| source.public_q18.as_mut())
        .unwrap();
    values[0] = 1; // Outside the original zero interval; digest is unchanged.
    let owners = SourceConstructionOwners {
        manifest: &manifest,
        incoming: None,
        public: None,
        retained_baseline_bytes: 0,
    };
    REGISTRY_HASH_FACTORIES.with(|n| n.set(0));
    descriptor_validation_quote(&plan.prefill, &shapes, &owners, MAX_PROGRAM_BYTES).unwrap();
    assert!(Program::validated_source_descriptors(&manifest).is_err());
    assert_eq!(REGISTRY_HASH_FACTORIES.with(|n| n.get()), 0);
}

#[test]
fn expanded_and_registry_checks_share_binding_file_count_digest_and_extent_failures() {
    let manifest = manifest(false);
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let program = Program::compile_with_norms(
        &manifest.config,
        manifest.rows,
        manifest.outputs,
        &manifest.tensors,
        manifest.weights,
        manifest.execution,
        &manifest.norms,
    )
    .unwrap();
    let agrees = |source: &PublicSource, valid| {
        assert_eq!(source.check(&program).is_ok(), valid);
        assert_eq!(source.check_registry(&registry).is_ok(), valid);
    };
    let mut source = import(&manifest);
    agrees(&source, true);
    let extra_bytes = [0u8; 8];
    let mut writer = PublicFileWriter::new(8, Sha256::digest(extra_bytes).into()).unwrap();
    writer.write(&extra_bytes).unwrap();
    source
        .files
        .insert("extra".into(), writer.finish().unwrap());
    agrees(&source, false);
    source.files.remove("extra");
    source.binding[0] ^= 1;
    agrees(&source, false);
    source.binding[0] ^= 1;
    let key = registry.files().next().unwrap().key.to_owned();
    let file = source.files.remove(&key).unwrap();
    agrees(&source, false);
    source.files.insert("extra".into(), file);
    agrees(&source, false);
    let file = source.files.remove("extra").unwrap();
    source.files.insert(key.clone(), file);
    agrees(&source, true);
    let bytes = vec![0u8; source.files[&key].size() as usize + 8];
    let mut writer =
        PublicFileWriter::new(bytes.len() as u64, Sha256::digest(&bytes).into()).unwrap();
    writer.write(&bytes).unwrap();
    source.files.insert(key.clone(), writer.finish().unwrap());
    agrees(&source, false);
    let bytes = vec![1u8; source.files[&key].size() as usize - 8];
    let mut writer =
        PublicFileWriter::new(bytes.len() as u64, Sha256::digest(&bytes).into()).unwrap();
    writer.write(&bytes).unwrap();
    source.files.insert(key, writer.finish().unwrap());
    agrees(&source, false);
    source.files.clear();
    agrees(&source, false);
}

#[test]
fn registry_checks_source_owner_without_successful_expanded_numeric_admission() {
    let mut manifest = manifest(false);
    let source = manifest
        .tensors
        .get_mut("model.embed_tokens.weight")
        .unwrap();
    let value = 1i64 << 40;
    source.lower = value;
    source.upper = value;
    source.row_l1 = None;
    source.row_squared_norm = None;
    source.row_l1_squared_sum = None;
    let mut hash = Sha256::new();
    for _ in 0..source.shape.iter().product::<usize>() {
        hash.update(value.to_le_bytes());
    }
    source.digest = hash.finalize().into();
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    import(&manifest).check_registry(&registry).unwrap();
    assert!(Program::compile_with_norms(
        &manifest.config,
        1,
        1,
        &manifest.tensors,
        manifest.weights,
        manifest.execution,
        &manifest.norms,
    )
    .is_err());
}

#[test]
fn registry_reuses_original_descriptor_validation_and_never_authenticates_raw_metadata() {
    let mut manifest = manifest(false);
    let key = "model.embed_tokens.weight";
    manifest.tensors.get_mut(key).unwrap().digest = [0; 32];
    assert!(SourceRegistry::from_manifest(&manifest).is_err());
    let mut manifest = self::manifest(false);
    manifest.tensors.get_mut(key).unwrap().shape[0] += 1;
    assert!(SourceRegistry::from_manifest(&manifest).is_err());
    let mut manifest = self::manifest(false);
    manifest.tensors.get_mut(key).unwrap().row_l1 = Some(1);
    assert!(SourceRegistry::from_manifest(&manifest).is_err());
    let mut manifest = self::manifest(false);
    manifest.tensors.get_mut(key).unwrap().public_q18 = Some(vec![0; 6]);
    assert!(SourceRegistry::from_manifest(&manifest).is_err());
    let mut manifest = self::manifest(true);
    manifest.norms.pop_first();
    assert!(SourceRegistry::from_manifest(&manifest).is_err());

    let mut manifest = self::manifest(false);
    let original = import(&manifest);
    // Structurally valid metadata is not authentication. Changing even a
    // statistic within its legal range changes the original source binding.
    let tensor = manifest.tensors.get_mut(key).unwrap();
    tensor.upper = 1;
    tensor.row_l1 = Some(1);
    tensor.row_squared_norm = Some(1);
    tensor.row_l1_squared_sum = Some(1);
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    assert!(original.check_registry(&registry).is_err());
    let mut importer = PublicSourceImport::new(&manifest).unwrap();
    // Ordered import still recomputes every claimed source statistic.
    let mut rejected = false;
    for (name, tensor) in &manifest.tensors {
        let bytes = vec![0; tensor.shape.iter().product::<usize>() * 4];
        if importer.append(name, 0, &bytes).is_err() {
            rejected = true;
            break;
        }
    }
    assert!(rejected);
    assert!(importer.finish().is_err());
}

fn bias_manifest() -> Manifest {
    // Use the existing bias-capable source adapter. The base dense-gated
    // fixture explicitly rejects attention_bias=true; it cannot exercise the
    // authenticated matrix-plus-bias handoff.
    let config = std::str::from_utf8(CONFIG)
        .unwrap()
        .replace("\"model_type\":\"llama\"", "\"model_type\":\"qwen2\"")
        .replace("\"attention_bias\":false", "\"attention_bias\":true");
    let mut manifest = manifest_for_config(config.as_bytes(), false);
    for (key, tensor) in &mut manifest.tensors {
        let value = if key.ends_with("q_proj.weight") {
            ONE
        } else if key.ends_with("q_proj.bias") {
            -ONE
        } else {
            continue;
        };
        tensor.lower = value;
        tensor.upper = value;
        tensor.row_l1 = None;
        tensor.row_squared_norm = None;
        tensor.row_l1_squared_sum = None;
        let mut hash = Sha256::new();
        for _ in 0..tensor.shape.iter().product::<usize>() {
            hash.update(value.to_le_bytes());
        }
        tensor.digest = hash.finalize().into();
    }
    manifest
}

#[derive(Clone)]
struct Descriptor {
    owner: usize,
    key: String,
    digest: [u8; 32],
    lanes: usize,
    bounds: Interval,
}

impl Descriptor {
    fn find(registry: &SourceRegistry<'_>, suffix: &str) -> Self {
        let (owner, key, tensor, _) = registry
            .records()
            .find(|(_, key, _, _)| key.ends_with(suffix))
            .unwrap();
        Self {
            owner,
            key: key.into(),
            digest: tensor.digest,
            lanes: tensor.shape.iter().product(),
            bounds: Interval::new(tensor.lower, tensor.upper).unwrap(),
        }
    }

    fn unchecked(&self) -> Arc<layer::Source> {
        layer::Source::new(
            self.owner,
            self.key.clone(),
            self.digest,
            self.lanes,
            self.bounds,
        )
        .unwrap()
    }
}

/// PUBLIC fixed seeds, fresh consuming owners and isolated routers only. No
/// Program, runtime clone/reset, or production entropy change supplies this owner.
fn semantic_runtimes(plan: &layer::Plan) -> [multiply::PartyRuntime; 3] {
    puma::prss::insecure_test_support::fresh_compound_parties(
        puma::Context::new([42; 32], plan.digest(), 0).unwrap(),
        plan.work_budget().limits().unwrap(),
    )
    .unwrap()
}

fn semantic_input(party: puma::Party, lanes: usize) -> Secret {
    let mut secret = Secret::new(lanes).unwrap();
    for i in 0..lanes {
        let value = ((i as i64 + 1) * ONE) as u64;
        if party == puma::Party::P0 {
            secret.first[i] = value;
        }
        if party == puma::Party::P2 {
            secret.second[i] = value;
        }
    }
    secret
}

fn single_source_plan(descriptor: &Descriptor, matrix: bool) -> Arc<layer::Plan> {
    let lanes = if matrix { 2 } else { descriptor.lanes };
    let mut builder = layer::Builder::new(
        WeightMode::Public,
        &[body::Register {
            lanes,
            bounds: Interval::new(ONE, (lanes as i64) * ONE).unwrap(),
        }],
    )
    .unwrap();
    let input = builder.input(0).unwrap();
    let output = if matrix {
        builder
            .public_matrix(
                &input,
                descriptor.unchecked(),
                1,
                2,
                descriptor.lanes / 2,
                false,
                true,
            )
            .unwrap()
    } else {
        builder
            .bias(&input, descriptor.unchecked(), descriptor.lanes)
            .unwrap()
    };
    builder.finish(&[output]).unwrap()
}

#[test]
fn semantic_registry_constructor_executes_imported_matrix_and_bias_without_program() {
    let manifest = bias_manifest();
    let public = Arc::new(import(&manifest));
    let (mut owners, plan) = {
        let registry = SourceRegistry::from_manifest(&manifest).unwrap();
        let weight = Descriptor::find(&registry, "q_proj.weight");
        let bias = Descriptor::find(&registry, "q_proj.bias");
        let mut builder = layer::Builder::new(
            WeightMode::Public,
            &[body::Register {
                lanes: 2,
                bounds: Interval::new(ONE, 2 * ONE).unwrap(),
            }],
        )
        .unwrap();
        let input = builder.input(0).unwrap();
        let matrix = builder
            .public_matrix(&input, weight.unchecked(), 1, 2, 2, false, true)
            .unwrap();
        let output = builder.bias(&matrix, bias.unchecked(), 2).unwrap();
        let plan = builder.finish(&[output]).unwrap();
        let owners = semantic_runtimes(&plan).map(|runtime| {
            let input = semantic_input(runtime.party(), 2);
            layer::Executor::new_with_registry(
                plan.clone(),
                runtime,
                vec![input],
                Some(public.clone()),
                &registry,
            )
            .unwrap()
        });
        (owners, plan)
    };
    // No registry/manifest lifetime is retained by the executors.
    drop(manifest);
    assert_eq!(Arc::strong_count(&public), 4);
    let parties = [puma::Party::P0, puma::Party::P1, puma::Party::P2];
    let mut queue = Vec::new();
    for _ in 0..100_000 {
        if owners.iter().all(layer::Executor::complete) {
            break;
        }
        if queue.is_empty() {
            for (i, owner) in owners.iter_mut().enumerate() {
                queue.extend(
                    owner
                        .advance()
                        .unwrap()
                        .into_iter()
                        .map(|(to, frame)| (parties[i], to, frame)),
                );
            }
        } else {
            for (from, to, frame) in std::mem::take(&mut queue) {
                queue.extend(
                    owners[to as usize]
                        .receive(frame.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, frame)| (to, next, frame)),
                );
            }
        }
    }
    assert!(queue.is_empty());
    assert!(owners.iter().all(layer::Executor::complete));
    let finished = owners.map(|owner| owner.finish().unwrap());
    for index in 0..2 {
        let shares = finished
            .each_ref()
            .map(|(_, values)| values.word(0, index).unwrap());
        for party in 0..3 {
            assert_eq!(shares[party].1, shares[(party + 1) % 3].0);
        }
        assert_eq!(
            shares
                .iter()
                .fold(0u64, |sum, value| sum.wrapping_add(value.0)),
            (2 * ONE) as u64
        );
    }
    for (runtime, _) in finished {
        let actual = runtime.cursors();
        let budget = plan.work_budget().cursors();
        assert_eq!(actual.public_blocks, budget.public_blocks);
        assert_eq!(actual.pair_blocks, budget.pair_blocks);
        assert!(actual.private_blocks <= budget.private_blocks);
        assert!(actual.aes_work <= budget.aes_work);
    }
    assert_eq!(Arc::strong_count(&public), 1);
}

#[test]
fn semantic_registry_constructor_rejects_every_unchecked_matrix_and_bias_descriptor() {
    let manifest = bias_manifest();
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let public = Arc::new(import(&manifest));
    for matrix in [true, false] {
        let descriptor = Descriptor::find(
            &registry,
            if matrix {
                "q_proj.weight"
            } else {
                "q_proj.bias"
            },
        );
        for field in 0..6 {
            let mut bad = descriptor.clone();
            match field {
                0 => bad.owner = usize::MAX,
                1 => bad.key = "unchecked.other".into(),
                2 => bad.digest = [9; 32],
                3 => bad.lanes += if matrix { 2 } else { 1 },
                4 => bad.bounds = Interval::new(-ONE, ONE).unwrap(),
                5 => {
                    bad.owner = Descriptor::find(
                        &registry,
                        if matrix {
                            "q_proj.bias"
                        } else {
                            "q_proj.weight"
                        },
                    )
                    .owner
                }
                _ => unreachable!(),
            }
            let plan = single_source_plan(&bad, matrix);
            let runtime = semantic_runtimes(&plan).into_iter().next().unwrap();
            let input = semantic_input(runtime.party(), if matrix { 2 } else { bad.lanes });
            assert!(
                matches!(
                    layer::Executor::new_with_registry(
                        plan,
                        runtime,
                        vec![input],
                        Some(public.clone()),
                        &registry,
                    ),
                    Err(Error::Binding)
                ),
                "matrix={matrix}, field={field}"
            );
            assert_eq!(Arc::strong_count(&public), 1);
        }
    }
}

#[test]
fn semantic_registry_constructor_rejects_missing_and_wrong_authenticated_file_owners() {
    let manifest = bias_manifest();
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    for matrix in [true, false] {
        let descriptor = Descriptor::find(
            &registry,
            if matrix {
                "q_proj.weight"
            } else {
                "q_proj.bias"
            },
        );
        let plan = single_source_plan(&descriptor, matrix);
        for failure in 0..6 {
            let mut public = import(&manifest);
            match failure {
                0 => public.binding[0] ^= 1,
                1 => {
                    public.files.remove(&descriptor.key);
                }
                2 | 3 => {
                    let lanes = descriptor.lanes + usize::from(failure == 3);
                    let bytes = vec![7u8; lanes * 8];
                    let mut writer =
                        PublicFileWriter::new(bytes.len() as u64, Sha256::digest(&bytes).into())
                            .unwrap();
                    writer.write(&bytes).unwrap();
                    public
                        .files
                        .insert(descriptor.key.clone(), writer.finish().unwrap());
                }
                4 => {
                    let bytes = [0u8; 8];
                    let mut writer =
                        PublicFileWriter::new(8, Sha256::digest(bytes).into()).unwrap();
                    writer.write(&bytes).unwrap();
                    public
                        .files
                        .insert("extra".into(), writer.finish().unwrap());
                }
                5 => (), // Missing PublicSource owner itself.
                _ => unreachable!(),
            }
            let runtime = semantic_runtimes(&plan).into_iter().next().unwrap();
            let input = semantic_input(runtime.party(), 2);
            let public = (failure != 5).then(|| Arc::new(public));
            assert!(
                matches!(
                    layer::Executor::new_with_registry(
                        plan.clone(),
                        runtime,
                        vec![input],
                        public,
                        &registry,
                    ),
                    Err(Error::Binding)
                ),
                "matrix={matrix}, failure={failure}"
            );
        }
    }
}
