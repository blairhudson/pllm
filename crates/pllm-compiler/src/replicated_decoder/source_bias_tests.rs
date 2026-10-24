//! Genuine guarded Qwen2 source plan; no trimmed semantic graph or oracle permit.
use super::*;
use source_bias::handoff::{self, Compiled};

fn fixture(width: usize) -> (Fixture, Vec<i64>, Vec<i64>) {
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type":"qwen2", "hidden_size":width, "intermediate_size":2,
        "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
        "head_dim":width, "vocab_size":3, "tie_word_embeddings":false,
        "hidden_act":"silu", "attention_bias":true, "max_position_embeddings":64,
        "rope_theta":10000, "rms_norm_eps":0.000001, "attention_dropout":0.0,
        "use_sliding_window":false, "use_cache":true, "max_window_layers":1
    }))
    .unwrap();
    let execution = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    // Metadata-only shape lowering, NOT an expanded executable/compiled oracle.
    let shapes =
        Program::source_shapes_with_policy(&config, 4, 2, WeightMode::Public, execution).unwrap();
    let mut tensors = BTreeMap::new();
    let mut values = BTreeMap::new();
    for (key, shape) in shapes {
        let lanes: usize = shape.iter().product();
        let clear: Vec<i64> = (0..lanes)
            .map(|i| {
                if key.contains("norm") {
                    ONE
                } else if key == "model.embed_tokens.weight" {
                    if i % width % 2 == 0 { ONE / 4 } else { ONE / 8 }
                } else if key.ends_with("q_proj.weight") {
                    if i % width == i / width {
                        if i / width % 2 == 0 { -1 } else { 1 }
                    } else {
                        0
                    }
                } else if key.ends_with("q_proj.bias") {
                    if i % 2 == 0 { -7 } else { 3 }
                } else {
                    0
                }
            })
            .collect();
        let digest = Sha256::digest(
            clear
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .into();
        tensors.insert(
            key.clone(),
            SourceTensor {
                shape,
                lower: *clear.iter().min().unwrap(),
                upper: *clear.iter().max().unwrap(),
                public_q18: key.contains("norm").then(|| clear.clone()),
                digest,
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
        );
        values.insert(key, clear);
    }
    let manifest = Manifest {
        schema: "pllm.replicated-decoder.manifest.v1".into(),
        weights: WeightMode::Public,
        execution,
        config,
        rows: 4,
        outputs: 2,
        source: [1; 32],
        composition: [2; 32],
        session: [3; 32],
        tensors,
        norms: BTreeMap::new(),
        program: [0; 32],
    };
    let mut importer = PublicSourceImport::new(&manifest).unwrap();
    for (key, clear) in &values {
        let encoded: Vec<_> = clear
            .iter()
            .flat_map(|v| (*v as f32 / ONE as f32).to_le_bytes())
            .collect();
        for (chunk, bytes) in encoded.chunks(16384).enumerate() {
            importer.append(key, chunk * 4096, bytes).unwrap();
        }
    }
    let public = Arc::new(importer.finish().unwrap());
    // Fixture data may name a tensor; production routing never parses these keys.
    let weight = values
        .remove("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    let bias = values
        .remove("model.layers.0.self_attn.q_proj.bias")
        .unwrap();
    (Fixture { manifest, public }, weight, bias)
}

#[test]
fn creation_to_body_bias_to_original_reshape_preserves_directed_wire_and_both_components() {
    for width in [2, 34] {
        let (fixture, weight, bias) = fixture(width);
        let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
        let mut case = Case::new(&fixture, &registry);
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        assert_eq!(fixture.manifest.program, [0; 32]);
        let mut owner = program.take_source_bias(0).unwrap();
        assert!(matches!(program.take_source_bias(0), Err(Error::Retired)));
        let d = owner.description();
        assert_eq!(d.geometry, [4, width, width]);
        let Code::TensorLoop {
            program: full,
            tail,
            ..
        } = &program.recipe(d.first + 2).code
        else {
            panic!("original rescale missing")
        };
        assert_eq!(full.lanes(), d.lanes().min(128));
        assert_eq!(tail.is_some(), width == 34);
        owner.check(&program, &registry).unwrap();
        owner.assert_foreign_bindings_rejected(&program, &registry);
        let compiled =
            Compiled::prepare(Arc::clone(&program), owner, &registry, MAX_PROGRAM_BYTES).unwrap();
        // Only NOW construct independent original executable oracle.
        let expected = original(&fixture);
        assert_eq!(program.digest(), expected.digest());
        assert_eq!(program.work_budget(), expected.work_budget());
        handoff::route_tests::qualify(
            compiled,
            fresh(expected, &fixture.public, false),
            fresh(program, &fixture.public, true),
            &registry,
            &weight,
            &bias,
        );
    }
}

#[test]
fn creation_rejects_foreign_same_digest_registry_and_program_without_minting_a_replacement() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let owner = program.take_source_bias(0).unwrap();
        let copy: Manifest =
            serde_json::from_slice(&serde_json::to_vec(&case.fixture.manifest).unwrap()).unwrap();
        let registry = SourceRegistry::from_manifest(&copy).unwrap();
        assert_eq!(registry.binding(), case.registry.binding());
        assert!(matches!(
            owner.check(&program, &registry),
            Err(Error::Binding)
        ));
        let foreign = original(case.fixture);
        assert_eq!(foreign.digest(), program.digest());
        assert!(matches!(
            owner.check(&foreign, case.registry),
            Err(Error::Binding)
        ));
        drop(owner); // Failed/abandoned publication burns original invocation.
        assert_eq!(
            program
                .request_custody
                .as_ref()
                .unwrap()
                .origin
                .payload()
                .state
                .load(Ordering::Acquire),
            BURNED
        );
    });
}

#[test]
fn source_bias_preparation_one_short_retained_baseline_rejects_before_body_allocation() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let owner = program.take_source_bias(0).unwrap();
        let baseline = program.request_custody.as_ref().unwrap().retained_bytes();
        let before = handoff::constructor_entries();
        assert!(matches!(
            Compiled::prepare(Arc::clone(&program), owner, case.registry, baseline - 1),
            Err(Error::Bounds)
        ));
        assert_eq!(handoff::constructor_entries(), before);
        assert_eq!(
            program
                .request_custody
                .as_ref()
                .unwrap()
                .origin
                .payload()
                .state
                .load(Ordering::Acquire),
            BURNED
        );
    });
}

#[test]
fn source_bias_exact_peak_and_one_short_preserve_named_lifetime_cap() {
    let peak = std::cell::Cell::new(0);
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let owner = program.take_source_bias(0).unwrap();
        let compiled = Compiled::prepare(program, owner, case.registry, MAX_PROGRAM_BYTES).unwrap();
        peak.set(compiled.admitted_bytes());
    });
    for short in [false, true] {
        with_case(|mut case| {
            let done = complete(&mut case);
            let mut program = finalize(&mut case, done);
            let owner = program.take_source_bias(0).unwrap();
            let cap = peak.get() - usize::from(short);
            let result = Compiled::prepare(program, owner, case.registry, cap);
            if short {
                assert!(matches!(result, Err(Error::Bounds)));
            } else {
                assert_eq!(result.unwrap().admitted_bytes(), peak.get());
            }
        });
    }
}

#[test]
fn prior_source_work_refusal_and_unwind_burn_before_publication_without_early_refund() {
    for unwind in [false, true] {
        with_case(|mut case| {
            let done = complete(&mut case);
            let mut program = finalize(&mut case, done);
            let owner = program.take_source_bias(0).unwrap();
            let used = case.root.budget().used_bytes();
            let before = handoff::constructor_entries();
            if unwind {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    let _owner = owner;
                    panic!("creation publication unwind");
                }));
                assert!(result.is_err());
            } else {
                let custody = program.request_custody.as_ref().unwrap();
                let guard = case
                    .root
                    .charge_external(OriginalSourcePlan::compiler_metadata_control_bytes().unwrap())
                    .unwrap();
                assert!(
                    custody
                        .origin
                        .payload()
                        .source
                        .prepay_compiler_metadata(&guard, u64::MAX)
                        .is_err()
                );
                assert!(matches!(
                    Compiled::prepare(
                        Arc::clone(&program),
                        owner,
                        case.registry,
                        MAX_PROGRAM_BYTES
                    ),
                    Err(Error::Bounds)
                ));
                drop(guard);
            }
            assert_eq!(handoff::constructor_entries(), before);
            assert_eq!(case.root.budget().used_bytes(), used);
            assert_eq!(
                program
                    .request_custody
                    .as_ref()
                    .unwrap()
                    .origin
                    .payload()
                    .state
                    .load(Ordering::Acquire),
                BURNED
            );
        });
    }
}

#[test]
fn same_digest_foreign_authenticated_source_cannot_replace_original_worker_source_allocation() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let owner = program.take_source_bias(0).unwrap();
        let compiled = Compiled::prepare(
            Arc::clone(&program),
            owner,
            case.registry,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
        let foreign = Fixture::new();
        assert!(!Arc::ptr_eq(&foreign.public, &case.fixture.public));
        foreign.public.check(&program).unwrap();
        let session = [92; 32];
        let context = Context::new(session, program.digest(), 0).unwrap();
        let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
            context,
            program.work_budget().limits().unwrap(),
        )
        .unwrap();
        for runtime in runtimes {
            assert!(matches!(
                Worker::new_with_shared_source(
                    Arc::clone(&program),
                    runtime,
                    session,
                    Arc::clone(&foreign.public),
                ),
                Err(Error::Binding)
            ));
        }
        drop(compiled);
    });
}

#[test]
fn real_bias_invocation_restore_error_and_unwind_erase_outputs_before_source_owners() {
    for panic in [false, true] {
        with_case(|mut case| {
            let done = complete(&mut case);
            let mut program = finalize(&mut case, done);
            let owner = program.take_source_bias(0).unwrap();
            let compiled = Compiled::prepare(
                Arc::clone(&program),
                owner,
                case.registry,
                MAX_PROGRAM_BYTES,
            )
            .unwrap();
            handoff::route_tests::failed_restore_erases_before_owner_release(
                compiled,
                fresh(program, &case.fixture.public, true),
                case.registry,
                panic,
            );
        });
    }
}

#[test]
fn pending_original_frames_cannot_become_bias_handoff_custody() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let owner = program.take_source_bias(0).unwrap();
        let compiled = Compiled::prepare(
            Arc::clone(&program),
            owner,
            case.registry,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
        handoff::route_tests::pending_frames_and_cancellation_retire_original_attempt(
            compiled,
            fresh(program, &case.fixture.public, true),
        );
    });
}

#[test]
fn detached_creation_retains_exact_program_and_all_guards_until_native_handles_drop() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let owner = program.take_source_bias(0).unwrap();
        let weak = Arc::downgrade(&program);
        let charged = case.root.budget().used_bytes();
        drop(program);
        let retained = weak
            .upgrade()
            .expect("creation must retain original Program charge");
        owner.check(&retained, case.registry).unwrap();
        assert_eq!(case.root.budget().used_bytes(), charged);
        drop(retained);
        drop(owner);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            case.root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    });
}
