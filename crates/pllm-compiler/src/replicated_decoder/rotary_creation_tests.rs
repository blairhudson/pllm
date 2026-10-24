//! Genuine guarded source request, original rank-four graph and authenticated
//! fixture bytes. No expanded executable exists until AFTER candidate issuance.
use super::*;
use rotary_creation::{
    CreatedOperator,
    handoff::{Compiled, tests as route},
};
use source_bias::handoff::route_tests as original_route;

fn fixture(heads: usize, width: usize, bias: bool) -> Fixture {
    fixture_kind(heads, width, bias, false)
}
fn fixture_kind(heads: usize, width: usize, bias: bool, qk_norm: bool) -> Fixture {
    let hidden = heads * width;
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type":if qk_norm{"qwen3"}else{"qwen2"},"hidden_size":hidden,"intermediate_size":2,
        "num_hidden_layers":1,"num_attention_heads":heads,"num_key_value_heads":heads,
        "head_dim":width,"vocab_size":3,"tie_word_embeddings":false,
        "hidden_act":"silu","attention_bias":bias,"max_position_embeddings":64,
        "rope_theta":10000,"rms_norm_eps":0.000001,"attention_dropout":0.0,
        "use_sliding_window":false,"use_cache":true,"max_window_layers":1
    }))
    .unwrap();
    let execution = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 2,
        ..ExecutionPolicy::default()
    };
    // Shapes only, not a compiled Program or a permit. OriginalSourcePlan is
    // created by Case::new's genuine owned source/config factory afterwards.
    let shapes =
        Program::source_shapes_with_policy(&config, 4, 2, WeightMode::Public, execution).unwrap();
    let mut tensors = BTreeMap::new();
    let mut values = BTreeMap::new();
    for (key, shape) in shapes {
        let lanes = shape.iter().product();
        let clear: Vec<i64> = (0..lanes)
            .map(|i| {
                if key.contains("norm") {
                    ONE
                } else if key == "model.embed_tokens.weight" {
                    if i % hidden % 2 == 0 {
                        ONE / 4
                    } else {
                        ONE / 8
                    }
                } else if key.ends_with("q_proj.weight") || key.ends_with("k_proj.weight") {
                    if i % hidden == i / hidden {
                        if i / hidden % 2 == 0 { -1 } else { 1 }
                    } else {
                        0
                    }
                } else if key.ends_with("q_proj.bias") || key.ends_with("k_proj.bias") {
                    if i % 2 == 0 { -7 } else { 3 }
                } else {
                    0
                }
            })
            .collect();
        let digest = Sha256::digest(
            clear
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .into();
        tensors.insert(
            key.clone(),
            SourceTensor {
                shape,
                lower: *clear.iter().min().unwrap(),
                upper: *clear.iter().max().unwrap(),
                digest,
                public_q18: key.contains("norm").then(|| clear.clone()),
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
    for (key, clear) in values {
        let bytes: Vec<_> = clear
            .iter()
            .flat_map(|v| (*v as f32 / ONE as f32).to_le_bytes())
            .collect();
        for (index, chunk) in bytes.chunks(16384).enumerate() {
            importer.append(&key, index * 4096, chunk).unwrap()
        }
    }
    Fixture {
        manifest,
        public: Arc::new(importer.finish().unwrap()),
    }
}
fn slot(program: &Program, nth: usize) -> usize {
    program
        .request_custody
        .as_ref()
        .unwrap()
        .projections
        .iter()
        .enumerate()
        .filter(|(_, v)| matches!(v, Some(CreatedOperator::Rotary(_))))
        .nth(nth)
        .unwrap()
        .0
}

#[test]
fn genuine_qk_rms_lineage_keeps_original_source_recipes_work_frames_and_no_rotary_owner() {
    let fixture = fixture_kind(1, 2, false, true);
    let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
    let mut case = Case::new(&fixture, &registry);
    let done = complete(&mut case);
    let program = finalize(&mut case, done);
    assert!(
        program
            .request_custody
            .as_ref()
            .unwrap()
            .projections
            .iter()
            .all(|e| !matches!(e, Some(CreatedOperator::Rotary(_))))
    );
    let expected = original(&fixture);
    assert_eq!(program.digest(), expected.digest());
    assert_eq!(program.work_budget(), expected.work_budget());
    assert_eq!(program.logical_len(), expected.logical_len());
    let mut a = fresh(expected, &fixture.public, false);
    let mut b = fresh(program, &fixture.public, true);
    let (mut frames_a, mut frames_b) = (Vec::new(), Vec::new());
    original_route::drive(&mut a, None, &mut frames_a);
    original_route::drive(&mut b, None, &mut frames_b);
    assert!(!frames_a.is_empty());
    assert_eq!(frames_a, frames_b);
    original_route::same(&a, &b);
}

#[test]
fn unsupported_extraction_is_binding_and_eligible_missing_producer_never_falls_back() {
    let fixture = fixture_kind(1, 2, false, true);
    let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
    let mut case = Case::new(&fixture, &registry);
    let done = complete(&mut case);
    let mut program = finalize(&mut case, done);
    for index in 0..program.request_custody.as_ref().unwrap().projections.len() {
        assert!(matches!(program.take_rotary(index), Err(Error::Binding)));
    }
    with_case(|mut case| {
        let permit = case.permit.take().unwrap();
        let controller = case
            .root
            .charge_external(permit.lowering_control_bytes(limits()).unwrap())
            .unwrap();
        rotary_creation::drop_next_producer_for_test();
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
    });
}

#[test]
fn genuine_rank_four_qkv_bias_and_biasless_rotary_rejoin_original_kv_attention_and_full_wire() {
    for (heads, width, bias, nth) in [
        (1, 2, true, 0),
        (1, 34, false, 1),
        (2, 34, true, 0),
        (2, 34, false, 1),
        (1, 128, false, 0),
    ] {
        let fixture = fixture(heads, width, bias);
        let registry = SourceRegistry::from_manifest(&fixture.manifest).unwrap();
        let mut case = Case::new(&fixture, &registry);
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let index = slot(&program, nth);
        let mut owner = program.take_rotary(index).unwrap();
        assert_eq!(owner.shape(), [1, heads, 4, width]);
        assert!(program.take_rotary(index).is_err());
        owner.assert_foreign_bindings_rejected(&program, &registry);
        let compiled =
            Compiled::prepare(Arc::clone(&program), owner, &registry, MAX_PROGRAM_BYTES).unwrap();
        // Oracle material cannot participate in creation or BODY authority.
        assert_eq!(fixture.manifest.program, [0; 32]);
        let expected = original(&fixture);
        assert_eq!(program.digest(), expected.digest());
        assert_eq!(program.work_budget(), expected.work_budget());
        route::qualify(
            compiled,
            fresh(expected, &fixture.public, false),
            fresh(program, &fixture.public, true),
            &registry,
        );
    }
}

#[test]
fn foreign_registry_and_equal_digest_program_are_not_rotary_creation_authority() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let index = slot(&program, 0);
        let owner = program.take_rotary(index).unwrap();
        let copy: Manifest =
            serde_json::from_slice(&serde_json::to_vec(&case.fixture.manifest).unwrap()).unwrap();
        let foreign = SourceRegistry::from_manifest(&copy).unwrap();
        assert_eq!(foreign.binding(), case.registry.binding());
        assert!(matches!(
            owner.check(&program, &foreign),
            Err(Error::Binding)
        ));
        let other = original(case.fixture);
        assert_eq!(other.digest(), program.digest());
        assert!(matches!(
            owner.check(&other, case.registry),
            Err(Error::Binding)
        ));
        owner.check(&program, case.registry).unwrap();
    });
}

#[test]
fn wrong_family_probe_does_not_consume_or_burn_genuine_bias_owner() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        assert!(matches!(program.take_rotary(0), Err(Error::Binding)));
        let bias = program.take_source_bias(0).unwrap();
        bias.check(&program, case.registry).unwrap();
    });
}

#[test]
fn detached_rotary_owner_keeps_original_native_source_and_history_charges() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let index = slot(&program, 0);
        let owner = program.take_rotary(index).unwrap();
        let weak = Arc::downgrade(&program);
        let used = case.root.budget().used_bytes();
        drop(program);
        let retained = weak.upgrade().unwrap();
        owner.check(&retained, case.registry).unwrap();
        assert_eq!(case.root.budget().used_bytes(), used);
        drop(retained);
        drop(owner);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            case.root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
    });
}

#[test]
fn original_rotary_body_named_peak_exact_and_one_short() {
    let peak = std::cell::Cell::new(0);
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let index = slot(&program, 0);
        let owner = program.take_rotary(index).unwrap();
        peak.set(
            Compiled::prepare(program, owner, case.registry, MAX_PROGRAM_BYTES)
                .unwrap()
                .admitted_bytes(),
        );
    });
    for short in [false, true] {
        with_case(|mut case| {
            let done = complete(&mut case);
            let mut program = finalize(&mut case, done);
            let index = slot(&program, 0);
            let owner = program.take_rotary(index).unwrap();
            let result = Compiled::prepare(
                program,
                owner,
                case.registry,
                peak.get() - usize::from(short),
            );
            if short {
                assert!(matches!(result, Err(Error::Bounds)))
            } else {
                assert_eq!(result.unwrap().admitted_bytes(), peak.get())
            }
        })
    }
}

#[test]
fn retained_baseline_refusal_precedes_any_rotary_body_owner_allocation() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let index = slot(&program, 0);
        let owner = program.take_rotary(index).unwrap();
        let retained = program.request_custody.as_ref().unwrap().retained_bytes();
        assert!(matches!(
            Compiled::prepare(Arc::clone(&program), owner, case.registry, retained - 1),
            Err(Error::Bounds)
        ));
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
fn prior_original_source_metadata_refusal_cannot_enter_rotary_body() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let index = slot(&program, 0);
        let owner = program.take_rotary(index).unwrap();
        let guard = case
            .root
            .charge_external(OriginalSourcePlan::compiler_metadata_control_bytes().unwrap())
            .unwrap();
        assert!(
            program
                .request_custody
                .as_ref()
                .unwrap()
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
fn undelivered_original_frames_cannot_become_rotary_drained_custody() {
    with_case(|mut case| {
        let done = complete(&mut case);
        let mut program = finalize(&mut case, done);
        let index = slot(&program, 0);
        let owner = program.take_rotary(index).unwrap();
        let compiled = Compiled::prepare(
            Arc::clone(&program),
            owner,
            case.registry,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
        route::pending_cancel(compiled, fresh(program, &case.fixture.public, true));
    });
}

#[test]
fn real_rotary_restore_failure_and_unwind_erase_before_native_and_source_owner_release() {
    for panic in [false, true] {
        with_case(|mut case| {
            let done = complete(&mut case);
            let mut program = finalize(&mut case, done);
            let index = slot(&program, 0);
            let owner = program.take_rotary(index).unwrap();
            let compiled = Compiled::prepare(
                Arc::clone(&program),
                owner,
                case.registry,
                MAX_PROGRAM_BYTES,
            )
            .unwrap();
            route::failed_restore(
                compiled,
                fresh(program, &case.fixture.public, true),
                case.registry,
                panic,
            );
        })
    }
}
