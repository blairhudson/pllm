//! Genuine config-factory custody, generated publisher fixture, tiny D22/D27.
//! No checkpoint authentication or whole-decoder execution claim.

use super::super::*;
use super::{copies, inventory_work, operation, reset, source_work, LIMIT, WORK};
use crate::{original_source as source, plan_input::json};
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use pllm_models::{DecoderExpansionBudget, DecoderWorkload};

const RAW: &[u8] = br#"{"model_type":"qwen3","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32,"max_position_embeddings":32,"head_dim":4,"hidden_act":"silu","rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true,"attention_bias":false,"attention_dropout":0.0,"rope_scaling":null,"sliding_window":null,"use_sliding_window":false,"use_cache":true,"max_window_layers":1,"layer_types":["full_attention"]}"#;

fn workload() -> DecoderWorkload {
    DecoderWorkload {
        batch: 1,
        max_input_tokens: 1,
        max_new_tokens: 1,
    }
}

fn original(root: &OwnedBudget, bytes: &[u8]) -> OriginalSourcePlan {
    let raw = root
        .charge_external(bytes.len())
        .unwrap()
        .try_build(|_| Ok(bytes.to_vec()))
        .unwrap();
    let config = json::quote(
        raw,
        root.charge_external(json::bootstrap_bytes().unwrap())
            .unwrap(),
        json::Limits {
            max_source_bytes: 4096,
            max_nodes: 256,
            max_depth: 16,
            max_string_bytes: 1024,
            max_number_bytes: 64,
            max_number_work: 1_000_000,
            max_work: 100_000_000,
            max_payload_bytes: 4 << 20,
        },
    )
    .unwrap()
    .construct()
    .unwrap();
    let limits = source::Limits {
        expansion: DecoderExpansionBudget {
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
            max_factory_bytes: LIMIT,
        },
        max_work: 1_000_000_000_000,
        max_working_payload: LIMIT,
    };
    let bootstrap = source::bootstrap_bytes(&config, limits).unwrap();
    source::quote(
        config,
        root.charge_external(bootstrap).unwrap(),
        workload(),
        limits,
    )
    .unwrap()
    .construct()
    .unwrap()
}

fn raw_weights() -> [u8; 8] {
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&(1.0_f32 / 4096.0).to_le_bytes());
    bytes[4..].copy_from_slice(&(4095.0_f32 / 4096.0).to_le_bytes());
    bytes
}

fn request<'a>(
    plan: &'a DecoderPlan,
    values: &'a [f64],
    encoded: &'a [i64],
) -> SigmaLinearActivationRequest<'a> {
    let mut r = super::fixture_request(plan, values, encoded);
    r.selection.content_sha256 = Sha256::digest(raw_weights()).into();
    r.selection.output_coordinates = vec![0];
    r.selection.inner_coordinates = vec![0, 1];
    let source = SourceIdentity::new(
        r.selection.content_sha256,
        r.source.numeric_profile(),
        r.source.implementation(),
    );
    let edge = |e: EdgeDescriptor, shape: [usize; 2], producer| {
        let id = e.identity();
        EdgeDescriptor::new(
            EdgeIdentity::new(source, id.plan(), producer, id.wire(), 0),
            Shape::new(&shape).unwrap(),
            e.numeric(),
            e.representation(),
            e.weight_visibility(),
        )
        .unwrap()
    };
    r.source = source;
    r.input = edge(r.input, [1, 2], r.input.identity().producer());
    r.private_weights = edge(
        r.private_weights,
        [1, 2],
        sigma_weight_selection_identity(&r.selection),
    );
    r.complete_sum = edge(r.complete_sum, [1, 1], r.complete_sum.identity().producer());
    r.rescaled = edge(r.rescaled, [1, 1], r.rescaled.identity().producer());
    r.activation_output = edge(
        r.activation_output,
        [1, 1],
        r.activation_output.identity().producer(),
    );
    r.next_use = ConsumerUse::new(
        r.activation_output,
        r.next_use.consumer(),
        r.next_use.operand_position(),
        r.next_use.ordinal(),
    );
    r
}

fn declaration(plan: &DecoderPlan) -> PublisherDeclaration {
    let r = request(plan, &[], &[]);
    PublisherDeclaration {
        artifact: NamedDigest {
            id: "generated-f32-original".into(),
            digest: Digest::from_sha256(r.selection.content_sha256),
        },
        selection: r.selection,
        expected_config: plan.config_digest.clone(),
        expected_semantic_plan: plan.digest(),
        selected_bytes: 8,
        dtype: SourceDtype::F32,
        endian: SourceEndian::Little,
        tensor_offset: 128,
    }
}

fn publisher(root: &OwnedBudget, plan: &DecoderPlan) -> OwnedCharged<PublisherDeclaration> {
    // Genuine test frontend: charge BEFORE strings/vectors are constructed.
    let owner = root
        .charge_external(16 << 10)
        .unwrap()
        .try_build(|_| Ok(declaration(plan)))
        .unwrap();
    assert!(owner.payload().backing_bytes().unwrap() <= owner.charged_bytes());
    owner
}

fn limits() -> SourceInputLimits {
    SourceInputLimits {
        inventory_work: WORK,
        max_additional_work: WORK * 2,
    }
}

fn build(root: &OwnedBudget) -> OwnedSigmaInput {
    let source = original(root, RAW);
    let publisher = publisher(root, source.plan());
    prepare_from_original_source(
        source,
        publisher,
        source_work(root),
        inventory_work(root),
        limits(),
    )
    .unwrap()
}

fn packet<'a>(selection: &'a CheckpointTensorSelection, bytes: &'a [u8]) -> SigmaSourceBytes<'a> {
    SigmaSourceBytes {
        artifact_name: "generated-f32-original",
        selection,
        dtype: SourceDtype::F32,
        endian: SourceEndian::Little,
        tensor_offset: 128,
        scalar_artifact_offsets: &[128, 132],
        bytes,
    }
}

fn issue(
    root: &OwnedBudget,
) -> (
    OwnedSigmaExecutionMask,
    OwnedSigmaExecutionRun,
    OwnedSigmaInputRetention,
    typed::OwnedLocalDealer,
) {
    let input = build(root);
    // Independent normal byte-factory oracle, never serialized/reimported.
    let oracle = pllm_models::lower_model_json(RAW, workload()).unwrap();
    let values = [1.0 / 4096.0, 4095.0 / 4096.0];
    let r = request(&oracle, &values, &[1, 4095]);
    let selection = r.selection.clone();
    let mut dealer =
        typed::OwnedLocalDealer::new(r.source, r.input.identity().plan(), [72; 32], root).unwrap();
    let (mask, run, retention) = input
        .execute_retained(
            packet(&selection, &raw_weights()),
            operation(r),
            root,
            &mut dealer,
            &[-1, -1],
        )
        .unwrap();
    (mask, run, retention, dealer)
}

#[test]
fn source_plan_is_moved_unchanged_without_json_roundtrip_or_guard_replacement() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let initial = root.budget().used_bytes();
    let source = original(&root, RAW);
    let plan_pointer = source.plan() as *const _;
    let raw_pointer = source.raw_configuration().as_ptr();
    let source_resources = source.resources();
    let publisher = publisher(&root, source.plan());
    let before = crate::plan_input::allocation_factories_for_test();
    let input = prepare_from_original_source(
        source,
        publisher,
        source_work(&root),
        inventory_work(&root),
        limits(),
    )
    .unwrap();
    assert_eq!(crate::plan_input::allocation_factories_for_test(), before);
    assert_eq!(input.semantic.plan() as *const _, plan_pointer);
    let OriginalPlanOwner::Source(source) = &input.semantic.input else {
        panic!("lost source owner")
    };
    assert_eq!(source.raw_configuration().as_ptr(), raw_pointer);
    assert_eq!(
        source.resources().prepaid_work,
        source_resources.prepaid_work
    );
    assert_eq!(
        source.resources().working_charge_bytes,
        source_resources.working_charge_bytes
    );
    assert!(input.is_budget(root.budget()));
    let work = input.preparation_work().unwrap();
    assert_eq!(work.parse_prepaid(), 0); // no parser invoked in this route
    assert_eq!(
        work.total_prepaid(),
        work.inventory_prepaid() + work.metadata_prepaid()
    );
    assert!(source_resources.prepaid_work > 0); // source factory work was NOT zeroed
    let oracle = pllm_models::lower_model_json(RAW, workload()).unwrap();
    assert_eq!(input.semantic.plan(), &oracle);
    assert_eq!(
        input.semantic.inventory().canonical_bytes(),
        lower_decoder_matrix_inventory(&oracle)
            .unwrap()
            .canonical_bytes()
    );
    drop(input);
    assert_eq!(root.budget().used_bytes(), initial);
}

#[test]
fn source_preparation_exact_and_one_short_work_refuse_before_inventory_or_copies() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let input = build(&root);
    let exact = input.preparation_work().unwrap().total_prepaid();
    drop(input);
    for short in [false, true] {
        let source = original(&root, RAW);
        let publisher = publisher(&root, source.plan());
        reset(None, None);
        let before = crate::plan_input::allocation_factories_for_test();
        let result = prepare_from_original_source(
            source,
            publisher,
            source_work(&root),
            inventory_work(&root),
            SourceInputLimits {
                max_additional_work: exact - u128::from(short),
                ..limits()
            },
        );
        assert_eq!(crate::plan_input::allocation_factories_for_test(), before);
        if short {
            assert!(matches!(result, Err(InputError::Work)));
            assert_eq!(copies(), 0);
        } else {
            assert_eq!(
                result
                    .as_ref()
                    .unwrap()
                    .preparation_work()
                    .unwrap()
                    .total_prepaid(),
                exact
            );
        }
    }
}

#[test]
fn source_preparation_actual_counter_exact_and_one_short_precede_metadata_factories() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let input = build(&root);
    let exact = root.budget().used_bytes();
    drop((input, root));
    for short in [false, true] {
        let root = OwnedBudget::new(exact - usize::from(short)).unwrap();
        let baseline = root.budget().used_bytes();
        let source = original(&root, RAW);
        let publisher = publisher(&root, source.plan());
        reset(None, None);
        let result = prepare_from_original_source(
            source,
            publisher,
            source_work(&root),
            inventory_work(&root),
            limits(),
        );
        if short {
            assert!(result.is_err());
            assert_eq!(copies(), 0);
        } else {
            assert!(result.is_ok());
            assert_eq!(root.budget().used_bytes(), exact);
        }
        drop(result);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn wrong_original_counter_and_bootstrap_refuse_without_cloning_or_json_upgrade() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let foreign = OwnedBudget::new(LIMIT).unwrap();
    for wrong in 0..4 {
        let initial = root.budget().used_bytes();
        let source = original(&root, RAW);
        let publisher = publisher(if wrong == 0 { &foreign } else { &root }, source.plan());
        let source_work = if wrong == 1 {
            source_work(&foreign)
        } else {
            root.charge_external(bootstrap_bytes().unwrap() - usize::from(wrong == 3))
                .unwrap()
        };
        let inventory = inventory_work(if wrong == 2 { &foreign } else { &root });
        reset(None, None);
        let result =
            prepare_from_original_source(source, publisher, source_work, inventory, limits());
        assert!(matches!(
            result,
            Err(InputError::BudgetIdentity | InputError::Resource)
        ));
        assert_eq!(copies(), 0);
        assert_eq!(root.budget().used_bytes(), initial);
    }
}

#[test]
fn different_actual_factory_source_does_not_inherit_publisher_selection() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let first = original(&root, RAW);
    let publisher = publisher(&root, first.plan());
    let changed = std::str::from_utf8(RAW)
        .unwrap()
        .replace("\"vocab_size\":32", "\"vocab_size\":33");
    let foreign = original(&root, changed.as_bytes());
    assert!(!first.same_origin(&foreign));
    reset(None, None);
    assert!(matches!(
        prepare_from_original_source(
            foreign,
            publisher,
            source_work(&root),
            inventory_work(&root),
            limits()
        ),
        Err(InputError::PublisherIdentity)
    ));
    assert_eq!(copies(), 0);
}

#[test]
fn unchanged_source_and_coefficient_verifiers_reject_bad_supplied_weights() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    for corrupt in [false, true] {
        let input = build(&root);
        let oracle = pllm_models::lower_model_json(RAW, workload()).unwrap();
        let values = [1.0 / 4096.0, 4095.0 / 4096.0];
        let encoded = if corrupt { [1, 4095] } else { [1, 4094] };
        let r = request(&oracle, &values, &encoded);
        let selection = r.selection.clone();
        let mut dealer =
            typed::OwnedLocalDealer::new(r.source, r.input.identity().plan(), [72; 32], &root)
                .unwrap();
        let mut raw = raw_weights();
        if corrupt {
            raw[0] ^= 1;
        }
        assert!(matches!(
            input.execute_retained(
                packet(&selection, &raw),
                operation(r),
                &root,
                &mut dealer,
                &[-1, -1]
            ),
            Err(InputError::Execution(SigmaExecutionError::Source(_)))
        ));
    }
}

#[test]
fn source_copy_failure_and_unwind_retire_genuine_original_guards() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let baseline = root.budget().used_bytes();
    for panic in [false, true] {
        let source = original(&root, RAW);
        let publisher = publisher(&root, source.plan());
        reset(
            if panic { None } else { Some(0) },
            if panic { Some(0) } else { None },
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prepare_from_original_source(
                source,
                publisher,
                source_work(&root),
                inventory_work(&root),
                limits(),
            )
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        assert_eq!(root.budget().used_bytes(), baseline);
    }
    reset(None, None);
}

#[test]
fn source_owned_d22_schedule_matches_independent_normal_facade() {
    fn trace(run: &mut OwnedSigmaExecutionRun) -> (Vec<[usize; 2]>, [EdgeDescriptor; 2]) {
        let mut frames = run.begin(&mut || false).unwrap();
        let mut schedule = Vec::new();
        for _ in 0..11 {
            schedule.push([frames[0].as_bytes().len(), frames[1].as_bytes().len()]);
            // Actual opaque peer bytes, not reconstructed/replaced packets.
            match run
                .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
                .unwrap()
            {
                OwnedSigmaExecutionStep::Frames(next) => frames = next,
                OwnedSigmaExecutionStep::Output(outputs) => {
                    return (schedule, [outputs[0].descriptor(), outputs[1].descriptor()])
                }
            }
        }
        panic!("original D22 exchange count exceeded");
    }
    let root = OwnedBudget::new(LIMIT).unwrap();
    let other = OwnedBudget::new(LIMIT).unwrap();
    let actual = {
        // Original singleton issuance permit must retire before the oracle run.
        let (_mask, mut run, _custody, _dealer) = issue(&root);
        trace(&mut run)
    };
    let oracle = pllm_models::lower_model_json(RAW, workload()).unwrap();
    let inventory = lower_decoder_matrix_inventory(&oracle).unwrap();
    let values = [1.0 / 4096.0, 4095.0 / 4096.0];
    let r = request(&oracle, &values, &[1, 4095]);
    let selection = r.selection.clone();
    let p = declaration(&oracle);
    let pin = ExpectedSigmaSourcePin::from_trusted_publisher(
        p.artifact,
        p.selected_bytes,
        p.selection,
        p.dtype,
        p.endian,
        p.tensor_offset,
        &p.expected_config,
        &p.expected_semantic_plan,
    )
    .unwrap();
    let mut dealer =
        typed::OwnedLocalDealer::new(r.source, r.input.identity().plan(), [72; 32], &other)
            .unwrap();
    let (_expected_mask, mut expected) = execute_owned_sigma_linear_activation(
        pin,
        packet(&selection, &raw_weights()),
        r,
        &inventory,
        &other,
        &mut dealer,
        &[-1, -1],
    )
    .unwrap();
    // Fresh native entropy makes independent masked packets intentionally
    // unequal. Exact public framing/phase order and final edge identities agree.
    assert_eq!(actual, trace(&mut expected));
}

#[test]
fn genuine_source_root_raw_value_guard_survives_d22_frames_and_last_output() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let baseline = root.budget().used_bytes();
    let (mask, mut run, residue, dealer) = issue(&root);
    let weak = Arc::downgrade(&residue.inner);
    let raw_pointer = match &residue.inner.semantic.input {
        OriginalPlanOwner::Source(s) => s.raw_configuration().as_ptr(),
        _ => panic!("wrong source"),
    };
    drop(residue); // no caller-side retention convention can supply this guarantee
    assert!(weak.upgrade().is_some());
    let mut frames = run.begin(&mut || false).unwrap();
    let mut prior = Vec::new();
    let mut outputs = None;
    for _ in 0..11 {
        let step = run
            .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
            .unwrap();
        prior.push(frames);
        match step {
            OwnedSigmaExecutionStep::Frames(next) => frames = next,
            OwnedSigmaExecutionStep::Output(out) => {
                outputs = Some(out);
                break;
            }
        }
    }
    let outputs = outputs.expect("original D22 exchange count exceeded");
    drop((run, mask, prior, dealer));
    let custody = weak.upgrade().unwrap();
    let OriginalPlanOwner::Source(source) = &custody.semantic.input else {
        panic!("lost source")
    };
    assert_eq!(source.raw_configuration().as_ptr(), raw_pointer);
    assert_eq!(source.raw_configuration(), RAW);
    drop(custody);
    let [left, right] = outputs;
    drop(left);
    assert!(weak.upgrade().is_some());
    drop(right);
    assert!(weak.upgrade().is_none());
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn genuine_source_survives_d27_role_split_frames_outputs_and_cancellation() {
    for cancel in [false, true] {
        let root = OwnedBudget::new(LIMIT).unwrap();
        let baseline = root.budget().used_bytes();
        let (mask, run, residue, dealer) = issue(&root);
        let weak = Arc::downgrade(&residue.inner);
        let (mut coordinator, [mut left, mut right]) = run.into_roles(&root).unwrap();
        drop(residue);
        let mut frames = [
            left.begin(&mut || false).unwrap(),
            right.begin(&mut || false).unwrap(),
        ];
        let mut prior = Vec::new();
        if cancel {
            let peer = frames[1].as_bytes().unwrap();
            coordinator.cancel();
            assert!(left.advance(peer, &mut || false).is_err());
            assert!(weak.upgrade().is_some());
            drop((left, right, coordinator, frames, mask, dealer));
        } else {
            let mut ready = false;
            for _ in 0..11 {
                let a = left
                    .advance(frames[1].as_bytes().unwrap(), &mut || false)
                    .unwrap();
                let b = right
                    .advance(frames[0].as_bytes().unwrap(), &mut || false)
                    .unwrap();
                prior.push(frames);
                match (a, b) {
                    (
                        OwnedSigmaExecutionRoleStep::Frame(a),
                        OwnedSigmaExecutionRoleStep::Frame(b),
                    ) => frames = [a, b],
                    (OwnedSigmaExecutionRoleStep::Ready, OwnedSigmaExecutionRoleStep::Ready) => {
                        ready = true;
                        break;
                    }
                    _ => panic!("different role schedules"),
                }
            }
            assert!(ready, "original D27 exchange count exceeded");
            let outputs = [left.finish().unwrap(), right.finish().unwrap()];
            assert!(weak.upgrade().is_some());
            drop((mask, dealer, prior));
            assert!(weak.upgrade().is_some());
            drop((outputs, coordinator));
        }
        assert!(weak.upgrade().is_none());
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}
