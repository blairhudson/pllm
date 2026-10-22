//! External D27 acceptance through the source-certified public compiler facade.
//!
//! Reuses sigma_source_facade's pinned BF16 submatrix, source-shaped semantic
//! declaration and independent integer encoding. This is local conditional
//! execution, not decoded parity, original-layer execution, SDK availability,
//! authenticated transport or publisher authentication. No raw typed output or
//! fresh-material bypass is used. Public budget counters observe retained
//! accounting, not pointer identity, erased bytes, entropy or transient updates;
//! the corresponding private native/compiler tests own those stronger checks.

#[path = "../../pllm-garble/tests/sigma_linear_activation/fixture.rs"]
mod fixture;

use pllm_compiler::{lower_decoder_matrix_inventory, sigma_linear_activation::*};
use pllm_core::edge_contract::{
    ConsumerUse, EdgeDescriptor, EdgeIdentity, Interpretation, Numeric, Representation, Shape,
    SourceIdentity, WeightVisibility,
};
use pllm_garble::{
    masked_wire_reference::typed,
    shared_allocation::{Budget, ExternalCharge},
};
use pllm_models::{lower_model_json, DecoderMode, DecoderPlan, DecoderWorkload, ModelOperator};
use pllm_types::{digest_bytes, Digest, NamedDigest};
use std::{mem::size_of, sync::Mutex};

const TENSOR_OFFSET: u64 = 281_019_680;
const LIMIT: usize = 64 << 20;
const PROVENANCE: SigmaExecutionProvenance =
    SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution;
// The matrix singleton is process-local; every test that issues native material
// in this external binary uses this lock. Other integration binaries are separate
// processes, not sibling owners of this test's singleton.
static SERIAL: Mutex<()> = Mutex::new(());

fn plan() -> DecoderPlan {
    lower_model_json(
        br#"{
            "model_type":"qwen2","hidden_size":896,"intermediate_size":4864,
            "num_hidden_layers":24,"num_attention_heads":14,"num_key_value_heads":2,
            "vocab_size":151936,"max_position_embeddings":32768,"hidden_act":"silu",
            "rms_norm_eps":1e-6,"rope_theta":1000000.0,"tie_word_embeddings":true
        }"#,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 1,
        },
    )
    .unwrap()
}

fn selected_digest() -> [u8; 32] {
    let digest: Digest =
        serde_json::from_value(serde_json::json!(fixture::SELECTED_SHA256)).unwrap();
    digest.as_sha256()
}

fn request<'a>(
    plan: &'a DecoderPlan,
    values: &'a [f64],
    encoded: &'a [i64],
) -> SigmaLinearActivationRequest<'a> {
    let graph = &plan.prefill;
    let linear = graph
        .operations
        .iter()
        .find(|op| {
            op.operator == ModelOperator::Linear && op.attributes["weight"] == fixture::TENSOR
        })
        .unwrap();
    let input = graph
        .operations
        .iter()
        .find(|op| op.id == linear.inputs[0])
        .unwrap();
    let activation = graph
        .operations
        .iter()
        .find(|op| op.operator == ModelOperator::Silu && op.inputs == [linear.id.clone()])
        .unwrap();
    let next = graph
        .operations
        .iter()
        .find(|op| op.inputs.contains(&activation.id))
        .unwrap();
    let selection = CheckpointTensorSelection {
        checkpoint: fixture::REPOSITORY.into(),
        immutable_revision: fixture::REVISION.into(),
        content_sha256: selected_digest(),
        tensor: fixture::TENSOR.into(),
        original_weight_shape: fixture::SOURCE_SHAPE_OI,
        weight_layout: "output_input_row_major".into(),
        output_coordinates: vec![0],
        inner_coordinates: (0..fixture::INNER).collect(),
        input_tensor: input.id.clone(),
        original_input_shape: [1, 896],
        input_layout: "row_inner_row_major".into(),
        input_row_coordinates: vec![0],
        encoding: SIGMA_LINEAR_ACTIVATION_ENCODING.into(),
    };
    let source = SourceIdentity::new(
        selected_digest(),
        digest_bytes(
            "pllm.test.sigma-d22.local-profile.v1",
            b"conditional-ring50-q12",
        )
        .as_sha256(),
        typed::approved_implementation(),
    );
    let plan_id = sigma_semantic_plan_identity(plan, DecoderMode::Prefill);
    let edge = |producer, dimensions: &[usize], fractional, visibility, wire| {
        EdgeDescriptor::new(
            EdgeIdentity::new(source, plan_id, producer, [wire; 32], 0),
            Shape::new(dimensions).unwrap(),
            Numeric::new(50, fractional, Interpretation::Signed).unwrap(),
            Representation::CommonAdditiveMasked,
            visibility,
        )
        .unwrap()
    };
    let activation_output = edge(
        sigma_semantic_operation_identity(activation),
        &[1, 1],
        12,
        None,
        5,
    );
    SigmaLinearActivationRequest {
        plan,
        mode: DecoderMode::Prefill,
        linear_operation: &linear.id,
        activation_operation: &activation.id,
        next_operation: &next.id,
        input: edge(
            sigma_semantic_operation_identity(input),
            &[1, fixture::INNER],
            12,
            None,
            1,
        ),
        private_weights: edge(
            sigma_weight_selection_identity(&selection),
            &[1, fixture::INNER],
            12,
            Some(WeightVisibility::Private),
            2,
        ),
        complete_sum: edge(
            sigma_semantic_operation_identity(linear),
            &[1, 1],
            24,
            None,
            3,
        ),
        rescaled: edge(
            sigma_gapars12_operation_identity(linear),
            &[1, 1],
            12,
            None,
            4,
        ),
        activation_output,
        rescale_operation: sigma_gapars12_operation_identity(linear),
        next_use: ConsumerUse::new(
            activation_output,
            sigma_semantic_operation_identity(next),
            next.inputs
                .iter()
                .position(|id| id == &activation.id)
                .unwrap() as u16,
            0,
        ),
        selection,
        source,
        input_domain: IntegerInterval::from_half_open(-4096, 4097).unwrap(),
        source_coefficients: values,
        encoded_private_coefficients: encoded,
    }
}

fn pin(plan: &DecoderPlan) -> ExpectedSigmaSourcePin {
    ExpectedSigmaSourcePin::from_trusted_publisher(
        NamedDigest {
            id: "model.safetensors".into(),
            digest: Digest::from_sha256(selected_digest()),
        },
        512,
        request(plan, &[], &[]).selection,
        SourceDtype::Bf16,
        SourceEndian::Little,
        TENSOR_OFFSET,
        &plan.config_digest,
        &plan.digest(),
    )
    .unwrap()
}

fn offsets() -> Vec<u64> {
    (0..fixture::INNER)
        .map(|i| TENSOR_OFFSET + 2 * i as u64)
        .collect()
}

fn packet<'a>(
    selection: &'a CheckpointTensorSelection,
    offsets: &'a [u64],
    bytes: &'a [u8],
) -> SigmaSourceBytes<'a> {
    SigmaSourceBytes {
        artifact_name: "model.safetensors",
        selection,
        dtype: SourceDtype::Bf16,
        endian: SourceEndian::Little,
        tensor_offset: TENSOR_OFFSET,
        scalar_artifact_offsets: offsets,
        bytes,
    }
}

fn coefficients() -> Vec<i64> {
    // Independent integer BF16 -> floor-Q12, identical fixture policy to the
    // existing external source test; never ask production encoding for an oracle.
    fixture::source_bytes()
        .chunks_exact(2)
        .map(|bytes| {
            let word = u16::from_le_bytes([bytes[0], bytes[1]]);
            let exponent = (word >> 7) & 255;
            assert_ne!(exponent, 255);
            let mantissa = i128::from(word & 127) + if exponent == 0 { 0 } else { 128 };
            let signed = if word & 0x8000 == 0 {
                mantissa
            } else {
                -mantissa
            };
            let shift = i32::from(exponent.max(1)) - 127 - 7 + 12;
            let encoded = if shift >= 0 {
                signed << shift
            } else if -shift >= 127 {
                if signed < 0 {
                    -1
                } else {
                    0
                }
            } else {
                signed >> -shift
            };
            i64::try_from(encoded).unwrap()
        })
        .collect()
}

fn dealer(budget: &Budget) -> typed::LocalDealer<'_> {
    let p = plan();
    let r = request(&p, &[], &[]);
    typed::LocalDealer::new(r.source, r.input.identity().plan(), [107; 32], budget).unwrap()
}

struct Issued<'b> {
    mask: SigmaExecutionMask<'b>,
    run: SigmaExecutionRun<'b>,
    output: EdgeDescriptor,
    source_floor: usize,
}

fn issue<'b>(
    budget: &'b Budget,
    dealer: &mut typed::LocalDealer<'b>,
) -> Result<Issued<'b>, SigmaExecutionError> {
    // All borrowed caller source/model/issuance/input buffers retire BEFORE any
    // role split or exchange. Only the original admitted compiler/native owners
    // may retain them; no test-side source keepalive substitutes for those owners.
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let bytes = fixture::source_bytes();
    let values = fixture::weights_io();
    let encoded = coefficients();
    let input: Vec<_> = fixture::input()
        .into_iter()
        .map(|x| (x * 4096.0) as i64)
        .collect();
    let r = request(&p, &values, &encoded);
    let output = r.activation_output;
    let expected = pin(&p);
    let source_floor = expected.certification_retained_quote(&r).unwrap()
        + expected
            .import_quote()
            .unwrap()
            .import_encoding_scratch_bytes
        + sigma_execution_compiler_preflight(&p, &inventory).unwrap();
    let selection = r.selection.clone();
    let (mask, run) = execute_sigma_linear_activation(
        expected,
        packet(&selection, &offsets(), &bytes),
        r,
        &inventory,
        budget,
        dealer,
        &input,
    )?;
    Ok(Issued {
        mask,
        run,
        output,
        source_floor,
    })
}

fn assert_source_charged(budget: &Budget, baseline: usize, source_floor: usize) {
    assert!(source_floor > 512);
    assert!(budget.used_bytes() >= baseline + source_floor);
}

fn assert_split_source_charged(budget: &Budget, baseline: usize, source_floor: usize) {
    // The new role reservation is NOT evidence for retaining the old source
    // guard: subtract its entire separately known charge before checking the
    // original source/compiler admission lower bound. Native private tests own
    // pointer identity and drop-order observations unavailable through this API.
    let new_charge = sigma_execution_role_split_resources()
        .unwrap()
        .additional_payload_bytes
        + size_of::<ExternalCharge<'static>>();
    assert_source_charged(budget, baseline + new_charge, source_floor);
}

fn ready<'b>(roles: &mut [SigmaExecutionRole<'b>; 2]) -> Vec<[SigmaExecutionRoleFrame<'b>; 2]> {
    let [left, right] = roles;
    let mut frames = [
        left.begin(&mut || false).unwrap(),
        right.begin(&mut || false).unwrap(),
    ];
    let mut retained = Vec::new();
    for round in 0..11 {
        let a = left
            .advance(frames[1].as_bytes().unwrap(), &mut || false)
            .unwrap();
        let b = right
            .advance(frames[0].as_bytes().unwrap(), &mut || false)
            .unwrap();
        retained.push(frames);
        match (a, b) {
            (SigmaExecutionRoleStep::Frame(a), SigmaExecutionRoleStep::Frame(b)) => {
                assert!(round < 10);
                frames = [a, b];
            }
            (SigmaExecutionRoleStep::Ready, SigmaExecutionRoleStep::Ready) => {
                assert_eq!(round, 10);
                return retained;
            }
            _ => panic!("independent role schedules diverged"),
        }
    }
    panic!("missing two-role final barrier")
}

#[test]
fn public_certified_source_roles_finish_eleven_exchanges_and_outputs_retain_source() {
    let _serial = SERIAL.lock().unwrap();
    let budget = Budget::new(LIMIT).unwrap();
    let mut dealer = dealer(&budget);
    let baseline = budget.used_bytes();
    let Issued {
        mask,
        run,
        output,
        source_floor,
    } = issue(&budget, &mut dealer).unwrap();
    let before = budget.used_bytes();
    let quote = sigma_execution_role_split_resources().unwrap();
    let (coordinator, mut roles) = run.into_roles(&budget).unwrap();
    assert_eq!(
        budget.used_bytes(),
        before + quote.additional_payload_bytes + size_of::<ExternalCharge<'static>>()
    );
    assert_eq!(coordinator.provenance(), PROVENANCE);
    assert_eq!(roles[0].role(), 0);
    assert_eq!(roles[1].role(), 1);
    assert_eq!(roles[0].provenance(), PROVENANCE);
    assert_eq!(roles[1].provenance(), PROVENANCE);
    let frames = ready(&mut roles);
    assert_eq!(frames.len(), 11);
    let [left, right] = roles;
    let outputs = [left.finish().unwrap(), right.finish().unwrap()];
    assert!(!coordinator.is_aborted());
    for (role, value) in outputs.iter().enumerate() {
        assert_eq!(value.role(), role as u8);
        assert_eq!(value.descriptor().unwrap(), output);
        assert_eq!(
            value.descriptor().unwrap().identity().source().checkpoint(),
            selected_digest()
        );
        assert_eq!(value.provenance(), PROVENANCE);
    }
    drop((frames, mask));
    assert_split_source_charged(&budget, baseline, source_floor);
    // Coordinator drop revokes native output access, not compiler source
    // retention. Escaped opaque outputs still own that same source lifetime.
    drop(coordinator);
    for value in &outputs {
        assert_eq!(
            value.descriptor(),
            Err(SigmaExecutionError::Native(typed::Error::Consumed))
        );
    }
    assert_split_source_charged(&budget, baseline, source_floor);
    let [left, right] = outputs;
    drop(left);
    assert_split_source_charged(&budget, baseline, source_floor);
    drop(right);
    assert_eq!(budget.used_bytes(), baseline);
    drop(dealer);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn coordinator_or_retained_frame_alone_preserves_source_and_real_singleton() {
    let _serial = SERIAL.lock().unwrap();
    let budget = Budget::new(LIMIT).unwrap();
    let mut dealer = dealer(&budget);
    let baseline = budget.used_bytes();
    // No frame keepalive: coordinator alone retains the checked source, even
    // after role drops have synchronously erased both native continuations.
    let Issued {
        mask,
        run,
        source_floor,
        ..
    } = issue(&budget, &mut dealer).unwrap();
    let (coordinator, roles) = run.into_roles(&budget).unwrap();
    drop((roles, mask));
    assert!(coordinator.is_aborted());
    assert_split_source_charged(&budget, baseline, source_floor);
    drop(coordinator);
    assert_eq!(budget.used_bytes(), baseline);

    let Issued {
        mask,
        run,
        source_floor,
        ..
    } = issue(&budget, &mut dealer).unwrap();
    let (coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
    let a = left.begin(&mut || false).unwrap();
    let b = right.begin(&mut || false).unwrap();
    assert!(left.advance(&[], &mut || false).is_err());
    assert!(coordinator.is_aborted());
    assert!(right.begin(&mut || false).is_err());
    drop((left, right, coordinator, mask));
    assert!(a.as_bytes().is_err());
    assert!(b.as_bytes().is_err());
    assert_split_source_charged(&budget, baseline, source_floor);
    let held = budget.used_bytes();
    assert!(matches!(
        issue(&budget, &mut dealer),
        Err(SigmaExecutionError::Native(typed::Error::Resource))
    ));
    assert_eq!(budget.used_bytes(), held);
    drop(a);
    assert_split_source_charged(&budget, baseline, source_floor);
    let held = budget.used_bytes();
    assert!(matches!(
        issue(&budget, &mut dealer),
        Err(SigmaExecutionError::Native(typed::Error::Resource))
    ));
    assert_eq!(budget.used_bytes(), held);
    drop(b);
    assert_eq!(budget.used_bytes(), baseline);
    // Prove guard retirement through the SAME certified-source facade, not an
    // arbitrary fresh native issuer or raw-owner reconstruction.
    drop(issue(&budget, &mut dealer).unwrap());
    assert_eq!(budget.used_bytes(), baseline);
}

#[test]
fn public_two_role_barrier_and_escaped_output_revoke_on_sibling_replay() {
    let _serial = SERIAL.lock().unwrap();
    let budget = Budget::new(LIMIT).unwrap();
    let mut dealer = dealer(&budget);
    let baseline = budget.used_bytes();
    for malformed_final_peer in [false, true] {
        let Issued {
            mask,
            run,
            source_floor,
            ..
        } = issue(&budget, &mut dealer).unwrap();
        let (coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
        let mut a = left.begin(&mut || false).unwrap();
        let mut b = right.begin(&mut || false).unwrap();
        for round in 0..11 {
            let next_a = left.advance(b.as_bytes().unwrap(), &mut || false).unwrap();
            if round == 10 {
                assert!(matches!(next_a, SigmaExecutionRoleStep::Ready));
                if malformed_final_peer {
                    assert!(right.advance(&[], &mut || false).is_err());
                    assert!(left.finish().is_err());
                } else {
                    // Locally ready does NOT release output while peer still
                    // awaits its own final reconstruction frame.
                    assert!(left.finish().is_err());
                    assert!(a.as_bytes().is_err());
                    assert!(right.advance(&[], &mut || false).is_err());
                }
                assert!(coordinator.is_aborted());
                break;
            }
            let next_b = right.advance(a.as_bytes().unwrap(), &mut || false).unwrap();
            match (next_a, next_b) {
                (SigmaExecutionRoleStep::Frame(x), SigmaExecutionRoleStep::Frame(y)) => {
                    a = x;
                    b = y;
                }
                _ => panic!("early ready"),
            }
        }
        drop((right, coordinator, mask));
        assert_split_source_charged(&budget, baseline, source_floor);
        drop((a, b));
        assert_eq!(budget.used_bytes(), baseline);
    }

    let Issued {
        mask,
        run,
        output,
        source_floor,
    } = issue(&budget, &mut dealer).unwrap();
    let (coordinator, mut roles) = run.into_roles(&budget).unwrap();
    let frames = ready(&mut roles);
    let [left, mut right] = roles;
    let escaped = left.finish().unwrap();
    assert_eq!(escaped.descriptor().unwrap(), output);
    assert!(right.advance(&[], &mut || false).is_err());
    assert!(coordinator.is_aborted());
    assert!(right.finish().is_err());
    assert_eq!(
        escaped.descriptor(),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    );
    drop((frames, coordinator, mask));
    assert_split_source_charged(&budget, baseline, source_floor);
    drop(escaped);
    assert_eq!(budget.used_bytes(), baseline);
}

#[test]
fn public_roles_reject_wrong_role_replay_and_same_source_foreign_issuance() {
    let _serial = SERIAL.lock().unwrap();
    let budget = Budget::new(LIMIT).unwrap();
    let mut dealer = dealer(&budget);
    let baseline = budget.used_bytes();
    let stale_peer = {
        let Issued { mask, run, .. } = issue(&budget, &mut dealer).unwrap();
        let (coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
        let a = left.begin(&mut || false).unwrap();
        let b = right.begin(&mut || false).unwrap();
        let stale = b.as_bytes().unwrap().to_vec();
        drop((left, right, coordinator, mask, a, b));
        stale
    };
    assert_eq!(budget.used_bytes(), baseline);
    for case in 0..4 {
        let Issued {
            mask,
            run,
            source_floor,
            ..
        } = issue(&budget, &mut dealer).unwrap();
        let (coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
        let a = left.begin(&mut || false).unwrap();
        let b = right.begin(&mut || false).unwrap();
        if case == 3 {
            drop(left);
        } else {
            match case {
                0 => assert!(left.advance(a.as_bytes().unwrap(), &mut || false).is_err()),
                1 => {
                    assert!(left.advance(b.as_bytes().unwrap(), &mut || false).is_ok());
                    assert!(left.advance(b.as_bytes().unwrap(), &mut || false).is_err());
                }
                2 => assert!(left.advance(&stale_peer, &mut || false).is_err()),
                _ => unreachable!(),
            }
            drop(left);
        }
        assert!(coordinator.is_aborted());
        assert!(right.advance(&[], &mut || false).is_err());
        assert!(right.finish().is_err());
        assert!(a.as_bytes().is_err());
        assert!(b.as_bytes().is_err());
        drop((coordinator, mask));
        assert_split_source_charged(&budget, baseline, source_floor);
        drop((a, b));
        assert_eq!(budget.used_bytes(), baseline);
    }
}

#[test]
fn public_split_exact_fit_one_short_wrong_budget_and_nonfresh_consumption() {
    let _serial = SERIAL.lock().unwrap();
    let q = sigma_execution_role_split_resources().unwrap();
    assert_eq!(
        q.additional_payload_bytes,
        q.native_reservation_bytes
            + q.retention_backing_bytes
            + q.wrapper_population_bytes
            + q.transition_bytes
    );
    for short in [true, false] {
        let budget = Budget::new(LIMIT).unwrap();
        let mut dealer = dealer(&budget);
        let baseline = budget.used_bytes();
        let Issued {
            mask,
            run,
            source_floor,
            ..
        } = issue(&budget, &mut dealer).unwrap();
        let additional = q.additional_payload_bytes + size_of::<ExternalCharge<'static>>();
        let filler_bytes = LIMIT - budget.used_bytes() - additional + usize::from(short);
        let filler = budget
            .charge_external(filler_bytes - size_of::<ExternalCharge<'static>>())
            .unwrap();
        let before = budget.used_bytes();
        let result = run.into_roles(&budget);
        if short {
            assert_eq!(before + additional, LIMIT + 1);
            assert!(matches!(
                result,
                Err(SigmaExecutionError::Native(typed::Error::Resource))
            ));
            // Failed split consumed native run but mask still holds source.
            drop(filler);
            assert_source_charged(&budget, baseline, source_floor);
            drop(mask);
        } else {
            let (coordinator, roles) = result.unwrap();
            assert_eq!(budget.used_bytes(), LIMIT);
            assert_eq!(budget.used_bytes(), before + additional);
            assert!(budget.charge_external(0).is_err());
            drop((roles, coordinator, mask, filler));
        }
        assert_eq!(budget.used_bytes(), baseline);
        drop(dealer);
        assert_eq!(budget.used_bytes(), 0);
    }

    let budget = Budget::new(LIMIT).unwrap();
    let foreign = Budget::new(LIMIT).unwrap();
    let mut dealer = dealer(&budget);
    let baseline = budget.used_bytes();
    let Issued {
        mask,
        run,
        source_floor,
        ..
    } = issue(&budget, &mut dealer).unwrap();
    assert!(matches!(
        run.into_roles(&foreign),
        Err(SigmaExecutionError::BudgetIdentity)
    ));
    assert_eq!(foreign.used_bytes(), 0);
    assert_source_charged(&budget, baseline, source_floor);
    drop(mask);
    assert_eq!(budget.used_bytes(), baseline);
    let Issued {
        mask,
        mut run,
        source_floor,
        ..
    } = issue(&budget, &mut dealer).unwrap();
    let frames = run.begin(&mut || false).unwrap();
    assert!(matches!(
        run.into_roles(&budget),
        Err(SigmaExecutionError::Native(typed::Error::Consumed))
    ));
    drop(mask);
    assert_source_charged(&budget, baseline, source_floor);
    drop(frames);
    assert_eq!(budget.used_bytes(), baseline);
}

#[test]
fn public_callback_reentrancy_panic_and_drop_burn_siblings_with_retained_frames() {
    let _serial = SERIAL.lock().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    // Everything is constructed inside this worker; no one-thread native owner
    // is sent across threads. On timeout do not join a deadlocked worker.
    let worker = std::thread::spawn(move || {
        for case in 0..4 {
            let budget = Budget::new(LIMIT).unwrap();
            let mut dealer = dealer(&budget);
            let baseline = budget.used_bytes();
            let Issued {
                mask,
                run,
                source_floor,
                ..
            } = issue(&budget, &mut dealer).unwrap();
            let (mut coordinator, [mut left, mut right]) = run.into_roles(&budget).unwrap();
            let a = left.begin(&mut || coordinator.is_aborted()).unwrap();
            let b = right.begin(&mut || coordinator.is_aborted()).unwrap();
            if case == 3 {
                drop(left);
            } else {
                match case {
                    0 => assert!(left
                        .advance(b.as_bytes().unwrap(), &mut || {
                            coordinator.cancel();
                            false
                        })
                        .is_err()),
                    1 => assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = left.advance(b.as_bytes().unwrap(), &mut || {
                            assert!(!coordinator.is_aborted());
                            panic!("external caller cancellation panic");
                        });
                    }))
                    .is_err()),
                    2 => left.cancel(),
                    _ => unreachable!(),
                }
                drop(left);
            }
            assert!(coordinator.is_aborted());
            assert!(right.begin(&mut || false).is_err());
            assert!(right.finish().is_err());
            drop((mask, coordinator));
            assert_split_source_charged(&budget, baseline, source_floor);
            assert!(a.as_bytes().is_err());
            assert!(b.as_bytes().is_err());
            drop((a, b));
            assert_eq!(budget.used_bytes(), baseline);
        }
        send.send(()).unwrap();
    });
    receive
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("public role callback deadlocked or panicked");
    worker.join().unwrap();
}

#[test]
fn frozen_source_and_semantic_identity_substitutions_fail_before_role_issuance() {
    let _serial = SERIAL.lock().unwrap();
    let publisher = plan();
    let budget = Budget::new(LIMIT).unwrap();
    let mut dealer = dealer(&budget);
    let baseline = budget.used_bytes();
    for case in 0..14 {
        let mut changed = publisher.clone();
        if case == 9 {
            changed.config_digest = Digest::from_sha256([83; 32]);
        }
        if case == 10 {
            changed.model_family.push_str("-not-publisher");
        }
        let inventory = lower_decoder_matrix_inventory(&changed).unwrap();
        let mut bytes = fixture::source_bytes();
        let mut values = fixture::weights_io();
        let mut encoded = coefficients();
        let mut coordinates = offsets();
        match case {
            0 => bytes[0] ^= 1,
            1 => coordinates.swap(0, 1),
            2 => encoded[0] += 1,
            3 => values[0] += 1.0,
            _ => {}
        }
        let mut r = request(&changed, &values, &encoded);
        match case {
            4 => r.selection.inner_coordinates.swap(0, 1),
            5 => r.selection.immutable_revision.push('0'),
            11 => {
                r.next_use = ConsumerUse::new(
                    r.activation_output,
                    [63; 32],
                    r.next_use.operand_position(),
                    r.next_use.ordinal(),
                )
            }
            12 => {
                r.source =
                    SourceIdentity::new(r.source.checkpoint(), [64; 32], r.source.implementation())
            }
            13 => {
                let id = r.input.identity();
                r.input = EdgeDescriptor::new(
                    EdgeIdentity::new(
                        id.source(),
                        id.plan(),
                        [65; 32],
                        id.wire(),
                        id.first_tensor_index(),
                    ),
                    r.input.shape(),
                    r.input.numeric(),
                    r.input.representation(),
                    r.input.weight_visibility(),
                )
                .unwrap();
            }
            _ => {}
        }
        let selection = r.selection.clone();
        let mut supplied = packet(&selection, &coordinates, &bytes);
        match case {
            6 => supplied.dtype = SourceDtype::F32,
            7 => supplied.endian = SourceEndian::Big,
            8 => supplied.tensor_offset += 2,
            _ => {}
        }
        assert!(
            matches!(
                execute_sigma_linear_activation(
                    pin(&publisher),
                    supplied,
                    r,
                    &inventory,
                    &budget,
                    &mut dealer,
                    &[0; fixture::INNER]
                ),
                Err(SigmaExecutionError::Source(_))
            ),
            "substitution {case}"
        );
        assert_eq!(budget.used_bytes(), baseline, "substitution {case}");
    }
    // Public execution exposes no entropy observer. These assertions establish
    // rejection at source/certificate admission and complete charge restoration,
    // not measured zero draws. Native injected-entropy tests establish that gate.
    drop(issue(&budget, &mut dealer).unwrap());
    assert_eq!(budget.used_bytes(), baseline);
}
