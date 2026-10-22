//! Rust-only regressions for the unregistered bridge. No Python input factory,
//! raw native issuance, certificate fabrication, secret export or parity claim.
//! The publisher tuple and independently encoded BF16 fixture match the public
//! compiler integration test `tests/sigma_role_facade.rs`.

use super::*;
use pllm_compiler::{lower_decoder_matrix_inventory, sigma_linear_activation::*};
use pllm_core::edge_contract::{
    ConsumerUse, EdgeIdentity, Interpretation, Numeric, Representation, Shape, SourceIdentity,
    WeightVisibility,
};
use pllm_garble::masked_wire_reference::typed;
use pllm_models::{lower_model_json, DecoderMode, DecoderPlan, DecoderWorkload, ModelOperator};
use pllm_types::{digest_bytes, Digest, NamedDigest};
use std::{
    mem::{align_of, size_of_val},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Mutex,
    time::Duration,
};

#[path = "../../pllm-garble/tests/sigma_linear_activation/fixture.rs"]
mod fixture;

const LIMIT: usize = 64 << 20;
const TENSOR_OFFSET: u64 = 281_019_680;
// All matrix issuances in this bridge test module share the native singleton.
// Other compiler/native crate test binaries run in separate processes.
static SERIAL: Mutex<()> = Mutex::new(());

// Compile-time negative auto-trait assertions, without a new dev dependency.
// If a type gains the trait, `_` has two candidates and compilation fails.
macro_rules! assert_not_trait {
    ($ty:ty, $bound:path) => {
        const _: fn() = || {
            trait Ambiguous<A> {
                fn check() {}
            }
            impl<T: ?Sized> Ambiguous<()> for T {}
            struct Implements;
            impl<T: ?Sized + $bound> Ambiguous<Implements> for T {}
            let _ = <$ty as Ambiguous<_>>::check;
        };
    };
}

assert_not_trait!(LocalCoordinator, Send);
assert_not_trait!(LocalCoordinator, Sync);
assert_not_trait!(LocalRole, Send);
assert_not_trait!(LocalRole, Sync);
assert_not_trait!(LocalFrame, Send);
assert_not_trait!(LocalFrame, Sync);
assert_not_trait!(LocalOutput, Send);
assert_not_trait!(LocalOutput, Sync);
assert_not_trait!(LocalStep, Send);
assert_not_trait!(LocalStep, Sync);
assert_not_trait!(RoleCall<'static>, Send);
assert_not_trait!(RoleCall<'static>, Sync);
assert_not_trait!(PendingStep<'static>, Send);
assert_not_trait!(PendingStep<'static>, Sync);
assert_not_trait!(PendingOutput<'static>, Send);
assert_not_trait!(PendingOutput<'static>, Sync);

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

fn coefficients() -> Vec<i64> {
    // Independent integer BF16 -> floor-Q12; no production encoder oracle.
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

fn dealer(budget: &OwnedBudget) -> OwnedLocalDealer {
    let p = plan();
    let r = request(&p, &[], &[]);
    OwnedLocalDealer::new(r.source, r.input.identity().plan(), [117; 32], budget).unwrap()
}

struct FixtureIssue {
    owners: LocalOwners,
    descriptor: EdgeDescriptor,
    source_floor: usize,
}

fn issue(budget: &OwnedBudget, dealer: &mut OwnedLocalDealer) -> Result<FixtureIssue, BridgeError> {
    issue_changed(budget, dealer, 0)
}

fn issue_changed(
    budget: &OwnedBudget,
    dealer: &mut OwnedLocalDealer,
    change: usize,
) -> Result<FixtureIssue, BridgeError> {
    // These caller-owned inputs ALL retire before any test begins exchanging
    // frames. No test-side keepalive substitutes for original compiler custody.
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let mut bytes = fixture::source_bytes();
    let mut values = fixture::weights_io();
    let mut encoded = coefficients();
    let mut offsets: Vec<_> = (0..fixture::INNER)
        .map(|i| TENSOR_OFFSET + 2 * i as u64)
        .collect();
    match change {
        1 => bytes[0] ^= 1,
        2 => encoded[0] += 1,
        3 => values[0] += 1.0,
        4 => offsets.swap(0, 1),
        5 => bytes.clear(),
        6 => bytes.resize(SELECTED_SOURCE_BYTE_CAPACITY + 1, 0),
        7 => offsets.push(TENSOR_OFFSET),
        8 => values.push(0.0),
        9 => encoded.push(0),
        _ => {}
    }
    let r = request(&p, &values, &encoded);
    let descriptor = r.activation_output;
    let expected = pin(&p);
    let source_floor = expected.certification_retained_quote(&r).unwrap()
        + expected
            .import_quote()
            .unwrap()
            .import_encoding_scratch_bytes
        + owned_sigma_execution_compiler_preflight(&p, &inventory).unwrap();
    let selection = r.selection.clone();
    let supplied = SigmaSourceBytes {
        artifact_name: "model.safetensors",
        selection: &selection,
        dtype: SourceDtype::Bf16,
        endian: SourceEndian::Little,
        tensor_offset: TENSOR_OFFSET,
        scalar_artifact_offsets: &offsets,
        bytes: &bytes,
    };
    let mut input: Vec<_> = fixture::input()
        .into_iter()
        .map(|x| (x * 4096.0) as i64)
        .collect();
    if change == 10 {
        input.push(0);
    }
    let owners =
        from_existing_rust_inputs(expected, supplied, r, &inventory, budget, dealer, &input)?;
    Ok(FixtureIssue {
        owners,
        descriptor,
        source_floor,
    })
}

fn commit_step(mut pending: PendingStep<'_>) -> LocalStep {
    let step = pending.take_step().unwrap();
    pending.commit().unwrap();
    step
}

fn frame(step: LocalStep) -> LocalFrame {
    match step {
        LocalStep::Frame(frame) => frame,
        LocalStep::Ready => panic!("premature Ready"),
    }
}

fn begin(role: &LocalRole) -> LocalFrame {
    frame(commit_step(role.begin(&mut || false).unwrap()))
}

fn advance(role: &LocalRole, peer: &LocalFrame) -> LocalStep {
    commit_step(role.advance(|| Ok(peer), &mut || false).unwrap())
}

fn finish(role: &LocalRole) -> LocalOutput {
    let mut pending = role.finish().unwrap();
    let output = pending.take_output().unwrap();
    pending.commit().unwrap();
    output
}

fn final_frames([left, right]: &[LocalRole; 2]) -> [LocalFrame; 2] {
    let mut frames = [begin(left), begin(right)];
    for _ in 0..10 {
        frames = [
            frame(advance(left, &frames[1])),
            frame(advance(right, &frames[0])),
        ];
    }
    frames
}

fn ready(roles: &[LocalRole; 2]) -> [LocalFrame; 2] {
    let frames = final_frames(roles);
    assert!(matches!(advance(&roles[0], &frames[1]), LocalStep::Ready));
    assert!(matches!(advance(&roles[1], &frames[0]), LocalStep::Ready));
    assert_eq!(roles[0].retained.frames.get(), 22);
    frames
}

fn assert_burned(coordinator: &LocalCoordinator, sibling: &LocalRole, frames: &[LocalFrame]) {
    assert_eq!(coordinator.is_aborted(), Ok(true));
    assert!(sibling.begin(&mut || false).is_err());
    assert!(sibling.finish().is_err());
    for frame in frames {
        assert!(frame.native.as_bytes().is_err());
    }
}

fn watchdog(body: impl FnOnce() + Send + 'static) {
    let (send, receive) = std::sync::mpsc::channel();
    // All !Send owners are constructed INSIDE the worker, never transferred.
    let worker = thread::spawn(move || {
        body();
        send.send(()).unwrap();
    });
    receive
        .recv_timeout(Duration::from_secs(30))
        .expect("bridge callback deadlocked or worker panicked");
    worker.join().unwrap();
}

#[test]
fn callbacks_query_cancel_drop_and_reenter_same_role_without_borrow_or_lock_failure() {
    let _serial = SERIAL.lock().unwrap();
    watchdog(|| {
        for case in 0..6 {
            let budget = OwnedBudget::new(LIMIT).unwrap();
            let mut dealer = dealer(&budget);
            let baseline = budget.budget().used_bytes();
            let FixtureIssue {
                owners: (coordinator, [left, right]),
                ..
            } = issue(&budget, &mut dealer).unwrap();
            let mut coordinator = Some(coordinator);
            let mut queried = false;
            let mut pending = left
                .begin(&mut || {
                    queried = true;
                    coordinator.as_ref().unwrap().is_aborted().unwrap()
                })
                .unwrap();
            let a = frame(pending.take_step().unwrap());
            pending.commit().unwrap();
            assert!(queried, "begin did not invoke cancellation callback");
            let b = begin(&right);
            let mut called = false;
            let result = left.advance(|| Ok(&b), &mut || {
                called = true;
                assert_eq!(coordinator.as_ref().unwrap().is_aborted(), Ok(false));
                match case {
                    0 => return false,
                    1 => coordinator.as_ref().unwrap().cancel().unwrap(),
                    2 => drop(coordinator.take()),
                    3 => assert!(matches!(
                        left.begin(&mut || false),
                        Err(BridgeError::Consumed)
                    )),
                    4 => assert!(matches!(
                        left.advance(|| Ok(&b), &mut || false),
                        Err(BridgeError::Consumed)
                    )),
                    5 => assert!(matches!(left.finish(), Err(BridgeError::Consumed))),
                    _ => unreachable!(),
                }
                false
            });
            assert!(called, "advance did not invoke cancellation callback");
            if case == 0 {
                drop(commit_step(result.unwrap()));
                assert_eq!(coordinator.as_ref().unwrap().is_aborted(), Ok(false));
                coordinator.as_ref().unwrap().cancel().unwrap();
            } else {
                assert!(result.map(|_| ()).is_err());
            }
            assert!(left.retained.coordinator.borrow().is_aborted());
            assert!(right.begin(&mut || false).is_err());
            assert!(a.native.as_bytes().is_err());
            assert!(b.native.as_bytes().is_err());
            drop((left, right, coordinator));
            assert!(budget.budget().used_bytes() > baseline);
            drop((a, b));
            assert_eq!(budget.budget().used_bytes(), baseline);
        }
    });
}

#[test]
fn parser_and_callback_errors_panics_and_parser_reentry_burn_sibling() {
    let _serial = SERIAL.lock().unwrap();
    watchdog(|| {
        for case in 0..5 {
            let budget = OwnedBudget::new(LIMIT).unwrap();
            let mut dealer = dealer(&budget);
            let baseline = budget.budget().used_bytes();
            let FixtureIssue {
                owners: (coordinator, [left, right]),
                ..
            } = issue(&budget, &mut dealer).unwrap();
            let frames = [begin(&left), begin(&right)];
            let mut callback_called = false;
            match case {
                0 => assert!(matches!(
                    left.advance(|| Err(BridgeError::Resource), &mut || {
                        callback_called = true;
                        false
                    }),
                    Err(BridgeError::Resource)
                )),
                1 => assert!(catch_unwind(AssertUnwindSafe(|| {
                    let _ = left.advance(|| panic!("peer parser panic"), &mut || {
                        callback_called = true;
                        false
                    });
                }))
                .is_err()),
                2 => assert!(left
                    .advance(|| Ok(&frames[1]), &mut || {
                        callback_called = true;
                        true
                    })
                    .is_err()),
                3 => assert!(catch_unwind(AssertUnwindSafe(|| {
                    let _ = left.advance(|| Ok(&frames[1]), &mut || {
                        callback_called = true;
                        assert_eq!(coordinator.is_aborted(), Ok(false));
                        panic!("cancellation callback panic");
                    });
                }))
                .is_err()),
                4 => assert!(left
                    .advance(
                        || {
                            assert!(matches!(left.finish(), Err(BridgeError::Consumed)));
                            Ok(&frames[1])
                        },
                        &mut || false
                    )
                    .is_err()),
                _ => unreachable!(),
            }
            if case < 2 {
                assert!(!callback_called, "native callback ran after parser failure");
            } else if case < 4 {
                assert!(callback_called);
            }
            assert_burned(&coordinator, &right, &frames);
            drop((coordinator, left, right));
            assert!(budget.budget().used_bytes() > baseline);
            drop(frames);
            assert_eq!(budget.budget().used_bytes(), baseline);
        }
    });
}

#[test]
fn step_packaging_error_or_panic_after_take_revokes_escaped_frame_or_ready() {
    let _serial = SERIAL.lock().unwrap();
    for at_ready in [false, true] {
        for panic in [false, true] {
            let budget = OwnedBudget::new(LIMIT).unwrap();
            let mut dealer = dealer(&budget);
            let FixtureIssue {
                owners: (coordinator, roles),
                ..
            } = issue(&budget, &mut dealer).unwrap();
            let frames = if at_ready {
                Some(final_frames(&roles))
            } else {
                None
            };
            let mut escaped = None;
            let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), BridgeError> {
                let mut pending = if let Some(frames) = &frames {
                    roles[0].advance(|| Ok(&frames[1]), &mut || false)?
                } else {
                    roles[0].begin(&mut || false)?
                };
                escaped = Some(pending.take_step()?);
                assert!(pending.step.is_none(), "payload must have left guard");
                if panic {
                    panic!("packaging step after take");
                }
                Err(BridgeError::Resource)
            }));
            if panic {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap(), Err(BridgeError::Resource));
            }
            assert_eq!(coordinator.is_aborted(), Ok(true));
            assert!(roles[1].begin(&mut || false).is_err());
            match escaped.as_ref().unwrap() {
                LocalStep::Frame(value) => assert!(value.native.as_bytes().is_err()),
                LocalStep::Ready => assert!(at_ready),
            }
            drop((escaped, frames, roles, coordinator));
        }
    }
}

#[test]
fn output_packaging_error_or_panic_after_take_revokes_already_finished_sibling() {
    let _serial = SERIAL.lock().unwrap();
    for panic in [false, true] {
        let budget = OwnedBudget::new(LIMIT).unwrap();
        let mut dealer = dealer(&budget);
        let baseline = budget.budget().used_bytes();
        let FixtureIssue {
            owners: (coordinator, roles),
            descriptor,
            ..
        } = issue(&budget, &mut dealer).unwrap();
        let frames = ready(&roles);
        let sibling = finish(&roles[0]);
        assert_eq!(sibling.descriptor(), Ok(descriptor));
        let mut escaped = None;
        let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), BridgeError> {
            let mut pending = roles[1].finish()?;
            escaped = Some(pending.take_output()?);
            assert!(pending.output.is_none());
            assert!(
                pending.call.native.is_none(),
                "native finish already consumed role"
            );
            assert_eq!(escaped.as_ref().unwrap().descriptor(), Ok(descriptor));
            if panic {
                panic!("packaging output after take");
            }
            Err(BridgeError::Resource)
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert_eq!(result.unwrap(), Err(BridgeError::Resource));
        }
        assert_eq!(coordinator.is_aborted(), Ok(true));
        assert!(sibling.descriptor().is_err());
        assert!(escaped.as_ref().unwrap().descriptor().is_err());
        drop((roles, coordinator, frames, sibling));
        assert!(budget.budget().used_bytes() > baseline);
        drop(escaped);
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn pending_guards_require_exactly_one_take_before_commit() {
    let _serial = SERIAL.lock().unwrap();
    for output in [false, true] {
        for duplicate_take in [false, true] {
            let budget = OwnedBudget::new(LIMIT).unwrap();
            let mut dealer = dealer(&budget);
            let FixtureIssue {
                owners: (coordinator, roles),
                ..
            } = issue(&budget, &mut dealer).unwrap();
            if output {
                let frames = ready(&roles);
                let mut pending = roles[0].finish().unwrap();
                let escaped = if duplicate_take {
                    let value = pending.take_output().unwrap();
                    assert!(matches!(pending.take_output(), Err(BridgeError::Consumed)));
                    Some(value)
                } else {
                    None
                };
                assert_eq!(pending.commit(), Err(BridgeError::Consumed));
                if let Some(value) = escaped {
                    assert!(value.descriptor().is_err());
                }
                drop(frames);
            } else {
                let mut pending = roles[0].begin(&mut || false).unwrap();
                let escaped = if duplicate_take {
                    let value = pending.take_step().unwrap();
                    assert!(matches!(pending.take_step(), Err(BridgeError::Consumed)));
                    Some(value)
                } else {
                    None
                };
                assert_eq!(pending.commit(), Err(BridgeError::Consumed));
                if let Some(LocalStep::Frame(value)) = escaped {
                    assert!(value.native.as_bytes().is_err());
                }
            }
            assert_eq!(coordinator.is_aborted(), Ok(true));
            assert!(roles[1].finish().is_err());
        }
    }
}

#[test]
fn packaging_cancellation_before_commit_and_population_exhaustion_burn_sibling() {
    let _serial = SERIAL.lock().unwrap();
    for output in [false, true] {
        for exhausted in [false, true] {
            let budget = OwnedBudget::new(LIMIT).unwrap();
            let mut dealer = dealer(&budget);
            let baseline = budget.budget().used_bytes();
            let FixtureIssue {
                owners: (coordinator, roles),
                ..
            } = issue(&budget, &mut dealer).unwrap();
            if output {
                let frames = ready(&roles);
                let sibling = finish(&roles[0]);
                if exhausted {
                    roles[1].retained.outputs.set(MAX_OUTPUTS);
                    assert!(matches!(roles[1].finish(), Err(BridgeError::Resource)));
                } else {
                    let mut pending = roles[1].finish().unwrap();
                    let escaped = pending.take_output().unwrap();
                    coordinator.cancel().unwrap();
                    assert_eq!(pending.commit(), Err(BridgeError::Consumed));
                    assert!(escaped.descriptor().is_err());
                }
                assert!(sibling.descriptor().is_err());
                drop((sibling, frames));
            } else if exhausted {
                roles[0].retained.frames.set(MAX_FRAMES);
                assert!(matches!(
                    roles[0].begin(&mut || false),
                    Err(BridgeError::Resource)
                ));
            } else {
                let mut pending = roles[0].begin(&mut || false).unwrap();
                let escaped = frame(pending.take_step().unwrap());
                coordinator.cancel().unwrap();
                assert_eq!(pending.commit(), Err(BridgeError::Consumed));
                assert!(escaped.native.as_bytes().is_err());
            }
            assert_eq!(coordinator.is_aborted(), Ok(true));
            assert!(roles[1].begin(&mut || false).is_err());
            drop((roles, coordinator));
            assert_eq!(budget.budget().used_bytes(), baseline);
        }
    }
}

#[test]
fn opaque_same_role_and_replayed_frames_burn_both_and_one_ready_cannot_finish() {
    let _serial = SERIAL.lock().unwrap();
    for case in 0..3 {
        let budget = OwnedBudget::new(LIMIT).unwrap();
        let mut dealer = dealer(&budget);
        let FixtureIssue {
            owners: (coordinator, roles),
            ..
        } = issue(&budget, &mut dealer).unwrap();
        let frames = if case == 2 {
            final_frames(&roles)
        } else {
            [begin(&roles[0]), begin(&roles[1])]
        };
        match case {
            0 => assert!(roles[0].advance(|| Ok(&frames[0]), &mut || false).is_err()),
            1 => {
                drop(advance(&roles[0], &frames[1]));
                assert!(roles[0].advance(|| Ok(&frames[1]), &mut || false).is_err());
            }
            2 => {
                assert!(matches!(advance(&roles[0], &frames[1]), LocalStep::Ready));
                assert!(!coordinator.is_aborted().unwrap());
                assert!(roles[0].finish().is_err(), "one Ready released output");
                assert!(roles[1].advance(|| Ok(&frames[0]), &mut || false).is_err());
            }
            _ => unreachable!(),
        }
        assert_burned(&coordinator, &roles[1], &frames);
    }
}

#[test]
fn duplicate_finish_revokes_both_previously_committed_outputs() {
    let _serial = SERIAL.lock().unwrap();
    let budget = OwnedBudget::new(LIMIT).unwrap();
    let mut dealer = dealer(&budget);
    let FixtureIssue {
        owners: (coordinator, roles),
        ..
    } = issue(&budget, &mut dealer).unwrap();
    let frames = ready(&roles);
    let outputs = [finish(&roles[0]), finish(&roles[1])];
    assert!(outputs.iter().all(|value| value.descriptor().is_ok()));
    assert!(matches!(roles[0].finish(), Err(BridgeError::Consumed)));
    assert_eq!(coordinator.is_aborted(), Ok(true));
    assert!(outputs.iter().all(|value| value.descriptor().is_err()));
    assert!(frames.iter().all(|value| value.native.as_bytes().is_err()));
}

#[test]
fn public_source_authority_blocks_second_live_issuance_needed_for_foreign_bridge_frame() {
    let _serial = SERIAL.lock().unwrap();
    let budget = OwnedBudget::new(LIMIT).unwrap();
    let mut dealer = dealer(&budget);
    let baseline = budget.budget().used_bytes();
    let FixtureIssue {
        owners: (coordinator, [left, right]),
        ..
    } = issue(&budget, &mut dealer).unwrap();
    let stale = begin(&right);
    drop((coordinator, left, right));
    let held = budget.budget().used_bytes();
    assert!(matches!(
        issue(&budget, &mut dealer),
        Err(BridgeError::Compiler(SigmaExecutionError::Native(
            typed::Error::Resource
        )))
    ));
    assert_eq!(budget.budget().used_bytes(), held);
    assert!(stale.native.as_bytes().is_err());
    // Exact blocked seam: a foreign LocalFrame needs two distinct Retentions
    // from genuine public issuances, but this old opaque frame still owns the
    // process-wide matrix singleton. Do not forge another Rc/native owner or
    // copy private frame bytes to pretend this reaches ForeignFrame checking.
    drop(stale);
    assert_eq!(budget.budget().used_bytes(), baseline);
    drop(issue(&budget, &mut dealer).unwrap());
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn bridge_exact_fit_one_short_and_first_reservation_boundary_restore_original_counter() {
    let _serial = SERIAL.lock().unwrap();
    // Measure the authentic compiler/native admitted population, not a guessed
    // native quote or fabricated certificate. Every reservation remains held.
    let sizing = OwnedBudget::new(LIMIT).unwrap();
    let mut d = dealer(&sizing);
    let baseline = sizing.budget().used_bytes();
    let issued = issue(&sizing, &mut d).unwrap();
    let additional = sizing.budget().used_bytes() - baseline;
    drop(issued);
    assert_eq!(sizing.budget().used_bytes(), baseline);
    drop((d, sizing));
    for short in [true, false] {
        let limit = baseline + additional - usize::from(short);
        let budget = OwnedBudget::new(limit).unwrap();
        let mut d = dealer(&budget);
        assert_eq!(budget.budget().used_bytes(), baseline);
        let result = issue(&budget, &mut d);
        if short {
            assert!(matches!(
                result,
                Err(BridgeError::Compiler(SigmaExecutionError::Native(
                    typed::Error::Resource
                )))
            ));
        } else {
            let FixtureIssue {
                owners: (coordinator, roles),
                ..
            } = result.unwrap();
            assert_eq!(budget.budget().used_bytes(), limit);
            let frames = ready(&roles);
            let outputs = [finish(&roles[0]), finish(&roles[1])];
            // Native may refund retired per-frame/continuation storage. The
            // bridge/source reservations remain resident; no new headroom was
            // needed to complete after exact-fit admission.
            assert!(budget.budget().used_bytes() <= limit);
            assert!(outputs.iter().all(|value| value.descriptor().is_ok()));
            drop((frames, outputs, roles, coordinator));
        }
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
    let q = BridgeResources::checked().unwrap();
    let bridge_charge = q.payload_bytes + size_of::<OwnedExternalCharge>();
    for short in [true, false] {
        let budget = OwnedBudget::new(baseline + bridge_charge - usize::from(short)).unwrap();
        let mut d = dealer(&budget);
        let result = issue(&budget, &mut d);
        if short {
            assert!(matches!(
                result,
                Err(BridgeError::Allocation(AllocationError::Resource))
            ));
        } else {
            // Bridge reservation fits exactly; next compiler reservation cannot.
            assert!(matches!(
                result,
                Err(BridgeError::Compiler(SigmaExecutionError::Native(
                    typed::Error::Resource
                )))
            ));
        }
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

fn assert_retention(
    retained: &Rc<Retention>,
    identity: *const pllm_garble::shared_allocation::Budget,
    floor: usize,
) {
    assert_eq!(std::ptr::from_ref(retained._charge.budget()), identity);
    assert_eq!(
        retained._charge.payload_bytes(),
        BridgeResources::checked().unwrap().payload_bytes
    );
    let role_charge = owned_sigma_execution_role_split_resources()
        .unwrap()
        .additional_payload_bytes
        + size_of::<OwnedExternalCharge>();
    // Subtract BOTH new reservations; they cannot masquerade as retained old
    // pinned-source/compiler custody. Native/source pointer identity is private
    // to compiler tests, not observable through its public facade here.
    assert!(
        retained._charge.budget().used_bytes()
            >= retained._charge.charged_bytes() + role_charge + floor
    );
}

#[test]
fn original_counter_source_and_bridge_charge_survive_roots_through_final_frame_or_output() {
    let _serial = SERIAL.lock().unwrap();
    for output_last in [false, true] {
        let (coordinator, roles, descriptor, floor, identity) = {
            let budget = OwnedBudget::new(LIMIT).unwrap();
            let identity = std::ptr::from_ref(budget.budget());
            let mut dealer = dealer(&budget);
            let FixtureIssue {
                owners: (coordinator, roles),
                descriptor,
                source_floor,
            } = issue(&budget, &mut dealer).unwrap();
            (coordinator, roles, descriptor, source_floor, identity)
        }; // Root, dealer and all caller source/plan/coefficient buffers retired.
        let weak = Rc::downgrade(&coordinator.retained);
        let original = Rc::as_ptr(&coordinator.retained);
        let frames = ready(&roles);
        let outputs = [finish(&roles[0]), finish(&roles[1])];
        for output in &outputs {
            assert_eq!(Rc::as_ptr(&output.retained), original);
            assert_eq!(output.descriptor(), Ok(descriptor));
            assert_eq!(
                descriptor.identity().source().checkpoint(),
                selected_digest()
            );
            assert_eq!(
                output.native.provenance(),
                SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution
            );
            assert_retention(&output.retained, identity, floor);
        }
        for frame in &frames {
            assert_eq!(Rc::as_ptr(&frame.retained), original);
            assert_retention(&frame.retained, identity, floor);
        }
        drop((roles, coordinator));
        assert!(outputs.iter().all(|value| value.descriptor().is_err()));
        assert!(frames.iter().all(|value| value.native.as_bytes().is_err()));
        if output_last {
            let [last, sibling] = outputs;
            drop((frames, sibling));
            assert_retention(&last.retained, identity, floor);
            assert!(weak.upgrade().is_some());
            drop(last);
        } else {
            let [last, sibling] = frames;
            drop((outputs, sibling));
            assert_retention(&last.retained, identity, floor);
            assert!(weak.upgrade().is_some());
            drop(last);
        }
        assert!(weak.upgrade().is_none());
        // Same authentic source factory now succeeds: last owner released real
        // singleton, not merely the new bridge reservation.
        let budget = OwnedBudget::new(LIMIT).unwrap();
        let mut dealer = dealer(&budget);
        let baseline = budget.budget().used_bytes();
        drop(issue(&budget, &mut dealer).unwrap());
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn source_substitutions_caps_and_foreign_budget_fail_without_residual_charge() {
    let _serial = SERIAL.lock().unwrap();
    let budget = OwnedBudget::new(LIMIT).unwrap();
    let mut dealer = dealer(&budget);
    let baseline = budget.budget().used_bytes();
    for case in 1..=10 {
        let result = issue_changed(&budget, &mut dealer, case);
        if case <= 4 {
            assert!(
                matches!(
                    result,
                    Err(BridgeError::Compiler(SigmaExecutionError::Source(_)))
                ),
                "substitution {case}"
            );
        } else {
            assert!(matches!(result, Err(BridgeError::Resource)), "cap {case}");
        }
        assert_eq!(budget.budget().used_bytes(), baseline, "case {case}");
    }
    let foreign = OwnedBudget::new(LIMIT).unwrap();
    let foreign_baseline = foreign.budget().used_bytes();
    assert!(matches!(
        issue(&foreign, &mut dealer),
        Err(BridgeError::Compiler(SigmaExecutionError::BudgetIdentity))
    ));
    assert_eq!(foreign.budget().used_bytes(), foreign_baseline);
    assert_eq!(budget.budget().used_bytes(), baseline);
    drop(issue(&budget, &mut dealer).unwrap());
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn simulated_process_and_thread_affinity_mismatch_precedes_role_take_or_parser() {
    let current = Affinity::current();
    assert_eq!(current.check(), Ok(()));
    let mut process = current;
    process.process ^= 1;
    assert_eq!(process.check(), Err(BridgeError::Affinity));
    let mut wrong_thread = current;
    wrong_thread.thread = thread::spawn(|| thread::current().id()).join().unwrap();
    assert_eq!(wrong_thread.check(), Err(BridgeError::Affinity));

    let _serial = SERIAL.lock().unwrap();
    let budget = OwnedBudget::new(LIMIT).unwrap();
    let mut dealer = dealer(&budget);
    let FixtureIssue {
        owners: (mut coordinator, roles),
        ..
    } = issue(&budget, &mut dealer).unwrap();
    drop(roles);
    Rc::get_mut(&mut coordinator.retained).unwrap().affinity = process;
    assert_eq!(coordinator.is_aborted(), Err(BridgeError::Affinity));
    assert_eq!(coordinator.cancel(), Err(BridgeError::Affinity));
    drop(coordinator);

    let FixtureIssue {
        owners: (coordinator, [mut left, right]),
        ..
    } = issue(&budget, &mut dealer).unwrap();
    drop((coordinator, right));
    Rc::get_mut(&mut left.retained).unwrap().affinity = wrong_thread;
    let mut parsed = false;
    assert!(matches!(
        left.advance(
            || {
                parsed = true;
                Err(BridgeError::Resource)
            },
            &mut || false
        ),
        Err(BridgeError::Affinity)
    ));
    assert!(!parsed);
    assert!(
        left.native.borrow().is_some(),
        "affinity failure took continuation"
    );
    assert!(matches!(
        left.begin(&mut || panic!("affinity called callback")),
        Err(BridgeError::Affinity)
    ));
    assert!(matches!(left.finish(), Err(BridgeError::Affinity)));
    assert!(left.native.borrow().is_some());
    drop(left);

    let FixtureIssue {
        owners: (coordinator, roles),
        ..
    } = issue(&budget, &mut dealer).unwrap();
    let frames = ready(&roles);
    let mut output = finish(&roles[0]);
    let sibling = finish(&roles[1]);
    drop((roles, coordinator, frames, sibling));
    Rc::get_mut(&mut output.retained).unwrap().affinity = process;
    assert_eq!(output.descriptor(), Err(BridgeError::Affinity));
}

#[test]
fn independent_bridge_census_covers_population_transitions_layout_and_overflow() {
    assert_eq!((MAX_FRAMES, MAX_OUTPUTS, MAX_VALUES), (22, 2, 256));
    let quote = BridgeResources::checked().unwrap();
    // Independent Rc layout arithmetic, not another call to Layout::extend.
    let round_up = |bytes: usize, align: usize| bytes.div_ceil(align) * align;
    let control = 2 * size_of::<usize>();
    let retention = round_up(
        round_up(control, align_of::<Retention>()) + size_of::<Retention>(),
        align_of::<usize>().max(align_of::<Retention>()),
    );
    let population_slots = [
        size_of::<LocalCoordinator>(),
        2 * size_of::<LocalRole>(),
        22 * size_of::<LocalFrame>(),
        22 * size_of::<LocalStep>(),
        2 * size_of::<LocalOutput>(),
    ];
    let construction = [
        size_of::<Admission<'_>>(),
        size_of::<ExpectedSigmaSourcePin>(),
        size_of::<SigmaLinearActivationRequest<'_>>(),
        size_of::<SigmaSourceBytes<'_>>(),
        size_of::<Issued>(),
        size_of::<Result<Issued, SigmaExecutionError>>(),
        size_of::<Split>(),
        size_of::<Result<Split, SigmaExecutionError>>(),
        size_of::<Retention>(),
        size_of::<Affinity>(),
        size_of::<Rc<Retention>>(),
        size_of::<[LocalRole; 2]>(),
        size_of::<LocalOwners>(),
        size_of::<Result<LocalOwners, BridgeError>>(),
        size_of::<OwnedExternalCharge>(),
        size_of::<Result<OwnedExternalCharge, AllocationError>>(),
        size_of::<BridgeResources>(),
        size_of::<Result<BridgeResources, BridgeError>>(),
    ];
    let operation = [
        size_of::<RoleCall<'_>>(),
        size_of::<Result<RoleCall<'_>, BridgeError>>(),
        size_of::<PendingStep<'_>>(),
        size_of::<Result<PendingStep<'_>, BridgeError>>(),
        size_of::<PendingOutput<'_>>(),
        size_of::<Result<PendingOutput<'_>, BridgeError>>(),
        size_of::<Option<OwnedSigmaExecutionRole>>(),
        size_of::<OwnedSigmaExecutionRoleFrame>(),
        size_of::<Result<OwnedSigmaExecutionRoleFrame, SigmaExecutionError>>(),
        size_of::<OwnedSigmaExecutionRoleStep>(),
        size_of::<Result<OwnedSigmaExecutionRoleStep, SigmaExecutionError>>(),
        size_of::<OwnedSigmaExecutionRoleOutput>(),
        size_of::<Result<OwnedSigmaExecutionRoleOutput, SigmaExecutionError>>(),
        size_of::<LocalFrame>(),
        size_of::<Result<LocalFrame, BridgeError>>(),
        size_of::<LocalStep>(),
        size_of::<Result<LocalStep, BridgeError>>(),
        size_of::<LocalOutput>(),
        size_of::<Result<LocalOutput, BridgeError>>(),
        size_of::<EdgeDescriptor>(),
        size_of::<Result<EdgeDescriptor, SigmaExecutionError>>(),
        size_of::<Result<EdgeDescriptor, BridgeError>>(),
        size_of::<Result<(), BridgeError>>(),
        size_of::<Result<&'static [u8], SigmaExecutionError>>(),
    ];
    let calculation = [
        size_of_val(&population_slots),
        size_of_val(&construction),
        size_of_val(&operation),
        size_of::<[usize; 3]>(),
        size_of::<[usize; 3]>(),
        size_of::<[usize; 9]>(),
        size_of::<Result<usize, BridgeError>>(),
        size_of::<Layout>(),
        size_of::<Result<(Layout, usize), std::alloc::LayoutError>>(),
    ];
    let population_bytes: usize = population_slots.into_iter().sum();
    let transition = construction.into_iter().sum::<usize>()
        + 2 * operation.into_iter().sum::<usize>()
        + calculation.into_iter().sum::<usize>();
    assert_eq!(quote.retention_bytes, retention);
    assert_eq!(quote.population_bytes, population_bytes);
    assert_eq!(quote.transition_bytes, transition);
    assert_eq!(
        quote.payload_bytes,
        retention + population_bytes + transition
    );
    assert_eq!(sum(&[]), Ok(0));
    assert_eq!(sum(&[usize::MAX, 1]), Err(BridgeError::Resource));
    assert_eq!(
        population::<LocalFrame>(usize::MAX),
        Err(BridgeError::Resource)
    );
}
