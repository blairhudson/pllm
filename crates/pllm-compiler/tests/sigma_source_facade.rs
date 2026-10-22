//! External-client D22 regression over the existing pinned BF16 submatrix.
//!
//! Uses only public compiler/native APIs. Outputs remain opaque local conditional
//! capabilities: successful completion is NOT decoded parity, SDK availability,
//! original-layer execution, publisher authentication, or role transport.

#[path = "../../pllm-garble/tests/sigma_linear_activation/fixture.rs"]
mod fixture;

use pllm_compiler::{lower_decoder_matrix_inventory, sigma_linear_activation::*};
use pllm_core::edge_contract::{
    ConsumerUse, EdgeDescriptor, EdgeIdentity, Interpretation, Numeric, Representation, Shape,
    SourceIdentity, WeightVisibility,
};
use pllm_garble::{masked_wire_reference::typed, shared_allocation::Budget};
use pllm_models::{lower_model_json, DecoderMode, DecoderPlan, DecoderWorkload, ModelOperator};
use pllm_types::{digest_bytes, Digest, NamedDigest};
use sha2::{Digest as _, Sha256};

const TENSOR_OFFSET: u64 = 281_019_680;

fn plan() -> DecoderPlan {
    // Source-shaped semantic metadata, not authenticated config bytes. The
    // publisher tuple below binds this exact local declaration and workload.
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
    // Test-local identity, not an advertised complete-method numeric profile.
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
    let edge = |producer, shape: &[usize], fraction, visibility, wire| {
        EdgeDescriptor::new(
            EdgeIdentity::new(source, plan_id, producer, [wire; 32], 0),
            Shape::new(shape).unwrap(),
            Numeric::new(50, fraction, Interpretation::Signed).unwrap(),
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
        // Public declared domain; NOT inferred from fixture input extrema.
        input_domain: IntegerInterval::from_half_open(-4096, 4097).unwrap(),
        source_coefficients: values,
        encoded_private_coefficients: encoded,
    }
}

fn pin(plan: &DecoderPlan) -> ExpectedSigmaSourcePin {
    ExpectedSigmaSourcePin::from_trusted_publisher(
        NamedDigest {
            id: "model.safetensors".into(),
            // Selected bytes, NOT the entire file's historical blob pin.
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

fn offsets() -> Vec<u64> {
    (0..fixture::INNER)
        .map(|i| TENSOR_OFFSET + 2 * i as u64)
        .collect()
}

fn weights() -> Vec<i64> {
    // Independent integer BF16 decoder and exact floor-Q12 encoder. No native
    // encoder or production source decoder defines the expected coefficients.
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
                // Tiny nonzero negatives floor to -1; positives floor to zero.
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

#[test]
fn pinned_bf16_submatrix_has_source_checked_conditional_certificate() {
    let plan = plan();
    let bytes = fixture::source_bytes();
    assert_eq!(fixture::FILE_SHA256.len(), 64); // Not fresh full-file verification.
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        fixture::SELECTED_SHA256
    );
    let values = fixture::weights_io();
    let encoded = weights();
    assert_eq!(encoded.iter().min(), Some(&-280));
    assert_eq!(encoded.iter().max(), Some(&324));
    let r = request(&plan, &values, &encoded);
    let source = import_sigma_source_bytes(
        pin(&plan),
        packet(&r.selection, &offsets(), &bytes),
        &encoded,
    )
    .unwrap();
    let checked = certify_source_checked_sigma_linear_activation(r, source).unwrap();
    let c = checked.certificate();
    assert_eq!(c.fixture_scope(), FixtureScope::CheckpointDerivedSubmatrix);
    assert_eq!(
        c.source_import_status(),
        SourceImportStatus::PinnedSourceBytesChecked
    );
    assert_eq!(c.encoded_private_coefficients(), encoded);
    assert_eq!(
        c.input_domain().half_open_endpoints().unwrap(),
        (-4096, 4097)
    );
    assert_eq!(c.sum_domain().endpoints(), (-339_738_624, 339_738_624));
    assert_eq!(c.rescaled_domain().endpoints(), (-82_944, 82_944));
    assert_eq!(c.multiply_accumulates(), 256);
    // Source certification does not mint transport or stronger native provenance.
    assert!(checked.native_elevation().is_err());
    assert!(c.native_elevation().is_err());
}

#[test]
fn public_raw_facade_completes_and_retires_real_source_owners_without_extracting_secrets() {
    // Sole successful issuance test in this integration-test process: matrix
    // singleton cannot be raced by another successful test in this binary.
    let budget = Budget::new(64 << 20).unwrap();
    let template = plan();
    let r = request(&template, &[], &[]);
    let mut dealer =
        typed::LocalDealer::new(r.source, r.input.identity().plan(), [97; 32], &budget).unwrap();
    let baseline = budget.used_bytes();
    drop(r);
    drop(template);

    // Source/input/issuance/plan/inventory caller buffers die BEFORE exchanges.
    let (mask, mut run, expected_output) = {
        let plan = plan();
        let inventory = lower_decoder_matrix_inventory(&plan).unwrap();
        let bytes = fixture::source_bytes();
        let values = fixture::weights_io();
        let encoded = weights();
        let input: Vec<_> = fixture::input()
            .into_iter()
            .map(|value| (value * 4096.0) as i64) // Exactly integral public fixture.
            .collect();
        let r = request(&plan, &values, &encoded);
        let expected_output = r.activation_output;
        let selection = r.selection.clone();
        let (mask, run) = execute_sigma_linear_activation(
            pin(&plan),
            packet(&selection, &offsets(), &bytes),
            r,
            &inventory,
            &budget,
            &mut dealer,
            &input,
        )
        .unwrap();
        (mask, run, expected_output)
    };
    let provenance = SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution;
    assert_eq!(run.provenance(), provenance);
    let mut frames = run.begin(&mut || false).unwrap();
    let mut retained = Vec::new();
    let output = loop {
        assert!(retained.len() < 11, "bounded native schedule exceeded");
        let step = run
            .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
            .unwrap();
        retained.push(frames);
        match step {
            SigmaExecutionStep::Frames(next) => frames = next,
            SigmaExecutionStep::Output(output) => break output,
        }
    };
    assert_eq!(retained.len(), 11);
    for party in &output {
        assert_eq!(party.descriptor(), expected_output);
        assert_eq!(party.provenance(), provenance);
    }
    // Completed/cancelled run rejects reuse, but does not reclaim escaped frames.
    run.cancel();
    assert!(run.begin(&mut || false).is_err());
    drop((run, mask));
    assert!(budget.used_bytes() > baseline);
    drop(retained);
    assert!(budget.used_bytes() > baseline);
    let [left, right] = output;
    drop(left);
    assert!(budget.used_bytes() > baseline);
    drop(right);
    assert_eq!(budget.used_bytes(), baseline);

    // Same public raw-source facade: malformed exchange and cancellation burn
    // both local peers; escaped frames keep source reservations until drop.
    for cancel in [false, true] {
        let plan = plan();
        let inventory = lower_decoder_matrix_inventory(&plan).unwrap();
        let values = fixture::weights_io();
        let encoded = weights();
        let bytes = fixture::source_bytes();
        let r = request(&plan, &values, &encoded);
        let selection = r.selection.clone();
        let (mask, mut run) = execute_sigma_linear_activation(
            pin(&plan),
            packet(&selection, &offsets(), &bytes),
            r,
            &inventory,
            &budget,
            &mut dealer,
            &[0; 256],
        )
        .unwrap();
        let frames = run.begin(&mut || false).unwrap();
        assert!(run.advance([&[], &[]], &mut || cancel).is_err());
        assert!(run.begin(&mut || false).is_err());
        run.cancel();
        drop((run, mask));
        assert!(budget.used_bytes() > baseline);
        drop(frames);
        assert_eq!(budget.used_bytes(), baseline);
    }
    drop(dealer);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn public_facade_rejects_source_coordinate_issuance_and_semantic_substitution() {
    let frozen_plan = plan();
    let budget = Budget::new(64 << 20).unwrap();
    let template = request(&frozen_plan, &[], &[]);
    let mut dealer = typed::LocalDealer::new(
        template.source,
        template.input.identity().plan(),
        [98; 32],
        &budget,
    )
    .unwrap();
    let baseline = budget.used_bytes();
    for case in 0..9 {
        let mut changed_plan = plan();
        if case == 6 {
            changed_plan.config_digest = Digest::from_sha256([81; 32]);
        }
        if case == 7 {
            changed_plan.model_family.push_str("-substituted");
        }
        let inventory = lower_decoder_matrix_inventory(&changed_plan).unwrap();
        let mut values = fixture::weights_io();
        let mut encoded = weights();
        let mut bytes = fixture::source_bytes();
        let mut offsets = offsets();
        match case {
            0 => bytes[0] ^= 1,
            1 => offsets.swap(0, 1),
            2 => encoded[0] += 1,
            3 => values[0] += 1.0,
            _ => {}
        }
        let mut r = request(&changed_plan, &values, &encoded);
        if case == 4 {
            r.selection.inner_coordinates.swap(0, 1);
        }
        if case == 5 {
            r.selection.weight_layout = "input_output_row_major".into();
        }
        let selection = r.selection.clone();
        let input = if case == 8 { [4097; 256] } else { [0; 256] };
        let result = execute_sigma_linear_activation(
            pin(&frozen_plan),
            packet(&selection, &offsets, &bytes),
            r,
            &inventory,
            &budget,
            &mut dealer,
            &input,
        );
        if case == 8 {
            assert!(matches!(result, Err(SigmaExecutionError::Resource)));
        } else {
            assert!(
                matches!(result, Err(SigmaExecutionError::Source(_))),
                "case {case}"
            );
        }
        assert_eq!(budget.used_bytes(), baseline, "case {case}");
    }
    drop(dealer);
    assert_eq!(budget.used_bytes(), 0);
}
