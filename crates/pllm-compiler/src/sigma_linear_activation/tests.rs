use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use pllm_garble::masked_wire_reference::typed::approved_implementation;
use pllm_models::{lower_model_json, DecoderWorkload};

fn interval(a: i128, b: i128) -> IntegerInterval {
    IntegerInterval::new(a, b).unwrap()
}

#[test]
fn bounded_certificate_schemas_preserve_exact_legacy_bytes_and_domains() {
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let context = SigmaCertificationContext::new(&p, &inventory).unwrap();
    assert_eq!(context.plan_identity, p.digest().as_sha256());
    assert_eq!(context.plan_hex.as_slice(), p.digest().as_str().as_bytes());
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        assert_eq!(
            context.phase_identity(mode).unwrap(),
            sigma_semantic_plan_identity(&p, mode)
        );
        let mut bytes = Vec::new();
        bounded_schema_write(SigmaSchema::Mode(mode), &mut bytes).unwrap();
        assert_eq!(bytes, canonical_bytes(&mode));
    }
    for op in p.prefill.operations.iter().chain(&p.decode.operations) {
        assert_eq!(
            bounded_operation_identity(op).unwrap(),
            sigma_semantic_operation_identity(op)
        );
    }
    let mut selection = request(&p, &[], &[]).selection;
    selection.checkpoint = "escape-\"\\\n\t-λ".into();
    let mut bytes = Vec::new();
    bounded_schema_write(SigmaSchema::Selection(&selection), &mut bytes).unwrap();
    assert_eq!(bytes, canonical_bytes(&selection));
    assert_eq!(
        bounded_selection_identity(&selection).unwrap(),
        sigma_weight_selection_identity(&selection)
    );
    let intervals = [
        interval(-RING_HALF, RING_HALF - 1),
        interval(-SUM_BOUND, SUM_BOUND - 1),
        interval(-RESCALED_BOUND, RESCALED_BOUND - 1),
        interval(-1, 0),
    ];
    bytes.clear();
    bounded_schema_write(SigmaSchema::Intervals(&intervals), &mut bytes).unwrap();
    assert_eq!(bytes, canonical_bytes(&intervals));
}

#[test]
fn bounded_certificate_digest_matches_legacy_composition() {
    let p = plan();
    let values = vec![0.25; 128];
    let encoded = vec![1024; 128];
    let r = request(&p, &values, &encoded);
    let rescale_operation = r.rescale_operation;
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let context = SigmaCertificationContext::new(&p, &inventory).unwrap();
    let c = certify_with_coefficients(r, SourceImport::DeclarationOnly, None, &context).unwrap();
    let mut legacy = Sha256::new();
    legacy.update(SIGMA_LINEAR_ACTIVATION_SCHEMA.as_bytes());
    legacy.update(canonical_bytes(c.selection()));
    legacy.update(p.digest().as_str().as_bytes());
    legacy.update(canonical_bytes(&c.mode()));
    legacy.update(inventory.digest().as_str().as_bytes());
    for edge in c.edges {
        legacy.update(edge.digest());
    }
    legacy.update(rescale_operation);
    legacy.update(c.activation_spec().digest());
    legacy.update(c.next_use().digest());
    legacy.update(canonical_bytes(&(
        c.input_domain(),
        c.weight_domain(),
        c.sum_domain(),
        c.rescaled_domain(),
    )));
    for value in values {
        legacy.update(value.to_bits().to_le_bytes());
    }
    for value in encoded {
        legacy.update(value.to_le_bytes());
    }
    legacy.update([0]);
    let expected: [u8; 32] = legacy.finalize().into();
    assert_eq!(c.digest(), expected);
}

#[test]
fn immutable_inventory_pairing_rejects_stale_plan_and_config() {
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    for config in [false, true] {
        let mut changed = p.clone();
        if config {
            changed.config_digest = pllm_types::Digest::from_sha256([91; 32]);
        } else {
            changed.model_family.push_str("-stale");
        }
        assert!(SigmaCertificationContext::new(&changed, &inventory).is_err());
    }
    let other = p.clone();
    let context = SigmaCertificationContext::new(&p, &inventory).unwrap();
    // Semantic equality does not permit switching borrows midway through proof.
    assert!(certify_with_coefficients(
        request(&other, &[], &[]),
        SourceImport::DeclarationOnly,
        None,
        &context,
    )
    .is_err());
}

#[test]
fn bounded_selection_rejection_does_not_touch_digest_writer() {
    let p = plan();
    let mut selection = request(&p, &[], &[]).selection;
    selection.checkpoint = "x".repeat(sigma_stream_limits().max_string_bytes + 1);
    let mut output = vec![71, 72];
    assert!(bounded_schema_write(SigmaSchema::Selection(&selection), &mut output).is_err());
    assert_eq!(output, [71, 72]);
    // Narrowing is exact-or-reject, never truncating a legacy i128 endpoint.
    let mut intervals = [interval(0, 0); 4];
    intervals[0] = interval(i128::MIN, i128::MAX);
    assert!(bounded_schema_write(SigmaSchema::Intervals(&intervals), &mut output).is_err());
    assert_eq!(output, [71, 72]);
}

#[test]
fn concrete_proof_workspace_reuses_real_state_and_resets_between_certificates() {
    use std::mem::size_of_val;
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let context = SigmaCertificationContext::new(&p, &inventory).unwrap();
    let values = vec![0.25; 128];
    let encoded = vec![1024; 128];
    let mut workspace = SigmaProofWorkspace::new(&request(&p, &values, &encoded));
    // Independent component inventory of the actual object, including any
    // struct padding. This is NOT a whole compiler/source admission quote.
    let fields = size_of_val(&workspace.identities)
        + size_of_val(&workspace.edges)
        + size_of_val(&workspace.intervals)
        + size_of_val(&workspace.activation)
        + size_of_val(&workspace.dot)
        + size_of_val(&workspace.hash)
        + size_of_val(&workspace.edge_hash)
        + size_of_val(&workspace.activation_metadata);
    assert!(size_of_val(&workspace) >= fields);
    assert!(size_of_val(&workspace) - fields < std::mem::align_of::<SigmaProofWorkspace>() * 8);
    assert_eq!(
        size_of::<SigmaStreamWorkspace<'static>>(),
        SIGMA_STREAM_DEPTH * StreamFrame::storage_bytes()
    );
    let first = certify_with_workspace(
        request(&p, &values, &encoded),
        SourceImport::DeclarationOnly,
        None,
        &context,
        &mut workspace,
    )
    .unwrap();
    assert!(workspace.activation.is_none());
    assert_eq!(workspace.intervals[1], first.weight_domain());
    assert_eq!(workspace.intervals[2], first.sum_domain());
    assert_eq!(workspace.edges, first.edges);
    // Deliberately dirty each reusable state category; next call reinitializes.
    workspace.hash.update(b"stale");
    workspace.identities.fill([99; 32]);
    workspace.intervals.fill(interval(0, 0));
    workspace.dot.products.fill(i128::MAX);
    workspace.edges.reverse();
    let second = certify_with_workspace(
        request(&p, &values, &encoded),
        SourceImport::DeclarationOnly,
        None,
        &context,
        &mut workspace,
    )
    .unwrap();
    assert_eq!(first.digest(), second.digest());
    let mut dot = SigmaDotWorkspace::default();
    let (sum, _) =
        complete_dot_interval_with_workspace(interval(-3, 2), interval(-2, 4), 2, &mut dot)
            .unwrap();
    assert_eq!(dot.products, [6, -12, -4, 8]);
    assert_eq!(sum.endpoints(), (-24, 16));
    let rejection: &'static str = checked_floor_encode_q12(f64::NAN).unwrap_err();
    assert_eq!(
        rejection,
        "nonfinite or unsupported signed-ring50 floor encoding"
    );
}

// Independent signed floor equation: Rust / truncates toward zero, then repair
// negative remainders. Deliberately does not call div_euclid or production code.
fn floor_once(value: i128) -> i128 {
    value / 4096 - i128::from(value < 0 && value % 4096 != 0)
}

#[test]
fn endpoint_enclosure_matches_exhaustive_independent_complete_dots() {
    for (xl, xu, wl, wu) in [(-3, 2, -2, 4), (-4, -1, 1, 3), (0, 3, -5, -2)] {
        for inner in 1..=3 {
            let (sum, z) =
                complete_dot_interval(interval(xl, xu), interval(wl, wu), inner).unwrap();
            // Enumerate all independent term products and then all possible
            // complete sums. This is not a second endpoint-product formula.
            let terms: Vec<i128> = (xl..=xu)
                .flat_map(|x| (wl..=wu).map(move |w| x * w))
                .collect();
            let mut sums = vec![0];
            for _ in 0..inner {
                sums = sums
                    .iter()
                    .flat_map(|s| terms.iter().map(move |t| s + t))
                    .collect();
            }
            assert_eq!(
                sum.endpoints(),
                (*sums.iter().min().unwrap(), *sums.iter().max().unwrap())
            );
            for total in sums {
                let expected = floor_once(total);
                assert!((z.minimum..=z.maximum).contains(&expected));
            }
        }
    }
}

#[test]
fn quarter_ring_endpoints_and_negative_floor_boundaries() {
    let (s, z) =
        complete_dot_interval(interval(-SUM_BOUND, -SUM_BOUND), interval(1, 1), 1).unwrap();
    assert_eq!(s.minimum, -SUM_BOUND);
    assert_eq!(z.minimum, -RESCALED_BOUND);
    let (_, z) =
        complete_dot_interval(interval(SUM_BOUND - 1, SUM_BOUND - 1), interval(1, 1), 1).unwrap();
    assert_eq!(z.maximum, RESCALED_BOUND - 1);
    assert!(complete_dot_interval(interval(SUM_BOUND, SUM_BOUND), interval(1, 1), 1).is_err());
    assert!(
        complete_dot_interval(interval(-SUM_BOUND - 1, -SUM_BOUND - 1), interval(1, 1), 1).is_err()
    );
    for value in [-8193, -8192, -4097, -4096, -4095, -1, 0, 1, 4095, 4096] {
        let (_, z) = complete_dot_interval(interval(value, value), interval(1, 1), 1).unwrap();
        assert_eq!(z.endpoints(), (floor_once(value), floor_once(value)));
    }
}

#[test]
fn partial_sums_and_per_term_floors_are_not_certified_equation() {
    let (_, z) = complete_dot_interval(interval(-1, -1), interval(1, 1), 2).unwrap();
    assert_eq!(z.endpoints(), (-1, -1));
    assert_ne!(z.minimum, 2 * floor_once(-1));
    assert!(
        complete_dot_interval(interval(SUM_BOUND / 2, SUM_BOUND / 2), interval(1, 1), 1).is_ok()
    );
    assert!(
        complete_dot_interval(interval(SUM_BOUND / 2, SUM_BOUND / 2), interval(1, 1), 2).is_err()
    );
}

#[test]
fn wide_arithmetic_rejects_without_wrap_or_narrowing() {
    assert!(IntegerInterval::new(1, 0).is_err());
    assert!(complete_dot_interval(interval(i128::MIN, i128::MAX), interval(1, 1), 1).is_err());
    assert!(complete_dot_interval(
        interval(-RING_HALF, -RING_HALF),
        interval(-RING_HALF, -RING_HALF),
        usize::MAX
    )
    .is_err());
    assert!(complete_dot_interval(interval(0, 0), interval(0, 0), 0).is_err());
    // Products greatly exceed i64 even when eventual rejection is expected.
    assert!(
        complete_dot_interval(interval(1 << 40, 1 << 40), interval(1 << 40, 1 << 40), 1).is_err()
    );
}

#[test]
fn exact_binary64_floor_encoding_and_unsupported_values() {
    for (value, expected) in [
        (-0.0, 0),
        (-0.00001, -1),
        (-1.0 / 4096.0, -1),
        (-1.5 / 4096.0, -2),
        (1.5 / 4096.0, 1),
        (1.0, 4096),
    ] {
        assert_eq!(sigma_floor_encode_q12(value).unwrap(), expected);
    }
    assert_eq!(
        sigma_floor_encode_q12(-((1_u64 << 37) as f64)).unwrap(),
        -(1_i64 << 49)
    );
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        (1_u64 << 37) as f64,
    ] {
        assert!(sigma_floor_encode_q12(value).is_err());
    }
}

pub(super) fn plan() -> DecoderPlan {
    lower_model_json(
        br#"{
        "model_type":"qwen3","hidden_size":8,"intermediate_size":16,
        "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
        "vocab_size":32,"max_position_embeddings":32,"head_dim":4,
        "hidden_act":"silu","rms_norm_eps":1e-6,"rope_theta":10000.0,
        "tie_word_embeddings":true,"attention_bias":false,"attention_dropout":0.0,
        "rope_scaling":null,"sliding_window":null,"use_sliding_window":false,
        "use_cache":true,"max_window_layers":1,"layer_types":["full_attention"]
    }"#,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 1,
        },
    )
    .unwrap()
}

pub(super) fn request<'a>(
    plan: &'a DecoderPlan,
    values: &'a [f64],
    encoded: &'a [i64],
) -> SigmaLinearActivationRequest<'a> {
    let graph = &plan.prefill;
    let activation = graph
        .operations
        .iter()
        .find(|o| o.operator == ModelOperator::Silu)
        .unwrap();
    let linear = graph
        .operations
        .iter()
        .find(|o| activation.inputs.contains(&o.id))
        .unwrap();
    let input = graph
        .operations
        .iter()
        .find(|o| linear.inputs.contains(&o.id))
        .unwrap();
    let next = graph
        .operations
        .iter()
        .find(|o| o.inputs.contains(&activation.id))
        .unwrap();
    let selection = CheckpointTensorSelection {
        checkpoint: "synthetic-test-only-not-a-trained-checkpoint".into(),
        immutable_revision: "fixture-v1".into(),
        content_sha256: [1; 32],
        tensor: linear.attributes["weight"].as_str().unwrap().into(),
        original_weight_shape: [16, 8],
        weight_layout: "output_input_row_major".into(),
        output_coordinates: (0..16).collect(),
        inner_coordinates: (0..8).collect(),
        input_tensor: input.id.clone(),
        original_input_shape: [1, 8],
        input_layout: "row_inner_row_major".into(),
        input_row_coordinates: vec![0],
        encoding: SIGMA_LINEAR_ACTIVATION_ENCODING.into(),
    };
    let source = SourceIdentity::new(
        [1; 32],
        hash(b"test-profile", SIGMA_LINEAR_ACTIVATION_ENCODING.as_bytes()),
        approved_implementation(),
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
        &[1, 16],
        12,
        None,
        5,
    );
    let position = next
        .inputs
        .iter()
        .position(|id| id == &activation.id)
        .unwrap() as u16;
    SigmaLinearActivationRequest {
        plan,
        mode: DecoderMode::Prefill,
        linear_operation: &linear.id,
        activation_operation: &activation.id,
        next_operation: &next.id,
        input: edge(
            sigma_semantic_operation_identity(input),
            &[1, 8],
            12,
            None,
            1,
        ),
        private_weights: edge(
            sigma_weight_selection_identity(&selection),
            &[16, 8],
            12,
            Some(WeightVisibility::Private),
            2,
        ),
        complete_sum: edge(
            sigma_semantic_operation_identity(linear),
            &[1, 16],
            24,
            None,
            3,
        ),
        rescaled: edge(
            sigma_gapars12_operation_identity(linear),
            &[1, 16],
            12,
            None,
            4,
        ),
        activation_output,
        rescale_operation: sigma_gapars12_operation_identity(linear),
        next_use: ConsumerUse::new(
            activation_output,
            sigma_semantic_operation_identity(next),
            position,
            0,
        ),
        selection,
        source,
        input_domain: interval(-100, 100),
        source_coefficients: values,
        encoded_private_coefficients: encoded,
    }
}

#[test]
fn checked_certificate_binds_actual_weights_and_never_elevates_metadata() {
    let p = plan();
    let values = vec![0.25; 128];
    let encoded = vec![1024; 128];
    let c = certify_sigma_linear_activation(
        request(&p, &values, &encoded),
        SourceImport::DeclarationOnly,
    )
    .unwrap();
    assert_eq!(c.fixture_scope(), FixtureScope::CompleteOperation);
    assert_eq!(
        c.source_import_status(),
        SourceImportStatus::UnauthenticatedDeclaration
    );
    assert_eq!(c.weight_domain().endpoints(), (1024, 1024));
    assert_eq!(c.sum_domain().endpoints(), (-819200, 819200));
    assert_eq!(c.rescaled_domain().endpoints(), (-200, 200));
    assert_eq!(c.multiply_accumulates(), 128);
    assert_eq!(c.encoded_private_coefficients(), encoded);
    assert!(c.native_elevation().is_err());
    let other_values = vec![0.5; 128];
    let other_encoded = vec![2048; 128];
    let other = certify_sigma_linear_activation(
        request(&p, &other_values, &other_encoded),
        SourceImport::DeclarationOnly,
    )
    .unwrap();
    assert_ne!(c.digest(), other.digest());
    assert!(certify_sigma_linear_activation(
        request(&p, &values, &other_encoded),
        SourceImport::DeclarationOnly
    )
    .is_err());
}

#[test]
fn selection_rejects_partial_reordered_geometry_and_oversized_matrices() {
    let p = plan();
    let values = vec![0.25; 128];
    let encoded = vec![1024; 128];
    let selection = request(&p, &values, &encoded).selection;
    let mut s = selection.clone();
    s.inner_coordinates.pop();
    assert_eq!(
        geometry(&s).unwrap().1,
        FixtureScope::CheckpointDerivedSubmatrix
    );
    let mut s = selection.clone();
    s.inner_coordinates.swap(0, 1);
    assert_eq!(
        geometry(&s).unwrap().1,
        FixtureScope::CheckpointDerivedSubmatrix
    );
    let mut s = selection.clone();
    s.output_coordinates[1] = 0;
    assert!(geometry(&s).is_err());
    let mut s = selection.clone();
    s.weight_layout = "input_output_row_major".into();
    assert!(geometry(&s).is_err());
    let mut s = selection.clone();
    s.original_input_shape = [usize::MAX, 8];
    assert!(geometry(&s).is_err());
    let mut s = selection.clone();
    s.original_weight_shape = [16, 32];
    s.original_input_shape = [1, 32];
    s.inner_coordinates = (0..32).collect();
    assert!(geometry(&s).is_err());
    let mut s = selection;
    s.output_coordinates = vec![1, 0];
    assert_eq!(
        geometry(&s).unwrap().1,
        FixtureScope::CheckpointDerivedSubmatrix
    );
    // All three matrices at the exact 256-element boundary; 256 outputs.
    s.original_weight_shape = [16, 16];
    s.original_input_shape = [16, 16];
    s.output_coordinates = (0..16).collect();
    s.input_row_coordinates = (0..16).collect();
    s.inner_coordinates = (0..16).collect();
    assert_eq!(geometry(&s).unwrap().0, [16, 16, 16]);
}

#[test]
fn whole_output_row_submatrix_is_labeled_not_original_layer_execution() {
    let p = plan();
    let values = vec![0.25; 16];
    let encoded = vec![1024; 16];
    let mut r = request(&p, &values, &encoded);
    r.selection.output_coordinates = vec![7, 2];
    let reshape = |edge: EdgeDescriptor, shape: &[usize], producer| {
        let old = edge.identity();
        EdgeDescriptor::new(
            EdgeIdentity::new(old.source(), old.plan(), producer, old.wire(), 0),
            Shape::new(shape).unwrap(),
            edge.numeric(),
            edge.representation(),
            edge.weight_visibility(),
        )
        .unwrap()
    };
    r.private_weights = reshape(
        r.private_weights,
        &[2, 8],
        sigma_weight_selection_identity(&r.selection),
    );
    r.complete_sum = reshape(
        r.complete_sum,
        &[1, 2],
        r.complete_sum.identity().producer(),
    );
    r.rescaled = reshape(r.rescaled, &[1, 2], r.rescaled.identity().producer());
    r.activation_output = reshape(
        r.activation_output,
        &[1, 2],
        r.activation_output.identity().producer(),
    );
    r.next_use = ConsumerUse::new(
        r.activation_output,
        r.next_use.consumer(),
        r.next_use.operand_position(),
        0,
    );
    let c = certify_sigma_linear_activation(r, SourceImport::DeclarationOnly).unwrap();
    assert_eq!(c.fixture_scope(), FixtureScope::CheckpointDerivedSubmatrix);
    assert_eq!(c.selection().output_coordinates, [7, 2]);
    assert_eq!(c.multiply_accumulates(), 16);
    assert!(c.native_elevation().is_err());
}

#[test]
fn substitution_nonfinite_partial_dot_and_next_consumer_rejected() {
    let p = plan();
    let values = vec![0.25; 128];
    let encoded = vec![1024; 128];
    let reject =
        |r| assert!(certify_sigma_linear_activation(r, SourceImport::DeclarationOnly).is_err());
    let mut r = request(&p, &values, &encoded);
    r.selection.content_sha256 = [2; 32];
    reject(r);
    let mut r = request(&p, &values, &encoded);
    r.selection.tensor.push_str(".wrong");
    reject(r);
    let mut r = request(&p, &values, &encoded);
    r.selection.inner_coordinates.pop();
    reject(r);
    let mut r = request(&p, &values, &encoded);
    r.rescale_operation = [99; 32];
    reject(r);
    let mut r = request(&p, &values, &encoded);
    r.next_use = ConsumerUse::new(r.activation_output, [99; 32], 0, 0);
    reject(r);
    let mut r = request(&p, &values, &encoded);
    r.complete_sum = r.rescaled;
    reject(r);
    let mut r = request(&p, &values, &encoded);
    r.input_domain = interval(-SUM_BOUND, SUM_BOUND - 1);
    reject(r);
    let mut bad_values = values.clone();
    bad_values[0] = f64::NAN;
    reject(request(&p, &bad_values, &encoded));
    reject(request(&p, &values[..127], &encoded[..127]));
}

#[test]
fn biased_and_tanh_or_attribute_modified_semantics_fail_closed() {
    let values = vec![0.25; 128];
    let encoded = vec![1024; 128];
    for case in 0..3 {
        let mut p = plan();
        let activation_id = p
            .prefill
            .operations
            .iter()
            .find(|o| o.operator == ModelOperator::Silu)
            .unwrap()
            .id
            .clone();
        // Keep lookup helper able to find SiLU; mutate after constructing the
        // operation identifiers, then construct request from unmodified plan
        // and replace only its borrowed plan/IDs with corresponding owned IDs.
        let original = p.clone();
        let mut r = request(&original, &values, &encoded);
        match case {
            0 => {
                for graph in [&mut p.prefill, &mut p.decode] {
                    let o = graph
                        .operations
                        .iter_mut()
                        .find(|o| o.id == r.linear_operation)
                        .unwrap();
                    o.attributes["bias"] = serde_json::json!("unsupported.bias");
                }
            }
            1 => {
                p.prefill
                    .operations
                    .iter_mut()
                    .find(|o| o.id == activation_id)
                    .unwrap()
                    .operator = ModelOperator::GeluTanh;
            }
            _ => {
                p.prefill
                    .operations
                    .iter_mut()
                    .find(|o| o.id == activation_id)
                    .unwrap()
                    .attributes = serde_json::json!({"approximate":"tanh"});
            }
        }
        r.plan = &p;
        assert!(certify_sigma_linear_activation(r, SourceImport::DeclarationOnly).is_err());
    }
}
