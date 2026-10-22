use super::super::tests::{plan, request};
use super::*;

#[test]
fn bounded_source_workspace_parity_and_actual_phase_storage_inventory() {
    let raw = bytes();
    let s = selection(&raw);
    let offsets = pin(s.clone()).offsets().unwrap();
    let encoded = coefficients();
    let expected = pin(s.clone());
    let quote = expected.checked_import_quote().unwrap();
    assert_eq!(quote, expected.import_quote().unwrap());
    assert_eq!(quote.source_byte_capacity, 1024);
    assert_eq!(quote.coefficient_capacity_bytes, 2048);
    let phase = SigmaSourcePhaseStorage::inventory().unwrap();
    assert_eq!(phase.workspace_bytes, size_of::<SigmaSourceWorkspace>());
    assert_eq!(phase.source_initialization_bytes, size_of::<[u8; 1024]>());
    assert_eq!(
        phase.coefficient_initialization_bytes,
        size_of::<[i64; 256]>()
    );
    assert_eq!(
        phase.hash_result_move_bytes,
        size_of::<sha2::digest::Output<Sha256>>() + size_of::<[u8; 32]>()
    );
    assert_eq!(
        phase.verification_move_bytes,
        size_of::<ExpectedSigmaSourcePin>()
            + size_of::<ExpectedSelectedSource>()
            + size_of::<VerifiedSelectedSource>()
            + size_of::<ZeroizingFixed<u8, 1024>>()
    );
    assert_eq!(
        phase.encoding_move_bytes,
        size_of::<ExpectedSigmaSourcePin>()
            + size_of::<VerifiedSelectedSource>()
            + size_of::<SigmaCoefficients>()
    );
    assert_eq!(phase.return_move_bytes, size_of::<CheckedSigmaSource>());
    let independently_counted_peak = size_of::<SigmaSourceWorkspace>()
        + (phase
            .source_initialization_bytes
            .max(phase.hash_result_move_bytes)
            + phase.verification_move_bytes)
            .max(phase.coefficient_initialization_bytes + phase.encoding_move_bytes)
            .max(size_of::<CheckedSigmaSource>());
    assert_eq!(
        quote.import_encoding_scratch_bytes,
        independently_counted_peak
    );
    let mut workspace = SigmaSourceWorkspace::new();
    let owner = checked_import_sigma_source_bytes_with_workspace(
        expected,
        packet(&s, &offsets, &raw),
        &encoded,
        &mut workspace,
    )
    .unwrap();
    let source_pointer = owner.verified.bytes().as_ptr();
    let coefficient_pointer = owner.encoded_private_coefficients().as_ptr();
    let binding = owner.source_binding();
    let values: Vec<_> = (0..128)
        .map(|i| {
            owner
                .verified
                .checked_finite_value_with_workspace(i, &mut workspace.selected.decode)
                .unwrap()
        })
        .collect();
    let p = plan();
    let inventory = lower_decoder_matrix_inventory(&p).unwrap();
    let make_request = || {
        let mut r = request(&p, &values, &encoded);
        r.selection = s.clone();
        let source = SourceIdentity::new(
            s.content_sha256,
            r.source.numeric_profile(),
            r.source.implementation(),
        );
        rebind_source(&mut r, source);
        r
    };
    let retained_quote = owner
        .checked_certification_retained_quote(&make_request())
        .unwrap();
    workspace.certification_hash.update(b"stale");
    let checked = certify_source_checked_with_inventory_and_workspace(
        make_request(),
        owner,
        &inventory,
        &mut workspace,
    )
    .unwrap();
    assert_eq!(checked.source.verified.bytes().as_ptr(), source_pointer);
    assert_eq!(
        checked
            .certificate()
            .encoded_private_coefficients()
            .as_ptr(),
        coefficient_pointer
    );
    assert_eq!(checked.source_binding(), binding);
    assert_eq!(
        checked.checked_retained_resource_bytes().unwrap(),
        retained_quote
    );
    let legacy_source =
        import_sigma_source_bytes(pin(s.clone()), packet(&s, &offsets, &raw), &encoded).unwrap();
    let legacy =
        certify_source_checked_sigma_linear_activation(make_request(), legacy_source).unwrap();
    assert_eq!(
        checked.certificate().digest(),
        legacy.certificate().digest()
    );
    assert!(checked.native_elevation().is_err());
}

#[test]
fn bounded_source_import_static_rejection_matches_legacy_and_workspace_recovers() {
    let raw = bytes();
    let s = selection(&raw);
    let offsets = pin(s.clone()).offsets().unwrap();
    let mut workspace = SigmaSourceWorkspace::new();
    for bad_issuance in [false, true] {
        let mut encoded = coefficients();
        if bad_issuance {
            encoded[0] += 1;
        }
        let make_packet = || {
            let mut p = packet(&s, &offsets, &raw);
            if !bad_issuance {
                p.artifact_name = "replacement.safetensors";
            }
            p
        };
        let error: &'static str = checked_import_sigma_source_bytes_with_workspace(
            pin(s.clone()),
            make_packet(),
            &encoded,
            &mut workspace,
        )
        .err()
        .unwrap();
        let legacy = import_sigma_source_bytes(pin(s.clone()), make_packet(), &encoded)
            .err()
            .unwrap();
        assert_eq!(error, legacy);
    }
    let owner = checked_import_sigma_source_bytes_with_workspace(
        pin(s.clone()),
        packet(&s, &offsets, &raw),
        &coefficients(),
        &mut workspace,
    )
    .unwrap();
    assert_eq!(owner.verified.bytes(), raw);
    assert_eq!(owner.encoded_private_coefficients(), coefficients());
}

fn bytes() -> Vec<u8> {
    // Exact BF16 source values: +0.25, -0.25, -2^-13, +2^-13.
    [0x3e80_u16, 0xbe80, 0xb900, 0x3900]
        .into_iter()
        .cycle()
        .take(128)
        .flat_map(u16::to_le_bytes)
        .collect()
}

fn coefficients() -> Vec<i64> {
    [1024, -1024, -1, 0].into_iter().cycle().take(128).collect()
}

fn selection(raw: &[u8]) -> CheckpointTensorSelection {
    let p = plan();
    let mut s = request(&p, &[], &[]).selection;
    s.immutable_revision = "0123456789abcdef0123456789abcdef01234567".into();
    s.content_sha256 = Sha256::digest(raw).into();
    s
}

fn pin(s: CheckpointTensorSelection) -> ExpectedSigmaSourcePin {
    ExpectedSigmaSourcePin::from_trusted_publisher(
        NamedDigest {
            id: "fixture-original.safetensors".into(),
            digest: Digest::from_sha256(s.content_sha256),
        },
        256,
        s,
        SourceDtype::Bf16,
        SourceEndian::Little,
        128,
        &plan().config_digest,
        &plan().digest(),
    )
    .unwrap()
}

fn packet<'a>(
    s: &'a CheckpointTensorSelection,
    offsets: &'a [u64],
    raw: &'a [u8],
) -> SigmaSourceBytes<'a> {
    SigmaSourceBytes {
        artifact_name: "fixture-original.safetensors",
        selection: s,
        dtype: SourceDtype::Bf16,
        endian: SourceEndian::Little,
        tensor_offset: 128,
        scalar_artifact_offsets: offsets,
        bytes: raw,
    }
}

#[test]
fn actual_bytes_owner_checks_every_coefficient_with_negative_floor() {
    let raw = bytes();
    let s = selection(&raw);
    let offsets = pin(s.clone()).offsets().unwrap();
    let owner =
        import_sigma_source_bytes(pin(s.clone()), packet(&s, &offsets, &raw), &coefficients())
            .unwrap();
    assert_eq!(owner.source_byte_count(), 256);
    assert_eq!(owner.encoded_private_coefficients(), coefficients());
    assert_eq!(
        (0..4)
            .map(|i| owner.source_value(i).unwrap())
            .collect::<Vec<_>>(),
        [0.25, -0.25, -1.0 / 8192.0, 1.0 / 8192.0]
    );
}

#[test]
fn f32_little_endian_source_uses_floor_not_ties_even_and_rejects_nonfinite() {
    for value in [-0.0001_f32, f32::NAN, f32::INFINITY, f32::MAX] {
        let raw: Vec<u8> = std::iter::repeat(value)
            .take(128)
            .flat_map(f32::to_le_bytes)
            .collect();
        let s = selection(&raw);
        let expected = ExpectedSigmaSourcePin::from_trusted_publisher(
            NamedDigest {
                id: "fixture-original.safetensors".into(),
                digest: Digest::from_sha256(s.content_sha256),
            },
            512,
            s.clone(),
            SourceDtype::F32,
            SourceEndian::Little,
            128,
            &plan().config_digest,
            &plan().digest(),
        )
        .unwrap();
        let offsets = expected.offsets().unwrap();
        let mut supplied = packet(&s, &offsets, &raw);
        supplied.dtype = SourceDtype::F32;
        let result = import_sigma_source_bytes(expected, supplied, &vec![-1; 128]);
        if value == -0.0001_f32 {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
        }
    }
}

#[test]
fn mutated_source_bytes_fail_fixed_expected_pin() {
    let raw = bytes();
    let s = selection(&raw);
    let offsets = pin(s.clone()).offsets().unwrap();
    for index in [0, 1, 127, 255] {
        let mut changed = raw.clone();
        changed[index] ^= 1;
        assert!(import_sigma_source_bytes(
            pin(s.clone()),
            packet(&s, &offsets, &changed),
            &coefficients()
        )
        .is_err());
    }
}

#[test]
fn wrong_offsets_artifact_layout_endian_dtype_revision_and_encoding_rejected() {
    let raw = bytes();
    let s = selection(&raw);
    let offsets = pin(s.clone()).offsets().unwrap();
    for case in 0..9 {
        let mut declared = s.clone();
        let mut changed_offsets = offsets.clone();
        match case {
            0 => declared.immutable_revision = "fedcba9876543210fedcba9876543210fedcba98".into(),
            1 => declared.weight_layout = "input_output_row_major".into(),
            2 => declared.encoding = "round-ties-even-q12".into(),
            3 => changed_offsets[64] += 2,
            _ => {}
        }
        let mut supplied = packet(&declared, &changed_offsets, &raw);
        match case {
            4 => supplied.endian = SourceEndian::Big,
            5 => supplied.dtype = SourceDtype::F32,
            6 => supplied.tensor_offset += 2,
            7 => supplied.artifact_name = "substituted.safetensors",
            8 => supplied.scalar_artifact_offsets = &changed_offsets[..127],
            _ => {}
        }
        assert!(
            import_sigma_source_bytes(pin(s.clone()), supplied, &coefficients()).is_err(),
            "case {case}"
        );
    }
    let mut reversed = raw.clone();
    for scalar in reversed.chunks_exact_mut(2) {
        scalar.swap(0, 1);
    }
    assert!(import_sigma_source_bytes(
        pin(s.clone()),
        packet(&s, &offsets, &reversed),
        &coefficients()
    )
    .is_err());
}

#[test]
fn selected_terms_and_canonical_issuance_coefficients_cannot_be_omitted() {
    let raw = bytes();
    let s = selection(&raw);
    let offsets = pin(s.clone()).offsets().unwrap();
    let encoded = coefficients();
    assert!(
        import_sigma_source_bytes(pin(s.clone()), packet(&s, &offsets, &raw[..254]), &encoded)
            .is_err()
    );
    assert!(
        import_sigma_source_bytes(pin(s.clone()), packet(&s, &offsets, &raw), &encoded[..127])
            .is_err()
    );
    for index in 0..encoded.len() {
        let mut changed = encoded.clone();
        changed[index] += 1;
        assert!(
            import_sigma_source_bytes(pin(s.clone()), packet(&s, &offsets, &raw), &changed)
                .is_err()
        );
    }
    let mut omitted = s.clone();
    omitted.inner_coordinates.pop();
    assert!(
        import_sigma_source_bytes(pin(s.clone()), packet(&omitted, &offsets, &raw), &encoded)
            .is_err()
    );
}

#[test]
fn nonfinite_and_out_of_ring_source_rejected_even_with_matching_expected_digest() {
    for bits in [0x7f80_u16, 0xff80, 0x7fc0, 0x7f7f] {
        let mut raw = bytes();
        raw[..2].copy_from_slice(&bits.to_le_bytes());
        let s = selection(&raw);
        let offsets = pin(s.clone()).offsets().unwrap();
        assert!(import_sigma_source_bytes(
            pin(s.clone()),
            packet(&s, &offsets, &raw),
            &coefficients()
        )
        .is_err());
    }
}

#[test]
fn pin_rejects_self_inconsistent_hash_extent_revision_and_unsupported_endian() {
    let raw = bytes();
    let s = selection(&raw);
    for case in 0..6 {
        let mut expected = s.clone();
        let mut artifact = NamedDigest {
            id: "fixture-original.safetensors".into(),
            digest: Digest::from_sha256(s.content_sha256),
        };
        let mut size = 256;
        let mut endian = SourceEndian::Little;
        let mut offset = 128;
        match case {
            0 => artifact.digest = Digest::from_sha256([0; 32]),
            1 => size = 254,
            2 => expected.immutable_revision = "main".into(),
            3 => endian = SourceEndian::Big,
            4 => offset = u64::MAX - 1,
            _ => offset = 129,
        }
        assert!(ExpectedSigmaSourcePin::from_trusted_publisher(
            artifact,
            size,
            expected,
            SourceDtype::Bf16,
            endian,
            offset,
            &plan().config_digest,
            &plan().digest(),
        )
        .is_err());
    }
}

fn rebind_source(r: &mut SigmaLinearActivationRequest<'_>, source: SourceIdentity) {
    let replace = |edge: EdgeDescriptor| {
        let id = edge.identity();
        EdgeDescriptor::new(
            pllm_core::edge_contract::EdgeIdentity::new(
                source,
                id.plan(),
                id.producer(),
                id.wire(),
                id.first_tensor_index(),
            ),
            edge.shape(),
            edge.numeric(),
            edge.representation(),
            edge.weight_visibility(),
        )
        .unwrap()
    };
    r.source = source;
    r.input = replace(r.input);
    r.private_weights = replace(r.private_weights);
    r.complete_sum = replace(r.complete_sum);
    r.rescaled = replace(r.rescaled);
    r.activation_output = replace(r.activation_output);
    let id = r.private_weights.identity();
    r.private_weights = EdgeDescriptor::new(
        pllm_core::edge_contract::EdgeIdentity::new(
            source,
            id.plan(),
            sigma_weight_selection_identity(&r.selection),
            id.wire(),
            0,
        ),
        r.private_weights.shape(),
        r.private_weights.numeric(),
        r.private_weights.representation(),
        r.private_weights.weight_visibility(),
    )
    .unwrap();
    r.next_use = ConsumerUse::new(
        r.activation_output,
        r.next_use.consumer(),
        r.next_use.operand_position(),
        r.next_use.ordinal(),
    );
}

#[test]
fn concrete_checked_bridge_preserves_semantics_and_still_blocks_native_elevation() {
    let raw = bytes();
    let s = selection(&raw);
    let offsets = pin(s.clone()).offsets().unwrap();
    let encoded = coefficients();
    let owner =
        import_sigma_source_bytes(pin(s.clone()), packet(&s, &offsets, &raw), &encoded).unwrap();
    let values: Vec<_> = (0..128).map(|i| owner.source_value(i).unwrap()).collect();
    let p = plan();
    let mut r = request(&p, &values, &encoded);
    r.selection = s.clone();
    let source = SourceIdentity::new(
        s.content_sha256,
        r.source.numeric_profile(),
        r.source.implementation(),
    );
    rebind_source(&mut r, source);
    let checked = certify_source_checked_sigma_linear_activation(r, owner).unwrap();
    assert_eq!(
        checked.certificate().source_import_status(),
        SourceImportStatus::PinnedSourceBytesChecked
    );
    assert!(checked.native_elevation().is_err());
}

#[test]
fn source_checked_bridge_rejects_bias_or_substituted_issuance_material() {
    let raw = bytes();
    let s = selection(&raw);
    let offsets = pin(s.clone()).offsets().unwrap();
    let encoded = coefficients();
    for biased in [false, true] {
        let owner = import_sigma_source_bytes(pin(s.clone()), packet(&s, &offsets, &raw), &encoded)
            .unwrap();
        let mut values: Vec<_> = (0..128).map(|i| owner.source_value(i).unwrap()).collect();
        let mut p = plan();
        if biased {
            let original = plan();
            let id = request(&original, &values, &encoded)
                .linear_operation
                .to_owned();
            for graph in [&mut p.prefill, &mut p.decode] {
                let linear = graph.operations.iter_mut().find(|o| o.id == id).unwrap();
                linear.attributes["bias"] = serde_json::json!("zero-but-declared-bias");
            }
        } else {
            values[0] = 0.5;
        }
        let mut r = request(&p, &values, &encoded);
        r.selection = s.clone();
        let source = SourceIdentity::new(
            s.content_sha256,
            r.source.numeric_profile(),
            r.source.implementation(),
        );
        rebind_source(&mut r, source);
        assert!(certify_source_checked_sigma_linear_activation(r, owner).is_err());
    }
}

#[test]
fn approved_checkpoint_submatrix_geometry_is_not_original_inner_dimension() {
    let raw = bytes();
    let mut s = selection(&raw);
    s.checkpoint = "Qwen/Qwen2.5-0.5B-Instruct".into();
    s.immutable_revision = "7ae557604adf67be50417f59c2c2f167def9a775".into();
    s.tensor = "model.layers.0.mlp.gate_proj.weight".into();
    s.original_weight_shape = [4864, 896];
    s.original_input_shape = [1, 896];
    s.output_coordinates = vec![0];
    s.inner_coordinates = (0..256).collect();
    let expected: Digest = serde_json::from_str(
        "\"780a723ba14459a03b2e72508debd6d4551d6ca13da75534c0f456de84c680de\"",
    )
    .unwrap();
    let mut digest = [0_u8; 32];
    for (i, value) in digest.iter_mut().enumerate() {
        *value = u8::from_str_radix(&expected.as_str()[2 * i..2 * i + 2], 16).unwrap();
    }
    s.content_sha256 = digest;
    assert_eq!(
        geometry(&s).unwrap(),
        ([1, 256, 1], FixtureScope::CheckpointDerivedSubmatrix)
    );
    let expected = ExpectedSigmaSourcePin::from_trusted_publisher(
        NamedDigest {
            id: "geometry-test-only-unverified-container-offset.safetensors".into(),
            digest: expected,
        },
        512,
        s,
        SourceDtype::Bf16,
        SourceEndian::Little,
        0,
        &plan().config_digest,
        &plan().digest(),
    )
    .unwrap();
    assert_eq!(
        expected.offsets().unwrap(),
        (0..512).step_by(2).collect::<Vec<_>>()
    );
    // No checkpoint bytes loaded: this test checks approved pin/geometry only,
    // never claims that synthetic source above matches the approved checkpoint.
}

#[test]
fn half_open_contract_uses_inclusive_integer_endpoints_for_products() {
    let x = IntegerInterval::from_half_open(-3, 3).unwrap();
    let w = IntegerInterval::from_half_open(-2, 5).unwrap();
    let (sum, z) = complete_dot_interval(x, w, 256).unwrap();
    assert_eq!(sum.half_open_endpoints().unwrap(), (-3072, 2049));
    assert_eq!(z.half_open_endpoints().unwrap(), (-1, 1));
    assert!(IntegerInterval::from_half_open(0, 0).is_err());
}

#[test]
fn import_quote_matches_retained_capacities_and_coefficient_owner_moves_once() {
    let raw = bytes();
    let s = selection(&raw);
    let expected = pin(s.clone());
    let offsets = expected.offsets().unwrap();
    let quote = expected.import_quote().unwrap();
    let encoded = coefficients();
    let values: Vec<_> = [0.25, -0.25, -1.0 / 8192.0, 1.0 / 8192.0]
        .into_iter()
        .cycle()
        .take(128)
        .collect();
    let p = plan();
    let mut r = request(&p, &values, &encoded);
    r.selection = s.clone();
    let source = SourceIdentity::new(
        s.content_sha256,
        r.source.numeric_profile(),
        r.source.implementation(),
    );
    rebind_source(&mut r, source);
    let before_copy_certificate_quote = expected.certification_retained_quote(&r).unwrap();
    let owner = import_sigma_source_bytes(expected, packet(&s, &offsets, &raw), &encoded).unwrap();
    assert_eq!(owner.resources().unwrap(), quote);
    assert_eq!(quote.source_byte_capacity, 1024);
    assert_eq!(quote.coefficient_capacity_bytes, 2048);
    assert_eq!(quote.decoded_f64_capacity_bytes, 0);
    assert_eq!(quote.arc_backing_bytes, 0);
    let coefficient_pointer = owner.encoded_private_coefficients().as_ptr();
    let retained_quote = owner.certification_retained_quote(&r).unwrap();
    assert_eq!(retained_quote, before_copy_certificate_quote);
    let checked = certify_source_checked_sigma_linear_activation(r, owner).unwrap();
    assert_eq!(checked.retained_resource_bytes().unwrap(), retained_quote);
    assert_eq!(
        checked
            .certificate()
            .encoded_private_coefficients()
            .as_ptr(),
        coefficient_pointer
    );
    assert!(checked.source.coefficients.is_none());
    assert_eq!(
        checked
            .source
            .resources()
            .unwrap()
            .coefficient_capacity_bytes,
        0
    );
}

#[test]
fn source_checked_owner_rejects_changed_config_or_plan_against_frozen_pin() {
    let raw = bytes();
    let s = selection(&raw);
    let offsets = pin(s.clone()).offsets().unwrap();
    let encoded = coefficients();
    for change_config in [false, true] {
        let owner = import_sigma_source_bytes(pin(s.clone()), packet(&s, &offsets, &raw), &encoded)
            .unwrap();
        let values: Vec<_> = (0..128).map(|i| owner.source_value(i).unwrap()).collect();
        let mut p = plan();
        if change_config {
            p.config_digest = Digest::from_sha256([99; 32]);
        } else {
            p.model_family.push_str(".substitution");
        }
        let mut r = request(&p, &values, &encoded);
        r.selection = s.clone();
        let source = SourceIdentity::new(
            s.content_sha256,
            r.source.numeric_profile(),
            r.source.implementation(),
        );
        rebind_source(&mut r, source);
        assert!(certify_source_checked_sigma_linear_activation(r, owner).is_err());
    }
}

#[test]
fn borrowed_inventory_certification_checks_both_pairing_and_original_publisher_pin() {
    let raw = bytes();
    let s = selection(&raw);
    let offsets = pin(s.clone()).offsets().unwrap();
    let encoded = coefficients();
    let p = plan();
    let original_inventory = lower_decoder_matrix_inventory(&p).unwrap();
    for case in 0..5 {
        let owner = import_sigma_source_bytes(pin(s.clone()), packet(&s, &offsets, &raw), &encoded)
            .unwrap();
        let values: Vec<_> = (0..128).map(|i| owner.source_value(i).unwrap()).collect();
        let source_pointer = owner.verified.bytes().as_ptr();
        let coefficient_pointer = owner.encoded_private_coefficients().as_ptr();
        let mut changed = p.clone();
        if case == 1 || case == 3 {
            changed.config_digest = Digest::from_sha256([88; 32]);
        } else if case == 2 || case == 4 {
            changed.model_family.push_str("-replacement");
        }
        let changed_inventory;
        let inventory = if case >= 3 {
            // Even a newly valid inventory cannot replace the original expected
            // publisher config/semantic plan. Neither binding subsumes the other.
            changed_inventory = lower_decoder_matrix_inventory(&changed).unwrap();
            &changed_inventory
        } else {
            &original_inventory
        };
        let mut r = request(&changed, &values, &encoded);
        r.selection = s.clone();
        let source = SourceIdentity::new(
            s.content_sha256,
            r.source.numeric_profile(),
            r.source.implementation(),
        );
        rebind_source(&mut r, source);
        let result = certify_source_checked_with_inventory(r, owner, inventory);
        if case == 0 {
            let checked = result.unwrap();
            assert_eq!(checked.source.verified.bytes().as_ptr(), source_pointer);
            assert_eq!(
                checked
                    .certificate()
                    .encoded_private_coefficients()
                    .as_ptr(),
                coefficient_pointer
            );
            assert!(checked.source.coefficients.is_none());
        } else {
            assert!(result.is_err(), "case {case}");
        }
    }
}

#[test]
fn finite_f32_ring50_overflow_import_rejected_despite_valid_byte_pin() {
    let raw: Vec<u8> = std::iter::repeat(f32::MAX)
        .take(128)
        .flat_map(f32::to_le_bytes)
        .collect();
    let s = selection(&raw);
    let expected = ExpectedSigmaSourcePin::from_trusted_publisher(
        NamedDigest {
            id: "fixture-original.safetensors".into(),
            digest: Digest::from_sha256(s.content_sha256),
        },
        512,
        s.clone(),
        SourceDtype::F32,
        SourceEndian::Little,
        128,
        &plan().config_digest,
        &plan().digest(),
    )
    .unwrap();
    let offsets = expected.offsets().unwrap();
    let mut supplied = packet(&s, &offsets, &raw);
    supplied.dtype = SourceDtype::F32;
    let result = import_sigma_source_bytes(expected, supplied, &vec![0; 128]);
    match result {
        Err(error) => assert!(error.contains("unsupported signed-ring50 floor encoding")),
        Ok(_) => panic!("finite source overflow accepted"),
    }
}
