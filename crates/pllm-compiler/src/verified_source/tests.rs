use super::*;

#[test]
fn static_capacity_digest_and_decode_errors_match_legacy_contracts() {
    let tuple = tuple();
    let independent = tuple.checkpoint.capacity()
        + tuple.immutable_revision.capacity()
        + tuple.tensor.capacity()
        + (tuple.output_coordinates.capacity() + tuple.inner_coordinates.capacity())
            * size_of::<usize>();
    assert_eq!(tuple.checked_metadata_backing_bytes().unwrap(), independent);
    assert_eq!(tuple.metadata_backing_bytes().unwrap(), independent);
    assert_eq!(tuple.checked_count().unwrap(), 2);
    let error: SourceError = checked_sum_bytes([usize::MAX, 1]).unwrap_err();
    assert_eq!(error, "resource byte overflow");
    assert_eq!(sum_bytes([usize::MAX, 1]).unwrap_err(), error);
    assert_eq!(
        checked_vector_bytes(&tuple.inner_coordinates).unwrap(),
        vector_bytes(&tuple.inner_coordinates).unwrap()
    );
    for digest_bytes in [[0; 32], [0xff; 32], [0x9a; 32]] {
        let digest = Digest::from_sha256(digest_bytes);
        assert_eq!(checked_digest_identity(&digest).unwrap(), digest_bytes);
        assert_eq!(digest_identity(&digest).unwrap(), digest_bytes);
    }
    for malformed in ["", "00", &"G".repeat(64), &"A".repeat(64), &"0".repeat(65)] {
        let error: SourceError = checked_digest_text_identity(malformed).unwrap_err();
        assert_eq!(error, "invalid expected digest");
    }
    let mut workspace = SourceDecodeWorkspace::new();
    assert_eq!(
        checked_decode_with_workspace(SourceDtype::Bf16, &[0], &mut workspace),
        Err("source scalar width")
    );
    assert_eq!(
        checked_decode_with_workspace(SourceDtype::F32, &f32::NAN.to_le_bytes(), &mut workspace),
        Err("nonfinite source scalar")
    );
    assert_eq!(
        checked_decode_with_workspace(SourceDtype::Bf16, &0x3e80u16.to_le_bytes(), &mut workspace)
            .unwrap(),
        0.25
    );
    assert_eq!(workspace.bytes, [0x80, 0x3e, 0, 0]);
}

// Independent framed transcript, not another call through the production hash
// helper. Test-owned Vec is not part of admitted source storage.
fn independent_binding(expected: &ExpectedSelectedSource, raw: &[u8]) -> [u8; 32] {
    let mut transcript = b"pllm.verified-selected-source.v1\0".to_vec();
    for text in [
        expected.artifact_name(),
        &expected.tuple().checkpoint,
        &expected.tuple().immutable_revision,
        &expected.tuple().tensor,
    ] {
        transcript.extend_from_slice(&(text.len() as u64).to_le_bytes());
        transcript.extend_from_slice(text.as_bytes());
    }
    transcript.extend_from_slice(&expected.digest());
    transcript.extend_from_slice(&expected.tuple().expected_config);
    transcript.extend_from_slice(&expected.tuple().expected_semantic_plan);
    for extent in expected.tuple().original_shape {
        transcript.extend_from_slice(&(extent as u64).to_le_bytes());
    }
    for coordinates in [
        &expected.tuple().output_coordinates,
        &expected.tuple().inner_coordinates,
    ] {
        transcript.extend_from_slice(&(coordinates.len() as u64).to_le_bytes());
        for &coordinate in coordinates {
            transcript.extend_from_slice(&(coordinate as u64).to_le_bytes());
        }
    }
    transcript.extend_from_slice(&[
        match expected.tuple().dtype {
            SourceDtype::Bf16 => 0,
            SourceDtype::F32 => 1,
        },
        0,
        1,
    ]);
    transcript.extend_from_slice(&expected.tuple().tensor_offset.to_le_bytes());
    transcript.extend_from_slice(raw);
    Sha256::digest(&transcript).into()
}

#[test]
fn caller_workspace_verifier_preserves_original_metadata_source_and_commitment() {
    let raw = raw();
    let declaration = tuple();
    let pin = expected(declaration.clone(), &raw);
    let name_pointer = pin.artifact_name().as_ptr();
    let checkpoint_pointer = pin.tuple().checkpoint.as_ptr();
    let coordinate_pointer = pin.tuple().output_coordinates.as_ptr();
    let binding = independent_binding(&pin, &raw);
    let quote = pin.checked_import_quote().unwrap();
    assert_eq!(quote, pin.import_quote().unwrap());
    assert_eq!(pin.checked_scalar_artifact_offset(0), Ok(132));
    assert_eq!(
        pin.checked_scalar_artifact_offset(2),
        Err("source offset index out of bounds")
    );
    let mut workspace = SelectedSourceWorkspace::new();
    let owner = checked_verify_selected_source_with_workspace(
        pin,
        SelectedSourceBytes {
            artifact_name: "test.safetensors",
            tuple: &declaration,
            scalar_artifact_offsets: &[132, 134],
            bytes: &raw,
        },
        &mut workspace,
    )
    .unwrap();
    assert_eq!(owner.source_binding(), binding);
    let artifact_digest: [u8; 32] = Sha256::digest(&raw).into();
    assert_eq!(workspace.actual_digest, artifact_digest);
    assert_eq!(workspace.binding, binding);
    assert_eq!(owner.expected.artifact_name().as_ptr(), name_pointer);
    assert_eq!(
        owner.expected_tuple().checkpoint.as_ptr(),
        checkpoint_pointer
    );
    assert_eq!(
        owner.expected_tuple().output_coordinates.as_ptr(),
        coordinate_pointer
    );
    assert_eq!(owner.checked_resources().unwrap(), quote);
    assert_eq!(
        owner.checked_finite_value_with_workspace(0, &mut workspace.decode),
        Ok(0.25)
    );
    assert_eq!(
        owner.checked_finite_value_with_workspace(2, &mut workspace.decode),
        Err("source scalar out of bounds")
    );
    let source_pointer = owner.bytes().as_ptr();
    let moved = owner;
    assert_eq!(moved.bytes().as_ptr(), source_pointer);
    assert_eq!(moved.bytes(), raw);
    assert_eq!(moved.bytes.len(), SELECTED_SOURCE_BYTE_CAPACITY);
    // A reused hash workspace never carries earlier successful/failed hash state
    // into a different publisher tuple or the next source binding.
    let mut another = declaration;
    another.expected_config = [9; 32];
    another.expected_semantic_plan = [10; 32];
    let pin = expected(another.clone(), &raw);
    let expected_binding = independent_binding(&pin, &raw);
    workspace.hash.update(b"stale");
    let owner = checked_verify_selected_source_with_workspace(
        pin,
        SelectedSourceBytes {
            artifact_name: "test.safetensors",
            tuple: &another,
            scalar_artifact_offsets: &[132, 134],
            bytes: &raw,
        },
        &mut workspace,
    )
    .unwrap();
    assert_eq!(owner.source_binding(), expected_binding);
    assert_ne!(owner.source_binding(), binding);
}

#[test]
fn static_and_legacy_verification_rejections_match_without_mutating_borrowed_source() {
    let original = raw();
    let mut workspace = SelectedSourceWorkspace::new();
    for case in 0..7 {
        let mut bytes = original.clone();
        let mut declaration = tuple();
        let mut offsets = [132, 134];
        let mut name = "test.safetensors";
        match case {
            0 => name = "wrong.safetensors",
            1 => {
                bytes.pop();
            }
            2 => offsets[0] += 2,
            3 => bytes[0] ^= 1,
            4 => declaration.expected_config = [99; 32],
            5 => declaration.expected_semantic_plan = [98; 32],
            _ => declaration.inner_coordinates.swap(0, 1),
        }
        let before = bytes.clone();
        let packet = || SelectedSourceBytes {
            artifact_name: name,
            tuple: &declaration,
            scalar_artifact_offsets: &offsets,
            bytes: &bytes,
        };
        let bounded: SourceError = checked_verify_selected_source_with_workspace(
            expected(tuple(), &original),
            packet(),
            &mut workspace,
        )
        .err()
        .unwrap();
        let legacy = verify_selected_source(expected(tuple(), &original), packet())
            .err()
            .unwrap();
        assert_eq!(legacy, bounded, "case {case}");
        assert_eq!(bytes, before);
    }
    let mut nonfinite = original.clone();
    nonfinite[..2].copy_from_slice(&0x7f80u16.to_le_bytes());
    let error: SourceError = checked_verify_pinned_source_bytes_with_workspace(
        expected(tuple(), &nonfinite),
        "test.safetensors",
        &[132, 134],
        &nonfinite,
        &mut workspace,
    )
    .err()
    .unwrap();
    assert_eq!(error, "nonfinite source scalar");
    let owner = checked_verify_pinned_source_bytes_with_workspace(
        expected(tuple(), &original),
        "test.safetensors",
        &[132, 134],
        &original,
        &mut workspace,
    )
    .unwrap();
    assert_eq!(owner.bytes(), original);
}

#[test]
fn static_constructor_and_resource_inventory_preserve_legacy_rejections() {
    let raw = raw();
    for case in 0..8 {
        let setup = || {
            let mut declaration = tuple();
            let mut name = "test.safetensors";
            let mut selected_bytes = raw.len();
            match case {
                0 => name = "",
                1 => selected_bytes += 1,
                2 => declaration.endian = SourceEndian::Big,
                3 => declaration.inner_coordinates = vec![0, 0],
                4 => {
                    declaration.checkpoint = String::with_capacity(65537);
                    declaration.checkpoint.push('x');
                }
                5 => declaration.tensor_offset = u64::MAX - 1,
                6 => declaration.immutable_revision = "g".repeat(40),
                _ => {
                    declaration.original_shape = [2, 256];
                    declaration.output_coordinates = vec![0, 1];
                    declaration.inner_coordinates = (0..256).collect();
                }
            }
            (
                NamedDigest {
                    id: name.into(),
                    digest: Digest::from_sha256(Sha256::digest(&raw).into()),
                },
                declaration,
                selected_bytes,
            )
        };
        let (artifact, declaration, bytes) = setup();
        let error: SourceError =
            ExpectedSelectedSource::checked_from_trusted_publisher(artifact, declaration, bytes)
                .err()
                .unwrap();
        let (artifact, declaration, bytes) = setup();
        let legacy = ExpectedSelectedSource::from_trusted_publisher(artifact, declaration, bytes)
            .err()
            .unwrap();
        assert_eq!(error, legacy, "case {case}");
    }
    let pin = expected(tuple(), &raw);
    let quote = pin.checked_import_quote().unwrap();
    let metadata = pin.artifact_name.capacity()
        + pin.tuple.checkpoint.capacity()
        + pin.tuple.immutable_revision.capacity()
        + pin.tuple.tensor.capacity()
        + (pin.tuple.output_coordinates.capacity() + pin.tuple.inner_coordinates.capacity())
            * size_of::<usize>();
    assert_eq!(quote.metadata_backing_bytes, metadata);
    assert_eq!(
        quote.retained_bytes,
        size_of::<VerifiedSelectedSource>() + metadata + 1024
    );
    assert_eq!(
        quote.import_scratch_bytes,
        size_of::<SelectedSourceWorkspace>()
            + 1024
            + size_of::<ExpectedSelectedSource>()
            + size_of::<VerifiedSelectedSource>()
            + size_of::<ZeroizingFixed<u8, 1024>>()
    );
    let owner = checked_verify_pinned_source_bytes_with_workspace(
        pin,
        "test.safetensors",
        &[132, 134],
        &raw,
        &mut SelectedSourceWorkspace::new(),
    )
    .unwrap();
    assert_eq!(owner.checked_metadata_backing_bytes().unwrap(), metadata);
}

#[test]
fn fixed_owner_erases_each_element_before_deallocation() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    // A Zeroize implementation need not implement DefaultIsZeroes, so generic
    // erasure must call each element rather than require slice specialization.
    struct Probe<'a>(&'a AtomicUsize);
    impl Zeroize for Probe<'_> {
        fn zeroize(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let erased = AtomicUsize::new(0);
    let owner = ZeroizingFixed {
        values: Box::new([Probe(&erased), Probe(&erased), Probe(&erased)]),
    };
    drop(owner);
    assert_eq!(erased.load(Ordering::SeqCst), 3);

    let mut bytes = ZeroizingFixed::<u8, 4>::new_zeroed();
    bytes.copy_from_slice(&[1, 2, 3, 4]);
    bytes.zeroize();
    assert_eq!(&bytes[..], &[0; 4]);
}

fn tuple() -> SelectedSourceTuple {
    let mut outputs = Vec::with_capacity(17);
    outputs.push(1);
    let mut inner = Vec::with_capacity(31);
    inner.extend([0, 1]);
    let mut checkpoint = String::with_capacity(127);
    checkpoint.push_str("independently-trusted-test-checkpoint");
    SelectedSourceTuple {
        checkpoint,
        immutable_revision: "0123456789abcdef0123456789abcdef01234567".into(),
        tensor: "projection.weight".into(),
        original_shape: [2, 2],
        output_coordinates: outputs,
        inner_coordinates: inner,
        output_input_row_major: true,
        dtype: SourceDtype::Bf16,
        endian: SourceEndian::Little,
        tensor_offset: 128,
        expected_config: [1; 32],
        expected_semantic_plan: [2; 32],
    }
}
fn expected(tuple: SelectedSourceTuple, raw: &[u8]) -> ExpectedSelectedSource {
    ExpectedSelectedSource::from_trusted_publisher(
        NamedDigest {
            id: "test.safetensors".into(),
            digest: Digest::from_sha256(Sha256::digest(raw).into()),
        },
        tuple,
        raw.len(),
    )
    .unwrap()
}
fn raw() -> Vec<u8> {
    [0x3e80_u16, 0xbe80]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect()
}

#[test]
fn neutral_owner_verifies_bytes_and_prices_actual_metadata_capacities() {
    let raw = raw();
    let tuple = tuple();
    let supplied_tuple = tuple.clone();
    let original_checkpoint_capacity = tuple.checkpoint.capacity();
    let original_output_capacity = tuple.output_coordinates.capacity();
    let expected = expected(tuple, &raw);
    let quote = expected.import_quote().unwrap();
    let tuple_capacities = expected.tuple().metadata_backing_bytes().unwrap();
    assert_eq!(
        expected.tuple().checkpoint.capacity(),
        original_checkpoint_capacity
    );
    assert_eq!(
        expected.tuple().output_coordinates.capacity(),
        original_output_capacity
    );
    assert_eq!(
        quote.metadata_backing_bytes,
        tuple_capacities + expected.artifact_name.capacity()
    );
    let owner = verify_selected_source(
        expected,
        SelectedSourceBytes {
            artifact_name: "test.safetensors",
            tuple: &supplied_tuple,
            scalar_artifact_offsets: &[132, 134],
            bytes: &raw,
        },
    )
    .unwrap();
    assert_eq!(quote, owner.resources().unwrap());
    assert_eq!(owner.bytes(), raw);
    assert_eq!(owner.finite_value(0).unwrap(), 0.25);
    assert_eq!(owner.finite_value(1).unwrap(), -0.25);
    assert_eq!(quote.source_byte_capacity, 1024);
    assert_eq!(quote.arc_backing_bytes, 0);
    assert_eq!(
        quote.import_overlap_bytes,
        quote.retained_bytes + quote.import_scratch_bytes
    );
}

#[test]
fn frozen_config_plan_and_original_source_mapping_cannot_be_redeclared() {
    let raw = raw();
    let original = tuple();
    for case in 0..7 {
        let mut changed = original.clone();
        match case {
            0 => changed.expected_config = [3; 32],
            1 => changed.expected_semantic_plan = [3; 32],
            2 => changed.immutable_revision = "ffffffffffffffffffffffffffffffffffffffff".into(),
            3 => changed.tensor_offset = 130,
            4 => changed.output_input_row_major = false,
            5 => changed.endian = SourceEndian::Big,
            _ => changed.inner_coordinates.swap(0, 1),
        }
        assert!(verify_selected_source(
            expected(original.clone(), &raw),
            SelectedSourceBytes {
                artifact_name: "test.safetensors",
                tuple: &changed,
                scalar_artifact_offsets: &[132, 134],
                bytes: &raw
            }
        )
        .is_err());
    }
}

#[test]
fn actual_byte_offset_and_nonfinite_mutations_fail() {
    let raw = raw();
    let tuple = tuple();
    let mut changed = raw.clone();
    changed[1] ^= 1;
    assert!(verify_selected_source(
        expected(tuple.clone(), &raw),
        SelectedSourceBytes {
            artifact_name: "test.safetensors",
            tuple: &tuple,
            scalar_artifact_offsets: &[132, 134],
            bytes: &changed
        }
    )
    .is_err());
    assert!(verify_selected_source(
        expected(tuple.clone(), &raw),
        SelectedSourceBytes {
            artifact_name: "test.safetensors",
            tuple: &tuple,
            scalar_artifact_offsets: &[130, 134],
            bytes: &raw
        }
    )
    .is_err());
    let mut nonfinite = raw;
    nonfinite[..2].copy_from_slice(&0x7f80_u16.to_le_bytes());
    assert!(verify_selected_source(
        expected(tuple.clone(), &nonfinite),
        SelectedSourceBytes {
            artifact_name: "test.safetensors",
            tuple: &tuple,
            scalar_artifact_offsets: &[132, 134],
            bytes: &nonfinite
        }
    )
    .is_err());
}

#[test]
fn finite_f32_overflow_is_not_a_generic_quantization_policy() {
    let mut tuple = tuple();
    tuple.dtype = SourceDtype::F32;
    let raw: Vec<u8> = [f32::MAX, -f32::MAX]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let owner = verify_selected_source(
        expected(tuple.clone(), &raw),
        SelectedSourceBytes {
            artifact_name: "test.safetensors",
            tuple: &tuple,
            scalar_artifact_offsets: &[136, 140],
            bytes: &raw,
        },
    )
    .unwrap();
    assert_eq!(owner.finite_value(0).unwrap(), f64::from(f32::MAX));
}

#[test]
fn parsed_digest_has_no_hidden_retained_string_capacity() {
    let digest = Digest::from_sha256([123; 32]);
    assert_eq!(digest_identity(&digest).unwrap(), [123; 32]);
    let mut tuple = tuple();
    tuple.checkpoint = String::with_capacity(65537);
    tuple.checkpoint.push('x');
    let raw = raw();
    assert!(ExpectedSelectedSource::from_trusted_publisher(
        NamedDigest {
            id: "test".into(),
            digest
        },
        tuple,
        raw.len()
    )
    .is_err());
    assert!(sum_bytes([usize::MAX, 1]).is_err());
}

#[test]
fn fixed_backing_wipes_before_field_deallocation_and_external_guard_refund() {
    use std::{cell::RefCell, rc::Rc};
    struct Probe {
        wiped: bool,
        events: Rc<RefCell<Vec<&'static str>>>,
    }
    impl Zeroize for Probe {
        fn zeroize(&mut self) {
            self.wiped = true;
            self.events.borrow_mut().push("wipe");
        }
    }
    impl Drop for Probe {
        fn drop(&mut self) {
            assert!(self.wiped);
            self.events.borrow_mut().push("element-drop");
        }
    }
    struct Guard(Rc<RefCell<Vec<&'static str>>>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.borrow_mut().push("refund");
        }
    }
    struct Guarded {
        _owner: ZeroizingFixed<Probe, 2>,
        _guard: Guard,
    }
    let events = Rc::new(RefCell::new(Vec::new()));
    {
        let _guarded = Guarded {
            _owner: ZeroizingFixed {
                values: Box::new([
                    Probe {
                        wiped: false,
                        events: events.clone(),
                    },
                    Probe {
                        wiped: false,
                        events: events.clone(),
                    },
                ]),
            },
            _guard: Guard(events.clone()),
        };
    }
    assert_eq!(
        &*events.borrow(),
        &["wipe", "wipe", "element-drop", "element-drop", "refund"]
    );
    let mut bytes: ZeroizingFixed<u8, 1024> = ZeroizingFixed::new_zeroed();
    bytes.fill(255);
    bytes.zeroize();
    assert!(bytes.iter().all(|&v| v == 0));
    let mut coefficients: ZeroizingFixed<i64, 256> = ZeroizingFixed::new_zeroed();
    coefficients.fill(-17);
    coefficients.zeroize();
    assert!(coefficients.iter().all(|&v| v == 0));
    assert_eq!(
        size_of::<ZeroizingFixed<u8, 1024>>(),
        size_of::<Box<[u8; 1024]>>()
    );
    assert_eq!(
        size_of::<ZeroizingFixed<i64, 256>>(),
        size_of::<Box<[i64; 256]>>()
    );
}
