//! Metadata-only fixtures: no Python, plan parsing, native execution or source I/O.
use super::*;

#[derive(Clone)]
struct Inputs {
    artifact: NamedDigest,
    selection: CheckpointTensorSelection,
    config: Digest,
    semantic_plan: Digest,
    selected_bytes: usize,
    dtype: SourceDtype,
    endian: SourceEndian,
    tensor_offset: u64,
}

impl Inputs {
    fn new() -> Self {
        let digest: [u8; 32] = Sha256::digest(raw()).into();
        Self {
            artifact: NamedDigest {
                id: "fixture-original.safetensors".into(),
                digest: Digest::from_sha256(digest),
            },
            selection: CheckpointTensorSelection {
                checkpoint: "synthetic-publisher-premise".into(),
                immutable_revision: "0123456789abcdef0123456789abcdef01234567".into(),
                content_sha256: digest,
                tensor: "layer.weight".into(),
                original_weight_shape: [4, 8],
                weight_layout: "output_input_row_major".into(),
                output_coordinates: vec![3, 1],
                inner_coordinates: vec![7, 2],
                input_tensor: "hidden.0".into(),
                original_input_shape: [2, 8],
                input_layout: "row_inner_row_major".into(),
                input_row_coordinates: vec![1],
                encoding: SIGMA_LINEAR_ACTIVATION_ENCODING.into(),
            },
            config: Digest::from_sha256([7; 32]),
            semantic_plan: Digest::from_sha256([9; 32]),
            selected_bytes: 8,
            dtype: SourceDtype::Bf16,
            endian: SourceEndian::Little,
            tensor_offset: 128,
        }
    }

    fn quote(&self, baseline: usize) -> Result<SigmaSourceConstructionResources, &'static str> {
        ExpectedSigmaSourcePin::checked_construction_quote(
            &self.artifact,
            &self.selection,
            &self.config,
            &self.semantic_plan,
            baseline,
        )
    }

    fn admitted(self, baseline: usize, cap: usize) -> Result<ExpectedSigmaSourcePin, &'static str> {
        ExpectedSigmaSourcePin::from_trusted_publisher_admitted(
            self.artifact,
            self.selected_bytes,
            self.selection,
            self.dtype,
            self.endian,
            self.tensor_offset,
            &self.config,
            &self.semantic_plan,
            baseline,
            cap,
        )
    }

    fn legacy(self) -> Result<ExpectedSigmaSourcePin, &'static str> {
        ExpectedSigmaSourcePin::checked_from_trusted_publisher(
            self.artifact,
            self.selected_bytes,
            self.selection,
            self.dtype,
            self.endian,
            self.tensor_offset,
            &self.config,
            &self.semantic_plan,
        )
    }
}

fn raw() -> [u8; 8] {
    // BF16: +0.25, -0.25, -2^-13, +2^-13, in selected coordinate order.
    [0x80, 0x3e, 0x80, 0xbe, 0x00, 0xb9, 0x00, 0x39]
}

fn copies() -> usize {
    SOURCE_METADATA_COPY_FACTORIES.with(|n| n.get())
}

fn reset_copies() {
    SOURCE_METADATA_COPY_FACTORIES.with(|n| n.set(0));
}

#[test]
fn construction_quote_and_one_short_precede_all_metadata_copy_factories() {
    let input = Inputs::new();
    let snapshot = input.clone();
    let checkpoint_pointer = input.selection.checkpoint.as_ptr();
    let coordinates_pointer = input.selection.output_coordinates.as_ptr();
    let baseline = 137;
    reset_copies();
    let quote = input.quote(baseline).unwrap();
    assert_eq!(copies(), 0);
    assert_eq!(input.selection, snapshot.selection);
    assert_eq!(input.artifact, snapshot.artifact);
    assert_eq!(input.selection.checkpoint.as_ptr(), checkpoint_pointer);
    assert_eq!(
        input.selection.output_coordinates.as_ptr(),
        coordinates_pointer
    );
    assert_eq!(
        quote.peak_bytes,
        baseline + quote.construction_overlap_bytes
    );
    assert_eq!(
        quote.construction_overlap_bytes,
        quote.retained_pin_bytes
            + input.artifact.digest.backing_capacity_bytes()
            + input.config.backing_capacity_bytes()
            + input.semantic_plan.backing_capacity_bytes()
            + quote.construction_scratch_bytes
    );
    assert_eq!(
        input.admitted(baseline, quote.peak_bytes - 1).err(),
        Some("expected source construction capacity limit exceeded")
    );
    assert_eq!(copies(), 0);

    let input = Inputs::new();
    let quote = input.quote(baseline).unwrap();
    let checkpoint_pointer = input.selection.checkpoint.as_ptr();
    let coordinates_pointer = input.selection.output_coordinates.as_ptr();
    let artifact_pointer = input.artifact.id.as_ptr();
    let pin = input.admitted(baseline, quote.peak_bytes).unwrap();
    assert_eq!(copies(), 5);
    // Originals move unchanged; only the three selected strings and two ordered
    // coordinate slices get new exact logical backings.
    assert_eq!(pin.selection.checkpoint.as_ptr(), checkpoint_pointer);
    assert_eq!(
        pin.selection.output_coordinates.as_ptr(),
        coordinates_pointer
    );
    assert_eq!(pin.selected.artifact_name().as_ptr(), artifact_pointer);
    assert_eq!(
        quote.retained_pin_bytes,
        size_of::<ExpectedSigmaSourcePin>()
            + pin.selection.checked_metadata_backing_bytes().unwrap()
            + pin.selected.checked_metadata_backing_bytes().unwrap()
    );
}

#[test]
fn constructor_requotes_changed_input_capacity_before_copying() {
    let mut input = Inputs::new();
    let old = input.quote(0).unwrap();
    input.selection.checkpoint.reserve_exact(4096);
    reset_copies();
    let current = input.quote(0).unwrap();
    assert!(current.peak_bytes > old.peak_bytes);
    assert_eq!(
        input.admitted(0, old.peak_bytes).err(),
        Some("expected source construction capacity limit exceeded")
    );
    assert_eq!(copies(), 0);
}

#[test]
fn spare_input_capacities_are_charged_but_never_inherited_by_tuple_copies() {
    let mut input = Inputs::new();
    let before = input.quote(0).unwrap();
    input.artifact.id.reserve_exact(53);
    for text in [
        &mut input.selection.checkpoint,
        &mut input.selection.immutable_revision,
        &mut input.selection.tensor,
        &mut input.selection.weight_layout,
        &mut input.selection.input_tensor,
        &mut input.selection.input_layout,
        &mut input.selection.encoding,
    ] {
        text.reserve_exact(31);
    }
    for coordinates in [
        &mut input.selection.output_coordinates,
        &mut input.selection.inner_coordinates,
        &mut input.selection.input_row_coordinates,
    ] {
        coordinates.reserve_exact(17);
    }
    reset_copies();
    let after = input.quote(0).unwrap();
    assert_eq!(copies(), 0);
    let input_backing = input.selection.checked_metadata_backing_bytes().unwrap()
        + input.artifact.id.capacity()
        + input.artifact.digest.backing_capacity_bytes()
        + input.config.backing_capacity_bytes()
        + input.semantic_plan.backing_capacity_bytes();
    assert_eq!(after.input_metadata_backing_bytes, input_backing);
    assert!(after.input_metadata_backing_bytes > before.input_metadata_backing_bytes);
    assert_eq!(
        after.tuple_copy_backing_bytes,
        before.tuple_copy_backing_bytes
    );
    assert_eq!(
        after.construction_scratch_bytes,
        before.construction_scratch_bytes
    );
    assert_eq!(
        after.peak_bytes - before.peak_bytes,
        after.input_metadata_backing_bytes - before.input_metadata_backing_bytes
    );
    let original_selection_capacity = input.selection.checked_metadata_backing_bytes().unwrap();
    let original_artifact_capacity = input.artifact.id.capacity();
    let pin = input.admitted(0, after.peak_bytes).unwrap();
    assert_eq!(copies(), 5);
    assert_eq!(
        pin.selection.checked_metadata_backing_bytes().unwrap(),
        original_selection_capacity
    );
    let tuple = pin.selected.tuple();
    for text in [&tuple.checkpoint, &tuple.immutable_revision, &tuple.tensor] {
        assert_eq!(text.capacity(), text.len());
    }
    for coordinates in [&tuple.output_coordinates, &tuple.inner_coordinates] {
        assert_eq!(coordinates.capacity(), coordinates.len());
    }
    assert_eq!(
        tuple.checked_metadata_backing_bytes().unwrap(),
        after.tuple_copy_backing_bytes
    );
    assert_eq!(
        pin.selected.checked_metadata_backing_bytes().unwrap(),
        original_artifact_capacity + after.tuple_copy_backing_bytes
    );
}

#[test]
fn original_validation_authority_and_identity_error_order_are_preserved() {
    for case in 0..9 {
        let mut input = Inputs::new();
        let expected = match case {
            0 => {
                input.selection.encoding = "wrong".into();
                input.artifact.digest = Digest::from_sha256([0; 32]);
                "unsupported source layout or floor encoding"
            }
            1 => {
                input.artifact.digest = Digest::from_sha256([0; 32]);
                input.selection.immutable_revision = "not-a-revision".into();
                "inconsistent expected source digest"
            }
            2 => {
                input.selection.immutable_revision = "not-a-revision".into();
                input.artifact.id.clear();
                "unsupported source revision/layout/endian/offset"
            }
            3 => {
                input.endian = SourceEndian::Big;
                input.selected_bytes = usize::MAX;
                "unsupported source revision/layout/endian/offset"
            }
            4 => {
                input.selected_bytes = usize::MAX;
                "invalid independently expected source pin"
            }
            5 => {
                input.artifact.id.clear();
                "invalid independently expected source pin"
            }
            6 => {
                input.tensor_offset = u64::MAX - 1;
                "source offset overflow"
            }
            7 => {
                input.selection.output_coordinates[1] = input.selection.output_coordinates[0];
                input.artifact.digest = Digest::from_sha256([0; 32]);
                "invalid, duplicate or unbounded selected coordinates"
            }
            _ => {
                input.selection.original_weight_shape = [usize::MAX, 8];
                input.artifact.digest = Digest::from_sha256([0; 32]);
                "original weight geometry overflow"
            }
        };
        assert_eq!(input.clone().legacy().err(), Some(expected), "case {case}");
        reset_copies();
        assert_eq!(
            input.admitted(0, usize::MAX).err(),
            Some(expected),
            "case {case}"
        );
        // Only the checks that originally preceded tuple construction must
        // precede copies. Shared revision/byte/offset checks still run shared.
        assert_eq!(copies(), if matches!(case, 0 | 1 | 7 | 8) { 0 } else { 5 });
    }
}

#[test]
fn overflow_rejects_before_any_copy_or_owner_mutation() {
    let input = Inputs::new();
    reset_copies();
    assert_eq!(
        input.quote(usize::MAX).err(),
        Some("resource byte overflow")
    );
    assert_eq!(copies(), 0);
    assert_eq!(
        input.admitted(usize::MAX, usize::MAX).err(),
        Some("resource byte overflow")
    );
    assert_eq!(copies(), 0);
    assert_eq!(
        checked_coordinate_copy_bytes(usize::MAX),
        Err("vector backing overflow")
    );
    for (selection, artifact, digest, copied) in [
        (usize::MAX, 1, 0, 0),
        (0, usize::MAX, 1, 0),
        (0, 0, 0, usize::MAX),
    ] {
        assert_eq!(
            SigmaSourceConstructionResources::checked_new(0, selection, artifact, digest, copied)
                .err(),
            Some("resource byte overflow")
        );
    }
    assert_eq!(copies(), 0);
}

#[test]
fn original_selected_identity_offsets_bytes_and_binding_match_legacy() {
    let input = Inputs::new();
    let selection = input.selection.clone();
    let artifact_name = input.artifact.id.clone();
    let config = input.config.as_sha256();
    let semantic_plan = input.semantic_plan.as_sha256();
    let legacy = input.clone().legacy().unwrap();
    let quote = input.quote(0).unwrap();
    let admitted = input.admitted(0, quote.peak_bytes).unwrap();
    assert_eq!(admitted.selection(), &selection);
    assert_eq!(admitted.selected.tuple(), legacy.selected.tuple());
    assert_eq!(admitted.selected.artifact_name(), artifact_name);
    assert_eq!(admitted.selected.digest(), selection.content_sha256);
    assert_eq!(admitted.selected.tuple().expected_config, config);
    assert_eq!(
        admitted.selected.tuple().expected_semantic_plan,
        semantic_plan
    );
    let offsets = admitted.offsets().unwrap();
    assert_eq!(offsets, [190, 180, 158, 148]);
    assert_eq!(legacy.offsets().unwrap(), offsets);
    assert_eq!(admitted.selected.selected_bytes(), 8);
    let bytes = raw();
    let packet = || SigmaSourceBytes {
        artifact_name: &artifact_name,
        selection: &selection,
        dtype: SourceDtype::Bf16,
        endian: SourceEndian::Little,
        tensor_offset: 128,
        scalar_artifact_offsets: &offsets,
        bytes: &bytes,
    };
    let encoded = [1024, -1024, -1, 0];
    let admitted = checked_import_sigma_source_bytes(admitted, packet(), &encoded).unwrap();
    let legacy = checked_import_sigma_source_bytes(legacy, packet(), &encoded).unwrap();
    assert_eq!(admitted.source_binding(), legacy.source_binding());
    assert_eq!(admitted.verified.bytes(), bytes);
    assert_eq!(admitted.encoded_private_coefficients(), encoded);
    for index in 0..4 {
        assert_eq!(admitted.source_value(index), legacy.source_value(index));
    }
}

#[test]
fn selected_byte_and_metadata_capacity_limits_keep_shared_authority() {
    let mut input = Inputs::new();
    input.dtype = SourceDtype::F32;
    input.selection.original_weight_shape = [16, 16];
    input.selection.original_input_shape = [1, 16];
    input.selection.output_coordinates = (0..16).collect();
    input.selection.inner_coordinates = (0..16).collect();
    input.selection.input_row_coordinates = vec![0];
    input.selected_bytes = SELECTED_SOURCE_BYTE_CAPACITY;
    let mut invalid = input.clone();
    invalid.selected_bytes += 1;
    let quote = input.quote(0).unwrap();
    let pin = input.admitted(0, quote.peak_bytes).unwrap();
    assert_eq!(pin.selected.selected_bytes(), SELECTED_SOURCE_BYTE_CAPACITY);
    assert_eq!(
        invalid.clone().legacy().err(),
        Some("invalid independently expected source pin")
    );
    assert_eq!(
        invalid.admitted(0, usize::MAX).err(),
        Some("invalid independently expected source pin")
    );

    let mut input = Inputs::new();
    input.artifact.id.reserve_exact(64 * 1024);
    assert_eq!(
        input.admitted(0, usize::MAX).err(),
        Some("selected-source metadata capacity limit exceeded")
    );
    // Selection itself remains below 64 KiB, while retained selection + tuple
    // exceeds the existing SIGMA combined cap. No new acceptance authority.
    let mut input = Inputs::new();
    let fixed = input.selection.checked_metadata_backing_bytes().unwrap()
        - input.selection.checkpoint.capacity();
    let target = 64 * 1024 - fixed;
    let mut backing = "x".repeat(target).into_boxed_str().into_string();
    backing.clear();
    backing.push_str(&input.selection.checkpoint);
    input.selection.checkpoint = backing;
    assert_eq!(
        input.selection.checked_metadata_backing_bytes().unwrap(),
        64 * 1024
    );
    assert_eq!(
        input.admitted(0, usize::MAX).err(),
        Some("source metadata capacity limit exceeded")
    );
}
