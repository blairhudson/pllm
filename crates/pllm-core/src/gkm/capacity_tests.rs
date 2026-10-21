use super::publication::{AdmissionPool, PROCESS_PUBLICATIONS};
use super::tests::ScriptEntropy;
use super::*;
use std::sync::{atomic::Ordering, Arc};

#[test]
fn capacity_is_admission_policy_not_a_new_artifact_identity() {
    let weights = vec![257, -1, -129, 32767];
    let bounds = vec![100; 2];
    let matrix = I16Matrix::from_encoded(2, 2, &weights, &bounds).unwrap();
    let wide =
        I16Matrix::from_encoded_owned_with_capacity(2, 2, weights, bounds, Capacity::Decoder)
            .unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &(-0.0f32).to_le_bytes()).unwrap();
    let wide_table = FloatTable::from_le_bytes_owned_with_capacity(
        1,
        1,
        (-0.0f32).to_le_bytes().to_vec(),
        Capacity::Decoder,
    )
    .unwrap();
    assert_eq!(matrix.digest(), wide.digest());
    assert_eq!(table.digest(), wide_table.digest());
    assert_eq!(matrix.capacity(), Capacity::Capsule);
    assert_eq!(wide.capacity(), Capacity::Decoder);
    let make = |matrix, table, capacity| {
        Publication::publish_in_pool_with_capacity(
            [9; 32],
            &[Stage {
                id: [1; 32],
                matrix,
            }],
            table,
            &mut ScriptEntropy::new(&[2, 3]),
            &PROCESS_PUBLICATIONS,
            capacity,
        )
        .unwrap()
    };
    let capsule = make(&matrix, &table, Capacity::Capsule);
    let decoder = make(&wide, &wide_table, Capacity::Decoder);
    assert_eq!(capsule.digest(), decoder.digest());
    // Pre-extension v1 schema, independently recomputed with Python hashlib,
    // struct.pack and affine integer secp256k1 multiplication (r = 2, 3).
    for (actual, expected) in [
        (
            matrix.digest(),
            "bec435b24e0dac2a707073e17028042fc191fcca9fa488f85117c42c3471beb4",
        ),
        (
            table.digest(),
            "bffb0198ddec33fd85fd78db9b3f8f3ffa4075aa10a30fbd459f38d99c516b54",
        ),
        (
            capsule.digest(),
            "9b131b1ff854bc69a5defe315eeef10dba0ca95e3b9c48b9252b98269d5f5453",
        ),
    ] {
        assert_eq!(actual.as_slice(), super::tests::hex(expected));
    }
    let bytes = capsule.export_wire().unwrap();
    assert_eq!(bytes.as_bytes(), decoder.export_wire().unwrap().as_bytes());
    let imported = Publication::import_wire_with_capacity(
        bytes.as_bytes(),
        capsule.digest(),
        Capacity::Decoder,
    )
    .unwrap();
    assert_eq!(imported.digest(), capsule.digest());
    assert_eq!(imported.capacity(), Capacity::Decoder);
    assert_eq!(
        Publication::import_wire(bytes.as_bytes(), capsule.digest())
            .unwrap()
            .capacity(),
        Capacity::Capsule
    );
}

#[test]
fn full_qwen_geometry_is_admitted_without_importing_or_allocating_model_payloads() {
    let mut shapes = Vec::new();
    for _ in 0..24 {
        shapes.extend([(1152, 896), (896, 896), (9728, 896), (896, 4864)]);
    }
    shapes.push((151_936, 896));
    assert_eq!(shapes.len(), 97);
    let resources =
        PublicationResources::for_shapes_with_capacity(&shapes, 151_936, 896, Capacity::Decoder)
            .unwrap();
    assert_eq!(resources.capacity, Capacity::Decoder);
    assert_eq!(
        resources.source_weight_bytes + resources.source_table_bytes,
        1_532_461_056
    );
    assert_eq!(resources.window_accumulator_bytes, 20_422_656);
    assert_eq!(shapes.iter().map(|s| s.1).sum::<usize>(), 182_144);
    assert!(resources.max_window_payload_bytes < Capacity::Decoder.max_window_bytes());
    assert!(resources.admitted_payload_bytes < Capacity::Decoder.max_publication_bytes());
    assert!(resources.seeded_window_extra_bytes >= 4864 * 8 + 151_936usize.div_ceil(8));
    assert!(matches!(
        PublicationResources::for_shapes(&shapes, 151_936, 896),
        Err(Error::Limit)
    ));
    // Geometry-only admission must remain allocation-free, including rejection.
    let _fail = payload::failure::after(0);
    assert!(PublicationResources::for_shapes_with_capacity(
        &shapes,
        151_936,
        896,
        Capacity::Decoder
    )
    .is_ok());
}

#[test]
fn decoder_caps_and_overflow_reject_before_payload_allocation() {
    let cap = Capacity::Decoder;
    let _fail = payload::failure::after(0);
    for shape in [
        (usize::MAX, 1),
        (1, usize::MAX),
        (0, 1),
        (262_145, 1),
        (1, 16_385),
        (262_144, 1025),
    ] {
        assert!(matches!(
            PublicationResources::for_shapes_with_capacity(&[shape], 1, 1, cap),
            Err(Error::Limit)
        ));
        assert!(matches!(
            I16Matrix::from_encoded_owned_with_capacity(
                shape.0,
                shape.1,
                Vec::new(),
                Vec::new(),
                cap
            ),
            Err(Error::Limit)
        ));
    }
    for (vocab, width) in [
        (usize::MAX, 1),
        (1, usize::MAX),
        (262_145, 1),
        (1, 16_385),
        (262_144, 16_384),
    ] {
        assert!(matches!(
            FloatTable::from_le_bytes_owned_with_capacity(vocab, width, Vec::new(), cap),
            Err(Error::Limit)
        ));
    }
    // Each matrix separately fits 2^28 elements; their aggregate source does not.
    assert!(matches!(
        PublicationResources::for_shapes_with_capacity(&[(262_144, 1024); 4], 1, 1, cap),
        Err(Error::Limit)
    ));
    assert!(matches!(
        PublicationResources::for_shapes_with_capacity(&[(1, 1); 257], 1, 1, cap),
        Err(Error::Limit)
    ));
    // Thin matrices still pay for every secret accumulator and public bound.
    assert!(matches!(
        PublicationResources::for_shapes_with_capacity(&[(262_144, 1); 16], 1, 1, cap),
        Err(Error::Limit)
    ));
    assert!(matches!(
        PublicationResources::for_shapes_with_capacity(&[(1, 16_384); 256], 1, 1, cap),
        Err(Error::Limit)
    ));
    assert_eq!(cap.max_wire_frame_bytes(), 4 << 20);
    assert_eq!(cap.max_publication_wire_bytes(), 32 << 20);
    assert_eq!(cap.max_window_steps(), 65_536);
    assert_eq!(Capacity::Capsule.max_rows(), MAX_DIMENSION);
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        if let Some(oversized) = capacity.max_process_wire_bytes().checked_add(1) {
            for bytes in [oversized, usize::MAX] {
                assert!(matches!(
                    super::window::wire::WirePermit::acquire(bytes, capacity),
                    Err(Error::Limit)
                ));
            }
        }
    }
}

#[test]
fn decoder_publication_failures_refund_envelope_and_retained_responses_hold_it() {
    static POOL: AdmissionPool = AdmissionPool::new(128 << 20);
    let matrix = I16Matrix::from_encoded_owned_with_capacity(
        257,
        1,
        vec![1; 257],
        vec![4],
        Capacity::Decoder,
    )
    .unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &[0; 4]).unwrap();
    let mut success = false;
    for after in 0..32 {
        let result = {
            let mut rng = ScriptEntropy::new(&[2; 257]);
            let _fail = payload::failure::after(after);
            Publication::publish_in_pool_with_capacity(
                [9; 32],
                &[Stage {
                    id: [1; 32],
                    matrix: &matrix,
                }],
                &table,
                &mut rng,
                &POOL,
                Capacity::Decoder,
            )
        };
        match result {
            Err(Error::Allocation) => assert_eq!(POOL.bytes.load(Ordering::Acquire), 0),
            Ok(publication) => {
                let bytes = publication.resources().admitted_payload_bytes;
                let publication = Arc::new(publication);
                let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
                drop(client.begin_seeded([8; 32], &[Step::Lookup]).unwrap());
                let pending = client
                    .begin([8; 32], &[Step::Linear([1; 32])])
                    .unwrap()
                    .linear(&[3])
                    .unwrap();
                let response = matrix.evaluate(&pending.share(Worker::Zero)).unwrap();
                drop((pending, client, publication));
                assert_eq!(POOL.bytes.load(Ordering::Acquire), bytes);
                drop(response);
                assert_eq!(POOL.bytes.load(Ordering::Acquire), 0);
                success = true;
                break;
            }
            Err(other) => panic!("unexpected {other:?}"),
        }
    }
    assert!(success);
}
