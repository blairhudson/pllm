use super::*;
use crate::gkm::{
    tests::ScriptEntropy, Client, Step, Worker, WorkerWire, MAX_CLIENTS_PER_PUBLICATION,
};
use std::sync::Arc;

fn fixture() -> (I16Matrix, FloatTable) {
    (
        I16Matrix::from_encoded(2, 3, &[3, -7, 11, -2, 0, 5], &[100; 3]).unwrap(),
        FloatTable::from_le_bytes(9, 1, &[0; 9 * 4]).unwrap(),
    )
}

#[test]
fn streamed_components_preserve_generic_pricing_and_exact_owned_overlap_difference() {
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let generic =
            PublicationResources::for_shapes_with_capacity(&[(2, 3)], 9, 1, capacity).unwrap();
        let streamed =
            PublicationResources::for_streamed_single_window(&[(2, 3)], 9, 1, capacity).unwrap();
        assert_eq!(generic.ownership, PublicationOwnership::General);
        assert_eq!(generic.max_import_workspace_bytes, 2 * 2 * 3);
        assert_eq!(
            generic.max_live_window_payload_bytes,
            MAX_LIVE_WINDOWS as usize * generic.max_window_payload_bytes
        );
        let source =
            generic.source_weight_bytes + generic.source_bound_bytes + generic.source_table_bytes;
        assert_eq!(
            generic.admitted_payload_bytes,
            source
                + generic.max_import_workspace_bytes
                + generic.native_publication_payload_bytes
                + generic.owner_scalar_workspace_bytes
                + generic.max_live_window_payload_bytes
        );
        assert_eq!(
            streamed.admitted_payload_bytes,
            source
                + STREAMED_IMPORT_WORKSPACE_BYTES
                + generic.native_publication_payload_bytes
                + generic.owner_scalar_workspace_bytes
                + generic.max_window_payload_bytes
        );
        assert_eq!(
            streamed.max_live_window_payload_bytes,
            generic.max_window_payload_bytes
        );
        assert_eq!(
            streamed.seeded_window_extra_bytes,
            generic.seeded_window_extra_bytes
        );
    }
    let mut shapes = Vec::new();
    for _ in 0..24 {
        shapes.extend([(1152, 896), (896, 896), (9728, 896), (896, 4864)]);
    }
    shapes.push((151_936, 896));
    let generic =
        PublicationResources::for_shapes_with_capacity(&shapes, 151_936, 896, Capacity::Decoder)
            .unwrap();
    let streamed =
        PublicationResources::for_streamed_single_window(&shapes, 151_936, 896, Capacity::Decoder)
            .unwrap();
    assert_eq!(
        generic.source_weight_bytes + generic.source_bound_bytes + generic.source_table_bytes,
        1_537_566_720
    );
    assert_eq!(generic.max_import_workspace_bytes, 272_269_312);
    assert_eq!(
        generic.admitted_payload_bytes - streamed.admitted_payload_bytes,
        generic.max_import_workspace_bytes + 7 * generic.max_window_payload_bytes
            - STREAMED_IMPORT_WORKSPACE_BYTES
    );
    let fail = payload::failure::after(0);
    for shapes in [&[(usize::MAX, 1)][..], &[(1, usize::MAX)][..], &[][..]] {
        assert!(matches!(
            PublicationResources::for_streamed_single_window(shapes, 9, 1, Capacity::Decoder),
            Err(Error::Limit)
        ));
    }
    for (vocabulary, width) in [(usize::MAX, 1), (1, usize::MAX)] {
        assert!(matches!(
            PublicationResources::for_streamed_single_window(
                &[(2, 3)],
                vocabulary,
                width,
                Capacity::Decoder
            ),
            Err(Error::Limit)
        ));
    }
    drop(fail);
}

#[test]
fn streamed_reservation_is_exact_one_lifetime_window_with_retained_frame_leases() {
    static POOL: AdmissionPool = AdmissionPool::new(16 << 20);
    let (matrix, table) = fixture();
    let stages = [Stage {
        id: [1; 32],
        matrix: &matrix,
    }];
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            let publication = Arc::new(
                Publication::publish_owned_in_pool(
                    [4; 32],
                    &stages,
                    &table,
                    &mut ScriptEntropy::new(&[2, 3]),
                    &POOL,
                    capacity,
                    PublicationOwnership::StreamedSingleWindow,
                )
                .unwrap(),
            );
            let priced = publication.resources();
            assert_eq!(publication._admission.bytes, priced.admitted_payload_bytes);
            assert_eq!(
                POOL.bytes.load(Ordering::Acquire),
                priced.admitted_payload_bytes
            );
            let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
            assert!(matches!(
                Client::new(Arc::clone(&publication), publication.digest()),
                Err(Error::Limit)
            ));
            let plan = [Step::Linear([1; 32])];
            let mut window = if seeded {
                client.begin_seeded_wire([5; 32], &plan)
            } else {
                client.begin_wire([5; 32], &plan)
            }
            .unwrap();
            assert!(matches!(client.begin([5; 32], &plan), Err(Error::Retired)));
            let starts = window.start_frames().unwrap();
            let public_frame = publication.export_wire().unwrap();
            let remote = Arc::new(
                Publication::import_wire_streamed_single_window(
                    public_frame.as_bytes(),
                    publication.digest(),
                    capacity,
                )
                .unwrap(),
            );
            let mut owner = WorkerWire::new(
                Arc::clone(&remote),
                remote.digest(),
                Worker::One,
                &stages,
                &table,
            )
            .unwrap();
            let mut worker = owner.begin(starts[1].as_bytes()).unwrap();
            let requests = window.linear(&[1, 2, 3]).unwrap();
            let reply = worker.execute(requests[1].as_bytes()).unwrap();
            assert!(worker.execute(requests[1].as_bytes()).is_err());
            drop((worker, owner));
            assert_eq!(remote.live_windows(), 1);
            assert_eq!(remote._admission.bytes, priced.admitted_payload_bytes);
            drop(reply);
            assert_eq!(remote.live_windows(), 0);
            let mut replacement_worker = WorkerWire::new(
                Arc::clone(&remote),
                remote.digest(),
                Worker::One,
                &stages,
                &table,
            )
            .unwrap();
            assert!(matches!(
                replacement_worker.begin(starts[1].as_bytes()),
                Err(Error::Retired)
            ));
            window.abort();
            drop((window, client));
            assert!(matches!(
                Client::new(Arc::clone(&publication), publication.digest()),
                Err(Error::Limit)
            ));
            assert_eq!(
                POOL.bytes.load(Ordering::Acquire),
                priced.admitted_payload_bytes
            );
            drop(starts);
            assert_eq!(
                POOL.bytes.load(Ordering::Acquire),
                priced.admitted_payload_bytes
            );
            drop(requests);
            let replacement = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
            assert!(matches!(
                replacement.begin([5; 32], &plan),
                Err(Error::Retired)
            ));
            drop((replacement, publication));
            assert_eq!(POOL.bytes.load(Ordering::Acquire), 0);
        }
    }
    // General publications retain the original multi-handle policy.
    let general = Arc::new(Publication::publish([4; 32], &stages, &table).unwrap());
    let clients: Vec<_> = (0..MAX_CLIENTS_PER_PUBLICATION)
        .map(|_| Client::new(Arc::clone(&general), general.digest()).unwrap())
        .collect();
    assert_eq!(clients.len(), MAX_CLIENTS_PER_PUBLICATION as usize);
}

#[test]
fn streamed_publication_failures_refund_actual_price_and_import_checks_metadata() {
    static POOL: AdmissionPool = AdmissionPool::new(16 << 20);
    let (matrix, table) = fixture();
    let stages = [Stage {
        id: [1; 32],
        matrix: &matrix,
    }];
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let mut succeeded = false;
        for after in 0..32 {
            let fail = payload::failure::after(after);
            let result = Publication::publish_owned_in_pool(
                [4; 32],
                &stages,
                &table,
                &mut ScriptEntropy::new(&[2, 3]),
                &POOL,
                capacity,
                PublicationOwnership::StreamedSingleWindow,
            );
            drop(fail);
            match result {
                Err(Error::Allocation) => assert_eq!(POOL.bytes.load(Ordering::Acquire), 0),
                Ok(publication) => {
                    let frame = publication.export_wire().unwrap();
                    let imported = Publication::import_wire_streamed_single_window(
                        frame.as_bytes(),
                        publication.digest(),
                        capacity,
                    )
                    .unwrap();
                    assert_eq!(
                        imported.resources().admitted_payload_bytes,
                        publication.resources().admitted_payload_bytes
                    );
                    assert_eq!(
                        imported._admission.bytes,
                        imported.resources().admitted_payload_bytes
                    );
                    assert_eq!(imported.digest(), publication.digest());
                    let general = Publication::publish_in_pool_with_capacity(
                        [4; 32],
                        &stages,
                        &table,
                        &mut ScriptEntropy::new(&[2, 3]),
                        &PROCESS_PUBLICATIONS,
                        capacity,
                    )
                    .unwrap();
                    assert_eq!(general.digest(), publication.digest());
                    assert_eq!(general.export_wire().unwrap().as_bytes(), frame.as_bytes());
                    let mut imported_success = false;
                    for index in 0..32 {
                        let fail = payload::failure::after(index);
                        let result = Publication::import_wire_streamed_single_window(
                            frame.as_bytes(),
                            publication.digest(),
                            capacity,
                        );
                        drop(fail);
                        match result {
                            Err(Error::Allocation) => (),
                            Ok(value) => {
                                assert_eq!(
                                    value._admission.bytes,
                                    value.resources().admitted_payload_bytes
                                );
                                imported_success = true;
                                break;
                            }
                            Err(error) => panic!("unexpected import error {error:?}"),
                        }
                    }
                    assert!(imported_success);
                    let mut malformed = frame.as_bytes().to_vec();
                    malformed[72..80].copy_from_slice(&0u64.to_le_bytes());
                    // Trusted-fixture repin reaches structural metadata preflight,
                    // rather than stopping only at the ordinary digest check.
                    let mut hash = Sha256::new();
                    hash.update(b"pllm.gkm.owner-publication.v1");
                    hash.update(NUMERIC_PROFILE);
                    hash.update(&malformed[40..]);
                    let pin: [u8; 32] = hash.finalize().into();
                    malformed[8..40].copy_from_slice(&pin);
                    let fail = payload::failure::after(0);
                    assert!(matches!(
                        Publication::import_wire_streamed_single_window(&malformed, pin, capacity),
                        Err(Error::Limit)
                    ));
                    drop(fail);
                    let fail = payload::failure::after(0);
                    assert!(matches!(
                        Publication::import_wire_streamed_single_window(
                            &frame.as_bytes()[..frame.len() - 1],
                            publication.digest(),
                            capacity
                        ),
                        Err(Error::Binding)
                    ));
                    drop(fail);
                    drop(publication);
                    assert_eq!(POOL.bytes.load(Ordering::Acquire), 0);
                    succeeded = true;
                    break;
                }
                Err(error) => panic!("unexpected {error:?}"),
            }
        }
        assert!(succeeded);
        for seeded in [false, true] {
            let publication = Arc::new(
                Publication::publish_owned_in_pool(
                    [4; 32],
                    &stages,
                    &table,
                    &mut ScriptEntropy::new(&[2, 3]),
                    &POOL,
                    capacity,
                    PublicationOwnership::StreamedSingleWindow,
                )
                .unwrap(),
            );
            let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
            let fail = payload::failure::after(0);
            let result = if seeded {
                client.begin_seeded_wire([5; 32], &[crate::gkm::Step::Lookup])
            } else {
                client.begin_wire([5; 32], &[crate::gkm::Step::Lookup])
            };
            drop(fail);
            assert!(matches!(result, Err(Error::Allocation)));
            assert!(matches!(
                client.begin_wire([5; 32], &[crate::gkm::Step::Lookup]),
                Err(Error::Retired)
            ));
            assert_eq!(
                POOL.bytes.load(Ordering::Acquire),
                publication.resources().admitted_payload_bytes
            );
            drop((client, publication));
            assert_eq!(POOL.bytes.load(Ordering::Acquire), 0);
        }
    }
}
