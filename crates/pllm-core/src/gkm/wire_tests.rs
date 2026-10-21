use super::*;

fn fixture() -> (I16Matrix, FloatTable, Arc<Publication>, Client) {
    let matrix = I16Matrix::from_encoded(1, 2, &[3, -7], &[100; 2]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &(-0.0f32).to_le_bytes()).unwrap();
    let publication = Arc::new(
        Publication::publish(
            [4; 32],
            &[Stage {
                id: [1; 32],
                matrix: &matrix,
            }],
            &table,
        )
        .unwrap(),
    );
    let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
    (matrix, table, publication, client)
}

#[test]
fn worker_window_permit_preserves_original_full_requirement_and_prices_only_seeded_one_extra() {
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let matrix = I16Matrix::from_encoded(11, 13, &[3; 11 * 13], &[100; 13]).unwrap();
        let table = FloatTable::from_le_bytes(17, 7, &[0; 17 * 7 * 4]).unwrap();
        let stages = [Stage {
            id: [1; 32],
            matrix: &matrix,
        }];
        let publication = Arc::new(
            Publication::publish_with_capacity([4; 32], &stages, &table, capacity).unwrap(),
        );
        let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
        let plan = [Step::Linear([1; 32]), Step::Lookup];
        // Original wire admission formula, retaining the actual current inline
        // owner size for every role/profile. No seeded endpoint allowance here.
        let original = plan.len() * size_of::<Step>()
            + (13usize * 8 + 11 * 16).max(7 * 4)
            + size_of::<WorkerWireWindow>()
            + 1024;
        for seeded in [false, true] {
            let mut window = if seeded {
                client.begin_seeded_wire([5; 32], &plan)
            } else {
                client.begin_wire([5; 32], &plan)
            }
            .unwrap();
            let starts = window.start_frames().unwrap();
            for (index, role) in [Worker::Zero, Worker::One].into_iter().enumerate() {
                let mut owner = WorkerWire::new(
                    Arc::clone(&publication),
                    publication.digest(),
                    role,
                    &stages,
                    &table,
                )
                .unwrap();
                let worker = owner.begin(starts[index].as_bytes()).unwrap();
                let extra = if seeded && role == Worker::One {
                    publication.resources.seeded_window_extra_bytes
                } else {
                    0
                };
                assert_eq!(worker._permit.bytes, original + extra);
                assert_eq!(worker.seeded.is_some(), seeded && role == Worker::One);
            }
        }
    }
}

#[test]
fn worker_uses_native_cumulative_window_step_and_shared_live_admission() {
    let (matrix, table, publication, client) = fixture();
    for exhausted_steps in [false, true] {
        let mut worker = WorkerWire::new(
            Arc::clone(&publication),
            publication.digest(),
            Worker::Zero,
            &[Stage {
                id: [1; 32],
                matrix: &matrix,
            }],
            &table,
        )
        .unwrap();
        let source = worker.source.as_ref().unwrap();
        if exhausted_steps {
            source
                .budget
                .steps
                .store(MAX_RESERVED_STEPS, Ordering::Release);
        } else {
            source.budget.windows.store(MAX_WINDOWS, Ordering::Release);
        }
        let mut window = client.begin_wire([5; 32], &[Step::Lookup]).unwrap();
        let start = window.start_frames().unwrap();
        assert!(matches!(
            worker.begin(start[0].as_bytes()),
            Err(Error::Limit)
        ));
        assert!(matches!(
            worker.begin(start[0].as_bytes()),
            Err(Error::Retired)
        ));
    }
    let mut worker = WorkerWire::new(
        Arc::clone(&publication),
        publication.digest(),
        Worker::Zero,
        &[Stage {
            id: [1; 32],
            matrix: &matrix,
        }],
        &table,
    )
    .unwrap();
    let held: Vec<_> = (0..MAX_LIVE_WINDOWS - 1)
        .map(|_| client.begin([5; 32], &[Step::Lookup]).unwrap())
        .collect();
    let mut window = client.begin_wire([5; 32], &[Step::Lookup]).unwrap();
    let start = window.start_frames().unwrap();
    assert_eq!(publication.live_windows(), MAX_LIVE_WINDOWS);
    assert!(matches!(
        worker.begin(start[0].as_bytes()),
        Err(Error::Limit)
    ));
    drop(held);
    assert!(matches!(
        worker.begin(start[0].as_bytes()),
        Err(Error::Retired)
    ));
}

#[test]
fn every_new_payload_allocation_failure_is_fallible_and_retires_its_owner() {
    let (matrix, table, publication, client) = fixture();
    let artifact = publication.export_wire().unwrap();
    // Import owns no live windows. A failed staged allocation must return its
    // complete original publication reservation, not just its wire allowance.
    let mut imported = false;
    for index in 0..32 {
        let fail = payload::failure::after(index);
        let result = Publication::import_wire(artifact.as_bytes(), publication.digest());
        drop(fail);
        match result {
            Ok(value) => {
                drop(value);
                imported = true;
                break;
            }
            Err(Error::Allocation) => (),
            Err(error) => panic!("unexpected {error:?}"),
        }
    }
    assert!(imported);
    let mut constructed = false;
    for index in 0..32 {
        let fail = payload::failure::after(index);
        let result = WorkerWire::new(
            Arc::clone(&publication),
            publication.digest(),
            Worker::Zero,
            &[Stage {
                id: [1; 32],
                matrix: &matrix,
            }],
            &table,
        );
        drop(fail);
        match result {
            Ok(value) => {
                drop(value);
                constructed = true;
                break;
            }
            Err(Error::Allocation) => (),
            Err(error) => panic!("unexpected {error:?}"),
        }
        assert_eq!(publication.clients.load(Ordering::Acquire), 1);
    }
    assert!(constructed);
    for phase in 0..5 {
        let mut succeeded = false;
        for index in 0..32 {
            let mut worker = WorkerWire::new(
                Arc::clone(&publication),
                publication.digest(),
                Worker::Zero,
                &[Stage {
                    id: [1; 32],
                    matrix: &matrix,
                }],
                &table,
            )
            .unwrap();
            let mut peer = WorkerWire::new(
                Arc::clone(&publication),
                publication.digest(),
                Worker::One,
                &[Stage {
                    id: [1; 32],
                    matrix: &matrix,
                }],
                &table,
            )
            .unwrap();
            let mut window = client
                .begin_wire([5; 32], &[Step::Linear([1; 32])])
                .unwrap();
            if phase == 0 {
                let fail = payload::failure::after(index);
                let result = window.start_frames();
                drop(fail);
                if result.is_ok() {
                    succeeded = true;
                    break;
                }
                assert!(matches!(result, Err(Error::Allocation)));
                assert!(matches!(window.start_frames(), Err(Error::Retired)));
                continue;
            }
            let start = window.start_frames().unwrap();
            if phase == 1 {
                let fail = payload::failure::after(index);
                let result = worker.begin(start[0].as_bytes());
                drop(fail);
                if result.is_ok() {
                    succeeded = true;
                    break;
                }
                assert!(matches!(result, Err(Error::Allocation)));
                assert!(matches!(
                    worker.begin(start[0].as_bytes()),
                    Err(Error::Retired)
                ));
                continue;
            }
            let mut left = worker.begin(start[0].as_bytes()).unwrap();
            let mut right = peer.begin(start[1].as_bytes()).unwrap();
            if phase == 2 {
                let fail = payload::failure::after(index);
                let result = window.linear(&[1, -2]);
                drop(fail);
                if result.is_ok() {
                    succeeded = true;
                    break;
                }
                assert!(matches!(result, Err(Error::Allocation)));
                assert!(matches!(window.linear(&[1, -2]), Err(Error::Retired)));
                continue;
            }
            let request = window.linear(&[1, -2]).unwrap();
            if phase == 3 {
                let fail = payload::failure::after(index);
                let result = left.execute(request[0].as_bytes());
                drop(fail);
                if result.is_ok() {
                    succeeded = true;
                    break;
                }
                assert!(matches!(result, Err(Error::Allocation)));
                assert!(matches!(
                    left.execute(request[0].as_bytes()),
                    Err(Error::Retired)
                ));
                continue;
            }
            let a = left.execute(request[0].as_bytes()).unwrap();
            let b = right.execute(request[1].as_bytes()).unwrap();
            let fail = payload::failure::after(index);
            let result = window.accept_replies(a.as_bytes(), b.as_bytes());
            drop(fail);
            if result.is_ok() {
                succeeded = true;
                break;
            }
            assert!(matches!(result, Err(Error::Allocation)));
            assert!(matches!(
                window.accept_replies(a.as_bytes(), b.as_bytes()),
                Err(Error::Retired)
            ));
        }
        assert!(succeeded);
        assert_eq!(publication.live_windows(), 0);
        assert_eq!(publication.clients.load(Ordering::Acquire), 1);
    }
}
