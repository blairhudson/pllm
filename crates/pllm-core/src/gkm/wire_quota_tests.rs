use super::*;

fn fixture(
    capacity: Capacity,
    specialized: bool,
    seeded: bool,
    steps: usize,
) -> (Arc<Publication>, ClientWireWindow, I16Matrix, FloatTable) {
    fixture_plan(capacity, specialized, seeded, &vec![Step::Lookup; steps])
}

fn fixture_plan(
    capacity: Capacity,
    specialized: bool,
    seeded: bool,
    plan: &[Step],
) -> (Arc<Publication>, ClientWireWindow, I16Matrix, FloatTable) {
    let matrix = I16Matrix::from_encoded(1, 1, &[3], &[100]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &(-0.0f32).to_le_bytes()).unwrap();
    let stages = [Stage {
        id: [1; 32],
        matrix: &matrix,
    }];
    let publication = Arc::new(
        if specialized {
            Publication::publish_streamed_single_window([4; 32], &stages, &table, capacity)
        } else {
            Publication::publish_with_capacity([4; 32], &stages, &table, capacity)
        }
        .unwrap(),
    );
    let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
    let window = if seeded {
        client.begin_seeded_wire([5; 32], plan)
    } else {
        client.begin_wire([5; 32], plan)
    }
    .unwrap();
    (publication, window, matrix, table)
}

fn get_lease(window: &ClientWireWindow) -> Arc<Lease> {
    let Some(ClientState::Ready(window)) = &window.state else {
        panic!("not ready")
    };
    Arc::clone(&window._lease)
}

fn header(lease: &Arc<Lease>) -> WireBytes {
    let permit = FramePermit::single(lease, HEADER).unwrap();
    let capacity = lease.budget.publication.capacity;
    let mut out = WireBytes::allocate(HEADER, capacity).unwrap();
    out._frame = Some(permit);
    out
}

fn used(quota: &WireQuota) -> (usize, usize) {
    *quota.0.lock().unwrap()
}

#[test]
fn streamed_four_handles_fifth_sticky_and_count_byte_caps_are_independent() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            let (publication, mut window, _, _) = fixture(capacity, true, seeded, 8);
            let lease = get_lease(&window);
            let frames: Vec<_> = (0..4).map(|_| header(&lease)).collect();
            assert_eq!(
                used(&lease.frames),
                (4, 4 * WireBytes::charge(HEADER).unwrap())
            );
            assert!(matches!(
                FramePermit::single(&lease, HEADER),
                Err(Error::Limit)
            ));
            window.abort();
            assert_eq!(used(&lease.frames).0, 4);
            assert_eq!(publication.live_windows(), 1);
            drop(frames);
            assert_eq!(used(&lease.frames), (0, 0));
            assert!(matches!(
                FramePermit::single(&lease, HEADER),
                Err(Error::Retired)
            ));
            assert!(window.lookup(0).is_err());

            let (_, window, _, _) = fixture(capacity, true, seeded, 8);
            let lease = get_lease(&window);
            let f = capacity.max_wire_frame_bytes();
            let first = FramePermit::single(&lease, f).unwrap();
            let second = FramePermit::single(&lease, f).unwrap();
            let third = FramePermit::single(&lease, f).unwrap();
            assert_eq!(used(&lease.frames).0, 3);
            assert!(matches!(FramePermit::single(&lease, f), Err(Error::Limit)));
            // Count had room for four; bytes (including owner/control) did not.
            assert!(used(&lease.frames).1 < 4 * f);
            drop((first, second, third));
            assert_eq!(used(&lease.frames), (0, 0));
        }
    }
}

#[test]
fn streamed_starts_requests_pair_preflight_and_allocation_rollback() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            let (_, mut window, _, _) = fixture(capacity, true, seeded, 8);
            let lease = get_lease(&window);
            let starts = window.start_frames().unwrap();
            let requests = window.lookup(0).unwrap();
            assert_eq!(used(&lease.frames).0, 4);
            drop(starts);
            assert_eq!(used(&lease.frames).0, 2);
            window.abort();
            assert_eq!(used(&lease.frames).0, 2);
            drop(requests);
            assert_eq!(used(&lease.frames), (0, 0));
            assert_eq!(used(&lease.scratch), (0, 0));

            let (_, mut window, _, _) = fixture(capacity, true, seeded, 8);
            let lease = get_lease(&window);
            let held: Vec<_> = (0..3).map(|_| header(&lease)).collect();
            let fail = payload::failure::after(0);
            assert!(matches!(
                FramePermit::pair(&lease, [HEADER; 2]),
                Err(Error::Limit)
            ));
            drop(fail);
            assert_eq!(used(&lease.frames).0, 3); // no partial pair admission
            drop(held);
            assert!(matches!(window.start_frames(), Err(Error::Retired)));

            for after in 0..3 {
                let (_, mut window, _, _) = fixture(capacity, true, seeded, 8);
                let lease = get_lease(&window);
                let fail = payload::failure::after(after);
                assert!(matches!(window.start_frames(), Err(Error::Allocation)));
                drop(fail);
                assert_eq!(used(&lease.frames), (0, 0));
                assert_eq!(used(&lease.scratch), (0, 0));
                assert!(matches!(window.start_frames(), Err(Error::Retired)));
            }
        }
    }
}

#[test]
fn streamed_scratch_preallocation_byte_count_overflow_and_refund() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            let (_, window, _, _) = fixture(capacity, true, seeded, 1);
            let lease = get_lease(&window);
            let scratch: Vec<_> = (0..4)
                .map(|_| WireScratch::filled(&lease, 1, 0u8).unwrap())
                .collect();
            assert_eq!(
                used(&lease.scratch),
                (4, 4 * WireScratch::<u8>::charge(1).unwrap())
            );
            assert!(matches!(
                WireScratch::filled(&lease, 1, 0u8),
                Err(Error::Limit)
            ));
            drop(scratch);
            assert_eq!(used(&lease.scratch), (0, 0));
            assert!(matches!(
                WireScratch::filled(&lease, 1, 0u8),
                Err(Error::Retired)
            ));
            drop(window);

            let (_, mut window, _, _) = fixture(capacity, true, seeded, 1);
            let lease = get_lease(&window);
            let held: Vec<_> = (0..4)
                .map(|_| WireScratch::filled(&lease, 1, 0u8).unwrap())
                .collect();
            assert!(matches!(window.start_frames(), Err(Error::Limit)));
            assert_eq!(used(&lease.frames), (0, 0));
            assert_eq!(used(&lease.scratch).0, 4);
            drop(held);
            assert_eq!(used(&lease.scratch), (0, 0));
            assert!(matches!(window.start_frames(), Err(Error::Retired)));

            for overflow in [false, true] {
                let (_, window, _, _) = fixture(capacity, true, seeded, 1);
                let lease = get_lease(&window);
                let fail = payload::failure::after(0);
                let length = if overflow {
                    usize::MAX
                } else {
                    4 * capacity.max_wire_frame_bytes() / size_of::<u64>()
                };
                assert!(matches!(
                    WireScratch::filled(&lease, length, 0u64),
                    Err(Error::Limit)
                ));
                drop(fail);
                assert_eq!(used(&lease.scratch), (0, 0));
                assert!(matches!(
                    WireScratch::filled(&lease, 1, 0u8),
                    Err(Error::Retired)
                ));
            }

            let (_, window, _, _) = fixture(capacity, true, seeded, 1);
            let lease = get_lease(&window);
            let fail = payload::failure::after(0);
            assert!(matches!(
                WireScratch::filled(&lease, 8, 0u8),
                Err(Error::Allocation)
            ));
            drop(fail);
            assert_eq!(used(&lease.scratch), (0, 0));
        }
    }
}

#[test]
fn streamed_linear_lookup_linear_transport_scratch_and_acceptance_both_roles() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            let plan = [Step::Linear([1; 32]), Step::Lookup, Step::Linear([1; 32])];
            let (publication, mut client, matrix, table) =
                fixture_plan(capacity, true, seeded, &plan);
            let client_lease = get_lease(&client);
            let public = publication.export_wire().unwrap();
            let starts = client.start_frames().unwrap();
            let mut owners = [Worker::Zero, Worker::One].map(|role| {
                let imported = Arc::new(
                    Publication::import_wire_streamed_single_window(
                        public.as_bytes(),
                        publication.digest(),
                        capacity,
                    )
                    .unwrap(),
                );
                WorkerWire::new(
                    imported,
                    publication.digest(),
                    role,
                    &[Stage {
                        id: [1; 32],
                        matrix: &matrix,
                    }],
                    &table,
                )
                .unwrap()
            });
            let mut zero = owners[0].begin(starts[0].as_bytes()).unwrap();
            let mut one = owners[1].begin(starts[1].as_bytes()).unwrap();
            for step in plan {
                let request = match step {
                    Step::Lookup => client.lookup(0),
                    _ => client.linear(&[7]),
                }
                .unwrap();
                assert_eq!(used(&client_lease.frames).0, 4); // retained starts + current pair
                let a = zero.execute(request[0].as_bytes()).unwrap();
                let b = one.execute(request[1].as_bytes()).unwrap();
                let tentative = client.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
                assert_eq!(
                    tentative.values[0].to_bits(),
                    match step {
                        Step::Lookup => (-0.0f32).to_bits(),
                        _ => (21.0f32 / 4_194_304.0).to_bits(),
                    }
                );
                for lease in [&client_lease, &zero.run.lease, &one.run.lease] {
                    assert_eq!(used(&lease.scratch), (0, 0));
                }
            }
            assert_eq!(used(&client_lease.frames).0, 2);
            assert_eq!(client.finish().unwrap().steps(), plan.len());
            zero.finish().unwrap();
            one.finish().unwrap();
            drop(starts);
            assert_eq!(used(&client_lease.frames), (0, 0));
        }
    }
}

#[test]
fn streamed_parallel_reservations_are_atomic_and_refund_after_abort() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let (_, mut window, _, _) = fixture(capacity, true, true, 8);
        let lease = get_lease(&window);
        let gate = std::sync::Barrier::new(5);
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..5)
                .map(|_| {
                    scope.spawn(|| {
                        gate.wait();
                        FramePermit::single(&lease, HEADER)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 4);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(Error::Limit)))
                .count(),
            1
        );
        assert_eq!(used(&lease.frames).0, 4);
        window.abort();
        drop(results);
        assert_eq!(used(&lease.frames), (0, 0));
        assert!(matches!(
            FramePermit::single(&lease, HEADER),
            Err(Error::Retired)
        ));
    }
}

#[test]
fn streamed_historical_worker_replies_exhaust_across_steps_and_never_revive() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            let (publication, mut client, matrix, table) = fixture(capacity, true, seeded, 8);
            let frame = publication.export_wire().unwrap();
            let starts = client.start_frames().unwrap();
            let mut owners = [Worker::Zero, Worker::One].map(|role| {
                let public = Arc::new(
                    Publication::import_wire_streamed_single_window(
                        frame.as_bytes(),
                        publication.digest(),
                        capacity,
                    )
                    .unwrap(),
                );
                WorkerWire::new(
                    public,
                    publication.digest(),
                    role,
                    &[Stage {
                        id: [1; 32],
                        matrix: &matrix,
                    }],
                    &table,
                )
                .unwrap()
            });
            let mut zero = owners[0].begin(starts[0].as_bytes()).unwrap();
            let mut one = owners[1].begin(starts[1].as_bytes()).unwrap();
            let quota = Arc::clone(&one.run.lease);
            drop(starts);
            let mut held = Vec::new();
            for _ in 0..4 {
                let request = client.lookup(0).unwrap();
                let a = zero.execute(request[0].as_bytes()).unwrap();
                let b = one.execute(request[1].as_bytes()).unwrap();
                client.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
                held.push(b);
                assert_eq!(used(&quota.scratch), (0, 0));
            }
            let request = client.lookup(0).unwrap();
            let fail = payload::failure::after(0);
            assert!(matches!(
                one.execute(request[1].as_bytes()),
                Err(Error::Limit)
            ));
            drop(fail);
            assert_eq!(used(&quota.frames).0, 4);
            drop(held);
            assert_eq!(used(&quota.frames), (0, 0));
            assert!(matches!(
                one.execute(request[1].as_bytes()),
                Err(Error::Retired)
            ));
            assert!(one.finish().is_err());
            client.abort();
        }
    }
}

#[test]
fn streamed_publication_one_live_handle_byte_cap_and_generic_bypass() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let (publication, _, _, _) = fixture(capacity, true, false, 1);
        let quota = Arc::clone(publication.export_quota.as_ref().unwrap());
        let frame = publication.export_wire().unwrap();
        assert_eq!(used(&quota), (1, WireBytes::charge(frame.len()).unwrap()));
        assert!(matches!(publication.export_wire(), Err(Error::Limit)));
        drop(frame);
        assert_eq!(used(&quota), (0, 0));
        let fail = payload::failure::after(0);
        assert!(matches!(
            WireBytes::allocate_publication(
                capacity.max_publication_wire_bytes(),
                capacity,
                Some(&quota)
            ),
            Err(Error::Limit)
        ));
        drop(fail);
        assert_eq!(used(&quota), (0, 0));
        let fail = payload::failure::after(0);
        assert!(matches!(publication.export_wire(), Err(Error::Allocation)));
        drop(fail);
        assert_eq!(used(&quota), (0, 0));
        let frame = publication.export_wire().unwrap();
        drop(publication);
        assert_eq!(used(&quota).0, 1);
        drop(frame);
        assert_eq!(used(&quota), (0, 0));

        let (publication, window, _, _) = fixture(capacity, false, false, 1);
        let lease = get_lease(&window);
        let frames: Vec<_> = (0..8).map(|_| header(&lease)).collect();
        let scratch: Vec<_> = (0..8)
            .map(|_| WireScratch::filled(&lease, 1, 0u8).unwrap())
            .collect();
        let publications: Vec<_> = (0..8).map(|_| publication.export_wire().unwrap()).collect();
        assert_eq!(used(&lease.frames), (0, 0));
        assert_eq!(used(&lease.scratch), (0, 0));
        drop((frames, scratch, publications));
    }
}
