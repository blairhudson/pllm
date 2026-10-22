use super::*;
use crate::gkm::{
    BatchPlan, BatchResources, Capacity, Client, FloatTable, I16Matrix, Publication, Stage,
};

struct Fixture {
    client: ClientBatchWireWindow,
    workers: [WorkerBatchWireWindow; 2],
    _sources: [WorkerWire; 2],
    resources: BatchResources,
}

#[test]
fn each_worker_operation_dispatches_one_shared_primitive_and_no_scalar_evaluator() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            for batch in [1, 2, 5] {
                let mut fixture = fixture(capacity, seeded, batch);
                let scalar: Vec<_> = [Step::Lookup, Step::Linear([7; 32]), Step::Lookup]
                    .into_iter()
                    .flat_map(|step| std::iter::repeat_n(step, batch))
                    .collect();
                let descriptors: Vec<_> = [Step::Lookup, Step::Linear([7; 32]), Step::Lookup]
                    .into_iter()
                    .enumerate()
                    .map(|(index, step)| BatchDescriptor {
                        phase_index: 0,
                        operation_index: index as u32,
                        first_claim: (index * batch) as u32,
                        rows: batch as u32,
                        step,
                    })
                    .collect();
                let plan = BatchPlan::new(&scalar, &descriptors).unwrap();
                let sizes = BatchResources::descriptor_wire_sizes(
                    &[([7; 32], 3, 5)],
                    9,
                    3,
                    &plan,
                    if seeded {
                        SharingProfile::Aes128WindowV1
                    } else {
                        SharingProfile::FullShares
                    },
                    capacity,
                )
                .unwrap();
                for (stage, linear) in [false, true, false].into_iter().enumerate() {
                    let requests = if linear {
                        fixture.client.linear(&vec![1; 5 * batch])
                    } else {
                        fixture.client.lookup(&vec![1; batch])
                    }
                    .unwrap();
                    let replies = [0, 1].map(|role| {
                        assert_eq!(requests[role].len(), sizes[stage].request_bytes[role]);
                        SCALAR_EVALUATIONS.with(|count| count.set(0));
                        crate::gkm::numeric::LINEAR_DISPATCHES.with(|count| count.set(0));
                        crate::gkm::lookup::LOOKUP_DISPATCHES.with(|count| count.set(0));
                        let reply = fixture.workers[role]
                            .execute(requests[role].as_bytes())
                            .unwrap();
                        assert_eq!(reply.len(), sizes[stage].reply_bytes[role]);
                        assert_eq!(SCALAR_EVALUATIONS.with(|count| count.get()), 0);
                        assert_eq!(
                            crate::gkm::numeric::LINEAR_DISPATCHES.with(|count| count.get()),
                            usize::from(linear)
                        );
                        assert_eq!(
                            crate::gkm::lookup::LOOKUP_DISPATCHES.with(|count| count.get()),
                            usize::from(!linear)
                        );
                        assert_eq!(
                            *fixture.workers[role]
                                .worker
                                .run
                                .lease
                                .scratch
                                .0
                                .lock()
                                .unwrap(),
                            (0, 0)
                        );
                        reply
                    });
                    fixture
                        .client
                        .accept_replies(replies[0].as_bytes(), replies[1].as_bytes())
                        .unwrap();
                }
                for worker in &mut fixture.workers {
                    worker.finish().unwrap();
                }
                fixture.client.finish().unwrap();
            }
        }
    }
}

#[test]
fn malformed_complete_client_operation_precedes_allocation_and_retires_without_entropy() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            for case in 0..6 {
                let linear = case >= 2;
                let mut fixture = fixture_stages(
                    capacity,
                    seeded,
                    2,
                    &[if linear {
                        Step::Linear([7; 32])
                    } else {
                        Step::Lookup
                    }],
                );
                let mut input = [0i64; 10];
                if case == 4 {
                    input[9] = 101;
                }
                if case == 5 {
                    input[9] = i64::MIN;
                }
                let failure = payload::failure::after(0);
                let result = match case {
                    0 => fixture.client.lookup(&[0]),
                    1 => fixture.client.lookup(&[0, 9]),
                    2 => fixture.client.linear(&input[..9]),
                    3 => fixture.client.lookup(&[0, 1]),
                    _ => fixture.client.linear(&input),
                };
                drop(failure);
                assert!(matches!(
                    (case, result),
                    (0 | 2, Err(Error::Shape))
                        | (1 | 4 | 5, Err(Error::Bound))
                        | (3, Err(Error::Schedule))
                ));
                assert!(matches!(
                    fixture.client.lookup(&[0, 1]),
                    Err(Error::Retired)
                ));
                assert!(fixture.client.finish().is_err());
            }
        }
    }
}

#[test]
fn worker_source_profile_and_schedule_binding_preflight_has_no_plan_key_or_lease_allocation() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for profile in [SharingProfile::FullShares, SharingProfile::Aes128WindowV1] {
            for role in [Worker::Zero, Worker::One] {
                for case in 0..5 {
                    let matrix = I16Matrix::from_encoded(1, 1, &[3], &[100]).unwrap();
                    let table = FloatTable::from_le_bytes(1, 1, &(-0.0f32).to_le_bytes()).unwrap();
                    let stages = [Stage {
                        id: [7; 32],
                        matrix: &matrix,
                    }];
                    let public = |context| {
                        Arc::new(
                            Publication::publish_streamed_single_window(
                                context, &stages, &table, capacity,
                            )
                            .unwrap(),
                        )
                    };
                    let owner = public([5; 32]);
                    let local = public([5; 32]);
                    let other = public([6; 32]);
                    let plan = |step| {
                        BatchPlan::new(
                            &[step; 2],
                            &[BatchDescriptor {
                                phase_index: 0,
                                operation_index: 0,
                                first_claim: 0,
                                rows: 2,
                                step,
                            }],
                        )
                        .unwrap()
                    };
                    let native = Client::new(Arc::clone(&owner), owner.digest()).unwrap();
                    let mut client = native
                        .begin_batch_wire(
                            [9; 32],
                            &[Step::Lookup; 2],
                            BatchAdmission::new(&owner, plan(Step::Lookup), profile).unwrap(),
                        )
                        .unwrap();
                    let starts = client.start_frames().unwrap();
                    let mut worker =
                        WorkerWire::new(Arc::clone(&local), local.digest(), role, &stages, &table)
                            .unwrap();
                    let admission = BatchAdmission::new(
                        if case == 0 { &other } else { &local },
                        plan(if case == 2 {
                            Step::Linear([7; 32])
                        } else {
                            Step::Lookup
                        }),
                        if case == 1 {
                            if profile == SharingProfile::FullShares {
                                SharingProfile::Aes128WindowV1
                            } else {
                                SharingProfile::FullShares
                            }
                        } else {
                            profile
                        },
                    )
                    .unwrap();
                    let retry = BatchAdmission::new(&local, plan(Step::Lookup), profile).unwrap();
                    let failure = payload::failure::after(0);
                    let result = if case >= 3 {
                        worker.begin_batch_bound(
                            starts[usize::from(role == Worker::One)].as_bytes(),
                            if case == 3 { [8; 32] } else { [9; 32] },
                            if case == 4 {
                                &[Step::Linear([7; 32]); 2]
                            } else {
                                &[Step::Lookup; 2]
                            },
                            admission,
                        )
                    } else {
                        worker.begin_batch(
                            starts[usize::from(role == Worker::One)].as_bytes(),
                            admission,
                        )
                    };
                    assert!(matches!(result, Err(Error::Binding)));
                    assert_eq!(local.live_windows(), 0);
                    assert!(worker.active.is_none());
                    assert!(worker.source.is_none());
                    assert!(matches!(
                        worker.begin_batch(
                            starts[usize::from(role == Worker::One)].as_bytes(),
                            retry
                        ),
                        Err(Error::Retired)
                    ));
                    drop(failure);
                }
            }
        }
    }
}

fn fixture(capacity: Capacity, seeded: bool, rows: usize) -> Fixture {
    fixture_stages(
        capacity,
        seeded,
        rows,
        &[Step::Lookup, Step::Linear([7; 32]), Step::Lookup],
    )
}

fn fixture_stages(capacity: Capacity, seeded: bool, rows: usize, steps: &[Step]) -> Fixture {
    let id = [7; 32];
    let matrix = I16Matrix::from_encoded(
        3,
        5,
        &[1, -2, 3, -4, 5, 2, 3, 4, 5, 6, -1, 0, 1, 0, -1],
        &[100; 5],
    )
    .unwrap();
    let table_bytes: Vec<_> = (0..27)
        .flat_map(|i| {
            let value = if i % 3 == 0 { -0.0f32 } else { i as f32 / 8.0 };
            value.to_le_bytes()
        })
        .collect();
    let table = FloatTable::from_le_bytes(9, 3, &table_bytes).unwrap();
    let stages = [Stage {
        id,
        matrix: &matrix,
    }];
    let public = Arc::new(
        Publication::publish_streamed_single_window([5; 32], &stages, &table, capacity).unwrap(),
    );
    let profile = if seeded {
        SharingProfile::Aes128WindowV1
    } else {
        SharingProfile::FullShares
    };
    let scalar_plan = steps
        .iter()
        .flat_map(|&step| std::iter::repeat_n(step, rows))
        .collect::<Vec<_>>();
    let descriptors = steps
        .iter()
        .enumerate()
        .map(|(index, &step)| BatchDescriptor {
            phase_index: 0,
            operation_index: index as u32,
            first_claim: (index * rows) as u32,
            rows: rows as u32,
            step,
        })
        .collect::<Vec<_>>();
    let plan = || BatchPlan::new(&scalar_plan, &descriptors).unwrap();
    let resources =
        BatchResources::for_streamed_single_window(&[(id, 3, 5)], 9, 3, &plan(), profile, capacity)
            .unwrap();
    let admission = BatchAdmission::new(&public, plan(), profile).unwrap();
    assert_eq!(*admission.resources(), resources);
    let source = public.export_wire().unwrap();
    let native = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let mut client = native
        .begin_batch_wire([6; 32], &scalar_plan, admission)
        .unwrap();
    let starts = client.start_frames().unwrap();
    let owners = [Worker::Zero, Worker::One].map(|role| {
        let local = Arc::new(
            Publication::import_wire_streamed_single_window(
                source.as_bytes(),
                public.digest(),
                capacity,
            )
            .unwrap(),
        );
        let admission = BatchAdmission::new(&local, plan(), profile).unwrap();
        let mut source =
            WorkerWire::new(Arc::clone(&local), public.digest(), role, &stages, &table).unwrap();
        let window = source
            .begin_batch_bound(
                starts[usize::from(role == Worker::One)].as_bytes(),
                [6; 32],
                &scalar_plan,
                admission,
            )
            .unwrap();
        (source, window)
    });
    let [(zero_source, zero), (one_source, one)] = owners;
    Fixture {
        client,
        workers: [zero, one],
        _sources: [zero_source, one_source],
        resources,
    }
}

#[test]
fn four_retained_aggregate_replies_fifth_fails_before_allocation_and_is_sticky() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            let mut fixture = fixture_stages(capacity, seeded, 2, &[Step::Lookup; 8]);
            let mut held = Vec::new();
            for _ in 0..4 {
                let request = fixture.client.lookup(&[0, 1]).unwrap();
                let a = fixture.workers[0].execute(request[0].as_bytes()).unwrap();
                let b = fixture.workers[1].execute(request[1].as_bytes()).unwrap();
                fixture
                    .client
                    .accept_replies(a.as_bytes(), b.as_bytes())
                    .unwrap();
                held.push(b);
            }
            let request = fixture.client.lookup(&[0, 1]).unwrap();
            let fail = payload::failure::after(0);
            assert!(matches!(
                fixture.workers[1].execute(request[1].as_bytes()),
                Err(Error::Limit)
            ));
            drop(fail);
            drop(held);
            assert!(matches!(
                fixture.workers[1].execute(request[1].as_bytes()),
                Err(Error::Retired)
            ));
            assert!(fixture.workers[1].finish().is_err());
            fixture.client.abort();
        }
    }
}

#[test]
fn aggregate_retained_after_abort_keeps_exact_credit_until_bytes_destroyed() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            let before = WireResources::current_with_capacity(capacity).process_live_bytes;
            let mut fixture = fixture(capacity, seeded, 2);
            let admitted = WireResources::current_with_capacity(capacity).process_live_bytes;
            let request = fixture.client.lookup(&[0, 1]).unwrap();
            let reply = fixture.workers[1].execute(request[1].as_bytes()).unwrap();
            assert_eq!(
                WireResources::current_with_capacity(capacity).process_live_bytes,
                admitted
            );
            let credit =
                fixture.resources.retained_frame_bytes + fixture.resources.transport_scratch_bytes;
            fixture.client.abort();
            fixture.workers[1].abort();
            drop(request);
            drop(fixture);
            assert_eq!(
                WireResources::current_with_capacity(capacity).process_live_bytes,
                before + credit
            );
            drop(reply);
            assert_eq!(
                WireResources::current_with_capacity(capacity).process_live_bytes,
                before
            );
        }
    }
}

#[test]
fn every_batch_issue_and_freeze_payload_allocation_failure_burns_owner() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            for freeze in [false, true] {
                let mut failures = 0;
                for after in 0..64 {
                    let mut fixture = fixture(capacity, seeded, 2);
                    if freeze {
                        let request = fixture.client.lookup(&[0, 1]).unwrap();
                        let a = fixture.workers[0].execute(request[0].as_bytes()).unwrap();
                        let b = fixture.workers[1].execute(request[1].as_bytes()).unwrap();
                        let fail = payload::failure::after(after);
                        let result = fixture
                            .client
                            .accept_replies(a.as_bytes(), b.as_bytes())
                            .map(|_| ());
                        drop(fail);
                        match result {
                            Ok(()) => break,
                            Err(Error::Allocation) => failures += 1,
                            Err(error) => panic!("unexpected freeze failure: {error:?}"),
                        }
                    } else {
                        let fail = payload::failure::after(after);
                        let result = fixture.client.lookup(&[0, 1]);
                        drop(fail);
                        match result {
                            Ok(_) => break,
                            Err(Error::Allocation) => failures += 1,
                            Err(error) => panic!("unexpected issue failure: {error:?}"),
                        }
                    }
                    assert!(matches!(
                        fixture.client.lookup(&[0, 1]),
                        Err(Error::Retired)
                    ));
                    assert!(fixture.client.finish().is_err());
                }
                assert!(failures >= 4);
            }
        }
    }
}

#[test]
fn batch_lookup_authentication_and_final_linear_verification_remain_authoritative() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            for corrupt_lookup in [false, true] {
                let mut fixture = fixture(capacity, seeded, 2);
                for (stage, linear) in [false, true, false].into_iter().enumerate() {
                    let request = if linear {
                        fixture.client.linear(&[-1, 2, -3, 4, -5, -1, 2, -3, 4, -5])
                    } else {
                        fixture.client.lookup(&[0, 1])
                    }
                    .unwrap();
                    let a = fixture.workers[0].execute(request[0].as_bytes()).unwrap();
                    let b = fixture.workers[1].execute(request[1].as_bytes()).unwrap();
                    let mut bytes = b.as_bytes().to_vec();
                    if stage == if corrupt_lookup { 0 } else { 1 } {
                        bytes[AGGREGATE_HEADER_BYTES + 8] ^= 1;
                    }
                    let result = fixture.client.accept_replies(a.as_bytes(), &bytes);
                    if corrupt_lookup {
                        assert!(matches!(result, Err(Error::Verification)));
                        assert!(fixture.client.finish().is_err());
                        break;
                    }
                    result.unwrap();
                }
                if !corrupt_lookup {
                    assert!(matches!(fixture.client.finish(), Err(Error::Verification)));
                }
            }
        }
    }
}

#[test]
fn aggregate_b1_b2_odd_aes_tails_and_exact_complete_acceptance_both_capacities_profiles() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            for rows in [1, 2, 5] {
                let mut fixture = fixture(capacity, seeded, rows);
                for linear in [false, true, false] {
                    let tokens: Vec<_> = (0..rows).map(|row| row % 9).collect();
                    let input: Vec<_> = (0..rows).flat_map(|_| [-1, 2, -3, 4, -5]).collect();
                    let request = if linear {
                        fixture.client.linear(&input)
                    } else {
                        fixture.client.lookup(&tokens)
                    }
                    .unwrap();
                    assert_eq!(&request[0].as_bytes()[..8], BATCH_MAGIC);
                    if seeded {
                        assert_eq!(request[1].len(), AGGREGATE_HEADER_BYTES + rows * 8);
                    }
                    assert!(request
                        .iter()
                        .all(|frame| frame.len() <= fixture.resources.max_aggregate_bytes));
                    let a = fixture.workers[0].execute(request[0].as_bytes()).unwrap();
                    let b = fixture.workers[1].execute(request[1].as_bytes()).unwrap();
                    let tentative = fixture
                        .client
                        .accept_replies(a.as_bytes(), b.as_bytes())
                        .unwrap();
                    assert_eq!(tentative.values.len(), rows * 3);
                    if linear {
                        let expected =
                            [-55i64, -18, 3].map(|value| (value as f32 / 4_194_304.0).to_bits());
                        for row in tentative.values.chunks_exact(3) {
                            assert_eq!(
                                row.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                                expected
                            );
                        }
                    } else {
                        for (row, values) in tentative.values.chunks_exact(3).enumerate() {
                            let index = row % 9;
                            let expected = [
                                (-0.0f32).to_bits(),
                                ((index * 3 + 1) as f32 / 8.0).to_bits(),
                                ((index * 3 + 2) as f32 / 8.0).to_bits(),
                            ];
                            assert_eq!(
                                values
                                    .iter()
                                    .map(|value| value.to_bits())
                                    .collect::<Vec<_>>(),
                                expected
                            );
                        }
                    }
                }
                let accepted = fixture.client.finish().unwrap();
                assert_eq!(accepted.steps(), 3 * rows);
                assert_eq!(accepted.values().len(), 3 * rows);
                for worker in &mut fixture.workers {
                    assert_eq!(worker.finish().unwrap().steps, 3 * rows);
                }
            }
        }
    }
}

#[test]
fn aggregate_header_lengths_profile_role_plan_and_replay_substitutions_are_sticky() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            for offset in [
                0, 8, 9, 42, 114, 146, 182, 214, 218, 222, 226, 259, 291, 292, 324, 332, 340,
            ] {
                let mut fixture = fixture(capacity, seeded, 2);
                let request = fixture.client.lookup(&[0, 1]).unwrap();
                let mut changed = request[1].as_bytes().to_vec();
                changed[offset] ^= 1;
                assert!(fixture.workers[1].execute(&changed).is_err());
                assert!(matches!(
                    fixture.workers[1].execute(request[1].as_bytes()),
                    Err(Error::Retired)
                ));
                fixture.client.abort();
            }
        }
    }
}

#[test]
fn metadata_admits_aggregate_above_four_mib_without_raising_scalar_row_limit() {
    let id = [1; 32];
    let scalar = [Step::Linear(id); 3];
    let descriptor = BatchDescriptor {
        phase_index: 0,
        operation_index: 0,
        first_claim: 0,
        rows: 3,
        step: scalar[0],
    };
    let plan = BatchPlan::new(&scalar, &[descriptor]).unwrap();
    let resources = BatchResources::for_streamed_single_window(
        &[(id, 262_144, 1)],
        1,
        1,
        &plan,
        SharingProfile::FullShares,
        Capacity::Decoder,
    )
    .unwrap();
    let f = Capacity::Decoder.max_wire_frame_bytes();
    assert!(resources.max_aggregate_bytes > f);
    assert_eq!(
        resources.max_aggregate_bytes,
        AGGREGATE_HEADER_BYTES + 3 * (8 + 262_144 * 8)
    );
    assert_eq!(
        resources.extra_wire_bytes,
        resources.retained_frame_bytes + resources.transport_scratch_bytes - 8 * f
    );
    assert!(BatchResources::for_streamed_single_window(
        &[(id, 262_145, 1)],
        1,
        1,
        &plan,
        SharingProfile::FullShares,
        Capacity::Decoder
    )
    .is_err());
    assert!(BatchResources::for_streamed_single_window(
        &[(id, usize::MAX, 1)],
        1,
        1,
        &plan,
        SharingProfile::FullShares,
        Capacity::Decoder
    )
    .is_err());
}

#[test]
fn both_complete_role_packets_stay_frozen_before_every_independent_row_challenge() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    struct Observe {
        lease: Arc<Lease>,
        calls: usize,
    }
    impl Entropy for Observe {
        fn fill(&mut self, purpose: Purpose, bytes: &mut [u8]) -> Result<(), Error> {
            assert_eq!(purpose, Purpose::ClaimChallenge);
            assert_eq!(self.lease.scratch.0.lock().unwrap().0, 2);
            self.calls += 1;
            bytes.fill(0);
            *bytes.last_mut().unwrap() = self.calls as u8;
            Ok(())
        }
    }
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            let mut fixture = fixture_stages(capacity, seeded, 5, &[Step::Linear([7; 32])]);
            let input: Vec<_> = (0..5).flat_map(|_| [-1, 2, -3, 4, -5]).collect();
            let request = fixture.client.linear(&input).unwrap();
            let a = fixture.workers[0].execute(request[0].as_bytes()).unwrap();
            let b = fixture.workers[1].execute(request[1].as_bytes()).unwrap();
            let Some(ClientState::Ready(window)) = &fixture.client.client.state else {
                panic!("not pending");
            };
            let mut rng = Observe {
                lease: Arc::clone(&window._lease),
                calls: 0,
            };
            fixture
                .client
                .accept_inner_with(a.as_bytes(), b.as_bytes(), &mut rng)
                .unwrap();
            assert_eq!(rng.calls, 5);
            assert_eq!(*rng.lease.scratch.0.lock().unwrap(), (0, 0));
            fixture.client.finish().unwrap();
            for worker in &mut fixture.workers {
                worker.finish().unwrap();
            }
        }
    }
}

#[test]
fn actual_lookup_aggregate_above_scalar_four_mib_limit_uses_only_batch_admission() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let capacity = Capacity::Decoder;
    let width = 16_384;
    let rows = 65;
    let id = [8; 32];
    for seeded in [false, true] {
        let matrix = I16Matrix::from_encoded(1, 1, &[3], &[100]).unwrap();
        let values: Vec<_> = (0..width)
            .map(|i| {
                if i == 0 {
                    -0.0f32
                } else {
                    (i % 31) as f32 / 8.0
                }
            })
            .collect();
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let table = FloatTable::from_le_bytes_with_capacity(1, width, &bytes, capacity).unwrap();
        let stages = [Stage {
            id,
            matrix: &matrix,
        }];
        let public = Arc::new(
            Publication::publish_streamed_single_window([1; 32], &stages, &table, capacity)
                .unwrap(),
        );
        let scalar = vec![Step::Lookup; rows];
        let descriptor = BatchDescriptor {
            phase_index: 0,
            operation_index: 0,
            first_claim: 0,
            rows: rows as u32,
            step: Step::Lookup,
        };
        let plan = || BatchPlan::new(&scalar, &[descriptor]).unwrap();
        let profile = if seeded {
            SharingProfile::Aes128WindowV1
        } else {
            SharingProfile::FullShares
        };
        let admission = BatchAdmission::new(&public, plan(), profile).unwrap();
        let resources = *admission.resources();
        assert!(resources.max_aggregate_bytes > capacity.max_wire_frame_bytes());
        let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
        let window = if seeded {
            client.begin_seeded_wire([2; 32], &scalar)
        } else {
            client.begin_wire([2; 32], &scalar)
        }
        .unwrap();
        let mut client = window.into_batch(admission).unwrap();
        let publication = public.export_wire().unwrap();
        let starts = client.start_frames().unwrap();
        let mut workers = [Worker::Zero, Worker::One].map(|role| {
            let local = Arc::new(
                Publication::import_wire_streamed_single_window(
                    publication.as_bytes(),
                    public.digest(),
                    capacity,
                )
                .unwrap(),
            );
            let admission = BatchAdmission::new(&local, plan(), profile).unwrap();
            let mut source =
                WorkerWire::new(local, public.digest(), role, &stages, &table).unwrap();
            let worker = source
                .begin(starts[usize::from(role == Worker::One)].as_bytes())
                .unwrap()
                .into_batch(admission)
                .unwrap();
            (source, worker)
        });
        let request = client.lookup(&vec![0; rows]).unwrap();
        let a = workers[0].1.execute(request[0].as_bytes()).unwrap();
        let b = workers[1].1.execute(request[1].as_bytes()).unwrap();
        assert_eq!(a.len(), resources.max_aggregate_bytes);
        assert_eq!(b.len(), resources.max_aggregate_bytes);
        assert!(a.len() > capacity.max_wire_frame_bytes());
        client.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
        let accepted = client.finish().unwrap();
        for row in accepted.values().chunks_exact(width) {
            assert!(row
                .iter()
                .zip(&values)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
        }
        for (_, worker) in &mut workers {
            worker.finish().unwrap();
        }
    }
}

#[test]
fn partial_role_reply_and_valid_previous_batch_replay_retire_without_retry() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            for partial in [false, true] {
                let mut fixture = fixture(capacity, seeded, 2);
                let request = fixture.client.lookup(&[0, 1]).unwrap();
                let a = fixture.workers[0].execute(request[0].as_bytes()).unwrap();
                let b = fixture.workers[1].execute(request[1].as_bytes()).unwrap();
                if partial {
                    assert!(fixture
                        .client
                        .accept_replies(a.as_bytes(), &b.as_bytes()[..b.len() - 1])
                        .is_err());
                    assert!(matches!(
                        fixture.client.accept_replies(a.as_bytes(), b.as_bytes()),
                        Err(Error::Retired)
                    ));
                } else {
                    fixture
                        .client
                        .accept_replies(a.as_bytes(), b.as_bytes())
                        .unwrap();
                    assert!(fixture.workers[1].execute(request[1].as_bytes()).is_err());
                    assert!(matches!(
                        fixture.workers[1].execute(request[1].as_bytes()),
                        Err(Error::Retired)
                    ));
                }
                fixture.client.abort();
                assert!(fixture.client.finish().is_err());
            }
        }
    }
}

#[test]
fn every_aggregate_worker_output_allocation_failure_refunds_frames_and_scratch_but_retires() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for seeded in [false, true] {
            let mut failures = 0;
            for after in 0..32 {
                let mut fixture = fixture(capacity, seeded, 2);
                let request = fixture.client.lookup(&[0, 1]).unwrap();
                let reserved = WireResources::current_with_capacity(capacity).process_live_bytes;
                let fail = payload::failure::after(after);
                let result = fixture.workers[1].execute(request[1].as_bytes());
                drop(fail);
                match result {
                    Ok(_) => break,
                    Err(Error::Allocation) => failures += 1,
                    Err(error) => panic!("unexpected worker allocation error: {error:?}"),
                }
                let lease = &fixture.workers[1].worker.run.lease;
                assert_eq!(fixture.workers[1].worker.completed, 0);
                assert_eq!(lease.seeded_completed.load(Ordering::Acquire), 0);
                assert_eq!(*lease.frames.0.lock().unwrap(), (0, 0));
                assert_eq!(*lease.scratch.0.lock().unwrap(), (0, 0));
                assert_eq!(
                    WireResources::current_with_capacity(capacity).process_live_bytes,
                    reserved
                );
                assert!(matches!(
                    fixture.workers[1].execute(request[1].as_bytes()),
                    Err(Error::Retired)
                ));
            }
            assert!(failures >= 3);
        }
    }
}
