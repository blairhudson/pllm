use super::*;

fn descriptor(first_claim: u32, rows: u32, step: Step) -> BatchDescriptor {
    BatchDescriptor {
        phase_index: 0,
        operation_index: 1,
        first_claim,
        rows,
        step,
    }
}

#[test]
fn canonical_batch_plan_matches_independent_fixed_width_vector() {
    let scalar = [Step::Lookup, Step::Lookup, Step::Linear([7; 32])];
    let descriptors = [descriptor(0, 2, Step::Lookup), descriptor(2, 1, scalar[2])];
    let plan = BatchPlan::new(&scalar, &descriptors).unwrap();
    let mut bytes = b"pllm.gkm.source-stage-batch.v1".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&[0; 66]);
    bytes.push(1);
    bytes.extend_from_slice(&[7; 32]);
    bytes.extend_from_slice(&2u32.to_le_bytes());
    for (first, rows, tag, id) in [(0u32, 2u32, 0u8, [0u8; 32]), (2, 1, 1, [7; 32])] {
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&first.to_le_bytes());
        bytes.extend_from_slice(&rows.to_le_bytes());
        bytes.push(tag);
        bytes.extend_from_slice(&id);
    }
    let expected: [u8; 32] = Sha256::digest(bytes).into();
    assert_eq!(plan.digest(), expected);
    assert_eq!(plan.descriptors(), descriptors);
    plan.validate_scalar_plan(&scalar).unwrap();
    assert!(matches!(
        plan.validate_scalar_plan(&[Step::Lookup; 3]),
        Err(Error::Binding)
    ));
}

#[test]
fn batch_plan_rejects_gaps_overlap_zero_mixed_steps_overflow_and_partial_coverage() {
    let scalar = [Step::Lookup, Step::Lookup, Step::Linear([7; 32])];
    for bad in [
        vec![descriptor(1, 2, Step::Lookup)],
        vec![
            descriptor(0, 1, Step::Lookup),
            descriptor(0, 2, Step::Lookup),
        ],
        vec![descriptor(0, 0, Step::Lookup)],
        vec![descriptor(0, 3, Step::Lookup)],
        vec![descriptor(0, 2, Step::Lookup)],
        vec![descriptor(0, u32::MAX, Step::Lookup)],
    ] {
        assert!(BatchPlan::new(&scalar, &bad).is_err());
    }
    assert!(BatchPlan::new(&[], &[]).is_err());
    assert!(BatchPlan::new(&scalar, &[]).is_err());
}

#[test]
fn batching_b1_b2_odd_preserves_scalar_fingerprint_and_binds_source_labels() {
    let scalar = [Step::Lookup; 5];
    let descriptors = [
        descriptor(0, 2, Step::Lookup),
        descriptor(2, 3, Step::Lookup),
    ];
    let plan = BatchPlan::new(&scalar, &descriptors).unwrap();
    let singles: Vec<_> = (0..5)
        .map(|first| descriptor(first, 1, Step::Lookup))
        .collect();
    let singles = BatchPlan::new(&scalar, &singles).unwrap();
    assert_eq!(plan.scalar_digest, singles.scalar_digest);
    assert_ne!(plan.digest(), singles.digest());
    for phase in [false, true] {
        let mut changed = descriptors;
        if phase {
            changed[0].phase_index += 1;
        } else {
            changed[0].operation_index += 1;
        }
        assert_ne!(
            plan.digest(),
            BatchPlan::new(&scalar, &changed).unwrap().digest()
        );
    }
}

#[test]
fn invalid_batch_plan_does_not_allocate_and_valid_metadata_allocation_is_fallible() {
    let fail = payload::failure::after(0);
    assert!(matches!(
        BatchPlan::new(&[Step::Lookup], &[descriptor(1, 1, Step::Lookup)]),
        Err(Error::Schedule)
    ));
    assert!(matches!(
        BatchPlan::new(&[Step::Lookup], &[descriptor(0, 1, Step::Lookup)]),
        Err(Error::Allocation)
    ));
    drop(fail);
}

fn publication(capacity: Capacity) -> Arc<Publication> {
    publication_with_context(capacity, [2; 32])
}

fn publication_with_context(capacity: Capacity, context: [u8; 32]) -> Arc<Publication> {
    let matrix = super::super::I16Matrix::from_encoded(1, 1, &[3], &[100]).unwrap();
    let table = super::super::FloatTable::from_le_bytes(1, 1, &(-0.0f32).to_le_bytes()).unwrap();
    Arc::new(
        Publication::publish_streamed_single_window(
            context,
            &[super::super::Stage {
                id: [1; 32],
                matrix: &matrix,
            }],
            &table,
            capacity,
        )
        .unwrap(),
    )
}

#[test]
fn client_admission_binding_precedes_scalar_allocation_and_burns_malformed_attempt() {
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for profile in [SharingProfile::FullShares, SharingProfile::Aes128WindowV1] {
            for wrong_source in [false, true] {
                let public = publication(capacity);
                let other = publication_with_context(capacity, [3; 32]);
                let client =
                    super::super::Client::new(Arc::clone(&public), public.digest()).unwrap();
                let plan =
                    BatchPlan::new(&[Step::Lookup; 2], &[descriptor(0, 2, Step::Lookup)]).unwrap();
                let admission =
                    BatchAdmission::new(if wrong_source { &other } else { &public }, plan, profile)
                        .unwrap();
                let failure = payload::failure::after(0);
                assert!(matches!(
                    client.begin_batch_wire(
                        [8; 32],
                        if wrong_source {
                            &[Step::Lookup; 2]
                        } else {
                            &[Step::Linear([1; 32]); 2]
                        },
                        admission
                    ),
                    Err(Error::Binding)
                ));
                assert_eq!(public.live_windows(), 0);
                assert!(matches!(
                    client.begin([8; 32], &[Step::Lookup; 2]),
                    Err(Error::Retired)
                ));
                drop(failure);
            }
            let public = publication(capacity);
            let plan = BatchPlan::new(&[Step::Lookup], &[descriptor(0, 1, Step::Lookup)]).unwrap();
            let admission = BatchAdmission::new(&public, plan, profile).unwrap();
            let other_profile = if profile == SharingProfile::FullShares {
                SharingProfile::Aes128WindowV1
            } else {
                SharingProfile::FullShares
            };
            let failure = payload::failure::after(0);
            assert!(matches!(
                admission.validate_binding(&public, &[Step::Lookup], other_profile),
                Err(Error::Binding)
            ));
            assert_eq!(public.live_windows(), 0);
            drop(failure);
        }
    }
}

#[test]
fn worker_concrete_layout_peaks_and_each_allocation_failure_refund_exclusive_owner() {
    assert_eq!(
        WorkerBatchPayloadLayout::new(&descriptor(0, 3, Step::Linear([1; 32])), 5, 7)
            .unwrap()
            .payload_bytes()
            .unwrap(),
        3 * (5 + 7) * 8
    );
    assert_eq!(
        WorkerBatchPayloadLayout::new(&descriptor(0, 3, Step::Lookup), 9usize.div_ceil(8), 17)
            .unwrap()
            .payload_bytes()
            .unwrap(),
        3 * 2 + 3 * 17 * 4
    );
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for profile in [SharingProfile::FullShares, SharingProfile::Aes128WindowV1] {
            let public = publication(capacity);
            for step in [Step::Lookup, Step::Linear([1; 32])] {
                let descriptor = descriptor(0, 3, step);
                let plan = BatchPlan::new(&[step; 3], &[descriptor]).unwrap();
                let admission = Arc::new(BatchAdmission::new(&public, plan, profile).unwrap());
                let bytes = PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire);
                for after in 0..2 {
                    let failure = payload::failure::after(after);
                    assert!(matches!(
                        WorkerBatchPayload::allocate(&admission, &descriptor, 1, 1),
                        Err(Error::Allocation)
                    ));
                    drop(failure);
                    assert!(!admission.payload_live.load(Ordering::Acquire));
                    assert_eq!(PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire), bytes);
                }
                let owner = WorkerBatchPayload::allocate(&admission, &descriptor, 1, 1).unwrap();
                assert!(matches!(
                    BatchPayload::allocate(&public, Arc::clone(&admission), 0),
                    Err(Error::Limit)
                ));
                drop(owner);
                assert!(!admission.payload_live.load(Ordering::Acquire));
                assert_eq!(PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire), bytes);
            }
        }
    }
}

#[test]
fn descriptor_wire_sizes_are_role_exact_include_headers_and_share_common_checked_geometry() {
    let plan = BatchPlan::new(
        &[
            Step::Lookup,
            Step::Lookup,
            Step::Linear([1; 32]),
            Step::Linear([1; 32]),
            Step::Linear([1; 32]),
        ],
        &[
            descriptor(0, 2, Step::Lookup),
            descriptor(2, 3, Step::Linear([1; 32])),
        ],
    )
    .unwrap();
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for profile in [SharingProfile::FullShares, SharingProfile::Aes128WindowV1] {
            let sizes = BatchResources::descriptor_wire_sizes(
                &[([1; 32], 1, 1)],
                1,
                1,
                &plan,
                profile,
                capacity,
            )
            .unwrap();
            assert_eq!(
                sizes[0].request_bytes,
                [
                    358,
                    if profile == SharingProfile::FullShares {
                        358
                    } else {
                        356
                    }
                ]
            );
            assert_eq!(sizes[0].reply_bytes, [364; 2]);
            assert_eq!(
                sizes[1].request_bytes,
                [
                    388,
                    if profile == SharingProfile::FullShares {
                        388
                    } else {
                        364
                    }
                ]
            );
            assert_eq!(sizes[1].reply_bytes, [388; 2]);
            let failure = payload::failure::after(0);
            assert!(matches!(
                BatchResources::descriptor_wire_sizes(
                    &[([2; 32], 1, 1)],
                    1,
                    1,
                    &plan,
                    profile,
                    capacity
                ),
                Err(Error::Binding)
            ));
            assert!(matches!(
                BatchResources::descriptor_wire_sizes(
                    &[([1; 32], usize::MAX, 1)],
                    1,
                    1,
                    &plan,
                    profile,
                    capacity
                ),
                Err(Error::Limit)
            ));
            drop(failure);
        }
    }
}

#[test]
fn metadata_recipe_matches_real_native_and_wire_reservations_and_exclusive_payload_owner() {
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for profile in [SharingProfile::FullShares, SharingProfile::Aes128WindowV1] {
            let public = publication(capacity);
            let plan = BatchPlan::new(
                &[Step::Linear([1; 32]); 2],
                &[descriptor(0, 2, Step::Linear([1; 32]))],
            )
            .unwrap();
            let resources = BatchResources::for_streamed_single_window(
                &[([1; 32], 1, 1)],
                1,
                1,
                &plan,
                profile,
                capacity,
            )
            .unwrap();
            let payload_bytes = if profile == SharingProfile::FullShares {
                72
            } else {
                56
            };
            assert_eq!(
                resources.extra_native_owner_bytes,
                size_of::<BatchDescriptor>()
                    + size_of::<BatchAdmission>()
                    + 2 * size_of::<usize>()
                    + batch_owner_control_bytes().unwrap()
                    + payload_bytes
            );
            let native_before = PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire);
            let wire_before =
                super::super::WireResources::current_with_capacity(capacity).process_live_bytes;
            let admission = Arc::new(BatchAdmission::new(&public, plan, profile).unwrap());
            assert_eq!(*admission.resources(), resources);
            assert_eq!(
                PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire),
                native_before + resources.extra_native_owner_bytes
            );
            let wire_reserved = resources.retained_frame_bytes + resources.transport_scratch_bytes;
            assert_eq!(
                super::super::WireResources::current_with_capacity(capacity).process_live_bytes,
                wire_before + wire_reserved
            );
            let owner = BatchPayload::allocate(&public, Arc::clone(&admission), 0).unwrap();
            assert!(matches!(
                BatchPayload::allocate(&public, Arc::clone(&admission), 0),
                Err(Error::Limit)
            ));
            drop(admission);
            assert_eq!(
                PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire),
                native_before + resources.extra_native_owner_bytes
            );
            drop(owner);
            assert_eq!(
                PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire),
                native_before
            );
            assert_eq!(
                super::super::WireResources::current_with_capacity(capacity).process_live_bytes,
                wire_before
            );
        }
    }
}

#[test]
fn every_flat_owner_allocation_failure_refunds_exclusive_slot_and_keeps_reservation() {
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        for profile in [SharingProfile::FullShares, SharingProfile::Aes128WindowV1] {
            let public = publication(capacity);
            let plan = BatchPlan::new(
                &[Step::Linear([1; 32]); 2],
                &[descriptor(0, 2, Step::Linear([1; 32]))],
            )
            .unwrap();
            let admission = Arc::new(BatchAdmission::new(&public, plan, profile).unwrap());
            let native_reserved = PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire);
            let wire_reserved =
                super::super::WireResources::current_with_capacity(capacity).process_live_bytes;
            let mut failures = 0;
            for after in 0..8 {
                let fail = payload::failure::after(after);
                let result = BatchPayload::allocate(&public, Arc::clone(&admission), 0);
                drop(fail);
                match result {
                    Ok(owner) => {
                        drop(owner);
                        break;
                    }
                    Err(Error::Allocation) => failures += 1,
                    Err(error) => panic!("unexpected allocation error: {error:?}"),
                }
                assert!(!admission.payload_live.load(Ordering::Acquire));
                assert_eq!(
                    PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire),
                    native_reserved
                );
                assert_eq!(
                    super::super::WireResources::current_with_capacity(capacity).process_live_bytes,
                    wire_reserved
                );
            }
            assert!(failures >= 4);
        }
    }
}

#[test]
fn artifact_identity_does_not_allow_batch_admission_capacity_or_generic_policy_substitution() {
    let public = publication(Capacity::Capsule);
    let frame = public.export_wire().unwrap();
    let pin = public.digest();
    let plan = || BatchPlan::new(&[Step::Lookup], &[descriptor(0, 1, Step::Lookup)]).unwrap();
    for generic in [false, true] {
        let local = Arc::new(
            if generic {
                Publication::import_wire_with_capacity(frame.as_bytes(), pin, Capacity::Capsule)
            } else {
                Publication::import_wire_streamed_single_window(
                    frame.as_bytes(),
                    pin,
                    Capacity::Decoder,
                )
            }
            .unwrap(),
        );
        assert_eq!(local.digest(), pin);
        let client = super::super::Client::new(Arc::clone(&local), pin).unwrap();
        let window = client.begin_wire([3; 32], &[Step::Lookup]).unwrap();
        let admission = BatchAdmission::new(&public, plan(), SharingProfile::FullShares).unwrap();
        assert!(matches!(window.into_batch(admission), Err(Error::Binding)));
    }
}

#[test]
fn batch_wire_admission_failure_refunds_native_and_destroys_plan_without_buffer_allocation() {
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let public = publication(capacity);
        let plan = BatchPlan::new(&[Step::Lookup], &[descriptor(0, 1, Step::Lookup)]).unwrap();
        let before = PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire);
        let wire_before =
            super::super::WireResources::current_with_capacity(capacity).process_live_bytes;
        let available = capacity.max_process_wire_bytes() - wire_before;
        let block = WirePermit::acquire(available, capacity).unwrap();
        let fail = payload::failure::after(0);
        assert!(matches!(
            BatchAdmission::new(&public, plan, SharingProfile::FullShares),
            Err(Error::Limit)
        ));
        drop(fail);
        assert_eq!(PROCESS_PUBLICATIONS.bytes.load(Ordering::Acquire), before);
        drop(block);
        assert_eq!(
            super::super::WireResources::current_with_capacity(capacity).process_live_bytes,
            wire_before
        );
    }
}
