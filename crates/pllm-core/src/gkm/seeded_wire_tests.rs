use super::*;

fn fixture() -> (Arc<Publication>, Client, [WorkerWire; 2]) {
    let matrix = I16Matrix::from_encoded(2, 3, &[3, -7, 11, -2, 0, 5], &[100; 3]).unwrap();
    let table_bytes: Vec<_> = [-0.0f32, 1.5, -2.25, 3.0, 4.0, -5.0, 6.0, 7.0, 8.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let table = FloatTable::from_le_bytes(9, 1, &table_bytes).unwrap();
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
    let bytes = publication.export_wire().unwrap();
    let workers = [Worker::Zero, Worker::One].map(|role| {
        let imported =
            Arc::new(Publication::import_wire(bytes.as_bytes(), publication.digest()).unwrap());
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
    (publication, client, workers)
}

const PLAN: [Step; 4] = [
    Step::Lookup,
    Step::Linear([1; 32]),
    Step::Lookup,
    Step::Linear([1; 32]),
];

fn run(seeded: bool) -> (Vec<Vec<u8>>, usize) {
    let (_, client, mut owners) = fixture();
    let mut window = if seeded {
        client.begin_seeded_wire([5; 32], &PLAN)
    } else {
        client.begin_wire([5; 32], &PLAN)
    }
    .unwrap();
    let starts = window.start_frames().unwrap();
    assert_eq!(starts[0].len(), HEADER + PLAN.len() * 33);
    assert_eq!(
        starts[1].len(),
        starts[0].len() + if seeded { 32 } else { 0 }
    );
    let mut workers = [
        owners[0].begin(starts[0].as_bytes()).unwrap(),
        owners[1].begin(starts[1].as_bytes()).unwrap(),
    ];
    let mut bytes = starts.iter().map(WireBytes::len).sum();
    let mut outputs = Vec::new();
    for (ordinal, step) in PLAN.iter().enumerate() {
        let requests = match step {
            Step::Lookup => window.lookup(if ordinal == 0 { 0 } else { 8 }),
            Step::Linear(_) => window.linear(&[-17, 21, 4]),
        }
        .unwrap();
        if seeded {
            assert_eq!(requests[1].len(), HEADER);
        }
        bytes += requests.iter().map(WireBytes::len).sum::<usize>();
        let zero = workers[0].execute(requests[0].as_bytes()).unwrap();
        let one = workers[1].execute(requests[1].as_bytes()).unwrap();
        let tentative = window
            .accept_replies(zero.as_bytes(), one.as_bytes())
            .unwrap();
        outputs.push(
            tentative
                .values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        );
    }
    let accepted = window.finish().unwrap();
    assert_eq!(
        accepted.sharing_profile(),
        if seeded {
            SharingProfile::Aes128WindowV1
        } else {
            SharingProfile::FullShares
        }
    );
    for worker in &mut workers {
        worker.finish().unwrap();
    }
    assert_eq!(outputs[0], (-0.0f32).to_le_bytes());
    assert_eq!(outputs[2], 8.0f32.to_le_bytes());
    let expected: Vec<_> = [-154i64, 54]
        .iter()
        .flat_map(|v| super::super::rescale_signed(*v).to_le_bytes())
        .collect();
    assert_eq!(outputs[1], expected);
    (outputs, bytes)
}

#[test]
fn full_seeded_wire_exact_parity_and_seed_only_byte_accounting() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let full = run(false);
    let seeded = run(true);
    assert_eq!(full.0, seeded.0);
    // One start adds exactly 32 key bytes; One's two linear/lookup bodies vanish.
    assert_eq!(seeded.1, full.1 + 32 - 2 * (3 * 8 + 2));
}

#[test]
fn seeded_wire_rejects_role_profile_ordinal_source_and_frame_substitutions() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    // All public binding fields plus numeric domain, role, request identity,
    // geometry/body framing. Every ignored failure remains retired.
    for offset in [
        0, 8, 9, 10, 42, 74, 106, 114, 146, 178, 182, 186, 219, 251, 252, 284, 288, 292,
    ] {
        let (_, client, mut owners) = fixture();
        let mut window = client.begin_seeded_wire([5; 32], &PLAN).unwrap();
        let starts = window.start_frames().unwrap();
        let mut worker = owners[1].begin(starts[1].as_bytes()).unwrap();
        let requests = window.lookup(0).unwrap();
        let mut bad = requests[1].as_bytes().to_vec();
        bad[offset] ^= 1;
        assert!(worker.execute(&bad).is_err(), "offset {offset}");
        assert!(matches!(
            worker.execute(requests[1].as_bytes()),
            Err(Error::Retired)
        ));
        assert!(worker.finish().is_err());
    }
    for role in [0, 1] {
        let (_, client, mut owners) = fixture();
        let mut window = client.begin_seeded_wire([5; 32], &PLAN).unwrap();
        let starts = window.start_frames().unwrap();
        assert!(owners[role].begin(starts[1 - role].as_bytes()).is_err());
        assert!(matches!(
            owners[role].begin(starts[role].as_bytes()),
            Err(Error::Retired)
        ));
    }
}

#[test]
fn seeded_wire_replay_cancellation_and_retained_frame_leases() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (publication, client, mut owners) = fixture();
    let mut window = client.begin_seeded_wire([5; 32], &PLAN).unwrap();
    let starts = window.start_frames().unwrap();
    let mut worker = owners[1].begin(starts[1].as_bytes()).unwrap();
    let request = window.lookup(0).unwrap();
    let reply = worker.execute(request[1].as_bytes()).unwrap();
    assert!(worker.execute(request[1].as_bytes()).is_err());
    assert!(worker.finish().is_err());
    assert!(owners[1].begin(starts[1].as_bytes()).is_err());
    window.abort();
    assert_eq!(publication.live_windows(), 1);
    drop(starts);
    assert_eq!(publication.live_windows(), 1);
    drop(request);
    assert_eq!(publication.live_windows(), 0);
    // Failed worker's payload/key state and reply retain worker-local lease.
    assert_eq!(worker.source.budget.publication.live_windows(), 1);
    let remote = Arc::clone(&worker.source.budget.publication);
    drop(worker);
    assert_eq!(remote.live_windows(), 1);
    drop(reply);
    assert_eq!(remote.live_windows(), 0);
}

#[test]
fn seeded_wire_start_and_expansion_allocation_failures_are_sticky() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for phase in 0..3 {
        let mut success = false;
        for index in 0..24 {
            let (_, client, mut owners) = fixture();
            let mut window = client.begin_seeded_wire([5; 32], &PLAN).unwrap();
            if phase == 0 {
                let fail = payload::failure::after(index);
                let result = window.start_frames();
                drop(fail);
                if result.is_ok() {
                    success = true;
                    break;
                }
                assert!(matches!(result, Err(Error::Allocation)));
                assert!(matches!(window.start_frames(), Err(Error::Retired)));
                continue;
            }
            let starts = window.start_frames().unwrap();
            if phase == 1 {
                let fail = payload::failure::after(index);
                let result = owners[1].begin(starts[1].as_bytes());
                drop(fail);
                if result.is_ok() {
                    success = true;
                    break;
                }
                assert!(matches!(result, Err(Error::Allocation)));
                assert!(matches!(
                    owners[1].begin(starts[1].as_bytes()),
                    Err(Error::Retired)
                ));
                continue;
            }
            let mut worker = owners[1].begin(starts[1].as_bytes()).unwrap();
            let request = window.lookup(0).unwrap();
            let fail = payload::failure::after(index);
            let result = worker.execute(request[1].as_bytes());
            drop(fail);
            if result.is_ok() {
                success = true;
                break;
            }
            assert!(matches!(result, Err(Error::Allocation)));
            assert!(matches!(
                worker.execute(request[1].as_bytes()),
                Err(Error::Retired)
            ));
        }
        assert!(success, "phase {phase}");
    }
}

#[test]
fn seeded_wire_corruption_still_requires_original_full_window_verification() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    for linear in [false, true] {
        let (_, client, mut owners) = fixture();
        let plan = [if linear {
            Step::Linear([1; 32])
        } else {
            Step::Lookup
        }];
        let mut window = client.begin_seeded_wire([5; 32], &plan).unwrap();
        let starts = window.start_frames().unwrap();
        let mut zero = owners[0].begin(starts[0].as_bytes()).unwrap();
        let mut one = owners[1].begin(starts[1].as_bytes()).unwrap();
        let request = if linear {
            window.linear(&[-17, 21, 4])
        } else {
            window.lookup(0)
        }
        .unwrap();
        let a = zero.execute(request[0].as_bytes()).unwrap();
        let b = one.execute(request[1].as_bytes()).unwrap();
        let mut corrupt = b.as_bytes().to_vec();
        corrupt[HEADER] ^= 1;
        if linear {
            // Fixed replies may produce tentative values, never acceptance.
            window.accept_replies(a.as_bytes(), &corrupt).unwrap();
            assert!(matches!(window.finish(), Err(Error::Verification)));
        } else {
            assert!(matches!(
                window.accept_replies(a.as_bytes(), &corrupt),
                Err(Error::Verification)
            ));
        }
        assert!(matches!(window.finish(), Err(Error::Retired)));
    }
}

#[test]
fn seeded_wire_fresh_purpose_keys_one_use_export_partial_admission_and_wrong_reply() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (_, client, mut owners) = fixture();
    let mut first = client.begin_seeded_wire([5; 32], &PLAN).unwrap();
    let a = first.start_frames().unwrap();
    let mut second = client.begin_seeded_wire([5; 32], &PLAN).unwrap();
    let b = second.start_frames().unwrap();
    let key_start = HEADER + PLAN.len() * 33;
    assert_ne!(
        &a[1].as_bytes()[key_start..key_start + 16],
        &a[1].as_bytes()[key_start + 16..]
    );
    assert_ne!(
        &a[1].as_bytes()[key_start..key_start + 16],
        &b[1].as_bytes()[key_start..key_start + 16]
    );
    assert_ne!(
        &a[1].as_bytes()[key_start + 16..],
        &b[1].as_bytes()[key_start + 16..]
    );
    let mut one = owners[1].begin(a[1].as_bytes()).unwrap();
    let request = first.lookup(0).unwrap();
    let reply = one.execute(request[1].as_bytes()).unwrap();
    // A reply from another native request cannot acknowledge remote progress.
    let _ = second.lookup(0).unwrap();
    assert!(second
        .accept_replies(reply.as_bytes(), reply.as_bytes())
        .is_err());
    assert!(matches!(second.finish(), Err(Error::Retired)));
    assert!(first.start_frames().is_err());
    assert!(matches!(first.finish(), Err(Error::Retired)));

    let (_, client, mut owners) = fixture();
    let mut window = client.begin_seeded_wire([5; 32], &PLAN).unwrap();
    let starts = window.start_frames().unwrap();
    let mut zero = owners[0].begin(starts[0].as_bytes()).unwrap();
    // Partial worker admission: missing key byte fails before stream construction.
    assert!(owners[1]
        .begin(&starts[1].as_bytes()[..starts[1].len() - 1])
        .is_err());
    zero.abort();
    window.abort();
    assert!(matches!(zero.execute(&[]), Err(Error::Retired)));
    assert!(matches!(
        owners[1].begin(starts[1].as_bytes()),
        Err(Error::Retired)
    ));
    assert!(matches!(window.finish(), Err(Error::Retired)));
}

#[test]
fn substituted_seed_material_cannot_bypass_reconstruction_or_final_verification() {
    let _accounting = crate::gkm::tests::process_accounting_guard();
    let (_, client, mut owners) = fixture();
    let mut window = client
        .begin_seeded_wire([5; 32], &[Step::Linear([1; 32])])
        .unwrap();
    let starts = window.start_frames().unwrap();
    let mut changed = starts[1].as_bytes().to_vec();
    // This deliberately bypasses channel authentication in the fixture.
    // Native correctness checks must remain authoritative regardless.
    changed[HEADER + 33] ^= 1;
    let mut zero = owners[0].begin(starts[0].as_bytes()).unwrap();
    let mut one = owners[1].begin(&changed).unwrap();
    let request = window.linear(&[-17, 21, 4]).unwrap();
    let a = zero.execute(request[0].as_bytes()).unwrap();
    let b = one.execute(request[1].as_bytes()).unwrap();
    if window.accept_replies(a.as_bytes(), b.as_bytes()).is_ok() {
        assert!(window.finish().is_err());
    }
    assert!(matches!(window.finish(), Err(Error::Retired)));
}
