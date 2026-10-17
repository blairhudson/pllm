use std::sync::{Arc, Mutex};

use pllm_core::gkm::{
    Client, Error, FloatTable, I16Matrix, Publication, Stage, Step, WireResources, Worker,
    WorkerWire, MAX_PROCESS_WIRE_BYTES, MAX_PUBLICATION_WIRE_BYTES, MAX_WIRE_FRAME_BYTES,
    NUMERIC_PROFILE,
};
use sha2::{Digest, Sha256};

const STAGE: [u8; 32] = [7; 32];
const HEADER: usize = 296;
// Resource checks share process counters with the behavioral fixtures. Keep
// this binary deterministic even when invoked without --test-threads=1.
static SERIAL: Mutex<()> = Mutex::new(());

fn source() -> (I16Matrix, FloatTable, Vec<u8>) {
    let matrix = I16Matrix::from_encoded(2, 3, &[2048, -300, 17, 0, 0, 0], &[4096; 3]).unwrap();
    let bytes: Vec<u8> = (0..9)
        .flat_map(|row| [row as f32 / 8.0, -0.0, -(row as f32) / 16.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let table = FloatTable::from_le_bytes(9, 3, &bytes).unwrap();
    (matrix, table, bytes)
}

fn publication(matrix: &I16Matrix, table: &FloatTable) -> Arc<Publication> {
    Arc::new(Publication::publish([2; 32], &[Stage { id: STAGE, matrix }], table).unwrap())
}

fn worker(
    publication: &Arc<Publication>,
    role: Worker,
    matrix: &I16Matrix,
    table: &FloatTable,
) -> WorkerWire {
    WorkerWire::new(
        Arc::clone(publication),
        publication.digest(),
        role,
        &[Stage { id: STAGE, matrix }],
        table,
    )
    .unwrap()
}

fn repin(bytes: &mut [u8]) -> [u8; 32] {
    // A deliberately malicious *trusted* fixture reaches structural checks even
    // after modifying the payload. Production callers must never self-pin this way.
    let mut hash = Sha256::new();
    hash.update(b"pllm.gkm.owner-publication.v1");
    hash.update(NUMERIC_PROFILE);
    hash.update(&bytes[40..]);
    let digest: [u8; 32] = hash.finalize().into();
    bytes[8..40].copy_from_slice(&digest);
    digest
}

#[test]
fn exact_oracle_publication_roundtrip_and_dependent_full_share_requests() {
    let _serial = SERIAL.lock().unwrap();
    let (matrix, table, bytes) = source();
    let original = publication(&matrix, &table);
    let exported = original.export_wire().unwrap();
    let imported =
        Arc::new(Publication::import_wire(exported.as_bytes(), original.digest()).unwrap());
    assert_eq!(imported.digest(), original.digest());
    assert_eq!(imported.context(), original.context());
    assert_eq!(
        imported.export_wire().unwrap().as_bytes(),
        exported.as_bytes()
    );
    let client = Client::new(Arc::clone(&imported), original.digest()).unwrap();
    let mut zero = worker(&imported, Worker::Zero, &matrix, &table);
    let mut one = worker(&imported, Worker::One, &matrix, &table);
    for token in 0..9 {
        let mut window = client
            .begin_wire([9; 32], &[Step::Lookup, Step::Linear(STAGE)])
            .unwrap();
        let start = window.start_frames().unwrap();
        let mut left = zero.begin(start[0].as_bytes()).unwrap();
        let mut right = one.begin(start[1].as_bytes()).unwrap();
        let request = window.lookup(token).unwrap();
        assert_eq!(request[0].len(), HEADER + 2);
        assert_eq!(request[1].len(), HEADER + 2);
        // Independent share reconstruction: only one role's selector per frame.
        for bit in 0..16 {
            assert_eq!(
                ((request[0].as_bytes()[HEADER + bit / 8]
                    ^ request[1].as_bytes()[HEADER + bit / 8])
                    >> (bit % 8))
                    & 1,
                u8::from(bit == token)
            );
        }
        let a = left.execute(request[0].as_bytes()).unwrap();
        let b = right.execute(request[1].as_bytes()).unwrap();
        let row = window.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
        assert!(row.signed.is_empty());
        assert_eq!(
            row.values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            bytes[token * 12..(token + 1) * 12]
        );
        let x = [token as i64 * 256, -13, -(token as i64) * 128];
        let request = window.linear(&x).unwrap();
        for (index, input) in x.iter().enumerate() {
            let offset = HEADER + index * 8;
            let a = u64::from_le_bytes(
                request[0].as_bytes()[offset..offset + 8]
                    .try_into()
                    .unwrap(),
            );
            let b = u64::from_le_bytes(
                request[1].as_bytes()[offset..offset + 8]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(a.wrapping_add(b), *input as u64);
        }
        let a = left.execute(request[0].as_bytes()).unwrap();
        let b = right.execute(request[1].as_bytes()).unwrap();
        let result = window.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
        let exact = 2048i128 * x[0] as i128 - 300i128 * x[1] as i128 + 17i128 * x[2] as i128;
        assert_eq!(result.signed, &[exact as i64, 0]);
        assert_eq!(result.values, &[(exact as i64 as f32) / 4_194_304.0, 0.0]);
        assert_eq!(left.finish().unwrap().worker, Worker::Zero);
        assert_eq!(right.finish().unwrap().request_id, window.request_id());
        let accepted = window.finish().unwrap();
        assert_eq!(accepted.steps(), 2);
        assert_eq!(accepted.request_id(), window.request_id());
        assert_eq!(accepted.signed_linear_output(), &[exact as i64, 0]);
        assert!(matches!(window.finish(), Err(Error::Retired)));
    }
}

#[test]
fn publication_pin_truncation_points_geometry_and_ranges_reject() {
    let _serial = SERIAL.lock().unwrap();
    let (matrix, table, _) = source();
    let public = publication(&matrix, &table);
    let exported = public.export_wire().unwrap();
    assert!(matches!(
        Publication::import_wire(exported.as_bytes(), [0; 32]),
        Err(Error::Binding)
    ));
    for end in 0..exported.len() {
        assert!(Publication::import_wire(&exported.as_bytes()[..end], public.digest()).is_err());
    }
    let mut bad = exported.as_bytes().to_vec();
    bad.push(0);
    let pin = repin(&mut bad);
    assert!(Publication::import_wire(&bad, pin).is_err());
    let mut bad = exported.as_bytes().to_vec();
    bad[72..80].copy_from_slice(&u64::MAX.to_le_bytes()); // stage count
    let pin = repin(&mut bad);
    assert!(matches!(
        Publication::import_wire(&bad, pin),
        Err(Error::Limit)
    ));
    let mut bad = exported.as_bytes().to_vec();
    bad[89] = 4; // compressed R point must not accept an uncompressed tag
    let pin = repin(&mut bad);
    assert!(matches!(
        Publication::import_wire(&bad, pin),
        Err(Error::Binding)
    ));
    let mut bad = exported.as_bytes().to_vec();
    bad[90..122].fill(0xff); // invalid coordinate
    let pin = repin(&mut bad);
    assert!(matches!(
        Publication::import_wire(&bad, pin),
        Err(Error::Binding)
    ));
    let mut at = 88;
    for _ in 0..2 {
        at += 1 + exported.as_bytes()[at] as usize;
    }
    let bound = at + 64 + 16;
    let mut bad = exported.as_bytes().to_vec();
    bad[bound..bound + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    let pin = repin(&mut bad);
    assert!(matches!(
        Publication::import_wire(&bad, pin),
        Err(Error::Bound)
    ));
    let mut bad = exported.as_bytes().to_vec();
    bad[bound + 24..bound + 32].copy_from_slice(&(1u64 << 63).to_le_bytes());
    let pin = repin(&mut bad);
    assert!(matches!(
        Publication::import_wire(&bad, pin),
        Err(Error::Bound)
    ));
    assert!(matches!(
        Publication::import_wire(&vec![0; MAX_PUBLICATION_WIRE_BYTES + 1], public.digest()),
        Err(Error::Limit)
    ));
}

#[test]
fn all_header_substitutions_and_truncations_retire_client_and_worker() {
    let _serial = SERIAL.lock().unwrap();
    let (matrix, table, _) = source();
    let public = publication(&matrix, &table);
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    // Magic, kind, sharing/numeric profiles, publication, session, sequence,
    // context, plan, total, ordinal, stage kind/id, artifact, role, window,
    // rows/columns/length. Every public binding field is exercised.
    let offsets = [
        0, 8, 9, 10, 42, 74, 106, 114, 146, 178, 182, 186, 187, 219, 251, 252, 284, 288, 292,
    ];
    for offset in offsets {
        let mut zero = worker(&public, Worker::Zero, &matrix, &table);
        let mut one = worker(&public, Worker::One, &matrix, &table);
        let mut window = client.begin_wire([9; 32], &[Step::Linear(STAGE)]).unwrap();
        let start = window.start_frames().unwrap();
        let mut left = zero.begin(start[0].as_bytes()).unwrap();
        let mut right = one.begin(start[1].as_bytes()).unwrap();
        let request = window.linear(&[1, 2, 3]).unwrap();
        let a = left.execute(request[0].as_bytes()).unwrap();
        let b = right.execute(request[1].as_bytes()).unwrap();
        let mut bad = a.as_bytes().to_vec();
        bad[offset] ^= 1;
        assert!(window.accept_replies(&bad, b.as_bytes()).is_err());
        assert!(matches!(
            window.accept_replies(a.as_bytes(), b.as_bytes()),
            Err(Error::Retired)
        ));
        assert!(window.finish().is_err());
        let mut zero2 = worker(&public, Worker::Zero, &matrix, &table);
        let mut again = zero2.begin(start[0].as_bytes()).unwrap();
        let mut bad = request[0].as_bytes().to_vec();
        bad[offset] ^= 1;
        assert!(again.execute(&bad).is_err());
        assert!(matches!(
            again.execute(request[0].as_bytes()),
            Err(Error::Retired)
        ));
        assert!(again.finish().is_err());
    }
    for cut in [0, 8, 40, HEADER - 1, HEADER, HEADER + 23] {
        let mut zero = worker(&public, Worker::Zero, &matrix, &table);
        let mut window = client.begin_wire([9; 32], &[Step::Linear(STAGE)]).unwrap();
        let start = window.start_frames().unwrap();
        let mut left = zero.begin(start[0].as_bytes()).unwrap();
        let request = window.linear(&[1, 2, 3]).unwrap();
        assert!(left.execute(&request[0].as_bytes()[..cut]).is_err());
        assert!(left.execute(request[0].as_bytes()).is_err());
        assert!(window
            .accept_replies(&request[0].as_bytes()[..cut], &[])
            .is_err());
        assert!(window.finish().is_err());
    }
}

#[test]
fn lookup_hash_and_final_linear_verification_stay_authoritative() {
    let _serial = SERIAL.lock().unwrap();
    let (matrix, table, _) = source();
    let public = publication(&matrix, &table);
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    for step in [Step::Lookup, Step::Linear(STAGE)] {
        let mut zero = worker(&public, Worker::Zero, &matrix, &table);
        let mut one = worker(&public, Worker::One, &matrix, &table);
        let mut window = client.begin_wire([9; 32], &[step]).unwrap();
        let start = window.start_frames().unwrap();
        let mut left = zero.begin(start[0].as_bytes()).unwrap();
        let mut right = one.begin(start[1].as_bytes()).unwrap();
        let request = match step {
            Step::Lookup => window.lookup(8),
            _ => window.linear(&[1, 2, 3]),
        }
        .unwrap();
        let a = left.execute(request[0].as_bytes()).unwrap();
        let b = right.execute(request[1].as_bytes()).unwrap();
        let mut bad = a.as_bytes().to_vec();
        if step == Step::Lookup {
            bad[HEADER] ^= 1;
            assert!(matches!(
                window.accept_replies(&bad, b.as_bytes()),
                Err(Error::Verification)
            ));
        } else {
            let original = u64::from_le_bytes(bad[HEADER..HEADER + 8].try_into().unwrap());
            bad[HEADER..HEADER + 8].copy_from_slice(&original.wrapping_add(1).to_le_bytes());
            window.accept_replies(&bad, b.as_bytes()).unwrap();
            assert!(matches!(window.finish(), Err(Error::Verification)));
        }
        assert!(window.finish().is_err());
    }
}

#[test]
fn worker_replay_wrong_source_schedule_and_cancellation_are_sticky() {
    let _serial = SERIAL.lock().unwrap();
    let (matrix, table, _) = source();
    let public = publication(&matrix, &table);
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let wrong = I16Matrix::from_encoded(2, 3, &[2047, -300, 17, 0, 0, 0], &[4096; 3]).unwrap();
    assert!(WorkerWire::new(
        Arc::clone(&public),
        public.digest(),
        Worker::Zero,
        &[Stage {
            id: STAGE,
            matrix: &wrong
        }],
        &table
    )
    .is_err());
    let mut zero = worker(&public, Worker::Zero, &matrix, &table);
    let mut window = client.begin_wire([9; 32], &[Step::Lookup]).unwrap();
    let start = window.start_frames().unwrap();
    let mut left = zero.begin(start[0].as_bytes()).unwrap();
    let request = window.lookup(8).unwrap();
    let reply = left.execute(request[0].as_bytes()).unwrap();
    left.finish().unwrap();
    assert!(zero.begin(start[0].as_bytes()).is_err()); // replay previous sequence
    assert!(zero.begin(start[0].as_bytes()).is_err());
    drop((left, zero, reply));
    let mut zero = worker(&public, Worker::Zero, &matrix, &table);
    let mut left = zero.begin(start[0].as_bytes()).unwrap();
    left.execute(request[0].as_bytes()).unwrap();
    assert!(left.execute(request[0].as_bytes()).is_err()); // duplicate ordinal
    assert!(left.finish().is_err());
    drop((left, zero));
    let mut zero = worker(&public, Worker::Zero, &matrix, &table);
    let mut left = zero.begin(start[0].as_bytes()).unwrap();
    left.abort();
    assert!(left.execute(request[0].as_bytes()).is_err());
    window.abort();
    assert!(window.lookup(8).is_err());
    drop((left, zero));
    let mut zero = worker(&public, Worker::Zero, &matrix, &table);
    let mut bad = start[0].as_bytes().to_vec();
    bad[HEADER + 1] = 1; // noncanonical lookup stage ID in authenticated schedule
    assert!(zero.begin(&bad).is_err());
    assert!(zero.begin(start[0].as_bytes()).is_err());
}

#[test]
fn seeded_network_rejected_and_native_seeded_path_remains_available() {
    let _serial = SERIAL.lock().unwrap();
    let (matrix, table, _) = source();
    let public = publication(&matrix, &table);
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let (window, seeded) = client.begin_seeded([9; 32], &[Step::Lookup]).unwrap();
    assert!(matches!(window.into_wire(), Err(Error::Binding)));
    drop(seeded);
    let mut window = client.begin_wire([9; 32], &[Step::Lookup]).unwrap();
    let start = window.start_frames().unwrap();
    let mut zero = worker(&public, Worker::Zero, &matrix, &table);
    let mut bad = start[0].as_bytes().to_vec();
    bad[9] = 1;
    assert!(matches!(zero.begin(&bad), Err(Error::Binding)));
    assert!(zero.begin(start[0].as_bytes()).is_err());
}

#[test]
fn wire_global_cap_retained_packets_and_publication_leases() {
    let _serial = SERIAL.lock().unwrap();
    // This binary is run with --test-threads=1; process counters are shared.
    let before = WireResources::current().process_live_bytes;
    let (matrix, table, _) = source();
    let public = publication(&matrix, &table);
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let mut retained = Vec::new();
    for _ in 0..8 {
        let mut window = client.begin_wire([9; 32], &[Step::Lookup]).unwrap();
        let start = window.start_frames().unwrap();
        drop(start);
        retained.push(window.lookup(8).unwrap());
        window.abort();
    }
    assert_eq!(public.live_windows(), 8);
    assert!(matches!(
        client.begin_wire([9; 32], &[Step::Lookup]),
        Err(Error::Limit)
    ));
    drop(retained);
    assert_eq!(public.live_windows(), 0);
    assert_eq!(WireResources::current().process_live_bytes, before);
    let mut artifacts = Vec::new();
    loop {
        match public.export_wire() {
            Ok(bytes) => artifacts.push(bytes),
            Err(Error::Limit) => break,
            Err(other) => panic!("unexpected {other:?}"),
        }
    }
    assert!(WireResources::current().process_live_bytes <= MAX_PROCESS_WIRE_BYTES);
    assert!(MAX_PROCESS_WIRE_BYTES - WireResources::current().process_live_bytes < 2048);
    assert!(matches!(public.export_wire(), Err(Error::Limit)));
    drop(artifacts);
    assert_eq!(WireResources::current().process_live_bytes, before);
    let mut window = client.begin_wire([9; 32], &[Step::Lookup]).unwrap();
    let start = window.start_frames().unwrap();
    let mut zero = worker(&public, Worker::Zero, &matrix, &table);
    let mut left = zero.begin(start[0].as_bytes()).unwrap();
    assert!(matches!(
        left.execute(&vec![0; MAX_WIRE_FRAME_BYTES + 1]),
        Err(Error::Limit)
    ));
    assert!(left.finish().is_err());
}

#[test]
fn canonical_identity_duplicate_stage_and_exact_table_hash_import_checks() {
    let _serial = SERIAL.lock().unwrap();
    let zero = I16Matrix::from_encoded(1, 1, &[0], &[1u64 << 63]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &(-0.0f32).to_le_bytes()).unwrap();
    let public = Arc::new(
        Publication::publish(
            [2; 32],
            &[
                Stage {
                    id: [1; 32],
                    matrix: &zero,
                },
                Stage {
                    id: [2; 32],
                    matrix: &zero,
                },
            ],
            &table,
        )
        .unwrap(),
    );
    let packet = public.export_wire().unwrap();
    let imported = Publication::import_wire(packet.as_bytes(), public.digest()).unwrap();
    assert_eq!(
        imported.export_wire().unwrap().as_bytes(),
        packet.as_bytes()
    );
    let first = 88 + 1 + packet.as_bytes()[88] as usize;
    // Each one-by-one matrix has 80 metadata, 16 bounds, and the canonical
    // two-byte length-prefixed identity S (length=1, byte=0).
    assert_eq!(&packet.as_bytes()[first + 96..first + 98], &[1, 0]);
    let second = first + 98;
    let mut duplicate = packet.as_bytes().to_vec();
    duplicate[second..second + 32].fill(1);
    let pin = repin(&mut duplicate);
    assert!(matches!(
        Publication::import_wire(&duplicate, pin),
        Err(Error::Binding)
    ));
    let mut row_hash = packet.as_bytes().to_vec();
    *row_hash.last_mut().unwrap() ^= 1;
    let pin = repin(&mut row_hash);
    let changed = Arc::new(Publication::import_wire(&row_hash, pin).unwrap());
    assert!(WorkerWire::new(
        changed,
        pin,
        Worker::Zero,
        &[
            Stage {
                id: [1; 32],
                matrix: &zero
            },
            Stage {
                id: [2; 32],
                matrix: &zero
            },
        ],
        &table
    )
    .is_err());
    let other = FloatTable::from_le_bytes(1, 1, &0.0f32.to_le_bytes()).unwrap();
    assert!(WorkerWire::new(
        Arc::clone(&public),
        public.digest(),
        Worker::Zero,
        &[
            Stage {
                id: [1; 32],
                matrix: &zero
            },
            Stage {
                id: [2; 32],
                matrix: &zero
            },
        ],
        &other
    )
    .is_err());
}

#[test]
fn ignored_duplicate_partial_reply_padding_and_incomplete_calls_burn_handles() {
    let _serial = SERIAL.lock().unwrap();
    let (matrix, table, _) = source();
    let public = publication(&matrix, &table);
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    for phase in 0..7 {
        let mut zero = worker(&public, Worker::Zero, &matrix, &table);
        let mut one = worker(&public, Worker::One, &matrix, &table);
        let mut window = client.begin_wire([9; 32], &[Step::Lookup]).unwrap();
        let start = window.start_frames().unwrap();
        if phase == 0 {
            assert!(window.start_frames().is_err());
            assert!(matches!(window.lookup(8), Err(Error::Retired)));
            continue;
        }
        let mut left = zero.begin(start[0].as_bytes()).unwrap();
        let mut right = one.begin(start[1].as_bytes()).unwrap();
        let request = window.lookup(8).unwrap();
        if phase == 1 {
            assert!(window.lookup(8).is_err());
            assert!(window.finish().is_err());
            continue;
        }
        if phase == 2 {
            let mut bad = request[0].as_bytes().to_vec();
            bad[HEADER + 1] |= 128;
            assert!(matches!(left.execute(&bad), Err(Error::Shape)));
            assert!(matches!(
                left.execute(request[0].as_bytes()),
                Err(Error::Retired)
            ));
            continue;
        }
        if phase == 3 {
            assert!(matches!(left.finish(), Err(Error::Incomplete)));
            assert!(matches!(
                left.execute(request[0].as_bytes()),
                Err(Error::Retired)
            ));
            continue;
        }
        let a = left.execute(request[0].as_bytes()).unwrap();
        let b = right.execute(request[1].as_bytes()).unwrap();
        if phase == 4 {
            assert!(window.accept_replies(a.as_bytes(), &[]).is_err());
            assert!(matches!(
                window.accept_replies(a.as_bytes(), b.as_bytes()),
                Err(Error::Retired)
            ));
            continue;
        }
        window.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
        if phase == 5 {
            assert!(window.accept_replies(a.as_bytes(), b.as_bytes()).is_err());
            assert!(matches!(window.finish(), Err(Error::Retired)));
        } else {
            left.finish().unwrap();
            assert!(left.finish().is_err());
            assert!(zero.begin(start[0].as_bytes()).is_err());
        }
    }
}

#[test]
fn valid_previous_window_packets_cannot_substitute_for_next_sequence() {
    let _serial = SERIAL.lock().unwrap();
    let (matrix, table, _) = source();
    let public = publication(&matrix, &table);
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let mut zero = worker(&public, Worker::Zero, &matrix, &table);
    let mut one = worker(&public, Worker::One, &matrix, &table);
    let mut old = client.begin_wire([9; 32], &[Step::Lookup]).unwrap();
    let old_start = old.start_frames().unwrap();
    let mut left = zero.begin(old_start[0].as_bytes()).unwrap();
    let mut right = one.begin(old_start[1].as_bytes()).unwrap();
    let old_request = old.lookup(8).unwrap();
    let a = left.execute(old_request[0].as_bytes()).unwrap();
    let b = right.execute(old_request[1].as_bytes()).unwrap();
    old.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
    left.finish().unwrap();
    right.finish().unwrap();
    old.finish().unwrap();
    let mut next = client.begin_wire([9; 32], &[Step::Lookup]).unwrap();
    let next_start = next.start_frames().unwrap();
    let mut next_left = zero.begin(next_start[0].as_bytes()).unwrap();
    let mut next_right = one.begin(next_start[1].as_bytes()).unwrap();
    let next_request = next.lookup(8).unwrap();
    assert!(matches!(
        next_left.execute(old_request[0].as_bytes()),
        Err(Error::Binding)
    ));
    assert!(matches!(
        next_left.execute(next_request[0].as_bytes()),
        Err(Error::Retired)
    ));
    let next_b = next_right.execute(next_request[1].as_bytes()).unwrap();
    // The old reply has a correct internal request hash and shape; only its
    // authenticated window/sequence differs. Native slot binding must reject it.
    assert!(matches!(
        next.accept_replies(a.as_bytes(), next_b.as_bytes()),
        Err(Error::Binding)
    ));
    assert!(matches!(next.finish(), Err(Error::Retired)));
}
