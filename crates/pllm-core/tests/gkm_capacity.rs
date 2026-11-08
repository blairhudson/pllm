//! Decoder admission on small, asymmetric fixtures. No model payloads or
//! maximum-dimension group operations are needed to exercise the new geometry.
use pllm_core::gkm::*;
use std::sync::{Arc, Mutex};

static SERIAL: Mutex<()> = Mutex::new(());
const CONTEXT: [u8; 32] = [3; 32];
const OUT: [u8; 32] = [1; 32];
const IN: [u8; 32] = [2; 32];

fn matrix(rows: usize, columns: usize) -> I16Matrix {
    let weights = (0..rows * columns).map(|i| (i % 7) as i16 - 3).collect();
    I16Matrix::from_encoded_owned_with_capacity(
        rows,
        columns,
        weights,
        vec![8; columns],
        Capacity::Decoder,
    )
    .unwrap()
}

fn table(vocabulary: usize, width: usize) -> FloatTable {
    let mut bytes = vec![0; vocabulary * width * 4];
    for (i, word) in bytes.chunks_exact_mut(4).enumerate() {
        let value = if i % 3 == 0 {
            -0.0
        } else {
            (i % 23) as f32 / 8.0
        };
        word.copy_from_slice(&value.to_le_bytes());
    }
    FloatTable::from_le_bytes_owned_with_capacity(vocabulary, width, bytes, Capacity::Decoder)
        .unwrap()
}

fn worker(
    publication: &Arc<Publication>,
    role: Worker,
    stages: &[Stage<'_>],
    table: &FloatTable,
) -> WorkerWire {
    WorkerWire::new(
        Arc::clone(publication),
        publication.digest(),
        role,
        stages,
        table,
    )
    .unwrap()
}

#[test]
fn decoder_high_token_ids_wire_lookup_reconstructs_exact_float_rows() {
    let _serial = SERIAL.lock().unwrap();
    const VOCABULARY: usize = 151646;
    const WIDTH: usize = 2;
    let tokens = [151644usize, 151645];
    let rows = [[1.25f32, -2.5], [-0.0, 17.75]];
    let alias_rows = [[32.0f32, -64.0], [-128.0, 256.0]];
    let mut expected = [[0u8; WIDTH * 4]; 2];
    let mut bytes = vec![0u8; VOCABULARY * WIDTH * 4];
    for (i, &token) in tokens.iter().enumerate() {
        let alias = usize::from(token as u16);
        assert!(token > usize::from(u16::MAX));
        assert_ne!(token, alias);
        for column in 0..WIDTH {
            assert!(rows[i][column].is_finite());
            assert!(alias_rows[i][column].is_finite());
            expected[i][column * 4..(column + 1) * 4]
                .copy_from_slice(&rows[i][column].to_le_bytes());
            bytes[(token * WIDTH + column) * 4..(token * WIDTH + column + 1) * 4]
                .copy_from_slice(&rows[i][column].to_le_bytes());
            bytes[(alias * WIDTH + column) * 4..(alias * WIDTH + column + 1) * 4]
                .copy_from_slice(&alias_rows[i][column].to_le_bytes());
        }
        assert_eq!(&bytes[token * WIDTH * 4..(token + 1) * WIDTH * 4], &expected[i]);
        assert_ne!(&bytes[alias * WIDTH * 4..(alias + 1) * WIDTH * 4], &expected[i]);
    }
    assert_ne!(expected[0], expected[1]);
    // Independent raw expected rows stay outside the table owner consumed here.
    let table = FloatTable::from_le_bytes_owned_with_capacity(
        VOCABULARY, WIDTH, bytes, Capacity::Decoder,
    ).unwrap();
    let matrix = matrix(1, 1);
    let stages = [Stage { id: OUT, matrix: &matrix }];
    let original = Publication::publish_with_capacity(
        CONTEXT, &stages, &table, Capacity::Decoder,
    ).unwrap();
    let encoded = original.export_wire().unwrap();
    let public = Arc::new(Publication::import_wire_with_capacity(
        encoded.as_bytes(), original.digest(), Capacity::Decoder,
    ).unwrap());
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let mut zero = worker(&public, Worker::Zero, &stages, &table);
    let mut one = worker(&public, Worker::One, &stages, &table);
    let plan = [Step::Lookup, Step::Lookup];
    let mut previous_request = None;
    // Reuse immutable worker owners, never window inventory or seeded material.
    for seeded in [false, true] {
        let mut window = if seeded {
            client.begin_seeded_wire(CONTEXT, &plan)
        } else {
            client.begin_wire(CONTEXT, &plan)
        }.unwrap();
        let request_id = window.request_id();
        if let Some(previous) = previous_request {
            assert_ne!(request_id, previous);
        }
        previous_request = Some(request_id);
        let start = window.start_frames().unwrap();
        let mut left = zero.begin_bound(start[0].as_bytes(), CONTEXT, &plan).unwrap();
        let mut right = one.begin_bound(start[1].as_bytes(), CONTEXT, &plan).unwrap();
        assert_eq!(left.request_id(), request_id);
        assert_eq!(right.request_id(), request_id);
        for (i, &token) in tokens.iter().enumerate() {
            let request = window.lookup(token).unwrap();
            let a = left.execute(request[0].as_bytes()).unwrap();
            let b = right.execute(request[1].as_bytes()).unwrap();
            let row = window.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
            assert_eq!(row.values.len(), WIDTH);
            let mut reconstructed = [0u8; WIDTH * 4];
            for (column, (&value, word)) in row.values.iter()
                .zip(expected[i].chunks_exact(4)).enumerate()
            {
                let expected_value = f32::from_le_bytes(word.try_into().unwrap());
                assert_eq!(value.to_bits(), expected_value.to_bits());
                reconstructed[column * 4..(column + 1) * 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
            assert_eq!(reconstructed, expected[i]);
            drop((request, a, b));
        }
        assert_eq!(left.finish().unwrap().steps, plan.len());
        assert_eq!(right.finish().unwrap().steps, plan.len());
        assert_eq!(window.finish().unwrap().steps(), plan.len());
        drop((start, window, left, right));
    }
    // Worker source permits and last-run leases persist across the two windows;
    // explicitly retire these owners before checking complete logical release.
    drop((zero, one, client, encoded, original));
    assert_eq!(public.live_windows(), 0);
    assert_eq!(
        WireResources::current_with_capacity(Capacity::Decoder).process_live_bytes,
        0
    );
}

#[test]
fn asymmetric_wire_requests_roundtrip_and_reconstruct_exact_signed_outputs() {
    let _serial = SERIAL.lock().unwrap();
    asymmetric_roundtrip(false);
    asymmetric_roundtrip(true);
}

fn asymmetric_roundtrip(seeded: bool) {
    let out = matrix(1031, 3);
    let input = matrix(2, 1033);
    let table = table(9, 2053);
    let stages = [
        Stage {
            id: OUT,
            matrix: &out,
        },
        Stage {
            id: IN,
            matrix: &input,
        },
    ];
    assert!(matches!(
        Publication::publish(CONTEXT, &stages, &table),
        Err(Error::Limit)
    ));
    let original =
        Publication::publish_with_capacity(CONTEXT, &stages, &table, Capacity::Decoder).unwrap();
    let encoded = original.export_wire().unwrap();
    assert_eq!(encoded.capacity(), Capacity::Decoder);
    assert!(matches!(
        Publication::import_wire(encoded.as_bytes(), original.digest()),
        Err(Error::Limit)
    ));
    let public = Arc::new(
        Publication::import_wire_with_capacity(
            encoded.as_bytes(),
            original.digest(),
            Capacity::Decoder,
        )
        .unwrap(),
    );
    assert_eq!(encoded.as_bytes(), public.export_wire().unwrap().as_bytes());
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let mut zero = worker(&public, Worker::Zero, &stages, &table);
    let mut one = worker(&public, Worker::One, &stages, &table);
    let plan = [Step::Lookup, Step::Linear(OUT), Step::Linear(IN)];
    let mut window = if seeded {
        client.begin_seeded_wire(CONTEXT, &plan)
    } else {
        client.begin_wire(CONTEXT, &plan)
    }
    .unwrap();
    let start = window.start_frames().unwrap();
    let mut left = zero
        .begin_bound(start[0].as_bytes(), CONTEXT, &plan)
        .unwrap();
    let mut right = one
        .begin_bound(start[1].as_bytes(), CONTEXT, &plan)
        .unwrap();
    let request = window.lookup(8).unwrap();
    let a = left.execute(request[0].as_bytes()).unwrap();
    let b = right.execute(request[1].as_bytes()).unwrap();
    assert!(a.len() > MAX_WIRE_FRAME_BYTES);
    let row = window.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
    let mut selector = [0u8; 2];
    selector[1] = 1;
    let expected = table.xor_rows(&selector).unwrap();
    for (value, bytes) in row.values.iter().zip(expected.chunks_exact(4)) {
        assert_eq!(value.to_le_bytes(), bytes);
    }
    for (rows, columns) in [out.shape(), input.shape()] {
        let x: Vec<_> = (0..columns).map(|i| (i % 9) as i64 - 4).collect();
        let request = window.linear(&x).unwrap();
        assert_eq!(request[0].len(), 296 + columns * 8);
        assert_eq!(request[1].len(), 296 + if seeded { 0 } else { columns * 8 });
        let a = left.execute(request[0].as_bytes()).unwrap();
        let b = right.execute(request[1].as_bytes()).unwrap();
        assert_eq!(a.len(), 296 + rows * 8);
        let actual = window.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
        for (row, (&signed, &value)) in actual.signed.iter().zip(actual.values).enumerate() {
            let exact: i128 = x
                .iter()
                .enumerate()
                .map(|(col, &x)| (((row * columns + col) % 7) as i128 - 3) * x as i128)
                .sum();
            assert_eq!(signed as i128, exact);
            assert_eq!(
                value.to_bits(),
                ((exact as i64 as f32) / 4_194_304.0).to_bits()
            );
        }
    }
    left.finish().unwrap();
    right.finish().unwrap();
    assert_eq!(window.finish().unwrap().steps(), plan.len());
    drop((
        start, request, a, b, window, left, right, zero, one, client, encoded, original,
    ));
    assert_eq!(public.live_windows(), 0);
    assert_eq!(
        WireResources::current_with_capacity(Capacity::Decoder).process_live_bytes,
        0
    );
}

#[test]
fn decoder_stage_and_schedule_limits_and_large_vocabulary_use_owner_geometry() {
    let _serial = SERIAL.lock().unwrap();
    let matrix = matrix(1, 1);
    let table = table(4097, 1);
    let ids: Vec<[u8; 32]> = (0..256u16)
        .map(|i| {
            let mut id = [0; 32];
            id[..2].copy_from_slice(&i.to_le_bytes());
            id
        })
        .collect();
    let stages: Vec<_> = ids
        .iter()
        .map(|&id| Stage {
            id,
            matrix: &matrix,
        })
        .collect();
    let original =
        Publication::publish_with_capacity(CONTEXT, &stages, &table, Capacity::Decoder).unwrap();
    let encoded = original.export_wire().unwrap();
    let public = Arc::new(
        Publication::import_wire_with_capacity(
            encoded.as_bytes(),
            original.digest(),
            Capacity::Decoder,
        )
        .unwrap(),
    );
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let plan = vec![Step::Lookup; 257];
    let mut window = client.begin_wire(CONTEXT, &plan).unwrap();
    let start = window.start_frames().unwrap();
    assert!(start[0].len() > MAX_WIRE_FRAME_BYTES);
    let mut zero = worker(&public, Worker::Zero, &stages, &table);
    let mut one = worker(&public, Worker::One, &stages, &table);
    let mut left = zero.begin(start[0].as_bytes()).unwrap();
    let mut right = one.begin(start[1].as_bytes()).unwrap();
    for token in 0..plan.len() {
        let requests = window.lookup(4096 - token).unwrap();
        assert_eq!(requests[0].len(), 296 + 513);
        let a = left.execute(requests[0].as_bytes()).unwrap();
        let b = right.execute(requests[1].as_bytes()).unwrap();
        window.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
    }
    left.finish().unwrap();
    right.finish().unwrap();
    assert_eq!(window.finish().unwrap().steps(), 257);
    let too_long = vec![Step::Lookup; Capacity::Decoder.max_window_steps() + 1];
    assert!(matches!(
        client.begin(CONTEXT, &too_long),
        Err(Error::Limit)
    ));
    let (window, mut seeded) = client.begin_seeded(CONTEXT, &[Step::Lookup]).unwrap();
    let pending = window.lookup(4096).unwrap();
    let a = table.evaluate(&pending.share(Worker::Zero)).unwrap();
    let b = seeded
        .evaluate_lookup(&table, &pending.share(Worker::One))
        .unwrap();
    let accepted = pending
        .freeze_first(Ok(a))
        .unwrap()
        .freeze_second(Ok(b))
        .unwrap()
        .finish()
        .unwrap();
    assert_eq!(accepted.sharing_profile(), SharingProfile::Aes128WindowV1);
    let mut too_many = stages;
    too_many.push(Stage {
        id: [255; 32],
        matrix: &matrix,
    });
    assert!(matches!(
        Publication::publish_with_capacity(CONTEXT, &too_many, &table, Capacity::Decoder),
        Err(Error::Limit)
    ));
}

#[test]
fn wide_owner_shape_failures_retire_transactions_and_retained_packets_keep_leases() {
    let _serial = SERIAL.lock().unwrap();
    let matrix = matrix(257, 1);
    let table = table(1, 1);
    let stages = [Stage {
        id: OUT,
        matrix: &matrix,
    }];
    let public = Arc::new(
        Publication::publish_with_capacity(CONTEXT, &stages, &table, Capacity::Decoder).unwrap(),
    );
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let pending = client
        .begin(CONTEXT, &[Step::Linear(OUT)])
        .unwrap()
        .linear(&[1])
        .unwrap();
    let share = pending.share(Worker::Zero);
    assert!(matches!(
        LinearResponse::from_untrusted(&share, share.binding(), &vec![0; 258]),
        Err(Error::Limit)
    ));
    assert!(matches!(
        matrix.evaluate(&pending.share(Worker::One)),
        Err(Error::Retired)
    ));
    drop(pending);
    assert_eq!(public.live_windows(), 0);

    let mut zero = worker(&public, Worker::Zero, &stages, &table);
    let mut one = worker(&public, Worker::One, &stages, &table);
    let mut window = client.begin_wire(CONTEXT, &[Step::Linear(OUT)]).unwrap();
    let start = window.start_frames().unwrap();
    let mut left = zero.begin(start[0].as_bytes()).unwrap();
    let mut right = one.begin(start[1].as_bytes()).unwrap();
    let request = window.linear(&[1]).unwrap();
    let a = left.execute(request[0].as_bytes()).unwrap();
    let b = right.execute(request[1].as_bytes()).unwrap();
    let mut wrong = a.as_bytes().to_vec();
    wrong[284..288].copy_from_slice(&258u32.to_le_bytes());
    assert!(matches!(
        window.accept_replies(&wrong, b.as_bytes()),
        Err(Error::Binding)
    ));
    assert!(matches!(
        window.accept_replies(a.as_bytes(), b.as_bytes()),
        Err(Error::Retired)
    ));
    drop((window, left, right, zero, one, client));
    assert_eq!(public.live_windows(), 3);
    assert!(WireResources::current_with_capacity(Capacity::Decoder).process_live_bytes > 0);
    drop((start, request));
    assert_eq!(public.live_windows(), 2);
    drop((a, b));
    assert_eq!(public.live_windows(), 0);
    assert_eq!(
        WireResources::current_with_capacity(Capacity::Decoder).process_live_bytes,
        0
    );
}
