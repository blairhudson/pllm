//! Two bounded fresh processes, no seed/cursor serialization or resume path.
use pllm_core::gkm::*;
use std::{fmt::Write, process::Command, sync::Arc};

fn hex(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(result, "{byte:02x}").unwrap();
    }
    result
}

#[test]
fn fresh_process_fixture() {
    if std::env::var("PLLM_GKM_RESTART_FIXTURE").as_deref() != Ok("1") {
        return;
    }
    let matrix = I16Matrix::from_encoded(1, 2, &[3, -2], &[100; 2]).unwrap();
    const VOCABULARY: usize = 256;
    const TOKEN: usize = VOCABULARY - 1;
    let table_bytes: Vec<_> = (0..VOCABULARY)
        .flat_map(|row| {
            if row == TOKEN {
                (-0.0_f32).to_le_bytes()
            } else {
                (row as f32 / 16.0).to_le_bytes()
            }
        })
        .collect();
    let table = FloatTable::from_le_bytes(VOCABULARY, 1, &table_bytes).unwrap();
    let publication = Arc::new(
        Publication::publish(
            [1; 32],
            &[Stage {
                id: [2; 32],
                matrix: &matrix,
            }],
            &table,
        )
        .unwrap(),
    );
    let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
    let (window, mut worker) = client
        .begin_seeded([3; 32], &[Step::Lookup, Step::Linear([2; 32])])
        .unwrap();
    let request = hex(&window.request_id());
    let pending = window.lookup(TOKEN).unwrap();
    let view = pending.share(Worker::Zero);
    let mut lookup_stream = [0u8; VOCABULARY / 8];
    lookup_stream.copy_from_slice(view.selector().unwrap());
    lookup_stream[TOKEN / 8] ^= 1 << (TOKEN % 8); // remove the known one-hot input
                                                  // Two full AES blocks, independently compared across processes. No request,
                                                  // publication or linear-stream entropy may enter this lookup fingerprint.
    let lookup_mask = hex(&lookup_stream);
    assert!(pending.share(Worker::One).selector().is_none());
    let first = table.evaluate(&view);
    let half = pending.freeze_first(first).unwrap();
    let second = worker.evaluate_lookup(&table, &half.remaining_share());
    let window = half.freeze_second(second).unwrap();
    assert_eq!(window.tentative_values()[0].to_bits(), (-0.0_f32).to_bits());
    let pending = window.linear(&[7, -11]).unwrap();
    let view = pending.share(Worker::Zero);
    let values = view.values().unwrap();
    let linear_mask = format!("{:016x}{:016x}", values[0], values[1]);
    assert!(pending.share(Worker::One).values().is_none());
    let first = matrix.evaluate(&view);
    let half = pending.freeze_first(first).unwrap();
    let second = worker.evaluate_linear(&matrix, &half.remaining_share());
    let accepted = half.freeze_second(second).unwrap().finish().unwrap();
    assert_eq!(accepted.signed_linear_output(), [43]);
    drop(worker);
    assert_eq!(publication.live_windows(), 1);
    drop(accepted);
    assert_eq!(publication.live_windows(), 0);
    println!("GKM_RESTART={request}:{linear_mask}:{lookup_mask}");
}

#[test]
fn fresh_processes_draw_new_keys_and_request_identities() {
    let mut prior = None;
    for _ in 0..2 {
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "fresh_process_fixture",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("PLLM_GKM_RESTART_FIXTURE", "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let stdout = String::from_utf8(result.stdout).unwrap();
        let frame = stdout
            .lines()
            .find_map(|line| line.split_once("GKM_RESTART=").map(|(_, frame)| frame))
            .unwrap();
        let fields: Vec<_> = frame.split(':').collect();
        assert_eq!(fields.len(), 3);
        let (request, linear_mask, lookup_mask) = (fields[0], fields[1], fields[2]);
        assert_eq!(
            (request.len(), linear_mask.len(), lookup_mask.len()),
            (64, 32, 64)
        );
        if let Some((old_request, old_linear, old_lookup)) = &prior {
            assert_ne!(request, old_request);
            assert_ne!(
                linear_mask, old_linear,
                "linear stream repeated across fresh processes"
            );
            assert_ne!(
                lookup_mask, old_lookup,
                "lookup stream repeated across fresh processes"
            );
        }
        prior = Some((
            request.to_owned(),
            linear_mask.to_owned(),
            lookup_mask.to_owned(),
        ));
    }
}
