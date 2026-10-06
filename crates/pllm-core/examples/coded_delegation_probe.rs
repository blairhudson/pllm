//! Isolated standard-mode costs with public synthetic weights, never a decoder.
use std::{env, hint::black_box, time::Instant};

use pllm_core::{
    coded_delegation_reference::{RaaDelegationReference, RaaDelegationResources},
    coded_linear::{CODED_LINEAR_FIELD, RAA_MAX_PREPROCESS_BYTES},
    Executor, Matrix,
};
use serde_json::json;
use sha2::{Digest, Sha256};

fn main() -> Result<(), String> {
    let args = env::args()
        .skip(1)
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| "expected rows columns samples".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if args.len() != 3 || !(1..=32).contains(&args[2]) {
        return Err("expected rows columns samples; samples must be 1..=32".into());
    }
    let (rows, columns, samples) = (args[0], args[1], args[2]);
    let resources = RaaDelegationResources::for_shape(rows, columns)?;
    let mut report = json!({
        "rows": rows, "columns": columns, "samples": samples,
        "privacy_matrix_bytes": resources.privacy_matrix_bytes,
        "verification_matrix_bytes": resources.verification_matrix_bytes,
        "code_payload_bytes": resources.code_payload_bytes,
        "persistent_client_payload_bytes": resources.persistent_payload_bytes,
        "max_live_query_payload_bytes": resources.max_live_query_payload_bytes,
        "reference_payload_bound_bytes": RAA_MAX_PREPROCESS_BYTES,
        "privacy_sparsity": resources.privacy_sparsity,
        "executable_sdk": false,
    });
    if !resources.within_reference_bound() {
        // The resource error must occur even without imported weight values.
        let error = RaaDelegationReference::preprocess(&[], rows, columns, [13; 32])
            .err()
            .ok_or("over-budget geometry unexpectedly admitted")?;
        report["status"] = json!("rejected_before_preprocessing");
        report["reason"] = json!(error);
        println!("{report}");
        return Ok(());
    }
    let raw_weights = (0..rows * columns)
        .map(|i| (i * 31 + i / columns * 7) as u8)
        .collect::<Vec<_>>();
    let weight_sha256 = format!("{:x}", Sha256::digest(&raw_weights));
    let matrix = Matrix::new(&raw_weights, rows, columns)?;
    drop(raw_weights);
    let executor = Executor::new(1, true)?;
    let start = Instant::now();
    let reference =
        RaaDelegationReference::preprocess(matrix.as_signed_slice(), rows, columns, [13; 32])?;
    let preprocessing_seconds = start.elapsed().as_secs_f64();
    let mut observations = Vec::new();
    let mut previous_masked = None;
    for sample in 0..samples {
        // Vary the vector while preserving signed i8 extremes and row geometry.
        let input = (0..columns)
            .map(|i| (i * 17 + sample * 19) as i8)
            .collect::<Vec<_>>();
        let start = Instant::now();
        let clear = black_box(matrix.clear(&executor, black_box(&input), 1)?);
        let clear_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let query = reference.issue(&input)?;
        let issue_seconds = start.elapsed().as_secs_f64();
        if previous_masked.as_deref() == Some(query.masked_input()) {
            return Err("repeated masked vector".into());
        }
        previous_masked = Some(query.masked_input().to_vec());
        let start = Instant::now();
        let claimed = matrix.modular(&executor, query.masked_input(), 1, CODED_LINEAR_FIELD)?;
        let server_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let decoded = reference.finish(query, &claimed)?;
        let finish_seconds = start.elapsed().as_secs_f64();
        if decoded != clear {
            return Err("decoded signed output differs from clear SIMD kernel".into());
        }
        observations.push(json!({
            "clear_seconds": clear_seconds, "issue_seconds": issue_seconds,
            "server_seconds": server_seconds, "finish_seconds": finish_seconds,
            "client_seconds": issue_seconds + finish_seconds,
            "aggregate_seconds": issue_seconds + server_seconds + finish_seconds,
        }));
    }
    let mut forgeries_rejected = 0;
    for row in [0, rows / 2, rows - 1] {
        let query = reference.issue(&vec![127; columns])?;
        let mut claimed = matrix.modular(&executor, query.masked_input(), 1, CODED_LINEAR_FIELD)?;
        claimed[row] = (claimed[row] + 1) % CODED_LINEAR_FIELD;
        if reference.finish(query, &claimed).is_ok() {
            return Err("changed output was accepted".into());
        }
        forgeries_rejected += 1;
    }
    report["status"] = json!("reference_checks_passed");
    report["synthetic_weight_sha256"] = json!(weight_sha256);
    report["preprocessing_seconds"] = json!(preprocessing_seconds);
    report["raw_input_output_field_bytes_per_query"] = json!(4 * (rows + columns));
    report["checked_outputs"] = json!(samples * rows);
    report["forgeries_rejected"] = json!(forgeries_rejected);
    report["observations"] = json!(observations);
    println!("{report}");
    Ok(())
}
