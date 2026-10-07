//! Fresh-process synthetic fixture, with no model values or private user inputs.
use pllm_core::retained_kv::RetainedKvCache;
use pllm_garble::retained_selection::{estimate, Context, Policy, SelectionOwner};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Instant;

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, byte| {
        write!(out, "{byte:02x}").unwrap();
        out
    })
}

fn main() -> Result<(), String> {
    let args = std::env::args()
        .skip(1)
        .map(|s| s.parse::<usize>().map_err(|_| "invalid fixture integer"))
        .collect::<Result<Vec<_>, _>>()?;
    let [rows, width, keep, recent, variant] = args.as_slice() else {
        return Err("expected rows width keep recent variant".into());
    };
    let (rows, width, variant) = (*rows, *width, *variant);
    let policy = Policy {
        keep: *keep,
        recent: *recent,
    };
    // Refuse unsupported dimensions before constructing even the public fixture.
    estimate(rows, width, policy, 0)?;
    if variant > 2 {
        return Err("fixture variant exceeds bound".into());
    }
    let started = Instant::now();
    let keys: Vec<_> = (0..2 * rows * width)
        .map(|i| ((i * 17 % 37) as i32 - 18) as f32 / 8.0)
        .collect();
    let values: Vec<_> = (0..keys.len())
        .map(|i| ((i * 11 % 53) as i32 - 26) as f32 / 16.0)
        .collect();
    let positions: Vec<_> = (0..rows).map(|i| 2 * i + 1).collect();
    let cache =
        RetainedKvCache::from_prefill(&keys, &values, width, &positions, rows + 2, 2 * rows + 2)?;
    drop(keys);
    drop(values);
    drop(positions);
    let mut owner = SelectionOwner::new(
        Context {
            plan: [1; 32],
            source: [2; 32],
            state_contract: [3; 32],
            layer: 0,
        },
        cache,
    )?;
    let state_before = owner.allocated_bytes();
    let fixture_seconds = started.elapsed().as_secs_f64();
    let scores: Vec<_> = (0..rows)
        .map(|i| {
            if variant == 0 {
                1.0
            } else {
                ((i * 7 + variant * 3) % rows) as f32 / 16.0
            }
        })
        .collect();
    let started = Instant::now();
    let (client, evaluator) = owner.issue(&scores, policy)?;
    let issuance_seconds = started.elapsed().as_secs_f64();
    let binding = evaluator.binding();
    let r = evaluator.resources();
    let started = Instant::now();
    let output = evaluator.evaluate(binding)?;
    let evaluation_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    client.finish(&mut owner, output)?;
    let commit_seconds = started.elapsed().as_secs_f64();
    let (key, value, positions) = owner.active()?;
    let mut hash = Sha256::new();
    for p in positions {
        hash.update(p.to_le_bytes());
    }
    for v in key.iter().chain(value) {
        hash.update(v.to_bits().to_le_bytes());
    }
    let snapshot_digest = hex(&hash.finalize());
    let result = json!({
        "schema": "pllm.retained_selection.native_fixture.v1",
        "rows": rows, "width": width, "keep": policy.keep, "recent": policy.recent, "variant": variant,
        "output_sha256": snapshot_digest,
        "public_schedule_sha256": hex(&r.public_schedule_digest),
        "allocation": {"state_before_bytes": state_before, "state_after_bytes": owner.allocated_bytes(),
            "combined_preflight_bound_bytes": r.estimate.combined_allocation_bound_bytes,
            "wire_bound": r.estimate.wire_bound, "actual_wires": r.wires},
        "material": {"and_gates": r.and_gates, "evaluator_ciphertext_bytes": r.evaluator_ciphertext_bytes,
            "constant_label_bytes": r.constant_label_bytes, "input_label_bytes": r.input_label_bytes,
            "output_label_bytes": r.output_label_bytes, "client_encoding_bytes": r.client_encoding_bytes,
            "public_metadata_bytes_estimate": r.public_metadata_bytes_estimate,
            "client_to_evaluator_material_bytes": r.evaluator_ciphertext_bytes
                + (r.constant_label_bytes + r.input_label_bytes) as u64,
            "evaluator_to_client_material_bytes": r.output_label_bytes},
        "timings_seconds": {"fixture": fixture_seconds, "issuance": issuance_seconds,
            "evaluation": evaluation_seconds, "client_commit": commit_seconds},
        "geometry_gates": {"qwen_width_128_rejected": estimate(16, 128, policy, 0).is_err(),
            "sixteen_by_sixteen_memory_rejected": estimate(16, 16, Policy { keep: 8, recent: 2 }, 0).is_err()}
    });
    owner.close();
    println!("{result}");
    Ok(())
}
