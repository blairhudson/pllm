//! Test-only complete frozen-GELU protected replay. Fixture words are public;
//! masks, input shares, FSS roots and triples use ordinary OS-random issuance.
use super::*;
use crate::fusefss::conformance::{
    export_protected_gelu, ExportResult, ProtectedBackendIdentity, ProtectedWorkAdmission,
};
use serde_json::json;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

// Separate initial test-work limits, not changes to native live allocation,
// descriptor, lane or model admission ceilings. Public local table-copy work is
// reported independently from cumulative private key material.
const PRIVATE_MATERIAL_LIMIT: u64 = 128 << 20;
const PUBLIC_COPY_LIMIT: u64 = 2 << 30;
const ELAPSED_LIMIT: Duration = Duration::from_secs(1200);

fn native<T>(result: Result<T, Error>) -> ExportResult<T> {
    result.map_err(|error| format!("bounded native execution: {error:?}").into())
}

fn hex_digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn evaluate(
    gate: &CompiledGelu<'_>,
    words: &[u64],
    context: Context,
) -> ExportResult<(Vec<u64>, u64)> {
    let mask = (1u64 << 50) - 1;
    let [mut a, mut b] = native(gate.issue(words.len(), context))?;
    let mut x = Zeroizing::new(Vec::with_capacity(words.len()));
    let mut y = Zeroizing::new(Vec::with_capacity(words.len()));
    for word in words {
        let share = Zeroizing::new(native(random())? & mask);
        x.push(*share);
        y.push(word.wrapping_sub(*share) & mask);
    }
    let aa = native(a.start(&x))?;
    let bb = native(b.start(&y))?;
    let ap = native(a.advance(bb.as_bytes()))?;
    let bp = native(b.advance(aa.as_bytes()))?;
    let result_a = native(a.finish(bp.as_bytes()))?;
    let result_b = native(b.finish(ap.as_bytes()))?;
    if !a.is_consumed() || !b.is_consumed() {
        return Err("protected owners not consumed".into());
    }
    let result = result_a
        .shares()
        .iter()
        .zip(result_b.shares())
        .map(|(&a, &b)| a.wrapping_add(b) & mask)
        .collect();
    let bytes = [aa.as_bytes(), bb.as_bytes(), ap.as_bytes(), bp.as_bytes()]
        .iter()
        .map(|frame| frame.len() as u64)
        .sum();
    // Every owned secret/frame/output retires when this function returns.
    Ok((result, bytes))
}

#[test]
#[ignore = "explicit bounded complete protected export; set PLLM_FUSEFSS_EXPORT_PROTECTED_GELU=1"]
fn export_public_boundary_frozen_gelu() -> ExportResult<()> {
    let started = Instant::now();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fusefss-native-protected-v1/runtime-receipt.json");
    for receipt in [path.clone(), path.with_file_name("protected-receipt.json")] {
        match std::fs::remove_file(receipt) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    let budget = native(Budget::new(MAX_ALLOCATION_BYTES))?;
    let gate = native(CompiledGelu::with_backend(
        &budget,
        LookupBackend::PublicBoundary,
    ))?;
    let source_table_bytes = native(gate.resources(1))?
        .issuance
        .public_source_table_bytes;
    let retained = budget.used_bytes();
    let mut material = [0u64; 4];
    let mut public_copies = [0u64; 4];
    let mut reservations = [0usize; 4];
    let mut frames = [0u64; 4];
    for lanes in 1..=4 {
        let r = native(gate.resources(lanes))?;
        material[lanes - 1] = 2 * r.issuance.party_key_logical_bytes as u64;
        public_copies[lanes - 1] = 2 * r.issuance.public_table_bytes_per_party as u64;
        reservations[lanes - 1] = r.combined_reservation_bytes;
        frames[lanes - 1] = r.issuance.total_peer_frame_bytes as u64;
    }
    // This exact whole-fixture count is also independently enforced by the
    // helper before any protected callback. Price table-copy work first too.
    if 43544 * public_copies[0] > PUBLIC_COPY_LIMIT {
        return Err("complete public table-copy work exceeds test allowance".into());
    }
    let source_bytes: &[(&str, &[u8])] = &[
        (
            "crates/pllm-garble/src/fusefss_backend/mod.rs",
            include_bytes!("mod.rs"),
        ),
        (
            "crates/pllm-garble/src/fusefss_backend/compiled.rs",
            include_bytes!("compiled.rs"),
        ),
        (
            "crates/pllm-garble/src/fusefss_backend/dcf.rs",
            include_bytes!("dcf.rs"),
        ),
        (
            "crates/pllm-garble/src/fusefss_backend/conformance.rs",
            include_bytes!("conformance.rs"),
        ),
        (
            "crates/pllm-garble/src/compact_dcf.rs",
            include_bytes!("../compact_dcf.rs"),
        ),
        (
            "crates/pllm-garble/src/interval_fss.rs",
            include_bytes!("../interval_fss.rs"),
        ),
        (
            "crates/pllm-garble/src/shared_affine_lookup_reference.rs",
            include_bytes!("../shared_affine_lookup_reference.rs"),
        ),
        (
            "crates/pllm-garble/src/shared_affine_lookup_reference/ring.rs",
            include_bytes!("../shared_affine_lookup_reference/ring.rs"),
        ),
        (
            "crates/pllm-garble/src/shared_wide_product/ring.rs",
            include_bytes!("../shared_wide_product/ring.rs"),
        ),
    ];
    let hashes: Vec<_> = source_bytes
        .iter()
        .map(|(path, bytes)| (*path, hex_digest(bytes)))
        .collect();
    let sources: Vec<_> = hashes
        .iter()
        .map(|(path, hash)| (*path, hash.as_str()))
        .collect();
    let mut counts = [0u64; 4];
    let mut observed_frames = 0u64;
    let receipt = export_protected_gelu(
        ProtectedBackendIdentity {
            name: "public-original-boundary-shared-dcf.v1",
            sources: &sources,
        },
        ProtectedWorkAdmission {
            material_bytes_by_lanes: material,
            cumulative_material_limit: PRIVATE_MATERIAL_LIMIT,
        },
        |words, descriptor, context| {
            if started.elapsed() > ELAPSED_LIMIT || gate.descriptor_digest() != descriptor.digest()
            {
                return Err("protected replay time or descriptor admission failed".into());
            }
            let (outputs, bytes) = evaluate(&gate, words, context)?;
            if bytes != frames[words.len() - 1] || budget.used_bytes() != retained {
                return Err("protected frame or owner accounting mismatch".into());
            }
            counts[words.len() - 1] += 1;
            observed_frames += bytes;
            Ok(outputs)
        },
    )?;
    let cumulative_material: u64 = counts.iter().zip(material).map(|(&n, b)| n * b).sum();
    let cumulative_public_copies: u64 = counts.iter().zip(public_copies).map(|(&n, b)| n * b).sum();
    let words: u64 = counts
        .iter()
        .enumerate()
        .map(|(i, n)| (i as u64 + 1) * n)
        .sum();
    if words != 43544
        || cumulative_material > PRIVATE_MATERIAL_LIMIT
        || cumulative_public_copies > PUBLIC_COPY_LIMIT
        || started.elapsed() > ELAPSED_LIMIT
    {
        return Err("protected replay complete-work reconciliation failed".into());
    }
    drop(gate);
    if budget.used_bytes() != 0 {
        return Err("protected replay retained owners".into());
    }
    let runtime = json!({
        "schema": "pllm.fusefss.protected-native-runtime-receipt.v1",
        "scope": "local two-party scalar reconstruction; not transport, whole-model or security qualification",
        "backend": "public-original-boundary-shared-dcf.v1", "gate_evaluations": words,
        "batch_counts_by_lanes": counts, "logical_private_material_bytes": cumulative_material,
        "public_local_table_copy_bytes": cumulative_public_copies,
        "actual_peer_frame_bytes": observed_frames, "frames_exclude": "keys, public tables, authentication, transport and client boundary",
        "admitted_combined_bytes_by_lanes": reservations, "live_native_limit_bytes": MAX_ALLOCATION_BYTES,
        "public_source_table_bytes": source_table_bytes, "final_budget_bytes": budget.used_bytes(),
        "test_work_limits": {"private_material_bytes": PRIVATE_MATERIAL_LIMIT,
            "public_local_copy_bytes": PUBLIC_COPY_LIMIT, "elapsed_seconds": ELAPSED_LIMIT.as_secs()},
        "elapsed_seconds_including_fixture_and_export": started.elapsed().as_secs_f64(),
        "adapter_output": receipt["output"], "backend_sources": receipt["backend"],
        "freshness": "ordinary OS randomness in CompiledGelu::issue and independent input sharing; public compiler-fixture mask never reaches protected evaluator",
        "unknown": ["full-wire bytes", "process peak RSS", "full-stack key delivery", "independent cryptographic review", "checkpoint quality"]
    });
    std::fs::write(path, serde_json::to_vec_pretty(&runtime)?)?;
    Ok(())
}
