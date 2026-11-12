//! Aggregate-only fixed toy facade. Intentionally unmounted pending shared integration.
use crate::invalid;
use pllm_core::torus_lwe as toy;
use pyo3::prelude::*;
use serde_json::{json, Value};

const JSON_LIMIT: usize = 8192;
const SOURCES: [&[u8]; 5] = [
    include_bytes!("torus_lwe.rs"),
    include_bytes!("../../pllm-core/src/torus_lwe.rs"),
    include_bytes!("../../pllm-core/src/kernel_support.rs"),
    include_bytes!("../../pllm-types/src/lib.rs"),
    include_bytes!("../../../Cargo.lock"),
];

fn admitted() -> PyResult<(toy::Quote, usize)> {
    let q = toy::preflight(toy::MAX_LOGICAL_BYTES, u64::MAX).map_err(|e| invalid(e.to_string()))?;
    let source_bytes = SOURCES.iter().try_fold(0usize, |n, s| n.checked_add(s.len()).and_then(|n| n.checked_add(8)))
        .ok_or_else(|| invalid("toy fingerprint size overflow".into()))?;
    // Embedded source pages + concatenation workspace; NOT a fictitious fixed
    // allowance for reading an arbitrarily large source/binary. No binary read.
    let total = source_bytes.checked_mul(2).and_then(|n| n.checked_add(q.logical_bytes))
        .ok_or_else(|| invalid("toy wrapper size overflow".into()))?;
    if total > toy::MAX_LOGICAL_BYTES { return Err(invalid("toy logical wrapper estimate exceeds ceiling".into())); }
    Ok((q, total))
}

fn implementation_digest(source_bytes: usize) -> String {
    let mut bytes = Vec::with_capacity(source_bytes);
    for source in SOURCES {
        bytes.extend((source.len() as u64).to_le_bytes());
        bytes.extend(source);
    }
    pllm_types::digest_bytes("pllm.torus_lwe.implementation.v1", &bytes).to_string()
}

fn metadata() -> PyResult<Value> {
    let (q, total) = admitted()?;
    Ok(json!({
        "contract": toy::CONTRACT, "lanes": 4,
        "requested_os_bytes": q.requested_os_bytes,
        "logical_key_table_bytes": toy::TABLE_BYTES,
        "logical_input_output_payload_bytes": toy::PAYLOAD_BYTES,
        "logical_allocation_estimate_bytes": total,
        "work_units": q.work_units, "error_bound": toy::ERROR_BOUND,
        "native_implementation_digest": implementation_digest((total - q.logical_bytes) / 2),
        "native_version": env!("CARGO_PKG_VERSION")
    }))
}

fn bounded_json(value: Value) -> PyResult<String> {
    let text = value.to_string();
    if text.len() > JSON_LIMIT { return Err(invalid("toy aggregate JSON exceeds fixed bound".into())); }
    Ok(text)
}

/// Deterministic and entropy-free; no caller parameters, masks or secret exports.
#[pyfunction]
fn torus_lwe_preflight() -> PyResult<String> { bounded_json(metadata()?) }

#[pyfunction]
fn torus_lwe_fixture(py: Python<'_>) -> PyResult<String> {
    let mut report = metadata()?; // repeat admission before material/RNG
    let result = py.detach(toy::fixture).map_err(|e| invalid(e.to_string()))?;
    report["decoded_matches"] = json!(result.decoded_matches);
    report["phase_equalities"] = json!(result.phase_equalities);
    report["maximum_centered_error"] = json!(result.maximum_centered_error);
    report["issuance_ns"] = json!(result.issuance_ns);
    report["switch_ns"] = json!(result.switch_ns);
    report["decrypt_ns"] = json!(result.decrypt_ns);
    report["oracle_ns"] = json!(result.oracle_ns);
    py.check_signals()?;
    bounded_json(report)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(torus_lwe_preflight, module)?)?;
    module.add_function(wrap_pyfunction!(torus_lwe_fixture, module)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_preflight_metadata_and_real_source_workspace() {
        let first = torus_lwe_preflight().unwrap();
        assert_eq!(first, torus_lwe_preflight().unwrap());
        let report: Value = serde_json::from_str(&first).unwrap();
        assert_eq!(report["requested_os_bytes"], 31952);
        assert_eq!(report["logical_key_table_bytes"], 38400);
        assert_eq!(report["logical_input_output_payload_bytes"], 448);
        assert!(report.get("decoded_matches").is_none());
        let (q, total) = admitted().unwrap();
        let actual: usize = SOURCES.iter().map(|source| source.len() + 8).sum();
        assert_eq!(total, q.logical_bytes + 2 * actual);
        assert!(first.len() <= JSON_LIMIT);
    }

    #[test]
    fn metadata_json_bound_refuses_oversize() {
        assert!(bounded_json(json!({"oversize": "x".repeat(JSON_LIMIT)})).is_err());
    }
}
