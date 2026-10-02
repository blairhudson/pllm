//! Immutable-byte boundary for offline native placement validation.
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// network_placement_requirements(model_plan: bytes, composition: bytes) -> bytes
///
/// Derive requirements before selecting parties, with the same native parsing,
/// composition admission, complete schedule and owned-tensor floors as placement
/// validation. No assignment, offers, resource estimates or capacity are needed.
/// Canonical model_plan JSON is bounded to 16 MiB; canonical resolved composition
/// JSON is bounded to 1 MiB. Invalid inputs raise ValueError; non-bytes arguments
/// raise TypeError.
///
/// Returns canonical JSON with schema pllm.network_placement_requirements.v1,
/// roles [{role_id, capability}], separate_operators [[role_id, role_id]],
/// minimum_weight_bytes and minimum_memory_bytes exact-role u64 maps, and
/// schedule_digest, model_plan_digest, composition_digest. Roles and separation
/// pairs are identity-sorted. Floors count one byte per owned tensor element,
/// deduplicated by explicit native artifact ID. Memory floor currently equals
/// weight floor; excludes float source storage, native copies, scales, buffers
/// and peak RAM. Separation describes required operator declarations, not proof
/// of independence. This document does not authorize execution or reserve capacity.
#[pyfunction]
fn network_placement_requirements<'py>(
    py: Python<'py>,
    model_plan: &Bound<'_, PyBytes>,
    composition: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    let model_plan = model_plan.as_bytes();
    let composition = composition.as_bytes();
    let output = py
        .detach(|| pllm_compiler::network_placement_requirements(model_plan, composition))
        .map_err(crate::invalid)?;
    Ok(PyBytes::new(py, &output))
}

/// validate_network_placement(model_plan: bytes, composition: bytes, document: bytes) -> bytes
///
/// Offline legality only. model_plan is canonical native decoder-plan JSON
/// (maximum 16 MiB); composition is canonical resolved pipeline JSON (1 MiB).
/// document is strict pllm.network_placement.v1 JSON (1 MiB, 16 roles, 64 parties).
/// Unknown/duplicate fields, invalid unsigned integers, expired offers, unknown
/// capabilities, unsupported compositions and incomplete native schedules fail
/// with ValueError. Non-bytes arguments fail with TypeError.
///
/// snapshot_digest and instance_epoch are nonzero lowercase 64-hex hashes.
/// allowed_models contains exact native plan.config_digest hashes or "*";
/// checkpoint/model names are not admitted. Role requirements aggregate per
/// party with checked arithmetic; each assigned role consumes one session slot.
/// Both resource maps must meet native per-role owned-tensor floors: one byte
/// per tensor element, deduplicated by artifact ID across both decoder phases.
/// Prepared providers retain all body weights even for client-owned stages;
/// offset workers retain all body weights; client-only owns every tensor.
/// These floors exclude float source storage, native copies, scales, working
/// buffers and peak RAM. They are not complete memory certificates.
///
/// Returns compact sorted JSON: request fields plus schedule_digest,
/// model_plan_digest, composition_digest, minimum_weight_bytes,
/// minimum_memory_bytes and placement_digest. Both minimum fields are exact-role
/// u64 maps; minimum_memory_bytes currently equals minimum_weight_bytes. Role/party arrays
/// and capability/model lists are sorted by identity. placement_digest is SHA256
/// of b"pllm.network_placement.validated.v1\0" plus canonical output without
/// placement_digest. To revalidate output, pass only its original request fields.
///
/// Operator separation checks declarations only. Offers, snapshot contents,
/// physical independence and resource estimates are not authenticated here;
/// this result neither reserves capacity nor authorizes live network execution.
#[pyfunction]
fn validate_network_placement<'py>(
    py: Python<'py>,
    model_plan: &Bound<'_, PyBytes>,
    composition: &Bound<'_, PyBytes>,
    document: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    // Borrow immutable Python bytes; bounds are checked natively before parsing.
    let model_plan = model_plan.as_bytes();
    let composition = composition.as_bytes();
    let document = document.as_bytes();
    let output = py
        .detach(|| pllm_compiler::validate_network_placement(model_plan, composition, document))
        .map_err(crate::invalid)?;
    Ok(PyBytes::new(py, &output))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(network_placement_requirements, module)?)?;
    module.add_function(wrap_pyfunction!(validate_network_placement, module)?)
}
