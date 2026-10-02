//! Bounded byte boundary only. All measured protocol work executes without GIL.
use crate::invalid;
use pllm_garble::projected_polynomial::{self as native, Contract, Dimensions, Layout};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyModule},
};
use serde_json::json;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn projected_polynomial_probe<'py>(
    py: Python<'py>,
    layout: &str,
    ring_bits: u8,
    rows: usize,
    hidden: usize,
    channels: usize,
    outputs: usize,
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    input: &[u8],
    binding: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    let mode = match layout {
        "dense" => Layout::Dense,
        "derived" => Layout::Derived,
        "contracted" => Layout::Contracted,
        "seeded" => Layout::Seeded,
        _ => return Err(invalid("unsupported projected polynomial layout".into())),
    };
    let binding: [u8; 32] = binding
        .try_into()
        .map_err(|_| invalid("polynomial binding requires 32 bytes".into()))?;
    // Contract bounds precede decoding/allocation, including direct binding callers.
    let contract = Contract::new(
        Dimensions {
            rows,
            hidden,
            channels,
            outputs,
            ring_bits,
        },
        mode,
        binding,
        [gate, up, down],
    )
    .map_err(invalid)?;
    if input.len() != rows * hidden * 8 {
        return Err(invalid("polynomial input byte length differs".into()));
    }
    let values: Vec<u64> = input
        .chunks_exact(8)
        .map(|p| u64::from_le_bytes(p.try_into().expect("eight bytes")))
        .collect();
    let result = py
        .detach(move || native::probe(contract, &values))
        .map_err(invalid)?;
    let bytes: Vec<u8> = result.output.iter().flat_map(|v| v.to_le_bytes()).collect();
    let metadata = json!({
        "material_bytes": result.material_bytes, "opening_bytes": result.opening_bytes,
        "expanded_array_storage_bytes_per_party": result.expanded_storage_bytes,
        "public_weight_array_storage_bytes_shared": result.shared_weight_bytes,
        "issuance_wall_seconds": result.issuance_seconds,
        "evaluation_wall_seconds": result.evaluation_seconds,
        "backend": "rust-pllm-garble/pllm-core", "mask_expansion": "aes256-counter-domain-sha256-v1",
    }).to_string();
    Ok((PyBytes::new(py, &bytes), metadata))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(projected_polynomial_probe, module)?)
}
