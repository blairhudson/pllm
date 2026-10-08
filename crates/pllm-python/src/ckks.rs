//! Public CKKS layout preparation; ciphertext ownership stays in the HE backend.
use crate::invalid;
use pllm_core::ckks_layout;
use pyo3::prelude::*;
use serde_json::json;
use std::sync::OnceLock;

fn digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let mut bytes = Vec::new();
        for source in [
            include_bytes!("ckks.rs").as_slice(),
            include_bytes!("../../pllm-core/src/ckks_layout.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ] {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.ckks_layout.implementation.v1", &bytes).to_string()
    })
}

#[pyfunction]
fn ckks_parameters(
    depth: usize,
    rotation_keys: usize,
    relinearization: bool,
    live_ciphertexts: usize,
) -> PyResult<String> {
    let parameters =
        ckks_layout::parameters(depth, rotation_keys, relinearization, live_ciphertexts)
            .map_err(invalid)?;
    Ok(
        json!({"parameters": parameters, "native_implementation_digest": digest(),
        "native_version": env!("CARGO_PKG_VERSION")})
        .to_string(),
    )
}

#[pyfunction]
fn ckks_softmax_parameters(width: usize) -> PyResult<String> {
    let parameters = ckks_layout::softmax_parameters(width).map_err(invalid)?;
    Ok(
        json!({"parameters": parameters, "native_implementation_digest": digest(),
        "native_version": env!("CARGO_PKG_VERSION")})
        .to_string(),
    )
}

#[pyfunction]
fn ckks_diagonals(
    py: Python<'_>,
    weights: &[u8],
    rows: usize,
    columns: usize,
) -> PyResult<Vec<Vec<f64>>> {
    ckks_layout::padded_width(rows, columns).map_err(invalid)?;
    if weights.len() != rows * columns * 8 {
        return Err(invalid(
            "CKKS weights require exact little-endian f64 shape".into(),
        ));
    }
    let weights = weights
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    py.detach(|| ckks_layout::diagonals(&weights, rows, columns))
        .map_err(invalid)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ckks_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(ckks_softmax_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(ckks_diagonals, module)?)?;
    Ok(())
}
