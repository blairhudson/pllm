//! Local Maverick algebra benchmark; no certified privacy/verifier or live handles.
use crate::invalid;
use pllm_core::{
    coded_delegation_reference::{RaaDelegationReference, RaaDelegationResources},
    coded_linear::{CODED_LINEAR_FIELD, RAA_MAX_PREPROCESS_BYTES},
    Executor, Matrix,
};
use pyo3::{prelude::*, types::PyBytes};
use serde_json::{json, Value};
use std::{sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

const MAX_PRODUCTS: usize = 64 << 20;

fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let mut bytes = Vec::new();
        for source in [
            include_bytes!("coded_delegation.rs").as_slice(),
            include_bytes!("../../pllm-core/src/coded_delegation_reference.rs").as_slice(),
            include_bytes!("../../pllm-core/src/coded_linear.rs").as_slice(),
            include_bytes!("../../pllm-core/src/kernels.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ] {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.coded_delegation.benchmark.v1", &bytes).to_string()
    })
}

fn projection(rows: usize, columns: usize, queries: usize) -> Result<Value, String> {
    if !(1..=256).contains(&queries) {
        return Err("Maverick benchmark requires 1..=256 serial queries".into());
    }
    let r = RaaDelegationResources::for_shape(rows, columns)?;
    // Dimensions are bounded before arithmetic. Include caller/native weight
    // copies, signed inputs, decoded/control/output copies and conservative
    // encode/syndrome/sampling scratch. Python's oracle is a separate owner.
    let allocation = r.persistent_payload_bytes
        + r.max_live_query_payload_bytes
        + 2 * rows * columns
        + 2 * queries * columns
        + 12 * queries * rows
        + 256 * (rows + columns);
    Ok(json!({
        "rows": rows, "columns": columns, "queries": queries,
        "field": CODED_LINEAR_FIELD,
        "privacy_matrix_bytes": r.privacy_matrix_bytes,
        "verification_matrix_bytes": r.verification_matrix_bytes,
        "code_payload_bytes": r.code_payload_bytes,
        "persistent_client_payload_bytes": r.persistent_payload_bytes,
        "evaluator_weight_bytes": rows * columns,
        "privacy_sparsity": r.privacy_sparsity,
        "serial_query_payload_bytes": 4 * (rows + columns),
        "total_query_payload_bytes": queries * 4 * (rows + columns),
        "field_boundary_payload_estimate_bytes": queries * 4 * (rows + columns),
        "protocol_allocation_estimate_bytes": allocation,
        "admitted": r.within_reference_bound() && allocation <= RAA_MAX_PREPROCESS_BYTES
            && queries * rows * columns <= MAX_PRODUCTS,
        "distance_certified": false, "privacy_certified": false,
        "native_implementation_digest": implementation_digest(),
        "native_version": env!("CARGO_PKG_VERSION"),
    }))
}

#[pyfunction]
fn coded_delegation_projection(rows: usize, columns: usize, queries: usize) -> PyResult<String> {
    projection(rows, columns, queries)
        .map(|r| r.to_string())
        .map_err(invalid)
}

#[pyfunction]
fn coded_delegation_benchmark<'py>(
    py: Python<'py>,
    rows: usize,
    columns: usize,
    queries: usize,
    public_seed: &[u8],
    weights: &[u8],
    inputs: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    let mut metadata = projection(rows, columns, queries).map_err(invalid)?;
    if metadata["admitted"] != true {
        return Err(invalid(
            "Maverick benchmark exceeds 256 MiB allocation or 64M product bound".into(),
        ));
    }
    let seed: [u8; 32] = public_seed
        .try_into()
        .map_err(|_| invalid("public code seed must contain 32 bytes".into()))?;
    if weights.len() != rows * columns || inputs.len() != queries * columns {
        return Err(invalid(
            "Maverick signed-i8 input/weight shape mismatch".into(),
        ));
    }
    let (output, metadata) = py.detach(|| -> PyResult<_> {
        let start = Instant::now();
        let matrix = Matrix::new(weights, rows, columns).map_err(invalid)?;
        let executor = Executor::new(1, true).map_err(invalid)?;
        let reference =
            RaaDelegationReference::preprocess(matrix.as_signed_slice(), rows, columns, seed)
                .map_err(invalid)?;
        metadata["public_setup_seconds"] = json!(start.elapsed().as_secs_f64());
        let mut output = Zeroizing::new(Vec::with_capacity(queries * rows * 4));
        let (mut issuance, mut evaluation, mut verification, mut clear_time) = (0.0, 0.0, 0.0, 0.0);
        for raw in inputs.chunks_exact(columns) {
            let input = Zeroizing::new(raw.iter().map(|v| *v as i8).collect::<Vec<_>>());
            let start = Instant::now();
            let clear = Zeroizing::new(matrix.clear(&executor, &input, 1).map_err(invalid)?);
            clear_time += start.elapsed().as_secs_f64();
            let start = Instant::now();
            let query = reference.issue(&input).map_err(invalid)?;
            issuance += start.elapsed().as_secs_f64();
            let start = Instant::now();
            let claimed = Zeroizing::new(
                matrix
                    .modular(&executor, query.masked_input(), 1, CODED_LINEAR_FIELD)
                    .map_err(invalid)?,
            );
            evaluation += start.elapsed().as_secs_f64();
            let start = Instant::now();
            let decoded = Zeroizing::new(reference.finish(query, &claimed).map_err(invalid)?);
            verification += start.elapsed().as_secs_f64();
            if decoded != clear {
                return Err(invalid(
                    "Maverick result differs from clear integer kernel".into(),
                ));
            }
            output.extend(decoded.iter().flat_map(|v| v.to_le_bytes()));
            // query has been consumed before issuing the next vector.
        }
        metadata["issuance_seconds"] = json!(issuance);
        metadata["evaluation_seconds"] = json!(evaluation);
        metadata["verification_seconds"] = json!(verification);
        metadata["online_seconds"] = json!(evaluation + verification);
        metadata["clear_seconds"] = json!(clear_time);
        Ok((output, metadata))
    })?;
    py.check_signals()?;
    Ok((PyBytes::new(py, &output), metadata.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(coded_delegation_projection, module)?)?;
    module.add_function(wrap_pyfunction!(coded_delegation_benchmark, module)?)
}
