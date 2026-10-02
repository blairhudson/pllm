//! Immutable-byte continuation extension, separate from the original schedule schema.
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn decoder_continuation<'py>(
    py: Python<'py>,
    plan: &Bound<'_, PyBytes>,
    composition: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (plan, composition) = (plan.as_bytes(), composition.as_bytes());
    let bytes = py
        .detach(|| pllm_compiler::decoder_continuation_bytes(plan, composition))
        .map_err(crate::invalid)?;
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
fn admit_decoder_continuation(
    py: Python<'_>,
    plan: &Bound<'_, PyBytes>,
    composition: &Bound<'_, PyBytes>,
    contract: &Bound<'_, PyBytes>,
    prefix: u64,
    query: u64,
    memory_bytes: u64,
) -> PyResult<u64> {
    let (plan, composition, contract) =
        (plan.as_bytes(), composition.as_bytes(), contract.as_bytes());
    py.detach(|| {
        pllm_compiler::admit_decoder_continuation(
            plan,
            composition,
            contract,
            prefix,
            query,
            memory_bytes,
        )
    })
    .map_err(crate::invalid)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decoder_continuation, module)?)?;
    module.add_function(wrap_pyfunction!(admit_decoder_continuation, module)?)
}
