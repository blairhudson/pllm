use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn _retained_decoder_contract<'py>(
    py: Python<'py>,
    plan: &Bound<'_, PyBytes>,
    composition: &Bound<'_, PyBytes>,
    policy: &Bound<'_, PyBytes>,
    memory_bytes: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let (plan, composition, policy) = (plan.as_bytes(), composition.as_bytes(), policy.as_bytes());
    let bytes = py
        .detach(|| pllm_compiler::retained_decoder_bytes(plan, composition, policy, memory_bytes))
        .map_err(crate::invalid)?;
    Ok(PyBytes::new(py, &bytes))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_retained_decoder_contract, module)?)
}
