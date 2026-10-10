use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn stage_kernel_plan<'py>(
    py: Python<'py>,
    plan: &Bound<'_, PyBytes>,
    composition: &Bound<'_, PyBytes>,
    evidence: &Bound<'_, PyBytes>,
    budget: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let (plan, composition, evidence) =
        (plan.as_bytes(), composition.as_bytes(), evidence.as_bytes());
    let bytes = py
        .detach(|| pllm_compiler::stage_kernel_plan(plan, composition, evidence, budget))
        .map_err(crate::invalid)?;
    Ok(PyBytes::new(py, &bytes))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(stage_kernel_plan, module)?)
}
