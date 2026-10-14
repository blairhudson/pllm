//! Immutable-byte continuation extension, separate from the original schedule schema.
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// A checked, graph-free public geometry handle; never a reservation or KV owner.
#[pyclass(frozen, name = "DecoderContinuationAdmission")]
struct PyContinuationAdmission {
    inner: pllm_compiler::ContinuationAdmission,
}

#[pymethods]
impl PyContinuationAdmission {
    #[new]
    fn new(
        py: Python<'_>,
        plan: &Bound<'_, PyBytes>,
        composition: &Bound<'_, PyBytes>,
        contract: &Bound<'_, PyBytes>,
    ) -> PyResult<Self> {
        let (plan, composition, contract) =
            (plan.as_bytes(), composition.as_bytes(), contract.as_bytes());
        let inner = py
            .detach(|| pllm_compiler::validate_decoder_continuation(plan, composition, contract))
            .map_err(crate::invalid)?;
        Ok(Self { inner })
    }

    fn admit(&self, prefix: u64, query: u64, memory_bytes: u64) -> PyResult<u64> {
        self.inner
            .admit(prefix, query, memory_bytes)
            .map_err(crate::invalid)
    }
}

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

#[pyfunction]
fn decoder_mlp_tiling<'py>(
    py: Python<'py>,
    plan: &Bound<'_, PyBytes>,
    composition: &Bound<'_, PyBytes>,
    tile_rows: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let (plan, composition) = (plan.as_bytes(), composition.as_bytes());
    let bytes = py
        .detach(|| pllm_compiler::mlp_tiling_bytes(plan, composition, tile_rows))
        .map_err(crate::invalid)?;
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
fn decoder_attention_tiling<'py>(
    py: Python<'py>,
    plan: &Bound<'_, PyBytes>,
    composition: &Bound<'_, PyBytes>,
    query_rows: u64,
    kv_heads: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let (plan, composition) = (plan.as_bytes(), composition.as_bytes());
    let bytes = py
        .detach(|| pllm_compiler::attention_tiling_bytes(plan, composition, query_rows, kv_heads))
        .map_err(crate::invalid)?;
    Ok(PyBytes::new(py, &bytes))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyContinuationAdmission>()?;
    module.add_function(wrap_pyfunction!(decoder_attention_tiling, module)?)?;
    module.add_function(wrap_pyfunction!(decoder_mlp_tiling, module)?)?;
    module.add_function(wrap_pyfunction!(decoder_continuation, module)?)?;
    module.add_function(wrap_pyfunction!(admit_decoder_continuation, module)?)
}
