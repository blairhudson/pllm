use crate::invalid;
use pllm_core::progressive_head::ProgressiveHead;
use pyo3::{prelude::*, types::PyBytes};

#[pyclass(frozen, name = "_ProgressiveHead", module = "pllm._native")]
struct Head(ProgressiveHead);
type QueryReport = (usize, Vec<(u8, usize, usize)>);
#[pymethods]
impl Head {
    #[new]
    fn new(py: Python<'_>, weights: &[u8], scales: Vec<f32>, columns: usize) -> PyResult<Self> {
        Ok(Self(
            py.detach(|| ProgressiveHead::new(weights, &scales, columns))
                .map_err(invalid)?,
        ))
    }
    fn query(&self, py: Python<'_>, input: &[u8], scale: f32, bits: u8) -> PyResult<QueryReport> {
        let x: Vec<i8> = input.iter().map(|&x| x as i8).collect();
        let report = py
            .detach(|| self.0.query(&x, scale, bits))
            .map_err(invalid)?;
        Ok((report.winner, report.rounds))
    }
    fn residual_table<'py>(&self, py: Python<'py>, bits: u8) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.0.residual_table(bits)).map_err(invalid)?;
        Ok(PyBytes::new(py, &bytes))
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Head>()
}
