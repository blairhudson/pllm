use crate::invalid;
use pllm_core::head_index::Index;
use pyo3::{prelude::*, types::PyBytes};

#[pyclass(name = "HeadRetrievalIndex", module = "pllm._native")]
struct HeadIndex {
    inner: Index,
}
#[pymethods]
impl HeadIndex {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        rows: usize,
        features: usize,
        rank: usize,
        weights: &[u8],
        projection: &[u8],
        profiles: &[u8],
    ) -> PyResult<Self> {
        Ok(Self {
            inner: py
                .detach(|| Index::new(rows, features, rank, weights, projection, profiles))
                .map_err(invalid)?,
        })
    }
    #[getter]
    fn payload_bytes(&self) -> usize {
        self.inner.payload_bytes()
    }
    fn candidates<'py>(
        &self,
        py: Python<'py>,
        input: &[u8],
        scale: f64,
        count: usize,
    ) -> PyResult<(Bound<'py, PyBytes>, f64)> {
        let (ids, upper) = py
            .detach(|| self.inner.candidates(input, scale, count))
            .map_err(invalid)?;
        let bytes: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
        Ok((PyBytes::new(py, &bytes), upper))
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<HeadIndex>()
}
