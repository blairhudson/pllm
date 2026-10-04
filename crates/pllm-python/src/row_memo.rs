use crate::invalid;
use pllm_core::row_memo::RowMemo;
use pyo3::{prelude::*, types::PyBytes};
use std::sync::Mutex;

#[pyclass(frozen, name = "_IntegerRowMemo", module = "pllm._native")]
struct Memo(Mutex<RowMemo>);
#[pymethods]
impl Memo {
    #[new]
    #[pyo3(signature = (weights, rows, columns, maximum, threads=1))]
    fn new(
        py: Python<'_>,
        weights: &[u8],
        rows: usize,
        columns: usize,
        maximum: usize,
        threads: usize,
    ) -> PyResult<Self> {
        let memo = py
            .detach(|| RowMemo::new(weights, rows, columns, maximum, threads))
            .map_err(invalid)?;
        Ok(Self(Mutex::new(memo)))
    }
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        input: &[u8],
        rows: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let data: Vec<u8> = py
            .detach(|| {
                self.0
                    .lock()
                    .map_err(|_| "memo lock failed".to_owned())?
                    .evaluate(input, rows)
                    .map(|values| values.into_iter().flat_map(i32::to_le_bytes).collect())
            })
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &data))
    }
    fn stats(&self) -> PyResult<(u64, u64, usize, usize)> {
        Ok(self
            .0
            .lock()
            .map_err(|_| invalid("memo lock failed".into()))?
            .stats())
    }
    fn clear(&self) -> PyResult<()> {
        self.0
            .lock()
            .map_err(|_| invalid("memo lock failed".into()))?
            .clear();
        Ok(())
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Memo>()
}
