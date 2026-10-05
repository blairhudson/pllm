use crate::invalid;
use pllm_core::mask_stream::MaskRows;
use pyo3::{prelude::*, types::PyBytes};
use std::sync::Mutex;

#[pyclass(frozen, name = "PreparedMaskRows", module = "pllm._native")]
struct Rows(Mutex<MaskRows>);

#[pymethods]
impl Rows {
    #[new]
    fn new(
        seed: &[u8],
        input_domain: &[u8],
        output_domain: &[u8],
        rows: usize,
        input_width: usize,
        output_width: usize,
        wire_bits: u8,
    ) -> PyResult<Self> {
        let seed = seed
            .try_into()
            .map_err(|_| invalid("mask seed must be 32 bytes".into()))?;
        Ok(Self(Mutex::new(
            MaskRows::new(
                seed,
                input_domain,
                output_domain,
                rows,
                input_width,
                output_width,
                wire_bits,
            )
            .map_err(invalid)?,
        )))
    }

    fn take<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        count: usize,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let (input, output) = py
            .detach(|| {
                self.0
                    .lock()
                    .map_err(|_| "mask lock failed".to_owned())?
                    .take(start, count)
            })
            .map_err(invalid)?;
        Ok((PyBytes::new(py, &input), PyBytes::new(py, &output)))
    }

    fn burn(&self, py: Python<'_>, start: usize, count: usize) -> PyResult<()> {
        py.detach(|| {
            self.0
                .lock()
                .map_err(|_| "mask lock failed".to_owned())?
                .burn(start, count)
        })
        .map_err(invalid)
    }

    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.0
                .lock()
                .map_err(|_| "mask lock failed".to_owned())?
                .cancel();
            Ok(())
        })
        .map_err(invalid)
    }

    #[getter]
    fn retained_bytes(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| {
            Ok(self
                .0
                .lock()
                .map_err(|_| "mask lock failed".to_owned())?
                .retained_bytes())
        })
        .map_err(invalid)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Rows>()
}
