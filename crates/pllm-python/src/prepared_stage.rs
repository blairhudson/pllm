use crate::{invalid, mask_stream::Rows};
use pllm_core::prepared_stage::{StageLayout, StageQuery};
use pyo3::{exceptions::PyBufferError, prelude::*, types::PyBytes};
use std::sync::{Arc, Mutex};
use zeroize::Zeroizing;

#[pyclass(frozen, name = "PreparedStageLayout", module = "pllm._native")]
struct Layout(Arc<StageLayout>);

#[pymethods]
impl Layout {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        stage: &str,
        input_width: usize,
        widths: &[u8],
        wire_bits: u8,
        input_bits: u8,
        activation_bits: u8,
        maximum_rows: usize,
        weight_scales: &[u8],
        bias: Option<&[u8]>,
    ) -> PyResult<Self> {
        // Refuse metadata copies outside the core's public geometry ceiling.
        if stage.is_empty()
            || stage.len() > 512
            || !(1..=4096).contains(&maximum_rows)
            || widths.is_empty()
            || widths.len() > 4_000_000 / maximum_rows
            || weight_scales.len() != widths.len() * 4
            || bias.is_some_and(|v| v.len() != widths.len() * 4)
        {
            return Err(invalid("invalid bounded prepared stage metadata".into()));
        }
        Ok(Self(Arc::new(
            StageLayout::new(
                stage.to_owned(),
                input_width,
                widths.to_vec(),
                wire_bits,
                input_bits,
                activation_bits,
                maximum_rows,
                weight_scales.to_vec(),
                bias.map(Vec::from),
            )
            .map_err(invalid)?,
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare<'py>(
        &self,
        py: Python<'py>,
        masks: &Rows,
        start: usize,
        rows: usize,
        chunk_rows: usize,
        tickets: Vec<String>,
        input: &[u8],
    ) -> PyResult<(Query, Vec<Bound<'py, PyBytes>>)> {
        let (query, requests) = py
            .detach(|| {
                self.0.prepare(
                    &mut *masks.0.lock().map_err(|_| "mask lock failed".to_owned())?,
                    start,
                    rows,
                    chunk_rows,
                    tickets,
                    input,
                )
            })
            .map_err(invalid)?;
        Ok((
            Query(Mutex::new(query)),
            requests
                .into_iter()
                .map(|data| PyBytes::new(py, &data))
                .collect(),
        ))
    }
}

#[pyclass(frozen, name = "PreparedStageQuery", module = "pllm._native")]
struct Query(Mutex<StageQuery>);

#[pymethods]
impl Query {
    fn receive(&self, py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = match payload.cast::<PyBytes>() {
            Ok(bytes) => bytes.as_bytes(),
            Err(error) => {
                self.cancel(py)?;
                return Err(error.into());
            }
        };
        py.detach(|| {
            self.0
                .lock()
                .map_err(|_| "stage lock failed".to_owned())?
                .receive(bytes)
        })
        .map_err(invalid)
    }

    fn finish(&self, py: Python<'_>) -> PyResult<(Output, u64)> {
        let (bytes, ns) = py
            .detach(|| {
                self.0
                    .lock()
                    .map_err(|_| "stage lock failed".to_owned())?
                    .finish()
            })
            .map_err(invalid)?;
        Ok((Output(bytes), ns))
    }

    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.0
                .lock()
                .map_err(|_| "stage lock failed".to_owned())?
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
                .map_err(|_| "stage lock failed".to_owned())?
                .retained_bytes())
        })
        .map_err(invalid)
    }
}

#[pyclass(frozen, name = "PreparedStageOutput", module = "pllm._native")]
struct Output(Zeroizing<Vec<u8>>);

#[pymethods]
impl Output {
    /// # Safety
    /// CPython supplies the checked output pointer. FillInfo retains this frozen
    /// owner and refuses writable views. No mutation or explicit close is exposed;
    /// only the final retained owner zeroizes the allocation.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("null prepared output buffer"));
        }
        let len = isize::try_from(slf.0.len())
            .map_err(|_| PyBufferError::new_err("prepared output buffer overflow"))?;
        // SAFETY: ownership, immutability and lifetime conditions above hold.
        let status = unsafe {
            pyo3::ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                slf.0.as_ptr().cast_mut().cast(),
                len,
                1,
                flags,
            )
        };
        if status == -1 {
            Err(PyErr::fetch(slf.py()))
        } else {
            Ok(())
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Layout>()?;
    module.add_class::<Query>()?;
    module.add_class::<Output>()
}
