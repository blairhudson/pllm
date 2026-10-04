use crate::invalid;
use pllm_core::masked_aggregate::{self, OutputMask, OutputReceiver, OutputRelay};
use pyo3::{
    prelude::*,
    types::{PyAny, PyBytes},
};
use std::sync::Mutex;

#[pyclass(frozen, name = "_OutputMask", module = "pllm._native")]
struct Worker(Mutex<Option<OutputMask>>);
#[pymethods]
impl Worker {
    fn apply<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let state = self
            .0
            .lock()
            .map_err(|_| invalid("mask lock failed".into()))?
            .take()
            .ok_or_else(|| invalid("output mask already consumed".into()))?;
        let value: &[u8] = value.extract()?;
        let result = py.detach(|| state.mask(value)).map_err(invalid)?;
        Ok(PyBytes::new(py, &result))
    }
}

#[pyclass(frozen, name = "_OutputRelay", module = "pllm._native")]
struct Relay(Mutex<Option<OutputRelay>>);
#[pymethods]
impl Relay {
    fn combine<'py>(
        &self,
        py: Python<'py>,
        own: &Bound<'py, PyAny>,
        peer: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let state = self
            .0
            .lock()
            .map_err(|_| invalid("relay lock failed".into()))?
            .take()
            .ok_or_else(|| invalid("output relay already consumed".into()))?;
        let own: &[u8] = own.extract()?;
        let peer: &[u8] = peer.extract()?;
        let result = py.detach(|| state.combine(own, peer)).map_err(invalid)?;
        Ok(PyBytes::new(py, &result))
    }
}

#[pyclass(frozen, name = "_OutputReceiver", module = "pllm._native")]
struct Receiver(Mutex<Option<OutputReceiver>>);
#[pymethods]
impl Receiver {
    fn finish<'py>(
        &self,
        py: Python<'py>,
        frame: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let state = self
            .0
            .lock()
            .map_err(|_| invalid("receiver lock failed".into()))?
            .take()
            .ok_or_else(|| invalid("output receiver already consumed".into()))?;
        let frame: &[u8] = frame.extract()?;
        let result = py.detach(|| state.finish(frame)).map_err(invalid)?;
        Ok(PyBytes::new(py, &result))
    }
}

#[pyfunction]
fn output_aggregation_issue(
    py: Python<'_>,
    context: &[u8],
    widths: &[u8],
    rows: usize,
) -> PyResult<(Worker, Relay, Receiver)> {
    let context: [u8; 32] = context
        .try_into()
        .map_err(|_| invalid("context must be 32 bytes".into()))?;
    let (w, r, c) = py
        .detach(|| masked_aggregate::issue(&context, widths, rows))
        .map_err(invalid)?;
    Ok((
        Worker(Mutex::new(Some(w))),
        Relay(Mutex::new(Some(r))),
        Receiver(Mutex::new(Some(c))),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Worker>()?;
    module.add_class::<Relay>()?;
    module.add_class::<Receiver>()?;
    module.add_function(wrap_pyfunction!(output_aggregation_issue, module)?)
}
