//! Internal research boundary. The trusted client gets only tentative values
//! until finish; invalid calls consume the request before Python conversion.

use std::collections::BTreeMap;

use pllm_compiler::{GkmDecoderContract, GkmDecoderSession};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict};

#[pyfunction]
fn gkm_decoder_contract<'py>(
    py: Python<'py>,
    config: &Bound<'_, PyBytes>,
    input_rows: usize,
    decode_steps: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let config = config.as_bytes();
    let contract = py
        .detach(|| GkmDecoderContract::compile(config, input_rows, decode_steps))
        .map_err(crate::invalid)?;
    Ok(PyBytes::new(py, &contract.canonical_bytes()))
}

#[pyclass(name = "_GkmDecoderSession")]
struct PySession {
    inner: Option<GkmDecoderSession>,
}

#[pymethods]
impl PySession {
    #[new]
    #[pyo3(signature = (config, input_rows, decode_steps, tensors, *, seeded=false))]
    fn new(
        py: Python<'_>,
        config: &Bound<'_, PyBytes>,
        input_rows: usize,
        decode_steps: usize,
        tensors: &Bound<'_, PyDict>,
        seeded: bool,
    ) -> PyResult<Self> {
        let config = config.as_bytes();
        let contract = py
            .detach(|| GkmDecoderContract::compile(config, input_rows, decode_steps))
            .map_err(crate::invalid)?;
        if tensors.len() != contract.source_shapes().len() {
            return Err(crate::invalid("GKM source tensor set mismatch".into()));
        }
        // Retain owned Python references while detached, even if another thread
        // mutates the dictionary. Borrow only immutable, exact-size byte objects.
        let mut owners = Vec::new();
        for (name, shape) in contract.source_shapes() {
            let value = tensors
                .get_item(name)?
                .ok_or_else(|| crate::invalid("missing GKM source tensor".into()))?;
            let bytes = value
                .cast_into::<PyBytes>()
                .map_err(|_| crate::invalid("GKM source must be bytes".into()))?;
            if bytes.as_bytes().len() != shape.iter().product::<usize>() * 4 {
                return Err(crate::invalid("GKM source shape mismatch".into()));
            }
            owners.push((name.clone(), bytes));
        }
        let borrowed: BTreeMap<_, _> = owners
            .iter()
            .map(|(name, bytes)| (name.clone(), bytes.as_bytes()))
            .collect();
        let inner = py
            .detach(|| GkmDecoderSession::new(contract, &borrowed, seeded))
            .map_err(crate::invalid)?;
        Ok(Self { inner: Some(inner) })
    }

    fn contract<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self.inner.as_ref().ok_or_else(retired)?;
        Ok(PyBytes::new(py, &inner.contract_bytes()))
    }

    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.inner.as_ref().ok_or_else(retired)?;
        let result = PyDict::new(py);
        for (name, bytes) in inner.local_tensors() {
            result.set_item(name, PyBytes::new(py, bytes))?;
        }
        Ok(result)
    }

    fn lookup<'py>(
        &mut self,
        py: Python<'py>,
        token: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        if token.is_instance_of::<PyBool>() {
            return Err(crate::invalid("GKM token must be an integer".into()));
        }
        let token = token.extract::<usize>()?;
        let bytes = py.detach(|| inner.lookup(token)).map_err(crate::invalid)?;
        let result = PyBytes::new(py, &bytes);
        self.inner = Some(inner);
        Ok(result)
    }

    fn linear<'py>(
        &mut self,
        py: Python<'py>,
        stage: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        if stage.is_instance_of::<PyBool>() {
            return Err(crate::invalid("GKM stage must be an integer".into()));
        }
        let stage = stage.extract::<usize>()?;
        let input = input
            .cast::<PyBytes>()
            .map_err(|_| crate::invalid("GKM input must be float32 bytes".into()))?
            .as_bytes();
        let bytes = py
            .detach(|| inner.linear(stage, input))
            .map_err(crate::invalid)?;
        let result = PyBytes::new(py, &bytes);
        self.inner = Some(inner);
        Ok(result)
    }

    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        let receipt = py.detach(|| inner.finish()).map_err(crate::invalid)?;
        Ok(PyBytes::new(py, &receipt))
    }

    fn abort(&mut self) {
        self.inner = None;
    }
}

fn retired() -> PyErr {
    crate::invalid("GKM decoder session is retired".into())
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySession>()?;
    module.add_function(wrap_pyfunction!(gkm_decoder_contract, module)?)
}
