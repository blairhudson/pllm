//! Separate bounded client/worker owners. Call parsing failure retires live state.

use std::collections::BTreeMap;

use pllm_compiler::{
    GkmRuntimeContract, GkmTransportClient, GkmTransportImport, GkmTransportWorker,
};
use pllm_core::gkm::{WireBytes, Worker};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyString, PyTuple};

fn invalid(message: &str) -> PyErr {
    crate::invalid(message.into())
}
fn retired() -> PyErr {
    invalid("GKM transport owner is retired")
}

/// Parse the fixed bounded signatures only after taking the
/// owner. Raw *args/**kwargs prevent PyO3 from rejecting arity/keyword errors
/// before retirement. Python errors before method dispatch cannot reach us.
/// Counts are bounded before inspecting values; no caller-sized collection or
/// caller-supplied keyword string copy is made here.
fn arguments<'py, const N: usize>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
) -> PyResult<[Bound<'py, PyAny>; N]> {
    crate::owned_arguments::arguments(args, kwargs, names, "GKM")
}

#[allow(clippy::too_many_arguments)]
fn contract(
    py: Python<'_>,
    experiment: &[u8],
    config: &[u8],
    rows: usize,
    outputs: usize,
    source: &str,
    tokenizer: &str,
) -> PyResult<GkmRuntimeContract> {
    let source = serde_json::from_value(serde_json::Value::String(source.into()))
        .map_err(|_| invalid("invalid GKM source digest"))?;
    let tokenizer = serde_json::from_value(serde_json::Value::String(tokenizer.into()))
        .map_err(|_| invalid("invalid GKM tokenizer digest"))?;
    py.detach(|| GkmRuntimeContract::compile(experiment, config, rows, outputs, source, tokenizer))
        .map_err(crate::invalid)
}

fn sources<'py>(
    contract: &GkmRuntimeContract,
    tensors: &Bound<'py, PyDict>,
) -> PyResult<Vec<(String, Bound<'py, PyBytes>)>> {
    let shapes = contract.decoder().source_shapes();
    if tensors.len() != shapes.len() {
        return Err(invalid("GKM source tensor set mismatch"));
    }
    let mut owners = Vec::new();
    for (name, shape) in shapes {
        let value = tensors
            .get_item(name)?
            .ok_or_else(|| invalid("missing GKM source tensor"))?;
        let bytes = value
            .cast_into::<PyBytes>()
            .map_err(|_| invalid("GKM source must be bytes"))?;
        if bytes.as_bytes().len() != shape.iter().product::<usize>() * 4 {
            return Err(invalid("GKM source shape mismatch"));
        }
        owners.push((name.clone(), bytes));
    }
    Ok(owners)
}

fn pair(py: Python<'_>, bytes: [WireBytes; 2]) -> (Bound<'_, PyBytes>, Bound<'_, PyBytes>) {
    (
        PyBytes::new(py, bytes[0].as_bytes()),
        PyBytes::new(py, bytes[1].as_bytes()),
    )
}

#[pyclass(name = "_GkmTransportImport")]
struct PyImport {
    inner: Option<GkmTransportImport>,
}

#[pymethods]
impl PyImport {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        experiment: &Bound<'_, PyBytes>,
        config: &Bound<'_, PyBytes>,
        input_rows: usize,
        output_tokens: usize,
        source_lock: &str,
        tokenizer: &str,
    ) -> PyResult<Self> {
        let contract = contract(
            py,
            experiment.as_bytes(),
            config.as_bytes(),
            input_rows,
            output_tokens,
            source_lock,
            tokenizer,
        )?;
        let inner = py
            .detach(|| GkmTransportImport::new(contract))
            .map_err(crate::invalid)?;
        Ok(Self { inner: Some(inner) })
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        let [name, offset, data] = arguments(args, kwargs, ["name", "offset", "data"])?;
        let name = name
            .cast::<PyString>()
            .map_err(|_| invalid("GKM source name must be a string"))?
            .to_str()?;
        if offset.is_instance_of::<PyBool>() {
            return Err(invalid("GKM source offset must be an integer"));
        }
        let offset = offset.extract::<usize>()?;
        let bytes = data
            .cast::<PyBytes>()
            .map_err(|_| invalid("GKM source frame must be bytes"))?
            .as_bytes();
        py.detach(|| inner.push(name, offset, bytes))
            .map_err(crate::invalid)?;
        self.inner = Some(inner);
        Ok(())
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn client<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyClient> {
        let inner = self.inner.take().ok_or_else(retired)?;
        arguments(args, kwargs, [])?;
        let client = py.detach(|| inner.client()).map_err(crate::invalid)?;
        Ok(PyClient {
            inner: Some(client),
        })
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn worker<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyWorker> {
        let inner = self.inner.take().ok_or_else(retired)?;
        let [publication, pin, role, start] =
            arguments(args, kwargs, ["publication", "pin", "role", "start"])?;
        let publication = publication
            .cast::<PyBytes>()
            .map_err(|_| invalid("GKM publication must be bytes"))?
            .as_bytes();
        let pin: [u8; 32] = pin
            .cast::<PyBytes>()
            .map_err(|_| invalid("GKM pin must be bytes"))?
            .as_bytes()
            .try_into()
            .map_err(|_| invalid("invalid GKM publication pin"))?;
        if role.is_instance_of::<PyBool>() {
            return Err(invalid("GKM worker role must be an integer"));
        }
        let role = match role.extract::<usize>()? {
            0 => Worker::Zero,
            1 => Worker::One,
            _ => return Err(invalid("invalid GKM worker role")),
        };
        let start = start
            .cast::<PyBytes>()
            .map_err(|_| invalid("GKM start must be bytes"))?
            .as_bytes();
        let worker = py
            .detach(|| inner.worker(publication, pin, role, start))
            .map_err(crate::invalid)?;
        Ok(PyWorker {
            inner: Some(worker),
        })
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn abort<'py>(
        &mut self,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        self.inner = None;
        arguments(args, kwargs, [])?;
        Ok(())
    }
}

#[pyclass(name = "_GkmTransportClient")]
struct PyClient {
    inner: Option<GkmTransportClient>,
}

#[pymethods]
impl PyClient {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        experiment: &Bound<'_, PyBytes>,
        config: &Bound<'_, PyBytes>,
        input_rows: usize,
        output_tokens: usize,
        source_lock: &str,
        tokenizer: &str,
        tensors: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let contract = contract(
            py,
            experiment.as_bytes(),
            config.as_bytes(),
            input_rows,
            output_tokens,
            source_lock,
            tokenizer,
        )?;
        let owners = sources(&contract, tensors)?;
        let borrowed: BTreeMap<_, _> = owners
            .iter()
            .map(|(k, v)| (k.clone(), v.as_bytes()))
            .collect();
        let inner = py
            .detach(|| GkmTransportClient::new(contract, &borrowed))
            .map_err(crate::invalid)?;
        Ok(Self { inner: Some(inner) })
    }

    fn contract<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            &self.inner.as_ref().ok_or_else(retired)?.contract_bytes(),
        ))
    }

    fn decoder<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            &self.inner.as_ref().ok_or_else(retired)?.decoder_bytes(),
        ))
    }

    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (name, bytes) in self.inner.as_ref().ok_or_else(retired)?.local_tensors() {
            result.set_item(name, PyBytes::new(py, bytes))?;
        }
        Ok(result)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn publication<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        arguments(args, kwargs, [])?;
        let bytes = py
            .detach(|| inner.publication_bytes())
            .map_err(crate::invalid)?;
        let result = PyBytes::new(py, bytes.as_bytes());
        self.inner = Some(inner);
        Ok(result)
    }

    fn publication_digest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            &self
                .inner
                .as_ref()
                .ok_or_else(retired)?
                .publication_digest(),
        ))
    }

    fn request_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            &self.inner.as_ref().ok_or_else(retired)?.request_id(),
        ))
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn starts<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        arguments(args, kwargs, [])?;
        let bytes = py.detach(|| inner.starts()).map_err(crate::invalid)?;
        let result = pair(py, bytes);
        self.inner = Some(inner);
        Ok(result)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn lookup<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        let [token] = arguments(args, kwargs, ["token"])?;
        if token.is_instance_of::<PyBool>() {
            return Err(invalid("GKM token must be an integer"));
        }
        let token = token.extract::<usize>()?;
        let bytes = py.detach(|| inner.lookup(token)).map_err(crate::invalid)?;
        let result = pair(py, bytes);
        self.inner = Some(inner);
        Ok(result)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn linear<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        let [stage, input] = arguments(args, kwargs, ["stage", "input"])?;
        if stage.is_instance_of::<PyBool>() {
            return Err(invalid("GKM stage must be an integer"));
        }
        let stage = stage.extract::<usize>()?;
        let input = input
            .cast::<PyBytes>()
            .map_err(|_| invalid("GKM input must be float32 bytes"))?
            .as_bytes();
        let bytes = py
            .detach(|| inner.linear(stage, input))
            .map_err(crate::invalid)?;
        let result = pair(py, bytes);
        self.inner = Some(inner);
        Ok(result)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn accept<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        let [zero, one] = arguments(args, kwargs, ["zero", "one"])?;
        let zero = zero
            .cast::<PyBytes>()
            .map_err(|_| invalid("GKM reply must be bytes"))?
            .as_bytes();
        let one = one
            .cast::<PyBytes>()
            .map_err(|_| invalid("GKM reply must be bytes"))?
            .as_bytes();
        let bytes = py
            .detach(|| inner.accept(zero, one))
            .map_err(crate::invalid)?;
        let result = PyBytes::new(py, &bytes);
        self.inner = Some(inner);
        Ok(result)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn finish<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        arguments(args, kwargs, [])?;
        let bytes = py.detach(|| inner.finish()).map_err(crate::invalid)?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn abort<'py>(
        &mut self,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        self.inner = None;
        arguments(args, kwargs, [])?;
        Ok(())
    }
}

#[pyclass(name = "_GkmTransportWorker")]
struct PyWorker {
    inner: Option<GkmTransportWorker>,
}

#[pymethods]
impl PyWorker {
    fn request_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let request = self
            .inner
            .as_ref()
            .ok_or_else(retired)?
            .request_id()
            .map_err(crate::invalid)?;
        Ok(PyBytes::new(py, &request))
    }

    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        experiment: &Bound<'_, PyBytes>,
        config: &Bound<'_, PyBytes>,
        input_rows: usize,
        output_tokens: usize,
        source_lock: &str,
        tokenizer: &str,
        tensors: &Bound<'_, PyDict>,
        publication: &Bound<'_, PyBytes>,
        pin: &Bound<'_, PyBytes>,
        role: usize,
        start: &Bound<'_, PyBytes>,
    ) -> PyResult<Self> {
        let role = match role {
            0 => Worker::Zero,
            1 => Worker::One,
            _ => return Err(invalid("invalid GKM worker role")),
        };
        let pin: [u8; 32] = pin
            .as_bytes()
            .try_into()
            .map_err(|_| invalid("invalid GKM publication pin"))?;
        let contract = contract(
            py,
            experiment.as_bytes(),
            config.as_bytes(),
            input_rows,
            output_tokens,
            source_lock,
            tokenizer,
        )?;
        let owners = sources(&contract, tensors)?;
        let borrowed: BTreeMap<_, _> = owners
            .iter()
            .map(|(k, v)| (k.clone(), v.as_bytes()))
            .collect();
        let publication = publication.as_bytes();
        let start = start.as_bytes();
        let inner = py
            .detach(|| GkmTransportWorker::new(contract, &borrowed, publication, pin, role, start))
            .map_err(crate::invalid)?;
        Ok(Self { inner: Some(inner) })
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        let [request] = arguments(args, kwargs, ["request"])?;
        let request = request
            .cast::<PyBytes>()
            .map_err(|_| invalid("GKM frame must be bytes"))?
            .as_bytes();
        let bytes = py
            .detach(|| inner.execute(request))
            .map_err(crate::invalid)?;
        let result = PyBytes::new(py, bytes.as_bytes());
        self.inner = Some(inner);
        Ok(result)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn finish<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut inner = self.inner.take().ok_or_else(retired)?;
        arguments(args, kwargs, [])?;
        let bytes = py.detach(|| inner.finish()).map_err(crate::invalid)?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn abort<'py>(
        &mut self,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        self.inner = None;
        arguments(args, kwargs, [])?;
        Ok(())
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyImport>()?;
    module.add_class::<PyClient>()?;
    module.add_class::<PyWorker>()
}
