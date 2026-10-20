//! Consuming trusted-client boundary; no worker execution or SDK serving claim.
//! Python copies are extra owners: inputs <=272 bytes, replies <=176 bytes,
//! tokens <=16 u32. Returned native words remain zeroizing through list creation.
use pllm_compiler::fss_decoder::{
    channel::{
        contract_digest, ClientRequest, MAX_INPUT_FRAME_BYTES, MAX_MANIFEST_BYTES,
        MAX_OUTPUT_FRAME_BYTES,
    },
    Error,
};
use pyo3::{
    prelude::*,
    types::{PyAny, PyBool, PyBytes, PyDict, PyInt, PyList, PyTuple},
};
use zeroize::Zeroizing;

fn invalid(message: impl ToString) -> PyErr {
    crate::invalid(message.to_string())
}
fn native_error(error: Error) -> PyErr {
    invalid(format!("FuseFSS client native rejection: {error:?}"))
}
fn arguments<'py, const N: usize>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
) -> PyResult<[Bound<'py, PyAny>; N]> {
    crate::owned_arguments::arguments(args, kwargs, names, "FuseFSS client")
}
fn bytes<'a>(value: &'a Bound<'_, PyAny>, maximum: usize) -> PyResult<&'a [u8]> {
    let bytes = value.cast_exact::<PyBytes>()?.as_bytes();
    if bytes.len() > maximum {
        return Err(invalid("FuseFSS client bytes exceed native bound"));
    }
    Ok(bytes)
}
fn integer(value: &Bound<'_, PyAny>) -> PyResult<u32> {
    if value.is_instance_of::<PyBool>() {
        return Err(invalid("FuseFSS client integer cannot be bool"));
    }
    value.cast_exact::<PyInt>()?.extract()
}

#[pyclass(module = "pllm._native", name = "FssClientRequest")]
pub struct NativeClientRequest {
    owner: Option<ClientRequest>,
    pid: u32,
}
impl NativeClientRequest {
    fn take(&mut self) -> PyResult<ClientRequest> {
        let owner = self
            .owner
            .take()
            .ok_or_else(|| invalid("FuseFSS client request retired"))?;
        if self.pid != std::process::id() {
            return Err(invalid("FuseFSS client owner cannot cross fork"));
        }
        Ok(owner)
    }
    fn retire(
        &mut self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        drop(self.owner.take());
        if self.pid != std::process::id() {
            return Err(invalid("FuseFSS client owner cannot cross fork"));
        }
        let [] = arguments(args, kwargs, [])?;
        Ok(())
    }
}
#[pymethods]
impl NativeClientRequest {
    #[new]
    #[pyo3(signature=(*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let [contract, session, manifests, tokens] = arguments(
            args,
            kwargs,
            ["contract_bytes", "session", "manifests", "tokens"],
        )?;
        let contract = bytes(&contract, 1 << 20)?;
        if contract.is_empty() {
            return Err(invalid("FuseFSS source contract is empty"));
        }
        let session: [u8; 32] = bytes(&session, 32)?
            .try_into()
            .map_err(|_| invalid("FuseFSS session must be 32 bytes"))?;
        let manifests = manifests.cast_exact::<PyTuple>()?;
        if manifests.len() != 3 {
            return Err(invalid("FuseFSS requires exactly three public manifests"));
        }
        let m0 = manifests.get_item(0)?;
        let m1 = manifests.get_item(1)?;
        let m2 = manifests.get_item(2)?;
        let manifests = [
            bytes(&m0, MAX_MANIFEST_BYTES)?,
            bytes(&m1, MAX_MANIFEST_BYTES)?,
            bytes(&m2, MAX_MANIFEST_BYTES)?,
        ];
        // Plain bounded list/tuple only: no unbounded collect, user iterator,
        // __len__/__getitem__ hook, integer coercion, or pre-admission RNG.
        let list = tokens.cast_exact::<PyList>().ok();
        let tuple = tokens.cast_exact::<PyTuple>().ok();
        let count = match (list, tuple) {
            (Some(v), _) => v.len(),
            (_, Some(v)) => v.len(),
            _ => return Err(invalid("FuseFSS tokens must be a plain list or tuple")),
        };
        if count == 0 || count > 16 {
            return Err(invalid("FuseFSS input token count outside 1..16"));
        }
        let mut words = Zeroizing::new([0u32; 16]);
        for (i, word) in words[..count].iter_mut().enumerate() {
            let item = match list {
                Some(v) => v.get_item(i)?,
                None => tuple.expect("checked tuple").get_item(i)?,
            };
            *word = integer(&item)?;
        }
        let tokens = &words[..count];
        let owner = py
            .detach(|| ClientRequest::new(contract_digest(contract)?, session, manifests, tokens))
            .map_err(native_error)?;
        Ok(Self {
            owner: Some(owner),
            pid: std::process::id(),
        })
    }

    #[pyo3(signature=(*args, **kwargs))]
    fn request_bytes<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let owner = self.take()?;
        let [] = arguments(args, kwargs, [])?;
        let result = PyBytes::new(py, owner.manifest_bytes());
        self.owner = Some(owner);
        Ok(result)
    }

    #[pyo3(signature=(*args, **kwargs))]
    fn take_input<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut owner = self.take()?;
        let [role] = arguments(args, kwargs, ["role"])?;
        let role = integer(&role)?;
        if role > 1 {
            return Err(invalid("FuseFSS holder role must be 0 or 1"));
        }
        let frame = py
            .detach(|| owner.take_input(role as u8))
            .map_err(native_error)?;
        debug_assert!(frame.len() <= MAX_INPUT_FRAME_BYTES);
        let result = PyBytes::new(py, &frame);
        self.owner = Some(owner);
        Ok(result)
    }

    #[pyo3(signature=(*args, **kwargs))]
    fn finish<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let owner = self.take()?;
        let [response0, response1] = arguments(args, kwargs, ["response0", "response1"])?;
        let response0 = bytes(&response0, MAX_OUTPUT_FRAME_BYTES)?;
        let response1 = bytes(&response1, MAX_OUTPUT_FRAME_BYTES)?;
        let words = py
            .detach(|| owner.finish(response0, response1))
            .map_err(native_error)?;
        PyList::new(py, words.iter().copied())
    }

    #[pyo3(signature=(*args, **kwargs))]
    fn close(
        &mut self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.retire(args, kwargs)
    }

    #[pyo3(signature=(*args, **kwargs))]
    fn cancel(
        &mut self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.retire(args, kwargs)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeClientRequest>()
}
