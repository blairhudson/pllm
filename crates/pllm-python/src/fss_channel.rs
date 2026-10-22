//! Consuming trusted-client boundary; no worker execution or SDK serving claim.
//! Python copies are extra owners: inputs <=272 bytes, replies <=184 bytes,
//! outputs <=4 u32. Returned native words remain zeroizing through list creation.
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
fn acceptance_error(error: Error, checked_ranges: bool) -> PyErr {
    if checked_ranges && error == Error::Domain {
        crate::replicated_transport::NumericDomainRejected::new_err(
            "FuseFSS request failed protected range checks",
        )
    } else {
        native_error(error)
    }
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
        let checked_ranges = owner.checked_ranges();
        let words = py
            .detach(|| owner.finish(response0, response1))
            .map_err(|error| acceptance_error(error, checked_ranges))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replicated_transport::NumericDomainRejected;
    use std::fmt::Write as _;

    fn client(py: Python<'_>, checked: bool) -> Bound<'_, PyAny> {
        // Codec-only opaque-contract fixture. No evaluator or authenticated
        // transport is simulated by these public, unkeyed frame checksums.
        let contract = b"{}";
        let mut digest = String::with_capacity(64);
        for byte in contract_digest(contract).unwrap() {
            write!(&mut digest, "{byte:02x}").unwrap();
        }
        let manifest = pllm_types::canonical_bytes(&serde_json::json!({
            "contract": digest,
            "initial": 2,
            "input_digest": "02".repeat(32),
            "profile": if checked {
                "fusefss.dense-rope-checked-ranges-ring50-q12.v1"
            } else {
                "fusefss.dense-rope-ring50-q12.v1"
            },
            "program": "03".repeat(32),
            "schema": "pllm.fss.channel.program.v1",
            "selections": 2,
            "source": "04".repeat(32),
            "vocabulary": 7,
        }));
        let manifest = PyBytes::new(py, &manifest);
        py.get_type::<NativeClientRequest>()
            .call1((
                PyBytes::new(py, contract),
                PyBytes::new(py, &[5; 32]),
                (&manifest, &manifest, &manifest),
                vec![3u32, 4],
            ))
            .unwrap()
    }

    fn reply(request: &[u8], role: u8, checked: bool, words: &[u64]) -> Vec<u8> {
        let mut operation = b"pllm.fss.channel.request.v1\0".to_vec();
        operation.extend_from_slice(request);
        let mut bytes = if checked { b"PLLMOC01" } else { b"PLLMOU01" }.to_vec();
        bytes.extend_from_slice(&contract_digest(&operation).unwrap());
        bytes.extend_from_slice(&request[120..152]);
        bytes.extend_from_slice(&request[40..72]);
        bytes.extend_from_slice(&[role, 50, if checked { 2 } else { 1 }, 0]);
        bytes.extend_from_slice(&(words.len() as u32).to_le_bytes());
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&contract_digest(&bytes).unwrap());
        bytes
    }

    #[test]
    fn numeric_rejection_is_typed_only_at_checked_final_acceptance() {
        Python::initialize();
        Python::attach(|py| {
            assert!(
                acceptance_error(Error::Domain, true).is_instance_of::<NumericDomainRejected>(py)
            );
            assert!(
                !acceptance_error(Error::Domain, false).is_instance_of::<NumericDomainRejected>(py)
            );
            assert!(!native_error(Error::Domain).is_instance_of::<NumericDomainRejected>(py));
            for error in [
                Error::Context,
                Error::Consumed,
                Error::Resource,
                Error::Allocation,
            ] {
                assert!(!acceptance_error(error, true).is_instance_of::<NumericDomainRejected>(py));
            }
        });
    }

    #[test]
    fn checked_python_finish_rejects_atomically_and_consumes_every_outcome() {
        Python::initialize();
        Python::attach(|py| {
            for (words, accepted, numeric) in [
                ([6, 0, 1], true, false),
                ([0, 0, 0], false, true),
                ([0, 1, 0], false, false),
                ([0, 0, 2], false, false),
                ([0, 7, 1], false, false),
                ([0, 0, 1 << 50], false, false),
            ] {
                let client = client(py, true);
                let request = client.call_method0("request_bytes").unwrap();
                let request = request.cast_exact::<PyBytes>().unwrap().as_bytes();
                client.call_method1("take_input", (0,)).unwrap();
                client.call_method1("take_input", (1,)).unwrap();
                let a = reply(request, 0, true, &[0, 0, 0]);
                let b = reply(request, 1, true, &words);
                let result =
                    client.call_method1("finish", (PyBytes::new(py, &a), PyBytes::new(py, &b)));
                if accepted {
                    assert_eq!(result.unwrap().extract::<Vec<u32>>().unwrap(), [6, 0]);
                } else {
                    let error = result.unwrap_err();
                    assert_eq!(error.is_instance_of::<NumericDomainRejected>(py), numeric);
                }
                let retired = client.call_method0("request_bytes").unwrap_err();
                assert!(!retired.is_instance_of::<NumericDomainRejected>(py));
            }
        });
    }

    #[test]
    fn python_legacy_vocabulary_errors_and_checked_namespace_downgrades_stay_protocol_errors() {
        Python::initialize();
        Python::attach(|py| {
            for checked in [false, true] {
                let client = client(py, checked);
                let request = client.call_method0("request_bytes").unwrap();
                let request = request.cast_exact::<PyBytes>().unwrap().as_bytes();
                client.call_method1("take_input", (0,)).unwrap();
                client.call_method1("take_input", (1,)).unwrap();
                // Legacy has an invalid vocabulary ID; checked has downgraded
                // namespace/count with no validity word. Neither is numeric rejection.
                let a = reply(request, 0, false, &[0, 0]);
                let b = reply(request, 1, false, &[7, 0]);
                let error = client
                    .call_method1("finish", (PyBytes::new(py, &a), PyBytes::new(py, &b)))
                    .unwrap_err();
                assert!(!error.is_instance_of::<NumericDomainRejected>(py));
                assert!(client.call_method0("request_bytes").is_err());
            }
        });
    }
}
