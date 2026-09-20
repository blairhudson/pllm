//! Python binding for the native integer core. Immutable byte buffers avoid
//! dangling NumPy borrows and retain the stable Python ABI. Matrices are copied
//! once at compilation. Arithmetic runs without the Python interpreter lock.
use pllm_core::{codec, kernels};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PySequence, PyTuple};
use std::sync::{
    atomic::{AtomicBool, AtomicU8, Ordering},
    Arc, Mutex,
};
use zeroize::Zeroizing;

fn invalid(error: String) -> PyErr {
    PyValueError::new_err(error)
}

fn checked_byte_sequence(
    value: &Bound<'_, PyAny>,
    expected: usize,
    maximum_bytes: usize,
    kind: &str,
) -> PyResult<Vec<Vec<u8>>> {
    let sequence = value
        .cast::<PySequence>()
        .map_err(|_| PyValueError::new_err(format!("Q7 SiLU {kind}s must be a sequence")))?;
    let count = sequence.len()?;
    if count != expected {
        return Err(PyValueError::new_err(format!(
            "Q7 SiLU expected {expected} {kind}s, got {count}"
        )));
    }
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let item = sequence.get_item(index)?;
        let bytes = item.cast::<PyBytes>().map_err(|_| {
            PyValueError::new_err(format!("Q7 SiLU {kind} at index {index} must be bytes"))
        })?;
        if bytes.as_bytes().len() > maximum_bytes {
            return Err(PyValueError::new_err(format!(
                "Q7 SiLU {kind} exceeds its byte bound"
            )));
        }
        output.push(bytes.as_bytes().to_vec());
    }
    Ok(output)
}

fn checked_gated_byte_sequence(
    value: &Bound<'_, PyAny>,
    expected: usize,
    kind: &str,
) -> PyResult<Vec<Vec<u8>>> {
    let sequence = value.cast::<PySequence>().map_err(|_| {
        PyValueError::new_err(format!("gated Q7 multiply {kind}s must be a sequence"))
    })?;
    let count = sequence.len()?;
    if count != expected {
        return Err(PyValueError::new_err(format!(
            "gated Q7 multiply expected {expected} {kind}s, got {count}"
        )));
    }
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let item = sequence.get_item(index)?;
        let bytes = item.cast::<PyBytes>().map_err(|_| {
            PyValueError::new_err(format!(
                "gated Q7 multiply {kind} at index {index} must be bytes"
            ))
        })?;
        if bytes.as_bytes().len() > pllm_compiler::SILU_Q7_MAX_LABEL_BYTES {
            return Err(PyValueError::new_err(format!(
                "gated Q7 multiply {kind} exceeds its byte bound"
            )));
        }
        output.push(bytes.as_bytes().to_vec());
    }
    Ok(output)
}

fn checked_i16_sequence(
    value: &Bound<'_, PyAny>,
    expected: usize,
    kind: &str,
) -> PyResult<Vec<i16>> {
    let sequence = value.cast::<PySequence>().map_err(|_| {
        PyValueError::new_err(format!("gated Q7 multiply {kind}s must be a sequence"))
    })?;
    let count = sequence.len()?;
    if count != expected {
        return Err(PyValueError::new_err(format!(
            "gated Q7 multiply expected {expected} {kind}s, got {count}"
        )));
    }
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let item = sequence.get_item(index)?;
        output.push(item.extract::<i16>().map_err(|_| {
            PyValueError::new_err(format!(
                "gated Q7 multiply {kind} at index {index} must fit i16"
            ))
        })?);
    }
    Ok(output)
}

fn compilation_invalid(diagnostics: Vec<pllm_compiler::Diagnostic>) -> PyErr {
    let json = pllm_compiler::diagnostics_json(&diagnostics);
    PyValueError::new_err(String::from_utf8_lossy(&json).into_owned())
}
fn bytes_u32(values: &[u32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output
}
fn bytes_i32(values: &[i32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output
}
fn py_bytes_u64<'py>(py: Python<'py>, values: &[u64]) -> PyResult<Bound<'py, PyBytes>> {
    PyBytes::new_with(py, std::mem::size_of_val(values), |output| {
        for (chunk, value) in output.chunks_exact_mut(8).zip(values) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    })
}
fn bytes_i64(values: &[i64]) -> Vec<u8> {
    let mut output = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output
}
fn bytes_f32(values: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output
}

fn signed_i8s(bytes: &[u8]) -> PyResult<Vec<i8>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len())
        .map_err(|_| PyRuntimeError::new_err("Freivalds weight allocation failed"))?;
    values.extend(bytes.iter().map(|value| *value as i8));
    Ok(values)
}

fn bounded_copy(bytes: &[u8], maximum: usize, kind: &str) -> PyResult<Vec<u8>> {
    if bytes.is_empty() || bytes.len() > maximum {
        return Err(invalid(format!("{kind} has an invalid byte length")));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes.len())
        .map_err(|_| PyRuntimeError::new_err(format!("{kind} allocation failed")))?;
    output.extend_from_slice(bytes);
    Ok(output)
}

fn exact_i32s(bytes: &[u8], expected: usize, kind: &str) -> PyResult<Zeroizing<Vec<i32>>> {
    if bytes.len()
        != expected
            .checked_mul(4)
            .ok_or_else(|| invalid(format!("{kind} length overflow")))?
    {
        return Err(invalid(format!("{kind} has an invalid byte length")));
    }
    let mut values = Zeroizing::new(Vec::new());
    values
        .try_reserve_exact(expected)
        .map_err(|_| PyRuntimeError::new_err(format!("{kind} allocation failed")))?;
    for chunk in bytes.chunks_exact(4) {
        values.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn exact_i64s(bytes: &[u8], expected: usize, kind: &str) -> PyResult<Zeroizing<Vec<i64>>> {
    if bytes.len()
        != expected
            .checked_mul(8)
            .ok_or_else(|| invalid(format!("{kind} length overflow")))?
    {
        return Err(invalid(format!("{kind} has an invalid byte length")));
    }
    let mut values = Zeroizing::new(Vec::new());
    values
        .try_reserve_exact(expected)
        .map_err(|_| PyRuntimeError::new_err(format!("{kind} allocation failed")))?;
    for chunk in bytes.chunks_exact(8) {
        values.push(i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(values)
}

fn exact_32(bytes: &[u8], kind: &str) -> PyResult<[u8; 32]> {
    bytes
        .try_into()
        .map_err(|_| invalid(format!("{kind} must contain exactly 32 bytes")))
}

fn exact_secret_32(bytes: &[u8], kind: &str) -> PyResult<Zeroizing<[u8; 32]>> {
    Ok(Zeroizing::new(exact_32(bytes, kind)?))
}

fn exact_16(bytes: &[u8], kind: &str) -> PyResult<[u8; 16]> {
    bytes
        .try_into()
        .map_err(|_| invalid(format!("{kind} must contain exactly 16 bytes")))
}

#[pyclass(name = "FreivaldsPolicy", frozen, module = "pllm._native")]
struct PyFreivaldsPolicy {
    policy: pllm_core::FreivaldsPolicy,
    resources: pllm_core::FreivaldsResourcePolicy,
    target_failure_bits: u32,
    max_attempts: u64,
    session_id: [u8; 32],
    session: Option<Arc<Mutex<pllm_core::FreivaldsSession>>>,
}

#[pymethods]
impl PyFreivaldsPolicy {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        target_failure_bits: u32,
        max_attempts: u64,
        max_matrix_elements: usize,
        max_projection_elements: usize,
        max_challenge_elements: usize,
        max_output_elements: usize,
        max_multiply_accumulates: u64,
        max_batch_rows: usize,
        session_id: &Bound<'_, PyBytes>,
        register_session: bool,
    ) -> PyResult<Self> {
        let policy =
            pllm_core::FreivaldsPolicy::sized_for_attempts(target_failure_bits, max_attempts)
                .map_err(|error| invalid(error.to_string()))?;
        let resources = pllm_core::FreivaldsResourcePolicy::new(
            max_matrix_elements,
            max_projection_elements,
            max_challenge_elements,
            max_output_elements,
            max_multiply_accumulates,
            max_batch_rows,
        )
        .map_err(|error| invalid(error.to_string()))?;
        let session_id = exact_32(session_id.as_bytes(), "Freivalds session ID")?;
        let session = if register_session {
            Some(Arc::new(Mutex::new(
                pllm_core::FreivaldsSession::register(session_id, policy)
                    .map_err(|error| invalid(error.to_string()))?,
            )))
        } else {
            None
        };
        Ok(Self {
            policy,
            resources,
            target_failure_bits,
            max_attempts,
            session_id,
            session,
        })
    }

    #[getter]
    fn checks(&self) -> usize {
        self.policy.checks()
    }

    #[getter]
    fn target_failure_bits(&self) -> u32 {
        self.target_failure_bits
    }

    #[getter]
    fn max_attempts(&self) -> u64 {
        self.max_attempts
    }

    #[getter]
    fn conservative_failure_bits(&self) -> u32 {
        self.policy.conservative_failure_bits()
    }
}

#[pyclass(module = "pllm._native")]
struct FreivaldsProjectionInventory {
    inner: Mutex<Option<pllm_core::FreivaldsProjectionBatch>>,
    resources: pllm_core::FreivaldsResourcePolicy,
    session: Option<Arc<Mutex<pllm_core::FreivaldsSession>>>,
}

#[pymethods]
impl FreivaldsProjectionInventory {
    #[getter]
    fn material_id<'python>(&self, py: Python<'python>) -> PyResult<Bound<'python, PyBytes>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds inventory lock poisoned"))?;
        let material = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Freivalds inventory is cancelled"))?;
        Ok(PyBytes::new(py, material.material_id()))
    }

    #[getter]
    fn inventory_rows(&self) -> PyResult<usize> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds inventory lock poisoned"))?;
        Ok(guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Freivalds inventory was cancelled"))?
            .inventory_rows())
    }

    #[getter]
    fn in_features(&self) -> PyResult<usize> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds inventory lock poisoned"))?;
        Ok(guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Freivalds inventory was cancelled"))?
            .in_features())
    }

    #[getter]
    fn out_features(&self) -> PyResult<usize> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds inventory lock poisoned"))?;
        Ok(guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Freivalds inventory was cancelled"))?
            .out_features())
    }

    #[getter]
    fn checks(&self) -> PyResult<usize> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds inventory lock poisoned"))?;
        Ok(guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Freivalds inventory was cancelled"))?
            .checks())
    }

    #[getter]
    fn max_row_l1(&self) -> PyResult<u64> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds inventory lock poisoned"))?;
        Ok(guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Freivalds inventory was cancelled"))?
            .max_row_l1())
    }

    fn payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds inventory lock poisoned"))?;
        let encoded = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Freivalds inventory was cancelled"))?
            .encode()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(PyBytes::new(py, &encoded))
    }

    fn authentication_tag<'py>(
        &self,
        py: Python<'py>,
        root_seed: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let seed = exact_secret_32(root_seed.as_bytes(), "Freivalds root seed")?;
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds inventory lock poisoned"))?;
        let tag = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Freivalds inventory was cancelled"))?
            .authentication_tag(&seed)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(PyBytes::new(py, &tag))
    }

    fn claim(
        &self,
        root_seed: &Bound<'_, PyBytes>,
        binding: &Bound<'_, PyBytes>,
        row_start: usize,
        batch_rows: usize,
    ) -> PyResult<FreivaldsVerifierHandle> {
        let seed = exact_secret_32(root_seed.as_bytes(), "Freivalds root seed")?;
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| invalid("Freivalds preparation material cannot be claimed".into()))?;
        let mut session = session
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds session lock poisoned"))?;
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds inventory lock poisoned"))?;
        let verifier = guard
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Freivalds inventory was cancelled"))?
            .claim_verifier(
                &seed,
                binding.as_bytes(),
                row_start,
                batch_rows,
                &mut session,
                self.resources,
            )
            .map_err(|error| invalid(error.to_string()))?;
        Ok(FreivaldsVerifierHandle {
            inner: Mutex::new(Some(verifier)),
            state: AtomicU8::new(0),
        })
    }

    fn cancel(&self) -> PyResult<bool> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds inventory lock poisoned"))?
            .take()
            .is_some())
    }
}

#[pyclass(name = "FreivaldsVerifier", module = "pllm._native")]
struct FreivaldsVerifierHandle {
    inner: Mutex<Option<pllm_core::FreivaldsVerifier>>,
    state: AtomicU8,
}

#[pymethods]
impl FreivaldsVerifierHandle {
    fn verify<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| PyRuntimeError::new_err("Freivalds verifier was already consumed"))?;
        let verifier = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds verifier lock poisoned"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Freivalds verifier was already consumed"))?;
        self.state.store(2, Ordering::Release);
        let batch_rows = verifier.batch_rows();
        let (out_features, in_features) = verifier.shape();
        let input = input
            .downcast::<PyBytes>()
            .map_err(|_| PyTypeError::new_err("Freivalds input must be bytes"))?;
        let output = output
            .downcast::<PyBytes>()
            .map_err(|_| PyTypeError::new_err("Freivalds output must be bytes"))?;
        let input_values = exact_i32s(
            input.as_bytes(),
            batch_rows
                .checked_mul(in_features)
                .ok_or_else(|| invalid("Freivalds input length overflow".into()))?,
            "Freivalds input",
        )?;
        let output_values = exact_i64s(
            output.as_bytes(),
            batch_rows
                .checked_mul(out_features)
                .ok_or_else(|| invalid("Freivalds output length overflow".into()))?,
            "Freivalds output",
        )?;
        let mut verifier = verifier;
        let verified = py
            .detach(move || verifier.verify(&input_values, &output_values))
            .map_err(|error| invalid(error.to_string()))?;
        Ok(PyBytes::new(py, &bytes_i64(verified.as_slice())))
    }

    fn cancel(&self) -> PyResult<bool> {
        if self
            .state
            .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(false);
        }
        Ok(self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Freivalds verifier lock poisoned"))?
            .take()
            .is_some())
    }
}

#[pyclass(frozen, module = "pllm._native")]
struct Matrix {
    inner: kernels::Matrix,
}
#[pymethods]
impl Matrix {
    #[new]
    fn new(data: &Bound<'_, PyBytes>, rows: usize, cols: usize) -> PyResult<Self> {
        Ok(Self {
            inner: kernels::Matrix::new(data.as_bytes(), rows, cols).map_err(invalid)?,
        })
    }
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    #[getter]
    fn weight_bytes(&self) -> usize {
        self.inner.weight_bytes()
    }
}

#[pyclass(frozen, module = "pllm._native")]
struct CompiledPlan {
    inner: pllm_compiler::CompiledPlan,
}

#[pyclass(frozen, module = "pllm._native")]
struct GarbledSiluQ7Material {
    inner: pllm_compiler::BoundSiluQ7Material,
}
#[pymethods]
impl GarbledSiluQ7Material {
    #[getter]
    fn gate<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.evaluator_payload())
    }
    fn encode<'py>(&self, py: Python<'py>, value: i16) -> PyResult<Bound<'py, PyBytes>> {
        let label = self
            .inner
            .encode(value)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(PyBytes::new(py, &label))
    }
    fn decode(&self, label: &Bound<'_, PyBytes>) -> PyResult<i16> {
        self.inner
            .decode(label.as_bytes())
            .map_err(|error| invalid(error.to_string()))
    }
}

#[pyclass(module = "pllm._native")]
struct SiluQ7Evaluator {
    inner: Mutex<pllm_compiler::SiluQ7Evaluator>,
    elements: usize,
}

#[pyclass(frozen, module = "pllm._native")]
struct GatedMultiplyQ7Region {
    inner: pllm_compiler::ModelGatedMultiplyQ7Region,
    plan: Arc<pllm_models::DecoderPlan>,
}

#[pymethods]
impl GatedMultiplyQ7Region {
    #[getter]
    fn digest(&self) -> String {
        pllm_compiler::model_gated_multiply_q7_region_digest(&self.inner).to_string()
    }

    #[getter]
    fn multiply_operation_id(&self) -> &str {
        &self.inner.multiply_operation_id
    }

    #[getter]
    fn method_component_id(&self) -> &str {
        &self.inner.method_component_id
    }

    #[getter]
    fn schedule_component_id(&self) -> &str {
        &self.inner.schedule_component_id
    }

    #[getter]
    fn max_tensor_elements(&self) -> usize {
        self.inner.max_tensor_elements
    }

    fn prepare_material(&self, py: Python<'_>) -> PyResult<GarbledGatedMultiplyQ7Material> {
        let inner = py
            .detach(|| {
                pllm_compiler::prepare_bound_gated_multiply_q7_material(&self.plan, &self.inner)
            })
            .map_err(invalid)?;
        Ok(GarbledGatedMultiplyQ7Material { inner })
    }

    fn prepare_evaluator(
        &self,
        py: Python<'_>,
        payload: &Bound<'_, PyBytes>,
    ) -> PyResult<GatedMultiplyQ7Evaluator> {
        if payload.as_bytes().len() > pllm_compiler::GATED_MULTIPLY_Q7_MAX_EVALUATOR_PAYLOAD_BYTES {
            return Err(invalid(
                "gated Q7 multiply evaluator payload exceeds its byte bound".into(),
            ));
        }
        let payload = payload.as_bytes().to_vec();
        let inner = py
            .detach(|| pllm_compiler::GatedMultiplyQ7Evaluator::new(&self.inner, &payload))
            .map_err(invalid)?;
        Ok(GatedMultiplyQ7Evaluator {
            inner: Mutex::new(inner),
            claimed: AtomicBool::new(false),
        })
    }
}

#[pyclass(frozen, module = "pllm._native")]
struct GarbledGatedMultiplyQ7Material {
    inner: pllm_compiler::BoundGatedMultiplyQ7Material,
}

#[pymethods]
impl GarbledGatedMultiplyQ7Material {
    #[getter]
    fn evaluator_payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.evaluator_payload())
    }

    #[getter]
    fn element_count(&self) -> usize {
        self.inner.element_count()
    }

    fn cancel(&self) -> PyResult<bool> {
        self.inner.cancel().map_err(invalid)
    }

    fn encode_gate<'py>(&self, py: Python<'py>, value: i16) -> PyResult<Bound<'py, PyBytes>> {
        let label = self
            .inner
            .encode_gate(value)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(PyBytes::new(py, &label))
    }

    fn encode_up<'py>(&self, py: Python<'py>, value: i16) -> PyResult<Bound<'py, PyBytes>> {
        let label = self
            .inner
            .encode_up(value)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(PyBytes::new(py, &label))
    }

    fn decode(&self, label: &Bound<'_, PyBytes>) -> PyResult<i16> {
        self.inner
            .decode(label.as_bytes())
            .map_err(|error| invalid(error.to_string()))
    }

    fn encode_gates<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let values = checked_i16_sequence(values, self.inner.element_count(), "gate value")?;
        let labels = self
            .inner
            .encode_gates(&values)
            .map_err(|error| invalid(error.to_string()))?;
        PyTuple::new(py, labels.iter().map(|label| PyBytes::new(py, label)))
    }

    fn encode_ups<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let values = checked_i16_sequence(values, self.inner.element_count(), "up value")?;
        let labels = self
            .inner
            .encode_ups(&values)
            .map_err(|error| invalid(error.to_string()))?;
        PyTuple::new(py, labels.iter().map(|label| PyBytes::new(py, label)))
    }

    fn decode_tensor(&self, labels: &Bound<'_, PyAny>) -> PyResult<Vec<i16>> {
        let labels =
            checked_gated_byte_sequence(labels, self.inner.element_count(), "output label")?;
        self.inner
            .decode_tensor(&labels)
            .map_err(|error| invalid(error.to_string()))
    }
}

#[pyclass(module = "pllm._native")]
struct GatedMultiplyQ7Evaluator {
    inner: Mutex<pllm_compiler::GatedMultiplyQ7Evaluator>,
    claimed: AtomicBool,
}

impl GatedMultiplyQ7Evaluator {
    fn claim(&self) -> PyResult<()> {
        if self.claimed.swap(true, Ordering::AcqRel) {
            return Err(invalid(
                "gated Q7 multiply evaluator was already consumed".into(),
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl GatedMultiplyQ7Evaluator {
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        gate_label: &Bound<'_, PyAny>,
        up_label: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.claim()?;
        let gate_label = match gate_label.cast::<PyBytes>() {
            Ok(label) => label,
            Err(_) => {
                self.inner
                    .lock()
                    .map_err(|_| {
                        PyRuntimeError::new_err("gated Q7 multiply evaluator lock was poisoned")
                    })?
                    .burn()
                    .map_err(invalid)?;
                return Err(invalid("gated Q7 multiply gate label must be bytes".into()));
            }
        };
        let up_label = match up_label.cast::<PyBytes>() {
            Ok(label) => label,
            Err(_) => {
                self.inner
                    .lock()
                    .map_err(|_| {
                        PyRuntimeError::new_err("gated Q7 multiply evaluator lock was poisoned")
                    })?
                    .burn()
                    .map_err(invalid)?;
                return Err(invalid("gated Q7 multiply up label must be bytes".into()));
            }
        };
        if gate_label.as_bytes().len() > pllm_compiler::SILU_Q7_MAX_LABEL_BYTES
            || up_label.as_bytes().len() > pllm_compiler::SILU_Q7_MAX_LABEL_BYTES
        {
            self.inner
                .lock()
                .map_err(|_| {
                    PyRuntimeError::new_err("gated Q7 multiply evaluator lock was poisoned")
                })?
                .burn()
                .map_err(invalid)?;
            return Err(invalid(
                "gated Q7 multiply label exceeds its byte bound".into(),
            ));
        }
        let gate_label = gate_label.as_bytes().to_vec();
        let up_label = up_label.as_bytes().to_vec();
        let output = py
            .detach(|| {
                self.inner
                    .lock()
                    .map_err(|_| "gated Q7 multiply evaluator lock was poisoned".to_string())?
                    .evaluate(&gate_label, &up_label)
            })
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &output))
    }

    fn evaluate_tensor<'py>(
        &self,
        py: Python<'py>,
        gate_labels: &Bound<'_, PyAny>,
        up_labels: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        self.claim()?;
        let expected = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("gated Q7 multiply evaluator lock was poisoned"))?
            .element_count();
        let gates = match checked_gated_byte_sequence(gate_labels, expected, "gate label") {
            Ok(labels) => labels,
            Err(error) => {
                self.inner
                    .lock()
                    .map_err(|_| {
                        PyRuntimeError::new_err("gated Q7 multiply evaluator lock was poisoned")
                    })?
                    .burn()
                    .map_err(invalid)?;
                return Err(error);
            }
        };
        let ups = match checked_gated_byte_sequence(up_labels, expected, "up label") {
            Ok(labels) => labels,
            Err(error) => {
                self.inner
                    .lock()
                    .map_err(|_| {
                        PyRuntimeError::new_err("gated Q7 multiply evaluator lock was poisoned")
                    })?
                    .burn()
                    .map_err(invalid)?;
                return Err(error);
            }
        };
        let gate_refs = gates.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let up_refs = ups.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let outputs = py
            .detach(|| {
                self.inner
                    .lock()
                    .map_err(|_| "gated Q7 multiply evaluator lock was poisoned".to_string())?
                    .evaluate_tensor(&gate_refs, &up_refs)
            })
            .map_err(invalid)?;
        PyTuple::new(py, outputs.iter().map(|output| PyBytes::new(py, output)))
    }
}
#[pymethods]
impl SiluQ7Evaluator {
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        labels: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let labels = match checked_byte_sequence(
            labels,
            self.elements,
            pllm_compiler::SILU_Q7_MAX_LABEL_BYTES,
            "label",
        ) {
            Ok(labels) => labels,
            Err(error) => {
                self.inner
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("Q7 SiLU evaluator lock was poisoned"))?
                    .burn()
                    .map_err(invalid)?;
                return Err(error);
            }
        };
        let outputs = py
            .detach(|| {
                self.inner
                    .lock()
                    .map_err(|_| "Q7 SiLU evaluator lock was poisoned".to_string())?
                    .evaluate(&labels)
            })
            .map_err(invalid)?;
        Ok(outputs
            .iter()
            .map(|output| PyBytes::new(py, output))
            .collect())
    }
}

#[pyclass(frozen, module = "pllm._native")]
struct ResolvedExperimentProfile {
    inner: pllm_compiler::ResolvedExperimentProfile,
}
#[pymethods]
impl ResolvedExperimentProfile {
    #[getter]
    fn canonical_profile<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.canonical_profile())
    }
    #[getter]
    fn configuration_digest(&self) -> String {
        self.inner.configuration_digest().to_string()
    }
    #[getter]
    fn model(&self) -> &str {
        self.inner.model()
    }
}
#[pymethods]
impl CompiledPlan {
    fn prepare_silu_q7_material(&self, py: Python<'_>) -> PyResult<GarbledSiluQ7Material> {
        let inner = py
            .detach(|| pllm_compiler::prepare_bound_silu_q7_material(&self.inner))
            .map_err(invalid)?;
        Ok(GarbledSiluQ7Material { inner })
    }

    #[getter]
    fn logical_plan<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.logical_json())
    }
    #[getter]
    fn execution_plan<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.execution_json())
    }
    #[getter]
    fn plan_lock<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.lock_json())
    }
    #[getter]
    fn region_program<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.region_program_json())
    }
    #[getter]
    fn configuration_digest(&self) -> String {
        self.inner.logical.configuration_digest.to_string()
    }
    #[getter]
    fn logical_plan_digest(&self) -> String {
        self.inner.lock.logical_plan_digest.to_string()
    }
    #[getter]
    fn execution_plan_digest(&self) -> String {
        self.inner.lock.execution_plan_digest.to_string()
    }
    #[getter]
    fn plan_lock_digest(&self) -> String {
        pllm_types::plan_lock_digest(&self.inner.lock).to_string()
    }
    #[getter]
    fn input_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.region_program.input.shape)
    }
    #[getter]
    fn output_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.region_program.output.shape)
    }
    fn prepare_silu_q7_evaluator(&self, gates: &Bound<'_, PyAny>) -> PyResult<SiluQ7Evaluator> {
        let elements = self
            .inner
            .region_program
            .input
            .shape
            .iter()
            .try_fold(1_usize, |count, dimension| {
                usize::try_from(*dimension)
                    .ok()
                    .and_then(|dimension| count.checked_mul(dimension))
            })
            .ok_or_else(|| invalid("Q7 SiLU tensor element count exceeds usize".into()))?;
        let gates = checked_byte_sequence(
            gates,
            elements,
            pllm_compiler::SILU_Q7_MAX_EVALUATOR_PAYLOAD_BYTES,
            "evaluator payload",
        )?;
        let inner = pllm_compiler::SiluQ7Evaluator::new(&self.inner, &gates).map_err(invalid)?;
        Ok(SiluQ7Evaluator {
            inner: Mutex::new(inner),
            elements,
        })
    }
    #[pyo3(signature=(weights,input,threads=1,simd=true))]
    fn execute_wrap32<'py>(
        &self,
        py: Python<'py>,
        weights: &Bound<'_, PyBytes>,
        input: &Bound<'_, PyBytes>,
        threads: usize,
        simd: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let input = codec::u32s(input.as_bytes()).map_err(invalid)?;
        let weights = weights.as_bytes();
        let output = py
            .detach(|| pllm_compiler::execute_wrap32(&self.inner, weights, &input, threads, simd))
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &bytes_u32(&output)))
    }
    #[pyo3(signature=(weights,input,benchmark_id,privacy_cohort,numeric_cohort,environment,warmups=3,repetitions=10,threads=1,simd=true))]
    #[allow(clippy::too_many_arguments)]
    fn benchmark_wrap32<'py>(
        &self,
        py: Python<'py>,
        weights: &Bound<'_, PyBytes>,
        input: &Bound<'_, PyBytes>,
        benchmark_id: String,
        privacy_cohort: String,
        numeric_cohort: String,
        environment: &Bound<'_, PyBytes>,
        warmups: u32,
        repetitions: u32,
        threads: usize,
        simd: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let input = codec::u32s(input.as_bytes()).map_err(invalid)?;
        let weights: Vec<i8> = weights
            .as_bytes()
            .iter()
            .map(|value| *value as i8)
            .collect();
        let environment = serde_json::from_slice(environment.as_bytes())
            .map_err(|error| invalid(format!("invalid benchmark environment JSON: {error}")))?;
        let options = pllm_bench::BenchmarkOptions {
            id: benchmark_id,
            privacy_cohort,
            numeric_cohort,
            warmups,
            repetitions,
            threads,
            simd,
            environment,
        };
        let report = py
            .detach(|| pllm_bench::benchmark_wrap32(&self.inner, &weights, &input, options))
            .map_err(|error| invalid(error.to_string()))?;
        let document = report
            .canonical_json()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(PyBytes::new(py, &document))
    }
}

#[pyfunction]
fn compile_plan(document: &Bound<'_, PyBytes>) -> PyResult<CompiledPlan> {
    Ok(CompiledPlan {
        inner: pllm_compiler::compile_document(document.as_bytes()).map_err(compilation_invalid)?,
    })
}

#[pyfunction]
fn silu_q7_contract(py: Python<'_>) -> PyResult<Bound<'_, PyBytes>> {
    let document = serde_json::to_vec(&pllm_compiler::silu_q7_installed_contract())
        .map_err(|error| invalid(error.to_string()))?;
    Ok(PyBytes::new(py, &document))
}

#[pyfunction]
fn resolve_experiment(document: &Bound<'_, PyBytes>) -> PyResult<ResolvedExperimentProfile> {
    Ok(ResolvedExperimentProfile {
        inner: pllm_compiler::resolve_experiment(document.as_bytes()).map_err(invalid)?,
    })
}

#[pyfunction]
fn assurance_report(py: Python<'_>) -> Bound<'_, PyBytes> {
    PyBytes::new(py, &pllm_assurance::report_bytes())
}

#[pyfunction]
fn assurance_results(py: Python<'_>) -> Bound<'_, PyBytes> {
    PyBytes::new(
        py,
        &pllm_types::canonical_bytes(&pllm_assurance::assurance_results()),
    )
}

#[pyfunction]
fn lower_model<'py>(
    py: Python<'py>,
    config: &Bound<'_, PyBytes>,
    batch: u64,
    max_input_tokens: u64,
    max_new_tokens: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let workload = pllm_models::DecoderWorkload {
        batch,
        max_input_tokens,
        max_new_tokens,
    };
    let plan = pllm_models::lower_model_json(config.as_bytes(), workload)
        .map_err(|error| invalid(error.to_string()))?;
    Ok(PyBytes::new(py, &pllm_types::canonical_bytes(&plan)))
}

#[pyfunction]
fn decoder_coverage<'py>(
    py: Python<'py>,
    plan: &Bound<'_, PyBytes>,
    profile: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let plan: pllm_models::DecoderPlan = serde_json::from_slice(plan.as_bytes())
        .map_err(|error| invalid(format!("invalid decoder model plan: {error}")))?;
    let report = pllm_compiler::decoder_coverage(&plan, profile);
    Ok(PyBytes::new(py, &pllm_types::canonical_bytes(&report)))
}

#[pyfunction]
fn decoder_runtime_schedule<'py>(
    py: Python<'py>,
    plan: &Bound<'_, PyBytes>,
    profile: &str,
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    if !matches!(
        profile,
        pllm_compiler::MASKED_LINEAR_RUNTIME_PROFILE
            | pllm_compiler::VERIFIED_MASKED_LINEAR_RUNTIME_PROFILE
    ) {
        return Err(invalid(format!(
            "unsupported decoder runtime schedule profile {profile:?}"
        )));
    }
    let plan: pllm_models::DecoderPlan = serde_json::from_slice(plan.as_bytes())
        .map_err(|error| invalid(format!("invalid decoder model plan: {error}")))?;
    let schedule = pllm_compiler::lower_decoder_runtime_schedule_for_profile(&plan, profile)
        .map_err(invalid)?;
    let digest = schedule.digest().to_string();
    Ok((
        PyBytes::new(py, &pllm_types::canonical_bytes(&schedule)),
        digest,
    ))
}

#[pyfunction(signature = (
    plan,
    mode,
    method_component_id = "pllm/r03-crt/v1",
    schedule_component_id = "pllm/scalar/v1",
    max_tensor_elements = 1
))]
fn lower_gated_multiply_q7(
    plan: &Bound<'_, PyBytes>,
    mode: &str,
    method_component_id: &str,
    schedule_component_id: &str,
    max_tensor_elements: usize,
) -> PyResult<Vec<GatedMultiplyQ7Region>> {
    let plan: pllm_models::DecoderPlan = serde_json::from_slice(plan.as_bytes())
        .map_err(|error| invalid(format!("invalid decoder model plan: {error}")))?;
    let mode = match mode {
        "prefill" => pllm_models::DecoderMode::Prefill,
        "decode" => pllm_models::DecoderMode::Decode,
        _ => return Err(invalid("decoder mode must be prefill or decode".into())),
    };
    let regions = pllm_compiler::lower_model_gated_multiply_q7_regions_with_components(
        &plan,
        mode,
        method_component_id,
        schedule_component_id,
        max_tensor_elements,
    )
    .map_err(invalid)?;
    let plan = Arc::new(plan);
    Ok(regions
        .into_iter()
        .map(|inner| GatedMultiplyQ7Region {
            inner,
            plan: Arc::clone(&plan),
        })
        .collect())
}

#[pyfunction]
fn apply_model_component<'py>(
    py: Python<'py>,
    plan: &[u8],
    component: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let plan: pllm_models::DecoderPlan = serde_json::from_slice(plan)
        .map_err(|error| invalid(format!("invalid decoder plan: {error}")))?;
    let mut component: serde_json::Value = serde_json::from_slice(component)
        .map_err(|error| invalid(format!("invalid model component: {error}")))?;
    let name = component
        .get("component")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid("model component requires component".to_owned()))?;
    if name != "pllm/kv-cache-eviction" {
        return Err(invalid(format!("unsupported model component {name:?}")));
    }
    let params = component
        .get_mut("params")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| invalid("model component requires object params".to_owned()))?;
    let implementation = params
        .remove("implementation")
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| invalid("KV-cache eviction requires implementation".to_owned()))?;
    if implementation != "pllm/mpcache/v1" {
        return Err(invalid(format!(
            "unsupported KV-cache eviction implementation {implementation:?}"
        )));
    }
    let policy: pllm_models::cache::MpcachePolicy =
        serde_json::from_value(serde_json::Value::Object(params.clone()))
            .map_err(|error| invalid(format!("invalid MPCache policy: {error}")))?;
    let optimized =
        pllm_models::cache::optimize(&plan, policy).map_err(|error| invalid(error.to_string()))?;
    Ok(PyBytes::new(py, &pllm_types::canonical_bytes(&optimized)))
}

#[pyfunction]
fn deployment_benchmark_report<'py>(
    py: Python<'py>,
    document: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    let request = serde_json::from_slice(document.as_bytes())
        .map_err(|error| invalid(format!("invalid deployment benchmark JSON: {error}")))?;
    let report =
        pllm_bench::deployment_report(request).map_err(|error| invalid(error.to_string()))?;
    let document = report
        .canonical_json()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(PyBytes::new(py, &document))
}

#[pyclass(frozen, module = "pllm._native")]
struct Executor {
    inner: kernels::Executor,
}
#[pymethods]
impl Executor {
    #[new]
    #[pyo3(signature=(threads=1,simd=true))]
    fn new(threads: usize, simd: bool) -> PyResult<Self> {
        Ok(Self {
            inner: kernels::Executor::new(threads, simd).map_err(invalid)?,
        })
    }
    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }
    #[getter]
    fn simd(&self) -> bool {
        self.inner.simd()
    }
    fn modular<'py>(
        &self,
        py: Python<'py>,
        matrix: PyRef<'_, Matrix>,
        input: &Bound<'_, PyBytes>,
        batch: usize,
        modulus: u32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let input = codec::u32s(input.as_bytes()).map_err(invalid)?;
        let m = &matrix.inner;
        let out = py
            .detach(|| m.modular(&self.inner, &input, batch, modulus))
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &bytes_u32(&out)))
    }
    fn wrap32<'py>(
        &self,
        py: Python<'py>,
        matrix: PyRef<'_, Matrix>,
        input: &Bound<'_, PyBytes>,
        batch: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let input = codec::u32s(input.as_bytes()).map_err(invalid)?;
        let m = &matrix.inner;
        let out = py
            .detach(|| m.wrap32(&self.inner, &input, batch))
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &bytes_u32(&out)))
    }
    fn wrap64<'py>(
        &self,
        py: Python<'py>,
        matrix: PyRef<'_, Matrix>,
        input: &Bound<'_, PyBytes>,
        batch: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let input = codec::u64s(input.as_bytes()).map_err(invalid)?;
        let m = &matrix.inner;
        let out = py
            .detach(|| m.wrap64(&self.inner, &input, batch))
            .map_err(invalid)?;
        py_bytes_u64(py, &out)
    }
    fn clear<'py>(
        &self,
        py: Python<'py>,
        matrix: PyRef<'_, Matrix>,
        input: &Bound<'_, PyBytes>,
        batch: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let input: Vec<i8> = input.as_bytes().iter().map(|&v| v as i8).collect();
        let m = &matrix.inner;
        let out = py
            .detach(|| m.clear(&self.inner, &input, batch))
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &bytes_i32(&out)))
    }
    fn coefficients<'py>(
        &self,
        py: Python<'py>,
        matrix: PyRef<'_, Matrix>,
        input: &Bound<'_, PyBytes>,
        columns: usize,
        modulus: u64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let input = codec::u64s(input.as_bytes()).map_err(invalid)?;
        let m = &matrix.inner;
        let out = py
            .detach(|| m.coefficients(&self.inner, &input, columns, modulus))
            .map_err(invalid)?;
        py_bytes_u64(py, &out)
    }
}
#[pyfunction]
fn pack_unsigned<'py>(
    py: Python<'py>,
    input: &Bound<'_, PyBytes>,
    width: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let values = codec::u32s(input.as_bytes()).map_err(invalid)?;
    let output = py.detach(|| codec::pack(&values, width)).map_err(invalid)?;
    Ok(PyBytes::new(py, &output))
}
#[pyfunction]
fn unpack_unsigned<'py>(
    py: Python<'py>,
    input: &Bound<'_, PyBytes>,
    width: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    // Python bytes cannot be mutated while this borrowed slice is detached.
    let source = input.as_bytes();
    let output = py
        .detach(|| codec::unpack(source, width))
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &bytes_u32(&output)))
}
#[pyfunction]
fn mask<'py>(
    py: Python<'py>,
    input: &Bound<'_, PyBytes>,
    masks: &Bound<'_, PyBytes>,
    modulus: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let x = input.as_bytes();
    let r = codec::u32s(masks.as_bytes()).map_err(invalid)?;
    let out = py.detach(|| codec::mask(x, &r, modulus)).map_err(invalid)?;
    Ok(PyBytes::new(py, &bytes_u32(&out)))
}
#[pyfunction]
fn unmask<'py>(
    py: Python<'py>,
    input: &Bound<'_, PyBytes>,
    masks: &Bound<'_, PyBytes>,
    modulus: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let x = codec::u32s(input.as_bytes()).map_err(invalid)?;
    let r = codec::u32s(masks.as_bytes()).map_err(invalid)?;
    let out = py
        .detach(|| codec::unmask(&x, &r, modulus))
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &bytes_i64(&out)))
}
#[pyfunction]
#[pyo3(signature=(input,rows,cols,bits=4,scales=None))]
fn quantize<'py>(
    py: Python<'py>,
    input: &Bound<'_, PyBytes>,
    rows: usize,
    cols: usize,
    bits: u8,
    scales: Option<&Bound<'_, PyBytes>>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let x = codec::f32s(input.as_bytes()).map_err(invalid)?;
    let scales = scales
        .map(|s| codec::f32s(s.as_bytes()))
        .transpose()
        .map_err(invalid)?;
    let (values, scales) = py
        .detach(|| codec::quantize(&x, rows, cols, bits, scales.as_deref()))
        .map_err(invalid)?;
    Ok((
        PyBytes::new(py, &values),
        PyBytes::new(py, &bytes_f32(&scales)),
    ))
}
#[pyfunction]
fn uniform_residues(py: Python<'_>, modulus: u64, count: usize) -> PyResult<Bound<'_, PyBytes>> {
    let values = py
        .detach(|| codec::random_residues(modulus, count))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &bytes_u32(&values)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prepare_freivalds(
    py: Python<'_>,
    weights: &Bound<'_, PyBytes>,
    out_features: usize,
    in_features: usize,
    inventory_rows: usize,
    root_seed: &Bound<'_, PyBytes>,
    binding: &Bound<'_, PyBytes>,
    material_id: &Bound<'_, PyBytes>,
    signed_input_bound: i64,
    signed_output_bound: i64,
    policy: PyRef<'_, PyFreivaldsPolicy>,
) -> PyResult<FreivaldsProjectionInventory> {
    let expected_weight_bytes = out_features
        .checked_mul(in_features)
        .ok_or_else(|| invalid("Freivalds weight length overflow".into()))?;
    if weights.as_bytes().len() != expected_weight_bytes
        || expected_weight_bytes > policy.resources.max_matrix_elements()
    {
        return Err(invalid("Freivalds weight dimensions exceed policy".into()));
    }
    let weight_values = signed_i8s(weights.as_bytes())?;
    let seed = exact_secret_32(root_seed.as_bytes(), "Freivalds root seed")?;
    let material_id = exact_32(material_id.as_bytes(), "Freivalds material id")?;
    let binding = bounded_copy(
        binding.as_bytes(),
        pllm_core::FREIVALDS_MAX_BINDING_BYTES,
        "Freivalds binding",
    )?;
    let native_policy = policy.policy;
    let resources = policy.resources;
    let session_id = policy.session_id;
    let inner = py
        .detach(move || {
            pllm_core::prepare_freivalds_projections(
                &weight_values,
                out_features,
                in_features,
                inventory_rows,
                &seed,
                &binding,
                session_id,
                material_id,
                signed_input_bound,
                signed_output_bound,
                native_policy,
                resources,
            )
        })
        .map_err(|error| invalid(error.to_string()))?;
    Ok(FreivaldsProjectionInventory {
        inner: Mutex::new(Some(inner)),
        resources,
        session: None,
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn import_freivalds(
    py: Python<'_>,
    payload: &Bound<'_, PyBytes>,
    authentication_tag: &Bound<'_, PyBytes>,
    root_seed: &Bound<'_, PyBytes>,
    inventory_rows: usize,
    in_features: usize,
    out_features: usize,
    binding: &Bound<'_, PyBytes>,
    material_id: &Bound<'_, PyBytes>,
    signed_input_bound: i64,
    signed_output_bound: i64,
    max_row_l1: u64,
    policy: PyRef<'_, PyFreivaldsPolicy>,
) -> PyResult<FreivaldsProjectionInventory> {
    let authentication_tag = exact_16(
        authentication_tag.as_bytes(),
        "Freivalds authentication tag",
    )?;
    let root_seed = exact_secret_32(root_seed.as_bytes(), "Freivalds root seed")?;
    let material_id = exact_32(material_id.as_bytes(), "Freivalds material id")?;
    let binding = bounded_copy(
        binding.as_bytes(),
        pllm_core::FREIVALDS_MAX_BINDING_BYTES,
        "Freivalds binding",
    )?;
    let expected_payload_bytes = inventory_rows
        .checked_mul(policy.policy.checks())
        .and_then(|value| value.checked_mul(in_features))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| invalid("Freivalds payload length overflow".into()))?;
    if payload.as_bytes().len() != expected_payload_bytes {
        return Err(invalid(format!(
            "Freivalds payload requires {expected_payload_bytes} bytes"
        )));
    }
    let payload = Zeroizing::new(bounded_copy(
        payload.as_bytes(),
        policy
            .resources
            .max_projection_elements()
            .checked_mul(4)
            .ok_or_else(|| invalid("Freivalds payload bound overflow".into()))?,
        "Freivalds payload",
    )?);
    let native_policy = policy.policy;
    let resources = policy.resources;
    let session =
        Arc::clone(policy.session.as_ref().ok_or_else(|| {
            invalid("Freivalds import requires a registered client session".into())
        })?);
    let import_session = Arc::clone(&session);
    let inner = py
        .detach(move || {
            let mut session = import_session
                .lock()
                .map_err(|_| pllm_core::FreivaldsError::Consumed)?;
            pllm_core::import_freivalds_projections(
                &payload,
                &authentication_tag,
                &root_seed,
                &mut session,
                inventory_rows,
                out_features,
                in_features,
                &binding,
                material_id,
                signed_input_bound,
                signed_output_bound,
                max_row_l1,
                native_policy,
                resources,
            )
        })
        .map_err(|error| invalid(error.to_string()))?;
    Ok(FreivaldsProjectionInventory {
        inner: Mutex::new(Some(inner)),
        resources,
        session: Some(session),
    })
}
#[pyfunction]
fn capabilities(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("api_version", 2)?;
    out.set_item("implementation", "rust")?;
    out.set_item("crate_version", env!("CARGO_PKG_VERSION"))?;
    out.set_item("avx2", kernels::has_avx2())?;
    out.set_item("neon", kernels::has_neon())?;
    out.set_item("parallel", "rayon")?;
    out.set_item("he_backend", "external-seal-tenseal")?;
    out.set_item("matrix_storage", "owned-int8")?;
    out.set_item("gpu", false)?;
    Ok(out)
}
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Matrix>()?;
    module.add_class::<CompiledPlan>()?;
    module.add_class::<ResolvedExperimentProfile>()?;
    module.add_class::<GarbledSiluQ7Material>()?;
    module.add_class::<SiluQ7Evaluator>()?;
    module.add_class::<GatedMultiplyQ7Region>()?;
    module.add_class::<GarbledGatedMultiplyQ7Material>()?;
    module.add_class::<GatedMultiplyQ7Evaluator>()?;
    module.add_class::<PyFreivaldsPolicy>()?;
    module.add_class::<FreivaldsProjectionInventory>()?;
    module.add_class::<FreivaldsVerifierHandle>()?;
    module.add_class::<Executor>()?;
    module.add_function(wrap_pyfunction!(compile_plan, module)?)?;
    module.add_function(wrap_pyfunction!(silu_q7_contract, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_experiment, module)?)?;
    module.add_function(wrap_pyfunction!(assurance_report, module)?)?;
    module.add_function(wrap_pyfunction!(assurance_results, module)?)?;
    module.add_function(wrap_pyfunction!(lower_model, module)?)?;
    module.add_function(wrap_pyfunction!(decoder_coverage, module)?)?;
    module.add_function(wrap_pyfunction!(decoder_runtime_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(lower_gated_multiply_q7, module)?)?;
    module.add_function(wrap_pyfunction!(apply_model_component, module)?)?;
    module.add_function(wrap_pyfunction!(deployment_benchmark_report, module)?)?;
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    module.add_function(wrap_pyfunction!(pack_unsigned, module)?)?;
    module.add_function(wrap_pyfunction!(unpack_unsigned, module)?)?;
    module.add_function(wrap_pyfunction!(mask, module)?)?;
    module.add_function(wrap_pyfunction!(unmask, module)?)?;
    module.add_function(wrap_pyfunction!(quantize, module)?)?;
    module.add_function(wrap_pyfunction!(uniform_residues, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_freivalds, module)?)?;
    module.add_function(wrap_pyfunction!(import_freivalds, module)?)?;
    Ok(())
}
