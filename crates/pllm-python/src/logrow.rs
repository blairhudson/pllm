//! Narrow, one-use binding for the in-process LogRow Q7 tensor reference.
//! This has no online provider transport or executable Experiment selection.

use crate::{invalid, CompactQ7Reference};
use pllm_compiler::{
    prepare_bound_logrow_q7_session, prepare_bound_logrow_q7_tensor, BoundLogRowQ7Session,
    BoundLogRowQ7TensorMaterial, ExperimentalLogRowQ7TensorPolicy,
};
use pllm_core::{
    compact::{compact_q7_from_f32, compact_q7_to_f32},
    CompactQ7Profile,
};
use pllm_models::{DecoderMode, DecoderPlan};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyModule};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

const MAX_PLAN_BYTES: usize = 16 * 1024 * 1024;
const MAX_FLOAT32_TENSOR_BYTES: usize = (64 * 1024 * 1024 / 2144) * 4;

#[pyclass(name = "LogRowQ7TensorReference", frozen, module = "pllm._native")]
pub(crate) struct LogRowQ7TensorReference {
    material: Mutex<Option<BoundLogRowQ7TensorMaterial>>,
    plan: Arc<DecoderPlan>,
    profile: Arc<CompactQ7Profile>,
    mode: DecoderMode,
    operation_id: String,
    policy: ExperimentalLogRowQ7TensorPolicy,
    elements: usize,
    evaluator_material_bytes: usize,
    binding_digest: String,
}

#[pyclass(name = "LogRowQ7SessionReference", frozen, module = "pllm._native")]
pub(crate) struct LogRowQ7SessionReference {
    material: Mutex<BoundLogRowQ7Session>,
    decoder_bound: AtomicBool,
    issuance_digest: String,
    estimate_bytes: Vec<u8>,
}

#[pymethods]
impl LogRowQ7SessionReference {
    fn bind_decoder(&self, plan_digest: &str, max_decode_steps: u64) -> PyResult<()> {
        let material = self
            .material
            .lock()
            .map_err(|_| PyRuntimeError::new_err("LogRow Q7 session lock was poisoned"))?;
        if material.estimate().plan_digest.to_string() != plan_digest
            || material.estimate().max_decode_steps != max_decode_steps
        {
            return Err(invalid(
                "LogRow Q7 decoder binding differs from its plan".into(),
            ));
        }
        if self
            .decoder_bound
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(invalid(
                "LogRow Q7 session is already bound to a decoder".into(),
            ));
        }
        Ok(())
    }

    #[getter]
    fn issuance_digest(&self) -> &str {
        &self.issuance_digest
    }

    #[getter]
    fn remaining_tensors(&self) -> PyResult<usize> {
        Ok(self
            .material
            .lock()
            .map_err(|_| PyRuntimeError::new_err("LogRow Q7 session lock was poisoned"))?
            .remaining_tensors())
    }

    fn estimate<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.estimate_bytes)
    }

    fn abort(&self) -> PyResult<()> {
        self.material
            .lock()
            .map_err(|_| PyRuntimeError::new_err("LogRow Q7 session lock was poisoned"))?
            .abort();
        Ok(())
    }

    fn evaluate_float32<'py>(
        &self,
        py: Python<'py>,
        mode: &Bound<'_, PyAny>,
        decode_step: &Bound<'_, PyAny>,
        operation_id: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        // Invalid Python types and lengths burn all pending tensors, too.
        let parsed = (|| {
            let mode = match mode.extract::<&str>()? {
                "prefill" => DecoderMode::Prefill,
                "decode" => DecoderMode::Decode,
                _ => return Err(invalid("LogRow Q7 mode must be prefill or decode".into())),
            };
            let decode_step = decode_step.extract::<u64>()?;
            let operation_id = operation_id.extract::<String>()?;
            let bytes = values
                .cast::<PyBytes>()
                .map_err(|_| invalid("LogRow Q7 float32 session input must be bytes".into()))?
                .as_bytes();
            if bytes.len() > MAX_FLOAT32_TENSOR_BYTES || bytes.len() % 4 != 0 {
                return Err(invalid(
                    "LogRow Q7 float32 session input is not bounded f32".into(),
                ));
            }
            let floats = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok((mode, decode_step, operation_id, floats))
        })();
        let (mode, decode_step, operation_id, floats) = match parsed {
            Ok(parsed) => parsed,
            Err(error) => {
                self.abort()?;
                return Err(error);
            }
        };
        let result = py
            .detach(|| {
                self.material
                    .lock()
                    .map_err(|_| "LogRow Q7 session lock was poisoned".to_owned())?
                    .evaluate_float32(mode, decode_step, &operation_id, &floats)
            })
            .map_err(invalid)?;
        let mut output = Vec::with_capacity(result.len() * 4);
        for value in result {
            output.extend_from_slice(&value.to_le_bytes());
        }
        Ok(PyBytes::new(py, &output))
    }
}

impl LogRowQ7TensorReference {
    fn claim(&self) -> PyResult<BoundLogRowQ7TensorMaterial> {
        self.material
            .lock()
            .map_err(|_| PyRuntimeError::new_err("LogRow Q7 material lock was poisoned"))?
            .take()
            .ok_or_else(|| invalid("LogRow Q7 tensor material was already consumed".into()))
    }

    fn run(
        &self,
        py: Python<'_>,
        material: BoundLogRowQ7TensorMaterial,
        values: Vec<i16>,
    ) -> PyResult<Vec<i16>> {
        py.detach(|| {
            let (evaluation, decoder) = material.encode(&values)?;
            let outputs = evaluation.evaluate(
                &self.plan,
                self.mode,
                &self.operation_id,
                &self.profile,
                &self.policy,
            )?;
            decoder.decode(outputs)
        })
        .map_err(invalid)
    }
}

#[pymethods]
impl LogRowQ7TensorReference {
    #[getter]
    fn elements(&self) -> usize {
        self.elements
    }

    #[getter]
    fn evaluator_material_bytes(&self) -> usize {
        self.evaluator_material_bytes
    }

    #[getter]
    fn binding_digest(&self) -> &str {
        &self.binding_digest
    }

    /// Little-endian signed i16 Q7 values, one per semantic tensor element.
    /// Claim material before checking the Python input so malformed calls burn it.
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let material = self.claim()?;
        let bytes = values
            .cast::<PyBytes>()
            .map_err(|_| invalid("LogRow Q7 tensor input must be bytes".into()))?
            .as_bytes();
        if bytes.len() != self.elements * 2 {
            return Err(invalid(
                "LogRow Q7 input must match the exact tensor shape".into(),
            ));
        }
        let input = bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        let result = self.run(py, material, input)?;
        let mut encoded = Vec::with_capacity(result.len() * 2);
        for value in result {
            encoded.extend_from_slice(&value.to_le_bytes());
        }
        Ok(PyBytes::new(py, &encoded))
    }

    /// Strict float32 bridge; private, data-dependent scales and saturation
    /// are forbidden. Out-of-domain input burns the entire one-use tensor.
    fn evaluate_float32<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let material = self.claim()?;
        let bytes = values
            .cast::<PyBytes>()
            .map_err(|_| invalid("LogRow float32 tensor input must be bytes".into()))?
            .as_bytes();
        if bytes.len() != self.elements * 4 {
            return Err(invalid(
                "LogRow float32 input must match the exact tensor shape".into(),
            ));
        }
        let input = bytes
            .chunks_exact(4)
            .map(|chunk| {
                compact_q7_from_f32(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .map_err(|error| invalid(error.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let result = self.run(py, material, input)?;
        let mut encoded = Vec::with_capacity(result.len() * 4);
        for value in result {
            encoded.extend_from_slice(&compact_q7_to_f32(value).to_le_bytes());
        }
        Ok(PyBytes::new(py, &encoded))
    }
}

#[pyfunction]
fn prepare_logrow_q7_tensor_reference(
    py: Python<'_>,
    plan: &Bound<'_, PyBytes>,
    mode: &str,
    operation_id: &str,
    profile: PyRef<'_, CompactQ7Reference>,
    max_elements: usize,
    max_evaluator_material_bytes: usize,
) -> PyResult<LogRowQ7TensorReference> {
    if plan.as_bytes().len() > MAX_PLAN_BYTES {
        return Err(invalid("LogRow Q7 decoder plan exceeds 16 MiB".into()));
    }
    let plan: DecoderPlan = serde_json::from_slice(plan.as_bytes())
        .map_err(|error| invalid(format!("invalid LogRow Q7 decoder plan: {error}")))?;
    let mode = match mode {
        "prefill" => DecoderMode::Prefill,
        "decode" => DecoderMode::Decode,
        _ => return Err(invalid("LogRow Q7 mode must be prefill or decode".into())),
    };
    let policy = ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(
        max_elements,
        max_evaluator_material_bytes,
    )
    .map_err(invalid)?;
    let profile = Arc::new(profile.inner.clone());
    let plan = Arc::new(plan);
    let material = py
        .detach(|| prepare_bound_logrow_q7_tensor(&plan, mode, operation_id, &profile, &policy))
        .map_err(invalid)?;
    Ok(LogRowQ7TensorReference {
        elements: material.elements(),
        evaluator_material_bytes: material.evaluator_material_bytes(),
        binding_digest: material.binding_digest().to_string(),
        material: Mutex::new(Some(material)),
        plan,
        profile,
        mode,
        operation_id: operation_id.to_owned(),
        policy,
    })
}

#[pyfunction]
fn estimate_logrow_q7_session_reference<'py>(
    py: Python<'py>,
    plan: &Bound<'_, PyBytes>,
    profile: PyRef<'_, CompactQ7Reference>,
    max_elements: usize,
    max_evaluator_material_bytes: usize,
    max_decode_steps: u64,
    max_session_evaluator_material_bytes: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    if plan.as_bytes().len() > MAX_PLAN_BYTES {
        return Err(invalid("LogRow Q7 decoder plan exceeds 16 MiB".into()));
    }
    let plan: DecoderPlan = serde_json::from_slice(plan.as_bytes())
        .map_err(|error| invalid(format!("invalid LogRow Q7 decoder plan: {error}")))?;
    let policy = ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(
        max_elements,
        max_evaluator_material_bytes,
    )
    .map_err(invalid)?;
    let profile = profile.inner.clone();
    let estimate = py
        .detach(|| {
            pllm_compiler::estimate_bound_logrow_q7_session(
                &plan,
                &profile,
                &policy,
                max_decode_steps,
                max_session_evaluator_material_bytes,
            )
        })
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &pllm_types::canonical_bytes(&estimate)))
}

#[pyfunction]
fn prepare_logrow_q7_session_reference(
    py: Python<'_>,
    plan: &Bound<'_, PyBytes>,
    profile: PyRef<'_, CompactQ7Reference>,
    max_elements: usize,
    max_evaluator_material_bytes: usize,
    max_decode_steps: u64,
    max_session_evaluator_material_bytes: usize,
) -> PyResult<LogRowQ7SessionReference> {
    if plan.as_bytes().len() > MAX_PLAN_BYTES {
        return Err(invalid("LogRow Q7 decoder plan exceeds 16 MiB".into()));
    }
    let plan: DecoderPlan = serde_json::from_slice(plan.as_bytes())
        .map_err(|error| invalid(format!("invalid LogRow Q7 decoder plan: {error}")))?;
    let policy = ExperimentalLogRowQ7TensorPolicy::acknowledge_unreviewed_public_profile(
        max_elements,
        max_evaluator_material_bytes,
    )
    .map_err(invalid)?;
    let profile = profile.inner.clone();
    let material = py
        .detach(|| {
            prepare_bound_logrow_q7_session(
                &plan,
                &profile,
                &policy,
                max_decode_steps,
                max_session_evaluator_material_bytes,
            )
        })
        .map_err(invalid)?;
    let estimate_bytes = pllm_types::canonical_bytes(material.estimate());
    let issuance_digest = material.issuance_digest().to_string();
    Ok(LogRowQ7SessionReference {
        material: Mutex::new(material),
        decoder_bound: AtomicBool::new(false),
        issuance_digest,
        estimate_bytes,
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LogRowQ7TensorReference>()?;
    module.add_class::<LogRowQ7SessionReference>()?;
    module.add_function(wrap_pyfunction!(
        prepare_logrow_q7_tensor_reference,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        estimate_logrow_q7_session_reference,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        prepare_logrow_q7_session_reference,
        module
    )?)?;
    Ok(())
}
