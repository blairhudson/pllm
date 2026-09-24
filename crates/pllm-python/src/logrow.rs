//! Narrow, one-use binding for the in-process LogRow Q7 tensor reference.
//! This has no online provider transport or executable Experiment selection.

use crate::{invalid, CompactQ7Reference};
use pllm_compiler::{
    prepare_bound_logrow_q7_tensor, BoundLogRowQ7TensorMaterial, ExperimentalLogRowQ7TensorPolicy,
};
use pllm_core::CompactQ7Profile;
use pllm_models::{DecoderMode, DecoderPlan};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyModule};
use std::sync::{Arc, Mutex};

const MAX_PLAN_BYTES: usize = 16 * 1024 * 1024;

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
        let material = self
            .material
            .lock()
            .map_err(|_| PyRuntimeError::new_err("LogRow Q7 material lock was poisoned"))?
            .take()
            .ok_or_else(|| invalid("LogRow Q7 tensor material was already consumed".into()))?;
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
        let result = py
            .detach(|| {
                let (evaluation, decoder) = material.encode(&input)?;
                let outputs = evaluation.evaluate(
                    &self.plan,
                    self.mode,
                    &self.operation_id,
                    &self.profile,
                    &self.policy,
                )?;
                decoder.decode(outputs)
            })
            .map_err(invalid)?;
        let mut encoded = Vec::with_capacity(result.len() * 2);
        for value in result {
            encoded.extend_from_slice(&value.to_le_bytes());
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LogRowQ7TensorReference>()?;
    module.add_function(wrap_pyfunction!(
        prepare_logrow_q7_tensor_reference,
        module
    )?)?;
    Ok(())
}
