//! Python binding for the native integer core. Immutable byte buffers avoid
//! dangling NumPy borrows and retain the stable Python ABI. Matrices are copied
//! once at compilation. Arithmetic runs without the Python interpreter lock.
use pllm_core::{codec, kernels};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PySequence, PyTuple};
use std::sync::Mutex;

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
    }
    (0..count)
        .map(|index| {
            sequence
                .get_item(index)?
                .cast::<PyBytes>()
                .map(|bytes| bytes.as_bytes().to_vec())
                .map_err(Into::into)
        })
        .collect()
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
    let policy: pllm_method_mpcache::MpcachePolicy =
        serde_json::from_value(serde_json::Value::Object(params.clone()))
            .map_err(|error| invalid(format!("invalid MPCache policy: {error}")))?;
    let optimized =
        pllm_method_mpcache::optimize(&plan, policy).map_err(|error| invalid(error.to_string()))?;
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
    module.add_class::<Executor>()?;
    module.add_function(wrap_pyfunction!(compile_plan, module)?)?;
    module.add_function(wrap_pyfunction!(silu_q7_contract, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_experiment, module)?)?;
    module.add_function(wrap_pyfunction!(assurance_report, module)?)?;
    module.add_function(wrap_pyfunction!(assurance_results, module)?)?;
    module.add_function(wrap_pyfunction!(lower_model, module)?)?;
    module.add_function(wrap_pyfunction!(decoder_coverage, module)?)?;
    module.add_function(wrap_pyfunction!(apply_model_component, module)?)?;
    module.add_function(wrap_pyfunction!(deployment_benchmark_report, module)?)?;
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    module.add_function(wrap_pyfunction!(pack_unsigned, module)?)?;
    module.add_function(wrap_pyfunction!(unpack_unsigned, module)?)?;
    module.add_function(wrap_pyfunction!(mask, module)?)?;
    module.add_function(wrap_pyfunction!(unmask, module)?)?;
    module.add_function(wrap_pyfunction!(quantize, module)?)?;
    module.add_function(wrap_pyfunction!(uniform_residues, module)?)?;
    Ok(())
}
