use pllm_compiler::ExecutionCapsule;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(frozen, name = "ExecutionCapsule")]
struct PyExecutionCapsule {
    inner: ExecutionCapsule,
}

#[pymethods]
impl PyExecutionCapsule {
    #[staticmethod]
    fn compile<'py>(
        py: Python<'py>,
        plan: &Bound<'_, PyBytes>,
        composition: &Bound<'_, PyBytes>,
        build: &str,
        binding: &str,
    ) -> PyResult<(Bound<'py, PyBytes>, String)> {
        let (plan, composition) = (plan.as_bytes(), composition.as_bytes());
        let bytes = py
            .detach(|| ExecutionCapsule::compile(plan, composition, build, binding))
            .map_err(crate::invalid)?;
        let digest = ExecutionCapsule::digest(&bytes);
        Ok((PyBytes::new(py, &bytes), digest))
    }

    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        bytes: &Bound<'_, PyBytes>,
        expected: &str,
        build: &str,
        binding: &str,
        model: &str,
        composition: &str,
    ) -> PyResult<Self> {
        let bytes = bytes.as_bytes();
        let inner = py
            .detach(|| ExecutionCapsule::load(bytes, expected, build, binding, model, composition))
            .map_err(crate::invalid)?;
        Ok(Self { inner })
    }

    fn metadata<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.metadata())
    }
    fn step_count(&self, phase: usize) -> PyResult<usize> {
        self.inner.step_count(phase).map_err(crate::invalid)
    }
    fn step<'py>(
        &self,
        py: Python<'py>,
        phase: usize,
        index: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            self.inner.step(phase, index).map_err(crate::invalid)?,
        ))
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExecutionCapsule>()
}
