//! Python binding for the native integer core. Immutable byte buffers avoid
//! dangling NumPy borrows and retain the stable Python ABI. Matrices are copied
//! once at compilation. Arithmetic runs without the Python interpreter lock.
use pllm_core::{codec, kernels};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

fn invalid(error: String) -> PyErr {
    PyValueError::new_err(error)
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
fn uniform_residues<'py>(
    py: Python<'py>,
    modulus: u64,
    count: usize,
) -> PyResult<Bound<'py, PyBytes>> {
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
    module.add_class::<Executor>()?;
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    module.add_function(wrap_pyfunction!(pack_unsigned, module)?)?;
    module.add_function(wrap_pyfunction!(unpack_unsigned, module)?)?;
    module.add_function(wrap_pyfunction!(mask, module)?)?;
    module.add_function(wrap_pyfunction!(unmask, module)?)?;
    module.add_function(wrap_pyfunction!(quantize, module)?)?;
    module.add_function(wrap_pyfunction!(uniform_residues, module)?)?;
    Ok(())
}
