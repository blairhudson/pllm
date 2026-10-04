use crate::invalid;
use pyo3::{prelude::*, types::PyBytes};

#[pyfunction]
fn public_bitplanes<'py>(
    py: Python<'py>,
    data: &[u8],
    inverse: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let output = py
        .detach(|| pllm_core::public_transforms::bitplanes(data, inverse))
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &output))
}

#[pyfunction]
fn public_hadamard<'py>(
    py: Python<'py>,
    data: &[u8],
    columns: usize,
    block: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let output = py
        .detach(|| pllm_core::public_transforms::hadamard(data, columns, block, seed))
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &output))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(public_entropy_encode, module)?)?;
    module.add_function(wrap_pyfunction!(public_entropy_decode, module)?)?;
    module.add_function(wrap_pyfunction!(projected_reshare_reference, module)?)?;
    module.add_function(wrap_pyfunction!(public_bitplanes, module)?)?;
    module.add_function(wrap_pyfunction!(public_hadamard, module)?)
}

#[pyfunction]
fn public_entropy_encode<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let output = py
        .detach(|| pllm_core::public_entropy::encode(data))
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &output))
}

#[pyfunction]
fn public_entropy_decode<'py>(
    py: Python<'py>,
    data: &[u8],
    expected_size: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let output = py
        .detach(|| pllm_core::public_entropy::decode(data, expected_size))
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &output))
}

#[pyfunction]
fn projected_reshare_reference<'py>(
    py: Python<'py>,
    weights: &[u8],
    a: &[u8],
    b: &[u8],
    before: bool,
) -> PyResult<(Bound<'py, PyBytes>, usize, usize)> {
    let result = py
        .detach(|| pllm_garble::projected_reshare::reference(weights, a, b, before))
        .map_err(invalid)?;
    Ok((
        PyBytes::new(py, &result.output),
        result.peer_bytes,
        result.input_share_bytes,
    ))
}
