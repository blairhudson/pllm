use crate::invalid;
use pyo3::{prelude::*, types::PyBytes};

#[pyfunction]
#[pyo3(signature = (seed, context, count, bits, input=None))]
fn offset_seeded_share<'py>(
    py: Python<'py>,
    seed: &[u8],
    context: &[u8],
    count: usize,
    bits: u8,
    input: Option<&[u8]>,
) -> PyResult<Bound<'py, PyBytes>> {
    let seed: [u8; 32] = seed
        .try_into()
        .map_err(|_| invalid("seed must be 32 bytes".into()))?;
    let context: [u8; 32] = context
        .try_into()
        .map_err(|_| invalid("context must be 32 bytes".into()))?;
    let data: Vec<u8> = py
        .detach(|| {
            pllm_core::offset_transport::seeded_share(&seed, &context, count, bits, input)
                .map(|values| values.into_iter().flat_map(u32::to_le_bytes).collect())
        })
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &data))
}

#[pyfunction]
fn offset_row_bits<'py>(
    py: Python<'py>,
    weights: &[u8],
    columns: usize,
    max_input: u8,
) -> PyResult<Bound<'py, PyBytes>> {
    let result = py
        .detach(|| pllm_core::offset_transport::row_residue_bits(weights, columns, max_input))
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &result))
}

#[pyfunction]
fn offset_pack_rows<'py>(
    py: Python<'py>,
    values: &[u8],
    widths: &[u8],
    rows: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let result = py
        .detach(|| pllm_core::offset_transport::pack_row_residues(values, widths, rows))
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &result))
}

#[pyfunction]
fn offset_reconstruct_rows<'py>(
    py: Python<'py>,
    a: &[u8],
    b: &[u8],
    widths: &[u8],
    rows: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let result = py
        .detach(|| pllm_core::offset_transport::reconstruct_rows(a, b, widths, rows))
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &result))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(offset_seeded_share, module)?)?;
    module.add_function(wrap_pyfunction!(offset_row_bits, module)?)?;
    module.add_function(wrap_pyfunction!(offset_pack_rows, module)?)?;
    module.add_function(wrap_pyfunction!(offset_reconstruct_rows, module)?)
}
