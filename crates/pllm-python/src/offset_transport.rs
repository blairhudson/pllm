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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(offset_seeded_share, module)?)
}
