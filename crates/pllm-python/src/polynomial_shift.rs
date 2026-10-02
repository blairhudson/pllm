use crate::invalid;
use pyo3::prelude::*;

#[pyfunction]
fn public_polynomial_shift_witness(
    ring_bits: u8,
    linear: u64,
    masked_gate: u64,
    masked_up: u64,
    coefficient_gu: u64,
    coefficient_g2: u64,
) -> PyResult<String> {
    let witness = pllm_assurance::polynomial_shift::evaluate(
        ring_bits,
        linear,
        masked_gate,
        masked_up,
        coefficient_gu,
        coefficient_g2,
    )
    .map_err(invalid)?;
    serde_json::to_string(&witness).map_err(|e| invalid(e.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(public_polynomial_shift_witness, module)?)
}
