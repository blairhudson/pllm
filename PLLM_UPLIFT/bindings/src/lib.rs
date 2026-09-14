use pyo3::{prelude::*,exceptions::PyValueError};
#[pyfunction]
#[pyo3(signature=(dim=64,lanes=18,repeats=7,bits=24,tile=32))]
fn benchmark_json(py:Python<'_>,dim:usize,lanes:usize,repeats:usize,bits:u8,tile:usize)->PyResult<String>{
 py.detach(move||pllm_bench::benchmark(dim,lanes,repeats,bits,tile)).map_err(PyValueError::new_err)
}
#[pyfunction]
fn assurance_json(py:Python<'_>)->PyResult<String>{py.detach(pllm_assurance::json).map_err(PyValueError::new_err)}
#[pymodule]
fn _native(m:&Bound<'_,PyModule>)->PyResult<()> {
 m.add_function(wrap_pyfunction!(benchmark_json,m)?)?;
 m.add_function(wrap_pyfunction!(assurance_json,m)?)?;
 Ok(())
}
