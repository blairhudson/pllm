use crate::invalid;
use pllm_garble::private_pages as native;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyList},
};

#[pyclass(name = "PrivatePageServer", module = "pllm._native")]
struct Server {
    inner: native::Server,
}
#[pymethods]
impl Server {
    #[new]
    fn new(py: Python<'_>, data: &[u8], record_bytes: usize, party: u8) -> PyResult<Self> {
        Ok(Self {
            inner: py
                .detach(|| native::Server::new(data, record_bytes, party))
                .map_err(invalid)?,
        })
    }
    fn descriptor<'py>(&self, py: Python<'py>) -> (Bound<'py, PyBytes>, usize, usize) {
        let d = self.inner.descriptor();
        (PyBytes::new(py, &d.digest), d.records, d.width)
    }
    fn evaluate<'py>(&mut self, py: Python<'py>, query: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let result = py.detach(|| self.inner.evaluate(query)).map_err(invalid)?;
        Ok(PyBytes::new(py, &result))
    }
    fn cancel(&mut self, py: Python<'_>, query: &[u8]) -> PyResult<()> {
        py.detach(|| self.inner.cancel(query)).map_err(invalid)
    }
    fn evaluate_batch<'py>(
        &mut self,
        py: Python<'py>,
        queries: &Bound<'py, PyList>,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        if !(1..=64).contains(&queries.len()) {
            return Err(invalid("private page batch exceeds 64 queries".into()));
        }
        let buffers: Vec<_> = queries
            .iter()
            .map(|q| q.cast_into::<PyBytes>())
            .collect::<Result<_, _>>()?;
        let data: Vec<_> = buffers.iter().map(|q| q.as_bytes()).collect();
        let result = py
            .detach(|| self.inner.evaluate_batch(&data))
            .map_err(invalid)?;
        Ok(result.iter().map(|r| PyBytes::new(py, r)).collect())
    }
}
#[pyclass(name = "PrivatePageDecoder", module = "pllm._native")]
struct Decoder {
    inner: Option<native::Decoder>,
}
#[pymethods]
impl Decoder {
    fn decode<'py>(
        &mut self,
        py: Python<'py>,
        a: &Bound<'py, PyAny>,
        b: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let decoder = self
            .inner
            .take()
            .ok_or_else(|| invalid("private page decoder already consumed".into()))?;
        let (a, b) = (a.extract::<&[u8]>()?, b.extract::<&[u8]>()?);
        let result = py.detach(|| decoder.decode(a, b)).map_err(invalid)?;
        Ok(PyBytes::new(py, &result))
    }
    fn cancel(&mut self) {
        self.inner = None;
    }
}
#[pyfunction]
fn private_page_issue<'py>(
    py: Python<'py>,
    digest: &[u8],
    records: usize,
    record_bytes: usize,
    index: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Decoder)> {
    let digest = digest
        .try_into()
        .map_err(|_| invalid("page source digest must be 32 bytes".into()))?;
    let (a, b, decoder) = py
        .detach(|| {
            native::issue(
                native::Descriptor {
                    digest,
                    records,
                    width: record_bytes,
                },
                index,
            )
        })
        .map_err(invalid)?;
    Ok((
        PyBytes::new(py, &a),
        PyBytes::new(py, &b),
        Decoder {
            inner: Some(decoder),
        },
    ))
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Server>()?;
    module.add_class::<Decoder>()?;
    module.add_function(wrap_pyfunction!(private_page_issue, module)?)
}
