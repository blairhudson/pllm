use crate::invalid;
use pllm_core::paged;
use pyo3::{prelude::*, types::PyBytes};
use std::path::Path;

fn digest(value: &str) -> PyResult<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid(
            "paged artifact needs a canonical SHA-256 digest".into(),
        ));
    }
    Ok(std::array::from_fn(|i| {
        u8::from_str_radix(&value[i * 2..i * 2 + 2], 16).expect("validated hexadecimal digest")
    }))
}
#[pyfunction]
fn export_paged_weights(
    py: Python<'_>,
    path: &str,
    data: &[u8],
    rows: usize,
    cols: usize,
    page_rows: usize,
    compressed: bool,
) -> PyResult<String> {
    let hash = py
        .detach(|| paged::export(Path::new(path), data, rows, cols, page_rows, compressed))
        .map_err(invalid)?;
    Ok(pllm_types::Digest::from_sha256(hash).to_string())
}
#[pyclass(name = "PagedMatrix", module = "pllm._native")]
struct Matrix {
    inner: Option<paged::PagedMatrix>,
}
impl Matrix {
    fn get(&self) -> PyResult<&paged::PagedMatrix> {
        self.inner
            .as_ref()
            .ok_or_else(|| invalid("paged matrix is closed".into()))
    }
}
#[pymethods]
impl Matrix {
    #[new]
    fn new(
        py: Python<'_>,
        path: &str,
        artifact_digest: &str,
        threads: usize,
        simd: bool,
    ) -> PyResult<Self> {
        let expected = digest(artifact_digest)?;
        Ok(Self {
            inner: Some(
                py.detach(|| paged::PagedMatrix::open(Path::new(path), expected, threads, simd))
                    .map_err(invalid)?,
            ),
        })
    }
    #[getter]
    fn shape(&self) -> PyResult<(usize, usize)> {
        Ok(self.get()?.shape())
    }
    #[getter]
    fn artifact_bytes(&self) -> PyResult<u64> {
        Ok(self.get()?.artifact_bytes())
    }
    #[getter]
    fn metadata_bytes(&self) -> PyResult<usize> {
        Ok(self.get()?.metadata_bytes())
    }
    #[getter]
    fn transient_weight_bytes(&self) -> PyResult<usize> {
        Ok(self.get()?.transient_weight_bytes())
    }
    #[getter]
    fn weight_digest(&self) -> PyResult<String> {
        Ok(pllm_types::Digest::from_sha256(self.get()?.weight_digest()).to_string())
    }
    fn close(&mut self) {
        self.inner = None;
    }
    fn clear<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        batch: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let matrix = self.get()?;
        let result = py
            .detach(|| {
                if data.len() > 4 * 1024 * 1024 {
                    return Err("paged input exceeds workspace policy".into());
                }
                let input: Vec<i8> = data.iter().map(|&x| x as i8).collect();
                matrix
                    .clear(&input, batch)
                    .map(|v| v.into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>())
            })
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &result))
    }
    #[pyo3(signature=(data,batch,modulus=None))]
    fn modular<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        batch: usize,
        modulus: Option<u32>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let matrix = self.get()?;
        let result = py
            .detach(|| {
                if data.len() % 4 != 0 || data.len() > 16 * 1024 * 1024 {
                    return Err("paged residue input exceeds policy".into());
                }
                let input: Vec<u32> = data
                    .chunks_exact(4)
                    .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
                    .collect();
                let values = match modulus {
                    Some(q) => matrix.modular(&input, batch, q),
                    None => matrix.wrap32(&input, batch),
                }?;
                Ok(values
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>())
            })
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &result))
    }
    fn gather<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let matrix = self.get()?;
        let result = py
            .detach(|| {
                if data.len() % 8 != 0 || data.len() > 4096 * 8 {
                    return Err("paged gather frame exceeds policy".into());
                }
                let ids: Vec<u64> = data
                    .chunks_exact(8)
                    .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
                    .collect();
                matrix.gather(&ids)
            })
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &result))
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Matrix>()?;
    module.add_function(wrap_pyfunction!(export_paged_weights, module)?)
}
