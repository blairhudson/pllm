use crate::invalid;
use pllm_core::paged;
use pllm_core::public_file;
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

#[pyclass(name = "PublicFile", module = "pllm._native")]
struct PublicFile {
    inner: Option<public_file::PublicFile>,
}
impl PublicFile {
    fn get(&self) -> PyResult<&public_file::PublicFile> {
        self.inner
            .as_ref()
            .ok_or_else(|| invalid("public snapshot is closed".into()))
    }
}
#[pymethods]
impl PublicFile {
    fn read<'py>(
        &self,
        py: Python<'py>,
        offset: u64,
        length: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let snapshot = self.get()?;
        if length == 0 || length > public_file::MAX_CHUNK_BYTES {
            return Err(invalid("public snapshot read exceeds bounds".into()));
        }
        // Fill unpublished Python bytes directly, without a second full chunk.
        PyBytes::new_with(py, length, |data| {
            py.detach(|| snapshot.read(offset, data)).map_err(invalid)
        })
    }
    fn close(&mut self) {
        self.inner = None;
    }
}

#[pyclass(name = "PublicFileWriter", module = "pllm._native")]
struct PublicFileWriter {
    inner: Option<public_file::PublicFileWriter>,
}
#[pymethods]
impl PublicFileWriter {
    #[new]
    fn new(py: Python<'_>, size: u64, content_digest: &str) -> PyResult<Self> {
        let expected = digest(content_digest)?;
        Ok(Self {
            inner: Some(
                py.detach(|| public_file::PublicFileWriter::new(size, expected))
                    .map_err(invalid)?,
            ),
        })
    }
    fn write(&mut self, py: Python<'_>, bytes: &[u8]) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| invalid("public snapshot writer is closed".into()))?;
        py.detach(|| writer.write(bytes)).map_err(invalid)
    }
    fn finish(&mut self, py: Python<'_>) -> PyResult<PublicFile> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| invalid("public snapshot writer is closed".into()))?;
        Ok(PublicFile {
            inner: Some(py.detach(|| writer.finish()).map_err(invalid)?),
        })
    }
    fn close(&mut self) {
        self.inner = None;
    }
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
    #[staticmethod]
    fn from_raw(
        py: Python<'_>,
        path: &str,
        content_digest: &str,
        rows: usize,
        cols: usize,
        executor: PyRef<'_, crate::Executor>,
    ) -> PyResult<Self> {
        let expected = digest(content_digest)?;
        let executor = std::sync::Arc::clone(&executor.inner);
        Ok(Self {
            inner: Some(
                py.detach(|| {
                    paged::PagedMatrix::from_raw_with_executor(
                        Path::new(path),
                        expected,
                        rows,
                        cols,
                        executor,
                    )
                })
                .map_err(invalid)?,
            ),
        })
    }
    #[staticmethod]
    fn from_public_file(
        py: Python<'_>,
        snapshot: PyRef<'_, PublicFile>,
        content_digest: &str,
        rows: usize,
        cols: usize,
        executor: PyRef<'_, crate::Executor>,
    ) -> PyResult<Self> {
        let expected = digest(content_digest)?;
        let executor = std::sync::Arc::clone(&executor.inner);
        let snapshot = snapshot.get()?;
        Ok(Self {
            inner: Some(
                py.detach(|| {
                    paged::PagedMatrix::from_public_file(snapshot, expected, rows, cols, executor)
                })
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
    fn digit8<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        batch: usize,
        bits: u32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let matrix = self.get()?;
        if data.len() > 16 << 20 {
            return Err(invalid("digit8 input exceeds 16 MiB".into()));
        }
        let result = py
            .detach(|| {
                let x = pllm_core::codec::u32s(data)?;
                matrix
                    .digit8(&x, batch, bits)
                    .map(|v| v.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>())
            })
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &result))
    }
    #[pyo3(signature=(data, columns=false))]
    fn gather<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        columns: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
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
                if columns {
                    matrix.gather_columns(&ids)
                } else {
                    matrix.gather(&ids)
                }
            })
            .map_err(invalid)?;
        Ok(PyBytes::new(py, &result))
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PublicFile>()?;
    module.add_class::<PublicFileWriter>()?;
    module.add_class::<Matrix>()?;
    module.add_function(wrap_pyfunction!(export_paged_weights, module)?)
}
