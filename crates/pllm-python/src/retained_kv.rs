//! Opaque client-local retained KV owner. Exported bytes are independent copies.

use pllm_core::cache_selection_reference;
use pllm_core::retained_kv::{self, RetainedKvCache};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use std::sync::Mutex;
use zeroize::Zeroizing;

fn invalid(error: &'static str) -> PyErr {
    PyValueError::new_err(error)
}

fn tensors(keys: &[u8], values: &[u8], width: usize, rows: usize) -> Result<(), &'static str> {
    if width == 0
        || keys.is_empty()
        || keys.len() != values.len()
        || keys.len() % (width * 4) != 0
        || keys.len() / (width * 4) > rows
        || keys.len() > retained_kv::MAX_PREFILL_BYTES / 2
    {
        return Err("retained KV byte tensors exceed their shape or resource bound");
    }
    Ok(())
}

fn floats(bytes: &[u8]) -> Zeroizing<Vec<f32>> {
    Zeroizing::new(
        bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect(),
    )
}

fn indices(bytes: &[u8]) -> Result<Zeroizing<Vec<usize>>, &'static str> {
    if bytes.is_empty() || bytes.len() % 4 != 0 || bytes.len() / 4 > retained_kv::MAX_ROWS {
        return Err("retained KV position frame is invalid");
    }
    Ok(Zeroizing::new(
        bytes
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()) as usize)
            .collect(),
    ))
}

#[pyclass(frozen, name = "_RetainedKvCacheF32", module = "pllm._native")]
struct Cache(Mutex<RetainedKvCache>);

#[pyfunction]
fn _cache_retain_f32<'py>(
    py: Python<'py>,
    scores: &[u8],
    keep: usize,
    recent: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    if scores.is_empty()
        || scores.len() % 4 != 0
        || scores.len() / 4 > cache_selection_reference::MAX_ROWS
    {
        return Err(invalid("retention score frame is invalid"));
    }
    let selected = py
        .detach(|| cache_selection_reference::retain(&floats(scores), keep, recent))
        .map_err(invalid)?;
    let bytes = Zeroizing::new(
        selected
            .iter()
            .flat_map(|&p| (p as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    );
    Ok(PyBytes::new(py, &bytes))
}

#[pymethods]
impl Cache {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        keys: &[u8],
        values: &[u8],
        width: usize,
        retained: &[u8],
        capacity: usize,
        maximum_position: usize,
    ) -> PyResult<Self> {
        retained_kv::storage_bytes(capacity, width).map_err(invalid)?;
        tensors(keys, values, width, retained_kv::MAX_ROWS).map_err(invalid)?;
        py.detach(|| {
            let retained = indices(retained)?;
            RetainedKvCache::from_prefill(
                &floats(keys),
                &floats(values),
                width,
                &retained,
                capacity,
                maximum_position,
            )
        })
        .map(|state| Self(Mutex::new(state)))
        .map_err(invalid)
    }

    fn append(&self, py: Python<'_>, position: usize, keys: &[u8], values: &[u8]) -> PyResult<()> {
        py.detach(|| {
            let mut state = self.0.lock().map_err(|_| "retained KV lock failed")?;
            tensors(keys, values, state.width(), state.capacity() - state.len())?;
            state.append(position, &floats(keys), &floats(values))
        })
        .map_err(invalid)
    }

    fn retain(&self, py: Python<'_>, positions: &[u8], capacity: usize) -> PyResult<()> {
        py.detach(|| {
            self.0
                .lock()
                .map_err(|_| "retained KV lock failed")?
                .retain(&indices(positions)?, capacity)
        })
        .map_err(invalid)
    }

    #[pyo3(signature = (positions=None))]
    fn export<'py>(
        &self,
        py: Python<'py>,
        positions: Option<&[u8]>,
    ) -> PyResult<(
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
    )> {
        let (keys, values, positions) = py
            .detach(|| {
                let state = self.0.lock().map_err(|_| "retained KV lock failed")?;
                let slots = match positions {
                    Some(bytes) => state.slots(&indices(bytes)?)?,
                    None => (0..state.len()).collect(),
                };
                let (keys, values, positions) = state.active()?;
                let width = state.width();
                let mut k = Zeroizing::new(Vec::with_capacity(slots.len() * width * 4));
                let mut v = Zeroizing::new(Vec::with_capacity(slots.len() * width * 4));
                let mut p = Zeroizing::new(Vec::with_capacity(slots.len() * 4));
                for slot in slots {
                    for value in &keys[slot * width..(slot + 1) * width] {
                        k.extend_from_slice(&value.to_le_bytes());
                    }
                    for value in &values[slot * width..(slot + 1) * width] {
                        v.extend_from_slice(&value.to_le_bytes());
                    }
                    p.extend_from_slice(&positions[slot].to_le_bytes());
                }
                Ok::<_, &'static str>((k, v, p))
            })
            .map_err(invalid)?;
        Ok((
            PyBytes::new(py, &keys),
            PyBytes::new(py, &values),
            PyBytes::new(py, &positions),
        ))
    }

    fn stats(&self) -> PyResult<(usize, usize, usize, usize, usize)> {
        let state = self
            .0
            .lock()
            .map_err(|_| invalid("retained KV lock failed"))?;
        Ok((
            state.len(),
            state.capacity(),
            state.width(),
            state.next_position(),
            state.allocated_bytes(),
        ))
    }

    #[pyo3(signature = (query, candidates, cluster_rows, keep_clusters, alpha=None))]
    fn clusters<'py>(
        &self,
        py: Python<'py>,
        query: &[u8],
        candidates: &[u8],
        cluster_rows: usize,
        keep_clusters: usize,
        alpha: Option<f32>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let (selected, scores) = py
            .detach(|| {
                let state = self.0.lock().map_err(|_| "retained KV lock failed")?;
                if query.len() / 4 != state.width() || query.len() % 4 != 0 {
                    return Err("retained KV cluster query shape differs");
                }
                let (keys, _, positions) = state.active()?;
                let positions =
                    Zeroizing::new(positions.iter().map(|&p| p as usize).collect::<Vec<_>>());
                cache_selection_reference::clusters_at_positions(
                    keys,
                    &floats(query),
                    &positions,
                    &indices(candidates)?,
                    cluster_rows,
                    keep_clusters,
                    alpha,
                )
            })
            .map_err(invalid)?;
        let selected = Zeroizing::new(
            selected
                .iter()
                .flat_map(|&p| (p as u32).to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let scores = Zeroizing::new(
            scores
                .iter()
                .flat_map(|&s| s.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        Ok((PyBytes::new(py, &selected), PyBytes::new(py, &scores)))
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.0
                .lock()
                .map_err(|_| "retained KV lock failed")?
                .close();
            Ok(())
        })
        .map_err(invalid)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Cache>()?;
    module.add_function(wrap_pyfunction!(_cache_retain_f32, module)?)
}
