//! Source-certified public Q18 snapshots. Checkpoint bytes arrive in bounded
//! frames; neither publication nor execution retains a resident weight matrix.
use super::*;
use pllm_core::public_file::{PublicFile, PublicFileWriter};
use pllm_core::streamed_matrix::{self, MatrixGeometry, MatrixLayout, SelectionGeometry};
use std::collections::BTreeMap;

pub struct PublicSource {
    files: BTreeMap<String, PublicFile>,
    binding: [u8; 32],
}

struct Pending {
    tensors: Vec<(String, SourceTensor)>,
    cursor: usize,
    offset: usize,
    writer: Option<PublicFileWriter>,
    hash: Sha256,
    files: BTreeMap<String, PublicFile>,
    row_sum: u64,
    row_l1: u64,
    row_square_sum: u128,
    row_squared_norm: u128,
    row_l1_squared_sum: u128,
    binding: [u8; 32],
    norms: SourceNorms,
    norm_scan: Option<source_norms::BlockScan>,
}

/// Any invalid append consumes the incomplete source and closes its files.
/// Public affine coefficients are checked against the checkpoint too.
pub struct PublicSourceImport(Option<Pending>);

impl PublicSourceImport {
    pub fn new(manifest: &Manifest) -> Result<Self, String> {
        if manifest.weights != WeightMode::Public {
            return Err("public source requires the public-weight contract".into());
        }
        let shapes = Program::source_shapes_with_policy(
            &manifest.config,
            manifest.rows,
            manifest.outputs,
            manifest.weights,
            manifest.execution,
        )?;
        if shapes.len() != manifest.tensors.len()
            || manifest.tensors.values().any(|t| {
                (t.row_l1.is_some()
                    || t.row_squared_norm.is_some()
                    || t.row_l1_squared_sum.is_some())
                    && t.shape.len() != 2
            })
            || shapes
                .iter()
                .any(|(key, shape)| manifest.tensors.get(key).is_none_or(|t| &t.shape != shape))
        {
            return Err("public source shape commitments differ".into());
        }
        if !manifest.norms.is_empty() && manifest.execution.range_policy != RangePolicy::Checked {
            return Err("public source norms require checked execution".into());
        }
        let plan = program::lower_with_weights(
            &manifest.config,
            manifest.rows,
            manifest.outputs,
            manifest.weights,
        )?;
        source_norms::validate_set(&plan.prefill, &shapes, &manifest.norms)
            .map_err(|e| e.to_string())?;
        source_norms::import_envelope(&manifest.norms).map_err(|e| e.to_string())?;
        Ok(Self(Some(Pending {
            tensors: manifest
                .tensors
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            cursor: 0,
            offset: 0,
            writer: None,
            hash: Sha256::new(),
            files: BTreeMap::new(),
            row_sum: 0,
            row_l1: 0,
            row_square_sum: 0,
            row_squared_norm: 0,
            row_l1_squared_sum: 0,
            binding: program::source_binding_with_norms(&manifest.tensors, &manifest.norms),
            norms: manifest.norms.clone(),
            norm_scan: None,
        })))
    }

    pub fn append(&mut self, key: &str, offset: usize, bytes: &[u8]) -> Result<(), String> {
        let mut pending = self.0.take().ok_or("public source importer is retired")?;
        let (name, tensor) = pending
            .tensors
            .get(pending.cursor)
            .ok_or("public source import is complete")?;
        let count = tensor.shape.iter().product::<usize>();
        if key != name
            || offset != pending.offset
            || bytes.is_empty()
            || bytes.len() > 16384
            || bytes.len() % 4 != 0
            || offset
                .checked_add(bytes.len() / 4)
                .is_none_or(|end| end > count)
        {
            return Err("invalid ordered public source frame".into());
        }
        if offset == 0 && tensor.public_q18.is_none() {
            pending.writer = Some(PublicFileWriter::new(count as u64 * 8, tensor.digest)?);
        }
        if offset == 0 {
            pending.norm_scan = pending
                .norms
                .get(key)
                .map(|norms| {
                    let (rows, columns, block_rows) = norms.geometry();
                    source_norms::BlockScan::new(rows, columns, block_rows)
                })
                .transpose()
                .map_err(|e| e.to_string())?;
        }
        let mut encoded = vec_zero::<u8>(bytes.len() * 2).map_err(|e| e.to_string())?;
        for (i, (raw, out)) in bytes
            .chunks_exact(4)
            .zip(encoded.chunks_exact_mut(8))
            .enumerate()
        {
            let value = super::source::encode_q18(raw)?;
            if let Some(scan) = &mut pending.norm_scan {
                scan.append(offset + i, std::slice::from_ref(&value))
                    .map_err(|e| e.to_string())?;
            }
            if value < tensor.lower
                || value > tensor.upper
                || tensor
                    .public_q18
                    .as_ref()
                    .is_some_and(|values| values.get(offset + i) != Some(&value))
            {
                return Err("public source coefficient/range differs".into());
            }
            out.copy_from_slice(&value.to_le_bytes());
            if tensor.row_l1.is_some()
                || tensor.row_squared_norm.is_some()
                || tensor.row_l1_squared_sum.is_some()
            {
                pending.row_sum = pending
                    .row_sum
                    .checked_add(value.unsigned_abs())
                    .ok_or("public row bound overflow")?;
                pending.row_square_sum = pending
                    .row_square_sum
                    .checked_add(u128::from(value.unsigned_abs()).pow(2))
                    .ok_or("public row square bound overflow")?;
                if (offset + i + 1) % tensor.shape[1] == 0 {
                    pending.row_l1 = pending.row_l1.max(pending.row_sum);
                    pending.row_l1_squared_sum = pending
                        .row_l1_squared_sum
                        .checked_add(u128::from(pending.row_sum).pow(2))
                        .ok_or("public row energy overflow")?;
                    pending.row_sum = 0;
                    pending.row_squared_norm = pending.row_squared_norm.max(pending.row_square_sum);
                    pending.row_square_sum = 0;
                }
            }
        }
        pending.hash.update(&encoded);
        if let Some(writer) = &mut pending.writer {
            writer.write(&encoded)?;
        }
        pending.offset += bytes.len() / 4;
        if pending.offset == count {
            if let Some(scan) = pending.norm_scan.take() {
                let actual = scan.finish().map_err(|e| e.to_string())?;
                if pending.norms.get(key) != Some(&actual) {
                    return Err("public source block norm certificate differs".into());
                }
            }
            if tensor.row_l1.is_some_and(|sum| sum != pending.row_l1)
                || tensor
                    .row_squared_norm
                    .is_some_and(|sum| sum != pending.row_squared_norm)
                || tensor
                    .row_l1_squared_sum
                    .is_some_and(|sum| sum != pending.row_l1_squared_sum)
                || pending.row_sum != 0
                || pending.row_square_sum != 0
            {
                return Err("public source row bound differs".into());
            }
            if <[u8; 32]>::from(std::mem::take(&mut pending.hash).finalize()) != tensor.digest {
                return Err("public source digest differs".into());
            }
            if let Some(writer) = pending.writer.take() {
                pending.files.insert(name.clone(), writer.finish()?);
            }
            pending.cursor += 1;
            pending.offset = 0;
            pending.row_l1 = 0;
            pending.row_squared_norm = 0;
            pending.row_l1_squared_sum = 0;
        }
        self.0 = Some(pending);
        Ok(())
    }

    pub fn finish(mut self) -> Result<PublicSource, String> {
        let pending = self.0.take().ok_or("public source importer is retired")?;
        if pending.cursor != pending.tensors.len()
            || pending.offset != 0
            || pending.writer.is_some()
        {
            return Err("incomplete public source import".into());
        }
        Ok(PublicSource {
            files: pending.files,
            binding: pending.binding,
        })
    }
}

impl PublicSource {
    #[cfg(test)]
    pub(super) fn test_values(
        program: &Program,
        values: &BTreeMap<String, Vec<i64>>,
    ) -> Option<Self> {
        let mut files = BTreeMap::new();
        for node in &program.nodes[..program.inputs] {
            if let Code::PublicInput { key, digest, .. } = &node.code {
                let bytes: Vec<u8> = values[key].iter().flat_map(|n| n.to_le_bytes()).collect();
                let mut writer = PublicFileWriter::new(bytes.len() as u64, *digest).unwrap();
                for chunk in bytes.chunks(65536) {
                    writer.write(chunk).unwrap();
                }
                files.insert(key.clone(), writer.finish().unwrap());
            }
        }
        (!files.is_empty()).then_some(Self {
            files,
            binding: program.source_binding,
        })
    }

    pub(super) fn check(&self, program: &Program) -> Result<(), Error> {
        if self.binding != program.source_binding {
            return Err(Error::Binding);
        }
        let mut count = 0;
        for node in &program.nodes[..program.inputs] {
            if let Code::PublicInput { key, digest, .. } = &node.code {
                let file = self.files.get(key).ok_or(Error::Binding)?;
                if file.digest() != *digest || file.size() != node.lanes as u64 * 8 {
                    return Err(Error::Binding);
                }
                count += 1;
            }
        }
        if count != self.files.len() || count == 0 {
            return Err(Error::Binding);
        }
        Ok(())
    }

    pub(super) fn file(&self, key: &str) -> Result<&PublicFile, Error> {
        self.files.get(key).ok_or(Error::Binding)
    }
}

/// Complete ring64 products, with a fixed 64 KiB public read buffer. Both
/// replicated components traverse every weight in the same public order.
/// One later tensor rescale acts on the complete sum, never on disk chunks.
pub(super) fn multiply(
    file: &PublicFile,
    input: &Secret,
    rows: usize,
    inner: usize,
    columns: usize,
    transposed: bool,
) -> Result<Secret, Error> {
    let layout = if transposed {
        MatrixLayout::InputOutput
    } else {
        MatrixLayout::OutputInput
    };
    let geometry = MatrixGeometry::new(rows, inner, columns, layout).map_err(stream_error)?;
    let inputs = [input.first.as_slice(), input.second.as_slice()];
    geometry
        .validate_inputs(file, &inputs)
        .map_err(stream_error)?;
    let mut out = Secret::new(geometry.output_elements())?;
    streamed_matrix::multiply(
        file,
        &inputs,
        &mut [out.first.as_mut_slice(), out.second.as_mut_slice()],
        geometry,
    )
    .map_err(stream_error)?;
    Ok(out)
}

/// Add a public contiguous vocabulary range to complete ring64 accumulators.
/// Selectors are row-major [rows, count]; the file remains [vocabulary, width].
/// Every public coefficient is read once and reused across both components of
/// every row. No rescale, secret-dependent seek, or resident weight copy occurs.
#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_rows(
    file: &PublicFile,
    selectors: &Secret,
    output: &mut Secret,
    rows: usize,
    vocabulary: usize,
    width: usize,
    start: usize,
    count: usize,
) -> Result<(), Error> {
    let geometry =
        SelectionGeometry::new(rows, vocabulary, width, start, count).map_err(stream_error)?;
    streamed_matrix::accumulate_rows(
        file,
        &[selectors.first.as_slice(), selectors.second.as_slice()],
        &mut [output.first.as_mut_slice(), output.second.as_mut_slice()],
        geometry,
    )
    .map_err(stream_error)
}

fn stream_error(error: streamed_matrix::Error) -> Error {
    match error {
        streamed_matrix::Error::Bounds => Error::Bounds,
        streamed_matrix::Error::Read => Error::Binding,
    }
}
