//! Source-certified public Q18 snapshots. Checkpoint bytes arrive in bounded
//! frames; neither publication nor execution retains a resident weight matrix.
use super::*;
use pllm_core::public_file::{PublicFile, PublicFileWriter};
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
}

/// Any invalid append consumes the incomplete source and closes its files.
/// Public affine coefficients are checked against the checkpoint too.
pub struct PublicSourceImport(Option<Pending>);

impl PublicSourceImport {
    pub fn new(manifest: &Manifest) -> Result<Self, String> {
        if manifest.weights != WeightMode::Public {
            return Err("public source requires the public-weight contract".into());
        }
        let shapes = Program::source_shapes_with_weights(
            &manifest.config,
            manifest.rows,
            manifest.outputs,
            manifest.weights,
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
            binding: program::source_binding(&manifest.tensors),
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
        let mut encoded = vec_zero::<u8>(bytes.len() * 2).map_err(|e| e.to_string())?;
        for (i, (raw, out)) in bytes
            .chunks_exact(4)
            .zip(encoded.chunks_exact_mut(8))
            .enumerate()
        {
            let value = super::source::encode_q18(raw)?;
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
    if rows == 0
        || inner == 0
        || columns == 0
        || rows.checked_mul(inner) != Some(input.first.len())
        || input.second.len() != input.first.len()
        || inner
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(8))
            .map(|n| n as u64)
            != Some(file.size())
        || (rows as u64)
            .checked_mul(inner as u64)
            .and_then(|n| n.checked_mul(columns as u64))
            .is_none_or(|n| n > 1 << 32)
    {
        return Err(Error::Bounds);
    }
    let mut out = Secret::new(rows.checked_mul(columns).ok_or(Error::Bounds)?)?;
    let mut buffer = [0u8; 65536];
    let mut offset = 0u64;
    while offset < file.size() {
        let n = buffer.len().min((file.size() - offset) as usize);
        file.read(offset, &mut buffer[..n])
            .map_err(|_| Error::Binding)?;
        for (i, raw) in buffer[..n].chunks_exact(8).enumerate() {
            let index = offset as usize / 8 + i;
            let (k, c) = if transposed {
                (index / columns, index % columns)
            } else {
                (index % inner, index / inner)
            };
            let weight = u64::from_le_bytes(raw.try_into().unwrap());
            for row in 0..rows {
                let x = row * inner + k;
                let y = row * columns + c;
                out.first[y] = out.first[y].wrapping_add(input.first[x].wrapping_mul(weight));
                out.second[y] = out.second[y].wrapping_add(input.second[x].wrapping_mul(weight));
            }
        }
        offset += n as u64;
    }
    Ok(out)
}
