//! Source-certified public Q18 snapshots. Checkpoint bytes arrive in bounded
//! frames; neither publication nor execution retains a resident weight matrix.
use super::source_registry::{
    add_bytes, array_bytes, binding_workspace_bytes, tree_entry_bytes, SourceConstructionQuote,
    SourceRegistry,
};
use super::*;
use pllm_core::public_file::{PublicFile, PublicFileWriter};
use pllm_core::streamed_matrix::{self, MatrixGeometry, MatrixLayout, SelectionGeometry};
use std::collections::BTreeMap;
use std::mem::size_of;

#[cfg(test)]
#[path = "source_registry_tests.rs"]
mod registry_tests;

#[cfg(test)]
#[path = "source_construction_tests.rs"]
mod construction_tests;

#[cfg(test)]
thread_local! {
    static CHECKED_IMPORT_FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

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
    /// Allocation-free quote of the checked import's complete native owner
    /// overlap. Source/norm maps remain borrowed baseline, separately inventoried
    /// by their ACTUAL capacities. The cloned destination is bounded by those
    /// capacities, not by file lengths. Keep the original norm import envelope
    /// in full; no sharing or retired-record refund lowers an existing charge.
    ///
    /// Includes retained file handles/tree keys, maximum scan/output overlap,
    /// 32 KiB encoded frame, binding SHA states and returned importer/source slots.
    /// Incoming frame backing, existing plans/authenticated files and Manifest
    /// inline/config/schema belong in retained_baseline_bytes. Kernel file cache,
    /// tempfile/OS internals and allocator bookkeeping are not native payload.
    pub(super) fn construction_quote(
        registry: &SourceRegistry<'_>,
        retained_baseline_bytes: usize,
    ) -> Result<SourceConstructionQuote, Error> {
        let sources = registry.source_records();
        let norms = registry.source_norms();
        let envelope = source_norms::import_envelope(norms)?;
        let mut owned = array_bytes::<(String, SourceTensor)>(sources.len())?;
        for (key, tensor) in sources {
            owned = add_bytes(owned, key.capacity())?;
            owned = add_bytes(owned, array_bytes::<usize>(tensor.shape.capacity())?)?;
            if let Some(values) = &tensor.public_q18 {
                owned = add_bytes(owned, array_bytes::<i64>(values.capacity())?)?;
            } else {
                let lanes = tensor
                    .shape
                    .iter()
                    .try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(Error::Bounds))?;
                let file_bytes = u64::try_from(lanes)
                    .map_err(|_| Error::Bounds)?
                    .checked_mul(8)
                    .ok_or(Error::Bounds)?;
                if file_bytes == 0 || file_bytes > pllm_core::public_file::MAX_BYTES {
                    return Err(Error::Bounds);
                }
                // Disk bytes are authenticated/reserved by PublicFileWriter,
                // never charged as a resident source tensor here.
                owned = add_bytes(owned, tree_entry_bytes::<(String, PublicFile)>()?)?;
                owned = add_bytes(owned, key.capacity())?;
                owned = add_bytes(owned, public_file_arc_bytes()?)?;
            }
        }
        owned = add_bytes(owned, envelope.total_bytes)?;
        owned = add_bytes(owned, binding_workspace_bytes()?)?;
        for bytes in [
            2 * 16384, // existing maximum encoded frame, NOT source length
            size_of::<Pending>(),
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<PublicSource>(),
            size_of::<Result<PublicSource, String>>(),
            size_of::<Result<PublicFile, String>>(),
            size_of::<Result<PublicFileWriter, String>>(),
            size_of::<Result<source_norms::BlockScan, Error>>(),
            size_of::<Result<source_norms::BlockNorms, Error>>(),
            size_of::<SourceConstructionQuote>(),
            size_of::<Result<SourceConstructionQuote, Error>>(),
            size_of::<source_norms::ImportEnvelope>(),
            // One cloned record/key may coexist with its destination slot.
            size_of::<(String, SourceTensor)>(),
        ] {
            owned = add_bytes(owned, bytes)?;
        }
        SourceConstructionQuote::new(sources, norms, retained_baseline_bytes, owned)
    }

    /// Compiler-private opt-in: validate/quote BEFORE copying any owned metadata.
    /// Parent first obtains SourceRegistry from compiler-minted descriptors under
    /// its separately preadmitted validation/Manifest-construction allowance.
    /// This entry reuses that authority instead of rerunning allocating semantic
    /// lowering/shape maps. It does not validate a manifest execution digest.
    /// Legacy `new(&Manifest)` and its default routing remain unchanged.
    #[cfg(test)]
    pub(super) fn new_checked(
        registry: &SourceRegistry<'_>,
        retained_baseline_bytes: usize,
        cap: usize,
    ) -> Result<Self, Error> {
        Self::construction_quote(registry, retained_baseline_bytes)?.admit(cap)?;
        #[cfg(test)]
        CHECKED_IMPORT_FACTORIES.with(|n| n.set(n.get() + 1));
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(registry.source_records().len())
            .map_err(|_| Error::Allocation)?;
        if tensors.capacity() != registry.source_records().len() {
            return Err(Error::Allocation);
        }
        for (key, tensor) in registry.source_records() {
            tensors.push((
                copy_string(key)?,
                SourceTensor {
                    shape: copy_slice(&tensor.shape)?,
                    lower: tensor.lower,
                    upper: tensor.upper,
                    digest: tensor.digest,
                    public_q18: tensor.public_q18.as_deref().map(copy_slice).transpose()?,
                    row_l1: tensor.row_l1,
                    row_squared_norm: tensor.row_squared_norm,
                    row_l1_squared_sum: tensor.row_l1_squared_sum,
                },
            ));
        }
        Ok(Self(Some(Pending {
            tensors,
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
            binding: registry.binding(),
            norms: registry.source_norms().clone(),
            norm_scan: None,
        })))
    }

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
    /// Existing immutable handle/key backing only. Retain this owner while using
    /// the census; do not infer aliasing from paths/digests or charge file length
    /// as resident bytes. Parent adds this inventory to its outside baseline.
    pub(super) fn resident_metadata_bytes(&self) -> Result<usize, Error> {
        let mut bytes = size_of::<Self>();
        for key in self.files.keys() {
            bytes = add_bytes(bytes, tree_entry_bytes::<(String, PublicFile)>()?)?;
            bytes = add_bytes(bytes, key.capacity())?;
            bytes = add_bytes(bytes, public_file_arc_bytes()?)?;
        }
        Ok(bytes)
    }

    #[cfg(test)]
    pub(super) fn test_values(
        program: &Program,
        values: &BTreeMap<String, Vec<i64>>,
    ) -> Option<Self> {
        let mut files = BTreeMap::new();
        for position in 0..program.input_count() {
            let (_, node) = program.input_at(position);
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
        // Preserve binding-first failure ordering before touching input nodes.
        if self.binding != program.source_binding {
            return Err(Error::Binding);
        }
        self.check_files(
            program.source_binding,
            (0..program.input_count()).filter_map(|position| {
                let (_, node) = program.input_at(position);
                if let Code::PublicInput { key, digest, .. } = &node.code {
                    Some(super::source_registry::FileDescriptor {
                        key,
                        digest: *digest,
                        lanes: node.lanes,
                    })
                } else {
                    None
                }
            }),
        )
    }

    pub(super) fn check_registry(
        &self,
        registry: &super::source_registry::SourceRegistry<'_>,
    ) -> Result<(), Error> {
        self.check_files(registry.binding(), registry.files())
    }

    fn check_files<'a>(
        &self,
        binding: [u8; 32],
        descriptors: impl Iterator<Item = super::source_registry::FileDescriptor<'a>>,
    ) -> Result<(), Error> {
        if self.binding != binding {
            return Err(Error::Binding);
        }
        let mut count = 0;
        for descriptor in descriptors {
            let file = self.files.get(descriptor.key).ok_or(Error::Binding)?;
            if file.digest() != descriptor.digest || file.size() != descriptor.lanes as u64 * 8 {
                return Err(Error::Binding);
            }
            count += 1;
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

fn public_file_arc_bytes() -> Result<usize, Error> {
    add_bytes(size_of::<std::fs::File>(), array_bytes::<usize>(2)?)
}

#[cfg(test)]
fn copy_slice<T: Copy>(source: &[T]) -> Result<Vec<T>, Error> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(source.len())
        .map_err(|_| Error::Allocation)?;
    if copy.capacity() != source.len() {
        return Err(Error::Allocation);
    }
    copy.extend_from_slice(source);
    Ok(copy)
}

#[cfg(test)]
fn copy_string(source: &str) -> Result<String, Error> {
    let mut copy = String::new();
    copy.try_reserve_exact(source.len())
        .map_err(|_| Error::Allocation)?;
    if copy.capacity() != source.len() {
        return Err(Error::Allocation);
    }
    copy.push_str(source);
    Ok(copy)
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
