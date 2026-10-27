//! Private compact client route; the ordinary Publisher remains the default.
use super::compact_publisher_scan::Scan;
use super::*;
use crate::replicated_decoder::program::source_entry::ingress::{
    CompiledSource, HostRawExpectations, RawHost, SourceCollector, SourceHeaderEntry,
};
use crate::replicated_decoder::source_registry;
use pllm_core::public_file::{PublicFile, PublicFileWriter};
use pllm_garble::shared_allocation::OwnedBudget;
use std::mem::size_of;

fn add(a: usize, b: usize) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| "compact publisher capacity overflow".into())
}
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| "compact publisher capacity overflow".into())
}
fn native(e: Error) -> String {
    e.to_string()
}
fn allocation(_: impl std::fmt::Display) -> Error {
    Error::Bounds
}

struct Snapshot {
    raw: PublicFile,
    encoded: PublicFile,
}
struct Collection {
    // Same genuine Header directory, not an ambient/client-selected path.
    directory: Arc<dyn pllm_core::public_file::SnapshotDirectory>,
    shapes: Vec<(String, Vec<usize>)>,
    requests: BTreeMap<String, (usize, usize, usize)>,
    gamma_keys: std::collections::BTreeSet<String>,
    snapshots: Vec<Snapshot>,
    writer: Option<PublicFileWriter>,
    raw_writer: Option<PublicFileWriter>,
    scan: Option<Scan>,
    gamma: Option<Zeroizing<Vec<i64>>>,
    norms: Option<source_norms::BlockScan>,
    tokens: Zeroizing<Vec<i64>>,
    cursor: usize,
    offset: usize,
    issued: usize,
    issue_offset: usize,
    vocabulary: usize,
    root: OwnedBudget,
}
#[cfg(test)]
impl Drop for Collection {
    fn drop(&mut self) {
        use crate::replicated_decoder::program::source_entry::ingress::handoff_tests;
        if handoff_tests::real_payload_probe_armed(self.root.budget()) {
            // Observe the actual retained file and heap, without retaining either.
            let backing = self.snapshots.first().and_then(|snapshot| snapshot.raw.backing().ok().flatten());
            handoff_tests::observe_real_payload_drop(
                self.root.budget(),
                backing,
                (self.snapshots.as_ptr() as usize, self.snapshots.len(), self.snapshots.capacity()),
            );
        }
    }
}
enum Stage {
    Collecting(SourceCollector<Collection>),
    Sealed(CompiledSource<Collection>),
    Retired,
}
pub(super) struct CompactPublisher {
    stage: Stage,
    manifest_bytes: usize,
    collection_bytes: usize,
    visits: u64,
    disk_bytes: usize,
}

impl CompactPublisher {
    /// Host-only provisioned expectations, NOT a worker/payload/SDK entrypoint.
    /// Session is supplied by the admitted host; no new randomness or activation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_raw_host(
        experiment: &[u8],
        config: &[u8],
        rows: usize,
        outputs: usize,
        session: [u8; 32],
        source: [u8; 32],
        composition: [u8; 32],
        expected: Option<HostRawExpectations<'_>>,
    ) -> Result<RawHost, String> {
        let header = SourceHeaderEntry::begin_raw_host(
            experiment, config, rows, outputs, session, source, composition, expected,
        )
        .map_err(native)?;
        RawHost::prepare(header).map_err(native)
    }

    pub(super) fn new(
        experiment: &[u8],
        config: &[u8],
        rows: usize,
        outputs: usize,
        source: [u8; 32],
        composition: [u8; 32],
    ) -> Result<Self, String> {
        if outputs != 2 {
            return Err("compact publisher requires exactly two output epochs".into());
        }
        let mut session = [0; 32];
        getrandom::fill(&mut session).map_err(|_| "replicated OS randomness failed")?;
        let mut header = SourceHeaderEntry::begin(
            experiment,
            config,
            rows,
            outputs,
            session,
            source,
            composition,
        )
        .map_err(native)?;
        // Prepay the bounded census itself before walking original semantic ops.
        let count = header.original().plan().prefill.operations.len();
        let census_work = mul(mul(add(count, 1)?, add(count, 1)?)?, 256)?;
        header
            .admit_collection(
                0,
                0,
                u64::try_from(census_work).map_err(|_| "compact census work overflow")?,
            )
            .map_err(native)?;
        let plan = header.original().plan();
        let mut names = 0usize;
        let mut width = 1usize;
        let mut records = 0usize;
        for op in &plan.prefill.operations {
            names = add(names, op.id.len())?;
            width = width.max(
                usize::try_from(*op.output_shape.last().ok_or("empty compact source shape")?)
                    .map_err(|_| "compact width overflow")?,
            );
            for attribute in ["weight", "bias"] {
                if let Some(key) = op.attributes.get(attribute).and_then(|v| v.as_str()) {
                    names = add(names, key.len())?;
                    records = add(records, 1)?;
                }
            }
        }
        // Semantic-plan-derived conservative destination/old-new capacities.
        // No MAX-byte value becomes credit. Actual capacities are checked below.
        let tree = source_registry::tree_entry_bytes::<(String, SourceTensor)>().map_err(native)?;
        let norm_tree = source_registry::tree_entry_bytes::<(String, source_norms::BlockNorms)>()
            .map_err(native)?;
        let shape_bytes = mul(records, add(tree, 2 * size_of::<usize>())?)?;
        let gamma_bytes = mul(mul(records, width)?, size_of::<i64>())?;
        let norm_bytes = mul(records, add(norm_tree, mul(width, 3 * size_of::<u128>())?)?)?;
        let manifest_bytes = add(
            add(
                size_of::<Manifest>(),
                add(
                    header.manifest().schema.capacity(),
                    header.manifest().config.capacity(),
                )?,
            )?,
            mul(
                4,
                add(add(shape_bytes, gamma_bytes)?, add(norm_bytes, names)?)?,
            )?,
        )?;
        let norm_scan = source_norms::BlockScan::payload_bytes(width, width, 1).map_err(native)?;
        let files = mul(
            records,
            add(
                add(
                    size_of::<Snapshot>(),
                    mul(2, pllm_core::public_file::backing_bytes()?)?,
                )?,
                add(
                    2 * size_of::<PublicFileWriter>() + 4 * size_of::<Sha256>(),
                    mul(2, pllm_core::public_file::writer_creation_bytes()?)?,
                )?,
            )?,
        )?;
        let scratch = add(
            add(
                compact_publisher_scan_bytes(),
                mul(
                    4,
                    size_of::<Arc<dyn pllm_core::public_file::SnapshotDirectory>>(),
                )?,
            )?,
            add(8 * CHUNK * size_of::<u64>(), mul(rows + outputs + 1, 32)?)?,
        )?;
        let collection_bytes = add(
            mul(4, names)?,
            add(
                mul(
                    3,
                    size_of::<Publisher>() + size_of::<Self>() + size_of::<Collection>(),
                )?,
                add(
                    scratch,
                    mul(
                        4,
                        add(
                            add(shape_bytes, names)?,
                            add(gamma_bytes, add(add(files, norm_scan)?, norm_bytes)?)?,
                        )?,
                    )?,
                )?,
            )?,
        )?;
        let visits = u64::try_from(add(
            census_work,
            mul(mul(add(width, 1)?, add(width, 1)?)?, 256)?,
        )?)
        .map_err(|_| "compact visits overflow")?;
        let root = header.clone_collection_budget().map_err(native)?;
        let directory =
            Arc::clone(header.directory()) as Arc<dyn pllm_core::public_file::SnapshotDirectory>;
        let owner = header
            .collect(manifest_bytes, collection_bytes, visits, |original, _| {
                let (shapes, vocabulary) =
                    program::source_shapes_from_plan(original.plan(), WeightMode::Public)
                        .map_err(allocation)?;
                let requests = source_norms::requests(&original.plan().prefill)?;
                let envelope = source_norms::requests_envelope(&requests)?;
                if envelope.total_bytes
                    > mul(
                        4,
                        add(norm_scan, add(norm_bytes, names).map_err(allocation)?)
                            .map_err(allocation)?,
                    )
                    .map_err(allocation)?
                {
                    return Err(Error::Bounds);
                }
                let gamma_keys = original
                    .plan()
                    .prefill
                    .operations
                    .iter()
                    .filter(|op| op.operator == pllm_models::ModelOperator::RmsNorm)
                    .map(|op| {
                        op.attributes["weight"]
                            .as_str()
                            .ok_or(Error::Binding)
                            .map(str::to_owned)
                    })
                    .collect::<Result<_, _>>()?;
                let mut ordered = Vec::new();
                ordered
                    .try_reserve_exact(shapes.len())
                    .map_err(allocation)?;
                ordered.extend(shapes);
                let mut snapshots = Vec::new();
                snapshots
                    .try_reserve_exact(ordered.len())
                    .map_err(allocation)?;
                if snapshots.capacity() != ordered.len() || ordered.capacity() != ordered.len() {
                    return Err(Error::Bounds);
                }
                Ok(Collection {
                    directory,
                    shapes: ordered,
                    requests,
                    gamma_keys,
                    snapshots,
                    writer: None,
                    raw_writer: None,
                    scan: None,
                    gamma: None,
                    norms: None,
                    tokens: Zeroizing::new(Vec::new()),
                    cursor: 0,
                    offset: 0,
                    issued: 0,
                    issue_offset: 0,
                    vocabulary,
                    root,
                })
            })
            .map_err(native)?;
        #[cfg(test)]
        let mut owner = owner;
        #[cfg(test)]
        crate::replicated_decoder::program::source_entry::ingress::handoff_tests::observe_real_collector(&mut owner);
        let mut words = 0usize;
        for (_, shape) in &owner.payload().map_err(native)?.shapes {
            let n = shape.iter().try_fold(1usize, |n, d| mul(n, *d))?;
            words = add(words, n)?;
        }
        // Client raw F32 + Q18, plus E's Q18 importer snapshot. Outer SDK disk
        // reservation MUST cover this for the whole handle lifetime; no native
        // resident-byte lease is misrepresented as a filesystem reservation.
        let disk_bytes = mul(words, 20)?;
        Ok(Self {
            stage: Stage::Collecting(owner),
            manifest_bytes,
            collection_bytes,
            visits,
            disk_bytes,
        })
    }
    pub(super) fn shapes(&self) -> &[(String, Vec<usize>)] {
        match &self.stage {
            Stage::Collecting(c) => c.payload().map(|v| v.shapes.as_slice()).unwrap_or(&[]),
            Stage::Sealed(c) => c.payload().map(|v| v.shapes.as_slice()).unwrap_or(&[]),
            Stage::Retired => &[],
        }
    }
    pub(super) fn disk_bytes(&self) -> usize {
        self.disk_bytes
    }
    pub(super) fn storage_kind(&self) -> Option<&'static str> {
        match &self.stage {
            Stage::Sealed(c) => c.program().ok().map(|p| p.program().storage_kind()),
            _ => None,
        }
    }
    pub(super) fn directory(&self) -> Result<&std::path::Path, String> {
        match &self.stage {
            Stage::Collecting(c) => Ok(c.directory().path()),
            Stage::Sealed(c) => c.directory().map(|d| d.path()).map_err(native),
            Stage::Retired => Err("compact publisher is retired".into()),
        }
    }
    pub(super) fn append(&mut self, key: &str, offset: usize, bytes: &[u8]) -> Result<(), String> {
        let result = match &mut self.stage {
            Stage::Collecting(c) => c
                .phase(
                    self.manifest_bytes,
                    self.collection_bytes,
                    self.visits
                        .checked_mul(bytes.len() as u64 + 1)
                        .ok_or("compact frame work overflow")?,
                    |state, records, original| {
                        if !std::ptr::eq(state.root.budget(), original.budget()) {
                            return Err(Error::Binding);
                        }
                        let (name, shape) = state.shapes.get(state.cursor).ok_or(Error::Binding)?;
                        if name != key || offset != state.offset {
                            return Err(Error::Binding);
                        }
                        let elements = shape
                            .iter()
                            .try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(Error::Bounds))?;
                        if bytes.is_empty()
                            || bytes.len() > CHUNK * 4
                            || bytes.len() % 4 != 0
                            || offset
                                .checked_add(bytes.len() / 4)
                                .is_none_or(|end| end > elements)
                        {
                            return Err(Error::Binding);
                        }
                        if offset == 0 {
                            state.scan = Some(
                                Scan::new(
                                    elements,
                                    if shape.len() == 2 {
                                        Some(shape[1])
                                    } else {
                                        None
                                    },
                                )
                                .map_err(allocation)?,
                            );
                            state.writer = Some(
                                PublicFileWriter::new_computed_in(
                                    (elements as u64).checked_mul(8).ok_or(Error::Bounds)?,
                                    Arc::clone(&state.directory),
                                )
                                .map_err(allocation)?,
                            );
                            state.raw_writer = Some(
                                PublicFileWriter::new_computed_in(
                                    (elements as u64).checked_mul(4).ok_or(Error::Bounds)?,
                                    Arc::clone(&state.directory),
                                )
                                .map_err(allocation)?,
                            );
                            state.gamma = if state.gamma_keys.contains(key) {
                                let values = vec_zero(elements)?;
                                if values.capacity() != elements {
                                    return Err(Error::Bounds);
                                }
                                Some(Zeroizing::new(values))
                            } else {
                                None
                            };
                            state.norms = state
                                .requests
                                .get(key)
                                .map(|&(r, c, b)| source_norms::BlockScan::new(r, c, b))
                                .transpose()?;
                        }
                        state
                            .scan
                            .as_mut()
                            .ok_or(Error::Binding)?
                            .append(
                                offset,
                                bytes,
                                state.writer.as_mut().ok_or(Error::Binding)?,
                                state.gamma.as_mut().map(|v| v.as_mut_slice()),
                                state.norms.as_mut(),
                            )
                            .map_err(allocation)?;
                        state
                            .raw_writer
                            .as_mut()
                            .ok_or(Error::Binding)?
                            .write(bytes)
                            .map_err(allocation)?;
                        state.offset = state
                            .offset
                            .checked_add(bytes.len() / 4)
                            .ok_or(Error::Bounds)?;
                        if state.offset == elements {
                            let summary = state
                                .scan
                                .take()
                                .ok_or(Error::Binding)?
                                .finish()
                                .map_err(allocation)?;
                            let encoded = state
                                .writer
                                .take()
                                .ok_or(Error::Binding)?
                                .finish()
                                .map_err(allocation)?;
                            let raw = state
                                .raw_writer
                                .take()
                                .ok_or(Error::Binding)?
                                .finish()
                                .map_err(allocation)?;
                            if encoded.digest() != summary.digest {
                                return Err(Error::Binding);
                            }
                            let gamma = state.gamma.take().map(|mut v| std::mem::take(&mut *v));
                            records.tensors.insert(
                                key.to_owned(),
                                SourceTensor {
                                    shape: shape.clone(),
                                    lower: summary.lower,
                                    upper: summary.upper,
                                    digest: summary.digest,
                                    public_q18: gamma,
                                    row_l1: summary.row_l1,
                                    row_squared_norm: summary.row_squared_norm,
                                    row_l1_squared_sum: summary.row_l1_squared_sum,
                                },
                            );
                            if let Some(norms) = state.norms.take() {
                                records.norms.insert(key.to_owned(), norms.finish()?);
                            }
                            state.snapshots.push(Snapshot { raw, encoded });
                            state.cursor += 1;
                            state.offset = 0;
                        }
                        Ok(())
                    },
                )
                .map_err(native),
            _ => Err("compact publisher is not collecting".into()),
        };
        if result.is_err() {
            self.stage = Stage::Retired;
        }
        result
    }
    pub(super) fn seal(&mut self, ids: &[u32]) -> Result<(), String> {
        let stage = std::mem::replace(&mut self.stage, Stage::Retired);
        let Stage::Collecting(mut owner) = stage else {
            return Err("compact publisher is not collecting".into());
        };
        let expected = owner.manifest().rows;
        let feed_visits = self
            .visits
            .checked_mul(
                (self.disk_bytes / 20)
                    .checked_add(ids.len())
                    .and_then(|n| n.checked_add(1))
                    .ok_or("compact feed work overflow")? as u64,
            )
            .ok_or("compact feed work overflow")?;
        owner
            .phase(
                self.manifest_bytes,
                self.collection_bytes,
                feed_visits,
                |state, records, _| {
                    if ids.len() != expected
                        || state.cursor != state.shapes.len()
                        || state.offset != 0
                        || ids.iter().any(|id| *id as usize >= state.vocabulary)
                    {
                        return Err(Error::Binding);
                    }
                    if state.snapshots.len() != state.shapes.len() {
                        return Err(Error::Binding);
                    }
                    for ((key, shape), snapshot) in state.shapes.iter().zip(&state.snapshots) {
                        let words = shape
                            .iter()
                            .try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(Error::Bounds))?;
                        let tensor = records.tensors.get(key).ok_or(Error::Binding)?;
                        if tensor.shape != *shape
                            || tensor.digest != snapshot.encoded.digest()
                            || snapshot.encoded.size() != words as u64 * 8
                            || snapshot.raw.size() != words as u64 * 4
                        {
                            return Err(Error::Binding);
                        }
                    }
                    state
                        .tokens
                        .try_reserve_exact(ids.len())
                        .map_err(allocation)?;
                    if state.tokens.capacity() != ids.len() {
                        return Err(Error::Bounds);
                    }
                    state.tokens.extend(ids.iter().map(|v| i64::from(*v)));
                    Ok(())
                },
            )
            .map_err(native)?;
        // Feed original retained F32 bytes, never lossy reconstruction from Q18.
        let compiled = owner
            .finish(|state, import| {
                let mut frame = Zeroizing::new([0u8; CHUNK * 4]);
                for ((key, shape), snapshot) in state.shapes.iter().zip(&state.snapshots) {
                    let count = shape
                        .iter()
                        .try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(Error::Bounds))?;
                    for offset in (0..count).step_by(CHUNK) {
                        let n = CHUNK.min(count - offset);
                        snapshot
                            .raw
                            .read((offset * 4) as u64, &mut frame[..n * 4])
                            .map_err(allocation)?;
                        import.append(key, offset, &frame[..n * 4])?;
                        #[cfg(test)]
                        {
                            use crate::replicated_decoder::program::source_entry::ingress::handoff_tests;
                            if handoff_tests::real_fixture_matches(state.root.budget()) {
                                // Only the first successful genuine raw read +
                                // importer append may reach the injected failure.
                                handoff_tests::after_real_import_append(
                                    import,
                                    snapshot.raw.backing().unwrap().unwrap(),
                                    (state.snapshots.as_ptr() as usize, state.snapshots.len(), state.snapshots.capacity()),
                                )?;
                            }
                        }
                    }
                }
                Ok(())
            })
            .map_err(native)?;
        if compiled.program().map_err(native)?.program().storage_kind() != "compact" {
            return Err("compact publisher storage differs".into());
        }
        self.stage = Stage::Sealed(compiled);
        Ok(())
    }
    pub(super) fn cancel(&mut self) {
        self.stage = Stage::Retired;
    }

    pub(super) fn manifest(&mut self) -> Result<Vec<u8>, String> {
        let Stage::Sealed(owner) = &mut self.stage else {
            return Err("replicated publisher not sealed".into());
        };
        // Clone ONLY the actual wrapper's retained Arc, never construct/adopt a
        // SourceRequestProgram from it. Clone allocates no new graph or backing.
        let program = Arc::clone(owner.program().map_err(native)?.program());
        let bytes = add(self.collection_bytes, mul(self.manifest_bytes, 16)?)?;
        owner
            .phase(bytes, self.visits, |_| {
                let manifest = program
                    .compact_storage()
                    .ok_or(Error::Binding)?
                    .source_manifest()?;
                let wire = Wire {
                    manifest,
                    program: program.digest(),
                };
                let result = serde_json::to_vec(&wire).map_err(allocation)?;
                if result.capacity() > bytes {
                    return Err(Error::Bounds);
                }
                Ok(result)
            })
            .map_err(native)
    }
    pub(super) fn next_chunk(&mut self) -> Result<Option<SourceChunk>, String> {
        let Stage::Sealed(owner) = &mut self.stage else {
            return Err("replicated publisher not sealed".into());
        };
        let wrapper = owner.program().map_err(native)?;
        let count = wrapper.input_count();
        let program = Arc::clone(wrapper.program());
        owner
            .phase(self.collection_bytes, self.visits, |state| {
                let access = program.schedule_access();
                while state.issued < count {
                    let node = access.node(state.issued)?;
                    match node.kind() {
                        program::schedule_access::Kind::PublicInput => {
                            state.issued += 1;
                        }
                        program::schedule_access::Kind::Input => {
                            if node.lanes() != state.tokens.len() {
                                return Err(Error::Binding);
                            }
                            let offset = state.issue_offset;
                            let n = CHUNK.min(
                                state
                                    .tokens
                                    .len()
                                    .checked_sub(offset)
                                    .ok_or(Error::Binding)?,
                            );
                            if n == 0 {
                                return Err(Error::Binding);
                            }
                            let result = source_chunk(
                                node.input_name()?.to_owned(),
                                offset,
                                &state.tokens[offset..offset + n],
                            )
                            .map_err(allocation)?;
                            state.issue_offset += n;
                            if state.issue_offset == state.tokens.len() {
                                state.tokens.zeroize();
                                state.issue_offset = 0;
                                state.issued += 1;
                            }
                            return Ok(Some(result));
                        }
                        _ => return Err(Error::Binding),
                    }
                }
                Ok(None)
            })
            .map_err(native)
    }
    pub(super) fn finish_outcome(mut self, frames: [&[u8]; 3]) -> Result<FinalOutcome, String> {
        let Stage::Sealed(owner) = &mut self.stage else {
            return Err("replicated publisher not sealed".into());
        };
        let program = Arc::clone(owner.program().map_err(native)?.program());
        let count = owner.program().map_err(native)?.input_count();
        owner
            .phase(self.collection_bytes, self.visits, |state| {
                // Any trailing public inputs are metadata, not extra secret shares.
                let access = program.schedule_access();
                while state.issued < count {
                    if access.node(state.issued)?.kind()
                        != program::schedule_access::Kind::PublicInput
                    {
                        return Err(Error::Binding);
                    }
                    state.issued += 1;
                }
                let manifest = program
                    .compact_storage()
                    .ok_or(Error::Binding)?
                    .source_manifest()?;
                finish_manifest_outcome(manifest, program.digest(), frames, Some(state.vocabulary))
                    .map_err(allocation)
            })
            .map_err(native)
    }
}

// Same wire Manifest fields/order; only program is the FINAL actual digest.
struct Wire<'a> {
    manifest: &'a Manifest,
    program: [u8; 32],
}
impl Serialize for Wire<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let m = self.manifest;
        let mut map = s.serialize_struct("Manifest", 11 + usize::from(!m.norms.is_empty()))?;
        map.serialize_field("schema", &m.schema)?;
        map.serialize_field("weights", &m.weights)?;
        map.serialize_field("execution", &m.execution)?;
        map.serialize_field("config", &m.config)?;
        map.serialize_field("rows", &m.rows)?;
        map.serialize_field("outputs", &m.outputs)?;
        map.serialize_field("source", &m.source)?;
        map.serialize_field("composition", &m.composition)?;
        map.serialize_field("session", &m.session)?;
        map.serialize_field("tensors", &m.tensors)?;
        if !m.norms.is_empty() {
            map.serialize_field("norms", &m.norms)?;
        }
        map.serialize_field("program", &self.program)?;
        map.end()
    }
}

#[cfg(test)]
#[path = "compact_publisher_tests.rs"]
mod tests;
