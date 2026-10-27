//! Conditional host-pin integrity, rooted reads and genuine BF16 TokenLookup.
//! Not publisher authentication, a CLEAR projection or external lease proof.
//! Lives under source_entry::ingress so header custody remains private. No SDK,
//! worker deserializer, fixture producer, source registry or Q18 import is added.
//!
//! Logical original-counter reservations cover the named owned slots and borrowed
//! chunk window, not allocator/RSS, native I/O call frames or caller spare capacity.
//! Caller must retain/admit its actual chunk backing separately; append borrows
//! only a nonempty <= PublicFile MAX_CHUNK_BYTES window for that call. All original
//! work is prepaid before writers, including worst-case one-byte chunk overhead,
//! complete hashes/writes, JSON scanner/parser visits, numeric/scale/Q8 passes and
//! head comparison. Raw codec quote alone does NOT cover these additional passes.
use super::*;
use crate::verified_source::{raw, SourceDecodeWorkspace};
use pllm_core::public_file::{self, PublicFile, PublicFileWriter, SnapshotDirectory};
use pllm_models::bounded_canonical as canonical;
use pllm_types::bounded_canonical::{StreamFrame, StreamLimits};
use sha2::{Digest as _, Sha256};

#[cfg(any(test, feature = "lifecycle-test-support"))]
#[path = "raw_fixture.rs"]
mod raw_fixture;

// Child of the raw custody owner, not a parallel source authority. Keeping this
// private declaration here avoids touching the provider-owned source adapter.
#[path = "../replicated_decoder/raw_clear_producer.rs"]
mod raw_clear_producer;
pub(in crate::replicated_decoder) use raw_clear_producer::{
    AdmittedRawInput, PrivateSourceInstance, RawLookupActivation,
};

const NAME: usize = 1024;
const JSON_BYTES: usize = 1 << 20; // existing source-entry JSON cap, not a cap raise
const FRAME: usize = 4096;
const CANONICAL_WORK: usize = 1 << 28;

/// Borrowed immutable independently provisioned HOST configuration only. No
/// Deserialize, Clone, payload setter, self-attestation or expected-pin issuer.
/// File order is config, index, shards. Tensor tuple = name, file ordinal, dtype
/// (F32/BF16/F8_E4M3), row-major shape, relative data begin/end, optional BF16
/// block-scale name. Safetensors supplies LE contiguous row-major layout only.
pub(in crate::replicated_decoder) struct HostRawExpectations<'a> {
    files: &'a [(&'a str, u64, [u8; 32])],
    tensors: &'a [(&'a str, usize, &'a str, &'a [usize], u64, u64, Option<&'a str>)],
    config: [u8; 32],
    semantic: [u8; 32],
    recipe: &'a str,
}
impl<'a> HostRawExpectations<'a> {
    /// Provisioning is a host trust premise; this constructor authenticates no
    /// issuer. It traverses nothing. The admitted header validates/copies later.
    pub(in crate::replicated_decoder) fn provision(
        files: &'a [(&'a str, u64, [u8; 32])],
        tensors: &'a [(&'a str, usize, &'a str, &'a [usize], u64, u64, Option<&'a str>)],
        config: [u8; 32],
        semantic: [u8; 32],
        recipe: &'a str,
    ) -> Self {
        Self { files, tensors, config, semantic, recipe }
    }

    // Fixed bounded FILE census only, before any original JSON/factory/directory.
    fn preflight(&self) -> Result<(), Error> {
        if !(3..=source_policy::DESCRIPTORS).contains(&self.files.len())
            || self.tensors.is_empty()
            || self.tensors.len() > source_policy::DESCRIPTORS
        { return Err(Error::Bounds); }
        for (_, size, _) in self.files {
            if *size == 0 || *size > public_file::MAX_BYTES { return Err(Error::Bounds); }
        }
        Ok(())
    }
}

struct ExpectedFile { name: String, size: u64, digest: [u8; 32] }
struct ExpectedTensor {
    name: String,
    file: usize,
    dtype: raw::RawScalarType,
    shape: [usize; 2],
    rank: usize,
    begin: u64,
    end: u64,
    scale: Option<String>,
    scale_index: Option<usize>,
}
impl ExpectedTensor {
    fn elements(&self) -> Result<usize, Error> {
        self.shape[..self.rank].iter().try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(Error::Bounds))
    }
    fn width(&self) -> u64 { match self.dtype { raw::RawScalarType::F32Le => 4, raw::RawScalarType::Bf16Le => 2, raw::RawScalarType::E4m3Fn => 1 } }
}

/// Only this CHILD can construct. Actual retained directory Arc identity plus
/// its original budget are custody, not equal labels, digests or fstat tuples.
/// Directory was privately created with this header's immutable original plan.
pub(super) struct TrustedRawPremise {
    files: Vec<ExpectedFile>,
    tensors: Vec<ExpectedTensor>,
    identity: OwnedBudget,
    work: u64,
    phase_bytes: usize,
}
impl TrustedRawPremise {
    fn check(&self, header: &SourceHeaderEntry) -> Result<(), Error> {
        if header.failed || header.directory.is_some()
            || !std::ptr::eq(self.identity.budget(), header.root.budget())
            || !std::ptr::eq(self.identity.budget(), header.original.budget())
            || !header.controller.is_for(self.identity.budget())
            || !header.collection_charge.is_for(self.identity.budget())
        { return Err(Error::Binding); }
        Ok(())
    }
    fn provision(header: &mut SourceHeaderEntry, expected: HostRawExpectations<'_>) -> Result<Self, Error> {
        let f = expected.files.len();
        let t = expected.tensors.len();
        let ops = header.original.plan().prefill.operations.len();
        // Bounds depend only on scalar counts and existing caps: no string or
        // inventory traversal before visitor/copy/validation/retirement prepay.
        let metadata = plus(
            mul(f, plus(size_of::<ExpectedFile>(), NAME)?)?,
            mul(t, plus(size_of::<ExpectedTensor>(), mul(2, NAME)?)?)?,
        )?;
        let shapes = mul(ops, plus(
            plus(source_registry::tree_entry_bytes::<(String, Vec<usize>)>()?, source_registry::tree_entry_bytes::<(String, usize)>()?)?,
            plus(mul(2, NAME)?, mul(2, size_of::<usize>())?)?,
        )?)?;
        let controls = plus(mul(3, size_of::<TrustedRawPremise>())?, plus(size_of::<OwnedBudget>(), mul(3, size_of::<HostRawExpectations<'_>>())?)?)?;
        // Synthetic preparation can already own funded payload on this SAME
        // header. Retain that whole debit while provisioning/runtime overlap.
        let initial = plus(header.collection_charge.payload_bytes(), plus(mul(2, source_policy::CONFIG_BYTES)?, plus(mul(2, metadata)?, plus(mul(2, shapes)?, plus(controls,
            canonical::resources::<std::io::Sink>(16).map_err(|_| Error::Bounds)?.hash_workspace_bytes)?)?)?)?)?;
        let comparisons = mul(mul(plus(plus(f, t)?, ops)?, plus(plus(f, t)?, ops)?)?, NAME)?;
        let visits = plus(mul(2, CANONICAL_WORK)?, plus(comparisons, plus(initial, mul(4, source_policy::CONFIG_BYTES)?)?)?)?;
        // The existing square covers deduplication and the other preparation
        // passes, NOT two additional independent scale-name searches. Each can
        // visit all t candidates for all t tensors, comparing NAME bytes even
        // for equal-length names with a common prefix and a late shared scale.
        // Quote from LEN alone: do not scan for FP8 tensors before paying. Burn
        // both passes on the SAME original ledger before any inventory allocation
        // or traversal; the later ingestion comparison charge is not retroactive.
        let scale_relationship_comparisons = mul(mul(t, t)?, NAME)?;
        let scale_index_comparisons = mul(mul(t, t)?, NAME)?;
        header.admit_collection(0, 0, u64::try_from(plus(
            scale_relationship_comparisons, scale_index_comparisons,
        )?).map_err(|_| Error::Bounds)?)?;
        header.admit_collection(0, initial, u64::try_from(visits).map_err(|_| Error::Bounds)?)?;
        if header.raw_premise.is_some() || expected.config == [0;32] || expected.semantic == [0;32]
            || !matches!(expected.recipe, "D31" | "D32") { return Err(Error::Binding); }
        if <[u8;32]>::from(Sha256::digest(header.original.raw_configuration())) != expected.config
            || expected.files[0].2 != expected.config
            || expected.files[0].1 != header.original.raw_configuration().len() as u64
            || semantic_identity(header.original.plan())? != expected.semantic
        { return Err(Error::Binding); }
        let mut files = exact_vec(f)?;
        let mut total = 0usize;
        for (i, (name, size, digest)) in expected.files.iter().enumerate() {
            if !valid_name(name) || *digest == [0;32] || expected.files[..i].iter().any(|(n,_,_)| n == name)
                || (i == 0 && *name != "config.json")
                || (i == 1 && !name.ends_with(".safetensors.index.json"))
                || (i >= 2 && !name.ends_with(".safetensors")) { return Err(Error::Binding); }
            if i < 2 && *size > JSON_BYTES as u64 { return Err(Error::Bounds); }
            total = plus(total, usize::try_from(*size).map_err(|_| Error::Bounds)?)?;
            files.push(ExpectedFile { name: copy_text(name)?, size: *size, digest: *digest });
        }
        let mut tensors: Vec<ExpectedTensor> = exact_vec(t)?;
        for (i, (name, file, dtype, shape, begin, end, scale)) in expected.tensors.iter().enumerate() {
            if !valid_name(name) || *file < 2 || *file >= f || !(1..=2).contains(&shape.len())
                || shape.contains(&0) || expected.tensors[..i].iter().any(|x| x.0 == *name)
                || scale.is_some_and(|s| !valid_name(s)) { return Err(Error::Binding); }
            let dtype = match *dtype { "F32" => raw::RawScalarType::F32Le, "BF16" => raw::RawScalarType::Bf16Le,
                "F8_E4M3" if expected.recipe == "D32" => raw::RawScalarType::E4m3Fn, _ => return Err(Error::Binding) };
            let mut dimensions = [1;2]; dimensions[..shape.len()].copy_from_slice(shape);
            let tensor = ExpectedTensor { name: copy_text(name)?, file: *file, dtype, shape: dimensions, rank: shape.len(), begin: *begin, end: *end, scale: scale.map(copy_text).transpose()?, scale_index: None };
            let quote = raw::quote(dtype, tensor.elements()?, usize::MAX, usize::MAX).map_err(|_| Error::Bounds)?;
            if begin.checked_add(quote.source_bytes as u64) != Some(*end) || *end > files[*file].size || *begin % tensor.width() != 0
                || (i > 0 && (tensors[i-1].file > *file || (tensors[i-1].file == *file && tensors[i-1].end != *begin)))
                || ((i == 0 || tensors[i-1].file != *file) && *begin != 0)
            { return Err(Error::Binding); }
            tensors.push(tensor);
        }
        for file in 2..f { if !tensors.iter().any(|x| x.file == file) { return Err(Error::Binding); } }
        for tensor in &tensors {
            match (tensor.dtype, tensor.scale.as_deref()) {
                (raw::RawScalarType::E4m3Fn, Some(name)) if tensor.rank == 2 => {
                    let scale = tensors.iter().find(|x| x.name == name).ok_or(Error::Binding)?;
                    raw::block128_scale_index(tensor.shape[0], tensor.shape[1], 0, 0).map_err(|_| Error::Shape)?;
                    if scale.dtype != raw::RawScalarType::Bf16Le || scale.rank != 2
                        || scale.shape != [tensor.shape[0]/128,tensor.shape[1]/128] || scale.scale.is_some()
                    { return Err(Error::Binding); }
                }
                (raw::RawScalarType::F32Le | raw::RawScalarType::Bf16Le, None) => (),
                _ => return Err(Error::Binding),
            }
        }
        for i in 0..tensors.len() {
            let index = tensors[i].scale.as_ref().map(|name| tensors.iter().position(|s| &s.name==name).ok_or(Error::Binding)).transpose()?;
            tensors[i].scale_index=index;
        }
        // Existing model-neutral semantic shape owner, not projection custody.
        let (required, _) = crate::replicated_decoder::program::source_shapes_from_plan(header.original.plan(), WeightMode::Public).map_err(|_| Error::Binding)?;
        for (name, shape) in &required {
            let found = tensors.iter().find(|x| &x.name == name).ok_or(Error::Binding)?;
            if found.shape[..found.rank] != shape[..] { return Err(Error::Shape); }
        }
        let config = parse_borrowed_configuration(header)?;
        if expected.recipe == "D32" {
            let q = &config.payload().value()["quantization_config"];
            if q["quant_method"].as_str() != Some("fp8") || q["fmt"].as_str() != Some("e4m3")
                || q["weight_block_size"].as_array().is_none_or(|a| a.len()!=2 || a[0].as_u64()!=Some(128) || a[1].as_u64()!=Some(128))
            { return Err(Error::Binding); }
        } else if config.payload().value().get("quantization_config").is_some_and(|q| !q.is_null()) { return Err(Error::Binding); }
        drop(config);
        let head = semantic_weight(header, pllm_models::ModelOperator::OutputHead)?;
        let embedding = semantic_weight(header, pllm_models::ModelOperator::TokenLookup)?;
        for name in [head, embedding] {
            if tensors.iter().find(|t| t.name == name).is_none_or(|t| t.dtype != raw::RawScalarType::Bf16Le) { return Err(Error::Binding); }
        }
        // Full future work prepaid once at prepare, before its first writer.
        // JSON calls each retain their existing 1e9 own cap; original ledger also
        // pays that complete cap, not just a scalar-decode quote or observed use.
        let ledger = PassLedger::quote(&files, &tensors, header.original.raw_configuration().len(),
            tensors.iter().find(|t| t.name == head).ok_or(Error::Binding)?.end
                .checked_sub(tensors.iter().find(|t| t.name == head).ok_or(Error::Binding)?.begin).ok_or(Error::Bounds)?)?;
        let work = plus(ledger.work()?, comparisons)?;
        let backing = [size_of::<RawBacking>(), public_arc_bytes()?,
            public_file::backing_bytes().map_err(|_| Error::Bounds)?]
            .into_iter().try_fold(0, plus)?;
        let phase_bytes = [initial, mul(f, backing)?,
            public_file::writer_creation_bytes().map_err(|_| Error::Bounds)?,
            host_control_bytes()?, public_file::MAX_CHUNK_BYTES, mul(2, JSON_BYTES)?,
            json::bootstrap_bytes().map_err(|_| Error::Bounds)?, 2*FRAME,
            size_of::<SourceDecodeWorkspace>(), mul(4, size_of::<Sha256>())?]
            .into_iter().try_fold(0, plus)?;
        // Feature-only SOURCE logical inventory, not a physical Rust stack or
        // CPython peak proof. Pay before header freeze and publisher construction.
        #[cfg(feature = "lifecycle-test-support")]
        let phase_bytes = plus(phase_bytes, lifecycle_storage_bytes()?)?;
        #[cfg(feature = "lifecycle-test-support")]
        let work = plus(work, plus(
            mul(LIFECYCLE_OBSERVATIONS, LIFECYCLE_OBSERVATION_WORK)?,
            mul(LIFECYCLE_CHUNKS, 3 * 17 + LIFECYCLE_OBSERVATION_WORK)?,
        )?)?;
        let identity = header.root.try_clone().map_err(|_| Error::Bounds)?;
        // Grow the ORIGINAL live charge, not merely its claimed future size.
        // This must precede the header Arc/Weak freeze and every host allocation.
        header.admit_collection(0, phase_bytes, 0)?;
        Ok(Self { files, tensors, identity, work: u64::try_from(work).map_err(|_| Error::Bounds)?, phase_bytes })
    }
}

impl SourceHeaderEntry {
    /// No raw expectation is accepted by ordinary begin or any worker payload.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder) fn begin_raw_host(
        experiment: &[u8], config: &[u8], rows: usize, outputs: usize,
        session: [u8;32], source: [u8;32], composition: [u8;32],
        expected: Option<HostRawExpectations<'_>>,
    ) -> Result<RawPending, Error> {
        let expected = expected.ok_or(Error::Binding)?;
        expected.preflight()?;
        Self::begin_mode(experiment, config, rows, outputs, session, source, composition, true)?.into_raw_host(expected)
    }
    // Private consuming tail: never constructs another original/root/ledger.
    fn into_raw_host(mut self, expected:HostRawExpectations<'_>) -> Result<RawPending,Error> {
        let directory=self.prepare_raw_host(expected)?;
        let header=Arc::new(self); let identity=Arc::downgrade(&header);
        Ok(RawPending {
            #[cfg(any(test,feature="lifecycle-test-support"))] fixture:None,
            header,identity,directory,
        })
    }
    fn prepare_raw_host(&mut self, expected:HostRawExpectations<'_>) -> Result<Arc<snapshot_directory::RawSnapshotDirectory>,Error> {
        expected.preflight()?;
        let premise=TrustedRawPremise::provision(self,expected)?;
        self.raw_premise=Some(premise);
        // Filesystem-free, separately charged location owner, prepared before
        // freezing the whole header. No parent metadata, mkdir or tempfile here.
        snapshot_directory::RawSnapshotDirectory::prepare(&self.original,&self.controller,&self.root)
    }
}

/// Custody of all manifest/session/config/execution/plan/limits and original
/// charges, including experiment and prepaid work. Not a digest permission.
pub(in crate::replicated_decoder) struct RawPending {
    // Generated files/descriptors first, covering original header LAST.
    #[cfg(any(test,feature="lifecycle-test-support"))]
    fixture:Option<raw_fixture::Fixture>,
    header: Arc<SourceHeaderEntry>,
    identity: std::sync::Weak<SourceHeaderEntry>,
    directory: Arc<snapshot_directory::RawSnapshotDirectory>,
}
impl std::ops::Deref for RawPending {
    type Target = SourceHeaderEntry;
    fn deref(&self) -> &SourceHeaderEntry { &self.header }
}
impl RawPending {
    fn check(&self) -> Result<(), Error> {
        let original = self.identity.upgrade().ok_or(Error::Binding)?;
        if !Arc::ptr_eq(&original, &self.header) { return Err(Error::Binding); }
        self.directory.check_budget(self.header.original.budget())?;
        self.raw_premise.as_ref().ok_or(Error::Binding)?.check(&self.header)
    }
}

/// Checked logical API/pass counts, NOT CPU fuel, RSS, native syscall retries,
/// elapsed time, allocator enforcement or thread/timeouts. C9 parser allocations
/// retain their separately checked actual quote/construct overlap below.
struct PassLedger { appends: usize, factories: usize, writes: usize, reads: usize,
    finalizations: usize, read_bytes: usize, json_bytes: usize, json_calls: usize,
    decodes: usize, comparisons: usize }
fn ceil(n: usize, d: usize) -> Result<usize, Error> {
    if d == 0 { return Err(Error::Bounds); }
    plus(n / d, usize::from(n % d != 0))
}
impl PassLedger {
    fn quote(files: &[ExpectedFile], tensors: &[ExpectedTensor], c: usize, h: u64) -> Result<Self, Error> {
        let f = files.len();
        let shards = f.checked_sub(2).ok_or(Error::Bounds)?;
        let s = files.iter().try_fold(0, |n, f| plus(n, usize::try_from(f.size).map_err(|_| Error::Bounds)?))?;
        let (mut e, mut p, mut d) = (0, 0, 0);
        for t in tensors {
            let n = t.elements()?;
            e = plus(e, n)?;
            d = plus(d, mul(n, t.width() as usize)?)?;
            if t.scale_index.is_some() { p = plus(p, n)?; }
        }
        let h = usize::try_from(h).map_err(|_| Error::Bounds)?;
        // J bounds the index plus each actual <=JSON_BYTES shard header. We
        // cannot read headers before admission; reject excess before allocation.
        let j = plus(usize::try_from(files.get(1).ok_or(Error::Bounds)?.size).map_err(|_| Error::Bounds)?, mul(shards, JSON_BYTES)?)?;
        let reads = [ceil(c, FRAME)?, 1, mul(2, shards)?, tensors.len(), e,
            mul(2, p)?, 2, mul(2, ceil(h, FRAME)?)?].into_iter().try_fold(0, plus)?;
        let read_bytes = [c, j, mul(8, shards)?, mul(8, tensors.len())?, d,
            mul(10, p)?, 16, mul(2, h)?].into_iter().try_fold(0, plus)?;
        Ok(Self { appends: plus(s, 1)?, factories: mul(2, f)?, finalizations: f, writes: mul(3, s)?,
            reads, read_bytes, json_bytes: j, json_calls: plus(shards, 1)?,
            decodes: plus(mul(2, e)?, mul(3, p)?)?, comparisons: mul(2, plus(c, h)?)? })
    }
    fn work(&self) -> Result<usize, Error> {
        // Successful nonempty appends <=S, plus ONE terminal invalid attempt.
        // Writes include write/SHA/max-i8 byte passes; appends include update and
        // handoff controls. Finishes/create <=F each, never application retries.
        [mul(3, self.appends)?, self.factories, self.writes, self.reads,
            // Outside C9: source-member scanner, retained node/edge walk,
            // destination zero-fill, and optional metadata-value walk. Each
            // is bounded by J; parser/copy internals also retain C9's full cap.
            self.read_bytes, mul(5, self.json_bytes)?,
            mul(self.json_calls, 1_000_000_000)?, self.decodes, self.comparisons,
            // At most two SHA padding blocks per file, plus final digest and
            // backing comparisons. No retry credits: terminal failures latch.
            mul(self.finalizations, 128 + 32)?, self.appends].into_iter().try_fold(0, plus)
    }
}

/// Private payload. Only the in-place rooted host may drive it. A failing core
/// writer can burn its incomplete backing/hash, but never these surviving owners.
struct RawRegistration {
    files: Vec<RawBacking>,
    writer: Option<PublicFileWriter>,
    cursor: usize,
    offset: u64,
    identity: std::sync::Weak<snapshot_directory::RawSnapshotDirectory>,
    directory: Arc<snapshot_directory::RawSnapshotDirectory>,
    header: RawPending,
    // Reader controls remain covered through directory retirement and genuine
    // outer lease exit. Reader Drop must not refund the host's final charge.
    reader_charge: Option<OwnedExternalCharge>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HostStage { Pending, Collecting, Sealed, Failed, DirectoryRetired, Released }
struct RawHostState {
    #[cfg(feature = "lifecycle-test-support")]
    observations_left: usize,
    #[cfg(feature = "lifecycle-test-support")]
    fixture_chunks_left: usize,
    payload: Option<RawRegistration>,
    retained: Option<Arc<RootedRawHost>>,
    stage: HostStage,
    retire_attempted: bool,
    retire_reserved: bool,
    reader_claimed: bool,
    reader_live: bool,
    reader_failed: bool,
}
struct RootedRawHost {
    state: std::sync::Mutex<RawHostState>,
}

/// Readiness is a typed lifecycle result, not disk/lease authority. Ready means
/// the exact rooted host has atomically stopped admission pending cleanup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) enum RetirementReadiness { Busy, Ready }
impl RetirementReadiness {
    pub(in crate::replicated_decoder) fn is_ready(self) -> bool { self==Self::Ready }
}

/// Filesystem-free opaque preparation result, NOT disk authority. No Clone,
/// receipt, callback, replacement boolean, or compiler-to-Python dependency.
///
/// PRIVATE HOST API FOR THE NEXT BRIDGE (not production-wired):
/// 1. Adopt this exact holder into a persistent genuine DiskLease owner first.
/// 2. Under that existing admitted lease call private start(device); device is
///    only a filesystem predicate, never proof of admission. No bound-backing
///    credit during ingestion: keep the full disk reservation UNBOUND.
/// 3. append/finish borrow the holder. Every terminal failure latches, retaining
///    completed files, pins, header, directory and original cumulative debits.
/// 4. Under the lease lock, begin_retire first: Busy changes nothing; Ready
///    reserves the exact holder before clearing any future backing credit.
///    retire consumes that reservation, burns writers/files, then removes the
///    directory. Failure must retain BOTH this holder and the real outer lease;
///    never exit/GC/refund that lease and never retry removal.
/// 5. ONLY after successful retire and actual lease exit exactly once call
///    release_final_charges. These two operations deliberately remain separate.
///
    /// The private host adapter enforces that order. Native code cannot certify
    /// a Python lease; device is a predicate, never caller-minted disk authority.
pub(in crate::replicated_decoder) struct RawHost {
    root: Arc<RootedRawHost>,
}

/// Selects an ORIGINAL graph; neither this selector nor an operation ordinal is
/// source authority. Only the sealed, single-claim reader supplies that custody.
#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) enum RawGraph { Prefill, Decode }

/// Non-Clone original-root retention. Every tensor/window borrows this owner;
/// therefore its last descendant dies before reader_live can be cleared.
/// No constructor accepts a file, label, digest, declaration or caller buffer.
pub(in crate::replicated_decoder) struct RawReader {
    root: Arc<RootedRawHost>,
}

/// Original semantic operation and ExpectedTensor index, never a detached file.
pub(in crate::replicated_decoder) struct RawTensor<'r> {
    reader: &'r RawReader,
    graph: RawGraph,
    operation: usize,
    tensor: usize,
    shape: [usize; 2],
    charge: OwnedExternalCharge,
}

/// Numeric bytes alone carry NO source authority. The window retains its whole
/// semantic tensor and reader until all bytes have been wiped, before refund.
pub(in crate::replicated_decoder) struct RawWindow<'t, 'r> {
    source: &'t RawTensor<'r>,
    bytes: [u8; FRAME],
    len: usize,
    charge: OwnedExternalCharge,
}

// One claim transition, original header/directory/owner checks, flag publication
// and eventual Drop flag clear. Logical control work, not measured CPU fuel.
const READER_CLAIM_WORK: u64 = 64;
fn reader_control_bytes() -> Result<usize, Error> {
    [mul(3,size_of::<RawReader>())?,
        mul(2,size_of::<Result<RawReader,Error>>())?,
        mul(2,size_of::<std::sync::MutexGuard<'_,RawHostState>>())?,
        size_of::<OwnedExternalCharge>()]
        .into_iter().try_fold(0,plus)
}

// Post-registration work is NOT drawn from the exhausted registration quote.
// Reserve against the actual original counter and debit the actual original
// ledger before any search, read, zero-fill, copy or destination publication.
fn admit_reader(payload: &RawRegistration, bytes: usize, work: u64) -> Result<OwnedExternalCharge,Error> {
    let header=&payload.header;
    header.original.prepay_primitive_work(&header.controller,work).map_err(|_|Error::Bounds)?;
    let peak=plus(header.root.budget().used_bytes(),plus(bytes,size_of::<OwnedExternalCharge>())?)?;
    if peak>MAX_PROGRAM_BYTES { return Err(Error::Bounds); }
    header.root.budget().admit_peak(peak).map_err(|_|Error::Bounds)?;
    header.root.charge_external(bytes).map_err(|_|Error::Bounds)
}

fn reader_payload(state: &RawHostState) -> Result<&RawRegistration,Error> {
    if state.stage!=HostStage::Sealed || state.retire_attempted || state.retire_reserved || !state.reader_live || state.reader_failed {
        return Err(Error::Retired);
    }
    state.payload.as_ref().ok_or(Error::Retired)
}

// Original-admission slice: private DEMAND and immutable owner-derived identity,
// not a numeric profile, source constructor or detached reservation authority.
#[derive(Clone, Copy)]
struct PrimitiveQuote { work: u64, resident_bytes: usize }
impl PrimitiveQuote {
    fn checked(work: u64, resident_bytes: usize) -> Result<Self,Error> {
        if resident_bytes>MAX_PROGRAM_BYTES { return Err(Error::Bounds); }
        Ok(Self {work,resident_bytes})
    }
    fn include(self, work: u64, bytes: usize) -> Result<Self,Error> {
        Self::checked(self.work.checked_add(work).ok_or(Error::Bounds)?,plus(self.resident_bytes,bytes)?)
    }
}
struct OriginalOperationKey { graph: RawGraph, operation: usize, tensor: usize }

fn primitive_admission_quote(payload: &RawRegistration, key: &OriginalOperationKey,
    demand: PrimitiveQuote) -> Result<PrimitiveQuote,Error> {
    let plan=payload.header.original.plan();
    let operations=match key.graph { RawGraph::Prefill=>plan.prefill.operations.len(),
        RawGraph::Decode=>plan.decode.operations.len() };
    let premise=payload.header.raw_premise.as_ref().ok_or(Error::Binding)?;
    // Counts only, not traversals. Covers owner validation AND concrete lookup
    // revalidation, complete bounded inventory/backing scans and cleanup. Work
    // is logical original-ledger work, not measured native CPU instructions.
    let visits=[mul(operations,4*NAME+64)?,mul(premise.tensors.len(),NAME+1)?,
        mul(payload.files.len(),64)?,1024].into_iter().try_fold(0,plus)?;
    let work=u64::try_from(mul(2,visits)?).map_err(|_|Error::Bounds)?;
    let controls=[mul(2,size_of::<std::sync::MutexGuard<'_,RawHostState>>())?,
        mul(2,size_of::<OriginalOperationKey>())?,mul(3,size_of::<PrimitiveQuote>())?,
        mul(2,size_of::<OwnedExternalCharge>())?,mul(4,size_of::<Arc<PublicFile>>())?,
        mul(2,size_of::<Arc<RootedRawHost>>())?,mul(16,size_of::<u64>())?]
        .into_iter().try_fold(0,plus)?;
    let work=work.checked_add(u64::try_from(controls).map_err(|_|Error::Bounds)?).ok_or(Error::Bounds)?;
    demand.include(work,controls)
}

// Owner-INTERNAL only. The charge is immediately enclosed with the concrete
// producer; no callable path exports/re-attaches it as source authority.
fn admit_primitive(reader: &RawReader, key: &OriginalOperationKey,
    demand: PrimitiveQuote) -> Result<OwnedExternalCharge,Error> {
    let mut state=reader.root.state.lock().map_err(|_|Error::Binding)?;
    let quote=primitive_admission_quote(reader_payload(&state)?,key,demand)?;
    state.reader_failed=true;
    let payload=state.payload.as_ref().ok_or(Error::Retired)?;
    // Work -> original peak/cap -> original charge, BEFORE semantic traversal.
    let charge=admit_reader(payload,quote.resident_bytes,quote.work)?;
    let name=original_weight(payload,key.graph,key.operation)?;
    let premise=payload.header.raw_premise.as_ref().ok_or(Error::Binding)?;
    let selected=premise.tensors.iter().position(|t|t.name==name).ok_or(Error::Binding)?;
    if selected!=key.tensor || premise.tensors.get(key.tensor).ok_or(Error::Binding)?.rank!=2
        || payload.files.len()!=premise.files.len() { return Err(Error::Binding); }
    for (backing,expected) in payload.files.iter().zip(&premise.files) {
        let original=backing.original.upgrade().ok_or(Error::Binding)?;
        let file=backing.file.as_ref().ok_or(Error::Binding)?;
        if !Arc::ptr_eq(&original,file) || file.size()!=expected.size { return Err(Error::Binding); }
    }
    let expected=&premise.tensors[key.tensor];
    let file=payload.files.get(expected.file).and_then(|b|b.file.as_ref()).ok_or(Error::Binding)?;
    let start=data_start(file)?;
    let begin=start.checked_add(expected.begin).ok_or(Error::Bounds)?;
    let end=start.checked_add(expected.end).ok_or(Error::Bounds)?;
    if begin>=end || end>file.size() { return Err(Error::Bounds); }
    state.reader_failed=false;
    Ok(charge)
}

fn original_weight(payload: &RawRegistration, graph: RawGraph, operation: usize) -> Result<&str,Error> {
    check_header(&payload.header,&payload.directory,&payload.identity)?;
    let plan=payload.header.original.plan();
    let graph=match graph { RawGraph::Prefill=>&plan.prefill, RawGraph::Decode=>&plan.decode };
    let op=graph.operations.get(operation).ok_or(Error::Binding)?;
    if !matches!(op.operator,pllm_models::ModelOperator::TokenLookup
        | pllm_models::ModelOperator::Linear | pllm_models::ModelOperator::OutputHead) {
        return Err(Error::Binding);
    }
    op.attributes.get("weight").and_then(|s|s.as_str()).ok_or(Error::Binding)
}

impl RawReader {
    pub(in crate::replicated_decoder) fn weight(&self, graph: RawGraph, operation: usize) -> Result<RawTensor<'_>,Error> {
        let mut state=self.root.state.lock().map_err(|_|Error::Binding)?;
        let payload=reader_payload(&state)?;
        let count=payload.header.raw_premise.as_ref().ok_or(Error::Binding)?.tensors.len();
        // Scan at most the full sealed inventory, including long equal prefixes.
        let work=u64::try_from(plus(mul(count,NAME+1)?,128)?).map_err(|_|Error::Bounds)?;
        state.reader_failed=true; // any admitted attempt consumes on late failure
        let payload=state.payload.as_ref().ok_or(Error::Retired)?;
        let charge=admit_reader(payload,plus(mul(3,size_of::<RawTensor<'_>>())?,
            mul(2,size_of::<Result<RawTensor<'_>,Error>>())?)?,work)?;
        let name=original_weight(payload,graph,operation)?;
        let tensors=&payload.header.raw_premise.as_ref().ok_or(Error::Binding)?.tensors;
        let tensor=tensors.iter().position(|t|t.name==name).ok_or(Error::Binding)?;
        let expected=&tensors[tensor];
        if expected.rank!=2 { return Err(Error::Shape); }
        let shape=expected.shape;
        state.reader_failed=false;
        Ok(RawTensor {reader:self,graph,operation,tensor,shape,charge})
    }
}

impl Drop for RawReader {
    fn drop(&mut self) {
        // Poisoned mutex still owns its exact root. Drop may clear only liveness;
        // it must not remove files, break the retention cycle or refund charges.
        let mut state=self.root.state.lock().unwrap_or_else(|e|e.into_inner());
        state.reader_live=false;
    }
}

impl<'r> RawTensor<'r> {
    pub(in crate::replicated_decoder) fn shape(&self) -> [usize;2] { self.shape }

    pub(in crate::replicated_decoder) fn read(&self, offset: u64, len: usize) -> Result<RawWindow<'_, 'r>,Error> {
        let mut state=self.reader.root.state.lock().map_err(|_|Error::Binding)?;
        reader_payload(&state)?;
        state.reader_failed=true;
        let payload=state.payload.as_ref().ok_or(Error::Retired)?;
        // Full fixed window zero-fill, payload I/O, result/move overlap and full
        // Drop wipe are paid even on a late invalid extent or failed final read.
        let charge=admit_reader(payload,raw_window_bytes()?,RAW_WINDOW_WORK)?;
        if len==0 || len>FRAME { return Err(Error::Bounds); }
        let name=original_weight(payload,self.graph,self.operation)?;
        let expected=payload.header.raw_premise.as_ref().ok_or(Error::Binding)?
            .tensors.get(self.tensor).ok_or(Error::Binding)?;
        if expected.name!=name || expected.rank!=2 || expected.shape!=self.shape { return Err(Error::Binding); }
        let backing=payload.files.get(expected.file).ok_or(Error::Binding)?;
        let original=backing.original.upgrade().ok_or(Error::Binding)?;
        let file=backing.file.as_ref().ok_or(Error::Binding)?;
        if !Arc::ptr_eq(&original,file) { return Err(Error::Binding); }
        let relative_end=offset.checked_add(len as u64).ok_or(Error::Bounds)?;
        let extent=expected.end.checked_sub(expected.begin).ok_or(Error::Bounds)?;
        if relative_end>extent { return Err(Error::Bounds); }
        let begin=data_start(file)?.checked_add(expected.begin)
            .and_then(|n|n.checked_add(offset)).ok_or(Error::Bounds)?;
        if begin.checked_add(len as u64).is_none_or(|end|end>file.size()) { return Err(Error::Bounds); }
        let mut window=RawWindow {source:self,bytes:[0;FRAME],len,charge};
        #[cfg(test)]
        WINDOW_FACTORIES.with(|n|n.set(n.get()+1));
        file.read(begin,&mut window.bytes[..len]).map_err(|_|Error::Binding)?;
        state.reader_failed=false;
        Ok(window)
    }
}

impl Drop for RawTensor<'_> {
    fn drop(&mut self) {
        // Explicit Drop keeps the reader borrow live through destruction (drop
        // checking), even when this tensor will perform no further reads. Its
        // reservation drops afterwards; there is no detached payload to erase.
    }
}

impl RawWindow<'_, '_> {
    /// Borrowed numeric representation only. Cannot mint a producer or replace
    /// this window's original operation/source custody in any downstream API.
    pub(in crate::replicated_decoder) fn bytes(&self) -> &[u8] { &self.bytes[..self.len] }
}
impl Drop for RawWindow<'_, '_> {
    fn drop(&mut self) {
        self.bytes.fill(0);
        #[cfg(test)]
        WINDOW_DROP.with(|s|s.set(Some((self.bytes.iter().all(|b|*b==0),self.charge.budget().used_bytes()))));
    }
}

// Fixed-capacity window, result and move slots are charged in full, not len.
// Old returned windows retain their own charges throughout the next read.
const RAW_WINDOW_WORK: u64 = (8*FRAME+NAME+256) as u64;
fn raw_window_bytes() -> Result<usize,Error> {
    [mul(3,size_of::<RawWindow<'_, '_>>())?,
        mul(2,size_of::<Result<RawWindow<'_, '_>,Error>>())?,
        size_of::<std::sync::MutexGuard<'_,RawHostState>>(),
        size_of::<[u8;8]>(), mul(2,size_of::<Arc<PublicFile>>())?,
        mul(4,size_of::<u64>())?].into_iter().try_fold(0,plus)
}

fn host_control_bytes() -> Result<usize, Error> {
    fn arc_bytes<T>() -> Result<usize, Error> {
        let (layout, _) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<T>()).map_err(|_| Error::Bounds)?;
        Ok(layout.pad_to_align().size())
    }
    // Named stack-to-owner move overlap, plus the actual Arc allocations and
    // whole-header custody handles. No anonymous capacity padding.
    [mul(2, size_of::<RawRegistration>())?, mul(2, size_of::<RawHostState>())?,
        mul(2, size_of::<RawHost>())?, mul(2, size_of::<RawPending>())?,
        mul(2, size_of::<std::sync::MutexGuard<'_,RawHostState>>())?,
        mul(2, size_of::<Result<RetirementReadiness,Error>>())?,
        arc_bytes::<RootedRawHost>()?, arc_bytes::<SourceHeaderEntry>()?]
        .into_iter().try_fold(0, plus)
}

#[cfg(feature = "lifecycle-test-support")]
const LIFECYCLE_OBSERVATIONS: usize = 128;
#[cfg(feature = "lifecycle-test-support")]
// Four ten-scalar snapshot moves, three digest moves, fixed lock/control work.
const LIFECYCLE_OBSERVATION_WORK: usize = 4 * 10 + 3 * 32 + 64;
#[cfg(feature = "lifecycle-test-support")]
const LIFECYCLE_CHUNKS: usize = 3 * (JSON_BYTES / 17 + 1);

#[cfg(feature = "lifecycle-test-support")]
fn lifecycle_storage_bytes() -> Result<usize, Error> {
    use crate::replicated_decoder::source::Publisher;
    // The real feature-enabled Publisher layout includes Option<RawReader>,
    // with compiler-selected padding/alignment. Named value/result/move slots;
    // no assumed CPython/PyCell layout or allocator/RSS claim. RawRegistration
    // and RawHostState's new inline fields also enter host_control_bytes above.
    [mul(3, std::alloc::Layout::new::<Publisher>().pad_to_align().size())?,
        mul(2, std::alloc::Layout::new::<Option<Publisher>>().pad_to_align().size())?,
        mul(2, size_of::<Result<Publisher, String>>())?,
        mul(4, size_of::<[usize; 10]>())?,
        mul(2, size_of::<Result<[usize; 10], Error>>())?,
        mul(3, plus(size_of::<(String, Vec<u8>)>(), plus(NAME, JSON_BYTES)?)?)?,
        mul(3, size_of::<Vec<(String, Vec<u8>)>>())?,
        mul(3, size_of::<[u8; 17]>())?,
        mul(2, size_of::<Result<([u8; 17], usize), Error>>())?,
        mul(3, size_of::<(u64, [u8; 32], usize)>())?,
        mul(2, size_of::<Result<(u64, [u8; 32], usize), Error>>())?]
        .into_iter().try_fold(0, plus)
}

impl RawHost {
    pub(in crate::replicated_decoder) fn prepare(header: RawPending) -> Result<Self, Error> {
        let payload = RawRegistration::prepare(header)?;
        Ok(Self { root: Arc::new(RootedRawHost { state: std::sync::Mutex::new(RawHostState {
            #[cfg(feature = "lifecycle-test-support")]
            observations_left: LIFECYCLE_OBSERVATIONS,
            #[cfg(feature = "lifecycle-test-support")]
            fixture_chunks_left: LIFECYCLE_CHUNKS,
            payload: Some(payload), retained: None, stage: HostStage::Pending,
            retire_attempted: false, retire_reserved: false,
            reader_claimed: false, reader_live: false, reader_failed: false,
        }) }) })
    }

    #[cfg(feature = "lifecycle-test-support")]
    pub(in crate::replicated_decoder) fn test_fixture() -> Result<Self, Error> {
        // External synthetic trust construction, not candidate-derived approval.
        // All exact lengths, digests, descriptors and D31 premises freeze here,
        // before prepare and before Python can append its first candidate byte.
        let fixture = raw_fixture::Fixture::try_positive(raw_fixture::CONFIG,false)?;
        if fixture.files.len() != 3 || fixture.files.capacity() != 3
            || fixture.tensors.len() > 64
            || fixture.files.iter().any(|(name, bytes)| name.capacity() > NAME || bytes.capacity() > JSON_BYTES)
        { return Err(Error::Bounds); }
        Self::prepare(fixture.header_result()?)
    }

    #[cfg(feature = "lifecycle-test-support")]
    fn observe_slot(state: &mut RawHostState) -> Result<(), Error> {
        // Work prepaid at ORIGINAL registration preparation, never debited by
        // Busy or by these observations. Limit calls rather than reset ledger.
        state.observations_left = state.observations_left.checked_sub(1).ok_or(Error::Bounds)?;
        Ok(())
    }

    #[cfg(feature = "lifecycle-test-support")]
    pub(in crate::replicated_decoder) fn test_snapshot(&self) -> Result<[usize; 10], Error> {
        let mut state = self.root.state.lock().map_err(|_| Error::Binding)?;
        Self::observe_slot(&mut state)?;
        let (used, collection, reader, files, descriptors) = match state.payload.as_ref() {
            Some(p) => (p.header.root.budget().used_bytes(), p.header.collection_charge.charged_bytes(),
                p.reader_charge.as_ref().map_or(0, |c| c.charged_bytes()), p.files.len(),
                p.header.raw_premise.as_ref().ok_or(Error::Binding)?.tensors.len()),
            None => (0, 0, 0, 0, 0),
        };
        let stage = match state.stage { HostStage::Pending => 0, HostStage::Collecting => 1,
            HostStage::Sealed => 2, HostStage::Failed => 3, HostStage::DirectoryRetired => 4, HostStage::Released => 5 };
        Ok([stage, used, collection, reader, files, usize::from(state.reader_live),
            usize::from(state.retire_reserved), usize::from(state.retire_attempted),
            usize::from(state.retained.is_some()), descriptors])
    }

    #[cfg(feature = "lifecycle-test-support")]
    pub(in crate::replicated_decoder) fn test_inventory(&self, file: usize) -> Result<(u64, [u8; 32], usize), Error> {
        let mut state = self.root.state.lock().map_err(|_| Error::Binding)?;
        Self::observe_slot(&mut state)?;
        let p = state.payload.as_ref().ok_or(Error::Retired)?;
        let premise = p.header.raw_premise.as_ref().ok_or(Error::Binding)?;
        let f = premise.files.get(file).ok_or(Error::Bounds)?;
        Ok((f.size, f.digest, premise.tensors.len()))
    }

    #[cfg(feature = "lifecycle-test-support")]
    pub(in crate::replicated_decoder) fn test_chunk(&self, file: usize, offset: usize) -> Result<([u8; 17], usize), Error> {
        let mut state = self.root.state.lock().map_err(|_| Error::Binding)?;
        state.fixture_chunks_left = state.fixture_chunks_left.checked_sub(1).ok_or(Error::Bounds)?;
        let p = state.payload.as_ref().ok_or(Error::Retired)?;
        let files = &p.header.fixture.as_ref().ok_or(Error::Binding)?.files;
        let bytes = &files.get(file).ok_or(Error::Bounds)?.1;
        let remaining = bytes.get(offset..).ok_or(Error::Bounds)?;
        let n = remaining.len().min(17);
        let mut chunk = [0; 17]; chunk[..n].copy_from_slice(&remaining[..n]);
        Ok((chunk, n))
    }

    pub(in crate::replicated_decoder) fn disk_bytes(&self) -> Result<usize, Error> {
        let state = self.root.state.lock().map_err(|_| Error::Binding)?;
        let payload = state.payload.as_ref().ok_or(Error::Retired)?;
        payload.header.raw_premise.as_ref().ok_or(Error::Binding)?.files.iter()
            .try_fold(0, |n, file| plus(n, usize::try_from(file.size).map_err(|_| Error::Bounds)?))
    }

    pub(in crate::replicated_decoder) fn disk_directory(&self) -> Result<&'static std::path::Path, Error> {
        let state = self.root.state.lock().map_err(|_| Error::Binding)?;
        if state.payload.is_none() { return Err(Error::Retired); }
        snapshot_directory::parent()
    }

    pub(in crate::replicated_decoder) fn start(&self, device: u64) -> Result<(), Error> {
        let mut state = self.root.state.lock().map_err(|_| Error::Binding)?;
        if state.stage != HostStage::Pending || state.retire_attempted || state.retire_reserved {
            self.reject_ingestion(&mut state);
            return Err(Error::Retired);
        }
        // Install the root BEFORE the first fallible filesystem operation. Even
        // dropping the last external handle cannot invoke automatic cleanup.
        state.retained = Some(Arc::clone(&self.root));
        state.stage = HostStage::Failed;
        state.payload.as_mut().ok_or(Error::Retired)?.start(device)?;
        state.stage = HostStage::Collecting;
        Ok(())
    }

    pub(in crate::replicated_decoder) fn append(&self, file: usize, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.append_with(|payload|payload.append(file,offset,bytes))
    }

    // Also lets unit fixtures borrow generated bytes under this SAME mutex and
    // covering header. No payload escapes while another handle could retire it.
    fn append_with(&self, append:impl FnOnce(&mut RawRegistration)->Result<(),Error>) -> Result<(),Error> {
        let mut state = self.root.state.lock().map_err(|_| Error::Binding)?;
        if state.stage != HostStage::Collecting || state.retire_attempted || state.retire_reserved {
            self.reject_ingestion(&mut state);
            return Err(Error::Retired);
        }
        state.stage = HostStage::Failed;
        append(state.payload.as_mut().ok_or(Error::Retired)?)?;
        state.stage = HostStage::Collecting;
        Ok(())
    }

    pub(in crate::replicated_decoder) fn finish(&self) -> Result<(), Error> {
        let mut state = self.root.state.lock().map_err(|_| Error::Binding)?;
        if state.stage != HostStage::Collecting || state.retire_attempted || state.retire_reserved {
            self.reject_ingestion(&mut state);
            return Err(Error::Retired);
        }
        state.stage = HostStage::Failed;
        state.payload.as_ref().ok_or(Error::Retired)?.finish()?;
        state.stage = HostStage::Sealed;
        Ok(())
    }

    pub(in crate::replicated_decoder) fn claim_reader(&self) -> Result<RawReader,Error> {
        let mut state=self.root.state.lock().map_err(|_|Error::Binding)?;
        if state.stage!=HostStage::Sealed || state.retire_attempted || state.retire_reserved || state.reader_claimed {
            return Err(Error::Retired);
        }
        state.reader_claimed=true; // failure cannot reopen the consuming claim
        let payload=state.payload.as_mut().ok_or(Error::Retired)?;
        let charge=admit_reader(payload,reader_control_bytes()?,READER_CLAIM_WORK)?;
        check_header(&payload.header,&payload.directory,&payload.identity)?;
        payload.reader_charge=Some(charge);
        #[cfg(test)]
        READER_FACTORIES.with(|n|n.set(n.get()+1));
        state.reader_live=true;
        Ok(RawReader {root:Arc::clone(&self.root)})
    }

    fn reject_ingestion(&self, state: &mut RawHostState) {
        if matches!(state.stage, HostStage::Pending | HostStage::Collecting | HostStage::Sealed) {
            state.retained = Some(Arc::clone(&self.root));
            state.stage = HostStage::Failed;
        }
    }

    pub(in crate::replicated_decoder) fn begin_retire(&self) -> Result<RetirementReadiness,Error> {
        let mut state = self.root.state.lock().map_err(|_| Error::Binding)?;
        self.begin_retire_locked(&mut state)
    }

    fn begin_retire_locked(&self, state: &mut RawHostState) -> Result<RetirementReadiness,Error> {
        if state.retire_reserved || state.retire_attempted
            || matches!(state.stage,HostStage::DirectoryRetired|HostStage::Released) {
            return Err(Error::Retired);
        }
        // SAME mutex as reader claim. Busy has no root, flag, stage, writer,
        // file, directory, ledger or charge mutation; caller may retry later.
        if state.reader_live { return Ok(RetirementReadiness::Busy); }
        // Pending hosts have no start/rejection root yet. Retain custody before
        // reservation publication. No allocation: clone the already charged Arc.
        state.retained = Some(Arc::clone(&self.root));
        state.retire_reserved = true;
        state.stage = HostStage::Failed;
        Ok(RetirementReadiness::Ready)
    }

    pub(in crate::replicated_decoder) fn retire(&self) -> Result<(), Error> {
        let mut state = self.root.state.lock().map_err(|_| Error::Binding)?;
        if state.stage == HostStage::DirectoryRetired { return Ok(()); }
        if state.retire_attempted || state.stage == HostStage::Released { return Err(Error::Retired); }
        // Preserve direct-native cleanup: reserve AND consume under this same
        // critical section. Never bypass Busy or permit claims between them.
        if !state.retire_reserved && self.begin_retire_locked(&mut state)?==RetirementReadiness::Busy {
            return Err(Error::Binding);
        }
        if state.reader_live { return Err(Error::Binding); }
        state.retire_reserved = false; // consume once; failure cannot reopen it
        state.retire_attempted = true;
        state.stage = HostStage::Failed; // stop ingestion BEFORE dropping files
        let payload = state.payload.as_mut().ok_or(Error::Retired)?;
        drop(payload.writer.take()); // incomplete backing burn is permitted
        payload.files.clear();
        payload.directory.retire()?;
        state.stage = HostStage::DirectoryRetired;
        Ok(())
    }

    pub(in crate::replicated_decoder) fn release_final_charges(&self) -> Result<(), Error> {
        let mut state = self.root.state.lock().map_err(|_| Error::Binding)?;
        if state.stage != HostStage::DirectoryRetired { return Err(Error::Retired); }
        // Future adapter must have exited the real lease BEFORE entering here.
        // No consuming Result API can accidentally lose custody on Err.
        drop(state.payload.take());
        drop(state.retained.take());
        state.stage = HostStage::Released;
        Ok(())
    }
}
struct RawBacking {
    file: Option<Arc<PublicFile>>,
    original: std::sync::Weak<PublicFile>,
    #[cfg(test)]
    observe: Option<(OwnedBudget, usize)>,
}
impl std::ops::Deref for RawBacking {
    type Target=PublicFile;
    fn deref(&self) -> &Self::Target { self.file.as_ref().expect("live private raw backing") }
}
impl Drop for RawBacking {
    fn drop(&mut self) {
        drop(self.file.take());
        #[cfg(test)]
        if let Some((counter,cover))=&self.observe {
            BACKING_DROP.with(|s| s.set(Some((self.original.upgrade().is_none(),counter.budget().used_bytes(),*cover))));
        }
    }
}
#[cfg(test)]
thread_local! { static BACKING_DROP:std::cell::Cell<Option<(bool,usize,usize)>>=const {std::cell::Cell::new(None)}; }
#[cfg(test)]
thread_local! { static WRITER_FACTORIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)}; }
#[cfg(test)]
thread_local! { static READER_FACTORIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)}; }
#[cfg(test)]
thread_local! { static WINDOW_FACTORIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)}; }
#[cfg(test)]
thread_local! { static WINDOW_DROP:std::cell::Cell<Option<(bool,usize)>>=const {std::cell::Cell::new(None)}; }
fn public_arc_bytes() -> Result<usize, Error> {
    let (layout,_) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize;2]>()
        .extend(std::alloc::Layout::new::<PublicFile>()).map_err(|_| Error::Bounds)?;
    Ok(layout.pad_to_align().size())
}
impl RawRegistration {
    fn prepare(header: RawPending) -> Result<Self, Error> {
        header.check()?;
        let premise = header.raw_premise.as_ref().ok_or(Error::Binding)?;
        premise.check(&header)?;
        let (bytes, work, count) = (premise.phase_bytes, premise.work, premise.files.len());
        // Frozen capsule's covering charge is sized before freeze, work burns
        // against its ORIGINAL source plan; never clone/refund prepared state.
        if header.collection_charge.payload_bytes() < bytes { return Err(Error::Bounds); }
        header.original.prepay_compiler_metadata(&header.controller, work).map_err(|_| Error::Bounds)?;
        let files = exact_vec(count)?;
        let directory = Arc::clone(&header.directory);
        let identity=Arc::downgrade(&directory);
        Ok(Self {
            files, writer: None, cursor: 0, offset: 0, identity, directory, header, reader_charge: None })
    }
    fn start(&mut self, device: u64) -> Result<(), Error> {
        self.header.check()?;
        self.directory.start(device)?;
        self.writer = Some(start_writer(&self.header, &self.directory, 0)?);
        Ok(())
    }
    fn append(&mut self, file: usize, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        let premise = self.header.raw_premise.as_ref().ok_or(Error::Binding)?;
        check_header(&self.header,&self.directory,&self.identity)?;
        let expected = premise.files.get(self.cursor).ok_or(Error::Binding)?;
        let end = offset.checked_add(bytes.len() as u64).ok_or(Error::Bounds)?;
        if file != self.cursor || offset != self.offset || bytes.is_empty() || bytes.len() > public_file::MAX_CHUNK_BYTES || end > expected.size { return Err(Error::Binding); }
        self.writer.as_mut().ok_or(Error::Retired)?.write(bytes).map_err(|_| Error::Binding)?;
        self.offset = end;
        if end == expected.size {
            let file = self.writer.take().ok_or(Error::Retired)?.finish().map_err(|_| Error::Binding)?;
            let file=Arc::new(file);
            let original=Arc::downgrade(&file);
            self.files.push(RawBacking { file:Some(file), original,
                #[cfg(test)] observe:None,
            }); // exact reserved inventory, no growth
            self.cursor += 1; self.offset = 0;
            if self.cursor < premise.files.len() { self.writer = Some(start_writer(&self.header, &self.directory, self.cursor)?); }
        }
        Ok(())
    }
    fn finish(&self) -> Result<(), Error> {
        let premise = self.header.raw_premise.as_ref().ok_or(Error::Binding)?;
        check_header(&self.header,&self.directory,&self.identity)?;
        if self.cursor != premise.files.len() || self.offset != 0 || self.writer.is_some() || self.files.len()!=premise.files.len() { return Err(Error::Binding); }
        // No tensor subrange has escaped. Every file, including unselected bytes,
        // is now complete and independently expected-digest verified.
        validate_backings(&self.header, &self.files)?;
        validate_inventory(&self.header, &self.files)?;
        Ok(())
    }
}

fn check_header(header: &RawPending, directory: &Arc<snapshot_directory::RawSnapshotDirectory>, identity: &std::sync::Weak<snapshot_directory::RawSnapshotDirectory>) -> Result<(), Error> {
    header.check()?;
    let original=identity.upgrade().ok_or(Error::Binding)?;
    if !Arc::ptr_eq(&original,directory) { return Err(Error::Binding); }
    directory.check_budget(header.original.budget())
}

fn start_writer(header: &SourceHeaderEntry, directory: &Arc<snapshot_directory::RawSnapshotDirectory>, file: usize) -> Result<PublicFileWriter, Error> {
    let p = header.raw_premise.as_ref().ok_or(Error::Binding)?; p.check(header)?;
    let f = p.files.get(file).ok_or(Error::Binding)?;
    let directory: Arc<dyn SnapshotDirectory> = Arc::clone(directory) as Arc<dyn SnapshotDirectory>;
    #[cfg(test)]
    WRITER_FACTORIES.with(|n| n.set(n.get()+1));
    PublicFileWriter::new_in(f.size, f.digest, directory).map_err(|_| Error::Binding)
}
fn validate_backings(header: &SourceHeaderEntry, files: &[RawBacking]) -> Result<(), Error> {
    let p = header.raw_premise.as_ref().ok_or(Error::Binding)?; p.check(header)?;
    if files.len()!=p.files.len() { return Err(Error::Binding); }
    for (file, expected) in files.iter().zip(&p.files) {
        let original=file.original.upgrade().ok_or(Error::Binding)?;
        if !Arc::ptr_eq(&original,file.file.as_ref().ok_or(Error::Binding)?) { return Err(Error::Binding); }
        if file.size()!=expected.size || file.digest()!=expected.digest { return Err(Error::Binding); }
    }
    Ok(())
}
fn mul(a: usize, b: usize) -> Result<usize, Error> { a.checked_mul(b).ok_or(Error::Bounds) }
fn exact_vec<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new(); v.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    if v.capacity()!=n { return Err(Error::Bounds); } Ok(v)
}
fn valid_name(s: &str) -> bool { !s.is_empty() && s.len()<=NAME && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) }
fn copy_text(s: &str) -> Result<String, Error> {
    let mut v = String::new(); v.try_reserve_exact(s.len()).map_err(|_| Error::Allocation)?;
    if v.capacity()!=s.len() { return Err(Error::Bounds); } v.push_str(s); Ok(v)
}
fn semantic_identity(plan: &pllm_models::DecoderPlan) -> Result<[u8;32], Error> {
    let limits = canonical::Limits {
        canonical: StreamLimits { max_nodes: 1<<20, max_depth: 16, max_bytes: 1<<24, max_string_bytes: NAME,
            max_key_bytes: NAME, max_total_string_bytes: 1<<20, max_work: CANONICAL_WORK,
            max_stack_bytes: size_of::<[StreamFrame<'_>;16]>() },
        max_operations: source_policy::DESCRIPTORS*2, max_states: source_policy::DESCRIPTORS*4,
        max_inputs: source_policy::DESCRIPTORS*16, max_rank: 5, max_transformations: 0,
        max_domain_bytes: pllm_models::DECODER_PLAN_SCHEMA_VERSION.len(),
    };
    canonical::bounded_plan_digest(plan, limits, &mut [StreamFrame::empty();16]).map_err(|_| Error::Binding)
}
fn semantic_weight(header: &SourceHeaderEntry, operator: pllm_models::ModelOperator) -> Result<&str, Error> {
    header.original.plan().prefill.operations.iter().find(|o| o.operator==operator)
        .and_then(|o| o.attributes.get("weight")).and_then(|s| s.as_str()).ok_or(Error::Binding)
}
fn parse_borrowed_configuration(header: &mut SourceHeaderEntry) -> Result<OwnedCharged<json::ImportedValue>, Error> {
    // Existing admitted JSON owner; its additional charge overlaps collection.
    // Prepay its complete existing work cap on the ORIGINAL ledger as well.
    header.admit_collection(0, header.collection_charge.payload_bytes(), 1_000_000_000)?;
    let mut bytes=exact_vec(header.original.raw_configuration().len())?;
    bytes.extend_from_slice(header.original.raw_configuration());
    import_raw_json(header,bytes)
}

/// Reuse C9 quote/construct unchanged, retaining its original source owner and
/// guards. Check the actual quoted old/new peak against the unchanged program
/// cap BEFORE its factories, in addition to the original-counter work prepay.
fn import_raw_json(header: &SourceHeaderEntry, bytes: Vec<u8>) -> Result<OwnedCharged<json::ImportedValue>, Error> {
    let members=source_members(&bytes);
    let baseline=header.root.budget().used_bytes();
    let bootstrap=json::bootstrap_bytes().map_err(|_| Error::Bounds)?;
    let start=plus(baseline,plus(bytes.len(),plus(bootstrap,plus(size_of::<OwnedCharged<Vec<u8>>>(),size_of::<OwnedExternalCharge>())?)?)?)?;
    if start>MAX_PROGRAM_BYTES { return Err(Error::Bounds); }
    header.root.budget().admit_peak(start).map_err(|_| Error::Bounds)?;
    let source=header.root.charge_external(bytes.len()).map_err(|_| Error::Bounds)?
        .try_build(|_| Ok(bytes)).map_err(|_| Error::Bounds)?;
    let quote=json::quote(source,header.root.charge_external(bootstrap).map_err(|_| Error::Bounds)?,json_limits()).map_err(|_| Error::Binding)?;
    let peak=plus(baseline,quote.resources().original_owners_peak_bytes)?;
    if peak>MAX_PROGRAM_BYTES { return Err(Error::Bounds); }
    header.root.budget().admit_peak(peak).map_err(|_| Error::Bounds)?;
    let value=quote.construct().map_err(|_| Error::Binding)?;
    if members!=retained_members(value.payload().value()) { return Err(Error::Binding); }
    Ok(value)
}

// C9 JSON intentionally has last-value-wins duplicate semantics. Keep that
// existing API unchanged. This raw consumer additionally compares the number of
// source ':' tokens outside strings against ALL retained object members. Any
// duplicate discards at least one member/subtree, so counts differ, even for
// escaped equivalent keys or duplicates nested in arrays. Syntax is C9's job.
fn source_members(bytes: &[u8]) -> usize {
    let (mut string, mut escape, mut count) = (false,false,0);
    for &b in bytes {
        if string { if escape { escape=false; } else if b==b'\\' { escape=true; } else if b==b'"' { string=false; } }
        else if b==b'"' { string=true; } else if b==b':' { count+=1; }
    } count
}
fn retained_members(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(o) => o.len()+o.values().map(retained_members).sum::<usize>(),
        serde_json::Value::Array(a) => a.iter().map(retained_members).sum(), _ => 0,
    }
}
fn parse_window(header: &SourceHeaderEntry, file: &PublicFile, offset: u64, len: usize) -> Result<OwnedCharged<json::ImportedValue>, Error> {
    if len==0 || len>JSON_BYTES { return Err(Error::Bounds); }
    let mut bytes = exact_vec(len)?; bytes.resize(len,0);
    file.read(offset,&mut bytes).map_err(|_| Error::Binding)?;
    import_raw_json(header,bytes)
}
fn data_start(file: &PublicFile) -> Result<u64, Error> {
    data_start_into(file,&mut [0;8])
}
fn data_start_into(file: &PublicFile, bytes: &mut [u8;8]) -> Result<u64, Error> {
    file.read(0,bytes).map_err(|_| Error::Binding)?;
    let n = u64::from_le_bytes(*bytes);
    if n==0 || n>JSON_BYTES as u64 { return Err(Error::Bounds); }
    n.checked_add(8).filter(|end| *end<file.size()).ok_or(Error::Bounds)
}
fn validate_inventory(header: &SourceHeaderEntry, files: &[RawBacking]) -> Result<(), Error> {
    let p = header.raw_premise.as_ref().ok_or(Error::Binding)?;
    // Verify original config using the SAME immutable completed file, not a path
    // reopen or the caller's declaration of observed offsets.
    let mut frame = [0u8;FRAME];
    for (offset, original) in header.original.raw_configuration().chunks(FRAME).enumerate() {
        files[0].read((offset*FRAME) as u64,&mut frame[..original.len()]).map_err(|_| Error::Binding)?;
        if &frame[..original.len()]!=original { return Err(Error::Binding); }
    }
    let index_owner = parse_window(header,&files[1],0,usize::try_from(files[1].size()).map_err(|_| Error::Bounds)?)?;
    let index = index_owner.payload().value().as_object().ok_or(Error::Binding)?;
    if index.keys().any(|k| k!="metadata" && k!="weight_map") { return Err(Error::Binding); }
    let map = index.get("weight_map").and_then(|m| m.as_object()).ok_or(Error::Binding)?;
    if map.len()!=p.tensors.len() { return Err(Error::Binding); }
    for tensor in &p.tensors {
        if map.get(&tensor.name).and_then(|v| v.as_str())!=Some(p.files[tensor.file].name.as_str()) { return Err(Error::Binding); }
    }
    drop(index_owner);
    for (ordinal,file) in files.iter().enumerate().skip(2) {
        let start = data_start(file)?;
        let document = parse_window(header,file,8,(start-8) as usize)?;
        let object = document.payload().value().as_object().ok_or(Error::Binding)?;
        let expected = p.tensors.iter().filter(|t| t.file==ordinal);
        let count = expected.clone().count();
        if object.len()!=count+usize::from(object.contains_key("__metadata__")) { return Err(Error::Binding); }
        if object.get("__metadata__").is_some_and(|m| m.as_object().is_none_or(|m| m.values().any(|v| !v.is_string()))) { return Err(Error::Binding); }
        let mut end = 0;
        for tensor in expected {
            let record = object.get(&tensor.name).and_then(|v| v.as_object()).ok_or(Error::Binding)?;
            if record.len()!=3 { return Err(Error::Binding); }
            let dtype = match tensor.dtype { raw::RawScalarType::F32Le=>"F32", raw::RawScalarType::Bf16Le=>"BF16", raw::RawScalarType::E4m3Fn=>"F8_E4M3" };
            let shape = record.get("shape").and_then(|v| v.as_array()).ok_or(Error::Binding)?;
            let offsets = record.get("data_offsets").and_then(|v| v.as_array()).ok_or(Error::Binding)?;
            if record.get("dtype").and_then(|v| v.as_str())!=Some(dtype) || shape.len()!=tensor.rank
                || shape.iter().zip(tensor.shape).any(|(a,b)| a.as_u64()!=u64::try_from(b).ok())
                || offsets.len()!=2 || offsets[0].as_u64()!=Some(tensor.begin) || offsets[1].as_u64()!=Some(tensor.end)
                || tensor.begin!=end { return Err(Error::Binding); }
            end=tensor.end;
        }
        if start.checked_add(end)!=Some(file.size()) { return Err(Error::Binding); }
    }
    // Full finite/scale/Q8 validation, not just the subsequently selected range.
    let mut workspace = SourceDecodeWorkspace::new();
    for tensor in &p.tensors {
        let file = &files[tensor.file]; let start = data_start(file)?;
        for coordinate in 0..tensor.elements()? {
            let mut bytes = [0;4];
            let offset = start.checked_add(tensor.begin).and_then(|n| n.checked_add((coordinate as u64).checked_mul(tensor.width())?)).ok_or(Error::Bounds)?;
            file.read(offset,&mut bytes[..tensor.width() as usize]).map_err(|_| Error::Binding)?;
            let value = if let Some(index) = tensor.scale_index {
                let scale = p.tensors.get(index).ok_or(Error::Binding)?;
                let tile = raw::block128_scale_index(tensor.shape[0],tensor.shape[1],coordinate/tensor.shape[1],coordinate%tensor.shape[1]).map_err(|_| Error::Shape)?;
                let mut scale_bytes=[0;2];
                let scale_offset=data_start(&files[scale.file])?.checked_add(scale.begin).and_then(|n| n.checked_add((tile as u64).checked_mul(2)?)).ok_or(Error::Bounds)?;
                files[scale.file].read(scale_offset,&mut scale_bytes).map_err(|_| Error::Binding)?;
                raw::positive_bf16_scale(&scale_bytes,&mut workspace).map_err(|_| Error::Binding)?;
                raw::scaled_e4m3fn(bytes[0],scale_bytes).map_err(|_| Error::Binding)?
            } else { raw::decode_finite(tensor.dtype,&bytes[..tensor.width() as usize],&mut workspace).map_err(|_| Error::Binding)? };
            raw::q8(value).map_err(|_| Error::Binding)?;
        }
    }
    let head_name=semantic_weight(header,pllm_models::ModelOperator::OutputHead)?;
    let embedding_name=semantic_weight(header,pllm_models::ModelOperator::TokenLookup)?;
    let head=p.tensors.iter().find(|t| t.name==head_name).ok_or(Error::Binding)?;
    let embedding=p.tensors.iter().find(|t| t.name==embedding_name).ok_or(Error::Binding)?;
    if head.shape!=embedding.shape || head.rank!=2 || embedding.rank!=2 { return Err(Error::Shape); }
    let mut other=[0u8;FRAME];
    let count=head.end-head.begin;
    let (hs,es)=(data_start(&files[head.file])?,data_start(&files[embedding.file])?);
    let mut offset=0;
    while offset<count {
        let len=usize::try_from((count-offset).min(FRAME as u64)).map_err(|_| Error::Bounds)?;
        files[head.file].read(hs.checked_add(head.begin).and_then(|n| n.checked_add(offset)).ok_or(Error::Bounds)?,&mut frame[..len]).map_err(|_| Error::Binding)?;
        files[embedding.file].read(es.checked_add(embedding.begin).and_then(|n| n.checked_add(offset)).ok_or(Error::Bounds)?,&mut other[..len]).map_err(|_| Error::Binding)?;
        if frame[..len]!=other[..len] { return Err(Error::Binding); } offset+=len as u64;
    }
    Ok(())
}

#[cfg(test)]
#[path = "raw_owner_tests.rs"]
mod tests;
