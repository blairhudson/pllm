//! Operator-provisioned startup premise, NOT publisher authentication.
//!
//! The operator independently approves the PAIR (exact existing full source-lock
//! digest, exact inventory/configuration/semantic recipe). We do not reconstruct
//! that digest from a raw subset, authenticate a publisher, or trust SDK labels.
//! Trusted in-process host code is outside this boundary; this is not isolation
//! from that code. No candidate file, index, tensor or checksum is read here.
//!
//! Accounting exception approved solely for the INITIAL std::env::var_os clone:
//! that allocation is NOT prepaid. Malicious in-process setenv can make it huge
//! before rejection. Its borrowed encoded length and existing capacity are then
//! checked without conversion; retained path storage is charged before open.
//! All later buffers, bounded parser construction overlap, owned inventory,
//! synchronization controls and serialized temporary borrowed views are admitted
//! on a separate bounded process HOST baseline, never an invented request ledger.
//! These are logical owned slots/work, not allocator/RSS/syscall/retry/time caps.
use crate::plan_input::json;
use crate::replicated_decoder::program::source_entry::ingress::HostRawExpectations;
use crate::verified_source::raw::{self, RawScalarType};
use pllm_core::public_file;
use pllm_garble::shared_allocation::{OwnedBudget, OwnedCharged, OwnedExternalCharge, MAX_ALLOCATION_BYTES};
use serde_json::Value;
use std::{ffi::OsString, fs::File, io::Read, mem::size_of, path::Path, sync::{Mutex, OnceLock}};

const JSON_BYTES: usize = 1 << 20;
const NAME: usize = 1024;
const PATH_BYTES: usize = 4096;
const DESCRIPTORS: usize = crate::replicated_decoder::source_policy::DESCRIPTORS;
const PARSER_BYTES: usize = 8 << 20; // unchanged source-entry parser payload cap
const PARSER_WORK: u128 = 1_000_000_000;
const PATH_READ_WORK: u128 = 4 * PATH_BYTES as u128 + 4 * (JSON_BYTES as u128 + 1);
type Error = &'static str;
type FileView<'a> = (&'a str, u64, [u8; 32]);
type TensorView<'a> = (&'a str, usize, &'a str, &'a [usize], u64, u64, Option<&'a str>);

struct ApprovedFile { name: String, size: u64, digest: [u8; 32] }
struct ApprovedTensor {
    name: String, file: usize, dtype: RawScalarType, shape: [usize; 2], rank: usize,
    begin: u64, end: u64, scale: Option<String>,
}
struct HostSourceConfig {
    approved_lock_digest: [u8; 32],
    files: Vec<ApprovedFile>,
    tensors: Vec<ApprovedTensor>,
    config_sha256: [u8; 32],
    semantic_digest: [u8; 32],
    recipe: &'static str,
    // Fields retire before these guards. Scratch covers ONE invocation under
    // Startup.config's mutex, including borrowed tuple assembly/callback overlap.
    _storage: OwnedExternalCharge,
    _scratch: OwnedExternalCharge,
}
struct Startup {
    config: Option<Mutex<HostSourceConfig>>,
    path: Option<OwnedCharged<OsString>>,
    failure: Option<Error>,
    work_left: u128,
    #[cfg(test)]
    validation_builds: usize,
    _controls: OwnedExternalCharge,
    root: OwnedBudget,
}
static STARTUP: OnceLock<Result<Startup, Error>> = OnceLock::new();

fn add(a: usize, b: usize) -> Result<usize, Error> { a.checked_add(b).ok_or("host configuration bounds") }
fn mul(a: usize, b: usize) -> Result<usize, Error> { a.checked_mul(b).ok_or("host configuration bounds") }
fn sum(parts: impl IntoIterator<Item = usize>) -> Result<usize, Error> { parts.into_iter().try_fold(0, add) }
fn resource<T>(result: Result<T, pllm_garble::shared_allocation::Error>) -> Result<T, Error> {
    result.map_err(|_| "host baseline exhausted")
}

fn control_bytes() -> Result<usize, Error> {
    sum([size_of::<OnceLock<Result<Startup, Error>>>(), 2 * size_of::<Startup>(),
        size_of::<Mutex<HostSourceConfig>>(), 2 * size_of::<HostSourceConfig>(),
        2 * size_of::<File>(), 2 * size_of::<std::fs::Metadata>(), size_of::<OsString>(),
        size_of::<json::Limits>(), size_of::<json::SourceCounts>(),
        size_of::<Result<Startup, Error>>(), size_of::<Result<(), Error>>()])
}
fn storage_bytes(files: usize, tensors: usize, strings: usize) -> Result<usize, Error> {
    // Two named inventories cover stack-to-retained construction moves; string
    // backing is copied only once. The parser/source owners remain independently
    // live and charged until this whole typed inventory has been validated.
    sum([mul(2, mul(files, size_of::<ApprovedFile>())?)?,
        mul(2, mul(tensors, size_of::<ApprovedTensor>())?)?, strings,
        2 * size_of::<Vec<ApprovedFile>>(), 2 * size_of::<Vec<ApprovedTensor>>(),
        2 * size_of::<ApprovedFile>(), 2 * size_of::<ApprovedTensor>(),
        2 * size_of::<String>(), 2 * size_of::<[u8; 32]>()])
}
fn scratch_bytes(files: usize, tensors: usize) -> Result<usize, Error> {
    sum([mul(files, size_of::<FileView<'_>>())?, mul(tensors, size_of::<TensorView<'_>>())?,
        2 * size_of::<Vec<FileView<'_>>>(), 2 * size_of::<Vec<TensorView<'_>>>(),
        2 * size_of::<FileView<'_>>(), 2 * size_of::<TensorView<'_>>(),
        2 * size_of::<HostRawExpectations<'_>>(), size_of::<std::sync::MutexGuard<'_, HostSourceConfig>>()])
}
fn path_io_bytes(bytes: usize) -> Result<usize, Error> {
    // Named temporary path-conversion slots for std filesystem calls, separate
    // from the retained OsString. No call receives an unbounded path. This is
    // logical conversion backing, not a bound on opaque OS/kernel allocation.
    sum([mul(2, add(bytes, 1)?)?, 2 * size_of::<Vec<u8>>()])
}
fn host_limit() -> Result<usize, Error> {
    // Declared reusable arithmetic, not MAX-to-fit: existing parser 8 MiB plus
    // <=1 MiB source, <=1 MiB decoded owned strings, descriptor slots/views,
    // bounded path and named controls. Existing global allocation cap is intact.
    let bytes = sum([resource(OwnedBudget::resources())?.initial_reserved_bytes,
        control_bytes()?, PATH_BYTES, size_of::<OwnedCharged<OsString>>(), path_io_bytes(PATH_BYTES)?, JSON_BYTES + 1,
        size_of::<OwnedCharged<Vec<u8>>>(), PARSER_BYTES,
        json::bootstrap_bytes().map_err(|_| "host parser bounds")?,
        size_of::<OwnedCharged<json::ImportedValue>>(),
        storage_bytes(DESCRIPTORS, DESCRIPTORS, JSON_BYTES)?,
        scratch_bytes(DESCRIPTORS, DESCRIPTORS)?,
        6 * size_of::<OwnedExternalCharge>()])?;
    if bytes > MAX_ALLOCATION_BYTES { return Err("host baseline exceeds existing cap"); }
    Ok(bytes)
}
fn validation_work(files: usize, tensors: usize) -> Result<u128, Error> {
    if files > DESCRIPTORS || tensors > DESCRIPTORS { return Err("host descriptor work bounds"); }
    let n = add(files, tensors)? as u128;
    // Includes duplicate-name/order/file-coverage and BOTH scale relationship
    // searches, even common-prefix names. No inventory traversal to derive quote.
    Ok(4 * n * n * NAME as u128 + 8 * DESCRIPTORS as u128 * NAME as u128)
}
fn work_limit() -> Result<u128, Error> {
    Ok(PATH_READ_WORK + PARSER_WORK + validation_work(DESCRIPTORS, DESCRIPTORS)? + 2 * DESCRIPTORS as u128)
}

impl Startup {
    fn new(limit: usize) -> Result<Self, Error> {
        let root = resource(OwnedBudget::new(limit))?;
        let controls = resource(root.charge_external(control_bytes()?))?;
        Ok(Self { config: None, path: None, failure: None, work_left: work_limit()?,
            #[cfg(test)] validation_builds: 0,
            _controls: controls, root })
    }
    fn pay(&mut self, amount: u128) -> Result<(), Error> {
        self.work_left = self.work_left.checked_sub(amount).ok_or("host logical work exhausted")?;
        Ok(())
    }
    fn load(&mut self, path: Option<OsString>) -> Result<(), Error> {
        let Some(path) = path else { return Ok(()); };
        self.pay(PATH_READ_WORK)?;
        let borrowed = path.as_os_str();
        if borrowed.is_empty() || borrowed.as_encoded_bytes().len() > PATH_BYTES
            || path.capacity() > PATH_BYTES || borrowed.as_encoded_bytes().contains(&0)
            || !Path::new(borrowed).is_absolute()
        { return Err("host configuration path must be bounded and absolute"); }
        let io_bytes = path_io_bytes(borrowed.as_encoded_bytes().len())?;
        let charge = resource(self.root.charge_external(path.capacity()))?;
        // Move, no conversion/clone. The sole initial-clone exception ends here.
        self.path = Some(resource(charge.try_build(|_| Ok(path)))?);
        let _path_io = resource(self.root.charge_external(io_bytes))?;
        let path = Path::new(self.path.as_ref().ok_or("host path missing")?.payload());
        let metadata = std::fs::metadata(path).map_err(|_| "host configuration metadata failed")?;
        if !metadata.is_file() { return Err("host configuration is not a regular file"); }
        if metadata.len() == 0 || metadata.len() > JSON_BYTES as u64 {
            return Err("host configuration size outside cap");
        }
        let mut file = File::open(path).map_err(|_| "host configuration open failed")?;
        let metadata = file.metadata().map_err(|_| "host configuration metadata failed")?;
        if !metadata.is_file() { return Err("host configuration is not a regular file"); }
        let n = usize::try_from(metadata.len()).map_err(|_| "host configuration too large")?;
        if n == 0 || n > JSON_BYTES { return Err("host configuration size outside cap"); }
        let source = read_source(&self.root, &mut file, n)?;
        self.parse(source)
    }
    fn parse(&mut self, source: OwnedCharged<Vec<u8>>) -> Result<(), Error> {
        self.pay(PARSER_WORK)?; // before quote's scanner, not after observed use
        let bootstrap = json::bootstrap_bytes().map_err(|_| "host parser bounds")?;
        let quote = json::quote(source, resource(self.root.charge_external(bootstrap))?, limits())
            .map_err(|_| "host configuration JSON rejected")?;
        // The existing scanner counts every source member, including escaped
        // duplicate keys; retained-member count detects its last-value-wins loss.
        let counts = quote.counts();
        let parsed = quote.construct().map_err(|_| "host configuration JSON rejected")?;
        self.pay(2 * DESCRIPTORS as u128)?;
        if retained_members(parsed.payload().value()) != counts.map_entries {
            return Err("duplicate host configuration field");
        }
        let value = parsed.payload().value();
        let object = object(value, &["version", "approved_lock_digest", "files", "tensors", "config_sha256", "semantic_digest", "recipe"])?;
        if number(&object["version"])? != 1 { return Err("unsupported host configuration version"); }
        let files = array(&object["files"])?;
        let tensors = array(&object["tensors"])?;
        if !(3..=DESCRIPTORS).contains(&files.len()) || tensors.is_empty() || tensors.len() > DESCRIPTORS {
            return Err("host descriptor count outside cap");
        }
        self.pay(validation_work(files.len(), tensors.len())?)?;
        let storage = resource(self.root.charge_external(storage_bytes(files.len(), tensors.len(), counts.decoded_string_bytes)?))?;
        let scratch = resource(self.root.charge_external(scratch_bytes(files.len(), tensors.len())?))?;
        // Test-local entry counter: no production field, work grant or API.
        #[cfg(test)] { self.validation_builds += 1; }
        let config = HostSourceConfig::build(object, files, tensors, storage, scratch)?;
        // Parser, original read buffer and their complete construction reservation
        // retire only after typed validation/copy overlap has finished.
        drop(parsed);
        self.config = Some(Mutex::new(config));
        Ok(())
    }
}

fn read_source(root: &OwnedBudget, reader: &mut impl Read, n: usize) -> Result<OwnedCharged<Vec<u8>>, Error> {
    if n == 0 || n > JSON_BYTES { return Err("host configuration size outside cap"); }
    let charge = resource(root.charge_external(add(n, 1)?))?;
    let mut failure = None;
    resource(charge.try_build(|_| {
        let result = (|| {
            let mut bytes = exact_vec(n + 1)?;
            bytes.resize(n + 1, 0);
            let mut used = 0;
            while used < bytes.len() {
                // No automatic retry loop: an interrupted/error read fails shut.
                let read = reader.read(&mut bytes[used..]).map_err(|_| "host configuration read failed")?;
                if read == 0 { break; }
                used = add(used, read)?;
            }
            if used != n { return Err("host configuration changed size while reading"); }
            bytes.truncate(n);
            Ok(bytes)
        })();
        result.map_err(|error| { failure = Some(error); pllm_garble::shared_allocation::Error::Resource })
    })).map_err(|error| failure.unwrap_or(error))
}
fn limits() -> json::Limits {
    json::Limits { max_source_bytes: JSON_BYTES, max_nodes: DESCRIPTORS, max_depth: 16,
        max_string_bytes: NAME, max_number_bytes: 64, max_number_work: 100_000_000,
        max_work: PARSER_WORK, max_payload_bytes: PARSER_BYTES }
}
fn retained_members(value: &Value) -> usize {
    match value {
        Value::Object(o) => o.len() + o.values().map(retained_members).sum::<usize>(),
        Value::Array(a) => a.iter().map(retained_members).sum(), _ => 0,
    }
}
fn object<'a>(value: &'a Value, keys: &[&str]) -> Result<&'a serde_json::Map<String, Value>, Error> {
    let object = value.as_object().ok_or("host object required")?;
    if object.len() != keys.len() || keys.iter().any(|key| !object.contains_key(*key)) {
        return Err("unknown or missing host configuration field");
    }
    Ok(object)
}
fn array(value: &Value) -> Result<&[Value], Error> { value.as_array().map(Vec::as_slice).ok_or("host array required") }
fn text(value: &Value) -> Result<&str, Error> {
    value.as_str().filter(|s| !s.is_empty() && s.len() <= NAME).ok_or("host text outside cap")
}
fn name(value: &Value) -> Result<&str, Error> {
    let text = text(value)?;
    if !text.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) { return Err("invalid host inventory name"); }
    Ok(text)
}
fn number(value: &Value) -> Result<u64, Error> { value.as_u64().ok_or("host unsigned integer required") }
fn ordinal(value: &Value) -> Result<usize, Error> { usize::try_from(number(value)?).map_err(|_| "host integer overflow") }
fn digest(value: &Value) -> Result<[u8; 32], Error> {
    let text = text(value)?;
    if text.len() != 64 { return Err("host digest must be 64 lowercase hex bytes"); }
    let mut out = [0; 32];
    for (i, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        let hex = |b| match b { b'0'..=b'9' => Ok(b - b'0'), b'a'..=b'f' => Ok(b - b'a' + 10), _ => Err("invalid host digest") };
        out[i] = hex(pair[0])? * 16 + hex(pair[1])?;
    }
    if out == [0; 32] { return Err("zero host digest"); }
    Ok(out)
}
fn exact_vec<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values.try_reserve_exact(n).map_err(|_| "host allocation failed")?;
    if values.capacity() != n { return Err("host vector capacity mismatch"); }
    Ok(values)
}
fn copy(text: &str) -> Result<String, Error> {
    let mut value = String::new();
    value.try_reserve_exact(text.len()).map_err(|_| "host allocation failed")?;
    if value.capacity() != text.len() { return Err("host string capacity mismatch"); }
    value.push_str(text);
    Ok(value)
}
fn dtype_name(dtype: RawScalarType) -> &'static str {
    match dtype { RawScalarType::F32Le => "F32", RawScalarType::Bf16Le => "BF16", RawScalarType::E4m3Fn => "F8_E4M3" }
}

impl HostSourceConfig {
    fn build(manifest: &serde_json::Map<String, Value>, file_values: &[Value], tensor_values: &[Value],
        storage: OwnedExternalCharge, scratch: OwnedExternalCharge) -> Result<Self, Error>
    {
        // The guards are arguments: all partial locals drop before the guards on
        // every error/unwind. No try_build closure can accidentally detach them.
        let approved_lock_digest = digest(&manifest["approved_lock_digest"])?;
        let config_sha256 = digest(&manifest["config_sha256"])?;
        let semantic_digest = digest(&manifest["semantic_digest"])?;
        let recipe = match text(&manifest["recipe"])? { "D31" => "D31", "D32" => "D32", _ => return Err("unsupported host recipe") };
        let mut files: Vec<ApprovedFile> = exact_vec(file_values.len())?;
        let mut total = 0usize;
        for (i, value) in file_values.iter().enumerate() {
            let o = object(value, &["name", "size", "sha256"])?;
            let name = name(&o["name"])?;
            let size = number(&o["size"])?;
            if size == 0 || size > public_file::MAX_BYTES || (i < 2 && size > JSON_BYTES as u64) {
                return Err("host file size outside existing cap");
            }
            if (i == 0 && name != "config.json") || (i == 1 && !name.ends_with(".safetensors.index.json"))
                || (i >= 2 && !name.ends_with(".safetensors")) || files.iter().any(|f| f.name == name)
            { return Err("invalid or duplicate host file"); }
            total = add(total, usize::try_from(size).map_err(|_| "host file size overflow")?)?;
            files.push(ApprovedFile { name: copy(name)?, size, digest: digest(&o["sha256"])? });
        }
        if files[0].digest != config_sha256 { return Err("host configuration digest mismatch"); }
        let mut tensors: Vec<ApprovedTensor> = exact_vec(tensor_values.len())?;
        for value in tensor_values {
            let o = object(value, &["name", "file", "dtype", "shape", "begin", "end", "scale"])?;
            let name = name(&o["name"])?;
            let file = ordinal(&o["file"])?;
            if file < 2 || file >= files.len() || tensors.iter().any(|t| t.name == name) { return Err("invalid or duplicate host tensor"); }
            let dtype = match text(&o["dtype"])? {
                "F32" => RawScalarType::F32Le, "BF16" => RawScalarType::Bf16Le,
                "F8_E4M3" if recipe == "D32" => RawScalarType::E4m3Fn, _ => return Err("invalid host dtype"),
            };
            let dimensions = array(&o["shape"])?;
            if !(1..=2).contains(&dimensions.len()) { return Err("invalid host tensor rank"); }
            let mut shape = [1; 2];
            let mut elements = 1usize;
            for (i, dimension) in dimensions.iter().enumerate() {
                shape[i] = ordinal(dimension)?;
                if shape[i] == 0 { return Err("zero host dimension"); }
                elements = mul(elements, shape[i])?;
            }
            let cap = usize::try_from(public_file::MAX_BYTES).map_err(|_| "host file cap overflow")?;
            let quote = raw::quote(dtype, elements, cap, cap).map_err(|_| "host shape outside file cap")?;
            let width = raw::quote(dtype, 1, cap, cap).map_err(|_| "host scalar quote failed")?.source_bytes as u64;
            let begin = number(&o["begin"])?;
            let end = number(&o["end"])?;
            let scale = if o["scale"].is_null() { None } else { Some(copy(super::host_source_config::name(&o["scale"])?)?) };
            if begin.checked_add(quote.source_bytes as u64) != Some(end) || end > files[file].size || begin % width != 0
                || tensors.last().is_some_and(|t| t.file > file || (t.file == file && t.end != begin))
                || (tensors.last().is_none_or(|t| t.file != file) && begin != 0)
            { return Err("invalid host tensor layout"); }
            tensors.push(ApprovedTensor { name: copy(name)?, file, dtype, shape, rank: dimensions.len(), begin, end, scale });
        }
        for file in 2..files.len() {
            if !tensors.iter().any(|t| t.file == file) { return Err("host shard has no tensors"); }
        }
        for tensor in &tensors {
            match (tensor.dtype, tensor.scale.as_deref()) {
                (RawScalarType::E4m3Fn, Some(name)) if tensor.rank == 2 => {
                    raw::block128_scale_index(tensor.shape[0], tensor.shape[1], 0, 0).map_err(|_| "invalid host scale geometry")?;
                    let scale = tensors.iter().find(|t| t.name == name).ok_or("host scale missing")?;
                    if scale.dtype != RawScalarType::Bf16Le || scale.rank != 2 || scale.scale.is_some()
                        || scale.shape != [tensor.shape[0] / 128, tensor.shape[1] / 128]
                    { return Err("invalid host scale mapping"); }
                }
                (RawScalarType::F32Le | RawScalarType::Bf16Le, None) => (),
                _ => return Err("invalid host scale relationship"),
            }
        }
        Ok(Self { approved_lock_digest, files, tensors, config_sha256, semantic_digest, recipe,
            _storage: storage, _scratch: scratch })
    }
    fn views(&self) -> Result<(Vec<FileView<'_>>, Vec<TensorView<'_>>), Error> {
        let mut files = exact_vec(self.files.len())?;
        let mut tensors = exact_vec(self.tensors.len())?;
        for file in &self.files { files.push((file.name.as_str(), file.size, file.digest)); }
        for tensor in &self.tensors {
            tensors.push((tensor.name.as_str(), tensor.file, dtype_name(tensor.dtype), &tensor.shape[..tensor.rank],
                tensor.begin, tensor.end, tensor.scale.as_deref()));
        }
        Ok((files, tensors))
    }
    fn with_expected<T>(&self, source: [u8; 32], callback: impl for<'a> FnOnce(Option<HostRawExpectations<'a>>) -> Result<T, String>) -> Result<T, String> {
        if source != self.approved_lock_digest { return callback(None); }
        // The retained baseline scratch charge predates even these allocations.
        // Its caller holds the mutex until vectors AND header copy overlap retire.
        // Each lookup has two finite length-bounded host assembly passes, not
        // uncharged request-premise validation: header copies/comparisons/hash
        // validation still prepay on their ORIGINAL header ledger downstream.
        let (files, tensors) = self.views().map_err(str::to_owned)?;
        callback(Some(HostRawExpectations::provision(&files, &tensors, self.config_sha256, self.semantic_digest, self.recipe)))
    }
}

fn initialize(cell: &OnceLock<Result<Startup, Error>>, capture: impl FnOnce() -> Option<OsString>) -> Result<(), Error> {
    let state = cell.get_or_init(|| {
        let mut state = Startup::new(host_limit()?)?;
        // Exactly ONE approved unprepaid allocation; no per-request env reads.
        let path = capture();
        if let Err(error) = state.load(path) { state.failure = Some(error); }
        Ok(state)
    });
    match state { Ok(state) => state.failure.map_or(Ok(()), Err), Err(error) => Err(*error) }
}
pub(super) fn initialize_from_startup() -> Result<(), Error> {
    initialize(&STARTUP, || std::env::var_os("PLLM_HOST_SOURCE_CONFIG"))
}
pub(super) fn with_expected<T>(source: [u8; 32], callback: impl for<'a> FnOnce(Option<HostRawExpectations<'a>>) -> Result<T, String>) -> Result<T, String> {
    let Some(Ok(state)) = STARTUP.get() else { return callback(None); };
    if state.failure.is_some() { return callback(None); }
    let Some(config) = &state.config else { return callback(None); };
    config.lock().map_err(|_| "host configuration lock poisoned".to_owned())?.with_expected(source, callback)
}

#[cfg(test)]
#[path = "host_source_config_tests.rs"]
mod tests;
