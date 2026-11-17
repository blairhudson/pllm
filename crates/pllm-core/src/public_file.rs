//! Immutable public bytes, shared without copying a private file.
//!
//! Only a completed bounded writer can mint a snapshot. No path, writable file,
//! or mutable byte view escapes; readers and paged matrices share its lifetime.
//! The default writer authenticates an expected digest; native producers may
//! instead commit the digest computed over their output. That does not authenticate
//! the producer's source.
use sha2::{Digest, Sha256};
use std::{fs::File, io::{self, Write}, path::Path, sync::Arc};
#[cfg(unix)]
use crate::{resident_fd::FileLease, fd_close_once::{
    observe_once, read_at_once, write_once, FileObservation, ObserveOnceError, IoOnceError,
}};

/// Retained location only, not coefficient/source/admission authority.
pub trait SnapshotDirectory: Send + Sync {
    fn path(&self) -> &Path;
    /// Receives genuine local fstat observation, NEVER a writable descriptor.
    fn check_file(&self, file: FileBacking) -> Result<(), String>;
    /// Additive fail-closed seam; never calls the legacy String boundary.
    fn check_file_typed(&self, _file: FileBacking) -> Result<(), SnapshotDirectoryError> {
        Err(SnapshotDirectoryError::Unsupported)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotDirectoryError { Unsupported, Unavailable, Poisoned, Retired, Io, DeviceMismatch }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublicBackingError { Io, AllocationOverflow }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublicFileCreateError {
    Bounds, DirectoryPolicy, Io, ObservationUnsupported,
    Observation(PublicBackingError), Directory(SnapshotDirectoryError),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublicFileFinishError { Closed, ContentMismatch }

// Each named io helper: item/map adapter once per reached map (also on Ok);
// on Err only: FnOnce self/(io::Error,) invocation, incoming formal, drop item,
// (io::Error,) actual/formal/move, opaque destructor, () return/receipt, fixed
// enum construction/return/caller receipt. Destructor unwind instead retires
// initialized caller prefix; no normal result then. No native capacity claim.
fn creation_io_error(error: std::io::Error) -> PublicFileCreateError {
    std::mem::drop(error);
    PublicFileCreateError::Io
}
fn backing_io_error(error: std::io::Error) -> PublicBackingError {
    std::mem::drop(error);
    PublicBackingError::Io
}
const fn finish_digest_callback<'a>(digest: &'a [u8; 32])
    -> impl FnOnce([u8; 32]) -> bool + 'a
{
    move |v: [u8; 32]| v != *digest
}

/// Every escaped reader retains this whole backing. For File storage, File closes
/// BEFORE directory (and covering lease). Resident storage only drops a reader
/// lease: an external coordinator owns descriptor/namespace custody and explicit
/// retirement. Last backing Arc drop is never a retirement acknowledgement.
pub(crate) struct PublicFileBacking<B> {
    file: B,
    _directory: Option<Arc<dyn SnapshotDirectory>>,
}
impl PublicFileBacking<File> {
    pub(crate) fn anonymous(file: File) -> Self {
        Self {
            file,
            _directory: None,
        }
    }
}
impl std::ops::Deref for PublicFileBacking<File> {
    type Target = File;
    fn deref(&self) -> &File {
        &self.file
    }
}
/// Actual shared backing layout, including the retained optional location handle.
pub fn backing_bytes() -> Result<usize, String> {
    let (layout, _) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(std::alloc::Layout::new::<PublicFileBacking<File>>())
        .map_err(|e| e.to_string())?;
    Ok(layout.pad_to_align().size())
}

/// Explicit-directory setup/return/path-copy overlap, not coefficient backing.
pub fn writer_creation_bytes() -> Result<usize, String> {
    let mut bytes = 8usize
        .checked_mul(MAX_DIRECTORY_PATH_BYTES)
        .ok_or("file setup quote overflow")?;
    for n in [
        std::mem::size_of::<Pending<File>>(),
        std::mem::size_of::<PublicFileWriter>(),
        std::mem::size_of::<Result<PublicFileWriter, String>>(),
        std::mem::size_of::<Result<File, std::io::Error>>(),
        2 * std::mem::size_of::<std::path::PathBuf>(),
    ] {
        bytes = bytes.checked_add(n).ok_or("file setup quote overflow")?;
    }
    Ok(bytes)
}
pub const MAX_DIRECTORY_PATH_BYTES: usize = 256;
pub const MAX_BYTES: u64 = 4 << 30;
pub const MAX_CHUNK_BYTES: usize = 1 << 20;

/// Read-only observation of an owned file's physical backing. Deduplicate using
/// `(device, file)` while retaining the file owner: identifiers may be reused
/// after the last owner closes. This value itself does not retain ownership.
/// Allocated bytes are a filesystem observation, not a logical length or a
/// reservation credit; aggregation and credit clamping belong to callers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileBacking {
    pub device: u64,
    pub file: u64,
    pub allocated_bytes: u64,
}

pub type PublicFile = PublicFileStorage<File>;
/// Reader-lease snapshot, with no legacy callback-free I/O methods:
/// ```compile_fail
/// use pllm_core::public_file::ResidentPublicFile;
/// fn denied(snapshot: &ResidentPublicFile) {
///     let _ = snapshot.read(0u64, &mut [0u8; 1]);
/// }
/// ```
/// File-backed equivalent remains available:
/// ```
/// use pllm_core::public_file::PublicFile;
/// fn allowed(snapshot: &PublicFile) {
///     let _ = snapshot.read(0u64, &mut [0u8; 1]);
/// }
/// ```
#[cfg(unix)]
pub type ResidentPublicFile = PublicFileStorage<FileLease>;

/// Immutable cached metadata and shared backing. Generic Rust ABI/layout is not
/// guaranteed; legacy layout qualification must be checked independently.
pub struct PublicFileStorage<B> {
    file: Arc<PublicFileBacking<B>>,
    size: u64,
    digest: [u8; 32],
    max_i8: u32,
}

impl<B> Clone for PublicFileStorage<B> {
    fn clone(&self) -> Self {
        Self { file: Arc::clone(&self.file), size: self.size,
            digest: self.digest, max_i8: self.max_i8 }
    }
}

/// Unformatted bounded-read failure. Native errors are retained, not rendered.
/// This type does not promise arbitrary io::Error payloads are heap-free. The
/// zero-owned-heap native-error premise is limited to the audited Rust 1.85.1
/// unmodified Unix/Windows native constructors, not every cfg(unix), stdlib,
/// backend or error producer. No physical-stack/RSS bound follows.
#[derive(Debug)]
pub enum PublicReadError {
    Bounds,
    Io(io::Error),
    Truncated,
}

// Checked compile-time logical slot inventory only; not native call-frame bytes.
const fn read_layout_mul(n: usize, bytes: usize) -> usize {
    match n.checked_mul(bytes) {
        Some(bytes) => bytes,
        None => panic!("public read control layout overflow"),
    }
}
const fn read_layout_sum<const N: usize>(parts: [usize; N]) -> usize {
    let mut bytes = 0usize;
    let mut i = 0;
    while i < N {
        bytes = match bytes.checked_add(parts[i]) {
            Some(bytes) => bytes,
            None => panic!("public read control layout overflow"),
        };
        i += 1;
    }
    bytes
}
const READ_METHOD_ARGS_BYTES: usize = read_layout_sum([
    std::mem::size_of::<&PublicFile>(), std::mem::size_of::<u64>(),
    std::mem::size_of::<&mut [u8]>(),
]);
const READ_KERNEL_ARGS_BYTES: usize = read_layout_sum([
    std::mem::size_of::<&File>(), std::mem::size_of::<&mut [u8]>(),
    std::mem::size_of::<u64>(),
]);
#[cfg(unix)]
const READ_PLATFORM_CONTROL_BYTES: usize = std::mem::size_of::<Result<(), io::Error>>();
#[cfg(windows)]
const READ_PLATFORM_CONTROL_BYTES: usize = read_layout_sum([
    std::mem::size_of::<Result<usize, io::Error>>(),
    read_layout_mul(5, std::mem::size_of::<usize>()),
    std::mem::size_of::<std::ops::RangeFrom<usize>>(),
    read_layout_mul(2, std::mem::size_of::<u64>()),
    read_layout_mul(2, std::mem::size_of::<bool>()),
]);
/// Fixed logical control slots for the typed method and kernel (D_TIO).
/// Actual target layouts; checked at compile time. Excludes error formatting,
/// String heap, stdlib formatter/destructor internals, native stack and payload
/// backing. This is not a full read-memory bound. Legacy paged tile/read-partition
/// formulas remain payload-only: routing through this kernel adds no such claim.
pub const READ_UNFORMATTED_CONTROL_BYTES: usize = read_layout_sum([
    read_layout_mul(2, READ_METHOD_ARGS_BYTES),
    read_layout_mul(3, READ_KERNEL_ARGS_BYTES),
    std::mem::size_of::<usize>(), std::mem::size_of::<Option<u64>>(),
    read_layout_mul(4, std::mem::size_of::<u64>()),
    read_layout_mul(4, std::mem::size_of::<bool>()),
    read_layout_mul(3, std::mem::size_of::<Result<(), PublicReadError>>()),
    std::mem::size_of::<PublicReadError>(), std::mem::size_of::<io::Error>(),
    READ_PLATFORM_CONTROL_BYTES,
]);

fn format_read_error(error: PublicReadError) -> String {
    match error {
        PublicReadError::Bounds => "public snapshot read exceeds bounds".into(),
        PublicReadError::Truncated => "private public snapshot is truncated".into(),
        PublicReadError::Io(error) => {
            let message = error.to_string();
            drop(error);
            message
        }
    }
}

impl PublicFile {
    /// Deliberately invalidate this anonymous snapshot for EOF regressions.
    /// Cached size/digest remain sealed; no file, path or descriptor escapes.
    /// Requires exclusive per-test ownership of snapshot mutation.
    /// Metadata check and truncation are not atomic against concurrent truncation.
    #[cfg(feature = "insecure-test-support")]
    #[doc(hidden)]
    pub fn truncate_snapshot_for_tests(&self, shorter_len: u64) -> io::Result<()> {
        let current_len = self.file.metadata()?.len();
        if shorter_len >= self.size || shorter_len >= current_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "test truncation must be shorter than current snapshot size",
            ));
        }
        self.file.set_len(shorter_len)
    }

    /// Observe the existing handle with local fstat, without exposing the handle
    /// or changing file contents. Unix allocation is checked st_blocks * 512,
    /// never length/set_len. Unsupported platforms return None (no credit).
    pub fn backing(&self) -> Result<Option<FileBacking>, String> {
        observe_backing(&self.file)
    }
    pub fn read(&self, offset: u64, bytes: &mut [u8]) -> Result<(), String> {
        self.read_unformatted(offset, bytes).map_err(format_read_error)
    }
    /// Same bounded positional read and partial-write behavior as read, without
    /// allocating a formatted error message. No additional I/O or retry layer.
    pub fn read_unformatted(&self, offset: u64, bytes: &mut [u8]) -> Result<(), PublicReadError> {
        let length = bytes.len();
        if length == 0 || length > MAX_CHUNK_BYTES {
            return Err(PublicReadError::Bounds);
        }
        let end = match offset.checked_add(length as u64) {
            Some(end) => end,
            None => return Err(PublicReadError::Bounds),
        };
        if end > self.size {
            return Err(PublicReadError::Bounds);
        }
        read_at_unformatted(&self.file, bytes, offset)
    }
}

impl<B> PublicFileStorage<B> {
    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub(crate) fn file(&self) -> Arc<PublicFileBacking<B>> {
        Arc::clone(&self.file)
    }
    pub(crate) fn max_i8(&self) -> u32 {
        self.max_i8
    }
}

// Complete local inventory: (&File,) actual/formal; metadata native Result and
// map receiver, mapper item/adapter, error drop, mapped Result/? residual/binding;
// blocks receiver/u64, (u64,512) checked product/Option, ok_or/error/? residual;
// dev/ino receivers/scalars, FileBacking/Some/Ok/received Result, metadata drop.
// Non-Unix: borrowed file/None/Ok instead. Metadata and io::Error destruction
// are opaque native endpoints, not capacity or work closure.
macro_rules! emit_observe_backing {
    ($name:ident, $file:ident, $metadata:ident, $allocated_bytes:ident,
     $error:ty, $mapper:expr, $overflow:expr) => {
fn $name($file: &File) -> Result<Option<FileBacking>, $error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let $metadata = $file.metadata().map_err($mapper)?;
        let $allocated_bytes = $metadata
            .blocks()
            .checked_mul(512)
            .ok_or($overflow)?;
        Ok(Some(FileBacking {
            device: $metadata.dev(),
            file: $metadata.ino(),
            $allocated_bytes,
        }))
    }
    #[cfg(not(unix))]
    {
        let _ = $file;
        Ok(None)
    }
}
    };
}
emit_observe_backing!(observe_backing, file, metadata, allocated_bytes, String,
    |e| e.to_string(), "public snapshot allocated byte count overflow");
emit_observe_backing!(observe_backing_typed, file, metadata, allocated_bytes, PublicBackingError,
    backing_io_error, PublicBackingError::AllocationOverflow);

struct Pending<B> {
    file: B,
    directory: Option<Arc<dyn SnapshotDirectory>>,
    size: u64,
    expected: Option<[u8; 32]>,
    written: u64,
    hash: Sha256,
    max_i8: u32,
}

pub type PublicFileWriter = PublicFileWriterStorage<File>;
#[cfg(unix)]
pub type ResidentPublicFileWriter = PublicFileWriterStorage<FileLease>;

pub struct PublicFileWriterStorage<B> {
    pending: Option<Pending<B>>,
}

/// Constant-payload append failure: no native error or owned heap escapes.
/// A transient std I/O error may own heap; it is dropped without formatting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicWriteError {
    Closed,
    Bounds,
    Io,
}

/// Semantic version of the logical write profile, not an authorization token.
pub const PUBLIC_WRITE_LOGICAL_VERSION: u32 = 1;

/// Logical source work and simultaneous control-slot envelope, not physical
/// memory, native stack, CPU, syscall or physical-attempt bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicWriteLogicalQuote {
    pub helper_calls: u64,
    pub visits: u64,
    pub potential_bytes: u64,
    pub control_bytes: usize,
    pub returned_owned_bytes: usize,
}

const fn write_layout_mul(n: usize, bytes: usize) -> usize {
    match n.checked_mul(bytes) {
        Some(bytes) => bytes,
        None => panic!("public write control layout overflow"),
    }
}
const fn write_layout_sum<const N: usize>(parts: [usize; N]) -> usize {
    let mut bytes = 0usize;
    let mut i = 0;
    while i < N {
        bytes = match bytes.checked_add(parts[i]) {
            Some(bytes) => bytes,
            None => panic!("public write control layout overflow"),
        };
        i += 1;
    }
    bytes
}

// Distinct nominal move/formal/received slots, not duplicated backing or an
// assertion that the compiler materializes these slots. Each row follows v1's
// source inventory; no callback endpoint belongs to this fixed owner pool.
const WRITE_FIXED_BYTES: usize = {
    use std::{convert::Infallible, mem::size_of as s, ops::ControlFlow};
    write_layout_sum([
        // Method and take/restore.
        s::<&mut PublicFileWriter>(), s::<&[u8]>(), s::<PublicFileWriter>(),
        s::<Option<Pending<File>>>(), s::<Pending<File>>(), s::<Option<Pending<File>>>(),
        // Take propagation.
        s::<Result<Pending<File>, PublicWriteError>>(),
        s::<ControlFlow<Result<Infallible, PublicWriteError>, Pending<File>>>(),
        s::<Result<Infallible, PublicWriteError>>(),
        // Guard, including captured size-field bound.
        s::<usize>(), s::<u64>(), s::<Option<u64>>(), s::<u64>(),
        write_layout_mul(4, s::<bool>()), s::<&u64>(),
        // Concrete std I/O and typed propagation.
        s::<&mut File>(), s::<&[u8]>(),
        write_layout_mul(2, s::<Result<(), io::Error>>()), s::<io::Error>(),
        s::<Result<(), PublicWriteError>>(),
        s::<ControlFlow<Result<Infallible, PublicWriteError>, ()>>(),
        s::<Result<Infallible, PublicWriteError>>(),
        // Hash and opaque adapter/reduction boundary.
        s::<&mut Sha256>(), s::<&[u8]>(), write_layout_mul(2, s::<()>()),
        s::<&[u8]>(), s::<Option<u32>>(), write_layout_mul(2, s::<u32>()),
        // Count update and return, including the leaf-owned caller endpoint.
        s::<usize>(), write_layout_mul(2, s::<u64>()),
        s::<PublicWriteError>(), write_layout_mul(2, s::<Result<(), PublicWriteError>>()),
        // Retirement endpoints, not destructor internals.
        s::<&mut io::Error>(), s::<&mut File>(),
        s::<&mut Option<Arc<dyn SnapshotDirectory>>>(),
        s::<&mut Pending<File>>(), s::<&mut PublicFileWriter>(),
    ])
};

// Literal |&v| (v as i8 as i32).unsigned_abs(): ten single-assignment
// endpoints, each initialized/retired once. Reads below are additional; writes
// are already in 2B. No owner/hash/error/adapter pool is replaced per byte.
// Nine visits: invoke/transfer; destructure/read byte; cast i8; widen i32;
// intrinsic call/transfer; intrinsic operation/result; receive intrinsic result;
// callback return; reduction receives return. Two helper calls per byte.
const WRITE_CALLBACK_BYTES: usize = {
    use std::mem::size_of as s;
    write_layout_sum([
        write_layout_mul(2, s::<&u8>()), s::<u8>(), s::<i8>(),
        write_layout_mul(2, s::<i32>()), write_layout_mul(4, s::<u32>()),
    ])
};
const WRITE_CALLBACK_READ_BYTES: usize = {
    use std::mem::size_of as s;
    write_layout_sum([
        write_layout_mul(2, s::<&u8>()), write_layout_mul(2, s::<u8>()), s::<i8>(),
        write_layout_mul(2, s::<i32>()), write_layout_mul(3, s::<u32>()),
    ])
};

fn write_logical_quote_parts(
    n: u64, fixed: usize, callback: usize, reads: usize,
) -> Option<PublicWriteLogicalQuote> {
    let control_bytes = fixed.checked_add(callback)?;
    let fixed = u64::try_from(fixed).ok()?;
    let callback = u64::try_from(callback).ok()?;
    let reads = u64::try_from(reads).ok()?;
    // Fixed branch union: 26 transitions and six opaque helper endpoints
    // (write_all, hash, reduction, native-error/File/directory retirement).
    // Visits: take/Closed/propagate/bind (4); guard/Bounds alternatives (6);
    // write/map/propagate (3); hash/count (2); reduction/unwrap/combine/store (4);
    // Some/restore/Ok (3); error/pending/result/local retirement endpoints (4).
    // reverse_control(Ufixed, 26, 1) = 2*Ufixed*(26+1).
    let helper_calls = 6u64.checked_add(n.checked_mul(2)?)?;
    let visits = 26u64.checked_add(n.checked_mul(9)?)?;
    let fixed_potential = fixed.checked_mul(2)?.checked_mul(27)?;
    let callback_potential = callback.checked_mul(2)?.checked_add(reads)?;
    let potential_bytes = fixed_potential.checked_add(n.checked_mul(2)?)?
        .checked_add(n.checked_mul(callback_potential)?)?;
    helper_calls.checked_add(visits)?.checked_add(potential_bytes)?;
    Some(PublicWriteLogicalQuote {
        helper_calls, visits, potential_bytes, control_bytes, returned_owned_bytes: 0,
    })
}

/// Pure bounded-length branch union for `PublicFileWriter::write_typed`.
/// Zero is quoted (Bounds); closed and remaining-size refusals need no inspection.
/// Oversized lengths or scalar conversion/arithmetic overflow return None.
/// Nominal layout sums/products are checked at compile time.
/// Uses actual target layouts; v1 supports pointer widths at most 64 bits.
///
/// Caller must prepay this query, length inspection, Option/result and admission
/// controls. This is not permission, payment or new authority. The leaf includes
/// its received return slot; a caller-mapped result is separate. Input backing
/// and retained writer/directory owners remain caller-funded, not new allocations.
/// Constructors, finish and eventual successful-owner retirement cost separately.
///
/// One exposed std write_all, hash update and std adapter/reduction are opaque
/// logical boundaries. The literal byte callback and unsigned_abs operation have
/// explicit scalar endpoints (K=2, V=9, P=2B+R per byte). Write/hash views add 2n;
/// R already includes the external byte read. No private std retries, hash rounds,
/// reduction internals, CPU, syscalls, physical attempts, scratch/RSS or allocator
/// backing are bounded. Drop endpoints promise neither physical close success,
/// erasure nor disk release; partial writes before error remain possible.
///
/// Before calling, retain another concrete directory owner or retain funding for
/// its possible last-Arc destructor and filesystem/namespace cleanup through
/// return/unwind and eventual final release. This quote does not fund arbitrary
/// SnapshotDirectory destructors. Panic-abort guarantees no cleanup completion.
pub fn public_write_logical_quote_v1(byte_len: usize) -> Option<PublicWriteLogicalQuote> {
    if byte_len > MAX_CHUNK_BYTES { return None; }
    write_logical_quote_parts(
        u64::try_from(byte_len).ok()?, WRITE_FIXED_BYTES,
        WRITE_CALLBACK_BYTES, WRITE_CALLBACK_READ_BYTES,
    )
}

/// Versioned source potential, not payment, allocation capacity or native work.
/// `control_bytes` is the conservative union B, not a stack/peak measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PublicFileLogicalWork {
    pub helper_calls: u64,
    pub visits: u64,
    pub potential_bytes: u64,
    pub control_bytes: usize,
}
impl PublicFileLogicalWork {
    /// Checked finite source recipe. N counts nominal events (including ZST),
    /// K exposed calls, B whole nominal layouts, D explicitly external reads.
    pub fn checked_source_union(n: u64, k: u64, b: usize, d: u64) -> Option<Self> {
        let visits = n.checked_add(k.checked_mul(2)?)?;
        let potential_bytes = u64::try_from(b).ok()?.checked_mul(2)?
            .checked_mul(visits.checked_add(1)?)?.checked_add(d)?;
        let result = Self { helper_calls: k, visits, potential_bytes, control_bytes: b };
        result.total_work()?;
        Some(result)
    }
    /// Compose already evaluated regions. Do not reapply the nonlinear recipe
    /// to their combined B/V, or count a future reserve as executed cleanup.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        let value = Self {
            helper_calls: self.helper_calls.checked_add(other.helper_calls)?,
            visits: self.visits.checked_add(other.visits)?,
            potential_bytes: self.potential_bytes.checked_add(other.potential_bytes)?,
            control_bytes: self.control_bytes.checked_add(other.control_bytes)?,
        };
        value.total_work()?;
        Some(value)
    }
    pub fn checked_scale(self, count: u64) -> Option<Self> {
        let value = Self {
            helper_calls: self.helper_calls.checked_mul(count)?,
            visits: self.visits.checked_mul(count)?,
            potential_bytes: self.potential_bytes.checked_mul(count)?,
            control_bytes: self.control_bytes.checked_mul(usize::try_from(count).ok()?)?,
        };
        value.total_work()?;
        Some(value)
    }
    pub fn total_work(self) -> Option<u64> {
        self.helper_calls.checked_add(self.visits)?.checked_add(self.potential_bytes)
    }
}

/// File-specialized source unions. Creation includes three possible path calls
/// and one possible typed check, NOT an observed trace of a refusal. The private
/// ambient-factory alternative is conservatively reserved too; it is never an
/// append occurrence. Concrete directory interiors are supplied by their owner.
/// The last three fields are separate remaining-owner obligations: a successful
/// finish executes none of them. Failed write retirement belongs to write v1;
/// failed finish partial retirement belongs to finish_file, not pending discard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicFileLifecycleLogicalQuote {
    pub create_computed_in: PublicFileLogicalWork,
    pub create_authenticated_in: PublicFileLogicalWork,
    pub finish_file: PublicFileLogicalWork,
    pub discard_pending_file: PublicFileLogicalWork,
    pub release_snapshot_handle: PublicFileLogicalWork,
    pub release_last_file_backing: PublicFileLogicalWork,
}

// Static source census only. No registry, operation, allocation or owner query.
// N includes zero-sized items/unit returns. Every row is a nominal endpoint;
// calls are counted separately, including Try branch/residual conversion unions.
#[derive(Default)]
struct LifecycleCensus { nominal: u64, bytes: usize, calls: u64 }
impl LifecycleCensus {
    fn row<T>(&mut self, n: usize) -> Option<()> {
        self.nominal = self.nominal.checked_add(u64::try_from(n).ok()?)?;
        self.bytes = self.bytes.checked_add(n.checked_mul(std::mem::size_of::<T>())?)?;
        Some(())
    }
    fn calls(&mut self, n: u64) -> Option<()> {
        self.calls = self.calls.checked_add(n)?;
        Some(())
    }
    fn quote(self) -> Option<PublicFileLogicalWork> {
        PublicFileLogicalWork::checked_source_union(self.nominal, self.calls, self.bytes, 0)
    }
}

fn lifecycle_mapper<R, E, T, F: FnOnce(E) -> T>(
    c: &mut LifecycleCensus, _item: F,
) -> Option<()> {
    c.row::<F>(1)?;
    c.row::<(Result<R, E>, F)>(1)?;
    c.row::<(F, (E,))>(1)?;
    c.row::<E>(1)?;
    c.row::<T>(3)?; // construction, normal return, receipt (not after unwind)
    c.calls(2)?; // adapter always; mapper invocation Err-only, union
    Some(())
}
fn lifecycle_io_drop(c: &mut LifecycleCensus) -> Option<()> {
    fn item<F>(c: &mut LifecycleCensus, _: F) -> Option<()> { c.row::<F>(1) }
    item(c, std::mem::drop::<io::Error>)?;
    c.row::<(io::Error,)>(2)?;
    c.row::<&mut io::Error>(1)?;
    c.row::<()>(2)?;
    c.calls(2)?; // drop invocation is distinct from native destructor boundary
    Some(())
}

fn lifecycle_creation(authenticated: bool) -> Option<PublicFileLogicalWork> {
    use std::{convert::Infallible, fs::Metadata, ops::ControlFlow};
    type A = Arc<dyn SnapshotDirectory>;
    type C = PublicFileCreateError;
    type O = PublicBackingError;
    type D = SnapshotDirectoryError;
    let mut c = LifecycleCensus::default();
    macro_rules! r { ($n:expr, $t:ty) => { c.row::<$t>($n)? }; }
    if authenticated { r!(2, (u64, [u8;32])); } else { r!(2, u64); }
    r!(3, A); r!(1, Option<[u8;32]>);
    r!(2, (u64, Option<[u8;32]>, Option<A>));
    r!(4, Result<PublicFileWriter,C>);
    c.calls(1)?; // explicit wrapper -> shared body
    r!(2, (u64, Option<[u8;32]>, Option<A>));
    r!(2, u64); r!(3, bool); r!(1, C);
    r!(1, &Option<A>); r!(1, &A);
    // Three dynamic boundaries belong here, never in concrete interiors.
    r!(6, (&A,)); r!(6, &dyn SnapshotDirectory);
    r!(6, &dyn SnapshotDirectory); r!(6, &Path);
    r!(2, &Path); r!(2, bool); r!(2, &Path);
    r!(4, &std::ffi::OsStr); r!(3, usize); r!(2, bool); r!(1, C);
    c.calls(9)?; // 3 Arc derefs + 3 path + absolute/as_os_str/len
    r!(2, (&Path,)); r!(2, ());
    r!(4, Result<File,io::Error>); r!(2, Result<File,C>);
    r!(2, ControlFlow<Result<Infallible,C>,File>); r!(2, Result<Infallible,C>);
    r!(3, File);
    lifecycle_mapper::<File,io::Error,C,_>(&mut c, creation_io_error)?;
    lifecycle_io_drop(&mut c)?;
    lifecycle_mapper::<File,io::Error,C,_>(&mut c, creation_io_error)?;
    lifecycle_io_drop(&mut c)?;
    c.calls(6)?; // two alternative factories, two Try branches/residuals each
    r!(2, (&File,)); r!(2, Result<Option<FileBacking>,O>);
    r!(1, Result<Option<FileBacking>,C>);
    r!(1, ControlFlow<Result<Infallible,C>,Option<FileBacking>>);
    r!(1, Result<Infallible,C>); r!(1, Option<FileBacking>); r!(1, C);
    r!(1, Result<FileBacking,C>); r!(1, ControlFlow<Result<Infallible,C>,FileBacking>);
    r!(1, Result<Infallible,C>);
    lifecycle_mapper::<Option<FileBacking>,O,C,_>(&mut c, C::Observation)?;
    c.calls(6)?; // observe, eager ok_or, two Try branch/residual pairs
    r!(2, (&dyn SnapshotDirectory,FileBacking)); r!(2, (&A,));
    r!(2, &dyn SnapshotDirectory); r!(2, Result<(),D>); r!(1, Result<(),C>);
    r!(1, ControlFlow<Result<Infallible,C>,()>); r!(1, Result<Infallible,C>);
    lifecycle_mapper::<(),D,C,_>(&mut c, C::Directory)?;
    c.calls(4)?; // Arc deref, dynamic check, Try branch/residual
    r!(1, File); r!(1, Option<A>); r!(2, u64); r!(1, Option<[u8;32]>);
    r!(3, Sha256); r!(1, u32); r!(1, Pending<File>);
    r!(1, Option<Pending<File>>); r!(1, PublicFileWriter);
    r!(2, Result<PublicFileWriter,C>);
    r!(1, &mut File); r!(1, &mut Option<A>);
    c.calls(3)?; // hash new; initialized File/directory refusal endpoints
    #[cfg(unix)] {
        r!(2, (&File,)); r!(2, Result<Metadata,io::Error>);
        r!(1, Result<Metadata,O>); r!(1, ControlFlow<Result<Infallible,O>,Metadata>);
        r!(1, Result<Infallible,O>); r!(1, Metadata);
        r!(6, (&Metadata,)); r!(6, u64); r!(2, (u64,u64));
        r!(2, Option<u64>); r!(1, O); r!(1, Result<u64,O>);
        r!(1, ControlFlow<Result<Infallible,O>,u64>); r!(1, Result<Infallible,O>);
        r!(4, u64); r!(1, FileBacking); r!(1, Option<FileBacking>);
        r!(2, Result<Option<FileBacking>,O>); r!(1, &mut Metadata);
        lifecycle_mapper::<Metadata,io::Error,O,_>(&mut c, backing_io_error)?;
        lifecycle_io_drop(&mut c)?;
        c.calls(11)?; // metadata/blocks/mul/ok_or/dev/ino/drop + four Try calls
    }
    #[cfg(not(unix))] { r!(1, &File); r!(2, Result<Option<FileBacking>,O>); }
    c.quote()
}

fn lifecycle_finish_callback<F: FnOnce([u8;32]) -> bool>(
    c: &mut LifecycleCensus, _callback: F,
) -> Option<()> {
    c.row::<F>(4)?; // construction/return/receipt OR unused None consumption
    c.row::<(Option<[u8;32]>,F)>(1)?;
    c.row::<(F,([u8;32],))>(1)?;
    Some(())
}
fn lifecycle_finish() -> Option<PublicFileLogicalWork> {
    use std::{convert::Infallible, ops::ControlFlow};
    type E = PublicFileFinishError;
    type A = Arc<dyn SnapshotDirectory>;
    let mut c = LifecycleCensus::default();
    macro_rules! r { ($n:expr, $t:ty) => { c.row::<$t>($n)? }; }
    r!(2, PublicFileWriter); r!(2, (&mut Option<Pending<File>>,));
    r!(2, Option<Pending<File>>); r!(1, E); r!(1, Result<Pending<File>,E>);
    r!(1, ControlFlow<Result<Infallible,E>,Pending<File>>); r!(1, Result<Infallible,E>);
    r!(1, Pending<File>); r!(2, (Sha256,)); r!(2, sha2::digest::Output<Sha256>);
    r!(2, [u8;32]); r!(2, u64); r!(2, bool); r!(1, Option<[u8;32]>);
    r!(2, (&[u8;32],)); r!(1, &[u8;32]); r!(1, [u8;32]);
    r!(2, ([u8;32],)); r!(2, [u8;32]); r!(1, &[u8;32]); r!(1, [u8;32]);
    // PartialEq::ne actual/formal arguments: reached Some only; None reserves
    // this alternative in the whole-body union, without invoking comparison.
    r!(2, (&[u8;32], &[u8;32]));
    r!(6, bool); r!(1, E);
    r!(3, File); r!(3, Option<A>); r!(3, PublicFileBacking<File>);
    r!(3, Arc<PublicFileBacking<File>>); r!(2, u64); r!(1, [u8;32]); r!(2, u32);
    r!(2, PublicFile); r!(2, Result<PublicFile,E>);
    r!(1, &mut File); r!(1, &mut Option<A>); r!(1, &mut PublicFileWriter);
    // No eventual backing retirement here: it is a separate future obligation.
    fn factory<F>(c: &mut LifecycleCensus, _: F) -> Option<()> { c.row::<F>(1) }
    factory(&mut c, finish_digest_callback)?;
    let digest = [0;32];
    lifecycle_finish_callback(&mut c, finish_digest_callback(&digest))?;
    // take/ok_or/Try x2/finalize/Into/factory/is_some_and/FnOnce/compare/
    // unused F disposal/Arc new/partial File+directory/empty self retirement.
    c.calls(15)?;
    c.quote()
}

fn lifecycle_remaining(kind: u8) -> Option<PublicFileLogicalWork> {
    type A = Arc<dyn SnapshotDirectory>;
    let mut c = LifecycleCensus::default();
    macro_rules! r { ($n:expr, $t:ty) => { c.row::<$t>($n)? }; }
    match kind {
        0 => {
            // Discard still-live Pending only. Wrapper/Option/fields retire;
            // no call to take is invented for Rust's generated drop glue.
            r!(2, (&mut PublicFileWriter,)); r!(1, Option<Pending<File>>);
            r!(1, bool); r!(2, (&mut Pending<File>,)); r!(1, File);
            r!(1, Option<A>); r!(2, u64); r!(1, Option<[u8;32]>);
            r!(1, Sha256); r!(1, u32); r!(2, (&mut File,));
            r!(2, (&mut Option<A>,)); r!(2, (&mut Sha256,)); r!(6, ());
            c.calls(4)?; // pending glue; File; directory option; hash retirement
        }
        1 => {
            r!(1, PublicFile); r!(2, (&mut PublicFile,));
            r!(2, (&mut Arc<PublicFileBacking<File>>,));
            r!(1, u64); r!(1, [u8;32]); r!(1, u32); r!(2, ());
            c.calls(2)?; // snapshot glue + Arc decrement, not last backing body
        }
        2 => {
            r!(2, (&mut PublicFileBacking<File>,));
            r!(2, (&mut File,)); r!(2, ());
            r!(2, (&mut Option<A>,)); r!(1, bool);
            r!(2, (&mut A,)); r!(2, ());
            r!(1, PublicFileBacking<File>); r!(2, (&mut PublicFileBacking<File>,));
            r!(2, ()); c.calls(4)?; // glue/File/Arc decrement/deallocation boundary
        }
        _ => return None,
    }
    c.quote()
}

/// Pure File lifecycle profile v1. Query setup, real-F witness construction and
/// disposal, checked arithmetic and cache transport require predecessor funding.
/// No filesystem call, constructor, lock or owner cleanup runs to obtain prices.
/// Native internals and arbitrary directory final-owner retirement remain OPEN;
/// keepalive requires genuine custody through every escaped reader, not a count.
pub fn public_file_lifecycle_logical_quote_v1() -> Option<PublicFileLifecycleLogicalQuote> {
    Some(PublicFileLifecycleLogicalQuote {
        create_computed_in: lifecycle_creation(false)?,
        create_authenticated_in: lifecycle_creation(true)?,
        finish_file: lifecycle_finish()?,
        discard_pending_file: lifecycle_remaining(0)?,
        release_snapshot_handle: lifecycle_remaining(1)?,
        release_last_file_backing: lifecycle_remaining(2)?,
    })
}

// Statement-only sharing preserves legacy bookkeeping, without another frame,
// argument slots, bindings, or cloning write authority.
macro_rules! complete_append {
    ($pending:ident, $bytes:ident) => {
        $pending.hash.update($bytes);
        $pending.written += $bytes.len() as u64;
        $pending.max_i8 = $pending.max_i8.max(
            $bytes
                .iter()
                .map(|&v| (v as i8 as i32).unsigned_abs())
                .max()
                .unwrap_or(0),
        );
    };
}

// Constructor census (one occurrence unless noted): size actual/formal, two
// comparisons/short circuit/Err; directory Option borrow/match; path actual/formal
// and borrowed Path three times, absolute/path scan and OsStr/length; policy Err;
// Some factory (&Path,) OR None factory (), native return/map receiver, mapper
// item/adapter (evaluated on Ok too), error invocation tuple/drop/enum return;
// observation (&File,) return/map/ok_or/?; owner (&dyn SnapshotDirectory,
// FileBacking) return/map/?; branch File move; Pending fields individually
// file/directory/size/expected/0/hash-new/0, Some/writer/Ok/received return.
// Every ? has mapped Result, ControlFlow<Result<Infallible,E>,T> and residual.
// Refusal/unwind retires only initialized prefix: local File before directory;
// success transfers both. Tempfile/RNG/path scans, metadata, arbitrary owner
// calls/drop, hash, File close and native error destructors remain opaque.
macro_rules! emit_with_directory {
    ($name:ident, $size:ident, $expected:ident, $directory:ident, $file:ident,
     $owner:ident, $error:ty, $bounds:expr, $policy:expr, $mapper:expr,
     $check:expr) => {
    fn $name(
        $size: u64,
        $expected: Option<[u8; 32]>,
        $directory: Option<Arc<dyn SnapshotDirectory>>,
    ) -> Result<Self, $error> {
        if $size == 0 || $size > MAX_BYTES {
            return Err($bounds);
        }
        let $file = match &$directory {
            Some($owner) => {
                if !$owner.path().is_absolute()
                    || $owner.path().as_os_str().len() > MAX_DIRECTORY_PATH_BYTES - 32
                {
                    return Err($policy);
                }
                let $file = tempfile::tempfile_in($owner.path()).map_err($mapper)?;
                $check;
                $file
            }
            None => tempfile::tempfile().map_err($mapper)?,
        };
        Ok(Self {
            pending: Some(Pending {
                $file,
                $directory,
                $size,
                $expected,
                written: 0,
                hash: Sha256::new(),
                max_i8: 0,
            }),
        })
    }
    };
}

impl PublicFileWriter {
    // Wrapper census, each once: size actual/formal, expected actual/formal
    // and Some OR None, directory actual/formal/Some move; with_directory_typed
    // item and (u64,Option<[u8;32]>,Option<Arc<dyn SnapshotDirectory>>) actual/
    // formal; callee Result return/wrapper receipt/outer return/caller receipt.
    /// Typed explicit-directory creation. Native transients and cleanup remain
    /// opaque; the fixed returned error owns no backing. No funding is minted.
    pub fn new_computed_in_typed(size: u64, directory: Arc<dyn SnapshotDirectory>)
        -> Result<Self, PublicFileCreateError>
    {
        Self::with_directory_typed(size, None, Some(directory))
    }
    pub fn new_in_typed(size: u64, expected: [u8; 32], directory: Arc<dyn SnapshotDirectory>)
        -> Result<Self, PublicFileCreateError>
    {
        Self::with_directory_typed(size, Some(expected), Some(directory))
    }
    pub fn new(size: u64, expected: [u8; 32]) -> Result<Self, String> {
        Self::with_expected(size, Some(expected))
    }
    /// Commit a native producer's bytes without a precomputed digest. The sealed
    /// digest is always computed over the exact output, never supplied by a caller.
    /// Source authentication and disk reservation remain the caller's responsibility.
    pub fn new_computed(size: u64) -> Result<Self, String> {
        Self::with_expected(size, None)
    }
    pub fn new_in(
        size: u64,
        expected: [u8; 32],
        directory: Arc<dyn SnapshotDirectory>,
    ) -> Result<Self, String> {
        Self::with_directory(size, Some(expected), Some(directory))
    }
    pub fn new_computed_in(
        size: u64,
        directory: Arc<dyn SnapshotDirectory>,
    ) -> Result<Self, String> {
        Self::with_directory(size, None, Some(directory))
    }
    fn with_expected(size: u64, expected: Option<[u8; 32]>) -> Result<Self, String> {
        Self::with_directory(size, expected, None)
    }
    emit_with_directory!(with_directory, size, expected, directory, file, owner, String,
        "public snapshot size exceeds bounds".into(),
        "snapshot directory exceeds explicit path policy".into(), |e| e.to_string(),
        owner.check_file(
            observe_backing(&file)?.ok_or("snapshot filesystem observation unsupported")?,
        )?);
    emit_with_directory!(with_directory_typed, size, expected, directory, file, owner,
        PublicFileCreateError, PublicFileCreateError::Bounds, PublicFileCreateError::DirectoryPolicy,
        creation_io_error,
        owner.check_file_typed(
            observe_backing_typed(&file).map_err(PublicFileCreateError::Observation)?
                .ok_or(PublicFileCreateError::ObservationUnsupported)?,
        ).map_err(PublicFileCreateError::Directory)?);
    /// Any failed append retires the incomplete file, including an invalid size.
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), String> {
        let mut pending = self
            .pending
            .take()
            .ok_or("public snapshot writer is closed")?;
        if bytes.is_empty()
            || bytes.len() > MAX_CHUNK_BYTES
            || pending
                .written
                .checked_add(bytes.len() as u64)
                .is_none_or(|end| end > pending.size)
        {
            return Err("public snapshot chunk exceeds bounds".into());
        }
        pending.file.write_all(bytes).map_err(|e| e.to_string())?;
        complete_append!(pending, bytes);
        self.pending = Some(pending);
        Ok(())
    }

    /// Optional constant-payload append, with the same bookkeeping as `write`.
    /// Taking Pending first makes Bounds and Io sticky retirement, not retries.
    /// Native I/O errors are dropped unformatted; transient std errors are not
    /// claimed heap-free. No permission is stored or minted. See
    /// `public_write_logical_quote_v1` for opaque boundaries and required caller
    /// funding/other directory ownership; this is not per-physical-attempt I/O.
    pub fn write_typed(&mut self, bytes: &[u8]) -> Result<(), PublicWriteError> {
        let mut pending = self.pending.take().ok_or(PublicWriteError::Closed)?;
        if bytes.is_empty()
            || bytes.len() > MAX_CHUNK_BYTES
            || pending.written.checked_add(bytes.len() as u64)
                .is_none_or(|end| end > pending.size)
        {
            return Err(PublicWriteError::Bounds);
        }
        pending.file.write_all(bytes).map_err(|_| PublicWriteError::Io)?;
        complete_append!(pending, bytes);
        self.pending = Some(pending);
        Ok(())
    }
}

// Finish census: self/Option take/ok_or/Result/ControlFlow/residual/Pending;
// hash partial move/finalize/output/Into/digest; written,size,!=,||; expected
// Option/is_some_and receiver. Typed factory only when sizes match: function
// item, (&digest,) actual/formal, borrowed capture/closure/return/receipt;
// (Option<[u8;32]>,F) adapter, FnOnce self/(v,) comparison/bool/receipt on Some,
// unused callback disposal on None. Digest referent stays live (32 bytes).
// Err/return/receipt OR backing field moves, Arc::new actual/formal/return,
// size/digest/max fields, snapshot/Ok/receipt. Retire remaining partial Pending
// fields and empty self; successful backing retires only at last strong owner.
// Generic B drop, hash, 32-byte comparison, Arc allocation/counters, File close,
// directory cleanup are named opaque endpoints, never native closure or ACK.
macro_rules! emit_finish {
    ($name:ident, $self:ident, $pending:ident, $digest:ident, $B:ident,
     $error:ty, $closed:expr, $mismatch:expr, $callback:expr) => {
    pub fn $name(mut $self) -> Result<PublicFileStorage<$B>, $error> {
        let $pending = $self
            .pending
            .take()
            .ok_or($closed)?;
        let $digest: [u8; 32] = $pending.hash.finalize().into();
        if $pending.written != $pending.size || $pending.expected.is_some_and($callback) {
            return Err($mismatch);
        }
        Ok(PublicFileStorage {
            file: Arc::new(PublicFileBacking {
                file: $pending.file,
                _directory: $pending.directory,
            }),
            size: $pending.size,
            $digest,
            max_i8: $pending.max_i8,
        })
    }
    };
}
impl<B> PublicFileWriterStorage<B> {
    emit_finish!(finish, self, pending, digest, B, String,
        "public snapshot writer is closed", "public snapshot digest or size differs".into(),
        |v| v != digest);
    emit_finish!(finish_typed, self, pending, digest, B, PublicFileFinishError,
        PublicFileFinishError::Closed, PublicFileFinishError::ContentMismatch,
        finish_digest_callback(&digest));
}

#[cfg(test)]
#[path = "public_file_lifecycle_tests.rs"]
mod lifecycle_tests;

/// One proposed physical attempt; authorization is borrowed, never retained.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoAttempt {
    Observe,
    Read { bytes: usize, offset: u64 },
    Write { bytes: usize, logical_offset: u64 },
}

#[cfg(unix)]
#[derive(Debug)]
pub enum FacadeError<E> {
    Authorization(E),
    Bounds,
    Closed,
    Backend,
    DescriptorUnavailable,
    Observe(ObserveOnceError),
    Io(IoOnceError),
    UnexpectedEof,
    WriteZero,
    InvalidBacking,
}

#[cfg(unix)]
impl PublicFileWriterStorage<FileLease> {
    /// Requires cooperative exclusive mutation and cursor zero before adoption,
    /// throughout writing, and immutability after sealing. Empty regular-file
    /// observation proves neither cursor position nor exclusivity. Caller keeps
    /// external coordinator custody alive on every result and unwind; dropping
    /// this writer only drops its reader lease, never closes or acknowledges.
    pub fn from_resident<E>(
        size: u64, expected: Option<[u8; 32]>, reader: FileLease,
        authorize: &mut impl FnMut(IoAttempt) -> Result<(), E>,
    ) -> Result<Self, FacadeError<E>> {
        if size == 0 || size > MAX_BYTES { return Err(FacadeError::Bounds); }
        authorize(IoAttempt::Observe).map_err(FacadeError::Authorization)?;
        let observation = observe_once(reader.descriptor().ok_or(FacadeError::DescriptorUnavailable)?)
            .map_err(FacadeError::Observe)?;
        if observation.mode & libc::S_IFMT as u32 != libc::S_IFREG as u32
            || observation.length != 0
        {
            return Err(FacadeError::InvalidBacking);
        }
        Ok(Self { pending: Some(Pending {
            file: reader, directory: None, size, expected, written: 0,
            hash: Sha256::new(), max_i8: 0,
        }) })
    }

    /// Every unsuccessful append poisons the writer, including authorization
    /// refusal, partial I/O, invalid input and unwinding. No cursor repair/retry.
    pub fn write_authorized<E>(
        &mut self, bytes: &[u8],
        authorize: &mut impl FnMut(IoAttempt) -> Result<(), E>,
    ) -> Result<(), FacadeError<E>> {
        self.write_authorized_with(bytes, authorize, write_once)
    }

    // Per-call scalar-result seam. No stored hooks, descriptor fabrication, or
    // callback-free public execution. Production and regressions share this loop.
    fn write_authorized_with<E>(
        &mut self, bytes: &[u8],
        authorize: &mut impl FnMut(IoAttempt) -> Result<(), E>,
        mut attempt: impl FnMut(std::os::fd::BorrowedFd<'_>, &[u8]) -> Result<usize, IoOnceError>,
    ) -> Result<(), FacadeError<E>> {
        let mut pending = self.pending.take().ok_or(FacadeError::Closed)?;
        if bytes.is_empty() || bytes.len() > MAX_CHUNK_BYTES
            || pending.written.checked_add(bytes.len() as u64).is_none_or(|end| end > pending.size)
        { return Err(FacadeError::Bounds); }
        let mut done = 0;
        while done < bytes.len() {
            authorize(IoAttempt::Write {
                bytes: bytes.len() - done, logical_offset: pending.written + done as u64,
            }).map_err(FacadeError::Authorization)?;
            let n = attempt(pending.file.descriptor().ok_or(FacadeError::DescriptorUnavailable)?,
                &bytes[done..]).map_err(FacadeError::Io)?;
            if n == 0 { return Err(FacadeError::WriteZero); }
            if n > bytes.len() - done { return Err(FacadeError::Backend); }
            done += n;
        }
        complete_append!(pending, bytes);
        self.pending = Some(pending);
        Ok(())
    }
}

#[cfg(unix)]
impl PublicFileStorage<FileLease> {
    /// Failure may leave a modified prefix; output is unusable on any error.
    pub fn read_authorized<E>(
        &self, offset: u64, dst: &mut [u8],
        authorize: &mut impl FnMut(IoAttempt) -> Result<(), E>,
    ) -> Result<(), FacadeError<E>> {
        self.file.read_authorized(self.size, offset, dst, authorize)
    }

    pub fn observe_authorized<E>(
        &self, authorize: &mut impl FnMut(IoAttempt) -> Result<(), E>,
    ) -> Result<FileObservation, FacadeError<E>> {
        authorize(IoAttempt::Observe).map_err(FacadeError::Authorization)?;
        observe_once(self.file.file.descriptor().ok_or(FacadeError::DescriptorUnavailable)?)
            .map_err(FacadeError::Observe)
    }
}

#[cfg(unix)]
impl PublicFileBacking<FileLease> {
    /// Caller supplies its sealed logical length; backing carries no such length.
    /// Failure may modify a prefix; output is unusable on any error. Controls are
    /// unqualified for funding. Reader-lease drop never acknowledges retirement;
    /// external coordinator retains descriptor custody for explicit retirement.
    pub(crate) fn read_authorized<E>(
        &self, size: u64, offset: u64, dst: &mut [u8],
        authorize: &mut impl FnMut(IoAttempt) -> Result<(), E>,
    ) -> Result<(), FacadeError<E>> {
        self.read_authorized_with(size, offset, dst, authorize, read_at_once)
    }

    fn read_authorized_with<E>(
        &self, size: u64, offset: u64, dst: &mut [u8],
        authorize: &mut impl FnMut(IoAttempt) -> Result<(), E>,
        mut attempt: impl FnMut(std::os::fd::BorrowedFd<'_>, &mut [u8], u64) -> Result<usize, IoOnceError>,
    ) -> Result<(), FacadeError<E>> {
        if dst.is_empty() || dst.len() > MAX_CHUNK_BYTES
            || offset.checked_add(dst.len() as u64).is_none_or(|end| end > size)
        { return Err(FacadeError::Bounds); }
        let mut done = 0;
        while done < dst.len() {
            authorize(IoAttempt::Read { bytes: dst.len() - done, offset: offset + done as u64 })
                .map_err(FacadeError::Authorization)?;
            let remaining = dst.len() - done;
            let n = attempt(self.file.descriptor().ok_or(FacadeError::DescriptorUnavailable)?,
                &mut dst[done..], offset + done as u64).map_err(FacadeError::Io)?;
            if n == 0 { return Err(FacadeError::UnexpectedEof); }
            if n > remaining { return Err(FacadeError::Backend); }
            done += n;
        }
        Ok(())
    }

}

#[cfg(all(test, unix))]
#[path = "public_file_resident_tests.rs"]
mod resident_tests;

pub(crate) fn read_at(file: &File, data: &mut [u8], offset: u64) -> Result<(), String> {
    read_at_unformatted(file, data, offset).map_err(format_read_error)
}

fn read_at_unformatted(file: &File, data: &mut [u8], offset: u64) -> Result<(), PublicReadError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        match file.read_exact_at(data, offset) {
            Ok(()) => Ok(()),
            Err(error) => Err(PublicReadError::Io(error)),
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < data.len() {
            let n = match file.seek_read(&mut data[done..], offset + done as u64) {
                Ok(n) => n,
                Err(error) => return Err(PublicReadError::Io(error)),
            };
            if n == 0 {
                return Err(PublicReadError::Truncated);
            }
            done += n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Location {
        directory: tempfile::TempDir,
        reject: bool,
    }
    impl SnapshotDirectory for Location {
        fn path(&self) -> &Path {
            self.directory.path()
        }
        fn check_file(&self, file: FileBacking) -> Result<(), String> {
            if self.reject {
                return Err("foreign filesystem".into());
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if file.device != self.directory.path().metadata().unwrap().dev() {
                    return Err("foreign filesystem".into());
                }
            }
            Ok(())
        }
    }

    #[test]
    fn typed_writes_match_legacy_all_bytes_and_signed_abs_edge_in_two_frames() {
        let all: Vec<u8> = (0..=255).collect();
        for bytes in [all.as_slice(), &[0, 127, 128, 255]] {
            let expected: [u8; 32] = Sha256::digest(bytes).into();
            for authenticate in [false, true] {
                let make = || if authenticate {
                    PublicFileWriter::new(bytes.len() as u64, expected).unwrap()
                } else {
                    PublicFileWriter::new_computed(bytes.len() as u64).unwrap()
                };
                let mut typed = make();
                let mut legacy = make();
                let mid = bytes.len() / 2;
                for frame in [&bytes[..mid], &bytes[mid..]] {
                    typed.write_typed(frame).unwrap();
                    legacy.write(frame).unwrap();
                }
                let typed = typed.finish().unwrap();
                let legacy = legacy.finish().unwrap();
                for snapshot in [&typed, &legacy] {
                    assert_eq!(snapshot.size(), bytes.len() as u64);
                    assert_eq!(snapshot.digest(), expected);
                    assert_eq!(snapshot.max_i8(), 128);
                    let mut actual = vec![0; bytes.len()];
                    snapshot.read(0, &mut actual).unwrap();
                    assert_eq!(actual, bytes);
                }
            }
        }
    }

    fn assert_typed_closed_sticky(mut writer: PublicFileWriter) {
        assert!(writer.pending.is_none());
        assert_eq!(writer.write_typed(b"x"), Err(PublicWriteError::Closed));
        assert_eq!(writer.write_typed(&[]), Err(PublicWriteError::Closed));
        assert_eq!(writer.write(b"x").unwrap_err(), "public snapshot writer is closed");
        assert!(writer.pending.is_none());
        assert_eq!(writer.finish().err().unwrap(), "public snapshot writer is closed");
    }

    #[test]
    fn typed_write_bounds_retire_and_preserve_legacy_diagnostics() {
        let oversized = vec![0; MAX_CHUNK_BYTES + 1];
        for bytes in [b"".as_slice(), b"abcde", oversized.as_slice()] {
            let mut writer = PublicFileWriter::new_computed(4).unwrap();
            assert_eq!(writer.write_typed(bytes), Err(PublicWriteError::Bounds));
            assert_typed_closed_sticky(writer);
            let mut legacy = PublicFileWriter::new_computed(4).unwrap();
            assert_eq!(legacy.write(bytes).unwrap_err(), "public snapshot chunk exceeds bounds");
            assert_typed_closed_sticky(legacy);
        }
        let mut writer = PublicFileWriter::new_computed(4).unwrap();
        writer.write_typed(b"abc").unwrap();
        assert_eq!(writer.write_typed(b"de"), Err(PublicWriteError::Bounds));
        assert_typed_closed_sticky(writer);

        // Private scalar injection exercises checked-add overflow only; not
        // evidence of a production source or ownership/admission authority.
        let mut writer = PublicFileWriter::new_computed(4).unwrap();
        writer.pending.as_mut().unwrap().written = u64::MAX;
        assert_eq!(writer.write_typed(b"x"), Err(PublicWriteError::Bounds));
        assert_typed_closed_sticky(writer);
    }

    #[test]
    fn typed_write_error_has_only_fixed_unit_payloads() {
        fn copy_eq<T: Copy + Eq>() {}
        copy_eq::<PublicWriteError>();
        assert!(!std::mem::needs_drop::<PublicWriteError>());
        assert!(!std::mem::needs_drop::<Result<(), PublicWriteError>>());
        assert_eq!(std::mem::size_of::<PublicWriteError>(), 1);
        assert_ne!(PublicWriteError::Closed, PublicWriteError::Bounds);
        assert_ne!(PublicWriteError::Closed, PublicWriteError::Io);
        assert_ne!(PublicWriteError::Bounds, PublicWriteError::Io);
    }

    #[cfg(unix)]
    #[test]
    fn typed_bounds_release_pending_directory_but_keep_other_owner() {
        let owner = Arc::new(Location { directory: tempfile::tempdir().unwrap(), reject: false });
        let weak = Arc::downgrade(&owner);
        let path = owner.path().to_path_buf();
        let mut writer = PublicFileWriter::new_computed_in(4, owner.clone()).unwrap();
        assert_eq!(Arc::strong_count(&owner), 2);
        assert_eq!(writer.write_typed(&[]), Err(PublicWriteError::Bounds));
        assert_eq!(Arc::strong_count(&owner), 1);
        assert!(weak.upgrade().is_some());
        assert!(path.is_dir());
        assert_typed_closed_sticky(writer);
        drop(owner);
        assert!(weak.upgrade().is_none());
        assert!(!path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn typed_write_real_readonly_os_failure_retires_pending_and_directory_arc() {
        let owner = Arc::new(Location { directory: tempfile::tempdir().unwrap(), reject: false });
        let weak = Arc::downgrade(&owner);
        let path = owner.path().to_path_buf();
        // New test-only fixture: a genuine owned read-only File replaces only
        // the private Pending.file of a constructor-created writer. No fake fd,
        // injected OS result or production failure hook. Setup is unpriced.
        let mut named = tempfile::NamedTempFile::new().unwrap();
        named.write_all(b"kept").unwrap();
        let readonly = File::open(named.path()).unwrap();
        let mut writer = PublicFileWriter::new_computed_in(4, owner.clone()).unwrap();
        let old = std::mem::replace(&mut writer.pending.as_mut().unwrap().file, readonly);
        drop(old);
        assert_eq!(Arc::strong_count(&owner), 2);
        assert_eq!(writer.write_typed(b"abcd"), Err(PublicWriteError::Io));
        assert_eq!(std::fs::read(named.path()).unwrap(), b"kept");
        assert_eq!(Arc::strong_count(&owner), 1);
        assert!(weak.upgrade().is_some());
        assert!(path.is_dir());
        assert_typed_closed_sticky(writer);
        drop(owner);
        assert!(weak.upgrade().is_none());
        assert!(!path.exists());
        // Proves actual OS error, not partial-write/retry behavior or close ACK.
    }

    #[cfg(unix)]
    #[test]
    fn typed_success_keeps_directory_until_last_snapshot_backing_release() {
        let owner = Arc::new(Location { directory: tempfile::tempdir().unwrap(), reject: false });
        let weak = Arc::downgrade(&owner);
        let path = owner.path().to_path_buf();
        let mut writer = PublicFileWriter::new_computed_in(4, owner.clone()).unwrap();
        writer.write_typed(b"abcd").unwrap();
        assert_eq!(Arc::strong_count(&owner), 2);
        let snapshot = writer.finish().unwrap();
        let backing = snapshot.file();
        drop(owner);
        drop(snapshot);
        assert!(weak.upgrade().is_some());
        let mut bytes = [0; 4];
        read_at(&backing, &mut bytes, 0).unwrap();
        assert_eq!(&bytes, b"abcd");
        assert!(path.is_dir());
        drop(backing);
        assert!(weak.upgrade().is_none());
        assert!(!path.exists());
    }

    // Independent row inventory, deliberately not the production layout sum.
    fn independent_write_pools() -> (usize, usize, usize) {
        use std::{convert::Infallible, mem::size_of as s, ops::ControlFlow};
        let method = s::<&mut PublicFileWriter>() + s::<&[u8]>() + s::<PublicFileWriter>();
        let take_restore = 2*s::<Option<Pending<File>>>() + s::<Pending<File>>();
        let take_propagation = s::<Result<Pending<File>, PublicWriteError>>()
            + s::<ControlFlow<Result<Infallible, PublicWriteError>, Pending<File>>>()
            + s::<Result<Infallible, PublicWriteError>>();
        let guard = s::<usize>() + 2*s::<u64>() + s::<Option<u64>>()
            + 4*s::<bool>() + s::<&u64>();
        let io = s::<&mut File>() + s::<&[u8]>()
            + 2*s::<Result<(), io::Error>>() + s::<io::Error>();
        let propagation = s::<Result<(), PublicWriteError>>()
            + s::<ControlFlow<Result<Infallible, PublicWriteError>, ()>>()
            + s::<Result<Infallible, PublicWriteError>>();
        let hash = s::<&mut Sha256>() + s::<&[u8]>() + 2*s::<()>();
        let stats = s::<&[u8]>() + s::<Option<u32>>() + 2*s::<u32>();
        let count = s::<usize>() + 2*s::<u64>();
        let returns = s::<PublicWriteError>() + 2*s::<Result<(), PublicWriteError>>();
        let retire = s::<&mut io::Error>() + s::<&mut File>()
            + s::<&mut Option<Arc<dyn SnapshotDirectory>>>()
            + s::<&mut Pending<File>>() + s::<&mut PublicFileWriter>();
        let fixed = method + take_restore + take_propagation + guard + io
            + propagation + hash + stats + count + returns + retire;
        let callback = 2*s::<&u8>() + s::<u8>() + s::<i8>() + 2*s::<i32>() + 4*s::<u32>();
        let reads = 2*s::<&u8>() + 2*s::<u8>() + s::<i8>() + 2*s::<i32>() + 3*s::<u32>();
        (fixed, callback, reads)
    }

    #[test]
    fn typed_write_quote_matches_independent_fixed_and_scalar_endpoint_equations() {
        let (fixed, callback, reads) = independent_write_pools();
        assert_eq!(PUBLIC_WRITE_LOGICAL_VERSION, 1);
        assert_eq!(WRITE_FIXED_BYTES, fixed);
        assert_eq!(WRITE_CALLBACK_BYTES, callback);
        assert_eq!(WRITE_CALLBACK_READ_BYTES, reads);
        for n in [0, 1, 32768, MAX_CHUNK_BYTES] {
            let quote = public_write_logical_quote_v1(n).unwrap();
            let n = u64::try_from(n).unwrap();
            assert_eq!(quote.helper_calls, 6 + 2*n);
            assert_eq!(quote.visits, 26 + 9*n);
            assert_eq!(quote.potential_bytes,
                54*fixed as u64 + 2*n + n*(2*callback as u64 + reads as u64));
            assert_eq!(quote.control_bytes, fixed + callback);
            assert_eq!(quote.returned_owned_bytes, 0);
            assert_eq!(quote.helper_calls.checked_add(quote.visits)
                .and_then(|kv| kv.checked_add(quote.potential_bytes)),
                Some(32 + 54*fixed as u64 + n*(13 + 2*callback as u64 + reads as u64)));
        }
        for n in [0, 1, 32768, MAX_CHUNK_BYTES - 1] {
            let a = public_write_logical_quote_v1(n).unwrap();
            let b = public_write_logical_quote_v1(n + 1).unwrap();
            assert_eq!(b.helper_calls - a.helper_calls, 2);
            assert_eq!(b.visits - a.visits, 9);
            assert_eq!(b.potential_bytes - a.potential_bytes, (2 + 2*callback + reads) as u64);
            assert_eq!(b.control_bytes, a.control_bytes);
        }
        if std::mem::size_of::<&u8>() == 8 {
            assert_eq!((callback, reads, 2*callback + reads), (42, 39, 123));
            assert_eq!(13 + 2*callback + reads, 136);
        }
        assert_eq!(public_write_logical_quote_v1(MAX_CHUNK_BYTES + 1), None);
        assert_eq!(public_write_logical_quote_v1(usize::MAX), None);
    }

    #[test]
    fn typed_write_quote_checked_overflow_and_owner_pool_affect_only_fixed_terms() {
        let (fixed, callback, reads) = independent_write_pools();
        for n in [0, 1, 32768, MAX_CHUNK_BYTES as u64] {
            // Private scalar extent injection, not a caller-settable tariff.
            let a = write_logical_quote_parts(n, fixed, callback, reads).unwrap();
            let b = write_logical_quote_parts(n, fixed + 1, callback, reads).unwrap();
            assert_eq!(b.helper_calls, a.helper_calls);
            assert_eq!(b.visits, a.visits);
            assert_eq!(b.potential_bytes - a.potential_bytes, 54);
            assert_eq!(b.control_bytes - a.control_bytes, 1);
        }
        assert_eq!(write_logical_quote_parts(u64::MAX, 0, 0, 0), None);
        assert_eq!(write_logical_quote_parts(0, usize::MAX, 1, 0), None); // S
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(write_logical_quote_parts(0, usize::MAX, 0, 0), None); // fixed P
            assert_eq!(write_logical_quote_parts(1, 0, usize::MAX, 0), None); // byte P
        }
        assert_eq!(write_logical_quote_parts(u64::MAX / 10, 0, 100, 100), None); // n*Pbyte
        assert_eq!(write_logical_quote_parts(u64::MAX / 9, 0, 0, 0), None); // V + 26
        if let Ok(fixed) = usize::try_from(u64::MAX / 54) {
            // floor(MAX/54) leaves 51 units, enough for K+V=32 at n=0.
            let p = u64::try_from(fixed).unwrap().checked_mul(54).unwrap();
            assert_eq!(u64::MAX - p, 51);
            let zero = write_logical_quote_parts(0, fixed, 0, 0).unwrap();
            assert_eq!(zero.potential_bytes, p);
            assert_eq!(zero.helper_calls.checked_add(zero.visits)
                .and_then(|kv| kv.checked_add(zero.potential_bytes)), Some(u64::MAX - 19));
            // The next fixed extent already overflows P, independently of K+V.
            let next = fixed.checked_add(1).unwrap();
            assert_eq!(u64::try_from(next).unwrap().checked_mul(54), None);
            assert_eq!(write_logical_quote_parts(0, next, 0, 0), None);
            // With empty callback pools, total slope is 2+9+2=13 per byte.
            let one = write_logical_quote_parts(1, fixed, 0, 0).unwrap();
            assert_eq!(one.helper_calls.checked_add(one.visits)
                .and_then(|kv| kv.checked_add(one.potential_bytes)), Some(u64::MAX - 6));
            // At n=2 each component fits, but their total exceeds MAX by 7.
            let two_p = p.checked_add(4).unwrap();
            assert_eq!(two_p, u64::MAX - 47);
            assert_eq!(10u64.checked_add(44).and_then(|kv| kv.checked_add(two_p)), None);
            assert_eq!(write_logical_quote_parts(2, fixed, 0, 0), None);
        }
        assert!(std::panic::catch_unwind(|| write_layout_sum([usize::MAX, 1])).is_err());
        assert!(std::panic::catch_unwind(|| write_layout_mul(usize::MAX, 2)).is_err());
    }

    #[test]
    fn explicit_directory_checks_actual_file_and_retains_last_backing() {
        let owner = Arc::new(Location {
            directory: tempfile::tempdir().unwrap(),
            reject: false,
        });
        let path = owner.path().to_path_buf();
        let mut writer = PublicFileWriter::new_computed_in(4, owner.clone()).unwrap();
        writer.write(b"abcd").unwrap();
        let snapshot = writer.finish().unwrap();
        let escaped = snapshot.file();
        owner
            .check_file(snapshot.backing().unwrap().unwrap())
            .unwrap();
        drop(owner);
        drop(snapshot);
        assert!(path.is_dir());
        let mut bytes = [0; 4];
        read_at(&escaped, &mut bytes, 0).unwrap();
        assert_eq!(&bytes, b"abcd");
        drop(escaped);
        assert!(!path.exists());
    }
    #[test]
    fn explicit_directory_refuses_failed_filesystem_predicate() {
        let owner = Arc::new(Location {
            directory: tempfile::tempdir().unwrap(),
            reject: true,
        });
        assert!(PublicFileWriter::new_computed_in(4, owner.clone()).is_err());
        assert!(PublicFileWriter::new_in(4, [0; 32], owner).is_err());
        assert!(PublicFileWriter::new_computed(4).is_ok());
    }

    #[test]
    fn backing_observations_share_identity_without_retaining_file_ownership() {
        let mut writer = PublicFileWriter::new_computed(4).unwrap();
        writer.write(b"abcd").unwrap();
        let snapshot = Arc::new(writer.finish().unwrap());
        let duplicate = Arc::clone(&snapshot);
        let handle = snapshot.file();
        handle.sync_all().unwrap();
        let owner = Arc::downgrade(&handle);
        let observation = snapshot.backing().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let metadata = handle.metadata().unwrap();
            let backing = observation.unwrap();
            assert_eq!(
                (backing.device, backing.file),
                (metadata.dev(), metadata.ino())
            );
            assert_eq!(
                backing.allocated_bytes,
                metadata.blocks().checked_mul(512).unwrap()
            );
            let mut other = PublicFileWriter::new_computed(4).unwrap();
            other.write(b"abcd").unwrap();
            let other = other.finish().unwrap();
            assert_eq!(other.digest(), snapshot.digest());
            let other = other.backing().unwrap().unwrap();
            assert_ne!((backing.device, backing.file), (other.device, other.file));
        }
        #[cfg(not(unix))]
        assert_eq!(observation, None);
        drop(handle);
        drop(snapshot);
        let after = duplicate.backing().unwrap();
        assert_eq!(
            observation.map(|b| (b.device, b.file)),
            after.map(|b| (b.device, b.file))
        );
        let mut bytes = [0; 4];
        duplicate.read(0, &mut bytes).unwrap();
        assert_eq!(&bytes, b"abcd");
        assert_eq!(duplicate.digest(), <[u8; 32]>::from(Sha256::digest(bytes)));
        drop(duplicate);
        assert!(owner.upgrade().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn sparse_extent_is_not_physical_backing() {
        use std::os::unix::fs::MetadataExt;
        // Exercise the same observer on a sparse fixture, without bypassing the
        // authenticated snapshot writer or modifying a sealed snapshot's bytes.
        let file = tempfile::tempfile().unwrap();
        let extent = 64 << 20;
        file.set_len(extent).unwrap();
        file.sync_all().unwrap();
        let metadata = file.metadata().unwrap();
        let backing = observe_backing(&file).unwrap().unwrap();
        assert_eq!(metadata.len(), extent);
        assert_eq!(
            (backing.device, backing.file),
            (metadata.dev(), metadata.ino())
        );
        assert_eq!(
            backing.allocated_bytes,
            metadata.blocks().checked_mul(512).unwrap()
        );
        assert!(backing.allocated_bytes < extent / 16);
        let mut hole = [1; 8];
        read_at(&file, &mut hole, extent - 8).unwrap();
        assert_eq!(hole, [0; 8]);
    }

    #[test]
    fn producer_commits_exact_content_and_keeps_size_and_lifetime_contracts() {
        assert!(PublicFileWriter::new_computed(0).is_err());
        assert!(PublicFileWriter::new_computed(MAX_BYTES + 1).is_err());
        let mut writer = PublicFileWriter::new_computed(4).unwrap();
        writer.write(b"ab").unwrap();
        writer.write(b"cd").unwrap();
        let snapshot = writer.finish().unwrap();
        assert_eq!(snapshot.digest(), <[u8; 32]>::from(Sha256::digest(b"abcd")));
        assert_eq!(snapshot.size(), 4);
        let file = snapshot.file();
        let owner = Arc::downgrade(&file);
        drop(snapshot);
        let mut bytes = [0; 4];
        read_at(&file, &mut bytes, 0).unwrap();
        assert_eq!(&bytes, b"abcd");
        drop(file);
        assert!(owner.upgrade().is_none());

        let mut incomplete = PublicFileWriter::new_computed(4).unwrap();
        incomplete.write(b"abc").unwrap();
        assert!(incomplete.finish().is_err());
        for data in [b"".as_slice(), b"abcde", &vec![0; MAX_CHUNK_BYTES + 1]] {
            let mut writer = PublicFileWriter::new_computed(4).unwrap();
            assert!(writer.write(data).is_err());
            assert!(writer.write(b"abcd").is_err());
            assert!(writer.finish().is_err());
        }
    }

    #[test]
    fn bounded_writers_retire_on_invalid_or_incomplete_content() {
        let hash = Sha256::digest(b"abcd").into();
        assert!(PublicFileWriter::new(0, hash).is_err());
        assert!(PublicFileWriter::new(MAX_BYTES + 1, hash).is_err());
        for data in [b"".as_slice(), b"abcde", &vec![0; MAX_CHUNK_BYTES + 1]] {
            let mut writer = PublicFileWriter::new(4, hash).unwrap();
            assert!(writer.write(data).is_err());
            assert!(writer.write(b"abcd").is_err());
            assert!(writer.finish().is_err());
        }
        for data in [b"abc".as_slice(), b"abce"] {
            let mut writer = PublicFileWriter::new(4, hash).unwrap();
            writer.write(data).unwrap();
            assert!(writer.finish().is_err());
        }
    }

    #[test]
    fn authenticated_snapshot_has_bounded_independent_readers() {
        let bytes = [0, 127, 128, 255];
        let hash = Sha256::digest(bytes).into();
        let mut writer = PublicFileWriter::new(4, hash).unwrap();
        writer.write(&bytes[..2]).unwrap();
        writer.write(&bytes[2..]).unwrap();
        let snapshot = writer.finish().unwrap();
        assert_eq!(snapshot.digest(), hash);
        assert_eq!(snapshot.size(), 4);
        assert_eq!(snapshot.max_i8(), 128);
        let mut partial = [0; 2];
        snapshot.read(1, &mut partial).unwrap();
        assert_eq!(partial, [127, 128]);
        let mut whole = [0; 4];
        snapshot.read(0, &mut whole).unwrap();
        assert_eq!(whole, bytes);
        for (offset, length) in [(0, 0), (0, MAX_CHUNK_BYTES + 1), (3, 2), (u64::MAX, 1)] {
            assert!(snapshot.read(offset, &mut vec![0; length]).is_err());
        }
        let file = snapshot.file();
        assert_eq!(Arc::strong_count(&file), 2);
        drop(snapshot);
        let mut result = [0; 4];
        read_at(&file, &mut result, 0).unwrap();
        assert_eq!(result, bytes);
    }

    #[test]
    fn typed_reads_preserve_bounds_exact_end_and_legacy_diagnostics() {
        let mut writer = PublicFileWriter::new_computed(4).unwrap();
        writer.write(b"abcd").unwrap();
        let snapshot = writer.finish().unwrap();
        for (offset, length) in [
            (u64::MAX, 0), (u64::MAX, MAX_CHUNK_BYTES + 1),
            (u64::MAX, 1), (3, 2), (4, 1),
        ] {
            let mut bytes = vec![0xa5; length];
            assert!(matches!(snapshot.read_unformatted(offset, &mut bytes), Err(PublicReadError::Bounds)));
            assert!(bytes.iter().all(|&b| b == 0xa5));
            assert_eq!(snapshot.read(offset, &mut bytes).unwrap_err(), "public snapshot read exceeds bounds");
            assert!(bytes.iter().all(|&b| b == 0xa5));
        }
        let mut typed = [0; 2];
        let mut legacy = [0; 2];
        snapshot.read_unformatted(2, &mut typed).unwrap();
        snapshot.read(2, &mut legacy).unwrap();
        assert_eq!(&typed, b"cd");
        assert_eq!(legacy, typed);
        let file = snapshot.file();
        typed.fill(0); legacy.fill(0);
        read_at_unformatted(&file, &mut typed, 2).unwrap();
        read_at(&file, &mut legacy, 2).unwrap();
        assert_eq!(&typed, b"cd");
        assert_eq!(legacy, typed);
        // Legacy unbounded kernel retains its original empty-read semantics.
        read_at(&file, &mut [], u64::MAX).unwrap();
        assert_eq!(format_read_error(PublicReadError::Truncated), "private public snapshot is truncated");
    }

    #[test]
    fn typed_and_legacy_eof_preserve_native_diagnostics_and_written_prefix() {
        let mut writer = PublicFileWriter::new_computed(4).unwrap();
        writer.write(b"abcd").unwrap();
        let snapshot = writer.finish().unwrap();
        // Privileged test-only external truncation AFTER the genuine writer
        // finished; production exposes no writable descriptor or fault callback.
        let file = snapshot.file();
        file.set_len(2).unwrap();
        #[cfg(unix)]
        let expected = {
            use std::os::unix::fs::FileExt;
            let mut bytes = [0xa5; 4];
            let error = file.read_exact_at(&mut bytes, 0).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
            assert_eq!(&bytes, b"ab\xa5\xa5");
            error.to_string() // independent native legacy diagnostic, unchanged
        };
        #[cfg(windows)]
        let expected = "private public snapshot is truncated".to_string();
        for kernel in [false, true] {
            let mut typed = [0xa5; 4];
            let error = if kernel {
                read_at_unformatted(&file, &mut typed, 0)
            } else {
                snapshot.read_unformatted(0, &mut typed)
            }.unwrap_err();
            #[cfg(unix)]
            match error {
                PublicReadError::Io(error) => {
                    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
                    assert_eq!(error.to_string(), expected);
                }
                error => panic!("native EOF changed variant: {error:?}"),
            }
            #[cfg(windows)]
            assert!(matches!(error, PublicReadError::Truncated));
            assert_eq!(&typed, b"ab\xa5\xa5");
            let mut legacy = [0xa5; 4];
            let message = if kernel {
                read_at(&file, &mut legacy, 0)
            } else {
                snapshot.read(0, &mut legacy)
            }.unwrap_err();
            assert_eq!(message, expected);
            assert_eq!(legacy, typed);
        }
    }

    #[test]
    fn typed_read_control_quote_matches_independent_target_inventory() {
        use std::mem::size_of as s;
        let b = s::<&PublicFile>() + s::<u64>() + s::<&mut [u8]>();
        let k = s::<&File>() + s::<&mut [u8]>() + s::<u64>();
        #[cfg(unix)]
        let platform = s::<Result<(), io::Error>>();
        #[cfg(windows)]
        let platform = s::<Result<usize, io::Error>>() + 5*s::<usize>()
            + s::<std::ops::RangeFrom<usize>>() + 2*s::<u64>() + 2*s::<bool>();
        let expected = 2*b + 3*k + s::<usize>() + s::<Option<u64>>()
            + 4*s::<u64>() + 4*s::<bool>() + 3*s::<Result<(), PublicReadError>>()
            + s::<PublicReadError>() + s::<io::Error>() + platform;
        assert_eq!(READ_UNFORMATTED_CONTROL_BYTES, expected);
    }

}
