//! Immutable public bytes, shared without copying a private file.
//!
//! Only a completed bounded writer can mint a snapshot. No path, writable file,
//! or mutable byte view escapes; readers and paged matrices share its lifetime.
//! The default writer authenticates an expected digest; native producers may
//! instead commit the digest computed over their output. That does not authenticate
//! the producer's source.
use sha2::{Digest, Sha256};
use std::{fs::File, io::{self, Write}, path::Path, sync::Arc};

/// Retained location only, not coefficient/source/admission authority.
pub trait SnapshotDirectory: Send + Sync {
    fn path(&self) -> &Path;
    /// Receives genuine local fstat observation, NEVER a writable descriptor.
    fn check_file(&self, file: FileBacking) -> Result<(), String>;
}

/// Every escaped reader retains this whole backing. File must close BEFORE the
/// directory (and its covering lease), including when only a paged reader lives.
pub(crate) struct PublicFileBacking {
    file: File,
    _directory: Option<Arc<dyn SnapshotDirectory>>,
}
impl PublicFileBacking {
    pub(crate) fn anonymous(file: File) -> Self {
        Self {
            file,
            _directory: None,
        }
    }
}
impl std::ops::Deref for PublicFileBacking {
    type Target = File;
    fn deref(&self) -> &File {
        &self.file
    }
}
/// Actual shared backing layout, including the retained optional location handle.
pub fn backing_bytes() -> Result<usize, String> {
    let (layout, _) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(std::alloc::Layout::new::<PublicFileBacking>())
        .map_err(|e| e.to_string())?;
    Ok(layout.pad_to_align().size())
}

/// Explicit-directory setup/return/path-copy overlap, not coefficient backing.
pub fn writer_creation_bytes() -> Result<usize, String> {
    let mut bytes = 8usize
        .checked_mul(MAX_DIRECTORY_PATH_BYTES)
        .ok_or("file setup quote overflow")?;
    for n in [
        std::mem::size_of::<Pending>(),
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

pub struct PublicFile {
    file: Arc<PublicFileBacking>,
    size: u64,
    digest: [u8; 32],
    max_i8: u32,
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

    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    /// Observe the existing handle with local fstat, without exposing the handle
    /// or changing file contents. Unix allocation is checked st_blocks * 512,
    /// never length/set_len. Unsupported platforms return None (no credit).
    pub fn backing(&self) -> Result<Option<FileBacking>, String> {
        observe_backing(&self.file)
    }
    pub(crate) fn file(&self) -> Arc<PublicFileBacking> {
        Arc::clone(&self.file)
    }
    pub(crate) fn max_i8(&self) -> u32 {
        self.max_i8
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

fn observe_backing(file: &File) -> Result<Option<FileBacking>, String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        let allocated_bytes = metadata
            .blocks()
            .checked_mul(512)
            .ok_or("public snapshot allocated byte count overflow")?;
        Ok(Some(FileBacking {
            device: metadata.dev(),
            file: metadata.ino(),
            allocated_bytes,
        }))
    }
    #[cfg(not(unix))]
    {
        let _ = file;
        Ok(None)
    }
}

struct Pending {
    file: File,
    directory: Option<Arc<dyn SnapshotDirectory>>,
    size: u64,
    expected: Option<[u8; 32]>,
    written: u64,
    hash: Sha256,
    max_i8: u32,
}

pub struct PublicFileWriter {
    pending: Option<Pending>,
}

impl PublicFileWriter {
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
    fn with_directory(
        size: u64,
        expected: Option<[u8; 32]>,
        directory: Option<Arc<dyn SnapshotDirectory>>,
    ) -> Result<Self, String> {
        if size == 0 || size > MAX_BYTES {
            return Err("public snapshot size exceeds bounds".into());
        }
        let file = match &directory {
            Some(owner) => {
                if !owner.path().is_absolute()
                    || owner.path().as_os_str().len() > MAX_DIRECTORY_PATH_BYTES - 32
                {
                    return Err("snapshot directory exceeds explicit path policy".into());
                }
                let file = tempfile::tempfile_in(owner.path()).map_err(|e| e.to_string())?;
                owner.check_file(
                    observe_backing(&file)?.ok_or("snapshot filesystem observation unsupported")?,
                )?;
                file
            }
            None => tempfile::tempfile().map_err(|e| e.to_string())?,
        };
        Ok(Self {
            pending: Some(Pending {
                file,
                directory,
                size,
                expected,
                written: 0,
                hash: Sha256::new(),
                max_i8: 0,
            }),
        })
    }
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
        pending.hash.update(bytes);
        pending.written += bytes.len() as u64;
        pending.max_i8 = pending.max_i8.max(
            bytes
                .iter()
                .map(|&v| (v as i8 as i32).unsigned_abs())
                .max()
                .unwrap_or(0),
        );
        self.pending = Some(pending);
        Ok(())
    }
    pub fn finish(mut self) -> Result<PublicFile, String> {
        let pending = self
            .pending
            .take()
            .ok_or("public snapshot writer is closed")?;
        let digest: [u8; 32] = pending.hash.finalize().into();
        if pending.written != pending.size || pending.expected.is_some_and(|v| v != digest) {
            return Err("public snapshot digest or size differs".into());
        }
        Ok(PublicFile {
            file: Arc::new(PublicFileBacking {
                file: pending.file,
                _directory: pending.directory,
            }),
            size: pending.size,
            digest,
            max_i8: pending.max_i8,
        })
    }
}

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
