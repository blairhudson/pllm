//! Immutable public bytes, shared without copying a private file.
//!
//! Only a completed bounded writer can mint a snapshot. No path, writable file,
//! or mutable byte view escapes; readers and paged matrices share its lifetime.
//! The default writer authenticates an expected digest; native producers may
//! instead commit the digest computed over their output. That does not authenticate
//! the producer's source.
use sha2::{Digest, Sha256};
use std::{fs::File, io::Write, path::Path, sync::Arc};

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

impl PublicFile {
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
        let length = bytes.len();
        if length == 0
            || length > MAX_CHUNK_BYTES
            || offset
                .checked_add(length as u64)
                .is_none_or(|end| end > self.size)
        {
            return Err("public snapshot read exceeds bounds".into());
        }
        read_at(&self.file, bytes, offset)
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
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(data, offset).map_err(|e| e.to_string())
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < data.len() {
            let n = file
                .seek_read(&mut data[done..], offset + done as u64)
                .map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("private public snapshot is truncated".into());
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
}
