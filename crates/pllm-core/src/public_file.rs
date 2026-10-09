//! Authenticated immutable public bytes, shared without copying a private file.
//!
//! Only a completed bounded writer can mint a snapshot. No path, writable file,
//! or mutable byte view escapes; readers and paged matrices share its lifetime.
use sha2::{Digest, Sha256};
use std::{fs::File, io::Write, sync::Arc};

pub const MAX_BYTES: u64 = 4 << 30;
pub const MAX_CHUNK_BYTES: usize = 1 << 20;

pub struct PublicFile {
    file: Arc<File>,
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
    pub(crate) fn file(&self) -> Arc<File> {
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

struct Pending {
    file: File,
    size: u64,
    expected: [u8; 32],
    written: u64,
    hash: Sha256,
    max_i8: u32,
}

pub struct PublicFileWriter {
    pending: Option<Pending>,
}

impl PublicFileWriter {
    pub fn new(size: u64, expected: [u8; 32]) -> Result<Self, String> {
        if size == 0 || size > MAX_BYTES {
            return Err("public snapshot size exceeds bounds".into());
        }
        Ok(Self {
            pending: Some(Pending {
                file: tempfile::tempfile().map_err(|e| e.to_string())?,
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
            || pending.written + bytes.len() as u64 > pending.size
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
        if pending.written != pending.size
            || <[u8; 32]>::from(pending.hash.finalize()) != pending.expected
        {
            return Err("public snapshot digest or size differs".into());
        }
        Ok(PublicFile {
            file: Arc::new(pending.file),
            size: pending.size,
            digest: pending.expected,
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
