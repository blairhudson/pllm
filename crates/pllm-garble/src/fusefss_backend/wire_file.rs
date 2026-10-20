//! Private exact replay storage. Only public nonce/recipient records reach disk.
//! No file handle/path or restore/reset interface is exposed by the wire API.
use super::{indexed_probes, indexed_slots, Error, ImportGuard};
use std::{fs::File, io, mem::size_of};
use zeroize::{Zeroize, Zeroizing};

pub const MAX_INDEXED_FILE_DISK_BYTES: u64 = 4 << 30;
const RECORD_BYTES: u64 = 33;

/// One guard's fixed native resident owner (including BOTH read/write buffers)
/// and separate full sparse-file extent. Two receiving guards require twice
/// these amounts. OS file cache/RSS and physical disk allocation are not Budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedFileResources {
    pub resident_bytes: usize,
    pub disk_bytes: u64,
}

pub(super) fn resources(capacity: usize) -> Result<IndexedFileResources, Error> {
    let disk_bytes = u64::try_from(indexed_slots(capacity)?)
        .ok()
        .and_then(|slots| slots.checked_mul(RECORD_BYTES))
        .filter(|bytes| *bytes <= MAX_INDEXED_FILE_DISK_BYTES)
        .ok_or(Error::Resource)?;
    Ok(IndexedFileResources {
        resident_bytes: size_of::<ImportGuard<'_>>(),
        disk_bytes,
    })
}

pub(super) struct Ledger {
    // These fields all retire before ImportGuard's final budget lease. Taking
    // file on failure permanently closes storage while the owner stays charged.
    pub(super) file: Option<File>,
    read: Zeroizing<[u8; 33]>,
    write: Zeroizing<[u8; 33]>,
    disk_bytes: u64,
    #[cfg(test)]
    pub(super) fault: Option<Fault>,
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) enum Fault {
    Read,
    Write,
    PartialWrite,
}

impl Ledger {
    #[cfg(test)]
    pub(super) fn scratch_is_zero(&self) -> bool {
        *self.read == [0; 33] && *self.write == [0; 33]
    }

    pub(super) fn new(disk_bytes: u64) -> Result<Self, Error> {
        // Same secure unlinked-tempfile primitive as core public-file owners.
        // No predictable pathname, named persistent file, or exported descriptor.
        let file = tempfile::tempfile().map_err(|_| Error::Resource)?;
        file.set_len(disk_bytes).map_err(|_| Error::Resource)?;
        Ok(Self {
            file: Some(file),
            read: Zeroizing::new([0; 33]),
            write: Zeroizing::new([0; 33]),
            disk_bytes,
            #[cfg(test)]
            fault: None,
        })
    }

    pub(super) fn claim(
        &mut self,
        role: u8,
        nonce: [u8; 32],
        slots: usize,
        exhausted: bool,
    ) -> Result<(), Error> {
        if self.file.is_none() {
            return Err(Error::Consumed);
        }
        if role > 1 {
            return Err(Error::Domain);
        }
        self.write[..32].copy_from_slice(&nonce);
        self.write[32] = role + 1;
        let result = self.try_claim(slots, exhausted);
        self.read.zeroize();
        self.write.zeroize();
        match result {
            Ok(decision) => decision,
            Err(_) => {
                // A write may have completed partly. Never try to roll it back,
                // reopen the file, or accept another claim after ANY I/O error.
                self.file.take();
                Err(Error::Resource)
            }
        }
    }

    fn try_claim(&mut self, slots: usize, exhausted: bool) -> io::Result<Result<(), Error>> {
        let file = self.file.as_ref().unwrap();
        // Detect truncation even when the next lookup would touch an earlier,
        // still-readable slot; otherwise lost history could silently reappear.
        if file.metadata()?.len() != self.disk_bytes {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        let (mut slot, stride) = indexed_probes(&self.write, slots);
        for _ in 0..slots {
            let offset = (slot as u64)
                .checked_mul(RECORD_BYTES)
                .filter(|offset| *offset <= self.disk_bytes - RECORD_BYTES)
                .ok_or(io::ErrorKind::InvalidData)?;
            #[cfg(test)]
            if matches!(self.fault, Some(Fault::Read)) {
                return Err(io::ErrorKind::Other.into());
            }
            read_at(file, &mut *self.read, offset)?;
            if self.read[32] == 0 {
                if self.read[..32] != [0; 32] {
                    return Err(io::ErrorKind::InvalidData.into());
                }
                if exhausted {
                    return Ok(Err(Error::Resource));
                }
                #[cfg(test)]
                match self.fault {
                    Some(Fault::Write) => return Err(io::ErrorKind::Other.into()),
                    Some(Fault::PartialWrite) => {
                        write_at(file, &self.write[..17], offset)?;
                        return Err(io::ErrorKind::WriteZero.into());
                    }
                    _ => {}
                }
                write_at(file, &*self.write, offset)?;
                return Ok(Ok(()));
            }
            if self.read[32] > 2 {
                return Err(io::ErrorKind::InvalidData.into());
            }
            if *self.read == *self.write {
                return Ok(Err(Error::Consumed));
            }
            slot = (slot + stride) & (slots - 1);
        }
        // Impossible for valid half-full geometry. Treat damaged state as fatal.
        Err(io::ErrorKind::InvalidData.into())
    }
}

fn read_at(file: &File, bytes: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(bytes, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < bytes.len() {
            match file.seek_read(&mut bytes[done..], offset + done as u64) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => done += n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, bytes, offset);
        Err(io::ErrorKind::Unsupported.into())
    }
}

fn write_at(file: &File, bytes: &[u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(bytes, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < bytes.len() {
            match file.seek_write(&bytes[done..], offset + done as u64) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => done += n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, bytes, offset);
        Err(io::ErrorKind::Unsupported.into())
    }
}
