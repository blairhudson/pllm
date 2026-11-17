//! Python-free, explicit descriptor close custody; not source admission.
//!
//! Only a successful Darwin close acknowledges physical closure. This owner
//! provides no source-retirement certificate, funding, backing acknowledgment,
//! directory cleanup, or PublicFile integration. Future integrations must price
//! their actual layouts and lifecycle before activation.
//!
//! There is deliberately no Drop close. Callers must retain open custody until
//! explicit close, and retain unknown custody without retrying a possibly reused
//! descriptor number. Borrowers must not close aliases and must prevent close for
//! their entire use. Other Unix targets refuse physical close with ENOSYS.

use std::ffi::CStr;
use std::os::fd::{BorrowedFd, IntoRawFd, OwnedFd, RawFd};

/// Close-only lifecycle, independent of acquisition or validation status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloseState {
    Vacant,
    Open,
    CloseActive,
    Closed,
    Unknown,
}

/// A refusal is not a close attempt; Unknown must never be retried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloseFailure {
    Eligibility,
    Unknown,
}

/// Exclusive close custody with no cloning, raw-descriptor constructor, state
/// setter, external acknowledgment, or implicit Drop implementation.
pub struct UnixCloseOnceFd {
    state: CloseState,
    fd: RawFd,
    close_errno: i32,
}

impl UnixCloseOnceFd {
    pub fn new() -> Self {
        Self { state: CloseState::Vacant, fd: -1, close_errno: 0 }
    }

    /// Only Vacant can adopt. Refusal returns the incoming owner unchanged;
    /// successful eligibility checking is followed only by infallible transfer.
    pub fn adopt(&mut self, owned: OwnedFd) -> Result<(), OwnedFd> {
        if self.state != CloseState::Vacant {
            return Err(owned);
        }
        self.fd = owned.into_raw_fd();
        self.state = CloseState::Open;
        Ok(())
    }

    pub fn state(&self) -> CloseState { self.state }

    pub fn close_errno(&self) -> i32 { self.close_errno }

    /// A borrow, never an ownership transfer or permission to close an alias.
    /// No descriptor can be borrowed after any uncertain close attempt.
    pub fn descriptor(&self) -> Option<BorrowedFd<'_>> {
        if self.state != CloseState::Open { return None; }
        // Adoption consumed a genuine OwnedFd; no close has yet been attempted.
        Some(unsafe { BorrowedFd::borrow_raw(self.fd) })
    }

    pub fn close(&mut self) -> Result<(), CloseFailure> {
        self.close_with(physical_close)
    }

    // Private per-call injection only. No public callback or process-global seam.
    fn close_with(&mut self, close: impl FnOnce(RawFd) -> Result<(), i32>) -> Result<(), CloseFailure> {
        match self.state {
            CloseState::Closed => return Ok(()),
            CloseState::Open => {}
            _ => return Err(CloseFailure::Eligibility),
        }
        self.state = CloseState::CloseActive;
        match close(self.fd) {
            Ok(()) => {
                self.fd = -1;
                self.state = CloseState::Closed;
                Ok(())
            }
            Err(errno) => {
                self.close_errno = errno;
                self.state = CloseState::Unknown;
                Err(CloseFailure::Unknown)
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn physical_close(fd: RawFd) -> Result<(), i32> {
    let result = unsafe { libc::close(fd) };
    // Capture errno immediately, with no intervening call or EINTR retry.
    if result == 0 { Ok(()) } else { Err(unsafe { *libc::__error() }) }
}

#[cfg(not(target_os = "macos"))]
fn physical_close(_: RawFd) -> Result<(), i32> { Err(libc::ENOSYS) }

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::{AsRawFd, FromRawFd};

    // Fake numbers are confined to this private test constructor AND injected
    // closers. Never call close() or borrow an Open synthetic descriptor.
    fn synthetic(state: CloseState) -> UnixCloseOnceFd {
        assert!(!std::mem::needs_drop::<UnixCloseOnceFd>());
        UnixCloseOnceFd { state, fd: 42, close_errno: 0 }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn real_owned_tempfile_close_and_idempotence() {
        let file = tempfile::tempfile().unwrap();
        let fd = file.as_raw_fd();
        let mut owner = UnixCloseOnceFd::new();
        assert!(owner.adopt(file.into()).is_ok());
        assert_eq!(owner.descriptor().unwrap().as_raw_fd(), fd);
        assert_eq!(owner.close(), Ok(()));
        assert_eq!(owner.state(), CloseState::Closed);
        assert_eq!(owner.fd, -1);
        assert_eq!(owner.close_errno(), 0);
        assert!(owner.descriptor().is_none());
        assert_eq!(owner.close(), Ok(()));
    }

    #[test]
    fn vacant_refusal_never_calls_closer() {
        let mut owner = UnixCloseOnceFd::new();
        assert_eq!(owner.state(), CloseState::Vacant);
        assert_eq!(owner.close_errno(), 0);
        assert!(owner.descriptor().is_none());
        assert_eq!(owner.close_with(|_| panic!("vacant close")), Err(CloseFailure::Eligibility));
        assert_eq!(owner.state(), CloseState::Vacant);
        let mut active = synthetic(CloseState::CloseActive);
        assert_eq!(active.close_with(|_| panic!("active close")), Err(CloseFailure::Eligibility));
        assert_eq!(active.state(), CloseState::CloseActive);
    }

    #[test]
    fn adopt_refusal_returns_incoming_owner() {
        for state in [CloseState::Open, CloseState::CloseActive, CloseState::Closed, CloseState::Unknown] {
            let mut owner = synthetic(state);
            let incoming: OwnedFd = tempfile::tempfile().unwrap().into();
            let fd = incoming.as_raw_fd();
            let returned = owner.adopt(incoming).unwrap_err();
            assert_eq!(returned.as_raw_fd(), fd);
            assert!(unsafe { libc::fcntl(returned.as_raw_fd(), libc::F_GETFD) } >= 0);
            assert_eq!(owner.state(), state);
            assert_eq!(owner.fd, 42);
            // Returned OwnedFd still owns the real resource and closes normally.
            drop(returned);
        }
    }

    #[test]
    fn injected_success_closes_once() {
        let mut owner = synthetic(CloseState::Open);
        let mut calls = 0;
        assert_eq!(owner.close_with(|fd| { assert_eq!(fd, 42); calls += 1; Ok(()) }), Ok(()));
        assert_eq!(owner.state(), CloseState::Closed);
        assert_eq!(owner.fd, -1);
        assert_eq!(owner.close_with(|_| panic!("second close")), Ok(()));
        assert_eq!(calls, 1);
    }

    #[test]
    fn injected_errors_never_retry_or_implicitly_close() {
        for errno in [libc::EINTR, libc::EBADF, libc::EIO] {
            let mut owner = synthetic(CloseState::Open);
            let mut calls = 0;
            assert_eq!(owner.close_with(|fd| { assert_eq!(fd, 42); calls += 1; Err(errno) }),
                Err(CloseFailure::Unknown));
            assert_eq!(owner.state(), CloseState::Unknown);
            assert_eq!(owner.close_errno(), errno);
            assert!(owner.descriptor().is_none());
            assert_eq!(owner.close_with(|_| panic!("unknown retry")), Err(CloseFailure::Eligibility));
            drop(owner);
            assert_eq!(calls, 1);
        }
    }

    #[test]
    fn dropping_open_owner_does_not_close() {
        let owned: OwnedFd = tempfile::tempfile().unwrap().into();
        let fd = owned.as_raw_fd();
        let mut owner = UnixCloseOnceFd::new();
        assert!(owner.adopt(owned).is_ok());
        drop(owner);
        assert!(unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0);
        // Recover only this still-open test descriptor for ordinary RAII cleanup.
        drop(unsafe { OwnedFd::from_raw_fd(fd) });
    }
}

/// Nominal data-operation failure, not admission or retirement authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoOnceError {
    Unsupported,
    OffsetRange,
    Errno(i32),
}

/// One synchronous positional read; no cursor management, retention, or retry.
///
/// macOS only. Profile refusal precedes checked offset conversion, then the
/// Rust 1.85.1 Apple count cap (c_int::MAX - 1). Empty buffers still attempt I/O
/// after validation. The caller must charge its original ledger before every
/// attempt, including any caller-directed retry. This grants no admission,
/// ownership, lifecycle, kernel/ioctl, or whole-work bound.
pub fn read_at_once<'fd>(
    fd: BorrowedFd<'fd>, dst: &mut [u8], offset: u64,
) -> Result<usize, IoOnceError> {
    read_at_once_with(
        std::os::fd::AsRawFd::as_raw_fd(&fd), dst, offset,
        cfg!(target_os = "macos"), physical_read_once,
    )
}

/// One synchronous write at the descriptor's current position; no retry.
///
/// Uses the same profile, count cap, empty-buffer, and caller-ledger contract as
/// `read_at_once`. The borrow and buffer are never retained. A short write, zero,
/// or EINTR is returned unchanged; no cursor repair or ownership transfer occurs.
pub fn write_once<'fd>(fd: BorrowedFd<'fd>, src: &[u8]) -> Result<usize, IoOnceError> {
    write_once_with(
        std::os::fd::AsRawFd::as_raw_fd(&fd), src,
        cfg!(target_os = "macos"), physical_write_once,
    )
}

fn io_once_count(len: usize) -> usize {
    len.min((libc::c_int::MAX - 1) as usize)
}

// Private per-call profile and fused result/errno injection. Synthetic numbers
// are usable only with injected attempts, never native I/O or BorrowedFd.
fn read_at_once_with(
    fd: RawFd, dst: &mut [u8], offset: u64, supported: bool,
    attempt: impl FnOnce(RawFd, &mut [u8], usize, libc::off_t) -> Result<usize, i32>,
) -> Result<usize, IoOnceError> {
    if !supported { return Err(IoOnceError::Unsupported); }
    let offset = libc::off_t::try_from(offset).map_err(|_| IoOnceError::OffsetRange)?;
    let count = io_once_count(dst.len());
    attempt(fd, dst, count, offset).map_err(IoOnceError::Errno)
}

fn write_once_with(
    fd: RawFd, src: &[u8], supported: bool,
    attempt: impl FnOnce(RawFd, &[u8], usize) -> Result<usize, i32>,
) -> Result<usize, IoOnceError> {
    if !supported { return Err(IoOnceError::Unsupported); }
    let count = io_once_count(src.len());
    attempt(fd, src, count).map_err(IoOnceError::Errno)
}

#[cfg(target_os = "macos")]
fn physical_read_once(
    fd: RawFd, dst: &mut [u8], count: usize, offset: libc::off_t,
) -> Result<usize, i32> {
    // The live synchronous borrow protects fd; count never exceeds dst.len().
    let result = unsafe { libc::pread(fd, dst.as_mut_ptr().cast(), count, offset) };
    if result == -1 {
        // Snapshot errno immediately, before any helper or retry.
        let errno = unsafe { *libc::__error() };
        Err(errno)
    } else {
        Ok(result as usize)
    }
}

#[cfg(target_os = "macos")]
fn physical_write_once(fd: RawFd, src: &[u8], count: usize) -> Result<usize, i32> {
    // The live synchronous borrow protects fd; count never exceeds src.len().
    let result = unsafe { libc::write(fd, src.as_ptr().cast(), count) };
    if result == -1 {
        // Snapshot errno immediately, before any helper or retry.
        let errno = unsafe { *libc::__error() };
        Err(errno)
    } else {
        Ok(result as usize)
    }
}

#[cfg(not(target_os = "macos"))]
fn physical_read_once(_: RawFd, _: &mut [u8], _: usize, _: libc::off_t) -> Result<usize, i32> {
    Err(libc::ENOSYS)
}

#[cfg(not(target_os = "macos"))]
fn physical_write_once(_: RawFd, _: &[u8], _: usize) -> Result<usize, i32> {
    Err(libc::ENOSYS)
}

/// Nominal path-operation failure, not namespace custody or adoption authority.
///
/// Callers must establish parent/name root identity and exclusive namespace
/// control before interpreting unlink success as ownership removal. Preowned
/// intent must retain ambiguous acquisition or unlink failures for quarantine;
/// a negative result is not proof that no namespace or resource change occurred.
/// Charge the original ledger before each attempt. One collision is returned,
/// not retried with another name. These are caller obligations, not guarantees
/// supplied by these wrappers. No source, backing-retirement, funding, admission,
/// security, cryptographic, capacity, or activation qualification follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathOnceError {
    Unsupported,
    InvalidBasename,
    Errno(i32),
}

/// Open a child directory with one Rust libc `openat` invocation, without retry.
///
/// Only macOS64 is qualified; other profiles refuse before basename validation.
/// A name must be nonempty, not `.` or `..`, and contain no slash. Its CStr bytes
/// are borrowed unchanged, without allocation or retention. Flags are exactly
/// O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC. Success returns a raw fd,
/// including fd 0, with no Drop close or intermediary OwnedFd. Negative results
/// snapshot errno immediately, including EINTR or an absent (zero) errno.
/// One Rust libc invocation is not a kernel-attempt or libc-internal CPU bound.
/// See `PathOnceError` for the caller's namespace and custody obligations.
/// Immediate raw-fd adoption is the caller's responsibility; the prepared-factory
/// handoff described on `create_child_once` also applies to directory acquisition.
pub fn open_directory_once(parent: BorrowedFd<'_>, name: &CStr) -> Result<RawFd, PathOnceError> {
    open_directory_once_with(
        std::os::fd::AsRawFd::as_raw_fd(&parent), name,
        cfg!(all(target_os = "macos", target_pointer_width = "64")), physical_open_directory_once,
    )
}

/// Exclusively create one child with O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW |
/// O_CLOEXEC and mode 0600, subject to the caller's umask. Never retry a collision.
///
/// Profile, basename, invocation, errno, and raw-result rules match
/// `open_directory_once`. The caller must immediately adopt every genuine
/// successful raw fd. With `ResidentFds::acquire_with`, invoke this inside the
/// prepared factory, then immediately convert the genuine raw fd using the
/// infallible unsafe `OwnedFd::from_raw_fd` and return Ok for existing prepared
/// adoption. No fallible work, callback, formatting, or allocation may intervene.
/// This handoff is a caller obligation; this module wires no importer and stores
/// no paths or new owners. Synthetic descriptors must never enter this handoff.
pub fn create_child_once(directory: BorrowedFd<'_>, name: &CStr) -> Result<RawFd, PathOnceError> {
    create_child_once_with(
        std::os::fd::AsRawFd::as_raw_fd(&directory), name,
        cfg!(all(target_os = "macos", target_pointer_width = "64")), physical_create_child_once,
    )
}

/// Unlink one child with flags 0 and one Rust libc `unlinkat` invocation.
///
/// Profile and basename validation match `open_directory_once`. Only return 0
/// acknowledges success; negative results snapshot errno immediately without
/// retry. This is not descriptor close, mapping retirement, or backing cleanup
/// authority. Ownership-removal interpretation requires the exclusive namespace
/// and root-identity prerequisites documented on `PathOnceError`.
pub fn unlink_child_once(directory: BorrowedFd<'_>, name: &CStr) -> Result<(), PathOnceError> {
    unlink_child_once_with(
        std::os::fd::AsRawFd::as_raw_fd(&directory), name,
        cfg!(all(target_os = "macos", target_pointer_width = "64")), physical_unlink_child_once,
    )
}

fn path_once_basename(name: &CStr, supported: bool) -> Result<(), PathOnceError> {
    if !supported { return Err(PathOnceError::Unsupported); }
    let bytes = name.to_bytes();
    if bytes.is_empty() || bytes == b"." || bytes == b".." || bytes.contains(&b'/') {
        return Err(PathOnceError::InvalidBasename);
    }
    Ok(())
}

// Same private fused result/errno seam as data I/O. Synthetic fd numbers are
// confined to injected attempts, never BorrowedFd, OwnedFd, or physical calls.
fn open_directory_once_with(
    parent: RawFd, name: &CStr, supported: bool,
    attempt: impl FnOnce(RawFd, &CStr, libc::c_int) -> Result<RawFd, i32>,
) -> Result<RawFd, PathOnceError> {
    path_once_basename(name, supported)?;
    attempt(parent, name, libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .map_err(PathOnceError::Errno)
}

fn create_child_once_with(
    directory: RawFd, name: &CStr, supported: bool,
    attempt: impl FnOnce(RawFd, &CStr, libc::c_int, libc::c_int) -> Result<RawFd, i32>,
) -> Result<RawFd, PathOnceError> {
    path_once_basename(name, supported)?;
    attempt(directory, name, libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW
        | libc::O_CLOEXEC, 0o600 as libc::c_int).map_err(PathOnceError::Errno)
}

fn unlink_child_once_with(
    directory: RawFd, name: &CStr, supported: bool,
    attempt: impl FnOnce(RawFd, &CStr, libc::c_int) -> Result<(), i32>,
) -> Result<(), PathOnceError> {
    path_once_basename(name, supported)?;
    attempt(directory, name, 0).map_err(PathOnceError::Errno)
}

#[cfg(all(target_os = "macos", target_pointer_width = "64"))]
fn physical_open_directory_once(
    parent: RawFd, name: &CStr, flags: libc::c_int,
) -> Result<RawFd, i32> {
    let result = unsafe { libc::openat(parent, name.as_ptr(), flags) };
    if result < 0 {
        let errno = unsafe { *libc::__error() };
        Err(errno)
    } else {
        Ok(result)
    }
}

#[cfg(all(target_os = "macos", target_pointer_width = "64"))]
fn physical_create_child_once(
    directory: RawFd, name: &CStr, flags: libc::c_int, mode: libc::c_int,
) -> Result<RawFd, i32> {
    // Darwin mode_t is u16, promoted to c_int for the variadic openat argument.
    let result = unsafe { libc::openat(directory, name.as_ptr(), flags, mode) };
    if result < 0 {
        let errno = unsafe { *libc::__error() };
        Err(errno)
    } else {
        Ok(result)
    }
}

#[cfg(all(target_os = "macos", target_pointer_width = "64"))]
fn physical_unlink_child_once(
    directory: RawFd, name: &CStr, flags: libc::c_int,
) -> Result<(), i32> {
    let result = unsafe { libc::unlinkat(directory, name.as_ptr(), flags) };
    if result == 0 {
        Ok(())
    } else {
        let errno = unsafe { *libc::__error() };
        Err(errno)
    }
}

#[cfg(not(all(target_os = "macos", target_pointer_width = "64")))]
fn physical_open_directory_once(_: RawFd, _: &CStr, _: libc::c_int) -> Result<RawFd, i32> {
    Err(libc::ENOSYS)
}

#[cfg(not(all(target_os = "macos", target_pointer_width = "64")))]
fn physical_create_child_once(
    _: RawFd, _: &CStr, _: libc::c_int, _: libc::c_int,
) -> Result<RawFd, i32> {
    Err(libc::ENOSYS)
}

#[cfg(not(all(target_os = "macos", target_pointer_width = "64")))]
fn physical_unlink_child_once(_: RawFd, _: &CStr, _: libc::c_int) -> Result<(), i32> {
    Err(libc::ENOSYS)
}

/// A point-in-time descriptor observation, not stable identity or resource authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileObservation {
    pub device: u64,
    pub inode: u64,
    pub mode: u32,
    pub length: u64,
    /// Reported st_blocks multiplied by 512, not backing-release acknowledgement.
    pub allocated_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObserveOnceError {
    Unsupported,
    Errno(i32),
    InvalidSize,
    InvalidBlocks,
    AllocationOverflow,
}

/// Observe a borrowed descriptor with one Rust-level libc `fstat` operation.
///
/// Only macOS64 is qualified; other profiles refuse before invoking the adapter.
/// A nonzero return snapshots errno immediately, without reading output fields
/// or retrying. Successful fields undergo checked size conversion first, then
/// checked block conversion and multiplication by 512. Device conversion matches
/// Rust MetadataExt's sign-preserving cast to u64; inode and mode bits are retained.
/// No allocation, retention, ownership transfer, duplication, or close occurs.
/// This snapshot grants no stable identity, resource bound, funding, admission,
/// retirement acknowledgement, or source authority. One Rust-level operation is
/// not a kernel-attempt or libc-internal CPU bound.
pub fn observe_once(fd: BorrowedFd<'_>) -> Result<FileObservation, ObserveOnceError> {
    observe_once_with(
        std::os::fd::AsRawFd::as_raw_fd(&fd),
        cfg!(all(target_os = "macos", target_pointer_width = "64")), physical_observe_once,
    )
}

// Scalar-only injection: tests never manufacture libc::stat or descriptor owners.
#[derive(Clone, Copy)]
struct RawFileObservation {
    device: i32,
    inode: u64,
    mode: u16,
    length: i64,
    blocks: i64,
}

fn observe_once_with(
    fd: RawFd, supported: bool,
    attempt: impl FnOnce(RawFd) -> Result<RawFileObservation, i32>,
) -> Result<FileObservation, ObserveOnceError> {
    if !supported { return Err(ObserveOnceError::Unsupported); }
    let raw = attempt(fd).map_err(ObserveOnceError::Errno)?;
    let length = u64::try_from(raw.length).map_err(|_| ObserveOnceError::InvalidSize)?;
    let blocks = u64::try_from(raw.blocks).map_err(|_| ObserveOnceError::InvalidBlocks)?;
    let allocated_bytes = blocks.checked_mul(512).ok_or(ObserveOnceError::AllocationOverflow)?;
    Ok(FileObservation {
        device: raw.device as u64, inode: raw.inode, mode: u32::from(raw.mode),
        length, allocated_bytes,
    })
}

#[cfg(all(target_os = "macos", target_pointer_width = "64"))]
fn physical_observe_once(fd: RawFd) -> Result<RawFileObservation, i32> {
    let mut output = std::mem::MaybeUninit::<libc::stat>::uninit();
    let result = unsafe { libc::fstat(fd, output.as_mut_ptr()) };
    if result != 0 {
        let errno = unsafe { *libc::__error() };
        return Err(errno);
    }
    let output = unsafe { output.assume_init() };
    Ok(RawFileObservation {
        device: output.st_dev, inode: output.st_ino, mode: output.st_mode,
        length: output.st_size, blocks: output.st_blocks,
    })
}

#[cfg(not(all(target_os = "macos", target_pointer_width = "64")))]
fn physical_observe_once(_: RawFd) -> Result<RawFileObservation, i32> {
    Err(libc::ENOSYS)
}

#[cfg(test)]
#[path = "fd_close_once_io_tests.rs"]
mod io_tests;
