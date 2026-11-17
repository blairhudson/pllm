use super::*;

#[cfg(target_os = "macos")]
#[test]
fn real_positional_read_cursor_preserved() {
    use std::io::{Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;

    let mut file = tempfile::tempfile().unwrap();
    file.write_all(b"abcdef").unwrap();
    file.seek(SeekFrom::Start(4)).unwrap();
    let mut owner = UnixCloseOnceFd::new();
    assert!(owner.adopt(file.into()).is_ok());
    let mut dst = [0; 3];
    let result = read_at_once(owner.descriptor().unwrap(), &mut dst, 1);
    let position = unsafe { libc::lseek(owner.descriptor().unwrap().as_raw_fd(), 0, libc::SEEK_CUR) };
    let retired = owner.close();
    assert_eq!(retired, Ok(()));
    assert_eq!(owner.state(), CloseState::Closed);
    assert_eq!(result, Ok(3));
    assert_eq!(&dst, b"bcd");
    assert_eq!(position, 4);
}

#[cfg(target_os = "macos")]
#[test]
fn real_write_cursor_advance() {
    use std::io::{Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;

    let mut file = tempfile::tempfile().unwrap();
    file.write_all(b"abcdef").unwrap();
    file.seek(SeekFrom::Start(2)).unwrap();
    let mut owner = UnixCloseOnceFd::new();
    assert!(owner.adopt(file.into()).is_ok());
    let result = write_once(owner.descriptor().unwrap(), b"XYZ");
    let position = unsafe { libc::lseek(owner.descriptor().unwrap().as_raw_fd(), 0, libc::SEEK_CUR) };
    let mut dst = [0; 6];
    let read_result = read_at_once(owner.descriptor().unwrap(), &mut dst, 0);
    let retired = owner.close();
    assert_eq!(retired, Ok(()));
    assert_eq!(owner.state(), CloseState::Closed);
    assert_eq!(result, Ok(3));
    assert_eq!(position, 5);
    assert_eq!(read_result, Ok(6));
    assert_eq!(&dst, b"abXYZf");
}

// Synthetic fd 42 is passed only to private injected closures. No native call,
// borrowed descriptor, or owning descriptor is constructed from that number.
#[test]
fn injected_eintr_one_call() {
    let mut reads = 0;
    let mut writes = 0;
    assert_eq!(read_at_once_with(42, &mut [0; 4], 7, true, |fd, _, count, offset| {
        reads += 1;
        assert_eq!((fd, count, offset), (42, 4, 7));
        Err(libc::EINTR)
    }), Err(IoOnceError::Errno(libc::EINTR)));
    assert_eq!(write_once_with(42, b"data", true, |fd, _, count| {
        writes += 1;
        assert_eq!((fd, count), (42, 4));
        Err(libc::EINTR)
    }), Err(IoOnceError::Errno(libc::EINTR)));
    assert_eq!((reads, writes), (1, 1));
}

#[test]
fn injected_short_zero_error_returns() {
    for outcome in [Ok(2), Ok(0), Err(libc::EIO), Err(libc::EBADF)] {
        let mut reads = 0;
        let mut writes = 0;
        let mut dst = [0; 4];
        let read = read_at_once_with(42, &mut dst, 0, true, |_, dst, count, _| {
            reads += 1;
            assert_eq!(count, 4);
            if let Ok(n) = outcome { dst[..n].fill(9); }
            outcome
        });
        let write = write_once_with(42, b"data", true, |_, src, count| {
            writes += 1;
            assert_eq!(src, b"data");
            assert_eq!(count, 4);
            outcome
        });
        assert_eq!(read, outcome.map_err(IoOnceError::Errno));
        assert_eq!(write, outcome.map_err(IoOnceError::Errno));
        assert_eq!((reads, writes), (1, 1));
        if let Ok(n) = outcome {
            assert!(dst[..n].iter().all(|&byte| byte == 9));
            assert!(dst[n..].iter().all(|&byte| byte == 0));
        }
    }
}

#[test]
fn checked_offset_count_clamp_empty_refusal_precedence() {
    let cap = (libc::c_int::MAX - 1) as usize;
    for len in [0, 1, cap - 1, cap, cap + 1, usize::MAX] {
        assert_eq!(io_once_count(len), len.min(cap));
    }
    let max_offset = libc::off_t::MAX as u64;
    let mut reads = 0;
    let mut writes = 0;
    assert_eq!(read_at_once_with(42, &mut [], max_offset, true, |_, dst, count, offset| {
        reads += 1;
        assert!(dst.is_empty());
        assert_eq!(count, 0);
        assert_eq!(offset, libc::off_t::MAX);
        Ok(0)
    }), Ok(0));
    assert_eq!(write_once_with(42, &[], true, |_, src, count| {
        writes += 1;
        assert!(src.is_empty());
        assert_eq!(count, 0);
        Ok(0)
    }), Ok(0));
    assert_eq!((reads, writes), (1, 1));
    for offset in [max_offset + 1, u64::MAX] {
        assert_eq!(read_at_once_with(42, &mut [], offset, true,
            |_, _, _, _| panic!("invalid offset attempted I/O")), Err(IoOnceError::OffsetRange));
        assert_eq!(read_at_once_with(42, &mut [], offset, false,
            |_, _, _, _| panic!("unsupported profile attempted I/O")), Err(IoOnceError::Unsupported));
    }
}

#[test]
fn unsupported_profile_zero_calls() {
    let mut calls = 0;
    for offset in [0, u64::MAX] {
        assert_eq!(read_at_once_with(42, &mut [0; 1], offset, false, |_, _, _, _| {
            calls += 1;
            Ok(1)
        }), Err(IoOnceError::Unsupported));
    }
    for src in [&b""[..], &b"data"[..]] {
        assert_eq!(write_once_with(42, src, false, |_, _, _| {
            calls += 1;
            Ok(0)
        }), Err(IoOnceError::Unsupported));
    }
    assert_eq!(calls, 0);
}

// Path tests use only private injected attempts. Neither input 42 nor returned
// synthetic fd numbers are borrowed, adopted, closed, or used for physical I/O.
#[test]
fn path_profile_refusal_precedes_name_validation() {
    let mut calls = 0;
    for bytes in [&b"\0"[..], &b".\0"[..], &b"..\0"[..], &b"a/b\0"[..], &b"child\0"[..]] {
        let name = CStr::from_bytes_with_nul(bytes).unwrap();
        assert_eq!(open_directory_once_with(42, name, false, |_, _, _| {
            calls += 1;
            Ok(0)
        }), Err(PathOnceError::Unsupported));
        assert_eq!(create_child_once_with(42, name, false, |_, _, _, _| {
            calls += 1;
            Ok(0)
        }), Err(PathOnceError::Unsupported));
        assert_eq!(unlink_child_once_with(42, name, false, |_, _, _| {
            calls += 1;
            Ok(())
        }), Err(PathOnceError::Unsupported));
    }
    assert_eq!(calls, 0);
}

#[test]
fn path_invalid_basenames_zero_calls() {
    let mut calls = 0;
    for bytes in [&b"\0"[..], &b".\0"[..], &b"..\0"[..], &b"/\0"[..],
                  &b"a/b\0"[..], &b"a/\0"[..], &b"/a\0"[..]] {
        let name = CStr::from_bytes_with_nul(bytes).unwrap();
        assert_eq!(open_directory_once_with(42, name, true, |_, _, _| {
            calls += 1;
            Ok(0)
        }), Err(PathOnceError::InvalidBasename));
        assert_eq!(create_child_once_with(42, name, true, |_, _, _, _| {
            calls += 1;
            Ok(0)
        }), Err(PathOnceError::InvalidBasename));
        assert_eq!(unlink_child_once_with(42, name, true, |_, _, _| {
            calls += 1;
            Ok(())
        }), Err(PathOnceError::InvalidBasename));
    }
    assert_eq!(calls, 0);
}

#[test]
fn open_directory_flags_and_success() {
    let name = CStr::from_bytes_with_nul(b"\xff.directory\0").unwrap();
    for returned in [0, 17] {
        let mut calls = 0;
        assert_eq!(open_directory_once_with(42, name, true, |parent, passed, flags| {
            calls += 1;
            assert_eq!(parent, 42);
            assert_eq!(passed.as_ptr(), name.as_ptr());
            assert_eq!(passed.to_bytes(), b"\xff.directory");
            assert_eq!(flags, libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC);
            Ok(returned)
        }), Ok(returned));
        assert_eq!(calls, 1);
    }
}

#[test]
fn open_directory_errors_once() {
    let name = CStr::from_bytes_with_nul(b"directory\0").unwrap();
    for code in [libc::EINTR, libc::EEXIST, libc::EIO, 0] {
        let mut calls = 0;
        let mut errno = code;
        let result = open_directory_once_with(42, name, true, |parent, passed, flags| {
            calls += 1;
            assert_eq!(parent, 42);
            assert_eq!(passed.as_ptr(), name.as_ptr());
            assert_eq!(flags, libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC);
            // The injected attempt returns a fused snapshot, as physical code
            // does immediately after a negative libc return, before any retry.
            let captured = errno;
            errno = libc::EBADF;
            Err(captured)
        });
        assert_eq!(result, Err(PathOnceError::Errno(code)));
        assert_eq!(errno, libc::EBADF);
        assert_eq!(calls, 1);
    }
}

#[test]
fn create_child_flags_mode_and_success() {
    let name = CStr::from_bytes_with_nul(b"\xff.child\0").unwrap();
    for returned in [0, 17] {
        let mut calls = 0;
        assert_eq!(create_child_once_with(42, name, true, |directory, passed, flags, mode| {
            calls += 1;
            assert_eq!(directory, 42);
            assert_eq!(passed.as_ptr(), name.as_ptr());
            assert_eq!(passed.to_bytes(), b"\xff.child");
            assert_eq!(flags, libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW
                | libc::O_CLOEXEC);
            let promoted: libc::c_int = mode;
            assert_eq!(promoted, 0o600 as libc::c_int);
            Ok(returned)
        }), Ok(returned));
        assert_eq!(calls, 1);
    }
    #[cfg(all(target_os = "macos", target_pointer_width = "64"))]
    assert_eq!(std::mem::size_of::<libc::mode_t>(), 2);
}

#[test]
fn create_child_errors_once() {
    let name = CStr::from_bytes_with_nul(b"child\0").unwrap();
    for code in [libc::EINTR, libc::EEXIST, libc::EIO, 0] {
        let mut calls = 0;
        let mut errno = code;
        let result = create_child_once_with(42, name, true, |directory, passed, flags, mode| {
            calls += 1;
            assert_eq!(directory, 42);
            assert_eq!(passed.as_ptr(), name.as_ptr());
            assert_eq!(flags, libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW
                | libc::O_CLOEXEC);
            assert_eq!(mode, 0o600 as libc::c_int);
            let captured = errno;
            errno = libc::EBADF;
            Err(captured)
        });
        assert_eq!(result, Err(PathOnceError::Errno(code)));
        assert_eq!(errno, libc::EBADF);
        assert_eq!(calls, 1);
    }
}

#[test]
fn unlink_child_flags_and_ack() {
    for bytes in [&b"child\0"[..], &b"\xff.child\0"[..]] {
        let name = CStr::from_bytes_with_nul(bytes).unwrap();
        let mut calls = 0;
        assert_eq!(unlink_child_once_with(42, name, true, |directory, passed, flags| {
            calls += 1;
            assert_eq!(directory, 42);
            assert_eq!(passed.as_ptr(), name.as_ptr());
            assert_eq!(passed.to_bytes_with_nul(), bytes);
            assert_eq!(flags, 0);
            Ok(())
        }), Ok(()));
        assert_eq!(calls, 1);
    }
}

#[test]
fn unlink_child_errors_once() {
    let name = CStr::from_bytes_with_nul(b"child\0").unwrap();
    for code in [libc::EINTR, libc::EEXIST, libc::EIO, 0] {
        let mut calls = 0;
        let mut errno = code;
        let result = unlink_child_once_with(42, name, true, |directory, passed, flags| {
            calls += 1;
            assert_eq!(directory, 42);
            assert_eq!(passed.as_ptr(), name.as_ptr());
            assert_eq!(flags, 0);
            let captured = errno;
            errno = libc::EBADF;
            Err(captured)
        });
        assert_eq!(result, Err(PathOnceError::Errno(code)));
        assert_eq!(errno, libc::EBADF);
        assert_eq!(calls, 1);
    }
}

#[cfg(all(target_os = "macos", target_pointer_width = "64"))]
#[test]
fn real_path_apis_resident_lifecycle() {
    use crate::resident_fd::{ResidentFds, RegistryError};
    use std::fs::File;
    use std::os::fd::{AsFd, FromRawFd};
    use std::os::unix::fs::MetadataExt;

    let fixture = tempfile::tempdir().unwrap();
    std::fs::create_dir(fixture.path().join("directory")).unwrap();
    let parent = File::open(fixture.path()).unwrap();
    let registry = ResidentFds::prepare(2).unwrap();
    let directory_name = CStr::from_bytes_with_nul(b"directory\0").unwrap();
    let child_name = CStr::from_bytes_with_nul(b"child\0").unwrap();
    let mut directory = Some(registry.acquire_with(0, || {
        let raw = open_directory_once(parent.as_fd(), directory_name)?;
        // Only a genuine successful fd enters this immediate prepared handoff.
        Ok::<OwnedFd, PathOnceError>(unsafe { OwnedFd::from_raw_fd(raw) })
    }).unwrap());
    let mut child = Some(registry.acquire_with(1, || {
        let raw = create_child_once(
            directory.as_ref().unwrap().descriptor().unwrap(), child_name)?;
        Ok::<OwnedFd, PathOnceError>(unsafe { OwnedFd::from_raw_fd(raw) })
    }).unwrap());

    let payload = b"resident path lifecycle";
    {
        let fd = child.as_ref().unwrap().descriptor().unwrap();
        assert_eq!(write_once(fd, payload), Ok(payload.len()));
        let mut bytes = [0; 23];
        assert_eq!(read_at_once(fd, &mut bytes, 0), Ok(payload.len()));
        assert_eq!(&bytes, payload);
    }
    {
        let metadata = std::fs::metadata(fixture.path().join("directory").join("child")).unwrap();
        let fd = child.as_ref().unwrap().descriptor().unwrap();
        assert_eq!(observe_once(fd), Ok(FileObservation {
            device: metadata.dev(), inode: metadata.ino(), mode: metadata.mode(),
            length: metadata.len(), allocated_bytes: metadata.blocks().checked_mul(512).unwrap(),
        }));
    }
    {
        let fd = directory.as_ref().unwrap().descriptor().unwrap();
        assert_eq!(unlink_child_once(fd, child_name), Ok(()));
    }
    {
        let fd = child.as_ref().unwrap().descriptor().unwrap();
        let mut bytes = [0; 23];
        assert_eq!(read_at_once(fd, &mut bytes, 0), Ok(payload.len()));
        assert_eq!(&bytes, payload);
    }

    let escaped = child.as_ref().unwrap().clone();
    assert_eq!(registry.try_retire(&mut child), Err(RegistryError::Busy));
    assert!(child.is_some());
    for lease in [child.as_ref().unwrap(), &escaped] {
        let fd = lease.descriptor().unwrap();
        let mut bytes = [0; 23];
        assert_eq!(read_at_once(fd, &mut bytes, 0), Ok(payload.len()));
        assert_eq!(&bytes, payload);
    }
    drop(escaped);
    assert_eq!(registry.try_retire(&mut child), Ok(()));
    assert!(child.is_none());
    assert_eq!(registry.try_retire(&mut directory), Ok(()));
    assert!(directory.is_none());
    // Fixture cleanup is ordinary test cleanup, not source-backing release proof.
}

#[test]
fn observe_unsupported_zero_calls() {
    let mut calls = 0;
    assert_eq!(observe_once_with(42, false, |_| {
        calls += 1;
        Err(libc::EIO)
    }), Err(ObserveOnceError::Unsupported));
    assert_eq!(calls, 0);
}

#[test]
fn observe_fields_and_negative_device() {
    for (device, expected_device) in [(17, 17), (-1, u64::MAX), (i32::MIN, 0xffff_ffff_8000_0000)] {
        let mut calls = 0;
        assert_eq!(observe_once_with(42, true, |fd| {
            calls += 1;
            assert_eq!(fd, 42);
            Ok(RawFileObservation {
                device, inode: u64::MAX, mode: u16::MAX, length: i64::MAX, blocks: 4,
            })
        }), Ok(FileObservation {
            device: expected_device, inode: u64::MAX, mode: u32::from(u16::MAX),
            length: i64::MAX as u64, allocated_bytes: 2048,
        }));
        assert_eq!(calls, 1);
    }
}

#[test]
fn observe_errno_once() {
    for code in [libc::EINTR, libc::EBADF, 0] {
        let mut calls = 0;
        let mut errno = code;
        let result = observe_once_with(42, true, |fd| {
            calls += 1;
            assert_eq!(fd, 42);
            // Model the adapter's fused errno snapshot without a synthetic stat.
            let captured = errno;
            errno = libc::EIO;
            Err(captured)
        });
        assert_eq!(result, Err(ObserveOnceError::Errno(code)));
        assert_eq!(errno, libc::EIO);
        assert_eq!(calls, 1);
    }
}

#[test]
fn observe_invalid_size_blocks_precedence() {
    for (length, blocks, expected) in [
        (-1, -1, ObserveOnceError::InvalidSize),
        (i64::MIN, i64::MAX, ObserveOnceError::InvalidSize),
        (-1, 0, ObserveOnceError::InvalidSize),
        (0, -1, ObserveOnceError::InvalidBlocks),
        (i64::MAX, i64::MIN, ObserveOnceError::InvalidBlocks),
    ] {
        let mut calls = 0;
        assert_eq!(observe_once_with(42, true, |fd| {
            calls += 1;
            assert_eq!(fd, 42);
            Ok(RawFileObservation { device: 1, inode: 2, mode: 3, length, blocks })
        }), Err(expected));
        assert_eq!(calls, 1);
    }
}

#[test]
fn observe_allocation_overflow_boundary() {
    let last = i64::try_from(u64::MAX / 512).unwrap();
    for blocks in [0, last, last + 1, i64::MAX] {
        let mut calls = 0;
        let result = observe_once_with(42, true, |fd| {
            calls += 1;
            assert_eq!(fd, 42);
            Ok(RawFileObservation { device: 1, inode: 2, mode: 3, length: 0, blocks })
        });
        let expected = if blocks <= last {
            Ok((blocks as u64) * 512)
        } else {
            Err(ObserveOnceError::AllocationOverflow)
        };
        assert_eq!(result.map(|observation| observation.allocated_bytes), expected);
        assert_eq!(calls, 1);
    }
}
