//! Real leases and descriptors; only per-call result/count injection below.
//! Native once-I/O and affirmative close are qualified on macOS only.
#![cfg(target_os = "macos")]

use super::*;
use crate::{kernels::Executor, paged::ResidentPagedMatrix,
    resident_fd::{ResidentFds, RegistryError}};
use std::cell::Cell;

fn fixture() -> (ResidentFds, Option<FileLease>) {
    let registry = ResidentFds::prepare(1).unwrap();
    let lease = registry.acquire_with(0, || tempfile::tempfile().map(Into::into)).unwrap();
    (registry, Some(lease))
}

fn allow(_: IoAttempt) -> Result<(), ()> { Ok(()) }

fn writer(coordinator: &Option<FileLease>, size: u64) -> ResidentPublicFileWriter {
    ResidentPublicFileWriter::from_resident(size, None,
        coordinator.as_ref().unwrap().clone(), &mut allow).unwrap()
}

fn sealed(coordinator: &Option<FileLease>) -> ResidentPublicFile {
    let mut writer = writer(coordinator, 4);
    writer.write_authorized(b"abcd", &mut allow).unwrap();
    writer.finish().unwrap()
}

#[test]
fn resident_constructor_bounds_precede_authorization_and_keep_coordinator() {
    let (registry, mut coordinator) = fixture();
    for size in [0, MAX_BYTES + 1] {
        assert!(matches!(ResidentPublicFileWriter::from_resident(size, None,
            coordinator.as_ref().unwrap().clone(), &mut |_| -> Result<(), ()> {
                panic!("bounds must precede authorization")
            }), Err(FacadeError::Bounds)));
        assert!(coordinator.is_some());
    }
    assert!(matches!(ResidentPublicFileWriter::from_resident(4, None,
        coordinator.as_ref().unwrap().clone(), &mut |_| Err(7)),
        Err(FacadeError::Authorization(7))));
    assert!(coordinator.is_some());
    registry.try_retire(&mut coordinator).unwrap();
    assert!(coordinator.is_none());
}

#[test]
fn resident_constructor_requires_empty_regular_file() {
    for directory in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let registry = ResidentFds::prepare(1).unwrap();
        let file = if directory {
            File::open(root.path()).unwrap()
        } else {
            let mut file = tempfile::tempfile().unwrap();
            file.write_all(b"x").unwrap();
            file
        };
        let mut coordinator = Some(registry.acquire_with(0,
            || Ok::<_, ()>(file.into())).unwrap());
        assert!(matches!(ResidentPublicFileWriter::from_resident(4, None,
            coordinator.as_ref().unwrap().clone(), &mut allow), Err(FacadeError::InvalidBacking)));
        assert!(coordinator.is_some());
        registry.try_retire(&mut coordinator).unwrap();
    }
}

#[test]
fn resident_write_denial_has_zero_attempts_and_poisoned_writer() {
    let (registry, mut coordinator) = fixture();
    let mut writer = writer(&coordinator, 4);
    let attempts = Cell::new(0);
    assert!(matches!(writer.write_authorized_with(b"abcd", &mut |_| Err(9), |_, _| {
        attempts.set(attempts.get() + 1); Ok(4)
    }), Err(FacadeError::Authorization(9))));
    assert_eq!(attempts.get(), 0);
    assert!(writer.pending.is_none());
    assert!(matches!(writer.write_authorized(b"abcd", &mut allow), Err(FacadeError::Closed)));
    assert_eq!(writer.finish().err().unwrap(), "public snapshot writer is closed");
    assert!(coordinator.is_some());
    registry.try_retire(&mut coordinator).unwrap();
}

#[test]
fn resident_partial_write_reauthorizes_remaining_and_advanced_offset() {
    let (registry, mut coordinator) = fixture();
    let mut writer = writer(&coordinator, 4);
    writer.write_authorized(b"a", &mut allow).unwrap();
    let mut trail = Vec::new();
    let mut calls = 0;
    writer.write_authorized_with(b"bcd", &mut |attempt| {
        trail.push(attempt); Ok::<_, ()>(())
    }, |fd, bytes| {
        calls += 1;
        write_once(fd, &bytes[..if calls == 1 { 1 } else { bytes.len() }])
    }).unwrap();
    assert_eq!(trail, [IoAttempt::Write { bytes: 3, logical_offset: 1 },
        IoAttempt::Write { bytes: 2, logical_offset: 2 }]);
    let snapshot = writer.finish().unwrap();
    assert_eq!(snapshot.digest(), <[u8; 32]>::from(Sha256::digest(b"abcd")));
    let mut dst = [0; 4];
    snapshot.read_authorized(0, &mut dst, &mut allow).unwrap();
    assert_eq!(&dst, b"abcd");
    drop(snapshot);
    registry.try_retire(&mut coordinator).unwrap();
}

#[test]
fn resident_partial_refusal_and_unwind_drop_only_reader_lease() {
    for unwind in [false, true] {
        let (registry, mut coordinator) = fixture();
        let mut writer = writer(&coordinator, 4);
        let mut charges = 0;
        let mut attempts = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            writer.write_authorized_with(b"abcd", &mut |_| {
                charges += 1;
                if charges == 2 {
                    if unwind { panic!("authorization unwind"); }
                    return Err(5);
                }
                Ok(())
            }, |fd, bytes| {
                attempts += 1;
                write_once(fd, &bytes[..1])
            })
        }));
        if unwind { assert!(result.is_err()); }
        else { assert!(matches!(result.unwrap(), Err(FacadeError::Authorization(5)))); }
        assert_eq!(attempts, 1);
        assert!(writer.pending.is_none());
        assert!(coordinator.is_some());
        let observation = observe_once(coordinator.as_ref().unwrap().descriptor().unwrap()).unwrap();
        assert_eq!(observation.length, 1);
        drop(writer);
        registry.try_retire(&mut coordinator).unwrap();
    }
}

#[test]
fn resident_write_eintr_zero_and_backend_error_never_retry() {
    for outcome in [Err(IoOnceError::Errno(libc::EINTR)), Ok(0), Ok(5)] {
        let (registry, mut coordinator) = fixture();
        let mut writer = writer(&coordinator, 4);
        let mut calls = 0;
        let result = writer.write_authorized_with(b"abcd", &mut allow, |_, _| {
            calls += 1; outcome
        });
        match outcome {
            Err(error) => assert!(matches!(result, Err(FacadeError::Io(e)) if e == error)),
            Ok(0) => assert!(matches!(result, Err(FacadeError::WriteZero))),
            _ => assert!(matches!(result, Err(FacadeError::Backend))),
        }
        assert_eq!(calls, 1);
        assert!(writer.pending.is_none());
        assert!(coordinator.is_some());
        drop(writer);
        registry.try_retire(&mut coordinator).unwrap();
    }
}

#[test]
fn resident_bounds_poison_before_callback_and_closed_stays_closed() {
    for bytes in [Vec::new(), vec![0; 5], vec![0; MAX_CHUNK_BYTES + 1]] {
        let (registry, mut coordinator) = fixture();
        let mut writer = writer(&coordinator, 4);
        assert!(matches!(writer.write_authorized(&bytes,
            &mut |_| -> Result<(), ()> { panic!("invalid append authorized") }), Err(FacadeError::Bounds)));
        assert!(writer.pending.is_none());
        assert!(matches!(writer.write_authorized(b"abcd", &mut allow), Err(FacadeError::Closed)));
        assert!(coordinator.is_some());
        drop(writer);
        registry.try_retire(&mut coordinator).unwrap();
    }
}

#[test]
fn resident_digest_and_size_failure_preserve_external_coordinator() {
    for bad_digest in [false, true] {
        let (registry, mut coordinator) = fixture();
        let mut writer = ResidentPublicFileWriter::from_resident(4,
            if bad_digest { Some([0; 32]) } else { None },
            coordinator.as_ref().unwrap().clone(), &mut allow).unwrap();
        writer.write_authorized(if bad_digest { b"abcd" } else { b"abc" }, &mut allow).unwrap();
        assert_eq!(writer.finish().err().unwrap(), "public snapshot digest or size differs");
        assert!(coordinator.is_some());
        registry.try_retire(&mut coordinator).unwrap();
    }
}

#[test]
fn resident_genuine_write_read_observe_and_snapshot_clone_share_backing() {
    let (registry, mut coordinator) = fixture();
    let snapshot = sealed(&coordinator);
    let duplicate = snapshot.clone();
    assert!(Arc::ptr_eq(&snapshot.file, &duplicate.file));
    assert_eq!(Arc::strong_count(&snapshot.file), 2);
    let weak = Arc::downgrade(&snapshot.file);
    let mut trail = Vec::new();
    let observation = snapshot.observe_authorized(&mut |a| {
        trail.push(a); Ok::<_, ()>(())
    }).unwrap();
    assert_eq!(observation.length, 4);
    let mut dst = [0; 2];
    duplicate.read_authorized(2, &mut dst, &mut |a| {
        trail.push(a); Ok::<_, ()>(())
    }).unwrap();
    assert_eq!(&dst, b"cd");
    assert_eq!(trail, [IoAttempt::Observe, IoAttempt::Read { bytes: 2, offset: 2 }]);
    assert!(matches!(snapshot.observe_authorized(&mut |_| Err(3)), Err(FacadeError::Authorization(3))));
    drop(snapshot);
    assert_eq!(registry.try_retire(&mut coordinator), Err(RegistryError::Busy));
    assert!(coordinator.is_some());
    drop(duplicate);
    assert!(weak.upgrade().is_none());
    // Last backing drop did not close or acknowledge: coordinator still lives.
    assert!(coordinator.as_ref().unwrap().descriptor().is_some());
    registry.try_retire(&mut coordinator).unwrap();
}

#[test]
fn resident_read_denial_bounds_and_partial_remaining_reauthorization() {
    let (registry, mut coordinator) = fixture();
    let snapshot = sealed(&coordinator);
    let mut dst = [0xa5; 3];
    assert!(matches!(snapshot.file.read_authorized_with(snapshot.size(), 1, &mut dst, &mut |_| Err(8),
        |_, _, _| panic!("denied read attempted")), Err(FacadeError::Authorization(8))));
    assert_eq!(dst, [0xa5; 3]);
    for (offset, length) in [(0, 0), (u64::MAX, 1), (3, 2), (0, MAX_CHUNK_BYTES + 1)] {
        assert!(matches!(snapshot.read_authorized(offset, &mut vec![0; length],
            &mut |_| -> Result<(), ()> { panic!("bounds authorized") }), Err(FacadeError::Bounds)));
    }
    let mut trail = Vec::new();
    let mut calls = 0;
    snapshot.file.read_authorized_with(snapshot.size(), 1, &mut dst, &mut |a| {
        trail.push(a); Ok::<_, ()>(())
    }, |fd, dst, offset| {
        calls += 1;
        let length = if calls == 1 { 1 } else { dst.len() };
        read_at_once(fd, &mut dst[..length], offset)
    }).unwrap();
    assert_eq!(&dst, b"bcd");
    assert_eq!(trail, [IoAttempt::Read { bytes: 3, offset: 1 },
        IoAttempt::Read { bytes: 2, offset: 2 }]);
    drop(snapshot);
    registry.try_retire(&mut coordinator).unwrap();
}

#[test]
fn resident_read_eintr_eof_backend_and_partial_refusal_never_hide_retry() {
    let (registry, mut coordinator) = fixture();
    let snapshot = sealed(&coordinator);
    for outcome in [Err(IoOnceError::Errno(libc::EINTR)), Ok(0), Ok(5)] {
        let mut calls = 0;
        let result = snapshot.file.read_authorized_with(snapshot.size(), 0, &mut [0; 4], &mut allow, |_, _, _| {
            calls += 1; outcome
        });
        match outcome {
            Err(error) => assert!(matches!(result, Err(FacadeError::Io(e)) if e == error)),
            Ok(0) => assert!(matches!(result, Err(FacadeError::UnexpectedEof))),
            _ => assert!(matches!(result, Err(FacadeError::Backend))),
        }
        assert_eq!(calls, 1);
    }
    let mut dst = [0xa5; 4];
    let mut charges = 0;
    let mut calls = 0;
    assert!(matches!(snapshot.file.read_authorized_with(snapshot.size(), 0, &mut dst, &mut |_| {
        charges += 1;
        if charges == 2 { Err(6) } else { Ok(()) }
    }, |fd, dst, offset| {
        calls += 1;
        read_at_once(fd, &mut dst[..1], offset)
    }), Err(FacadeError::Authorization(6))));
    assert_eq!(calls, 1);
    assert_eq!(dst, [b'a', 0xa5, 0xa5, 0xa5]);
    assert!(coordinator.is_some());
    drop(snapshot);
    registry.try_retire(&mut coordinator).unwrap();
}

#[test]
fn resident_escaped_paged_arc_clone_blocks_then_allows_explicit_retirement() {
    let (registry, mut coordinator) = fixture();
    let snapshot = sealed(&coordinator);
    let weak = Arc::downgrade(&snapshot.file);
    let executor = Arc::new(Executor::new(1, false).unwrap());
    let matrix = Arc::new(ResidentPagedMatrix::from_public_file(&snapshot,
        snapshot.digest(), 2, 2, executor).unwrap());
    let escaped = Arc::clone(&matrix);
    assert_eq!(matrix.shape(), (2, 2));
    assert_eq!(matrix.artifact_bytes(), 4);
    assert_eq!(matrix.weight_digest(), snapshot.digest());
    assert!(matrix.metadata_bytes() > 0);
    drop(snapshot);
    drop(matrix);
    assert_eq!(registry.try_retire(&mut coordinator), Err(RegistryError::Busy));
    assert!(coordinator.is_some());
    drop(escaped);
    assert!(weak.upgrade().is_none());
    assert!(coordinator.is_some());
    registry.try_retire(&mut coordinator).unwrap();
    assert!(coordinator.is_none());
}

#[test]
fn resident_row_reads_first_middle_last_and_page_boundaries() {
    let (registry, mut coordinator) = fixture();
    let (rows, cols) = (1031usize, 17usize);
    let bytes: Vec<u8> = (0..rows).flat_map(|row| {
        (0..cols).map(move |col| (row * 37 + col * 13) as u8)
    }).collect();
    let mut writer = writer(&coordinator, bytes.len() as u64);
    writer.write_authorized(&bytes, &mut allow).unwrap();
    let snapshot = writer.finish().unwrap();
    let executor = Arc::new(Executor::new(1, false).unwrap());
    let matrix = ResidentPagedMatrix::from_public_file(&snapshot,
        snapshot.digest(), rows, cols, executor).unwrap();
    for row in [0, 511, 512, 515, 1023, 1024, 1030] {
        let mut dst = [0xa5; 17];
        let mut trail = Vec::new();
        matrix.read_row_authorized(row, &mut dst, &mut |attempt| {
            trail.push(attempt); Ok::<_, ()>(())
        }).unwrap();
        for (col, byte) in dst.into_iter().enumerate() {
            assert_eq!(byte, (row * 37 + col * 13) as u8);
        }
        assert_eq!(trail[0], IoAttempt::Read { bytes: cols, offset: (row * cols) as u64 });
    }
    drop(snapshot);
    drop(matrix);
    assert!(coordinator.is_some());
    registry.try_retire(&mut coordinator).unwrap();
}

#[test]
fn resident_row_bounds_and_denial_leave_destination_unchanged() {
    let (registry, mut coordinator) = fixture();
    let snapshot = sealed(&coordinator);
    let executor = Arc::new(Executor::new(1, false).unwrap());
    let matrix = ResidentPagedMatrix::from_public_file(&snapshot,
        snapshot.digest(), 2, 2, executor).unwrap();
    let mut callbacks = 0;
    for (row, length) in [(2, 2), (usize::MAX, 2), (0, 0), (0, 1), (0, 3),
        (0, MAX_CHUNK_BYTES + 1)]
    {
        let mut dst = vec![0xa5; length];
        assert!(matches!(matrix.read_row_authorized(row, &mut dst, &mut |_| {
            callbacks += 1; Ok::<_, ()>(())
        }), Err(FacadeError::Bounds)));
        assert!(dst.iter().all(|&byte| byte == 0xa5));
    }
    assert_eq!(callbacks, 0);
    let mut dst = [0xa5; 2];
    assert!(matches!(matrix.read_row_authorized(1, &mut dst, &mut |attempt| {
        callbacks += 1;
        assert_eq!(attempt, IoAttempt::Read { bytes: 2, offset: 2 });
        Err(11)
    }), Err(FacadeError::Authorization(11))));
    assert_eq!(callbacks, 1);
    assert_eq!(dst, [0xa5; 2]);
    drop(snapshot);
    drop(matrix);
    assert!(coordinator.is_some());
    registry.try_retire(&mut coordinator).unwrap();
}

#[test]
fn resident_row_escaped_matrix_clone_requires_explicit_retirement() {
    let (registry, mut coordinator) = fixture();
    let snapshot = sealed(&coordinator);
    let weak = Arc::downgrade(&snapshot.file);
    let executor = Arc::new(Executor::new(1, false).unwrap());
    let matrix = Arc::new(ResidentPagedMatrix::from_public_file(&snapshot,
        snapshot.digest(), 2, 2, executor).unwrap());
    let escaped = Arc::clone(&matrix);
    drop(snapshot);
    drop(matrix);
    assert_eq!(registry.try_retire(&mut coordinator), Err(RegistryError::Busy));
    assert!(coordinator.is_some());
    let mut dst = [0; 2];
    escaped.read_row_authorized(1, &mut dst, &mut allow).unwrap();
    assert_eq!(&dst, b"cd");
    drop(escaped);
    assert!(weak.upgrade().is_none());
    assert!(coordinator.as_ref().unwrap().descriptor().is_some());
    registry.try_retire(&mut coordinator).unwrap();
    assert!(coordinator.is_none());
}

#[test]
fn generic_snapshot_clone_requires_no_backing_clone() {
    struct NotClone;
    let snapshot = PublicFileStorage {
        file: Arc::new(PublicFileBacking { file: NotClone, _directory: None }),
        size: 7, digest: [3; 32], max_i8: 128,
    };
    let duplicate = snapshot.clone();
    assert!(Arc::ptr_eq(&snapshot.file, &duplicate.file));
    assert_eq!((duplicate.size(), duplicate.digest(), duplicate.max_i8()), (7, [3; 32], 128));
}

#[test]
fn legacy_typed_layout_inventory_stays_file_specialized() {
    use std::mem::{align_of, offset_of, size_of};
    #[allow(dead_code)]
    struct LegacyBacking { file: File, directory: Option<Arc<dyn SnapshotDirectory>> }
    #[allow(dead_code)]
    struct LegacyPending {
        file: File, directory: Option<Arc<dyn SnapshotDirectory>>, size: u64,
        expected: Option<[u8; 32]>, written: u64, hash: Sha256, max_i8: u32,
    }
    #[allow(dead_code)]
    struct LegacyWriter { pending: Option<LegacyPending> }
    #[allow(dead_code)]
    struct LegacySnapshot {
        file: Arc<LegacyBacking>, size: u64, digest: [u8; 32], max_i8: u32,
    }
    assert_eq!(size_of::<PublicFileBacking<File>>(), size_of::<LegacyBacking>());
    assert_eq!(align_of::<PublicFileBacking<File>>(), align_of::<LegacyBacking>());
    assert_eq!(size_of::<Pending<File>>(), size_of::<LegacyPending>());
    assert_eq!(align_of::<Pending<File>>(), align_of::<LegacyPending>());
    assert_eq!(size_of::<PublicFileWriter>(), size_of::<LegacyWriter>());
    assert_eq!(align_of::<PublicFileWriter>(), align_of::<LegacyWriter>());
    assert_eq!(size_of::<PublicFile>(), size_of::<LegacySnapshot>());
    assert_eq!(align_of::<PublicFile>(), align_of::<LegacySnapshot>());
    assert_eq!(offset_of!(PublicFileBacking<File>, file), offset_of!(LegacyBacking, file));
    assert_eq!(offset_of!(PublicFileBacking<File>, _directory), offset_of!(LegacyBacking, directory));
    assert_eq!(offset_of!(Pending<File>, file), offset_of!(LegacyPending, file));
    assert_eq!(offset_of!(Pending<File>, directory), offset_of!(LegacyPending, directory));
    assert_eq!(offset_of!(Pending<File>, size), offset_of!(LegacyPending, size));
    assert_eq!(offset_of!(Pending<File>, expected), offset_of!(LegacyPending, expected));
    assert_eq!(offset_of!(Pending<File>, written), offset_of!(LegacyPending, written));
    assert_eq!(offset_of!(Pending<File>, hash), offset_of!(LegacyPending, hash));
    assert_eq!(offset_of!(Pending<File>, max_i8), offset_of!(LegacyPending, max_i8));
    assert_eq!(offset_of!(PublicFileWriter, pending), offset_of!(LegacyWriter, pending));
    assert_eq!(offset_of!(PublicFile, file), offset_of!(LegacySnapshot, file));
    assert_eq!(offset_of!(PublicFile, size), offset_of!(LegacySnapshot, size));
    assert_eq!(offset_of!(PublicFile, digest), offset_of!(LegacySnapshot, digest));
    assert_eq!(offset_of!(PublicFile, max_i8), offset_of!(LegacySnapshot, max_i8));
    let (layout, _) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(std::alloc::Layout::new::<LegacyBacking>()).unwrap();
    assert_eq!(backing_bytes().unwrap(), layout.pad_to_align().size());
    let expected = 8 * MAX_DIRECTORY_PATH_BYTES + size_of::<LegacyPending>()
        + size_of::<LegacyWriter>() + size_of::<Result<LegacyWriter, String>>()
        + size_of::<Result<File, io::Error>>() + 2 * size_of::<std::path::PathBuf>();
    assert_eq!(writer_creation_bytes().unwrap(), expected);
}
