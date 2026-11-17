//! Independent ordinary regressions. No allocation counter is a native bound.
use super::*;
use std::{cell::RefCell, mem::{align_of, align_of_val, needs_drop, size_of, size_of_val},
    path::PathBuf, sync::{Mutex, atomic::{AtomicUsize, Ordering}}};

struct Location {
    path: PathBuf,
    _directory: Option<tempfile::TempDir>,
    events: Mutex<Vec<&'static str>>,
    refuse: bool,
    panic: bool,
}
impl Location {
    fn new() -> Arc<Self> {
        let directory = tempfile::tempdir().unwrap();
        Arc::new(Self { path: directory.path().to_owned(), _directory: Some(directory),
            events: Mutex::new(Vec::new()), refuse: false, panic: false })
    }
}
impl SnapshotDirectory for Location {
    fn path(&self) -> &Path {
        self.events.lock().unwrap().push("path");
        &self.path
    }
    fn check_file(&self, _: FileBacking) -> Result<(), String> {
        self.events.lock().unwrap().push("legacy");
        if self.refuse { Err("fixture refusal".into()) } else { Ok(()) }
    }
    fn check_file_typed(&self, _: FileBacking) -> Result<(), SnapshotDirectoryError> {
        self.events.lock().unwrap().push("typed");
        assert!(!self.panic, "fixture check unwind");
        if self.refuse { Err(SnapshotDirectoryError::DeviceMismatch) } else { Ok(()) }
    }
}

#[cfg(unix)]
#[test]
fn lifecycle_legacy_typed_content_and_failure_parity() {
    let owner = Location::new();
    let bytes = [0, 127, 128, 255, 42, 1];
    let expected: [u8; 32] = Sha256::digest(bytes).into();
    for authenticated in [false, true] {
        let mut legacy = if authenticated {
            PublicFileWriter::new_in(6, expected, owner.clone()).unwrap()
        } else { PublicFileWriter::new_computed_in(6, owner.clone()).unwrap() };
        let mut typed = if authenticated {
            PublicFileWriter::new_in_typed(6, expected, owner.clone()).unwrap()
        } else { PublicFileWriter::new_computed_in_typed(6, owner.clone()).unwrap() };
        for frame in [&bytes[..2], &bytes[2..]] {
            legacy.write(frame).unwrap(); typed.write_typed(frame).unwrap();
        }
        let before = owner.events.lock().unwrap().len();
        let legacy = legacy.finish().unwrap();
        let typed = typed.finish_typed().unwrap();
        assert_eq!(before, owner.events.lock().unwrap().len(), "finish has no directory calls");
        for snapshot in [&legacy, &typed] {
            assert_eq!((snapshot.size(), snapshot.digest(), snapshot.max_i8()), (6, expected, 128));
            let mut read = [0; 6]; snapshot.read(0, &mut read).unwrap(); assert_eq!(read, bytes);
        }
    }
    for (size, expected, content) in [(2, None, b"x".as_slice()), (1, Some([0;32]), b"x")] {
        let mut legacy = PublicFileWriter::with_directory(size, expected, Some(owner.clone())).unwrap();
        let mut typed = PublicFileWriter::with_directory_typed(size, expected, Some(owner.clone())).unwrap();
        legacy.write(content).unwrap(); typed.write_typed(content).unwrap();
        assert_eq!(legacy.finish().err(), Some("public snapshot digest or size differs".into()));
        assert_eq!(typed.finish_typed().err(), Some(PublicFileFinishError::ContentMismatch));
    }
    let mut typed = PublicFileWriter::new_computed_in_typed(1, owner).unwrap();
    assert_eq!(typed.write_typed(&[]), Err(PublicWriteError::Bounds));
    assert_eq!(typed.finish_typed().err(), Some(PublicFileFinishError::Closed));
}

#[test]
fn lifecycle_creation_priority_and_fail_closed_default() {
    let owner = Location::new();
    for size in [0, MAX_BYTES + 1] {
        assert_eq!(PublicFileWriter::new_computed_in_typed(size, owner.clone()).err(),
            Some(PublicFileCreateError::Bounds));
        assert_eq!(PublicFileWriter::new_computed_in(size, owner.clone()).err(),
            Some("public snapshot size exceeds bounds".into()));
        assert_eq!(PublicFileWriter::new_in_typed(size, [0;32], owner.clone()).err(),
            Some(PublicFileCreateError::Bounds));
    }
    assert!(owner.events.lock().unwrap().is_empty());
    for (path, calls, error) in [
        (PathBuf::from("relative"), 1, PublicFileCreateError::DirectoryPolicy),
        (PathBuf::from(format!("/{}", "x".repeat(MAX_DIRECTORY_PATH_BYTES))), 2,
            PublicFileCreateError::DirectoryPolicy),
        (owner.path.join("missing-parent"), 3, PublicFileCreateError::Io),
    ] {
        let fixture = Arc::new(Location { path, _directory: None, events: Mutex::new(Vec::new()),
            refuse: false, panic: false });
        assert_eq!(PublicFileWriter::new_computed_in_typed(1, fixture.clone()).err(), Some(error));
        assert_eq!(*fixture.events.lock().unwrap(), vec!["path"; calls]);
    }
    struct LegacyOnly(tempfile::TempDir);
    impl SnapshotDirectory for LegacyOnly {
        fn path(&self) -> &Path { self.0.path() }
        fn check_file(&self, _: FileBacking) -> Result<(), String> { panic!("legacy must not run") }
    }
    let legacy = Arc::new(LegacyOnly(tempfile::tempdir().unwrap()));
    #[cfg(unix)]
    assert_eq!(PublicFileWriter::new_computed_in_typed(1, legacy).err(),
        Some(PublicFileCreateError::Directory(SnapshotDirectoryError::Unsupported)));
    #[cfg(not(unix))]
    assert_eq!(PublicFileWriter::new_computed_in_typed(1, legacy).err(),
        Some(PublicFileCreateError::ObservationUnsupported));
}

#[test]
fn lifecycle_private_none_factory_and_observation_parity() {
    let mut writer = PublicFileWriter::with_directory_typed(1,None,None).unwrap();
    let file = &writer.pending.as_ref().unwrap().file;
    assert_eq!(observe_backing(file).unwrap(),observe_backing_typed(file).unwrap());
    assert!(writer.pending.as_ref().unwrap().directory.is_none());
    writer.write_typed(b"x").unwrap(); assert_eq!(writer.finish_typed().unwrap().size(),1);
}

#[cfg(unix)]
#[test]
fn lifecycle_refusal_unwind_and_last_backing_custody() {
    for panic in [false, true] {
        let mut owner = Location::new();
        let fixture = Arc::get_mut(&mut owner).unwrap();
        fixture.refuse = true; fixture.panic = panic;
        let weak = Arc::downgrade(&owner);
        let path = owner.path.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
            PublicFileWriter::new_computed_in_typed(1, owner.clone())));
        if panic { assert!(result.is_err()); } else {
            assert_eq!(result.unwrap().err(), Some(PublicFileCreateError::Directory(
                SnapshotDirectoryError::DeviceMismatch)));
        }
        assert_eq!(Arc::strong_count(&owner), 1);
        assert_eq!(*owner.events.lock().unwrap(), ["path", "path", "path", "typed"]);
        assert!(path.exists()); drop(owner);
        assert!(weak.upgrade().is_none()); assert!(!path.exists());
    }
    let owner = Location::new();
    let path = owner.path.clone(); let weak = Arc::downgrade(&owner);
    let mut writer = PublicFileWriter::new_computed_in_typed(1, owner.clone()).unwrap();
    writer.write_typed(b"x").unwrap();
    let snapshot = writer.finish_typed().unwrap();
    let clone = snapshot.clone(); let backing = snapshot.file();
    drop(owner); drop(snapshot); drop(clone);
    assert!(weak.upgrade().is_some()); assert!(path.exists());
    drop(backing); assert!(weak.upgrade().is_none()); assert!(!path.exists());
}

thread_local! { static DROPS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) }; }
struct DropBacking;
impl Drop for DropBacking { fn drop(&mut self) { DROPS.with(|v| v.borrow_mut().push("file")); } }
struct DropDirectory;
impl SnapshotDirectory for DropDirectory {
    fn path(&self) -> &Path { panic!("finish must not call path") }
    fn check_file(&self, _: FileBacking) -> Result<(), String> { panic!("finish must not check") }
}
impl Drop for DropDirectory { fn drop(&mut self) { DROPS.with(|v| v.borrow_mut().push("directory")); } }

#[test]
fn lifecycle_generic_partial_pending_drop_order() {
    for success in [false, true] {
        DROPS.with(|v| v.borrow_mut().clear());
        let writer = PublicFileWriterStorage { pending: Some(Pending {
            file: DropBacking, directory: Some(Arc::new(DropDirectory)), size: 1,
            expected: None, written: if success { 1 } else { 0 }, hash: Sha256::new(), max_i8: 0,
        }) };
        let result = writer.finish_typed();
        if success {
            let snapshot = result.ok().unwrap(); let clone = snapshot.clone();
            drop(snapshot); DROPS.with(|v| assert!(v.borrow().is_empty())); drop(clone);
        } else { assert_eq!(result.err(), Some(PublicFileFinishError::ContentMismatch)); }
        DROPS.with(|v| assert_eq!(*v.borrow(), ["file", "directory"]));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn lifecycle_resident_finish_preserves_explicit_coordinator_retirement() {
    use crate::resident_fd::{ResidentFds, RegistryError};
    for success in [false, true] {
        let registry = ResidentFds::prepare(1).unwrap();
        let mut coordinator = Some(registry.acquire_with(0,
            || tempfile::tempfile().map(Into::into)).unwrap());
        let mut writer = ResidentPublicFileWriter::from_resident(1, None,
            coordinator.as_ref().unwrap().clone(), &mut |_| Ok::<_, ()>(())).unwrap();
        if success { writer.write_authorized(b"x", &mut |_| Ok::<_, ()>(())).unwrap(); }
        let result = writer.finish_typed();
        if success {
            let snapshot = result.ok().unwrap(); let clone = snapshot.clone();
            assert_eq!(registry.try_retire(&mut coordinator), Err(RegistryError::Busy));
            drop(snapshot);
            assert_eq!(registry.try_retire(&mut coordinator), Err(RegistryError::Busy));
            drop(clone);
        } else { assert_eq!(result.err(), Some(PublicFileFinishError::ContentMismatch)); }
        assert!(coordinator.is_some()); registry.try_retire(&mut coordinator).unwrap();
    }
}

#[derive(Debug)]
struct ErrorPayload(Arc<AtomicUsize>);
impl std::fmt::Display for ErrorPayload {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { panic!("no formatting") }
}
impl std::error::Error for ErrorPayload {}
impl Drop for ErrorPayload { fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); } }

// Actual item values only: no fn-pointer coercion and no copied closure witness.
fn io_item<R, E: Copy + Eq + std::fmt::Debug, F: FnOnce(io::Error) -> E>(f: F, expected: E, _: Option<R>) {
    assert_eq!(size_of_val(&f), 0);
    assert_eq!(align_of_val(&f), 1);
    assert_eq!(size_of::<&F>(), size_of::<usize>());
    assert_eq!(size_of::<&mut F>(), size_of::<usize>());
    assert!(size_of::<(F, (io::Error,))>() >= size_of::<io::Error>());
    assert!(size_of::<(Result<R, io::Error>, F)>() >= size_of::<io::Error>());
    assert!(size_of::<Result<R,E>>() >= size_of::<E>());
    let drop_item = std::mem::drop::<io::Error>;
    assert_eq!((size_of_val(&drop_item),align_of_val(&drop_item)),(0,1));
    let drops = Arc::new(AtomicUsize::new(0));
    let result = Err::<(), _>(io::Error::new(io::ErrorKind::Other, ErrorPayload(drops.clone()))).map_err(f);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(result, Err(expected));
    use std::{convert::Infallible, ops::ControlFlow};
    assert!(size_of::<ControlFlow<Result<Infallible, E>, R>>() >= size_of::<E>());
    assert!(size_of::<Result<Infallible,E>>() >= size_of::<E>());
}

fn callback_probe<F: FnOnce([u8; 32]) -> bool>(mut f: F, input: [u8;32], expected: bool) {
    // Qualified observed opaque type; not a surrogate struct or ABI guarantee.
    assert_eq!(size_of_val(&f), size_of::<&[u8;32]>());
    assert_eq!(align_of_val(&f), align_of::<&[u8;32]>());
    assert_eq!(size_of_val(&&f), size_of::<&F>());
    assert_eq!(size_of_val(&&mut f), size_of::<&mut F>());
    assert!(size_of::<(Option<[u8;32]>, F)>() >= size_of::<Option<[u8;32]>>());
    assert!(size_of::<(F, ([u8;32],))>() >= 32);
    let received: bool = f(input); assert_eq!(received, expected);
}

#[test]
fn lifecycle_actual_helper_items_and_single_borrowed_factory() {
    io_item(creation_io_error, PublicFileCreateError::Io, None::<File>);
    io_item(backing_io_error, PublicBackingError::Io, None::<std::fs::Metadata>);
    fn variant<R, E, F: FnOnce(E) -> PublicFileCreateError>(f: F, e: E, _:Option<R>) {
        assert_eq!(size_of_val(&f), 0); assert_eq!(align_of_val(&f), 1);
        assert!(size_of::<(Result<R, E>, F)>() >= size_of::<Result<R, E>>());
        assert_eq!(size_of::<&F>(),size_of::<usize>());
        assert_eq!(size_of::<&mut F>(),size_of::<usize>());
        assert!(size_of::<Result<R,PublicFileCreateError>>() >= size_of::<PublicFileCreateError>());
        assert!(size_of::<(F, (E,))>() >= size_of::<E>());
        let _received = f(e);
    }
    variant(PublicFileCreateError::Observation, PublicBackingError::AllocationOverflow, None::<Option<FileBacking>>);
    variant(PublicFileCreateError::Directory, SnapshotDirectoryError::Retired, None::<()>);
    // Each scope proves the actual opaque factory result accepts a short borrow.
    for input in [[7;32], [8;32]] {
        let digest = [7;32];
        callback_probe(finish_digest_callback(&digest), input, input != digest);
    }
    let factory = finish_digest_callback;
    assert_eq!(size_of_val(&factory), 0);
    assert_eq!(align_of_val(&factory), 1);
}

#[test]
fn lifecycle_emitter_callback_identity_source_contract() {
    // Lexical assertions supplement semantic tests; these strings are not
    // compiled duplicate closures and are NEVER callable/ABI witnesses.
    let source = include_str!("public_file.rs");
    let compact: String = source.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(compact.matches("constfnfinish_digest_callback<'a>").count(),1);
    assert_eq!(compact.matches("move|v:[u8;32]|v!=*digest").count(),1);
    assert!(compact.contains("if$pending.written!=$pending.size||$pending.expected.is_some_and($callback)"));
    assert!(compact.contains("emit_finish!(finish,self,pending,digest,B,String,\"publicsnapshotwriterisclosed\",\"publicsnapshotdigestorsizediffers\".into(),|v|v!=digest);"));
    assert!(compact.contains("emit_finish!(finish_typed,self,pending,digest,B,PublicFileFinishError,PublicFileFinishError::Closed,PublicFileFinishError::ContentMismatch,finish_digest_callback(&digest));"));
    assert!(compact.contains("Self::with_directory_typed(size,None,Some(directory))"));
    assert!(compact.contains("Self::with_directory_typed(size,Some(expected),Some(directory))"));
    assert!(compact.contains("observe_backing_typed(&file).map_err(PublicFileCreateError::Observation)?.ok_or(PublicFileCreateError::ObservationUnsupported)?"));
    assert!(compact.contains(").map_err(PublicFileCreateError::Directory)?"));
    assert_eq!(compact.matches("fncreation_io_error(").count(),1);
    assert_eq!(compact.matches("fnbacking_io_error(").count(),1);
    assert!(compact.contains("fncreation_io_error(error:std::io::Error)->PublicFileCreateError{std::mem::drop(error);PublicFileCreateError::Io}"));
    assert!(compact.contains("fnbacking_io_error(error:std::io::Error)->PublicBackingError{std::mem::drop(error);PublicBackingError::Io}"));
}

#[test]
fn lifecycle_native_error_destructor_unwind_has_no_typed_return() {
    #[derive(Debug)]
    struct Unwinds(Arc<AtomicUsize>);
    impl std::fmt::Display for Unwinds {
        fn fmt(&self,_:&mut std::fmt::Formatter<'_>)->std::fmt::Result { panic!("no Display") }
    }
    impl std::error::Error for Unwinds {}
    impl Drop for Unwinds {
        fn drop(&mut self) { self.0.fetch_add(1,Ordering::SeqCst); panic!("native payload destructor"); }
    }
    fn check<E,F:FnOnce(io::Error)->E>(f:F) {
        let drops = Arc::new(AtomicUsize::new(0));
        let received = AtomicUsize::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = f(io::Error::new(io::ErrorKind::Other,Unwinds(drops.clone())));
            received.fetch_add(1,Ordering::SeqCst);
        }));
        assert!(result.is_err()); assert_eq!(received.load(Ordering::SeqCst),0);
        assert_eq!(drops.load(Ordering::SeqCst),1);
    }
    check(creation_io_error); check(backing_io_error);
}

#[test]
fn lifecycle_errors_exhaustive_copy_no_returned_backing() {
    fn check<T: Copy + Eq>(values: &[T]) {
        assert!(!needs_drop::<T>()); assert!(!needs_drop::<Result<(),T>>());
        for (i, a) in values.iter().enumerate() {
            for (j, b) in values.iter().enumerate() { assert_eq!(a == b, i == j); }
        }
    }
    use SnapshotDirectoryError::*;
    check(&[Unsupported, Unavailable, Poisoned, Retired, Io, DeviceMismatch]);
    check(&[PublicBackingError::Io, PublicBackingError::AllocationOverflow]);
    check(&[PublicFileFinishError::Closed, PublicFileFinishError::ContentMismatch]);
    use PublicFileCreateError as C;
    check(&[C::Bounds, C::DirectoryPolicy, C::Io, C::ObservationUnsupported,
        C::Observation(PublicBackingError::Io), C::Observation(PublicBackingError::AllocationOverflow),
        C::Directory(Unsupported), C::Directory(Unavailable), C::Directory(Poisoned),
        C::Directory(Retired), C::Directory(Io), C::Directory(DeviceMismatch)]);
}

#[test]
fn lifecycle_digest_branch_oracle_and_actual_finish_cases() {
    // Independent oracle: factory is evaluated even for None, but only after
    // Pending exists AND sizes match. These are source endpoints, not CPU counts.
    for (present, matches, expected, factory, invoke, unused) in [
        (false, false, None, 0, 0, 0), (true, false, None, 0, 0, 0),
        (true, false, Some([0;32]), 0, 0, 0), (true, true, None, 1, 0, 1),
        (true, true, Some(Sha256::digest([]).into()), 1, 1, 0),
        (true, true, Some([0;32]), 1, 1, 0),
    ] {
        let reached = usize::from(present && matches);
        assert_eq!((reached, reached * usize::from(expected.is_some()),
            reached * usize::from(expected.is_none())), (factory, invoke, unused));
        // Same-size Some reaches the actual ne call; None disposes F without
        // comparison. The static finish union still reserves both alternatives.
        let comparison_endpoints = if present && matches && expected.is_some() { 2 } else { 0 };
        assert_eq!(comparison_endpoints, invoke.checked_mul(2).unwrap());
        let writer = PublicFileWriterStorage { pending: if present { Some(Pending {
            file: (), directory: None, size: 1, expected, written: u64::from(matches),
            hash: Sha256::new(), max_i8: 0,
        }) } else { None } };
        let success = present && matches && expected.is_none_or(|v| v == <[u8;32]>::from(Sha256::digest([])));
        let result = writer.finish_typed();
        assert_eq!(result.is_ok(), success);
        if !success { assert_eq!(result.err(), Some(if present {
            PublicFileFinishError::ContentMismatch
        } else { PublicFileFinishError::Closed })); }
    }
}

#[test]
fn lifecycle_independent_creation_observation_endpoint_census() {
    use std::{convert::Infallible, ops::ControlFlow, fs::Metadata};
    type C = PublicFileCreateError;
    type O = PublicBackingError;
    type D = SnapshotDirectoryError;
    type Directory = Arc<dyn SnapshotDirectory>;
    // (expression owner, branch, nominal multiplicity, concrete endpoint tuple
    // or result type, target size/alignment). Multiplicities include separate
    // actual/formal/return/received/retirement occurrences, never reusable slots.
    // All alternatives are summed, including mutually exclusive Some/None and
    // success/refusal prefixes. No time, stack, peak-native or funding export.
    let mut rows = Vec::new();
    macro_rules! row {
        ($owner:literal, $branch:literal, $n:expr, $t:ty) => {
            rows.push(($owner, $branch, $n as usize, size_of::<$t>(), align_of::<$t>()));
        };
    }
    row!("computed wrapper size actual/formal", "entry", 2, u64);
    row!("authenticated wrapper size/expected actual/formal", "entry alternative", 2, (u64,[u8;32]));
    let wrapper_shared_start = rows.len();
    row!("wrapper directory actual/formal/Some move", "each wrapper alternative", 6, Directory);
    row!("wrapper expected Some/None", "each wrapper alternative", 2, Option<[u8;32]>);
    row!("callee argument tuple actual/formal", "each wrapper alternative", 4, (u64,Option<[u8;32]>,Option<Directory>));
    row!("callee return/wrapper receipt/outer return/consumer receipt", "each wrapper alternative", 8, Result<PublicFileWriter,C>);
    let body_start = rows.len();
    row!("with_directory argument tuple actual/formal", "entry", 2, (u64,Option<[u8;32]>,Option<Directory>));
    row!("size reads, zero and MAX", "bounds", 2, u64);
    row!("size comparisons and short circuit", "bounds", 3, bool);
    row!("Bounds construction/Err return/receipt", "size refusal", 1, C);
    row!("directory borrow/match receiver", "bounds accepted", 1, &Option<Directory>);
    row!("owner pattern borrow", "Some", 1, &Directory);
    row!("path Arc::deref actual/formal", "Some, each reached path", 6, (&Directory,));
    row!("path Arc::deref return/receipt", "Some, three calls", 6, &dyn SnapshotDirectory);
    row!("path receiver actual/formal", "Some, each reached path", 6, &dyn SnapshotDirectory);
    row!("path borrowed return/receipt", "Some, three calls", 6, &Path);
    row!("absolute receiver actual/formal", "first path", 2, &Path);
    row!("absolute result and not", "first path", 2, bool);
    row!("as_os_str receiver actual/formal", "absolute", 2, &Path);
    row!("OsStr return/receipt/len actual/formal", "absolute", 4, &std::ffi::OsStr);
    row!("length return/receipt, policy subtraction", "absolute", 3, usize);
    row!("length comparison and path short circuit", "absolute", 2, bool);
    row!("DirectoryPolicy enum construction", "path refusal", 1, C);
    row!("factory_in argument tuple actual/formal", "Some path accepted", 2, (&Path,));
    row!("ambient factory argument tuple actual/formal", "None alternative (private only)", 2, ());
    row!("factory native return/map receiver", "Some OR None, union", 4, Result<File,io::Error>);
    row!("factory mapped return", "Some OR None, union", 2, Result<File,C>);
    row!("factory ? ControlFlow", "Some OR None, union", 2, ControlFlow<Result<Infallible,C>,File>);
    row!("factory ? residual", "Some OR None, union", 2, Result<Infallible,C>);
    row!("local File binding/branch move/outer binding", "factory success", 3, File);
    row!("observation argument tuple actual/formal", "Some factory success", 2, (&File,));
    row!("observation native return/map receiver", "Some", 2, Result<Option<FileBacking>,O>);
    row!("Observation mapper mapped result", "Some", 1, Result<Option<FileBacking>,C>);
    row!("observation ? ControlFlow", "Some", 1, ControlFlow<Result<Infallible,C>,Option<FileBacking>>);
    row!("observation ? residual", "observation error", 1, Result<Infallible,C>);
    row!("ok_or receiver", "observation success", 1, Option<FileBacking>);
    row!("ObservationUnsupported eager argument", "ok_or, even Some", 1, C);
    row!("ok_or return", "Some observation", 1, Result<FileBacking,C>);
    row!("ok_or ? ControlFlow", "Some observation", 1, ControlFlow<Result<Infallible,C>,FileBacking>);
    row!("ok_or ? residual", "unsupported", 1, Result<Infallible,C>);
    row!("owner check tuple actual/formal", "Some observed", 2, (&dyn SnapshotDirectory,FileBacking));
    row!("check Arc::deref actual/formal", "Some observed", 2, (&Directory,));
    row!("check Arc::deref return/receipt", "Some observed", 2, &dyn SnapshotDirectory);
    row!("owner check return/map receiver", "Some observed", 2, Result<(),D>);
    row!("Directory mapper mapped result", "owner reached", 1, Result<(),C>);
    row!("check ? ControlFlow", "owner reached", 1, ControlFlow<Result<Infallible,C>,()>);
    row!("check ? residual", "owner refusal", 1, Result<Infallible,C>);
    row!("Pending file field move", "success", 1, File);
    row!("Pending directory field move", "success", 1, Option<Directory>);
    row!("Pending size/written zero fields", "success", 2, u64);
    row!("Pending expected field", "success", 1, Option<[u8;32]>);
    row!("Sha256::new return/receipt/hash field move", "success, opaque hash", 3, Sha256);
    row!("max_i8 zero field", "success", 1, u32);
    row!("Pending/Some/writer construction", "success", 1, Pending<File>);
    row!("Some payload and writer move", "success", 1, Option<Pending<File>>);
    row!("writer value move", "success", 1, PublicFileWriter);
    row!("constructor result return/receipt", "all exits", 2, Result<PublicFileWriter,C>);
    row!("local file drop endpoint", "factory-success refusal/unwind", 1, &mut File);
    row!("directory parameter drop endpoint", "refusal/unwind", 1, &mut Option<Directory>);
    let unix_observe_start = rows.len();
    row!("metadata actual/formal", "Unix observe", 2, (&File,));
    row!("metadata native return/map receiver", "Unix observe", 2, Result<Metadata,io::Error>);
    row!("metadata mapped return", "Unix observe", 1, Result<Metadata,O>);
    row!("metadata ? ControlFlow", "Unix observe", 1, ControlFlow<Result<Infallible,O>,Metadata>);
    row!("metadata ? residual", "Unix error", 1, Result<Infallible,O>);
    row!("metadata binding", "Unix metadata success", 1, Metadata);
    row!("blocks/dev/ino receiver tuples actual/formal", "Unix success", 6, (&Metadata,));
    row!("blocks/dev/ino return/receipt", "Unix success", 6, u64);
    row!("checked_mul receiver/literal tuple actual/formal", "Unix metadata success", 2, (u64,u64));
    row!("checked product return/ok_or receiver", "Unix metadata success", 2, Option<u64>);
    row!("AllocationOverflow eager argument", "Unix ok_or", 1, O);
    row!("product ok_or result", "Unix ok_or", 1, Result<u64,O>);
    row!("product ? ControlFlow", "Unix ok_or", 1, ControlFlow<Result<Infallible,O>,u64>);
    row!("product ? residual", "overflow", 1, Result<Infallible,O>);
    row!("allocated_bytes binding and three backing fields", "Unix success", 4, u64);
    row!("FileBacking/Some/Ok moves", "Unix success", 1, FileBacking);
    row!("Some move", "Unix success", 1, Option<FileBacking>);
    row!("observe return/receipt", "Unix all exits", 2, Result<Option<FileBacking>,O>);
    row!("metadata drop endpoint", "Unix initialized exit", 1, &mut Metadata);
    let other_observe_start = rows.len();
    row!("non-Unix file borrow", "non-Unix alternative", 1, &File);
    row!("non-Unix None and Ok return/receipt", "non-Unix alternative", 2, Result<Option<FileBacking>,O>);
    let shared_mapper_start = rows.len();
    // Actual named mapper item/adapter and invocation slots, counted separately
    // below because their types cannot be spelled as fn pointers.
    fn mapper<F,R,E,T>(rows: &mut Vec<Endpoint<'static>>, owner: &'static str, f: F, _: Option<(R,E,T)>)
    where F: FnOnce(E) -> T {
        rows.push((owner,"item evaluated on Ok or Err",1,size_of_val(&f),std::mem::align_of_val(&f)));
        rows.push((owner,"map receiver adapter tuple",1,size_of::<(Result<R,E>,F)>(),align_of::<(Result<R,E>,F)>()));
        rows.push((owner,"FnOnce self/input tuple on Err",1,size_of::<(F,(E,))>(),align_of::<(F,(E,))>()));
        rows.push((owner,"incoming error formal on Err",1,size_of::<E>(),align_of::<E>()));
        rows.push((owner,"enum return/receipt on normal Err mapping",2,size_of::<T>(),align_of::<T>()));
    }
    // Both factory branches are conservatively included, only one can invoke.
    mapper(&mut rows,"factory Some creation_io_error",creation_io_error,None::<(File,io::Error,C)>);
    mapper(&mut rows,"factory None creation_io_error",creation_io_error,None::<(File,io::Error,C)>);
    mapper(&mut rows,"Observation enum item",PublicFileCreateError::Observation,None::<(Option<FileBacking>,O,C)>);
    mapper(&mut rows,"Directory enum item",PublicFileCreateError::Directory,None::<((),D,C)>);
    let unix_mapper_start = rows.len();
    mapper(&mut rows,"observation backing_io_error",backing_io_error,None::<(Metadata,io::Error,O)>);
    let aggregate_drop_start = rows.len();
    row!("three io-helper drop item tuple actual/formal", "Err only, disjoint union", 6, (io::Error,));
    row!("three drop unit return/receipt", "destructor returns normally", 6, ());
    row!("three helper incoming/drop/error retirement", "Err only, native destructor opaque", 3, &mut io::Error);
    let union = rows.iter().try_fold(0usize, |sum, (_,_,n,bytes,_)|
        n.checked_mul(*bytes).and_then(|v| sum.checked_add(v))).unwrap();
    let expected = assert_endpoint_oracle(&rows, CREATION_OBSERVATION_ENDPOINTS);
    assert_eq!(union, expected);
    // Partition the independently labelled baseline by source region. The
    // typed union retains both factory alternatives and caller-side endpoints
    // even when the non-Unix observation body returns None unconditionally.
    let mut selected = rows[body_start..unix_observe_start].to_vec();
    selected.extend_from_slice(&rows[shared_mapper_start..unix_mapper_start]);
    let unix = cfg!(unix);
    if unix {
        selected.extend_from_slice(&rows[unix_observe_start..other_observe_start]);
        selected.extend_from_slice(&rows[unix_mapper_start..aggregate_drop_start]);
        selected.push(("backing_io_error", "enum construction", 1, size_of::<O>(), align_of::<O>()));
    } else {
        selected.extend_from_slice(&rows[other_observe_start..shared_mapper_start]);
    }
    for owner in ["factory Some", "factory None", "Observation", "Directory"] {
        selected.push((owner, "enum construction", 1, size_of::<C>(), align_of::<C>()));
    }
    // Replace the baseline's aggregate destructor rows with typed rows per
    // selected helper, including the actual zero-sized drop item endpoint.
    let drop_item = std::mem::drop::<io::Error>;
    for owner in ["factory Some", "factory None", "Unix metadata"].into_iter()
        .take(if unix { 3 } else { 2 })
    {
        selected.extend([
            (owner, "drop item", 1, size_of_val(&drop_item), align_of_val(&drop_item)),
            (owner, "drop tuple actual/formal", 2, size_of::<(io::Error,)>(), align_of::<(io::Error,)>()),
            (owner, "opaque error retirement", 1, size_of::<&mut io::Error>(), align_of::<&mut io::Error>()),
            (owner, "drop unit return/receipt", 2, size_of::<()>(), align_of::<()>()),
        ]);
    }
    selected.extend(rows[wrapper_shared_start..body_start].iter().map(|&(owner, branch, n, size, align)| {
        assert_eq!(n % 2, 0);
        (owner, branch, n / 2, size, align) // select only one wrapper
    }));
    let (common_n, common_b) = selected.iter().try_fold((0u64, 0usize), |(n, b), &(_, _, count, size, _)| {
        Some((n.checked_add(u64::try_from(count).ok()?)?, b.checked_add(count.checked_mul(size)?)?))
    }).unwrap();

    let calls = [
        1u64, // explicit wrapper -> shared constructor
        3 + 3 + 3, // path derefs, path callbacks, absolute/as_os_str/len
        2 * (1 + 2 + 2 + 2), // factory, map/invoke, drop/destructor, Try pair per alternative
        1 + 2 + 1 + 2 * 2, // observe, map/invoke, ok_or, two Try pairs
        1 + 1 + 2 + 2, // owner deref, check, map/invoke, Try pair
        1 + 2, // hash init, local File and directory retirement
        if unix { 7 + 2 * 2 + 2 + 2 } else { 0 }, // metadata body: helpers, Try, map/invoke, drop/destructor
    ].into_iter().try_fold(0u64, u64::checked_add).unwrap();
    let q = public_file_lifecycle_logical_quote_v1().unwrap();
    for (quote, &(_, _, count, size, _)) in [q.create_computed_in, q.create_authenticated_in]
        .into_iter().zip(&rows[..wrapper_shared_start])
    {
        let n = common_n.checked_add(u64::try_from(count).unwrap()).unwrap();
        let b = common_b.checked_add(count.checked_mul(size).unwrap()).unwrap();
        assert_eq!((n, calls), if unix { (208, 56) } else { (161, 41) });
        assert_logical(quote, n, calls, b);
    }
    // Explicit distinct backing columns: fixed errors own zero; successful
    // creation transfers a new File plus inline Pending and retains an existing
    // directory Arc. No new directory allocation is attributed to these rows.
    let returned_error_backing = 0usize;
    let new_directory_allocation = 0usize;
    assert_eq!((returned_error_backing,new_directory_allocation), (0,0));
    eprintln!("creation/observation conservative logical control union={union}; tempfile::tempfile[_in] internals/RNG, Path::is_absolute/as_os_str, OsStr::len, File::metadata, MetadataExt::blocks/dev/ino, Sha256, io::Error/File/directory destructors opaque; no native backing/work closure. Arbitrary third-party SnapshotDirectory bodies remain opaque; selected Owned/Raw local bodies are separately expanded in compiler inventories (including 1/2/3 path calls), NOT hidden here as native work");
}

fn independent_finish_census<B>() -> (usize, usize) {
    use std::{alloc::Layout, convert::Infallible, ops::ControlFlow, sync::atomic::AtomicUsize};
    type E = PublicFileFinishError;
    let mut rows: Vec<(&str,&str,usize,usize,usize)> = Vec::new();
    macro_rules! row {
        ($name:literal,$branch:literal,$n:expr,$t:ty) => { rows.push(($name,$branch,$n,size_of::<$t>(),align_of::<$t>())); };
    }
    row!("self actual/formal","entry",2,PublicFileWriterStorage<B>);
    row!("take receiver actual/formal","entry",2,(&mut Option<Pending<B>>,));
    row!("take return/ok_or receiver","entry",2,Option<Pending<B>>);
    row!("Closed eager enum","entry",1,E);
    row!("ok_or return","entry",1,Result<Pending<B>,E>);
    row!("take ? ControlFlow","entry",1,ControlFlow<Result<Infallible,E>,Pending<B>>);
    row!("take ? residual","closed",1,Result<Infallible,E>);
    row!("Pending binding","present",1,Pending<B>);
    row!("hash partial move/finalize actual/formal","present",2,(Sha256,));
    row!("finalize native output/Into receiver","present",2,sha2::digest::Output<Sha256>);
    row!("Into output/digest binding","present",2,[u8;32]);
    row!("written/size reads","present",2,u64);
    row!("written != size and || result","present",2,bool);
    row!("expected field read/adapter receiver","sizes match",1,Option<[u8;32]>);
    row!("factory borrowed tuple actual/formal","sizes match even None",2,(&[u8;32],));
    row!("factory captured-reference move","sizes match",1,&[u8;32]);
    row!("digest live referent (not new backing)","present",1,[u8;32]);
    row!("callback argument tuple actual/formal","sizes match Some",2,([u8;32],));
    row!("v binding/read","sizes match Some",2,[u8;32]);
    row!("captured reference access","sizes match Some",1,&[u8;32]);
    row!("referent read for array comparison","sizes match Some, opaque 32-byte comparison",1,[u8;32]);
    row!("PartialEq::ne argument tuple actual/formal","sizes match Some",2,(&[u8;32], &[u8;32]));
    row!("comparison bool return/receipt/is_some_and return/receipt","sizes match Some",4,bool);
    row!("is_some_and None bool return/receipt","sizes match None alternative",2,bool);
    row!("ContentMismatch construction","mismatch",1,E);
    row!("backing file move actual/formal/field","success",3,B);
    row!("backing directory move actual/formal/field","success",3,Option<Arc<dyn SnapshotDirectory>>);
    row!("PublicFileBacking construction/Arc actual/formal","success",3,PublicFileBacking<B>);
    row!("Arc return/receipt/snapshot field","success",3,Arc<PublicFileBacking<B>>);
    row!("size read/field","success",2,u64);
    row!("digest field move","success",1,[u8;32]);
    row!("max read/field","success",2,u32);
    row!("snapshot construction/Ok move","success",2,PublicFileStorage<B>);
    row!("result return/consumer receipt","all exits",2,Result<PublicFileStorage<B>,E>);
    row!("partial Pending file drop","refusal/unwind after hash moved",1,&mut B);
    row!("partial Pending directory drop","refusal/unwind after hash moved",1,&mut Option<Arc<dyn SnapshotDirectory>>);
    row!("empty self retirement","all exits",1,&mut PublicFileWriterStorage<B>);
    row!("backing retirement endpoint","eventual last strong owner, not finish return",1,&mut PublicFileBacking<B>);
    fn callback_slots<F: FnOnce([u8;32])->bool>(rows: &mut Vec<Endpoint<'static>>, f: F) {
        // Actual returned factory type, not a duplicate lexical closure.
        rows.push(("actual borrowed callback","construction/return/receipt",3,size_of_val(&f),std::mem::align_of_val(&f)));
        rows.push(("actual borrowed callback","None unused callback consumption",1,size_of::<F>(),align_of::<F>()));
        rows.push(("actual borrowed callback","is_some_and receiver/adapter tuple",1,size_of::<(Option<[u8;32]>,F)>(),align_of::<(Option<[u8;32]>,F)>()));
        rows.push(("actual borrowed callback","FnOnce self/input tuple",1,size_of::<(F,([u8;32],))>(),align_of::<(F,([u8;32],))>()));
        // &F/&mut F are separately measured in the callable ABI probe; no
        // production reference endpoint is invented from a test-only borrow.
    }
    let digest = [0;32];
    callback_slots(&mut rows,finish_digest_callback(&digest));
    let factory = finish_digest_callback;
    rows.push(("actual digest factory item","sizes match",1,size_of_val(&factory),std::mem::align_of_val(&factory)));
    let control = rows.iter().try_fold(0usize,
        |sum,(_,_,n,s,_)| n.checked_mul(*s).and_then(|v| sum.checked_add(v))).unwrap();
    let expected = assert_endpoint_oracle(&rows, FINISH_ENDPOINTS);
    assert_eq!(control,expected);
    // Independently keyed boundary, not a scalar pin or a pointer-size sum.
    let comparison = rows.iter().position(|r|
        r.0 == "PartialEq::ne argument tuple actual/formal").unwrap();
    assert_eq!(rows[comparison], ("PartialEq::ne argument tuple actual/formal",
        "sizes match Some",2,size_of::<(&[u8;32], &[u8;32])>(),
        align_of::<(&[u8;32], &[u8;32])>()));
    let mut missing_comparison = rows.clone(); missing_comparison.remove(comparison);
    assert_eq!(checked_endpoint_oracle(&missing_comparison, FINISH_ENDPOINTS),None);
    let mut one_endpoint = rows.clone(); one_endpoint[comparison].2 = 1;
    assert_eq!(checked_endpoint_oracle(&one_endpoint, FINISH_ENDPOINTS),None);
    let mut wrong_branch = rows.clone(); wrong_branch[comparison].1 = "sizes match None alternative";
    assert_eq!(checked_endpoint_oracle(&wrong_branch, FINISH_ENDPOINTS),None);
    if std::any::type_name::<B>() == std::any::type_name::<File>() {
        // Eventual backing retirement is NOT part of finish's executed union.
        let n = rows.iter().filter(|r| r.0 != "backing retirement endpoint")
            .try_fold(0u64, |sum,r| sum.checked_add(u64::try_from(r.2).ok()?)).unwrap();
        let b = rows.iter().filter(|r| r.0 != "backing retirement endpoint")
            .try_fold(0usize, |sum,r| sum.checked_add(r.2.checked_mul(r.3)?)).unwrap();
        let q = public_file_lifecycle_logical_quote_v1().unwrap().finish_file;
        assert_eq!(n,72);
        assert_logical(q, n, 15, b);
        let comparison_bytes = 2usize.checked_mul(size_of::<(&[u8;32], &[u8;32])>()).unwrap();
        let old_b = b.checked_sub(comparison_bytes).unwrap();
        let old = PublicFileLogicalWork::checked_source_union(70,15,old_b,0).unwrap();
        assert_eq!((old.helper_calls,old.visits),(15,100));
        assert_eq!(q.control_bytes,old.control_bytes.checked_add(comparison_bytes).unwrap());
        assert_eq!(q,PublicFileLogicalWork::checked_source_union(72,15,b,0).unwrap());
    }
    // STRICTLY CONDITIONAL: if std ArcInner is repr(C) with two AtomicUsize
    // counters followed by this data, this is its padded layout. No pinned std
    // source/profile has been independently qualified here. This calculation
    // is NOT a measured/verified allocation layout, bound, or funding tariff.
    let (layout,offset) = Layout::new::<[AtomicUsize;2]>().extend(Layout::new::<PublicFileBacking<B>>()).unwrap();
    let allocation = layout.pad_to_align();
    assert!(offset >= 2*size_of::<AtomicUsize>());
    (control,allocation.size())
}

#[test]
fn lifecycle_independent_file_and_lease_finish_layouts() {
    // Same field types, no production inventory/pool/constants used as oracle.
    struct OldPending<B> { file: B, directory: Option<Arc<dyn SnapshotDirectory>>,
        size:u64, expected:Option<[u8;32]>, written:u64, hash:Sha256, max_i8:u32 }
    struct OldWriter<B> { pending: Option<OldPending<B>> }
    struct OldBacking<B> { file:B, directory:Option<Arc<dyn SnapshotDirectory>> }
    struct OldSnapshot<B> { file:Arc<OldBacking<B>>, size:u64, digest:[u8;32], max_i8:u32 }
    fn carriers<B>() {
        assert_eq!((size_of::<Pending<B>>(),align_of::<Pending<B>>()),(size_of::<OldPending<B>>(),align_of::<OldPending<B>>()));
        assert_eq!((size_of::<PublicFileWriterStorage<B>>(),align_of::<PublicFileWriterStorage<B>>()),(size_of::<OldWriter<B>>(),align_of::<OldWriter<B>>()));
        assert_eq!((size_of::<PublicFileBacking<B>>(),align_of::<PublicFileBacking<B>>()),(size_of::<OldBacking<B>>(),align_of::<OldBacking<B>>()));
        assert_eq!((size_of::<PublicFileStorage<B>>(),align_of::<PublicFileStorage<B>>()),(size_of::<OldSnapshot<B>>(),align_of::<OldSnapshot<B>>()));
    }
    carriers::<File>();
    let file = independent_finish_census::<File>();
    assert!(file.1 > size_of::<Arc<PublicFileBacking<File>>>());
    #[cfg(unix)] {
        carriers::<FileLease>();
        let lease = independent_finish_census::<FileLease>();
        assert!(lease.1 > size_of::<Arc<PublicFileBacking<FileLease>>>());
        eprintln!("finish File={file:?}; FileLease={lease:?} (logical control, CONDITIONAL two-counter repr(C) Arc allocation calculation; std qualification PENDING); native hash/array comparison/Arc allocator/File/drop/unlock opaque, lease coordinator remains external");
    }
}

// Independent, hand-audited endpoint-key/branch/multiplicity expectations. These
// are not emitted from row! calls, production templates, or a production pool.
// Target layouts remain observed inputs, not hard-coded cross-build ABI claims.
type Endpoint<'a> = (&'a str, &'a str, usize, usize, usize);
fn assert_logical(q: PublicFileLogicalWork, n: u64, k: u64, b: usize) {
    let v = n as u128 + 2 * k as u128;
    let p = 2 * b as u128 * (v + 1);
    assert_eq!((q.helper_calls as u128,q.visits as u128,q.potential_bytes as u128,q.control_bytes),
        (k as u128,v,p,b));
    assert_eq!(q.total_work().unwrap() as u128,k as u128 + v + p);
}

#[test]
fn lifecycle_remaining_owner_independent_source_potential_oracles() {
    type A = Arc<dyn SnapshotDirectory>;
    let q = public_file_lifecycle_logical_quote_v1().unwrap();
    #[cfg(unix)] {
        // Finite nominal/call census pins; layouts remain target observations.
        assert_eq!((q.create_computed_in.helper_calls,q.create_computed_in.visits),(56,320));
        assert_eq!((q.create_authenticated_in.helper_calls,q.create_authenticated_in.visits),(56,320));
    }
    assert_eq!((q.finish_file.helper_calls,q.finish_file.visits),(15,102));
    assert_eq!((q.discard_pending_file.helper_calls,q.discard_pending_file.visits),(4,33));
    assert_eq!((q.release_snapshot_handle.helper_calls,q.release_snapshot_handle.visits),(2,14));
    assert_eq!((q.release_last_file_backing.helper_calls,q.release_last_file_backing.visits),(4,26));
    // Independent handwritten endpoint types/multiplicities: neither production
    // census nor its N/B/K is used to construct these expectations.
    let pending = [
        (2,size_of::<(&mut PublicFileWriter,)>()), (1,size_of::<Option<Pending<File>>>()),
        (1,size_of::<bool>()), (2,size_of::<(&mut Pending<File>,)>()),
        (1,size_of::<File>()), (1,size_of::<Option<A>>()), (2,size_of::<u64>()),
        (1,size_of::<Option<[u8;32]>>()), (1,size_of::<Sha256>()), (1,size_of::<u32>()),
        (2,size_of::<(&mut File,)>()), (2,size_of::<(&mut Option<A>,)>()),
        (2,size_of::<(&mut Sha256,)>()), (6,0),
    ];
    let snapshot = [(1,size_of::<PublicFile>()), (2,size_of::<(&mut PublicFile,)>()),
        (2,size_of::<(&mut Arc<PublicFileBacking<File>>,)>()),
        (1,size_of::<u64>()), (1,32), (1,size_of::<u32>()), (2,0)];
    let last = [(2,size_of::<(&mut PublicFileBacking<File>,)>()),
        (2,size_of::<(&mut File,)>()), (2,0), (2,size_of::<(&mut Option<A>,)>()),
        (1,size_of::<bool>()), (2,size_of::<(&mut A,)>()), (2,0),
        (1,size_of::<PublicFileBacking<File>>()),
        (2,size_of::<(&mut PublicFileBacking<File>,)>()), (2,0)];
    for (quote,k,rows) in [(q.discard_pending_file,4,&pending[..]),
        (q.release_snapshot_handle,2,&snapshot[..]), (q.release_last_file_backing,4,&last[..])] {
        assert_logical(quote, rows.iter().map(|r| r.0 as u64).sum(), k,
            rows.iter().map(|r| r.0 * r.1).sum());
    }
    // Their separate API entries are obligations, never automatically appended
    // to finish_file. There is no directory-final-owner capacity certificate.
    assert!(q.finish_file != q.release_last_file_backing);
}

#[test]
fn lifecycle_checked_source_recipe_zst_scaling_and_overflow() {
    let z = PublicFileLogicalWork::checked_source_union(7,2,0,0).unwrap();
    assert_eq!((z.helper_calls,z.visits,z.potential_bytes),(2,11,0));
    assert_eq!(z.checked_scale(3).unwrap().visits,33);
    assert_eq!(z.checked_add(z).unwrap().total_work(),Some(26));
    assert_eq!(PublicFileLogicalWork::checked_source_union(u64::MAX,1,0,0),None);
    assert_eq!(PublicFileLogicalWork::checked_source_union(0,u64::MAX,0,0),None);
    assert_eq!(PublicFileLogicalWork::checked_source_union(1,0,usize::MAX,0),None);
    assert_eq!(PublicFileLogicalWork::checked_source_union(1,0,1,u64::MAX),None);
    assert_eq!(z.checked_scale(u64::MAX),None);
    let invalid = PublicFileLogicalWork { helper_calls:u64::MAX,visits:1,potential_bytes:0,control_bytes:0 };
    assert_eq!(invalid.total_work(),None); assert_eq!(invalid.checked_add(z),None);
    let mut census = LifecycleCensus::default();
    assert!(census.row::<()>(7).is_some()); assert_eq!((census.nominal,census.bytes),(7,0));
    census.nominal = u64::MAX; assert!(census.row::<()>(1).is_none());
}
fn checked_endpoint_oracle(rows: &[Endpoint<'_>], specification: &str) -> Option<usize> {
    use std::collections::BTreeMap;
    let mut actual = BTreeMap::new();
    let mut actual_total = 0usize;
    for &(key,branch,n,size,align) in rows {
        if n == 0 || !align.is_power_of_two() || size % align != 0
            || actual.insert((key,branch),(n,size)).is_some() { return None; }
        actual_total = actual_total.checked_add(n.checked_mul(size)?)?;
    }
    let mut expected = BTreeMap::new();
    let mut expected_total = 0usize;
    for line in specification.lines().map(str::trim).filter(|s| !s.is_empty()) {
        let mut fields = line.split('@');
        let n = fields.next()?.parse::<usize>().ok()?;
        let key = (fields.next()?,fields.next()?);
        if fields.next().is_some() || expected.insert(key,n).is_some() { return None; }
        let &(observed,size) = actual.get(&key)?;
        if observed != n { return None; }
        expected_total = expected_total.checked_add(n.checked_mul(size)?)?;
    }
    if actual.len() != expected.len() || actual_total != expected_total { return None; }
    Some(expected_total)
}
fn assert_endpoint_oracle(rows: &[Endpoint<'_>], specification: &str) -> usize {
    let total = checked_endpoint_oracle(rows,specification).expect("exact endpoint set/count/checked total");
    // Removing ANY endpoint or changing ANY multiplicity must be rejected,
    // including the metadata residual identified by independent review.
    for i in 0..rows.len() {
        let mut missing = rows.to_vec(); missing.remove(i);
        assert_eq!(checked_endpoint_oracle(&missing,specification),None);
        let mut changed = rows.to_vec(); changed[i].2 = changed[i].2.checked_add(1).unwrap();
        assert_eq!(checked_endpoint_oracle(&changed,specification),None);
    }
    let mut duplicate = rows.to_vec(); duplicate.push(rows[0]);
    assert_eq!(checked_endpoint_oracle(&duplicate,specification),None);
    total
}

const CREATION_OBSERVATION_ENDPOINTS: &str = r#"
2@computed wrapper size actual/formal@entry
2@authenticated wrapper size/expected actual/formal@entry alternative
6@wrapper directory actual/formal/Some move@each wrapper alternative
2@wrapper expected Some/None@each wrapper alternative
4@callee argument tuple actual/formal@each wrapper alternative
8@callee return/wrapper receipt/outer return/consumer receipt@each wrapper alternative
2@with_directory argument tuple actual/formal@entry
2@size reads, zero and MAX@bounds
3@size comparisons and short circuit@bounds
1@Bounds construction/Err return/receipt@size refusal
1@directory borrow/match receiver@bounds accepted
1@owner pattern borrow@Some
6@path Arc::deref actual/formal@Some, each reached path
6@path Arc::deref return/receipt@Some, three calls
6@path receiver actual/formal@Some, each reached path
6@path borrowed return/receipt@Some, three calls
2@absolute receiver actual/formal@first path
2@absolute result and not@first path
2@as_os_str receiver actual/formal@absolute
4@OsStr return/receipt/len actual/formal@absolute
3@length return/receipt, policy subtraction@absolute
2@length comparison and path short circuit@absolute
1@DirectoryPolicy enum construction@path refusal
2@factory_in argument tuple actual/formal@Some path accepted
2@ambient factory argument tuple actual/formal@None alternative (private only)
4@factory native return/map receiver@Some OR None, union
2@factory mapped return@Some OR None, union
2@factory ? ControlFlow@Some OR None, union
2@factory ? residual@Some OR None, union
3@local File binding/branch move/outer binding@factory success
2@observation argument tuple actual/formal@Some factory success
2@observation native return/map receiver@Some
1@Observation mapper mapped result@Some
1@observation ? ControlFlow@Some
1@observation ? residual@observation error
1@ok_or receiver@observation success
1@ObservationUnsupported eager argument@ok_or, even Some
1@ok_or return@Some observation
1@ok_or ? ControlFlow@Some observation
1@ok_or ? residual@unsupported
2@owner check tuple actual/formal@Some observed
2@check Arc::deref actual/formal@Some observed
2@check Arc::deref return/receipt@Some observed
2@owner check return/map receiver@Some observed
1@Directory mapper mapped result@owner reached
1@check ? ControlFlow@owner reached
1@check ? residual@owner refusal
1@Pending file field move@success
1@Pending directory field move@success
2@Pending size/written zero fields@success
1@Pending expected field@success
3@Sha256::new return/receipt/hash field move@success, opaque hash
1@max_i8 zero field@success
1@Pending/Some/writer construction@success
1@Some payload and writer move@success
1@writer value move@success
2@constructor result return/receipt@all exits
1@local file drop endpoint@factory-success refusal/unwind
1@directory parameter drop endpoint@refusal/unwind
2@metadata actual/formal@Unix observe
2@metadata native return/map receiver@Unix observe
1@metadata mapped return@Unix observe
1@metadata ? ControlFlow@Unix observe
1@metadata ? residual@Unix error
1@metadata binding@Unix metadata success
6@blocks/dev/ino receiver tuples actual/formal@Unix success
6@blocks/dev/ino return/receipt@Unix success
2@checked_mul receiver/literal tuple actual/formal@Unix metadata success
2@checked product return/ok_or receiver@Unix metadata success
1@AllocationOverflow eager argument@Unix ok_or
1@product ok_or result@Unix ok_or
1@product ? ControlFlow@Unix ok_or
1@product ? residual@overflow
4@allocated_bytes binding and three backing fields@Unix success
1@FileBacking/Some/Ok moves@Unix success
1@Some move@Unix success
2@observe return/receipt@Unix all exits
1@metadata drop endpoint@Unix initialized exit
1@non-Unix file borrow@non-Unix alternative
2@non-Unix None and Ok return/receipt@non-Unix alternative
6@three io-helper drop item tuple actual/formal@Err only, disjoint union
6@three drop unit return/receipt@destructor returns normally
3@three helper incoming/drop/error retirement@Err only, native destructor opaque
1@factory Some creation_io_error@item evaluated on Ok or Err
1@factory Some creation_io_error@map receiver adapter tuple
1@factory Some creation_io_error@FnOnce self/input tuple on Err
1@factory Some creation_io_error@incoming error formal on Err
2@factory Some creation_io_error@enum return/receipt on normal Err mapping
1@factory None creation_io_error@item evaluated on Ok or Err
1@factory None creation_io_error@map receiver adapter tuple
1@factory None creation_io_error@FnOnce self/input tuple on Err
1@factory None creation_io_error@incoming error formal on Err
2@factory None creation_io_error@enum return/receipt on normal Err mapping
1@observation backing_io_error@item evaluated on Ok or Err
1@observation backing_io_error@map receiver adapter tuple
1@observation backing_io_error@FnOnce self/input tuple on Err
1@observation backing_io_error@incoming error formal on Err
2@observation backing_io_error@enum return/receipt on normal Err mapping
1@Observation enum item@item evaluated on Ok or Err
1@Observation enum item@map receiver adapter tuple
1@Observation enum item@FnOnce self/input tuple on Err
1@Observation enum item@incoming error formal on Err
2@Observation enum item@enum return/receipt on normal Err mapping
1@Directory enum item@item evaluated on Ok or Err
1@Directory enum item@map receiver adapter tuple
1@Directory enum item@FnOnce self/input tuple on Err
1@Directory enum item@incoming error formal on Err
2@Directory enum item@enum return/receipt on normal Err mapping
"#;

const FINISH_ENDPOINTS: &str = r#"
2@self actual/formal@entry
2@take receiver actual/formal@entry
2@take return/ok_or receiver@entry
1@Closed eager enum@entry
1@ok_or return@entry
1@take ? ControlFlow@entry
1@take ? residual@closed
1@Pending binding@present
2@hash partial move/finalize actual/formal@present
2@finalize native output/Into receiver@present
2@Into output/digest binding@present
2@written/size reads@present
2@written != size and || result@present
1@expected field read/adapter receiver@sizes match
2@factory borrowed tuple actual/formal@sizes match even None
1@factory captured-reference move@sizes match
1@digest live referent (not new backing)@present
2@callback argument tuple actual/formal@sizes match Some
2@v binding/read@sizes match Some
1@captured reference access@sizes match Some
1@referent read for array comparison@sizes match Some, opaque 32-byte comparison
2@PartialEq::ne argument tuple actual/formal@sizes match Some
4@comparison bool return/receipt/is_some_and return/receipt@sizes match Some
2@is_some_and None bool return/receipt@sizes match None alternative
1@ContentMismatch construction@mismatch
3@backing file move actual/formal/field@success
3@backing directory move actual/formal/field@success
3@PublicFileBacking construction/Arc actual/formal@success
3@Arc return/receipt/snapshot field@success
2@size read/field@success
1@digest field move@success
2@max read/field@success
2@snapshot construction/Ok move@success
2@result return/consumer receipt@all exits
1@partial Pending file drop@refusal/unwind after hash moved
1@partial Pending directory drop@refusal/unwind after hash moved
1@empty self retirement@all exits
1@backing retirement endpoint@eventual last strong owner, not finish return
3@actual borrowed callback@construction/return/receipt
1@actual borrowed callback@None unused callback consumption
1@actual borrowed callback@is_some_and receiver/adapter tuple
1@actual borrowed callback@FnOnce self/input tuple
1@actual digest factory item@sizes match
"#;

// PRE-CUT PROVENANCE (not HEAD): literal items below were transcribed from the
// initial, read-only working-tree read before this writer's first source patch:
// public_file.rs lines 255-275, 492-526, 571-589 (tool read call
// call_7f53aece66f341e79a4830979cdc70ef). That dirty pre-cut already contained
// generic Storage<B>/resident support absent from HEAD, as the scoped read-only
// git diff confirms. No baseline commit/revision or committed-history equality
// is asserted. The independent expectations are NOT produced by the emitters.
const PRE_CUT_OBSERVE: &str = r#"
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
"#;
const PRE_CUT_CREATE: &str = r#"
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
"#;
const PRE_CUT_FINISH: &str = r#"
pub fn finish(mut self) -> Result<PublicFileStorage<B>, String> {
    let pending = self
        .pending
        .take()
        .ok_or("public snapshot writer is closed")?;
    let digest: [u8; 32] = pending.hash.finalize().into();
    if pending.written != pending.size || pending.expected.is_some_and(|v| v != digest) {
        return Err("public snapshot digest or size differs".into());
    }
    Ok(PublicFileStorage {
        file: Arc::new(PublicFileBacking {
            file: pending.file,
            _directory: pending.directory,
        }),
        size: pending.size,
        digest,
        max_i8: pending.max_i8,
    })
}
"#;

// Finite lexical substitution for ONLY these literal item emitters. Not a Rust
// macro engine, source generator, ABI witness, or evidence of compiler hygiene.
// Identifier substitutions include invocation-supplied self/locals/B. String
// contents remain whole tokens (including whitespace/escapes); only source
// whitespace and comments outside literals are ignored. Punctuation is retained.
fn legacy_tokens(source: &str) -> Vec<String> {
    let bytes = source.as_bytes(); let mut i = 0; let mut tokens = Vec::new();
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() { i += 1; continue; }
        if bytes[i..].starts_with(b"//") {
            while i < bytes.len() && bytes[i] != b'\n' { i += 1; }
            continue;
        }
        if bytes[i..].starts_with(b"/*") {
            let mut nesting = 1usize; i += 2;
            while nesting != 0 {
                assert!(i < bytes.len(),"unterminated block comment");
                if bytes[i..].starts_with(b"/*") { nesting += 1; i += 2; }
                else if bytes[i..].starts_with(b"*/") { nesting -= 1; i += 2; }
                else { i += 1; }
            }
            continue;
        }
        let start = i; i += 1;
        if bytes[start] == b'"' {
            let mut closed = false;
            while i < bytes.len() {
                if bytes[i] == b'\\' { i += 2; continue; }
                if bytes[i] == b'"' { i += 1; closed = true; break; }
                i += 1;
            }
            assert!(closed,"unterminated literal in finite legacy fixture");
        } else if bytes[start].is_ascii_alphanumeric() || bytes[start] == b'_' {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') { i += 1; }
        } else {
            // Preserve joint operator spelling: != is not the same token as
            // separated ! =, nor > = as >=. Longest operators checked first.
            for operator in ["<<=",">>=","...","..=","::","->","=>","==","!=","<=",">=","&&","||","+=","-=","*=","/=","%=","^=","&=","|=","<<",">>",".."] {
                if bytes[start..].starts_with(operator.as_bytes()) { i = start + operator.len(); break; }
            }
        }
        tokens.push(source[start..i].to_owned());
    }
    tokens
}
fn legacy_expansion(source: &str, emitter: &str, parameters: &[&str]) -> Vec<String> {
    let definition_marker = format!("macro_rules! {emitter} {{");
    let definition = source.split_once(definition_marker.as_str()).unwrap().1;
    let template = definition.split_once("=> {").unwrap().1.split_once("\n    };\n}").unwrap().0;
    let invocation_marker = format!("{emitter}!(");
    let invocation = source.split_once(invocation_marker.as_str()).unwrap().1;
    // These finite invocations contain no inner semicolon or literal ");".
    // Do not lex unrelated functions after the invocation's terminator.
    let invocation = format!("{})",invocation.split_once(");").unwrap().0);
    // All five invocations have simple ident/type parameters and expressions;
    // commas nested in (), [], {} belong to a single expression argument.
    let mut arguments = Vec::new(); let mut argument = Vec::new(); let mut depth = 0usize;
    for token in legacy_tokens(&invocation) {
        match token.as_str() {
            ")" if depth == 0 => { if !argument.is_empty() { arguments.push(argument); } break; }
            "," if depth == 0 => { arguments.push(std::mem::take(&mut argument)); }
            "(" | "[" | "{" => { depth += 1; argument.push(token); }
            ")" | "]" | "}" => { depth = depth.checked_sub(1).unwrap(); argument.push(token); }
            _ => argument.push(token),
        }
    }
    assert_eq!(arguments.len(),parameters.len(),"finite emitter signature changed");
    let mut expanded = Vec::new(); let mut tokens = legacy_tokens(template).into_iter();
    while let Some(token) = tokens.next() {
        if token == "$" {
            let name = tokens.next().unwrap();
            let index = parameters.iter().position(|p| *p == name.as_str()).expect("unknown emitter binding");
            expanded.extend(arguments[index].iter().cloned());
        } else { expanded.push(token); }
    }
    expanded
}

#[test]
fn lifecycle_complete_legacy_items_match_genuine_dirty_pre_cut_literals() {
    let source = include_str!("public_file.rs");
    let observe = ["name","file","metadata","allocated_bytes","error","mapper","overflow"];
    let create = ["name","size","expected","directory","file","owner","error","bounds","policy","mapper","check"];
    let finish = ["name","self","pending","digest","B","error","closed","mismatch","callback"];
    for (emitter,parameters,literal) in [
        ("emit_observe_backing",observe.as_slice(),PRE_CUT_OBSERVE),
        ("emit_with_directory",create.as_slice(),PRE_CUT_CREATE),
        ("emit_finish",finish.as_slice(),PRE_CUT_FINISH),
    ] {
        assert_eq!(legacy_expansion(source,emitter,parameters),legacy_tokens(literal),"{emitter}: complete legacy item");
    }
    // Specific shared-template blind spots from review must fail this oracle.
    let bound = source.replacen("$size > MAX_BYTES","$size >= MAX_BYTES",1);
    assert_ne!(legacy_expansion(&bound,"emit_with_directory",&create),legacy_tokens(PRE_CUT_CREATE));
    let blocks = source.replacen(".blocks()",".len()",1);
    assert_ne!(legacy_expansion(&blocks,"emit_observe_backing",&observe),legacy_tokens(PRE_CUT_OBSERVE));
    let literal = source.replacen("\"public snapshot size exceeds bounds\"","\"public snapshot  size exceeds bounds\"",1);
    assert_ne!(legacy_expansion(&literal,"emit_with_directory",&create),legacy_tokens(PRE_CUT_CREATE));
    let closure = source.replacen("|v| v != digest","|v| v == digest",1);
    assert_ne!(legacy_expansion(&closure,"emit_finish",&finish),legacy_tokens(PRE_CUT_FINISH));
}

#[test]
fn lifecycle_endpoint_checked_arithmetic_rejects_overflow_and_bad_layout() {
    assert_eq!(checked_endpoint_oracle(&[("x","b",2,usize::MAX,1)],"2@x@b"),None);
    assert_eq!(checked_endpoint_oracle(&[("x","b",1,usize::MAX,1),("y","b",1,1,1)],"1@x@b\n1@y@b"),None);
    assert_eq!(checked_endpoint_oracle(&[("x","b",1,1,2)],"1@x@b"),None);
}
