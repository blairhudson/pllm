use libloading::{Library, Symbol};
use pllm_plugin_api::{
    PllmAbiHeader, PllmByteSlice, PllmHandle, PllmHostVTable, PllmOwnedBuffer, PllmPluginEntryFn,
    PllmPluginVTable, PllmStatus, PLLM_PLUGIN_ABI_VERSION, PLLM_PLUGIN_ENTRY_SYMBOL,
};
use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

fn events() -> &'static Mutex<Vec<Vec<u8>>> {
    static EVENTS: OnceLock<Mutex<Vec<Vec<u8>>>> = OnceLock::new();
    EVENTS.get_or_init(|| Mutex::new(Vec::new()))
}

static CANCELLED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" fn emit_event(host: PllmHandle, event: PllmByteSlice) -> PllmStatus {
    if host != PllmHandle::new(7, 1) {
        return PllmStatus::INVALID_ARGUMENT;
    }
    let event = match unsafe { event.as_slice() } {
        Some(event) => event,
        None => return PllmStatus::INVALID_ARGUMENT,
    };
    match events().lock() {
        Ok(mut events) => {
            events.push(event.to_vec());
            PllmStatus::OK
        }
        Err(_) => PllmStatus::INTERNAL,
    }
}

unsafe extern "C" fn is_cancelled(host: PllmHandle) -> u8 {
    u8::from(host == PllmHandle::new(7, 1) && CANCELLED.load(Ordering::SeqCst))
}

fn fixture_manifest() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plugin/Cargo.toml")
}

fn build_fixture(target: &Path) -> PathBuf {
    let output = Command::new(env!("CARGO"))
        .args(["build", "--quiet", "--manifest-path"])
        .arg(fixture_manifest())
        .arg("--target-dir")
        .arg(target)
        .output()
        .expect("fixture cargo build must start");
    assert!(
        output.status.success(),
        "fixture build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    target.join("debug").join(format!(
        "{DLL_PREFIX}pllm_native_plugin_fixture{DLL_SUFFIX}"
    ))
}

unsafe fn bytes(buffer: &PllmOwnedBuffer) -> Vec<u8> {
    if buffer.len == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(buffer.data, buffer.len as usize) }.to_vec()
}

#[test]
fn dynamic_fixture_negotiates_executes_releases_and_rejects_stale_handles() {
    let target = tempfile::tempdir().expect("target directory");
    let library_path = build_fixture(target.path());
    let library = unsafe { Library::new(library_path) }.expect("fixture library");
    let entry: Symbol<PllmPluginEntryFn> =
        unsafe { library.get(PLLM_PLUGIN_ENTRY_SYMBOL) }.expect("entry symbol");
    let host = PllmHostVTable {
        header: PllmAbiHeader::new::<PllmHostVTable>(),
        host: PllmHandle::new(7, 1),
        emit_event: Some(emit_event),
        is_cancelled: Some(is_cancelled),
        reserved: [ptr::null_mut(); 4],
    };
    let mut plugin = ptr::null::<PllmPluginVTable>();
    assert_eq!(unsafe { entry(&host, &mut plugin) }, PllmStatus::OK);
    assert!(!plugin.is_null());
    let plugin = unsafe { &*plugin };
    assert!(plugin.is_compatible());
    assert_eq!(plugin.header.abi_version, PLLM_PLUGIN_ABI_VERSION);

    let mut instance = PllmHandle::INVALID;
    assert_eq!(
        unsafe {
            plugin.create.expect("create")(
                &host,
                PllmByteSlice::try_from_slice(br#"{"threads":2}"#).expect("configuration slice"),
                &mut instance,
            )
        },
        PllmStatus::OK
    );
    assert!(instance.is_valid());

    let mut descriptor = PllmOwnedBuffer::EMPTY;
    assert_eq!(
        unsafe { plugin.describe.expect("describe")(instance, &mut descriptor) },
        PllmStatus::OK
    );
    let descriptor_bytes = unsafe { bytes(&descriptor) };
    assert!(descriptor_bytes.starts_with(br#"{"schema":"pllm.native_plugin_descriptor.v1""#));
    assert_eq!(
        unsafe { plugin.release_buffer.expect("release")(instance, descriptor) },
        PllmStatus::OK
    );
    let mut error = PllmOwnedBuffer::EMPTY;
    assert_eq!(
        unsafe { plugin.last_error.expect("last error")(instance, &mut error) },
        PllmStatus::OK
    );
    assert_eq!(unsafe { bytes(&error) }, b"fixture has no pending error");
    assert_eq!(
        unsafe { plugin.release_buffer.expect("release")(instance, error) },
        PllmStatus::OK
    );

    let mut output = PllmOwnedBuffer::EMPTY;
    assert_eq!(
        unsafe {
            plugin.execute_region.expect("execute")(
                instance,
                PllmByteSlice::try_from_slice(b"abc\0def").expect("request slice"),
                &mut output,
            )
        },
        PllmStatus::OK
    );
    assert_eq!(unsafe { bytes(&output) }, b"fed\0cba");
    assert_eq!(
        unsafe { plugin.release_buffer.expect("release")(instance, output) },
        PllmStatus::OK
    );
    assert_eq!(
        events().lock().expect("events").as_slice(),
        &[b"executed".to_vec()]
    );

    let mut operation = PllmHandle::INVALID;
    assert_eq!(
        unsafe {
            plugin.submit_region.expect("submit")(
                instance,
                PllmByteSlice::try_from_slice(b"async").expect("request slice"),
                &mut operation,
            )
        },
        PllmStatus::OK
    );
    assert!(operation.is_valid());
    let mut ready = 0;
    let mut async_output = PllmOwnedBuffer::EMPTY;
    assert_eq!(
        unsafe { plugin.poll.expect("poll")(instance, operation, &mut ready, &mut async_output,) },
        PllmStatus::OK
    );
    assert_eq!(ready, 1);
    assert_eq!(unsafe { bytes(&async_output) }, b"cnysa");
    assert_eq!(
        unsafe { plugin.release_buffer.expect("release")(instance, async_output) },
        PllmStatus::OK
    );

    let mut cancelled_operation = PllmHandle::INVALID;
    assert_eq!(
        unsafe {
            plugin.submit_region.expect("submit")(
                instance,
                PllmByteSlice::try_from_slice(b"drop").expect("request slice"),
                &mut cancelled_operation,
            )
        },
        PllmStatus::OK
    );
    assert_eq!(
        unsafe { plugin.cancel.expect("cancel")(instance, cancelled_operation) },
        PllmStatus::OK
    );
    let mut cancelled_ready = 0;
    let mut dropped_output = PllmOwnedBuffer::EMPTY;
    assert_eq!(
        unsafe {
            plugin.poll.expect("poll")(
                instance,
                cancelled_operation,
                &mut cancelled_ready,
                &mut dropped_output,
            )
        },
        PllmStatus::NOT_FOUND
    );

    CANCELLED.store(true, Ordering::SeqCst);
    let mut cancelled_output = PllmOwnedBuffer::EMPTY;
    assert_eq!(
        unsafe {
            plugin.execute_region.expect("execute")(
                instance,
                PllmByteSlice::try_from_slice(b"cancel").expect("request slice"),
                &mut cancelled_output,
            )
        },
        PllmStatus::CANCELLED
    );
    CANCELLED.store(false, Ordering::SeqCst);

    assert_eq!(
        unsafe { plugin.destroy.expect("destroy")(instance) },
        PllmStatus::OK
    );
    let mut next = PllmHandle::INVALID;
    assert_eq!(
        unsafe { plugin.create.expect("create")(&host, PllmByteSlice::EMPTY, &mut next) },
        PllmStatus::OK
    );
    assert_ne!(instance, next);
    let mut stale_output = PllmOwnedBuffer::EMPTY;
    assert_eq!(
        unsafe {
            plugin.execute_region.expect("execute")(
                instance,
                PllmByteSlice::try_from_slice(b"stale").expect("request slice"),
                &mut stale_output,
            )
        },
        PllmStatus::NOT_FOUND
    );
    assert_eq!(
        unsafe { plugin.destroy.expect("destroy")(next) },
        PllmStatus::OK
    );
}

#[test]
fn dynamic_fixture_rejects_incompatible_host_header() {
    let target = tempfile::tempdir().expect("target directory");
    let library = unsafe { Library::new(build_fixture(target.path())) }.expect("fixture library");
    let entry: Symbol<PllmPluginEntryFn> =
        unsafe { library.get(PLLM_PLUGIN_ENTRY_SYMBOL) }.expect("entry symbol");
    let host = PllmHostVTable {
        header: PllmAbiHeader {
            size: PllmHostVTable::minimum_size(),
            abi_version: PLLM_PLUGIN_ABI_VERSION + 1,
        },
        host: PllmHandle::new(7, 1),
        emit_event: None,
        is_cancelled: None,
        reserved: [ptr::null_mut(); 4],
    };
    let mut plugin = ptr::null();
    assert_eq!(
        unsafe { entry(&host, &mut plugin) },
        PllmStatus::UNSUPPORTED
    );
    assert!(plugin.is_null());
}
