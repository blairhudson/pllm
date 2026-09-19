#![deny(unsafe_op_in_unsafe_fn)]

use pllm_plugin_api::{
    ffi_guard, PllmByteSlice, PllmCancelFn, PllmCreateFn, PllmDescribeFn, PllmDestroyFn,
    PllmEmitEventFn, PllmExecuteRegionFn, PllmHandle, PllmHostVTable, PllmIsCancelledFn,
    PllmLastErrorFn, PllmOwnedBuffer, PllmPluginVTable, PllmPollFn, PllmReleaseBufferFn,
    PllmStatus, PllmSubmitRegionFn,
};
use std::ptr;
use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct State {
    generation: u64,
    live: bool,
    host: PllmHandle,
    emit_event: Option<PllmEmitEventFn>,
    is_cancelled: Option<PllmIsCancelledFn>,
    operation_generation: u64,
    pending: Option<(PllmHandle, Vec<u8>)>,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::default()))
}

fn valid(state: &State, handle: PllmHandle) -> bool {
    state.live && handle == PllmHandle::new(1, state.generation)
}

fn output(handle: PllmHandle, mut bytes: Vec<u8>, target: *mut PllmOwnedBuffer) -> PllmStatus {
    if target.is_null() {
        return PllmStatus::INVALID_ARGUMENT;
    }
    let buffer = if bytes.is_empty() {
        PllmOwnedBuffer {
            data: ptr::null_mut(),
            len: 0,
            capacity: 0,
            owner: handle,
        }
    } else {
        let len = match u64::try_from(bytes.len()) {
            Ok(len) => len,
            Err(_) => return PllmStatus::RESOURCE_EXHAUSTED,
        };
        let capacity = match u64::try_from(bytes.capacity()) {
            Ok(capacity) => capacity,
            Err(_) => return PllmStatus::RESOURCE_EXHAUSTED,
        };
        let buffer = PllmOwnedBuffer {
            data: bytes.as_mut_ptr(),
            len,
            capacity,
            owner: handle,
        };
        std::mem::forget(bytes);
        buffer
    };
    unsafe { target.write(buffer) };
    PllmStatus::OK
}

unsafe extern "C" fn create(
    host: *const PllmHostVTable,
    configuration: PllmByteSlice,
    instance: *mut PllmHandle,
) -> PllmStatus {
    ffi_guard(|| {
        if host.is_null() || instance.is_null() || !configuration.is_well_formed() {
            return PllmStatus::INVALID_ARGUMENT;
        }
        let host = unsafe { &*host };
        if !host.is_compatible() {
            return PllmStatus::UNSUPPORTED;
        }
        let mut state = match state().lock() {
            Ok(state) => state,
            Err(_) => return PllmStatus::INTERNAL,
        };
        if state.live {
            return PllmStatus::RESOURCE_EXHAUSTED;
        }
        state.generation = state.generation.wrapping_add(1).max(1);
        state.live = true;
        state.host = host.host;
        state.emit_event = host.emit_event;
        state.is_cancelled = host.is_cancelled;
        unsafe { instance.write(PllmHandle::new(1, state.generation)) };
        PllmStatus::OK
    })
}

unsafe extern "C" fn destroy(instance: PllmHandle) -> PllmStatus {
    ffi_guard(|| {
        let mut state = match state().lock() {
            Ok(state) => state,
            Err(_) => return PllmStatus::INTERNAL,
        };
        if !valid(&state, instance) {
            return PllmStatus::NOT_FOUND;
        }
        state.live = false;
        state.host = PllmHandle::INVALID;
        state.emit_event = None;
        state.is_cancelled = None;
        state.pending = None;
        PllmStatus::OK
    })
}

unsafe extern "C" fn describe(
    instance: PllmHandle,
    target: *mut PllmOwnedBuffer,
) -> PllmStatus {
    ffi_guard(|| {
        let state = match state().lock() {
            Ok(state) => state,
            Err(_) => return PllmStatus::INTERNAL,
        };
        if !valid(&state, instance) {
            return PllmStatus::NOT_FOUND;
        }
        output(
            instance,
            br#"{"schema":"pllm.native_plugin_descriptor.v1","fixture":true}"#.to_vec(),
            target,
        )
    })
}

unsafe extern "C" fn execute_region(
    instance: PllmHandle,
    request: PllmByteSlice,
    target: *mut PllmOwnedBuffer,
) -> PllmStatus {
    ffi_guard(|| {
        let request = match unsafe { request.as_slice() } {
            Some(request) => request,
            None => return PllmStatus::INVALID_ARGUMENT,
        };
        let state = match state().lock() {
            Ok(state) => state,
            Err(_) => return PllmStatus::INTERNAL,
        };
        if !valid(&state, instance) {
            return PllmStatus::NOT_FOUND;
        }
        if state
            .is_cancelled
            .map(|callback| unsafe { callback(state.host) } != 0)
            .unwrap_or(false)
        {
            return PllmStatus::CANCELLED;
        }
        if let Some(callback) = state.emit_event {
            let event = match PllmByteSlice::try_from_slice(b"executed") {
                Some(event) => event,
                None => return PllmStatus::RESOURCE_EXHAUSTED,
            };
            let status = unsafe { callback(state.host, event) };
            if !status.is_ok() {
                return status;
            }
        }
        let mut response = request.to_vec();
        response.reverse();
        output(instance, response, target)
    })
}

unsafe extern "C" fn submit_region(
    instance: PllmHandle,
    request: PllmByteSlice,
    operation: *mut PllmHandle,
) -> PllmStatus {
    ffi_guard(|| {
        if operation.is_null() {
            return PllmStatus::INVALID_ARGUMENT;
        }
        let request = match unsafe { request.as_slice() } {
            Some(request) => request,
            None => return PllmStatus::INVALID_ARGUMENT,
        };
        let mut state = match state().lock() {
            Ok(state) => state,
            Err(_) => return PllmStatus::INTERNAL,
        };
        if !valid(&state, instance) {
            return PllmStatus::NOT_FOUND;
        }
        if state.pending.is_some() {
            return PllmStatus::RESOURCE_EXHAUSTED;
        }
        if state
            .is_cancelled
            .map(|callback| unsafe { callback(state.host) } != 0)
            .unwrap_or(false)
        {
            return PllmStatus::CANCELLED;
        }
        state.operation_generation = state.operation_generation.wrapping_add(1).max(1);
        let handle = PllmHandle::new(2, state.operation_generation);
        let mut response = request.to_vec();
        response.reverse();
        state.pending = Some((handle, response));
        unsafe { operation.write(handle) };
        PllmStatus::OK
    })
}

unsafe extern "C" fn poll(
    instance: PllmHandle,
    operation: PllmHandle,
    ready: *mut u8,
    target: *mut PllmOwnedBuffer,
) -> PllmStatus {
    ffi_guard(|| {
        if ready.is_null() || target.is_null() {
            return PllmStatus::INVALID_ARGUMENT;
        }
        let mut state = match state().lock() {
            Ok(state) => state,
            Err(_) => return PllmStatus::INTERNAL,
        };
        if !valid(&state, instance) {
            return PllmStatus::NOT_FOUND;
        }
        if state.pending.as_ref().map(|(handle, _)| *handle) != Some(operation) {
            return PllmStatus::NOT_FOUND;
        }
        let (_, response) = match state.pending.take() {
            Some(pending) => pending,
            None => return PllmStatus::NOT_FOUND,
        };
        unsafe { ready.write(1) };
        drop(state);
        output(instance, response, target)
    })
}

unsafe extern "C" fn cancel(instance: PllmHandle, operation: PllmHandle) -> PllmStatus {
    ffi_guard(|| {
        let mut state = match state().lock() {
            Ok(state) => state,
            Err(_) => return PllmStatus::INTERNAL,
        };
        if !valid(&state, instance) {
            return PllmStatus::NOT_FOUND;
        }
        if state.pending.as_ref().map(|(handle, _)| *handle) != Some(operation) {
            return PllmStatus::NOT_FOUND;
        }
        state.pending = None;
        PllmStatus::OK
    })
}

unsafe extern "C" fn release_buffer(
    instance: PllmHandle,
    buffer: PllmOwnedBuffer,
) -> PllmStatus {
    ffi_guard(|| {
        if !buffer.is_well_formed() || buffer.owner != instance {
            return PllmStatus::INVALID_ARGUMENT;
        }
        let state = match state().lock() {
            Ok(state) => state,
            Err(_) => return PllmStatus::INTERNAL,
        };
        if !valid(&state, instance) {
            return PllmStatus::NOT_FOUND;
        }
        if buffer.capacity > 0 {
            unsafe {
                drop(Vec::from_raw_parts(
                    buffer.data,
                    buffer.len as usize,
                    buffer.capacity as usize,
                ))
            };
        }
        PllmStatus::OK
    })
}

unsafe extern "C" fn last_error(
    instance: PllmHandle,
    target: *mut PllmOwnedBuffer,
) -> PllmStatus {
    ffi_guard(|| {
        let state = match state().lock() {
            Ok(state) => state,
            Err(_) => return PllmStatus::INTERNAL,
        };
        if !valid(&state, instance) {
            return PllmStatus::NOT_FOUND;
        }
        output(instance, b"fixture has no pending error".to_vec(), target)
    })
}

static VTABLE: PllmPluginVTable = PllmPluginVTable {
    header: pllm_plugin_api::PllmAbiHeader::new::<PllmPluginVTable>(),
    create: Some(create as PllmCreateFn),
    destroy: Some(destroy as PllmDestroyFn),
    describe: Some(describe as PllmDescribeFn),
    execute_region: Some(execute_region as PllmExecuteRegionFn),
    submit_region: Some(submit_region as PllmSubmitRegionFn),
    poll: Some(poll as PllmPollFn),
    cancel: Some(cancel as PllmCancelFn),
    release_buffer: Some(release_buffer as PllmReleaseBufferFn),
    last_error: Some(last_error as PllmLastErrorFn),
    reserved: [ptr::null_mut(); 8],
};

#[no_mangle]
pub unsafe extern "C" fn pllm_plugin_v1(
    host: *const PllmHostVTable,
    plugin: *mut *const PllmPluginVTable,
) -> PllmStatus {
    ffi_guard(|| {
        if host.is_null() || plugin.is_null() {
            return PllmStatus::INVALID_ARGUMENT;
        }
        if !unsafe { &*host }.is_compatible() || !VTABLE.is_compatible() {
            return PllmStatus::UNSUPPORTED;
        }
        unsafe { plugin.write(&VTABLE) };
        PllmStatus::OK
    })
}
