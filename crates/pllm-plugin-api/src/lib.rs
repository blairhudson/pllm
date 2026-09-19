//! Stable C-compatible native plugin ABI contracts.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_void;
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

pub const PLLM_PLUGIN_ABI_VERSION: u32 = 1;
pub const PLLM_PLUGIN_ENTRY_SYMBOL: &[u8] = b"pllm_plugin_v1\0";

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PllmStatus(pub i32);

impl PllmStatus {
    pub const OK: Self = Self(0);
    pub const INVALID_ARGUMENT: Self = Self(1);
    pub const UNSUPPORTED: Self = Self(2);
    pub const NOT_FOUND: Self = Self(3);
    pub const RESOURCE_EXHAUSTED: Self = Self(4);
    pub const CANCELLED: Self = Self(5);
    pub const INTERNAL: Self = Self(6);
    pub const PANIC: Self = Self(7);

    pub const fn is_ok(self) -> bool {
        self.0 == Self::OK.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PllmAbiHeader {
    pub size: u32,
    pub abi_version: u32,
}

impl PllmAbiHeader {
    pub const fn new<T>() -> Self {
        Self {
            size: size_of::<T>() as u32,
            abi_version: PLLM_PLUGIN_ABI_VERSION,
        }
    }

    pub const fn supports(self, minimum_size: u32) -> bool {
        self.abi_version == PLLM_PLUGIN_ABI_VERSION && self.size >= minimum_size
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PllmHandle {
    pub slot: u64,
    pub generation: u64,
}

impl PllmHandle {
    pub const INVALID: Self = Self {
        slot: 0,
        generation: 0,
    };

    pub const fn new(slot: u64, generation: u64) -> Self {
        Self { slot, generation }
    }

    pub const fn is_valid(self) -> bool {
        self.slot != 0 && self.generation != 0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PllmByteSlice {
    pub data: *const u8,
    pub len: u64,
}

impl PllmByteSlice {
    pub const EMPTY: Self = Self {
        data: ptr::null(),
        len: 0,
    };

    pub fn try_from_slice(value: &[u8]) -> Option<Self> {
        Some(Self {
            data: value.as_ptr(),
            len: u64::try_from(value.len()).ok()?,
        })
    }

    pub const fn is_well_formed(self) -> bool {
        (self.len == 0 || !self.data.is_null()) && self.len <= usize::MAX as u64
    }

    /// # Safety
    /// `data` must remain readable for `len` bytes for the returned lifetime.
    pub unsafe fn as_slice<'a>(self) -> Option<&'a [u8]> {
        if !self.is_well_formed() {
            return None;
        }
        if self.len == 0 {
            return Some(&[]);
        }
        Some(unsafe { std::slice::from_raw_parts(self.data, self.len as usize) })
    }
}

impl Default for PllmByteSlice {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct PllmOwnedBuffer {
    pub data: *mut u8,
    pub len: u64,
    pub capacity: u64,
    pub owner: PllmHandle,
}

impl PllmOwnedBuffer {
    pub const EMPTY: Self = Self {
        data: ptr::null_mut(),
        len: 0,
        capacity: 0,
        owner: PllmHandle::INVALID,
    };

    pub const fn is_well_formed(&self) -> bool {
        self.len <= self.capacity
            && self.capacity <= usize::MAX as u64
            && ((self.capacity == 0 && self.data.is_null())
                || (self.capacity > 0 && !self.data.is_null() && self.owner.is_valid()))
    }
}

impl Default for PllmOwnedBuffer {
    fn default() -> Self {
        Self::EMPTY
    }
}

pub type PllmEmitEventFn =
    unsafe extern "C" fn(host: PllmHandle, event: PllmByteSlice) -> PllmStatus;
pub type PllmIsCancelledFn = unsafe extern "C" fn(host: PllmHandle) -> u8;

#[repr(C)]
pub struct PllmHostVTable {
    pub header: PllmAbiHeader,
    pub host: PllmHandle,
    pub emit_event: Option<PllmEmitEventFn>,
    pub is_cancelled: Option<PllmIsCancelledFn>,
    pub reserved: [*mut c_void; 4],
}

unsafe impl Send for PllmHostVTable {}
unsafe impl Sync for PllmHostVTable {}

impl PllmHostVTable {
    pub const fn minimum_size() -> u32 {
        size_of::<Self>() as u32
    }

    pub const fn is_compatible(&self) -> bool {
        self.header.supports(Self::minimum_size()) && self.host.is_valid()
    }
}

pub type PllmCreateFn = unsafe extern "C" fn(
    host: *const PllmHostVTable,
    configuration: PllmByteSlice,
    instance: *mut PllmHandle,
) -> PllmStatus;
pub type PllmDestroyFn = unsafe extern "C" fn(instance: PllmHandle) -> PllmStatus;
pub type PllmDescribeFn =
    unsafe extern "C" fn(instance: PllmHandle, output: *mut PllmOwnedBuffer) -> PllmStatus;
pub type PllmExecuteRegionFn = unsafe extern "C" fn(
    instance: PllmHandle,
    request: PllmByteSlice,
    output: *mut PllmOwnedBuffer,
) -> PllmStatus;
pub type PllmSubmitRegionFn = unsafe extern "C" fn(
    instance: PllmHandle,
    request: PllmByteSlice,
    operation: *mut PllmHandle,
) -> PllmStatus;
pub type PllmPollFn = unsafe extern "C" fn(
    instance: PllmHandle,
    operation: PllmHandle,
    ready: *mut u8,
    output: *mut PllmOwnedBuffer,
) -> PllmStatus;
pub type PllmCancelFn =
    unsafe extern "C" fn(instance: PllmHandle, operation: PllmHandle) -> PllmStatus;
pub type PllmReleaseBufferFn =
    unsafe extern "C" fn(instance: PllmHandle, buffer: PllmOwnedBuffer) -> PllmStatus;
pub type PllmLastErrorFn =
    unsafe extern "C" fn(instance: PllmHandle, output: *mut PllmOwnedBuffer) -> PllmStatus;

#[repr(C)]
pub struct PllmPluginVTable {
    pub header: PllmAbiHeader,
    pub create: Option<PllmCreateFn>,
    pub destroy: Option<PllmDestroyFn>,
    pub describe: Option<PllmDescribeFn>,
    pub execute_region: Option<PllmExecuteRegionFn>,
    pub submit_region: Option<PllmSubmitRegionFn>,
    pub poll: Option<PllmPollFn>,
    pub cancel: Option<PllmCancelFn>,
    pub release_buffer: Option<PllmReleaseBufferFn>,
    pub last_error: Option<PllmLastErrorFn>,
    pub reserved: [*mut c_void; 8],
}

unsafe impl Send for PllmPluginVTable {}
unsafe impl Sync for PllmPluginVTable {}

impl PllmPluginVTable {
    pub const fn minimum_size() -> u32 {
        size_of::<Self>() as u32
    }

    pub const fn is_compatible(&self) -> bool {
        self.header.supports(Self::minimum_size())
            && self.create.is_some()
            && self.destroy.is_some()
            && self.describe.is_some()
            && self.execute_region.is_some()
            && self.submit_region.is_some()
            && self.poll.is_some()
            && self.cancel.is_some()
            && self.release_buffer.is_some()
            && self.last_error.is_some()
    }
}

pub type PllmPluginEntryFn = unsafe extern "C" fn(
    host: *const PllmHostVTable,
    plugin: *mut *const PllmPluginVTable,
) -> PllmStatus;

pub fn ffi_guard<F>(operation: F) -> PllmStatus
where
    F: FnOnce() -> PllmStatus,
{
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or(PllmStatus::PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_and_handles_are_stable_scalars() {
        assert_eq!(size_of::<PllmStatus>(), size_of::<i32>());
        assert_eq!(size_of::<PllmHandle>(), 16);
        assert!(PllmStatus::OK.is_ok());
        assert!(!PllmStatus::INTERNAL.is_ok());
        assert!(!PllmHandle::INVALID.is_valid());
        assert!(PllmHandle::new(1, 1).is_valid());
        assert_ne!(PllmHandle::new(1, 1), PllmHandle::new(1, 2));
    }

    #[test]
    fn slices_and_buffers_reject_invalid_shapes() {
        assert!(PllmByteSlice::EMPTY.is_well_formed());
        assert!(!PllmByteSlice {
            data: ptr::null(),
            len: 1,
        }
        .is_well_formed());
        assert!(PllmOwnedBuffer::EMPTY.is_well_formed());
        assert!(!PllmOwnedBuffer {
            data: ptr::dangling_mut(),
            len: 0,
            capacity: 0,
            owner: PllmHandle::new(1, 1),
        }
        .is_well_formed());
        assert!(!PllmOwnedBuffer {
            data: ptr::null_mut(),
            len: 1,
            capacity: 1,
            owner: PllmHandle::new(1, 1),
        }
        .is_well_formed());
    }

    #[test]
    fn headers_fail_closed_on_version_or_size() {
        let header = PllmAbiHeader::new::<PllmPluginVTable>();
        assert!(header.supports(PllmPluginVTable::minimum_size()));
        assert!(!PllmAbiHeader {
            size: header.size - 1,
            abi_version: header.abi_version,
        }
        .supports(PllmPluginVTable::minimum_size()));
        assert!(!PllmAbiHeader {
            size: header.size,
            abi_version: header.abi_version + 1,
        }
        .supports(PllmPluginVTable::minimum_size()));
    }

    #[test]
    fn guard_translates_panics_to_status() {
        assert_eq!(ffi_guard(|| PllmStatus::OK), PllmStatus::OK);
        assert_eq!(ffi_guard(|| panic!("fixture panic")), PllmStatus::PANIC);
    }
}
