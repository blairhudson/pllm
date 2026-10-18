//! macOS reclaimable physical memory and process-lifetime pressure observation.
//!
//! Mach's free count includes speculative pages, which also occur in the
//! file-backed count. Purgeable and MADV_FREE_REUSABLE pages belong to internal
//! (anonymous) objects. Their aggregate counters do not report overlap, so credit
//! only the larger count. Other inactive anonymous memory, compressor capacity,
//! and swap add no capacity.

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct HostMemory {
    pub total: u64,
    pub available: u64,
    pub free: u64,
    pub cached: u64,
    pub purgeable: u64,
    pub reusable: u64,
    pub pressure: &'static str,
}

#[cfg(any(target_os = "macos", test))]
#[derive(Clone, Copy)]
struct PageCounts {
    free: u64,
    speculative: u64,
    cached: u64,
    purgeable: u64,
    reusable: u64,
}

#[cfg(any(target_os = "macos", test))]
fn pressure_name(flags: usize) -> Result<&'static str, &'static str> {
    if flags & 4 != 0 {
        Ok("critical")
    } else if flags & 2 != 0 {
        Ok("warning")
    } else if flags == 1 {
        Ok("normal")
    } else {
        Err("macOS memory pressure is unavailable")
    }
}

#[cfg(any(target_os = "macos", test))]
fn pressure_event(flags: usize, current: Result<usize, &'static str>) -> usize {
    // GCD may coalesce a warning and its recovery. Keeping both bits forever
    // would block later launches even after the host returns to normal.
    if flags & 1 != 0 {
        if let Ok(level) = current {
            if pressure_name(level).is_ok() {
                return level;
            }
        }
    }
    flags
}

#[cfg(any(target_os = "macos", test))]
fn from_counts(
    total: u64,
    page_size: u64,
    counts: PageCounts,
    pressure: usize,
) -> Result<HostMemory, &'static str> {
    let invalid = "invalid macOS physical memory counters";
    if total == 0 || !page_size.is_power_of_two() || !(4096..=65536).contains(&page_size) {
        return Err(invalid);
    }
    let PageCounts {
        free,
        speculative,
        cached,
        purgeable,
        reusable,
    } = counts;
    let free = free.checked_sub(speculative).ok_or(invalid)?;
    let available = free
        .checked_add(cached)
        .and_then(|n| n.checked_add(purgeable.max(reusable)))
        .and_then(|n| n.checked_mul(page_size))
        .ok_or(invalid)?;
    if available > total {
        return Err(invalid);
    }
    Ok(HostMemory {
        total,
        available,
        free: free * page_size,
        cached: cached * page_size,
        purgeable: purgeable * page_size,
        reusable: reusable * page_size,
        pressure: pressure_name(pressure)?,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn sample() -> Result<HostMemory, &'static str> {
    Err("Mach memory observation requires macOS")
}

#[cfg(target_os = "macos")]
pub fn sample() -> Result<HostMemory, &'static str> {
    macos::sample()
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{from_counts, HostMemory, PageCounts};
    use std::ffi::{c_char, c_int, c_void};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    };

    // Stable HOST_VM_INFO64 rev1 prefix (38 natural_t words). Later kernel
    // revisions can append fields without changing this explicitly sized query.
    #[repr(C)]
    #[derive(Default)]
    pub(super) struct VmStatistics {
        free: u32,
        active: u32,
        inactive: u32,
        wired: u32,
        counters: [u64; 9],
        purgeable: u32,
        speculative: u32,
        compressor_counters: [u64; 4],
        compressor: u32,
        throttled: u32,
        external: u32,
        internal: u32,
        uncompressed: u64,
    }

    #[link(name = "System")]
    unsafe extern "C" {
        fn mach_host_self() -> u32;
        static mach_task_self_: u32;
        fn mach_port_deallocate(task: u32, name: u32) -> c_int;
        fn host_page_size(host: u32, size: *mut usize) -> c_int;
        fn host_statistics64(host: u32, flavor: c_int, out: *mut c_int, count: *mut u32) -> c_int;
        fn sysctlbyname(
            name: *const c_char,
            out: *mut c_void,
            size: *mut usize,
            input: *const c_void,
            input_size: usize,
        ) -> c_int;
        static _dispatch_source_type_memorypressure: u8;
        fn dispatch_get_global_queue(identifier: isize, flags: usize) -> *mut c_void;
        fn dispatch_source_create(
            kind: *const u8,
            handle: usize,
            mask: usize,
            queue: *mut c_void,
        ) -> *mut c_void;
        fn dispatch_set_context(object: *mut c_void, context: *mut c_void);
        fn dispatch_source_set_event_handler_f(
            source: *mut c_void,
            handler: unsafe extern "C" fn(*mut c_void),
        );
        fn dispatch_source_get_data(source: *mut c_void) -> usize;
        fn dispatch_resume(object: *mut c_void);
    }

    static SOURCE: OnceLock<Result<(), &'static str>> = OnceLock::new();
    static EVENTS: AtomicUsize = AtomicUsize::new(1);

    unsafe extern "C" fn event(source: *mut c_void) {
        // Only native telemetry/atomics run here; no interpreter, allocation,
        // model owner, or callback lifetime can be retained here.
        let flags = unsafe { dispatch_source_get_data(source) };
        EVENTS.store(
            super::pressure_event(
                flags,
                sysctl::<u32>(b"kern.memorystatus_vm_pressure_level\0").map(|n| n as usize),
            ),
            Ordering::Release,
        );
    }

    fn monitor() -> Result<(), &'static str> {
        *SOURCE.get_or_init(|| {
            // One bounded source intentionally lives until process exit. Its
            // callback and context remain valid for that entire lifetime.
            let source = unsafe {
                dispatch_source_create(
                    &raw const _dispatch_source_type_memorypressure,
                    0,
                    1 | 2 | 4,
                    dispatch_get_global_queue(0, 0),
                )
            };
            if source.is_null() {
                return Err("cannot create macOS memory-pressure source");
            }
            unsafe {
                dispatch_set_context(source, source);
                dispatch_source_set_event_handler_f(source, event);
                dispatch_resume(source);
            }
            Ok(())
        })
    }

    fn sysctl<T: Default>(name: &'static [u8]) -> Result<T, &'static str> {
        let mut value = T::default();
        let mut size = size_of::<T>();
        let result = unsafe {
            sysctlbyname(
                name.as_ptr().cast(),
                (&mut value as *mut T).cast(),
                &mut size,
                std::ptr::null(),
                0,
            )
        };
        if result != 0 || size != size_of::<T>() {
            return Err("cannot query macOS memory state");
        }
        Ok(value)
    }

    pub(super) fn sample() -> Result<HostMemory, &'static str> {
        monitor()?;
        let total = sysctl::<u64>(b"hw.memsize\0")?;
        // Seed/current-state query prevents missing pressure already elevated
        // before source registration; GCD supplies asynchronous transitions.
        let current = sysctl::<u32>(b"kern.memorystatus_vm_pressure_level\0")? as usize;
        super::pressure_name(current)?;
        let mut stats = VmStatistics::default();
        let mut count = (size_of::<VmStatistics>() / size_of::<u32>()) as u32;
        let mut page_size = 0;
        let (page_result, result) = unsafe {
            let host = mach_host_self();
            let page_result = host_page_size(host, &mut page_size);
            let result = host_statistics64(
                host,
                4,
                (&mut stats as *mut VmStatistics).cast(),
                &mut count,
            );
            mach_port_deallocate(mach_task_self_, host);
            (page_result, result)
        };
        if page_result != 0 || result != 0 || count != 38 {
            return Err("cannot query macOS physical memory counters");
        }
        // XNU exports this resident-page count separately from HOST_VM_INFO64.
        // vm_object_deactivate_pages marks only internal pages reusable; it
        // clears their dirty state so the kernel can discard them without swap.
        let reusable = sysctl::<u32>(b"vm.page_reusable_count\0")?;
        from_counts(
            total,
            page_size as u64,
            PageCounts {
                free: stats.free.into(),
                speculative: stats.speculative.into(),
                cached: stats.external.into(),
                purgeable: stats.purgeable.into(),
                reusable: reusable.into(),
            },
            current | EVENTS.load(Ordering::Acquire),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_reclaimable_pages_and_pressure() {
        let counts = PageCounts {
            free: 10,
            speculative: 3,
            cached: 20,
            purgeable: 5,
            reusable: 0,
        };
        let sample = from_counts(4096 * 100, 4096, counts, 1).unwrap();
        assert_eq!(sample.available, 32 * 4096);
        assert_eq!(
            sample.available,
            sample.free + sample.cached + sample.purgeable.max(sample.reusable)
        );
        assert_eq!(sample.pressure, "normal");
        assert_eq!(pressure_name(1 | 2).unwrap(), "warning");
        assert_eq!(pressure_name(1 | 2 | 4).unwrap(), "critical");
        assert!(pressure_name(0).is_err());
        assert!(from_counts(4096, 4096, counts, 1).is_err());
        assert!(from_counts(4096, 4096, PageCounts { free: 0, ..counts }, 1).is_err());
        assert!(from_counts(
            u64::MAX,
            4096,
            PageCounts {
                free: u64::MAX,
                speculative: 0,
                ..counts
            },
            1
        )
        .is_err());
    }

    #[test]
    fn reusable_allocator_pages_count_without_double_crediting_purgeable_pages() {
        let counts = PageCounts {
            free: 10,
            speculative: 3,
            cached: 20,
            purgeable: 5,
            reusable: 40,
        };
        let sample = from_counts(4096 * 100, 4096, counts, 1).unwrap();
        assert_eq!(sample.available, 67 * 4096);
        assert_eq!(sample.reusable, 40 * 4096);
        // Either counter can include pages from the other. Credit their maximum,
        // never their sum or a process's larger RSS-minus-footprint difference.
        let sample = from_counts(
            4096 * 100,
            4096,
            PageCounts {
                purgeable: 45,
                ..counts
            },
            1,
        )
        .unwrap();
        assert_eq!(sample.available, 72 * 4096);
        assert!(from_counts(4096 * 50, 4096, counts, 1).is_err());
        assert!(from_counts(
            u64::MAX,
            4096,
            PageCounts {
                reusable: u64::MAX,
                ..counts
            },
            1
        )
        .is_err());
    }

    #[test]
    fn coalesced_recovery_does_not_leave_pressure_stuck() {
        assert_eq!(
            pressure_name(pressure_event(1 | 4, Ok(1))).unwrap(),
            "normal"
        );
        assert_eq!(
            pressure_name(pressure_event(1 | 4, Ok(2))).unwrap(),
            "warning"
        );
        assert_eq!(
            pressure_name(pressure_event(1 | 4, Err("unavailable"))).unwrap(),
            "critical"
        );
        // A warning without a recovery event is still observed even if brief.
        assert_eq!(pressure_name(pressure_event(2, Ok(1))).unwrap(), "warning");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn mach_and_dispatch_live_observation() {
        assert_eq!(size_of::<macos::VmStatistics>(), 152);
        for _ in 0..3 {
            let sample = sample().unwrap();
            assert!(sample.total > 0 && sample.available <= sample.total);
            assert_eq!(
                sample.available,
                sample.free + sample.cached + sample.purgeable.max(sample.reusable)
            );
        }
    }
}
