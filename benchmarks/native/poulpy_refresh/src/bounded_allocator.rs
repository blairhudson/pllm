//! Qualification-only anonymous-mapping allocator; no backend dependency.
//!
//! Every request owns one mapping. Charge includes its entire page-rounded
//! extent (header and alignment slack included) plus a bookkeeping ticket.
//! Successful munmap is the only event that releases a mapped allocation's
//! charge. Realloc reserves the new mapping while the old mapping is charged.
//!
//! The ticket and external runtime/page-table reserves are OPERATIONAL BUDGETS,
//! not a proved bound on XNU kernel memory or on non-Rust runtime allocations.
//! Thus the enforced charge ceiling alone is not whole-process HE admission.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("This qualification allocator is reviewed only for aarch64-apple-darwin");

use std::alloc::{GlobalAlloc, Layout};
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub const PAGE_BYTES: usize = 16_384;
pub const OWNER_CEILING: usize = 8 << 30;
pub const GUARD_RESERVE: usize = 64 << 20;
pub const RUNTIME_RESERVE: usize = 64 << 20;
pub const PAGE_TABLE_RESERVE: usize = 16 << 20;
pub const BOOKKEEPING_TICKET: usize = 2 * PAGE_BYTES;
pub const CHARGE_CEILING: usize = OWNER_CEILING - GUARD_RESERVE - RUNTIME_RESERVE - PAGE_TABLE_RESERVE;
pub const MAX_MAPPINGS: usize = 4096;
const MAGIC: usize = 0x504c_4c4d_4d41_5031;
const MAP_FAILED: *mut u8 = usize::MAX as *mut u8;

// Values/signatures checked against the installed macOS SDK sys/mman.h and
// unistd.h. No libc crate, malloc, formatting, unwinding, or heap-owned registry.
#[link(name = "System")]
extern "C" {
    fn getpagesize() -> i32;
    fn mmap(address: *mut c_void, length: usize, protection: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn munmap(address: *mut c_void, length: usize) -> i32;
    fn abort() -> !;
}

/// An allocation-free mapping interface, also used for deterministic fault tests.
///
/// # Safety
/// Implementations must not allocate, unwind, or reenter this allocator. A
/// successful map must return a distinct read/write anonymous zero-filled extent
/// of `length` bytes, aligned to page_size(). Only usize::MAX denotes failure.
/// Successful unmap must remove that complete extent before returning zero.
pub unsafe trait Mapper: Sync {
    fn page_size(&self) -> usize;
    unsafe fn map(&self, length: usize) -> *mut u8;
    unsafe fn unmap(&self, address: *mut u8, length: usize) -> i32;
}

pub struct SystemMapper;

// SAFETY: these direct OS calls obey the interface; no Rust heap operation is
// performed by this wrapper. Non-Rust runtime costs are separately budgeted.
unsafe impl Mapper for SystemMapper {
    fn page_size(&self) -> usize {
        let page = unsafe { getpagesize() };
        if page > 0 { page as usize } else { 0 }
    }

    unsafe fn map(&self, length: usize) -> *mut u8 {
        // PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, no fd, zero offset.
        unsafe { mmap(ptr::null_mut(), length, 0x01 | 0x02, 0x0002 | 0x1000, -1, 0).cast() }
    }

    unsafe fn unmap(&self, address: *mut u8, length: usize) -> i32 {
        unsafe { munmap(address.cast(), length) }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Header {
    base: *mut u8,
    mapped: usize,
    charge: usize,
    requested: usize,
    alignment: usize,
    offset: usize,
    magic: usize,
}

pub const HEADER_BYTES: usize = std::mem::size_of::<Header>();
const _: () = assert!(HEADER_BYTES == 64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub mapped: usize,
    pub charge: usize,
    pub alignment: usize,
    pub maximum_offset: usize,
}

fn round_up(value: usize, alignment: usize) -> Option<usize> {
    value.checked_add(alignment.checked_sub(1)?).map(|v| v & !(alignment - 1))
}

/// Checked geometry only; does not reserve memory or call the OS.
pub fn geometry(layout: Layout, page: usize, bookkeeping: usize) -> Option<Geometry> {
    if layout.size() == 0 || !page.is_power_of_two() || page < HEADER_BYTES {
        return None;
    }
    let alignment = layout.align().max(std::mem::align_of::<Header>());
    // A page-aligned base is also aligned to every smaller power of two. For
    // larger alignment, the next aligned point is at most `alignment` away;
    // the gap is a positive multiple of page and therefore fits the header.
    let maximum_offset = if alignment > page { alignment } else { round_up(HEADER_BYTES, alignment)? };
    let mapped = round_up(maximum_offset.checked_add(layout.size())?, page)?;
    if mapped > isize::MAX as usize {
        return None;
    }
    Some(Geometry { mapped, charge: mapped.checked_add(bookkeeping)?, alignment, maximum_offset })
}

#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    /// Includes in-flight reservations and bookkeeping tickets.
    pub charged: usize,
    pub peak_charged: usize,
    /// Successful mappings, including mappings leaked on failed munmap.
    pub mapped: usize,
    /// Includes in-flight map reservations and failed-unmap extents.
    pub mappings: usize,
    pub denied: usize,
    pub map_failures: usize,
    pub unmap_failures: usize,
    pub poisoned: bool,
}

pub struct BoundedAllocator<M: Mapper = SystemMapper> {
    mapper: M,
    ceiling: usize,
    max_mappings: usize,
    bookkeeping: usize,
    charged: AtomicUsize,
    peak: AtomicUsize,
    mapped: AtomicUsize,
    mappings: AtomicUsize,
    denied: AtomicUsize,
    map_failures: AtomicUsize,
    unmap_failures: AtomicUsize,
    poisoned: AtomicBool,
}

impl BoundedAllocator<SystemMapper> {
    pub const fn qualification() -> Self {
        Self::new(SystemMapper, CHARGE_CEILING, MAX_MAPPINGS, BOOKKEEPING_TICKET)
    }
}

impl<M: Mapper> BoundedAllocator<M> {
    /// Smaller ceilings are useful for isolated tests. Production qualification
    /// installs qualification(), with no environment or command-line override.
    pub const fn new(mapper: M, ceiling: usize, max_mappings: usize, bookkeeping: usize) -> Self {
        Self {
            mapper, ceiling, max_mappings, bookkeeping,
            charged: AtomicUsize::new(0), peak: AtomicUsize::new(0),
            mapped: AtomicUsize::new(0), mappings: AtomicUsize::new(0),
            denied: AtomicUsize::new(0), map_failures: AtomicUsize::new(0),
            unmap_failures: AtomicUsize::new(0), poisoned: AtomicBool::new(false),
        }
    }

    /// Individual atomic observations, not a coherent concurrent RSS snapshot.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            charged: self.charged.load(Ordering::Acquire), peak_charged: self.peak.load(Ordering::Acquire),
            mapped: self.mapped.load(Ordering::Acquire), mappings: self.mappings.load(Ordering::Acquire),
            denied: self.denied.load(Ordering::Acquire), map_failures: self.map_failures.load(Ordering::Acquire),
            unmap_failures: self.unmap_failures.load(Ordering::Acquire), poisoned: self.poisoned.load(Ordering::Acquire),
        }
    }

    fn deny(&self) -> *mut u8 {
        increment(&self.denied);
        ptr::null_mut()
    }

    fn reserve(&self, charge: usize) -> bool {
        if self.poisoned.load(Ordering::Acquire) {
            return false;
        }
        if self.mappings.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            n.checked_add(1).filter(|&new| new <= self.max_mappings)
        }).is_err() {
            return false;
        }
        let previous = self.charged.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            n.checked_add(charge).filter(|&new| new <= self.ceiling)
        });
        match previous {
            Ok(old) => {
                // Success of the checked update proves this addition safe.
                self.peak.fetch_max(old + charge, Ordering::AcqRel);
                // Another thread may poison while this reservation is pending.
                if self.poisoned.load(Ordering::Acquire) {
                    self.rollback(charge);
                    return false;
                }
                true
            }
            Err(_) => {
                self.mappings.fetch_sub(1, Ordering::AcqRel);
                false
            }
        }
    }

    fn rollback(&self, charge: usize) {
        self.charged.fetch_sub(charge, Ordering::AcqRel);
        self.mappings.fetch_sub(1, Ordering::AcqRel);
    }

    unsafe fn release(&self, base: *mut u8, mapped: usize, charge: usize) {
        if unsafe { self.mapper.unmap(base, mapped) } == 0 {
            self.mapped.fetch_sub(mapped, Ordering::AcqRel);
            self.rollback(charge);
        } else {
            // Never subtract a charge when the mapping may still exist. Do not
            // allocate a diagnostic or unwind from GlobalAlloc. The run must be
            // rejected; further allocation attempts return null.
            increment(&self.unmap_failures);
            self.poisoned.store(true, Ordering::Release);
        }
    }

    unsafe fn allocate(&self, layout: Layout) -> *mut u8 {
        let page = self.mapper.page_size();
        // Freeze the Apple-M page geometry; other targets need their own review.
        if page != PAGE_BYTES {
            return self.deny();
        }
        let Some(g) = geometry(layout, page, self.bookkeeping) else { return self.deny(); };
        if !self.reserve(g.charge) {
            return self.deny();
        }
        let base = unsafe { self.mapper.map(g.mapped) };
        if base == MAP_FAILED {
            increment(&self.map_failures);
            self.rollback(g.charge);
            return self.deny();
        }
        self.mapped.fetch_add(g.mapped, Ordering::AcqRel);
        let address = base as usize;
        let offset = address.checked_add(HEADER_BYTES)
            .and_then(|p| round_up(p, g.alignment))
            .and_then(|p| p.checked_sub(address));
        let valid = !base.is_null() && address % page == 0
            && address.checked_add(g.mapped).is_some()
            && offset.is_some_and(|o| o >= HEADER_BYTES && o <= g.maximum_offset
                && o.checked_add(layout.size()).is_some_and(|end| end <= g.mapped));
        if !valid {
            unsafe { self.release(base, g.mapped, g.charge); }
            return self.deny();
        }
        // `valid` established Some(offset) and bounds within this mapped object.
        let offset = match offset { Some(value) => value, None => unsafe { abort() } };
        let user = unsafe { base.add(offset) };
        let header = Header {
            base, mapped: g.mapped, charge: g.charge, requested: layout.size(),
            alignment: layout.align(), offset, magic: MAGIC,
        };
        unsafe { user.sub(HEADER_BYTES).cast::<Header>().write(header); }
        user
    }

    unsafe fn header(&self, pointer: *mut u8, layout: Layout) -> Header {
        // As required by GlobalAlloc, pointer must be an allocation of THIS
        // allocator with this original nonzero layout, and not already freed.
        // Checks diagnose internal mismatches; they do not make invalid pointers
        // safe to dereference. abort never unwinds and performs no Rust allocation.
        if pointer.is_null() || layout.size() == 0 { unsafe { abort() } }
        let header = unsafe { pointer.sub(HEADER_BYTES).cast::<Header>().read() };
        if header.magic != MAGIC || header.requested != layout.size() || header.alignment != layout.align()
            || header.offset < HEADER_BYTES || header.offset > header.mapped
            || header.base as usize != (pointer as usize).wrapping_sub(header.offset) {
            unsafe { abort() }
        }
        header
    }
}

fn increment(counter: &AtomicUsize) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_add(1)));
}

// SAFETY: mappings are disjoint, zero-filled, layout-aligned and bounded above;
// all budget mutations are atomic. No method unwinds or recursively allocates.
// Callers still owe the ordinary GlobalAlloc pointer/layout/lifetime contract.
unsafe impl<M: Mapper> GlobalAlloc for BoundedAllocator<M> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { self.allocate(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Anonymous mmap zeroes the user range. Header writes precede that range.
        unsafe { self.allocate(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let h = unsafe { self.header(pointer, layout) };
        unsafe { self.release(h.base, h.mapped, h.charge); }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = unsafe { self.header(pointer, layout) };
        // GlobalAlloc requires positive new_size; defensively refuse zero
        // without freeing the old allocation. Failed growth/shrink also keeps it.
        if new_size == 0 { return self.deny(); }
        let Ok(new_layout) = Layout::from_size_align(new_size, layout.align()) else { return self.deny(); };
        if new_size == layout.size() { return pointer; }
        let new_pointer = unsafe { self.allocate(new_layout) };
        if new_pointer.is_null() { return new_pointer; }
        unsafe { ptr::copy_nonoverlapping(pointer, new_pointer, layout.size().min(new_size)); }
        unsafe { self.dealloc(pointer, layout); }
        new_pointer
    }
}

#[cfg(test)]
#[path = "../tests/allocator/unit.rs"]
mod tests;
