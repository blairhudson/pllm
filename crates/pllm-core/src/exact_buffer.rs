//! Exact logical backing with initialized-prefix ownership, extracted from the
//! compiler's bounded index storage. This is not admission or budget authority.
//!
//! Successful backing has exactly the requested slice length, without reserve,
//! resizing or a finish-time copy. Allocator rounding, fragmentation, RSS, nested
//! element storage and caller state are outside this logical contract.

use std::alloc::Layout;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ptr::NonNull;

pub(crate) mod linked_rows;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Overflow,
    Allocation,
    Incomplete,
}

/// Check the array's byte extent and alignment without entering the allocator.
pub fn layout_bytes<T>(count: usize) -> Result<usize, Error> {
    Layout::array::<T>(count)
        .map(|layout| layout.size())
        .map_err(|_| Error::Overflow)
}

/// Exact array backing and an initialized prefix. Element-owned backing is not
/// included. No permits, reservations or source authenticity are conferred.
pub struct ExactBuffer<T> {
    slots: Box<[MaybeUninit<T>]>,
    initialized: usize,
}

impl<T> ExactBuffer<T> {
    /// Allocate through the standard global allocator. A returned null pointer
    /// becomes `Allocation`; an allocator may still abort instead of returning.
    /// This does not promise recovery from process OOM or allocator aborts.
    pub fn new(count: usize) -> Result<Self, Error> {
        let layout = Layout::array::<T>(count).map_err(|_| Error::Overflow)?;
        let pointer = if layout.size() == 0 {
            // Aligned and non-null even for over-aligned ZSTs. Never call alloc
            // with a zero-sized layout; Box also skips deallocation in this case.
            NonNull::<MaybeUninit<T>>::dangling()
        } else {
            // SAFETY: Layout::array validated the extent, alignment and isize
            // bound, and this branch has strictly positive allocation size.
            let raw = unsafe { allocate_nonzero(layout) }.cast::<MaybeUninit<T>>();
            NonNull::new(raw).ok_or(Error::Allocation)?
        };
        // SAFETY: positive backing comes from the global allocator, aligned for
        // T and uniquely owned, with Layout::array::<T>(count). MaybeUninit<T>
        // has the same layout as T; exact slice metadata gives Box the identical
        // deallocation layout and allocator. Zero-sized backing uses aligned
        // dangling storage and requires no deallocation. No references to live T
        // exist yet. Install ownership immediately: no callbacks, allocations or
        // panicking/fallible operations intervene after a non-null allocation.
        Ok(Self {
            slots: unsafe {
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(pointer.as_ptr(), count))
            },
            initialized: 0,
        })
    }

    /// Append one element, or return the incoming owned value unchanged if full.
    /// Earlier initialized elements remain owned by this buffer on either path.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        let Some(slot) = self.slots.get_mut(self.initialized) else {
            return Err(value);
        };
        slot.write(value);
        // A slot existed, so initialized < len <= usize::MAX. Cannot overflow.
        self.initialized += 1;
        Ok(())
    }

    /// Consume a complete buffer and transfer its backing without copy/resize.
    /// An incomplete call also consumes self, dropping its initialized prefix.
    /// Callers needing to retain a pending owner must not use this failure path.
    pub fn finish(self) -> Result<Box<[T]>, Error> {
        if self.initialized != self.slots.len() {
            return Err(Error::Incomplete);
        }
        let this = ManuallyDrop::new(self);
        // SAFETY: move backing exactly once, suppressing this owner's prefix Drop.
        let slots = unsafe { std::ptr::read(&this.slots) };
        // SAFETY: every slot is initialized. This only transfers ownership.
        Ok(unsafe { slots.assume_init() })
    }

    /// Transfer the initialized prefix and all backing into a `Vec`, without
    /// allocation, copying, cloning or initializing the unused tail.
    ///
    /// The returned length is the number of successful pushes. For non-zero-sized
    /// elements, capacity is the original requested slot count (including zero),
    /// and the allocation/pointer is unchanged. Zero-sized elements keep their
    /// initialized length but have `Vec` capacity `usize::MAX`, independently of
    /// the requested slots; their aligned dangling storage is never allocated.
    ///
    /// This transfers ownership, not persistent fixed-capacity enforcement:
    /// callers must guard subsequent pushes if growth is forbidden. Element-owned
    /// backing, allocator rounding and RSS remain outside the logical contract.
    pub fn into_initialized_vec(self) -> Vec<T> {
        let initialized = self.initialized;
        let this = ManuallyDrop::new(self);
        // SAFETY: move the uniquely owned Box exactly once. Its old owner cannot
        // run prefix Drop or field Drop, and is not accessed again. Stable boxed
        // slice into_vec transfers that same global allocation without allocation,
        // copying or running element destructors. Suppress the temporary Vec's
        // Drop before transferring its ownership; only infallible header/pointer
        // operations intervene, with no callbacks, initialization or assertions.
        let slots = unsafe { std::ptr::read(&this.slots) };
        let mut slots = ManuallyDrop::new(slots.into_vec());
        let capacity = slots.capacity();
        let pointer = slots.as_mut_ptr().cast::<T>();
        // SAFETY: MaybeUninit<T> and T have identical size/alignment. The pointer
        // comes from the actual owning Vec, not a guessed header or allocation.
        // Its actual capacity preserves the original global deallocation layout
        // and validated isize byte bound. Exactly `initialized` leading elements
        // are live T, initialized once by push; the tail stays spare capacity.
        // initialized <= original slots <= capacity, also for ZST capacity MAX.
        // Empty/ZST pointers remain non-null and aligned, with no deallocation.
        // The returned Vec alone drops the prefix and frees any positive backing;
        // both previous owners are suppressed, and no fallible operation occurs
        // before this new owner is installed.
        unsafe { Vec::from_raw_parts(pointer, initialized, capacity) }
    }
}

impl<T> Drop for ExactBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: exactly this prefix contains live T values, each initialized
        // once by push. Slice drop glue cleans the remaining prefix if one
        // destructor panics; field drop then frees the MaybeUninit backing.
        // A second panic during unwind may abort, as with ordinary slice drop.
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.slots.as_mut_ptr().cast::<T>(),
                self.initialized,
            ));
        }
    }
}

/// Private allocator seam: tests may refuse an allocation, never supply pointers
/// or alternate deallocator authority.
///
/// # Safety
/// `layout` must have nonzero size. The caller must take ownership immediately
/// on a non-null return and ultimately free with the same global layout.
unsafe fn allocate_nonzero(layout: Layout) -> *mut u8 {
    #[cfg(test)]
    if tests::refuse_allocation() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller supplies a validated, nonzero Layout.
    unsafe { std::alloc::alloc(layout) }
}

#[cfg(test)]
#[path = "exact_buffer_tests.rs"]
mod tests;
#[cfg(test)]
pub(crate) use tests::AllocationScope;
