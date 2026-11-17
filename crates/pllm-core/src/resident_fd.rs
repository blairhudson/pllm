//! Resident descriptor custody, not source authority, funding, or activation.
//!
//! Keep the registry rooted until affirmative retirement. There is no Drop
//! close, directory back-reference, or cycle. Reader and mapping leases must
//! retain a FileLease to prevent close; descriptor close is not mapping retirement.
//!
//! [`retirement_work`] describes conditional logical source work only; it does
//! not change custody, retained-storage quotes, funding, or admission.

pub mod retirement_work;

use crate::fd_close_once::{CloseState, UnixCloseOnceFd};
use std::marker::PhantomData;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistryError {
    Bounds,
    WrongTable,
    State,
    Busy,
    Poisoned,
    Unknown,
    Allocation,
    Overflow,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AcquireError<E> {
    Registry(RegistryError),
    Factory(E),
}

/// Descriptive retained storage only, never prebirth admission authority.
/// Excludes allocator overhead, initialization-race loser Box transients,
/// construction transients, factory work, and RSS. Future whole-work funding
/// must price these separately. Arc backing models Rust 1.85.1 ArcInner, not
/// an Arc pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageQuote {
    /// Inline registry plus one potential retained mutex backend allocation,
    /// including at count zero, under the qualified profile.
    pub registry_bytes: usize,
    pub slot_backing_bytes: usize,
    pub entry_backing_bytes: usize,
    pub total_bytes: usize,
}

#[repr(C)]
struct ArcInnerModel<T> {
    strong: AtomicUsize,
    weak: AtomicUsize,
    data: T,
}

// Target selection only: Rust 1.85.1 is an external qualification, not a runtime
// compiler-version check. Other Unix mutex implementations are not qualified.
fn qualified_mutex_backing_bytes() -> Option<usize> {
    #[cfg(all(target_os = "macos", target_pointer_width = "64"))]
    { Some(std::mem::size_of::<libc::pthread_mutex_t>()) }
    #[cfg(not(all(target_os = "macos", target_pointer_width = "64")))]
    { None }
}

// On unsupported targets, zero backend bytes permit structural allocation
// validation only. That subtotal must never escape as a retained-storage quote.
fn checked_preparation_extents(
    count: usize,
) -> Result<(usize, usize, usize, usize), RegistryError> {
    let registry = std::mem::size_of::<ResidentFds>()
        .checked_add(qualified_mutex_backing_bytes().unwrap_or(0))
        .ok_or(RegistryError::Overflow)?;
    let slots = count.checked_mul(std::mem::size_of::<Slot>())
        .ok_or(RegistryError::Overflow)?;
    let entries = count
        .checked_mul(std::mem::size_of::<ArcInnerModel<Entry<UnixCloseOnceFd>>>())
        .ok_or(RegistryError::Overflow)?;
    let total = registry.checked_add(slots).and_then(|n| n.checked_add(entries))
        .ok_or(RegistryError::Overflow)?;
    Ok((registry, slots, entries, total))
}

pub struct ResidentFds {
    slots: Mutex<Vec<Slot>>,
}

type Slot = EngineSlot<UnixCloseOnceFd>;

pub struct FileLease {
    inner: Lease<UnixCloseOnceFd>,
}

impl Clone for FileLease {
    fn clone(&self) -> Self { Self { inner: self.inner.clone() } }
}

impl FileLease {
    pub fn descriptor(&self) -> Option<BorrowedFd<'_>> {
        self.inner.entry.fd.as_ref()?.descriptor()
    }
}

impl ResidentFds {
    /// All slots and Arc entries are born before any factory can run.
    /// Vec reservation is fallible; Arc::new may abort on OOM. No descriptor
    /// exists during this construction, so this is not an all-fallible claim.
    pub fn prepare(count: usize) -> Result<Self, RegistryError> {
        checked_preparation_extents(count)?;
        Ok(Self { slots: Engine::<UnixCloseOnceFd>::prepare(count)? })
    }

    /// Requested logical retained storage, qualified only for externally pinned
    /// Rust 1.85.1, macOS64, and the default Global allocator. Unknown means the
    /// target's mutex backing is unqualified; it is returned before arithmetic.
    /// No partial quote is exposed for other Unix targets; custody stays portable.
    /// This is not allocator usable size, RSS, or physical capacity.
    /// For the empty non-ZST Vec used here, try_reserve_exact(n) uses
    /// grow_exact/finish_grow/set_ptr_and_cap to store n: Global::alloc_impl
    /// returns a slice whose length is layout.size(). Count zero has zero Vec
    /// allocation, not zero whole-table heap storage; construction pushes exactly
    /// n slots, with no later growth. The inline Mutex header is already counted.
    /// Its pthread OnceBox can retain one additional 64-byte, align-8
    /// libc::pthread_mutex_t payload, priced even before the first lock and even
    /// at count zero. No lock is forced; actual_capacity's first lock can allocate
    /// that payload. Together with the pinned ArcInner model, this qualifies
    /// requested logical backing sizes only under those implementation assumptions.
    /// Future toolchains are not implicitly qualified. Observing capacity after
    /// allocation does not establish prebirth funding or admission authority.
    pub fn quote(count: usize) -> Result<StorageQuote, RegistryError> {
        qualified_mutex_backing_bytes().ok_or(RegistryError::Unknown)?;
        let (registry_bytes, slot_backing_bytes, entry_backing_bytes, total_bytes) =
            checked_preparation_extents(count)?;
        Ok(StorageQuote { registry_bytes, slot_backing_bytes, entry_backing_bytes, total_bytes })
    }

    /// Observed Vec capacity, not a prebirth storage authorization or an implicit
    /// qualification of unsupported future toolchains for the static quote.
    pub fn actual_capacity(&self) -> Result<usize, RegistryError> {
        Ok(self.slots.lock().map_err(|_| RegistryError::Poisoned)?.capacity())
    }

    pub fn acquire_with<E>(
        &self, index: usize, factory: impl FnOnce() -> Result<OwnedFd, E>,
    ) -> Result<FileLease, AcquireError<E>> {
        Engine::acquire(&self.slots, index, factory).map(|inner| FileLease { inner })
    }

    pub fn retire(&self, lease: FileLease) -> Result<(), RegistryError> {
        Engine::retire(&self.slots, lease.inner)
    }

    /// Retryable retirement: refusal preserves the caller's exact lease.
    /// An absent lease is not a retirement acknowledgement. Once unique close
    /// begins, the caller is empty even if close fails or unwinds.
    pub fn try_retire(&self, lease: &mut Option<FileLease>) -> Result<(), RegistryError> {
        Engine::try_retire(&self.slots, lease)
    }
}

// Private owner abstraction: fake owners exercise precisely the same engine
// without manufacturing descriptor numbers or calling native descriptor APIs.
trait Owner: Sized {
    type Incoming;
    fn new() -> Self;
    fn vacant(&self) -> bool;
    fn adopt_prepared(&mut self, incoming: Self::Incoming);
    fn close(&mut self) -> Result<(), ()>;
    #[cfg(test)]
    fn shared_retirement(&self) {}
}

impl Owner for UnixCloseOnceFd {
    type Incoming = OwnedFd;
    fn new() -> Self { UnixCloseOnceFd::new() }
    fn vacant(&self) -> bool { self.state() == CloseState::Vacant }
    fn adopt_prepared(&mut self, incoming: OwnedFd) {
        // Eligibility was established before calling the factory. Nothing can
        // mutate this private prepared owner between validation and adoption.
        if let Err(unreachable_owner) = self.adopt(incoming) {
            // Even an internal invariant failure must not implicitly close.
            std::mem::forget(unreachable_owner);
            unreachable!("prepared vacant close owner refused adoption");
        }
    }
    fn close(&mut self) -> Result<(), ()> { UnixCloseOnceFd::close(self).map_err(|_| ()) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase { Vacant, AcquisitionUnknown, Open, RetiringUnknown, Closed, Unknown }

struct Entry<T> { fd: Option<T> }
enum Custody<T> { Shared(Arc<Entry<T>>), Unique(Entry<T>) }
struct EngineSlot<T> { phase: Phase, custody: Option<Custody<T>> }
struct Lease<T> { index: usize, entry: Arc<Entry<T>> }

impl<T> Clone for Lease<T> {
    fn clone(&self) -> Self { Self { index: self.index, entry: Arc::clone(&self.entry) } }
}

// Stack-local adapters only: moving a lease never allocates or panics.
trait LeaseCell<T> {
    fn lease(&self) -> Option<&Lease<T>>;
    fn take(&mut self) -> Option<Lease<T>>;
    fn restore(&mut self, lease: Lease<T>);
}

impl<T> LeaseCell<T> for Option<Lease<T>> {
    fn lease(&self) -> Option<&Lease<T>> { self.as_ref() }
    fn take(&mut self) -> Option<Lease<T>> { Option::take(self) }
    fn restore(&mut self, lease: Lease<T>) { *self = Some(lease); }
}

impl LeaseCell<UnixCloseOnceFd> for Option<FileLease> {
    fn lease(&self) -> Option<&Lease<UnixCloseOnceFd>> {
        self.as_ref().map(|lease| &lease.inner)
    }
    fn take(&mut self) -> Option<Lease<UnixCloseOnceFd>> {
        Option::take(self).map(|lease| lease.inner)
    }
    fn restore(&mut self, inner: Lease<UnixCloseOnceFd>) { *self = Some(FileLease { inner }); }
}

// Own the lock before extraction, and custody before any injected operation.
// Recovery moves the exact owner back without allocating or acquiring a lock.
struct RecoveryGuard<'a, 'c, T> {
    slots: MutexGuard<'a, Vec<EngineSlot<T>>>,
    index: usize,
    custody: Option<Custody<T>>,
    fallback: Phase,
    caller: Option<&'c mut dyn LeaseCell<T>>,
}

impl<T> RecoveryGuard<'_, '_, T> {
    fn restore_caller(&mut self) {
        if let (Some(caller), Some(Custody::Shared(entry))) =
            (self.caller.as_mut(), self.custody.as_ref())
        {
            if caller.lease().is_none() {
                caller.restore(Lease { index: self.index, entry: Arc::clone(entry) });
            }
        }
    }

    fn finish(mut self, phase: Phase) {
        self.restore_caller();
        self.slots[self.index].custody = self.custody.take();
        self.slots[self.index].phase = phase;
    }
}

impl<T> Drop for RecoveryGuard<'_, '_, T> {
    fn drop(&mut self) {
        self.restore_caller();
        if let Some(custody) = self.custody.take() {
            self.slots[self.index].custody = Some(custody);
            self.slots[self.index].phase = self.fallback;
        }
    }
}

struct Engine<T>(PhantomData<T>);

impl<T: Owner> Engine<T> {
    fn prepare(count: usize) -> Result<Mutex<Vec<EngineSlot<T>>>, RegistryError> {
        let mut slots = Vec::new();
        slots.try_reserve_exact(count).map_err(|_| RegistryError::Allocation)?;
        for _ in 0..count {
            slots.push(EngineSlot {
                phase: Phase::Vacant,
                custody: Some(Custody::Shared(Arc::new(Entry { fd: None }))),
            });
        }
        Ok(Mutex::new(slots))
    }

    fn acquire<E>(
        table: &Mutex<Vec<EngineSlot<T>>>, index: usize,
        factory: impl FnOnce() -> Result<T::Incoming, E>,
    ) -> Result<Lease<T>, AcquireError<E>> {
        let registry = AcquireError::Registry;
        let mut slots = table.lock().map_err(|_| registry(RegistryError::Poisoned))?;
        let slot = slots.get_mut(index).ok_or_else(|| registry(RegistryError::Bounds))?;
        if slot.phase != Phase::Vacant { return Err(registry(RegistryError::State)); }
        match slot.custody.as_mut() {
            Some(Custody::Shared(entry)) => {
                let entry = Arc::get_mut(entry).ok_or_else(|| registry(RegistryError::Busy))?;
                if entry.fd.is_some() { return Err(registry(RegistryError::State)); }
            }
            _ => return Err(registry(RegistryError::State)),
        }
        let mut owner = T::new();
        if !owner.vacant() { return Err(registry(RegistryError::State)); }
        let mut guard = RecoveryGuard {
            slots, index, custody: None, fallback: Phase::AcquisitionUnknown, caller: None,
        };
        guard.slots[index].phase = Phase::AcquisitionUnknown;
        guard.custody = guard.slots[index].custody.take();
        let entry = match guard.custody.as_mut() {
            Some(Custody::Shared(entry)) => entry,
            _ => unreachable!("validated shared custody"),
        };
        let writable = Arc::get_mut(entry).expect("validated unique entry");
        let incoming = factory().map_err(AcquireError::Factory)?;
        owner.adopt_prepared(incoming);
        writable.fd = Some(owner);
        let lease = Lease { index, entry: Arc::clone(entry) };
        guard.finish(Phase::Open);
        Ok(lease)
    }

    fn retire(table: &Mutex<Vec<EngineSlot<T>>>, lease: Lease<T>) -> Result<(), RegistryError> {
        let mut caller = Some(lease);
        Self::try_retire(table, &mut caller)
    }

    fn try_retire(
        table: &Mutex<Vec<EngineSlot<T>>>, caller: &mut dyn LeaseCell<T>,
    ) -> Result<(), RegistryError> {
        let lease = caller.lease().ok_or(RegistryError::State)?;
        let slots = table.lock().map_err(|_| RegistryError::Poisoned)?;
        let index = lease.index;
        let slot = slots.get(index).ok_or(RegistryError::Bounds)?;
        match slot.custody.as_ref() {
            Some(Custody::Shared(entry)) if Arc::ptr_eq(entry, &lease.entry) => {}
            _ => return Err(RegistryError::WrongTable),
        }
        if slot.phase != Phase::Open { return Err(RegistryError::State); }
        let mut guard = RecoveryGuard {
            slots, index, custody: None, fallback: Phase::Open, caller: Some(caller),
        };
        guard.slots[index].phase = Phase::RetiringUnknown;
        guard.custody = guard.slots[index].custody.take();
        drop(guard.caller.as_mut().expect("retirement caller").take());
        #[cfg(test)]
        if let Some(Custody::Shared(entry)) = guard.custody.as_ref() {
            entry.fd.as_ref().expect("open entry owns descriptor").shared_retirement();
        }
        let shared = match guard.custody.take() {
            Some(Custody::Shared(entry)) => entry,
            _ => unreachable!("validated shared custody"),
        };
        match Arc::try_unwrap(shared) {
            Ok(entry) => {
                guard.custody = Some(Custody::Unique(entry));
                guard.fallback = Phase::Unknown;
            }
            Err(entry) => {
                guard.custody = Some(Custody::Shared(entry));
                guard.finish(Phase::Open);
                return Err(RegistryError::Busy);
            }
        }
        let result = match guard.custody.as_mut() {
            Some(Custody::Unique(entry)) => entry.fd.as_mut()
                .expect("open entry owns descriptor").close(),
            _ => unreachable!("unique retirement custody"),
        };
        match result {
            Ok(()) => { guard.finish(Phase::Closed); Ok(()) }
            Err(()) => { guard.finish(Phase::Unknown); Err(RegistryError::Unknown) }
        }
    }
}

#[cfg(test)]
#[path = "resident_fd_tests.rs"]
mod tests;
