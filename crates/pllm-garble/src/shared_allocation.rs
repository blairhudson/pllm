//! Shared logical capacity budget; not an allocator/RSS bound.
use std::sync::atomic::{AtomicUsize, Ordering};

pub const MAX_ALLOCATION_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Domain,
    Unsupported,
    Resource,
    Allocation,
    Randomness,
    Context,
    Consumed,
}

/// Borrowed by every lease, including exported output owners.
pub struct Budget {
    limit: usize,
    used: AtomicUsize,
    #[cfg(test)]
    pub(crate) fail_after: std::sync::atomic::AtomicIsize,
    #[cfg(test)]
    accounting_updates: AtomicUsize,
}
impl Budget {
    pub fn new(limit: usize) -> Result<Self, Error> {
        if limit == 0 || limit > MAX_ALLOCATION_BYTES {
            return Err(Error::Resource);
        }
        Ok(Self {
            limit,
            used: AtomicUsize::new(0),
            #[cfg(test)]
            fail_after: std::sync::atomic::AtomicIsize::new(-1),
            #[cfg(test)]
            accounting_updates: AtomicUsize::new(0),
        })
    }
    pub fn used_bytes(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }
    /// Charge caller-owned payload and the guard's inline metadata to this budget.
    ///
    /// This is logical accounting only: the guard does not allocate, own, or erase
    /// the payload, and grants no native execution or provenance authority. The
    /// caller must retain it until the actual payload owner has been dropped.
    /// Owner wrappers should enforce that ordering with private fields.
    pub fn charge_external(&self, payload_bytes: usize) -> Result<ExternalCharge<'_>, Error> {
        let bytes = payload_bytes
            .checked_add(std::mem::size_of::<ExternalCharge<'_>>())
            .ok_or(Error::Resource)?;
        Ok(ExternalCharge {
            lease: self.reserve(bytes)?,
        })
    }
    pub(crate) fn reserve(&self, bytes: usize) -> Result<Lease<'_>, Error> {
        self.used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map_err(|_| Error::Resource)?;
        #[cfg(test)]
        self.accounting_updates.fetch_add(1, Ordering::SeqCst);
        Ok(Lease {
            budget: self,
            bytes,
        })
    }
    pub(crate) fn allocation(&self) -> Result<(), Error> {
        #[cfg(test)]
        {
            let n = self.fail_after.load(Ordering::SeqCst);
            if n == 0 {
                return Err(Error::Allocation);
            }
            if n > 0 {
                self.fail_after.fetch_sub(1, Ordering::SeqCst);
            }
        }
        Ok(())
    }
}

/// Non-cloneable logical accounting for caller-owned external storage.
///
/// Includes its own inline metadata in the same budget as the payload. It does
/// not own or erase caller buffers, certify their contents, or grant native
/// execution authority. Retain it through the actual payload owner's drop;
/// cancellation of work alone is not grounds for releasing a retained payload's
/// charge. Private owner-wrapper fields must enforce payload-before-guard drop.
///
/// ```compile_fail
/// use pllm_garble::shared_allocation::Budget;
/// let budget = Budget::new(1024).unwrap();
/// let charge = budget.charge_external(8).unwrap();
/// let duplicate = charge.clone();
/// ```
pub struct ExternalCharge<'a> {
    lease: Lease<'a>,
}

impl<'a> ExternalCharge<'a> {
    pub fn payload_bytes(&self) -> usize {
        self.lease.bytes - std::mem::size_of::<Self>()
    }

    pub fn charged_bytes(&self) -> usize {
        self.lease.bytes
    }

    /// Whether this guard charges this exact budget, not an equal-sized budget.
    pub fn is_for(&self, budget: &Budget) -> bool {
        std::ptr::eq(self.lease.budget, budget)
    }

    /// Move already-charged payload into a native owner without touching `used`.
    /// The source keeps its inline metadata and any untransferred payload. This
    /// is accounting only, never provenance or permission to execute.
    pub(crate) fn split_payload(
        &mut self,
        budget: &Budget,
        bytes: usize,
    ) -> Result<Lease<'a>, Error> {
        if !self.is_for(budget) || bytes > self.payload_bytes() {
            return Err(Error::Resource);
        }
        self.lease.split(bytes)
    }

    /// Change the logical payload charge, retaining the inline metadata charge.
    ///
    /// Growth reserves the positive increment atomically before changing this
    /// guard; failure leaves its original charge intact. Call before growing the
    /// actual payload. Shrink only after the corresponding storage is released.
    pub fn resize_payload(&mut self, payload_bytes: usize) -> Result<(), Error> {
        let bytes = payload_bytes
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(Error::Resource)?;
        if bytes > self.lease.bytes {
            let mut increment = self.lease.budget.reserve(bytes - self.lease.bytes)?;
            self.lease.bytes = bytes;
            // Transfer accounting into this guard without releasing the increment.
            increment.bytes = 0;
        } else {
            let released = self.lease.bytes - bytes;
            self.lease.bytes = bytes;
            #[cfg(test)]
            self.lease
                .budget
                .accounting_updates
                .fetch_add(1, Ordering::SeqCst);
            self.lease.budget.used.fetch_sub(released, Ordering::SeqCst);
        }
        Ok(())
    }
}

pub(crate) struct Lease<'a> {
    pub(crate) budget: &'a Budget,
    pub(crate) bytes: usize,
}
impl<'a> Lease<'a> {
    pub(crate) fn split(&mut self, bytes: usize) -> Result<Lease<'a>, Error> {
        self.bytes = self.bytes.checked_sub(bytes).ok_or(Error::Resource)?;
        Ok(Lease {
            budget: self.budget,
            bytes,
        })
    }
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        #[cfg(test)]
        self.budget
            .accounting_updates
            .fetch_add(1, Ordering::SeqCst);
        self.budget.used.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
pub(crate) fn allocate<T>(capacity: usize, budget: &Budget) -> Result<Vec<T>, Error> {
    budget.allocation()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Allocation)?;
    exact_capacity(values, capacity)
}
pub(crate) fn exact_capacity<T>(values: Vec<T>, capacity: usize) -> Result<Vec<T>, Error> {
    if values.capacity() != capacity {
        return Err(Error::Resource);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_transfer_keeps_exact_full_budget_and_inline_guard() {
        let metadata = std::mem::size_of::<ExternalCharge<'_>>();
        let budget = Budget::new(metadata + 31).unwrap();
        let mut charge = budget.charge_external(31).unwrap();
        let updates = budget.accounting_updates.load(Ordering::SeqCst);
        let lease = charge.split_payload(&budget, 31).unwrap();
        // Unlike a final used-byte comparison, this catches a transient
        // decrement/reacquire window even if nobody wins the race for capacity.
        assert_eq!(budget.accounting_updates.load(Ordering::SeqCst), updates);
        assert_eq!(lease.bytes, 31);
        assert_eq!(charge.payload_bytes(), 0);
        assert_eq!(charge.charged_bytes(), metadata);
        assert_eq!(budget.used_bytes(), metadata + 31);
        // Transfer did not make even one byte available to a competing owner.
        assert!(matches!(budget.reserve(1), Err(Error::Resource)));
        drop(charge);
        assert_eq!(budget.used_bytes(), 31);
        drop(lease);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn payload_transfer_failure_is_atomic_and_partial_slices_drop_independently() {
        let metadata = std::mem::size_of::<ExternalCharge<'_>>();
        let budget = Budget::new(metadata + 31).unwrap();
        let foreign = Budget::new(metadata + 31).unwrap();
        let mut charge = budget.charge_external(31).unwrap();
        for (target, bytes) in [(&foreign, 1), (&foreign, 0), (&budget, 32)] {
            assert!(matches!(
                charge.split_payload(target, bytes),
                Err(Error::Resource)
            ));
            assert_eq!(charge.payload_bytes(), 31);
            assert_eq!(budget.used_bytes(), metadata + 31);
            assert_eq!(foreign.used_bytes(), 0);
        }
        let first = charge.split_payload(&budget, 12).unwrap();
        let second = charge.split_payload(&budget, 19).unwrap();
        drop(first);
        assert_eq!(budget.used_bytes(), metadata + 19);
        drop(second);
        assert_eq!(budget.used_bytes(), metadata);
        charge.resize_payload(31).unwrap();
        drop(charge);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn zero_and_one_byte_transfers_preserve_metadata_in_either_drop_order() {
        let metadata = std::mem::size_of::<ExternalCharge<'_>>();
        for bytes in [0, 1] {
            for retained in [0, 1] {
                for source_first in [false, true] {
                    let budget = Budget::new(metadata + bytes + retained).unwrap();
                    let mut source = budget.charge_external(bytes + retained).unwrap();
                    let updates = budget.accounting_updates.load(Ordering::SeqCst);
                    let native = source.split_payload(&budget, bytes).unwrap();
                    assert_eq!(budget.accounting_updates.load(Ordering::SeqCst), updates);
                    assert_eq!(source.payload_bytes(), retained);
                    assert_eq!(source.charged_bytes(), metadata + retained);
                    assert_eq!(budget.used_bytes(), metadata + bytes + retained);
                    if source_first {
                        drop(source);
                        assert_eq!(budget.used_bytes(), bytes);
                        drop(native);
                    } else {
                        drop(native);
                        assert_eq!(budget.used_bytes(), metadata + retained);
                        drop(source);
                    }
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}
