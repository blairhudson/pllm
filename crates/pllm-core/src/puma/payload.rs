//! Live payload accounting, independent of source numeric/PRSS identity.
use super::{Error, FIXED_RESERVE_BYTES, MAX_LANES};
use std::ops::{Deref, DerefMut};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use zeroize::Zeroize;

struct Ledger {
    live: AtomicUsize,
    peak: AtomicUsize,
    limit: usize,
}

/// Read-only per-runtime accounting handle. Cloning this handle neither clones
/// secret material nor grants issuance/reset/refund authority. Original exported
/// payloads keep the ledger alive after cancellation or destruction of the owner.
/// This is not a process-global limit or an allocator/RSS measurement.
#[derive(Clone)]
pub struct PayloadUsage {
    ledger: Arc<Ledger>,
}

impl PayloadUsage {
    pub(super) fn new(max_bytes: usize) -> Self {
        // One fixed-size Arc allocation per runtime: stable Rust aborts on OOM
        // here. Its metadata is covered by the fixed reserve, not payload bytes.
        Self {
            ledger: Arc::new(Ledger {
                live: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                limit: max_bytes - FIXED_RESERVE_BYTES,
            }),
        }
    }

    pub fn live_bytes(&self) -> usize {
        self.ledger.live.load(Ordering::Acquire)
    }
    pub fn peak_bytes(&self) -> usize {
        self.ledger.peak.load(Ordering::Acquire)
    }
    pub fn limit_bytes(&self) -> usize {
        self.ledger.limit
    }

    fn reserve(&self, bytes: usize) -> Result<Reservation, Error> {
        let before = self
            .ledger
            .live
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(bytes).filter(|&sum| sum <= self.ledger.limit)
            })
            .map_err(|_| Error::Bounds)?;
        self.ledger.peak.fetch_max(before + bytes, Ordering::AcqRel);
        Ok(Reservation {
            usage: self.clone(),
            bytes,
        })
    }

    pub(super) fn zeros(&self, n: usize) -> Result<Words, Error> {
        Words::new(n, Some(self))
    }
    pub(super) fn copy(&self, values: &[u64], n: usize) -> Result<Words, Error> {
        if values.len() != n {
            return Err(Error::Shape);
        }
        let mut result = self.zeros(n)?;
        result.copy_from_slice(values);
        Ok(result)
    }
}

struct Reservation {
    usage: PayloadUsage,
    bytes: usize,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.usage
            .ledger
            .live
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// A buffer can carry its issuer's lease and a receiving runtime's lease.
/// Protocol inputs are consumed, not forwarded as new outputs, so two suffice.
/// Raw import constructors are outer-caller allocations until admission binds
/// them. No public API can detach a lease or obtain the underlying Vec.
pub(super) struct Words {
    data: Vec<u64>,
    leases: [Option<Reservation>; 2],
}

impl Words {
    pub(super) fn new(n: usize, usage: Option<&PayloadUsage>) -> Result<Self, Error> {
        if n > 2 * MAX_LANES {
            return Err(Error::Bounds);
        }
        // Acquire BEFORE the fallible payload allocation. Local declaration
        // order also frees a failed allocation before refunding its reservation.
        let lease = usage.map(|u| u.reserve(n * 8)).transpose()?;
        let mut data = Vec::new();
        data.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
        if data.capacity() != n {
            // Reject a capacity-changing allocator implementation before any
            // secret copy. No hidden extra capacity is accepted by the ledger.
            data.resize(data.capacity(), 0);
            return Err(Error::Allocation);
        }
        data.resize(n, 0);
        Ok(Self {
            data,
            leases: [lease, None],
        })
    }

    pub(super) fn charge_to(&mut self, usage: &PayloadUsage) -> Result<(), Error> {
        if self
            .leases
            .iter()
            .flatten()
            .any(|r| Arc::ptr_eq(&r.usage.ledger, &usage.ledger))
        {
            return Ok(());
        }
        let slot = self
            .leases
            .iter_mut()
            .find(|r| r.is_none())
            .ok_or(Error::Bounds)?;
        *slot = Some(usage.reserve(self.data.capacity() * 8)?);
        Ok(())
    }
}

impl Deref for Words {
    type Target = [u64];
    fn deref(&self) -> &[u64] {
        &self.data
    }
}
impl DerefMut for Words {
    fn deref_mut(&mut self) -> &mut [u64] {
        &mut self.data
    }
}
impl Drop for Words {
    fn drop(&mut self) {
        self.data.as_mut_slice().zeroize();
        // Refund must occur AFTER the allocator's deallocation callback returns,
        // not merely after wiping or logical output extraction. Field leases
        // drop only after this body (and its real Vec deallocation) completes.
        drop(std::mem::take(&mut self.data));
    }
}
