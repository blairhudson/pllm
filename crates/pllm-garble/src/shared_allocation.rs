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
        })
    }
    pub fn used_bytes(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }
    pub(crate) fn reserve(&self, bytes: usize) -> Result<Lease<'_>, Error> {
        self.used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map_err(|_| Error::Resource)?;
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
