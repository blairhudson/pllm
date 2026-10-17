//! All variable-size owned payloads use fallible exact reservations. No growth,
//! fallible-iterator collection, shrinking conversion to Box, or allocating
//! Clone is used on capsule payload paths. Empty vectors do not allocate.
//!
//! Stable Rust's fixed-size Arc handles still use Arc::new and may abort on OOM;
//! this is not universal allocator-failure recovery. These handles are explicitly
//! bounded by client/window admission. Callers' Arc::new(Publication), caller
//! buffers/copies and allocator bookkeeping are outside payload recovery.

use super::Error;

pub(super) fn capacity<T>(length: usize) -> Result<Vec<T>, Error> {
    length.checked_mul(size_of::<T>()).ok_or(Error::Limit)?;
    #[cfg(test)]
    failure::allocation(length)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Error::Allocation)?;
    Ok(values)
}

pub(super) fn copy<T: Copy>(source: &[T]) -> Result<Vec<T>, Error> {
    let mut values = capacity(source.len())?;
    values.extend_from_slice(source);
    Ok(values)
}

pub(super) fn filled<T: Copy>(length: usize, value: T) -> Result<Vec<T>, Error> {
    let mut values = capacity(length)?;
    values.resize(length, value);
    Ok(values)
}

pub(super) fn collect<T>(source: impl ExactSizeIterator<Item = T>) -> Result<Vec<T>, Error> {
    let mut values = capacity(source.len())?;
    for value in source {
        values.push(value);
    }
    Ok(values)
}

// Private deterministic failure injection at the exact payload reservation seam.
// Production always calls Vec::try_reserve_exact. A separate native allocator
// regression also injects real null allocations at that seam.
#[cfg(test)]
pub(super) mod failure {
    use super::Error;
    use std::cell::Cell;

    thread_local! {
        static FAIL_AFTER: Cell<Option<usize>> = const { Cell::new(None) };
    }

    pub struct Guard;
    pub fn after(successful_reservations: usize) -> Guard {
        FAIL_AFTER.with(|slot| {
            assert!(slot.get().is_none());
            slot.set(Some(successful_reservations));
        });
        Guard
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            FAIL_AFTER.with(|slot| slot.set(None));
        }
    }
    pub(super) fn allocation(length: usize) -> Result<(), Error> {
        if length == 0 {
            return Ok(());
        }
        FAIL_AFTER.with(|slot| match slot.get() {
            None => Ok(()),
            Some(0) => Err(Error::Allocation),
            Some(n) => {
                slot.set(Some(n - 1));
                Ok(())
            }
        })
    }
}
