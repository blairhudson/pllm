//! Shared process-local native custody, not a numeric protocol or transport.
//!
//! Extracted from D27 without changing its custody graph or lock order. Only
//! private native states implement retirement; no caller-provided callback or
//! trait implementation may run while either metadata/state lock is held.
use super::*;
use std::alloc::Layout;
use std::sync::{atomic::AtomicUsize, Arc, Mutex, MutexGuard};

pub(super) trait Retire {
    /// Erase output and cancel native continuation, synchronously. No caller
    /// callback, THIS coordinator's metadata-lock acquisition or account
    /// detachment is permitted. A closed nested region may retire a strictly
    /// downstream child registry with no parent backlink or user callbacks.
    fn retire(&mut self);
}

pub(super) struct RoleSlotFor<B: AllocationContext, S: Retire> {
    pub(super) state: Mutex<S>,
    pub(super) _lease: LeaseFor<B>,
}

#[derive(Default)]
pub(super) struct SplitControl {
    pub(super) aborted: bool,
    pub(super) ready: [bool; 2],
}

impl SplitControl {
    pub(super) fn mark_ready(&mut self, role: u8) {
        self.ready[usize::from(role)] = true;
    }

    /// Only inside operation: aborted dispatch has already failed closed.
    pub(super) fn require_both_ready(&self) -> Result<(), Error> {
        if self.ready != [true; 2] {
            Err(Error::Consumed)
        } else {
            Ok(())
        }
    }
}

/// Strong cancellation-only registry, with exactly the original D27 field
/// order. Slots never link back: no cycles, Weak-accounting gap or leaked owner.
pub(super) struct CoordinationFor<B: AllocationContext, S: Retire> {
    pub(super) control: Mutex<SplitControl>,
    pub(super) slots: [Arc<RoleSlotFor<B, S>>; 2],
    pub(super) _original_run: Option<LeaseFor<B>>,
    pub(super) _additional: LeaseFor<B>,
}

pub(super) fn arc_bytes<T>() -> Result<usize, Error> {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map(|(layout, _)| layout.pad_to_align().size())
        .map_err(|_| Error::Resource)
}

impl<B: AllocationContext, S: Retire> CoordinationFor<B, S> {
    pub(super) fn abort_locked(&self, control: &mut SplitControl) {
        control.aborted = true;
        for slot in &self.slots {
            let mut state = slot.state.lock().unwrap_or_else(|p| p.into_inner());
            state.retire();
        }
    }

    pub(super) fn control(&self) -> MutexGuard<'_, SplitControl> {
        match self.control.lock() {
            Ok(control) => control,
            Err(poisoned) => {
                let mut control = poisoned.into_inner();
                self.abort_locked(&mut control);
                control
            }
        }
    }

    pub(super) fn abort(&self) {
        self.abort_locked(&mut self.control());
    }

    /// Poll with BOTH locks released. Reentry/cancellation is allowed; panic
    /// retires both native states before escaping. Internal native polls use
    /// only the locked aborted flag, not this caller callback.
    pub(super) fn poll_cancelled(&self, cancelled: &mut impl FnMut() -> bool) -> Result<(), Error> {
        let aborted = {
            let control = self.control();
            control.aborted
        };
        if aborted {
            return Err(Error::Consumed);
        }
        let mut poll = CallbackPollFor {
            coordination: self,
            armed: true,
        };
        let cancelled = cancelled();
        poll.armed = false;
        if cancelled {
            self.abort();
            Err(Error::Consumed)
        } else {
            Ok(())
        }
    }

    // Internal native operation, NOT an extension/callback API. The concrete
    // role implementation locks its own state and performs only native work.
    pub(super) fn operation<T>(
        &self,
        f: impl FnOnce(&mut SplitControl) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut operation = OperationFor {
            coordination: self,
            control: self.control(),
            armed: true,
        };
        let result = if operation.control.aborted {
            Err(Error::Consumed)
        } else {
            f(&mut operation.control)
        };
        if result.is_err() {
            self.abort_locked(&mut operation.control);
        }
        operation.armed = false;
        result
    }
}

// On native-work panic the concrete state guard unwinds FIRST. Then this guard
// erases both slots before unlocking metadata. External callback guard runs
// without either lock; those two unwind paths must not be merged.
pub(super) struct OperationFor<'a, B: AllocationContext, S: Retire> {
    coordination: &'a CoordinationFor<B, S>,
    control: MutexGuard<'a, SplitControl>,
    armed: bool,
}

pub(super) struct CallbackPollFor<'a, B: AllocationContext, S: Retire> {
    coordination: &'a CoordinationFor<B, S>,
    armed: bool,
}

impl<B: AllocationContext, S: Retire> Drop for CallbackPollFor<'_, B, S> {
    fn drop(&mut self) {
        if self.armed {
            self.coordination.abort();
        }
    }
}

impl<B: AllocationContext, S: Retire> Drop for OperationFor<'_, B, S> {
    fn drop(&mut self) {
        if self.armed {
            self.coordination.abort_locked(&mut self.control);
        }
    }
}

#[cfg(test)]
#[path = "role_coordination_nexp_tests.rs"]
mod nexp_tests;
