use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use zeroize::Zeroizing;

use super::{Error, Result, BATCH_SIZE, DIMENSION, MAX_ATTEMPTS};

pub const ALLOCATION_CEILING: usize = 16 << 20;
const BUFFER_OVERHEAD: usize = 128;
const CONTROL_ALLOWANCE: usize = 256 << 10;
const TRANSIENT_ALLOWANCE: usize = 512 << 10;

/// Checked complete capsule allocation envelope, not a process-RSS bound.
///
/// External charge covers one bounded set of borrowed matrices/code descriptors
/// and incoming P/Q/Qaux while the component owns copies. Runtime reservations
/// additionally precede every variable-sized component allocation. Control,
/// stack and allocator allowances are explicit rather than omitted payload.
#[derive(Debug, Clone, Copy)]
pub struct AllocationPlan {
    public_payload_bytes: usize,
    external_payload_bytes: usize,
    server_payload_bytes: usize,
    transient_allowance_bytes: usize,
    control_allowance_bytes: usize,
    retained_output_bytes: usize,
    planned_peak_bytes: usize,
    ceiling: usize,
}

impl AllocationPlan {
    pub fn public_payload_bytes(&self) -> usize {
        self.public_payload_bytes
    }
    pub fn external_payload_bytes(&self) -> usize {
        self.external_payload_bytes
    }
    pub fn server_payload_bytes(&self) -> usize {
        self.server_payload_bytes
    }
    pub fn transient_allowance_bytes(&self) -> usize {
        self.transient_allowance_bytes
    }
    pub fn control_allowance_bytes(&self) -> usize {
        self.control_allowance_bytes
    }
    pub fn retained_output_bytes(&self) -> usize {
        self.retained_output_bytes
    }
    pub fn planned_peak_bytes(&self) -> usize {
        self.planned_peak_bytes
    }

    pub fn for_shape(m: usize, n: usize, batch: usize, ceiling: usize) -> Result<Self> {
        // No allocation or hashing before all geometry and size checks.
        let mn = m.checked_mul(n).ok_or(Error::Allocation)?;
        let nx = n.checked_mul(8).ok_or(Error::Allocation)?;
        let ny = m.checked_mul(8).ok_or(Error::Allocation)?;
        let secondary = batch
            .checked_mul(n)
            .and_then(|v| v.checked_mul(8))
            .ok_or(Error::Allocation)?;
        if (m, n, batch) != (DIMENSION, DIMENSION, BATCH_SIZE) {
            return Err(Error::Shape);
        }
        let matrices = mn.checked_mul(batch * 4).ok_or(Error::Allocation)?;
        let p = m
            .checked_mul(nx)
            .and_then(|v| v.checked_mul(batch * 4))
            .ok_or(Error::Allocation)?;
        let q = ny
            .checked_mul(n)
            .and_then(|v| v.checked_mul(batch * 4))
            .ok_or(Error::Allocation)?;
        let auxiliary = ny
            .checked_mul(secondary)
            .and_then(|v| v.checked_mul(4))
            .ok_or(Error::Allocation)?;
        let codes = (batch * nx + ny + secondary) * 16;
        let public_payload_bytes = [matrices, p, q, auxiliary, codes]
            .into_iter()
            .try_fold(0_usize, |a, b| a.checked_add(b).ok_or(Error::Allocation))?;
        let server_payload_bytes = matrices + q;
        let retained_output_bytes = MAX_ATTEMPTS as usize * batch * m * 4;
        let planned_peak_bytes = [
            public_payload_bytes,
            public_payload_bytes,
            server_payload_bytes,
            TRANSIENT_ALLOWANCE,
            CONTROL_ALLOWANCE,
            retained_output_bytes,
        ]
        .into_iter()
        .try_fold(0_usize, |a, b| a.checked_add(b).ok_or(Error::Allocation))?;
        if ceiling > ALLOCATION_CEILING || planned_peak_bytes > ceiling {
            return Err(Error::Allocation);
        }
        Ok(Self {
            public_payload_bytes,
            external_payload_bytes: public_payload_bytes,
            server_payload_bytes,
            transient_allowance_bytes: TRANSIENT_ALLOWANCE,
            control_allowance_bytes: CONTROL_ALLOWANCE,
            retained_output_bytes,
            planned_peak_bytes,
            ceiling,
        })
    }

    pub fn bounded() -> Result<Self> {
        Self::for_shape(DIMENSION, DIMENSION, BATCH_SIZE, ALLOCATION_CEILING)
    }

    pub(super) fn start(self) -> Memory {
        let base = self.external_payload_bytes + self.control_allowance_bytes;
        // Fixed metadata allocation: Arc::new follows Rust's abort-on-OOM
        // policy. Fallible word-buffer allocation does not change that policy.
        let memory = Memory(Arc::new(State {
            used: AtomicUsize::new(base),
            peak: AtomicUsize::new(base),
            ceiling: self.ceiling,
            allocations: AtomicUsize::new(0),
        }));
        #[cfg(test)]
        audit::capture(&memory);
        memory
    }
}

struct State {
    used: AtomicUsize,
    peak: AtomicUsize,
    ceiling: usize,
    allocations: AtomicUsize,
}

#[derive(Clone)]
pub(super) struct Memory(Arc<State>);

#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    /// Includes the fixed external/control reservation as well as live buffers.
    pub charged_live_bytes: usize,
    pub charged_peak_bytes: usize,
    pub ceiling_bytes: usize,
    pub buffer_allocations: usize,
}

impl Memory {
    #[cfg(test)]
    pub(super) fn owner_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    pub(super) fn stats(&self) -> MemoryStats {
        MemoryStats {
            charged_live_bytes: self.0.used.load(Ordering::Acquire),
            charged_peak_bytes: self.0.peak.load(Ordering::Acquire),
            ceiling_bytes: self.0.ceiling,
            buffer_allocations: self.0.allocations.load(Ordering::Acquire),
        }
    }

    fn reserve(&self, bytes: usize) -> Result<Reservation> {
        let previous = self
            .0
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|&n| n <= self.0.ceiling)
            })
            .map_err(|_| Error::Allocation)?;
        self.0.peak.fetch_max(previous + bytes, Ordering::AcqRel);
        Ok(Reservation {
            memory: self.clone(),
            bytes,
        })
    }
}

struct Reservation {
    memory: Memory,
    bytes: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.memory.0.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Every variable-sized module buffer uses this owner. No Clone/Debug/serde.
/// Data drops (and is zeroized) before its reservation is released.
pub(super) struct Words {
    data: Zeroizing<Vec<u32>>,
    _reservation: Reservation,
}

impl Words {
    pub(super) fn zeros(memory: &Memory, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(4)
            .and_then(|n| n.checked_add(BUFFER_OVERHEAD))
            .ok_or(Error::Allocation)?;
        let reservation = memory.reserve(bytes)?;
        // Identifies the actual allocator call to the test allocator. This
        // observes reservations; it neither fails nor replaces any allocation.
        #[cfg(test)]
        let _allocation_scope = audit::AllocationScope::new(memory, bytes);
        let mut data = Zeroizing::new(Vec::new());
        data.try_reserve_exact(len).map_err(|_| Error::Allocation)?;
        // Global's Vec allocation requests exactly the admitted layout. Reject
        // any unexpected capacity rather than adopting an unaccounted buffer.
        if data.capacity() != len {
            return Err(Error::Allocation);
        }
        data.resize(len, 0);
        memory.0.allocations.fetch_add(1, Ordering::AcqRel);
        Ok(Self {
            data,
            _reservation: reservation,
        })
    }

    pub(super) fn copy(memory: &Memory, source: &[u32]) -> Result<Self> {
        let mut result = Self::zeros(memory, source.len())?;
        result.as_mut().copy_from_slice(source);
        Ok(result)
    }

    pub(super) fn as_slice(&self) -> &[u32] {
        &self.data
    }

    pub(super) fn as_mut(&mut self) -> &mut [u32] {
        &mut self.data
    }
}

/// Test-only observers: real failures are injected by the integration binary's
/// GlobalAlloc implementation returning null at a selected allocation index.
#[cfg(test)]
pub(super) mod audit {
    use std::cell::{Cell, RefCell};

    use super::{AtomicUsize, Memory};

    thread_local! {
        static ALLOCATION: Cell<Option<(*const AtomicUsize, usize)>> = const { Cell::new(None) };
        static CAPTURE_NEXT: Cell<bool> = const { Cell::new(false) };
        static CAPTURED: RefCell<Option<Memory>> = const { RefCell::new(None) };
    }

    pub(super) struct AllocationScope(Option<(*const AtomicUsize, usize)>);

    impl AllocationScope {
        pub(super) fn new(memory: &Memory, charge: usize) -> Self {
            Self(ALLOCATION.with(|slot| slot.replace(Some((&memory.0.used, charge)))))
        }
    }

    impl Drop for AllocationScope {
        fn drop(&mut self) {
            ALLOCATION.with(|slot| slot.set(self.0));
        }
    }

    pub(in super::super) fn allocation_context() -> Option<(*const AtomicUsize, usize)> {
        ALLOCATION.try_with(Cell::get).ok().flatten()
    }

    pub(in super::super) fn capture_next() {
        CAPTURED.with(|slot| assert!(slot.borrow().is_none()));
        CAPTURE_NEXT.with(|slot| slot.set(true));
    }

    pub(super) fn capture(memory: &Memory) {
        if CAPTURE_NEXT.with(|slot| slot.replace(false)) {
            CAPTURED.with(|slot| *slot.borrow_mut() = Some(memory.clone()));
        }
    }

    pub(in super::super) fn take_capture() -> Memory {
        CAPTURE_NEXT.with(|slot| slot.set(false));
        CAPTURED.with(|slot| {
            slot.borrow_mut()
                .take()
                .expect("memory capture was not created")
        })
    }
}
