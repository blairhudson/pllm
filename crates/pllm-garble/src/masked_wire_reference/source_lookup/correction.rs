//! Private consuming scan-to-correction boundary. Not a raw-mask issuer.
use super::*;
use zeroize::Zeroize;

#[cfg(test)]
thread_local! {
    pub(super) static PANIC_BEGIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static PANIC_ADVANCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) struct ScanResultFor<C: AllocationContext> {
    pub(super) sums: Zeroizing<Vec<u64>>,
    pub(super) counts: Zeroizing<Vec<i64>>,
    pub(super) role: u8,
    pub(super) spec: Spec,
    pub(super) binding: [u8; 32],
    pub(super) digest: [u8; 32],
    pub(super) lease: LeaseFor<C>,
}

pub(super) struct CorrectionState {
    pub(super) arithmetic: Zeroizing<Vec<u64>>,
    pub(super) correction: Zeroizing<Vec<u64>>,
    pub(super) digest: [u8; 32],
    pub(super) phase: u8,
}

pub(super) struct CorrectionFor<C: AllocationContext> {
    pub(super) masks: Zeroizing<Vec<[u64; 2]>>,
    pub(super) state: Option<CorrectionState>,
    pub(super) select: Option<selection::PartyFor<C>>,
    pub(super) scan_lease: Option<LeaseFor<C>>,
    pub(super) role: u8,
    pub(super) spec: Spec,
    pub(super) binding: [u8; 32],
    pub(super) lease: LeaseFor<C>,
}

// Named live call/result slots, included before issuance, not a Budget token.
pub(super) struct Controls<'a, C: AllocationContext> {
    _scan: Option<ScanResultFor<C>>,
    _state: Option<CorrectionState>,
    _frame: Result<FrameFor<C>, Error>,
    _step: Result<StepFor<C>, Error>,
    _abort: Option<AbortOnDrop<'a, C>>,
    _scanner_transition: Option<Execution>,
}

pub(super) struct AbortOnDrop<'a, C: AllocationContext> {
    pub(super) correction: &'a mut CorrectionFor<C>,
    pub(super) armed: bool,
}
impl<C: AllocationContext> Drop for AbortOnDrop<'_, C> {
    fn drop(&mut self) {
        if self.armed {
            self.correction.cancel();
        }
    }
}

pub(super) fn scan_bytes<C: AllocationContext>(spec: Spec) -> Result<usize, Error> {
    add(&[
        size_of::<ScanResultFor<C>>(),
        mul(16, spec.lanes)?,
        2 * ALLOCATOR,
    ])
}

pub(super) fn control_bytes<C: AllocationContext>() -> usize {
    size_of::<Controls<'_, C>>()
}

impl<C: AllocationContext> CorrectionFor<C> {
    pub(super) fn cancel(&mut self) {
        self.state = None;
        self.masks.zeroize();
        if let Some(select) = &mut self.select {
            select.cancel();
        }
    }

    pub(super) fn begin(&mut self, scan: ScanResultFor<C>) -> Result<FrameFor<C>, Error> {
        let result = self.begin_inner(scan);
        if result.is_err() {
            self.cancel();
        }
        result
    }

    fn begin_inner(&mut self, scan: ScanResultFor<C>) -> Result<FrameFor<C>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        if state.phase != 0 {
            return Err(Error::Consumed);
        }
        if !std::ptr::eq(scan.lease.budget.budget(), self.lease.budget.budget())
            || scan.role != self.role
            || scan.spec != self.spec
            || scan.binding != self.binding
        {
            return Err(Error::Context);
        }
        if scan.sums.len() != self.spec.lanes || scan.counts.len() != self.spec.lanes {
            return Err(Error::Domain);
        }
        // Validate provenance before reading scan words. No arbitrary raw words
        // can enter: only the parent scanner constructs ScanResultFor.
        let ScanResultFor {
            mut sums,
            counts,
            digest,
            lease,
            ..
        } = scan;
        self.scan_lease = Some(lease);
        #[cfg(test)]
        PANIC_BEGIN.with(|armed| {
            if armed.replace(false) {
                panic!("injected correction begin unwind");
            }
        });
        for i in 0..self.spec.lanes {
            sums[i] = sums[i].wrapping_add(self.masks[i][0]) & mask(self.spec.output_bits);
            state.correction[i] = super::correction(self.role, counts[i]) ^ self.masks[i][1];
        }
        state.arithmetic = sums;
        self.masks.zeroize();
        state.digest = digest;
        // Scan count backing retires before its original reservation.
        drop(counts);
        let frame = self.first_frame(&state)?;
        state.phase = 1;
        self.state = Some(state);
        Ok(frame)
    }
}
