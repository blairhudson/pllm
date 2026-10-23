//! TEST-ONLY owning bridge. No production coordinator/import issuer.
//!
//! Destination seals resolve into the SAME live Worker store and runtime/PRSS.
//! An original alias stays an alias. Copy mode copies the containing physical
//! owner (including packed offsets), replaces its backing, then erases the old
//! shares. Thus later original Worker reads actually use the installed backing;
//! this is not a detached receipt over an unexecuted export recipe.
//! Scope: continuation/aliasing inside the ORIGINAL Program. Switching Program,
//! request origin/epoch, public-source qualification and production drain remain
//! separate sealed requirements. No cross-growing-prefill relation is inferred.
use super::*;
use crate::replicated_decoder::body_tensor_integration::CoordinatorDrained;
use crate::replicated_decoder::program::compact_lowering::frozen_region::prefix_custody::{
    DestinationKey, PendingPrefix,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) enum Install {
    RetainAlias,
    Copy,
}

/// Non-cloneable receipt AND actual destination value store. It cannot escape
/// its consuming prefix owner. Workers erase before that owner's native/source
/// custody can be released. No raw-worker extraction or detached import token.
pub(in crate::replicated_decoder) struct Installed<'p, 'a> {
    workers: [Worker; 3],
    slots: Vec<usize>,
    pending: &'p PendingPrefix<'a>,
    failed: bool,
}

fn source_word(worker: &Worker, word: (usize, usize)) -> Result<(u64, u64), Error> {
    let (a, b) = worker.local_slices(word.0)?;
    Ok((
        *a.get(word.1).ok_or(Error::Shape)?,
        *b.get(word.1).ok_or(Error::Shape)?,
    ))
}

// No new lanes/Nodes/recipes: enumerate the existing bounded mappings lazily.
fn each_word(
    pending: &PendingPrefix<'_>,
    mut f: impl FnMut((usize, usize)) -> Result<(), Error>,
) -> Result<(), Error> {
    for edge in 0..pending.destination_count() {
        for lane in 0..pending.destination_lanes(edge)? {
            f(pending.mapping(edge, lane)?.0)?;
        }
    }
    if let Some(sticky) = pending.sticky_source() {
        f((sticky, 0))?;
    }
    Ok(())
}

fn mapped_slot(worker: &Worker, pending: &PendingPrefix<'_>, slot: usize) -> Result<bool, Error> {
    let state = worker.oracle_values.as_ref().ok_or(Error::Binding)?;
    let mut found = false;
    each_word(pending, |word| {
        found |= state.layout.span(word.0)?.stored()?.0 == slot;
        Ok(())
    })?;
    Ok(found)
}

fn preflight(
    q: &Qualified<'_>,
    pending: &PendingPrefix<'_>,
    workers: [&Worker; 3],
    mode: Install,
) -> Result<usize, Error> {
    if !Arc::ptr_eq(&q.origin, pending.source().execution_origin()) {
        return Err(Error::Binding);
    }
    // This work already includes all future native full/tail calls in the
    // original Program. Copy/alias adds no secret protocol and no native recipe.
    pending.resources().accumulated_work.limits()?;
    q.program.work_budget().limits()?;
    for (party, worker) in workers.iter().enumerate() {
        q.check(worker)?;
        if worker.party() as usize != party
            || !worker.boundary_ready()
            || worker.step != workers[0].step
        {
            return Err(Error::Binding);
        }
        each_word(pending, |word| {
            let span = q.layout.span(word.0)?;
            source_word(worker, word)?;
            // Original exported storage must remain protected throughout the
            // continuation, not merely happen to contain equal bytes now.
            if span.last_use < q.program.logical_len() {
                return Err(Error::Binding);
            }
            Ok(())
        })?;
    }
    // Replace the precisely inventoried frozen owner with the sealed pending
    // owner's census; retain all original oracle/layout/routing overlap above it.
    let mut peak = q
        .admitted_peak
        .checked_sub(pending.source().execution_retained_bytes())
        .ok_or(Error::Binding)?;
    peak = add(peak, pending.resources().retained_bytes)?;
    peak = add(
        peak,
        size_of::<Installed<'_, '_>>() + 2 * size_of::<Result<Installed<'_, '_>, Error>>(),
    )?;
    peak = add(peak, mul(q.layout.owners.len(), size_of::<usize>())?)?;
    if mode == Install::Copy {
        for slot in 0..workers[0].values.len() {
            if !mapped_slot(workers[0], pending, slot)? {
                continue;
            }
            for worker in workers {
                let old = worker.values[slot].as_ref().ok_or(Error::Binding)?;
                // All old storage is already in the original execution quote;
                // quote replacement capacity before the first Secret factory.
                peak = add(
                    peak,
                    mul(add(old.first.len(), old.second.len())?, size_of::<u64>())?,
                )?;
            }
        }
    }
    Ok(peak.max(pending.resources().peak_bytes))
}

/// Pure quote, not an import/drain issuer or authority to reuse a prefix.
#[allow(dead_code)] // Keep source-free quote alongside the exercised drained entry.
pub(in crate::replicated_decoder) fn quote(
    q: &Qualified<'_>,
    pending: &PendingPrefix<'_>,
    workers: &[Worker; 3],
    mode: Install,
) -> Result<usize, Error> {
    preflight(q, pending, workers.each_ref(), mode)
}

pub(in crate::replicated_decoder) fn quote_drained(
    q: &Qualified<'_>,
    pending: &PendingPrefix<'_>,
    workers: &[CoordinatorDrained<Worker>; 3],
    mode: Install,
) -> Result<usize, Error> {
    preflight(
        q,
        pending,
        workers
            .each_ref()
            .map(CoordinatorDrained::frozen_prefix_worker_ref),
        mode,
    )
}

pub(in crate::replicated_decoder) fn install<'p, 'a>(
    q: &Qualified<'_>,
    pending: &'p PendingPrefix<'a>,
    drained: [CoordinatorDrained<Worker>; 3],
    mode: Install,
    cap: usize,
) -> Result<Installed<'p, 'a>, Error> {
    install_inner(q, pending, drained, mode, cap, None)
}

// Fault injection is private test support, never an approval/receipt callback.
pub(in crate::replicated_decoder) fn install_inner<'p, 'a>(
    q: &Qualified<'_>,
    pending: &'p PendingPrefix<'a>,
    drained: [CoordinatorDrained<Worker>; 3],
    mode: Install,
    cap: usize,
    fail_after: Option<usize>,
) -> Result<Installed<'p, 'a>, Error> {
    pending.begin_runtime_attempt()?; // error/cancel/unwind burns all siblings
    let [a, b, c] = drained;
    let workers = [
        q.consume_for_prefix(a, pending.source())?,
        q.consume_for_prefix(b, pending.source())?,
        q.consume_for_prefix(c, pending.source())?,
    ];
    let peak = preflight(q, pending, workers.each_ref(), mode)?;
    if peak > cap.min(MAX_PROGRAM_BYTES) {
        return Err(Error::Bounds);
    }
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(q.layout.owners.len())
        .map_err(|_| Error::Allocation)?;
    if slots.capacity() != q.layout.owners.len() {
        return Err(Error::Allocation);
    }
    for slot in 0..workers[0].values.len() {
        if mapped_slot(&workers[0], pending, slot)? {
            slots.push(slot);
        }
    }
    // Private, uncommitted candidate. No reference escapes until every role's
    // mappings are installed; its Drop covers partial-copy errors/unwind too.
    let mut installed = Installed {
        workers,
        slots,
        pending,
        failed: true,
    };
    let mut copied = 0;
    if mode == Install::Copy {
        for &slot in &installed.slots {
            for worker in &mut installed.workers {
                if fail_after == Some(copied) {
                    return Err(Error::Phase);
                }
                let old = worker.values[slot].as_ref().ok_or(Error::Binding)?;
                COPY_FACTORIES.with(|n| n.set(n.get() + 1));
                let mut value = Secret::new(old.first.len())?;
                if value.first.capacity() != old.first.len()
                    || value.second.capacity() != old.second.len()
                {
                    return Err(Error::Allocation);
                }
                value.first.copy_from_slice(&old.first);
                value.second.copy_from_slice(&old.second);
                drop(worker.values[slot].replace(value)); // Zeroizing erases old shares NOW
                copied += 1;
            }
        }
    }
    // Verify ALL safe data and sticky words are installed before minting receipt.
    for worker in &installed.workers {
        each_word(pending, |word| source_word(worker, word).map(|_| ()))?;
    }
    installed.failed = false;
    Ok(installed)
}

impl Installed<'_, '_> {
    fn checked<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.failed = true;
            for worker in &mut self.workers {
                worker.cancel();
            }
        }
        result
    }
    pub(in crate::replicated_decoder) fn word(
        &mut self,
        party: usize,
        key: &DestinationKey<'_, '_>,
        lane: usize,
    ) -> Result<(u64, u64), Error> {
        let result = (|| {
            if self.failed {
                return Err(Error::Retired);
            }
            let edge = (0..self.pending.destination_count())
                .find(|edge| key.matches(self.pending, *edge))
                .ok_or(Error::Binding)?;
            source_word(
                self.workers.get(party).ok_or(Error::Binding)?,
                self.pending.mapping(edge, lane)?.0,
            )
        })();
        self.checked(result)
    }
    pub(in crate::replicated_decoder) fn sticky(
        &mut self,
        party: usize,
    ) -> Result<Option<(u64, u64)>, Error> {
        let result = (|| {
            if self.failed {
                return Err(Error::Retired);
            }
            self.pending
                .sticky_source()
                .map(|word| source_word(self.workers.get(party).ok_or(Error::Binding)?, (word, 0)))
                .transpose()
        })();
        self.checked(result)
    }
    pub(in crate::replicated_decoder) fn advance(
        &mut self,
        q: &Qualified<'_>,
        party: usize,
        before: Option<usize>,
    ) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = (|| {
            if self.failed {
                return Err(Error::Retired);
            }
            if !Arc::ptr_eq(&q.origin, self.pending.source().execution_origin()) {
                return Err(Error::Binding);
            }
            q.advance(self.workers.get_mut(party).ok_or(Error::Binding)?, before)
        })();
        self.checked(result)
    }
    pub(in crate::replicated_decoder) fn receive(
        &mut self,
        party: usize,
        bytes: &[u8],
        from: Party,
    ) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = (|| {
            if self.failed {
                return Err(Error::Retired);
            }
            self.workers
                .get_mut(party)
                .ok_or(Error::Binding)?
                .receive(bytes, from)
        })();
        self.checked(result)
    }
    pub(in crate::replicated_decoder) fn complete(&self) -> bool {
        !self.failed && self.workers.iter().all(Worker::complete)
    }
    pub(in crate::replicated_decoder) fn assert_same(
        &self,
        q: &Qualified<'_>,
        control: &[Worker; 3],
    ) {
        for (a, b) in control.iter().zip(&self.workers) {
            q.assert_same_state(a, b);
        }
    }
    pub(in crate::replicated_decoder) fn backing(&self, party: usize) -> Result<*const u64, Error> {
        Ok(self.workers.get(party).ok_or(Error::Binding)?.values
            [*self.slots.first().ok_or(Error::Binding)?]
        .as_ref()
        .ok_or(Error::Binding)?
        .first
        .as_ptr())
    }
}

impl Drop for Installed<'_, '_> {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            worker.cancel();
        }
        // Read the actual borrowed source owner AFTER erasure. Besides making
        // lifecycle ordering explicit, Drop keeps Rust dropck's source borrow
        // alive until cancellation, including unwinding and unused receipts.
        if self
            .workers
            .iter()
            .all(|w| w.retired() && w.values.is_empty())
            && Arc::strong_count(self.pending.source().execution_origin()) > 0
        {
            ERASED_WHILE_SOURCE_LIVE.with(|n| n.set(n.get() + 1));
        }
    }
}

thread_local! { pub(in crate::replicated_decoder) static COPY_FACTORIES:std::cell::Cell<usize>=const { std::cell::Cell::new(0) }; }
thread_local! { pub(in crate::replicated_decoder) static ERASED_WHILE_SOURCE_LIVE:std::cell::Cell<usize>=const { std::cell::Cell::new(0) }; }
