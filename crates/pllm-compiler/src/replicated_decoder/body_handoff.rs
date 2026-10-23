//! Compiler-private, consuming handoff of the original decoder worker.
//!
//! Adapters own immutable interval/export admission and their executor's checked
//! completion. They must globally drain all three roles and the router before
//! begin AND restore: neither local boundary readiness nor local completion can
//! certify that drain. This module never creates another execution identity.
use super::super::{Active, BoundaryProgress};
use super::{Party, PartyRuntime};
use crate::replicated_decoder::{Code, Error, Program, Secret, Worker};
use std::{mem::size_of, sync::Arc};

/// Retain every original source, share, sticky-validity/feedback owner and
/// last-use ledger. Only the original runtime leaves this owner. Worker::drop
/// cancels on every failed transition or abandoned suspension; no failure can
/// release a partially restored worker. No extra flag, Arc or allocation.
#[repr(transparent)]
pub(in crate::replicated_decoder) struct SuspendedWorker {
    worker: Worker,
}

type Begun = Result<(SuspendedWorker, PartyRuntime, Vec<Secret>), Error>;

// The original Worker slot is already priced inside the enclosing invocation.
// Adapters must additionally price these transition/result-slot excesses and
// their nonallocating checked-export closure. Do not charge another Worker, or
// silently discount the extra returned runtime/input-vector control slots.
pub(in crate::replicated_decoder) const TRANSITION_INLINE_BYTES: usize = size_of::<Begun>()
    .saturating_sub(size_of::<Worker>())
    + size_of::<Result<Worker, Error>>().saturating_sub(size_of::<Worker>());

const _: () = assert!(size_of::<SuspendedWorker>() == size_of::<Worker>());

/// Failure-sensitive work only BORROWS these owners. On error or unwind, outputs
/// erase before either original execution owner is canceled. On success, outputs
/// erase explicitly before the original runtime/worker may leave this guard.
/// No custom Drop and no cloned runtime/source identity or detached output.
struct RestoreOwners<Outputs, Runtime, Suspended> {
    outputs: Outputs,
    runtime: Runtime,
    suspended: Suspended,
}

impl<Outputs, Runtime, Suspended> RestoreOwners<Outputs, Runtime, Suspended> {
    fn copy_then_release(
        mut self,
        copy: impl FnOnce(&Outputs, &mut Suspended) -> Result<(), Error>,
    ) -> Result<(Runtime, Suspended), Error> {
        copy(&self.outputs, &mut self.suspended)?;
        // If erasure itself unwinds, the remaining runtime/worker fields still
        // stay here until output destruction has finished unwinding.
        drop(self.outputs);
        Ok((self.runtime, self.suspended))
    }
}

/// NEW owned-restoration controls, additive to the adapter's existing quote.
/// Count complete named slots, not presumed slack in older transition reserves.
/// Caller must preflight this census BEFORE invoking restore_owned/allocation.
pub(in crate::replicated_decoder) fn owned_restore_inline_bytes<Outputs>() -> Result<usize, Error> {
    [
        size_of::<RestoreOwners<Outputs, PartyRuntime, SuspendedWorker>>(), // Guard candidate.
        size_of::<RestoreOwners<Outputs, PartyRuntime, SuspendedWorker>>(), // Guard move/callee.
        size_of::<Outputs>(), // Consumed output-owner argument.
        size_of::<Outputs>(), // Explicit output-destruction move.
        size_of::<(PartyRuntime, SuspendedWorker)>(), // Released original tuple.
        size_of::<Result<(PartyRuntime, SuspendedWorker), Error>>(),
        size_of::<Result<Worker, Error>>(), // Owned helper return.
        size_of::<Result<(), Error>>(),     // Borrowed copy/replay result.
        size_of::<&Outputs>(),              // Read-only reader borrow.
        size_of::<&mut SuspendedWorker>(),  // Borrowed restoration receiver.
    ]
    .into_iter()
    .try_fold(0usize, |sum, bytes| {
        sum.checked_add(bytes).ok_or(Error::Bounds)
    })
}

// Exercise the SAME production guard with nonnative drop probes. This is not a
// Worker/runtime fixture, drain issuer, or producer/consumer proof constructor.
#[cfg(test)]
pub(in crate::replicated_decoder) fn test_restore_owner_order<Outputs, Runtime, Suspended>(
    outputs: Outputs,
    runtime: Runtime,
    suspended: Suspended,
    copy: impl FnOnce(&Outputs, &mut Suspended) -> Result<(), Error>,
) -> Result<(Runtime, Suspended), Error> {
    RestoreOwners {
        outputs,
        runtime,
        suspended,
    }
    .copy_then_release(copy)
}

impl SuspendedWorker {
    /// Census the original worker's independently owned authenticated files.
    /// Worker currently owns PublicSource by value, not Arc: an adapter lending a
    /// separate existing Arc to its executor must charge BOTH owners, never infer
    /// aliasing from equal bindings/digests or silently reimport either source.
    pub(in crate::replicated_decoder) fn public_metadata_bytes(
        worker: &Worker,
        program: &Arc<Program>,
    ) -> Result<usize, Error> {
        worker.live()?;
        if !Arc::ptr_eq(program, &worker.program) {
            return Err(Error::Binding);
        }
        let public = worker.public.as_ref().ok_or(Error::Binding)?;
        public.check(program)?;
        public.resident_metadata_bytes()
    }

    /// Nonmutating LOCAL precondition, not a distributed drain certificate.
    /// Coordinator custody must be established separately before consuming begin.
    pub(in crate::replicated_decoder) fn check_start(
        worker: &Worker,
        program: &Arc<Program>,
        start: usize,
    ) -> Result<(), Error> {
        worker.live()?;
        // Expanded checkpoint capture requires every original retired owner.
        // Reject until a compact-body mapping exists; never fabricate captures.
        #[cfg(test)]
        if super::super::oracle_trace::enabled() {
            return Err(Error::Binding);
        }
        if !Arc::ptr_eq(program, &worker.program)
            || worker.imported != worker.program.input_count()
            || !worker.boundary_ready()
        {
            return Err(Error::Binding);
        }
        if start < program.input_count() || start >= program.logical_len() {
            return Err(Error::Bounds);
        }
        if worker.step != start && !(worker.step == 0 && start == program.input_count()) {
            return Err(Error::Phase);
        }
        Ok(())
    }

    pub(in crate::replicated_decoder) fn advance_to_start(
        program: &Arc<Program>,
        start: usize,
        worker: &mut Worker,
    ) -> Result<BoundaryProgress, Error> {
        if !Arc::ptr_eq(program, &worker.program) {
            worker.cancel();
            return Err(Error::Binding);
        }
        worker.advance_before(start)
    }

    /// Consume the exact admitted Program Arc at a complete import/local idle
    /// boundary. Input copies retain both components and are charged by the
    /// adapter, alongside all still-live original owners and export overlap.
    pub(in crate::replicated_decoder) fn begin(
        worker: Worker,
        program: &Arc<Program>,
        start: usize,
        input_ids: &[usize],
    ) -> Begun {
        let mut suspended = Self { worker };
        let worker = &mut suspended.worker;
        Self::check_start(worker, program, start)?;
        if worker.step == 0 && start == worker.program.input_count() {
            while worker.step < start {
                worker.retire_inputs();
            }
        }
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(input_ids.len())
            .map_err(|_| Error::Allocation)?;
        // Compiler mappings hold logical owners; only Worker storage uses slots.
        for &id in input_ids {
            let value = worker
                .values
                .get(worker.program.value_slot(id))
                .and_then(Option::as_ref)
                .ok_or(Error::Binding)?;
            let mut copy = Secret::new(value.first.len())?;
            copy.first.copy_from_slice(&value.first);
            copy.second.copy_from_slice(&value.second);
            inputs.push(copy);
        }
        let Active::Idle(runtime) = std::mem::replace(&mut worker.active, Active::Retired) else {
            return Err(Error::Phase);
        };
        Ok((suspended, runtime, inputs))
    }

    pub(in crate::replicated_decoder) fn party(&self) -> Party {
        self.worker.party()
    }

    pub(in crate::replicated_decoder) fn cancel(&mut self) {
        self.worker.cancel();
    }

    /// Restore only after the adapter's consuming finish returns the ORIGINAL
    /// runtime and checked, read-only exports. The closure maps (output, lane)
    /// through the adapter's admitted owner/layout and returns BOTH components;
    /// it must not allocate. End/output IDs come from immutable compiler mapping,
    /// not caller-provided descriptors. Every original node must be complete.
    pub(in crate::replicated_decoder) fn restore(
        mut self,
        runtime: PartyRuntime,
        end: usize,
        output_ids: &[usize],
        word: impl Fn(usize, usize) -> Result<(u64, u64), Error>,
    ) -> Result<Worker, Error> {
        self.copy_restore_exports(end, output_ids, word)?;
        self.worker.active = Active::Idle(runtime);
        Ok(self.worker)
    }

    /// Unlike a borrowed export closure, this consumes the complete output owner.
    /// Every fallible copy/ledger operation runs with runtime AND Worker still in
    /// the ordered guard. Only after outputs erase may either owner be moved out.
    pub(in crate::replicated_decoder) fn restore_owned<Outputs>(
        self,
        runtime: PartyRuntime,
        end: usize,
        output_ids: &[usize],
        outputs: Outputs,
        word: impl Fn(&Outputs, usize, usize) -> Result<(u64, u64), Error>,
    ) -> Result<Worker, Error> {
        let owners = RestoreOwners {
            outputs,
            runtime,
            suspended: self,
        };
        let (runtime, mut suspended) = owners.copy_then_release(|outputs, suspended| {
            suspended
                .copy_restore_exports(end, output_ids, |output, lane| word(outputs, output, lane))
        })?;
        suspended.worker.active = Active::Idle(runtime);
        Ok(suspended.worker)
    }

    /// Shared numeric/export/retirement implementation. Borrowing here is safe
    /// only inside a consuming outer transition; never expose this partial state.
    fn copy_restore_exports(
        &mut self,
        end: usize,
        output_ids: &[usize],
        word: impl Fn(usize, usize) -> Result<(u64, u64), Error>,
    ) -> Result<(), Error> {
        for (output, &id) in output_ids.iter().enumerate() {
            let mut value = Secret::new(self.worker.program.recipe(id).lanes)?;
            for i in 0..value.first.len() {
                (value.first[i], value.second[i]) = word(output, i)?;
            }
            self.worker.values[self.worker.program.value_slot(id)] = Some(value);
        }
        // Replay the original ledger only after ALL exports have been copied.
        // A whole original TensorLoop has completed all tiles, not just its first
        // tile. Set its completion cursor before the original retirement guard;
        // binary intervals never take this branch and retain their exact order.
        while self.worker.step < end {
            if matches!(
                self.worker.program.recipe(self.worker.step).code,
                Code::TensorLoop { .. }
            ) {
                self.worker.loop_offset = self.worker.program.recipe(self.worker.step).lanes;
            }
            self.worker.retire_inputs();
        }
        self.worker.started = true;
        Ok(())
    }
}
