//! D27 process-local ownership split. No serialization or new numeric boundary.
use super::*;
use std::alloc::Layout;
use std::sync::{atomic::AtomicUsize, Arc, Mutex, MutexGuard};

/// Incremental owner inventory beyond the original D22 run. Available before
/// constructing a coordination object, hashing, allocation or entropy. Includes
/// full Arc backings/counters/padding, all returned wrappers, and concrete
/// construction/transition slots; excludes borrowed peer slices and the original
/// run's already-charged native inventory. Logical storage, not stack or RSS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearActivationRoleSplitResources {
    pub role_backing_bytes: usize,
    pub coordination_backing_bytes: usize,
    pub wrapper_population_bytes: usize,
    pub transition_bytes: usize,
    pub reservation_bytes: usize,
}

fn arc_bytes<T>() -> Result<usize, Error> {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map(|(layout, _)| layout.pad_to_align().size())
        .map_err(|_| Error::Resource)
}

pub fn role_split_resources() -> Result<LinearActivationRoleSplitResources, Error> {
    let role_backing_bytes = arc_bytes::<RoleSlot<'static>>()?;
    let coordination_backing_bytes = arc_bytes::<Coordination<'static>>()?;
    let wrapper_population_bytes = metadata_sum(&[
        size_of::<LinearActivationRoleCoordinator<'static>>(),
        2 * size_of::<LinearActivationRole<'static>>(),
        2 * EXCHANGES * size_of::<LinearActivationRoleFrame<'static>>(),
        2 * EXCHANGES * size_of::<LinearActivationRoleStep<'static>>(),
        2 * size_of::<LinearActivationRoleOutput<'static>>(),
    ])?;
    // Full origin/destination slots, conservatively including original native
    // fields already quoted by D22. No guessed spare bytes or per-node allowance.
    let transition_bytes = metadata_sum(&[
        size_of::<LinearActivationRoleSplitResources>(),
        size_of::<Result<LinearActivationRoleSplitResources, Error>>(),
        size_of::<LinearActivationRun<'static>>(),
        2 * size_of::<RoleSlot<'static>>(),
        size_of::<Coordination<'static>>(),
        size_of::<SplitControl>(),
        size_of::<[Arc<RoleSlot<'static>>; 2]>(),
        size_of::<[LinearActivationRole<'static>; 2]>(),
        size_of::<RoleSplitResult<'static>>(),
        size_of::<Result<RoleSplitResult<'static>, Error>>(),
        size_of::<Operation<'static, 'static>>(),
        size_of::<CallbackPoll<'static, 'static>>(),
        size_of::<MutexGuard<'static, RoleState<'static>>>(),
        size_of::<PartyStep<'static>>(),
        size_of::<Result<PartyStep<'static>, Error>>(),
        size_of::<Result<LinearActivationFrame<'static>, Error>>(),
        size_of::<Result<LinearActivationRoleFrame<'static>, Error>>(),
        size_of::<Result<LinearActivationRoleStep<'static>, Error>>(),
        size_of::<Result<LinearActivationRoleOutput<'static>, Error>>(),
        size_of::<Result<EdgeDescriptor, Error>>(),
        size_of::<Result<&'static [u8], Error>>(),
        size_of::<Result<(), Error>>(),
        4 * size_of::<Arc<Coordination<'static>>>(),
        3 * size_of::<Lease<'static>>(),
        size_of::<Option<Lease<'static>>>(),
    ])?;
    let reservation_bytes = metadata_sum(&[
        role_backing_bytes,
        role_backing_bytes,
        coordination_backing_bytes,
        wrapper_population_bytes,
        transition_bytes,
    ])?;
    Ok(LinearActivationRoleSplitResources {
        role_backing_bytes,
        coordination_backing_bytes,
        wrapper_population_bytes,
        transition_bytes,
        reservation_bytes,
    })
}

struct RoleState<'b> {
    // Keep the original cancelled inline native wrappers charged until drop.
    party: LinearParty<'b>,
    output: Option<CommonMasked<'b, Ring>>,
}
struct RoleSlot<'b> {
    state: Mutex<RoleState<'b>>,
    _lease: Lease<'b>,
}
#[derive(Default)]
struct SplitControl {
    aborted: bool,
    ready: [bool; 2],
}

/// Cancellation-only registry. Private strong links retain allocation accounting
/// until the registry drops (Weak links would retain uncharged Arc allocations).
/// No public method can evaluate, borrow, extract or import a sibling's state.
struct Coordination<'b> {
    control: Mutex<SplitControl>,
    slots: [Arc<RoleSlot<'b>>; 2],
    _original_run: Option<Lease<'b>>,
    _additional: Lease<'b>,
}

impl Coordination<'_> {
    fn abort_locked(&self, control: &mut SplitControl) {
        control.aborted = true;
        for slot in &self.slots {
            let mut state = slot.state.lock().unwrap_or_else(|p| p.into_inner());
            state.output = None;
            state.party.cancel();
        }
    }

    fn control(&self) -> MutexGuard<'_, SplitControl> {
        match self.control.lock() {
            Ok(control) => control,
            Err(poisoned) => {
                let mut control = poisoned.into_inner();
                self.abort_locked(&mut control);
                control
            }
        }
    }

    fn abort(&self) {
        self.abort_locked(&mut self.control());
    }

    /// Caller code must never run under either mutex: callbacks may inspect or
    /// cancel this same coordinator. An unwind burns both before propagating.
    fn poll_cancelled(&self, cancelled: &mut impl FnMut() -> bool) -> Result<(), Error> {
        let aborted = {
            let control = self.control();
            control.aborted
        };
        if aborted {
            return Err(Error::Consumed);
        }
        let mut poll = CallbackPoll {
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

    fn operation<T>(
        &self,
        f: impl FnOnce(&mut SplitControl) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut operation = Operation {
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

// If locked native work panics, the role-state lock unwinds first; this guard
// then burns BOTH continuations before releasing the coordination lock. Caller
// callbacks use CallbackPoll instead and never execute under either mutex.
struct Operation<'a, 'b> {
    coordination: &'a Coordination<'b>,
    control: MutexGuard<'a, SplitControl>,
    armed: bool,
}

struct CallbackPoll<'a, 'b> {
    coordination: &'a Coordination<'b>,
    armed: bool,
}
impl Drop for CallbackPoll<'_, '_> {
    fn drop(&mut self) {
        if self.armed {
            self.coordination.abort();
        }
    }
}
impl Drop for Operation<'_, '_> {
    fn drop(&mut self) {
        if self.armed {
            self.coordination.abort_locked(&mut self.control);
        }
    }
}

/// Local coordinator, with cancellation authority only. Dropping it aborts both
/// roles, including unclaimed/completed outputs. Keep it through output use.
pub struct LinearActivationRoleCoordinator<'b> {
    shared: Arc<Coordination<'b>>,
}
impl LinearActivationRoleCoordinator<'_> {
    pub fn cancel(&mut self) {
        self.shared.abort();
    }
    pub fn is_aborted(&self) -> bool {
        self.shared.control().aborted
    }
}
impl Drop for LinearActivationRoleCoordinator<'_> {
    fn drop(&mut self) {
        self.shared.abort();
    }
}

/// Unique evaluator handle for exactly one role. Neither sibling state nor a
/// raw typed owner escapes. Dropping an unfinished handle aborts BOTH roles.
/// The original native one-thread/local profile remains in force; Arc/Mutex
/// here implement synchronous retirement, not authenticated remote deployment.
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::LocalLinearActivationRole;
/// fn duplicate(role: LocalLinearActivationRole<'_>) { let _ = role.clone(); }
/// ```
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::LocalLinearActivationRole;
/// fn escape(role: LocalLinearActivationRole<'_>) { let _ = role.slot; }
/// ```
pub struct LinearActivationRole<'b> {
    slot: Arc<RoleSlot<'b>>,
    shared: Arc<Coordination<'b>>,
    role: u8,
    handed_off: bool,
}

/// Retained outgoing frame. After abort, fresh byte access fails closed; slices
/// borrowed before cancellation cannot be revoked. Original guards/leases stay
/// in the native frame until drop. No material serialization authority.
pub struct LinearActivationRoleFrame<'b> {
    native: LinearActivationFrame<'b>,
    shared: Arc<Coordination<'b>>,
}
impl LinearActivationRoleFrame<'_> {
    pub fn as_bytes(&self) -> Result<&[u8], Error> {
        self.shared.operation(|_| Ok(self.native.as_bytes()))
    }
}

#[allow(clippy::large_enum_variant)]
pub enum LinearActivationRoleStep<'b> {
    Frame(LinearActivationRoleFrame<'b>),
    /// Local computation complete, not permission to extract an output. Both
    /// peers must be Ready before either handle may be consumed by finish.
    Ready,
}

/// Opaque final local conditional output. Can outlive the coordinator, but its
/// descriptor then fails closed. Abort erases its actual mailbox-owned backing.
/// No raw CommonMasked, dealer mask, share or capability extraction is exposed.
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::LocalLinearActivationRoleOutput;
/// fn escape(output: LocalLinearActivationRoleOutput<'_>) { let _ = output.slot; }
/// ```
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::LocalLinearActivationRoleOutput;
/// fn duplicate(output: LocalLinearActivationRoleOutput<'_>) { let _ = output.clone(); }
/// ```
pub struct LinearActivationRoleOutput<'b> {
    slot: Arc<RoleSlot<'b>>,
    shared: Arc<Coordination<'b>>,
    role: u8,
}
impl LinearActivationRoleOutput<'_> {
    pub fn descriptor(&self) -> Result<EdgeDescriptor, Error> {
        self.shared.operation(|_| {
            self.slot
                .state
                .lock()
                .map_err(|_| Error::Consumed)?
                .output
                .as_ref()
                .map(CommonMasked::descriptor)
                .ok_or(Error::Consumed)
        })
    }
    pub fn role(&self) -> u8 {
        self.role
    }
}
impl Drop for LinearActivationRoleOutput<'_> {
    fn drop(&mut self) {
        // Serialize output retirement with abort/evaluation. No accounting guard
        // is detached; the allocation and original metadata remain retained.
        let _control = self.shared.control();
        self.slot
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .output = None;
    }
}

impl<'b> LinearActivationRole<'b> {
    pub fn role(&self) -> u8 {
        self.role
    }
    pub fn cancel(&mut self) {
        self.shared.abort();
    }
    pub fn begin(
        &mut self,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<LinearActivationRoleFrame<'b>, Error> {
        self.shared.poll_cancelled(cancelled)?;
        self.shared.operation(|control| {
            let native = self
                .slot
                .state
                .lock()
                .map_err(|_| Error::Consumed)?
                .party
                // Native matrix cancellation polls only local coordinator
                // state, never arbitrary caller code while locks are held.
                .begin(&mut || control.aborted)?;
            Ok(LinearActivationRoleFrame {
                native,
                shared: Arc::clone(&self.shared),
            })
        })
    }
    pub fn advance(
        &mut self,
        peer: &[u8],
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<LinearActivationRoleStep<'b>, Error> {
        self.shared.poll_cancelled(cancelled)?;
        self.shared.operation(|control| {
            let mut state = self.slot.state.lock().map_err(|_| Error::Consumed)?;
            match state.party.advance(peer)? {
                PartyStep::Frame(native) => {
                    Ok(LinearActivationRoleStep::Frame(LinearActivationRoleFrame {
                        native,
                        shared: Arc::clone(&self.shared),
                    }))
                }
                PartyStep::Output(output) => {
                    state.output = Some(output);
                    control.ready[usize::from(self.role)] = true;
                    #[cfg(test)]
                    tests::panic_after_locked_output(self.role, &state, control);
                    Ok(LinearActivationRoleStep::Ready)
                }
            }
        })
    }

    /// Consume once, only after BOTH local roles are ready. Early/repeated/wrong
    /// schedule attempts burn both. No ordinary numeric continuation is added.
    ///
    /// ```compile_fail
    /// use pllm_garble::masked_wire_reference::typed::LocalLinearActivationRole;
    /// fn reuse(role: LocalLinearActivationRole<'_>) {
    ///     let _ = role.finish();
    ///     let _ = role.finish();
    /// }
    /// ```
    pub fn finish(mut self) -> Result<LinearActivationRoleOutput<'b>, Error> {
        let output = self.shared.operation(|control| {
            if control.ready != [true; 2] {
                return Err(Error::Consumed);
            }
            if self
                .slot
                .state
                .lock()
                .map_err(|_| Error::Consumed)?
                .output
                .is_none()
            {
                return Err(Error::Consumed);
            }
            Ok(LinearActivationRoleOutput {
                slot: Arc::clone(&self.slot),
                shared: Arc::clone(&self.shared),
                role: self.role,
            })
        });
        self.handed_off = output.is_ok();
        output
    }
}
impl Drop for LinearActivationRole<'_> {
    fn drop(&mut self) {
        if !self.handed_off {
            self.shared.abort();
        }
    }
}

type RoleSplitResult<'b> = (
    LinearActivationRoleCoordinator<'b>,
    [LinearActivationRole<'b>; 2],
);

impl<'b> LinearActivationRun<'b> {
    /// Budget object identity only; not admission or execution authority.
    pub fn is_budget(&self, budget: &Budget) -> bool {
        self.lease
            .as_ref()
            .is_some_and(|lease| std::ptr::eq(lease.budget, budget))
    }

    fn role_split_preflight(&self) -> Result<&'b Budget, Error> {
        if self.parties.iter().any(|party| {
            party.phase != Phase::Fresh || party.ledger.next != 0 || party.roots.is_none()
        }) {
            return Err(Error::Consumed);
        }
        Ok(self.lease.as_ref().ok_or(Error::Consumed)?.budget)
    }

    /// Consume a FRESH local run into distinct party-local evaluators. The
    /// coordinator has cancellation authority, never evaluation/extraction.
    /// Reserve all added owners before any allocation; no entropy or payload
    /// copies. Any rejection consumes/drops the original run, never permits retry.
    ///
    /// ```compile_fail
    /// use pllm_garble::masked_wire_reference::typed::LocalLinearActivationRun;
    /// fn reuse(run: LocalLinearActivationRun<'_>) {
    ///     let _ = run.into_roles();
    ///     let _ = run.into_roles();
    /// }
    /// ```
    pub fn into_roles(self) -> Result<RoleSplitResult<'b>, Error> {
        let cost = role_split_resources()?;
        let budget = self.role_split_preflight()?;
        let additional = budget.reserve(cost.reservation_bytes)?;
        self.split_roles_reserved(cost, additional)
    }

    /// Same split, transferring its exact incremental slice from a preadmitted
    /// guard on the ORIGINAL budget. No refund/reacquire, imported authority or
    /// caller-specified allowance. Original native/source charges stay intact.
    pub fn into_roles_from_reserved(
        self,
        charge: &mut ExternalCharge<'b>,
    ) -> Result<RoleSplitResult<'b>, Error> {
        let cost = role_split_resources()?;
        let budget = self.role_split_preflight()?;
        let additional = charge.split_payload(budget, cost.reservation_bytes)?;
        self.split_roles_reserved(cost, additional)
    }

    fn split_roles_reserved(
        self,
        cost: LinearActivationRoleSplitResources,
        mut additional: Lease<'b>,
    ) -> Result<RoleSplitResult<'b>, Error> {
        let budget = additional.budget;
        let left_lease = additional.split(cost.role_backing_bytes)?;
        let right_lease = additional.split(cost.role_backing_bytes)?;
        let Self {
            parties: [left, right],
            lease,
        } = self;
        budget.allocation()?;
        let left = Arc::new(RoleSlot {
            state: Mutex::new(RoleState {
                party: left,
                output: None,
            }),
            _lease: left_lease,
        });
        budget.allocation()?;
        let right = Arc::new(RoleSlot {
            state: Mutex::new(RoleState {
                party: right,
                output: None,
            }),
            _lease: right_lease,
        });
        budget.allocation()?;
        let shared = Arc::new(Coordination {
            control: Mutex::new(SplitControl::default()),
            slots: [Arc::clone(&left), Arc::clone(&right)],
            _original_run: lease,
            _additional: additional,
        });
        Ok((
            LinearActivationRoleCoordinator {
                shared: Arc::clone(&shared),
            },
            [
                LinearActivationRole {
                    slot: left,
                    shared: Arc::clone(&shared),
                    role: 0,
                    handed_off: false,
                },
                LinearActivationRole {
                    slot: right,
                    shared,
                    role: 1,
                    handed_off: false,
                },
            ],
        ))
    }
}

#[cfg(test)]
mod tests;
