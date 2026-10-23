//! D27 process-local ownership split. No serialization or new numeric boundary.
use super::super::role_coordination::{self as custody, arc_bytes, SplitControl};
use super::*;
use std::sync::{Arc, Mutex, MutexGuard};

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

pub fn role_split_resources() -> Result<LinearActivationRoleSplitResources, Error> {
    role_split_resources_for::<&Budget>()
}
pub fn role_split_resources_for<B: AllocationContext>(
) -> Result<LinearActivationRoleSplitResources, Error> {
    let role_backing_bytes = arc_bytes::<RoleSlotFor<B>>()?;
    let coordination_backing_bytes = arc_bytes::<CoordinationFor<B>>()?;
    let wrapper_population_bytes = metadata_sum(&[
        size_of::<LinearActivationRoleCoordinatorFor<B>>(),
        2 * size_of::<LinearActivationRoleFor<B>>(),
        2 * EXCHANGES * size_of::<LinearActivationRoleFrameFor<B>>(),
        2 * EXCHANGES * size_of::<LinearActivationRoleStepFor<B>>(),
        2 * size_of::<LinearActivationRoleOutputFor<B>>(),
    ])?;
    // Full origin/destination slots, conservatively including original native
    // fields already quoted by D22. No guessed spare bytes or per-node allowance.
    let transition_bytes = metadata_sum(&[
        size_of::<LinearActivationRoleSplitResources>(),
        size_of::<Result<LinearActivationRoleSplitResources, Error>>(),
        size_of::<LinearActivationRunFor<B>>(),
        2 * size_of::<RoleSlotFor<B>>(),
        size_of::<CoordinationFor<B>>(),
        size_of::<SplitControl>(),
        size_of::<[Arc<RoleSlotFor<B>>; 2]>(),
        size_of::<[LinearActivationRoleFor<B>; 2]>(),
        size_of::<RoleSplitResultFor<B>>(),
        size_of::<Result<RoleSplitResultFor<B>, Error>>(),
        size_of::<OperationFor<'_, B>>(),
        size_of::<CallbackPollFor<'_, B>>(),
        size_of::<MutexGuard<'_, RoleStateFor<B>>>(),
        size_of::<PartyStepFor<B>>(),
        size_of::<Result<PartyStepFor<B>, Error>>(),
        size_of::<Result<LinearActivationFrameFor<B>, Error>>(),
        size_of::<Result<LinearActivationRoleFrameFor<B>, Error>>(),
        size_of::<Result<LinearActivationRoleStepFor<B>, Error>>(),
        size_of::<Result<LinearActivationRoleOutputFor<B>, Error>>(),
        size_of::<Result<EdgeDescriptor, Error>>(),
        size_of::<Result<&'static [u8], Error>>(),
        size_of::<Result<(), Error>>(),
        4 * size_of::<Arc<CoordinationFor<B>>>(),
        3 * size_of::<LeaseFor<B>>(),
        size_of::<Option<LeaseFor<B>>>(),
        // Continuity returned by preflight and named construction context.
        2 * B::TRANSIENT_LINK_BYTES,
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

struct RoleStateFor<B: AllocationContext> {
    // Keep the original cancelled inline native wrappers charged until drop.
    party: LinearPartyFor<B>,
    output: Option<CommonMaskedFor<B, Ring>>,
}
impl<B: AllocationContext> custody::Retire for RoleStateFor<B> {
    fn retire(&mut self) {
        self.output = None;
        self.party.cancel();
    }
}
type RoleSlotFor<B> = custody::RoleSlotFor<B, RoleStateFor<B>>;
type CoordinationFor<B> = custody::CoordinationFor<B, RoleStateFor<B>>;
type OperationFor<'a, B> = custody::OperationFor<'a, B, RoleStateFor<B>>;
type CallbackPollFor<'a, B> = custody::CallbackPollFor<'a, B, RoleStateFor<B>>;

/// Local coordinator, with cancellation authority only. Dropping it aborts both
/// roles, including unclaimed/completed outputs. Keep it through output use.
pub struct LinearActivationRoleCoordinatorFor<B: AllocationContext> {
    shared: Arc<CoordinationFor<B>>,
}
impl<B: AllocationContext> LinearActivationRoleCoordinatorFor<B> {
    pub fn cancel(&mut self) {
        self.shared.abort();
    }
    pub fn is_aborted(&self) -> bool {
        self.shared.control().aborted
    }
}
impl<B: AllocationContext> Drop for LinearActivationRoleCoordinatorFor<B> {
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
pub struct LinearActivationRoleFor<B: AllocationContext> {
    slot: Arc<RoleSlotFor<B>>,
    shared: Arc<CoordinationFor<B>>,
    role: u8,
    handed_off: bool,
}

/// Retained outgoing frame. After abort, fresh byte access fails closed; slices
/// borrowed before cancellation cannot be revoked. Original guards/leases stay
/// in the native frame until drop. No material serialization authority.
pub struct LinearActivationRoleFrameFor<B: AllocationContext> {
    native: LinearActivationFrameFor<B>,
    shared: Arc<CoordinationFor<B>>,
}
impl<B: AllocationContext> LinearActivationRoleFrameFor<B> {
    pub fn as_bytes(&self) -> Result<&[u8], Error> {
        self.shared.operation(|_| Ok(self.native.as_bytes()))
    }
}

#[allow(clippy::large_enum_variant)]
pub enum LinearActivationRoleStepFor<B: AllocationContext> {
    Frame(LinearActivationRoleFrameFor<B>),
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
pub struct LinearActivationRoleOutputFor<B: AllocationContext> {
    slot: Arc<RoleSlotFor<B>>,
    shared: Arc<CoordinationFor<B>>,
    role: u8,
}
impl<B: AllocationContext> LinearActivationRoleOutputFor<B> {
    pub fn descriptor(&self) -> Result<EdgeDescriptor, Error> {
        self.shared.operation(|_| {
            self.slot
                .state
                .lock()
                .map_err(|_| Error::Consumed)?
                .output
                .as_ref()
                .map(CommonMaskedFor::descriptor)
                .ok_or(Error::Consumed)
        })
    }
    pub fn role(&self) -> u8 {
        self.role
    }
}
impl<B: AllocationContext> Drop for LinearActivationRoleOutputFor<B> {
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

impl<B: AllocationContext> LinearActivationRoleFor<B> {
    pub fn role(&self) -> u8 {
        self.role
    }
    pub fn cancel(&mut self) {
        self.shared.abort();
    }
    pub fn begin(
        &mut self,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<LinearActivationRoleFrameFor<B>, Error> {
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
            Ok(LinearActivationRoleFrameFor {
                native,
                shared: Arc::clone(&self.shared),
            })
        })
    }
    pub fn advance(
        &mut self,
        peer: &[u8],
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<LinearActivationRoleStepFor<B>, Error> {
        self.shared.poll_cancelled(cancelled)?;
        self.shared.operation(|control| {
            let mut state = self.slot.state.lock().map_err(|_| Error::Consumed)?;
            match state.party.advance(peer)? {
                PartyStepFor::Frame(native) => Ok(LinearActivationRoleStepFor::Frame(
                    LinearActivationRoleFrameFor {
                        native,
                        shared: Arc::clone(&self.shared),
                    },
                )),
                PartyStepFor::Output(output) => {
                    state.output = Some(output);
                    control.mark_ready(self.role);
                    #[cfg(test)]
                    tests::panic_after_locked_output(self.role, &state, control);
                    Ok(LinearActivationRoleStepFor::Ready)
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
    pub fn finish(mut self) -> Result<LinearActivationRoleOutputFor<B>, Error> {
        let output = self.shared.operation(|control| {
            control.require_both_ready()?;
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
            Ok(LinearActivationRoleOutputFor {
                slot: Arc::clone(&self.slot),
                shared: Arc::clone(&self.shared),
                role: self.role,
            })
        });
        self.handed_off = output.is_ok();
        output
    }
}
impl<B: AllocationContext> Drop for LinearActivationRoleFor<B> {
    fn drop(&mut self) {
        if !self.handed_off {
            self.shared.abort();
        }
    }
}

type RoleSplitResultFor<B> = (
    LinearActivationRoleCoordinatorFor<B>,
    [LinearActivationRoleFor<B>; 2],
);

impl<B: AllocationContext> LinearActivationRunFor<B> {
    /// Budget object identity only; not admission or execution authority.
    pub fn is_budget(&self, budget: &Budget) -> bool {
        self.lease
            .as_ref()
            .is_some_and(|lease| std::ptr::eq(lease.budget.budget(), budget))
    }

    fn role_split_context(&self) -> Result<&B, Error> {
        if self.parties.iter().any(|party| {
            party.phase != Phase::Fresh || party.ledger.next != 0 || party.roots.is_none()
        }) {
            return Err(Error::Consumed);
        }
        Ok(&self.lease.as_ref().ok_or(Error::Consumed)?.budget)
    }

    // Only preadmitted callers materialize a strong continuity slot here. The
    // returned value owns its context; no borrow from the consumed run escapes.
    fn role_split_preflight(&self) -> Result<B, Error> {
        Ok(self.role_split_context()?.share())
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
    pub fn into_roles(self) -> Result<RoleSplitResultFor<B>, Error> {
        let cost = role_split_resources_for::<B>()?;
        // Reserve on a short view of the still-live original run before making
        // any additional strong context link or allocating role backing.
        let additional = LeaseFor::reserve(self.role_split_context()?, cost.reservation_bytes)?;
        self.split_roles_reserved(cost, additional)
    }
}
impl<'b> LinearActivationRun<'b> {
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
}
impl<B: AllocationContext> LinearActivationRunFor<B> {
    // Deliberately preserve the qualified Arc/Mutex custody graph and its exact
    // preadmitted backing/control census. Arc supplies shared destruction here,
    // not thread permission: the original native Live marker still makes every
    // role non-Send/non-Sync. Replacing this graph with Rc/RefCell would also
    // change the established poison/abort and accounting paths unnecessarily.
    #[allow(clippy::arc_with_non_send_sync)]
    fn split_roles_reserved(
        self,
        cost: LinearActivationRoleSplitResources,
        mut additional: LeaseFor<B>,
    ) -> Result<RoleSplitResultFor<B>, Error> {
        let allocation = additional.budget.share();
        let budget = allocation.budget();
        let left_lease = additional.split(cost.role_backing_bytes)?;
        let right_lease = additional.split(cost.role_backing_bytes)?;
        let Self {
            parties: [left, right],
            lease,
        } = self;
        budget.allocation()?;
        let left = Arc::new(RoleSlotFor {
            state: Mutex::new(RoleStateFor {
                party: left,
                output: None,
            }),
            _lease: left_lease,
        });
        budget.allocation()?;
        let right = Arc::new(RoleSlotFor {
            state: Mutex::new(RoleStateFor {
                party: right,
                output: None,
            }),
            _lease: right_lease,
        });
        budget.allocation()?;
        let shared = Arc::new(CoordinationFor {
            control: Mutex::new(SplitControl::default()),
            slots: [Arc::clone(&left), Arc::clone(&right)],
            _original_run: lease,
            _additional: additional,
        });
        Ok((
            LinearActivationRoleCoordinatorFor {
                shared: Arc::clone(&shared),
            },
            [
                LinearActivationRoleFor {
                    slot: left,
                    shared: Arc::clone(&shared),
                    role: 0,
                    handed_off: false,
                },
                LinearActivationRoleFor {
                    slot: right,
                    shared,
                    role: 1,
                    handed_off: false,
                },
            ],
        ))
    }
}

pub type LinearActivationRoleCoordinator<'b> = LinearActivationRoleCoordinatorFor<&'b Budget>;
pub type LinearActivationRole<'b> = LinearActivationRoleFor<&'b Budget>;
pub type LinearActivationRoleFrame<'b> = LinearActivationRoleFrameFor<&'b Budget>;
pub type LinearActivationRoleOutput<'b> = LinearActivationRoleOutputFor<&'b Budget>;
pub type LinearActivationRoleStep<'b> = LinearActivationRoleStepFor<&'b Budget>;
type RoleSplitResult<'b> = RoleSplitResultFor<&'b Budget>;

impl LinearActivationRunFor<OwnedBudgetContext> {
    pub fn into_roles_from_reserved(
        self,
        charge: &mut OwnedExternalCharge,
    ) -> Result<RoleSplitResultFor<OwnedBudgetContext>, Error> {
        let cost = role_split_resources_for::<OwnedBudgetContext>()?;
        let continuity = self.role_split_preflight()?;
        let additional = charge.split_native(continuity.budget(), cost.reservation_bytes)?;
        self.split_roles_reserved(cost, additional)
    }
}

#[cfg(test)]
type RoleSlot<'b> = RoleSlotFor<&'b Budget>;
#[cfg(test)]
type Coordination<'b> = CoordinationFor<&'b Budget>;

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "roles_legacy_oracle_tests.rs"]
mod legacy_oracle_tests;
