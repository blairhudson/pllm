//! D27 source-retaining local role facade. No SDK or authenticated role transport.
use super::*;

/// Incremental logical storage beyond the already-admitted source/native run.
/// Pure preconstruction query: no allocation, hashing, source copying or entropy.
/// Original source/import/compiler reservations remain continuously retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SigmaExecutionRoleSplitResources {
    pub native_reservation_bytes: usize,
    pub retention_backing_bytes: usize,
    pub wrapper_population_bytes: usize,
    pub transition_bytes: usize,
    /// Additional guard payload; its own metadata is charged by Budget.
    pub additional_payload_bytes: usize,
}

pub fn sigma_execution_role_split_resources(
) -> Result<SigmaExecutionRoleSplitResources, SigmaExecutionError> {
    role_split_resources_for::<&Budget>()
}

pub fn owned_sigma_execution_role_split_resources(
) -> Result<SigmaExecutionRoleSplitResources, SigmaExecutionError> {
    role_split_resources_for::<OwnedBudgetContext>()
}

fn role_split_resources_for<C: SigmaExecutionContext>(
) -> Result<SigmaExecutionRoleSplitResources, SigmaExecutionError> {
    let native = typed::local_linear_activation_role_split_resources_for::<C>()
        .map_err(SigmaExecutionError::Native)?;
    let retention_backing_bytes = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<RetainedRolesFor<C>>())
        .map(|(layout, _)| layout.pad_to_align().size())
        .map_err(|_| SigmaExecutionError::Resource)?;
    let wrapper_population_bytes = execution_sum([
        size_of::<SigmaExecutionRoleCoordinatorFor<C>>(),
        2 * size_of::<SigmaExecutionRoleFor<C>>(),
        22 * size_of::<SigmaExecutionRoleFrameFor<C>>(),
        22 * size_of::<C::RoleStep>(),
        2 * size_of::<SigmaExecutionRoleOutputFor<C>>(),
    ])?;
    // Conservative named logical transition census, including this quote's
    // return slot. This is not an ABI-stack, allocator or RSS measurement.
    let transition_bytes = execution_sum([
        size_of::<SigmaExecutionRoleSplitResources>(),
        size_of::<Result<SigmaExecutionRoleSplitResources, SigmaExecutionError>>(),
        size_of::<typed::LocalLinearActivationRoleSplitResources>(),
        size_of::<RoleAdmissionFor<C>>(),
        size_of::<SigmaExecutionRunFor<C>>(),
        size_of::<RetainedRolesFor<C>>(),
        size_of::<typed::LocalLinearActivationRunFor<C>>(),
        size_of::<NativeRolesFor<C>>(),
        size_of::<Result<NativeRolesFor<C>, typed::Error>>(),
        size_of::<CompilerRolesFor<C>>(),
        size_of::<Result<CompilerRolesFor<C>, SigmaExecutionError>>(),
        size_of::<Result<typed::LocalLinearActivationRoleFrameFor<C>, typed::Error>>(),
        size_of::<Result<typed::LocalLinearActivationRoleStepFor<C>, typed::Error>>(),
        size_of::<Result<typed::LocalLinearActivationRoleOutputFor<C>, typed::Error>>(),
        size_of::<Result<SigmaExecutionRoleFrameFor<C>, SigmaExecutionError>>(),
        size_of::<Result<C::RoleStep, SigmaExecutionError>>(),
        size_of::<Result<SigmaExecutionRoleOutputFor<C>, SigmaExecutionError>>(),
        size_of::<Result<EdgeDescriptor, SigmaExecutionError>>(),
        4 * size_of::<Arc<RetainedRolesFor<C>>>(),
        size_of::<Arc<RetainedSourceFor<C>>>(),
        size_of::<std::array::IntoIter<usize, 5>>(),
        size_of::<std::array::IntoIter<usize, 22>>(),
    ])?;
    let transition_bytes = if C::OWNED {
        execution_sum([
            transition_bytes,
            execution_sum([
                size_of::<C::Charge>(),
                size_of::<Result<C::Charge, typed::Error>>(),
                size_of::<Result<C::Charge, SigmaExecutionError>>(),
                size_of::<std::array::IntoIter<usize, 4>>(),
            ])?,
            size_of::<std::array::IntoIter<usize, 3>>(),
        ])?
    } else {
        transition_bytes
    };
    let additional_payload_bytes = execution_sum([
        native.reservation_bytes,
        retention_backing_bytes,
        wrapper_population_bytes,
        transition_bytes,
    ])?;
    Ok(SigmaExecutionRoleSplitResources {
        native_reservation_bytes: native.reservation_bytes,
        retention_backing_bytes,
        wrapper_population_bytes,
        transition_bytes,
        additional_payload_bytes,
    })
}

#[doc(hidden)]
pub type NativeRolesFor<C> = (
    typed::LocalLinearActivationRoleCoordinatorFor<C>,
    [typed::LocalLinearActivationRoleFor<C>; 2],
);
type CompilerRolesFor<C> = (
    SigmaExecutionRoleCoordinatorFor<C>,
    [SigmaExecutionRoleFor<C>; 2],
);
#[cfg(test)]
type NativeRoles<'b> = NativeRolesFor<&'b Budget>;
type CompilerRoles<'b> = CompilerRolesFor<&'b Budget>;

struct RoleAdmissionFor<C: SigmaExecutionContext> {
    // Original native/source owners retire before the new accounting guard on
    // every rejection. No source or coefficient backing is copied or detached.
    run: Option<SigmaExecutionRunFor<C>>,
    charge: Option<C::Charge>,
}
struct RetainedRolesFor<C: SigmaExecutionContext> {
    _source: Arc<RetainedSourceFor<C>>,
    _charge: C::Charge,
}
#[cfg(test)]
type RoleAdmission<'b> = RoleAdmissionFor<&'b Budget>;
#[cfg(test)]
type RetainedRoles<'b> = RetainedRolesFor<&'b Budget>;

/// Source-retaining local coordinator. Native cancellation burns both original
/// role continuations immediately; callers need not remember sibling retirement.
/// Keep alive until outputs retire. Dropping it aborts both, even if ready.
pub struct SigmaExecutionRoleCoordinatorFor<C: SigmaExecutionContext> {
    native: typed::LocalLinearActivationRoleCoordinatorFor<C>,
    _retained: Arc<RetainedRolesFor<C>>,
}
impl<C: SigmaExecutionContext> SigmaExecutionRoleCoordinatorFor<C> {
    pub fn cancel(&mut self) {
        self.native.cancel();
    }
    pub fn is_aborted(&self) -> bool {
        self.native.is_aborted()
    }
    pub fn provenance(&self) -> SigmaExecutionProvenance {
        SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution
    }
}

/// Exactly one evaluator role; neither sibling owners nor raw native capability
/// extraction exists. Cancellation, protocol error and unfinished drop burn both.
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::SigmaExecutionRole;
/// fn duplicate(role: SigmaExecutionRole<'_>) { let _ = role.clone(); }
/// ```
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::SigmaExecutionRole;
/// fn escape(role: SigmaExecutionRole<'_>) { let _ = role.native; }
/// ```
pub struct SigmaExecutionRoleFor<C: SigmaExecutionContext> {
    native: typed::LocalLinearActivationRoleFor<C>,
    retained: Arc<RetainedRolesFor<C>>,
}

pub struct SigmaExecutionRoleFrameFor<C: SigmaExecutionContext> {
    native: typed::LocalLinearActivationRoleFrameFor<C>,
    _retained: Arc<RetainedRolesFor<C>>,
}
impl<C: SigmaExecutionContext> SigmaExecutionRoleFrameFor<C> {
    pub fn as_bytes(&self) -> Result<&[u8], SigmaExecutionError> {
        self.native.as_bytes().map_err(SigmaExecutionError::Native)
    }
}

#[allow(clippy::large_enum_variant)]
pub enum SigmaExecutionRoleStepFor<C: SigmaExecutionContext> {
    Frame(SigmaExecutionRoleFrameFor<C>),
    Ready,
}

#[allow(clippy::large_enum_variant)]
pub enum SigmaExecutionRoleStep<'b> {
    Frame(SigmaExecutionRoleFrame<'b>),
    Ready,
}

/// Opaque revocable local output, not an ordinary decoder output or source-proof
/// elevation. Its original native backing is erased on coordinated abort.
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::SigmaExecutionRoleOutput;
/// fn escape(output: SigmaExecutionRoleOutput<'_>) { let _ = output.native; }
/// ```
///
/// ```compile_fail
/// use pllm_compiler::sigma_linear_activation::SigmaExecutionRoleOutput;
/// fn duplicate(output: SigmaExecutionRoleOutput<'_>) { let _ = output.clone(); }
/// ```
pub struct SigmaExecutionRoleOutputFor<C: SigmaExecutionContext> {
    native: typed::LocalLinearActivationRoleOutputFor<C>,
    _retained: Arc<RetainedRolesFor<C>>,
}

pub type SigmaExecutionRoleCoordinator<'b> = SigmaExecutionRoleCoordinatorFor<&'b Budget>;
pub type SigmaExecutionRole<'b> = SigmaExecutionRoleFor<&'b Budget>;
pub type SigmaExecutionRoleFrame<'b> = SigmaExecutionRoleFrameFor<&'b Budget>;
pub type SigmaExecutionRoleOutput<'b> = SigmaExecutionRoleOutputFor<&'b Budget>;
/// Owning abort authority; dropping it revokes even finished outputs.
///
/// ```compile_fail,E0599
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleCoordinator;
/// fn duplicate(coordinator: OwnedSigmaExecutionRoleCoordinator) { let _ = coordinator.clone(); }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleCoordinator;
/// fn extract(coordinator: OwnedSigmaExecutionRoleCoordinator) { let _ = coordinator.native; }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleCoordinator;
/// fn extract_guard(coordinator: OwnedSigmaExecutionRoleCoordinator) {
///     let _ = coordinator._retained._charge;
/// }
/// ```
pub type OwnedSigmaExecutionRoleCoordinator = SigmaExecutionRoleCoordinatorFor<OwnedBudgetContext>;
/// One owned evaluator, not a clone, native escape or borrowed-role upgrade.
/// Finishing consumes its sole public authority, even when finishing fails.
///
/// ```compile_fail,E0599
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRole;
/// fn duplicate(role: OwnedSigmaExecutionRole) { let _ = role.clone(); }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRole;
/// fn extract(role: OwnedSigmaExecutionRole) { let _ = role.native; }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRole;
/// fn extract_guard(role: OwnedSigmaExecutionRole) { let _ = role.retained._charge; }
/// ```
///
/// ```compile_fail,E0382
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRole;
/// fn reuse(role: OwnedSigmaExecutionRole) {
///     let _ = role.finish();
///     let _ = role.finish();
/// }
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRole;
/// fn needs_send<T: Send>() {}
/// needs_send::<OwnedSigmaExecutionRole>();
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRole;
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<OwnedSigmaExecutionRole>();
/// ```
///
/// ```compile_fail,E0308
/// use pllm_compiler::sigma_linear_activation::{SigmaExecutionRole, OwnedSigmaExecutionRole};
/// fn upgrade(role: SigmaExecutionRole<'_>) -> OwnedSigmaExecutionRole { role }
/// ```
pub type OwnedSigmaExecutionRole = SigmaExecutionRoleFor<OwnedBudgetContext>;
/// Revocable frame retains the original owning source and role reservations.
///
/// ```compile_fail,E0599
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleFrame;
/// fn duplicate(frame: OwnedSigmaExecutionRoleFrame) { let _ = frame.clone(); }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleFrame;
/// fn extract(frame: OwnedSigmaExecutionRoleFrame) { let _ = frame.native; }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleFrame;
/// fn extract_guard(frame: OwnedSigmaExecutionRoleFrame) { let _ = frame._retained._charge; }
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleFrame;
/// fn needs_send<T: Send>() {}
/// needs_send::<OwnedSigmaExecutionRoleFrame>();
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleFrame;
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<OwnedSigmaExecutionRoleFrame>();
/// ```
///
/// ```compile_fail,E0308
/// use pllm_compiler::sigma_linear_activation::{SigmaExecutionRoleFrame, OwnedSigmaExecutionRoleFrame};
/// fn upgrade(frame: SigmaExecutionRoleFrame<'_>) -> OwnedSigmaExecutionRoleFrame { frame }
/// ```
pub type OwnedSigmaExecutionRoleFrame = SigmaExecutionRoleFrameFor<OwnedBudgetContext>;
/// Opaque local output, revocable even after both evaluators have finished.
/// Owning storage grants no cloning, extraction or thread-transfer permission.
///
/// ```compile_fail,E0599
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleOutput;
/// fn duplicate(output: OwnedSigmaExecutionRoleOutput) { let _ = output.clone(); }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleOutput;
/// fn extract(output: OwnedSigmaExecutionRoleOutput) { let _ = output.native; }
/// ```
///
/// ```compile_fail,E0616
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleOutput;
/// fn extract_guard(output: OwnedSigmaExecutionRoleOutput) { let _ = output._retained._charge; }
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleOutput;
/// fn needs_send<T: Send>() {}
/// needs_send::<OwnedSigmaExecutionRoleOutput>();
/// ```
///
/// ```compile_fail,E0277
/// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRoleOutput;
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<OwnedSigmaExecutionRoleOutput>();
/// ```
///
/// ```compile_fail,E0308
/// use pllm_compiler::sigma_linear_activation::{SigmaExecutionRoleOutput, OwnedSigmaExecutionRoleOutput};
/// fn upgrade(output: SigmaExecutionRoleOutput<'_>) -> OwnedSigmaExecutionRoleOutput { output }
/// ```
pub type OwnedSigmaExecutionRoleOutput = SigmaExecutionRoleOutputFor<OwnedBudgetContext>;
pub type OwnedSigmaExecutionRoleStep = SigmaExecutionRoleStepFor<OwnedBudgetContext>;

impl<C: SigmaExecutionContext> SigmaExecutionRoleOutputFor<C> {
    pub fn descriptor(&self) -> Result<EdgeDescriptor, SigmaExecutionError> {
        self.native
            .descriptor()
            .map_err(SigmaExecutionError::Native)
    }
    pub fn role(&self) -> u8 {
        self.native.role()
    }
    pub fn provenance(&self) -> SigmaExecutionProvenance {
        SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution
    }
}

impl<C: SigmaExecutionContext> SigmaExecutionRoleFor<C> {
    pub fn role(&self) -> u8 {
        self.native.role()
    }
    pub fn provenance(&self) -> SigmaExecutionProvenance {
        SigmaExecutionProvenance::PinnedSourceBytesWithLocalConditionalExecution
    }
    pub fn cancel(&mut self) {
        self.native.cancel();
    }
    pub fn begin(
        &mut self,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<SigmaExecutionRoleFrameFor<C>, SigmaExecutionError> {
        self.native
            .begin(cancelled)
            .map(|native| SigmaExecutionRoleFrameFor {
                native,
                _retained: Arc::clone(&self.retained),
            })
            .map_err(SigmaExecutionError::Native)
    }
    pub fn advance(
        &mut self,
        peer: &[u8],
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<C::RoleStep, SigmaExecutionError> {
        match self.native.advance(peer, cancelled) {
            Ok(typed::LocalLinearActivationRoleStepFor::Frame(native)) => {
                Ok(C::role_frame_step(SigmaExecutionRoleFrameFor {
                    native,
                    _retained: Arc::clone(&self.retained),
                }))
            }
            Ok(typed::LocalLinearActivationRoleStepFor::Ready) => Ok(C::role_ready_step()),
            Err(error) => Err(SigmaExecutionError::Native(error)),
        }
    }

    /// Consume once, after both role computations are Ready; early finish burns
    /// both siblings. No post-D22 numeric operation or native owner escape.
    ///
    /// ```compile_fail
    /// use pllm_compiler::sigma_linear_activation::SigmaExecutionRole;
    /// fn reuse(role: SigmaExecutionRole<'_>) {
    ///     let _ = role.finish();
    ///     let _ = role.finish();
    /// }
    /// ```
    pub fn finish(self) -> Result<SigmaExecutionRoleOutputFor<C>, SigmaExecutionError> {
        let Self { native, retained } = self;
        native
            .finish()
            .map(|native| SigmaExecutionRoleOutputFor {
                native,
                _retained: retained,
            })
            .map_err(SigmaExecutionError::Native)
    }
}

impl<'b> SigmaExecutionRun<'b> {
    /// Consume a FRESH source-checked run into two local role handles and an
    /// abort-only coordinator. Checks the original budget before reserving all
    /// added owners. Native receives its exact slice from this guard with no
    /// release/reacquire; source and original ExternalCharge remain retained.
    /// Not a deployment/router API: peer frames remain unauthenticated local data.
    ///
    /// ```compile_fail
    /// use pllm_compiler::sigma_linear_activation::SigmaExecutionRun;
    /// use pllm_garble::shared_allocation::Budget;
    /// fn reuse<'b>(run: SigmaExecutionRun<'b>, budget: &'b Budget) {
    ///     let _ = run.into_roles(budget);
    ///     let _ = run.into_roles(budget);
    /// }
    /// ```
    pub fn into_roles(self, budget: &'b Budget) -> Result<CompilerRoles<'b>, SigmaExecutionError> {
        self.into_roles_with_root(&budget)
    }
}

impl OwnedSigmaExecutionRun {
    /// Same both-Ready, revocable local role split; original owning counter only.
    /// Any rejection consumes the run; no retry or borrowed-owner upgrade exists.
    ///
    /// ```compile_fail,E0382
    /// use pllm_compiler::sigma_linear_activation::OwnedSigmaExecutionRun;
    /// use pllm_garble::shared_allocation::OwnedBudget;
    /// fn reuse(run: OwnedSigmaExecutionRun, budget: &OwnedBudget) {
    ///     let _ = run.into_roles(budget);
    ///     let _ = run.into_roles(budget);
    /// }
    /// ```
    pub fn into_roles(
        self,
        budget: &OwnedBudget,
    ) -> Result<CompilerRolesFor<OwnedBudgetContext>, SigmaExecutionError> {
        self.into_roles_with_root(budget)
    }
}

impl<C: SigmaExecutionContext> SigmaExecutionRunFor<C> {
    fn into_roles_with_root(
        self,
        root: &C::Root,
    ) -> Result<CompilerRolesFor<C>, SigmaExecutionError> {
        let budget = C::root_budget(root);
        if !self
            .native
            .as_ref()
            .ok_or(SigmaExecutionError::Native(typed::Error::Consumed))?
            .is_budget(budget)
        {
            return Err(SigmaExecutionError::BudgetIdentity);
        }
        let quote = role_split_resources_for::<C>()?;
        let charge = C::reserve(root, quote.additional_payload_bytes)
            .map_err(SigmaExecutionError::Native)?;
        let mut admitted = RoleAdmissionFor::<C> {
            run: Some(self),
            charge: Some(charge),
        };
        let native = admitted
            .run
            .as_mut()
            .and_then(|run| run.native.take())
            .ok_or(SigmaExecutionError::Native(typed::Error::Consumed))?;
        let (coordinator, [left, right]) = C::split(
            native,
            admitted
                .charge
                .as_mut()
                .ok_or(SigmaExecutionError::Resource)?,
        )
        .map_err(SigmaExecutionError::Native)?;
        // This additional Arc backing was quoted/reserved BEFORE native splitting.
        // Clone only an ownership link to the original source, never its payload.
        let retained = Arc::new(RetainedRolesFor::<C> {
            _source: Arc::clone(
                &admitted
                    .run
                    .as_ref()
                    .ok_or(SigmaExecutionError::Resource)?
                    .retained,
            ),
            _charge: admitted
                .charge
                .take()
                .ok_or(SigmaExecutionError::Resource)?,
        });
        Ok((
            SigmaExecutionRoleCoordinatorFor {
                native: coordinator,
                _retained: Arc::clone(&retained),
            },
            [
                SigmaExecutionRoleFor {
                    native: left,
                    retained: Arc::clone(&retained),
                },
                SigmaExecutionRoleFor {
                    native: right,
                    retained,
                },
            ],
        ))
    }
}

#[cfg(test)]
#[path = "execution_roles_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "execution_roles_owned_tests.rs"]
mod owned_tests;
