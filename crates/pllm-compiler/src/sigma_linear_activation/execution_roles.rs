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
    let native = typed::local_linear_activation_role_split_resources()
        .map_err(SigmaExecutionError::Native)?;
    let retention_backing_bytes = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<RetainedRoles<'static>>())
        .map(|(layout, _)| layout.pad_to_align().size())
        .map_err(|_| SigmaExecutionError::Resource)?;
    let wrapper_population_bytes = execution_sum([
        size_of::<SigmaExecutionRoleCoordinator<'static>>(),
        2 * size_of::<SigmaExecutionRole<'static>>(),
        22 * size_of::<SigmaExecutionRoleFrame<'static>>(),
        22 * size_of::<SigmaExecutionRoleStep<'static>>(),
        2 * size_of::<SigmaExecutionRoleOutput<'static>>(),
    ])?;
    // Conservative named logical transition census, including this quote's
    // return slot. This is not an ABI-stack, allocator or RSS measurement.
    let transition_bytes = execution_sum([
        size_of::<SigmaExecutionRoleSplitResources>(),
        size_of::<Result<SigmaExecutionRoleSplitResources, SigmaExecutionError>>(),
        size_of::<typed::LocalLinearActivationRoleSplitResources>(),
        size_of::<RoleAdmission<'static>>(),
        size_of::<SigmaExecutionRun<'static>>(),
        size_of::<RetainedRoles<'static>>(),
        size_of::<typed::LocalLinearActivationRun<'static>>(),
        size_of::<NativeRoles<'static>>(),
        size_of::<Result<NativeRoles<'static>, typed::Error>>(),
        size_of::<CompilerRoles<'static>>(),
        size_of::<Result<CompilerRoles<'static>, SigmaExecutionError>>(),
        size_of::<Result<typed::LocalLinearActivationRoleFrame<'static>, typed::Error>>(),
        size_of::<Result<typed::LocalLinearActivationRoleStep<'static>, typed::Error>>(),
        size_of::<Result<typed::LocalLinearActivationRoleOutput<'static>, typed::Error>>(),
        size_of::<Result<SigmaExecutionRoleFrame<'static>, SigmaExecutionError>>(),
        size_of::<Result<SigmaExecutionRoleStep<'static>, SigmaExecutionError>>(),
        size_of::<Result<SigmaExecutionRoleOutput<'static>, SigmaExecutionError>>(),
        size_of::<Result<EdgeDescriptor, SigmaExecutionError>>(),
        4 * size_of::<Arc<RetainedRoles<'static>>>(),
        size_of::<Arc<RetainedSource<'static>>>(),
        size_of::<std::array::IntoIter<usize, 5>>(),
        size_of::<std::array::IntoIter<usize, 22>>(),
    ])?;
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

type NativeRoles<'b> = (
    typed::LocalLinearActivationRoleCoordinator<'b>,
    [typed::LocalLinearActivationRole<'b>; 2],
);
type CompilerRoles<'b> = (
    SigmaExecutionRoleCoordinator<'b>,
    [SigmaExecutionRole<'b>; 2],
);

struct RoleAdmission<'b> {
    // Original native/source owners retire before the new accounting guard on
    // every rejection. No source or coefficient backing is copied or detached.
    run: Option<SigmaExecutionRun<'b>>,
    charge: Option<ExternalCharge<'b>>,
}
struct RetainedRoles<'b> {
    _source: Arc<RetainedSource<'b>>,
    _charge: ExternalCharge<'b>,
}

/// Source-retaining local coordinator. Native cancellation burns both original
/// role continuations immediately; callers need not remember sibling retirement.
/// Keep alive until outputs retire. Dropping it aborts both, even if ready.
pub struct SigmaExecutionRoleCoordinator<'b> {
    native: typed::LocalLinearActivationRoleCoordinator<'b>,
    _retained: Arc<RetainedRoles<'b>>,
}
impl SigmaExecutionRoleCoordinator<'_> {
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
pub struct SigmaExecutionRole<'b> {
    native: typed::LocalLinearActivationRole<'b>,
    retained: Arc<RetainedRoles<'b>>,
}

pub struct SigmaExecutionRoleFrame<'b> {
    native: typed::LocalLinearActivationRoleFrame<'b>,
    _retained: Arc<RetainedRoles<'b>>,
}
impl SigmaExecutionRoleFrame<'_> {
    pub fn as_bytes(&self) -> Result<&[u8], SigmaExecutionError> {
        self.native.as_bytes().map_err(SigmaExecutionError::Native)
    }
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
pub struct SigmaExecutionRoleOutput<'b> {
    native: typed::LocalLinearActivationRoleOutput<'b>,
    _retained: Arc<RetainedRoles<'b>>,
}
impl SigmaExecutionRoleOutput<'_> {
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

impl<'b> SigmaExecutionRole<'b> {
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
    ) -> Result<SigmaExecutionRoleFrame<'b>, SigmaExecutionError> {
        self.native
            .begin(cancelled)
            .map(|native| SigmaExecutionRoleFrame {
                native,
                _retained: Arc::clone(&self.retained),
            })
            .map_err(SigmaExecutionError::Native)
    }
    pub fn advance(
        &mut self,
        peer: &[u8],
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<SigmaExecutionRoleStep<'b>, SigmaExecutionError> {
        match self.native.advance(peer, cancelled) {
            Ok(typed::LocalLinearActivationRoleStep::Frame(native)) => {
                Ok(SigmaExecutionRoleStep::Frame(SigmaExecutionRoleFrame {
                    native,
                    _retained: Arc::clone(&self.retained),
                }))
            }
            Ok(typed::LocalLinearActivationRoleStep::Ready) => Ok(SigmaExecutionRoleStep::Ready),
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
    pub fn finish(self) -> Result<SigmaExecutionRoleOutput<'b>, SigmaExecutionError> {
        let Self { native, retained } = self;
        native
            .finish()
            .map(|native| SigmaExecutionRoleOutput {
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
        if !self
            .native
            .as_ref()
            .ok_or(SigmaExecutionError::Native(typed::Error::Consumed))?
            .is_budget(budget)
        {
            return Err(SigmaExecutionError::BudgetIdentity);
        }
        let quote = sigma_execution_role_split_resources()?;
        let charge = budget
            .charge_external(quote.additional_payload_bytes)
            .map_err(SigmaExecutionError::Native)?;
        let mut admitted = RoleAdmission {
            run: Some(self),
            charge: Some(charge),
        };
        let native = admitted
            .run
            .as_mut()
            .and_then(|run| run.native.take())
            .ok_or(SigmaExecutionError::Native(typed::Error::Consumed))?;
        let (coordinator, [left, right]) = native
            .into_roles_from_reserved(
                admitted
                    .charge
                    .as_mut()
                    .ok_or(SigmaExecutionError::Resource)?,
            )
            .map_err(SigmaExecutionError::Native)?;
        // This additional Arc backing was quoted/reserved BEFORE native splitting.
        // Clone only an ownership link to the original source, never its payload.
        let retained = Arc::new(RetainedRoles {
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
            SigmaExecutionRoleCoordinator {
                native: coordinator,
                _retained: Arc::clone(&retained),
            },
            [
                SigmaExecutionRole {
                    native: left,
                    retained: Arc::clone(&retained),
                },
                SigmaExecutionRole {
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
