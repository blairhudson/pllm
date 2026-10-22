//! Private, unregistered same-process SIGMA bridge core.
//!
//! Registration gate: compiler admission starts with an ALREADY constructed
//! DecoderPlan, DecoderMatrixInventory, ExpectedSigmaSourcePin, selections and
//! borrowed numeric/source buffers. The trusted pin now has a separate admitted
//! metadata constructor, but neither compiler preflight nor source/role quotes
//! admit plan/inventory construction from Python, serde allocation/scratch, or
//! Python object packaging. A byte limit is not that missing construction census.
//! Do not register these classes or add a
//! Python factory until those owners provide enforceable preconstruction quotes
//! and actual immutable Python buffer custody is retained through decoding.
//!
//! This core accepts only that existing Rust caller baseline. It never decodes
//! bytes into a plan/pin, derives an expected pin from supplied bytes, constructs
//! a certificate, or calls the native primitive directly. Source validation,
//! original coefficients, original counter/charges and the native pre-entropy
//! IssuanceTicket path remain exclusively in the consuming compiler facade.
//!
//! The inventory below covers this Rust core, NOT CPython allocation, arbitrary
//! caller callbacks, allocator headers, machine stack or RSS. Pyclass annotations
//! enforce thread affinity if registration is completed later; no pymethods,
//! Python constructors, registration entry point or SDK surface exists here.
#![allow(dead_code)]

use pllm_compiler::{
    sigma_linear_activation::{
        execute_owned_sigma_linear_activation, ExpectedSigmaSourcePin, OwnedSigmaExecutionMask,
        OwnedSigmaExecutionRole, OwnedSigmaExecutionRoleCoordinator, OwnedSigmaExecutionRoleFrame,
        OwnedSigmaExecutionRoleOutput, OwnedSigmaExecutionRoleStep, OwnedSigmaExecutionRun,
        SigmaExecutionError, SigmaLinearActivationRequest, SigmaSourceBytes,
    },
    verified_source::SELECTED_SOURCE_BYTE_CAPACITY,
    DecoderMatrixInventory,
};
use pllm_core::edge_contract::EdgeDescriptor;
use pllm_garble::{
    masked_wire_reference::typed::OwnedLocalDealer,
    shared_allocation::{Error as AllocationError, OwnedBudget, OwnedExternalCharge},
};
use pyo3::prelude::*;
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    mem::size_of,
    rc::Rc,
    thread::{self, ThreadId},
};

// Additional bridge bounds, not alternative numeric/protocol declarations.
// Refuse schedule growth rather than silently underprice new frame owners.
const MAX_FRAMES: usize = 22;
const MAX_OUTPUTS: usize = 2;
const MAX_VALUES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BridgeError {
    Resource,
    Affinity,
    Consumed,
    ForeignFrame,
    Allocation(AllocationError),
    Compiler(SigmaExecutionError),
}

type Issued = (OwnedSigmaExecutionMask, OwnedSigmaExecutionRun);
type Split = (
    OwnedSigmaExecutionRoleCoordinator,
    [OwnedSigmaExecutionRole; 2],
);
type LocalOwners = (LocalCoordinator, [LocalRole; 2]);

#[derive(Clone, Copy)]
struct Affinity {
    process: u32,
    thread: ThreadId,
}

impl Affinity {
    fn current() -> Self {
        Self {
            process: std::process::id(),
            thread: thread::current().id(),
        }
    }

    fn check(self) -> Result<(), BridgeError> {
        if self.process == std::process::id() && self.thread == thread::current().id() {
            Ok(())
        } else {
            Err(BridgeError::Affinity)
        }
    }
}

// Native payloads retire before the final bridge charge. No public mask access.
// Only this individual coordinator slot is borrowed, never across caller code.
struct Retention {
    coordinator: RefCell<OwnedSigmaExecutionRoleCoordinator>,
    _mask: OwnedSigmaExecutionMask,
    affinity: Affinity,
    frames: Cell<usize>,
    outputs: Cell<usize>,
    _charge: OwnedExternalCharge,
}

impl Retention {
    fn cancel(&self) {
        self.coordinator.borrow_mut().cancel();
    }

    fn frame(
        self: &Rc<Self>,
        native: OwnedSigmaExecutionRoleFrame,
    ) -> Result<LocalFrame, BridgeError> {
        let next = self
            .frames
            .get()
            .checked_add(1)
            .ok_or(BridgeError::Resource)?;
        if next > MAX_FRAMES {
            return Err(BridgeError::Resource);
        }
        self.frames.set(next);
        Ok(LocalFrame {
            native,
            retained: Rc::clone(self),
        })
    }

    fn output(
        self: &Rc<Self>,
        native: OwnedSigmaExecutionRoleOutput,
    ) -> Result<LocalOutput, BridgeError> {
        let next = self
            .outputs
            .get()
            .checked_add(1)
            .ok_or(BridgeError::Resource)?;
        if next > MAX_OUTPUTS {
            return Err(BridgeError::Resource);
        }
        self.outputs.set(next);
        Ok(LocalOutput {
            native,
            retained: Rc::clone(self),
        })
    }
}

#[pyclass(unsendable, name = "_SigmaLocalCoordinator", module = "pllm._native")]
struct LocalCoordinator {
    retained: Rc<Retention>,
}

impl LocalCoordinator {
    fn cancel(&self) -> Result<(), BridgeError> {
        self.retained.affinity.check()?;
        self.retained.cancel();
        Ok(())
    }

    fn is_aborted(&self) -> Result<bool, BridgeError> {
        self.retained.affinity.check()?;
        Ok(self.retained.coordinator.borrow().is_aborted())
    }
}

impl Drop for LocalCoordinator {
    fn drop(&mut self) {
        // Native coordinator stays alive in Retention while any frame/output
        // exists; dropping THIS public-facing owner must still revoke them now.
        self.retained.cancel();
    }
}

#[pyclass(unsendable, name = "_SigmaLocalRole", module = "pllm._native")]
struct LocalRole {
    native: RefCell<Option<OwnedSigmaExecutionRole>>,
    retained: Rc<Retention>,
}

#[pyclass(unsendable, name = "_SigmaLocalFrame", module = "pllm._native")]
struct LocalFrame {
    native: OwnedSigmaExecutionRoleFrame,
    retained: Rc<Retention>,
}

#[pyclass(unsendable, name = "_SigmaLocalOutput", module = "pllm._native")]
struct LocalOutput {
    native: OwnedSigmaExecutionRoleOutput,
    retained: Rc<Retention>,
}

impl LocalOutput {
    fn descriptor(&self) -> Result<EdgeDescriptor, BridgeError> {
        self.retained.affinity.check()?;
        self.native.descriptor().map_err(BridgeError::Compiler)
    }
}

// No byte export, buffer protocol, serialization, native owner or mask getters.
#[allow(clippy::large_enum_variant)]
enum LocalStep {
    Frame(LocalFrame),
    Ready,
}

/// Armed before argument decoding; its mutable slot borrow ends immediately.
/// On errors/unwind, even AFTER native finish, shared cancellation revokes both
/// siblings and any output already moved into a partially packaged object.
struct RoleCall<'a> {
    role: &'a LocalRole,
    native: Option<OwnedSigmaExecutionRole>,
    armed: bool,
}

impl<'a> RoleCall<'a> {
    fn take(role: &'a LocalRole) -> Result<Self, BridgeError> {
        role.retained.affinity.check()?;
        let native = role.native.borrow_mut().take();
        match native {
            Some(native) => Ok(Self {
                role,
                native: Some(native),
                armed: true,
            }),
            None => {
                // Reentry sees the vacant slot, not a RefCell panic. It burns
                // the active outer operation as well as its sibling.
                role.retained.cancel();
                Err(BridgeError::Consumed)
            }
        }
    }

    fn native(&mut self) -> Result<&mut OwnedSigmaExecutionRole, BridgeError> {
        self.native.as_mut().ok_or(BridgeError::Consumed)
    }

    fn restore(mut self) -> Result<(), BridgeError> {
        // A callback may have cancelled/reentered even if it returned false.
        if self.role.retained.coordinator.borrow().is_aborted() {
            return Err(BridgeError::Consumed);
        }
        let mut slot = self.role.native.borrow_mut();
        if slot.is_some() {
            return Err(BridgeError::Consumed);
        }
        *slot = self.native.take();
        self.armed = false;
        Ok(())
    }
}

impl Drop for RoleCall<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.role.retained.cancel();
            // Retire the taken native continuation before its retention link.
            self.native.take();
        }
    }
}

/// Keep this guard alive until ALL return-value packaging succeeds. Taking the
/// step is not commit. Any packaging failure/unwind burns both continuations.
#[must_use = "taking a step is not commit; keep this guard through packaging"]
struct PendingStep<'a> {
    step: Option<LocalStep>,
    call: RoleCall<'a>,
}

impl PendingStep<'_> {
    fn take_step(&mut self) -> Result<LocalStep, BridgeError> {
        self.step.take().ok_or_else(|| {
            self.call.role.retained.cancel();
            BridgeError::Consumed
        })
    }

    fn commit(self) -> Result<(), BridgeError> {
        if self.step.is_some() {
            return Err(BridgeError::Consumed);
        }
        self.call.restore()
    }
}

/// Unlike native role drop AFTER finish, this still has abort authority. Thus
/// packaging failure of either finished output also revokes its finished sibling.
#[must_use = "keep abort authority until finished-output packaging succeeds"]
struct PendingOutput<'a> {
    output: Option<LocalOutput>,
    call: RoleCall<'a>,
}

impl PendingOutput<'_> {
    fn take_output(&mut self) -> Result<LocalOutput, BridgeError> {
        self.output.take().ok_or_else(|| {
            self.call.role.retained.cancel();
            BridgeError::Consumed
        })
    }

    fn commit(mut self) -> Result<(), BridgeError> {
        if self.output.is_some() || self.call.role.retained.coordinator.borrow().is_aborted() {
            return Err(BridgeError::Consumed);
        }
        self.call.armed = false;
        Ok(())
    }
}

impl LocalRole {
    fn begin(&self, cancelled: &mut impl FnMut() -> bool) -> Result<PendingStep<'_>, BridgeError> {
        let mut call = RoleCall::take(self)?;
        let frame = call
            .native()?
            .begin(cancelled)
            .map_err(BridgeError::Compiler)?;
        let step = LocalStep::Frame(self.retained.frame(frame)?);
        Ok(PendingStep {
            step: Some(step),
            call,
        })
    }

    /// Future PyO3 entry point must accept raw &Bound<PyAny> and perform its
    /// extraction INSIDE parse_peer (or use raw *args/**kwargs). Never extract a
    /// PyRef/frame, numeric argument or callback before RoleCall::take succeeds.
    fn advance<'a, 'peer>(
        &'a self,
        parse_peer: impl FnOnce() -> Result<&'peer LocalFrame, BridgeError>,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<PendingStep<'a>, BridgeError> {
        let mut call = RoleCall::take(self)?;
        let peer = parse_peer()?;
        peer.retained.affinity.check()?;
        if !Rc::ptr_eq(&self.retained, &peer.retained) {
            return Err(BridgeError::ForeignFrame);
        }
        let bytes = peer.native.as_bytes().map_err(BridgeError::Compiler)?;
        let step = match call
            .native()?
            .advance(bytes, cancelled)
            .map_err(BridgeError::Compiler)?
        {
            OwnedSigmaExecutionRoleStep::Frame(frame) => {
                LocalStep::Frame(self.retained.frame(frame)?)
            }
            OwnedSigmaExecutionRoleStep::Ready => LocalStep::Ready,
        };
        Ok(PendingStep {
            step: Some(step),
            call,
        })
    }

    fn finish(&self) -> Result<PendingOutput<'_>, BridgeError> {
        let mut call = RoleCall::take(self)?;
        // The unchanged compiler/native finish enforces BOTH-Ready. No wrapper
        // ready flag, separate calculation or early-output authority exists.
        let native = call.native.take().ok_or(BridgeError::Consumed)?;
        let output = native.finish().map_err(BridgeError::Compiler)?;
        let output = self.retained.output(output)?;
        Ok(PendingOutput {
            output: Some(output),
            call,
        })
    }
}

// Every field holding a native owner precedes the final bridge charge.
struct Admission<'a> {
    expected: Option<ExpectedSigmaSourcePin>,
    request: Option<SigmaLinearActivationRequest<'a>>,
    issued: Option<Issued>,
    split: Option<Split>,
    charge: Option<OwnedExternalCharge>,
}

/// Pure checked logical inventory of the concrete Rust wrappers introduced here.
/// Entire nested objects are conservatively counted, including native metadata
/// already priced upstream. No guessed per-node or Python-object allowance.
struct BridgeResources {
    retention_bytes: usize,
    population_bytes: usize,
    transition_bytes: usize,
    payload_bytes: usize,
}

fn sum(parts: &[usize]) -> Result<usize, BridgeError> {
    parts.iter().try_fold(0usize, |sum, &n| {
        sum.checked_add(n).ok_or(BridgeError::Resource)
    })
}

fn population<T>(count: usize) -> Result<usize, BridgeError> {
    size_of::<T>()
        .checked_mul(count)
        .ok_or(BridgeError::Resource)
}

impl BridgeResources {
    fn checked() -> Result<Self, BridgeError> {
        // Logical Rc owner inventory: two reference counters and payload with
        // alignment padding, matching the shared native Arc inventory convention.
        let retention_bytes = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<Retention>())
            .map_err(|_| BridgeError::Resource)?
            .0
            .pad_to_align()
            .size();
        let population_bytes = sum(&[
            size_of::<LocalCoordinator>(),
            population::<LocalRole>(2)?,
            population::<LocalFrame>(MAX_FRAMES)?,
            population::<LocalStep>(MAX_FRAMES)?,
            population::<LocalOutput>(MAX_OUTPUTS)?,
        ])?;
        let construction = sum(&[
            size_of::<Admission<'_>>(),
            size_of::<ExpectedSigmaSourcePin>(),
            size_of::<SigmaLinearActivationRequest<'_>>(),
            size_of::<SigmaSourceBytes<'_>>(),
            size_of::<Issued>(),
            size_of::<Result<Issued, SigmaExecutionError>>(),
            size_of::<Split>(),
            size_of::<Result<Split, SigmaExecutionError>>(),
            size_of::<Retention>(),
            size_of::<Affinity>(),
            size_of::<Rc<Retention>>(),
            size_of::<[LocalRole; 2]>(),
            size_of::<LocalOwners>(),
            size_of::<Result<LocalOwners, BridgeError>>(),
            size_of::<OwnedExternalCharge>(),
            size_of::<Result<OwnedExternalCharge, AllocationError>>(),
            size_of::<Self>(),
            size_of::<Result<Self, BridgeError>>(),
        ])?;
        let operation = sum(&[
            size_of::<RoleCall<'_>>(),
            size_of::<Result<RoleCall<'_>, BridgeError>>(),
            size_of::<PendingStep<'_>>(),
            size_of::<Result<PendingStep<'_>, BridgeError>>(),
            size_of::<PendingOutput<'_>>(),
            size_of::<Result<PendingOutput<'_>, BridgeError>>(),
            size_of::<Option<OwnedSigmaExecutionRole>>(),
            size_of::<OwnedSigmaExecutionRoleFrame>(),
            size_of::<Result<OwnedSigmaExecutionRoleFrame, SigmaExecutionError>>(),
            size_of::<OwnedSigmaExecutionRoleStep>(),
            size_of::<Result<OwnedSigmaExecutionRoleStep, SigmaExecutionError>>(),
            size_of::<OwnedSigmaExecutionRoleOutput>(),
            size_of::<Result<OwnedSigmaExecutionRoleOutput, SigmaExecutionError>>(),
            size_of::<LocalFrame>(),
            size_of::<Result<LocalFrame, BridgeError>>(),
            size_of::<LocalStep>(),
            size_of::<Result<LocalStep, BridgeError>>(),
            size_of::<LocalOutput>(),
            size_of::<Result<LocalOutput, BridgeError>>(),
            size_of::<EdgeDescriptor>(),
            size_of::<Result<EdgeDescriptor, SigmaExecutionError>>(),
            size_of::<Result<EdgeDescriptor, BridgeError>>(),
            size_of::<Result<(), BridgeError>>(),
            size_of::<Result<&'static [u8], SigmaExecutionError>>(),
        ])?;
        // Concrete size-query tables and Layout return slots are themselves
        // logical owned scratch, not the storage their entries describe. These
        // phases are sequential; summing all tables conservatively overcounts.
        let calculation = sum(&[
            size_of::<[usize; 5]>(),  // population
            size_of::<[usize; 18]>(), // construction
            size_of::<[usize; 24]>(), // operation
            size_of::<[usize; 3]>(),  // transition
            size_of::<[usize; 3]>(),  // payload
            size_of::<[usize; 9]>(),  // this calculation table
            size_of::<Result<usize, BridgeError>>(),
            size_of::<Layout>(),
            size_of::<Result<(Layout, usize), std::alloc::LayoutError>>(),
        ])?;
        // Reentrant sibling calls can overlap: price both complete populations.
        let transition_bytes = sum(&[
            construction,
            operation.checked_mul(2).ok_or(BridgeError::Resource)?,
            calculation,
        ])?;
        let payload_bytes = sum(&[retention_bytes, population_bytes, transition_bytes])?;
        Ok(Self {
            retention_bytes,
            population_bytes,
            transition_bytes,
            payload_bytes,
        })
    }
}

/// Bounded Rust-only local factory. All incoming complex inputs are preexisting
/// caller-owned Rust baseline; this is deliberately NOT a Python-input factory.
/// The independent trusted expected pin and supplied byte tuple remain separate.
#[allow(clippy::too_many_arguments)]
fn from_existing_rust_inputs(
    expected: ExpectedSigmaSourcePin,
    supplied: SigmaSourceBytes<'_>,
    request: SigmaLinearActivationRequest<'_>,
    inventory: &DecoderMatrixInventory,
    budget: &OwnedBudget,
    dealer: &mut OwnedLocalDealer,
    input: &[i64],
) -> Result<LocalOwners, BridgeError> {
    if !dealer.is_budget(budget.budget()) {
        return Err(BridgeError::Compiler(SigmaExecutionError::BudgetIdentity));
    }
    // No serde, Vec conversion, source copy or callbacks before these caps.
    if supplied.bytes.is_empty()
        || supplied.bytes.len() > SELECTED_SOURCE_BYTE_CAPACITY
        || supplied.scalar_artifact_offsets.len() > MAX_VALUES
        || request.source_coefficients.len() > MAX_VALUES
        || request.encoded_private_coefficients.len() > MAX_VALUES
        || input.len() > MAX_VALUES
    {
        return Err(BridgeError::Resource);
    }
    let quote = BridgeResources::checked()?;
    let charge = budget
        .charge_external(quote.payload_bytes)
        .map_err(BridgeError::Allocation)?;
    let mut admitted = Admission {
        expected: Some(expected),
        request: Some(request),
        issued: None,
        split: None,
        charge: Some(charge),
    };
    admitted.issued = Some(
        execute_owned_sigma_linear_activation(
            admitted.expected.take().ok_or(BridgeError::Consumed)?,
            supplied,
            admitted.request.take().ok_or(BridgeError::Consumed)?,
            inventory,
            budget,
            dealer,
            input,
        )
        .map_err(BridgeError::Compiler)?,
    );
    let (mask, run) = admitted.issued.take().ok_or(BridgeError::Consumed)?;
    admitted.split = Some(run.into_roles(budget).map_err(BridgeError::Compiler)?);
    let (coordinator, [left, right]) = admitted.split.take().ok_or(BridgeError::Consumed)?;
    // Sole additional heap allocation. Complete core inventory was charged
    // before compiler import/issuance; original compiler guards never move out.
    let retained = Rc::new(Retention {
        coordinator: RefCell::new(coordinator),
        _mask: mask,
        affinity: Affinity::current(),
        frames: Cell::new(0),
        outputs: Cell::new(0),
        _charge: admitted.charge.take().ok_or(BridgeError::Consumed)?,
    });
    Ok((
        LocalCoordinator {
            retained: Rc::clone(&retained),
        },
        [
            LocalRole {
                native: RefCell::new(Some(left)),
                retained: Rc::clone(&retained),
            },
            LocalRole {
                native: RefCell::new(Some(right)),
                retained,
            },
        ],
    ))
}

#[cfg(test)]
#[path = "sigma_linear_activation_tests.rs"]
mod tests;
