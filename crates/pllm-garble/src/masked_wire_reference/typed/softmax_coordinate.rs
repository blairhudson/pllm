//! D36 private unsigned denominator TR and raw-code coordinate loan.
//!
//! Only causal_softmax can issue the consumed denominator owners. Declarations,
//! ranges and hashes alone cannot enter this bridge. Native TR14,6 is unchanged;
//! public typed TR remains signed and is deliberately not used or extended.
//! Outer D27 custody MUST retire both subparties on error/unwind and hold outputs
//! until both siblings are ready. This module creates no independent coordinator.
//! Outer frame wrappers retain that source/custody; native frames alone carry no
//! source-Arc promise. Entropy-provider captures/callback work are caller-owned,
//! additional to this quote, and must never run under the interactive locks.
//! All slots below are logical inventories, not compiled stack, allocator or RSS.

use super::causal_softmax::{DenominatorMaskFor, DenominatorValueFor};
use super::*;

const DOMAIN: &[u8] = b"pllm.sigma.private-u14-tr6-raw-u8-coordinate.v1\0";

#[derive(Clone, Copy)]
pub(super) struct Spec {
    pub(super) operation: [u8; 32],
    pub(super) quotient: EdgeDescriptor,
    pub(super) coordinate: EdgeDescriptor,
    pub(super) next: [u8; 32],
}

fn native_spec() -> arithmetic::Spec {
    arithmetic::Spec {
        bits: 14,
        lanes: 1,
        gate: arithmetic::Gate::Tr { shift: 6 },
    }
}

/// Fixed logical work categories for one normal resource-query/declaration/issue
/// route and both evaluator continuations. Not CPU instructions or permission;
/// the enclosing root must admit these together with ALL prior/pending work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Work {
    pub(super) resource_queries: usize,
    pub(super) declaration_checks: usize,
    pub(super) original_owner_checks: usize,
    pub(super) bindings_issued: usize,
    pub(super) binding_digests: usize,
    pub(super) descriptor_digests: usize,
    pub(super) context_hashes: usize,
    pub(super) input_word_reads: usize,
    pub(super) coordinate_loans: usize,
    pub(super) paired_native_issue_aes: usize,
    pub(super) native_evaluate_aes_per_role: usize,
    pub(super) random_byte_requests: usize,
}

#[cfg(test)]
impl Work {
    pub(super) fn checked_total(self) -> Result<usize, Error> {
        sum(&[
            self.resource_queries,
            self.declaration_checks,
            self.original_owner_checks,
            self.bindings_issued,
            self.binding_digests,
            self.descriptor_digests,
            self.context_hashes,
            self.input_word_reads,
            self.coordinate_loans,
            self.paired_native_issue_aes,
            self.native_evaluate_aes_per_role
                .checked_mul(2)
                .ok_or(Error::Resource)?,
            self.random_byte_requests,
        ])
    }

    /// Numeric cumulative check only. Does not issue a Budget or work permit.
    pub(super) fn check(self, prior: usize, ceiling: usize) -> Result<usize, Error> {
        let total = prior
            .checked_add(self.checked_total()?)
            .ok_or(Error::Resource)?;
        if total > ceiling {
            Err(Error::Resource)
        } else {
            Ok(total)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Resources {
    pub(super) native: arithmetic::Resources,
    pub(super) declaration_bytes: usize,
    pub(super) mask_wrapper_bytes: usize,
    pub(super) party_wrapper_bytes: usize,
    pub(super) output_wrapper_bytes: usize,
    pub(super) transition_bytes: usize,
    pub(super) total_bytes: usize,
    pub(super) work: Work,
}

fn sum(parts: &[usize]) -> Result<usize, Error> {
    parts
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b).ok_or(Error::Resource))
}

pub(super) struct DeclarationFor<C: AllocationContext> {
    input: Binding,
    quotient: Binding,
    coordinate: Binding,
    operation: [u8; 32],
    native: LeaseFor<C>,
    parties: [LeaseFor<C>; 2],
    outputs: [LeaseFor<C>; 2],
    mask: LeaseFor<C>,
}

pub(super) struct MaskFor<C: AllocationContext> {
    native: arithmetic::OutputMasksFor<C>,
    #[allow(dead_code)] // Original source/native lineage retained by sealed Q0 view.
    original: DenominatorMaskFor<C>,
    #[allow(dead_code)] // Q6 provenance retained, not a retagged native owner.
    quotient: Binding,
    coordinate: Binding,
    loaned: bool,
    _lease: LeaseFor<C>,
}

pub(super) struct PartyFor<C: AllocationContext> {
    native: Option<arithmetic::PartyFor<C>>,
    original: Option<DenominatorValueFor<C>>,
    input: Binding,
    quotient: Binding,
    coordinate: Binding,
    role: u8,
    // Inline retired party stays charged even after its output escapes.
    _lease: LeaseFor<C>,
    output_lease: Option<LeaseFor<C>>,
}

pub(super) struct OutputFor<C: AllocationContext> {
    native: arithmetic::MaskedOutputFor<C>,
    #[allow(dead_code)] // Original source/native lineage retained by sealed Q0 view.
    original: DenominatorValueFor<C>,
    #[allow(dead_code)] // Q6 provenance retained, not a retagged native owner.
    quotient: Binding,
    coordinate: Binding,
    loaned: bool,
    _lease: LeaseFor<C>,
}

#[allow(clippy::large_enum_variant)] // Exact inline owner transfer; no unquoted boxing factory.
pub(super) enum StepFor<C: AllocationContext> {
    Frame(arithmetic::FrameFor<C>),
    Output(OutputFor<C>),
}

type Issued<C> = (MaskFor<C>, [PartyFor<C>; 2]);
type NativeIssued<C> = (arithmetic::OutputMasksFor<C>, [arithmetic::PartyFor<C>; 2]);

/// One loan, tied to the actual complete native output and original denominator.
/// No owned/cloneable capability, mask getter on an evaluator, or public retag.
pub(super) struct MaskCoordinate<'a, C: AllocationContext> {
    owner: &'a MaskFor<C>,
}
pub(super) struct ValueCoordinate<'a, C: AllocationContext> {
    owner: &'a OutputFor<C>,
}

// Concrete caller/callee/Result slots, conservatively summed, not guessed padding.
// Native-private workspace remains the unchanged native resources contract.
#[allow(dead_code)]
#[repr(C)]
struct DealerSlots<C: AllocationContext> {
    dealer: *mut LocalDealerFor<C>,
    input: Binding,
    spec: Spec,
    resources: Resources,
    resource_result: Result<Resources, Error>,
    reservation: LeaseFor<C>,
    declaration: DeclarationFor<C>,
    declaration_result: Result<DeclarationFor<C>, Error>,
    original: DenominatorMaskFor<C>,
    native_result: Result<NativeIssued<C>, Error>,
    native_origin: NativeIssued<C>,
    result: Result<Issued<C>, Error>,
    native_context: arithmetic::Context,
    bindings: [Binding; 3],
    binding_result: Result<Binding, Error>,
    interval: Interval,
    fixed_result: Result<(), Error>,
    hash: Sha256,
    digests: [[u8; 32]; 6],
    parts: [&'static [u8]; 5],
    arithmetic_spec: arithmetic::Spec,
    native_resources: arithmetic::Resources,
    native_query: Result<arithmetic::Resources, Error>,
    mask_slice: &'static [u64],
    descriptor_iteration: std::array::IntoIter<EdgeDescriptor, 3>,
    descriptor: EdgeDescriptor,
    numeric: Result<Numeric, Error>,
    work: Work,
    lease_split: Result<LeaseFor<C>, Error>,
    party_leases: [LeaseFor<C>; 2],
    output_leases: [LeaseFor<C>; 2],
    total_bytes: usize,
    party_bytes: usize,
    output_bytes: usize,
    declaration_bytes: usize,
    native_bytes: usize,
    mask_bytes: usize,
    transition_bytes: usize,
    work_total: usize,
    prior_work: usize,
    work_ceiling: usize,
    resource_sum: Result<usize, Error>,
    work_sum: Result<usize, Error>,
    doubled_work: Result<usize, Error>,
    cumulative_work: Result<usize, Error>,
}

#[allow(dead_code)]
#[repr(C)]
struct PartySlots<C: AllocationContext> {
    party: *mut PartyFor<C>,
    original: DenominatorValueFor<C>,
    native_step: Result<arithmetic::StepFor<C>, Error>,
    native_origin: arithmetic::StepFor<C>,
    outgoing: Result<arithmetic::FrameFor<C>, Error>,
    result: Result<StepFor<C>, Error>,
    output_origin: OutputFor<C>,
    output_destination: StepFor<C>,
    peer: &'static [u8],
    input_slice: &'static [u64],
    fixed_result: Result<(), Error>,
}

// Lifetime-erased references are priced as pointer/length slots instead of
// requiring C:'static. No raw pointer is dereferenced anywhere in this module.
fn dealer_transition_bytes<C: AllocationContext>() -> usize {
    size_of::<DealerSlots<C>>()
}
fn party_transition_bytes<C: AllocationContext>() -> usize {
    size_of::<PartySlots<C>>()
}
fn output_transition_bytes<C: AllocationContext>() -> Result<usize, Error> {
    sum(&[
        size_of::<*mut OutputFor<C>>(),
        size_of::<ValueCoordinate<'_, C>>(),
        size_of::<Result<ValueCoordinate<'_, C>, Error>>(),
    ])
}

/// Quote before declarations, caller control construction or native factories.
/// Original incoming owners/source backing stay separately and continuously
/// charged; full wrapper layouts include their retained inline handles. No heap
/// is allocated by the bridge apart from the original native TR implementation.
pub(super) fn resources_for<C: AllocationContext>() -> Result<Resources, Error> {
    let native = arithmetic::resources_for_context::<C>(native_spec())?;
    let declaration_bytes = size_of::<DeclarationFor<C>>();
    let mask_wrapper_bytes = sum(&[
        size_of::<MaskFor<C>>(),
        size_of::<MaskCoordinate<'_, C>>(),
        size_of::<Result<MaskCoordinate<'_, C>, Error>>(),
    ])?;
    let party_wrapper_bytes = size_of::<PartyFor<C>>();
    let output_wrapper_bytes = sum(&[size_of::<OutputFor<C>>(), output_transition_bytes::<C>()?])?;
    let transition_bytes = sum(&[
        dealer_transition_bytes::<C>(),
        party_transition_bytes::<C>(),
        party_transition_bytes::<C>(),
    ])?;
    let total_bytes = sum(&[
        native.combined_reservation_bytes,
        declaration_bytes,
        mask_wrapper_bytes,
        party_wrapper_bytes,
        party_wrapper_bytes,
        output_wrapper_bytes,
        output_wrapper_bytes,
        transition_bytes,
    ])?;
    if total_bytes > crate::shared_allocation::MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        native,
        declaration_bytes,
        mask_wrapper_bytes,
        party_wrapper_bytes,
        output_wrapper_bytes,
        transition_bytes,
        total_bytes,
        work: Work {
            resource_queries: 3,
            declaration_checks: 1,
            original_owner_checks: 3,
            bindings_issued: 1,
            binding_digests: 3,
            // Three inside Binding::digest, one in dealer.binding, two in
            // native Context. Do not silently hide nested descriptor hashing.
            descriptor_digests: 6,
            // Original typed mask-id derivation and this bridge's operation.
            context_hashes: 2,
            input_word_reads: 3,
            coordinate_loans: 3,
            paired_native_issue_aes: native.paired_issue_aes_blocks,
            native_evaluate_aes_per_role: native.evaluation_aes_blocks_per_party,
            random_byte_requests: native.random_bytes,
        },
    })
}

fn checked_declaration<C: AllocationContext>(
    dealer: &LocalDealerFor<C>,
    input: Binding,
    s: Spec,
) -> Result<Interval, Error> {
    let edge = input.descriptor;
    if input.session != dealer.session
        || input.namespace != dealer.namespace
        || input.consumer != s.operation
        || edge.identity().source() != dealer.source
        || edge.identity().plan() != dealer.plan
    {
        return Err(Error::Context);
    }
    let unsigned =
        |bits, f| Numeric::new(bits, f, Interpretation::Unsigned).map_err(|_| Error::Domain);
    if edge.numeric() != unsigned(14, 12)?
        || s.quotient.numeric() != unsigned(8, 6)?
        || s.coordinate.numeric() != unsigned(8, 0)?
    {
        return Err(Error::Domain);
    }
    for descriptor in [edge, s.quotient, s.coordinate] {
        if descriptor.representation() != Representation::CommonAdditiveMasked
            || descriptor.weight_visibility() == Some(WeightVisibility::Public)
            || descriptor.shape().elements() != 1
        {
            return Err(Error::Domain);
        }
    }
    if edge.identity().source() != s.quotient.identity().source()
        || edge.identity().plan() != s.quotient.identity().plan()
        || edge.shape() != s.quotient.shape()
        || edge.identity().first_tensor_index() != s.quotient.identity().first_tensor_index()
        || edge.identity().wire() == s.quotient.identity().wire()
        || s.quotient.identity() != s.coordinate.identity()
        || s.quotient.shape() != s.coordinate.shape()
        || s.quotient.weight_visibility() != s.coordinate.weight_visibility()
    {
        return Err(Error::Context);
    }
    edge.identity()
        .first_tensor_index()
        .checked_add(1)
        .ok_or(Error::Resource)?;
    let range = input.interval.ok_or(Error::Domain)?;
    if range.lower != 4096 || ![8193, 12289].contains(&range.upper) {
        return Err(Error::Domain);
    }
    Ok(range.shift(6))
}

/// Acyclic metadata precommit, NOT a denominator proof or permission to issue.
/// Only issue_with can consume B's genuine original mask owner.
pub(super) fn declare_reserved<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    input: Binding,
    spec: Spec,
    mut reservation: LeaseFor<C>,
) -> Result<DeclarationFor<C>, Error> {
    let range = checked_declaration(dealer, input, spec)?;
    let quote = resources_for::<C>()?;
    if reservation.bytes != quote.total_bytes
        || !std::ptr::eq(reservation.budget.budget(), dealer.budget.budget())
    {
        return Err(Error::Resource);
    }
    let native = reservation.split(quote.native.combined_reservation_bytes)?;
    let party_bytes = sum(&[quote.party_wrapper_bytes, party_transition_bytes::<C>()])?;
    let parties = [
        reservation.split(party_bytes)?,
        reservation.split(party_bytes)?,
    ];
    let outputs = [
        reservation.split(quote.output_wrapper_bytes)?,
        reservation.split(quote.output_wrapper_bytes)?,
    ];
    let quotient = dealer.binding(spec.quotient, spec.next, Some(range))?;
    // Same actual native issuance/mask, not a second independent output root.
    let coordinate = Binding {
        descriptor: spec.coordinate,
        ..quotient
    };
    Ok(DeclarationFor {
        input,
        quotient,
        coordinate,
        operation: spec.operation,
        native,
        parties,
        outputs,
        mask: reservation,
    })
}

impl<C: AllocationContext> DeclarationFor<C> {
    pub(super) fn output_binding(&self) -> Binding {
        self.coordinate
    }
    #[cfg(test)]
    pub(super) fn quotient_binding(&self) -> Binding {
        self.quotient
    }

    pub(super) fn issue_with(
        self,
        original: DenominatorMaskFor<C>,
        fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<Issued<C>, Error> {
        if original.binding() != self.input
            || original.values().len() != 1
            || !std::ptr::eq(original.budget(), self.native.budget.budget())
        {
            return Err(Error::Context);
        }
        let operation = digest(&[
            DOMAIN,
            &self.operation,
            &self.input.digest(),
            &self.quotient.digest(),
            &self.coordinate.digest(),
        ]);
        let context = arithmetic::Context {
            plan: self.input.descriptor.identity().plan(),
            session: self.input.session,
            operation,
            input_wire: self.input.descriptor.digest(),
            output_wire: self.quotient.descriptor.digest(),
            first_tensor_index: self.input.descriptor.identity().first_tensor_index(),
        };
        let (native, [left, right]) = arithmetic::issue_reserved_with(
            native_spec(),
            context,
            original.values(),
            self.native,
            fill,
        )?;
        let [left_lease, right_lease] = self.parties;
        let [left_output, right_output] = self.outputs;
        let party = |native, role, lease, output_lease| PartyFor {
            native: Some(native),
            original: None,
            input: self.input,
            quotient: self.quotient,
            coordinate: self.coordinate,
            role,
            _lease: lease,
            output_lease: Some(output_lease),
        };
        Ok((
            MaskFor {
                native,
                original,
                quotient: self.quotient,
                coordinate: self.coordinate,
                loaned: false,
                _lease: self.mask,
            },
            [
                party(left, 0, left_lease, left_output),
                party(right, 1, right_lease, right_output),
            ],
        ))
    }
}

impl<C: AllocationContext> MaskFor<C> {
    pub(super) fn coordinate_once(&mut self) -> Result<MaskCoordinate<'_, C>, Error> {
        if std::mem::replace(&mut self.loaned, true) {
            return Err(Error::Consumed);
        }
        Ok(MaskCoordinate { owner: self })
    }
}

impl<C: AllocationContext> OutputFor<C> {
    pub(super) fn coordinate_once(&mut self) -> Result<ValueCoordinate<'_, C>, Error> {
        if std::mem::replace(&mut self.loaned, true) {
            return Err(Error::Consumed);
        }
        Ok(ValueCoordinate { owner: self })
    }
}

#[allow(dead_code)] // Sealed provenance inspection methods; no caller-minted proof.
impl<C: AllocationContext> MaskCoordinate<'_, C> {
    pub(super) fn binding(&self) -> Binding {
        self.owner.coordinate
    }
    pub(super) fn quotient_binding(&self) -> Binding {
        self.owner.quotient
    }
    pub(super) fn parent_binding(&self) -> Binding {
        self.owner.original.binding()
    }
    pub(super) fn values(&self) -> &[u64] {
        self.owner.native.values()
    }
    pub(super) fn budget(&self) -> &Budget {
        self.owner.original.budget()
    }
    /// Valid only with this original coordinate; never a reusable range token.
    pub(super) fn reachable_inverse(&self) -> Interval {
        Interval {
            lower: 1365,
            upper: 4097,
        }
    }
}

#[allow(dead_code)] // Sealed provenance inspection methods; no caller-minted proof.
impl<C: AllocationContext> ValueCoordinate<'_, C> {
    pub(super) fn binding(&self) -> Binding {
        self.owner.coordinate
    }
    pub(super) fn quotient_binding(&self) -> Binding {
        self.owner.quotient
    }
    pub(super) fn parent_binding(&self) -> Binding {
        self.owner.original.binding()
    }
    pub(super) fn values(&self) -> &[u64] {
        self.owner.native.values()
    }
    pub(super) fn budget(&self) -> &Budget {
        self.owner.original.budget()
    }
    pub(super) fn role(&self) -> u8 {
        self.owner.original.role()
    }
    pub(super) fn reachable_inverse(&self) -> Interval {
        Interval {
            lower: 1365,
            upper: 4097,
        }
    }
}

impl<C: AllocationContext> PartyFor<C> {
    pub(super) fn cancel(&mut self) {
        if let Some(native) = self.native.as_mut() {
            native.cancel();
        }
        self.native = None;
        self.original = None;
        // Retain own inline/output reservation until actual wrapper drop.
    }

    pub(super) fn begin(
        &mut self,
        original: DenominatorValueFor<C>,
    ) -> Result<arithmetic::FrameFor<C>, Error> {
        let result = (|| {
            if self.original.is_some() {
                return Err(Error::Consumed);
            }
            if original.binding() != self.input
                || original.role() != self.role
                || original.values().len() != 1
                || !std::ptr::eq(original.budget(), self._lease.budget.budget())
            {
                return Err(Error::Context);
            }
            let frame = self
                .native
                .as_mut()
                .ok_or(Error::Consumed)?
                .begin(original.values())?;
            self.original = Some(original);
            Ok(frame)
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }

    pub(super) fn advance(&mut self, peer: &[u8]) -> Result<StepFor<C>, Error> {
        let result = (|| match self.native.as_mut().ok_or(Error::Consumed)?.advance(peer)? {
            arithmetic::StepFor::Frame(frame) => Ok(StepFor::Frame(frame)),
            arithmetic::StepFor::Output(native) => {
                let original = self.original.take().ok_or(Error::Consumed)?;
                let lease = self.output_lease.take().ok_or(Error::Consumed)?;
                self.native = None;
                Ok(StepFor::Output(OutputFor {
                    native,
                    original,
                    quotient: self.quotient,
                    coordinate: self.coordinate,
                    loaned: false,
                    _lease: lease,
                }))
            }
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
}

#[cfg(test)]
#[path = "softmax_coordinate_tests.rs"]
mod tests;
