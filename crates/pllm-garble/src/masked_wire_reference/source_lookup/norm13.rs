//! Closed D43 inverse profile over ORIGINAL certified table and parent loan.
//! No public Spec13, arbitrary table/mask adoption or source authority here.
use super::super::typed::layernorm4::{Norm13Binding, Norm13MaskLoan, Norm13ValueLoan, SourceFor};
use super::*;
use std::sync::Arc;

#[cfg_attr(not(test), allow(dead_code))] // Sealed private native entry; production parent not activated.
fn spec() -> Spec {
    Spec {
        index_bits: 13,
        output_bits: 50,
        lanes: 1,
        boundary: Boundary::Reconstructed,
    }
}
#[cfg_attr(not(test), allow(dead_code))] // Exact private wrapper charge; native qualification only until parent issuance.
fn arc_meta<C: AllocationContext>() -> Result<usize, Error> {
    add(&[size_of::<Arc<SourceFor<C>>>(), size_of::<LeaseFor<C>>()])
}
#[allow(dead_code)] // Retention fields own actual source/counter even without parent activation.
pub(in crate::masked_wire_reference) struct MasksFor<C: AllocationContext> {
    native: OutputMasksFor<C>,
    source: Arc<SourceFor<C>>,
    lease: LeaseFor<C>,
}
#[allow(dead_code)] // Closed native role, no public or SDK activation.
pub(in crate::masked_wire_reference) struct PartyFor<C: AllocationContext> {
    native: super::PartyFor<C>,
    source: Arc<SourceFor<C>>,
    input: Norm13Binding,
    evaluator_controls: LeaseFor<C>,
}
#[allow(dead_code)] // Original source/lease must outlive retained frame bytes.
pub(in crate::masked_wire_reference) struct FrameFor<C: AllocationContext> {
    native: super::FrameFor<C>,
    source: Arc<SourceFor<C>>,
    lease: LeaseFor<C>,
}
#[allow(dead_code)] // Original source/lease must outlive retained output backing.
pub(in crate::masked_wire_reference) struct ValueFor<C: AllocationContext> {
    native: MaskedOutputFor<C>,
    source: Arc<SourceFor<C>>,
    lease: LeaseFor<C>,
}
#[cfg_attr(not(test), allow(dead_code))] // Only private native qualification invokes this step until parent issuance.
pub(in crate::masked_wire_reference) enum Step<C: AllocationContext> {
    Frame(FrameFor<C>),
    Output(ValueFor<C>),
}

#[allow(dead_code)]
struct IssuanceControls<'a, C: AllocationContext> {
    loan: Norm13MaskLoan<'a, C>,
    profile: correction::ValidatedDescriptor,
    gpu: gpu127::Resources,
    resources: Resources,
    resource_result: Result<Resources, Error>,
    seal: Norm13IssueSeal,
    lane: Norm13KeyLane,
    original: Result<(OutputMasksFor<C>, [super::PartyFor<C>; 2]), Error>,
    returned: Result<(MasksFor<C>, [PartyFor<C>; 2]), Error>,
    source: Arc<SourceFor<C>>,
    leases: [Option<LeaseFor<C>>; 4],
    lease_results: [Result<LeaseFor<C>, Error>; 4],
}

// EACH evaluator retains these concrete begin/advance/move/split slots in its
// original party lease. Dealer-mask Drop must not refund future evaluator work.
#[allow(dead_code)]
struct EvaluatorControls<'a, C: AllocationContext> {
    value_loan: Norm13ValueLoan<'a, C>,
    native_frame: Result<super::FrameFor<C>, Error>,
    native_step: Result<StepFor<C>, Error>,
    returned_frame: Result<FrameFor<C>, Error>,
    returned_step: Result<Step<C>, Error>,
    frame_move: Option<FrameFor<C>>,
    output_move: Option<ValueFor<C>>,
    source_clone: Arc<SourceFor<C>>,
    split: Result<LeaseFor<C>, Error>,
    moving_lease: Option<LeaseFor<C>>,
}

/// Allocation/hash/entropy free. Root certified table/source/caller remain
/// additional, NOT a reconstructed numeric baseline credit.
#[cfg_attr(not(test), allow(dead_code))] // Native-only resource query; containing parent still must price itself.
pub(in crate::masked_wire_reference) fn resources<C: AllocationContext>() -> Result<Resources, Error>
{
    let profile = correction::ValidatedDescriptor::norm_inverse();
    let child = correction::resources_validated::<C>(profile)?;
    let gpu = Profile::new(13)
        .map_err(gpu_error)?
        .layernorm13_resources()
        .map_err(gpu_error)?;
    let mut q = resources_parts::<C>(spec(), gpu, child)?;
    // Party's source Arc, two outstanding frame wrappers, output wrapper.
    let party_extra = add(&[
        size_of::<Arc<SourceFor<C>>>(),
        size_of::<Norm13Binding>(),
        size_of::<LeaseFor<C>>(),
        mul(3, arc_meta::<C>()?)?,
        size_of::<EvaluatorControls<'_, C>>(),
    ])?;
    let dealer_extra = add(&[arc_meta::<C>()?, size_of::<IssuanceControls<'_, C>>()])?;
    q.party_reservation_bytes = add(&[q.party_reservation_bytes, party_extra])?;
    q.dealer_reservation_bytes = add(&[q.dealer_reservation_bytes, dealer_extra])?;
    q.combined_reservation_bytes = add(&[
        q.dealer_reservation_bytes,
        mul(2, q.party_reservation_bytes)?,
    ])?;
    if q.combined_reservation_bytes > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(q)
}

/// Consume exact original-counter parent slice and creation-issued input loan.
/// Any error consumes the loan/reservation: no alternative table/profile retry.
#[cfg_attr(not(test), allow(dead_code))] // Closed real issuer, deliberately not promoted before parent completion.
pub(in crate::masked_wire_reference) fn issue_reserved_with<C: AllocationContext>(
    loan: Norm13MaskLoan<'_, C>,
    reservation: LeaseFor<C>,
    fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(MasksFor<C>, [PartyFor<C>; 2]), Error> {
    let q = resources::<C>()?;
    if reservation.bytes != q.combined_reservation_bytes
        || !std::ptr::eq(reservation.budget.budget(), loan.table().budget())
    {
        return Err(Error::Resource);
    }
    let source = Arc::clone(loan.source());
    let input = loan.input_binding();
    let profile = correction::ValidatedDescriptor::norm_inverse();
    let child = correction::resources_validated::<C>(profile)?;
    let (mut masks, [mut left, mut right]) = issue_parts(
        spec(),
        loan.context(),
        loan.table().values(),
        loan.values(),
        reservation,
        fill,
        q,
        child,
        Some(profile),
    )?;
    let lease = masks._scanner_dealer.split(arc_meta::<C>()?)?;
    let left_controls = left.lease.split(size_of::<EvaluatorControls<'_, C>>())?;
    let right_controls = right.lease.split(size_of::<EvaluatorControls<'_, C>>())?;
    Ok((
        MasksFor {
            native: masks,
            source: Arc::clone(&source),
            lease,
        },
        [
            PartyFor {
                native: left,
                source: Arc::clone(&source),
                input,
                evaluator_controls: left_controls,
            },
            PartyFor {
                native: right,
                source,
                input,
                evaluator_controls: right_controls,
            },
        ],
    ))
}

#[cfg_attr(not(test), allow(dead_code))] // Original native mask loan; private qualification only.
impl<C: AllocationContext> MasksFor<C> {
    pub(in crate::masked_wire_reference) fn values(&self) -> &[u64] {
        self.native.values()
    }
}
#[cfg_attr(not(test), allow(dead_code))] // Original codec frame loan; no new public framing.
impl<C: AllocationContext> FrameFor<C> {
    pub(in crate::masked_wire_reference) fn as_bytes(&self) -> &[u8] {
        self.native.as_bytes()
    }
}
#[cfg_attr(not(test), allow(dead_code))] // Original masked result loan; parent activation remains absent.
impl<C: AllocationContext> ValueFor<C> {
    pub(in crate::masked_wire_reference) fn values(&self) -> &[u64] {
        self.native.values()
    }
}
#[allow(dead_code)] // Real begin/advance/cancel awaiting production parent caller, not a stub.
impl<C: AllocationContext> PartyFor<C> {
    #[cfg(test)]
    pub(in crate::masked_wire_reference) fn evaluator_reserved_bytes(&self) -> usize {
        self.evaluator_controls.bytes
    }
    pub(in crate::masked_wire_reference) fn cancel(&mut self) {
        self.native.cancel();
    }
    pub(in crate::masked_wire_reference) fn begin(
        &mut self,
        loan: Norm13ValueLoan<'_, C>,
    ) -> Result<FrameFor<C>, Error> {
        let result = (|| {
            if !Arc::ptr_eq(loan.source(), &self.source)
                || !loan.is_for(self.input)
                || loan.role() != self.native.role
            {
                return Err(Error::Context);
            }
            let native = self.native.begin(loan.values())?;
            let lease = self.native.lease.split(arc_meta::<C>()?)?;
            Ok(FrameFor {
                native,
                source: Arc::clone(&self.source),
                lease,
            })
        })();
        if result.is_err() {
            self.native.cancel();
        }
        result
    }
    pub(in crate::masked_wire_reference) fn advance(
        &mut self,
        peer: &[u8],
    ) -> Result<Step<C>, Error> {
        let result = (|| {
            let step = self.native.advance_residency(peer, true)?;
            let lease = self.native.lease.split(arc_meta::<C>()?)?;
            Ok(match step {
                StepFor::Frame(native) => Step::Frame(FrameFor {
                    native,
                    source: Arc::clone(&self.source),
                    lease,
                }),
                StepFor::Reconstructed(native) => Step::Output(ValueFor {
                    native,
                    source: Arc::clone(&self.source),
                    lease,
                }),
                StepFor::Deferred(_) => return Err(Error::Context),
            })
        })();
        if result.is_err() {
            self.native.cancel();
        }
        result
    }
}
