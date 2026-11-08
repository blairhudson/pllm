//! Bounded party-local masked matrix owners. Local caller-mask issuance does not
//! establish mask freshness or trusted provenance. Frames are binding checks,
//! not authentication, malicious-output verification, or a transport protocol.
//! The coordinator must cancel the sibling on any failure.
use super::{binding, core, random, Context, Live, Release, Spec, ISSUANCE};
pub use super::{Budget, Error};
use crate::shared_allocation::{
    allocate, AllocationContext, Lease, LeaseFor, OwnedBudget, OwnedBudgetContext,
    OwnedExternalCharge, MAX_ALLOCATION_BYTES,
};
use sha2::{Digest, Sha256};
use std::{
    marker::PhantomData,
    mem::{align_of, size_of},
    rc::Rc,
    sync::atomic::Ordering,
};
use zeroize::Zeroizing;

pub(crate) mod material;

const MAGIC: &[u8; 8] = b"PLLMMX01";
const HEADER: usize = 8 + 32 + 32 + 1 + 1 + 2;
const ALLOCATOR: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub input_elements: usize,
    pub weight_elements: usize,
    pub output_elements: usize,
    pub party_key_bits: usize,
    pub dealer_macs: usize,
    pub party_macs: [usize; 2],
    pub complete_macs: usize,
    pub random_bytes: usize,
    pub party_material_payload_bytes: usize,
    pub local_output_payload_bytes: usize,
    pub frame_bytes: usize,
    pub total_frame_bytes: usize,
    pub dealer_scratch_bytes: usize,
    /// Borrowed masks are additional outer owners, not reserved here.
    pub borrowed_mask_bytes: usize,
    /// Borrowed common operands are additional outer owners, not reserved here.
    /// Actual backing capacities, aliases and retained clear inputs also belong
    /// to the caller and must be priced by the enclosing coordinator.
    pub borrowed_masked_operand_bytes: usize,
    pub owner_metadata_bytes: usize,
    pub stack_allowance_bytes: usize,
    pub allocator_allowance_bytes: usize,
    pub party_reservation_bytes: usize,
    pub dealer_reservation_bytes: usize,
    /// Combined dealer / both parties / retained frames / outputs peak.
    /// Logical capacity admission, not an allocator or process RSS guarantee.
    pub reservation_bytes: usize,
}

pub fn resources(spec: Spec) -> Result<Resources, Error> {
    resources_for_context::<&Budget>(spec)
}

/// Same numeric/work inventory with actual owning-container layouts.
pub fn owned_resources(spec: Spec) -> Result<Resources, Error> {
    resources_for_context::<OwnedBudgetContext>(spec)
}

pub(crate) fn resources_for_context<C: AllocationContext>(spec: Spec) -> Result<Resources, Error> {
    let dimensions_result = core::dimensions(spec);
    let dimensions_tuple = match dimensions_result {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    let (a, b, c, d) = dimensions_tuple;
    let dn = match PartyFor::<C>::FINISHED_TRANSFER_BYTES {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n00 = match a.checked_add(b) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n01 = match n00.checked_add(c) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n02 = match 8usize.checked_mul(n01) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n03 = match 8usize.checked_mul(c) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n04 = match HEADER.checked_add(n03) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n05 = match size_of::<PartyFor<C>>().checked_add(size_of::<FrameFor<C>>()) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n06 = match n05.checked_add(size_of::<MaskedOutputFor<C>>()) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n07 = match n06.checked_add(dn) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n08 = match n02.checked_add(n03) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n09 = match n08.checked_add(n04) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n10 = match n09.checked_add(n07) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n11 = match 5usize.checked_mul(ALLOCATOR) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n12 = match n10.checked_add(n11) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n13 = match n12.checked_add(8192) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n14 = match size_of::<LeaseFor<C>>().checked_add(size_of::<Live>()) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n15 = match n14.checked_add(C::TRANSIENT_LINK_BYTES) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n16 = match n03.checked_add(n15) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n17 = match n16.checked_add(ALLOCATOR) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n18 = match n17.checked_add(16384) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n19 = match 2usize.checked_mul(n13) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n20 = match n19.checked_add(n18) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n21 = match usize::from(spec.bits).checked_mul(n01) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n22 = match 2usize.checked_mul(d) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n23 = match 3usize.checked_mul(d) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n24 = match 6usize.checked_mul(d) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n25 = match 32usize.checked_add(n02) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n26 = match 2usize.checked_mul(n04) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n27 = match 8usize.checked_mul(n00) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n28 = match 2usize.checked_mul(n07) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n29 = match n28.checked_add(n15) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let n30 = match 11usize.checked_mul(ALLOCATOR) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    if n20 > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    let out: Resources = Resources {
        input_elements: a,
        weight_elements: b,
        output_elements: c,
        party_key_bits: n21,
        dealer_macs: d,
        party_macs: [n22, n23],
        complete_macs: n24,
        random_bytes: n25,
        party_material_payload_bytes: n02,
        local_output_payload_bytes: n03,
        frame_bytes: n04,
        total_frame_bytes: n26,
        dealer_scratch_bytes: n03,
        borrowed_mask_bytes: n02,
        borrowed_masked_operand_bytes: n27,
        owner_metadata_bytes: n29,
        stack_allowance_bytes: 32768,
        allocator_allowance_bytes: n30,
        party_reservation_bytes: n13,
        dealer_reservation_bytes: n18,
        reservation_bytes: n20,
    };
    Ok(out)
}

enum State {
    Fresh(core::Material),
    Pending {
        local: Zeroizing<Vec<u64>>,
        operand_digest: [u8; 32],
    },
}

/// Owns only this party's material, public binding, phase and reservation.
/// No sibling reference or shared secret/reconstruction owner. Failures consume.
pub type Party<'b> = PartyFor<&'b Budget>;
pub type OwnedParty = PartyFor<OwnedBudgetContext>;
pub struct PartyFor<C: AllocationContext> {
    id: u8,
    spec: Spec,
    binding: [u8; 32],
    state: Option<State>,
    // Secret owners must erase/deallocate before refund or singleton release.
    lease: Option<LeaseFor<C>>,
    live: Option<Live>,
}
pub type Frame<'b> = FrameFor<&'b Budget>;
pub type OwnedFrame = FrameFor<OwnedBudgetContext>;
pub struct FrameFor<C: AllocationContext> {
    bytes: Zeroizing<Vec<u8>>,
    _lease: LeaseFor<C>,
    _live: Live,
}
impl<C: AllocationContext> FrameFor<C> {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}
pub type MaskedOutput<'b> = MaskedOutputFor<&'b Budget>;
pub type OwnedMaskedOutput = MaskedOutputFor<OwnedBudgetContext>;
pub struct MaskedOutputFor<C: AllocationContext> {
    values: Zeroizing<Vec<u64>>,
    _lease: LeaseFor<C>,
    _live: Live,
}
impl<C: AllocationContext> MaskedOutputFor<C> {
    pub fn values(&self) -> &[u64] {
        &self.values
    }

    // Only finished native ownership crosses this boundary, never typed authority.
    pub(in crate::masked_wire_reference) fn into_opaque(self) -> OpaqueNative<C> {
        let Self { values, _lease: lease, _live: live } = self;
        let release = live.into_release();
        let out: OpaqueNative<C> = OpaqueNative { values, native_lease: lease, release };
        out
    }
}

// Field order is the erasure/refund/release contract. No getters or constructors.
pub(in crate::masked_wire_reference) struct OpaqueNative<C: AllocationContext> {
    values: Zeroizing<Vec<u64>>,
    native_lease: LeaseFor<C>,
    release: Release,
}

pub fn issue_local<'b>(
    spec: Spec,
    context: Context,
    rx: &[u64],
    rw: &[u64],
    ry: &[u64],
    budget: &'b Budget,
) -> Result<[Party<'b>; 2], Error> {
    issue_local_cancellable(spec, context, rx, rw, ry, budget, &mut || false)
}

pub fn issue_local_cancellable<'b>(
    spec: Spec,
    context: Context,
    rx: &[u64],
    rw: &[u64],
    ry: &[u64],
    budget: &'b Budget,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<[Party<'b>; 2], Error> {
    issue_with(spec, context, [rx, rw, ry], budget, &mut random, cancelled)
}

/// Same process-local material issuance from a preadmitted owning counter.
/// No borrowed owner is upgraded, no material codec is enabled, and Live keeps
/// the original one-thread profile. Exact native capacity transfers once.
#[allow(clippy::too_many_arguments)]
pub fn issue_owned_from_reserved(
    spec: Spec,
    context: Context,
    masks: [&[u64]; 3],
    budget: &OwnedBudget,
    charge: &mut OwnedExternalCharge,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<[OwnedParty; 2], Error> {
    issue_owned_from_reserved_with(spec, context, masks, budget, charge, &mut random, cancelled)
}

#[allow(clippy::too_many_arguments)]
fn issue_owned_from_reserved_with(
    spec: Spec,
    context: Context,
    [rx, rw, ry]: [&[u64]; 3],
    budget: &OwnedBudget,
    charge: &mut OwnedExternalCharge,
    fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<[OwnedParty; 2], Error> {
    let cost = owned_resources(spec)?;
    context.validate()?;
    core::validate(rx, cost.input_elements, spec.bits)?;
    core::validate(rw, cost.weight_elements, spec.bits)?;
    core::validate(ry, cost.output_elements, spec.bits)?;
    let lease = charge.split_native(budget.budget(), cost.reservation_bytes)?;
    issue_reserved_with(spec, context, [rx, rw, ry], lease, fill, cancelled)
}

pub(super) fn issue_with<'b>(
    spec: Spec,
    context: Context,
    [rx, rw, ry]: [&[u64]; 3],
    budget: &'b Budget,
    fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<[Party<'b>; 2], Error> {
    let cost = resources(spec)?;
    context.validate()?;
    core::validate(rx, cost.input_elements, spec.bits)?;
    core::validate(rw, cost.weight_elements, spec.bits)?;
    core::validate(ry, cost.output_elements, spec.bits)?;
    // Entire peak admitted before entropy, control ownership or allocation.
    let lease = budget.reserve(cost.reservation_bytes)?;
    issue_reserved_with(spec, context, [rx, rw, ry], lease, fill, cancelled)
}

/// Consume exactly `resources(spec)?.reservation_bytes` from original admission.
/// Geometry, canonical masks and exact capacity are checked before entropy.
pub(crate) fn issue_reserved_with<C: AllocationContext>(
    spec: Spec,
    context: Context,
    [rx, rw, ry]: [&[u64]; 3],
    lease: LeaseFor<C>,
    fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<[PartyFor<C>; 2], Error> {
    let cost = resources_for_context::<C>(spec)?;
    context.validate()?;
    core::validate(rx, cost.input_elements, spec.bits)?;
    core::validate(rw, cost.weight_elements, spec.bits)?;
    core::validate(ry, cost.output_elements, spec.bits)?;
    if lease.bytes != cost.reservation_bytes {
        return Err(Error::Resource);
    }
    // Legacy entry points still validate/admit before acquiring control, then
    // check cancellation before nonce entropy. Their codec/order is unchanged.
    let ticket = IssuanceTicket::acquire()?;
    issue_reserved_ticket_with(spec, context, [rx, rw, ry], lease, ticket, fill, cancelled)
}

/// Actual singleton ownership and a burned native issuance number. An enclosing
/// operation acquires this before root entropy, then moves it into issuance.
/// Private fields prevent reconstruction from metadata. Drop releases the real
/// guard on every partial failure; no peek/reacquire race or thread-local state.
pub(crate) struct IssuanceTicket {
    live: Live,
    issuance: u64,
}
impl IssuanceTicket {
    pub(crate) fn acquire() -> Result<Self, Error> {
        let live = Live::acquire()?;
        let issuance = ISSUANCE
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_add(1))
            .map_err(|_| Error::Resource)?;
        Ok(Self { live, issuance })
    }
}

/// Same native issuance, with control admitted before enclosing root entropy.
/// All checks remain here as well: a ticket authorizes no geometry or masks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn issue_reserved_ticket_with<C: AllocationContext>(
    spec: Spec,
    context: Context,
    [rx, rw, ry]: [&[u64]; 3],
    mut lease: LeaseFor<C>,
    ticket: IssuanceTicket,
    fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<[PartyFor<C>; 2], Error> {
    let cost = resources_for_context::<C>(spec)?;
    context.validate()?;
    core::validate(rx, cost.input_elements, spec.bits)?;
    core::validate(rw, cost.weight_elements, spec.bits)?;
    core::validate(ry, cost.output_elements, spec.bits)?;
    if lease.bytes != cost.reservation_bytes {
        return Err(Error::Resource);
    }
    let allocation = lease.budget.share();
    let budget = allocation.budget();
    let IssuanceTicket { live, issuance } = ticket;
    if cancelled() {
        return Err(Error::Consumed);
    }
    let mut nonce = [0; 32];
    fill(&mut nonce)?;
    let mut hash = Sha256::new();
    hash.update(b"pllm.masked.matrix.party.v1\0");
    hash.update(binding(context, spec, issuance, &nonce));
    let binding: [u8; 32] = hash.finalize().into();
    let materials = core::deal(spec, [rx, rw, ry], budget, fill, cancelled)?;
    let left_lease = lease.split(cost.party_reservation_bytes)?;
    let right_lease = lease.split(cost.party_reservation_bytes)?;
    let [left, right] = materials;
    let other_live = live.fork();
    // Dealer scratch already erased/deallocated by core::deal before its refund.
    Ok([
        PartyFor {
            id: 0,
            spec,
            binding,
            state: Some(State::Fresh(left)),
            lease: Some(left_lease),
            live: Some(live),
        },
        PartyFor {
            id: 1,
            spec,
            binding,
            state: Some(State::Fresh(right)),
            lease: Some(right_lease),
            live: Some(other_live),
        },
    ])
}

#[cfg(test)]
pub(crate) fn with_exhausted_issuance<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(u64);
    impl Drop for Restore {
        fn drop(&mut self) {
            ISSUANCE.store(self.0, Ordering::SeqCst);
        }
    }
    // Caller must hold matrix::TEST_SERIAL, as for every native matrix test.
    let _restore = Restore(ISSUANCE.swap(u64::MAX, Ordering::SeqCst));
    f()
}

fn operands(xhat: &[u64], what: &[u64]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"pllm.masked.matrix.operands.v1\0");
    for values in [xhat, what] {
        hash.update((values.len() as u64).to_le_bytes());
        for value in values {
            hash.update(value.to_le_bytes());
        }
    }
    hash.finalize().into()
}

impl<C: AllocationContext> PartyFor<C> {
    // Original-birth slot allowance. Ordinary output retains unused slots too.
    // Four Result<O, E> slots: closure, named result, return, typed receiver.
    // H(X) = checked size_of::<X>() + checked (align_of::<X>() - 1).
    const FINISHED_TRANSFER_BYTES: Option<usize> = 'inventory: {
        let padding0 = match align_of::<Result<MaskedOutputFor<C>, Error>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h0 = match size_of::<Result<MaskedOutputFor<C>, Error>>().checked_add(padding0) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p0 = match h0.checked_mul(4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s0 = p0;
        let padding1 = match align_of::<MaskedOutputFor<C>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h1 = match size_of::<MaskedOutputFor<C>>().checked_add(padding1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s1 = match s0.checked_add(h1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding2 = match align_of::<Zeroizing<Vec<u64>>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h2 = match size_of::<Zeroizing<Vec<u64>>>().checked_add(padding2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s2 = match s1.checked_add(h2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding3 = match align_of::<LeaseFor<C>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h3 = match size_of::<LeaseFor<C>>().checked_add(padding3) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s3 = match s2.checked_add(h3) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding4 = match align_of::<Live>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h4 = match size_of::<Live>().checked_add(padding4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p4 = match h4.checked_mul(2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s4 = match s3.checked_add(p4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding5 = match align_of::<Release>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h5 = match size_of::<Release>().checked_add(padding5) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p5 = match h5.checked_mul(3) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s5 = match s4.checked_add(p5) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding6 = match align_of::<OpaqueNative<C>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h6 = match size_of::<OpaqueNative<C>>().checked_add(padding6) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p6 = match h6.checked_mul(2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s6 = match s5.checked_add(p6) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding7 = match align_of::<PhantomData<Rc<()>>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h7 = match size_of::<PhantomData<Rc<()>>>().checked_add(padding7) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s7 = match s6.checked_add(h7) {
            Some(value) => value,
            None => break 'inventory None,
        };
        Some(s7)
    };
    const GEOMETRY_INVENTORY: Option<usize> = 'inventory: {
        let s0 = size_of::<Spec>();
        let s1 = match s0.checked_add(size_of::<Result<pllm_core::streamed_matrix::MatrixGeometry, pllm_core::streamed_matrix::Error>>()) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s2 = match s1.checked_add(size_of::<Result<pllm_core::streamed_matrix::MatrixGeometry, Error>>()) {
            Some(value) => value,
            None => break 'inventory None,
        };
        Some(s2)
    };
    const DIMENSIONS_INVENTORY: Option<usize> = 'inventory: {
        let geometry = match Self::GEOMETRY_INVENTORY {
            Some(value) => value,
            None => break 'inventory None,
        };
        let phase = if size_of::<[usize; 3]>() > geometry {
            size_of::<[usize; 3]>()
        } else {
            geometry
        };
        let records = match size_of::<Spec>().checked_add(size_of::<Result<(usize, usize, usize, usize), Error>>()) {
            Some(value) => value,
            None => break 'inventory None,
        };
        records.checked_add(phase)
    };
    // 31 nodes: 62 operand U + 31 named U + 4 dimensions + dn = 98 U;
    // 31 node Options + dn read = 32; 31 node errors + dn + forwarding + cap = 34.
    pub(in crate::masked_wire_reference) const NATIVE_CALC: Option<usize> = 'inventory: {
        let s0 = match Self::DIMENSIONS_INVENTORY { Some(value) => value, None => break 'inventory None };
        let padding1 = match align_of::<Spec>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h1 = match size_of::<Spec>().checked_add(padding1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p1 = match h1.checked_mul(2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s1 = match s0.checked_add(p1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding2 = match align_of::<Result<(usize, usize, usize, usize), Error>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h2 = match size_of::<Result<(usize, usize, usize, usize), Error>>().checked_add(padding2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s2 = match s1.checked_add(h2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding3 = match align_of::<(usize, usize, usize, usize)>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h3 = match size_of::<(usize, usize, usize, usize)>().checked_add(padding3) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s3 = match s2.checked_add(h3) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding4 = match align_of::<usize>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h4 = match size_of::<usize>().checked_add(padding4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p4 = match h4.checked_mul(98) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s4 = match s3.checked_add(p4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding5 = match align_of::<Option<usize>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h5 = match size_of::<Option<usize>>().checked_add(padding5) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p5 = match h5.checked_mul(32) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s5 = match s4.checked_add(p5) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding6 = match align_of::<Error>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h6 = match size_of::<Error>().checked_add(padding6) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p6 = match h6.checked_mul(34) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s6 = match s5.checked_add(p6) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding7 = match align_of::<Resources>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h7 = match size_of::<Resources>().checked_add(padding7) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s7 = match s6.checked_add(h7) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding8 = match align_of::<Result<Resources, Error>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h8 = match size_of::<Result<Resources, Error>>().checked_add(padding8) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p8 = match h8.checked_mul(4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s8 = match s7.checked_add(p8) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding9 = match align_of::<[usize; 2]>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h9 = match size_of::<[usize; 2]>().checked_add(padding9) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s9 = match s8.checked_add(h9) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding10 = match align_of::<bool>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h10 = match size_of::<bool>().checked_add(padding10) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s10 = match s9.checked_add(h10) {
            Some(value) => value,
            None => break 'inventory None,
        };
        Some(s10)
    };

    pub fn begin(
        &mut self,
        xhat: &[u64],
        what: &[u64],
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<FrameFor<C>, Error> {
        let result = (|| {
            // Take before validation: a wrong phase or any error consumes.
            let state = self.state.take().ok_or(Error::Consumed)?;
            let State::Fresh(material) = state else {
                return Err(Error::Consumed);
            };
            let cost = resources_for_context::<C>(self.spec)?;
            let lease = self.lease.as_mut().ok_or(Error::Consumed)?;
            let local = core::evaluate(
                self.spec,
                self.id,
                &material,
                xhat,
                what,
                lease.budget.budget(),
                cancelled,
            )?;
            let operand_digest = operands(xhat, what);
            let mut bytes = Zeroizing::new(allocate(cost.frame_bytes, lease.budget.budget())?);
            bytes.extend_from_slice(MAGIC);
            bytes.extend_from_slice(&self.binding);
            bytes.extend_from_slice(&operand_digest);
            bytes.push(self.id);
            bytes.push(1); // Output-share phase, only phase in this protocol.
            bytes.extend_from_slice(&(cost.output_elements as u16).to_le_bytes());
            for value in local.iter() {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let frame_lease =
                lease.split(cost.frame_bytes + size_of::<FrameFor<C>>() + ALLOCATOR)?;
            let live = self.live.as_ref().ok_or(Error::Consumed)?.fork();
            self.state = Some(State::Pending {
                local,
                operand_digest,
            });
            Ok(FrameFor {
                bytes,
                _lease: frame_lease,
                _live: live,
            })
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }

    pub fn finish(&mut self, peer: &[u8]) -> Result<MaskedOutputFor<C>, Error> {
        let result = (|| {
            let state = self.state.take().ok_or(Error::Consumed)?;
            let State::Pending {
                mut local,
                operand_digest,
            } = state
            else {
                return Err(Error::Consumed);
            };
            let cost = resources_for_context::<C>(self.spec)?;
            if peer.len() != cost.frame_bytes {
                return Err(Error::Domain);
            }
            if &peer[..8] != MAGIC
                || peer[8..40] != self.binding
                || peer[40..72] != operand_digest
                || peer[72] != 1 - self.id
                || peer[73] != 1
                || peer[74..76] != (cost.output_elements as u16).to_le_bytes()
            {
                return Err(Error::Context);
            }
            // Validate ALL residues before mutating. No payload decode/copy.
            for word in peer[HEADER..].chunks_exact(8) {
                let value = u64::from_le_bytes(word.try_into().map_err(|_| Error::Domain)?);
                if value & !core::mask(self.spec.bits) != 0 {
                    return Err(Error::Domain);
                }
            }
            for (out, word) in local.iter_mut().zip(peer[HEADER..].chunks_exact(8)) {
                let value = u64::from_le_bytes(word.try_into().map_err(|_| Error::Domain)?);
                *out = out.wrapping_add(value) & core::mask(self.spec.bits);
            }
            // Same local allocation becomes common masked output; reservation
            // transfers only after every validation succeeds.
            Ok(MaskedOutputFor {
                values: local,
                _lease: {
                    let lease = self.lease.as_mut().ok_or(Error::Consumed)?;
                    lease.split(lease.bytes - size_of::<Self>())?
                },
                _live: self.live.take().ok_or(Error::Consumed)?,
            })
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }

    pub fn cancel(&mut self) {
        // Do not combine these assignments: drop secrets before refund.
        self.state = None;
        if let Some(lease) = self.lease.as_mut() {
            // Retained inline owner still exists after cancellation/completion.
            let refund = lease.bytes.saturating_sub(size_of::<Self>());
            drop(lease.split(refund));
        }
        self.live = None;
    }
}

#[cfg(test)]
mod owned_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn current_native_slot_funding() {
        let _serial = super::super::TEST_SERIAL.lock().unwrap();
        // Test observations are not workload funding, ABI stack measurements,
        // zeroization evidence, or a way to reach the wrong-variant branch.
        fn layout<T>(label: &str) -> usize {
            println!("{label}: {} size={} align={}", std::any::type_name::<T>(),
                size_of::<T>(), align_of::<T>());
            size_of::<T>()
        }
        #[derive(Default)]
        struct Ledger { bytes: usize }
        impl Ledger {
            fn bytes(&mut self, label: &str, bytes: usize) {
                println!("slot {label}: {bytes}");
                self.bytes = self.bytes.checked_add(bytes).expect(label);
            }
            fn slot<T>(&mut self, label: &str) {
                let size = layout::<T>(label);
                let slack = align_of::<T>().checked_sub(1).unwrap();
                self.bytes(label, size.checked_add(slack).expect(label));
            }
            fn raw<T>(&mut self, label: &str) { self.bytes(label, layout::<T>(label)); }
        }
        fn sum(values: &[usize]) -> usize {
            values.iter().try_fold(0usize, |n, &v| n.checked_add(v)).unwrap()
        }
        fn node(ledger: &mut Ledger, label: &str) {
            println!("checked node {label}");
            ledger.slot::<usize>("operand/left");
            ledger.slot::<usize>("operand/right");
            ledger.slot::<usize>("node/named-success");
            ledger.slot::<Option<usize>>("node/checked-result");
            ledger.slot::<Error>("error/node-resource");
        }
        fn native_calculation() -> usize {
            type Q = (usize, usize, usize, usize);
            type Geometry = pllm_core::streamed_matrix::MatrixGeometry;
            type GeometryError = pllm_core::streamed_matrix::Error;
            let mut geometry = Ledger::default();
            geometry.raw::<Spec>("helper/geometry-input");
            geometry.raw::<Result<Geometry, GeometryError>>("producer/core-geometry-result");
            geometry.raw::<Result<Geometry, Error>>("producer/mapped-geometry-result");
            layout::<Geometry>("geometry/record");
            layout::<GeometryError>("geometry/error");
            let dimensions_phase = layout::<[usize; 3]>("helper/dimensions-elements")
                .max(geometry.bytes);
            let mut ledger = Ledger::default();
            ledger.raw::<Spec>("helper/dimensions-input");
            ledger.raw::<Result<Q, Error>>("producer/dimensions-result");
            ledger.bytes("helper/dimensions-phase-maximum", dimensions_phase);
            ledger.slot::<Spec>("helper/public-resources-input");
            ledger.slot::<Spec>("helper/context-resources-input");
            ledger.slot::<Result<Q, Error>>("receiver/dimensions_result");
            ledger.slot::<Q>("receiver/dimensions_tuple");
            ledger.slot::<usize>("destructure/a");
            ledger.slot::<usize>("destructure/b");
            ledger.slot::<usize>("destructure/c");
            ledger.slot::<usize>("destructure/d");
            ledger.slot::<usize>("receiver/dn");
            ledger.slot::<Option<usize>>("producer/dn-allowance-read");
            ledger.slot::<Error>("error/dn-read-resource");
            ledger.slot::<Error>("error/dimensions-forward");
            node(&mut ledger, "n00:a+b");
            node(&mut ledger, "n01:n00+c");
            node(&mut ledger, "n02:8*n01");
            node(&mut ledger, "n03:8*c");
            node(&mut ledger, "n04:header+n03");
            node(&mut ledger, "n05:party-owner+frame-owner");
            node(&mut ledger, "n06:n05+output-owner");
            node(&mut ledger, "n07:n06+dn");
            node(&mut ledger, "n08:n02+n03");
            node(&mut ledger, "n09:n08+n04");
            node(&mut ledger, "n10:n09+n07");
            node(&mut ledger, "n11:party-allocators");
            node(&mut ledger, "n12:n10+n11");
            node(&mut ledger, "n13:n12+party-stack");
            node(&mut ledger, "n14:lease+live");
            node(&mut ledger, "n15:n14+temporary-link");
            node(&mut ledger, "n16:n03+n15");
            node(&mut ledger, "n17:n16+dealer-allocator");
            node(&mut ledger, "n18:n17+dealer-stack");
            node(&mut ledger, "n19:both-parties");
            node(&mut ledger, "n20:n19+n18");
            node(&mut ledger, "n21:bits*n01");
            node(&mut ledger, "n22:party-zero-macs");
            node(&mut ledger, "n23:party-one-macs");
            node(&mut ledger, "n24:complete-macs");
            node(&mut ledger, "n25:nonce+n02");
            node(&mut ledger, "n26:both-frames");
            node(&mut ledger, "n27:borrowed-operands");
            node(&mut ledger, "n28:both-party-metadata");
            node(&mut ledger, "n29:n28+n15");
            node(&mut ledger, "n30:all-allocators");
            ledger.slot::<Error>("error/combined-cap");
            ledger.slot::<Resources>("receiver/named-out");
            ledger.slot::<Result<Resources, Error>>("producer/context-resources-return");
            ledger.slot::<Result<Resources, Error>>("receiver/public-forwarding");
            ledger.slot::<Result<Resources, Error>>("producer/public-resources-return");
            ledger.slot::<Result<Resources, Error>>("receiver/public-caller");
            ledger.slot::<[usize; 2]>("producer/party-macs-array");
            ledger.slot::<bool>("operand/cap-comparison");
            ledger.bytes
        }
        // These are individual semantic slots, not a polynomial copied from Dn.
        fn transfer_slots<C: AllocationContext>() -> usize {
            let mut ledger = Ledger::default();
            ledger.slot::<Result<MaskedOutputFor<C>, Error>>("producer/finish-closure-return");
            ledger.slot::<Result<MaskedOutputFor<C>, Error>>("receiver/finish-named-result");
            ledger.slot::<Result<MaskedOutputFor<C>, Error>>("producer/finish-function-return");
            ledger.slot::<Result<MaskedOutputFor<C>, Error>>("receiver/typed-finish");
            ledger.slot::<MaskedOutputFor<C>>("receiver/into_opaque-self");
            ledger.slot::<Zeroizing<Vec<u64>>>("destructure/values");
            ledger.slot::<LeaseFor<C>>("destructure/lease");
            ledger.slot::<Live>("destructure/live");
            ledger.slot::<Live>("receiver/into_release-self");
            ledger.slot::<Release>("producer/into_release-return");
            ledger.slot::<Release>("receiver/named-release");
            ledger.slot::<Release>("producer/native-release-initializer");
            ledger.slot::<OpaqueNative<C>>("receiver/named-native-out");
            ledger.slot::<OpaqueNative<C>>("producer/into_opaque-return");
            ledger.slot::<PhantomData<Rc<()>>>("destructure/confined-live-remainder");
            ledger.bytes
        }
        #[derive(Clone, Copy)]
        struct Expected { party: usize, frame: usize, dealer: usize, total: usize }
        impl Expected {
            fn retained<C: AllocationContext>(&self) -> usize {
                self.party.checked_sub(self.frame).unwrap()
                    .checked_sub(size_of::<PartyFor<C>>()).unwrap()
            }
        }
        fn expected<C: AllocationContext>(link: usize) -> Expected {
            println!("native context {}", std::any::type_name::<C>());
            layout::<Live>("native/Live");
            layout::<Option<Live>>("native/OptionLive");
            layout::<Release>("native/Release");
            layout::<LeaseFor<C>>("native/Lease");
            layout::<PartyFor<C>>("native/Party");
            layout::<FrameFor<C>>("native/Frame");
            layout::<MaskedOutputFor<C>>("native/O");
            layout::<OpaqueNative<C>>("native/N");
            let dn = transfer_slots::<C>();
            assert_eq!(PartyFor::<C>::FINISHED_TRANSFER_BYTES, Some(dn));
            assert_eq!(PartyFor::<C>::NATIVE_CALC, Some(native_calculation()));
            // Tiny geometry is exactly one input, weight, output and MAC.
            // Wire header fields are measured individually, without HEADER.
            let frame_payload = sum(&[8, 32, 32, size_of::<u8>(), size_of::<u8>(),
                size_of::<u16>(), size_of::<u64>()]);
            let mut party = Ledger::default();
            party.raw::<u64>("payload/material-a");
            party.raw::<u64>("payload/material-b");
            party.raw::<u64>("payload/material-c");
            party.raw::<u64>("payload/local-output");
            party.bytes("payload/frame", frame_payload);
            party.raw::<PartyFor<C>>("owner/party");
            party.raw::<FrameFor<C>>("owner/frame");
            party.raw::<MaskedOutputFor<C>>("owner/output");
            party.bytes("slots/native-transfer", dn);
            party.bytes("allocator/material-a", 64);
            party.bytes("allocator/material-b", 64);
            party.bytes("allocator/material-c", 64);
            party.bytes("allocator/local-output", 64);
            party.bytes("allocator/frame", 64);
            party.bytes("fixed/party-stack", 8192);
            let mut dealer = Ledger::default();
            dealer.raw::<u64>("payload/correction-scratch");
            dealer.raw::<LeaseFor<C>>("owner/dealer-lease");
            dealer.raw::<Live>("owner/dealer-live");
            dealer.bytes("helper/temporary-owning-link", link);
            dealer.bytes("allocator/dealer-scratch", 64);
            dealer.bytes("fixed/dealer-stack", 16384);
            let expected = Expected {
                party: party.bytes,
                frame: sum(&[frame_payload, size_of::<FrameFor<C>>(), 64]),
                dealer: dealer.bytes,
                total: sum(&[party.bytes, party.bytes, dealer.bytes]),
            };
            let actual = resources_for_context::<C>(spec()).unwrap();
            assert_eq!(actual.input_elements, 1);
            assert_eq!(actual.weight_elements, 1);
            assert_eq!(actual.output_elements, 1);
            assert_eq!(actual.dealer_macs, 1);
            assert_eq!(actual.frame_bytes, frame_payload);
            assert_eq!(actual.party_reservation_bytes, expected.party);
            assert_eq!(actual.dealer_reservation_bytes, expected.dealer);
            assert_eq!(actual.reservation_bytes, expected.total);
            expected
        }
        fn transfer<C: AllocationContext>(
            [mut a, mut b]: [PartyFor<C>; 2], expected: Expected,
        ) -> OpaqueNative<C> {
            assert_eq!(a.lease.as_ref().unwrap().bytes, expected.party);
            assert_eq!(b.lease.as_ref().unwrap().bytes, expected.party);
            let af = a.begin(&[7], &[10], &mut || false).unwrap();
            let bf = b.begin(&[7], &[10], &mut || false).unwrap();
            assert_eq!(af._lease.bytes, expected.frame);
            assert_eq!(bf._lease.bytes, expected.frame);
            assert_eq!(a.lease.as_ref().unwrap().bytes,
                expected.party.checked_sub(expected.frame).unwrap());
            a.lease.as_ref().unwrap().budget.budget().fail_after.store(0, Ordering::SeqCst);
            let output = a.finish(bf.as_bytes()).unwrap();
            let other = b.finish(af.as_bytes()).unwrap();
            assert_eq!(output.values(), &[17]);
            let pointer = output.values.as_ptr();
            let capacity = output.values.capacity();
            let counter = output._lease.budget.budget() as *const Budget;
            let used = output._lease.budget.budget().used_bytes();
            let updates = output._lease.budget.budget().test_accounting_updates();
            let handles = super::super::LIVE_HANDLES.load(Ordering::SeqCst);
            assert_eq!(output._lease.bytes, expected.retained::<C>());
            assert_eq!(other._lease.bytes, expected.retained::<C>());
            assert_eq!(a.lease.as_ref().unwrap().bytes, size_of::<PartyFor<C>>());
            assert_eq!(b.lease.as_ref().unwrap().bytes, size_of::<PartyFor<C>>());
            let opaque = output.into_opaque();
            assert_eq!(opaque.values.as_ptr(), pointer);
            assert_eq!(opaque.values.capacity(), capacity);
            assert_eq!(&*opaque.values, &[17]);
            assert_eq!(opaque.native_lease.bytes, expected.retained::<C>());
            assert_eq!(opaque.native_lease.budget.budget() as *const Budget, counter);
            assert_eq!(opaque.native_lease.budget.budget().used_bytes(), used);
            assert_eq!(opaque.native_lease.budget.budget().test_accounting_updates(), updates);
            assert_eq!(super::super::LIVE_HANDLES.load(Ordering::SeqCst), handles);
            drop((a, b, af, bf, other));
            assert_eq!(super::super::LIVE_HANDLES.load(Ordering::SeqCst), 1);
            assert!(super::super::LIVE.load(Ordering::SeqCst));
            opaque
        }
        layout::<material::ImportPermit<'static>>("borrowed-only/material-permit");
        layout::<material::Packet<'static>>("borrowed-only/material-packet");
        layout::<(material::ImportPermit<'static>, material::Packet<'static>)>("borrowed-only/material-return-pair");
        layout::<material::HandoffResources>("borrowed-only/material-resources");
        let borrowed = expected::<&Budget>(0);
        let owned = expected::<OwnedBudgetContext>(layout::<OwnedBudgetContext>("owned/temporary-link"));
        let bootstrap = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let fixed = bootstrap.checked_sub(size_of::<OwnedBudget>()).unwrap();
        for short in [true, false] {
            let budget = Budget::new(borrowed.total.checked_sub(usize::from(short)).unwrap()).unwrap();
            if short { budget.fail_after.store(0, Ordering::SeqCst); }
            let before = budget.test_accounting_updates();
            let mut draws = 0;
            let result = issue_with(spec(), context(), [&[5], &[7], &[11]], &budget,
                &mut |bytes| { draws += bytes.len(); fill(bytes) }, &mut || false);
            if short {
                assert!(matches!(result, Err(Error::Resource)));
                assert_eq!(draws, 0);
                assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
                assert_eq!(budget.test_accounting_updates(), before);
                assert_eq!(budget.used_bytes(), 0);
            } else {
                assert_eq!(draws, 56); // Nonce and three one-word material draws.
                assert_eq!(budget.used_bytes(), sum(&[borrowed.party, borrowed.party]));
                let opaque = transfer(result.unwrap(), borrowed);
                assert!(std::ptr::eq(opaque.native_lease.budget, &budget));
                assert_eq!(budget.used_bytes(), borrowed.retained::<&Budget>());
                drop(opaque);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
        for short in [true, false] {
            let payload = owned.total.checked_sub(usize::from(short)).unwrap();
            let limit = sum(&[bootstrap, size_of::<OwnedExternalCharge>(), payload]);
            let budget = OwnedBudget::new(limit).unwrap();
            let weak = budget.test_counter_weak();
            let counter = budget.budget() as *const Budget;
            let mut charge = budget.charge_external(payload).unwrap();
            if short { budget.budget().fail_after.store(0, Ordering::SeqCst); }
            let before = budget.budget().test_accounting_updates();
            let mut draws = 0;
            let result = issue_owned_from_reserved_with(spec(), context(), [&[5], &[7], &[11]],
                &budget, &mut charge,
                &mut |bytes| { draws += bytes.len(); fill(bytes) }, &mut || false);
            if short {
                assert!(matches!(result, Err(Error::Resource)));
                assert_eq!(draws, 0);
                assert_eq!(budget.budget().fail_after.load(Ordering::SeqCst), 0);
                assert_eq!(budget.budget().test_accounting_updates(), before);
                assert_eq!(budget.budget().used_bytes(), limit);
                drop(charge);
                assert_eq!(budget.budget().used_bytes(), bootstrap);
            } else {
                assert_eq!(draws, 56);
                assert_eq!(charge.payload_bytes(), 0);
                assert_eq!(budget.budget().used_bytes(),
                    sum(&[bootstrap, size_of::<OwnedExternalCharge>(), owned.party, owned.party]));
                let opaque = transfer(result.unwrap(), owned);
                assert_eq!(opaque.native_lease.budget.budget() as *const Budget, counter);
                drop((budget, charge));
                let observer = weak.upgrade().unwrap();
                assert_eq!(observer.used_bytes(), sum(&[fixed, owned.retained::<OwnedBudgetContext>()]));
                assert_eq!(&*opaque.values, &[17]);
                drop(opaque);
                assert_eq!(observer.used_bytes(), fixed);
                drop(observer);
                // Reclamation is separate from arithmetic and erasure evidence.
                assert!(weak.upgrade().is_none());
            }
        }
        let over_cap = MAX_ALLOCATION_BYTES.checked_add(1).unwrap();
        assert_eq!(MAX_ALLOCATION_BYTES, 64 * 1024 * 1024);
        assert!(matches!(Budget::new(over_cap), Err(Error::Resource)));
        assert!(matches!(OwnedBudget::new(over_cap), Err(Error::Resource)));
        let mut overflow = spec();
        overflow.rows = usize::MAX;
        overflow.inner = 2;
        let budget = Budget::new(borrowed.total).unwrap();
        budget.fail_after.store(0, Ordering::SeqCst);
        let before = budget.test_accounting_updates();
        let mut draws = 0;
        assert!(matches!(issue_with(overflow, context(), [&[], &[], &[]], &budget,
            &mut |bytes| { draws += bytes.len(); fill(bytes) }, &mut || false), Err(Error::Resource)));
        assert_eq!(draws, 0);
        assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
        assert_eq!(budget.test_accounting_updates(), before);
        assert_eq!(budget.used_bytes(), 0);
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut charge = root.charge_external(owned.total).unwrap();
        root.budget().fail_after.store(0, Ordering::SeqCst);
        let before = root.budget().test_snapshot();
        let updates = root.budget().test_accounting_updates();
        assert!(matches!(issue_owned_from_reserved_with(overflow, context(), [&[], &[], &[]],
            &root, &mut charge, &mut |bytes| { draws += bytes.len(); fill(bytes) },
            &mut || false), Err(Error::Resource)));
        assert_eq!(draws, 0);
        assert_eq!(root.budget().test_snapshot(), before);
        assert_eq!(root.budget().test_accounting_updates(), updates);
        assert_eq!(root.budget().fail_after.load(Ordering::SeqCst), 0);
        drop(charge);
        assert_eq!(root.budget().used_bytes(), bootstrap);
        assert_eq!(super::super::LIVE_HANDLES.load(Ordering::SeqCst), 0);
        assert!(!super::super::LIVE.load(Ordering::SeqCst));
        // No synthetic ledger overflow is advertised as exercising a production
        // size-constant overflow; those constants have no controllable test seam.
    }

    fn spec() -> Spec {
        Spec {
            bits: 50,
            rows: 1,
            inner: 1,
            columns: 1,
            layout: super::super::MatrixLayout::OutputInput,
        }
    }

    fn context() -> Context {
        Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            input_wire: [4; 32],
            weight_wire: [5; 32],
            output_wire: [6; 32],
        }
    }

    // Test-only deterministic injection into the unchanged native issuance seam.
    // Production entry points still use the original random() implementation.
    fn fill(bytes: &mut [u8]) -> Result<(), Error> {
        bytes.fill(19);
        Ok(())
    }

    #[test]
    fn owned_native_owners_preserve_one_thread_profile() {
        // Same safe ambiguity pattern as the public source-lookup Clone check:
        // gaining either auto trait makes the corresponding marker ambiguous.
        trait AmbiguousIfSend<Marker> {
            fn check() {}
        }
        impl<T: ?Sized> AmbiguousIfSend<()> for T {}
        impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
        trait AmbiguousIfSync<Marker> {
            fn check() {}
        }
        impl<T: ?Sized> AmbiguousIfSync<()> for T {}
        impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for T {}

        let _ = <OwnedParty as AmbiguousIfSend<_>>::check;
        let _ = <OwnedParty as AmbiguousIfSync<_>>::check;
        let _ = <OwnedFrame as AmbiguousIfSend<_>>::check;
        let _ = <OwnedFrame as AmbiguousIfSync<_>>::check;
        let _ = <OwnedMaskedOutput as AmbiguousIfSend<_>>::check;
        let _ = <OwnedMaskedOutput as AmbiguousIfSync<_>>::check;
    }

    #[test]
    fn borrowed_layout_and_work_inventory_remain_unchanged() {
        assert_eq!(size_of::<Lease<'_>>(), 2 * size_of::<usize>());
        let borrowed = resources(spec()).unwrap();
        let owned = owned_resources(spec()).unwrap();
        assert_eq!(owned.party_key_bits, borrowed.party_key_bits);
        assert_eq!(owned.dealer_macs, borrowed.dealer_macs);
        assert_eq!(owned.party_macs, borrowed.party_macs);
        assert_eq!(owned.random_bytes, borrowed.random_bytes);
        assert_eq!(owned.frame_bytes, borrowed.frame_bytes);
        assert_eq!(
            owned.party_reservation_bytes,
            borrowed.party_reservation_bytes
        );
        assert_eq!(
            owned.reservation_bytes,
            borrowed.reservation_bytes + OwnedBudgetContext::TRANSIENT_LINK_BYTES
        );
        assert_eq!(size_of::<OwnedParty>(), size_of::<Party<'_>>());
        assert_eq!(size_of::<OwnedFrame>(), size_of::<Frame<'_>>());
        assert_eq!(
            size_of::<OwnedMaskedOutput>(),
            size_of::<MaskedOutput<'_>>()
        );
    }

    #[test]
    fn owned_exact_short_and_foreign_counter_reject_before_entropy() {
        let _serial = super::super::TEST_SERIAL.lock().unwrap();
        let bootstrap = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let quote = owned_resources(spec()).unwrap();
        for short in [true, false] {
            let payload = quote.reservation_bytes - usize::from(short);
            let limit = bootstrap + size_of::<OwnedExternalCharge>() + payload;
            let budget = OwnedBudget::new(limit).unwrap();
            let foreign = OwnedBudget::new(limit).unwrap();
            let mut charge = budget.charge_external(payload).unwrap();
            let draws = Cell::new(0usize);
            let mut observed_fill = |bytes: &mut [u8]| {
                draws.set(draws.get() + bytes.len());
                assert_eq!(budget.budget().used_bytes(), limit);
                fill(bytes)
            };
            assert!(matches!(
                issue_owned_from_reserved_with(
                    spec(),
                    context(),
                    [&[5], &[7], &[11]],
                    &foreign,
                    &mut charge,
                    &mut observed_fill,
                    &mut || false,
                ),
                Err(Error::Resource)
            ));
            assert_eq!(draws.get(), 0);
            assert_eq!(charge.payload_bytes(), payload);
            assert_eq!(foreign.budget().used_bytes(), bootstrap);
            let result = issue_owned_from_reserved_with(
                spec(),
                context(),
                [&[5], &[7], &[11]],
                &budget,
                &mut charge,
                &mut observed_fill,
                &mut || false,
            );
            if short {
                assert!(matches!(result, Err(Error::Resource)));
                assert_eq!(draws.get(), 0);
                assert_eq!(charge.payload_bytes(), payload);
            } else {
                let parties = result.unwrap();
                assert_eq!(draws.get(), quote.random_bytes);
                assert_eq!(charge.payload_bytes(), 0);
                drop(parties);
            }
            drop(charge);
            assert_eq!(budget.budget().used_bytes(), bootstrap);
        }
    }

    #[test]
    fn owned_original_material_outputs_and_frames_outlive_root_without_upgrade() {
        let _serial = super::super::TEST_SERIAL.lock().unwrap();
        let [mut left, mut right] = {
            let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let mut charge = budget
                .charge_external(owned_resources(spec()).unwrap().reservation_bytes)
                .unwrap();
            issue_owned_from_reserved_with(
                spec(),
                context(),
                [&[5], &[7], &[11]],
                &budget,
                &mut charge,
                &mut fill,
                &mut || false,
            )
            .unwrap()
        }; // Both original root handle and remaining external guard have retired.
        let a = left.begin(&[7], &[10], &mut || false).unwrap();
        let b = right.begin(&[7], &[10], &mut || false).unwrap();
        let x = left.finish(b.as_bytes()).unwrap();
        let y = right.finish(a.as_bytes()).unwrap();
        assert_eq!(x.values(), &[17]); // 2*3 + original output mask 11.
        assert_eq!(x.values(), y.values());
        assert!(std::ptr::eq(
            a._lease.budget.budget(),
            x._lease.budget.budget()
        ));
        assert!(std::ptr::eq(
            b._lease.budget.budget(),
            y._lease.budget.budget()
        ));
        drop((left, right, b, x, y));

        // One retained original frame alone still owns the real singleton. A
        // different owning counter cannot manufacture replacement authority.
        let retry = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut charge = retry
            .charge_external(owned_resources(spec()).unwrap().reservation_bytes)
            .unwrap();
        let draws = Cell::new(0);
        assert!(matches!(
            issue_owned_from_reserved_with(
                spec(),
                context(),
                [&[5], &[7], &[11]],
                &retry,
                &mut charge,
                &mut |bytes| {
                    draws.set(draws.get() + bytes.len());
                    fill(bytes)
                },
                &mut || false,
            ),
            Err(Error::Resource)
        ));
        assert_eq!(draws.get(), 0);
        drop(a);
        // The failed native attempt consumed its transferred slice. Reserve a
        // fresh slice only for this separate test issuance after full retirement.
        charge
            .resize_payload(owned_resources(spec()).unwrap().reservation_bytes)
            .unwrap();
        drop(
            issue_owned_from_reserved_with(
                spec(),
                context(),
                [&[5], &[7], &[11]],
                &retry,
                &mut charge,
                &mut fill,
                &mut || false,
            )
            .unwrap(),
        );
    }

    #[test]
    fn owned_output_alone_retains_original_counter_and_singleton_after_root_drop() {
        let _serial = super::super::TEST_SERIAL.lock().unwrap();
        let bootstrap = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let quote = owned_resources(spec()).unwrap();
        let output = {
            let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let mut charge = budget.charge_external(quote.reservation_bytes).unwrap();
            let [mut left, mut right] = issue_owned_from_reserved_with(
                spec(),
                context(),
                [&[5], &[7], &[11]],
                &budget,
                &mut charge,
                &mut fill,
                &mut || false,
            )
            .unwrap();
            assert_eq!(charge.payload_bytes(), 0);
            let a = left.begin(&[7], &[10], &mut || false).unwrap();
            let b = right.begin(&[7], &[10], &mut || false).unwrap();
            let x = left.finish(b.as_bytes()).unwrap();
            let y = right.finish(a.as_bytes()).unwrap();
            assert_eq!(x.values(), &[17]);
            assert_eq!(x.values(), y.values());
            assert!(std::ptr::eq(x._lease.budget.budget(), budget.budget()));
            drop((charge, budget, left, right, a, b, y));
            assert_eq!(
                x._lease.budget.budget().used_bytes(),
                bootstrap - size_of::<OwnedBudget>() + x._lease.bytes
            );
            x
        };

        // No root, external guard, party, frame or sibling output survives.
        // A separately funded counter must fail on Live, not lack of capacity.
        let retained_bytes = output._lease.budget.budget().used_bytes();
        let foreign = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut charge = foreign.charge_external(quote.reservation_bytes).unwrap();
        let draws = Cell::new(0);
        let mut observed_fill = |bytes: &mut [u8]| {
            draws.set(draws.get() + bytes.len());
            fill(bytes)
        };
        assert!(matches!(
            issue_owned_from_reserved_with(
                spec(),
                context(),
                [&[5], &[7], &[11]],
                &foreign,
                &mut charge,
                &mut observed_fill,
                &mut || false,
            ),
            Err(Error::Resource)
        ));
        assert_eq!(draws.get(), 0);
        assert_eq!(charge.payload_bytes(), 0);
        assert_eq!(
            foreign.budget().used_bytes(),
            bootstrap + charge.charged_bytes()
        );
        assert_eq!(output.values(), &[17]);
        assert_eq!(output._lease.budget.budget().used_bytes(), retained_bytes);

        drop(output);
        charge.resize_payload(quote.reservation_bytes).unwrap();
        let parties = issue_owned_from_reserved_with(
            spec(),
            context(),
            [&[5], &[7], &[11]],
            &foreign,
            &mut charge,
            &mut observed_fill,
            &mut || false,
        )
        .unwrap();
        assert_eq!(draws.get(), quote.random_bytes);
        drop((parties, charge));
        assert_eq!(foreign.budget().used_bytes(), bootstrap);
    }

    #[test]
    fn owned_allocation_and_cancellation_failures_retire_original_material() {
        let _serial = super::super::TEST_SERIAL.lock().unwrap();
        let bootstrap = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        // Six material allocations followed by dealer correction scratch. The
        // final failure must retire both already-allocated material sets.
        for fail_after in 0..=6 {
            let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let mut charge = budget
                .charge_external(owned_resources(spec()).unwrap().reservation_bytes)
                .unwrap();
            budget
                .budget()
                .fail_after
                .store(fail_after, Ordering::SeqCst);
            let result = issue_owned_from_reserved_with(
                spec(),
                context(),
                [&[5], &[7], &[11]],
                &budget,
                &mut charge,
                &mut fill,
                &mut || false,
            );
            assert!(
                matches!(result, Err(Error::Allocation)),
                "fail_after={fail_after}"
            );
            assert_eq!(charge.payload_bytes(), 0);
            assert!(charge.is_for(budget.budget()));
            assert_eq!(
                budget.budget().used_bytes(),
                bootstrap + charge.charged_bytes()
            );
            drop(charge);
            assert_eq!(budget.budget().used_bytes(), bootstrap);

            // Successful issuance on the same original counter also proves
            // that the failed attempt released the process-wide singleton.
            budget.budget().fail_after.store(-1, Ordering::SeqCst);
            let mut retry_charge = budget
                .charge_external(owned_resources(spec()).unwrap().reservation_bytes)
                .unwrap();
            let parties = issue_owned_from_reserved_with(
                spec(),
                context(),
                [&[5], &[7], &[11]],
                &budget,
                &mut retry_charge,
                &mut fill,
                &mut || false,
            )
            .unwrap();
            assert_eq!(retry_charge.payload_bytes(), 0);
            drop((parties, retry_charge));
            assert_eq!(budget.budget().used_bytes(), bootstrap);
        }
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut charge = budget
            .charge_external(owned_resources(spec()).unwrap().reservation_bytes)
            .unwrap();
        let draws = Cell::new(0);
        assert!(matches!(
            issue_owned_from_reserved_with(
                spec(),
                context(),
                [&[5], &[7], &[11]],
                &budget,
                &mut charge,
                &mut |bytes| {
                    draws.set(draws.get() + bytes.len());
                    fill(bytes)
                },
                &mut || true,
            ),
            Err(Error::Consumed)
        ));
        assert_eq!(draws.get(), 0);
        drop(charge);
        assert_eq!(budget.budget().used_bytes(), bootstrap);
    }
    #[test]
    fn owned_opaque_native_survives_root_retirement() {
        let _serial = super::super::TEST_SERIAL.lock().unwrap();
        let bootstrap = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let weak = budget.test_counter_weak();
        let mut charge = budget
            .charge_external(owned_resources(spec()).unwrap().reservation_bytes)
            .unwrap();
        let [mut left, mut right] = issue_owned_from_reserved_with(
            spec(),
            context(),
            [&[5], &[7], &[11]],
            &budget,
            &mut charge,
            &mut fill,
            &mut || false,
        )
        .unwrap();
        let a = left.begin(&[7], &[10], &mut || false).unwrap();
        let b = right.begin(&[7], &[10], &mut || false).unwrap();
        let output = left.finish(b.as_bytes()).unwrap();
        let other = right.finish(a.as_bytes()).unwrap();
        let pointer = output.values.as_ptr();
        let capacity = output.values.capacity();
        let native_bytes = output._lease.bytes;
        let counter = output._lease.budget.budget() as *const Budget;
        assert_eq!(counter, budget.budget() as *const Budget);
        let used = budget.budget().used_bytes();
        let handles = super::super::LIVE_HANDLES.load(Ordering::SeqCst);
        let opaque = output.into_opaque();
        assert_eq!(opaque.values.as_ptr(), pointer);
        assert_eq!(opaque.values.capacity(), capacity);
        assert_eq!(opaque.native_lease.bytes, native_bytes);
        assert_eq!(opaque.native_lease.budget.budget() as *const Budget, counter);
        assert_eq!(opaque.native_lease.budget.budget().used_bytes(), used);
        assert_eq!(super::super::LIVE_HANDLES.load(Ordering::SeqCst), handles);

        drop(budget);
        drop((charge, left, right, a, b, other));
        assert_eq!(opaque.values.as_ptr(), pointer);
        assert_eq!(opaque.values.capacity(), capacity);
        assert_eq!(opaque.native_lease.bytes, native_bytes);
        assert_eq!(opaque.native_lease.budget.budget() as *const Budget, counter);
        assert_eq!(
            opaque.native_lease.budget.budget().used_bytes(),
            bootstrap - size_of::<OwnedBudget>() + native_bytes,
        );
        assert_eq!(super::super::LIVE_HANDLES.load(Ordering::SeqCst), 1);
        assert!(super::super::LIVE.load(Ordering::SeqCst));
        drop(opaque);
        assert_eq!(super::super::LIVE_HANDLES.load(Ordering::SeqCst), 0);
        assert!(!super::super::LIVE.load(Ordering::SeqCst));
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn opaque_native_moves_backing_lease_and_one_release_without_fork() {
        let _serial = super::super::TEST_SERIAL.lock().unwrap();
        fn send_sync<T: Send + Sync>() {}
        send_sync::<OpaqueNative<OwnedBudgetContext>>();
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let [mut left, mut right] = issue_with(
            spec(), context(), [&[5], &[7], &[11]], &budget, &mut fill, &mut || false,
        ).unwrap();
        let a = left.begin(&[7], &[10], &mut || false).unwrap();
        let b = right.begin(&[7], &[10], &mut || false).unwrap();
        let output = left.finish(b.as_bytes()).unwrap();
        let other = right.finish(a.as_bytes()).unwrap();
        let pointer = output.values.as_ptr();
        let capacity = output.values.capacity();
        let bytes = output._lease.bytes;
        let used = budget.used_bytes();
        let handles = super::super::LIVE_HANDLES.load(Ordering::SeqCst);
        let opaque = output.into_opaque();
        assert_eq!(opaque.values.as_ptr(), pointer);
        assert_eq!(opaque.values.capacity(), capacity);
        assert_eq!(opaque.native_lease.bytes, bytes);
        assert!(std::ptr::eq(opaque.native_lease.budget, &budget));
        assert_eq!(budget.used_bytes(), used);
        assert_eq!(super::super::LIVE_HANDLES.load(Ordering::SeqCst), handles);
        drop((left, right, a, b, other));
        assert_eq!(super::super::LIVE_HANDLES.load(Ordering::SeqCst), 1);
        assert!(super::super::LIVE.load(Ordering::SeqCst));
        drop(opaque);
        assert_eq!(super::super::LIVE_HANDLES.load(Ordering::SeqCst), 0);
        assert!(!super::super::LIVE.load(Ordering::SeqCst));
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn finished_slots_are_in_original_layout_aware_quote() {
        let _serial = super::super::TEST_SERIAL.lock().unwrap();
        fn check<C: AllocationContext>() {
            let quote = resources_for_context::<C>(spec()).unwrap();
            let metadata = size_of::<PartyFor<C>>() + size_of::<FrameFor<C>>()
                + size_of::<MaskedOutputFor<C>>()
                + PartyFor::<C>::FINISHED_TRANSFER_BYTES.unwrap();
            assert_eq!(quote.party_reservation_bytes,
                quote.party_material_payload_bytes + quote.local_output_payload_bytes
                + quote.frame_bytes + metadata + 5 * ALLOCATOR + 8192);
            assert_eq!(quote.reservation_bytes,
                2 * quote.party_reservation_bytes + quote.dealer_reservation_bytes);
            assert!(PartyFor::<C>::NATIVE_CALC.is_some());
        }
        check::<&Budget>();
        check::<OwnedBudgetContext>();
        let mut overflow = spec();
        overflow.rows = usize::MAX;
        overflow.inner = 2;
        assert!(matches!(resources(overflow), Err(Error::Resource)));
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        budget.fail_after.store(0, Ordering::SeqCst);
        let mut draws = 0;
        assert!(matches!(issue_with(
            overflow, context(), [&[], &[], &[]], &budget,
            &mut |_| { draws += 1; Ok(()) }, &mut || false,
        ), Err(Error::Resource)));
        assert_eq!(draws, 0);
        assert_eq!(budget.used_bytes(), 0);
    }

}
