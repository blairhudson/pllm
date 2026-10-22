//! Bounded party-local masked matrix owners. Local caller-mask issuance does not
//! establish mask freshness or trusted provenance. Frames are binding checks,
//! not authentication, malicious-output verification, or a transport protocol.
//! The coordinator must cancel the sibling on any failure.
use super::{binding, core, random, Context, Live, Spec, ISSUANCE};
pub use super::{Budget, Error};
use crate::shared_allocation::{
    allocate, AllocationContext, Lease, LeaseFor, OwnedBudget, OwnedBudgetContext,
    OwnedExternalCharge, MAX_ALLOCATION_BYTES,
};
use sha2::{Digest, Sha256};
use std::{mem::size_of, sync::atomic::Ordering};
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
    let (a, b, c, d) = core::dimensions(spec)?;
    let material = 8 * (a + b + c);
    let frame = HEADER + 8 * c;
    let party_metadata =
        size_of::<PartyFor<C>>() + size_of::<FrameFor<C>>() + size_of::<MaskedOutputFor<C>>();
    // No heap control block: Live is a non-secret, statically counted guard.
    // Party: three material allocations, one local output and one frame.
    let party = material + 8 * c + frame + party_metadata + 5 * ALLOCATOR + (8 << 10);
    // Dealer scratch allocation and bounded issuance / SHA / arithmetic stack.
    let dealer_metadata = size_of::<LeaseFor<C>>() + size_of::<Live>() + C::TRANSIENT_LINK_BYTES;
    let dealer = 8 * c + dealer_metadata + ALLOCATOR + (16 << 10);
    let reservation = 2 * party + dealer;
    if reservation > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        input_elements: a,
        weight_elements: b,
        output_elements: c,
        party_key_bits: usize::from(spec.bits) * (a + b + c),
        dealer_macs: d,
        party_macs: [2 * d, 3 * d],
        complete_macs: 6 * d,
        random_bytes: 32 + 8 * (a + b + c),
        party_material_payload_bytes: material,
        local_output_payload_bytes: 8 * c,
        frame_bytes: frame,
        total_frame_bytes: 2 * frame,
        dealer_scratch_bytes: 8 * c,
        borrowed_mask_bytes: 8 * (a + b + c),
        borrowed_masked_operand_bytes: 8 * (a + b),
        owner_metadata_bytes: 2 * party_metadata + dealer_metadata,
        stack_allowance_bytes: 32 << 10,
        allocator_allowance_bytes: 11 * ALLOCATOR,
        party_reservation_bytes: party,
        dealer_reservation_bytes: dealer,
        reservation_bytes: reservation,
    })
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
}
