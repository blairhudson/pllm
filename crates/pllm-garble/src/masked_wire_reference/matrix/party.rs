//! Bounded party-local masked matrix owners. Local caller-mask issuance does not
//! establish mask freshness or trusted provenance. Frames are binding checks,
//! not authentication, malicious-output verification, or a transport protocol.
//! The coordinator must cancel the sibling on any failure.
use super::{binding, core, random, Context, Live, Spec, ISSUANCE};
pub use super::{Budget, Error};
use crate::shared_allocation::{allocate, Lease, MAX_ALLOCATION_BYTES};
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
    let (a, b, c, d) = core::dimensions(spec)?;
    let material = 8 * (a + b + c);
    let frame = HEADER + 8 * c;
    let party_metadata = size_of::<Party<'static>>()
        + size_of::<Frame<'static>>()
        + size_of::<MaskedOutput<'static>>();
    // No heap control block: Live is a non-secret, statically counted guard.
    // Party: three material allocations, one local output and one frame.
    let party = material + 8 * c + frame + party_metadata + 5 * ALLOCATOR + (8 << 10);
    // Dealer scratch allocation and bounded issuance / SHA / arithmetic stack.
    let dealer_metadata = size_of::<Lease<'static>>() + size_of::<Live>();
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
pub struct Party<'b> {
    id: u8,
    spec: Spec,
    binding: [u8; 32],
    state: Option<State>,
    // Secret owners must erase/deallocate before refund or singleton release.
    lease: Option<Lease<'b>>,
    live: Option<Live>,
}
pub struct Frame<'b> {
    bytes: Zeroizing<Vec<u8>>,
    _lease: Lease<'b>,
    _live: Live,
}
impl Frame<'_> {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}
pub struct MaskedOutput<'b> {
    values: Zeroizing<Vec<u64>>,
    _lease: Lease<'b>,
    _live: Live,
}
impl MaskedOutput<'_> {
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
    let mut lease = budget.reserve(cost.reservation_bytes)?;
    let live = Live::acquire()?;
    let issuance = ISSUANCE
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_add(1))
        .map_err(|_| Error::Resource)?;
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
        Party {
            id: 0,
            spec,
            binding,
            state: Some(State::Fresh(left)),
            lease: Some(left_lease),
            live: Some(live),
        },
        Party {
            id: 1,
            spec,
            binding,
            state: Some(State::Fresh(right)),
            lease: Some(right_lease),
            live: Some(other_live),
        },
    ])
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

impl<'b> Party<'b> {
    pub fn begin(
        &mut self,
        xhat: &[u64],
        what: &[u64],
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<Frame<'b>, Error> {
        let result = (|| {
            // Take before validation: a wrong phase or any error consumes.
            let state = self.state.take().ok_or(Error::Consumed)?;
            let State::Fresh(material) = state else {
                return Err(Error::Consumed);
            };
            let cost = resources(self.spec)?;
            let lease = self.lease.as_mut().ok_or(Error::Consumed)?;
            let local = core::evaluate(
                self.spec,
                self.id,
                &material,
                xhat,
                what,
                lease.budget,
                cancelled,
            )?;
            let operand_digest = operands(xhat, what);
            let mut bytes = Zeroizing::new(allocate(cost.frame_bytes, lease.budget)?);
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
                lease.split(cost.frame_bytes + size_of::<Frame<'static>>() + ALLOCATOR)?;
            let live = self.live.as_ref().ok_or(Error::Consumed)?.fork();
            self.state = Some(State::Pending {
                local,
                operand_digest,
            });
            Ok(Frame {
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

    pub fn finish(&mut self, peer: &[u8]) -> Result<MaskedOutput<'b>, Error> {
        let result = (|| {
            let state = self.state.take().ok_or(Error::Consumed)?;
            let State::Pending {
                mut local,
                operand_digest,
            } = state
            else {
                return Err(Error::Consumed);
            };
            let cost = resources(self.spec)?;
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
            Ok(MaskedOutput {
                values: local,
                _lease: self.lease.take().ok_or(Error::Consumed)?,
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
        self.lease = None;
        self.live = None;
    }
}
