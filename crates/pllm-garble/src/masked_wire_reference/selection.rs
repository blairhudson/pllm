//! Bounded native SIGMA five-ring-word Select, independently implemented from
//! EzPC aa0f2f860266fb79901405ef5e9965b8c9fe2e88 select.cpp's numeric contract.
//!
//! Common inputs are `b_hat = b XOR beta` and `x_hat = x + r_x`.
//! Reconstructed mode opens only `b*x + r_z` in one exchange, with a fresh
//! full-ring output mask. Deferred mode retains distinct local additive shares.
//! Caller owns input-mask freshness. Local research only: frames bind views,
//! roles and invocation, but are unauthenticated, not malicious-output verified,
//! and provide neither persistent replay protection nor a crypto-review claim.

use super::mask;
use crate::shared_allocation::{allocate, Lease, MAX_ALLOCATION_BYTES};
pub use crate::shared_allocation::{Budget, Error};
use sha2::{Digest, Sha256};
use std::mem::size_of;
use zeroize::Zeroizing;

pub const DOMAIN: &str = "pllm.sigma.five-word-select.v1";
pub const MAX_LANES: usize = super::MAX_LANES;
const MAGIC: &[u8; 8] = b"PLLMSL01";
const HEADER: usize = 76;
const OUTPUT_PHASE: u8 = 1;
const ALLOCATOR: usize = 64;
const WORKSPACE: usize = 8 << 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    Reconstructed,
    Deferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub bits: u8,
    pub lanes: usize,
}
impl Spec {
    fn validate(self) -> Result<(), Error> {
        if !(3..=64).contains(&self.bits) || !(1..=MAX_LANES).contains(&self.lanes) {
            return Err(Error::Domain);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Context {
    pub plan: [u8; 32],
    pub session: [u8; 32],
    pub operation: [u8; 32],
    pub predicate_wire: [u8; 32],
    pub input_wire: [u8; 32],
    pub output_wire: [u8; 32],
    pub first_tensor_index: u64,
}
impl Context {
    fn validate(self, lanes: usize) -> Result<(), Error> {
        if self.predicate_wire == self.input_wire
            || self.output_wire == self.input_wire
            || self.output_wire == self.predicate_wire
        {
            return Err(Error::Context);
        }
        self.first_tensor_index
            .checked_add(lanes as u64)
            .ok_or(Error::Resource)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub output_bits: u8,
    pub logical_key_bits_per_party: usize,
    /// Exact requested bytes, not a security/entropy estimate.
    pub random_bytes: usize,
    pub output_mask_entropy_bits: usize,
    pub paired_issue_aes_blocks: usize,
    pub evaluation_aes_blocks_per_party: usize,
    pub frames_per_party: usize,
    pub frame_bytes: usize,
    pub total_peer_frame_bytes: usize,
    pub party_material_payload_bytes: usize,
    /// Two borrowed dealer mask slices, predicate represented as u64.
    pub borrowed_input_mask_bytes: usize,
    /// Both input slices for both parties (even if callers share backing).
    pub borrowed_masked_input_bytes: usize,
    pub dealer_reservation_bytes: usize,
    pub party_reservation_bytes: usize,
    pub combined_reservation_bytes: usize,
}

fn frame_charge(spec: Spec) -> usize {
    HEADER + 8 * spec.lanes + size_of::<Frame<'static>>() + ALLOCATOR
}
fn output_charge(spec: Spec) -> usize {
    8 * spec.lanes + size_of::<MaskedOutput<'static>>() + ALLOCATOR
}
fn deferred_charge(spec: Spec) -> usize {
    8 * spec.lanes + size_of::<DeferredShare<'static>>() + ALLOCATOR
}

fn checked_sum(parts: &[usize]) -> Result<usize, Error> {
    parts.iter().try_fold(0usize, |sum, part| {
        sum.checked_add(*part).ok_or(Error::Resource)
    })
}

/// Public admission only. Checked bounded geometry makes all subsequent size
/// arithmetic safe. Logical capacity bound, not an allocator/RSS guarantee.
/// Each party prices 40L material + 8L local + 16L borrowed inputs, two vector
/// allowances, retained frame/output overlap, inline metadata and 8 KiB scratch.
/// Dealer prices 8L output masks + 16L borrowed input masks and one allowance.
pub fn resources(spec: Spec) -> Result<Resources, Error> {
    resources_for(spec, Boundary::Reconstructed)
}

/// Deferred mode emits no frames. It retains the original party charge while
/// moving local backing into a separately charged share owner. Both metadata
/// owners and backing overlap are conservatively priced until the party drops.
pub fn resources_for(spec: Spec, boundary: Boundary) -> Result<Resources, Error> {
    spec.validate()?;
    // Spec bounds all lane products and individual owner charges before these
    // calculations; aggregate sums and multiplication remain checked.
    let material = 40 * spec.lanes;
    let overlap = match boundary {
        Boundary::Reconstructed => checked_sum(&[frame_charge(spec), output_charge(spec)])?,
        Boundary::Deferred => deferred_charge(spec),
    };
    let party = checked_sum(&[
        size_of::<Party<'static>>(),
        64 * spec.lanes,
        2 * ALLOCATOR,
        overlap,
        WORKSPACE,
    ])?;
    let dealer = checked_sum(&[
        size_of::<OutputMasks<'static>>(),
        24 * spec.lanes,
        ALLOCATOR,
    ])?;
    let combined = checked_sum(&[party.checked_mul(2).ok_or(Error::Resource)?, dealer])?;
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        output_bits: spec.bits,
        logical_key_bits_per_party: 5 * usize::from(spec.bits) * spec.lanes,
        random_bytes: 32 + 48 * spec.lanes,
        output_mask_entropy_bits: usize::from(spec.bits) * spec.lanes,
        paired_issue_aes_blocks: 0,
        evaluation_aes_blocks_per_party: 0,
        frames_per_party: usize::from(boundary == Boundary::Reconstructed),
        frame_bytes: if boundary == Boundary::Reconstructed {
            HEADER + 8 * spec.lanes
        } else {
            0
        },
        total_peer_frame_bytes: if boundary == Boundary::Reconstructed {
            2 * (HEADER + 8 * spec.lanes)
        } else {
            0
        },
        party_material_payload_bytes: material,
        borrowed_input_mask_bytes: 16 * spec.lanes,
        borrowed_masked_input_bytes: 32 * spec.lanes,
        dealer_reservation_bytes: dealer,
        party_reservation_bytes: party,
        combined_reservation_bytes: combined,
    })
}

fn zeros<T: Default + zeroize::Zeroize>(
    len: usize,
    budget: &Budget,
) -> Result<Zeroizing<Vec<T>>, Error> {
    let mut values = Zeroizing::new(allocate(len, budget)?);
    values.resize_with(len, T::default);
    Ok(values)
}
fn draw(bits: u8, fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>) -> Result<u64, Error> {
    let mut bytes = Zeroizing::new([0; 8]);
    fill(&mut *bytes)?;
    Ok(u64::from_le_bytes(*bytes) & mask(bits))
}

struct Execution {
    // a, B, c, d1, d2; one party's independent additive shares only.
    material: Zeroizing<Vec<[u64; 5]>>,
    local: Zeroizing<Vec<u64>>,
    input_digest: [u8; 32],
    begun: bool,
}

/// Non-cloneable party-local material. Every failed execution burns state.
/// Explicitly cancel the sibling on failure. Cancel keeps its lease until drop.
pub struct Party<'a> {
    id: u8,
    boundary: Boundary,
    spec: Spec,
    binding: [u8; 32],
    state: Option<Execution>,
    // Fields drop in declaration order: erase/deallocate before refund.
    lease: Lease<'a>,
}
pub struct OutputMasks<'a> {
    values: Zeroizing<Vec<u64>>,
    _lease: Lease<'a>,
}
impl OutputMasks<'_> {
    /// Trusted dealer/composition boundary only, never an evaluator input.
    pub fn values(&self) -> &[u64] {
        &self.values
    }
}
pub struct Frame<'a> {
    bytes: Zeroizing<Vec<u8>>,
    _lease: Lease<'a>,
}
impl Frame<'_> {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}
pub struct MaskedOutput<'a> {
    values: Zeroizing<Vec<u64>>,
    _lease: Lease<'a>,
}
impl MaskedOutput<'_> {
    /// Common b*x + r_z in the original ring; no clear-output getter.
    pub fn values(&self) -> &[u64] {
        &self.values
    }
}

/// One party's additive share of b*x + r_z, not a reconstructed masked value.
/// No constructor, cloning, frame conversion or peer-share access is provided.
pub struct DeferredShare<'a> {
    role: u8,
    binding: [u8; 32],
    spec: Spec,
    values: Zeroizing<Vec<u64>>,
    // Erase/deallocate the backing before refunding its original budget lease.
    _lease: Lease<'a>,
}
impl DeferredShare<'_> {
    pub fn role(&self) -> u8 {
        self.role
    }
    pub fn binding(&self) -> &[u8; 32] {
        &self.binding
    }
    pub fn bits(&self) -> u8 {
        self.spec.bits
    }
    pub fn lanes(&self) -> usize {
        self.spec.lanes
    }
    /// Low-level own-share getter; never a common masked output.
    pub fn values(&self) -> &[u64] {
        &self.values
    }
}

pub fn issue<'a>(
    spec: Spec,
    context: Context,
    predicate_masks: &[u64],
    input_masks: &[u64],
    budget: &'a Budget,
) -> Result<(OutputMasks<'a>, [Party<'a>; 2]), Error> {
    issue_with(
        spec,
        context,
        predicate_masks,
        input_masks,
        budget,
        |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
    )
}

pub fn issue_deferred<'a>(
    spec: Spec,
    context: Context,
    predicate_masks: &[u64],
    input_masks: &[u64],
    budget: &'a Budget,
) -> Result<(OutputMasks<'a>, [Party<'a>; 2]), Error> {
    let cost = resources_for(spec, Boundary::Deferred)?;
    context.validate(spec.lanes)?;
    validate_inputs(spec, predicate_masks, input_masks)?;
    let reservation = budget.reserve(cost.combined_reservation_bytes)?;
    issue_reserved_with(
        spec,
        context,
        predicate_masks,
        input_masks,
        Boundary::Deferred,
        reservation,
        |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
    )
}

fn validate_inputs(spec: Spec, predicate: &[u64], input: &[u64]) -> Result<(), Error> {
    if predicate.len() != spec.lanes
        || input.len() != spec.lanes
        || predicate.iter().any(|&v| v > 1)
        || input.iter().any(|&v| v & !mask(spec.bits) != 0)
    {
        return Err(Error::Domain);
    }
    Ok(())
}

pub(super) fn issue_with<'a>(
    spec: Spec,
    context: Context,
    predicate_masks: &[u64],
    input_masks: &[u64],
    budget: &'a Budget,
    fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(OutputMasks<'a>, [Party<'a>; 2]), Error> {
    let cost = resources(spec)?;
    context.validate(spec.lanes)?;
    validate_inputs(spec, predicate_masks, input_masks)?;
    let reservation = budget.reserve(cost.combined_reservation_bytes)?;
    issue_reserved_with(
        spec,
        context,
        predicate_masks,
        input_masks,
        Boundary::Reconstructed,
        reservation,
        fill,
    )
}

/// Aggregate admission entry: split an exact reservation from the original
/// budget, never reacquire capacity inside an already admitted parent.
pub(super) fn issue_reserved_with<'a>(
    spec: Spec,
    context: Context,
    predicate_masks: &[u64],
    input_masks: &[u64],
    boundary: Boundary,
    mut reservation: Lease<'a>,
    mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(OutputMasks<'a>, [Party<'a>; 2]), Error> {
    let cost = resources_for(spec, boundary)?;
    context.validate(spec.lanes)?;
    validate_inputs(spec, predicate_masks, input_masks)?;
    if reservation.bytes != cost.combined_reservation_bytes {
        return Err(Error::Resource);
    }
    let budget = reservation.budget;
    // All leases precede secret owners, including on allocation/RNG unwind.
    let a = reservation.split(cost.party_reservation_bytes)?;
    let b = reservation.split(cost.party_reservation_bytes)?;
    let mut masks = zeros(spec.lanes, budget)?;
    let new_execution = || -> Result<Execution, Error> {
        Ok(Execution {
            material: zeros(spec.lanes, budget)?,
            local: zeros(spec.lanes, budget)?,
            input_digest: [0; 32],
            begun: false,
        })
    };
    let mut left = new_execution()?;
    let mut right = new_execution()?;
    let mut nonce = [0; 32];
    fill(&mut nonce)?;
    let binding = match boundary {
        Boundary::Reconstructed => binding(spec, context, nonce),
        Boundary::Deferred => {
            let mut hash = Sha256::new();
            hash.update(b"pllm.sigma.five-word-select.deferred.v1");
            hash.update(binding(spec, context, nonce));
            hash.finalize().into()
        }
    };
    for (i, (&beta, &rx)) in predicate_masks.iter().zip(input_masks).enumerate() {
        let beta = Zeroizing::new(beta);
        let rx = Zeroizing::new(rx);
        let rz = Zeroizing::new(draw(spec.bits, &mut fill)?);
        masks[i] = *rz;
        // Independently derive the five pinned correlations. Never reinterpret
        // the XOR predicate as an arithmetic masked bit.
        let correlations = Zeroizing::new([
            *beta,
            *rx,
            beta.wrapping_mul(*rx).wrapping_add(*rz),
            beta.wrapping_mul(2),
            0u64.wrapping_sub(beta.wrapping_mul(2).wrapping_mul(*rx)),
        ]);
        for j in 0..5 {
            let share = Zeroizing::new(draw(spec.bits, &mut fill)?);
            left.material[i][j] = *share;
            right.material[i][j] = correlations[j].wrapping_sub(*share) & mask(spec.bits);
        }
    }
    Ok((
        OutputMasks {
            values: masks,
            _lease: reservation,
        },
        [
            Party {
                id: 0,
                boundary,
                spec,
                binding,
                state: Some(left),
                lease: a,
            },
            Party {
                id: 1,
                boundary,
                spec,
                binding,
                state: Some(right),
                lease: b,
            },
        ],
    ))
}

fn binding(spec: Spec, context: Context, nonce: [u8; 32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(DOMAIN.as_bytes());
    hash.update([spec.bits]);
    hash.update((spec.lanes as u64).to_le_bytes());
    for part in [
        context.plan,
        context.session,
        context.operation,
        context.predicate_wire,
        context.input_wire,
        context.output_wire,
        nonce,
    ] {
        hash.update(part);
    }
    hash.update(context.first_tensor_index.to_le_bytes());
    hash.finalize().into()
}

fn evaluate(id: u8, bits: u8, bhat: u64, xhat: u64, words: &[u64; 5]) -> u64 {
    let mut share = words[2]
        .wrapping_sub(words[0].wrapping_mul(xhat))
        .wrapping_sub(words[1].wrapping_mul(bhat));
    if bhat == 0 {
        share = share
            .wrapping_add(words[3].wrapping_mul(xhat))
            .wrapping_add(words[4]);
    }
    if id == 1 {
        share = share.wrapping_add(bhat.wrapping_mul(xhat));
    }
    share & mask(bits)
}

impl<'a> Party<'a> {
    pub fn cancel(&mut self) {
        self.state = None;
    }

    pub fn begin(&mut self, predicate: &[u64], input: &[u64]) -> Result<Frame<'a>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        if self.boundary != Boundary::Reconstructed || state.begun {
            return Err(Error::Domain);
        }
        // Preflight BOTH complete arrays before writing any working value.
        validate_inputs(self.spec, predicate, input)?;
        let mut hash = Sha256::new();
        hash.update(b"pllm.sigma.select-masked-inputs.v1");
        hash.update((self.spec.lanes as u64).to_le_bytes());
        for (i, (&bhat, &xhat)) in predicate.iter().zip(input).enumerate() {
            hash.update(bhat.to_le_bytes());
            hash.update(xhat.to_le_bytes());
            state.local[i] = evaluate(self.id, self.spec.bits, bhat, xhat, &state.material[i]);
        }
        state.input_digest = hash.finalize().into();
        state.begun = true;
        let frame = self.frame(&state)?;
        self.state = Some(state);
        Ok(frame)
    }

    /// Consume the one peer frame. Moves the preallocated output backing;
    /// performs no allocation. Success and failure both retire local material.
    pub fn finish(&mut self, peer: &[u8]) -> Result<MaskedOutput<'a>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        if self.boundary != Boundary::Reconstructed {
            return Err(Error::Domain);
        }
        self.validate_peer(&state, peer)?;
        for (i, word) in peer[HEADER..].chunks_exact(8).enumerate() {
            state.local[i] = state.local[i]
                .wrapping_add(u64::from_le_bytes(word.try_into().unwrap()))
                & mask(self.spec.bits);
        }
        let lease = self.lease.split(output_charge(self.spec))?;
        Ok(MaskedOutput {
            values: state.local,
            _lease: lease,
        })
    }

    /// Fresh-only consuming evaluation. Moves preallocated local backing into
    /// a distinct own-share capability, without any allocation or wire frame.
    /// Every failure burns this owner; explicitly cancel its sibling if needed.
    pub fn into_deferred(
        mut self,
        predicate: &[u64],
        input: &[u64],
    ) -> Result<DeferredShare<'a>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        if self.boundary != Boundary::Deferred || state.begun {
            return Err(Error::Domain);
        }
        validate_inputs(self.spec, predicate, input)?;
        for (i, (&bhat, &xhat)) in predicate.iter().zip(input).enumerate() {
            state.local[i] = evaluate(self.id, self.spec.bits, bhat, xhat, &state.material[i]);
        }
        let lease = self.lease.split(deferred_charge(self.spec))?;
        drop(state.material);
        Ok(DeferredShare {
            role: self.id,
            binding: self.binding,
            spec: self.spec,
            values: state.local,
            _lease: lease,
        })
    }

    fn frame(&mut self, state: &Execution) -> Result<Frame<'a>, Error> {
        let mut bytes = Zeroizing::new(allocate(HEADER + 8 * self.spec.lanes, self.lease.budget)?);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.binding);
        bytes.extend_from_slice(&state.input_digest);
        bytes.extend_from_slice(&[self.id, OUTPUT_PHASE]);
        bytes.extend_from_slice(&(self.spec.lanes as u16).to_le_bytes());
        for value in &*state.local {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let lease = self.lease.split(frame_charge(self.spec))?;
        Ok(Frame {
            bytes,
            _lease: lease,
        })
    }

    fn validate_peer(&self, state: &Execution, peer: &[u8]) -> Result<(), Error> {
        if !state.begun
            || peer.len() != HEADER + 8 * self.spec.lanes
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != state.input_digest
            || peer[72] != 1 - self.id
            || peer[73] != OUTPUT_PHASE
            || peer[74..76] != (self.spec.lanes as u16).to_le_bytes()
        {
            return Err(Error::Context);
        }
        if peer[HEADER..]
            .chunks_exact(8)
            .any(|word| u64::from_le_bytes(word.try_into().unwrap()) & !mask(self.spec.bits) != 0)
        {
            return Err(Error::Domain);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
