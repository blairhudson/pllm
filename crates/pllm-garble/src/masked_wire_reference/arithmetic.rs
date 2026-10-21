//! D16/D17 bounded CPU masked-wire TR, GapARS and DReLU.
//!
//! Distinct approved GPU-derived DPF / full-output-ring mask profile, NOT the
//! leaking pinned CPU implementation or the existing Q12 gated-block profile.
//! A local research dealer issues two separate party-local owners. Each owns
//! only its own key/constants. Frames bind public context, input, role and phase;
//! they are NOT authenticated, replay-persistent or malicious-output verified.
//! GapARS requires the caller's unwrapped input in [-2^(n-2), 2^(n-2)); this
//! owner does not reconstruct or prove that private range. Caller input masks
//! must be fresh; only output masks are freshly sampled here.

use super::{mask, rescale};
pub use crate::shared_allocation::{Budget, Error};
use crate::{
    compact_dpf::gpu127::{self, GpuKey, Profile},
    shared_allocation::{allocate, Lease, MAX_ALLOCATION_BYTES},
};
use sha2::{Digest, Sha256};
use std::mem::size_of;
use zeroize::Zeroizing;

pub const DOMAIN: &str = "pllm.sigma.gpu127-full-ring-masks.v1";
pub const MAX_LANES: usize = super::MAX_LANES;
const MAGIC: &[u8; 8] = b"PLLMSA01";
const HEADER: usize = 8 + 32 + 32 + 1 + 1 + 2;
const ALLOCATOR: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    /// Exact unsigned floor, reduced to the n-shift output ring. Signed floor
    /// has the same reduced-ring representation.
    Tr { shift: u8 },
    /// Exact signed floor in the original ring, conditional on the quarter-ring
    /// public promise above. No saturation or runtime range disclosure.
    GapArs { shift: u8 },
    /// One-bit XOR-masked [signed input >= 0], including zero.
    Drelu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub bits: u8,
    pub lanes: usize,
    pub gate: Gate,
}
impl Spec {
    fn geometry(self) -> Result<(Profile, u8, usize), Error> {
        if !(3..=64).contains(&self.bits) || !(1..=MAX_LANES).contains(&self.lanes) {
            return Err(Error::Domain);
        }
        let (comparison, output, frames) = match self.gate {
            Gate::Tr { shift } if shift > 0 && shift < self.bits => (shift, self.bits - shift, 2),
            Gate::GapArs { shift } if shift > 0 && shift < self.bits - 1 => (shift, self.bits, 2),
            Gate::Drelu => (self.bits - 1, 1, 1),
            _ => return Err(Error::Domain),
        };
        // Public zero-extension, not truncation: GPU-derived DPF supports >=8
        // bits. Short low-bit comparisons retain their exact threshold/query.
        let profile = Profile::new(comparison.max(8)).map_err(primitive)?;
        Ok((profile, output, frames))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Context {
    pub plan: [u8; 32],
    pub session: [u8; 32],
    pub operation: [u8; 32],
    pub input_wire: [u8; 32],
    pub output_wire: [u8; 32],
    pub first_tensor_index: u64,
}
impl Context {
    fn validate(self, lanes: usize) -> Result<(), Error> {
        if self.input_wire == self.output_wire {
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
    pub comparison_bits: u8,
    pub logical_key_bits_per_party: usize,
    /// Exact RNG byte requests, including nonce, key roots, full output masks
    /// and arithmetic/Boolean correlation splits. Not an entropy/security claim.
    pub random_bytes: usize,
    pub output_mask_entropy_bits: usize,
    pub paired_issue_aes_blocks: usize,
    pub evaluation_aes_blocks_per_party: usize,
    pub frames_per_party: usize,
    pub frame_bytes: usize,
    pub total_peer_frame_bytes: usize,
    /// Delivery has no codec/transport yet. Logical key bits are NOT wire bytes.
    pub party_material_payload_bytes: usize,
    pub borrowed_input_mask_bytes: usize,
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

/// Public-only checked admission, before allocation or randomness. Prices fixed
/// inline DPF keys, all lane vectors, retained frames and output owners, masks,
/// and a 16 KiB logical stack/AES allowance. Not an allocator/RSS guarantee.
pub fn resources(spec: Spec) -> Result<Resources, Error> {
    let (profile, output_bits, frames) = spec.geometry()?;
    let key = profile.resources();
    let constants = match spec.gate {
        Gate::Tr { .. } => 2 * usize::from(output_bits),
        Gate::GapArs { .. } => 3 * usize::from(output_bits),
        Gate::Drelu => 1,
    };
    let per_lane_random = match spec.gate {
        Gate::Tr { .. } => 64,
        Gate::GapArs { .. } => 72,
        Gate::Drelu => 48,
    };
    let material = spec.lanes * size_of::<Lane>();
    let party = size_of::<Party<'static>>()
        + material
        + 24 * spec.lanes
        + 3 * ALLOCATOR
        + frames * frame_charge(spec)
        + output_charge(spec)
        + (8 << 10);
    let dealer = size_of::<OutputMasks<'static>>() + 16 * spec.lanes + ALLOCATOR;
    let combined = 2 * party + dealer;
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        output_bits,
        comparison_bits: profile.bits(),
        logical_key_bits_per_party: spec.lanes * (key.logical_key_bits + constants),
        random_bytes: 32 + per_lane_random * spec.lanes,
        output_mask_entropy_bits: usize::from(output_bits) * spec.lanes,
        paired_issue_aes_blocks: key.paired_issue_aes_blocks * spec.lanes,
        evaluation_aes_blocks_per_party: key.point_or_less_aes_blocks * spec.lanes,
        frames_per_party: frames,
        frame_bytes: HEADER + 8 * spec.lanes,
        total_peer_frame_bytes: 2 * frames * (HEADER + 8 * spec.lanes),
        party_material_payload_bytes: material,
        borrowed_input_mask_bytes: 8 * spec.lanes,
        borrowed_masked_input_bytes: 16 * spec.lanes,
        dealer_reservation_bytes: dealer,
        party_reservation_bytes: party,
        combined_reservation_bytes: combined,
    })
}

fn primitive(error: gpu127::Error) -> Error {
    match error {
        gpu127::Error::Domain | gpu127::Error::Encoding => Error::Domain,
        gpu127::Error::Resource => Error::Resource,
        gpu127::Error::Allocation => Error::Allocation,
        gpu127::Error::Randomness => Error::Randomness,
    }
}
fn zeros(len: usize, budget: &Budget) -> Result<Zeroizing<Vec<u64>>, Error> {
    let mut v = allocate(len, budget)?;
    v.resize(len, 0);
    Ok(Zeroizing::new(v))
}
fn draw(bits: u8, fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>) -> Result<u64, Error> {
    let mut bytes = Zeroizing::new([0; 8]);
    fill(&mut *bytes)?;
    Ok(u64::from_le_bytes(*bytes) & mask(bits))
}
fn split(
    value: u64,
    bits: u8,
    fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<[u64; 2], Error> {
    // Keep the correlation erased even when the split's RNG draw fails.
    let value = Zeroizing::new(value);
    let left = Zeroizing::new(draw(bits, fill)?);
    Ok([*left, value.wrapping_sub(*left) & mask(bits)])
}

struct Lane {
    key: GpuKey,
    constants: Zeroizing<[u64; 3]>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fresh,
    Carry,
    Output,
}
struct Execution {
    lanes: Vec<Lane>,
    input: Zeroizing<Vec<u64>>,
    local: Zeroizing<Vec<u64>>,
    input_digest: [u8; 32],
    phase: Phase,
}

/// No Clone/Debug/clear-mask getters. One party's material only. Failed calls
/// consume it; enclosing orchestration must also retire the other party.
pub struct Party<'a> {
    id: u8,
    spec: Spec,
    binding: [u8; 32],
    state: Option<Execution>,
    lease: Lease<'a>,
}
pub struct OutputMasks<'a> {
    values: Zeroizing<Vec<u64>>,
    _lease: Lease<'a>,
}
impl OutputMasks<'_> {
    /// Trusted dealer/composition boundary only. Never supplied to an evaluator.
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
    /// Common output + full-ring mask, or DReLU XOR its independent bit mask.
    pub fn values(&self) -> &[u64] {
        &self.values
    }
}
pub enum Step<'a> {
    Frame(Frame<'a>),
    Output(MaskedOutput<'a>),
}

pub fn issue<'a>(
    spec: Spec,
    context: Context,
    input_masks: &[u64],
    budget: &'a Budget,
) -> Result<(OutputMasks<'a>, [Party<'a>; 2]), Error> {
    issue_with(spec, context, input_masks, budget, |bytes| {
        getrandom::fill(bytes).map_err(|_| Error::Randomness)
    })
}

fn issue_with<'a>(
    spec: Spec,
    context: Context,
    input_masks: &[u64],
    budget: &'a Budget,
    mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(OutputMasks<'a>, [Party<'a>; 2]), Error> {
    let cost = resources(spec)?;
    context.validate(spec.lanes)?;
    if input_masks.len() != spec.lanes || input_masks.iter().any(|v| v & !mask(spec.bits) != 0) {
        return Err(Error::Domain);
    }
    let mut reservation = budget.reserve(cost.combined_reservation_bytes)?;
    // Declare every lease before secret owners: failure unwinding must erase
    // material before returning any of its capacity to the shared budget.
    // Split errors also occur before allocation or randomness.
    let a = reservation.split(cost.party_reservation_bytes)?;
    let b = reservation.split(cost.party_reservation_bytes)?;
    let mut masks = zeros(spec.lanes, budget)?;
    let new_execution = || -> Result<Execution, Error> {
        Ok(Execution {
            lanes: allocate(spec.lanes, budget)?,
            input: zeros(spec.lanes, budget)?,
            local: zeros(spec.lanes, budget)?,
            input_digest: [0; 32],
            phase: Phase::Fresh,
        })
    };
    let mut left = new_execution()?;
    let mut right = new_execution()?;
    let mut nonce = [0; 32];
    fill(&mut nonce)?;
    let binding = binding(spec, context, nonce);
    let (profile, output_bits, _) = spec.geometry()?;
    for (index, &rin) in input_masks.iter().enumerate() {
        let rin = Zeroizing::new(rin);
        // D17: always sample the COMPLETE output ring, never a one-bit mask
        // for a multi-bit reconstructed arithmetic result.
        let rout = Zeroizing::new(draw(output_bits, &mut fill)?);
        masks[index] = *rout;
        let (point, constants) = match spec.gate {
            Gate::Tr { shift } | Gate::GapArs { shift } => {
                let gap = matches!(spec.gate, Gate::GapArs { .. });
                let boolean = Zeroizing::new(draw(1, &mut fill)?);
                let carry = Zeroizing::new(split(*boolean, output_bits, &mut fill)?);
                let wrap = Zeroizing::new(if gap {
                    split(
                        (*rin >> (spec.bits - 1)) << (spec.bits - shift),
                        spec.bits,
                        &mut fill,
                    )?
                } else {
                    [0; 2]
                });
                let correction = Zeroizing::new(split(
                    rout.wrapping_sub(*rin >> shift).wrapping_sub(if gap {
                        1 << (spec.bits - shift - 2)
                    } else {
                        0
                    }),
                    output_bits,
                    &mut fill,
                )?);
                (
                    *rin & mask(shift),
                    Zeroizing::new([
                        [carry[0], wrap[0], correction[0]],
                        [carry[1], wrap[1], correction[1]],
                    ]),
                )
            }
            Gate::Drelu => {
                let neg = Zeroizing::new(0u64.wrapping_sub(*rin) & mask(spec.bits));
                let value = Zeroizing::new(*rout ^ (*neg >> (spec.bits - 1)) ^ 1);
                let shares = Zeroizing::new(split(*value, 1, &mut fill)?);
                (
                    *neg & mask(spec.bits - 1),
                    Zeroizing::new([[shares[0], 0, 0], [shares[1], 0, 0]]),
                )
            }
        };
        let point = Zeroizing::new(point);
        let [a, b] = GpuKey::issue_with(profile, *point, |bytes| {
            fill(bytes).map_err(|_| gpu127::Error::Randomness)
        })
        .map_err(primitive)?;
        left.lanes.push(Lane {
            key: a,
            constants: Zeroizing::new(constants[0]),
        });
        right.lanes.push(Lane {
            key: b,
            constants: Zeroizing::new(constants[1]),
        });
    }
    Ok((
        OutputMasks {
            values: masks,
            _lease: reservation,
        },
        [
            Party {
                id: 0,
                spec,
                binding,
                state: Some(left),
                lease: a,
            },
            Party {
                id: 1,
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
    hash.update(Profile::DOMAIN.as_bytes());
    hash.update([spec.bits]);
    hash.update((spec.lanes as u64).to_le_bytes());
    hash.update(match spec.gate {
        Gate::Tr { shift } => [0, shift],
        Gate::GapArs { shift } => [1, shift],
        Gate::Drelu => [2, 0],
    });
    for part in [
        context.plan,
        context.session,
        context.operation,
        context.input_wire,
        context.output_wire,
        nonce,
    ] {
        hash.update(part);
    }
    hash.update(context.first_tensor_index.to_le_bytes());
    hash.finalize().into()
}

impl<'a> Party<'a> {
    pub fn cancel(&mut self) {
        self.state = None;
    }

    pub fn begin(&mut self, masked: &[u64]) -> Result<Frame<'a>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        if state.phase != Phase::Fresh
            || masked.len() != self.spec.lanes
            || masked.iter().any(|v| v & !mask(self.spec.bits) != 0)
        {
            return Err(Error::Domain);
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm.sigma.masked-arithmetic-input.v1");
        for (index, (&input, lane)) in masked.iter().zip(&state.lanes).enumerate() {
            state.input[index] = input;
            hash.update(input.to_le_bytes());
            state.local[index] = match self.spec.gate {
                Gate::Tr { shift } | Gate::GapArs { shift } => {
                    let adjusted = rescale::adjusted(
                        input,
                        self.spec.bits,
                        matches!(self.spec.gate, Gate::GapArs { .. }),
                    );
                    u64::from(lane.key.less(adjusted & mask(shift)).map_err(primitive)?)
                        ^ (lane.constants[0] & 1)
                }
                Gate::Drelu => {
                    (u64::from(self.id) * (input >> (self.spec.bits - 1)))
                        ^ u64::from(
                            lane.key
                                .less(mask(self.spec.bits - 1) - (input & mask(self.spec.bits - 1)))
                                .map_err(primitive)?,
                        )
                        ^ lane.constants[0]
                }
            };
        }
        state.input_digest = hash.finalize().into();
        state.phase = if self.spec.gate == Gate::Drelu {
            Phase::Output
        } else {
            Phase::Carry
        };
        let frame = self.frame(&state)?;
        self.state = Some(state);
        Ok(frame)
    }

    pub fn advance(&mut self, peer: &[u8]) -> Result<Step<'a>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        self.validate_peer(&state, peer)?;
        if state.phase == Phase::Carry {
            let (shift, gap) = match self.spec.gate {
                Gate::Tr { shift } => (shift, false),
                Gate::GapArs { shift } => (shift, true),
                Gate::Drelu => return Err(Error::Domain),
            };
            for (index, word) in peer[HEADER..].chunks_exact(8).enumerate() {
                let opened = state.local[index] ^ u64::from_le_bytes(word.try_into().unwrap());
                state.local[index] = rescale::finish(
                    self.id,
                    state.input[index],
                    self.spec.bits,
                    shift,
                    gap,
                    &state.lanes[index].constants,
                    opened,
                );
            }
            state.phase = Phase::Output;
            let frame = self.frame(&state)?;
            self.state = Some(state);
            Ok(Step::Frame(frame))
        } else {
            let output_bits = self.spec.geometry()?.1;
            for (index, word) in peer[HEADER..].chunks_exact(8).enumerate() {
                let other = u64::from_le_bytes(word.try_into().unwrap());
                state.local[index] = if self.spec.gate == Gate::Drelu {
                    state.local[index] ^ other
                } else {
                    state.local[index].wrapping_add(other) & mask(output_bits)
                };
            }
            let lease = self.lease.split(output_charge(self.spec))?;
            Ok(Step::Output(MaskedOutput {
                values: state.local,
                _lease: lease,
            }))
        }
    }

    fn frame(&mut self, state: &Execution) -> Result<Frame<'a>, Error> {
        let mut bytes = Zeroizing::new(allocate(HEADER + 8 * self.spec.lanes, self.lease.budget)?);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.binding);
        bytes.extend_from_slice(&state.input_digest);
        bytes.extend_from_slice(&[self.id, if state.phase == Phase::Carry { 0 } else { 1 }]);
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
        if state.phase == Phase::Fresh {
            return Err(Error::Context);
        }
        if peer.len() != HEADER + 8 * self.spec.lanes
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != state.input_digest
            || peer[72] != 1 - self.id
            || peer[73] != if state.phase == Phase::Carry { 0 } else { 1 }
            || peer[74..76] != (self.spec.lanes as u16).to_le_bytes()
        {
            return Err(Error::Context);
        }
        let bits = if state.phase == Phase::Carry {
            1
        } else {
            self.spec.geometry()?.1
        };
        if peer[HEADER..]
            .chunks_exact(8)
            .any(|word| u64::from_le_bytes(word.try_into().unwrap()) & !mask(bits) != 0)
        {
            return Err(Error::Domain);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
