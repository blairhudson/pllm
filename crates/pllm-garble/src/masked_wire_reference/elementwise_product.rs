//! Consuming two-input product material for D29's bounded local composition.
//!
//! Private until canonical-record/owner review. No public Multiply or material
//! importer. Uses the existing negative-mask algebra, not a matrix singleton.
//! Capacities are logical owners; allocator overhead/native stacks are excluded.
//! Ordinary backing OOM may abort; this is not recoverable allocator admission.

use super::{mask, product};
use crate::shared_allocation::{AllocationContext, Budget, Error, LeaseFor, MAX_ALLOCATION_BYTES};
use sha2::{Digest, Sha256};
use std::mem::size_of;
use zeroize::Zeroizing;

pub(super) const DOMAIN: &[u8] = b"pllm.sigma.elementwise-product.v1";
const INPUT_DOMAIN: &[u8] = b"pllm.sigma.elementwise-product.common-inputs.v1";
const MAGIC: &[u8; 8] = b"PLLMSP01";
const BITS: u8 = 50;
const HEADER: usize = 76;

#[derive(Clone, Copy, Debug)]
pub(super) struct Context {
    pub region: [u8; 32],
    pub stage: [u8; 32],
    pub inputs: [[u8; 32]; 2],
    pub output: [u8; 32],
    pub first: u64,
}
impl Context {
    fn check(self, lanes: usize) -> Result<(), Error> {
        if self.inputs[0] == self.inputs[1]
            || self.inputs.contains(&self.output)
            || self.first.checked_add(lanes as u64).is_none()
        {
            return Err(Error::Context);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Resources {
    pub lanes: usize,
    pub frame_bytes: usize,
    pub total_peer_bytes: usize,
    pub random_bytes: usize,
    pub dealer_bytes: usize,
    pub party_bytes: usize,
    pub combined_bytes: usize,
    pub construction_controls: usize,
    pub factory_transition_bytes: usize,
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Resource)
}

struct Execution {
    coefficients: Zeroizing<Box<[[u64; 3]]>>,
    local: Zeroizing<Box<[u64]>>,
    common: [u8; 32],
    begun: bool,
}
pub(super) struct PartyFor<C: AllocationContext> {
    context: Context,
    binding: [u8; 32],
    lanes: usize,
    role: u8,
    execution: Option<Execution>,
    // Even cancel retains its actual wrapper/control reservation until drop.
    lease: LeaseFor<C>,
}
pub(super) struct MasksFor<C: AllocationContext> {
    values: Zeroizing<Box<[u64]>>,
    _lease: LeaseFor<C>,
}
pub(super) struct OutputFor<C: AllocationContext> {
    values: Zeroizing<Box<[u64]>>,
    _lease: LeaseFor<C>,
}
pub(super) struct FrameFor<C: AllocationContext> {
    bytes: Zeroizing<Box<[u8]>>,
    _lease: LeaseFor<C>,
}
impl<C: AllocationContext> MasksFor<C> {
    pub(super) fn values(&self) -> &[u64] {
        &self.values
    }
}
impl<C: AllocationContext> OutputFor<C> {
    pub(super) fn values(&self) -> &[u64] {
        &self.values
    }
}
impl<C: AllocationContext> FrameFor<C> {
    pub(super) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Concrete source-level controls, intentionally retained in the aggregate.
/// These are NOT a guessed stack/workspace allowance. Box construction uses an
/// exact vec![Copy; n] backing and a backing-preserving Vec -> Box transition.
fn controls<C: AllocationContext>() -> Result<usize, Error> {
    [
        size_of::<Context>(),
        size_of::<Resources>(),
        size_of::<Result<Resources, Error>>(),
        size_of::<[LeaseFor<C>; 3]>(),
        C::TRANSIENT_LINK_BYTES,
        size_of::<[Sha256; 2]>(),
        size_of::<[sha2::digest::Output<Sha256>; 2]>(),
        size_of::<[&[u8]; 10]>(),
        size_of::<[u8; 2]>(),
        size_of::<[u8; 8]>(),
        size_of::<Zeroizing<[u8; 32]>>(),
        size_of::<Zeroizing<[u8; 8]>>(),
        size_of::<Zeroizing<[u64; 4]>>(),
        size_of::<[Zeroizing<[u64; 3]>; 3]>(),
        size_of::<Vec<[u64; 3]>>(),
        size_of::<Box<[[u64; 3]]>>(),
        size_of::<Vec<u64>>(),
        size_of::<Box<[u64]>>(),
        size_of::<Vec<u8>>(),
        size_of::<Box<[u8]>>(),
        size_of::<(MasksFor<C>, [PartyFor<C>; 2])>(),
        size_of::<Result<(MasksFor<C>, [PartyFor<C>; 2]), Error>>(),
        size_of::<Result<FrameFor<C>, Error>>(),
        size_of::<Result<OutputFor<C>, Error>>(),
    ]
    .into_iter()
    .try_fold(0, add)
}

/// Allocation-/hash-free. Input owners remain separately charged and live.
pub(super) fn resources_for<C: AllocationContext>(lanes: usize) -> Result<Resources, Error> {
    if !(1..=256).contains(&lanes) {
        return Err(Error::Domain);
    }
    let frame_bytes = add(HEADER, mul(8, lanes)?)?;
    // Party execution, retained send, own output and a complete borrowed receive
    // coexist; no credit from transferring the already allocated local backing.
    let construction_controls = controls::<C>()?;
    // Also retain a complete largest factory backing through Vec -> Box return.
    // No credit depends on the standard library preserving that backing in place.
    let factory_transition_bytes = mul(size_of::<[u64; 3]>(), lanes)?.max(frame_bytes);
    let party_bytes = [
        size_of::<PartyFor<C>>(),
        mul(32, lanes)?,
        size_of::<FrameFor<C>>(),
        frame_bytes,
        frame_bytes,
        size_of::<OutputFor<C>>(),
        mul(8, lanes)?,
        construction_controls,
        factory_transition_bytes,
    ]
    .into_iter()
    .try_fold(0, add)?;
    let dealer_bytes = [
        size_of::<MasksFor<C>>(),
        mul(8, lanes)?,
        construction_controls,
        factory_transition_bytes,
    ]
    .into_iter()
    .try_fold(0, add)?;
    let combined_bytes = add(dealer_bytes, mul(2, party_bytes)?)?;
    if combined_bytes > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        lanes,
        frame_bytes,
        total_peer_bytes: mul(2, frame_bytes)?,
        random_bytes: add(32, mul(32, lanes)?)?,
        dealer_bytes,
        party_bytes,
        combined_bytes,
        construction_controls,
        factory_transition_bytes,
    })
}

fn zeros_u64(n: usize, budget: &Budget) -> Result<Zeroizing<Box<[u64]>>, Error> {
    budget.allocation()?;
    Ok(Zeroizing::new(vec![0; n].into_boxed_slice()))
}
fn coefficients(n: usize, budget: &Budget) -> Result<Zeroizing<Box<[[u64; 3]]>>, Error> {
    budget.allocation()?;
    Ok(Zeroizing::new(vec![[0; 3]; n].into_boxed_slice()))
}
fn canonical(values: &[u64], lanes: usize) -> Result<(), Error> {
    if values.len() != lanes || values.iter().any(|v| v & !mask(BITS) != 0) {
        return Err(Error::Domain);
    }
    Ok(())
}
fn part(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}
fn binding(context: Context, lanes: usize, nonce: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    for p in [
        DOMAIN,
        &context.region,
        &context.stage,
        &context.inputs[0],
        &context.inputs[1],
        &context.output,
        &[50, 24],
        &(lanes as u64).to_le_bytes(),
        &context.first.to_le_bytes(),
        nonce,
    ] {
        part(&mut h, p);
    }
    h.finalize().into()
}
fn common(context: Context, binding: &[u8; 32], first: &[u64], second: &[u64]) -> [u8; 32] {
    let mut h = Sha256::new();
    for p in [
        INPUT_DOMAIN,
        binding,
        &context.inputs[0],
        &context.inputs[1],
        &(first.len() as u64).to_le_bytes(),
    ] {
        part(&mut h, p);
    }
    for values in [first, second] {
        h.update((8 * values.len() as u64).to_le_bytes());
        for value in values {
            h.update(value.to_le_bytes());
        }
    }
    h.finalize().into()
}

/// Internal reserved entry only; no masks or evaluator values are imported from
/// callers. Typed owner adapter validates the two genuine lookup capabilities.
pub(super) fn issue_reserved<C: AllocationContext>(
    context: Context,
    first: &[u64],
    second: &[u64],
    mut lease: LeaseFor<C>,
    mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(MasksFor<C>, [PartyFor<C>; 2]), Error> {
    let cost = resources_for::<C>(first.len())?;
    context.check(cost.lanes)?;
    canonical(first, cost.lanes)?;
    canonical(second, cost.lanes)?;
    if lease.bytes != cost.combined_bytes {
        return Err(Error::Resource);
    }
    let allocation = lease.budget.share();
    let budget = allocation.budget();
    let left_lease = lease.split(cost.party_bytes)?;
    let right_lease = lease.split(cost.party_bytes)?;
    let mut masks = zeros_u64(cost.lanes, budget)?;
    let mut left = Execution {
        coefficients: coefficients(cost.lanes, budget)?,
        local: zeros_u64(cost.lanes, budget)?,
        common: [0; 32],
        begun: false,
    };
    let mut right = Execution {
        coefficients: coefficients(cost.lanes, budget)?,
        local: zeros_u64(cost.lanes, budget)?,
        common: [0; 32],
        begun: false,
    };
    let mut nonce = Zeroizing::new([0; 32]);
    fill(&mut *nonce)?;
    let binding = binding(context, cost.lanes, &nonce);
    for lane in 0..cost.lanes {
        // Four independent draws: rP, A0, B0, C0; never reuse a split for another
        // coefficient/lane/invocation. Reduction is uniform in the power-of-two ring.
        let mut draws = Zeroizing::new([0u64; 4]);
        for draw in draws.iter_mut() {
            let mut bytes = Zeroizing::new([0; 8]);
            fill(&mut *bytes)?;
            *draw = u64::from_le_bytes(*bytes) & mask(BITS);
        }
        masks[lane] = draws[0];
        let totals = Zeroizing::new([
            product::negative_mask(first[lane]) & mask(BITS),
            product::negative_mask(second[lane]) & mask(BITS),
            product::correlation(first[lane], second[lane], draws[0]) & mask(BITS),
        ]);
        for coefficient in 0..3 {
            left.coefficients[lane][coefficient] = draws[coefficient + 1];
            right.coefficients[lane][coefficient] =
                totals[coefficient].wrapping_sub(draws[coefficient + 1]) & mask(BITS);
        }
    }
    Ok((
        MasksFor {
            values: masks,
            _lease: lease,
        },
        [
            PartyFor {
                context,
                binding,
                lanes: cost.lanes,
                role: 0,
                execution: Some(left),
                lease: left_lease,
            },
            PartyFor {
                context,
                binding,
                lanes: cost.lanes,
                role: 1,
                execution: Some(right),
                lease: right_lease,
            },
        ],
    ))
}

impl<C: AllocationContext> PartyFor<C> {
    pub(super) fn cancel(&mut self) {
        self.execution = None;
    }

    pub(super) fn begin(&mut self, first: &[u64], second: &[u64]) -> Result<FrameFor<C>, Error> {
        let mut execution = self.execution.take().ok_or(Error::Consumed)?;
        if execution.begun {
            return Err(Error::Consumed);
        }
        canonical(first, self.lanes)?;
        canonical(second, self.lanes)?;
        execution.common = common(self.context, &self.binding, first, second);
        for lane in 0..self.lanes {
            execution.local[lane] = product::evaluate(
                self.role,
                first[lane],
                second[lane],
                &execution.coefficients[lane],
            ) & mask(BITS);
        }
        let cost = resources_for::<C>(self.lanes)?;
        let frame_lease = self
            .lease
            .split(add(size_of::<FrameFor<C>>(), cost.frame_bytes)?)?;
        self.lease.budget.budget().allocation()?;
        let mut bytes = Zeroizing::new(vec![0; cost.frame_bytes].into_boxed_slice());
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..40].copy_from_slice(&self.binding);
        bytes[40..72].copy_from_slice(&execution.common);
        bytes[72] = self.role;
        bytes[73] = 1;
        bytes[74..76].copy_from_slice(&(self.lanes as u16).to_le_bytes());
        for (word, value) in bytes[HEADER..]
            .chunks_exact_mut(8)
            .zip(execution.local.iter())
        {
            word.copy_from_slice(&value.to_le_bytes());
        }
        execution.begun = true;
        self.execution = Some(execution);
        Ok(FrameFor {
            bytes,
            _lease: frame_lease,
        })
    }

    pub(super) fn finish(&mut self, peer: &[u8]) -> Result<OutputFor<C>, Error> {
        let mut execution = self.execution.take().ok_or(Error::Consumed)?;
        if !execution.begun {
            return Err(Error::Consumed);
        }
        if peer.len() != HEADER + 8 * self.lanes
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != execution.common
            || peer[72] != 1 - self.role
            || peer[73] != 1
            || u16::from_le_bytes(peer[74..76].try_into().unwrap()) as usize != self.lanes
        {
            return Err(Error::Context);
        }
        // Whole peer validates BEFORE any addition/output mutation.
        for word in peer[HEADER..].chunks_exact(8) {
            if u64::from_le_bytes(word.try_into().unwrap()) & !mask(BITS) != 0 {
                return Err(Error::Domain);
            }
        }
        for (value, word) in execution
            .local
            .iter_mut()
            .zip(peer[HEADER..].chunks_exact(8))
        {
            *value = value.wrapping_add(u64::from_le_bytes(word.try_into().unwrap())) & mask(BITS);
        }
        let lease = self
            .lease
            .split(add(size_of::<OutputFor<C>>(), mul(8, self.lanes)?)?)?;
        Ok(OutputFor {
            values: execution.local,
            _lease: lease,
        })
    }
}

#[cfg(test)]
#[path = "elementwise_product_tests.rs"]
mod tests;
