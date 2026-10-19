//! Bounded, independently implemented arithmetic FuseFSS backend research slice.
//!
//! Governing source: arXiv:2606.09551v1, Appendix I.2 (pp.20–21) and N
//! (pp.26–27). I.2 specifies a private vector-payload lookup interface, not a
//! particular key algorithm. We instantiate it by the telescoping equation
//! `LUT(u)=v_last + sum_(i=1..M-1) (v_(i-1)-v_i)*[u<alpha_i]`.
//! Each summand is a *private payload* vector DCF over (Z/2^n)^2. An additional
//! zero-threshold/zero-payload DCF is always issued/evaluated, so exactly M keys
//! per party are charged, including padding and zero thresholds. That private
//! control gives neither party boundaries, the mask, deltas or clear coefficients.
//! The separate public-boundary specialization reuses the original PUBLIC table
//! and universal DCF; it never exposes mask-shifted order or selected clear
//! coefficients. Neither choice claims the authors' evaluated backend or costs.
//!
//! Security assumes a trusted independent dealer, at most one semi-honest online
//! corruption, fresh material, and the PRG/FSS assumptions of compact DCF.
//! Functional tests do not constitute independent cryptographic/side-channel
//! review. Frames bind public context, role, fresh issuance and order; they are
//! NOT authenticated network transport. The affine adapter still rejects Boolean
//! outputs/nonidentity Phi; separate consuming predicate/conversion owners expose
//! those bounded primitives without declaring a complete compiled gate schedule.
//! Original input shares feed a fresh degree-one Beaver Horner product, with no
//! rescale. Initial limit: four lanes; no automatic whole-tensor execution.
mod compiled;
pub mod conversions;
mod dcf;
#[cfg(test)]
mod gate_conformance;
pub mod lookup;
pub mod predicates;

pub use compiled::{CompiledGelu, CompiledResources};

use crate::shared_affine_lookup_reference::{issue_ring, RingParty, RingTable};
use crate::shared_rescale_reference::Context;
use crate::shared_wide_product::ring;
use sha2::{Digest, Sha256};
use std::{
    mem::size_of,
    sync::atomic::{AtomicUsize, Ordering},
};
use zeroize::Zeroizing;

pub const MAX_ALLOCATION_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_LANES: usize = 4;
pub const HEADER_BYTES: usize = 80;
// Frozen clear-instance ceiling plus canonical descriptor ceiling. Callback
// must include all live compiler owners/scratch within this combined allowance.
const COMPILER_OVERLAP_BYTES: usize = 1024 * 1024 + 256 * 1024;
const MAGIC: &[u8; 8] = b"PLLMFS01";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Domain,
    Unsupported,
    Resource,
    Allocation,
    Randomness,
    Context,
    Consumed,
}

fn primitive_error(error: crate::shared_rescale_reference::Error) -> Error {
    use crate::shared_rescale_reference::Error as Primitive;
    match error {
        Primitive::Domain => Error::Domain,
        Primitive::Resource => Error::Resource,
        Primitive::Randomness => Error::Randomness,
        Primitive::Consumed => Error::Consumed,
        Primitive::Context => Error::Context,
    }
}

/// Shared logical-allocation envelope. Borrowing prevents budget destruction
/// before exported parties/frames/outputs. No global allocator/RSS claim.
pub struct Budget {
    limit: usize,
    used: AtomicUsize,
    #[cfg(test)]
    fail_after: std::sync::atomic::AtomicIsize,
}
impl Budget {
    pub fn new(limit: usize) -> Result<Self, Error> {
        if limit == 0 || limit > MAX_ALLOCATION_BYTES {
            return Err(Error::Resource);
        }
        Ok(Self {
            limit,
            used: AtomicUsize::new(0),
            #[cfg(test)]
            fail_after: std::sync::atomic::AtomicIsize::new(-1),
        })
    }
    pub fn used_bytes(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }
    fn reserve(&self, bytes: usize) -> Result<Lease<'_>, Error> {
        self.used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map_err(|_| Error::Resource)?;
        Ok(Lease {
            budget: self,
            bytes,
        })
    }
    fn allocation(&self) -> Result<(), Error> {
        #[cfg(test)]
        {
            let n = self.fail_after.load(Ordering::SeqCst);
            if n == 0 {
                return Err(Error::Allocation);
            }
            if n > 0 {
                self.fail_after.fetch_sub(1, Ordering::SeqCst);
            }
        }
        Ok(())
    }
}
struct Lease<'a> {
    budget: &'a Budget,
    bytes: usize,
}
impl<'a> Lease<'a> {
    fn split(&mut self, bytes: usize) -> Result<Lease<'a>, Error> {
        self.bytes = self.bytes.checked_sub(bytes).ok_or(Error::Resource)?;
        Ok(Lease {
            budget: self.budget,
            bytes,
        })
    }
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
fn allocate<T>(capacity: usize, budget: &Budget) -> Result<Vec<T>, Error> {
    budget.allocation()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Allocation)?;
    exact_capacity(values, capacity)
}
// Exported frames/outputs may outlive all party slack. Their leases charge the
// backing capacity, so reject allocator over-grants before secret data is filled.
fn exact_capacity<T>(values: Vec<T>, capacity: usize) -> Result<Vec<T>, Error> {
    if values.capacity() != capacity {
        return Err(Error::Resource);
    }
    Ok(values)
}
fn random() -> Result<u64, Error> {
    let mut bytes = Zeroizing::new([0; 8]);
    getrandom::fill(&mut *bytes).map_err(|_| Error::Randomness)?;
    Ok(u64::from_le_bytes(*bytes))
}

#[derive(Clone, Copy)]
pub struct Spec {
    pub bits: u8,
    pub source_intervals: usize,
    pub lanes: usize,
    pub descriptor_digest: [u8; 32],
    pub source_digest: [u8; 32],
    pub boolean_outputs: usize,
    pub identity_phi: bool,
}
impl Spec {
    pub fn padded_intervals(&self) -> Result<usize, Error> {
        if !(1..=50).contains(&self.bits)
            || !(1..=512).contains(&self.source_intervals)
            || !(1..=MAX_LANES).contains(&self.lanes)
            || self.source_intervals as u64 > 1u64 << self.bits
        {
            return Err(Error::Domain);
        }
        if self.boolean_outputs != 0 || !self.identity_phi {
            return Err(Error::Unsupported);
        }
        Ok(((self.source_intervals + 1) as u64).min(1u64 << self.bits) as usize)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Resources {
    pub lookup_backend: LookupBackend,
    pub padded_intervals: usize,
    pub dcf_keys_per_party: usize,
    pub dcf_evaluations_per_party: usize,
    pub party_key_logical_bytes: usize,
    pub public_table_bytes_per_party: usize,
    pub public_source_table_bytes: usize,
    pub masked_frame_bytes: usize,
    pub product_frame_bytes: usize,
    pub total_peer_frame_bytes: usize,
    pub output_bytes_per_party: usize,
    pub compiler_overlap_bytes: usize,
    pub combined_reservation_bytes: usize,
    pub party_reservation_bytes: usize,
}

/// Separate key constructions with identical frozen scalar arithmetic. Public
/// boundary reuse requires an original public descriptor, not a private LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupBackend {
    PrivateVector,
    PublicBoundary,
}

pub fn resources(spec: Spec) -> Result<Resources, Error> {
    resources_for(spec, None)
}

fn resources_for(spec: Spec, public: Option<&RingTable>) -> Result<Resources, Error> {
    let m = spec.padded_intervals()?;
    let lanes = spec.lanes;
    let masked = HEADER_BYTES + 8 * lanes;
    let product = HEADER_BYTES + 16 * lanes;
    let (
        backend,
        key_count,
        key_bytes,
        public_bytes,
        source_bytes,
        party_keys,
        issue_scratch,
        eval_scratch,
    ) = if let Some(table) = public {
        if table.bits() != spec.bits || table.intervals() != m {
            return Err(Error::Domain);
        }
        let cost: crate::shared_affine_lookup_reference::RingResources =
            table.resources().map_err(primitive_error)?;
        (
            LookupBackend::PublicBoundary,
            lanes,
            lanes * (cost.party_private_key_payload_bytes[0] + 32),
            lanes * cost.public_table_payload_bytes,
            cost.public_table_payload_bytes,
            lanes
                * (size_of::<Lane>()
                    + cost.party_table_capacity_bytes[0]
                    + cost.party_private_key_capacity_bytes[0]),
            cost.issuance_scratch_bytes,
            cost.evaluation_scratch_bytes_per_party,
        )
    } else {
        let cost = dcf::Key::resources(spec.bits)?;
        (
            LookupBackend::PrivateVector,
            lanes * m,
            lanes * (m * dcf::Key::logical_bytes(spec.bits) + 48),
            0,
            0,
            lanes * (m * (cost.key_owner_bytes + cost.key_capacity_bytes) + size_of::<Lane>()),
            cost.issuance_scratch_bytes,
            0,
        )
    };
    // All state variants coexist in this conservative envelope: original x,
    // own opening, coefficients, product openings, exported output, both frames.
    // Extra 4096 covers stack cryptographic scratch and fixed-size owner metadata.
    let party_reservation = party_keys + lanes * 128 + masked + product + 4096 + eval_scratch;
    let artifact = (m + 1) * 8 + m * 16;
    let combined = 2 * party_reservation + COMPILER_OVERLAP_BYTES + artifact + 4096 + issue_scratch;
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        lookup_backend: backend,
        padded_intervals: m,
        dcf_keys_per_party: key_count,
        dcf_evaluations_per_party: lanes * m,
        party_key_logical_bytes: key_bytes,
        public_table_bytes_per_party: public_bytes,
        public_source_table_bytes: source_bytes,
        masked_frame_bytes: masked,
        product_frame_bytes: product,
        total_peer_frame_bytes: 2 * (masked + product),
        output_bytes_per_party: lanes * 8,
        compiler_overlap_bytes: COMPILER_OVERLAP_BYTES,
        combined_reservation_bytes: combined,
        party_reservation_bytes: party_reservation,
    })
}

/// Dealer-only compiled affine partition. Constructor copies bounded slices;
/// callers must also erase their original compiler-owned secret buffers.
/// Intentionally no Clone, Debug, clear-data getters, or serialization.
pub struct DealerPartition {
    mask: Zeroizing<u64>,
    boundaries: Zeroizing<Vec<u64>>,
    payloads: Zeroizing<Vec<[u64; 2]>>,
}
impl DealerPartition {
    pub fn from_slices(
        mask: u64,
        boundaries: &[u64],
        payloads: &[[u64; 2]],
    ) -> Result<Self, Error> {
        if payloads.is_empty() || payloads.len() > 513 || boundaries.len() != payloads.len() + 1 {
            return Err(Error::Domain);
        }
        let mut b = Zeroizing::new(Vec::new());
        let mut p = Zeroizing::new(Vec::new());
        b.try_reserve_exact(boundaries.len())
            .map_err(|_| Error::Allocation)?;
        p.try_reserve_exact(payloads.len())
            .map_err(|_| Error::Allocation)?;
        b.extend_from_slice(boundaries);
        p.extend_from_slice(payloads);
        Ok(Self {
            mask: Zeroizing::new(mask),
            boundaries: b,
            payloads: p,
        })
    }
    fn validate(&self, bits: u8, mask: u64, m: usize) -> Result<(), Error> {
        let n = 1u64 << bits;
        if *self.mask != mask
            || self.payloads.len() != m
            || self.boundaries.len() != m + 1
            || self.boundaries[0] != 0
            || self.boundaries[m] != n
            || self.boundaries.windows(2).any(|w| w[0] >= w[1])
            || self.payloads.iter().flatten().any(|v| *v >= n)
        {
            return Err(Error::Domain);
        }
        Ok(())
    }
}

enum Lookup {
    Private {
        keys: Vec<dcf::Key>,
        base: Zeroizing<[u64; 2]>,
    },
    Public(RingParty),
    Consumed,
}

impl Lookup {
    fn evaluate(self, party: u8, opened: u64, mask: u64) -> Result<Zeroizing<[u64; 2]>, Error> {
        match self {
            Self::Private { keys, base } => {
                let mut result = base;
                for key in &keys {
                    let value = Zeroizing::new(key.eval(party, opened)?);
                    for coordinate in 0..2 {
                        result[coordinate] =
                            result[coordinate].wrapping_add(value[coordinate]) & mask;
                    }
                }
                Ok(result)
            }
            Self::Public(key) => key.evaluate(opened).map_err(primitive_error),
            Self::Consumed => Err(Error::Consumed),
        }
    }
}

struct Lane {
    lookup: Lookup,
    mask: Zeroizing<u64>,
    triple: ring::TripleShare,
}
enum State {
    Ready(Vec<Lane>),
    Masked(Vec<Lane>, Zeroizing<Vec<u64>>, Zeroizing<Vec<u64>>),
    Product(
        Vec<Lane>,
        Zeroizing<Vec<[u64; 2]>>,
        Zeroizing<Vec<[u64; 2]>>,
    ),
}
/// One-use local evaluator; no cloning or key export. Any failed online method
/// takes/burns state, including wrong order, malformed input/frame and replay.
pub struct Party<'a> {
    id: u8,
    spec: Spec,
    binding: [u8; 32],
    nonce: [u8; 32],
    state: Option<State>,
    lease: Lease<'a>,
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
pub struct Output<'a> {
    values: Zeroizing<Vec<u64>>,
    _lease: Lease<'a>,
}
impl Output<'_> {
    pub fn shares(&self) -> &[u64] {
        &self.values
    }
}

/// Preflight/reserve *before* randomness or compilation. Callback runs once per
/// lane with a fresh OS-random mask; return original mask and translated padded
/// partition, in [constant, linear] order. Callback is trusted dealer code, must
/// honor the 1.25 MiB compilation/descriptor-overlap bound and erase its own
/// secret artifacts. Backend additionally charges its copied partition.
/// Public digests bind provenance; backend does not reimplement compiler parity.
pub fn issue<F>(
    spec: Spec,
    context: Context,
    budget: &Budget,
    compile: F,
) -> Result<[Party<'_>; 2], Error>
where
    F: FnMut(usize, u64) -> Result<DealerPartition, Error>,
{
    issue_inner(spec, context, budget, compile, random)
}
fn issue_inner<F, R>(
    spec: Spec,
    context: Context,
    budget: &Budget,
    compile: F,
    next_mask: R,
) -> Result<[Party<'_>; 2], Error>
where
    F: FnMut(usize, u64) -> Result<DealerPartition, Error>,
    R: FnMut() -> Result<u64, Error>,
{
    issue_with_lookup(spec, context, budget, compile, next_mask, None)
}

fn issue_public<'a>(
    spec: Spec,
    context: Context,
    budget: &'a Budget,
    table: &RingTable,
) -> Result<[Party<'a>; 2], Error> {
    issue_with_lookup(
        spec,
        context,
        budget,
        |_, _| Err(Error::Unsupported),
        random,
        Some(table),
    )
}

fn issue_with_lookup<'a, F, R>(
    spec: Spec,
    context: Context,
    budget: &'a Budget,
    mut compile: F,
    mut next_mask: R,
    public: Option<&RingTable>,
) -> Result<[Party<'a>; 2], Error>
where
    F: FnMut(usize, u64) -> Result<DealerPartition, Error>,
    R: FnMut() -> Result<u64, Error>,
{
    let r = resources_for(spec, public)?;
    context
        .first_tensor_index
        .checked_add(spec.lanes as u64)
        .ok_or(Error::Domain)?;
    let mut reservation = budget.reserve(r.combined_reservation_bytes)?;
    let mut hash = Sha256::new();
    hash.update(b"pllm.fusefss.affine-backend.v1\0");
    hash.update(spec.descriptor_digest);
    hash.update(spec.source_digest);
    hash.update([spec.bits]);
    hash.update((spec.source_intervals as u64).to_le_bytes());
    hash.update((spec.lanes as u64).to_le_bytes());
    hash.update(context.plan);
    hash.update(context.session);
    hash.update(context.operation);
    hash.update(context.first_tensor_index.to_le_bytes());
    if let Some(table) = public {
        hash.update(b"public-original-boundary.v1\0");
        for word in table
            .boundaries()
            .iter()
            .chain(table.coefficients().iter().flatten())
        {
            hash.update(word.to_le_bytes());
        }
    }
    let binding = hash.finalize().into();
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let mask = (1u64 << spec.bits) - 1;
    let mut left = allocate(spec.lanes, budget)?;
    let mut right = allocate(spec.lanes, budget)?;
    for lane in 0..spec.lanes {
        let input_mask = Zeroizing::new(next_mask()? & mask);
        let [ka, kb] = if let Some(table) = public {
            budget.allocation()?;
            let limit = table
                .resources()
                .map_err(primitive_error)?
                .peak_logical_allocation_bytes;
            let ([a, b], _) = issue_ring(table, *input_mask, limit).map_err(primitive_error)?;
            [Lookup::Public(a), Lookup::Public(b)]
        } else {
            let partition = compile(lane, *input_mask)?;
            partition.validate(spec.bits, *input_mask, r.padded_intervals)?;
            let mut ka = allocate(r.padded_intervals, budget)?;
            let mut kb = allocate(r.padded_intervals, budget)?;
            // Include alpha_0=0, even though its payload is zero. No sampled-value
            // pruning, deduplication, or per-mask shape changes.
            for i in 0..r.padded_intervals {
                let delta = Zeroizing::new(if i == 0 {
                    [0; 2]
                } else {
                    std::array::from_fn(|j| {
                        partition.payloads[i - 1][j].wrapping_sub(partition.payloads[i][j]) & mask
                    })
                });
                let [a, b] = dcf::Key::issue(spec.bits, partition.boundaries[i], *delta, budget)?;
                ka.push(a);
                kb.push(b);
            }
            let ba = Zeroizing::new([random()? & mask, random()? & mask]);
            let bb = Zeroizing::new(std::array::from_fn(|j| {
                partition.payloads[r.padded_intervals - 1][j].wrapping_sub(ba[j]) & mask
            }));
            [
                Lookup::Private { keys: ka, base: ba },
                Lookup::Private { keys: kb, base: bb },
            ]
        };
        let ma = Zeroizing::new(random()? & mask);
        let [ta, tb] = ring::issue(spec.bits).map_err(primitive_error)?;
        left.push(Lane {
            lookup: ka,
            mask: Zeroizing::new(*ma),
            triple: ta,
        });
        right.push(Lane {
            lookup: kb,
            mask: Zeroizing::new(input_mask.wrapping_sub(*ma) & mask),
            triple: tb,
        });
    }
    Ok([
        Party {
            id: 0,
            spec,
            binding,
            nonce,
            state: Some(State::Ready(left)),
            lease: reservation.split(r.party_reservation_bytes)?,
        },
        Party {
            id: 1,
            spec,
            binding,
            nonce,
            state: Some(State::Ready(right)),
            lease: reservation.split(r.party_reservation_bytes)?,
        },
    ])
}

impl<'a> Party<'a> {
    pub fn cancel(&mut self) {
        self.state = None;
    }
    pub fn is_consumed(&self) -> bool {
        self.state.is_none()
    }
    fn mask(&self) -> u64 {
        (1u64 << self.spec.bits) - 1
    }
    fn frame(&mut self, phase: u8, values: &[u64]) -> Result<Frame<'a>, Error> {
        let length = HEADER_BYTES + 8 * values.len();
        let mut bytes = Zeroizing::new(allocate(length, self.lease.budget)?);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.binding);
        bytes.extend_from_slice(&self.nonce);
        bytes.extend_from_slice(&[self.spec.bits, phase, self.id, 0]);
        bytes.extend_from_slice(&(self.spec.lanes as u32).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(Frame {
            bytes,
            _lease: self.lease.split(length)?,
        })
    }
    fn check(&self, peer: &[u8], phase: u8, words: usize) -> Result<(), Error> {
        if peer.len() != HEADER_BYTES + 8 * words
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != self.nonce
            || peer[72..76] != [self.spec.bits, phase, 1 - self.id, 0]
            || peer[76..80] != (self.spec.lanes as u32).to_le_bytes()
            || peer[HEADER_BYTES..]
                .chunks_exact(8)
                .any(|b| u64::from_le_bytes(b.try_into().unwrap()) > self.mask())
        {
            return Err(Error::Context);
        }
        Ok(())
    }
    pub fn start(&mut self, input_shares: &[u64]) -> Result<Frame<'a>, Error> {
        let State::Ready(keys) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        if input_shares.len() != self.spec.lanes || input_shares.iter().any(|x| *x > self.mask()) {
            return Err(Error::Domain);
        }
        let mut x = Zeroizing::new(allocate(self.spec.lanes, self.lease.budget)?);
        let mut own = Zeroizing::new(allocate(self.spec.lanes, self.lease.budget)?);
        x.extend_from_slice(input_shares);
        for (k, v) in keys.iter().zip(input_shares) {
            own.push(v.wrapping_add(*k.mask) & self.mask());
        }
        let frame = self.frame(0, &own)?;
        self.state = Some(State::Masked(keys, x, own));
        Ok(frame)
    }
    pub fn advance(&mut self, peer_masked: &[u8]) -> Result<Frame<'a>, Error> {
        let State::Masked(mut keys, x, own) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        self.check(peer_masked, 0, self.spec.lanes)?;
        let mut coefficients = Zeroizing::new(allocate(self.spec.lanes, self.lease.budget)?);
        let mut de = Zeroizing::new(allocate(self.spec.lanes, self.lease.budget)?);
        let mut words = Zeroizing::new(allocate(2 * self.spec.lanes, self.lease.budget)?);
        for (i, k) in keys.iter_mut().enumerate() {
            let u = Zeroizing::new(own[i].wrapping_add(read_word(peer_masked, i)) & self.mask());
            let lookup = std::mem::replace(&mut k.lookup, Lookup::Consumed);
            let c = lookup.evaluate(self.id, *u, self.mask())?;
            let pair = k.triple.open(c[1], x[i]).map_err(primitive_error)?;
            coefficients.push(*c);
            de.push(*pair);
            words.extend_from_slice(&*pair);
        }
        let frame = self.frame(1, &words)?;
        self.state = Some(State::Product(keys, coefficients, de));
        Ok(frame)
    }
    pub fn finish(&mut self, peer_product: &[u8]) -> Result<Output<'a>, Error> {
        let State::Product(keys, coefficients, own) = self.state.take().ok_or(Error::Consumed)?
        else {
            return Err(Error::Context);
        };
        self.check(peer_product, 1, 2 * self.spec.lanes)?;
        let mut values = Zeroizing::new(allocate(self.spec.lanes, self.lease.budget)?);
        for (i, k) in keys.into_iter().enumerate() {
            let peer = Zeroizing::new([
                read_word(peer_product, 2 * i),
                read_word(peer_product, 2 * i + 1),
            ]);
            let product = Zeroizing::new(
                k.triple
                    .finish(self.id, &own[i], &peer)
                    .map_err(primitive_error)?,
            );
            values.push(product.wrapping_add(coefficients[i][0]) & self.mask());
        }
        Ok(Output {
            values,
            _lease: self.lease.split(8 * self.spec.lanes)?,
        })
    }
}
fn read_word(frame: &[u8], index: usize) -> u64 {
    u64::from_le_bytes(
        frame[HEADER_BYTES + 8 * index..HEADER_BYTES + 8 * (index + 1)]
            .try_into()
            .unwrap(),
    )
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod conformance;
