//! Bounded D8 conversions and public conditional Phi, using the backend's
//! allocation/lifetime envelope. No evaluator reconstructs an input or predicate.
//!
//! For `u=(x+r) mod 2^ell`, `q=2^(ell-s)`, exact signed floor is
//! ```text
//! u>>s - r>>s - [u_low<r_low] + q*[u<(r+2^(ell-1)) mod 2^ell]
//!     - q*[r+2^(ell-1)<2^ell]
//! ```
//! The last constant stays dealer-shared.
//! Reduced-ring TR drops q modulo its output ring; ARS retains it. Zero
//! extension instead uses `q*[u<r]`. Comparisons are independent payload DCFs,
//! never clear predicate openings. Widths 1..=50 match existing compact DCF.
//! This original decomposition is not an evaluated-source protocol identity or
//! independently reviewed security claim. Transport authentication remains external.
//!
//! Admission counts private vector DCFs honestly (second coordinate is unused,
//! but its key/correction bytes are charged). It excludes allocator metadata and
//! RSS. `resources` includes both roles, dealer/evaluation scratch, frame and
//! output coexistence; exported backend Frame/Output owners retain split leases.
//! Supported Phi topology is deliberately bounded to one arithmetic-bit select
//! plus a public ring add, not an arbitrary compiled post-circuit. Compose B2A
//! first when selectors are XOR-shared, issuing independent material for both.
//! Frozen per-edge scale/source identities, source-range proof verification,
//! complete CPU schedules and graph integration remain caller/source gates.

use super::{
    allocate, primitive_error, random, read_word, Budget, Error, Frame, Lease, Output,
    HEADER_BYTES, MAX_ALLOCATION_BYTES, MAX_LANES,
};
use crate::{
    compact_dcf::VectorComparison as Key, shared_rescale_reference::Context,
    shared_wide_product::ring,
};
use sha2::{Digest, Sha256};
use std::mem::size_of;
use zeroize::Zeroizing;

const MAGIC: &[u8; 8] = b"PLLMFC02";

fn mask(bits: u8) -> u64 {
    u64::MAX >> (64 - bits)
}
fn width(bits: u8) -> Result<(), Error> {
    if bits == 0 || bits > 64 {
        Err(Error::Domain)
    } else if bits > 50 {
        Err(Error::Unsupported)
    } else {
        Ok(())
    }
}

/// Public source-lowering contract, NOT a membership test or a proof verifier.
/// A narrower envelope must be proved by the caller without revealing inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedRange {
    ring_bits: u8,
    effective_bits: u8,
    proof_digest: [u8; 32],
}
impl SignedRange {
    /// Interpret every ring element in two's complement; no narrower claim.
    pub fn full_ring(bits: u8) -> Result<Self, Error> {
        width(bits)?;
        Ok(Self {
            ring_bits: bits,
            effective_bits: bits,
            proof_digest: [0; 32],
        })
    }
    /// Caller asserts a source proof of `-2^(effective_bits-1) <= X <
    /// 2^(effective_bits-1)`. This constructor only validates public metadata.
    /// It neither verifies the proof digest nor inspects/reveals secret values.
    pub fn source_contract(
        bits: u8,
        effective_bits: u8,
        proof_digest: [u8; 32],
    ) -> Result<Self, Error> {
        width(bits)?;
        if effective_bits == 0 || effective_bits > bits || proof_digest == [0; 32] {
            return Err(Error::Domain);
        }
        Ok(Self {
            ring_bits: bits,
            effective_bits,
            proof_digest,
        })
    }
}

/// Exact full-ring operations. PhiSelect computes `no + bit*(yes-no) + add`;
/// its selector must already be an arithmetic sharing of a bit. That semantic
/// bit contract is caller-owned, not checked by opening the selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Tr {
        shift: u8,
        range: SignedRange,
    },
    Ars {
        shift: u8,
        range: SignedRange,
    },
    ZeroExtend,
    SignExtend {
        range: SignedRange,
    },
    BooleanToArithmetic,
    PhiSelect {
        add: u64,
        bit_contract: [u8; 32],
    },
    /// Ordinary ring product, without a selector-bit or signed no-wrap claim.
    Multiply,
}

#[derive(Clone, Copy)]
pub struct Spec {
    pub input_bits: u8,
    pub output_bits: u8,
    pub lanes: usize,
    pub operation: Operation,
    pub source_digest: [u8; 32],
    pub profile_digest: [u8; 32],
}
impl Spec {
    fn validate(self) -> Result<(), Error> {
        width(self.input_bits)?;
        width(self.output_bits)?;
        if !(1..=MAX_LANES).contains(&self.lanes)
            || self.source_digest == [0; 32]
            || self.profile_digest == [0; 32]
        {
            return Err(Error::Domain);
        }
        match self.operation {
            Operation::Tr { shift, range } | Operation::Ars { shift, range } => {
                if shift >= self.input_bits || range.ring_bits != self.input_bits {
                    return Err(Error::Domain);
                }
                let expected = if matches!(self.operation, Operation::Tr { .. }) {
                    self.input_bits - shift
                } else {
                    self.input_bits
                };
                if self.output_bits != expected {
                    return Err(Error::Domain);
                }
            }
            Operation::SignExtend { range } => {
                if self.output_bits < self.input_bits || range.ring_bits != self.input_bits {
                    return Err(Error::Domain);
                }
            }
            Operation::ZeroExtend => {
                if self.output_bits < self.input_bits {
                    return Err(Error::Domain);
                }
            }
            Operation::BooleanToArithmetic => {
                if self.input_bits != 1 {
                    return Err(Error::Domain);
                }
            }
            Operation::PhiSelect { add, bit_contract } => {
                if self.input_bits != self.output_bits
                    || add > mask(self.output_bits)
                    || bit_contract == [0; 32]
                {
                    return Err(Error::Domain);
                }
            }
            Operation::Multiply => {
                if self.input_bits != self.output_bits {
                    return Err(Error::Domain);
                }
            }
        }
        Ok(())
    }
    fn shift(self) -> u8 {
        match self.operation {
            Operation::Tr { shift, .. } | Operation::Ars { shift, .. } => shift,
            _ => 0,
        }
    }
    fn product(self) -> bool {
        matches!(
            self.operation,
            Operation::BooleanToArithmetic | Operation::PhiSelect { .. } | Operation::Multiply
        )
    }
    fn weight(self) -> u64 {
        (1u64 << (self.input_bits - self.shift())) & mask(self.output_bits)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Resources {
    pub dcf_keys_per_party: usize,
    pub dcf_evaluations_per_party: usize,
    pub dcf_encoded_bytes_per_party: usize,
    /// Complete issued private payload, not just DCF keys: DCF encoding plus
    /// input-mask/base shares (16 bytes/lane), or triple shares (24 bytes/lane).
    /// Excludes public bindings/nonces, online scratch, owner padding and dealer
    /// temporaries. Logical layout count only; there is no private key codec.
    pub private_material_bytes_per_party: usize,
    /// Both online roles' private payload. Not peak allocation or RSS.
    pub private_material_bytes_total: usize,
    pub triples: usize,
    pub frame_bytes_per_party: usize,
    pub output_bytes_per_party: usize,
    pub party_reservation_bytes: usize,
    pub combined_reservation_bytes: usize,
}
pub fn resources(spec: Spec) -> Result<Resources, Error> {
    spec.validate()?;
    let mut keys = 0;
    let mut encoded = 0;
    let mut capacity = 0;
    let mut scratch = 4096;
    if !spec.product() {
        for bits in [
            if spec.weight() != 0 {
                spec.input_bits
            } else {
                0
            },
            spec.shift(),
        ] {
            if bits != 0 {
                let cost = Key::resources(bits, spec.output_bits).map_err(primitive_error)?;
                keys += 1;
                encoded += cost.encoded_key_bytes;
                capacity += cost.key_capacity_bytes;
                scratch += cost.issuance_scratch_bytes;
            }
        }
    }
    let words = if spec.product() { 2 } else { 1 };
    let frame = HEADER_BYTES + 8 * words * spec.lanes;
    // Vec owners/enum owners, both live/exported phases, output, product operands,
    // own openings, per-key evaluation scratch and cryptographic stack scratch.
    let party = size_of::<Party<'_>>()
        + spec.lanes * (size_of::<Lane>() + capacity + 256)
        + frame
        + 8 * spec.lanes
        + 4096;
    let combined = 2 * party + scratch + 2 * size_of::<Vec<Lane>>();
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    let private = spec.lanes
        * (encoded
            + if spec.product() {
                ring::TRIPLE_BYTES
            } else {
                16
            });
    Ok(Resources {
        dcf_keys_per_party: keys * spec.lanes,
        dcf_evaluations_per_party: keys * spec.lanes,
        dcf_encoded_bytes_per_party: encoded * spec.lanes,
        private_material_bytes_per_party: private,
        private_material_bytes_total: 2 * private,
        triples: if spec.product() { spec.lanes } else { 0 },
        frame_bytes_per_party: frame,
        output_bytes_per_party: 8 * spec.lanes,
        party_reservation_bytes: party,
        combined_reservation_bytes: combined,
    })
}

enum Lane {
    Convert {
        full: Option<Key>,
        low: Option<Key>,
        input_mask: Zeroizing<u64>,
        base: Zeroizing<u64>,
    },
    Product(ring::TripleShare),
}
enum State {
    Ready(Vec<Lane>),
    Waiting(Vec<Lane>, Zeroizing<Vec<u64>>, Zeroizing<Vec<u64>>),
}
/// One-use owner. Failed methods burn state, including malformed/replayed frames.
pub struct Party<'a> {
    id: u8,
    spec: Spec,
    binding: [u8; 32],
    nonce: [u8; 32],
    state: Option<State>,
    lease: Lease<'a>,
}

fn binding(spec: Spec, context: Context) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"pllm.fusefss.conversions.d8.v1\0");
    hash.update(spec.source_digest);
    hash.update(spec.profile_digest);
    hash.update([spec.input_bits, spec.output_bits, spec.shift()]);
    hash.update((spec.lanes as u64).to_le_bytes());
    let (tag, range) = match spec.operation {
        Operation::Tr { range, .. } => (0, Some(range)),
        Operation::Ars { range, .. } => (1, Some(range)),
        Operation::ZeroExtend => (2, None),
        Operation::SignExtend { range } => (3, Some(range)),
        Operation::BooleanToArithmetic => (4, None),
        Operation::PhiSelect { add, bit_contract } => {
            hash.update(add.to_le_bytes());
            hash.update(bit_contract);
            (5, None)
        }
        Operation::Multiply => (6, None),
    };
    hash.update([tag]);
    if let Some(range) = range {
        hash.update([range.ring_bits, range.effective_bits]);
        hash.update(range.proof_digest);
    }
    hash.update(context.plan);
    hash.update(context.session);
    hash.update(context.operation);
    hash.update(context.first_tensor_index.to_le_bytes());
    hash.finalize().into()
}

/// Dealer-only issuance. Aggregate admission precedes entropy and allocation.
/// Fresh independent masks/DCF roots/triples for every lane and invocation.
pub fn issue(spec: Spec, context: Context, budget: &Budget) -> Result<[Party<'_>; 2], Error> {
    issue_inner(spec, context, budget, random)
}
fn issue_inner(
    spec: Spec,
    context: Context,
    budget: &Budget,
    mut next: impl FnMut() -> Result<u64, Error>,
) -> Result<[Party<'_>; 2], Error> {
    let cost = resources(spec)?;
    context
        .first_tensor_index
        .checked_add(spec.lanes as u64)
        .ok_or(Error::Domain)?;
    let mut lease = budget.reserve(cost.combined_reservation_bytes)?;
    let binding = binding(spec, context);
    let mut nonce = [0; 32];
    for chunk in nonce.chunks_exact_mut(8) {
        chunk.copy_from_slice(&next()?.to_le_bytes());
    }
    let mut left = allocate(spec.lanes, budget)?;
    let mut right = allocate(spec.lanes, budget)?;
    for _ in 0..spec.lanes {
        if spec.product() {
            let [a, b] = ring::issue(spec.output_bits).map_err(primitive_error)?;
            left.push(Lane::Product(a));
            right.push(Lane::Product(b));
            continue;
        }
        let r = Zeroizing::new(next()? & mask(spec.input_bits));
        let share = Zeroizing::new(next()? & mask(spec.input_bits));
        let signed = !matches!(spec.operation, Operation::ZeroExtend);
        let half = 1u64 << (spec.input_bits - 1);
        let threshold = Zeroizing::new(if signed {
            r.wrapping_add(half) & mask(spec.input_bits)
        } else {
            *r
        });
        let mut base = Zeroizing::new(0u64.wrapping_sub(*r >> spec.shift()));
        if signed && *r < half {
            *base = base.wrapping_sub(spec.weight());
        }
        let base_a = Zeroizing::new(next()? & mask(spec.output_bits));
        let full = if spec.weight() != 0 {
            Some(issue_key(
                spec.input_bits,
                spec.output_bits,
                *threshold,
                spec.weight(),
                budget,
            )?)
        } else {
            None
        };
        let low = if spec.shift() != 0 {
            Some(issue_key(
                spec.shift(),
                spec.output_bits,
                *r & mask(spec.shift()),
                mask(spec.output_bits),
                budget,
            )?)
        } else {
            None
        };
        let (fa, fb) = match full {
            Some([a, b]) => (Some(a), Some(b)),
            None => (None, None),
        };
        let (la, lb) = match low {
            Some([a, b]) => (Some(a), Some(b)),
            None => (None, None),
        };
        left.push(Lane::Convert {
            full: fa,
            low: la,
            input_mask: Zeroizing::new(*share),
            base: Zeroizing::new(*base_a),
        });
        right.push(Lane::Convert {
            full: fb,
            low: lb,
            input_mask: Zeroizing::new(r.wrapping_sub(*share) & mask(spec.input_bits)),
            base: Zeroizing::new(base.wrapping_sub(*base_a) & mask(spec.output_bits)),
        });
    }
    Ok([
        Party {
            id: 0,
            spec,
            binding,
            nonce,
            state: Some(State::Ready(left)),
            lease: lease.split(cost.party_reservation_bytes)?,
        },
        Party {
            id: 1,
            spec,
            binding,
            nonce,
            state: Some(State::Ready(right)),
            lease: lease.split(cost.party_reservation_bytes)?,
        },
    ])
}
fn issue_key(
    bits: u8,
    output: u8,
    threshold: u64,
    payload: u64,
    budget: &Budget,
) -> Result<[Key; 2], Error> {
    let cost = Key::resources(bits, output).map_err(primitive_error)?;
    let mut allocation_failed = false;
    let result = Key::issue_with_reservation(
        bits,
        output,
        threshold,
        [payload, 0],
        cost.combined_allocation_bytes,
        |_| {
            budget.allocation().map_err(|_| {
                allocation_failed = true;
                crate::shared_rescale_reference::Error::Resource
            })
        },
    );
    if allocation_failed {
        Err(Error::Allocation)
    } else {
        result.map_err(primitive_error)
    }
}

impl<'a> Party<'a> {
    pub fn cancel(&mut self) {
        self.state = None;
    }
    pub fn is_consumed(&self) -> bool {
        self.state.is_none()
    }
    fn frame(&mut self, words: &[u64]) -> Result<Frame<'a>, Error> {
        let length = HEADER_BYTES + words.len() * 8;
        let mut bytes = Zeroizing::new(allocate(length, self.lease.budget)?);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.binding);
        bytes.extend_from_slice(&self.nonce);
        bytes.extend_from_slice(&[self.spec.output_bits, 0, self.id, 0]);
        bytes.extend_from_slice(&(self.spec.lanes as u32).to_le_bytes());
        for value in words {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(Frame {
            bytes,
            _lease: self.lease.split(length)?,
        })
    }
    fn check(&self, peer: &[u8], words: usize, bits: u8) -> Result<(), Error> {
        if peer.len() != HEADER_BYTES + words * 8
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != self.nonce
            || peer[72..76] != [self.spec.output_bits, 0, 1 - self.id, 0]
            || peer[76..80] != (self.spec.lanes as u32).to_le_bytes()
            || peer[HEADER_BYTES..]
                .chunks_exact(8)
                .any(|v| u64::from_le_bytes(v.try_into().unwrap()) > mask(bits))
        {
            return Err(Error::Context);
        }
        Ok(())
    }
    /// TR/ARS/extensions take one arithmetic share per lane. B2A takes one
    /// local Boolean XOR share (0/1) per lane; no XOR bit is ever opened.
    pub fn start(&mut self, input: &[u64]) -> Result<Frame<'a>, Error> {
        let State::Ready(keys) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        if matches!(
            self.spec.operation,
            Operation::PhiSelect { .. } | Operation::Multiply
        ) {
            return Err(Error::Context);
        }
        if input.len() != self.spec.lanes || input.iter().any(|v| *v > mask(self.spec.input_bits)) {
            return Err(Error::Domain);
        }
        let mut own = Zeroizing::new(allocate(
            self.spec.lanes * if self.spec.product() { 2 } else { 1 },
            self.lease.budget,
        )?);
        let mut additive = Zeroizing::new(allocate(self.spec.lanes, self.lease.budget)?);
        for (key, value) in keys.iter().zip(input) {
            match key {
                Lane::Convert { input_mask, .. } => {
                    own.push(value.wrapping_add(**input_mask) & mask(self.spec.input_bits))
                }
                Lane::Product(t) => {
                    let pair = t
                        .open(
                            if self.id == 0 { *value } else { 0 },
                            if self.id == 1 { *value } else { 0 },
                        )
                        .map_err(primitive_error)?;
                    own.extend_from_slice(&*pair);
                }
            }
            additive.push(*value);
        }
        let frame = self.frame(&own)?;
        self.state = Some(State::Waiting(keys, own, additive));
        Ok(frame)
    }
    /// Arithmetic-bit conditional selection. `yes`, `no`, `bit` use output-ring
    /// additive shares. Each invocation consumes its own independent triple.
    pub fn start_phi(&mut self, bit: &[u64], yes: &[u64], no: &[u64]) -> Result<Frame<'a>, Error> {
        let State::Ready(keys) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        if !matches!(self.spec.operation, Operation::PhiSelect { .. }) {
            return Err(Error::Context);
        }
        if [bit, yes, no].iter().any(|v| {
            v.len() != self.spec.lanes || v.iter().any(|x| *x > mask(self.spec.output_bits))
        }) {
            return Err(Error::Domain);
        }
        let mut own = Zeroizing::new(allocate(2 * self.spec.lanes, self.lease.budget)?);
        let mut additive = Zeroizing::new(allocate(self.spec.lanes, self.lease.budget)?);
        for (i, key) in keys.iter().enumerate() {
            let Lane::Product(t) = key else {
                return Err(Error::Context);
            };
            let difference =
                Zeroizing::new(yes[i].wrapping_sub(no[i]) & mask(self.spec.output_bits));
            let pair = t.open(bit[i], *difference).map_err(primitive_error)?;
            own.extend_from_slice(&*pair);
            additive.push(no[i]);
        }
        let frame = self.frame(&own)?;
        self.state = Some(State::Waiting(keys, own, additive));
        Ok(frame)
    }
    /// Exact product modulo `2^output_bits`, on two arithmetic-share vectors.
    /// No signed-range or bit-membership assertion is made. Every operand pair
    /// uses its own fresh triple; an exact signed rescale is a separate owner.
    pub fn start_product(&mut self, left: &[u64], right: &[u64]) -> Result<Frame<'a>, Error> {
        let State::Ready(keys) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        if !matches!(self.spec.operation, Operation::Multiply) {
            return Err(Error::Context);
        }
        if [left, right].iter().any(|values| {
            values.len() != self.spec.lanes
                || values
                    .iter()
                    .any(|value| *value > mask(self.spec.output_bits))
        }) {
            return Err(Error::Domain);
        }
        let mut own = Zeroizing::new(allocate(2 * self.spec.lanes, self.lease.budget)?);
        for (i, key) in keys.iter().enumerate() {
            let Lane::Product(triple) = key else {
                return Err(Error::Context);
            };
            let pair = triple.open(left[i], right[i]).map_err(primitive_error)?;
            own.extend_from_slice(&*pair);
        }
        let frame = self.frame(&own)?;
        self.state = Some(State::Waiting(keys, own, Zeroizing::new(Vec::new())));
        Ok(frame)
    }
    pub fn finish(&mut self, peer: &[u8]) -> Result<Output<'a>, Error> {
        let State::Waiting(keys, own, additive) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        let words = self.spec.lanes * if self.spec.product() { 2 } else { 1 };
        self.check(
            peer,
            words,
            if self.spec.product() {
                self.spec.output_bits
            } else {
                self.spec.input_bits
            },
        )?;
        let mut values = Zeroizing::new(allocate(self.spec.lanes, self.lease.budget)?);
        for (i, key) in keys.into_iter().enumerate() {
            let value = Zeroizing::new(match key {
                Lane::Convert {
                    full, low, base, ..
                } => {
                    let u = Zeroizing::new(
                        own[i].wrapping_add(read_word(peer, i)) & mask(self.spec.input_bits),
                    );
                    let mut result = Zeroizing::new(*base);
                    if self.id == 0 {
                        *result = result.wrapping_add(*u >> self.spec.shift());
                    }
                    if let Some(key) = full {
                        let v = Zeroizing::new(key.eval(self.id, *u).map_err(primitive_error)?);
                        *result = result.wrapping_add(v[0]);
                    }
                    if let Some(key) = low {
                        let v = Zeroizing::new(
                            key.eval(self.id, *u & mask(self.spec.shift()))
                                .map_err(primitive_error)?,
                        );
                        *result = result.wrapping_add(v[0]);
                    }
                    *result
                }
                Lane::Product(t) => {
                    let a = Zeroizing::new([own[2 * i], own[2 * i + 1]]);
                    let b = Zeroizing::new([read_word(peer, 2 * i), read_word(peer, 2 * i + 1)]);
                    let product =
                        Zeroizing::new(t.finish(self.id, &a, &b).map_err(primitive_error)?);
                    match self.spec.operation {
                        Operation::Multiply => *product,
                        Operation::BooleanToArithmetic => {
                            additive[i].wrapping_sub(product.wrapping_mul(2))
                        }
                        Operation::PhiSelect { add, .. } => additive[i]
                            .wrapping_add(*product)
                            .wrapping_add(if self.id == 0 { add } else { 0 }),
                        _ => return Err(Error::Context),
                    }
                }
            });
            values.push(*value & mask(self.spec.output_bits));
        }
        Ok(Output {
            values,
            _lease: self.lease.split(self.spec.lanes * 8)?,
        })
    }
}

#[cfg(test)]
#[path = "conversions_tests.rs"]
mod tests;
