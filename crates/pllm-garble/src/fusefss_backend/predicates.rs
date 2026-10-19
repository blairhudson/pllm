//! Bounded scalar Boolean projection of an actual Protocol-3 compilation.
//!
//! Trusted issuance borrows an already typed public Descriptor. Its construction,
//! generic parsing, input bytes and retained owner are OUTSIDE this reservation.
//! No frozen-GELU identity or arithmetic/Phi execution is asserted here. Frames
//! are context-bound local protocol messages, not authenticated transport.
use super::{allocate, primitive_error, random, Budget, Error, Frame, Lease, HEADER_BYTES};
use crate::{
    compact_dcf::RingComparison,
    fusefss::{self, Descriptor, Opcode, Shape},
    shared_rescale_reference::Context,
    shared_wide_product::ring,
};
use sha2::{Digest, Sha256};
use std::mem::size_of;
use zeroize::Zeroizing;

const MAGIC: &[u8; 8] = b"PLLMFB01";

/// Public shape only. DCF storage bounds every query at the descriptor word
/// width (low-bit queries may use less). Traffic counts are exact, not bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub dcf_keys_per_party: usize,
    pub dcf_evaluations_per_party: usize,
    pub triples_per_party: usize,
    pub frames_per_party: usize,
    pub total_peer_frame_bytes: usize,
    pub output_bytes_per_party: usize,
    pub compiler_overlap_bytes: usize,
    pub party_reservation_bytes: usize,
    pub combined_reservation_bytes: usize,
}

/// Does not allocate, parse, or compile. Admission precedes randomness.
pub fn resources(descriptor: &Descriptor) -> Result<Resources, Error> {
    let s = descriptor.shape();
    if s.boolean_outputs == 0 {
        return Err(Error::Unsupported);
    }
    let key = RingComparison::resources(descriptor.word().bits, 1).map_err(primitive_error)?;
    // Compiler handoff capacities are exact; retain the conservative 2x envelope.
    // Starts and ordinals overlap retained handoff allocations conservatively.
    let compiler = 2
        * (s.source_intervals * 16
            + (s.padded_intervals + 1) * 8
            + s.padded_intervals * 24
            + s.queries * size_of::<fusefss::PredicateQuery>()
            + s.vm_ops * size_of::<fusefss::Instruction>())
        + 8192;
    let frames = HEADER_BYTES + 8 + s.boolean_ands * (HEADER_BYTES + 16);
    let output = 8 * s.boolean_outputs;
    let party = s.queries * (size_of::<Query>() + key.key_capacity_bytes + 8)
        + s.vm_ops * (16 + 8) // instruction shares and maximum VM stack
        + s.boolean_ands * size_of::<Option<ring::TripleShare>>()
        + frames
        + output
        + 8192; // fixed owners, pending/opening scratch, crypto stack allowance
    let combined = 2 * party + compiler + key.issuance_scratch_bytes;
    if combined > super::MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        dcf_keys_per_party: s.queries,
        dcf_evaluations_per_party: s.queries,
        triples_per_party: s.boolean_ands,
        frames_per_party: 1 + s.boolean_ands,
        total_peer_frame_bytes: 2 * frames,
        output_bytes_per_party: output,
        compiler_overlap_bytes: compiler,
        party_reservation_bytes: party,
        combined_reservation_bytes: combined,
    })
}

fn compiler_error(e: fusefss::Error) -> Error {
    match e {
        fusefss::Error::Allocation => Error::Allocation,
        _ => Error::Domain,
    }
}

struct Query {
    // Both public, source-determined. Neither threshold nor mask is retained.
    bits: u8,
    shift: u64,
    key: RingComparison,
}
struct Material {
    mask: Zeroizing<u64>,
    queries: Vec<Query>,
    // [public opcode, private XOR-share for Bit / public index for Query].
    code: Zeroizing<Vec<[u64; 2]>>,
    triples: Vec<Option<ring::TripleShare>>,
}
struct Execution {
    material: Material,
    queries: Zeroizing<Vec<u64>>,
    stack: Zeroizing<Vec<u64>>,
    outputs: Zeroizing<Vec<u64>>,
    pc: usize,
    output_index: usize,
    end: usize,
    gate: usize,
}
struct Pending {
    triple: ring::TripleShare,
    opening: Zeroizing<[u64; 2]>,
    or_xor: Zeroizing<u64>,
}
enum State {
    Ready(Material),
    Masked(Material, Zeroizing<u64>),
    Gate(Execution, Pending),
}

/// One scalar issuance, input additive shares over Descriptor.word(), outputs
/// XOR shares. No Clone, key codec, clear predicate or clear-output API.
pub struct Party<'a> {
    id: u8,
    bits: u8,
    shape: Shape,
    binding: [u8; 32],
    nonce: [u8; 32],
    round: u32,
    state: Option<State>,
    lease: Lease<'a>,
}

pub struct BooleanOutput<'a> {
    values: Zeroizing<Vec<u64>>,
    _lease: Lease<'a>,
}
impl BooleanOutput<'_> {
    /// Canonical one-bit XOR shares, in descriptor output order.
    pub fn shares(&self) -> &[u64] {
        &self.values
    }
}

pub enum Step<'a> {
    Frame(Frame<'a>),
    Output(BooleanOutput<'a>),
}

/// Source identity is public caller provenance, not a claim that a synthetic
/// descriptor is a published model gate. Descriptor digest binds exact bytes.
pub fn issue<'a>(
    descriptor: &Descriptor,
    source_digest: [u8; 32],
    context: Context,
    budget: &'a Budget,
) -> Result<[Party<'a>; 2], Error> {
    issue_inner(descriptor, source_digest, context, budget, random)
}

/// Independent in-crate conformance can pin the input mask while all FSS keys,
/// mask shares, Bit shares, triples and nonce still use fresh randomness.
#[cfg(test)]
pub(crate) fn issue_with_mask<'a>(
    descriptor: &Descriptor,
    source_digest: [u8; 32],
    context: Context,
    budget: &'a Budget,
    mask: u64,
) -> Result<[Party<'a>; 2], Error> {
    if mask >= 1u64 << descriptor.word().bits {
        return Err(Error::Domain);
    }
    issue_inner(descriptor, source_digest, context, budget, || Ok(mask))
}

fn issue_inner<'a>(
    descriptor: &Descriptor,
    source_digest: [u8; 32],
    context: Context,
    budget: &'a Budget,
    mut draw_mask: impl FnMut() -> Result<u64, Error>,
) -> Result<[Party<'a>; 2], Error> {
    let cost = resources(descriptor)?;
    context
        .first_tensor_index
        .checked_add(1)
        .ok_or(Error::Domain)?;
    let mut reservation = budget.reserve(cost.combined_reservation_bytes)?;
    let bits = descriptor.word().bits;
    let s = descriptor.shape();
    let mask = Zeroizing::new(draw_mask()? & ((1u64 << bits) - 1));
    budget.allocation()?;
    let plan = descriptor.compile(*mask).map_err(compiler_error)?;
    let [a, b] = plan
        .with_dealer_handoff(|h| {
            Ok((|| -> Result<[Material; 2], Error> {
                if h.descriptor_digest != descriptor.digest()
                    || h.word != descriptor.word()
                    || h.shape != s
                    || h.predicate_queries.len() != s.queries
                    || h.boolean_program.len() != s.vm_ops
                {
                    return Err(Error::Domain);
                }
                let ma = Zeroizing::new(random()? & ((1u64 << bits) - 1));
                let mut a = Material {
                    mask: Zeroizing::new(*ma),
                    queries: allocate(s.queries, budget)?,
                    code: Zeroizing::new(allocate(s.vm_ops, budget)?),
                    triples: allocate(s.boolean_ands, budget)?,
                };
                let mut b = Material {
                    mask: Zeroizing::new(mask.wrapping_sub(*ma) & ((1u64 << bits) - 1)),
                    queries: allocate(s.queries, budget)?,
                    code: Zeroizing::new(allocate(s.vm_ops, budget)?),
                    triples: allocate(s.boolean_ands, budget)?,
                };
                for q in h.predicate_queries {
                    budget.allocation()?;
                    let limit = RingComparison::resources(q.bits(), 1).map_err(primitive_error)?;
                    let [ka, kb] = RingComparison::issue(
                        q.bits(),
                        1,
                        q.threshold(),
                        1,
                        limit.combined_allocation_bytes,
                    )
                    .map_err(primitive_error)?;
                    // The primitive checks actual capacities before its first draw.
                    if ka.allocation().key_capacity_bytes > limit.key_capacity_bytes
                        || kb.allocation().key_capacity_bytes > limit.key_capacity_bytes
                    {
                        return Err(Error::Resource);
                    }
                    a.queries.push(Query {
                        bits: q.bits(),
                        shift: q.shift(),
                        key: ka,
                    });
                    b.queries.push(Query {
                        bits: q.bits(),
                        shift: q.shift(),
                        key: kb,
                    });
                }
                for ins in h.boolean_program {
                    let op = ins.opcode();
                    let arg = Zeroizing::new(u64::from(ins.argument()));
                    let left = Zeroizing::new(if op == Opcode::Bit {
                        random()? & 1
                    } else {
                        *arg
                    });
                    let right = Zeroizing::new(if op == Opcode::Bit {
                        *left ^ *arg
                    } else {
                        *arg
                    });
                    a.code.push([op as u64, *left]);
                    b.code.push([op as u64, *right]);
                    if matches!(op, Opcode::And | Opcode::Or) {
                        let [ta, tb] = ring::issue(1).map_err(primitive_error)?;
                        a.triples.push(Some(ta));
                        b.triples.push(Some(tb));
                    }
                }
                if a.triples.len() != s.boolean_ands {
                    return Err(Error::Domain);
                }
                Ok([a, b])
            })())
        })
        .map_err(compiler_error)??;
    let binding = Sha256::new()
        .chain_update(b"pllm.fusefss.boolean-vm.v1\0")
        .chain_update(descriptor.digest())
        .chain_update(source_digest)
        .chain_update(context.plan)
        .chain_update(context.session)
        .chain_update(context.operation)
        .chain_update(context.first_tensor_index.to_le_bytes())
        .finalize()
        .into();
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let make = |id, material, lease| Party {
        id,
        bits,
        shape: s,
        binding,
        nonce,
        round: 0,
        state: Some(State::Ready(material)),
        lease,
    };
    Ok([
        make(0, a, reservation.split(cost.party_reservation_bytes)?),
        make(1, b, reservation.split(cost.party_reservation_bytes)?),
    ])
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
        bytes.extend_from_slice(&[self.bits, self.id, 0, 0]);
        bytes.extend_from_slice(&self.round.to_le_bytes());
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        Ok(Frame {
            bytes,
            _lease: self.lease.split(length)?,
        })
    }
    fn check(&self, peer: &[u8], words: usize, max: u64) -> Result<(), Error> {
        if peer.len() != HEADER_BYTES + 8 * words
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != self.nonce
            || peer[72..76] != [self.bits, 1 - self.id, 0, 0]
            || peer[76..80] != self.round.to_le_bytes()
            || peer[HEADER_BYTES..]
                .chunks_exact(8)
                .any(|v| u64::from_le_bytes(v.try_into().unwrap()) > max)
        {
            return Err(Error::Context);
        }
        Ok(())
    }
    pub fn start(&mut self, input_share: u64) -> Result<Frame<'a>, Error> {
        let State::Ready(material) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        let max = (1u64 << self.bits) - 1;
        if input_share > max {
            return Err(Error::Domain);
        }
        let own = Zeroizing::new(input_share.wrapping_add(*material.mask) & max);
        let frame = self.frame(&[*own])?;
        self.state = Some(State::Masked(material, own));
        Ok(frame)
    }
    /// Consume exactly one peer frame, returning the next bound round or final
    /// XOR shares. Any error burns all remaining key/triple/VM state.
    pub fn advance(&mut self, peer: &[u8]) -> Result<Step<'a>, Error> {
        let state = self.state.take().ok_or(Error::Consumed)?;
        let mut e = match state {
            State::Masked(material, own) => {
                self.check(peer, 1, (1u64 << self.bits) - 1)?;
                let opened = Zeroizing::new(
                    own.wrapping_add(super::read_word(peer, 0)) & ((1u64 << self.bits) - 1),
                );
                let mut queries = Zeroizing::new(allocate(self.shape.queries, self.lease.budget)?);
                for q in &material.queries {
                    queries.push(
                        q.key
                            .eval(
                                self.id,
                                opened.wrapping_add(q.shift) & ((1u64 << q.bits) - 1),
                            )
                            .map_err(primitive_error)?,
                    );
                }
                Execution {
                    material,
                    queries,
                    stack: Zeroizing::new(allocate(self.shape.vm_ops, self.lease.budget)?),
                    outputs: Zeroizing::new(allocate(
                        self.shape.boolean_outputs,
                        self.lease.budget,
                    )?),
                    pc: 0,
                    output_index: 0,
                    end: self.shape.program_lengths[0],
                    gate: 0,
                }
            }
            State::Gate(mut e, pending) => {
                self.check(peer, 2, 1)?;
                let theirs = Zeroizing::new([super::read_word(peer, 0), super::read_word(peer, 1)]);
                let v = pending
                    .triple
                    .finish(self.id, &pending.opening, &theirs)
                    .map_err(primitive_error)?
                    ^ *pending.or_xor;
                e.stack.push(v);
                e
            }
            State::Ready(_) => return Err(Error::Context),
        };
        self.round += 1;
        loop {
            if e.pc == e.end {
                if e.stack.len() != 1 {
                    return Err(Error::Domain);
                }
                e.outputs.push(e.stack.pop().ok_or(Error::Domain)?);
                e.output_index += 1;
                if e.output_index == self.shape.boolean_outputs {
                    if e.pc != e.material.code.len() || e.gate != self.shape.boolean_ands {
                        return Err(Error::Domain);
                    }
                    return Ok(Step::Output(BooleanOutput {
                        values: e.outputs,
                        _lease: self.lease.split(8 * self.shape.boolean_outputs)?,
                    }));
                }
                e.end += self.shape.program_lengths[e.output_index];
            }
            let [op, arg] = *e.material.code.get(e.pc).ok_or(Error::Domain)?;
            e.pc += 1;
            match op {
                0 => e.stack.push(arg),
                1 => e
                    .stack
                    .push(*e.queries.get(arg as usize).ok_or(Error::Domain)?),
                2 => *e.stack.last_mut().ok_or(Error::Domain)? ^= u64::from(self.id == 0),
                3..=5 => {
                    let right = Zeroizing::new(e.stack.pop().ok_or(Error::Domain)?);
                    let left = Zeroizing::new(e.stack.pop().ok_or(Error::Domain)?);
                    if op == 3 {
                        e.stack.push(*left ^ *right);
                        continue;
                    }
                    let triple = e
                        .material
                        .triples
                        .get_mut(e.gate)
                        .and_then(Option::take)
                        .ok_or(Error::Consumed)?;
                    e.gate += 1;
                    let opening = triple.open(*left, *right).map_err(primitive_error)?;
                    let frame = self.frame(&*opening)?;
                    let pending = Pending {
                        triple,
                        opening,
                        or_xor: Zeroizing::new(if op == 5 { *left ^ *right } else { 0 }),
                    };
                    self.state = Some(State::Gate(e, pending));
                    return Ok(Step::Frame(frame));
                }
                _ => return Err(Error::Domain),
            }
        }
    }
}

#[cfg(test)]
#[path = "predicates_tests.rs"]
mod tests;
