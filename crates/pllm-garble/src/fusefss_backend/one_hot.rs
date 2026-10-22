//! One-use D9 token equality via the raw arithmetic DCF cyclic derivative.
//!
//! A50 input membership in [0,V) is a compiler-owned promise, NOT checked here.
//! Only t=x+r mod 2^ceil(log2 V) opens. Outputs stay additive A50, including
//! non-bit local shares. One fresh key may evaluate its complete public domain
//! once; it must never serve another token, retry, lane, or arbitrary query.
//!
//! Privacy assumes existing function-private DCF/PRG, independent uniform mask
//! splitting, authenticated transport, and noncolluding workers/dealer. A dealer
//! observing t can recover x. No malicious-security or side-channel certification.
use super::{
    allocate, primitive_error, wire, Budget, Error, Frame, Lease, Output, HEADER_BYTES,
    MAX_ALLOCATION_BYTES,
};
use crate::{
    compact_dcf::RingComparison,
    shared_rescale_reference::{Context, Error as PrimitiveError},
};
use sha2::{Digest, Sha256};
use std::mem::size_of;
use zeroize::Zeroizing;

pub const MAX_VOCABULARY: usize = 1 << 18;
const A50_MASK: u64 = (1u64 << 50) - 1;
const MAGIC: &[u8; 8] = b"PLLMFO01";
// Existing wire envelope, distinct primitive kind. No new namespace/parser.
const WIRE_KIND: u8 = 5;
// Covers fixed DFS stack (19 nodes), crypto scratch and owner/alignment staging.
const FIXED_BYTES: usize = 8192;
// First/previous leaf retained by the derivative fold, plus the visitor's current
// zeroizing scalar leaf. This is additional to the unchanged DFS/crypto bound.
const FOLD_BYTES: usize = size_of::<[u64; 3]>();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub vocabulary: usize,
    /// Authenticated public model/embedding/source commitment (authentication is external).
    pub source_digest: [u8; 32],
    /// Compiler's source/domain/row-selection binding; not a membership proof.
    pub profile_digest: [u8; 32],
}
impl Spec {
    fn bits(self) -> Result<u8, Error> {
        if self.vocabulary == 0
            || self.vocabulary > MAX_VOCABULARY
            || self.source_digest == [0; 32]
            || self.profile_digest == [0; 32]
        {
            return Err(Error::Domain);
        }
        Ok((usize::BITS - (self.vocabulary - 1).leading_zeros()) as u8)
    }
    pub fn digest(self) -> Result<[u8; 32], Error> {
        let bits = self.bits()?;
        let mut h = Sha256::new();
        h.update(b"pllm.fusefss.d9-one-hot.raw-dcf.cyclic-derivative.a50.v1\0");
        h.update(b"uniform-plus-mask;independent-split;full-natural-dfs19;role0-public-z0;one-lane;compiler-promised-token-domain\0");
        h.update(self.source_digest);
        h.update(self.profile_digest);
        h.update((self.vocabulary as u64).to_le_bytes());
        h.update([50, bits, 50]);
        h.update((1u64 << bits).to_le_bytes());
        Ok(h.finalize().into())
    }
    fn binding(self, context: Context) -> Result<[u8; 32], Error> {
        context
            .first_tensor_index
            .checked_add(1)
            .ok_or(Error::Domain)?;
        Ok(Sha256::new()
            .chain_update(self.digest()?)
            .chain_update(context.plan)
            .chain_update(context.session)
            .chain_update(context.operation)
            .chain_update(context.first_tensor_index.to_le_bytes())
            .finalize()
            .into())
    }
}

/// Logical fixed public work/storage, not timing, RSS, or whole-model admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub vocabulary: usize,
    pub input_bits: u8,
    pub domain_entries: usize,
    pub dcf_keys_per_party: usize,
    /// Complete scalar leaf evaluations, NOT N independent b-level point paths.
    pub dcf_evaluations_per_party: usize,
    pub tree_expansions_per_party: usize,
    pub prg_blocks_per_party: usize,
    pub derivative_rows_per_party: usize,
    pub triples_per_party: usize,
    pub private_material_bytes_per_party: usize,
    pub private_material_bytes_total: usize,
    pub material_packet_bytes_per_party: usize,
    pub public_table_bytes_per_party: usize,
    /// Fixed first/previous/current leaf storage; no full-domain scalar Vec.
    pub scalar_scratch_bytes_per_party: usize,
    pub output_bytes_per_party: usize,
    pub frames_per_party: usize,
    pub frame_bytes_per_party: usize,
    pub frame_bytes_total: usize,
    pub party_reservation_bytes: usize,
    /// Includes both role owners, issuance crypto scratch, and both packet exports.
    pub combined_reservation_bytes: usize,
}
pub fn resources(spec: Spec) -> Result<Resources, Error> {
    let b = spec.bits()?;
    let n = 1usize << b;
    let cost = if b == 0 {
        None
    } else {
        Some(RingComparison::resources(b, 50).map_err(primitive_error)?)
    };
    let private = cost.map_or(0, |r| r.encoded_key_bytes + 8);
    let packet = wire::ENVELOPE_BYTES + private;
    let fold = if b == 0 { 0 } else { FOLD_BYTES };
    let output = 8 * spec.vocabulary;
    let frame = HEADER_BYTES + 8;
    let party = size_of::<Party<'_>>()
        + size_of::<State>()
        + FIXED_BYTES
        + fold
        + output
        + frame
        + cost.map_or(0, |r| r.key_capacity_bytes);
    let combined = 2 * party + cost.map_or(0, |r| r.issuance_scratch_bytes) + 2 * packet;
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        vocabulary: spec.vocabulary,
        input_bits: b,
        domain_entries: n,
        dcf_keys_per_party: usize::from(b != 0),
        dcf_evaluations_per_party: if b == 0 { 0 } else { n },
        tree_expansions_per_party: n - 1,
        prg_blocks_per_party: 5 * (n - 1),
        derivative_rows_per_party: spec.vocabulary,
        triples_per_party: 0,
        private_material_bytes_per_party: private,
        private_material_bytes_total: 2 * private,
        material_packet_bytes_per_party: packet,
        public_table_bytes_per_party: 0,
        scalar_scratch_bytes_per_party: fold,
        output_bytes_per_party: output,
        frames_per_party: 1,
        frame_bytes_per_party: frame,
        frame_bytes_total: 2 * frame,
        party_reservation_bytes: party,
        combined_reservation_bytes: combined,
    })
}

struct Secret {
    key: Option<RingComparison>,
    mask: Zeroizing<u64>,
}
struct Work {
    output: Zeroizing<Vec<u64>>,
    frame: Zeroizing<Vec<u8>>,
}
impl Work {
    fn new(r: Resources, budget: &Budget) -> Result<Self, Error> {
        let mut output = Zeroizing::new(allocate(r.vocabulary, budget)?);
        output.resize(r.vocabulary, 0);
        let frame = Zeroizing::new(allocate(r.frame_bytes_per_party, budget)?);
        Ok(Self { output, frame })
    }
}
enum State {
    Ready(Secret, Work),
    Opened(Secret, Work, Zeroizing<u64>),
}
/// Role-only, non-clonable, consuming owner. Drop/cancel erases keys and scratch.
pub struct Party<'a> {
    id: u8,
    binding: [u8; 32],
    nonce: [u8; 32],
    resources: Resources,
    state: Option<State>,
    lease: Lease<'a>,
}

pub fn issue(spec: Spec, context: Context, budget: &Budget) -> Result<[Party<'_>; 2], Error> {
    issue_with(spec, context, budget, |bytes| {
        getrandom::fill(bytes).map_err(|_| PrimitiveError::Randomness)
    })
}
fn issue_with(
    spec: Spec,
    context: Context,
    budget: &Budget,
    mut random: impl FnMut(&mut [u8]) -> Result<(), PrimitiveError>,
) -> Result<[Party<'_>; 2], Error> {
    let r = resources(spec)?;
    let binding = spec.binding(context)?;
    let mut reservation = budget.reserve(r.combined_reservation_bytes)?;
    // Every online array/frame capacity and both key correction vectors are
    // fallibly admitted before threshold/split/root/nonce entropy is requested.
    let work_a = Work::new(r, budget)?;
    let work_b = Work::new(r, budget)?;
    let (keys, shares) = if r.input_bits == 0 {
        ([None, None], Zeroizing::new([0, 0]))
    } else {
        budget.allocation()?;
        budget.allocation()?;
        let cost = RingComparison::resources(r.input_bits, 50).map_err(primitive_error)?;
        let (keys, shares) = RingComparison::issue_uniform_split(
            r.input_bits,
            50,
            cost.combined_allocation_bytes,
            &mut random,
        )
        .map_err(primitive_error)?;
        (keys.map(Some), Zeroizing::new(shares))
    };
    let mut nonce = [0u8; 32];
    random(&mut nonce).map_err(primitive_error)?;
    let [key_a, key_b] = keys;
    let make = |id, key, work, lease| Party {
        id,
        binding,
        nonce,
        resources: r,
        state: Some(State::Ready(
            Secret {
                key,
                mask: Zeroizing::new(shares[id as usize]),
            },
            work,
        )),
        lease,
    };
    Ok([
        make(
            0,
            key_a,
            work_a,
            reservation.split(r.party_reservation_bytes)?,
        ),
        make(
            1,
            key_b,
            work_b,
            reservation.split(r.party_reservation_bytes)?,
        ),
    ])
}

impl<'a> Party<'a> {
    pub fn resources(&self) -> Resources {
        self.resources
    }
    pub fn cancel(&mut self) {
        self.state = None;
    }
    pub fn is_consumed(&self) -> bool {
        self.state.is_none()
    }

    /// One canonical A50 share. Only the input share is locally narrowed to b.
    /// V=1 is an explicit public constant: its frame contains public zero only.
    pub fn start(&mut self, share: u64) -> Result<Frame<'a>, Error> {
        let State::Ready(secret, mut work) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        if share > A50_MASK {
            return Err(Error::Domain);
        }
        let mask = (1u64 << self.resources.input_bits) - 1;
        let opened = Zeroizing::new(share.wrapping_add(*secret.mask) & mask);
        work.frame.extend_from_slice(MAGIC);
        work.frame.extend_from_slice(&self.binding);
        work.frame.extend_from_slice(&self.nonce);
        work.frame
            .extend_from_slice(&[self.resources.input_bits, 0, self.id, 0]);
        work.frame.extend_from_slice(&1u32.to_le_bytes());
        work.frame.extend_from_slice(&opened.to_le_bytes());
        let bytes = std::mem::take(&mut work.frame);
        let frame = Frame {
            bytes,
            _lease: self.lease.split(self.resources.frame_bytes_per_party)?,
        };
        self.state = Some(State::Opened(secret, work, opened));
        Ok(frame)
    }

    pub fn finish(&mut self, peer: &[u8]) -> Result<Output<'a>, Error> {
        let State::Opened(secret, mut work, own) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        let mask = (1u64 << self.resources.input_bits) - 1;
        if peer.len() != self.resources.frame_bytes_per_party
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != self.nonce
            || peer[72..76] != [self.resources.input_bits, 0, 1 - self.id, 0]
            || peer[76..80] != 1u32.to_le_bytes()
        {
            return Err(Error::Context);
        }
        let other = u64::from_le_bytes(peer[HEADER_BYTES..].try_into().unwrap());
        if other > mask {
            return Err(Error::Context);
        }
        let t = own.wrapping_add(other) & mask;
        if work.output.len() != self.resources.vocabulary
            || work.output.capacity() != self.resources.vocabulary
        {
            return Err(Error::Resource);
        }
        if let Some(key) = secret.key {
            let mut endpoints = Zeroizing::new([0u64; 2]); // first, previous
            key.eval_all_visit(self.id, |z, value| {
                if z == 0 {
                    endpoints[0] = value;
                } else {
                    // The opening t and every index are public. Every DCF leaf
                    // is still evaluated, including the padded vocabulary tail.
                    let i = (t.wrapping_sub(z as u64) & mask) as usize;
                    if i < work.output.len() {
                        work.output[i] = endpoints[1].wrapping_sub(value) & A50_MASK;
                    }
                }
                endpoints[1] = value;
            })
            .map_err(primitive_error)?;
            // The cyclic z=0 derivative needs the final and first leaves. Its
            // output index is t; retain the original role-zero correction even
            // when the public opening wraps or lies outside the vocabulary.
            if (t as usize) < work.output.len() {
                work.output[t as usize] = endpoints[1]
                    .wrapping_sub(endpoints[0])
                    .wrapping_add(u64::from(self.id == 0))
                    & A50_MASK;
            }
        } else {
            work.output[0] = u64::from(self.id == 0);
        }
        Ok(Output {
            values: work.output,
            _lease: self.lease.split(self.resources.output_bytes_per_party)?,
        })
    }

    /// Consumes unused issuance ownership. Transport must authenticate envelope.
    pub fn export_material(mut self) -> Result<wire::MaterialPacket<'a>, Error> {
        let State::Ready(secret, _work) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        wire::packet(
            WIRE_KIND,
            self.id,
            self.binding,
            self.nonce,
            self.resources.private_material_bytes_per_party,
            self.lease.budget,
            |out| {
                if let Some(key) = secret.key {
                    wire::word(out, *secret.mask);
                    key.write_material(out);
                }
                Ok(())
            },
        )
    }

    /// Replay claim burns before public/packet validation or allocation. Retain
    /// ImportGuard across the authenticated session; restart protection is outer.
    pub fn import_material(
        spec: Spec,
        context: Context,
        role: u8,
        nonce: [u8; 32],
        packet: wire::MaterialPacket<'_>,
        guard: &mut wire::ImportGuard<'_>,
        budget: &'a Budget,
    ) -> Result<Self, Error> {
        guard.claim(role, nonce)?;
        let r = resources(spec)?;
        let binding = spec.binding(context)?;
        let mut input = wire::open(
            &packet,
            WIRE_KIND,
            role,
            binding,
            nonce,
            r.private_material_bytes_per_party,
        )?;
        let mut scan = input.fork();
        let key_bytes = if r.input_bits == 0 {
            0
        } else {
            let _mask = Zeroizing::new(scan.word(r.input_bits)?);
            let len = r.private_material_bytes_per_party - 8;
            RingComparison::validate_material(scan.take(len)?, r.input_bits, 50)?;
            len
        };
        scan.finish()?;
        let lease = budget.reserve(r.party_reservation_bytes)?;
        let work = Work::new(r, budget)?;
        let secret = if r.input_bits == 0 {
            Secret {
                key: None,
                mask: Zeroizing::new(0),
            }
        } else {
            let mask = Zeroizing::new(input.word(r.input_bits)?);
            let key =
                RingComparison::read_material(input.take(key_bytes)?, r.input_bits, 50, budget)?;
            Secret {
                key: Some(key),
                mask,
            }
        };
        input.finish()?;
        Ok(Self {
            id: role,
            binding,
            nonce,
            resources: r,
            state: Some(State::Ready(secret, work)),
            lease,
        })
    }
}

#[cfg(test)]
#[path = "one_hot_tests.rs"]
mod tests;
