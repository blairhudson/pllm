//! Bounded public-boundary research candidate, not a correlated-key security
//! proof or a complete FuseFSS/ROSETTA implementation. Two immutable public
//! piecewise-constant linear forms share ONE universal comparison per lane.
//! The existing ring owner supplies independent full-ring base shares and the
//! exact dealer correction. No translated boundary, correction or mask survives
//! in an evaluator. This module adds consuming, context-bound scalar framing.
//!
//! Trust: independent in-process research dealer; frames are NOT authenticated.
//! Source/descriptor digests are commitments, not provenance or range proofs.
//! No secret reconstruction, input-membership proof, aggregate client acceptance,
//! raw material codec, import ledger or provider registration is supplied here.
//! Existing 513-interval ceiling remains: full 2049-interval SiLU is NOT admitted.
use super::{ring::MAX_INTERVALS, RingParty, RingResources, RingTable};
use crate::{
    fusefss_backend::primitive_error,
    shared_allocation::{Budget, Error, Lease, MAX_ALLOCATION_BYTES},
    shared_rescale_reference::Context,
    shared_wide_product::ring as product,
};
use sha2::{Digest, Sha256};
use std::mem::size_of;
use zeroize::Zeroizing;

const MAGIC: &[u8; 8] = b"PLLMJB01";
const HEADER: usize = 80;
const FRAME: usize = HEADER + 16;
const SCRATCH: usize = 4096;

pub(crate) struct JointBasis<'a> {
    table: RingTable,
    digest: [u8; 32],
    source: [u8; 32],
    _lease: Lease<'a>,
}

/// Counts live logical owners/capacities, not cumulative preprocessing bytes,
/// work admission, allocator metadata, spills or RSS. Repeated issuance has no
/// request-wide material/work ledger here; no whole-method admission is implied.
/// Caller-owned borrowed import slices remain an external charge. No tape or
/// material envelope exists in this candidate; do not quote zero as a codec cost.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JointResources {
    pub ring: RingResources,
    pub source_reservation_bytes: usize,
    pub issuance_reservation_bytes: usize,
    pub combined_reservation_bytes: usize,
    pub frame_backing_bytes_per_party: usize,
    pub output_bytes_per_party: usize,
    pub private_payload_bytes_per_party: usize,
    pub dcf_path_levels_per_party: usize,
    pub triples_per_party: usize,
}

impl<'a> JointBasis<'a> {
    /// Fixed arity two. Every row is two public constant-channel values, NOT
    /// [constant, slope] until issue/start/advance_affine/finish_affine selects
    /// that interpretation. One scalar lane; no implicit membership check.
    pub(crate) fn new(
        bits: u8,
        boundaries: &[u64],
        channels: &[[u64; 2]],
        source: [u8; 32],
        descriptor: [u8; 32],
        budget: &'a Budget,
    ) -> Result<Self, Error> {
        if source == [0; 32] || descriptor == [0; 32] {
            return Err(Error::Domain);
        }
        // Bound lengths before byte arithmetic and reserve before copying.
        if channels.len() > MAX_INTERVALS || boundaries.len() > MAX_INTERVALS + 1 {
            return Err(Error::Resource);
        }
        let payload = boundaries.len() * 8 + channels.len() * 16;
        let lease = budget.reserve(size_of::<Self>() + payload + SCRATCH)?;
        budget.allocation()?;
        let table = RingTable::new(bits, boundaries, channels, size_of::<RingTable>() + payload)
            .map_err(primitive_error)?;
        let mut hash = Sha256::new();
        hash.update(b"pllm.joint-public-boundary-basis.reference.v1\0");
        hash.update(source);
        hash.update(descriptor);
        hash.update([bits, 2]);
        hash.update((channels.len() as u64).to_le_bytes());
        for &p in boundaries {
            hash.update(p.to_le_bytes());
        }
        for row in channels {
            for v in row {
                hash.update(v.to_le_bytes());
            }
        }
        Ok(Self { table, digest: hash.finalize().into(), source, _lease: lease })
    }

    pub(crate) fn resources(&self) -> Result<JointResources, Error> {
        let ring = self.table.resources().map_err(primitive_error)?;
        // Conservatively keep ring's source charge even though our source lease
        // already charges it. Includes BOTH holders/copies, dealer workspace,
        // nonce/context, moved Result staging, two exported frames plus imported
        // peer-frame copies per holder, product/output/mask staging. No heap
        // growth occurs in the added framing/output path.
        let extra = 4 * size_of::<JointParty<'_>>()
            + 8 * size_of::<JointFrame<'_>>()
            + 4 * size_of::<JointOutput<'_>>()
            + 4 * size_of::<JointAffineOutput<'_>>()
            + 2 * size_of::<JointResources>()
            + SCRATCH;
        let issuance = ring.peak_logical_allocation_bytes
            .checked_add(extra).ok_or(Error::Resource)?;
        let combined = self._lease.bytes.checked_add(issuance).ok_or(Error::Resource)?;
        if combined > MAX_ALLOCATION_BYTES { return Err(Error::Resource); }
        Ok(JointResources {
            ring,
            source_reservation_bytes: self._lease.bytes,
            issuance_reservation_bytes: issuance,
            combined_reservation_bytes: combined,
            frame_backing_bytes_per_party: 2 * FRAME,
            output_bytes_per_party: 16,
            private_payload_bytes_per_party: ring.party_private_key_payload_bytes[0]
                + 8 + product::TRIPLE_BYTES,
            dcf_path_levels_per_party: ring.dcf_path_levels_per_party,
            triples_per_party: 1,
        })
    }

    /// All issuance includes one fresh Beaver product; callers may discard it
    /// after selecting channels. This keeps one immutable, honestly priced
    /// schedule rather than a secret-dependent optional product policy.
    pub(crate) fn issue(&self, context: Context, epoch: u64) -> Result<[JointParty<'a>; 2], Error> {
        self.issue_with(context, epoch, |bytes| {
            #[cfg(test)]
            if let Some(result) = super::joint_basis_test_hooks::fill(
                super::joint_basis_test_hooks::Kind::JointEntropy, bytes,
            ) {
                return result.map_err(primitive_error);
            }
            #[cfg(test)]
            if let Some(result) = tests::fill_override(bytes) {
                return result;
            }
            getrandom::fill(bytes).map_err(|_| Error::Randomness)
        })
    }

    fn issue_with(
        &self,
        context: Context,
        epoch: u64,
        mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<[JointParty<'a>; 2], Error> {
        context.first_tensor_index.checked_add(1).ok_or(Error::Domain)?;
        let cost = self.resources()?;
        let budget = self._lease.budget;
        // Atomic reservation BEFORE any mask, nonce, comparison or triple RNG.
        let mut lease = budget.reserve(cost.issuance_reservation_bytes)?;
        budget.allocation()?;
        let mut hash = Sha256::new();
        hash.update(b"pllm.joint-public-boundary-basis.context.v1\0");
        hash.update(self.digest);
        hash.update(self.source);
        hash.update(context.plan);
        hash.update(context.session);
        hash.update(context.operation);
        hash.update(context.first_tensor_index.to_le_bytes());
        hash.update(epoch.to_le_bytes());
        let binding = hash.finalize().into();
        let mask = (1u64 << self.table.bits()) - 1;
        let mut entropy = Zeroizing::new([0; 48]);
        fill(&mut *entropy)?;
        let r = Zeroizing::new(u64::from_le_bytes(entropy[..8].try_into().unwrap()) & mask);
        let left = Zeroizing::new(u64::from_le_bytes(entropy[8..16].try_into().unwrap()) & mask);
        let mut nonce = [0; 32];
        nonce.copy_from_slice(&entropy[16..]);
        let ([a, b], _) = super::issue_ring(&self.table, *r, cost.ring.peak_logical_allocation_bytes)
            .map_err(primitive_error)?;
        let [ta, tb] = product::issue(self.table.bits()).map_err(primitive_error)?;
        let party_bytes = cost.issuance_reservation_bytes / 2;
        let la = lease.split(party_bytes)?;
        let lb = lease.split(cost.issuance_reservation_bytes - party_bytes)?;
        Ok([(0, a, ta, *left, la), (1, b, tb, r.wrapping_sub(*left) & mask, lb)]
            .map(|(id, key, triple, input_mask, lease)| JointParty {
                id, bits: self.table.bits(), binding, nonce,
                state: Some(State::Ready(key, Zeroizing::new(input_mask), triple)),
                lease,
            }))
    }
}

enum State {
    Ready(RingParty, Zeroizing<u64>, product::TripleShare),
    Waiting(RingParty, Zeroizing<u64>, Zeroizing<u64>, product::TripleShare),
    Product(Zeroizing<u64>, Zeroizing<[u64; 2]>, product::TripleShare),
}

/// One holder only; no Clone/Debug/material export. Any invalid call burns.
pub(crate) struct JointParty<'a> {
    id: u8,
    bits: u8,
    binding: [u8; 32],
    nonce: [u8; 32],
    state: Option<State>,
    lease: Lease<'a>,
}

pub(crate) struct JointFrame<'a> {
    bytes: Zeroizing<[u8; FRAME]>,
    len: usize,
    _lease: Lease<'a>,
}
impl JointFrame<'_> {
    pub(crate) fn as_bytes(&self) -> &[u8] { &self.bytes[..self.len] }
}
pub(crate) struct JointOutput<'a> {
    values: Zeroizing<[u64; 2]>,
    _lease: Lease<'a>,
}
impl JointOutput<'_> {
    pub(crate) fn shares(&self) -> &[u64; 2] { &self.values }
}
pub(crate) struct JointAffineOutput<'a> {
    value: Zeroizing<u64>,
    _lease: Lease<'a>,
}
impl JointAffineOutput<'_> {
    pub(crate) fn share(&self) -> u64 { *self.value }
}

impl<'a> JointParty<'a> {
    pub(crate) fn cancel(&mut self) { self.state = None; }
    fn mask(&self) -> u64 { (1u64 << self.bits) - 1 }
    fn frame(&mut self, phase: u8, words: &[u64]) -> Result<JointFrame<'a>, Error> {
        let mut bytes = Zeroizing::new([0; FRAME]);
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..40].copy_from_slice(&self.binding);
        bytes[40..72].copy_from_slice(&self.nonce);
        bytes[72..76].copy_from_slice(&[self.id, phase, self.bits, 2]);
        bytes[76..80].copy_from_slice(&1u32.to_le_bytes());
        for (dst, word) in bytes[HEADER..].chunks_exact_mut(8).zip(words) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        Ok(JointFrame { bytes, len: HEADER + words.len() * 8,
            _lease: self.lease.split(size_of::<JointFrame<'_>>())? })
    }
    fn peer(&self, bytes: &[u8], phase: u8, count: usize) -> Result<Zeroizing<[u64; 2]>, Error> {
        if bytes.len() != HEADER + count * 8
            || &bytes[..8] != MAGIC || bytes[8..40] != self.binding
            || bytes[40..72] != self.nonce
            || bytes[72..76] != [1 - self.id, phase, self.bits, 2]
            || bytes[76..80] != 1u32.to_le_bytes()
        { return Err(Error::Context); }
        let mut result = Zeroizing::new([0; 2]);
        for (dst, raw) in result.iter_mut().zip(bytes[HEADER..].chunks_exact(8)) {
            *dst = u64::from_le_bytes(raw.try_into().unwrap());
            if *dst > self.mask() { return Err(Error::Domain); }
        }
        Ok(result)
    }
    pub(crate) fn start(&mut self, share: u64) -> Result<JointFrame<'a>, Error> {
        let State::Ready(key, mask, triple) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        if share > self.mask() { return Err(Error::Domain); }
        let own = Zeroizing::new(share.wrapping_add(*mask) & self.mask());
        let frame = self.frame(0, &[*own])?;
        self.state = Some(State::Waiting(key, own, Zeroizing::new(share), triple));
        Ok(frame)
    }
    /// Finish two constant channels; the unused fresh triple is burned.
    pub(crate) fn finish(&mut self, peer: &[u8]) -> Result<JointOutput<'a>, Error> {
        let State::Waiting(key, own, _, _) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        let remote = self.peer(peer, 0, 1)?;
        let values = key.evaluate(own.wrapping_add(remote[0]) & self.mask())
            .map_err(primitive_error)?;
        Ok(JointOutput { values, _lease: self.lease.split(size_of::<JointOutput<'_>>())? })
    }
    /// Fixed affine continuation: channel[0] + channel[1]*original X, modulo N.
    /// One fresh existing Beaver product, no ARS or implicit scale adjustment.
    /// For a Q0 D and Q12 X/C, this returns Q12 directly.
    pub(crate) fn advance_affine(&mut self, peer: &[u8]) -> Result<JointFrame<'a>, Error> {
        let State::Waiting(key, own, x, triple) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        let remote = self.peer(peer, 0, 1)?;
        let values = key.evaluate(own.wrapping_add(remote[0]) & self.mask())
            .map_err(primitive_error)?;
        let opening = triple.open(values[1], *x).map_err(primitive_error)?;
        let frame = self.frame(1, &*opening)?;
        self.state = Some(State::Product(Zeroizing::new(values[0]), opening, triple));
        Ok(frame)
    }
    pub(crate) fn finish_affine(&mut self, peer: &[u8]) -> Result<JointAffineOutput<'a>, Error> {
        let State::Product(base, own, triple) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        let remote = self.peer(peer, 1, 2)?;
        let value = Zeroizing::new(base.wrapping_add(
            triple.finish(self.id, &own, &remote).map_err(primitive_error)?) & self.mask());
        Ok(JointAffineOutput { value,
            _lease: self.lease.split(size_of::<JointAffineOutput<'_>>())? })
    }
}

#[cfg(test)]
#[path = "joint_basis_tests.rs"]
mod tests;

#[cfg(test)]
pub(crate) fn test_source_active() -> bool {
    tests::source_active()
}
