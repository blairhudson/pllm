//! Bounded composition of original-public-table lookups. Every chunk evaluates
//! its complete full-ring function: the selected public slice, zero elsewhere.
//! Parties sum all chunk shares. No private chunk number, index, or table value
//! is revealed. Every lane/chunk receives independent fresh masks and keys.
//!
//! This deliberately preserves the existing 513-interval primitive ceiling.
//! The larger table is a separately bounded composition, not a larger primitive
//! or a source range certificate. Inputs outside the original table return zero;
//! callers must separately establish source-specific index/domain obligations.
use super::{
    allocate as exact, primitive_error, random, Budget, Error, Frame, Lease, Output, HEADER_BYTES,
    MAX_ALLOCATION_BYTES, MAX_LANES,
};
use crate::interval_fss::RingComparison;
use crate::shared_affine_lookup_reference::{issue_ring, RingParty, RingTable};
use crate::shared_rescale_reference::Context;
use sha2::{Digest, Sha256};
use std::{mem::size_of, sync::Arc};
use zeroize::Zeroizing;

/// Bounded public composition only: 17-bit indices cover the source reciprocal
/// table at row width 1024. The 64 MiB aggregate and four-lane caps still apply.
/// This neither raises the underlying 513-interval cap nor certifies index ranges.
pub const MAX_TABLE_ENTRIES: usize = 1 << 17;
/// At most 2 KiB of original u64 values per chunk. Native boundary/coefficient
/// copies are larger and are charged separately; this is not a memory estimate.
pub const CHUNK_ENTRIES: usize = 256;
const MAGIC: &[u8; 8] = b"PLLMFL01";
const FIXED_BYTES: usize = 4096;
const TABLE_DOMAIN: &[u8] = b"pllm.fusefss.zero-extended-public-table.v1\0";

/// Immutable original public table, with its own live logical reservation.
/// Caller buffers and delivery remain separate owners. The source digest is a
/// caller-supplied commitment, not publisher authentication or a range proof.
pub struct Table<'a> {
    values: Vec<u64>,
    bits: u8,
    source_digest: [u8; 32],
    digest: [u8; 32],
    lowered: Option<Arc<Lowered<'a>>>,
    lease: Lease<'a>,
}

// Public original partitions only. This independent lease must outlive the raw
// Table/cache whenever an issued or imported party still holds these chunks.
struct Lowered<'a> {
    chunks: Vec<Arc<RingTable>>,
    _lease: Lease<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub entries: usize,
    pub chunks_per_lane: usize,
    pub dcf_keys_per_party: usize,
    pub dcf_evaluations_per_party: usize,
    pub private_key_bytes_per_party: usize,
    /// Actual public ownership copies: payload for legacy parties, Arc handles
    /// only for shared parties. Shared backing is charged once below instead.
    pub public_table_bytes_per_party: usize,
    pub peer_frame_bytes_per_party: usize,
    pub output_bytes_per_party: usize,
    /// Raw source plus any independently leased immutable lowered backing.
    pub table_reservation_bytes: usize,
    /// Additional reservation; the immutable Table retains its separate lease.
    pub issuance_reservation_bytes: usize,
    pub party_reservation_bytes: usize,
}

fn checked_add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}

fn checked_mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Resource)
}

impl<'a> Table<'a> {
    pub fn new_zero_extended(
        bits: u8,
        values: &[u64],
        source_digest: [u8; 32],
        budget: &'a Budget,
    ) -> Result<Self, Error> {
        if !(1..=50).contains(&bits)
            || values.is_empty()
            || values.len() > MAX_TABLE_ENTRIES
            || values.len() as u64 > 1u64 << bits
            || values.iter().any(|&v| v >= 1u64 << bits)
        {
            return Err(Error::Domain);
        }
        let lease = budget.reserve(checked_add(
            size_of::<Self>(),
            checked_mul(values.len(), 8)?,
        )?)?;
        let mut copy = exact(values.len(), budget)?;
        copy.extend_from_slice(values);
        let mut hash = Sha256::new();
        hash.update(TABLE_DOMAIN);
        hash.update(source_digest);
        hash.update([bits]);
        hash.update((values.len() as u64).to_le_bytes());
        hash.update((CHUNK_ENTRIES as u64).to_le_bytes());
        for value in values {
            hash.update(value.to_le_bytes());
        }
        Ok(Self {
            values: copy,
            bits,
            source_digest,
            digest: hash.finalize().into(),
            lowered: None,
            lease,
        })
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub(crate) fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }

    pub(crate) fn construction_hash_bytes(entries: usize) -> usize {
        TABLE_DOMAIN.len() + 32 + 1 + 8 + 8 + 8 * entries
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn entries(&self) -> usize {
        self.values.len()
    }

    fn intervals(&self, chunk: usize) -> usize {
        let start = chunk * CHUNK_ENTRIES;
        let end = (start + CHUNK_ENTRIES).min(self.entries());
        end - start + usize::from(start != 0) + usize::from((end as u64) < 1u64 << self.bits)
    }

    pub(crate) fn lowered_reservation_bytes(&self) -> usize {
        let chunks = self.entries().div_ceil(CHUNK_ENTRIES);
        size_of::<Lowered<'_>>()
            + 2 * size_of::<usize>()
            + chunks * size_of::<Arc<RingTable>>()
            + (0..chunks)
                .map(|c| {
                    size_of::<RingTable>() + 2 * size_of::<usize>() + self.intervals(c) * 24 + 8
                })
                .sum::<usize>()
    }

    #[cfg(test)]
    pub(crate) fn lowered_payload_bytes(&self) -> usize {
        (0..self.entries().div_ceil(CHUNK_ENTRIES))
            .map(|c| self.intervals(c) * 24 + 8)
            .sum()
    }

    /// Request-local opt-in. Default constructors and codecs keep their original
    /// behavior. All canonical chunk construction is admitted before allocation;
    /// no masks, nonces, keys or randomized bases enter this immutable owner.
    #[cfg(test)]
    pub(crate) fn with_shared_chunks(self) -> Result<Self, Error> {
        self.with_shared_chunks_counted(|_, _| {})
    }

    /// Report completed public construction even if a later allocation fails.
    /// Retained bytes are committed separately, only after all chunks succeed.
    pub(crate) fn with_shared_chunks_counted(
        mut self,
        mut constructed: impl FnMut(usize, usize),
    ) -> Result<Self, Error> {
        if self.lowered.is_some() {
            return Ok(self);
        }
        let budget = self.lease.budget;
        let lease = budget.reserve(self.lowered_reservation_bytes())?;
        let count = self.entries().div_ceil(CHUNK_ENTRIES);
        let mut chunks = exact(count, budget)?;
        for chunk in 0..count {
            let (boundaries, coefficients) = self.chunk_buffers(chunk)?;
            let table = RingTable::from_owned(
                self.bits,
                boundaries,
                coefficients,
                size_of::<RingTable>() + self.intervals(chunk) * 24 + 8,
            )
            .map_err(primitive_error)?;
            // Payload and validated RingTable header already exist even if the
            // following Arc allocation fails. Count metadata independently.
            constructed(self.intervals(chunk) * 24 + 8, size_of::<RingTable>());
            budget.allocation()?;
            chunks.push(Arc::new(table));
            constructed(0, 2 * size_of::<usize>() + size_of::<Arc<RingTable>>());
        }
        budget.allocation()?;
        self.lowered = Some(Arc::new(Lowered {
            chunks,
            _lease: lease,
        }));
        constructed(
            0,
            size_of::<Lowered<'_>>() + 2 * size_of::<usize>() + size_of::<Arc<Lowered<'_>>>(),
        );
        Ok(self)
    }

    pub fn resources(&self, lanes: usize) -> Result<Resources, Error> {
        if !(1..=MAX_LANES).contains(&lanes) {
            return Err(Error::Domain);
        }
        let chunks = self.entries().div_ceil(CHUNK_ENTRIES);
        let keys = checked_mul(chunks, lanes)?;
        let dcf = RingComparison::resources(self.bits, self.bits).map_err(primitive_error)?;
        let mut public_bytes = 0;
        let mut evaluations = 0;
        let mut largest_public = 0;
        for chunk in 0..chunks {
            let intervals = self.intervals(chunk);
            let bytes = checked_add(checked_mul(intervals, 24)?, 8)?;
            public_bytes = checked_add(public_bytes, bytes)?;
            evaluations = checked_add(evaluations, intervals)?;
            largest_public = largest_public.max(bytes);
        }
        public_bytes = if self.lowered.is_some() {
            checked_add(
                checked_mul(keys, size_of::<Arc<RingTable>>())?,
                size_of::<Arc<Lowered<'_>>>(),
            )?
        } else {
            checked_mul(public_bytes, lanes)?
        };
        let frame = checked_add(HEADER_BYTES, checked_mul(keys, 8)?)?;
        let output = checked_mul(lanes, 8)?;
        let unit_bytes = checked_mul(
            keys,
            checked_add(size_of::<Unit>(), dcf.key_capacity_bytes)?,
        )?;
        // All original public copies (or shared handles), key owners, masked
        // openings and exported frame/output can coexist. No secret scratch discount.
        let party = checked_add(
            checked_add(unit_bytes, public_bytes)?,
            checked_add(
                checked_mul(keys, 8)?,
                checked_add(frame, output + FIXED_BYTES + dcf.issuance_scratch_bytes)?,
            )?,
        )?;
        // While issuing one chunk, both construction slices and the intermediate
        // RingTable coexist. Its party copies are already included above. Fixed
        // metadata and both native comparison scratch contracts are charged too.
        let scratch = checked_add(
            checked_mul(largest_public, 2)?,
            checked_add(2 * FIXED_BYTES, checked_mul(dcf.issuance_scratch_bytes, 2)?)?,
        )?;
        let issuance = checked_add(checked_mul(party, 2)?, scratch)?;
        let table_bytes = checked_add(
            self.lease.bytes,
            self.lowered.as_ref().map_or(0, |v| v._lease.bytes),
        )?;
        if checked_add(issuance, table_bytes)? > MAX_ALLOCATION_BYTES {
            return Err(Error::Resource);
        }
        Ok(Resources {
            entries: self.entries(),
            chunks_per_lane: chunks,
            dcf_keys_per_party: keys,
            dcf_evaluations_per_party: checked_mul(evaluations, lanes)?,
            private_key_bytes_per_party: checked_mul(keys, dcf.encoded_key_bytes + 24)?,
            public_table_bytes_per_party: public_bytes,
            peer_frame_bytes_per_party: frame,
            output_bytes_per_party: output,
            table_reservation_bytes: table_bytes,
            issuance_reservation_bytes: issuance,
            party_reservation_bytes: party,
        })
    }

    fn chunk(&self, index: usize) -> Result<RingTable, Error> {
        let (boundaries, coefficients) = self.chunk_buffers(index)?;
        self.lease.budget.allocation()?;
        RingTable::new(
            self.bits,
            &boundaries,
            &coefficients,
            size_of::<RingTable>() + boundaries.len() * 8 + coefficients.len() * 16,
        )
        .map_err(primitive_error)
    }

    fn chunk_buffers(&self, index: usize) -> Result<(Vec<u64>, Vec<[u64; 2]>), Error> {
        let start = index * CHUNK_ENTRIES;
        let end = (start + CHUNK_ENTRIES).min(self.entries());
        let count = self.intervals(index);
        let mut boundaries = exact(count + 1, self.lease.budget)?;
        let mut coefficients = exact(count, self.lease.budget)?;
        boundaries.push(0);
        if start != 0 {
            coefficients.push([0, 0]);
            boundaries.push(start as u64);
        }
        for i in start..end {
            coefficients.push([self.values[i], 0]);
            boundaries.push((i + 1) as u64);
        }
        if (end as u64) < 1u64 << self.bits {
            coefficients.push([0, 0]);
            boundaries.push(1u64 << self.bits);
        }
        Ok((boundaries, coefficients))
    }

    /// Trusted dealer issuance. The Table budget also owns the issued parties.
    /// Public Table may retire immediately: each evaluator retains immutable
    /// public copies or independently leased shared chunks, plus fresh private keys.
    pub fn issue(&self, lanes: usize, context: Context) -> Result<[Party<'a>; 2], Error> {
        let r = self.resources(lanes)?;
        context
            .first_tensor_index
            .checked_add(lanes as u64)
            .ok_or(Error::Domain)?;
        let budget = self.lease.budget;
        let mut reservation = budget.reserve(r.issuance_reservation_bytes)?;
        let mut left = exact(r.dcf_keys_per_party, budget)?;
        let mut right = exact(r.dcf_keys_per_party, budget)?;
        let binding = self.binding(lanes, context);
        let mut nonce = [0; 32];
        getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
        let mask = (1u64 << self.bits) - 1;
        // Chunk-major order is public and fixed, including all-zero slices.
        for chunk in 0..r.chunks_per_lane {
            let local = if self.lowered.is_none() {
                Some(self.chunk(chunk)?)
            } else {
                None
            };
            let table = self
                .lowered
                .as_ref()
                .map(|v| v.chunks[chunk].as_ref())
                .or(local.as_ref())
                .ok_or(Error::Domain)?;
            let cost = table.resources().map_err(primitive_error)?;
            for _ in 0..lanes {
                let offset = Zeroizing::new(random()? & mask);
                budget.allocation()?;
                let [a, b] = if let Some(lowered) = &self.lowered {
                    let (pair, _) = lowered.chunks[chunk]
                        .issue_shared(*offset, cost.peak_logical_allocation_bytes)
                        .map_err(primitive_error)?;
                    pair.map(Lookup::Shared)
                } else {
                    let (pair, _) = issue_ring(table, *offset, cost.peak_logical_allocation_bytes)
                        .map_err(primitive_error)?;
                    pair.map(Lookup::Local)
                };
                let share = Zeroizing::new(random()? & mask);
                left.push(Unit {
                    lookup: a,
                    mask: Zeroizing::new(*share),
                });
                right.push(Unit {
                    lookup: b,
                    mask: Zeroizing::new(offset.wrapping_sub(*share) & mask),
                });
            }
        }
        Ok([
            Party {
                id: 0,
                bits: self.bits,
                lanes,
                binding,
                nonce,
                resources: r,
                state: Some(State::Ready(left)),
                _lowered: self.lowered.as_ref().map(Arc::clone),
                lease: reservation.split(r.party_reservation_bytes)?,
            },
            Party {
                id: 1,
                bits: self.bits,
                lanes,
                binding,
                nonce,
                resources: r,
                state: Some(State::Ready(right)),
                _lowered: self.lowered.as_ref().map(Arc::clone),
                lease: reservation.split(r.party_reservation_bytes)?,
            },
        ])
    }
}

impl Table<'_> {
    fn binding(&self, lanes: usize, context: Context) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"pllm.fusefss.chunked-public-lookup.v1\0");
        hash.update(self.digest);
        hash.update(self.source_digest);
        hash.update(context.plan);
        hash.update(context.session);
        hash.update(context.operation);
        hash.update(context.first_tensor_index.to_le_bytes());
        hash.update((lanes as u64).to_le_bytes());
        hash.finalize().into()
    }
}

enum Lookup {
    Local(RingParty),
    Shared(RingParty<Arc<RingTable>>),
}
impl Lookup {
    fn write_material(&self, out: &mut Vec<u8>) {
        match self {
            Self::Local(p) => p.write_material(out),
            Self::Shared(p) => p.write_material(out),
        }
    }
    fn evaluate(
        self,
        opened: u64,
    ) -> Result<Zeroizing<[u64; 2]>, crate::shared_rescale_reference::Error> {
        match self {
            Self::Local(p) => p.evaluate(opened),
            Self::Shared(p) => p.evaluate(opened),
        }
    }
}

struct Unit {
    lookup: Lookup,
    mask: Zeroizing<u64>,
}

enum State {
    Ready(Vec<Unit>),
    Opened(Vec<Unit>, Zeroizing<Vec<u64>>),
}

/// Consuming two-party lookup. Methods take state before any validation; errors,
/// wrong order, cancellation and replay burn every remaining chunk together.
pub struct Party<'a> {
    id: u8,
    bits: u8,
    lanes: usize,
    binding: [u8; 32],
    nonce: [u8; 32],
    resources: Resources,
    state: Option<State>,
    _lowered: Option<Arc<Lowered<'a>>>,
    lease: Lease<'a>,
}

impl<'a> Party<'a> {
    pub fn export_material(mut self) -> Result<super::wire::MaterialPacket<'a>, Error> {
        let state = self.state.take().ok_or(Error::Consumed)?;
        let State::Ready(keys) = state else {
            return Err(Error::Context);
        };
        super::wire::packet(
            3,
            self.id,
            self.binding,
            self.nonce,
            self.resources.private_key_bytes_per_party,
            self.lease.budget,
            |out| {
                for unit in &keys {
                    super::wire::word(out, *unit.mask);
                    unit.lookup.write_material(out);
                }
                Ok(())
            },
        )
    }

    pub fn import_material(
        table: &Table<'a>,
        lanes: usize,
        context: Context,
        role: u8,
        nonce: [u8; 32],
        packet: super::wire::MaterialPacket<'_>,
        guard: &mut super::wire::ImportGuard<'_>,
    ) -> Result<Self, Error> {
        guard.claim(role, nonce)?;
        let r = table.resources(lanes)?;
        context
            .first_tensor_index
            .checked_add(lanes as u64)
            .ok_or(Error::Domain)?;
        let binding = table.binding(lanes, context);
        let mut input = super::wire::open(
            &packet,
            3,
            role,
            binding,
            nonce,
            r.private_key_bytes_per_party,
        )?;
        let mut scan = input.fork();
        let key_bytes = crate::compact_dcf::RingComparison::resources(table.bits, table.bits)
            .map_err(primitive_error)?
            .encoded_key_bytes;
        for _ in 0..r.dcf_keys_per_party {
            for _ in 0..3 {
                let _ = Zeroizing::new(scan.word(table.bits)?);
            }
            crate::compact_dcf::RingComparison::validate_material(
                scan.take(key_bytes)?,
                table.bits,
                table.bits,
            )?;
        }
        scan.finish()?;
        let budget = table.lease.budget;
        // One public chunk constructor overlaps imported copies, never dealer RNG.
        let mut reservation =
            budget.reserve(r.issuance_reservation_bytes - r.party_reservation_bytes)?;
        let mut keys = exact(r.dcf_keys_per_party, budget)?;
        for chunk in 0..r.chunks_per_lane {
            let public = if table.lowered.is_none() {
                Some(table.chunk(chunk)?)
            } else {
                None
            };
            for _ in 0..lanes {
                let mask = Zeroizing::new(input.word(table.bits)?);
                let lookup = if let Some(lowered) = &table.lowered {
                    Lookup::Shared(RingParty::read_shared_material(
                        &mut input,
                        role,
                        Arc::clone(&lowered.chunks[chunk]),
                        budget,
                    )?)
                } else {
                    Lookup::Local(RingParty::read_material(
                        &mut input,
                        role,
                        public.as_ref().ok_or(Error::Domain)?,
                        budget,
                    )?)
                };
                keys.push(Unit { lookup, mask });
            }
        }
        input.finish()?;
        Ok(Self {
            id: role,
            bits: table.bits,
            lanes,
            binding,
            nonce,
            resources: r,
            state: Some(State::Ready(keys)),
            _lowered: table.lowered.as_ref().map(Arc::clone),
            lease: reservation.split(r.party_reservation_bytes)?,
        })
    }

    pub fn cancel(&mut self) {
        self.state = None;
        self._lowered = None;
    }

    pub fn is_consumed(&self) -> bool {
        self.state.is_none()
    }

    pub fn start(&mut self, shares: &[u64]) -> Result<Frame<'a>, Error> {
        let lowered = self._lowered.take();
        let State::Ready(keys) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        let mask = (1u64 << self.bits) - 1;
        if shares.len() != self.lanes || shares.iter().any(|&v| v > mask) {
            return Err(Error::Domain);
        }
        let mut opened = Zeroizing::new(exact(keys.len(), self.lease.budget)?);
        for (i, key) in keys.iter().enumerate() {
            opened.push(shares[i % self.lanes].wrapping_add(*key.mask) & mask);
        }
        let mut bytes = Zeroizing::new(exact(
            self.resources.peer_frame_bytes_per_party,
            self.lease.budget,
        )?);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.binding);
        bytes.extend_from_slice(&self.nonce);
        bytes.extend_from_slice(&[self.bits, 0, self.id, 0]);
        bytes.extend_from_slice(&(self.lanes as u32).to_le_bytes());
        for word in opened.iter() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let frame = Frame {
            bytes,
            _lease: self
                .lease
                .split(self.resources.peer_frame_bytes_per_party)?,
        };
        self.state = Some(State::Opened(keys, opened));
        self._lowered = lowered;
        Ok(frame)
    }

    pub fn finish(&mut self, peer: &[u8]) -> Result<Output<'a>, Error> {
        let _lowered = self._lowered.take();
        let State::Opened(keys, opened) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        let mask = (1u64 << self.bits) - 1;
        if peer.len() != self.resources.peer_frame_bytes_per_party
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != self.nonce
            || peer[72..76] != [self.bits, 0, 1 - self.id, 0]
            || peer[76..80] != (self.lanes as u32).to_le_bytes()
            || peer[HEADER_BYTES..]
                .chunks_exact(8)
                .any(|b| u64::from_le_bytes(b.try_into().unwrap()) > mask)
        {
            return Err(Error::Context);
        }
        let mut values = Zeroizing::new(exact(self.lanes, self.lease.budget)?);
        values.resize(self.lanes, 0u64);
        for (i, key) in keys.into_iter().enumerate() {
            let other = super::read_word(peer, i);
            let input = Zeroizing::new(opened[i].wrapping_add(other) & mask);
            let contribution = key.lookup.evaluate(*input).map_err(primitive_error)?;
            let lane = i % self.lanes;
            values[lane] = values[lane].wrapping_add(contribution[0]) & mask;
        }
        Ok(Output {
            values,
            _lease: self.lease.split(self.resources.output_bytes_per_party)?,
        })
    }
}

#[cfg(test)]
#[path = "lookup_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "table_cache_tests.rs"]
mod table_cache_tests;
