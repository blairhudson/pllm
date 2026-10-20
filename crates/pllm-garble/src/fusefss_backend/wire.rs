//! Native one-party material transport. Outer transport MUST authenticate sender,
//! recipient and issuance. SHA-256 here detects corruption, not malicious forgery.
//! Never serialize a dealer, seed generator, translated partition or peer share.
//!
//! Schema `PLLMFW01`: 8-byte magic; kind/role bytes and six zero reserved bytes;
//! 32-byte public-context binding; 32-byte issuance nonce; u64 body length;
//! body; SHA-256 of all preceding bytes. Integers are native little endian,
//! without Rust padding, platform usize fields, optional counts or extensions.
//! Kinds: 1 affine, 2 conversion, 3 public lookup, 4 Boolean predicates. Public
//! geometry selects every key width/type/count; no serialized count allocates.
//! Affine lanes contain own mask, triple, base pair and key(s). Conversions
//! contain triples OR own mask/base and full/low vector keys in that order.
//! Lookup units contain own mask/base pair/universal scalar key, chunk-major.
//! Predicates contain own mask; fixed-count (u64 width, u64 public shift, scalar
//! key) queries; fixed-count (u64 opcode, u64 argument share) VM instructions;
//! fixed-count Boolean triples. Their public descriptor bounds and validates VM
//! stack geometry before allocation. Public tables never travel in this packet.
use super::{allocate, Budget, Error, Lease, MAX_ALLOCATION_BYTES};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

#[path = "wire_file.rs"]
mod file;
pub use file::{IndexedFileResources, MAX_INDEXED_FILE_DISK_BYTES};

const MAGIC: &[u8; 8] = b"PLLMFW01";
const HEADER: usize = 88;
/// Fixed namespace/type/recipient/binding/issuance/length header plus checksum.
pub const ENVELOPE_BYTES: usize = HEADER + 32;
const OVERHEAD: usize = ENVELOPE_BYTES;

/// Secret packet; backing capacity remains charged until bytes are erased.
/// Copies made by external transports require their own accounting and erasure.
pub struct MaterialPacket<'a> {
    bytes: Zeroizing<Vec<u8>>,
    _lease: Lease<'a>,
}
impl<'a> MaterialPacket<'a> {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Bounded copy at the authenticated transport boundary. No parsing or RNG.
    pub fn from_bytes(bytes: &[u8], budget: &'a Budget) -> Result<Self, Error> {
        if !(OVERHEAD..=MAX_ALLOCATION_BYTES).contains(&bytes.len()) {
            return Err(Error::Resource);
        }
        let lease = budget.reserve(bytes.len())?;
        let mut copy = Zeroizing::new(allocate(bytes.len(), budget)?);
        copy.extend_from_slice(bytes);
        Ok(Self {
            bytes: copy,
            _lease: lease,
        })
    }
    pub fn nonce(&self) -> [u8; 32] {
        self.bytes[48..80].try_into().unwrap()
    }
    pub(crate) fn body_len(&self) -> usize {
        self.bytes.len() - OVERHEAD
    }
}

/// Validated, allocation-free replay storage geometry. Private fields prevent
/// callers from underpricing a guard. The historical small ledger and the
/// request-sized indexed ledger have deliberately separate constructors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportGuardLayout {
    capacity: usize,
    slots: usize,
    reservation_bytes: usize,
}
impl ImportGuardLayout {
    /// Historical ImportGuard::new geometry and accounting, unchanged.
    pub fn legacy(capacity: usize) -> Result<Self, Error> {
        if !(1..=4096).contains(&capacity) {
            return Err(Error::Resource);
        }
        Ok(Self {
            capacity,
            slots: capacity,
            reservation_bytes: capacity * 33,
        })
    }

    /// Price the entire known request, including any allowed failed imports.
    /// Each distinct (role, nonce) consumes one slot of the claim allowance;
    /// callers sharing an owner between both roles must count both claims.
    ///
    /// A power-of-two open-addressed table stays at most half full. Full keys,
    /// not fingerprints, are retained. Charge all table slots plus inline owner
    /// metadata under the existing 64 MiB ceiling. No allocation or RNG here.
    pub fn for_requests(capacity: usize) -> Result<Self, Error> {
        let slots = indexed_slots(capacity)?;
        let reservation_bytes = slots
            .checked_mul(std::mem::size_of::<[u8; 33]>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<ImportGuard<'_>>()))
            .filter(|bytes| *bytes <= MAX_ALLOCATION_BYTES)
            .ok_or(Error::Resource)?;
        Ok(Self {
            capacity,
            slots,
            reservation_bytes,
        })
    }

    pub fn capacity(self) -> usize {
        self.capacity
    }
    pub fn reservation_bytes(self) -> usize {
        self.reservation_bytes
    }
    fn indexed(self) -> bool {
        self.slots != self.capacity
    }
}

/// Bounded exact replay ledger for one authenticated receiving endpoint. Keep
/// this owner across ALL contexts for the entire request/session; never recreate
/// it to retry a failed import or after cancelling an individual runtime.
/// Claims burn before parsing/allocation. Persistent/restart replay prevention
/// belongs to outer authenticated transport; byte codecs cannot provide it.
/// This owner does not enforce schedule order: the waiting runtime/transport does.
pub struct ImportGuard<'a> {
    // Field order is significant: erase/free every slot BEFORE refunding lease.
    claims: Zeroizing<Vec<[u8; 33]>>,
    file: Option<file::Ledger>,
    layout: ImportGuardLayout,
    used: usize,
    _lease: Lease<'a>,
}
impl<'a> ImportGuard<'a> {
    /// Historical bounded linear ledger; existing importers retain its exact
    /// capacity, allocation, error precedence and capacity * 33 byte accounting.
    pub fn new(capacity: usize, budget: &'a Budget) -> Result<Self, Error> {
        Self::with_layout(ImportGuardLayout::legacy(capacity)?, budget)
    }

    /// Request-long exact indexed ledger with the full known claim allowance.
    /// This is the same allocation priced by indexed_reservation_bytes.
    pub fn new_indexed(capacity: usize, budget: &'a Budget) -> Result<Self, Error> {
        Self::with_layout(ImportGuardLayout::for_requests(capacity)?, budget)
    }

    /// Allocation-free, checked price of new_indexed, including all backing
    /// slots and owner metadata. Zero/oversized capacities fail with Resource.
    pub fn indexed_reservation_bytes(capacity: usize) -> Result<usize, Error> {
        Ok(ImportGuardLayout::for_requests(capacity)?.reservation_bytes())
    }

    /// Fixed resident price and separate full-request disk reservation for one
    /// receiving endpoint. Includes both fixed read/write scratch buffers.
    pub fn indexed_file_resources(capacity: usize) -> Result<IndexedFileResources, Error> {
        file::resources(capacity)
    }

    /// Optional private, unlinked, sparse-file exact ledger. The caller must
    /// reserve disk_bytes for its WHOLE lifetime, independently of native Budget
    /// and other source/tape files; set_len is not a physical-space reservation.
    /// No path or file handle escapes. Nonces are public: this makes no physical
    /// secure-erasure or crash/restart persistence claim. Any I/O failure closes
    /// the file and permanently poisons this guard, never permitting a retry.
    pub fn new_indexed_file(
        capacity: usize,
        max_disk_bytes: u64,
        budget: &'a Budget,
    ) -> Result<Self, Error> {
        let resources = Self::indexed_file_resources(capacity)?;
        if max_disk_bytes > MAX_INDEXED_FILE_DISK_BYTES || resources.disk_bytes > max_disk_bytes {
            return Err(Error::Resource);
        }
        // All resident scratch is admitted BEFORE filesystem creation. Failure
        // drops any opened file before this lease can be refunded.
        let lease = budget.reserve(resources.resident_bytes)?;
        budget.allocation()?;
        let file = file::Ledger::new(resources.disk_bytes)?;
        Ok(Self {
            claims: Zeroizing::new(Vec::new()),
            file: Some(file),
            layout: ImportGuardLayout {
                capacity,
                slots: indexed_slots(capacity)?,
                reservation_bytes: resources.resident_bytes,
            },
            used: 0,
            _lease: lease,
        })
    }

    /// Allocate the complete reviewed layout once, before accepting any claim.
    /// Indexed claims need neither heap allocation nor table growth/rehashing.
    pub fn with_layout(layout: ImportGuardLayout, budget: &'a Budget) -> Result<Self, Error> {
        let lease = budget.reserve(layout.reservation_bytes())?;
        let mut claims = Zeroizing::new(allocate(layout.slots, budget)?);
        if layout.indexed() {
            // Zero recipient tag means empty; occupied tags are role + 1. Thus
            // every 256-bit nonce, including all-zero, remains representable.
            claims.resize(layout.slots, [0; 33]);
        }
        Ok(Self {
            claims,
            file: None,
            layout,
            used: 0,
            _lease: lease,
        })
    }

    pub(crate) fn claim(&mut self, role: u8, nonce: [u8; 32]) -> Result<(), Error> {
        if let Some(file) = &mut self.file {
            file.claim(
                role,
                nonce,
                self.layout.slots,
                self.used == self.layout.capacity,
            )?;
            self.used += 1;
            return Ok(());
        }
        if role > 1 {
            return Err(Error::Domain);
        }
        let mut claim = [0; 33];
        claim[..32].copy_from_slice(&nonce);
        if !self.layout.indexed() {
            claim[32] = role;
            if self.claims.contains(&claim) {
                return Err(Error::Consumed);
            }
            if self.used == self.layout.capacity {
                return Err(Error::Resource);
            }
            self.claims.push(claim);
        } else {
            claim[32] = role + 1;
            let slot = self.find_slot(&claim)?;
            // Search precedes exhaustion so duplicates still report Consumed
            // when the request has used its entire allowance.
            if self.used == self.layout.capacity {
                return Err(Error::Resource);
            }
            self.claims[slot] = claim;
        }
        self.used += 1;
        Ok(())
    }

    fn find_slot(&self, claim: &[u8; 33]) -> Result<usize, Error> {
        // SHA-256 only selects probes. Equality ALWAYS compares all 33 bytes;
        // hash collisions cannot cause false acceptance or duplicate rejection.
        // Double hashing avoids linear clusters for random issuer nonces, giving
        // expected constant lookup work rather than scanning all prior claims.
        let mask = self.layout.slots - 1;
        let (mut slot, stride) = indexed_probes(claim, self.layout.slots);
        for _ in 0..self.layout.slots {
            let entry = &self.claims[slot];
            if entry[32] == 0 {
                return Ok(slot);
            }
            if entry == claim {
                return Err(Error::Consumed);
            }
            slot = (slot + stride) & mask;
        }
        // The half-full invariant ensures a free slot, but fail closed even if
        // internal geometry ever becomes inconsistent. Never evict a claim.
        Err(Error::Resource)
    }
}

fn indexed_slots(capacity: usize) -> Result<usize, Error> {
    if capacity == 0 {
        return Err(Error::Resource);
    }
    capacity
        .checked_mul(2)
        .and_then(usize::checked_next_power_of_two)
        .ok_or(Error::Resource)
}

fn indexed_probes(claim: &[u8; 33], slots: usize) -> (usize, usize) {
    let digest = Sha256::digest(claim);
    let mask = slots - 1;
    let slot = u64::from_le_bytes(digest[..8].try_into().unwrap()) as usize & mask;
    // An odd stride visits every slot in a power-of-two table.
    let stride = (u64::from_le_bytes(digest[8..16].try_into().unwrap()) as usize | 1) & mask;
    (slot, stride)
}

pub(crate) fn packet(
    kind: u8,
    role: u8,
    binding: [u8; 32],
    nonce: [u8; 32],
    body: usize,
    budget: &Budget,
    write: impl FnOnce(&mut Vec<u8>) -> Result<(), Error>,
) -> Result<MaterialPacket<'_>, Error> {
    let len = body
        .checked_add(OVERHEAD)
        .filter(|n| *n <= MAX_ALLOCATION_BYTES)
        .ok_or(Error::Resource)?;
    let lease = budget.reserve(len)?;
    let mut bytes = Zeroizing::new(allocate(len, budget)?);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&[kind, role, 0, 0, 0, 0, 0, 0]);
    bytes.extend_from_slice(&binding);
    bytes.extend_from_slice(&nonce);
    bytes.extend_from_slice(&(body as u64).to_le_bytes());
    write(&mut bytes)?;
    if bytes.len() != len - 32 || bytes.capacity() != len {
        return Err(Error::Resource);
    }
    let digest = Sha256::digest(&*bytes);
    bytes.extend_from_slice(&digest);
    Ok(MaterialPacket {
        bytes,
        _lease: lease,
    })
}

pub(crate) fn open<'a>(
    packet: &'a MaterialPacket<'_>,
    kind: u8,
    role: u8,
    binding: [u8; 32],
    nonce: [u8; 32],
    body: usize,
) -> Result<Reader<'a>, Error> {
    let bytes = packet.as_bytes();
    if bytes.len() != body.checked_add(OVERHEAD).ok_or(Error::Resource)?
        || &bytes[..8] != MAGIC
        || bytes[8..16] != [kind, role, 0, 0, 0, 0, 0, 0]
        || bytes[16..48] != binding
        || bytes[48..80] != nonce
        || bytes[80..88] != (body as u64).to_le_bytes()
        || bytes[bytes.len() - 32..] != Sha256::digest(&bytes[..bytes.len() - 32])[..]
    {
        return Err(Error::Context);
    }
    Ok(Reader {
        bytes: &bytes[HEADER..bytes.len() - 32],
        offset: 0,
    })
}

pub(crate) fn word(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    pub(crate) fn fork(&self) -> Self {
        Self {
            bytes: self.bytes,
            offset: self.offset,
        }
    }
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.offset.checked_add(n).ok_or(Error::Context)?;
        let result = self.bytes.get(self.offset..end).ok_or(Error::Context)?;
        self.offset = end;
        Ok(result)
    }
    pub(crate) fn word(&mut self, bits: u8) -> Result<u64, Error> {
        let value = u64::from_le_bytes(self.take(8)?.try_into().unwrap());
        if !(1..=50).contains(&bits) || value >= 1u64 << bits {
            return Err(Error::Context);
        }
        Ok(value)
    }
    pub(crate) fn finish(self) -> Result<(), Error> {
        if self.offset != self.bytes.len() {
            Err(Error::Context)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;
