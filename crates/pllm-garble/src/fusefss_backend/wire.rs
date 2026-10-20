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

/// Bounded replay ledger for one authenticated receiving endpoint. Keep this
/// owner for the entire session; never recreate it to retry a failed import.
/// Claims burn before parsing/allocation. Persistent/restart replay prevention
/// belongs to outer authenticated transport; byte codecs cannot provide it.
pub struct ImportGuard<'a> {
    claims: Zeroizing<Vec<[u8; 33]>>,
    capacity: usize,
    _lease: Lease<'a>,
}
impl<'a> ImportGuard<'a> {
    pub fn new(capacity: usize, budget: &'a Budget) -> Result<Self, Error> {
        if capacity == 0 || capacity > 4096 {
            return Err(Error::Resource);
        }
        let lease = budget.reserve(capacity * 33)?;
        Ok(Self {
            claims: Zeroizing::new(allocate(capacity, budget)?),
            capacity,
            _lease: lease,
        })
    }
    pub(crate) fn claim(&mut self, role: u8, nonce: [u8; 32]) -> Result<(), Error> {
        if role > 1 {
            return Err(Error::Domain);
        }
        let mut claim = [0; 33];
        claim[..32].copy_from_slice(&nonce);
        claim[32] = role;
        if self.claims.contains(&claim) {
            return Err(Error::Consumed);
        }
        if self.claims.len() == self.capacity {
            return Err(Error::Resource);
        }
        self.claims.push(claim);
        Ok(())
    }
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
