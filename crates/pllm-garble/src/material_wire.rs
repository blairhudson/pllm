//! Shared bounded secret-packet storage and caller-namespaced envelope.
//! SHA-256 detects corruption, not malicious forgery. Authentication and replay
//! prevention belong to the caller's transport/lifecycle, not this codec.
use crate::shared_allocation::{allocate, Budget, Error, Lease, MAX_ALLOCATION_BYTES};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub(crate) const HEADER: usize = 88;
/// Fixed namespace/type/recipient/binding/issuance/length header plus checksum.
pub const ENVELOPE_BYTES: usize = HEADER + 32;

/// Secret packet; backing capacity remains charged until bytes are erased.
/// Copies made by external transports require their own accounting and erasure.
pub struct MaterialPacket<'b> {
    // Field order erases/frees bytes BEFORE refunding the capacity lease.
    bytes: Zeroizing<Vec<u8>>,
    _lease: Lease<'b>,
}
impl<'b> MaterialPacket<'b> {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Full backing capacity, including spare bytes; not the envelope length.
    pub fn capacity_bytes(&self) -> usize {
        self.bytes.capacity()
    }
    /// Consume this standalone byte owner into another admitted Budget lifetime.
    /// Reserve the FULL backing capacity before refunding its original charge,
    /// even for the same Budget. No allocation, copy, parsing or RNG occurs.
    /// Failure consumes/erases this packet; success keeps erasure before the
    /// final destination refund. No native party or borrowed payload is moved.
    pub fn into_budget(self, budget: &Budget) -> Result<MaterialPacket<'_>, Error> {
        let lease = budget.reserve(self.capacity_bytes())?;
        let Self {
            bytes,
            _lease: source_lease,
        } = self;
        let packet = MaterialPacket {
            bytes,
            _lease: lease,
        };
        drop(source_lease);
        Ok(packet)
    }
    /// Bounded copy at the authenticated transport boundary. No parsing or RNG.
    pub fn from_bytes(bytes: &[u8], budget: &'b Budget) -> Result<Self, Error> {
        if !(ENVELOPE_BYTES..=MAX_ALLOCATION_BYTES).contains(&bytes.len()) {
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
        self.bytes.len() - ENVELOPE_BYTES
    }
}

// Keep namespace and every envelope binding explicit at the shared boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn packet<'b>(
    magic: &[u8; 8],
    kind: u8,
    role: u8,
    binding: [u8; 32],
    nonce: [u8; 32],
    body: usize,
    budget: &'b Budget,
    write: impl FnOnce(&mut Vec<u8>) -> Result<(), Error>,
) -> Result<MaterialPacket<'b>, Error> {
    let len = body
        .checked_add(ENVELOPE_BYTES)
        .filter(|n| *n <= MAX_ALLOCATION_BYTES)
        .ok_or(Error::Resource)?;
    let lease = budget.reserve(len)?;
    let mut bytes = Zeroizing::new(allocate(len, budget)?);
    bytes.extend_from_slice(magic);
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

/// Validate caller-selected geometry and bindings without allocating or claiming
/// issuance. Returned body borrows the packet's existing secret storage.
pub(crate) fn open<'a>(
    packet: &'a MaterialPacket<'_>,
    magic: &[u8; 8],
    kind: u8,
    role: u8,
    binding: [u8; 32],
    nonce: [u8; 32],
    body: usize,
) -> Result<&'a [u8], Error> {
    let bytes = packet.as_bytes();
    if bytes.len() != body.checked_add(ENVELOPE_BYTES).ok_or(Error::Resource)?
        || &bytes[..8] != magic
        || bytes[8..16] != [kind, role, 0, 0, 0, 0, 0, 0]
        || bytes[16..48] != binding
        || bytes[48..80] != nonce
        || bytes[80..88] != (body as u64).to_le_bytes()
        || bytes[bytes.len() - 32..] != Sha256::digest(&bytes[..bytes.len() - 32])[..]
    {
        return Err(Error::Context);
    }
    Ok(&bytes[HEADER..bytes.len() - 32])
}

/// Decode a borrowed fixed little-endian record without heap/packet staging.
/// Geometry is checked before reading any word; partial private output erases on
/// error/unwind. The caller prices the fixed inline transient, retains the byte
/// owner's charge, and still owns envelope/context/recipient/issuance validation.
/// Width matches the existing arithmetic-material Reader (1..=50), not every
/// algebraic ring supported elsewhere. No masking or alternate encoding occurs.
pub(crate) fn decode_fixed_words<const N: usize>(
    bytes: &[u8],
    bits: u8,
) -> Result<Zeroizing<[u64; N]>, Error> {
    if bytes.len()
        != N.checked_mul(std::mem::size_of::<u64>())
            .ok_or(Error::Context)?
        || !(1..=50).contains(&bits)
    {
        return Err(Error::Context);
    }
    let limit = 1u64 << bits;
    let mut words = Zeroizing::new([0; N]);
    for (word, encoded) in words.iter_mut().zip(bytes.chunks_exact(8)) {
        *word = u64::from_le_bytes(encoded.try_into().unwrap());
        if *word >= limit {
            return Err(Error::Context);
        }
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use zeroize::Zeroize;

    const MAGIC: &[u8; 8] = b"PLLMFW01";
    const BODY: &[u8] = &[0xa5, 0, 0xff];
    // Frozen historical PLLMFW01 bytes, including an independently computed
    // checksum. Do not regenerate from the implementation under test.
    const GOLDEN: &str = concat!(
        "504c4c4d465730310301000000000000",
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
        "0300000000000000a500ff",
        "0c5e67c6b4d4655312ab0d824232daed75f5450d63c4d1d85aff2b0ecbd7c2cb"
    );

    fn binding() -> [u8; 32] {
        std::array::from_fn(|i| i as u8)
    }
    fn nonce() -> [u8; 32] {
        std::array::from_fn(|i| (i + 32) as u8)
    }
    fn golden() -> Vec<u8> {
        GOLDEN
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let digit = |b: u8| match b {
                    b'0'..=b'9' => b - b'0',
                    b'a'..=b'f' => b - b'a' + 10,
                    _ => panic!("invalid golden hex"),
                };
                digit(pair[0]) * 16 + digit(pair[1])
            })
            .collect()
    }
    fn make(budget: &Budget) -> MaterialPacket<'_> {
        packet(MAGIC, 3, 1, binding(), nonce(), BODY.len(), budget, |out| {
            out.extend_from_slice(BODY);
            Ok(())
        })
        .unwrap()
    }
    fn read<'a>(packet: &'a MaterialPacket<'_>) -> Result<&'a [u8], Error> {
        open(packet, MAGIC, 3, 1, binding(), nonce(), BODY.len())
    }
    fn checksum(bytes: &mut [u8]) {
        let end = bytes.len() - 32;
        let digest = Sha256::digest(&bytes[..end]);
        bytes[end..].copy_from_slice(&digest);
    }

    #[test]
    fn historical_bytes_and_public_fusefss_paths_match() {
        let budget = Budget::new(1024).unwrap();
        assert_eq!(HEADER, 88);
        assert_eq!(ENVELOPE_BYTES, 120);
        let packet = make(&budget);
        assert_eq!(packet.as_bytes(), golden());
        assert_eq!(packet.nonce(), nonce());
        assert_eq!(packet.body_len(), BODY.len());
        assert_eq!(read(&packet), Ok(BODY));
        let legacy = crate::fusefss_backend::wire::packet(
            3,
            1,
            binding(),
            nonce(),
            BODY.len(),
            &budget,
            |out| {
                out.extend_from_slice(BODY);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(legacy.as_bytes(), packet.as_bytes());
        let copied: crate::fusefss_backend::wire::MaterialPacket<'_> =
            MaterialPacket::from_bytes(&golden(), &budget).unwrap();
        assert_eq!(copied.as_bytes(), legacy.as_bytes());
        let mut reader =
            crate::fusefss_backend::wire::open(&copied, 3, 1, binding(), nonce(), BODY.len())
                .unwrap();
        assert_eq!(reader.take(BODY.len()).unwrap(), BODY);
        reader.finish().unwrap();
    }

    #[test]
    fn envelope_fields_checked_even_with_valid_checksum() {
        let budget = Budget::new(1024).unwrap();
        for offset in 0..HEADER {
            let mut bytes = golden();
            bytes[offset] ^= 1;
            checksum(&mut bytes);
            let packet = MaterialPacket::from_bytes(&bytes, &budget).unwrap();
            assert_eq!(read(&packet), Err(Error::Context), "offset {offset}");
        }
        for offset in HEADER..ENVELOPE_BYTES + BODY.len() {
            let mut bytes = golden();
            bytes[offset] ^= 1;
            let packet = MaterialPacket::from_bytes(&bytes, &budget).unwrap();
            assert_eq!(read(&packet), Err(Error::Context), "offset {offset}");
        }
        for extra in [false, true] {
            let mut bytes = golden();
            if extra {
                bytes.push(0);
            } else {
                bytes.pop();
            }
            let packet = MaterialPacket::from_bytes(&bytes, &budget).unwrap();
            assert_eq!(read(&packet), Err(Error::Context));
        }
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn caller_namespace_is_used_and_open_borrows_without_budget_allocation() {
        let budget = Budget::new(ENVELOPE_BYTES + BODY.len()).unwrap();
        let packet = packet(
            b"OTHER001",
            3,
            1,
            binding(),
            nonce(),
            BODY.len(),
            &budget,
            |out| {
                out.extend_from_slice(BODY);
                Ok(())
            },
        )
        .unwrap();
        budget.fail_after.store(0, Ordering::SeqCst);
        assert_eq!(&packet.as_bytes()[..8], b"OTHER001");
        assert_eq!(read(&packet), Err(Error::Context));
        let body = open(&packet, b"OTHER001", 3, 1, binding(), nonce(), BODY.len()).unwrap();
        assert_eq!(body, BODY);
        assert_eq!(body.as_ptr(), packet.as_bytes()[HEADER..].as_ptr());
        assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
        assert_eq!(budget.used_bytes(), ENVELOPE_BYTES + BODY.len());
        drop(packet);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn bounds_error_precedence_and_failed_writes_refund() {
        let budget = Budget::new(1024).unwrap();
        budget.fail_after.store(0, Ordering::SeqCst);
        for body in [usize::MAX, MAX_ALLOCATION_BYTES, 1024] {
            assert!(matches!(
                packet(MAGIC, 3, 1, binding(), nonce(), body, &budget, |_| panic!(
                    "write called"
                )),
                Err(Error::Resource)
            ));
        }
        assert!(matches!(
            MaterialPacket::from_bytes(&[0; ENVELOPE_BYTES - 1], &budget),
            Err(Error::Resource)
        ));
        assert!(matches!(
            MaterialPacket::from_bytes(&golden(), &budget),
            Err(Error::Allocation)
        ));
        assert!(matches!(
            packet(MAGIC, 3, 1, binding(), nonce(), 3, &budget, |_| panic!(
                "write called"
            )),
            Err(Error::Allocation)
        ));
        assert_eq!(budget.used_bytes(), 0);
        budget.fail_after.store(-1, Ordering::SeqCst);
        assert!(matches!(
            packet(MAGIC, 3, 1, binding(), nonce(), 3, &budget, |out| {
                out.push(0xa5);
                Err(Error::Randomness)
            }),
            Err(Error::Randomness)
        ));
        assert!(matches!(
            packet(MAGIC, 3, 1, binding(), nonce(), 3, &budget, |_| Ok(())),
            Err(Error::Resource)
        ));
        assert!(matches!(
            packet(MAGIC, 3, 1, binding(), nonce(), 3, &budget, |out| {
                out.extend_from_slice(&[0; 4]);
                Ok(())
            }),
            Err(Error::Resource)
        ));
        assert!(matches!(
            packet(MAGIC, 3, 1, binding(), nonce(), 3, &budget, |out| {
                out.reserve_exact(1024);
                out.extend_from_slice(BODY);
                Ok(())
            }),
            Err(Error::Resource)
        ));
        assert_eq!(budget.used_bytes(), 0);
        let packet = make(&budget);
        assert_eq!(
            open(&packet, MAGIC, 0, 0, [0; 32], [0; 32], usize::MAX),
            Err(Error::Resource)
        );
    }

    #[test]
    fn secret_storage_erases_before_lease_refund() {
        let budget = Budget::new(1024).unwrap();
        let mut packet = make(&budget);
        let capacity = packet.bytes.capacity();
        assert_eq!(budget.used_bytes(), capacity);
        packet.bytes.zeroize();
        assert!(packet.bytes.is_empty());
        assert_eq!(budget.used_bytes(), capacity);
        drop(packet);
        assert_eq!(budget.used_bytes(), 0);
        let copied = MaterialPacket::from_bytes(&golden(), &budget).unwrap();
        assert_eq!(budget.used_bytes(), golden().len());
        drop(copied);
        assert_eq!(budget.used_bytes(), 0);
    }

    // Test-only private construction: public packet constructors remain exact
    // capacity. Initialize a secret tail before truncation to model spare backing.
    pub(super) fn with_spare(budget: &Budget, spare: usize) -> MaterialPacket<'_> {
        let envelope = golden();
        let capacity = envelope.len() + spare;
        let lease = budget.reserve(capacity).unwrap();
        let mut bytes = Zeroizing::new(allocate(capacity, budget).unwrap());
        bytes.extend_from_slice(&envelope);
        bytes.resize(capacity, 0xa5);
        bytes.truncate(envelope.len());
        MaterialPacket {
            bytes,
            _lease: lease,
        }
    }

    #[test]
    fn transfer_keeps_pointer_capacity_envelope_and_outlives_original_budget() {
        let destination = Budget::new(4096).unwrap();
        let (moved, pointer, capacity) = {
            let source = Budget::new(4096).unwrap();
            let packet = with_spare(&source, 73);
            let pointer = packet.as_bytes().as_ptr();
            let capacity = packet.capacity_bytes();
            assert!(capacity > packet.as_bytes().len());
            let moved = packet.into_budget(&destination).unwrap();
            assert_eq!(source.used_bytes(), 0);
            (moved, pointer, capacity)
        }; // Original Budget is gone; no lease or payload borrows it.
        assert_eq!(moved.as_bytes().as_ptr(), pointer);
        assert_eq!(moved.capacity_bytes(), capacity);
        assert_eq!(moved.as_bytes(), golden());
        assert_eq!(moved.nonce(), nonce());
        assert_eq!(read(&moved), Ok(BODY));
        assert_eq!(destination.used_bytes(), capacity);
        drop(moved);
        assert_eq!(destination.used_bytes(), 0);
    }

    #[test]
    fn transfer_admits_actual_capacity_exact_fit_and_consumes_on_shortfall() {
        for spare in [0, 1, 4096] {
            let capacity = golden().len() + spare;
            for short in [false, true] {
                let source = Budget::new(capacity).unwrap();
                let destination = Budget::new(capacity - usize::from(short)).unwrap();
                let packet = with_spare(&source, spare);
                let pointer = packet.as_bytes().as_ptr();
                let result = packet.into_budget(&destination);
                assert_eq!(source.used_bytes(), 0);
                if short {
                    assert!(matches!(result, Err(Error::Resource)));
                    assert_eq!(destination.used_bytes(), 0);
                } else {
                    let moved = result.unwrap();
                    assert_eq!(moved.as_bytes().as_ptr(), pointer);
                    assert_eq!(destination.used_bytes(), capacity);
                    assert_eq!(moved.capacity_bytes(), capacity);
                    drop(moved);
                    assert_eq!(destination.used_bytes(), 0);
                }
            }
        }
    }

    #[test]
    fn same_budget_transfer_requires_full_overlap_before_source_refund() {
        for spare in [0, 41] {
            let capacity = golden().len() + spare;
            for limit in [capacity, 2 * capacity - 1, 2 * capacity] {
                let budget = Budget::new(limit).unwrap();
                let packet = with_spare(&budget, spare);
                let pointer = packet.as_bytes().as_ptr();
                let result = packet.into_budget(&budget);
                if limit < 2 * capacity {
                    // Refunding before admission would incorrectly succeed here.
                    assert!(matches!(result, Err(Error::Resource)));
                    assert_eq!(budget.used_bytes(), 0);
                } else {
                    let moved = result.unwrap();
                    assert_eq!(moved.as_bytes().as_ptr(), pointer);
                    assert_eq!(budget.used_bytes(), capacity);
                    drop(moved);
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }

    #[test]
    fn transfer_removes_allocation_not_destination_admission_or_from_bytes_failpoint() {
        let capacity = golden().len();
        let source = Budget::new(capacity).unwrap();
        let destination = Budget::new(capacity).unwrap();
        let packet = make(&source);
        source.fail_after.store(0, Ordering::SeqCst);
        destination.fail_after.store(0, Ordering::SeqCst);
        let moved = packet.into_budget(&destination).unwrap();
        assert_eq!(source.used_bytes(), 0);
        assert_eq!(destination.used_bytes(), capacity);
        assert_eq!(source.fail_after.load(Ordering::SeqCst), 0);
        assert_eq!(destination.fail_after.load(Ordering::SeqCst), 0);
        drop(moved);
        // The existing copying API still reaches its original allocation hook.
        assert!(matches!(
            MaterialPacket::from_bytes(&golden(), &destination),
            Err(Error::Allocation)
        ));
        assert_eq!(destination.used_bytes(), 0);
        let source = Budget::new(capacity).unwrap();
        let short = Budget::new(capacity - 1).unwrap();
        let packet = make(&source);
        short.fail_after.store(0, Ordering::SeqCst);
        assert!(matches!(packet.into_budget(&short), Err(Error::Resource)));
        assert_eq!(source.used_bytes(), 0);
        assert_eq!(short.used_bytes(), 0);
        assert_eq!(short.fail_after.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn transferred_secret_erases_before_final_destination_refund() {
        let capacity = golden().len() + 73;
        let source = Budget::new(capacity).unwrap();
        let destination = Budget::new(capacity).unwrap();
        let mut moved = with_spare(&source, 73).into_budget(&destination).unwrap();
        assert_eq!(source.used_bytes(), 0);
        moved.bytes.zeroize();
        assert!(moved.as_bytes().is_empty());
        assert_eq!(moved.capacity_bytes(), capacity);
        assert_eq!(destination.used_bytes(), capacity);
        assert!(matches!(destination.reserve(1), Err(Error::Resource)));
        drop(moved);
        assert_eq!(destination.used_bytes(), 0);
    }

    #[test]
    fn transfer_unwind_drops_backing_before_destination_lease_refund() {
        let source = Budget::new(4096).unwrap();
        let destination = Budget::new(4096).unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let moved = with_spare(&source, 73).into_budget(&destination).unwrap();
            assert_eq!(source.used_bytes(), 0);
            assert_eq!(destination.used_bytes(), moved.capacity_bytes());
            assert_eq!(read(&moved), Ok(BODY));
            panic!("unwind transferred packet");
        }));
        assert!(outcome.is_err());
        // MaterialPacket's unchanged field order drops Zeroizing before Lease.
        assert_eq!(source.used_bytes(), 0);
        assert_eq!(destination.used_bytes(), 0);
    }
}

#[cfg(test)]
#[path = "material_wire_queue_tests.rs"]
mod queue_tests;

#[cfg(test)]
#[path = "material_wire_fixed_tests.rs"]
mod fixed_tests;
