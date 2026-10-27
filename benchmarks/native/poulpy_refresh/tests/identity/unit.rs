//! Tiny ordinary native regressions. Written, not executed by this lane.
use super::*;
use std::io::Cursor;

fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new().unwrap();
    hash.update(bytes).unwrap();
    hash.finish().unwrap()
}

fn fixture(bytes: &[u8]) -> OneShot {
    // Private mechanics fixture, never a production constructor or HE profile.
    let binding = Binding {
        profile: digest(b"fixture-only"),
        p: -1,
        capacity: 64,
        alignment: 64,
    };
    let mut hash = binding.start().unwrap();
    hash.update(bytes).unwrap();
    OneShot {
        expected: Expected {
            binding,
            digest: hash.finish().unwrap(),
            wire_bytes: bytes.len() as u64,
        },
        attempted: false,
    }
}

#[test]
fn standard_sha256_vectors_and_sdk_context_abi() {
    assert_eq!(std::mem::size_of::<Sha256Context>(), 104);
    assert_eq!(
        digest(b""),
        [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ]
    );
    assert_eq!(
        digest(b"abc"),
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ]
    );
    let mut split = Sha256::new().unwrap();
    split.update(b"a").unwrap();
    split.update(b"b").unwrap();
    split.update(b"c").unwrap();
    assert_eq!(split.finish().unwrap(), digest(b"abc"));
}

#[test]
fn preflight_burns_success_and_every_failure_without_retry() {
    for bytes in [b"abcd".as_slice(), b"abc", b"abcde", b"abXd"] {
        let mut record = fixture(b"abcd");
        let result = record.check(Cursor::new(bytes));
        assert_eq!(result.is_ok(), bytes == b"abcd");
        assert!(matches!(
            record.check(Cursor::new(b"abcd")),
            Err(IdentityError::Burned)
        ));
    }
}

struct BrokenSeek;
impl Read for BrokenSeek {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        Ok(0)
    }
}
impl Seek for BrokenSeek {
    fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
        Err(io::ErrorKind::PermissionDenied.into())
    }
}

#[test]
fn failure_before_first_byte_also_burns_attempt() {
    let mut record = fixture(b"abcd");
    assert!(record.check(BrokenSeek).is_err());
    assert!(matches!(
        record.check(Cursor::new(b"abcd")),
        Err(IdentityError::Burned)
    ));
}

#[test]
fn geometry_profile_and_galois_binding_changes_digest() {
    let original = fixture(b"abcd");
    let expected = original.expected.digest;
    for change in 0..4 {
        let mut modified = fixture(b"abcd");
        match change {
            0 => modified.expected.binding.p = 3,
            1 => modified.expected.binding.capacity = 128,
            2 => modified.expected.binding.alignment = 128,
            _ => modified.expected.binding.profile[0] ^= 1,
        }
        modified.expected.digest = expected;
        assert!(matches!(
            modified.check(Cursor::new(b"abcd")),
            Err(IdentityError::Digest)
        ));
    }
}

#[test]
fn actual_reader_bytes_are_rehashed_not_preflight_path_or_metadata() {
    let mut record = fixture(b"abcd");
    let _verified = record.check(Cursor::new(b"abcd")).unwrap();
    // Simulate a File alias changing bytes after preflight. Production import
    // hashes exactly the bytes read by upstream ReaderFrom a second time.
    let mut reader = HashReader {
        input: Cursor::new(b"abXd"),
        hash: record.expected.binding.start().unwrap(),
        bytes: 0,
        limit: 5,
    };
    let mut parsed = [0u8; 4];
    read_record(&mut reader, 4, |input| input.read_exact(&mut parsed)).unwrap();
    assert_ne!(reader.hash.finish().unwrap(), record.expected.digest);
}

struct PartialWrite(Vec<u8>);
impl Write for PartialWrite {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let count = bytes.len().min(2);
        self.0.extend_from_slice(&bytes[..count]);
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn actual_partial_export_writes_hash_only_delivered_bytes() {
    let mut output = PartialWrite(Vec::new());
    let expected = fixture(b"abcd");
    let mut writer = HashWriter {
        output: &mut output,
        hash: expected.expected.binding.start().unwrap(),
        bytes: 0,
    };
    writer.write_all(b"abcd").unwrap();
    assert_eq!(writer.bytes, 4);
    assert_eq!(writer.hash.finish().unwrap(), expected.expected.digest);
    assert_eq!(output.0.as_slice(), b"abcd");
}

#[test]
fn retained_original_is_an_additional_charged_owner() {
    let layout = std::alloc::Layout::from_size_align(RAW_CAPACITY as usize, 64).unwrap();
    let allocation = crate::bounded_allocator::geometry(
        layout,
        crate::bounded_allocator::PAGE_BYTES,
        crate::bounded_allocator::BOOKKEEPING_TICKET,
    )
    .unwrap();
    assert_eq!(allocation.charge, 242_270_208);
}

#[test]
fn original_eager_generator_cannot_be_wrapped_after_return_under_frozen_cap() {
    // ca353257 poulpy-ckks/src/layouts/bootstrapping_keys.rs:279-319 retains
    // all 33 automorphisms, tensor, and both switches before returning. Context
    // already owns both prepared DFTs. These are only a subset lower bound:
    // unknown setup/secret/runtime/export owners are NOT assumed zero.
    let raw_bundle = 35 * RAW_CAPACITY + 3_145_728;
    assert_eq!(raw_bundle, 8_480_882_688);
    let known_subset = raw_bundle + 75_758_592 + 67_108_864;
    let frozen_ceiling = 8u64 << 30;
    assert_eq!(known_subset, 8_623_750_144);
    assert_eq!(known_subset - frozen_ceiling, 33_815_552);
    assert!(known_subset > frozen_ceiling);
}

#[test]
fn serial_keygen_scratch_and_private_heap_are_simultaneously_charged_subset() {
    // Existing source audit, not a new setup budget. All ten full-capacity
    // leaves overlap during tensor key generation: no reduced logical width,
    // advertised scratch alone, or exported file size can replace this charge.
    // core/layouts/glwe_secret_tensor.rs:268-280 owns four private heap leaves
    // even while the caller's 156-MiB keygen scratch remains allocated.
    let leaves = [
        RAW_CAPACITY as usize,
        524_288,
        524_288,
        2_097_152,
        2_097_152,
        156 << 20,
        2_097_152,
        2_097_152,
        1_048_576,
        3_145_728,
    ];
    let payload: usize = leaves.iter().sum();
    assert_eq!(payload, 419_430_400);
    let charged: usize = leaves
        .iter()
        .map(|bytes| {
            let layout = std::alloc::Layout::from_size_align(*bytes, 64).unwrap();
            crate::bounded_allocator::geometry(
                layout,
                crate::bounded_allocator::PAGE_BYTES,
                crate::bounded_allocator::BOOKKEEPING_TICKET,
            )
            .unwrap()
            .charge
        })
        .sum();
    assert_eq!(charged, 419_921_920);
    // A retained previous raw export is NOT the same raw owner. This overlap
    // would be extra, not free space credited merely because writing completed.
    assert_eq!(charged + 242_270_208, 662_192_128);
}
