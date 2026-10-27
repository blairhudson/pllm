//! Owner-bound content identity, NOT publisher authenticity or role authorization.
//!
//! Expected bytes come only from WriterTo on the retained original raw owner;
//! callers cannot supply an expected digest, parameter hash, or "dense valid"
//! flag. The digest binds the frozen profile, actual role/type, Galois element,
//! original coefficient Layout and exact serialized bytes. It does NOT prove
//! which secret generated an already-existing raw evaluation key.
//!
//! Precise missing seam: upstream GLWEAutomorphismKey::read_from/write_to carries
//! p + GGLWE noise/base2k/dsize/k_aux/matrix, not a dense-key identity or epoch.
//! Only the trusted client that calls glwe_automorphism_key_encrypt_sk with its
//! original sk_dense can establish that relationship. No serial client/export
//! owner currently preserves that provenance across all 36 keys, the same sparse
//! secret for both switches, and the ciphertext owner. Thus quarantined imports
//! have NO promotion to preparation, even after byte identity passes. Do not hash
//! secret material or invent a signed/publisher-authenticated manifest here.
//!
//! CommonCrypto SHA-256 uses the installed SDK CommonDigest.h ABI: CC_LONG=u32,
//! CC_SHA256_CTX={count[2],hash[8],wbuf[16]}, Init/Update/Final -> int, libSystem.
//! No Cargo dependency, custom hash algorithm, HE arithmetic or source edit.
//! Hash/I/O runtime and kernel costs remain unqualified; this is not admission.
//! Retaining the original adds a full raw owner: 242,221,056 payload bytes and
//! 242,270,208 mapping-plus-ticket bytes. It is not free in the staged ledger.
//! This diagnostic ownership path has no complete admitted phase upper bound.
//!
//! Original-generation integration audit (pinned ca3532570ff62b96599a74f7adb85de7726b0446):
//! - This repository's main.rs:96-127 is admission-only; no client generator exists.
//! - poulpy-ckks/src/test_suite/presets.rs:112-129 samples original dense secret,
//!   prepares it, then calls context.generate_keys. Lines 155-165 encrypt the
//!   original ciphertext under that same prepared secret. This is a reference
//!   fixture, NOT the approved client: seeds at 113/119/156 are fixed/public.
//! - poulpy-ckks/src/layouts/bootstrapping_keys.rs:279-283 generates/inserts each
//!   rotation; 286-289 generates tensor; 298-308 samples ONE sparse secret and
//!   generates both switches. No export callback exists before keys accumulate.
//!
//! Local serial client IS supported by existing public primitives; no upstream
//! observer API is required. Exact candidate, with NO new arithmetic/order:
//! 1. Privately own full raw/prepared dense buffers; call GLWESecretSampling::
//!    glwe_secret_fill_ternary_hw(...,1024,...) then GLWESecretPreparedFactory::
//!    glwe_secret_prepare. No adopted buffer/tag establishes sampling provenance.
//! 2. For the original sorted Galois inventory call GLWEAutomorphismKeyEncryptSk::
//!    glwe_automorphism_key_encrypt_sk on ONE full seven-row raw owner. Stream
//!    WriterTo, derive immutable content record inside this original generation
//!    call, checked-retire raw owner BEFORE allocating the next. Tensor follows
//!    with GLWETensorKeyEncryptSk::glwe_tensor_key_encrypt_sk, same dense owner.
//! 3. Sample ONE h=32 sparse secret; preserve upstream ENCAPSULATED tag. Call
//!    GLWESwitchingKeyEncryptSk::glwe_switching_key_encrypt_sk(dense,sparse), then
//!    (...sparse,dense), with the original low/high layouts. Retire/wipe sparse
//!    only after both exports. Never resample per switch or reload.
//! 4. CKKSEncryptOps::ckks_encrypt_sk uses the SAME dense prepared owner and
//!    original input layout after key exports. Only a private owner that performs
//!    these calls may derive opaque LOCAL origin bindings; never hash secrets or
//!    reconstruct bindings from serialized keys. No remote authenticity follows.
//!
//! NEW concrete backend-contract blocker, not an absent keygen API:
//! poulpy-hal/src/source.rs:9-17 explicitly says Source (ChaCha8Rng) is NOT suitable
//! for cryptographic key generation. Above public APIs require concrete &mut
//! Source; poulpy-core/src/delegates/sampling.rs:16-23 takes source.new_seed(),
//! then poulpy-cpu-portable/src/sampling.rs:25-36 constructs another Source(seed)
//! for actual secret coefficients (65/86 do likewise for noise). Fresh separate
//! OS-random 32-byte domain seeds do NOT replace/qualify that internal stream.
//! DECISIONS.md requires secure randomness. Do not silently override this source
//! contract, swap RNG/backend, or turn an OS-seeded reference RNG into a claimed
//! secure client. Explicit qualification-only disposition of this RNG contract,
//! or an approved secure RNG sampling/encryption seam, is needed before literal
//! client generation is implemented here. This is not a claim of an RNG exploit.
//!
//! Source-proved keygen leaf SUBSET: raw key 242,221,056 + both raw/prepared
//! secret pairs 5,242,880 + 156-MiB reserved keygen scratch 163,577,856 + four
//! tensor-preparation heap leaves 8,388,608 = 419,430,400 payload. The ten leaf
//! mapping charges total 419,921,920, not a complete setup bound. The 1-GiB setup
//! target in the lifetime appendix is NOT an admitted ceiling/extra allowance.
//! Module/context/solver/codec, sources, metadata, exports and kernel I/O remain
//! additional. Ciphertext/plaintext leaf payloads at entry are 1,048,576/524,288;
//! encode/encrypt scratch and full codec ownership must also be priced.
//! Existing RotationExportOwner RETAINS raw storage: retaining all exports would
//! recreate the eager failure, so eventual trusted export must checked-retire
//! raw storage and retain only origin-bound immutable records. No such authority
//! is minted by current byte-only helper. Setup/runtime/I/O admission stays shut.

#![deny(unsafe_op_in_unsafe_fn)]

use std::{
    ffi::c_void,
    io::{self, Read, Seek, SeekFrom, Write},
};

use poulpy_core::layouts::GetGaloisElement;
use poulpy_hal::layouts::{ReaderFrom, WriterTo};

use crate::staged_preparation::{read_record, RawRotation, RotationPreparation};
use crate::staged_storage::{healthy, CheckedOwner, StorageError};

const RAW_CAPACITY: u64 = 242_221_056;
const MAX_WIRE: u64 = RAW_CAPACITY + 125;
const CHUNK: usize = 64 << 10;
const DOMAIN: &[u8] = b"pllm.poulpy.original-rotation-content.v1\0";

#[repr(C)]
struct Sha256Context {
    count: [u32; 2],
    hash: [u32; 8],
    wbuf: [u32; 16],
}
const _: () = assert!(std::mem::size_of::<Sha256Context>() == 104);
const _: () = assert!(std::mem::align_of::<Sha256Context>() == 4);

#[link(name = "System")]
unsafe extern "C" {
    fn CC_SHA256_Init(context: *mut Sha256Context) -> i32;
    fn CC_SHA256_Update(context: *mut Sha256Context, bytes: *const c_void, count: u32) -> i32;
    fn CC_SHA256_Final(digest: *mut u8, context: *mut Sha256Context) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityError {
    Io(io::ErrorKind),
    Digest,
    Length,
    Shape,
    Burned,
    HashApi,
    Storage(StorageError),
    MissingTrustedClientExportEpoch,
}

impl From<io::Error> for IdentityError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.kind())
    }
}
impl From<StorageError> for IdentityError {
    fn from(error: StorageError) -> Self {
        Self::Storage(error)
    }
}

struct Sha256 {
    context: Sha256Context,
    bytes: u64,
}

impl Sha256 {
    fn new() -> Result<Self, IdentityError> {
        // Init need not initialize wbuf. Start with fully initialized Rust data.
        let mut context = Sha256Context {
            count: [0; 2],
            hash: [0; 8],
            wbuf: [0; 16],
        };
        if unsafe { CC_SHA256_Init(&mut context) } != 1 {
            return Err(IdentityError::HashApi);
        }
        Ok(Self { context, bytes: 0 })
    }

    fn update(&mut self, bytes: &[u8]) -> Result<(), IdentityError> {
        let count = u64::try_from(bytes.len()).map_err(|_| IdentityError::Length)?;
        self.bytes = self
            .bytes
            .checked_add(count)
            .filter(|n| *n <= u64::MAX / 8)
            .ok_or(IdentityError::Length)?;
        for chunk in bytes.chunks(u32::MAX as usize) {
            if unsafe {
                CC_SHA256_Update(&mut self.context, chunk.as_ptr().cast(), chunk.len() as u32)
            } != 1
            {
                return Err(IdentityError::HashApi);
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<[u8; 32], IdentityError> {
        let mut digest = [0; 32];
        if unsafe { CC_SHA256_Final(digest.as_mut_ptr(), &mut self.context) } != 1 {
            return Err(IdentityError::HashApi);
        }
        Ok(digest)
    }
}

/// Private immutable binding. "Rotation" names a storage type, not permission
/// to evaluate or serve. There is deliberately no dense identity/epoch field.
struct Binding {
    profile: [u8; 32],
    p: i64,
    capacity: u64,
    alignment: u64,
}

impl Binding {
    fn rotation(p: i64) -> Result<Self, IdentityError> {
        let mut profile = Sha256::new()?;
        profile.update(include_bytes!("../contract.json"))?;
        Ok(Self {
            profile: profile.finish()?,
            p,
            capacity: RAW_CAPACITY,
            alignment: 64,
        })
    }

    fn start(&self) -> Result<Sha256, IdentityError> {
        let mut hash = Sha256::new()?;
        hash.update(DOMAIN)?;
        hash.update(&self.profile)?;
        hash.update(b"GLWEAutomorphismKey/NTT4x30Portable<Standard>/raw\0")?;
        hash.update(&self.p.to_le_bytes())?;
        hash.update(&self.capacity.to_le_bytes())?;
        hash.update(&self.alignment.to_le_bytes())?;
        // Actual owner was checked against the frozen full seven-row shape.
        for field in [65536u32, 52, 1, 1, 7, 4, 224, 1680, 33, 1, 8] {
            hash.update(&field.to_le_bytes())?;
        }
        Ok(hash)
    }
}

struct Expected {
    binding: Binding,
    digest: [u8; 32],
    wire_bytes: u64,
}

struct HashWriter<'a, W> {
    output: &'a mut W,
    hash: Sha256,
    bytes: u64,
}

impl<W: Write> Write for HashWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if (bytes.len() as u64) > MAX_WIRE.saturating_sub(self.bytes) {
            return Err(io::ErrorKind::InvalidData.into());
        }
        let written = self.output.write(bytes)?;
        if written > bytes.len() {
            return Err(io::ErrorKind::InvalidData.into());
        }
        self.hash
            .update(&bytes[..written])
            .map_err(|_| io::Error::from(io::ErrorKind::Other))?;
        self.bytes += written as u64;
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

struct HashReader<R> {
    input: R,
    hash: Sha256,
    bytes: u64,
    limit: u64,
}

impl<R: Read> Read for HashReader<R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let take = bytes
            .len()
            .min(self.limit.saturating_sub(self.bytes) as usize);
        if take == 0 {
            return Ok(0);
        }
        let read = self.input.read(&mut bytes[..take])?;
        if read > take {
            return Err(io::ErrorKind::InvalidData.into());
        }
        self.hash
            .update(&bytes[..read])
            .map_err(|_| io::Error::from(io::ErrorKind::Other))?;
        self.bytes += read as u64;
        Ok(read)
    }
}

/// One candidate attempt per retained export. Failure consumes the attempt even
/// before seek/read; correcting the candidate cannot rehabilitate this record.
struct OneShot {
    expected: Expected,
    attempted: bool,
}

impl OneShot {
    fn check<R: Read + Seek>(&mut self, mut input: R) -> Result<R, IdentityError> {
        if self.attempted {
            return Err(IdentityError::Burned);
        }
        self.attempted = true;
        input.seek(SeekFrom::Start(0))?;
        let mut reader = HashReader {
            input,
            hash: self.expected.binding.start()?,
            bytes: 0,
            limit: self
                .expected
                .wire_bytes
                .checked_add(1)
                .ok_or(IdentityError::Length)?,
        };
        let mut buffer = [0u8; CHUNK];
        loop {
            if reader.read(&mut buffer)? == 0 {
                break;
            }
        }
        if reader.bytes != self.expected.wire_bytes {
            return Err(IdentityError::Length);
        }
        if reader.hash.finish()? != self.expected.digest {
            return Err(IdentityError::Digest);
        }
        reader.input.seek(SeekFrom::Start(0))?;
        Ok(reader.input)
    }
}

#[must_use = "retain or checked-release the original export owner"]
pub struct RotationExportOwner {
    original: CheckedOwner<RawRotation>,
    record: OneShot,
}

impl RotationExportOwner {
    /// Adopt original raw storage and derive expectation while actually exporting
    /// it with upstream WriterTo. This makes no claim that this already-existing
    /// key came from a particular secret; that trusted client seam is still open.
    pub fn write_original<W: Write>(
        raw: RawRotation,
        output: &mut W,
    ) -> Result<Self, IdentityError> {
        let original = CheckedOwner::raw_rotation(raw)?;
        let binding = Binding::rotation(original.borrow().p())?;
        let mut writer = HashWriter {
            output,
            hash: binding.start()?,
            bytes: 0,
        };
        original.borrow().write_to(&mut writer)?;
        writer.flush()?;
        if writer.bytes < RAW_CAPACITY || writer.bytes > MAX_WIRE {
            return Err(IdentityError::Length);
        }
        if !healthy(crate::ALLOCATOR.snapshot()) {
            return Err(StorageError::AllocatorRejected.into());
        }
        let expected = Expected {
            binding,
            digest: writer.hash.finish()?,
            wire_bytes: writer.bytes,
        };
        Ok(Self {
            original,
            record: OneShot {
                expected,
                attempted: false,
            },
        })
    }

    /// Hash candidate before any import/preparation allocation. Own its reader;
    /// callers cannot swap the path/reader through this ticket. A file's external
    /// aliases can still mutate it, so import rehashes the ACTUALLY parsed bytes.
    pub fn preflight<R: Read + Seek>(
        &mut self,
        input: R,
    ) -> Result<RotationTicket<'_, R>, IdentityError> {
        if !healthy(crate::ALLOCATOR.snapshot()) {
            self.record.attempted = true;
            return Err(StorageError::AllocatorRejected.into());
        }
        let input = self.record.check(input)?;
        Ok(RotationTicket {
            export: self,
            input,
        })
    }

    pub fn release(self) -> Result<(), IdentityError> {
        self.original.release().map_err(Into::into)
    }
}

#[must_use = "content preflight is not dense-key or resource admission"]
pub struct RotationTicket<'a, R> {
    export: &'a RotationExportOwner,
    input: R,
}

impl<'a, R: Read> RotationTicket<'a, R> {
    /// Consuming ticket adopts EXISTING approved storage, not an allocator or a
    /// callback that could allocate before identity verification. Original export
    /// remains immutably borrowed throughout parse/quarantine. No raw owner leaks.
    pub fn inspect_existing(
        self,
        destination: RawRotation,
    ) -> Result<QuarantinedRotation<'a>, IdentityError> {
        let mut raw = CheckedOwner::raw_rotation(destination)?;
        let expected = &self.export.record.expected;
        let mut reader = HashReader {
            input: self.input,
            hash: expected.binding.start()?,
            bytes: 0,
            limit: expected.wire_bytes + 1,
        };
        let parsed = read_record(&mut reader, expected.wire_bytes, |record| {
            raw.borrow_mut().read_from(record)
        });
        let result = parsed.map_err(IdentityError::from).and_then(|_| {
            crate::staged_storage::high_key_layout(raw.borrow())?;
            if raw.borrow().p() != expected.binding.p {
                return Err(IdentityError::Shape);
            }
            if reader.bytes != expected.wire_bytes {
                return Err(IdentityError::Length);
            }
            if reader.hash.finish()? != expected.digest {
                return Err(IdentityError::Digest);
            }
            if !healthy(crate::ALLOCATOR.snapshot()) {
                return Err(StorageError::AllocatorRejected.into());
            }
            Ok(())
        });
        if let Err(error) = result {
            // Even overwritten/malformed imported fields cannot change the
            // original allocation Layout saved before parsing. No retry API.
            raw.release()?;
            return Err(error);
        }
        Ok(QuarantinedRotation {
            export: self.export,
            raw,
        })
    }

    /// Reject BEFORE any preparation allocation. Hash equality cannot mint a
    /// dense-key epoch. Dropping this consuming ticket does not lose its export.
    pub fn require_client_epoch(self) -> Result<(), IdentityError> {
        Err(IdentityError::MissingTrustedClientExportEpoch)
    }
}

#[must_use = "byte equality does not qualify preparation"]
pub struct QuarantinedRotation<'a> {
    export: &'a RotationExportOwner,
    raw: CheckedOwner<RawRotation>,
}

impl QuarantinedRotation<'_> {
    pub fn release(self) -> Result<(), IdentityError> {
        // Retain the original export borrow until checked candidate cleanup ends.
        let _original = self.export;
        self.raw.release().map_err(Into::into)
    }

    pub fn into_preparation(self) -> Result<RotationPreparation, IdentityError> {
        self.release()?;
        Err(IdentityError::MissingTrustedClientExportEpoch)
    }
}

#[cfg(test)]
#[path = "../tests/identity/unit.rs"]
mod tests;
