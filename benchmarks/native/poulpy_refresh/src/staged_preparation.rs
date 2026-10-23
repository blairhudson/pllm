//! Real raw-import/preparation ownership for the pinned staged design.
//!
//! This module allocates no key/context/ciphertext. It adopts existing original
//! owners. Import parsing does NOT authenticate immutable key material: parsed
//! imports deliberately have no conversion into preparation/provider material.
//! The preparation transaction uses the real pinned public factory, but its
//! entry stays blocked by the unclosed qualification admission gate.
//! Arbitrary raw-key adoption now also rejects: the pinned specialized owner
//! exposes a full borrowed span, but not original allocation capacity/alignment.
//! No trusted-client provenance, secure sampling seam, or HE allocation is added.

use std::{
    alloc::Layout,
    io::{self, Read},
};

use poulpy_core::layouts::{
    GGLWEToBackendRef, GLWEAutomorphismKey, GLWEAutomorphismKeyPreparedFactory, GetDegree,
    GetGaloisElement,
};
use poulpy_hal::{
    AlignedBuf,
    layouts::{Backend, DataView, Module, ReaderFrom, ScratchOwned},
};

use crate::staged_storage::{
    CheckedOwner, Cpu, RefillGuard, Rotation, StorageError, admit_encrypted_execution, healthy,
    high_key_layout, quote_rotation_preparation, require_original_layout,
};

pub type RawRotation = GLWEAutomorphismKey<AlignedBuf, <Cpu as Backend>::ZnxWord>;
pub type PreparationScratch = ScratchOwned<Cpu>;
const RAW_BYTES: usize = 242_221_056;
const RAW_FRAMING_MAX: u64 = 125;
const PREPARATION_BYTES: usize = 2 << 20;

fn raw_layout(key: &RawRotation) -> Result<Layout, StorageError> {
    high_key_layout(key)?;
    let bytes = Cpu::bytes_of_mat_znx(65536, 7, 1, 2, 33);
    if bytes != RAW_BYTES {
        return Err(StorageError::Shape);
    }
    let expected = Layout::from_size_align(bytes, 64).map_err(|_| StorageError::Shape)?;
    let backend = <RawRotation as GGLWEToBackendRef<Cpu>>::to_backend_ref(key);
    let span = Cpu::len_bytes_ref(DataView::data(backend.data()));
    raw_backing_layout(span, expected)
}

/// A full backend borrow exposes the actual visible backing span, NOT the
/// AlignedBuf's original capacity or alignment. MatZnx::read_from may shorten
/// the logical shape without shortening even that span. RawRotation does not
/// encode its allocating backend and has no public original-buffer accessor;
/// an arbitrary adopted raw owner therefore cannot establish an exact Layout.
/// Do not turn a matching span or aligned pointer into a capacity certificate.
/// A future private factory seam must retain the real owning AlignedBuf/Layout,
/// or a reviewed upstream accessor must expose it. No such authority exists here.
pub(crate) fn raw_backing_layout(span: usize, expected: Layout) -> Result<Layout, StorageError> {
    require_original_layout(span, expected)
}

impl CheckedOwner<RawRotation> {
    pub fn raw_rotation(key: RawRotation) -> Result<Self, StorageError> {
        let layout = raw_layout(&key)?;
        Self::new(key, layout)
    }
}

impl CheckedOwner<PreparationScratch> {
    pub fn preparation_scratch(scratch: PreparationScratch) -> Result<Self, StorageError> {
        // data is the original AlignedBuf; retain its allocation capacity even
        // if the visible scratch length has been shortened.
        let layout = Layout::from_size_align(scratch.data.capacity(), scratch.data.align())
            .map_err(|_| StorageError::Shape)?;
        if scratch.data.len() < PREPARATION_BYTES {
            return Err(StorageError::Capacity);
        }
        if scratch.data.align() < Cpu::SCRATCH_ALIGN {
            return Err(StorageError::Shape);
        }
        Self::new(scratch, layout)
    }
}

/// Parse exactly one bounded record into existing storage, without a whole-file
/// Vec. The enclosing owner permanently invalidates itself on any parse failure.
/// The external deadline/kernel-I/O bound remains a separate admission gate.
pub(crate) fn read_record<R: Read>(
    reader: &mut R,
    bytes: u64,
    parse: impl FnOnce(&mut std::io::Take<&mut R>) -> io::Result<()>,
) -> io::Result<()> {
    let mut bounded = reader.take(bytes);
    parse(&mut bounded)?;
    if bounded.limit() != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record length mismatch",
        ));
    }
    let mut extra = [0u8; 1];
    if bounded.into_inner().read(&mut extra)? != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing key bytes",
        ));
    }
    Ok(())
}

/// Untrusted import owns one full seven-row raw key and its original Layout.
/// Metadata/shape/trailing checks are real, but no hash or dense-key identity is
/// invented. There is deliberately no API exposing this raw key to preparation.
/// Adoption is closed until the original opaque key allocation can be proved.
pub struct RotationImport {
    raw: CheckedOwner<RawRotation>,
    expected_p: i64,
    attempted: bool,
    failed: bool,
}

impl RotationImport {
    pub fn adopt(raw: RawRotation, expected_p: i64) -> Result<Self, StorageError> {
        Ok(Self {
            raw: CheckedOwner::raw_rotation(raw)?,
            expected_p,
            attempted: false,
            failed: false,
        })
    }

    pub fn read_from<R: Read>(
        &mut self,
        reader: &mut R,
        wire_bytes: u64,
    ) -> Result<(), StorageError> {
        if self.attempted || self.failed || !healthy(crate::ALLOCATOR.snapshot()) {
            self.failed = true;
            return Err(StorageError::AllocatorRejected);
        }
        self.attempted = true;
        // Failure is sticky even if the parser partially overwrites metadata.
        self.failed = true;
        if wire_bytes < RAW_BYTES as u64 || wire_bytes > RAW_BYTES as u64 + RAW_FRAMING_MAX {
            return Err(StorageError::Capacity);
        }
        let raw = self.raw.borrow_mut();
        read_record(reader, wire_bytes, |record| raw.read_from(record))
            .map_err(|_| StorageError::Shape)?;
        raw_layout(raw)?;
        if raw.p() != self.expected_p || !healthy(crate::ALLOCATOR.snapshot()) {
            return Err(StorageError::Shape);
        }
        self.failed = false;
        Ok(())
    }

    /// Release remains possible after malformed metadata, using the saved
    /// ORIGINAL Layout rather than trusting fields overwritten by the parser.
    pub fn release(self) -> Result<(), StorageError> {
        self.raw.release()
    }
}

/// Same-role preparation transaction. Old phase views must end before taking
/// the destination out of RotationPhase. No prepared views escape this owner.
pub struct RotationPreparation {
    raw: CheckedOwner<RawRotation>,
    destination: CheckedOwner<Rotation>,
    scratch: CheckedOwner<PreparationScratch>,
    expected_p: i64,
}

/// The originating phase remains exclusively borrowed until checked commit.
/// Field order burns the guard before dropping loading owners on abandonment.
#[must_use = "checked commit is required; dropping burns the originating phase"]
pub struct RotationRefillPreparation<'a> {
    refill: RefillGuard<'a, Rotation>,
    loading: RotationPreparation,
}

impl RotationRefillPreparation<'_> {
    pub fn prepare_and_commit(self, module: &Module<Cpu>) -> Result<(), StorageError> {
        let Self { refill, loading } = self;
        let ready = loading.prepare_and_release(module)?;
        // No ready owner escapes between checked loading release and insertion.
        refill.commit_rotation(ready)
    }
}

impl RotationPreparation {
    /// Raw material must originate from trusted local client generation, NOT
    /// RotationImport. Authenticated disk-store promotion remains unimplemented.
    pub fn adopt_local_raw<'a>(
        raw: RawRotation,
        mut refill: RefillGuard<'a, Rotation>,
        scratch: PreparationScratch,
        expected_p: i64,
    ) -> Result<RotationRefillPreparation<'a>, StorageError> {
        if raw.p() != expected_p {
            return Err(StorageError::Shape);
        }
        // Keep the destination inside its checked abort guard until BOTH loading
        // owners validate. In particular an unprovable raw adoption cannot detach
        // and silently drop the originating destination before checked cleanup.
        let raw = CheckedOwner::raw_rotation(raw)?;
        let scratch = match CheckedOwner::preparation_scratch(scratch) {
            Ok(scratch) => scratch,
            Err(error) => {
                raw.release()?;
                return Err(error);
            }
        };
        let destination = refill.take_destination()?;
        let loading = Self {
            raw,
            destination,
            scratch,
            expected_p,
        };
        Ok(RotationRefillPreparation { refill, loading })
    }

    /// Real pinned arithmetic, unreachable until complete qualification bounds
    /// are admitted. No test override, forged admission token, or fallback.
    fn prepare_and_release(
        mut self,
        module: &Module<Cpu>,
    ) -> Result<CheckedOwner<Rotation>, StorageError> {
        admit_encrypted_execution()?;
        // This is the first supported seam over EXISTING typed operands, not a
        // preconstruction bound on Module/raw/output/scratch. The public pure
        // preconstruction check also stays closed; callers must not allocate a
        // graph and mistake this later gate for permission to have built it.
        {
            let quote = quote_rotation_preparation(
                module,
                &self.raw,
                &self.destination,
                &self.scratch,
                None,
            )?;
            quote.check_workspace_capacity(
                crate::bounded_allocator::CHARGE_CEILING,
                crate::bounded_allocator::MAX_MAPPINGS,
            )?;
        }
        if !healthy(crate::ALLOCATOR.snapshot()) {
            return Err(StorageError::AllocatorRejected);
        }
        if module.ring_degree().as_usize() != 65536 {
            return Err(StorageError::Shape);
        }
        let required = module.glwe_automorphism_key_prepare_tmp_bytes(self.raw.borrow());
        if required != PREPARATION_BYTES {
            return Err(StorageError::Shape);
        }
        {
            let mut arena = self.scratch.borrow_mut().arena();
            if arena.available() < required {
                return Err(StorageError::Capacity);
            }
            module.glwe_automorphism_key_prepare(
                self.destination.borrow_mut(),
                self.raw.borrow(),
                &mut arena,
            );
        }
        if self.destination.borrow().p() != self.expected_p || !healthy(crate::ALLOCATOR.snapshot())
        {
            return Err(StorageError::AllocatorRejected);
        }
        // Only return the prepared owner after both real loading owners have
        // dropped and their mapping release checks passed. VMP private Vecs use
        // the same global allocator; a failed unmap already poisoned the run.
        self.raw.release()?;
        self.scratch.release()?;
        Ok(self.destination)
    }
}
