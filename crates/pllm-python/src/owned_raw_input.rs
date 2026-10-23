//! Private, model-neutral ingress for an immutable, original-counter byte owner.
//!
//! Only an EXACT pre-existing Python `bytes` is borrowed. Its CPython backing,
//! creation, allocator/GC/error/traceback machinery and dependency call frames
//! remain EXTERNAL: this is not admission of that earlier allocation or a full
//! CPython physical peak. No exporter, `__bytes__`, integer conversion, serializer,
//! Python exception construction or caller callback runs with the working charge.
//!
//! One existing OwnedBudget reserves controls, then backing and the shared
//! OwnedCharged inline payload BEFORE the buffer factory. A consuming scratch
//! freeze moves the same backing into OwnedCharged<Vec<u8>>; it never refunds,
//! clones, rebuilds or substitutes the accounting root. The named conservative
//! ingress inventory stays with those bytes through the existing parser/input
//! custody spine. No authority, registration or native/Python contract gate is
//! enabled here. Callers must compose this copy work with subsequent parser work.
//!
//! Standard Box allocation can abort on allocator OOM. Quotation admits finite
//! logical backing/control slots, not allocator overhead or recoverable OOM.
//! Failed scratch is erased before its charge retires. Successful source bytes
//! use the existing plain Vec owner and make NO secure-erasure promise.

#![allow(dead_code)] // Rust-only prerequisite; deliberately no Python registration.

use pllm_garble::shared_allocation::{
    Budget, Error as AllocationError, OwnedBudget, OwnedCharged, OwnedExternalCharge,
    MAX_ALLOCATION_BYTES,
};
use pyo3::{prelude::*, types::PyBytes, DowncastError};
use std::{mem::size_of, mem::size_of_val, mem::MaybeUninit, slice};
use zeroize::Zeroize;

const COPY_CHUNK_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Limits {
    pub max_bytes: usize,
    /// Logical byte initialization/copy/failed-scratch erasure plus chunk visits.
    /// Does not authorize or pay later parser/classifier/native work.
    pub max_work: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Error {
    ExactBytes,
    ByteLimit,
    WorkLimit,
    Overflow,
    Native(AllocationError),
}

impl From<AllocationError> for Error {
    fn from(error: AllocationError) -> Self {
        Self::Native(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Resources {
    pub input_bytes: usize,
    pub controls_bytes: usize,
    /// External payload reserved before invoking shared try_build.
    pub payload_bytes: usize,
    /// Exact additional counter charge, including shared inline owner metadata.
    pub reserved_bytes: usize,
    pub work_bound: u128,
}

// Named logical slots, conservatively summed even when phases are sequential.
// The final OwnedCharged inline payload/guard is ALSO added by the shared spine;
// these extra move/result slots do not grant any credit against its reservation.
// Scalar quote arithmetic, compiler/ABI call frames and dependency internals are
// not measured here. Source CPython objects/pointees are explicitly external.
const CONTROL_TERMS: &[usize] = &[
    size_of::<Limits>(),
    2 * size_of::<Resources>(),
    size_of::<Result<Resources, Error>>(),
    size_of::<(&OwnedBudget, &Bound<'static, PyAny>, Limits)>(),
    size_of::<(&Bound<'static, PyAny>, &Bound<'static, PyBytes>)>(),
    size_of::<Bound<'static, PyAny>>(),
    size_of::<Bound<'static, PyBytes>>(),
    size_of::<Result<&Bound<'static, PyBytes>, DowncastError<'static, 'static>>>(),
    size_of::<(&Budget, &[u8])>(),
    2 * size_of::<OwnedExternalCharge>(),
    size_of::<Result<OwnedExternalCharge, AllocationError>>(),
    2 * size_of::<OwnedCharged<Vec<u8>>>(),
    size_of::<Result<OwnedCharged<Vec<u8>>, AllocationError>>(),
    2 * size_of::<FrozenInput>(),
    2 * size_of::<Result<FrozenInput, Error>>(),
    size_of::<Writer<'static>>(),
    size_of::<Box<[MaybeUninit<u8>]>>(),
    size_of::<Box<[u8]>>(),
    size_of::<Vec<u8>>(),
    size_of::<slice::IterMut<'static, MaybeUninit<u8>>>(),
    size_of::<Option<&mut MaybeUninit<u8>>>(),
    size_of::<&mut MaybeUninit<u8>>(),
    size_of::<&mut u8>(),
    size_of::<slice::Chunks<'static, u8>>(),
    size_of::<Option<&[u8]>>(),
    size_of::<(&mut Writer<'static>, &[u8])>(),
    size_of::<Result<(), AllocationError>>(),
    // freeze result, closed producer result, shared try_build's build result.
    3 * size_of::<Result<Vec<u8>, AllocationError>>(),
    size_of::<&mut Vec<u8>>(),
    // write end, source/expected lengths, cursor and chunk-visit arithmetic slots.
    size_of::<[usize; 8]>(),
];

fn controls_bytes() -> Result<usize, Error> {
    let slots = CONTROL_TERMS
        .iter()
        .try_fold(0usize, |sum, bytes| sum.checked_add(*bytes))
        .ok_or(Error::Overflow)?;
    // Actual CLOSED producer type, not a surrogate tuple/unknown caller closure.
    // Include quote-time instance, live producer and its moved try_build argument.
    let producer_slots = size_of_val(&producer(&[]))
        .checked_mul(3)
        .ok_or(Error::Overflow)?;
    slots.checked_add(producer_slots).ok_or(Error::Overflow)
}

impl Resources {
    /// Pure native quote; no Budget, Python conversion or buffer factory.
    pub(crate) fn checked(input_bytes: usize, limits: Limits) -> Result<Self, Error> {
        if limits.max_bytes > MAX_ALLOCATION_BYTES || input_bytes > limits.max_bytes {
            return Err(Error::ByteLimit);
        }
        let chunks = (input_bytes / COPY_CHUNK_BYTES)
            .checked_add(usize::from(input_bytes % COPY_CHUNK_BYTES != 0))
            .ok_or(Error::Overflow)?;
        // Entire zero initialization, at most N copied bytes, and up to N bytes
        // erased on failure. Chunk visits are independently bounded. This entire
        // per-call bound must be composed with later work, never lent/refunded
        // when success does not need the failure erasure allowance.
        let work_bound = (input_bytes as u128)
            .checked_mul(3)
            .and_then(|work| work.checked_add(chunks as u128))
            .ok_or(Error::Overflow)?;
        if work_bound > limits.max_work {
            return Err(Error::WorkLimit);
        }
        let controls_bytes = controls_bytes()?;
        let payload_bytes = controls_bytes
            .checked_add(input_bytes)
            .ok_or(Error::Overflow)?;
        let reserved_bytes = payload_bytes
            .checked_add(size_of::<OwnedCharged<Vec<u8>>>())
            .ok_or(Error::Overflow)?;
        Ok(Self {
            input_bytes,
            controls_bytes,
            payload_bytes,
            reserved_bytes,
            work_bound,
        })
    }
}

/// No mutation, Clone, guard detachment, Python reference or budget adoption API.
pub(crate) struct FrozenInput {
    original: OwnedCharged<Vec<u8>>,
    copy_work_bound: u128,
}

impl FrozenInput {
    pub(crate) fn bytes(&self) -> &[u8] {
        self.original.payload()
    }

    pub(crate) fn budget(&self) -> &Budget {
        self.original.budget()
    }

    /// Validated source-bound per-call copy bound, not downstream work credit.
    pub(crate) fn copy_work_bound(&self) -> u128 {
        self.copy_work_bound
    }

    /// Moves the SAME original owner into an existing consuming parser/spine.
    /// Its complete ingress charge remains resident, not a detached side guard.
    /// SIGMA prepare_from_plan_bytes retains it in semantic.raw, then opaque
    /// OwnedSigmaInputRetention; existing role/frame handles retain that residue.
    pub(crate) fn into_original(self) -> OwnedCharged<Vec<u8>> {
        self.original
    }
}

/// Typed Rust limits only: an eventual Python entry must independently qualify
/// argument packaging/conversion BEFORE entering any live protocol ownership.
/// No pymethod/pyfunction registration or PyErr conversion is provided here.
pub(crate) fn from_exact_bytes(
    root: &OwnedBudget,
    value: &Bound<'_, PyAny>,
    limits: Limits,
) -> Result<FrozenInput, Error> {
    // Even borrow/cast/quotation controls have a real original-counter guard.
    // Failure here precedes Python inspection and the first buffer factory.
    let mut charge = root.charge_external(controls_bytes()?)?;
    let bytes = value
        .cast_exact::<PyBytes>()
        .map_err(|_| Error::ExactBytes)?;
    let source = bytes.as_bytes(); // builtin immutable borrow, never an exporter.
    let quote = Resources::checked(source.len(), limits)?;
    charge.resize_payload(quote.payload_bytes)?;
    // Shared try_build adds inline payload/padding BEFORE invoking producer.
    // Keep the ingress controls charged after construction as well, including
    // when original moves into parser/input/session custody and roots retire.
    let original = charge.try_build(producer(source))?;
    Ok(FrozenInput {
        original,
        copy_work_bound: quote.work_bound,
    })
}

fn producer(source: &[u8]) -> impl FnOnce(&Budget) -> Result<Vec<u8>, AllocationError> + '_ {
    move |budget| {
        let mut writer = Writer::new(source.len(), budget);
        for chunk in source.chunks(COPY_CHUNK_BYTES) {
            writer.append(chunk)?;
        }
        writer.freeze()
    }
}

/// This mutable type never escapes the fully charged factory. All writes append
/// a nonempty bounded slice; no mutable byte view or arbitrary producer callback
/// is exposed. Failed/incomplete state is consumed and erased before guard drop.
struct Writer<'a> {
    bytes: Vec<u8>,
    written: usize,
    budget: &'a Budget,
}

impl<'a> Writer<'a> {
    fn new(length: usize, budget: &'a Budget) -> Self {
        #[cfg(test)]
        tests::factory_entry(budget);

        // Exact logical allocation, avoiding Vec's at-least-capacity request.
        // Box's allocator overhead/physical usable size remain excluded. No
        // caller callback can interrupt the complete initialization loop.
        let mut uninitialized = Box::<[u8]>::new_uninit_slice(length);
        for slot in uninitialized.iter_mut() {
            slot.write(0);
        }
        // SAFETY: every element of this exact slice was initialized to 0 above;
        // no early return, fallible operation or caller code intervenes. Box ->
        // Vec transfers the exact same backing without cloning or allocation.
        let bytes = unsafe { uninitialized.assume_init() }.into_vec();
        Self {
            bytes,
            written: 0,
            budget,
        }
    }

    fn append(&mut self, source: &[u8]) -> Result<(), AllocationError> {
        let end = self
            .written
            .checked_add(source.len())
            .filter(|end| *end <= self.bytes.len())
            .ok_or(AllocationError::Resource)?;
        if source.is_empty() {
            return Err(AllocationError::Resource);
        }
        self.bytes[self.written..end].copy_from_slice(source);
        self.written = end;
        Ok(())
    }

    fn freeze(mut self) -> Result<Vec<u8>, AllocationError> {
        if self.written != self.bytes.len() {
            return Err(AllocationError::Resource);
        }
        Ok(std::mem::take(&mut self.bytes))
    }
}

impl Drop for Writer<'_> {
    fn drop(&mut self) {
        self.bytes.as_mut_slice().zeroize();
        #[cfg(test)]
        tests::scratch_retired(self.budget, &self.bytes);
    }
}

#[cfg(test)]
#[path = "owned_raw_input_tests.rs"]
mod tests;

#[cfg(any(test, doctest))]
#[path = "owned_raw_input_independent_tests.rs"]
mod independent_regressions;
