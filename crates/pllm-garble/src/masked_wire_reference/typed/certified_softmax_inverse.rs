//! D36 fixed inverse -> ORIGINAL native lookup, no duplicate table owner/cache.
//!
//! Mount under activation exactly like certified_exp_lookup. Its containing root
//! admits the complete original-counter aggregate BEFORE quote/prepare/hash and
//! retains Verified/source/original charge through last output/frame. Numeric
//! quotes/Verified do NOT grant that admission, coordinate range or source power.
//! `issue` consumes an exact already-live original native lease and preserves all
//! caller context/numeric fields. B owns this outer admission/custody invariant.
//! No caller table/certificate/digest is accepted; no R<=4096 claim is made.

use super::{lookup_resource_calculation_bytes, resource_metadata_sum};
use crate::fixed_inverse_table::{self as tables, Definition, Verified};
use crate::masked_wire_reference::source_lookup;
use crate::shared_allocation::{AllocationContext, Error as NativeError, LeaseFor};
use sha2::{Digest, Sha256};
use std::mem::size_of;

#[derive(Debug, Eq, PartialEq)]
pub(in crate::masked_wire_reference::typed) enum Error {
    Verification(tables::Error),
    Identity,
    Native(NativeError),
}

/// Additional logical preparation/control census plus unchanged native costs.
/// Dependency-private SHA callframes/physical stack/RSS are NOT measured here;
/// outer existing bounded hash-work allowance covers public prep and pin rehash,
/// in addition to these named hash states/word/output/control slots. Caller quote
/// records/source/input originals remain additional, not credited as retired.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::masked_wire_reference::typed) struct Quote {
    pub(crate) verification: tables::Quote,
    pub(crate) lookup: source_lookup::Resources,
    pub(crate) native_query_phase_bytes: usize,
    pub(crate) preparation_peak_bytes: usize,
    pub(crate) issuance_adapter_bytes: usize,
    pub(crate) native_reservation_bytes: usize,
    /// Informational SUBSET already included in native_reservation_bytes.
    pub(crate) native_table_payload_bytes: usize,
    pub(crate) preparation_hash_bytes: usize,
    pub(crate) preparation_hash_blocks: usize,
    pub(crate) issuance_pin_hash_bytes_per_lookup: usize,
    pub(crate) issuance_pin_hash_blocks_per_lookup: usize,
}

#[allow(dead_code)]
struct IdentitySlots {
    definition: Definition,
    hash: Sha256,
    final_hash: sha2::digest::Output<Sha256>,
    digest: [u8; 32],
    word: [u8; 8],
    returned: Result<[u8; 32], Error>,
}

#[allow(dead_code)]
struct PreparationSlots {
    verification_result: Result<Verified, tables::Error>,
    identity_result: Result<[u8; 32], Error>,
    digest: [u8; 32],
    returned: Result<(Verified, [u8; 32]), Error>,
}

#[allow(dead_code)]
struct IssuanceSlots<C: AllocationContext> {
    context: source_lookup::Context,
    spec: source_lookup::Spec,
    identity_result: Result<[u8; 32], Error>,
    native: Result<
        (
            source_lookup::OutputMasksFor<C>,
            [source_lookup::PartyFor<C>; 2],
        ),
        NativeError,
    >,
    returned: Result<
        (
            source_lookup::OutputMasksFor<C>,
            [source_lookup::PartyFor<C>; 2],
        ),
        Error,
    >,
}

fn spec(lanes: usize) -> source_lookup::Spec {
    source_lookup::Spec {
        index_bits: 8,
        output_bits: 50,
        lanes,
        boundary: source_lookup::Boundary::Reconstructed,
    }
}

/// Query is public arithmetic, not admission. B includes all phases and native
/// copies/work in its complete global quote before proof/hash/entropy/construction.
pub(in crate::masked_wire_reference::typed) fn quote<C: AllocationContext>(
    lanes: usize,
) -> Result<Quote, Error> {
    let verification = tables::quote().map_err(Error::Verification)?;
    let lookup = source_lookup::resources_for_context::<C>(spec(lanes)).map_err(Error::Native)?;
    let sum = |parts: &[usize]| resource_metadata_sum(parts).map_err(Error::Native);
    let verification_phase = sum(&[
        verification.validation_slots_bytes,
        size_of::<PreparationSlots>(),
    ])?;
    let hash_phase = sum(&[size_of::<PreparationSlots>(), size_of::<IdentitySlots>()])?;
    Ok(Quote {
        native_query_phase_bytes: sum(&[
            size_of::<tables::Quote>(),
            size_of::<source_lookup::Spec>(),
            lookup_resource_calculation_bytes().map_err(Error::Native)?,
        ])?,
        preparation_peak_bytes: verification_phase.max(hash_phase),
        issuance_adapter_bytes: sum(&[size_of::<IssuanceSlots<C>>(), size_of::<IdentitySlots>()])?,
        native_reservation_bytes: lookup.combined_reservation_bytes,
        native_table_payload_bytes: verification
            .selected_literal_bytes
            .checked_mul(2)
            .ok_or(Error::Native(NativeError::Resource))?,
        preparation_hash_bytes: verification.selected_literal_bytes,
        preparation_hash_blocks: 33, // 2048-byte SHA256 data + one padding block.
        issuance_pin_hash_bytes_per_lookup: verification.selected_literal_bytes,
        issuance_pin_hash_blocks_per_lookup: 33,
        verification,
        lookup,
    })
}

fn identity(checked: &Verified) -> Result<[u8; 32], Error> {
    let d = tables::definition();
    if d.entries != 256
        || d.numerator != 1 << 18
        || d.sentinel != 0
        || (
            d.coordinate_bits,
            d.coordinate_fractional_bits,
            d.denominator_fractional_bits,
        ) != (8, 0, 6)
        || (d.payload_bits, d.payload_fractional_bits) != (50, 12)
        || d.global_payload_max != 262144
    {
        return Err(Error::Identity);
    }
    let mut hash = Sha256::new();
    for value in checked.values() {
        hash.update(value.to_le_bytes());
    }
    let digest: [u8; 32] = hash.finalize().into();
    if digest != d.sha256_u64le {
        return Err(Error::Identity);
    }
    Ok(digest)
}

// Production failure boundary. No hash/value access before all 256 checks pass.
fn after_verification(
    result: Result<Verified, tables::Error>,
) -> Result<(Verified, [u8; 32]), Error> {
    let checked = result.map_err(Error::Verification)?;
    let digest = identity(&checked)?;
    Ok((checked, digest))
}

/// All 256 checks before ANY table hash/private operation. Caller must already
/// retain its genuine original aggregate lease; a numeric quote is insufficient.
pub(in crate::masked_wire_reference::typed) fn prepare() -> Result<(Verified, [u8; 32]), Error> {
    after_verification(tables::verify())
}

/// Actual unchanged native lookup owners/copies. This helper does not replace
/// numeric_contract with a table digest or manufacture coordinate/source proofs.
pub(in crate::masked_wire_reference::typed) fn issue<C: AllocationContext>(
    checked: &Verified,
    context: source_lookup::Context,
    input_masks: &[u64],
    reservation: LeaseFor<C>,
    fill: impl FnMut(&mut [u8]) -> Result<(), NativeError>,
) -> Result<
    (
        source_lookup::OutputMasksFor<C>,
        [source_lookup::PartyFor<C>; 2],
    ),
    Error,
> {
    let _ = identity(checked)?;
    source_lookup::issue_reserved_with(
        spec(input_masks.len()),
        context,
        checked.values(),
        input_masks,
        reservation,
        fill,
    )
    .map_err(Error::Native)
}

#[cfg(test)]
#[path = "certified_softmax_inverse_tests.rs"]
mod tests;
