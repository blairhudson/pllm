//! Private hookup of certified public exp bytes to the existing source lookup.
//!
//! `fixed_exp_tables::Verified` is already the immutable selected-table owner:
//! its private constructor checks ALL 512 floors before exposing either static
//! array. Do not wrap/copy/cache it in another public-table owner. The unchanged
//! source lookup owns/charges the two actual party copies for each lookup.
//!
//! Mount as a child of `typed::activation` to reuse its existing method-neutral
//! lookup calculation quote without copying that machinery:
//! `#[path = "certified_exp_lookup.rs"] pub(super) mod certified_exp_lookup;`
//!
//! The containing region admits these public phases BEFORE `prepare`, then
//! completes its source/declaration/root preflight before ANY operation entropy
//! or private construction. It passes an original exact native lease to `issue`.
//! Numbers returned here are NOT leases, source/range/custody certificates, or
//! complete region quotes. Context/mask provenance and fixed uses remain the
//! containing owner's responsibility. Nothing here exposes an SDK adapter.

use super::{lookup_resource_calculation_bytes, resource_metadata_sum};
use crate::exp_floor::Workspace;
use crate::fixed_exp_tables::{self as tables, Definition, Table, Verified};
use crate::masked_wire_reference::source_lookup;
use crate::shared_allocation::{AllocationContext, Error as NativeError, LeaseFor};
use sha2::{Digest, Sha256};
use std::mem::size_of;

#[derive(Debug, Eq, PartialEq)]
pub(in crate::masked_wire_reference::typed) enum Error {
    Floor(tables::Error),
    Identity(Table),
    Native(NativeError),
}

/// Additional public preparation/adapter census plus unchanged native costs.
/// Scoped logical capacity, not machine stack/RSS or an aggregate nExp quote.
/// Caller-retained quote records, source/context originals and input storage
/// remain additional. These fields count hash objects/words/results, NOT
/// dependency-private SHA scratch. The containing owner's existing bounded hash
/// allowance must cover public preparation too, not just native issuance.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::masked_wire_reference::typed) struct Quote {
    pub(crate) floors: tables::Quote,
    pub(crate) lookup: source_lookup::Resources,
    pub(crate) native_query_phase_bytes: usize,
    pub(crate) preparation_peak_bytes: usize,
    pub(crate) issuance_adapter_bytes: usize,
    pub(crate) paired_native_reservation_bytes: usize,
    /// Informational subset ALREADY included in paired_native_reservation_bytes.
    pub(crate) native_table_payload_bytes_per_pair: usize,
    pub(crate) preparation_hash_bytes: usize,
    pub(crate) issuance_pin_hash_bytes_per_lookup: usize,
}

#[allow(dead_code)]
struct IdentitySlots {
    table: Table,
    definition: Definition,
    hash: Sha256,
    final_hash: sha2::digest::Output<Sha256>,
    digest: [u8; 32],
    word: [u8; 8],
    returned: Result<[u8; 32], Error>,
}

#[allow(dead_code)]
struct PreparationSlots {
    floor_result: Result<Verified, tables::Error>,
    returned: Result<(Verified, [[u8; 32]; 2]), Error>,
    identities: [[u8; 32]; 2],
    identity_result: Result<[[u8; 32]; 2], Error>,
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

/// Public, allocation/hash/entropy-free query. The existing native calculation
/// helper quotes its actual nested query slots; no D27/matrix byte constant.
pub(in crate::masked_wire_reference::typed) fn quote<C: AllocationContext>(
    lanes: usize,
) -> Result<Quote, Error> {
    let floors = tables::quote().map_err(|e| Error::Floor(tables::Error::Floor(e)))?;
    let lookup = source_lookup::resources_for_context::<C>(spec(lanes)).map_err(Error::Native)?;
    let sum = |parts: &[usize]| resource_metadata_sum(parts).map_err(Error::Native);
    let floor_phase = sum(&[
        floors.proof.workspace_and_call_bytes,
        floors.validation_slots_bytes,
        size_of::<PreparationSlots>(),
    ])?;
    let hash_phase = sum(&[
        size_of::<Workspace>(),
        size_of::<PreparationSlots>(),
        size_of::<IdentitySlots>(),
    ])?;
    let table_bytes = floors.selected_literal_bytes / 2;
    Ok(Quote {
        native_query_phase_bytes: sum(&[
            size_of::<tables::Quote>(),
            size_of::<source_lookup::Spec>(),
            lookup_resource_calculation_bytes().map_err(Error::Native)?,
        ])?,
        preparation_peak_bytes: floor_phase.max(hash_phase),
        issuance_adapter_bytes: sum(&[size_of::<IssuanceSlots<C>>(), size_of::<IdentitySlots>()])?,
        paired_native_reservation_bytes: lookup
            .combined_reservation_bytes
            .checked_mul(2)
            .ok_or(Error::Native(NativeError::Resource))?,
        native_table_payload_bytes_per_pair: table_bytes
            .checked_mul(4)
            .ok_or(Error::Native(NativeError::Resource))?,
        preparation_hash_bytes: floors.selected_literal_bytes,
        issuance_pin_hash_bytes_per_lookup: table_bytes,
        floors,
        lookup,
    })
}

fn matches_hex(digest: &[u8; 32], expected: &str) -> bool {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = expected.as_bytes();
    bytes.len() == 64
        && digest
            .iter()
            .zip(bytes.chunks_exact(2))
            .all(|(&byte, pair)| {
                pair[0] == HEX[usize::from(byte >> 4)] && pair[1] == HEX[usize::from(byte & 15)]
            })
}

fn identity(checked: &Verified, table: Table) -> Result<[u8; 32], Error> {
    let definition = table.definition();
    let denominator = match table {
        Table::LowByte => 12,
        Table::HighByte => 4,
    };
    if definition.entries != 256
        || definition.argument_denominator_bits != denominator
        || definition.payload_scale != 4096
        || (
            definition.coordinate_bits,
            definition.coordinate_fractional_bits,
        ) != (8, 0)
        || (definition.payload_bits, definition.payload_fractional_bits) != (50, 12)
    {
        return Err(Error::Identity(table));
    }
    let mut hash = Sha256::new();
    for value in checked.values(table) {
        hash.update(value.to_le_bytes());
    }
    let digest: [u8; 32] = hash.finalize().into();
    if !matches_hex(&digest, definition.sha256_u64le) {
        return Err(Error::Identity(table));
    }
    Ok(digest)
}

/// Actual byte digests, ordered E0/E1. No caller-provided certificate/hash is
/// accepted. Construct canonical region records only AFTER this succeeds.
fn identities(checked: &Verified) -> Result<[[u8; 32]; 2], Error> {
    Ok([
        identity(checked, Table::LowByte)?,
        identity(checked, Table::HighByte)?,
    ])
}

// Keep the failure boundary explicit: an arithmetic failure cannot enter either
// hash or native construction. Tests inject ONLY rejection, never valid evidence.
fn after_floors(
    result: Result<Verified, tables::Error>,
) -> Result<(Verified, [[u8; 32]; 2]), Error> {
    let checked = result.map_err(Error::Floor)?;
    let identities = identities(&checked)?;
    Ok((checked, identities))
}

/// No allocator, Budget, entropy callback or private input enters this function.
/// Every call checks all 512 Rust floors before exposing/hashing either table.
/// Returns the SAME Verified owner plus observed E0/E1 digests, not another owner.
pub(in crate::masked_wire_reference::typed) fn prepare(
    work: &mut Workspace,
) -> Result<(Verified, [[u8; 32]; 2]), Error> {
    after_floors(tables::verify(work))
}

/// Same original native table owners, lease and caller context. In particular,
/// do NOT replace numeric_contract with a digest supplied as arithmetic evidence:
/// B's source/coordinate/template commitment must remain inside that context.
/// Fresh source-lookup entropy/material is generated only after the pin check.
pub(in crate::masked_wire_reference::typed) fn issue<C: AllocationContext>(
    checked: &Verified,
    table: Table,
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
    let _ = identity(checked, table)?;
    let native = source_lookup::issue_reserved_with(
        spec(input_masks.len()),
        context,
        checked.values(table),
        input_masks,
        reservation,
        fill,
    );
    native.map_err(Error::Native)
}

#[cfg(test)]
#[path = "certified_exp_lookup_tests.rs"]
mod tests;
