//! D29 prerequisite declarations. No region execution/issuance constructor is
//! exported until the canonical records, table floors and full phase inventory
//! are reviewed. This is NOT an nExp capability, inherited adapter, or admission
//! of the eleven-exchange region. The independently implemented private product
//! owner is usable by that future region, not by public post-D22 Multiply.

use super::super::elementwise_product;
use super::*;
use pllm_core::edge_contract::ConsumerUse;

const REGION_DOMAIN: &[u8] = b"pllm.sigma.nexp-region.v1";
const COORDINATE_DOMAIN: &[u8] = b"pllm.sigma.nexp-coordinate.v1";
const EQUATION: &[u8] = b"floor(E0[min(X,65535)%256]*E1[min(X,65535)/256]/4096)";

// Layouts below are not a serialized struct layout or public capability. The
// private region's explicit versioned wire order (including paper/source pins,
// intervals, all fourteen edges and eighteen uses) is in nexp_region_records.md.
// That record and its independent vectors require review before hash/issuance;
// do not infer a shorter wire order from the fields of this metadata DTO.
#[allow(dead_code)]
struct Declaration {
    source: SourceIdentity,
    input: EdgeDescriptor,
    output: EdgeDescriptor,
    operation: [u8; 32],
    next: [u8; 32],
    table_hashes: [[u8; 32]; 2],
    coordinates: [CoordinateDeclaration; 2],
    uses: [ConsumerUse; 18],
}
// Canonical order: domain, declaration.v1, mapping tag (low8.v1/high8-tr16-8.v1), parent edge,
// coordinate edge, target table hash, u64-LE first/lane count, unsigned(8,0),
// u16-LE lower=0/upper=256. Runtime bindings must NOT enter Declaration.
#[allow(dead_code)]
struct CoordinateDeclaration {
    parent: EdgeDescriptor,
    coordinate: EdgeDescriptor,
    table_hash: [u8; 32],
    high: bool,
    first: u64,
    lanes: u64,
}
// Separate acyclic runtime record: coordinate domain, runtime.v1, declaration digest,
// ORIGINAL parent Binding digest, ORIGINAL coordinate Binding digest, role.
#[allow(dead_code)]
struct CoordinateProvenance {
    declaration: [u8; 32],
    parent: Binding,
    coordinate: Binding,
    role: u8,
    digest: [u8; 32],
}

/// Metadata-only logical owner quote. NOT the full nExp execution peak and NOT
/// Budget authority. Original caller/source/table owners remain additional.
/// No constructor accepts this number as a substitute for an original lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NexpDeclarationResources {
    pub lanes: usize,
    pub declaration_bytes: usize,
    pub coordinate_provenance_bytes: usize,
    pub ledger_bytes: usize,
    pub hash_controls_bytes: usize,
    pub query_transition_bytes: usize,
    pub metadata_bytes: usize,
    pub exchanges: usize,
    pub frames: usize,
    pub total_peer_bytes: usize,
}
fn sum(values: &[usize]) -> Result<usize, Error> {
    values
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b).ok_or(Error::Resource))
}

/// Hash-/allocation-free prerequisite query, before records/table hashing. The
/// containing owner must separately charge this query's fixed input/output slots
/// before invocation. It prices concrete metadata only; missing nExp role/abort/
/// mailbox/output states mean no aggregate execution quote is claimed here.
pub fn nexp_declaration_resources(lanes: usize) -> Result<NexpDeclarationResources, Error> {
    if !(1..=256).contains(&lanes) {
        return Err(Error::Domain);
    }
    // Query new product's actual source layouts, not a matrix/D27 byte constant.
    let _product = elementwise_product::resources_for::<&Budget>(lanes)?;
    let declaration_bytes = sum(&[
        size_of::<Declaration>(),
        size_of::<Result<Declaration, Error>>(),
        size_of::<Declaration>(),
    ])?;
    let coordinate_provenance_bytes = sum(&[
        size_of::<[CoordinateProvenance; 2]>(),
        size_of::<Result<[CoordinateProvenance; 2], Error>>(),
        size_of::<[CoordinateDeclaration; 2]>(),
    ])?;
    let ledger_bytes = size_of::<[internal_use::Ledger<18>; 3]>();
    let hash_controls_bytes = sum(&[
        size_of::<[Sha256; 2]>(),
        size_of::<[sha2::digest::Output<Sha256>; 2]>(),
        size_of::<[&[u8]; 18]>(),
        size_of::<[u8; 8]>(),
        size_of::<[u8; 16]>(),
        size_of::<[Binding; 18]>(),
        size_of::<[ConsumerUse; 18]>(),
    ])?;
    let query_transition_bytes = sum(&[
        size_of::<elementwise_product::Resources>(),
        size_of::<Result<elementwise_product::Resources, Error>>(),
        size_of::<NexpDeclarationResources>(),
        size_of::<Result<NexpDeclarationResources, Error>>(),
        size_of::<Result<usize, Error>>(),
        size_of::<usize>(),
        size_of::<[usize; 5]>(),
        size_of::<[usize; 3]>(),
        size_of::<[usize; 7]>(),
        size_of::<[usize; 11]>(),
        size_of::<std::slice::Iter<'_, usize>>(),
    ])?;
    let metadata_bytes = sum(&[
        declaration_bytes,
        coordinate_provenance_bytes,
        ledger_bytes,
        hash_controls_bytes,
        query_transition_bytes,
    ])?;
    let total_peer_bytes = 1688usize
        .checked_add(208usize.checked_mul(lanes).ok_or(Error::Resource)?)
        .ok_or(Error::Resource)?;
    let _ = (REGION_DOMAIN, COORDINATE_DOMAIN, EQUATION);
    Ok(NexpDeclarationResources {
        lanes,
        declaration_bytes,
        coordinate_provenance_bytes,
        ledger_bytes,
        hash_controls_bytes,
        query_transition_bytes,
        metadata_bytes,
        exchanges: 11,
        frames: 22,
        total_peer_bytes,
    })
}

/// Complete clear-root domain preflight. No masks, keys, entropy, hash, operation
/// counter or secret allocation. This checks a local premise, not compiler/source
/// proof. It does not call prepare_root, whose mask generation precedes encode.
/// No inherited capability can be minted from these values or descriptors.
pub fn validate_nexp_clear_root(input: &EdgeDescriptor, values: &[i64]) -> Result<(), Error> {
    let numeric = input.numeric();
    if numeric.ring_bits() != 50
        || numeric.fractional_bits() != 12
        || numeric.interpretation() != Interpretation::Signed
        || input.representation() != Representation::CommonAdditiveMasked
        || input.weight_visibility() == Some(WeightVisibility::Public)
    {
        return Err(Error::Domain);
    }
    if !(1..=256).contains(&input.shape().elements()) {
        return Err(Error::Resource);
    }
    input
        .identity()
        .first_tensor_index()
        .checked_add(input.shape().elements() as u64)
        .ok_or(Error::Resource)?;
    if values.len() != input.shape().elements() {
        return Err(Error::Domain);
    }
    if values
        .iter()
        .any(|value| !(0..(1i64 << 38)).contains(value))
    {
        return Err(Error::Domain);
    }
    Ok(())
}

#[cfg(test)]
#[path = "nexp_tests.rs"]
mod tests;

#[path = "nexp_region.rs"]
#[allow(dead_code)] // Private local composition, not exported/registered/adopted.
pub(super) mod region;
