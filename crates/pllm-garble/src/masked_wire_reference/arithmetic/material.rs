//! Single-use, process-local arithmetic material transfer. No remote
//! authentication, source authority, routing or compiler permission is conferred.
//! Output masks stay with their original dealer/composition owner.
//!
//! Logical slots below are not physical stack, allocator/RSS, CPU or SourceWork
//! proofs. Zeroization covers owned secret backing, not stack/register erasure.
use super::*;
use crate::material_wire::{self, MaterialPacket, ENVELOPE_BYTES};
use crate::shared_allocation::Lease;
use zeroize::Zeroize;

const MAGIC: &[u8; 8] = b"PLLMSA02";
const KIND: u8 = 1;
const DOMAIN: &[u8] = b"pllm.masked-wire.arithmetic-material.v1\0";
const HASH_STACK: usize = 8 << 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub key_codec_bytes: usize,
    pub body_bytes: usize,
    pub packet_bytes: usize,
    pub owner_metadata_bytes: usize,
    pub transient_bytes: usize,
    pub allocator_allowance_bytes: usize,
    pub stack_allowance_bytes: usize,
    pub original_reservation_bytes: usize,
    pub incremental_reservation_bytes: usize,
    pub combined_reservation_bytes: usize,
}

fn sum(values: &[usize]) -> Result<usize, Error> {
    values.iter().try_fold(0usize, |a, b| a.checked_add(*b)).ok_or(Error::Resource)
}

/// Checked concrete-owner inventory. R remains live throughout export/import;
/// P owns packet backing, K covers exactly one temporary canonical key.
pub fn resources(spec: Spec) -> Result<Resources, Error> {
    let original = super::resources(spec)?.party_reservation_bytes;
    let key = spec.geometry()?.0.resources().codec_bytes;
    let body = spec.lanes.checked_mul(key.checked_add(24).ok_or(Error::Resource)?)
        .ok_or(Error::Resource)?;
    let packet = sum(&[ENVELOPE_BYTES, body])?;
    let metadata = sum(&[
        size_of::<Handoff<'static>>(), size_of::<ImportPermit<'static>>(),
        size_of::<Packet<'static>>(), size_of::<ExportStage<'static>>(),
        size_of::<ImportStage<'static>>(), size_of::<Result<Handoff<'static>, Error>>(),
        size_of::<Result<Party<'static>, Error>>(),
    ])?;
    let transient = sum(&[
        size_of::<std::vec::IntoIter<Lane>>(), size_of::<Lane>(),
        size_of::<Result<GpuKey, gpu127::Error>>(), size_of::<crate::compact_dpf::Bits>(),
        size_of::<crate::compact_dpf::BitReader<'static>>(),
        size_of::<Zeroizing<Vec<u8>>>(),
        size_of::<Result<Zeroizing<Vec<u8>>, gpu127::Error>>(),
        // Additional actual owners: shared packet return/local, decoded constants,
        // rebuilt Execution return, and allocation/zero-vector return slots.
        size_of::<MaterialPacket<'static>>(),
        size_of::<Result<MaterialPacket<'static>, Error>>(),
        size_of::<Zeroizing<[u64; 3]>>(), size_of::<Execution>(),
        size_of::<Result<Vec<Lane>, Error>>(), size_of::<Vec<Lane>>(),
        size_of::<Result<Zeroizing<Vec<u64>>, Error>>(),
        size_of::<Zeroizing<Vec<u64>>>(), size_of::<Result<Vec<u64>, Error>>(),
        size_of::<Vec<u64>>(), size_of::<Result<Vec<u8>, Error>>(),
        size_of::<Vec<u8>>(), size_of::<Lease<'static>>(),
        size_of::<Result<Lease<'static>, Error>>(),
        // Iterator next, inline decode local, take/split return and moved scratch
        // tuple are distinct logical transition slots, not physical stack copies.
        size_of::<Option<Lane>>(), size_of::<GpuKey>(),
        size_of::<Option<Execution>>(), size_of::<Option<Packet<'static>>>(),
        size_of::<(ImportPermit<'static>, Packet<'static>)>(),
        size_of::<(Zeroizing<Vec<u64>>, Zeroizing<Vec<u64>>)>(),
        // Borrowing export and in-place retirement slots. Retain earlier
        // conservative slots above; these are logical, not ABI stack claims.
        size_of::<std::slice::Iter<'static, Lane>>(),
        size_of::<std::slice::IterMut<'static, Lane>>(),
        size_of::<&'static Execution>(), size_of::<&'static mut Execution>(),
        size_of::<&'static Lane>(), size_of::<&'static mut Lane>(),
        size_of::<&'static mut GpuKey>(),
    ])?;
    let incremental = sum(&[packet, key, metadata, transient, 2 * ALLOCATOR, HASH_STACK])?;
    let combined = sum(&[original, incremental])?;
    if combined > MAX_ALLOCATION_BYTES { return Err(Error::Resource); }
    Ok(Resources {
        key_codec_bytes: key, body_bytes: body, packet_bytes: packet,
        owner_metadata_bytes: metadata, transient_bytes: transient,
        allocator_allowance_bytes: 2 * ALLOCATOR, stack_allowance_bytes: HASH_STACK,
        original_reservation_bytes: original, incremental_reservation_bytes: incremental,
        combined_reservation_bytes: combined,
    })
}

// Declaration order is intentional: secret owners retire before any leases.
struct ExportStage<'b> {
    execution: Option<Execution>,
    original: Option<Lease<'b>>,
    workspace: Option<Lease<'b>>,
}
impl ExportStage<'_> {
    fn retire_execution(&mut self) {
        if let Some(execution) = self.execution.as_mut() {
            // Never move Lane/key/constants out: clear every original heap slot,
            // including already encoded and not-yet-encoded lanes, while live.
            for lane in &mut execution.lanes {
                lane.key.clear_secret_fields();
                lane.constants.zeroize();
            }
            #[cfg(test)]
            tests::observe_retired_lanes(&execution.lanes);
        }
        // Cleared lanes and scratch retire under the unchanged original lease.
        drop(self.execution.take());
    }
}
impl Drop for ExportStage<'_> {
    fn drop(&mut self) {
        self.retire_execution();
    }
}
struct ImportStage<'b> {
    execution: Option<Execution>,
    packet: Option<Packet<'b>>,
    permit: Option<ImportPermit<'b>>,
}

/// Opaque byte owner. No raw constructor, mutation, Clone or permit extraction.
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::arithmetic::material::Packet;
/// fn duplicate(p: Packet<'_>) { let _ = p.clone(); }
/// ```
pub struct Packet<'b> {
    packet: MaterialPacket<'b>,
    _metadata: Lease<'b>,
}
impl Packet<'_> {
    pub fn as_bytes(&self) -> &[u8] { self.packet.as_bytes() }
}

/// One trusted issuance, consumed even on failed import.
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::arithmetic::material::ImportPermit;
/// fn duplicate(p: ImportPermit<'_>) { let _ = p.clone(); }
/// ```
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::arithmetic::material::{ImportPermit, Packet};
/// fn reuse<'b>(p: ImportPermit<'b>, a: Packet<'b>, b: Packet<'b>) {
///     let _ = p.import(a);
///     let _ = p.import(b);
/// }
/// ```
pub struct ImportPermit<'b> {
    role: u8,
    spec: Spec,
    binding: [u8; 32],
    descriptor: [u8; 32],
    packet_digest: [u8; 32],
    original: Lease<'b>,
    _workspace: Lease<'b>,
}

pub struct Handoff<'b> {
    packet: Packet<'b>,
    permit: ImportPermit<'b>,
}
impl<'b> Handoff<'b> {
    pub fn into_parts(self) -> (ImportPermit<'b>, Packet<'b>) {
        (self.permit, self.packet)
    }
}

fn descriptor(spec: Spec, role: u8, binding: [u8; 32]) -> Result<[u8; 32], Error> {
    let (profile, output, _) = spec.geometry()?;
    let (gate, shift) = match spec.gate {
        Gate::Tr { shift } => (0, shift),
        Gate::GapArs { shift } => (1, shift),
        Gate::Drelu => (2, 0),
    };
    let mut hash = Sha256::new();
    hash.update(DOMAIN);
    for domain in [super::DOMAIN.as_bytes(), Profile::DOMAIN.as_bytes()] {
        hash.update((domain.len() as u64).to_le_bytes());
        hash.update(domain);
    }
    hash.update([spec.bits, profile.bits(), output, gate, shift, role]);
    hash.update((spec.lanes as u64).to_le_bytes());
    hash.update(binding);
    Ok(hash.finalize().into())
}

fn validate_constants(spec: Spec, constants: &[u64; 3]) -> Result<(), Error> {
    let output = spec.geometry()?.1;
    if constants.iter().any(|c| c & !mask(output) != 0)
        || (matches!(spec.gate, Gate::Tr { .. }) && constants[1] != 0)
        || (spec.gate == Gate::Drelu && (constants[1] != 0 || constants[2] != 0))
    { return Err(Error::Domain); }
    Ok(())
}

impl<'b> PartyFor<&'b Budget> {
    pub fn export_material(self) -> Result<Handoff<'b>, Error> {
        let PartyFor { id, spec, binding, state, lease } = self;
        let mut stage = ExportStage { execution: state, original: Some(lease), workspace: None };
        let cost = resources(spec)?;
        if id > 1 { return Err(Error::Context); }
        let execution = stage.execution.as_ref().ok_or(Error::Consumed)?;
        if execution.phase != Phase::Fresh { return Err(Error::Consumed); }
        let profile = spec.geometry()?.0;
        if execution.lanes.len() != spec.lanes || execution.input.len() != spec.lanes
            || execution.local.len() != spec.lanes || execution.input_digest != [0; 32]
            || execution.lanes.capacity() != spec.lanes || execution.input.capacity() != spec.lanes
            || execution.local.capacity() != spec.lanes
            || execution.input.iter().chain(execution.local.iter()).any(|v| *v != 0)
        { return Err(Error::Domain); }
        if stage.original.as_ref().ok_or(Error::Consumed)?.bytes != cost.original_reservation_bytes {
            return Err(Error::Resource);
        }
        for lane in &execution.lanes {
            if lane.key.profile() != profile || !lane.key.matches_party(id) {
                return Err(Error::Domain);
            }
            validate_constants(spec, &lane.constants)?;
        }
        let descriptor = descriptor(spec, id, binding)?;
        let budget = stage.original.as_ref().ok_or(Error::Consumed)?.budget;
        stage.workspace = Some(budget.reserve(cost.incremental_reservation_bytes - cost.packet_bytes)?);
        let packet = material_wire::packet(MAGIC, KIND, id, descriptor, binding,
            cost.body_bytes, budget, |bytes| {
                let execution = stage.execution.as_ref().ok_or(Error::Consumed)?;
                for lane in &execution.lanes {
                    let encoded = lane.key.material_with_reservation(cost.key_codec_bytes, |requested| {
                        if requested != cost.key_codec_bytes { return Err(gpu127::Error::Resource); }
                        budget.allocation().map_err(|_| gpu127::Error::Allocation)
                    }).map_err(primitive)?;
                    // Post-encoding capacity check is NOT proof against transient
                    // allocator overgrant. Workspace covers one exact K only.
                    if encoded.capacity() != cost.key_codec_bytes { return Err(Error::Resource); }
                    bytes.extend_from_slice(&encoded);
                    for constant in lane.constants.iter() { bytes.extend_from_slice(&constant.to_le_bytes()); }
                }
                Ok(())
            })?;
        stage.retire_execution();
        let packet_digest = Sha256::digest(packet.as_bytes()).into();
        let metadata = stage.workspace.as_mut().ok_or(Error::Consumed)?
            .split(size_of::<Packet<'static>>() + ALLOCATOR)?;
        Ok(Handoff {
            packet: Packet { packet, _metadata: metadata },
            permit: ImportPermit {
                role: id, spec, binding, descriptor, packet_digest,
                original: stage.original.take().ok_or(Error::Consumed)?,
                _workspace: stage.workspace.take().ok_or(Error::Consumed)?,
            },
        })
    }
}

// Whole body geometry and all canonical keys/constants are checked without heap
// allocation. Decode/drop each inline key before moving to the next lane.
fn validate_body(spec: Spec, role: u8, body: &[u8]) -> Result<(), Error> {
    let profile = spec.geometry()?.0;
    let key_bytes = profile.resources().codec_bytes;
    let stride = key_bytes.checked_add(24).ok_or(Error::Resource)?;
    if body.len() != spec.lanes.checked_mul(stride).ok_or(Error::Resource)? {
        return Err(Error::Domain);
    }
    let mut reader = body.chunks_exact(stride);
    for lane in &mut reader {
        drop(GpuKey::decode(profile, role, &lane[..key_bytes]).map_err(primitive)?);
        let mut constants = Zeroizing::new([0; 3]);
        for (value, bytes) in constants.iter_mut().zip(lane[key_bytes..].chunks_exact(8)) {
            *value = u64::from_le_bytes(bytes.try_into().map_err(|_| Error::Domain)?);
        }
        validate_constants(spec, &constants)?;
    }
    if !reader.remainder().is_empty() { return Err(Error::Domain); }
    Ok(())
}

impl<'b> ImportPermit<'b> {
    pub fn import(self, packet: Packet<'b>) -> Result<Party<'b>, Error> {
        let mut stage = ImportStage { execution: None, packet: Some(packet), permit: Some(self) };
        let permit = stage.permit.as_ref().ok_or(Error::Consumed)?;
        let packet = stage.packet.as_ref().ok_or(Error::Consumed)?;
        let cost = resources(permit.spec)?;
        if permit.role > 1 || descriptor(permit.spec, permit.role, permit.binding)? != permit.descriptor {
            return Err(Error::Context);
        }
        let digest: [u8; 32] = Sha256::digest(packet.as_bytes()).into();
        if digest != permit.packet_digest { return Err(Error::Context); }
        let body = material_wire::open(&packet.packet, MAGIC, KIND, permit.role,
            permit.descriptor, permit.binding, cost.body_bytes)?;
        validate_body(permit.spec, permit.role, body)?;
        let spec = permit.spec;
        let budget = permit.original.budget;
        stage.execution = Some(Execution {
            lanes: allocate(spec.lanes, budget)?,
            input: zeros(spec.lanes, budget)?, local: zeros(spec.lanes, budget)?,
            input_digest: [0; 32], phase: Phase::Fresh,
        });
        let execution = stage.execution.as_mut().ok_or(Error::Consumed)?;
        let profile = spec.geometry()?.0;
        let key_bytes = cost.key_codec_bytes;
        for lane in body.chunks_exact(key_bytes + 24) {
            let key = GpuKey::decode(profile, permit.role, &lane[..key_bytes]).map_err(primitive)?;
            let mut constants = Zeroizing::new([0; 3]);
            for (value, bytes) in constants.iter_mut().zip(lane[key_bytes..].chunks_exact(8)) {
                *value = u64::from_le_bytes(bytes.try_into().map_err(|_| Error::Domain)?);
            }
            execution.lanes.push(Lane { key, constants });
        }
        drop(stage.packet.take());
        let permit = stage.permit.take().ok_or(Error::Consumed)?;
        // Original lease moves unchanged; workspace releases only after packet
        // backing and every temporary decoded secret have retired.
        Ok(PartyFor { id: permit.role, spec, binding: permit.binding,
            state: stage.execution.take(), lease: permit.original })
    }
}

#[cfg(test)]
mod tests;
