//! Consuming, process-local material handoff. The permit, not packet bytes,
//! authorizes one import. No raw-byte import or transport authentication claim.
use super::*;
use crate::material_wire::{self, MaterialPacket, ENVELOPE_BYTES};

const MATERIAL_MAGIC: &[u8; 8] = b"PLLMSM01";
const KIND: u8 = 1;
const HASH_STACK: usize = 8 << 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HandoffResources {
    pub(crate) body_bytes: usize,
    pub(crate) packet_bytes: usize,
    pub(crate) owner_metadata_bytes: usize,
    pub(crate) allocator_allowance_bytes: usize,
    pub(crate) stack_allowance_bytes: usize,
    /// Existing party reservation remains live and covers recreated A/B/C.
    pub(crate) import_overlap_bytes: usize,
    /// Additional to the original party reservation, including codec charge.
    pub(crate) incremental_reservation_bytes: usize,
}

pub(crate) fn handoff_resources(spec: Spec) -> Result<HandoffResources, Error> {
    let cost = resources(spec)?;
    let body = cost.party_material_payload_bytes;
    let packet = ENVELOPE_BYTES.checked_add(body).ok_or(Error::Resource)?;
    // Actual permit/packet layouts include revised Live and Option<Live> owners.
    // The original enlarged party lease (including finished slots) stays intact.
    let metadata = size_of::<ImportPermit<'static>>()
        .checked_add(size_of::<Packet<'static>>())
        .ok_or(Error::Resource)?;
    let incremental = packet
        .checked_add(metadata)
        .and_then(|n| n.checked_add(ALLOCATOR))
        .and_then(|n| n.checked_add(HASH_STACK))
        .ok_or(Error::Resource)?;
    let combined = cost.party_reservation_bytes
        .checked_add(incremental)
        .ok_or(Error::Resource)?;
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(HandoffResources {
        body_bytes: body,
        packet_bytes: packet,
        owner_metadata_bytes: metadata,
        allocator_allowance_bytes: ALLOCATOR,
        stack_allowance_bytes: HASH_STACK,
        import_overlap_bytes: body + packet,
        incremental_reservation_bytes: incremental,
    })
}

pub(crate) struct Packet<'b> {
    // Erase/deallocate before either lease or singleton guard drops.
    packet: MaterialPacket<'b>,
    _lease: Lease<'b>,
    _live: Live,
}
impl Packet<'_> {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        self.packet.as_bytes()
    }
}

pub(crate) struct ImportPermit<'b> {
    role: u8,
    spec: Spec,
    binding: [u8; 32],
    descriptor_digest: [u8; 32],
    packet_digest: [u8; 32],
    lease: Lease<'b>,
    _workspace: Lease<'b>,
    live: Live,
}

impl<'b> Party<'b> {
    pub(crate) fn material_identity(&self) -> Result<(u8, Spec, [u8; 32]), Error> {
        if !matches!(self.state, Some(State::Fresh(_))) {
            return Err(Error::Consumed);
        }
        if self.id > 1 {
            return Err(Error::Context);
        }
        resources(self.spec)?;
        Ok((self.id, self.spec, self.binding))
    }

    pub(crate) fn export_material(
        mut self,
        descriptor_digest: [u8; 32],
    ) -> Result<(ImportPermit<'b>, Packet<'b>), Error> {
        let (role, spec, binding) = self.material_identity()?;
        let cost = handoff_resources(spec)?;
        let budget = self.lease.as_ref().ok_or(Error::Consumed)?.budget;
        // Codec separately reserves packet length. All additional owner/stack
        // allowances are reserved first, before any allocation or copy.
        let mut workspace =
            budget.reserve(cost.incremental_reservation_bytes - cost.packet_bytes)?;
        let State::Fresh(material) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Consumed);
        };
        let (a, b, c, _) = core::dimensions(spec)?;
        core::validate(&material.a, a, spec.bits)?;
        core::validate(&material.b, b, spec.bits)?;
        core::validate(&material.c, c, spec.bits)?;
        let packet = material_wire::packet(
            MATERIAL_MAGIC,
            KIND,
            role,
            descriptor_digest,
            binding,
            cost.body_bytes,
            budget,
            |bytes| {
                for values in [&material.a, &material.b, &material.c] {
                    for value in values.iter() {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                Ok(())
            },
        )?;
        let packet_digest = Sha256::digest(packet.as_bytes()).into();
        let packet_lease = workspace.split(size_of::<Packet<'static>>() + ALLOCATOR)?;
        let packet_live = self.live.as_ref().ok_or(Error::Consumed)?.fork();
        // Original arrays coexist with packet until here. Their reservation is
        // never refunded/reacquired: it moves to permit, then imported Party.
        drop(material);
        Ok((
            ImportPermit {
                role,
                spec,
                binding,
                descriptor_digest,
                packet_digest,
                lease: self.lease.take().ok_or(Error::Consumed)?,
                _workspace: workspace,
                live: self.live.take().ok_or(Error::Consumed)?,
            },
            Packet {
                packet,
                _lease: packet_lease,
                _live: packet_live,
            },
        ))
    }
}

impl<'b> ImportPermit<'b> {
    pub(crate) fn import(self, packet: Packet<'b>) -> Result<Party<'b>, Error> {
        let cost = handoff_resources(self.spec)?;
        if self.role > 1 {
            return Err(Error::Context);
        }
        // Trusted digest covers complete envelope AND body. Recomputing the
        // public checksum after changing canonical shares cannot authorize it.
        let digest: [u8; 32] = Sha256::digest(packet.as_bytes()).into();
        if digest != self.packet_digest {
            return Err(Error::Context);
        }
        let body = material_wire::open(
            &packet.packet,
            MATERIAL_MAGIC,
            KIND,
            self.role,
            self.descriptor_digest,
            self.binding,
            cost.body_bytes,
        )?;
        let (a, b, c, _) = core::dimensions(self.spec)?;
        // Validate every residue before the first recreated-array allocation.
        for word in body.chunks_exact(8) {
            let value = u64::from_le_bytes(word.try_into().map_err(|_| Error::Domain)?);
            if value & !core::mask(self.spec.bits) != 0 {
                return Err(Error::Domain);
            }
        }
        let mut material = core::Material {
            a: core::zeros(a, self.lease.budget)?,
            b: core::zeros(b, self.lease.budget)?,
            c: core::zeros(c, self.lease.budget)?,
        };
        let mut words = body.chunks_exact(8);
        for values in [&mut material.a, &mut material.b, &mut material.c] {
            for value in values.iter_mut() {
                *value = u64::from_le_bytes(
                    words
                        .next()
                        .ok_or(Error::Domain)?
                        .try_into()
                        .map_err(|_| Error::Domain)?,
                );
            }
        }
        // Packet erases before its independent charge/guard are released.
        drop(packet);
        Ok(Party {
            id: self.role,
            spec: self.spec,
            binding: self.binding,
            state: Some(State::Fresh(material)),
            lease: Some(self.lease),
            live: Some(self.live),
        })
    }
}

#[cfg(test)]
mod tests;
