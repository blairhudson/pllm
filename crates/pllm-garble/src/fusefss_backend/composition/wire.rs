//! Optional role-separated material transport for immutable composition plans.
//! This module is a child of `runtime`, publicly re-exported by composition.
//! It never calls dealer issuance and does not alter historical plan resources
//! or digests. Outer transport must authenticate packet recipient and issuance.
use super::super::*;
use super::{Material, Primitive};
pub use crate::fusefss_backend::wire::{
    ImportGuard, ImportGuardLayout, IndexedFileResources, MaterialPacket,
};
use crate::fusefss_backend::{wire::ENVELOPE_BYTES, CompiledGelu, LookupBackend};

/// Additional logical storage at one public step. Public Plan/Table owners and
/// runtime registers are separate. Packet bytes include the 120-byte envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepTransportResources {
    pub packet_bytes_bound: usize,
    pub decoded_party_reservation_bytes: usize,
    pub import_temporary_bytes: usize,
    pub import_overlap_bytes: usize,
    pub export_overlap_bytes: usize,
}

/// Optional, conservative streaming admission; historical Resources stay intact.
/// One receiving role, one active primitive, previous step's complete frame set,
/// and caller-selected extra retained packet slots. Older frames must be retired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransportResources {
    pub secret_steps: usize,
    pub cumulative_packet_bytes_per_party_bound: usize,
    pub largest_packet_bytes_bound: usize,
    pub largest_import_overlap_bytes: usize,
    pub largest_export_overlap_bytes: usize,
    pub retained_packet_bytes: usize,
    pub retained_frame_bytes_per_party: usize,
    pub guard_reservation_bytes: usize,
    pub combined_worker_reservation_bytes: usize,
    /// Trusted dealer only: historical paired issuance peak plus both current
    /// packet owners, public plan and additional retained packets; no runtimes.
    pub combined_dealer_export_reservation_bytes: usize,
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Resource)
}
fn step_cost(
    packet: usize,
    party: usize,
    temporary: usize,
) -> Result<StepTransportResources, Error> {
    let export = add(packet, party)?;
    let import = add(export, temporary)?;
    if import > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(StepTransportResources {
        packet_bytes_bound: packet,
        decoded_party_reservation_bytes: party,
        import_temporary_bytes: temporary,
        import_overlap_bytes: import,
        export_overlap_bytes: export,
    })
}

impl<'a> Material<'a> {
    /// Consume this fresh party-local owner. Native exporters reject started or
    /// consumed owners; no peer share, dealer mask, seed generator or clear
    /// translated partition is serialized. Packet may outlive the immutable Plan.
    pub fn export_packet(self) -> Result<MaterialPacket<'a>, Error> {
        match self.primitive {
            Primitive::Conversion(p) => p.export_material(),
            Primitive::Predicate(p) => p.export_material(),
            Primitive::Lookup(p) => p.export_material(),
            Primitive::Gelu(p, false) => p.export_material(),
            Primitive::Gelu(_, true) => Err(Error::Consumed),
            Primitive::OneHot(p) => (*p).export_material(),
        }
    }
}

impl<'a> Plan<'a> {
    /// Public per-step geometry, without issuance or private compilation. Local
    /// arithmetic steps cannot receive material and return Domain.
    pub fn step_transport_resources(&self, pc: usize) -> Result<StepTransportResources, Error> {
        match self.instruction(pc)? {
            Instruction::Convert(..) | Instruction::Multiply(..) | Instruction::Select(..) => {
                let r = conversions::resources(self.conversion_spec(pc)?)?;
                step_cost(
                    add(ENVELOPE_BYTES, r.private_material_bytes_per_party)?,
                    r.party_reservation_bytes,
                    0,
                )
            }
            Instruction::Predicate(value, signed, threshold) => {
                let _scratch = self.lease.budget.reserve(DESCRIPTOR_SCRATCH)?;
                self.lease.budget.allocation()?;
                let descriptor =
                    Descriptor::simple_comparison(self.register(value)?.bits, signed, threshold)
                        .map_err(descriptor_error)?;
                let r = predicates::resources(&descriptor)?;
                step_cost(
                    r.material_packet_bytes_bound,
                    r.party_reservation_bytes,
                    DESCRIPTOR_SCRATCH,
                )
            }
            Instruction::Lookup(_, table) => {
                let r = self.tables[table].resources(1)?;
                let temporary = r
                    .issuance_reservation_bytes
                    .checked_sub(mul(2, r.party_reservation_bytes)?)
                    .ok_or(Error::Resource)?;
                step_cost(
                    add(ENVELOPE_BYTES, r.private_key_bytes_per_party)?,
                    r.party_reservation_bytes,
                    temporary,
                )
            }
            Instruction::Gelu(_) => {
                let public =
                    CompiledGelu::with_backend(self.lease.budget, LookupBackend::PublicBoundary)?;
                let r = public.resources(1)?;
                step_cost(
                    add(ENVELOPE_BYTES, r.issuance.party_key_logical_bytes)?,
                    r.issuance.party_reservation_bytes,
                    r.descriptor_reservation_bytes,
                )
            }
            Instruction::CompactTokenOneHot(_) => {
                let r = one_hot::resources(self.one_hot_spec(pc)?)?;
                // Native party reservation includes output, full-domain scratch,
                // key storage and the Party moved into the adapter's Box.
                step_cost(
                    r.material_packet_bytes_per_party,
                    r.party_reservation_bytes,
                    0,
                )
            }
            _ => Err(Error::Domain),
        }
    }

    /// Price explicit transport overlap under unchanged 64 MiB Budget. Guard
    /// capacity is the actual session-long ImportGuard capacity (1..=4096).
    /// Exhaustion fails closed; this API never resets or grows a guard.
    /// Caller also needs one unused claim per secret step/failed request. This
    /// is a storage bound, not a promise about previously consumed guard slots
    /// or concurrent Budget headroom. `retained_packet_slots` includes any
    /// retained native delivery buffer while import_step_bytes makes its copy.
    pub fn transport_resources(
        &self,
        retained_packet_slots: usize,
        guard_capacity: usize,
    ) -> Result<TransportResources, Error> {
        self.transport_resources_with_guard(
            retained_packet_slots,
            ImportGuardLayout::legacy(guard_capacity)?,
        )
    }

    /// Request-sized counterpart to transport_resources. Charges the exact
    /// ImportGuard::new_indexed allocation for the entire request-long claim
    /// allowance, rather than capping its price at the legacy 4096-entry ledger.
    pub fn transport_resources_indexed(
        &self,
        retained_packet_slots: usize,
        guard_capacity: usize,
    ) -> Result<TransportResources, Error> {
        self.transport_resources_with_guard(
            retained_packet_slots,
            ImportGuardLayout::for_requests(guard_capacity)?,
        )
    }

    /// Price the exact validated layout used by ImportGuard::with_layout. Use
    /// ImportGuardLayout::for_requests for the full request-long claim count,
    /// including all contexts and any failed-import allowance. The same owner
    /// must survive every Plan/runtime, cancellation and malformed packet.
    /// This is overlap admission, not a claim that unused slots remain. Public
    /// source/table owners and the native aggregate budget remain separate.
    pub fn transport_resources_with_guard(
        &self,
        retained_packet_slots: usize,
        guard_layout: ImportGuardLayout,
    ) -> Result<TransportResources, Error> {
        self.transport_resources_with_guard_bytes(
            retained_packet_slots,
            guard_layout.reservation_bytes(),
        )
    }

    /// Resident overlap for one file-backed guard, including its two fixed I/O
    /// scratch buffers. Reserve indexed_file_resources(capacity).disk_bytes
    /// separately for the full request; two holders need two disk/resident owners.
    pub fn transport_resources_indexed_file(
        &self,
        retained_packet_slots: usize,
        guard_capacity: usize,
    ) -> Result<TransportResources, Error> {
        self.transport_resources_with_guard_bytes(
            retained_packet_slots,
            ImportGuard::indexed_file_resources(guard_capacity)?.resident_bytes,
        )
    }

    fn transport_resources_with_guard_bytes(
        &self,
        retained_packet_slots: usize,
        guard_bytes: usize,
    ) -> Result<TransportResources, Error> {
        let mut largest_packet = 0;
        let mut largest_import = 0;
        let mut largest_export = 0;
        let mut cumulative = 0;
        for pc in 0..self.resources.instructions {
            let instruction = self.instruction(pc)?;
            if matches!(
                instruction,
                Instruction::Constant(_)
                    | Instruction::Add(..)
                    | Instruction::Sub(..)
                    | Instruction::PublicMultiply(..)
                    | Instruction::Narrow(_)
            ) {
                continue;
            }
            let r = self.step_transport_resources(pc)?;
            largest_packet = largest_packet.max(r.packet_bytes_bound);
            largest_import = largest_import.max(r.import_overlap_bytes);
            largest_export = largest_export.max(r.export_overlap_bytes);
            cumulative = add(cumulative, r.packet_bytes_bound)?;
        }
        let retained_packet_bytes = mul(retained_packet_slots, largest_packet)?;
        let retained_frames = self.resources.retained_frame_overlap_bytes / 2;
        let worker = add(
            add(
                add(
                    add(
                        add(
                            self.resources.plan_reservation_bytes,
                            self.resources.register_owner_bytes_per_party,
                        )?,
                        largest_import,
                    )?,
                    retained_packet_bytes,
                )?,
                retained_frames,
            )?,
            guard_bytes,
        )?;
        let dealer = add(
            add(
                add(
                    self.resources.plan_reservation_bytes,
                    self.resources.largest_primitive_reservation_bytes,
                )?,
                mul(2, largest_packet)?,
            )?,
            retained_packet_bytes,
        )?;
        if worker > MAX_ALLOCATION_BYTES || dealer > MAX_ALLOCATION_BYTES {
            return Err(Error::Resource);
        }
        Ok(TransportResources {
            secret_steps: self.resources.secret_steps,
            cumulative_packet_bytes_per_party_bound: cumulative,
            largest_packet_bytes_bound: largest_packet,
            largest_import_overlap_bytes: largest_import,
            largest_export_overlap_bytes: largest_export,
            retained_packet_bytes,
            retained_frame_bytes_per_party: retained_frames,
            guard_reservation_bytes: guard_bytes,
            combined_worker_reservation_bytes: worker,
            combined_dealer_export_reservation_bytes: dealer,
        })
    }

    /// Import exactly the public instruction requested by Event::NeedMaterial.
    /// All operation parameters, bit widths, table identities and step bindings
    /// come from this immutable Plan, never packet-selected instructions. Keep
    /// guard alive across every step, failed import and cancelled runtime.
    /// Feed the Result through Engine::feed_result so import failure also burns
    /// the waiting runtime's input/register state.
    pub fn import_step(
        &self,
        pc: usize,
        context: Context,
        role: u8,
        nonce: [u8; 32],
        packet: MaterialPacket<'_>,
        guard: &mut ImportGuard<'_>,
    ) -> Result<Material<'a>, Error> {
        let mut dispatched = false;
        let result = (|| {
            if role > 1 {
                return Err(Error::Domain);
            }
            context
                .first_tensor_index
                .checked_add(1)
                .ok_or(Error::Domain)?;
            let instruction = self.instruction(pc)?;
            let context = self.step_context(context, pc);
            let budget = self.lease.budget;
            let primitive = match instruction {
                Instruction::Convert(..) | Instruction::Multiply(..) | Instruction::Select(..) => {
                    let spec = self.conversion_spec(pc)?;
                    dispatched = true;
                    Primitive::Conversion(conversions::Party::import_material(
                        spec, context, role, nonce, packet, guard, budget,
                    )?)
                }
                Instruction::Predicate(value, signed, threshold) => {
                    let _scratch = budget.reserve(DESCRIPTOR_SCRATCH)?;
                    budget.allocation()?;
                    let descriptor = Descriptor::simple_comparison(
                        self.register(value)?.bits,
                        signed,
                        threshold,
                    )
                    .map_err(descriptor_error)?;
                    dispatched = true;
                    Primitive::Predicate(predicates::Party::import_material(
                        &descriptor,
                        self.source,
                        context,
                        role,
                        nonce,
                        packet,
                        guard,
                        budget,
                    )?)
                }
                Instruction::Lookup(_, table) => {
                    dispatched = true;
                    Primitive::Lookup(lookup::Party::import_material(
                        &self.tables[table],
                        1,
                        context,
                        role,
                        nonce,
                        packet,
                        guard,
                    )?)
                }
                Instruction::Gelu(_) => {
                    let public = CompiledGelu::with_backend(budget, LookupBackend::PublicBoundary)?;
                    dispatched = true;
                    Primitive::Gelu(
                        public.import_material(1, context, role, nonce, packet, guard)?,
                        false,
                    )
                }
                Instruction::CompactTokenOneHot(_) => {
                    let spec = self.one_hot_spec(pc)?;
                    dispatched = true;
                    let party = one_hot::Party::import_material(
                        spec, context, role, nonce, packet, guard, budget,
                    )?;
                    budget.allocation()?;
                    Primitive::OneHot(Box::new(party))
                }
                _ => return Err(Error::Domain),
            };
            Ok(Material {
                plan: self.digest,
                operation: context.operation,
                pc,
                role,
                primitive,
            })
        })();
        // Failures before reaching a native importer must burn issuance too.
        if !dispatched {
            guard.claim(role, nonce)?;
        }
        result
    }

    /// Python/SDK-friendly borrowed authenticated bytes boundary. Reject packet
    /// capacities exceeding this exact step's public bound before copying. The
    /// original caller buffer remains externally owned/accounted and must erase.
    pub fn import_step_bytes(
        &self,
        pc: usize,
        context: Context,
        role: u8,
        nonce: [u8; 32],
        bytes: &[u8],
        guard: &mut ImportGuard<'_>,
    ) -> Result<Material<'a>, Error> {
        let prepare = (|| {
            let r = self.step_transport_resources(pc)?;
            if bytes.len() > r.packet_bytes_bound {
                return Err(Error::Resource);
            }
            MaterialPacket::from_bytes(bytes, self.lease.budget)
        })();
        match prepare {
            Ok(packet) => self.import_step(pc, context, role, nonce, packet, guard),
            Err(error) => {
                guard.claim(role, nonce)?;
                Err(error)
            }
        }
    }
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;
