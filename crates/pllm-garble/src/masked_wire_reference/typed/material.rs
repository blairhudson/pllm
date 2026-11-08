//! One local matrix-material transfer, not a public byte decoder or transport.
use super::*;
use crate::shared_allocation::MAX_ALLOCATION_BYTES;

/// Classification of the output's declared interval provenance. Neither variant
/// is a compiler range proof; the complete bound remains in the binding digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixMaterialProvenance {
    LocalAssumption,
    RingOnly,
}

/// Inspectable declaration, not authorization or an evaluator/dealer capability.
/// Fields stay private; only a live owner can mint its corresponding permit.
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::{MatrixMaterialDescriptor, MatrixParty};
/// fn owner<'b>(d: MatrixMaterialDescriptor) -> MatrixParty<'b> { d.into() }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixMaterialDescriptor {
    implementation: [u8; 32],
    operation: [u8; 32],
    role: u8,
    spec: matrix::Spec,
    inputs: [[u8; 32]; 2],
    output: [u8; 32],
    native_binding: [u8; 32],
    provenance: MatrixMaterialProvenance,
}

impl MatrixMaterialDescriptor {
    pub fn implementation(&self) -> [u8; 32] {
        self.implementation
    }
    pub fn operation(&self) -> [u8; 32] {
        self.operation
    }
    pub fn role(&self) -> u8 {
        self.role
    }
    pub fn spec(&self) -> matrix::Spec {
        self.spec
    }
    pub fn input_binding_digests(&self) -> [[u8; 32]; 2] {
        self.inputs
    }
    pub fn output_binding_digest(&self) -> [u8; 32] {
        self.output
    }
    pub fn native_execution_binding(&self) -> [u8; 32] {
        self.native_binding
    }
    pub fn provenance(&self) -> MatrixMaterialProvenance {
        self.provenance
    }
    /// Canonical v1 material-codec commitment, separate from edge/binding hashes.
    pub fn digest(&self) -> [u8; 32] {
        let layout = match self.spec.layout {
            matrix::MatrixLayout::InputOutput => 0,
            matrix::MatrixLayout::OutputInput => 1,
        };
        let provenance = match self.provenance {
            MatrixMaterialProvenance::LocalAssumption => 1,
            MatrixMaterialProvenance::RingOnly => 0,
        };
        digest(&[
            b"pllm.sigma.typed.matrix-material-codec.v1\0",
            &self.implementation,
            &self.operation,
            &[self.role],
            &[self.spec.bits],
            &[layout],
            &(self.spec.rows as u64).to_le_bytes(),
            &(self.spec.inner as u64).to_le_bytes(),
            &(self.spec.columns as u64).to_le_bytes(),
            &self.inputs[0],
            &self.inputs[1],
            &self.output,
            &self.native_binding,
            &[provenance],
        ])
    }
}

/// Incremental logical peak beyond the already-reserved live typed party.
/// Native charge includes native packet/material restoration; typed charge has
/// no heap allocation. Metadata is conservatively counted for handoff, permit,
/// packet and descriptor simultaneously, plus bounded SHA/move stack allowance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixMaterialResources {
    pub native_incremental_reservation_bytes: usize,
    pub native_owner_metadata_bytes: usize,
    pub native_allocator_allowance_bytes: usize,
    pub native_stack_allowance_bytes: usize,
    /// Informational overlap, already covered by original/native reservations;
    /// not another additive charge.
    pub native_import_overlap_bytes: usize,
    pub packet_bytes: usize,
    pub body_bytes: usize,
    pub typed_permit_metadata_bytes: usize,
    pub typed_packet_metadata_bytes: usize,
    pub typed_stack_allowance_bytes: usize,
    pub typed_incremental_reservation_bytes: usize,
    pub incremental_reservation_bytes: usize,
}

pub fn matrix_material_resources(spec: matrix::Spec) -> Result<MatrixMaterialResources, Error> {
    let native = matrix::party::material::handoff_resources(spec)?;
    let permit = size_of::<MatrixImportPermit<'static>>()
        .checked_add(size_of::<MatrixHandoff<'static>>())
        .and_then(|n| n.checked_add(size_of::<MatrixMaterialDescriptor>()))
        .ok_or(Error::Resource)?;
    let packet = size_of::<MatrixMaterialPacket<'static>>();
    let stack = 8usize << 10;
    let typed = permit
        .checked_add(packet)
        .and_then(|n| n.checked_add(stack))
        .ok_or(Error::Resource)?;
    let total = native
        .incremental_reservation_bytes
        .checked_add(typed)
        .filter(|&n| n <= MAX_ALLOCATION_BYTES)
        .ok_or(Error::Resource)?;
    Ok(MatrixMaterialResources {
        native_incremental_reservation_bytes: native.incremental_reservation_bytes,
        native_owner_metadata_bytes: native.owner_metadata_bytes,
        native_allocator_allowance_bytes: native.allocator_allowance_bytes,
        native_stack_allowance_bytes: native.stack_allowance_bytes,
        native_import_overlap_bytes: native.import_overlap_bytes,
        packet_bytes: native.packet_bytes,
        body_bytes: native.body_bytes,
        typed_permit_metadata_bytes: permit,
        typed_packet_metadata_bytes: packet,
        typed_stack_allowance_bytes: stack,
        typed_incremental_reservation_bytes: typed,
        incremental_reservation_bytes: total,
    })
}

/// Noncloneable same-budget local handoff. Splitting does not duplicate rights.
pub struct MatrixHandoff<'b> {
    permit: MatrixImportPermit<'b>,
    packet: MatrixMaterialPacket<'b>,
}
impl<'b> MatrixHandoff<'b> {
    pub fn descriptor(&self) -> MatrixMaterialDescriptor {
        self.permit.descriptor
    }
    pub fn into_parts(self) -> (MatrixImportPermit<'b>, MatrixMaterialPacket<'b>) {
        (self.permit, self.packet)
    }
}

/// Opaque secret bytes, read-only. No raw constructor or borrowed-byte importer.
/// Its independent lease stays live until the native packet has been erased.
pub struct MatrixMaterialPacket<'b> {
    native: matrix::party::material::Packet<'b>,
    _lease: Lease<'b>,
}
impl MatrixMaterialPacket<'_> {
    pub fn as_bytes(&self) -> &[u8] {
        self.native.as_bytes()
    }
}

/// Single-use local authority minted only by consuming a fresh typed owner.
/// Not cloneable or serializable. Import consumes both owners even on failure.
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::MatrixImportPermit;
/// fn duplicate<'b>(p: MatrixImportPermit<'b>) { let _ = p.clone(); }
/// ```
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::{MatrixImportPermit, MatrixMaterialPacket};
/// fn reuse<'b>(p: MatrixImportPermit<'b>, a: MatrixMaterialPacket<'b>, b: MatrixMaterialPacket<'b>) {
///     let _ = p.import(a);
///     let _ = p.import(b);
/// }
/// ```
pub struct MatrixImportPermit<'b> {
    native: matrix::party::material::ImportPermit<'b>,
    inputs: [Binding; 2],
    output: Binding,
    operation: [u8; 32],
    role: u8,
    descriptor: MatrixMaterialDescriptor,
    // Original enlarged output/wrapper/finished-slot charge never leaves this
    // borrowed ownership chain. Enclosing layouts above are measured, not assumed.
    lease: Lease<'b>,
    material_lease: Lease<'b>,
}
impl<'b> MatrixImportPermit<'b> {
    pub fn import(self, packet: MatrixMaterialPacket<'b>) -> Result<MatrixParty<'b>, Error> {
        let MatrixMaterialPacket { native, _lease } = packet;
        // Native authority checks its original complete packet hash, expected
        // role, descriptor and execution binding, not merely a public checksum.
        let native = self.native.import(native)?;
        Ok(MatrixParty {
            native: Some(native),
            inputs: self.inputs,
            output: self.output,
            operation: self.operation,
            role: self.role,
            imported: true,
            lease: Some(self.lease),
            material_lease: Some(self.material_lease),
        })
    }
}

impl<'b> MatrixParty<'b> {
    /// Consume fresh material into one local permit/packet pair. Any failure
    /// burns this party; the coordinator must explicitly retire its sibling.
    /// No entropy, external authorization or replacement input owners are used.
    pub fn export_material(mut self) -> Result<MatrixHandoff<'b>, Error> {
        if self.imported {
            return Err(Error::Consumed);
        }
        let native = self.native.take().ok_or(Error::Consumed)?;
        let (role, spec, native_binding) = native.material_identity()?;
        if role != self.role {
            return Err(Error::Context);
        }
        let cost = matrix_material_resources(spec)?;
        // Keep original lease inside self until native export succeeds, so a
        // failed admission drops native secret material before its wrapper charge.
        let budget = self.lease.as_ref().ok_or(Error::Consumed)?.budget;
        let mut extra = budget.reserve(cost.typed_incremental_reservation_bytes)?;
        let packet_lease = extra.split(cost.typed_packet_metadata_bytes)?;
        let descriptor = MatrixMaterialDescriptor {
            implementation: approved_implementation(),
            operation: self.operation,
            role,
            spec,
            inputs: self.inputs.map(Binding::digest),
            output: self.output.digest(),
            native_binding,
            provenance: if self.output.interval.is_some() {
                MatrixMaterialProvenance::LocalAssumption
            } else {
                MatrixMaterialProvenance::RingOnly
            },
        };
        // Reserve typed overhead first; native reserves its own incremental
        // resources while the original native and typed charges remain held.
        let (permit, packet) = native.export_material(descriptor.digest())?;
        let original = self.lease.take().ok_or(Error::Consumed)?;
        Ok(MatrixHandoff {
            permit: MatrixImportPermit {
                native: permit,
                inputs: self.inputs,
                output: self.output,
                operation: self.operation,
                role,
                descriptor,
                lease: original,
                material_lease: extra,
            },
            packet: MatrixMaterialPacket {
                native: packet,
                _lease: packet_lease,
            },
        })
    }
}

#[cfg(test)]
mod tests;
