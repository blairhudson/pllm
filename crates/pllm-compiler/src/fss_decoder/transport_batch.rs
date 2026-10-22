//! Bounded cross-Plan delivery ownership. Packet storage/erasure is the shared
//! native wire owner; this adapter only retains public descriptors and ordering.
use super::{add, mul, DealerTransport, Error, MaterialRequest};
#[cfg(test)]
use super::{Budget, MaterialPacket};
#[cfg(not(test))]
use pllm_garble::fusefss_backend::{wire::MaterialPacket, Budget};
use std::mem::size_of;

pub const CALLBACK_METADATA_BYTES_BOUND: usize = 1024;
pub const MATERIAL_BATCH_RECORDS_BOUND: usize = 64;
pub const MATERIAL_BATCH_BODY_BYTES_BOUND: usize = 1 << 20;
pub const MATERIAL_BATCH_ENVELOPE_BYTES_BOUND: usize = 1024;
/// Existing length-prefixed frame header plus the inner metadata-length word.
pub const MATERIAL_BATCH_RECORD_OVERHEAD_BYTES: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaterialDelivery {
    #[default]
    Single,
    BatchV1,
}
impl MaterialDelivery {
    pub fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::BatchV1 => "batch-v1",
        }
    }
}

/// Borrowed callback access to an independently budgeted, erasing packet owner.
/// Neither this record nor its packet borrows the issuing Plan's Budget.
pub struct MaterialRecord<'a> {
    request: MaterialRequest,
    packet: MaterialPacket<'a>,
}
impl MaterialRecord<'_> {
    pub fn request(&self) -> MaterialRequest {
        self.request
    }
    pub fn packet(&self) -> &[u8] {
        self.packet.as_bytes()
    }
}

/// Additional ownership only; original current-step issue/export/import overlap
/// remains priced by the shared transport_resources API. These are logical
/// bounded owners, not Python allocator/RSS or arbitrary callback-retention claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchResources {
    native_bytes: usize,
    python_bytes: usize,
}
impl BatchResources {
    pub(super) fn new() -> Result<Self, Error> {
        let descriptors = add(
            mul(
                2 * MATERIAL_BATCH_RECORDS_BOUND,
                size_of::<MaterialRecord<'static>>(),
            )?,
            add(size_of::<Queues<'static>>(), size_of::<Budget>())?,
        )?;
        // Callback originals, inner/outer framing staging, packed HTTP bodies,
        // send/receive copies and decoding. Only one RPC/append is active. Price
        // ten complete body-sized copies plus bounded sender/receiver Python
        // tuple/dict/bytes metadata. The Python adapter must enforce this lifetime.
        let python_bytes = add(
            mul(10, MATERIAL_BATCH_BODY_BYTES_BOUND)?,
            add(mul(2 * MATERIAL_BATCH_RECORDS_BOUND, 4096)?, 4096)?,
        )?;
        Ok(Self {
            native_bytes: add(Self::packet_budget_bytes(), descriptors)?,
            python_bytes,
        })
    }
    pub fn records_bound(&self) -> usize {
        MATERIAL_BATCH_RECORDS_BOUND
    }
    pub fn body_bytes_bound(&self) -> usize {
        MATERIAL_BATCH_BODY_BYTES_BOUND
    }
    pub fn envelope_bytes_bound(&self) -> usize {
        MATERIAL_BATCH_ENVELOPE_BYTES_BOUND
    }
    pub fn record_overhead_bytes(&self) -> usize {
        MATERIAL_BATCH_RECORD_OVERHEAD_BYTES
    }
    pub fn native_reservation_bytes(&self) -> usize {
        self.native_bytes
    }
    pub fn python_overlap_bytes(&self) -> usize {
        self.python_bytes
    }
    pub(super) fn packet_budget_bytes() -> usize {
        2 * MATERIAL_BATCH_BODY_BYTES_BOUND
    }
}

pub(super) fn record_bytes(packet: usize) -> Result<usize, Error> {
    add(
        MATERIAL_BATCH_RECORD_OVERHEAD_BYTES + CALLBACK_METADATA_BYTES_BOUND,
        packet,
    )
}

/// The Budget is owned by prepare(), outside Program::visit_material(). Consuming
/// packet handoff admits the existing backing here before its original charge
/// retires; no packet borrowing a visit's Budget escapes.
pub(super) struct Queues<'a> {
    budget: &'a Budget,
    records: [Vec<MaterialRecord<'a>>; 2],
    body_bytes: [usize; 2],
    accepted: [usize; 2],
}
impl<'a> Queues<'a> {
    pub(super) fn new(budget: &'a Budget) -> Result<Self, Error> {
        let mut records = [Vec::new(), Vec::new()];
        for queue in &mut records {
            queue
                .try_reserve_exact(MATERIAL_BATCH_RECORDS_BOUND)
                .map_err(|_| Error::Allocation)?;
            if queue.capacity() != MATERIAL_BATCH_RECORDS_BOUND {
                return Err(Error::Resource);
            }
        }
        Ok(Self {
            budget,
            records,
            body_bytes: [MATERIAL_BATCH_ENVELOPE_BYTES_BOUND; 2],
            accepted: [0; 2],
        })
    }
    /// Check the public packet bound BEFORE issuing the next pair. A large
    /// legitimate packet must use the original singleton path after flush().
    pub(super) fn singleton(bound: usize) -> Result<bool, Error> {
        Ok(
            add(MATERIAL_BATCH_ENVELOPE_BYTES_BOUND, record_bytes(bound)?)?
                > MATERIAL_BATCH_BODY_BYTES_BOUND,
        )
    }
    pub(super) fn needs_flush(&self, bound: usize) -> Result<bool, Error> {
        let size = record_bytes(bound)?;
        Ok(self
            .records
            .iter()
            .any(|q| q.len() == MATERIAL_BATCH_RECORDS_BOUND)
            || self.body_bytes.iter().any(|&bytes| {
                bytes
                    .checked_add(size)
                    .is_none_or(|n| n > MATERIAL_BATCH_BODY_BYTES_BOUND)
            }))
    }
    pub(super) fn full(&self) -> bool {
        self.records
            .iter()
            .any(|q| q.len() == MATERIAL_BATCH_RECORDS_BOUND)
            || self.body_bytes.contains(&MATERIAL_BATCH_BODY_BYTES_BOUND)
    }
    pub(super) fn push_pair(
        &mut self,
        requests: [MaterialRequest; 2],
        packets: [MaterialPacket<'_>; 2],
    ) -> Result<(), Error> {
        let mut next = self.body_bytes;
        for role in 0..2 {
            if requests[role].role != role as u8
                || packets[role].as_bytes().len() > requests[role].packet_bytes_bound
                || self.records[role].len() >= MATERIAL_BATCH_RECORDS_BOUND
            {
                return Err(Error::Context);
            }
            // Bound actual backing as well as wire bytes, without relaxing the
            // existing per-role body or aggregate packet Budget caps. Ordinary
            // exporters use exact capacity; spare capacity may reject earlier.
            next[role] = add(next[role], record_bytes(packets[role].capacity_bytes())?)?;
            if next[role] > MATERIAL_BATCH_BODY_BYTES_BOUND {
                return Err(Error::Resource);
            }
        }
        // Both destination admissions must succeed before either queue changes.
        // A failed second handoff erases/refunds the first temporary owner.
        let [left, right] = packets;
        let transferred = [
            left.into_budget(self.budget)?,
            right.into_budget(self.budget)?,
        ];
        for (role, packet) in transferred.into_iter().enumerate() {
            self.records[role].push(MaterialRecord {
                request: requests[role],
                packet,
            });
        }
        self.body_bytes = next;
        Ok(())
    }
    pub(super) fn flush(&mut self, transport: &mut impl DealerTransport) -> Result<(), Error> {
        if self.records[0].len() != self.records[1].len() {
            return Err(Error::Context);
        }
        if self.records[0].is_empty() {
            return Ok(());
        }
        for role in 0..2 {
            transport.check_active()?;
            transport.deliver_batch(role as u8, &self.records[role])?;
            transport.check_active()?;
        }
        // Neither role advances to a fresh batch until BOTH complete ACKs.
        for role in 0..2 {
            self.accepted[role] = add(self.accepted[role], self.records[role].len())?;
        }
        self.clear();
        Ok(())
    }
    pub(super) fn accept_singleton(&mut self) -> Result<(), Error> {
        if self.records.iter().any(|q| !q.is_empty()) {
            return Err(Error::Context);
        }
        for accepted in &mut self.accepted {
            *accepted = add(*accepted, 1)?;
        }
        Ok(())
    }
    pub(super) fn complete(&self, requests: usize) -> bool {
        self.records.iter().all(Vec::is_empty) && self.accepted == [requests; 2]
    }
    pub(super) fn clear(&mut self) {
        for queue in &mut self.records {
            queue.clear(); // Shared MaterialPacket Drop erases/refunds each backing.
        }
        self.body_bytes = [MATERIAL_BATCH_ENVELOPE_BYTES_BOUND; 2];
    }
}
