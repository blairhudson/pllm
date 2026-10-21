//! Explicit source-stage grouping of an unchanged scalar claim schedule.
//! The compiler owns the proof that each descriptor is one complete source
//! operation. Core validation only proves exact homogeneous scalar coverage.

use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use zeroize::Zeroizing;

use super::publication::{PublicationPermit, PROCESS_PUBLICATIONS};
use super::window::wire::{batch_owner_control_bytes, batch_scratch_charge, WireBytes, WirePermit};
use super::{
    payload, Capacity, Error, Publication, PublicationOwnership, PublicationResources,
    SharingProfile, Step,
};

const PLAN_DOMAIN: &[u8] = b"pllm.gkm.source-stage-batch.v1";
const SCALAR_DOMAIN: &[u8] = b"pllm.gkm.batch-scalar-schedule.v1";

/// PLGKMB01: magic(8), kind/profile(2), numeric/publication/session(96),
/// sequence(8), context/scalar-plan(64), scalar-count(4), batch-plan(32),
/// batch-ordinal/first-claim/count(12), step(33), artifact(32), role(1),
/// request-id(32), dimensions(8), total encoded length(8). Rows follow as
/// [u64 LE row byte length, row payload]. Scalar framing is not extended.
pub(super) const AGGREGATE_HEADER_BYTES: usize = 340;
const SCALAR_HEADER_BYTES: usize = 296;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchResources {
    pub max_aggregate_bytes: usize,
    pub retained_frame_bytes: usize,
    pub transport_scratch_bytes: usize,
    pub extra_native_owner_bytes: usize,
    pub extra_wire_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchWireSizes {
    pub request_bytes: [usize; 2],
    pub reply_bytes: [usize; 2],
}

pub(super) fn aggregate_length(rows: u32, stride: usize) -> Result<usize, Error> {
    add(AGGREGATE_HEADER_BYTES, mul(rows as usize, add(8, stride)?)?)
}

pub(super) fn row_payload_lengths(
    step: Step,
    input: usize,
    output: usize,
) -> Result<[usize; 2], Error> {
    Ok(match step {
        Step::Linear(_) => [mul(input, 8)?, mul(output, 8)?],
        Step::Lookup => [input, mul(output, 4)?],
    })
}

impl BatchWireSizes {
    fn new(
        rows: u32,
        request: usize,
        reply: usize,
        profile: SharingProfile,
    ) -> Result<Self, Error> {
        Ok(Self {
            request_bytes: [
                aggregate_length(rows, request)?,
                aggregate_length(
                    rows,
                    if profile == SharingProfile::FullShares {
                        request
                    } else {
                        0
                    },
                )?,
            ],
            reply_bytes: [aggregate_length(rows, reply)?; 2],
        })
    }
}

/// Non-cloneable source-/schedule-/profile-bound reservation. Execution entry
/// points consume this owner, never accept caller-authored resources.
/// Buffer owners and aggregate lease guards retain its Arc through destruction.
pub struct BatchAdmission {
    pub(super) publication: [u8; 32],
    pub(super) plan: BatchPlan,
    pub(super) profile: SharingProfile,
    pub(super) capacity: Capacity,
    resources: BatchResources,
    payload_live: AtomicBool,
    // Last: plan metadata and all native/wire owners must die before refunds.
    _native: PublicationPermit,
    _wire_credit: WirePermit,
}

struct AdmittedBatchPlan {
    plan: BatchPlan,
    native: PublicationPermit,
}

impl BatchAdmission {
    pub fn new(
        publication: &Publication,
        plan: BatchPlan,
        profile: SharingProfile,
    ) -> Result<Self, Error> {
        if publication.resources.ownership != PublicationOwnership::StreamedSingleWindow {
            return Err(Error::Binding);
        }
        let resources = BatchResources::for_stage_iter(
            publication
                .matrices
                .iter()
                .map(|m| (m.id, m.rows, m.columns)),
            publication.row_hashes.len(),
            publication.width,
            &plan,
            profile,
            publication.capacity,
        )?;
        let native = PublicationPermit::acquire(
            &PROCESS_PUBLICATIONS,
            resources.extra_native_owner_bytes,
            publication.capacity,
        )?;
        let owned = AdmittedBatchPlan { plan, native };
        // Aggregate allocation must consume this credit for bytes above the
        // unchanged scalar allowance; it must NOT charge those bytes twice.
        // Scalar host envelope already contains 8F. Selected execution replaces
        // that transport reservation with R+S; the serialized delta remains
        // (R-4F)+(S-4F). Buffer permits consume this credit, never reserve again.
        let wire_credit = WirePermit::acquire(
            add(
                resources.retained_frame_bytes,
                resources.transport_scratch_bytes,
            )?,
            publication.capacity,
        )?;
        Ok(Self {
            publication: publication.digest,
            plan: owned.plan,
            profile,
            capacity: publication.capacity,
            resources,
            payload_live: AtomicBool::new(false),
            _native: owned.native,
            _wire_credit: wire_credit,
        })
    }

    pub fn resources(&self) -> &BatchResources {
        &self.resources
    }

    fn validate_publication(
        &self,
        publication: &Publication,
        profile: SharingProfile,
    ) -> Result<(), Error> {
        if self.publication != publication.digest
            || self.capacity != publication.capacity
            || self.profile != profile
            || publication.resources.ownership != PublicationOwnership::StreamedSingleWindow
        {
            return Err(Error::Binding);
        }
        Ok(())
    }

    pub fn validate_binding(
        &self,
        publication: &Publication,
        scalar_plan: &[Step],
        profile: SharingProfile,
    ) -> Result<(), Error> {
        self.validate_publication(publication, profile)?;
        self.plan.validate_scalar_plan(scalar_plan)
    }

    pub(super) fn validate_wire_binding(
        &self,
        publication: &Publication,
        schedule: &[u8],
        count: u32,
        profile: SharingProfile,
    ) -> Result<(), Error> {
        self.validate_publication(publication, profile)?;
        if count != self.plan.scalar_count || schedule.len() != mul(count as usize, 33)? {
            return Err(Error::Binding);
        }
        let mut hash = Sha256::new();
        hash.update(SCALAR_DOMAIN);
        hash.update(count.to_le_bytes());
        hash.update(schedule);
        if <[u8; 32]>::from(hash.finalize()) != self.plan.scalar_digest {
            return Err(Error::Binding);
        }
        Ok(())
    }
}

impl BatchResources {
    /// Metadata only; one entry per descriptor, in canonical plan order.
    pub fn descriptor_wire_sizes(
        stages: &[([u8; 32], usize, usize)],
        vocabulary: usize,
        width: usize,
        plan: &BatchPlan,
        profile: SharingProfile,
        capacity: Capacity,
    ) -> Result<Vec<BatchWireSizes>, Error> {
        Self::for_streamed_single_window(stages, vocabulary, width, plan, profile, capacity)?;
        let mut result = payload::capacity(plan.descriptors().len())?;
        for descriptor in plan.descriptors() {
            // Geometry was checked by the exact same estimation traversal above.
            let (input, output) = match descriptor.step {
                Step::Lookup => (vocabulary.div_ceil(8), width),
                Step::Linear(id) => {
                    let (_, rows, columns) = stages
                        .iter()
                        .find(|(stage, _, _)| *stage == id)
                        .ok_or(Error::Binding)?;
                    (*columns, *rows)
                }
            };
            let [request, reply] = row_payload_lengths(descriptor.step, input, output)?;
            result.push(BatchWireSizes::new(
                descriptor.rows,
                request,
                reply,
                profile,
            )?);
        }
        Ok(result)
    }
    pub fn for_streamed_single_window(
        stages: &[([u8; 32], usize, usize)],
        vocabulary: usize,
        width: usize,
        plan: &BatchPlan,
        profile: SharingProfile,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        Self::for_stage_iter(
            stages.iter().copied(),
            vocabulary,
            width,
            plan,
            profile,
            capacity,
        )
    }

    fn for_stage_iter(
        stages: impl ExactSizeIterator<Item = ([u8; 32], usize, usize)> + Clone,
        vocabulary: usize,
        width: usize,
        plan: &BatchPlan,
        profile: SharingProfile,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        let base = PublicationResources::for_shape_iter_owned(
            stages.clone().map(|(_, rows, columns)| (rows, columns)),
            vocabulary,
            width,
            capacity,
            PublicationOwnership::StreamedSingleWindow,
        )?;
        for (index, (id, _, _)) in stages.clone().enumerate() {
            if stages
                .clone()
                .take(index)
                .any(|(previous, _, _)| previous == id)
            {
                return Err(Error::Binding);
            }
        }
        if plan.scalar_count as usize > capacity.max_window_steps() {
            return Err(Error::Limit);
        }
        let f = capacity.max_wire_frame_bytes();
        let schedule_bytes = mul(plan.scalar_count as usize, 33)?;
        let start_body = add(
            schedule_bytes,
            if profile == SharingProfile::Aes128WindowV1 {
                32
            } else {
                0
            },
        )?;
        let start_bytes = add(SCALAR_HEADER_BYTES, start_body)?;
        if start_bytes > f {
            return Err(Error::Limit);
        }
        let mut max_frame_charge = WireBytes::charge(start_bytes)?;
        let mut max_scratch_charge = batch_scratch_charge(start_body)?;
        let mut max_aggregate_bytes = 0;
        let mut max_native_payload = 0;
        for descriptor in plan.descriptors() {
            let (input_width, output_width) = match descriptor.step {
                Step::Linear(id) => {
                    let (_, rows, columns) = stages
                        .clone()
                        .find(|(stage, _, _)| *stage == id)
                        .ok_or(Error::Binding)?;
                    (columns, rows)
                }
                Step::Lookup => (vocabulary.div_ceil(8), width),
            };
            let [request_row, reply_row] =
                row_payload_lengths(descriptor.step, input_width, output_width)?;
            let wire = BatchWireSizes::new(descriptor.rows, request_row, reply_row, profile)?;
            for row in [request_row, reply_row] {
                if add(SCALAR_HEADER_BYTES, row)? > f {
                    return Err(Error::Limit);
                }
            }
            for aggregate in wire.request_bytes.into_iter().chain(wire.reply_bytes) {
                max_aggregate_bytes = max_aggregate_bytes.max(aggregate);
                max_frame_charge = max_frame_charge.max(WireBytes::charge(aggregate)?);
            }
            // Only full reply freeze and row decoding are transport scratch.
            // Contiguous worker input/output buffers are native owners below.
            // Two complete replies stay frozen throughout all independent row
            // challenges, alongside at most two row-decode scratch guards.
            max_scratch_charge = max_scratch_charge.max(batch_scratch_charge(wire.reply_bytes[0])?);
            let layout = BatchPayloadLayout::new(descriptor, input_width, output_width, profile)?;
            let worker = WorkerBatchPayloadLayout::new(descriptor, input_width, output_width)?;
            max_native_payload = max_native_payload
                .max(layout.payload_bytes()?)
                .max(worker.payload_bytes()?);
        }
        let descriptor_bytes = mul(plan.descriptors.len(), size_of::<BatchDescriptor>())?;
        // Arc control words are the only out-of-line fixed admission allocation.
        // Scalar row preparation/verification remains in the original envelope;
        // all additional flat buffers below are priced in full, not via a factor
        // applied to the scalar reservation. Previous batch buffers must be
        // destroyed before constructing the next BatchPayload owner.
        let control = add(
            add(size_of::<BatchAdmission>(), 2 * size_of::<usize>())?,
            batch_owner_control_bytes()?,
        )?;
        let extra_native_owner_bytes = add(add(descriptor_bytes, control)?, max_native_payload)?;
        if add(base.admitted_payload_bytes, extra_native_owner_bytes)?
            > capacity.max_publication_bytes()
            || add(base.max_window_payload_bytes, extra_native_owner_bytes)?
                > capacity.max_window_bytes()
        {
            return Err(Error::Limit);
        }
        let baseline = mul(4, f)?;
        let retained_frame_bytes = mul(4, f.max(max_frame_charge))?;
        let transport_scratch_bytes = mul(4, f.max(max_scratch_charge))?;
        let extra_wire_bytes = add(
            retained_frame_bytes
                .checked_sub(baseline)
                .ok_or(Error::Limit)?,
            transport_scratch_bytes
                .checked_sub(baseline)
                .ok_or(Error::Limit)?,
        )?;
        if add(mul(2, baseline)?, extra_wire_bytes)? > capacity.max_process_wire_bytes() {
            return Err(Error::Limit);
        }
        Ok(Self {
            max_aggregate_bytes,
            retained_frame_bytes,
            transport_scratch_bytes,
            extra_native_owner_bytes,
            extra_wire_bytes,
        })
    }
}

fn add(left: usize, right: usize) -> Result<usize, Error> {
    left.checked_add(right).ok_or(Error::Limit)
}
fn mul(left: usize, right: usize) -> Result<usize, Error> {
    left.checked_mul(right).ok_or(Error::Limit)
}

/// Concrete extra owner: flattened private inputs/selectors/shares plus complete
/// tentative batch output. One ordinary scalar pending/verification workspace
/// remains native-priced; these buffers coexist with it and are charged in FULL.
pub(super) struct BatchPayload {
    pub(super) input: Zeroizing<Vec<i64>>,
    pub(super) tokens: Zeroizing<Vec<usize>>,
    pub(super) shares: [Zeroizing<Vec<u8>>; 2],
    pub(super) values: Zeroizing<Vec<f32>>,
    pub(super) signed: Zeroizing<Vec<i64>>,
    _admission: BatchPayloadPermit,
}

struct BatchPayloadPermit(Arc<BatchAdmission>);
impl BatchPayloadPermit {
    fn acquire(admission: &Arc<BatchAdmission>) -> Result<Self, Error> {
        admission
            .payload_live
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::Limit)?;
        Ok(Self(Arc::clone(admission)))
    }
}
impl Drop for BatchPayloadPermit {
    fn drop(&mut self) {
        self.0.payload_live.store(false, Ordering::Release);
    }
}

struct BatchPayloadLayout {
    input: usize,
    tokens: usize,
    shares: [usize; 2],
    values: usize,
    signed: usize,
}

/// Exact simultaneous worker vectors, including decoded/expanded contiguous
/// shares. No scalar response slots, row encoding or scalar scratch reused.
pub(super) struct WorkerBatchPayloadLayout {
    linear_input: usize,
    linear_output: usize,
    selectors: usize,
    bytes: usize,
}
impl WorkerBatchPayloadLayout {
    fn new(descriptor: &BatchDescriptor, input: usize, output: usize) -> Result<Self, Error> {
        let batch = descriptor.rows as usize;
        Ok(match descriptor.step {
            Step::Linear(_) => Self {
                linear_input: mul(batch, input)?,
                linear_output: mul(batch, output)?,
                selectors: 0,
                bytes: 0,
            },
            Step::Lookup => Self {
                linear_input: 0,
                linear_output: 0,
                selectors: mul(batch, input)?,
                bytes: mul(mul(batch, output)?, 4)?,
            },
        })
    }
    fn payload_bytes(&self) -> Result<usize, Error> {
        add(
            mul(add(self.linear_input, self.linear_output)?, 8)?,
            add(self.selectors, self.bytes)?,
        )
    }
}

pub(super) struct WorkerBatchPayload {
    pub(super) linear_input: Zeroizing<Vec<u64>>,
    pub(super) linear_output: Zeroizing<Vec<u64>>,
    pub(super) selectors: Zeroizing<Vec<u8>>,
    pub(super) bytes: Zeroizing<Vec<u8>>,
    _admission: BatchPayloadPermit,
}
impl WorkerBatchPayload {
    pub(super) fn allocate(
        admission: &Arc<BatchAdmission>,
        descriptor: &BatchDescriptor,
        input: usize,
        output: usize,
    ) -> Result<Self, Error> {
        let permit = BatchPayloadPermit::acquire(admission)?;
        let layout = WorkerBatchPayloadLayout::new(descriptor, input, output)?;
        let linear_input = Zeroizing::new(payload::filled(layout.linear_input, 0u64)?);
        let linear_output = Zeroizing::new(payload::filled(layout.linear_output, 0u64)?);
        let selectors = Zeroizing::new(payload::filled(layout.selectors, 0u8)?);
        let bytes = Zeroizing::new(payload::filled(layout.bytes, 0u8)?);
        if linear_input.capacity() != layout.linear_input
            || linear_output.capacity() != layout.linear_output
            || selectors.capacity() != layout.selectors
            || bytes.capacity() != layout.bytes
        {
            return Err(Error::Allocation);
        }
        Ok(Self {
            linear_input,
            linear_output,
            selectors,
            bytes,
            _admission: permit,
        })
    }
}

impl BatchPayloadLayout {
    fn new(
        descriptor: &BatchDescriptor,
        input_width: usize,
        output_width: usize,
        profile: SharingProfile,
    ) -> Result<Self, Error> {
        let rows = descriptor.rows as usize;
        let second = profile == SharingProfile::FullShares;
        match descriptor.step {
            Step::Linear(_) => {
                let input = mul(rows, input_width)?;
                let share = mul(input, 8)?;
                let output = mul(rows, output_width)?;
                Ok(Self {
                    input,
                    tokens: 0,
                    shares: [share, if second { share } else { 0 }],
                    values: output,
                    signed: output,
                })
            }
            Step::Lookup => {
                let share = mul(rows, input_width)?;
                Ok(Self {
                    input: 0,
                    tokens: rows,
                    shares: [share, if second { share } else { 0 }],
                    values: mul(rows, output_width)?,
                    signed: 0,
                })
            }
        }
    }

    fn payload_bytes(&self) -> Result<usize, Error> {
        let mut bytes = add(
            mul(self.input, size_of::<i64>())?,
            mul(self.tokens, size_of::<usize>())?,
        )?;
        bytes = add(bytes, add(self.shares[0], self.shares[1])?)?;
        bytes = add(bytes, mul(self.values, size_of::<f32>())?)?;
        add(bytes, mul(self.signed, size_of::<i64>())?)
    }
}

impl BatchPayload {
    pub(super) fn allocate(
        publication: &Publication,
        admission: Arc<BatchAdmission>,
        ordinal: usize,
    ) -> Result<Self, Error> {
        if publication.digest != admission.publication || publication.capacity != admission.capacity
        {
            return Err(Error::Binding);
        }
        admission
            .payload_live
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::Limit)?;
        let permit = BatchPayloadPermit(Arc::clone(&admission));
        let descriptor = admission
            .plan
            .descriptors()
            .get(ordinal)
            .ok_or(Error::Schedule)?;
        let (input, output) = match descriptor.step {
            Step::Lookup => (publication.row_hashes.len().div_ceil(8), publication.width),
            Step::Linear(id) => {
                let m = &publication.matrices[publication.matrix_index(&id)?];
                (m.columns, m.rows)
            }
        };
        let layout = BatchPayloadLayout::new(descriptor, input, output, admission.profile)?;
        // Admission is already live. Every field drops before its final guard,
        // including partial construction failure (locals declared after guard).
        let input = Zeroizing::new(payload::filled(layout.input, 0i64)?);
        let tokens = Zeroizing::new(payload::filled(layout.tokens, 0usize)?);
        let left = Zeroizing::new(payload::filled(layout.shares[0], 0u8)?);
        let right = Zeroizing::new(payload::filled(layout.shares[1], 0u8)?);
        let values = Zeroizing::new(payload::filled(layout.values, 0f32)?);
        let signed = Zeroizing::new(payload::filled(layout.signed, 0i64)?);
        if input.capacity() != layout.input
            || tokens.capacity() != layout.tokens
            || left.capacity() != layout.shares[0]
            || right.capacity() != layout.shares[1]
            || values.capacity() != layout.values
            || signed.capacity() != layout.signed
        {
            return Err(Error::Allocation);
        }
        Ok(Self {
            input,
            tokens,
            shares: [left, right],
            values,
            signed,
            _admission: permit,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchDescriptor {
    pub phase_index: u32,
    pub operation_index: u32,
    pub first_claim: u32,
    pub rows: u32,
    pub step: Step,
}

/// Public metadata, not an execution/admission token or a source-bound proof.
/// No scalar plan is copied: its independent fingerprint is checked again by
/// the consuming native execution path against the actual scalar schedule.
pub struct BatchPlan {
    descriptors: Vec<BatchDescriptor>,
    scalar_count: u32,
    scalar_digest: [u8; 32],
    digest: [u8; 32],
}

impl BatchPlan {
    pub fn new(scalar_plan: &[Step], descriptors: &[BatchDescriptor]) -> Result<Self, Error> {
        if scalar_plan.is_empty()
            || scalar_plan.len() > Capacity::Decoder.max_window_steps()
            || descriptors.is_empty()
            || descriptors.len() > scalar_plan.len()
        {
            return Err(Error::Limit);
        }
        let scalar_count = u32::try_from(scalar_plan.len()).map_err(|_| Error::Limit)?;
        let mut first = 0u32;
        for descriptor in descriptors {
            if descriptor.rows == 0 || descriptor.first_claim != first {
                return Err(Error::Schedule);
            }
            let end = first.checked_add(descriptor.rows).ok_or(Error::Limit)?;
            let covered = scalar_plan
                .get(first as usize..end as usize)
                .ok_or(Error::Schedule)?;
            if covered.iter().any(|step| *step != descriptor.step) {
                return Err(Error::Schedule);
            }
            first = end;
        }
        if first != scalar_count {
            return Err(Error::Schedule);
        }
        let mut hash = Sha256::new();
        hash.update(PLAN_DOMAIN);
        hash.update(scalar_count.to_le_bytes());
        for step in scalar_plan {
            hash.update(step.canonical_bytes());
        }
        hash.update(
            u32::try_from(descriptors.len())
                .map_err(|_| Error::Limit)?
                .to_le_bytes(),
        );
        for descriptor in descriptors {
            hash.update(descriptor.phase_index.to_le_bytes());
            hash.update(descriptor.operation_index.to_le_bytes());
            hash.update(descriptor.first_claim.to_le_bytes());
            hash.update(descriptor.rows.to_le_bytes());
            hash.update(descriptor.step.canonical_bytes());
        }
        let digest = hash.finalize().into();
        // Validate/hash before the first fallible descriptor allocation. Caller
        // metadata exists before publication; admission later charges its owner.
        let descriptors = payload::copy(descriptors)?;
        if descriptors.capacity() != descriptors.len() {
            return Err(Error::Allocation);
        }
        Ok(Self {
            descriptors,
            scalar_count,
            scalar_digest: scalar_digest(scalar_plan)?,
            digest,
        })
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn descriptors(&self) -> &[BatchDescriptor] {
        &self.descriptors
    }

    pub(super) fn validate_scalar_plan(&self, plan: &[Step]) -> Result<(), Error> {
        if plan.len() != self.scalar_count as usize || scalar_digest(plan)? != self.scalar_digest {
            return Err(Error::Binding);
        }
        Ok(())
    }
}

fn scalar_digest(plan: &[Step]) -> Result<[u8; 32], Error> {
    let mut hash = Sha256::new();
    hash.update(SCALAR_DOMAIN);
    hash.update(
        u32::try_from(plan.len())
            .map_err(|_| Error::Limit)?
            .to_le_bytes(),
    );
    for step in plan {
        hash.update(step.canonical_bytes());
    }
    Ok(hash.finalize().into())
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
