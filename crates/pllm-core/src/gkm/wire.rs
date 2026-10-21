//! Party-local full-share and explicitly selected seeded transport owners.
//! See `wire-api.md` for the original full-share adapter
//! contract. Bytes are framing, not authentication or encryption. In particular,
//! the authenticated start frame and publication pin must come from the configured
//! client/owner channels, never from an unauthenticated peer's self-description.

use std::sync::atomic::{AtomicU8, AtomicUsize};
use std::sync::Mutex;

use super::super::{Stage, NUMERIC_PROFILE};
use super::*;

/// Capsule process allowance, also subject to the shared Decoder process cap.
pub const MAX_PROCESS_WIRE_BYTES: usize = 8 << 20;
pub const MAX_PUBLICATION_WIRE_BYTES: usize = 512 << 10;
pub const MAX_WIRE_FRAME_BYTES: usize = 8192;

static WIRE_BYTES: AtomicUsize = AtomicUsize::new(0);
static CAPSULE_WIRE_BYTES: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
std::thread_local! { static SCALAR_EVALUATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

/// Fixed-size synchronized window/publication accounting. Payload admission is
/// atomic across both dimensions, including pair preflight; no queue allocation.
pub(crate) struct WireQuota(Mutex<(usize, usize)>);

impl WireQuota {
    pub(crate) fn new() -> Self {
        Self(Mutex::new((0, 0)))
    }

    fn reserve(
        &self,
        count: usize,
        bytes: usize,
        max_count: usize,
        max_bytes: usize,
    ) -> Result<(), Error> {
        let mut used = self.0.lock().map_err(|_| Error::Retired)?;
        let next_count = used
            .0
            .checked_add(count)
            .filter(|&n| n <= max_count)
            .ok_or(Error::Limit)?;
        let next_bytes = used
            .1
            .checked_add(bytes)
            .filter(|&n| n <= max_bytes)
            .ok_or(Error::Limit)?;
        *used = (next_count, next_bytes);
        Ok(())
    }

    fn release(&self, bytes: usize) {
        // Destruction must refund even if an unrelated panic poisoned the mutex.
        let mut used = self.0.lock().unwrap_or_else(|error| error.into_inner());
        used.0 -= 1;
        used.1 -= bytes;
    }
}

fn streamed(lease: &Lease) -> bool {
    lease.budget.publication.resources.ownership
        == super::super::PublicationOwnership::StreamedSingleWindow
}

struct FramePermit {
    lease: Arc<Lease>,
    bytes: usize,
    length: usize,
}

impl FramePermit {
    fn reserve(lease: &Arc<Lease>, lengths: &[usize]) -> Result<(), Error> {
        Self::reserve_kind(lease, lengths, false)
    }

    fn reserve_kind(lease: &Arc<Lease>, lengths: &[usize], aggregate: bool) -> Result<(), Error> {
        lease.ensure_open()?;
        let result = (|| {
            let limit = lease.budget.publication.capacity.max_wire_frame_bytes();
            let admission = lease.batch_admission()?;
            let length_limit = if aggregate {
                admission
                    .as_ref()
                    .ok_or(Error::Binding)?
                    .resources()
                    .max_aggregate_bytes
            } else {
                limit
            };
            let mut bytes = 0usize;
            for &length in lengths {
                if length > length_limit {
                    return Err(Error::Limit);
                }
                bytes = bytes
                    .checked_add(WireBytes::charge(length)?)
                    .ok_or(Error::Limit)?;
            }
            if streamed(lease) {
                lease.frames.reserve(
                    lengths.len(),
                    bytes,
                    4,
                    match admission {
                        Some(admission) => admission.resources().retained_frame_bytes,
                        None => limit.checked_mul(4).ok_or(Error::Limit)?,
                    },
                )?;
            }
            Ok(())
        })();
        if result.is_err() {
            lease.poisoned.store(true, Ordering::Release);
        }
        result
    }

    fn aggregate_pair(lease: &Arc<Lease>, lengths: [usize; 2]) -> Result<[Self; 2], Error> {
        Self::reserve_kind(lease, &lengths, true)?;
        Ok(lengths.map(|length| Self {
            lease: Arc::clone(lease),
            bytes: WireBytes::charge(length).expect("preflighted aggregate"),
            length,
        }))
    }

    fn aggregate(lease: &Arc<Lease>, length: usize) -> Result<Self, Error> {
        Self::reserve_kind(lease, &[length], true)?;
        Ok(Self {
            lease: Arc::clone(lease),
            bytes: WireBytes::charge(length)?,
            length,
        })
    }

    fn single(lease: &Arc<Lease>, length: usize) -> Result<Self, Error> {
        Self::reserve(lease, &[length])?;
        Ok(Self {
            lease: Arc::clone(lease),
            bytes: if streamed(lease) {
                WireBytes::charge(length)?
            } else {
                0
            },
            length,
        })
    }

    fn pair(lease: &Arc<Lease>, lengths: [usize; 2]) -> Result<[Self; 2], Error> {
        Self::reserve(lease, &lengths)?;
        Ok(lengths.map(|length| Self {
            lease: Arc::clone(lease),
            bytes: if streamed(lease) {
                WireBytes::charge(length).expect("preflighted charge")
            } else {
                0
            },
            length,
        }))
    }
}

impl Drop for FramePermit {
    fn drop(&mut self) {
        if self.bytes != 0 {
            self.lease.frames.release(self.bytes);
        }
    }
}

pub(crate) struct PublicationFramePermit {
    quota: Arc<WireQuota>,
    bytes: usize,
}

impl PublicationFramePermit {
    pub(crate) fn acquire(
        quota: &Arc<WireQuota>,
        length: usize,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        let bytes = WireBytes::charge(length)?;
        quota.reserve(1, bytes, 1, capacity.max_publication_wire_bytes())?;
        Ok(Self {
            quota: Arc::clone(quota),
            bytes,
        })
    }
}

impl Drop for PublicationFramePermit {
    fn drop(&mut self) {
        self.quota.release(self.bytes);
    }
}

struct ScratchPermit {
    lease: Arc<Lease>,
    bytes: usize,
}

impl ScratchPermit {
    fn acquire(lease: &Arc<Lease>, bytes: usize) -> Result<Option<Self>, Error> {
        lease.ensure_open()?;
        if !streamed(lease) {
            return Ok(None);
        }
        let admission = lease.batch_admission()?;
        let result = lease
            .budget
            .publication
            .capacity
            .max_wire_frame_bytes()
            .checked_mul(4)
            .ok_or(Error::Limit)
            .and_then(|limit| {
                lease.scratch.reserve(
                    1,
                    bytes,
                    4,
                    admission.as_ref().map_or(limit, |admission| {
                        admission.resources().transport_scratch_bytes
                    }),
                )
            });
        if result.is_err() {
            lease.poisoned.store(true, Ordering::Release);
        }
        result?;
        Ok(Some(Self {
            lease: Arc::clone(lease),
            bytes,
        }))
    }
}

impl Drop for ScratchPermit {
    fn drop(&mut self) {
        self.lease.scratch.release(self.bytes);
    }
}

/// Transport-only duplicate/serialization buffers. Native pending masks,
/// response owners, accumulators, worker schedule and SeededWorker expansion
/// already belong to the complete native window envelope and are NOT charged
/// again here. Borrowed inbound frames require caller admission, not a copy.
struct WireScratch<T: zeroize::Zeroize> {
    values: Zeroizing<Vec<T>>,
    _quota: Option<ScratchPermit>,
}

impl<T: zeroize::Zeroize + Copy> WireScratch<T> {
    pub(crate) fn charge(length: usize) -> Result<usize, Error> {
        length
            .checked_mul(size_of::<T>())
            .and_then(|n| n.checked_add(size_of::<Self>() + 64))
            .ok_or(Error::Limit)
    }
    fn filled(lease: &Arc<Lease>, length: usize, value: T) -> Result<Self, Error> {
        let bytes = Self::charge(length).inspect_err(|_| {
            lease.poisoned.store(true, Ordering::Release);
        })?;
        let quota = ScratchPermit::acquire(lease, bytes)?;
        let values = Zeroizing::new(payload::filled(length, value)?);
        if values.capacity() != length {
            return Err(Error::Allocation);
        }
        Ok(Self {
            values,
            _quota: quota,
        })
    }
}

impl<T: zeroize::Zeroize> std::ops::Deref for WireScratch<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}
impl<T: zeroize::Zeroize> std::ops::DerefMut for WireScratch<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

/// Logical payload accounting, not RSS. Includes wire-owned public source
/// copies, schedules, control/scratch allowances, and every retained output frame.
#[derive(Debug, Clone, Copy)]
pub struct WireResources {
    pub process_live_bytes: usize,
    pub process_limit_bytes: usize,
}

impl WireResources {
    pub fn current() -> Self {
        Self::current_with_capacity(Capacity::Capsule)
    }

    /// Capsule reports its unchanged sub-pool; Decoder reports the complete
    /// process pool, including Capsule reservations. These are overlapping caps.
    pub fn current_with_capacity(capacity: Capacity) -> Self {
        Self {
            process_live_bytes: if capacity == Capacity::Capsule {
                CAPSULE_WIRE_BYTES.load(Ordering::Acquire)
            } else {
                WIRE_BYTES.load(Ordering::Acquire)
            },
            process_limit_bytes: capacity.max_process_wire_bytes(),
        }
    }
}

pub(crate) struct WirePermit {
    bytes: usize,
    capacity: Capacity,
}

impl WirePermit {
    pub(crate) fn acquire(bytes: usize, capacity: Capacity) -> Result<Self, Error> {
        if capacity == Capacity::Capsule {
            CAPSULE_WIRE_BYTES
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                    used.checked_add(bytes)
                        .filter(|&n| n <= MAX_PROCESS_WIRE_BYTES)
                })
                .map_err(|_| Error::Limit)?;
        }
        let result = WIRE_BYTES
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes)
                    .filter(|&n| n <= Capacity::Decoder.max_process_wire_bytes())
            })
            .map_err(|_| Error::Limit);
        if result.is_err() && capacity == Capacity::Capsule {
            CAPSULE_WIRE_BYTES.fetch_sub(bytes, Ordering::AcqRel);
        }
        result?;
        Ok(Self { bytes, capacity })
    }
}

impl Drop for WirePermit {
    fn drop(&mut self) {
        WIRE_BYTES.fetch_sub(self.bytes, Ordering::AcqRel);
        if self.capacity == Capacity::Capsule {
            CAPSULE_WIRE_BYTES.fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

/// Non-cloneable, zeroizing output bytes. Borrow for transport; there is no
/// unleased buffer extraction. Request/schedule frames retain the native window
/// lease even after cancellation. Caller-made copies require caller admission.
pub struct WireBytes {
    bytes: Zeroizing<Vec<u8>>,
    _frame: Option<FramePermit>,
    _publication: Option<PublicationFramePermit>,
    _permit: WirePermit,
}

impl WireBytes {
    pub(crate) fn charge(length: usize) -> Result<usize, Error> {
        length
            .checked_add(size_of::<Self>() + 64)
            .ok_or(Error::Limit)
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn capacity(&self) -> Capacity {
        self._permit.capacity
    }

    fn allocate(length: usize, capacity: Capacity) -> Result<Self, Error> {
        if length > capacity.max_publication_wire_bytes() {
            return Err(Error::Limit);
        }
        let permit = WirePermit::acquire(Self::charge(length)?, capacity)?;
        let bytes = payload::capacity(length)?;
        // Never grow a buffer beyond the capacity admitted before allocation.
        if bytes.capacity() != length {
            return Err(Error::Allocation);
        }
        Ok(Self {
            bytes: Zeroizing::new(bytes),
            _frame: None,
            _publication: None,
            _permit: permit,
        })
    }

    fn allocate_leased(
        length: usize,
        lease: &Arc<Lease>,
        permit: FramePermit,
        aggregate: bool,
    ) -> Result<Self, Error> {
        lease.ensure_open()?;
        if permit.length != length || !Arc::ptr_eq(&permit.lease, lease) {
            return Err(Error::Binding);
        }
        let capacity = lease.budget.publication.capacity;
        let admission = lease.batch_admission()?;
        let maximum = if aggregate {
            admission
                .as_ref()
                .ok_or(Error::Binding)?
                .resources()
                .max_aggregate_bytes
        } else {
            capacity.max_wire_frame_bytes()
        };
        if length > maximum {
            return Err(Error::Limit);
        }
        // Selected lease's fixed R+S reservation pays for every operation frame
        // and scratch owner. General/scalar paths retain original permits.
        let global = WirePermit::acquire(
            if admission.is_some() {
                0
            } else {
                Self::charge(length)?
            },
            capacity,
        )?;
        let bytes = payload::capacity(length)?;
        if bytes.capacity() != length {
            return Err(Error::Allocation);
        }
        Ok(Self {
            bytes: Zeroizing::new(bytes),
            _frame: Some(permit),
            _publication: None,
            _permit: global,
        })
    }

    pub(crate) fn allocate_publication(
        length: usize,
        capacity: Capacity,
        quota: Option<&Arc<WireQuota>>,
    ) -> Result<Self, Error> {
        let reservation = quota
            .map(|quota| PublicationFramePermit::acquire(quota, length, capacity))
            .transpose()?;
        let mut out = Self::allocate(length, capacity)?;
        out._publication = reservation;
        Ok(out)
    }

    pub(crate) fn extend(&mut self, value: &[u8]) {
        assert!(value.len() <= self.bytes.capacity() - self.bytes.len());
        self.bytes.extend_from_slice(value);
    }
}

pub(crate) fn batch_scratch_charge(length: usize) -> Result<usize, Error> {
    WireScratch::<u8>::charge(length)
}

pub(crate) fn batch_owner_control_bytes() -> Result<usize, Error> {
    batch_wire::owner_control_bytes()
}

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }
    pub(crate) fn position(&self) -> usize {
        self.at
    }
    pub(crate) fn take(&mut self, length: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(length).ok_or(Error::Limit)?;
        let value = self.bytes.get(self.at..end).ok_or(Error::Shape)?;
        self.at = end;
        Ok(value)
    }
    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        self.take(N)?.try_into().map_err(|_| Error::Shape)
    }
    pub(crate) fn byte(&mut self) -> Result<u8, Error> {
        Ok(self.array::<1>()?[0])
    }
    pub(crate) fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    pub(crate) fn finish(self) -> Result<(), Error> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::Shape)
        }
    }
}

const MAGIC: &[u8; 8] = b"PLGKM001";
const HEADER: usize = 296;
const START: u8 = 0;
const REQUEST: u8 = 1;
const REPLY: u8 = 2;

fn request_id(binding: &Binding) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"pllm.gkm.request.v1");
    hash.update(binding.publication);
    hash.update(binding.session);
    hash.update(binding.sequence.to_le_bytes());
    hash.update(binding.context);
    hash.update(binding.plan);
    hash.finalize().into()
}

fn shape(publication: &Publication, step: Step) -> Result<(usize, usize), Error> {
    match step {
        Step::Lookup => Ok((publication.row_hashes.len(), publication.width)),
        Step::Linear(id) => {
            let matrix = &publication.matrices[publication.matrix_index(&id)?];
            Ok((matrix.rows, matrix.columns))
        }
    }
}

fn artifact(publication: &Publication, step: Step) -> Result<[u8; 32], Error> {
    match step {
        Step::Lookup => Ok(publication.table_digest),
        Step::Linear(id) => Ok(publication.matrices[publication.matrix_index(&id)?].digest),
    }
}

fn step_bytes(step: Step) -> [u8; 33] {
    step.canonical_bytes()
}

fn read_step(cursor: &mut Cursor<'_>) -> Result<Step, Error> {
    let tag = cursor.byte()?;
    let id = cursor.array()?;
    match tag {
        0 if id == [0; 32] => Ok(Step::Lookup),
        1 => Ok(Step::Linear(id)),
        _ => Err(Error::Binding),
    }
}

fn encode_frame(
    kind: u8,
    binding: Binding,
    dimensions: (usize, usize),
    body: &[u8],
    lease: &Arc<Lease>,
    permit: FramePermit,
) -> Result<WireBytes, Error> {
    lease.ensure_open()?;
    let length = HEADER.checked_add(body.len()).ok_or(Error::Limit)?;
    let capacity = lease.budget.publication.capacity;
    if length > capacity.max_wire_frame_bytes() {
        return Err(Error::Limit);
    }
    if permit.length != length || !Arc::ptr_eq(&permit.lease, lease) {
        return Err(Error::Binding);
    }
    let mut out = WireBytes::allocate_leased(length, lease, permit, false)?;
    out.extend(MAGIC);
    out.extend(&[
        kind,
        if binding.profile == SharingProfile::FullShares {
            0
        } else {
            1
        },
    ]);
    out.extend(&Sha256::digest(NUMERIC_PROFILE));
    out.extend(&binding.publication);
    out.extend(&binding.session);
    out.extend(&binding.sequence.to_le_bytes());
    out.extend(&binding.context);
    out.extend(&binding.plan);
    out.extend(&binding.total.to_le_bytes());
    out.extend(&binding.ordinal.to_le_bytes());
    out.extend(&step_bytes(binding.step));
    out.extend(&binding.artifact);
    out.extend(&[if binding.worker == Worker::Zero { 0 } else { 1 }]);
    out.extend(&request_id(&binding));
    out.extend(&(dimensions.0 as u32).to_le_bytes());
    out.extend(&(dimensions.1 as u32).to_le_bytes());
    out.extend(&(body.len() as u32).to_le_bytes());
    out.extend(body);
    Ok(out)
}

struct Frame<'a> {
    binding: Binding,
    dimensions: (usize, usize),
    body: &'a [u8],
}

impl<'a> Frame<'a> {
    fn decode(bytes: &'a [u8], kind: u8, owner: &Publication) -> Result<Self, Error> {
        if bytes.len() > owner.capacity.max_wire_frame_bytes() {
            return Err(Error::Limit);
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.take(8)? != MAGIC || cursor.byte()? != kind {
            return Err(Error::Binding);
        }
        let profile = match cursor.byte()? {
            0 => SharingProfile::FullShares,
            1 => SharingProfile::Aes128WindowV1,
            _ => return Err(Error::Binding),
        };
        if cursor.take(32)? != Sha256::digest(NUMERIC_PROFILE).as_slice() {
            return Err(Error::Binding);
        }
        let publication = cursor.array()?;
        let session = cursor.array()?;
        let sequence = cursor.u64()?;
        let context = cursor.array()?;
        let plan = cursor.array()?;
        let total = cursor.u32()?;
        let ordinal = cursor.u32()?;
        let step = read_step(&mut cursor)?;
        let artifact = cursor.array()?;
        let worker = match cursor.byte()? {
            0 => Worker::Zero,
            1 => Worker::One,
            _ => return Err(Error::Binding),
        };
        let binding = Binding {
            publication,
            session,
            sequence,
            context,
            plan,
            total,
            ordinal,
            step,
            artifact,
            worker,
            profile,
        };
        if total == 0
            || total as usize > owner.capacity.max_window_steps()
            || ordinal >= total
            || sequence >= MAX_WINDOWS
            || cursor.array::<32>()? != request_id(&binding)
        {
            return Err(Error::Binding);
        }
        let dimensions = (cursor.u32()? as usize, cursor.u32()? as usize);
        if publication != owner.digest || dimensions != shape(owner, step)? {
            return Err(Error::Binding);
        }
        let length = cursor.u32()? as usize;
        let body = cursor.take(length)?;
        cursor.finish()?;
        Ok(Self {
            binding,
            dimensions,
            body,
        })
    }

    fn validate(&self, binding: Binding, publication: &Publication) -> Result<(), Error> {
        if self.binding != binding || self.dimensions != shape(publication, binding.step)? {
            return Err(Error::Binding);
        }
        Ok(())
    }
}

enum ClientState {
    Ready(Window),
    Linear(PendingLinear),
    Lookup(PendingLookup),
}

/// Owns the existing native window/pending transaction across external I/O.
/// Every fallible state transition takes the old state first: ignored errors,
/// duplicate issue/import/start/finish, and cancellation cannot permit retries.
pub struct ClientWireWindow {
    seed_keys: Option<WireSeedKeys>,
    state: Option<ClientState>,
    started: bool,
    request: [u8; 32],
    _permit: WirePermit,
}

struct WireSeedKeys {
    keys: Zeroizing<[[u8; 16]; 2]>,
    _lease: Arc<Lease>,
}

/// Borrowed, unaccepted values for trusted client-local dependent execution.
/// No reusable decoder state or user output may escape before `finish` succeeds.
pub struct WireTentative<'a> {
    pub values: &'a [f32],
    pub signed: &'a [i64],
}

impl Client {
    /// Fresh window keys travel only in worker One's confidential authenticated
    /// start channel. Neither this owner nor returned frames support retries.
    pub fn begin_seeded_wire(
        &self,
        context: [u8; 32],
        plan: &[Step],
    ) -> Result<ClientWireWindow, Error> {
        let (window, keys) = self.seeded_wire_parts(context, plan)?;
        window.into_wire_inner(Some(keys))
    }

    fn seeded_wire_parts(
        &self,
        context: [u8; 32],
        plan: &[Step],
    ) -> Result<(Window, WireSeedKeys), Error> {
        let (window, worker) = self.begin_seeded(context, plan)?;
        let keys = WireSeedKeys {
            keys: worker.transfer_keys(&window._lease)?,
            _lease: Arc::clone(&window._lease),
        };
        Ok((window, keys))
    }

    pub fn begin_wire(&self, context: [u8; 32], plan: &[Step]) -> Result<ClientWireWindow, Error> {
        self.begin(context, plan)?.into_wire()
    }
}

impl Window {
    /// Convert an untouched full-share window. Unpaired seeded windows reject
    /// and retire: use `Client::begin_seeded_wire` for consuming seed transfer.
    pub fn into_wire(self) -> Result<ClientWireWindow, Error> {
        self.into_wire_inner(None)
    }

    fn into_wire_inner(self, keys: Option<WireSeedKeys>) -> Result<ClientWireWindow, Error> {
        self._lease.ensure_open()?;
        if (self.profile == SharingProfile::Aes128WindowV1) != keys.is_some() || self.completed != 0
        {
            return Err(Error::Binding);
        }
        let max_values = self
            .publication
            .matrices
            .iter()
            .map(|m| m.rows.max(m.columns))
            .max()
            .unwrap();
        let permit = WirePermit::acquire(
            if self._lease.batch_admission()?.is_some() {
                0
            } else {
                size_of::<ClientWireWindow>() + 2 * max_values * 8 + self.plan.len() * 33 + 1024
            },
            self.publication.capacity,
        )?;
        Ok(ClientWireWindow {
            request: self.request_id(),
            state: Some(ClientState::Ready(self)),
            seed_keys: keys,
            started: false,
            _permit: permit,
        })
    }
}

impl ClientWireWindow {
    pub fn request_id(&self) -> [u8; 32] {
        self.request
    }

    /// Role-specific starts, exported exactly once before private shares.
    /// Seeded worker One's start contains both purpose keys: its channel MUST
    /// provide confidentiality and authentication. Zero receives only schedule.
    pub fn start_frames(&mut self) -> Result<[WireBytes; 2], Error> {
        // Take even on malformed state: seed material must not survive a failed export.
        let keys = self.seed_keys.take();
        let state = self.state.take().ok_or(Error::Retired)?;
        let ClientState::Ready(window) = state else {
            return Err(Error::Schedule);
        };
        if self.started {
            return Err(Error::Retired);
        }
        let schedule_bytes = window.plan.len().checked_mul(33).ok_or(Error::Limit)?;
        let body_length = schedule_bytes
            .checked_add(if keys.is_some() { 32 } else { 0 })
            .ok_or(Error::Limit)?;
        let [left_permit, right_permit] = FramePermit::pair(
            &window._lease,
            [
                HEADER.checked_add(schedule_bytes).ok_or(Error::Limit)?,
                HEADER.checked_add(body_length).ok_or(Error::Limit)?,
            ],
        )?;
        let mut body = WireScratch::filled(&window._lease, body_length, 0u8)?;
        for (bytes, &step) in body[..schedule_bytes]
            .chunks_exact_mut(33)
            .zip(window.plan.iter())
        {
            bytes.copy_from_slice(&step_bytes(step));
        }
        let first = window.binding(Worker::Zero)?;
        let dimensions = shape(&window.publication, first.step)?;
        let left = encode_frame(
            START,
            first,
            dimensions,
            &body[..window.plan.len() * 33],
            &window._lease,
            left_permit,
        )?;
        if let Some(keys) = keys {
            body[schedule_bytes..schedule_bytes + 16].copy_from_slice(&keys.keys[0]);
            body[schedule_bytes + 16..].copy_from_slice(&keys.keys[1]);
        }
        let right = encode_frame(
            START,
            window.binding(Worker::One)?,
            dimensions,
            &body,
            &window._lease,
            right_permit,
        )?;
        self.started = true;
        self.state = Some(ClientState::Ready(window));
        Ok([left, right])
    }

    fn take_ready(&mut self) -> Result<Window, Error> {
        let state = self.state.take().ok_or(Error::Retired)?;
        if !self.started {
            return Err(Error::Schedule);
        }
        match state {
            ClientState::Ready(window) => Ok(window),
            _ => Err(Error::Schedule),
        }
    }

    pub fn linear(&mut self, input: &[i64]) -> Result<[WireBytes; 2], Error> {
        let pending = self.take_ready()?.linear(input)?;
        let length = |worker| {
            pending
                .share(worker)
                .values()
                .unwrap_or(&[])
                .len()
                .checked_mul(8)
                .and_then(|n| n.checked_add(HEADER))
                .ok_or(Error::Limit)
        };
        let [left_permit, right_permit] = FramePermit::pair(
            &pending.window._lease,
            [length(Worker::Zero)?, length(Worker::One)?],
        )?;
        let encode = |worker, permit| {
            let share = pending.share(worker);
            let values = share.values().unwrap_or(&[]);
            let mut bytes = WireScratch::filled(
                share.lease,
                values.len().checked_mul(8).ok_or(Error::Limit)?,
                0u8,
            )?;
            for (out, value) in bytes.chunks_exact_mut(8).zip(values) {
                out.copy_from_slice(&value.to_le_bytes());
            }
            encode_frame(
                REQUEST,
                share.binding(),
                shape(&pending.window.publication, share.binding().step)?,
                &bytes[..values.len() * 8],
                share.lease,
                permit,
            )
        };
        let left = encode(Worker::Zero, left_permit)?;
        let right = encode(Worker::One, right_permit)?;
        self.state = Some(ClientState::Linear(pending));
        Ok([left, right])
    }

    pub fn lookup(&mut self, token: usize) -> Result<[WireBytes; 2], Error> {
        let pending = self.take_ready()?.lookup(token)?;
        let length = |worker| {
            HEADER
                .checked_add(pending.share(worker).selector().unwrap_or(&[]).len())
                .ok_or(Error::Limit)
        };
        let [left_permit, right_permit] = FramePermit::pair(
            &pending.window._lease,
            [length(Worker::Zero)?, length(Worker::One)?],
        )?;
        let encode = |worker, permit| {
            let share = pending.share(worker);
            encode_frame(
                REQUEST,
                share.binding(),
                shape(&pending.window.publication, Step::Lookup)?,
                share.selector().unwrap_or(&[]),
                share.lease,
                permit,
            )
        };
        let left = encode(Worker::Zero, left_permit)?;
        let right = encode(Worker::One, right_permit)?;
        self.state = Some(ClientState::Lookup(pending));
        Ok([left, right])
    }

    pub fn accept_replies(&mut self, zero: &[u8], one: &[u8]) -> Result<WireTentative<'_>, Error> {
        let state = self.state.take().ok_or(Error::Retired)?;
        let window = match state {
            ClientState::Linear(pending) => {
                let decode = |worker, bytes| {
                    let share = pending.share(worker);
                    let frame = Frame::decode(bytes, REPLY, &pending.window.publication)?;
                    frame.validate(share.binding(), &pending.window.publication)?;
                    if frame.body.len() != share.rows * 8 {
                        return Err(Error::Shape);
                    }
                    let mut values = WireScratch::filled(share.lease, share.rows, 0u64)?;
                    for (out, bytes) in values.iter_mut().zip(frame.body.chunks_exact(8)) {
                        *out = u64::from_le_bytes(bytes.try_into().unwrap());
                    }
                    LinearResponse::from_untrusted(&share, frame.binding, &values[..share.rows])
                };
                let left = decode(Worker::Zero, zero)?;
                let right = decode(Worker::One, one)?;
                acknowledge_seeded_reply(&pending.window);
                pending.freeze_first(Ok(left))?.freeze_second(Ok(right))?
            }
            ClientState::Lookup(pending) => {
                let decode = |worker, bytes| {
                    let share = pending.share(worker);
                    let frame = Frame::decode(bytes, REPLY, &pending.window.publication)?;
                    frame.validate(share.binding(), &pending.window.publication)?;
                    LookupResponse::from_untrusted(&share, frame.binding, frame.body)
                };
                let left = decode(Worker::Zero, zero)?;
                let right = decode(Worker::One, one)?;
                acknowledge_seeded_reply(&pending.window);
                pending.freeze_first(Ok(left))?.freeze_second(Ok(right))?
            }
            ClientState::Ready(_) => return Err(Error::Schedule),
        };
        self.state = Some(ClientState::Ready(window));
        let Some(ClientState::Ready(window)) = &self.state else {
            unreachable!()
        };
        Ok(WireTentative {
            values: window.tentative_values(),
            signed: window.tentative_signed(),
        })
    }

    pub fn finish(&mut self) -> Result<AcceptedWindow, Error> {
        self.take_ready()?.finish()
    }

    pub fn abort(&mut self) {
        self.state = None;
        self.seed_keys = None;
    }
}

// Remote expansion cannot share an in-process lease. A bound, fixed reply marks
// transport progress, NOT trust in worker computation. Original full-window
// numeric/authentication verification remains mandatory in Window::finish.
fn acknowledge_seeded_reply(window: &Window) {
    if window.profile == SharingProfile::Aes128WindowV1 {
        window
            ._lease
            .seeded_completed
            .store(window.completed as u32 + 1, Ordering::Release);
    }
}

// Constructor rollback must destroy every admitted payload before releasing its
// wire reservation. A guard fixes that order independently of function parameter
// or local-variable drop order, including a failed native Budget admission.
struct AdmittedWorkerSources {
    matrices: Vec<I16Matrix>,
    table: FloatTable,
    permit: WirePermit,
}

struct WorkerSource {
    matrices: Vec<I16Matrix>,
    table: FloatTable,
    worker: Worker,
    budget: Arc<Budget>,
    _permit: WirePermit,
}

const ACTIVE: u8 = 0;
const FINISHED: u8 = 1;
const FAILED: u8 = 2;

struct WorkerRun {
    status: AtomicU8,
    lease: Arc<Lease>,
}

impl WorkerRun {
    fn fail(&self) {
        self.status.store(FAILED, Ordering::Release);
        self.lease.poisoned.store(true, Ordering::Release);
    }
}

/// One role, one authenticated client session, immutable public sources. A new
/// schedule must advance the sequence; a live, failed, replayed or substituted
/// request cannot be replaced. No complementary share, token, client input or
/// publisher scalar is stored here. Restart-persistent replay admission belongs
/// to the deployment, not this process-local owner.
pub struct WorkerWire {
    source: Option<Arc<WorkerSource>>,
    session: Option<[u8; 32]>,
    last_sequence: Option<u64>,
    active: Option<Arc<WorkerRun>>,
}

impl WorkerWire {
    /// Copies already-admitted public sources under the additional global wire
    /// allowance. Caller source/import buffers retain their original ownership.
    /// Stages must occur in publication order with exact IDs, geometry and hashes.
    pub fn new(
        publication: Arc<Publication>,
        authenticated_owner_digest: [u8; 32],
        worker: Worker,
        stages: &[Stage<'_>],
        table: &FloatTable,
    ) -> Result<Self, Error> {
        let permit = Self::validate_sources(
            &publication,
            authenticated_owner_digest,
            stages.iter().map(|s| (s.id, s.matrix)),
            table,
        )?;
        // Reject exhausted publication handles before any variable-size source
        // copy. On later copy/validation failures, locals destroy those payloads
        // before this budget and the earlier wire permit are released.
        let budget = Budget::new(&publication)?;
        let mut matrices = payload::capacity(stages.len())?;
        for stage in stages {
            matrices.push(I16Matrix::from_encoded_with_capacity(
                stage.matrix.rows,
                stage.matrix.columns,
                &stage.matrix.weights,
                &stage.matrix.input_bounds,
                publication.capacity,
            )?);
        }
        let table = FloatTable::from_le_bytes_with_capacity(
            table.vocabulary,
            table.width,
            &table.bytes,
            publication.capacity,
        )?;
        Ok(Self::with_sources(
            budget,
            worker,
            AdmittedWorkerSources {
                matrices,
                table,
                permit,
            },
        ))
    }

    /// Moves immutable source owners into one party without duplicating their
    /// payloads. IDs, source bytes, public bounds and the trusted pin receive the
    /// same checks and global wire allowance as the copying constructor.
    pub fn from_owned(
        publication: Arc<Publication>,
        authenticated_owner_digest: [u8; 32],
        worker: Worker,
        ids: &[[u8; 32]],
        matrices: Vec<I16Matrix>,
        table: FloatTable,
    ) -> Result<Self, Error> {
        if ids.len() != matrices.len()
            || matrices.len() != publication.matrices.len()
            || matrices.capacity() != matrices.len()
        {
            return Err(Error::Shape);
        }
        let permit = Self::validate_sources(
            &publication,
            authenticated_owner_digest,
            ids.iter().copied().zip(&matrices),
            &table,
        )?;
        // These owners arrived as parameters, whose destruction would follow
        // local permit destruction on `?`. Adopt them before fallible admission.
        let sources = AdmittedWorkerSources {
            matrices,
            table,
            permit,
        };
        let budget = Budget::new(&publication)?;
        Ok(Self::with_sources(budget, worker, sources))
    }

    fn validate_sources<'a>(
        publication: &Publication,
        authenticated_owner_digest: [u8; 32],
        stages: impl ExactSizeIterator<Item = ([u8; 32], &'a I16Matrix)>,
        table: &FloatTable,
    ) -> Result<WirePermit, Error> {
        if publication.digest != authenticated_owner_digest
            || publication.compute_digest() != authenticated_owner_digest
            || stages.len() != publication.matrices.len()
            || table.digest != publication.table_digest
            || table.shape() != (publication.row_hashes.len(), publication.width)
        {
            return Err(Error::Binding);
        }
        let count = stages.len();
        for ((id, matrix), expected) in stages.zip(&publication.matrices) {
            if id != expected.id
                || matrix.digest != expected.digest
                || matrix.shape() != (expected.rows, expected.columns)
                || matrix.input_bounds != expected.input_bounds
                || matrix.output_bounds != expected.output_bounds
            {
                return Err(Error::Binding);
            }
        }
        for (row, expected) in table
            .bytes
            .chunks_exact(table.width * 4)
            .zip(&publication.row_hashes)
        {
            if Sha256::digest(row).as_slice() != expected {
                return Err(Error::Binding);
            }
        }
        let resources = publication.resources;
        WirePermit::acquire(
            resources.source_weight_bytes
                + resources.source_bound_bytes
                + resources.source_table_bytes
                + count * size_of::<I16Matrix>()
                + size_of::<WorkerSource>()
                + size_of::<Self>()
                + 1024,
            publication.capacity,
        )
    }

    fn with_sources(budget: Arc<Budget>, worker: Worker, sources: AdmittedWorkerSources) -> Self {
        // No fallible transition after moving fields out of the rollback guard.
        // WorkerSource likewise keeps its permit after all payload fields.
        Self {
            source: Some(Arc::new(WorkerSource {
                matrices: sources.matrices,
                table: sources.table,
                worker,
                budget,
                _permit: sources.permit,
            })),
            session: None,
            last_sequence: None,
            active: None,
        }
    }

    /// `authenticated_start` must be authenticated to this worker's configured
    /// client channel. This method checks framing/binding, not channel identity.
    pub fn begin(&mut self, authenticated_start: &[u8]) -> Result<WorkerWireWindow, Error> {
        let result = self.begin_inner(authenticated_start);
        if result.is_err() {
            self.abort();
        }
        result
    }

    /// Additionally enforce the independently admitted application's context
    /// and complete schedule. Authentication alone cannot substitute a caller's
    /// shorter or differently ordered schedule for the compiler's public plan.
    /// A mismatch retires the worker exactly like any other failed begin.
    pub fn begin_bound(
        &mut self,
        authenticated_start: &[u8],
        expected_context: [u8; 32],
        expected_plan: &[Step],
    ) -> Result<WorkerWireWindow, Error> {
        let result = self.begin(authenticated_start).and_then(|window| {
            if window.binding.context != expected_context || window.plan.as_slice() != expected_plan
            {
                return Err(Error::Binding);
            }
            Ok(window)
        });
        if result.is_err() {
            self.abort();
        }
        result
    }

    fn validate_start_envelope<'a>(&self, bytes: &'a [u8]) -> Result<Frame<'a>, Error> {
        let source = self.source.as_ref().ok_or(Error::Retired)?;
        if self
            .active
            .as_ref()
            .is_some_and(|run| run.status.load(Ordering::Acquire) != FINISHED)
        {
            return Err(Error::Retired);
        }
        let publication = &source.budget.publication;
        let frame = Frame::decode(bytes, START, publication)?;
        let binding = frame.binding;
        if binding.publication != publication.digest
            || binding.worker != source.worker
            || binding.ordinal != 0
            || self.session.is_some_and(|s| s != binding.session)
            || self
                .last_sequence
                .is_some_and(|last| binding.sequence <= last)
            || binding.artifact != artifact(publication, binding.step)?
            || frame.dimensions != shape(publication, binding.step)?
            || frame.body.len()
                != binding.total as usize * 33
                    + if binding.profile == SharingProfile::Aes128WindowV1
                        && binding.worker == Worker::One
                    {
                        32
                    } else {
                        0
                    }
        {
            return Err(Error::Binding);
        }
        Ok(frame)
    }

    fn validate_start_schedule(&self, frame: &Frame<'_>) -> Result<(), Error> {
        let publication = &self
            .source
            .as_ref()
            .ok_or(Error::Retired)?
            .budget
            .publication;
        let binding = frame.binding;
        let mut hash = Sha256::new();
        hash.update(match binding.profile {
            SharingProfile::FullShares => b"pllm.gkm.window-plan.v1".as_slice(),
            SharingProfile::Aes128WindowV1 => super::super::SEEDED_SHARING_PROFILE.as_bytes(),
        });
        hash.update(publication.digest);
        hash.update(binding.context);
        hash.update((binding.total as u64).to_le_bytes());
        let schedule_bytes = binding.total as usize * 33;
        let mut cursor = Cursor::new(&frame.body[..schedule_bytes]);
        // Validate the entire authenticated schedule without allocating.
        for ordinal in 0..binding.total {
            let step = read_step(&mut cursor)?;
            if ordinal == 0 && step != binding.step {
                return Err(Error::Binding);
            }
            match step {
                Step::Lookup => hash.update([0]),
                Step::Linear(id) => {
                    publication.matrix_index(&id)?;
                    hash.update([1]);
                    hash.update(id);
                }
            }
        }
        cursor.finish()?;
        let digest: [u8; 32] = hash.finalize().into();
        if digest != binding.plan {
            return Err(Error::Binding);
        }
        Ok(())
    }

    fn begin_inner(&mut self, bytes: &[u8]) -> Result<WorkerWireWindow, Error> {
        self.begin_inner_admitted(bytes, None)
    }

    fn begin_inner_admitted(
        &mut self,
        bytes: &[u8],
        admission: Option<super::super::BatchAdmission>,
    ) -> Result<WorkerWireWindow, Error> {
        let source = self.source.as_ref().ok_or(Error::Retired)?;
        if self
            .active
            .as_ref()
            .is_some_and(|run| run.status.load(Ordering::Acquire) != FINISHED)
        {
            return Err(Error::Retired);
        }
        // Preserve scalar cumulative-counter and error ordering. Batch preflight
        // borrows these same validators before any plan/key/lease allocation.
        source.budget.publication.reserve_window()?;
        reserve(&source.budget.windows, 1, MAX_WINDOWS)?;
        reserve(&PROCESS_WINDOWS, 1, MAX_PROCESS_WINDOWS)?;
        let frame = self.validate_start_envelope(bytes)?;
        let binding = frame.binding;
        let publication = &source.budget.publication;
        reserve(
            &source.budget.steps,
            binding.total as u64,
            MAX_RESERVED_STEPS,
        )?;
        reserve(
            &PROCESS_STEPS,
            binding.total as u64,
            MAX_PROCESS_RESERVED_STEPS,
        )?;
        self.validate_start_schedule(&frame)?;
        let schedule_bytes = binding.total as usize * 33;
        let seeded_one =
            binding.profile == SharingProfile::Aes128WindowV1 && binding.worker == Worker::One;
        let permit = WirePermit::acquire(
            if admission.is_some() {
                0
            } else {
                binding.total as usize * size_of::<Step>()
                + publication
                    .matrices
                    .iter()
                    .map(|m| m.columns * 8 + m.rows * 16)
                    .max()
                    .unwrap()
                    .max(publication.width * 4)
                + size_of::<WorkerWireWindow>()
                // Explicit full/Zero evaluation uses the original linear input,
                // output and encoding scratch above. Only One expands selectors
                // and owns seeded stream/control state. The complete inline
                // window (including its inactive Option<SeededWorker>) is always
                // charged separately, regardless of profile or role.
                + if seeded_one {
                    publication.resources.seeded_window_extra_bytes
                } else {
                    0
                }
                + 1024
            },
            publication.capacity,
        )?;
        let lease = Lease::acquire(&source.budget)?;
        let mut plan = payload::capacity(binding.total as usize)?;
        let mut cursor = Cursor::new(&frame.body[..schedule_bytes]);
        for _ in 0..binding.total {
            plan.push(read_step(&mut cursor)?);
        }
        let plan = Arc::new(plan);
        if let Some(admission) = admission {
            lease.attach_batch(admission, &plan, binding.profile)?;
        }
        let seeded = if seeded_one {
            let mut keys = Zeroizing::new([[0; 16]; 2]);
            keys[0].copy_from_slice(&frame.body[schedule_bytes..schedule_bytes + 16]);
            keys[1].copy_from_slice(&frame.body[schedule_bytes + 16..]);
            Some(SeededWorker::from_wire(
                &keys,
                Arc::clone(&plan),
                binding,
                Arc::clone(&lease),
            )?)
        } else {
            None
        };
        let run = Arc::new(WorkerRun {
            status: AtomicU8::new(ACTIVE),
            lease,
        });
        self.session = Some(binding.session);
        self.last_sequence = Some(binding.sequence);
        self.active = Some(Arc::clone(&run));
        Ok(WorkerWireWindow {
            source: Arc::clone(source),
            seeded,
            plan,
            binding,
            completed: 0,
            run,
            _permit: permit,
        })
    }

    pub fn abort(&mut self) {
        if let Some(run) = &self.active {
            run.fail();
        }
        self.source = None;
        self.active = None;
    }
}

impl Drop for WorkerWire {
    fn drop(&mut self) {
        if let Some(run) = &self.active {
            if run.status.load(Ordering::Acquire) == ACTIVE {
                run.fail();
            }
        }
    }
}

/// Monotonic one-use role-local request. It can evaluate only the next explicit
/// or locally expanded share in its authenticated public plan. Every failure is sticky, including
/// duplicate execute/finish calls, even when callers ignore the error.
pub struct WorkerWireWindow {
    source: Arc<WorkerSource>,
    seeded: Option<SeededWorker>,
    plan: Arc<Vec<Step>>,
    binding: Binding,
    completed: usize,
    run: Arc<WorkerRun>,
    _permit: WirePermit,
}

/// Worker schedule completion only; this is NOT a client acceptance receipt.
#[derive(Debug, Clone, Copy)]
pub struct WorkerWireReceipt {
    pub request_id: [u8; 32],
    pub worker: Worker,
    pub steps: usize,
}

impl WorkerWireWindow {
    pub fn publication_resources(&self) -> super::super::PublicationResources {
        self.source.budget.publication.resources()
    }
    pub fn sharing_profile(&self) -> SharingProfile {
        self.binding.profile
    }

    /// Public native request identity, for binding an authenticated outer ledger.
    pub fn request_id(&self) -> [u8; 32] {
        request_id(&self.binding)
    }

    pub fn execute(&mut self, request: &[u8]) -> Result<WireBytes, Error> {
        let result = self.execute_inner(request);
        if result.is_err() {
            self.run.fail();
        }
        result
    }

    fn execute_inner(&mut self, bytes: &[u8]) -> Result<WireBytes, Error> {
        self.run.lease.ensure_open()?;
        if self.run.status.load(Ordering::Acquire) != ACTIVE {
            return Err(Error::Retired);
        }
        let publication = &self.source.budget.publication;
        let step = *self.plan.get(self.completed).ok_or(Error::Schedule)?;
        let mut expected = self.binding;
        expected.ordinal = self.completed as u32;
        expected.step = step;
        expected.artifact = artifact(publication, step)?;
        let frame = Frame::decode(bytes, REQUEST, publication)?;
        frame.validate(expected, publication)?;
        let reply_body = match step {
            Step::Linear(id) => publication.matrices[publication.matrix_index(&id)?]
                .rows
                .checked_mul(8),
            Step::Lookup => publication.width.checked_mul(4),
        }
        .ok_or(Error::Limit)?;
        let reply_permit = FramePermit::single(
            &self.run.lease,
            HEADER.checked_add(reply_body).ok_or(Error::Limit)?,
        )?;
        self.evaluate_row(frame, |binding, dimensions, body, lease| {
            encode_frame(REPLY, binding, dimensions, body, lease, reply_permit)
        })
    }

    // One model-neutral evaluator used by scalar framing and source-stage
    // aggregates. The emitter borrows the original native/scratch response;
    // scalar transport emits after its one primitive completes. Batch transport
    // dispatches the same shared kernels on complete contiguous token-major data.
    fn evaluate_row<T>(
        &mut self,
        frame: Frame<'_>,
        emit: impl FnOnce(Binding, (usize, usize), &[u8], &Arc<Lease>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        #[cfg(test)]
        SCALAR_EVALUATIONS.with(|count| count.set(count.get() + 1));
        self.run.lease.ensure_open()?;
        if self.run.status.load(Ordering::Acquire) != ACTIVE {
            return Err(Error::Retired);
        }
        let publication = &self.source.budget.publication;
        let step = *self.plan.get(self.completed).ok_or(Error::Schedule)?;
        let mut expected = self.binding;
        expected.ordinal = self.completed as u32;
        expected.step = step;
        expected.artifact = artifact(publication, step)?;
        frame.validate(expected, publication)?;
        let reply_body = match step {
            Step::Linear(id) => publication.matrices[publication.matrix_index(&id)?]
                .rows
                .checked_mul(8),
            Step::Lookup => publication.width.checked_mul(4),
        }
        .ok_or(Error::Limit)?;
        let response = match step {
            Step::Linear(id) => {
                let matrix = &self.source.matrices[publication.matrix_index(&id)?];
                if frame.body.len()
                    != if self.seeded.is_some() {
                        0
                    } else {
                        matrix.columns * 8
                    }
                {
                    return Err(Error::Shape);
                }
                // Explicit wire workers have no native pending/response owner;
                // reserve their primitive output before allocation. Seeded
                // responses already own native-priced output/expansion storage.
                let _output_quota = if self.seeded.is_none() {
                    ScratchPermit::acquire(
                        &self.run.lease,
                        WireScratch::<u64>::charge(matrix.rows)?,
                    )?
                } else {
                    None
                };
                let values = if let Some(worker) = &mut self.seeded {
                    let slot = ResponseSlot::new();
                    worker
                        .evaluate_linear(
                            matrix,
                            &LinearShare {
                                binding: expected,
                                values: None,
                                rows: matrix.rows,
                                slot: &slot,
                                lease: &self.run.lease,
                            },
                        )?
                        .values
                } else {
                    let mut input = WireScratch::filled(&self.run.lease, matrix.columns, 0u64)?;
                    for (out, word) in input.iter_mut().zip(frame.body.chunks_exact(8)) {
                        *out = u64::from_le_bytes(word.try_into().unwrap());
                    }
                    matrix.multiply_share(&input[..matrix.columns])?
                };
                let mut body = WireScratch::filled(
                    &self.run.lease,
                    matrix.rows.checked_mul(8).ok_or(Error::Limit)?,
                    0u8,
                )?;
                for (out, value) in body.chunks_exact_mut(8).zip(values.iter()) {
                    out.copy_from_slice(&value.to_le_bytes());
                }
                emit(
                    expected,
                    frame.dimensions,
                    &body[..matrix.rows * 8],
                    &self.run.lease,
                )?
            }
            Step::Lookup => {
                let _output_quota = if self.seeded.is_none() {
                    ScratchPermit::acquire(&self.run.lease, WireScratch::<u8>::charge(reply_body)?)?
                } else {
                    None
                };
                let values = if let Some(worker) = &mut self.seeded {
                    if !frame.body.is_empty() {
                        return Err(Error::Shape);
                    }
                    let slot = ResponseSlot::new();
                    worker
                        .evaluate_lookup(
                            &self.source.table,
                            &LookupShare {
                                binding: expected,
                                selector: None,
                                bytes: publication.width * 4,
                                slot: &slot,
                                lease: &self.run.lease,
                            },
                        )?
                        .bytes
                } else {
                    self.source.table.xor_rows(frame.body)?
                };
                emit(expected, frame.dimensions, &values, &self.run.lease)?
            }
        };
        self.completed += 1;
        Ok(response)
    }

    pub fn finish(&mut self) -> Result<WorkerWireReceipt, Error> {
        let result = (|| {
            self.run.lease.ensure_open()?;
            if self.run.status.load(Ordering::Acquire) != ACTIVE {
                return Err(Error::Retired);
            }
            if self.completed != self.plan.len() {
                return Err(Error::Incomplete);
            }
            if self.seeded.is_some()
                && self.run.lease.seeded_completed.load(Ordering::Acquire) as usize
                    != self.plan.len()
            {
                return Err(Error::Incomplete);
            }
            self.run.status.store(FINISHED, Ordering::Release);
            self.run.lease.poisoned.store(true, Ordering::Release);
            Ok(WorkerWireReceipt {
                request_id: request_id(&self.binding),
                worker: self.source.worker,
                steps: self.completed,
            })
        })();
        if result.is_err() {
            self.run.fail();
        }
        result
    }

    pub fn abort(&mut self) {
        self.run.fail();
    }
}

impl Drop for WorkerWireWindow {
    fn drop(&mut self) {
        if self.run.status.load(Ordering::Acquire) == ACTIVE {
            self.run.fail();
        }
    }
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "seeded_wire_tests.rs"]
mod seeded_tests;

#[path = "batch_wire.rs"]
mod batch_wire;
pub use batch_wire::{ClientBatchWireWindow, WorkerBatchWireWindow};

#[cfg(test)]
#[path = "wire_quota_tests.rs"]
mod quota_tests;
