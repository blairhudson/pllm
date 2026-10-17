use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    Arc,
};

use k256::{ProjectivePoint, Scalar};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use super::{
    payload, reconstruct_signed, rescale_signed, sample_scalar, signed_scalar, Capacity, Entropy,
    Error, FloatTable, I16Matrix, OsEntropy, Publication, Purpose, SharingProfile,
    MAX_CLIENTS_PER_PUBLICATION, MAX_LIVE_WINDOWS, MAX_PROCESS_LIVE_WINDOWS,
    MAX_PROCESS_RESERVED_STEPS, MAX_PROCESS_WINDOWS, MAX_RESERVED_STEPS, MAX_WINDOWS,
};

#[cfg(test)]
use super::{MAX_DIMENSION, MAX_WINDOW_STEPS};

use super::seeded::Streams;
#[path = "seeded_window.rs"]
mod seeded_window;
pub use seeded_window::SeededWorker;
#[path = "wire.rs"]
pub(super) mod wire;
pub use wire::{
    ClientWireWindow, WireBytes, WireResources, WireTentative, WorkerWire, WorkerWireReceipt,
    WorkerWireWindow, MAX_PROCESS_WIRE_BYTES, MAX_PUBLICATION_WIRE_BYTES, MAX_WIRE_FRAME_BYTES,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Worker {
    Zero,
    One,
}

/// Exact public order is frozen at request admission. Tokens and inputs are not
/// part of this public schedule. This capsule uses one authenticated row table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Lookup,
    Linear([u8; 32]),
}

/// Public reply binding. It contains no selector token or reconstructed input.
/// Copying this framing does not copy the private one-use transaction owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    publication: [u8; 32],
    session: [u8; 32],
    sequence: u64,
    context: [u8; 32],
    plan: [u8; 32],
    total: u32,
    ordinal: u32,
    step: Step,
    artifact: [u8; 32],
    worker: Worker,
    profile: SharingProfile,
}

impl Binding {
    pub fn sharing_profile(&self) -> SharingProfile {
        self.profile
    }
    pub fn worker(&self) -> Worker {
        self.worker
    }
    pub fn context(&self) -> [u8; 32] {
        self.context
    }
    pub fn publication(&self) -> [u8; 32] {
        self.publication
    }
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }
    pub fn total(&self) -> u32 {
        self.total
    }
    pub fn step(&self) -> Step {
        self.step
    }
}

pub struct LinearShare<'a> {
    binding: Binding,
    values: Option<&'a [u64]>,
    rows: usize,
    slot: &'a ResponseSlot,
    lease: &'a Arc<Lease>,
}
impl LinearShare<'_> {
    pub fn binding(&self) -> Binding {
        self.binding
    }
    /// None for the seed-only designated worker: no expanded input is delivered.
    /// Even an optional explicit payload cannot outlive its pending owner.
    ///
    /// ```compile_fail,E0505
    /// use pllm_core::gkm::{PendingLinear, Worker};
    /// fn after_owner_drop(pending: PendingLinear) {
    ///     let share = pending.share(Worker::Zero);
    ///     let payload: Option<&[u64]> = share.values();
    ///     drop(pending);
    ///     assert!(!payload.unwrap().is_empty());
    /// }
    /// ```
    pub fn values(&self) -> Option<&[u64]> {
        self.values
    }
}

pub struct LookupShare<'a> {
    binding: Binding,
    selector: Option<&'a [u8]>,
    bytes: usize,
    slot: &'a ResponseSlot,
    lease: &'a Arc<Lease>,
}
impl LookupShare<'_> {
    pub fn binding(&self) -> Binding {
        self.binding
    }
    /// None for the seed-only designated worker.
    /// Even an optional explicit payload cannot outlive its pending owner.
    ///
    /// ```compile_fail,E0505
    /// use pllm_core::gkm::{PendingLookup, Worker};
    /// fn after_owner_drop(pending: PendingLookup) {
    ///     let share = pending.share(Worker::Zero);
    ///     let payload: Option<&[u8]> = share.selector();
    ///     drop(pending);
    ///     assert!(!payload.unwrap().is_empty());
    /// }
    /// ```
    pub fn selector(&self) -> Option<&[u8]> {
        self.selector
    }
}

struct ResponseSlot(AtomicBool);

impl ResponseSlot {
    fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    fn produce<T>(
        &self,
        lease: &Arc<Lease>,
        make: impl FnOnce() -> Result<T, Error>,
    ) -> Result<T, Error> {
        let result = (|| {
            lease.ensure_open()?;
            self.0
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| Error::Retired)?;
            make()
        })();
        if result.is_err() {
            lease.poisoned.store(true, Ordering::Release);
        }
        result
    }
}

/// One owned response per pending worker-role slot. A framing, allocation or
/// duplicate-slot failure poisons the entire transaction even if ignored.
/// Its shared lease keeps capacity reserved until this buffer actually drops.
/// Copying a Binding alone cannot allocate an admitted response.
///
/// ```compile_fail
/// use pllm_core::gkm::{Binding, LinearResponse};
/// fn unleased(binding: Binding) { let _ = LinearResponse::from_untrusted(binding, &[0]); }
/// ```
pub struct LinearResponse {
    binding: Binding,
    values: Zeroizing<Vec<u64>>,
    _lease: Arc<Lease>,
}
impl LinearResponse {
    pub fn from_untrusted(
        share: &LinearShare<'_>,
        binding: Binding,
        values: &[u64],
    ) -> Result<Self, Error> {
        share.slot.produce(share.lease, || {
            if binding != share.binding {
                return Err(Error::Binding);
            }
            if values.len() > share.rows {
                return Err(Error::Limit);
            }
            if values.len() != share.rows {
                return Err(Error::Shape);
            }
            Ok(Self {
                binding,
                values: Zeroizing::new(payload::copy(values)?),
                _lease: Arc::clone(share.lease),
            })
        })
    }
}

pub struct LookupResponse {
    binding: Binding,
    bytes: Zeroizing<Vec<u8>>,
    _lease: Arc<Lease>,
}
impl LookupResponse {
    pub fn from_untrusted(
        share: &LookupShare<'_>,
        binding: Binding,
        bytes: &[u8],
    ) -> Result<Self, Error> {
        share.slot.produce(share.lease, || {
            if binding != share.binding {
                return Err(Error::Binding);
            }
            if bytes.len() > share.bytes {
                return Err(Error::Limit);
            }
            if bytes.len() != share.bytes {
                return Err(Error::Shape);
            }
            Ok(Self {
                binding,
                bytes: Zeroizing::new(payload::copy(bytes)?),
                _lease: Arc::clone(share.lease),
            })
        })
    }
}

impl I16Matrix {
    pub fn evaluate(&self, share: &LinearShare<'_>) -> Result<LinearResponse, Error> {
        share.slot.produce(share.lease, || {
            if share.binding.artifact != self.digest {
                return Err(Error::Binding);
            }
            Ok(LinearResponse {
                binding: share.binding,
                values: self.multiply_share(share.values.ok_or(Error::Binding)?)?,
                _lease: Arc::clone(share.lease),
            })
        })
    }
}

impl FloatTable {
    pub fn evaluate(&self, share: &LookupShare<'_>) -> Result<LookupResponse, Error> {
        share.slot.produce(share.lease, || {
            if share.binding.artifact != self.digest {
                return Err(Error::Binding);
            }
            Ok(LookupResponse {
                binding: share.binding,
                bytes: self.xor_rows(share.selector.ok_or(Error::Binding)?)?,
                _lease: Arc::clone(share.lease),
            })
        })
    }
}

static PROCESS_WINDOWS: AtomicU64 = AtomicU64::new(0);
static PROCESS_STEPS: AtomicU64 = AtomicU64::new(0);
static PROCESS_LIVE: AtomicU32 = AtomicU32::new(0);

struct Budget {
    windows: AtomicU64,
    steps: AtomicU64,
    live: AtomicU32,
    publication: Arc<Publication>,
}
impl Drop for Budget {
    fn drop(&mut self) {
        self.publication.clients.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Budget {
    fn new(publication: &Arc<Publication>) -> Result<Arc<Self>, Error> {
        publication
            .clients
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_CLIENTS_PER_PUBLICATION).then_some(n + 1)
            })
            .map_err(|_| Error::Limit)?;
        Ok(Arc::new(Self {
            windows: AtomicU64::new(0),
            steps: AtomicU64::new(0),
            live: AtomicU32::new(0),
            publication: Arc::clone(publication),
        }))
    }
}

fn reserve(counter: &AtomicU64, amount: u64, cap: u64) -> Result<u64, Error> {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
            count.checked_add(amount).filter(|&next| next <= cap)
        })
        .map_err(|_| Error::Limit)
}

struct Lease {
    budget: Arc<Budget>,
    poisoned: AtomicBool,
    seeded_completed: AtomicU32,
}
impl Lease {
    fn ensure_open(&self) -> Result<(), Error> {
        if self.poisoned.load(Ordering::Acquire) {
            Err(Error::Retired)
        } else {
            Ok(())
        }
    }

    fn acquire(budget: &Arc<Budget>) -> Result<Arc<Self>, Error> {
        budget
            .live
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_LIVE_WINDOWS).then_some(n + 1)
            })
            .map_err(|_| Error::Limit)?;
        if budget
            .publication
            .live
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_LIVE_WINDOWS).then_some(n + 1)
            })
            .is_err()
        {
            budget.live.fetch_sub(1, Ordering::AcqRel);
            return Err(Error::Limit);
        }
        if PROCESS_LIVE
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_PROCESS_LIVE_WINDOWS).then_some(n + 1)
            })
            .is_err()
        {
            budget.live.fetch_sub(1, Ordering::AcqRel);
            budget.publication.live.fetch_sub(1, Ordering::AcqRel);
            return Err(Error::Limit);
        }
        // Fixed-size Arc allocation is the explicit stable-Rust OOM exception.
        Ok(Arc::new(Self {
            budget: Arc::clone(budget),
            poisoned: AtomicBool::new(false),
            seeded_completed: AtomicU32::new(0),
        }))
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.budget.live.fetch_sub(1, Ordering::AcqRel);
        self.budget.publication.live.fetch_sub(1, Ordering::AcqRel);
        PROCESS_LIVE.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A client admission owner. Failed/malformed/cancelled admitted attempts burn
/// their window and entire reserved step budget; no refund on drop or failure.
/// Process counters prevent resetting these caps by constructing another client
/// in this process. Restart-safe deployment security budgets are out of scope.
/// Live admission additionally belongs to the shared Publication, not the client.
/// Stable Rust's fixed-size Budget/Lease/plan Arc handles can abort on allocation
/// failure; all variable-size payload reservations return Error::Allocation.
pub struct Client {
    publication: Arc<Publication>,
    session: [u8; 32],
    budget: Arc<Budget>,
}

impl Client {
    /// `authenticated_owner_digest` MUST be supplied by a trusted owner channel,
    /// not copied blindly from an untrusted publication. This is an in-process
    /// trust pin, not an implementation of signatures or artifact delivery.
    pub fn new(
        publication: Arc<Publication>,
        authenticated_owner_digest: [u8; 32],
    ) -> Result<Self, Error> {
        Self::new_with(publication, authenticated_owner_digest, &mut OsEntropy)
    }

    fn new_with(
        publication: Arc<Publication>,
        expected: [u8; 32],
        rng: &mut impl Entropy,
    ) -> Result<Self, Error> {
        if publication.digest != expected || publication.compute_digest() != expected {
            return Err(Error::Binding);
        }
        let mut session = [0; 32];
        rng.fill(Purpose::Session, &mut session)?;
        // Fixed-size, admitted Arc handle; payload allocation recovery does not
        // claim to make Arc::new fallible on stable Rust.
        let budget = Budget::new(&publication)?;
        Ok(Self {
            publication,
            session,
            budget,
        })
    }

    pub fn begin(&self, context: [u8; 32], plan: &[Step]) -> Result<Window, Error> {
        self.begin_profile(context, plan, SharingProfile::FullShares)
    }

    fn begin_profile(
        &self,
        context: [u8; 32],
        plan: &[Step],
        profile: SharingProfile,
    ) -> Result<Window, Error> {
        // Attempts are charged before validating caller-controlled shape/order.
        let sequence = reserve(&self.budget.windows, 1, MAX_WINDOWS)?;
        reserve(&PROCESS_WINDOWS, 1, MAX_PROCESS_WINDOWS)?;
        if plan.is_empty()
            || plan.len() > self.publication.capacity.max_window_steps()
            || (self.publication.capacity == Capacity::Decoder
                && profile != SharingProfile::FullShares)
        {
            return Err(Error::Limit);
        }
        reserve(&self.budget.steps, plan.len() as u64, MAX_RESERVED_STEPS)?;
        reserve(
            &PROCESS_STEPS,
            plan.len() as u64,
            MAX_PROCESS_RESERVED_STEPS,
        )?;
        let mut hash = Sha256::new();
        match profile {
            SharingProfile::FullShares => hash.update(b"pllm.gkm.window-plan.v1"),
            SharingProfile::Aes128WindowV1 => hash.update(super::SEEDED_SHARING_PROFILE.as_bytes()),
        }
        hash.update(self.publication.digest);
        hash.update(context);
        hash.update((plan.len() as u64).to_le_bytes());
        for step in plan {
            match step {
                Step::Lookup => hash.update([0]),
                Step::Linear(id) => {
                    self.publication.matrix_index(id)?;
                    hash.update([1]);
                    hash.update(id);
                }
            }
        }
        // Publication admission already budgets every accumulator, plan and
        // maximum pending pair, even if some matrices are unused in this plan.
        let lease = Lease::acquire(&self.budget)?;
        let mut accumulators = payload::capacity(self.publication.matrices.len())?;
        for matrix in &self.publication.matrices {
            accumulators.push(Accumulator {
                x: Zeroizing::new(payload::filled(matrix.columns, Scalar::ZERO)?),
                y: Zeroizing::new(payload::filled(matrix.rows, Scalar::ZERO)?),
                calls: 0,
            });
        }
        Ok(Window {
            publication: Arc::clone(&self.publication),
            session: self.session,
            sequence,
            context,
            plan: Arc::new(payload::copy(plan)?),
            profile,
            streams: None,
            plan_digest: hash.finalize().into(),
            completed: 0,
            accumulators,
            tentative: Zeroizing::new(Vec::new()),
            signed: Zeroizing::new(Vec::new()),
            _lease: lease,
        })
    }
}

struct Accumulator {
    x: Zeroizing<Vec<Scalar>>,
    y: Zeroizing<Vec<Scalar>>,
    calls: usize,
}

/// Request-owned, non-cloneable tentative transaction. No serialization, Debug,
/// mutable response access, or accepted-result constructor is exposed.
///
/// ```compile_fail
/// use pllm_core::gkm::Window;
/// fn clone_window(window: &Window) { let _: Window = window.clone(); }
/// ```
/// ```compile_fail
/// use pllm_core::gkm::Window;
/// fn replay(window: Window) { let _ = window.finish(); let _ = window.finish(); }
/// ```
pub struct Window {
    publication: Arc<Publication>,
    session: [u8; 32],
    sequence: u64,
    context: [u8; 32],
    plan: Arc<Vec<Step>>,
    profile: SharingProfile,
    streams: Option<Streams>,
    plan_digest: [u8; 32],
    completed: usize,
    accumulators: Vec<Accumulator>,
    tentative: Zeroizing<Vec<f32>>,
    signed: Zeroizing<Vec<i64>>,
    _lease: Arc<Lease>,
}

impl Drop for Window {
    fn drop(&mut self) {
        // Cancel/close the shared transaction before retained workers/responses
        // can be used again. Payload fields still drop before the last lease.
        self._lease.poisoned.store(true, Ordering::Release);
    }
}

impl Window {
    pub fn sharing_profile(&self) -> SharingProfile {
        self.profile
    }
    /// Unique in this client session even when the caller reuses its context.
    /// Retained by the final acceptance receipt; no private input is hashed here.
    pub fn request_id(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"pllm.gkm.request.v1");
        hash.update(self.publication.digest);
        hash.update(self.session);
        hash.update(self.sequence.to_le_bytes());
        hash.update(self.context);
        hash.update(self.plan_digest);
        hash.finalize().into()
    }

    /// Borrow for trusted client-local dependent computation only. This is not
    /// an acceptance receipt. An adversarial linear result can be present here.
    pub fn tentative_values(&self) -> &[f32] {
        &self.tentative
    }
    pub fn tentative_signed(&self) -> &[i64] {
        &self.signed
    }

    fn binding(&self, worker: Worker) -> Result<Binding, Error> {
        let step = *self.plan.get(self.completed).ok_or(Error::Schedule)?;
        let artifact = match step {
            Step::Lookup => self.publication.table_digest,
            Step::Linear(id) => {
                self.publication.matrices[self.publication.matrix_index(&id)?].digest
            }
        };
        Ok(Binding {
            publication: self.publication.digest,
            session: self.session,
            sequence: self.sequence,
            context: self.context,
            plan: self.plan_digest,
            total: self.plan.len() as u32,
            ordinal: self.completed as u32,
            step,
            artifact,
            worker,
            profile: self.profile,
        })
    }

    /// Inputs are already nearest/2048 encoded. Standalone `encode_input` defines
    /// that boundary; admitting encoded integers also supports exact toy oracles.
    pub fn linear(self, input: &[i64]) -> Result<PendingLinear, Error> {
        self.linear_with(input, &mut OsEntropy)
    }

    fn linear_with(
        mut self,
        input: &[i64],
        rng: &mut impl Entropy,
    ) -> Result<PendingLinear, Error> {
        self._lease.ensure_open()?;
        let Some(Step::Linear(id)) = self.plan.get(self.completed) else {
            return Err(Error::Schedule);
        };
        let stage = self.publication.matrix_index(id)?;
        let matrix = &self.publication.matrices[stage];
        if input.len() != matrix.columns {
            return Err(Error::Shape);
        }
        if input
            .iter()
            .zip(matrix.input_bounds.iter())
            .any(|(&x, &bound)| x.unsigned_abs() > bound)
        {
            return Err(Error::Bound);
        }
        let mut bytes = Zeroizing::new(payload::filled(matrix.columns * 8, 0)?);
        let (left, right) = if let Some(streams) = &mut self.streams {
            streams.linear.fill(&mut bytes)?;
            let complement =
                payload::collect(input.iter().zip(bytes.chunks_exact(8)).map(|(&x, b)| {
                    (x as u64).wrapping_sub(u64::from_le_bytes(b.try_into().unwrap()))
                }))?;
            (Zeroizing::new(complement), Zeroizing::new(Vec::new()))
        } else {
            rng.fill(Purpose::LinearMask, &mut bytes)?;
            let left = Zeroizing::new(payload::collect(
                bytes
                    .chunks_exact(8)
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap())),
            )?);
            let right = Zeroizing::new(payload::collect(
                input
                    .iter()
                    .zip(left.iter())
                    .map(|(&x, &a)| (x as u64).wrapping_sub(a)),
            )?);
            (left, right)
        };
        let input = Zeroizing::new(payload::copy(input)?);
        Ok(PendingLinear {
            stage,
            input,
            shares: [left, right],
            slots: [ResponseSlot::new(), ResponseSlot::new()],
            window: self,
        })
    }

    pub fn lookup(self, token: usize) -> Result<PendingLookup, Error> {
        self.lookup_with(token, &mut OsEntropy)
    }

    fn lookup_with(mut self, token: usize, rng: &mut impl Entropy) -> Result<PendingLookup, Error> {
        self._lease.ensure_open()?;
        if self.plan.get(self.completed) != Some(&Step::Lookup) {
            return Err(Error::Schedule);
        }
        let vocabulary = self.publication.row_hashes.len();
        if token >= vocabulary {
            return Err(Error::Bound);
        }
        let mut left = Zeroizing::new(payload::filled(vocabulary.div_ceil(8), 0)?);
        if let Some(streams) = &mut self.streams {
            streams.lookup.fill(&mut left)?;
        } else {
            rng.fill(Purpose::LookupMask, &mut left)?;
        }
        if vocabulary % 8 != 0 {
            let last = left.len() - 1;
            left[last] &= (1u8 << (vocabulary % 8)) - 1;
        }
        let right = if self.profile == SharingProfile::Aes128WindowV1 {
            left[token / 8] ^= 1 << (token % 8);
            Zeroizing::new(Vec::new())
        } else {
            let mut right = Zeroizing::new(payload::copy(&left)?);
            right[token / 8] ^= 1 << (token % 8);
            right
        };
        Ok(PendingLookup {
            window: self,
            token: Zeroizing::new(token),
            selectors: [left, right],
            slots: [ResponseSlot::new(), ResponseSlot::new()],
        })
    }

    /// Consumes the complete original output window. No result is returned on
    /// incomplete schedules, count mismatch, entropy failure or a failed check.
    pub fn finish(self) -> Result<AcceptedWindow, Error> {
        self.finish_with(&mut OsEntropy)
    }

    fn finish_with(mut self, rng: &mut impl Entropy) -> Result<AcceptedWindow, Error> {
        self._lease.ensure_open()?;
        if self.completed != self.plan.len() {
            return Err(Error::Incomplete);
        }
        if self.profile == SharingProfile::Aes128WindowV1
            && (self._lease.seeded_completed.load(Ordering::Acquire) as usize != self.plan.len()
                || !self.streams.as_ref().is_some_and(Streams::complete))
        {
            return Err(Error::Incomplete);
        }
        // Verify exact per-stage counts before drawing any final challenge.
        for (matrix, accumulator) in self.publication.matrices.iter().zip(&self.accumulators) {
            let expected = self
                .plan
                .iter()
                .filter(|step| **step == Step::Linear(matrix.id))
                .count();
            if accumulator.calls != expected {
                return Err(Error::Incomplete);
            }
        }
        let mut left = ProjectivePoint::IDENTITY;
        let mut combined_y =
            Zeroizing::new(payload::filled(self.publication.r.len(), Scalar::ZERO)?);
        for (matrix, accumulator) in self.publication.matrices.iter().zip(&self.accumulators) {
            if accumulator.calls == 0 {
                continue;
            }
            let beta = Zeroizing::new(sample_scalar(rng, Purpose::MatrixChallenge)?);
            for (s, x) in matrix.s.iter().zip(accumulator.x.iter()) {
                left += *s * (*beta * x);
            }
            // Different output widths use the corresponding prefix of global R.
            for (sum, y) in combined_y.iter_mut().zip(accumulator.y.iter()) {
                *sum += *beta * y;
            }
        }
        let right = self
            .publication
            .r
            .iter()
            .zip(combined_y.iter())
            .fold(ProjectivePoint::IDENTITY, |sum, (r, y)| sum + r * y);
        if !bool::from(left.ct_eq(&right)) {
            return Err(Error::Verification);
        }
        Ok(AcceptedWindow {
            request: self.request_id(),
            context: self.context,
            publication: self.publication.digest,
            plan: self.plan_digest,
            steps: self.completed,
            profile: self.profile,
            values: Zeroizing::new(std::mem::take(&mut *self.tentative)),
            signed: Zeroizing::new(std::mem::take(&mut *self.signed)),
            _lease: Arc::clone(&self._lease),
        })
    }
}

/// First response is frozen independently, allowing bounded partial-reply
/// cancellation. The worker-zero/worker-one order is explicit; reordered or
/// duplicated roles reject and consume the whole transaction.
/// Forward the complete evaluation/decoding `Result` to `freeze_first` (and
/// subsequently `freeze_second`): oversized/failed decoding must burn the owner,
/// not be discarded for a retry. Dropping the owner also cancels it.
///
/// ```compile_fail
/// use pllm_core::gkm::{PendingLinear, LinearResponse, Error, Worker};
/// fn replay(pending: PendingLinear, reply: Result<LinearResponse, Error>) {
///     let _ = pending.freeze_first(reply);
///     let _ = pending.share(Worker::Zero); // consumed even if reply was Err
/// }
/// ```
pub struct PendingLinear {
    stage: usize,
    input: Zeroizing<Vec<i64>>,
    shares: [Zeroizing<Vec<u64>>; 2],
    slots: [ResponseSlot; 2],
    // Last: masks and input must drop before the window's last lease reference.
    window: Window,
}
pub struct HalfLinear {
    pending: PendingLinear,
    first: LinearResponse,
}

impl PendingLinear {
    pub fn share(&self, worker: Worker) -> LinearShare<'_> {
        LinearShare {
            binding: self
                .window
                .binding(worker)
                .expect("admitted pending linear"),
            values: (!(self.window.profile == SharingProfile::Aes128WindowV1
                && worker == Worker::One))
                .then_some(self.shares[if worker == Worker::Zero { 0 } else { 1 }].as_slice()),
            rows: self.window.publication.matrices[self.stage].rows,
            slot: &self.slots[if worker == Worker::Zero { 0 } else { 1 }],
            lease: &self.window._lease,
        }
    }

    pub fn freeze_first(
        self,
        response: Result<LinearResponse, Error>,
    ) -> Result<HalfLinear, Error> {
        let response = response?;
        self.validate(&response, Worker::Zero)?;
        Ok(HalfLinear {
            pending: self,
            first: response,
        })
    }

    fn validate(&self, response: &LinearResponse, worker: Worker) -> Result<(), Error> {
        self.window._lease.ensure_open()?;
        if response.binding != self.window.binding(worker)?
            || !Arc::ptr_eq(&response._lease, &self.window._lease)
        {
            return Err(Error::Binding);
        }
        if response.values.len() != self.window.publication.matrices[self.stage].rows {
            return Err(Error::Shape);
        }
        Ok(())
    }
}

impl HalfLinear {
    pub fn remaining_share(&self) -> LinearShare<'_> {
        self.pending.share(Worker::One)
    }

    pub fn freeze_second(self, response: Result<LinearResponse, Error>) -> Result<Window, Error> {
        self.freeze_second_with(response, &mut OsEntropy)
    }

    fn freeze_second_with(
        self,
        response: Result<LinearResponse, Error>,
        rng: &mut impl Entropy,
    ) -> Result<Window, Error> {
        let response = response?;
        self.pending.validate(&response, Worker::One)?;
        // Both buffers are now owned, immutable, exact-shaped and context-bound.
        // No challenge has existed while either response could be replaced.
        let mut pending = self.pending;
        let matrix = &pending.window.publication.matrices[pending.stage];
        let signed = Zeroizing::new(payload::collect(
            self.first
                .values
                .iter()
                .zip(response.values.iter())
                .map(|(&a, &b)| reconstruct_signed(a, b)),
        )?);
        if signed
            .iter()
            .zip(matrix.output_bounds.iter())
            .any(|(&y, &bound)| y.unsigned_abs() > bound)
        {
            return Err(Error::Bound);
        }
        let alpha = Zeroizing::new(sample_scalar(rng, Purpose::ClaimChallenge)?);
        let accumulator = &mut pending.window.accumulators[pending.stage];
        for (sum, &x) in accumulator.x.iter_mut().zip(pending.input.iter()) {
            *sum += *alpha * signed_scalar(x);
        }
        for (sum, &y) in accumulator.y.iter_mut().zip(signed.iter()) {
            *sum += *alpha * signed_scalar(y);
        }
        accumulator.calls += 1;
        pending.window.tentative =
            Zeroizing::new(payload::collect(signed.iter().map(|&y| rescale_signed(y)))?);
        pending.window.signed = signed;
        pending.window.completed += 1;
        Ok(pending.window)
    }
}

/// Dense lookup's one-use owner. Forward decoding/evaluation errors into the
/// consuming freeze methods, exactly as for `PendingLinear`.
pub struct PendingLookup {
    token: Zeroizing<usize>,
    selectors: [Zeroizing<Vec<u8>>; 2],
    slots: [ResponseSlot; 2],
    window: Window,
}
pub struct HalfLookup {
    pending: PendingLookup,
    first: LookupResponse,
}

impl PendingLookup {
    pub fn share(&self, worker: Worker) -> LookupShare<'_> {
        LookupShare {
            binding: self
                .window
                .binding(worker)
                .expect("admitted pending lookup"),
            selector: (!(self.window.profile == SharingProfile::Aes128WindowV1
                && worker == Worker::One))
                .then_some(self.selectors[if worker == Worker::Zero { 0 } else { 1 }].as_slice()),
            bytes: self.window.publication.width * 4,
            slot: &self.slots[if worker == Worker::Zero { 0 } else { 1 }],
            lease: &self.window._lease,
        }
    }

    pub fn freeze_first(
        self,
        response: Result<LookupResponse, Error>,
    ) -> Result<HalfLookup, Error> {
        let response = response?;
        self.validate(&response, Worker::Zero)?;
        Ok(HalfLookup {
            pending: self,
            first: response,
        })
    }

    fn validate(&self, response: &LookupResponse, worker: Worker) -> Result<(), Error> {
        self.window._lease.ensure_open()?;
        if response.binding != self.window.binding(worker)?
            || !Arc::ptr_eq(&response._lease, &self.window._lease)
        {
            return Err(Error::Binding);
        }
        if response.bytes.len() != self.window.publication.width * 4 {
            return Err(Error::Shape);
        }
        Ok(())
    }
}

impl HalfLookup {
    pub fn remaining_share(&self) -> LookupShare<'_> {
        self.pending.share(Worker::One)
    }

    pub fn freeze_second(self, response: Result<LookupResponse, Error>) -> Result<Window, Error> {
        let response = response?;
        self.pending.validate(&response, Worker::One)?;
        let mut pending = self.pending;
        let bytes = Zeroizing::new(payload::collect(
            self.first
                .bytes
                .iter()
                .zip(response.bytes.iter())
                .map(|(&a, &b)| a ^ b),
        )?);
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        if !bool::from(digest.ct_eq(&pending.window.publication.row_hashes[*pending.token])) {
            return Err(Error::Verification);
        }
        // No float interpretation or local use occurs before the row hash check.
        let values = Zeroizing::new(payload::collect(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        )?);
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numeric);
        }
        pending.window.tentative = values;
        pending.window.signed = Zeroizing::new(Vec::new());
        pending.window.completed += 1;
        Ok(pending.window)
    }
}

/// Only successful complete-window verification can construct this value. It
/// owns the last toy-stage output; it is not a KV/sampler/decoder continuation.
/// Its live-capacity lease remains reserved until the accepted payload drops.
///
/// ```compile_fail
/// use pllm_core::gkm::AcceptedWindow;
/// fn fabricate() -> AcceptedWindow { AcceptedWindow { steps: 1, ..todo!() } }
/// ```
pub struct AcceptedWindow {
    request: [u8; 32],
    context: [u8; 32],
    publication: [u8; 32],
    plan: [u8; 32],
    steps: usize,
    profile: SharingProfile,
    values: Zeroizing<Vec<f32>>,
    signed: Zeroizing<Vec<i64>>,
    _lease: Arc<Lease>,
}
impl AcceptedWindow {
    pub fn sharing_profile(&self) -> SharingProfile {
        self.profile
    }
    pub fn request_id(&self) -> [u8; 32] {
        self.request
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
    /// Empty when the final stage was a lookup.
    pub fn signed_linear_output(&self) -> &[i64] {
        &self.signed
    }
    pub fn context(&self) -> [u8; 32] {
        self.context
    }
    pub fn publication(&self) -> [u8; 32] {
        self.publication
    }
    pub fn plan_digest(&self) -> [u8; 32] {
        self.plan
    }
    pub fn steps(&self) -> usize {
        self.steps
    }
}

#[cfg(test)]
#[path = "window_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "resource_tests.rs"]
mod resource_tests;
