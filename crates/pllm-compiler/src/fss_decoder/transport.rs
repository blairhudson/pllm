//! Bounded callback transport for the public-weight two-worker/dealer contract.
//!
//! Workers own one Runtime and one request-long replay guard. Only prepare() may
//! issue material, through the public no-input Program material visitor. External
//! transport MUST authenticate the dealer, peer, role and fresh session; native
//! checksums/context binding do not provide network authentication. Callbacks may
//! retain only the admitted bounded streaming slots and must erase/drain them on
//! cancellation. Python/HTTP/queue implementations are deliberately outside here.
use super::{Driver, Error, MaterialVisitor, NativeOutput, Program, TokenShares, Worker};
use pllm_garble::{
    fusefss_backend::{
        composition::{wavefront, Event, Material, Plan, Runtime as SerialRuntime},
        wire::{ImportGuard, MaterialPacket, ENVELOPE_BYTES},
        Budget, MAX_ALLOCATION_BYTES,
    },
    shared_rescale_reference::Context,
};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

#[path = "transport_batch.rs"]
mod batch;
pub use batch::{
    BatchResources, MaterialDelivery, MaterialRecord, CALLBACK_METADATA_BYTES_BOUND,
    MATERIAL_BATCH_BODY_BYTES_BOUND, MATERIAL_BATCH_ENVELOPE_BYTES_BOUND,
    MATERIAL_BATCH_RECORDS_BOUND, MATERIAL_BATCH_RECORD_OVERHEAD_BYTES,
};

// Includes native reply/delivery buffers, Python byte copies, and single-slot
// authenticated queue/HTTP ownership at each end. Arbitrary callback retention
// is forbidden, not silently included in a process-wide/RSS claim.
const PACKET_SLOTS: usize = 6;
const FRAME_COPY_SLOTS: usize = 8;

#[derive(Clone, Copy)]
pub struct Kernel {
    pub context: Context,
    pub plan_digest: [u8; 32],
    /// Zero-based public kernel ordinal, identical for dealer and both workers.
    pub ordinal: usize,
}
impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("ordinal", &self.ordinal)
            .field("plan_digest", &self.plan_digest)
            .field("program", &self.context.plan)
            .field("session", &self.context.session)
            .field("operation", &self.context.operation)
            .field("first_tensor_index", &self.context.first_tensor_index)
            .finish()
    }
}
#[derive(Clone, Copy, Debug)]
pub struct MaterialRequest {
    pub kernel: Kernel,
    pub role: u8,
    pub pc: usize,
    pub packet_bytes_bound: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct PeerRequest {
    pub kernel: Kernel,
    pub role: u8,
    /// Zero-based within this kernel, including all primitive rounds.
    pub frame: usize,
    pub frame_bytes: usize,
}

/// Native-owned, fallibly allocated, erasing callback response. No caller can
/// replace its allocation or widen its exact public bound. Exactly one write;
/// a rejected/second write burns the reply even if the callback ignores the error.
pub struct Reply {
    bytes: Zeroizing<Vec<u8>>,
    bound: usize,
    state: u8,
}
impl Reply {
    fn new(bound: usize) -> Result<Self, Error> {
        if bound == 0 || bound > MAX_ALLOCATION_BYTES {
            return Err(Error::Resource);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(bound)
            .map_err(|_| Error::Allocation)?;
        if bytes.capacity() != bound {
            return Err(Error::Resource);
        }
        Ok(Self {
            bytes: Zeroizing::new(bytes),
            bound,
            state: 0,
        })
    }
    pub fn bound(&self) -> usize {
        self.bound
    }
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let fresh = self.state == 0;
        self.state = 2;
        if !fresh {
            return Err(Error::Consumed);
        }
        if bytes.is_empty() || bytes.len() > self.bound {
            return Err(Error::Resource);
        }
        self.bytes.extend_from_slice(bytes);
        self.state = 1;
        Ok(())
    }
    fn bytes(&self) -> Result<&[u8], Error> {
        if self.state != 1 {
            return Err(Error::Consumed);
        }
        Ok(&self.bytes)
    }
}

pub trait DealerTransport {
    /// Accept exactly this recipient's opaque material before returning. Both
    /// recipients must accept the current pc before the dealer advances.
    fn deliver(&mut self, request: MaterialRequest, packet: &[u8]) -> Result<(), Error>;
    /// Explicit opt-in batch ACK. Every record must be accepted before success;
    /// a partial/lost ACK burns the entire request, never a retryable suffix.
    /// Oversized records still use deliver() with its original ACK contract.
    fn deliver_batch(&mut self, _role: u8, _records: &[MaterialRecord<'_>]) -> Result<(), Error> {
        Err(Error::Unsupported)
    }
    /// A consuming bridge may expose cancellation between bounded deliveries.
    fn check_active(&mut self) -> Result<(), Error> {
        Ok(())
    }
    /// No unsent/undelivered retained buffers may remain after success.
    fn finish(&mut self) -> Result<(), Error>;
    /// Erase retained copies, close queues and wake blocked consumers/producers.
    /// Must not panic. Native adapters invoke this at most once per failed run.
    fn cancel(&mut self);
}
pub trait WorkerTransport {
    fn material(&mut self, request: MaterialRequest, reply: &mut Reply) -> Result<(), Error>;
    /// Send one own-party opaque frame and receive the matching peer frame.
    /// Transport must permit simultaneous sends without unbounded buffering.
    fn exchange(
        &mut self,
        request: PeerRequest,
        frame: &[u8],
        reply: &mut Reply,
    ) -> Result<(), Error>;
    /// Verify the role endpoint has consumed its complete admitted stream.
    fn finish(&mut self) -> Result<(), Error>;
    /// Must not panic; retire retained buffers and wake blocked endpoints.
    fn cancel(&mut self);
    #[cfg(test)]
    fn observe(&mut self, _selection: usize, _node: usize, _shape: &[usize], _shares: &[u64]) {}
}

/// Compiler-owned public admission, never accepted back as a caller's proof.
/// Every prepare/execute recomputes this before any issuance/input acceptance.
#[derive(Clone, Copy, Debug)]
pub struct Admission {
    online: super::OnlineSchedule,
    delivery: MaterialDelivery,
    batch: Option<BatchResources>,
    kernels: usize,
    requests: usize,
    frames: usize,
    peer_frame_bytes: usize,
    packet_bytes: usize,
    cumulative_packet_bytes: usize,
    guard_capacity: usize,
    guard_reservation_bytes: usize,
    guard_disk_bytes: u64,
    combined_bytes: usize,
    transcript: [u8; 32],
}
impl Admission {
    pub fn material_delivery(&self) -> MaterialDelivery {
        self.delivery
    }
    pub fn delivery_batch(&self) -> Option<&BatchResources> {
        self.batch.as_ref()
    }
    pub fn kernels(&self) -> usize {
        self.kernels
    }
    pub fn primitive_requests_per_party(&self) -> usize {
        self.requests
    }
    pub fn frames_per_party(&self) -> usize {
        self.frames
    }
    /// Largest retained per-party primitive frame set. This bounds each single
    /// peer frame; cumulative request traffic is a separate resource.
    pub fn peer_frame_bytes_bound(&self) -> usize {
        self.peer_frame_bytes
    }
    pub fn largest_packet_bytes(&self) -> usize {
        self.packet_bytes
    }
    pub fn cumulative_packet_bytes_per_party_bound(&self) -> usize {
        self.cumulative_packet_bytes
    }
    pub fn guard_capacity(&self) -> usize {
        self.guard_capacity
    }
    pub fn guard_reservation_bytes(&self) -> usize {
        self.guard_reservation_bytes
    }
    pub fn guard_disk_bytes(&self) -> u64 {
        self.guard_disk_bytes
    }
    pub fn combined_reservation_bytes(&self) -> usize {
        self.combined_bytes
    }
    pub fn transcript_digest(&self) -> [u8; 32] {
        self.transcript
    }
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Resource)
}
fn hash_start(program: &Program) -> Sha256 {
    Sha256::new()
        .chain_update(b"pllm.fss-decoder.transport-schedule.v1\0")
        .chain_update(program.digest())
}
fn hash_kernel(hash: &mut Sha256, k: Kernel) {
    hash.update((k.ordinal as u64).to_le_bytes());
    hash.update(k.plan_digest);
    hash.update(k.context.plan);
    hash.update(k.context.session);
    hash.update(k.context.operation);
    hash.update(k.context.first_tensor_index.to_le_bytes());
}

pub fn admit(program: &Program, context: Context) -> Result<Admission, Error> {
    admit_with_delivery(program, context, MaterialDelivery::Single)
}

pub fn admit_with_delivery(
    program: &Program,
    context: Context,
    delivery: MaterialDelivery,
) -> Result<Admission, Error> {
    // Keep every exact (role,nonce) claim across the complete request. Indexed,
    // preallocated storage removes the quadratic scan; its complete capacity is
    // priced before kernel visitation and private input. No per-kernel reset.
    let requests = program.resources().primitive_requests_per_party;
    let guard_capacity = requests.max(1);
    let file_guard = program.capacity() == super::Capacity::Decoder;
    let (guard_reservation_bytes, guard_disk_bytes) = if file_guard {
        let resources = ImportGuard::indexed_file_resources(guard_capacity)?;
        (resources.resident_bytes, resources.disk_bytes)
    } else {
        (ImportGuard::indexed_reservation_bytes(guard_capacity)?, 0)
    };
    struct Scan {
        online: super::OnlineSchedule,
        kernels: usize,
        requests: usize,
        frames: usize,
        peer_frame_bytes: usize,
        packet: usize,
        cumulative: usize,
        peak: usize,
        guard: usize,
        file_guard: bool,
        max_cumulative: usize,
        hash: Sha256,
    }
    impl MaterialVisitor for Scan {
        fn visit(&mut self, context: Context, plan: &Plan<'_>) -> Result<(), Error> {
            let r = if self.file_guard {
                plan.transport_resources_indexed_file(PACKET_SLOTS, self.guard)?
            } else {
                plan.transport_resources_indexed(PACKET_SLOTS, self.guard)?
            };
            let k = Kernel {
                context,
                plan_digest: plan.digest(),
                ordinal: self.kernels,
            };
            hash_kernel(&mut self.hash, k);
            let wave = if self.online.uses_wavefront() && !plan.is_atomic_vector() {
                Some(wavefront::Schedule::new(plan)?)
            } else {
                None
            };
            if let Some(w) = &wave {
                self.hash.update(w.digest());
            }
            self.kernels = add(self.kernels, 1)?;
            self.requests = add(self.requests, r.secret_steps)?;
            self.frames = add(
                self.frames,
                wave.as_ref()
                    .map_or(plan.resources().frames_per_party, |w| w.resources().rounds),
            )?;
            let frame_bytes = wave.as_ref().map_or(r.retained_frame_bytes_per_party, |w| {
                w.resources().largest_frame_bytes
            });
            self.peer_frame_bytes = self.peer_frame_bytes.max(frame_bytes);
            self.packet = self.packet.max(r.largest_packet_bytes_bound);
            self.cumulative = add(self.cumulative, r.cumulative_packet_bytes_per_party_bound)?;
            if self.cumulative > self.max_cumulative {
                return Err(Error::Resource);
            }
            let native = add(
                mul(2, r.combined_worker_reservation_bytes)?,
                r.combined_dealer_export_reservation_bytes,
            )?;
            let frame_copies = mul(2 * FRAME_COPY_SLOTS, frame_bytes)?;
            let extra = wave
                .as_ref()
                .map_or(0, |w| w.resources().additional_reservation_bytes);
            self.peak = self.peak.max(add(add(native, frame_copies)?, extra)?);
            Ok(())
        }
        fn cancel(&mut self) {}
    }
    let mut scan = Scan {
        online: program.online_schedule(),
        kernels: 0,
        requests: 0,
        frames: 0,
        peer_frame_bytes: 0,
        packet: 0,
        cumulative: 0,
        peak: 0,
        guard: guard_capacity,
        file_guard,
        max_cumulative: if file_guard { 32 << 30 } else { 1 << 30 },
        hash: hash_start(program),
    };
    program.visit_material(context, &mut scan)?;
    if scan.requests != requests || scan.kernels != program.resources().kernels {
        return Err(Error::Context);
    }
    let r = program.resources();
    let owners = add(
        add(
            mul(3, r.metadata_bytes)?,
            mul(2, r.live_array_bytes_per_party)?,
        )?,
        mul(3, r.source_table_cache_bytes_per_role)?,
    )?;
    // Each role has independent request-long cache metadata and retained tables,
    // including tables unrelated to the current Plan. Cache ownership grows
    // monotonically: its final size conservatively overlaps every transport peak.
    // Current Plan table allowances remain in scan.peak to cover miss-time
    // generated-value staging alongside the final immutable cached owner.
    // Fixed native adapter/ledger/callback metadata slack, in addition to exact
    // guard, packet, frame, Plan, source/program and tensor-array reservations.
    let mut combined = add(add(owners, scan.peak)?, 64 * 1024)?;
    let batch = match delivery {
        MaterialDelivery::Single => None,
        MaterialDelivery::BatchV1 => {
            let resources = BatchResources::new()?;
            combined = add(combined, resources.native_reservation_bytes())?;
            combined = add(combined, resources.python_overlap_bytes())?;
            Some(resources)
        }
    };
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Admission {
        online: program.online_schedule(),
        delivery,
        batch,
        kernels: scan.kernels,
        requests,
        frames: scan.frames,
        peer_frame_bytes: scan.peer_frame_bytes,
        packet_bytes: scan.packet,
        cumulative_packet_bytes: scan.cumulative,
        guard_capacity,
        guard_reservation_bytes,
        guard_disk_bytes,
        combined_bytes: combined,
        transcript: scan.hash.finalize().into(),
    })
}

struct Dealer<'t, 'b, T: DealerTransport> {
    transport: &'t mut T,
    batches: Option<batch::Queues<'b>>,
    admission: Admission,
    kernels: usize,
    requests: usize,
    hash: Sha256,
    cancelled: bool,
    complete: bool,
}
impl<T: DealerTransport> Dealer<'_, '_, T> {
    fn abort(&mut self) {
        if !self.cancelled {
            self.cancelled = true;
            if let Some(queues) = &mut self.batches {
                queues.clear();
            }
            self.transport.cancel();
        }
    }
    fn before_pair(&mut self, bound: usize) -> Result<bool, Error> {
        self.transport.check_active()?;
        if bound > self.admission.packet_bytes || self.requests >= self.admission.requests {
            return Err(Error::Resource);
        }
        let singleton = self.batches.is_none() || batch::Queues::singleton(bound)?;
        if let Some(queues) = &mut self.batches {
            if singleton || queues.needs_flush(bound)? {
                queues.flush(self.transport)?;
            }
        }
        Ok(singleton)
    }
    fn deliver_pair(
        &mut self,
        requests: [MaterialRequest; 2],
        packets: [MaterialPacket<'_>; 2],
        singleton: bool,
    ) -> Result<(), Error> {
        for (request, packet) in requests.iter().zip(&packets) {
            if packet.as_bytes().len() > request.packet_bytes_bound {
                return Err(Error::Resource);
            }
        }
        if let Some(queues) = self.batches.as_mut().filter(|_| !singleton) {
            queues.push_pair(requests, [packets[0].as_bytes(), packets[1].as_bytes()])?;
            drop(packets);
            if queues.full() {
                queues.flush(self.transport)?;
            }
        } else {
            for (request, packet) in requests.into_iter().zip(&packets) {
                self.transport.deliver(request, packet.as_bytes())?;
                self.transport.check_active()?;
            }
            if let Some(queues) = &mut self.batches {
                queues.accept_singleton()?;
            }
        }
        self.requests = add(self.requests, 1)?;
        Ok(())
    }
}
impl<T: DealerTransport> Drop for Dealer<'_, '_, T> {
    fn drop(&mut self) {
        if !self.complete {
            self.abort();
        }
    }
}
impl<T: DealerTransport> MaterialVisitor for Dealer<'_, '_, T> {
    fn visit(&mut self, context: Context, plan: &Plan<'_>) -> Result<(), Error> {
        let k = Kernel {
            context,
            plan_digest: plan.digest(),
            ordinal: self.kernels,
        };
        hash_kernel(&mut self.hash, k);
        let wave = if self.admission.online.uses_wavefront() && !plan.is_atomic_vector() {
            Some(wavefront::Schedule::new(plan)?)
        } else {
            None
        };
        if let Some(w) = &wave {
            self.hash.update(w.digest());
        }
        self.kernels = add(self.kernels, 1)?;
        let mut deliver = |pc| {
            let bound = plan.step_transport_resources(pc)?.packet_bytes_bound;
            let singleton = self.before_pair(bound)?;
            let [left, right] = plan.issue_step(pc, context)?;
            let left = left.export_packet()?;
            let right = right.export_packet()?;
            let requests = [0, 1].map(|role| MaterialRequest {
                kernel: k,
                role,
                pc,
                packet_bytes_bound: bound,
            });
            self.deliver_pair(requests, [left, right], singleton)?;
            Ok::<_, Error>(())
        };
        if let Some(w) = &wave {
            for pc in w.material_steps() {
                deliver(pc)?;
            }
        } else {
            for pc in plan.material_steps() {
                deliver(pc)?;
            }
        }
        Ok(())
    }
    fn cancel(&mut self) {
        self.abort();
    }
}

/// Issue only through the immutable public schedule. Caller must reserve a fresh
/// authenticated session externally; never retry a failed run with the same
/// context. No input shares or peer-frame receiver exist on this interface.
pub fn prepare(
    program: &Program,
    context: Context,
    transport: &mut impl DealerTransport,
) -> Result<(), Error> {
    prepare_with_delivery(program, context, MaterialDelivery::Single, transport)
}

pub fn prepare_with_delivery(
    program: &Program,
    context: Context,
    delivery: MaterialDelivery,
    transport: &mut impl DealerTransport,
) -> Result<(), Error> {
    let admission = match admit_with_delivery(program, context, delivery) {
        Ok(a) => a,
        Err(e) => {
            transport.cancel();
            return Err(e);
        }
    };
    // This independent owner outlives Program::visit_material and every Plan
    // Budget it creates. Only copied opaque packets borrow it.
    let budget = if admission.batch.is_some() {
        match Budget::new(BatchResources::packet_budget_bytes()) {
            Ok(b) => Some(b),
            Err(error) => {
                transport.cancel();
                return Err(error);
            }
        }
    } else {
        None
    };
    let batches = match budget.as_ref().map(batch::Queues::new).transpose() {
        Ok(queues) => queues,
        Err(error) => {
            transport.cancel();
            return Err(error);
        }
    };
    let mut dealer = Dealer {
        transport,
        batches,
        admission,
        kernels: 0,
        requests: 0,
        hash: hash_start(program),
        cancelled: false,
        complete: false,
    };
    let result = (|| {
        program.visit_material(context, &mut dealer)?;
        if let Some(queues) = &mut dealer.batches {
            queues.flush(dealer.transport)?;
            if !queues.complete(admission.requests) {
                return Err(Error::Context);
            }
        }
        if dealer.kernels != admission.kernels
            || dealer.requests != admission.requests
            || dealer.hash.clone().finalize().as_slice() != admission.transcript
        {
            return Err(Error::Context);
        }
        dealer.transport.finish()
    })();
    if result.is_err() {
        dealer.abort();
    } else {
        dealer.complete = true;
    }
    result
}

struct Role<'g, 't, T: WorkerTransport> {
    transport: &'t mut T,
    role: u8,
    guard: ImportGuard<'g>,
    admission: Admission,
    kernels: usize,
    requests: usize,
    frames: usize,
    hash: Sha256,
    cancelled: bool,
    complete: bool,
}

// Fixed stack storage avoids another unpriced/fallible heap owner per kernel.
#[allow(clippy::large_enum_variant)]
pub(super) enum Execution<'s, 'p, 'a> {
    Serial(SerialRuntime<'p, 'a>),
    Wave(wavefront::Runtime<'s, 'p, 'a>),
}
impl<'a> Execution<'_, '_, 'a> {
    pub(super) fn poll(&mut self) -> Result<Event<'a>, Error> {
        match self {
            Self::Serial(r) => r.poll(),
            Self::Wave(r) => r.poll(),
        }
    }
    pub(super) fn feed_result(
        &mut self,
        m: Result<Material<'a>, Error>,
    ) -> Result<Event<'a>, Error> {
        match self {
            Self::Serial(r) => r.feed_result(m),
            Self::Wave(r) => r.feed_result(m),
        }
    }
    pub(super) fn advance(&mut self, p: &[u8]) -> Result<Event<'a>, Error> {
        match self {
            Self::Serial(r) => r.advance(p),
            Self::Wave(r) => r.advance(p),
        }
    }
    fn cancel(&mut self) {
        match self {
            Self::Serial(r) => r.cancel(),
            Self::Wave(r) => r.cancel(),
        }
    }
}
impl<T: WorkerTransport> Role<'_, '_, T> {
    fn abort(&mut self) {
        if !self.cancelled {
            self.cancelled = true;
            self.transport.cancel();
        }
    }
}
impl<T: WorkerTransport> Drop for Role<'_, '_, T> {
    fn drop(&mut self) {
        if !self.complete {
            self.abort();
        }
    }
}
impl<T: WorkerTransport> Driver for Role<'_, '_, T> {
    fn run(
        &mut self,
        context: Context,
        plan: &Plan<'_>,
        input: &[u64],
    ) -> Result<NativeOutput, Error> {
        let k = Kernel {
            context,
            plan_digest: plan.digest(),
            ordinal: self.kernels,
        };
        hash_kernel(&mut self.hash, k);
        let wave = if self.admission.online.uses_wavefront() && !plan.is_atomic_vector() {
            Some(wavefront::Schedule::new(plan)?)
        } else {
            None
        };
        if let Some(w) = &wave {
            self.hash.update(w.digest());
        }
        self.kernels = add(self.kernels, 1)?;
        if self.kernels > self.admission.kernels {
            return Err(Error::Context);
        }
        let expected_frames = wave
            .as_ref()
            .map_or(plan.resources().frames_per_party, |w| w.resources().rounds);
        let mut runtime = if let Some(w) = &wave {
            Execution::Wave(w.runtime(self.role, context, input)?)
        } else {
            Execution::Serial(plan.runtime(self.role, context, input)?)
        };
        let result = (|| {
            let mut event = runtime.poll()?;
            let mut frames = 0usize;
            loop {
                event = match event {
                    Event::NeedMaterial { pc } => {
                        if self.requests >= self.admission.requests {
                            return Err(Error::Resource);
                        }
                        let bound = plan.step_transport_resources(pc)?.packet_bytes_bound;
                        if bound > self.admission.packet_bytes {
                            return Err(Error::Resource);
                        }
                        let mut reply = Reply::new(bound)?;
                        self.transport.material(
                            MaterialRequest {
                                kernel: k,
                                role: self.role,
                                pc,
                                packet_bytes_bound: bound,
                            },
                            &mut reply,
                        )?;
                        let bytes = reply.bytes()?;
                        if bytes.len() < ENVELOPE_BYTES {
                            return Err(Error::Context);
                        }
                        let nonce = bytes[48..80].try_into().map_err(|_| Error::Context)?;
                        self.requests = add(self.requests, 1)?;
                        let material = plan.import_step_bytes(
                            pc,
                            context,
                            self.role,
                            nonce,
                            bytes,
                            &mut self.guard,
                        );
                        runtime.feed_result(material)?
                    }
                    Event::Frame(frame) => {
                        if self.frames >= self.admission.frames {
                            return Err(Error::Resource);
                        }
                        let len = frame.as_bytes().len();
                        if len > self.admission.peer_frame_bytes {
                            return Err(Error::Resource);
                        }
                        let mut reply = Reply::new(len)?;
                        self.transport.exchange(
                            PeerRequest {
                                kernel: k,
                                role: self.role,
                                frame: frames,
                                frame_bytes: len,
                            },
                            frame.as_bytes(),
                            &mut reply,
                        )?;
                        let peer = reply.bytes()?;
                        if peer.len() != len {
                            return Err(Error::Context);
                        }
                        frames = add(frames, 1)?;
                        self.frames = add(self.frames, 1)?;
                        runtime.advance(peer)?
                    }
                    Event::Output(output) => {
                        if frames != expected_frames {
                            return Err(Error::Context);
                        }
                        return NativeOutput::from_native_plan(output, plan);
                    }
                };
            }
        })();
        if result.is_err() {
            runtime.cancel();
            self.abort();
        }
        result
    }
    fn cancel(&mut self) {
        self.abort();
    }
    #[cfg(test)]
    fn observe(&mut self, selection: usize, node: usize, shape: &[usize], shares: &[u64]) {
        self.transport.observe(selection, node, shape, shares);
    }
}

/// Consume exactly one party's input. Admission runs before Worker construction.
/// One guard persists across every kernel until success, failure or unwind; it
/// cannot be reset through this API. The SDK must enforce one execution per fresh
/// authenticated session, including across process restarts and failed calls.
pub fn execute(
    program: &Program,
    role: u8,
    context: Context,
    input: Zeroizing<Vec<u64>>,
    transport: &mut impl WorkerTransport,
) -> Result<TokenShares, Error> {
    execute_with_delivery(
        program,
        role,
        context,
        input,
        MaterialDelivery::Single,
        transport,
    )
}

pub fn execute_with_delivery(
    program: &Program,
    role: u8,
    context: Context,
    input: Zeroizing<Vec<u64>>,
    delivery: MaterialDelivery,
    transport: &mut impl WorkerTransport,
) -> Result<TokenShares, Error> {
    let admission = match admit_with_delivery(program, context, delivery) {
        Ok(a) => a,
        Err(e) => {
            transport.cancel();
            return Err(e);
        }
    };
    if role > 1 {
        transport.cancel();
        return Err(Error::Domain);
    }
    let budget = match Budget::new(admission.guard_reservation_bytes) {
        Ok(b) => b,
        Err(e) => {
            transport.cancel();
            return Err(e);
        }
    };
    let guard = match if admission.guard_disk_bytes != 0 {
        ImportGuard::new_indexed_file(
            admission.guard_capacity,
            admission.guard_disk_bytes,
            &budget,
        )
    } else {
        ImportGuard::new_indexed(admission.guard_capacity, &budget)
    } {
        Ok(g) => g,
        Err(e) => {
            transport.cancel();
            return Err(e);
        }
    };
    let mut driver = Role {
        transport,
        role,
        guard,
        admission,
        kernels: 0,
        requests: 0,
        frames: 0,
        hash: hash_start(program),
        cancelled: false,
        complete: false,
    };
    let result = (|| {
        let output = Worker::new(program, role, context, input)?.execute(&mut driver)?;
        if driver.kernels != admission.kernels
            || driver.requests != admission.requests
            || driver.frames != admission.frames
            || driver.hash.clone().finalize().as_slice() != admission.transcript
        {
            return Err(Error::Context);
        }
        driver.transport.finish()?;
        Ok(output)
    })();
    if result.is_err() {
        driver.abort();
    } else {
        driver.complete = true;
    }
    result
}

#[cfg(test)]
#[path = "transport_tests.rs"]
pub(super) mod tests;
