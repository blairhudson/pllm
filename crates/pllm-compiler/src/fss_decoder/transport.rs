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
        composition::{Event, Plan},
        wire::{ImportGuard, ENVELOPE_BYTES},
        Budget, MAX_ALLOCATION_BYTES,
    },
    shared_rescale_reference::Context,
};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

// Includes native reply/delivery buffers, Python byte copies, and single-slot
// authenticated queue/HTTP ownership at each end. Arbitrary callback retention
// is forbidden, not silently included in a process-wide/RSS claim.
const PACKET_SLOTS: usize = 6;
const FRAME_COPY_SLOTS: usize = 8;
const MAX_GUARD: usize = 4096;

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
    kernels: usize,
    requests: usize,
    frames: usize,
    packet_bytes: usize,
    cumulative_packet_bytes: usize,
    guard_capacity: usize,
    combined_bytes: usize,
    transcript: [u8; 32],
}
impl Admission {
    pub fn kernels(&self) -> usize {
        self.kernels
    }
    pub fn primitive_requests_per_party(&self) -> usize {
        self.requests
    }
    pub fn frames_per_party(&self) -> usize {
        self.frames
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
    // Whole-request guard: no scope reset, retry, capacity increase or per-kernel
    // exception. SDK must admit a fresh authenticated session before accepting
    // input. Existing 4096-claim ceiling remains an explicit fail-closed limit.
    let requests = program.resources().primitive_requests_per_party;
    if requests > MAX_GUARD {
        return Err(Error::Resource);
    }
    let guard_capacity = requests.max(1);
    struct Scan {
        kernels: usize,
        requests: usize,
        frames: usize,
        packet: usize,
        cumulative: usize,
        peak: usize,
        guard: usize,
        hash: Sha256,
    }
    impl MaterialVisitor for Scan {
        fn visit(&mut self, context: Context, plan: &Plan<'_>) -> Result<(), Error> {
            let r = plan.transport_resources(PACKET_SLOTS, self.guard)?;
            let k = Kernel {
                context,
                plan_digest: plan.digest(),
                ordinal: self.kernels,
            };
            hash_kernel(&mut self.hash, k);
            self.kernels = add(self.kernels, 1)?;
            self.requests = add(self.requests, r.secret_steps)?;
            self.frames = add(self.frames, plan.resources().frames_per_party)?;
            self.packet = self.packet.max(r.largest_packet_bytes_bound);
            self.cumulative = add(self.cumulative, r.cumulative_packet_bytes_per_party_bound)?;
            if self.cumulative > 1 << 30 {
                return Err(Error::Resource);
            }
            let native = add(
                mul(2, r.combined_worker_reservation_bytes)?,
                r.combined_dealer_export_reservation_bytes,
            )?;
            let frame_copies = mul(2 * FRAME_COPY_SLOTS, r.retained_frame_bytes_per_party)?;
            self.peak = self.peak.max(add(native, frame_copies)?);
            Ok(())
        }
        fn cancel(&mut self) {}
    }
    let mut scan = Scan {
        kernels: 0,
        requests: 0,
        frames: 0,
        packet: 0,
        cumulative: 0,
        peak: 0,
        guard: guard_capacity,
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
    let combined = add(add(owners, scan.peak)?, 64 * 1024)?;
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Admission {
        kernels: scan.kernels,
        requests,
        frames: scan.frames,
        packet_bytes: scan.packet,
        cumulative_packet_bytes: scan.cumulative,
        guard_capacity,
        combined_bytes: combined,
        transcript: scan.hash.finalize().into(),
    })
}

struct Dealer<'t, T: DealerTransport> {
    transport: &'t mut T,
    admission: Admission,
    kernels: usize,
    requests: usize,
    hash: Sha256,
    cancelled: bool,
    complete: bool,
}
impl<T: DealerTransport> Dealer<'_, T> {
    fn abort(&mut self) {
        if !self.cancelled {
            self.cancelled = true;
            self.transport.cancel();
        }
    }
}
impl<T: DealerTransport> Drop for Dealer<'_, T> {
    fn drop(&mut self) {
        if !self.complete {
            self.abort();
        }
    }
}
impl<T: DealerTransport> MaterialVisitor for Dealer<'_, T> {
    fn visit(&mut self, context: Context, plan: &Plan<'_>) -> Result<(), Error> {
        let k = Kernel {
            context,
            plan_digest: plan.digest(),
            ordinal: self.kernels,
        };
        hash_kernel(&mut self.hash, k);
        self.kernels = add(self.kernels, 1)?;
        for pc in plan.material_steps() {
            let bound = plan.step_transport_resources(pc)?.packet_bytes_bound;
            if bound > self.admission.packet_bytes {
                return Err(Error::Resource);
            }
            let [left, right] = plan.issue_step(pc, context)?;
            let left = left.export_packet()?;
            let right = right.export_packet()?;
            for (role, packet) in [(0, &left), (1, &right)] {
                if packet.as_bytes().len() > bound {
                    return Err(Error::Resource);
                }
                self.transport.deliver(
                    MaterialRequest {
                        kernel: k,
                        role,
                        pc,
                        packet_bytes_bound: bound,
                    },
                    packet.as_bytes(),
                )?;
            }
            self.requests = add(self.requests, 1)?;
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
    let admission = match admit(program, context) {
        Ok(a) => a,
        Err(e) => {
            transport.cancel();
            return Err(e);
        }
    };
    let mut dealer = Dealer {
        transport,
        admission,
        kernels: 0,
        requests: 0,
        hash: hash_start(program),
        cancelled: false,
        complete: false,
    };
    let result = (|| {
        program.visit_material(context, &mut dealer)?;
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
        self.kernels = add(self.kernels, 1)?;
        if self.kernels > self.admission.kernels {
            return Err(Error::Context);
        }
        let mut runtime = plan.runtime(self.role, context, input)?;
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
                        if frames != plan.resources().frames_per_party {
                            return Err(Error::Context);
                        }
                        return NativeOutput::from_native(output);
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
    let admission = match admit(program, context) {
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
    let budget = match mul(admission.guard_capacity, 33).and_then(Budget::new) {
        Ok(b) => b,
        Err(e) => {
            transport.cancel();
            return Err(e);
        }
    };
    let guard = match ImportGuard::new(admission.guard_capacity, &budget) {
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
mod tests;
