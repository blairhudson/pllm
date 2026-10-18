use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use zeroize::Zeroizing;

use super::arithmetic::{add, canonical, dot, mul, sub, OsRandom};
use super::code::Sparse;
use super::memory::{Memory, MemoryStats, Words};
use super::publication::{Material, ValidatedPublication};
use super::{
    Error, Result, BATCH_SIZE, CHECK_WEIGHT, DIMENSION, LENGTH, MAX_ATTEMPTS, PRIVACY_WEIGHT,
    REPETITIONS, SECONDARY_DIMENSION, SECONDARY_LENGTH,
};

const BATCH_SLOT: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Single,
    Batch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub publication: [u8; 32],
    pub session: [u8; 32],
    pub attempt: u32,
    pub mode: Mode,
    pub slot: u8,
}

#[derive(Clone, Copy)]
pub struct RequestView<'a> {
    pub header: Header,
    pub values: &'a [u32],
}

#[derive(Clone, Copy)]
pub struct ReplyView<'a> {
    pub header: Header,
    pub values: &'a [u32],
}

#[derive(Clone, Copy)]
pub struct PrimaryChallengeView<'a> {
    pub header: Header,
    /// Interleaved position/nonzero-coefficient u32 pairs, not dense masks.
    pub entries: [&'a [u32]; REPETITIONS],
}

#[derive(Clone, Copy)]
pub struct AuxiliaryReplyView<'a> {
    pub header: Header,
    /// Repetition-major, each response is the ordered B*n stacked vector.
    pub values: &'a [u32],
}

pub struct Request {
    header: Header,
    values: Words,
}

impl Request {
    pub fn view(&self) -> RequestView<'_> {
        RequestView {
            header: self.header,
            values: self.values.as_slice(),
        }
    }
}

pub struct Reply {
    header: Header,
    values: Words,
}

impl Reply {
    pub fn view(&self) -> ReplyView<'_> {
        ReplyView {
            header: self.header,
            values: self.values.as_slice(),
        }
    }
}

pub struct PrimaryChallenge {
    header: Header,
    entries: [Sparse; REPETITIONS],
}

impl PrimaryChallenge {
    pub fn view(&self) -> PrimaryChallengeView<'_> {
        PrimaryChallengeView {
            header: self.header,
            entries: [self.entries[0].pairs(), self.entries[1].pairs()],
        }
    }
}

pub struct AuxiliaryReply {
    header: Header,
    values: Words,
}

impl AuxiliaryReply {
    pub fn view(&self) -> AuxiliaryReplyView<'_> {
        AuxiliaryReplyView {
            header: self.header,
            values: self.values.as_slice(),
        }
    }
}

struct Control {
    issued: AtomicU32,
    live: AtomicBool,
}

struct Lease(Arc<Control>);

impl Drop for Lease {
    fn drop(&mut self) {
        self.0.live.store(false, Ordering::Release);
    }
}

pub(super) struct Claim {
    pub(super) header: Header,
    pub(super) input: Words,
    pub(super) correction: Words,
}

/// Private one-use owner: no Clone, Debug, serde, or unverified output accessor.
///
/// ```compile_fail
/// use pllm_core::maverick_batch::Query;
/// fn duplicate(query: Query) { let _ = query.clone(); }
/// ```
///
/// ```compile_fail
/// use pllm_core::maverick_batch::{Client, Query, ReplyView};
/// fn reuse(client: &mut Client, query: Query, reply: ReplyView<'_>) {
///     let _ = client.finish_query(query, reply);
///     let _ = client.finish_query(query, reply);
/// }
/// ```
pub struct Query {
    pub(super) claim: Claim,
    _lease: Lease,
}

/// Private owner for exactly two masked claims in the bound ordered matrix list.
///
/// ```compile_fail
/// use pllm_core::maverick_batch::Batch;
/// fn duplicate(batch: Batch) { let _ = batch.clone(); }
/// ```
///
/// ```compile_fail
/// use pllm_core::maverick_batch::{Batch, Client, ReplyView};
/// fn reuse(client: &mut Client, batch: Batch, replies: &[ReplyView<'_>]) {
///     let _ = client.begin_verification(batch, replies);
///     let _ = client.begin_verification(batch, replies);
/// }
/// ```
pub struct Batch {
    claims: [Claim; BATCH_SIZE],
    lease: Lease,
}

/// All original responses have been copied before this owner can exist.
///
/// ```compile_fail
/// use pllm_core::maverick_batch::ChallengedBatch;
/// fn duplicate(batch: ChallengedBatch) { let _ = batch.clone(); }
/// ```
///
/// ```compile_fail
/// use pllm_core::maverick_batch::{AuxiliaryReplyView, ChallengedBatch, Client};
/// fn reuse(client: &mut Client, batch: ChallengedBatch, reply: AuxiliaryReplyView<'_>) {
///     let _ = client.finish_batch(batch, reply);
///     let _ = client.finish_batch(batch, reply);
/// }
/// ```
pub struct ChallengedBatch {
    claims: [Claim; BATCH_SIZE],
    outputs: [Words; BATCH_SIZE],
    encoded: [Words; BATCH_SIZE],
    primary: [Sparse; REPETITIONS],
    header: Header,
    _lease: Lease,
}

pub struct AcceptedOutput {
    header: Header,
    values: Words,
}

impl AcceptedOutput {
    pub fn header(&self) -> Header {
        self.header
    }
    /// Exact field result only; this is not centered integer/FP8/BF16 decoding.
    pub fn values(&self) -> &[u32] {
        self.values.as_slice()
    }
}

pub struct AcceptedBatch {
    outputs: [AcceptedOutput; BATCH_SIZE],
}

impl AcceptedBatch {
    pub fn outputs(&self) -> &[AcceptedOutput; BATCH_SIZE] {
        &self.outputs
    }
}

/// Holds public client preprocessing and private issuance ownership only.
pub struct Client {
    pub(super) material: Material,
    session: [u8; 32],
    control: Arc<Control>,
}

/// Party-local server owner: only public matrices/Q, framing and replay state.
/// No P, Gx, private correction, input, client controller or secret RNG access.
pub struct Server {
    memory: Memory,
    matrices: [Words; BATCH_SIZE],
    q: [Words; BATCH_SIZE],
    binding: [u8; 32],
    session: [u8; 32],
    seen: [u8; MAX_ATTEMPTS as usize],
    evaluated: [u8; MAX_ATTEMPTS as usize],
    modes: [Option<Mode>; MAX_ATTEMPTS as usize],
}

impl ValidatedPublication {
    /// Copies public server-owned buffers; private state is never shared.
    /// The shared allocation meter contains only resource counters.
    pub fn split(self) -> Result<(Client, Server)> {
        let mut session = [0; 32];
        getrandom::fill(&mut session).map_err(|_| Error::Randomness)?;
        let m = self.material;
        let server = Server {
            memory: m.memory.clone(),
            matrices: [
                Words::copy(&m.memory, m.matrices[0].as_slice())?,
                Words::copy(&m.memory, m.matrices[1].as_slice())?,
            ],
            q: [
                Words::copy(&m.memory, m.q[0].as_slice())?,
                Words::copy(&m.memory, m.q[1].as_slice())?,
            ],
            binding: m.binding,
            session,
            seen: [0; MAX_ATTEMPTS as usize],
            evaluated: [0; MAX_ATTEMPTS as usize],
            modes: [None; MAX_ATTEMPTS as usize],
        };
        let client = Client {
            material: m,
            session,
            // Fixed metadata retains Arc::new's abort-on-OOM semantics. The
            // fallible/retired Error::Allocation paths apply to word buffers.
            control: Arc::new(Control {
                issued: AtomicU32::new(0),
                live: AtomicBool::new(false),
            }),
        };
        Ok((client, server))
    }
}

impl Client {
    pub fn memory_stats(&self) -> MemoryStats {
        self.material.memory.stats()
    }
    pub fn attempts_issued(&self) -> u32 {
        self.control.issued.load(Ordering::Acquire)
    }
    pub fn has_live_window(&self) -> bool {
        self.control.live.load(Ordering::Acquire)
    }

    fn reserve(&self, mode: Mode) -> Result<(Header, Lease)> {
        let attempt = self
            .control
            .issued
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_ATTEMPTS).then_some(n + 1)
            })
            .map_err(|_| Error::Attempts)?;
        self.control
            .live
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::Busy)?;
        Ok((
            Header {
                publication: self.material.binding,
                session: self.session,
                attempt,
                mode,
                slot: BATCH_SLOT,
            },
            Lease(Arc::clone(&self.control)),
        ))
    }

    fn check_header(&self, actual: Header, expected: Header) -> Result<()> {
        if actual != expected
            || expected.publication != self.material.binding
            || expected.session != self.session
        {
            return Err(Error::Binding);
        }
        Ok(())
    }

    fn claim(
        &self,
        mut header: Header,
        slot: usize,
        input: &[u32],
        noise: Sparse,
    ) -> Result<(Claim, Request)> {
        let m = &self.material;
        header.slot = slot as u8;
        let mask = m.gx[slot].syndrome(&m.memory, &noise)?;
        let mut masked = Words::zeros(&m.memory, DIMENSION)?;
        let mut correction = Words::zeros(&m.memory, DIMENSION)?;
        for i in 0..DIMENSION {
            masked.as_mut()[i] = add(input[i], mask.as_slice()[i]);
            correction.as_mut()[i] = noise.dot(&m.p[slot].as_slice()[i * LENGTH..][..LENGTH]);
        }
        let request = Request {
            header,
            values: Words::copy(&m.memory, masked.as_slice())?,
        };
        Ok((
            Claim {
                header,
                input: masked,
                correction,
            },
            request,
        ))
    }

    pub fn issue_query(&mut self, slot: usize, input: &[u32]) -> Result<(Query, Request)> {
        self.issue_query_with(slot, input, |memory| {
            Sparse::sample(memory, &mut OsRandom, LENGTH, PRIVACY_WEIGHT)
        })
    }

    pub(super) fn issue_query_with(
        &mut self,
        slot: usize,
        input: &[u32],
        noise: impl FnOnce(&Memory) -> Result<Sparse>,
    ) -> Result<(Query, Request)> {
        let (header, lease) = self.reserve(Mode::Single)?;
        if slot >= BATCH_SIZE {
            return Err(Error::Shape);
        }
        canonical(input, DIMENSION)?;
        let noise = noise(&self.material.memory)?;
        let (claim, request) = self.claim(header, slot, input, noise)?;
        Ok((
            Query {
                claim,
                _lease: lease,
            },
            request,
        ))
    }

    pub fn issue_batch(
        &mut self,
        inputs: [&[u32]; BATCH_SIZE],
    ) -> Result<(Batch, [Request; BATCH_SIZE])> {
        self.issue_batch_with(inputs, |memory| {
            Ok([
                Sparse::sample(memory, &mut OsRandom, LENGTH, PRIVACY_WEIGHT)?,
                Sparse::sample(memory, &mut OsRandom, LENGTH, PRIVACY_WEIGHT)?,
            ])
        })
    }

    pub(super) fn issue_batch_with(
        &mut self,
        inputs: [&[u32]; BATCH_SIZE],
        noise: impl FnOnce(&Memory) -> Result<[Sparse; BATCH_SIZE]>,
    ) -> Result<(Batch, [Request; BATCH_SIZE])> {
        let (header, lease) = self.reserve(Mode::Batch)?;
        for input in inputs {
            canonical(input, DIMENSION)?;
        }
        let [e0, e1] = noise(&self.material.memory)?;
        let (c0, r0) = self.claim(header, 0, inputs[0], e0)?;
        let (c1, r1) = self.claim(header, 1, inputs[1], e1)?;
        Ok((
            Batch {
                claims: [c0, c1],
                lease,
            },
            [r0, r1],
        ))
    }

    /// Consumes the query on EVERY path, including wrong binding/shape/field.
    /// Incoming response is copied before fresh secret challenges are sampled.
    pub fn finish_query(
        &mut self,
        query: Query,
        response: ReplyView<'_>,
    ) -> Result<AcceptedOutput> {
        self.finish_query_with(query, response, |memory| checks(memory, LENGTH))
            .map(|v| v.0)
    }

    pub(super) fn finish_query_with(
        &mut self,
        query: Query,
        response: ReplyView<'_>,
        challenges: impl FnOnce(&Memory) -> Result<[Sparse; REPETITIONS]>,
    ) -> Result<(AcceptedOutput, Zeroizing<[[u32; 2]; REPETITIONS]>)> {
        self.check_header(response.header, query.claim.header)?;
        canonical(response.values, DIMENSION)?;
        let m = &self.material;
        let fixed = Words::copy(&m.memory, response.values)?;
        let encoded = m.gy.encode(&m.memory, fixed.as_slice())?;
        let challenges = challenges(&m.memory)?;
        let slot = query.claim.header.slot as usize;
        let mut mismatch = 0;
        let mut trace = Zeroizing::new([[0; 2]; REPETITIONS]);
        for (i, e) in challenges.iter().enumerate() {
            let mut row = Words::zeros(&m.memory, DIMENSION)?;
            project(&m.q[slot], e, row.as_mut());
            let left = dot(row.as_slice(), query.claim.input.as_slice());
            let right = e.dot(encoded.as_slice());
            trace[i] = [left, right];
            mismatch |= left ^ right;
        }
        if mismatch != 0 {
            return Err(Error::Rejected);
        }
        let output = decrypt(&m.memory, &query.claim, fixed.as_slice())?;
        Ok((output, trace))
    }

    /// Fix ALL originals before sampling/revealing ANY primary challenge.
    /// Omission, duplication, wrong order or malformed replies retire the batch.
    pub fn begin_verification(
        &mut self,
        batch: Batch,
        responses: &[ReplyView<'_>],
    ) -> Result<(ChallengedBatch, PrimaryChallenge)> {
        self.begin_with(batch, responses, |memory| checks(memory, LENGTH))
    }

    pub(super) fn begin_with(
        &mut self,
        batch: Batch,
        responses: &[ReplyView<'_>],
        challenges: impl FnOnce(&Memory) -> Result<[Sparse; REPETITIONS]>,
    ) -> Result<(ChallengedBatch, PrimaryChallenge)> {
        if responses.len() != BATCH_SIZE {
            return Err(Error::Shape);
        }
        for (claim, response) in batch.claims.iter().zip(responses) {
            self.check_header(response.header, claim.header)?;
            canonical(response.values, DIMENSION)?;
        }
        let m = &self.material;
        let outputs = [
            Words::copy(&m.memory, responses[0].values)?,
            Words::copy(&m.memory, responses[1].values)?,
        ];
        let encoded = [
            m.gy.encode(&m.memory, outputs[0].as_slice())?,
            m.gy.encode(&m.memory, outputs[1].as_slice())?,
        ];
        // Only after both copies and code encodings are fixed.
        let primary = challenges(&m.memory)?;
        let mut header = batch.claims[0].header;
        header.slot = BATCH_SLOT;
        let disclosed = PrimaryChallenge {
            header,
            entries: [primary[0].copy(&m.memory)?, primary[1].copy(&m.memory)?],
        };
        Ok((
            ChallengedBatch {
                claims: batch.claims,
                outputs,
                encoded,
                primary,
                header,
                _lease: batch.lease,
            },
            disclosed,
        ))
    }

    /// Fix the complete auxiliary reply, sample independent SECRET secondary
    /// challenges, then check all four secondary and all four original equations.
    pub fn finish_batch(
        &mut self,
        batch: ChallengedBatch,
        auxiliary: AuxiliaryReplyView<'_>,
    ) -> Result<AcceptedBatch> {
        self.finish_batch_with(batch, auxiliary, |memory| checks(memory, SECONDARY_LENGTH))
            .map(|v| v.0)
    }

    pub(super) fn finish_batch_with(
        &mut self,
        batch: ChallengedBatch,
        auxiliary: AuxiliaryReplyView<'_>,
        challenges: impl FnOnce(&Memory) -> Result<[Sparse; REPETITIONS]>,
    ) -> Result<(AcceptedBatch, BatchTrace)> {
        self.check_header(auxiliary.header, batch.header)?;
        canonical(auxiliary.values, REPETITIONS * SECONDARY_DIMENSION)?;
        let m = &self.material;
        let fixed = Words::copy(&m.memory, auxiliary.values)?;
        let secondary = challenges(&m.memory)?;
        let mut mismatch = 0;
        let mut trace = BatchTrace {
            original: Zeroizing::new([[0; 2]; 4]),
            secondary: Zeroizing::new([[0; 2]; 4]),
        };
        // Each fresh e' is shared across the already-fixed auxiliary responses.
        for (j, ep) in secondary.iter().enumerate() {
            let r = m.gp.syndrome(&m.memory, ep)?;
            for (q, e) in batch.primary.iter().enumerate() {
                let mut left = 0;
                for (i, a) in e.entries() {
                    for (k, b) in ep.entries() {
                        left = add(
                            left,
                            mul(mul(a, b), m.aux.as_slice()[i * SECONDARY_LENGTH + k]),
                        );
                    }
                }
                let right = dot(
                    r.as_slice(),
                    &fixed.as_slice()[q * SECONDARY_DIMENSION..][..SECONDARY_DIMENSION],
                );
                trace.secondary[q * REPETITIONS + j] = [left, right];
                mismatch |= left ^ right;
            }
        }
        for (q, e) in batch.primary.iter().enumerate() {
            for b in 0..BATCH_SIZE {
                let start = q * SECONDARY_DIMENSION + b * DIMENSION;
                let left = dot(
                    &fixed.as_slice()[start..][..DIMENSION],
                    batch.claims[b].input.as_slice(),
                );
                let right = e.dot(batch.encoded[b].as_slice());
                trace.original[q * BATCH_SIZE + b] = [left, right];
                mismatch |= left ^ right;
            }
        }
        if mismatch != 0 {
            return Err(Error::Rejected);
        }
        Ok((
            AcceptedBatch {
                outputs: [
                    decrypt(&m.memory, &batch.claims[0], batch.outputs[0].as_slice())?,
                    decrypt(&m.memory, &batch.claims[1], batch.outputs[1].as_slice())?,
                ],
            },
            trace,
        ))
    }
}

pub(super) struct BatchTrace {
    pub(super) original: Zeroizing<[[u32; 2]; 4]>,
    pub(super) secondary: Zeroizing<[[u32; 2]; 4]>,
}

fn checks(memory: &Memory, length: usize) -> Result<[Sparse; REPETITIONS]> {
    Ok([
        Sparse::sample(memory, &mut OsRandom, length, CHECK_WEIGHT)?,
        Sparse::sample(memory, &mut OsRandom, length, CHECK_WEIGHT)?,
    ])
}

fn decrypt(memory: &Memory, claim: &Claim, fixed: &[u32]) -> Result<AcceptedOutput> {
    let mut values = Words::zeros(memory, DIMENSION)?;
    for (i, out) in values.as_mut().iter_mut().enumerate() {
        *out = sub(fixed[i], claim.correction.as_slice()[i]);
    }
    Ok(AcceptedOutput {
        header: claim.header,
        values,
    })
}

fn project(matrix: &Words, e: &Sparse, result: &mut [u32]) {
    debug_assert_eq!(result.len(), DIMENSION);
    for (position, value) in e.entries() {
        for (j, out) in result.iter_mut().enumerate() {
            *out = add(
                *out,
                mul(value, matrix.as_slice()[position * DIMENSION + j]),
            );
        }
    }
}

impl Server {
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory.stats()
    }

    fn check(&self, header: Header) -> Result<usize> {
        if header.publication != self.binding
            || header.session != self.session
            || header.attempt >= MAX_ATTEMPTS
        {
            return Err(Error::Binding);
        }
        Ok(header.attempt as usize)
    }

    pub fn evaluate(&mut self, request: RequestView<'_>) -> Result<Reply> {
        let at = self.check(request.header)?;
        let slot = request.header.slot as usize;
        if slot >= BATCH_SIZE {
            return Err(Error::Shape);
        }
        if self.modes[at].is_some_and(|mode| mode != request.header.mode)
            || self.seen[at] & (1 << slot) != 0
            || (request.header.mode == Mode::Single && self.seen[at] != 0)
        {
            return Err(Error::Replay);
        }
        // Burn this slot even on a malformed input or resource failure.
        self.modes[at] = Some(request.header.mode);
        self.seen[at] |= 1 << slot;
        canonical(request.values, DIMENSION)?;
        let mut values = Words::zeros(&self.memory, DIMENSION)?;
        for row in 0..DIMENSION {
            values.as_mut()[row] = dot(
                &self.matrices[slot].as_slice()[row * DIMENSION..][..DIMENSION],
                request.values,
            );
        }
        self.evaluated[at] |= 1 << slot;
        Ok(Reply {
            header: request.header,
            values,
        })
    }

    pub fn project(&mut self, challenge: PrimaryChallengeView<'_>) -> Result<AuxiliaryReply> {
        let at = self.check(challenge.header)?;
        if challenge.header.mode != Mode::Batch
            || challenge.header.slot != BATCH_SLOT
            || self.modes[at] != Some(Mode::Batch)
            || self.seen[at] != 3
            || self.evaluated[at] != 3
        {
            return Err(Error::Replay);
        }
        self.seen[at] |= 4;
        let mut values = Words::zeros(&self.memory, REPETITIONS * SECONDARY_DIMENSION)?;
        for (q, pairs) in challenge.entries.iter().enumerate() {
            let sparse = Sparse::from_pairs(&self.memory, LENGTH, CHECK_WEIGHT, pairs)?;
            for b in 0..BATCH_SIZE {
                let start = q * SECONDARY_DIMENSION + b * DIMENSION;
                project(
                    &self.q[b],
                    &sparse,
                    &mut values.as_mut()[start..][..DIMENSION],
                );
            }
        }
        Ok(AuxiliaryReply {
            header: challenge.header,
            values,
        })
    }
}
