//! Exactly two ordered raw-field Protocol-2 batches, one exclusive lease.
//! No dependency transform, tentative output, numeric decoder or SDK authority.

use super::super::memory::{RequestAllocationPlan, Reservation};
use super::*;

/// Four private claims. Drop cancels both spent attempts. Not Clone/Debug/serde.
///
/// ```compile_fail
/// use pllm_core::maverick_batch::Protocol2Request;
/// fn duplicate(r: Protocol2Request) { let _ = r.clone(); }
/// ```
///
/// ```compile_fail
/// use pllm_core::maverick_batch::{Client, Protocol2Request, ReplyView};
/// fn replay(c: &mut Client, r: Protocol2Request, replies: &[ReplyView<'_>]) {
///     let _ = c.begin_request_verification(r, replies);
///     let _ = c.begin_request_verification(r, replies);
/// }
/// ```
pub struct Protocol2Request {
    claims: [[Claim; BATCH_SIZE]; 2],
    _controls: Reservation,
    lease: Lease,
}

/// All four original replies and encodings precede every primary challenge.
///
/// ```compile_fail
/// use pllm_core::maverick_batch::ChallengedRequest;
/// fn tentative(r: &ChallengedRequest) -> &[u32] { r.outputs()[0].values() }
/// ```
pub struct ChallengedRequest {
    batches: [ChallengedBatch; 2],
    _controls: Reservation,
    _lease: Lease,
}

/// All four demasks appear atomically, only after both batches pass.
pub struct AcceptedRequest {
    batches: [AcceptedBatch; 2],
}

impl AcceptedRequest {
    pub fn batches(&self) -> &[AcceptedBatch; 2] {
        &self.batches
    }
}

impl Client {
    /// Observation against the actual validated material's shared allocation
    /// meter, not a reservation or caller-declared trusted-material credential.
    /// Issuance rechecks; every later buffer still reserves on the same meter.
    pub fn request_allocation_plan(&self) -> Result<RequestAllocationPlan> {
        RequestAllocationPlan::quote(&self.material.memory)
    }

    pub fn issue_request(
        &mut self,
        inputs: [[&[u32]; BATCH_SIZE]; 2],
    ) -> Result<(Protocol2Request, [[Request; BATCH_SIZE]; 2])> {
        self.issue_request_with(inputs, |memory| {
            Sparse::sample(memory, &mut OsRandom, LENGTH, PRIVACY_WEIGHT)
        })
    }

    fn issue_request_with(
        &mut self,
        inputs: [[&[u32]; BATCH_SIZE]; 2],
        mut noise: impl FnMut(&Memory) -> Result<Sparse>,
    ) -> Result<(Protocol2Request, [[Request; BATCH_SIZE]; 2])> {
        // The same counter burns TWO ordinals even on later malformed input,
        // failed preflight, entropy/allocation failure, cancellation or Busy.
        let (first, lease) = self.reserve_ordinals(Mode::Batch, 2)?;
        let mut second = first;
        second.attempt = second.attempt.checked_add(1).ok_or(Error::Attempts)?;
        for batch in inputs {
            for input in batch {
                canonical(input, DIMENSION)?;
            }
        }
        self.request_allocation_plan()?; // no allocation or entropy before quote
        let controls = self.material.memory.reserve(4096)?;
        let memory = &self.material.memory;
        let (a, ra) = self.claim(first, 0, inputs[0][0], noise(memory)?)?;
        let (b, rb) = self.claim(first, 1, inputs[0][1], noise(memory)?)?;
        let (c, rc) = self.claim(second, 0, inputs[1][0], noise(memory)?)?;
        let (d, rd) = self.claim(second, 1, inputs[1][1], noise(memory)?)?;
        Ok((
            Protocol2Request {
                claims: [[a, b], [c, d]],
                _controls: controls,
                lease,
            },
            [[ra, rb], [rc, rd]],
        ))
    }

    pub fn begin_request_verification(
        &mut self,
        request: Protocol2Request,
        responses: &[ReplyView<'_>],
    ) -> Result<(ChallengedRequest, [PrimaryChallenge; 2])> {
        self.begin_request_with(request, responses, |memory| checks(memory, LENGTH))
    }

    fn begin_request_with(
        &mut self,
        request: Protocol2Request,
        responses: &[ReplyView<'_>],
        mut challenges: impl FnMut(&Memory) -> Result<[Sparse; REPETITIONS]>,
    ) -> Result<(ChallengedRequest, [PrimaryChallenge; 2])> {
        // Declare the lease FIRST so every moved private local dies before it
        // on errors/unwind. Owners also declare their lease last for Drop.
        let lease = request.lease;
        let controls = request._controls;
        let [a, b] = request.claims;
        if responses.len() != 4 {
            return Err(Error::Shape);
        }
        // Validate ALL framing/canonical values before copying any original.
        for (claim, reply) in a.iter().chain(b.iter()).zip(responses) {
            self.check_header(reply.header, claim.header)?;
            canonical(reply.values, DIMENSION)?;
        }
        let oa = self.fix_originals(&a, &responses[..2])?;
        let ob = self.fix_originals(&b, &responses[2..])?;
        let ea = self.encode_originals(&oa)?;
        let eb = self.encode_originals(&ob)?;
        let (a, pa) = self.prepare_encoded_batch(a, oa, ea, &mut challenges)?;
        let (b, pb) = self.prepare_encoded_batch(b, ob, eb, &mut challenges)?;
        Ok((
            ChallengedRequest {
                batches: [a, b],
                _controls: controls,
                _lease: lease,
            },
            [pa, pb],
        ))
    }

    pub fn finish_request(
        &mut self,
        request: ChallengedRequest,
        auxiliary: &[AuxiliaryReplyView<'_>],
    ) -> Result<AcceptedRequest> {
        self.finish_request_with(request, auxiliary, |memory| {
            checks(memory, SECONDARY_LENGTH)
        })
    }

    fn finish_request_with(
        &mut self,
        request: ChallengedRequest,
        auxiliary: &[AuxiliaryReplyView<'_>],
        mut challenges: impl FnMut(&Memory) -> Result<[Sparse; REPETITIONS]>,
    ) -> Result<AcceptedRequest> {
        if auxiliary.len() != 2 {
            return Err(Error::Shape);
        }
        // Header/canonical validation of BOTH precedes every secondary draw.
        for (batch, reply) in request.batches.iter().zip(auxiliary) {
            self.check_header(reply.header, batch.header)?;
            canonical(reply.values, REPETITIONS * SECONDARY_DIMENSION)?;
        }
        let a = self.fix_auxiliary(&request.batches[0], auxiliary[0])?;
        let b = self.fix_auxiliary(&request.batches[1], auxiliary[1])?;
        let sa = challenges(&self.material.memory)?;
        let sb = challenges(&self.material.memory)?;
        let (ma, _ta) = self.check_equations(&request.batches[0], &a, &sa)?;
        let (mb, _tb) = self.check_equations(&request.batches[1], &b, &sb)?;
        // Arithmetic failures accumulate privately; never return first success
        // or short-circuit verification of a well-formed second batch.
        if ma | mb != 0 {
            return Err(Error::Rejected);
        }
        let outputs = [
            self.accept_batch(&request.batches[0])?,
            self.accept_batch(&request.batches[1])?,
        ];
        Ok(AcceptedRequest { batches: outputs })
    }
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
