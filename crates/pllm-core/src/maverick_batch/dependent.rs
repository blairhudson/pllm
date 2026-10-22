//! Fixed two-claim field dependency, using the ordinary Protocol-2 verifier.
//! No numerical decoder, arbitrary local callback, transport or security claim.

use super::super::arithmetic::{canonical, OsRandom};
use super::super::code::Sparse;
use super::super::memory::{Memory, Words};
use super::super::{Result, DIMENSION, LENGTH, PRIVACY_WEIGHT, REPETITIONS};
use super::{
    checks, demask_values, Batch, ChallengedBatch, Claim, Client, Lease, Mode, PrimaryChallenge,
    ReplyView, Request,
};

/// Consuming first stage of `x -> M0*x -> M1*(M0*x)` over BabyBear.
/// Only masked requests escape. Drop cancels; all errors consume this owner.
///
/// ```compile_fail
/// use pllm_core::maverick_batch::DependentRequest;
/// fn clone_owner(owner: DependentRequest) { let _ = owner.clone(); }
/// ```
///
/// ```compile_fail
/// use pllm_core::maverick_batch::{Client, DependentRequest, ReplyView};
/// fn reuse(c: &mut Client, owner: DependentRequest, reply: ReplyView<'_>) {
///     let _ = c.continue_dependent(owner, reply);
///     let _ = c.continue_dependent(owner, reply);
/// }
/// ```
///
/// ```compile_fail
/// use pllm_core::maverick_batch::DependentRequest;
/// fn escape(owner: &DependentRequest) -> &[u32] { owner.values() }
/// ```
pub struct DependentRequest {
    claim: Claim,
    lease: Lease,
}

/// First original is immutable; the second claim uses its private field demask.
/// There is no tentative output, callback, or arbitrary private transform API.
/// Acceptance uses `Client::finish_batch` only after BOTH originals and the
/// complete auxiliary reply have crossed the existing fixation barriers.
///
/// ```compile_fail
/// use pllm_core::maverick_batch::DependentBatch;
/// fn clone_owner(owner: DependentBatch) { let _ = owner.clone(); }
/// ```
///
/// ```compile_fail
/// use pllm_core::maverick_batch::{Client, DependentBatch, ReplyView};
/// fn reuse(c: &mut Client, owner: DependentBatch, reply: ReplyView<'_>) {
///     let _ = c.begin_dependent_verification(owner, reply);
///     let _ = c.begin_dependent_verification(owner, reply);
/// }
/// ```
///
/// ```compile_fail
/// use pllm_core::maverick_batch::DependentBatch;
/// fn escape(owner: &DependentBatch) -> &[u32] { owner.outputs()[0].values() }
/// ```
///
/// ```compile_fail
/// use pllm_core::maverick_batch::{Client, Batch, DependentBatch};
/// fn bypass(_: &mut Client, owner: DependentBatch) -> Batch { owner.into() }
/// ```
///
/// ```compile_fail
/// use pllm_core::maverick_batch::DependentBatch;
/// fn private_escape(owner: &DependentBatch) -> &[u32] { owner.first_output.as_slice() }
/// ```
pub struct DependentBatch {
    // Drop all retained originals before Batch drops its claims and live lease.
    first_output: Words,
    batch: Batch,
}

impl Client {
    /// Start the fixed ordered field chain. Same live-window and attempt ledger
    /// as coupled batches; this selects no FP8/BF16 or decoder numeric contract.
    pub fn issue_dependent(&mut self, input: &[u32]) -> Result<(DependentRequest, Request)> {
        self.issue_dependent_with(input, |memory| {
            Sparse::sample(memory, &mut OsRandom, LENGTH, PRIVACY_WEIGHT)
        })
    }

    pub(in super::super) fn issue_dependent_with(
        &mut self,
        input: &[u32],
        noise: impl FnOnce(&Memory) -> Result<Sparse>,
    ) -> Result<(DependentRequest, Request)> {
        let (header, lease) = self.reserve(Mode::Batch)?;
        canonical(input, DIMENSION)?;
        let (claim, request) = self.claim(header, 0, input, noise(&self.material.memory)?)?;
        Ok((DependentRequest { claim, lease }, request))
    }

    /// Fix the first reply BEFORE privately deriving and masking the second
    /// input. No verification challenges are sampled or disclosed at this stage.
    pub fn continue_dependent(
        &mut self,
        owner: DependentRequest,
        response: ReplyView<'_>,
    ) -> Result<(DependentBatch, Request)> {
        self.continue_dependent_with(owner, response, |memory| {
            Sparse::sample(memory, &mut OsRandom, LENGTH, PRIVACY_WEIGHT)
        })
    }

    pub(in super::super) fn continue_dependent_with(
        &mut self,
        owner: DependentRequest,
        response: ReplyView<'_>,
        noise: impl FnOnce(&Memory) -> Result<Sparse>,
    ) -> Result<(DependentBatch, Request)> {
        self.check_header(response.header, owner.claim.header)?;
        canonical(response.values, DIMENSION)?;
        let memory = &self.material.memory;
        let first_output = Words::copy(memory, response.values)?;
        #[cfg(test)]
        super::super::testing::record(super::super::testing::FixationEvent::FirstOriginalFixed);
        let next_input = demask_values(memory, &owner.claim, first_output.as_slice())?;
        let (second, request) =
            self.claim(owner.claim.header, 1, next_input.as_slice(), noise(memory)?)?;
        // next_input is a zeroizing, charged private owner, never AcceptedOutput.
        Ok((
            DependentBatch {
                batch: Batch {
                    claims: [owner.claim, second],
                    lease: owner.lease,
                },
                first_output,
            },
            request,
        ))
    }

    /// Fix the second original, then enter the SAME primary/secondary verifier
    /// as coupled Protocol 2. The first original cannot be replaced by caller.
    pub fn begin_dependent_verification(
        &mut self,
        owner: DependentBatch,
        response: ReplyView<'_>,
    ) -> Result<(ChallengedBatch, PrimaryChallenge)> {
        self.begin_dependent_with(owner, response, |memory| checks(memory, LENGTH))
    }

    pub(in super::super) fn begin_dependent_with(
        &mut self,
        owner: DependentBatch,
        response: ReplyView<'_>,
        challenges: impl FnOnce(&Memory) -> Result<[Sparse; REPETITIONS]>,
    ) -> Result<(ChallengedBatch, PrimaryChallenge)> {
        self.check_header(owner.batch.claims[0].header, owner.batch.claims[0].header)?;
        self.check_header(response.header, owner.batch.claims[1].header)?;
        canonical(response.values, DIMENSION)?;
        let second = Words::copy(&self.material.memory, response.values)?;
        self.begin_fixed_outputs(owner.batch, [owner.first_output, second], challenges)
    }
}
