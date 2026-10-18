//! Party-local Q18 multiplication followed by source-pinned TruncAPr.
//!
//! Public intervals must bound the intended, unwrapped operands. This owner
//! checks their complete product hull before randomness or messages; it cannot
//! prove that secret shares satisfy a caller's interval declaration. A compiler
//! must derive those declarations independently of the private request.
//!
//! Each role holds only its adjacent replicated pairs. Fresh pairwise zero
//! shares mask the local product before one rotation, followed by the existing
//! source PRSS/truncation owner. Setup, frames and outputs are non-cloneable.
//! Typed messages require authenticated private transport at the outer boundary.

use super::{blocks, expand, Cursors, PrssLimits, PublicContribution, SeedMessage, SourceParty};
use crate::puma::{
    Context, Error, Header, Limits, Message, MessagePhase, Operation, Party, PayloadUsage,
    ReplicatedShare, Status as TruncationStatus, Words, FIXED_RESERVE_BYTES, INPUT_MAX_EXCLUSIVE,
    INPUT_MIN,
};
use sha2::{Digest, Sha256};

pub const PROFILE: &str = "puma.spu-a42c261.mul-aa-truncapr18.ring64.quarter-gap.replicated3.v1";
pub const NUMERIC_PROFILE: &str = "puma.source-derived-q18.v1";
pub const FRACTION_BITS: u8 = 18;

/// Public admission ticket, before any PRSS draw. This has no chosen truncation
/// roles. The input/rotation binding is distinct from the later TruncAPr binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    input: Operation,
    zero_start: u64,
}

impl Plan {
    pub fn digest(self) -> [u8; 32] {
        self.input.digest()
    }
    pub fn sequence(self) -> u64 {
        self.input.sequence()
    }
    pub fn lanes(self) -> usize {
        self.input.lanes()
    }

    /// Binding only for the two input pairs and multiplication rotation. Its
    /// placeholder pivot is not an evaluator assignment; use the runtime's
    /// `truncation_operation` after rotation for actual source-derived roles.
    pub fn input_binding(self) -> Operation {
        self.input
    }

    pub fn share(
        self,
        party: Party,
        first: &[u64],
        second: &[u64],
        limits: Limits,
    ) -> Result<ReplicatedShare, Error> {
        ReplicatedShare::from_local(self.input, party, first, second, limits)
    }
}

/// Inclusive public bounds on an unwrapped signed integer, not its ring residue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    lower: i64,
    upper: i64,
}

impl Interval {
    pub fn new(lower: i64, upper: i64) -> Result<Self, Error> {
        if lower > upper {
            return Err(Error::Bounds);
        }
        Ok(Self { lower, upper })
    }

    pub fn lower(self) -> i64 {
        self.lower
    }

    pub fn upper(self) -> i64 {
        self.upper
    }
}

/// One lane's public operand obligations. Fields carry no private observations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputBounds {
    pub left: Interval,
    pub right: Interval,
}

impl InputBounds {
    /// Check all four corners in i128 before any ring reduction.
    pub fn product(self) -> Result<Interval, Error> {
        let mut lower = i128::MAX;
        let mut upper = i128::MIN;
        for x in [self.left.lower, self.left.upper] {
            for y in [self.right.lower, self.right.upper] {
                let product = i128::from(x)
                    .checked_mul(i128::from(y))
                    .ok_or(Error::Bounds)?;
                lower = lower.min(product);
                upper = upper.max(product);
            }
        }
        if lower < i128::from(INPUT_MIN) || upper >= i128::from(INPUT_MAX_EXCLUSIVE) {
            return Err(Error::Bounds);
        }
        Interval::new(
            i64::try_from(lower).map_err(|_| Error::Bounds)?,
            i64::try_from(upper).map_err(|_| Error::Bounds)?,
        )
    }

    /// Conservative hull of stochastic Q18 outputs, inclusive of every mask.
    pub fn output(self) -> Result<Interval, Error> {
        let product = self.product()?;
        let scale = 1_i64 << FRACTION_BITS;
        Interval::new(
            product.lower.div_euclid(scale),
            -(-product.upper).div_euclid(scale),
        )
    }
}

/// The common PRSS setup with a distinct compound execution namespace.
///
/// ```compile_fail
/// use pllm_core::puma::multiply::SetupParty;
/// fn duplicate(owner: SetupParty) { let _ = owner.clone(); }
/// ```
pub struct SetupParty(super::SetupParty);

impl SetupParty {
    pub fn new(context: Context, party: Party, limits: PrssLimits) -> Result<Self, Error> {
        Ok(Self(super::SetupParty::new(
            bound_context(context),
            party,
            limits,
        )?))
    }

    pub fn status(&self) -> super::SetupStatus {
        self.0.status()
    }

    pub fn contribution(&mut self) -> Result<PublicContribution, Error> {
        self.0.contribution()
    }

    pub fn receive_public(&mut self, message: PublicContribution) -> Result<(), Error> {
        self.0.receive_public(message)
    }

    pub fn share_seed(&mut self) -> Result<SeedMessage, Error> {
        self.0.share_seed()
    }

    pub fn receive_seed(&mut self, message: SeedMessage) -> Result<(), Error> {
        self.0.receive_seed(message)
    }

    pub fn finish(&mut self) -> Result<PartyRuntime, Error> {
        Ok(PartyRuntime {
            source: self.0.finish()?,
            phase: Phase::Ready,
        })
    }

    pub fn cancel(&mut self) {
        self.0.cancel();
    }
}

fn bound_context(mut context: Context) -> Context {
    let mut hash = Sha256::new();
    hash.update(b"pllm.puma.compound-setup.v1");
    hash.update(PROFILE.as_bytes());
    hash.update(NUMERIC_PROFILE.as_bytes());
    hash.update(context.schedule);
    context.schedule = hash.finalize().into();
    context
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ready,
    Planned,
    AwaitingRotate,
    AwaitingOpen,
    AwaitingReshare,
    Complete,
    Retired,
}

enum Phase {
    Ready,
    Planned(Plan),
    Rotating { plan: Plan, product: Words },
    Truncating(Operation),
    Retired,
}

/// A sequential compound owner: multiplication and truncation share one ledger,
/// one source cursor schedule and one failure domain. No intermediate result or
/// reusable correlation is exported. Returned original frames/shares keep their
/// byte reservations through erasure and deallocation, including after cancel.
///
/// ```compile_fail
/// use pllm_core::puma::multiply::PartyRuntime;
/// fn duplicate(owner: PartyRuntime) { let _ = owner.clone(); }
/// ```
pub struct PartyRuntime {
    source: SourceParty,
    phase: Phase,
}

impl PartyRuntime {
    pub fn context(&self) -> Context {
        self.source.context()
    }

    pub fn party(&self) -> Party {
        self.source.party()
    }

    pub fn cursors(&self) -> Cursors {
        self.source.cursors()
    }

    pub fn payload_usage(&self) -> PayloadUsage {
        self.source.payload_usage()
    }

    pub fn status(&self) -> Status {
        match self.phase {
            Phase::Ready => Status::Ready,
            Phase::Planned(_) => Status::Planned,
            Phase::Rotating { .. } => Status::AwaitingRotate,
            Phase::Truncating(_) => match self.source.status() {
                TruncationStatus::AwaitingOpen => Status::AwaitingOpen,
                TruncationStatus::AwaitingReshare => Status::AwaitingReshare,
                TruncationStatus::Complete => Status::Complete,
                _ => Status::Retired,
            },
            Phase::Retired => Status::Retired,
        }
    }

    /// Available only after the multiplication rotation has been accepted and
    /// the next public source draw has selected the truncation roles.
    pub fn truncation_operation(&self) -> Option<Operation> {
        match self.phase {
            Phase::Truncating(op) => Some(op),
            _ => None,
        }
    }

    pub fn cancel(&mut self) {
        self.phase = Phase::Retired;
        self.source.cancel();
    }

    fn checked<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.cancel();
        }
        result
    }

    /// Admit every lane and the complete operation before consuming PRSS.
    /// Worst-case remaining helper work is priced before the public pivot draw;
    /// actual AES accounting still charges only this role's source expansions.
    pub fn plan(&mut self, bounds: &[InputBounds]) -> Result<Plan, Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            let n = bounds.len();
            let working = self.source.limits.core.admit(n)? - FIXED_RESERVE_BYTES;
            let usage = self.payload_usage();
            if usage
                .live_bytes()
                .checked_add(working)
                .ok_or(Error::Bounds)?
                > usage.limit_bytes()
            {
                return Err(Error::Bounds);
            }
            let mut hash = Sha256::new();
            hash.update(PROFILE.as_bytes());
            hash.update(NUMERIC_PROFILE.as_bytes());
            for bound in bounds {
                bound.product()?;
                for interval in [bound.left, bound.right] {
                    hash.update(interval.lower.to_le_bytes());
                    hash.update(interval.upper.to_le_bytes());
                }
            }
            let q = blocks(n)?;
            self.source.work(6 * q + n as u64 + 1)?;
            if self.source.runtime.next_sequence() >= self.source.limits.core.max_operations {
                return Err(Error::Bounds);
            }
            self.source
                .cursors
                .public_blocks
                .checked_add(1)
                .ok_or(Error::Bounds)?;
            self.source
                .cursors
                .private_blocks
                .checked_add(n as u64)
                .ok_or(Error::Bounds)?;
            let zero_start = self.source.cursors.pair_blocks;
            let mask_start = zero_start.checked_add(q).ok_or(Error::Bounds)?;
            mask_start.checked_add(2 * q).ok_or(Error::Bounds)?;
            hash.update(b"pairwise-product-zero");
            hash.update(zero_start.to_le_bytes());
            let mut input = Operation::new(
                self.context(),
                self.source.runtime.next_sequence(),
                Party::P0,
                n,
                FRACTION_BITS,
                self.source.limits.core,
            )?;
            input.source = Some(crate::puma::SourceBinding {
                setup: self.source.setup,
                public_start: self.source.cursors.public_blocks,
                mask_start,
                output_start: mask_start + q,
            });
            input.consumer = Some(hash.finalize().into());
            let plan = Plan { input, zero_start };
            self.phase = Phase::Planned(plan);
            Ok(plan)
        })();
        self.checked(result)
    }

    /// Consume this party's two adjacent operand pairs, returning one fresh
    /// masked product component to PrevRank. No clear product is reconstructed.
    pub fn begin(
        &mut self,
        left: ReplicatedShare,
        right: ReplicatedShare,
    ) -> Result<Message, Error> {
        let result = self.begin_inner(left, right);
        self.checked(result)
    }

    fn begin_inner(
        &mut self,
        mut left: ReplicatedShare,
        mut right: ReplicatedShare,
    ) -> Result<Message, Error> {
        self.source.live()?;
        let Phase::Planned(plan) = self.phase else {
            return Err(Error::Phase);
        };
        let operation = plan.input;
        let zero_start = plan.zero_start;
        if left.operation != operation || right.operation != operation {
            return Err(Error::Binding);
        }
        let party = self.party();
        if left.party != party || right.party != party {
            return Err(Error::Role);
        }
        let usage = self.payload_usage();
        left.charge_to(&usage)?;
        right.charge_to(&usage)?;
        let n = operation.lanes();
        // Source order: Both fill, rotation, THEN public pivot and truncation.
        // Consume before allocation; cancellation never rewinds either stream.
        let q = blocks(n)?;
        self.source.cursors.pair_blocks = zero_start.checked_add(q).ok_or(Error::Bounds)?;
        self.source.cursors.aes_work = self.source.work(2 * q)?;
        let seeds = self.source.seeds.as_ref().ok_or(Error::Retired)?;
        let mut product = expand(&seeds.own, zero_start, n, Some(&usage))?;
        let next = expand(
            seeds.next.as_ref().ok_or(Error::Retired)?,
            zero_start,
            n,
            Some(&usage),
        )?;
        for i in 0..n {
            product[i] = left.first[i]
                .wrapping_mul(right.first[i])
                .wrapping_add(left.first[i].wrapping_mul(right.second[i]))
                .wrapping_add(left.second[i].wrapping_mul(right.first[i]))
                .wrapping_add(product[i])
                .wrapping_sub(next[i]);
        }
        drop(next);
        let words = usage.copy(&product, n)?;
        self.phase = Phase::Rotating { plan, product };
        Ok(Message {
            header: Header {
                binding: operation.digest(),
                from: party,
                to: party.next().next(),
                phase: MessagePhase::Multiply,
            },
            words,
        })
    }

    /// Accept NextRank's product component, then start truncation at the next
    /// pairwise cursor. The multiplication buffer moves into the original owner.
    pub fn rotate(&mut self, mut next: Message) -> Result<[Option<Message>; 2], Error> {
        let result = (|| {
            self.source.live()?;
            let Phase::Rotating { plan, product } =
                std::mem::replace(&mut self.phase, Phase::Retired)
            else {
                return Err(Error::Phase);
            };
            next.validate(
                plan.input,
                self.party().next(),
                self.party(),
                MessagePhase::Multiply,
            )?;
            next.words.charge_to(&self.payload_usage())?;
            let mut operation = self.source.plan(plan.lanes(), FRACTION_BITS)?;
            operation.consumer = Some(plan.digest());
            self.source.pending = Some(operation);
            let input = ReplicatedShare {
                operation,
                party: self.party(),
                first: product,
                second: next.words,
            };
            let messages = self.source.begin(input)?;
            self.phase = Phase::Truncating(operation);
            Ok(messages)
        })();
        self.checked(result)
    }

    fn truncating(&self) -> Result<(), Error> {
        self.source.live()?;
        if !matches!(self.phase, Phase::Truncating(_)) {
            return Err(Error::Phase);
        }
        Ok(())
    }

    pub fn open(&mut self, peer: Message, correlation: Message) -> Result<Message, Error> {
        let result = self
            .truncating()
            .and_then(|()| self.source.open(peer, correlation));
        self.checked(result)
    }

    pub fn reshare(&mut self, peer: Message) -> Result<(), Error> {
        let result = self.truncating().and_then(|()| self.source.reshare(peer));
        self.checked(result)
    }

    pub fn take_output(&mut self) -> Result<ReplicatedShare, Error> {
        let result = (|| {
            self.truncating()?;
            let output = self.source.take_output()?;
            self.phase = Phase::Ready;
            Ok(output)
        })();
        self.checked(result)
    }
}

impl Drop for PartyRuntime {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
#[path = "multiply_tests.rs"]
mod tests;
