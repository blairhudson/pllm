//! Bounded source-derived Maverick Protocols 3 and 2 over BabyBear.
//!
//! Algebra conformance only: no code-distance or dual-LPN certification, model
//! quantizer, decoder, transport, or SDK admission. This module is independent of
//! the historical seeded/i8 RAA reference and does not alter its identity.

mod arithmetic;
mod code;
mod memory;
mod protocol;
mod publication;
pub mod window_plan;

pub use code::CodeDescription;
pub use memory::{AllocationPlan, MemoryStats, RequestAllocationPlan, ALLOCATION_CEILING};
pub use protocol::{
    AcceptedBatch, AcceptedOutput, AcceptedRequest, AuxiliaryReply, AuxiliaryReplyView, Batch,
    ChallengedBatch, ChallengedRequest, Client, DependentBatch, DependentRequest, Header, Mode,
    PrimaryChallenge, PrimaryChallengeView, Protocol2Request, Query, Reply, ReplyView, Request,
    RequestView, Server,
};
pub use publication::{
    preprocess, PublicationParts, PublicationView, SetupInputs, UnvalidatedPublication,
    ValidatedPublication,
};

pub const FIELD: u32 = 2_013_265_921;
pub const DIMENSION: usize = 64;
pub const BATCH_SIZE: usize = 2;
pub const REPEAT: usize = 8;
pub const LENGTH: usize = REPEAT * DIMENSION;
pub const SECONDARY_DIMENSION: usize = BATCH_SIZE * DIMENSION;
pub const SECONDARY_LENGTH: usize = REPEAT * SECONDARY_DIMENSION;
pub const PRIVACY_WEIGHT: usize = 165;
pub const CHECK_WEIGHT: usize = 31;
pub const REPETITIONS: usize = 2;
pub const MAX_ATTEMPTS: u32 = 16;

/// Errors contain no private input, mask, challenge, or arithmetic diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Shape,
    Field,
    Code,
    Publication,
    Binding,
    Replay,
    Busy,
    Attempts,
    Randomness,
    Allocation,
    Rejected,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Maverick bounded capsule: {self:?}")
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
pub mod testing;
