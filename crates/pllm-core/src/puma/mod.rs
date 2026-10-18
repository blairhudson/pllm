//! Bounded PUMA candidate replicated-three-party stochastic truncation.
//!
//! Each [`PartyRuntime`] owns one party's replicated pair and authorized local
//! preprocessing, never all three input shares. The helper knows the random
//! mask, as in the source, but never receives the masked opening. [`prss`]
//! supplies the pinned pairwise setup and call-reset AES stream without a dealer.
//!
//! This is the pinned SPU `TruncAPr` candidate, not a completed PUMA decoder or
//! a complete implementation of the approved named source-derived Q18 profile.
//! Implementation/range/transport/SDK/quality gates remain open. See
//! `docs/research/puma-truncation-reference.md` for ownership and setup gates.

pub mod oracle;
mod payload;
mod protocol;
pub mod prss;

pub use payload::PayloadUsage;
use payload::Words;
pub use protocol::PartyRuntime;
pub use prss::multiply;

use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub const PROFILE: &str = "puma.spu-a42c261.truncapr.ring64.quarter-gap.replicated3.v1";
pub const MAX_LANES: usize = 1024;
pub const MAX_OPERATIONS: u64 = 4096;
pub const MAX_ADMITTED_BYTES: usize = 256 << 10;
pub const INPUT_MIN: i64 = -(1_i64 << 62);
pub const INPUT_MAX_EXCLUSIVE: i64 = 1_i64 << 62;
const FIXED_RESERVE_BYTES: usize = 4096;
const WORDS_PER_LANE_RESERVE: usize = 16;

/// Coarse local errors; never include secret values or failure positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Bounds,
    Allocation,
    Binding,
    Role,
    Phase,
    Shape,
    Entropy,
    Retired,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PUMA truncation rejected: {self:?}")
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Party {
    P0 = 0,
    P1 = 1,
    P2 = 2,
}

impl Party {
    pub fn next(self) -> Self {
        match self {
            Self::P0 => Self::P1,
            Self::P1 => Self::P2,
            Self::P2 => Self::P0,
        }
    }
}

/// Shape-only admission keeps the existing conservative 16-word/lane estimate.
/// Actual live payloads, including original returned frames and shares, are
/// additionally capped by a per-runtime ledger until wiped AND deallocated.
/// Caller-created copies/containers, allocator overhead and other runtimes are
/// separate. The fixed 4096-byte reserve includes ledger/owner metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    max_lanes: usize,
    max_operations: u64,
    max_bytes: usize,
}

impl Limits {
    pub fn new(max_lanes: usize, max_operations: u64, max_bytes: usize) -> Result<Self, Error> {
        if max_lanes == 0
            || max_lanes > MAX_LANES
            || max_operations == 0
            || max_operations > MAX_OPERATIONS
            || max_bytes > MAX_ADMITTED_BYTES
        {
            return Err(Error::Bounds);
        }
        let result = Self {
            max_lanes,
            max_operations,
            max_bytes,
        };
        result.admit(max_lanes)?;
        Ok(result)
    }

    pub fn admitted_bytes(self, lanes: usize) -> Result<usize, Error> {
        self.admit(lanes)
    }

    /// Working-set estimate with this operation's original outgoing frames
    /// retained: H uses at most 10 words/lane; A/B at most 8. This does not replace
    /// lifetime accounting for buffers retained from earlier operations.
    /// Compound consumers use the full 16-word/lane bound for their joint owners.
    pub fn role_working_bytes(self, operation: Operation, party: Party) -> Result<usize, Error> {
        if operation.consumer.is_some() {
            return self.admit(operation.lanes);
        }
        self.admit(operation.lanes)?;
        let words = if party == operation.helper() { 10 } else { 8 };
        Ok(FIXED_RESERVE_BYTES + operation.lanes * words * 8)
    }

    fn admit(self, lanes: usize) -> Result<usize, Error> {
        let bytes = lanes
            .checked_mul(WORDS_PER_LANE_RESERVE)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(FIXED_RESERVE_BYTES))
            .ok_or(Error::Bounds)?;
        if lanes == 0 || lanes > self.max_lanes || bytes > self.max_bytes {
            return Err(Error::Bounds);
        }
        Ok(bytes)
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_lanes: MAX_LANES,
            max_operations: MAX_OPERATIONS,
            max_bytes: MAX_ADMITTED_BYTES,
        }
    }
}

/// Public binding supplied by the authenticated request/compiler boundary.
/// Session IDs must be fresh across restarts. No persistent replay store exists
/// in this bounded core; source PRSS issuance must share this retirement domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Context {
    session: [u8; 32],
    schedule: [u8; 32],
    phase: u64,
}

impl Context {
    pub fn new(session: [u8; 32], schedule: [u8; 32], phase: u64) -> Result<Self, Error> {
        if session == [0; 32] || schedule == [0; 32] {
            return Err(Error::Binding);
        }
        Ok(Self {
            session,
            schedule,
            phase,
        })
    }
}

/// Public source parameters. Input is a ring64 signed integer with an admitted
/// bound INPUT_MIN <= X < INPUT_MAX_EXCLUSIVE. The core cannot check that bound
/// from one local replicated pair; the model compiler/caller must prove it.
/// For products/accumulators the proof is about the intended UNWRAPPED integer,
/// before ring reduction. An overflowing value wrapping into the gap is invalid.
/// `pivot` is the source's public role relabeling, not an extra trusted party.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operation {
    context: Context,
    sequence: u64,
    pivot: Party,
    lanes: usize,
    shift: u8,
    source: Option<SourceBinding>,
    consumer: Option<[u8; 32]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SourceBinding {
    setup: [u8; 32],
    public_start: u64,
    mask_start: u64,
    output_start: u64,
}

impl Operation {
    pub fn new(
        context: Context,
        sequence: u64,
        pivot: Party,
        lanes: usize,
        shift: u8,
        limits: Limits,
    ) -> Result<Self, Error> {
        limits.admit(lanes)?;
        if sequence >= limits.max_operations || !(1..=62).contains(&shift) {
            return Err(Error::Bounds);
        }
        Ok(Self {
            context,
            sequence,
            pivot,
            lanes,
            shift,
            source: None,
            consumer: None,
        })
    }

    pub fn lanes(self) -> usize {
        self.lanes
    }
    pub fn shift(self) -> u8 {
        self.shift
    }
    pub fn sequence(self) -> u64 {
        self.sequence
    }
    pub fn evaluator_a(self) -> Party {
        self.pivot
    }
    pub fn evaluator_b(self) -> Party {
        self.pivot.next()
    }
    pub fn helper(self) -> Party {
        self.pivot.next().next()
    }

    /// Includes request phase, complete schedule identity, ordinal, shape,
    /// shift, role pivot and fixed numerical profile. No private data is hashed.
    pub fn digest(self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(PROFILE.as_bytes());
        hash.update(self.context.session);
        hash.update(self.context.schedule);
        hash.update(self.context.phase.to_le_bytes());
        hash.update(self.sequence.to_le_bytes());
        hash.update([self.pivot as u8, self.shift]);
        hash.update((self.lanes as u64).to_le_bytes());
        if let Some(source) = self.source {
            hash.update(prss::PRSS_PROFILE.as_bytes());
            hash.update(source.setup);
            hash.update(b"public-pivot");
            hash.update(source.public_start.to_le_bytes());
            hash.update(b"pairwise-mask");
            hash.update(source.mask_start.to_le_bytes());
            hash.update(b"pairwise-reshare");
            hash.update(source.output_start.to_le_bytes());
        }
        if let Some(consumer) = self.consumer {
            hash.update(b"compound-consumer");
            hash.update(consumer);
        }
        hash.finalize().into()
    }
}

fn zero_words(n: usize) -> Result<Words, Error> {
    Words::new(n, None)
}

fn copy_words(values: &[u64], n: usize) -> Result<Words, Error> {
    if values.len() != n {
        return Err(Error::Shape);
    }
    let mut result = zero_words(n)?;
    result.copy_from_slice(values);
    Ok(result)
}

/// One party's (x_i, x_(i+1)) pair. Non-cloneable and wiped on drop.
/// Accessors expose only that local party's existing authorized view.
pub struct ReplicatedShare {
    operation: Operation,
    party: Party,
    first: Words,
    second: Words,
}

impl ReplicatedShare {
    pub(super) fn charge_to(&mut self, usage: &PayloadUsage) -> Result<(), Error> {
        self.first.charge_to(usage)?;
        self.second.charge_to(usage)
    }

    /// Outer-caller import, bounded per object. Admission to a runtime prices
    /// both original buffers before arithmetic or further owned allocations.
    pub fn from_local(
        operation: Operation,
        party: Party,
        first: &[u64],
        second: &[u64],
        limits: Limits,
    ) -> Result<Self, Error> {
        limits.admit(operation.lanes)?;
        if first.len() != operation.lanes || second.len() != operation.lanes {
            return Err(Error::Shape);
        }
        Ok(Self {
            operation,
            party,
            first: copy_words(first, operation.lanes)?,
            second: copy_words(second, operation.lanes)?,
        })
    }
    pub fn first(&self) -> &[u64] {
        &self.first
    }
    pub fn second(&self) -> &[u64] {
        &self.second
    }
    pub fn party(&self) -> Party {
        self.party
    }
    pub fn operation(&self) -> Operation {
        self.operation
    }
}

/// One-use, party-scoped input from the source's PRSS setup boundary.
/// No public constructor generates all parties' correlations. The setup owner
/// must provide fresh uniform, pairwise-consistent values and burn its issuance
/// ticket on handoff. The local dealer exists only in conformance tests.
pub struct LocalMaterial {
    operation: Operation,
    party: Party,
    data: MaterialData,
}

enum MaterialData {
    Evaluator {
        mask: Words,
        output: Words,
    },
    Helper {
        mask_a: Words,
        mask_b: Words,
        split: Option<Words>,
        output_a: Words,
        output_b: Words,
    },
}

impl LocalMaterial {
    fn charge_to(&mut self, usage: &PayloadUsage) -> Result<(), Error> {
        match &mut self.data {
            MaterialData::Evaluator { mask, output } => {
                mask.charge_to(usage)?;
                output.charge_to(usage)
            }
            MaterialData::Helper {
                mask_a,
                mask_b,
                split,
                output_a,
                output_b,
            } => {
                for words in [mask_a, mask_b, output_a, output_b] {
                    words.charge_to(usage)?;
                }
                if let Some(words) = split {
                    words.charge_to(usage)?;
                }
                Ok(())
            }
        }
    }

    pub fn evaluator(
        operation: Operation,
        party: Party,
        mask_component: &[u64],
        output_component: &[u64],
        limits: Limits,
    ) -> Result<Self, Error> {
        limits.admit(operation.lanes)?;
        if party == operation.helper() {
            return Err(Error::Role);
        }
        if mask_component.len() != operation.lanes || output_component.len() != operation.lanes {
            return Err(Error::Shape);
        }
        Ok(Self {
            operation,
            party,
            data: MaterialData::Evaluator {
                mask: copy_words(mask_component, operation.lanes)?,
                output: copy_words(output_component, operation.lanes)?,
            },
        })
    }

    /// Helper receives the same mask/output component as each evaluator under
    /// its respective pairwise PRSS channel. It never receives evaluator inputs.
    pub fn helper(
        operation: Operation,
        mask_a: &[u64],
        mask_b: &[u64],
        output_a: &[u64],
        output_b: &[u64],
        limits: Limits,
    ) -> Result<Self, Error> {
        limits.admit(operation.lanes)?;
        if [mask_a.len(), mask_b.len(), output_a.len(), output_b.len()] != [operation.lanes; 4] {
            return Err(Error::Shape);
        }
        Ok(Self {
            operation,
            party: operation.helper(),
            data: MaterialData::Helper {
                mask_a: copy_words(mask_a, operation.lanes)?,
                mask_b: copy_words(mask_b, operation.lanes)?,
                split: None,
                output_a: copy_words(output_a, operation.lanes)?,
                output_b: copy_words(output_b, operation.lanes)?,
            },
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessagePhase {
    Multiply,
    MaskedInput,
    Correlation,
    Reshare,
}

/// Public envelope; the receiving runtime checks every field against its own
/// admitted operation. Transport must authenticate sender and keep payload secret.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub binding: [u8; 32],
    pub from: Party,
    pub to: Party,
    pub phase: MessagePhase,
}

/// Owned, non-cloneable typed frame. This is not a network codec or authenticated
/// channel. `from_received` bounds its copy before allocation; a future wire
/// decoder must also bound raw bytes before allocating its own receive buffer.
/// Original runtime-issued payloads retain issuer reservations after export.
/// A raw received copy is charged on receiver admission; outer copies/containers
/// need their own transport policy and are not a process-global cap here.
pub struct Message {
    header: Header,
    words: Words,
}

impl Message {
    pub fn from_received(
        header: Header,
        words: &[u64],
        operation: Operation,
        limits: Limits,
    ) -> Result<Self, Error> {
        limits.admit(operation.lanes)?;
        if header.binding != operation.digest() {
            return Err(Error::Binding);
        }
        let expected = operation.lanes
            * if header.phase == MessagePhase::Correlation {
                2
            } else {
                1
            };
        Ok(Self {
            header,
            words: copy_words(words, expected)?,
        })
    }
    pub fn header(&self) -> Header {
        self.header
    }
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    fn validate(
        &self,
        operation: Operation,
        from: Party,
        to: Party,
        phase: MessagePhase,
    ) -> Result<(), Error> {
        if self.header.binding != operation.digest() {
            return Err(Error::Binding);
        }
        if self.header.from != from || self.header.to != to {
            return Err(Error::Role);
        }
        if self.header.phase != phase {
            return Err(Error::Phase);
        }
        let n = operation.lanes
            * if phase == MessagePhase::Correlation {
                2
            } else {
                1
            };
        if self.words.len() != n {
            return Err(Error::Shape);
        }
        Ok(())
    }
}

/// No secret-bearing object implements Clone or Debug. Public runtime state is
/// inspectable without displaying shares, masks or correlation payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ready,
    AwaitingOpen,
    AwaitingReshare,
    Complete,
    Retired,
}

#[cfg(test)]
mod tests;
