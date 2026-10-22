//! Pinned SPU/YACL source PRSS for bounded, sequential TruncAPr schedules.
//!
//! SPU a42c261 pins YACL 5418371. Each fill resets AES-128-CTR to the
//! little-endian u128 representation of its u64 block cursor. Inside that fill,
//! the AES counter increments big-endian. Partial blocks are discarded.
//! This is deliberately not a conventional continuously buffered CTR stream.
//!
//! Setup types are party-local and non-cloneable. No seed import, export,
//! persistence, cursor reset or source `fork` API exists. Typed seed messages
//! must travel over authenticated private channels; no wire codec is supplied.
//! The nondefault `insecure-test-support` feature adds a separate fresh-graph
//! fixture using publicly known keys, never production entropy or live owners.

use super::*;
use aes::cipher::{generic_array::GenericArray, BlockEncrypt, KeyInit};
use aes::Aes128;

#[path = "multiply.rs"]
pub mod multiply;

#[cfg(feature = "insecure-test-support")]
#[path = "insecure_test_support.rs"]
pub mod insecure_test_support;

pub const PRSS_PROFILE: &str = "puma.spu-a42c261.yacl-5418371.prss.le64.aes128-ctr-call-reset.v1";
pub const MAX_AES_BLOCKS: u64 = (3 * MAX_LANES as u64 + 3) * MAX_OPERATIONS;
/// The original maximum complete-program AES allowance, independently priced
/// from top-level ordinals: a conversion has many nested Boolean rounds.
pub const MAX_PROGRAM_AES_BLOCKS: u64 = (3 * MAX_LANES as u64 + 3) * MAX_PROGRAM_OPERATIONS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrssLimits {
    core: Limits,
    aes_blocks: u64,
}

impl PrssLimits {
    pub fn program(max_operations: u64, aes_blocks: u64) -> Result<Self, Error> {
        let core = Limits::program(max_operations)?;
        if aes_blocks > MAX_PROGRAM_AES_BLOCKS {
            return Err(Error::Bounds);
        }
        Ok(Self { core, aes_blocks })
    }
    pub fn new(core: Limits, aes_blocks: u64) -> Result<Self, Error> {
        if aes_blocks == 0 || aes_blocks > MAX_AES_BLOCKS {
            return Err(Error::Bounds);
        }
        Ok(Self { core, aes_blocks })
    }
}
impl Default for PrssLimits {
    fn default() -> Self {
        Self {
            core: Limits::default(),
            aes_blocks: MAX_AES_BLOCKS,
        }
    }
}

/// Public audit counters. Pair blocks count the shared logical cursor once,
/// even when H expands both adjacent streams. AES work counts both expansions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursors {
    pub public_blocks: u64,
    pub pair_blocks: u64,
    pub private_blocks: u64,
    pub aes_work: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupStatus {
    Gathering,
    SeedExchange,
    Ready,
    Retired,
}

/// This seed contribution is intentionally public (source AllGather).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicContribution {
    pub(super) context: Context,
    pub(super) from: Party,
    pub(super) value: u128,
}
impl PublicContribution {
    pub fn sender(self) -> Party {
        self.from
    }
    pub fn value(self) -> u128 {
        self.value
    }
}

/// One directed secret seed, sent to the previous party. Never Clone/Debug.
/// The seed is inaccessible to a central router through this typed API.
pub struct SeedMessage {
    pub(super) setup: [u8; 32],
    pub(super) from: Party,
    pub(super) to: Party,
    pub(super) seed: Zeroizing<[u8; 16]>,
}
impl SeedMessage {
    pub fn sender(&self) -> Party {
        self.from
    }
    pub fn recipient(&self) -> Party {
        self.to
    }
}

struct LocalSeeds {
    private: Zeroizing<[u8; 16]>,
    // The only two adjacent edge keys held by this party.
    own: Zeroizing<[u8; 16]>,
    next: Option<Zeroizing<[u8; 16]>>,
}

/// Source setup, with fresh independent OS draws for public, private and own
/// pairwise seeds. Gathering binds each role's public contribution before the
/// directed seed exchange, so stale seed messages fail even if a caller reuses
/// its original Context. No reconstruction or all-party seed registry exists.
pub struct SetupParty {
    context: Context,
    party: Party,
    limits: PrssLimits,
    status: SetupStatus,
    contributions: [Option<u128>; 3],
    offered: bool,
    seeds: Option<LocalSeeds>,
    setup: [u8; 32],
    public_seed: u128,
    pid: u32,
}

impl SetupParty {
    pub fn new(context: Context, party: Party, limits: PrssLimits) -> Result<Self, Error> {
        Self::with_entropy(context, party, limits, &mut OsSeeds)
    }

    fn with_entropy(
        context: Context,
        party: Party,
        limits: PrssLimits,
        rng: &mut impl SeedEntropy,
    ) -> Result<Self, Error> {
        // Fixed-size seed buffers only; all variable payload admission is later.
        limits.core.admit(1)?;
        let public = rng.draw(SeedPurpose::Public)?;
        let private = rng.draw(SeedPurpose::Private)?;
        let own = rng.draw(SeedPurpose::Pairwise)?;
        let mut contributions = [None; 3];
        contributions[party as usize] = Some(u128::from_le_bytes(*public));
        Ok(Self {
            context,
            party,
            limits,
            status: SetupStatus::Gathering,
            contributions,
            offered: false,
            seeds: Some(LocalSeeds {
                private,
                own,
                next: None,
            }),
            setup: [0; 32],
            public_seed: 0,
            pid: std::process::id(),
        })
    }

    pub fn status(&self) -> SetupStatus {
        self.status
    }
    pub fn cancel(&mut self) {
        self.seeds = None;
        self.status = SetupStatus::Retired;
    }

    fn checked<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.cancel();
        }
        result
    }

    fn phase(&self, expected: SetupStatus) -> Result<(), Error> {
        if self.status == SetupStatus::Retired || self.pid != std::process::id() {
            return Err(Error::Retired);
        }
        if self.status != expected {
            return Err(Error::Phase);
        }
        Ok(())
    }

    /// Issue once; broadcast this public Copy value to the other two roles.
    pub fn contribution(&mut self) -> Result<PublicContribution, Error> {
        let result = (|| {
            self.phase(SetupStatus::Gathering)?;
            if self.offered {
                return Err(Error::Phase);
            }
            self.offered = true;
            Ok(PublicContribution {
                context: self.context,
                from: self.party,
                value: self.contributions[self.party as usize].ok_or(Error::Phase)?,
            })
        })();
        self.checked(result)
    }

    pub fn receive_public(&mut self, message: PublicContribution) -> Result<(), Error> {
        let result = (|| {
            self.phase(SetupStatus::Gathering)?;
            if !self.offered {
                return Err(Error::Phase);
            }
            if message.context != self.context {
                return Err(Error::Binding);
            }
            if message.from == self.party || self.contributions[message.from as usize].is_some() {
                return Err(Error::Role);
            }
            self.contributions[message.from as usize] = Some(message.value);
            Ok(())
        })();
        self.checked(result)
    }

    /// Follow executable source direction: send self_seed to PrevRank, receive
    /// NextRank's self_seed. The source comment describes the opposite direction.
    pub fn share_seed(&mut self) -> Result<SeedMessage, Error> {
        let result = (|| {
            self.phase(SetupStatus::Gathering)?;
            if !self.offered {
                return Err(Error::Phase);
            }
            let mut hash = Sha256::new();
            hash.update(PRSS_PROFILE.as_bytes());
            hash.update(self.context.session);
            hash.update(self.context.schedule);
            hash.update(self.context.phase.to_le_bytes());
            let mut public_seed = 0_u128;
            for (role, value) in self.contributions.iter().enumerate() {
                let value = value.ok_or(Error::Phase)?;
                hash.update([role as u8]);
                hash.update(value.to_le_bytes());
                public_seed = public_seed.wrapping_add(value);
            }
            self.setup = hash.finalize().into();
            self.public_seed = public_seed;
            self.status = SetupStatus::SeedExchange;
            Ok(SeedMessage {
                setup: self.setup,
                from: self.party,
                to: self.party.next().next(),
                seed: Zeroizing::new(*self.seeds.as_ref().ok_or(Error::Retired)?.own),
            })
        })();
        self.checked(result)
    }

    pub fn receive_seed(&mut self, message: SeedMessage) -> Result<(), Error> {
        let result = (|| {
            self.phase(SetupStatus::SeedExchange)?;
            if message.setup != self.setup {
                return Err(Error::Binding);
            }
            if message.from != self.party.next() || message.to != self.party {
                return Err(Error::Role);
            }
            self.seeds.as_mut().ok_or(Error::Retired)?.next = Some(message.seed);
            self.status = SetupStatus::Ready;
            Ok(())
        })();
        self.checked(result)
    }

    pub fn finish(&mut self) -> Result<SourceParty, Error> {
        let result = (|| {
            self.phase(SetupStatus::Ready)?;
            let seeds = self.seeds.take().ok_or(Error::Retired)?;
            if seeds.next.is_none() {
                return Err(Error::Phase);
            }
            self.status = SetupStatus::Retired;
            let context = Context {
                session: self.setup,
                ..self.context
            };
            Ok(SourceParty {
                runtime: PartyRuntime::new(context, self.party, self.limits.core),
                setup_context: self.context,
                context,
                party: self.party,
                limits: self.limits,
                setup: self.setup,
                public_seed: self.public_seed,
                seeds: Some(seeds),
                cursors: Cursors::default(),
                pending: None,
                pid: self.pid,
            })
        })();
        self.checked(result)
    }
}
impl Drop for SetupParty {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// One role's PRSS state and truncation runtime, inseparable on success/failure.
/// No API releases material to a detachable owner, resets cursors, or imports
/// old seeds. Production setup always draws new OS seeds, including after restart.
pub struct SourceParty {
    runtime: PartyRuntime,
    setup_context: Context,
    context: Context,
    party: Party,
    limits: PrssLimits,
    setup: [u8; 32],
    public_seed: u128,
    seeds: Option<LocalSeeds>,
    cursors: Cursors,
    pending: Option<Operation>,
    pid: u32,
}

impl SourceParty {
    pub(crate) fn setup_context(&self) -> Context {
        self.setup_context
    }
    pub fn context(&self) -> Context {
        self.context
    }
    pub fn party(&self) -> Party {
        self.party
    }
    pub fn cursors(&self) -> Cursors {
        self.cursors
    }
    pub fn payload_usage(&self) -> PayloadUsage {
        self.runtime.payload_usage()
    }
    pub fn status(&self) -> Status {
        self.runtime.status()
    }
    pub fn has_pending_operation(&self) -> bool {
        self.pending.is_some()
    }
    pub fn cancel(&mut self) {
        self.seeds = None;
        self.pending = None;
        self.runtime.cancel();
    }

    fn live(&self) -> Result<(), Error> {
        if self.seeds.is_none() || self.pid != std::process::id() {
            return Err(Error::Retired);
        }
        Ok(())
    }
    fn checked<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.cancel();
        }
        result
    }
    fn work(&self, additional: u64) -> Result<u64, Error> {
        self.cursors
            .aes_work
            .checked_add(additional)
            .filter(|&n| n <= self.limits.aes_blocks)
            .ok_or(Error::Bounds)
    }

    fn admit_work(&self, budget: WorkBudget) -> Result<(), Error> {
        budget.admit(
            self.runtime.next_sequence(),
            self.cursors,
            self.limits.core.max_operations,
            self.limits.aes_blocks,
        )
    }

    /// Reserve one public pivot draw after validating shape/shift/work. A second
    /// reservation, or abandoning this plan via cancel/drop, retires all keys.
    pub fn plan(&mut self, lanes: usize, shift: u8) -> Result<Operation, Error> {
        let result = (|| {
            self.live()?;
            if self.pending.is_some() || self.runtime.status() != Status::Ready {
                return Err(Error::Phase);
            }
            let mut op = Operation::new(
                self.context,
                self.runtime.next_sequence(),
                Party::P0,
                lanes,
                shift,
                self.limits.core,
            )?;
            let public_start = self.cursors.public_blocks;
            let next_public = public_start.checked_add(1).ok_or(Error::Bounds)?;
            let q = blocks(lanes)?;
            let output_start = self
                .cursors
                .pair_blocks
                .checked_add(q)
                .ok_or(Error::Bounds)?;
            output_start.checked_add(q).ok_or(Error::Bounds)?;
            let work = self.work(1)?;
            self.cursors.public_blocks = next_public; // burn before expansion
            self.cursors.aes_work = work;
            let mut word = [0_u64; 1]; // public, never secret
            fill(&self.public_seed.to_le_bytes(), public_start, &mut word)?;
            op.pivot = pivot(word[0])?;
            op.source = Some(SourceBinding {
                setup: self.setup,
                public_start,
                mask_start: self.cursors.pair_blocks,
                output_start,
            });
            self.pending = Some(op);
            Ok(op)
        })();
        self.checked(result)
    }

    pub fn begin(&mut self, input: ReplicatedShare) -> Result<[Option<Message>; 2], Error> {
        let result = self.begin_inner(input);
        self.checked(result)
    }

    fn begin_inner(&mut self, mut input: ReplicatedShare) -> Result<[Option<Message>; 2], Error> {
        self.live()?;
        let op = self.pending.take().ok_or(Error::Phase)?;
        if input.operation != op {
            return Err(Error::Binding);
        }
        if input.party != self.party {
            return Err(Error::Role);
        }
        self.limits.core.admit(op.lanes)?;
        let source = op.source.ok_or(Error::Binding)?;
        if source.mask_start != self.cursors.pair_blocks || source.setup != self.setup {
            return Err(Error::Binding);
        }
        let n = op.lanes;
        let q = blocks(n)?;
        let helper = self.party == op.helper();
        let private_start = self.cursors.private_blocks;
        let private_end = private_start
            .checked_add(if helper { n as u64 } else { 0 })
            .ok_or(Error::Bounds)?;
        let pair_end = source.output_start.checked_add(q).ok_or(Error::Bounds)?;
        let work = self.work(if helper { 4 * q + n as u64 } else { 2 * q })?;
        self.cursors.pair_blocks = pair_end;
        self.cursors.private_blocks = private_end;
        self.cursors.aes_work = work; // reserve all expansions BEFORE buffers
        let usage = self.runtime.payload_usage();
        // Price the original caller-to-runtime transfer before any PRSS buffer.
        input.charge_to(&usage)?;
        let seeds = self.seeds.as_ref().ok_or(Error::Retired)?;
        let next = seeds.next.as_ref().ok_or(Error::Phase)?;
        let data = if helper {
            MaterialData::Helper {
                mask_a: expand(next, source.mask_start, n, Some(&usage))?,
                mask_b: expand(&seeds.own, source.mask_start, n, Some(&usage))?,
                split: Some(expand(&seeds.private, private_start, 2 * n, Some(&usage))?),
                output_a: expand(next, source.output_start, n, Some(&usage))?,
                output_b: expand(&seeds.own, source.output_start, n, Some(&usage))?,
            }
        } else {
            let key = if self.party == op.evaluator_a() {
                &seeds.own
            } else {
                next
            };
            MaterialData::Evaluator {
                mask: expand(key, source.mask_start, n, Some(&usage))?,
                output: expand(key, source.output_start, n, Some(&usage))?,
            }
        };
        let material = LocalMaterial {
            operation: op,
            party: self.party,
            data,
        };
        self.runtime
            .begin_with_randomness(op, input, material, &mut IssuedSplits)
    }

    pub fn open(&mut self, peer_open: Message, correlation: Message) -> Result<Message, Error> {
        let result = self
            .live()
            .and_then(|()| self.runtime.open(peer_open, correlation));
        self.checked(result)
    }
    pub fn reshare(&mut self, peer: Message) -> Result<(), Error> {
        let result = self.live().and_then(|()| self.runtime.reshare(peer));
        self.checked(result)
    }
    pub fn take_output(&mut self) -> Result<ReplicatedShare, Error> {
        let result = self.live().and_then(|()| self.runtime.take_output());
        self.checked(result)
    }
}
impl Drop for SourceParty {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn blocks(words: usize) -> Result<u64, Error> {
    typed_blocks(words, 8)
}

fn typed_blocks(words: usize, word_bytes: usize) -> Result<u64, Error> {
    if words == 0 || words > 2 * MAX_LANES {
        return Err(Error::Bounds);
    }
    if !matches!(word_bytes, 1 | 2 | 4 | 8) {
        return Err(Error::Bounds);
    }
    Ok((words as u64 * word_bytes as u64).div_ceil(16))
}

fn fill(key: &[u8; 16], start: u64, out: &mut [u64]) -> Result<u64, Error> {
    fill_typed(key, start, out, 8)
}

// SPU Boolean shares select the smallest 8/16/32/64-bit backing type.
// PRSS fills that byte width, not one ring64 word per lane. Storage in this
// bounded owner is still u64 and fully charged; only source AES consumption
// follows the narrow type. Unused bytes at the end of each fill are discarded.
fn fill_typed(
    key: &[u8; 16],
    start: u64,
    out: &mut [u64],
    word_bytes: usize,
) -> Result<u64, Error> {
    let q = typed_blocks(out.len(), word_bytes)?;
    let end = start.checked_add(q).ok_or(Error::Bounds)?;
    let cipher = Aes128::new(GenericArray::from_slice(key));
    let initial = u128::from_be_bytes(u128::from(start).to_le_bytes());
    for (offset, words) in out.chunks_mut(16 / word_bytes).enumerate() {
        let value = initial.checked_add(offset as u128).ok_or(Error::Bounds)?;
        let mut block = Zeroizing::new(value.to_be_bytes());
        cipher.encrypt_block(GenericArray::from_mut_slice(&mut *block));
        for (i, word) in words.iter_mut().enumerate() {
            *word = 0;
            for j in 0..word_bytes {
                *word |= u64::from(block[word_bytes * i + j]) << (8 * j);
            }
        }
    }
    Ok(end)
}

fn expand(
    key: &[u8; 16],
    start: u64,
    n: usize,
    usage: Option<&PayloadUsage>,
) -> Result<Words, Error> {
    let q = blocks(n)?;
    start.checked_add(q).ok_or(Error::Bounds)?;
    let mut out = Words::new(n, usage)?;
    fill(key, start, &mut out)?;
    Ok(out)
}

fn pivot(raw: u64) -> Result<Party, Error> {
    // Source computes (raw+1)%3 and (raw+2)%3 before reduction. Its two
    // overflowing raw values alias roles. Reject/burn, never silently resample.
    if raw > u64::MAX - 2 {
        return Err(Error::Bounds);
    }
    Ok(match raw % 3 {
        0 => Party::P0,
        1 => Party::P1,
        _ => Party::P2,
    })
}

// Source-owned helpers must carry their already-issued private split. Never
// silently replace it with OS entropy at the arithmetic boundary.
struct IssuedSplits;
impl protocol::Entropy for IssuedSplits {
    fn fill(&mut self, _: &mut [u64]) -> Result<(), Error> {
        Err(Error::Phase)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeedPurpose {
    Public,
    Private,
    Pairwise,
}
trait SeedEntropy {
    fn draw(&mut self, purpose: SeedPurpose) -> Result<Zeroizing<[u8; 16]>, Error>;
}
struct OsSeeds;
impl SeedEntropy for OsSeeds {
    fn draw(&mut self, _: SeedPurpose) -> Result<Zeroizing<[u8; 16]>, Error> {
        let mut bytes = Zeroizing::new([0; 16]);
        getrandom::fill(&mut *bytes).map_err(|_| Error::Entropy)?;
        Ok(bytes)
    }
}

#[cfg(test)]
#[path = "prss_tests.rs"]
mod tests;
