//! Deterministic injection and public fixture construction, compiled ONLY under
//! cfg(test). Not a production source-ensemble/noise API or security certificate.

use super::arithmetic::{below, OsRandom, RandomWords};
use super::code::{Code, Sparse};
use super::memory::Words;
use super::*;

std::thread_local! {
    static LIVE_LEASE: std::cell::Cell<Option<*const std::sync::atomic::AtomicBool>> = const {
        std::cell::Cell::new(None)
    };
}

pub(super) struct LeaseScope<'a> {
    previous: Option<*const std::sync::atomic::AtomicBool>,
    _flag: &'a std::sync::atomic::AtomicBool,
}

impl<'a> LeaseScope<'a> {
    pub(super) fn new(flag: &'a std::sync::atomic::AtomicBool) -> Self {
        Self {
            previous: LIVE_LEASE.with(|slot| slot.replace(Some(flag))),
            _flag: flag,
        }
    }
}

impl Drop for LeaseScope<'_> {
    fn drop(&mut self) {
        LIVE_LEASE.with(|slot| slot.set(self.previous));
    }
}

/// Only allocator callbacks during the scoped consuming operation may inspect
/// this flag. Client::observe_live_lease keeps the original Control Arc alive.
pub fn observed_lease_is_live() -> Option<bool> {
    LIVE_LEASE
        .try_with(|slot| {
            slot.get().map(|flag| {
                // SAFETY: the synchronous scope is nested inside the retained Control
                // owner's lifetime; callbacks and TLS access execute on the same thread.
                unsafe { &*flag }.load(std::sync::atomic::Ordering::Acquire)
            })
        })
        .ok()
        .flatten()
}

pub fn with_live_lease<T>(client: &mut Client, operation: impl FnOnce(&mut Client) -> T) -> T {
    client.observe_live_lease(operation)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixationEvent {
    FirstOriginalFixed,
    DependentNoise,
    OriginalsFixed,
    PrimaryDraw,
    AuxiliaryFixed,
    SecondaryDraw,
}

std::thread_local! {
    static EVENTS: std::cell::RefCell<Option<Vec<FixationEvent>>> = const {
        std::cell::RefCell::new(None)
    };
}

pub(super) fn record(event: FixationEvent) {
    EVENTS.with(|events| {
        if let Some(events) = events.borrow_mut().as_mut() {
            events.push(event);
        }
    });
}

pub fn begin_fixation_trace() {
    EVENTS.with(|events| *events.borrow_mut() = Some(Vec::with_capacity(8)));
}

pub fn take_fixation_trace() -> Vec<FixationEvent> {
    EVENTS.with(|events| events.borrow_mut().take().unwrap())
}

pub fn issue_dependent_fixed(
    client: &mut Client,
    input: &[u32],
    pairs: &[u32],
) -> Result<(DependentRequest, Request)> {
    client.issue_dependent_with(input, |memory| {
        Sparse::from_pairs(memory, LENGTH, PRIVACY_WEIGHT, pairs)
    })
}

pub fn continue_dependent_fixed(
    client: &mut Client,
    owner: DependentRequest,
    response: ReplyView<'_>,
    pairs: &[u32],
) -> Result<(DependentBatch, Request)> {
    client.continue_dependent_with(owner, response, |memory| {
        record(FixationEvent::DependentNoise);
        Sparse::from_pairs(memory, LENGTH, PRIVACY_WEIGHT, pairs)
    })
}

pub fn begin_dependent_fixed(
    client: &mut Client,
    owner: DependentBatch,
    response: ReplyView<'_>,
    pairs: [&[u32]; 2],
) -> Result<(ChallengedBatch, PrimaryChallenge)> {
    client.begin_dependent_with(owner, response, |memory| {
        record(FixationEvent::PrimaryDraw);
        Ok([
            Sparse::from_pairs(memory, LENGTH, CHECK_WEIGHT, pairs[0])?,
            Sparse::from_pairs(memory, LENGTH, CHECK_WEIGHT, pairs[1])?,
        ])
    })
}

/// Fail first privacy entropy without a new production seam.
pub fn issue_dependent_rng_failure(
    client: &mut Client,
    input: &[u32],
) -> Result<(DependentRequest, Request)> {
    client.issue_dependent_with(input, |memory| {
        Sparse::sample(memory, &mut FailedRandom, LENGTH, PRIVACY_WEIGHT)
    })
}

pub fn continue_dependent_rng_failure(
    client: &mut Client,
    owner: DependentRequest,
    response: ReplyView<'_>,
) -> Result<(DependentBatch, Request)> {
    client.continue_dependent_with(owner, response, |memory| {
        Sparse::sample(memory, &mut FailedRandom, LENGTH, PRIVACY_WEIGHT)
    })
}

pub fn begin_dependent_rng_failure(
    client: &mut Client,
    owner: DependentBatch,
    response: ReplyView<'_>,
) -> Result<(ChallengedBatch, PrimaryChallenge)> {
    client.begin_dependent_with(owner, response, |memory| {
        Ok([
            Sparse::sample(memory, &mut FailedRandom, LENGTH, CHECK_WEIGHT)?,
            Sparse::sample(memory, &mut FailedRandom, LENGTH, CHECK_WEIGHT)?,
        ])
    })
}

pub struct FixtureCode {
    pub dimension: usize,
    pub pi1: Vec<u32>,
    pub pi2: Vec<u32>,
    pub v1: Vec<u32>,
    pub v2: Vec<u32>,
}

impl FixtureCode {
    pub fn new(dimension: usize, tag: usize) -> Self {
        let length = 8 * dimension;
        Self {
            dimension,
            pi1: (0..length)
                .map(|j| ((5 * j + tag) % length) as u32)
                .collect(),
            pi2: (0..length)
                .map(|j| ((7 * j + 3 * tag) % length) as u32)
                .collect(),
            v1: (0..length)
                .map(|j| {
                    if j % 7 == 0 {
                        FIELD - 1 - j as u32
                    } else {
                        (17 * j + tag + 1) as u32
                    }
                })
                .collect(),
            v2: (0..length)
                .map(|j| {
                    if j % 5 == 0 {
                        FIELD - 1 - j as u32
                    } else {
                        (31 * j + tag + 1) as u32
                    }
                })
                .collect(),
        }
    }

    pub fn description(&self) -> CodeDescription<'_> {
        CodeDescription {
            field: FIELD,
            dimension: self.dimension,
            first_permutation: &self.pi1,
            second_permutation: &self.pi2,
            first_scale: &self.v1,
            second_scale: &self.v2,
        }
    }
}

pub struct Fixture {
    pub matrices: [Vec<u32>; 2],
    pub inputs: [Vec<u32>; 2],
    pub privacy: [FixtureCode; 2],
    pub primary: FixtureCode,
    pub secondary: FixtureCode,
    pub noise: [Vec<u32>; 2],
    pub primary_challenges: [Vec<u32>; 2],
    pub secondary_challenges: [Vec<u32>; 2],
}

pub fn challenge(length: usize, weight: usize, tag: usize) -> Vec<u32> {
    (0..weight)
        .flat_map(|j| {
            [
                ((3 * j + tag) % length) as u32,
                if j % 2 == 1 {
                    FIELD - 1 - j as u32
                } else {
                    7919 * (j + 1) as u32
                },
            ]
        })
        .collect()
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}

impl Fixture {
    pub fn new() -> Self {
        let matrix: Vec<u32> = (0..64 * 64)
            .map(|k| {
                let row = k / 64;
                let col = k % 64;
                (((row + 1) * 65537 - (col + 1) * 257) as u64 % u64::from(FIELD)) as u32
            })
            .collect();
        let input: Vec<u32> = (0..64)
            .map(|j| {
                if j % 3 == 0 {
                    FIELD - 1 - j
                } else {
                    j * 1234567
                }
            })
            .collect();
        Self {
            matrices: [
                matrix.clone(),
                matrix.iter().map(|&a| FIELD - 1 - a).collect(),
            ],
            inputs: [
                input.clone(),
                input.iter().map(|&a| FIELD - 1 - a).collect(),
            ],
            privacy: [FixtureCode::new(64, 2), FixtureCode::new(64, 3)],
            primary: FixtureCode::new(64, 11),
            secondary: FixtureCode::new(128, 13),
            noise: [challenge(512, 165, 5), challenge(512, 165, 6)],
            primary_challenges: [challenge(512, 31, 0), challenge(512, 31, 1)],
            secondary_challenges: [challenge(1024, 31, 2), challenge(1024, 31, 7)],
        }
    }

    pub fn source(&self) -> SetupInputs<'_> {
        SetupInputs {
            matrices: [&self.matrices[0], &self.matrices[1]],
            privacy_codes: [self.privacy[0].description(), self.privacy[1].description()],
            primary_code: self.primary.description(),
            secondary_code: self.secondary.description(),
        }
    }

    pub fn roles(&self) -> Result<(Client, Server)> {
        preprocess(AllocationPlan::bounded()?, self.source())?
            .validate(self.source())?
            .split()
    }

    pub fn direct(&self, slot: usize) -> Vec<u32> {
        self.matrices[slot]
            .chunks_exact(64)
            .map(|row| {
                (row.iter()
                    .zip(&self.inputs[slot])
                    .map(|(&a, &b)| u128::from(a) * u128::from(b))
                    .sum::<u128>()
                    % u128::from(FIELD)) as u32
            })
            .collect()
    }
}

pub fn issue_fixed(
    client: &mut Client,
    slot: usize,
    input: &[u32],
    pairs: &[u32],
) -> Result<(Query, Request)> {
    client.issue_query_with(slot, input, |memory| {
        Sparse::from_pairs(memory, LENGTH, PRIVACY_WEIGHT, pairs)
    })
}

pub fn issue_batch_fixed(
    client: &mut Client,
    inputs: [&[u32]; 2],
    pairs: [&[u32]; 2],
) -> Result<(Batch, [Request; 2])> {
    client.issue_batch_with(inputs, |memory| {
        Ok([
            Sparse::from_pairs(memory, LENGTH, PRIVACY_WEIGHT, pairs[0])?,
            Sparse::from_pairs(memory, LENGTH, PRIVACY_WEIGHT, pairs[1])?,
        ])
    })
}

pub fn query_snapshot(query: &Query) -> (&[u32], &[u32]) {
    (
        query.claim.input.as_slice(),
        query.claim.correction.as_slice(),
    )
}

pub fn finish_query_fixed(
    client: &mut Client,
    query: Query,
    response: ReplyView<'_>,
    pairs: [&[u32]; 2],
) -> Result<(AcceptedOutput, [[u32; 2]; 2])> {
    client
        .finish_query_with(query, response, |memory| {
            Ok([
                Sparse::from_pairs(memory, LENGTH, CHECK_WEIGHT, pairs[0])?,
                Sparse::from_pairs(memory, LENGTH, CHECK_WEIGHT, pairs[1])?,
            ])
        })
        .map(|(accepted, trace)| (accepted, *trace))
}

pub fn begin_fixed(
    client: &mut Client,
    batch: Batch,
    responses: &[ReplyView<'_>],
    pairs: [&[u32]; 2],
) -> Result<(ChallengedBatch, PrimaryChallenge)> {
    client.begin_with(batch, responses, |memory| {
        Ok([
            Sparse::from_pairs(memory, LENGTH, CHECK_WEIGHT, pairs[0])?,
            Sparse::from_pairs(memory, LENGTH, CHECK_WEIGHT, pairs[1])?,
        ])
    })
}

pub type FixedBatchTrace = [[u32; 2]; 4];

pub fn finish_batch_fixed(
    client: &mut Client,
    batch: ChallengedBatch,
    response: AuxiliaryReplyView<'_>,
    pairs: [&[u32]; 2],
) -> Result<(AcceptedBatch, FixedBatchTrace, FixedBatchTrace)> {
    client
        .finish_batch_with(batch, response, |memory| {
            record(FixationEvent::SecondaryDraw);
            Ok([
                Sparse::from_pairs(memory, SECONDARY_LENGTH, CHECK_WEIGHT, pairs[0])?,
                Sparse::from_pairs(memory, SECONDARY_LENGTH, CHECK_WEIGHT, pairs[1])?,
            ])
        })
        .map(|(accepted, trace)| (accepted, *trace.secondary, *trace.original))
}

struct FailedRandom;
impl RandomWords for FailedRandom {
    fn word(&mut self) -> Result<u32> {
        Err(Error::Randomness)
    }
}

/// Deterministic test entropy only. Zeros cause no rejection draws, so exactly
/// two successful words fill each sparse pair. The failing word itself receives
/// a partial fill through the same zeroizing byte-scratch helper as OsRandom.
struct FailAfter {
    remaining: usize,
    calls: usize,
}

impl RandomWords for FailAfter {
    fn word(&mut self) -> Result<u32> {
        self.calls += 1;
        if self.remaining == 0 {
            super::arithmetic::word_from_fill(|bytes| {
                bytes[..2].copy_from_slice(&[0x97, 0x42]);
                Err(Error::Randomness)
            })
        } else {
            self.remaining -= 1;
            Ok(0)
        }
    }
}

fn failing_noise(memory: &super::memory::Memory, successful_words: usize) -> Result<Sparse> {
    assert!(successful_words < 2 * PRIVACY_WEIGHT);
    let mut rng = FailAfter {
        remaining: successful_words,
        calls: 0,
    };
    let result = Sparse::sample(memory, &mut rng, LENGTH, PRIVACY_WEIGHT);
    assert_eq!(rng.calls, successful_words + 1);
    assert_eq!(result.as_ref().err(), Some(&Error::Randomness));
    result
}

fn failing_checks(
    memory: &super::memory::Memory,
    length: usize,
    successful_words: usize,
) -> Result<[Sparse; 2]> {
    assert!(successful_words < 4 * CHECK_WEIGHT);
    let mut rng = FailAfter {
        remaining: successful_words,
        calls: 0,
    };
    let result = (|| {
        Ok([
            Sparse::sample(memory, &mut rng, length, CHECK_WEIGHT)?,
            Sparse::sample(memory, &mut rng, length, CHECK_WEIGHT)?,
        ])
    })();
    assert_eq!(rng.calls, successful_words + 1);
    assert_eq!(result.as_ref().err(), Some(&Error::Randomness));
    result
}

pub fn issue_dependent_rng_failure_after(
    client: &mut Client,
    input: &[u32],
    successful_words: usize,
) -> Result<(DependentRequest, Request)> {
    client.issue_dependent_with(input, |memory| failing_noise(memory, successful_words))
}

pub fn continue_dependent_rng_failure_after(
    client: &mut Client,
    owner: DependentRequest,
    response: ReplyView<'_>,
    successful_words: usize,
) -> Result<(DependentBatch, Request)> {
    client.continue_dependent_with(owner, response, |memory| {
        failing_noise(memory, successful_words)
    })
}

pub fn begin_dependent_rng_failure_after(
    client: &mut Client,
    owner: DependentBatch,
    response: ReplyView<'_>,
    successful_words: usize,
) -> Result<(ChallengedBatch, PrimaryChallenge)> {
    client.begin_dependent_with(owner, response, |memory| {
        failing_checks(memory, LENGTH, successful_words)
    })
}

pub fn finish_batch_rng_failure_after(
    client: &mut Client,
    owner: ChallengedBatch,
    response: AuxiliaryReplyView<'_>,
    successful_words: usize,
) -> Result<AcceptedBatch> {
    client
        .finish_batch_with(owner, response, |memory| {
            failing_checks(memory, SECONDARY_LENGTH, successful_words)
        })
        .map(|v| v.0)
}

pub fn issue_rng_failure(client: &mut Client, input: &[u32]) -> Result<(Query, Request)> {
    client.issue_query_with(0, input, |memory| {
        Sparse::sample(memory, &mut FailedRandom, LENGTH, PRIVACY_WEIGHT)
    })
}

pub fn finish_query_rng_failure(
    client: &mut Client,
    query: Query,
    response: ReplyView<'_>,
) -> Result<AcceptedOutput> {
    client
        .finish_query_with(query, response, |memory| {
            Ok([
                Sparse::sample(memory, &mut FailedRandom, LENGTH, CHECK_WEIGHT)?,
                Sparse::sample(memory, &mut FailedRandom, LENGTH, CHECK_WEIGHT)?,
            ])
        })
        .map(|v| v.0)
}

pub fn begin_rng_failure(
    client: &mut Client,
    batch: Batch,
    responses: &[ReplyView<'_>],
) -> Result<(ChallengedBatch, PrimaryChallenge)> {
    client.begin_with(batch, responses, |memory| {
        Ok([
            Sparse::sample(memory, &mut FailedRandom, LENGTH, CHECK_WEIGHT)?,
            Sparse::sample(memory, &mut FailedRandom, LENGTH, CHECK_WEIGHT)?,
        ])
    })
}

pub fn finish_batch_rng_failure(
    client: &mut Client,
    batch: ChallengedBatch,
    response: AuxiliaryReplyView<'_>,
) -> Result<AcceptedBatch> {
    client
        .finish_batch_with(batch, response, |memory| {
            Ok([
                Sparse::sample(memory, &mut FailedRandom, SECONDARY_LENGTH, CHECK_WEIGHT)?,
                Sparse::sample(memory, &mut FailedRandom, SECONDARY_LENGTH, CHECK_WEIGHT)?,
            ])
        })
        .map(|v| v.0)
}

struct WordStream<'a>(std::slice::Iter<'a, u32>);
impl RandomWords for WordStream<'_> {
    fn word(&mut self) -> Result<u32> {
        self.0.next().copied().ok_or(Error::Randomness)
    }
}

pub fn bounded_draw(upper: u32, words: &[u32]) -> Result<u32> {
    below(&mut WordStream(words.iter()), upper)
}

pub fn sampled_pairs(length: usize, weight: usize, words: Option<&[u32]>) -> Result<Vec<u32>> {
    let memory = AllocationPlan::bounded()?.start();
    let sparse = if let Some(words) = words {
        Sparse::sample(&memory, &mut WordStream(words.iter()), length, weight)?
    } else {
        Sparse::sample(&memory, &mut OsRandom, length, weight)?
    };
    Ok(sparse.pairs().to_vec())
}

pub fn encode(description: CodeDescription<'_>, input: &[u32]) -> Result<Vec<u32>> {
    let memory = AllocationPlan::bounded()?.start();
    Ok(Code::copy(&memory, description)?
        .encode(&memory, input)?
        .as_slice()
        .to_vec())
}

pub fn syndrome(description: CodeDescription<'_>, pairs: &[u32]) -> Result<Vec<u32>> {
    let memory = AllocationPlan::bounded()?.start();
    let sparse = Sparse::from_pairs(
        &memory,
        description.dimension * REPEAT,
        pairs.len() / 2,
        pairs,
    )?;
    Ok(Code::copy(&memory, description)?
        .syndrome(&memory, &sparse)?
        .as_slice()
        .to_vec())
}

/// Used only by the allocator test to observe erasure immediately before free.
pub fn erase_probe(client: &Client, observe_pointer: impl FnOnce(*const u32, usize)) -> Result<()> {
    let mut secret = Words::zeros(&client.material.memory, 64)?;
    secret.as_mut().fill(0x12345678);
    observe_pointer(secret.as_slice().as_ptr(), 64 * 4);
    drop(secret);
    Ok(())
}

/// Non-secret counter lifetime retained across consuming failure paths.
pub struct MemoryWatch(super::memory::Memory);

impl MemoryWatch {
    pub fn stats(&self) -> MemoryStats {
        self.0.stats()
    }
    pub fn owner_count(&self) -> usize {
        self.0.owner_count()
    }
}

pub fn capture_next_memory() {
    super::memory::audit::capture_next();
}

pub fn take_memory_watch() -> MemoryWatch {
    MemoryWatch(super::memory::audit::take_capture())
}

/// Pointer remains alive throughout the observed allocation and every owned
/// buffer deallocation. The test allocator may read this atomic counter only.
pub fn allocation_context() -> Option<(*const std::sync::atomic::AtomicUsize, usize)> {
    super::memory::audit::allocation_context()
}

pub fn partial_entropy_failure() -> Result<u32> {
    super::arithmetic::word_from_fill(|bytes| {
        bytes[..2].copy_from_slice(&[0x97, 0x42]);
        Err(Error::Randomness)
    })
}
