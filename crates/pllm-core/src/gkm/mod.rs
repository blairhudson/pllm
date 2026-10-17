//! Bounded, model-neutral Gupta--Katz--Miers protocol capsule (Figs. 1--5).
//!
//! Numeric completion: exact binary scaling by 2048, nearest ties-to-even,
//! finite inputs, signed i16 weights, unsigned wrapping ring64 share arithmetic,
//! signed reconstruction, then **i64 -> f32 -> division by 2^22**. Embeddings
//! are original finite IEEE-754 binary32 **little-endian bytes**, including signed
//! zero, with SHA-256 over those exact bytes. Fig. 3's self-reference typo is
//! resolved using the source prose: the second selector is `r XOR one_hot`.
//!
//! Owner publication uses secp256k1's scalar field, not its coordinate field.
//! Every linear response pair is owned/frozen before sampling a fresh unbiased
//! field scalar. Checks concern reconstructed signed relations under public
//! no-overflow bounds, NEVER individual wrapped share outputs. Repeated calls
//! accumulate per matrix; final fresh coefficients combine the matrix checks.
//! All dependent values remain tentative until the complete declared window
//! passes. Dropping any pending/partial/window owner cancels it. There is no
//! streaming acceptance, embedding cache, or detached verification queue.
//!
//! This scalar CPU conformance capsule does not implement a decoder, model
//! import, SDK, prefill batching, SIMD/GPU acceleration,
//! serialized publication, authenticated/confidential transport, persistent KV,
//! sampler, or deployment security/peak-memory/performance qualification. Two
//! worker views are in-process. The caller is the trusted client; it must not
//! release borrowed tentative values as accepted outputs. Public artifact pins
//! must arrive through an independently authenticated owner channel.
//! Optional AES seeded windows are separately identified, source-derived and
//! computationally private. They reuse the same transaction/verification owners;
//! verifier challenges never come from a shared seed. See `Client::begin_seeded`.
//!
//! Variable-size payload allocation is fallible and returns `Error::Allocation`.
//! Shared publication admission covers all clients, retained responses and
//! accepted-output owners; complete publication envelopes also share a process
//! cap. Fixed-size `Arc::new` admission handles (and callers' Arc wrappers) still
//! have stable Rust's abort-on-OOM semantics. This is not universal OOM recovery
//! or an RSS bound; standalone caller-owned primitives/copies and allocator
//! overhead are outside publication-envelope accounting.

mod lookup;
mod numeric;
mod payload;
mod publication;
mod seeded;
mod window;

pub use lookup::FloatTable;
pub use numeric::{encode_input, encode_weight, reconstruct_signed, rescale_signed, I16Matrix};
pub use publication::{Publication, PublicationResources, Stage};
pub use seeded::{SharingProfile, SEEDED_SHARING_PROFILE};
pub use window::{
    AcceptedWindow, Binding, Client, HalfLinear, HalfLookup, LinearResponse, LinearShare,
    LookupResponse, LookupShare, PendingLinear, PendingLookup, SeededWorker, Step, Window, Worker,
};

use k256::{elliptic_curve::PrimeField, Scalar};
use zeroize::Zeroizing;

/// A source-derived profile, not the author's truncating-cast profile.
pub const NUMERIC_PROFILE: &str =
    "gkm.nearest-even.2048.i16.ring64.signed.f32-before-div22.le-finite-f32.sha256.secp256k1.v1";
pub const MAX_DIMENSION: usize = 256;
pub const MAX_VOCABULARY: usize = 4096;
pub const MAX_STAGES: usize = 16;
pub const MAX_WINDOW_STEPS: usize = 128;
pub const MAX_PUBLICATION_BYTES: usize = 16 << 20;
/// Sum of complete admitted publication envelopes, including their reserved
/// live-window capacity. Additional publications share this existing-size cap.
pub const MAX_PROCESS_PUBLICATION_BYTES: usize = MAX_PUBLICATION_BYTES;
pub const MAX_CLIENTS_PER_PUBLICATION: u32 = 64;
pub const MAX_WINDOW_BYTES: usize = 1 << 20;
pub const MAX_LIVE_WINDOWS: u32 = 8;
pub const MAX_WINDOWS: u64 = 4096;
pub const MAX_RESERVED_STEPS: u64 = 65_536;
pub const MAX_PROCESS_LIVE_WINDOWS: u32 = 64;
pub const MAX_PROCESS_WINDOWS: u64 = 1_048_576;
pub const MAX_PROCESS_RESERVED_STEPS: u64 = 16_777_216;

/// Local errors are deliberately coarse. These are not worker diagnostic frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Shape,
    Limit,
    Numeric,
    Bound,
    Entropy,
    Binding,
    Schedule,
    Incomplete,
    Verification,
    Allocation,
    Retired,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GKM capsule rejected: {self:?}")
    }
}

impl std::error::Error for Error {}

// Production has no injectable/public RNG or seed-import interface. Independent
// OS draws are used for each purpose; verifier randomness never uses a shared PRG.
// The private trait permits deterministic failure/rejection tests only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Purpose {
    Owner,
    Session,
    LinearMask,
    LookupMask,
    LinearSeed,
    LookupSeed,
    ClaimChallenge,
    MatrixChallenge,
}

trait Entropy {
    fn fill(&mut self, purpose: Purpose, out: &mut [u8]) -> Result<(), Error>;
}

struct OsEntropy;

impl Entropy for OsEntropy {
    fn fill(&mut self, _: Purpose, out: &mut [u8]) -> Result<(), Error> {
        getrandom::fill(out).map_err(|_| Error::Entropy)
    }
}

fn sample_scalar(rng: &mut impl Entropy, purpose: Purpose) -> Result<Scalar, Error> {
    let mut bytes = Zeroizing::new([0u8; 32]);
    // Rejection, not reduction, gives uniform F_p including zero. A bounded
    // entropy/rejection failure aborts; it never falls back or biases a scalar.
    for _ in 0..128 {
        rng.fill(purpose, &mut *bytes)?;
        if let Some(value) = Option::<Scalar>::from(Scalar::from_repr((*bytes).into())) {
            return Ok(value);
        }
    }
    Err(Error::Entropy)
}

fn signed_scalar(value: i64) -> Scalar {
    let magnitude = Scalar::from(value.unsigned_abs());
    if value < 0 {
        -magnitude
    } else {
        magnitude
    }
}

#[cfg(test)]
mod tests;
