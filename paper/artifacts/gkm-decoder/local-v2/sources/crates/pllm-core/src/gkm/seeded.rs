//! Source-qualified AES byte stream; not the stochastic sampler or a verifier RNG.
//! See docs/research/gkm-seeded-shares.md for the frozen source profile.
use aes::{
    cipher::{Block, BlockEncrypt, KeyInit},
    Aes128Enc,
};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use super::Error;

pub const SEEDED_SHARING_PROFILE: &str = "gkm.seeded.aes128.duplicate-le64.two-purpose.window.v1";

/// Sharing privacy is distinct from the unchanged numeric/verification profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharingProfile {
    /// Independent OS-random full shares; the default admission identity.
    FullShares,
    /// Computational privacy under AES PRG security and worker non-collusion.
    Aes128WindowV1,
}

// No Clone, Debug, seed export, cursor import, or reset. AES's enabled zeroize
// feature wipes its expanded key on drop. We retain no redundant raw key.
pub(super) struct Stream {
    cipher: Aes128Enc,
    buffer: Zeroizing<[u8; 16]>,
    position: u128,
    limit: u128,
    retired: bool,
}

impl Stream {
    const DOMAIN_BYTES: u128 = 1u128 << 68;

    pub(super) fn new(key: &[u8; 16], limit: u128) -> Result<Self, Error> {
        if limit > Self::DOMAIN_BYTES {
            return Err(Error::Limit);
        }
        Ok(Self {
            cipher: Aes128Enc::new(key.into()),
            buffer: Zeroizing::new([0; 16]),
            position: 0,
            limit,
            retired: false,
        })
    }

    pub(super) fn fill(&mut self, mut out: &mut [u8]) -> Result<(), Error> {
        if self.retired {
            return Err(Error::Retired);
        }
        // Preflight before touching output/cursor; overflow never wraps a block.
        if self
            .position
            .checked_add(out.len() as u128)
            .is_none_or(|end| end > self.limit || end > Self::DOMAIN_BYTES)
        {
            self.retired = true;
            return Err(Error::Limit);
        }
        while !out.is_empty() {
            let offset = (self.position % 16) as usize;
            if offset == 0 {
                let counter = (self.position / 16) as u64;
                self.buffer[..8].copy_from_slice(&counter.to_le_bytes());
                self.buffer[8..].copy_from_slice(&counter.to_le_bytes());
                self.cipher
                    .encrypt_block(Block::<Aes128Enc>::from_mut_slice(&mut *self.buffer));
            }
            let count = out.len().min(16 - offset);
            out[..count].copy_from_slice(&self.buffer[offset..offset + count]);
            self.position += count as u128;
            out = &mut out[count..];
        }
        Ok(())
    }

    pub(super) fn complete(&self) -> bool {
        !self.retired && self.position == self.limit
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.position.zeroize();
        self.limit.zeroize();
        self.retired.zeroize();
        // cipher and buffer then zeroize through their own drops.
    }
}
impl ZeroizeOnDrop for Stream {}

pub(super) struct Streams {
    pub(super) linear: Stream,
    pub(super) lookup: Stream,
}

impl Streams {
    pub(super) fn new(
        linear_key: &[u8; 16],
        lookup_key: &[u8; 16],
        linear_bytes: usize,
        lookup_bytes: usize,
    ) -> Result<Self, Error> {
        Ok(Self {
            linear: Stream::new(linear_key, linear_bytes as u128)?,
            lookup: Stream::new(lookup_key, lookup_bytes as u128)?,
        })
    }

    pub(super) fn complete(&self) -> bool {
        self.linear.complete() && self.lookup.complete()
    }
}

/// Intentional extra owned state/copies beyond the existing full-share envelope.
/// Schedule payload is shared, never duplicated; its existing full-window charge
/// remains. Fixed handle/control allowance includes the shared plan Arc.
pub(super) const WINDOW_EXTRA_BYTES: usize = 2 * size_of::<Streams>()
    + super::MAX_DIMENSION * 8
    + super::MAX_VOCABULARY.div_ceil(8)
    + 32 // two temporary OS seed arrays, copied once into each endpoint's cipher
    + 512; // public control/Arc metadata and fixed block/key temporaries

#[cfg(test)]
#[path = "seeded_tests.rs"]
pub(super) mod tests;
