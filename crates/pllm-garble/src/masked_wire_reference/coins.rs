//! Fresh party-local coin tapes, never client wire masks.
//!
//! Only independent key roots/leaves and party zero's uniform additive shares
//! use these tapes. Correlation-dependent corrections remain explicit. The
//! 256-bit seeds are independent for each party and each bounded issuance;
//! domain separation binds role, plan/session/tensor context and fresh nonce.
use super::*;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialEncoding {
    Full,
    Seeded,
}

pub(super) struct Tape {
    cipher: Aes256,
    counter: u64,
}

impl Tape {
    pub(super) fn new(seed: &[u8; 32], binding: &[u8; 32], nonce: &[u8; 32], party: u8) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"pllm.masked_wire.party-key-coins.v1\0");
        hash.update(seed);
        hash.update(binding);
        hash.update(nonce);
        hash.update([party]);
        let key = Zeroizing::new(<[u8; 32]>::from(hash.finalize()));
        Self {
            cipher: Aes256::new((&*key).into()),
            counter: 0,
        }
    }

    pub(super) fn word(&mut self) -> u128 {
        // At most 256 lanes and fewer than 64 random fields per lane. This
        // internal limit catches accidental unbounded use before counter wrap.
        assert!(self.counter < 256 * 64);
        let mut block = (u128::from(self.counter)).to_le_bytes().into();
        self.cipher.encrypt_block(&mut block);
        self.counter += 1;
        u128::from_le_bytes(block.into())
    }
}

#[derive(Default)]
pub(super) struct Coins {
    tapes: Option<[Tape; 2]>,
}

impl Coins {
    pub(super) fn new(seeds: &[[u8; 32]; 2], binding: &[u8; 32], nonce: &[u8; 32]) -> Self {
        Self {
            tapes: Some([
                Tape::new(&seeds[0], binding, nonce, 0),
                Tape::new(&seeds[1], binding, nonce, 1),
            ]),
        }
    }

    pub(super) fn key(&mut self, bits: u8, point: u64) -> Result<[Key; 2], Error> {
        match self.tapes.as_mut() {
            Some(tapes) => Key::issue_with(bits, point, |party| Ok(tapes[party].word())),
            None => Key::issue(bits, point),
        }
        .map_err(|_| Error::Randomness)
    }

    pub(super) fn share(&mut self, value: u64, bits: u8) -> Result<[u64; 2], Error> {
        match self.tapes.as_mut() {
            Some(tapes) => {
                let left = tapes[0].word() as u64 & mask(bits);
                Ok([left, value.wrapping_sub(left) & mask(bits)])
            }
            None => share(value, bits),
        }
    }
}
