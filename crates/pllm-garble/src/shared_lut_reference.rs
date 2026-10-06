//! Bounded clean-room reference for Curl §4.1.1, Fig. 4.
//!
//! The dealer exists only in this in-process research API. Each evaluator owns
//! one additive share. Input indices are already in the table's ring: protected
//! truncation, distributed issuance, transport authentication and decoder
//! integration are separate, unimplemented obligations.

use sha2::{Digest, Sha256};
use std::sync::Arc;
use zeroize::Zeroizing;

pub const MAX_ROWS: usize = 4096;
pub const OPENING_BYTES: usize = 68;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Domain,
    Randomness,
    Consumed,
    Context,
}

/// Immutable public u32-ring table. Construction binds every value and its size.
pub struct Table {
    values: Arc<[u32]>,
    digest: [u8; 32],
}

impl Table {
    pub fn new(values: &[u32]) -> Result<Self, Error> {
        if !(2..=MAX_ROWS).contains(&values.len()) || !values.len().is_power_of_two() {
            return Err(Error::Domain);
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm.shared_lut_reference.v1\0");
        hash.update((values.len() as u32).to_le_bytes());
        for value in values {
            hash.update(value.to_le_bytes());
        }
        Ok(Self {
            values: values.into(),
            digest: hash.finalize().into(),
        })
    }

    pub fn rows(&self) -> usize {
        self.values.len()
    }

    /// Secret coefficient payload only, for both parties; excludes public table,
    /// issuance metadata, transport and any input truncation.
    pub fn dealer_share_payload_bytes(&self) -> usize {
        2 * (2 + 4 * self.rows())
    }

    pub fn issue_reference(&self) -> Result<[Party; 2], Error> {
        let mut random = Zeroizing::new(vec![0u8; 4 + 4 * self.rows()]);
        getrandom::fill(&mut random).map_err(|_| Error::Randomness)?;
        let mut nonce = [0; 32];
        getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
        let mask = (self.rows() - 1) as u16;
        let r = u16::from_le_bytes([random[0], random[1]]) & mask;
        let r0 = u16::from_le_bytes([random[2], random[3]]) & mask;
        let mut left = Zeroizing::new(Vec::with_capacity(self.rows()));
        let mut right = Zeroizing::new(Vec::with_capacity(self.rows()));
        for (index, bytes) in random[4..].chunks_exact(4).enumerate() {
            let share = u32::from_le_bytes(bytes.try_into().unwrap());
            left.push(share);
            right.push(u32::from(index == usize::from(r)).wrapping_sub(share));
        }
        let make = |id, r, vector| Party {
            id,
            nonce,
            digest: self.digest,
            table: Arc::clone(&self.values),
            secret: Some(Secret {
                r: Zeroizing::new(r),
                vector,
            }),
        };
        Ok([make(0, r0, left), make(1, r.wrapping_sub(r0) & mask, right)])
    }
}

struct Secret {
    r: Zeroizing<u16>,
    vector: Zeroizing<Vec<u32>>,
}

/// No Clone, export or debug representation of party-local one-use material.
pub struct Party {
    id: u8,
    nonce: [u8; 32],
    digest: [u8; 32],
    table: Arc<[u32]>,
    secret: Option<Secret>,
}

pub struct Pending {
    id: u8,
    nonce: [u8; 32],
    digest: [u8; 32],
    table: Arc<[u32]>,
    delta: u16,
    secret: Secret,
}

impl Party {
    /// Burns the key before checking the input-share domain.
    pub fn start(&mut self, input_share: u16) -> Result<(Pending, [u8; OPENING_BYTES]), Error> {
        let secret = self.secret.take().ok_or(Error::Consumed)?;
        if usize::from(input_share) >= self.table.len() {
            return Err(Error::Domain);
        }
        let delta = input_share.wrapping_sub(*secret.r) & (self.table.len() - 1) as u16;
        let mut frame = [0; OPENING_BYTES];
        frame[0] = 1;
        frame[1] = self.id;
        frame[2..34].copy_from_slice(&self.nonce);
        frame[34..66].copy_from_slice(&self.digest);
        frame[66..].copy_from_slice(&delta.to_le_bytes());
        Ok((
            Pending {
                id: self.id,
                nonce: self.nonce,
                digest: self.digest,
                table: Arc::clone(&self.table),
                delta,
                secret,
            },
            frame,
        ))
    }
}

impl Pending {
    /// Consumes pending state even if peer framing/context validation fails.
    /// The returned word is one opaque output share, not a reconstructed value.
    pub fn finish(self, peer: &[u8]) -> Result<u32, Error> {
        if peer.len() != OPENING_BYTES
            || peer[0] != 1
            || peer[1] != 1 - self.id
            || peer[2..34] != self.nonce
            || peer[34..66] != self.digest
        {
            return Err(Error::Context);
        }
        let other = u16::from_le_bytes([peer[66], peer[67]]);
        if usize::from(other) >= self.table.len() {
            return Err(Error::Domain);
        }
        let mask = self.table.len() - 1;
        let shift = usize::from(self.delta.wrapping_add(other)) & mask;
        Ok(self
            .secret
            .vector
            .iter()
            .enumerate()
            .fold(0u32, |sum, (index, share)| {
                sum.wrapping_add(share.wrapping_mul(self.table[(index + shift) & mask]))
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_input_and_additive_share_wraparound() {
        let values = [0, u32::MAX, 42, 17, 1, 1 << 31, 98, 0];
        let table = Table::new(&values).unwrap();
        for x in 0..8u16 {
            for a in 0..8u16 {
                let [mut left, mut right] = table.issue_reference().unwrap();
                let (l, lf) = left.start(a).unwrap();
                let (r, rf) = right.start(x.wrapping_sub(a) & 7).unwrap();
                assert_eq!(
                    l.finish(&rf).unwrap().wrapping_add(r.finish(&lf).unwrap()),
                    values[x as usize]
                );
                assert!(matches!(left.start(a), Err(Error::Consumed)));
            }
        }
    }

    #[test]
    fn malformed_and_foreign_openings_consume_material() {
        let table = Table::new(&[5, 8, 10, 3]).unwrap();
        for field in [0, 1, 2, 34, 67] {
            let [mut left, mut right] = table.issue_reference().unwrap();
            let (l, _) = left.start(1).unwrap();
            let (_, mut rf) = right.start(2).unwrap();
            rf[field] ^= 128;
            assert!(l.finish(&rf).is_err());
            assert!(matches!(left.start(1), Err(Error::Consumed)));
        }
        let [mut left, _] = table.issue_reference().unwrap();
        assert!(matches!(left.start(4), Err(Error::Domain)));
        assert!(matches!(left.start(1), Err(Error::Consumed)));
        let [mut left, _] = table.issue_reference().unwrap();
        let [_, mut foreign] = table.issue_reference().unwrap();
        let (l, _) = left.start(1).unwrap();
        let (_, frame) = foreign.start(2).unwrap();
        assert!(matches!(l.finish(&frame), Err(Error::Context)));
        assert!(Table::new(&[1; 3]).is_err());
        assert!(Table::new(&[1; MAX_ROWS + 1]).is_err());
    }
}
