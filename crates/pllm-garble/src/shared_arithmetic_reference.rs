//! Bounded test-dealer Beaver multiplication over an exact power-of-two ring.
//! Each evaluator owns one triple share. No authenticated transport or malicious
//! security is supplied by this component-only reference.
use crate::shared_rescale_reference::{Context, Error};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub const HEADER_BYTES: usize = 78;
const MAGIC: &[u8; 8] = b"PLLMMU01";
pub const MAX_LANES: usize = 1024;

struct Triple(Zeroizing<Vec<[u32; 3]>>);
enum State {
    Ready(Triple),
    Waiting(Triple, Zeroizing<Vec<[u32; 2]>>),
}

pub struct Party {
    id: u8,
    mask: u32,
    lanes: usize,
    binding: [u8; 32],
    nonce: [u8; 32],
    state: Option<State>,
}

fn word() -> Result<u32, Error> {
    let mut raw = Zeroizing::new([0; 4]);
    getrandom::fill(&mut *raw).map_err(|_| Error::Randomness)?;
    Ok(u32::from_le_bytes(*raw))
}

pub fn issue_reference(bits: u8, lanes: usize, context: Context) -> Result<[Party; 2], Error> {
    if !(2..=32).contains(&bits) || !(1..=MAX_LANES).contains(&lanes) {
        return Err(Error::Resource);
    }
    context
        .first_tensor_index
        .checked_add(lanes as u64)
        .ok_or(Error::Domain)?;
    let mask = ((1u64 << bits) - 1) as u32;
    let mut hash = Sha256::new();
    hash.update(b"pllm.shared_arithmetic_reference.v1\0");
    hash.update([bits]);
    hash.update((lanes as u64).to_le_bytes());
    hash.update(context.plan);
    hash.update(context.session);
    hash.update(context.operation);
    hash.update(context.first_tensor_index.to_le_bytes());
    let binding = hash.finalize().into();
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let mut left = Zeroizing::new(Vec::with_capacity(lanes));
    let mut right = Zeroizing::new(Vec::with_capacity(lanes));
    for _ in 0..lanes {
        let a = Zeroizing::new(word()? & mask);
        let b = Zeroizing::new(word()? & mask);
        let c = Zeroizing::new(a.wrapping_mul(*b) & mask);
        let shares = Zeroizing::new([word()? & mask, word()? & mask, word()? & mask]);
        left.push(*shares);
        right.push([
            a.wrapping_sub(shares[0]) & mask,
            b.wrapping_sub(shares[1]) & mask,
            c.wrapping_sub(shares[2]) & mask,
        ]);
    }
    Ok([(0, left), (1, right)].map(|(id, triple)| Party {
        id,
        mask,
        lanes,
        binding,
        nonce,
        state: Some(State::Ready(Triple(triple))),
    }))
}

impl Party {
    pub fn cancel(&mut self) {
        self.state = None;
    }
    pub fn key_payload_bytes(&self) -> usize {
        if self.state.is_some() {
            12 * self.lanes
        } else {
            0
        }
    }

    pub fn start(&mut self, a: &[u32], b: &[u32]) -> Result<Vec<u8>, Error> {
        let state = self.state.take().ok_or(Error::Consumed)?;
        let State::Ready(triple) = state else {
            return Err(Error::Context);
        };
        if a.len() != self.lanes
            || b.len() != self.lanes
            || a.iter().chain(b).any(|&v| v > self.mask)
        {
            return Err(Error::Domain);
        }
        let own = Zeroizing::new(
            a.iter()
                .zip(b)
                .zip(triple.0.iter())
                .map(|((&a, &b), t)| {
                    [
                        a.wrapping_sub(t[0]) & self.mask,
                        b.wrapping_sub(t[1]) & self.mask,
                    ]
                })
                .collect::<Vec<_>>(),
        );
        let mut frame = Vec::with_capacity(HEADER_BYTES + 8 * self.lanes);
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&self.binding);
        frame.extend_from_slice(&self.nonce);
        frame.extend_from_slice(&[0, self.id]);
        frame.extend_from_slice(&(self.lanes as u32).to_le_bytes());
        for pair in own.iter() {
            for word in pair {
                frame.extend_from_slice(&word.to_le_bytes());
            }
        }
        self.state = Some(State::Waiting(triple, own));
        Ok(frame)
    }

    pub fn finish(&mut self, peer: &[u8]) -> Result<Zeroizing<Vec<u32>>, Error> {
        let state = self.state.take().ok_or(Error::Consumed)?;
        let State::Waiting(triple, own) = state else {
            return Err(Error::Context);
        };
        if peer.len() != HEADER_BYTES + 8 * self.lanes
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != self.nonce
            || peer[72] != 0
            || peer[73] != 1 - self.id
            || u32::from_le_bytes(peer[74..78].try_into().unwrap()) as usize != self.lanes
        {
            return Err(Error::Context);
        }
        let mut output = Zeroizing::new(Vec::with_capacity(self.lanes));
        for ((raw, own), triple) in peer[HEADER_BYTES..]
            .chunks_exact(8)
            .zip(own.iter())
            .zip(triple.0.iter())
        {
            let d = u32::from_le_bytes(raw[..4].try_into().unwrap());
            let e = u32::from_le_bytes(raw[4..].try_into().unwrap());
            if d > self.mask || e > self.mask {
                return Err(Error::Domain);
            }
            let d = d.wrapping_add(own[0]) & self.mask;
            let e = e.wrapping_add(own[1]) & self.mask;
            output.push(
                triple[2]
                    .wrapping_add(d.wrapping_mul(triple[1]))
                    .wrapping_add(e.wrapping_mul(triple[0]))
                    .wrapping_add(if self.id == 0 { d.wrapping_mul(e) } else { 0 })
                    & self.mask,
            );
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn context() -> Context {
        Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: 0,
        }
    }
    #[test]
    fn exact_ring_products_and_one_use() {
        for bits in [2, 8, 24, 32] {
            let mask = ((1u64 << bits) - 1) as u32;
            for x in [0, 1, mask, mask / 2] {
                for y in [0, 1, mask, mask / 2] {
                    let [mut a, mut b] = issue_reference(bits, 1, context()).unwrap();
                    let left = a.start(&[mask], &[mask]).unwrap();
                    let right = b
                        .start(
                            &[x.wrapping_sub(mask) & mask],
                            &[y.wrapping_sub(mask) & mask],
                        )
                        .unwrap();
                    let aa = a.finish(&right).unwrap();
                    let bb = b.finish(&left).unwrap();
                    assert_eq!(aa[0].wrapping_add(bb[0]) & mask, x.wrapping_mul(y) & mask);
                    assert_eq!(a.key_payload_bytes(), 0);
                    assert_eq!(a.finish(&right), Err(Error::Consumed));
                }
            }
        }
    }
    #[test]
    fn malformed_domain_context_replay_and_cancel_burn() {
        for offset in [0, 8, 40, 72, 73, 74, 81] {
            let [mut a, mut b] = issue_reference(24, 1, context()).unwrap();
            a.start(&[0], &[0]).unwrap();
            let mut frame = b.start(&[0], &[0]).unwrap();
            frame[offset] ^= 128;
            assert!(a.finish(&frame).is_err());
            assert_eq!(a.key_payload_bytes(), 0);
        }
        let [mut a, mut b] = issue_reference(24, 1, context()).unwrap();
        assert!(a.start(&[1 << 24], &[0]).is_err());
        assert_eq!(a.key_payload_bytes(), 0);
        b.cancel();
        assert!(b.start(&[0], &[0]).is_err());
        assert!(issue_reference(24, MAX_LANES + 1, context()).is_err());
    }
}
