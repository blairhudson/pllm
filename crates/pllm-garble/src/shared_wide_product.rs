//! One-use 2^64 Beaver product for exact bounded fixed-point intermediates.
//! The trusted local dealer supplies independent triples; each Party owns one
//! triple share. Frames bind context and issuance, not transport authentication.
use crate::shared_circuit::random_word;
use crate::shared_rescale_reference::{Context, Error};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub const HEADER: usize = 78;
pub const FRAME_BYTES: usize = HEADER + 16;
pub const KEY_BYTES: usize = 24;
enum State {
    Ready(Zeroizing<[u64; 3]>),
    Waiting(Zeroizing<[u64; 3]>, Zeroizing<[u64; 2]>),
}
pub struct Party {
    id: u8,
    binding: [u8; 32],
    nonce: [u8; 32],
    state: Option<State>,
}

pub fn issue(context: Context) -> Result<[Party; 2], Error> {
    let mut hash = Sha256::new();
    hash.update(b"pllm.shared_wide_product.v1\0");
    hash.update(context.plan);
    hash.update(context.session);
    hash.update(context.operation);
    hash.update(context.first_tensor_index.to_le_bytes());
    let binding = hash.finalize().into();
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let a = Zeroizing::new(random_word()?);
    let b = Zeroizing::new(random_word()?);
    let left = Zeroizing::new([random_word()?, random_word()?, random_word()?]);
    let right = Zeroizing::new([
        a.wrapping_sub(left[0]),
        b.wrapping_sub(left[1]),
        a.wrapping_mul(*b).wrapping_sub(left[2]),
    ]);
    Ok([(0, left), (1, right)].map(|(id, t)| Party {
        id,
        binding,
        nonce,
        state: Some(State::Ready(t)),
    }))
}

impl Party {
    pub fn cancel(&mut self) {
        self.state = None;
    }
    pub fn key_bytes(&self) -> usize {
        if self.state.is_some() {
            KEY_BYTES
        } else {
            0
        }
    }
    pub fn start(&mut self, x: u64, y: u64) -> Result<Vec<u8>, Error> {
        let State::Ready(t) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        let own = Zeroizing::new([x.wrapping_sub(t[0]), y.wrapping_sub(t[1])]);
        let mut frame = Vec::with_capacity(FRAME_BYTES);
        frame.extend(b"PLLMMU64");
        frame.extend(self.binding);
        frame.extend(self.nonce);
        frame.extend([0, self.id]);
        frame.extend(1_u32.to_le_bytes());
        for v in own.iter() {
            frame.extend(v.to_le_bytes());
        }
        self.state = Some(State::Waiting(t, own));
        Ok(frame)
    }
    pub fn finish(&mut self, frame: &[u8]) -> Result<u64, Error> {
        let State::Waiting(t, own) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        if frame.len() != FRAME_BYTES
            || &frame[..8] != b"PLLMMU64"
            || frame[8..40] != self.binding
            || frame[40..72] != self.nonce
            || frame[72] != 0
            || frame[73] != 1 - self.id
            || frame[74..78] != 1_u32.to_le_bytes()
        {
            return Err(Error::Context);
        }
        let d = u64::from_le_bytes(frame[78..86].try_into().unwrap()).wrapping_add(own[0]);
        let e = u64::from_le_bytes(frame[86..94].try_into().unwrap()).wrapping_add(own[1]);
        Ok(t[2]
            .wrapping_add(d.wrapping_mul(t[1]))
            .wrapping_add(e.wrapping_mul(t[0]))
            .wrapping_add(if self.id == 0 { d.wrapping_mul(e) } else { 0 }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ctx() -> Context {
        Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: 0,
        }
    }
    #[test]
    fn full_ring_signed_products_one_use_and_foreign_frames() {
        for x in [0, 1, u64::MAX, 1 << 63, (1 << 58) - 1] {
            for y in [0, 1, u64::MAX, 1 << 63, 1 << 20] {
                let [mut a, mut b] = issue(ctx()).unwrap();
                let sx = random_word().unwrap();
                let sy = random_word().unwrap();
                let f = a.start(sx, sy).unwrap();
                let g = b.start(x.wrapping_sub(sx), y.wrapping_sub(sy)).unwrap();
                assert_eq!(
                    a.finish(&g).unwrap().wrapping_add(b.finish(&f).unwrap()),
                    (u128::from(x) * u128::from(y)) as u64
                );
                assert_eq!(a.key_bytes() + b.key_bytes(), 0);
                assert!(a.finish(&g).is_err());
            }
        }
        let [mut a, mut b] = issue(ctx()).unwrap();
        a.start(0, 0).unwrap();
        let mut f = b.start(0, 0).unwrap();
        f[8] ^= 1;
        assert!(a.finish(&f).is_err());
        assert_eq!(a.key_bytes(), 0);
        b.cancel();
        assert!(b.start(0, 0).is_err());
    }
}
