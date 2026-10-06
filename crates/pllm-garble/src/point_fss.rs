//! Private implementation of a bounded prefix-DPF comparison reference.
//!
//! A threshold is the disjoint union of at most `bits` binary prefixes. We
//! always issue all `bits` point keys, including zero-payload keys: neither
//! threshold bits nor zero thresholds change the public shape. This deliberately
//! simple O(bits^2) construction is NOT SIGMA's optimized comparison backend.

use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes128,
};
use zeroize::{Zeroize, Zeroizing};

use crate::shared_rescale_reference::Error;

#[derive(Clone)]
struct Correction {
    seed: [u8; 16],
    control: [u8; 2],
}

impl Drop for Correction {
    fn drop(&mut self) {
        self.seed.zeroize();
        self.control.zeroize();
    }
}

struct Point {
    root: Zeroizing<[u8; 16]>,
    words: Vec<Correction>,
    final_word: Zeroizing<u32>,
}

fn expand(seed: &[u8; 16]) -> ([[u8; 16]; 2], [u8; 2]) {
    let cipher = Aes128::new(seed.into());
    // Independent public counter inputs, with a full 128-bit child seed and
    // separate control bits. No seed bit is removed to store a control bit.
    let mut blocks = [0u8, 1, 2].map(|counter| {
        let mut block = *b"PLLM.FSS.PRG.v1\0";
        block[15] = counter;
        block.into()
    });
    cipher.encrypt_blocks(&mut blocks);
    let result = (
        [blocks[0].into(), blocks[1].into()],
        [blocks[2][0] & 1, blocks[2][1] & 1],
    );
    for block in &mut blocks {
        block.as_mut_slice().zeroize();
    }
    result
}

fn leaf(seed: &[u8; 16]) -> u32 {
    let cipher = Aes128::new(seed.into());
    let mut block = (*b"PLLM.FSS.PRG.v1\x03").into();
    cipher.encrypt_block(&mut block);
    let word = u32::from_le_bytes(block[..4].try_into().unwrap());
    block.as_mut_slice().zeroize();
    word
}

fn correct(seed: &mut [u8; 16], word: &[u8; 16], control: u8) {
    let mask = 0u8.wrapping_sub(control);
    for (value, delta) in seed.iter_mut().zip(word) {
        *value ^= delta & mask;
    }
}

impl Point {
    fn issue(bits: u8, point: u32, payload: u32) -> Result<[Self; 2], Error> {
        let mut roots = Zeroizing::new([[0; 16]; 2]);
        for seed in roots.iter_mut() {
            getrandom::fill(seed).map_err(|_| Error::Randomness)?;
        }
        let mut seeds = Zeroizing::new(*roots);
        let mut controls = Zeroizing::new([0u8, 1]);
        let mut words = Vec::with_capacity(usize::from(bits));
        for level in 0..bits {
            let keep = ((point >> (bits - level - 1)) & 1) as usize;
            let lose = 1 - keep;
            let mut children = Zeroizing::new([expand(&seeds[0]), expand(&seeds[1])]);
            let mut word = Correction {
                seed: [0; 16],
                control: [0; 2],
            };
            for (i, value) in word.seed.iter_mut().enumerate() {
                *value = children[0].0[lose][i] ^ children[1].0[lose][i];
            }
            word.control = [
                children[0].1[0] ^ children[1].1[0] ^ (1 - keep as u8),
                children[0].1[1] ^ children[1].1[1] ^ keep as u8,
            ];
            for party in 0..2 {
                correct(&mut children[party].0[keep], &word.seed, controls[party]);
                seeds[party] = children[party].0[keep];
                controls[party] = children[party].1[keep] ^ (controls[party] & word.control[keep]);
            }
            words.push(word);
        }
        let delta = payload
            .wrapping_sub(leaf(&seeds[0]))
            .wrapping_add(leaf(&seeds[1]));
        let final_word = if controls[0] == 1 {
            delta
        } else {
            delta.wrapping_neg()
        };
        Ok([
            Self {
                root: Zeroizing::new(roots[0]),
                words: words.clone(),
                final_word: Zeroizing::new(final_word),
            },
            Self {
                root: Zeroizing::new(roots[1]),
                words,
                final_word: Zeroizing::new(final_word),
            },
        ])
    }

    fn eval(&self, party: u8, point: u32) -> u32 {
        let mut seed = Zeroizing::new(*self.root);
        let mut control = party;
        for (level, word) in self.words.iter().enumerate() {
            let direction = ((point >> (self.words.len() - level - 1)) & 1) as usize;
            let (mut children, children_control) = expand(&seed);
            correct(&mut children[direction], &word.seed, control);
            *seed = children[direction];
            children.zeroize();
            control = children_control[direction] ^ (control & word.control[direction]);
        }
        let value = leaf(&seed).wrapping_add(u32::from(control).wrapping_mul(*self.final_word));
        if party == 0 {
            value
        } else {
            value.wrapping_neg()
        }
    }

    fn payload_bytes(&self) -> usize {
        20 + 18 * self.words.len()
    }
}

/// No public evaluation/export API: the owning one-use gate burns the batch.
pub(crate) struct Comparison {
    bits: u8,
    prefixes: Vec<Point>,
}

impl Comparison {
    pub(crate) fn bytes(bits: u8) -> usize {
        let n = usize::from(bits);
        20 * n + 9 * n * (n + 1)
    }

    pub(crate) fn issue(bits: u8, threshold: u32) -> Result<[Self; 2], Error> {
        if !(1..=32).contains(&bits) || u64::from(threshold) >= 1u64 << bits {
            return Err(Error::Domain);
        }
        let mut left = Vec::with_capacity(usize::from(bits));
        let mut right = Vec::with_capacity(usize::from(bits));
        for length in 1..=bits {
            let prefix = threshold >> (bits - length);
            let [a, b] = Point::issue(length, prefix & !1, prefix & 1)?;
            left.push(a);
            right.push(b);
        }
        Ok([
            Self {
                bits,
                prefixes: left,
            },
            Self {
                bits,
                prefixes: right,
            },
        ])
    }

    pub(crate) fn eval(&self, party: u8, input: u32) -> u32 {
        self.prefixes
            .iter()
            .enumerate()
            .fold(0u32, |sum, (index, key)| {
                sum.wrapping_add(key.eval(party, input >> (usize::from(self.bits) - index - 1)))
            })
    }

    pub(crate) fn payload_bytes(&self) -> usize {
        self.prefixes.iter().map(Point::payload_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_thresholds_include_zero_and_fixed_shape() {
        for bits in 1..=6 {
            for threshold in 0..1 << bits {
                let [left, right] = Comparison::issue(bits, threshold).unwrap();
                assert_eq!(left.payload_bytes(), Comparison::bytes(bits));
                assert_eq!(right.payload_bytes(), left.payload_bytes());
                for value in 0..1 << bits {
                    assert_eq!(
                        left.eval(0, value).wrapping_add(right.eval(1, value)),
                        u32::from(value < threshold)
                    );
                }
            }
        }
    }

    #[test]
    fn full_width_extreme_thresholds_and_zero_payload() {
        for threshold in [0, 1, 1 << 31, u32::MAX] {
            let [left, right] = Comparison::issue(32, threshold).unwrap();
            for value in [
                0,
                1,
                threshold.wrapping_sub(1),
                threshold,
                threshold.wrapping_add(1),
                u32::MAX,
            ] {
                assert_eq!(
                    left.eval(0, value).wrapping_add(right.eval(1, value)),
                    u32::from(value < threshold)
                );
            }
        }
        for payload in [0, 1, u32::MAX] {
            let [left, right] = Point::issue(4, 7, payload).unwrap();
            for point in 0..16 {
                assert_eq!(
                    left.eval(0, point).wrapping_add(right.eval(1, point)),
                    if point == 7 { payload } else { 0 }
                );
            }
        }
    }
}
