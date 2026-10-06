//! Bounded distributed comparison from Boyle et al., FSS for Mixed-Mode Secure
//! Computation, ePrint 2020/1392, Figure 1. Independently implemented from the
//! published equations over the additive group Z/(2^32), with 128-bit seeds.
//!
//! One correction per input bit replaces the quadratic prefix-DPF reference.
//! Control bits occupy separate bytes; no seed entropy is used as a control bit.
//! Keys remain private to the owning one-use gate. This implementation has not
//! received independent cryptographic or side-channel review.

use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes128,
};
use zeroize::{Zeroize, Zeroizing};

use crate::shared_rescale_reference::Error;

#[derive(Clone)]
struct Correction {
    seed: [u8; 16],
    value: u32,
    control: [u8; 2],
}

impl Drop for Correction {
    fn drop(&mut self) {
        self.seed.zeroize();
        self.value.zeroize();
        self.control.zeroize();
    }
}

/// No cloning, public evaluation, or export interface. The gate owns its keys.
pub(crate) struct Comparison {
    root: Zeroizing<[u8; 16]>,
    words: Vec<Correction>,
    final_word: Zeroizing<u32>,
}

fn convert(seed: &[u8; 16]) -> u32 {
    // Figure 1's Convert_G is truncation when |G| is a power of two <= 2^128.
    u32::from_le_bytes(seed[..4].try_into().unwrap())
}

fn expand(seed: &[u8; 16]) -> ([[u8; 16]; 2], [u32; 2], [u8; 2]) {
    let cipher = Aes128::new(seed.into());
    let mut blocks = [0u8, 1, 2, 3, 4].map(|counter| {
        let mut block = *b"PLLM.DCF.PRG.v1\0";
        block[15] = counter;
        block.into()
    });
    cipher.encrypt_blocks(&mut blocks);
    // Four independent full-width outputs provide the two child seeds and the
    // two value seeds. A fifth supplies only controls. Convert discards unused
    // value-seed bits after expansion, exactly as permitted for this group.
    let result = (
        [blocks[0].into(), blocks[1].into()],
        [
            u32::from_le_bytes(blocks[2][..4].try_into().unwrap()),
            u32::from_le_bytes(blocks[3][..4].try_into().unwrap()),
        ],
        [blocks[4][0] & 1, blocks[4][1] & 1],
    );
    for block in &mut blocks {
        block.as_mut_slice().zeroize();
    }
    result
}

fn signed(value: u32, negative: u8) -> u32 {
    value.wrapping_mul(1u32.wrapping_sub(2 * u32::from(negative)))
}

fn correct(seed: &mut [u8; 16], word: &[u8; 16], control: u8) {
    let mask = 0u8.wrapping_sub(control);
    for (value, delta) in seed.iter_mut().zip(word) {
        *value ^= delta & mask;
    }
}

impl Comparison {
    pub(crate) fn bytes(bits: u8) -> usize {
        // Root + final group word + (seed + group word + two control bytes)/bit.
        20 + 22 * usize::from(bits)
    }

    pub(crate) fn issue(bits: u8, threshold: u32) -> Result<[Self; 2], Error> {
        Self::issue_payload(bits, threshold, 1)
    }

    fn issue_payload(bits: u8, threshold: u32, payload: u32) -> Result<[Self; 2], Error> {
        if !(1..=32).contains(&bits) || u64::from(threshold) >= 1u64 << bits {
            return Err(Error::Domain);
        }
        let mut roots = Zeroizing::new([[0; 16]; 2]);
        for root in roots.iter_mut() {
            getrandom::fill(root).map_err(|_| Error::Randomness)?;
        }
        let mut seeds = Zeroizing::new(*roots);
        let mut controls = Zeroizing::new([0u8, 1]);
        let mut path_value = Zeroizing::new(0u32);
        let mut words = Vec::with_capacity(usize::from(bits));
        for level in 0..bits {
            let keep = ((threshold >> (bits - level - 1)) & 1) as usize;
            let lose = 1 - keep;
            let mut children = Zeroizing::new([expand(&seeds[0]), expand(&seeds[1])]);
            let mut word = Correction {
                seed: [0; 16],
                value: signed(
                    children[1].1[lose]
                        .wrapping_sub(children[0].1[lose])
                        .wrapping_sub(*path_value)
                        .wrapping_add(payload.wrapping_mul(keep as u32)),
                    controls[1],
                ),
                control: [
                    children[0].2[0] ^ children[1].2[0] ^ (1 - keep as u8),
                    children[0].2[1] ^ children[1].2[1] ^ keep as u8,
                ],
            };
            for (i, value) in word.seed.iter_mut().enumerate() {
                *value = children[0].0[lose][i] ^ children[1].0[lose][i];
            }
            *path_value = path_value
                .wrapping_add(children[0].1[keep])
                .wrapping_sub(children[1].1[keep])
                .wrapping_add(signed(word.value, controls[1]));
            for party in 0..2 {
                correct(&mut children[party].0[keep], &word.seed, controls[party]);
                seeds[party] = children[party].0[keep];
                controls[party] = children[party].2[keep] ^ (controls[party] & word.control[keep]);
            }
            words.push(word);
        }
        let final_word = signed(
            convert(&seeds[1])
                .wrapping_sub(convert(&seeds[0]))
                .wrapping_sub(*path_value),
            controls[1],
        );
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

    pub(crate) fn eval(&self, party: u8, input: u32) -> u32 {
        let mut seed = Zeroizing::new(*self.root);
        let mut control = Zeroizing::new(party);
        let mut sum = Zeroizing::new(0u32);
        for (level, word) in self.words.iter().enumerate() {
            let direction = ((input >> (self.words.len() - level - 1)) & 1) as usize;
            let mut children = Zeroizing::new(expand(&seed));
            *sum = sum.wrapping_add(signed(
                children.1[direction].wrapping_add(u32::from(*control).wrapping_mul(word.value)),
                party,
            ));
            correct(&mut children.0[direction], &word.seed, *control);
            *seed = children.0[direction];
            *control = children.2[direction] ^ (*control & word.control[direction]);
        }
        sum.wrapping_add(signed(
            convert(&seed).wrapping_add(u32::from(*control).wrapping_mul(*self.final_word)),
            party,
        ))
    }

    pub(crate) fn payload_bytes(&self) -> usize {
        Self::bytes(self.words.len() as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_thresholds_payloads_and_fixed_key_shape() {
        for bits in 1..=6 {
            for threshold in 0..1 << bits {
                for payload in [0, 1, u32::MAX, 0x80000000] {
                    let [left, right] =
                        Comparison::issue_payload(bits, threshold, payload).unwrap();
                    assert_eq!(left.payload_bytes(), 20 + 22 * usize::from(bits));
                    assert_eq!(right.payload_bytes(), left.payload_bytes());
                    for x in 0..1 << bits {
                        assert_eq!(
                            left.eval(0, x).wrapping_add(right.eval(1, x)),
                            if x < threshold { payload } else { 0 },
                            "{bits}-bit threshold {threshold}, input {x}, payload {payload}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn full_width_boundaries_and_fresh_roots() {
        for bits in [16, 24, 32] {
            let max = ((1u64 << bits) - 1) as u32;
            for threshold in [0, 1, 1 << (bits - 1), max] {
                let [left, right] = Comparison::issue(bits, threshold).unwrap();
                let [again, _] = Comparison::issue(bits, threshold).unwrap();
                assert_ne!(*left.root, *right.root);
                assert_ne!(*left.root, *again.root);
                for x in [
                    0,
                    1,
                    threshold.wrapping_sub(1),
                    threshold,
                    threshold.wrapping_add(1),
                    max,
                ] {
                    let x = x & max;
                    assert_eq!(
                        left.eval(0, x).wrapping_add(right.eval(1, x)),
                        u32::from(x < threshold)
                    );
                }
            }
        }
        for (bits, threshold) in [(0, 0), (33, 0), (1, 2), (31, u32::MAX)] {
            assert!(matches!(
                Comparison::issue(bits, threshold),
                Err(Error::Domain)
            ));
        }
    }
}
