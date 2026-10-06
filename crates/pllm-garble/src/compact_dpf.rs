//! Independent BGI early-terminated Boolean DPF and Grotto parity comparison.
//! BGI 2016 Figure 1 / §3.2.3; Grotto §5–6; SIGMA §2.5–2.6.
//! Seeds retain 128 secret bits; control bits are separate (SIGMA uses 127).
//! Internal reusable key algebra, never a public reusable protocol handle.

use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use zeroize::{Zeroize, Zeroizing};

pub(crate) struct Bits {
    pub data: Zeroizing<Vec<u8>>,
    pub len: usize,
}

impl Bits {
    pub fn new() -> Self {
        Self {
            data: Zeroizing::new(Vec::new()),
            len: 0,
        }
    }
    pub fn put(&mut self, value: u128, width: usize) {
        for i in 0..width {
            if self.len % 8 == 0 {
                self.data.push(0);
            }
            self.data[self.len / 8] |= (((value >> i) & 1) as u8) << (self.len % 8);
            self.len += 1;
        }
    }
}

struct Correction {
    seed: u128,
    control: [u8; 2],
}
impl Drop for Correction {
    fn drop(&mut self) {
        self.seed.zeroize();
        self.control.zeroize();
    }
}

pub(crate) struct Key {
    bits: u8,
    root: u128,
    words: Vec<Correction>,
    leaf: u128,
}
impl Drop for Key {
    fn drop(&mut self) {
        self.root.zeroize();
        self.leaf.zeroize();
    }
}

fn random() -> Result<u128, getrandom::Error> {
    let mut b = Zeroizing::new([0; 16]);
    getrandom::fill(&mut *b)?;
    Ok(u128::from_le_bytes(*b))
}

// AES-CTR under the seed. Distinct counter 2 supplies control bits without
// taking bits out of the secret seed. This CPU reference recomputes the flags:
// two child calls use 4 AES invocations; one child call uses 2.
fn child(seed: u128, branch: usize) -> (u128, u8) {
    let cipher = Aes128::new(&seed.to_le_bytes().into());
    let mut block = (branch as u128).to_le_bytes().into();
    let mut flags = 2u128.to_le_bytes().into();
    cipher.encrypt_block(&mut block);
    cipher.encrypt_block(&mut flags);
    (u128::from_le_bytes(block.into()), (flags[0] >> branch) & 1)
}

impl Key {
    pub fn encoded_bits(bits: u8) -> usize {
        if bits <= 7 {
            1usize << bits
        } else {
            256 + 130 * usize::from(bits - 7)
        }
    }

    pub fn issue(bits: u8, point: u64) -> Result<[Self; 2], getrandom::Error> {
        assert!((1..=63).contains(&bits) && point < 1u64 << bits);
        if bits <= 7 {
            let mask = if bits == 7 {
                u128::MAX
            } else {
                (1u128 << (1 << bits)) - 1
            };
            let left = random()? & mask;
            return Ok([left, left ^ (1u128 << point)].map(|leaf| Self {
                bits,
                root: 0,
                words: vec![],
                leaf,
            }));
        }
        let roots = Zeroizing::new([random()?, random()?]);
        let mut seeds = Zeroizing::new(*roots);
        let mut controls = [0u8, 1];
        let mut words = [Vec::new(), Vec::new()];
        for level in (7..bits).rev() {
            let keep = ((point >> level) & 1) as usize;
            let a = [child(seeds[0], 0), child(seeds[0], 1)];
            let b = [child(seeds[1], 0), child(seeds[1], 1)];
            let seed = a[1 - keep].0 ^ b[1 - keep].0;
            let cw = [
                a[0].1 ^ b[0].1 ^ keep as u8 ^ 1,
                a[1].1 ^ b[1].1 ^ keep as u8,
            ];
            for party in 0..2 {
                let expanded = if party == 0 { a } else { b };
                let flag = u128::from(controls[party]);
                seeds[party] = expanded[keep].0 ^ (seed & 0u128.wrapping_sub(flag));
                controls[party] = expanded[keep].1 ^ (cw[keep] & flag as u8);
                words[party].push(Correction { seed, control: cw });
            }
        }
        // Identity Convert into 128 Boolean output bits; no seed entropy is
        // discarded. Matching off-path seeds/control cancel in both keys.
        let leaf = seeds[0] ^ seeds[1] ^ (1u128 << (point & 127));
        let [left, right] = words;
        Ok([
            Self {
                bits,
                root: roots[0],
                words: left,
                leaf,
            },
            Self {
                bits,
                root: roots[1],
                words: right,
                leaf,
            },
        ])
    }

    fn descend(&self, level: usize, seed: u128, flag: u8, branch: usize) -> (u128, u8) {
        let (next, control) = child(seed, branch);
        let cw = &self.words[level];
        (
            next ^ (cw.seed & 0u128.wrapping_sub(u128::from(flag))),
            control ^ (flag & cw.control[branch]),
        )
    }

    pub fn point(&self, party: u8, input: u64) -> u8 {
        assert!(party <= 1 && input < 1u64 << self.bits);
        if self.bits <= 7 {
            return ((self.leaf >> input) & 1) as u8;
        }
        let (mut seed, mut flag) = (self.root, party);
        for i in 0..self.words.len() {
            (seed, flag) = self.descend(
                i,
                seed,
                flag,
                ((input >> (self.bits as usize - 1 - i)) & 1) as usize,
            );
        }
        (((seed ^ (self.leaf & 0u128.wrapping_sub(u128::from(flag)))) >> (input & 127)) & 1) as u8
    }

    /// XOR shares of input < point. Grotto suffix parity with one traversed
    /// child per level: a right-subtree parity is parent XOR left-child.
    pub fn less(&self, party: u8, input: u64) -> u8 {
        assert!(party <= 1 && input < 1u64 << self.bits);
        let (mut seed, mut flag, mut parity) = (self.root, party, 0);
        if self.bits > 7 {
            for i in 0..self.words.len() {
                let branch = ((input >> (self.bits as usize - 1 - i)) & 1) as usize;
                let (next, control) = self.descend(i, seed, flag, branch);
                parity ^= (flag ^ control) & (1 ^ branch as u8);
                (seed, flag) = (next, control);
            }
        }
        let leaf = if self.bits <= 7 {
            self.leaf
        } else {
            seed ^ (self.leaf & 0u128.wrapping_sub(u128::from(flag)))
        };
        let shift = (input & 127) + 1;
        parity
            ^ if shift == 128 {
                0
            } else {
                ((leaf >> shift).count_ones() & 1) as u8
            }
    }

    /// Bounded full-domain expansion, one visit per tree edge. Lookup parties
    /// perform all public table work; there is no secret-indexed memory query.
    pub fn all(&self, party: u8) -> Zeroizing<Vec<u8>> {
        assert!(self.bits <= 10 && party <= 1);
        fn walk(key: &Key, depth: usize, seed: u128, flag: u8, out: &mut Vec<u8>) {
            if depth == key.words.len() {
                let leaf = seed ^ (key.leaf & 0u128.wrapping_sub(u128::from(flag)));
                out.extend((0..128).map(|i| ((leaf >> i) & 1) as u8));
            } else {
                for branch in 0..2 {
                    let (s, t) = key.descend(depth, seed, flag, branch);
                    walk(key, depth + 1, s, t, out);
                }
            }
        }
        let mut out = Zeroizing::new(Vec::with_capacity(1 << self.bits));
        if self.bits <= 7 {
            out.extend((0..1 << self.bits).map(|i| ((self.leaf >> i) & 1) as u8));
        } else {
            walk(self, 0, self.root, party, &mut out);
        }
        out
    }

    pub fn encode(&self, output: &mut Bits) {
        if self.bits <= 7 {
            output.put(self.leaf, 1 << self.bits);
        } else {
            output.put(self.root, 128);
            for cw in &self.words {
                output.put(cw.seed, 128);
                output.put(u128::from(cw.control[0]), 1);
                output.put(u128::from(cw.control[1]), 1);
            }
            output.put(self.leaf, 128);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_small_points_and_parities_and_tree_boundaries() {
        for bits in 1..=10 {
            let points: Vec<_> = if bits <= 7 {
                (0..1u64 << bits).collect()
            } else {
                vec![0, 1, 126, 127, 128, 129, (1 << bits) - 1]
            };
            for point in points {
                let keys = Key::issue(bits, point).unwrap();
                let all = [keys[0].all(0), keys[1].all(1)];
                for x in 0..1 << bits {
                    assert_eq!(
                        keys[0].point(0, x) ^ keys[1].point(1, x),
                        u8::from(x == point)
                    );
                    assert_eq!(keys[0].less(0, x) ^ keys[1].less(1, x), u8::from(x < point));
                    for b in 0..2 {
                        assert_eq!(all[b][x as usize], keys[b].point(b as u8, x));
                    }
                }
                let mut encoded = Bits::new();
                keys[0].encode(&mut encoded);
                assert_eq!(encoded.len, Key::encoded_bits(bits));
            }
        }
    }

    #[test]
    fn wide_comparisons_do_not_expand_the_domain() {
        for bits in [12, 25, 32, 40, 63] {
            for p in [
                0,
                1,
                (1 << (bits - 1)) - 1,
                1 << (bits - 1),
                (1 << bits) - 1,
            ] {
                let k = Key::issue(bits, p).unwrap();
                for x in [
                    0,
                    p.saturating_sub(1),
                    p,
                    (p + 1) & ((1 << bits) - 1),
                    (1 << bits) - 1,
                ] {
                    assert_eq!(k[0].less(0, x) ^ k[1].less(1, x), u8::from(x < p));
                    assert_eq!(k[0].point(0, x) ^ k[1].point(1, x), u8::from(x == p));
                }
            }
        }
    }
}
