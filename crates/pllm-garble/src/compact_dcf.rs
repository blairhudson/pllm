//! Bounded distributed comparison from Boyle et al., FSS for Mixed-Mode Secure
//! Computation, ePrint 2020/1392, Figure 1. Independently implemented from the
//! published equations over the additive group Z/(2^32), with 128-bit seeds.
//! RingComparison extends input/output widths independently through 50 bits,
//! using the same PRG and correction algebra and checked u64 ring words.
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

struct Correction<W: Word> {
    seed: [u8; 16],
    value: W,
    control: [u8; 2],
}

impl<W: Word> Drop for Correction<W> {
    fn drop(&mut self) {
        self.seed.zeroize();
        self.value.zeroize();
        self.control.zeroize();
    }
}

/// No cloning, public evaluation, or export interface. The gate owns its keys.
pub(crate) struct Key<W: Word> {
    root: Zeroizing<[u8; 16]>,
    words: Vec<Correction<W>>,
    final_word: Zeroizing<W>,
}

pub(crate) type Comparison = Key<u32>;

// Scalar and vector layouts use one PRG, correction and traversal algorithm.
// Group operations are coordinatewise modulo the public power-of-two mask.
// Scalar Convert and the PRG domain preserve historical u32/u64 identities.
pub(crate) trait Word: Copy + Zeroize {
    fn zero() -> Self;
    fn convert(seed: &[u8], mask: u64) -> Self;
    fn add(self, other: Self, mask: u64) -> Self;
    fn sub(self, other: Self, mask: u64) -> Self;
    fn scale(self, value: u64, mask: u64) -> Self;
}
macro_rules! scalar_word {
    ($word:ty) => {
        impl Word for $word {
            fn zero() -> Self {
                0
            }
            fn convert(seed: &[u8], mask: u64) -> Self {
                (u64::from_le_bytes(seed[..8].try_into().unwrap()) & mask) as Self
            }
            fn add(self, other: Self, mask: u64) -> Self {
                ((self as u64).wrapping_add(other as u64) & mask) as Self
            }
            fn sub(self, other: Self, mask: u64) -> Self {
                ((self as u64).wrapping_sub(other as u64) & mask) as Self
            }
            fn scale(self, value: u64, mask: u64) -> Self {
                ((self as u64).wrapping_mul(value) & mask) as Self
            }
        }
    };
}
scalar_word!(u32);
scalar_word!(u64);
impl Word for [u64; 2] {
    fn zero() -> Self {
        [0; 2]
    }
    fn convert(seed: &[u8], mask: u64) -> Self {
        std::array::from_fn(|j| {
            u64::from_le_bytes(seed[8 * j..8 * (j + 1)].try_into().unwrap()) & mask
        })
    }
    fn add(self, other: Self, mask: u64) -> Self {
        std::array::from_fn(|j| self[j].wrapping_add(other[j]) & mask)
    }
    fn sub(self, other: Self, mask: u64) -> Self {
        std::array::from_fn(|j| self[j].wrapping_sub(other[j]) & mask)
    }
    fn scale(self, value: u64, mask: u64) -> Self {
        self.map(|word| word.wrapping_mul(value) & mask)
    }
}

fn convert<W: Word>(seed: &[u8], mask: u64) -> W {
    W::convert(seed, mask)
}

fn expand<W: Word>(seed: &[u8; 16], mask: u64) -> ([[u8; 16]; 2], [W; 2], [u8; 2]) {
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
            convert::<W>(blocks[2].as_slice(), mask),
            convert::<W>(blocks[3].as_slice(), mask),
        ],
        [blocks[4][0] & 1, blocks[4][1] & 1],
    );
    for block in &mut blocks {
        block.as_mut_slice().zeroize();
    }
    result
}

fn signed<W: Word>(value: W, negative: u8, mask: u64) -> W {
    value.scale(1u64.wrapping_sub(2 * u64::from(negative)), mask)
}

fn correct(seed: &mut [u8; 16], word: &[u8; 16], control: u8) {
    let mask = 0u8.wrapping_sub(control);
    for (value, delta) in seed.iter_mut().zip(word) {
        *value ^= delta & mask;
    }
}

impl Key<u32> {
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
        generate(
            bits,
            u64::from(threshold),
            payload,
            u64::from(u32::MAX),
            std::mem::size_of::<Self>(),
            usize::MAX,
            |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
        )
    }

    pub(crate) fn eval(&self, party: u8, input: u32) -> u32 {
        evaluate(self, party, u64::from(input), u64::from(u32::MAX))
    }

    pub(crate) fn payload_bytes(&self) -> usize {
        Self::bytes(self.words.len() as u8)
    }
}

/// Logical source-managed storage, not RSS or allocator metadata. Encoded bytes
/// are a layout count only: this primitive deliberately has no raw key codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Resources {
    pub encoded_key_bytes: usize,
    pub key_owner_bytes: usize,
    pub key_capacity_bytes: usize,
    pub issuance_scratch_bytes: usize,
    pub combined_allocation_bytes: usize,
}

fn resources<W: Word>(bits: u8, owner_bytes: usize, capacities: [usize; 2]) -> Resources {
    let size = std::mem::size_of::<W>();
    // Conservative logical stack-owner budget: roots, evolving seeds/controls,
    // threshold/payload/path, paired children and one expansion return/correction,
    // final word, five AES output blocks, the AES key schedule, return staging.
    // Backend-internal temporary registers/stack, compiler spill/copy choices and
    // allocator bookkeeping are not a claim about measured process peak/RSS.
    let scratch = 64
        + 2
        + 8
        + 2 * size
        + 3 * std::mem::size_of::<([[u8; 16]; 2], [W; 2], [u8; 2])>()
        + std::mem::size_of::<Correction<W>>()
        + size
        + 80
        + std::mem::size_of::<Aes128>()
        + 2 * owner_bytes;
    let capacity = capacities.map(|n| n * std::mem::size_of::<Correction<W>>());
    Resources {
        encoded_key_bytes: 16 + size + usize::from(bits) * (18 + size),
        key_owner_bytes: owner_bytes,
        key_capacity_bytes: capacity[0],
        issuance_scratch_bytes: scratch,
        combined_allocation_bytes: 2 * owner_bytes + capacity[0] + capacity[1] + scratch,
    }
}

fn generate<W: Word>(
    bits: u8,
    threshold: u64,
    payload: W,
    mask: u64,
    owner_bytes: usize,
    limit: usize,
    random: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<[Key<W>; 2], Error> {
    generate_with_reservation(
        bits,
        threshold,
        payload,
        mask,
        owner_bytes,
        limit,
        |_| Ok(()),
        random,
    )
}

// Common bounded reservation seam: callback runs before each of the two exact
// capacity allocations, and all allocation/admission completes before entropy.
// The caller owns any longer-lived reservation; this primitive owns no lease.
#[allow(clippy::too_many_arguments)]
fn generate_with_reservation<W: Word>(
    bits: u8,
    threshold: u64,
    payload: W,
    mask: u64,
    owner_bytes: usize,
    limit: usize,
    mut before_reserve: impl FnMut(usize) -> Result<(), Error>,
    mut random: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<[Key<W>; 2], Error> {
    let threshold = Zeroizing::new(threshold);
    let payload = Zeroizing::new(payload);
    if resources::<W>(bits, owner_bytes, [usize::from(bits); 2]).combined_allocation_bytes > limit {
        return Err(Error::Resource);
    }
    // Reserve both complete correction owners before requesting any entropy.
    // Neither a key nor a correction vector is Clone.
    let mut words: [Vec<Correction<W>>; 2] = [Vec::new(), Vec::new()];
    for party in &mut words {
        before_reserve(usize::from(bits) * std::mem::size_of::<Correction<W>>())?;
        party
            .try_reserve_exact(usize::from(bits))
            .map_err(|_| Error::Resource)?;
    }
    if resources::<W>(bits, owner_bytes, words.each_ref().map(|w| w.capacity()))
        .combined_allocation_bytes
        > limit
    {
        return Err(Error::Resource);
    }
    let mut roots = Zeroizing::new([[0; 16]; 2]);
    for root in roots.iter_mut() {
        random(root)?;
    }
    let mut seeds = Zeroizing::new(*roots);
    let mut controls = Zeroizing::new([0u8, 1]);
    let mut path_value = Zeroizing::new(W::zero());
    for level in 0..bits {
        let keep = ((*threshold >> (bits - level - 1)) & 1) as usize;
        let lose = 1 - keep;
        let mut children =
            Zeroizing::new([expand::<W>(&seeds[0], mask), expand::<W>(&seeds[1], mask)]);
        let mut word = Correction {
            seed: [0; 16],
            value: signed(
                children[1].1[lose]
                    .sub(children[0].1[lose], mask)
                    .sub(*path_value, mask)
                    .add(payload.scale(keep as u64, mask), mask),
                controls[1],
                mask,
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
            .add(children[0].1[keep], mask)
            .sub(children[1].1[keep], mask)
            .add(signed(word.value, controls[1], mask), mask);
        for party in 0..2 {
            correct(&mut children[party].0[keep], &word.seed, controls[party]);
            seeds[party] = children[party].0[keep];
            controls[party] = children[party].2[keep] ^ (controls[party] & word.control[keep]);
        }
        words[0].push(Correction {
            seed: word.seed,
            value: word.value,
            control: word.control,
        });
        words[1].push(word);
    }
    let final_word = Zeroizing::new(signed(
        convert::<W>(&seeds[1], mask)
            .sub(convert::<W>(&seeds[0], mask), mask)
            .sub(*path_value, mask),
        controls[1],
        mask,
    ));
    let [left, right] = words;
    Ok([
        Key {
            root: Zeroizing::new(roots[0]),
            words: left,
            final_word: Zeroizing::new(*final_word),
        },
        Key {
            root: Zeroizing::new(roots[1]),
            words: right,
            final_word: Zeroizing::new(*final_word),
        },
    ])
}

fn evaluate<W: Word>(key: &Key<W>, party: u8, input: u64, mask: u64) -> W {
    let mut seed = Zeroizing::new(*key.root);
    let mut control = Zeroizing::new(party);
    let mut sum = Zeroizing::new(W::zero());
    for (level, word) in key.words.iter().enumerate() {
        let direction = ((input >> (key.words.len() - level - 1)) & 1) as usize;
        let mut children = Zeroizing::new(expand::<W>(&seed, mask));
        *sum = sum.add(
            signed(
                children.1[direction].add(word.value.scale(u64::from(*control), mask), mask),
                party,
                mask,
            ),
            mask,
        );
        correct(&mut children.0[direction], &word.seed, *control);
        *seed = children.0[direction];
        *control = children.2[direction] ^ (*control & word.control[direction]);
    }
    sum.add(
        signed(
            convert::<W>(&*seed, mask).add(key.final_word.scale(u64::from(*control), mask), mask),
            party,
            mask,
        ),
        mask,
    )
}

/// Checked payload-hiding scalar DCF. Owner must be private to a consuming gate;
/// this internal algebra is not itself a one-use online protocol or raw codec.
pub(crate) struct RingComparison {
    key: Key<u64>,
    input_bits: u8,
    output_bits: u8,
}
impl RingComparison {
    pub(crate) fn resources(input_bits: u8, output_bits: u8) -> Result<Resources, Error> {
        Self::validate(input_bits, output_bits)?;
        Ok(resources::<u64>(
            input_bits,
            std::mem::size_of::<Self>(),
            [usize::from(input_bits); 2],
        ))
    }
    fn validate(input_bits: u8, output_bits: u8) -> Result<(), Error> {
        if !(1..=50).contains(&input_bits) || !(1..=50).contains(&output_bits) {
            return Err(Error::Domain);
        }
        Ok(())
    }
    pub(crate) fn issue(
        input_bits: u8,
        output_bits: u8,
        threshold: u64,
        payload: u64,
        allocation_limit: usize,
    ) -> Result<[Self; 2], Error> {
        Self::issue_with(
            input_bits,
            output_bits,
            threshold,
            payload,
            allocation_limit,
            |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
        )
    }
    fn issue_with(
        input_bits: u8,
        output_bits: u8,
        threshold: u64,
        payload: u64,
        allocation_limit: usize,
        random: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<[Self; 2], Error> {
        let cost = Self::resources(input_bits, output_bits)?;
        if threshold >= 1u64 << input_bits || payload >= 1u64 << output_bits {
            return Err(Error::Domain);
        }
        if cost.combined_allocation_bytes > allocation_limit {
            return Err(Error::Resource);
        }
        // Recheck actual reserved capacities and full wrapper storage before RNG.
        let keys = generate(
            input_bits,
            threshold,
            payload,
            (1u64 << output_bits) - 1,
            std::mem::size_of::<Self>(),
            allocation_limit,
            random,
        )?;
        Ok(keys.map(|key| Self {
            key,
            input_bits,
            output_bits,
        }))
    }
    pub(crate) fn eval(&self, party: u8, input: u64) -> Result<u64, Error> {
        if party > 1 || input >= 1u64 << self.input_bits {
            return Err(Error::Domain);
        }
        Ok(evaluate(
            &self.key,
            party,
            input,
            (1u64 << self.output_bits) - 1,
        ))
    }
    pub(crate) fn allocation(&self) -> Resources {
        resources::<u64>(
            self.input_bits,
            std::mem::size_of::<Self>(),
            [self.key.words.capacity(); 2],
        )
    }
}

/// Private two-coordinate payload DCF over (Z/2^output_bits)^2. This wraps the
/// same Key/Correction/PRG/generation/traversal as both scalar variants. One
/// vector key has one root and one correction path, NOT two scalar DCF keys.
/// Conversion uses both 64-bit halves of each 128-bit value seed; at most 100
/// output bits are retained. Width metadata is public and bound into this owner.
pub(crate) struct VectorComparison {
    key: Key<[u64; 2]>,
    input_bits: u8,
    output_bits: u8,
}
impl VectorComparison {
    pub(crate) fn resources(input_bits: u8, output_bits: u8) -> Result<Resources, Error> {
        RingComparison::validate(input_bits, output_bits)?;
        Ok(resources::<[u64; 2]>(
            input_bits,
            std::mem::size_of::<Self>(),
            [usize::from(input_bits); 2],
        ))
    }

    /// Caller preflights/retains its enclosing reservation. `before_reserve`
    /// runs once for each correction Vec, before actual reservation and before
    /// either root is sampled. Callback receives requested capacity bytes; it
    /// can enforce a bounded allocator or inject allocation failure. No key is
    /// returned on callback/capacity/entropy failure; partial secrets are erased.
    pub(crate) fn issue_with_reservation(
        input_bits: u8,
        output_bits: u8,
        threshold: u64,
        payload: [u64; 2],
        allocation_limit: usize,
        before_reserve: impl FnMut(usize) -> Result<(), Error>,
    ) -> Result<[Self; 2], Error> {
        Self::issue_with(
            input_bits,
            output_bits,
            threshold,
            payload,
            allocation_limit,
            before_reserve,
            |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
        )
    }

    fn issue_with(
        input_bits: u8,
        output_bits: u8,
        threshold: u64,
        payload: [u64; 2],
        allocation_limit: usize,
        before_reserve: impl FnMut(usize) -> Result<(), Error>,
        random: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<[Self; 2], Error> {
        let threshold = Zeroizing::new(threshold);
        let payload = Zeroizing::new(payload);
        let cost = Self::resources(input_bits, output_bits)?;
        if *threshold >= 1u64 << input_bits || payload.iter().any(|v| *v >= 1u64 << output_bits) {
            return Err(Error::Domain);
        }
        if cost.combined_allocation_bytes > allocation_limit {
            return Err(Error::Resource);
        }
        let keys = generate_with_reservation(
            input_bits,
            *threshold,
            *payload,
            (1u64 << output_bits) - 1,
            std::mem::size_of::<Self>(),
            allocation_limit,
            before_reserve,
            random,
        )?;
        Ok(keys.map(|key| Self {
            key,
            input_bits,
            output_bits,
        }))
    }
    pub(crate) fn eval(&self, party: u8, input: u64) -> Result<[u64; 2], Error> {
        if party > 1 || input >= 1u64 << self.input_bits {
            return Err(Error::Domain);
        }
        Ok(evaluate(
            &self.key,
            party,
            input,
            (1u64 << self.output_bits) - 1,
        ))
    }
    #[cfg(test)]
    pub(crate) fn allocation(&self) -> Resources {
        resources::<[u64; 2]>(
            self.input_bits,
            std::mem::size_of::<Self>(),
            [self.key.words.capacity(); 2],
        )
    }
}

#[cfg(test)]
mod ring_tests;

#[cfg(test)]
mod group_tests;

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
