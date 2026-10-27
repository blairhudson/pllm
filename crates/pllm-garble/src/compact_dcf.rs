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
        #[cfg(test)]
        crate::shared_affine_lookup_reference::joint_basis_test_hooks::observe(
            crate::shared_affine_lookup_reference::joint_basis_test_hooks::Event {
                kind: crate::shared_affine_lookup_reference::joint_basis_test_hooks::Kind::CorrectionDrop,
                amount: std::mem::size_of::<Self>(),
            },
        );
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

// ENTRY only. Automatic root/words/final_word field destruction retains its
// original order; no claim of completed erasure/deallocation is made here.
#[cfg(test)]
impl<W: Word> Drop for Key<W> {
    fn drop(&mut self) {
        crate::shared_affine_lookup_reference::joint_basis_test_hooks::observe(
            crate::shared_affine_lookup_reference::joint_basis_test_hooks::Event {
                kind: crate::shared_affine_lookup_reference::joint_basis_test_hooks::Kind::KeyDrop,
                amount: self.words.len(),
            },
        );
    }
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
    before_reserve: impl FnMut(usize) -> Result<(), Error>,
    random: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<[Key<W>; 2], Error> {
    let threshold = Zeroizing::new(threshold);
    let payload = Zeroizing::new(payload);
    generate_with_setup(
        bits,
        mask,
        owner_bytes,
        limit,
        before_reserve,
        |_| Ok((*threshold, *payload)),
        random,
    )
}

// Setup runs only after both correction vectors have passed exact-capacity
// admission. Existing fixed-threshold issuers retain their exact RNG sequence;
// bounded one-hot may sample its threshold and independent split here instead.
fn generate_with_setup<W: Word, R: FnMut(&mut [u8]) -> Result<(), Error>>(
    bits: u8,
    mask: u64,
    owner_bytes: usize,
    limit: usize,
    mut before_reserve: impl FnMut(usize) -> Result<(), Error>,
    setup: impl FnOnce(&mut R) -> Result<(u64, W), Error>,
    mut random: R,
) -> Result<[Key<W>; 2], Error> {
    if resources::<W>(bits, owner_bytes, [usize::from(bits); 2]).combined_allocation_bytes > limit {
        return Err(Error::Resource);
    }
    // Reserve both complete correction owners before requesting any entropy.
    // Neither a key nor a correction vector is Clone.
    let mut words: [Vec<Correction<W>>; 2] = [Vec::new(), Vec::new()];
    for party in &mut words {
        before_reserve(usize::from(bits) * std::mem::size_of::<Correction<W>>())?;
        #[cfg(test)]
        crate::shared_affine_lookup_reference::joint_basis_test_hooks::reserve(
            crate::shared_affine_lookup_reference::joint_basis_test_hooks::Kind::DcfReserve,
            usize::from(bits) * std::mem::size_of::<Correction<W>>(),
        )?;
        party
            .try_reserve_exact(usize::from(bits))
            .map_err(|_| Error::Resource)?;
        #[cfg(test)]
        crate::shared_affine_lookup_reference::joint_basis_test_hooks::observe(
            crate::shared_affine_lookup_reference::joint_basis_test_hooks::Event {
                kind: crate::shared_affine_lookup_reference::joint_basis_test_hooks::Kind::DcfCapacity,
                amount: party.capacity() * std::mem::size_of::<Correction<W>>(),
            },
        );
    }
    if resources::<W>(bits, owner_bytes, words.each_ref().map(|w| w.capacity()))
        .combined_allocation_bytes
        > limit
    {
        return Err(Error::Resource);
    }
    let (threshold, payload) = setup(&mut random)?;
    let threshold = Zeroizing::new(threshold);
    let payload = Zeroizing::new(payload);
    let mut roots = Zeroizing::new([[0; 16]; 2]);
    for root in roots.iter_mut() {
        #[cfg(test)]
        if let Some(result) = crate::shared_affine_lookup_reference::joint_basis_test_hooks::fill(
            crate::shared_affine_lookup_reference::joint_basis_test_hooks::Kind::DcfEntropy, root,
        ) {
            result?;
        } else {
            random(root)?;
        }
        #[cfg(not(test))]
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
    /// Uniform raw threshold and independent additive threshold-mask split.
    /// Both correction allocations complete before this method requests entropy.
    /// Only the enclosing one-use operator receives the split; neither threshold
    /// nor the sibling key may reach an evaluator.
    pub(crate) fn issue_uniform_split(
        input_bits: u8,
        output_bits: u8,
        allocation_limit: usize,
        random: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<([Self; 2], [u64; 2]), Error> {
        Self::validate(input_bits, output_bits)?;
        let mut shares = Zeroizing::new([0u64; 2]);
        let keys = generate_with_setup(
            input_bits,
            (1u64 << output_bits) - 1,
            std::mem::size_of::<Self>(),
            allocation_limit,
            |_| Ok(()),
            |rng| {
                let mut bytes = Zeroizing::new([0u8; 16]);
                rng(&mut *bytes)?;
                let mask = (1u64 << input_bits) - 1;
                let threshold =
                    Zeroizing::new(u64::from_le_bytes(bytes[..8].try_into().unwrap()) & mask);
                shares[0] = u64::from_le_bytes(bytes[8..].try_into().unwrap()) & mask;
                shares[1] = threshold.wrapping_sub(shares[0]) & mask;
                Ok((*threshold, 1u64))
            },
            random,
        )?;
        Ok((
            keys.map(|key| Self {
                key,
                input_bits,
                output_bits,
            }),
            *shares,
        ))
    }

    /// Fixed natural-order scalar traversal. At most 19 live DFS nodes, no heap
    /// allocation, pruning, new PRG, or changed point-evaluation arithmetic.
    /// Deliberately bounded to the one-hot operator's 18-bit public domain.
    #[cfg_attr(not(test), allow(dead_code))] // Compatibility buffer sink.
    pub(crate) fn eval_all_into(&self, party: u8, output: &mut [u64]) -> Result<(), Error> {
        if party > 1 || self.input_bits > 18 || output.len() != 1usize << self.input_bits {
            return Err(Error::Domain);
        }
        self.eval_all_visit(party, |index, value| output[index] = value)
    }

    /// Visit EVERY leaf in unsigned/MSB order using the same fixed DFS as the
    /// buffer sink. Validation precedes the first callback; an infallible visitor
    /// cannot stop, prune or turn a partial traversal into a successful result.
    /// The enclosing owner prices its visitor state separately from DFS scratch.
    pub(crate) fn eval_all_visit(
        &self,
        party: u8,
        mut visit: impl FnMut(usize, u64),
    ) -> Result<(), Error> {
        if party > 1 || self.input_bits > 18 {
            return Err(Error::Domain);
        }
        type Node = ([u8; 16], u8, u64, usize, usize);
        let mut stack = Zeroizing::new([([0u8; 16], 0u8, 0u64, 0usize, 0usize); 19]);
        let mask = (1u64 << self.output_bits) - 1;
        stack[0] = (*self.key.root, party, 0, 0, 0);
        let mut len = 1;
        while len != 0 {
            len -= 1;
            let node = Zeroizing::<Node>::new(stack[len]);
            stack[len].zeroize();
            let (seed, control, sum, depth, prefix) = &*node;
            if *depth == usize::from(self.input_bits) {
                let value = Zeroizing::new(sum.add(
                    signed(
                        convert::<u64>(seed, mask)
                            .add(self.key.final_word.scale(u64::from(*control), mask), mask),
                        party,
                        mask,
                    ),
                    mask,
                ));
                visit(*prefix, *value);
                continue;
            }
            let word = &self.key.words[*depth];
            let mut children = Zeroizing::new(expand::<u64>(seed, mask));
            // Push right then left: leaves emerge in ordinary unsigned/MSB order.
            for direction in [1usize, 0] {
                let value = Zeroizing::new(
                    sum.add(
                        signed(
                            children.1[direction]
                                .add(word.value.scale(u64::from(*control), mask), mask),
                            party,
                            mask,
                        ),
                        mask,
                    ),
                );
                correct(&mut children.0[direction], &word.seed, *control);
                let next_control = children.2[direction] ^ (*control & word.control[direction]);
                stack[len] = (
                    children.0[direction],
                    next_control,
                    *value,
                    depth + 1,
                    (prefix << 1) | direction,
                );
                len += 1;
            }
        }
        Ok(())
    }

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

#[cfg(test)]
mod full_domain_tests {
    use super::*;
    #[test]
    fn every_leaf_matches_point_eval_for_fresh_and_imported_scalar_keys() {
        use crate::fusefss_backend::{Budget, MAX_ALLOCATION_BYTES};
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mask = (1u64 << 50) - 1;
        for bits in 1..=6 {
            let n = 1usize << bits;
            let cost = RingComparison::resources(bits, 50).unwrap();
            for r in 0..n {
                let mut setup = true;
                let (keys, shares) = RingComparison::issue_uniform_split(
                    bits,
                    50,
                    cost.combined_allocation_bytes,
                    |bytes| {
                        if setup {
                            bytes[..8].copy_from_slice(&(r as u64).to_le_bytes());
                            bytes[8..].copy_from_slice(&((r * 37 + 11) as u64).to_le_bytes());
                            setup = false;
                            Ok(())
                        } else {
                            getrandom::fill(bytes).map_err(|_| Error::Randomness)
                        }
                    },
                )
                .unwrap();
                assert_eq!(shares[0].wrapping_add(shares[1]) & (n as u64 - 1), r as u64);
                let mut full = [vec![0; n], vec![0; n]];
                for role in 0..2 {
                    keys[role]
                        .eval_all_into(role as u8, &mut full[role])
                        .unwrap();
                    let mut visited = 0;
                    keys[role]
                        .eval_all_visit(role as u8, |index, value| {
                            assert_eq!(index, visited);
                            assert_eq!(value, full[role][index]);
                            assert_eq!(value, keys[role].eval(role as u8, index as u64).unwrap());
                            visited += 1;
                        })
                        .unwrap();
                    assert_eq!(visited, n);
                    let mut encoded = Zeroizing::new(Vec::new());
                    keys[role].write_material(&mut encoded);
                    let imported =
                        RingComparison::read_material(&encoded, bits, 50, &budget).unwrap();
                    let mut decoded_full = vec![0; n];
                    imported
                        .eval_all_into(role as u8, &mut decoded_full)
                        .unwrap();
                    assert_eq!(decoded_full, full[role]);
                    let mut visited = 0;
                    imported
                        .eval_all_visit(role as u8, |index, value| {
                            assert_eq!(index, visited);
                            assert_eq!(value, decoded_full[index]);
                            assert_eq!(value, imported.eval(role as u8, index as u64).unwrap());
                            visited += 1;
                        })
                        .unwrap();
                    assert_eq!(visited, n);
                    for x in 0..n {
                        assert_eq!(
                            full[role][x],
                            keys[role].eval(role as u8, x as u64).unwrap()
                        );
                    }
                }
                for x in 0..n {
                    assert_eq!(full[0][x].wrapping_add(full[1][x]) & mask, u64::from(x < r));
                }
            }
        }
    }

    #[test]
    fn visitor_matches_buffer_every_leaf_at_width_and_full_domain_boundaries() {
        use crate::fusefss_backend::MAX_ALLOCATION_BYTES;
        for (bits, output_bits) in [(1, 1), (2, 32), (7, 50), (18, 50)] {
            let n = 1usize << bits;
            let keys = RingComparison::issue(
                bits,
                output_bits,
                n as u64 - 1,
                (1u64 << output_bits) - 1,
                MAX_ALLOCATION_BYTES,
            )
            .unwrap();
            for (role, key) in keys.iter().enumerate() {
                let mut full = Zeroizing::new(vec![0; n]);
                key.eval_all_into(role as u8, &mut full).unwrap();
                let mut visited = 0;
                key.eval_all_visit(role as u8, |index, value| {
                    assert_eq!(index, visited);
                    assert_eq!(value, full[index]);
                    // Independent scalar path at the natural point coordinate,
                    // not just another invocation of the shared visitor sink.
                    assert_eq!(value, key.eval(role as u8, index as u64).unwrap());
                    visited += 1;
                })
                .unwrap();
                assert_eq!(visited, n);
                for index in [0, 1, n / 2, n - 1] {
                    assert_eq!(full[index], key.eval(role as u8, index as u64).unwrap());
                }
            }
        }
    }

    #[test]
    fn invalid_sink_geometry_and_roles_have_no_partial_writes_or_visits() {
        use crate::fusefss_backend::MAX_ALLOCATION_BYTES;
        for bits in [1, 18, 19] {
            let [key, _] = RingComparison::issue(bits, 50, 0, 1, MAX_ALLOCATION_BYTES).unwrap();
            let n = 1usize << bits;
            for role in [0, 1, 2, u8::MAX] {
                for len in [0, n - 1, n, n + 1] {
                    if role < 2 && bits <= 18 && len == n {
                        continue;
                    }
                    let mut output = Zeroizing::new(vec![u64::MAX; len]);
                    assert_eq!(key.eval_all_into(role, &mut output), Err(Error::Domain));
                    assert!(output.iter().all(|&value| value == u64::MAX));
                }
                if role > 1 || bits > 18 {
                    let mut called = false;
                    assert_eq!(
                        key.eval_all_visit(role, |_, _| called = true),
                        Err(Error::Domain)
                    );
                    assert!(!called);
                }
            }
        }
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

// Private native codec seam for enclosing consuming party owners. Widths and
// exact length are caller-supplied public geometry, never attacker counts.
pub(crate) trait WireWord: Word {
    fn write(self, out: &mut Vec<u8>);
    fn read(
        input: &mut crate::fusefss_backend::wire::Reader<'_>,
        bits: u8,
    ) -> Result<Self, crate::fusefss_backend::Error>;
}
impl WireWord for u64 {
    fn write(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    fn read(
        input: &mut crate::fusefss_backend::wire::Reader<'_>,
        bits: u8,
    ) -> Result<Self, crate::fusefss_backend::Error> {
        input.word(bits)
    }
}
impl WireWord for [u64; 2] {
    fn write(self, out: &mut Vec<u8>) {
        for value in self {
            value.write(out);
        }
    }
    fn read(
        input: &mut crate::fusefss_backend::wire::Reader<'_>,
        bits: u8,
    ) -> Result<Self, crate::fusefss_backend::Error> {
        Ok([input.word(bits)?, input.word(bits)?])
    }
}
impl<W: WireWord> Key<W> {
    fn write_material(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&*self.root);
        self.final_word.write(out);
        for correction in &self.words {
            out.extend_from_slice(&correction.seed);
            correction.value.write(out);
            out.extend_from_slice(&correction.control);
        }
    }
    fn validate_material(
        bytes: &[u8],
        input_bits: u8,
        output_bits: u8,
    ) -> Result<(), crate::fusefss_backend::Error> {
        use crate::fusefss_backend::{wire::Reader, Error as E};
        RingComparison::validate(input_bits, output_bits).map_err(|_| E::Domain)?;
        if bytes.len()
            != 16
                + std::mem::size_of::<W>()
                + usize::from(input_bits) * (18 + std::mem::size_of::<W>())
        {
            return Err(E::Context);
        }
        // Complete canonical preflight before correction allocation.
        let scan = |reader: &mut Reader<'_>| -> Result<(), E> {
            reader.take(16)?;
            let _ = Zeroizing::new(W::read(reader, output_bits)?);
            for _ in 0..input_bits {
                reader.take(16)?;
                let _ = Zeroizing::new(W::read(reader, output_bits)?);
                if reader.take(2)?.iter().any(|b| *b > 1) {
                    return Err(E::Context);
                }
            }
            Ok(())
        };
        let mut reader = Reader::new(bytes);
        scan(&mut reader)?;
        reader.finish()
    }
    fn read_material(
        bytes: &[u8],
        input_bits: u8,
        output_bits: u8,
        budget: &crate::fusefss_backend::Budget,
    ) -> Result<Self, crate::fusefss_backend::Error> {
        use crate::fusefss_backend::wire::Reader;
        Self::validate_material(bytes, input_bits, output_bits)?;
        let mut words = crate::fusefss_backend::allocate(usize::from(input_bits), budget)?;
        let mut reader = Reader::new(bytes);
        let root = Zeroizing::new(reader.take(16)?.try_into().unwrap());
        let final_word = Zeroizing::new(W::read(&mut reader, output_bits)?);
        for _ in 0..input_bits {
            let seed = Zeroizing::new(reader.take(16)?.try_into().unwrap());
            let value = Zeroizing::new(W::read(&mut reader, output_bits)?);
            let control = reader.take(2)?.try_into().unwrap();
            words.push(Correction {
                seed: *seed,
                value: *value,
                control,
            });
        }
        Ok(Self {
            root,
            words,
            final_word,
        })
    }
}
macro_rules! material_codec {
    ($owner:ty, $word:ty) => {
        impl $owner {
            pub(crate) fn validate_material(
                bytes: &[u8],
                input_bits: u8,
                output_bits: u8,
            ) -> Result<(), crate::fusefss_backend::Error> {
                Key::<$word>::validate_material(bytes, input_bits, output_bits)
            }
            pub(crate) fn write_material(&self, out: &mut Vec<u8>) {
                self.key.write_material(out);
            }
            pub(crate) fn read_material(
                bytes: &[u8],
                input_bits: u8,
                output_bits: u8,
                budget: &crate::fusefss_backend::Budget,
            ) -> Result<Self, crate::fusefss_backend::Error> {
                Ok(Self {
                    key: Key::<$word>::read_material(bytes, input_bits, output_bits, budget)?,
                    input_bits,
                    output_bits,
                })
            }
        }
    };
}
material_codec!(RingComparison, u64);
material_codec!(VectorComparison, [u64; 2]);
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
