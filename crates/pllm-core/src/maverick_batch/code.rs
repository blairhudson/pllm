use sha2::{Digest, Sha256};

use super::arithmetic::{add, below, canonical, mul, RandomWords};
use super::memory::{Memory, Words};
use super::{Error, Result, FIELD, REPEAT, SECONDARY_DIMENSION, SECONDARY_LENGTH};

/// Explicit public source-ensemble description. This is not a certificate of
/// sampling provenance, minimum distance, or dual-LPN security.
#[derive(Clone, Copy)]
pub struct CodeDescription<'a> {
    pub field: u32,
    pub dimension: usize,
    pub first_permutation: &'a [u32],
    pub second_permutation: &'a [u32],
    pub first_scale: &'a [u32],
    pub second_scale: &'a [u32],
}

impl CodeDescription<'_> {
    pub(super) fn same_as(self, other: CodeDescription<'_>) -> bool {
        self.field == other.field
            && self.dimension == other.dimension
            && self.first_permutation == other.first_permutation
            && self.second_permutation == other.second_permutation
            && self.first_scale == other.first_scale
            && self.second_scale == other.second_scale
    }

    pub(super) fn validate(self, dimension: usize) -> Result<()> {
        if self.field != FIELD {
            return Err(Error::Field);
        }
        if self.dimension != dimension || !(1..=SECONDARY_DIMENSION).contains(&dimension) {
            return Err(Error::Code);
        }
        let length = dimension * REPEAT;
        for permutation in [self.first_permutation, self.second_permutation] {
            if permutation.len() != length {
                return Err(Error::Code);
            }
            let mut seen = [false; SECONDARY_LENGTH];
            for &index in permutation {
                let index = index as usize;
                if index >= length || seen[index] {
                    return Err(Error::Code);
                }
                seen[index] = true;
            }
        }
        for scale in [self.first_scale, self.second_scale] {
            canonical(scale, length)?;
            if scale.contains(&0) {
                return Err(Error::Code);
            }
        }
        Ok(())
    }

    pub fn digest(self) -> Result<[u8; 32]> {
        self.validate(self.dimension)?;
        let mut hash = Sha256::new();
        hash.update(b"pllm.maverick.raa.v1\0");
        hash.update(FIELD.to_le_bytes());
        hash.update((self.dimension as u64).to_le_bytes());
        hash.update((REPEAT as u32).to_le_bytes());
        hash.update(((self.dimension * REPEAT) as u64).to_le_bytes());
        for part in [
            self.first_permutation,
            self.second_permutation,
            self.first_scale,
            self.second_scale,
        ] {
            for &value in part {
                hash.update(value.to_le_bytes());
            }
        }
        Ok(hash.finalize().into())
    }
}

pub(super) struct Code {
    dimension: usize,
    pi1: Words,
    pi2: Words,
    v1: Words,
    v2: Words,
}

impl Code {
    pub(super) fn copy(memory: &Memory, description: CodeDescription<'_>) -> Result<Self> {
        description.validate(description.dimension)?;
        Ok(Self {
            dimension: description.dimension,
            pi1: Words::copy(memory, description.first_permutation)?,
            pi2: Words::copy(memory, description.second_permutation)?,
            v1: Words::copy(memory, description.first_scale)?,
            v2: Words::copy(memory, description.second_scale)?,
        })
    }

    pub(super) fn description(&self) -> CodeDescription<'_> {
        CodeDescription {
            field: FIELD,
            dimension: self.dimension,
            first_permutation: self.pi1.as_slice(),
            second_permutation: self.pi2.as_slice(),
            first_scale: self.v1.as_slice(),
            second_scale: self.v2.as_slice(),
        }
    }

    pub(super) fn length(&self) -> usize {
        self.dimension * REPEAT
    }

    /// G^T*w. The explicit source convention is Pi[i,j]=[pi[j]=i].
    pub(super) fn encode(&self, memory: &Memory, input: &[u32]) -> Result<Words> {
        canonical(input, self.dimension)?;
        let length = self.length();
        let mut first = Words::zeros(memory, length)?;
        let mut sum = 0;
        for (j, out) in first.as_mut().iter_mut().enumerate() {
            sum = add(
                sum,
                mul(
                    input[self.pi1.as_slice()[j] as usize / REPEAT],
                    self.v1.as_slice()[j],
                ),
            );
            *out = sum;
        }
        let mut result = Words::zeros(memory, length)?;
        sum = 0;
        for (j, out) in result.as_mut().iter_mut().enumerate() {
            sum = add(
                sum,
                mul(
                    first.as_slice()[self.pi2.as_slice()[j] as usize],
                    self.v2.as_slice()[j],
                ),
            );
            *out = sum;
        }
        Ok(result)
    }

    /// G*e. Suffix sums and inverse data movement, then contiguous repeat sums.
    pub(super) fn syndrome(&self, memory: &Memory, sparse: &Sparse) -> Result<Words> {
        if sparse.length != self.length() {
            return Err(Error::Shape);
        }
        let mut dense = Words::zeros(memory, self.length())?;
        for (position, value) in sparse.entries() {
            dense.as_mut()[position] = value;
        }
        let mut first = Words::zeros(memory, self.length())?;
        let mut sum = 0;
        for j in (0..self.length()).rev() {
            sum = add(sum, dense.as_slice()[j]);
            first.as_mut()[self.pi2.as_slice()[j] as usize] = mul(sum, self.v2.as_slice()[j]);
        }
        let mut result = Words::zeros(memory, self.dimension)?;
        sum = 0;
        for j in (0..self.length()).rev() {
            sum = add(sum, first.as_slice()[j]);
            let row = self.pi1.as_slice()[j] as usize / REPEAT;
            result.as_mut()[row] = add(result.as_slice()[row], mul(sum, self.v1.as_slice()[j]));
        }
        Ok(result)
    }
}

pub(super) struct Sparse {
    length: usize,
    pairs: Words,
}

impl Sparse {
    pub(super) fn sample(
        memory: &Memory,
        rng: &mut impl RandomWords,
        length: usize,
        weight: usize,
    ) -> Result<Self> {
        if length > SECONDARY_LENGTH || weight == 0 || weight > length {
            return Err(Error::Shape);
        }
        let mut pool = Words::zeros(memory, length)?;
        for (i, value) in pool.as_mut().iter_mut().enumerate() {
            *value = i as u32;
        }
        let mut pairs = Words::zeros(memory, 2 * weight)?;
        for i in 0..weight {
            let j = i + below(rng, (length - i) as u32)? as usize;
            pool.as_mut().swap(i, j);
            pairs.as_mut()[2 * i] = pool.as_slice()[i];
            pairs.as_mut()[2 * i + 1] = below(rng, FIELD - 1)? + 1;
        }
        Ok(Self { length, pairs })
    }

    pub(super) fn from_pairs(
        memory: &Memory,
        length: usize,
        weight: usize,
        pairs: &[u32],
    ) -> Result<Self> {
        if length > SECONDARY_LENGTH || weight == 0 || weight > length || pairs.len() != 2 * weight
        {
            return Err(Error::Shape);
        }
        let mut seen = [false; SECONDARY_LENGTH];
        for pair in pairs.chunks_exact(2) {
            let position = pair[0] as usize;
            if position >= length || seen[position] || pair[1] == 0 || pair[1] >= FIELD {
                return Err(Error::Field);
            }
            seen[position] = true;
        }
        Ok(Self {
            length,
            pairs: Words::copy(memory, pairs)?,
        })
    }

    pub(super) fn copy(&self, memory: &Memory) -> Result<Self> {
        Ok(Self {
            length: self.length,
            pairs: Words::copy(memory, self.pairs.as_slice())?,
        })
    }

    pub(super) fn pairs(&self) -> &[u32] {
        self.pairs.as_slice()
    }

    pub(super) fn entries(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.pairs
            .as_slice()
            .chunks_exact(2)
            .map(|p| (p[0] as usize, p[1]))
    }

    pub(super) fn dot(&self, vector: &[u32]) -> u32 {
        debug_assert_eq!(self.length, vector.len());
        self.entries()
            .fold(0, |sum, (i, value)| add(sum, mul(value, vector[i])))
    }
}
