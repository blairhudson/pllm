use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use super::{payload, Capacity, Error, NUMERIC_PROFILE};

/// Immutable original little-endian float32 bytes, never quantized/reserialized.
pub struct FloatTable {
    pub(super) capacity: Capacity,
    pub(super) vocabulary: usize,
    pub(super) width: usize,
    pub(super) bytes: Vec<u8>,
    pub(super) digest: [u8; 32],
}

impl FloatTable {
    pub fn from_le_bytes(vocabulary: usize, width: usize, bytes: &[u8]) -> Result<Self, Error> {
        Self::from_le_bytes_with_capacity(vocabulary, width, bytes, Capacity::Capsule)
    }

    pub fn from_le_bytes_with_capacity(
        vocabulary: usize,
        width: usize,
        bytes: &[u8],
        capacity: Capacity,
    ) -> Result<Self, Error> {
        Self::validate(vocabulary, width, bytes, capacity)?;
        Self::from_le_bytes_owned_with_capacity(vocabulary, width, payload::copy(bytes)?, capacity)
    }

    fn validate(
        vocabulary: usize,
        width: usize,
        bytes: &[u8],
        capacity: Capacity,
    ) -> Result<(), Error> {
        if capacity.table_bytes(vocabulary, width)? != bytes.len() {
            return Err(Error::Shape);
        }
        // Explicit finite-only profile; inspect bits, but preserve original bytes.
        if bytes
            .chunks_exact(4)
            .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        {
            return Err(Error::Numeric);
        }
        Ok(())
    }

    /// Retains the original finite float32 bytes and allocation without copying.
    /// An import may not hide additional retained capacity behind its row count.
    pub fn from_le_bytes_owned(
        vocabulary: usize,
        width: usize,
        bytes: Vec<u8>,
    ) -> Result<Self, Error> {
        Self::from_le_bytes_owned_with_capacity(vocabulary, width, bytes, Capacity::Capsule)
    }

    pub fn from_le_bytes_owned_with_capacity(
        vocabulary: usize,
        width: usize,
        bytes: Vec<u8>,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        Self::validate(vocabulary, width, &bytes, capacity)?;
        if bytes.capacity() != bytes.len() {
            return Err(Error::Limit);
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm.gkm.float-table.v1");
        hash.update(NUMERIC_PROFILE);
        hash.update((vocabulary as u64).to_le_bytes());
        hash.update((width as u64).to_le_bytes());
        hash.update(&bytes);
        Ok(Self {
            capacity,
            vocabulary,
            width,
            bytes,
            digest: hash.finalize().into(),
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.vocabulary, self.width)
    }

    pub fn capacity(&self) -> Capacity {
        self.capacity
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Dense V-bit selector, least-significant bit first; unused high bits zero.
    /// XORs whole original row bit strings. Intermediates need not be valid floats.
    pub fn xor_rows(&self, selector: &[u8]) -> Result<Zeroizing<Vec<u8>>, Error> {
        self.validate_selectors(selector, 1)?;
        let mut output = Zeroizing::new(payload::filled(self.width * 4, 0)?);
        self.xor_selectors_into(selector, 1, &mut output)?;
        Ok(output)
    }

    fn validate_selectors(&self, selectors: &[u8], batch: usize) -> Result<(), Error> {
        let stride = self.vocabulary.div_ceil(8);
        if batch == 0 || selectors.len() != batch.checked_mul(stride).ok_or(Error::Limit)? {
            return Err(Error::Shape);
        }
        let used = self.vocabulary % 8;
        if used != 0
            && selectors
                .chunks_exact(stride)
                .any(|row| row[stride - 1] >> used != 0)
        {
            return Err(Error::Shape);
        }
        Ok(())
    }

    /// Dense token-major selectors and raw outputs. Each table byte is loaded
    /// once across tokens, including rows with zero selector bits for all tokens.
    pub fn xor_selectors_into(
        &self,
        selectors: &[u8],
        batch: usize,
        output: &mut [u8],
    ) -> Result<(), Error> {
        self.validate_selectors(selectors, batch)?;
        let width = self.width.checked_mul(4).ok_or(Error::Limit)?;
        if output.len() != batch.checked_mul(width).ok_or(Error::Limit)? {
            return Err(Error::Shape);
        }
        #[cfg(test)]
        LOOKUP_DISPATCHES.with(|count| count.set(count.get() + 1));
        output.fill(0);
        let stride = self.vocabulary.div_ceil(8);
        const TILE: usize = 64;
        let mut tile = [0u8; TILE];
        for row in 0..self.vocabulary {
            for first in (0..width).step_by(TILE) {
                let count = TILE.min(width - first);
                tile[..count]
                    .copy_from_slice(&self.bytes[row * width + first..row * width + first + count]);
                for (feature, &byte) in tile[..count].iter().enumerate() {
                    for token in 0..batch {
                        let mask = 0u8
                            .wrapping_sub((selectors[token * stride + row / 8] >> (row % 8)) & 1);
                        output[token * width + first + feature] ^= byte & mask;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
std::thread_local! { pub(super) static LOOKUP_DISPATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
