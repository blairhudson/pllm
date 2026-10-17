use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use super::{payload, Error, MAX_DIMENSION, MAX_VOCABULARY, NUMERIC_PROFILE};

/// Immutable original little-endian float32 bytes, never quantized/reserialized.
pub struct FloatTable {
    pub(super) vocabulary: usize,
    pub(super) width: usize,
    pub(super) bytes: Vec<u8>,
    pub(super) digest: [u8; 32],
}

impl FloatTable {
    pub fn from_le_bytes(vocabulary: usize, width: usize, bytes: &[u8]) -> Result<Self, Error> {
        Self::validate(vocabulary, width, bytes)?;
        Self::from_le_bytes_owned(vocabulary, width, payload::copy(bytes)?)
    }

    fn validate(vocabulary: usize, width: usize, bytes: &[u8]) -> Result<(), Error> {
        if vocabulary == 0 || vocabulary > MAX_VOCABULARY || width == 0 || width > MAX_DIMENSION {
            return Err(Error::Limit);
        }
        if vocabulary.checked_mul(width).and_then(|n| n.checked_mul(4)) != Some(bytes.len()) {
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
        Self::validate(vocabulary, width, &bytes)?;
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
            vocabulary,
            width,
            bytes,
            digest: hash.finalize().into(),
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.vocabulary, self.width)
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Dense V-bit selector, least-significant bit first; unused high bits zero.
    /// XORs whole original row bit strings. Intermediates need not be valid floats.
    pub fn xor_rows(&self, selector: &[u8]) -> Result<Zeroizing<Vec<u8>>, Error> {
        if selector.len() != self.vocabulary.div_ceil(8) {
            return Err(Error::Shape);
        }
        let used = self.vocabulary % 8;
        if used != 0 && selector[selector.len() - 1] >> used != 0 {
            return Err(Error::Shape);
        }
        let mut output = Zeroizing::new(payload::filled(self.width * 4, 0)?);
        for (row, bytes) in self.bytes.chunks_exact(self.width * 4).enumerate() {
            let mask = 0u8.wrapping_sub((selector[row / 8] >> (row % 8)) & 1);
            for (out, &byte) in output.iter_mut().zip(bytes) {
                *out ^= byte & mask;
            }
        }
        Ok(output)
    }
}
