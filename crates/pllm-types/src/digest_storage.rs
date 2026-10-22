//! Allocation-free access to validated digest identity and logical backing.

use crate::Digest;

impl Digest {
    /// Decode the canonical lowercase hexadecimal identity without allocating.
    /// Construction and deserialization enforce the 32-byte SHA-256 invariant.
    pub fn as_sha256(&self) -> [u8; 32] {
        let nibble = |byte: u8| {
            if byte <= b'9' {
                byte - b'0'
            } else {
                byte - b'a' + 10
            }
        };
        let mut result = [0; 32];
        for (output, pair) in result.iter_mut().zip(self.0.as_bytes().chunks_exact(2)) {
            *output = (nibble(pair[0]) << 4) | nibble(pair[1]);
        }
        result
    }

    /// Actual logical String capacity, excluding this digest's inline metadata.
    /// This is not allocator overhead or a process-RSS measurement.
    pub fn backing_capacity_bytes(&self) -> usize {
        self.0.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_round_trips_through_validated_hex() {
        for offset in 0..=u8::MAX {
            let bytes = std::array::from_fn(|i| offset.wrapping_add(i as u8));
            let digest = Digest::from_sha256(bytes);
            assert_eq!(digest.as_sha256(), bytes);
            let decoded: Digest =
                serde_json::from_str(&serde_json::to_string(&digest).unwrap()).unwrap();
            assert_eq!(decoded.as_sha256(), bytes);
        }
    }

    #[test]
    fn capacity_counts_retained_backing_not_hex_length() {
        let mut text = String::with_capacity(4096);
        text.push_str(Digest::from_sha256([0xab; 32]).as_str());
        let capacity = text.capacity();
        let digest = Digest(text);
        assert_eq!(digest.backing_capacity_bytes(), capacity);
        assert!(capacity > digest.as_str().len());
        assert_eq!(digest.as_sha256(), [0xab; 32]);
    }
}
