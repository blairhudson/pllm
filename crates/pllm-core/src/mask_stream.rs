//! Bounded, one-use row views of the existing SHAKE256 preparation stream.
//! Domains and wire identity are supplied by the authenticated protocol boundary.
//! No full mask inventory is retained; sequential claims reuse the XOF cursor.
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256, Shake256Reader,
};

pub const MAX_ROWS: usize = 1 << 20;
pub const MAX_ELEMENTS: usize = 1 << 26;
pub const MAX_DOMAIN_BYTES: usize = 8192;

struct Stream {
    base: Shake256Reader,
    reader: Shake256Reader,
    position: usize,
}

impl Stream {
    fn new(domain: &[u8], seed: &[u8; 32]) -> Self {
        let mut hash = Shake256::default();
        hash.update(domain);
        hash.update(seed);
        let base = hash.finalize_xof();
        Self {
            reader: base.clone(),
            base,
            position: 0,
        }
    }

    fn read(&mut self, start: usize, elements: usize, word_bytes: usize) -> Vec<u8> {
        if start < self.position {
            self.reader = self.base.clone();
            self.position = 0;
        }
        // Out-of-order disjoint leases are allowed. Skip without allocating the
        // prefix; ordinary ordered prefill/decode never repeats that work.
        let mut scratch = [0u8; 4096];
        while self.position < start {
            let count = (start - self.position).min(scratch.len());
            self.reader.read(&mut scratch[..count]);
            self.position += count;
        }
        let mut result = vec![0u8; elements * 4];
        for chunk in result.chunks_mut(4 * 1024) {
            let count = chunk.len() / 4;
            self.reader.read(&mut scratch[..count * word_bytes]);
            for (src, dst) in scratch[..count * word_bytes]
                .chunks_exact(word_bytes)
                .zip(chunk.chunks_exact_mut(4))
            {
                dst[..word_bytes].copy_from_slice(src);
            }
        }
        self.position += elements * word_bytes;
        result
    }
}

/// Not cloneable or serializable. A row is claimed once across both streams.
pub struct MaskRows {
    streams: Option<(Stream, Stream)>,
    used: Vec<bool>,
    rows: usize,
    input_width: usize,
    output_width: usize,
    word_bytes: usize,
}

impl MaskRows {
    pub fn new(
        seed: &[u8; 32],
        input_domain: &[u8],
        output_domain: &[u8],
        rows: usize,
        input_width: usize,
        output_width: usize,
        wire_bits: u8,
    ) -> Result<Self, String> {
        if rows == 0
            || rows > MAX_ROWS
            || input_width == 0
            || output_width == 0
            || !matches!(wire_bits, 16 | 24 | 32)
            || input_domain.is_empty()
            || output_domain.is_empty()
            || input_domain == output_domain
            || input_domain.len().max(output_domain.len()) > MAX_DOMAIN_BYTES
            || rows
                .checked_mul(input_width.max(output_width))
                .is_none_or(|n| n > MAX_ELEMENTS)
        {
            return Err("prepared mask stream exceeds its shape/domain bounds".into());
        }
        Ok(Self {
            streams: Some((
                Stream::new(input_domain, seed),
                Stream::new(output_domain, seed),
            )),
            used: vec![false; rows],
            rows,
            input_width,
            output_width,
            word_bytes: usize::from(wire_bits / 8),
        })
    }

    fn range(&self, start: usize, count: usize) -> Result<std::ops::Range<usize>, String> {
        if self.streams.is_none() {
            return Err("prepared mask stream is closed".into());
        }
        let end = start
            .checked_add(count)
            .ok_or("prepared mask range overflow")?;
        if count == 0 || end > self.rows {
            return Err("prepared mask range exceeds its inventory".into());
        }
        Ok(start..end)
    }

    pub fn take(&mut self, start: usize, count: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
        let range = match self.range(start, count) {
            Ok(range) if !self.used[range.clone()].iter().any(|used| *used) => range,
            _ => {
                self.cancel();
                return Err("prepared mask range is invalid, consumed or closed".into());
            }
        };
        self.used[range].fill(true); // burn before expansion or allocation
        let (input, output) = self
            .streams
            .as_mut()
            .ok_or("prepared mask stream is closed")?;
        Ok((
            input.read(
                start * self.input_width * self.word_bytes,
                count * self.input_width,
                self.word_bytes,
            ),
            output.read(
                start * self.output_width * self.word_bytes,
                count * self.output_width,
                self.word_bytes,
            ),
        ))
    }

    /// Cancellation of an unconsumed reservation is idempotent and cannot affect
    /// already returned owned buffers, including an in-flight request's masks.
    pub fn burn(&mut self, start: usize, count: usize) -> Result<(), String> {
        if self.streams.is_none() {
            return Ok(());
        }
        let range = self.range(start, count)?;
        self.used[range].fill(true);
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.streams = None;
        self.used.fill(true);
    }

    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.used.capacity() * std::mem::size_of::<bool>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_partitions_match_whole_xof() {
        for bits in [16, 24, 32] {
            let mut rows = MaskRows::new(&[9; 32], b"input", b"output", 71, 47, 29, bits).unwrap();
            for (start, count) in [(10, 54), (64, 1), (65, 6), (0, 10)] {
                let (input, output) = rows.take(start, count).unwrap();
                for (domain, width, data) in [
                    (b"input".as_slice(), 47, input),
                    (b"output".as_slice(), 29, output),
                ] {
                    let mut hash = Shake256::default();
                    hash.update(domain);
                    hash.update(&[9; 32]);
                    let mut reference = vec![0u8; 71 * width * usize::from(bits / 8)];
                    hash.finalize_xof().read(&mut reference);
                    let word = usize::from(bits / 8);
                    for (actual, expected) in data.chunks_exact(4).zip(
                        reference[start * width * word..(start + count) * width * word]
                            .chunks_exact(word),
                    ) {
                        assert_eq!(&actual[..word], expected);
                        assert!(actual[word..].iter().all(|byte| *byte == 0));
                    }
                }
            }
            assert!(rows.take(0, 1).is_err());
        }
    }

    #[test]
    fn cancellation_replay_and_bounds() {
        let make = || MaskRows::new(&[0; 32], b"in", b"out", 4, 5, 8, 24).unwrap();
        let mut rows = make();
        rows.burn(0, 2).unwrap();
        rows.burn(0, 2).unwrap();
        assert!(rows.take(0, 1).is_err());
        assert!(rows.take(2, 1).is_err());
        for (start, count) in [(0, 0), (0, 5), (usize::MAX, 2)] {
            let mut rows = make();
            assert!(rows.take(start, count).is_err());
            assert!(rows.take(0, 1).is_err());
        }
        let mut rows = make();
        let value = rows.take(0, 1).unwrap();
        rows.cancel();
        assert_ne!(value.0, vec![0; 20]);
        assert!(rows.take(1, 1).is_err());
        assert!(MaskRows::new(&[0; 32], b"in", b"out", MAX_ROWS + 1, 1, 1, 16).is_err());
        assert!(MaskRows::new(&[0; 32], b"in", b"out", 1, MAX_ELEMENTS + 1, 1, 16).is_err());
        assert!(make().retained_bytes() < 2048);
    }
}
