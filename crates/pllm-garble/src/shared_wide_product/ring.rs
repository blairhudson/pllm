//! Allocation-free Beaver algebra shared by the legacy 2^64 product and
//! ring-aware callers. This is not another framed protocol or lifecycle owner.
//! The enclosing Party must enforce one opening, peer/context validation,
//! cancellation/replay burns and memory admission. Triple reconstruction consumes
//! the share. No clear triple material or deterministic issuance API is exported.
use crate::shared_rescale_reference::Error;
use zeroize::Zeroizing;

/// Private triple payload, independent of public ring metadata/struct padding.
pub(crate) const TRIPLE_BYTES: usize = 3 * std::mem::size_of::<u64>();

const ENTROPY_BYTES: usize = 5 * std::mem::size_of::<u64>();

// Logical source-managed overlap: raw entropy, a/b, the ring mask, both returned
// owners and one decoded word. This is 136 bytes, within the existing native
// callers' fixed crypto-stack allowances (4096 bytes, 8192 for predicates).
// DCF issuance is sequential, not nested in this call. No heap owner is added;
// no reservation/cap or material/wire quote needs changing.
const ISSUANCE_SCRATCH_BYTES: usize =
    ENTROPY_BYTES + 4 * std::mem::size_of::<u64>() + 2 * std::mem::size_of::<TripleShare>();

/// Noncloneable and deliberately not Debug. Public ring metadata is retained to
/// prevent accidental reconstruction under a different modulus. Exactly three
/// zeroizing u64s constitute the 24-byte secret payload at every width.
pub(crate) struct TripleShare {
    words: Zeroizing<[u64; 3]>,
    mask: u64,
}

fn mask(bits: u8) -> Result<u64, Error> {
    if !(1..=64).contains(&bits) {
        return Err(Error::Domain);
    }
    Ok(u64::MAX >> (64 - bits))
}

/// Trusted dealer: independent OS-random a, b and three uniform left shares,
/// then complementary right shares. Width is validated *before any RNG call*.
/// One fixed-size OS fill supplies the same five full-width words, in existing
/// a/b/left-a/left-b/left-c little-endian order. There is no seed expansion or
/// change of entropy source. Low-bit masking is exactly uniform in every 2^n
/// ring, including 2^64. A failed/partial fill exposes no material and is not
/// retried here; Zeroizing erases the entire buffer on error or unwind.
pub(crate) fn issue(bits: u8) -> Result<[TripleShare; 2], Error> {
    issue_from_fill(bits, |bytes| {
        #[cfg(test)]
        if let Some(result) = bulk_tests::fill_override(bytes) {
            return result;
        }
        getrandom::fill(bytes).map_err(|_| Error::Randomness)
    })
}

fn issue_from_fill(
    bits: u8,
    fill: impl FnOnce(&mut [u8]) -> Result<(), Error>,
) -> Result<[TripleShare; 2], Error> {
    const { assert!(ISSUANCE_SCRATCH_BYTES <= 4096) };
    let mask = mask(bits)?;
    // Geometry first; no allocation or admission occurs inside this primitive.
    // Enclosing native parties reserve their complete envelope before calling.
    let mut raw = Zeroizing::new([0; ENTROPY_BYTES]);
    fill(&mut *raw)?;
    let word = |index: usize| {
        u64::from_le_bytes(raw[index * 8..(index + 1) * 8].try_into().unwrap()) & mask
    };
    let a = Zeroizing::new(word(0));
    let b = Zeroizing::new(word(1));
    let left = Zeroizing::new([word(2), word(3), word(4)]);
    let right = Zeroizing::new([
        a.wrapping_sub(left[0]) & mask,
        b.wrapping_sub(left[1]) & mask,
        a.wrapping_mul(*b).wrapping_sub(left[2]) & mask,
    ]);
    Ok([
        TripleShare { words: left, mask },
        TripleShare { words: right, mask },
    ])
}

impl TripleShare {
    pub(crate) fn write_material(&self, out: &mut Vec<u8>) {
        for value in self.words.iter() {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    pub(crate) fn read_material(
        input: &mut crate::fusefss_backend::wire::Reader<'_>,
        bits: u8,
    ) -> Result<Self, crate::fusefss_backend::Error> {
        // Valid fixed records use two whole-span bounds checks (peek/commit),
        // rather than three word reads/offset updates/width checks. Both old
        // and new paths are allocation-free. Keep malformed-input cursor/error
        // precedence exactly: only the local invalid bytes use the old reads,
        // never another provider call, material issuance or import-guard claim.
        let decoded = input
            .fork()
            .take(TRIPLE_BYTES)
            .and_then(|bytes| crate::material_wire::decode_fixed_words::<3>(bytes, bits));
        let words = match decoded {
            Ok(words) => {
                input.take(TRIPLE_BYTES)?;
                words
            }
            Err(_) => {
                let mut words = Zeroizing::new([0; 3]);
                for word in words.iter_mut() {
                    *word = input.word(bits)?;
                }
                words
            }
        };
        // Two fixed-array factory/result slots, this triple and local Reader
        // staging fit the existing 4096-byte caller crypto-stack allowance.
        // No heap owner or refund/credit is added; quotes stay conservative.
        // Listed logical owners only, not compiler-stack/allocator/RSS proof.
        const _: () = assert!(
            2 * TRIPLE_BYTES
                + std::mem::size_of::<TripleShare>()
                + 2 * std::mem::size_of::<crate::fusefss_backend::wire::Reader<'_>>()
                <= 4096
        );
        Ok(Self {
            words,
            mask: mask(bits).map_err(crate::fusefss_backend::primitive_error)?,
        })
    }
    /// Local [x-a, y-b] share, with canonical input validation rather than silent
    /// narrowing. Borrowing allows state to remain at the enclosing Party;
    /// callers must not expose repeated openings with different operands.
    pub(crate) fn open(&self, x: u64, y: u64) -> Result<Zeroizing<[u64; 2]>, Error> {
        self.canonical(&[x, y])?;
        Ok(Zeroizing::new([
            x.wrapping_sub(self.words[0]) & self.mask,
            y.wrapping_sub(self.words[1]) & self.mask,
        ]))
    }

    /// Consume one party's share to compute c_b + d*b_b + e*a_b + [b=0]*d*e.
    /// Both opening pairs and role are validated before arithmetic. Inputs and
    /// output are canonical ring elements. Full-u64 wrapping followed by masking
    /// preserves the exact product in every admitted 2^n ring, with no rescale.
    /// This validates algebraic geometry, not peer identity or authentication.
    pub(crate) fn finish(self, party: u8, own: &[u64; 2], peer: &[u64; 2]) -> Result<u64, Error> {
        if party > 1 {
            return Err(Error::Domain);
        }
        self.canonical(own)?;
        self.canonical(peer)?;
        let d = Zeroizing::new(own[0].wrapping_add(peer[0]) & self.mask);
        let e = Zeroizing::new(own[1].wrapping_add(peer[1]) & self.mask);
        Ok(self.words[2]
            .wrapping_add(d.wrapping_mul(self.words[1]))
            .wrapping_add(e.wrapping_mul(self.words[0]))
            .wrapping_add(d.wrapping_mul(*e).wrapping_mul(u64::from(party == 0)))
            & self.mask)
    }

    fn canonical(&self, values: &[u64]) -> Result<(), Error> {
        if values.iter().any(|value| *value > self.mask) {
            return Err(Error::Domain);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "ring_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ring_bulk_tests.rs"]
mod bulk_tests;

// Preserve the older word-injection tests without retaining a five-call
// production path. Their adapter feeds the very same bulk decoder/algebra.
#[cfg(test)]
use bulk_tests::issue_with;
