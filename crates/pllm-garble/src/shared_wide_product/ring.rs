//! Allocation-free Beaver algebra shared by the legacy 2^64 product and
//! ring-aware callers. This is not another framed protocol or lifecycle owner.
//! The enclosing Party must enforce one opening, peer/context validation,
//! cancellation/replay burns and memory admission. Triple reconstruction consumes
//! the share. No clear triple material or deterministic issuance API is exported.
use crate::{shared_circuit::random_word, shared_rescale_reference::Error};
use zeroize::Zeroizing;

/// Private triple payload, independent of public ring metadata/struct padding.
pub(crate) const TRIPLE_BYTES: usize = 3 * std::mem::size_of::<u64>();

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
/// Five full-width draws retain existing PLLM randomness and draw ordering;
/// low-bit masking is uniform for every power-of-two ring, including 2^64.
pub(crate) fn issue(bits: u8) -> Result<[TripleShare; 2], Error> {
    issue_with(bits, random_word)
}

fn issue_with(
    bits: u8,
    mut draw: impl FnMut() -> Result<u64, Error>,
) -> Result<[TripleShare; 2], Error> {
    let mask = mask(bits)?;
    let a = Zeroizing::new(draw()? & mask);
    let b = Zeroizing::new(draw()? & mask);
    let left = Zeroizing::new([draw()? & mask, draw()? & mask, draw()? & mask]);
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
        Ok(Self {
            words: Zeroizing::new([input.word(bits)?, input.word(bits)?, input.word(bits)?]),
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
