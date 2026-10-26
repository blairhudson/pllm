//! Public, exact rank-one coefficient algebra. Not encryption, authenticated
//! provenance, ROSETTA roundtrip/security, rescaling, or key switching.
//!
//! Restriction selects one explicitly ordered RNS limb without division. Sample
//! extraction preserves the degree-N secret domain: (b0,a0,-a[N-1],...,-a1).
//! Labels express caller consistency only; they do not authenticate origins.
//! Ordinary bounded public algebra only: no work ledger or budget authority.
//! Compiler reference admission owns those separate caller-storage obligations.
use crate::coefficient_layout::CoefficientRing;

pub const MAX_CASES: usize = 32;
pub const MAX_WORD_OPS: usize = 1 << 20;
pub const MAX_OWNER_BYTES: usize = 1 << 20;
pub const MAX_OUTPUT_BYTES: usize = 1 << 20;
// Conservative inline envelope for <=32 caller descriptors, retained wrappers,
// Vec controls and compiler admission metadata, not just coefficient payloads.
const CONTROL_BYTES: usize = 32768;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Descriptor,
    Length,
    NonCanonical,
    Resource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limb {
    pub id: u64,
    pub modulus: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    /// For each limb in order: degree coefficients b, then degree coefficients a.
    OrderedLimbBA,
    /// b0 followed by a0,-a[N-1],...,-a1; dot with unchanged secret order.
    ConstantBASecretOrder,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sign {
    BPlusAS,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Scale {
    pub numerator: u64,
    pub denominator: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KeyDomain {
    pub label: u64,
    pub dimension: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Source {
    pub degree: usize,
    pub rank: usize,
    /// Only the first limb_count entries are active. Unused entry must be None.
    pub limbs: [Option<Limb>; 2],
    pub limb_count: usize,
    pub layout: Layout,
    pub sign: Sign,
    pub scale: Scale,
    pub key: KeyDomain,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Destination {
    pub degree: usize,
    pub rank: usize,
    pub limb: Limb,
    pub layout: Layout,
    pub sign: Sign,
    pub scale: Scale,
    /// Same label and dimension, not a key switch or dimension reduction.
    pub key: KeyDomain,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Case {
    pub source: Source,
    pub destination: Destination,
    pub selected: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Quote {
    pub input_words: usize,
    pub output_words: usize,
    pub oracle_words: usize,
    pub owner_bytes: usize,
    /// Conservative scalar word operations, including validation, initialization,
    /// output, cleanup and one independent direct-constant/dot oracle per case.
    /// Not machine instructions, elapsed time, or ciphertext cost.
    pub word_ops: usize,
}

impl Case {
    fn validate(self) -> Result<(), Error> {
        let s = self.source;
        let d = self.destination;
        if !(1..=2).contains(&s.limb_count)
            || self.selected >= s.limb_count
            || s.rank != 1
            || s.layout != Layout::OrderedLimbBA
            || d.layout != Layout::ConstantBASecretOrder
            || d.rank != 1
            || s.degree != d.degree
            || s.key.dimension != s.degree
            || s.key != d.key
            || s.sign != d.sign
            || s.scale != d.scale
            || s.scale.numerator == 0
            || s.scale.denominator == 0
        {
            return Err(Error::Descriptor);
        }
        for (i, limb) in s.limbs.iter().enumerate() {
            if i < s.limb_count {
                let limb = limb.ok_or(Error::Descriptor)?;
                CoefficientRing::new(s.degree, limb.modulus).map_err(|_| Error::Descriptor)?;
            } else if limb.is_some() {
                return Err(Error::Descriptor);
            }
        }
        if s.limb_count == 2 {
            let l = s.limbs[0].ok_or(Error::Descriptor)?;
            let r = s.limbs[1].ok_or(Error::Descriptor)?;
            if l.id == r.id || l.modulus == r.modulus {
                return Err(Error::Descriptor);
            }
        }
        if s.limbs[self.selected] != Some(d.limb) {
            return Err(Error::Descriptor);
        }
        Ok(())
    }
    pub fn quote(self) -> Result<Quote, Error> {
        self.validate()?;
        let n = self.source.degree;
        let input_words = 2 * n * self.source.limb_count;
        let output_words = n + 1;
        Ok(Quote {
            input_words,
            output_words,
            oracle_words: n,
            owner_bytes: (input_words + output_words + n) * 8,
            // Includes complete canonical scans, full output initialization and
            // cleanup, extraction, secret scans and both O(N) exact oracles.
            // Fixed allowance covers <=2 limbs, descriptors and case controls.
            word_ops: 4 * input_words + 48 * n + 512,
        })
    }
}

pub fn quote(cases: &[Case]) -> Result<Quote, Error> {
    if cases.is_empty() || cases.len() > MAX_CASES {
        return Err(Error::Resource);
    }
    let mut q = Quote {
        input_words: 0,
        output_words: 0,
        oracle_words: 0,
        owner_bytes: CONTROL_BYTES,
        word_ops: 512,
    };
    for case in cases {
        let c = case.quote()?;
        // All terms bounded by degree<=4096, limbs<=2, cases<=32.
        q.input_words += c.input_words;
        q.output_words += c.output_words;
        q.oracle_words += c.oracle_words;
        q.owner_bytes += c.owner_bytes;
        q.word_ops += c.word_ops;
    }
    if q.owner_bytes > MAX_OWNER_BYTES
        || q.output_words * 8 > MAX_OUTPUT_BYTES
        || q.word_ops > MAX_WORD_OPS
    {
        return Err(Error::Resource);
    }
    Ok(q)
}

pub struct Extraction<'a> {
    case: Case,
    // Complete input backing, not merely selected limb, stays borrowed.
    input: &'a [u64],
    words: &'a mut [u64],
}

/// Exact per-case backing lengths; validate ALL limbs before any output write.
/// Safe borrows keep input immutable and output disjoint through cleanup.
pub fn extract<'a>(
    case: Case,
    input: &'a [u64],
    output: &'a mut [u64],
) -> Result<Extraction<'a>, Error> {
    let q = case.quote()?;
    if input.len() != q.input_words || output.len() != q.output_words {
        return Err(Error::Length);
    }
    let n = case.source.degree;
    for limb in 0..case.source.limb_count {
        let modulus = case.source.limbs[limb].ok_or(Error::Descriptor)?.modulus;
        if input[2 * n * limb..2 * n * (limb + 1)]
            .iter()
            .any(|&v| v >= modulus)
        {
            return Err(Error::NonCanonical);
        }
    }
    output.fill(0);
    let at = case.selected * 2 * n;
    let a = &input[at + n..at + 2 * n];
    let modulus = case.destination.limb.modulus;
    output[0] = input[at];
    output[1] = a[0];
    for j in 1..n {
        output[j + 1] = if a[n - j] == 0 { 0 } else { modulus - a[n - j] };
    }
    Ok(Extraction {
        case,
        input,
        words: output,
    })
}

impl Extraction<'_> {
    pub fn descriptor(&self) -> Destination {
        self.case.destination
    }
    pub fn words(&self) -> &[u64] {
        self.words
    }
    /// Independent constant-of-negacyclic-product versus extracted dot check.
    /// Ordinary bounded math, NOT prepayment. Compiler limits admitted calls.
    /// Public deterministic secret vector only; no error assumption.
    /// u128 products of <2^61 words fit; reduce each product BEFORE accumulation.
    pub fn check_public_relation(&self, secret: &[u64]) -> Result<(u64, u64), Error> {
        let n = self.case.source.degree;
        let q = self.case.destination.limb.modulus;
        if secret.len() != n {
            return Err(Error::Length);
        }
        if secret.iter().any(|&v| v >= q) {
            return Err(Error::NonCanonical);
        }
        let at = self.case.selected * 2 * n;
        let a = &self.input[at + n..at + 2 * n];
        let mul = |x: u64, y: u64| ((u128::from(x) * u128::from(y)) % u128::from(q)) as u64;
        let mut direct = (self.input[at] + mul(a[0], secret[0])) % q;
        for i in 1..n {
            direct = (direct + q - mul(a[i], secret[n - i])) % q;
        }
        let mut dot = self.words[0];
        for (x, s) in self.words[1..].iter().zip(secret) {
            dot = (dot + mul(*x, *s)) % q;
        }
        Ok((direct, dot))
    }
}
impl Drop for Extraction<'_> {
    fn drop(&mut self) {
        self.words.fill(0);
    }
}

#[cfg(test)]
#[path = "coefficient_extraction_tests.rs"]
mod tests;
