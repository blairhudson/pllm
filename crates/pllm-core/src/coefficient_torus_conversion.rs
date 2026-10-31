//! Allocation-free rank-one coefficient-LWE to Torus64 arithmetic, R-S/T64-v1.
//!
//! PUBLIC facts: extraction uses (b,a), b+sum(a_i*s_i), and carries its existing
//! key domain, dimension and public scale. Those fields are preserved, not
//! authenticated. Binary secrets are a semantic precondition, never an argument.
//! INFERRED mapping: adapt that convention to (B,A), B-sum(A_i*s_i), by negating
//! rounded masks. This is an implementable native component, not an author-faithful
//! ROSETTA reproduction or a complete encrypted protocol.
//! APPROXIMATE profile: deterministic component rounding, not phase correction.
//! R_q(x)=floor((2^64*x+floor(q/2))/q) mod 2^64. Odd q excludes exact half ties;
//! this is nearest rounding, not an invented ties-even case or floating arithmetic.
//! Incoming public scale Delta becomes exactly 2^64*Delta/q torus units.
//! Circular phase rounding error is <=(h+1)/2 torus units for binary weight h;
//! conservative bound uses h=dimension. Incoming absolute error E adds 2^64*E/q.
//! Divide these bounds by 2^64 for the circular metric modulo one; they are not
//! linear distances across the wrap. Deterministic mask images are NOT fresh
//! uniform torus samples and do not restore per-sample bias.
//!
//! No secret, key generation, RNG, allocation, decryption, lookup, bootstrap,
//! plaintext fallback, callback, or runtime activation. Key switching, external
//! products, CMUX and bootstrapping remain future actual native work. No security,
//! trained-model, D37 Quant4096, exact-floor, equality, or unconditional total
//! phase-error <1/8192 guarantee (incoming E must also be bounded).
//! Kernel quotes are NOT custody authority: shared admission must still charge
//! original source/key/output owners and all transport/scheduling when integrated.
//! Quotes are per-call arithmetic limits, not cumulative debits. The original
//! caller must pay both directional calls and failed attempts, and retain all
//! required input/source/key/output owners itself. Output wrappers borrow only
//! their output buffers; they neither retain input custody nor authorize release.
//! Safe Rust slices enforce disjoint input/output. All refusals precede writes.
//! Valid calls have public fixed scan/write counts; division is NOT claimed
//! constant-time, and malformed public descriptors can refuse early.
//!
//! Reverse is a generic arithmetic utility, NOT adopted ROSETTA HomComp encoding,
//! security or a numeric profile. U(x)=RNE(q*x/2^64) mod q uses ties-to-even;
//! (B,A) maps to (U(B),-U(A) mod q): round BEFORE negating, including half ties.
//! Circular reverse error is <=(h+1)/2 q-units, plus q*E_T/2^64 for incoming
//! torus phase error E_T (torus WORD units). Forward/reverse composition adds
//! q*(h+1)/(2*2^64) q-units only when E_T excludes that forward contribution;
//! original coefficient error E_C contributes E_C, not q*E_C/2^64. These are
//! circular bounds, not exact phases.
//! Unchanged forward components recover canonical originals: forward error
//! <=1/2 torus unit rescales to <=q/(2*2^64)<1/4 q-unit. This also holds
//! after the forward mask negation. Arbitrary reverse/forward is not identity.
//! Neither direction removes noise or restores uniform masks.

use crate::coefficient_extraction::{Destination, Layout, Sign, MAX_OWNER_BYTES, MAX_WORD_OPS};
pub use crate::coefficient_layout::LayoutError as Error;

pub const MAX_DIMENSION: usize = 2048;
const TORUS: u128 = 1_u128 << 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Quote {
    pub input_words: usize,
    pub output_words: usize,
    pub scratch_words: usize,
    /// Simultaneously live distinct caller coefficient buffers, not custody.
    pub caller_bytes: usize,
    pub descriptor_bytes: usize,
    /// Conservative abstract scalar-word work, not CPU instructions or latency.
    pub word_ops: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Torus64Conversion {
    source: Destination,
    quote: Quote,
}

/// Borrows caller output without allocation or implicit cleanup/ownership transfer.
/// Components are (B,A); the output phase convention is B-sum(A_i*s_i).
/// `source_descriptor` is deliberately NOT an odd-modulus output descriptor.
pub struct Torus64Ciphertext<'a> {
    source: Destination,
    words: &'a mut [u64],
}

/// Borrowed reverse output (b,a), phase b+sum(a_i*s_i), in the original domain.
/// No allocation, ownership transfer or zero-on-drop: caller retains cleanup.
/// The descriptor is consistency metadata, not authenticated custody or a new
/// scale authority. Its scale applies only to the bound unchanged-scale view.
/// This arithmetic view borrows output only, not input custody or dependencies.
pub struct CoefficientCiphertext<'a> {
    source: Destination,
    words: &'a mut [u64],
}

impl CoefficientCiphertext<'_> {
    pub fn descriptor(&self) -> Destination {
        self.source
    }
    pub fn words(&self) -> &[u64] {
        self.words
    }
}

impl Torus64Ciphertext<'_> {
    pub fn source_descriptor(&self) -> Destination {
        self.source
    }
    pub fn words(&self) -> &[u64] {
        self.words
    }
    /// Edit components while retaining the original borrowed descriptor.
    /// This does not certify an operation, key, scale, error or custody change.
    /// Reverse scale interpretation requires unchanged scale/domain/convention;
    /// callers must establish those semantics independently of this raw access.
    pub fn words_mut(&mut self) -> &mut [u64] {
        self.words
    }
}

impl Torus64Conversion {
    pub fn new(source: Destination) -> Result<Self, Error> {
        // Check sizing arithmetic even for hostile, unbounded descriptors.
        let words = source.key.dimension.checked_add(1).ok_or(Error::Overflow)?;
        let caller_bytes = words.checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u64>()))
            .ok_or(Error::Overflow)?;
        let word_ops = words.checked_mul(64)
            .and_then(|v| v.checked_add(128)).ok_or(Error::Overflow)?;
        if source.key.dimension == 0 || source.key.dimension > MAX_DIMENSION {
            return Err(Error::InvalidDegree);
        }
        if source.rank != 1 || source.degree != source.key.dimension
            || source.layout != Layout::ConstantBASecretOrder
            || source.sign != Sign::BPlusAS
            || source.scale.numerator == 0 || source.scale.denominator == 0
        {
            return Err(Error::InvalidGeometry);
        }
        let q = source.limb.modulus;
        if q <= 4096 || q >= (1_u64 << 63) || q & 1 == 0 {
            return Err(Error::InvalidModulus);
        }
        if caller_bytes > MAX_OWNER_BYTES || word_ops > MAX_WORD_OPS {
            return Err(Error::ResourceLimit);
        }
        // For every canonical x, x*2^64+floor(q/2) fits in u128.
        u128::from(q - 1).checked_mul(TORUS)
            .and_then(|v| v.checked_add(u128::from(q / 2)))
            .ok_or(Error::Overflow)?;
        // Reverse accepts every torus word, not only forward images. Prove its
        // largest numerator, doubled remainder and possible upward quotient.
        u128::from(q).checked_mul(u128::from(u64::MAX)).ok_or(Error::Overflow)?;
        (TORUS - 1).checked_mul(2).ok_or(Error::Overflow)?;
        u128::from(q - 1).checked_add(1).ok_or(Error::Overflow)?;
        Ok(Self {
            source,
            quote: Quote {
                input_words: words,
                output_words: words,
                scratch_words: 0,
                caller_bytes,
                descriptor_bytes: std::mem::size_of::<Self>()
                    + std::mem::size_of::<Torus64Ciphertext<'static>>(),
                word_ops,
            },
        })
    }

    pub fn source_descriptor(self) -> Destination {
        self.source
    }

    pub fn quote(self) -> Quote {
        self.quote
    }

    /// Reverse has two live buffers and both borrowed wrappers plus this plan.
    /// The same conservative 64 ops/word +128 covers length/descriptor/cap
    /// preflight, each checked multiply/divide/remainder/tie test, negation/write.
    /// No canonical scan is needed: every u64 is a valid torus component.
    /// Additional retained forward inputs/owners require external accounting.
    pub fn reverse_quote(self) -> Quote {
        Quote {
            descriptor_bytes: self.quote.descriptor_bytes
                + std::mem::size_of::<CoefficientCiphertext<'static>>(),
            ..self.quote
        }
    }

    /// Exact unreduced rational scale in torus units; no f64 conversion.
    pub fn torus_scale(self) -> (u128, u128) {
        // u64 * 2^64 and u64 * q (<2^63) both fit u128.
        (u128::from(self.source.scale.numerator) * TORUS,
         u128::from(self.source.scale.denominator) * u128::from(self.source.limb.modulus))
    }

    /// Checked local work/storage admission, not prepayment or owner authority.
    /// Exact slice lengths required. Input is canonical (b,a), output is (B,A).
    /// Caps cover quoted work/caller buffers, not separately reported wrapper
    /// bytes or cumulative admission; one-short refusal leaves output intact.
    pub fn convert_into<'a>(
        self,
        input: &[u64],
        output: &'a mut [u64],
        max_word_ops: usize,
        max_caller_bytes: usize,
    ) -> Result<Torus64Ciphertext<'a>, Error> {
        if input.len() != self.quote.input_words || output.len() != self.quote.output_words {
            return Err(Error::BufferLength);
        }
        if max_word_ops < self.quote.word_ops || max_caller_bytes < self.quote.caller_bytes {
            return Err(Error::ResourceLimit);
        }
        // Full public-length scan (no short-circuit on invalid components).
        let mut invalid = false;
        for &x in input {
            invalid |= x >= self.source.limb.modulus;
        }
        if invalid {
            return Err(Error::NonCanonical);
        }
        // No remaining fallible operations. Constructor proved arithmetic capacity.
        output[0] = self.round_canonical(input[0]);
        for i in 1..self.quote.output_words {
            output[i] = self.round_canonical(input[i]).wrapping_neg();
        }
        Ok(Torus64Ciphertext { source: self.source, words: output })
    }

    /// Reverse only a view bound to this exact original descriptor, not raw
    /// words with a newly supplied scale/key. All descriptor/length/cap checks
    /// precede writes; safe disjoint caller slices, no scratch or cleanup change.
    /// Caps cover reverse_quote's work and caller buffers; descriptor_bytes is
    /// separately quoted wrapper storage, as in the forward API.
    pub fn reverse_into<'a>(
        self,
        input: &Torus64Ciphertext<'_>,
        output: &'a mut [u64],
        max_word_ops: usize,
        max_caller_bytes: usize,
    ) -> Result<CoefficientCiphertext<'a>, Error> {
        if input.source != self.source {
            return Err(Error::InvalidGeometry);
        }
        let quote = self.reverse_quote();
        if input.words.len() != quote.input_words || output.len() != quote.output_words {
            return Err(Error::BufferLength);
        }
        if max_word_ops < quote.word_ops || max_caller_bytes < quote.caller_bytes {
            return Err(Error::ResourceLimit);
        }
        // Constructor's q<2^63 proves q*u64::MAX <2^127, twice remainder
        // <2^65, and rounded quotient <=q. No fallible operation after writes.
        output[0] = self.round_torus(input.words[0]);
        for i in 1..quote.output_words {
            let rounded = self.round_torus(input.words[i]);
            output[i] = if rounded == 0 { 0 } else { self.source.limb.modulus - rounded };
        }
        Ok(CoefficientCiphertext { source: self.source, words: output })
    }

    fn round_torus(self, x: u64) -> u64 {
        let q = u128::from(self.source.limb.modulus);
        let numerator = q.checked_mul(u128::from(x))
            .expect("torus numerator capacity proved by constructor");
        let quotient = numerator / TORUS;
        let twice_remainder = (numerator % TORUS).checked_mul(2)
            .expect("twice torus remainder fits u128");
        let increment = twice_remainder > TORUS
            || (twice_remainder == TORUS && quotient & 1 != 0);
        let rounded = quotient.checked_add(u128::from(increment))
            .expect("rounded quotient at most q");
        (rounded % q) as u64
    }

    fn round_canonical(self, x: u64) -> u64 {
        let q = self.source.limb.modulus;
        let numerator = u128::from(x).checked_mul(TORUS)
            .and_then(|v| v.checked_add(u128::from(q / 2)))
            .expect("canonical numerator capacity proved by constructor");
        let quotient = numerator / u128::from(q);
        // Explicit reduction handles quotient == 2^64 by mapping it to zero,
        // rather than rejecting a valid modular rounding wrap. Under this slice's
        // q<2^63 and x<q, even x=q-1 stays >2 torus units below 2^64,
        // so that particular quotient is unreachable (zero still maps to zero).
        (quotient % TORUS) as u64
    }
}
