//! Exact clear coefficient algebra from NEXUS §III.B (PDF p.5), Algorithm 1,
//! and its Appendix A proof (pp.15–16). Pinned PDF SHA-256:
//! c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2.
//!
//! Compression places scalars in ascending coefficient order. Decompression
//! executes the proved shift/substitution tree, then multiplies by N^-1 in an
//! odd modular limb. This is NOT CKKS encoding, ciphertext compression, rounding,
//! or a security/numeric profile. No keys, masks, randomness or noise are used.
//! The paper's N' is the ring degree here; packing fewer values zero-pads that
//! ring, never substitutes a smaller automorphism degree. Algorithm 1's printed
//! child indices/loop endpoint are inconsistent; the Appendix A recurrence
//! determines exactly log2(N) stages. Children are written by coefficient residue
//! rather than tree position, implementing the stated final reordering directly.

use crate::coefficient_layout::{CoefficientRing, LayoutError};

/// Fixed work ceiling, in coefficient visits; callers may impose a tighter cap.
pub const MAX_COMPRESSION_WORK: usize = 1 << 28;

/// O(1) address/count algebra, independently usable for source-sized geometries.
/// This does not admit execution at degrees unsupported by `CoefficientRing`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressionGeometry {
    degree: usize,
    scalars: usize,
    polynomials: usize,
    packed_words: usize,
}

impl CompressionGeometry {
    pub fn new(degree: usize, scalars: usize) -> Result<Self, LayoutError> {
        if !degree.is_power_of_two() || scalars == 0 {
            return Err(LayoutError::InvalidGeometry);
        }
        let polynomials = scalars.div_ceil(degree);
        let packed_words = polynomials
            .checked_mul(degree)
            .ok_or(LayoutError::Overflow)?;
        packed_words.checked_mul(8).ok_or(LayoutError::Overflow)?;
        Ok(Self {
            degree,
            scalars,
            polynomials,
            packed_words,
        })
    }

    pub fn polynomial_count(self) -> usize {
        self.polynomials
    }
    pub fn packed_words(self) -> usize {
        self.packed_words
    }

    /// Scalar order remains unchanged across consecutive compressed polynomials.
    pub fn address(self, scalar: usize) -> Result<(usize, usize), LayoutError> {
        if scalar >= self.scalars {
            return Err(LayoutError::Index);
        }
        Ok((scalar / self.degree, scalar % self.degree))
    }

    pub fn valid_coefficients(self, polynomial: usize) -> Result<usize, LayoutError> {
        if polynomial >= self.polynomials {
            return Err(LayoutError::Index);
        }
        Ok((self.scalars - polynomial * self.degree).min(self.degree))
    }
}

/// One compressed polynomial -> N constant polynomials, coefficient ordered.
/// All buffers caller-owned; no allocation. Work and storage bounded at creation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoefficientCompression {
    ring: CoefficientRing,
    output_words: usize,
    work: usize,
}

impl CoefficientCompression {
    pub fn new(ring: CoefficientRing, max_work: usize) -> Result<Self, LayoutError> {
        let n = ring.degree();
        let output_words = n.checked_mul(n).ok_or(LayoutError::Overflow)?;
        // Conservative coefficient-visit bound, including canonical scans,
        // copies, initialization, shared monomial validation and normalization.
        let work = output_words
            .checked_mul(10)
            .and_then(|v| n.checked_mul(4).and_then(|extra| v.checked_add(extra)))
            .ok_or(LayoutError::Overflow)?;
        if work > max_work.min(MAX_COMPRESSION_WORK) {
            return Err(LayoutError::ResourceLimit);
        }
        Ok(Self {
            ring,
            output_words,
            work,
        })
    }

    pub fn output_words(self) -> usize {
        self.output_words
    }
    pub fn scratch_words(self) -> usize {
        2 * self.ring.degree()
    }
    pub fn coefficient_work(self) -> usize {
        self.work
    }

    fn canonical(self, input: &[u64]) -> Result<(), LayoutError> {
        if input.iter().any(|&x| x >= self.ring.modulus()) {
            return Err(LayoutError::NonCanonical);
        }
        Ok(())
    }

    /// Clear polynomial construction, not compression of arbitrary polynomials.
    pub fn compress(self, scalars: &[u64], output: &mut [u64]) -> Result<(), LayoutError> {
        if scalars.is_empty() || scalars.len() > self.ring.degree() {
            return Err(LayoutError::InvalidGeometry);
        }
        if output.len() != self.ring.degree() {
            return Err(LayoutError::BufferLength);
        }
        self.canonical(scalars)?;
        output.fill(0);
        output[..scalars.len()].copy_from_slice(scalars);
        Ok(())
    }

    /// Output polynomial s is (input[s],0,...,0). Every rejection occurs before
    /// either output or scratch changes. Safe borrowing excludes buffer aliasing.
    /// Not constant time. Uses the shared ring's signed monomial and inverse law.
    pub fn decompress(
        self,
        input: &[u64],
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), LayoutError> {
        let n = self.ring.degree();
        if input.len() != n
            || output.len() != self.output_words
            || scratch.len() != self.scratch_words()
        {
            return Err(LayoutError::BufferLength);
        }
        self.canonical(input)?;
        let inverse = self.ring.inverse_power_two(n)?;
        let q = self.ring.modulus();
        output.fill(0);
        output[..n].copy_from_slice(input);
        let (parent, shifted) = scratch.split_at_mut(n);
        let mut h = 1;
        while h < n {
            // Parent s contains coefficients i == s (mod h), shifted by -s.
            // Even child keeps residue s; odd child keeps residue s+h.
            let element = n / h + 1;
            for s in 0..h {
                parent.copy_from_slice(&output[s * n..(s + 1) * n]);
                // Validated geometry/canonical inputs make this infallible.
                self.ring
                    .monomial(parent, -(h as i64), shifted)
                    .expect("validated signed monomial");
                trace_child(parent, &mut output[s * n..(s + 1) * n], element, q);
                trace_child(
                    shifted,
                    &mut output[(s + h) * n..(s + h + 1) * n],
                    element,
                    q,
                );
            }
            h *= 2;
        }
        for value in output {
            *value = (u128::from(*value) * u128::from(inverse) % u128::from(q)) as u64;
        }
        Ok(())
    }
}

// The source substitution is an odd automorphism of Z_q[X]/(X^N+1).
// Its signed address is not a cyclic slot rotation: X^N = -1.
fn trace_child(input: &[u64], output: &mut [u64], element: usize, q: u64) {
    let n = input.len();
    output.copy_from_slice(input);
    for (i, &v) in input.iter().enumerate() {
        let at = (i * element) % (2 * n);
        let signed = if at >= n && v != 0 { q - v } else { v };
        output[at % n] = (output[at % n] + signed) % q;
    }
}

#[cfg(test)]
#[path = "coefficient_compression/tests.rs"]
mod tests;
