//! Exact rounded removal of one odd coefficient limb, using caller-owned buffers.
//! Public, variable-time algebra only: no transforms, ciphertexts or secret issuer.

use super::modular::{InverseError, inverse, mul, sub};
use super::{CoefficientRing, LayoutError};

/// One retained modulus q and one removed modulus p, of equal degree.
/// Coprime odd composites are legal; no primality, roots or HE profile is implied.
/// Construction and application allocate no storage and retain no buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoundedLimbDrop {
    retained: CoefficientRing,
    removed: CoefficientRing,
    inverse_p: u64,
}

impl RoundedLimbDrop {
    /// Uses validated ring bounds (legacy 4096, or explicitly public 8192).
    /// Degree mismatch is InvalidDegree; noncoprime moduli are InvalidModulus.
    /// The extracted inverse's iteration-cap failure maps to Overflow.
    pub fn new(
        retained: CoefficientRing,
        removed: CoefficientRing,
    ) -> Result<Self, LayoutError> {
        if retained.degree() != removed.degree() {
            return Err(LayoutError::InvalidDegree);
        }
        let inverse_p = inverse(removed.modulus() % retained.modulus(), retained.modulus())
            .map_err(|error| match error {
                InverseError::NotCoprime => LayoutError::InvalidModulus,
                InverseError::Capacity => LayoutError::Overflow,
            })?;
        Ok(Self { retained, removed, inverse_p })
    }

    /// For x in [0,p*q) represented by (vq,vp), returns
    /// floor((x+(p-1)/2)/p) mod q coefficientwise. Odd p excludes half ties.
    /// All three lengths and both complete canonical inputs are checked before
    /// any write. Atomicity covers this one retained-limb projection only:
    /// callers projecting several limbs must preflight the whole operation.
    pub fn apply_into(
        self,
        vq: &[u64],
        vp: &[u64],
        output: &mut [u64],
    ) -> Result<(), LayoutError> {
        let degree = self.retained.degree();
        if vq.len() != degree || vp.len() != degree || output.len() != degree {
            return Err(LayoutError::BufferLength);
        }
        self.retained.canonical(vq)?;
        self.removed.canonical(vp)?;
        for ((&aq, &ap), out) in vq.iter().zip(vp).zip(output) {
            *out = self.project(aq, ap);
        }
        Ok(())
    }

    // Infallible for canonical residues under this validated descriptor.
    fn project(self, aq: u64, ap: u64) -> u64 {
        let q = self.retained.modulus();
        let p = self.removed.modulus();
        let half = (p - 1) / 2;
        // Reduce the centered remainder BEFORE subtraction: p may exceed q.
        // Negative center is represented modulo q without a signed cast.
        let center_q = if ap <= half {
            ap % q
        } else {
            sub(0, (p - ap) % q, q)
        };
        mul(sub(aq, center_q, q), self.inverse_p, q)
    }
}

/// Named semantic counts for one construction plus one application.
/// These are not CPU tariffs, prepaid/cumulative funding, or total execution
/// bounds. Reusing a chain retains a conservative per-call quote only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DropChainWork {
    pub degree_checks: usize,
    pub pair_checks: usize,
    pub canonical_checks: usize,
    pub projections: usize,
    pub word_stores: usize,
}

/// Required two-component payload prefixes and borrowed descriptor storage.
/// Payload bytes exclude excess caller capacity, slice headers, stack storage,
/// and allocator overhead; descriptor bytes are quoted separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DropChainGeometry {
    pub degree: usize,
    pub input_limbs: usize,
    pub retained_limbs: usize,
    pub steps: usize,
    pub input_words: usize,
    pub output_words: usize,
    pub scratch_words: usize,
    pub payload_bytes: usize,
    pub descriptor_bytes: usize,
}

/// Allocation-free public coefficient rounding, removing trailing limbs.
/// Pairs use triangular order j*(j-1)/2+i for every i<j, including pairs
/// between retained limbs. Ring order fixes the removal order, last first.
/// No keys, trust/issuance state, ciphertext semantics, or HE capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoundedDropChain<'a> {
    rings: &'a [CoefficientRing],
    pairs: &'a [RoundedLimbDrop],
    geometry: DropChainGeometry,
    work: DropChainWork,
}

// Divide an even factor first; avoid overflowing an otherwise fitting half
// product. Callers establish that at least one factor is even.
fn checked_half_product(a: usize, b: usize) -> Result<usize, LayoutError> {
    let (a, b) = if a % 2 == 0 { (a / 2, b) } else { (a, b / 2) };
    a.checked_mul(b).ok_or(LayoutError::Overflow)
}

// Pure count helper also permits overflow tests without fabricated huge slices.
fn chain_quote(
    m: usize,
    degree: usize,
    steps: usize,
) -> Result<(DropChainGeometry, DropChainWork), LayoutError> {
    if m < 2 || steps == 0 || steps >= m {
        return Err(LayoutError::InvalidGeometry);
    }
    let r = m - steps;
    let pairs = checked_half_product(m, m - 1)?;
    let stage_sum = (m - 1).checked_add(r).ok_or(LayoutError::Overflow)?;
    let projections_per_coefficient = checked_half_product(steps, stage_sum)?;
    let input_words = m.checked_mul(degree)
        .and_then(|v| v.checked_mul(2)).ok_or(LayoutError::Overflow)?;
    let output_words = r.checked_mul(degree)
        .and_then(|v| v.checked_mul(2)).ok_or(LayoutError::Overflow)?;
    let projections = projections_per_coefficient.checked_mul(degree)
        .and_then(|v| v.checked_mul(2)).ok_or(LayoutError::Overflow)?;
    let word_stores = input_words.checked_add(projections)
        .and_then(|v| v.checked_add(output_words)).ok_or(LayoutError::Overflow)?;
    let payload_bytes = input_words.checked_add(input_words)
        .and_then(|v| v.checked_add(output_words))
        .and_then(|v| v.checked_mul(size_of::<u64>())).ok_or(LayoutError::Overflow)?;
    let descriptor_bytes = m.checked_mul(size_of::<CoefficientRing>())
        .and_then(|v| pairs.checked_mul(size_of::<RoundedLimbDrop>())
            .and_then(|p| v.checked_add(p))).ok_or(LayoutError::Overflow)?;
    Ok((DropChainGeometry {
        degree,
        input_limbs: m,
        retained_limbs: r,
        steps,
        input_words,
        output_words,
        scratch_words: input_words,
        payload_bytes,
        descriptor_bytes,
    }, DropChainWork {
        degree_checks: m,
        pair_checks: pairs,
        canonical_checks: input_words,
        projections,
        word_stores,
    }))
}

// Every row and row+i are bounded by the checked triangular pair count.
fn pair_row(j: usize) -> usize {
    if j % 2 == 0 { (j / 2) * (j - 1) } else { j * ((j - 1) / 2) }
}

impl<'a> RoundedDropChain<'a> {
    /// Check geometry and every semantic ceiling before scanning descriptors.
    /// Existing validated ring degrees apply; no extra limb-count/prime gate.
    pub fn new(
        rings: &'a [CoefficientRing],
        pairs: &'a [RoundedLimbDrop],
        steps: usize,
        ceiling: DropChainWork,
    ) -> Result<Self, LayoutError> {
        let m = rings.len();
        if m < 2 || steps == 0 || steps >= m {
            return Err(LayoutError::InvalidGeometry);
        }
        let (geometry, work) = chain_quote(m, rings[0].degree(), steps)?;
        if work.degree_checks > ceiling.degree_checks
            || work.pair_checks > ceiling.pair_checks
            || work.canonical_checks > ceiling.canonical_checks
            || work.projections > ceiling.projections
            || work.word_stores > ceiling.word_stores
        {
            return Err(LayoutError::ResourceLimit);
        }
        if pairs.len() != work.pair_checks {
            return Err(LayoutError::BufferLength);
        }
        for ring in rings {
            if ring.degree() != geometry.degree {
                return Err(LayoutError::InvalidDegree);
            }
        }
        for j in 1..m {
            for i in 0..j {
                let pair = pairs[pair_row(j) + i];
                if pair.retained.degree() != rings[i].degree()
                    || pair.removed.degree() != rings[j].degree()
                {
                    return Err(LayoutError::InvalidDegree);
                }
                if pair.retained.modulus() != rings[i].modulus()
                    || pair.removed.modulus() != rings[j].modulus()
                {
                    return Err(LayoutError::InvalidModulus);
                }
            }
        }
        Ok(Self { rings, pairs, geometry, work })
    }

    pub fn geometry(&self) -> DropChainGeometry {
        self.geometry
    }

    pub fn work(&self) -> DropChainWork {
        self.work
    }

    /// Component-major, then limb-major: component*m*N + limb*N + coefficient.
    /// Exact input/output lengths and sufficient scratch are checked first,
    /// then both entire canonical inputs, before either mutable buffer changes.
    /// Success touches only the required scratch prefix. After its first write,
    /// all scalar projections and copies are infallible under checked geometry.
    pub fn apply_into(
        &self,
        input: &[u64],
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), LayoutError> {
        let g = self.geometry;
        if input.len() != g.input_words || output.len() != g.output_words
            || scratch.len() < g.scratch_words
        {
            return Err(LayoutError::BufferLength);
        }
        // All offsets below are bounded by checked input_words/output_words;
        // degree is nonzero and descriptors cannot change through shared borrows.
        let component_words = g.input_words / 2;
        let retained_words = g.output_words / 2;
        for component in input.chunks_exact(component_words) {
            for (ring, limb) in self.rings.iter().zip(component.chunks_exact(g.degree)) {
                ring.canonical(limb)?;
            }
        }
        let scratch = &mut scratch[..g.scratch_words];
        scratch.copy_from_slice(input);
        for component in scratch.chunks_exact_mut(component_words) {
            for j in (g.retained_limbs..g.input_limbs).rev() {
                let (retained, removed) = component.split_at_mut(j * g.degree);
                // This limb includes earlier stages, but stays immutable during
                // its own stage while every lower limb is updated in place.
                let removed = &removed[..g.degree];
                for (i, limb) in retained.chunks_exact_mut(g.degree).enumerate() {
                    let pair = self.pairs[pair_row(j) + i];
                    for (value, &residue) in limb.iter_mut().zip(removed) {
                        *value = pair.project(*value, residue);
                    }
                }
            }
        }
        output[..retained_words].copy_from_slice(&scratch[..retained_words]);
        output[retained_words..].copy_from_slice(
            &scratch[component_words..component_words + retained_words],
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod chain_tests;
