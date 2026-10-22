//! Bounded, exact coefficient-ring algebra with caller-owned storage.
//!
//! This is not an HE backend or an executable secure-multiplication capability.
//! No keys, encryption, H2A, OT, noise, rounding, or numerical profile are supplied.
//! The inverse-power-of-two and negacyclic trace/merge algebra extends the existing
//! `coefficient_repacking` primitives without changing their serialized API.
//! Matrix indexing uses the explicitly approved paper-derived corrections, not
//! an assertion of author errata; see `docs/plans/bumblebee-indexing-corrections.md`.

/// Preserve the existing coefficient-repacking degree and input-count bounds.
pub const MAX_DEGREE: usize = 4096;
pub const MAX_INTERLEAVE_INPUTS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutError {
    InvalidDegree,
    InvalidModulus,
    InvalidGeometry,
    Overflow,
    ResourceLimit,
    BufferLength,
    NonCanonical,
    Index,
}

/// One odd coefficient-modulus limb, not a plaintext or security parameter set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoefficientRing {
    degree: usize,
    modulus: u64,
}

impl CoefficientRing {
    pub fn new(degree: usize, modulus: u64) -> Result<Self, LayoutError> {
        if !degree.is_power_of_two() || degree > MAX_DEGREE {
            return Err(LayoutError::InvalidDegree);
        }
        // Same limb bound as coefficient_repacking::divide_power_two.
        if !(3..(1_u64 << 61)).contains(&modulus) || modulus & 1 == 0 {
            return Err(LayoutError::InvalidModulus);
        }
        Ok(Self { degree, modulus })
    }

    pub fn degree(self) -> usize {
        self.degree
    }

    pub fn modulus(self) -> u64 {
        self.modulus
    }

    fn canonical(self, values: &[u64]) -> Result<(), LayoutError> {
        if values.iter().any(|&v| v >= self.modulus) {
            return Err(LayoutError::NonCanonical);
        }
        Ok(())
    }

    fn neg(self, v: u64) -> u64 {
        if v == 0 {
            0
        } else {
            self.modulus - v
        }
    }

    fn add(self, a: u64, b: u64) -> u64 {
        (a + b) % self.modulus
    }

    fn sub(self, a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            self.modulus - (b - a)
        }
    }

    fn mul(self, a: u64, b: u64) -> u64 {
        (u128::from(a) * u128::from(b) % u128::from(self.modulus)) as u64
    }

    /// Exact inverse in the odd coefficient ring, never ring-2^ell division.
    pub fn inverse_power_two(self, divisor: usize) -> Result<u64, LayoutError> {
        if !divisor.is_power_of_two() || divisor > self.degree {
            return Err(LayoutError::InvalidGeometry);
        }
        // Same repeated inverse-of-two law as coefficient_repacking; no prime
        // assumption is needed (odd composite moduli also work).
        let mut inverse = 1;
        for _ in 0..divisor.trailing_zeros() {
            inverse = self.mul(inverse, self.modulus.div_ceil(2));
        }
        Ok(inverse)
    }

    /// Multiply by X^exponent modulo X^N+1, including negative exponents.
    /// Every error is detected before any output is changed.
    pub fn monomial(
        self,
        input: &[u64],
        exponent: i64,
        output: &mut [u64],
    ) -> Result<(), LayoutError> {
        if input.len() != self.degree || output.len() != self.degree {
            return Err(LayoutError::BufferLength);
        }
        self.canonical(input)?;
        let shift = exponent.rem_euclid((2 * self.degree) as i64) as usize;
        for (i, &v) in input.iter().enumerate() {
            let at = i + shift;
            output[at % self.degree] = if at / self.degree % 2 == 0 {
                v
            } else {
                self.neg(v)
            };
        }
        Ok(())
    }

    fn interleave_geometry(self, stride: usize, count: usize) -> Result<usize, LayoutError> {
        if !stride.is_power_of_two()
            || stride > self.degree
            || count == 0
            || count > stride.min(MAX_INTERLEAVE_INPUTS)
        {
            return Err(LayoutError::InvalidGeometry);
        }
        Ok(count.next_power_of_two())
    }

    /// Scratch contains padded input polynomials plus one temporary polynomial.
    /// No allocation, retention, or buffer ownership transfer occurs.
    pub fn interleave_scratch_words(
        self,
        stride: usize,
        count: usize,
    ) -> Result<usize, LayoutError> {
        Ok((self.interleave_geometry(stride, count)? + 1) * self.degree)
    }

    /// Optimized FFT-style trace/merge, retaining input j at lane j:
    /// `output[k*stride+j] = input[j*N+k*stride]`; absent lanes are zero.
    /// All other input coefficients are removed, even for count=1.
    /// Inputs are exact ring polynomials, NOT ciphertexts or H2A shares.
    /// Safe Rust's exclusive mutable borrows require output/scratch to be disjoint
    /// from input and each other; sizes below account for those distinct buffers.
    pub fn interleave(
        self,
        input: &[u64],
        stride: usize,
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), LayoutError> {
        let padded = self.interleave_geometry(stride, count)?;
        if input.len() != count * self.degree
            || output.len() != self.degree
            || scratch.len() != (padded + 1) * self.degree
        {
            return Err(LayoutError::BufferLength);
        }
        self.canonical(input)?;
        let inverse = self.inverse_power_two(stride)?;
        let (values, temp) = scratch.split_at_mut(padded * self.degree);
        values.fill(0);
        for (v, &a) in values.iter_mut().zip(input) {
            *v = self.mul(a, inverse);
        }
        // Partial groups first trace from padded lanes up to the full stride.
        // This is the same generalized trace used by coefficient_repacking.py.
        for value in values.chunks_exact_mut(self.degree) {
            let mut h = padded;
            while h < stride {
                temp.copy_from_slice(value);
                self.add_automorphism(value, temp, self.degree / h + 1);
                h *= 2;
            }
        }
        let mut h = padded / 2;
        while h != 0 {
            let (left, right) = values.split_at_mut(h * self.degree);
            for (a, b) in left
                .chunks_exact_mut(self.degree)
                .zip(right.chunks_exact(self.degree))
            {
                for i in 0..self.degree {
                    let shifted = if i >= h {
                        b[i - h]
                    } else {
                        self.neg(b[self.degree + i - h])
                    };
                    temp[i] = self.sub(a[i], shifted);
                    a[i] = self.add(a[i], shifted);
                }
                self.add_automorphism(a, temp, self.degree / h + 1);
            }
            h /= 2;
        }
        output.copy_from_slice(&values[..self.degree]);
        Ok(())
    }

    fn add_automorphism(self, output: &mut [u64], input: &[u64], element: usize) {
        for (i, &v) in input.iter().enumerate() {
            let at = (i * element) % (2 * self.degree);
            let v = if at < self.degree { v } else { self.neg(v) };
            output[at % self.degree] = self.add(output[at % self.degree], v);
        }
    }
}

/// Conservative, simultaneously-live storage for a streaming block evaluator.
/// Every word is caller-owned u64 storage. Descriptor owns no heap storage.
/// Group holds accumulated products; scratch does not alias it. Packed output
/// includes the currently written group, so no additional N-word output is charged.
/// This is algebra workspace only, NOT a complete HE/key/transport resource price.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayoutStorage {
    pub input_matrix_words: usize,
    pub output_matrix_words: usize,
    pub encoding_words: usize,
    pub group_words: usize,
    pub interleave_scratch_words: usize,
    pub packed_words: usize,
    pub caller_bytes: usize,
    pub descriptor_bytes: usize,
}

/// Immutable row-major matrix/block geometry. Matrix shape is (k,m,n), window
/// is (kw,mw,nw); edge blocks are zero padded on encoding and cropped on parsing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoefficientLayout {
    ring: CoefficientRing,
    shape: [usize; 3],
    window: [usize; 3],
    blocks: [usize; 3],
    groups: usize,
    storage: LayoutStorage,
}

fn product(a: usize, b: usize) -> Result<usize, LayoutError> {
    a.checked_mul(b).ok_or(LayoutError::Overflow)
}

impl CoefficientLayout {
    /// Explicit caller-byte admission, not an allocation. Degree <=4096 and
    /// mw<=256 preserve existing repacking bounds; no HE capacity is increased.
    pub fn new(
        ring: CoefficientRing,
        shape: [usize; 3],
        window: [usize; 3],
        max_caller_bytes: usize,
    ) -> Result<Self, LayoutError> {
        let [k, m, n] = shape;
        let [kw, mw, nw] = window;
        if shape.contains(&0)
            || window.contains(&0)
            || !mw.is_power_of_two()
            || mw > MAX_INTERLEAVE_INPUTS
        {
            return Err(LayoutError::InvalidGeometry);
        }
        if product(product(kw, mw)?, nw)? > ring.degree {
            return Err(LayoutError::InvalidGeometry);
        }
        let blocks = [k.div_ceil(kw), m.div_ceil(mw), n.div_ceil(nw)];
        let count = product(blocks[0], blocks[2])?;
        let groups = count.div_ceil(mw);
        let mut storage = LayoutStorage {
            input_matrix_words: product(k, m)?
                .checked_add(product(m, n)?)
                .ok_or(LayoutError::Overflow)?,
            output_matrix_words: product(k, n)?,
            encoding_words: product(2, ring.degree)?,
            group_words: product(mw, ring.degree)?,
            interleave_scratch_words: ring.interleave_scratch_words(mw, count.min(mw))?,
            packed_words: product(groups, ring.degree)?,
            caller_bytes: 0,
            descriptor_bytes: std::mem::size_of::<Self>(),
        };
        let words = [
            storage.input_matrix_words,
            storage.output_matrix_words,
            storage.encoding_words,
            storage.group_words,
            storage.interleave_scratch_words,
            storage.packed_words,
        ]
        .into_iter()
        .try_fold(0_usize, |a, b| {
            a.checked_add(b).ok_or(LayoutError::Overflow)
        })?;
        storage.caller_bytes = product(words, std::mem::size_of::<u64>())?;
        if storage.caller_bytes > max_caller_bytes || storage.caller_bytes > isize::MAX as usize {
            return Err(LayoutError::ResourceLimit);
        }
        Ok(Self {
            ring,
            shape,
            window,
            blocks,
            groups,
            storage,
        })
    }

    pub fn ring(self) -> CoefficientRing {
        self.ring
    }

    pub fn shape(self) -> [usize; 3] {
        self.shape
    }

    pub fn window(self) -> [usize; 3] {
        self.window
    }

    pub fn blocks(self) -> [usize; 3] {
        self.blocks
    }

    pub fn storage(self) -> LayoutStorage {
        self.storage
    }

    pub fn group_count(self) -> usize {
        self.groups
    }

    /// Valid (non-padding) inputs in one interleaving group.
    pub fn group_inputs(self, group: usize) -> Result<usize, LayoutError> {
        if group >= self.groups {
            return Err(LayoutError::Index);
        }
        Ok((self.blocks[0] * self.blocks[2] - group * self.window[1]).min(self.window[1]))
    }

    /// Qhat = sum Q[r,h] X^(r*nw*mw-h), reduced modulo X^N+1.
    pub fn pack_lhs(
        self,
        matrix: &[u64],
        row_block: usize,
        inner_block: usize,
        output: &mut [u64],
    ) -> Result<(), LayoutError> {
        if matrix.len() != self.shape[0] * self.shape[1] || output.len() != self.ring.degree {
            return Err(LayoutError::BufferLength);
        }
        if row_block >= self.blocks[0] || inner_block >= self.blocks[1] {
            return Err(LayoutError::Index);
        }
        self.ring.canonical(matrix)?;
        output.fill(0);
        let [kw, mw, nw] = self.window;
        let row_start = row_block * kw;
        let inner_start = inner_block * mw;
        for r in 0..kw.min(self.shape[0] - row_start) {
            for h in 0..mw.min(self.shape[1] - inner_start) {
                let v = matrix[(row_start + r) * self.shape[1] + inner_start + h];
                let base = r * nw * mw;
                if base >= h {
                    output[base - h] = v;
                } else {
                    output[self.ring.degree - (h - base)] = self.ring.neg(v);
                }
            }
        }
        Ok(())
    }

    /// Vhat = sum V[h,c] X^(c*mw+h), with zero-padded edge blocks.
    pub fn pack_rhs(
        self,
        matrix: &[u64],
        inner_block: usize,
        col_block: usize,
        output: &mut [u64],
    ) -> Result<(), LayoutError> {
        if matrix.len() != self.shape[1] * self.shape[2] || output.len() != self.ring.degree {
            return Err(LayoutError::BufferLength);
        }
        if inner_block >= self.blocks[1] || col_block >= self.blocks[2] {
            return Err(LayoutError::Index);
        }
        self.ring.canonical(matrix)?;
        output.fill(0);
        let [_, mw, nw] = self.window;
        let inner_start = inner_block * mw;
        let col_start = col_block * nw;
        for h in 0..mw.min(self.shape[1] - inner_start) {
            for c in 0..nw.min(self.shape[2] - col_start) {
                output[c * mw + h] = matrix[(inner_start + h) * self.shape[2] + col_start + c];
            }
        }
        Ok(())
    }

    /// (packed polynomial, coefficient) for a global output matrix coordinate.
    pub fn packed_address(self, row: usize, col: usize) -> Result<(usize, usize), LayoutError> {
        if row >= self.shape[0] || col >= self.shape[2] {
            return Err(LayoutError::Index);
        }
        let [kw, mw, nw] = self.window;
        let t = (row / kw) * self.blocks[2] + col / nw;
        Ok((t / mw, ((row % kw) * nw + col % nw) * mw + t % mw))
    }

    /// ParseMat with global t=alpha*n_prime+gamma and unrotated lane t%mw.
    /// Packed polynomials are concatenated. Padded lanes/rows/columns are ignored.
    pub fn parse(self, packed: &[u64], output: &mut [u64]) -> Result<(), LayoutError> {
        if packed.len() != self.storage.packed_words
            || output.len() != self.storage.output_matrix_words
        {
            return Err(LayoutError::BufferLength);
        }
        self.ring.canonical(packed)?;
        for row in 0..self.shape[0] {
            for col in 0..self.shape[2] {
                let (group, coefficient) = self.packed_address(row, col)?;
                output[row * self.shape[2] + col] = packed[group * self.ring.degree + coefficient];
            }
        }
        Ok(())
    }
}

/// Algebraic expansion only: each tuple names (lhs share owner, rhs share owner).
/// Local terms stay with their owner; cross terms require two independently
/// implemented private products. No OLT execution or security is supplied here.
/// Source: final paper section III, p.5 explicitly names Q1*V0 and Q0*V1.
pub struct TwoPartyProductPlan;

impl TwoPartyProductPlan {
    pub const LOCAL_TERMS: [(u8, u8); 2] = [(0, 0), (1, 1)];
    pub const CROSS_TERMS: [(u8, u8); 2] = [(1, 0), (0, 1)];
}

#[cfg(test)]
#[path = "coefficient_layout/tests.rs"]
mod tests;
