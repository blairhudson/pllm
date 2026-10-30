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
pub mod coordinates;
pub(crate) mod modular;
pub mod schedule;
pub(crate) mod trace;

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
        if v == 0 { 0 } else { self.modulus - v }
    }

    /// Exact inverse in the odd coefficient ring, never ring-2^ell division.
    pub fn inverse_power_two(self, divisor: usize) -> Result<u64, LayoutError> {
        if !divisor.is_power_of_two() || divisor > self.degree {
            return Err(LayoutError::InvalidGeometry);
        }
        // Same repeated inverse-of-two law as coefficient_repacking; no prime
        // assumption is needed (odd composite moduli also work).
        Ok(trace::inverse_power_two(self.modulus, divisor))
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
        Ok(schedule::InterleaveSchedule::bounded(self.degree, stride, count, MAX_DEGREE)?.padded())
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
        trace::interleave(
            self.degree,
            self.modulus,
            MAX_DEGREE,
            input,
            stride,
            count,
            output,
            scratch,
        )
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
    fn coordinates(self) -> Result<coordinates::Coordinates, LayoutError> {
        coordinates::Coordinates::bounded(self.ring.degree, self.shape, self.window, MAX_DEGREE)
    }
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
        for at in self.coordinates()?.lhs(row_block, inner_block)? {
            let v = matrix[at.matrix];
            output[at.coefficient] = if at.negative { self.ring.neg(v) } else { v };
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
        for at in self.coordinates()?.rhs(inner_block, col_block)? {
            output[at.coefficient] = matrix[at.matrix];
        }
        Ok(())
    }

    /// (packed polynomial, coefficient) for a global output matrix coordinate.
    pub fn packed_address(self, row: usize, col: usize) -> Result<(usize, usize), LayoutError> {
        if row >= self.shape[0] || col >= self.shape[2] {
            return Err(LayoutError::Index);
        }
        self.coordinates()?.address(row, col)
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
