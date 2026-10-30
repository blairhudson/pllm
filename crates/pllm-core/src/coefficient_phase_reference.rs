//! Clear-only packed linear-region phase reference. Not an HE implementation.
//!
//! Observed floor-whole U(m)=floor(q*m/2^64), canonical full-width plaintext
//! multiplication, exact D21 trace/merge and global CRT/Down. No nearest-Up,
//! Delta substitution, keys, sampler, encryption noise or private Share values.
//! The existing coefficient_conversion APIs and odd layout defaults are unchanged.
//! Source observations: target/bumblebee-pinned-phase-astra/observations.md;
//! proposed limb tuple: docs/plans/bumblebee-executable-profile-d37.md B1.
//!
//! Allocation-free kernel: all storage belongs to the caller. Native ownership
//! adapter must reserve the complete quote on its ORIGINAL budget before factories.
//! Work is consumed cumulatively before any value-dependent validation or loop.

pub use crate::coefficient_layout::coordinates;
#[path = "coefficient_phase_reference/wide.rs"]
pub mod wide;
#[path = "coefficient_phase_reference/work.rs"]
pub mod work;
use crate::coefficient_layout::TwoPartyProductPlan;
use coordinates::Coordinates;
pub use wide::{Rns3, Wide};
use crate::coefficient_layout::modular::{mul, sub};

pub const DEGREE: usize = 8192;
pub const SHAPE: [usize; 3] = [3, 65, 129];
pub const WINDOW: [usize; 3] = [1, 64, 128];
pub const LIMBS: [u64; 3] = [562949952847873, 562949952798721, 562949952700417];
pub const MAX_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_WORK: u64 = 1 << 28;
pub const CELLS: usize = SHAPE[0] * SHAPE[2];
pub const SOURCE_WORDS: usize = 2 * (SHAPE[0] * SHAPE[1] + SHAPE[1] * SHAPE[2]);
/// Six products, two encodings, two public support-index planes and nine trace
/// polynomials per limb. No coordinate iterator runs inside a convolution pair.
pub const SCRATCH_WORDS: usize = 3 * 19 * DEGREE;
pub const PHASE_WORDS: usize = 2 * 3 * DEGREE;
pub const CONVERTED_WORDS: usize = 2 * 2 * DEGREE;
/// Four cross outputs followed by two final party-local product outputs.
pub const MATRIX_WORDS: usize = 6 * CELLS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Geometry,
    Overflow,
    Capacity,
    Unsupported,
    Resource,
    BufferLength,
    NonCanonical,
    UnknownPhaseError,
}
impl From<crate::coefficient_layout::LayoutError> for Error {
    fn from(e: crate::coefficient_layout::LayoutError) -> Self {
        use crate::coefficient_layout::LayoutError as L;
        match e {
            L::Overflow => Self::Overflow,
            L::BufferLength => Self::BufferLength,
            L::NonCanonical => Self::NonCanonical,
            L::ResourceLimit => Self::Resource,
            _ => Self::Geometry,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CryptoError {
    ExplicitZero,
    Unknown,
}

/// One non-renewing arithmetic-work counter for all admitted clear calls.
/// Units count bounded word/u128 arithmetic, not top-level arbitrary-wide calls.
#[derive(Debug)]
pub struct Work {
    limit: u64,
    used: u64,
}
impl Work {
    pub fn new(limit: u64) -> Result<Self, Error> {
        if limit > MAX_WORK {
            return Err(Error::Resource);
        }
        Ok(Self { limit, used: 0 })
    }
    pub fn used(&self) -> u64 {
        self.used
    }
    pub fn consume(&mut self, units: u64) -> Result<(), Error> {
        let next = self.used.checked_add(units).ok_or(Error::Overflow)?;
        if next > self.limit {
            return Err(Error::Resource);
        }
        self.used = next;
        Ok(())
    }
}

/// Named complete-kernel prequote, including the independent dense oracle budget.
/// Counts two directions, three limbs, both inner blocks and all padded traces.
/// ```compile_fail
/// use pllm_core::coefficient_phase_reference::Quote;
/// let mut q = Quote::qualification().unwrap();
/// q.total_bytes = 0;
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Quote {
    source_bytes: usize,
    mask_bytes: usize,
    scratch_bytes: usize,
    retained_bytes: usize,
    control_bytes: usize,
    total_bytes: usize,
    convolution_pairs: u64,
    inventory: work::Inventory,
}
impl Quote {
    pub fn source_bytes(&self) -> usize {
        self.source_bytes
    }
    pub fn mask_bytes(&self) -> usize {
        self.mask_bytes
    }
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
    pub fn control_bytes(&self) -> usize {
        self.control_bytes
    }
    pub fn convolution_pairs(&self) -> u64 {
        self.convolution_pairs
    }
    pub fn arithmetic(&self) -> u64 {
        self.inventory.total()
    }
    pub fn inventory(&self) -> &work::Inventory {
        &self.inventory
    }
    pub fn qualification() -> Result<Self, Error> {
        let inventory = work::Inventory::qualification()?;
        let pairs = inventory.convolution_pairs();
        let source_bytes = SOURCE_WORDS * 8;
        let mask_bytes = 2 * DEGREE * std::mem::size_of::<Wide>();
        let scratch_bytes = SCRATCH_WORDS * 8;
        let retained_bytes = (PHASE_WORDS + CONVERTED_WORDS + MATRIX_WORDS) * 8;
        // Named fixed upper envelope for descriptors, Wide/CRT temporaries,
        // iterator/control slots and native constructor/Result transition slots.
        // Adapter rechecks its concrete inline inventory against this envelope.
        let control_bytes = 64 * 1024;
        let total_bytes =
            source_bytes + mask_bytes + scratch_bytes + retained_bytes + control_bytes;
        if total_bytes > MAX_BYTES || inventory.total() > MAX_WORK {
            return Err(Error::Resource);
        }
        Ok(Self {
            source_bytes,
            mask_bytes,
            scratch_bytes,
            retained_bytes,
            control_bytes,
            total_bytes,
            convolution_pairs: pairs,
            inventory,
        })
    }
    pub fn admit(self, bytes: usize, work: &mut Work) -> Result<(), Error> {
        if bytes < self.total_bytes || self.total_bytes > MAX_BYTES {
            return Err(Error::Resource);
        }
        work.consume(self.inventory.total())
    }
}

/// All values are immutable, caller-supplied CLEAR integers. No freshness claim.
pub struct Inputs<'a> {
    pub lhs: [&'a [u64]; 2],
    pub rhs: [&'a [u64]; 2],
    /// Complete canonical R_q masks, in TwoPartyProductPlan::CROSS_TERMS order.
    pub masks: [&'a [Wide]; 2],
    pub crypto_error: CryptoError,
}
impl Inputs<'_> {
    pub fn validate_lengths(&self) -> Result<(), Error> {
        if self.lhs.iter().any(|s| s.len() != SHAPE[0] * SHAPE[1])
            || self.rhs.iter().any(|s| s.len() != SHAPE[1] * SHAPE[2])
            || self.masks.iter().any(|s| s.len() != DEGREE)
        {
            return Err(Error::BufferLength);
        }
        if self.crypto_error != CryptoError::ExplicitZero {
            return Err(Error::UnknownPhaseError);
        }
        Ok(())
    }
}

pub struct Buffers<'a> {
    pub scratch: &'a mut [u64],
    /// Unmasked actual packed-product phases, direction/limb/coefficient order.
    pub phases: &'a mut [u64],
    /// Exact Down after complete CRT: direction/(receiver,sender)/coefficient.
    pub converted: &'a mut [u64],
    /// Cross receiver/sender matrices in plan order, then final P0/P1 matrices.
    pub matrices: &'a mut [u64],
}

/// Pre-admitted, single-use kernel permit. No storage allocation occurs here.
pub struct Admitted {
    rns: Rns3,
    coordinates: Coordinates,
}
impl Admitted {
    pub fn qualification(bytes: usize, work: &mut Work, error: CryptoError) -> Result<Self, Error> {
        if error != CryptoError::ExplicitZero {
            return Err(Error::UnknownPhaseError);
        }
        Quote::qualification()?.admit(bytes, work)?;
        let rns = Rns3::new(LIMBS)?;
        // Verify the document's exact 147-bit product, not merely bit length.
        if rns.modulus() != Wide([0xfc73a11bdfe2c001, 0xff8b000c0237bf8a, 0x7ffff, 0]) {
            return Err(Error::Unsupported);
        }
        Self::with_checked_product(rns)
    }
    fn with_checked_product(rns: Rns3) -> Result<Self, Error> {
        // Conservative whole-polynomial residual bound:
        // |t*E| < 2*N*t^2; require 16*(2*N*t^2) < q.
        if rns.modulus() <= Wide::from((32 * DEGREE) as u64).checked_shl(128)? {
            return Err(Error::Unsupported);
        }
        Ok(Self {
            rns,
            coordinates: Coordinates::bounded(DEGREE, SHAPE, WINDOW, DEGREE)?,
        })
    }
    /// Call after admission but before allocating native workspace.
    pub fn validate_inputs(&self, input: &Inputs<'_>) -> Result<(), Error> {
        input.validate_lengths()?;
        if input
            .masks
            .iter()
            .flat_map(|m| m.iter())
            .any(|&x| x >= self.rns.modulus())
        {
            return Err(Error::NonCanonical);
        }
        Ok(())
    }
    pub fn evaluate(self, input: &Inputs<'_>, buffers: Buffers<'_>) -> Result<(), Error> {
        input.validate_lengths()?;
        if buffers.scratch.len() != SCRATCH_WORDS
            || buffers.phases.len() != PHASE_WORDS
            || buffers.converted.len() != CONVERTED_WORDS
            || buffers.matrices.len() != MATRIX_WORDS
        {
            return Err(Error::BufferLength);
        }
        // Reject all masks before changing any destination.
        self.validate_inputs(input)?;
        let Buffers {
            scratch,
            phases,
            converted,
            matrices,
        } = buffers;
        for (direction, (lhs_owner, rhs_owner)) in
            TwoPartyProductPlan::CROSS_TERMS.into_iter().enumerate()
        {
            let lhs = input.lhs[lhs_owner as usize];
            let rhs = input.rhs[rhs_owner as usize];
            for (limb, p) in LIMBS.into_iter().enumerate() {
                let plane = &mut scratch[limb * 19 * DEGREE..(limb + 1) * 19 * DEGREE];
                let (group, tail) = plane.split_at_mut(6 * DEGREE);
                let (encoding, trace) = tail.split_at_mut(4 * DEGREE);
                let (packed, indices) = encoding.split_at_mut(2 * DEGREE);
                let (a, b) = packed.split_at_mut(DEGREE);
                let (ai, bi) = indices.split_at_mut(DEGREE);
                group.fill(0);
                for row in 0..3 {
                    for col in 0..2 {
                        let product =
                            &mut group[(row * 2 + col) * DEGREE..(row * 2 + col + 1) * DEGREE];
                        // COMPLETE inner-block accumulation before trace or ParseMat.
                        for inner in 0..2 {
                            a.fill(0);
                            b.fill(0);
                            let mut alen = 0;
                            let mut blen = 0;
                            for at in self.coordinates.lhs(row, inner)? {
                                let v = lhs[at.matrix];
                                // Negacyclic sign in R_(2^64) BEFORE applying U.
                                let canonical = if at.negative { v.wrapping_neg() } else { v };
                                a[at.coefficient] = self.rns.encode(canonical)?.rem_word(p);
                                ai[alen] = at.coefficient as u64;
                                alen += 1;
                            }
                            for at in self.coordinates.rhs(inner, col)? {
                                b[at.coefficient] = rhs[at.matrix] % p;
                                bi[blen] = at.coefficient as u64;
                                blen += 1;
                            }
                            for &aindex in &ai[..alen] {
                                let aindex = aindex as usize;
                                let av = a[aindex];
                                for &bindex in &bi[..blen] {
                                    let bindex = bindex as usize;
                                    let at = aindex + bindex;
                                    let v = mul(av, b[bindex], p);
                                    let cell = &mut product[at % DEGREE];
                                    *cell = if at >= DEGREE {
                                        sub(*cell, v, p)
                                    } else {
                                        (*cell + v) % p
                                    };
                                }
                            }
                        }
                    }
                }
                trace_interleave(
                    DEGREE,
                    p,
                    group,
                    64,
                    6,
                    &mut phases
                        [(direction * 3 + limb) * DEGREE..(direction * 3 + limb + 1) * DEGREE],
                    trace,
                )?;
            }
            for at in 0..DEGREE {
                // Public padding/tails, not secret sparsity: traced phase is
                // identically zero outside visible cells. Down the full mask
                // once there; this is exactly Down(0+mask), not a skipped share.
                let mask_down = self.rns.down(input.masks[direction][at])?;
                converted[(direction * 2 + 1) * DEGREE + at] = mask_down.wrapping_neg();
                if !self.coordinates.visible(0, at)? {
                    converted[direction * 2 * DEGREE + at] = mask_down;
                    continue;
                }
                let phase = self.rns.reconstruct(std::array::from_fn(|limb| {
                    phases[(direction * 3 + limb) * DEGREE + at]
                }))?;
                let mask = input.masks[direction][at];
                converted[direction * 2 * DEGREE + at] =
                    self.rns.down(phase.add_mod(mask, self.rns.modulus())?)?;
            }
            for row in 0..SHAPE[0] {
                for col in 0..SHAPE[2] {
                    let (group, at) = self.coordinates.address(row, col)?;
                    debug_assert_eq!(group, 0);
                    for side in 0..2 {
                        matrices[(direction * 2 + side) * CELLS + row * SHAPE[2] + col] =
                            converted[(direction * 2 + side) * DEGREE + at];
                    }
                }
            }
        }
        // Local terms remain exact. No H2A, truncation or phase re-encoding here.
        for party in 0..2 {
            for row in 0..SHAPE[0] {
                for col in 0..SHAPE[2] {
                    let cell = row * SHAPE[2] + col;
                    let mut value = 0_u64;
                    for h in 0..SHAPE[1] {
                        value = value.wrapping_add(
                            input.lhs[party][row * SHAPE[1] + h]
                                .wrapping_mul(input.rhs[party][h * SHAPE[2] + col]),
                        );
                    }
                    for (d, (owner, _)) in TwoPartyProductPlan::CROSS_TERMS.into_iter().enumerate()
                    {
                        let side = usize::from(party != owner as usize);
                        value = value.wrapping_add(matrices[(d * 2 + side) * CELLS + cell]);
                    }
                    matrices[(4 + party) * CELLS + cell] = value;
                }
            }
        }
        Ok(())
    }
}

/// Explicit bounded clear geometry; delegates to the SAME legacy trace kernel.
/// Does NOT relax CoefficientRing::new or its inverse contract.
pub fn trace_interleave(
    n: usize,
    p: u64,
    input: &[u64],
    stride: usize,
    count: usize,
    output: &mut [u64],
    scratch: &mut [u64],
) -> Result<(), Error> {
    crate::coefficient_layout::trace::interleave(
        n, p, DEGREE, input, stride, count, output, scratch,
    )
    .map_err(Error::from)
}

#[cfg(test)]
#[path = "coefficient_phase_reference/tests.rs"]
mod tests;
