//! Bounded row/output kernels. Bounds are explicit public source contracts, not
//! secret membership proofs unless the explicit checked-range kernel is used.
//! No whole-matrix schedule or approximation lives here.
use super::super::{
    conversions::{Operation, SignedRange},
    Budget, Error,
};
pub use super::gates::InputRange;
use super::{Builder, Plan, Value, MAX_INPUTS};
use sha2::{Digest, Sha256};

const HALF: i128 = 1i128 << 49;
const MASK: u64 = (1u64 << 50) - 1;
const SCRATCH: usize = MAX_INPUTS + 4096;

/// Both commitments must be nonzero. Compiler owns provenance and input validity.
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub source_digest: [u8; 32],
    pub contract_digest: [u8; 32],
}

/// Only final additive outputs are exported; `range` applies to every output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub outputs: usize,
    pub range: InputRange,
    pub raw_range: Option<InputRange>,
}

fn signed(x: i128) -> Result<i128, Error> {
    if (-HALF..HALF).contains(&x) {
        Ok(x)
    } else {
        Err(Error::Domain)
    }
}
fn checked(r: InputRange) -> Result<InputRange, Error> {
    signed(r.min.into())?;
    signed(r.max.into())?;
    if r.min > r.max {
        return Err(Error::Domain);
    }
    Ok(r)
}
fn interval(lo: i128, hi: i128) -> Result<InputRange, Error> {
    Ok(InputRange {
        min: signed(lo)? as i64,
        max: signed(hi)? as i64,
    })
}
fn product(a: InputRange, b: InputRange) -> Result<InputRange, Error> {
    let (lo, hi) = product_limits(a, b)?;
    interval(lo, hi)
}
// Keep wide exact endpoints until an optional independent bound is intersected.
// All input endpoints still satisfy the same signed50 contract.
fn product_limits(a: InputRange, b: InputRange) -> Result<(i128, i128), Error> {
    checked(a)?;
    checked(b)?;
    let products = [
        i128::from(a.min).checked_mul(b.min.into()),
        i128::from(a.min).checked_mul(b.max.into()),
        i128::from(a.max).checked_mul(b.min.into()),
        i128::from(a.max).checked_mul(b.max.into()),
    ];
    let mut lo = i128::MAX;
    let mut hi = i128::MIN;
    for p in products {
        let p = p.ok_or(Error::Domain)?;
        lo = lo.min(p);
        hi = hi.max(p);
    }
    Ok((lo, hi))
}
fn sum(a: InputRange, b: InputRange) -> Result<InputRange, Error> {
    interval(
        i128::from(a.min)
            .checked_add(b.min.into())
            .ok_or(Error::Domain)?,
        i128::from(a.max)
            .checked_add(b.max.into())
            .ok_or(Error::Domain)?,
    )
}
fn word(x: i64) -> u64 {
    (x as u64) & MASK
}
fn hash(binding: Binding, kind: &[u8], ranges: &[InputRange]) -> Result<Sha256, Error> {
    if binding.source_digest == [0; 32] || binding.contract_digest == [0; 32] {
        return Err(Error::Domain);
    }
    let mut h = Sha256::new();
    h.update(b"pllm.fusefss.bounded-kernels.ring50.q12.v1\0");
    h.update(binding.contract_digest);
    h.update((kind.len() as u64).to_le_bytes());
    h.update(kind);
    h.update((ranges.len() as u64).to_le_bytes());
    for r in ranges {
        checked(*r)?;
        h.update(r.min.to_le_bytes());
        h.update(r.max.to_le_bytes());
    }
    Ok(h)
}
fn builder(
    binding: Binding,
    h: Sha256,
    inputs: usize,
    budget: &Budget,
) -> Result<Builder<'_>, Error> {
    if inputs == 0 || inputs > MAX_INPUTS {
        return Err(Error::Domain);
    }
    let contract: [u8; 32] = h.finalize().into();
    let widths = [50; MAX_INPUTS];
    Builder::new(binding.source_digest, &contract, &widths[..inputs], budget)
}
fn ars12(b: &mut Builder<'_>, raw: Value) -> Result<Value, Error> {
    b.convert(
        raw,
        Operation::Ars {
            shift: 12,
            range: SignedRange::full_ring(50)?,
        },
        50,
    )
}
fn finish_dot(
    b: Builder<'_>,
    output: Value,
    raw: InputRange,
) -> Result<(Plan<'_>, Metadata), Error> {
    let range = interval(
        i128::from(raw.min).div_euclid(4096),
        i128::from(raw.max).div_euclid(4096),
    )?;
    Ok((
        b.finish(&[output])?,
        Metadata {
            outputs: 1,
            range,
            raw_range: Some(raw),
        },
    ))
}

/// Public-linear contraction: sum all exact raw products, then ONE signed ARS12.
/// Public coefficients must also satisfy their explicit source bound.
pub fn public_dot<'a>(
    binding: Binding,
    ranges: &[InputRange],
    coefficients: &[i64],
    coefficient_range: InputRange,
    budget: &'a Budget,
) -> Result<(Plan<'a>, Metadata), Error> {
    if ranges.is_empty() || ranges.len() != coefficients.len() || ranges.len() > MAX_INPUTS {
        return Err(Error::Domain);
    }
    checked(coefficient_range)?;
    let mut h = hash(binding, b"public-dot-complete-accumulation-ars12", ranges)?;
    h.update(coefficient_range.min.to_le_bytes());
    h.update(coefficient_range.max.to_le_bytes());
    let mut raw = InputRange { min: 0, max: 0 };
    for (&r, &c) in ranges.iter().zip(coefficients) {
        if c < coefficient_range.min || c > coefficient_range.max {
            return Err(Error::Domain);
        }
        raw = sum(raw, product(r, InputRange { min: c, max: c })?)?;
        h.update(c.to_le_bytes());
    }
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, ranges.len(), budget)?;
    let mut acc = b.constant(50, 0)?;
    for (i, &c) in coefficients.iter().enumerate() {
        let x = b.input(i)?;
        let term = b.mul_public(x, word(c))?;
        acc = b.add(acc, term)?;
    }
    let output = ars12(&mut b, acc)?;
    finish_dot(b, output, raw)
}

/// Private/private contraction. Inputs are all left shares, then all right shares.
pub fn private_dot<'a>(
    binding: Binding,
    left: &[InputRange],
    right: &[InputRange],
    budget: &'a Budget,
) -> Result<(Plan<'a>, Metadata), Error> {
    private_dot_inner(binding, left, right, None, budget)
}

/// Same complete accumulation and one ARS12, conditional on independently
/// established complete-vector squared energies. The compiler authenticates
/// source/view provenance. No private membership check or raw-range override is
/// accepted; this kernel derives and intersects the integer Cauchy bound itself.
/// `energies` are [sum(left_i^2), sum(right_i^2)] upper bounds on ordinary signed
/// Q12 codes, not modular residues and not per-coordinate allowances. Both must
/// cover the full declared vector. Nonnegative energies bound every coordinate
/// and every prefix too, even when the final dot cancels to zero. Each product
/// and prefix must fit signed50 BEFORE the sole final ARS12; Metadata includes
/// the asymmetric signed floor interval. Premises are pinned in the Plan domain.
/// Lower-bound consistency checks are not authentication or private membership
/// proofs. Returned Plan borrows only Budget, not input intervals/proof owners.
pub fn private_dot_with_row_energy<'a>(
    binding: Binding,
    left: &[InputRange],
    right: &[InputRange],
    energies: [u128; 2],
    budget: &'a Budget,
) -> Result<(Plan<'a>, Metadata), Error> {
    private_dot_inner(binding, left, right, Some(energies), budget)
}

fn private_dot_inner<'a>(
    binding: Binding,
    left: &[InputRange],
    right: &[InputRange],
    energies: Option<[u128; 2]>,
    budget: &'a Budget,
) -> Result<(Plan<'a>, Metadata), Error> {
    if left.is_empty() || left.len() != right.len() || left.len() > MAX_INPUTS / 2 {
        return Err(Error::Domain);
    }
    let kind: &[u8] = if energies.is_some() {
        b"private-dot-complete-row-energy-ars12.v1"
    } else {
        b"private-dot-complete-accumulation-ars12"
    };
    let mut h = hash(binding, kind, left)?;
    let radius = if let Some(energies) = energies {
        for (ranges, energy) in [(left, energies[0]), (right, energies[1])] {
            if minimum_row_energy(ranges)? > energy {
                return Err(Error::Domain);
            }
            h.update(energy.to_le_bytes());
        }
        Some(
            i128::try_from(
                pllm_core::integer_bounds::dot_radius(energies[0], energies[1])
                    .map_err(|_| Error::Domain)?,
            )
            .map_err(|_| Error::Domain)?,
        )
    } else {
        None
    };
    let mut raw = InputRange { min: 0, max: 0 };
    for (&a, &r) in left.iter().zip(right) {
        raw = if let Some(radius) = radius {
            // A coordinate product and every prefix dot are subsets of these
            // complete vectors. For every prefix J, E_left(J)<=E_left and
            // E_right(J)<=E_right; shared integer Cauchy bounds |dot(J)|.
            // Check the term BEFORE the addition: later cancellation cannot
            // justify a wrapped product or prefix. Keep tighter scalar bounds.
            let (lo, hi) = product_limits(a, r)?;
            let term = intersect_dot_radius(lo, hi, radius)?;
            intersect_dot_radius(
                i128::from(raw.min)
                    .checked_add(term.min.into())
                    .ok_or(Error::Domain)?,
                i128::from(raw.max)
                    .checked_add(term.max.into())
                    .ok_or(Error::Domain)?,
                radius,
            )?
        } else {
            sum(raw, product(a, r)?)?
        };
        h.update(r.min.to_le_bytes());
        h.update(r.max.to_le_bytes());
    }
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, left.len() * 2, budget)?;
    let mut acc = b.constant(50, 0)?;
    for i in 0..left.len() {
        let a = b.input(i)?;
        let c = b.input(left.len() + i)?;
        let term = b.multiply(a, c)?;
        acc = b.add(acc, term)?;
    }
    let output = ars12(&mut b, acc)?;
    finish_dot(b, output, raw)
}

fn minimum_row_energy(ranges: &[InputRange]) -> Result<u128, Error> {
    ranges.iter().try_fold(0u128, |sum, &range| {
        checked(range)?;
        let nearest = if range.min > 0 {
            range.min
        } else if range.max < 0 {
            range.max
        } else {
            0
        };
        // Shared exact signed-code energy arithmetic, no second square routine.
        let square =
            pllm_core::integer_bounds::row_energy(&[nearest]).map_err(|_| Error::Domain)?;
        sum.checked_add(square).ok_or(Error::Domain)
    })
}

fn intersect_dot_radius(lo: i128, hi: i128, radius: i128) -> Result<InputRange, Error> {
    if radius < 0 {
        return Err(Error::Domain);
    }
    let lo = lo.max(-radius);
    let hi = hi.min(radius);
    if lo > hi {
        return Err(Error::Domain);
    }
    interval(lo, hi)
}

/// Explicit numeric order: exact source-code product followed by signed ARS12.
/// Does not claim this order is approved for any unresolved decoder profile.
pub fn public_scale_q12(
    binding: Binding,
    range: InputRange,
    coefficient: i64,
    coefficient_range: InputRange,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    public_dot(binding, &[range], &[coefficient], coefficient_range, budget)
}

/// Standalone full-vocabulary D9 plan over the existing consuming raw-DCF owner.
/// The declared interval must lie in [0,V); the compiler must establish actual
/// membership and authenticate source/profile provenance. Local A50 shares are
/// not bits. No scalar register/input/instruction limit is increased.
pub fn compact_token_one_hot(
    binding: Binding,
    id_range: InputRange,
    vocabulary: usize,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    checked(id_range)?;
    if vocabulary == 0
        || vocabulary > super::one_hot::MAX_VOCABULARY
        || id_range.min < 0
        || id_range.max >= vocabulary as i64
    {
        return Err(Error::Domain);
    }
    let mut h = hash(binding, b"compact-token-one-hot-raw-dcf", &[id_range])?;
    h.update((vocabulary as u64).to_le_bytes());
    let plan = Plan::compact_token_one_hot(
        binding.source_digest,
        h.finalize().into(),
        vocabulary,
        budget,
    )?;
    Ok((plan, metadata(vocabulary, InputRange { min: 0, max: 1 })))
}

/// Equality-derived one-hot; no caller-supplied selector is treated as proven bit.
/// Compiler must establish initial ID membership in the declared vocabulary.
pub fn token_one_hot(
    binding: Binding,
    id_range: InputRange,
    vocabulary: usize,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    checked(id_range)?;
    if vocabulary == 0
        || vocabulary > MAX_INPUTS
        || id_range.min < 0
        || id_range.max as usize >= vocabulary
    {
        return Err(Error::Domain);
    }
    let mut h = hash(binding, b"token-equality-one-hot", &[id_range])?;
    h.update((vocabulary as u64).to_le_bytes());
    let _scratch = budget.reserve(SCRATCH + 8 * vocabulary)?;
    let mut b = builder(binding, h, 1, budget)?;
    let id = b.input(0)?;
    let zero = b.constant(50, 0)?;
    let mut outputs = super::super::allocate(vocabulary, budget)?;
    for i in 0..vocabulary {
        let lower = b.lt_unsigned(id, i as u64)?;
        let upper = b.lt_unsigned(id, i as u64 + 1)?;
        outputs.push(b.select(lower, zero, upper)?);
    }
    Ok((
        b.finish(&outputs)?,
        Metadata {
            outputs: vocabulary,
            range: InputRange { min: 0, max: 1 },
            raw_range: None,
        },
    ))
}

/// Stable lowest-index argmax. Exports ONLY additive token ID, never best score.
pub fn argmax<'a>(
    binding: Binding,
    ranges: &[InputRange],
    budget: &'a Budget,
) -> Result<(Plan<'a>, Metadata), Error> {
    if ranges.is_empty() || ranges.len() > MAX_INPUTS {
        return Err(Error::Domain);
    }
    let h = hash(binding, b"argmax-strict-improvement-lowest-index", ranges)?;
    let mut best = ranges[0];
    for &r in &ranges[1..] {
        interval(
            i128::from(best.min) - i128::from(r.max),
            i128::from(best.max) - i128::from(r.min),
        )?;
        best = InputRange {
            min: best.min.max(r.min),
            max: best.max.max(r.max),
        };
    }
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, ranges.len(), budget)?;
    let mut best = b.input(0)?;
    let mut id = b.constant(50, 0)?;
    for i in 1..ranges.len() {
        let candidate = b.input(i)?;
        let delta = b.sub(best, candidate)?;
        let improve = b.lt_signed(delta, 0)?;
        best = b.select(improve, candidate, best)?;
        let next_id = b.constant(50, i as u64)?;
        id = b.select(improve, next_id, id)?;
    }
    Ok((
        b.finish(&[id])?,
        Metadata {
            outputs: 1,
            range: InputRange {
                min: 0,
                max: ranges.len() as i64 - 1,
            },
            raw_range: None,
        },
    ))
}

fn metadata(outputs: usize, range: InputRange) -> Metadata {
    Metadata {
        outputs,
        range,
        raw_range: None,
    }
}

/// Public zero-fallback policy, committed into the checked-range Plan domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeFallback {
    /// Preserve each accepted coordinate; replace each rejected coordinate by zero.
    PerInput,
    /// Preserve the entire row iff every coordinate is accepted; otherwise zero it.
    WholeRow,
}

/// Opt-in private inclusive signed50 membership check with fixed PerInput fallback.
/// Outputs are sanitized A50 inputs in order, then ONE A50 bit equal to the AND
/// of all membership predicates. Zero must belong to `accepted`. No predicate or
/// failing position is opened; the complete public schedule runs on every input.
///
/// Every signed50 ring value is checked, even outside a declared `input_ranges`
/// interval. Those public intervals are identity metadata, never a reason to elide
/// a check. The caller must route only sanitized outputs to downstream arithmetic,
/// accumulate validity secretly, and make any rejection decision only at its final
/// boundary. This kernel alone does not implement request-level rejection.
///
/// Metadata.range covers ALL outputs, including the final bit: the first n outputs
/// have the tighter `accepted` range and the last has range [0,1].
pub fn checked_range<'a>(
    binding: Binding,
    input_ranges: &[InputRange],
    accepted: InputRange,
    budget: &'a Budget,
) -> Result<(Plan<'a>, Metadata), Error> {
    checked_range_with_fallback(
        binding,
        input_ranges,
        accepted,
        RangeFallback::PerInput,
        budget,
    )
}

/// Same protected check, with an explicit public whole-row/per-input policy.
pub fn checked_range_with_fallback<'a>(
    binding: Binding,
    input_ranges: &[InputRange],
    accepted: InputRange,
    fallback: RangeFallback,
    budget: &'a Budget,
) -> Result<(Plan<'a>, Metadata), Error> {
    checked_range_inner(binding, input_ranges, accepted, fallback, false, budget)
}

/// Same inclusive predicate and zero fallback, with a balanced protected AND.
/// Membership remains secret; the public tree visits every coordinate.
pub fn checked_range_balanced<'a>(
    binding: Binding,
    input_ranges: &[InputRange],
    accepted: InputRange,
    fallback: RangeFallback,
    budget: &'a Budget,
) -> Result<(Plan<'a>, Metadata), Error> {
    checked_range_inner(binding, input_ranges, accepted, fallback, true, budget)
}

fn checked_range_inner<'a>(
    binding: Binding,
    input_ranges: &[InputRange],
    accepted: InputRange,
    fallback: RangeFallback,
    balanced: bool,
    budget: &'a Budget,
) -> Result<(Plan<'a>, Metadata), Error> {
    checked(accepted)?;
    if input_ranges.is_empty()
        || input_ranges.len() > MAX_INPUTS
        || accepted.min > 0
        || accepted.max < 0
    {
        return Err(Error::Domain);
    }
    let mut h = hash(
        binding,
        b"checked-signed50-inclusive-range-zero-fallback.v1",
        input_ranges,
    )?;
    h.update(accepted.min.to_le_bytes());
    if balanced {
        h.update(b"balanced-complete-membership.v1\0");
    }
    h.update(accepted.max.to_le_bytes());
    h.update([match fallback {
        RangeFallback::PerInput => 0,
        RangeFallback::WholeRow => 1,
    }]);
    let count = input_ranges.len().checked_add(1).ok_or(Error::Resource)?;
    let scratch = count
        .checked_mul(std::mem::size_of::<Value>() * if balanced { 2 } else { 1 })
        .and_then(|n| n.checked_add(SCRATCH))
        .ok_or(Error::Resource)?;
    let _scratch = budget.reserve(scratch)?;
    let mut b = builder(binding, h, input_ranges.len(), budget)?;
    let mut outputs = super::super::allocate(count, budget)?;
    let zero = b.constant(50, 0)?;
    let one = b.constant(50, 1)?;
    let sign_bit = b.constant(50, HALF as u64)?;
    let mut valid = one;
    let mut members = if balanced {
        super::super::allocate(input_ranges.len(), budget)?
    } else {
        Vec::new()
    };
    // (word(x) + 2^49) mod 2^50 == signed(x) + 2^49. This public
    // sign-bit rotation gives exact unsigned order over the ENTIRE signed ring.
    // It intentionally wraps; neither subtraction nor signed no-wrap is assumed.
    // Thresholds include 0 and 2^50, both supported by the existing DCF predicate.
    let lower_threshold = (i128::from(accepted.min) + HALF) as u64;
    let upper_threshold = (i128::from(accepted.max) + HALF + 1) as u64;
    for i in 0..input_ranges.len() {
        let input = b.input(i)?;
        let ordered = b.add(input, sign_bit)?;
        let below = b.lt_unsigned(ordered, lower_threshold)?;
        let below_upper = b.lt_unsigned(ordered, upper_threshold)?;
        // Both comparison outputs are proven A50 bits. Phi preserves bit typing
        // for membership and its AND chain, and uses existing fresh products.
        let member = b.select(below, zero, below_upper)?;
        if balanced {
            members.push(member);
        } else {
            valid = b.select(member, valid, zero)?;
        }
        outputs.push(match fallback {
            RangeFallback::PerInput => b.select(member, input, zero)?,
            RangeFallback::WholeRow => input,
        });
    }
    if balanced {
        while members.len() > 1 {
            let length = members.len();
            for pair in 0..length / 2 {
                members[pair] = b.select(members[2 * pair], members[2 * pair + 1], zero)?;
            }
            if length % 2 != 0 {
                members[length / 2] = members[length - 1];
            }
            members.truncate(length.div_ceil(2));
        }
        valid = members[0];
    }
    if fallback == RangeFallback::WholeRow {
        for output in &mut outputs {
            *output = b.select(valid, *output, zero)?;
        }
    }
    outputs.push(valid);
    Ok((
        b.finish(&outputs)?,
        metadata(
            count,
            InputRange {
                min: accepted.min,
                max: accepted.max.max(1),
            },
        ),
    ))
}

/// Exact sticky AND of [prior_valid, new_valid], returning one A50 bit. Both
/// reconstructed inputs must be bits established by protected checks or earlier
/// ANDs; arbitrary local A50 shares are not Boolean inputs. Caller owns provenance.
/// This is an unscaled existing private product, with no opening or early exit.
pub fn validity_and(binding: Binding, budget: &Budget) -> Result<(Plan<'_>, Metadata), Error> {
    let bit = InputRange { min: 0, max: 1 };
    let h = hash(binding, b"private-validity-bit-and.v1", &[bit; 2])?;
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, 2, budget)?;
    let prior = b.input(0)?;
    let next = b.input(1)?;
    let valid = b.multiply(prior, next)?;
    Ok((b.finish(&[valid])?, metadata(1, bit)))
}

/// Inputs [values..., validity]; outputs only validity * values, without ARS or
/// opening. Validity must be a caller-proven reconstructed A50 bit, e.g. the output
/// of checked_range/validity_and. Multiplication by 0/1 is signed50-safe for every
/// value in the declared intervals. Suitable for final token masking; the caller
/// retains the separate shared aggregate bit for its final-only rejection boundary.
pub fn mask_by_validity<'a>(
    binding: Binding,
    input_ranges: &[InputRange],
    budget: &'a Budget,
) -> Result<(Plan<'a>, Metadata), Error> {
    if input_ranges.is_empty() || input_ranges.len() >= MAX_INPUTS {
        return Err(Error::Domain);
    }
    let h = hash(
        binding,
        b"private-values-first-validity-last-unscaled-mask.v1",
        input_ranges,
    )?;
    let mut output_range = InputRange { min: 0, max: 0 };
    for range in input_ranges {
        output_range.min = output_range.min.min(range.min);
        output_range.max = output_range.max.max(range.max);
    }
    let scratch = input_ranges
        .len()
        .checked_mul(std::mem::size_of::<Value>())
        .and_then(|n| n.checked_add(SCRATCH))
        .ok_or(Error::Resource)?;
    let _scratch = budget.reserve(scratch)?;
    let mut b = builder(binding, h, input_ranges.len() + 1, budget)?;
    let valid = b.input(input_ranges.len())?;
    let mut outputs = super::super::allocate(input_ranges.len(), budget)?;
    for i in 0..input_ranges.len() {
        let value = b.input(i)?;
        outputs.push(b.multiply(valid, value)?);
    }
    Ok((b.finish(&outputs)?, metadata(outputs.len(), output_range)))
}

/// Signed floor division by 2^shift, retaining the signed50 output ring.
pub fn floor(
    binding: Binding,
    range: InputRange,
    shift: u8,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    if shift >= 50 {
        return Err(Error::Domain);
    }
    let mut h = hash(binding, b"signed-floor", &[range])?;
    h.update([shift]);
    let output_range = interval(
        i128::from(range.min).div_euclid(1i128 << shift),
        i128::from(range.max).div_euclid(1i128 << shift),
    )?;
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, 1, budget)?;
    let x = b.input(0)?;
    let y = if shift == 0 {
        x
    } else {
        b.convert(
            x,
            Operation::Ars {
                shift,
                range: SignedRange::full_ring(50)?,
            },
            50,
        )?
    };
    Ok((b.finish(&[y])?, metadata(1, output_range)))
}

fn equality_node(b: &mut Builder<'_>, id: Value, row: usize) -> Result<Value, Error> {
    let lower = b.lt_unsigned(id, row as u64)?;
    let upper = b.lt_unsigned(id, row as u64 + 1)?;
    let zero = b.constant(50, 0)?;
    b.select(lower, zero, upper)
}

/// Equality to one public row. Out-of-range rows legitimately produce zero.
pub fn equality(
    binding: Binding,
    id_range: InputRange,
    public_row: usize,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    checked(id_range)?;
    if id_range.min < 0 || public_row as u128 >= HALF as u128 {
        return Err(Error::Domain);
    }
    let mut h = hash(binding, b"token-equality-public-row", &[id_range])?;
    h.update((public_row as u64).to_le_bytes());
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, 1, budget)?;
    let id = b.input(0)?;
    let result = equality_node(&mut b, id, public_row)?;
    Ok((
        b.finish(&[result])?,
        metadata(1, InputRange { min: 0, max: 1 }),
    ))
}

/// Stream disjoint row chunks over arbitrary fixed vocabularies without raising
/// per-Plan caps. Caller controls chunk size; normal admission rejects oversize.
pub fn token_one_hot_chunk(
    binding: Binding,
    id_range: InputRange,
    vocabulary: usize,
    rows: std::ops::Range<usize>,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    checked(id_range)?;
    if vocabulary == 0
        || vocabulary as u128 > HALF as u128
        || id_range.min < 0
        || id_range.max as usize >= vocabulary
        || rows.start >= rows.end
        || rows.end > vocabulary
        || rows.len() > MAX_INPUTS
    {
        return Err(Error::Domain);
    }
    let mut h = hash(binding, b"token-equality-one-hot-chunk", &[id_range])?;
    for v in [vocabulary, rows.start, rows.end] {
        h.update((v as u64).to_le_bytes());
    }
    let _scratch = budget.reserve(SCRATCH + 8 * rows.len())?;
    let mut b = builder(binding, h, 1, budget)?;
    let id = b.input(0)?;
    let mut outputs = super::super::allocate(rows.len(), budget)?;
    for row in rows {
        outputs.push(equality_node(&mut b, id, row)?);
    }
    Ok((
        b.finish(&outputs)?,
        metadata(outputs.len(), InputRange { min: 0, max: 1 }),
    ))
}

/// Inputs [left value, left index, right value, right index]. Equal values select
/// lowest numeric index even when inputs arrive in arbitrary index order.
pub fn stable_max(
    binding: Binding,
    ranges: [InputRange; 2],
    index_range: InputRange,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    checked(index_range)?;
    if index_range.min < 0 {
        return Err(Error::Domain);
    }
    let mut h = hash(binding, b"stable-max-lowest-numeric-index", &ranges)?;
    h.update(index_range.min.to_le_bytes());
    h.update(index_range.max.to_le_bytes());
    interval(
        i128::from(ranges[0].min) - i128::from(ranges[1].max),
        i128::from(ranges[0].max) - i128::from(ranges[1].min),
    )?;
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, 4, budget)?;
    let left = b.input(0)?;
    let li = b.input(1)?;
    let right = b.input(2)?;
    let ri = b.input(3)?;
    let delta = b.sub(left, right)?;
    let less = b.lt_signed(delta, 0)?;
    let le = b.lt_signed(delta, 1)?;
    let zero = b.constant(50, 0)?;
    let one = b.constant(50, 1)?;
    let equal = b.select(less, zero, le)?;
    let index_delta = b.sub(ri, li)?;
    let lower_index = b.lt_signed(index_delta, 0)?;
    let tie = b.select(equal, lower_index, zero)?;
    let choose_right = b.select(less, one, tie)?;
    let value = b.select(choose_right, right, left)?;
    let index = b.select(choose_right, ri, li)?;
    let result_range = InputRange {
        min: ranges[0].min.max(ranges[1].min).min(index_range.min),
        max: ranges[0].max.max(ranges[1].max).max(index_range.max),
    };
    Ok((b.finish(&[value, index])?, metadata(2, result_range)))
}

/// Maximum of two scores and their opaque payloads; ties retain the left pair.
/// Inputs are [left_score, left_payload, right_score, right_payload]. This does
/// not compare payloads. An ordered tournament whose left subtree precedes its
/// right subtree therefore retains the original lowest leaf on equal scores.
pub fn max_left_on_tie(
    binding: Binding,
    ranges: [InputRange; 2],
    payload_range: InputRange,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    checked(payload_range)?;
    let mut h = hash(binding, b"max-pair-left-on-tie", &ranges)?;
    h.update(payload_range.min.to_le_bytes());
    h.update(payload_range.max.to_le_bytes());
    interval(
        i128::from(ranges[0].min) - i128::from(ranges[1].max),
        i128::from(ranges[0].max) - i128::from(ranges[1].min),
    )?;
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, 4, budget)?;
    let left = b.input(0)?;
    let lp = b.input(1)?;
    let right = b.input(2)?;
    let rp = b.input(3)?;
    let delta = b.sub(left, right)?;
    let choose_right = b.lt_signed(delta, 0)?;
    let value = b.select(choose_right, right, left)?;
    let payload = b.select(choose_right, rp, lp)?;
    let result_range = InputRange {
        min: ranges[0].min.max(ranges[1].min).min(payload_range.min),
        max: ranges[0].max.max(ranges[1].max).max(payload_range.max),
    };
    Ok((b.finish(&[value, payload])?, metadata(2, result_range)))
}

/// Private additive residual; inputs [left,right], with explicit no-wrap proof.
pub fn add(
    binding: Binding,
    ranges: [InputRange; 2],
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    let h = hash(binding, b"add-no-wrap", &ranges)?;
    let output_range = sum(ranges[0], ranges[1])?;
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, 2, budget)?;
    let x = b.input(0)?;
    let y = b.input(1)?;
    let output = b.add(x, y)?;
    Ok((b.finish(&[output])?, metadata(1, output_range)))
}

/// Add one public integer-code bias, not a Q12 product or implicit rescale.
pub fn public_add(
    binding: Binding,
    range: InputRange,
    bias: i64,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    let mut h = hash(binding, b"public-add-no-wrap", &[range])?;
    let output_range = sum(
        range,
        checked(InputRange {
            min: bias,
            max: bias,
        })?,
    )?;
    h.update(bias.to_le_bytes());
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, 1, budget)?;
    let x = b.input(0)?;
    let bias = b.constant(50, word(bias))?;
    let output = b.add(x, bias)?;
    Ok((b.finish(&[output])?, metadata(1, output_range)))
}

/// Explicit affine order floor(x * coefficient / 4096) + bias.
pub fn public_affine(
    binding: Binding,
    range: InputRange,
    coefficient: i64,
    coefficient_range: InputRange,
    bias: i64,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    checked(coefficient_range)?;
    if coefficient < coefficient_range.min || coefficient > coefficient_range.max {
        return Err(Error::Domain);
    }
    let raw = product(
        range,
        InputRange {
            min: coefficient,
            max: coefficient,
        },
    )?;
    let scaled = interval(
        i128::from(raw.min).div_euclid(4096),
        i128::from(raw.max).div_euclid(4096),
    )?;
    let result = sum(
        scaled,
        checked(InputRange {
            min: bias,
            max: bias,
        })?,
    )?;
    let mut h = hash(
        binding,
        b"public-affine-product-ars12-add-bias",
        &[range, coefficient_range],
    )?;
    h.update(coefficient.to_le_bytes());
    h.update(bias.to_le_bytes());
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, 1, budget)?;
    let x = b.input(0)?;
    let raw_node = b.mul_public(x, word(coefficient))?;
    let scaled = ars12(&mut b, raw_node)?;
    let bias_node = b.constant(50, word(bias))?;
    let output = b.add(scaled, bias_node)?;
    Ok((
        b.finish(&[output])?,
        Metadata {
            outputs: 1,
            range: result,
            raw_range: Some(raw),
        },
    ))
}

/// Approved D13 half-split RoPE: [floor(l*c/4096)+floor((-r)*s/4096),
/// floor(r*c/4096)+floor(l*s/4096)]. Rotate-half's sign belongs INSIDE
/// the product before flooring; negating a floored positive term is not exact.
pub fn rope(
    binding: Binding,
    ranges: [InputRange; 2],
    cosine: i64,
    sine: i64,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    let negative_sine = sine.checked_neg().ok_or(Error::Domain)?;
    signed(i128::from(negative_sine))?;
    let mut h = hash(
        binding,
        b"d13-half-split-rope-signed-terms-per-term-ars12.v2",
        &ranges,
    )?;
    h.update(cosine.to_le_bytes());
    h.update(sine.to_le_bytes());
    let mut scaled = [InputRange { min: 0, max: 0 }; 4];
    for (i, (r, c)) in [
        (ranges[0], cosine),
        (ranges[1], negative_sine),
        (ranges[1], cosine),
        (ranges[0], sine),
    ]
    .into_iter()
    .enumerate()
    {
        let raw = product(r, InputRange { min: c, max: c })?;
        scaled[i] = interval(
            i128::from(raw.min).div_euclid(4096),
            i128::from(raw.max).div_euclid(4096),
        )?;
    }
    let left = sum(scaled[0], scaled[1])?;
    let right = sum(scaled[2], scaled[3])?;
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, 2, budget)?;
    let mut nodes = [0; 4];
    for (i, (input, c)) in [(0, cosine), (1, negative_sine), (1, cosine), (0, sine)]
        .into_iter()
        .enumerate()
    {
        let input = b.input(input)?;
        let raw = b.mul_public(input, word(c))?;
        nodes[i] = ars12(&mut b, raw)?;
    }
    let a = b.add(nodes[0], nodes[1])?;
    let d = b.add(nodes[2], nodes[3])?;
    Ok((
        b.finish(&[a, d])?,
        metadata(
            2,
            InputRange {
                min: left.min.min(right.min),
                max: left.max.max(right.max),
            },
        ),
    ))
}

fn descriptor_error(e: crate::fusefss::Error) -> Error {
    match e {
        crate::fusefss::Error::Allocation => Error::Allocation,
        _ => Error::Domain,
    }
}
fn signed_word(x: u64) -> i128 {
    if x & (1 << 49) == 0 {
        i128::from(x)
    } else {
        i128::from(x) - (1i128 << 50)
    }
}

/// Frozen source GELU, not a new approximation. Derive output envelope from only
/// PUBLIC zero-mask affine partitions, then execute existing native CompiledGelu.
pub fn gelu(
    binding: Binding,
    range: InputRange,
    budget: &Budget,
) -> Result<(Plan<'_>, Metadata), Error> {
    let mut h = hash(binding, b"frozen-source-gelu-signed38", &[range])?;
    if range.min < -(1i64 << 37) || range.max >= 1i64 << 37 {
        return Err(Error::Domain);
    }
    let output_range = {
        let _overlap = budget.reserve(super::super::COMPILER_OVERLAP_BYTES)?;
        budget.allocation()?;
        let descriptor = crate::fusefss::Descriptor::gelu().map_err(descriptor_error)?;
        h.update(descriptor.digest());
        let public = descriptor.compile(0).map_err(descriptor_error)?;
        public
            .with_dealer_handoff(|table| {
                let bounds = (|| {
                    let mut lo = i128::MAX;
                    let mut hi = i128::MIN;
                    for (ends, &[c, l]) in
                        table.shifted_boundaries.windows(2).zip(table.coefficients)
                    {
                        for (domain_lo, domain_hi, offset) in [
                            (0i128, HALF - 1, 0i128),
                            (HALF, (1i128 << 50) - 1, 1i128 << 50),
                        ] {
                            let start = i128::from(ends[0])
                                .max(domain_lo)
                                .max(i128::from(range.min) + offset);
                            let end = (i128::from(ends[1]) - 1)
                                .min(domain_hi)
                                .min(i128::from(range.max) + offset);
                            if start > end {
                                continue;
                            }
                            let c = signed_word(c);
                            let l = signed_word(l);
                            for x in [start - offset, end - offset] {
                                let y = c
                                    .checked_add(l.checked_mul(x).ok_or(Error::Domain)?)
                                    .ok_or(Error::Domain)?;
                                signed(y)?;
                                lo = lo.min(y);
                                hi = hi.max(y);
                            }
                        }
                    }
                    if lo == i128::MAX {
                        return Err(Error::Domain);
                    }
                    interval(lo, hi)
                })();
                Ok(bounds)
            })
            .map_err(descriptor_error)??
    };
    let _scratch = budget.reserve(SCRATCH)?;
    let mut b = builder(binding, h, 1, budget)?;
    let x = b.input(0)?;
    let output = b.gelu(x)?;
    Ok((b.finish(&[output])?, metadata(1, output_range)))
}

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod tests;
