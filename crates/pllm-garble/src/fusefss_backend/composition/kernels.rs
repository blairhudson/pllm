//! Bounded row/output kernels. Bounds are explicit public source contracts, not
//! secret membership proofs. No whole-matrix schedule or approximation lives here.
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
    interval(lo, hi)
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
    if left.is_empty() || left.len() != right.len() || left.len() > MAX_INPUTS / 2 {
        return Err(Error::Domain);
    }
    let mut h = hash(binding, b"private-dot-complete-accumulation-ars12", left)?;
    let mut raw = InputRange { min: 0, max: 0 };
    for (&a, &r) in left.iter().zip(right) {
        raw = sum(raw, product(a, r)?)?;
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
