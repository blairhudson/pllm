//! Exact source LayerNorm/RMSNorm and unsigned50 nonzero normalization index.
//!
//! Admission checks only PUBLIC interval arithmetic. The caller must establish
//! that private inputs belong to the declared interval; this is not a checkpoint
//! verifier or a private membership test. Every private operation below goes
//! through the consuming protected Builder; no exponent/index is opened.
use super::super::{
    allocate,
    conversions::{Operation, SignedRange},
    Budget, Error,
};
use super::{Builder, Plan, Value};
use sha2::{Digest, Sha256};
use std::mem::size_of;

const BITS: u8 = 50;
const HALF: i128 = 1i128 << 49;
const MASK: u64 = (1u64 << BITS) - 1;
const CONTRACT: &[u8] = b"pllm.fusefss.source-norm.v1\0A50/Q12;sum:A50/Q12;mean=floor((4096//k)*sum/4096);center:A50/Q12;squares:S=A50/Q24+1;unsigned-nonzero-index:e=floor(log2(S)),m=floor(128*S/2^e)-128,index=64*m+e;rsqrt-table:k-specific,8192-slots,A50,isqrt((2^55*k)//((128+m)*2^e));product:A50/Q24;ARS12:Q12;affine-product:A50/Q24;ARS12:Q12;beta-after-floor;caller-interval-assumption-not-membership-proof\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Layer,
    Rms,
}

/// Public affine parameters are signed Q12 codes, not host floating point.
pub struct Spec<'a> {
    pub kind: Kind,
    pub input_min: i64,
    pub input_max: i64,
    pub gamma: &'a [i64],
    pub beta: &'a [i64],
}

#[derive(Clone, Copy, Debug)]
pub struct Index {
    pub exponent: Value,
    pub mantissa: Value,
    pub index: Value,
}

/// All 50 public candidates are evaluated, regardless of S. For e<7 the
/// candidate's local left shift may wrap for an UNUSED bin; multiplication by
/// a secret bit/select is ring algebra, so this cannot affect the chosen bin.
/// In the chosen bin 128 <= floor(128*S/2^e) <= 255 and no wrap occurs.
/// For e>=7 canonical reduced-ring TR followed by ZeroExtend is exactly the
/// UNSIGNED right shift, even for S with bit49 set. ARS would be wrong there.
///
/// At public step e>=1, [S<2^e] retains the previous candidate; otherwise the
/// e candidate replaces it. Induction gives min(floor(log2(S)),e), hence exactly
/// one final bin for every ASSUMED 1<=S<2^50. Comparison bits are fresh protected
/// B2A results; each select has fresh product material and checked provenance.
pub fn index(builder: &mut Builder, s: Value) -> Result<Index, Error> {
    let bias = builder.constant(BITS, 128)?;
    let scaled = builder.mul_public(s, 128)?;
    let mut mantissa = builder.sub(scaled, bias)?;
    let mut exponent = builder.constant(BITS, 0)?;
    for e in 1u8..50 {
        let scaled = if e <= 7 {
            builder.mul_public(s, 1u64 << (7 - e))?
        } else {
            let shift = e - 7;
            let reduced = builder.convert(
                s,
                Operation::Tr {
                    shift,
                    range: SignedRange::full_ring(BITS)?,
                },
                BITS - shift,
            )?;
            builder.convert(reduced, Operation::ZeroExtend, BITS)?
        };
        let candidate = builder.sub(scaled, bias)?;
        let candidate_e = builder.constant(BITS, e as u64)?;
        let below = builder.lt_unsigned(s, 1u64 << e)?;
        mantissa = builder.select(below, mantissa, candidate)?;
        exponent = builder.select(below, exponent, candidate_e)?;
    }
    let high = builder.mul_public(mantissa, 64)?;
    let index = builder.add(high, exponent)?;
    Ok(Index {
        exponent,
        mantissa,
        index,
    })
}

/// Standalone source index gate; input is an assumed unsigned nonzero A50 word.
/// Three final arithmetic shares are returned in exponent/mantissa/index order.
pub fn compile_index(source_digest: [u8; 32], budget: &Budget) -> Result<Plan<'_>, Error> {
    let mut builder = Builder::new(source_digest,
        b"pllm.fusefss.norm-index.v1;A50/Q24,unsigned,1<=S<2^50;50-fixed-public-bins;TR+ZeroExtend;outputs=A50/Q0:e,m,index;no-input-membership-proof",
        &[BITS], budget)?;
    let s = builder.input(0)?;
    let extracted = index(&mut builder, s)?;
    builder.finish(&[extracted.exponent, extracted.mantissa, extracted.index])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Interval {
    lo: i128,
    hi: i128,
}

impl Interval {
    fn checked(lo: i128, hi: i128) -> Result<Self, Error> {
        if lo > hi || lo < -HALF || hi >= HALF {
            return Err(Error::Domain);
        }
        Ok(Self { lo, hi })
    }
    fn scale(self, n: i128) -> Result<Self, Error> {
        let a = self.lo.checked_mul(n).ok_or(Error::Domain)?;
        let b = self.hi.checked_mul(n).ok_or(Error::Domain)?;
        Self::checked(a.min(b), a.max(b))
    }
    fn floor12(self) -> Self {
        Self {
            lo: self.lo.div_euclid(4096),
            hi: self.hi.div_euclid(4096),
        }
    }
}

/// Conservative sufficient signed50 no-wrap certificate, using checked i128.
/// A rejected interval can contain safe particular rows; no private admission
/// predicate is opened to recover those rows.
#[derive(Clone, Copy, Debug)]
struct Proof {
    mean: Interval,
    centered: Interval,
    s: Interval,
}

fn preflight(spec: &Spec, inverse_max: u64) -> Result<Proof, Error> {
    let k = spec.gamma.len();
    if !(1..=4096).contains(&k) || spec.beta.len() != k || inverse_max > 1u64 << 30 {
        return Err(Error::Domain);
    }
    let x = Interval::checked(spec.input_min as i128, spec.input_max as i128)?;
    let total = x.scale(k as i128)?;
    let mean = match spec.kind {
        Kind::Layer => total.scale((4096 / k) as i128)?.floor12(),
        Kind::Rms => Interval::checked(0, 0)?,
    };
    let centered = Interval::checked(
        x.lo.checked_sub(mean.hi).ok_or(Error::Domain)?,
        x.hi.checked_sub(mean.lo).ok_or(Error::Domain)?,
    )?;
    let a = centered.lo.checked_mul(centered.lo).ok_or(Error::Domain)?;
    let b = centered.hi.checked_mul(centered.hi).ok_or(Error::Domain)?;
    let square_min = if centered.lo <= 0 && centered.hi >= 0 {
        0
    } else {
        a.min(b)
    };
    let square_max = a.max(b);
    Interval::checked(square_min, square_max)?;
    // Nonnegative squares make every partial sum fit whenever this sum fits.
    let squares = Interval::checked(
        square_min.checked_mul(k as i128).ok_or(Error::Domain)?,
        square_max.checked_mul(k as i128).ok_or(Error::Domain)?,
    )?;
    let s = Interval::checked(
        squares.lo.checked_add(1).ok_or(Error::Domain)?,
        squares.hi.checked_add(1).ok_or(Error::Domain)?,
    )?;
    let product = Interval::checked(
        centered
            .lo
            .checked_mul(inverse_max as i128)
            .ok_or(Error::Domain)?
            .min(0),
        centered
            .hi
            .checked_mul(inverse_max as i128)
            .ok_or(Error::Domain)?
            .max(0),
    )?;
    let q = product.floor12();
    for (&gamma, &beta) in spec.gamma.iter().zip(spec.beta) {
        Interval::checked(gamma as i128, gamma as i128)?;
        Interval::checked(beta as i128, beta as i128)?;
        let affine = q.scale(gamma as i128)?.floor12();
        Interval::checked(
            affine.lo.checked_add(beta as i128).ok_or(Error::Domain)?,
            affine.hi.checked_add(beta as i128).ok_or(Error::Domain)?,
        )?;
    }
    Ok(Proof { mean, centered, s })
}

fn word(value: i64) -> u64 {
    (value as u64) & MASK
}

fn floor12(builder: &mut Builder, value: Value) -> Result<Value, Error> {
    builder.convert(
        value,
        Operation::Ars {
            shift: 12,
            range: SignedRange::full_ring(BITS)?,
        },
        BITS,
    )
}

/// Compile one row; production outputs contain ONLY final arithmetic shares.
/// Runtime's cumulative instruction/register and <=64MiB limits remain binding;
/// source width <=4096 does not promise every such plan fits native capacity.
pub fn compile<'a>(
    source_digest: [u8; 32],
    spec: &Spec,
    budget: &'a Budget,
) -> Result<Plan<'a>, Error> {
    compile_inner(source_digest, spec, budget, false, None)
}

/// Identical source formula, range proof and Plan digest. Only immutable rsqrt
/// generation/validation/copy/hash ownership is shared across request rows.
pub fn compile_cached<'a>(
    source_digest: [u8; 32],
    spec: &Spec,
    cache: &super::cache::SourceTableCache<'a>,
) -> Result<Plan<'a>, Error> {
    compile_inner(source_digest, spec, cache.budget(), false, Some(cache))
}

/// Test-only layout: mean, S, exponent, mantissa, index, inverse, then Y[0..k].
/// These remain arithmetic shares until a test explicitly reconstructs them.
#[cfg(test)]
pub(super) fn compile_fixture<'a>(
    source_digest: [u8; 32],
    spec: &Spec,
    budget: &'a Budget,
) -> Result<Plan<'a>, Error> {
    compile_inner(source_digest, spec, budget, true, None)
}

fn compile_inner<'a>(
    source_digest: [u8; 32],
    spec: &Spec,
    budget: &'a Budget,
    intermediates: bool,
    cache: Option<&super::cache::SourceTableCache<'a>>,
) -> Result<Plan<'a>, Error> {
    let k = spec.gamma.len();
    if source_digest == [0; 32] || (intermediates && !cfg!(test)) {
        return Err(Error::Domain);
    }
    // First reject public algebra before allocating tables or constructing gates.
    // Exact table maximum is slot0 = floor(sqrt(2^48*k)); source generator also
    // checks every payload. Integer binary search avoids a floating rsqrt.
    if !(1..=4096).contains(&k) {
        return Err(Error::Domain);
    }
    let n = (1u128 << 48).checked_mul(k as u128).ok_or(Error::Domain)?;
    let (mut lo, mut hi) = (0u64, 1u64 << 30);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if (mid as u128) * (mid as u128) <= n {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let proof = preflight(spec, lo)?;
    // Conservatively charge borrowed affine buffers too; callers retain these
    // owners while source generation and Plan construction overlap.
    let _affine = budget.reserve(k.checked_mul(2 * size_of::<i64>()).ok_or(Error::Resource)?)?;
    let table = if cache.is_none() {
        Some(super::source_tables::rsqrt(k, budget)?)
    } else {
        None
    };
    if let Some(table) = &table {
        if table.len() != 8192 || table.iter().any(|&v| v > lo || v >= 1u64 << BITS) {
            return Err(Error::Domain);
        }
    }
    let mut hash = Sha256::new();
    hash.update(CONTRACT);
    hash.update([match spec.kind {
        Kind::Layer => 0,
        Kind::Rms => 1,
    }]);
    hash.update((k as u64).to_le_bytes());
    hash.update(spec.input_min.to_le_bytes());
    hash.update(spec.input_max.to_le_bytes());
    for (&g, &b) in spec.gamma.iter().zip(spec.beta) {
        hash.update(g.to_le_bytes());
        hash.update(b.to_le_bytes());
    }
    for bound in [proof.mean, proof.centered, proof.s] {
        hash.update(bound.lo.to_le_bytes());
        hash.update(bound.hi.to_le_bytes());
    }
    hash.update([u8::from(intermediates)]);
    let contract: [u8; 32] = hash.finalize().into();
    // Temporary register-id vectors remain budgeted while Builder/Plan coexist.
    let capacity = k
        .checked_mul(2)
        .and_then(|n| n.checked_add(6))
        .ok_or(Error::Resource)?;
    let _scratch = budget.reserve(
        capacity
            .checked_mul(size_of::<Value>())
            .ok_or(Error::Resource)?,
    )?;
    let mut centered = allocate(k, budget)?;
    let mut outputs = allocate(k + 6, budget)?;
    let input_bits = [BITS; 4096];
    let mut builder = Builder::new(source_digest, &contract, &input_bits[..k], budget)?;
    let zero = builder.constant(BITS, 0)?;
    let mean = if spec.kind == Kind::Layer {
        let mut total = zero;
        for i in 0..k {
            let x = builder.input(i)?;
            total = builder.add(total, x)?;
        }
        let scaled = builder.mul_public(total, (4096 / k) as u64)?;
        floor12(&mut builder, scaled)?
    } else {
        zero
    };
    let mut squares = zero;
    for i in 0..k {
        let x = builder.input(i)?;
        let z = if spec.kind == Kind::Layer {
            builder.sub(x, mean)?
        } else {
            x
        };
        centered.push(z);
        let square = builder.multiply(z, z)?;
        squares = builder.add(squares, square)?;
    }
    let one = builder.constant(BITS, 1)?;
    let s = builder.add(squares, one)?;
    let extracted = index(&mut builder, s)?;
    // Cached factory admits ONLY this closed source formula/k, never caller
    // values. Its exact maximum is the same slot0 value used by preflight above.
    let table_index = match cache {
        Some(c) => builder.source_table(c, super::cache::Kind::Rsqrt { k })?,
        None => builder.table(BITS, table.as_ref().ok_or(Error::Domain)?.as_ref())?,
    };
    let inverse = builder.lookup(extracted.index, table_index)?;
    if intermediates {
        outputs.extend_from_slice(&[
            mean,
            s,
            extracted.exponent,
            extracted.mantissa,
            extracted.index,
            inverse,
        ]);
    }
    for (i, z) in centered.into_iter().enumerate() {
        let product = builder.multiply(z, inverse)?;
        let q = floor12(&mut builder, product)?;
        let affine = builder.mul_public(q, word(spec.gamma[i]))?;
        let scaled = floor12(&mut builder, affine)?;
        let beta = builder.constant(BITS, word(spec.beta[i]))?;
        outputs.push(builder.add(scaled, beta)?);
    }
    builder.finish(&outputs)
}

#[cfg(test)]
#[path = "norm_tests.rs"]
mod tests;
