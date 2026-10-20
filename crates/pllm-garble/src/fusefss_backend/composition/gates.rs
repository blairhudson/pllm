//! Fixed D8 gate lowerings. Public intervals are caller assumptions, not verified
//! checkpoint provenance or secret membership checks. Every numeric obligation is
//! checked before constructing tables; no assertion digest replaces these checks.
use super::super::{
    conversions::{Operation, SignedRange},
    Budget, Error,
};
use super::{source_tables, Builder, Plan, Value};
use sha2::{Digest, Sha256};

const PROFILE: &[u8] = b"fusefss.source-derived-ring50-q12.v1";
const HALF: i128 = 1i128 << 49;
// Contract/width encoding plus two fixed row work arrays coexist with Builder.
// Reserve this public stack workspace too; retained plan owners charge separately.
const LOWERING_SCRATCH_BYTES: usize = 64 * 1024;

/// Inclusive public bounds on reconstructed integer codes. Not an input validator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputRange {
    pub min: i64,
    pub max: i64,
}

impl InputRange {
    fn checked(self) -> Result<(i128, i128), Error> {
        let lo = i128::from(self.min);
        let hi = i128::from(self.max);
        if lo > hi {
            return Err(Error::Domain);
        }
        signed50(lo)?;
        signed50(hi)?;
        Ok((lo, hi))
    }
}

/// Positions in Plan's returned share vector, never internal node identifiers.
/// Production plans expose only `final_outputs`. Extra fixture fields exist only
/// in test builds and still remain shares until test-only reconstruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputLayout {
    pub final_outputs: std::ops::Range<usize>,
    #[cfg(test)]
    pub intermediates: Vec<(&'static str, std::ops::Range<usize>)>,
}

fn signed50(x: i128) -> Result<i128, Error> {
    if (-HALF..HALF).contains(&x) {
        Ok(x)
    } else {
        Err(Error::Domain)
    }
}
fn add(a: i128, b: i128) -> Result<i128, Error> {
    signed50(a.checked_add(b).ok_or(Error::Domain)?)
}
fn sub(a: i128, b: i128) -> Result<i128, Error> {
    signed50(a.checked_sub(b).ok_or(Error::Domain)?)
}
fn mul(a: i128, b: i128) -> Result<i128, Error> {
    signed50(a.checked_mul(b).ok_or(Error::Domain)?)
}

fn source_digest() -> [u8; 32] {
    source_tables::source_digest()
}

fn builder<'a>(
    gate: &[u8],
    ranges: &[InputRange],
    active: &[bool],
    parameter: usize,
    budget: &'a Budget,
) -> Result<Builder<'a>, Error> {
    builder_source(gate, ranges, active, parameter, budget, source_digest())
}
fn builder_source<'a>(
    gate: &[u8],
    ranges: &[InputRange],
    active: &[bool],
    parameter: usize,
    budget: &'a Budget,
    source: [u8; 32],
) -> Result<Builder<'a>, Error> {
    // Internally computed commitment to numerically checked metadata. This is
    // not a caller-supplied proof digest. Streaming avoids unreserved buffers and
    // preserves Builder's fixed contract-size ceiling for full 1024-entry rows.
    let mut contract = Sha256::new();
    for bytes in [PROFILE, b"\0caller-assumed-inclusive-ranges\0", gate] {
        contract.update((bytes.len() as u64).to_le_bytes());
        contract.update(bytes);
    }
    contract.update((ranges.len() as u64).to_le_bytes());
    contract.update((parameter as u64).to_le_bytes());
    for range in ranges {
        for value in [range.min, range.max] {
            contract.update(value.to_le_bytes());
        }
    }
    for &bit in active {
        contract.update([u8::from(bit)]);
    }
    let contract: [u8; 32] = contract.finalize().into();
    let widths = [50u8; 1024];
    Builder::new(source, &contract, &widths[..ranges.len()], budget)
}

fn finish<'a>(
    b: Builder<'a>,
    finals: &[Value],
    intermediates: &[(&'static str, &[Value])],
) -> Result<(Plan<'a>, OutputLayout), Error> {
    #[cfg(not(test))]
    {
        let _ = intermediates;
        Ok((
            b.finish(finals)?,
            OutputLayout {
                final_outputs: 0..finals.len(),
            },
        ))
    }
    #[cfg(test)]
    {
        let mut outputs = finals.to_vec();
        let mut fields = Vec::new();
        for &(name, values) in intermediates {
            let start = outputs.len();
            outputs.extend_from_slice(values);
            fields.push((name, start..outputs.len()));
        }
        Ok((
            b.finish(&outputs)?,
            OutputLayout {
                final_outputs: 0..finals.len(),
                intermediates: fields,
            },
        ))
    }
}

fn tr_positive(b: &mut Builder<'_>, x: Value, shift: u8) -> Result<Value, Error> {
    let x = b.convert(
        x,
        Operation::Tr {
            shift,
            range: SignedRange::full_ring(50)?,
        },
        50 - shift,
    )?;
    b.convert(x, Operation::ZeroExtend, 50)
}

/// Source-effective signed38/Q12 SiLU, retaining ORIGINAL input in Phi.
pub fn silu(range: InputRange, budget: &Budget) -> Result<(Plan<'_>, OutputLayout), Error> {
    silu_inner(range, budget, None)
}
/// Same immutable schedule/numeric identity, retaining a request-owned source
/// Table rather than copying/hashing its payload at every scalar invocation.
pub fn silu_cached<'a>(
    range: InputRange,
    cache: &super::cache::SourceTableCache<'a>,
) -> Result<(Plan<'a>, OutputLayout), Error> {
    silu_inner(range, cache.budget(), Some(cache))
}
fn silu_inner<'a>(
    range: InputRange,
    budget: &'a Budget,
    cache: Option<&super::cache::SourceTableCache<'a>>,
) -> Result<(Plan<'a>, OutputLayout), Error> {
    let (lo, hi) = range.checked()?;
    if lo < -(1i128 << 37) || hi >= 1i128 << 37 {
        return Err(Error::Domain);
    }
    // abs(floor(X/64)) and both selector differences remain signed50, including
    // X=-2^37, whose reduced-ring TR word denotes -2^31.
    let abs = (lo.div_euclid(64)).abs().max((hi.div_euclid(64)).abs());
    mul(abs, 2)?;
    sub(lo.min(0), 65_536)?;
    signed50(hi.max(0))?;
    let _scratch = budget.reserve(LOWERING_SCRATCH_BYTES)?;
    let mut b = match cache {
        Some(c) => builder_source(b"silu", &[range], &[], 0, budget, c.numeric_revision())?,
        None => builder(b"silu", &[range], &[], 0, budget)?,
    };
    let table = match cache {
        Some(c) => b.source_table(c, super::cache::Kind::Silu)?,
        None => b.table(50, source_tables::silu())?,
    };
    let original = b.input(0)?;
    let x38 = b.narrow(original, 38)?;
    let y32 = b.convert(
        x38,
        Operation::Tr {
            shift: 6,
            range: SignedRange::full_ring(38)?,
        },
        32,
    )?;
    let y = b.convert(
        y32,
        Operation::SignExtend {
            range: SignedRange::full_ring(32)?,
        },
        50,
    )?;
    let negative = b.lt_signed(y, 0)?;
    let zero = b.constant(50, 0)?;
    let negated = b.sub(zero, y)?;
    let absolute = b.select(negative, negated, y)?;
    let in_table = b.lt_unsigned(absolute, 1024)?;
    let cap = b.constant(50, 1023)?;
    let index = b.select(in_table, absolute, cap)?;
    let correction = b.lookup(index, table)?;
    let selected = b.select(negative, zero, original)?;
    let result = b.sub(selected, correction)?;
    finish(b, &[result], &[("floor", &[y]), ("index", &[index])])
}

fn check_nexp(range: InputRange) -> Result<(), Error> {
    let (lo, hi) = range.checked()?;
    if lo < 0 || hi > 65535 {
        return Err(Error::Domain);
    }
    mul(4096, 4096)?;
    Ok(())
}

fn nexp_node(b: &mut Builder<'_>, c: Value, tables: (usize, usize)) -> Result<Value, Error> {
    let low = b.narrow(c, 8)?;
    let low = b.convert(low, Operation::ZeroExtend, 50)?;
    let high = tr_positive(b, c, 8)?;
    let a = b.lookup(low, tables.0)?;
    let d = b.lookup(high, tables.1)?;
    let product = b.multiply(a, d)?;
    tr_positive(b, product, 12)
}

/// Exact source nExp on unsigned C in [0,65535], Q12 output.
pub fn nexp(range: InputRange, budget: &Budget) -> Result<(Plan<'_>, OutputLayout), Error> {
    check_nexp(range)?;
    let _scratch = budget.reserve(LOWERING_SCRATCH_BYTES)?;
    let mut b = builder(b"nexp", &[range], &[], 0, budget)?;
    let tables = (
        b.table(50, source_tables::exp0())?,
        b.table(50, source_tables::exp1())?,
    );
    let c = b.input(0)?;
    let result = nexp_node(&mut b, c, tables)?;
    finish(b, &[result], &[])
}

fn reciprocal_len(k: usize) -> Result<usize, Error> {
    if !(1..=1024).contains(&k) {
        return Err(Error::Domain);
    }
    let q = 6 + (usize::BITS - k.leading_zeros());
    1usize.checked_shl(q).ok_or(Error::Resource)
}

fn reciprocal_table(b: &mut Builder<'_>, k: usize, budget: &Budget) -> Result<usize, Error> {
    reciprocal_len(k)?;
    let values = source_tables::reciprocal(k, budget)?;
    b.table(50, &values)
}

/// Reciprocal of Q6 denominator; public k binds full table including sentinel.
pub fn reciprocal(
    k: usize,
    range: InputRange,
    budget: &Budget,
) -> Result<(Plan<'_>, OutputLayout), Error> {
    reciprocal_len(k)?;
    let (lo, hi) = range.checked()?;
    let upper = mul(64, k as i128)?;
    if lo < 64 || hi > upper {
        return Err(Error::Domain);
    }
    let _scratch = budget.reserve(LOWERING_SCRATCH_BYTES)?;
    let mut b = builder(b"reciprocal", &[range], &[], k, budget)?;
    let table = reciprocal_table(&mut b, k, budget)?;
    let z = b.input(0)?;
    let result = b.lookup(z, table)?;
    finish(b, &[result], &[])
}

fn check_softmax(ranges: &[InputRange], active: &[bool]) -> Result<usize, Error> {
    if ranges.is_empty() || ranges.len() > 1024 || ranges.len() != active.len() {
        return Err(Error::Domain);
    }
    let mut lo = HALF;
    let mut hi = -HALF;
    let mut k = 0;
    for (&range, &enabled) in ranges.iter().zip(active) {
        let (a, z) = range.checked()?;
        if enabled {
            lo = lo.min(a);
            hi = hi.max(z);
            k += 1;
        }
    }
    reciprocal_len(k)?;
    let span = sub(hi, lo)?;
    // Protected max compares signed differences, and Phi evaluates yes-no.
    signed50(-span)?;
    mul(4096, k as i128)?;
    mul(4096, 4096)?;
    add(0, mul(4096, k as i128)?)?;
    Ok(k)
}

/// Full source row softmax under public causal mask. Masked entries are never
/// compared, exponentiated, or accumulated. No output renormalization.
pub fn softmax<'a>(
    ranges: &[InputRange],
    active: &[bool],
    budget: &'a Budget,
) -> Result<(Plan<'a>, OutputLayout), Error> {
    softmax_inner(ranges, active, budget, None)
}
/// Cached source constants only; every lookup/input still receives fresh native
/// material and executes every original public-table evaluation.
pub fn softmax_cached<'a>(
    ranges: &[InputRange],
    active: &[bool],
    cache: &super::cache::SourceTableCache<'a>,
) -> Result<(Plan<'a>, OutputLayout), Error> {
    softmax_inner(ranges, active, cache.budget(), Some(cache))
}
fn softmax_inner<'a>(
    ranges: &[InputRange],
    active: &[bool],
    budget: &'a Budget,
    cache: Option<&super::cache::SourceTableCache<'a>>,
) -> Result<(Plan<'a>, OutputLayout), Error> {
    let k = check_softmax(ranges, active)?;
    let _scratch = budget.reserve(LOWERING_SCRATCH_BYTES)?;
    let mut b = match cache {
        Some(c) => builder_source(
            b"softmax-public-causal-mask",
            ranges,
            active,
            k,
            budget,
            c.numeric_revision(),
        )?,
        None => builder(b"softmax-public-causal-mask", ranges, active, k, budget)?,
    };
    let (tables, inverse_table) = match cache {
        Some(c) => (
            (
                b.source_table(c, super::cache::Kind::Exp0)?,
                b.source_table(c, super::cache::Kind::Exp1)?,
            ),
            b.source_table(c, super::cache::Kind::Reciprocal { k })?,
        ),
        None => (
            (
                b.table(50, source_tables::exp0())?,
                b.table(50, source_tables::exp1())?,
            ),
            reciprocal_table(&mut b, k, budget)?,
        ),
    };
    let zero = b.constant(50, 0)?;
    let first = active.iter().position(|&bit| bit).ok_or(Error::Domain)?;
    let mut maximum = b.input(first)?;
    for (i, &enabled) in active.iter().enumerate().skip(first + 1) {
        if !enabled {
            continue;
        }
        let x = b.input(i)?;
        let difference = b.sub(x, maximum)?;
        let less = b.lt_signed(difference, 0)?;
        maximum = b.select(less, maximum, x)?;
    }
    let cap = b.constant(50, 65535)?;
    let mut exps = [zero; 1024];
    let mut total = zero;
    for (i, &enabled) in active.iter().enumerate() {
        if !enabled {
            continue;
        }
        let x = b.input(i)?;
        let difference = b.sub(maximum, x)?;
        let small = b.lt_unsigned(difference, 65536)?;
        let c = b.select(small, difference, cap)?;
        exps[i] = nexp_node(&mut b, c, tables)?;
        total = b.add(total, exps[i])?;
    }
    let denominator = tr_positive(&mut b, total, 6)?;
    let inverse = b.lookup(denominator, inverse_table)?;
    let mut outputs = [zero; 1024];
    for (i, &enabled) in active.iter().enumerate() {
        if enabled {
            let product = b.multiply(exps[i], inverse)?;
            outputs[i] = tr_positive(&mut b, product, 12)?;
        }
    }
    finish(
        b,
        &outputs[..ranges.len()],
        &[
            ("exp", &exps[..ranges.len()]),
            ("denominator", &[denominator]),
            ("reciprocal", &[inverse]),
        ],
    )
}

#[cfg(test)]
#[path = "gates_tests.rs"]
mod tests;
