//! Clear local equations, not a model profile, source owner or admission token.
//! D31 numeric-source-recovery §3.2–3.3: no stock-Qwen kernel equivalence claim.
use super::{add_rn32, f32_to_bf16, mul_rn32, promote_bf16, Error, FINITE_R32_WORK};

// OTHER is separate from prepaid primitive calls. Per coordinate: at most eight
// promotions and three casts (<64 logical operations each), iterator/index/sign
// glue and Result dispatch (<256), plus full active-output cleanup (<256).
// Reserve 2048 even for the simpler elementwise/dot paths. Fixed 4096 covers
// length/geometry checks, checked quote arithmetic, slicing and error dispatch.
// Logical work only: no CPU/time/physical-stack promise or original-ledger credit.
const OTHER_PER_COORDINATE: usize = 2048;
const OTHER_FIXED: usize = 4096;

fn quote(width: usize, calls: usize) -> Result<usize, Error> {
    width
        .checked_mul(calls)
        .and_then(|n| n.checked_mul(FINITE_R32_WORK))
        .and_then(|n| width.checked_mul(OTHER_PER_COORDINATE)?.checked_add(n))
        .and_then(|n| n.checked_add(OTHER_FIXED))
        .ok_or(Error::WorkLimit)
}

/// Checked logical allowance, including one RN32 call and full failure cleanup
/// per coordinate. Empty elementwise arrays are permitted.
pub fn add_bf16_work(width: usize) -> Result<usize, Error> {
    quote(width, 1)
}

/// Same allowance as addition; no allocation or serving admission implied.
pub fn mul_bf16_work(width: usize) -> Result<usize, Error> {
    quote(width, 1)
}

/// Two separately rounded RN32 calls per coordinate. Empty dot returns +0.
pub fn dot_bf16_rn32_work(width: usize) -> Result<usize, Error> {
    quote(width, 2)
}

/// One supplied-coefficient half-rotation head; width must be positive and even.
/// Three RN32 calls per coordinate, with independent BF16 product casts.
/// Generic scalar algebra does not change any model's fixed head geometry.
pub fn rope_bf16_supplied_work(width: usize) -> Result<usize, Error> {
    if width == 0 || width % 2 != 0 {
        return Err(Error::Geometry);
    }
    quote(width, 3)
}

fn budget(work: usize, limit: usize) -> Result<(), Error> {
    if work > limit { Err(Error::WorkLimit) } else { Ok(()) }
}

fn lengths(width: usize, other: usize, capacity: usize) -> Result<(), Error> {
    if width != other {
        return Err(Error::Length);
    }
    if capacity < width {
        return Err(Error::Capacity);
    }
    Ok(())
}

type Binary = fn(f32, f32) -> Result<f32, Error>;

fn binary_into(
    left: &[u16],
    right: &[u16],
    output: &mut [u16],
    work_limit: usize,
    operation: Binary,
) -> Result<(), Error> {
    lengths(left.len(), right.len(), output.len())?;
    budget(quote(left.len(), 1)?, work_limit)?;
    let active = &mut output[..left.len()];
    let result = (|| {
        for ((&a, &b), slot) in left.iter().zip(right).zip(active.iter_mut()) {
            *slot = f32_to_bf16(operation(promote_bf16(a)?, promote_bf16(b)?)?)?;
        }
        Ok(())
    })();
    if result.is_err() {
        active.fill(0);
    }
    result
}

/// R16(R32(a+b)) coordinatewise. Length/capacity/work preflight visits no
/// elements and leaves output unchanged. Arithmetic/conversion failure clears
/// the entire active prefix; spare tail is always untouched. Publish only on Ok.
pub fn add_bf16_into(
    left: &[u16], right: &[u16], output: &mut [u16], work_limit: usize,
) -> Result<(), Error> {
    binary_into(left, right, output, work_limit, add_rn32)
}

/// R16(R32(a*b)); same preflight/cleanup/tail contract as `add_bf16_into`.
pub fn mul_bf16_into(
    left: &[u16], right: &[u16], output: &mut [u16], work_limit: usize,
) -> Result<(), Error> {
    binary_into(left, right, output, work_limit, mul_rn32)
}

/// Increasing-coordinate, NONFUSED RN32 multiply then RN32 add starting +0.
/// No BF16 accumulator cast, reassociation, FTZ, host float arithmetic or FMA.
/// Nonfinite operands/intermediate overflow propagate the native primitive error.
pub fn dot_bf16_rn32(left: &[u16], right: &[u16], work_limit: usize) -> Result<f32, Error> {
    if left.len() != right.len() {
        return Err(Error::Length);
    }
    budget(dot_bf16_rn32_work(left.len())?, work_limit)?;
    let mut sum = f32::from_bits(0);
    for (&a, &b) in left.iter().zip(right) {
        let product = mul_rn32(promote_bf16(a)?, promote_bf16(b)?)?;
        sum = add_rn32(sum, product)?;
    }
    Ok(sum)
}

/// Supplied finite BF16 cos/sin only; coefficient generation is out of scope.
/// For one even-width head, rot(x)=concat(-x[half..],x[..half]). Evaluate
/// add16(mul16(x[k],cos[k]),mul16(rot(x)[k],sin[k])) in that order.
/// Coefficients have full head length; caller supplies any required duplication.
/// Preflight/active-cleanup/spare-tail contract matches `add_bf16_into`.
pub fn rope_bf16_supplied_into(
    input: &[u16], cos: &[u16], sin: &[u16], output: &mut [u16], work_limit: usize,
) -> Result<(), Error> {
    let work = rope_bf16_supplied_work(input.len())?;
    lengths(input.len(), cos.len(), output.len())?;
    if sin.len() != input.len() {
        return Err(Error::Length);
    }
    budget(work, work_limit)?;
    let half = input.len() / 2;
    let active = &mut output[..input.len()];
    let result = (|| {
        for (k, slot) in active.iter_mut().enumerate() {
            let rotated = if k < half { input[k + half] ^ 0x8000 } else { input[k - half] };
            let first = f32_to_bf16(mul_rn32(promote_bf16(input[k])?, promote_bf16(cos[k])?)?)?;
            let second = f32_to_bf16(mul_rn32(promote_bf16(rotated)?, promote_bf16(sin[k])?)?)?;
            *slot = f32_to_bf16(add_rn32(promote_bf16(first)?, promote_bf16(second)?)?)?;
        }
        Ok(())
    })();
    if result.is_err() {
        active.fill(0);
    }
    result
}
