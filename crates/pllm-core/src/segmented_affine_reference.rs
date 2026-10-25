//! Synthetic clear R-S reference, NOT HomComp, PBS, encrypted lookup or a kernel.
//! Fixed proposed grid/secants: docs/plans/rosetta-reciprocal-round-trip.md §2.
//! No parameter/key/source authority or physical/native-stack accounting follows.

use std::mem::size_of;

pub const PAIRS: usize = 8192;
pub const GRID: i64 = 1 << 20;
pub const RESULT_DENOMINATOR: i128 = 1 << 44;
pub const MAX_OWNER_BYTES: usize = 64 << 20;
pub const MAX_OUTPUT_BYTES: usize = 1 << 20;
pub const MAX_CASES: usize = 32768;
/// Qualification harness ceiling, not a guarantee about a host's execution time.
pub const QUALIFICATION_SECONDS: u64 = 30;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Coefficient {
    /// Integer units of 2^-24.
    pub slope: i64,
    /// Integer units of 2^-44.
    pub intercept: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Domain,
    OffGrid,
    Overflow,
    Resource,
    Poisoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Evaluation {
    pub segment: usize,
    pub local_index: usize,
    /// Exact affine value = numerator / RESULT_DENOMINATOR; no output rounding.
    pub numerator: i128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub workspace_bytes: usize,
    pub control_bytes: usize,
    pub retained_owner_bytes: usize,
    pub aggregate_bytes: usize,
    /// Conservative explicit scalar-operation bound, not elapsed/CPU time.
    pub work: u128,
}

// Name the concrete live arguments/results and old/new move slots. No claim is
// made about compiler/OS stacks, allocator arenas or encrypted backend owners.
struct Controls<'a> {
    _workspace: Option<Workspace<'a>>,
    _quote: [Resources; 2],
    _pair: [Coefficient; 2],
    _evaluation: [Result<Evaluation, Error>; 2],
    _rational: [i128; 12],
    _wide: [u128; 12],
    _indices: [usize; 8],
    _borrow: &'a mut [Coefficient],
    _storage_header: Vec<Coefficient>,
    _error_returns: [Result<Resources, Error>; 2],
}

/// Prices actual capacity of existing caller storage BEFORE initialization.
/// The allocation creating this Vec is upstream; this is no Budget authority.
pub fn resources(
    storage: &Vec<Coefficient>,
    retained_owner_bytes: usize,
    cases: usize,
) -> Result<Resources, Error> {
    if storage.len() != PAIRS || storage.capacity() != PAIRS || cases > MAX_CASES {
        return Err(Error::Resource);
    }
    let workspace_bytes = storage
        .capacity()
        .checked_mul(size_of::<Coefficient>())
        .ok_or(Error::Overflow)?;
    let control_bytes = size_of::<Controls<'static>>();
    let aggregate_bytes = retained_owner_bytes
        .checked_add(workspace_bytes)
        .and_then(|n| n.checked_add(control_bytes))
        .ok_or(Error::Overflow)?;
    if workspace_bytes > MAX_OUTPUT_BYTES || aggregate_bytes > MAX_OWNER_BYTES {
        return Err(Error::Resource);
    }
    // Scalar-source operations, each checked arithmetic/comparison/conversion,
    // loop visit or write costs one (NOT machine instructions/division latency).
    // Each coefficient call <=64: domain4, endpoint4, denominator4, slope
    // shift4+RNE12, intercept4+RNE12, conversions/results8, remaining12 for
    // checked branch/early-return controls. Each pass adds <=8 loop/index/write
    // operations; retirement adds2 per pair. Fixed quote controls <=128.
    // Each evaluation <=128 including threshold/index/arithmetic/result slots.
    let work = 128 + (PAIRS as u128) * (2 * (64 + 8) + 2) + (cases as u128) * 128;
    Ok(Resources {
        workspace_bytes,
        control_bytes,
        retained_owner_bytes,
        aggregate_bytes,
        work,
    })
}

fn rne(n: u128, d: u128) -> Result<u128, Error> {
    if d == 0 {
        return Err(Error::Domain);
    }
    let q = n / d;
    let r = n % d;
    let twice = r.checked_mul(2).ok_or(Error::Overflow)?;
    q.checked_add(u128::from(twice > d || (twice == d && q & 1 != 0)))
        .ok_or(Error::Overflow)
}

fn coefficient(segment: usize, index: usize) -> Result<Coefficient, Error> {
    if segment >= 4 || index >= 2048 {
        return Err(Error::Domain);
    }
    let n = 2048 + index as u128;
    let d = n.checked_mul(n + 1).ok_or(Error::Overflow)?;
    // l=2^s*n/2048, u=2^s*(n+1)/2048. RNE is symmetric for negatives.
    let a = rne(
        1u128
            .checked_shl(46 - 2 * segment as u32)
            .ok_or(Error::Overflow)?,
        d,
    )?;
    let b = rne(
        (1u128 << (55 - segment))
            .checked_mul(2 * n + 1)
            .ok_or(Error::Overflow)?,
        d,
    )?;
    Ok(Coefficient {
        slope: i64::try_from(a)
            .map_err(|_| Error::Overflow)?
            .checked_neg()
            .ok_or(Error::Overflow)?,
        intercept: i64::try_from(b).map_err(|_| Error::Overflow)?,
    })
}

/// Exact rational fixture conversion; negative/off-grid/end16 inputs refuse.
pub fn grid_input(numerator: i128, denominator: u128) -> Result<i64, Error> {
    if denominator == 0 || numerator < 0 {
        return Err(Error::Domain);
    }
    let scaled = (numerator as u128)
        .checked_mul(GRID as u128)
        .ok_or(Error::Overflow)?;
    if scaled % denominator != 0 {
        return Err(Error::OffGrid);
    }
    let x = i64::try_from(scaled / denominator).map_err(|_| Error::Overflow)?;
    index(x)?;
    Ok(x)
}

/// Binary64 fixture admission only; exact grid reconstruction, never rounding.
pub fn float_grid_input(x: f64) -> Result<i64, Error> {
    if !x.is_finite() || !(1.0..16.0).contains(&x) {
        return Err(Error::Domain);
    }
    let scaled = x * GRID as f64; // exact power-of-two scaling on this domain
    let code = scaled as i64;
    if code as f64 != scaled {
        return Err(Error::OffGrid);
    }
    Ok(code)
}

fn index(x: i64) -> Result<(usize, usize), Error> {
    if !(GRID..16 * GRID).contains(&x) {
        return Err(Error::Domain);
    }
    let segment =
        usize::from(x >= 2 * GRID) + usize::from(x >= 4 * GRID) + usize::from(x >= 8 * GRID);
    Ok((segment, ((x - (GRID << segment)) >> (9 + segment)) as usize))
}

pub struct Workspace<'a> {
    pairs: &'a mut [Coefficient],
    remaining: usize,
    poisoned: bool,
}

impl<'a> Workspace<'a> {
    /// All resource checks and an arithmetic replay precede the first write.
    /// Caller retains any real original reservation through this borrow.
    pub fn prepare(
        storage: &'a mut Vec<Coefficient>,
        retained_owner_bytes: usize,
        cases: usize,
        byte_limit: usize,
        work_limit: u128,
    ) -> Result<Self, Error> {
        let quote = resources(storage, retained_owner_bytes, cases)?;
        if byte_limit < quote.aggregate_bytes || work_limit < quote.work {
            return Err(Error::Resource);
        }
        for s in 0..4 {
            for i in 0..2048 {
                coefficient(s, i)?;
            }
        }
        for (at, pair) in storage.iter_mut().enumerate() {
            *pair = coefficient(at / 2048, at % 2048)?;
        }
        Ok(Self {
            pairs: storage,
            remaining: cases,
            poisoned: false,
        })
    }

    pub fn pairs(&self) -> Result<&[Coefficient], Error> {
        if self.poisoned {
            Err(Error::Poisoned)
        } else {
            Ok(self.pairs)
        }
    }
}

pub fn evaluate_reference(x: i64, workspace: &mut Workspace<'_>) -> Result<Evaluation, Error> {
    if workspace.poisoned {
        return Err(Error::Poisoned);
    }
    // Failure permanently poisons; no suffix retry or truncation of a campaign.
    workspace.poisoned = true;
    if workspace.remaining == 0 {
        return Err(Error::Resource);
    }
    let (segment, local_index) = index(x)?;
    let c = workspace.pairs[segment * 2048 + local_index];
    let numerator = i128::from(c.slope)
        .checked_mul(i128::from(x))
        .and_then(|v| v.checked_add(i128::from(c.intercept)))
        .ok_or(Error::Overflow)?;
    workspace.remaining -= 1;
    workspace.poisoned = false;
    Ok(Evaluation {
        segment,
        local_index,
        numerator,
    })
}

impl Drop for Workspace<'_> {
    fn drop(&mut self) {
        self.pairs.fill(Coefficient::default());
    }
}

#[cfg(test)]
#[path = "segmented_affine_reference_tests.rs"]
mod tests;
