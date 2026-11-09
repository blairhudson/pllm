//! Private, allocation-free raw equations in Z_q[X]/(X^N+1).
//!
//! Supplied coefficients only: no sampler, entropy, distribution, paper default,
//! issued key, authentication label, owner import, or protected execution. Raw
//! buffers cannot establish key identity, provenance, or addition depth. The
//! correctness bound covers valid constructions and at most ONE addition, not
//! arbitrary ciphertexts or repeated addition. Decoding arbitrary canonical
//! ciphertexts still implements the stated rounding equation.
//!
//! Reuse audit: CoefficientRing::new is a pure nonallocating descriptor check;
//! it preserves the legacy 4096 degree cap and admits odd composite moduli.
//! Existing public modular helpers and public NTT/repacking use value-dependent
//! arithmetic and are NOT used on secret coefficients here. Existing clear phase
//! Work and row_fold ledgers carry different contracts; this counter is local,
//! not CallerWorkLedger, funding, CPU time, key purpose, or admission authority.
//! GKM transport, seeded entropy, and source custody do not transfer here.
//!
//! Source-level fixed schedules are not machine constant-time certification.
//! Zeroize guards cover borrowed scratch/destination on unwind, not aborts,
//! registers, compiler copies, caller inputs, or universal OOM recovery. Caller
//! retains input-erasure responsibility. Funded original ownership, custody,
//! security review, RNS/CKKS, rotations, key switching, and private indexing are
//! absent. No new dependencies, allocation, callbacks, or native constructor.
//! Shared guard mechanics do not change this kernel's arm policy: scratch is
//! armed after debit BEFORE value validation; destination immediately before
//! its first write, disarmed only on success. Invalid values still wipe scratch.

mod arithmetic;
mod bfv;

use crate::coefficient_layout::CoefficientRing;
use crate::kernel_support::{KernelWorkLimit, WipeGuard};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KernelError {
    Parameters,
    Geometry,
    Overflow,
    Work,
    InvalidValue,
}

/// Explicit magnitude bounds, each strictly below q. No distribution implied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Supports {
    pub(super) s: u64,
    pub(super) u: u64,
    pub(super) e: u64,
    pub(super) e0: u64,
    pub(super) e1: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Params {
    n: usize,
    two_n: usize,
    three_n: usize,
    q: u64,
    t: u64,
    delta: u64,
    supports: Supports,
}

impl Params {
    pub(super) fn new(n: usize, q: u64, t: u64, supports: Supports) -> Result<Self, KernelError> {
        CoefficientRing::new(n, q).map_err(|_| KernelError::Parameters)?;
        let (two_n, three_n) = geometry(n)?;
        if t < 2 || t >= q {
            return Err(KernelError::Parameters);
        }
        for bound in [supports.s, supports.u, supports.e, supports.e0, supports.e1] {
            if bound >= q {
                return Err(KernelError::Parameters);
            }
        }
        let n_wide = n as u128;
        let first = n_wide
            .checked_mul(u128::from(supports.e))
            .and_then(|v| v.checked_mul(u128::from(supports.u)))
            .ok_or(KernelError::Overflow)?;
        let last = n_wide
            .checked_mul(u128::from(supports.e1))
            .and_then(|v| v.checked_mul(u128::from(supports.s)))
            .ok_or(KernelError::Overflow)?;
        let noise = first
            .checked_add(u128::from(supports.e0))
            .and_then(|v| v.checked_add(last))
            .ok_or(KernelError::Overflow)?;
        // B=N*E*U+E0+N*E1*S. With r=q%t, require
        // 2*t*(2*B+2*r)<q, including floor-Delta's encoding gap.
        // For two valid messages M<=2(t-1), |V|<=2B, phase numerator
        // error is t*V-r*M. This strict bound places it inside (-q/2,q/2),
        // including negative error and plaintext wrap. No primality needed.
        let gap = u128::from(q % t);
        let bound = noise
            .checked_mul(2)
            .and_then(|v| gap.checked_mul(2).and_then(|g| v.checked_add(g)))
            .and_then(|v| v.checked_mul(u128::from(t)))
            .and_then(|v| v.checked_mul(2))
            .ok_or(KernelError::Overflow)?;
        if bound >= u128::from(q) {
            return Err(KernelError::Parameters);
        }
        Ok(Self {
            n,
            two_n,
            three_n,
            q,
            t,
            delta: q / t,
            supports,
        })
    }
}

fn geometry(n: usize) -> Result<(usize, usize), KernelError> {
    let two = n.checked_mul(2).ok_or(KernelError::Overflow)?;
    let three = n.checked_mul(3).ok_or(KernelError::Overflow)?;
    Ok((two, three))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Operation {
    Key,
    Encrypt,
    Add,
    Decrypt,
}

/// Logical source upper bound, NOT instructions, CPU time, or production tariff.
///
/// Each coefficient scan/conversion counts one unit. Each product counts 64
/// double-and-add rounds plus one accumulation (embedding uses this bound too).
/// Decryption adds 128 restoring-division steps and eight scalar decode units
/// per coefficient. Buffer units count initialization, final modular arithmetic,
/// destination/phase stores, scratch wipe, and failure-destination wipe:
/// Key: N+N+2N+3N+2N=9N; Encrypt: 2N+3N+2N+3N+2N=12N;
/// Add: 0+2N+2N+0+2N=6N; Decrypt: N+N+2N+3N+N=8N.
/// Both success stores and worst failure cleanup are reserved together. Public
/// loop/control overhead is not a separate unit: these are logical pass/round
/// counts, not word-instruction counts or a runtime upper bound.
///
///                 scans  conversions  products       decode    buffer reserve
/// Key              3N       2N        N^2              0           9N
/// Encrypt          6N       3N        2N^2+N           0          12N
/// Add              4N        0        0                0           6N
/// Decrypt          3N        N        N^2             136N          8N
///
/// All public quote arithmetic is checked before any coefficient is read.
pub(super) fn quote(operation: Operation, n: usize) -> Result<u64, KernelError> {
    let n = u64::try_from(n).map_err(|_| KernelError::Overflow)?;
    let square = n.checked_mul(n).ok_or(KernelError::Overflow)?;
    let (products, linear) = match operation {
        Operation::Key => (square, 14),
        Operation::Encrypt => (
            square
                .checked_mul(2)
                .and_then(|v| v.checked_add(n))
                .ok_or(KernelError::Overflow)?,
            21,
        ),
        Operation::Add => (0, 10),
        Operation::Decrypt => (square, 148),
    };
    products
        .checked_mul(65)
        .and_then(|v| n.checked_mul(linear).and_then(|l| v.checked_add(l)))
        .ok_or(KernelError::Overflow)
}

/// Geometry and quote rejection leave all buffers and counter unchanged.
/// Params are constructible only through new; repeat checked geometry here so
/// each operation establishes its own indexing/loop bounds before reading data.
fn preflight(
    params: Params,
    operation: Operation,
    lengths: &[(usize, usize)],
    work: &mut KernelWorkLimit,
) -> Result<(), KernelError> {
    let (two_n, three_n) = geometry(params.n)?;
    if two_n != params.two_n || three_n != params.three_n {
        return Err(KernelError::Geometry);
    }
    for &(actual, expected) in lengths {
        if actual != expected {
            return Err(KernelError::Geometry);
        }
    }
    let units = quote(operation, params.n)?;
    work.debit(units).map_err(|_| KernelError::Work)
}

#[cfg(test)]
mod tests;
