//! Closed private convolution choice, not an extensible backend or HE owner.
//! Schoolbook remains the default, including odd composite q and old quotes.
//! Fixed NTT borrows an existing validated prime/root descriptor. No sampler,
//! entropy, profile, authority, activation, or machine-CT qualification follows.
//! Caller workspace is 3N BFV words plus separate 2N NTT words: 5N total,
//! 163840 declared bytes at N=4096, excluding all other buffers and stack data.

use super::arithmetic;
use super::{KernelError, KernelWorkLimit, Operation, Params, geometry};
use crate::coefficient_layout::MAX_DEGREE;
use crate::coefficient_ntt::{
    FixedError, FixedOperation, NegacyclicNtt, fixed_quote, negacyclic_product_into,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NttError {
    PlanMismatch,
    Fixed(FixedError),
}

fn fixed_error(error: FixedError) -> KernelError {
    KernelError::Ntt(NttError::Fixed(error))
}

/// FixedNtt's scratch is a reborrow of an OUTER armed WipeGuard, held by its
/// wrapper across the entire equation, including gaps between both products.
/// The nested NTT guards can never disarm that outer guard. All operands/output
/// are disjoint ordinary borrows; no alias tricks or early output publication.
pub(super) enum Convolution<'a> {
    Schoolbook,
    FixedNtt {
        plan: &'a NegacyclicNtt,
        scratch: &'a mut [u64],
        work: &'a mut KernelWorkLimit,
    },
}

impl Convolution<'_> {
    pub(super) fn product(
        &mut self,
        lhs: &[u64],
        rhs: &[u64],
        output: &mut [u64],
        q: u64,
    ) -> Result<(), KernelError> {
        match self {
            Self::Schoolbook => {
                arithmetic::convolution(lhs, rhs, output, q);
                Ok(())
            }
            Self::FixedNtt { plan, scratch, work } => {
                negacyclic_product_into(*plan, lhs, rhs, output, &mut **scratch, &mut **work)
                    .map_err(fixed_error)
            }
        }
    }
}

/// BFV-local CONSERVATIVE shell units, not a disjoint exact event inventory,
/// CPU/runtime/RSS bound, tariff, original-owner funding, or paid authority.
/// Remove 65N^2 per replaced schoolbook product, retain old linear envelopes:
/// Key 14N+2N=16N; Encrypt 21N+65N embedding+2N=88N;
/// Decrypt 148N+2N=150N. Extra 2N reserves the OUTER workspace wipe, even
/// though each nested fixed product also reserves its own scratch wipe.
/// Old linear reserves still include initialization now performed by NTT:
/// intentional conservative overlap, not exact/disjoint event accounting.
/// Separate NTT counter instances use fixed_quote(Product) units; never sum
/// across these inventories, mint child limits, reserve/refund, or convert units.
pub(super) fn shell_quote(operation: Operation, n: usize) -> Result<u64, KernelError> {
    let weight = match operation {
        Operation::Key => 16,
        Operation::Encrypt => 88,
        Operation::Decrypt => 150,
        Operation::Add => return Err(KernelError::Parameters),
    };
    u64::try_from(n)
        .map_err(|_| KernelError::Overflow)?
        .checked_mul(weight)
        .ok_or(KernelError::Overflow)
}

pub(super) fn product_count(operation: Operation) -> Result<u64, KernelError> {
    match operation {
        Operation::Key | Operation::Decrypt => Ok(1),
        Operation::Encrypt => Ok(2),
        Operation::Add => Err(KernelError::Parameters),
    }
}

/// Entirely PUBLIC preflight before coefficient reads or EITHER debit. Both
/// mutable counter borrows are exclusive through the wrapper's operation.
/// A shortage preserves both balances and every buffer. Only shell is debited
/// here; each actual product debits the ORIGINAL supplied NTT counter itself.
/// Wrappers immediately arm both outer scratch guards after this succeeds.
pub(super) fn preflight_ntt(
    params: Params,
    operation: Operation,
    lengths: &[(usize, usize)],
    shell: &mut KernelWorkLimit,
    plan: &NegacyclicNtt,
    workspace_len: usize,
    ntt: &mut KernelWorkLimit,
) -> Result<(), KernelError> {
    let (two_n, three_n) = geometry(params.n)?;
    if two_n != params.two_n || three_n != params.three_n {
        return Err(KernelError::Geometry);
    }
    if !params.n.is_power_of_two() || params.n > MAX_DEGREE {
        return Err(fixed_error(FixedError::Degree));
    }
    // fixed_quote gates the descriptor's ACTUAL degree too. Public8192 is
    // never an opt-in to secret degree 8192, even when shapes are also wrong.
    let ntt_units = fixed_quote(plan, FixedOperation::Product)
        .map_err(fixed_error)?
        .total();
    if plan.ring().degree() != params.n || plan.ring().modulus() != params.q {
        return Err(KernelError::Ntt(NttError::PlanMismatch));
    }
    for &(actual, expected) in lengths {
        if actual != expected {
            return Err(KernelError::Geometry);
        }
    }
    if workspace_len != two_n {
        return Err(fixed_error(FixedError::Geometry));
    }
    let shell_units = shell_quote(operation, params.n)?;
    let all_products = product_count(operation)?
        .checked_mul(ntt_units)
        .ok_or(KernelError::Overflow)?;
    if shell.remaining() < shell_units {
        return Err(KernelError::Work);
    }
    if ntt.remaining() < all_products {
        return Err(fixed_error(FixedError::Work));
    }
    shell.debit(shell_units).map_err(|_| KernelError::Work)
}
