//! Crate-private caller-buffered transforms with source-fixed arithmetic.
//!
//! Borrow the existing validated public plan; do not create a new plan owner,
//! ciphertext handle, key import, admission authority, or protected backend.
//! Actual degree >4096 is refused, including Public8192 descriptors. Existing
//! public variable-time arithmetic is never used on these coefficients.
//! Public-index traversal is shared; no machine constant-time claim follows.
//!
//! Gate order: degree/checked quote/exact lengths, debit, public Montgomery
//! setup, FULL input audit, then arm guards before writes. Invalid values spend
//! work but preserve both output and scratch. After mutation, guards zeroize
//! scratch and failed destinations on unwind; only successful output disarms.
//! No allocation, entropy, callbacks, input erasure, or abort/copy guarantees.

use super::{NegacyclicNtt, schedule};
use crate::coefficient_layout::MAX_DEGREE;
use crate::coefficient_layout::modular::Montgomery64;
use crate::kernel_support::{KernelWorkLimit, WipeGuard};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Error {
    Degree,
    Geometry,
    Overflow,
    Work,
    Parameters,
    NonCanonical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Operation {
    Forward,
    Inverse,
    Product,
}

/// Separate LOCAL logical inventory, not legacy LogicalOperations, BFV units,
/// CallerWorkLedger, time, physical memory, original funding, or a tariff.
/// Distinct inventories require separate counter instances; no implicit sum.
///
/// Setup: six Newton steps, two public remainders, three modulus predicates,
/// three descriptor stores =14. Conversion: one encode/decode REDC. Scans visit
/// every input word. Each permutation reserves N index visits and <=N swap
/// stores. Each butterfly reserves two Montgomery products (INCLUDING the
/// final twiddle update), add/sub, and two coefficient stores =6 units.
/// Each nontrivial transform reserves <=N block-twiddle initializations.
/// Forward twist: two products and one store per word; inverse: three and one.
/// Pointwise reserves one product and one store per word. Boundary writes are
/// separate from conversions. Cleanup reserves success scratch AND worst
/// failure destination wipes together. Index/control and internal scalar
/// instructions are not separately priced: these are logical source events.
///
/// With L=log2(N), B=NL/2 and N>1:
/// Forward =16+12N+L+6B; Inverse =17+13N+L+6B;
/// Product =21+32N+3L+18B. N=1 keeps the original transform early return:
/// unary =14+6N; product =14+13N. Public setup/root conversions are included,
/// unlike the existing public plan's already-constructed arithmetic inventory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Quote {
    pub(crate) public_setup: u64,
    pub(crate) scans: u64,
    pub(crate) conversions: u64,
    pub(crate) permutations: u64,
    pub(crate) butterflies: u64,
    pub(crate) block_initializations: u64,
    pub(crate) twists: u64,
    pub(crate) pointwise: u64,
    pub(crate) boundary_writes: u64,
    pub(crate) cleanup: u64,
    total: u64,
}

impl Quote {
    pub(crate) fn total(self) -> u64 {
        self.total
    }
}

fn times(lhs: u64, rhs: u64) -> Result<u64, Error> {
    lhs.checked_mul(rhs).ok_or(Error::Overflow)
}

fn plus(lhs: u64, rhs: u64) -> Result<u64, Error> {
    lhs.checked_add(rhs).ok_or(Error::Overflow)
}

/// Arithmetic-only inventory seam, not a coefficient entrypoint or degree
/// opt-in. Actual quote and every executable entrypoint gate the plan first.
pub(super) fn inventory(operation: Operation, n: u64, stages: u64) -> Result<Quote, Error> {
    let product = operation == Operation::Product;
    let transforms = if product { 3 } else { 1 };
    let nontrivial = n != 1;
    let butterflies = times(n / 2, stages)?;
    let roots = if !nontrivial {
        0
    } else if product {
        plus(times(stages, 3)?, 7)?
    } else {
        plus(stages, if operation == Operation::Inverse { 3 } else { 2 })?
    };
    let mut quote = Quote {
        public_setup: 14,
        scans: times(n, if product { 2 } else { 1 })?,
        conversions: plus(times(n, if product { 3 } else { 2 })?, roots)?,
        permutations: if nontrivial { times(times(n, 2)?, transforms)? } else { 0 },
        butterflies: if nontrivial { times(times(butterflies, 6)?, transforms)? } else { 0 },
        block_initializations: if nontrivial { times(n, transforms)? } else { 0 },
        twists: if nontrivial {
            times(n, if product { 10 } else if operation == Operation::Inverse { 4 } else { 3 })?
        } else {
            0
        },
        pointwise: if product { times(n, 2)? } else { 0 },
        boundary_writes: times(n, if product { 3 } else { 2 })?,
        cleanup: times(n, if product { 3 } else { 1 })?,
        total: 0,
    };
    for units in [
        quote.public_setup,
        quote.scans,
        quote.conversions,
        quote.permutations,
        quote.butterflies,
        quote.block_initializations,
        quote.twists,
        quote.pointwise,
        quote.boundary_writes,
        quote.cleanup,
    ] {
        quote.total = plus(quote.total, units)?;
    }
    Ok(quote)
}

fn geometry(plan: &NegacyclicNtt) -> Result<(usize, usize), Error> {
    let n = plan.ring.degree();
    // Bound selector is deliberately not stored by CoefficientRing. Test the
    // actual degree, not how this descriptor happened to be constructed.
    if !n.is_power_of_two() || n > MAX_DEGREE {
        return Err(Error::Degree);
    }
    let two_n = n.checked_mul(2).ok_or(Error::Overflow)?;
    Ok((n, two_n))
}

pub(crate) fn quote(plan: &NegacyclicNtt, operation: Operation) -> Result<Quote, Error> {
    let (n, _) = geometry(plan)?;
    inventory(
        operation,
        u64::try_from(n).map_err(|_| Error::Overflow)?,
        u64::from(n.trailing_zeros()),
    )
}

fn preflight(
    plan: &NegacyclicNtt,
    operation: Operation,
    lengths: &[(usize, usize)],
    work: &mut KernelWorkLimit,
) -> Result<(), Error> {
    let units = quote(plan, operation)?.total();
    for &(actual, expected) in lengths {
        if actual != expected {
            return Err(Error::Geometry);
        }
    }
    work.debit(units).map_err(|_| Error::Work)
}

/// All words, roots, twists, unity, and N^-1 stay canonical Montgomery values.
/// Caller has validated geometry/values and encoded every coefficient. Root
/// setup depends only on the borrowed public descriptor. No decoding here.
fn transform_mont(plan: &NegacyclicNtt, domain: &Montgomery64, words: &mut [u64], inverse: bool) {
    let n = words.len();
    if n == 1 {
        return;
    }
    let one = domain.encode(1);
    let psi = domain.encode(if inverse { plan.psi_inv } else { plan.psi });
    if !inverse {
        let mut twist = one;
        for value in words.iter_mut() {
            *value = domain.mul(*value, twist);
            twist = domain.mul(twist, psi);
        }
    }
    for (low, high) in schedule::swaps(n) {
        words.swap(low, high);
    }
    let roots = if inverse { &plan.inverse_stage_roots } else { &plan.forward_stage_roots };
    for stage in schedule::stages(n) {
        let root = domain.encode(roots[stage.index]);
        for start in stage.blocks() {
            let mut twiddle = one;
            for (low, high) in stage.butterflies(start) {
                let a = words[low];
                let b = domain.mul(words[high], twiddle);
                words[low] = domain.add(a, b);
                words[high] = domain.sub(a, b);
                twiddle = domain.mul(twiddle, root);
            }
        }
    }
    if inverse {
        let n_inv = domain.encode(plan.n_inv);
        let mut twist = one;
        for value in words.iter_mut() {
            *value = domain.mul(domain.mul(*value, n_inv), twist);
            twist = domain.mul(twist, psi);
        }
    }
}

fn transform_into(
    plan: &NegacyclicNtt,
    input: &[u64],
    out: &mut [u64],
    work: &mut KernelWorkLimit,
    inverse: bool,
) -> Result<(), Error> {
    let (n, _) = geometry(plan)?;
    let operation = if inverse { Operation::Inverse } else { Operation::Forward };
    preflight(plan, operation, &[(input.len(), n), (out.len(), n)], work)?;
    let domain = Montgomery64::new(plan.ring.modulus()).ok_or(Error::Parameters)?;
    if domain.invalid(input) != 0 {
        return Err(Error::NonCanonical);
    }
    let output = WipeGuard::new(out);
    for (destination, &value) in output.words.iter_mut().zip(input) {
        *destination = domain.encode(value);
    }
    transform_mont(plan, &domain, output.words, inverse);
    for value in output.words.iter_mut() {
        *value = domain.decode(*value);
    }
    output.finish();
    Ok(())
}

/// Natural-order evaluations at psi^(2k+1); no scratch or allocation.
pub(crate) fn forward_into(
    plan: &NegacyclicNtt,
    input: &[u64],
    out: &mut [u64],
    work: &mut KernelWorkLimit,
) -> Result<(), Error> {
    transform_into(plan, input, out, work, false)
}

/// Natural-order inverse, including N^-1 and inverse twist.
pub(crate) fn inverse_into(
    plan: &NegacyclicNtt,
    input: &[u64],
    out: &mut [u64],
    work: &mut KernelWorkLimit,
) -> Result<(), Error> {
    transform_into(plan, input, out, work, true)
}

/// Raw product in Z_q[X]/(X^N+1), never an HE multiplication capability.
/// lhs/rhs/out lengths N; scratch EXACTLY 2N. Immutable inputs may alias.
pub(crate) fn negacyclic_product_into(
    plan: &NegacyclicNtt,
    lhs: &[u64],
    rhs: &[u64],
    out: &mut [u64],
    scratch: &mut [u64],
    work: &mut KernelWorkLimit,
) -> Result<(), Error> {
    let (n, two_n) = geometry(plan)?;
    preflight(
        plan,
        Operation::Product,
        &[(lhs.len(), n), (rhs.len(), n), (out.len(), n), (scratch.len(), two_n)],
        work,
    )?;
    let domain = Montgomery64::new(plan.ring.modulus()).ok_or(Error::Parameters)?;
    // Deliberately separate, unconditional complete scans: no || between arrays.
    let invalid_lhs = domain.invalid(lhs);
    let invalid_rhs = domain.invalid(rhs);
    if (invalid_lhs | invalid_rhs) != 0 {
        return Err(Error::NonCanonical);
    }
    let scratch = WipeGuard::new(scratch);
    let (left, right) = scratch.words.split_at_mut(n);
    for i in 0..n {
        left[i] = domain.encode(lhs[i]);
        right[i] = domain.encode(rhs[i]);
    }
    transform_mont(plan, &domain, left, false);
    transform_mont(plan, &domain, right, false);
    for i in 0..n {
        left[i] = domain.mul(left[i], right[i]);
    }
    transform_mont(plan, &domain, left, true);
    let output = WipeGuard::new(out);
    for i in 0..n {
        output.words[i] = domain.decode(left[i]);
    }
    output.finish();
    Ok(())
}
