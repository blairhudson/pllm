//! Actual raw key/encryption/addition/decryption equations, not issued keys.
//! Signed inputs have explicit support bounds; all other inputs are canonical.
//! Scratch is exactly 3N, pk/ct exactly 2N, all other polynomials exactly N.
//! After successful preflight, failures consume work and wipe scratch, preserving
//! untouched output. Once writing starts, destination wipes on failure/unwind.
//! Opt-in wrappers additionally borrow separate 2N NTT workspace and its own
//! caller counter. Both balances/shapes/plans are checked before either debit.

use super::arithmetic as ar;
use super::convolution::{Convolution, preflight_ntt};
use super::{KernelError, KernelWorkLimit, Operation, Params, WipeGuard, preflight};
use crate::coefficient_ntt::NegacyclicNtt;

/// pk=(-a*s+e,a). Inputs borrowed, never sampled or retained.
pub(super) fn key_equation_into(
    params: Params,
    s: &[i64],
    a: &[u64],
    e: &[i64],
    pk: &mut [u64],
    scratch: &mut [u64],
    work: &mut KernelWorkLimit,
) -> Result<(), KernelError> {
    let n = params.n;
    preflight(
        params,
        Operation::Key,
        &[
            (s.len(), n),
            (a.len(), n),
            (e.len(), n),
            (pk.len(), params.two_n),
            (scratch.len(), params.three_n),
        ],
        work,
    )?;
    key_equation_body(params, s, a, e, pk, scratch, Convolution::Schoolbook)
}

/// Opt-in only: no plan construction, fallback, key import, or authority.
pub(super) fn key_equation_ntt_into(
    params: Params,
    s: &[i64],
    a: &[u64],
    e: &[i64],
    pk: &mut [u64],
    scratch: &mut [u64],
    work: &mut KernelWorkLimit,
    plan: &NegacyclicNtt,
    workspace: &mut [u64],
    ntt_work: &mut KernelWorkLimit,
) -> Result<(), KernelError> {
    let n = params.n;
    preflight_ntt(
        params,
        Operation::Key,
        &[
            (s.len(), n),
            (a.len(), n),
            (e.len(), n),
            (pk.len(), params.two_n),
            (scratch.len(), params.three_n),
        ],
        work,
        plan,
        workspace.len(),
        ntt_work,
    )?;
    let workspace = WipeGuard::new(workspace);
    key_equation_body(
        params,
        s,
        a,
        e,
        pk,
        scratch,
        Convolution::FixedNtt { plan, scratch: &mut *workspace.words, work: ntt_work },
    )
}

fn key_equation_body(
    params: Params,
    s: &[i64],
    a: &[u64],
    e: &[i64],
    pk: &mut [u64],
    scratch: &mut [u64],
    mut convolution: Convolution<'_>,
) -> Result<(), KernelError> {
    let n = params.n;
    let scratch = WipeGuard::new(scratch);
    let mut invalid = ar::invalid_signed(s, params.supports.s);
    invalid |= ar::invalid_canonical(a, params.q);
    invalid |= ar::invalid_signed(e, params.supports.e);
    if invalid != 0 {
        return Err(KernelError::InvalidValue);
    }
    let (secret, rest) = scratch.words.split_at_mut(n);
    let (error, product) = rest.split_at_mut(n);
    ar::convert(s, secret, params.q);
    ar::convert(e, error, params.q);
    convolution.product(a, secret, product, params.q)?;
    let output = WipeGuard::new(pk);
    for i in 0..n {
        output.words[i] = ar::sub(error[i], product[i], params.q);
        output.words[n + i] = a[i];
    }
    output.finish();
    Ok(())
}

/// ct=(pk0*u+e0+Delta*m, pk1*u+e1), Delta=floor(q/t).
pub(super) fn encrypt_into(
    params: Params,
    pk: &[u64],
    message: &[u64],
    u: &[i64],
    e0: &[i64],
    e1: &[i64],
    ct: &mut [u64],
    scratch: &mut [u64],
    work: &mut KernelWorkLimit,
) -> Result<(), KernelError> {
    let n = params.n;
    preflight(
        params,
        Operation::Encrypt,
        &[
            (pk.len(), params.two_n),
            (message.len(), n),
            (u.len(), n),
            (e0.len(), n),
            (e1.len(), n),
            (ct.len(), params.two_n),
            (scratch.len(), params.three_n),
        ],
        work,
    )?;
    encrypt_body(params, pk, message, u, e0, e1, ct, scratch, Convolution::Schoolbook)
}

pub(super) fn encrypt_ntt_into(
    params: Params,
    pk: &[u64],
    message: &[u64],
    u: &[i64],
    e0: &[i64],
    e1: &[i64],
    ct: &mut [u64],
    scratch: &mut [u64],
    work: &mut KernelWorkLimit,
    plan: &NegacyclicNtt,
    workspace: &mut [u64],
    ntt_work: &mut KernelWorkLimit,
) -> Result<(), KernelError> {
    let n = params.n;
    preflight_ntt(
        params,
        Operation::Encrypt,
        &[
            (pk.len(), params.two_n),
            (message.len(), n),
            (u.len(), n),
            (e0.len(), n),
            (e1.len(), n),
            (ct.len(), params.two_n),
            (scratch.len(), params.three_n),
        ],
        work,
        plan,
        workspace.len(),
        ntt_work,
    )?;
    let workspace = WipeGuard::new(workspace);
    encrypt_body(
        params,
        pk,
        message,
        u,
        e0,
        e1,
        ct,
        scratch,
        Convolution::FixedNtt { plan, scratch: &mut *workspace.words, work: ntt_work },
    )
}

fn encrypt_body(
    params: Params,
    pk: &[u64],
    message: &[u64],
    u: &[i64],
    e0: &[i64],
    e1: &[i64],
    ct: &mut [u64],
    scratch: &mut [u64],
    mut convolution: Convolution<'_>,
) -> Result<(), KernelError> {
    let n = params.n;
    let scratch = WipeGuard::new(scratch);
    let mut invalid = ar::invalid_canonical(pk, params.q);
    invalid |= ar::invalid_canonical(message, params.t);
    invalid |= ar::invalid_signed(u, params.supports.u);
    invalid |= ar::invalid_signed(e0, params.supports.e0);
    invalid |= ar::invalid_signed(e1, params.supports.e1);
    if invalid != 0 {
        return Err(KernelError::InvalidValue);
    }
    let (ephemeral, rest) = scratch.words.split_at_mut(n);
    let (first, second) = rest.split_at_mut(n);
    ar::convert(u, ephemeral, params.q);
    convolution.product(&pk[..n], ephemeral, first, params.q)?;
    convolution.product(&pk[n..], ephemeral, second, params.q)?;
    let output = WipeGuard::new(ct);
    for i in 0..n {
        let embedded = ar::mul(params.delta, message[i], params.q);
        let noise0 = ar::signed(e0[i], params.q);
        let noise1 = ar::signed(e1[i], params.q);
        output.words[i] = ar::add(ar::add(first[i], noise0, params.q), embedded, params.q);
        output.words[n + i] = ar::add(second[i], noise1, params.q);
    }
    output.finish();
    Ok(())
}

/// Componentwise addition; no raw key-identity or addition-depth enforcement.
pub(super) fn add_into(
    params: Params,
    lhs: &[u64],
    rhs: &[u64],
    output: &mut [u64],
    work: &mut KernelWorkLimit,
) -> Result<(), KernelError> {
    preflight(
        params,
        Operation::Add,
        &[
            (lhs.len(), params.two_n),
            (rhs.len(), params.two_n),
            (output.len(), params.two_n),
        ],
        work,
    )?;
    let mut invalid = ar::invalid_canonical(lhs, params.q);
    invalid |= ar::invalid_canonical(rhs, params.q);
    if invalid != 0 {
        return Err(KernelError::InvalidValue);
    }
    let output = WipeGuard::new(output);
    for i in 0..params.two_n {
        output.words[i] = ar::add(lhs[i], rhs[i], params.q);
    }
    output.finish();
    Ok(())
}

/// phase=(c0+c1*s) mod q, then exact half-up decode mod t. Canonical arbitrary
/// ciphertexts receive this equation, NOT a valid-encryption/noise guarantee.
pub(super) fn decrypt_into(
    params: Params,
    s: &[i64],
    ct: &[u64],
    message: &mut [u64],
    scratch: &mut [u64],
    work: &mut KernelWorkLimit,
) -> Result<(), KernelError> {
    let n = params.n;
    preflight(
        params,
        Operation::Decrypt,
        &[
            (s.len(), n),
            (ct.len(), params.two_n),
            (message.len(), n),
            (scratch.len(), params.three_n),
        ],
        work,
    )?;
    decrypt_body(params, s, ct, message, scratch, Convolution::Schoolbook)
}

pub(super) fn decrypt_ntt_into(
    params: Params,
    s: &[i64],
    ct: &[u64],
    message: &mut [u64],
    scratch: &mut [u64],
    work: &mut KernelWorkLimit,
    plan: &NegacyclicNtt,
    workspace: &mut [u64],
    ntt_work: &mut KernelWorkLimit,
) -> Result<(), KernelError> {
    let n = params.n;
    preflight_ntt(
        params,
        Operation::Decrypt,
        &[
            (s.len(), n),
            (ct.len(), params.two_n),
            (message.len(), n),
            (scratch.len(), params.three_n),
        ],
        work,
        plan,
        workspace.len(),
        ntt_work,
    )?;
    let workspace = WipeGuard::new(workspace);
    decrypt_body(
        params,
        s,
        ct,
        message,
        scratch,
        Convolution::FixedNtt { plan, scratch: &mut *workspace.words, work: ntt_work },
    )
}

fn decrypt_body(
    params: Params,
    s: &[i64],
    ct: &[u64],
    message: &mut [u64],
    scratch: &mut [u64],
    mut convolution: Convolution<'_>,
) -> Result<(), KernelError> {
    let n = params.n;
    let scratch = WipeGuard::new(scratch);
    let mut invalid = ar::invalid_signed(s, params.supports.s);
    invalid |= ar::invalid_canonical(ct, params.q);
    if invalid != 0 {
        return Err(KernelError::InvalidValue);
    }
    let (secret, rest) = scratch.words.split_at_mut(n);
    let (product, phase) = rest.split_at_mut(n);
    ar::convert(s, secret, params.q);
    convolution.product(&ct[n..], secret, product, params.q)?;
    for i in 0..n {
        phase[i] = ar::add(ct[i], product[i], params.q);
    }
    let output = WipeGuard::new(message);
    for i in 0..n {
        output.words[i] = ar::decode(phase[i], params.q, params.t);
    }
    output.finish();
    Ok(())
}
