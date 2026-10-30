//! Diagnostic exact-ring reference, not encryption or an admitted capability.
//!
//! NEXUS Appendix B, printed p.16, PDF SHA-256
//! c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2:
//! b = A*s_S + m + e; add_client_layer computes b + A*s_C + e';
//! remove_server_layer then computes b' - A*s_S. The residual phase is
//! m + e + e', NOT m. Factors and signed noise are explicit caller fixtures;
//! there are no keys, sampling, ciphertext descriptors, encoding or security claims.
//! Section III.C, p.6 separately gives (A-U)W + UW = AW. Matrix A there
//! is unrelated to Appendix B's polynomial A. No CKKS slot mapping is implied.
//!
//! Shared CoefficientRing owns modulus/degree and signed negacyclic reduction.
//! Multiplication streams its monomial action; no parallel ring implementation.
//! CoefficientCompression remains the owner of packing/decompression (see tests).
//! All buffers are caller-owned and already admitted by their original owner.
//! Quotes are conservative coefficient-visit envelopes, not native instructions,
//! latency or complete resource admission. The caller's SAME cumulative ledger
//! is debited before canonicality scans. Quote/shape refusals occur before debit;
//! a failed debit leaves its balance unchanged. Once admitted, canonicality-scan
//! work is nonrefundable even on rejection. Every refusal leaves buffers unchanged.

use crate::coefficient_layout::{CoefficientRing, LayoutError};
use crate::row_fold::{MAX_INVOCATION_WORK, WorkLedger};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Layout(LayoutError),
    Work(crate::row_fold::Error),
}
impl From<LayoutError> for Error {
    fn from(error: LayoutError) -> Self {
        Self::Layout(error)
    }
}
impl From<crate::row_fold::Error> for Error {
    fn from(error: crate::row_fold::Error) -> Self {
        Self::Work(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
    AddClientLayer,
    RemoveServerLayer,
    /// Row-major scalar matrices (rows, inner, columns), not SIMD encoding.
    ComposeProjection([usize; 3]),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Quote {
    pub input_words: usize,
    pub output_words: usize,
    pub scratch_words: usize,
    /// All input, output and scratch payload bytes, excluding external owners.
    pub caller_bytes: usize,
    pub work: usize,
}
fn add(a: usize, b: usize) -> Result<usize, LayoutError> {
    a.checked_add(b).ok_or(LayoutError::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize, LayoutError> {
    a.checked_mul(b).ok_or(LayoutError::Overflow)
}

/// Literal limits: no budget enlargement or quote capping to make work fit.
pub fn quote(
    ring: CoefficientRing,
    op: Operation,
    max_bytes: usize,
    max_work: usize,
) -> Result<Quote, Error> {
    let n = ring.degree();
    if n > crate::coefficient_layout::MAX_DEGREE {
        return Err(LayoutError::InvalidDegree.into());
    }
    let (input_words, output_words, scratch_words, work) = match op {
        Operation::AddClientLayer | Operation::RemoveServerLayer => {
            let inputs = if op == Operation::AddClientLayer {
                4
            } else {
                3
            };
            // Includes repeated canonical scans inside n shared monomial calls,
            // n*n scalar accumulations, initialization and external validation.
            (
                mul(inputs, n)?,
                n,
                n,
                add(mul(16, mul(n, n)?)?, mul(16, n)?)?,
            )
        }
        Operation::ComposeProjection([rows, inner, cols]) => {
            if rows == 0 || inner == 0 || cols == 0 {
                return Err(LayoutError::InvalidGeometry.into());
            }
            let left = mul(rows, inner)?;
            let right = mul(inner, cols)?;
            let out = mul(rows, cols)?;
            let inputs = add(mul(2, left)?, right)?;
            // Two ordered dots (A-U)W and UW, validation, final addition.
            (
                inputs,
                out,
                0,
                add(
                    mul(16, mul(out, inner)?)?,
                    add(mul(4, inputs)?, mul(4, out)?)?,
                )?,
            )
        }
    };
    let caller_bytes = mul(add(add(input_words, output_words)?, scratch_words)?, 8)?;
    if caller_bytes > max_bytes
        || caller_bytes > isize::MAX as usize
        || work > max_work
        || work > MAX_INVOCATION_WORK
    {
        return Err(LayoutError::ResourceLimit.into());
    }
    Ok(Quote {
        input_words,
        output_words,
        scratch_words,
        caller_bytes,
        work,
    })
}

fn canonical(ring: CoefficientRing, inputs: &[&[u64]]) -> Result<(), Error> {
    if inputs
        .iter()
        .any(|xs| xs.iter().any(|&x| x >= ring.modulus()))
    {
        return Err(LayoutError::NonCanonical.into());
    }
    Ok(())
}
fn product_add(
    ring: CoefficientRing,
    a: &[u64],
    factor: &[u64],
    subtract: bool,
    output: &mut [u64],
    scratch: &mut [u64],
) {
    let q = ring.modulus();
    for (i, &scalar) in factor.iter().enumerate() {
        ring.monomial(a, i as i64, scratch)
            .expect("prevalidated ring buffers");
        for (dst, &shifted) in output.iter_mut().zip(scratch.iter()) {
            let term = (u128::from(scalar) * u128::from(shifted) % u128::from(q)) as u64;
            *dst = if subtract {
                (*dst + q - term) % q
            } else {
                (*dst + term) % q
            };
        }
    }
}

/// Appendix B step 2, preserving the caller's existing b (including e).
/// Signed e' coefficients are reduced modulo q; no distribution is asserted.
#[allow(clippy::too_many_arguments)]
pub fn add_client_layer(
    ring: CoefficientRing,
    a: &[u64],
    b: &[u64],
    client_factor: &[u64],
    noise_prime: &[i64],
    output: &mut [u64],
    scratch: &mut [u64],
    max_bytes: usize,
    max_work: usize,
    ledger: &mut WorkLedger,
) -> Result<(), Error> {
    let price = quote(ring, Operation::AddClientLayer, max_bytes, max_work)?;
    let n = ring.degree();
    if [
        a.len(),
        b.len(),
        client_factor.len(),
        noise_prime.len(),
        output.len(),
        scratch.len(),
    ]
    .iter()
    .any(|&len| len != n)
    {
        return Err(LayoutError::BufferLength.into());
    }
    ledger.debit(price.work, 0)?;
    canonical(ring, &[a, b, client_factor])?;
    output.copy_from_slice(b);
    product_add(ring, a, client_factor, false, output, scratch);
    for (dst, &noise) in output.iter_mut().zip(noise_prime) {
        *dst = (*dst + noise.rem_euclid(ring.modulus() as i64) as u64) % ring.modulus();
    }
    Ok(())
}

/// Appendix B removal equation. Does not decrypt, decode, or discard e/e'.
#[allow(clippy::too_many_arguments)]
pub fn remove_server_layer(
    ring: CoefficientRing,
    a: &[u64],
    layered: &[u64],
    server_factor: &[u64],
    output: &mut [u64],
    scratch: &mut [u64],
    max_bytes: usize,
    max_work: usize,
    ledger: &mut WorkLedger,
) -> Result<(), Error> {
    let price = quote(ring, Operation::RemoveServerLayer, max_bytes, max_work)?;
    let n = ring.degree();
    if [
        a.len(),
        layered.len(),
        server_factor.len(),
        output.len(),
        scratch.len(),
    ]
    .iter()
    .any(|&len| len != n)
    {
        return Err(LayoutError::BufferLength.into());
    }
    ledger.debit(price.work, 0)?;
    canonical(ring, &[a, layered, server_factor])?;
    output.copy_from_slice(layered);
    product_add(ring, a, server_factor, true, output, scratch);
    Ok(())
}

/// Section III.C's exact linear identity ONLY. U is a supplied matrix, not a
/// sampled/approved one-time pad. No noisy phase is silently identified with AW.
#[allow(clippy::too_many_arguments)]
pub fn compose_projection(
    ring: CoefficientRing,
    shape: [usize; 3],
    matrix: &[u64],
    u: &[u64],
    weights: &[u64],
    output: &mut [u64],
    max_bytes: usize,
    max_work: usize,
    ledger: &mut WorkLedger,
) -> Result<(), Error> {
    let price = quote(
        ring,
        Operation::ComposeProjection(shape),
        max_bytes,
        max_work,
    )?;
    let [rows, inner, cols] = shape;
    if matrix.len() != rows * inner
        || u.len() != matrix.len()
        || weights.len() != inner * cols
        || output.len() != price.output_words
    {
        return Err(LayoutError::BufferLength.into());
    }
    ledger.debit(price.work, 0)?;
    canonical(ring, &[matrix, u, weights])?;
    let q = u128::from(ring.modulus());
    for row in 0..rows {
        for col in 0..cols {
            let (mut online, mut offline) = (0, 0);
            for h in 0..inner {
                let a = u128::from(matrix[row * inner + h]);
                let mask = u128::from(u[row * inner + h]);
                let w = u128::from(weights[h * cols + col]);
                online = (online + ((a + q - mask) % q) * w) % q;
                offline = (offline + mask * w) % q;
            }
            output[row * cols + col] = ((online + offline) % q) as u64;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "commutable_projection_reference_tests.rs"]
mod tests;
