//! SYNTHETIC Torus64 encrypted CMUX arithmetic; not source-exact ROSETTA.
//!
//! Rank-one GLWE over R = (Z/2^64 Z)[X]/(X^N+1), component-major (B,A),
//! phase B + A*s. All coefficients use full u64 wrapping arithmetic. This is
//! NOT the B-A*s output convention of `coefficient_torus_conversion`: negate
//! that output's mask polynomial before adapting it to this convention.
//!
//! Selector is an encrypted GGSW with 16 GLWE rows, eight per input component.
//! Flattened index: (((component * 8 + digit) * 2 + row_component) * N + k).
//! Components and row_components are B=0, A=1; digit 0 is least significant.
//! Let g_d=2^(8*d). Row (c,d) is an encryption of zero Z_(c,d), plus a
//! selector*g_d constant-polynomial entry on component c's diagonal. Thus
//! row phase is e_(0,d)+selector*g_d for c=0 and
//! e_(1,d)+selector*g_d*s for c=1, with PLUS diagonal and PLUS phase signs.
//! Decomposing both components into unsigned base-256 digits reconstructs
//! every bit exactly: x = sum_d digit_d(x)*g_d mod 2^64. Consequently
//! phase(ExtProduct(delta,S)) = selector*phase(delta) + sum D_(c,d)*e_(c,d).
//! CMUX computes f + ExtProduct(t-f,S); it does not inspect a clear selector.
//!
//! Encryption validity, selector bitness, key agreement and noise bounds are
//! semantic preconditions. This kernel checks none of them. It accepts no key,
//! secret, clear selector, decryptor, epoch label or callback. No heap allocation,
//! factory, registration, privilege, runtime activation or constant-time claim.
//! Degree-eight tests are deterministic SYNTHETIC fixtures, NOT security
//! parameters, cryptographic RNG, blind rotation or a complete HE implementation.
//!
//! Quotes/limits are primitive-local only, NOT original cumulative admission.
//! Missing trusted original encrypted-material/key/backing/attempt owner remains
//! an activation blocker. PublicCiphertextOwner and TraceAdmission are metadata;
//! OriginalSourcePlan owns model source, not GGSW material. No caller labels can
//! repair that gap. Existing external implementations are not dependencies.

pub use crate::coefficient_layout::LayoutError as Error;

pub const DIGITS: usize = 8;
pub const COMPONENTS: usize = 2;
/// Local quadratic-kernel size cap only; no inherited numeric/security profile.
pub const MAX_DEGREE: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    pub rank: usize,
    pub degree: usize,
}

/// Exact counts of the named logical events, NOT scalar instruction/CPU fuel.
/// One MAC event multiplies a digit by a row word, then adds OR subtracts it.
/// Indexing, validation, quote construction and loop controls are not events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Work {
    pub difference_words: usize,
    pub digit_words: usize,
    pub output_copy_words: usize,
    pub negacyclic_macs: usize,
    pub total_events: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Quote {
    /// Each of f, t and output has this exact length.
    pub glwe_words: usize,
    pub ggsw_words: usize,
    /// Delta (2N words) and one reusable digit polynomial (N words).
    pub scratch_words: usize,
    /// Sum of five slice extents (f,t,S,output,scratch), in u64 bytes.
    /// Assumes distinct input extents; immutable f/t/S MAY alias, in which case
    /// this overcounts. Excludes backing outside slices, descriptors and stack.
    /// This is NOT retained-owner, allocator, transport or process memory price.
    pub caller_bytes: usize,
    pub work: Work,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    pub max_work_events: usize,
    pub max_caller_bytes: usize,
}

fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}

/// Checked geometry, slice products, storage and exact named-event quotation.
/// Arithmetic is checked even for hostile geometry before any buffer access.
pub fn quote(geometry: Geometry) -> Result<Quote, Error> {
    let n = geometry.degree;
    let components = add(geometry.rank, 1)?;
    let glwe_words = mul(components, n)?;
    let rows = mul(components, DIGITS)?;
    let ggsw_words = mul(rows, glwe_words)?;
    let scratch_words = add(glwe_words, n)?;
    let caller_words = add(add(mul(glwe_words, 3)?, ggsw_words)?, scratch_words)?;
    let caller_bytes = mul(caller_words, std::mem::size_of::<u64>())?;
    let digit_words = mul(rows, n)?;
    let negacyclic_macs = mul(digit_words, glwe_words)?;
    let total_events = add(add(mul(glwe_words, 2)?, digit_words)?, negacyclic_macs)?;
    // Also prove the maximum convolution index i+j fits before kernel writes.
    mul(n, 2)?;
    if geometry.rank != 1 {
        return Err(Error::InvalidGeometry);
    }
    if !n.is_power_of_two() || n > MAX_DEGREE {
        return Err(Error::InvalidDegree);
    }
    Ok(Quote {
        glwe_words,
        ggsw_words,
        scratch_words,
        caller_bytes,
        work: Work {
            difference_words: glwe_words,
            digit_words,
            output_copy_words: glwe_words,
            negacyclic_macs,
            total_events,
        },
    })
}

/// Write f + ExtProduct(t-f, encrypted_selector), without allocation.
/// All slices must have EXACT quoted lengths (including scratch). All refusals
/// precede any output OR scratch mutation. On success scratch is overwritten;
/// no zeroization or retention guarantee is implied. Safe Rust mutable borrows
/// require output/scratch disjoint from each other and every input slice.
/// Caps are repeatable local limits, not a spendable/cumulative owner ledger.
pub fn cmux_into(
    geometry: Geometry,
    f: &[u64],
    t: &[u64],
    encrypted_selector: &[u64],
    output: &mut [u64],
    scratch: &mut [u64],
    limits: Limits,
) -> Result<(), Error> {
    let q = quote(geometry)?;
    if f.len() != q.glwe_words
        || t.len() != q.glwe_words
        || encrypted_selector.len() != q.ggsw_words
        || output.len() != q.glwe_words
        || scratch.len() != q.scratch_words
    {
        return Err(Error::BufferLength);
    }
    if limits.max_work_events < q.work.total_events || limits.max_caller_bytes < q.caller_bytes {
        return Err(Error::ResourceLimit);
    }
    // No fallible operations remain. All u64 coefficients are canonical torus
    // words; no odd-modulus canonicality scan or rounding profile is applicable.
    let n = geometry.degree;
    let (delta, digits) = scratch.split_at_mut(q.glwe_words);
    for k in 0..q.glwe_words {
        delta[k] = t[k].wrapping_sub(f[k]);
    }
    output.copy_from_slice(f);
    for component in 0..COMPONENTS {
        for digit in 0..DIGITS {
            for i in 0..n {
                digits[i] = (delta[component * n + i] >> (8 * digit)) & 255;
            }
            let row = (component * DIGITS + digit) * COMPONENTS * n;
            for out_component in 0..COMPONENTS {
                for i in 0..n {
                    for j in 0..n {
                        let product = digits[i]
                            .wrapping_mul(encrypted_selector[row + out_component * n + j]);
                        let k = i + j;
                        let at = out_component * n + k % n;
                        output[at] = if k < n {
                            output[at].wrapping_add(product)
                        } else {
                            output[at].wrapping_sub(product)
                        };
                    }
                }
            }
        }
    }
    Ok(())
}
