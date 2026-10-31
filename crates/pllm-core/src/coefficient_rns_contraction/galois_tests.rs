//! Source-only authored tests: ALL UNRUN. Public synthetic rows, not HE keys.
//! Independent coefficient sigma, unsigned digits, signed schoolbook, BigInt
//! pairwise qi*p CRT and nonnegative floor division; inputs use direct DFT,
//! never production transforms. Historical tranche has no full-Qp CRT oracle:
//! each retained output depends only on its qi,p pair, giving equivalent rounded
//! residues. Large cases use sparse supports, not dense quadratic transforms.
//! Work assertions are QUOTE-CONTRACT REGRESSIONS and exact-fit/one-short
//! admission checks, not independently proven operation-event envelopes.
//! NEW relation_ tranche below adds independent signed BigInt arithmetic and
//! full four-limb Qp CRT for three synthetic source-row relation regressions.
//! Neither tranche establishes plaintext correctness, security, or source ownership.
use super::*;
use crate::coefficient_layout::{CoefficientRing, DegreeBound};
use num_bigint::BigInt;

fn residue(x: BigInt, q: u64) -> u64 {
    let q = BigInt::from(q);
    let x = ((x % &q) + &q) % &q;
    x.to_u64_digits().1.first().copied().unwrap_or(0)
}

fn power(a: u64, exponent: u64, q: u64) -> u64 {
    residue(BigInt::from(a).modpow(&BigInt::from(exponent), &BigInt::from(q)), q)
}

fn dft(values: &[u64], root: u64, q: u64, inverse: bool) -> Vec<u64> {
    let n = values.len();
    let support: Vec<_> = values.iter().copied().enumerate().filter(|&(_, v)| v != 0).collect();
    let root = if inverse { power(root, q - 2, q) } else { root };
    let scale = if inverse { power(n as u64, q - 2, q) } else { 1 };
    (0..n).map(|at| {
        let mut sum = BigInt::from(0);
        for &(from, value) in &support {
            let exponent = if inverse { (2 * from + 1) * at } else { (2 * at + 1) * from };
            sum += BigInt::from(value) * BigInt::from(power(root, exponent as u64, q));
        }
        residue(sum * BigInt::from(scale), q)
    }).collect()
}

fn sigma(values: &[u64], g: u64, q: u64) -> Vec<u64> {
    let n = values.len();
    let mut out = vec![BigInt::from(0); n];
    // Independent signed monomials in Z[X]/(X^N+1), not spectral permutation.
    for (j, &value) in values.iter().enumerate() {
        let exponent = j as u64 * g;
        let index = (exponent % n as u64) as usize;
        if (exponent / n as u64) % 2 == 0 {
            out[index] += BigInt::from(value);
        } else {
            out[index] -= BigInt::from(value);
        }
    }
    out.into_iter().map(|x| residue(x, q)).collect()
}

fn schoolbook(a: &[BigInt], b: &[u64]) -> Vec<BigInt> {
    let n = a.len();
    let left: Vec<_> = a.iter().enumerate().filter(|(_, v)| **v != BigInt::from(0)).collect();
    let right: Vec<_> = b.iter().copied().enumerate().filter(|&(_, v)| v != 0).collect();
    let mut out = vec![BigInt::from(0); n];
    for &(i, a) in &left {
        for &(j, b) in &right {
            let term = a * BigInt::from(b);
            if i + j >= n { out[i + j - n] -= term; } else { out[i + j] += term; }
        }
    }
    out
}

fn rounded(q_value: u64, p_value: u64, q: u64, p: u64) -> u64 {
    let inverse = BigInt::from(q).modinv(&BigInt::from(p)).unwrap();
    let t = residue((BigInt::from(p_value) - BigInt::from(q_value)) * inverse, p);
    let x = BigInt::from(q_value) + BigInt::from(q) * BigInt::from(t);
    assert!(x >= BigInt::from(0));
    residue((x + BigInt::from((p - 1) / 2)) / BigInt::from(p), q)
}

struct Fixture {
    n: usize,
    q: [u64; 4],
    roots: [u64; 4],
    sources: [NegacyclicNtt; 3],
    targets: [NegacyclicNtt; 4],
}

impl Fixture {
    fn new(n: usize, q: [u64; 4]) -> Self {
        let roots = q.map(|q| (2..=64).find_map(|a| {
            let root = power(a, (q - 1) / (2 * n as u64), q);
            (power(root, n as u64, q) == q - 1).then_some(root)
        }).expect("bounded independent root witness required"));
        let targets = core::array::from_fn(|i| NegacyclicNtt::new(
            CoefficientRing::bounded(n, q[i], DegreeBound::Public8192).unwrap(), roots[i],
        ).unwrap());
        let sources = core::array::from_fn(|i| targets[i].clone());
        Self { n, q, roots, sources, targets }
    }

    fn plan(&self) -> RnsContraction<'_> { RnsContraction::new(&self.sources, &self.targets).unwrap() }

    fn inputs(&self, coefficients: &[u64], rows: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let source = (0..6).flat_map(|poly| {
            let i = poly % 3;
            dft(&coefficients[poly * self.n..(poly + 1) * self.n], self.roots[i], self.q[i], false)
        }).collect();
        let rows = (0..24).flat_map(|poly| {
            let i = poly % 4;
            dft(&rows[poly * self.n..(poly + 1) * self.n], self.roots[i], self.q[i], false)
        }).collect();
        (source, rows)
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum Variant {
    Correct, Inverse, Centered, OrdinateCopy, ComponentSwap, BeforeDrop,
    EachRowDrop, LateSigma, RowPayloadPermutation, OmitC0, UntransformedC0,
}

fn oracle(f: &Fixture, coefficients: &[u64], rows: &[u64], g: u64, variant: Variant) -> Vec<u64> {
    let n = f.n;
    let g = if variant == Variant::Inverse {
        (1..2 * n as u64).step_by(2).find(|&h| h * g % (2 * n as u64) == 1).unwrap()
    } else { g };
    let transformed: Vec<_> = (0..6).map(|poly| {
        // Wrong LateSigma leaves c1 untransformed until AFTER contraction/drop.
        let input_g = if variant == Variant::LateSigma && poly >= 3 { 1 } else { g };
        sigma(&coefficients[poly * n..(poly + 1) * n], input_g, f.q[poly % 3])
    }).collect();
    let mut contracted = vec![0; 8 * n];
    let mut each_row = vec![0; 24 * n];
    for component in 0..2 {
        for limb in 0..4 {
            let mut sums = vec![BigInt::from(0); n];
            for row in 0..3 {
                let digits = &transformed[3 + row];
                let lifted: Vec<BigInt> = if variant == Variant::OrdinateCopy {
                    let spectral = dft(digits, f.roots[row], f.q[row], false);
                    let copied: Vec<_> = spectral.into_iter().map(|x| x % f.q[limb]).collect();
                    dft(&copied, f.roots[limb], f.q[limb], true).into_iter().map(BigInt::from).collect()
                } else {
                    digits.iter().map(|&x| {
                        if variant == Variant::Centered && x > f.q[row] / 2 {
                            BigInt::from(x) - BigInt::from(f.q[row])
                        } else { BigInt::from(x) }
                    }).collect()
                };
                let row_component = if variant == Variant::ComponentSwap { 1 - component } else { component };
                // Swap whole row0/row1 DATA payloads, retaining all descriptor,
                // component and target-limb positions. No context permutation.
                let payload_row = if variant == Variant::RowPayloadPermutation && row < 2 { 1-row } else { row };
                let start = ((payload_row * 2 + row_component) * 4 + limb) * n;
                for (at, (sum, term)) in sums.iter_mut().zip(schoolbook(&lifted, &rows[start..start + n])).enumerate() {
                    each_row[((row * 2 + component) * 4 + limb) * n + at] = residue(term.clone(), f.q[limb]);
                    *sum += term;
                }
            }
            for (at, sum) in sums.into_iter().enumerate() {
                contracted[(component * 4 + limb) * n + at] = residue(sum, f.q[limb]);
            }
        }
    }
    if variant == Variant::BeforeDrop {
        // Explicit wrong lift: retained c0 limbs and unsigned q0->p c0 lift
        // injected before Dp, rather than adding sigma(c0) after Dp.
        for limb in 0..4 {
            for at in 0..n {
                let c0 = transformed[if limb == 3 { 0 } else { limb }][at];
                contracted[limb * n + at] = residue(
                    BigInt::from(contracted[limb * n + at]) + BigInt::from(c0), f.q[limb]);
            }
        }
    }
    let mut out = vec![0; 6 * n];
    for component in 0..2 {
        for limb in 0..3 {
            let dropped: Vec<_> = (0..n).map(|at| {
                if variant == Variant::EachRowDrop {
                    // Explicit wrong semantics: Dp(v0)+Dp(v1)+Dp(v2), not
                    // Dp(v0+v1+v2). Round each qi,p pair before summation.
                    let sum: BigInt = (0..3).map(|row| BigInt::from(rounded(
                        each_row[((row * 2 + component) * 4 + limb) * n + at],
                        each_row[((row * 2 + component) * 4 + 3) * n + at], f.q[limb], f.q[3],
                    ))).sum();
                    residue(sum, f.q[limb])
                } else {
                    rounded(contracted[(component * 4 + limb) * n + at],
                        contracted[(component * 4 + 3) * n + at], f.q[limb], f.q[3])
                }
            }).collect();
            // Wrong LateSigma transforms the dropped row product, including
            // its row polynomial, instead of transforming only input c1.
            let dropped = if variant == Variant::LateSigma { sigma(&dropped, g, f.q[limb]) } else { dropped };
            for at in 0..n {
                let c0 = if component != 0 || variant == Variant::BeforeDrop || variant == Variant::OmitC0 { 0 }
                    else if variant == Variant::UntransformedC0 { coefficients[limb * n + at] }
                    else { transformed[limb][at] };
                out[(component * 3 + limb) * n + at] = residue(BigInt::from(dropped[at]) + BigInt::from(c0), f.q[limb]);
            }
        }
    }
    out
}

fn sentinels(n: usize) -> Vec<u64> { (0..n).map(|i| u64::MAX - 19 * i as u64).collect() }

fn check(f: &Fixture, coefficients: &[u64], rows: &[u64], g: u64) -> Vec<u64> {
    let plan = f.plan();
    let storage = plan.galois_storage().unwrap();
    assert_eq!((storage.source_words, storage.row_words, storage.output_words), (6*f.n, 24*f.n, 6*f.n));
    assert_eq!((storage.scratch_words, storage.total_words, storage.total_bytes), (29*f.n, 65*f.n, 520*f.n));
    let old = plan.composition_storage().unwrap();
    assert_eq!((old.scratch_words, old.total_words, old.total_bytes), (19*f.n, 52*f.n, 416*f.n));
    let (source, rows_spectral) = f.inputs(coefficients, rows);
    let source_before = source.clone();
    let rows_before = rows_spectral.clone();
    let mut output_backing = sentinels(6*f.n + 7);
    let mut scratch = sentinels(29*f.n + 11);
    let output_tail = output_backing[6*f.n..].to_vec();
    let scratch_tail = scratch[29*f.n..].to_vec();
    plan.galois_contract_drop_into(
        ComponentSpectra { contexts: &f.sources, values: &source },
        RowSpectra { contexts: &f.targets, values: &rows_spectral }, g,
        &mut output_backing[..6*f.n], &mut scratch, plan.galois_work().unwrap(),
    ).unwrap();
    let expected = oracle(f, coefficients, rows, g, Variant::Correct);
    assert_eq!(&output_backing[..6*f.n], expected);
    assert_eq!(&output_backing[6*f.n..], output_tail);
    assert_eq!(&scratch[29*f.n..], scratch_tail);
    assert_eq!(source, source_before);
    assert_eq!(rows_spectral, rows_before);
    output_backing[..6*f.n].to_vec()
}

fn dense(f: &Fixture) -> (Vec<u64>, Vec<u64>) {
    let coefficients = (0..6*f.n).map(|at| {
        let q = f.q[(at / f.n) % 3];
        if at % 3 == 0 { q - 1 } else { (at as u64 * 7 + 1) % q }
    }).collect();
    let rows = (0..24*f.n).map(|at| {
        let q = f.q[(at / f.n) % 4];
        if at % 2 == 0 { q - 1 } else { (at as u64 * 11 + 3) % q }
    }).collect();
    (coefficients, rows)
}

#[test]
fn tiny_all_odd_g_identity_signed_wrap_both_components_p_above_and_below() {
    for n in [1, 2, 4, 8] {
        for q in [[17, 97, 113, 193], [97, 113, 193, 17]] {
            let f = Fixture::new(n, q);
            let (coefficients, rows) = dense(&f);
            for g in (1..2*n as u64).step_by(2) { check(&f, &coefficients, &rows, g); }
        }
    }
}

#[test]
fn zero_and_unsigned_extremes() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let (mut coefficients, rows) = dense(&f);
    coefficients.fill(0);
    assert_eq!(check(&f, &coefficients, &rows, 7), vec![0; 24]);
    for (at, value) in coefficients.iter_mut().enumerate() { *value = f.q[(at / f.n) % 3] - 1; }
    check(&f, &coefficients, &rows, 3);
    check(&f, &coefficients, &vec![0; 24*f.n], 5);
}

#[test]
fn distinct_valid_roots_have_same_coefficient_result() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let (coefficients, rows) = dense(&f);
    let expected = check(&f, &coefficients, &rows, 3);
    let roots = core::array::from_fn(|i| power(f.roots[i], 3, f.q[i]));
    let targets = core::array::from_fn(|i| NegacyclicNtt::new(f.targets[i].ring(), roots[i]).unwrap());
    let sources = core::array::from_fn(|i| targets[i].clone());
    let other = Fixture { n: f.n, q: f.q, roots, sources, targets };
    assert_ne!(f.targets, other.targets);
    assert_eq!(check(&other, &coefficients, &rows, 3), expected);
}

#[test]
fn independent_counterexamples_reject_five_wrong_algorithms() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let (coefficients, rows) = dense(&f);
    // g=3 at N=4 is self-inverse; use N=8, g=3, inverse=11 instead.
    let f8 = Fixture::new(8, [17, 97, 113, 193]);
    let (coefficients8, rows8) = dense(&f8);
    let correct8 = oracle(&f8, &coefficients8, &rows8, 3, Variant::Correct);
    assert_ne!(correct8, oracle(&f8, &coefficients8, &rows8, 3, Variant::Inverse));
    assert_eq!(check(&f8, &coefficients8, &rows8, 3), correct8);
    let correct = oracle(&f, &coefficients, &rows, 3, Variant::Correct);
    for wrong in [Variant::Centered, Variant::OrdinateCopy, Variant::ComponentSwap, Variant::BeforeDrop] {
        assert_ne!(correct, oracle(&f, &coefficients, &rows, 3, wrong), "{wrong:?}");
    }
    assert_eq!(check(&f, &coefficients, &rows, 3), correct);
}

#[test]
fn rowwise_drop_loses_derived_rounding_carry() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let n = f.n;
    let mut coefficients = vec![0; 6*n];
    for limb in 0..3 { coefficients[limb*n+1] = limb as u64 + 2; }
    coefficients[3*n] = 1;
    coefficients[4*n] = 1;
    let mut rows = vec![0; 24*n];
    // p=193, h=96: round(96/193)=round(95/193)=0,
    // but round(192/193)=round(190/193)=1. Two rows expose the carry.
    for row in 0..2 {
        for limb in 0..4 {
            rows[((row*2)*4+limb)*n] = residue(BigInt::from(96), f.q[limb]);
            rows[((row*2+1)*4+limb)*n] = residue(BigInt::from(95), f.q[limb]);
        }
    }
    let correct = oracle(&f, &coefficients, &rows, 3, Variant::Correct);
    let wrong = oracle(&f, &coefficients, &rows, 3, Variant::EachRowDrop);
    assert_ne!(correct, wrong);
    for poly in 0..6 { assert_eq!((correct[poly*n], wrong[poly*n]), (1, 0)); }
    assert_eq!(check(&f, &coefficients, &rows, 3), correct);
}

#[test]
fn sigma_after_drop_wrongly_transforms_row_polynomial() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let n = f.n;
    let mut coefficients = vec![0; 6*n];
    for limb in 0..3 { coefficients[limb*n] = limb as u64 + 2; }
    coefficients[3*n+1] = 1;
    let mut rows = vec![0; 24*n];
    // c1=X, b0=pX, a0=2pX^2, g=3. Correct products after drop:
    // X^3*X=-1 and 2X^3*X^2=-2X. Wrong late sigma gives
    // sigma(X^2)=-X^2 and sigma(2X^3)=+2X. c0 is constant.
    for limb in 0..4 {
        rows[limb*n+1] = residue(BigInt::from(f.q[3]), f.q[limb]);
        rows[(4+limb)*n+2] = residue(BigInt::from(2)*BigInt::from(f.q[3]), f.q[limb]);
    }
    let correct = oracle(&f, &coefficients, &rows, 3, Variant::Correct);
    let wrong = oracle(&f, &coefficients, &rows, 3, Variant::LateSigma);
    assert_ne!(correct, wrong);
    for limb in 0..3 {
        assert_eq!((correct[limb*n], wrong[limb*n]), (limb as u64 + 1, limb as u64 + 2));
        assert_eq!((correct[(3+limb)*n+1], wrong[(3+limb)*n+1]), (f.q[limb]-2, 2));
    }
    assert_eq!(check(&f, &coefficients, &rows, 3), correct);
}

#[test]
fn row_payload_swap_without_context_change_breaks_digit_row_binding() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let n = f.n;
    let mut coefficients = vec![0; 6*n];
    for limb in 0..3 { coefficients[limb*n+1] = limb as u64 + 2; }
    coefficients[3*n] = 1;
    coefficients[4*n] = 2;
    let mut rows = vec![0; 24*n];
    // d0=1,d1=2; b=(p,3p), a=(2p,4p). Divisibility by p
    // removes rounding ambiguity: correct constants (7,10), swapped (5,8).
    for row in 0..2 {
        for component in 0..2 {
            for limb in 0..4 {
                let multiplier = 1 + 2*row + component;
                rows[((row*2+component)*4+limb)*n] = residue(
                    BigInt::from(multiplier)*BigInt::from(f.q[3]), f.q[limb]);
            }
        }
    }
    let correct = oracle(&f, &coefficients, &rows, 3, Variant::Correct);
    let wrong = oracle(&f, &coefficients, &rows, 3, Variant::RowPayloadPermutation);
    assert_ne!(correct, wrong);
    for limb in 0..3 {
        assert_eq!((correct[limb*n], wrong[limb*n]), (7, 5));
        assert_eq!((correct[(3+limb)*n], wrong[(3+limb)*n]), (10, 8));
    }
    assert_eq!(check(&f, &coefficients, &rows, 3), correct);
}

#[test]
fn omitted_or_untransformed_c0_loses_independent_sigma_placement() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let n = f.n;
    let mut coefficients = vec![0; 6*n];
    for limb in 0..3 { coefficients[limb*n+1] = limb as u64 + 2; }
    let rows = vec![0; 24*n];
    // Zero rows isolate c0: sigma((limb+2)X)=(limb+2)X^3.
    let correct = oracle(&f, &coefficients, &rows, 3, Variant::Correct);
    let omitted = oracle(&f, &coefficients, &rows, 3, Variant::OmitC0);
    let unchanged = oracle(&f, &coefficients, &rows, 3, Variant::UntransformedC0);
    assert_ne!(correct, omitted);
    assert_ne!(correct, unchanged);
    for limb in 0..3 {
        assert_eq!(correct[limb*n+3], limb as u64 + 2);
        assert_eq!((omitted[limb*n+3], unchanged[limb*n+3]), (0, 0));
        assert_eq!(unchanged[limb*n+1], limb as u64 + 2);
    }
    assert_eq!(check(&f, &coefficients, &rows, 3), correct);
}

fn refusal(
    f: &Fixture, contexts: &[NegacyclicNtt; 3], target_contexts: &[NegacyclicNtt; 4],
    source: &[u64], rows: &[u64], g: u64, out_len: usize, scratch_len: usize,
    ceiling: GaloisWork, expected: GaloisError,
) {
    let mut out = sentinels(out_len + 5);
    let mut scratch = sentinels(scratch_len + 7);
    let out_before = out.clone();
    let scratch_before = scratch.clone();
    let source_before = source.to_vec();
    let rows_before = rows.to_vec();
    assert_eq!(f.plan().galois_contract_drop_into(
        ComponentSpectra { contexts, values: source }, RowSpectra { contexts: target_contexts, values: rows },
        g, &mut out[..out_len], &mut scratch[..scratch_len], ceiling,
    ), Err(expected));
    assert_eq!(out, out_before);
    assert_eq!(scratch, scratch_before);
    assert_eq!(source, source_before);
    assert_eq!(rows, rows_before);
}

#[test]
fn malformed_lengths_and_elements_preserve_complete_backing_buffers() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let (coefficients, row_coefficients) = dense(&f);
    let (source, rows) = f.inputs(&coefficients, &row_coefficients);
    let work = f.plan().galois_work().unwrap();
    for len in [0, 23, 25] {
        let mut bad = source.clone(); bad.resize(len, 0);
        refusal(&f, &f.sources, &f.targets, &bad, &rows, 3, 24, 116, work, GaloisError::BufferLength);
        let mut bad = rows.clone(); bad.resize(if len == 0 { 0 } else { len + 72 }, 0);
        refusal(&f, &f.sources, &f.targets, &source, &bad, 3, 24, 116, work, GaloisError::BufferLength);
    }
    for len in [0, 23, 25] {
        refusal(&f, &f.sources, &f.targets, &source, &rows, 3, len, 116, work, GaloisError::BufferLength);
    }
    for len in [0, 115] {
        refusal(&f, &f.sources, &f.targets, &source, &rows, 3, 24, len, work, GaloisError::BufferLength);
    }
    for g in [0, 2, 8, 9, u64::MAX] {
        refusal(&f, &f.sources, &f.targets, &source, &rows, g, 24, 119, work, GaloisError::InvalidAutomorphism);
    }
}

#[test]
fn every_input_and_row_residue_including_last_scanned_before_any_write() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let (coefficients, row_coefficients) = dense(&f);
    let (source, rows) = f.inputs(&coefficients, &row_coefficients);
    let work = f.plan().galois_work().unwrap();
    for at in 0..source.len() {
        for value in [f.q[(at / f.n) % 3], u64::MAX] {
            let mut bad = source.clone(); bad[at] = value;
            refusal(&f, &f.sources, &f.targets, &bad, &rows, 3, 24, 119, work, GaloisError::NonCanonical);
        }
    }
    for at in 0..rows.len() {
        for value in [f.q[(at / f.n) % 4], u64::MAX] {
            let mut bad = rows.clone(); bad[at] = value;
            refusal(&f, &f.sources, &f.targets, &source, &bad, 3, 24, 119, work, GaloisError::NonCanonical);
        }
    }
}

#[test]
fn context_order_duplicate_root_degree_and_constructor_refusals() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let (coefficients, row_coefficients) = dense(&f);
    let (source, rows) = f.inputs(&coefficients, &row_coefficients);
    let work = f.plan().galois_work().unwrap();
    for limb in 0..3 {
        for kind in 0..4 {
            let mut bad = f.sources.clone();
            bad[limb] = match kind {
                0 => f.sources[(limb + 1) % 3].clone(),
                1 => NegacyclicNtt::new(f.sources[limb].ring(), power(f.roots[limb], 3, f.q[limb])).unwrap(),
                2 => NegacyclicNtt::new(CoefficientRing::new(2, f.q[limb]).unwrap(), power(f.roots[limb], 2, f.q[limb])).unwrap(),
                _ => { bad.swap(0, 1); bad[limb].clone() },
            };
            refusal(&f, &bad, &f.targets, &source, &rows, 3, 24, 119, work, GaloisError::ContextMismatch);
        }
    }
    for limb in 0..4 {
        for kind in 0..4 {
            let mut bad = f.targets.clone();
            bad[limb] = match kind {
                0 => f.targets[(limb + 1) % 4].clone(),
                1 => NegacyclicNtt::new(bad[limb].ring(), power(f.roots[limb], 3, f.q[limb])).unwrap(),
                2 => NegacyclicNtt::new(CoefficientRing::new(2, f.q[limb]).unwrap(), power(f.roots[limb], 2, f.q[limb])).unwrap(),
                _ => { bad.swap(0, 3); bad[limb].clone() },
            };
            refusal(&f, &f.sources, &bad, &source, &rows, 3, 24, 119, work, GaloisError::ContextMismatch);
        }
    }
    let mut bad = f.targets.clone(); bad.swap(0, 1);
    assert!(matches!(RnsContraction::new(&f.sources, &bad), Err(Error::ContextMismatch)));
    let mut bad = f.targets.clone(); bad[3] = bad[0].clone();
    assert!(matches!(RnsContraction::new(&f.sources, &bad), Err(Error::ContextMismatch)));
    let mut bad = f.targets.clone();
    bad[3] = NegacyclicNtt::new(CoefficientRing::new(2, f.q[3]).unwrap(), power(f.roots[3], 2, f.q[3])).unwrap();
    assert!(matches!(RnsContraction::new(&f.sources, &bad), Err(Error::ContextMismatch)));
}

#[test]
fn every_positive_ceiling_field_one_short_is_atomic() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    let (coefficients, row_coefficients) = dense(&f);
    let (source, rows) = f.inputs(&coefficients, &row_coefficients);
    let work = f.plan().galois_work().unwrap();
    let mut attempts = 0;
    macro_rules! short {
        ($($field:ident).+) => {{
            if work.$($field).+ > 0 {
                let mut bad = work;
                bad.$($field).+ -= 1;
                refusal(&f, &f.sources, &f.targets, &source, &rows, 3, 24, 119, bad, GaloisError::WorkLimit);
                attempts += 1;
            }
        }};
    }
    macro_rules! events {
        ($($prefix:ident).+) => {
            short!($($prefix).+.checks); short!($($prefix).+.residue_checks);
            short!($($prefix).+.moves); short!($($prefix).+.swaps);
            short!($($prefix).+.modular_additions); short!($($prefix).+.modular_subtractions);
            short!($($prefix).+.modular_multiplications); short!($($prefix).+.exponentiations);
            short!($($prefix).+.exponent_steps); short!($($prefix).+.primality_rounds);
        };
    }
    macro_rules! construction {
        ($($prefix:ident).+) => {
            short!($($prefix).+.checks); short!($($prefix).+.descriptor_stores);
            short!($($prefix).+.euclid_iterations); short!($($prefix).+.signed_divisions);
            short!($($prefix).+.signed_multiplications); short!($($prefix).+.signed_subtractions);
            short!($($prefix).+.signed_remainders); short!($($prefix).+.unsigned_remainders);
        };
    }
    events!(composition.contraction.events); short!(composition.contraction.coefficient_reductions);
    events!(composition.extra.inverse_ntts);
    short!(composition.extra.drop.checks); short!(composition.extra.drop.residue_checks);
    short!(composition.extra.drop.moves); short!(composition.extra.drop.modular_subtractions);
    short!(composition.extra.drop.modular_multiplications); short!(composition.extra.drop.unsigned_reductions);
    short!(composition.extra.drop.ordinary_subtractions); short!(composition.extra.drop.ordinary_divisions);
    construction!(composition.extra.construction);
    short!(composition.extra.control.capacity_outcomes); short!(composition.extra.control.adapters);
    short!(composition.extra.control.result_inspections); short!(composition.extra.control.ceiling_comparisons);
    short!(composition.extra.control.length_comparisons); short!(composition.extra.control.dispatches);
    short!(composition.extra.control.selected_loop_polls); short!(composition.extra.control.slice_partitions);
    short!(composition.extra.control.selected_phase_calls);
    events!(extra.automorphisms); events!(extra.inverse_ntts);
    short!(extra.signed_modular_additions); short!(extra.final_copies);
    construction!(extra.construction);
    short!(extra.control.validation_checks); short!(extra.control.residue_checks);
    short!(extra.control.capacity_outcomes); short!(extra.control.adapters);
    short!(extra.control.result_inspections); short!(extra.control.ceiling_comparisons);
    short!(extra.control.borrowed_component_visits); short!(extra.control.selected_loop_polls);
    short!(extra.control.slice_partitions); short!(extra.control.phase_calls);
    short!(extra.control.index_products); short!(extra.control.index_additions);
    short!(extra.control.index_remainders); short!(extra.control.index_comparisons);
    short!(extra.control.index_subtractions); short!(extra.control.index_divisions);
    // 92 fields visited, 16 zero NTT categories intentionally not decremented.
    assert_eq!(attempts, 76);
}

#[test]
fn explicit_public8192_sparse_analytic_and_legacy_refusal() {
    let n = 8192;
    let f = Fixture::new(n, [562949952847873, 562949952798721, 562949952700417, 562949952274433]);
    assert!(CoefficientRing::new(n, f.q[0]).is_err());
    assert!(CoefficientRing::new(4096, f.q[0]).is_ok());
    let mut coefficients = vec![0; 6*n];
    for limb in 0..3 {
        coefficients[limb*n + n-1] = f.q[limb] - 1;
        coefficients[(3+limb)*n] = limb as u64 + 1;
    }
    let mut rows = vec![0; 24*n];
    // Only row0 active: constant 2p in b, constant p in a, both are 0 mod p.
    // Thus Dp gives exactly 2 and 1; c1 row0 digit is constant 1 for every g.
    for limb in 0..4 {
        rows[limb*n] = (2*f.q[3]) % f.q[limb];
        rows[(4+limb)*n] = f.q[3] % f.q[limb];
    }
    for g in [1, 3, 16383] {
        let out = check(&f, &coefficients, &rows, g);
        for limb in 0..3 {
            let mut expected = sigma(&coefficients[limb*n..(limb+1)*n], g, f.q[limb]);
            expected[0] = (expected[0] + 2) % f.q[limb];
            assert_eq!(&out[limb*n..(limb+1)*n], expected);
            assert_eq!(out[(3+limb)*n], 1);
            assert!(out[(3+limb)*n+1..(4+limb)*n].iter().all(|&v| v == 0));
        }
    }
}

#[test]
fn quote_contract_regression_and_exact_storage_endpoints() {
    // QUOTE-CONTRACT REGRESSION: expected quote fields intentionally pin the
    // published formulas/constants. This does not independently enumerate real
    // operation events or prove envelopes. The separate one-short test covers
    // all 76 positive fields, with 16 zero markers among 92 total fields.
    for n in [1, 2, 4, 8] {
        let f = Fixture::new(n, [17, 97, 113, 193]);
        let plan = f.plan();
        let work = plan.galois_work().unwrap();
        let count = n as u64;
        let butterflies = count / 2 * u64::from(n.trailing_zeros());
        assert_eq!(work.extra.automorphisms, LogicalOperations {
            checks: 3*count + 15, residue_checks: 3*count, moves: 3*count,
            ..LogicalOperations::default()
        });
        assert_eq!(work.extra.inverse_ntts, LogicalOperations {
            checks: 3*count + 6, residue_checks: 3*count,
            moves: 6*count + 6*butterflies, swaps: 3*(count/2),
            modular_additions: 3*butterflies, modular_subtractions: 3*butterflies,
            modular_multiplications: if n == 1 { 0 } else { 6*butterflies + 9*count },
            ..LogicalOperations::default()
        });
        assert_eq!(work.extra.signed_modular_additions, 3*count);
        assert_eq!(work.extra.final_copies, 6*count);
        assert_eq!(work.extra.construction, DropConstructionWork {
            checks: 390, descriptor_stores: 15, euclid_iterations: 384,
            signed_divisions: 384, signed_multiplications: 768,
            signed_subtractions: 768, signed_remainders: 3, unsigned_remainders: 3,
        });
        assert_eq!(work.extra.control, GaloisControlWork {
            validation_checks: 30*count+1024, residue_checks: 30*count,
            capacity_outcomes: 1024, adapters: 1024, result_inspections: 2048,
            ceiling_comparisons: 92, borrowed_component_visits: 2,
            selected_loop_polls: 40*count+1024, slice_partitions: 256, phase_calls: 128,
            index_products: 12*count+256, index_additions: 12*count+256,
            index_remainders: 6*count+256, index_comparisons: 3*count+256,
            index_subtractions: 6*count+256, index_divisions: 3*count+256,
        });
        let storage = plan.galois_storage().unwrap();
        assert_eq!((storage.component_words, storage.composition_end, storage.result_end), (3*n, 22*n, 28*n));
        assert_eq!(storage.scratch_words, 29*n);
        // Valid descriptor degree bound makes overflow unreachable at operation
        // entry. Independently check 65N/520N fit even at maximum admitted N.
        assert_eq!(storage.total_words.checked_mul(8), Some(storage.total_bytes));
    }
    assert_eq!(8192_usize.checked_mul(520), Some(4_259_840));
}

#[test]
fn independent_named_payload_partition_excludes_descriptors_control_and_tails() {
    // Derive storage from simultaneously live named slices, independently of
    // production's storage arithmetic. These are payload words, NOT physical
    // peak memory; descriptor/scalar/control storage and caller tails excluded.
    for n in [1, 4, 8192] {
        let f = Fixture::new(n, if n == 8192 {
            [562949952847873, 562949952798721, 562949952700417, 562949952274433]
        } else { [17, 97, 113, 193] });
        let substituted_c1 = 3*n;
        let contracted_spectra = 2*4*n;
        let inverse_coefficients = 2*4*n;
        let contraction_scratch = n+n+n;
        let staged_result = 2*3*n;
        let c0_inverse = n;
        let old_composition = contracted_spectra + inverse_coefficients + contraction_scratch;
        let scratch = substituted_c1 + old_composition + staged_result + c0_inverse;
        let external_source = 2*3*n;
        let external_rows = 3*2*4*n;
        let external_output = 2*3*n;
        let total = external_source + external_rows + external_output + scratch;
        assert_eq!((old_composition, scratch, total, total*8), (19*n, 29*n, 65*n, 520*n));
        let actual = f.plan().galois_storage().unwrap();
        assert_eq!((actual.source_words, actual.row_words, actual.output_words),
            (external_source, external_rows, external_output));
        assert_eq!((actual.component_words, actual.composition_end, actual.result_end),
            (substituted_c1, substituted_c1+old_composition, substituted_c1+old_composition+staged_result));
        assert_eq!((actual.scratch_words, actual.total_words, actual.total_bytes), (scratch, total, total*8));
    }
}

#[test]
fn all_tiny_basis_positions_rows_components_and_signed_wrap() {
    let f = Fixture::new(4, [17, 97, 113, 193]);
    for row in 0..3 {
        for component in 0..2 {
            for from in 0..f.n {
                for at in 0..f.n {
                    let mut coefficients = vec![0; 6*f.n];
                    // Distinct c0 independently detects component placement.
                    for limb in 0..3 { coefficients[limb*f.n + (from+1)%f.n] = limb as u64 + 1; }
                    coefficients[(3+row)*f.n + from] = f.q[row] - 1;
                    let mut rows = vec![0; 24*f.n];
                    for limb in 0..4 { rows[((row*2+component)*4+limb)*f.n + at] = f.q[limb] - 1; }
                    for g in [1, 3, 5, 7] { check(&f, &coefficients, &rows, g); }
                }
            }
        }
    }
}

#[test]
fn public8192_late_residue_length_context_and_work_refusals_are_atomic() {
    let n = 8192;
    let f = Fixture::new(n, [562949952847873, 562949952798721, 562949952700417, 562949952274433]);
    let source = vec![0; 6*n];
    let rows = vec![0; 24*n];
    let work = f.plan().galois_work().unwrap();
    let mut bad = source.clone(); bad[6*n-1] = f.q[2];
    refusal(&f, &f.sources, &f.targets, &bad, &rows, 3, 6*n, 29*n+3, work, GaloisError::NonCanonical);
    let mut bad = rows.clone(); bad[24*n-1] = f.q[3];
    refusal(&f, &f.sources, &f.targets, &source, &bad, 3, 6*n, 29*n+3, work, GaloisError::NonCanonical);
    refusal(&f, &f.sources, &f.targets, &source, &rows, 3, 6*n, 29*n-1, work, GaloisError::BufferLength);
    refusal(&f, &f.sources, &f.targets, &source, &rows, 3, 6*n+1, 29*n+3, work, GaloisError::BufferLength);
    let mut contexts = f.sources.clone(); contexts.swap(0, 1);
    refusal(&f, &contexts, &f.targets, &source, &rows, 3, 6*n, 29*n+3, work, GaloisError::ContextMismatch);
    let mut bad = work; bad.extra.final_copies -= 1;
    refusal(&f, &f.sources, &f.targets, &source, &rows, 3, 6*n, 29*n+3, bad, GaloisError::WorkLimit);
}

// Source equation only: three Garner rows, not eight base-2^20 rows.
// Authored and UNRUN. No keys, imported source, or production expected-value path.
const RELATION_Q: u64 = 186337;
const RELATION_P: u64 = 193;
const RELATION_MODULI: [u64; 4] = [17, 97, 113, 193];

fn relation_poly(values: &[i64]) -> Vec<BigInt> {
    assert!(values.len() <= 8);
    let mut out = vec![BigInt::from(0); 8];
    for (to, &from) in out.iter_mut().zip(values) { *to = BigInt::from(from); }
    out
}

fn relation_canonical(value: &BigInt, modulus: u64) -> BigInt {
    let m = BigInt::from(modulus);
    ((value % &m) + &m) % &m
}

fn relation_add(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}

fn relation_sub(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    a.iter().zip(b).map(|(a, b)| a - b).collect()
}

fn relation_scale(a: &[BigInt], scale: &BigInt) -> Vec<BigInt> {
    a.iter().map(|a| a * scale).collect()
}

fn relation_mul(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    assert_eq!((a.len(), b.len()), (8, 8));
    let mut out = relation_poly(&[]);
    // Both operands signed, including secret and row errors. No unsigned oracle.
    for i in 0..8 {
        for j in 0..8 {
            if i + j < 8 { out[i+j] += &a[i] * &b[j]; }
            else { out[i+j-8] -= &a[i] * &b[j]; }
        }
    }
    out
}

fn relation_sigma(a: &[BigInt], g: u64) -> Vec<BigInt> {
    let mut out = relation_poly(&[]);
    for (i, value) in a.iter().enumerate() {
        let exponent = i as u64 * g;
        let at = (exponent % 8) as usize;
        if (exponent / 8) % 2 == 0 { out[at] += value; }
        else { out[at] -= value; }
    }
    out
}

fn relation_abs(value: &BigInt) -> BigInt {
    if value < &BigInt::from(0) { -value } else { value.clone() }
}

fn relation_l1(a: &[BigInt]) -> BigInt { a.iter().map(relation_abs).sum() }

fn relation_linf(a: &[BigInt]) -> BigInt {
    a.iter().map(relation_abs).max().unwrap()
}

fn relation_garner() -> [BigInt; 3] {
    let weights = core::array::from_fn(|i| {
        let qi = RELATION_MODULI[i];
        let quotient = BigInt::from(RELATION_Q / qi);
        // Canonical inverse BEFORE multiplying by q/qi; do not reduce G mod qi.
        let inverse = quotient.modinv(&BigInt::from(qi)).unwrap();
        assert!(inverse >= BigInt::from(0) && inverse < BigInt::from(qi));
        quotient * inverse
    });
    assert_eq!(weights, [43844, 97971, 44523].map(BigInt::from));
    assert_eq!(weights.iter().cloned().sum::<BigInt>(), BigInt::from(RELATION_Q + 1));
    weights
}

fn relation_source_rows(
    secret: &[BigInt], g: u64, a: &[Vec<BigInt>; 3], errors: &[Vec<BigInt>; 3],
    weights: &[BigInt; 3], factor: u64,
) -> [[Vec<BigInt>; 2]; 3] {
    let switched_secret = relation_sigma(secret, g);
    core::array::from_fn(|i| {
        let message = relation_scale(&switched_secret, &(BigInt::from(factor) * &weights[i]));
        let b = relation_add(&relation_sub(&errors[i], &relation_mul(&a[i], secret)), &message);
        [b, a[i].clone()]
    })
}

fn relation_encode_rows(rows: &[[Vec<BigInt>; 2]; 3]) -> Vec<u64> {
    let mut out = Vec::with_capacity(192);
    for row in rows {
        for component in row {
            for modulus in RELATION_MODULI {
                out.extend(component.iter().map(|x| residue(x.clone(), modulus)));
            }
        }
    }
    out
}

fn relation_encode_components(components: &[Vec<BigInt>; 2]) -> Vec<u64> {
    let mut out = Vec::with_capacity(48);
    for component in components {
        for modulus in &RELATION_MODULI[..3] {
            out.extend(component.iter().map(|x| residue(x.clone(), *modulus)));
        }
    }
    out
}

fn relation_actual(f: &Fixture, c0: &[BigInt], c1: &[BigInt], rows: &[u64], g: u64) -> Vec<u64> {
    let coefficients = relation_encode_components(&[c0.to_vec(), c1.to_vec()]);
    let (source, rows) = f.inputs(&coefficients, rows);
    let plan = f.plan();
    let storage = plan.galois_storage().unwrap();
    let mut output = vec![0; storage.output_words];
    let mut scratch = vec![0; storage.scratch_words];
    // Actual side ONLY. In particular, do not call historical check()/rounded().
    plan.galois_contract_drop_into(
        ComponentSpectra { contexts: &f.sources, values: &source },
        RowSpectra { contexts: &f.targets, values: &rows }, g,
        &mut output, &mut scratch, plan.galois_work().unwrap(),
    ).unwrap();
    output
}

fn relation_crt(residues: &[BigInt; 4]) -> BigInt {
    let m = RELATION_Q * RELATION_P;
    let mut total = BigInt::from(0);
    for (i, modulus) in RELATION_MODULI.iter().copied().enumerate() {
        let quotient = BigInt::from(m / modulus);
        let inverse = quotient.modinv(&BigInt::from(modulus)).unwrap();
        total += &residues[i] * quotient * inverse;
    }
    let z = relation_canonical(&total, m);
    for i in 0..4 {
        assert_eq!(relation_canonical(&z, RELATION_MODULI[i]), residues[i]);
    }
    z
}

fn relation_drop(canonical_m: &[BigInt]) -> Vec<BigInt> {
    canonical_m.iter().map(|z| {
        assert!(z >= &BigInt::from(0) && z < &BigInt::from(RELATION_Q * RELATION_P));
        // Nonnegative floor division, full M representative, not qi,p rounding.
        relation_canonical(&((z + BigInt::from(96)) / BigInt::from(193)), RELATION_Q)
    }).collect()
}

struct RelationOracle {
    components: [Vec<BigInt>; 2],
    v: [Vec<BigInt>; 2],
    digits: [Vec<BigInt>; 3],
}

fn relation_oracle(c0: &[BigInt], c1: &[BigInt], rows: &[u64], g: u64) -> RelationOracle {
    assert_eq!(rows.len(), 192);
    let substituted = relation_sigma(c1, g);
    let digits: [Vec<BigInt>; 3] = core::array::from_fn(|i| {
        substituted.iter().map(|x| relation_canonical(x, RELATION_MODULI[i])).collect()
    });
    let v: [Vec<BigInt>; 2] = core::array::from_fn(|component| {
        let limbs: [Vec<BigInt>; 4] = core::array::from_fn(|limb| {
            let mut sum = relation_poly(&[]);
            for row in 0..3 {
                let start = ((row * 2 + component) * 4 + limb) * 8;
                let payload: Vec<_> = rows[start..start+8].iter().copied().map(BigInt::from).collect();
                // Lift same canonical qi digit unchanged to every target limb.
                sum = relation_add(&sum, &relation_mul(&digits[row], &payload));
            }
            sum.iter().map(|x| relation_canonical(x, RELATION_MODULI[limb])).collect()
        });
        (0..8).map(|at| relation_crt(&core::array::from_fn(|limb| limbs[limb][at].clone()))).collect()
    });
    let components = [relation_add(&relation_sigma(c0, g), &relation_drop(&v[0])), relation_drop(&v[1])];
    RelationOracle { components, v, digits }
}

fn relation_assert_phase(actual: &[u64], secret: &[BigInt], expected: &[BigInt]) {
    for limb in 0..3 {
        let b: Vec<_> = actual[limb*8..(limb+1)*8].iter().copied().map(BigInt::from).collect();
        let a: Vec<_> = actual[(3+limb)*8..(4+limb)*8].iter().copied().map(BigInt::from).collect();
        let phase = relation_add(&b, &relation_mul(&a, secret));
        for at in 0..8 {
            assert_eq!(relation_canonical(&phase[at], RELATION_MODULI[limb]),
                relation_canonical(&expected[at], RELATION_MODULI[limb]), "limb={limb}, at={at}");
        }
    }
}

#[test]
fn relation_synthetic_source_rows_full_m_phase_identity_and_bound() {
    let f = Fixture::new(8, RELATION_MODULI);
    let secret = relation_poly(&[1, 1, 0, 0, 0, 0, 0, -1]);
    let c0 = relation_poly(&[3, 0, -1]);
    let c1 = relation_poly(&[-1, 2, 0, 0, 0, 0, 0, 1]);
    let a = core::array::from_fn(|i| {
        let mut value = relation_poly(&[i as i64 + 1]); value[7] = BigInt::from(2); value
    });
    let weights = relation_garner();
    let mut nonzero_xi = [false; 2];
    let mut cases = 0;
    for (noise_case, noisy) in [false, true].into_iter().enumerate() {
        let errors: [Vec<BigInt>; 3] = core::array::from_fn(|i| {
            let mut value = relation_poly(&[]);
            if noisy { value[i] = BigInt::from(if i % 2 == 0 { 1 } else { -1 }); }
            value
        });
        for g in [1, 3, 11, 15] {
            let rows = relation_source_rows(&secret, g, &a, &errors, &weights, RELATION_P);
            let encoded = relation_encode_rows(&rows);
            // Verify source row phase in ALL four limbs, including p.
            for row in 0..3 {
                for limb in 0..4 {
                    let start = row * 64 + limb * 8;
                    let b: Vec<_> = encoded[start..start+8].iter().copied().map(BigInt::from).collect();
                    let a: Vec<_> = encoded[start+32..start+40].iter().copied().map(BigInt::from).collect();
                    let lhs = relation_add(&b, &relation_mul(&a, &secret));
                    let rhs = relation_add(&errors[row], &relation_scale(&relation_sigma(&secret, g),
                        &(BigInt::from(RELATION_P) * &weights[row])));
                    for at in 0..8 {
                        assert_eq!(relation_canonical(&lhs[at], RELATION_MODULI[limb]),
                            relation_canonical(&rhs[at], RELATION_MODULI[limb]));
                    }
                }
            }
            let oracle = relation_oracle(&c0, &c1, &encoded, g);
            let actual = relation_actual(&f, &c0, &c1, &encoded, g);
            assert_eq!(actual, relation_encode_components(&oracle.components));
            let substituted = relation_sigma(&c1, g);
            let mut reconstructed = relation_poly(&[]);
            let mut error = relation_poly(&[]);
            let mut noise_bound = BigInt::from(0);
            for row in 0..3 {
                for at in 0..8 {
                    assert_eq!(oracle.digits[row][at], relation_canonical(&substituted[at], RELATION_MODULI[row]));
                    assert!(oracle.digits[row][at] >= BigInt::from(0));
                    assert!(oracle.digits[row][at] < BigInt::from(RELATION_MODULI[row]));
                }
                reconstructed = relation_add(&reconstructed, &relation_scale(&oracle.digits[row], &weights[row]));
                error = relation_add(&error, &relation_mul(&oracle.digits[row], &errors[row]));
                noise_bound += relation_l1(&oracle.digits[row]) * relation_linf(&errors[row]);
            }
            for at in 0..8 {
                assert_eq!(relation_canonical(&reconstructed[at], RELATION_Q), relation_canonical(&substituted[at], RELATION_Q));
            }
            let centered: [Vec<BigInt>; 2] = core::array::from_fn(|component| {
                oracle.v[component].iter().map(|z| {
                    let r = relation_canonical(z, RELATION_P);
                    if r > BigInt::from(96) { r - BigInt::from(193) } else { r }
                }).collect()
            });
            let numerator = relation_sub(&relation_sub(&error, &centered[0]), &relation_mul(&centered[1], &secret));
            let xi: Vec<_> = numerator.iter().map(|x| {
                assert_eq!(x % BigInt::from(193), BigInt::from(0), "exact integer divisibility");
                x / BigInt::from(193)
            }).collect();
            // Canonical-M wraps contribute multiples of q after division by p.
            let phase = relation_add(&relation_sigma(&relation_add(&c0, &relation_mul(&c1, &secret)), g), &xi);
            relation_assert_phase(&actual, &secret, &phase);
            let bound = &noise_bound + BigInt::from(96) * (BigInt::from(1) + relation_l1(&secret));
            assert!(BigInt::from(193) * relation_linf(&xi) <= bound);
            assert!(bound > BigInt::from(0));
            if noisy {
                assert!(noise_bound > BigInt::from(0));
                assert!(relation_linf(&error) > BigInt::from(0));
            } else { assert_eq!(error, relation_poly(&[])); }
            nonzero_xi[noise_case] |= relation_linf(&xi) > BigInt::from(0);
            cases += 1;
        }
    }
    assert_eq!(cases, 8);
    assert_eq!(nonzero_xi, [true, true], "zero row noise need not mean zero switch error");
}

#[test]
fn relation_analytic_row_semantic_mutations_and_error_sign_boundary() {
    let f = Fixture::new(8, RELATION_MODULI);
    let zero = relation_poly(&[]);
    let one = relation_poly(&[1]);
    let secret = relation_poly(&[0, 1]);
    let a = core::array::from_fn(|_| zero.clone());
    let errors = core::array::from_fn(|_| zero.clone());
    let weights = relation_garner();
    let correct = relation_source_rows(&secret, 3, &a, &errors, &weights, RELATION_P);
    let quotient_weights = core::array::from_fn(|i| BigInt::from(RELATION_Q / RELATION_MODULI[i]));
    // Analytic constants: sum(q/qi)=10961+1921+1649=14531;
    // sum(G)=186338=193*965+93, so omitted-p rounding gives 965.
    assert_eq!(quotient_weights.iter().cloned().sum::<BigInt>(), BigInt::from(14531));
    assert_eq!(186338_u64, 193 * 965 + 93);
    let wrong_weights = relation_source_rows(&secret, 3, &a, &errors, &quotient_weights, RELATION_P);
    let omitted_p = relation_source_rows(&secret, 3, &a, &errors, &weights, 1);
    let inverse_g = relation_source_rows(&secret, 11, &a, &errors, &weights, RELATION_P);
    let omitted_sigma = relation_source_rows(&secret, 1, &a, &errors, &weights, RELATION_P);
    let swapped = core::array::from_fn(|i| [correct[i][1].clone(), correct[i][0].clone()]);
    // sigma_11(X)=X^11=-X^3, and swapping puts X^3 in a: phase X^4.
    let cases = [
        (correct, relation_poly(&[0, 0, 0, 1])),
        (wrong_weights, relation_poly(&[0, 0, 0, 14531])),
        (omitted_p, relation_poly(&[0, 0, 0, 965])),
        (inverse_g, relation_poly(&[0, 0, 0, -1])),
        (omitted_sigma, relation_poly(&[0, 1])),
        (swapped, relation_poly(&[0, 0, 0, 0, 1])),
    ];
    assert_eq!(cases.len(), 6);
    for (rows, literal_phase) in cases {
        let encoded = relation_encode_rows(&rows);
        let oracle = relation_oracle(&zero, &one, &encoded, 3);
        let actual = relation_actual(&f, &zero, &one, &encoded, 3);
        assert_eq!(actual, relation_encode_components(&oracle.components));
        relation_assert_phase(&actual, &secret, &literal_phase);
    }
    let boundary_a = [relation_poly(&[97]), zero.clone(), zero.clone()];
    // a0=97 rounds to 1. Summed b offsets: -96 rounds to 0,
    // whereas -98 rounds to -1; including sum(G) yields phases 2 and 1.
    for (sign, phase) in [(1, 2), (-1, 1)] {
        let boundary_e = [relation_poly(&[sign]), zero.clone(), zero.clone()];
        let rows = relation_source_rows(&one, 3, &boundary_a, &boundary_e, &weights, RELATION_P);
        let encoded = relation_encode_rows(&rows);
        let oracle = relation_oracle(&zero, &one, &encoded, 3);
        let actual = relation_actual(&f, &zero, &one, &encoded, 3);
        assert_eq!(actual, relation_encode_components(&oracle.components));
        relation_assert_phase(&actual, &one, &relation_poly(&[phase]));
    }
}

#[test]
fn relation_rounding_secret_noncommutation_with_zero_row_noise() {
    let f = Fixture::new(8, RELATION_MODULI);
    let zero = relation_poly(&[]);
    let one = relation_poly(&[1]);
    let secret = relation_poly(&[1, 1]);
    let a = [relation_poly(&[96, 96]), zero.clone(), zero.clone()];
    let errors = core::array::from_fn(|_| zero.clone());
    let rows = relation_source_rows(&secret, 3, &a, &errors, &relation_garner(), RELATION_P);
    let encoded = relation_encode_rows(&rows);
    let oracle = relation_oracle(&zero, &one, &encoded, 3);
    assert_eq!(oracle.v[1], relation_poly(&[96, 96]));
    assert_eq!(relation_drop(&oracle.v[1]), zero);
    let v1_times_secret = relation_mul(&oracle.v[1], &secret);
    assert_eq!(v1_times_secret, relation_poly(&[96, 192, 96]));
    assert_eq!(relation_drop(&v1_times_secret), relation_poly(&[0, 1]));
    assert_eq!(relation_mul(&relation_drop(&oracle.v[1]), &secret), zero);
    let actual = relation_actual(&f, &zero, &one, &encoded, 3);
    assert_eq!(actual, relation_encode_components(&oracle.components));
    let literal_phase = relation_poly(&[1, -1, 0, 1]);
    assert_eq!(relation_encode_components(&oracle.components),
        relation_encode_components(&[literal_phase.clone(), zero.clone()]));
    relation_assert_phase(&actual, &secret, &literal_phase);
    // E=0, r1=96+96X, r0=-96+X-96X^2, hence xi=-X exactly.
    let centered: [Vec<BigInt>; 2] = core::array::from_fn(|j| oracle.v[j].iter().map(|z| {
        let r = relation_canonical(z, RELATION_P);
        if r > BigInt::from(96) { r - BigInt::from(193) } else { r }
    }).collect());
    assert_eq!(centered[0], relation_poly(&[-96, 1, -96]));
    assert_eq!(centered[1], relation_poly(&[96, 96]));
    let numerator = relation_sub(&relation_sub(&zero, &centered[0]), &relation_mul(&centered[1], &secret));
    assert_eq!(numerator, relation_poly(&[0, -193]));
    let xi: Vec<_> = numerator.iter().map(|x| {
        assert_eq!(x % BigInt::from(193), BigInt::from(0));
        x / BigInt::from(193)
    }).collect();
    assert_eq!(xi, relation_poly(&[0, -1]));
    let combined = relation_add(&oracle.v[0], &v1_times_secret);
    let canonical_combined: Vec<_> = combined.iter().map(|x| relation_canonical(x, RELATION_Q * RELATION_P)).collect();
    let once_rounded_wrong_phase = relation_drop(&canonical_combined);
    assert_eq!(once_rounded_wrong_phase, relation_poly(&[1, 0, 0, 1]));
    assert_ne!(once_rounded_wrong_phase, literal_phase);
    assert_eq!(BigInt::from(193) * relation_linf(&xi), BigInt::from(193));
    assert!(BigInt::from(193) <= BigInt::from(96) * (BigInt::from(1) + relation_l1(&secret)));
}
