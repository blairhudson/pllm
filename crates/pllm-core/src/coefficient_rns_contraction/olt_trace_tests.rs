//! Authored source regressions, ALL UNRUN under Native HOLD.
//! Independent signed schoolbook + GLOBAL four-limb BigInt CRT/rounded drop,
//! recursive tree (not production schedule), direct DFT fixture encoding.
//! Expected arithmetic never calls production NTT/contraction/trace kernels.
//! Synthetic row errors are public, not source key provenance or HE evidence.
//! Existing selector/marker payloads do not establish a source relation or phase
//! law. The separate phase_ fixture below checks a public deterministic TEST
//! convention only: unadopted c0+c1*s / forward-g, not a B-N1 decision, paper key
//! orientation, RLWE/issuer/security evidence or noise-admission claim.
//! Ideal selector rows remain exponent-independent; order needs other witnesses.
use super::olt_trace::*;
use super::*;
use crate::coefficient_layout::{CoefficientRing, DegreeBound};
use crate::coefficient_layout::schedule::{InterleaveSchedule, InterleaveStep};
use num_bigint::BigInt;

// Child tests reuse the private independent DFT/BigInt CRT/tree fixtures, not
// production arithmetic, and do not widen a test fixture into a public API.
#[path = "olt_tests.rs"]
mod olt_tests;

fn residue(value: BigInt, q: u64) -> u64 {
    let modulus = BigInt::from(q);
    (((value % &modulus) + &modulus) % &modulus).to_u64_digits()
        .1.first().copied().unwrap_or(0)
}

fn power(a: u64, exponent: u64, q: u64) -> u64 {
    residue(BigInt::from(a).modpow(&BigInt::from(exponent), &BigInt::from(q)), q)
}

/// Independent sparse-support direct evaluation at psi^(2k+1).
fn direct_dft(a: &[u64], psi: u64, q: u64) -> Vec<u64> {
    let support: Vec<_> = a.iter().copied().enumerate().filter(|&(_, v)| v != 0).collect();
    (0..a.len()).map(|k| {
        let mut sum = BigInt::from(0);
        for &(j, value) in &support {
            sum += BigInt::from(value) * BigInt::from(power(psi, ((2*k+1)*j) as u64, q));
        }
        residue(sum, q)
    }).collect()
}

fn monomial(a: &[u64], exponent: usize, q: u64) -> Vec<u64> {
    let n = a.len();
    let mut out = vec![BigInt::from(0); n];
    for (j, &value) in a.iter().enumerate() {
        let degree = j * exponent;
        let signed = if (degree / n) % 2 == 0 { BigInt::from(value) } else { -BigInt::from(value) };
        out[degree % n] += signed;
    }
    out.into_iter().map(|v| residue(v, q)).collect()
}

fn schoolbook(a: &[BigInt], b: &[u64]) -> Vec<BigInt> {
    let n = a.len();
    let support: Vec<_> = b.iter().copied().enumerate().filter(|&(_, v)| v != 0).collect();
    let mut out = vec![BigInt::from(0); n];
    for (i, value) in a.iter().enumerate() {
        if *value == BigInt::from(0) { continue; }
        for &(j, b) in &support {
            let product = value * BigInt::from(b);
            if i+j < n { out[i+j] += product; } else { out[i+j-n] -= product; }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mutation {
    None, Centered, BeforeDrop, Components, SourceOrder, TargetOrder, NoErrors,
    InverseAutomorphism, ReverseTraceLevels, WrongTraceLevel, OmitNormalization,
    FloorRescale, WrongRow { exponent: u64, replacement: u64 },
}

struct Fixture {
    n: usize,
    q: [u64; 4],
    roots: [u64; 4],
    sources: [NegacyclicNtt; 3],
    targets: [NegacyclicNtt; 4],
    input: Vec<u64>,
    exponents: Vec<u64>,
    row_coefficients: Vec<Vec<u64>>,
    row_spectra: Vec<Vec<u64>>,
    crt_modulus: BigInt,
    crt_weights: [BigInt; 4],
    direction: u64,
}

impl Fixture {
    fn new(n: usize, stride: usize, count: usize, q: [u64; 4], direction: u64, errors: bool) -> Self {
        let roots = q.map(|q| (2..=128).find_map(|base| {
            let root = power(base, (q-1)/(2*n as u64), q);
            (power(root, n as u64, q) == q-1).then_some(root)
        }).expect("independent bounded root witness"));
        let targets = core::array::from_fn(|i| NegacyclicNtt::new(
            CoefficientRing::bounded(n, q[i], DegreeBound::Public8192).unwrap(), roots[i],
        ).unwrap());
        let sources = core::array::from_fn(|i| targets[i].clone());
        let mut input = vec![0; count*6*n];
        for lane in 0..count {
            for poly in 0..6 {
                let q = q[poly%3];
                input[(lane*6+poly)*n] = (direction+3+lane as u64+poly as u64)%q;
                if n > 1 {
                    // Explicit unsigned high digit and negacyclic sign witness.
                    input[(lane*6+poly)*n+n-1] = q-1-(lane as u64%3);
                }
            }
        }
        let mut exponents: Vec<_> = InterleaveSchedule::bounded(n, stride, count, 8192).unwrap()
            .steps().filter_map(|step| match step {
                InterleaveStep::Trace { element, .. } | InterleaveStep::Merge { element, .. } => Some(element as u64),
                _ => None,
            }).collect();
        exponents.sort_unstable();
        exponents.dedup();
        let row_coefficients: Vec<_> = exponents.iter().map(|&g| Self::rows(n, q, g, direction, errors)).collect();
        let row_spectra: Vec<Vec<u64>> = row_coefficients.iter().map(|row| (0..24).flat_map(|poly| {
            direct_dft(&row[poly*n..(poly+1)*n], roots[poly%4], q[poly%4])
        }).collect()).collect();
        let crt_modulus = q.iter().fold(BigInt::from(1), |product, &q| product*BigInt::from(q));
        let crt_weights = core::array::from_fn(|i| {
            let partial = &crt_modulus / BigInt::from(q[i]);
            &partial * partial.modinv(&BigInt::from(q[i])).unwrap()
        });
        if errors {
            assert_distinct_payloads(&row_coefficients);
            assert_distinct_payloads(&row_spectra);
        }
        Self { n, q, roots, sources, targets, input, exponents, row_coefficients, row_spectra,
            crt_modulus, crt_weights, direction }
    }

    fn rows(n: usize, q: [u64; 4], g: u64, direction: u64, errors: bool) -> Vec<u64> {
        let mut rows = vec![0; 24*n];
        for source in 0..3 {
            for component in 0..2 {
                for target in 0..4 {
                    let at = ((source*2+component)*4+target)*n;
                    // Zero-error selector: b=0, a_source=p in its own qi only.
                    // Dp then reproduces sigma(c1) in all ordered retained limbs.
                    let base = if component == 1 && source == target { q[3]%q[target] } else { 0 };
                    rows[at] = base;
                    if errors {
                        if source == 0 && component == 0 {
                            // This entire polynomial is reserved for an
                            // injective marker: odd g in [1,2N) -> floor(g/2).
                            // Direction 1/2 gives value 2/3, nonzero even for
                            // smallest fixture q=17. No g%small-prime collision.
                            assert!(g & 1 == 1 && g < 2*n as u64);
                            assert!((1..=2).contains(&direction));
                            rows[at+g as usize/2] = direction+1;
                            continue;
                        }
                        let error = BigInt::from((source+1)*(component+1))
                            * BigInt::from(direction+1);
                        let error = if source%2 == 0 { error } else { -error };
                        // Small nonzero public row errors in both components,
                        // all four limbs. Also constant error for N=1.
                        let j = usize::from(n > 1);
                        rows[at+j] = residue(BigInt::from(rows[at+j])+error, q[target]);
                    }
                }
            }
        }
        rows
    }

    /// Test-only replacement; fixture spectra still use independent direct DFT.
    fn replace_rows(&mut self, rows: Vec<Vec<u64>>) {
        assert_eq!(rows.len(), self.exponents.len());
        self.row_spectra = rows.iter().map(|row| (0..24).flat_map(|poly| {
            direct_dft(&row[poly*self.n..(poly+1)*self.n], self.roots[poly%4], self.q[poly%4])
        }).collect()).collect();
        self.row_coefficients = rows;
    }

    fn bindings(&self, stride: usize, count: usize) -> Vec<SwitchBinding<'_>> {
        InterleaveSchedule::bounded(self.n, stride, count, 8192).unwrap().steps()
            .filter_map(|step| match step {
                InterleaveStep::Trace { element, .. } | InterleaveStep::Merge { element, .. } => {
                    let at = self.exponents.iter().position(|&g| g == element as u64).unwrap();
                    Some(SwitchBinding { exponent: element as u64,
                        rows: RowSpectra { contexts: &self.targets, values: &self.row_spectra[at] } })
                }
                _ => None,
            }).collect()
    }

    /// Full canonical global Qp CRT, then floor((x+(p-1)/2)/p).
    fn drop_global(&self, residues: [u64; 4], mutation: Mutation) -> [u64; 3] {
        let sum = residues.into_iter().zip(&self.crt_weights)
            .fold(BigInt::from(0), |sum, (value, weight)| sum+BigInt::from(value)*weight);
        let x = ((sum % &self.crt_modulus)+&self.crt_modulus)%&self.crt_modulus;
        // Explicit alternative-rescale mutant: floor instead of nearest.
        // This does NOT claim coverage of all possible rescale substitutions.
        let offset = if mutation == Mutation::FloorRescale { 0 } else { (self.q[3]-1)/2 };
        let rounded = (x+BigInt::from(offset))/BigInt::from(self.q[3]);
        core::array::from_fn(|i| residue(rounded.clone(), self.q[i]))
    }

    fn switch(&self, value: &[u64], g: usize, mutation: Mutation) -> Vec<u64> {
        let n = self.n;
        let row_exponent = match mutation {
            Mutation::WrongRow { exponent, replacement } if exponent == g as u64 => replacement,
            _ => g as u64,
        };
        let at = self.exponents.iter().position(|&e| e == row_exponent).unwrap();
        let automorphism = if mutation == Mutation::InverseAutomorphism {
            residue(BigInt::from(g).modinv(&BigInt::from(2*n)).unwrap(), (2*n) as u64) as usize
        } else { g };
        let no_errors;
        let rows = if mutation == Mutation::NoErrors {
            no_errors = Self::rows(n, self.q, g as u64, self.direction, false);
            &no_errors
        } else { &self.row_coefficients[at] };
        let c0: Vec<_> = (0..3).map(|i| monomial(&value[i*n..(i+1)*n], automorphism, self.q[i])).collect();
        let digits: Vec<Vec<BigInt>> = (0..3).map(|i| {
            monomial(&value[(3+i)*n..(4+i)*n], automorphism, self.q[i]).into_iter().map(|digit| {
                let digit = BigInt::from(digit);
                if mutation == Mutation::Centered && digit > BigInt::from(self.q[i]/2) {
                    digit-BigInt::from(self.q[i])
                } else { digit }
            }).collect()
        }).collect();
        let mut out = vec![0; 6*n];
        for component in 0..2 {
            let row_component = if mutation == Mutation::Components { 1-component } else { component };
            let polys: Vec<Vec<u64>> = (0..4).map(|target| {
                let row_target = if mutation == Mutation::TargetOrder && target < 3 { (target+1)%3 } else { target };
                let mut sum = vec![BigInt::from(0); n];
                for source in 0..3 {
                    let row_source = if mutation == Mutation::SourceOrder { (source+1)%3 } else { source };
                    let start = ((row_source*2+row_component)*4+row_target)*n;
                    let product = schoolbook(&digits[source], &rows[start..start+n]);
                    for (sum, value) in sum.iter_mut().zip(product) { *sum += value; }
                }
                if mutation == Mutation::BeforeDrop && component == 0 && target < 3 {
                    for (sum, &c0) in sum.iter_mut().zip(&c0[target]) { *sum += BigInt::from(c0); }
                }
                sum.into_iter().map(|sum| residue(sum, self.q[target])).collect()
            }).collect();
            for j in 0..n {
                let dropped = self.drop_global(core::array::from_fn(|i| polys[i][j]), mutation);
                for limb in 0..3 {
                    let mut value = BigInt::from(dropped[limb]);
                    if component == 0 && mutation != Mutation::BeforeDrop { value += BigInt::from(c0[limb][j]); }
                    out[(component*3+limb)*n+j] = residue(value, self.q[limb]);
                }
            }
        }
        out
    }

    fn plus(&self, a: &[u64], b: &[u64], subtract: bool) -> Vec<u64> {
        a.iter().zip(b).enumerate().map(|(at, (&a, &b))| {
            let b = if subtract { -BigInt::from(b) } else { BigInt::from(b) };
            residue(BigInt::from(a)+b, self.q[(at/self.n)%3])
        }).collect()
    }

    fn shift(&self, value: &[u64], h: usize) -> Vec<u64> {
        let n = self.n;
        let mut out = vec![0; 6*n];
        for poly in 0..6 {
            for j in 0..n {
                let degree = j+h;
                let value = BigInt::from(value[poly*n+j]);
                let signed = if degree/n%2 == 0 { value } else { -value };
                out[poly*n+degree%n] = residue(signed, self.q[poly%3]);
            }
        }
        out
    }

    /// Independent recursion, not shared InterleaveSchedule iteration.
    fn oracle(&self, stride: usize, count: usize, mutation: Mutation) -> Vec<u64> {
        fn tree(f: &Fixture, lanes: &[Vec<u64>], lane: usize, h: usize, mutation: Mutation) -> Vec<u64> {
            if h == lanes.len() { return lanes[lane].clone(); }
            let left = tree(f, lanes, lane, 2*h, mutation);
            let right = f.shift(&tree(f, lanes, lane+h, 2*h, mutation), h);
            let difference = f.plus(&left, &right, true);
            let switched = f.switch(&difference, f.n/h+1, mutation);
            f.plus(&f.plus(&left, &right, false), &switched, false)
        }
        let padded = count.next_power_of_two();
        let mut lanes = vec![vec![0; 6*self.n]; padded];
        for lane in 0..count {
            for at in 0..6*self.n {
                let q = self.q[(at/self.n)%3];
                let inverse = if mutation == Mutation::OmitNormalization { BigInt::from(1) }
                    else { BigInt::from(stride).modinv(&BigInt::from(q)).unwrap() };
                lanes[lane][at] = residue(BigInt::from(self.input[lane*6*self.n+at])*inverse, q);
            }
        }
        for lane in &mut lanes {
            let mut h = padded;
            let mut levels = Vec::new();
            while h < stride {
                levels.push(h);
                h *= 2;
            }
            if mutation == Mutation::ReverseTraceLevels { levels.reverse(); }
            for h in levels {
                let element = if mutation == Mutation::WrongTraceLevel { self.n/padded+1 }
                    else { self.n/h+1 };
                *lane = self.plus(lane, &self.switch(lane, element, mutation), false);
            }
        }
        tree(self, &lanes, 0, 1, mutation)
    }
}

// Purpose-sized independent N8 phase arithmetic. No candidate/schedule/recursive
// oracle supplies expected components, digits, remainders, phases or errors.
type PhasePair = [Vec<BigInt>; 2];

fn phase_poly(values: &[i64]) -> Vec<BigInt> {
    assert!(values.len() <= 8);
    let mut out = vec![BigInt::from(0); 8];
    for (out, &value) in out.iter_mut().zip(values) { *out = BigInt::from(value); }
    out
}

fn phase_pair(c0: &[i64], c1: &[i64]) -> PhasePair {
    [phase_poly(c0), phase_poly(c1)]
}

fn phase_add(a: &[BigInt], b: &[BigInt], subtract: bool) -> Vec<BigInt> {
    assert_eq!(a.len(), 8);
    assert_eq!(b.len(), 8);
    a.iter().zip(b).map(|(a, b)| if subtract { a-b } else { a+b }).collect()
}

fn phase_scale(a: &[BigInt], scalar: &BigInt) -> Vec<BigInt> {
    a.iter().map(|a| a*scalar).collect()
}

fn phase_sigma(a: &[BigInt], g: usize) -> Vec<BigInt> {
    let mut out = phase_poly(&[]);
    for (j, value) in a.iter().enumerate() {
        let degree = j*g;
        if degree/8%2 == 0 { out[degree%8] += value; }
        else { out[degree%8] -= value; }
    }
    out
}

fn phase_x(a: &[BigInt]) -> Vec<BigInt> {
    schoolbook(a, &[0, 1, 0, 0, 0, 0, 0, 0])
}

fn phase_pair_add(a: &PhasePair, b: &PhasePair, subtract: bool) -> PhasePair {
    core::array::from_fn(|i| phase_add(&a[i], &b[i], subtract))
}

fn phase_value(c: &PhasePair) -> Vec<BigInt> {
    phase_add(&c[0], &schoolbook(&c[1], &[1, 1, 0, 0, 0, 0, 0, 0]), false)
}

fn phase_encode(f: &Fixture, c: &PhasePair) -> Vec<u64> {
    (0..6).flat_map(|poly| c[poly/3].iter().map(move |v| residue(v.clone(), f.q[poly%3]))).collect()
}

fn phase_linf(a: &[BigInt]) -> BigInt {
    a.iter().map(|v| if *v < BigInt::from(0) { -v } else { v.clone() }).max().unwrap()
}

fn phase_l1(a: &[BigInt]) -> BigInt {
    a.iter().map(|v| if *v < BigInt::from(0) { -v } else { v.clone() }).sum()
}

fn phase_assert_law(f: &Fixture, c: &PhasePair, ideal: &[BigInt], h: &[BigInt]) {
    let phase = phase_value(c);
    let expected = phase_add(ideal, h, false);
    for &q in &f.q[..3] {
        for j in 0..8 { assert_eq!(residue(phase[j].clone(), q), residue(expected[j].clone(), q)); }
    }
}

fn phase_fixture(error: i64) -> Fixture {
    assert!(error == 0 || error == 65);
    let mut f = Fixture::new(8, 4, 2, [17, 97, 113, 193], 1, false);
    assert_eq!(f.exponents, [5, 9]); // discovery is only a separate constraint
    for &q in &f.q {
        let mut divisor = 2;
        while divisor*divisor <= q {
            assert_ne!(q%divisor, 0, "tiny literal must be prime");
            divisor += 1;
        }
        assert_eq!(q%16, 1);
    }
    f.roots = f.q.map(|q| (2..q).find(|&psi| {
        power(psi, 8, q) == q-1 && power(psi, 16, q) == 1
    }).unwrap());
    for (i, &psi) in f.roots.iter().enumerate() {
        let q = f.q[i];
        assert_eq!(power(psi, 8, q), q-1);
        assert_eq!(power(psi, 16, q), 1);
        assert!((2..psi).all(|r| power(r, 8, q) != q-1 || power(r, 16, q) != 1));
    }
    f.targets = core::array::from_fn(|i| NegacyclicNtt::new(
        CoefficientRing::bounded(8, f.q[i], DegreeBound::Public8192).unwrap(), f.roots[i],
    ).unwrap());
    f.sources = core::array::from_fn(|i| f.targets[i].clone());
    assert_eq!(f.crt_modulus, BigInt::from(35_963_041));
    assert_eq!(17_u64*97*113, 186_337);
    let projectors = [43_844_i64, 97_971, 44_523];
    assert_eq!(projectors.iter().sum::<i64>(), 186_338);
    for (i, &projector) in projectors.iter().enumerate() {
        for j in 0..3 { assert_eq!(residue(BigInt::from(projector), f.q[j]), u64::from(i == j)); }
    }
    for i in 0..4 {
        for j in 0..4 { assert_eq!(residue(f.crt_weights[i].clone(), f.q[j]), u64::from(i == j)); }
    }
    let s = phase_poly(&[1, 1]);
    let rows = [5, 9].into_iter().map(|g| {
        let image = phase_sigma(&s, g);
        assert_eq!(image, if g == 5 { phase_poly(&[1, 0, 0, 0, 0, 1]) } else { phase_poly(&[1, -1]) });
        let mut rows = vec![0; 24*8];
        for (source, &projector) in projectors.iter().enumerate() {
            let rhs = phase_add(&phase_scale(&image, &BigInt::from(193*projector)), &phase_poly(&[error]), false);
            let b = phase_add(&rhs, &s, true);
            let a = phase_poly(&[1]);
            for target in 0..4 {
                let q = f.q[target];
                let b_limb: Vec<_> = b.iter().map(|v| residue(v.clone(), q)).collect();
                let a_limb: Vec<_> = a.iter().map(|v| residue(v.clone(), q)).collect();
                // Verify the actual coefficient row in ALL FOUR limbs, BEFORE
                // replace_rows performs any spectral encoding.
                let lhs = phase_add(&b_limb.iter().copied().map(BigInt::from).collect::<Vec<_>>(),
                    &schoolbook(&s, &a_limb), false);
                for j in 0..8 { assert_eq!(residue(lhs[j].clone(), q), residue(rhs[j].clone(), q)); }
                let at = (source*2*4+target)*8;
                rows[at..at+8].copy_from_slice(&b_limb);
                let at = ((source*2+1)*4+target)*8;
                rows[at..at+8].copy_from_slice(&a_limb);
            }
        }
        rows
    }).collect();
    f.replace_rows(rows); // no scaffold selector/marker rows survive
    f.input = [phase_pair(&[0, -4, 4], &[4]), phase_pair(&[4, -4, 4], &[4])]
        .iter().flat_map(|c| phase_encode(&f, c)).collect();
    f
}

struct PhaseWitness {
    components: PhasePair,
    digits: [Vec<BigInt>; 3],
    v: PhasePair,
    r: PhasePair,
    dropped: PhasePair,
    e: Vec<BigInt>,
    numerator: Vec<BigInt>,
    xi: Vec<BigInt>,
}

fn phase_switch_witness(f: &Fixture, c: &PhasePair, g: usize, error: i64) -> PhaseWitness {
    assert!(g == 5 || g == 9);
    let row = &f.row_coefficients[if g == 5 { 0 } else { 1 }];
    // Encode current lifts per qi BEFORE direct substitution and unsigned digit
    // extraction. Earlier errors are carried in c, not replaced by ideal input.
    let digits: [Vec<BigInt>; 3] = core::array::from_fn(|i| {
        let encoded: Vec<_> = c[1].iter().map(|v| BigInt::from(residue(v.clone(), f.q[i]))).collect();
        phase_sigma(&encoded, g).into_iter().map(|v| BigInt::from(residue(v, f.q[i]))).collect()
    });
    let contracted: Vec<Vec<u64>> = (0..8).map(|poly| {
        let component = poly/4;
        let target = poly%4;
        let mut sum = phase_poly(&[]);
        for source in 0..3 {
            let at = ((source*2+component)*4+target)*8;
            sum = phase_add(&sum, &schoolbook(&digits[source], &row[at..at+8]), false);
        }
        sum.into_iter().map(|v| residue(v, f.q[target])).collect()
    }).collect();
    let mut v: PhasePair = [phase_poly(&[]), phase_poly(&[])];
    let mut r = v.clone();
    let mut dropped = v.clone();
    let p = BigInt::from(193);
    for component in 0..2 {
        for j in 0..8 {
            let limbs = core::array::from_fn(|i| contracted[component*4+i][j]);
            let sum: BigInt = limbs.iter().zip(&f.crt_weights).map(|(&v, w)| BigInt::from(v)*w).sum();
            let canonical = ((sum % &f.crt_modulus)+&f.crt_modulus)%&f.crt_modulus;
            let remainder = residue(canonical.clone(), 193);
            let centered = BigInt::from(remainder) - if remainder > 96 { p.clone() } else { BigInt::from(0) };
            assert!(centered >= BigInt::from(-96) && centered <= BigInt::from(96));
            let numerator = &canonical-&centered;
            assert_eq!(&numerator%&p, BigInt::from(0));
            let quotient = numerator/&p;
            let rounded = (&canonical+BigInt::from(96))/&p;
            let existing = f.drop_global(limbs, Mutation::None);
            for i in 0..3 {
                assert_eq!(residue(quotient.clone(), f.q[i]), residue(rounded.clone(), f.q[i]));
                assert_eq!(residue(quotient.clone(), f.q[i]), existing[i]);
            }
            // Center the quotient modulo Q only to choose small component lifts;
            // this is NOT a claim that global CRT/intermediate values never wrap.
            let q = BigInt::from(186_337);
            let canonical_q = ((quotient%&q)+&q)%&q;
            dropped[component][j] = if canonical_q > BigInt::from(93_168) { canonical_q-&q } else { canonical_q };
            v[component][j] = canonical;
            r[component][j] = centered;
        }
    }
    let mut e = phase_poly(&[]);
    for digit in &digits { e = phase_add(&e, &phase_scale(digit, &BigInt::from(error)), false); }
    let numerator = phase_add(&phase_add(&e, &r[0], true),
        &schoolbook(&r[1], &[1, 1, 0, 0, 0, 0, 0, 0]), true);
    let xi: Vec<_> = numerator.iter().map(|v| {
        assert_eq!(v%&p, BigInt::from(0));
        v/&p
    }).collect();
    let components = [phase_add(&dropped[0], &phase_sigma(&c[0], g), false), dropped[1].clone()];
    let image = phase_sigma(&phase_value(c), g);
    phase_assert_law(f, &components, &image, &xi);
    let secret_l1 = phase_l1(&phase_poly(&[1, 1]));
    assert_eq!(secret_l1, BigInt::from(2));
    let rhs: BigInt = digits.iter().map(|d| phase_l1(d)*BigInt::from(error)).sum::<BigInt>()
        + BigInt::from(96)*(BigInt::from(1)+secret_l1);
    assert!(BigInt::from(193)*phase_linf(&xi) <= rhs);
    if error == 65 { assert_eq!(rhs, BigInt::from(if g == 5 { 483 } else { 678 })); }
    PhaseWitness { components, digits, v, r, dropped, e, numerator, xi }
}

// Candidate adapter ONLY: direct DFT encodes inputs; no output is used to
// manufacture any expected value in phase_switch_witness or the explicit path.
fn phase_actual_switch(f: &Fixture, c: &PhasePair, g: usize) -> Vec<u64> {
    let encoded = phase_encode(f, c);
    let spectra: Vec<_> = (0..6).flat_map(|poly| {
        direct_dft(&encoded[poly*8..(poly+1)*8], f.roots[poly%3], f.q[poly%3])
    }).collect();
    let contraction = RnsContraction::new(&f.sources, &f.targets).unwrap();
    let storage = contraction.galois_storage().unwrap();
    let work = contraction.galois_work().unwrap();
    let mut out = vec![u64::MAX; storage.output_words];
    let mut scratch = vec![u64::MAX; storage.scratch_words];
    assert!(g == 5 || g == 9);
    contraction.galois_contract_drop_into(
        ComponentSpectra { contexts: &f.sources, values: &spectra },
        RowSpectra { contexts: &f.targets, values: &f.row_spectra[if g == 5 { 0 } else { 1 }] },
        g as u64, &mut out, &mut scratch, work,
    ).unwrap();
    assert_eq!(out.len(), 48);
    out
}

fn phase_assert_witness(w: &PhaseWitness, merge: bool, error: i64) {
    let noisy = error == 65;
    let digit = if merge { phase_poly(&[1, 1]) } else { phase_poly(&[1]) };
    for d in &w.digits { assert_eq!(*d, digit); }
    assert_eq!(w.e, phase_scale(&digit, &BigInt::from(3*error)));
    let m = 35_963_041;
    let v0 = if merge {
        if noisy { phase_poly(&[385, 189, m-196]) } else { phase_poly(&[190, m-6, m-196]) }
    } else if noisy { phase_poly(&[385, m-3, 0, 0, 0, 193]) }
    else { phase_poly(&[190, m-3, 0, 0, 0, 193]) };
    assert_eq!(w.v[0], v0);
    assert_eq!(w.v[1], phase_scale(&digit, &BigInt::from(3)));
    let r0 = if merge {
        if noisy { phase_poly(&[-1, -4, -3]) } else { phase_poly(&[-3, -6, -3]) }
    } else if noisy { phase_poly(&[-1, -3]) } else { phase_poly(&[-3, -3]) };
    assert_eq!(w.r[0], r0);
    assert_eq!(w.r[1], phase_scale(&digit, &BigInt::from(3)));
    let xi = if noisy { digit } else { phase_poly(&[]) };
    assert_eq!(w.numerator, phase_scale(&xi, &BigInt::from(193)));
    assert_eq!(w.xi, xi);
    let drop0 = if merge {
        if noisy { phase_poly(&[2, 1, -1]) } else { phase_poly(&[1, 0, -1]) }
    } else if noisy { phase_poly(&[2, 0, 0, 0, 0, 1]) }
    else { phase_poly(&[1, 0, 0, 0, 0, 1]) };
    assert_eq!(w.dropped, [drop0, phase_poly(&[])]);
}

struct PhasePath {
    components: PhasePair,
    ideal: Vec<BigInt>,
    h: Vec<BigInt>,
    prior: [Vec<BigInt>; 2],
    merge_xi: Vec<BigInt>,
}

fn phase_three_switch_path(f: &Fixture, error: i64) -> PhasePath {
    let noisy = error == 65;
    let raw = [phase_pair(&[0, -4, 4], &[4]), phase_pair(&[4, -4, 4], &[4])];
    let normalized = [phase_pair(&[0, -1, 1], &[1]), phase_pair(&[1, -1, 1], &[1])];
    let mut ideals = [phase_poly(&[1, 0, 1]), phase_poly(&[2, 0, 1])];
    let mut h = [phase_poly(&[]), phase_poly(&[])];
    let mut traced = normalized.clone();
    for lane in 0..2 {
        assert_eq!(&f.input[lane*48..(lane+1)*48], phase_encode(f, &raw[lane]));
        assert_eq!(phase_value(&raw[lane]), if lane == 0 { phase_poly(&[4, 0, 4]) } else { phase_poly(&[8, 0, 4]) });
        // Modular inversion is independent, not an assumption of real division.
        let encoded = phase_encode(f, &raw[lane]);
        let normalized_words: Vec<_> = encoded.iter().enumerate().map(|(at, &v)| {
            let q = f.q[(at/8)%3];
            let inverse = BigInt::from(4).modinv(&BigInt::from(q)).unwrap();
            assert_eq!(residue(BigInt::from(4)*&inverse, q), 1);
            residue(BigInt::from(v)*inverse, q)
        }).collect();
        assert_eq!(normalized_words, phase_encode(f, &normalized[lane]));
        phase_assert_law(f, &normalized[lane], &ideals[lane], &h[lane]);
        let w = phase_switch_witness(f, &normalized[lane], 5, error);
        phase_assert_witness(&w, false, error);
        let literal = phase_pair(&[if noisy { 2+lane as i64 } else { 1+lane as i64 }, 0, -1], &[]);
        assert_eq!(w.components, literal);
        let candidate = phase_actual_switch(f, &normalized[lane], 5);
        assert_eq!(candidate, phase_encode(f, &w.components));
        assert_eq!(candidate, phase_encode(f, &literal)); // all 48 words
        traced[lane] = phase_pair_add(&normalized[lane], &w.components, false);
        ideals[lane] = phase_add(&ideals[lane], &phase_sigma(&ideals[lane], 5), false);
        h[lane] = phase_add(&phase_add(&h[lane], &phase_sigma(&h[lane], 5), false), &w.xi, false);
        assert_eq!(traced[lane], phase_pair(&[if noisy { 2+2*lane as i64 } else { 1+2*lane as i64 }, -1], &[1]));
        assert_eq!(ideals[lane], phase_poly(&[2+2*lane as i64]));
        assert_eq!(h[lane], phase_poly(&[if noisy { 1 } else { 0 }]));
        assert_eq!(phase_linf(&w.xi), BigInt::from(if noisy { 1 } else { 0 }));
        assert_eq!(phase_linf(&h[lane]), BigInt::from(if noisy { 1 } else { 0 }));
        phase_assert_law(f, &traced[lane], &ideals[lane], &h[lane]);
        assert_eq!(phase_encode(f, &traced[lane]), f.plus(&normalized_words, &candidate, false));
    }
    let shifted = core::array::from_fn(|i| phase_x(&traced[1][i]));
    let difference = phase_pair_add(&traced[0], &shifted, true);
    assert_eq!(difference, if noisy { phase_pair(&[2, -5, 1], &[1, -1]) }
        else { phase_pair(&[1, -4, 1], &[1, -1]) });
    let p_difference = phase_add(&ideals[0], &phase_x(&ideals[1]), true);
    let h_difference = phase_add(&h[0], &phase_x(&h[1]), true);
    assert_eq!(p_difference, phase_poly(&[2, -4]));
    assert_eq!(h_difference, if noisy { phase_poly(&[1, -1]) } else { phase_poly(&[]) });
    phase_assert_law(f, &difference, &p_difference, &h_difference);
    assert_eq!(phase_encode(f, &difference), f.plus(&phase_encode(f, &traced[0]),
        &f.shift(&phase_encode(f, &traced[1]), 1), true));
    let w = phase_switch_witness(f, &difference, 9, error);
    phase_assert_witness(&w, true, error);
    let switch_literal = if noisy { phase_pair(&[4, 6], &[]) } else { phase_pair(&[2, 4], &[]) };
    assert_eq!(w.components, switch_literal);
    let candidate = phase_actual_switch(f, &difference, 9);
    assert_eq!(candidate, phase_encode(f, &w.components));
    assert_eq!(candidate, phase_encode(f, &switch_literal));
    let components = phase_pair_add(&phase_pair_add(&traced[0], &shifted, false), &w.components, false);
    let ideal = phase_add(&phase_add(&ideals[0], &phase_x(&ideals[1]), false), &phase_sigma(&p_difference, 9), false);
    let final_h = phase_add(&phase_add(&h[0], &phase_x(&h[1]), false),
        &phase_add(&phase_sigma(&h_difference, 9), &w.xi, false), false);
    assert_eq!(components, if noisy { phase_pair(&[6, 9, -1], &[1, 1]) }
        else { phase_pair(&[3, 6, -1], &[1, 1]) });
    assert_eq!(ideal, phase_poly(&[4, 8]));
    assert_eq!(final_h, if noisy { phase_poly(&[3, 3]) } else { phase_poly(&[]) });
    phase_assert_law(f, &components, &ideal, &final_h);
    assert_eq!(phase_linf(&w.xi), BigInt::from(if noisy { 1 } else { 0 }));
    let recurrence_bound = BigInt::from(2)*(phase_linf(&h[0])+phase_linf(&h[1]))+phase_linf(&w.xi);
    assert_eq!(recurrence_bound, BigInt::from(if noisy { 5 } else { 0 }));
    assert!(phase_linf(&final_h) <= recurrence_bound);
    // Literal binding sequence constrains wrapper only; no expectation came
    // from these bindings. Stage additions above are NOT wrapper observations.
    assert_eq!(f.bindings(4, 2).iter().map(|b| b.exponent).collect::<Vec<_>>(), [5, 5, 9]);
    assert_eq!(actual(f, 4, 2), phase_encode(f, &components));
    PhasePath { components, ideal, h: final_h, prior: h, merge_xi: w.xi }
}

fn phase_actual_value(f: &Fixture, words: &[u64]) -> [Vec<u64>; 3] {
    assert_eq!(words.len(), 48);
    core::array::from_fn(|i| {
        let c0: Vec<_> = words[i*8..(i+1)*8].iter().copied().map(BigInt::from).collect();
        let c1: Vec<_> = words[(3+i)*8..(4+i)*8].iter().copied().map(BigInt::from).collect();
        phase_value(&[c0, c1]).into_iter().map(|v| residue(v, f.q[i])).collect()
    })
}

#[test]
fn source_related_n8_three_switch_stage_components_and_recursive_phase() {
    let f = phase_fixture(65);
    let path = phase_three_switch_path(&f, 65);
    assert_eq!(phase_value(&path.components), phase_poly(&[7, 11]));
    assert_eq!(phase_linf(&path.h), BigInt::from(3));
    let words = actual(&f, 4, 2); // complete actual TracePlan::apply
    assert_eq!(words, phase_encode(&f, &phase_pair(&[6, 9, -1], &[1, 1])));
    let phase = phase_actual_value(&f, &words);
    for i in 0..3 {
        assert_eq!(phase[i], phase_poly(&[7, 11]).into_iter().map(|v| residue(v, f.q[i])).collect::<Vec<_>>());
    }
    // Reconstruct ACTUAL phase over Q, then center the ideal difference. Only
    // this final difference has a no-wrap assertion; not each qi or global V.
    let projectors = [43_844, 97_971, 44_523];
    let mut centered = phase_poly(&[]);
    for j in 0..8 {
        let reconstructed: BigInt = (0..3).map(|i| BigInt::from(phase[i][j])*BigInt::from(projectors[i])).sum();
        let difference = residue(reconstructed-&path.ideal[j], 186_337);
        centered[j] = BigInt::from(difference)-if difference > 93_168 { BigInt::from(186_337) } else { BigInt::from(0) };
    }
    assert_eq!(centered, phase_poly(&[3, 3]));
    assert!(BigInt::from(2)*phase_linf(&centered) < BigInt::from(186_337));
}

#[test]
fn source_related_n8_frozen_local_error_controls_and_noise_free_trace() {
    let f = phase_fixture(65);
    let path = phase_three_switch_path(&f, 65);
    assert_eq!(path.prior, [phase_poly(&[1]), phase_poly(&[1])]);
    assert_eq!(path.merge_xi, phase_poly(&[1, 1]));
    let words = actual(&f, 4, 2);
    let actual_phase = phase_actual_value(&f, &words);
    let base = phase_add(&path.prior[0], &phase_x(&path.prior[1]), false);
    let branch = phase_sigma(&phase_add(&path.prior[0], &phase_x(&path.prior[1]), true), 9);
    let propagated = phase_add(&base, &branch, false);
    assert_eq!(propagated, phase_poly(&[2, 2]));
    // ORACLE-ONLY recurrence mutations, NOT mutated candidate executions.
    // Freeze actual independent local xi. Earlier b-only errors leave c1,
    // later digits and contracted remainders unchanged in this fixture.
    // The last two bad predictions coincide: separately discriminate against
    // actual output, NOT mutually identifiable, nor coverage of all sign/digit
    // or profile faults.
    for i in 0..3 {
        let q = f.q[i];
        let correct = phase_add(&path.ideal, &path.h, false);
        let correct: Vec<_> = correct.into_iter().map(|v| residue(v, q)).collect();
        assert_eq!(actual_phase[i], correct);
        let drop_prior = path.merge_xi.clone();
        let inverse_two = BigInt::from(2).modinv(&BigInt::from(q)).unwrap();
        assert_eq!(residue(BigInt::from(2)*&inverse_two, q), 1);
        let halve_prior = phase_add(&phase_scale(&propagated, &inverse_two), &path.merge_xi, false);
        let omit_branch = phase_add(&base, &path.merge_xi, false);
        for (name, bad_h, literal_h, literal_phase, literal_difference) in [
            ("ORACLE-ONLY drop both prior errors", drop_prior, phase_poly(&[1, 1]), phase_poly(&[5, 9]), phase_poly(&[2, 2])),
            ("ORACLE-ONLY halve propagated prior errors", halve_prior, phase_poly(&[2, 2]), phase_poly(&[6, 10]), phase_poly(&[1, 1])),
            ("ORACLE-ONLY omit switched-difference branch", omit_branch, phase_poly(&[2, 2]), phase_poly(&[6, 10]), phase_poly(&[1, 1])),
        ] {
            let bad_phase = phase_add(&path.ideal, &bad_h, false);
            let mut difference = Vec::new();
            for j in 0..8 {
                assert_eq!(residue(bad_h[j].clone(), q), residue(literal_h[j].clone(), q), "{name}");
                let bad = residue(bad_phase[j].clone(), q);
                assert_eq!(bad, residue(literal_phase[j].clone(), q), "{name}");
                difference.push(residue(BigInt::from(actual_phase[i][j])-BigInt::from(bad), q));
            }
            assert_eq!(difference, literal_difference.into_iter().map(|v| residue(v, q)).collect::<Vec<_>>(), "{name}");
            assert!(difference.iter().any(|&v| v != 0), "{name}");
        }
    }
    // Separate genuinely related zero-e rows; NEVER Mutation::NoErrors (which
    // would reconstruct selector rows). Zero noise is not generally zero drop
    // error; this particular fixture independently proves zero local xi.
    let zero = phase_fixture(0);
    let zero_path = phase_three_switch_path(&zero, 0);
    assert_eq!(zero_path.h, phase_poly(&[]));
    assert_eq!(zero_path.merge_xi, phase_poly(&[]));
    assert_eq!(zero_path.components, phase_pair(&[3, 6, -1], &[1, 1]));
    assert_eq!(phase_value(&zero_path.components), phase_poly(&[4, 8]));
    let zero_words = actual(&zero, 4, 2);
    assert_eq!(zero_words, phase_encode(&zero, &zero_path.components));
    let zero_phase = phase_actual_value(&zero, &zero_words);
    for i in 0..3 {
        assert_eq!(zero_phase[i], zero_path.ideal.iter().map(|v| residue(v.clone(), zero.q[i])).collect::<Vec<_>>());
    }
}

fn assert_distinct_payloads(payloads: &[Vec<u64>]) {
    for i in 0..payloads.len() {
        for j in 0..i { assert_ne!(payloads[i], payloads[j], "payload collision {i}/{j}"); }
    }
}

fn actual(f: &Fixture, stride: usize, count: usize) -> Vec<u64> {
    let contraction = RnsContraction::new(&f.sources, &f.targets).unwrap();
    let plan = TracePlan::new(&contraction, stride, count).unwrap();
    let storage = plan.storage().unwrap();
    let bindings = f.bindings(stride, count);
    let mut out = vec![u64::MAX; storage.output_words+3];
    let mut scratch = vec![u64::MAX-1; storage.scratch_words+5];
    let pointer = out.as_ptr();
    plan.apply(Coefficients { contexts: &f.sources, values: &f.input }, &bindings,
        &mut out, &mut scratch, plan.work().unwrap()).unwrap();
    assert_eq!(pointer, out.as_ptr());
    assert_eq!(&out[storage.output_words..], &[u64::MAX; 3]);
    assert_eq!(&scratch[storage.scratch_words..], &[u64::MAX-1; 5]);
    out.truncate(storage.output_words);
    out
}

#[test]
fn recursive_global_crt_nonzero_errors_component_sign_digit_and_limb_witnesses() {
    let f = Fixture::new(8, 8, 3, [17, 97, 113, 193], 1, true);
    let expected = f.oracle(8, 3, Mutation::None);
    assert_eq!(actual(&f, 8, 3), expected);
    for mutation in [Mutation::Centered, Mutation::BeforeDrop, Mutation::Components,
        Mutation::SourceOrder, Mutation::TargetOrder, Mutation::NoErrors]
    {
        assert_ne!(f.oracle(8, 3, mutation), expected, "missing discriminating {mutation:?} fixture");
    }
}

#[test]
fn tiny_count_one_partial_padding_and_two_distinct_direction_contexts() {
    for (n, stride, count) in [(1, 1, 1), (2, 2, 1), (4, 4, 3), (8, 8, 6), (8, 4, 3)] {
        for direction in [1, 2] {
            let q = if direction == 1 { [17, 97, 113, 193] } else { [193, 113, 97, 17] };
            let f = Fixture::new(n, stride, count, q, direction, true);
            assert_eq!(actual(&f, stride, count), f.oracle(stride, count, Mutation::None));
        }
    }
}

#[test]
fn source_sized_row_payload_metadata_is_injective_without_ntt_or_ring_evaluation() {
    // AUTHOR ONLY / UNRUN: coefficient-payload construction and comparisons,
    // no NTT descriptors, primality claims, contraction or trace evaluation.
    let exponents = [257_u64, 513, 1025, 2049, 4097, 8193];
    let mut scheduled: Vec<_> = InterleaveSchedule::bounded(8192, 64, 6, 8192).unwrap()
        .steps().filter_map(|step| match step {
            InterleaveStep::Trace { element, .. } | InterleaveStep::Merge { element, .. } => Some(element as u64),
            _ => None,
        }).collect();
    scheduled.sort_unstable();
    scheduled.dedup();
    assert_eq!(scheduled, exponents);
    let primes = [562949952847873, 562949952798721, 562949952700417, 562949952274433];
    let mut both_directions = Vec::new();
    for direction in [1, 2] {
        let q = if direction == 1 { primes } else { [primes[2], primes[0], primes[1], primes[3]] };
        let payloads: Vec<_> = exponents.iter().map(|&g| Fixture::rows(8192, q, g, direction, true)).collect();
        assert_distinct_payloads(&payloads);
        for (&g, row) in exponents.iter().zip(&payloads) {
            assert_eq!(row[g as usize/2], direction+1);
            assert_eq!(row[..8192].iter().filter(|&&v| v != 0).count(), 1);
        }
        both_directions.extend(payloads);
    }
    // Different public payloads, NOT evidence of two genuine source keys.
    assert_distinct_payloads(&both_directions);
}

#[test]
fn hand_derived_nonideal_rows_separate_inverse_level_order_and_wrong_row_whole_trace() {
    for direction in [1, 2] {
        let q = if direction == 1 { [17, 97, 113, 193] } else { [113, 17, 97, 193] };
        let mut f = Fixture::new(8, 4, 1, q, direction, false);
        assert_eq!(f.exponents, [5, 9]);
        f.input.fill(0);
        for limb in 0..3 { f.input[(3+limb)*8+1] = 4; }
        let rows: Vec<_> = f.exponents.iter().map(|&g| {
            let mut row = vec![0; 24*8];
            // g9 switch rows zero. g5: b_i=p only in retained qi=i;
            // a_i=0. Therefore switch_g5(c0,c1)=(sigma(c0)+sigma(c1),0),
            // whereas switch_g9(c0,c1)=(sigma(c0),0), exactly after drop.
            if g == 5 {
                for source in 0..3 { row[((source*2)*4+source)*8] = q[3]%q[source]; }
            }
            row
        }).collect();
        assert_distinct_payloads(&rows);
        f.replace_rows(rows);
        assert_distinct_payloads(&f.row_spectra);

        // Normalize (0,4X)/4=(0,X). Correct g9 then g5 gives (X^5,X).
        // Inverse g5=13 mod16 instead gives (-X^5,X). Reversed levels give
        // X^5+sigma9(X^5)=0. Repeating level g9 or using its zero rows at
        // g5 also gives (0,X). These are explicit nonideal arithmetic rows;
        // ideal trace group projectors need NOT distinguish inverse/order.
        let mut correct_literal = vec![0; 6*8];
        let mut inverse_literal = vec![0; 6*8];
        let mut zero_c0_literal = vec![0; 6*8];
        for limb in 0..3 {
            correct_literal[limb*8+5] = 1;
            inverse_literal[limb*8+5] = q[limb]-1;
            for result in [&mut correct_literal, &mut inverse_literal, &mut zero_c0_literal] {
                result[(3+limb)*8+1] = 1;
            }
        }
        let expected = f.oracle(4, 1, Mutation::None);
        assert_eq!(expected, correct_literal);
        assert_eq!(actual(&f, 4, 1), expected);
        let inverse = f.oracle(4, 1, Mutation::InverseAutomorphism);
        assert_eq!(inverse, inverse_literal);
        assert_ne!(inverse, expected);
        for mutant in [Mutation::ReverseTraceLevels, Mutation::WrongTraceLevel,
            Mutation::WrongRow { exponent: 5, replacement: 9 }]
        {
            let wrong = f.oracle(4, 1, mutant);
            assert_eq!(wrong, zero_c0_literal);
            assert_ne!(wrong, expected, "hand witness failed for {mutant:?}");
        }
        // Labels/context stay correct while payload is substituted. API is
        // public arithmetic, so it accepts canonical synthetic wrong rows;
        // its DIFFERENT result must match independent wrong-row oracle.
        let wrong_expected = f.oracle(4, 1, Mutation::WrongRow { exponent: 5, replacement: 9 });
        let mut wrong_rows = f.row_coefficients.clone();
        wrong_rows[0] = wrong_rows[1].clone();
        f.replace_rows(wrong_rows);
        assert_eq!(actual(&f, 4, 1), wrong_expected);
        assert_ne!(wrong_expected, expected);
    }
}

#[test]
fn constant_zero_c1_whole_trace_pins_omitted_inverse_stride_counterexample() {
    let mut f = Fixture::new(8, 4, 1, [17, 97, 113, 193], 1, false);
    f.input.fill(0);
    for limb in 0..3 { f.input[limb*8] = 1; }
    // Two constant-doubling traces: 4*(1/4)=1, versus omitted scale ->4.
    let mut correct_literal = vec![0; 6*8];
    let mut omitted_literal = vec![0; 6*8];
    for limb in 0..3 { correct_literal[limb*8] = 1; omitted_literal[limb*8] = 4; }
    let expected = f.oracle(4, 1, Mutation::None);
    let omitted = f.oracle(4, 1, Mutation::OmitNormalization);
    assert_eq!(expected, correct_literal);
    assert_eq!(omitted, omitted_literal);
    assert_ne!(omitted, expected);
    assert_eq!(actual(&f, 4, 1), expected);
}

#[test]
fn half_boundary_whole_trace_pins_floor_instead_of_rounded_special_limb_rescale() {
    let mut f = Fixture::new(2, 2, 1, [17, 97, 113, 193], 1, false);
    f.input.fill(0);
    for limb in 0..3 { f.input[(3+limb)*2] = 2; }
    let mut row = vec![0; 24*2];
    // After normalization c1=1, only digit row0 contributes x=97 globally.
    // p=193: nearest floor((97+96)/193)=1, alternative floor(97/193)=0.
    // a=0; c1 remains 1 through the single complete trace, not a lone switch.
    for target in 0..4 { row[target*2] = 97%f.q[target]; }
    f.replace_rows(vec![row]);
    let mut correct_literal = vec![0; 6*2];
    let mut floor_literal = vec![0; 6*2];
    for limb in 0..3 {
        correct_literal[limb*2] = 1;
        correct_literal[(3+limb)*2] = 1;
        floor_literal[(3+limb)*2] = 1;
    }
    let expected = f.oracle(2, 1, Mutation::None);
    let floor = f.oracle(2, 1, Mutation::FloorRescale);
    assert_eq!(expected, correct_literal);
    assert_eq!(floor, floor_literal);
    assert_ne!(floor, expected);
    assert_eq!(actual(&f, 2, 1), expected);
    // Only this specified floor-rescale mutant is covered here. No blanket
    // coverage claim for CKKS rescale, rowwise drop, merge-level reorderings,
    // or recursively propagated source-secret phase/error laws.
}

#[test]
fn zero_error_selector_inverse_stride_both_components_and_rectangular_coordinates() {
    let mut f = Fixture::new(8, 8, 3, [17, 97, 113, 193], 1, false);
    f.input.fill(0);
    // Each lane contains two contiguous result-column blocks, stride 8 here:
    // kept constant terms expose inverse-stride cancellation and lane placement.
    for lane in 0..3 {
        for poly in 0..6 { f.input[(lane*6+poly)*8] = (lane+poly+1) as u64; }
    }
    let out = actual(&f, 8, 3);
    assert_eq!(out, f.oracle(8, 3, Mutation::None));
    for poly in 0..6 {
        for j in 0..8 { assert_eq!(out[poly*8+j], if j < 3 { (j+poly+1) as u64 } else { 0 }); }
    }
    // Rectangular 3x2 ParseMat geometry, independent global coordinate rule.
    let mut rectangular = Fixture::new(16, 8, 3, [97, 193, 257, 353], 1, false);
    rectangular.input.fill(0);
    for column in 0..3 {
        for poly in 0..6 {
            for row in 0..2 {
                rectangular.input[(column*6+poly)*16+row*8] = (10*row+column+poly+1) as u64;
            }
        }
    }
    let out = actual(&rectangular, 8, 3);
    assert_eq!(out, rectangular.oracle(8, 3, Mutation::None));
    for poly in 0..6 {
        for row in 0..2 {
            for column in 0..3 { assert_eq!(out[poly*16+row*8+column], (10*row+column+poly+1) as u64); }
        }
    }
}

#[test]
fn whole_call_exact_fit_every_positive_work_field_one_short_preserves_all_tails() {
    let f = Fixture::new(8, 8, 3, [17, 97, 113, 193], 1, true);
    let contraction = RnsContraction::new(&f.sources, &f.targets).unwrap();
    let plan = TracePlan::new(&contraction, 8, 3).unwrap();
    let st = plan.storage().unwrap();
    let work = plan.work().unwrap();
    let bindings = f.bindings(8, 3);
    let mut ceilings = Vec::new();
    for index in 0..92 {
        if work.galois[index] != 0 { let mut short = work; short.galois[index] -= 1; ceilings.push(short); }
    }
    for index in 0..10 {
        if work.forwards[index] != 0 { let mut short = work; short.forwards[index] -= 1; ceilings.push(short); }
        if work.wrapper[index] != 0 { let mut short = work; short.wrapper[index] -= 1; ceilings.push(short); }
    }
    for index in 0..8 {
        if work.preflight_construction[index] != 0 {
            let mut short = work; short.preflight_construction[index] -= 1; ceilings.push(short);
        }
    }
    let mut short = work; short.switches -= 1; ceilings.push(short);
    let mut short = work; short.control -= 1; ceilings.push(short);
    let mut short = work; short.indices -= 1; ceilings.push(short);
    for ceiling in ceilings {
        let mut out = vec![987; st.output_words+7];
        let mut scratch = vec![654; st.scratch_words+9];
        assert_eq!(plan.apply(Coefficients { contexts: &f.sources, values: &f.input },
            &bindings, &mut out, &mut scratch, ceiling), Err(TraceError::WorkLimit));
        assert!(out.iter().all(|&v| v == 987));
        assert!(scratch.iter().all(|&v| v == 654));
    }
    // Exact full-call workspace and quote succeed, without relying on tails.
    let mut out = vec![0; st.output_words];
    let mut scratch = vec![0; st.scratch_words];
    plan.apply(Coefficients { contexts: &f.sources, values: &f.input }, &bindings,
        &mut out, &mut scratch, work).unwrap();
    assert_eq!(out, f.oracle(8, 3, Mutation::None));
    let mut scratch = vec![654; st.scratch_words-1];
    out.fill(987);
    assert_eq!(plan.apply(Coefficients { contexts: &f.sources, values: &f.input }, &bindings,
        &mut out, &mut scratch, work), Err(TraceError::Length));
    assert!(out.iter().all(|&v| v == 987));
    assert!(scratch.iter().all(|&v| v == 654));
}

#[test]
fn cumulative_quote_pins_forward_transitions_and_nested_repeated_scans() {
    for (n, stride, count) in [(1, 1, 1), (4, 4, 3), (8, 8, 3), (16, 8, 3)] {
        let q = if n == 16 { [97, 193, 257, 353] } else { [17, 97, 113, 193] };
        let f = Fixture::new(n, stride, count, q, 1, true);
        let contraction = RnsContraction::new(&f.sources, &f.targets).unwrap();
        let plan = TracePlan::new(&contraction, stride, count).unwrap();
        let st = plan.storage().unwrap();
        let work = plan.work().unwrap();
        let p = count.next_power_of_two();
        let c = (p*((stride/p).trailing_zeros() as usize)+p-1) as u64;
        let n = n as u64;
        let e = u64::from(stride.trailing_zeros());
        let k = count as u64;
        let m = p as u64 - 1;
        let t = c - m;
        assert_eq!(work.control, 7765+52*e+7*e*e+117*k+66*k*n
            +634*c+132*c*n+10*c*e+34*t+96*m+96*m*n+314);
        assert_eq!(work.indices, 74+5*e+2*p as u64+36*k+18*k*n
            +74*c+36*c*n+14*m+84*m*n+9*t);
        if n == 1 {
            assert_eq!((c, p, k, e), (0, 1, 1, 0));
            assert_eq!((work.control, work.indices), (8262, 130));
        }
        let butterflies = (n/2)*u64::from(n.trailing_zeros());
        let transforms = 6*c;
        assert_eq!(work.switches, c);
        assert_eq!(work.forwards, [transforms*(n+2), transforms*n,
            transforms*(2*n+2*butterflies), transforms*(n/2),
            transforms*butterflies, transforms*butterflies,
            transforms*if n == 1 { 0 } else { 2*butterflies+2*n }, 0, 0, 0]);
        // Parent quote is retained C times, not replaced by a fresh single-call
        // ceiling. Representative endpoints pin flattening order independently.
        let nested = contraction.galois_work().unwrap();
        assert_eq!(work.galois[0], c*nested.composition.contraction.events.checks);
        assert_eq!(work.galois[1], c*nested.composition.contraction.events.residue_checks);
        assert_eq!(work.galois[10], c*nested.composition.contraction.coefficient_reductions);
        assert_eq!(work.galois[91], c*nested.extra.control.index_divisions);
        assert_eq!(work.wrapper[1], n*(6*count as u64+24*c));
        assert_eq!(st.scratch_words, (6*p+47)*n as usize);
        assert_eq!(st.total_words, (6*count+24*c as usize+6*p+53)*n as usize);
        assert_eq!(work.preflight_construction[0], nested.composition.extra.construction.checks);
    }
}

#[test]
fn final_binding_corruption_context_exponent_lengths_and_input_refuse_before_normalization() {
    let f = Fixture::new(8, 8, 3, [17, 97, 113, 193], 1, true);
    let other = Fixture::new(8, 8, 3, [193, 113, 97, 17], 2, true);
    let contraction = RnsContraction::new(&f.sources, &f.targets).unwrap();
    let plan = TracePlan::new(&contraction, 8, 3).unwrap();
    let st = plan.storage().unwrap();
    let last_row = f.row_spectra.last().unwrap();
    let mut corrupted = last_row.clone();
    *corrupted.last_mut().unwrap() = f.q[3];
    let mut bad_input = f.input.clone();
    *bad_input.last_mut().unwrap() = f.q[2];
    for case in 0..9 {
        let mut bindings = f.bindings(8, 3);
        let last = bindings.last_mut().unwrap();
        let mut input = Coefficients { contexts: &f.sources, values: &f.input };
        let mut out = vec![987; st.output_words+7];
        let mut scratch = vec![654; st.scratch_words+9];
        let expected = match case {
            0 => { last.rows.values = &corrupted; TraceError::NonCanonical }
            1 => { last.rows.contexts = &other.targets; TraceError::Context }
            2 => { last.exponent = 1; TraceError::Exponent }
            3 => { last.rows.values = &last_row[..last_row.len()-1]; TraceError::Length }
            4 => { input.values = &bad_input; TraceError::NonCanonical }
            5 => { input.contexts = &other.sources; TraceError::Context }
            6 => { input.values = &f.input[..f.input.len()-1]; TraceError::Length }
            7 => { bindings.pop(); TraceError::Length }
            _ => { out.truncate(st.output_words-1); TraceError::Length }
        };
        assert_eq!(plan.apply(input, &bindings, &mut out, &mut scratch, plan.work().unwrap()), Err(expected));
        assert!(out.iter().all(|&v| v == 987));
        assert!(scratch.iter().all(|&v| v == 654));
    }
}

#[test]
fn checked_quote_capacity_and_source_sized_geometry_without_execution() {
    // Exercise real checked helpers, no giant allocations or invalid NTT plan.
    assert_eq!(super::olt_trace::times(u64::MAX, 2), Err(TraceError::Capacity));
    assert_eq!(super::olt_trace::add(u64::MAX, 1), Err(TraceError::Capacity));
    assert_eq!(super::olt_trace::usize_words(usize::MAX, 2), Err(TraceError::Capacity));
    assert_eq!(super::olt_trace::usize_sum(usize::MAX, 1), Err(TraceError::Capacity));
    let mut total = 0;
    assert_eq!(super::olt_trace::bump(&mut total, u128::MAX, 2), Err(TraceError::Capacity));
    assert_eq!(total, 0);
    total = u128::MAX;
    assert_eq!(super::olt_trace::bump(&mut total, 1, 1), Err(TraceError::Capacity));
    assert_eq!(total, u128::MAX);
    let schedule = InterleaveSchedule::bounded(8192, 64, 6, 8192).unwrap();
    let exponents: Vec<_> = schedule.steps().filter_map(|step| match step {
        InterleaveStep::Trace { element, .. } | InterleaveStep::Merge { element, .. } => Some(element),
        _ => None,
    }).collect();
    assert_eq!(exponents.len(), 31);
    assert_eq!(&exponents[..24], &[1025, 513, 257].repeat(8));
    assert_eq!(&exponents[24..], &[2049, 2049, 2049, 2049, 4097, 4097, 8193]);
    assert_eq!(881usize*8192*8, 57_737_216);
}

#[test]
fn scalar_no_switch_fixed_wrapper_costs_one_short_preserve_full_buffers() {
    let f = Fixture::new(1, 1, 1, [17, 97, 113, 193], 1, true);
    let contraction = RnsContraction::new(&f.sources, &f.targets).unwrap();
    let plan = TracePlan::new(&contraction, 1, 1).unwrap();
    let st = plan.storage().unwrap();
    let work = plan.work().unwrap();
    assert_eq!((st.switches, st.padded), (0, 1));
    // Independently: 7765+117+66+314; 74+2+36+18.
    assert_eq!((work.control, work.indices), (8262, 130));
    let mut control_short = work;
    control_short.control -= 1;
    let mut indices_short = work;
    indices_short.indices -= 1;
    for ceiling in [control_short, indices_short] {
        let mut out = vec![987; st.output_words+7];
        let mut scratch = vec![654; st.scratch_words+9];
        let original_out = out.clone();
        let original_scratch = scratch.clone();
        assert_eq!(plan.apply(Coefficients { contexts: &f.sources, values: &f.input },
            &[], &mut out, &mut scratch, ceiling), Err(TraceError::WorkLimit));
        assert_eq!(out, original_out);
        assert_eq!(scratch, original_scratch);
    }
    assert_eq!(actual(&f, 1, 1), f.input);
}

#[test]
#[ignore = "unadopted N8192 public arithmetic fixture; Native HOLD, explicit later authorization required"]
fn public8192_six_padded_eight_stride64_both_directions_recursive_global_crt() {
    let primes = [562949952847873, 562949952798721, 562949952700417, 562949952274433];
    for direction in [1, 2] {
        let q = if direction == 1 { primes } else { [primes[2], primes[0], primes[1], primes[3]] };
        let f = Fixture::new(8192, 64, 6, q, direction, true);
        let contraction = RnsContraction::new(&f.sources, &f.targets).unwrap();
        let plan = TracePlan::new(&contraction, 64, 6).unwrap();
        let st = plan.storage().unwrap();
        assert_eq!(st.switches, 31);
        assert_eq!(st.padded, 8);
        assert_eq!(st.scratch_words, 95*8192);
        assert_eq!(st.total_words, 881*8192);
        assert_eq!(plan.work().unwrap().switches, 31);
        let mut bindings = f.bindings(64, 6);
        let mut damaged = bindings.last().unwrap().rows.values.to_vec();
        *damaged.last_mut().unwrap() = q[3];
        bindings.last_mut().unwrap().rows.values = &damaged;
        let mut out = vec![987; st.output_words+3];
        let mut scratch = vec![654; st.scratch_words+5];
        assert_eq!(plan.apply(Coefficients { contexts: &f.sources, values: &f.input },
            &bindings, &mut out, &mut scratch, plan.work().unwrap()), Err(TraceError::NonCanonical));
        assert!(out.iter().all(|&v| v == 987));
        assert!(scratch.iter().all(|&v| v == 654));
        assert_eq!(actual(&f, 64, 6), f.oracle(64, 6, Mutation::None));
    }
}
