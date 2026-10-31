//! Authored source regressions, ALL UNRUN under Native HOLD.
//! Independent signed schoolbook + GLOBAL four-limb BigInt CRT/rounded drop,
//! recursive tree (not production schedule), direct DFT fixture encoding.
//! Expected arithmetic never calls production NTT/contraction/trace kernels.
//! Synthetic row errors are public, not source key provenance or HE evidence.
//! Distinct error-row payloads do not establish source-related rotation keys or
//! a whole-trace phase/error law. Ideal selector rows intentionally remain
//! exponent-independent; inverse-group/order mutants need separate witnesses.
use super::olt_trace::*;
use super::*;
use crate::coefficient_layout::{CoefficientRing, DegreeBound};
use crate::coefficient_layout::schedule::{InterleaveSchedule, InterleaveStep};
use num_bigint::BigInt;

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
