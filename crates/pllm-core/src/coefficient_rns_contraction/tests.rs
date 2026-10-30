//! Authored ordinary tests, UNRUN. BigInt expected values never call production
//! NTT/modular/CRT helpers. Large fixtures stay sparse, not quadratic dense runs.
use super::*;
use crate::coefficient_layout::{CoefficientRing, DegreeBound};
use num_bigint::BigInt;

fn residue(value: BigInt, q: u64) -> u64 {
    let modulus = BigInt::from(q);
    let value = ((value % &modulus) + &modulus) % &modulus;
    let (_, digits) = value.to_u64_digits();
    digits.first().copied().unwrap_or(0)
}

fn power(a: u64, exponent: u64, q: u64) -> u64 {
    residue(BigInt::from(a).modpow(&BigInt::from(exponent), &BigInt::from(q)), q)
}

// Independent natural-order evaluation. Sparse forward costs O(N*support);
// inverse is used only at small degree, including inverse scale and twist.
fn evaluate(values: &[u64], psi: u64, q: u64, inverse: bool) -> Vec<u64> {
    let n = values.len();
    let support: Vec<_> = values.iter().copied().enumerate().filter(|&(_, v)| v != 0).collect();
    let root = if inverse { power(psi, q - 2, q) } else { psi };
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

// Signed schoolbook in Z[X]/(X^N+1), with zero terms elided at large degree.
// No production transforms or fixed-width modular products form expected values.
fn schoolbook(a: &[u64], b: &[u64]) -> Vec<BigInt> {
    let n = a.len();
    assert_eq!(b.len(), n);
    let left: Vec<_> = a.iter().copied().enumerate().filter(|&(_, v)| v != 0).collect();
    let right: Vec<_> = b.iter().copied().enumerate().filter(|&(_, v)| v != 0).collect();
    let mut output = vec![BigInt::from(0); n];
    for &(i, a) in &left {
        for &(j, b) in &right {
            let term = BigInt::from(a) * BigInt::from(b);
            if i + j >= n {
                output[i + j - n] -= term;
            } else {
                output[i + j] += term;
            }
        }
    }
    output
}

struct Fixture {
    sources: [NegacyclicNtt; 3],
    targets: [NegacyclicNtt; 4],
    roots: [u64; 4],
}

impl Fixture {
    fn new(n: usize, moduli: [u64; 4]) -> Self {
        // Conditional public mathematical fixtures, not adopted HE parameters.
        // Independent bounded BigInt root search must succeed, never skip/fallback.
        let roots = moduli.map(|q| {
            assert_eq!((q - 1) % (2 * n as u64), 0);
            (2..=64).find_map(|a| {
                let psi = power(a, (q - 1) / (2 * n as u64), q);
                (power(psi, n as u64, q) == q - 1
                    && power(psi, 2 * n as u64, q) == 1).then_some(psi)
            }).expect("conditional fixture has no bounded exact-order root witness")
        });
        let targets = core::array::from_fn(|i| {
            let ring = CoefficientRing::bounded(n, moduli[i], DegreeBound::Public8192).unwrap();
            NegacyclicNtt::new(ring, roots[i]).unwrap()
        });
        let sources = core::array::from_fn(|i| targets[i].clone());
        Self { sources, targets, roots }
    }

    fn small(n: usize) -> Self {
        Self::new(n, [17, 97, 113, 193])
    }

    fn plan(&self) -> RnsContraction<'_> {
        RnsContraction::new(&self.sources, &self.targets).unwrap()
    }

    fn source_spectra(&self, digits: &[u64]) -> Vec<u64> {
        let n = self.sources[0].ring().degree();
        (0..3).flat_map(|i| {
            evaluate(&digits[i * n..(i + 1) * n], self.roots[i], self.sources[i].ring().modulus(), false)
        }).collect()
    }

    fn row_spectra(&self, coefficients: &[u64]) -> Vec<u64> {
        let n = self.sources[0].ring().degree();
        (0..24).flat_map(|poly| {
            let limb = poly % 4;
            evaluate(&coefficients[poly * n..(poly + 1) * n], self.roots[limb], self.targets[limb].ring().modulus(), false)
        }).collect()
    }
}

fn sentinels(n: usize) -> Vec<u64> {
    (0..n).map(|i| u64::MAX - 37 * i as u64).collect()
}

// Complete expected contraction: signed schoolbook sum, then independent
// natural-order evaluation, not a production NTT roundtrip oracle.
fn check_case(fixture: &Fixture, digits: &[u64], row_coefficients: &[u64]) -> Vec<u64> {
    let plan = fixture.plan();
    let n = plan.degree();
    let source = fixture.source_spectra(digits);
    let rows = fixture.row_spectra(row_coefficients);
    let source_before = source.clone();
    let rows_before = rows.clone();
    let mut embedded = sentinels(12 * n);
    let mut embed_scratch = sentinels(2 * n + 3);
    let embed_tail = embed_scratch[2 * n..].to_vec();
    plan.embed_into(
        SourceSpectra { contexts: &fixture.sources, values: &source },
        &mut embedded, &mut embed_scratch, plan.work(Evaluation::Embed).unwrap(),
    ).unwrap();
    assert_eq!(&embed_scratch[2 * n..], embed_tail);
    for i in 0..3 {
        for limb in 0..4 {
            let q = fixture.targets[limb].ring().modulus();
            let lifted: Vec<_> = digits[i * n..(i + 1) * n].iter().map(|&v| residue(BigInt::from(v), q)).collect();
            assert_eq!(&embedded[(i * 4 + limb) * n..(i * 4 + limb + 1) * n],
                evaluate(&lifted, fixture.roots[limb], q, false));
        }
    }
    let mut output = sentinels(8 * n);
    let mut scratch = sentinels(3 * n + 3);
    let tail = scratch[3 * n..].to_vec();
    plan.contract_into(
        SourceSpectra { contexts: &fixture.sources, values: &source },
        RowSpectra { contexts: &fixture.targets, values: &rows },
        &mut output, &mut scratch, plan.work(Evaluation::Contract).unwrap(),
    ).unwrap();
    assert_eq!(&scratch[3 * n..], tail);
    for component in 0..2 {
        for limb in 0..4 {
            let q = fixture.targets[limb].ring().modulus();
            let mut expected = vec![BigInt::from(0); n];
            for i in 0..3 {
                let row_start = ((i * 2 + component) * 4 + limb) * n;
                let product = schoolbook(&digits[i * n..(i + 1) * n], &row_coefficients[row_start..row_start + n]);
                for (sum, term) in expected.iter_mut().zip(product) {
                    *sum += term;
                }
            }
            let coefficients: Vec<_> = expected.into_iter().map(|v| residue(v, q)).collect();
            let start = (component * 4 + limb) * n;
            assert_eq!(&output[start..start + n], evaluate(&coefficients, fixture.roots[limb], q, false));
        }
    }
    assert_eq!(source, source_before);
    assert_eq!(rows, rows_before);
    output
}

#[test]
fn dense_small_degrees_unsigned_extremes_and_negative_wrap_bigint() {
    for n in [1, 2, 4, 8] {
        let fixture = Fixture::small(n);
        let rows: Vec<_> = (0..24 * n).map(|at| {
            let q = fixture.targets[(at / n) % 4].ring().modulus();
            if at % 2 == 0 { q - 1 } else { (at as u64 * 7 + 3) % q }
        }).collect();
        for case in 0..4 {
            let digits: Vec<_> = (0..3 * n).map(|at| {
                let q = fixture.sources[at / n].ring().modulus();
                match case {
                    0 => 0,
                    1 => q - 1,
                    2 => if at % 2 == 0 { q - 1 } else { 1 },
                    _ => (at as u64 * 11 + 2) % q,
                }
            }).collect();
            check_case(&fixture, &digits, &rows);
        }
    }
}

#[test]
fn degree_one_exhausts_each_unsigned_digit_domain() {
    let fixture = Fixture::small(1);
    let rows: Vec<_> = (0..24).map(|at| fixture.targets[at % 4].ring().modulus() - 1 - at as u64 % 3).collect();
    for i in 0..3 {
        for digit in 0..fixture.sources[i].ring().modulus() {
            let mut digits = vec![0; 3];
            digits[i] = digit;
            check_case(&fixture, &digits, &rows);
        }
    }
}

#[test]
fn exhaustive_small_basis_positions_rows_components_limbs_and_wrap() {
    let n = 4;
    let fixture = Fixture::small(n);
    for i in 0..3 {
        for component in 0..2 {
            for limb in 0..4 {
                for from in 0..n {
                    for at in 0..n {
                        let mut digits = vec![0; 3 * n];
                        digits[i * n + from] = fixture.sources[i].ring().modulus() - 1;
                        let mut rows = vec![0; 24 * n];
                        rows[((i * 2 + component) * 4 + limb) * n + at] = fixture.targets[limb].ring().modulus() - 1;
                        check_case(&fixture, &digits, &rows);
                    }
                }
            }
        }
    }
}

#[test]
fn unsigned_lift_is_not_centering_or_copying_ntt_ordinates() {
    let n = 4;
    let fixture = Fixture::small(n);
    let mut digits = vec![0; 3 * n];
    digits[1] = 16; // -1 mod 17, but the unsigned digit is positive 16.
    let source = fixture.source_spectra(&digits);
    let mut output = vec![0; 12 * n];
    let mut scratch = vec![0; 2 * n];
    fixture.plan().embed_into(
        SourceSpectra { contexts: &fixture.sources, values: &source },
        &mut output, &mut scratch, fixture.plan().work(Evaluation::Embed).unwrap(),
    ).unwrap();
    let q = fixture.targets[1].ring().modulus();
    let unsigned = evaluate(&[0, 16, 0, 0], fixture.roots[1], q, false);
    let centered = evaluate(&[0, q - 1, 0, 0], fixture.roots[1], q, false);
    let copied: Vec<_> = source[..n].iter().map(|&v| v % q).collect();
    assert_eq!(&output[n..2 * n], unsigned);
    assert_ne!(unsigned, centered);
    assert_ne!(unsigned, copied);
}

#[test]
fn sparse_public_8192_conditional_fixture_bigint_not_profile_or_drop() {
    let n = 8192;
    let fixture = Fixture::new(n, [
        562949952847873, 562949952798721, 562949952700417, 562949952274433,
    ]);
    assert!(CoefficientRing::new(n, fixture.sources[0].ring().modulus()).is_err());
    let mut digits = vec![0; 3 * n];
    for i in 0..3 {
        digits[i * n + n - 1] = fixture.sources[i].ring().modulus() - 1;
        digits[i * n + n / 2] = 1;
    }
    let mut rows = vec![0; 24 * n];
    for poly in 0..24 {
        rows[poly * n + 1] = fixture.targets[poly % 4].ring().modulus() - 1;
        rows[poly * n + n / 2] = 2 + poly as u64;
    }
    check_case(&fixture, &digits, &rows);
}

fn refuse(
    fixture: &Fixture,
    evaluation: Evaluation,
    source_contexts: &[NegacyclicNtt; 3],
    target_contexts: &[NegacyclicNtt; 4],
    source: &[u64],
    rows: &[u64],
    output_len: usize,
    scratch_len: usize,
    ceiling: Work,
    error: Error,
) {
    let mut output = sentinels(output_len);
    let mut scratch = sentinels(scratch_len);
    let output_before = output.clone();
    let scratch_before = scratch.clone();
    let source_before = source.to_vec();
    let rows_before = rows.to_vec();
    let source_input = SourceSpectra { contexts: source_contexts, values: source };
    let result = match evaluation {
        Evaluation::Embed => fixture.plan().embed_into(source_input, &mut output, &mut scratch, ceiling),
        Evaluation::Contract => fixture.plan().contract_into(
            source_input, RowSpectra { contexts: target_contexts, values: rows },
            &mut output, &mut scratch, ceiling,
        ),
    };
    assert_eq!(result, Err(error));
    assert_eq!(output, output_before);
    assert_eq!(scratch, scratch_before);
    assert_eq!(source, source_before);
    assert_eq!(rows, rows_before);
}

#[test]
fn every_residue_including_final_row_component_limb_refuses_atomically() {
    let n = 4;
    let fixture = Fixture::small(n);
    for evaluation in [Evaluation::Embed, Evaluation::Contract] {
        let plan = fixture.plan();
        let storage = plan.storage(evaluation).unwrap();
        let quote = plan.work(evaluation).unwrap();
        for at in 0..3 * n {
            for bad in [fixture.sources[at / n].ring().modulus(), u64::MAX] {
                let mut source = vec![0; 3 * n];
                source[at] = bad;
                refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &vec![0; 24 * n],
                    storage.output_words, storage.scratch_words + 3, quote, Error::NonCanonical);
            }
        }
        if evaluation == Evaluation::Contract {
            for at in 0..24 * n {
                for bad in [fixture.targets[(at / n) % 4].ring().modulus(), u64::MAX] {
                    let mut rows = vec![0; 24 * n];
                    rows[at] = bad;
                    refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &vec![0; 3 * n], &rows,
                        storage.output_words, storage.scratch_words + 3, quote, Error::NonCanonical);
                }
            }
        }
    }
}

#[test]
fn all_lengths_and_each_one_short_work_category_refuse_atomically() {
    let fixture = Fixture::small(4);
    for evaluation in [Evaluation::Embed, Evaluation::Contract] {
        let plan = fixture.plan();
        let storage = plan.storage(evaluation).unwrap();
        let quote = plan.work(evaluation).unwrap();
        let source = vec![0; storage.source_words];
        let rows = vec![0; 24 * plan.degree()];
        for length in [0, storage.source_words - 1, storage.source_words + 1] {
            refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &vec![u64::MAX; length], &rows,
                storage.output_words, storage.scratch_words + 3, quote, Error::BufferLength);
        }
        if evaluation == Evaluation::Contract {
            for length in [0, storage.row_words - 1, storage.row_words + 1] {
                refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &vec![u64::MAX; length],
                    storage.output_words, storage.scratch_words + 3, quote, Error::BufferLength);
            }
        }
        for length in [0, storage.output_words - 1, storage.output_words + 1] {
            refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &rows,
                length, storage.scratch_words + 3, quote, Error::BufferLength);
        }
        for length in [0, storage.scratch_words - 1] {
            refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &rows,
                storage.output_words, length, quote, Error::BufferLength);
        }
        macro_rules! one_short {
            ($($field:ident),*) => { $(
                if quote.events.$field != 0 {
                    let mut ceiling = quote;
                    ceiling.events.$field -= 1;
                    refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &rows,
                        storage.output_words, storage.scratch_words + 3, ceiling, Error::WorkLimit);
                }
            )* };
        }
        one_short!(checks, residue_checks, moves, swaps, modular_additions,
            modular_subtractions, modular_multiplications, exponentiations,
            exponent_steps, primality_rounds);
        let mut ceiling = quote;
        ceiling.coefficient_reductions -= 1;
        refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &rows,
            storage.output_words, storage.scratch_words + 3, ceiling, Error::WorkLimit);
    }
}

#[test]
fn public_8192_final_word_and_capacity_refusals_are_atomic() {
    let fixture = Fixture::new(8192, [65537, 786433, 5767169, 7340033]);
    let n = fixture.plan().degree();
    let source = vec![0; 3 * n];
    let rows = vec![0; 24 * n];
    for evaluation in [Evaluation::Embed, Evaluation::Contract] {
        let storage = fixture.plan().storage(evaluation).unwrap();
        let quote = fixture.plan().work(evaluation).unwrap();
        let mut bad_source = source.clone();
        bad_source[3 * n - 1] = fixture.sources[2].ring().modulus();
        refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &bad_source, &rows,
            storage.output_words, storage.scratch_words + 3, quote, Error::NonCanonical);
        if evaluation == Evaluation::Contract {
            let mut bad_rows = rows.clone();
            bad_rows[24 * n - 1] = fixture.targets[3].ring().modulus();
            refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &bad_rows,
                storage.output_words, storage.scratch_words + 3, quote, Error::NonCanonical);
        }
        refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &rows,
            storage.output_words - 1, storage.scratch_words + 3, quote, Error::BufferLength);
        refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &rows,
            storage.output_words, storage.scratch_words - 1, quote, Error::BufferLength);
        let mut short_work = quote;
        short_work.events.modular_multiplications -= 1;
        refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &rows,
            storage.output_words, storage.scratch_words + 3, short_work, Error::WorkLimit);
    }
}

#[test]
fn ordered_context_degree_and_root_identity_refusals_are_atomic() {
    let fixture = Fixture::small(4);
    let other_degree = Fixture::small(2);
    let mut swapped_source = fixture.sources.clone();
    swapped_source.swap(0, 2);
    let mut swapped_target = fixture.targets.clone();
    swapped_target.swap(0, 2);
    let mut changed_root_source = fixture.sources.clone();
    changed_root_source[2] = NegacyclicNtt::new(fixture.sources[2].ring(),
        power(fixture.roots[2], 3, fixture.sources[2].ring().modulus())).unwrap();
    let mut changed_root_target = fixture.targets.clone();
    changed_root_target[3] = NegacyclicNtt::new(fixture.targets[3].ring(),
        power(fixture.roots[3], 3, fixture.targets[3].ring().modulus())).unwrap();
    for sources in [&swapped_source, &changed_root_source, &other_degree.sources] {
        assert!(matches!(RnsContraction::new(sources, &fixture.targets), Err(Error::ContextMismatch)));
        for evaluation in [Evaluation::Embed, Evaluation::Contract] {
            let storage = fixture.plan().storage(evaluation).unwrap();
            refuse(&fixture, evaluation, sources, &fixture.targets, &vec![0; 12], &vec![0; 96],
                storage.output_words, storage.scratch_words + 3, fixture.plan().work(evaluation).unwrap(), Error::ContextMismatch);
        }
    }
    for targets in [&swapped_target, &changed_root_target, &other_degree.targets] {
        refuse(&fixture, Evaluation::Contract, &fixture.sources, targets, &vec![0; 12], &vec![0; 96],
            32, 15, fixture.plan().work(Evaluation::Contract).unwrap(), Error::ContextMismatch);
    }
    assert!(matches!(RnsContraction::new(&fixture.sources, &swapped_target), Err(Error::ContextMismatch)));
    assert!(matches!(RnsContraction::new(&fixture.sources, &other_degree.targets), Err(Error::ContextMismatch)));
    let mut duplicate = fixture.targets.clone();
    duplicate[3] = duplicate[0].clone();
    assert!(matches!(RnsContraction::new(&fixture.sources, &duplicate), Err(Error::ContextMismatch)));
    // A different valid root for p is allowed when selected consistently at plan
    // construction; it is refused only when input metadata disagrees with plan.
    assert!(RnsContraction::new(&fixture.sources, &changed_root_target).is_ok());
}

#[test]
fn mixed_refusal_precedence_preserves_entire_output_and_excess_scratch_tail() {
    let fixture = Fixture::small(4);
    let mut wrong_sources = fixture.sources.clone();
    wrong_sources.swap(0, 2);
    let mut wrong_targets = fixture.targets.clone();
    wrong_targets.swap(0, 3);
    let mut source = vec![0; 12];
    source[0] = u64::MAX;
    source[11] = u64::MAX;
    let mut rows = vec![0; 96];
    rows[0] = u64::MAX;
    rows[95] = u64::MAX;
    for evaluation in [Evaluation::Embed, Evaluation::Contract] {
        let storage = fixture.plan().storage(evaluation).unwrap();
        let quote = fixture.plan().work(evaluation).unwrap();
        let mut short_work = quote;
        short_work.events.checks -= 1;
        // All four faults coexist: context wins over lengths, work and residues.
        refuse(&fixture, evaluation, &wrong_sources, &wrong_targets, &source[..11], &rows[..95],
            storage.output_words - 1, storage.scratch_words + 7, short_work, Error::ContextMismatch);
        if evaluation == Evaluation::Contract {
            // Row context alone also precedes the other three fault categories.
            refuse(&fixture, evaluation, &fixture.sources, &wrong_targets, &source[..11], &rows[..95],
                storage.output_words - 1, storage.scratch_words + 7, short_work, Error::ContextMismatch);
        }
        // With contexts repaired, lengths win over bad work and bad residues.
        refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source[..11], &rows[..95],
            storage.output_words - 1, storage.scratch_words + 7, short_work, Error::BufferLength);
        // Source-length refusal alone precedes bad work and noncanonical source.
        refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source[..11], &rows,
            storage.output_words, storage.scratch_words + 7, short_work, Error::BufferLength);
        if evaluation == Evaluation::Contract {
            // Row-length refusal, with source/output lengths now exact.
            refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &rows[..95],
                storage.output_words, storage.scratch_words + 7, short_work, Error::BufferLength);
        }
        // With all geometry repaired, ceiling refusal precedes residue scanning.
        refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &rows,
            storage.output_words, storage.scratch_words + 7, short_work, Error::WorkLimit);
        refuse(&fixture, evaluation, &fixture.sources, &fixture.targets, &source, &rows,
            storage.output_words, storage.scratch_words + 7, quote, Error::NonCanonical);
    }
    // ArithmeticCapacity remains a defensive unreachable path at valid bounds;
    // no forged private descriptor or panic/rollback experiment is introduced.
}

#[test]
fn non_residue_comparison_envelope_has_independent_source_census() {
    // Ordinary regression against the attested field/control layout. This is a
    // logical source census, NOT size_of/ABI/CPU measurement, and uses no
    // production count helper for expected bounds. Layout changes require review.
    fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        source.split_once(start).unwrap().1.split_once(end).unwrap().0
    }
    let ntt = include_str!("../coefficient_ntt.rs");
    let layout = include_str!("../coefficient_layout.rs");
    let own = include_str!("../coefficient_rns_contraction.rs");
    let ring = between(layout, "pub struct CoefficientRing {", "\n}");
    assert!(ring.contains("degree: usize,"));
    assert!(ring.contains("modulus: u64,"));
    let ring_scalars = ring.matches(':').count();
    assert_eq!(ring_scalars, 2);
    let descriptor = between(ntt, "pub struct NegacyclicNtt {", "\n}");
    assert_eq!(descriptor.matches(':').count(), 6);
    assert!(descriptor.contains("ring: CoefficientRing,"));
    for field in ["psi: u64,", "psi_inv: u64,", "n_inv: u64,"] {
        assert!(descriptor.contains(field));
    }
    assert!(descriptor.contains("forward_stage_roots: [u64; 13],"));
    assert!(descriptor.contains("inverse_stage_roots: [u64; 13],"));
    let scalar_fields = descriptor.matches(": u64,").count();
    let root_arrays = descriptor.matches(": [u64; 13],").count();
    let descriptor_comparisons = ring_scalars + scalar_fields + root_arrays * 13;
    assert_eq!(descriptor_comparisons, 31);

    let event_struct = between(ntt, "pub struct LogicalOperations {", "\n}");
    let event_fields = event_struct.matches(": u64,").count();
    assert_eq!(event_fields, 10);
    let macro_fields = between(own, "$action!($left, $right;", ")");
    let macro_fields: Vec<_> = macro_fields.split(',').map(str::trim).filter(|v| !v.is_empty()).collect();
    assert_eq!(macro_fields, ["checks", "residue_checks", "moves", "swaps", "modular_additions",
        "modular_subtractions", "modular_multiplications", "exponentiations", "exponent_steps", "primality_rounds"]);
    let fits = between(own, "pub fn fits(", "\n}\n");
    assert_eq!(fits.matches("event_fields!(fits,").count(), 1);
    assert_eq!(fits.matches("self.coefficient_reductions <= ceiling.coefficient_reductions").count(), 1);
    let plan_fields = between(own, "pub struct RnsContraction<'a> {", "\n}");
    assert!(plan_fields.contains("sources: &'a [NegacyclicNtt; 3],"));
    assert!(plan_fields.contains("targets: &'a [NegacyclicNtt; 4],"));

    let storage = between(own, "pub fn storage(", "\n    ///");
    let embed_branch = between(storage, "Evaluation::Embed =>", "\n");
    let contract_branch = between(storage, "Evaluation::Contract =>", "\n");
    assert_eq!(embed_branch.matches("words(").count(), 2);
    assert_eq!(contract_branch.matches("words(").count(), 3);
    assert_eq!(storage.matches(".checked_mul(").count(), 2); // word helper + bytes
    assert_eq!(storage.matches(".checked_add(").count(), 3);
    assert_eq!(storage.matches(".ok_or(").count(), 3); // helper + total + bytes
    assert_eq!(storage.matches(".and_then(").count(), 2);
    assert_eq!(storage.matches('?').count(), 8); // includes both exclusive branches
    let word_multiplications = 1 + contract_branch.matches("words(").count();
    let storage_capacity = word_multiplications + storage.matches(".checked_add(").count() + 1;
    let storage_adapters = word_multiplications + storage.matches(".and_then(").count()
        + storage.matches(".ok_or(").count() - 1;
    let storage_results = storage.matches('?').count() - embed_branch.matches('?').count();
    assert_eq!((storage_capacity, storage_adapters, storage_results), (8, 8, 6));

    let work = between(own, "pub fn work(", "\n    fn preflight(");
    let add = between(own, "fn add_ntt(", "\n    /// No scalarization");
    assert_eq!(work.matches("u64::try_from(").count(), 1);
    assert_eq!(work.matches(".map_err(").count(), 1);
    assert_eq!(work.matches("scaled(").count(), 5);
    assert_eq!(work.matches(".checked_mul(").count(), 1);
    assert_eq!(work.matches(".checked_add(").count(), 1);
    assert_eq!(work.matches(".ok_or(").count(), 2);
    assert_eq!(work.matches("work.add_ntt(").count(), 2);
    assert_eq!(work.matches('?').count(), 9);
    assert_eq!(add.matches(".checked_add(").count(), 1);
    assert_eq!(add.matches(".ok_or(").count(), 1);
    assert_eq!(add.matches('?').count(), 1);
    assert!(work.contains("for source in self.sources"));
    assert!(work.contains("for target in self.targets"));
    let source_limbs = 3;
    let target_limbs = 4;
    let transform_calls = source_limbs + source_limbs * target_limbs;
    let checked_field_additions = transform_calls * event_fields;
    let work_capacity = work.matches("u64::try_from(").count()
        + work.matches("scaled(").count() + work.matches(".checked_add(").count()
        + checked_field_additions;
    let work_adapters = work.matches(".map_err(").count()
        + work.matches("scaled(").count() + 1 + checked_field_additions;
    // Seven non-helper ? sites + 15 helper ? sites + 150 expanded field ? sites.
    let work_results = work.matches('?').count() - work.matches("work.add_ntt(").count()
        + transform_calls + checked_field_additions;
    assert_eq!((work_capacity, work_adapters, work_results), (157, 157, 172));

    let preflight = between(own, "fn preflight(", "\n    /// d_i");
    let ntt_quote = between(ntt, "pub fn operations(", "\n    fn validate(");
    assert_eq!(preflight.matches(".len()").count(), 4);
    assert_eq!(preflight.matches(".fits(").count(), 1);
    assert_eq!(preflight.matches('?').count(), 2);
    assert_eq!(ntt_quote.matches("match operation").count(), 1);
    assert_eq!(ntt_quote.matches("if n == 1").count(), 1);
    assert_eq!(ntt_quote.matches("operation == Operation::Inverse").count(), 1);
    assert_eq!(preflight.matches("rows.is_some()").count(), 1);
    assert_eq!(preflight.matches("rows.is_some_and(").count(), 2);
    assert_eq!(preflight.matches("if let Some(rows)").count(), 1);
    let row_shapes = preflight.matches("rows.is_some()").count()
        + preflight.matches("rows.is_some_and(").count() + preflight.matches("if let Some(rows)").count();
    let evaluation_selections = storage.matches("match evaluation").count()
        + work.matches("evaluation == Evaluation::Contract").count();
    let ntt_quote_selections = transform_calls * (ntt_quote.matches("match operation").count()
        + ntt_quote.matches("if n == 1").count() + ntt_quote.matches("operation == Operation::Inverse").count());
    let result_tags = storage_results + work_results + preflight.matches('?').count()
        + 1 /* caller preflight? */ + transform_calls /* NTT expect */;
    let census = (source_limbs + target_limbs) * descriptor_comparisons
        + preflight.matches(".len()").count() + event_fields + 1 /* reductions ceiling */
        + storage_capacity + work_capacity + storage_adapters + work_adapters
        + result_tags + 4 /* aggregate context/length/work/canonical refusals */
        + row_shapes + evaluation_selections + ntt_quote_selections;
    assert_eq!(result_tags, 196);
    assert_eq!(census, 813);
    let fixture = Fixture::small(4);
    for evaluation in [Evaluation::Embed, Evaluation::Contract] {
        let own_residues = if evaluation == Evaluation::Contract { 27 * 4 } else { 3 * 4 };
        let separately_quoted_ntt_checks = transform_calls * (4 + 2);
        let quoted = fixture.plan().work(evaluation).unwrap().events.checks;
        let non_residue_envelope = quoted - own_residues as u64 - separately_quoted_ntt_checks as u64;
        assert!(non_residue_envelope >= census as u64);
        // Exact 1024 compatibility remains checked in the stage-formula test;
        // this independently-derived inequality is the sufficiency proof.
    }
}

#[test]
fn logical_inventory_and_declared_storage_not_tariff_or_physical_claim() {
    for n in [1, 8, 8192] {
        let fixture = if n <= 8 { Fixture::small(n) } else {
            Fixture::new(n, [65537, 786433, 5767169, 7340033])
        };
        let plan = fixture.plan();
        for evaluation in [Evaluation::Embed, Evaluation::Contract] {
            let contract = evaluation == Evaluation::Contract;
            let storage = plan.storage(evaluation).unwrap();
            assert_eq!(storage.source_words, 3 * n);
            assert_eq!(storage.row_words, if contract { 24 * n } else { 0 });
            assert_eq!(storage.output_words, if contract { 8 * n } else { 12 * n });
            assert_eq!(storage.scratch_words, if contract { 3 * n } else { 2 * n });
            assert_eq!(storage.total_words, if contract { 38 * n } else { 17 * n });
            assert_eq!(storage.total_bytes, storage.total_words * 8);
            let quote = plan.work(evaluation).unwrap();
            let mut expected = Work {
                events: LogicalOperations {
                    checks: 1024 + if contract { 27 * n as u64 } else { 3 * n as u64 },
                    residue_checks: if contract { 27 * n as u64 } else { 3 * n as u64 },
                    moves: if contract { 44 * n as u64 } else { 12 * n as u64 },
                    modular_additions: if contract { 24 * n as u64 } else { 0 },
                    modular_multiplications: if contract { 24 * n as u64 } else { 0 },
                    ..LogicalOperations::default()
                },
                coefficient_reductions: 12 * n as u64,
            };
            // Independent stage formula, not production operations() as oracle.
            let butterflies = n as u64 / 2 * u64::from(n.trailing_zeros());
            expected.events.checks += 15 * (2 + n as u64);
            expected.events.residue_checks += 15 * n as u64;
            expected.events.moves += 15 * (2 * n as u64 + 2 * butterflies);
            expected.events.swaps += 15 * (n as u64 / 2);
            expected.events.modular_additions += 15 * butterflies;
            expected.events.modular_subtractions += 15 * butterflies;
            if n != 1 {
                expected.events.modular_multiplications += 30 * butterflies + 33 * n as u64;
            }
            assert_eq!(quote, expected);
            assert!(quote.fits(quote));
            assert!(!quote.fits(Work::default()));
            assert_eq!(quote.events.exponentiations, 0);
            assert_eq!(quote.events.primality_rounds, 0);
        }
    }
}

#[test]
fn public_synthetic_row_phase_crt_identity_only_no_rounded_drop_integration() {
    let n = 4;
    let fixture = Fixture::small(n);
    let moduli = fixture.sources.each_ref().map(|context| context.ring().modulus());
    let q_product = moduli.iter().fold(BigInt::from(1), |a, &q| a * BigInt::from(q));
    let p = fixture.targets[3].ring().modulus();
    let basis: [BigInt; 3] = core::array::from_fn(|i| {
        let partial = &q_product / BigInt::from(moduli[i]);
        let inverse = power(residue(partial.clone(), moduli[i]), moduli[i] - 2, moduli[i]);
        let value = partial * BigInt::from(inverse);
        for (j, &q) in moduli.iter().enumerate() {
            assert_eq!(residue(value.clone(), q), u64::from(i == j));
        }
        value
    });
    // Public c and public t stand for supplied sigma(c1), sigma(s). No secret,
    // key issuance, or automorphism implementation is present in this test.
    let c = [BigInt::from(-1), BigInt::from(2), BigInt::from(0), BigInt::from(-3)];
    let digits: Vec<_> = (0..3).flat_map(|i| c.iter().map(move |v| residue(v.clone(), moduli[i]))).collect();
    for at in 0..n {
        let reconstructed = (0..3).fold(BigInt::from(0), |sum, i| sum + BigInt::from(digits[i * n + at]) * &basis[i]);
        assert_eq!((reconstructed - &c[at]) % &q_product, BigInt::from(0));
    }
    let s = [1, 0, 1, 0];
    let t = [0, 1, 0, 1];
    let mut rows = vec![0; 24 * n];
    let mut errors = vec![0; 3 * n];
    for i in 0..3 {
        errors[i * n + i] = 1 + i as u64;
        let a = [1 + i as u64, 1, 0, 2];
        let a_s = schoolbook(&a, &s);
        for limb in 0..4 {
            let q = fixture.targets[limb].ring().modulus();
            for at in 0..n {
                let b = BigInt::from(p) * &basis[i] * BigInt::from(t[at])
                    - &a_s[at] + BigInt::from(errors[i * n + at]);
                rows[(i * 2 * 4 + limb) * n + at] = residue(b, q);
                rows[((i * 2 + 1) * 4 + limb) * n + at] = a[at] % q;
            }
        }
    }
    let output = check_case(&fixture, &digits, &rows);
    for limb in 0..4 {
        let q = fixture.targets[limb].ring().modulus();
        let b = evaluate(&output[limb * n..(limb + 1) * n], fixture.roots[limb], q, true);
        let a = evaluate(&output[(4 + limb) * n..(5 + limb) * n], fixture.roots[limb], q, true);
        let a_s = schoolbook(&a, &s);
        let c_mod: Vec<_> = c.iter().map(|v| residue(v.clone(), q)).collect();
        let c_t = schoolbook(&c_mod, &t);
        let mut noise = vec![BigInt::from(0); n];
        for i in 0..3 {
            for (sum, term) in noise.iter_mut().zip(schoolbook(&digits[i * n..(i + 1) * n], &errors[i * n..(i + 1) * n])) {
                *sum += term;
            }
        }
        for at in 0..n {
            assert_eq!(residue(BigInt::from(b[at]) + &a_s[at], q),
                residue(BigInt::from(p) * &c_t[at] + &noise[at], q));
        }
    }
}
