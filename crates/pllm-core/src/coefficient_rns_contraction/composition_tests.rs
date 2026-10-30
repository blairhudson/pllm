//! Composition tests authored UNRUN against the frozen public API (Sol21).
//! Existing contraction tests/census stay separate. Constructor-before-first-write
//! remains a static diff review gate, never a text grep test or panic rollback claim.
//!
//! Authored freeze-round refusal inventory:
//! - source/rows: 0, required-1, required+1; output: 0, 6N-1, 6N+5;
//!   scratch: 0, 19N-1; exact errors and complete input/output/scratch snapshots,
//!   including unpassed backing tails and guard gaps, at small/public degree;
//! - source and target reorder/degree/root mismatches, including removed p;
//! - every source/row position at N=4 set to its modulus and u64::MAX;
//! - 35 extra and 11 contraction fields enumerated, each positive field one short,
//!   all other fields at independent exact expectations; zero fields never decremented;
//! - precedence: storage -> extra quote -> outer lengths -> extra ceiling ->
//!   old storage/work -> contexts -> source/row lengths -> old ceiling ->
//!   canonical scan -> three constructors -> first output OR scratch write.
//!
//! Independent finite-quote expectations for that round, not CPU scalar funding:
//! controls: capacity=111, adapters=118, results=141, ceiling=35, length=2,
//! dispatch=29, selected_loop=18N+405, slices=39, selected_calls=43.
//! B=(N/2)*log2(N); eight inverse transforms: checks=8N+16, residue=8N,
//! moves=16N+16B, swaps=8*floor(N/2), add=sub=8B,
//! mul=(N==1 ? 0 : 16B+24N), all remaining inverse categories zero.
//! Six drop applications: checks=18N+30, residue=12N, moves=6N, modsub=12N,
//! modmul=6N, unsigned_reductions=6N, plain_sub=6N+6, plain_div=6.
//! Three constructors: checks=390, stores=15, Euclid=384, signed_div=384,
//! signed_mul=768, signed_sub=768, signed_rem=3, unsigned_rem=3.
//! These independent counts must be reviewed, never weakened to match a quote.
use super::*;

fn expected_contraction(n: usize) -> Work {
    let count = n as u64;
    let butterflies = (count / 2) * u64::from(n.trailing_zeros());
    Work {
        events: LogicalOperations {
            checks: 1024 + 27 * count + 15 * (count + 2),
            residue_checks: 42 * count,
            moves: 74 * count + 30 * butterflies,
            swaps: 15 * (count / 2),
            modular_additions: 24 * count + 15 * butterflies,
            modular_subtractions: 15 * butterflies,
            modular_multiplications: 24 * count + if n == 1 { 0 } else { 30 * butterflies + 33 * count },
            ..LogicalOperations::default()
        },
        coefficient_reductions: 12 * count,
    }
}

fn expected_extra(n: usize) -> CompositionExtraWork {
    let count = n as u64;
    let butterflies = (count / 2) * u64::from(n.trailing_zeros());
    CompositionExtraWork {
        inverse_ntts: LogicalOperations {
            checks: 8 * count + 16,
            residue_checks: 8 * count,
            moves: 16 * count + 16 * butterflies,
            swaps: 8 * (count / 2),
            modular_additions: 8 * butterflies,
            modular_subtractions: 8 * butterflies,
            modular_multiplications: if n == 1 { 0 } else { 16 * butterflies + 24 * count },
            exponentiations: 0,
            exponent_steps: 0,
            primality_rounds: 0,
        },
        drop: DropWork {
            checks: 18 * count + 30,
            residue_checks: 12 * count,
            moves: 6 * count,
            modular_subtractions: 12 * count,
            modular_multiplications: 6 * count,
            unsigned_reductions: 6 * count,
            ordinary_subtractions: 6 * count + 6,
            ordinary_divisions: 6,
        },
        construction: DropConstructionWork {
            checks: 390,
            descriptor_stores: 15,
            euclid_iterations: 384,
            signed_divisions: 384,
            signed_multiplications: 768,
            signed_subtractions: 768,
            signed_remainders: 3,
            unsigned_remainders: 3,
        },
        control: CompositionControlWork {
            capacity_outcomes: 111,
            adapters: 118,
            result_inspections: 141,
            ceiling_comparisons: 35,
            length_comparisons: 2,
            dispatches: 29,
            selected_loop_polls: 18 * count + 405,
            slice_partitions: 39,
            selected_phase_calls: 43,
        },
    }
}

fn expected_work(n: usize) -> CompositionWork {
    CompositionWork { contraction: expected_contraction(n), extra: expected_extra(n) }
}

// Positive remainder plus BigInt inverse; no production arithmetic or Fermat
// inverse is used for CRT/drop. Division below has nonnegative numerator.
fn rounded_crt(vq: u64, vp: u64, q: u64, p: u64) -> u64 {
    let inverse = BigInt::from(q).modinv(&BigInt::from(p)).unwrap();
    let t = residue((BigInt::from(vp) - BigInt::from(vq)) * inverse, p);
    let x = BigInt::from(vq) + BigInt::from(q) * BigInt::from(t);
    let h = (p - 1) / 2;
    assert!(x >= BigInt::from(0));
    let y = residue((x + BigInt::from(h)) / BigInt::from(p), q);
    let r = if vp <= h { BigInt::from(vp) } else { BigInt::from(vp) - BigInt::from(p) };
    assert_eq!(residue(BigInt::from(p) * BigInt::from(y), q),
        residue(BigInt::from(vq) - r, q));
    y
}

// O(sd*sR) schoolbook products plus O(N) materialization. No inverse transform,
// dense large-N convolution, or transform of an expected output occurs here.
fn contraction_coefficients(fixture: &Fixture, digits: &[u64], rows: &[u64]) -> Vec<u64> {
    let n = fixture.sources[0].ring().degree();
    assert_eq!(digits.len(), 3 * n);
    assert_eq!(rows.len(), 24 * n);
    let mut coefficients = vec![0; 8 * n];
    for component in 0..2 {
        for limb in 0..4 {
            let q = fixture.targets[limb].ring().modulus();
            let mut sums = vec![BigInt::from(0); n];
            for i in 0..3 {
                let start = ((i * 2 + component) * 4 + limb) * n;
                for (sum, term) in sums.iter_mut().zip(schoolbook(
                    &digits[i * n..(i + 1) * n], &rows[start..start + n],
                )) {
                    *sum += term;
                }
            }
            for (at, sum) in sums.into_iter().enumerate() {
                coefficients[(component * 4 + limb) * n + at] = residue(sum, q);
            }
        }
    }
    coefficients
}

fn rounded_coefficients(fixture: &Fixture, coefficients: &[u64]) -> Vec<u64> {
    let n = fixture.sources[0].ring().degree();
    assert_eq!(coefficients.len(), 8 * n);
    let p = fixture.targets[3].ring().modulus();
    let mut output = vec![0; 6 * n];
    for component in 0..2 {
        for limb in 0..3 {
            let q = fixture.targets[limb].ring().modulus();
            for at in 0..n {
                output[(component * 3 + limb) * n + at] = rounded_crt(
                    coefficients[(component * 4 + limb) * n + at],
                    coefficients[(component * 4 + 3) * n + at], q, p,
                );
            }
        }
    }
    output
}

fn assert_storage(fixture: &Fixture) {
    let n = fixture.sources[0].ring().degree();
    let storage: CompositionStorage = fixture.plan().composition_storage().unwrap();
    assert_eq!(storage.source_words, 3 * n);
    assert_eq!(storage.row_words, 24 * n);
    assert_eq!(storage.output_words, 6 * n);
    assert_eq!(storage.scratch_words, 19 * n);
    assert_eq!(storage.total_words, 52 * n);
    assert_eq!(storage.total_bytes, 416 * n);
    // Declared disjoint buffers, not a physical-peak allocation claim.
}

// Every admitted fixture executes all three retained-limb drops through the
// public composition call. Input spectra use independent direct evaluation.
// Guard gaps and tails share backing allocations but use safe exclusive splits.
fn check_composition(fixture: &Fixture, digits: &[u64], rows: &[u64], extra_scratch: usize) -> Vec<u64> {
    let n = fixture.sources[0].ring().degree();
    assert_storage(fixture);
    let coefficients = contraction_coefficients(fixture, digits, rows);
    let expected = rounded_coefficients(fixture, &coefficients);
    let mut source_backing = sentinels(3 * n + 5);
    source_backing[..3 * n].copy_from_slice(&fixture.source_spectra(digits));
    let mut row_backing = sentinels(24 * n + 5);
    row_backing[..24 * n].copy_from_slice(&fixture.row_spectra(rows));
    let source_before = source_backing.clone();
    let rows_before = row_backing.clone();
    // Physical output backing has 6N+5 words; only its exact 6N prefix is passed.
    let scratch_len = 19 * n + extra_scratch;
    let mut backing = sentinels(3 + 6 * n + 5 + scratch_len + 4);
    let before = backing.clone();
    let actual;
    {
        let (_, rest) = backing.split_at_mut(3);
        let (output_backing, rest) = rest.split_at_mut(6 * n + 5);
        let (output, _) = output_backing.split_at_mut(6 * n);
        let (scratch, _) = rest.split_at_mut(scratch_len);
        let plan = fixture.plan();
        let ceiling = expected_work(n);
        assert_eq!(plan.work(Evaluation::Contract).unwrap(), ceiling.contraction);
        assert_eq!(plan.composition_extra_work().unwrap(), ceiling.extra);
        plan.contract_drop_into(
            SourceSpectra { contexts: &fixture.sources, values: &source_backing[..3 * n] },
            RowSpectra { contexts: &fixture.targets, values: &row_backing[..24 * n] },
            output, scratch,
            ceiling,
        ).unwrap();
        assert_eq!(output, expected.as_slice());
        for component in 0..2 {
            for limb in 0..3 {
                let q = fixture.targets[limb].ring().modulus();
                assert!(output[(component * 3 + limb) * n..(component * 3 + limb + 1) * n]
                    .iter().all(|&value| value < q));
            }
        }
        actual = output.to_vec();
    }
    assert_eq!(source_backing, source_before);
    assert_eq!(row_backing, rows_before);
    assert_eq!(&backing[..3], &before[..3]);
    let output_end = 3 + 6 * n;
    assert_eq!(&backing[output_end..output_end + 5], &before[output_end..output_end + 5]);
    let scratch_tail = output_end + 5 + 19 * n;
    assert_eq!(&backing[scratch_tail..], &before[scratch_tail..]);
    actual
}

fn dense_rows(fixture: &Fixture) -> Vec<u64> {
    let n = fixture.sources[0].ring().degree();
    (0..24 * n).map(|at| {
        let poly = at / n;
        let q = fixture.targets[poly % 4].ring().modulus();
        if (at + poly) % 3 == 0 { q - 1 } else { (11 * at as u64 + 7 * poly as u64 + 3) % q }
    }).collect()
}

#[test]
fn composition_dense_bigint_all_small_degrees_prime_orders_and_unsigned_extremes() {
    for n in [1, 2, 4, 8] {
        for moduli in [[17, 97, 113, 193], [113, 17, 97, 193], [193, 97, 113, 17]] {
            let fixture = Fixture::new(n, moduli);
            let rows = dense_rows(&fixture);
            for case in 0..3 {
                let digits: Vec<_> = (0..3 * n).map(|at| {
                    let q = fixture.sources[at / n].ring().modulus();
                    match case {
                        0 => q - 1,
                        1 => if at % 2 == 0 { q - 1 } else { 1 },
                        _ => (5 * at as u64 + 2) % q,
                    }
                }).collect();
                check_composition(&fixture, &digits, &rows, 0);
            }
        }
    }
}

#[test]
fn composition_consistently_changed_valid_roots_preserve_coefficient_results() {
    let fixture = Fixture::small(4);
    let mut changed = Fixture::small(4);
    for limb in 0..4 {
        let q = changed.targets[limb].ring().modulus();
        changed.roots[limb] = power(changed.roots[limb], 3, q);
        changed.targets[limb] = NegacyclicNtt::new(changed.targets[limb].ring(), changed.roots[limb]).unwrap();
    }
    changed.sources = core::array::from_fn(|i| changed.targets[i].clone());
    let digits: Vec<_> = (0..12).map(|at| fixture.sources[at / 4].ring().modulus() - 1 - at as u64 % 3).collect();
    let rows = dense_rows(&fixture);
    assert_eq!(check_composition(&fixture, &digits, &rows, 0),
        check_composition(&changed, &digits, &rows, 7));
}

#[test]
fn composition_zero_source_or_zero_rows_clear_every_poisoned_output_word() {
    for n in [1, 2, 4, 8] {
        let fixture = Fixture::small(n);
        assert_eq!(check_composition(&fixture, &vec![0; 3 * n], &dense_rows(&fixture), 7), vec![0; 6 * n]);
        let digits: Vec<_> = (0..3 * n).map(|at| fixture.sources[at / n].ring().modulus() - 1).collect();
        assert_eq!(check_composition(&fixture, &digits, &vec![0; 24 * n], 0), vec![0; 6 * n]);
    }
}

#[test]
fn composition_midpoints_common_whole_integer_signed_quotients_and_crt_crosscheck() {
    let n = 4;
    for moduli in [[17, 97, 113, 193], [193, 97, 113, 17]] {
        let fixture = Fixture::new(n, moduli);
        let p = moduli[3];
        let h = (p - 1) / 2;
        assert_eq!(p % 2, 1); // Odd p: no half-integer tie exists.
        let mut ks = vec![BigInt::from(0), BigInt::from(1), BigInt::from(-1)];
        for q in moduli[..3].iter().copied() {
            for center in [BigInt::from(q), BigInt::from(p), BigInt::from(p) * BigInt::from(q)] {
                for sign in [-1, 1] {
                    for neighbor in [-1, 0, 1] {
                        ks.push(&center * BigInt::from(sign) + BigInt::from(neighbor));
                    }
                }
            }
        }
        for k in ks {
            for r in [0, 1, h - 1, h, h + 1, p - 1] {
                let whole = &k * BigInt::from(p) + BigInt::from(r);
                let quotient = &k + BigInt::from(u64::from(r > h));
                for i in 0..3 {
                    for component in 0..2 {
                        let at = (i + component) % n;
                        let mut digits = vec![0; 3 * n];
                        digits[i * n] = 1;
                        let mut rows = vec![0; 24 * n];
                        // The SAME whole integer is reduced into all four limbs.
                        for limb in 0..4 {
                            rows[((i * 2 + component) * 4 + limb) * n + at] = residue(whole.clone(), moduli[limb]);
                        }
                        let actual = check_composition(&fixture, &digits, &rows, 0);
                        let mut expected = vec![0; 6 * n];
                        for limb in 0..3 {
                            let q = moduli[limb];
                            let value = residue(quotient.clone(), q);
                            expected[(component * 3 + limb) * n + at] = value;
                            assert_eq!(rounded_crt(residue(whole.clone(), q), r, q, p), value);
                        }
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn composition_basis_routes_every_source_component_limb_and_negacyclic_wrap() {
    let n = 4;
    let fixture = Fixture::small(n);
    for i in 0..3 {
        for component in 0..2 {
            for limb in 0..4 {
                for (from, at) in [(0, 0), (1, 1), (3, 1), (3, 3)] {
                    let mut digits = vec![0; 3 * n];
                    digits[i * n + from] = fixture.sources[i].ring().modulus() - 1;
                    let mut rows = vec![0; 24 * n];
                    rows[((i * 2 + component) * 4 + limb) * n + at] = fixture.targets[limb].ring().modulus() - 1;
                    check_composition(&fixture, &digits, &rows, 0);
                }
            }
        }
    }
}

#[test]
fn composition_p_only_independent_residues_have_constructive_nonzero_retained_witness() {
    let fixture = Fixture::small(4);
    let mut digits = vec![0; 12];
    digits[0] = 1;
    let mut rows = vec![0; 96];
    rows[3 * 4] = 97; // Legal independent residues, NOT a common-W fixture.
    let actual = check_composition(&fixture, &digits, &rows, 7);
    let mut expected = vec![0; 24];
    for limb in 0..3 {
        let q = fixture.targets[limb].ring().modulus();
        let inverse = BigInt::from(193).modinv(&BigInt::from(q)).unwrap();
        expected[limb * 4] = residue(BigInt::from(96) * inverse, q);
        assert_ne!(expected[limb * 4], 0);
    }
    assert_eq!(actual, expected);
}

#[test]
fn composition_sparse_public_8192_all_coefficients_guards_and_exact_or_extra_scratch() {
    let n = 8192;
    let fixture = Fixture::new(n, [
        562949952847873, 562949952798721, 562949952700417, 562949952274433,
    ]);
    let mut digits = vec![0; 3 * n];
    for i in 0..3 {
        digits[i * n + n - 1] = fixture.sources[i].ring().modulus() - 1;
        digits[i * n + n / 2] = 1 + i as u64;
    }
    let mut rows = vec![0; 24 * n];
    for poly in 0..24 {
        rows[poly * n + 1] = fixture.targets[poly % 4].ring().modulus() - 1;
        rows[poly * n + n / 2] = 2 + poly as u64;
    }
    let coefficients = contraction_coefficients(&fixture, &digits, &rows);
    let p = fixture.targets[3].ring().modulus();
    let removed: Vec<_> = (0..2).flat_map(|component| {
        coefficients[(component * 4 + 3) * n..(component * 4 + 4) * n].iter().copied()
    }).collect();
    assert!(removed.iter().any(|&value| value != 0));
    assert!(removed.iter().any(|&value| value > (p - 1) / 2), "negative centered p residue must actually occur");
    let exact = check_composition(&fixture, &digits, &rows, 0);
    assert_eq!(exact, check_composition(&fixture, &digits, &rows, 7));
}

#[test]
fn composition_contraction_ceiling_has_independent_stage_formula_before_extra_api_freeze() {
    for n in [1, 2, 4, 8, 8192] {
        let fixture = if n <= 8 { Fixture::small(n) } else {
            Fixture::new(n, [65537, 786433, 5767169, 7340033])
        };
        let count = n as u64;
        let butterflies = (count / 2) * u64::from(n.trailing_zeros());
        // Three inverse source transforms plus twelve forward lifted transforms;
        // inverse normalization adds 3N multiplies, forward twists add 12N.
        let expected = Work {
            events: LogicalOperations {
                checks: 1024 + 27 * count + 15 * (count + 2),
                residue_checks: 42 * count,
                moves: 74 * count + 30 * butterflies,
                swaps: 15 * (count / 2),
                modular_additions: 24 * count + 15 * butterflies,
                modular_subtractions: 15 * butterflies,
                modular_multiplications: 24 * count + if n == 1 { 0 } else { 30 * butterflies + 33 * count },
                ..LogicalOperations::default()
            },
            coefficient_reductions: 12 * count,
        };
        assert_eq!(fixture.plan().work(Evaluation::Contract).unwrap(), expected);
    }
}

#[test]
fn composition_extra_quote_all_35_fields_match_independent_formulas() {
    for n in [1, 2, 4, 8, 8192] {
        for moduli in if n <= 8 {
            vec![[17, 97, 113, 193], [113, 17, 97, 193], [193, 97, 113, 17]]
        } else {
            vec![[65537, 786433, 5767169, 7340033]]
        } {
            let fixture = Fixture::new(n, moduli);
            let expected = expected_extra(n);
            assert_eq!(fixture.plan().composition_extra_work().unwrap(), expected,
                "extra quote mismatch at degree {n}, ordered moduli {moduli:?}");
            assert!(expected.fits(expected));
            assert!(!expected.fits(CompositionExtraWork::default()));
            assert_eq!(expected.inverse_ntts.exponentiations, 0);
            assert_eq!(expected.inverse_ntts.exponent_steps, 0);
            assert_eq!(expected.inverse_ntts.primality_rounds, 0);
            if n == 1 {
                assert_eq!(expected.inverse_ntts.swaps, 0);
                assert_eq!(expected.inverse_ntts.modular_additions, 0);
                assert_eq!(expected.inverse_ntts.modular_subtractions, 0);
                assert_eq!(expected.inverse_ntts.modular_multiplications, 0);
            }
        }
    }
}

// Reused, disjoint mutable views from one allocation. Every refusal snapshots
// the ENTIRE allocation, not merely passed prefixes. Immutable inputs also have
// unpassed tails. No rollback wrapper/hook is installed around production.
struct RefusalBuffers {
    words: Vec<u64>,
    output_capacity: usize,
    scratch_capacity: usize,
}

impl RefusalBuffers {
    fn new(n: usize) -> Self {
        let output_capacity = 6 * n + 10;
        let scratch_capacity = 19 * n + 12;
        Self { words: sentinels(3 + output_capacity + 5 + scratch_capacity + 4),
            output_capacity, scratch_capacity }
    }

    fn refuse(
        &mut self,
        fixture: &Fixture,
        source_contexts: &[NegacyclicNtt; 3],
        target_contexts: &[NegacyclicNtt; 4],
        source: &[u64],
        rows: &[u64],
        output_len: usize,
        scratch_len: usize,
        ceiling: CompositionWork,
        expected_error: CompositionError,
    ) {
        assert!(output_len <= self.output_capacity);
        assert!(scratch_len <= self.scratch_capacity);
        let mut source_backing = sentinels(source.len() + 5);
        source_backing[..source.len()].copy_from_slice(source);
        let mut row_backing = sentinels(rows.len() + 7);
        row_backing[..rows.len()].copy_from_slice(rows);
        let source_before = source_backing.clone();
        let rows_before = row_backing.clone();
        let before = self.words.clone();
        let result = {
            let (_, rest) = self.words.split_at_mut(3);
            let (output_backing, rest) = rest.split_at_mut(self.output_capacity);
            let (_, rest) = rest.split_at_mut(5);
            let (scratch_backing, _) = rest.split_at_mut(self.scratch_capacity);
            fixture.plan().contract_drop_into(
                SourceSpectra { contexts: source_contexts, values: &source_backing[..source.len()] },
                RowSpectra { contexts: target_contexts, values: &row_backing[..rows.len()] },
                &mut output_backing[..output_len], &mut scratch_backing[..scratch_len], ceiling,
            )
        };
        assert_eq!(result, Err(expected_error));
        assert_eq!(self.words, before, "refusal changed output/scratch/backing tail/guard");
        assert_eq!(source_backing, source_before);
        assert_eq!(row_backing, rows_before);
        assert_eq!(source, &source_before[..source.len()]);
        assert_eq!(rows, &rows_before[..rows.len()]);
    }
}

fn public_or_small(n: usize) -> Fixture {
    if n <= 8 { Fixture::small(n) } else {
        Fixture::new(n, [65537, 786433, 5767169, 7340033])
    }
}

#[test]
fn composition_isolated_lengths_small_and_public_entire_reused_buffers_unchanged() {
    for n in [4, 8192] {
        let fixture = public_or_small(n);
        let source = vec![0; 3 * n];
        let rows = vec![0; 24 * n];
        let ceiling = expected_work(n);
        let mut buffers = RefusalBuffers::new(n);
        for length in [0, 3 * n - 1, 3 * n + 1] {
            buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &vec![0; length], &rows,
                6 * n, 19 * n + 7, ceiling, CompositionError::Contraction(Error::BufferLength));
        }
        for length in [0, 24 * n - 1, 24 * n + 1] {
            buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &vec![0; length],
                6 * n, 19 * n + 7, ceiling, CompositionError::Contraction(Error::BufferLength));
        }
        for length in [0, 6 * n - 1, 6 * n + 5] {
            buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows,
                length, 19 * n + 7, ceiling, CompositionError::BufferLength);
        }
        for length in [0, 19 * n - 1] {
            buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows,
                6 * n, length, ceiling, CompositionError::BufferLength);
        }
    }
}

#[test]
fn composition_context_identity_refusals_small_and_public_including_p_root() {
    for n in [4, 8192] {
        let fixture = public_or_small(n);
        let other_degree = Fixture::new(n / 2, fixture.targets.each_ref().map(|ctx| ctx.ring().modulus()));
        let source = vec![0; 3 * n];
        let rows = vec![0; 24 * n];
        let ceiling = expected_work(n);
        let mut buffers = RefusalBuffers::new(n);
        let mut swapped_sources = fixture.sources.clone();
        swapped_sources.swap(0, 2);
        buffers.refuse(&fixture, &swapped_sources, &fixture.targets, &source, &rows,
            6 * n, 19 * n + 7, ceiling, CompositionError::Contraction(Error::ContextMismatch));
        for i in 0..3 {
            for replacement in [
                other_degree.sources[i].clone(),
                fixture.sources[(i + 1) % 3].clone(),
                NegacyclicNtt::new(fixture.sources[i].ring(),
                    power(fixture.roots[i], 3, fixture.sources[i].ring().modulus())).unwrap(),
            ] {
                let mut contexts = fixture.sources.clone();
                contexts[i] = replacement;
                buffers.refuse(&fixture, &contexts, &fixture.targets, &source, &rows,
                    6 * n, 19 * n + 7, ceiling, CompositionError::Contraction(Error::ContextMismatch));
            }
        }
        // Bad row metadata belongs on RowSpectra, not on a newly invalid plan.
        for pair in [(0, 2), (0, 3)] {
            let mut contexts = fixture.targets.clone();
            contexts.swap(pair.0, pair.1);
            buffers.refuse(&fixture, &fixture.sources, &contexts, &source, &rows,
                6 * n, 19 * n + 7, ceiling, CompositionError::Contraction(Error::ContextMismatch));
        }
        for limb in 0..4 {
            for replacement in [
                other_degree.targets[limb].clone(),
                fixture.targets[(limb + 1) % 4].clone(),
                NegacyclicNtt::new(fixture.targets[limb].ring(),
                    power(fixture.roots[limb], 3, fixture.targets[limb].ring().modulus())).unwrap(),
            ] {
                let mut contexts = fixture.targets.clone();
                contexts[limb] = replacement;
                buffers.refuse(&fixture, &fixture.sources, &contexts, &source, &rows,
                    6 * n, 19 * n + 7, ceiling, CompositionError::Contraction(Error::ContextMismatch));
            }
        }
    }
}

#[test]
fn composition_every_source_and_row_residue_refuses_including_last_word() {
    for n in [4, 8192] {
        let fixture = public_or_small(n);
        let mut buffers = RefusalBuffers::new(n);
        let ceiling = expected_work(n);
        let mut source = vec![0; 3 * n];
        let mut rows = vec![0; 24 * n];
        // Complete N=4 census; public degree checks final words independently.
        let source_positions: Vec<_> = if n == 4 { (0..3 * n).collect() } else { vec![3 * n - 1] };
        for at in source_positions {
            for bad in [fixture.sources[at / n].ring().modulus(), u64::MAX] {
                source[at] = bad;
                buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows,
                    6 * n, 19 * n + 7, ceiling, CompositionError::Contraction(Error::NonCanonical));
            }
            source[at] = 0;
        }
        let row_positions: Vec<_> = if n == 4 { (0..24 * n).collect() } else { vec![24 * n - 1] };
        for at in row_positions {
            for bad in [fixture.targets[(at / n) % 4].ring().modulus(), u64::MAX] {
                rows[at] = bad;
                buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows,
                    6 * n, 19 * n + 7, ceiling, CompositionError::Contraction(Error::NonCanonical));
            }
            rows[at] = 0;
        }
    }
}

#[test]
fn composition_each_positive_extra_and_contraction_field_one_short_independently() {
    let n = 4;
    let fixture = Fixture::small(n);
    let exact = expected_work(n);
    assert_eq!(fixture.plan().composition_extra_work().unwrap(), exact.extra);
    assert_eq!(fixture.plan().work(Evaluation::Contract).unwrap(), exact.contraction);
    let mut digits = vec![0; 3 * n];
    for i in 0..3 { digits[i * n + i] = fixture.sources[i].ring().modulus() - 1; }
    let row_coefficients = dense_rows(&fixture);
    // Exact independently specified ceilings must admit a nonzero oracle case.
    let admitted = check_composition(&fixture, &digits, &row_coefficients, 7);
    assert!(admitted.iter().any(|&value| value != 0));
    let source = fixture.source_spectra(&digits);
    let rows = fixture.row_spectra(&row_coefficients);
    let mut buffers = RefusalBuffers::new(n);
    let mut extra_fields = 0;
    let mut extra_positive = 0;
    let mut contraction_fields = 0;
    let mut contraction_positive = 0;
    macro_rules! extra_fields {
        ($group:ident; $($field:ident),+ $(,)?) => { $(
            extra_fields += 1;
            let value = exact.extra.$group.$field;
            if value > 0 {
                extra_positive += 1;
                let mut short = exact;
                short.extra.$group.$field = value - 1;
                assert!(!exact.extra.fits(short.extra), stringify!($field));
                buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows,
                    6 * n, 19 * n + 7, short, CompositionError::WorkLimit);
            } else {
                assert_eq!(value, 0, stringify!($field));
            }
        )+ };
    }
    extra_fields!(inverse_ntts; checks, residue_checks, moves, swaps, modular_additions,
        modular_subtractions, modular_multiplications, exponentiations, exponent_steps, primality_rounds);
    extra_fields!(drop; checks, residue_checks, moves, modular_subtractions,
        modular_multiplications, unsigned_reductions, ordinary_subtractions, ordinary_divisions);
    extra_fields!(construction; checks, descriptor_stores, euclid_iterations, signed_divisions,
        signed_multiplications, signed_subtractions, signed_remainders, unsigned_remainders);
    extra_fields!(control; capacity_outcomes, adapters, result_inspections, ceiling_comparisons,
        length_comparisons, dispatches, selected_loop_polls, slice_partitions, selected_phase_calls);
    macro_rules! contraction_fields {
        ($($field:ident),+ $(,)?) => { $(
            contraction_fields += 1;
            let value = exact.contraction.events.$field;
            if value > 0 {
                contraction_positive += 1;
                let mut short = exact;
                short.contraction.events.$field = value - 1;
                assert!(!exact.contraction.fits(short.contraction), stringify!($field));
                buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows,
                    6 * n, 19 * n + 7, short, CompositionError::Contraction(Error::WorkLimit));
            } else {
                assert_eq!(value, 0, stringify!($field));
            }
        )+ };
    }
    contraction_fields!(checks, residue_checks, moves, swaps, modular_additions,
        modular_subtractions, modular_multiplications, exponentiations, exponent_steps, primality_rounds);
    contraction_fields += 1;
    assert!(exact.contraction.coefficient_reductions > 0);
    contraction_positive += 1;
    let mut short = exact;
    short.contraction.coefficient_reductions -= 1;
    buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows,
        6 * n, 19 * n + 7, short, CompositionError::Contraction(Error::WorkLimit));
    assert_eq!((extra_fields, extra_positive), (35, 32));
    assert_eq!((contraction_fields, contraction_positive), (11, 8));
    assert_eq!(exact.contraction.events.exponentiations, 0);
    assert_eq!(exact.contraction.events.exponent_steps, 0);
    assert_eq!(exact.contraction.events.primality_rounds, 0);
}

#[test]
fn composition_mixed_refusal_precedence_exact_errors_and_reused_buffer_atomicity() {
    let n = 4;
    let fixture = Fixture::small(n);
    let exact = expected_work(n);
    let mut short_extra = exact;
    short_extra.extra.control.selected_phase_calls -= 1;
    let mut short_old = exact;
    short_old.contraction.events.checks -= 1;
    let mut short_both = short_extra;
    short_both.contraction = short_old.contraction;
    let mut wrong_sources = fixture.sources.clone();
    wrong_sources.swap(0, 2);
    let mut wrong_targets = fixture.targets.clone();
    wrong_targets.swap(0, 3);
    let mut source = vec![0; 3 * n];
    let mut rows = vec![0; 24 * n];
    source[0] = u64::MAX;
    source[3 * n - 1] = u64::MAX;
    rows[0] = u64::MAX;
    rows[24 * n - 1] = u64::MAX;
    let mut buffers = RefusalBuffers::new(n);
    // Outer output and scratch faults isolated from one another; each outranks
    // wrong contexts, both ceilings, bad old lengths and noncanonical values.
    for (output_len, scratch_len) in [(6 * n - 1, 19 * n + 7), (6 * n + 5, 19 * n + 7), (6 * n, 19 * n - 1)] {
        buffers.refuse(&fixture, &wrong_sources, &wrong_targets, &source[..3 * n - 1], &rows[..24 * n - 1],
            output_len, scratch_len, short_both, CompositionError::BufferLength);
    }
    // With outer geometry valid, extra ceiling wins context and old length/work.
    buffers.refuse(&fixture, &wrong_sources, &wrong_targets, &source[..3 * n - 1], &rows[..24 * n - 1],
        6 * n, 19 * n + 7, short_both, CompositionError::WorkLimit);
    for (sources, targets) in [(&wrong_sources, &fixture.targets), (&fixture.sources, &wrong_targets)] {
        buffers.refuse(&fixture, sources, targets, &source, &rows,
            6 * n, 19 * n + 7, short_extra, CompositionError::WorkLimit);
    }
    // Isolate extra-vs-input-length without a context fault masking that relation.
    buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source[..3 * n - 1], &rows,
        6 * n, 19 * n + 7, short_extra, CompositionError::WorkLimit);
    buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows[..24 * n - 1],
        6 * n, 19 * n + 7, short_extra, CompositionError::WorkLimit);
    // Valid extra ceiling: independently isolate source and row context faults.
    for (sources, targets) in [(&wrong_sources, &fixture.targets), (&fixture.sources, &wrong_targets)] {
        buffers.refuse(&fixture, sources, targets, &source[..3 * n - 1], &rows[..24 * n - 1],
            6 * n, 19 * n + 7, short_old, CompositionError::Contraction(Error::ContextMismatch));
        // Pairwise relations also stand without another earlier old-preflight
        // fault masking context-vs-work, context-vs-scan, or either length.
        buffers.refuse(&fixture, sources, targets, &source, &rows,
            6 * n, 19 * n + 7, short_old, CompositionError::Contraction(Error::ContextMismatch));
        buffers.refuse(&fixture, sources, targets, &source, &rows,
            6 * n, 19 * n + 7, exact, CompositionError::Contraction(Error::ContextMismatch));
        buffers.refuse(&fixture, sources, targets, &source[..3 * n - 1], &rows,
            6 * n, 19 * n + 7, exact, CompositionError::Contraction(Error::ContextMismatch));
        buffers.refuse(&fixture, sources, targets, &source, &rows[..24 * n - 1],
            6 * n, 19 * n + 7, exact, CompositionError::Contraction(Error::ContextMismatch));
    }
    // Valid contexts: source and row length faults each beat old ceiling/scan.
    buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source[..3 * n - 1], &rows,
        6 * n, 19 * n + 7, short_old, CompositionError::Contraction(Error::BufferLength));
    buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows[..24 * n - 1],
        6 * n, 19 * n + 7, short_old, CompositionError::Contraction(Error::BufferLength));
    buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source[..3 * n - 1], &rows,
        6 * n, 19 * n + 7, exact, CompositionError::Contraction(Error::BufferLength));
    buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows[..24 * n - 1],
        6 * n, 19 * n + 7, exact, CompositionError::Contraction(Error::BufferLength));
    // Valid lengths: old ceiling beats noncanonical source AND row.
    buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows,
        6 * n, 19 * n + 7, short_old, CompositionError::Contraction(Error::WorkLimit));
    // Both ceilings valid: combined source+row noncanonical faults are wrapped.
    buffers.refuse(&fixture, &fixture.sources, &fixture.targets, &source, &rows,
        6 * n, 19 * n + 7, exact, CompositionError::Contraction(Error::NonCanonical));
    // Valid plans have same-degree distinct prime targets, so public callers
    // cannot force InvalidDegree/NotCoprime constructor errors. Storage/quote
    // overflow and DropConstruction(LayoutError::Overflow) remain defensive
    // variants: no forged descriptor or unreachable overflow witness. Exact
    // constructor error mapping / all-three-before-write is static-review only.
}
