//! Source-authored only; UNRUN. Synthetic public rows, no authentic-key claim.
//! Expected products: independent BigInt signed schoolbook negacyclic algebra;
//! expected trace: existing independent global-CRT/recursive-tree fixture.
use super::*;
use crate::coefficient_rns_contraction::{Olt, OltLhsSpectra, OltWork};
use crate::coefficient_layout::coordinates::Coordinates;

struct CompositionFixture {
    trace: Fixture,
    shape: [usize; 3],
    window: [usize; 3],
    blocks: [usize; 3],
    lhs_coefficients: Vec<u64>,
    lhs_spectra: Vec<u64>,
    rhs: Vec<u64>,
}

impl CompositionFixture {
    fn new(n: usize, shape: [usize; 3], window: [usize; 3], direction: u64) -> Self {
        let blocks: [usize; 3] = core::array::from_fn(|i| shape[i].div_ceil(window[i]));
        let count = blocks[0] * blocks[2];
        let trace = Fixture::new(n, window[1], count, [97, 193, 257, 353], direction, true);
        // Arbitrary canonical component polynomials, not an assumed plaintext
        // layout/phase. Dense support exercises coefficient/spectral transitions
        // and forbids selected-cell cropping. Components and inner blocks differ.
        let lhs_coefficients: Vec<_> = (0..blocks[0] * blocks[1] * 6 * n).map(|at| {
            let poly = at / n;
            let q = trace.q[poly % 3];
            (q - 1 - ((at * 7 + poly * 11 + 3) as u64 % q)) % q
        }).collect();
        let lhs_spectra = lhs_coefficients.chunks_exact(n).enumerate().flat_map(|(poly, a)| {
            direct_dft(a, trace.roots[poly % 3], trace.q[poly % 3])
        }).collect();
        let full_width = [u64::MAX, 1_u64 << 63, 0, 7, (1_u64 << 63) + 19, 1];
        let rhs = (0..shape[1] * shape[2]).map(|i| full_width[i % full_width.len()]).collect();
        let mut result = Self { trace, shape, window, blocks, lhs_coefficients, lhs_spectra, rhs };
        result.trace.input = result.products(false, false, false);
        result
    }

    /// NO Coordinates, NTT, production contraction or schedule in product oracle.
    /// `signed_rhs` deliberately substitutes a WRONG int64 interpretation.
    fn products(&self, signed_rhs: bool, omit_last_inner: bool, zero_c1: bool) -> Vec<u64> {
        let n = self.trace.n;
        let mut result = Vec::new();
        for row_block in 0..self.blocks[0] {
            for col_block in 0..self.blocks[2] {
                let mut lane = vec![BigInt::from(0); 6 * n];
                for inner in 0..self.blocks[1] {
                    if omit_last_inner && inner + 1 == self.blocks[1] { continue; }
                    let mut packed = vec![BigInt::from(0); n];
                    for h in 0..self.window[1] {
                        for c in 0..self.window[2] {
                            let global_h = inner * self.window[1] + h;
                            let global_c = col_block * self.window[2] + c;
                            if global_h < self.shape[1] && global_c < self.shape[2] {
                                let value = self.rhs[global_h * self.shape[2] + global_c];
                                let value = BigInt::from(value) - if signed_rhs && value >= 1_u64 << 63 {
                                    BigInt::from(1_u64) << 64_usize
                                } else { BigInt::from(0) };
                                packed[c * self.window[1] + h] = value;
                            }
                        }
                    }
                    for poly in 0..6 {
                        if zero_c1 && poly >= 3 { continue; }
                        let at = ((row_block * self.blocks[1] + inner) * 6 + poly) * n;
                        // Commutative schoolbook accepts BigInt first argument.
                        let product = schoolbook(&packed, &self.lhs_coefficients[at..at + n]);
                        for j in 0..n { lane[poly * n + j] += &product[j]; }
                    }
                }
                result.extend(lane.into_iter().enumerate().map(|(at, v)| residue(v, self.trace.q[(at / n) % 3])));
            }
        }
        result
    }

    fn rows(&self) -> Vec<SwitchBinding<'_>> {
        self.trace.exponents.iter().enumerate().map(|(i, &exponent)| SwitchBinding {
            exponent, rows: RowSpectra { contexts: &self.trace.targets, values: &self.trace.row_spectra[i] },
        }).collect()
    }

    fn run(&self) -> Vec<u64> {
        let contraction = RnsContraction::new(&self.trace.sources, &self.trace.targets).unwrap();
        let plan = Olt::new(&contraction, self.shape, self.window).unwrap();
        let st = plan.storage().unwrap();
        let mut output = vec![987; st.output_words + 7];
        let mut scratch = vec![654; st.scratch_words + 9];
        let lhs = OltLhsSpectra { contexts: &self.trace.sources, values: &self.lhs_spectra };
        let rows = self.rows();
        let census = plan.census(&lhs, &self.rhs, &rows, &output, &scratch).unwrap();
        assert_eq!(census.total_words, st.total_words + 16);
        assert_eq!(census.total_bytes, 8 * census.total_words);
        plan.apply(lhs, &self.rhs, &rows, &mut output, &mut scratch, plan.work().unwrap()).unwrap();
        assert!(output[st.output_words..].iter().all(|&v| v == 987));
        assert!(scratch[st.scratch_words..].iter().all(|&v| v == 654));
        output.truncate(st.output_words);
        output
    }
}

#[test]
fn public_composition_independent_both_components_all_inner_blocks_full_width_partials() {
    for direction in [1, 2] {
        // Non-power K=3, P=4, stride=8: padded trace then every merge level.
        // Last inner block and last column block are partial.
        let mut f = CompositionFixture::new(16, [1, 9, 5], [1, 8, 2], direction);
        let expected = f.trace.oracle(8, 3, Mutation::None);
        assert_eq!(f.run(), expected);
        let original = f.trace.input.clone();
        for products in [f.products(true, false, false), f.products(false, true, false),
            f.products(false, false, true)]
        {
            assert_ne!(products, original);
            f.trace.input = products;
            assert_ne!(f.trace.oracle(8, 3, Mutation::None), expected);
        }
        f.trace.input = original;
        assert_ne!(f.trace.oracle(8, 3, Mutation::OmitNormalization), expected);
        assert_ne!(f.trace.oracle(8, 3, Mutation::NoErrors), expected);
        // Every unique exponent has distinct synthetic payload and affects
        // output. These are arithmetic witnesses, NOT authentic evaluation keys.
        for &g in &f.trace.exponents {
            let other = *f.trace.exponents.iter().find(|&&e| e != g).unwrap();
            assert_ne!(f.trace.oracle(8, 3, Mutation::WrongRow { exponent: g, replacement: other }), expected);
        }
        // No cropping: full 6N output, not a selected matrix-cell facade.
        assert_eq!(expected.len(), 6 * 16);
        let coordinates = Coordinates::bounded(16, f.shape, f.window, 8192).unwrap();
        assert!(expected.iter().enumerate().any(|(at, &v)|
            !coordinates.visible(0, at % 16).unwrap() && v != 0));
    }
}

#[test]
fn public_composition_no_switch_trace_only_merge_only_and_row_repacking() {
    for (shape, window) in [
        ([1, 3, 1], [1, 1, 1]), // mw=1, K=P=1: no rows or switches.
        ([1, 5, 1], [1, 4, 1]), // P=1, mw>1: trace only.
        ([2, 5, 2], [1, 4, 1]), // K=mw=4: merge only, multiple row blocks.
        ([1, 5, 3], [1, 4, 1]), // nonpower K=3, P=mw=4.
    ] {
        let f = CompositionFixture::new(8, shape, window, 1);
        let count = f.blocks[0] * f.blocks[2];
        assert_eq!(f.run(), f.trace.oracle(window[1], count, Mutation::None));
        let contraction = RnsContraction::new(&f.trace.sources, &f.trace.targets).unwrap();
        let plan = Olt::new(&contraction, shape, window).unwrap();
        let st = plan.storage().unwrap();
        assert_eq!(st.scratch_words, (6 * count.next_power_of_two() + 51) * 8);
        assert_eq!(st.unique_exponents, window[1].trailing_zeros() as usize);
        assert_eq!(plan.required_exponent_count(), st.unique_exponents);
        let inventory: std::collections::BTreeSet<_> = (0..st.unique_exponents)
            .map(|level| plan.required_exponent(level).unwrap()).collect();
        assert_eq!(inventory, f.trace.exponents.iter().copied().collect());
        assert_eq!(plan.required_exponent(st.unique_exponents), Err(TraceError::Exponent));
        assert_eq!(plan.required_exponent(usize::MAX), Err(TraceError::Exponent));
        assert_eq!(st.unique_row_words, st.unique_exponents * 24 * 8);
        assert_eq!(st.scheduled_row_reference_words, st.switches * 24 * 8);
        if window[1] == 1 {
            assert!(f.rows().is_empty());
            assert_eq!(st.switches, 0);
            let unused = vec![0; 24 * 8];
            let extra = [SwitchBinding { exponent: 9, rows: RowSpectra {
                contexts: &f.trace.targets, values: &unused } }];
            refuse(&plan, OltLhsSpectra { contexts: &f.trace.sources, values: &f.lhs_spectra },
                &f.rhs, &extra, plan.work().unwrap(), TraceError::Length, false, false);
        }
    }
    let scalar = CompositionFixture::new(1, [1, 3, 1], [1, 1, 1], 1);
    assert_eq!(scalar.run(), scalar.trace.oracle(1, 1, Mutation::None));
}

fn refuse(plan: &Olt<'_>, lhs: OltLhsSpectra<'_>, rhs: &[u64], rows: &[SwitchBinding<'_>],
    ceiling: OltWork, expected: TraceError, output_short: bool, scratch_short: bool)
{
    let st = plan.storage().unwrap();
    let mut output = vec![u64::MAX - 17; if output_short { st.output_words - 1 } else { st.output_words + 7 }];
    let mut scratch = vec![u64::MAX - 23; if scratch_short { st.scratch_words - 1 } else { st.scratch_words + 9 }];
    let original_output = output.clone();
    let original_scratch = scratch.clone();
    assert_eq!(plan.apply(lhs, rhs, rows, &mut output, &mut scratch, ceiling), Err(expected));
    assert_eq!(output, original_output);
    assert_eq!(scratch, original_scratch);
}

#[test]
fn public_composition_refusals_preserve_every_mutable_word_and_tails() {
    let f = CompositionFixture::new(16, [1, 9, 5], [1, 8, 2], 1);
    let contraction = RnsContraction::new(&f.trace.sources, &f.trace.targets).unwrap();
    let plan = Olt::new(&contraction, f.shape, f.window).unwrap();
    let lhs = || OltLhsSpectra { contexts: &f.trace.sources, values: &f.lhs_spectra };
    let work = plan.work().unwrap();
    let rows = f.rows();
    refuse(&plan, lhs(), &f.rhs, &rows, work, TraceError::Length, true, false);
    refuse(&plan, lhs(), &f.rhs, &rows, work, TraceError::Length, false, true);
    refuse(&plan, lhs(), &f.rhs[..f.rhs.len() - 1], &rows, work, TraceError::Length, false, false);
    refuse(&plan, OltLhsSpectra { contexts: &f.trace.sources, values: &f.lhs_spectra[1..] },
        &f.rhs, &rows, work, TraceError::Length, false, false);
    let mut noncanonical = f.lhs_spectra.clone();
    *noncanonical.last_mut().unwrap() = f.trace.q[2];
    refuse(&plan, OltLhsSpectra { contexts: &f.trace.sources, values: &noncanonical },
        &f.rhs, &rows, work, TraceError::NonCanonical, false, false);
    let mut wrong_sources = f.trace.sources.clone();
    wrong_sources.swap(0, 1);
    refuse(&plan, OltLhsSpectra { contexts: &wrong_sources, values: &f.lhs_spectra },
        &f.rhs, &rows, work, TraceError::Context, false, false);
    wrong_sources = f.trace.sources.clone();
    wrong_sources[0] = NegacyclicNtt::new(wrong_sources[0].ring(),
        power(f.trace.roots[0], 3, f.trace.q[0])).unwrap();
    refuse(&plan, OltLhsSpectra { contexts: &wrong_sources, values: &f.lhs_spectra },
        &f.rhs, &rows, work, TraceError::Context, false, false);
    let mut wrong_targets = f.trace.targets.clone();
    wrong_targets.swap(0, 1);
    let mut root_targets = f.trace.targets.clone();
    root_targets[3] = NegacyclicNtt::new(root_targets[3].ring(),
        power(f.trace.roots[3], 3, f.trace.q[3])).unwrap();
    for contexts in [&wrong_targets, &root_targets] {
        let mut bad = f.rows();
        bad.last_mut().unwrap().rows.contexts = contexts;
        refuse(&plan, lhs(), &f.rhs, &bad, work, TraceError::Context, false, false);
    }
    assert!(matches!(RnsContraction::new(&f.trace.sources, &wrong_targets), Err(Error::ContextMismatch)));
    let mut duplicate_targets = f.trace.targets.clone();
    duplicate_targets[3] = duplicate_targets[0].clone();
    assert!(matches!(RnsContraction::new(&f.trace.sources, &duplicate_targets), Err(Error::ContextMismatch)));
    assert!(matches!(RnsContraction::new(&wrong_sources, &f.trace.targets), Err(Error::ContextMismatch)));
    let mut bad_values = rows.last().unwrap().rows.values.to_vec();
    *bad_values.last_mut().unwrap() = f.trace.q[3];
    let mut bad = f.rows();
    bad.last_mut().unwrap().rows.values = &bad_values;
    refuse(&plan, lhs(), &f.rhs, &bad, work, TraceError::NonCanonical, false, false);
    bad = f.rows();
    bad.last_mut().unwrap().rows.values = &rows.last().unwrap().rows.values[1..];
    refuse(&plan, lhs(), &f.rhs, &bad, work, TraceError::Length, false, false);
    bad = f.rows();
    bad[1].exponent = bad[0].exponent;
    refuse(&plan, lhs(), &f.rhs, &bad, work, TraceError::Exponent, false, false);
    bad = f.rows();
    bad[0].exponent = 1; // unused inventory entry, despite valid automorphism.
    refuse(&plan, lhs(), &f.rhs, &bad, work, TraceError::Exponent, false, false);
    refuse(&plan, lhs(), &f.rhs, &rows[..rows.len() - 1], work, TraceError::Length, false, false);
    bad = f.rows();
    bad.push(SwitchBinding { exponent: 1, rows: RowSpectra {
        contexts: &f.trace.targets, values: rows[0].rows.values } });
    refuse(&plan, lhs(), &f.rhs, &bad, work, TraceError::Length, false, false);
    // Inventory order is irrelevant; exact exponent selection is not positional.
    let mut reversed = f.rows();
    reversed.reverse();
    let st = plan.storage().unwrap();
    let mut out = vec![0; st.output_words];
    let mut scratch = vec![0; st.scratch_words];
    plan.apply(lhs(), &f.rhs, &reversed, &mut out, &mut scratch, work).unwrap();
    assert_eq!(out, f.trace.oracle(8, 3, Mutation::None));
}

#[test]
fn public_composition_every_positive_work_field_one_short_is_atomic() {
    let f = CompositionFixture::new(16, [1, 9, 5], [1, 8, 2], 1);
    let contraction = RnsContraction::new(&f.trace.sources, &f.trace.targets).unwrap();
    let plan = Olt::new(&contraction, f.shape, f.window).unwrap();
    let work = plan.work().unwrap();
    let mut ceilings = Vec::new();
    macro_rules! short {
        ($($field:ident).+) => {
            if work.$($field).+ != 0 {
                let mut v = work;
                v.$($field).+ -= 1;
                ceilings.push(v);
            }
        };
    }
    short!(forward_ntts); short!(rhs_forward_ntts); short!(inverse_ntts); short!(pointwise_products);
    short!(packing_stores); short!(ring_negations); short!(coordinate_visits);
    short!(lane_copies); short!(spectral_accumulations); short!(control); short!(indices);
    short!(outer.coefficient_reductions);
    short!(outer.events.checks); short!(outer.events.residue_checks);
    short!(outer.events.moves); short!(outer.events.swaps);
    short!(outer.events.modular_additions); short!(outer.events.modular_subtractions);
    short!(outer.events.modular_multiplications); short!(outer.events.exponentiations);
    short!(outer.events.exponent_steps); short!(outer.events.primality_rounds);
    short!(trace.switches); short!(trace.control); short!(trace.indices);
    for index in 0..92 {
        if work.trace.galois[index] != 0 {
            let mut v = work; v.trace.galois[index] -= 1; ceilings.push(v);
        }
    }
    for index in 0..10 {
        if work.trace.forwards[index] != 0 {
            let mut v = work; v.trace.forwards[index] -= 1; ceilings.push(v);
        }
        if work.trace.wrapper[index] != 0 {
            let mut v = work; v.trace.wrapper[index] -= 1; ceilings.push(v);
        }
    }
    for index in 0..8 {
        if work.trace.preflight_construction[index] != 0 {
            let mut v = work; v.trace.preflight_construction[index] -= 1; ceilings.push(v);
        }
    }
    for ceiling in ceilings {
        refuse(&plan, OltLhsSpectra { contexts: &f.trace.sources, values: &f.lhs_spectra },
            &f.rhs, &f.rows(), ceiling, TraceError::WorkLimit, false, false);
    }
    assert_eq!(f.run(), f.trace.oracle(8, 3, Mutation::None));
}

#[test]
fn public_composition_geometry_and_checked_scalar_capacity_refusals() {
    let f = CompositionFixture::new(8, [1, 3, 1], [1, 1, 1], 1);
    let contraction = RnsContraction::new(&f.trace.sources, &f.trace.targets).unwrap();
    for (shape, window) in [([3, 1, 3], [1, 4, 1]), ([0, 1, 1], [1, 1, 1]),
        ([1, 1, 1], [1, 3, 1]), ([1, 1, 1], [2, 8, 1])]
    {
        assert!(Olt::new(&contraction, shape, window).is_err());
    }
    assert!(matches!(Olt::new(&contraction, [usize::MAX, 2, 2], [1, 1, 1]),
        Err(TraceError::Geometry(LayoutError::Overflow))));
    // Shape products can fit while 6N spectral payload/byte capacity overflows.
    assert!(matches!(Olt::new(&contraction, [1, usize::MAX, 1], [1, 1, 1]),
        Err(TraceError::Capacity)));
    assert_eq!(crate::coefficient_rns_contraction::olt_trace::times(u64::MAX, 2), Err(TraceError::Capacity));
    assert_eq!(crate::coefficient_rns_contraction::olt_trace::add(u64::MAX, 1), Err(TraceError::Capacity));
    let mut work = Work::default();
    assert_eq!(work.add_scaled_ntt(LogicalOperations { moves: u64::MAX,
        ..LogicalOperations::default() }, 2), Err(Error::ArithmeticCapacity));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn public_composition_scalar_control_narrowing_refuses_without_large_allocations() {
    let f = CompositionFixture::new(1, [1, 1, 1], [1, 1, 1], 1);
    let contraction = RnsContraction::new(&f.trace.sources, &f.trace.targets).unwrap();
    let length = usize::MAX / 100;
    let l = length as u64;
    let shape = [1, length, 1];
    let window = [1, 1, 1];
    assert!(Coordinates::bounded(1, shape, window, 8192).is_ok());
    // N=K=P=1, C=E=0: storage and numeric fields fit; only control narrows out.
    let storage_bytes = length.checked_mul(56).unwrap().checked_add(504).unwrap();
    assert_eq!(storage_bytes as u128, 56*u128::from(l)+504);
    let mut numeric = Work::default();
    for context in &f.trace.sources {
        numeric.add_scaled_ntt(context.operations(Operation::Forward), l).unwrap();
        numeric.add_scaled_ntt(context.operations(Operation::Inverse), 2).unwrap();
    }
    numeric.add_ntt(LogicalOperations {
        checks: 6*l, residue_checks: 6*l, moves: 17*l+12,
        modular_multiplications: 6*l, modular_additions: 6*l,
        ..LogicalOperations::default()
    }).unwrap();
    numeric.coefficient_reductions = 3*l;
    assert!(u64::try_from(96*u128::from(l)+129).is_ok());
    assert!(u64::try_from(298*u128::from(l)+5998).is_err());
    assert!(matches!(Olt::new(&contraction, shape, window), Err(TraceError::Capacity)));
}

#[test]
fn public_composition_finite_incremental_wrapper_formulas_and_scalar_fixed_costs() {
    for (n, shape, window) in [
        (1, [1, 1, 1], [1, 1, 1]),
        (8, [1, 3, 1], [1, 1, 1]),
        (8, [1, 5, 1], [1, 4, 1]),
        (8, [2, 5, 2], [1, 4, 1]),
        (16, [1, 9, 5], [1, 8, 2]),
    ] {
        let f = CompositionFixture::new(n, shape, window, 1);
        let contraction = RnsContraction::new(&f.trace.sources, &f.trace.targets).unwrap();
        let plan = Olt::new(&contraction, shape, window).unwrap();
        let work = plan.work().unwrap();
        let k = (f.blocks[0]*f.blocks[2]) as u64;
        let a = (f.blocks[0]*f.blocks[1]) as u64;
        let b = k*f.blocks[1] as u64;
        let v = (f.blocks[0]*shape[1]*shape[2]) as u64;
        let e = u64::from(window[1].trailing_zeros());
        let degree = n as u64;
        assert_eq!(work.control, 5397+7*e+72*a+30*a*degree+65*k
            +118*b+51*b*degree+27*v+314+222);
        assert_eq!(work.indices, 91+e+6*a+38*k+65*b+18*b*degree+7*v);
        // Outer fields are incremental, not a second copy of nested trace.
        let trace = TracePlan::new(&contraction, window[1], k as usize).unwrap();
        assert_eq!(work.trace, trace.work().unwrap());
        if n == 1 {
            assert_eq!((a, b, v, e, work.trace.switches), (1, 1, 1, 0, 0));
            // Independently: 5397+72+30+65+118+51+27+314+222;
            // indices 91+6+38+65+18+7. Nested quote stays separate.
            assert_eq!((work.control, work.indices), (6296, 225));
            assert_eq!((work.trace.control, work.trace.indices), (8262, 130));
            let mut control_short = work;
            control_short.control -= 1;
            let mut indices_short = work;
            indices_short.indices -= 1;
            let mut trace_control_short = work;
            trace_control_short.trace.control -= 1;
            let mut trace_indices_short = work;
            trace_indices_short.trace.indices -= 1;
            for ceiling in [control_short, indices_short, trace_control_short, trace_indices_short] {
                refuse(&plan, OltLhsSpectra { contexts: &f.trace.sources, values: &f.lhs_spectra },
                    &f.rhs, &f.rows(), ceiling, TraceError::WorkLimit, false, false);
            }
        }
        assert_eq!(f.run(), f.trace.oracle(window[1], k as usize, Mutation::None));
    }
}

#[test]
fn selected_geometry_387_distinct_addresses_and_padded_schedule_without_crypto_profile() {
    let coordinates = Coordinates::bounded(8192, [3, 65, 129], [1, 64, 128], 8192).unwrap();
    let mut addresses = std::collections::BTreeSet::new();
    for row in 0..3 {
        for column in 0..129 {
            let (group, coefficient) = coordinates.address(row, column).unwrap();
            assert_eq!(group, 0);
            assert!(coefficient < 8192);
            assert!(coordinates.visible(group, coefficient).unwrap());
            assert!(addresses.insert((group, coefficient)));
        }
    }
    assert_eq!(addresses.len(), 387);
    let schedule = InterleaveSchedule::bounded(8192, 64, 6, 8192).unwrap();
    let steps: Vec<_> = schedule.steps().collect();
    assert_eq!(schedule.padded(), 8);
    assert_eq!(steps.iter().filter(|s| matches!(s, InterleaveStep::Trace { .. })).count(), 24);
    assert_eq!(steps.iter().filter(|s| matches!(s, InterleaveStep::Merge { .. })).count(), 7);
    for lane in [6, 7] {
        assert_eq!(steps.iter().filter(|s| matches!(s, InterleaveStep::Trace { lane: actual, .. } if *actual == lane)).count(), 3);
    }
}

#[test]
#[ignore = "reuses existing unadopted N8192 arithmetic tuple; source-authored only, UNRUN"]
fn selected_geometry_storage_and_nested_call_quotes_reuse_existing_tuple() {
    // EXACT existing olt_trace_tests arithmetic fixture; NOT a new selected
    // crypto/security profile, parameter adoption, or executed validation.
    let f = Fixture::new(8192, 64, 6,
        [562949952847873, 562949952798721, 562949952700417, 562949952274433], 1, true);
    let contraction = RnsContraction::new(&f.sources, &f.targets).unwrap();
    let plan = Olt::new(&contraction, [3, 65, 129], [1, 64, 128]).unwrap();
    let st = plan.storage().unwrap();
    let work = plan.work().unwrap();
    assert_eq!(st.lhs_words, 36 * 8192); // six ciphertext-shaped public inputs.
    assert_eq!(st.rhs_words, 8385);
    assert_eq!(st.unique_exponents, 6);
    assert_eq!(st.unique_row_words, 144 * 8192);
    assert_eq!(st.scheduled_row_reference_words, 31 * 24 * 8192);
    assert_eq!(st.output_words, 6 * 8192);
    assert_eq!(st.scratch_words, 99 * 8192);
    assert_eq!(st.total_words, (36 + 144 + 6 + 99) * 8192 + 8385);
    assert_eq!(work.pointwise_products, 72);
    assert_eq!(work.rhs_forward_ntts, 36); // per-output-lane RHS recomputation.
    assert_eq!(work.forward_ntts, 222);
    assert_eq!(work.inverse_ntts, 36);
    assert_eq!(work.trace.switches, 31);
    assert_eq!(work.rhs_forward_ntts + 6 * work.trace.switches, 222);
    // Existing Galois composition: twelve target forwards, fourteen inverses
    // per switch (3 digit + 8 rounded-drop + 3 original c0). Invocation census,
    // not instrumentation, and never a substitute for owner fieldwise work.
    assert_eq!(work.forward_ntts + 12 * work.trace.switches, 594);
    assert_eq!(work.inverse_ntts + 14 * work.trace.switches, 470);
    let contexts = &f.sources;
    let mut expected = Work::default();
    for context in contexts {
        expected.add_scaled_ntt(context.operations(Operation::Forward), 12).unwrap();
        expected.add_scaled_ntt(context.operations(Operation::Inverse), 12).unwrap();
    }
    assert!(work.outer.events.moves >= expected.events.moves);
    assert!(work.outer.events.modular_multiplications >= expected.events.modular_multiplications);
}
