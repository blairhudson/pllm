//! Inactive native cut only. Reconstruction/dealer belongs to these test
//! fixtures; production still holds one replicated share per party/worker.
use super::*;
use crate::puma::multiply::arithmetic::{IndexedAffine, IndexedAffineRow, IndexedTerm};
use crate::puma::multiply::tensor::attention_work::{IndexedWorkWorkspace, SoftmaxSink};
use crate::puma::multiply::tensor::predicate_provenance::{
    self, Audit, Fault, IndexedInventory, IndexedSink, Rule, CAPACITY,
};

const PIVOTS: [i64; 3] = [-2_097_152, -1_048_576, 1_048_577];

#[test]
fn indexed_sparse_chain_full_node_capacity_caches_and_allocations() {
    let bounds = Interval::new(-9, 9).unwrap();
    let mut native = Builder::indexed(1, bounds).unwrap();
    let mut metadata = IndexedWorkWorkspace::new(1, bounds).unwrap();
    let scope = crate::exact_buffer::AllocationScope::new(false);
    let mut value = Value(0);
    let mut maps = 0;
    for i in 1..MAX_NODES {
        let next = if i % 3 == 0 {
            let coefficients = Affine { left: 1, right: 0, constant: 0 };
            let a = native.affine(value, None, coefficients).unwrap();
            assert_eq!(a, metadata.affine(value, None, coefficients).unwrap());
            a
        } else {
            maps += 1;
            let rows = [row(0, &[(0, 1)])];
            let a = native.indexed_affine(value, &rows).unwrap();
            assert_eq!(a, metadata.indexed_affine(value, &rows).unwrap());
            a
        };
        value = next;
    }
    assert_eq!(scope.entries(), 2 * maps);
    let program = native.finish(value).unwrap();
    assert_eq!(program.len, 192);
    assert_eq!(metadata.finish(value).unwrap(), program.work_budget());
    metadata.assert_native_metadata(&program);
    assert_eq!(scope.entries(), 2 * maps); // cache/finish have no acquisition
    let mut audit = Audit::default();
    predicate_provenance::validate(&program, &mut audit).unwrap();
    assert!(audit.visits <= predicate_provenance::REPAIRED_VALIDATION_BOUND);
    assert!(predicate_provenance::REPAIRED_VALIDATION_BOUND < 806_701_248);
    assert_eq!(program.work_budget().operations(), 191);
    let cloned = program.clone();
    assert_eq!(scope.entries(), 3 * maps); // no proof or next-wrapper allocation
    let rebuilt = Builder(cloned).finish(value).unwrap();
    assert_eq!(rebuilt.digest(), program.digest());
    assert_eq!(scope.entries(), 3 * maps);
    let mut parties = owners(&program, &[7]);
    let usage = parties.each_ref().map(Executor::payload_usage);
    for _ in 1..MAX_NODES { step(&mut parties); }
    let output = parties.map(|p| p.finish().unwrap().1);
    assert_eq!(clear(&output), [7]);
    drop(output);
    assert!(usage.iter().all(|u| u.live_bytes() == 0));
}

#[test]
fn indexed_sparse_multiple_nested_maps_mask_only_and_clone_exact_m_plus_q() {
    let bounds = Interval::new(-3_000_000, 3_000_000).unwrap();
    let (offsets, mask) = descriptors(2, PIVOTS);
    let mut native = Builder::indexed(2, bounds).unwrap();
    let mut metadata = IndexedWorkWorkspace::new(2, bounds).unwrap();
    let scope = crate::exact_buffer::AllocationScope::new(false);
    let mut output = Value(0);
    for _ in 0..2 {
        output = emit(&mut native, &offsets, &mask).unwrap();
        assert_eq!(output, emit(&mut metadata, &offsets, &mask).unwrap());
    }
    assert_eq!(scope.entries(), 12); // each sink m=4, q=2, no mask allocation
    let program = native.finish(output).unwrap();
    assert_eq!(metadata.finish(output).unwrap(), program.work_budget());
    metadata.assert_native_metadata(&program);
    assert_eq!(scope.entries(), 12);
    let cloned = program.clone();
    assert_eq!(scope.entries(), 18);
    let old = program.extension.owners().unwrap();
    let new = cloned.extension.owners().unwrap();
    for i in [1, 4, 6, 9] {
        assert_eq!(old.table(i), new.table(i));
        assert_ne!(old.table(i).as_ptr(), new.table(i).as_ptr());
    }
    for i in [4, 9] {
        assert_eq!(old.proofs(i), new.proofs(i));
        assert_ne!(old.proofs(i).as_ptr(), new.proofs(i).as_ptr());
    }
    for i in [5, 10] { assert!(old.table(i).is_empty()); assert!(old.rules[i].is_some()); }
    assert_eq!(Builder(cloned).finish(output).unwrap().digest(), program.digest());
    assert_eq!(scope.entries(), 18);
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Program>();
}

#[test]
fn indexed_sparse_physical_orphans_duplicates_missing_incomplete_and_proof_counts_refuse() {
    let original = resident(2, Interval::new(-3_000_000, 3_000_000).unwrap(), PIVOTS).unwrap();
    for corruption in 0..9 {
        let mut program = original.clone();
        let owners = program.extension.owners_mut().unwrap();
        match corruption {
            0 => { let rows = owners.table(1).to_vec(); owners.insert_record_for_test(Value(1), &rows); },
            1 => { let rows = owners.table(1).to_vec(); owners.insert_record_for_test(Value(191), &rows); },
            2 => *owners.records.find_mut(Value(1)).unwrap().identity_mut() = Value(2),
            3 => { drop(owners.records.remove(Value(1)).unwrap()); },
            4 => *owners.records.find_mut(Value(1)).unwrap().copied_mut() -= 1,
            5 => *owners.records.find_mut(Value(4)).unwrap().proofs_slot() = Box::default(),
            6 => owners.backing_bytes += 1,
            7 => owners.rules[191] = Some(Rule::NestedDifference { packed: Value(3) }),
            _ => owners.rules[4] = None,
        }
        assert!(predicate_provenance::validate(&program, &mut Audit::default()).is_err());
        assert!(program.resources().is_err());
        // Cloned corruption never aliases or retires the original owner.
        predicate_provenance::validate(&original, &mut Audit::default()).unwrap();
    }
}

fn row(constant: i64, terms: &[(usize, i64)]) -> IndexedAffineRow {
    assert!(terms.len() <= 2);
    let mut row = IndexedAffineRow { constant, len: terms.len() as u8,
        terms: [IndexedTerm::default(); 2] };
    for (slot, &(lane, coefficient)) in row.terms.iter_mut().zip(terms) {
        *slot = IndexedTerm { lane, coefficient };
    }
    row
}

fn descriptors(n: usize, thresholds: [i64; 3]) -> (Vec<IndexedAffineRow>, Vec<IndexedAffineRow>) {
    let offsets = thresholds.into_iter().flat_map(|t| (0..n).map(move |j| row(-t, &[(j, 1)]))).collect();
    let mask = (0..n).map(|j| row(0, &[(2 * n + j, 1), (j, -1)])).collect();
    (offsets, mask)
}

fn emit<S: IndexedSink>(b: &mut S, offsets: &[IndexedAffineRow], mask: &[IndexedAffineRow]) -> Result<Value, Error> {
    let x = b.input();
    let offset = b.indexed_affine(x, offsets)?;
    let sign = b.sign(offset)?;
    let packed = b.from_bit(sign)?;
    let inside = b.nested_difference(packed, mask)?;
    b.masked_operand(x, inside)
}

fn resident(n: usize, bounds: Interval, thresholds: [i64; 3]) -> Result<Program, Error> {
    let (offsets, mask) = descriptors(n, thresholds);
    let mut b = ResidentBuilder::from_builder(Builder::indexed(n, bounds)?);
    let offset = b.indexed_affine(Value(0), &offsets)?;
    let sign = b.predicate_sign(offset)?;
    let packed = b.predicate_from_bit(sign)?;
    let inside = b.nested_difference(packed, &mask)?;
    let safe = b.masked_operand(Value(0), inside)?;
    b.finish(safe)?;
    let mut out = None;
    b.transfer_complete(&mut out)?;
    Ok(out.unwrap())
}

fn through_b2a(n: usize) -> Builder {
    let (offsets, _) = descriptors(n, PIVOTS);
    let mut b = Builder::indexed(n, Interval::new(-3_000_000, 3_000_000).unwrap()).unwrap();
    let offset = b.indexed_affine(Value(0), &offsets).unwrap();
    let sign = b.sign(offset).unwrap();
    b.from_bit(sign).unwrap();
    b
}

// Independent mathematical oracle: no descriptor, interval, native arithmetic,
// activation implementation, proof checker, or secret membership assertion.
fn oracle(values: &[i64], thresholds: [i64; 3]) -> [Vec<i64>; 5] {
    let offsets: Vec<_> = thresholds.into_iter().flat_map(|t| values.iter()
        .map(move |&x| i64::try_from(i128::from(x) - i128::from(t)).unwrap())).collect();
    let bits: Vec<_> = thresholds.into_iter().flat_map(|t| values.iter()
        .map(move |&x| i64::from(i128::from(x) < i128::from(t)))).collect();
    let mask: Vec<_> = values.iter().map(|&x| i64::from(
        i128::from(thresholds[0]) <= i128::from(x) && i128::from(x) < i128::from(thresholds[2]))).collect();
    let safe = values.iter().zip(&mask).map(|(&x, &p)| x * p).collect();
    [offsets, bits.clone(), bits, mask, safe]
}

fn current_clear(parties: &[Executor; 3], value: usize) -> Vec<i64> {
    let slot = parties[0].program.slots[value] as usize;
    match parties[0].registers[slot].as_ref().unwrap() {
        Register::Arithmetic(_) => clear_refs(parties.each_ref().map(|p|
            match p.registers[slot].as_ref().unwrap() { Register::Arithmetic(s) => s, _ => panic!() })),
        Register::Bit(_) => {
            let shares = parties.each_ref().map(|p| match p.registers[slot].as_ref().unwrap() {
                Register::Bit(s) => s, _ => panic!(),
            });
            for j in 0..3 {
                assert_eq!(shares[j].bits(), 1);
                assert_eq!(shares[j].second(), shares[(j + 1) % 3].first());
            }
            (0..shares[0].first().len()).map(|j| shares.iter()
                .fold(0u64, |x, s| x ^ s.first()[j]) as i64 & 1).collect()
        }
    }
}

fn execute_and_check(p: &Program, values: &[i64], thresholds: [i64; 3]) {
    let expected = oracle(values, thresholds);
    let mut parties = owners(p, values);
    let usages = parties.each_ref().map(Executor::payload_usage);
    for (index, expected) in expected.iter().enumerate() {
        step(&mut parties); // also checks every actual ordinal/PRSS delta
        assert_eq!(current_clear(&parties, index + 1), *expected);
        let node = p.nodes[index + 1].unwrap();
        for &value in expected {
            assert!(node.bounds.lower() <= value && value <= node.bounds.upper());
        }
    }
    for p in &parties {
        assert_eq!(p.runtime.as_ref().unwrap().source.runtime.next_sequence(), 5);
        assert_eq!(p.cursors().pair_blocks, p.program.work_budget().cursors().pair_blocks);
    }
    let results = parties.map(|p| p.finish().unwrap());
    assert_eq!(clear_refs(results.each_ref().map(|r| &r.1)), expected[4]);
    drop(results);
    assert!(usages.iter().all(|u| u.live_bytes() == 0 && u.peak_bytes() <= u.limit_bytes()
        && u.peak_bytes() + FIXED_RESERVE_BYTES <= p.admitted_bytes()));
}

#[test]
fn indexed_prefix_n1_n128_native_metadata_and_three_party_integer_oracle() {
    let boundary = [-2_097_153, -2_097_152, -2_097_151, -1_048_577, -1_048_576,
        -1_048_575, 1_048_575, 1_048_576, 1_048_577, 1_048_578];
    for n in [1, 128] {
        let bounds = Interval::new(-3_000_000, 3_000_000).unwrap();
        let (offsets, mask) = descriptors(n, PIVOTS);
        let p = resident(n, bounds, PIVOTS).unwrap();
        assert_eq!(p.lanes(), n);
        assert_eq!(p.len, 6);
        assert_eq!(p.instructions(), 5);
        assert_eq!(p.nodes[1].unwrap().lanes, 3 * n);
        assert_eq!(p.nodes[2].unwrap().lanes, 3 * n);
        assert_eq!(p.nodes[3].unwrap().lanes, 3 * n);
        assert_eq!(p.nodes[4].unwrap().bounds, Interval::new(0, 1).unwrap());
        assert_eq!(p.output_bounds(), Interval::new(PIVOTS[0], PIVOTS[2] - 1).unwrap());
        assert_eq!(p.nodes[5].unwrap().instruction, Instruction::Product {
            left: Value(0), right: Value(4), rescale: false, certified: None });
        let expected_work = WorkBudget::local().checked_mul(2).unwrap()
            .checked_add(WorkBudget::conversion(3 * n, true).unwrap()).unwrap()
            .checked_add(WorkBudget::from_boolean(3 * n, 1).unwrap()).unwrap()
            .checked_add(WorkBudget::product(n, false).unwrap()).unwrap();
        assert_eq!(p.work_budget(), expected_work);
        assert_eq!(expected_work.operations(), 5);
        let factories = (work_preflight::row_entries(), attention_work::factory_attempts());
        let mut metadata = IndexedWorkWorkspace::new(n, bounds).unwrap();
        let output = emit(&mut metadata, &offsets, &mask).unwrap();
        assert_eq!(metadata.finish(output).unwrap(), expected_work);
        metadata.assert_native_metadata(&p);
        assert_eq!((work_preflight::row_entries(), attention_work::factory_attempts()), factories);
        assert_eq!(metadata.finish(output), Err(Error::Phase));
        assert!(metadata.checking_visits() > 0);
        if n == 1 {
            for x in boundary { execute_and_check(&p, &[x], PIVOTS); }
        } else {
            let values: Vec<_> = (0..n).map(|j| boundary[j % boundary.len()]).collect();
            execute_and_check(&p, &values, PIVOTS);
        }
    }
}

#[test]
fn indexed_prefix_support_empty_equal_threshold_and_signed_offset_endpoints() {
    for bounds in [Interval::new(-3_000_000, -2_500_000).unwrap(),
        Interval::new(-17, 29).unwrap(), Interval::new(2_000_000, 3_000_000).unwrap()] {
        let p = resident(1, bounds, PIVOTS).unwrap();
        let lo = bounds.lower().max(PIVOTS[0]);
        let hi = bounds.upper().min(PIVOTS[2] - 1);
        assert_eq!(p.output_bounds(), if lo <= hi {
            Interval::new(lo.min(0), hi.max(0)).unwrap()
        } else { Interval::new(0, 0).unwrap() });
        execute_and_check(&p, &[bounds.lower()], PIVOTS);
        execute_and_check(&p, &[bounds.upper()], PIVOTS);
    }
    let equal = [7, 7, 7];
    let p = resident(1, Interval::new(-9, 9).unwrap(), equal).unwrap();
    assert_eq!(p.output_bounds(), Interval::new(0, 0).unwrap());
    for x in [-9, 6, 7, 8, 9] { execute_and_check(&p, &[x], equal); }
    let lower = i64::MIN + PIVOTS[2];
    let upper = i64::MAX + PIVOTS[0];
    let p = resident(1, Interval::new(lower, upper).unwrap(), PIVOTS).unwrap();
    for x in [lower, upper] { execute_and_check(&p, &[x], PIVOTS); }
    for bounds in [Interval::new(lower - 1, upper).unwrap(), Interval::new(lower, upper + 1).unwrap()] {
        let (offsets, _) = descriptors(1, PIVOTS);
        let mut b = ResidentBuilder::from_builder(Builder::indexed(1, bounds).unwrap());
        assert_eq!(b.indexed_affine(Value(0), &offsets), Err(Error::Bounds));
        assert_eq!(b.state, ResidentState::Failed);
        assert_eq!(b.builder.as_ref().unwrap().0.len, 1);
        assert!(b.pending_indexed.is_empty());
        let mut metadata = IndexedWorkWorkspace::new(1, bounds).unwrap();
        assert_eq!(metadata.indexed_affine(Value(0), &offsets), Err(Error::Bounds));
        assert_eq!(metadata.indexed_affine(Value(0), &offsets), Err(Error::Phase));
        assert_eq!(metadata.finish(Value(0)), Err(Error::Phase));
    }
}

#[test]
fn indexed_affine_shared_descriptor_ordered_bounds_and_native1024_materialization() {
    let domain = Interval::new(-11, 17).unwrap();
    let rows = vec![row(7, &[(1, -3), (0, 2)]), row(-5, &[]), row(0, &[(0, 1)])];
    assert_eq!(IndexedAffine(&rows).output_bounds(2, domain).unwrap(), Interval::new(-66, 74).unwrap());
    // Final cancellation cannot excuse an overflowing coefficient or partial.
    for (bounds, bad) in [
        (Interval::new(i64::MAX, i64::MAX).unwrap(), row(1, &[(0, 1), (0, -1)])),
        (Interval::new(i64::MIN, i64::MIN).unwrap(), row(0, &[(0, -1)])),
        (Interval::new(i64::MAX, i64::MAX).unwrap(), row(0, &[(0, 2), (0, -2)])),
    ] { assert_eq!(IndexedAffine(&[bad]).output_bounds(1, bounds), Err(Error::Bounds)); }
    let mut bad = row(0, &[(0, 1)]);
    bad.terms[1] = IndexedTerm { lane: 0, coefficient: 1 };
    assert!(IndexedAffine(&[bad]).output_bounds(1, domain).is_err());
    bad = row(0, &[(1, 1)]);
    assert_eq!(IndexedAffine(&[bad]).output_bounds(1, domain), Err(Error::Shape));
    bad.len = 3;
    assert!(IndexedAffine(&[bad]).output_bounds(2, domain).is_err());
    assert!(IndexedAffine(&[]).output_bounds(1, domain).is_err());
    let wide = vec![row(3, &[(0, -1), (0, 1)]); 1024];
    assert!(IndexedAffine(&wide).output_bounds(1, domain).is_ok());
    assert!(IndexedAffine(&vec![row(0, &[]); 1025]).output_bounds(1, domain).is_err());
    let mut runtimes = parties(WorkBudget::local().limits().unwrap());
    let mut shares = inputs(&runtimes, &[9]);
    let outputs = [0, 1, 2].map(|j| runtimes[j]
        .indexed_affine(&mut shares[j], IndexedAffine(&wide), domain).unwrap());
    assert_eq!(clear(&outputs), vec![3; 1024]);
    for r in &runtimes {
        assert_eq!(r.source.runtime.next_sequence(), 1);
        assert_eq!(r.cursors().pair_blocks, 0);
    }
    // The public constructor is NOT widened; private input width is still128,
    // private temporary width384. Descriptor/native1024 is a distinct bound.
    assert!(Builder::new(129, domain).is_err());
    assert!(Builder::indexed(129, domain).is_err());
    let mut b = Builder::indexed(128, domain).unwrap();
    assert!(b.indexed_affine(Value(0), &vec![row(0, &[(0, 1)]); CAPACITY + 1]).is_err());
    let mut scalar = SoftmaxWorkWorkspace::empty();
    scalar.reset(1, domain);
    assert!(scalar.push(Instruction::IndexedAffine { input: Value(0), table: 1, output_lanes: 1 }, domain, false).is_err());
}

#[test]
fn indexed_local_binding_and_terminal_refusal_precede_ordinals_and_randomness() {
    let bounds = Interval::new(-9, 9).unwrap();
    let descriptor = [row(7, &[(0, 1)])];
    for failure in 0..4 {
        let work = if failure == 2 {
            WorkBudget::conversion(1, false).unwrap().checked_add(WorkBudget::local()).unwrap()
        } else {
            WorkBudget::local()
        };
        let mut runtime = parties(work.limits().unwrap());
        let mut shares = inputs(&runtime, &[3]);
        match failure {
            0 => shares.swap(0, 1),
            1 => shares[0].operation.context.phase += 1,
            2 => {
                let before = runtime[0].cursors();
                let sequence = runtime[0].source.runtime.next_sequence();
                let plan = runtime[0].plan_conversion(1, Kind::Bits).unwrap();
                assert_eq!(plan.kind(), Kind::Bits);
                assert_eq!(runtime[0].status(), Status::Planned);
                assert_eq!(runtime[0].cursors(), before);
                assert_eq!(runtime[0].source.runtime.next_sequence(), sequence);
            },
            _ => (),
        }
        let bad_descriptor = [row(0, &[(1, 1)])];
        let actual = if failure == 3 { &bad_descriptor } else { &descriptor };
        let before = runtime[0].cursors();
        let sequence = runtime[0].source.runtime.next_sequence();
        let result = runtime[0].indexed_affine(&mut shares[0], IndexedAffine(actual), bounds);
        assert!(result.is_err());
        if failure == 2 { assert_eq!(result.err(), Some(Error::Phase)); }
        assert_eq!(runtime[0].status(), Status::Retired);
        assert_eq!(runtime[0].cursors(), before);
        assert_eq!(runtime[0].source.runtime.next_sequence(), sequence);
        assert!(runtime[0].indexed_affine(&mut shares[0], IndexedAffine(&descriptor), bounds).is_err());
    }
    let consumer = |rows: &[IndexedAffineRow], domain: Interval| {
        let mut runtime = parties(WorkBudget::local().limits().unwrap());
        let mut shares = inputs(&runtime, &[3]);
        let output = runtime[0].indexed_affine(&mut shares[0], IndexedAffine(rows), domain).unwrap();
        output.operation.consumer.unwrap()
    };
    assert_ne!(consumer(&descriptor, bounds), consumer(&[row(8, &[(0, 1)])], bounds));
    assert_ne!(consumer(&descriptor, bounds), consumer(&descriptor, Interval::new(-10, 9).unwrap()));
    assert_ne!(consumer(&descriptor, bounds), consumer(&[descriptor[0], descriptor[0]], bounds));
}

#[test]
fn indexed_predicate_checker_rejects_wrong_coeff_lane_threshold_lineage_and_fake_bits() {
    let (_, good) = descriptors(2, PIVOTS);
    for variant in 0..11 {
        let mut b = through_b2a(2);
        let mut mask = good.clone();
        match variant {
            0 => mask[0].terms[1].coefficient = 1,
            1 => mask[0].constant = 1,
            2 => mask[0].terms[1].lane = 1,
            3 => mask[0].terms.swap(0, 1),
            4 => b.0.nodes[3].as_mut().unwrap().instruction = Instruction::FromBit(Value(1)),
            5 => b.0.nodes[2].as_mut().unwrap().instruction = Instruction::CompactSign(Value(1), 64),
            6 => b.0.nodes[3].as_mut().unwrap().instruction = Instruction::Affine {
                left: Value(0), right: None, coefficients: Affine { left: 0, right: 0, constant: 0 } },
            7 => b.0.bit_widths[2] = 2,
            8 => b.0.extension.owners_mut().unwrap().table_mut(1)[0].terms[0].coefficient = 2,
            9 => b.0.nodes[2].as_mut().unwrap().instruction = Instruction::Sign(Value(3)),
            _ => b.0.extension.owners_mut().unwrap().table_mut(1)[0].constant = -PIVOTS[2] - 1,
        }
        let mut resident = ResidentBuilder::from_builder(b);
        assert!(resident.nested_difference(Value(3), &mask).is_err(), "variant={variant}");
        assert_eq!(resident.state, ResidentState::Failed);
        assert_eq!(resident.builder.as_ref().unwrap().0.len, 4);
        assert!(resident.builder.as_ref().unwrap().0.extension.owners().unwrap().rules[4].is_none());
        assert!(resident.nested_difference(Value(3), &good).is_err());
        let mut out = None;
        assert!(resident.transfer_complete(&mut out).is_err());
        assert!(out.is_none());
    }
    // A genuine mask on another (even equal-valued) arithmetic root cannot
    // refine x. Raw node indices/digests do not convey transferable facts.
    let mut b = Builder::indexed(2, Interval::new(-3_000_000, 3_000_000).unwrap()).unwrap();
    let other = b.affine(Value(0), None, Affine { left: 1, right: 0, constant: 0 }).unwrap();
    let (offsets, mask) = descriptors(2, PIVOTS);
    let offset = b.indexed_affine(other, &offsets).unwrap();
    let sign = b.sign(offset).unwrap();
    let packed = b.from_bit(sign).unwrap();
    let inside = b.nested_difference(packed, &mask).unwrap();
    assert!(b.masked_operand(Value(0), inside).is_err());
    assert!(b.masked_operand(other, inside).is_ok());
    // Failed plain Builder has no attempt guard; it is not evidence of sticky
    // invalidation. The enclosing ResidentBuilder is what enforces it.
    let mut b = through_b2a(2);
    assert!(b.nested_difference(Value(3), &[row(0, &[])]).is_err());
    assert!(b.nested_difference(Value(3), &mask).is_ok());
    // -i64::MIN is not an admitted signed threshold, even where the actual
    // offset happens to fit. Recover in i128, then reject, never wrap it.
    let mut b = Builder::indexed(1, Interval::new(i64::MAX, i64::MAX).unwrap()).unwrap();
    let offset = b.indexed_affine(Value(0), &[row(i64::MIN, &[(0, 1)]),
        row(0, &[(0, 1)]), row(0, &[(0, 1)])]).unwrap();
    let sign = b.sign(offset).unwrap();
    let packed = b.from_bit(sign).unwrap();
    assert!(b.nested_difference(packed, &[row(0, &[(2, 1), (0, -1)])]).is_err());
}

#[test]
fn indexed_finish_revalidates_exact_proofs_causality_and_cached_storage() {
    let p = resident(2, Interval::new(-3_000_000, 3_000_000).unwrap(), PIVOTS).unwrap();
    for variant in 0..8 {
        let mut b = Builder(p.clone());
        match variant {
            0 => b.0.extension.owners_mut().unwrap().proofs_mut(4).swap(0, 1),
            1 => b.0.extension.owners_mut().unwrap().rules[5] = Some(Rule::MaskedOperand { original: Value(1), inside: Value(4) }),
            2 => b.0.nodes[2].as_mut().unwrap().instruction = Instruction::Sign(Value(3)),
            3 => b.0.extension.owners_mut().unwrap().table_mut(4)[0].terms[0].lane += 1,
            4 => b.0.nodes[5].as_mut().unwrap().instruction = Instruction::Product {
                left: Value(0), right: Value(4), rescale: false, certified: Some(p.output_bounds()) },
            5 => b.0.nodes[5].as_mut().unwrap().instruction = Instruction::Product {
                left: Value(0), right: Value(4), rescale: true, certified: None },
            6 => b.0.nodes[5].as_mut().unwrap().bounds = Interval::new(0, 0).unwrap(),
            _ => b.0.extension.owners_mut().unwrap().backing_bytes -= 1,
        }
        assert!(b.finish_in_place(Value(5)).is_err(), "variant={variant}");
    }
    let mut b = Builder::indexed(1, Interval::new(-9, 9).unwrap()).unwrap();
    assert!(b.push(Instruction::Affine { left: Value(1), right: None,
        coefficients: Affine { left: 1, right: 0, constant: 0 } }, Interval::new(-9, 9).unwrap(), false).is_err());
    assert_eq!(b.0.len, 1);
}

#[test]
fn indexed_instruction_without_owners_refuses_hash_resources_and_execution_not_fallback() {
    let mut program = resident(1, Interval::new(-3_000_000, 3_000_000).unwrap(), PIVOTS).unwrap();
    let mut execution = owners(&program, &[0]);
    program.extension = predicate_provenance::Extension::Absent;
    assert_eq!(predicate_provenance::Dag::table(&program, Value(1)), Err(Error::Phase));
    assert_eq!(program.resources(), Err(Error::Phase));
    let mut builder = Builder(program);
    assert_eq!(builder.finish_in_place(Value(5)), Err(Error::Phase));
    for owner in &mut execution {
        // Same retained executor, test-only corruption of its original owned
        // Program; no replacement resource-view endpoint or descriptor fallback.
        let ProgramOwner::Owned(program) = &mut owner.program else { panic!("expected owned test executor"); };
        program.extension = predicate_provenance::Extension::Absent;
        assert!(owner.start_step().is_err());
    }
}

#[test]
fn indexed_digest_binds_capacity_descriptor_geometry_and_proof_antecedents() {
    let bounds = Interval::new(-3_000_000, 3_000_000).unwrap();
    let p = resident(2, bounds, PIVOTS).unwrap();
    assert_ne!(p.digest(), resident(2, bounds, [PIVOTS[0] + 1, PIVOTS[1], PIVOTS[2]]).unwrap().digest());
    assert_ne!(p.digest(), resident(2, Interval::new(-3_000_001, 3_000_000).unwrap(), PIVOTS).unwrap().digest());
    assert_ne!(p.digest(), resident(1, bounds, PIVOTS).unwrap().digest());
    let proof_hash = |p: &Program, index| {
        let mut h = Sha256::new();
        let cache = predicate_provenance::RecordIndex::new(p, &mut Audit::default()).unwrap();
        predicate_provenance::hash_rule(p, Some(&cache), index, &mut h);
        <[u8; 32]>::from(h.finalize())
    };
    let mut changed = p.clone();
    changed.extension.owners_mut().unwrap().proofs_mut(4).swap(0, 1);
    assert_ne!(proof_hash(&p, 4), proof_hash(&changed, 4));
    changed.extension.owners_mut().unwrap().rules[5] = Some(Rule::MaskedOperand { original: Value(1), inside: Value(4) });
    assert_ne!(proof_hash(&p, 5), proof_hash(&changed, 5));
    let mut a = Builder::new(1, Interval::new(-9, 9).unwrap()).unwrap();
    let x = a.affine(Value(0), None, Affine { left: 1, right: 0, constant: 0 }).unwrap();
    let ordinary = a.finish(x).unwrap();
    // Independently specified pre-extension byte stream. No extension tag,
    // capacity, descriptor count or absent proof marker may leak into it.
    let mut legacy = Sha256::new();
    legacy.update(b"pllm.puma.public-tensor-program.q18.v1");
    legacy.update(crate::puma::multiply::NUMERIC_PROFILE.as_bytes());
    legacy.update(1u64.to_le_bytes());
    legacy.update(2u64.to_le_bytes());
    legacy.update([0]);
    legacy.update((-9i64).to_le_bytes());
    legacy.update(9i64.to_le_bytes());
    legacy.update([1, 0, u8::MAX]);
    for c in [1i64, 0, 0] { legacy.update(c.to_le_bytes()); }
    legacy.update((-9i64).to_le_bytes());
    legacy.update(9i64.to_le_bytes());
    assert_eq!(ordinary.digest(), <[u8; 32]>::from(legacy.finalize()));
    assert_eq!(ordinary.work_budget(), WorkBudget::local());
    let mut b = Builder::indexed(1, Interval::new(-9, 9).unwrap()).unwrap();
    let x = b.affine(Value(0), None, Affine { left: 1, right: 0, constant: 0 }).unwrap();
    let extended = b.finish(x).unwrap();
    assert_ne!(ordinary.digest(), extended.digest());
    assert_eq!(ordinary.nodes, extended.nodes);
}

#[test]
fn indexed_attempt_errors_keep_exact_pending_owners_and_never_publish_partial_metadata() {
    for stage in [Fault::AcquireRows, Fault::Row, Fault::Publish] {
        let (offsets, _) = descriptors(1, PIVOTS);
        let mut b = ResidentBuilder::from_builder(Builder::indexed(1, Interval::new(-9, 9).unwrap()).unwrap());
        predicate_provenance::arm_fault(stage);
        assert_eq!(b.indexed_affine(Value(0), &offsets), Err(Error::Allocation));
        assert_eq!(b.state, ResidentState::Failed);
        let p = &b.builder.as_ref().unwrap().0;
        assert_eq!(p.len, 1);
        assert!(p.extension.owners().unwrap().table(1).is_empty());
        assert!(p.extension.owners().unwrap().proofs(1).is_empty());
        assert!(p.extension.owners().unwrap().rules[1].is_none());
        assert_eq!(p.extension.backing_bytes(), 0);
        if stage == Fault::Row { assert_eq!(b.pending_indexed.rows.len(), 1); }
        if stage == Fault::Publish { assert_eq!(b.pending_indexed.rows.len(), 3); }
        assert_eq!(b.indexed_affine(Value(0), &offsets), Err(Error::Phase));
        assert_eq!(b.finish(Value(0)), Err(Error::Phase));
    }
    for stage in [Fault::AcquireProofs, Fault::Proof, Fault::Publish] {
        let (_, mask) = descriptors(2, PIVOTS);
        let mut b = ResidentBuilder::from_builder(through_b2a(2));
        predicate_provenance::arm_fault(stage);
        assert_eq!(b.nested_difference(Value(3), &mask), Err(Error::Allocation));
        assert_eq!(b.state, ResidentState::Failed);
        assert_eq!(b.builder.as_ref().unwrap().0.len, 4);
        assert!(b.builder.as_ref().unwrap().0.extension.owners().unwrap().table(4).is_empty());
        assert!(b.builder.as_ref().unwrap().0.extension.owners().unwrap().proofs(4).is_empty());
        assert!(b.builder.as_ref().unwrap().0.extension.owners().unwrap().rules[4].is_none());
        if stage == Fault::AcquireProofs {
            assert_eq!(b.pending_indexed.rows.len(), 0);
            assert_eq!(b.pending_indexed.rows.extent(), 2);
            assert!(matches!(b.pending_indexed.proofs, ExactStorage::Empty));
        }
        if stage == Fault::Proof {
            assert_eq!(b.pending_indexed.rows.len(), 1);
            assert_eq!(b.pending_indexed.proofs.len(), 1);
        }
        assert!(b.masked_operand(Value(0), Value(4)).is_err());
        assert!(b.finish(Value(3)).is_err());
    }
    // New metadata failure is terminal as well; there is no reset escape.
    let (offsets, mask) = descriptors(1, PIVOTS);
    for stage in [Fault::AcquireRows, Fault::Row, Fault::Publish] {
        let mut w = IndexedWorkWorkspace::new(1, Interval::new(-9, 9).unwrap()).unwrap();
        predicate_provenance::arm_fault(stage);
        assert_eq!(w.indexed_affine(Value(0), &offsets), Err(Error::Allocation));
        assert_eq!(emit(&mut w, &offsets, &mask), Err(Error::Phase));
        assert_eq!(w.finish(Value(5)), Err(Error::Phase));
    }
    for stage in [Fault::AcquireProofs, Fault::Proof, Fault::Publish] {
        let mut w = IndexedWorkWorkspace::new(1, Interval::new(-9, 9).unwrap()).unwrap();
        let offset = w.indexed_affine(Value(0), &offsets).unwrap();
        let sign = w.sign(offset).unwrap();
        let packed = w.from_bit(sign).unwrap();
        let backing = w.backing_bytes();
        predicate_provenance::arm_fault(stage);
        assert_eq!(w.nested_difference(packed, &mask), Err(Error::Allocation));
        let snapshot = w.attempt_snapshot();
        assert_eq!(snapshot.0, ResidentState::Failed);
        assert_eq!(snapshot.1, 4);
        assert_eq!(snapshot.4, backing);
        assert_eq!(w.finish(packed), Err(Error::Phase));
    }
}

#[test]
fn indexed_unwind_keeps_pending_backing_in_original_resident_and_metadata_owners() {
    let (offsets, mask) = descriptors(2, PIVOTS);
    for stage in [Fault::PanicAfterProof, Fault::PanicBeforePublish] {
        let mut b = ResidentBuilder::from_builder(through_b2a(2));
        predicate_provenance::arm_fault(stage);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = b.nested_difference(Value(3), &mask);
        })).is_err());
        assert_eq!(b.state, ResidentState::Failed);
        assert_eq!(b.builder.as_ref().unwrap().0.len, 4);
        assert!(b.builder.as_ref().unwrap().0.extension.owners().unwrap().table(4).is_empty());
        assert!(b.builder.as_ref().unwrap().0.extension.owners().unwrap().proofs(4).is_empty());
        assert!(b.pending_indexed.rows.len() > 0);
        assert!(b.pending_indexed.proofs.len() > 0);
        assert!(b.finish(Value(3)).is_err());
        let mut out = None;
        assert!(b.transfer_complete(&mut out).is_err());
        assert!(out.is_none());
    }
    for stage in [Fault::PanicAfterRow, Fault::PanicBeforePublish] {
        let mut w = IndexedWorkWorkspace::new(2, Interval::new(-9, 9).unwrap()).unwrap();
        predicate_provenance::arm_fault(stage);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = w.indexed_affine(Value(0), &offsets);
        })).is_err());
        let snapshot = w.attempt_snapshot();
        assert_eq!(snapshot.0, ResidentState::Failed);
        assert_eq!(snapshot.1, 1);
        assert!(snapshot.2 > 0);
        assert_eq!(snapshot.4, 0);
        assert_eq!(w.finish(Value(5)), Err(Error::Phase));
    }
    for stage in [Fault::PanicAfterProof, Fault::PanicBeforePublish] {
        let mut w = IndexedWorkWorkspace::new(2, Interval::new(-9, 9).unwrap()).unwrap();
        let offset = w.indexed_affine(Value(0), &offsets).unwrap();
        let sign = w.sign(offset).unwrap();
        let packed = w.from_bit(sign).unwrap();
        let backing = w.backing_bytes();
        predicate_provenance::arm_fault(stage);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = w.nested_difference(packed, &mask);
        })).is_err());
        let snapshot = w.attempt_snapshot();
        assert_eq!(snapshot.0, ResidentState::Failed);
        assert_eq!(snapshot.1, 4);
        assert!(snapshot.2 > 0 && snapshot.3 > 0);
        assert_eq!(snapshot.4, backing);
        assert_eq!(w.finish(packed), Err(Error::Phase));
    }
}

#[test]
fn indexed_workspace_indices_carry_no_cross_graph_or_reset_authority() {
    let (offsets, mask) = descriptors(1, PIVOTS);
    let bounds = Interval::new(-3_000_000, 3_000_000).unwrap();
    let mut old = IndexedWorkWorkspace::new(1, bounds).unwrap();
    let output = emit(&mut old, &offsets, &mask).unwrap();
    old.finish(output).unwrap();
    let mut fresh = IndexedWorkWorkspace::new(1, bounds).unwrap();
    assert!(fresh.masked_operand(Value(0), Value(4)).is_err());
    assert_eq!(fresh.indexed_affine(Value(0), &offsets), Err(Error::Phase));
    let mut scalar = SoftmaxWorkWorkspace::empty();
    scalar.reset(1, bounds);
    scalar.reset(1, bounds); // reset of legacy scalar metadata issues no new fact
    assert!(scalar.node(Value(4), false).is_err());
    let mut other = IndexedWorkWorkspace::new(1, bounds).unwrap();
    let arbitrary = other.affine(Value(0), None, Affine { left: 0, right: 0, constant: 0 }).unwrap();
    assert!(other.nested_difference(arbitrary, &mask).is_err());
    assert_eq!(other.finish(arbitrary), Err(Error::Phase));
    // Coincident valid current nodes are interpreted locally, not rejected by
    // foreign-index branding that this private Value type does not possess.
    let mut local = IndexedWorkWorkspace::new(1, bounds).unwrap();
    let local_output = emit(&mut local, &offsets, &mask).unwrap();
    assert_eq!(output, local_output);
    assert!(local.finish(local_output).is_ok());
    for operation in 0..3 {
        let mut w = IndexedWorkWorkspace::new(1, bounds).unwrap();
        let result = match operation {
            0 => w.affine(Value(99), None, Affine { left: 1, right: 0, constant: 0 }),
            1 => w.product(Value(0), Value(99), false),
            _ => w.from_bit(Value(0)),
        };
        assert!(result.is_err());
        assert_eq!(w.attempt_snapshot().0, ResidentState::Failed);
        assert_eq!(w.indexed_affine(Value(0), &offsets), Err(Error::Phase));
    }
    for from_bit in [false, true] {
        let mut b = ResidentBuilder::from_builder(Builder::indexed(1, bounds).unwrap());
        let result = if from_bit { b.predicate_from_bit(Value(0)) } else { b.predicate_sign(Value(99)) };
        assert!(result.is_err());
        assert_eq!(b.state, ResidentState::Failed);
        assert_eq!(b.indexed_affine(Value(0), &offsets), Err(Error::Phase));
    }
}

#[test]
fn indexed_liveness_storage_inventory_and_actual_checker_walks_are_covered() {
    fn phases<R>() -> [usize; 4] {
        use crate::exact_buffer::{ExactBuffer, Error as ExactError};
        use crate::exact_buffer::linked_rows::Link;
        use predicate_provenance::LaneProof;
        use std::mem::{size_of, ManuallyDrop, MaybeUninit};
        use std::ptr::NonNull;
        let common = 3 * size_of::<IndexedAffineRow>() + 2 * size_of::<Value>()
            + 3 * size_of::<usize>() + size_of::<ExactBuffer<R>>()
            + 2 * size_of::<Result<Box<dyn Link<IndexedAffineRow, Value, LaneProof>>, ExactError>>();
        // Independent literal type inventory, no production layout/phase helper.
        // new: count/layout/allocator argument, pointer/raw/checked pointer,
        // ownership construction, returned buffer and exclusive failure control.
        let acquisition = size_of::<usize>() + 2 * size_of::<std::alloc::Layout>()
            + size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>()
            + size_of::<Result<std::alloc::Layout, ExactError>>() + size_of::<std::alloc::LayoutError>()
            + size_of::<NonNull<MaybeUninit<R>>>() + size_of::<*mut MaybeUninit<R>>()
            + size_of::<*mut u8>() + size_of::<Option<NonNull<MaybeUninit<R>>>>()
            + size_of::<Result<NonNull<MaybeUninit<R>>, ExactError>>()
            + size_of::<*mut [MaybeUninit<R>]>() + size_of::<Box<[MaybeUninit<R>]>>()
            + size_of::<ExactBuffer<R>>() + size_of::<Result<ExactBuffer<R>, ExactError>>()
            + size_of::<bool>() + size_of::<ExactError>();
        let copy = 2 * size_of::<R>() + size_of::<Result<(), R>>()
            + size_of::<&mut ExactBuffer<R>>() + size_of::<Option<&mut MaybeUninit<R>>>()
            + size_of::<&mut MaybeUninit<R>>() + size_of::<&Result<(), R>>() + size_of::<bool>();
        // finish: caller buffer is in common; these are DISTINCT consumed-self,
        // this, slots, borrowed/raw read arguments and owned return endpoints.
        let finish = size_of::<ExactBuffer<R>>() + size_of::<ManuallyDrop<ExactBuffer<R>>>()
            + size_of::<Box<[MaybeUninit<R>]>>() + size_of::<&Box<[MaybeUninit<R>]>>()
            + size_of::<*const Box<[MaybeUninit<R>]>>() + size_of::<Box<[R]>>()
            + size_of::<Result<Box<[R]>, ExactError>>() + 2 * size_of::<usize>()
            + size_of::<bool>() + size_of::<ExactError>();
        let conversion = size_of::<Result<Box<[R]>, ExactError>>() + size_of::<Box<[R]>>()
            + size_of::<Result<Box<[R; 1]>, Box<[R]>>>() + 2 * size_of::<Box<[R; 1]>>()
            + size_of::<Box<dyn Link<IndexedAffineRow, Value, LaneProof>>>();
        [acquisition, copy, finish, conversion].map(|phase| common.checked_add(phase).unwrap())
    }
    fn staging<R>() -> usize {
        phases::<R>().into_iter().max().unwrap()
    }
    // Check EACH phase even when the large seed-copy phase wins the max. Actual
    // sealed record types at both dispatch endpoints, not mirrored ABI structs.
    use crate::exact_buffer::linked_rows::Record;
    use predicate_provenance::LaneProof;
    assert_eq!(Record::<IndexedAffineRow, Value, LaneProof, 1>::STAGING_PHASES.unwrap(),
        phases::<Record<IndexedAffineRow, Value, LaneProof, 1>>());
    assert_eq!(Record::<IndexedAffineRow, Value, LaneProof, 384>::STAGING_PHASES.unwrap(),
        phases::<Record<IndexedAffineRow, Value, LaneProof, 384>>());
    for n in [1, 128] {
        let p = resident(n, Interval::new(-3_000_000, 3_000_000).unwrap(), PIVOTS).unwrap();
        assert_eq!(&p.last_use[..6], &[5, 2, 3, 4, 5, 6]);
        assert_eq!(p.nodes[1].unwrap().instruction.inputs(), [Some(Value(0)), None]);
        assert_eq!(p.nodes[4].unwrap().instruction.inputs(), [Some(Value(3)), None]);
        assert_eq!(p.nodes[5].unwrap().instruction.inputs(), [Some(Value(0)), Some(Value(4))]);
        let quote = IndexedInventory::quote(n, &[(3 * n, false), (n, true)]).unwrap();
        let backing = quote.descriptor_bytes + quote.proof_bytes;
        // Independent concrete physical records, not row bytes or a call back
        // into the quote/layout dispatcher under test. Caller slices stay raw.
        use crate::exact_buffer::linked_rows::Record;
        use predicate_provenance::LaneProof;
        use std::mem::size_of;
        let physical_rows = if n == 1 {
            std::mem::size_of::<Record<IndexedAffineRow, Value, LaneProof, 3>>()
                + std::mem::size_of::<Record<IndexedAffineRow, Value, LaneProof, 1>>()
        } else {
            std::mem::size_of::<Record<IndexedAffineRow, Value, LaneProof, 384>>()
                + std::mem::size_of::<Record<IndexedAffineRow, Value, LaneProof, 128>>()
        };
        assert_eq!(quote.descriptor_bytes, physical_rows);
        assert_eq!(quote.proof_bytes, n * std::mem::size_of::<LaneProof>());
        let allocation = if n == 1 {
            staging::<Record<IndexedAffineRow, Value, LaneProof, 3>>()
                .max(staging::<Record<IndexedAffineRow, Value, LaneProof, 1>>())
        } else {
            staging::<Record<IndexedAffineRow, Value, LaneProof, 384>>()
                .max(staging::<Record<IndexedAffineRow, Value, LaneProof, 128>>())
        } + size_of::<predicate_provenance::CachedDag<'static>>() + size_of::<Audit>();
        let cache_build = size_of::<predicate_provenance::RecordIndex<'static>>()
            + size_of::<Result<predicate_provenance::RecordIndex<'static>, Error>>()
            + size_of::<crate::exact_buffer::linked_rows::Layouts>()
            + size_of::<Result<crate::exact_buffer::linked_rows::Layouts, Error>>()
            + size_of::<Option<&'static predicate_provenance::RecordLink>>()
            + size_of::<&'static predicate_provenance::RecordLink>()
            + size_of::<Node>() + size_of::<Value>() + 6 * size_of::<usize>()
            + size_of::<&'static predicate_provenance::IndexedOwners>()
            + size_of::<&'static dyn predicate_provenance::Dag>() + size_of::<&'static mut Audit>();
        let checking = 12 * size_of::<Node>() + 8 * size_of::<predicate_provenance::Predicate>()
            + 6 * size_of::<LaneProof>() + 12 * size_of::<Interval>() + 8 * size_of::<[i128; 2]>()
            + size_of::<Sha256>() + size_of::<Audit>() + cache_build
            + size_of::<predicate_provenance::CachedDag<'static>>()
            + size_of::<Result<predicate_provenance::CachedDag<'static>, Error>>()
            + 3 * size_of::<Option<&'static predicate_provenance::RecordLink>>()
            + 2 * size_of::<&'static predicate_provenance::RecordLink>()
            + 2 * size_of::<&'static crate::exact_buffer::linked_rows::Chain<IndexedAffineRow, Value, LaneProof>>()
            + 2 * size_of::<&'static predicate_provenance::IndexedOwners>()
            + size_of::<&'static predicate_provenance::Extension>()
            + 2 * size_of::<usize>() + 2 * size_of::<Value>()
            + size_of::<&'static [IndexedAffineRow]>()
            + size_of::<Result<&'static [IndexedAffineRow], Error>>()
            + size_of::<Result<&'static predicate_provenance::IndexedOwners, Error>>();
        let finish = size_of::<[usize; MAX_NODES]>() + size_of::<Sha256>()
            + 2 * size_of::<predicate_provenance::Predicate>() + size_of::<Rule>()
            + size_of::<Node>() + size_of::<WorkBudget>()
            + size_of::<Option<predicate_provenance::RecordIndex<'static>>>() + size_of::<Audit>()
            + cache_build;
        assert_eq!(quote.allocation_scratch_bytes, allocation);
        assert_eq!(quote.checker_scratch_bytes, checking);
        assert_eq!(quote.finish_scratch_bytes, finish);
        let active = physical_rows + n * size_of::<LaneProof>() + 4 * n * size_of::<IndexedAffineRow>()
            + allocation.max(checking).max(finish) + size_of::<predicate_provenance::PreparedMap>()
            + size_of::<Result<predicate_provenance::PreparedMap, Error>>();
        assert_eq!(quote.native_known_peak_bytes, active + size_of::<ResidentBuilder>()
            + size_of::<Option<Program>>() + size_of::<Result<(), Error>>());
        let view = size_of::<&'static attention_work::SoftmaxWorkWorkspace>()
            + size_of::<&'static predicate_provenance::IndexedOwners>();
        assert_eq!(attention_work::INDEXED_WORK_VIEW_BYTES, view);
        let metadata_scratch = allocation.max(checking).checked_add(view).unwrap().max(finish);
        let metadata = physical_rows + n * size_of::<LaneProof>() + 4 * n * size_of::<IndexedAffineRow>()
            + metadata_scratch + size_of::<predicate_provenance::PreparedMap>()
            + size_of::<Result<predicate_provenance::PreparedMap, Error>>()
            + size_of::<IndexedWorkWorkspace>() + size_of::<Result<WorkBudget, Error>>();
        assert_eq!(quote.metadata_known_peak_bytes, metadata);
        assert_eq!(quote.hash_visits_bound,
            (192 + 4 * n as u64 * 8 + n as u64 * 16) * 64 + (192 * 9 + 8) * 64 + 192 * 64);
        let r = p.resources().unwrap();
        assert_eq!(backing, p.public_table_bytes().unwrap());
        assert_eq!(quote.allocation_attempts, 3);
        assert_eq!(quote.caller_descriptor_bytes, 4 * n * std::mem::size_of::<IndexedAffineRow>());
        assert_eq!(quote.pending_owner_bytes, std::mem::size_of::<predicate_provenance::Pending>());
        assert!(quote.native_known_peak_bytes >= std::mem::size_of::<ResidentBuilder>() + backing);
        assert!(quote.metadata_known_peak_bytes >= IndexedWorkWorkspace::workspace_bytes() + backing);
        assert!(quote.finish_scratch_bytes >= std::mem::size_of::<[usize; MAX_NODES]>());
        assert!(quote.hash_visits_bound > 0);
        assert!(r.working_bytes >= quote.checker_scratch_bytes);
        assert!(r.working_bytes >= p.working_bytes().unwrap());
        let mut audit = Audit::default();
        predicate_provenance::validate(&p, &mut audit).unwrap();
        assert!(audit.visits > 0 && audit.visits <= quote.finish_validation_visits);
        assert!(audit.visits <= predicate_provenance::REPAIRED_VALIDATION_BOUND);
        assert!(predicate_provenance::REPAIRED_VALIDATION_BOUND < 806_701_248);
        assert_eq!(predicate_provenance::REPAIRED_VALIDATION_BOUND,
            192 * (1024 * 3072 + 1024) + (192 * 9 + 8) * 64);
        assert_eq!(predicate_provenance::RUNTIME_LOOKUP_VISITS_BOUND, 192 * 192 * 64);
        assert!(predicate_provenance::REPAIRED_VALIDATION_BOUND
            + predicate_provenance::RUNTIME_LOOKUP_VISITS_BOUND < 806_701_248);
        let max_hash = (192_u64 + 192 * 384 * 24) * 64 + (192 * 9 + 8) * 64 + 192 * 64;
        assert_eq!(predicate_provenance::MAX_HASH_VISITS_BOUND, max_hash);
        assert!(quote.hash_visits_bound <= max_hash);
        assert!(192 * (1024 * 3072 + 1024) + (192 * 9 + 8) * 64
            + max_hash + 192 * 192 * 64 < 806_701_248);
        assert!(Program::full_resource_validation_visits().unwrap() >= audit.visits);
        assert!(p.extension.owners().unwrap().check_visits >= audit.visits);
        let mut owners = owners(&p, &vec![0; n]);
        for index in 1..6 {
            step(&mut owners);
            for owner in &owners {
                for prior in 0..index {
                    if p.last_use[prior] as usize == index && p.slots[prior] != p.slots[index] {
                        assert!(owner.registers[p.slots[prior] as usize].is_none());
                    }
                }
            }
        }
    }
}

#[test]
fn indexed_shared_owner_wire_parity_replay_cancel_and_exact_work_refusal() {
    let p = Arc::new(resident(1, Interval::new(-3_000_000, 3_000_000).unwrap(), PIVOTS).unwrap());
    let mut owned = parity_owners(&p, &[1_048_576], false, 1);
    let mut shared = parity_owners(&p, &[1_048_576], true, 1);
    for o in &shared {
        assert!(std::ptr::eq::<Program>(&*o.program, &*p));
        assert_eq!(o.program.extension.owners().unwrap().table(1).as_ptr(), p.extension.owners().unwrap().table(1).as_ptr());
        assert_eq!(o.program.extension.owners().unwrap().proofs(4).as_ptr(), p.extension.owners().unwrap().proofs(4).as_ptr());
    }
    compare_execution(&mut owned, &mut shared);
    let a = owned.map(|o| o.finish().unwrap());
    let b = shared.map(|o| o.finish().unwrap());
    for (a, b) in a.iter().zip(&b) {
        assert_eq!(a.1.first(), b.1.first());
        assert_eq!(a.1.second(), b.1.second());
    }
    drop((a, b));
    let weak = Arc::downgrade(&p);
    let mut retained = parity_owners(&p, &[7], true, 1);
    drop(p);
    for o in &mut retained { o.cancel(); }
    assert_eq!(weak.strong_count(), 3);
    drop(retained);
    assert_eq!(weak.strong_count(), 0);

    let p = resident(1, Interval::new(-3_000_000, 3_000_000).unwrap(), PIVOTS).unwrap();
    cancel_at_every_event(&p, &[7]);
    let work = p.work_budget();
    for limits in [PrssLimits::program(work.operations() - 1, work.cursors().aes_work).unwrap(),
        PrssLimits::program(work.operations(), work.cursors().aes_work - 1).unwrap()] {
        let runtimes = parties(limits);
        let shares = inputs(&runtimes, &[7]);
        for (r, input) in runtimes.into_iter().zip(shares) {
            let usage = r.payload_usage();
            assert!(Executor::new(r, p.clone(), input).is_err());
            assert_eq!(usage.live_bytes(), 0);
        }
    }
    // Capture actual Sign wire, replay into an already advanced or different
    // immutable graph; existing program/invocation/step binding must retire it.
    let mut first = owners(&p, &[7]);
    step(&mut first);
    let wire = encoded_step(&mut first, &mut Vec::new(), false);
    let (sender, recipient, bytes) = wire[0].clone();
    assert!(first[recipient].receive_encoded(&bytes, sender).is_err());
    assert!(first[recipient].retired());
    let changed = resident(1, Interval::new(-3_000_000, 3_000_000).unwrap(),
        [PIVOTS[0] + 1, PIVOTS[1], PIVOTS[2]]).unwrap();
    let mut other = owners(&changed, &[7]);
    step(&mut other);
    assert!(other[recipient].receive_encoded(&bytes, sender).is_err());
    assert!(other[recipient].retired());
}
