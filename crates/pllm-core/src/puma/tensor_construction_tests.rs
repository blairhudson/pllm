use super::*;
use std::mem::{align_of, size_of};

#[test]
fn checked_rows_quote_geometry_and_private_origin_identity() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    for (rows, width, error) in [(0, usize::MAX, Error::Shape), (usize::MAX, 0, Error::Shape),
        (usize::MAX, 2, Error::Bounds), (9, 1, Error::Bounds), (1, 129, Error::Bounds), (8, 128, Error::Bounds)] {
        assert_eq!(Construction::CheckedRangeRows { rows, width }.quote(), Err(ConstructionQuoteError::Geometry(error)));
    }
    let a = Program::checked_range_rows(2, 64, full, full, 0).unwrap();
    let b = Program::checked_range_rows(4, 32, full, full, 0).unwrap();
    let legacy = Program::checked_range(128, full, full, 0).unwrap();
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.nodes, legacy.nodes);
    assert_ne!(a.digest(), b.digest());
    assert_ne!(a.digest(), legacy.digest());
    assert_eq!(a.work_budget(), legacy.work_budget());
    assert!(!a.extension.absent());
    assert!(!predicate_provenance::Dag::extended(&a));
    assert!(a.extension.owners().is_err());
    assert_eq!(a.extension.backing_bytes(), 0);
    assert_eq!(a.extension.capacity(a.lanes), 128);
    for index in 0..a.len {
        let v = Value(index as u8);
        assert!(predicate_provenance::Dag::table(&a, v).unwrap().is_empty());
        assert!(predicate_provenance::Dag::rule(&a, v).is_none());
        assert!(predicate_provenance::Dag::proofs(&a, v).is_empty());
    }
    assert!(legacy.extension.absent());
    let before = predicate_provenance::access_counts();
    let clone = a.clone();
    let _ = clone.resources().unwrap();
    drop(clone);
    assert_eq!(predicate_provenance::access_counts(), before);
}

// Independent literal encodings, never hash_header/hash_rule/BitMap::hash.
// Track every opaque update occurrence and byte, including legacy header stream.
fn checked_rows_expected_hash(p: &Program, geometry: Option<(u64, u64)>) -> ([u8; 32], usize, usize) {
    let mut h = Sha256::new();
    let mut calls = 0usize;
    let mut bytes = 0usize;
    let mut update = |raw: &[u8]| { h.update(raw); calls += 1; bytes += raw.len(); };
    update(b"pllm.puma.public-tensor-program.q18.v1");
    update(b"puma.checked-range-q18.v1");
    update(&(p.lanes as u64).to_le_bytes());
    update(&(p.len as u64).to_le_bytes());
    update(b"checked-validity-output.v1");
    update(&[p.checked_validity.unwrap().0]);
    if let Some((rows, width)) = geometry {
        update(b"puma.checked-rows.v1");
        update(&rows.to_le_bytes());
        update(&width.to_le_bytes());
    }
    for node in p.nodes[..p.len].iter().flatten() {
        match node.instruction {
            Instruction::Input => update(&[0]),
            Instruction::Bits(v) => update(&[6, v.0]),
            Instruction::BitMap(v, map) => {
                update(&[7, v.0]);
                let (tag, parameter) = match map {
                    BitMap::LowBits(n) => (0, u64::from(n)),
                    BitMap::ShiftRight(n) => (1, u64::from(n)),
                    BitMap::ShiftLeft(n) => (2, u64::from(n)),
                    BitMap::Parity(mask) => (5, mask),
                    _ => panic!("unexpected Rows map"),
                };
                update(&[tag]); update(&parameter.to_le_bytes());
            }
            Instruction::BitConstantXor(v, c) => { update(&[12, v.0]); update(&c.to_le_bytes()); }
            Instruction::BitOr(a, b) => update(&[8, a.0, b.0]),
            Instruction::BitXor(a, b) => update(&[9, a.0, b.0]),
            Instruction::FromBit(v) => update(&[5, v.0]),
            Instruction::Affine { left, right, coefficients } => {
                update(&[1, left.0, right.map_or(255, |v| v.0)]);
                update(&coefficients.left.to_le_bytes());
                update(&coefficients.right.to_le_bytes());
                update(&coefficients.constant.to_le_bytes());
            }
            _ => panic!("unexpected Rows node"),
        }
        update(&node.bounds.lower().to_le_bytes()); update(&node.bounds.upper().to_le_bytes());
    }
    drop(update);
    (h.finalize().into(), calls, bytes)
}

#[test]
fn checked_rows_complete_hash_literal_census_and_unchanged_legacy_stream() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    let range = Interval::new(-7, 11).unwrap();
    let legacy = Program::checked_range(128, full, range, 3).unwrap();
    let (digest, updates, bytes) = checked_rows_expected_hash(&legacy, None);
    assert_eq!(legacy.digest(), digest);
    assert_eq!((updates, bytes), (272, 1682));
    for (rows, width) in [(1, 128), (2, 64), (4, 128), (8, 64)] {
        let p = Program::checked_range_rows(rows, width, full, range, 3).unwrap();
        let (digest, updates, bytes) = checked_rows_expected_hash(&p, Some((rows as u64, width as u64)));
        assert_eq!(p.digest(), digest);
        assert_eq!((updates, bytes), (275, 1718));
        assert_eq!((2 * bytes + 32, 64 * (updates + 2), updates + 2), (3468, 17_728, 277));
    }
}

#[test]
fn checked_rows_real_resources_keep_full_liveness_and_inline_owners() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    for (rows, width) in [(1, 128), (3, 128), (4, 128), (8, 64)] {
        let p = Program::checked_range_rows(rows, width, full, full, 0).unwrap();
        let r = p.resources().unwrap();
        assert_eq!(r.public_table_bytes, 0);
        assert_eq!(r.immutable_template_bytes, size_of::<Program>() + 2 * size_of::<usize>());
        assert_eq!(r.register_container_bytes, p.register_count * size_of::<Option<Register>>());
        assert_eq!(r.invocation_container_bytes, size_of::<Executor>() + r.register_container_bytes);
        assert_eq!(r.frame_bytes, 3 * size_of::<Frame>());
        let mut peak = 0;
        for step in 0..p.len {
            let live: usize = (0..=step).filter(|&i| i == step || usize::from(p.last_use[i]) >= step)
                .map(|i| 2 * p.nodes[i].unwrap().lanes).sum();
            let node = p.nodes[step].unwrap();
            let temporary = match node.instruction {
                Instruction::Input => 0,
                Instruction::Affine { right, .. } => if right.is_some() { 10 } else { 6 },
                Instruction::Bits(_) => 34,
                Instruction::BitMap(..) | Instruction::BitConstantXor(..) => 6,
                Instruction::BitOr(..) => 20,
                Instruction::BitXor(..) => 10,
                Instruction::FromBit(v) => { let bits = usize::from(p.bit_widths[v.0 as usize]);
                    if bits <= 8 { 16 * bits + 10 } else { 42 } }
                _ => panic!("unexpected Rows resource instruction"),
            } * node.lanes;
            peak = peak.max(8 * (live + temporary));
        }
        assert_eq!(r.working_bytes, peak);
        assert_eq!(r.owned_admitted_bytes, FIXED_RESERVE_BYTES + r.invocation_container_bytes + r.frame_bytes + peak);
        assert_eq!(r.shared_admitted_bytes, r.owned_admitted_bytes + r.immutable_template_bytes);
        let q = Construction::CheckedRangeRows { rows, width }.quote().unwrap();
        assert_eq!(q.public_table_bytes, 0);
        assert_eq!(q.immutable_template_bytes, r.immutable_template_bytes);
        assert!(q.return_slot_bytes >= size_of::<Result<Program, Error>>());
        assert!(size_of::<ProgramOwner>() >= size_of::<Program>());
        assert!(size_of::<Executor>() >= size_of::<ProgramOwner>());
    }
}

#[test]
fn checked_rows_identity_validator_refuses_profile_validity_geometry_and_indexed_forgery() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    let p = Program::checked_range_rows(2, 8, full, full, 0).unwrap();
    for fault in 0..5 {
        let mut broken = p.clone();
        match fault {
            0 => broken.numeric_profile = super::super::NUMERIC_PROFILE,
            1 => broken.checked_validity = None,
            2 => broken.lanes = 8,
            3 => broken.extension = predicate_provenance::Extension::CheckedRowsIdentity(CheckedRowsIdentity { rows: 0, width: 8 }),
            _ => broken.extension = predicate_provenance::Extension::CheckedRowsIdentity(CheckedRowsIdentity { rows: u64::MAX, width: 2 }),
        }
        assert!(broken.resources().is_err());
    }
    let mut forged = p.clone();
    forged.nodes[1].as_mut().unwrap().instruction = Instruction::IndexedAffine { input: Value(0), table: 0, output_lanes: 16 };
    assert!(matches!(predicate_provenance::Dag::table(&forged, Value(1)), Err(Error::Phase)));
    assert!(matches!(forged.resources(), Err(Error::Phase)));
}

// Literal pre-extraction composition. Never call equal_zero_recipe or its work
// visitor for the expected instructions/digest/slot plan.
fn original_equal_zero(lanes: usize, input: Interval) -> Result<Program, Error> {
    let mut b = Builder::new(lanes, input)?;
    let mut bits = b.bits(b.input())?;
    for shift in [32, 16, 8, 4, 2, 1] {
        let other = b.bit_map(bits, BitMap::ShiftRight(shift))?;
        bits = b.bit_or(bits, other)?;
    }
    let bit = b.bit_map(bits, BitMap::LowBits(1))?;
    let nonzero = b.from_bit(bit)?;
    let out = b.affine(
        nonzero,
        None,
        Affine {
            left: -1,
            right: 0,
            constant: 1,
        },
    )?;
    b.finish(out)
}

fn equality_loop_work(lanes: usize, input: Interval) -> Result<WorkBudget, Error> {
    if lanes == 0 {
        return Err(Error::Bounds);
    }
    let width = lanes.min(MAX_TILE_LANES);
    let mut work = Program::equal_zero_work(width, input)?
        .checked_mul(u64::try_from(lanes / width).map_err(|_| Error::Bounds)?)?;
    if lanes % width != 0 {
        work = work.checked_add(Program::equal_zero_work(lanes % width, input)?)?;
    }
    Ok(work)
}

// Test consumer of the ordinary native WorkBudget admission contract. This is
// not a new production Budget/constructor API or complete-owner memory claim.
fn admitted_equality_loop(
    lanes: usize,
    input: Interval,
    max_operations: u64,
    max_aes: u64,
) -> Result<(Program, Option<Program>), Error> {
    let work = equality_loop_work(lanes, input)?;
    work.admit(
        0,
        crate::puma::prss::Cursors::default(),
        max_operations,
        max_aes,
    )?;
    let width = lanes.min(MAX_TILE_LANES);
    let full = Program::equal_zero(width, input)?;
    let tail = if lanes % width == 0 {
        None
    } else {
        Some(Program::equal_zero(lanes % width, input)?)
    };
    Ok((full, tail))
}

#[test]
fn equal_zero_work_preflight_matches_literal_factory_every_width_and_domain_endpoints() {
    for lanes in 1..=MAX_TILE_LANES {
        for (lower, upper) in [
            (i64::MIN, i64::MAX),
            (i64::MIN, i64::MIN),
            (i64::MAX, i64::MAX),
            (INPUT_MIN, INPUT_MAX_EXCLUSIVE - 1),
            (-151935, 151935),
            (-1, 1),
            (0, 0),
        ] {
            let input = Interval::new(lower, upper).unwrap();
            let attempts = numeric::equal_zero_factory_attempts();
            let quote = Program::equal_zero_work(lanes, input).unwrap();
            assert_eq!(numeric::equal_zero_factory_attempts(), attempts);
            let actual = Program::equal_zero(lanes, input).unwrap();
            assert_eq!(numeric::equal_zero_factory_attempts(), attempts + 1);
            let original = original_equal_zero(lanes, input).unwrap();
            assert_eq!(quote, actual.work_budget());
            assert_eq!(quote.cursors(), original.work_budget().cursors());
            assert_eq!(quote.operations(), 28);
            assert_eq!(actual.nodes, original.nodes);
            assert_eq!(actual.len, original.len);
            assert_eq!(actual.bit_widths, original.bit_widths);
            assert_eq!(actual.digest(), original.digest());
            assert_eq!(actual.numeric_profile(), original.numeric_profile());
            assert_eq!(actual.last_use, original.last_use);
            assert_eq!(actual.slots, original.slots);
            assert_eq!(actual.register_count, original.register_count);
        }
    }
}

#[test]
fn equal_zero_work_preflight_full_tail_and_call_reset_cursors_match_native_programs() {
    let input = Interval::new(-1024, 1024).unwrap();
    for lanes in [1, 2, 7, 127, 128, 129, 255, 256, 257, 1025] {
        let attempts = numeric::equal_zero_factory_attempts();
        let quote = equality_loop_work(lanes, input).unwrap();
        assert_eq!(numeric::equal_zero_factory_attempts(), attempts);
        let width = lanes.min(MAX_TILE_LANES);
        let full = Program::equal_zero(width, input).unwrap();
        let mut actual = full
            .work_budget()
            .checked_mul((lanes / width) as u64)
            .unwrap();
        if lanes % width != 0 {
            let tail = Program::equal_zero(lanes % width, input).unwrap();
            actual = actual.checked_add(tail.work_budget()).unwrap();
        }
        assert_eq!(quote, actual);
        assert_eq!(quote.cursors(), actual.cursors());
    }
    // Odd invocation tails are priced separately, never one merged PRSS fill.
    let one = Program::equal_zero_work(1, input).unwrap();
    let two_calls = one.checked_mul(2).unwrap();
    assert_ne!(
        two_calls.cursors(),
        Program::equal_zero_work(2, input).unwrap().cursors()
    );
}

#[test]
fn equal_zero_work_preflight_geometry_errors_match_factory_without_entering_it() {
    assert!(matches!(Interval::new(1, 0), Err(Error::Bounds)));
    for lanes in [0, MAX_TILE_LANES + 1, usize::MAX] {
        for input in [
            Interval::new(0, 0).unwrap(),
            Interval::new(i64::MIN, i64::MAX).unwrap(),
        ] {
            let attempts = numeric::equal_zero_factory_attempts();
            let quote = Program::equal_zero_work(lanes, input);
            assert!(matches!(quote, Err(Error::Bounds)));
            assert_eq!(numeric::equal_zero_factory_attempts(), attempts);
            let actual = Program::equal_zero(lanes, input).map(|p| p.work_budget());
            assert_eq!(quote, actual);
            assert_eq!(numeric::equal_zero_factory_attempts(), attempts + 1);
        }
    }
}

#[test]
fn equal_zero_work_preflight_zero_one_short_and_overflow_refuse_before_all_factories() {
    let input = Interval::new(i64::MIN, i64::MAX).unwrap();
    for lanes in [1, 127, 128, 129, 257] {
        let quote = equality_loop_work(lanes, input).unwrap();
        let (ops, aes) = (quote.operations(), quote.cursors().aes_work);
        let attempts = numeric::equal_zero_factory_attempts();
        for (max_ops, max_aes) in [(0, aes), (ops - 1, aes), (ops, 0), (ops, aes - 1)] {
            assert!(matches!(
                admitted_equality_loop(lanes, input, max_ops, max_aes),
                Err(Error::Bounds)
            ));
            assert_eq!(numeric::equal_zero_factory_attempts(), attempts);
        }
        let (full, tail) = admitted_equality_loop(lanes, input, ops, aes).unwrap();
        assert_eq!(
            numeric::equal_zero_factory_attempts(),
            attempts + 1 + usize::from(tail.is_some())
        );
        assert_eq!(
            full.work_budget(),
            Program::equal_zero_work(lanes.min(MAX_TILE_LANES), input).unwrap()
        );
    }
    let attempts = numeric::equal_zero_factory_attempts();
    let quote = Program::equal_zero_work(1, input).unwrap();
    assert!(matches!(quote.checked_mul(u64::MAX), Err(Error::Bounds)));
    assert!(matches!(
        WorkBudget::local()
            .checked_mul(u64::MAX)
            .unwrap()
            .checked_add(quote),
        Err(Error::Bounds)
    ));
    for cursors in [
        crate::puma::prss::Cursors {
            public_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            pair_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            aes_work: u64::MAX,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            quote.admit(0, cursors, u64::MAX, u64::MAX),
            Err(Error::Bounds)
        ));
    }
    // Equality itself draws no private blocks. A following ordinary rescale
    // does; aggregate preflight must retain that independent cursor overflow.
    assert_eq!(quote.cursors().private_blocks, 0);
    let with_rescale = quote.checked_add(WorkBudget::rescale(1).unwrap()).unwrap();
    assert!(matches!(
        with_rescale.admit(
            0,
            crate::puma::prss::Cursors {
                private_blocks: u64::MAX,
                ..Default::default()
            },
            u64::MAX,
            u64::MAX
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(numeric::equal_zero_factory_attempts(), attempts);
}

#[test]
fn equal_zero_work_preflight_keeps_existing_rescale_work_contract_and_bounded_scope_distinct() {
    let input = Interval::new(-37, 37).unwrap();
    for lanes in [1, 3, 127, MAX_TILE_LANES] {
        let mut native = Builder::new(lanes, input).unwrap();
        let output = native.rescale(native.input(), 18).unwrap();
        assert_eq!(
            native.finish(output).unwrap().work_budget(),
            WorkBudget::rescale(lanes).unwrap()
        );
        let legacy = Program::equal_zero_work(lanes, input).unwrap();
        assert_ne!(
            legacy,
            Program::equal_zero_bounded(lanes, input)
                .unwrap()
                .work_budget()
        );
        assert_eq!(
            legacy,
            Program::equal_zero_work(lanes, Interval::new(i64::MIN, i64::MAX).unwrap()).unwrap()
        );
    }
}

fn verify_phases(q: ConstructionQuote) {
    let inline = size_of::<Program>();
    assert_eq!(size_of::<Builder>(), inline);
    assert_eq!(
        q.immutable_template_bytes,
        inline + 2 * size_of::<usize>() + q.public_table_bytes
    );
    assert!(q.return_slot_bytes >= size_of::<Result<Program, Error>>());
    let owners = q.return_slot_bytes + size_of::<Builder>() + q.public_table_bytes
        + size_of::<ExactStorage<Affine>>();
    let scratch = q.constructor_buffer_bytes + q.proof_buffer_bytes;
    assert_eq!(
        q.finish_buffer_bytes,
        size_of::<Sha256>() + MAX_NODES * size_of::<usize>()
            + size_of::<Option<predicate_provenance::RecordIndex<'static>>>()
            + size_of::<predicate_provenance::Audit>()
    );
    assert_eq!(q.new_build_bytes, inline + owners + scratch);
    assert_eq!(q.new_finish_bytes, owners + q.finish_buffer_bytes);
    // RMS additionally covers its sealed resident handoff/take/install peak;
    // that exact owner-local inventory is reconstructed in rms_resident::tests.
    assert!(q.new_publish_bytes >= q.return_slot_bytes + q.immutable_template_bytes);
    assert_eq!(
        q.new_peak_bytes,
        q.new_build_bytes
            .max(q.new_finish_bytes)
            .max(q.new_publish_bytes)
    );
    assert_eq!(
        q.peak_with_borrowed_public_bytes,
        q.new_peak_bytes + q.borrowed_public_bytes
    );
}

#[test]
fn native_construction_phase_inventories_are_explicit_not_execution_allowances() {
    for family in [
        Construction::Builder { lanes: 1 },
        Construction::CheckedRange { lanes: 128 },
        Construction::AllValid { lanes: 1024 },
    ] {
        let q = family.quote().unwrap();
        verify_phases(q);
        assert_eq!(
            (
                q.public_table_bytes,
                q.constructor_buffer_bytes,
                q.proof_buffer_bytes,
                q.borrowed_public_bytes
            ),
            (0, 0, 0, 0)
        );
    }
    for family in [
        Construction::RmsNormCertified {
            rows: 1,
            width: 1024,
            gamma_len: 1024,
        },
        Construction::RmsNormWithInputNorm {
            rows: 1024,
            width: 1,
            gamma_len: 1,
        },
        Construction::SoftmaxCertified {
            rows: 1,
            width: 1024,
        },
    ] {
        // Geometry endpoint only; this is not numeric/work admission.
        verify_phases(family.quote().unwrap());
    }
    let rope = Construction::RotaryHalf {
        rows: 2,
        width: 64,
        cosine_len: 64,
        sine_len: 64,
    }
    .quote()
    .unwrap();
    verify_phases(rope);
    assert_eq!(rope.borrowed_public_bytes, 128 * size_of::<i64>());
    assert_eq!(rope.public_table_bytes, 2 * 128 * size_of::<Affine>());
    assert_eq!(
        rope.constructor_buffer_bytes,
        2 * size_of::<[Affine; 128]>()
    );
    assert_eq!(rope.proof_buffer_bytes, 0);
    assert_eq!(
        rope.return_slot_bytes,
        size_of::<Result<Program, Error>>().max(size_of::<Result<Builder, Error>>())
            .max(size_of::<Result<FactoryOutput, Error>>()).max(size_of::<Option<FactoryOutput>>())
    );

    let boxes = size_of::<[Option<Interval>; 36]>();
    let ranged = boxes + align_of::<Option<Interval>>();
    for family in [
        Construction::RmsNormCertified {
            rows: 2,
            width: 3,
            gamma_len: 3,
        },
        Construction::RmsNormWithInputNorm {
            rows: 2,
            width: 3,
            gamma_len: 3,
        },
    ] {
        let q = family.quote().unwrap();
        verify_phases(q);
        assert_eq!(q.borrowed_public_bytes, 3 * size_of::<i64>());
        assert_eq!(q.public_table_bytes, 6 * size_of::<Affine>());
        assert_eq!(q.constructor_buffer_bytes, 2 * size_of::<[Affine; 1024]>());
        assert_eq!(
            q.return_slot_bytes,
            size_of::<Result<(Program, u128), Error>>()
                .max(size_of::<Result<(Builder, Value, u128), Error>>())
                .max(size_of::<Result<Builder, Error>>())
                .max(size_of::<Result<FactoryOutput, Error>>())
                .max(size_of::<Option<FactoryOutput>>())
        );
        // Independent lexical inventory: twelve rsqrt bindings, two chained /
        // returned temporaries, four RMS bindings/temporaries, helper return.
        let proof_values = 12 + 2 + 4 + 1;
        let proof_arrays = 2 + 1 + 1 + 1;
        assert_eq!(
            q.proof_buffer_bytes,
            proof_values * ranged
                + proof_arrays * boxes
                + size_of::<[i128; 4]>()
                + 2 * size_of::<[(Interval, i64); 3]>()
                + 2 * size_of::<[(usize, i64); 3]>()
        );
    }
    let softmax = Construction::SoftmaxCertified { rows: 1, width: 3 }
        .quote()
        .unwrap();
    verify_phases(softmax);
    assert_eq!(
        softmax.return_slot_bytes,
        size_of::<Result<(Program, u64), Error>>()
            .max(size_of::<Result<(Program, Option<u64>), Error>>())
            .max(size_of::<Result<Builder, Error>>())
            .max(size_of::<Result<FactoryOutput, Error>>())
            .max(size_of::<Option<FactoryOutput>>())
    );
    assert_eq!(
        softmax.proof_buffer_bytes,
        (11 + 1 + 1) * ranged + 2 * boxes + size_of::<[i128; 4]>() + 2 * size_of::<Interval>()
    );
    assert_eq!(
        (
            softmax.public_table_bytes,
            softmax.borrowed_public_bytes,
            softmax.constructor_buffer_bytes
        ),
        (0, 0, 0)
    );
    let silu = Construction::GuardedSilu { lanes: 128 }.quote().unwrap();
    verify_phases(silu);
    assert_eq!(
        silu.constructor_buffer_bytes,
        size_of::<[(Value, i64); 2]>()
            + size_of::<[(Value, i64); 4]>()
            + (6 * 3 + 2 * 4 + 2 * 5) * size_of::<i64>()
            + 2 * size_of::<[(i64, i64, bool); 2]>()
            + 3 * size_of::<[(Interval, i64); 4]>()
            + 4 * size_of::<[Value; 3]>()
            + size_of::<Option<[Value; 3]>>()
    );
}

#[test]
fn negative_and_equal_zero_have_separate_native_buffer_and_return_inventories() {
    for lanes in [1, 3, 127, 128] {
        let negative = Construction::Negative { lanes }.quote().unwrap();
        let equality = Construction::EqualZero { lanes }.quote().unwrap();
        for q in [negative, equality] {
            verify_phases(q);
            // Native APIs return Program/Builder; closed dispatch additionally
            // overlaps the actual producer-bearing FactoryOutput result/slot.
            assert_eq!(
                q.return_slot_bytes,
                size_of::<Result<Program, Error>>().max(size_of::<Result<Builder, Error>>())
                    .max(size_of::<Result<FactoryOutput, Error>>()).max(size_of::<Option<FactoryOutput>>())
            );
            assert_eq!(q.borrowed_public_bytes, 0);
            assert_eq!(q.public_table_bytes, 0);
            assert_eq!(q.proof_buffer_bytes, 0);
            assert_eq!(
                q.immutable_template_bytes,
                size_of::<Program>() + 2 * size_of::<usize>()
            );
        }
        // Negative has no bit maps or reduction array. Equality has one u8
        // six-shift expression and one owning IntoIter (buffer + cursor state).
        // This independently names both slots rather than borrowing quote math.
        let shifts = size_of::<[u8; 6]>();
        let consuming_shifts = size_of::<std::array::IntoIter<u8, 6>>();
        assert_eq!(negative.constructor_buffer_bytes, 0);
        assert_eq!(equality.constructor_buffer_bytes, shifts + consuming_shifts);
        assert_eq!(
            equality.new_peak_bytes - negative.new_peak_bytes,
            shifts + consuming_shifts
        );
        // The larger equality inventory must not silently widen Builder scope.
        assert_eq!(
            Construction::Builder { lanes }
                .quote()
                .unwrap()
                .constructor_buffer_bytes,
            0
        );
    }
}

#[test]
fn native_predicate_quotes_match_actual_geometry_errors_including_overflow_inputs() {
    for lanes in [0, 129, MAX_ROW_LANES, usize::MAX / 2 + 1, usize::MAX] {
        let input = Interval::new(i64::MIN, i64::MAX).unwrap();
        for (family, actual) in [
            (
                Construction::Negative { lanes },
                Program::negative(lanes, input),
            ),
            (
                Construction::EqualZero { lanes },
                Program::equal_zero(lanes, input),
            ),
        ] {
            assert_eq!(
                family.quote(),
                Err(ConstructionQuoteError::Geometry(Error::Bounds)),
                "{family:?}"
            );
            assert!(matches!(actual, Err(Error::Bounds)), "{family:?}");
        }
    }
}

#[test]
fn native_predicate_quotes_cover_retained_resources_and_original_instruction_contracts() {
    // Independent, explicit instruction inventories. In particular Negative
    // must not inherit equality's bit maps or compact/bounded conversion path.
    let negative_instructions = [
        Instruction::Input,
        Instruction::Sign(Value(0)),
        Instruction::FromBit(Value(1)),
    ];
    let equality_instructions = [
        Instruction::Input,
        Instruction::Bits(Value(0)),
        Instruction::BitMap(Value(1), BitMap::ShiftRight(32)),
        Instruction::BitOr(Value(1), Value(2)),
        Instruction::BitMap(Value(3), BitMap::ShiftRight(16)),
        Instruction::BitOr(Value(3), Value(4)),
        Instruction::BitMap(Value(5), BitMap::ShiftRight(8)),
        Instruction::BitOr(Value(5), Value(6)),
        Instruction::BitMap(Value(7), BitMap::ShiftRight(4)),
        Instruction::BitOr(Value(7), Value(8)),
        Instruction::BitMap(Value(9), BitMap::ShiftRight(2)),
        Instruction::BitOr(Value(9), Value(10)),
        Instruction::BitMap(Value(11), BitMap::ShiftRight(1)),
        Instruction::BitOr(Value(11), Value(12)),
        Instruction::BitMap(Value(13), BitMap::LowBits(1)),
        Instruction::FromBit(Value(14)),
        Instruction::Affine {
            left: Value(15),
            right: None,
            coefficients: Affine {
                left: -1,
                right: 0,
                constant: 1,
            },
        },
    ];
    let negative_widths = [0, 1, 0];
    let equality_widths = [
        0, 64, 32, 64, 48, 64, 56, 64, 60, 64, 62, 64, 63, 64, 1, 0, 0,
    ];
    for lanes in [1, 3, 127, 128] {
        for (lower, upper) in [(i64::MIN, i64::MAX), (-3, 3), (0, 0), (-3, -1), (1, 3)] {
            let input = Interval::new(lower, upper).unwrap();
            let from_bit = WorkBudget::from_boolean(lanes, 1).unwrap();
            let negative_work = WorkBudget::conversion(lanes, true)
                .unwrap()
                .checked_add(from_bit)
                .unwrap();
            let reduction_pair = WorkBudget::local()
                .checked_add(WorkBudget::bit_or(lanes, 64).unwrap())
                .unwrap();
            let equality_work = WorkBudget::conversion(lanes, false)
                .unwrap()
                .checked_add(reduction_pair.checked_mul(6).unwrap())
                .unwrap()
                .checked_add(WorkBudget::local()) // LowBits(1)
                .unwrap()
                .checked_add(from_bit)
                .unwrap()
                .checked_add(WorkBudget::local()) // final affine
                .unwrap();
            for (family, program, instructions, widths, work) in [
                (
                    Construction::Negative { lanes },
                    Program::negative(lanes, input).unwrap(),
                    negative_instructions.as_slice(),
                    negative_widths.as_slice(),
                    negative_work,
                ),
                (
                    Construction::EqualZero { lanes },
                    Program::equal_zero(lanes, input).unwrap(),
                    equality_instructions.as_slice(),
                    equality_widths.as_slice(),
                    equality_work,
                ),
            ] {
                let q = family.quote().unwrap();
                verify_phases(q);
                let resources = program.resources().unwrap();
                assert_eq!(q.public_table_bytes, resources.public_table_bytes);
                assert_eq!(
                    q.immutable_template_bytes,
                    resources.immutable_template_bytes
                );
                assert!(q.new_peak_bytes >= resources.immutable_template_bytes);
                assert_eq!(program.public_affine_count, 0);
                for table in &program.public_affines {
                    assert!(table.is_empty());
                    // Empty non-ZST boxed slice has no backing allocation.
                    assert_eq!(crate::exact_buffer::layout_bytes::<Affine>(table.len()).unwrap(), 0);
                }
                assert_eq!(program.lanes(), lanes);
                assert_eq!(program.output_lanes(), lanes);
                assert_eq!(program.output_bounds(), Interval::new(0, 1).unwrap());
                assert_eq!(program.validity_lanes(), None);
                assert_eq!(program.numeric_profile(), super::super::NUMERIC_PROFILE);
                assert_eq!(program.work_budget(), work);
                assert_eq!(program.instructions(), instructions.len() - 1);
                assert_eq!(program.len, instructions.len());
                assert_eq!(&program.bit_widths[..program.len], widths);
                for (node, instruction) in program.nodes[..program.len].iter().zip(instructions) {
                    let node = node.unwrap();
                    assert_eq!(node.instruction, *instruction);
                    assert_eq!(node.lanes, lanes);
                }
                assert!(program.nodes[program.len..].iter().all(Option::is_none));
            }
            // Existing before/after harness also freezes digest/work/output
            // identity across advisory quoting, without executing any kernel.
            compare_result(Construction::Negative { lanes }, || {
                Program::negative(lanes, input)
            });
            compare_result(Construction::EqualZero { lanes }, || {
                Program::equal_zero(lanes, input)
            });
        }
    }
}

#[test]
fn geometry_quote_rejects_zero_overflow_bad_lengths_and_unsupported_rope_generation() {
    use ConstructionQuoteError::*;
    for family in [
        Construction::Builder { lanes: 0 },
        Construction::Builder { lanes: 129 },
        Construction::CheckedRange { lanes: usize::MAX },
        Construction::AllValid { lanes: 0 },
        Construction::AllValid { lanes: 1025 },
        Construction::GuardedSilu { lanes: 129 },
        Construction::RmsNormCertified {
            rows: usize::MAX,
            width: 2,
            gamma_len: 2,
        },
        Construction::RmsNormWithInputNorm {
            rows: 0,
            width: 3,
            gamma_len: 3,
        },
        Construction::RmsNormCertified {
            rows: 1,
            width: 3,
            gamma_len: 2,
        },
        Construction::RotaryHalf {
            rows: 1,
            width: 3,
            cosine_len: 1,
            sine_len: 1,
        },
        Construction::RotaryHalf {
            rows: 2,
            width: 128,
            cosine_len: 128,
            sine_len: 128,
        },
        Construction::RotaryHalf {
            rows: 1,
            width: 64,
            cosine_len: 32,
            sine_len: 31,
        },
        Construction::SoftmaxCertified { rows: 1, width: 0 },
        Construction::SoftmaxCertified {
            rows: usize::MAX,
            width: 2,
        },
    ] {
        assert!(matches!(family.quote(), Err(Geometry(_))), "{family:?}");
    }
    for (theta, position, width) in [
        (f32::NAN, 0, 2),
        (f32::INFINITY, 0, 2),
        (0.5, 0, 2),
        (1.0, 513, 2),
        (1.0, 0, 0),
        (1.0, 0, 3),
        (1.0, 0, 130),
    ] {
        assert_eq!(
            Construction::RotaryCoefficients {
                theta,
                position,
                width
            }
            .quote(),
            Err(Geometry(Error::Bounds))
        );
        assert!(matches!(
            Program::rotary_coefficients(theta, position, width),
            Err(Error::Bounds)
        ));
    }
    for (theta, position, width) in [(1.0, 0, 2), (10000.0, 512, 128)] {
        assert_eq!(
            Construction::RotaryCoefficients {
                theta,
                position,
                width
            }
            .quote(),
            Err(UncheckedRotaryCoefficientCapacity)
        );
        // Refusing a quote must not silently disable the original API.
        let (cos, sin) = Program::rotary_coefficients(theta, position, width).unwrap();
        assert_eq!((cos.len(), sin.len()), (width / 2, width / 2));
    }
}

fn compare_result(family: Construction, construct: impl Fn() -> Result<Program, Error>) {
    let before = construct();
    let q = family.quote().unwrap();
    let after = construct();
    match (before, after) {
        (Ok(a), Ok(b)) => {
            let resources = a.resources().unwrap();
            assert_eq!(q.public_table_bytes, resources.public_table_bytes);
            assert_eq!(
                q.immutable_template_bytes,
                resources.immutable_template_bytes
            );
            assert!(q.new_peak_bytes >= resources.immutable_template_bytes);
            assert_eq!(a.digest(), b.digest());
            assert_eq!(a.work_budget(), b.work_budget());
            assert_eq!(a.instructions(), b.instructions());
            assert_eq!(a.output_bounds(), b.output_bounds());
            assert_eq!(a.validity_lanes(), b.validity_lanes());
        }
        (Err(a), Err(b)) => panic!("positive fixture failed before/after quote: {a:?}/{b:?}"),
        _ => panic!("quoting changed constructor outcome"),
    }
}

fn compare_error(family: Construction, construct: impl Fn() -> Result<Program, Error>) {
    let before = construct().err().expect("negative fixture must fail");
    family.quote().unwrap();
    let after = construct()
        .err()
        .expect("quote must not admit negative fixture");
    assert_eq!(before, after);
}

#[test]
fn quoted_public_capacities_cover_native_results_without_changing_numerics_or_errors() {
    let bound = Interval::new(-(1 << 18), 1 << 18).unwrap();
    for n in [1, 127, 128] {
        compare_result(Construction::Builder { lanes: n }, || {
            let mut b = Builder::new(n, bound)?;
            let out = b.affine(
                b.input(),
                None,
                Affine {
                    left: 1,
                    right: 0,
                    constant: 7,
                },
            )?;
            let out = b.product(out, b.input(), true)?;
            b.finish(out)
        });
        compare_result(Construction::GuardedSilu { lanes: n }, || {
            Program::activation_guarded(Activation::Silu, n, bound)
        });
        compare_result(Construction::CheckedRange { lanes: n }, || {
            Program::checked_range(n, Interval::new(i64::MIN, i64::MAX)?, bound, 0)
        });
    }
    for n in [1, 128, 1024] {
        compare_result(Construction::AllValid { lanes: n }, || {
            Program::all_valid(n)
        });
    }
    for width in [1, 3, 896] {
        let gamma = vec![1 << 18; width];
        compare_result(
            Construction::RmsNormCertified {
                rows: 1,
                width,
                gamma_len: width,
            },
            || Program::rms_norm_certified(1, width, bound, &gamma).map(|v| v.0),
        );
        compare_result(
            Construction::RmsNormWithInputNorm {
                rows: 1,
                width,
                gamma_len: width,
            },
            || {
                Program::rms_norm_with_input_norm(1, width, bound, &gamma, (width as u128) << 36)
                    .map(|v| v.0)
            },
        );
    }
    for width in [2, 64, 128] {
        let cos = vec![1 << 18; width / 2];
        let sin = vec![0; width / 2];
        compare_result(
            Construction::RotaryHalf {
                rows: 1,
                width,
                cosine_len: cos.len(),
                sine_len: sin.len(),
            },
            || Program::rotary_half(1, width, bound, &cos, &sin),
        );
    }
    for width in [1, 3, 128] {
        compare_result(Construction::SoftmaxCertified { rows: 1, width }, || {
            Program::softmax_certified(1, width, bound).map(|v| v.0)
        });
    }
    // Quote validates storage geometry only: unsafe numerical inputs still fail
    // in their ORIGINAL constructor, with the same error after an advisory quote.
    compare_error(Construction::CheckedRange { lanes: 3 }, || {
        Program::checked_range(3, bound, bound, i64::MAX)
    });
    compare_error(
        Construction::RmsNormWithInputNorm {
            rows: 1,
            width: 3,
            gamma_len: 3,
        },
        || {
            Program::rms_norm_with_input_norm(1, 3, Interval::new(1, 2)?, &[1 << 18; 3], 0)
                .map(|v| v.0)
        },
    );
}

#[test]
fn construction_quote_supports_public_literal_update_and_exhaustive_destructure() {
    let literal = ConstructionQuote {
        return_slot_bytes: 11,
        borrowed_public_bytes: 23,
        public_table_bytes: 31,
        constructor_buffer_bytes: 43,
        proof_buffer_bytes: 59,
        finish_buffer_bytes: 61,
        immutable_template_bytes: 73,
        new_build_bytes: 89,
        new_finish_bytes: 97,
        new_publish_bytes: 101,
        new_peak_bytes: 109,
        peak_with_borrowed_public_bytes: 127,
    };
    let updated = ConstructionQuote { new_peak_bytes: 137, ..literal };
    let ConstructionQuote {
        return_slot_bytes,
        borrowed_public_bytes,
        public_table_bytes,
        constructor_buffer_bytes,
        proof_buffer_bytes,
        finish_buffer_bytes,
        immutable_template_bytes,
        new_build_bytes,
        new_finish_bytes,
        new_publish_bytes,
        new_peak_bytes,
        peak_with_borrowed_public_bytes,
    } = updated;
    assert_eq!(return_slot_bytes, 11);
    assert_eq!(borrowed_public_bytes, 23);
    assert_eq!(public_table_bytes, 31);
    assert_eq!(constructor_buffer_bytes, 43);
    assert_eq!(proof_buffer_bytes, 59);
    assert_eq!(finish_buffer_bytes, 61);
    assert_eq!(immutable_template_bytes, 73);
    assert_eq!(new_build_bytes, 89);
    assert_eq!(new_finish_bytes, 97);
    assert_eq!(new_publish_bytes, 101);
    assert_eq!(new_peak_bytes, 137);
    assert_eq!(peak_with_borrowed_public_bytes, 127);
    assert_eq!(literal.new_peak_bytes, 109);
}

#[test]
fn installed_resident_quote_retains_inline_return_overlap_and_fallback_builder() {
    // Arithmetic consistency only, not observed lifetimes or request funding.
    for construction in [
        Construction::RmsNormCertified { rows: 1, width: 3, gamma_len: 3 },
        Construction::RmsNormWithInputNorm { rows: 1, width: 3, gamma_len: 3 },
        Construction::RotaryHalf { rows: 1, width: 4, cosine_len: 2, sine_len: 2 },
    ] {
        let (q, peak) = construction.quote_for_installed_resident().unwrap();
        assert_eq!(q, construction.quote().unwrap());
        verify_phases(q);
        let build = size_of::<Program>() + q.return_slot_bytes + q.public_table_bytes
            + q.constructor_buffer_bytes + q.proof_buffer_bytes;
        let finish = q.return_slot_bytes + q.public_table_bytes + q.finish_buffer_bytes;
        let publish = q.return_slot_bytes + q.immutable_template_bytes;
        assert!(peak >= build.max(finish).max(publish));
        if matches!(construction, Construction::RotaryHalf { .. }) {
            assert_eq!(peak, build.max(finish).max(publish));
        } // exact sealed RMS peak is reconstructed in its private owner tests
        assert!(q.return_slot_bytes >= size_of::<Result<FactoryOutput, Error>>());
        assert!(q.return_slot_bytes >= size_of::<Option<FactoryOutput>>());
        assert!(q.new_build_bytes >= 2 * size_of::<Builder>() + q.return_slot_bytes
            + q.public_table_bytes + q.constructor_buffer_bytes + q.proof_buffer_bytes);
    }
    for construction in [
        Construction::Builder { lanes: 3 },
        Construction::Negative { lanes: 3 },
        Construction::EqualZero { lanes: 3 },
        Construction::CheckedRange { lanes: 3 },
        Construction::AllValid { lanes: 3 },
        Construction::GuardedSilu { lanes: 3 },
        Construction::SoftmaxCertified { rows: 1, width: 3 },
    ] {
        let (q, peak) = construction.quote_for_installed_resident().unwrap();
        let ordinary = construction.quote().unwrap();
        assert_eq!(q, ordinary);
        assert_eq!(peak, ordinary.new_peak_bytes);
        verify_phases(q);
        assert!(q.new_build_bytes >= 2 * size_of::<Builder>() + q.return_slot_bytes);
    }
    let coefficients = Construction::RotaryCoefficients { theta: 10_000.0, position: 0, width: 4 };
    assert_eq!(coefficients.quote(), Err(ConstructionQuoteError::UncheckedRotaryCoefficientCapacity));
    assert_eq!(coefficients.quote_for_installed_resident(),
        Err(ConstructionQuoteError::UncheckedRotaryCoefficientCapacity));
}

#[test]
fn native_event_categories_separate_affine_attempts_from_caller_arc_and_polynomial() {
    let rotary = Construction::RotaryHalf { rows: 1, width: 4, cosine_len: 2, sine_len: 2 }.logical_events().unwrap();
    assert_eq!(rotary.allocation_attempts, 2); // no caller Arc new hidden here
    assert_eq!(rotary.cleanup_dispatches, 3 + 2 * 2);
    let rms = Construction::RmsNormCertified { rows: 1, width: 3, gamma_len: 3 }.logical_events().unwrap();
    assert_eq!(rms.allocation_attempts, 1);
    let polynomial_writes = 36 * 257 * (9 * (4 + 4 + 8 + 4 + 8 + 4) + 32);
    assert!(rms.writes >= polynomial_writes);
    assert!(rms.validation > rotary.validation);
    assert_eq!(rms.total().unwrap(), rms.writes + rms.validation + rms.transitions
        + rms.allocation_attempts + rms.cleanup_dispatches);
    assert!(LogicalEvents { writes: u64::MAX, ..LogicalEvents::default() }
        .checked_add(LogicalEvents { writes: 1, ..LogicalEvents::default() }).is_err());
    assert!(rms.checked_mul(u64::MAX).is_err());
    // These recipes have no adopted/new logical inventory in this cut. Numeric
    // constructors and their original protocol-work contracts are unchanged.
    assert!(Construction::SoftmaxCertified { rows: 1, width: 3 }.logical_events().is_err());
    let n = MAX_NODES as u64;
    assert_eq!(Program::full_resource_validation_visits().unwrap(),
        64 * (3 + n + n * (n + 1) / 2) + predicate_provenance::MAX_VALIDATION_VISITS);
}

#[test]
fn construction_quote_does_not_enter_frozen_digest_or_certificate_equations() {
    let q = Construction::Builder { lanes: 3 }.quote().unwrap();
    let mut b = Builder::new(3, Interval::new(-3, 3).unwrap()).unwrap();
    let out = b
        .affine(
            b.input(),
            None,
            Affine {
                left: 2,
                right: 0,
                constant: 1,
            },
        )
        .unwrap();
    let p = b.finish(out).unwrap();
    let mut h = Sha256::new();
    h.update(b"pllm.puma.public-tensor-program.q18.v1");
    h.update(b"puma.source-derived-q18.v1");
    h.update(3_u64.to_le_bytes());
    h.update(2_u64.to_le_bytes());
    h.update([0]);
    h.update((-3_i64).to_le_bytes());
    h.update(3_i64.to_le_bytes());
    h.update([1, 0, 255]);
    for c in [2_i64, 0, 1] {
        h.update(c.to_le_bytes());
    }
    h.update((-5_i64).to_le_bytes());
    h.update(7_i64.to_le_bytes());
    assert_eq!(p.digest(), <[u8; 32]>::from(h.finalize()));
    assert_eq!(p.work_budget(), WorkBudget::local());
    assert_eq!(
        q.immutable_template_bytes,
        p.resources().unwrap().immutable_template_bytes
    );

    pre_extension_legacy_vectors::assert_factories();

    // Retain the original nontrivial-width quote NONINTERFERENCE checks too.
    // These snapshots test quote purity only, not historical identity: the
    // independent Git-derived expectations above supply that separate oracle.
    let bound = Interval::new(-(1 << 18), 1 << 18).unwrap();
    let (a, radius_a) = Program::rms_norm_certified(1, 3, bound, &[1 << 18; 3]).unwrap();
    Construction::RmsNormCertified { rows: 1, width: 3, gamma_len: 3 }.quote().unwrap();
    let (b, radius_b) = Program::rms_norm_certified(1, 3, bound, &[1 << 18; 3]).unwrap();
    assert_eq!((a.digest(), a.work_budget(), radius_a),
        (b.digest(), b.work_budget(), radius_b));
    let (a, mass_a) = Program::softmax_certified(1, 3, bound).unwrap();
    Construction::SoftmaxCertified { rows: 1, width: 3 }.quote().unwrap();
    let (b, mass_b) = Program::softmax_certified(1, 3, bound).unwrap();
    assert_eq!((a.digest(), a.work_budget(), mass_a),
        (b.digest(), b.work_budget(), mass_b));
}

mod pre_extension_legacy_vectors {
    use super::*;

    // Git provenance: 3a21be5a3a8922ab2b9d363805cbe584ef1240cd,
    // crates/pllm-core/src/puma/{tensor.rs,tensor_numeric.rs,
    // tensor_attention_work.rs,arithmetic.rs,work.rs}. `git show` confirms that
    // Program has no indexed extension; numeric.rs matches the current source.
    // These ONE-LANE fixtures transcribe the old RMS/softmax compositions and
    // finish byte tags, NOT a second general numeric implementation. Their
    // single positive exponent/point proof cell needs no bucket/partition code.
    // Expectations never use Builder, production interval/proof/work helpers,
    // Instruction::inputs, BitMap::hash, finish, or a current constructor.
    const Q: i64 = 262_144;
    const LOCAL: [u64; 5] = [1, 0, 0, 0, 0];
    const RESCALE: [u64; 5] = [1, 1, 2, 1, 6];
    // Work columns: operations, public, pair, private, AES. Each source fill
    // rounds separately: at one lane an 8-byte fill and a 2*8-byte fill each
    // cost one block; full A2B has two initial fills and six carry fills.
    struct Stream {
        rows: Vec<(Vec<u8>, [i64; 2], u8)>,
        work: [u64; 5],
    }
    impl Stream {
        fn new() -> Self {
            let mut s = Self { rows: Vec::new(), work: [0; 5] };
            s.emit(vec![0], [0, 0], 0, [0; 5]);
            s
        }
        fn emit(&mut self, bytes: Vec<u8>, bounds: [i64; 2], width: u8, work: [u64; 5]) -> u8 {
            let id = u8::try_from(self.rows.len()).unwrap();
            self.rows.push((bytes, bounds, width));
            for (total, add) in self.work.iter_mut().zip(work) { *total += add; }
            id
        }
        fn bounds(&self, id: u8) -> [i64; 2] { self.rows[id as usize].1 }
        fn affine(&mut self, a: u8, b: Option<u8>, c: [i64; 3]) -> u8 {
            let term = |v: [i64; 2], c: i64| {
                let x = i128::from(v[0]) * i128::from(c);
                let y = i128::from(v[1]) * i128::from(c);
                [x.min(y), x.max(y)]
            };
            let left = term(self.bounds(a), c[0]);
            let right = term(b.map_or([0, 0], |b| self.bounds(b)), c[1]);
            let bounds = std::array::from_fn(|i|
                i64::try_from(left[i] + right[i] + i128::from(c[2])).unwrap());
            let mut bytes = vec![1, a, b.unwrap_or(255)];
            for coefficient in c { bytes.extend(coefficient.to_le_bytes()); }
            self.emit(bytes, bounds, 0, LOCAL)
        }
        fn product(&mut self, a: u8, b: u8, rescale: bool, certified: bool) -> u8 {
            let x = self.bounds(a);
            let y = self.bounds(b);
            let corners = [i128::from(x[0]) * i128::from(y[0]),
                i128::from(x[0]) * i128::from(y[1]), i128::from(x[1]) * i128::from(y[0]),
                i128::from(x[1]) * i128::from(y[1])];
            let raw = [i64::try_from(*corners.iter().min().unwrap()).unwrap(),
                i64::try_from(*corners.iter().max().unwrap()).unwrap()];
            let mut bytes = vec![2, a, b, u8::from(rescale)];
            if certified {
                bytes.extend(b"conditional-product-envelope.v1");
                for bound in raw { bytes.extend(bound.to_le_bytes()); }
            }
            self.emit(bytes, if rescale { rounded(raw, 18) } else { raw }, 0,
                if rescale { [1, 1, 3, 1, 8] } else { [1, 0, 1, 0, 2] })
        }
        fn rescale(&mut self, a: u8, shift: u8) -> u8 {
            self.emit(vec![3, a, shift], rounded(self.bounds(a), shift), 0, RESCALE)
        }
        fn map(&mut self, a: u8, tag: u8, parameter: u64, width: u8) -> u8 {
            let mut bytes = vec![7, a, tag];
            bytes.extend(parameter.to_le_bytes());
            self.emit(bytes, [0, 0], width, LOCAL)
        }
        fn highest(&mut self, a: u8, width: u8) -> u8 {
            let bits = self.emit(vec![6, a], [0, 0], 64, [1, 0, 8, 0, 16]);
            let mut spread = self.map(bits, 0, u64::from(width), width);
            for shift in [1_u8, 2, 4, 8, 16, 32] {
                let lower = self.map(spread, 1, u64::from(shift), width.saturating_sub(shift));
                spread = self.emit(vec![8, spread, lower], [0, 0], width, [3, 0, 1, 0, 2]);
            }
            let lower = self.map(spread, 1, 1, width - 1);
            self.emit(vec![9, spread, lower], [0, 0], width, LOCAL)
        }
        fn from(&mut self, a: u8, width: u8, exact: Option<i64>) -> u8 {
            self.emit(vec![5, a], exact.map_or([0, (1_i64 << width) - 1], |v| [v, v]), 0,
                if width == 1 { [1, 1, 3, 0, 5] } else {
                    assert!(width > 8); [1, 0, 9, 0, 18]
                })
        }
        fn row(&mut self, a: u8, broadcast: bool) -> u8 {
            let mut bytes = vec![10, a];
            bytes.extend(1_u64.to_le_bytes());
            bytes.extend((u64::MAX - u64::from(broadcast)).to_le_bytes());
            self.emit(bytes, self.bounds(a), 0, LOCAL)
        }
        fn assert_program(&self, p: &Program, profile: &[u8]) {
            let mut hash = Sha256::new();
            hash.update(b"pllm.puma.public-tensor-program.q18.v1");
            hash.update(profile);
            hash.update(1_u64.to_le_bytes());
            hash.update((self.rows.len() as u64).to_le_bytes());
            assert_eq!(p.len, self.rows.len());
            assert_eq!(p.numeric_profile().as_bytes(), profile);
            assert!(p.extension.absent());
            assert_eq!(p.checked_validity, None);
            for (i, (bytes, bounds, width)) in self.rows.iter().enumerate() {
                hash.update(bytes);
                for bound in bounds { hash.update(bound.to_le_bytes()); }
                assert_eq!(p.bit_widths[i], *width, "legacy width at {i}");
                let node = p.nodes[i].unwrap();
                assert_eq!((node.bounds.lower(), node.bounds.upper()), (bounds[0], bounds[1]));
                assert_eq!((node.lanes, node.bit), (1, *width != 0));
            }
            assert_eq!(p.digest(), <[u8; 32]>::from(hash.finalize()));
            let c = p.work_budget().cursors();
            assert_eq!([p.work_budget().operations(), c.public_blocks, c.pair_blocks,
                c.private_blocks, c.aes_work], self.work);
        }
    }
    fn rounded(bounds: [i64; 2], shift: u8) -> [i64; 2] {
        let q = 1_i128 << shift;
        [i64::try_from(i128::from(bounds[0]).div_euclid(q)).unwrap(),
            i64::try_from(-(-i128::from(bounds[1])).div_euclid(q)).unwrap()]
    }

    // Zero row, width one: epsilon-one denominator=1, exponent=0, U=Q/4.
    // All three powers are exact integers. The partitioned polynomial proof
    // has ONE point cell, so its result is exactly this literal interval chain.
    fn rms(gamma: i64) -> Stream {
        let mut s = Stream::new();
        let square = s.product(0, 0, true, true);
        let sum = s.row(square, false);
        let raw_mean = s.affine(sum, None, [Q, 0, 0]);
        let mean = s.rescale(raw_mean, 18);
        let denominator = s.affine(mean, None, [1, 0, 1]);
        let highest = s.highest(denominator, 35);
        let z = s.map(highest, 2, 1, 36);
        let reversed = s.map(z, 3, 36, 36);
        let factor = s.from(reversed, 36, Some(1 << 34));
        let u = s.product(denominator, factor, true, true);
        let u2 = s.product(u, u, true, true);
        let u3 = s.product(u2, u, true, true);
        let u4 = s.product(u3, u, true, true);
        let mut raw = s.affine(u, None, [-4_057_974, 0, 0]);
        for (power, coefficient) in [(u2, 10_085_068), (u3, -13_072_088), (u4, 6_823_457)] {
            raw = s.affine(raw, Some(power), [1, coefficient, 0]);
        }
        let scaled = s.rescale(raw, 18);
        let polynomial = s.affine(scaled, None, [1, 0, 1_086_023]);
        let parity = s.map(z, 5, 0x5555_5555_5555_5555, 1);
        let parity = s.from(parity, 1, Some(0));
        let coefficient = s.affine(parity, None, [212, 0, 512]);
        let folded = s.map(z, 4, 0, 18);
        let reversed = s.map(folded, 3, 18, 18);
        let scale = s.from(reversed, 18, Some(1 << 17));
        let compensation = s.product(coefficient, scale, false, true);
        let inverse = s.product(polynomial, compensation, true, true);
        let inverse = s.row(inverse, true);
        let normalized = s.product(0, inverse, true, true);
        let mut bytes = vec![11, normalized, 0];
        bytes.extend(gamma.to_le_bytes());
        bytes.extend(0_i64.to_le_bytes());
        let scaled = s.emit(bytes, [0, 0], 0, LOCAL);
        s.rescale(scaled, 18);
        s
    }

    // Zero score, width one: total's public lower bound refines [0,Q] to
    // [Q,Q]. Reciprocal mass proof has ONE point cell at exponent 18; no use
    // of the current reciprocal primitive or reciprocal_mass_bounds here.
    fn softmax() -> (Stream, u64) {
        let mut s = Stream::new();
        let z = s.affine(0, Some(0), [1, -1, 0]);
        let threshold = s.affine(z, None, [-1, 0, -14 * Q]);
        let sign = s.emit(vec![4, threshold], [0, 1], 1, [1, 0, 8, 0, 16]);
        let keep = s.from(sign, 1, None);
        let safe = s.product(z, keep, false, false);
        let reduced = s.rescale(safe, 5);
        let mut exp = s.affine(reduced, None, [1, 0, Q]);
        for _ in 0..5 { exp = s.product(exp, exp, true, true); }
        let exp = s.product(exp, keep, false, false);
        let total = s.row(exp, false);
        s.rows[total as usize].1 = [Q, Q];
        let highest = s.highest(total, 36);
        let reversed = s.map(highest, 3, 36, 36);
        let factor = s.from(reversed, 36, Some(1 << 17));
        let c = s.product(total, factor, true, true);
        let guess = s.affine(c, None, [-2, 0, 763_940]);
        let w = s.product(guess, factor, true, true);
        let xw = s.product(total, w, true, true);
        let e = s.affine(xw, None, [-1, 0, Q]);
        let one_e = s.affine(e, None, [1, 0, Q]);
        let r = s.product(w, one_e, true, true);
        let e2 = s.product(e, e, true, true);
        let one_e2 = s.affine(e2, None, [1, 0, Q]);
        let inverse = s.product(r, one_e2, true, true);
        // ceil(Q*inverse.upper/Q), width=1; the independent point-cell
        // arithmetic above is also the old certificate calculation.
        let mass = u64::try_from(s.bounds(inverse)[1]).unwrap();
        let inverse = s.row(inverse, true);
        s.product(exp, inverse, true, false);
        (s, mass)
    }

    pub(super) fn assert_factories() {
        for input_norm in [false, true] {
            let gamma = if input_norm { -Q } else { Q };
            let expected = rms(gamma);
            let construction = if input_norm {
                Construction::RmsNormWithInputNorm { rows: 1, width: 1, gamma_len: 1 }
            } else { Construction::RmsNormCertified { rows: 1, width: 1, gamma_len: 1 } };
            // Reconstruct before AND after quote; neither current output is an
            // oracle for the other. E<=0 refines declared [-1,1] to zero first.
            for _ in 0..2 {
                let (program, energy) = if input_norm {
                    Program::rms_norm_with_input_norm(1, 1, Interval::new(-1, 1).unwrap(), &[gamma], 0)
                } else { Program::rms_norm_certified(1, 1, Interval::new(0, 0).unwrap(), &[gamma]) }.unwrap();
                expected.assert_program(&program, b"puma.rmsnorm-epsilon1-q18.v1");
                assert_eq!(program.public_affine_count, 1);
                assert_eq!(program.public_affines[0].as_ref(), &[Affine { left: gamma, right: 0, constant: 0 }]);
                assert!(program.public_affines[1].is_empty());
                // E=0 => ceil(sqrt(E*r^2)/Q)+ceil(sqrt(1))=1;
                // |gamma|=Q adds the final rounding guard: (1+1)^2=4.
                assert_eq!(energy, 4);
                let gamma = [gamma];
                let parameters = if input_norm {
                    FactoryParameters::RmsNormWithInputNorm { input: Interval::new(-1, 1).unwrap(), gamma: &gamma, squared_norm: 0 }
                } else { FactoryParameters::RmsNormCertified { input: Interval::new(0, 0).unwrap(), gamma: &gamma } };
                let output = build(construction, parameters);
                expected.assert_program(output.program(), b"puma.rmsnorm-epsilon1-q18.v1");
                assert_eq!(output.certificate(), ProducedCertificate::GeneralOutputRowSquared(4));
                construction.quote().unwrap();
            }
        }
        let (expected, mass) = softmax();
        for _ in 0..2 {
            let (program, actual_mass) = Program::softmax_certified(1, 1, Interval::new(0, 0).unwrap()).unwrap();
            expected.assert_program(&program, b"puma.source-derived-q18.v1");
            assert_eq!(actual_mass, mass);
            assert_eq!(program.public_affine_count, 0);
            assert!(program.public_affines.iter().all(|table| table.is_empty()));
            let output = build(Construction::SoftmaxCertified { rows: 1, width: 1 },
                FactoryParameters::SoftmaxCertified { input: Interval::new(0, 0).unwrap() });
            expected.assert_program(output.program(), b"puma.source-derived-q18.v1");
            assert_eq!(output.certificate(), ProducedCertificate::AttentionMass(mass));
            Construction::SoftmaxCertified { rows: 1, width: 1 }.quote().unwrap();
        }
    }
    fn build(construction: Construction, parameters: FactoryParameters<'_>) -> FactoryOutput {
        let factory = construction.bind(parameters).unwrap();
        let mut workspace = WorkPreflightWorkspace::empty();
        let ticket = factory.preflight(1, &mut workspace).unwrap();
        let mut total = WorkBudget::default();
        let admitted = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap()
            .admit(ticket).unwrap();
        let (full, tail) = admitted.into_parts();
        assert!(tail.is_none());
        full.build().unwrap()
    }
}
