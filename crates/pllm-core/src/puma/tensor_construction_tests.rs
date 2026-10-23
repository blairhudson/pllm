use super::*;
use std::mem::{align_of, size_of};

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
    let owners = q.return_slot_bytes + size_of::<Builder>() + q.public_table_bytes;
    let scratch = q.constructor_buffer_bytes + q.proof_buffer_bytes;
    assert_eq!(
        q.finish_buffer_bytes,
        size_of::<Sha256>() + MAX_NODES * size_of::<usize>()
    );
    assert_eq!(q.new_build_bytes, owners + scratch);
    assert_eq!(q.new_finish_bytes, owners + scratch + q.finish_buffer_bytes);
    assert_eq!(
        q.new_publish_bytes,
        q.return_slot_bytes + q.immutable_template_bytes + scratch
    );
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
            // Both APIs return Result<Program>; new/row returns Result<Builder>.
            // Neither path returns a certificate, tuple or coefficient owner.
            assert_eq!(
                q.return_slot_bytes,
                size_of::<Result<Program, Error>>().max(size_of::<Result<Builder, Error>>())
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
                    assert_eq!(table.capacity(), 0); // no backing or deep copy
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

    let bound = Interval::new(-(1 << 18), 1 << 18).unwrap();
    let (a, radius_a) = Program::rms_norm_certified(1, 3, bound, &[1 << 18; 3]).unwrap();
    Construction::RmsNormCertified {
        rows: 1,
        width: 3,
        gamma_len: 3,
    }
    .quote()
    .unwrap();
    let (b, radius_b) = Program::rms_norm_certified(1, 3, bound, &[1 << 18; 3]).unwrap();
    assert_eq!(
        (a.digest(), a.work_budget(), radius_a),
        (b.digest(), b.work_budget(), radius_b)
    );
    let (a, mass_a) = Program::softmax_certified(1, 3, bound).unwrap();
    Construction::SoftmaxCertified { rows: 1, width: 3 }
        .quote()
        .unwrap();
    let (b, mass_b) = Program::softmax_certified(1, 3, bound).unwrap();
    assert_eq!(
        (a.digest(), a.work_budget(), mass_a),
        (b.digest(), b.work_budget(), mass_b)
    );
}
