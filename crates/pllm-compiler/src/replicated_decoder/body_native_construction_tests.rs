//! Native quote integration tests; no allocator instrumentation or replacement
//! executor. Factory closures observe the original chunks boundary directly.
use super::*;
use std::{cell::Cell, mem::size_of_val};

fn builder(lanes: usize) -> Builder {
    Builder::new(
        WeightMode::Public,
        &[Register {
            lanes,
            bounds: Interval::new(-ONE, ONE).unwrap(),
        }],
    )
    .unwrap()
}

fn assert_unpublished_and_retired(b: &mut Builder) {
    assert!(b.proof_admission_retired);
    assert!(b.templates.is_empty());
    assert!(b.template_metadata.is_empty());
    assert!(b.ops.is_empty());
    assert!(b.proofs.is_empty());
    assert_eq!(b.registers.len(), 1);
    assert_eq!(b.regions.len(), 1);
    assert_eq!(b.work, WorkBudget::default());
    assert_eq!(b.advances, 1);
    assert!(matches!(b.input(0), Err(Error::Binding)));
}

#[test]
fn native_gate_exact_one_short_adds_caller_and_borrowed_baselines() {
    for construction in [
        tensor::Construction::Builder { lanes: 3 },
        tensor::Construction::GuardedSilu { lanes: 3 },
        tensor::Construction::CheckedRange { lanes: 3 },
        tensor::Construction::AllValid { lanes: 3 },
        tensor::Construction::RmsNormCertified {
            rows: 1,
            width: 3,
            gamma_len: 3,
        },
        tensor::Construction::RmsNormWithInputNorm {
            rows: 1,
            width: 3,
            gamma_len: 3,
        },
        tensor::Construction::SoftmaxCertified { rows: 1, width: 3 },
        tensor::Construction::RotaryHalf {
            rows: 1,
            width: 2,
            cosine_len: 1,
            sine_len: 1,
        },
    ] {
        let q = construction.quote().unwrap();
        // Independent composition: fresh Builder's exact container census,
        // caller-held capacity, supplied public elements, native owner peak.
        let required = Builder::input_metadata_bytes(1).unwrap()
            + 37
            + q.borrowed_public_bytes
            + q.new_peak_bytes;
        for short in [false, true] {
            let mut b = builder(3);
            b.proof_metadata_limit = required - usize::from(short);
            let result = b.native_construction_preflight(construction, 37);
            if short {
                assert!(matches!(result, Err(Error::Bounds)), "{construction:?}");
                assert_unpublished_and_retired(&mut b);
            } else {
                result.unwrap();
                assert!(!b.proof_admission_retired);
                assert!(b.templates.is_empty());
            }
        }
    }
}

#[test]
fn native_gate_rejects_addition_geometry_overflow_and_unsupported_generator() {
    for (construction, caller) in [
        (tensor::Construction::Builder { lanes: 3 }, usize::MAX),
        (
            tensor::Construction::RmsNormCertified {
                rows: usize::MAX,
                width: 2,
                gamma_len: 2,
            },
            0,
        ),
        (
            tensor::Construction::RotaryHalf {
                rows: 1,
                width: 2,
                cosine_len: 1,
                sine_len: 0,
            },
            0,
        ),
        (
            tensor::Construction::RotaryCoefficients {
                theta: 10000.0,
                position: 0,
                width: 2,
            },
            0,
        ),
    ] {
        let mut b = builder(3);
        assert!(b
            .native_construction_preflight(construction, caller)
            .is_err());
        assert_unpublished_and_retired(&mut b);
    }
}

#[test]
fn chunks_one_short_never_calls_factory_exact_reaches_original_factory_error() {
    for guarded in [false, true] {
        for short in [false, true] {
            let mut b = builder(TILE + 1);
            let input = b.input(0).unwrap();
            let construction = |lanes| {
                if guarded {
                    tensor::Construction::GuardedSilu { lanes }
                } else {
                    tensor::Construction::Builder { lanes }
                }
            };
            let count = input.len().div_ceil(TILE);
            let required = Builder::input_metadata_bytes(1).unwrap()
                + count * size_of::<Arc<tensor::Program>>()
                + Builder::proof_row_scratch(&input).unwrap()
                + construction(TILE).quote().unwrap().new_peak_bytes;
            b.proof_metadata_limit = required - usize::from(short);
            let called = Cell::new(false);
            let result = b.chunks(&input, construction, |_, _| {
                called.set(true);
                // Deliberate ORIGINAL factory failure, after successful gate.
                Err(Error::Allocation)
            });
            assert_eq!(called.get(), !short);
            assert!(matches!(result, Err(error) if error == if short {
                Error::Bounds
            } else {
                Error::Allocation
            }));
            assert_unpublished_and_retired(&mut b);
        }
    }
}

#[test]
fn checked_and_producer_family_one_short_rejects_before_template_publication() {
    // Both guard-list backings coexist before the first checked native factory.
    let mut b = builder(3);
    let input = b.input(0).unwrap();
    let required = Builder::input_metadata_bytes(1).unwrap()
        + 2 * size_of::<Arc<tensor::Program>>()
        + Builder::proof_row_scratch(&input).unwrap()
        + tensor::Construction::CheckedRange { lanes: 3 }
            .quote()
            .unwrap()
            .new_peak_bytes;
    b.proof_metadata_limit = required - 1;
    assert!(matches!(
        b.checked(&input, ONE / 2, None),
        Err(Error::Bounds)
    ));
    assert_unpublished_and_retired(&mut b);

    for rms in [false, true] {
        let mut b = builder(3);
        let input = b.input(0).unwrap();
        let construction = if rms {
            tensor::Construction::RmsNormCertified {
                rows: 1,
                width: 3,
                gamma_len: 3,
            }
        } else {
            tensor::Construction::SoftmaxCertified { rows: 1, width: 3 }
        };
        let quote = construction.quote().unwrap();
        let required = Builder::input_metadata_bytes(1).unwrap()
            + size_of::<Arc<tensor::Program>>()
            + size_of::<(usize, Option<u128>, Option<u64>)>()
            + Builder::proof_row_scratch(&input).unwrap()
            + quote.new_peak_bytes
            + if rms { 3 * size_of::<i64>() } else { 0 };
        b.proof_metadata_limit = required - 1;
        if rms {
            assert!(matches!(
                b.rms(&input, 3, &[ONE; 3], None, None),
                Err(Error::Bounds)
            ));
        } else {
            assert!(matches!(b.softmax(&input, 1, 2), Err(Error::Bounds)));
        }
        assert_unpublished_and_retired(&mut b);
    }
}

#[test]
fn checked_retains_original_native_guard_and_aggregate_identity() {
    let mut b = builder(3);
    let input = b.input(0).unwrap();
    let accepted = Interval::new(-ONE / 2, ONE / 2).unwrap();
    let guard = tensor::Program::checked_range(3, b.bounds(&input).unwrap(), accepted, 0).unwrap();
    let all_valid = tensor::Program::all_valid(3).unwrap();
    let (safe, valid) = b.checked(&input, ONE / 2, None).unwrap();
    assert!(b.templates.contains_key(&guard.digest()));
    assert!(b.templates.contains_key(&all_valid.digest()));
    assert_eq!(
        b.work,
        guard
            .work_budget()
            .checked_add(all_valid.work_budget())
            .unwrap()
    );
    assert_eq!(b.bounds(&safe).unwrap(), accepted);
    assert_eq!(
        b.bounds(&valid.unwrap()).unwrap(),
        Interval::new(0, 1).unwrap()
    );
}

#[test]
fn native_gate_recensuses_templates_instead_of_reusing_initial_budget() {
    let construction = tensor::Construction::AllValid { lanes: 3 };
    let required =
        Builder::input_metadata_bytes(1).unwrap() + construction.quote().unwrap().new_peak_bytes;
    let mut b = builder(3);
    b.native_construction_preflight(construction, 0).unwrap();
    b.intern(tensor::Program::all_valid(3).unwrap()).unwrap();
    let work = b.work;
    b.proof_metadata_limit = required;
    assert!(matches!(
        b.native_construction_preflight(construction, 0),
        Err(Error::Bounds)
    ));
    assert!(b.proof_admission_retired);
    assert_eq!(b.templates.len(), 1);
    assert_eq!(b.work, work);
    assert!(b.ops.is_empty());
}

#[test]
fn legacy_generated_vec_quote_remains_unsupported() {
    assert_eq!(
        tensor::Construction::RotaryCoefficients {
            theta: 10000.0,
            position: 0,
            width: 2,
        }
        .quote(),
        Err(tensor::ConstructionQuoteError::UncheckedRotaryCoefficientCapacity)
    );
}

#[test]
fn rope_initialization_gate_prices_full_arrays_and_source_destination_overlap() {
    assert_eq!(ROTARY_COEFFICIENT_BYTES, 2 * size_of::<[i64; 64]>());
    assert_eq!(ROTARY_INITIALIZATION_BYTES, 4 * size_of::<[i64; 64]>());
    for rows in [1, 4] {
        for short in [false, true] {
            let mut b = builder(rows * 2);
            let input = b.input(0).unwrap();
            let required = Builder::input_metadata_bytes(1).unwrap()
                + rows * size_of::<Arc<tensor::Program>>()
                + Builder::proof_row_scratch(&input).unwrap()
                // Independent inventory: two arrays, each initializer plus
                // destination. Width two still owns all 128 backing words.
                + 4 * 64 * size_of::<i64>();
            b.proof_metadata_limit = required - usize::from(short);
            // Exactly the production pre-array/list gate, no fill or native
            // program has run yet. Destination capacity is checked by its owner.
            let result = b.new_template_list(&input, rows, ROTARY_INITIALIZATION_BYTES);
            if short {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_unpublished_and_retired(&mut b);
            } else {
                assert_eq!(result.unwrap().capacity(), rows);
                assert!(!b.proof_admission_retired);
            }
        }
    }
}

#[test]
fn rope_native_gate_exact_one_short_precedes_fill_and_program_factory() {
    for width in [2, 64, 128] {
        for short in [false, true] {
            let mut b = builder(width);
            let input = b.input(0).unwrap();
            let quote = tensor::Construction::RotaryHalf {
                rows: 1,
                width,
                cosine_len: width / 2,
                sine_len: width / 2,
            }
            .quote()
            .unwrap();
            let caller =
                size_of::<Arc<tensor::Program>>() + Builder::proof_row_scratch(&input).unwrap();
            let required = Builder::input_metadata_bytes(1).unwrap()
                + caller
                + 2 * 64 * size_of::<i64>()
                + quote.new_peak_bytes
                + width * size_of::<i64>();
            // This is the later, larger gate, not the initialization gate.
            assert!(
                required
                    > Builder::input_metadata_bytes(1).unwrap()
                        + caller
                        + ROTARY_INITIALIZATION_BYTES
            );
            b.proof_metadata_limit = required - usize::from(short);
            let called = Cell::new(false);
            let result = b.rope_with_fill(&input, width, 1, 512, 10000.0, |_, _, _, c, s| {
                called.set(true);
                assert_eq!((c.len(), s.len()), (width / 2, width / 2));
                // Stop at the original fill boundary, before the native factory.
                Err(Error::Allocation)
            });
            assert_eq!(called.get(), !short);
            assert!(matches!(result, Err(error) if error == if short {
                Error::Bounds
            } else {
                Error::Allocation
            }));
            assert_unpublished_and_retired(&mut b);
        }
    }
}

#[test]
fn rope_narrow_slices_never_refund_full_retained_array_backing() {
    let mut b = builder(2);
    let input = b.input(0).unwrap();
    let quote = tensor::Construction::RotaryHalf {
        rows: 1,
        width: 2,
        cosine_len: 1,
        sine_len: 1,
    }
    .quote()
    .unwrap();
    // An incorrect active-elements-only baseline would reach the fill. The
    // actual two full arrays are 1,024 bytes even though slices expose 16 bytes.
    b.proof_metadata_limit = Builder::input_metadata_bytes(1).unwrap()
        + size_of::<Arc<tensor::Program>>()
        + Builder::proof_row_scratch(&input).unwrap()
        + 2 * size_of::<i64>()
        + quote.peak_with_borrowed_public_bytes;
    let called = Cell::new(false);
    assert!(matches!(
        b.rope_with_fill(&input, 2, 1, 0, 10000.0, |_, _, _, _, _| {
            called.set(true);
            Err(Error::Allocation)
        }),
        Err(Error::Bounds)
    ));
    assert!(!called.get());
    assert_unpublished_and_retired(&mut b);
}

#[test]
fn rope_prevalidates_public_extents_positions_and_theta_and_retires() {
    for (width, sequence, position, theta, expected) in [
        (0, 1, 0, 10000.0, Error::Shape),
        (usize::MAX, 1, 0, 10000.0, Error::Shape),
        (3, 1, 0, 10000.0, Error::Shape),
        (2, 0, 0, 10000.0, Error::Shape),
        (2, 3, 0, 10000.0, Error::Shape),
        (2, 2, usize::MAX, 10000.0, Error::Bounds),
        (2, 1, usize::MAX, 10000.0, Error::Bounds),
        (2, 2, 512, 10000.0, Error::Bounds),
        (2, 1, 513, 10000.0, Error::Bounds),
        (2, 1, 0, f32::NAN, Error::Bounds),
        (2, 1, 0, f32::INFINITY, Error::Bounds),
        (2, 1, 0, 0.5, Error::Bounds),
    ] {
        let mut b = builder(4);
        let input = b.input(0).unwrap();
        let called = Cell::new(false);
        let capacities = (
            b.registers.capacity(),
            b.regions.capacity(),
            b.ops.capacity(),
        );
        let result = b.rope_with_fill(&input, width, sequence, position, theta, |_, _, _, _, _| {
            called.set(true);
            Err(Error::Allocation)
        });
        assert!(matches!(result, Err(error) if error == expected));
        assert!(!called.get());
        assert_unpublished_and_retired(&mut b);
        assert_eq!(
            (
                b.registers.capacity(),
                b.regions.capacity(),
                b.ops.capacity()
            ),
            capacities
        );
    }
}

#[test]
fn fixed_buffer_rope_matches_original_generated_coefficients_and_native_programs() {
    for (theta, position, width, sequence, rows) in [
        (1.0, 0, 2, 1, 1),
        (10000.0, 511, 2, 2, 4),
        (10000.0, 0, 64, 2, 4),
        (10000.0, 512, 128, 1, 1),
    ] {
        let mut legacy = builder(rows * width);
        let input = legacy.input(0).unwrap();
        let mut programs = Vec::with_capacity(rows);
        for row in 0..rows {
            let (cosine, sine) = tensor::Program::rotary_coefficients(
                theta,
                (position + row % sequence) as u32,
                width,
            )
            .unwrap();
            let view = input.slice(row * width, width).unwrap();
            let native = tensor::Program::rotary_half(
                1,
                width,
                legacy.bounds(&view).unwrap(),
                &cosine,
                &sine,
            )
            .unwrap();
            programs.push(legacy.intern(native).unwrap());
        }
        let expected = legacy.tensors(&input, width, programs, vec![], 0).unwrap();
        let expected = legacy.finish(&[expected]).unwrap();

        let mut actual = builder(rows * width);
        let input = actual.input(0).unwrap();
        let output = actual
            .rope(&input, width, sequence, position, theta)
            .unwrap();
        assert!(actual.proofs.is_empty());
        let actual = actual.finish(&[output]).unwrap();
        assert_eq!(actual.digest(), expected.digest());
        assert_eq!(actual.work_budget(), expected.work_budget());
        assert_eq!(
            actual.coordinator_advances(),
            expected.coordinator_advances()
        );
        assert_eq!(
            actual.resources().admitted_bytes,
            expected.resources().admitted_bytes
        );
    }
}

#[test]
fn rope_reuses_fixed_buffers_without_mutating_previously_interned_tables() {
    let mut b = builder(8);
    let input = b.input(0).unwrap();
    let cosine_address = Cell::new(0);
    let sine_address = Cell::new(0);
    let fills = Cell::new(0);
    let output = b
        .rope_with_fill(
            &input,
            2,
            2,
            511,
            10000.0,
            |theta, position, width, c, s| {
                if fills.get() == 0 {
                    cosine_address.set(c.as_ptr() as usize);
                    sine_address.set(s.as_ptr() as usize);
                } else {
                    assert_eq!(c.as_ptr() as usize, cosine_address.get());
                    assert_eq!(s.as_ptr() as usize, sine_address.get());
                }
                fills.set(fills.get() + 1);
                tensor::Program::rotary_coefficients_into(theta, position, width, c, s)
            },
        )
        .unwrap();
    assert_eq!(fills.get(), 4);
    assert_eq!(output.len(), 8);
    assert_eq!(b.templates.len(), 2);
    let Op::Tensor { programs, .. } = b.ops.last().unwrap() else {
        panic!("RoPE must retain original tensor family");
    };
    for (row, retained) in programs.iter().enumerate() {
        let (cosine, sine) =
            tensor::Program::rotary_coefficients(10000.0, 511 + (row % 2) as u32, 2).unwrap();
        let original =
            tensor::Program::rotary_half(1, 2, Interval::new(-ONE, ONE).unwrap(), &cosine, &sine)
                .unwrap();
        assert_eq!(retained.digest(), original.digest());
        assert_eq!(retained.work_budget(), original.work_budget());
        assert_eq!(retained.resources().unwrap(), original.resources().unwrap());
    }
}

#[test]
fn rope_later_fill_failure_retires_without_discarding_previous_template() {
    let mut b = builder(4);
    let input = b.input(0).unwrap();
    let fills = Cell::new(0);
    let result = b.rope_with_fill(
        &input,
        2,
        2,
        511,
        10000.0,
        |theta, position, width, c, s| {
            fills.set(fills.get() + 1);
            if fills.get() == 2 {
                return Err(Error::Allocation);
            }
            tensor::Program::rotary_coefficients_into(theta, position, width, c, s)
        },
    );
    assert!(matches!(result, Err(Error::Allocation)));
    assert_eq!(fills.get(), 2);
    assert!(b.proof_admission_retired);
    assert_eq!(b.templates.len(), 1);
    assert_eq!(b.template_metadata.len(), 1);
    assert!(b.ops.is_empty());
    assert!(b.proofs.is_empty());
    assert_eq!(b.work, WorkBudget::default());
    assert_eq!(b.advances, 1);
    let (cosine, sine) = tensor::Program::rotary_coefficients(10000.0, 511, 2).unwrap();
    let original =
        tensor::Program::rotary_half(1, 2, Interval::new(-ONE, ONE).unwrap(), &cosine, &sine)
            .unwrap();
    assert!(b.templates.contains_key(&original.digest()));
    assert!(matches!(b.input(0), Err(Error::Binding)));
}

// Independent witnesses for the existing two-map policy. These are test
// equations, not new production allowances or a whole-request memory census.
fn rope_map_entry_bytes() -> usize {
    let edges_and_header = 16 * size_of::<usize>() + 128;
    11 * size_of::<([u8; 32], Arc<tensor::Program>)>()
        + edges_and_header
        + 11 * size_of::<([u8; 32], usize)>()
        + edges_and_header
}

fn rope_template_only_census(b: &Builder) -> usize {
    assert_eq!((b.registers.len(), b.registers.capacity()), (1, 1));
    assert_eq!((b.regions.len(), b.regions.capacity()), (1, 1));
    assert_eq!(b.regions[0].len(), 1);
    assert_eq!((b.ops.len(), b.ops.capacity()), (0, 0));
    assert_eq!((b.proofs.len(), b.proofs.capacity()), (0, 0));
    assert_eq!(b.templates.len(), b.template_metadata.len());
    Builder::input_metadata_bytes(1).unwrap()
        + b.template_metadata
            .values()
            .map(|bytes| bytes + 64)
            .sum::<usize>()
        + b.templates.len() * rope_map_entry_bytes()
}

fn fill_rope_fixture(
    width: usize,
    position: u32,
    cosine: &mut [i64; 64],
    sine: &mut [i64; 64],
) -> tensor::Program {
    let half = width / 2;
    tensor::Program::rotary_coefficients_into(
        10000.0,
        position,
        width,
        &mut cosine[..half],
        &mut sine[..half],
    )
    .unwrap();
    tensor::Program::rotary_half(
        1,
        width,
        Interval::new(-ONE, ONE).unwrap(),
        &cosine[..half],
        &sine[..half],
    )
    .unwrap()
}

#[test]
fn rope_later_row_exact_one_short_recensuses_before_coefficient_fill() {
    let (first_digest, resources) = {
        let mut cosine = [0; 64];
        let mut sine = [0; 64];
        let program = fill_rope_fixture(2, 511, &mut cosine, &mut sine);
        (program.digest(), program.resources().unwrap())
    };
    for short in [false, true] {
        let mut b = builder(4);
        let input = b.input(0).unwrap();
        let initial = Builder::input_metadata_bytes(1).unwrap();
        let backing = 2 * size_of::<[i64; 64]>();
        let caller = 2 * size_of::<Arc<tensor::Program>>()
            + Builder::proof_row_scratch(&input).unwrap()
            + backing;
        let native = tensor::Construction::RotaryHalf {
            rows: 1,
            width: 2,
            cosine_len: 1,
            sine_len: 1,
        }
        .quote()
        .unwrap()
        .peak_with_borrowed_public_bytes;
        let after_first =
            initial + resources.immutable_template_bytes + 64 + rope_map_entry_bytes();
        let second_required = after_first + caller + native;
        let limit = second_required - usize::from(short);
        // Prove that row one passes both construction and NEW insertion at this
        // limit. The failing boundary must be row two's re-census, not row one.
        let first_insert = initial
            + caller
            + 2 * (resources.immutable_template_bytes + 64)
            + rope_map_entry_bytes();
        let initialization = initial + caller - backing + 2 * backing;
        assert!(initialization <= limit);
        assert!(initial + caller + native <= limit);
        assert!(first_insert <= limit);
        assert!(second_required <= MAX_PROGRAM_BYTES);
        b.proof_metadata_limit = limit;
        let fills = Cell::new(0);
        let result = b.rope_with_fill(
            &input,
            2,
            2,
            511,
            10000.0,
            |theta, position, width, c, s| {
                fills.set(fills.get() + 1);
                if fills.get() == 2 {
                    // At EXACT budget, prove gate reached row two's fill, without
                    // entering a later insertion/publication phase with another cap.
                    return Err(Error::Allocation);
                }
                tensor::Program::rotary_coefficients_into(theta, position, width, c, s)
            },
        );
        assert_eq!(fills.get(), if short { 1 } else { 2 });
        assert!(matches!(result, Err(error) if error == if short {
            Error::Bounds
        } else {
            Error::Allocation
        }));
        assert!(b.proof_admission_retired);
        assert_eq!(b.templates.len(), 1);
        assert_eq!(b.template_metadata.len(), 1);
        assert_eq!(b.templates[&first_digest].resources().unwrap(), resources);
        assert_eq!(rope_template_only_census(&b), after_first);
        assert!(b.ops.is_empty());
        assert_eq!(b.work, WorkBudget::default());
        assert_eq!(b.advances, 1);
        assert!(matches!(b.input(0), Err(Error::Binding)));
        assert!(matches!(b.finish(&[input]), Err(Error::Binding)));
    }
}

#[test]
fn rope_post_fill_insert_and_dedup_exact_one_short_keep_full_array_baseline() {
    for width in [2, 128] {
        for dedup in [false, true] {
            for short in [false, true] {
                let mut b = builder(2 * width);
                let input = b.input(0).unwrap();
                let mut programs = b
                    .new_template_list(&input, 2, ROTARY_INITIALIZATION_BYTES)
                    .unwrap();
                let mut cosine = [0; 64];
                let mut sine = [0; 64];
                let backing = size_of_val(&cosine) + size_of_val(&sine);
                assert_eq!(backing, ROTARY_COEFFICIENT_BYTES);
                // Real RoPE caller buffers: both full arrays, retained list
                // CAPACITY (two, even when empty) and input/view scratch.
                let caller = backing
                    + programs.capacity() * size_of::<Arc<tensor::Program>>()
                    + Builder::proof_row_scratch(&input).unwrap();
                assert_eq!(
                    caller,
                    Builder::template_call_temporary(&input, &programs, ROTARY_COEFFICIENT_BYTES)
                        .unwrap()
                );
                if dedup {
                    let first = fill_rope_fixture(width, 511, &mut cosine, &mut sine);
                    programs.push(b.intern_with_temporary(first, caller).unwrap());
                }
                // Same position for both rows is the ordinary sequence=1 dedup
                // path. Program already exists: this gate is explicitly AFTER
                // fill/construction, not a new preconstruction claim.
                let native = fill_rope_fixture(width, 511, &mut cosine, &mut sine);
                let digest = native.digest();
                let resources = native.resources().unwrap();
                let arrays_before = (cosine, sine);
                let old_count = b.templates.len();
                let existing = programs.first().cloned();
                let incoming = resources.immutable_template_bytes + 64;
                let insertion = if dedup {
                    0
                } else {
                    incoming + (old_count + 1) * rope_map_entry_bytes()
                };
                let required = rope_template_only_census(&b) + caller + incoming + insertion;
                assert!(required <= MAX_PROGRAM_BYTES);
                b.proof_metadata_limit = required - usize::from(short);
                let result = b.intern_with_temporary(native, caller);
                assert_eq!((cosine, sine), arrays_before);
                assert_eq!(programs.capacity(), 2);
                assert_eq!(programs.len(), usize::from(dedup));
                assert!(b.ops.is_empty());
                assert!(b.proofs.is_empty());
                assert_eq!(b.work, WorkBudget::default());
                assert_eq!(b.advances, 1);
                if short {
                    assert!(matches!(result, Err(Error::Bounds)));
                    assert_eq!(b.templates.len(), old_count);
                    assert_eq!(b.template_metadata.len(), old_count);
                    if let Some(existing) = existing {
                        assert!(Arc::ptr_eq(&existing, &b.templates[&digest]));
                    }
                    assert!(b.proof_admission_retired);
                    assert!(matches!(b.input(0), Err(Error::Binding)));
                    assert!(matches!(b.finish(&[input]), Err(Error::Binding)));
                } else {
                    let retained = result.unwrap();
                    assert_eq!(retained.digest(), digest);
                    assert_eq!(retained.resources().unwrap(), resources);
                    assert_eq!(b.templates.len(), 1);
                    assert!(Arc::ptr_eq(&retained, &b.templates[&digest]));
                    if let Some(existing) = existing {
                        assert!(Arc::ptr_eq(&existing, &retained));
                    }
                    assert!(!b.proof_admission_retired);
                }
            }
        }
    }
}

#[test]
fn rope_tensor_publication_exact_one_short_keeps_full_arrays_after_fill() {
    for width in [2, 128] {
        for short in [false, true] {
            let rows = 2;
            let mut b = builder(rows * width);
            let input = b.input(0).unwrap();
            let mut programs = b
                .new_template_list(&input, rows, ROTARY_INITIALIZATION_BYTES)
                .unwrap();
            let mut cosine = [0; 64];
            let mut sine = [0; 64];
            let backing = size_of_val(&cosine) + size_of_val(&sine);
            let caller = backing
                + programs.capacity() * size_of::<Arc<tensor::Program>>()
                + Builder::proof_row_scratch(&input).unwrap();
            for _ in 0..rows {
                let native = fill_rope_fixture(width, 511, &mut cosine, &mut sine);
                programs.push(b.intern_with_temporary(native, caller).unwrap());
            }
            assert_eq!(b.templates.len(), 1);
            let retained = programs[0].clone();
            let arrays_before = (cosine, sine);
            let initial = rope_template_only_census(&b);
            // tensors() pre-region phase: source Vec + boxed Arc destination,
            // region Vec + boxed destination, fixed arrays, borrowed input.
            let construction = (programs.capacity() + rows) * size_of::<Arc<tensor::Program>>()
                + 2 * rows * size_of::<Region>()
                + backing
                + input.metadata_bytes()
                + 256;
            // push phase: incoming Tensor's layout/Arc box, BOTH region owners,
            // full array baseline, view/output scratch, all three replacement
            // Vecs and their existing owner-defined outer-container allowances.
            let incoming_op =
                input.metadata_bytes() + 64 + rows * size_of::<Arc<tensor::Program>>() + 64;
            let replacements = 4 * size_of::<Register>()
                + 64
                + 4 * size_of::<Box<[Region]>>()
                + 64
                + 4 * size_of::<Op>()
                + 64;
            let publication = incoming_op
                + backing
                + 2 * (rows * size_of::<Region>() + 64)
                + Builder::proof_row_scratch(&input).unwrap()
                + replacements;
            let output = Layout::quote_contiguous(1, input.len())
                .unwrap()
                .construction_peak_bytes;
            assert!(publication > construction);
            assert!(publication > output);
            let required = initial + construction.max(publication).max(output);
            assert!(required <= MAX_PROGRAM_BYTES);
            b.proof_metadata_limit = required - usize::from(short);
            let capacities = (
                b.registers.capacity(),
                b.regions.capacity(),
                b.ops.capacity(),
            );
            let result = b.tensors(&input, width, programs, vec![], backing);
            assert_eq!((cosine, sine), arrays_before);
            assert_eq!(b.templates.len(), 1);
            assert!(Arc::ptr_eq(&retained, &b.templates[&retained.digest()]));
            assert!(b.proofs.is_empty());
            if short {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_eq!(
                    (
                        b.registers.capacity(),
                        b.regions.capacity(),
                        b.ops.capacity()
                    ),
                    capacities
                );
                assert_eq!(b.registers.len(), 1);
                assert!(b.ops.is_empty());
                assert_eq!(b.work, WorkBudget::default());
                assert_eq!(b.advances, 1);
                assert!(b.proof_admission_retired);
                assert!(matches!(b.input(0), Err(Error::Binding)));
                assert!(matches!(b.finish(&[input]), Err(Error::Binding)));
            } else {
                let output = result.unwrap();
                assert_eq!(output.len(), rows * width);
                assert_eq!(b.registers.len(), 2);
                assert_eq!(b.ops.len(), 1);
                assert_eq!(
                    b.work,
                    retained
                        .work_budget()
                        .checked_add(retained.work_budget())
                        .unwrap()
                );
                assert_eq!(b.advances, 1 + rows as u64 * retained.instructions() as u64);
                let Op::Tensor { programs, .. } = &b.ops[0] else {
                    panic!("expected Tensor");
                };
                assert_eq!(programs.len(), rows);
                assert!(programs.iter().all(|p| Arc::ptr_eq(p, &retained)));
                assert!(!b.proof_admission_retired);
            }
        }
    }
}

#[test]
fn rms_both_certificate_paths_keep_native_digest_work_tables_and_energy() {
    for supplied in [false, true] {
        let mut b = builder(3);
        let input = b.input(0).unwrap();
        let bounds = b.bounds(&input).unwrap();
        let gamma = [ONE; 3];
        let norm = 3_u128 << 36;
        let (expected, squared) = if supplied {
            b.record_proof(
                Proof {
                    key: Builder::proof_row_key(&input, 0, 3).unwrap(),
                    energy: Some(norm),
                    mass: None,
                },
                0,
            )
            .unwrap();
            tensor::Program::rms_norm_with_input_norm(1, 3, bounds, &gamma, norm).unwrap()
        } else {
            tensor::Program::rms_norm_certified(1, 3, bounds, &gamma).unwrap()
        };
        let (output, valid) = b.rms(&input, 3, &gamma, None, None).unwrap();
        assert!(valid.is_none());
        let retained = b.templates.get(&expected.digest()).unwrap();
        assert_eq!(retained.work_budget(), expected.work_budget());
        assert_eq!(retained.resources().unwrap(), expected.resources().unwrap());
        assert_eq!(b.bounds(&output).unwrap(), expected.output_bounds());
        assert_eq!(b.proof(&output).unwrap().energy, Some(squared));
        assert_eq!(b.work, expected.work_budget());
    }
}

#[test]
fn softmax_keeps_original_native_certificate_and_numeric_failure_burns() {
    let mut b = builder(3);
    let input = b.input(0).unwrap();
    let (expected, mass) =
        tensor::Program::softmax_certified(1, 3, b.bounds(&input).unwrap()).unwrap();
    let output = b.softmax(&input, 1, 2).unwrap();
    assert!(b.templates.contains_key(&expected.digest()));
    assert_eq!(b.work, expected.work_budget());
    assert_eq!(b.bounds(&output).unwrap(), expected.output_bounds());
    assert_eq!(b.proof(&output).unwrap().mass, Some(mass));

    let mut b = Builder::new(
        WeightMode::Public,
        &[Register {
            lanes: 3,
            bounds: Interval::new(i64::MIN, i64::MAX).unwrap(),
        }],
    )
    .unwrap();
    let input = b.input(0).unwrap();
    assert!(matches!(b.softmax(&input, 1, 2), Err(Error::Bounds)));
    assert_unpublished_and_retired(&mut b);
}
