use super::*;

fn corners(bits: u8) -> [u64; 7] {
    let ring = 1u64 << bits;
    [
        0,
        1,
        ring / 2 - 1,
        ring / 2,
        ring / 2 + 1,
        ring - 2,
        ring - 1,
    ]
}
fn choices(index: usize) -> Choices {
    let ring = corners(50);
    Choices {
        input_mask: ring[index % 7],
        predicate_mask: (index % 2) as u8,
        clip_mask: [0, 1, 32768, 65535][index % 4],
        high_mask: [0, 1, 128, 255][index % 4],
        lookup: [
            LookupChoices {
                r: ring[(index + 1) % 7],
                u: ring[(index + 2) % 7],
                q: (index % 2) as u8,
                sigma_positive: index % 4 < 2,
            },
            LookupChoices {
                r: ring[(index + 3) % 7],
                u: ring[(index + 4) % 7],
                q: ((index + 1) % 2) as u8,
                sigma_positive: index % 4 >= 2,
            },
        ],
        product: ProductChoices {
            mask: ring[(index + 5) % 7],
            left_coefficients: [
                ring[index % 7],
                ring[(index + 2) % 7],
                ring[(index + 4) % 7],
            ],
        },
        output_mask: ring[(index + 6) % 7],
    }
}

#[test]
fn all_selected_literals_match_independent_fixture_after_rust_floor_gate() {
    // Entire512 gate BEFORE hashes. This is an actual Rust proof test definition,
    // not a claim that it ran, a caller certificate or Decimal-only substitute.
    let mut work = crate::exp_floor::Workspace::default();
    let checked = crate::fixed_exp_tables::verify(&mut work).unwrap();
    let tables = fixture_tables().unwrap();
    use crate::fixed_exp_tables::Table;
    use sha2::{Digest, Sha256};
    for (index, table, expected) in [
        (
            0,
            Table::LowByte,
            "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b",
        ),
        (
            1,
            Table::HighByte,
            "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9",
        ),
    ] {
        assert_eq!(checked.values(table), &tables[index]);
        let mut hash = Sha256::new();
        for value in tables[index] {
            hash.update(value.to_le_bytes());
        }
        assert_eq!(format!("{:x}", hash.finalize()), expected);
    }
}

#[test]
fn full_clear_integer_schedule_matches_literal_equation_all65536_clip_codes() {
    let tables = fixture_tables().unwrap();
    for x in 0..=65535i64 {
        let trace = lane(x, &tables).unwrap();
        let direct = tables[0][x as usize % 256] * tables[1][x as usize / 256] / 4096;
        assert_eq!(
            (trace.c16, trace.j0, trace.j1),
            (x as u64, x as u64 % 256, x as u64 / 256)
        );
        assert_eq!(trace.e, direct);
        assert!(trace.p <= 1 << 24 && trace.e <= 4096);
    }
    for x in [65536, 65537, 131071, (1i64 << 38) - 1] {
        let trace = lane(x, &tables).unwrap();
        assert_eq!(
            (trace.h, trace.b, trace.c16, trace.j0, trace.j1, trace.e),
            (1, 0, 65535, 255, 255, 0)
        );
    }
    let zero = lane(0, &tables).unwrap();
    assert_eq!((zero.p, zero.e), (1 << 24, 4096));
    assert_eq!(
        (lane(255, &tables).unwrap().e, lane(256, &tables).unwrap().e),
        (3848, 3847)
    );
}

#[test]
fn unsigned_tr_borrow_ties_wrap_and_all_high_byte_indices() {
    let mut saw_borrow = false;
    let mut saw_no_borrow = false;
    for clear in [
        0, 1, 127, 128, 254, 255, 256, 257, 32767, 32768, 65534, 65535,
    ] {
        for mask in corners(16) {
            for output_mask in corners(8) {
                let shift = tr(Additive::new(clear, mask, 16).unwrap(), 8, output_mask).unwrap();
                assert_eq!(shift.output.clear().unwrap(), clear / 256);
                assert_eq!(shift.output.bits, 8);
                saw_borrow |= shift.low_borrow;
                saw_no_borrow |= !shift.low_borrow;
            }
        }
    }
    assert!(saw_borrow && saw_no_borrow);
    for high in 0..256 {
        let shift = tr(Additive::new(high * 256 + 255, 65535, 16).unwrap(), 8, 255).unwrap();
        assert_eq!(shift.output.clear().unwrap(), high);
    }
    // Signed ring8 reinterpretation would turn required index128 into -128.
    let high = tr(Additive::new(32768, 32768, 16).unwrap(), 8, 128).unwrap();
    assert_eq!(high.output.clear().unwrap(), 128);
    assert_eq!(signed(high.output.clear().unwrap(), 8).unwrap(), -128);
}

#[test]
fn signed_gap_ars_uses_floor_and_rejects_unproved_quarter_domain() {
    let quarter = 1i128 << 48;
    let mut borrow = [false; 2];
    for clear in [
        -quarter,
        -6144,
        -4097,
        -4096,
        -4095,
        -2048,
        -1,
        0,
        1,
        2048,
        4095,
        4096,
        4097,
        6144,
        1 << 24,
        quarter - 1,
    ] {
        for mask in corners(50) {
            for output_mask in corners(50) {
                let shift = gap_ars(
                    Additive::new(reduce(clear, 50).unwrap(), mask, 50).unwrap(),
                    12,
                    output_mask,
                )
                .unwrap();
                assert_eq!(
                    signed(shift.output.clear().unwrap(), 50).unwrap(),
                    clear.div_euclid(4096)
                );
                borrow[usize::from(shift.low_borrow)] = true;
            }
        }
    }
    assert_eq!(borrow, [true, true]);
    for invalid in [-quarter - 1, quarter] {
        assert_eq!(
            gap_ars(
                Additive::new(reduce(invalid, 50).unwrap(), 0, 50).unwrap(),
                12,
                0
            ),
            Err(Error::Domain)
        );
    }
    // Both ties are floors, not truncation toward zero or nearest-even.
    assert_eq!(
        signed(
            gap_ars(
                Additive::new(reduce(-2048, 50).unwrap(), 0, 50).unwrap(),
                12,
                0
            )
            .unwrap()
            .output
            .clear()
            .unwrap(),
            50
        )
        .unwrap(),
        -1
    );
    assert_eq!(
        gap_ars(Additive::new(6144, 0, 50).unwrap(), 12, 0)
            .unwrap()
            .output
            .clear()
            .unwrap(),
        1
    );
}

#[test]
fn xor_complement_and_select_preserve_distinct_mask_algebras() {
    for bit in [0, 1] {
        for boolean_mask in [0, 1] {
            let predicate = XorBit::new(bit, boolean_mask).unwrap();
            let complemented = predicate.not().unwrap();
            assert_eq!(
                (complemented.clear().unwrap(), complemented.mask),
                (bit ^ 1, boolean_mask)
            );
            for value in [0, 1, 32767, 32768, 65535] {
                for input_mask in corners(16) {
                    let encoded = Additive::new(value, input_mask, 16).unwrap();
                    for output_mask in corners(16) {
                        assert_eq!(
                            select(predicate, encoded, output_mask)
                                .unwrap()
                                .clear()
                                .unwrap(),
                            u64::from(bit) * value
                        );
                    }
                }
            }
        }
    }
    assert!(select(
        XorBit { word: 2, mask: 0 },
        Additive::new(1, 0, 16).unwrap(),
        0
    )
    .is_err());
    // Treating a masked Boolean word as arithmetic would select the wrong arm.
    assert_eq!(
        select(
            XorBit::new(0, 1).unwrap(),
            Additive::new(65535, 1, 16).unwrap(),
            0
        )
        .unwrap()
        .clear()
        .unwrap(),
        0
    );
}

#[test]
fn d19_both_signed_count_branches_full_counts_and_rho_dependency() {
    for sigma in [-1i64, 1] {
        for left_count in 0..=256i64 {
            let right_count = sigma - left_count;
            if !(-256..=0).contains(&right_count) {
                continue;
            }
            for q0 in [0, 1] {
                for q1 in [0, 1] {
                    let opened = count_correction(left_count, q0).unwrap()
                        ^ count_correction(right_count, q1).unwrap();
                    assert_eq!(opened ^ q0 ^ q1, u8::from(sigma == 1));
                }
            }
        }
    }
    for payload in [0, 1, 3847, 3848, 4095, 4096] {
        for r in corners(50) {
            for u in corners(50) {
                for q in [0, 1] {
                    for positive in [false, true] {
                        let result = lookup(
                            payload,
                            LookupChoices {
                                r,
                                u,
                                q,
                                sigma_positive: positive,
                            },
                        )
                        .unwrap();
                        assert_eq!(result.output.clear().unwrap(), payload);
                        assert_eq!(
                            result.selected.clear().unwrap(),
                            if positive { payload } else { 0 }
                        );
                        assert_eq!(
                            result.output.mask,
                            reduce(2 * i128::from(u) - i128::from(r), 50).unwrap()
                        );
                        assert_eq!(result.output.mask % 2, r % 2);
                        assert_eq!(result.output.word % 2, result.opened_r % 2);
                        assert_eq!(result.opened_correction ^ q, u8::from(positive));
                    }
                }
            }
        }
    }
}

#[test]
fn product_uses_original_lookup_masks_and_role1_only_public_term() {
    for payloads in [[0, 0], [4096, 4096], [4095, 3847], [1, 4096]] {
        for index in 0..28 {
            let settings = choices(index);
            let left = lookup(payloads[0], settings.lookup[0]).unwrap();
            let right = lookup(payloads[1], settings.lookup[1]).unwrap();
            let result = product(left.output, right.output, settings.product).unwrap();
            assert_eq!(result.output.clear().unwrap(), payloads[0] * payloads[1]);
            let [a, b, c] = result.coefficients[0];
            let role0 = (u128::from(a) * u128::from(right.output.word)
                + u128::from(left.output.word) * u128::from(b)
                + u128::from(c))
                % u128::from(N50);
            assert_eq!(u128::from(result.contributions[0]), role0);
            assert_eq!(
                (result.contributions[0] + result.contributions[1]) % N50,
                result.output.word
            );
            assert_eq!(result.output.mask, settings.product.mask);
        }
    }
}

#[test]
fn whole_masked_schedule_preserves_affine_projection_and_rounding_relations() {
    let tables = fixture_tables().unwrap();
    for input in BOUNDARIES {
        for index in 0..28 {
            let trace = masked_lane(input, &tables, choices(index)).unwrap();
            let wanted = trace.expected;
            assert_eq!(trace.input.clear().unwrap(), input as u64);
            assert_eq!(
                (trace.x40.clear().unwrap(), trace.affine40.clear().unwrap()),
                (wanted.x40, wanted.affine40)
            );
            assert_eq!(trace.affine40.mask, trace.x40.mask);
            assert_eq!(trace.x40.mask, trace.input.mask % (1u64 << 40));
            assert_eq!(
                (trace.h.clear().unwrap(), trace.b.clear().unwrap()),
                (wanted.h, wanted.b)
            );
            assert_eq!(trace.h.mask, trace.b.mask);
            assert_eq!(trace.w16.mask, trace.x40.mask % 65536);
            assert_eq!(
                (trace.w16.clear().unwrap(), trace.v16.clear().unwrap()),
                (wanted.w16, wanted.v16)
            );
            assert_eq!(trace.c16.mask, trace.v16.mask);
            assert_eq!(
                (
                    trace.c16.clear().unwrap(),
                    trace.j0.clear().unwrap(),
                    trace.j1.output.clear().unwrap()
                ),
                (wanted.c16, wanted.j0, wanted.j1)
            );
            assert_eq!(trace.j0.mask, trace.c16.mask % 256);
            assert_eq!(
                (
                    trace.lookup[0].output.clear().unwrap(),
                    trace.lookup[1].output.clear().unwrap()
                ),
                (wanted.u0, wanted.u1)
            );
            assert_eq!(trace.product.output.clear().unwrap(), wanted.p);
            assert_eq!(trace.final_shift.output.clear().unwrap(), wanted.e);
            assert!(wanted.p <= 1 << 24 && wanted.e < 4097);
        }
    }
}

#[test]
fn all_lane_widths_1_through256_and_complete_root_rejection() {
    let tables = fixture_tables().unwrap();
    for width in 1..=256 {
        let inputs: Vec<_> = (0..width)
            .map(|i| BOUNDARIES[i % BOUNDARIES.len()])
            .collect();
        let expected = lanes(&inputs, &tables).unwrap();
        assert_eq!(expected.len(), width);
        for (index, (input, output)) in inputs.into_iter().zip(expected).enumerate() {
            assert_eq!(output, lane(input, &tables).unwrap());
            assert_eq!(
                masked_lane(input, &tables, choices(index))
                    .unwrap()
                    .final_shift
                    .output
                    .clear()
                    .unwrap(),
                output.e
            );
        }
    }
    for invalid in [-1, 1i64 << 38] {
        let mut inputs = vec![0; 256];
        inputs[255] = invalid;
        assert_eq!(lanes(&inputs, &tables), Err(Error::Domain));
        assert_eq!(lane(invalid, &tables), Err(Error::Domain));
    }
    assert_eq!(lanes(&[], &tables), Err(Error::Shape));
    assert_eq!(lanes(&[0; 257], &tables), Err(Error::Shape));
}

#[test]
fn all18_clear_use_types_and_operand_ordinals_match_source_contract() {
    let trace = lane(32768, &fixture_tables().unwrap()).unwrap();
    let uses = use_values(&trace);
    assert_eq!((uses.len(), USES.len()), (18, 18));
    for index in [11, 12, 13] {
        assert_eq!(
            (uses[index].bits, uses[index].fractional, uses[index].kind),
            (8, 0, Kind::Unsigned)
        );
    }
    assert_eq!(uses[11].word, 128);
    assert_eq!(
        (uses[4].kind, uses[6].kind),
        (Kind::Predicate, Kind::Predicate)
    );
    assert_eq!(
        (uses[16].bits, uses[16].fractional, uses[17].fractional),
        (50, 24, 12)
    );
    assert_eq!(uses[9], uses[10]); // SAME parent C, both coordinate reads.
    assert_eq!(
        USES.iter()
            .enumerate()
            .filter_map(|(i, (_, operand))| (*operand == 1).then_some(i))
            .collect::<Vec<_>>(),
        [7, 15]
    );
    // Moving floors into the product operands erases valid Q12 payloads.
    let one = lane(1, &fixture_tables().unwrap()).unwrap();
    assert_eq!(one.e, 4095);
    assert_ne!(one.e, (one.u0 / 4096) * (one.u1 / 4096) * 4096);
}

#[test]
fn exact11_stage_shapes_22_frames_and_directed_bytes_at_every_lane_width() {
    assert_eq!(STAGES.len(), 11);
    assert_eq!(
        STAGES.map(|stage| stage.phase),
        [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    );
    assert_eq!(
        STAGES
            .iter()
            .filter(|stage| stage.magic == *b"PLLMLU01")
            .count(),
        2
    );
    for width in 1..=256 {
        let total: usize = (1..=11)
            .map(|stage| 2 * frame_bytes(stage, width).unwrap())
            .sum();
        assert_eq!(total, 1688 + 208 * width);
        for stage in 1..=11 {
            assert_eq!(
                payload_bits(stage).unwrap().len(),
                STAGES[stage - 1].words_per_lane
            );
        }
    }
    assert_eq!(frame_bytes(1, 1), Ok(84));
    assert_eq!(frame_bytes(5, 256), Ok(4176));
    for invalid in [0, 12, usize::MAX] {
        assert_eq!(frame_bytes(invalid, 1), Err(Error::Domain));
    }
    for invalid in [0, 257, usize::MAX] {
        assert_eq!(frame_bytes(1, invalid), Err(Error::Shape));
    }
}

#[test]
fn noncanonical_masks_coordinates_bits_and_shifts_never_silently_normalize() {
    assert_eq!(Additive::new(1u64 << 50, 0, 50), Err(Error::Domain));
    assert_eq!(Additive::new(0, 1u64 << 50, 50), Err(Error::Domain));
    assert_eq!(Additive::new(0, 0, 0), Err(Error::Domain));
    assert_eq!(XorBit::new(0, 2), Err(Error::Domain));
    let c = Additive::new(65535, 1, 16).unwrap();
    for shift in [0, 16, 255] {
        assert_eq!(tr(c, shift, 0), Err(Error::Domain));
    }
    assert_eq!(tr(c, 8, 256), Err(Error::Domain));
    assert_eq!(c.project(50), Err(Error::Domain));
    let p = Additive::new(1 << 24, 0, 50).unwrap();
    for shift in [0, 49, 50, 255] {
        assert_eq!(gap_ars(p, shift, 0), Err(Error::Domain));
    }
    let mut invalid = choices(0);
    invalid.high_mask = 256;
    assert_eq!(
        masked_lane(0, &fixture_tables().unwrap(), invalid),
        Err(Error::Domain)
    );
}
