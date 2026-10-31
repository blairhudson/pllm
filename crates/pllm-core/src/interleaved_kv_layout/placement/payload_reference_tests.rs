use super::*;
use super::payload_reference::I64ReferenceWork;

fn recipe(g: Geometry, token: usize) -> AppendPlacement {
    let t = g.tokens_per_key_tile();
    let group = g.tokens_per_value_group();
    AppendPlacement::new(g, AppendDescriptor {
        token,
        key_tile: token / t,
        key_lane: token % t,
        new_key_ciphertexts: usize::from(token % t == 0),
        value_group: token / group,
        new_value_ciphertexts: if token % group == 0 { g.head_dim() } else { 0 },
    }).unwrap()
}

#[test]
fn exact_lengths_and_parent_presence_refuse_without_writes() {
    let g = Geometry::new(8, 1, 4, 10).unwrap();
    let source = [3i64; 9];
    let parent = [5i64; 9];
    for token in [0, 1] {
        for op in recipe(g, token) {
            for length in [0, 7, 9] {
                for field in 0..3 {
                    let mut output = [701i64; 9];
                    let src = &source[..if field == 0 { length } else { 8 }];
                    let old = &parent[..if field == 1 { length } else { 8 }];
                    // Supply parent even for boundary recipes to test malformed
                    // lengths independently of presence precedence.
                    let result = op.apply_i64_reference(src, Some(old),
                        &mut output[..if field == 2 { length } else { 8 }]);
                    assert_eq!(result, Err(LayoutError::BufferLength));
                    assert_eq!(output, [701; 9]);
                    assert_eq!(source, [3; 9]);
                    assert_eq!(parent, [5; 9]);
                }
            }
            let mut output = [709; 8];
            let unexpected = if op.retain_parent { None } else { Some(&parent[..8]) };
            assert_eq!(op.apply_i64_reference(&source[..8], unexpected, &mut output),
                Err(LayoutError::Geometry));
            assert_eq!(output, [709; 8]);
        }
    }
}

#[test]
fn complete_mask_geometry_including_capacity_must_match() {
    let g = Geometry::new(8, 1, 4, 10).unwrap();
    let good = recipe(g, 0).next().unwrap();
    for other in [
        Geometry::new(16, 1, 4, 10).unwrap(),
        Geometry::new(4, 1, 4, 10).unwrap(),
        Geometry::new(8, 2, 2, 10).unwrap(),
        Geometry::new(8, 1, 2, 10).unwrap(),
        Geometry::new(8, 1, 4, 11).unwrap(),
    ] {
        for source_side in [false, true] {
            let mut op = good;
            if source_side { op.source_mask.geometry = other; }
            else { op.destination_mask.geometry = other; }
            let mut output = [719; 8];
            assert_eq!(op.apply_i64_reference(&[1; 8], None, &mut output),
                Err(LayoutError::Geometry));
            assert_eq!(op.i64_reference_work(), Err(LayoutError::Geometry));
            assert_eq!(output, [719; 8]);
        }
    }
}

#[test]
fn hostile_masks_and_public_destination_fields_refuse_atomically() {
    let g = Geometry::new(8, 1, 4, 10).unwrap();
    let key = recipe(g, 0).next().unwrap();
    let value = recipe(g, 0).nth(1).unwrap();
    for (base, field, error) in [
        (key, 0, LayoutError::Geometry), (key, 1, LayoutError::Geometry),
        (value, 2, LayoutError::Geometry), (value, 3, LayoutError::Geometry),
        (value, 4, LayoutError::Geometry), (key, 5, LayoutError::Geometry),
        (key, 6, LayoutError::Index), (value, 7, LayoutError::Index),
        (value, 8, LayoutError::Index), (value, 9, LayoutError::Index),
        (key, 10, LayoutError::Geometry),
    ] {
        let mut op = base;
        match field {
            0 => op.source_mask.lane = 2,
            1 => op.destination_mask.lane = usize::MAX,
            2 => op.source_mask.feature = Some(4),
            3 => op.destination_mask.feature = Some(usize::MAX),
            4 => op.destination_mask.feature = Some(1),
            5 => {
                op.source_mask.feature = Some(0);
                op.destination_mask.feature = Some(0);
            }
            6 => op.destination = PlacementDestination::Key { tile: usize::MAX },
            7 => op.destination = PlacementDestination::Value { group: usize::MAX, ciphertext: 0 },
            8 => op.destination = PlacementDestination::Value { group: 0, ciphertext: usize::MAX },
            9 => op.destination = PlacementDestination::Value { group: 0, ciphertext: 4 },
            _ => op.destination = PlacementDestination::Value { group: 0, ciphertext: 0 },
        }
        let mut output = [727; 8];
        assert_eq!(op.apply_i64_reference(&[1; 8], None, &mut output), Err(error));
        assert_eq!(output, [727; 8]);
        assert_eq!(op.i64_reference_work(), Err(error));
    }
}

#[test]
fn every_signed_rotation_is_arithmetic_not_append_authority() {
    let g = Geometry::new(8, 1, 1, 10).unwrap();
    let base = recipe(g, 0).next().unwrap();
    for (rotation, destination_lane) in [
        (0, 0), (-1, 1), (1, 7), (-9, 1), (9, 7),
        (i32::MIN, 0), (i32::MAX, 1),
    ] {
        let mut op = base;
        op.rotation = rotation;
        op.destination_mask.lane = destination_lane;
        let source = [i64::MIN, 991, 992, 993, 994, 995, 996, i64::MAX];
        let mut output = [733; 8];
        op.apply_i64_reference(&source, None, &mut output).unwrap();
        let mut expected = [0; 8];
        expected[destination_lane] = i64::MIN;
        assert_eq!(output, expected);
        assert_eq!(source, [i64::MIN, 991, 992, 993, 994, 995, 996, i64::MAX]);
    }
}

#[test]
fn final_slot_overflow_leaves_entire_output_and_originals_unchanged() {
    let g = Geometry::new(8, 1, 4, 10).unwrap();
    let mut cases = 0;
    for op in recipe(g, 1) {
        // K and the V contribution containing feature three reach final slot.
        if !op.destination_mask.selected(7).unwrap() { continue; }
        for (extreme, increment) in [(i64::MAX, 1), (i64::MIN, -1)] {
            let mut source = [2; 8];
            source[6] = extreme;
            let mut parent = [3; 8];
            parent[7] = increment;
            let original_source = source;
            let original_parent = parent;
            let mut output = [739; 11];
            assert_eq!(op.apply_i64_reference(&source, Some(&parent), &mut output[..8]),
                Err(LayoutError::Overflow));
            assert_eq!(output, [739; 11], "even preceding slots and spare tail stay untouched");
            assert_eq!(source, original_source);
            assert_eq!(parent, original_parent);
            cases += 1;
        }
    }
    assert_eq!(cases, 4);
}

#[test]
fn poison_partial_and_boundary_key_plus_every_value_preserve_originals_and_tail() {
    let g = Geometry::new(8, 1, 4, 10).unwrap();
    let source = [11, i64::MIN, -13, i64::MAX, 17, i64::MIN, -19, i64::MAX];
    let parent = [101, 103, 107, 109, 113, 127, 131, 137];
    let mut cases = 0;
    for token in [0, 1, 2, 7, 8, 9] {
        for (index, op) in recipe(g, token).enumerate() {
            let mut output = [743; 11];
            op.apply_i64_reference(&source,
                if op.retain_parent { Some(&parent) } else { None }, &mut output[..8]).unwrap();
            // Independent Figure 6 forward equation, not mask/address helpers.
            let mut expected = if op.retain_parent { parent } else { [0; 8] };
            for feature in 0..4 {
                let selected = index == 0 || (feature + index - 1) % 4 == (token % 8) / 2;
                if selected { expected[feature * 2 + token % 2] += source[feature * 2]; }
            }
            assert_eq!(&output[..8], &expected);
            assert_eq!(&output[8..], &[743; 3]);
            assert_eq!(source, [11, i64::MIN, -13, i64::MAX, 17, i64::MIN, -19, i64::MAX]);
            assert_eq!(parent, [101, 103, 107, 109, 113, 127, 131, 137]);
            cases += 1;
        }
    }
    assert_eq!(cases, 30);
}

#[test]
fn all_i64_extremes_are_valid_without_parent_and_zero_delta_preserves_parent() {
    let g = Geometry::new(8, 1, 4, 10).unwrap();
    for op in recipe(g, 0) {
        let mut output = [751; 8];
        op.apply_i64_reference(&[i64::MAX; 8], None, &mut output).unwrap();
        assert!(output.iter().all(|&x| x == 0 || x == i64::MAX));
        op.apply_i64_reference(&[i64::MIN; 8], None, &mut output).unwrap();
        assert!(output.iter().all(|&x| x == 0 || x == i64::MIN));
        let retained = PlacementContribution { retain_parent: true, ..op };
        let parent = [i64::MIN, i64::MAX, i64::MIN, i64::MAX,
            i64::MIN, i64::MAX, i64::MIN, i64::MAX];
        retained.apply_i64_reference(&[0; 8], Some(&parent), &mut output).unwrap();
        assert_eq!(output, parent);
    }
}

#[test]
fn checked_work_names_both_scans_not_hypothetical_one_pass_or_public_mask_calls() {
    for slots in [8, 32768] {
        let g = Geometry::new(slots, 1, 1, 2).unwrap();
        for token in [0, 1] {
            let op = recipe(g, token).next().unwrap();
            CENSUS.with(|c| c.set(PlacementWork::default()));
            let work = op.i64_reference_work().unwrap();
            assert_eq!(work, I64ReferenceWork {
                validation_slots: slots, write_slots: slots,
                mask_evaluations: 4 * slots,
                parent_additions: if token == 1 { 2 * slots } else { 0 },
            });
            assert_eq!(work.checked_repeated(3).unwrap(), I64ReferenceWork {
                validation_slots: 3 * slots, write_slots: 3 * slots,
                mask_evaluations: 12 * slots,
                parent_additions: if token == 1 { 6 * slots } else { 0 },
            });
            assert_eq!(work.checked_repeated(usize::MAX), Err(LayoutError::Overflow));
            assert_eq!(work.checked_repeated(0).unwrap(), I64ReferenceWork {
                validation_slots: 0, write_slots: 0, mask_evaluations: 0, parent_additions: 0,
            });
            assert_eq!(CENSUS.with(|c| c.get()), PlacementWork::default(),
                "quote does not traverse selected()/bits() or execute payload");
        }
    }
}
