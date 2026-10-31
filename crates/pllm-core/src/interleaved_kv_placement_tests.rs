//! Independent Figure 5/6 oracle. No production address/mask helpers used to
//! derive expected destinations. Production masks are used ONLY by interpreter.
use super::*;

fn descriptor(g: Geometry, n: usize) -> AppendDescriptor {
    let t = g.slots() / (g.heads() * g.head_dim());
    let group = g.slots() / g.heads();
    AppendDescriptor {
        token: n,
        key_tile: n / t,
        key_lane: n % t,
        new_key_ciphertexts: usize::from(n % t == 0),
        value_group: n / group,
        new_value_ciphertexts: if n % group == 0 { g.head_dim() } else { 0 },
    }
}

fn interpret(op: PlacementContribution, input: &[i64]) -> Vec<i64> {
    // These callers ask for delta only, independently of append retention.
    let delta = PlacementContribution { retain_parent: false, ..op };
    let mut output = vec![0; input.len()];
    delta.apply_i64_reference(input, None, &mut output).unwrap();
    output
}

// Forward physical-slot interpretation from Figure 6: at local x in ct c,
// token=(x+c*t)%G, feature=floor(x/t). Heads interleave in t-slot blocks.
// This scans physical destinations, not inversion via ValueLayout::address.
fn paper_delta(g: Geometry, n: usize, dest: PlacementDestination, input: &[i64]) -> Vec<i64> {
    let t = g.slots() / (g.heads() * g.head_dim());
    let group = g.slots() / g.heads();
    (0..g.slots()).map(|slot| {
        let block = slot / t;
        let h = block % g.heads();
        let f = block / g.heads();
        let lane = slot % t;
        let token = match dest {
            PlacementDestination::Key { tile } => tile * t + lane,
            PlacementDestination::Value { group: b, ciphertext } => {
                let c = ciphertext % g.head_dim();
                b * group + (f * t + lane + c * t) % group
            }
        };
        if token == n { input[(f * g.heads() + h) * t] } else { 0 }
    }).collect()
}

#[test]
fn every_source_basis_and_poison_against_forward_paper_oracle() {
    for (slots, heads, p) in [(8, 1, 4), (16, 2, 4), (32, 2, 2), (8, 2, 4), (8, 1, 1)] {
        let group = slots / heads;
        let g = Geometry::new(slots, heads, p, group * 2 + 2).unwrap();
        let t = slots / (heads * p);
        for n in [0, 1, t - 1, t, t + 1, group - 1, group, group + 1] {
            for basis in 0..slots {
                for sign in [-1, 1] {
                    let mut input = vec![0; slots];
                    input[basis] = sign;
                    for op in AppendPlacement::new(g, descriptor(g, n)).unwrap() {
                        assert_eq!(interpret(op, &input), paper_delta(g, n, op.destination, &input));
                    }
                }
            }
            let mut poison: Vec<_> = (0..slots).map(|j| -(j as i64) - 101).collect();
            for h in 0..heads {
                for f in 0..p {
                    let slot = (f * heads + h) * t;
                    assert_eq!(g.output_slot(h, f).unwrap(), slot);
                    poison[slot] = 1 + (h * p + f) as i64;
                }
            }
            for op in AppendPlacement::new(g, descriptor(g, n)).unwrap() {
                assert_eq!(interpret(op, &poison), paper_delta(g, n, op.destination, &poison));
                assert_eq!(op.rotation, -((n % t) as i32));
                assert_eq!(op.source_mask.selected(slots), Err(LayoutError::Index));
                assert_eq!(op.destination_mask.bits().len(), slots);
            }
        }
    }
}

#[test]
fn figure_five_and_six_token_five_order_sign_and_mask_order() {
    let g = Geometry::new(8, 1, 4, 10).unwrap();
    let input = [50, -999, 51, -999, 52, -999, 53, -999];
    let mut plan = AppendPlacement::new(g, descriptor(g, 5)).unwrap();
    assert_eq!(plan.len(), 5);
    let k = plan.next().unwrap();
    assert_eq!(k.destination, PlacementDestination::Key { tile: 2 });
    assert!(k.retain_parent);
    assert_eq!(k.rotation, -1);
    assert_eq!(interpret(k, &input), [0, 50, 0, 51, 0, 52, 0, 53]);
    let mut wrong_sign = k;
    wrong_sign.rotation = 1;
    assert_ne!(interpret(wrong_sign, &input), interpret(k, &input));
    // WRONG: destination_mask[j] * source_mask[j] * Rot(input,-1)[j].
    // Exactly ONE rotation, no call to interpret (which would rotate again).
    let wrong_order: Vec<_> = (0..8).map(|j| {
        input[(j + 7) % 8]
            * i64::from(k.source_mask.selected(j).unwrap())
            * i64::from(k.destination_mask.selected(j).unwrap())
    }).collect();
    // Hand derivation: source mask keeps even slots, destination keeps odd;
    // wrong order intersects disjoint masks, discarding every valid coordinate.
    assert_eq!(wrong_order, [0; 8]);
    assert_eq!(paper_delta(g, 5, k.destination, &input), [0, 50, 0, 51, 0, 52, 0, 53]);
    assert_ne!(wrong_order, interpret(k, &input));
    // Figure 6(d): ct2 receives v50; ct1 v51; ct0 v52; ct3 v53.
    let expected = [
        [0, 0, 0, 0, 0, 52, 0, 0],
        [0, 0, 0, 51, 0, 0, 0, 0],
        [0, 50, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 53],
    ];
    for (c, want) in expected.iter().enumerate() {
        let op = plan.next().unwrap();
        assert_eq!(op.destination, PlacementDestination::Value { group: 0, ciphertext: c });
        assert!(op.retain_parent);
        assert_eq!(&interpret(op, &input), want);
    }
    assert_eq!(plan.len(), 0);
    assert_eq!(plan.next(), None);
    assert_eq!(plan.next(), None);
}

#[test]
fn complete_value_partition_and_ideal_norm_contraction() {
    let g = Geometry::new(32, 2, 4, 34).unwrap();
    for n in 0..g.capacity() {
        let input: Vec<_> = (0..32).map(|j| j as i64 - 19).collect();
        let norm = input.iter().map(|x| x.abs()).max().unwrap();
        let mut seen = [0u8; 32];
        let mut count = 0;
        for op in AppendPlacement::new(g, descriptor(g, n)).unwrap() {
            count += 1;
            assert!(interpret(op, &input).iter().all(|x| x.abs() <= norm));
            if matches!(op.destination, PlacementDestination::Value { .. }) {
                for (i, bit) in op.source_mask.bits().enumerate() { seen[i] += bit; }
            }
        }
        assert_eq!(count, 5, "incomplete V inventory is not a complete append");
        for (i, copies) in seen.into_iter().enumerate() {
            assert_eq!(copies, u8::from(i % 4 == 0));
        }
    }
}

#[test]
fn repeated_fresh_outputs_match_unchanged_retained_cache_and_dense_reads() {
    for (slots, heads, p) in [(8, 1, 4), (32, 2, 4), (16, 4, 1)] {
        let capacity = 2 * slots / heads + 2;
        let g = Geometry::new(slots, heads, p, capacity).unwrap();
        let t = slots / (heads * p);
        let group = slots / heads;
        let mut ref_k = vec![0; g.key_layout().storage_words().unwrap()];
        let mut ref_v = vec![0; g.value_layout().storage_words().unwrap()];
        let mut cache = RetainedCache::new(g, &mut ref_k, &mut ref_v).unwrap();
        let mut keys = vec![0; g.key_layout().storage_words().unwrap()];
        let mut values = vec![0; g.value_layout().storage_words().unwrap()];
        let mut dense_k = Vec::new();
        let mut dense_v = Vec::new();
        for n in 0..capacity {
            let key: Vec<_> = (0..heads * p).map(|i| (n * 3 + i) as i64 - 17).collect();
            let val: Vec<_> = (0..heads * p).map(|i| 19 - (n + i * 2) as i64).collect();
            let mut input_k = vec![-777; slots];
            let mut input_v = vec![888; slots];
            for h in 0..heads {
                for f in 0..p {
                    input_k[(f * heads + h) * t] = key[h * p + f];
                    input_v[(f * heads + h) * t] = val[h * p + f];
                }
            }
            let a = cache.append_descriptor().unwrap();
            assert_eq!(a, descriptor(g, n));
            for op in AppendPlacement::new(g, a).unwrap() {
                let (storage, ct, input, boundary) = match op.destination {
                    PlacementDestination::Key { tile } => (&mut keys, tile, &input_k, n % t == 0),
                    PlacementDestination::Value { group: b, ciphertext } => {
                        assert_eq!(b, n / group);
                        (&mut values, ciphertext, &input_v, n % group == 0)
                    }
                };
                assert_eq!(op.retain_parent, !boundary);
                let old = storage[ct * slots..(ct + 1) * slots].to_vec();
                let contribution = paper_delta(g, n, op.destination, input);
                // Fresh output, immutable old parent; overlap would corrupt old tokens.
                for (&x, &delta) in old.iter().zip(&contribution) {
                    if delta != 0 { assert_eq!(x, 0); }
                    if boundary { assert_eq!(x, 0); }
                }
                let mut fresh = vec![0; slots];
                op.apply_i64_reference(input, if boundary { None } else { Some(&old) }, &mut fresh).unwrap();
                storage[ct * slots..(ct + 1) * slots].copy_from_slice(&fresh);
            }
            cache.append(&key, &val).unwrap();
            dense_k.push(key);
            dense_v.push(val);
            let view = cache.derived_view();
            assert_eq!(keys, view.key_slots());
            assert_eq!(values, view.value_slots());
            let query: Vec<_> = (0..heads * p).map(|i| i as i64 - 3).collect();
            let mut scores = vec![0; g.score_words()];
            view.query_key(&query, &mut scores).unwrap();
            let mut weights = vec![0; g.score_words()];
            let mut want_scores = vec![0; g.score_words()];
            let mut want_out = vec![0; slots];
            for token in 0..=n {
                for h in 0..heads {
                    let local = token % group;
                    let score = token / group * slots + (local / t * heads + h) * t + local % t;
                    let weight = (token % 5) as i64 - 2 + h as i64;
                    weights[score] = weight;
                    for f in 0..p {
                        want_scores[score] += query[h * p + f] * dense_k[token][h * p + f];
                        want_out[(f * heads + h) * t] += weight * dense_v[token][h * p + f];
                    }
                }
            }
            // Poison unused score slots; attention_value must ignore them.
            for local in n + 1..capacity.div_ceil(group) * group {
                for h in 0..heads {
                    let x = local % group;
                    weights[local / group * slots + (x / t * heads + h) * t + x % t] = -999;
                }
            }
            assert_eq!(scores, want_scores);
            let mut output = vec![0; slots];
            view.attention_value(&weights, &mut output).unwrap();
            assert_eq!(output, want_out);
        }
        assert_eq!(cache.append_descriptor(), Err(LayoutError::Capacity));
        assert_eq!(cache.append(&vec![0; heads * p], &vec![0; heads * p]), Err(LayoutError::Capacity));
        assert!(matches!(AppendPlacement::new(g, descriptor(g, capacity)), Err(LayoutError::Capacity)));
        assert_eq!(cache.descriptor().active_tokens, capacity);
        assert_eq!(cache.derived_view().key_slots(), keys);
        assert_eq!(cache.derived_view().value_slots(), values);
    }
}

#[test]
fn rejects_each_forged_descriptor_field_before_iteration() {
    let g = Geometry::new(16, 2, 4, 17).unwrap();
    let good = descriptor(g, 3);
    for field in 0..6 {
        let mut bad = good;
        match field {
            0 => bad.key_tile += 1,
            1 => bad.key_lane += 1,
            2 => bad.new_key_ciphertexts += 1,
            3 => bad.value_group += 1,
            4 => bad.new_value_ciphertexts += 1,
            _ => bad.token += 1,
        }
        assert!(matches!(AppendPlacement::new(g, bad), Err(LayoutError::Geometry)));
    }
    assert!(AppendPlacement::new(g, descriptor(g, 16)).is_ok());
    assert!(matches!(AppendPlacement::new(g, descriptor(g, 17)), Err(LayoutError::Capacity)));
    assert!(matches!(AppendPlacement::new(g, descriptor(g, usize::MAX)), Err(LayoutError::Capacity)));
    let mut doubly_invalid = descriptor(g, 17);
    doubly_invalid.key_tile += 1;
    assert!(matches!(AppendPlacement::new(g, doubly_invalid), Err(LayoutError::Capacity)));
}

#[test]
fn work_quote_matches_real_api_invocation_census_and_repeated_scans() {
    use super::placement::CENSUS;
    for (slots, heads, p) in [(8, 1, 4), (16, 2, 4), (8, 2, 4), (8, 1, 1)] {
        let group = slots / heads;
        let t = slots / (heads * p);
        let g = Geometry::new(slots, heads, p, group * 2 + 2).unwrap();
        for n in [0, 1, t - 1, t, t + 1, group - 1, group, group + 1] {
            let plan = AppendPlacement::new(g, descriptor(g, n)).unwrap();
            let sample = plan.clone().next().unwrap();
            CENSUS.with(|c| c.set(PlacementWork::default()));
            // One complete script: construct, traverse through None, scan BOTH
            // masks through None for EVERY contribution, then call next twice
            // more after each exhaustion and query one invalid mask index.
            let mask_quote = sample.source_mask.scan_work().unwrap()
                .checked_add(PlacementWork { mask_next_calls: 2, mask_predicate_calls: 1,
                    ..PlacementWork::default() }).unwrap();
            let quoted = AppendPlacement::construction_work()
                .checked_add(plan.traversal_work().unwrap()).unwrap()
                .checked_add(AppendPlacement::next_work().checked_repeated(2).unwrap()).unwrap()
                .checked_add(mask_quote.checked_repeated(2 * (p + 1)).unwrap()).unwrap();
            let literal = PlacementWork {
                constructor_checks: 6,
                recipe_next_calls: p + 4,
                mask_next_calls: 2 * (p + 1) * (slots + 3),
                mask_predicate_calls: 2 * (p + 1) * (slots + 1),
            };
            assert_eq!(quoted, literal);
            assert_eq!(CENSUS.with(|c| c.get()), PlacementWork::default(), "quotes execute no API traversal");
            let mut cumulative = PlacementWork::default();
            for repetition in 1..=3 {
                let mut actual = AppendPlacement::new(g, descriptor(g, n)).unwrap();
                while let Some(op) = actual.next() {
                    for mask in [op.source_mask, op.destination_mask] {
                        let mut bits = mask.bits();
                        let mut emitted = 0;
                        while let Some(bit) = bits.next() {
                            assert!(bit <= 1);
                            emitted += 1;
                        }
                        assert_eq!(emitted, slots, "zero bits count too");
                        assert_eq!(bits.next(), None);
                        assert_eq!(bits.next(), None);
                        assert_eq!(mask.selected(slots), Err(LayoutError::Index));
                    }
                }
                assert_eq!(actual.next(), None);
                assert_eq!(actual.next(), None);
                cumulative = cumulative.checked_add(quoted).unwrap();
                assert_eq!(cumulative, literal.checked_repeated(repetition).unwrap());
                assert_eq!(CENSUS.with(|c| c.get()), cumulative);
            }
            // Partial traversal quote counts only remaining next calls, still
            // includes terminal None. Exhausted traversal is one paid call.
            let mut partial = plan;
            partial.next().unwrap();
            assert_eq!(partial.traversal_work().unwrap().recipe_next_calls, p + 1);
            while partial.next().is_some() {}
            assert_eq!(partial.traversal_work().unwrap().recipe_next_calls, 1);
            assert_eq!(sample.source_mask.query_work().mask_predicate_calls, 1);
        }
    }
}

#[test]
fn maximum_legal_geometry_quotes_are_arithmetic_only_with_checked_overflow() {
    // No mask scans, payload allocation or rotations: metadata arithmetic only.
    // Cover maximum S/capacity, maximum H with t=1, and large legal p with t=1.
    for (slots, heads, p, capacity) in [
        (32768, 1, 1, 32768), (32768, 32768, 1, 1), (2048, 1, 2048, 1),
    ] {
        let g = Geometry::new(slots, heads, p, capacity).unwrap();
        let plan = AppendPlacement::new(g, descriptor(g, 0)).unwrap();
        assert_eq!(plan.traversal_work().unwrap().recipe_next_calls, p + 2);
        // Obtain mask metadata via bounded single next(), never scan large S.
        let op = plan.clone().next().unwrap();
        let scan = op.source_mask.scan_work().unwrap();
        assert_eq!(scan.mask_next_calls, slots + 1);
        assert_eq!(scan.mask_predicate_calls, slots);
        let complete = AppendPlacement::construction_work()
            .checked_add(plan.traversal_work().unwrap()).unwrap()
            .checked_add(scan.checked_repeated(2 * (p + 1)).unwrap()).unwrap();
        assert_eq!(complete, PlacementWork {
            constructor_checks: 6, recipe_next_calls: p + 2,
            mask_next_calls: 2 * (p + 1) * (slots + 1),
            mask_predicate_calls: 2 * (p + 1) * slots,
        });
        assert_eq!(complete.checked_repeated(usize::MAX), Err(LayoutError::Overflow));
        let payload = op.ideal_payload_work();
        assert_eq!(payload, PlacementPayloadWork {
            source_mask_products: slots, rotation_slots: slots,
            destination_mask_products: slots, parent_additions: 0,
            inactive_initializations: 0,
        });
        assert_eq!(payload.checked_repeated(usize::MAX), Err(LayoutError::Overflow));
    }
    for field in 0..4 {
        let mut unit = PlacementWork::default();
        let count = match field {
            0 => &mut unit.constructor_checks,
            1 => &mut unit.recipe_next_calls,
            2 => &mut unit.mask_next_calls,
            _ => &mut unit.mask_predicate_calls,
        };
        *count = 1;
        let maximum = unit.checked_repeated(usize::MAX).unwrap();
        assert_eq!(maximum.checked_add(unit), Err(LayoutError::Overflow));
        assert_eq!(maximum.checked_repeated(2), Err(LayoutError::Overflow));
        assert_eq!(maximum.checked_repeated(0).unwrap(), PlacementWork::default());
    }
    // Bound covers refusal too; actual short-circuit predicate count is smaller.
    use super::placement::CENSUS;
    let g = Geometry::new(8, 1, 4, 10).unwrap();
    for (mut bad, want_checks) in [(descriptor(g, 10), 1), (descriptor(g, 0), 2)] {
        bad.key_tile += 1;
        CENSUS.with(|c| c.set(PlacementWork::default()));
        assert!(AppendPlacement::new(g, bad).is_err());
        let actual = CENSUS.with(|c| c.get());
        assert_eq!(actual.constructor_checks, want_checks);
        assert!(actual.constructor_checks <= AppendPlacement::construction_work().constructor_checks);
        assert_eq!(actual.recipe_next_calls, 0);
        assert_eq!(actual.mask_next_calls, 0);
        assert_eq!(actual.mask_predicate_calls, 0);
    }
}

#[test]
fn hypothetical_payload_counts_include_full_rings_not_sparse_coordinates() {
    let g = Geometry::new(16, 2, 4, 17).unwrap();
    for n in [0, 1, 2, 3, 7, 8, 9, 16] {
        for op in AppendPlacement::new(g, descriptor(g, n)).unwrap() {
            let expected_additions = match op.destination {
                PlacementDestination::Key { .. } => if n % 2 == 0 { 0 } else { 16 },
                PlacementDestination::Value { .. } => if n % 8 == 0 { 0 } else { 16 },
            };
            assert_eq!(op.ideal_payload_work(), PlacementPayloadWork {
                source_mask_products: 16, rotation_slots: 16,
                destination_mask_products: 16, parent_additions: expected_additions,
                inactive_initializations: 0,
            });
        }
    }
    // No admission/limits API exists here: no invented one-short grant test.
}

#[test]
fn partial_boundary_delta_poison_and_partial_mask_census() {
    use super::placement::CENSUS;
    let g = Geometry::new(16, 2, 4, 10).unwrap();
    let mut contributions = 0;
    let mut masks = 0;
    for n in [1usize, 2, 7, 8, 9] {
        // Oracle inventory and masks come directly from logical coordinates.
        let destinations = [
            PlacementDestination::Key { tile: n / 2 },
            PlacementDestination::Value { group: n / 8, ciphertext: n / 8 * 4 },
            PlacementDestination::Value { group: n / 8, ciphertext: n / 8 * 4 + 1 },
            PlacementDestination::Value { group: n / 8, ciphertext: n / 8 * 4 + 2 },
            PlacementDestination::Value { group: n / 8, ciphertext: n / 8 * 4 + 3 },
        ];
        let mut expected_masks = [[[0u8; 16]; 2]; 5];
        let mut expected_delta = [[0i64; 16]; 5];
        let mut input_k = [-701i64; 16];
        let mut input_v = [809i64; 16];
        for head in 0..2 {
            for feature in 0..4 {
                let source = (feature * 2 + head) * 2;
                let dst = source + n % 2;
                let magnitude = (n * 20 + feature * 2 + head + 1) as i64;
                input_k[source] = if (feature + head) % 2 == 0 { magnitude } else { -magnitude };
                input_v[source] = if (feature + head) % 2 == 0 { -magnitude - 503 } else { magnitude + 503 };
                let value = 1 + ((n % 8) / 2 + 4 - feature) % 4;
                for (index, payload) in [(0, input_k[source]), (value, input_v[source])] {
                    expected_masks[index][0][source] = 1;
                    expected_masks[index][1][dst] = 1;
                    expected_delta[index][dst] = payload;
                }
            }
        }
        CENSUS.with(|c| c.set(PlacementWork::default()));
        let mut plan = AppendPlacement::new(g, descriptor(g, n)).unwrap();
        assert_eq!(CENSUS.with(|c| c.get()), PlacementWork {
            constructor_checks: 6, ..PlacementWork::default()
        });
        let mut want_census = PlacementWork {
            constructor_checks: 6, ..PlacementWork::default()
        };
        let mut actual_ops = Vec::new();
        for index in 0..5 {
            let op = plan.next().unwrap();
            want_census.recipe_next_calls += 1;
            assert_eq!(CENSUS.with(|c| c.get()), want_census);
            assert_eq!(op.destination, destinations[index]);
            assert_eq!(op.rotation, -((n % 2) as i32));
            assert_eq!(op.retain_parent, if index == 0 { n % 2 != 0 } else { n % 8 != 0 });
            let mut actual_masks = [[0u8; 16]; 2];
            for (kind, mask) in [op.source_mask, op.destination_mask].into_iter().enumerate() {
                let before = CENSUS.with(|c| c.get());
                let partial = PlacementWork {
                    mask_next_calls: 3, mask_predicate_calls: 3,
                    ..PlacementWork::default()
                };
                let complete = mask.scan_work().unwrap()
                    .checked_add(PlacementWork { mask_next_calls: 1, ..PlacementWork::default() }).unwrap()
                    .checked_add(mask.query_work()).unwrap();
                assert_eq!(complete, PlacementWork {
                    mask_next_calls: 18, mask_predicate_calls: 17,
                    ..PlacementWork::default()
                });
                assert_eq!(CENSUS.with(|c| c.get()), before, "quotes do not traverse APIs");
                let mut bits = mask.bits();
                assert_eq!(bits.len(), 16);
                for slot in 0..3 {
                    let bit = bits.next().unwrap();
                    actual_masks[kind][slot] = bit;
                    assert_eq!(bit, expected_masks[index][kind][slot]);
                    want_census.mask_next_calls += 1;
                    want_census.mask_predicate_calls += 1;
                    assert_eq!(CENSUS.with(|c| c.get()), want_census);
                }
                assert_eq!(bits.len(), 13);
                assert_eq!(CENSUS.with(|c| c.get()), before.checked_add(partial).unwrap());
                for slot in 3..16 {
                    let bit = bits.next().unwrap();
                    actual_masks[kind][slot] = bit;
                    assert_eq!(bit, expected_masks[index][kind][slot]);
                    want_census.mask_next_calls += 1;
                    want_census.mask_predicate_calls += 1;
                    assert_eq!(CENSUS.with(|c| c.get()), want_census);
                }
                assert_eq!(bits.len(), 0);
                assert_eq!(bits.next(), None);
                want_census.mask_next_calls += 1;
                assert_eq!(CENSUS.with(|c| c.get()), want_census);
                assert_eq!(bits.next(), None);
                want_census.mask_next_calls += 1;
                assert_eq!(CENSUS.with(|c| c.get()), want_census);
                assert_eq!(mask.selected(16), Err(LayoutError::Index));
                want_census.mask_predicate_calls += 1;
                assert_eq!(CENSUS.with(|c| c.get()), before.checked_add(complete).unwrap());
                assert_eq!(CENSUS.with(|c| c.get()), want_census);
                masks += 1;
            }
            actual_ops.push(op);
            contributions += 1;
        }
        assert_eq!(plan.next(), None);
        want_census.recipe_next_calls += 1;
        let literal = PlacementWork {
            constructor_checks: 6, recipe_next_calls: 6,
            mask_next_calls: 180, mask_predicate_calls: 170,
        };
        assert_eq!(want_census, literal);
        assert_eq!(CENSUS.with(|c| c.get()), literal);
        let before_quotes = CENSUS.with(|c| c.get());
        let quote = AppendPlacement::construction_work()
            .checked_add(AppendPlacement::next_work().checked_repeated(6).unwrap()).unwrap()
            .checked_add(PlacementWork {
                mask_next_calls: 18, mask_predicate_calls: 17,
                ..PlacementWork::default()
            }.checked_repeated(10).unwrap()).unwrap();
        assert_eq!(quote, literal);
        assert_eq!(CENSUS.with(|c| c.get()), before_quotes);

        // Delta/caller algebra, NOT a retained-state append or sanitization claim.
        // Old logical tokens are legitimate; every other parent coordinate is
        // finite nonzero poison, including coordinates targeted by this delta.
        for (index, op) in actual_ops.into_iter().enumerate() {
            let retain = if index == 0 { n % 2 != 0 } else { n % 8 != 0 };
            let mut parent = [12001i64; 16];
            for token in 0..n {
                for head in 0..2 {
                    for feature in 0..4 {
                        let ciphertext = if index == 0 { token / 2 } else {
                            token / 8 * 4 + ((token % 8) / 2 + 4 - feature) % 4
                        };
                        let expected_ciphertext = if index == 0 { n / 2 } else { n / 8 * 4 + index - 1 };
                        if ciphertext == expected_ciphertext {
                            let dst = (feature * 2 + head) * 2 + token % 2;
                            parent[dst] = -((token * 8 + head * 4 + feature + 1) as i64);
                        }
                    }
                }
            }
            let original = parent;
            let input = if index == 0 { &input_k } else { &input_v };
            // Reference uses internal infallible mask evaluation, not selected()
            // callbacks; the earlier public-API census remains independent.
            let mut delta = [0i64; 16];
            let mut fresh = [-23003i64; 16];
            PlacementContribution { retain_parent: false, ..op }
                .apply_i64_reference(input, None, &mut delta).unwrap();
            op.apply_i64_reference(input, if retain { Some(&parent) } else { None }, &mut fresh).unwrap();
            assert_eq!(delta, expected_delta[index]);
            let mut expected_fresh = [0i64; 16];
            for slot in 0..16 {
                expected_fresh[slot] = if retain { original[slot] + expected_delta[index][slot] }
                    else { expected_delta[index][slot] };
                if expected_masks[index][1][slot] != 0 {
                    assert_eq!(original[slot], 12001);
                    if retain {
                        assert_eq!(fresh[slot], 12001 + expected_delta[index][slot]);
                        assert_ne!(fresh[slot], expected_delta[index][slot], "target poison adds, never overwrites");
                    }
                }
            }
            assert_eq!(fresh, expected_fresh);
            assert_eq!(parent, original);
            let work = op.i64_reference_work().unwrap();
            assert_eq!((work.validation_slots, work.write_slots, work.mask_evaluations,
                work.parent_additions), (16, 16, 64, if retain { 32 } else { 0 }));
            // Caller buffer initialization and separate delta call are not free;
            // this quote names only the retained/fresh reference call itself.
        }
        assert_eq!(CENSUS.with(|c| c.get()), literal);
    }
    assert_eq!(contributions, 25);
    assert_eq!(masks, 50);
}

#[test]
fn partial_validity_masks_exact_prefix_tails_and_atomic_refusal() {
    let g = Geometry::new(16, 2, 4, 10).unwrap();
    let mut cases = 0;
    for active in [1usize, 2, 7, 8, 9] {
        for is_value in [false, true] {
            // Fixed allocation sizes independently: ceil(10/2)*16 and
            // ceil(10/8)*4*16. Expected bits derive only from logical tokens.
            let words = if is_value { 128 } else { 80 };
            let mut expected = vec![0u8; words];
            for token in 0..active {
                for head in 0..2 {
                    for feature in 0..4 {
                        let source = (feature * 2 + head) * 2;
                        let dst = source + token % 2;
                        let ciphertext = if is_value {
                            token / 8 * 4 + ((token % 8) / 2 + 4 - feature) % 4
                        } else { token / 2 };
                        expected[ciphertext * 16 + dst] = 1;
                    }
                }
            }
            let write = |mask: &mut [u8]| {
                if is_value { g.value_layout().validity_mask(active, mask) }
                else { g.key_layout().validity_mask(active, mask) }
            };
            let mut exact = vec![197u8; words];
            assert_eq!(write(&mut exact), Ok(()));
            assert_eq!(exact, expected);
            cases += 1;
            // Only the supplied exact prefix is in contract; spare allocation
            // bytes belong to this caller, not to the placement recipe.
            let mut allocation = vec![211u8; words + 3];
            assert_eq!(write(&mut allocation[..words]), Ok(()));
            assert_eq!(&allocation[..words], expected.as_slice());
            assert_eq!(&allocation[words..], &[211; 3]);
            cases += 1;
            for length in [words + 3, words - 1] {
                let mut refused = vec![223u8; length];
                let original = refused.clone();
                assert_eq!(write(&mut refused), Err(LayoutError::BufferLength));
                assert_eq!(refused, original);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 40);
    for is_value in [false, true] {
        let mut refused = vec![227u8; if is_value { 128 } else { 80 }];
        let original = refused.clone();
        let result = if is_value { g.value_layout().validity_mask(11, &mut refused) }
            else { g.key_layout().validity_mask(11, &mut refused) };
        assert_eq!(result, Err(LayoutError::Capacity));
        assert_eq!(refused, original);
        cases += 1;
    }
    // RetainedCache::new requires ENTIRE exact-sized storage to be zero.
    // These are refusals, never claims that poisoned storage is admitted.
    for case in 0..4 {
        let mut keys = vec![0i64; if case == 2 { 81 } else { 80 }];
        let mut values = vec![0i64; if case == 3 { 129 } else { 128 }];
        if case == 0 { keys[79] = -31; }
        if case == 1 { values[127] = 37; }
        let old_keys = keys.clone();
        let old_values = values.clone();
        let error = if case < 2 { LayoutError::NonZeroStorage } else { LayoutError::BufferLength };
        assert!(matches!(RetainedCache::new(g, &mut keys, &mut values), Err(e) if e == error));
        assert_eq!(keys, old_keys);
        assert_eq!(values, old_values);
        cases += 1;
    }
    assert_eq!(cases, 46);
}
