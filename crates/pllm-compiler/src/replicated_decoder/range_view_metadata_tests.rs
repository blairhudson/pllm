//! Bounded, construction-only coverage of checked-range descriptor admission.
use super::*;

fn input(tiles: usize, bounds: Interval) -> (Builder, Words) {
    let lanes = tiles.checked_mul(tensor::MAX_TILE_LANES).unwrap();
    let mut builder = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let owner = builder
        .push(Code::Input("range.input".into()), lanes, bounds)
        .unwrap();
    let words = (0..lanes).map(|lane| (owner, lane)).collect();
    (builder, words)
}

fn view_fee(tiles: usize) -> usize {
    Words::span_pool_bytes(tiles)
        .unwrap()
        .checked_mul(2)
        .unwrap()
}

// Independently construct the original per-tile recipe and doubling Words pool.
// Exclude only the returned-view fee so the exact new debit is measured apart
// from unchanged node, gather, primitive-interning and sticky-product charges.
fn legacy_recipe(builder: &mut Builder, words: &Words) -> Words {
    let bound = builder.checked_bound.unwrap();
    let accepted = Interval::new(-bound, bound).unwrap();
    let mut output = Words::new();
    for tile in words.chunks(tensor::MAX_TILE_LANES) {
        let n = tile.len();
        let input = builder.gather(&tile).unwrap();
        let program =
            tensor::Program::checked_range(n, builder.nodes[input].bounds, accepted, 0).unwrap();
        let digest = program.digest();
        let program = if let Some(program) = builder.tensors.get(&digest) {
            program.clone()
        } else {
            builder
                .charge_program_bytes(program.admitted_bytes().checked_add(1024).unwrap())
                .unwrap();
            let program = Arc::new(program);
            builder.tensors.insert(digest, program.clone());
            program
        };
        let owner = builder
            .push(
                Code::CheckedRange { input, program },
                n.checked_mul(2).unwrap(),
                Interval::new(-bound, bound.max(1)).unwrap(),
            )
            .unwrap();
        let safe = (0..n).map(|lane| (owner, lane)).collect();
        let safe = builder.push(Code::Gather(safe), n, accepted).unwrap();
        let valid = (n..2 * n).map(|lane| (owner, lane)).collect();
        let valid = builder
            .push(Code::Gather(valid), n, Interval::new(0, 1).unwrap())
            .unwrap();
        let valid = builder
            .tensor(valid, tensor::Program::all_valid(n).unwrap())
            .unwrap();
        builder.validity = Some(match builder.validity {
            Some(previous) => builder.product(previous, valid, None, false).unwrap(),
            None => valid,
        });
        output.extend((0..n).map(|lane| (safe, lane)));
    }
    output
}

#[test]
fn checked_range_exact_pool_1_2_3_5_tiles_preserves_legacy_schedule_and_identity() {
    for tiles in [1usize, 2, 3, 5] {
        let incoming = Interval::new(-12, 12).unwrap();
        let accepted = Interval::new(-9, 9).unwrap();
        let (mut builder, words) = input(tiles, incoming);
        let (mut legacy, legacy_words) = input(tiles, incoming);
        let safe = builder.checked_view(&words).unwrap();
        let legacy_safe = legacy_recipe(&mut legacy, &legacy_words);

        assert_eq!(safe.len(), words.len());
        assert_eq!(safe.metadata_bytes(), Words::span_pool_bytes(tiles).unwrap());
        assert_eq!(
            legacy_safe.metadata_bytes(),
            Words::span_pool_bytes(tiles.next_power_of_two()).unwrap()
        );
        assert_eq!(safe.owners().count(), tiles);
        let expected: Vec<_> = builder
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| match &node.code {
                Code::Gather(gathered) if node.bounds == accepted => {
                    assert_eq!(gathered.len(), tensor::MAX_TILE_LANES);
                    Some((0..node.lanes).map(move |lane| (id, lane)))
                }
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(safe, expected);
        assert!(safe.iter().eq(legacy_safe.iter()));
        assert_eq!(builder.bounds(&words).unwrap(), incoming);
        assert_eq!(builder.nodes[0].bounds, incoming);
        assert_eq!(builder.bounds(&safe).unwrap(), accepted);
        assert_eq!(builder.validity, legacy.validity);
        assert_eq!(builder.nodes.len(), legacy.nodes.len());
        assert_eq!(builder.allocated, legacy.allocated + view_fee(tiles));
        assert!(builder.allocated < MAX_PROGRAM_BYTES);
        assert_eq!(builder.workspace_bytes, legacy.workspace_bytes);
        for (actual, original) in builder.nodes.iter().zip(&legacy.nodes) {
            assert_eq!(actual.lanes, original.lanes);
            assert_eq!(actual.bounds, original.bounds);
            assert_eq!(actual.inputs(), original.inputs());
            assert_eq!(
                actual.work_budget().unwrap(),
                original.work_budget().unwrap()
            );
            match (&actual.code, &original.code) {
                (Code::Gather(a), Code::Gather(b)) => {
                    assert!(a.iter().eq(b.iter()));
                    assert_eq!(a.metadata_bytes(), b.metadata_bytes());
                }
                (
                    Code::CheckedRange { program: a, .. },
                    Code::CheckedRange { program: b, .. },
                )
                | (Code::Tensor { program: a, .. }, Code::Tensor { program: b, .. }) => {
                    assert_eq!(a.digest(), b.digest());
                    assert_eq!(a.admitted_bytes(), b.admitted_bytes());
                    assert!(Arc::ptr_eq(a, &builder.tensors[&a.digest()]));
                    assert!(Arc::ptr_eq(b, &legacy.tensors[&b.digest()]));
                }
                (Code::Input(a), Code::Input(b)) => assert_eq!(a, b),
                (Code::Product { .. }, Code::Product { .. }) => {}
                _ => panic!("range recipe changed node kind"),
            }
        }
        assert!(builder.tensors.keys().eq(legacy.tensors.keys()));
        assert_eq!(
            schedule_work(&builder.nodes).unwrap(),
            schedule_work(&legacy.nodes).unwrap()
        );

        // Original finalizer hashes every ordered node and output, and derives
        // retirement/work/PRSS identity without any runtime or native execution.
        let actual = builder
            .finish(
                1,
                safe.iter().collect(),
                &mut AdmissionSite::default(),
                |_| Ok(()),
                || [0; 32],
            )
            .unwrap();
        let original = legacy
            .finish(
                1,
                legacy_safe.iter().collect(),
                &mut AdmissionSite::default(),
                |_| Ok(()),
                || [0; 32],
            )
            .unwrap();
        assert_eq!(actual.digest(), original.digest());
        assert_eq!(actual.outputs, original.outputs);
        assert_eq!(actual.retire, original.retire);
        assert_eq!(actual.work_budget(), original.work_budget());
        assert_eq!(
            actual.work_budget().limits().unwrap(),
            original.work_budget().limits().unwrap()
        );
        assert_eq!(
            actual.payload_bytes(),
            original.payload_bytes() + view_fee(tiles)
        );
    }
}

#[test]
fn checked_range_exact_pool_partial_last_tile_preserves_legacy_order_and_digest() {
    let width = tensor::MAX_TILE_LANES;
    let lanes = width.checked_mul(2).unwrap().checked_add(1).unwrap();
    let incoming = Interval::new(-12, 12).unwrap();
    let accepted = Interval::new(-9, 9).unwrap();
    let (mut builder, full_words) = input(3, incoming);
    let (mut legacy, legacy_full_words) = input(3, incoming);
    let words = full_words.slice(..lanes);
    let legacy_words = legacy_full_words.slice(..lanes);
    let safe = builder.checked_view(&words).unwrap();
    let legacy_safe = legacy_recipe(&mut legacy, &legacy_words);
    assert_eq!(safe.len(), lanes);
    assert_eq!(safe.owners().count(), 3);
    assert_eq!(safe.metadata_bytes(), Words::span_pool_bytes(3).unwrap());
    assert_eq!(legacy_safe.metadata_bytes(), Words::span_pool_bytes(4).unwrap());
    assert!(safe.iter().eq(legacy_safe.iter()));
    let mut expected = Vec::new();
    let mut tile_lanes = Vec::new();
    for (owner, node) in builder.nodes.iter().enumerate() {
        if let Code::CheckedRange { input, .. } = &node.code {
            let Code::Gather(gathered) = &builder.nodes[*input].code else {
                panic!("range input must remain an ordered gather");
            };
            let start = expected.len();
            assert!(gathered.iter().eq(words.slice(start..start + gathered.len()).iter()));
            tile_lanes.push(gathered.len());
            expected.extend((0..gathered.len()).map(|lane| (owner + 1, lane)));
        }
    }
    assert_eq!(tile_lanes, vec![width, width, 1]);
    assert_eq!(safe, expected);
    assert_eq!(builder.bounds(&words).unwrap(), incoming);
    assert_eq!(builder.bounds(&safe).unwrap(), accepted);
    assert_eq!(builder.validity, legacy.validity);
    assert_eq!(builder.nodes.len(), legacy.nodes.len());
    for (actual, original) in builder.nodes.iter().zip(&legacy.nodes) {
        assert_eq!(actual.lanes, original.lanes);
        assert_eq!(actual.bounds, original.bounds);
        assert_eq!(actual.inputs(), original.inputs());
        assert_eq!(actual.work_budget().unwrap(), original.work_budget().unwrap());
    }
    assert_eq!(builder.allocated, legacy.allocated + view_fee(3));
    assert!(builder.allocated < MAX_PROGRAM_BYTES);
    assert_eq!(builder.workspace_bytes, legacy.workspace_bytes);
    assert!(builder.tensors.keys().eq(legacy.tensors.keys()));
    assert_eq!(
        schedule_work(&builder.nodes).unwrap(),
        schedule_work(&legacy.nodes).unwrap()
    );
    let actual = builder
        .finish(
            1,
            safe.iter().collect(),
            &mut AdmissionSite::default(),
            |_| Ok(()),
            || [0; 32],
        )
        .unwrap();
    let original = legacy
        .finish(
            1,
            legacy_safe.iter().collect(),
            &mut AdmissionSite::default(),
            |_| Ok(()),
            || [0; 32],
        )
        .unwrap();
    assert_eq!(actual.digest(), original.digest());
    assert_eq!(actual.outputs, original.outputs);
    assert_eq!(actual.retire, original.retire);
    assert_eq!(actual.work_budget(), original.work_budget());
    assert_eq!(actual.payload_bytes(), original.payload_bytes() + view_fee(3));
}

#[test]
fn checked_range_exact_pool_fragmented_inputs_preserve_legacy_order_and_digest() {
    let width = tensor::MAX_TILE_LANES;
    let lanes = width.checked_mul(2).unwrap().checked_add(1).unwrap();
    let incoming = Interval::new(-12, 12).unwrap();
    let accepted = Interval::new(-9, 9).unwrap();
    let mut builder = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let mut legacy = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    for owner in 0..3 {
        for graph in [&mut builder, &mut legacy] {
            assert_eq!(
                graph
                    .push(Code::Input(format!("range.input.{owner}")), lanes, incoming)
                    .unwrap(),
                owner
            );
        }
    }
    // Reordered owners, strided/broadcast runs and unequal span lengths. The
    // middle run crosses the first tile boundary; the final run crosses the next.
    let ordered: Vec<_> = (0..width - 3)
        .map(|lane| (2, lane * 2))
        .chain(std::iter::repeat_n((0, 7), width + 1))
        .chain((0..3).map(|lane| (1, 1 + lane * 3)))
        .collect();
    let words: Words = ordered.iter().copied().collect();
    let legacy_words: Words = ordered.iter().copied().collect();
    assert_eq!(words.len(), lanes);
    assert_eq!(words.owners().collect::<Vec<_>>(), vec![2, 0, 1]);
    assert_eq!(
        words
            .chunks(width)
            .map(|tile| tile.owners().count())
            .collect::<Vec<_>>(),
        vec![2, 2, 1]
    );
    let safe = builder.checked_view(&words).unwrap();
    let legacy_safe = legacy_recipe(&mut legacy, &legacy_words);
    assert_eq!(safe.len(), lanes);
    assert_eq!(safe.owners().count(), 3);
    assert_eq!(safe.metadata_bytes(), Words::span_pool_bytes(3).unwrap());
    assert_eq!(legacy_safe.metadata_bytes(), Words::span_pool_bytes(4).unwrap());
    assert!(safe.iter().eq(legacy_safe.iter()));
    let mut expected = Vec::new();
    for (owner, node) in builder.nodes.iter().enumerate() {
        if let Code::CheckedRange { input, .. } = &node.code {
            let Code::Gather(gathered) = &builder.nodes[*input].code else {
                panic!("range input must remain an ordered gather");
            };
            let start = expected.len();
            assert!(gathered.iter().eq(ordered[start..start + gathered.len()].iter().copied()));
            expected.extend((0..gathered.len()).map(|lane| (owner + 1, lane)));
        }
    }
    assert_eq!(safe, expected);
    assert_eq!(builder.bounds(&words).unwrap(), incoming);
    assert!(builder.nodes[..3].iter().all(|node| node.bounds == incoming));
    assert_eq!(builder.bounds(&safe).unwrap(), accepted);
    assert_eq!(builder.validity, legacy.validity);
    assert_eq!(builder.nodes.len(), legacy.nodes.len());
    for (actual, original) in builder.nodes.iter().zip(&legacy.nodes) {
        assert_eq!(actual.lanes, original.lanes);
        assert_eq!(actual.bounds, original.bounds);
        assert_eq!(actual.inputs(), original.inputs());
        assert_eq!(actual.work_budget().unwrap(), original.work_budget().unwrap());
    }
    assert_eq!(builder.allocated, legacy.allocated + view_fee(3));
    assert!(builder.allocated < MAX_PROGRAM_BYTES);
    assert_eq!(builder.workspace_bytes, legacy.workspace_bytes);
    assert!(builder.tensors.keys().eq(legacy.tensors.keys()));
    assert_eq!(
        schedule_work(&builder.nodes).unwrap(),
        schedule_work(&legacy.nodes).unwrap()
    );
    let actual = builder
        .finish(
            3,
            safe.iter().collect(),
            &mut AdmissionSite::default(),
            |_| Ok(()),
            || [0; 32],
        )
        .unwrap();
    let original = legacy
        .finish(
            3,
            legacy_safe.iter().collect(),
            &mut AdmissionSite::default(),
            |_| Ok(()),
            || [0; 32],
        )
        .unwrap();
    assert_eq!(actual.digest(), original.digest());
    assert_eq!(actual.outputs, original.outputs);
    assert_eq!(actual.retire, original.retire);
    assert_eq!(actual.work_budget(), original.work_budget());
    assert_eq!(actual.payload_bytes(), original.payload_bytes() + view_fee(3));
}

#[test]
fn checked_range_exact_pool_fee_refusal_precedes_graph_mutation() {
    for tiles in [1usize, 2, 3, 5] {
        let incoming = Interval::new(-12, 12).unwrap();
        let (mut builder, words) = input(tiles, incoming);
        let fee = view_fee(tiles);
        builder.allocated = MAX_PROGRAM_BYTES.checked_sub(fee).unwrap() + 1;
        let allocated = builder.allocated;
        let work = schedule_work(&builder.nodes).unwrap();
        assert!(matches!(builder.checked_view(&words), Err(Error::Bounds)));
        assert_eq!(builder.admission_stage, "checked_range_view_metadata");
        assert_eq!(builder.admission_limit, Some("program_bytes"));
        assert_eq!(builder.requested_bytes, Some(MAX_PROGRAM_BYTES + 1));
        assert_eq!(builder.allocated, allocated);
        assert_eq!(builder.nodes.len(), 1);
        assert!(matches!(&builder.nodes[0].code, Code::Input(key) if key == "range.input"));
        assert_eq!(builder.nodes[0].bounds, incoming);
        assert_eq!(builder.bounds(&words).unwrap(), incoming);
        assert!(builder.tensors.is_empty());
        assert_eq!(builder.validity, None);
        assert_eq!(schedule_work(&builder.nodes).unwrap(), work);
    }
}

#[test]
fn checked_range_contained_alias_skips_pool_fee_at_full_metadata_cap() {
    for tiles in [1usize, 2, 3, 5] {
        let incoming = Interval::new(-9, 9).unwrap();
        let (mut builder, words) = input(tiles, incoming);
        builder.allocated = MAX_PROGRAM_BYTES;
        let requested = builder.requested_bytes;
        let safe = builder.checked_view(&words).unwrap();
        assert_eq!(safe, words);
        assert_eq!(safe.metadata_bytes(), words.metadata_bytes());
        assert_eq!(builder.allocated, MAX_PROGRAM_BYTES);
        assert_eq!(builder.requested_bytes, requested);
        assert_eq!(builder.admission_stage, "checked_range_domain");
        assert_eq!(builder.nodes.len(), 1);
        assert!(builder.tensors.is_empty());
        assert_eq!(builder.validity, None);
        assert_eq!(builder.bounds(&safe).unwrap(), incoming);
    }
}

#[test]
fn checked_range_contained_duplicate_spans_preserve_multiplicity_at_cap() {
    let width = tensor::MAX_TILE_LANES;
    let incoming = Interval::new(-9, 9).unwrap();
    let (mut builder, full_words) = input(1, incoming);
    let owner = full_words.at(0).0;
    let ordered: Vec<_> = (0..3)
        .flat_map(|_| (0..width - 1).map(|lane| (owner, lane)))
        .collect();
    let words: Words = ordered.iter().copied().collect();
    assert_eq!(words.len(), 3 * (width - 1));
    assert_eq!(words.owners().collect::<Vec<_>>(), vec![owner; 3]);
    assert!(words.iter().eq(ordered.iter().copied()));
    // Earlier overallocation may fit three spans without another growth, so
    // four-span capacity is not guaranteed. Derived Clone also need not retain
    // spare capacity or allocate exactly three; only three-span minima apply.
    let original_metadata = words.metadata_bytes();
    assert!(original_metadata >= Words::span_pool_bytes(3).unwrap());
    // Synthetic logical-program-budget test: checked_view adds no program charge.
    // Words::clone does allocate a descriptor Vec; this does not establish an admitted
    // construction frontier, combined host peak, or wipe guarantee.
    builder.allocated = MAX_PROGRAM_BYTES;
    let allocated = builder.allocated;
    let requested = builder.requested_bytes;
    let nodes = builder.nodes.len();
    let tensors: Vec<_> = builder.tensors.keys().copied().collect();
    let validity = builder.validity;

    let safe = builder.checked_view(&words).unwrap();

    assert_eq!(safe.len(), 3 * (width - 1));
    assert!(safe.iter().eq(ordered.iter().copied()));
    assert!(safe.iter().eq(words.iter()));
    assert_eq!(safe.owners().collect::<Vec<_>>(), vec![owner; 3]);
    assert_eq!(safe, words);
    assert!(safe.metadata_bytes() >= Words::span_pool_bytes(3).unwrap());
    assert_eq!(words.metadata_bytes(), original_metadata);
    assert_eq!(builder.allocated, allocated);
    assert_eq!(builder.requested_bytes, requested);
    assert_eq!(builder.admission_stage, "checked_range_domain");
    assert_eq!(builder.nodes.len(), nodes);
    assert!(matches!(&builder.nodes[owner].code, Code::Input(key) if key == "range.input"));
    assert_eq!(builder.nodes[owner].lanes, width);
    assert_eq!(builder.nodes[owner].bounds, incoming);
    assert_eq!(builder.tensors.keys().copied().collect::<Vec<_>>(), tensors);
    assert_eq!(builder.validity, validity);
    assert_eq!(builder.bounds(&safe).unwrap(), incoming);
}

#[test]
fn checked_range_contained_invalid_second_lane_rejected_before_alias_or_fee() {
    let width = tensor::MAX_TILE_LANES;
    let incoming = Interval::new(-9, 9).unwrap();
    let (mut builder, full_words) = input(1, incoming);
    assert_eq!(full_words.at(0).0, 0);
    assert_eq!(builder.nodes[0].lanes, width);
    let words: Words = [(0, width - 1), (0, width)].into_iter().collect();
    assert_eq!(words.len(), 2);
    assert!(words.iter().eq([(0, width - 1), (0, width)]));
    // The owner interval is contained, but the second lane is outside its backing.
    // A full logical byte budget must not mask shape validation with a fee refusal.
    builder.allocated = MAX_PROGRAM_BYTES;
    let allocated = builder.allocated;
    let requested = builder.requested_bytes;
    let limit = builder.admission_limit;
    let nodes = builder.nodes.len();
    let tensors: Vec<_> = builder.tensors.keys().copied().collect();
    let validity = builder.validity;
    let workspace = builder.workspace_bytes;
    let work = schedule_work(&builder.nodes).unwrap();

    assert!(matches!(builder.checked_view(&words), Err(Error::Shape)));

    assert_eq!(builder.admission_stage, "checked_range_domain");
    assert_eq!(builder.admission_limit, limit);
    assert_eq!(builder.allocated, allocated);
    assert_eq!(builder.requested_bytes, requested);
    assert_eq!(builder.nodes.len(), nodes);
    assert!(matches!(&builder.nodes[0].code, Code::Input(key) if key == "range.input"));
    assert_eq!(builder.nodes[0].lanes, width);
    assert_eq!(builder.nodes[0].bounds, incoming);
    assert_eq!(builder.tensors.keys().copied().collect::<Vec<_>>(), tensors);
    assert_eq!(builder.validity, validity);
    assert_eq!(builder.workspace_bytes, workspace);
    assert_eq!(schedule_work(&builder.nodes).unwrap(), work);
}
