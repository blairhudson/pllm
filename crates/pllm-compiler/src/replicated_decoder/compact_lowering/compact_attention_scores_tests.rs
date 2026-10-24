//! Source-only definitions; E continuity plus geometry-only product debit needed.
//! Expanded control is TEST-only, never candidate source/proof authority.
use super::*;
use crate::replicated_decoder::program::request_origin::compact_storage::{
    EmissionContext, EmissionScope,
};
use crate::replicated_decoder::program::source_entry::SourceRequestImport;

#[test]
fn shared_recorder_refinement_changes_bounds_only_not_recipe_or_work() {
    source(2, 1, 4, 4, |builder, context, _| {
        let graph = at_scores(builder, context)?;
        let mut scores = lower(&graph)?;
        let tile = &scores.tiles[0];
        let recipe = (
            tile.left_offset,
            tile.right_offset,
            tile.shape,
            tile.rescale,
            tile.proof.as_ref().map(|p| p.digest()),
        );
        let ordinary = tile.ordinary_bounds;
        let inputs = tile.inputs;
        let before = context.work();
        // Numeric-helper test, NOT authenticated mass for score queries and NOT
        // Values implementation. I's future caller owns that authority.
        let expected = matrix::with_mass(ordinary, inputs.right, ONE as u64, true)?;
        refine_product_mass(&mut scores.tiles, recipe.1 + 1, recipe.1, ONE as u64)?;
        let tile = &scores.tiles[0];
        assert_eq!(tile.bounds, expected);
        assert_eq!(tile.ordinary_bounds, ordinary);
        assert_eq!(
            (
                tile.left_offset,
                tile.right_offset,
                tile.shape,
                tile.rescale,
                tile.proof.as_ref().map(|p| p.digest())
            ),
            recipe
        );
        assert_eq!(context.work(), before);
        assert!(
            refine_product_mass(&mut scores.tiles, recipe.1 + 1, recipe.1 + 9, ONE as u64).is_err()
        );
        Ok(())
    })
    .unwrap();
}

fn source<T>(
    heads: usize,
    kv: usize,
    width: usize,
    sequence: usize,
    run: impl for<'a> FnOnce(&'a Builder, &'a EmissionContext<'a>, &Manifest) -> Result<T, Error>,
) -> Result<T, Error> {
    let (config, _, clear) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    let mut config: serde_json::Value = serde_json::from_slice(&config).unwrap();
    config["hidden_size"] = serde_json::json!(heads * width);
    config["intermediate_size"] = serde_json::json!(heads * width);
    config["num_attention_heads"] = serde_json::json!(heads);
    config["num_key_value_heads"] = serde_json::json!(kv);
    config["head_dim"] = serde_json::json!(width);
    let config = serde_json::to_vec(&config).unwrap();
    let policy = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    let mut publisher = Publisher::new_with_policy(
        &config,
        sequence,
        2,
        [1; 32],
        [2; 32],
        WeightMode::Public,
        policy,
    )
    .unwrap();
    let shapes = publisher
        .shapes()
        .iter()
        .map(|(k, s)| (k.clone(), s.clone()))
        .collect::<Vec<_>>();
    let mut payloads = BTreeMap::new();
    for (key, shape) in shapes {
        let values = &clear[&key];
        let bytes = (0..shape.iter().product())
            .flat_map(|i| (values[i % values.len()] as f32 / ONE as f32).to_le_bytes())
            .collect::<Vec<_>>();
        for (frame, chunk) in bytes.chunks(4096).enumerate() {
            publisher.append(&key, frame * 1024, chunk).unwrap();
        }
        payloads.insert(key, bytes);
    }
    let tokens = (0..sequence)
        .map(|i| 1 + (i % 2) as u32)
        .collect::<Vec<_>>();
    publisher.seal(&tokens).unwrap();
    let mut manifest: Manifest = serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
    manifest.program = [0; 32];
    let mut importer = SourceRequestImport::new(&serde_json::to_vec(&manifest).unwrap())?;
    for (key, bytes) in payloads {
        for (frame, chunk) in bytes.chunks(4096).enumerate() {
            importer.append(&key, frame * 1024, chunk)?;
        }
    }
    importer.with_created(|created, guard| {
        let mut scope = EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
        let (builder, context, _) = scope.parts();
        run(builder, context, &manifest)
    })
}
fn at_scores<'a>(
    builder: &'a Builder,
    context: &'a EmissionContext<'a>,
) -> Result<CompactGraph<'a, 'a>, Error> {
    let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
    while graph
        .frontier
        .operations()
        .nth(graph.cursor)
        .ok_or(Error::Binding)?
        .1
        .operator
        != Op::AttentionScores
    {
        graph.lower_next()?;
    }
    Ok(graph)
}
fn lower(graph: &CompactGraph<'_, '_>) -> Result<Scores, Error> {
    let op = graph
        .frontier
        .operations()
        .nth(graph.cursor)
        .ok_or(Error::Binding)?
        .1;
    let (q, k) = graph.frontier.inputs(op);
    let q = q.ok_or(Error::Binding)?;
    let k = k.ok_or(Error::Binding)?;
    graph.lower_attention_scores(
        op,
        &[
            1,
            q.logical.dimensions[1],
            q.logical.dimensions[2],
            k.logical.dimensions[2],
        ],
        q,
        k,
    )
}
#[test]
fn genuine_qkv_rotary_gqa_recipes_match_independent_expanded_control() {
    for (heads, kv, width, seq) in [(1, 1, 2, 4), (2, 1, 4, 4), (2, 2, 8, 4), (1, 1, 128, 11)] {
        source(heads, kv, width, seq, |builder, context, manifest| {
            let graph = at_scores(builder, context)?;
            let original_prefix = builder.nodes.len();
            let before = context.work();
            let first = graph.next;
            let scores = lower(&graph)?;
            let layout = scores.layout()?;
            assert_eq!(
                builder.nodes.len(),
                original_prefix,
                "no expanded entry used to mint ownership"
            );
            assert_eq!(context.work(), before.checked_add(scores.work)?);
            assert_eq!(scores.rows.len(), heads * seq);
            let oracle = Program::compile_with_norms(
                &manifest.config,
                seq,
                2,
                &manifest.tensors,
                WeightMode::Public,
                manifest.execution,
                &manifest.norms,
            )
            .unwrap();
            let mut work = WorkBudget::default();
            let mut ordinal = 0;
            for (row_index, row) in scores.rows.iter().enumerate() {
                assert!(Arc::ptr_eq(&row.gather.view.origin, &scores.query.origin));
                assert_eq!(row.prefix, row_index % seq + 1);
                assert_eq!(row.suffix, seq - row_index % seq - 1);
                assert_eq!(row.node_offset, ordinal);
                let Code::Gather(words) = &oracle.nodes[first + ordinal].code else {
                    panic!("query gather once");
                };
                assert_eq!(oracle.nodes[first + ordinal].bounds, row.query_bounds);
                assert_eq!(row.query_start, row_index * width);
                assert_eq!(
                    row.key_start,
                    (row_index / seq) / (heads / kv) * seq * width
                );
                let gather = scores.gather(ordinal)?;
                assert_eq!(gather.len(), width);
                for j in 0..width {
                    assert_eq!(words.at(j), gather.lane_ref(j)?);
                }
                ordinal += 1;
                for tile in &scores.tiles[row.tile_start..row.tile_start + row.tile_count] {
                    assert!(Arc::ptr_eq(&tile.gather.view.origin, &scores.keys.origin));
                    assert_eq!(tile.right_offset, ordinal);
                    let Code::Gather(words) = &oracle.nodes[first + ordinal].code else {
                        panic!("right gather");
                    };
                    assert_eq!(oracle.nodes[first + ordinal].bounds, tile.inputs.right);
                    let gather = scores.gather(ordinal)?;
                    for j in 0..gather.len() {
                        assert_eq!(words.at(j), gather.lane_ref(j)?);
                    }
                    let original = &oracle.nodes[first + ordinal + 1];
                    let Code::Product {
                        left,
                        right,
                        shape,
                        rescale,
                        norm,
                    } = &original.code
                    else {
                        panic!("original product");
                    };
                    assert_eq!(*left, first + row.node_offset);
                    assert_eq!(*right, first + ordinal);
                    assert_eq!(*shape, Some(tile.shape));
                    assert!(*rescale);
                    assert_eq!(original.bounds, tile.bounds);
                    assert_eq!(
                        norm.as_deref().map(|p| p.digest()),
                        tile.proof.as_ref().map(|p| p.digest())
                    );
                    work = work.checked_add(WorkBudget::product(tile.count, true)?)?;
                    ordinal += 2;
                }
                for column in 0..seq {
                    let word = scores.word(first, row_index * seq + column)?;
                    assert_eq!(layout.word(first, row_index * seq + column)?, word);
                    if column >= row.prefix {
                        assert_eq!(word, scores.zero);
                    } else {
                        assert!(word.0 >= first + row.node_offset);
                    }
                }
            }
            assert_eq!(ordinal, scores.nodes);
            assert_eq!(work, scores.work);
            if width == 128 && seq == 11 {
                assert_eq!(puma::MAX_LANES, 1024, "actual tile 8+3, not assumed MAX");
                let last = scores.rows.last().unwrap();
                assert_eq!(
                    scores.tiles[last.tile_start..last.tile_start + last.tile_count]
                        .iter()
                        .map(|t| t.count)
                        .collect::<Vec<_>>(),
                    vec![8, 3]
                );
                let split =
                    WorkBudget::product(8, true)?.checked_add(WorkBudget::product(3, true)?)?;
                assert_ne!(
                    split,
                    WorkBudget::product(11, true)?,
                    "per-invocation rounding/cost is not collapsed"
                );
            }
            Ok(())
        })
        .unwrap();
    }
}
#[test]
fn owned_proof_uses_query_row_and_max_actual_intact_key_rows() {
    source(2, 1, 4, 4, |builder, context, _| {
        let graph = at_scores(builder, context)?;
        let scores = lower(&graph)?;
        for row in &scores.rows {
            let q = graph
                .attention_energy(&scores.query, row.query_start, scores.width)?
                .expect("genuine RoPE bound");
            for tile in &scores.tiles[row.tile_start..row.tile_start + row.tile_count] {
                let mut max = 0;
                for key in 0..tile.count {
                    max = max.max(
                        graph
                            .attention_energy(
                                &scores.keys,
                                tile.right_start + key * scores.width,
                                scores.width,
                            )?
                            .unwrap(),
                    );
                }
                let proof = tile.proof.as_ref().unwrap();
                assert_eq!(proof.squared_energy_maxima(), [q, max]);
                assert_eq!(proof.input_bounds(), tile.inputs);
            }
        }
        assert!(graph.row_norm(&scores.query)?.is_none());
        assert!(graph
            .attention_energy(&scores.query, 1, scores.width)?
            .is_none());
        let mut forged = scores.keys.clone();
        forged.start += 1;
        assert!(graph.attention_energy(&forged, 0, scores.width).is_err());
        assert!(
            score_energy(&graph, &forged, 0, scores.width).is_err(),
            "identity errors never select GENERAL fallback"
        );
        Ok(())
    })
    .unwrap();
}
#[test]
fn genuine_general_norm_is_not_attention_certificate() {
    source(1, 1, 4, 4, |builder, context, _| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        loop {
            let op = graph
                .frontier
                .operations()
                .nth(graph.cursor)
                .ok_or(Error::Binding)?
                .1
                .operator;
            let view = graph.lower_next()?;
            if op == Op::RmsNorm {
                assert!(graph.row_norm(&view)?.is_some());
                assert!(graph.attention_energy(&view, 0, 4)?.is_none());
                assert!(
                    score_energy(&graph, &view, 0, 4)?.is_some(),
                    "GENERAL input fallback only after actual ATT absence"
                );
                break;
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn preexisting_work_and_actual_split_preserve_rescaled_97_not_collapsed_96() -> Result<(), Error> {
    // Numeric work-contract test only; no source authority minted here. Real
    // source-ledger accumulation is independently checked above.
    let prior = WorkBudget::product(11, false)?;
    let split = WorkBudget::product(8, false)?.checked_add(WorkBudget::product(3, false)?)?;
    assert_eq!(prior.checked_add(split)?.cursors().aes_work, 24);
    assert_eq!(
        prior
            .checked_add(WorkBudget::product(11, false)?)?
            .cursors()
            .aes_work,
        24
    );
    // Actual Scores include rescale: 48 prior + 49 split = 97, versus
    // 48 prior + 48 collapsed = 96. Keep both primitive contracts explicit.
    let prior = WorkBudget::product(11, true)?;
    let split = WorkBudget::product(8, true)?.checked_add(WorkBudget::product(3, true)?)?;
    assert_eq!(prior.checked_add(split)?.cursors().aes_work, 97);
    assert_eq!(
        prior
            .checked_add(WorkBudget::product(11, true)?)?
            .cursors()
            .aes_work,
        96
    );
    Ok(())
}
#[test]
fn exact_and_one_short_preentry_reservation_before_product_debit() {
    for short in [false, true] {
        source(2, 1, 4, 4, |builder, context, _| {
            let mut graph = at_scores(builder, context)?;
            let mut attempt = context.attempt();
            let peak = graph.score_peak(8, 8, size_of_val(&attempt))?;
            attempt.commit();
            graph.test_lower_limit(peak - usize::from(short));
            let before = context.work();
            let result = lower(&graph);
            if short {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_eq!(context.work(), before);
                assert!(context.check_open().is_err());
            } else {
                result?;
            }
            Ok(())
        })
        .unwrap();
    }
}
#[test]
fn postdebit_publication_failure_burns_without_inner_phase_refund() {
    for publication in [false, true] {
        let plan = source(2, 1, 4, 4, |builder, context, _| {
            let graph = at_scores(builder, context)?;
            let before = context.work();
            let charged = context.reservation().payload_bytes();
            if publication {
                FAIL_PUBLICATION.with(|f| f.set(true));
            } else {
                FAIL_AFTER_DEBIT.with(|f| f.set(true));
            }
            assert!(matches!(lower(&graph), Err(Error::Allocation)));
            assert!(context.work().operations() > before.operations());
            assert!(context.check_open().is_err());
            assert!(context.reservation().payload_bytes() >= charged);
            Ok(Arc::clone(context.plan()))
        })
        .unwrap();
        assert!(plan.budget().used_bytes() < plan.budget().admitted_peak_bytes());
    }
}

#[test]
fn changed_bound_input_is_rejected_before_work_not_reinterpreted() {
    source(2, 1, 4, 4, |builder, context, _| {
        let graph = at_scores(builder, context)?;
        let op = graph.frontier.operations().nth(graph.cursor).unwrap().1;
        let (q, k) = graph.frontier.inputs(op);
        let q = q.unwrap();
        let k = k.unwrap();
        let mut changed = q.clone();
        changed.start += 1;
        let before = context.work();
        assert!(graph
            .lower_attention_scores(op, &[1, 2, 4, 4], &changed, k)
            .is_err());
        assert_eq!(context.work(), before);
        assert!(context.check_open().is_err());
        Ok(())
    })
    .unwrap();
}

#[test]
fn causal_layout_geometry_matches_literal_prefix_full_tail_node_order() {
    // Public geometry test only: these coordinates issue no producer/proof.
    for sequence in [1, 4, 11] {
        for past in [0, 3, 17] {
            for tile in [8, 128] {
                let columns = past + sequence;
                let layout = ScoreLayout {
                    rows: 2 * sequence,
                    sequence,
                    columns,
                    tile,
                    zero: (1, 0),
                };
                let mut next = 0;
                for row in 0..2 * sequence {
                    let prefix = past + row % sequence + 1;
                    next += 1; // one query gather
                    for start in (0..prefix).step_by(tile) {
                        next += 1; // physical right gather
                        let count = tile.min(prefix - start);
                        for j in 0..count {
                            assert_eq!(
                                layout.word(20, row * columns + start + j).unwrap(),
                                (20 + next, j)
                            );
                        }
                        next += 1; // product
                    }
                    for j in prefix..columns {
                        assert_eq!(layout.word(20, row * columns + j).unwrap(), (1, 0));
                    }
                }
                assert!(layout.word(20, 2 * sequence * columns).is_err());
            }
        }
    }
}
