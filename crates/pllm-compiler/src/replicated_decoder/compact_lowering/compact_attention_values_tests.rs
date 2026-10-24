//! UNCOMPILED integration definitions. Require genuine E Scores/Scale/Softmax
//! publication and softmax_rows_for bridge. No synthetic PublishedOp/producer,
//! no primitive-only substitute for the source-driven acceptance tests below.
use super::*;
use crate::replicated_decoder::program::request_origin::compact_storage::{
    EmissionContext, EmissionScope,
};
use crate::replicated_decoder::program::source_entry::SourceRequestImport;

fn source<T>(
    heads: usize,
    kv: usize,
    width: usize,
    sequence: usize,
    uniform: bool,
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
            .flat_map(|i| {
                let x = if uniform {
                    if key.contains("norm.weight") {
                        1.0f32
                    } else {
                        1.0f32 / 1024.0
                    }
                } else {
                    values[i % values.len()] as f32 / ONE as f32
                };
                x.to_le_bytes()
            })
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
fn at_values<'a>(
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
        != Op::AttentionValues
    {
        // Genuine Scores -> original Scale -> certified Softmax must lower here.
        // If integration is absent this is BLOCKED, never mint a fake producer.
        graph.lower_next().inspect_err(|error| {
            eprintln!(
                "source Values prerequisite: {:?}, operator {:?}",
                error,
                graph.failure_site().map(|site| site.operator)
            );
        })?;
    }
    Ok(graph)
}
fn lower<'s>(graph: &CompactGraph<'_, 's>) -> Result<Values<'s>, Error> {
    let op = graph
        .frontier
        .operations()
        .nth(graph.cursor)
        .ok_or(Error::Binding)?
        .1;
    let (p, v) = graph.frontier.inputs(op);
    let p = p.ok_or(Error::Binding)?;
    let v = v.ok_or(Error::Binding)?;
    graph.lower_attention_values(
        op,
        &[
            1,
            p.logical.dimensions[1],
            p.logical.dimensions[2],
            v.logical.dimensions[3],
        ],
        p,
        v,
    )
}

#[test]
fn genuine_source_scores_scale_softmax_values_match_original_gqa_refs_bounds_proofs_work() {
    for (heads, kv, width, seq) in [(1, 1, 2, 4), (2, 1, 4, 4), (2, 2, 8, 4), (1, 1, 128, 11)] {
        eprintln!("Values original geometry heads={heads} kv={kv} width={width} seq={seq}");
        source(
            heads,
            kv,
            width,
            seq,
            false,
            |builder, context, manifest| {
                let graph = at_values(builder, context)?;
                let before = context.work();
                let prefix_nodes = builder.nodes.len();
                let first = graph.next;
                let values = lower(&graph).inspect_err(|error| {
                    eprintln!("Values emission: {error:?}");
                })?;
                let layout = values.layout()?;
                assert_eq!(
                    builder.nodes.len(),
                    prefix_nodes,
                    "no expanded entry prerequisite"
                );
                assert_eq!(context.work(), before.checked_add(values.work)?);
                // Independent TEST-only control constructed AFTER actual candidate.
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
                let mut ordinal = 0;
                let mut work = WorkBudget::default();
                for (index, row) in values.rows.iter().enumerate() {
                    assert_eq!(row.inner, index % seq + 1);
                    assert_eq!(row.gather_offset, ordinal);
                    assert_eq!(row.probability_start, index * seq);
                    assert_eq!(row.value_start, (index / seq) / (heads / kv) * seq * width);
                    let evidence = row
                        .probability
                        .as_ref()
                        .expect("actual certified Softmax producer");
                    evidence.check()?;
                    assert_eq!(evidence.width, row.inner);
                    assert_eq!(row.input_bounds, evidence.output);
                    let Code::Gather(words) = &oracle.nodes[first + ordinal].code else {
                        panic!("original native prefix Gather");
                    };
                    for k in 0..row.inner {
                        assert_eq!(row.gather.lane_ref(k)?, words.at(k));
                        assert_eq!(words.at(k), (evidence.tensor, k));
                    }
                    assert_eq!(oracle.nodes[first + ordinal].bounds, row.input_bounds);
                    ordinal += 1;
                    for tile in &values.tiles[row.tile_start..row.tile_start + row.tile_count] {
                        assert_eq!(tile.right_offset, ordinal);
                        assert_eq!(tile.left_offset, row.gather_offset);
                        assert!(tile.proof.is_none());
                        assert!(tile.rescale);
                        let Code::Gather(words) = &oracle.nodes[first + ordinal].code else {
                            panic!("right Gather");
                        };
                        for k in 0..row.inner {
                            for j in 0..tile.count {
                                let i = k * tile.count + j;
                                assert_eq!(
                                    tile.gather.lane_ref(i)?,
                                    values.values.word(
                                        row.value_start + k * width + tile.column_start + j
                                    )?
                                );
                                assert_eq!(tile.gather.lane_ref(i)?, words.at(i));
                            }
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
                            panic!("ordinary Product recipe");
                        };
                        assert_eq!(*left, first + row.gather_offset);
                        assert_eq!(*right, first + tile.right_offset);
                        assert_eq!(*shape, Some(tile.shape));
                        assert!(*rescale);
                        assert!(norm.is_none());
                        assert_eq!(
                            tile.ordinary_bounds,
                            product_bounds(tile.inputs, row.inner, true, true, None)?
                        );
                        assert_eq!(
                            tile.bounds,
                            matrix::with_mass(
                                tile.ordinary_bounds,
                                tile.inputs.right,
                                evidence.mass,
                                true
                            )?
                        );
                        assert_eq!(original.bounds, tile.bounds);
                        work = work.checked_add(WorkBudget::product(tile.count, true)?)?;
                        ordinal += 2;
                    }
                    for d in 0..width {
                        let word = values.word(first, index * width + d)?;
                        assert_eq!(layout.word(first, index * width + d)?, word);
                        assert!(word.0 >= first + row.gather_offset);
                        assert!(word.1 < width);
                    }
                }
                assert_eq!(ordinal, values.nodes);
                assert_eq!(work, values.work);
                if width == 128 && seq == 11 {
                    assert_eq!(puma::MAX_LANES, 1024);
                    let row = values.rows.last().unwrap();
                    assert_eq!(
                        values.tiles[row.tile_start..row.tile_start + row.tile_count]
                            .iter()
                            .map(|t| t.count)
                            .collect::<Vec<_>>(),
                        vec![93, 35]
                    );
                }
                Ok(())
            },
        )
        .unwrap();
    }
}

#[test]
fn same_native_program_arc_never_merges_invocation_ordinals_or_mass_lineage() {
    source(2, 1, 4, 1, true, |builder, context, _| {
        let graph = at_values(builder, context)?;
        let values = lower(&graph)?;
        let a = values.rows[0].probability.as_ref().unwrap();
        let b = values.rows[1].probability.as_ref().unwrap();
        assert!(
            Arc::ptr_eq(a.program(), b.program()),
            "uniform generated source exercises native template reuse"
        );
        assert_ne!(a.tensor_ordinal(), b.tensor_ordinal());
        assert_ne!(a.gather_ordinal(), b.gather_ordinal());
        assert_eq!(a.native.word(0)?, (a.tensor, 0));
        assert_eq!(b.native.word(0)?, (b.tensor, 0));
        assert!(Arc::ptr_eq(
            &a.predecessor.origin,
            &values.probabilities.origin
        ));
        Ok(())
    })
    .unwrap();
}

#[test]
fn exact_prefix_lookup_rejects_padding_shifted_prefix_and_broken_known_association() {
    source(2, 1, 4, 4, false, |builder, context, _| {
        let graph = at_values(builder, context)?;
        let op = graph.frontier.operations().nth(graph.cursor).unwrap().1;
        let (p, _) = graph.frontier.inputs(op);
        let p = p.unwrap();
        let prefix = PublicationSink::slice(p, 0, 1);
        let rows = graph.softmax_rows_for(p)?.ok_or(Error::Binding)?;
        let evidence = Probability::from_rows(rows, &prefix, 1)?;
        evidence.check()?;
        let padded = PublicationSink::slice(p, 0, 4);
        assert!(Probability::from_rows(rows, &padded, 4).is_err());
        let shifted = PublicationSink::slice(p, 1, 2);
        assert!(Probability::from_rows(rows, &shifted, 1).is_err());
        // No scan/zero trimming and no None-to-certificate conversion.
        assert!(rows.lookup_published(&padded, 1).is_err());
        Ok(())
    })
    .unwrap();
}

#[test]
fn exact_and_one_short_phase_gate_precede_native_probability_clones_and_work() {
    for short in [false, true] {
        source(2, 1, 4, 4, false, |builder, context, _| {
            let mut graph = at_values(builder, context)?;
            let mut attempt = context.attempt();
            let peak = graph.values_peak(8, 8, size_of_val(&attempt))?;
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
fn postdebit_and_publication_failures_burn_attempt_without_inner_refund() {
    for publication in [false, true] {
        let plan = source(2, 1, 4, 4, false, |builder, context, _| {
            let graph = at_values(builder, context)?;
            let before = context.work();
            let held = context.reservation().payload_bytes();
            if publication {
                FAIL_PUBLICATION.with(|f| f.set(true));
            } else {
                FAIL_AFTER_DEBIT.with(|f| f.set(true));
            }
            assert!(matches!(lower(&graph), Err(Error::Allocation)));
            assert!(context.work().operations() > before.operations());
            assert!(context.check_open().is_err());
            assert!(context.reservation().payload_bytes() >= held);
            Ok(Arc::clone(context.plan()))
        })
        .unwrap();
        assert!(plan.budget().used_bytes() < plan.budget().admitted_peak_bytes());
    }
}

#[test]
fn ordinary_product_cannot_be_rescued_by_mass_refinement() {
    // Numeric negative only, NOT a producer, certificate or acceptance path.
    let extreme = Interval::new(-(1i64 << 55), 1i64 << 55).unwrap();
    assert!(product_bounds(
        InputBounds {
            left: extreme,
            right: extreme
        },
        2,
        true,
        true,
        None
    )
    .is_err());
}

#[test]
fn constant_space_selected_span_checks_original_d_outer_k_inner_order() {
    for width in [1, 4, 128] {
        for inner in [1, 3, 11] {
            let mut selected = Span::default();
            for d in 0..width {
                for k in 0..inner {
                    Sink::push(
                        &mut selected,
                        Word {
                            index: 40 + k * width + d,
                            width,
                            valid: true,
                        },
                    );
                }
            }
            let view = Sink::view(selected, &[width, inner]);
            assert!(view.words.valid);
            assert_eq!(view.words.start, 40);
            assert_eq!(view.words.inner, inner);
            let mut broken = Span::default();
            Sink::push(
                &mut broken,
                Word {
                    index: 40,
                    width,
                    valid: true,
                },
            );
            Sink::push(
                &mut broken,
                Word {
                    index: usize::MAX,
                    width,
                    valid: true,
                },
            );
            assert!(!broken.valid);
        }
    }
}

#[test]
fn geometry_refuses_large_inner_without_chunking_or_other_numeric_path() {
    source(1, 1, 4, 4, false, |builder, context, _| {
        let graph = at_values(builder, context)?;
        let op = graph.frontier.operations().nth(graph.cursor).unwrap().1;
        let (p, v) = graph.frontier.inputs(op);
        let mut p = p.unwrap().clone();
        let mut v = v.unwrap().clone();
        p.logical.dimensions[3] = puma::MAX_LANES + 1;
        v.logical.dimensions[2] = puma::MAX_LANES + 1;
        assert!(geometry(&p, &v, &[1, 1, 4, 4], 1).is_err());
        Ok(())
    })
    .unwrap();
}
