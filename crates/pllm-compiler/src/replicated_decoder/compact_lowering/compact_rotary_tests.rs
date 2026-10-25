//! Definitions for the E6 RMS + R8 + rotary merge. No expanded executable is
//! used to mint a source context, row premise or candidate native owner.
use super::*;
use crate::replicated_decoder::program::request_origin::compact_storage::{
    EmissionContext, EmissionScope,
};
use crate::replicated_decoder::program::source_entry::SourceRequestImport;

fn construction(width: usize) -> tensor::Construction {
    tensor::Construction::RotaryHalf {
        rows: 1,
        width,
        cosine_len: width / 2,
        sine_len: width / 2,
    }
}

fn with_source<T>(
    heads: usize,
    width: usize,
    run: impl for<'a> FnOnce(&'a Builder, &'a EmissionContext<'a>, &Manifest) -> Result<T, Error>,
) -> Result<T, Error> {
    with_source_profile(heads, width, false, run)
}

fn with_source_profile<T>(
    heads: usize,
    width: usize,
    signed_qkv: bool,
    run: impl for<'a> FnOnce(&'a Builder, &'a EmissionContext<'a>, &Manifest) -> Result<T, Error>,
) -> Result<T, Error> {
    let (config, _, clear) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    let mut config: serde_json::Value = serde_json::from_slice(&config).unwrap();
    config["hidden_size"] = serde_json::json!(heads * width);
    config["intermediate_size"] = serde_json::json!(heads * width);
    config["num_attention_heads"] = serde_json::json!(heads);
    config["num_key_value_heads"] = serde_json::json!(heads);
    config["head_dim"] = serde_json::json!(width);
    let config = serde_json::to_vec(&config).unwrap();
    let policy = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    let mut publisher =
        Publisher::new_with_policy(&config, 4, 2, [1; 32], [2; 32], WeightMode::Public, policy)
            .unwrap();
    let shapes = publisher
        .shapes()
        .iter()
        .map(|(k, s)| (k.clone(), s.clone()))
        .collect::<Vec<_>>();
    let mut payloads = BTreeMap::new();
    for (key, shape) in shapes {
        let values = &clear[&key];
        let signed_projection = signed_qkv
            && ["q_proj.", "k_proj.", "v_proj."]
                .iter()
                .any(|name| key.contains(name));
        let bytes = (0..shape.iter().product())
            .flat_map(|i| {
                let value = if signed_projection {
                    let magnitude = if key.ends_with(".bias") { 1 } else { ONE / 64 };
                    if i % 2 == 0 {
                        -magnitude
                    } else {
                        magnitude
                    }
                } else {
                    values[i % values.len()]
                };
                (value as f32 / ONE as f32).to_le_bytes()
            })
            .collect::<Vec<_>>();
        for (chunk, frame) in bytes.chunks(4096).enumerate() {
            publisher.append(&key, chunk * 1024, frame).unwrap();
        }
        payloads.insert(key, bytes);
    }
    publisher.seal(&[1, 2, 1, 2]).unwrap();
    let mut manifest: Manifest = serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
    manifest.program = [0; 32];
    let mut importer = SourceRequestImport::new(&serde_json::to_vec(&manifest).unwrap())?;
    for (key, bytes) in payloads {
        for (chunk, frame) in bytes.chunks(4096).enumerate() {
            importer.append(&key, chunk * 1024, frame)?;
        }
    }
    importer.with_created(|created, guard| {
        let mut scope = EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
        let (builder, context, _) = scope.parts();
        run(builder, context, &manifest)
    })
}

fn at_rotary<'a>(
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
        != Op::RotaryEmbedding
    {
        graph.lower_next()?;
    }
    Ok(graph)
}

#[test]
fn genuine_projection_reshape_rotary_matches_original_coordinates_bounds_digest_work() {
    for (heads, width) in [(1, 2), (2, 4), (2, 8), (1, 36)] {
        with_source(heads, width, |builder, context, manifest| {
            let mut graph = at_rotary(builder, context)?;
            let prefix = builder.nodes.len();
            let before = context.work();
            let output = graph.lower_next()?;
            assert_eq!(
                builder.nodes.len(),
                prefix,
                "no expanded entry prerequisite"
            );
            let op = graph.ops.last().unwrap();
            let PublishedKind::Rotary(rows) = &op.kind else {
                panic!("rotary owner");
            };
            assert_eq!(output.logical.dimensions(), &[1, heads, 4, width]);
            assert_eq!(output.order, LaneOrder::Linear);
            assert_eq!(context.work(), before.checked_add(rows.work)?);
            if heads > 1 {
                assert!(matches!(op.input.order, LaneOrder::HeadsSequence { .. }));
            }
            // Independent expanded TEST control, only AFTER real source-driven
            // compact publication. Numeric identity alone grants no authority.
            let oracle = Program::compile_with_norms(
                &manifest.config,
                4,
                2,
                &manifest.tensors,
                WeightMode::Public,
                manifest.execution,
                &manifest.norms,
            )
            .unwrap();
            for (r, row) in rows.native.iter().enumerate() {
                let gather = op.first + 2 * r;
                let original = &oracle.nodes[gather];
                assert_eq!(original.bounds, row.incoming);
                let Code::Gather(words) = &original.code else {
                    panic!("original gather");
                };
                assert_eq!(words.len(), width);
                for j in 0..width {
                    assert_eq!(words.at(j), op.input.word(r * width + j)?);
                    assert_eq!(output.word(r * width + j)?, (gather + 1, j));
                }
                let Code::Tensor { program, input } = &oracle.nodes[gather + 1].code else {
                    panic!("original native row");
                };
                assert_eq!(*input, gather);
                assert_eq!(program.digest(), row.program.digest());
                assert_eq!(program.work_budget(), row.program.work_budget());
                assert_eq!(program.output_bounds(), row.program.output_bounds());
                assert_eq!(row.position, (r % 4) as u32);
            }
            assert!(
                graph.row_norm(&output)?.is_none(),
                "rotary is NEVER GENERAL"
            );
            assert!(graph.complete_row(&output, 0)?.is_none());
            assert!(graph.attention_energy(&output, 1, width)?.is_none());
            assert!(graph.attention_energy(&output, 0, width - 1)?.is_none());
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn genuine_biased_width36_row_crosses_product_tile_with_original_rope_owner() {
    use crate::replicated_decoder::program::schedule_finalization::{
        BorrowedOrderedNodes, BorrowedRecipe,
    };
    with_source_profile(1, 36, true, |builder, context, manifest| {
        let mut graph = at_rotary(builder, context)?;
        let operation = graph.frontier.operations().nth(graph.cursor).unwrap().1;
        let input = graph.frontier.inputs(operation).0.unwrap().clone();
        assert_eq!(input.logical.dimensions(), &[1, 1, 4, 36]);
        let (publication, start) = graph.alias_range(&input, 108, 36)?.ok_or(Error::Binding)?;
        let publication_slot = publication.slot.ok_or(Error::Binding)?;
        assert!(matches!(
            graph.ops[publication_slot].kind,
            PublishedKind::Projection { bias: Some(_), .. }
        ));
        let Mapping::Products(first) = publication.mapping else {
            panic!("actual bias triples");
        };
        assert_eq!(start, 108);
        assert_eq!(input.word(108)?, (first + 2, 108));
        assert_eq!(input.word(128)?, (first + 5, 0));
        let before = context.work();
        let output = graph.lower_next()?;
        let op = graph.ops.last().unwrap();
        let PublishedKind::Rotary(rows) = &op.kind else {
            panic!("rotary");
        };
        assert_eq!(rows.native.len(), 4);
        assert_eq!(context.work(), before.checked_add(rows.work)?);
        let row = &rows.native[3];
        let authority = row.premise.as_ref().expect("genuine attention ancestry");
        assert_eq!(authority.family, EnergyFamily::Attention);
        assert_eq!(authority.producer_slot, publication_slot);
        assert_eq!(authority.start, start);
        assert_eq!(authority.width, 36);
        assert_eq!(authority.first_word, input.word(108)?);
        assert!(row.incoming.lower() < 0, "signed original interval");
        assert!(graph.row_norm(&output)?.is_none(), "never GENERAL");
        let native = Arc::clone(&row.program);
        let ordinal = op.first + 7;
        // Independent expanded control only AFTER source-created publication.
        let original = Program::compile_with_norms(
            &manifest.config,
            4,
            2,
            &manifest.tensors,
            WeightMode::Public,
            manifest.execution,
            &manifest.norms,
        )
        .unwrap();
        let Code::Tensor { program, .. } = &original.nodes[ordinal].code else {
            panic!("original row");
        };
        assert_eq!(native.digest(), program.digest());
        assert_eq!(native.work_budget(), program.work_budget());
        assert_eq!(native.output_bounds(), program.output_bounds());
        let Code::Gather(words) = &original.nodes[ordinal - 1].code else {
            panic!("original gather");
        };
        for j in 0..36 {
            assert_eq!(words.at(j), input.word(108 + j)?);
        }
        let frozen = graph.freeze(vec![output], MAX_PROGRAM_BYTES)?;
        let reader = frozen.borrow_schedule(MAX_PROGRAM_BYTES)?;
        let BorrowedRecipe::Tensor { program, .. } = reader.node(ordinal)?.recipe else {
            panic!("frozen same native");
        };
        assert!(Arc::ptr_eq(program, &native));
        Ok(())
    })
    .unwrap();
}

#[test]
fn cross_tile_rotary_rejects_reordered_and_foreign_publications_before_factory() {
    let foreign = with_source(1, 36, |builder, context, _| {
        let graph = at_rotary(builder, context)?;
        let op = graph.frontier.operations().nth(graph.cursor).unwrap().1;
        Ok(graph.frontier.inputs(op).0.unwrap().clone())
    })
    .unwrap();
    for wrong_order in [true, false] {
        with_source(1, 36, |builder, context, _| {
            let graph = at_rotary(builder, context)?;
            let op = graph
                .frontier
                .operations()
                .nth(graph.cursor)
                .unwrap()
                .1
                .clone();
            let mut input = graph.frontier.inputs(&op).0.unwrap().clone();
            if wrong_order {
                input.order = LaneOrder::HeadsSequence {
                    heads: 2,
                    rows: 2,
                    width: 36,
                    to_heads: true,
                    base_start: 0,
                };
            } else {
                input = foreign.clone();
            }
            let factories = FACTORIES.with(|n| n.get());
            let work = context.work();
            assert!(matches!(
                graph.lower_rotary(&op, &input),
                Err(Error::Binding)
            ));
            assert_eq!(FACTORIES.with(|n| n.get()), factories);
            assert_eq!(context.work(), work);
            assert!(context.check_open().is_err());
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn frozen_reader_and_existing_alias_consumer_keep_original_row_arcs() {
    use crate::replicated_decoder::program::schedule_finalization::{
        BorrowedOrderedNodes, BorrowedRecipe,
    };
    with_source(2, 4, |builder, context, _| {
        let mut graph = at_rotary(builder, context)?;
        let output = graph.lower_next()?;
        let descriptor = graph.ops.len() - 1;
        let op = &graph.ops[descriptor];
        let first = op.first;
        let PublishedKind::Rotary(rows) = &op.kind else {
            panic!("rotary");
        };
        let arcs = rows
            .native
            .iter()
            .map(|r| Arc::clone(&r.program))
            .collect::<Vec<_>>();
        let coordinates = (0..op.input.lanes)
            .map(|i| op.input.word(i))
            .collect::<Result<Vec<_>, _>>()?;
        let alias = output.clone(); // existing exact-view export consumer
        let frozen = graph.freeze(vec![output], MAX_PROGRAM_BYTES)?;
        let reader = frozen.borrow_schedule(MAX_PROGRAM_BYTES)?;
        for (r, arc) in arcs.iter().enumerate() {
            let gather = reader.node(first + 2 * r)?;
            let BorrowedRecipe::Gather(words) = gather.recipe else {
                panic!("borrowed gather");
            };
            for j in 0..4 {
                assert_eq!(words.lane_ref(j)?, coordinates[r * 4 + j]);
            }
            let node = reader.node(first + 2 * r + 1)?;
            let BorrowedRecipe::Tensor { program, input } = node.recipe else {
                panic!("borrowed native");
            };
            assert_eq!(input, first + 2 * r);
            assert!(Arc::ptr_eq(arc, program));
            for j in 0..4 {
                assert_eq!(frozen.word(&alias, r * 4 + j)?, (first + 2 * r + 1, j));
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn genuine_q_and_k_rows_precede_test_only_expanded_control() {
    with_source(2, 4, |builder, context, manifest| {
        let mut graph = at_rotary(builder, context)?;
        let prefix = builder.nodes.len();
        let mut actual = Vec::new();
        while actual.len() < 2 {
            graph.lower_next()?;
            if let Some(op) = graph.ops.last() {
                if let PublishedKind::Rotary(rows) = &op.kind {
                    actual.push((
                        op.first,
                        rows.native
                            .iter()
                            .map(|row| Arc::clone(&row.program))
                            .collect::<Vec<_>>(),
                    ));
                }
            }
        }
        assert_eq!(
            builder.nodes.len(),
            prefix,
            "source input only, no expanded entry"
        );
        assert_ne!(actual[0].0, actual[1].0, "distinct original Q/K consumers");
        let oracle = Program::compile_with_norms(
            &manifest.config,
            4,
            2,
            &manifest.tensors,
            WeightMode::Public,
            manifest.execution,
            &manifest.norms,
        )
        .unwrap();
        for (first, programs) in actual {
            for (row, program) in programs.iter().enumerate() {
                let Code::Tensor {
                    program: original,
                    input,
                } = &oracle.nodes[first + 2 * row + 1].code
                else {
                    panic!("original Q/K native row");
                };
                assert_eq!(*input, first + 2 * row);
                assert_eq!(program.digest(), original.digest());
                assert_eq!(program.work_budget(), original.work_budget());
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn native_positions_coefficients_keep_two_floor_original_program() {
    for width in [2, 4, 128] {
        for position in [0, 1, 511, 512] {
            let input = Interval::new(-ONE, ONE).unwrap();
            let mut c = [0i64; 64];
            let mut s = [0i64; 64];
            tensor::Program::rotary_coefficients_into(
                10000.0,
                position,
                width,
                &mut c[..width / 2],
                &mut s[..width / 2],
            )
            .unwrap();
            let oracle =
                tensor::Program::rotary_half(1, width, input, &c[..width / 2], &s[..width / 2])
                    .unwrap();
            let mut workspace = tensor::WorkPreflightWorkspace::empty();
            let ticket = construction(width)
                .bind(tensor::FactoryParameters::RotaryHalf {
                    input,
                    cosine: &c[..width / 2],
                    sine: &s[..width / 2],
                })
                .unwrap()
                .preflight(1, &mut workspace)
                .unwrap();
            let mut ledger = WorkBudget::default();
            let (part, tail) = tensor::CallerWorkLedger::new(
                &mut ledger,
                puma::MAX_PROGRAM_OPERATIONS,
                puma::prss::MAX_PROGRAM_AES_BLOCKS,
            )
            .unwrap()
            .admit(ticket)
            .unwrap()
            .into_parts();
            assert!(tail.is_none());
            let (program, certificate) = part.build().unwrap().into_parts();
            assert_eq!(certificate, tensor::ProducedCertificate::None);
            assert_eq!(program.digest(), oracle.digest());
            assert_eq!(ledger, oracle.work_budget());
        }
    }
}

#[test]
fn graph_baseline_is_not_new_construction_headroom() {
    with_source(2, 4, |builder, context, _| {
        let mut graph = at_rotary(builder, context)?;
        // Retained baseline fits, but no space is granted for the new row Vec,
        // native templates, coefficient backing or query/build overlap. This is
        // a regression for canonical lower_rotary's missing graph-local gate;
        // not an invented exact one-byte construction quote.
        let baseline = add_bytes(graph.outside_owner_bytes, graph.baseline()?)?;
        graph.test_lower_limit(baseline);
        let before = context.work();
        assert!(matches!(graph.lower_next(), Err(Error::Bounds)));
        assert_eq!(
            context.work(),
            before,
            "refuse before native row debit/factory"
        );
        assert!(context.check_open().is_err());
        Ok(())
    })
    .unwrap();
}

#[test]
fn exact_and_one_short_peak_gate_precedes_first_factory() {
    for short in [false, true] {
        with_source(2, 4, |builder, context, _| {
            let mut graph = at_rotary(builder, context)?;
            let mut attempt = context.attempt();
            let peak = graph.rotary_admission_peak(8, 4, size_of_val(&attempt))?;
            attempt.commit();
            graph.test_lower_limit(peak - usize::from(short));
            let before = FACTORIES.with(|n| n.get());
            let result = graph.lower_next();
            if short {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_eq!(FACTORIES.with(|n| n.get()), before);
                assert!(context.check_open().is_err());
            } else {
                result?;
                assert_eq!(FACTORIES.with(|n| n.get()) - before, 8);
            }
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn public_norm_gate_and_lazy_general_first_preserve_absence_and_errors() -> Result<(), Error> {
    assert_eq!(
        choose_energy(
            false,
            || panic!("no GENERAL lookup without public norms"),
            || panic!("no attention lookup")
        )?,
        None
    );
    assert_eq!(
        choose_energy(
            true,
            || Ok(Some(17)),
            || panic!("must not evaluate fallback")
        )?,
        Some((EnergyFamily::General, 17))
    );
    assert_eq!(
        choose_energy(true, || Ok(None), || Ok(Some(23)))?,
        Some((EnergyFamily::Attention, 23))
    );
    assert_eq!(choose_energy(true, || Ok(None), || Ok(None))?, None);
    assert_eq!(
        choose_energy(
            true,
            || Err(Error::Binding),
            || panic!("error is not absence")
        ),
        Err(Error::Binding)
    );
    assert_eq!(
        choose_energy(true, || Ok(None), || Err(Error::Bounds)),
        Err(Error::Bounds)
    );
    Ok(())
}

#[test]
fn debit_row_and_publication_failures_burn_and_drop_payload_before_refund() {
    for stage in 0..3 {
        let source = with_source(2, 4, |builder, context, _| {
            let mut graph = at_rotary(builder, context)?;
            let before = context.work();
            let factories = FACTORIES.with(|n| n.get());
            let charged = context.reservation().payload_bytes();
            OWNERS.with(|p| p.borrow_mut().clear());
            match stage {
                0 => FAIL_AFTER_DEBIT.with(|f| f.set(true)),
                1 => FAIL_AFTER_ROW.with(|f| f.set(Some(1))),
                _ => FAIL_PUBLICATION.with(|f| f.set(true)),
            }
            assert!(matches!(graph.lower_next(), Err(Error::Allocation)));
            assert!(context.work().operations() > before.operations());
            assert!(context.check_open().is_err());
            let expected = match stage {
                0 => 0,
                1 => 2,
                _ => 8,
            };
            assert_eq!(FACTORIES.with(|n| n.get()) - factories, expected);
            OWNERS.with(|p| {
                assert_eq!(p.borrow().len(), expected);
                assert!(p.borrow().iter().all(|arc| arc.upgrade().is_none()));
            });
            assert!(
                context.reservation().payload_bytes() >= charged,
                "outer phase still held after payload destruction"
            );
            Ok(Arc::clone(context.plan()))
        })
        .unwrap();
        assert!(
            source.budget().used_bytes() < source.budget().admitted_peak_bytes(),
            "scope retired after failed payloads"
        );
    }
}

#[test]
fn only_attention_survives_complete_ordered_output_rows() {
    with_source(2, 4, |builder, context, _| {
        let mut graph = at_rotary(builder, context)?;
        let output = graph.lower_next()?;
        let slot = graph.ops.len() - 1;
        let op = &graph.ops[slot];
        let PublishedKind::Rotary(rows) = &op.kind else {
            panic!("rotary");
        };
        assert!(
            rows.native.iter().any(|row| row.attention_energy.is_some()),
            "genuine source-block premise"
        );
        for (r, row) in rows.native.iter().enumerate() {
            if let Some(bound) = row.attention_energy {
                assert_eq!(graph.attention_energy(&output, r * 4, 4)?, Some(bound));
                let premise = row.premise.as_ref().expect("never retain a bare bound");
                let (producer, start) = graph.alias_range(&op.input, r * 4, 4)?.unwrap();
                assert_eq!(premise.producer_slot, producer.slot.unwrap());
                assert_eq!(premise.start, start);
                assert_eq!(premise.width, 4);
                assert_eq!(premise.first_word, op.input.word(r * 4)?);
                assert!(premise.input_squared > 0);
                assert_eq!(premise.attention_squared, bound);
                assert_eq!(premise.family, EnergyFamily::Attention);
            }
        }
        let mut forged = output.clone();
        forged.start = 1;
        assert!(graph.attention_energy(&forged, 0, 4).is_err());
        assert!(graph.row_norm(&output)?.is_none());
        Ok(())
    })
    .unwrap();
}

#[test]
fn source_rotary_rejects_scaling_pairing_and_missing_theta_without_work() {
    for rejected in ["frequency_scaling", "pairing", "theta"] {
        with_source(2, 4, |builder, context, _| {
            let graph = at_rotary(builder, context)?;
            let mut op = graph
                .frontier
                .operations()
                .nth(graph.cursor)
                .unwrap()
                .1
                .clone();
            let input = graph.frontier.inputs(&op).0.unwrap().clone();
            match rejected {
                "frequency_scaling" => {
                    op.attributes[rejected] = serde_json::json!(2);
                }
                "pairing" => {
                    op.attributes[rejected] = serde_json::json!("adjacent");
                }
                _ => {
                    op.attributes.as_object_mut().unwrap().remove(rejected);
                }
            }
            let before = context.work();
            assert!(matches!(
                graph.lower_rotary(&op, &input),
                Err(Error::Binding)
            ));
            assert_eq!(context.work(), before);
            Ok(())
        })
        .unwrap();
    }
    for (theta, position, width) in [(0.0, 0, 4), (10000.0, 513, 4), (10000.0, 0, 3)] {
        let mut c = [0i64; 2];
        let mut s = [0i64; 2];
        assert!(matches!(
            tensor::Program::rotary_coefficients_into(
                theta,
                position,
                width,
                &mut c[..width / 2],
                &mut s[..width / 2]
            ),
            Err(Error::Bounds)
        ));
    }
}
