//! Definitions only: genuine request-route tests require complete E compact
//! predecessor integration. No fake PublishedOp/cursor or finished Program as
//! constructor authority. Numeric observer test is explicitly composition-only.
use super::*;
use crate::replicated_decoder::program::request_origin::compact_storage::{
    EmissionContext, EmissionScope,
};
use crate::replicated_decoder::program::source_entry::SourceRequestImport;

fn source<T>(
    vocab: usize,
    run: impl for<'a> FnOnce(&'a Builder, &'a EmissionContext<'a>, &Manifest) -> Result<T, Error>,
) -> Result<T, Error> {
    let (importer, manifest) = imported(vocab)?;
    importer.with_created(|created, guard| {
        let mut scope = EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)
            .inspect_err(|e| eprintln!("argmax fixture emission scope: {e:?}"))?;
        let (b, c, _) = scope.parts();
        run(b, c, &manifest)
    })
}
fn imported(vocab: usize) -> Result<(SourceRequestImport, Manifest), Error> {
    assert!(
        matches!(vocab, 4 | 5),
        "existing tiny request-source scope only"
    );
    let (config, _, clear) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    let mut config: serde_json::Value = serde_json::from_slice(&config).unwrap();
    config["vocab_size"] = serde_json::json!(vocab);
    let config = serde_json::to_vec(&config).unwrap();
    let policy = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    let mut publisher =
        Publisher::new_with_policy(&config, 1, 2, [1; 32], [2; 32], WeightMode::Public, policy)
            .unwrap();
    let shapes = publisher
        .shapes()
        .iter()
        .map(|(k, s)| (k.clone(), s.clone()))
        .collect::<Vec<_>>();
    let mut payloads = BTreeMap::new();
    for (key, shape) in shapes {
        let v = &clear[&key];
        let bytes = (0..shape.iter().product())
            .flat_map(|i| (v[i % v.len()] as f32 / ONE as f32).to_le_bytes())
            .collect::<Vec<_>>();
        publisher.append(&key, 0, &bytes).unwrap();
        payloads.insert(key, bytes);
    }
    publisher.seal(&[1]).unwrap();
    let mut manifest: Manifest = serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
    manifest.program = [0; 32];
    let mut importer = SourceRequestImport::new(&serde_json::to_vec(&manifest).unwrap())
        .inspect_err(|e| eprintln!("argmax fixture import: {e:?}"))?;
    for (k, v) in payloads {
        importer.append(&k, 0, &v)?;
    }
    Ok((importer, manifest))
}
fn advance(graph: &mut CompactGraph<'_, '_>) -> Result<(), Error> {
    while graph
        .frontier
        .operations()
        .nth(graph.cursor)
        .ok_or(Error::Binding)?
        .1
        .operator
        != Op::GreedyTokenSelection
    {
        let operator = format!(
            "{:?}",
            graph
                .frontier
                .operations()
                .nth(graph.cursor)
                .ok_or(Error::Binding)?
                .1
                .operator
        );
        graph.lower_next().inspect_err(|e| {
            eprintln!(
                "argmax predecessor cursor {} ({operator}): {e:?}",
                graph.cursor
            )
        })?;
    }
    Ok(())
}
fn input(graph: &CompactGraph<'_, '_>) -> Result<PublishedView, Error> {
    let op = graph
        .frontier
        .operations()
        .nth(graph.cursor)
        .ok_or(Error::Binding)?
        .1;
    Ok(graph.frontier.inputs(op).0.ok_or(Error::Binding)?.clone())
}

#[test]
fn genuine_tiny_four_five_original_tree_bounds_hash_reader_work() {
    for vocab in [4, 5] {
        source(vocab, |builder, context, manifest| {
            let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
            advance(&mut graph)?;
            let input = input(&graph)?;
            let first = graph.next;
            let before = context.work();
            let nodes = builder.nodes.len();
            NATIVE_FACTORIES.with(|n| n.set(0));
            let plan = prepare_request_argmax(&graph, &input, first)?;
            assert_eq!(
                builder.nodes.len(),
                nodes,
                "no expanded constructor prerequisite"
            );
            assert_eq!(context.work(), before.checked_add(plan.work)?);
            let mut observed = WorkBudget::default();
            let mut owners = 0;
            plan.visit(&input, &mut |event| {
                owners += 1;
                if let ArgmaxCode::Tensor { program, .. } = &event.code {
                    observed = observed.checked_add(program.work_budget())?;
                    assert!(plan
                        .programs
                        .iter()
                        .any(|p| std::ptr::eq(&p.program, *program)));
                } else if matches!(event.code, ArgmaxCode::Product { .. }) {
                    observed = observed.checked_add(WorkBudget::product(event.lanes, false)?)?;
                }
                Ok(())
            })?;
            assert_eq!(observed, plan.work);
            assert_eq!(owners, plan.geometry.nodes()?);
            assert_eq!(NATIVE_FACTORIES.with(|n| n.get()), plan.programs.len());
            // Independent TEST-only expanded control AFTER request construction.
            let original = Program::compile_with_norms(
                &manifest.config,
                1,
                2,
                &manifest.tensors,
                WeightMode::Public,
                manifest.execution,
                &manifest.norms,
            )
            .unwrap();
            plan.visit(&input, &mut |event| {
                let node = &original.nodes[event.owner];
                assert_eq!(node.lanes, event.lanes);
                assert_eq!(node.bounds, event.bounds);
                match (&event.code, &node.code) {
                    (ArgmaxCode::PublicIndices, Code::PublicValues(values)) => {
                        assert_eq!(values.len(), event.lanes);
                        for (i, value) in values.iter().enumerate() {
                            assert_eq!(*value, i as i64);
                        }
                    }
                    (ArgmaxCode::Gather(g), Code::Gather(words)) => {
                        for (i, word) in g.words().enumerate() {
                            assert_eq!(word, words.at(i));
                        }
                    }
                    (
                        ArgmaxCode::Tensor { input, program },
                        Code::Tensor {
                            input: other,
                            program: native,
                        },
                    ) => {
                        assert_eq!(input, other);
                        assert_eq!(program.digest(), native.digest());
                    }
                    (
                        ArgmaxCode::Affine {
                            left,
                            right,
                            coefficients,
                        },
                        Code::Affine {
                            left: a,
                            right: b,
                            coefficients: c,
                        },
                    ) => {
                        assert_eq!(left, a);
                        assert_eq!(Some(*right), *b);
                        assert_eq!(coefficients, c);
                    }
                    (
                        ArgmaxCode::Product { left, right },
                        Code::Product {
                            left: a, right: b, ..
                        },
                    ) => assert_eq!((left, right), (a, b)),
                    _ => panic!("original symbolic recipe differs"),
                }
                Ok(())
            })?;
            assert_eq!(
                plan.output,
                plan.geometry
                    .word(&input, first, plan.geometry.rounds, ArgmaxField::Indices, 0)?
            );
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn genuine_second_selection_reuses_original_arcs_but_charges_every_invocation() {
    let (entry, _) = imported(5).unwrap();
    NATIVE_FACTORIES.with(|n| n.set(0));
    let program = entry.compile_compact().unwrap();
    let (first, reused) = program
        .compact_storage()
        .unwrap()
        .test_argmax_reuse()
        .unwrap();
    assert!(first > 0 && reused > 0);
    assert_eq!(
        NATIVE_FACTORIES.with(|n| n.get()),
        first,
        "epoch one reuses actual epoch-zero owners with no second factory"
    );
}

#[test]
fn original_count_observer_composition_full_tail_repetitions_not_authority() {
    // Component metadata ONLY: not a semantic view, producer or request result.
    let mut records = Vec::new();
    records.try_reserve_exact(2).unwrap();
    let mut count = RequestCount {
        requirements: records,
        native: WorkBudget::default(),
        products: WorkBudget::default(),
    };
    let bounds = Interval::new(-20, 20).unwrap();
    let full = tensor::Program::negative_work(128, bounds).unwrap();
    let tail = tensor::Program::negative_work(1, bounds).unwrap();
    count.negative(128, bounds, full).unwrap();
    count.negative(1, bounds, tail).unwrap();
    count.negative(128, bounds, full).unwrap();
    assert_eq!(count.requirements[0].uses, 2);
    assert_eq!(count.requirements[1].uses, 1);
    assert_eq!(
        count.native,
        full.checked_add(tail).unwrap().checked_add(full).unwrap()
    );
    let product = WorkBudget::product(128, false).unwrap();
    count.product(product).unwrap();
    assert_eq!(count.products, product);
    assert!(
        count
            .negative(
                2,
                bounds,
                tensor::Program::negative_work(2, bounds).unwrap()
            )
            .is_err(),
        "bounded requirement capacity"
    );
}

#[test]
fn exact_and_one_short_phase_before_count_vec_and_factory() {
    for short in [false, true] {
        source(5, |builder, context, _| {
            let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
            advance(&mut graph)?;
            let view = input(&graph)?;
            let mut a = context.attempt();
            let (_, bytes) = peak(&graph, view.lanes, size_of_val(&a))?;
            a.commit();
            graph.test_lower_limit(bytes - usize::from(short));
            let before = context.work();
            NATIVE_FACTORIES.with(|n| n.set(0));
            let r = prepare_request_argmax(&graph, &view, graph.next);
            if short {
                assert!(matches!(r, Err(Error::Bounds)));
                assert_eq!(context.work(), before);
                assert_eq!(NATIVE_FACTORIES.with(|n| n.get()), 0);
                assert!(context.check_open().is_err());
            } else {
                r?;
            }
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn unreconciled_prior_component_work_cannot_enter_factory() {
    source(5, |builder, context, _| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        advance(&mut graph)?;
        let view = input(&graph)?;
        // Real component debit; do NOT invent graph recipes or overwrite ledger.
        context.admit_protocol_work(WorkBudget::product(1, false)?)?;
        let prior = context.work();
        NATIVE_FACTORIES.with(|n| n.set(0));
        assert!(prepare_request_argmax(&graph, &view, graph.next).is_err());
        assert_eq!(context.work(), prior);
        assert_eq!(NATIVE_FACTORIES.with(|n| n.get()), 0);
        Ok(())
    })
    .unwrap();
}

#[test]
fn postdebit_error_and_native_panic_burn_without_inner_refund() {
    for mode in 0..3 {
        let plan = source(5, |builder, context, _| {
            let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
            advance(&mut graph)?;
            let view = input(&graph)?;
            let before = context.work();
            let held = context.reservation().payload_bytes();
            NATIVE_FACTORIES.with(|n| n.set(0));
            if mode == 0 {
                FAIL_AFTER_DEBIT.with(|f| f.set(true));
            } else if mode == 1 {
                FAIL_PUBLICATION.with(|f| f.set(true));
            } else {
                PANIC_AFTER_FACTORY.with(|f| f.set(true));
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                prepare_request_argmax(&graph, &view, graph.next)
            }));
            if mode == 2 {
                assert!(result.is_err());
            } else {
                assert!(matches!(result.unwrap(), Err(Error::Allocation)));
            }
            if mode == 0 {
                assert_eq!(NATIVE_FACTORIES.with(|n| n.get()), 0);
            }
            assert!(context.work().operations() > before.operations());
            assert!(context.check_open().is_err());
            assert!(context.reservation().payload_bytes() >= held);
            Ok(Arc::clone(context.plan()))
        })
        .unwrap();
        assert!(plan.budget().used_bytes() < plan.budget().admitted_peak_bytes());
    }
}
