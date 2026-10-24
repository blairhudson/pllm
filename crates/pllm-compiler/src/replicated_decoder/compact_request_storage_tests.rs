//! Real source-create/import/declaration funnel. The expanded compiler below is
//! only an independent post-issuance numeric/recipe oracle, never an entry seal.
use super::super::source_entry::SourceRequestImport;
use super::*;

fn importer(vocabulary: usize) -> (SourceRequestImport, Manifest) {
    let (config, _, clear) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    let mut config: serde_json::Value = serde_json::from_slice(&config).unwrap();
    config["vocab_size"] = serde_json::json!(vocabulary);
    let config = serde_json::to_vec(&config).unwrap();
    let execution = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    let mut publisher = Publisher::new_with_policy(
        &config,
        4,
        2,
        [1; 32],
        [2; 32],
        WeightMode::Public,
        execution,
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
        publisher.append(&key, 0, &bytes).unwrap();
        payloads.insert(key, bytes);
    }
    publisher.seal(&TOKENS).unwrap();
    let mut manifest: Manifest = serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
    manifest.program = [0; 32]; // no precompiled identity authorizes construction
    let mut importer = SourceRequestImport::new(&serde_json::to_vec(&manifest).unwrap()).unwrap();
    for (key, bytes) in payloads {
        importer.append(&key, 0, &bytes).unwrap();
    }
    (importer, manifest)
}

#[test]
fn compact_request_first_instruction_constructs_original_lookup() {
    for vocabulary in [4, 128] {
        let (importer, manifest) = importer(vocabulary);
        importer
            .with_created(|created, guard| {
                let mut scope =
                    compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
                let (builder, context, _) = scope.parts();
                let prefix = builder.nodes.len();
                let mut graph = compact_lowering::CompactGraph::from_source_request(
                    builder,
                    context,
                    MAX_PROGRAM_BYTES,
                )?;
                let output = graph.lower_next()?;
                let census = graph.token_lookup_census().next().unwrap();
                assert_eq!(census.first, prefix + 2);
                assert_eq!(census.geometry, [4, vocabulary, 2]);
                assert_eq!(builder.nodes.len(), prefix, "no expanded request graph");
                let mut work = census
                    .program
                    .work_budget()
                    .checked_mul((vocabulary / census.program.lanes()) as u64)?
                    .checked_mul(4)?;
                if let Some(tail) = census.tail {
                    work = work.checked_add(tail.work_budget().checked_mul(4)?)?;
                }
                assert_eq!(context.work(), work);
                assert_eq!(graph.container_census()?.partial_work, work);
                // Independent expanded control starts ONLY AFTER candidate issuance.
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
                let Code::TensorLoop {
                    program,
                    tail,
                    input,
                } = &oracle.nodes[prefix + 2].code
                else {
                    panic!("original lookup loop")
                };
                assert_eq!(program.digest(), census.program.digest());
                assert_eq!(
                    tail.as_ref().map(|p| p.digest()),
                    census.tail.map(|p| p.digest())
                );
                assert!(matches!(input,LoopInput::TokenOffsets{owner,index:0} if *owner==prefix-1));
                for lane in 0..8 {
                    assert_eq!(
                        output.word(lane)?,
                        (prefix + 2 + (lane / 2) * 3 + 2, lane % 2)
                    );
                }
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn source_chain_temporary_retirement_preserves_original_admitted_peak() {
    let (importer, _) = importer(4);
    let (source, peak) = importer
        .with_created(|created, guard| {
            let mut scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            let (_, context, _) = scope.parts();
            let current = context.plan().budget().used_bytes();
            let before = context.historical_peak();
            assert!(before >= current);
            Ok((Arc::clone(context.plan()), before))
        })
        .unwrap();
    assert!(
        source.budget().used_bytes() < peak,
        "real manifest/import/permit temporaries retired"
    );
    assert_eq!(source.budget().admitted_peak_bytes(), peak);
}

#[test]
fn outer_public_payload_retires_before_its_own_reservation_on_success_and_failure() {
    for fail in [false, true] {
        let (importer, _) = importer(4);
        let mut retained_plan = None;
        let result: Result<(), Error> = importer.with_created(|created, guard| {
            let mut scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            let (_, context, _) = scope.parts();
            retained_plan = Some(Arc::clone(context.plan()));
            if fail {
                Err(Error::Allocation)
            } else {
                Ok(())
            }
        });
        assert_eq!(result.is_err(), fail);
        let (last_public_payload_retired, before_refund, charge) =
            super::super::source_entry::test_public_owner_drop().unwrap();
        assert!(
            last_public_payload_retired,
            "outer source payload dies before its guard"
        );
        assert!(charge > size_of::<OwnedExternalCharge>());
        assert!(
            before_refund >= charge,
            "same-counter reservation still live at payload retirement"
        );
        assert!(retained_plan.unwrap().budget().used_bytes() <= before_refund - charge);
    }
}

#[test]
fn consuming_scope_failure_burns_original_invocation() {
    let (importer, _) = importer(4);
    importer
        .with_created(|created, guard| {
            let mut scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            let (_, context, _) = scope.parts();
            let origin = context.test_origin();
            drop(scope);
            assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
            Ok(())
        })
        .unwrap();
}

#[test]
fn request_lookup_exact_and_one_short_before_factory() {
    let (source_import, _) = importer(128);
    let exact = source_import
        .with_created(|created, guard| {
            let mut scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            let (builder, context, _) = scope.parts();
            let mut graph = compact_lowering::CompactGraph::from_source_request(
                builder,
                context,
                MAX_PROGRAM_BYTES,
            )?;
            graph.lower_next()?;
            Ok(graph.construction_peak_bytes())
        })
        .unwrap();
    for limit in [exact, exact - 1] {
        let (importer, _) = importer(128);
        importer
            .with_created(|created, guard| {
                let mut scope =
                    compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
                let (builder, context, _) = scope.parts();
                let origin = context.test_origin();
                let mut graph = compact_lowering::CompactGraph::from_source_request(
                    builder,
                    context,
                    MAX_PROGRAM_BYTES,
                )?;
                graph.test_lower_limit(limit);
                let before = compact_lowering::test_lookup_attempts();
                let result = graph.lower_next();
                if limit == exact {
                    result?;
                    assert!(compact_lowering::test_lookup_attempts() > before);
                } else {
                    assert!(matches!(result, Err(Error::Bounds)));
                    assert_eq!(compact_lowering::test_lookup_attempts(), before);
                    assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
                    assert!(matches!(
                        compact_lowering::CompactGraph::from_source_request(
                            builder,
                            context,
                            MAX_PROGRAM_BYTES
                        ),
                        Err(Error::Binding)
                    ));
                }
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn request_lookup_factory_and_publication_failure_burn_without_early_refund() {
    for stage in [1, 2] {
        let (importer, _) = importer(128);
        importer
            .with_created(|created, guard| {
                let mut scope =
                    compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
                let (builder, context, _) = scope.parts();
                let origin = context.test_origin();
                let mut graph = compact_lowering::CompactGraph::from_source_request(
                    builder,
                    context,
                    MAX_PROGRAM_BYTES,
                )?;
                let before = context.reservation().payload_bytes();
                let attempts = compact_lowering::test_lookup_attempts();
                compact_lowering::test_lookup_failure(stage);
                assert!(matches!(graph.lower_next(), Err(Error::Allocation)));
                assert_eq!(
                    origin.payload().state.load(Ordering::Acquire),
                    BURNED,
                    "scope still alive"
                );
                assert!(context.work().operations() > 0);
                assert!(compact_lowering::test_lookup_attempts() > attempts);
                let charged = context.reservation().payload_bytes();
                assert!(charged >= before);
                let native = graph
                    .token_lookup_census()
                    .next()
                    .map(|c| Arc::downgrade(c.program));
                assert_eq!(native.is_some(), stage == 2);
                if let Some(native) = &native {
                    assert!(native.upgrade().is_some());
                }
                assert!(matches!(
                    compact_lowering::CompactGraph::from_source_request(
                        builder,
                        context,
                        MAX_PROGRAM_BYTES
                    ),
                    Err(Error::Binding)
                ));
                drop(graph);
                if let Some(native) = native {
                    assert!(native.upgrade().is_none());
                }
                assert_eq!(
                    context.reservation().payload_bytes(),
                    charged,
                    "no refund on payload destruction/failure"
                );
                Ok(())
            })
            .unwrap();
    }
}
