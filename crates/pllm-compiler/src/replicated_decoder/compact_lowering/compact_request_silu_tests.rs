use super::*;
use crate::replicated_decoder::program::request_rescale_tests::with_context;

#[test]
fn actual_route_prior_one_short_refuses_before_factory() {
    with_context(|builder, context| {
        let graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let input = Interval::new(-4 * ONE, 4 * ONE)?;
        let mut workspace = tensor::WorkPreflightWorkspace::empty();
        let pending = tensor::Construction::GuardedSilu { lanes: 128 }
            .bind(tensor::FactoryParameters::GuardedSilu { input })
            .unwrap()
            .preflight_full_tail(
                1,
                tensor::Construction::GuardedSilu { lanes: 1 }
                    .bind(tensor::FactoryParameters::GuardedSilu { input })
                    .unwrap(),
                &mut workspace,
            )
            .unwrap()
            .summary()
            .work;
        let count = puma::MAX_PROGRAM_OPERATIONS
            .checked_sub(pending.operations())
            .unwrap()
            + 1;
        let prior = tensor::Construction::Builder { lanes: 1 }
            .bind(tensor::FactoryParameters::Rescale { input, shift: 18 })
            .unwrap()
            .preflight(count, &mut workspace)
            .unwrap();
        assert_eq!(
            prior.summary().work.operations() + pending.operations(),
            puma::MAX_PROGRAM_OPERATIONS + 1
        );
        context.admit_work(prior)?;
        let before = context.work();
        let attempts = super::super::super::prepared_tensor_map::native_attempts();
        assert!(matches!(
            graph.prepare(MapKind::GuardedSilu, 129, input),
            Err(Error::Bounds)
        ));
        assert_eq!(context.work(), before);
        assert_eq!(
            super::super::super::prepared_tensor_map::native_attempts(),
            attempts
        );
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn actual_route_unwind_after_full_keeps_entire_work_and_charge() {
    with_context(|builder, context| {
        let graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let input = Interval::new(-4 * ONE, 4 * ONE)?;
        let mut workspace = tensor::ActivationWorkWorkspace::empty();
        let full = tensor::Program::activation_guarded_work(
            tensor::Activation::Silu,
            128,
            input,
            &mut workspace,
        )?;
        let tail = tensor::Program::activation_guarded_work(
            tensor::Activation::Silu,
            1,
            input,
            &mut workspace,
        )?;
        let expected = context.work().checked_add(full.checked_add(tail)?)?;
        super::super::super::prepared_tensor_map::observe_native_owners();
        super::super::super::prepared_tensor_map::request_silu::panic_after_full();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            graph.prepare(MapKind::GuardedSilu, 129, input)
        }));
        assert!(result.is_err());
        let weak = super::super::super::prepared_tensor_map::take_native_owners();
        assert_eq!(weak.len(), 1);
        assert!(weak.iter().all(|p| p.upgrade().is_none()));
        assert_eq!(context.work(), expected);
        assert!(context.reservation().payload_bytes() > 0);
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    })
    .unwrap();
}

// Cache/publication bookkeeping fixture ONLY, as in request-rescale tests.
// No execution or source-semantic correctness is inferred from this recipe.
fn publish(graph: &mut CompactGraph<'_, '_>, prepared: MapOwner) {
    let view = graph.bound_imports[0].view.clone();
    assert!(graph.ops.len() < graph.ops.capacity());
    graph.ops.push(PublishedOp {
        first: graph.next,
        input: view.clone(),
        output: view,
        kind: PublishedKind::Silu,
        prepared: Some(prepared),
    });
}

#[test]
fn exact_reuse_keeps_original_full_tail_arcs_and_debits_all_invocations_again() {
    for lanes in [1, 128, 129, 256, 257] {
        with_context(|builder, context| {
            let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
            let input = Interval::new(-4 * ONE, 4 * ONE)?;
            let first = graph.prepare(MapKind::GuardedSilu, lanes, input)?;
            let work = graph.map(&first)?.work_budget()?;
            let full = Arc::clone(graph.map(&first)?.program_at(0)?);
            let tail = if lanes % 128 != 0 && lanes > 128 {
                Some(Arc::clone(
                    graph.map(&first)?.program_at(lanes - lanes % 128)?,
                ))
            } else {
                None
            };
            publish(&mut graph, first);
            let before = context.work();
            let attempts = super::super::super::prepared_tensor_map::native_attempts();
            let reused = graph.prepare(MapKind::GuardedSilu, lanes, input)?;
            assert!(matches!(reused.storage, MapStorage::Previous(0)));
            assert!(Arc::ptr_eq(&full, graph.map(&reused)?.program_at(0)?));
            if let Some(tail) = tail {
                assert!(Arc::ptr_eq(
                    &tail,
                    graph.map(&reused)?.program_at(lanes - lanes % 128)?
                ));
            }
            assert_eq!(context.work(), before.checked_add(work)?);
            assert_eq!(
                super::super::super::prepared_tensor_map::native_attempts(),
                attempts
            );
            let charge = context.reservation().payload_bytes();
            drop(graph);
            assert_eq!(Arc::strong_count(&full), 1);
            assert_eq!(context.reservation().payload_bytes(), charge);
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn graph_inclusive_exact_one_short_boundary_precedes_native_factory() {
    // Observed admission boundary regression, not an independent completeness
    // proof of the underlying AssemblyInventory/native resource contracts.
    let exact = with_context(|builder, context| {
        let graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let _map = graph.prepare(MapKind::GuardedSilu, 257, Interval::new(-4 * ONE, 4 * ONE)?)?;
        Ok(graph.construction_peak_bytes())
    })
    .unwrap();
    for cap in [exact, exact - 1] {
        with_context(|builder, context| {
            let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
            graph.test_lower_limit(cap);
            let attempts = super::super::super::prepared_tensor_map::native_attempts();
            let result =
                graph.prepare(MapKind::GuardedSilu, 257, Interval::new(-4 * ONE, 4 * ONE)?);
            if cap == exact {
                result?;
            } else {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_eq!(
                    super::super::super::prepared_tensor_map::native_attempts(),
                    attempts
                );
                assert!(matches!(context.check_open(), Err(Error::Binding)));
            }
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn reuse_still_checks_current_graph_cap_before_alias_and_work() {
    with_context(|builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let input = Interval::new(-4 * ONE, 4 * ONE)?;
        let first = graph.prepare(MapKind::GuardedSilu, 129, input)?;
        let full = Arc::clone(graph.map(&first)?.program_at(0)?);
        publish(&mut graph, first);
        let before = context.work();
        graph.test_lower_limit(0);
        assert!(matches!(
            graph.prepare(MapKind::GuardedSilu, 129, input),
            Err(Error::Bounds)
        ));
        assert_eq!(context.work(), before);
        assert!(Arc::ptr_eq(&full, graph.kernel(0)?.program_at(0)?));
        Ok(())
    })
    .unwrap();
}

#[test]
fn partial_full_template_collision_refuses_without_rebinding_original_arc() {
    with_context(|builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let input = Interval::new(-4 * ONE, 4 * ONE)?;
        let first = graph.prepare(MapKind::GuardedSilu, 128, input)?;
        let full = Arc::clone(graph.map(&first)?.program_at(0)?);
        publish(&mut graph, first);
        let before = context.work();
        assert!(matches!(
            graph.prepare(MapKind::GuardedSilu, 129, input),
            Err(Error::Binding)
        ));
        assert!(Arc::ptr_eq(&full, graph.kernel(0)?.program_at(0)?));
        assert!(context.work().operations() > before.operations());
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    })
    .unwrap();
}
