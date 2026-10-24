use super::*;
use crate::replicated_decoder::program::request_rescale_tests::with_context;

#[test]
fn new_and_previous_keep_exact_full_tail_arc_and_both_work_debits() {
    for lanes in [128, 129, 257] {
        with_context(|builder, context| {
            let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
            let input = Interval::new(-1048576, 1048576)?;
            let first = graph.prepare_request_rescale(lanes, input)?;
            let prepared = graph.map(&first)?;
            let work = prepared.work_budget()?;
            let full = Arc::clone(prepared.program_at(0)?);
            let tail = if lanes % 128 != 0 {
                Some(Arc::clone(prepared.program_at(lanes - lanes % 128)?))
            } else {
                None
            };
            // Publish via the existing graph/container, not a parallel map cache.
            // Test-only recipe uses the original imported view; execution is not
            // attempted, and projection/source correctness is not inferred here.
            let view = graph.bound_imports[0].view.clone();
            assert!(
                graph.ops.len() < graph.ops.capacity(),
                "reuse fixture uses preadmitted storage"
            );
            graph.ops.push(PublishedOp {
                first: graph.next,
                input: view.clone(),
                output: view,
                kind: PublishedKind::Silu,
                prepared: Some(first),
            });
            let before = context.work();
            let attempts = super::super::super::prepared_tensor_map::native_attempts();
            let reused = graph.prepare_request_rescale(lanes, input)?;
            assert!(matches!(reused.storage, MapStorage::Previous(0)));
            let prepared = graph.map(&reused)?;
            assert!(Arc::ptr_eq(&full, prepared.program_at(0)?));
            if let Some(tail) = tail {
                assert!(Arc::ptr_eq(
                    &tail,
                    prepared.program_at(lanes - lanes % 128)?
                ));
            }
            assert_eq!(context.work(), before.checked_add(work)?);
            assert_eq!(
                super::super::super::prepared_tensor_map::native_attempts(),
                attempts
            );
            let charged = context.reservation().payload_bytes();
            drop(graph);
            assert!(Arc::strong_count(&full) == 1);
            assert_eq!(context.reservation().payload_bytes(), charged);
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn graph_exact_and_one_short_gate_include_live_graph_before_factory() {
    let exact = with_context(|builder, context| {
        let graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let _prepared = graph.prepare_request_rescale(257, Interval::new(-1048576, 1048576)?)?;
        Ok(graph.construction_peak_bytes())
    })
    .unwrap();
    for cap in [exact, exact - 1] {
        with_context(|builder, context| {
            let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
            graph.test_lower_limit(cap);
            let before = super::super::super::prepared_tensor_map::native_attempts();
            let result = graph.prepare_request_rescale(257, Interval::new(-1048576, 1048576)?);
            if cap == exact {
                result?;
            } else {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_eq!(
                    super::super::super::prepared_tensor_map::native_attempts(),
                    before
                );
                assert!(matches!(context.check_open(), Err(Error::Binding)));
            }
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn reuse_still_checks_memory_before_alias_or_work_and_keeps_prior_arc() {
    with_context(|builder, context| {
        let mut graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let input = Interval::new(-1048576, 1048576)?;
        let first = graph.prepare_request_rescale(129, input)?;
        let full = Arc::clone(graph.map(&first)?.program_at(0)?);
        let view = graph.bound_imports[0].view.clone();
        assert!(
            graph.ops.len() < graph.ops.capacity(),
            "reuse fixture uses preadmitted storage"
        );
        graph.ops.push(PublishedOp {
            first: graph.next,
            input: view.clone(),
            output: view,
            kind: PublishedKind::Silu,
            prepared: Some(first),
        });
        let before = context.work();
        graph.test_lower_limit(0);
        assert!(matches!(
            graph.prepare_request_rescale(129, input),
            Err(Error::Bounds)
        ));
        assert_eq!(context.work(), before);
        assert!(Arc::ptr_eq(&full, graph.kernel(0)?.program_at(0)?));
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    })
    .unwrap();
}
