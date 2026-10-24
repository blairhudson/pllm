//! Component tests only. Genuine source-input fixture; no fabricated PublishedOp
//! and no claim of published Scores->Scale or complete attention execution.
use super::*;
use crate::replicated_decoder::program::request_rescale_tests::with_context;

fn op() -> pllm_models::ModelOperation {
    pllm_models::ModelOperation {
        id: "component.scale".into(),
        operator: pllm_models::ModelOperator::AttentionScale,
        layer: None,
        state_kind: None,
        inputs: vec!["component.input".into()],
        output_shape: vec![],
        attributes: serde_json::json!({"head_dim":2}),
    }
}
// Literal original Builder composition: neither new scale18 nor the new sink.
fn original(lanes: usize, incoming: Interval, coefficient: i64) -> Result<tensor::Program, Error> {
    let mut b = tensor::Builder::new(lanes, incoming)?;
    let input = b.input();
    let scaled = b.affine(
        input,
        None,
        Affine {
            left: coefficient,
            right: 0,
            constant: 0,
        },
    )?;
    let output = b.rescale(scaled, 18)?;
    b.finish(output)
}
fn entries() -> usize {
    BUILD_ENTRIES.with(std::cell::Cell::get)
}

#[test]
fn zero_coefficient_still_consumes_original_affine_then_rescale() {
    with_context(|_, context| {
        let mut operation = op();
        operation.attributes["head_dim"] = serde_json::json!(u64::MAX);
        let declaration = attention_driver::scale_declaration(&operation)?;
        assert_eq!(declaration.coefficient(), 0);
        let mut attempt = context.attempt();
        context.admit_phase(storage(1, size_of_val(&attempt))?, 0)?;
        let mut workspace = tensor::WorkPreflightWorkspace::empty();
        let incoming = Interval::new(0, 0)?;
        let mut pending = query(context, 0, 1, incoming, &declaration, &mut workspace)?;
        let actual = build(&mut pending)?;
        let expected = original(1, incoming, declaration.coefficient())?;
        assert_eq!(actual.program.digest(), expected.digest());
        assert_eq!(actual.program.instructions(), expected.instructions());
        assert_eq!(actual.program.work_budget(), expected.work_budget());
        assert_eq!(actual.program.validity_lanes(), None);
        attempt.commit();
        Ok(())
    })
    .unwrap();
}

#[test]
fn genuine_imported_primitive_uses_driver_and_original_native_owner() {
    with_context(|builder, context| {
        let graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let input = &graph.bound_imports[0].view;
        let decl = attention_driver::scale_declaration(&op())?;
        let before = context.work();
        let rows = graph.lower_attention_scale(&op(), input, input.logical.dimensions())?;
        assert_eq!(rows.coefficient, decl.coefficient());
        assert_eq!(context.work(), before.checked_add(rows.work)?);
        assert_eq!(
            rows.virtual_nodes()?,
            2 * rows.native.len(),
            "Gather then Tensor, including one chunk"
        );
        let mut expected_start = 0;
        for chunk in &rows.native {
            let expected = original(chunk.lanes, chunk.incoming, decl.coefficient())?;
            assert_eq!(chunk.start, expected_start);
            expected_start += chunk.lanes;
            assert_eq!(chunk.program.digest(), expected.digest());
            assert_eq!(chunk.program.output_bounds(), expected.output_bounds());
            assert_eq!(chunk.program.work_budget(), expected.work_budget());
            assert_eq!(chunk.program.validity_lanes(), None);
        }
        assert_eq!(expected_start, input.lanes);
        let retained = context.reservation().payload_bytes();
        let owner = Arc::clone(&rows.native[0].program);
        let weak = Arc::downgrade(&rows.native[0].program);
        assert!(Arc::ptr_eq(&owner, &weak.upgrade().unwrap()));
        drop(rows);
        assert_eq!(Arc::strong_count(&owner), 1);
        drop(owner);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            context.reservation().payload_bytes(),
            retained,
            "no inner retirement"
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn primitive_numeric_components_all_chunks_debit_before_first_factory() {
    // Query/build COMPONENT only: interval declarations are native numeric
    // parameters, NOT source/score/secret-range certificates or a graph input.
    for lanes in [1usize, 128, 129, 256, 257] {
        with_context(|_, context| {
            let mut attempt = context.attempt();
            context.admit_phase(storage(lanes, size_of_val(&attempt))?, 0)?;
            let decl = attention_driver::scale_declaration(&op())?;
            let mut pending = reserve::<Pending>(lanes.div_ceil(128))?;
            let before = context.work();
            let factories = entries();
            let mut workspace = tensor::WorkPreflightWorkspace::empty();
            let mut expected = WorkBudget::default();
            // Independent original per128 geometry oracle, not the sink driver.
            for start in (0..lanes).step_by(128) {
                let width = (lanes - start).min(128);
                let incoming = Interval::new(-(start as i64) - 1, start as i64 + 1)?;
                let p = original(width, incoming, decl.coefficient())?;
                expected = expected.checked_add(p.work_budget())?;
                pending.push(query(
                    context,
                    start,
                    width,
                    incoming,
                    &decl,
                    &mut workspace,
                )?);
            }
            assert_eq!(
                entries(),
                factories,
                "query never constructs native Program"
            );
            assert_eq!(context.work(), before.checked_add(expected)?);
            for part in &mut pending {
                let native = build(part)?;
                let expected = original(native.lanes, native.incoming, decl.coefficient())?;
                assert_eq!(native.program.digest(), expected.digest());
                assert_eq!(native.program.output_bounds(), expected.output_bounds());
                assert_eq!(native.program.validity_lanes(), None);
            }
            attempt.commit();
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn request_prior_one_short_refuses_before_build_and_burns_existing_attempt() {
    with_context(|builder, context| {
        let graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let input = &graph.bound_imports[0].view;
        let mut ws = tensor::WorkPreflightWorkspace::empty();
        let decl = attention_driver::scale_declaration(&op())?;
        let mut native_workspace = tensor::SoftmaxWorkWorkspace::empty();
        let mut pending_work = WorkBudget::default();
        for chunk in chunks(input, 128) {
            pending_work = pending_work.checked_add(tensor::Program::scale18_work(
                chunk.lanes,
                graph.scale_incoming(&chunk)?,
                decl.coefficient(),
                &mut native_workspace,
            )?)?;
        }
        let prior_count = puma::MAX_PROGRAM_OPERATIONS
            .checked_sub(context.work().operations())
            .unwrap()
            .checked_sub(pending_work.operations())
            .unwrap()
            + 1;
        let prior = tensor::Construction::Builder { lanes: 1 }
            .bind(tensor::FactoryParameters::Rescale {
                input: Interval::new(-1, 1)?,
                shift: 18,
            })
            .unwrap()
            .preflight(prior_count, &mut ws)
            .unwrap();
        context.admit_work(prior)?;
        let work = context.work();
        assert_eq!(
            work.operations() + pending_work.operations(),
            puma::MAX_PROGRAM_OPERATIONS + 1
        );
        let factories = entries();
        assert!(matches!(
            graph.lower_attention_scale(&op(), input, input.logical.dimensions()),
            Err(Error::Bounds)
        ));
        assert_eq!(entries(), factories);
        assert_eq!(context.work(), work);
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn post_debit_failure_burns_retains_charge_and_never_builds() {
    with_context(|builder, context| {
        let graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let input = &graph.bound_imports[0].view;
        let before = context.work();
        let factories = entries();
        FAIL_AFTER_DEBIT.with(|flag| flag.set(true));
        assert!(matches!(
            graph.lower_attention_scale(&op(), input, input.logical.dimensions()),
            Err(Error::Allocation)
        ));
        assert_eq!(entries(), factories);
        assert!(context.work().operations() > before.operations());
        assert!(context.reservation().payload_bytes() > 0);
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn unresolved_published_word_refuses_instead_of_using_global_view_bounds() {
    with_context(|builder, context| {
        let graph = CompactGraph::from_source_request(builder, context, MAX_PROGRAM_BYTES)?;
        let mut input = graph.bound_imports[0].view.clone();
        // Malformed component metadata only, NOT a fabricated publication. No
        // PublishedOp inserted; an unknown original word cannot become authority.
        input.mapping = Mapping::Dense(graph.next);
        let factories = entries();
        assert!(graph
            .lower_attention_scale(&op(), &input, input.logical.dimensions())
            .is_err());
        assert_eq!(entries(), factories);
        Ok(())
    })
    .unwrap();
}
