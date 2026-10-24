//! Native-owner component tests, NOT a fake semantic Attention publication.
//! Uses the existing genuine config/public-source/token/context fixture. These
//! definitions may populate its tiny generated source only under a later grant.
use super::*;
use crate::replicated_decoder::program::request_rescale_tests::with_context;

fn input(graph: &CompactGraph<'_, '_>) -> PublishedView {
    graph
        .bound_imports
        .iter()
        .find(|i| i.semantic == "input.tokens")
        .expect("genuine token creation")
        .view
        .clone()
}
fn attempts() -> usize {
    FACTORIES.with(|n| n.get())
}

#[test]
fn original_builder_causal_prefixes_native_certificates_and_zero_suffix_match() {
    with_context(|imports, context| {
        let graph = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        let input = input(&graph);
        assert_eq!(input.lanes, 4);
        let before = context.work();
        let rows = prepare(&graph, &input, 2, 0)?;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.first(), graph.next);
        assert_eq!(rows.cursor(), graph.cursor);
        assert_eq!(context.work(), before.checked_add(rows.work())?);
        // Independent normal Builder; source-free oracle is NEVER admission or
        // source authority input to the component under test.
        let mut oracle = Builder {
            public_weights: true,
            ..Builder::default()
        };
        let original = oracle.push(Code::Input("oracle".into()), 4, input.bounds)?;
        let view = oracle.view(original, vec![1, 4]);
        let anchor = oracle.gather(&view.words.slice(..1))?;
        let zero = oracle.constant(anchor, 0)?;
        let first = oracle.nodes.len();
        let output = driver::softmax(&mut oracle, &[1, 4], Some(&view), 2, 0, 2, (zero, 0))?;
        assert_eq!(oracle.nodes.len() - first, 4);
        assert_eq!(output.words.len(), 4);
        for i in 0..2 {
            let row = rows.row(i)?;
            assert_eq!(row.width, i + 1);
            assert_eq!(row.gather, graph.next + 2 * i);
            assert_eq!(row.tensor, graph.next + 2 * i + 1);
            let Code::Tensor { program, .. } = &oracle.nodes[first + 2 * i + 1].code else {
                panic!("original Tensor");
            };
            assert_eq!(row.program().digest(), program.digest());
            assert_eq!(row.program().work_budget(), program.work_budget());
            assert_eq!(row.mass_bound(), oracle.row_masses[i].1);
            assert_eq!(row.output_bounds(), program.output_bounds());
            for j in 0..row.width {
                assert_eq!(row.input_word(j)?, input.word(i * 2 + j)?);
            }
            assert!(rows.lookup(&rows.prefix(i)?, i + 1)?.is_some());
            rows.check_padding(i, &rows.padded(i)?)?;
        }
        assert!(
            rows.lookup(&rows.padded(0)?, 2).is_err(),
            "never trim padded row into certificate"
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn same_program_reuse_retains_distinct_invocation_ordinals_and_debits_both() {
    with_context(|imports, context| {
        let graph = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        let input = input(&graph);
        let before = context.work();
        let n = attempts();
        let rows = prepare(&graph, &input, 1, 1)?;
        assert_eq!(rows.len(), 2);
        assert_eq!(attempts() - n, 1);
        let a = rows.row(0)?;
        let b = rows.row(1)?;
        assert!(Arc::ptr_eq(a.program(), b.program()));
        assert_ne!(a.tensor_ordinal(), b.tensor_ordinal());
        assert!(!Rc::ptr_eq(&rows.native[0], &rows.native[1]));
        assert_eq!(
            context.work(),
            before.checked_add(a.program().work_budget().checked_mul(2)?)?
        );
        assert_eq!(
            rows.lookup(&rows.prefix(1)?, 2)?.unwrap().tensor_ordinal(),
            b.tensor_ordinal()
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn prefix_partial_reorder_padding_and_foreign_same_shape_refuse() {
    with_context(|imports, context| {
        let graph = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        let source = input(&graph);
        let rows = prepare(&graph, &source, 1, 1)?;
        let good = rows.prefix(0)?;
        let mut partial = good.clone();
        partial.lanes = 1;
        assert!(rows.lookup(&partial, 1).is_err());
        let mut reordered = good.clone();
        reordered.reversed = true;
        assert!(rows.lookup(&reordered, 2).is_err());
        let mut offset = good.clone();
        offset.start = 1;
        assert!(rows.lookup(&offset, 2).is_err());
        // A second genuine graph/context loan with equal shape/coordinates does
        // not become the first creation's native row, even when hashes match.
        let other = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        let foreign = prepare(&other, &input(&other), 1, 1)?;
        assert!(rows.lookup(&foreign.prefix(0)?, 2).is_err());
        assert!(rows.check_padding(0, &foreign.padded(0)?).is_err());
        assert!(rows.lookup_published(&input(&other), 2).is_err());
        assert!(
            rows.lookup_published(&source, 4)?.is_none(),
            "actual absence is not a guessed mass"
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn cap_exact_one_short_before_native_factory() {
    let exact = with_context(|imports, context| {
        let graph = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        quote(&graph, &input(&graph), 2, 0)?;
        Ok(graph.construction_peak_bytes())
    })
    .unwrap();
    for cap in [exact, exact - 1] {
        with_context(|imports, context| {
            let mut graph = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
            graph.test_lower_limit(cap);
            let before = attempts();
            let result = prepare(&graph, &input(&graph), 2, 0);
            if cap == exact {
                assert!(result.is_ok());
            } else {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_eq!(attempts(), before);
                assert!(context.check_open().is_err());
            }
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn complete_pending_work_refuses_before_first_factory_no_refund() {
    with_context(|imports, context| {
        let graph = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        let input = input(&graph);
        let bounds = prefix_bounds(&graph, &input, 0, 1)?;
        let mut workspace = tensor::WorkPreflightWorkspace::empty();
        let first = tensor::Construction::SoftmaxCertified { rows: 1, width: 1 }
            .bind(tensor::FactoryParameters::SoftmaxCertified { input: bounds })
            .unwrap()
            .preflight(1, &mut workspace)
            .unwrap()
            .summary()
            .work;
        let second = tensor::Construction::SoftmaxCertified { rows: 1, width: 2 }
            .bind(tensor::FactoryParameters::SoftmaxCertified { input: bounds })
            .unwrap()
            .preflight(1, &mut workspace)
            .unwrap()
            .summary()
            .work;
        let pending = first.checked_add(second)?;
        let operations = puma::MAX_PROGRAM_OPERATIONS
            .checked_sub(pending.operations())
            .unwrap()
            + 1;
        let prior = tensor::Construction::Builder { lanes: 1 }
            .bind(tensor::FactoryParameters::Rescale {
                input: bounds,
                shift: 18,
            })
            .unwrap()
            .preflight(operations, &mut workspace)
            .unwrap();
        context.admit_work(prior)?;
        let before = attempts();
        assert!(matches!(prepare(&graph, &input, 2, 0), Err(Error::Bounds)));
        assert_eq!(attempts(), before);
        assert!(context.check_open().is_err());
        Ok(())
    })
    .unwrap();
}

#[test]
fn native_row_loan_retains_source_context_and_program_until_last_view() {
    with_context(|imports, context| {
        let graph = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        let rows = prepare(&graph, &input(&graph), 2, 0)?;
        let loan = rows.prefix(1)?;
        let weak = Arc::downgrade(rows.row(1)?.program());
        let charge = context.reservation().payload_bytes();
        drop(rows);
        drop(graph);
        assert!(weak.upgrade().is_some());
        assert_eq!(loan.word(0)?.0, loan.owner.tensor);
        drop(loan);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            context.reservation().payload_bytes(),
            charge,
            "only outer owner retires phases"
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn foreign_input_and_private_weight_factory_never_adopt_certified_owner() {
    assert!(require_public_weights(false).is_err());
    with_context(|imports, context| {
        let graph = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        let other = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        let before = attempts();
        assert!(prepare(&graph, &input(&other), 2, 0).is_err());
        assert_eq!(attempts(), before);
        Ok(())
    })
    .unwrap();
    // Actual private route is unavailable from this public source fixture; no
    // changed bool/test issuer is used to manufacture a private source context.
}

#[test]
fn unwind_after_first_native_burns_outer_attempt_without_inner_refund() {
    with_context(|imports, context| {
        let graph = CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        let source = input(&graph);
        let mut expected = context.work();
        let mut workspace = tensor::WorkPreflightWorkspace::empty();
        for (start, width) in [(0, 1), (2, 2)] {
            let b = prefix_bounds(&graph, &source, start, width)?;
            let ticket = tensor::Construction::SoftmaxCertified { rows: 1, width }
                .bind(tensor::FactoryParameters::SoftmaxCertified { input: b })
                .unwrap()
                .preflight(1, &mut workspace)
                .unwrap();
            expected = expected.checked_add(ticket.summary().work)?;
        }
        PANIC_AFTER_NATIVE.with(|v| v.set(true));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prepare(&graph, &source, 2, 0)
        }));
        assert!(result.is_err());
        assert!(context.check_open().is_err());
        assert_eq!(context.work(), expected);
        assert!(context.reservation().payload_bytes() > 0);
        Ok(())
    })
    .unwrap();
}

// Deliberately NO RoPE->Scores->Scale->Softmax->Values "positive" definition:
// A's exact Scale18 ticket + E/R's real Scale/Scores publication, E mapping,
// native-row alias lookup and frozen reader must first exist in one fixture.
