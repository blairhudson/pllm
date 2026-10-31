//! Independent literal original arm compositions, not a second production sink.
use super::*;

fn original_arm(
    b: &mut Builder,
    op: &ModelOperation,
    shape: &[usize],
    input: Option<&View>,
    tokens: &View,
    sources: &BTreeMap<String, SourceView>,
    vocabulary: usize,
) -> Result<View, Error> {
    let source = |name: &str| {
        sources
            .get(
                op.attributes
                    .get(name)
                    .and_then(|v| v.as_str())
                    .ok_or(Error::Binding)?,
            )
            .ok_or(Error::Binding)
    };
    match op.operator {
        Op::Linear | Op::OutputHead => {
            let mut output = b.matrix(
                input.ok_or(Error::Binding)?,
                MatrixView::source(source("weight")?, false)?,
                true,
            )?;
            if op.attributes.get("bias").is_some_and(|v| !v.is_null()) {
                let bias = source("bias")?;
                if bias.shape != [*shape.last().ok_or(Error::Shape)?] {
                    return Err(Error::Shape);
                }
                let repeated: Words = (0..output.words.len())
                    .map(|i| (bias.owner, i % bias.shape[0]))
                    .collect();
                output.words = b.binary(&output.words, &repeated, false)?;
            }
            output.shape = shape.to_vec();
            b.projection_energies(
                input.ok_or(Error::Binding)?,
                &output,
                source("weight")?,
                op.attributes
                    .get("bias")
                    .filter(|v| !v.is_null())
                    .map(|_| source("bias"))
                    .transpose()?,
            )?;
            Ok(output)
        }
        Op::TokenLookup => {
            let table = source("weight")?;
            if table.shape != [vocabulary, *shape.last().ok_or(Error::Shape)?] {
                return Err(Error::Shape);
            }
            if b.public_weights && b.lookup == policy::LookupPolicy::Trie {
                return b.trie_lookup(tokens, table, shape);
            }
            if b.public_weights && b.lookup == policy::LookupPolicy::Tiled {
                return b.tiled_lookup(tokens, table, shape);
            }
            let table_bound = b.nodes[table.owner].bounds;
            let mut words = Words::new();
            for token in tokens.words.iter() {
                let mut one_hot = Words::new();
                if b.public_weights {
                    let original = b.nodes[token.0].bounds;
                    let bounds = Interval::new(
                        original
                            .lower()
                            .checked_sub((vocabulary - 1) as i64)
                            .ok_or(Error::Bounds)?,
                        original.upper(),
                    )?;
                    let equal = b.tensor_loop(
                        LoopInput::TokenOffsets {
                            owner: token.0,
                            index: token.1,
                        },
                        vocabulary,
                        |n| tensor::Program::equal_zero(n, bounds),
                    )?;
                    one_hot = b.view(equal, vec![1, vocabulary]).words;
                } else {
                    for start in (0..vocabulary).step_by(128) {
                        let n = 128.min(vocabulary - start);
                        let x = b.gather(&std::iter::repeat_n(token, n).collect())?;
                        let mut builder = tensor::Builder::new(n, b.nodes[x].bounds)?;
                        let coefficients: Vec<_> = (start..start + n)
                            .map(|i| affine(1, 0, -(i as i64)))
                            .collect();
                        let shifted = builder.affine_lanes(builder.input(), &coefficients)?;
                        let program = builder.finish(shifted)?;
                        let difference = b.tensor(x, program)?;
                        let program = tensor::Program::equal_zero(n, b.nodes[difference].bounds)?;
                        let equal = b.tensor(difference, program)?;
                        one_hot.extend((0..n).map(|i| (equal, i)));
                    }
                }
                let selectors = View {
                    words: one_hot,
                    shape: vec![1, vocabulary],
                };
                let row = b.matrix(&selectors, MatrixView::source(table, true)?, false)?;
                for (id, _) in row.words.iter() {
                    b.nodes[id].bounds = table_bound;
                }
                words.extend(row.words);
            }
            Ok(View {
                words,
                shape: shape.to_vec(),
            })
        }
        _ => unreachable!(),
    }
}

fn old_graph(
    b: &mut Builder,
    graph: &DecoderGraph,
    tokens: &View,
    hidden: &View,
    sources: &BTreeMap<String, SourceView>,
    vocabulary: usize,
) -> Result<View, Error> {
    let mut values = BTreeMap::new();
    if !graph.state_inputs.is_empty() {
        values.insert("hidden".to_owned(), hidden.clone());
    }
    values.insert("input.tokens".to_owned(), tokens.clone());
    let anchor = b.gather(&tokens.words.slice(..1))?;
    b.constant(anchor, 0)?;
    for op in &graph.operations {
        b.admission_stage = "operator_lowering";
        let shape: Vec<_> = op.output_shape.iter().map(|v| *v as usize).collect();
        let a = op.inputs.first().and_then(|id| values.get(id));
        let c = op.inputs.get(1).and_then(|id| values.get(id));
        for view in [a, c].into_iter().flatten() {
            b.bounds(&view.words)?;
        }
        let output = original_arm(b, op, &shape, a, tokens, sources, vocabulary)?;
        if output.words.len() != shape.iter().product::<usize>() {
            return Err(Error::Shape);
        }
        values.insert(op.id.clone(), output);
    }
    values.remove(&graph.output).ok_or(Error::Binding)
}

fn fixture(
    public: bool,
    rows: usize,
    inner: usize,
    columns: usize,
    vocabulary: usize,
) -> (Builder, View, View, BTreeMap<String, SourceView>) {
    let mut b = Builder {
        public_weights: public,
        token_source_binding: [7; 32],
        ..Builder::default()
    };
    let t = b
        .push(
            Code::Input("input.tokens".into()),
            rows,
            Interval::new(0, vocabulary as i64 - 1).unwrap(),
        )
        .unwrap();
    let tokens = b.view(t, vec![1, rows]);
    let h = b
        .push(
            Code::Input("hidden".into()),
            rows * inner,
            Interval::new(-ONE / 16, ONE / 16).unwrap(),
        )
        .unwrap();
    let hidden = b.view(h, vec![1, rows, inner]);
    let mut sources = BTreeMap::new();
    for (key, shape) in [
        ("weight", vec![columns, inner]),
        ("bias", vec![columns]),
        ("embedding", vec![vocabulary, inner]),
    ] {
        let code = if public {
            Code::PublicInput {
                key: key.into(),
                digest: [key.len() as u8; 32],
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            }
        } else {
            Code::Input(key.into())
        };
        let bounds = if key == "bias" {
            Interval::new(1, 1).unwrap()
        } else {
            Interval::new(-1, 1).unwrap()
        };
        let id = b.push(code, shape.iter().product(), bounds).unwrap();
        sources.insert(key.into(), SourceView { owner: id, shape });
    }
    (b, tokens, hidden, sources)
}

fn semantics(op: Op, rows: usize, columns: usize, bias: bool) -> DecoderGraph {
    DecoderGraph {
        mode: DecoderMode::Prefill,
        batch: 1,
        query_sequence: rows as u64,
        maximum_key_sequence: rows as u64,
        operations: vec![ModelOperation {
            id: "projection".into(),
            operator: op,
            layer: Some(7),
            state_kind: None,
            inputs: vec!["hidden".into(), "input.tokens".into()],
            output_shape: vec![1, rows as u64, columns as u64],
            attributes: serde_json::json!({"weight": if op == Op::TokenLookup { "embedding" } else { "weight" }, "bias": if bias { serde_json::json!("bias") } else { serde_json::Value::Null } }),
        }],
        state_inputs: vec![StateTensor {
            id: "hidden".into(),
            layer: Some(7),
            kind: StateKind::Key,
            shape: vec![1],
            maximum_sequence: rows as u64,
        }],
        state_outputs: vec![],
        output: "projection".into(),
    }
}

fn actual_graph(
    b: &mut Builder,
    graph: &DecoderGraph,
    tokens: &View,
    hidden: &View,
    sources: &BTreeMap<String, SourceView>,
    vocabulary: usize,
    site: &mut AdmissionSite,
) -> Result<View, Error> {
    let mut state = BTreeMap::from([((7, StateKind::Key), hidden.clone())]);
    b.graph(
        graph,
        tokens,
        sources,
        &BTreeMap::new(),
        vocabulary,
        0,
        &mut state,
        site,
    )
}

fn same_result(a: Result<View, Error>, e: Result<View, Error>) {
    match (a, e) {
        (Ok(a), Ok(e)) => {
            assert_eq!(a.words, e.words);
            assert_eq!(a.shape, e.shape);
        }
        (Err(a), Err(e)) => assert_eq!(a, e),
        _ => panic!("result winner changed"),
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<String>() {
        message
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        message
    } else {
        panic!("unexpected non-string panic payload")
    }
}

fn same_contraction_state(a: &Builder, e: &Builder) {
    assert_same_schedule(a, e);
    assert_eq!(a.admission_stage, e.admission_stage);
    assert_eq!(a.admission_limit, e.admission_limit);
    assert_eq!(a.requested_bytes, e.requested_bytes);
    assert_eq!(a.requested_lanes, e.requested_lanes);
}

fn projection_proofs(
    builder: &mut Builder,
    input: &View,
    sources: &BTreeMap<String, SourceView>,
    general: bool,
) {
    // Test-private setup: complete scalar bounds and coefficient scans. No new
    // public source-import authority; ledger placement deliberately varies.
    for row in input.words.chunks_exact(3) {
        let energy = builder.scalar_row_energy(&row).unwrap();
        if general {
            builder.record_row_energy(&row, energy).unwrap();
        } else {
            builder.record_attention_energy(&row, energy).unwrap();
        }
    }
    for (key, columns) in [("weight", 3), ("bias", 1)] {
        let mut scan = source_norms::BlockScan::new(3, columns, 2).unwrap();
        scan.append(0, &vec![1; 3 * columns]).unwrap();
        builder
            .public_norms
            .insert(sources[key].owner, scan.finish().unwrap());
    }
}

#[test]
fn original_projection_heads_biases_shapes_and_ordered_inputs_match() {
    for public in [false, true] {
        for (rows, inner, columns) in [(1, 1, 1), (2, 3, 5), (2, 65, 129)] {
            for op in [Op::Linear, Op::OutputHead] {
                for bias in [false, true] {
                    let graph = semantics(op, rows, columns, bias);
                    let (mut a, t, h, sources) = fixture(public, rows, inner, columns, 3);
                    let (mut e, et, eh, es) = fixture(public, rows, inner, columns, 3);
                    let mut site = AdmissionSite::default();
                    let av = actual_graph(&mut a, &graph, &t, &h, &sources, 3, &mut site);
                    let ev = old_graph(&mut e, &graph, &et, &eh, &es, 3);
                    assert!(av.is_ok());
                    same_result(av, ev);
                    same_contraction_state(&a, &e);
                    assert_eq!(site.inputs, [Some((-ONE / 16, ONE / 16)), Some((0, 2))]);
                }
            }
        }
    }
}

#[test]
fn original_embedding_tails_tied_head_and_original_native_arcs_match() {
    for public in [false, true] {
        for vocabulary in [1, 3, 129] {
            for lookup in [
                policy::LookupPolicy::PerToken,
                policy::LookupPolicy::Tiled,
                policy::LookupPolicy::Trie,
            ] {
                let (mut a, t, h, sources) = fixture(public, 2, 3, vocabulary, vocabulary);
                let (mut e, et, eh, es) = fixture(public, 2, 3, vocabulary, vocabulary);
                a.lookup = lookup;
                e.lookup = lookup;
                if public && lookup == policy::LookupPolicy::Trie {
                    a.register_initial_tokens(&t.words, vocabulary).unwrap();
                    e.register_initial_tokens(&et.words, vocabulary).unwrap();
                }
                let mut graph = semantics(Op::TokenLookup, 2, 3, false);
                let mut head = semantics(Op::OutputHead, 2, vocabulary, false)
                    .operations
                    .remove(0);
                head.id = "head".into();
                head.inputs = vec!["projection".into()];
                head.attributes["weight"] = "embedding".into();
                graph.operations.push(head);
                graph.output = "head".into();
                let av = actual_graph(
                    &mut a,
                    &graph,
                    &t,
                    &h,
                    &sources,
                    vocabulary,
                    &mut AdmissionSite::default(),
                );
                let ev = old_graph(&mut e, &graph, &et, &eh, &es, vocabulary);
                let original_success = ev.is_ok();
                if public && lookup == policy::LookupPolicy::Trie && vocabulary == 1 {
                    assert!(av.is_err()); // original Trie requires vocabulary >= 2
                } else {
                    assert!(av.is_ok());
                }
                same_result(av, ev);
                same_contraction_state(&a, &e);
                if !original_success {
                    continue;
                }
                let previous = a.tensors.clone();
                actual_graph(
                    &mut a,
                    &graph,
                    &t,
                    &h,
                    &sources,
                    vocabulary,
                    &mut AdmissionSite::default(),
                )
                .unwrap();
                for (digest, program) in previous {
                    assert!(Arc::ptr_eq(&program, &a.tensors[&digest]));
                }
            }
        }
    }
}

#[test]
fn original_contraction_binding_shape_panic_and_late_bias_winners_match() {
    for fault in [
        "input_first",
        "ordered_input_shape",
        "weight_rank",
        "bias_missing",
        "bias_shape",
        "bias_zero_panic",
        "empty_declared",
        "lookup_shape",
        "lookup_owner_panic",
    ] {
        let op = if fault.starts_with("lookup") {
            Op::TokenLookup
        } else {
            Op::Linear
        };
        let mut graph = semantics(op, 2, 3, true);
        let (mut a, t, h, mut sources) = fixture(true, 2, 3, 3, 3);
        let (mut e, et, eh, mut es) = fixture(true, 2, 3, 3, 3);
        match fault {
            "input_first" => {
                graph.state_inputs.clear();
                sources.remove("weight");
                es.remove("weight");
            }
            "ordered_input_shape" => graph.operations[0].inputs.swap(0, 1),
            "weight_rank" => {
                sources.get_mut("weight").unwrap().shape = vec![9];
                es.get_mut("weight").unwrap().shape = vec![9];
            }
            "bias_missing" => {
                sources.remove("bias");
                es.remove("bias");
            }
            "bias_shape" => {
                sources.get_mut("bias").unwrap().shape = vec![4];
                es.get_mut("bias").unwrap().shape = vec![4];
            }
            "bias_zero_panic" => {
                graph.operations[0].output_shape[2] = 0;
                sources.get_mut("bias").unwrap().shape = vec![0];
                es.get_mut("bias").unwrap().shape = vec![0];
            }
            "empty_declared" => graph.operations[0].output_shape.clear(),
            "lookup_shape" => {
                sources.get_mut("embedding").unwrap().shape = vec![3, 4];
                es.get_mut("embedding").unwrap().shape = vec![3, 4];
            }
            "lookup_owner_panic" => {
                sources.get_mut("embedding").unwrap().owner = usize::MAX;
                es.get_mut("embedding").unwrap().owner = usize::MAX;
            }
            _ => unreachable!(),
        }
        let before = a.nodes.len();
        let av = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            actual_graph(
                &mut a,
                &graph,
                &t,
                &h,
                &sources,
                3,
                &mut AdmissionSite::default(),
            )
        }));
        let ev = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            old_graph(&mut e, &graph, &et, &eh, &es, 3)
        }));
        match (av, ev) {
            (Ok(a), Ok(e)) => same_result(a, e),
            (Err(a), Err(e)) => {
                assert!(matches!(fault, "lookup_owner_panic" | "bias_zero_panic"));
                assert_eq!(a.is::<String>(), e.is::<String>());
                assert_eq!(a.is::<&'static str>(), e.is::<&'static str>());
                assert_eq!(panic_message(a.as_ref()), panic_message(e.as_ref()));
            }
            _ => panic!("panic/error precedence changed"),
        }
        same_contraction_state(&a, &e);
        if matches!(fault, "bias_missing" | "bias_shape" | "empty_declared") {
            assert!(a.nodes.len() > before + 1);
            assert!(!a.tensors.is_empty());
        }
    }
}

#[test]
fn original_contraction_full_byte_cap_stops_before_factory_and_retains_diagnostics() {
    let graph = semantics(Op::Linear, 2, 3, true);
    let (mut quote, qt, _, _) = fixture(true, 2, 3, 3, 3);
    let before = quote.allocated;
    let anchor = quote.gather(&qt.words.slice(..1)).unwrap();
    quote.constant(anchor, 0).unwrap();
    let prelude = quote.allocated - before;
    let (mut a, t, h, sources) = fixture(true, 2, 3, 3, 3);
    let (mut e, et, eh, es) = fixture(true, 2, 3, 3, 3);
    a.allocated = MAX_PROGRAM_BYTES - prelude;
    e.allocated = a.allocated;
    same_result(
        actual_graph(
            &mut a,
            &graph,
            &t,
            &h,
            &sources,
            3,
            &mut AdmissionSite::default(),
        ),
        old_graph(&mut e, &graph, &et, &eh, &es, 3),
    );
    same_contraction_state(&a, &e);
    assert!(a.tensors.is_empty());
    assert_eq!(a.admission_stage, e.admission_stage);
    let control = driver::contraction_control::<Builder>(&t).unwrap();
    assert!(control.projection_bytes > 0 && control.lookup_bytes > 0);
    assert_eq!(control.sink_owned_heap_bytes, None);
    assert_eq!(control.compiled_callframe_bytes, None);
}

#[test]
fn contraction_layout_needs_no_iterator_view_sink_or_hook() {
    fn trap<T>() -> T {
        panic!("type-only quote must not construct or invoke sink hooks")
    }
    struct TrapTokens;
    impl Iterator for TrapTokens {
        type Item = (usize, usize);
        fn next(&mut self) -> Option<Self::Item> {
            panic!("type-only quote must not advance an iterator")
        }
    }
    struct TrapSink;
    impl driver::ContractionSink for TrapSink {
        type View = View;
        type Source = SourceView;
        type Words = Words;
        type Token = (usize, usize);

        fn source_shape(_: &SourceView) -> &[usize] {
            trap()
        }
        fn public_weights(&self) -> bool {
            trap()
        }
        fn lookup_policy(&self) -> policy::LookupPolicy {
            trap()
        }
        fn tokens(_: &View) -> impl Iterator<Item = Self::Token> + '_ {
            trap::<TrapTokens>()
        }
        fn table_bounds(&self, _: &SourceView) -> Interval {
            trap()
        }
        fn trie(&mut self, _: &View, _: &SourceView, _: &[usize]) -> Result<View, Error> {
            trap()
        }
        fn tiled(&mut self, _: &View, _: &SourceView, _: &[usize]) -> Result<View, Error> {
            trap()
        }
        fn public_selectors(&mut self, _: Self::Token, _: usize) -> Result<Words, Error> {
            trap()
        }
        fn private_selectors(
            &mut self,
            _: &mut Words,
            _: Self::Token,
            _: usize,
            _: usize,
        ) -> Result<(), Error> {
            trap()
        }
        fn view(_: Words, _: &[usize]) -> View {
            trap()
        }
        fn selector_view(_: Words, _: usize) -> View {
            trap()
        }
        fn matrix(&mut self, _: &View, _: &SourceView, _: bool, _: bool) -> Result<View, Error> {
            trap()
        }
        fn restore_table_bounds(&mut self, _: &View, _: Interval) {
            trap()
        }
        fn append_row(_: &mut Words, _: View) {
            trap()
        }
        fn len(_: &View) -> usize {
            trap()
        }
        fn bias(
            &mut self,
            _: &mut View,
            _: &SourceView,
            _: driver::BiasIndices,
        ) -> Result<(), Error> {
            trap()
        }
        fn set_shape(_: &mut View, _: &[usize]) {
            trap()
        }
        fn projection_energies(
            &mut self,
            _: &View,
            _: &View,
            _: &SourceView,
            _: Option<&SourceView>,
        ) -> Result<(), Error> {
            trap()
        }
    }

    // No value of any of these types is constructed. The explicit pairing is
    // test-only; it supplies no production admission authority.
    let quote = driver::contraction_control_layout::<TrapSink, TrapTokens>().unwrap();
    assert!(quote.projection_bytes > 0 && quote.lookup_bytes > 0);
    assert_eq!(quote.sink_owned_heap_bytes, None);
    assert_eq!(quote.compiled_callframe_bytes, None);
}

#[test]
fn builder_contraction_layout_matches_actual_iterator_quote() {
    fn layout<I: Iterator<Item = (usize, usize)>>(_: &I) -> driver::ContractionControl {
        driver::contraction_control_layout::<Builder, I>().unwrap()
    }
    let (_, tokens, _, _) = fixture(true, 2, 3, 3, 3);
    let iterator = <Builder as driver::ContractionSink>::tokens(&tokens);
    let expected = layout(&iterator);
    let actual = driver::contraction_control::<Builder>(&tokens).unwrap();
    assert_eq!(actual.projection_bytes, expected.projection_bytes);
    assert_eq!(actual.lookup_bytes, expected.lookup_bytes);
    assert_eq!(actual.sink_owned_heap_bytes, expected.sink_owned_heap_bytes);
    assert_eq!(
        actual.compiled_callframe_bytes,
        expected.compiled_callframe_bytes
    );
}

#[test]
fn contraction_layout_prices_iterator_storage_only_in_lookup() {
    use std::mem::{align_of, size_of};

    struct SizedTokens<const N: usize> {
        _storage: [u8; N],
    }
    impl<const N: usize> Iterator for SizedTokens<N> {
        type Item = (usize, usize);

        fn next(&mut self) -> Option<Self::Item> {
            panic!("type-only quote must not advance an iterator")
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            panic!("type-only quote must not inspect an iterator")
        }
    }

    assert_eq!(size_of::<SizedTokens<0>>(), 0);
    assert_eq!(size_of::<SizedTokens<1>>(), 1);
    assert_eq!(size_of::<SizedTokens<17>>(), 17);
    assert_eq!(align_of::<SizedTokens<0>>(), 1);
    assert_eq!(align_of::<SizedTokens<1>>(), 1);
    assert_eq!(align_of::<SizedTokens<17>>(), 1);

    // No iterator value is constructed. Synthetic pairings exercise accounting
    // sensitivity only, not compatibility with Builder's actual iterator.
    let zero = driver::contraction_control_layout::<Builder, SizedTokens<0>>().unwrap();
    let one = driver::contraction_control_layout::<Builder, SizedTokens<1>>().unwrap();
    let seventeen = driver::contraction_control_layout::<Builder, SizedTokens<17>>().unwrap();
    assert_eq!(one.projection_bytes, zero.projection_bytes);
    assert_eq!(seventeen.projection_bytes, zero.projection_bytes);
    // Independent literal oracle: two iterator slots in lookup, none in projection.
    assert_eq!(one.lookup_bytes - zero.lookup_bytes, 2);
    assert_eq!(seventeen.lookup_bytes - zero.lookup_bytes, 34);
    for quote in [zero, one, seventeen] {
        assert_eq!(quote.sink_owned_heap_bytes, None);
        assert_eq!(quote.compiled_callframe_bytes, None);
    }
}

#[test]
fn original_projection_proof_growth_and_late_proof_failure_keep_published_owners() {
    let graph = semantics(Op::OutputHead, 2, 3, true);
    for missing_bias_proof in [false, true] {
        let (mut a, t, h, sources) = fixture(true, 2, 3, 3, 3);
        let (mut e, et, eh, es) = fixture(true, 2, 3, 3, 3);
        // Test-private ledger setup derived from scalar bounds and a complete
        // coefficient scan. Not a new public source-import/trust boundary.
        for (builder, input, source) in [(&mut a, &h, &sources), (&mut e, &eh, &es)] {
            for row in input.words.chunks_exact(3) {
                let energy = builder.scalar_row_energy(&row).unwrap();
                builder.record_row_energy(&row, energy).unwrap();
            }
            for (key, columns) in [("weight", 3), ("bias", 1)] {
                if key == "bias" && missing_bias_proof {
                    continue;
                }
                let mut scan = source_norms::BlockScan::new(3, columns, 2).unwrap();
                scan.append(0, &vec![1; 3 * columns]).unwrap();
                builder
                    .public_norms
                    .insert(source[key].owner, scan.finish().unwrap());
            }
        }
        let av = actual_graph(
            &mut a,
            &graph,
            &t,
            &h,
            &sources,
            3,
            &mut AdmissionSite::default(),
        );
        let ev = old_graph(&mut e, &graph, &et, &eh, &es, 3);
        if missing_bias_proof {
            assert!(matches!(&av, Err(Error::Binding)));
        } else {
            assert!(av.is_ok());
            assert_eq!(a.attention_energies.len(), 4);
        }
        same_result(av, ev);
        same_contraction_state(&a, &e);
        assert!(!a.tensors.is_empty()); // late proof failure retains original natives
    }
}

#[test]
fn original_rank_two_and_declared_rank_four_contractions_remain_unrepaired() {
    for op in [Op::Linear, Op::OutputHead, Op::TokenLookup] {
        for output_rank in [2, 4] {
            let mut graph = semantics(op, 2, 3, false);
            graph.operations[0].output_shape = if output_rank == 2 {
                vec![2, 3]
            } else {
                vec![1, 1, 2, 3]
            };
            let (mut a, t, mut h, sources) = fixture(true, 2, 3, 3, 3);
            let (mut e, et, mut eh, es) = fixture(true, 2, 3, 3, 3);
            h.shape = vec![2, 3];
            eh.shape = vec![2, 3];
            let av = actual_graph(
                &mut a,
                &graph,
                &t,
                &h,
                &sources,
                3,
                &mut AdmissionSite::default(),
            );
            let ev = old_graph(&mut e, &graph, &et, &eh, &es, 3);
            assert!(av.is_ok());
            same_result(av, ev);
            same_contraction_state(&a, &e);
        }
    }
}

#[test]
fn original_public_raw_publication_survives_later_native_template_budget_failure() {
    let graph = semantics(Op::OutputHead, 2, 3, false);
    let (mut quote, qt, qh, qs) = fixture(true, 2, 3, 3, 3);
    let before = quote.allocated;
    let anchor = quote.gather(&qt.words.slice(..1)).unwrap();
    quote.constant(anchor, 0).unwrap();
    let input = quote.gather(&qh.words).unwrap();
    // Independent original raw-owner storage quote, NOT a numeric certificate.
    // Bounds affect neither this Code's metadata nor its six-lane backing cost.
    quote
        .push(
            Code::PublicProduct {
                input,
                weight: qs["weight"].owner,
                rows: 2,
                inner: 3,
                columns: 3,
                transposed: false,
            },
            6,
            Interval::new(-1, 1).unwrap(),
        )
        .unwrap();
    let prefix = quote.allocated - before;
    let (mut a, t, h, sources) = fixture(true, 2, 3, 3, 3);
    let (mut e, et, eh, es) = fixture(true, 2, 3, 3, 3);
    a.allocated = MAX_PROGRAM_BYTES - prefix;
    e.allocated = a.allocated;
    let av = actual_graph(
        &mut a,
        &graph,
        &t,
        &h,
        &sources,
        3,
        &mut AdmissionSite::default(),
    );
    let ev = old_graph(&mut e, &graph, &et, &eh, &es, 3);
    assert!(matches!(&av, Err(Error::Bounds)));
    same_result(av, ev);
    same_contraction_state(&a, &e);
    assert!(matches!(
        a.nodes.last().unwrap().code,
        Code::PublicProduct { .. }
    ));
    assert!(a.tensors.is_empty()); // native template was not published/interned
    assert_eq!(a.allocated, MAX_PROGRAM_BYTES);
}

#[test]
fn original_numeric_rescale_factory_rejects_after_raw_publication_without_budget_pressure() {
    let graph = semantics(Op::Linear, 2, 3, false);
    let (mut a, t, h, sources) = fixture(true, 2, 3, 3, 3);
    let (mut e, et, eh, es) = fixture(true, 2, 3, 3, 3);
    let radius = 1_i64 << 61;
    for (builder, hidden) in [(&mut a, &h), (&mut e, &eh)] {
        let owner = hidden.words.at(0).0;
        builder.nodes[owner].bounds = Interval::new(-radius, radius).unwrap();
    }
    // Scalar product fits quarter-ring; the original subsequent affine sum
    // fits i64 but exceeds rescale's quarter-ring input. No repaired bound.
    let raw_bounds = Interval::new(-3 * radius, 3 * radius).unwrap();
    let av = actual_graph(
        &mut a,
        &graph,
        &t,
        &h,
        &sources,
        3,
        &mut AdmissionSite::default(),
    );
    let ev = old_graph(&mut e, &graph, &et, &eh, &es, 3);
    assert!(matches!(&av, Err(Error::Bounds)));
    same_result(av, ev);
    same_contraction_state(&a, &e);
    let raw = a.nodes.last().unwrap();
    assert!(matches!(raw.code, Code::PublicProduct { .. }));
    assert_eq!(raw.bounds, raw_bounds);
    assert!(a.tensors.is_empty());
    assert!(a.allocated < MAX_PROGRAM_BYTES / 2);
    assert_eq!(a.admission_limit, None);
    // Independently identify the real failing numeric constructor, not a
    // publication-capacity error made to look like a factory rejection.
    let mut native = tensor::Builder::new(6, raw_bounds).unwrap();
    let input = native.input();
    assert!(matches!(native.rescale(input, 18), Err(Error::Bounds)));
}

#[test]
fn original_projection_certificate_bytes_accept_exact_and_reject_one_short_after_three_entries() {
    let graph = semantics(Op::OutputHead, 2, 3, true);
    let (mut quote, qt, qh, qs) = fixture(true, 2, 3, 3, 3);
    projection_proofs(&mut quote, &qh, &qs, true);
    let before = quote.allocated;
    old_graph(&mut quote, &graph, &qt, &qh, &qs, 3).unwrap();
    let total = quote.allocated - before;
    assert_eq!(quote.attention_energies.len(), 4);
    let certificate_bytes: usize = quote
        .attention_energies
        .iter()
        .map(|(words, _)| words.metadata_bytes() + std::mem::size_of::<(Words, u128)>())
        .sum();
    let last_bytes = quote.attention_energies.last().unwrap().0.metadata_bytes()
        + std::mem::size_of::<(Words, u128)>();
    assert!(certificate_bytes > last_bytes);
    let (mut no_general, nt, nh, ns) = fixture(true, 2, 3, 3, 3);
    projection_proofs(&mut no_general, &nh, &ns, false);
    let no_general_before = no_general.allocated;
    old_graph(&mut no_general, &graph, &nt, &nh, &ns, 3).unwrap();
    assert_eq!(
        total,
        no_general.allocated - no_general_before + certificate_bytes
    );
    for short in [0, 1] {
        let (mut a, t, h, sources) = fixture(true, 2, 3, 3, 3);
        let (mut e, et, eh, es) = fixture(true, 2, 3, 3, 3);
        projection_proofs(&mut a, &h, &sources, true);
        projection_proofs(&mut e, &eh, &es, true);
        a.allocated = MAX_PROGRAM_BYTES - total + short;
        e.allocated = a.allocated;
        let av = actual_graph(
            &mut a,
            &graph,
            &t,
            &h,
            &sources,
            3,
            &mut AdmissionSite::default(),
        );
        let ev = old_graph(&mut e, &graph, &et, &eh, &es, 3);
        if short == 0 {
            assert!(av.is_ok());
            assert_eq!(a.allocated, MAX_PROGRAM_BYTES);
            assert_eq!(a.attention_energies.len(), 4);
        } else {
            assert!(matches!(&av, Err(Error::Bounds)));
            assert_eq!(a.attention_energies.len(), 3);
            assert_eq!(a.allocated, MAX_PROGRAM_BYTES - last_bytes + 1);
        }
        same_result(av, ev);
        same_contraction_state(&a, &e);
        assert!(!a.tensors.is_empty());
        assert_eq!(a.row_energies.len(), 2); // attention output never general norm
        assert_eq!(
            a.attention_energies,
            quote.attention_energies[..(4 - short)]
        );
    }
}

#[test]
fn original_projection_does_not_consume_attention_only_input_energy_as_general_certificate() {
    let graph = semantics(Op::OutputHead, 2, 3, true);
    let (mut a, t, h, sources) = fixture(true, 2, 3, 3, 3);
    let (mut e, et, eh, es) = fixture(true, 2, 3, 3, 3);
    projection_proofs(&mut a, &h, &sources, false);
    projection_proofs(&mut e, &eh, &es, false);
    let original_attention = a.attention_energies.clone();
    let original_bytes = a.allocated;
    assert_eq!(a.row_norm(&h.words, 3), None);
    let av = actual_graph(
        &mut a,
        &graph,
        &t,
        &h,
        &sources,
        3,
        &mut AdmissionSite::default(),
    );
    let ev = old_graph(&mut e, &graph, &et, &eh, &es, 3);
    let output = av.as_ref().unwrap();
    assert_eq!(a.row_norm(&output.words, 3), None);
    assert_eq!(a.attention_energy(&output.words, 3), None);
    assert!(a.allocated > original_bytes); // actual native projection did publish
    same_result(av, ev);
    same_contraction_state(&a, &e);
    assert!(a.row_energies.is_empty());
    assert_eq!(a.attention_energies, original_attention);
}
