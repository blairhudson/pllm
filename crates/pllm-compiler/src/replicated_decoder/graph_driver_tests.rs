//! Parity definitions use the original expanded entry point and an independent
//! pre-extraction composition. They never call the new driver as their oracle.
//! Existing program_graph_tests frozen complete-program identities remain intact.
use super::*;
use pllm_models::{DecoderMode, ModelOperation, StateKind, StateTensor};

fn semantic(width: usize) -> DecoderGraph {
    DecoderGraph {
        mode: DecoderMode::Prefill,
        batch: 1,
        query_sequence: 2,
        maximum_key_sequence: 2,
        operations: vec![ModelOperation {
            id: "layer.7.original.norm".into(),
            operator: Op::RmsNorm,
            layer: Some(7),
            state_kind: None,
            inputs: vec!["hidden".into()],
            output_shape: vec![1, 2, width as u64],
            attributes: serde_json::json!({"weight": "gamma", "weight_offset": 0}),
        }],
        state_inputs: vec![StateTensor {
            id: "hidden".into(),
            layer: Some(7),
            kind: StateKind::Key,
            shape: vec![1, 2, width as u64],
            maximum_sequence: 2,
        }],
        state_outputs: vec![],
        output: "layer.7.original.norm".into(),
    }
}

fn gamma(width: usize) -> BTreeMap<String, SourceTensor> {
    let values = vec![ONE; width];
    let digest = Sha256::digest(
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .into();
    BTreeMap::from([(
        "gamma".into(),
        SourceTensor {
            shape: vec![width],
            lower: ONE,
            upper: ONE,
            public_q18: Some(values),
            digest,
            row_l1: None,
            row_squared_norm: None,
            row_l1_squared_sum: None,
        },
    )])
}

fn inputs(width: usize, public: bool, checked: Option<i64>) -> (Builder, View, View) {
    let mut b = Builder {
        public_weights: public,
        checked_bound: checked,
        ..Builder::default()
    };
    let tokens = b
        .push(
            Code::Input("input.tokens".into()),
            2,
            Interval::new(0, 2).unwrap(),
        )
        .unwrap();
    let tokens = b.view(tokens, vec![1, 2]);
    let hidden = b
        .push(
            Code::Input("hidden".into()),
            2 * width,
            Interval::new(-2 * ONE, 2 * ONE).unwrap(),
        )
        .unwrap();
    let hidden = b.view(hidden, vec![1, 2, width]);
    // Test-private synthetic ledger fixtures exercise lookup/order parity only.
    // These entries are not authority derived from a checked numeric producer.
    for row in hidden.words.chunks_exact(width) {
        b.record_row_energy(&row, (ONE as u128).pow(2)).unwrap();
        b.record_attention_energy(&row, 2 * (ONE as u128).pow(2))
            .unwrap();
    }
    (b, tokens, hidden)
}

/// Frozen pre-extraction RMSNorm composition. Only original Builder primitives
/// are used; no RmsNormSink methods or graph_driver helpers supply expectations.
fn original(
    b: &mut Builder,
    tokens: &View,
    hidden: Option<&View>,
    op: &ModelOperation,
    public: &BTreeMap<String, SourceTensor>,
) -> Result<Words, Error> {
    let anchor = b.gather(&tokens.words.slice(..1))?;
    b.constant(anchor, 0)?;
    b.admission_stage = "operator_lowering";
    if let Some(hidden) = hidden {
        b.bounds(&hidden.words)?;
    }
    if op.attributes.get("weight_offset").and_then(|v| v.as_i64()) != Some(0) {
        return Err(Error::Binding);
    }
    let gamma = public
        .get(
            op.attributes
                .get("weight")
                .and_then(|v| v.as_str())
                .ok_or(Error::Binding)?,
        )
        .ok_or(Error::Binding)?;
    let width = *op.output_shape.last().ok_or(Error::Shape)? as usize;
    if gamma.shape != [width] {
        return Err(Error::Shape);
    }
    let mut words = Words::new();
    let original = &hidden.ok_or(Error::Binding)?.words;
    let checked = if b.checked_bound.is_some() {
        Some(b.checked_rows(original, width)?)
    } else {
        None
    };
    for row in checked.as_ref().unwrap_or(original).chunks_exact(width) {
        b.admission_stage = "rms_norm_input_gather";
        let input = b.gather(&row)?;
        b.admission_stage = "rms_norm_numeric_constructor";
        let (program, squared) = if b.public_weights {
            let gamma = gamma.public_q18.as_ref().ok_or(Error::Binding)?;
            let (program, squared) = match b.row_norm(&row, width) {
                Some(norm) => tensor::Program::rms_norm_with_input_norm(
                    1,
                    width,
                    b.nodes[input].bounds,
                    gamma,
                    norm,
                )?,
                None => {
                    tensor::Program::rms_norm_certified(1, width, b.nodes[input].bounds, gamma)?
                }
            };
            (program, Some(squared))
        } else {
            (
                tensor::Program::rms_norm(
                    1,
                    width,
                    b.nodes[input].bounds,
                    gamma.public_q18.as_ref().ok_or(Error::Binding)?,
                )?,
                None,
            )
        };
        b.admission_stage = "rms_norm_tensor_owner";
        let output = b.tensor(input, program)?;
        if let Some(squared) = squared {
            b.admission_stage = "rms_norm_output_certificate";
            b.charge_program_bytes(std::mem::size_of::<(usize, u128)>())?;
            b.row_norms
                .try_reserve_exact(1)
                .map_err(|_| Error::Allocation)?;
            b.row_norms.push((output, squared));
        }
        words.extend((0..width).map(|i| (output, i)));
    }
    Ok(words)
}

fn lower(
    b: &mut Builder,
    tokens: &View,
    hidden: Option<&View>,
    semantic: &DecoderGraph,
    public: &BTreeMap<String, SourceTensor>,
    site: &mut AdmissionSite,
) -> Result<Words, Error> {
    let mut state = BTreeMap::new();
    if let Some(hidden) = hidden {
        state.insert((7, StateKind::Key), hidden.clone());
    }
    b.graph(
        semantic,
        tokens,
        &BTreeMap::new(),
        public,
        3,
        0,
        &mut state,
        site,
    )
    .map(|output| {
        assert_eq!(
            output.shape,
            [1, 2, semantic.operations[0].output_shape[2] as usize]
        );
        output.words
    })
}

fn assert_same_schedule(actual: &Builder, expected: &Builder) {
    assert_eq!(actual.nodes.len(), expected.nodes.len());
    for (a, e) in actual.nodes.iter().zip(&expected.nodes) {
        assert_eq!(a.lanes, e.lanes);
        assert_eq!(a.bounds, e.bounds);
        match (&a.code, &e.code) {
            (Code::Input(a), Code::Input(e)) => assert_eq!(a, e),
            (Code::Gather(a), Code::Gather(e)) => assert_eq!(a, e),
            (
                Code::Affine {
                    left: al,
                    right: ar,
                    coefficients: ac,
                },
                Code::Affine {
                    left: el,
                    right: er,
                    coefficients: ec,
                },
            ) => {
                let (mut a, mut e) = (Sha256::new(), Sha256::new());
                hash_affine(&mut a, *al, *ar, *ac);
                hash_affine(&mut e, *el, *er, *ec);
                assert_eq!(a.finalize(), e.finalize());
            }
            (
                Code::Tensor {
                    input: ai,
                    program: ap,
                },
                Code::Tensor {
                    input: ei,
                    program: ep,
                },
            )
            | (
                Code::CheckedRange {
                    input: ai,
                    program: ap,
                },
                Code::CheckedRange {
                    input: ei,
                    program: ep,
                },
            ) => {
                assert_eq!(ai, ei);
                assert_eq!(ap.digest(), ep.digest());
                assert!(Arc::ptr_eq(ap, actual.tensors.get(&ap.digest()).unwrap()));
                assert!(Arc::ptr_eq(ep, expected.tensors.get(&ep.digest()).unwrap()));
            }
            (
                Code::Product {
                    left: al,
                    right: ar,
                    shape: ash,
                    rescale: ac,
                    norm: an,
                },
                Code::Product {
                    left: el,
                    right: er,
                    shape: esh,
                    rescale: ec,
                    norm: en,
                },
            ) => {
                let (mut a, mut e) = (Sha256::new(), Sha256::new());
                hash_product(&mut a, *al, *ar, *ash, *ac, an.as_deref());
                hash_product(&mut e, *el, *er, *esh, *ec, en.as_deref());
                assert_eq!(a.finalize(), e.finalize());
            }
            _ => panic!("original graph node kind/order changed"),
        }
    }
    assert_eq!(
        schedule_work(&actual.nodes).unwrap(),
        schedule_work(&expected.nodes).unwrap()
    );
    assert_eq!(actual.allocated, expected.allocated);
    assert_eq!(actual.value_payload_bytes, expected.value_payload_bytes);
    assert_eq!(actual.workspace_bytes, expected.workspace_bytes);
    assert_eq!(actual.validity, expected.validity);
    assert_eq!(actual.row_norms, expected.row_norms);
    assert_eq!(actual.row_energies, expected.row_energies);
    assert_eq!(actual.attention_energies, expected.attention_energies);
    assert_eq!(actual.row_masses, expected.row_masses);
    assert_eq!(
        actual.tensors.keys().collect::<Vec<_>>(),
        expected.tensors.keys().collect::<Vec<_>>()
    );
    assert_eq!(actual.admission_ledger(), expected.admission_ledger());
}

#[test]
fn expanded_rms_family_preserves_original_checked_rows_native_owners_and_proof_scopes() {
    // 65-wide rows cross the guard's 128-lane tile boundary; 3-wide rows cover
    // ordinary multirow lowering. Both must guard the WHOLE input before RMS.
    for width in [3, 65] {
        for public_weights in [false, true] {
            for checked in [None, Some(ONE), Some(2 * ONE)] {
                let graph = semantic(width);
                let public = gamma(width);
                let (mut actual, tokens, hidden) = inputs(width, public_weights, checked);
                let (mut expected, old_tokens, old_hidden) = inputs(width, public_weights, checked);
                let mut site = AdmissionSite {
                    selection: 11,
                    ..AdmissionSite::default()
                };
                let output = lower(
                    &mut actual,
                    &tokens,
                    Some(&hidden),
                    &graph,
                    &public,
                    &mut site,
                )
                .unwrap();
                let old = original(
                    &mut expected,
                    &old_tokens,
                    Some(&old_hidden),
                    &graph.operations[0],
                    &public,
                )
                .unwrap();
                assert_eq!(output, old);
                assert_same_schedule(&actual, &expected);
                assert_eq!(site.selection, 11);
                assert_eq!(site.step, Some(0));
                assert_eq!(site.operator, Some(Op::RmsNorm));
                assert_eq!(site.layer, Some(7));
                assert_eq!(site.inputs, [Some((-2 * ONE, 2 * ONE)), None]);
                assert_eq!(actual.attention_energies.len(), 2);
                assert!(actual.row_masses.is_empty());
            }
        }
    }
}

#[test]
fn expanded_rms_errors_keep_original_precedence_partial_schedule_and_admission_site() {
    for fault in [
        "offset",
        "binding",
        "shape_before_input",
        "input",
        "guard",
        "native",
    ] {
        let width = 3;
        let mut graph = semantic(width);
        let mut public = gamma(width);
        let checked = if fault == "guard" {
            Some(-1)
        } else {
            Some(ONE)
        };
        let missing = matches!(fault, "shape_before_input" | "input");
        match fault {
            "offset" => {
                graph.operations[0].attributes["weight_offset"] = 1.into();
                public.clear();
            }
            "binding" => public.clear(),
            "shape_before_input" => public.get_mut("gamma").unwrap().shape = vec![4],
            "native" => public.get_mut("gamma").unwrap().public_q18 = None,
            _ => (),
        }
        if missing {
            graph.state_inputs.clear();
        }
        let (mut actual, tokens, hidden) = inputs(width, true, checked);
        let (mut expected, old_tokens, old_hidden) = inputs(width, true, checked);
        let mut site = AdmissionSite {
            selection: 11,
            ..AdmissionSite::default()
        };
        let error = lower(
            &mut actual,
            &tokens,
            (!missing).then_some(&hidden),
            &graph,
            &public,
            &mut site,
        )
        .expect_err("original fault must stop graph publication");
        let old_error = original(
            &mut expected,
            &old_tokens,
            (!missing).then_some(&old_hidden),
            &graph.operations[0],
            &public,
        )
        .err()
        .unwrap();
        assert_eq!(error, old_error);
        assert_eq!(
            error,
            match fault {
                "shape_before_input" => Error::Shape,
                "guard" => Error::Bounds,
                _ => Error::Binding,
            }
        );
        assert_same_schedule(&actual, &expected);
        assert_eq!(site.selection, 11);
        assert_eq!(site.step, Some(0));
        assert_eq!(site.operator, Some(Op::RmsNorm));
        assert_eq!(site.layer, Some(7));
        assert_eq!(site.inputs[0], (!missing).then_some((-2 * ONE, 2 * ONE)));
    }
}

#[test]
fn expanded_rms_guard_budget_failure_precedes_native_rows_and_keeps_original_cap() {
    let width = 3;
    let graph = semantic(width);
    let public = gamma(width);
    // Quote only the unchanged scalar graph prelude using original primitives.
    // The whole-input guard then encounters an exactly full retained byte cap.
    let (mut quote, tokens, _) = inputs(width, true, Some(ONE));
    let before = quote.allocated;
    let anchor = quote.gather(&tokens.words.slice(..1)).unwrap();
    quote.constant(anchor, 0).unwrap();
    let prelude = quote.allocated - before;

    let (mut actual, tokens, hidden) = inputs(width, true, Some(ONE));
    let (mut expected, old_tokens, old_hidden) = inputs(width, true, Some(ONE));
    actual.allocated = MAX_PROGRAM_BYTES - prelude;
    expected.allocated = MAX_PROGRAM_BYTES - prelude;
    let mut site = AdmissionSite::default();
    assert!(matches!(
        lower(
            &mut actual,
            &tokens,
            Some(&hidden),
            &graph,
            &public,
            &mut site
        ),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        original(
            &mut expected,
            &old_tokens,
            Some(&old_hidden),
            &graph.operations[0],
            &public
        ),
        Err(Error::Bounds)
    ));
    assert_same_schedule(&actual, &expected);
    assert_eq!(actual.admission_stage, "checked_range_view_metadata");
    assert_eq!(actual.requested_bytes, Some(MAX_PROGRAM_BYTES + 128));
    assert!(actual.tensors.is_empty());
    assert!(actual.validity.is_none());
    assert_eq!(site.step, Some(0));
    assert_eq!(site.operator, Some(Op::RmsNorm));
}

#[test]
fn expanded_rms_attention_only_ledger_does_not_refine_general_norm_admission() {
    let width = 3;
    let graph = semantic(width);
    let public = gamma(width);
    let (mut actual, tokens, hidden) = inputs(width, true, None);
    let (mut expected, old_tokens, old_hidden) = inputs(width, true, None);
    // Forged test-private negative control: retain only the synthetic attention
    // ledger, not a general row certificate. Its tighter energy is NOT authority
    // for RMSNorm. Clearing entries keeps their original byte charge conservative.
    actual.row_energies.clear();
    expected.row_energies.clear();
    assert!(actual.row_norms.is_empty());
    assert!(expected.row_norms.is_empty());
    for row in hidden.words.chunks_exact(width) {
        assert_eq!(actual.row_norm(&row, width), None);
        assert_eq!(
            actual.attention_energy(&row, width),
            Some(2 * (ONE as u128).pow(2))
        );
    }
    let attention = actual.attention_energies.clone();
    let bounds = actual.bounds(&hidden.words).unwrap();
    let (certified, _) = tensor::Program::rms_norm_certified(
        1,
        width,
        bounds,
        public["gamma"].public_q18.as_ref().unwrap(),
    )
    .unwrap();
    let mut site = AdmissionSite::default();
    let output = lower(
        &mut actual,
        &tokens,
        Some(&hidden),
        &graph,
        &public,
        &mut site,
    )
    .unwrap();
    let old = original(
        &mut expected,
        &old_tokens,
        Some(&old_hidden),
        &graph.operations[0],
        &public,
    )
    .unwrap();
    assert_eq!(output, old);
    assert_same_schedule(&actual, &expected);
    assert_eq!(actual.attention_energies, attention);
    assert!(actual.row_energies.is_empty());
    assert_eq!(actual.row_norms.len(), 2);
    for row in output.chunks_exact(width) {
        let Code::Tensor { program, .. } = &actual.nodes[row.at(0).0].code else {
            panic!("RMSNorm row must retain its original native tensor owner");
        };
        assert_eq!(program.digest(), certified.digest());
    }
}

#[test]
fn expanded_rms_certificate_budget_failure_keeps_published_tensor_and_original_order() {
    let width = 3;
    let graph = semantic(width);
    let public = gamma(width);
    let certificate_bytes = std::mem::size_of::<(usize, u128)>();
    let (mut quote, quote_tokens, quote_hidden) = inputs(width, true, None);
    quote.row_energies.clear();
    // Quote the independent original prelude and first-row composition only,
    // before any complete-graph output-count check. The native constructor derives
    // the output certificate from scalar bounds; no input-row authority is forged.
    let first_row = View {
        words: quote_hidden.words.slice(..width),
        shape: vec![1, 1, width],
    };
    let before = quote.allocated;
    let quoted = original(
        &mut quote,
        &quote_tokens,
        Some(&first_row),
        &graph.operations[0],
        &public,
    )
    .unwrap();
    assert_eq!(quote.row_norms.len(), 1);
    let publication_bytes = quote.allocated - before - certificate_bytes;
    let first_owner = quoted.at(0).0;
    assert_eq!(first_owner, quote.nodes.len() - 1);

    let (mut actual, tokens, hidden) = inputs(width, true, None);
    let (mut expected, old_tokens, old_hidden) = inputs(width, true, None);
    actual.row_energies.clear();
    expected.row_energies.clear();
    // Test-private counter saturation, not an allocator/whole-request quote:
    // publication exactly fills the existing cap; only its certificate exceeds it.
    actual.allocated = MAX_PROGRAM_BYTES - publication_bytes;
    expected.allocated = MAX_PROGRAM_BYTES - publication_bytes;
    let mut site = AdmissionSite::default();
    assert!(matches!(
        lower(
            &mut actual,
            &tokens,
            Some(&hidden),
            &graph,
            &public,
            &mut site
        ),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        original(
            &mut expected,
            &old_tokens,
            Some(&old_hidden),
            &graph.operations[0],
            &public
        ),
        Err(Error::Bounds)
    ));
    assert_same_schedule(&actual, &expected);
    assert_eq!(actual.nodes.len(), quote.nodes.len());
    assert_eq!(actual.allocated, MAX_PROGRAM_BYTES);
    assert_eq!(
        actual.requested_bytes,
        Some(MAX_PROGRAM_BYTES + certificate_bytes)
    );
    assert_eq!(actual.admission_limit, Some("program_bytes"));
    assert_eq!(actual.admission_stage, "rms_norm_output_certificate");
    assert!(actual.row_norms.is_empty());
    assert_eq!(actual.row_norms.capacity(), expected.row_norms.capacity());
    let Code::Tensor { program, .. } = &actual.nodes[first_owner].code else {
        panic!("certificate failure must occur after native tensor publication");
    };
    assert!(Arc::ptr_eq(
        program,
        actual.tensors.get(&program.digest()).unwrap()
    ));
    assert_eq!(actual.tensors.len(), 1);
    assert_eq!(site.step, Some(0));
    assert_eq!(site.operator, Some(Op::RmsNorm));
}

#[test]
fn expanded_frontier_duplicate_inputs_and_dead_outputs_match_original_retirement() {
    let operation = |id: &str, operator, inputs: &[&str]| ModelOperation {
        id: id.into(),
        operator,
        layer: Some(7),
        state_kind: None,
        inputs: inputs.iter().map(|id| (*id).into()).collect(),
        output_shape: vec![1, 2, 3],
        attributes: serde_json::json!({}),
    };
    let mut graph = semantic(3);
    graph.operations = vec![
        operation("dead_before", Op::TokenFeedback, &["hidden"]),
        operation("duplicate", Op::ResidualAdd, &["hidden", "hidden"]),
        operation("selected", Op::TokenFeedback, &["duplicate"]),
        operation("dead_after", Op::ResidualAdd, &["selected", "selected"]),
    ];
    graph.output = "selected".into();
    let (mut actual, tokens, hidden) = inputs(3, false, None);
    let (mut expected, old_tokens, old_hidden) = inputs(3, false, None);
    let mut site = AdmissionSite::default();
    let output = lower_rank(&mut actual, &tokens, Some(&hidden), &graph, &mut site).unwrap();

    // Original inline Builder::graph frontier and expanded compositions. Compare
    // each retirement boundary through child-module private state, not a new hook.
    let mut values: BTreeMap<String, View> = BTreeMap::from([
        ("hidden".into(), old_hidden.clone()),
        ("input.tokens".into(), old_tokens.clone()),
    ]);
    let mut uses: BTreeMap<&str, usize> = BTreeMap::new();
    for op in &graph.operations {
        for input in &op.inputs {
            *uses.entry(input).or_default() += 1;
        }
    }
    let mut frontier = OrderedGraph::<View>::new(&graph);
    frontier.bind("hidden".into(), old_hidden);
    frontier.bind("input.tokens".into(), old_tokens.clone());
    let anchor = expected.gather(&old_tokens.words.slice(..1)).unwrap();
    expected.constant(anchor, 0).unwrap();
    let retained = [
        ["hidden", "input.tokens"],
        ["duplicate", "input.tokens"],
        ["input.tokens", "selected"],
        ["input.tokens", "selected"],
    ];
    for (step, op) in graph.operations.iter().enumerate() {
        expected.admission_stage = "operator_lowering";
        let a = op.inputs.first().and_then(|id| values.get(id));
        let b = op.inputs.get(1).and_then(|id| values.get(id));
        let (front_a, front_b) = frontier.inputs(op);
        for (original, extracted) in [(a, front_a), (b, front_b)] {
            assert_eq!(original.is_some(), extracted.is_some());
            if let (Some(original), Some(extracted)) = (original, extracted) {
                assert_eq!(original.words, extracted.words);
                assert_eq!(original.shape, extracted.shape);
                expected.bounds(&original.words).unwrap();
            }
        }
        let lowered = match op.operator {
            Op::TokenFeedback => a.unwrap().clone(),
            Op::ResidualAdd => View {
                words: expected.residual(a.unwrap(), b.unwrap()).unwrap(),
                shape: vec![1, 2, 3],
            },
            _ => unreachable!(),
        };
        assert_eq!(lowered.words.len(), 6);
        values.insert(op.id.clone(), lowered.clone());
        for input in &op.inputs {
            let count = uses.get_mut(input.as_str()).unwrap();
            *count -= 1;
            if *count == 0 && input != &graph.output {
                values.remove(input);
            }
        }
        if !uses.contains_key(op.id.as_str()) && op.id != graph.output {
            values.remove(&op.id);
        }
        frontier.finish(op, lowered).unwrap();
        assert_eq!(frontier.uses, uses);
        assert_eq!(
            values.keys().map(String::as_str).collect::<Vec<_>>(),
            retained[step]
        );
        assert_eq!(
            frontier.values.keys().collect::<Vec<_>>(),
            values.keys().collect::<Vec<_>>()
        );
        for (id, original) in &values {
            let extracted = &frontier.values[id];
            assert_eq!(extracted.words, original.words);
            assert_eq!(extracted.shape, original.shape);
        }
    }
    assert_eq!(uses["hidden"], 0);
    assert_eq!(uses["selected"], 0);
    let old = values.remove(&graph.output).unwrap();
    let retired = frontier.output().unwrap();
    assert_eq!(retired.words, old.words);
    assert_eq!(retired.shape, old.shape);
    assert_eq!(output.words, old.words);
    assert_eq!(output.shape, old.shape);
    assert_same_schedule(&actual, &expected);
    assert_eq!(site.step, Some(3));
    assert_eq!(site.operator, Some(Op::ResidualAdd));
}

fn rank_operation(operator: Op, shape: &[u64], attributes: serde_json::Value) -> ModelOperation {
    ModelOperation {
        id: "layer.7.original.view".into(),
        operator,
        layer: Some(7),
        state_kind: None,
        inputs: vec!["hidden".into()],
        output_shape: shape.to_vec(),
        attributes,
    }
}

fn rank_graph(op: ModelOperation, input_shape: &[usize]) -> DecoderGraph {
    let mut graph = semantic(6);
    graph.output = op.id.clone();
    graph.operations = vec![op];
    graph.state_inputs[0].shape = input_shape.iter().map(|n| *n as u64).collect();
    graph
}

fn rank_inputs(shape: &[usize], checked: bool) -> (Builder, View, View) {
    let (mut b, tokens, mut hidden) = inputs(6, true, checked.then_some(ONE));
    if checked {
        // Original checked producer supplies native Arcs and sticky validity,
        // and propagates the test-private ledger fixtures above, not authority.
        // View lowering must not replace any of those existing entries/owners.
        hidden.words = b.checked_rows(&hidden.words, 6).unwrap();
        assert!(!b.tensors.is_empty());
        assert!(b.validity.is_some());
    }
    hidden.shape = shape.to_vec();
    (b, tokens, hidden)
}

fn lower_rank(
    b: &mut Builder,
    tokens: &View,
    hidden: Option<&View>,
    graph: &DecoderGraph,
    site: &mut AdmissionSite,
) -> Result<View, Error> {
    let mut state = BTreeMap::new();
    if let Some(hidden) = hidden {
        state.insert((7, StateKind::Key), hidden.clone());
    }
    b.graph(
        graph,
        tokens,
        &BTreeMap::new(),
        &BTreeMap::new(),
        3,
        0,
        &mut state,
        site,
    )
}

/// Frozen single-operation expanded composition, not a second traversal or a
/// driver oracle. Keep the original inline indexing, guards and final count.
fn original_rank(
    b: &mut Builder,
    tokens: &View,
    input: Option<&View>,
    op: &ModelOperation,
) -> Result<View, Error> {
    let anchor = b.gather(&tokens.words.slice(..1))?;
    b.constant(anchor, 0)?;
    b.admission_stage = "operator_lowering";
    let mut shape: Vec<usize> = op.output_shape.iter().map(|v| *v as usize).collect();
    if op.operator == Op::CacheSuffix {
        shape[2] = tokens.words.len();
    }
    if let Some(input) = input {
        b.bounds(&input.words)?;
    }
    let output = match op.operator {
        Op::Reshape => {
            let a = input.ok_or(Error::Binding)?;
            let words = match op.attributes.get("layout").and_then(|v| v.as_str()) {
                Some("batch_heads_sequence_feature") if shape.len() == 4 => {
                    let (heads, rows, width) = (shape[1], shape[2], shape[3]);
                    (0..heads)
                        .flat_map(|h| {
                            (0..rows).flat_map(move |r| {
                                (0..width).map(move |d| a.words.at((r * heads + h) * width + d))
                            })
                        })
                        .collect()
                }
                Some("batch_sequence_hidden") if a.shape.len() == 4 => {
                    let (heads, rows, width) = (a.shape[1], a.shape[2], a.shape[3]);
                    (0..rows)
                        .flat_map(|r| {
                            (0..heads).flat_map(move |h| {
                                (0..width).map(move |d| a.words.at((h * rows + r) * width + d))
                            })
                        })
                        .collect()
                }
                _ => return Err(Error::Binding),
            };
            View {
                words,
                shape: shape.clone(),
            }
        }
        Op::LastToken => {
            let a = input.ok_or(Error::Binding)?;
            let width = *shape.last().unwrap();
            View {
                words: a.words.slice(a.words.len() - width..),
                shape: shape.clone(),
            }
        }
        Op::CacheSuffix => {
            if op.attributes.get("semantics").and_then(|v| v.as_str())
                != Some("visible_valid_prefix")
            {
                return Err(Error::Binding);
            }
            input.ok_or(Error::Binding)?.clone()
        }
        _ => unreachable!(),
    };
    if output.words.len() != shape.iter().product::<usize>() {
        return Err(Error::Shape);
    }
    Ok(output)
}

#[test]
fn expanded_rank_views_preserve_original_words_native_arcs_and_certificate_scopes() {
    let split = serde_json::json!({"layout": "batch_heads_sequence_feature"});
    let merge = serde_json::json!({"layout": "batch_sequence_hidden"});
    let prefix = serde_json::json!({"semantics": "visible_valid_prefix"});
    let cases = [
        (
            rank_operation(Op::Reshape, &[1, 3, 2, 2], split.clone()),
            vec![1, 2, 6],
            vec![1, 3, 2, 2],
            vec![0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11],
        ),
        (
            rank_operation(Op::Reshape, &[1, 2, 6], merge.clone()),
            vec![1, 3, 2, 2],
            vec![1, 2, 6],
            vec![0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11],
        ),
        // Original guards are asymmetric: no new input-rank, batch, or
        // output-rank validation may be invented by this extraction.
        (
            rank_operation(Op::Reshape, &[1, 3, 2, 2], split),
            vec![12],
            vec![1, 3, 2, 2],
            vec![0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11],
        ),
        (
            rank_operation(Op::Reshape, &[12], merge),
            vec![2, 3, 2, 2],
            vec![12],
            vec![0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11],
        ),
        (
            rank_operation(Op::LastToken, &[1, 6], serde_json::json!({})),
            vec![1, 2, 6],
            vec![1, 6],
            vec![6, 7, 8, 9, 10, 11],
        ),
        (
            rank_operation(Op::LastToken, &[6], serde_json::json!({})),
            vec![12],
            vec![6],
            vec![6, 7, 8, 9, 10, 11],
        ),
        (
            rank_operation(Op::LastToken, &[1, 0], serde_json::json!({})),
            vec![1, 2, 6],
            vec![1, 0],
            vec![],
        ),
        // Declared key extent is rewritten by graph, but alias keeps the
        // actual input shape, even when equal lane counts hide different ranks.
        (
            rank_operation(Op::CacheSuffix, &[1, 2, 99, 3], prefix.clone()),
            vec![1, 3, 2, 2],
            vec![1, 3, 2, 2],
            (0..12).collect(),
        ),
        (
            rank_operation(Op::CacheSuffix, &[1, 3, 99, 2], prefix),
            vec![99],
            vec![99],
            (0..12).collect(),
        ),
    ];
    for (op, input_shape, output_shape, indices) in cases {
        for checked in [false, true] {
            let graph = rank_graph(op.clone(), &input_shape);
            let (mut actual, tokens, hidden) = rank_inputs(&input_shape, checked);
            let (mut expected, old_tokens, old_hidden) = rank_inputs(&input_shape, checked);
            let natives = actual.tensors.clone();
            let rows = actual.row_energies.clone();
            let attention = actual.attention_energies.clone();
            let validity = actual.validity;
            let mut site = AdmissionSite {
                selection: 11,
                ..AdmissionSite::default()
            };
            let output =
                lower_rank(&mut actual, &tokens, Some(&hidden), &graph, &mut site).unwrap();
            let old = original_rank(&mut expected, &old_tokens, Some(&old_hidden), &op).unwrap();
            assert_eq!(output.words, old.words);
            assert_eq!(output.shape, old.shape);
            assert_eq!(output.shape, output_shape);
            // Literal index expectations are independent of both algorithms.
            assert_eq!(
                output.words.iter().collect::<Vec<_>>(),
                indices
                    .iter()
                    .map(|i| hidden.words.at(*i))
                    .collect::<Vec<_>>()
            );
            assert_same_schedule(&actual, &expected);
            assert_eq!(actual.row_energies, rows);
            assert_eq!(actual.attention_energies, attention);
            assert_eq!(actual.validity, validity);
            for (digest, native) in &natives {
                assert!(Arc::ptr_eq(native, &actual.tensors[digest]));
            }
            assert_eq!(site.selection, 11);
            assert_eq!(site.step, Some(0));
            assert_eq!(site.operator, Some(op.operator));
            assert_eq!(site.layer, Some(7));
            let bounds = actual.bounds(&hidden.words).unwrap();
            assert_eq!(site.inputs, [Some((bounds.lower(), bounds.upper())), None]);
        }
    }
}

#[test]
fn expanded_rank_malformed_geometry_keeps_original_errors_panics_and_partial_schedule() {
    let split = serde_json::json!({"layout": "batch_heads_sequence_feature"});
    let merge = serde_json::json!({"layout": "batch_sequence_hidden"});
    let prefix = serde_json::json!({"semantics": "visible_valid_prefix"});
    // None denotes an original panic, not permission to repair malformed rank
    // geometry or replace the original behavior with a new Error::Shape.
    let cases = [
        (
            Op::Reshape,
            vec![1, 2, 6],
            vec![1, 2, 6],
            split.clone(),
            false,
            Some(Error::Binding),
        ),
        (
            Op::Reshape,
            vec![1, 3, 2, 2],
            vec![1, 2, 6],
            merge.clone(),
            false,
            Some(Error::Binding),
        ),
        (
            Op::Reshape,
            vec![1, 3, 2, 2],
            vec![1, 2, 6],
            serde_json::json!({"layout": 7}),
            false,
            Some(Error::Binding),
        ),
        (
            Op::Reshape,
            vec![2, 3, 2, 2],
            vec![1, 2, 6],
            split.clone(),
            false,
            Some(Error::Shape),
        ),
        (
            Op::Reshape,
            vec![1, 4, 2, 2],
            vec![1, 2, 6],
            split.clone(),
            false,
            None,
        ),
        (
            Op::Reshape,
            vec![1, 2, 7],
            vec![1, 3, 2, 2],
            merge.clone(),
            false,
            Some(Error::Shape),
        ),
        (
            Op::Reshape,
            vec![1, 2, 6],
            vec![1, 4, 2, 2],
            merge,
            false,
            None,
        ),
        (
            Op::Reshape,
            vec![1, 4, 2, 2],
            vec![1, 2, 6],
            split,
            true,
            Some(Error::Binding),
        ),
        (
            Op::LastToken,
            vec![],
            vec![1, 2, 6],
            serde_json::json!({}),
            true,
            Some(Error::Binding),
        ),
        (
            Op::LastToken,
            vec![],
            vec![1, 2, 6],
            serde_json::json!({}),
            false,
            None,
        ),
        (
            Op::LastToken,
            vec![1, 13],
            vec![1, 2, 6],
            serde_json::json!({}),
            false,
            None,
        ),
        (
            Op::LastToken,
            vec![2, 6],
            vec![1, 2, 6],
            serde_json::json!({}),
            false,
            Some(Error::Shape),
        ),
        (
            Op::CacheSuffix,
            vec![1, 3, 2, 2],
            vec![1, 3, 2, 2],
            serde_json::json!({"semantics": "suffix"}),
            true,
            Some(Error::Binding),
        ),
        (
            Op::CacheSuffix,
            vec![1, 3, 2, 2],
            vec![1, 3, 2, 2],
            prefix.clone(),
            true,
            Some(Error::Binding),
        ),
        (
            Op::CacheSuffix,
            vec![1, 2],
            vec![1, 3, 2, 2],
            prefix.clone(),
            false,
            None,
        ),
        (
            Op::CacheSuffix,
            vec![1, 2, 99, 4],
            vec![1, 3, 2, 2],
            prefix,
            false,
            Some(Error::Shape),
        ),
    ];
    for (operator, shape, input_shape, attributes, missing, error) in cases {
        let op = rank_operation(operator, &shape, attributes);
        let mut graph = rank_graph(op.clone(), &input_shape);
        if missing {
            graph.state_inputs.clear();
        }
        let (mut actual, tokens, hidden) = rank_inputs(&input_shape, true);
        let (mut expected, old_tokens, old_hidden) = rank_inputs(&input_shape, true);
        let mut site = AdmissionSite {
            selection: 11,
            ..AdmissionSite::default()
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lower_rank(
                &mut actual,
                &tokens,
                (!missing).then_some(&hidden),
                &graph,
                &mut site,
            )
        }));
        let old = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            original_rank(
                &mut expected,
                &old_tokens,
                (!missing).then_some(&old_hidden),
                &op,
            )
        }));
        match (result, old, error) {
            (Ok(Err(actual)), Ok(Err(original)), Some(error)) => {
                assert_eq!(actual, original);
                assert_eq!(actual, error);
            }
            (Err(actual), Err(original), None) => {
                let message = |payload: Box<dyn std::any::Any + Send>| {
                    if let Some(message) = payload.downcast_ref::<String>() {
                        message.clone()
                    } else {
                        payload.downcast_ref::<&str>().unwrap().to_string()
                    }
                };
                assert_eq!(message(actual), message(original));
            }
            _ => panic!("original rank/view failure changed: {operator:?}, {shape:?}"),
        }
        assert_same_schedule(&actual, &expected);
        assert_eq!(site.selection, 11);
        assert_eq!(site.step, Some(0));
        assert_eq!(site.operator, Some(operator));
        assert_eq!(site.layer, Some(7));
        // CacheSuffix's short-rank panic occurs before input bounds diagnostics.
        let input_bounds = if missing || (operator == Op::CacheSuffix && shape.len() < 3) {
            None
        } else {
            let bounds = actual.bounds(&hidden.words).unwrap();
            Some((bounds.lower(), bounds.upper()))
        };
        assert_eq!(site.inputs, [input_bounds, None]);
        assert_eq!(actual.admission_stage, "operator_lowering");
    }
}

// Test-local expanded encodings; production finalization remains unchanged.
fn hash_affine(h: &mut Sha256, left: usize, right: Option<usize>, c: Affine) {
    h.update([2]);
    h.update((left as u64).to_le_bytes());
    h.update(right.map_or(u64::MAX, |v| v as u64).to_le_bytes());
    for v in [c.left, c.right, c.constant] {
        h.update(v.to_le_bytes());
    }
}

fn hash_product(
    h: &mut Sha256,
    left: usize,
    right: usize,
    shape: Option<MatrixShape>,
    rescale: bool,
    norm: Option<&puma::multiply::norm::RowEnergyProof>,
) {
    // A certified complete dot binds its obligations as well as operand owners.
    h.update([if norm.is_some() { 13 } else { 3 }, u8::from(rescale)]);
    h.update((left as u64).to_le_bytes());
    h.update((right as u64).to_le_bytes());
    for n in shape.map_or([0, 0, 0], MatrixShape::dimensions) {
        h.update((n as u64).to_le_bytes());
    }
    if let Some(proof) = norm {
        h.update(proof.digest());
    }
}
