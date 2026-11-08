//! Original-expanded controls only; none qualifies compact request execution.
use super::*;
use crate::replicated_decoder::tests::complete_decoder_parity_tests::{
    checked_case, early_rejection_case, nonzero_case, run_expanded, three_selection_case,
    DecoderCase,
};

fn config(family: &str) -> Vec<u8> {
    // Same metadata as the existing shared complete-decoder fixture. Numeric
    // sources, authenticated input descriptors and expected IDs come from it.
    serde_json::to_vec(&serde_json::json!({
        "model_type": family, "hidden_size": 2, "intermediate_size": 2,
        "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
        "head_dim": 2, "vocab_size": 3, "tie_word_embeddings": false,
        "hidden_act": "silu", "attention_bias": family == "qwen2",
        "max_position_embeddings": 64, "rope_theta": 10000,
        "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
        "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
    }))
    .unwrap()
}

/// Traversal-only spy: no executable Program, numeric result or work authority.
struct Traversal {
    expected: Vec<DecoderPlan>,
    histories: Vec<Vec<(usize, usize)>>,
    next: usize,
    result_lanes: usize,
}

impl RequestSink for Traversal {
    type View = Vec<(usize, usize)>;
    type Selected = Vec<(usize, usize)>;

    fn lanes(view: &Self::View) -> usize {
        view.len()
    }

    fn prefill(
        &mut self,
        plan: &DecoderPlan,
        tokens: &Self::View,
        position: usize,
        state: &mut BTreeMap<(u64, pllm_models::StateKind), Self::View>,
        site: &mut AdmissionSite,
    ) -> Result<Self::View, Error> {
        assert_eq!(plan, &self.expected[self.next]);
        assert_eq!(position, 0);
        assert!(state.is_empty(), "semantic KV must not survive selection");
        assert_eq!(site.selection, self.next);
        assert!(site.step.is_none() && site.operator.is_none() && site.layer.is_none());
        assert_eq!(site.inputs, [None; 2]);
        assert_eq!(site.complete_work, None);
        // Deliberately poison state/context. The driver must replace both at
        // the next selection, while leaving sink-owned numeric state intact.
        state.insert((0, pllm_models::StateKind::Key), tokens.clone());
        site.step = Some(123);
        site.operator = Some(Op::KvCacheAppend);
        self.next += 1;
        Ok(std::iter::repeat_n((70 + self.next, 9), self.result_lanes).collect())
    }

    fn append_feedback(
        &mut self,
        tokens: &mut Self::View,
        result: &Self::View,
        selected: &mut Self::Selected,
    ) -> Result<(), Error> {
        selected.extend(result);
        tokens.extend(result);
        Ok(())
    }

    fn boundary(&self, selection: usize, id: &str, view: &Self::View) {
        assert_eq!(
            view,
            &self.histories[2 * selection + usize::from(id == "feedback.history")]
        );
    }
}

#[test]
fn traversal_reuses_exact_lowerer_fresh_prefill_state_and_original_coordinate_order() {
    for family in ["qwen2", "qwen3", "llama"] {
        for weights in [WeightMode::Private, WeightMode::Public] {
            for outputs in [1, 3] {
                let config = config(family);
                let prompt = vec![(3, 8), (3, 2)];
                let mut history = prompt.clone();
                let mut histories = vec![];
                for selection in 0..outputs {
                    histories.push(history.clone());
                    history.push((71 + selection, 9));
                    histories.push(history.clone());
                }
                let mut sink = Traversal {
                    expected: (0..outputs)
                        .map(|selection| {
                            lower_with_weights(&config, 2 + selection, outputs - selection, weights)
                                .unwrap()
                        })
                        .collect(),
                    histories,
                    next: 0,
                    result_lanes: 1,
                };
                let mut tokens = prompt;
                let selected = drive(
                    &mut sink,
                    &config,
                    2,
                    outputs,
                    weights,
                    &mut tokens,
                    &mut AdmissionSite::default(),
                )
                .unwrap();
                assert_eq!(sink.next, outputs);
                assert_eq!(
                    selected,
                    (0..outputs).map(|i| (71 + i, 9)).collect::<Vec<_>>()
                );
                assert_eq!(tokens, history);
            }
        }
    }
}

#[test]
fn nonscalar_selection_rejected_before_feedback_publication() {
    let config = config("qwen2");
    for result_lanes in [0, 2] {
        let prompt = vec![(3, 8), (3, 2)];
        let mut sink = Traversal {
            expected: vec![lower_with_weights(&config, 2, 2, WeightMode::Public).unwrap()],
            histories: vec![prompt.clone()],
            next: 0,
            result_lanes,
        };
        let mut tokens = prompt.clone();
        assert_eq!(
            drive(
                &mut sink,
                &config,
                2,
                2,
                WeightMode::Public,
                &mut tokens,
                &mut AdmissionSite::default()
            ),
            Err(Error::Shape)
        );
        assert_eq!(tokens, prompt);
        assert_eq!(sink.next, 1);
    }
}

#[test]
fn semantic_lowering_failure_preserves_binding_error_and_selection_context() {
    let mut sink = Traversal {
        expected: vec![],
        histories: vec![],
        next: 0,
        result_lanes: 1,
    };
    let prompt = vec![(3, 8), (3, 2)];
    let mut tokens = prompt.clone();
    let mut site = AdmissionSite {
        selection: 99,
        step: Some(123),
        ..AdmissionSite::default()
    };
    assert_eq!(
        drive(
            &mut sink,
            b"{}",
            2,
            2,
            WeightMode::Public,
            &mut tokens,
            &mut site
        ),
        Err(Error::Binding)
    );
    assert_eq!(site.selection, 0);
    assert_eq!(site.step, None);
    assert_eq!(sink.next, 0);
    assert_eq!(tokens, prompt);
}

/// Borrow metadata from existing genuine expanded fixture. Reconstruct only the
/// source descriptor map required by original graph lowering; verify its binding
/// against the fixture before using it. No new numeric fixture or source bytes.
fn sources(case: &DecoderCase, config: &[u8]) -> BTreeMap<String, SourceTensor> {
    let selections = case.expected.len() - usize::from(case.first_validity.is_some());
    let sources: BTreeMap<_, _> = Program::source_shapes(config, 2, selections)
        .unwrap()
        .into_iter()
        .map(|(key, shape)| {
            let values = &case.clear[&key];
            let imported = case.program.nodes[..case.program.inputs].iter().find(
                |node| matches!(&node.code, Code::PublicInput { key: name, .. } if name == &key),
            );
            let source = if let Some(node) = imported {
                let Code::PublicInput {
                    digest,
                    row_l1,
                    row_squared_norm,
                    row_l1_squared_sum,
                    ..
                } = &node.code
                else {
                    unreachable!()
                };
                SourceTensor {
                    shape,
                    lower: node.bounds.lower(),
                    upper: node.bounds.upper(),
                    public_q18: None,
                    digest: *digest,
                    row_l1: *row_l1,
                    row_squared_norm: *row_squared_norm,
                    row_l1_squared_sum: *row_l1_squared_sum,
                }
            } else {
                assert!(key.contains("norm"));
                SourceTensor {
                    shape,
                    lower: *values.iter().min().unwrap(),
                    upper: *values.iter().max().unwrap(),
                    public_q18: Some(values.clone()),
                    digest: Sha256::digest(
                        values
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .into(),
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                }
            };
            (key, source)
        })
        .collect();
    assert_eq!(source_binding(&sources), case.program.source_binding);
    sources
}

fn original_imports(
    case: &DecoderCase,
    sources: &BTreeMap<String, SourceTensor>,
) -> (Builder, BTreeMap<String, SourceView>, usize) {
    let checked = case.first_validity.is_some();
    let mut builder = Builder {
        public_weights: true,
        checked_bound: checked.then_some(ONE),
        ..Builder::default()
    };
    let mut inputs = BTreeMap::new();
    let mut token_input = None;
    for (owner, node) in case.program.nodes[..case.program.inputs].iter().enumerate() {
        assert_eq!(
            builder
                .push(node.code.clone(), node.lanes, node.bounds)
                .unwrap(),
            owner
        );
        match &node.code {
            Code::PublicInput { key, .. } => {
                inputs.insert(
                    key.clone(),
                    SourceView {
                        owner,
                        shape: sources[key].shape.clone(),
                    },
                );
            }
            Code::Input(key) if key == "input.tokens" => token_input = Some(owner),
            _ => panic!("unexpected fixture import"),
        }
    }
    let input = token_input.unwrap();
    (builder, inputs, input)
}

/// Real original graph failures, observed through the extracted driver. The
/// second-prefix faults consume no replacement numeric/Program authority.
struct FailingExpanded<'a> {
    original: Expanded<'a>,
    pressure: bool,
    visits: Vec<usize>,
    published: Vec<(usize, usize)>,
    boundaries: std::cell::RefCell<Vec<(usize, String)>>,
    failure: Option<(String, String, usize)>,
}

impl RequestSink for FailingExpanded<'_> {
    type View = View;
    type Selected = Vec<(usize, usize)>;

    fn lanes(view: &View) -> usize {
        view.words.len()
    }

    fn prefill(
        &mut self,
        plan: &DecoderPlan,
        tokens: &View,
        position: usize,
        state: &mut BTreeMap<(u64, pllm_models::StateKind), View>,
        site: &mut AdmissionSite,
    ) -> Result<View, Error> {
        self.visits.push(site.selection);
        assert_eq!(position, 0);
        assert!(state.is_empty());
        if site.selection != 1 {
            return self.original.prefill(plan, tokens, position, state, site);
        }
        let result = if self.pressure {
            // The next original gather must exceed the ordinary retained cap.
            self.original.builder.allocated = MAX_PROGRAM_BYTES;
            self.original.builder.admission_stage = "later_prefix_budget";
            self.original.prefill(plan, tokens, position, state, site)
        } else {
            // Missing original source binding fails at a real semantic operator,
            // after the graph's original anchor/constant publication.
            self.original.builder.graph(
                &plan.prefill,
                tokens,
                &BTreeMap::new(),
                self.original.sources,
                self.original.vocabulary,
                position,
                state,
                site,
            )
        };
        assert!(result.is_err(), "second original prefill must fail");
        self.failure = Some((
            site.to_string(),
            self.original.builder.admission_ledger(),
            self.original.builder.nodes.len(),
        ));
        result
    }

    fn append_feedback(
        &mut self,
        tokens: &mut View,
        result: &View,
        selected: &mut Self::Selected,
    ) -> Result<(), Error> {
        self.published.extend(result.words.iter());
        self.original.append_feedback(tokens, result, selected)
    }

    fn boundary(&self, selection: usize, id: &str, _view: &View) {
        self.boundaries.borrow_mut().push((selection, id.into()));
    }
}

#[test]
fn later_original_prefill_errors_preserve_site_ledger_and_stop_before_feedback_or_finish() {
    use std::cell::Cell;

    let case = three_selection_case();
    let config = config("qwen2");
    let sources = sources(&case, &config);
    for pressure in [false, true] {
        let (mut builder, inputs, input) = original_imports(&case, &sources);
        let mut tokens = builder.view(input, vec![1, 2]);
        let mut site = AdmissionSite::default();
        let mut sink = FailingExpanded {
            original: Expanded {
                builder: &mut builder,
                inputs: &inputs,
                sources: &sources,
                vocabulary: 3,
            },
            pressure,
            visits: vec![],
            published: vec![],
            boundaries: Default::default(),
            failure: None,
        };
        let finishes = Cell::new(0);
        let result = (|| -> Result<(), Error> {
            let _selected = drive(
                &mut sink,
                &config,
                2,
                3,
                WeightMode::Public,
                &mut tokens,
                &mut site,
            )?;
            finishes.set(finishes.get() + 1);
            panic!("failed original prefill reached finalization/publication boundary")
        })();
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("original second prefix must stop request"),
        };
        assert_eq!(
            error,
            if pressure {
                Error::Bounds
            } else {
                Error::Binding
            }
        );
        assert_eq!(sink.visits, [0, 1]); // Third prefix never starts.
        assert_eq!(
            *sink.boundaries.borrow(),
            [
                (0, "input.tokens".into()),
                (0, "feedback.history".into()),
                (1, "input.tokens".into())
            ]
        );
        assert_eq!(sink.published.len(), 1); // Failed prefix never appends feedback.
        assert_eq!(
            tokens.words.iter().collect::<Vec<_>>(),
            vec![(input, 0), (input, 1), sink.published[0]]
        );
        assert_eq!(tokens.shape, [1, 3]);
        assert_eq!(finishes.get(), 0);
        let (failure_site, failure_ledger, failure_end) = sink.failure.as_ref().unwrap();
        assert_eq!(&site.to_string(), failure_site);
        assert_eq!(&sink.original.builder.admission_ledger(), failure_ledger);
        assert_eq!(sink.original.builder.nodes.len(), *failure_end);
        assert_eq!(site.selection, 1);
        assert_eq!(site.complete_work, None);
        assert!(!sink.original.builder.nodes.is_empty()); // No consuming finish.
        if pressure {
            assert!(site.step.is_none() && site.operator.is_none());
            assert_eq!(sink.original.builder.admission_stage, "later_prefix_budget");
            assert_eq!(sink.original.builder.admission_limit, Some("program_bytes"));
            assert!(sink.original.builder.requested_bytes.unwrap() > MAX_PROGRAM_BYTES);
        } else {
            assert_eq!(site.step, Some(0));
            assert_eq!(site.operator, Some(Op::TokenLookup));
            assert_eq!(sink.original.builder.admission_stage, "operator_lowering");
        }

        // Independent pre-extraction loop with the SAME genuine fault. Compare
        // full public diagnostics, partial owner schedule, work and history—not
        // just an injected sentinel returned by a mock prefill.
        let (mut control, control_inputs, control_input) = original_imports(&case, &sources);
        let mut control_tokens = control.view(control_input, vec![1, 2]);
        let mut control_site = AdmissionSite::default();
        let mut control_selected = vec![];
        let control_result = (|| -> Result<(), Error> {
            for selection in 0..3 {
                control_site = AdmissionSite {
                    selection,
                    ..AdmissionSite::default()
                };
                let plan =
                    lower_with_weights(&config, 2 + selection, 3 - selection, WeightMode::Public)
                        .map_err(|_| Error::Binding)?;
                if pressure && selection == 1 {
                    control.allocated = MAX_PROGRAM_BYTES;
                    control.admission_stage = "later_prefix_budget";
                }
                let absent = BTreeMap::new();
                let inputs = if !pressure && selection == 1 {
                    &absent
                } else {
                    &control_inputs
                };
                let result = control.graph(
                    &plan.prefill,
                    &control_tokens,
                    inputs,
                    &sources,
                    3,
                    0,
                    &mut BTreeMap::new(),
                    &mut control_site,
                )?;
                assert_eq!(result.words.len(), 1);
                control_selected.extend(result.words.iter());
                control_tokens.words.extend(result.words.iter());
                control_tokens.shape = vec![1, control_tokens.words.len()];
            }
            Ok(())
        })();
        assert_eq!(control_result, Err(error));
        assert_eq!(control_site.to_string(), site.to_string());
        assert_eq!(
            control.admission_ledger(),
            sink.original.builder.admission_ledger()
        );
        assert_eq!(control_selected, sink.published);
        assert_eq!(control_tokens.words, tokens.words);
        assert_eq!(control.nodes.len(), sink.original.builder.nodes.len());
        assert_eq!(
            schedule_work(&control.nodes).unwrap(),
            schedule_work(&sink.original.builder.nodes).unwrap()
        );
    }
}

#[test]
fn checked_range_program_metadata_refusal_preserves_bounds_and_partial_graph() {
    // Tiny public envelopes, not a reconstruction of any historical request.
    // Use the existing retained-byte pressure pattern; allocate no large graph.
    let incoming = Interval::new(-12, 12).unwrap();
    let accepted = Interval::new(-9, 9).unwrap();
    let mut builder = Builder {
        checked_bound: Some(9),
        ..Builder::default()
    };
    let input = builder
        .push(Code::Input("range.input".into()), 3, incoming)
        .unwrap();
    let words: Words = (0..3).map(|lane| (input, lane)).collect();

    // A successful real constructor distinguishes metadata admission from a
    // numeric-constructor failure. Measure the preceding ordinary gather too.
    let program = tensor::Program::checked_range(3, incoming, accepted, 0).unwrap();
    let metadata_bytes = program.admitted_bytes().checked_add(1024).unwrap();
    let mut probe = Builder::default();
    probe
        .push(Code::Input("range.input".into()), 3, incoming)
        .unwrap();
    let before_gather = probe.allocated;
    probe.gather(&words).unwrap();
    let gather_bytes = probe.allocated - before_gather;
    assert_eq!(MAX_PROGRAM_BYTES, 64 * 1024 * 1024);
    // One tile's view allowance and gather fit; metadata exceeds the unchanged
    // cap by one byte. Only the accounting is pressured, not actual allocation.
    builder.allocated = MAX_PROGRAM_BYTES
        .checked_sub(Words::span_pool_bytes(1).unwrap() * 2 + gather_bytes + metadata_bytes)
        .unwrap()
        + 1;

    assert!(matches!(builder.checked_view(&words), Err(Error::Bounds)));
    assert_eq!(builder.admission_stage, "checked_range_program_metadata");
    assert_eq!(builder.admission_limit, Some("program_bytes"));
    let requested = builder.requested_bytes.unwrap();
    assert_eq!(requested, MAX_PROGRAM_BYTES + 1);
    assert!(requested > MAX_PROGRAM_BYTES);
    assert_eq!(builder.allocated + metadata_bytes, requested);
    let ledger = builder.admission_ledger();
    assert!(ledger.contains("stage: checked_range_program_metadata"));
    assert!(ledger.contains("limit: Some(\"program_bytes\")"));
    assert!(ledger.contains(&format!("requested_bytes: Some({requested})")));

    // Refusal leaves the original input and emitted gather inspectable, without
    // interning the rejected program or publishing checked outputs/validity.
    assert_eq!(builder.nodes.len(), 2);
    assert!(matches!(&builder.nodes[input].code, Code::Input(key) if key == "range.input"));
    assert_eq!(builder.nodes[input].bounds, incoming);
    assert_eq!(builder.bounds(&words).unwrap(), incoming);
    assert!(matches!(&builder.nodes[1].code, Code::Gather(gathered) if gathered == &words));
    assert_eq!(builder.nodes[1].bounds, incoming);
    assert!(builder.tensors.is_empty());
    assert_eq!(builder.validity, None);
}

#[test]
fn checked_final_acceptance_budget_failure_retains_sticky_site_and_never_publishes() {
    use std::cell::Cell;

    // Borrowed original canonical metadata only: no consuming finalizer or
    // replacement Program. Compare every recipe, coordinate and work charge.
    let config = config("qwen2");
    let final_plan = lower_with_weights(&config, 3, 1, WeightMode::Public).unwrap();
    let (last_step, last_op) = final_plan
        .prefill
        .operations
        .iter()
        .enumerate()
        .next_back()
        .unwrap();
    let mut fixed_prefix_work = None;
    for reject_prompt in [false, true] {
        // Both genuine checked fixtures complete every prefix. Early private
        // rejection changes neither construction nor the sticky guard schedule.
        let case = early_rejection_case(reject_prompt);
        let sources = sources(&case, &config);
        let Code::Product {
            left: prefix_end,
            rescale: false,
            ..
        } = case.program.nodes[case.program.outputs[0].0].code
        else {
            panic!("original final acceptance mask required")
        };
        let Code::Gather(expected_selected) = &case.program.nodes[prefix_end].code else {
            panic!("original ordered selected-ID gather required")
        };
        // Fail either the first acceptance gather or the mask product after
        // both real acceptance gathers have succeeded. Never inject an error.
        for acceptance_gathers in [0, 2] {
            let mut oracle = None;
            for extracted in [false, true] {
                let (mut builder, inputs, input) = original_imports(&case, &sources);
                let mut tokens = builder.view(input, vec![1, 2]);
                let mut site = AdmissionSite::default();
                let mut selected = if extracted {
                    drive(
                        &mut Expanded {
                            builder: &mut builder,
                            inputs: &inputs,
                            sources: &sources,
                            vocabulary: 3,
                        },
                        &config,
                        2,
                        2,
                        WeightMode::Public,
                        &mut tokens,
                        &mut site,
                    )
                    .unwrap()
                } else {
                    // Independent original loop, including its final site and
                    // exact private feedback aliases, before the same real cap.
                    let mut selected = vec![];
                    for selection in 0..2 {
                        site = AdmissionSite {
                            selection,
                            ..AdmissionSite::default()
                        };
                        let plan = lower_with_weights(
                            &config,
                            2 + selection,
                            2 - selection,
                            WeightMode::Public,
                        )
                        .unwrap();
                        let result = builder
                            .graph(
                                &plan.prefill,
                                &tokens,
                                &inputs,
                                &sources,
                                3,
                                0,
                                &mut BTreeMap::new(),
                                &mut site,
                            )
                            .unwrap();
                        assert_eq!(result.words.len(), 1);
                        selected.extend(result.words.iter());
                        tokens.words.extend(result.words.iter());
                        tokens.shape = vec![1, tokens.words.len()];
                    }
                    selected
                };
                let original_selected = selected.clone();
                let history = tokens.words.clone();
                assert_eq!(selected, expected_selected.iter().collect::<Vec<_>>());
                assert_eq!(tokens.shape, [1, 4]);
                assert_eq!(
                    tokens.words.iter().collect::<Vec<_>>(),
                    [(input, 0), (input, 1)]
                        .into_iter()
                        .chain(selected.iter().copied())
                        .collect::<Vec<_>>()
                );
                assert_eq!(builder.nodes.len(), prefix_end);
                let prefix = canonical(&builder.nodes, &selected);
                assert_eq!(
                    prefix,
                    canonical(&case.program.nodes[..prefix_end], &selected)
                );
                if let Some(work) = fixed_prefix_work {
                    assert_eq!(prefix.1, work);
                } else {
                    fixed_prefix_work = Some(prefix.1);
                }
                let valid = builder.validity.expect("real sticky guards required");
                assert_eq!(valid, case.program.outputs[2].0);
                let mut ancestor = valid;
                while ancestor != case.first_validity.unwrap() {
                    let Code::Product {
                        left,
                        rescale: false,
                        ..
                    } = builder.nodes[ancestor].code
                    else {
                        panic!("later guards must retain first-prefix sticky validity")
                    };
                    assert!(left < ancestor);
                    ancestor = left;
                }
                assert!(ancestor < case.feedback.0 && case.feedback.0 < valid);
                assert_eq!(site.selection, 1);
                assert_eq!(site.step, Some(last_step));
                assert_eq!(site.operator, Some(last_op.operator));
                assert_eq!(site.layer, last_op.layer);
                assert!(site.inputs[0].is_some());
                assert_eq!(site.complete_work, None);
                let prefix_site = site.to_string();
                let finishes = Cell::new(0);
                let result = (|| -> Result<(), Error> {
                    if extracted {
                        // Leave exactly the original successful gathers' ordinary
                        // charges available before calling production acceptance.
                        // The same real push then fails, without an internal hook
                        // or a second acceptance algorithm on the driver path.
                        let gather_bytes: usize = case.program.nodes
                            [prefix_end..prefix_end + acceptance_gathers]
                            .iter()
                            .map(|node| {
                                let Code::Gather(words) = &node.code else {
                                    panic!("original acceptance gather required")
                                };
                                words.metadata_bytes()
                                    + std::mem::size_of::<Node>()
                                    + 16 * node.lanes
                            })
                            .sum();
                        builder.allocated = MAX_PROGRAM_BYTES - gather_bytes;
                        builder.final_acceptance(input, 2, &mut selected)?;
                    } else {
                        // Independent original checked epilogue. Only the retained
                        // counter changes; numeric recipes and policy stay fixed.
                        builder.admission_stage = "final_acceptance";
                        let valid = builder.validity.ok_or(Error::Binding)?;
                        if acceptance_gathers == 0 {
                            builder.allocated = MAX_PROGRAM_BYTES;
                        }
                        let token_node = builder.gather(&selected.iter().copied().collect())?;
                        let flags = std::iter::repeat_n((valid, 0), 2).collect();
                        let flag_node = builder.gather(&flags)?;
                        if acceptance_gathers == 2 {
                            builder.allocated = MAX_PROGRAM_BYTES;
                        }
                        let masked = builder.product(token_node, flag_node, None, false)?;
                        selected = (0..2).map(|i| (masked, i)).collect();
                        selected.push((valid, 0));
                    }
                    finishes.set(finishes.get() + 1);
                    panic!("failed original acceptance reached finalization/publication boundary")
                })();
                assert_eq!(
                    match result {
                        Err(error) => error,
                        Ok(_) => panic!("real acceptance budget must reject"),
                    },
                    Error::Bounds
                );
                assert_eq!(finishes.get(), 0);
                assert_eq!(selected, original_selected); // No masked output publication.
                assert_eq!(tokens.words, history);
                assert_eq!(tokens.shape, [1, 4]);
                assert_eq!(builder.validity, Some(valid));
                assert_eq!(site.to_string(), prefix_site); // No finish-time reset.
                assert_eq!(site.complete_work, None);
                assert_eq!(builder.admission_stage, "final_acceptance");
                assert_eq!(builder.admission_limit, Some("program_bytes"));
                assert_eq!(builder.allocated, MAX_PROGRAM_BYTES);
                assert!(builder.requested_bytes.unwrap() > MAX_PROGRAM_BYTES);
                let end = prefix_end + acceptance_gathers;
                assert_eq!(builder.nodes.len(), end); // Failed node never published.
                let partial = canonical(&builder.nodes, &selected);
                assert_eq!(partial, canonical(&case.program.nodes[..end], &selected));
                let observation = (site.to_string(), builder.admission_ledger(), partial);
                if let Some(expected) = &oracle {
                    assert_eq!(&observation, expected);
                } else {
                    oracle = Some(observation);
                }
            }
        }
    }
}

/// Test-only pre-extraction loop oracle, using original Builder::graph. Compare
/// only the real request prefix; checked acceptance remains the production
/// epilogue. This does not construct a replacement Program or compact compiler.
fn assert_original_loop_parity(case: &DecoderCase) {
    let config = config("qwen2");
    let sources = sources(case, &config);
    let checked = case.first_validity.is_some();
    let selections = case.expected.len() - usize::from(checked);
    let (mut builder, inputs, input) = original_imports(case, &sources);
    let mut tokens = builder.view(input, vec![1, 2]);
    let mut selected = vec![];
    for selection in 0..selections {
        let plan = lower_with_weights(
            &config,
            2 + selection,
            selections - selection,
            WeightMode::Public,
        )
        .unwrap();
        let result = builder
            .graph(
                &plan.prefill,
                &tokens,
                &inputs,
                &sources,
                3,
                0,
                &mut BTreeMap::new(),
                &mut AdmissionSite {
                    selection,
                    ..AdmissionSite::default()
                },
            )
            .unwrap();
        assert_eq!(result.words.len(), 1);
        selected.extend(result.words.iter());
        tokens.words.extend(result.words.iter());
        tokens.shape = vec![1, tokens.words.len()];
    }
    let (prefix_end, actual_selected) = if checked {
        let Code::Product {
            left,
            rescale: false,
            ..
        } = case.program.nodes[case.program.outputs[0].0].code
        else {
            panic!("original final acceptance mask required")
        };
        let Code::Gather(words) = &case.program.nodes[left].code else {
            panic!("original ordered selected-ID gather required")
        };
        assert_eq!(builder.validity, Some(case.program.outputs[selections].0));
        (left, words.iter().collect::<Vec<_>>())
    } else {
        (case.program.nodes.len(), case.program.outputs.clone())
    };
    assert_eq!(selected, actual_selected);
    assert_eq!(selected[0], case.feedback);
    assert_eq!(
        tokens.words.iter().collect::<Vec<_>>(),
        [(input, 0), (input, 1)]
            .into_iter()
            .chain(selected.iter().copied())
            .collect::<Vec<_>>()
    );
    assert_eq!(builder.nodes.len(), prefix_end);

    // Independent original borrowed encoder checks EVERY original node recipe,
    // bounds, template digest, logical gather coordinate, complete prefix work
    // and retirement. Existing frozen whole-program graph snapshots separately
    // constrain production identity, complete work and final checked epilogue.
    let expected = canonical(&builder.nodes, &selected);
    let actual = canonical(&case.program.nodes[..prefix_end], &selected);
    assert_eq!(actual, expected);

    let lookup_coordinates: Vec<_> = case.program.nodes[..prefix_end]
        .iter()
        .filter_map(|node| match node.code {
            Code::TensorLoop {
                input: LoopInput::TokenOffsets { owner, index },
                ..
            } => Some((owner, index)),
            _ => None,
        })
        .collect();
    let expected_lookups: Vec<_> = (0..selections)
        .flat_map(|selection| {
            [(input, 0), (input, 1)]
                .into_iter()
                .chain(selected[..selection].iter().copied())
        })
        .collect();
    assert_eq!(lookup_coordinates, expected_lookups);
}

#[test]
fn original_expanded_request_canonical_selected_order_and_full_prefix_parity() {
    assert_original_loop_parity(&nonzero_case([0, 0]));
    assert_original_loop_parity(&checked_case(false));
    assert_original_loop_parity(&checked_case(true));
    assert_original_loop_parity(&three_selection_case());
    assert_original_loop_parity(&early_rejection_case(false));
    assert_original_loop_parity(&early_rejection_case(true));
}

#[test]
fn original_expanded_full_prefix_frames_components_sticky_validity_and_work() {
    // Genuine complete-decoder helpers verify ordered wire, both private share
    // components at feedback boundaries, full PRSS cursors and final frames.
    // Ordinary-versus-ordinary control only, NOT compact execution evidence.
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            for case in [
                nonzero_case([0, 0]),
                nonzero_case([1, 0]),
                checked_case(false),
                checked_case(true),
                three_selection_case(),
            ] {
                let trace = run_expanded(&case);
                assert_eq!(trace.outputs, case.expected);
                assert_eq!(trace.work, case.program.work_budget());
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Independent borrowed copy of the pre-extraction expanded node encoder and
/// last-use calculation. No Program publication, native cloning/construction,
/// replacement identity prefix, admission token or compact finalizer lives here.
type CanonicalSchedule = ([u8; 32], WorkBudget, Vec<Vec<usize>>);

fn canonical(nodes: &[Node], selected: &[(usize, usize)]) -> CanonicalSchedule {
    let mut last: Vec<_> = (0..nodes.len()).collect();
    let work = schedule_work(nodes).unwrap();
    let mut h = Sha256::new();
    for (id, node) in nodes.iter().enumerate() {
        for input in node.inputs() {
            last[input] = id;
        }
        h.update((node.lanes as u64).to_le_bytes());
        h.update(node.bounds.lower().to_le_bytes());
        h.update(node.bounds.upper().to_le_bytes());
        match &node.code {
            Code::Input(key) => {
                h.update([0]);
                h.update((key.len() as u64).to_le_bytes());
                h.update(key);
            }
            Code::PublicInput { key, digest, .. } => {
                h.update([5]);
                h.update((key.len() as u64).to_le_bytes());
                h.update(key);
                h.update(digest);
            }
            Code::PublicValues(values) => {
                h.update([7]);
                for n in values {
                    h.update(n.to_le_bytes());
                }
            }
            Code::PublicProduct {
                input,
                weight,
                rows,
                inner,
                columns,
                transposed,
            } => {
                h.update([6, u8::from(*transposed)]);
                for n in [input, weight, rows, inner, columns] {
                    h.update((*n as u64).to_le_bytes());
                }
            }
            Code::PublicLookup {
                input,
                weight,
                plan,
            } => {
                h.update([10]);
                for n in [*input, *weight] {
                    h.update((n as u64).to_le_bytes());
                }
                h.update(plan.digest());
            }
            Code::Selection { input, plan } => {
                h.update([11]);
                h.update((*input as u64).to_le_bytes());
                h.update(plan.digest());
            }
            Code::TrieLookup {
                input,
                weight,
                plan,
            } => {
                h.update([12]);
                h.update((*input as u64).to_le_bytes());
                h.update((*weight as u64).to_le_bytes());
                h.update(plan.digest());
            }
            Code::CheckedRange { input, program } => {
                h.update([9]);
                h.update((*input as u64).to_le_bytes());
                h.update(program.digest());
            }
            Code::Gather(items) => {
                h.update([1]);
                for (a, b) in items.iter() {
                    h.update((a as u64).to_le_bytes());
                    h.update((b as u64).to_le_bytes());
                }
            }
            Code::SourceGather(view) => {
                h.update([1]);
                for index in view.indices() {
                    h.update((view.owner as u64).to_le_bytes());
                    h.update((index as u64).to_le_bytes());
                }
            }
            Code::Affine {
                left,
                right,
                coefficients: c,
            } => {
                h.update([2]);
                h.update((*left as u64).to_le_bytes());
                h.update(right.map_or(u64::MAX, |v| v as u64).to_le_bytes());
                for v in [c.left, c.right, c.constant] {
                    h.update(v.to_le_bytes());
                }
            }
            Code::Product {
                left,
                right,
                shape,
                rescale,
                norm,
            } => {
                h.update([if norm.is_some() { 13 } else { 3 }, u8::from(*rescale)]);
                h.update((*left as u64).to_le_bytes());
                h.update((*right as u64).to_le_bytes());
                for n in shape.map_or([0, 0, 0], MatrixShape::dimensions) {
                    h.update((n as u64).to_le_bytes());
                }
                if let Some(proof) = norm {
                    h.update(proof.digest());
                }
            }
            Code::Tensor { input, program } => {
                h.update([4]);
                h.update((*input as u64).to_le_bytes());
                h.update(program.digest());
            }
            Code::TensorLoop {
                input,
                program,
                tail,
            } => {
                h.update([8]);
                match input {
                    LoopInput::Slice(id) => {
                        h.update([0]);
                        h.update((*id as u64).to_le_bytes());
                    }
                    LoopInput::TokenOffsets { owner, index } => {
                        h.update([1]);
                        h.update((*owner as u64).to_le_bytes());
                        h.update((*index as u64).to_le_bytes());
                    }
                }
                h.update(program.digest());
                h.update([tail.is_some() as u8]);
                if let Some(tail) = tail {
                    h.update(tail.digest());
                }
            }
        }
    }
    for &(id, index) in selected {
        last[id] = nodes.len();
        h.update((id as u64).to_le_bytes());
        h.update((index as u64).to_le_bytes());
    }
    let mut retire = vec![vec![]; nodes.len()];
    for (id, end) in last.into_iter().enumerate() {
        if end < retire.len() {
            retire[end].push(id);
        }
    }
    (h.finalize().into(), work, retire)
}
