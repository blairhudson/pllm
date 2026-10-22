use super::*;
use crate::replicated_decoder::tests::{execute, workers};

fn finish_builder(builder: Builder, inputs: usize, outputs: Vec<(usize, usize)>) -> Arc<Program> {
    let work = schedule_work(&builder.nodes).unwrap();
    let mut last: Vec<_> = (0..builder.nodes.len()).collect();
    for (step, node) in builder.nodes.iter().enumerate() {
        for id in node.inputs() {
            last[id] = step;
        }
    }
    for &(id, _) in &outputs {
        last[id] = builder.nodes.len();
    }
    let mut retire = vec![vec![]; builder.nodes.len()];
    for (id, step) in last.into_iter().enumerate() {
        if step < retire.len() {
            retire[step].push(id);
        }
    }
    Arc::new(Program {
        nodes: builder.nodes,
        inputs,
        outputs,
        retire,
        digest: [63; 32],
        source_binding: [0; 32],
        payload_bytes: builder.allocated,
        work,
    })
}

fn tournament(values: &[i64], retain_original: bool, alias_gather: bool) -> Arc<Program> {
    let mut builder = Builder {
        public_weights: true,
        selection: SelectionPolicy::Tiled,
        ..Builder::default()
    };
    let input = builder
        .push(
            Code::Input("logits".into()),
            values.len(),
            Interval::new(*values.iter().min().unwrap(), *values.iter().max().unwrap()).unwrap(),
        )
        .unwrap();
    let logits = builder.view(input, vec![values.len()]);
    let output = builder.argmax_tree(&logits).unwrap();
    let mut outputs: Vec<_> = output.words.iter().collect();
    if retain_original {
        outputs.push((input, values.len() - 1));
    }
    if alias_gather {
        let Code::Selection {
            input: dedicated, ..
        } = &builder.nodes[output.words.at(0).0].code
        else {
            unreachable!()
        };
        let alias = builder
            .gather(&std::iter::once((*dedicated, 0)).collect())
            .unwrap();
        outputs.push((alias, 0));
    }
    assert!(builder.allocated > values.len() * 32); // Original and dedicated copy both charged.
    finish_builder(builder, 1, outputs)
}

type DecoderFixture = (
    Vec<u8>,
    BTreeMap<String, SourceTensor>,
    Arc<Program>,
    BTreeMap<String, Vec<i64>>,
);

fn decoder_fixture(selection: SelectionPolicy, lookup: LookupPolicy) -> DecoderFixture {
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type":"qwen2", "hidden_size":2, "intermediate_size":2,
        "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
        "head_dim":2, "vocab_size":5, "tie_word_embeddings":false,
        "hidden_act":"silu", "attention_bias":true, "max_position_embeddings":64,
        "rope_theta":10000, "rms_norm_eps":0.000001, "attention_dropout":0.0,
        "use_sliding_window":false, "use_cache":true, "max_window_layers":1
    }))
    .unwrap();
    let mut sources = BTreeMap::new();
    let mut clear = BTreeMap::new();
    for (key, shape) in
        Program::source_shapes_with_weights(&config, 2, 2, WeightMode::Public).unwrap()
    {
        let values = if key == "model.embed_tokens.weight" {
            vec![
                ONE / 4,
                ONE / 4,
                -ONE / 4,
                -ONE / 4,
                ONE / 4,
                ONE / 4,
                ONE / 4,
                ONE / 4,
                ONE / 4,
                ONE / 4,
            ]
        } else if key == "lm_head.weight" {
            vec![-ONE, 0, 0, 0, ONE, 0, ONE, 0, -ONE, 0]
        } else if key.contains("norm") {
            vec![ONE; shape.iter().product()]
        } else {
            vec![0; shape.iter().product()]
        };
        let digest = Sha256::digest(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .into();
        sources.insert(
            key.clone(),
            SourceTensor {
                shape,
                lower: *values.iter().min().unwrap(),
                upper: *values.iter().max().unwrap(),
                public_q18: key.contains("norm").then(|| values.clone()),
                digest,
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
        );
        clear.insert(key, values);
    }
    let policy = ExecutionPolicy {
        selection,
        lookup,
        ..ExecutionPolicy::default()
    };
    let program = Arc::new(
        Program::compile_with_policy(&config, 2, 2, &sources, WeightMode::Public, policy).unwrap(),
    );
    clear.insert("input.tokens".into(), vec![0, 1]);
    (config, sources, program, clear)
}

fn test_stack(test: fn()) {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(test)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn stable_selection_compiler_runtime_preserves_original_live_logits_and_stable_order() {
    test_stack(|| {
        for n in [1, 3, 65, 129, 257] {
            let mut values = vec![-17; n];
            values[n - 1] = 99;
            let program = tournament(&values, true, false);
            assert_eq!(
                program
                    .nodes
                    .iter()
                    .filter(|n| matches!(n.code, Code::Selection { .. }))
                    .count(),
                1
            );
            assert_eq!(
                execute(workers(
                    program,
                    &BTreeMap::from([("logits".into(), values)])
                )),
                [n as u64 - 1, 99]
            );
        }
        let values = vec![5, 5, -1];
        assert_eq!(
            execute(workers(
                tournament(&values, false, false),
                &BTreeMap::from([("logits".into(), values)])
            )),
            [0]
        );
    });
}

#[test]
fn stable_selection_generated_complete_decoder_controls_and_private_feedback() {
    test_stack(|| {
        let (_, _, expanded, mut clear) =
            decoder_fixture(SelectionPolicy::Expanded, LookupPolicy::PerToken);
        assert!(!expanded
            .nodes
            .iter()
            .any(|n| matches!(n.code, Code::Selection { .. })));
        for lookup in [LookupPolicy::PerToken, LookupPolicy::Tiled] {
            let (_, _, tiled, _) = decoder_fixture(SelectionPolicy::Tiled, lookup);
            assert_eq!(
                tiled
                    .nodes
                    .iter()
                    .filter(|n| matches!(n.code, Code::Selection { .. }))
                    .count(),
                2
            );
            assert_ne!(expanded.digest(), tiled.digest());
            for (tokens, expected) in [(vec![0, 1], vec![0, 2]), (vec![1, 0], vec![2, 2])] {
                clear.insert("input.tokens".into(), tokens);
                assert_eq!(execute(workers(expanded.clone(), &clear)), expected);
                assert_eq!(execute(workers(tiled.clone(), &clear)), expected);
            }
        }
        let (config, sources, _, mut clear) =
            decoder_fixture(SelectionPolicy::Tiled, LookupPolicy::PerToken);
        clear.insert("input.tokens".into(), vec![0, 1]);
        let checked = Arc::new(
            Program::compile_with_policy(
                &config,
                2,
                2,
                &sources,
                WeightMode::Public,
                ExecutionPolicy {
                    range_policy: RangePolicy::Checked,
                    selection: SelectionPolicy::Tiled,
                    ..ExecutionPolicy::default()
                },
            )
            .unwrap(),
        );
        assert_eq!(execute(workers(checked, &clear)), [0, 2, 1]);
    });
}

#[test]
fn stable_selection_manifest_selector_and_source_substitution_are_rejected() {
    let (config, sources, program, clear) =
        decoder_fixture(SelectionPolicy::Tiled, LookupPolicy::PerToken);
    let policy = ExecutionPolicy {
        selection: SelectionPolicy::Tiled,
        ..ExecutionPolicy::default()
    };
    let mut bound =
        Program::compile_with_policy(&config, 2, 2, &sources, WeightMode::Public, policy).unwrap();
    bound.bind([1; 32], [2; 32]);
    let mut manifest = Manifest {
        schema: "pllm.replicated-decoder.manifest.v1".into(),
        weights: WeightMode::Public,
        execution: policy,
        config,
        rows: 2,
        outputs: 2,
        source: [1; 32],
        composition: [2; 32],
        session: [3; 32],
        tensors: sources,
        norms: BTreeMap::new(),
        program: bound.digest(),
    };
    assert!(manifest.compile().is_ok());
    assert_eq!(
        serde_json::to_value(&manifest).unwrap()["execution"]["selection"],
        "tiled"
    );
    manifest.execution.selection = SelectionPolicy::Expanded;
    assert!(manifest
        .compile()
        .err()
        .unwrap()
        .contains("program differs"));
    manifest.execution.selection = SelectionPolicy::Tiled;
    manifest.tensors.get_mut("lm_head.weight").unwrap().digest = [77; 32];
    assert!(manifest
        .compile()
        .err()
        .unwrap()
        .contains("program differs"));
    let substituted = Program::compile_with_policy(
        &manifest.config,
        2,
        2,
        &manifest.tensors,
        WeightMode::Public,
        policy,
    )
    .unwrap();
    let original_source = PublicSource::test_values(&program, &clear).unwrap();
    assert_eq!(original_source.check(&substituted), Err(Error::Binding));
    assert!(program.nodes.iter().all(|n| n.work_budget().is_ok()));
}

#[test]
fn stable_selection_worker_rejects_nonexclusive_operand_and_retires_both_phases() {
    test_stack(|| {
        let values = vec![3, 8, 8];
        let clear = BTreeMap::from([("logits".into(), values.clone())]);
        let mut invalid = workers(tournament(&values, false, true), &clear);
        for worker in &mut invalid {
            assert!(worker.advance().is_err());
            assert!(worker.retired());
            assert_eq!(worker.loop_test_state().1, 0);
        }
        for phase in [1, 2] {
            for replay in [false, true] {
                let mut roles = workers(tournament(&values, false, false), &clear);
                let mut queue = Vec::new();
                for _ in 0..1000 {
                    if queue.is_empty() {
                        for role in &mut roles {
                            let from = role.party();
                            queue.extend(
                                role.advance()
                                    .unwrap()
                                    .into_iter()
                                    .map(|(to, frame)| (from, to, frame)),
                            );
                        }
                    } else if roles
                        .iter()
                        .all(|r| r.selection_test_phase() == Some(phase))
                    {
                        break;
                    } else {
                        for (from, to, frame) in std::mem::take(&mut queue) {
                            queue.extend(
                                roles[to as usize]
                                    .receive(frame.as_bytes(), from)
                                    .unwrap()
                                    .into_iter()
                                    .map(|(next, frame)| (to, next, frame)),
                            );
                        }
                    }
                }
                assert!(roles
                    .iter()
                    .all(|r| r.selection_test_phase() == Some(phase)));
                let (from, to, frame) = queue.first().unwrap();
                let target = &mut roles[*to as usize];
                if replay {
                    drop(target.receive(frame.as_bytes(), *from).unwrap());
                    assert!(target.receive(frame.as_bytes(), *from).is_err());
                } else {
                    assert!(target.receive(b"bad", *from).is_err());
                }
                assert!(target.retired());
                assert_eq!(target.loop_test_state().1, 0);
                drop(queue);
                for role in &mut roles {
                    role.cancel();
                    assert_eq!(role.loop_test_state().1, 0);
                }
            }
        }
    });
}
