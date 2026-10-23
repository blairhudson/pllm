use super::*;
use crate::replicated_decoder::tests::{execute, workers};

fn decoder(family: &str, policy: ExecutionPolicy) -> (Arc<Program>, BTreeMap<String, Vec<i64>>) {
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type":family, "hidden_size":2, "intermediate_size":2,
        "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
        "head_dim":2,"vocab_size":5,"tie_word_embeddings":false,"hidden_act":"silu",
        "attention_bias":family=="qwen2","max_position_embeddings":64,"rope_theta":10000,
        "rms_norm_eps":0.000001,"attention_dropout":0.0,"use_sliding_window":false,
        "use_cache":true,"max_window_layers":1
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
        } else if key.ends_with("v_proj.weight") {
            vec![ONE, 0, 0, ONE]
        } else if key.ends_with("o_proj.weight") {
            vec![-ONE / 2, 0, 0, -ONE / 2]
        } else if key.contains("mlp.") {
            vec![ONE / 64; shape.iter().product()]
        } else {
            vec![0; shape.iter().product()]
        };
        let mut hash = Sha256::new();
        for v in &values {
            hash.update(v.to_le_bytes());
        }
        let digest: [u8; 32] = hash.finalize().into();
        let source: SourceTensor = serde_json::from_value(serde_json::json!({
            "shape":shape,"lower":values.iter().min().unwrap(),"upper":values.iter().max().unwrap(),
            "public_q18":key.contains("norm").then(||values.clone()),"digest":digest
        }))
        .unwrap();
        sources.insert(key.clone(), source);
        clear.insert(key, values);
    }
    let program = Arc::new(
        Program::compile_with_policy(&config, 2, 2, &sources, WeightMode::Public, policy).unwrap(),
    );
    clear.insert("input.tokens".into(), vec![0, 1]);
    (program, clear)
}

#[test]
fn trie_lookup_generated_qwen2_qwen3_llama_complete_decoder_controls_default_stack() {
    for family in ["qwen2", "qwen3", "llama"] {
        for selection in [SelectionPolicy::Expanded, SelectionPolicy::Tiled] {
            let policy = ExecutionPolicy {
                lookup: LookupPolicy::Trie,
                selection,
                ..ExecutionPolicy::default()
            };
            let (program, mut clear) = decoder(family, policy);
            let (legacy, _) = decoder(family, ExecutionPolicy::default());
            assert_ne!(program.digest(), legacy.digest());
            assert_eq!(
                program
                    .nodes
                    .iter()
                    .filter(|n| matches!(n.code, Code::TrieLookup { .. }))
                    .count(),
                2
            );
            for (ids, expected) in [(vec![0, 1], vec![0, 2]), (vec![1, 0], vec![2, 2])] {
                clear.insert("input.tokens".into(), ids);
                assert_eq!(
                    execute(workers(program.clone(), &clear)),
                    expected,
                    "{family} {selection:?}"
                );
                assert_eq!(execute(workers(legacy.clone(), &clear)), expected);
            }
        }
    }
}

#[test]
fn trie_lookup_checked_decoder_retains_final_aggregate_acceptance_contract() {
    let (program, clear) = decoder(
        "qwen2",
        ExecutionPolicy {
            lookup: LookupPolicy::Trie,
            selection: SelectionPolicy::Tiled,
            range_policy: RangePolicy::Checked,
            ..ExecutionPolicy::default()
        },
    );
    assert_eq!(execute(workers(program, &clear)), [0, 2, 1]);
}

fn lookup_builder(rows: usize) -> (Builder, usize, SourceView) {
    let mut b = Builder {
        public_weights: true,
        lookup: LookupPolicy::Trie,
        token_source_binding: [19; 32],
        ..Builder::default()
    };
    let table = b
        .push(
            Code::PublicInput {
                key: "table".into(),
                digest: [23; 32],
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
            131 * 3,
            Interval::new(-9, 9).unwrap(),
        )
        .unwrap();
    let tokens = b
        .push(
            Code::Input("input.tokens".into()),
            rows,
            Interval::new(0, 130).unwrap(),
        )
        .unwrap();
    (
        b,
        tokens,
        SourceView {
            owner: table,
            shape: vec![131, 3],
        },
    )
}

#[test]
fn trie_lookup_rejects_interval_only_and_copied_or_substituted_token_references() {
    let (mut b, input, table) = lookup_builder(9);
    let tokens = b.view(input, vec![9]);
    assert!(b.trie_lookup(&tokens, &table, &[9, 3]).is_err());
    b.register_initial_tokens(&tokens.words, 131).unwrap();
    let copy = b.gather(&tokens.words).unwrap();
    assert!(b
        .trie_lookup(&b.view(copy, vec![9]), &table, &[9, 3])
        .is_err());
    assert!(b
        .register_initial_tokens(&b.view(copy, vec![9]).words, 131)
        .is_err());
    let output = b.trie_lookup(&tokens, &table, &[9, 3]).unwrap();
    assert_eq!(output.words.len(), 27);
    let lookups: Vec<_> = b
        .nodes
        .iter()
        .filter_map(|n| {
            if let Code::TrieLookup {
                input,
                weight,
                plan,
            } = &n.code
            {
                assert_eq!(*weight, table.owner);
                assert_eq!(n.bounds, Interval::new(-9, 9).unwrap());
                assert!(matches!(b.nodes[*input].code, Code::Gather(_)));
                Some(plan.geometry())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(lookups, [(8, 131, 3), (1, 131, 3)]);
    assert!(b.tensors.is_empty());
    assert!(
        b.allocated
            > runtime::trie::Plan::work_for_geometry(8, 131)
                .unwrap()
                .operations() as usize
    );
    let forged = SourceView {
        owner: input,
        shape: table.shape.clone(),
    };
    assert!(b.trie_lookup(&tokens, &forged, &[9, 3]).is_err());
}

#[test]
fn trie_lookup_floor_native_budget_reconciles_without_provenance_fabrication() {
    let work = public_lookup_work_floor_with_policy(4, 2, 151936, LookupPolicy::Trie).unwrap();
    assert_eq!(work.operations(), 10707);
    assert_eq!(
        (
            work.cursors().public_blocks,
            work.cursors().pair_blocks,
            work.cursors().private_blocks,
            work.cursors().aes_work
        ),
        (2, 683981, 0, 1367802)
    );
    let large = public_lookup_work_floor_with_policy(64, 8, 151936, LookupPolicy::Trie).unwrap();
    assert_eq!(large.operations(), 641966);
    assert_eq!(
        (
            large.cursors().public_blocks,
            large.cursors().pair_blocks,
            large.cursors().private_blocks,
            large.cursors().aes_work
        ),
        (135, 41038412, 0, 82067239)
    );
}

#[test]
fn trie_lookup_resource_peak_is_charged_once_without_tensor_template_duplicates() {
    let mut b = Builder {
        public_weights: true,
        lookup: LookupPolicy::Trie,
        token_source_binding: [19; 32],
        ..Builder::default()
    };
    let weight = b
        .push(
            Code::PublicInput {
                key: "table".into(),
                digest: [23; 32],
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
            151936 * 896,
            Interval::new(-9, 9).unwrap(),
        )
        .unwrap();
    let input = b
        .push(
            Code::Input("input.tokens".into()),
            8,
            Interval::new(0, 151935).unwrap(),
        )
        .unwrap();
    let tokens = b.view(input, vec![8]);
    b.register_initial_tokens(&tokens.words, 151936).unwrap();
    let table = SourceView {
        owner: weight,
        shape: vec![151936, 896],
    };
    b.trie_lookup(&tokens, &table, &[8, 896]).unwrap();
    let Code::TrieLookup { plan, .. } = &b.nodes.last().unwrap().code else {
        unreachable!()
    };
    let peak = plan.peak_extra_bytes().unwrap();
    assert!(b.allocated >= peak);
    assert!(b.allocated < 32 << 20);
    let previous = b.allocated;
    b.trie_lookup(&tokens, &table, &[8, 896]).unwrap();
    assert!(b.allocated - previous < 8 * 896 * 16 + 8192);
    assert!(b.tensors.is_empty());
}

fn row_program(alias: bool) -> (Arc<Program>, BTreeMap<String, Vec<i64>>, Vec<u64>) {
    let values: Vec<i64> = (0..131 * 3).map(|i| i as i64 * 19 - 900).collect();
    let mut h = Sha256::new();
    for v in &values {
        h.update(v.to_le_bytes());
    }
    let mut b = Builder {
        public_weights: true,
        lookup: LookupPolicy::Trie,
        token_source_binding: [19; 32],
        ..Builder::default()
    };
    let weight = b
        .push(
            Code::PublicInput {
                key: "weights".into(),
                digest: h.finalize().into(),
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
            values.len(),
            Interval::new(*values.first().unwrap(), *values.last().unwrap()).unwrap(),
        )
        .unwrap();
    let input = b
        .push(
            Code::Input("input.tokens".into()),
            9,
            Interval::new(0, 130).unwrap(),
        )
        .unwrap();
    let tokens = View {
        words: (0..9).rev().map(|i| (input, i)).collect(),
        shape: vec![9],
    };
    b.register_initial_tokens(&tokens.words, 131).unwrap();
    let output = b
        .trie_lookup(
            &tokens,
            &SourceView {
                owner: weight,
                shape: vec![131, 3],
            },
            &[9, 3],
        )
        .unwrap();
    let mut outputs: Vec<_> = output.words.iter().collect();
    outputs.push((input, 8)); // Original token owner stays live and must not move.
    if alias {
        let dedicated = b
            .nodes
            .iter()
            .find_map(|n| {
                if let Code::TrieLookup { input, .. } = n.code {
                    Some(input)
                } else {
                    None
                }
            })
            .unwrap();
        outputs.push((dedicated, 0)); // Deliberately violates dedicated last-use contract.
    }
    let mut last: Vec<_> = (0..b.nodes.len()).collect();
    for (step, node) in b.nodes.iter().enumerate() {
        for id in node.inputs() {
            last[id] = step;
        }
    }
    for &(id, _) in &outputs {
        last[id] = b.nodes.len();
    }
    let mut retire = vec![vec![]; b.nodes.len()];
    for (id, step) in last.into_iter().enumerate() {
        if step < retire.len() {
            retire[step].push(id);
        }
    }
    let ids: Vec<i64> = vec![130, 0, 1, 130, 16, 17, 128, 129, 0];
    let mut expected: Vec<u64> = ids
        .iter()
        .rev()
        .flat_map(|&id| {
            values[id as usize * 3..id as usize * 3 + 3]
                .iter()
                .map(|&v| v as u64)
        })
        .collect();
    expected.push(ids[8] as u64);
    let work = schedule_work(&b.nodes).unwrap();
    let program = Arc::new(Program {
        nodes: b.nodes,
        inputs: 2,
        outputs,
        retire,
        digest: [63; 32],
        source_binding: [19; 32],
        request_custody: None,
        payload_bytes: b.allocated,
        work,
    });
    (
        program,
        BTreeMap::from([("weights".into(), values), ("input.tokens".into(), ids)]),
        expected,
    )
}

#[test]
fn trie_lookup_integrated_partial_row_tiles_preserve_original_live_input() {
    let (program, clear, expected) = row_program(false);
    assert_eq!(execute(workers(program, &clear)), expected);
}

#[test]
fn trie_lookup_runtime_rejects_aliased_owner_source_substitution_and_malformed_frame() {
    let (aliased, clear, _) = row_program(true);
    for mut worker in workers(aliased, &clear) {
        assert!(worker.advance().is_err());
        assert!(worker.retired());
    }
    let (program, clear, _) = row_program(false);
    let mut owned = Arc::try_unwrap(program).ok().unwrap();
    owned.source_binding = [29; 32];
    for mut worker in workers(Arc::new(owned), &clear) {
        assert!(worker.advance().is_err());
        assert!(worker.retired());
    }
    let (program, clear, _) = row_program(false);
    for mut worker in workers(program, &clear) {
        drop(worker.advance().unwrap());
        assert!(worker.receive(b"malformed", puma::Party::P1).is_err());
        assert!(worker.retired());
    }
}
