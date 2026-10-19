use super::*;
use puma::{prss::PublicContribution, Party};
use std::collections::BTreeMap;

#[test]
fn largest_live_prefix_views_reject_before_source_values() {
    let config = br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":8192,
        "num_hidden_layers":4,"num_attention_heads":1,"num_key_value_heads":1,
        "vocab_size":5,"max_position_embeddings":64,"rope_theta":10000,
        "rms_norm_eps":0.000001,"tie_word_embeddings":true,"hidden_act":"silu",
        "attention_bias":true,"head_dim":2,"attention_dropout":0.0,
        "use_sliding_window":false,"use_cache":true,"max_window_layers":4}"#;
    Program::source_shapes(config, 1, 1).unwrap();
    Program::source_shapes(config, 9, 1).unwrap();
    Program::source_shapes(config, 9, 8).unwrap();
    assert!(Program::source_shapes(config, 32, 1)
        .unwrap_err()
        .contains("compiler view payload limit"));
}

#[test]
fn pinned_qwen_public_metadata_prices_live_frontier_without_source_values() {
    let replay: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../paper/artifacts/puma-sdk/local-v4/qwen-metadata-replay.json"
    ))
    .unwrap();
    let config = replay["normalized_config"].as_str().unwrap().as_bytes();
    Program::source_shapes_with_weights(config, 4, 2, WeightMode::Public).unwrap();
    let reason =
        Program::source_shapes_with_weights(config, 16, 2, WeightMode::Public).unwrap_err();
    assert!(reason.contains("public lookup operation count"), "{reason}");
    assert!(!reason.contains("compiler view"));
    assert!(Program::source_shapes_with_weights(config, 4, 2, WeightMode::Private).is_err());
    assert!(
        Publisher::new_with_weights(config, 16, 2, [1; 32], [2; 32], WeightMode::Public)
            .err()
            .unwrap()
            .contains("public lookup operation count")
    );
    // The larger workload exceeds the original sequence ceiling before import.
}

fn fixture(family: &str) -> (Arc<Program>, BTreeMap<String, Vec<i64>>) {
    fixture_with_body(family, false)
}

fn fixture_with_body(family: &str, active: bool) -> (Arc<Program>, BTreeMap<String, Vec<i64>>) {
    fixture_with_weights(family, active, WeightMode::Private)
}

fn fixture_with_weights(
    family: &str,
    active: bool,
    mode: WeightMode,
) -> (Arc<Program>, BTreeMap<String, Vec<i64>>) {
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type": family, "hidden_size": 2, "intermediate_size": 2,
        "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
        "head_dim": 2, "vocab_size": 3, "tie_word_embeddings": false,
        "hidden_act": "silu", "attention_bias": family == "qwen2",
        "max_position_embeddings": 64, "rope_theta": 10000,
        "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
        "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
    }))
    .unwrap();
    let shapes = Program::source_shapes(&config, 2, 2).unwrap();
    let mut source = BTreeMap::new();
    let mut clear = BTreeMap::new();
    for (key, shape) in shapes {
        let values = if active && key == "model.embed_tokens.weight" {
            vec![ONE / 4, ONE / 4, -ONE / 4, -ONE / 4, 0, 0]
        } else if key == "model.embed_tokens.weight" {
            vec![ONE, ONE, ONE, 0, -ONE, 0]
        } else if key == "lm_head.weight" {
            vec![ONE, 0, -ONE, 0, 0, 0]
        } else if key.contains("norm") {
            vec![ONE; shape.iter().product()]
        } else if active && key.ends_with("v_proj.weight") {
            vec![ONE, 0, 0, ONE]
        } else if active && key.ends_with("o_proj.weight") {
            vec![-ONE / 2, 0, 0, -ONE / 2]
        } else if active && key.contains("mlp.") {
            vec![ONE / 64; shape.iter().product()]
        } else {
            vec![0; shape.iter().product()]
        };
        let digest: [u8; 32] = Sha256::digest(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .into();
        source.insert(
            key.clone(),
            SourceTensor {
                shape,
                lower: *values.iter().min().unwrap(),
                upper: *values.iter().max().unwrap(),
                public_q18: key.contains("norm").then(|| values.clone()),
                digest,
            },
        );
        clear.insert(key, values);
    }
    clear.insert("input.tokens".into(), vec![1, 2]);
    let program = Program::compile_with_weights(&config, 2, 2, &source, mode).unwrap();
    (Arc::new(program), clear)
}

#[test]
fn compact_storage_preserves_frozen_full_prefix_execution_identities() {
    // Captured before this representation change from the installed local-v2
    // quality wheel (native SHA256 29c8b78201352a51348431adb728db33f1
    // 716c21dd8e8ac1332b3be4111e4deb), with these exact source/config values.
    // The source adapter retains epsilon spelling; use serde's `1e-6` rather
    // than Python JSON's `1e-06` when replaying this fixture through the binding.
    for (family, active, expected) in [
        (
            "qwen2",
            false,
            "333aa62a92ec54bdfb694e6bc356d3b4d873629d0cbee5bbbbaa3fb447d28c6d",
        ),
        (
            "qwen2",
            true,
            "4529a304429056c986676ea4b8da1adf4cf864e23e41f40a0778a32268b671a1",
        ),
        (
            "qwen3",
            false,
            "83886ad38965f9244ba95b403855fb56c6298f44392c44e022b400b19118f7d2",
        ),
        (
            "qwen3",
            true,
            "98044e61b32ee4cde96ee63a0c0accc70a776b51b689ef763930353aea60d45e",
        ),
        (
            "llama",
            false,
            "f9b5f1dcc3c06ac87013adca91751f7acae25e5993fae0137b0b90ca98c08eca",
        ),
        (
            "llama",
            true,
            "9e9593392a6058d6dac04ecb442ec83a31d6e017291e5007144ac91731e4ce8c",
        ),
    ] {
        let (program, _) = fixture_with_body(family, active);
        let mut program = Arc::try_unwrap(program).ok().unwrap();
        program.bind([7; 32], [9; 32]);
        let actual: String = program
            .digest()
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect();
        assert_eq!(actual, expected, "{family} {active}");
    }
}

#[test]
fn public_weights_keep_prefix_and_feedback_private_with_original_final_id_contract() {
    for family in ["qwen2", "qwen3", "llama"] {
        for active in [false, true] {
            let (program, mut clear) = fixture_with_weights(family, active, WeightMode::Public);
            let (private, _) = fixture_with_body(family, active);
            assert_ne!(program.digest(), private.digest());
            assert_eq!(program.inputs().collect::<Vec<_>>(), [("input.tokens", 2)]);
            assert!(program
                .nodes
                .iter()
                .any(|n| matches!(n.code, Code::PublicProduct { .. })));
            for tokens in [vec![1, 2], vec![0, 2]] {
                clear.insert("input.tokens".into(), tokens);
                // Fixtures have clear sign margins. Fresh source stochastic
                // rounding may differ internally; only final IDs are exported.
                let expected = execute(workers(private.clone(), &clear));
                assert_eq!(
                    execute(workers(program.clone(), &clear)),
                    expected,
                    "{family} {active}"
                );
            }
        }
    }
}

#[test]
fn shared_public_tensor_programs_keep_distinct_invocations_and_complete_work() {
    let (program, clear) = fixture("qwen2");
    let mut unique: BTreeMap<[u8; 32], &Arc<tensor::Program>> = BTreeMap::new();
    let mut invocations = 0;
    for node in &program.nodes {
        if let Code::Tensor { program, .. } = &node.code {
            invocations += 1;
            if let Some(previous) = unique.insert(program.digest(), program) {
                assert!(Arc::ptr_eq(previous, program));
            }
        }
    }
    assert!(invocations > unique.len());
    let expected_operations: u64 = program
        .nodes
        .iter()
        .map(|node| match &node.code {
            Code::Product { .. } => 1,
            Code::Tensor { program, .. } => program.work_budget().operations(),
            _ => 0,
        })
        .sum();
    assert_eq!(program.operation_limit(), expected_operations);
    assert_eq!(
        program.work_budget(),
        schedule_work(&program.nodes).unwrap()
    );
    assert_eq!(execute(workers(program, &clear)), [1, 0]);
}

pub(super) fn workers(program: Arc<Program>, clear: &BTreeMap<String, Vec<i64>>) -> [Worker; 3] {
    workers_with_programs([program.clone(), program.clone(), program], clear)
}

pub(super) fn workers_with_programs(
    programs: [Arc<Program>; 3],
    clear: &BTreeMap<String, Vec<i64>>,
) -> [Worker; 3] {
    let mut setups: [_; 3] = std::array::from_fn(|i| {
        programs[i]
            .setup([42; 32], [Party::P0, Party::P1, Party::P2][i])
            .unwrap()
    });
    let public = setups.each_mut().map(|p| p.contribution().unwrap());
    for (i, p) in setups.iter_mut().enumerate() {
        for (j, v) in public.iter().enumerate() {
            if i != j {
                p.receive_public(
                    PublicContribution::decode(&v.encode(), [Party::P0, Party::P1, Party::P2][j])
                        .unwrap(),
                )
                .unwrap();
            }
        }
    }
    let mut seeds = setups
        .each_mut()
        .map(|p| Some(p.share_seed().unwrap().encode().unwrap()));
    for (i, p) in setups.iter_mut().enumerate() {
        let from = [Party::P0, Party::P1, Party::P2][(i + 1) % 3];
        let seed = seeds[(i + 1) % 3].take().unwrap();
        p.receive_seed(
            puma::prss::SeedMessage::decode(
                seed.as_bytes(),
                from,
                [Party::P0, Party::P1, Party::P2][i],
            )
            .unwrap(),
        )
        .unwrap();
    }
    let mut workers: [_; 3] = std::array::from_fn(|i| {
        let public = PublicSource::test_values(&programs[i], clear);
        Worker::new_with_source(
            programs[i].clone(),
            setups[i].finish().unwrap(),
            [42; 32],
            public,
        )
        .unwrap()
    });
    for (key, n) in programs[0].inputs() {
        let x = &clear[key];
        assert_eq!(x.len(), n);
        let first: Vec<_> = (0..n)
            .map(|i| 0xc751f00d12345678u64.wrapping_add(i as u64 * 17))
            .collect();
        let second: Vec<_> = (0..n)
            .map(|i| 0x254277811928f93du64.wrapping_add(i as u64 * 311))
            .collect();
        let third: Vec<_> = x
            .iter()
            .zip(first.iter().zip(&second))
            .map(|(x, (a, b))| (*x as u64).wrapping_sub(*a).wrapping_sub(*b))
            .collect();
        let shares = [first, second, third];
        for (i, w) in workers.iter_mut().enumerate() {
            w.import(key, &shares[i], &shares[(i + 1) % 3]).unwrap();
        }
    }
    workers
}

pub(super) fn execute(mut workers: [Worker; 3]) -> Vec<u64> {
    let mut queue = vec![];
    let mut rounds = 0;
    while !workers.iter().all(Worker::complete) {
        if queue.is_empty() {
            for w in &mut workers {
                if !w.complete() {
                    let from = w.party();
                    queue.extend(
                        w.advance()
                            .unwrap()
                            .into_iter()
                            .map(|(to, v)| (from, to, v)),
                    );
                }
            }
        } else {
            let current = std::mem::take(&mut queue);
            for (from, to, frame) in current {
                queue.extend(
                    workers[to as usize]
                        .receive(frame.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, v)| (to, next, v)),
                );
            }
        }
        rounds += 1;
        assert!(rounds < 100_000);
    }
    assert!(queue.is_empty());
    let outputs = workers.map(|w| w.finish().unwrap());
    (0..outputs[0].len())
        .map(|i| {
            outputs[0][i]
                .wrapping_add(outputs[1][i])
                .wrapping_add(outputs[2][i])
        })
        .collect()
}

#[test]
fn complete_protected_prefill_feedback_and_final_ids_cross_encoded_party_boundaries() {
    for family in ["qwen2", "qwen3", "llama"] {
        let (program, clear) = fixture(family);
        let result = execute(workers(program, &clear));
        // Zero body projections preserve the final input embedding. Q18 RMS
        // normalization cannot change its sign. The head compares ±x[0], with
        // index 2 fixed at zero. Token 2 -> 1, then token 1 -> 0.
        assert_eq!(result, vec![1, 0], "{family}");
    }
}

#[test]
fn every_selection_reexecutes_original_private_tokens_and_consumes_fresh_work() {
    let (program, _) = fixture("qwen2");
    let input = program
        .nodes
        .iter()
        .position(|n| matches!(&n.code, Code::Input(key) if key == "input.tokens"))
        .unwrap();
    for row in 0..2 {
        // The equality -> one-hot lookup gathers each original token once per
        // selection. Incremental KV execution would gather it only once.
        let count = program
            .nodes
            .iter()
            .filter(|n| matches!(&n.code, Code::Gather(words) if words == &vec![(input, row); 3]))
            .count();
        assert_eq!(count, 2);
    }
    let feedback = program.outputs[0];
    assert!(program
        .nodes
        .iter()
        .any(|n| matches!(&n.code, Code::Gather(words) if words == &vec![feedback; 3])));
    // Every input/source and temporary can retire; only final token shares
    // survive to the client-only reveal.
    for id in 0..program.nodes.len() {
        if !program.outputs.iter().any(|&(out, _)| out == id) {
            assert!(program.retire.iter().any(|ids| ids.contains(&id)));
        }
    }
}

#[test]
fn nonzero_attention_and_mlp_preserve_private_prefix_dependence() {
    for family in ["qwen2", "qwen3", "llama"] {
        let (program, mut clear) = fixture_with_body(family, true);
        // Independent sign oracle: Q=K=0 gives uniform attention, V is identity,
        // O=-I/2. At the final row x - mean(norm(prefix))/2 is negative for
        // [positive,positive] and positive for [negative,positive]. The small
        // positive gated MLP cannot cross either margin. Full-prefix feedback
        // then gives [+,+,-] -> negative and [-,+,+] -> positive. These margins
        // exceed 1/16 real unit, including stochastic rescale error.
        clear.insert("input.tokens".into(), vec![0, 0]);
        assert_eq!(
            execute(workers(program.clone(), &clear)),
            vec![1, 1],
            "{family}"
        );
        clear.insert("input.tokens".into(), vec![1, 0]);
        assert_eq!(execute(workers(program, &clear)), vec![0, 0], "{family}");
    }
}

#[test]
fn malformed_input_and_peer_frames_retire_complete_worker_owner() {
    let (program, clear) = fixture("qwen2");
    let mut owners = workers(program, &clear);
    let mut out = vec![];
    while out.is_empty() {
        for owner in &mut owners {
            let from = owner.party();
            out.extend(
                owner
                    .advance()
                    .unwrap()
                    .into_iter()
                    .map(|(to, v)| (from, to, v)),
            );
        }
    }
    let (from, to, frame) = out.into_iter().next().unwrap();
    let mut bad = frame.as_bytes().to_vec();
    bad[0] ^= 1;
    assert!(owners[to as usize].receive(&bad, from).is_err());
    assert!(owners[to as usize].retired());
    assert!(owners[to as usize].advance().is_err());
    for owner in &mut owners {
        owner.cancel();
    }
}
