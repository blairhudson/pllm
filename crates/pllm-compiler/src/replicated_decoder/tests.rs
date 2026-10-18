use super::*;
use puma::{prss::PublicContribution, Party};
use std::collections::BTreeMap;

fn fixture(family: &str) -> (Arc<Program>, BTreeMap<String, Vec<i64>>) {
    fixture_with_body(family, false)
}

fn fixture_with_body(family: &str, active: bool) -> (Arc<Program>, BTreeMap<String, Vec<i64>>) {
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
    let program = Program::compile(&config, 2, 2, &source).unwrap();
    (Arc::new(program), clear)
}

fn workers(program: Arc<Program>, clear: &BTreeMap<String, Vec<i64>>) -> [Worker; 3] {
    let mut setups = [Party::P0, Party::P1, Party::P2].map(|p| program.setup([42; 32], p).unwrap());
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
    let mut workers = setups
        .each_mut()
        .map(|p| Worker::new(program.clone(), p.finish().unwrap(), [42; 32]).unwrap());
    for (key, n) in program.inputs() {
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

fn execute(mut workers: [Worker; 3]) -> Vec<u64> {
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
