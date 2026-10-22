use super::*;
use crate::replicated_decoder::tests::workers;

fn fixture(family: &str, outputs: usize) -> (Publisher, [Worker; 3]) {
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type":family,"hidden_size":2,"intermediate_size":2,
        "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
        "head_dim":2,"vocab_size":3,"tie_word_embeddings":false,
        "hidden_act":"silu","attention_bias":family=="qwen2",
        "max_position_embeddings":64,"rope_theta":10000,"rms_norm_eps":0.000001,
        "attention_dropout":0.0,"use_sliding_window":false,"use_cache":true,
        "max_window_layers":1
    }))
    .unwrap();
    let execution = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    let mut publisher = Publisher::new_with_policy(
        &config,
        1,
        outputs,
        [1; 32],
        [2; 32],
        WeightMode::Public,
        execution,
    )
    .unwrap();
    // The existing test-only worker harness binds this fixed session. Production
    // construction continues to sample the session and all source shares freshly.
    publisher.manifest.session = [42; 32];
    let mut clear = BTreeMap::new();
    let shapes = publisher.shapes().to_vec();
    for (key, shape) in shapes {
        let values: Vec<f32> = if key == "model.embed_tokens.weight" {
            vec![0.25, -0.25, 1.5, -1.5, 0.75, -0.75]
        } else if key == "lm_head.weight" {
            vec![-0.25, 0.25, 0.25, -0.25, 0.0, 0.0]
        } else if key.contains("norm") {
            vec![1.0; shape.iter().product()]
        } else {
            vec![0.0; shape.iter().product()]
        };
        let raw: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        publisher.append(&key, 0, &raw).unwrap();
        clear.insert(
            key,
            raw.chunks_exact(4)
                .map(|v| encode_q18(v).unwrap())
                .collect(),
        );
    }
    publisher.seal(&[0]).unwrap();
    let program = publisher.program.as_ref().unwrap().clone();
    assert!(program
        .nodes
        .iter()
        .any(|n| matches!(n.code, Code::CheckedRange { .. })));
    assert_eq!(program.outputs.len(), outputs + 1);
    clear.insert("input.tokens".into(), vec![0]);
    while publisher.next_chunk().unwrap().is_some() {}
    let workers = workers(program, &clear);
    (publisher, workers)
}

fn final_frames(mut workers: [Worker; 3]) -> [Zeroizing<Vec<u8>>; 3] {
    let mut queue = Vec::new();
    for _ in 0..100_000 {
        if workers.iter().all(Worker::complete) {
            assert!(queue.is_empty());
            return workers.map(|w| w.finish_frame().unwrap());
        }
        if queue.is_empty() {
            for worker in &mut workers {
                if !worker.complete() {
                    let from = worker.party();
                    queue.extend(
                        worker
                            .advance()
                            .unwrap()
                            .into_iter()
                            .map(|(to, v)| (from, to, v)),
                    );
                }
            }
        } else {
            for (from, to, frame) in std::mem::take(&mut queue) {
                queue.extend(
                    workers[to as usize]
                        .receive(frame.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, v)| (to, next, v)),
                );
            }
        }
    }
    panic!("checked decoder did not complete");
}

fn opened_words(frames: &[Zeroizing<Vec<u8>>; 3]) -> Vec<u64> {
    (73..frames[0].len())
        .step_by(8)
        .map(|offset| {
            frames.iter().fold(0u64, |sum, frame| {
                sum.wrapping_add(u64::from_le_bytes(
                    frame[offset..offset + 8].try_into().unwrap(),
                ))
            })
        })
        .collect()
}

#[test]
fn checked_request_acceptance_masks_earlier_tokens_on_later_private_feedback_failure() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            for family in ["qwen2", "qwen3", "llama"] {
                // Independent margin oracle: normalization preserves opposite signs;
                // zero body weights preserve the embedding. The head selects token 1.
                let (publisher, workers) = fixture(family, 1);
                let frames = final_frames(workers);
                assert_eq!(opened_words(&frames), [1, 1]);
                assert_eq!(
                    publisher
                        .finish(frames.each_ref().map(|v| v.as_slice()))
                        .unwrap(),
                    [1]
                );

                // Feedback token 1 has ±1.5 activations, beyond the frozen ±1 range.
                // Its protected guard fails only during the SECOND full-prefix pass.
                // Even the already computed first token must be masked before reveal.
                let (publisher, workers) = fixture(family, 2);
                let frames = final_frames(workers);
                assert_eq!(opened_words(&frames), [0, 0, 0]);
                assert!(publisher
                    .finish(frames.each_ref().map(|v| v.as_slice()))
                    .unwrap_err()
                    .contains("failed protected range checks"));
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn checked_request_acceptance_binds_policy_and_rejects_nonbit_validity() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            let (publisher, workers) = fixture("qwen2", 1);
            let manifest = publisher.manifest().unwrap();
            for kind in 0..3 {
                let mut substituted: Manifest = serde_json::from_slice(&manifest).unwrap();
                match kind {
                    0 => substituted.execution.max_abs_activation = 2,
                    1 => substituted.execution.range_policy = RangePolicy::Static,
                    _ => substituted.execution = ExecutionPolicy::default(),
                }
                assert!(substituted.compile().is_err());
            }
            let mut frames = final_frames(workers);
            let offset = frames[0].len() - 8;
            let value = u64::from_le_bytes(frames[0][offset..].try_into().unwrap());
            frames[0][offset..].copy_from_slice(&value.wrapping_add(1).to_le_bytes());
            assert_eq!(opened_words(&frames), [1, 2]);
            assert!(publisher
                .finish(frames.each_ref().map(|v| v.as_slice()))
                .unwrap_err()
                .contains("failed protected range checks"));
        })
        .unwrap()
        .join()
        .unwrap();
}
