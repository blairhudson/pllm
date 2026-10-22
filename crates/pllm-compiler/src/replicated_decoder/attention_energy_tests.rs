use super::*;
use crate::replicated_decoder::tests::{execute, workers};

fn program(builder: Builder, inputs: usize, output: Words) -> Arc<Program> {
    let work = schedule_work(&builder.nodes).unwrap();
    Arc::new(Program {
        retire: vec![vec![]; builder.nodes.len()],
        nodes: builder.nodes,
        inputs,
        outputs: output.iter().collect(),
        digest: [113; 32],
        source_binding: [0; 32],
        payload_bytes: builder.allocated,
        work,
    })
}

#[test]
fn attention_norm_admits_complete_dots_and_preserves_integer_oracle_and_work() {
    let width = 64;
    let columns = 19;
    let big = 1i64 << 30;
    let mut x = vec![0; width];
    x[3] = big;
    let mut keys = vec![0; columns * width];
    for column in 0..columns {
        let feature = if column == columns - 1 {
            3
        } else {
            column % width
        };
        keys[column * width + feature] = if column % 2 == 0 { big } else { -big };
    }
    let bounds = Interval::new(-big, big).unwrap();
    assert!(InputBounds {
        left: bounds,
        right: bounds
    }
    .dot_output(width)
    .is_err());
    for rescale in [false, true] {
        let mut builder = Builder::default();
        let left = builder
            .push(Code::Input("q".into()), width, bounds)
            .unwrap();
        let right = builder
            .push(Code::Input("k".into()), keys.len(), bounds)
            .unwrap();
        let q = builder.view(left, vec![1, width]);
        let k = builder.view(right, vec![columns, width]);
        builder
            .record_attention_energy(&q.words, (big as u128).pow(2))
            .unwrap();
        for row in k.words.chunks_exact(width) {
            builder
                .record_attention_energy(&row, (big as u128).pow(2))
                .unwrap();
        }
        // Ordinary compiler tiling crosses the 1,024-word operand boundary and
        // retains a nonzero final key in the incomplete final tile.
        let output = builder
            .matrix_with_attention(&q, MatrixView::Values(&k), rescale, true)
            .unwrap();
        assert_eq!(
            builder
                .nodes
                .iter()
                .filter(|n| matches!(n.code, Code::Product { norm: Some(_), .. }))
                .count(),
            2
        );
        let program = program(builder, 2, output.words);
        let actual = execute(workers(
            program.clone(),
            &BTreeMap::from([("q".into(), x.clone()), ("k".into(), keys.clone())]),
        ));
        for (column, actual) in actual.into_iter().enumerate() {
            let raw: i128 = (0..width)
                .map(|i| i128::from(x[i]) * i128::from(keys[column * width + i]))
                .sum();
            let expected = if rescale { raw / i128::from(ONE) } else { raw };
            assert_eq!(actual, expected as u64);
        }
        let mut ordinary = program.nodes.clone();
        for node in &mut ordinary {
            if let Code::Product { norm, .. } = &mut node.code {
                *norm = None;
            }
        }
        assert_eq!(schedule_work(&ordinary).unwrap(), program.work);
    }

    let mut builder = Builder::default();
    let left = builder
        .push(Code::Input("q".into()), width, bounds)
        .unwrap();
    let right = builder
        .push(Code::Input("k".into()), keys.len(), bounds)
        .unwrap();
    let q = builder.view(left, vec![1, width]);
    let k = builder.view(right, vec![columns, width]);
    builder
        .record_attention_energy(&q.words, (big as u128).pow(2))
        .unwrap();
    for row in k.words.chunks_exact(width).take(columns - 1) {
        builder
            .record_attention_energy(&row, (big as u128).pow(2))
            .unwrap();
    }
    assert!(builder
        .matrix_with_attention(&q, MatrixView::Values(&k), true, true)
        .is_err());
}

#[test]
fn attention_energy_requires_complete_ordered_rows_without_general_norm_side_effects() {
    let mut builder = Builder::default();
    let id = builder
        .push(Code::Input("row".into()), 8, Interval::new(-5, 5).unwrap())
        .unwrap();
    let row: Words = (0..8).map(|i| (id, i)).collect();
    builder.record_attention_energy(&row, 80).unwrap();
    assert_eq!(builder.attention_energy(&row, 8), Some(80));
    assert_eq!(builder.row_norm(&row, 8), None);
    assert_eq!(builder.attention_energy(&row.slice(..4), 4), None);
    assert_eq!(builder.attention_energy(&row.slice(..7), 8), None);
    let reversed: Words = (0..8).rev().map(|i| (id, i)).collect();
    let other: Words = (0..8).map(|i| (id + 1, i)).collect();
    assert_eq!(builder.attention_energy(&reversed, 8), None);
    assert_eq!(builder.attention_energy(&other, 8), None);
    builder.allocated = MAX_PROGRAM_BYTES;
    assert!(builder.record_attention_energy(&row, 100).is_err());
    assert_eq!(builder.attention_energies.len(), 1);
}

#[test]
fn checked_public_source_norms_are_recomputed_and_source_bound() {
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type":"qwen2", "hidden_size":8, "intermediate_size":8,
        "num_hidden_layers":1, "num_attention_heads":4, "num_key_value_heads":2,
        "vocab_size":3, "tie_word_embeddings":false, "hidden_act":"silu",
        "attention_bias":true, "max_position_embeddings":64, "rope_theta":10000,
        "rms_norm_eps":0.000001, "attention_dropout":0.0, "use_cache":true,
        "use_sliding_window":false, "max_window_layers":1,
    }))
    .unwrap();
    let policy = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        ..ExecutionPolicy::default()
    };
    let mut publisher =
        Publisher::new_with_policy(&config, 2, 2, [1; 32], [2; 32], WeightMode::Public, policy)
            .unwrap();
    let mut frames = Vec::new();
    let shapes = publisher.shapes().to_vec();
    for (key, shape) in shapes {
        let bytes: Vec<_> = (0..shape.iter().product::<usize>())
            .flat_map(|i| {
                let value = if key.contains("norm") {
                    1.0f32
                } else if key == "model.embed_tokens.weight" {
                    (i as f32 % 7.0 - 3.0) / 16.0
                } else {
                    ((i * 7 + key.len()) as f32 % 13.0 - 6.0) / 1024.0
                };
                value.to_le_bytes()
            })
            .collect();
        for (i, chunk) in bytes.chunks(12).enumerate() {
            publisher.append(&key, i * 3, chunk).unwrap();
            frames.push((key.clone(), i * 3, chunk.to_vec()));
        }
    }
    publisher.seal(&[0, 1]).unwrap();
    let raw = publisher.manifest().unwrap();
    let manifest: Manifest = serde_json::from_slice(&raw).unwrap();
    let program = manifest.compile().unwrap();
    assert_eq!(manifest.norms.len(), 4); // Both Q/K weights and their biases.
    assert!(program
        .nodes
        .iter()
        .any(|n| matches!(n.code, Code::Product { norm: Some(_), .. })));
    let mut import = PublicSourceImport::new(&manifest).unwrap();
    for (key, offset, bytes) in &frames {
        import.append(key, *offset, bytes).unwrap();
    }
    let source = import.finish().unwrap();
    source.check(&program).unwrap();

    let serialized: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    for field in ["maximum_row_l1", "maximum_column_l1", "frobenius_squared"] {
        let mut bad = serialized.clone();
        let record = bad["norms"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap();
        record["blocks"][0][field] = serde_json::json!(0);
        let bad: Manifest = serde_json::from_value(bad).unwrap();
        let mut import = PublicSourceImport::new(&bad).unwrap();
        let reason = frames
            .iter()
            .find_map(|(key, offset, bytes)| import.append(key, *offset, bytes).err())
            .unwrap();
        assert!(
            reason.contains("block norm certificate differs"),
            "{reason}"
        );
        assert!(import.finish().is_err());
        if let Ok(mut changed) = Program::compile_with_norms(
            &bad.config,
            bad.rows,
            bad.outputs,
            &bad.tensors,
            bad.weights,
            bad.execution,
            &bad.norms,
        ) {
            changed.bind(bad.source, bad.composition);
            assert_ne!(changed.digest(), program.digest());
            assert!(source.check(&changed).is_err());
        }
    }
    let mut missing: Manifest = serde_json::from_slice(&raw).unwrap();
    let first = missing.norms.keys().next().unwrap().clone();
    missing.norms.remove(&first);
    assert!(missing.compile().is_err());
    assert!(PublicSourceImport::new(&missing).is_err());
    let mut absent: Manifest = serde_json::from_slice(&raw).unwrap();
    absent.norms.clear();
    let mut legacy = Program::compile_with_policy(
        &absent.config,
        absent.rows,
        absent.outputs,
        &absent.tensors,
        absent.weights,
        absent.execution,
    )
    .unwrap();
    legacy.bind(absent.source, absent.composition);
    assert_ne!(legacy.digest(), program.digest());
    assert!(source.check(&legacy).is_err());
}
