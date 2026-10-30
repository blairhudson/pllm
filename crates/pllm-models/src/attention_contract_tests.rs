use super::*;

fn baseline(index: usize) -> (ModelOperation, BTreeMap<String, Vec<u64>>) {
    let (operator, left, right, group_size, output) = match index {
        0 => (
            ModelOperator::AttentionScores,
            vec![1, 3, 2, 5],
            vec![1, 3, 7, 5],
            1,
            vec![1, 3, 2, 7],
        ),
        1 => (
            ModelOperator::AttentionScores,
            vec![1, 6, 3, 5],
            vec![1, 2, 3, 4, 5],
            3,
            vec![1, 6, 3, 4],
        ),
        2 => (
            ModelOperator::AttentionValues,
            vec![1, 4, 1, 9],
            vec![1, 1, 9, 7],
            4,
            vec![1, 4, 1, 7],
        ),
        3 => (
            ModelOperator::AttentionValues,
            vec![1, 6, 2, 5],
            vec![1, 2, 2, 5, 7],
            3,
            vec![1, 6, 2, 7],
        ),
        _ => unreachable!(),
    };
    (
        ModelOperation {
            id: "attention_contract".into(),
            operator,
            layer: None,
            state_kind: None,
            inputs: vec!["left".into(), "right".into()],
            output_shape: output,
            attributes: json!({"group_size": group_size}),
        },
        BTreeMap::from([("left".into(), left), ("right".into(), right)]),
    )
}

fn check(operation: &ModelOperation, shapes: &BTreeMap<String, Vec<u64>>) -> bool {
    [DecoderMode::Prefill, DecoderMode::Decode]
        .into_iter()
        .all(|mode| validate_operation(mode, operation, shapes).is_ok())
}

fn reject(operation: &ModelOperation, shapes: &BTreeMap<String, Vec<u64>>) {
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        assert!(validate_operation(mode, operation, shapes).is_err());
    }
}

#[test]
fn accepts_full_mha_scores_with_distinct_query_and_key_extents() {
    let (operation, shapes) = baseline(0);
    assert!(check(&operation, &shapes));
}

#[test]
fn accepts_window_gqa_scores() {
    let (operation, shapes) = baseline(1);
    assert!(check(&operation, &shapes));
}

#[test]
fn accepts_full_mqa_values_with_independent_value_feature_width() {
    let (operation, shapes) = baseline(2);
    assert!(check(&operation, &shapes));
}

#[test]
fn accepts_window_gqa_values_with_independent_value_feature_width() {
    let (operation, shapes) = baseline(3);
    assert!(check(&operation, &shapes));
}

#[test]
fn rejects_score_output_key_extent_and_feature_mismatch() {
    for index in [0, 1] {
        let (mut operation, shapes) = baseline(index);
        operation.output_shape[3] += 1;
        reject(&operation, &shapes);

        let (operation, mut shapes) = baseline(index);
        shapes.get_mut("left").unwrap()[3] += 1;
        reject(&operation, &shapes);
    }
}

#[test]
fn rejects_values_probability_contraction_mismatch() {
    for index in [2, 3] {
        let (operation, mut shapes) = baseline(index);
        shapes.get_mut("left").unwrap()[3] += 1;
        reject(&operation, &shapes);
    }
}

#[test]
fn rejects_window_query_mismatch() {
    for index in [1, 3] {
        let (operation, mut shapes) = baseline(index);
        shapes.get_mut("right").unwrap()[2] += 1;
        reject(&operation, &shapes);
    }
}

#[test]
fn rejects_missing_zero_noninteger_and_inconsistent_groups() {
    for index in 0..4 {
        for attributes in [
            json!({}),
            json!({"group_size": 0}),
            json!({"group_size": 2}),
            json!({"group_size": 1.5}),
            json!({"group_size": "3"}),
            json!({"group_size": -1}),
            json!({"group_size": null}),
            json!({"group_size": u64::MAX}),
        ] {
            let (mut operation, shapes) = baseline(index);
            operation.attributes = attributes;
            reject(&operation, &shapes);
        }
    }
}

#[test]
fn rejects_nondivisible_and_zero_heads() {
    for index in 0..4 {
        for (query_heads, kv_heads) in [(5, 2), (0, 2), (6, 0), (0, 0)] {
            let (mut operation, mut shapes) = baseline(index);
            shapes.get_mut("left").unwrap()[1] = query_heads;
            shapes.get_mut("right").unwrap()[1] = kv_heads;
            operation.output_shape[1] = query_heads;
            operation.attributes = json!({"group_size": 2});
            reject(&operation, &shapes);
        }
    }
}

#[test]
fn rejects_batch_output_axes_and_input_output_rank_mismatches() {
    for index in 0..4 {
        let (operation, mut shapes) = baseline(index);
        shapes.get_mut("right").unwrap()[0] += 1;
        reject(&operation, &shapes);

        for axis in 0..4 {
            let (mut operation, shapes) = baseline(index);
            operation.output_shape[axis] += 1;
            reject(&operation, &shapes);
        }

        for input in ["left", "right"] {
            let (operation, mut shapes) = baseline(index);
            shapes.get_mut(input).unwrap().truncate(3);
            reject(&operation, &shapes);
        }

        let (mut operation, shapes) = baseline(index);
        operation.output_shape.pop();
        reject(&operation, &shapes);
    }
}

#[test]
fn accepts_large_exact_head_ratio_without_multiplication() {
    let (mut operation, mut shapes) = baseline(0);
    shapes.get_mut("left").unwrap()[1] = u64::MAX;
    shapes.get_mut("right").unwrap()[1] = 1;
    operation.output_shape[1] = u64::MAX;
    operation.attributes = json!({"group_size": u64::MAX});
    assert!(check(&operation, &shapes));
}
