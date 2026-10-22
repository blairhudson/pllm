use pllm_compiler::{
    decoder_matrix_inventory_bytes, lower_decoder_matrix_inventory, lower_decoder_runtime_schedule,
    DECODER_MATRIX_INVENTORY_MAX_PLAN_BYTES, DECODER_MATRIX_INVENTORY_SCHEMA_VERSION,
};
use pllm_models::{
    lower_model_json, DecoderGraph, DecoderPlan, DecoderWorkload, ModelOperation, ModelOperator,
};
use pllm_types::{canonical_bytes, digest_bytes};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

fn config(family: &str, tied: bool) -> Value {
    let mut value = json!({
        "model_type": family, "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 32, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": tied
    });
    if family == "qwen3" {
        value["head_dim"] = json!(4);
        value["attention_bias"] = json!(false);
        value["attention_dropout"] = json!(0.0);
        value["rope_scaling"] = Value::Null;
        value["sliding_window"] = Value::Null;
        value["use_sliding_window"] = json!(false);
        value["use_cache"] = json!(true);
        value["max_window_layers"] = json!(2);
        value["layer_types"] = json!(["full_attention", "full_attention"]);
    }
    value
}

fn plan(family: &str, tied: bool) -> DecoderPlan {
    lower_model_json(
        &canonical_bytes(&config(family, tied)),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 3,
            max_new_tokens: 2,
        },
    )
    .unwrap()
}

fn composition() -> Vec<u8> {
    canonical_bytes(&json!({
        "components": {
            "inference": {"component": "pllm/inference", "params": {}},
            "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
            "linear": {"component": "pllm/masked-linear", "params": {}},
            "preparation": {"component": "pllm/model-aware-corrections", "params": {}}
        },
        "model": {"source": "model.fixture"}
    }))
}

#[test]
fn default_runtime_schedule_digest_regression() {
    for (family, expected) in [
        (
            "qwen2",
            "9b2f3bed2dff2d83598bebff5cabf5a22f0ed30f470d8b7d35215d1aaa57719d",
        ),
        (
            "qwen3",
            "b328c0862f8dbec47aa85a692337d74159596d44bd7a9a803cd59dd183fabe3b",
        ),
        (
            "llama",
            "2b3f35981d00f18aeaf95a06d30943a4e6c6cb88e06a8047b22af3bfacdeefac",
        ),
    ] {
        let schedule = lower_decoder_runtime_schedule(&plan(family, true), &composition()).unwrap();
        assert_eq!(schedule.digest().to_string(), expected);
    }
}

fn inventory(plan: &DecoderPlan) -> Value {
    let inventory = lower_decoder_matrix_inventory(plan).unwrap();
    let parsed: Value = serde_json::from_slice(inventory.canonical_bytes()).unwrap();
    assert_eq!(canonical_bytes(&parsed), inventory.canonical_bytes());
    assert_eq!(
        inventory.digest(),
        &digest_bytes(
            DECODER_MATRIX_INVENTORY_SCHEMA_VERSION,
            inventory.canonical_bytes()
        )
    );
    assert_eq!(parsed["model_plan_digest"], json!(plan.digest()));
    assert_eq!(parsed["model_config_digest"], json!(plan.config_digest));
    assert_eq!(
        inventory.semantic_plan_identity(),
        plan.digest().as_sha256()
    );
    assert_eq!(inventory.config_identity(), plan.config_digest.as_sha256());
    assert!(
        inventory.retained_payload_bytes().unwrap()
            >= std::mem::size_of_val(&inventory) + inventory.canonical_bytes().len() + 64
    );
    let bytes_entry = decoder_matrix_inventory_bytes(&canonical_bytes(plan)).unwrap();
    assert_eq!(bytes_entry.canonical_bytes(), inventory.canonical_bytes());
    assert_eq!(bytes_entry.digest(), inventory.digest());
    assert_eq!(
        bytes_entry.semantic_plan_identity(),
        inventory.semantic_plan_identity()
    );
    assert_eq!(bytes_entry.config_identity(), inventory.config_identity());
    parsed
}

#[test]
fn inventory_identity_snapshot_cannot_follow_later_plan_mutation() {
    let mut model = plan("llama", true);
    let snapshot = lower_decoder_matrix_inventory(&model).unwrap();
    let original = model.digest().as_sha256();
    model.prefill.output.push_str(".changed");
    assert_eq!(snapshot.semantic_plan_identity(), original);
    assert_ne!(
        snapshot.semantic_plan_identity(),
        model.digest().as_sha256()
    );
    model.config_digest = digest_bytes("changed-config", b"fixture");
    assert_ne!(snapshot.config_identity(), model.config_digest.as_sha256());
}

fn linear(graph: &mut DecoderGraph) -> &mut ModelOperation {
    graph
        .operations
        .iter_mut()
        .find(|op| op.operator == ModelOperator::Linear)
        .unwrap()
}

fn head(graph: &mut DecoderGraph) -> &mut ModelOperation {
    graph
        .operations
        .iter_mut()
        .find(|op| op.operator == ModelOperator::OutputHead)
        .unwrap()
}

#[test]
fn tiny_families_have_independently_expected_geometry_and_runtime_members() {
    for family in ["qwen2", "qwen3", "llama"] {
        for tied in [true, false] {
            let model = plan(family, tied);
            model.validate().unwrap();
            let result = inventory(&model);
            let schedule = lower_decoder_runtime_schedule(&model, &composition()).unwrap();
            for (mode, runtime) in [("prefill", &schedule.prefill), ("decode", &schedule.decode)] {
                let groups = result[mode]["groups"].as_array().unwrap();
                assert_eq!(groups.len(), 10); // lookup + 4 groups/layer + head
                assert_eq!(groups[0]["logical_shape"], json!([32, 8]));
                assert_eq!(groups[0]["artifact_kind"], "token_table");
                assert_eq!(groups[0]["orientation"], "vocabulary_feature");
                assert_eq!(groups[9]["logical_shape"], json!([32, 8]));
                assert_eq!(groups[9]["artifact_kind"], "matrix");
                assert_eq!(groups[9]["orientation"], "output_input");
                let table = &groups[0]["members"][0]["weight_id"];
                let output = &groups[9]["members"][0]["weight_id"];
                assert_eq!(table == output, tied);
                for layer in 0..2 {
                    let index = 1 + layer * 4;
                    for (offset, shape, widths) in [
                        (0, [16, 8], vec![8, 4, 4]),
                        (1, [8, 8], vec![8]),
                        (2, [32, 8], vec![16, 16]),
                        (3, [8, 16], vec![8]),
                    ] {
                        let group = &groups[index + offset];
                        assert_eq!(group["layer"], layer);
                        assert_eq!(group["logical_shape"], json!(shape));
                        let members = group["members"].as_array().unwrap();
                        let mut start = 0;
                        for (member, width) in members.iter().zip(&widths) {
                            assert_eq!(member["output_offset"], start);
                            assert_eq!(member["output_width"], *width);
                            assert_eq!(member["source_shape"], json!([width, shape[1]]));
                            assert_eq!(
                                member["bias_id"].is_string(),
                                family == "qwen2" && offset == 0
                            );
                            start += width;
                        }
                        assert_eq!(members.len(), widths.len());
                    }
                }
                let steps: Vec<_> = runtime
                    .steps
                    .iter()
                    .filter(|step| !step.weight_ids.is_empty())
                    .collect();
                assert_eq!(steps.len(), groups.len());
                for (order, (group, step)) in groups.iter().zip(steps).enumerate() {
                    assert_eq!(group["order"], order);
                    assert_eq!(group["layer"], json!(step.layer));
                    assert_eq!(group["input_id"], step.input_ids[0]);
                    let members = group["members"].as_array().unwrap();
                    assert_eq!(
                        members
                            .iter()
                            .map(|m| m["operation_id"].as_str().unwrap())
                            .collect::<Vec<_>>(),
                        step.operation_ids
                    );
                    assert_eq!(
                        members
                            .iter()
                            .map(|m| m["weight_id"].as_str().unwrap())
                            .collect::<Vec<_>>(),
                        step.weight_ids
                    );
                    for (member, output) in members.iter().zip(&step.outputs) {
                        assert_eq!(member["output_offset"], output.stage_offset);
                        assert_eq!(member["output_width"], output.stage_width);
                        assert_eq!(member["output_shape"], json!(output.output_shape));
                    }
                }
            }
            for forbidden in [
                "composition_digest",
                "complete",
                "protected_execution",
                "quantization",
                "executor",
            ] {
                assert!(!String::from_utf8(canonical_bytes(&result))
                    .unwrap()
                    .contains(forbidden));
            }
        }
    }
}

fn rewrite(value: &mut Value, replacements: &BTreeMap<String, String>) {
    match value {
        Value::String(string) => {
            if let Some(new) = replacements.get(string) {
                *string = new.clone();
            }
        }
        Value::Array(values) => {
            for value in values {
                rewrite(value, replacements);
            }
        }
        Value::Object(values) => {
            for (key, value) in values.iter_mut() {
                if key != "operator" {
                    rewrite(value, replacements);
                }
            }
        }
        _ => {}
    }
}

#[test]
fn arbitrary_operation_and_artifact_names_preserve_geometry_and_order() {
    for family in ["qwen2", "qwen3", "llama"] {
        let model = plan(family, true);
        let before = inventory(&model);
        let mut ids = BTreeSet::new();
        for graph in [&model.prefill, &model.decode] {
            for op in &graph.operations {
                ids.insert(op.id.clone());
                for field in ["weight", "bias"] {
                    if let Some(id) = op.attributes.get(field).and_then(Value::as_str) {
                        ids.insert(id.to_owned());
                    }
                }
            }
            for state in graph.state_inputs.iter().chain(&graph.state_outputs) {
                ids.insert(state.id.clone());
            }
        }
        // Deliberately reverse lexical ordering, use punctuation and Unicode.
        let renamed: BTreeMap<_, _> = ids
            .into_iter()
            .enumerate()
            .map(|(index, id)| (id, format!("opaque/λ:{}", 10000 - index)))
            .collect();
        let mut document = serde_json::to_value(&model).unwrap();
        rewrite(&mut document, &renamed);
        let renamed_model: DecoderPlan = serde_json::from_value(document).unwrap();
        let after = inventory(&renamed_model);
        let mut expected = before.clone();
        rewrite(&mut expected, &renamed);
        expected["model_plan_digest"] = json!(renamed_model.digest());
        assert_ne!(after["model_plan_digest"], before["model_plan_digest"]);
        assert_eq!(after, expected);
    }
}

#[test]
fn tables_share_public_input_group_but_keep_separate_artifacts_and_tied_head_owner() {
    let mut model = plan("qwen2", true);
    for graph in [&mut model.prefill, &mut model.decode] {
        let mut table = graph.operations[0].clone();
        table.id = "auxiliary-id".into();
        table.attributes = json!({"weight": "opaque-table-key", "weight_shape": [32, 6]});
        table.output_shape[2] = 6;
        graph.operations.insert(2, table); // Nonadjacent to first table.
    }
    model.validate().unwrap();
    let result = inventory(&model);
    let runtime = lower_decoder_runtime_schedule(&model, &composition()).unwrap();
    for (phase, runtime) in [("prefill", runtime.prefill), ("decode", runtime.decode)] {
        let lookup = &result[phase]["groups"][0];
        assert_eq!(lookup["logical_shape"], json!([32, 14]));
        assert_eq!(lookup["members"].as_array().unwrap().len(), 2);
        assert_eq!(lookup["members"][1]["source_shape"], json!([32, 6]));
        assert_eq!(lookup["members"][1]["output_offset"], 8);
        assert_eq!(lookup["members"][1]["weight_id"], "opaque-table-key");
        assert_eq!(
            runtime.steps[0].weight_ids,
            ["model.embed_tokens.weight", "opaque-table-key"]
        );
        assert_eq!(runtime.steps[0].outputs[1].stage_offset, 8);
        assert_eq!(result[phase]["groups"][9]["artifact_kind"], "matrix");
    }
}

// Complete config.json at Qwen/Qwen2.5-0.5B-Instruct revision
// 7ae557604adf67be50417f59c2c2f167def9a775; Git blob
// 0dbb161213629a23f0fc00ef286e6b1e366d180f. No weights or cache access.
const PINNED_CONFIG: &[u8] = br#"{
  "architectures": ["Qwen2ForCausalLM"], "attention_dropout": 0.0,
  "bos_token_id": 151643, "eos_token_id": 151645, "hidden_act": "silu",
  "hidden_size": 896, "initializer_range": 0.02, "intermediate_size": 4864,
  "max_position_embeddings": 32768, "max_window_layers": 21, "model_type": "qwen2",
  "num_attention_heads": 14, "num_hidden_layers": 24, "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "sliding_window": 32768,
  "tie_word_embeddings": true, "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1", "use_cache": true,
  "use_sliding_window": false, "vocab_size": 151936
}"#;

#[test]
fn full_pinned_config_has_96_body_groups_head_and_lookup_without_weights() {
    let model = lower_model_json(
        PINNED_CONFIG,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 128,
            max_new_tokens: 32,
        },
    )
    .unwrap();
    let result = inventory(&model);
    for mode in ["prefill", "decode"] {
        let groups = result[mode]["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 98);
        assert_eq!(groups[0]["logical_shape"], json!([151936, 896]));
        assert_eq!(groups[97]["logical_shape"], json!([151936, 896]));
        for layer in 0..24 {
            for (i, shape) in [[1152, 896], [896, 896], [9728, 896], [896, 4864]]
                .iter()
                .enumerate()
            {
                let group = &groups[1 + layer * 4 + i];
                assert_eq!(group["logical_shape"], json!(shape));
                assert_eq!(group["layer"], layer);
            }
        }
        let graph = if mode == "prefill" {
            &model.prefill
        } else {
            &model.decode
        };
        let by_id: BTreeMap<_, _> = graph
            .operations
            .iter()
            .map(|op| (op.id.as_str(), op))
            .collect();
        for member in groups.iter().flat_map(|g| g["members"].as_array().unwrap()) {
            let source = by_id[member["operation_id"].as_str().unwrap()];
            assert_eq!(member["weight_id"], source.attributes["weight"]);
            assert_eq!(
                member["bias_id"],
                source
                    .attributes
                    .get("bias")
                    .unwrap_or(&Value::Null)
                    .clone()
            );
        }
    }
}

#[test]
fn rejects_malformed_weight_bias_and_declared_shape_metadata() {
    for (key, value) in [
        ("weight", json!(null)),
        ("weight", json!("")),
        ("weight", json!(42)),
        ("bias", json!(false)),
        ("bias", json!(3)),
        ("bias", json!([])),
        ("bias", json!("")),
        ("weight_shape", json!([8, 7])),
        ("weight_shape", json!([8.0, 8])),
        ("weight_shape", json!("8x8")),
        ("bias_shape", json!([4])),
        ("transpose", json!(true)),
        ("input_dtype", json!(false)),
    ] {
        let mut model = plan("qwen2", true);
        for graph in [&mut model.prefill, &mut model.decode] {
            linear(graph).attributes[key] = value.clone();
        }
        assert!(
            lower_decoder_matrix_inventory(&model).is_err(),
            "{key}={value}"
        );
    }
    let mut valid = plan("qwen2", true);
    for graph in [&mut valid.prefill, &mut valid.decode] {
        linear(graph).attributes["weight_shape"] = json!([8, 8]);
        linear(graph).attributes["bias_shape"] = json!([8]);
    }
    inventory(&valid);
}

#[test]
fn rejects_duplicate_unbound_ids_and_incompatible_phases() {
    for case in 0..12 {
        let mut model = plan("qwen2", true);
        match case {
            0 => model.prefill.operations[1].id = model.prefill.operations[0].id.clone(),
            1 => linear(&mut model.prefill).inputs[0] = "missing-id".into(),
            2 => linear(&mut model.prefill)
                .inputs
                .push("input.tokens".into()),
            3 => model.prefill.operations[0].inputs[0] = "input.positions".into(),
            4 => linear(&mut model.prefill).attributes["weight"] = json!("input.tokens"),
            5 => linear(&mut model.decode).attributes["weight"] = json!("another-key"),
            6 => model.decode.operations.swap(2, 3), // Reorders q/k members.
            7 => model.decode.operations[0].output_shape[2] = 7, // Same source, conflicting shape.
            8 => head(&mut model.prefill).output_shape[1] = 31,
            9 => {
                head(&mut model.prefill).attributes["tied"] = json!(false);
            }
            10 => {
                let mut extra = head(&mut model.prefill).clone();
                extra.id = "extra-head".into();
                let n = model.prefill.operations.len();
                model.prefill.operations.insert(n - 1, extra);
            }
            11 => model.prefill.output = "unbound-result".into(),
            _ => unreachable!(),
        }
        assert!(
            lower_decoder_matrix_inventory(&model).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn rejects_source_shape_conflicts_within_phase_and_bad_prefixes() {
    for case in 0..7 {
        let mut model = plan("qwen2", true);
        for graph in [&mut model.prefill, &mut model.decode] {
            let indices: Vec<_> = graph
                .operations
                .iter()
                .enumerate()
                .filter(|(_, op)| op.operator == ModelOperator::Linear)
                .map(|(i, _)| i)
                .collect();
            let first_weight = graph.operations[indices[0]].attributes["weight"].clone();
            let first_bias = graph.operations[indices[0]].attributes["bias"].clone();
            match case {
                0 => graph.operations[indices[1]].attributes["weight"] = first_weight,
                1 => graph.operations[indices[1]].attributes["bias"] = first_bias,
                2 => graph.operations[indices[0]].attributes["bias"] = first_weight,
                3 => graph.operations[indices[0]].output_shape[0] = 2,
                4 => graph.operations[indices[0]].output_shape[1] += 1,
                5 => graph.operations[indices[0]].output_shape.push(1),
                6 => graph.operations[0].output_shape[1] += 1,
                _ => unreachable!(),
            }
        }
        assert!(
            lower_decoder_matrix_inventory(&model).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn bounded_admission_rejects_transforms_overflows_and_resource_excess() {
    for case in 0..8 {
        let mut model = plan("qwen2", true);
        match case {
            0 => model.transformations.push(pllm_models::AppliedMethod {
                component: "unsupported".into(),
                implementation: "unsupported".into(),
                method_id: "unsupported".into(),
                input_digest: model.digest(),
                configuration_digest: model.config_digest.clone(),
            }),
            1 => model.prefill.operations = vec![model.prefill.operations[0].clone(); 4097],
            2 => model.prefill.operations[0].id = "x".repeat(1025),
            3 => linear(&mut model.prefill).output_shape = vec![u64::MAX, 2],
            4 => model.prefill.operations[0].output_shape[2] = 0,
            5 => model.prefill.batch = 2,
            6 => {
                let mut nested = Value::Null;
                for _ in 0..25 {
                    nested = json!([nested]);
                }
                model.prefill.operations[0].attributes["extra"] = nested;
            }
            7 => {
                let mut table = model.prefill.operations[0].clone();
                table.id = "second-table".into();
                table.attributes["weight"] = json!("second-table-source");
                table.output_shape[2] = 1 << 20;
                model.prefill.operations.insert(1, table);
            }
            _ => unreachable!(),
        }
        assert!(
            lower_decoder_matrix_inventory(&model).is_err(),
            "case {case}"
        );
    }
    assert!(decoder_matrix_inventory_bytes(&vec![
        b' ';
        DECODER_MATRIX_INVENTORY_MAX_PLAN_BYTES + 1
    ])
    .is_err());
    for bytes in [
        b"{} {}".as_slice(),
        b"{\"a\":1,\"a\":2}",
        b"{\"bias\":1,\"bias\":null}",
    ] {
        assert!(decoder_matrix_inventory_bytes(bytes).is_err());
    }
    assert!(decoder_matrix_inventory_bytes(b"{\"a\":1,\"a\":2}")
        .unwrap_err()
        .contains("duplicate JSON key"));
    let mut fields = serde_json::to_value(plan("qwen2", true)).unwrap();
    fields["prefill"]["operations"][0]["attributes"]["weight"] = json!("x".repeat(7000));
    assert!(decoder_matrix_inventory_bytes(&canonical_bytes(&fields))
        .unwrap_err()
        .contains("encoded string"));
}

#[test]
fn resource_preflight_precedes_grouping_and_plan_allocation() {
    let mut model = plan("qwen2", true);
    let member = linear(&mut model.prefill).clone();
    model.prefill.operations = vec![member; 4000];
    // Duplicate IDs would fail later; conservative output admission fails first.
    assert!(lower_decoder_matrix_inventory(&model)
        .unwrap_err()
        .contains("output byte bound"));

    let mut model = plan("qwen2", true);
    model.prefill.operations = vec![model.prefill.operations[0].clone(); 4097];
    assert!(decoder_matrix_inventory_bytes(&canonical_bytes(&model))
        .unwrap_err()
        .contains("array limit"));

    let mut model = plan("qwen2", true);
    let mut member = model.prefill.operations[0].clone();
    member.attributes["extra"] = json!("x".repeat(1024));
    model.prefill.operations = vec![member; 1024];
    assert!(lower_decoder_matrix_inventory(&model)
        .unwrap_err()
        .contains("string limit"));
}

#[test]
fn matching_dimensions_do_not_merge_distinct_inputs_or_layers() {
    let mut model = plan("qwen3", true);
    for graph in [&mut model.prefill, &mut model.decode] {
        let token = graph.operations[0].id.clone();
        let k = graph
            .operations
            .iter_mut()
            .filter(|op| op.operator == ModelOperator::Linear)
            .nth(1)
            .unwrap();
        k.inputs[0] = token; // Same [batch, query, 8] shape, different semantic producer.
        let mut extra = linear(graph).clone();
        extra.id = "different-layer-member".into();
        extra.layer = Some(1);
        extra.attributes["weight"] = json!("different-layer-source");
        graph.operations.insert(3, extra);
    }
    let result = inventory(&model);
    for mode in ["prefill", "decode"] {
        let groups = result[mode]["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 12);
        assert_eq!(groups[1]["members"].as_array().unwrap().len(), 2); // q + v only
        assert_eq!(groups[1]["logical_shape"], json!([12, 8]));
        assert_eq!(groups[2]["layer"], 1);
        assert_eq!(groups[2]["members"].as_array().unwrap().len(), 1);
        assert_eq!(groups[3]["input_id"], "token_lookup");
        assert_eq!(groups[3]["members"].as_array().unwrap().len(), 1);
    }
}

fn local_extra(model: &mut DecoderPlan) -> &mut Value {
    &mut model
        .prefill
        .operations
        .iter_mut()
        .find(|op| op.operator == ModelOperator::RmsNorm)
        .unwrap()
        .attributes["extra"]
}

// Independent whole-JSON accounting, including keys, enum names and nulls.
// This deliberately uses the actual serialized schema rather than the native
// walk's field lists, so schema/count drift breaks the boundary regressions.
fn json_metadata_counts(value: &Value, depth: usize) -> (usize, usize, usize) {
    let (mut nodes, mut strings, mut maximum_depth) = (1, 0, depth);
    match value {
        Value::String(value) => strings += value.len(),
        Value::Array(values) => {
            for value in values {
                let (child_nodes, child_strings, child_depth) =
                    json_metadata_counts(value, depth + 1);
                nodes += child_nodes;
                strings += child_strings;
                maximum_depth = maximum_depth.max(child_depth);
            }
        }
        Value::Object(fields) => {
            for (key, value) in fields {
                let (child_nodes, child_strings, child_depth) =
                    json_metadata_counts(value, depth + 1);
                nodes += child_nodes;
                strings += key.len() + child_strings;
                maximum_depth = maximum_depth.max(child_depth);
            }
        }
        _ => {}
    }
    (nodes, strings, maximum_depth)
}

fn assert_entry_point_rejection(model: &DecoderPlan, reason: &str) {
    assert!(lower_decoder_matrix_inventory(model)
        .unwrap_err()
        .contains(reason));
    assert!(decoder_matrix_inventory_bytes(&canonical_bytes(model))
        .unwrap_err()
        .contains(reason));
}

#[test]
fn typed_and_bytes_apply_whole_document_depth_limits() {
    for arrays in [18, 19, 20, 21] {
        let mut model = plan("qwen2", true);
        let mut extra = Value::Null;
        for _ in 0..arrays {
            extra = json!([extra]);
        }
        *local_extra(&mut model) = extra;
        // root=0 -> prefill=1 -> operations=2 -> op=3 -> attributes=4
        // -> extra=5; the null leaf is 5 + arrays levels deep.
        let (_, _, depth) = json_metadata_counts(&serde_json::to_value(&model).unwrap(), 0);
        assert_eq!(depth, 5 + arrays);
        if depth <= 24 {
            inventory(&model); // Both entry points agree on bytes and digest.
        } else {
            assert_entry_point_rejection(&model, "depth or node limit");
        }
    }
}

#[test]
fn typed_and_bytes_count_all_document_nodes_at_boundary() {
    let mut model = plan("qwen3", true);
    *local_extra(&mut model) = json!([]);
    let (nodes, _, _) = json_metadata_counts(&serde_json::to_value(&model).unwrap(), 0);
    let mut remaining = 131072 - nodes;
    let extra = local_extra(&mut model).as_array_mut().unwrap();
    while remaining > 0 {
        if remaining == 1 {
            extra.push(Value::Null);
            remaining -= 1;
        } else {
            let count = (remaining - 1).min(4096);
            extra.push(Value::Array(vec![Value::Null; count]));
            remaining -= count + 1;
        }
    }
    assert_eq!(
        json_metadata_counts(&serde_json::to_value(&model).unwrap(), 0).0,
        131072
    );
    inventory(&model);
    local_extra(&mut model)
        .as_array_mut()
        .unwrap()
        .push(Value::Null);
    assert_entry_point_rejection(&model, "depth or node limit");
}

#[test]
fn typed_and_bytes_count_keys_and_value_strings_at_boundary() {
    let mut model = plan("llama", false);
    *local_extra(&mut model) = json!([]);
    let (_, strings, _) = json_metadata_counts(&serde_json::to_value(&model).unwrap(), 0);
    let mut remaining = (1 << 20) - strings;
    let extra = local_extra(&mut model).as_array_mut().unwrap();
    while remaining > 0 {
        let count = remaining.min(1024);
        extra.push(json!("x".repeat(count)));
        remaining -= count;
    }
    assert_eq!(
        json_metadata_counts(&serde_json::to_value(&model).unwrap(), 0).1,
        1 << 20
    );
    inventory(&model);
    local_extra(&mut model)
        .as_array_mut()
        .unwrap()
        .push(json!("x"));
    assert_entry_point_rejection(&model, "string limit");
}
