//! Source-only Gate1 contracts. No fixture/model downloads or execution effects.
use super::*;
use super::numeric_contract_tests::{grouping_regression_plan, legacy_dispatch_ids, legacy_groups};
use pllm_models::{lower_model_json, DecoderWorkload};
use pllm_types::canonical_bytes;
use serde_json::{json, Value};

fn absolute_plan() -> DecoderPlan {
    lower_model_json(&canonical_bytes(&json!({
        "model_type":"gpt2", "n_embd":4, "n_head":2, "n_layer":1,
        "n_positions":7, "vocab_size":11, "n_inner":9
    })), DecoderWorkload { batch:1, max_input_tokens:2, max_new_tokens:2 }).unwrap()
}

fn composition_value() -> Value {
    json!({"components": {
        "inference":{"component":"pllm/inference","params":{}},
        "kernels":{"component":"pllm/cpu","params":{"threads":1}},
        "linear":{"component":"pllm/masked-linear","params":{}},
        "preparation":{"component":"pllm/model-aware-corrections","params":{}}
    }, "model":{"source":"model.fixture"}})
}

fn baseline_phase(graph: &DecoderGraph) -> Result<DecoderRuntimePhaseSchedule, String> {
    lower_phase_with_admission(graph, DecoderRuntimeExecutor::RemoteStage, 0,
        &BTreeSet::new(), &BTreeSet::new(), LocalAdmission::Baseline)
}

fn construct(graph: &DecoderGraph) -> RemoteGroups<'_> {
    let quote = RemoteGroupingQuote::checked(graph).unwrap();
    let resources = quote.resources();
    let groups = quote.construct(resources.workspace_bytes, resources.work_bound).unwrap();
    assert!(groups.usage().work <= resources.work_bound);
    assert!(groups.usage().comparisons <= resources.comparison_bound);
    assert_eq!(groups.retained_index_bytes(), resources.heap_bytes);
    groups
}

#[test]
fn imported_absolute_plan_covers_every_operation_once_with_local_positions_norm_and_gelu() {
    let plan = absolute_plan();
    let before = canonical_bytes(&plan);
    let digest = plan.digest();
    plan.validate().unwrap();
    let bytes = canonical_bytes(&composition_value());
    let schedule = lower_decoder_runtime_schedule(&plan, &bytes).unwrap();
    assert!(!schedule.protected_execution);
    assert!(schedule.complete);
    assert!(schedule.private_boundary.is_none() && schedule.nonlinear.is_none());
    assert_eq!(schedule.schema_version, "pllm.decoder_runtime_schedule.v2");
    for (graph, phase) in [(&plan.prefill, &schedule.prefill), (&plan.decode, &schedule.decode)] {
        let mut covered = BTreeSet::new();
        for step in &phase.steps {
            for id in &step.operation_ids { assert!(covered.insert(id.as_str()), "{id}"); }
            for id in &step.operation_ids {
                let op = graph.operations.iter().find(|op| &op.id == id).unwrap();
                if op.id == "position_lookup" || op.id == "embedding_sum"
                    || matches!(op.operator, ModelOperator::LayerNorm | ModelOperator::GeluTanh | ModelOperator::CausalMask) {
                    assert_eq!(step.operation_ids.len(), 1);
                    assert_eq!(step.executor, DecoderRuntimeExecutor::ClientLocal);
                    assert!(step.weight_ids.is_empty());
                    assert_eq!(step.outputs.len(), 1);
                    assert_eq!(step.outputs[0].output_shape, op.output_shape);
                    assert_eq!(step.outputs[0].stage_offset, 0);
                }
            }
        }
        assert_eq!(covered, graph.operations.iter().map(|op| op.id.as_str()).collect());
        let token = phase.steps.iter().find(|step| step.operation_ids == ["token_lookup"]).unwrap();
        assert_eq!(token.executor, DecoderRuntimeExecutor::RemoteStage);
        assert_eq!(token.weight_ids.len(), 1);
        let head = phase.steps.iter().find(|step| step.operators == [ModelOperator::OutputHead]).unwrap();
        assert_eq!(head.outputs[0].stage_width, 11);
        assert_eq!(head.weight_ids.len(), 1);
        let position = graph.operations.iter().find(|op| op.id == "position_lookup").unwrap();
        assert_eq!(pllm_models::describe_embedding(position).unwrap().cardinality, Some(7));
        let qkv = phase.steps.iter().find(|step| step.operation_ids == ["layer.0.qkv_linear"]).unwrap();
        assert_eq!(qkv.outputs[0].stage_width, 12);
    }
    let coverage = crate::decoder_coverage(&plan, Some(&bytes)).unwrap();
    let lookup = coverage.operators.iter().find(|row| row.operator == ModelOperator::TokenLookup).unwrap();
    assert!(lookup.blocker.contains("absolute-position"));
    assert_eq!(canonical_bytes(&plan), before);
    assert_eq!(plan.digest(), digest);
}

#[test]
fn position_structural_classification_never_parses_attributes_or_creates_remote_members() {
    let mut graph = absolute_plan().prefill;
    let resources = RemoteGroupingQuote::checked(&graph).unwrap().resources();
    let position_index = graph.operations.iter().position(|op| op.id == "position_lookup").unwrap();
    let groups = construct(&graph);
    assert_eq!(groups.rows[position_index].class, GroupingClass::PositionCandidate);
    assert!(!groups.is_remote(position_index));
    assert!(groups.at(position_index).is_none());
    assert!(groups.get("position_lookup").is_none());
    assert!(groups.members.iter().all(|member| member.index != position_index));
    let expected_work = groups.usage().work;
    drop(groups);
    graph.operations[position_index].attributes = Value::Null;
    // Structural quote/construction has no recursive JSON interpretation. A
    // malformed candidate is NOT semantically admitted by this successful owner.
    assert_eq!(RemoteGroupingQuote::checked(&graph).unwrap().resources(), resources);
    assert_eq!(construct(&graph).usage().work, expected_work);
    GROUPING_ALLOCATIONS.with(|count| count.set(0));
    assert!(baseline_phase(&graph).is_err());
    GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 0));
}

#[test]
fn source_comparison_work_is_byte_charged_and_conservatively_quoted() {
    let mut op = absolute_plan().prefill.operations[0].clone();
    for source in ["input.tokens", "input.positions", "", "x", "input.segments", "éééééééé", "input.tokens.extra"] {
        op.inputs = vec![source.into()];
        let quote = grouping_class_work(&op).unwrap();
        assert_eq!(quote, (source.len().min(12) + 1 + source.len().min(15) + 1) as u128);
        let mut usage = GroupingUsage::default();
        let class = grouping_class(&op, |a, b| grouping_bytes_cmp(a, b, &mut usage).is_eq());
        assert!(usage.work <= quote);
        assert_eq!(usage.comparisons, 0);
        assert_eq!(usage.work, if source == "input.tokens" { 13 } else { quote });
        assert_eq!(class, match source {
            "input.tokens" => GroupingClass::TokenBoundary,
            "input.positions" => GroupingClass::PositionCandidate,
            _ => GroupingClass::UnsupportedLookup,
        });
    }
    op.inputs.clear();
    assert_eq!(grouping_class_work(&op).unwrap(), 0);
    op.inputs = vec!["input.tokens".into(), "input.positions".into()];
    assert_eq!(grouping_class_work(&op).unwrap(), 0);
    op.operator = ModelOperator::Linear;
    assert_eq!(grouping_class_work(&op).unwrap(), 0);
    // The exact checked operations used for Q accumulation, Wexisting+Q,
    // and the concrete cached-row allocation must refuse overflow.
    assert_eq!(grouping_add(u128::MAX, 29), Err(RemoteGroupingError::Overflow));
    assert_eq!(grouping_add(u128::MAX - 28, 29), Err(RemoteGroupingError::Overflow));
    assert_eq!(grouping_layout::<GroupRow>(usize::MAX), Err(RemoteGroupingError::Overflow));
}

#[test]
fn actual_absolute_grouping_exact_quote_and_one_short_precede_all_factories() {
    let plan = absolute_plan();
    for graph in [&plan.prefill, &plan.decode] {
        let resources = RemoteGroupingQuote::checked(graph).unwrap().resources();
        let remote_count = graph.operations.iter().filter(|op| {
            matches!(op.operator, ModelOperator::Linear | ModelOperator::OutputHead)
                || (op.operator == ModelOperator::TokenLookup && op.inputs == ["input.tokens"])
        }).count();
        assert_eq!(resources.heap_bytes,
            grouping_layout::<usize>(graph.operations.len()).unwrap()
            + grouping_layout::<GroupRow>(graph.operations.len()).unwrap()
            + grouping_layout::<RemoteMember<'_>>(remote_count).unwrap());
        GROUPING_ALLOCATIONS.with(|count| count.set(0));
        assert!(matches!(RemoteGroupingQuote::checked(graph).unwrap()
            .construct(resources.workspace_bytes - 1, resources.work_bound), Err(RemoteGroupingError::Capacity)));
        GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 0));
        assert!(matches!(RemoteGroupingQuote::checked(graph).unwrap()
            .construct(resources.workspace_bytes, resources.work_bound - 1), Err(RemoteGroupingError::WorkLimit)));
        GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 0));
        let actual = construct(graph);
        GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 3));
        assert_eq!(actual.members.len(), remote_count);
        let expected = legacy_groups(graph).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, op) in graph.operations.iter().enumerate() {
            assert_eq!(actual.at(index).map(|group| group.operations.iter().map(|op| op.id.as_str()).collect::<Vec<_>>()),
                expected.get(op.id.as_str()).map(|group| group.iter().map(|op| op.id.as_str()).collect::<Vec<_>>()));
        }
    }
}

#[test]
fn position_candidates_do_not_satisfy_token_requirement_or_alias_token_ids() {
    let mut graph = absolute_plan().prefill;
    graph.operations.retain(|op| op.id != "token_lookup");
    assert!(matches!(checked_remote_groups(&graph), Err(RemoteGroupingError::LookupHead)));
    for position_first in [false, true] {
        let mut graph = absolute_plan().prefill;
        graph.operations[1].id = graph.operations[0].id.clone();
        if position_first { graph.operations.swap(0, 1); }
        let groups = construct(&graph);
        let position_index = usize::from(!position_first);
        let token_index = usize::from(position_first);
        assert!(!groups.is_remote(position_index));
        assert!(groups.at(position_index).is_none());
        assert!(groups.is_remote(token_index));
        assert_eq!(groups.at(token_index).unwrap().operations.len(), 1);
        let expected = legacy_dispatch_ids(&graph).unwrap_err();
        assert_eq!(baseline_phase(&graph).unwrap_err(), expected);
    }
}

#[test]
fn late_lookup_domain_table_and_numeric_failures_publish_no_schedule() {
    let plan = absolute_plan();
    let original = plan.prefill.operations.iter().find(|op| op.id == "position_lookup").unwrap();
    for (key, value) in [
        ("index_domain", Value::Null), ("index_domain", json!("tokens")), ("index_domain", json!("unknown")),
        ("weight_shape", Value::Null), ("weight_shape", json!([0,4])), ("weight_shape", json!([7,0])),
        ("weight_shape", json!([7,5])), ("weight_shape", json!([7,4,1])),
        ("compute_dtype", Value::Null), ("compute_dtype", json!("bfloat16")),
        ("output_dtype", Value::Null), ("output_dtype", json!("float64")),
        ("weight", Value::Null), ("position_policy", json!("wrapped")), ("out_of_bounds", json!("clip")),
    ] {
        let mut graph = plan.prefill.clone();
        let mut late = original.clone();
        late.id = "late.position".into();
        late.attributes[key] = value;
        graph.operations.push(late);
        GROUPING_ALLOCATIONS.with(|count| count.set(0));
        assert!(baseline_phase(&graph).is_err(), "{key}");
        GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 0));
    }
    for source in ["input.segments", "input.unknown", "input.tokens"] {
        let mut graph = plan.prefill.clone();
        let position = graph.operations.iter_mut().find(|op| op.id == "position_lookup").unwrap();
        position.inputs = vec![source.into()];
        if source == "input.segments" { position.attributes["index_domain"] = json!("segments"); }
        assert!(baseline_phase(&graph).is_err());
    }
    let mut graph = plan.prefill.clone();
    graph.operations[1].inputs.push("input.tokens".into());
    assert!(matches!(checked_remote_groups(&graph), Err(RemoteGroupingError::LookupSource)));
    assert!(baseline_phase(&graph).is_err());
}

#[test]
fn baseline_context_is_explicit_and_never_grants_delegated_private_or_nonlinear_rights() {
    let plan = absolute_plan();
    for graph in [&plan.prefill, &plan.decode] {
        assert!(lower_phase(graph, DecoderRuntimeExecutor::RemoteStage, 0,
            &BTreeSet::new(), &BTreeSet::new()).is_err());
        for op in &graph.operations {
            if matches!(op.operator, ModelOperator::LayerNorm | ModelOperator::GeluTanh) || op.id == "position_lookup" {
                assert!(!admitted_local_operator(op, LocalAdmission::Existing));
                assert!(admitted_local_operator(op, LocalAdmission::Baseline), "{}", op.id);
            }
        }
    }
    let pipeline: crate::ExperimentPipeline = serde_json::from_value(composition_value()).unwrap();
    assert_eq!(baseline_local_admission(crate::DecoderCompositionKind::MaskedLinear, &pipeline), LocalAdmission::Baseline);
    assert_eq!(baseline_local_admission(crate::DecoderCompositionKind::ClientOnlyLinear, &pipeline), LocalAdmission::Baseline);
    for kind in [crate::DecoderCompositionKind::Other, crate::DecoderCompositionKind::VerifiedMaskedLinear,
        crate::DecoderCompositionKind::TwoOnlineOffsetLinear] {
        assert_eq!(baseline_local_admission(kind, &pipeline), LocalAdmission::Existing);
    }
    for (slot, component) in [("boundary", "pllm/private-lookup-full-head/v1"),
        ("nonlinear", "pllm/masked-wire") ] {
        let mut value = composition_value();
        value["components"][slot] = json!({"component":component,"params":{}});
        let pipeline: crate::ExperimentPipeline = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(baseline_local_admission(crate::DecoderCompositionKind::MaskedLinear, &pipeline), LocalAdmission::Existing);
        assert!(lower_decoder_runtime_schedule(&plan, &canonical_bytes(&value)).is_err());
    }
    let mut unsupported = composition_value();
    unsupported["components"]["linear"]["component"] = json!("pllm/not-admitted");
    assert!(lower_decoder_runtime_schedule(&plan, &canonical_bytes(&unsupported)).is_err());
}

#[test]
fn f32_norm_gelu_reject_malformed_numeric_contracts_without_changing_bf16_branch() {
    let graph = absolute_plan().prefill;
    let norm = graph.operations.iter().find(|op| op.operator == ModelOperator::LayerNorm).unwrap();
    for (key, value) in [("epsilon", json!("NaN")), ("epsilon", json!("1e-1000")),
        ("compute_dtype", json!("float64")), ("output_dtype", Value::Null),
        ("output_dtype", json!("bfloat16")), ("bias", Value::Null), ("weight_shape", json!([5]))] {
        let mut op = norm.clone();
        op.attributes[key] = value;
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline));
    }
    let gelu = graph.operations.iter().find(|op| op.operator == ModelOperator::GeluTanh).unwrap();
    assert!(baseline_f32_gelu(gelu));
    for (key, value) in [("formula", json!("erf")), ("compute_dtype", json!("float64")),
        ("compute_dtype", Value::Null), ("output_dtype", json!("bfloat16")),
        ("approximation", json!("tanh"))] {
        let mut op = gelu.clone();
        op.attributes[key] = value;
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline));
    }
    let mut bf16 = gelu.clone();
    bf16.attributes = json!({"compute_dtype":"bfloat16", "output_dtype":"bfloat16", "approximation":"tanh"});
    assert!(local_operator(&bf16));
    assert!(admitted_local_operator(&bf16, LocalAdmission::Existing));
    assert!(admitted_local_operator(&bf16, LocalAdmission::Baseline));
    bf16.attributes["approximation"] = json!("erf");
    assert!(!admitted_local_operator(&bf16, LocalAdmission::Baseline));
}

#[test]
fn imported_add_and_visible_prefix_mask_are_exact_baseline_local_forms_in_both_phases() {
    let plan = absolute_plan();
    let before = canonical_bytes(&plan);
    let digest = plan.digest();
    plan.validate().unwrap();
    for graph in [&plan.prefill, &plan.decode] {
        let add = graph.operations.iter().find(|op| op.id == "embedding_sum").unwrap();
        let mask = graph.operations.iter().find(|op| op.operator == ModelOperator::CausalMask).unwrap();
        assert!(baseline_inference_add(add));
        assert!(baseline_visible_prefix_mask(mask));
        for op in [add, mask] {
            assert!(!local_operator(op));
            assert!(!admitted_local_operator(op, LocalAdmission::Existing));
            assert!(admitted_local_operator(op, LocalAdmission::Baseline));
        }
        let phase = baseline_phase(graph).unwrap();
        for op in [add, mask] {
            let step = phase.steps.iter().find(|step| step.operation_ids == [op.id.as_str()]).unwrap();
            assert_eq!(step.executor, DecoderRuntimeExecutor::ClientLocal);
            assert_eq!(step.operators, [op.operator]);
            assert_eq!(step.input_ids, op.inputs);
            assert!(step.weight_ids.is_empty());
            assert_eq!(step.outputs.len(), 1);
            assert_eq!(step.outputs[0].output_shape, op.output_shape);
        }
    }
    assert_eq!(canonical_bytes(&plan), before);
    assert_eq!(plan.digest(), digest);
}

#[test]
fn inference_add_rejects_training_nonstring_extra_dtype_and_malformed_shapes() {
    let graph = absolute_plan().prefill;
    let original = graph.operations.iter().find(|op| op.id == "embedding_sum").unwrap();
    for value in [Value::Null, json!(true), json!(false), json!(0), json!(0.0), json!([]), json!({}),
        json!(""), json!("disabled"), json!("inference"), json!("training"), json!("enabled"), json!("0")] {
        let mut op = original.clone();
        op.attributes["dropout"] = value;
        assert!(!baseline_inference_add(&op));
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline));
        assert!(!admitted_local_operator(&op, LocalAdmission::Existing));
    }
    for (key, value) in [("extra", Value::Null), ("training", json!(false)), ("mode", json!("inference")),
        ("compute_dtype", json!("float32")), ("output_dtype", json!("float32")),
        ("output_dtype", json!("bfloat16")), ("output_dtype", Value::Null)] {
        let mut op = original.clone();
        op.attributes[key] = value;
        assert!(!baseline_inference_add(&op));
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline));
    }
    for attributes in [Value::Null, json!(false), json!([]), json!("disabled_inference")] {
        let mut op = original.clone();
        op.attributes = attributes;
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline));
    }
    for inputs in [vec![], vec!["token_lookup".into()], vec!["token_lookup".into(), "position_lookup".into(), "other".into()]] {
        let mut op = original.clone();
        op.inputs = inputs;
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline));
    }
    for shape in [vec![1,4], vec![1,1,2,4], vec![1,0,4]] {
        let mut op = original.clone();
        op.output_shape = shape;
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline));
    }
    let mut ordinary = original.clone();
    ordinary.attributes.as_object_mut().unwrap().remove("dropout");
    assert!(!baseline_inference_add(&ordinary));
    // Removing the source alias yields the PRE-EXISTING {} residual profile,
    // not a new inference-dropout permission. Preserve its old acceptance.
    assert!(local_operator(&ordinary));
    assert!(admitted_local_operator(&ordinary, LocalAdmission::Existing));
    assert!(admitted_local_operator(&ordinary, LocalAdmission::Baseline));
    ordinary.attributes = json!({"compute_dtype":"bfloat16", "output_dtype":"bfloat16"});
    assert!(local_operator(&ordinary));
    assert!(admitted_local_operator(&ordinary, LocalAdmission::Existing));
    assert!(admitted_local_operator(&ordinary, LocalAdmission::Baseline));
}

#[test]
fn visible_prefix_mask_rejects_every_missing_field_nonstring_wrong_mode_sentinel_and_extra() {
    let graph = absolute_plan().prefill;
    let original = graph.operations.iter().find(|op| op.operator == ModelOperator::CausalMask).unwrap();
    for key in ["padding_mask_input", "valid_lengths_input", "cache_validity", "masked_value"] {
        let mut missing = original.clone();
        missing.attributes.as_object_mut().unwrap().remove(key);
        assert!(!admitted_local_operator(&missing, LocalAdmission::Baseline), "missing {key}");
        for value in [Value::Null, json!(true), json!(false), json!(0), json!(-1), json!([]), json!({}), json!(""), json!("unknown")] {
            let mut op = original.clone();
            op.attributes[key] = value;
            assert!(!admitted_local_operator(&op, LocalAdmission::Baseline), "{key}");
            assert!(!admitted_local_operator(&op, LocalAdmission::Existing), "{key}");
        }
    }
    for (key, value) in [("masked_value", json!("-inf")), ("masked_value", json!("-1e9")),
        ("masked_value", json!("positive_infinity")), ("cache_validity", json!("valid_lengths_fixed_capacity")),
        ("cache_validity", json!("valid_lengths_bounded_suffix")), ("cache_validity", json!("training")),
        ("padding_mask_input", json!("input.sequence_lengths")), ("valid_lengths_input", json!("input.attention_mask")),
        ("extra", Value::Null), ("mode", json!("inference")), ("kind", json!("full_causal")),
        ("absolute_positions_input", json!("input.positions")),
        ("compute_dtype", json!("float32")), ("output_dtype", json!("float32")),
        ("output_dtype", json!("bfloat16")), ("output_dtype", Value::Null)] {
        let mut op = original.clone();
        op.attributes[key] = value;
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline), "{key}");
    }
    for suffix_index in 1..4 {
        let mut op = original.clone();
        op.inputs[suffix_index] = "input.tokens".into();
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline));
    }
    for inputs in [vec![], original.inputs[..3].to_vec(), {
        let mut inputs = original.inputs.clone(); inputs.push("input.extra".into()); inputs
    }] {
        let mut op = original.clone();
        op.inputs = inputs;
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline));
    }
    for shape in [vec![1,2,4], vec![1,2,2,0], vec![1,2,2,2,1]] {
        let mut op = original.clone();
        op.output_shape = shape;
        assert!(!admitted_local_operator(&op, LocalAdmission::Baseline));
    }
}

#[test]
fn actual_source_add_shape_compatibility_and_mask_dtype_fail_before_schedule_storage() {
    let original = absolute_plan();
    for id in ["embedding_sum", "layer.0.causal_mask"] {
        for mutate_shape in [false, true] {
            let mut plan = original.clone();
            let op = plan.decode.operations.iter_mut().find(|op| op.id == id).unwrap();
            if mutate_shape { *op.output_shape.last_mut().unwrap() += 1; }
            else { op.attributes["output_dtype"] = json!("bfloat16"); }
            GROUPING_ALLOCATIONS.with(|count| count.set(0));
            assert!(lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition_value())).is_err());
            GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 0));
        }
    }
    // Although {} remains a valid legacy add profile, stripping source attrs
    // from an imported plan must still fail its original source replay check.
    for (id, key) in [("embedding_sum", "dropout"), ("layer.0.causal_mask", "masked_value")] {
        let mut plan = original.clone();
        let op = plan.decode.operations.iter_mut().find(|op| op.id == id).unwrap();
        op.attributes.as_object_mut().unwrap().remove(key);
        GROUPING_ALLOCATIONS.with(|count| count.set(0));
        assert!(lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition_value())).is_err());
        GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 0));
    }
}

#[test]
fn old_token_decoder_phase_bytes_digest_head_gqa_and_terminal_pruning_stay_unchanged() {
    for family in ["qwen2", "qwen3", "llama"] {
        for tied in [false, true] {
            let plan = grouping_regression_plan(family, tied);
            let before = canonical_bytes(&plan);
            let digest = plan.digest();
            let bytes = canonical_bytes(&composition_value());
            let schedule = lower_decoder_runtime_schedule(&plan, &bytes).unwrap();
            let mut old = schedule.clone();
            for (graph, phase) in [(&plan.prefill, &mut old.prefill), (&plan.decode, &mut old.decode)] {
                *phase = lower_phase(graph, DecoderRuntimeExecutor::RemoteStage, 0,
                    &BTreeSet::new(), &BTreeSet::new()).unwrap();
                assert_eq!(phase.steps.iter().map(|step| step.operation_ids.iter().map(String::as_str).collect::<Vec<_>>()).collect::<Vec<_>>(),
                    legacy_dispatch_ids(graph).unwrap());
                let qkv = phase.steps.iter().find(|step| step.operators == [ModelOperator::Linear; 3]).unwrap();
                let mut widths: Vec<_> = qkv.outputs.iter().map(|output| output.stage_width).collect();
                widths.sort_unstable();
                assert_eq!(widths, [4,4,8]);
                let mut offset = 0;
                for output in &qkv.outputs { assert_eq!(output.stage_offset, offset); offset += output.stage_width; }
                assert_eq!(offset, 16);
                let head = phase.steps.iter().find(|step| step.operators == [ModelOperator::OutputHead]).unwrap();
                assert_eq!(head.outputs[0].stage_width, 32);
            }
            assert_eq!(canonical_bytes(&schedule), canonical_bytes(&old));
            assert_eq!(schedule.digest(), old.digest());
            assert_eq!(canonical_bytes(&plan), before);
            assert_eq!(plan.digest(), digest);
            let mut pruning = composition_value();
            pruning["components"]["linear"]["params"]["prefill_pruning"] = json!("terminal");
            let pruned = lower_decoder_runtime_schedule(&plan, &canonical_bytes(&pruning)).unwrap();
            let demand = terminal_demand(&plan.prefill);
            for (plain, pruned) in schedule.prefill.steps.iter().zip(&pruned.prefill.steps) {
                assert_eq!(plain.operation_ids, pruned.operation_ids);
                assert_eq!(plain.weight_ids, pruned.weight_ids);
                let expected = plain.operators.iter().all(|op| *op == ModelOperator::Linear)
                    && plain.operation_ids.iter().all(|id| demand.get(id.as_str()) == Some(&1))
                    && plain.outputs.iter().all(|out| matches!(out.output_shape.as_slice(), [1, rows, _] if *rows == plan.prefill.query_sequence));
                assert_eq!(pruned.terminal_row_only, expected);
            }
            assert_eq!(canonical_bytes(&schedule.decode), canonical_bytes(&pruned.decode));
        }
    }
}
