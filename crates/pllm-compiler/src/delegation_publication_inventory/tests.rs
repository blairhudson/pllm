use super::*;
use crate::decoder_matrix_inventory::lower_decoder_matrix_inventory;
use pllm_models::{lower_model_json, DecoderWorkload, ModelOperation};
use serde_json::{json, Value};

fn digest(label: &str) -> Digest {
    digest_bytes("publication-test", label.as_bytes())
}

fn op(
    id: &str,
    operator: ModelOperator,
    layer: Option<u64>,
    input: &str,
    shape: Vec<u64>,
    attributes: Value,
) -> ModelOperation {
    ModelOperation {
        id: id.into(),
        operator,
        layer,
        state_kind: None,
        inputs: vec![input.into()],
        output_shape: shape,
        attributes,
    }
}

// Semantic metadata fixture, deliberately not a valid complete Qwen decoder.
// Inventory lowering accepts model-neutral graph references/geometry, not family
// semantics. No checkpoint, code arrays, native execution or secret is needed.
fn plan(head_width: u64, bias: bool) -> DecoderPlan {
    let mut plan = lower_model_json(
        &canonical_bytes(&json!({
            "model_type": "qwen2", "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 2, "vocab_size": head_width,
            "max_position_embeddings": 32, "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false
        })),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 3,
            max_new_tokens: 2,
        },
    )
    .unwrap();
    for graph in [&mut plan.prefill, &mut plan.decode] {
        let sequence = graph.query_sequence;
        graph.state_inputs.clear();
        graph.state_outputs.clear();
        graph.operations = vec![
            op(
                "opaque-a",
                ModelOperator::TokenLookup,
                None,
                "input.tokens",
                vec![1, sequence, 64],
                json!({"weight": "table"}),
            ),
            op(
                "opaque-b",
                ModelOperator::Linear,
                Some(0),
                "opaque-a",
                vec![1, sequence, 64],
                if bias {
                    json!({"weight": "body", "bias": "body-bias"})
                } else {
                    json!({"weight": "body"})
                },
            ),
            op(
                "opaque-c",
                ModelOperator::LastToken,
                None,
                "opaque-b",
                vec![1, 64],
                json!({"axis": 1}),
            ),
            op(
                "opaque-d",
                ModelOperator::OutputHead,
                None,
                "opaque-c",
                vec![1, head_width],
                json!({"weight": "head", "tied": false}),
            ),
            op(
                "opaque-e",
                ModelOperator::TokenFeedback,
                None,
                "opaque-d",
                vec![1, 1],
                json!({}),
            ),
        ];
        graph.output = "opaque-e".into();
    }
    plan
}

fn source(plan: &DecoderPlan) -> PublicationSourceIdentity {
    PublicationSourceIdentity {
        repository: "independent/metadata-fixture".into(),
        revision: "pinned-revision".into(),
        checkpoint_manifest_digest: digest("checkpoint"),
        tokenizer_digest: digest("tokenizer"),
        config_digest: plan.config_digest.clone(),
    }
}

fn artifacts(head_width: u64, bias: bool) -> Vec<PublicationArtifactIdentity> {
    let mut artifacts: Vec<_> = [
        (
            PublicationArtifactKind::TokenTable,
            "table",
            vec![head_width, 64],
            PublicationSourceLayout::VocabularyFeature,
        ),
        (
            PublicationArtifactKind::Matrix,
            "body",
            vec![64, 64],
            PublicationSourceLayout::OutputInput,
        ),
        (
            PublicationArtifactKind::Matrix,
            "head",
            vec![head_width, 64],
            PublicationSourceLayout::OutputInput,
        ),
    ]
    .into_iter()
    .map(|(kind, id, shape, layout)| PublicationArtifactIdentity {
        kind,
        weight_id: id.into(),
        source_artifact_digest: digest(id),
        publication_artifact_digest: digest(&format!("encoded-{id}")),
        source_shape: shape,
        source_layout: layout,
    })
    .collect();
    if bias {
        artifacts.push(PublicationArtifactIdentity {
            kind: PublicationArtifactKind::Bias,
            weight_id: "body-bias".into(),
            source_artifact_digest: digest("bias"),
            publication_artifact_digest: digest("encoded-bias"),
            source_shape: vec![64],
            source_layout: PublicationSourceLayout::OutputVector,
        });
    }
    artifacts
}

fn lower(
    plan: &DecoderPlan,
    artifacts: Vec<PublicationArtifactIdentity>,
) -> DelegationPublicationInventory {
    lower_delegation_publication_inventory(
        plan,
        &lower_decoder_matrix_inventory(plan).unwrap(),
        source(plan),
        artifacts,
        DelegationPublicationContract::source_derived_maverick(),
    )
    .unwrap()
}

fn request() -> MaverickBatchAdmissionRequest {
    MaverickBatchAdmissionRequest {
        rows: 64,
        columns: 64,
        batch_size: 2,
        capsule_ceiling_bytes: 16 << 20,
        aggregate_ceiling_bytes: 16 << 20,
        live_windows: 1,
        attempts_per_client: 16,
        publications: vec![MaverickBatchPublication {
            group_orders: [1, 2],
            privacy_code_digests: [[1; 32], [2; 32]],
            primary_code_digest: [3; 32],
            secondary_code_digest: [4; 32],
        }],
    }
}

fn json_inventory(inventory: &DelegationPublicationInventory) -> Value {
    serde_json::from_slice(inventory.canonical_bytes()).unwrap()
}

#[test]
fn complete_ordered_manifest_includes_full_head_every_row_bias_and_local_semantics() {
    let model = plan(64, true);
    let inventory = lower(&model, artifacts(64, true));
    let value = json_inventory(&inventory);
    assert_eq!(
        value["inventory"]["prefill"]["groups"][2]["logical_shape"],
        json!([64, 64])
    );
    assert_eq!(value["uses"].as_array().unwrap().len(), 6);
    assert_eq!(value["uses"][1]["operation_position"], 1);
    assert_eq!(value["uses"][1]["claim_row_count"], 3);
    assert_eq!(value["uses"][4]["claim_row_count"], 1);
    assert_eq!(value["uses"][2]["claim_row_count"], 1);
    assert_eq!(value["uses"][1]["bias_artifact_position"], 3);
    assert_eq!(
        value["uses"][1]["bias_coverage"],
        "bias_addition_numeric_contract_blocked"
    );
    assert_eq!(value["semantic_coverage"].as_array().unwrap().len(), 10);
    assert_eq!(
        value["semantic_coverage"][0]["coverage"],
        "token_lookup_ownership_and_precision_blocked"
    );
    assert_eq!(
        value["semantic_coverage"][2]["coverage"],
        "local_semantic_numeric_and_state_contract_blocked"
    );
    assert_eq!(value["source_identity_authenticated"], false);
    assert_eq!(value["status"], json!({"status": "blocked", "gate": "M-N"}));
    assert_eq!(canonical_bytes(&value), inventory.canonical_bytes());
    assert_eq!(
        inventory.digest(),
        &digest_bytes(
            DELEGATION_PUBLICATION_INVENTORY_SCHEMA_VERSION,
            inventory.canonical_bytes()
        )
    );
}

#[test]
fn ordered_artifact_and_source_and_publication_identity_changes_bind() {
    let model = plan(64, false);
    let baseline = lower(&model, artifacts(64, false));
    let mut swapped = artifacts(64, false);
    swapped.swap(1, 2);
    let changed = lower(&model, swapped);
    assert_ne!(baseline.digest(), changed.digest());
    assert_eq!(json_inventory(&changed)["uses"][1]["artifact_position"], 2);
    for publication_digest in [false, true] {
        let mut changed = artifacts(64, false);
        if publication_digest {
            changed[1].publication_artifact_digest = digest("different-publication");
        } else {
            changed[1].source_artifact_digest = digest("different-source");
        }
        assert_ne!(baseline.digest(), lower(&model, changed).digest());
    }
    let mut pin = source(&model);
    pin.revision = "different-revision".into();
    let changed = lower_delegation_publication_inventory(
        &model,
        &lower_decoder_matrix_inventory(&model).unwrap(),
        pin,
        artifacts(64, false),
        DelegationPublicationContract::source_derived_maverick(),
    )
    .unwrap();
    assert_ne!(baseline.digest(), changed.digest());
}

#[test]
fn shape_and_operation_order_bind_without_family_name_parsing() {
    let baseline = lower(&plan(64, false), artifacts(64, false));
    let changed = lower(&plan(65, false), artifacts(65, false));
    assert_ne!(baseline.digest(), changed.digest());
    assert_eq!(
        json_inventory(&changed)["inventory"]["prefill"]["groups"][2]["logical_shape"],
        json!([65, 64])
    );
    let mut model = plan(64, false);
    for graph in [&mut model.prefill, &mut model.decode] {
        graph.operations.insert(
            2,
            op(
                "opaque-extra",
                ModelOperator::Silu,
                Some(0),
                "opaque-b",
                vec![1, graph.query_sequence, 64],
                json!({}),
            ),
        );
    }
    let changed = lower(&model, artifacts(64, false));
    assert_ne!(baseline.digest(), changed.digest());
    assert_eq!(json_inventory(&changed)["uses"][2]["operation_position"], 4);
}

#[test]
fn exact_artifact_coverage_rejects_omissions_extras_duplicates_wrong_layout_and_shape() {
    let model = plan(64, true);
    for case in 0..7 {
        let mut identities = artifacts(64, true);
        match case {
            0 => {
                identities.remove(2);
            } // Full head may not be omitted.
            1 => {
                identities.pop();
            } // Nor biases.
            2 => identities.push(identities[1].clone()),
            3 => {
                let mut extra = identities[1].clone();
                extra.weight_id = "extra".into();
                identities.push(extra);
            }
            4 => identities[1].source_layout = PublicationSourceLayout::VocabularyFeature,
            5 => identities[1].source_shape = vec![64, 63],
            6 => identities[1].source_shape = vec![u64::MAX, 2],
            _ => unreachable!(),
        }
        assert!(
            lower_delegation_publication_inventory(
                &model,
                &lower_decoder_matrix_inventory(&model).unwrap(),
                source(&model),
                identities,
                DelegationPublicationContract::source_derived_maverick(),
            )
            .is_err(),
            "case {case}"
        );
    }
}

#[test]
fn source_config_and_inventory_plan_identity_cannot_be_substituted() {
    let model = plan(64, false);
    let other = plan(65, false);
    assert!(lower_delegation_publication_inventory(
        &other,
        &lower_decoder_matrix_inventory(&model).unwrap(),
        source(&other),
        artifacts(65, false),
        DelegationPublicationContract::source_derived_maverick(),
    )
    .is_err());
    let mut wrong = source(&model);
    wrong.config_digest = digest("wrong-config");
    assert!(lower_delegation_publication_inventory(
        &model,
        &lower_decoder_matrix_inventory(&model).unwrap(),
        wrong,
        artifacts(64, false),
        DelegationPublicationContract::source_derived_maverick(),
    )
    .is_err());
}

#[test]
fn native_price_matches_independent_fixed_payload_and_real_allocation_getters() {
    let inventory = lower(&plan(64, true), artifacts(64, true));
    let admission = admit_maverick_batch_publication(&inventory, request()).unwrap();
    let value: Value = serde_json::from_slice(admission.canonical_bytes()).unwrap();
    let a = &value["native_allocation_per_capsule"];
    assert_eq!(a["p_bytes"], 262144);
    assert_eq!(a["q_bytes"], 262144);
    assert_eq!(a["qaux_bytes"], 2097152);
    assert_eq!(a["matrix_bytes"], 32768);
    assert_eq!(a["code_bytes"], 40960);
    assert_eq!(a["public_payload_bytes"], 2695168);
    assert_eq!(a["source_and_incoming_validation_overlap_bytes"], 2695168);
    assert_eq!(a["separate_server_matrix_and_q_bytes"], 294912);
    assert_eq!(
        a["transient_transcript_and_validation_allowance_bytes"],
        524288
    );
    assert_eq!(a["control_stack_allocator_allowance_bytes"], 262144);
    assert_eq!(a["retained_output_bytes"], 8192);
    assert_eq!(a["planned_peak_bytes"], 6479872);
    assert_eq!(
        a["planned_peak_bytes"],
        AllocationPlan::bounded().unwrap().planned_peak_bytes()
    );
    assert_eq!(value["aggregate_planned_peak_bytes"], 6479872);
    assert_eq!(value["native_parameters"]["secondary_code_dimension"], 128);
    assert_eq!(value["native_parameters"]["primary_exact_weight"], 31);
    assert_eq!(value["status"]["gate"], "M-N");
    for key in [
        "executable",
        "checkpoint_authenticated",
        "native_publication_validation_completed",
        "code_security_qualified",
    ] {
        assert_eq!(value[key], false);
    }
    assert_eq!(admission.status(), inventory.status());
    assert!(!String::from_utf8_lossy(admission.canonical_bytes())
        .contains(HISTORICAL_MAVERICK_SCALAR_IDENTITY));
}

#[test]
fn native_geometry_overflow_caps_and_lifetime_reject_without_tensor_access() {
    let inventory = lower(&plan(64, false), artifacts(64, false));
    for case in 0..9 {
        let mut req = request();
        match case {
            0 => req.rows = 65,
            1 => req.columns = usize::MAX,
            2 => req.batch_size = 1,
            3 => req.capsule_ceiling_bytes = (16 << 20) + 1,
            4 => req.capsule_ceiling_bytes = 6479871,
            5 => req.aggregate_ceiling_bytes = 6479871,
            6 => req.live_windows = 2,
            7 => req.attempts_per_client = 15,
            8 => req.rows = 0,
            _ => unreachable!(),
        }
        assert!(
            admit_maverick_batch_publication(&inventory, req).is_err(),
            "case {case}"
        );
    }
    let wide_head = lower(&plan(65, false), artifacts(65, false));
    assert!(admit_maverick_batch_publication(&wide_head, request())
        .unwrap_err()
        .contains("full-head geometry"));
    let mut exact = request();
    exact.capsule_ceiling_bytes = 6479872;
    exact.aggregate_ceiling_bytes = 6479872;
    assert!(admit_maverick_batch_publication(&inventory, exact).is_ok());
}

#[test]
fn publication_slots_are_complete_unique_ordered_and_code_bound() {
    let inventory = lower(&plan(64, false), artifacts(64, false));
    let baseline = admit_maverick_batch_publication(&inventory, request()).unwrap();
    for slots in [[1, 1], [1, 0], [1, 99]] {
        let mut req = request();
        req.publications[0].group_orders = slots;
        assert!(admit_maverick_batch_publication(&inventory, req).is_err());
    }
    let mut empty = request();
    empty.publications.clear();
    assert!(admit_maverick_batch_publication(&inventory, empty).is_err());
    let mut reversed = request();
    reversed.publications[0].group_orders = [2, 1];
    assert_ne!(
        baseline.digest(),
        admit_maverick_batch_publication(&inventory, reversed)
            .unwrap()
            .digest()
    );
    for code in 0..4 {
        let mut changed = request();
        match code {
            0 => changed.publications[0].privacy_code_digests[0][0] ^= 1,
            1 => changed.publications[0].privacy_code_digests[1][0] ^= 1,
            2 => changed.publications[0].primary_code_digest[0] ^= 1,
            3 => changed.publications[0].secondary_code_digest[0] ^= 1,
            _ => unreachable!(),
        }
        assert_ne!(
            baseline.digest(),
            admit_maverick_batch_publication(&inventory, changed)
                .unwrap()
                .digest()
        );
    }
}

#[test]
fn field_change_has_no_historical_or_w8a8_fallback() {
    let model = plan(64, false);
    let mut contract = DelegationPublicationContract::source_derived_maverick();
    contract.field_modulus = 2147483659;
    assert!(lower_delegation_publication_inventory(
        &model,
        &lower_decoder_matrix_inventory(&model).unwrap(),
        source(&model),
        artifacts(64, false),
        contract,
    )
    .is_err());
    assert!(HISTORICAL_MAVERICK_SCALAR_IDENTITY.contains("signed-i8"));
    assert!(HISTORICAL_MAVERICK_SCALAR_IDENTITY.contains("256MiB"));
}

#[test]
fn no_matrix_omissions_and_resident_capsules_are_charged_independently() {
    let mut model = plan(64, false);
    for graph in [&mut model.prefill, &mut model.decode] {
        for layer in 1..=2 {
            graph.operations.insert(
                1 + layer as usize,
                op(
                    &format!("independent-{layer}"),
                    ModelOperator::Linear,
                    Some(layer),
                    "opaque-a",
                    vec![1, graph.query_sequence, 64],
                    json!({"weight": format!("independent-weight-{layer}")}),
                ),
            );
        }
    }
    let mut identities = artifacts(64, false);
    for layer in 1..=2 {
        let mut artifact = identities[1].clone();
        artifact.weight_id = format!("independent-weight-{layer}");
        artifact.source_artifact_digest = digest(&artifact.weight_id);
        artifact.publication_artifact_digest = digest(&format!("encoded-{}", artifact.weight_id));
        identities.push(artifact);
    }
    let inventory = lower(&model, identities);
    // Native B=2 is not permission to drop the other body matrix or full head.
    assert!(admit_maverick_batch_publication(&inventory, request())
        .unwrap_err()
        .contains("omits matrix/full-head"));
    let mut req = request();
    let mut second = req.publications[0].clone();
    second.group_orders = [3, 4];
    req.publications.push(second);
    req.aggregate_ceiling_bytes = 2 * 6479872 - 1;
    assert!(admit_maverick_batch_publication(&inventory, req.clone()).is_err());
    req.aggregate_ceiling_bytes += 1;
    let result = admit_maverick_batch_publication(&inventory, req).unwrap();
    let value: Value = serde_json::from_slice(result.canonical_bytes()).unwrap();
    assert_eq!(value["aggregate_planned_peak_bytes"], 2 * 6479872);
    assert_eq!(
        value["request"]["publications"].as_array().unwrap().len(),
        2
    );
}

#[test]
fn grouped_source_member_order_and_offsets_bind_without_merging_artifact_identity() {
    let mut model = plan(64, false);
    for graph in [&mut model.prefill, &mut model.decode] {
        graph.operations.insert(
            2,
            op(
                "second-member",
                ModelOperator::Linear,
                Some(0),
                "opaque-a",
                vec![1, graph.query_sequence, 64],
                json!({"weight": "second-weight"}),
            ),
        );
    }
    let mut identities = artifacts(64, false);
    let mut second = identities[1].clone();
    second.weight_id = "second-weight".into();
    second.source_artifact_digest = digest("second-weight");
    second.publication_artifact_digest = digest("encoded-second-weight");
    identities.push(second);
    let baseline = lower(&model, identities.clone());
    let value = json_inventory(&baseline);
    let group = &value["inventory"]["prefill"]["groups"][1];
    assert_eq!(group["logical_shape"], json!([128, 64]));
    assert_eq!(group["members"][0]["weight_id"], "body");
    assert_eq!(group["members"][1]["output_offset"], 64);
    for graph in [&mut model.prefill, &mut model.decode] {
        graph.operations.swap(1, 2);
    }
    let changed = lower(&model, identities);
    assert_ne!(baseline.digest(), changed.digest());
    assert_eq!(
        json_inventory(&changed)["inventory"]["prefill"]["groups"][1]["members"][0]["weight_id"],
        "second-weight"
    );
    assert!(admit_maverick_batch_publication(&changed, request()).is_err());
}

#[test]
fn tied_source_keeps_table_and_full_head_distinct_publication_owners() {
    let mut model = plan(64, false);
    for graph in [&mut model.prefill, &mut model.decode] {
        graph.operations[3].attributes = json!({"weight": "table", "tied": true});
    }
    let mut identities = artifacts(64, false);
    identities[2].weight_id = "table".into();
    identities[2].source_artifact_digest = identities[0].source_artifact_digest.clone();
    let inventory = lower(&model, identities);
    let value = json_inventory(&inventory);
    assert_eq!(value["artifacts"].as_array().unwrap().len(), 3);
    assert_eq!(value["uses"][0]["artifact_position"], 0);
    assert_eq!(value["uses"][2]["artifact_position"], 2);
    assert_ne!(
        value["artifacts"][0]["publication_artifact_digest"],
        value["artifacts"][2]["publication_artifact_digest"]
    );
    assert_eq!(
        inventory.status(),
        PublicationAdmissionStatus::Blocked(PublicationBlockedGate::MaverickSourceNumericProfile)
    );
}

#[test]
fn replacement_plan_is_bounded_before_identity_hashing_or_serialization() {
    let model = plan(64, false);
    let inventory = lower_decoder_matrix_inventory(&model).unwrap();
    for case in 0..4 {
        let mut replacement = model.clone();
        let expected = match case {
            0 => {
                replacement.model_family = "x".repeat(1025);
                "metadata string limit"
            }
            1 => {
                replacement.prefill.operations[0].attributes = json!({"weight": "x".repeat(1025)});
                "metadata string limit"
            }
            2 => {
                let operation = replacement.prefill.operations[0].clone();
                replacement.prefill.operations = vec![operation; 4097];
                "operation or state limit"
            }
            3 => {
                replacement.prefill.operations[0].output_shape = vec![u64::MAX, 2];
                "shape product overflow"
            }
            _ => unreachable!(),
        };
        let error = lower_delegation_publication_inventory(
            &replacement,
            &inventory,
            source(&model),
            artifacts(64, false),
            DelegationPublicationContract::source_derived_maverick(),
        )
        .unwrap_err();
        assert!(error.contains(expected), "case {case}: {error}");
        assert!(!error.contains("identity mismatch"));
    }
}

fn generated_decoder() -> DecoderPlan {
    lower_model_json(
        &canonical_bytes(&json!({
            "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "vocab_size": 32,
            "max_position_embeddings": 32, "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": true
        })),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 3,
            max_new_tokens: 2,
        },
    )
    .unwrap()
}

fn declared_artifacts(inventory: &DecoderMatrixInventory) -> Vec<PublicationArtifactIdentity> {
    let document: InventoryDocument = serde_json::from_slice(inventory.canonical_bytes()).unwrap();
    let mut artifacts = BTreeMap::new();
    for group in &document.prefill.groups {
        for member in &group.members {
            artifacts
                .entry((group.artifact_kind, member.weight_id.clone()))
                .or_insert_with(|| PublicationArtifactIdentity {
                    kind: group.artifact_kind,
                    weight_id: member.weight_id.clone(),
                    source_artifact_digest: digest(&member.weight_id),
                    publication_artifact_digest: digest(&format!(
                        "encoded-{:?}-{}",
                        group.artifact_kind, member.weight_id
                    )),
                    source_shape: member.source_shape.to_vec(),
                    source_layout: group.orientation,
                });
            if let Some(id) = &member.bias_id {
                artifacts
                    .entry((PublicationArtifactKind::Bias, id.clone()))
                    .or_insert_with(|| PublicationArtifactIdentity {
                        kind: PublicationArtifactKind::Bias,
                        weight_id: id.clone(),
                        source_artifact_digest: digest(id),
                        publication_artifact_digest: digest(&format!("encoded-{id}")),
                        source_shape: member.bias_shape.unwrap().to_vec(),
                        source_layout: PublicationSourceLayout::OutputVector,
                    });
            }
        }
    }
    artifacts.into_values().collect()
}

#[test]
fn complete_generated_decoder_keeps_full_head_and_rejects_native_geometry() {
    let model = generated_decoder();
    model.validate().unwrap();
    let matrix_inventory = lower_decoder_matrix_inventory(&model).unwrap();
    let inventory = lower_delegation_publication_inventory(
        &model,
        &matrix_inventory,
        source(&model),
        declared_artifacts(&matrix_inventory),
        DelegationPublicationContract::source_derived_maverick(),
    )
    .unwrap();
    let groups = &inventory.manifest.inventory.prefill.groups;
    assert_eq!(groups.len(), 10); // Table, four groups/layer, complete head.
    let head = groups
        .iter()
        .find(|g| g.operator == ModelOperator::OutputHead)
        .unwrap();
    assert_eq!(head.logical_shape, [32, 8]);
    assert_eq!(head.members[0].source_shape, [32, 8]);
    assert_eq!(
        inventory.manifest.semantic_coverage.len(),
        model.prefill.operations.len() + model.decode.operations.len()
    );
    let mut req = request();
    // Put the actual complete head first: no synthetic stand-in, truncation,
    // tiling or missing-group error can disguise the unsupported head geometry.
    req.publications[0].group_orders = [head.order, 1];
    let error = admit_maverick_batch_publication(&inventory, req).unwrap_err();
    assert!(error.contains("matrix/full-head geometry unsupported"));
    assert_eq!(
        inventory.status(),
        PublicationAdmissionStatus::Blocked(PublicationBlockedGate::MaverickSourceNumericProfile)
    );
}

#[test]
fn fixed_legacy_runtime_schedule_digest_unchanged_by_publication_lowering() {
    let model = generated_decoder();
    let composition = canonical_bytes(&json!({
        "components": {
            "inference": {"component": "pllm/inference", "params": {}},
            "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
            "linear": {"component": "pllm/masked-linear", "params": {}},
            "preparation": {"component": "pllm/model-aware-corrections", "params": {}}
        },
        "model": {"source": "model.fixture"}
    }));
    let before =
        crate::decoder_runtime_schedule::lower_decoder_runtime_schedule(&model, &composition)
            .unwrap();
    const EXPECTED: &str = "9b2f3bed2dff2d83598bebff5cabf5a22f0ed30f470d8b7d35215d1aaa57719d";
    assert_eq!(before.digest().as_str(), EXPECTED);
    let matrix_inventory = lower_decoder_matrix_inventory(&model).unwrap();
    let publication = lower_delegation_publication_inventory(
        &model,
        &matrix_inventory,
        source(&model),
        declared_artifacts(&matrix_inventory),
        DelegationPublicationContract::source_derived_maverick(),
    )
    .unwrap();
    let after =
        crate::decoder_runtime_schedule::lower_decoder_runtime_schedule(&model, &composition)
            .unwrap();
    assert_eq!(after.digest().as_str(), EXPECTED);
    assert_eq!(before, after);
    assert_eq!(
        publication.status(),
        PublicationAdmissionStatus::Blocked(PublicationBlockedGate::MaverickSourceNumericProfile)
    );
    assert_eq!(
        HISTORICAL_MAVERICK_SCALAR_IDENTITY,
        "MaverickDelegatedLinear/seeded-raa/signed-i8/BabyBear/ty41/256MiB"
    );
}
