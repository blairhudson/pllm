//! Experiment-bound admission for the distinct verified ring64 runtime.
//! Ordinary masked/int8 scheduling cannot execute this contract.

use pllm_types::{canonical_bytes, canonical_digest, Digest};
use serde::Serialize;

use super::batch::SourceStageBatchContract;
use super::{reject, GkmDecoderContract};
use crate::{ExperimentDeployment, ExperimentDocument, ExperimentPipeline};
use crate::component_binding::SelectedComponents;

pub const PROTOCOL: &str = "pllm/open-weight-verifiable-outsourcing/v1";
pub const VERIFIER: &str = "pllm/published-linear-verify/v1";
pub const TOPOLOGY: &str = "pllm/two-online-delegation-workers/v1";
const SCHEMA: &str = "pllm.gkm.sdk-contract.v1";
const MAX_EXPERIMENT_BYTES: usize = 32 << 10;
const EXPLICIT_BINDINGS: [(&str, &str); 4] = [
    ("input", "pllm/two-share-input/v1"),
    ("lookup", "pllm/verified-public-f32-lookup/v1"),
    ("regions", "pllm/client-f32-decoder-regions/v1"),
    ("numeric", "pllm/q11-i16-ring64-boundary/v1"),
];

const TARGET_SLOTS: [&str; 17] = [
    "linear", "input", "lookup", "numeric", "rmsnorm", "rotary", "attention",
    "activation", "multiply", "bias", "add", "head", "selection", "verification",
    "topology", "state", "kernels",
];

fn validate_target(pipeline: &ExperimentPipeline) -> Result<(), String> {
    let expected = [
        ("linear", PROTOCOL, serde_json::json!({"capacity": "decoder", "sharing": "full", "generation": "fixed_length_greedy"})),
        ("input", "pllm/two-share-input/v1", serde_json::json!({})),
        ("lookup", "pllm/verified-public-f32-lookup/v1", serde_json::json!({})),
        ("numeric", "pllm/q11-i16-ring64-boundary/v1", serde_json::json!({})),
        ("rmsnorm", "pllm/f32-rmsnorm/v1", serde_json::json!({"epsilon": "source"})),
        ("rotary", "pllm/f32-rotary/v1", serde_json::json!({"configuration": "source"})),
        ("attention", "pllm/f32-causal-attention/v1", serde_json::json!({"heads": "source", "kv": "state"})),
        ("activation", "pllm/f32-activation/v1", serde_json::json!({"function": "source"})),
        ("multiply", "pllm/f32-multiply/v1", serde_json::json!({})),
        ("bias", "pllm/f32-add-bias/v1", serde_json::json!({"placement": "after-reconstruction"})),
        ("add", "pllm/f32-add/v1", serde_json::json!({})),
        ("head", "pllm/full-vocabulary-head/v1", serde_json::json!({"linear": "linear", "rows": "last-prefill-and-each-decode"})),
        ("selection", "pllm/greedy-selection/v1", serde_json::json!({"eos": "continue", "logits": "complete"})),
        ("verification", VERIFIER, serde_json::json!({"acceptance": "request"})),
        ("topology", TOPOLOGY, serde_json::json!({})),
        ("state", "pllm/client-local-kv", serde_json::json!({})),
        ("kernels", "pllm/cpu", serde_json::json!({"threads": 1})),
    ];
    if pipeline.components.len() != expected.len()
        || !matches!(pipeline.model.kind.as_str(), "huggingface" | "safetensors")
    {
        return Err(reject("unsupported SDK source or component composition"));
    }
    for (slot, identity, params) in expected {
        let binding = pipeline.components.get(slot)
            .ok_or_else(|| reject("incomplete SDK component bindings"))?;
        if binding.component != identity || serde_json::to_value(&binding.params).unwrap() != params {
            return Err(reject("incompatible SDK component binding"));
        }
    }
    Ok(())
}

pub(crate) fn selected(pipeline: &ExperimentPipeline) -> bool {
    pipeline
        .components
        .values()
        .any(|item| matches!(item.component.as_str(), PROTOCOL | VERIFIER | TOPOLOGY)
            || EXPLICIT_BINDINGS.iter().any(|(_, id)| item.component == *id))
}

#[cfg(test)]
mod batch_metadata_tests {
    use super::*;
    use pllm_core::gkm::{BatchResources, Capacity, SharingProfile};
    use serde_json::{json, Value};

    fn pipeline() -> ExperimentPipeline {
        serde_json::from_value(json!({
            "model": {"source": "fixture", "kind": "safetensors"},
            "components": {
                "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
                "linear": {"component": PROTOCOL, "params": {
                    "sharing": "full", "generation": "fixed_length_greedy"
                }},
                "verification": {"component": VERIFIER, "params": {"acceptance": "request"}},
                "topology": {"component": TOPOLOGY, "params": {}},
                "state": {"component": "pllm/client-local-kv", "params": {}}
            }
        }))
        .unwrap()
    }

    fn target_pipeline() -> ExperimentPipeline {
        // Independent literal wire fixture for examples/papers/gkm.py. Do not
        // derive expected parameters from the admission table under test.
        serde_json::from_value(json!({
            "model": {"source": "fixture", "kind": "safetensors"},
            "components": {
                "linear": {"component": "pllm/open-weight-verifiable-outsourcing/v1", "params": {
                    "capacity": "decoder", "sharing": "full", "generation": "fixed_length_greedy"}},
                "input": {"component": "pllm/two-share-input/v1", "params": {}},
                "lookup": {"component": "pllm/verified-public-f32-lookup/v1", "params": {}},
                "numeric": {"component": "pllm/q11-i16-ring64-boundary/v1", "params": {}},
                "rmsnorm": {"component": "pllm/f32-rmsnorm/v1", "params": {"epsilon": "source"}},
                "rotary": {"component": "pllm/f32-rotary/v1", "params": {"configuration": "source"}},
                "attention": {"component": "pllm/f32-causal-attention/v1", "params": {"heads": "source", "kv": "state"}},
                "activation": {"component": "pllm/f32-activation/v1", "params": {"function": "source"}},
                "multiply": {"component": "pllm/f32-multiply/v1", "params": {}},
                "bias": {"component": "pllm/f32-add-bias/v1", "params": {"placement": "after-reconstruction"}},
                "add": {"component": "pllm/f32-add/v1", "params": {}},
                "head": {"component": "pllm/full-vocabulary-head/v1", "params": {"linear": "linear", "rows": "last-prefill-and-each-decode"}},
                "selection": {"component": "pllm/greedy-selection/v1", "params": {"eos": "continue", "logits": "complete"}},
                "verification": {"component": "pllm/published-linear-verify/v1", "params": {"acceptance": "request"}},
                "topology": {"component": "pllm/two-online-delegation-workers/v1", "params": {}},
                "state": {"component": "pllm/client-local-kv", "params": {}},
                "kernels": {"component": "pllm/cpu", "params": {"threads": 1}}
            }
        })).unwrap()
    }

    fn target_document() -> Value {
        json!({"schema": "pllm.experiment.v2", "name": "target-selection",
            "pipeline": target_pipeline(), "deployment": {"kind": "local", "root": "target/gkm-test"},
            "budget": {"requests": 1, "max_input_tokens": 4, "max_new_tokens": 3}})
    }

    fn source_config() -> Vec<u8> {
        canonical_bytes(&json!({
            "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "vocab_size": 32, "max_position_embeddings": 16, "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": true
        }))
    }

    fn compile_document(document: &Value, config: &[u8], rows: usize, tokens: usize) -> Result<GkmRuntimeContract, String> {
        GkmRuntimeContract::compile(&canonical_bytes(document), config, rows, tokens,
            Digest::from_sha256([1; 32]), Digest::from_sha256([2; 32]))
    }

    #[test]
    fn exact_target_wire_map_rejects_every_slot_identity_parameter_and_link_mutation() {
        let fixture = serde_json::to_value(target_pipeline()).unwrap();
        validate_pipeline(&serde_json::from_value(fixture.clone()).unwrap()).unwrap();
        for slot in TARGET_SLOTS {
            let mut mutations = Vec::new();
            let mut missing = fixture.clone();
            missing["components"].as_object_mut().unwrap().remove(slot);
            mutations.push(missing);
            for field in ["component", "params"] {
                let mut missing = fixture.clone();
                missing["components"][slot].as_object_mut().unwrap().remove(field);
                mutations.push(missing);
            }
            for params in [Value::Null, json!([]), json!("wrong"), json!({})] {
                if params == fixture["components"][slot]["params"] { continue; }
                let mut changed = fixture.clone();
                changed["components"][slot]["params"] = params;
                mutations.push(changed);
            }
            for identity in [json!("unknown/implementation/v1"), fixture["components"]["input"]["component"].clone(), Value::Null] {
                if identity == fixture["components"][slot]["component"] { continue; }
                let mut changed = fixture.clone();
                changed["components"][slot]["component"] = identity;
                mutations.push(changed);
            }
            let mut extra = fixture.clone();
            extra["components"][slot]["params"]["unknown"] = json!(true);
            mutations.push(extra);
            for key in fixture["components"][slot]["params"].as_object().unwrap().keys() {
                let mut missing = fixture.clone();
                missing["components"][slot]["params"].as_object_mut().unwrap().remove(key);
                mutations.push(missing);
                for wrong in [Value::Null, json!("wrong-link-or-value"), json!(false), json!(2)] {
                    let mut changed = fixture.clone();
                    changed["components"][slot]["params"][key] = wrong;
                    mutations.push(changed);
                }
            }
            for mutation in mutations {
                if let Ok(pipeline) = serde_json::from_value::<ExperimentPipeline>(mutation) {
                    assert!(validate_pipeline(&pipeline).is_err(), "{slot}");
                    let mut document = target_document();
                    document["pipeline"] = serde_json::to_value(pipeline).unwrap();
                    let error = compile_document(&document, b"poisoned-config", 1, 1).err().unwrap();
                    assert_eq!(error, compile_document(&document, &source_config(), 1, 1).err().unwrap());
                }
            }
        }
        let mut extra = fixture;
        extra["components"]["regions"] = json!({"component": "pllm/client-f32-decoder-regions/v1", "params": {}});
        assert!(validate_pipeline(&serde_json::from_value(extra).unwrap()).is_err());
    }

    #[test]
    fn selected_target_envelope_and_actual_request_preserve_legacy_decoder_bytes() {
        let document = target_document();
        for (rows, tokens) in [(4, 3), (1, 1), (2, 2)] {
            let runtime = compile_document(&document, &source_config(), rows, tokens).unwrap();
            let legacy = GkmDecoderContract::compile_streamed_with_capacity(
                &source_config(), rows, tokens - 1, pllm_core::gkm::Capacity::Decoder).unwrap();
            assert_eq!(runtime.decoder.canonical_bytes(), legacy.canonical_bytes());
            assert_eq!(runtime.decoder.source_binding_digest(), legacy.source_binding_digest());
            assert!(runtime.decoder.validate_bindings().is_ok());
        }
        let mut oversized = document;
        oversized["budget"]["max_input_tokens"] = json!(513);
        assert!(compile_document(&oversized, &source_config(), 1, 1).is_err());
    }

    #[test]
    fn selected_capabilities_resolve_each_actual_operation_in_both_phases() {
        use crate::component_binding::{bind_selected, classify, validate_selected_operation};
        let pipeline = target_pipeline();
        validate_pipeline(&pipeline).unwrap();
        let descriptors = TARGET_SLOTS.map(|slot| (slot, pipeline.components[slot].component.as_str()));
        let runtime = compile_document(&target_document(), &source_config(), 2, 2).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for (phase, graph) in [("prefill", &runtime.decoder.plan.prefill), ("decode", &runtime.decoder.plan.decode)] {
            for operation in &graph.operations {
                let implementation = bind_selected(operation.operator, SelectedComponents::Explicit(&descriptors)).unwrap();
                assert_eq!(implementation, classify(operation.operator).unwrap());
                assert_eq!(runtime.decoder.bound_step(phase, &operation.id), Ok(implementation as u8));
                validate_selected_operation(graph, operation, implementation).unwrap();
                seen.insert(implementation as u8);
                let mut unknown_attribute = operation.clone();
                unknown_attribute.attributes["unconsumed"] = json!(true);
                assert!(validate_selected_operation(graph, &unknown_attribute, implementation).is_err());
                // Exactly one selected capability supplies each operation code;
                // changing that capability must fail, not fall back to classify.
                let failures = descriptors.iter().enumerate().filter(|(index, _)| {
                    let mut changed = descriptors;
                    changed[*index].1 = "unknown/implementation/v1";
                    bind_selected(operation.operator, SelectedComponents::Explicit(&changed)).is_err()
                }).count();
                assert_eq!(failures, 1, "{}", operation.id);
                assert!(bind_selected(operation.operator, SelectedComponents::Explicit(&[])).is_err());
            }
        }
        assert_eq!(seen, (1u8..=19).collect());
        use pllm_models::ModelOperator as O;
        for unsupported in [
            O::AttentionImportance, O::SecureTopK, O::SecureGather, O::ClusterBounds,
            O::ClusterSimilarity, O::SharedIndices, O::CacheActiveIndices, O::Scale,
            O::Slice, O::GeluTanh, O::Softcap, O::StateInitialize, O::Permute,
            O::Sigmoid, O::GatedDeltaDecay, O::CausalConvolution,
            O::ConvolutionStateUpdate, O::GatedDeltaRule, O::GatedDeltaStateUpdate,
            O::RmsNormGated, O::LayerNorm, O::GeluErf, O::Tanh, O::AttentionMask,
        ] {
            assert!(bind_selected(unsupported, SelectedComponents::Explicit(&descriptors)).is_err());
        }
    }

    #[test]
    fn selected_semantics_reject_shape_relationship_and_policy_mutations() {
        use crate::component_binding::{classify, validate_selected_operation};
        let runtime = compile_document(&target_document(), &source_config(), 2, 2).unwrap();
        for graph in [&runtime.decoder.plan.prefill, &runtime.decoder.plan.decode] {
            for operation in &graph.operations {
                let implementation = classify(operation.operator).unwrap();
                for key in operation.attributes.as_object().unwrap().keys() {
                    // Source weight/bias names are checked by member/source
                    // construction, not by this method-neutral semantic helper.
                    if matches!(key.as_str(), "weight" | "bias") { continue; }
                    let mut changed = operation.clone();
                    changed.attributes[key] = Value::Null;
                    assert!(validate_selected_operation(graph, &changed, implementation).is_err(),
                        "{}: {key}", operation.id);
                }
                if matches!(operation.operator, pllm_models::ModelOperator::RmsNorm
                    | pllm_models::ModelOperator::Silu | pllm_models::ModelOperator::Multiply
                    | pllm_models::ModelOperator::ResidualAdd | pllm_models::ModelOperator::Softmax
                    | pllm_models::ModelOperator::CausalMask | pllm_models::ModelOperator::RotaryEmbedding)
                {
                    let mut changed = operation.clone();
                    *changed.output_shape.last_mut().unwrap() += 1;
                    assert!(validate_selected_operation(graph, &changed, implementation).is_err());
                }
            }
        }
    }

    #[test]
    fn selected_target_refuses_nonserial_kernel_and_partial_or_unsupported_source() {
        for threads in [json!(0), json!(2), json!(1.0), Value::Null] {
            let mut document = target_document();
            document["pipeline"]["components"]["kernels"]["params"]["threads"] = threads;
            assert!(compile_document(&document, &source_config(), 1, 1).is_err());
        }
        let mut document = target_document();
        document["pipeline"]["components"]["kernels"]["component"] = json!("pllm/apple-metal-int8/v1");
        assert!(compile_document(&document, &source_config(), 1, 1).is_err());
        document = target_document();
        document["pipeline"]["components"]["linear"]["component"] = json!("pllm/maverick-delegated-linear/v1");
        assert!(compile_document(&document, &source_config(), 1, 1).is_err());
        for model_type in ["maverick", "llama4"] {
            let mut config: Value = serde_json::from_slice(&source_config()).unwrap();
            config["model_type"] = json!(model_type);
            assert!(compile_document(&target_document(), &canonical_bytes(&config), 1, 1).is_err());
        }
        let mut config: Value = serde_json::from_slice(&source_config()).unwrap();
        config["hidden_act"] = json!("gelu");
        assert!(compile_document(&target_document(), &canonical_bytes(&config), 1, 1).is_err());
    }

    #[test]
    fn selected_member_source_bias_and_state_relationships_remain_checked() {
        for mutation in 0..6 {
            let mut runtime = compile_document(&target_document(), &source_config(), 2, 2).unwrap();
            match mutation {
                0 => runtime.decoder.matrices[0].members[0].bias = Some("unbound.bias".into()),
                1 => runtime.decoder.matrices[0].members[0].weight = "unbound.weight".into(),
                2 => { runtime.decoder.source_shapes.remove(&runtime.decoder.table); },
                3 => runtime.decoder.plan.decode.state_inputs[0].kind = pllm_models::StateKind::Recurrent,
                4 => runtime.decoder.plan.prefill.state_outputs[0].kind = pllm_models::StateKind::CacheIndices,
                _ => runtime.decoder.decode.steps[0].executor = crate::decoder_runtime_schedule::DecoderRuntimeExecutor::ClientLocal,
            }
            assert!(runtime.decoder.validate_bindings().is_err());
        }
    }

    #[test]
    fn selected_cpu_declaration_and_existing_ring64_kernel_contract() {
        let runtime = compile_document(&target_document(), &source_config(), 1, 1).unwrap();
        assert_eq!(runtime.decoder.numeric_profile, pllm_core::gkm::NUMERIC_PROFILE);
        // Existing worker scalar/batch paths both call this serial native kernel;
        // Cpu(threads=1) does not impose a NumPy/Rayon or worker-launch limit.
        let matrix = pllm_core::gkm::I16Matrix::from_encoded(1, 2, &[-2, 3], &[4, 4]).unwrap();
        assert_eq!(matrix.multiply_share(&[1, 2]).unwrap().as_slice(), &[4]);
        let signed = pllm_core::gkm::reconstruct_signed(u64::MAX, 0);
        assert_eq!(signed, -1);
        assert_eq!(pllm_core::gkm::rescale_signed(signed), (-1i64 as f32) / 4_194_304.0);
    }

    fn runtime(capacity: Capacity, profile: SharingProfile, streamed: bool) -> GkmRuntimeContract {
        let config = canonical_bytes(&json!({
            "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "vocab_size": 32, "max_position_embeddings": 16, "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": true
        }));
        GkmRuntimeContract {
            schema_version: SCHEMA,
            configuration: Digest::from_sha256([1; 32]),
            composition: Digest::from_sha256([2; 32]),
            source_lock: Digest::from_sha256([3; 32]),
            tokenizer: Digest::from_sha256([4; 32]),
            decoder: GkmDecoderContract::compile_owned(&config, 2, 1, capacity, streamed, SelectedComponents::Legacy).unwrap(),
            sharing_profile: (profile == SharingProfile::Aes128WindowV1)
                .then_some(pllm_core::gkm::SEEDED_SHARING_PROFILE),
            batch_execution: None,
        }
    }

    #[test]
    fn batch_quote_consumes_core_totals_for_both_roles_and_profiles() {
        for capacity in [Capacity::Capsule, Capacity::Decoder] {
            for profile in [SharingProfile::FullShares, SharingProfile::Aes128WindowV1] {
                let runtime = runtime(capacity, profile, true);
                let decoder = runtime.decoder();
                let plan = decoder.batch_plan().unwrap();
                let stages: Vec<_> = decoder
                    .matrices
                    .iter()
                    .map(|matrix| (super::super::stage_id(matrix), matrix.rows, matrix.columns))
                    .collect();
                let native = BatchResources::for_streamed_single_window(
                    &stages,
                    decoder.vocabulary,
                    decoder.width,
                    &plan,
                    profile,
                    capacity,
                )
                .unwrap();
                let quote = runtime.source_stage_batch_quote().unwrap();
                let metadata = serde_json::to_value(quote).unwrap();
                for (key, expected) in [
                    ("max_aggregate_bytes", native.max_aggregate_bytes),
                    ("retained_frame_bytes", native.retained_frame_bytes),
                    ("transport_scratch_bytes", native.transport_scratch_bytes),
                    ("extra_native_owner_bytes", native.extra_native_owner_bytes),
                    ("extra_wire_bytes", native.extra_wire_bytes),
                ] {
                    assert_eq!(metadata["resources"][key], json!(expected));
                }
                let delta = native.extra_native_owner_bytes + native.extra_wire_bytes;
                assert_eq!(
                    metadata["native_admitted_payload_bytes"],
                    json!(
                        decoder
                            .native_publication_resources()
                            .unwrap()
                            .admitted_payload_bytes
                            + native.extra_native_owner_bytes
                    )
                );
                assert_eq!(
                    metadata["client_payload_bytes"],
                    json!(
                        decoder.execution_payload_estimate_bytes
                            + delta
                            + metadata["compiler_encoded_input_peak_bytes"]
                                .as_u64()
                                .unwrap() as usize
                    )
                );
                let scalar_worker = decoder
                    .decoder_resources
                    .as_ref()
                    .map_or(decoder.execution_payload_estimate_bytes, |resources| {
                        resources.worker_payload_bytes
                    });
                assert_eq!(
                    metadata["worker_payload_bytes"],
                    json!(scalar_worker + delta)
                );
                assert_eq!(
                    metadata["batch_plan"],
                    json!(Digest::from_sha256(plan.digest()))
                );
                assert_eq!(metadata["execution"], "source_stage_batch_v1");
                assert_eq!(
                    metadata["sharing_profile"],
                    match profile {
                        SharingProfile::FullShares => "full",
                        SharingProfile::Aes128WindowV1 => pllm_core::gkm::SEEDED_SHARING_PROFILE,
                    }
                );
            }
        }
    }

    #[test]
    fn batch_metadata_binds_sdk_identity_only_and_scalar_omission_is_exact() {
        let mut runtime = runtime(Capacity::Decoder, SharingProfile::FullShares, true);
        let scalar_bytes = runtime.canonical_bytes();
        let scalar_digest = runtime.digest();
        let decoder_bytes = runtime.decoder.canonical_bytes();
        let source_digest = runtime.decoder.source_binding_digest();
        let value: Value = serde_json::from_slice(&scalar_bytes).unwrap();
        assert!(value.get("batch_execution").is_none());
        let quote = runtime.source_stage_batch_quote().unwrap();
        assert_eq!(runtime.canonical_bytes(), scalar_bytes); // Quoting cannot select execution.
        runtime.batch_execution = Some(quote); // Private test-only installation.
        assert_ne!(runtime.digest(), scalar_digest);
        assert_eq!(runtime.decoder.canonical_bytes(), decoder_bytes);
        assert_eq!(runtime.decoder.source_binding_digest(), source_digest);
        runtime.batch_execution = None;
        assert_eq!(runtime.canonical_bytes(), scalar_bytes);
        assert_eq!(runtime.digest(), scalar_digest);
    }

    #[test]
    fn batch_metadata_rejects_nonstreamed_and_unknown_execution_selector() {
        let runtime = runtime(Capacity::Capsule, SharingProfile::FullShares, false);
        assert!(runtime.source_stage_batch_quote().is_err());
        let mut pipeline = pipeline();
        validate_pipeline(&pipeline).unwrap();
        pipeline
            .components
            .get_mut("linear")
            .unwrap()
            .params
            .insert("execution".to_owned(), json!("source_stage_batch_v1"));
        validate_pipeline(&pipeline).unwrap();
        pipeline
            .components
            .get_mut("linear")
            .unwrap()
            .params
            .insert("execution".to_owned(), json!("unknown"));
        assert!(validate_pipeline(&pipeline).is_err());
    }
}

pub(crate) fn validate_pipeline(pipeline: &ExperimentPipeline) -> Result<(), String> {
    if pipeline.components.len() == TARGET_SLOTS.len() {
        return validate_target(pipeline);
    }
    let expected = [
        ("kernels", "pllm/cpu", serde_json::json!({"threads": 1})),
        (
            "linear",
            PROTOCOL,
            serde_json::json!({
                "sharing": "full", "generation": "fixed_length_greedy"
            }),
        ),
        ("topology", TOPOLOGY, serde_json::json!({})),
        ("state", "pllm/client-local-kv", serde_json::json!({})),
        (
            "verification",
            VERIFIER,
            serde_json::json!({"acceptance": "request"}),
        ),
    ];
    // Legacy five imply these same fixed implementations internally. Never
    // modify the document: its original bytes remain configuration authority.
    if pipeline.components.len() != expected.len() {
        if pipeline.components.len() != expected.len() + EXPLICIT_BINDINGS.len() {
            return Err(reject("incomplete SDK component bindings"));
        }
        for (slot, identity) in EXPLICIT_BINDINGS {
            let binding = pipeline.components.get(slot)
                .ok_or_else(|| reject("incomplete SDK component bindings"))?;
            if binding.component != identity || !binding.params.is_empty() {
                return Err(reject("incompatible SDK component binding"));
            }
        }
    }
    if !matches!(pipeline.components.len(), 5 | 9)
        || !matches!(pipeline.model.kind.as_str(), "huggingface" | "safetensors")
    {
        return Err(reject("unsupported SDK source or component composition"));
    }
    for (slot, identity, params) in expected {
        let component = pipeline
            .components
            .get(slot)
            .ok_or_else(|| reject("incomplete SDK composition"))?;
        let mut actual = serde_json::to_value(&component.params).unwrap();
        if slot == "linear" {
            if let Some(execution) = actual.as_object_mut().unwrap().remove("execution") {
                if execution != "source_stage_batch_v1" {
                    return Err(reject("unsupported SDK execution"));
                }
            }
            if actual.get("sharing").and_then(serde_json::Value::as_str) == Some("seeded") {
                actual["sharing"] = serde_json::json!("full");
            }
            if let Some(capacity) = actual.as_object_mut().unwrap().remove("capacity") {
                if capacity != "decoder" {
                    return Err(reject("unsupported SDK capacity"));
                }
            }
        }
        if component.component != identity || actual != params {
            return Err(reject(
                "incompatible SDK numerical, role, kernel or acceptance choice",
            ));
        }
    }
    Ok(())
}

fn capacity(pipeline: &ExperimentPipeline) -> pllm_core::gkm::Capacity {
    if pipeline.components["linear"]
        .params
        .get("capacity")
        .and_then(serde_json::Value::as_str)
        == Some("decoder")
    {
        pllm_core::gkm::Capacity::Decoder
    } else {
        pllm_core::gkm::Capacity::Capsule
    }
}

pub(crate) fn validate_experiment(document: &ExperimentDocument) -> Result<(), String> {
    validate_pipeline(&document.pipeline)?;
    if !matches!(document.deployment, ExperimentDeployment::Local { .. }) {
        return Err(reject("network-planned delegation is not admitted"));
    }
    let expanded = capacity(&document.pipeline) == pllm_core::gkm::Capacity::Decoder;
    if !(1..=if expanded { 512 } else { 32 }).contains(&document.budget.max_input_tokens)
        || !(1..=if expanded { 65 } else { 9 }).contains(&document.budget.max_new_tokens)
        || !(1..=64).contains(&document.budget.requests)
    {
        return Err(reject("SDK workload exceeds the bounded request envelope"));
    }
    Ok(())
}

/// Actual-byte source and tokenizer pins are supplied by the trusted SDK loader.
/// This record binds them; it is not a substitute for checking source files.
#[derive(Serialize)]
pub struct GkmRuntimeContract {
    schema_version: &'static str,
    configuration: Digest,
    composition: Digest,
    source_lock: Digest,
    tokenizer: Digest,
    decoder: GkmDecoderContract,
    #[serde(skip_serializing_if = "Option::is_none")]
    sharing_profile: Option<&'static str>,
    // Execution identity is separate from the unchanged numeric/source contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    batch_execution: Option<SourceStageBatchContract>,
}

impl GkmRuntimeContract {
    pub fn compile(
        experiment: &[u8],
        config: &[u8],
        input_rows: usize,
        output_tokens: usize,
        source_lock: Digest,
        tokenizer: Digest,
    ) -> Result<Self, String> {
        if experiment.is_empty() || experiment.len() > MAX_EXPERIMENT_BYTES {
            return Err(reject("SDK configuration byte limit"));
        }
        let document: ExperimentDocument =
            serde_json::from_slice(experiment).map_err(|_| reject("invalid SDK Experiment"))?;
        if canonical_bytes(&document) != experiment {
            return Err(reject("SDK Experiment must be canonical"));
        }
        crate::validate_experiment(&document)?;
        crate::validate_experiment_model(&document.pipeline.model)?;
        validate_experiment(&document)?;
        if input_rows == 0
            || input_rows as u64 > document.budget.max_input_tokens
            || output_tokens == 0
            || output_tokens as u64 > document.budget.max_new_tokens
            || [source_lock.as_str(), tokenizer.as_str()]
                .iter()
                .any(|s| s.bytes().all(|b| b == b'0'))
        {
            return Err(reject(
                "SDK request or source pins differ from admitted envelope",
            ));
        }
        // Admit the complete declared envelope before lowering the actual query.
        // A small request must never disguise an oversized saved configuration.
        let capacity = capacity(&document.pipeline);
        let descriptors = (document.pipeline.components.len() == TARGET_SLOTS.len()).then(|| {
            TARGET_SLOTS.map(|slot| (slot, document.pipeline.components[slot].component.as_str()))
        });
        let selected = descriptors.as_ref().map_or(SelectedComponents::Legacy, |items| {
            SelectedComponents::Explicit(items)
        });
        let envelope = GkmDecoderContract::compile_owned(
            config,
            document.budget.max_input_tokens as usize,
            document.budget.max_new_tokens as usize - 1,
            capacity,
            true,
            selected,
        )?;
        let batch = document.pipeline.components["linear"]
            .params
            .contains_key("execution");
        let profile = if document.pipeline.components["linear"]
            .params
            .get("sharing")
            .and_then(serde_json::Value::as_str)
            == Some("seeded")
        {
            pllm_core::gkm::SharingProfile::Aes128WindowV1
        } else {
            pllm_core::gkm::SharingProfile::FullShares
        };
        if batch {
            SourceStageBatchContract::derive(&envelope, profile)?;
        }
        let decoder = if input_rows as u64 == document.budget.max_input_tokens
            && output_tokens as u64 == document.budget.max_new_tokens
        {
            envelope
        } else {
            drop(envelope);
            GkmDecoderContract::compile_owned(
                config,
                input_rows,
                output_tokens - 1,
                capacity,
                true,
                selected,
            )?
        };
        let batch_execution = if batch {
            Some(SourceStageBatchContract::derive(&decoder, profile)?)
        } else {
            None
        };
        Ok(Self {
            schema_version: SCHEMA,
            configuration: crate::configuration_digest_bytes(experiment),
            composition: crate::pipeline_digest(&document.pipeline),
            source_lock,
            tokenizer,
            decoder,
            sharing_profile: (document.pipeline.components["linear"]
                .params
                .get("sharing")
                .and_then(serde_json::Value::as_str)
                == Some("seeded"))
            .then_some(pllm_core::gkm::SEEDED_SHARING_PROFILE),
            batch_execution,
        })
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        canonical_bytes(self)
    }
    pub fn digest(&self) -> Digest {
        canonical_digest(SCHEMA, self)
    }
    pub fn decoder(&self) -> &GkmDecoderContract {
        &self.decoder
    }

    pub fn execution(&self) -> Option<&'static str> {
        self.batch_execution
            .as_ref()
            .map(|_| "source_stage_batch_v1")
    }

    /// Scalar starts and each scalar-equivalent row retain this bound.
    pub fn scalar_frame_bytes(&self) -> usize {
        self.decoder.capacity().max_wire_frame_bytes()
    }

    pub fn max_aggregate_bytes(&self) -> Option<usize> {
        self.batch_execution
            .as_ref()
            .map(SourceStageBatchContract::max_aggregate_bytes)
    }

    pub fn max_operation_frame_bytes(&self) -> usize {
        self.scalar_frame_bytes()
            .max(self.max_aggregate_bytes().unwrap_or(0))
    }

    /// Selected batch pricing/descriptor identity for outer-language admission.
    /// This quote is not a native reservation or a transferable capability.
    pub fn batch_execution_bytes(&self) -> Option<Vec<u8>> {
        self.batch_execution.as_ref().map(canonical_bytes)
    }

    pub(super) fn batch_admission(
        &self,
        publication: &pllm_core::gkm::Publication,
    ) -> Result<Option<pllm_core::gkm::BatchAdmission>, String> {
        let Some(metadata) = &self.batch_execution else {
            return Ok(None);
        };
        let admission = pllm_core::gkm::BatchAdmission::new(
            publication,
            self.decoder.batch_plan()?,
            self.sharing_profile(),
        )
        .map_err(|error| error.to_string())?;
        metadata.check_resources(admission.resources())?;
        Ok(Some(admission))
    }

    /// Metadata only: does not install an execution mode on this SDK contract.
    #[cfg(test)]
    pub(super) fn source_stage_batch_quote(&self) -> Result<SourceStageBatchContract, String> {
        SourceStageBatchContract::derive(&self.decoder, self.sharing_profile())
    }

    pub fn sharing_profile(&self) -> pllm_core::gkm::SharingProfile {
        if self.sharing_profile.is_some() {
            pllm_core::gkm::SharingProfile::Aes128WindowV1
        } else {
            pllm_core::gkm::SharingProfile::FullShares
        }
    }
}
