//! Exact coupled boundary admission; prepared body scheduling remains unchanged.
use crate::ExperimentPipeline;
use pllm_models::{DecoderGraph, DecoderPlan, ModelOperation, ModelOperator, StateKind};
use pllm_types::{pipeline_digest_bytes, Digest};
use serde::{Deserialize, Serialize};

pub(crate) const COMPONENT: &str = "pllm/private-lookup-full-head/v1";
pub(crate) const TOPOLOGY: &str = "pllm/prepared-body-two-online-boundary/v1";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrivateBoundaryPhase {
    pub lookup_operation: String,
    pub head_operation: String,
    pub lookup_rows: u64,
    pub head_rows: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrivateBoundaryContract {
    pub schema_version: String,
    pub model_plan_digest: Digest,
    pub model_config_digest: Digest,
    pub composition_digest: Digest,
    pub weight_id: String,
    pub vocabulary: u64,
    pub hidden_width: u64,
    pub owners: [String; 2],
    pub lookup_protocol: String,
    pub head_protocol: String,
    #[serde(default = "serial", skip_serializing_if = "is_serial")]
    pub head_dispatch: String,
    #[serde(default = "u32_encoding", skip_serializing_if = "is_u32_encoding")]
    pub head_output_encoding: String,
    #[serde(
        default = "existing_kernel",
        skip_serializing_if = "is_existing_kernel"
    )]
    pub head_kernel: String,
    pub weight_bits: u64,
    pub activation_bits: u64,
    pub head_ring_bits: u64,
    pub head_signed_output_bound: u64,
    pub embedding_numeric: String,
    pub head_numeric: String,
    pub lookup_batch_bound: u64,
    pub lookup_query_bound: u64,
    pub matrix_bytes: u64,
    pub scale_bytes: u64,
    pub lookup_scratch_bytes: u64,
    pub client_weight_bytes: u64,
    pub prefill: PrivateBoundaryPhase,
    pub decode: PrivateBoundaryPhase,
}

fn serial() -> String {
    "serial".into()
}
fn u32_encoding() -> String {
    "u32".into()
}
fn is_serial(value: &str) -> bool {
    value == "serial"
}
fn is_u32_encoding(value: &str) -> bool {
    value == "u32"
}

fn existing_kernel() -> String {
    "existing".into()
}
fn is_existing_kernel(value: &str) -> bool {
    value == "existing"
}

fn options(pipeline: &ExperimentPipeline) -> Result<(String, String, String), String> {
    let params = &pipeline
        .components
        .get("boundary")
        .ok_or("private boundary component is absent")?
        .params;
    if params.keys().any(|key| {
        !matches!(
            key.as_str(),
            "head_dispatch" | "head_output_encoding" | "head_kernel"
        )
    }) {
        return Err("unknown private boundary option".into());
    }
    let choice = |key: &str, default: &str, allowed: &[&str]| -> Result<String, String> {
        match params.get(key) {
            None => Ok(default.into()),
            Some(value) => value
                .as_str()
                .filter(|s| allowed.contains(s))
                .map(str::to_owned)
                .ok_or_else(|| format!("unsupported private boundary {key}")),
        }
    };
    Ok((
        choice("head_dispatch", "serial", &["serial", "parallel"])?,
        choice("head_output_encoding", "u32", &["u32", "uniform_residues"])?,
        choice("head_kernel", "existing", &["existing", "byte_dot"])?,
    ))
}

pub(crate) fn selected(pipeline: &ExperimentPipeline) -> bool {
    pipeline
        .components
        .get("boundary")
        .is_some_and(|c| c.component == COMPONENT)
}

pub(crate) fn validate_composition(pipeline: &ExperimentPipeline) -> Result<(), String> {
    options(pipeline)?;
    let topology = pipeline.components.get("topology");
    if !topology.is_some_and(|c| c.component == TOPOLOGY && c.params.is_empty())
        || !pipeline
            .components
            .get("kernels")
            .is_some_and(|c| c.component == "pllm/cpu")
        || [
            "verification",
            "nonlinear",
            "scheduler",
            "state",
            "cache",
            "public_prefix",
            "placement",
        ]
        .iter()
        .any(|slot| pipeline.components.contains_key(*slot))
        || pipeline.components.get("inventory").is_some_and(|c| {
            c.params.contains_key("decode_epoch_tokens")
                || c.params.contains_key("decode_epoch_schedule")
        })
        || pipeline.components.get("quantization").is_some_and(|c| {
            c.component != "pllm/symmetric-per-row-quantization/v1"
                || c.params
                    .get("weight_bits")
                    .and_then(serde_json::Value::as_u64)
                    != Some(8)
                || c.params
                    .get("activation_bits")
                    .and_then(serde_json::Value::as_u64)
                    != Some(8)
        })
    {
        return Err("private boundary requires explicit prepared boundary roles, CPU W8A8 and ordinary prepared body without verification, state reuse, regions or epochs".into());
    }
    Ok(())
}

fn unique(graph: &DecoderGraph, operator: ModelOperator) -> Result<&ModelOperation, String> {
    let mut rows = graph.operations.iter().filter(|op| op.operator == operator);
    let row = rows
        .next()
        .ok_or("private boundary is missing its semantic operation")?;
    if rows.next().is_some() {
        return Err("private boundary requires one main lookup and one full head".into());
    }
    Ok(row)
}

fn phase(graph: &DecoderGraph) -> Result<(PrivateBoundaryPhase, String, u64, u64), String> {
    let lookup = unique(graph, ModelOperator::TokenLookup)?;
    let head = unique(graph, ModelOperator::OutputHead)?;
    let weight = lookup
        .attributes
        .get("weight")
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or("private boundary lookup lacks weight identity")?;
    if lookup.inputs.as_slice() != ["input.tokens"]
        || lookup
            .attributes
            .as_object()
            .is_none_or(|attrs| attrs.len() != 1)
        || lookup.layer.is_some()
        || head.layer.is_some()
        || lookup.state_kind.is_some()
        || head.state_kind.is_some()
        || lookup.output_shape.len() != 3
        || !matches!(head.output_shape.len(), 2 | 3)
        || lookup.output_shape[0] != 1
        || head.output_shape[0] != 1
        || lookup.output_shape[1] != graph.query_sequence
        || head
            .attributes
            .get("weight")
            .and_then(serde_json::Value::as_str)
            != Some(weight)
        || head
            .attributes
            .get("tied")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || lookup.attributes.get("bias").is_some_and(|v| !v.is_null())
        || head.attributes.get("bias").is_some_and(|v| !v.is_null())
        || graph
            .state_outputs
            .iter()
            .any(|state| !matches!(state.kind, StateKind::Key | StateKind::Value))
        || graph.operations.iter().any(|op| {
            !matches!(
                op.operator,
                ModelOperator::TokenLookup
                    | ModelOperator::Linear
                    | ModelOperator::OutputHead
                    | ModelOperator::Reshape
                    | ModelOperator::RmsNorm
                    | ModelOperator::RotaryEmbedding
                    | ModelOperator::KvCacheAppend
                    | ModelOperator::CacheSuffix
                    | ModelOperator::AttentionScores
                    | ModelOperator::AttentionScale
                    | ModelOperator::CausalMask
                    | ModelOperator::Softmax
                    | ModelOperator::AttentionValues
                    | ModelOperator::ResidualAdd
                    | ModelOperator::Silu
                    | ModelOperator::Multiply
                    | ModelOperator::LastToken
                    | ModelOperator::GreedyTokenSelection
                    | ModelOperator::TokenFeedback
            )
        })
        || graph.operations.iter().any(|op| {
            op.attributes
                .get("output_dtype")
                .is_some_and(|v| !matches!(v.as_str(), Some("float32" | "model_native")))
        })
    {
        return Err(
            "private boundary requires a tied dense float32 main lookup and full head".into(),
        );
    }
    let region = crate::lower_model_output_head_region(graph, graph.mode, head)?;
    let hidden = lookup.output_shape[2];
    let vocabulary = *head.output_shape.last().unwrap();
    if hidden == 0 || vocabulary == 0 || region.input.shape.last() != Some(&hidden) {
        return Err("private boundary lookup/head dimensions differ".into());
    }
    Ok((
        PrivateBoundaryPhase {
            lookup_operation: lookup.id.clone(),
            head_operation: head.id.clone(),
            lookup_rows: lookup.output_shape[1],
            head_rows: if head.output_shape.len() == 2 {
                1
            } else {
                head.output_shape[1]
            },
        },
        weight.into(),
        vocabulary,
        hidden,
    ))
}

pub(crate) fn lower(
    plan: &DecoderPlan,
    composition: &[u8],
) -> Result<PrivateBoundaryContract, String> {
    let pipeline: ExperimentPipeline = serde_json::from_slice(composition)
        .map_err(|error| format!("invalid private boundary composition: {error}"))?;
    validate_composition(&pipeline)?;
    let (head_dispatch, head_output_encoding, head_kernel) = options(&pipeline)?;
    if !plan.transformations.is_empty() || !plan.token_feedback {
        return Err("private boundary requires original full-vocabulary token feedback".into());
    }
    let (prefill, weight_id, vocabulary, hidden_width) = phase(&plan.prefill)?;
    let (decode, decode_weight, decode_vocabulary, decode_hidden) = phase(&plan.decode)?;
    if (decode_weight.as_str(), decode_vocabulary, decode_hidden)
        != (weight_id.as_str(), vocabulary, hidden_width)
    {
        return Err("private boundary prefill/decode artifacts differ".into());
    }
    let matrix_bytes = vocabulary
        .checked_mul(hidden_width)
        .ok_or("private boundary matrix size overflow")?;
    let scale_bytes = vocabulary
        .checked_mul(4)
        .ok_or("private boundary scale size overflow")?;
    if head_kernel == "byte_dot" && (hidden_width > 131071 || matrix_bytes > 256 * 1024 * 1024) {
        return Err("private boundary byte-dot geometry exceeds native bounds".into());
    }
    let lookup_scratch_bytes = vocabulary
        .checked_mul(64)
        .ok_or("private boundary scratch size overflow")?;
    let head_signed_output_bound = hidden_width
        .checked_mul(127 * 127)
        .ok_or("private boundary head bound overflow")?;
    let head_ring_bits = [16_u64, 24, 32]
        .into_iter()
        .find(|bits| head_signed_output_bound < (1_u64 << (bits - 1)))
        .ok_or("private boundary head exceeds exact wrapping ring")?;
    let lookup_query_bound = prefill
        .lookup_rows
        .checked_add(
            plan.decode
                .maximum_key_sequence
                .checked_sub(plan.prefill.query_sequence)
                .ok_or("private boundary sequence bounds differ")?,
        )
        .ok_or("private boundary query bound overflow")?;
    if vocabulary > 262_144
        || hidden_width > 65_532
        || matrix_bytes
            .checked_add(scale_bytes)
            .is_none_or(|n| n > 512 * 1024 * 1024)
        || lookup_query_bound > 4096
    {
        return Err("private boundary exceeds native table or one-use query bounds".into());
    }
    Ok(PrivateBoundaryContract {
        schema_version: "pllm.private_boundary.v1".into(),
        model_plan_digest: plan.digest(),
        model_config_digest: plan.config_digest.clone(),
        composition_digest: pipeline_digest_bytes(composition),
        weight_id,
        vocabulary,
        hidden_width,
        owners: ["boundary_a".into(), "boundary_b".into()],
        lookup_protocol: "packed128_xor_dpf_v2".into(),
        head_protocol: "two_party_additive_full_head_v1".into(),
        head_dispatch,
        head_output_encoding,
        head_kernel,
        weight_bits: 8,
        activation_bits: 8,
        head_ring_bits,
        head_signed_output_bound,
        embedding_numeric: "tied_per_token_row_f32_v1".into(),
        head_numeric: "ordinary_symmetric_per_row_integer_then_f32_v1".into(),
        lookup_batch_bound: 64,
        lookup_query_bound,
        matrix_bytes,
        scale_bytes,
        lookup_scratch_bytes,
        client_weight_bytes: 0,
        prefill,
        decode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_models::{lower_model_json, DecoderWorkload};
    use pllm_types::canonical_bytes;
    use serde_json::{json, Value};

    fn composition() -> Value {
        json!({"model":{"source":"model.fixture"},"components":{
            "linear":{"component":"pllm/masked-linear","params":{}},
            "inference":{"component":"pllm/inference","params":{}},
            "preparation":{"component":"pllm/model-aware-corrections","params":{}},
            "kernels":{"component":"pllm/cpu","params":{"threads":1}},
            "boundary":{"component":COMPONENT,"params":{}},
            "topology":{"component":TOPOLOGY,"params":{}}
        }})
    }

    fn plan(family: &str, tied: bool) -> DecoderPlan {
        let config = json!({"model_type":family,"hidden_size":8,"intermediate_size":16,
            "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
            "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":tied,
            "attention_bias":false,"attention_dropout":0.0,"rope_scaling":null,
            "sliding_window":null,"use_sliding_window":false,"use_cache":true,"max_window_layers":2});
        lower_model_json(
            &canonical_bytes(&config),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 3,
                max_new_tokens: 2,
            },
        )
        .unwrap()
    }

    #[test]
    fn private_boundary_transport_choices_preserve_numeric_contract_and_default_bytes() {
        let plan = plan("qwen2", true);
        let baseline =
            crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition())).unwrap();
        let baseline_value =
            serde_json::to_value(baseline.private_boundary.as_ref().unwrap()).unwrap();
        assert!(baseline_value.get("head_dispatch").is_none());
        assert!(baseline_value.get("head_output_encoding").is_none());
        assert!(baseline_value.get("head_kernel").is_none());
        let restored: PrivateBoundaryContract =
            serde_json::from_value(baseline_value.clone()).unwrap();
        assert_eq!(restored.head_dispatch, "serial");
        assert_eq!(restored.head_output_encoding, "u32");
        assert_eq!(restored.head_kernel, "existing");
        assert_eq!(serde_json::to_value(restored).unwrap(), baseline_value);
        for dispatch in ["serial", "parallel"] {
            for encoding in ["u32", "uniform_residues"] {
                let mut composition = composition();
                composition["components"]["boundary"]["params"] = json!({
                    "head_dispatch": dispatch, "head_output_encoding": encoding});
                let schedule =
                    crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition))
                        .unwrap();
                let contract = schedule.private_boundary.as_ref().unwrap();
                assert_eq!(contract.head_dispatch, dispatch);
                assert_eq!(contract.head_output_encoding, encoding);
                let mut numeric = serde_json::to_value(contract).unwrap();
                numeric.as_object_mut().unwrap().remove("head_dispatch");
                numeric
                    .as_object_mut()
                    .unwrap()
                    .remove("head_output_encoding");
                numeric["composition_digest"] = baseline_value["composition_digest"].clone();
                assert_eq!(numeric, baseline_value);
                assert_eq!(schedule.prefill, baseline.prefill);
                assert_eq!(schedule.decode, baseline.decode);
                assert_ne!(
                    contract.composition_digest,
                    baseline
                        .private_boundary
                        .as_ref()
                        .unwrap()
                        .composition_digest
                );
                assert_ne!(schedule.digest(), baseline.digest());
            }
        }
    }

    #[test]
    fn private_boundary_transport_choices_fail_closed() {
        let plan = plan("qwen2", true);
        for params in [
            json!({"head_dispatch":"striped"}),
            json!({"head_dispatch":true}),
            json!({"head_dispatch":null}),
            json!({"head_output_encoding":"row_residues"}),
            json!({"head_output_encoding":32}),
            json!({"head_kernel":"sdot"}),
            json!({"head_kernel":true}),
            json!({"head_kernel":null}),
            json!({"unknown":"parallel"}),
        ] {
            let mut composition = composition();
            composition["components"]["boundary"]["params"] = params;
            assert!(
                crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition))
                    .is_err()
            );
            assert!(lower(&plan, &canonical_bytes(&composition)).is_err());
        }
    }

    #[test]
    fn byte_dot_binds_model_neutral_schedule_without_changing_steps() {
        for family in ["qwen2", "qwen3", "llama"] {
            let plan = plan(family, true);
            let baseline =
                crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition()))
                    .unwrap();
            for dispatch in ["serial", "parallel"] {
                for encoding in ["u32", "uniform_residues"] {
                    let mut selected = composition();
                    selected["components"]["boundary"]["params"] = json!({
                        "head_dispatch":dispatch, "head_output_encoding":encoding, "head_kernel":"byte_dot"});
                    let schedule =
                        crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&selected))
                            .unwrap();
                    assert_eq!(
                        schedule.private_boundary.as_ref().unwrap().head_kernel,
                        "byte_dot"
                    );
                    assert_eq!(schedule.prefill, baseline.prefill);
                    assert_eq!(schedule.decode, baseline.decode);
                    assert_ne!(schedule.digest(), baseline.digest());
                    assert_ne!(
                        schedule
                            .private_boundary
                            .as_ref()
                            .unwrap()
                            .composition_digest,
                        baseline
                            .private_boundary
                            .as_ref()
                            .unwrap()
                            .composition_digest
                    );
                }
            }
        }
    }

    #[test]
    fn private_boundary_preserves_body_steps_and_binds_full_artifact() {
        for family in ["qwen2", "qwen3", "llama"] {
            let plan = plan(family, true);
            let composition = composition();
            let schedule =
                crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition))
                    .unwrap();
            let boundary = schedule.private_boundary.as_ref().unwrap();
            assert_eq!((boundary.vocabulary, boundary.hidden_width), (32, 8));
            assert_eq!(boundary.owners, ["boundary_a", "boundary_b"]);
            assert_eq!(boundary.matrix_bytes, 256);
            assert_eq!(boundary.scale_bytes, 128);
            assert_eq!(boundary.lookup_scratch_bytes, 2048);
            assert_eq!(boundary.lookup_query_bound, 4);
            assert_eq!(boundary.client_weight_bytes, 0);
            assert_eq!(boundary.head_signed_output_bound, 8 * 127 * 127);
            let mut ordinary = composition.clone();
            ordinary["components"]
                .as_object_mut()
                .unwrap()
                .remove("boundary");
            ordinary["components"]
                .as_object_mut()
                .unwrap()
                .remove("topology");
            let baseline =
                crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&ordinary)).unwrap();
            assert_eq!(schedule.prefill, baseline.prefill);
            assert_eq!(schedule.decode, baseline.decode);
            assert!(serde_json::to_value(&baseline)
                .unwrap()
                .get("private_boundary")
                .is_none());
            assert_ne!(schedule.digest(), baseline.digest());
        }
    }

    #[test]
    fn private_boundary_rejects_missing_roles_unsupported_options_and_untied_model() {
        let plan = plan("qwen2", true);
        for slot in [
            "verification",
            "nonlinear",
            "scheduler",
            "state",
            "cache",
            "public_prefix",
            "placement",
        ] {
            let mut composition = composition();
            composition["components"][slot] = json!({"component":"pllm/unsupported","params":{}});
            assert!(
                crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition))
                    .is_err()
            );
        }
        let mut no_roles = composition();
        no_roles["components"]
            .as_object_mut()
            .unwrap()
            .remove("topology");
        assert!(crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&no_roles)).is_err());
        assert!(crate::lower_decoder_runtime_schedule(
            &self::plan("qwen2", false),
            &canonical_bytes(&composition())
        )
        .is_err());
        for reduction in [None, Some("prefix_f32")] {
            let mut composition = composition();
            composition["components"]["quantization"] = json!({
                "component":"pllm/symmetric-per-row-quantization/v1",
                "params":{"weight_bits":8,"activation_bits":8}});
            if let Some(reduction) = reduction {
                composition["components"]["quantization"]["params"]["causal_reduction"] =
                    json!(reduction);
            }
            assert!(
                crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition))
                    .is_ok()
            );
            composition["components"]["quantization"]["params"]["weight_bits"] = json!(4);
            assert!(
                crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition))
                    .is_err()
            );
        }
    }

    #[test]
    fn private_boundary_ownership_floors_and_unpriced_placement_rejection() {
        let plan = plan("qwen2", true);
        let composition = canonical_bytes(&composition());
        let admission =
            crate::NetworkPlacementAdmission::new(&canonical_bytes(&plan), &composition).unwrap();
        let requirements: Value = serde_json::from_slice(&admission.requirements()).unwrap();
        assert_eq!(requirements["roles"].as_array().unwrap().len(), 5);
        assert_eq!(requirements["minimum_weight_bytes"]["boundary_a"], 256);
        assert_eq!(requirements["minimum_weight_bytes"]["boundary_b"], 256);
        assert_eq!(requirements["minimum_weight_bytes"]["client"], 40);
        assert_eq!(
            requirements["minimum_memory_bytes"]["boundary_a"],
            256 + 128 + 2048
        );
        assert_eq!(
            requirements["separate_operators"].as_array().unwrap().len(),
            2
        );
        assert!(admission
            .validate(b"{}")
            .unwrap_err()
            .contains("not supported"));
    }

    #[test]
    fn private_boundary_preserves_demand_and_body_encoding_admission() {
        let plan = plan("qwen2", true);
        let mut composition = composition();
        composition["components"]["linear"]["params"] = json!({
            "request_encoding":"stage_packed", "output_encoding":"row_residues"});
        composition["components"]["inventory"] = json!({
            "component":"pllm/prepared-inventory-policy/v1", "params":{
                "policy":"request-sized", "rows":1,"refill":"on-demand","allocation":"demand"}});
        composition["components"]["delivery"] = json!({
            "component":"pllm/client-bundle-transport/v1", "params":{
                "encoding":"artifacts","compression":"zlib","batch_objects":8}});
        let bytes = canonical_bytes(&composition);
        let schedule = crate::lower_decoder_runtime_schedule(&plan, &bytes).unwrap();
        assert!(schedule.private_boundary.is_some());
        assert!(crate::network_placement_requirements(&canonical_bytes(&plan), &bytes).is_ok());
        composition["components"]["inventory"]["params"]["decode_epoch_tokens"] = json!(1);
        assert!(
            crate::lower_decoder_runtime_schedule(&plan, &canonical_bytes(&composition)).is_err()
        );
    }

    #[test]
    fn private_boundary_rejects_auxiliary_feedback_and_unsupported_native_regions() {
        let plan = plan("qwen2", true);
        let composition = canonical_bytes(&composition());
        let mut extra_lookup = plan.clone();
        extra_lookup
            .prefill
            .operations
            .push(plan.prefill.operations[0].clone());
        assert!(lower(&extra_lookup, &composition).is_err());
        let mut auxiliary = plan.clone();
        auxiliary.prefill.operations[0].attributes["auxiliary_weight"] = json!("extra");
        assert!(lower(&auxiliary, &composition).is_err());
        let mut no_feedback = plan.clone();
        no_feedback.token_feedback = false;
        assert!(lower(&no_feedback, &composition).is_err());
        assert!(crate::lower_decoder_continuation(&plan, &composition).is_err());
        assert!(
            crate::attention_tiling_bytes(&canonical_bytes(&plan), &composition, 1, 1).is_err()
        );
        assert!(crate::mlp_tiling_bytes(&canonical_bytes(&plan), &composition, 1).is_err());
        assert!(crate::ExecutionCapsule::compile(
            &canonical_bytes(&plan),
            &composition,
            &"a".repeat(64),
            &"b".repeat(64)
        )
        .is_err());
    }
}
