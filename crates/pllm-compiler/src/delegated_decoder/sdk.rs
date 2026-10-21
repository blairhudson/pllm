//! Experiment-bound admission for the distinct verified ring64 runtime.
//! Ordinary masked/int8 scheduling cannot execute this contract.

use pllm_types::{canonical_bytes, canonical_digest, Digest};
use serde::Serialize;

use super::batch::SourceStageBatchContract;
use super::{reject, GkmDecoderContract};
use crate::{ExperimentDeployment, ExperimentDocument, ExperimentPipeline};

pub const PROTOCOL: &str = "pllm/open-weight-verifiable-outsourcing/v1";
pub const VERIFIER: &str = "pllm/published-linear-verify/v1";
pub const TOPOLOGY: &str = "pllm/two-online-delegation-workers/v1";
const SCHEMA: &str = "pllm.gkm.sdk-contract.v1";
const MAX_EXPERIMENT_BYTES: usize = 32 << 10;

pub(crate) fn selected(pipeline: &ExperimentPipeline) -> bool {
    pipeline
        .components
        .values()
        .any(|item| matches!(item.component.as_str(), PROTOCOL | VERIFIER | TOPOLOGY))
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
            decoder: GkmDecoderContract::compile_owned(&config, 2, 1, capacity, streamed).unwrap(),
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
    if pipeline.components.len() != expected.len()
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
        let envelope = GkmDecoderContract::compile_streamed_with_capacity(
            config,
            document.budget.max_input_tokens as usize,
            document.budget.max_new_tokens as usize - 1,
            capacity,
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
            GkmDecoderContract::compile_streamed_with_capacity(
                config,
                input_rows,
                output_tokens - 1,
                capacity,
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
