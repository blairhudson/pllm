//! Experiment-bound admission for the distinct verified ring64 runtime.
//! Ordinary masked/int8 scheduling cannot execute this contract.

use pllm_types::{canonical_bytes, canonical_digest, Digest};
use serde::Serialize;

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
        let envelope = GkmDecoderContract::compile_with_capacity(
            config,
            document.budget.max_input_tokens as usize,
            document.budget.max_new_tokens as usize - 1,
            capacity,
        )?;
        let decoder = if input_rows as u64 == document.budget.max_input_tokens
            && output_tokens as u64 == document.budget.max_new_tokens
        {
            envelope
        } else {
            drop(envelope);
            GkmDecoderContract::compile_with_capacity(
                config,
                input_rows,
                output_tokens - 1,
                capacity,
            )?
        };
        Ok(Self {
            schema_version: SCHEMA,
            configuration: crate::configuration_digest_bytes(experiment),
            composition: crate::pipeline_digest(&document.pipeline),
            source_lock,
            tokenizer,
            decoder,
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
}
