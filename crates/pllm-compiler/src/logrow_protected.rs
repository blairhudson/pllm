//! Plan-bound, single-SiLU-element research reference for LogRow's fitted Q7
//! table. This is not a Python Experiment slot or a protected decoder runtime.

use crate::{canonical_digest, digest_bytes, lower_model_silu_operation, tensor_elements, Digest};
use pllm_core::{compact::COMPACT_SILU_Q7_PROFILE, CompactQ7Profile};
use pllm_garble::logrow::{
    prepare_compact_q7_logrow, CompactQ7LogRowClient, CompactQ7LogRowDecoder,
    CompactQ7LogRowProgram, LogRowInputs, LogRowOutputs,
};
use pllm_models::{DecoderMode, DecoderPlan};
use serde::Serialize;

const CONTEXT_DOMAIN: &str = "pllm.protected.logrow_q7.element_context.v1";
const MATERIAL_DOMAIN: &str = "pllm.protected.logrow_q7.element_material.v1";
const HARD_MAX_MATERIAL_BYTES: usize = 64 * 1024;

/// Explicit acknowledgement of unreviewed, in-process material and a maximum
/// evaluator body size; zero and unbounded limits fail before material issuance.
pub struct ExperimentalLogRowQ7Policy {
    max_evaluator_material_bytes: usize,
}

impl ExperimentalLogRowQ7Policy {
    pub fn acknowledge_unreviewed_public_profile(
        max_evaluator_material_bytes: usize,
    ) -> Result<Self, String> {
        if max_evaluator_material_bytes == 0
            || max_evaluator_material_bytes > HARD_MAX_MATERIAL_BYTES
        {
            return Err("protected LogRow Q7 material limit must be in (0, 64 KiB]".into());
        }
        Ok(Self {
            max_evaluator_material_bytes,
        })
    }
}

#[derive(Serialize)]
struct ElementContext<'a> {
    plan_digest: Digest,
    mode: DecoderMode,
    operation_id: &'a str,
    element_index: usize,
    profile_id: &'static str,
    profile_digest: [u8; 32],
    artifact_digest: Digest,
    max_evaluator_material_bytes: usize,
}

#[derive(Serialize)]
struct MaterialContext<'a> {
    element_context: &'a Digest,
    evaluator_material_bytes: usize,
    issuance_id: [u8; 32],
}

pub struct BoundLogRowQ7ClientMaterial {
    context_digest: Digest,
    binding_digest: Digest,
    client: CompactQ7LogRowClient,
    program: CompactQ7LogRowProgram,
}

pub struct BoundLogRowQ7Evaluation {
    context_digest: Digest,
    binding_digest: Digest,
    inputs: LogRowInputs,
    program: CompactQ7LogRowProgram,
}

pub struct BoundLogRowQ7Decoder(CompactQ7LogRowDecoder);
pub struct BoundLogRowQ7Outputs(LogRowOutputs);

fn method_artifact_digest() -> Digest {
    canonical_digest(
        "pllm.artifact.logrow_q7.complete_source_set.v1",
        &[
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("logrow_protected.rs"),
            ),
            digest_bytes("pllm.artifact.rust-source.v1", include_bytes!("lib.rs")),
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-models/src/lib.rs"),
            ),
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-core/src/compact.rs"),
            ),
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-garble/src/boolean.rs"),
            ),
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-garble/src/logrow.rs"),
            ),
        ],
    )
}

fn validate_context(
    plan: &DecoderPlan,
    mode: DecoderMode,
    operation_id: &str,
    element_index: usize,
    profile: &CompactQ7Profile,
    policy: &ExperimentalLogRowQ7Policy,
) -> Result<Digest, String> {
    let (tensor, _) = lower_model_silu_operation(plan, mode, operation_id)?;
    if element_index >= tensor_elements(&tensor.shape)? {
        return Err("protected LogRow Q7 element is outside the semantic SiLU tensor".into());
    }
    Ok(canonical_digest(
        CONTEXT_DOMAIN,
        &ElementContext {
            plan_digest: plan.digest(),
            mode,
            operation_id,
            element_index,
            profile_id: COMPACT_SILU_Q7_PROFILE,
            profile_digest: profile.digest(),
            artifact_digest: method_artifact_digest(),
            max_evaluator_material_bytes: policy.max_evaluator_material_bytes,
        },
    ))
}

/// Bind exactly one semantic SiLU element to the fitted-table reference.
/// Rejected material is dropped before any handle is published to the caller.
pub fn prepare_bound_logrow_q7_element(
    plan: &DecoderPlan,
    mode: DecoderMode,
    operation_id: &str,
    element_index: usize,
    profile: &CompactQ7Profile,
    policy: &ExperimentalLogRowQ7Policy,
) -> Result<BoundLogRowQ7ClientMaterial, String> {
    let context_digest =
        validate_context(plan, mode, operation_id, element_index, profile, policy)?;
    let (client, program) = prepare_compact_q7_logrow(profile)?;
    let evaluator_material_bytes = program.evaluator_material_bytes();
    if evaluator_material_bytes > policy.max_evaluator_material_bytes {
        return Err(format!(
            "protected LogRow Q7 element requires {evaluator_material_bytes} material bytes, exceeding the admitted limit {}",
            policy.max_evaluator_material_bytes,
        ));
    }
    let binding_digest = canonical_digest(
        MATERIAL_DOMAIN,
        &MaterialContext {
            element_context: &context_digest,
            evaluator_material_bytes,
            issuance_id: program.issuance_id(),
        },
    );
    Ok(BoundLogRowQ7ClientMaterial {
        context_digest,
        binding_digest,
        client,
        program,
    })
}

impl BoundLogRowQ7ClientMaterial {
    pub fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }

    pub fn evaluator_material_bytes(&self) -> usize {
        self.program.evaluator_material_bytes()
    }

    /// Invalid Q7 input also drops the one-use circuit and client material.
    pub fn encode(
        self,
        value: i16,
    ) -> Result<(BoundLogRowQ7Evaluation, BoundLogRowQ7Decoder), String> {
        let (inputs, decoder) = self.client.encode(value)?;
        Ok((
            BoundLogRowQ7Evaluation {
                context_digest: self.context_digest,
                binding_digest: self.binding_digest,
                inputs,
                program: self.program,
            },
            BoundLogRowQ7Decoder(decoder),
        ))
    }
}

impl BoundLogRowQ7Evaluation {
    pub fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }

    /// Context failure consumes both input labels and circuit before decoding.
    pub fn evaluate(
        self,
        plan: &DecoderPlan,
        mode: DecoderMode,
        operation_id: &str,
        element_index: usize,
        profile: &CompactQ7Profile,
        policy: &ExperimentalLogRowQ7Policy,
    ) -> Result<BoundLogRowQ7Outputs, String> {
        let expected = validate_context(plan, mode, operation_id, element_index, profile, policy)?;
        if expected != self.context_digest {
            return Err("protected LogRow Q7 material differs from its semantic context".into());
        }
        self.program.evaluate(self.inputs).map(BoundLogRowQ7Outputs)
    }
}

impl BoundLogRowQ7Decoder {
    pub fn decode(self, outputs: BoundLogRowQ7Outputs) -> Result<i16, String> {
        self.0.decode(outputs.0)
    }
}
