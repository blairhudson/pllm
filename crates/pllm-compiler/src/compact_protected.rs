//! A plan-bound *single-element research reference* for protected Compact Q7.
//!
//! No generic Experiment component, Python runtime, model execution schedule,
//! or cross-role transport is authorized by this specialized boundary.

use crate::{canonical_digest, digest_bytes, lower_model_silu_operation, tensor_elements, Digest};
use pllm_core::{compact::COMPACT_SILU_Q7_PROFILE, CompactQ7Profile};
use pllm_garble::compact_polynomial::{
    prepare_compact_q7_polynomial, CompactQ7PolynomialClient, CompactQ7PolynomialDecoder,
    CompactQ7PolynomialInputs, CompactQ7PolynomialOutputs, CompactQ7PolynomialProgram,
};
use pllm_models::{DecoderMode, DecoderPlan};
use serde::Serialize;

const CONTEXT_DOMAIN: &str = "pllm.protected.compact_q7.element_context.v1";
const MATERIAL_DOMAIN: &str = "pllm.protected.compact_q7.element_material.v1";
const HARD_MAX_CIPHERTEXT_BYTES: u64 = 8 * 1024 * 1024;

/// Explicit resource bound and acknowledgement for an unreviewed, in-process
/// public-profile reference, not a whole-model privacy or performance claim.
pub struct ExperimentalCompactQ7Policy {
    max_evaluator_ciphertext_bytes: u64,
}

impl ExperimentalCompactQ7Policy {
    pub fn acknowledge_unreviewed_public_profile(
        max_evaluator_ciphertext_bytes: u64,
    ) -> Result<Self, String> {
        if max_evaluator_ciphertext_bytes == 0
            || max_evaluator_ciphertext_bytes > HARD_MAX_CIPHERTEXT_BYTES
        {
            return Err("protected Compact Q7 ciphertext bound must be in (0, 8 MiB]".into());
        }
        Ok(Self {
            max_evaluator_ciphertext_bytes,
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
    max_evaluator_ciphertext_bytes: u64,
}

#[derive(Serialize)]
struct MaterialContext<'a> {
    element_context: &'a Digest,
    and_gate_count: usize,
    evaluator_ciphertext_bytes: u64,
    issuance_id: [u8; 32],
}

pub struct BoundCompactQ7ClientMaterial {
    context_digest: Digest,
    binding_digest: Digest,
    client: CompactQ7PolynomialClient,
    program: CompactQ7PolynomialProgram,
}

pub struct BoundCompactQ7Evaluation {
    context_digest: Digest,
    binding_digest: Digest,
    program: CompactQ7PolynomialProgram,
    inputs: CompactQ7PolynomialInputs,
}

pub struct BoundCompactQ7Decoder(CompactQ7PolynomialDecoder);
pub struct BoundCompactQ7Outputs(CompactQ7PolynomialOutputs);

fn method_artifact_digest() -> Digest {
    canonical_digest(
        "pllm.artifact.compact_q7.complete_source_set.v1",
        &[
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("compact_protected.rs"),
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
                include_bytes!("../../pllm-garble/src/compact_coordinate.rs"),
            ),
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-garble/src/compact_polynomial.rs"),
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
    policy: &ExperimentalCompactQ7Policy,
) -> Result<Digest, String> {
    let (tensor, _) = lower_model_silu_operation(plan, mode, operation_id)?;
    let elements = tensor_elements(&tensor.shape)?;
    if element_index >= elements {
        return Err("protected Compact Q7 element is outside the semantic SiLU tensor".into());
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
            max_evaluator_ciphertext_bytes: policy.max_evaluator_ciphertext_bytes,
        },
    ))
}

/// Admit exactly one semantic SiLU tensor element. The requested policy is
/// checked before publishing material to the caller; failure drops it.
pub fn prepare_bound_compact_q7_element(
    plan: &DecoderPlan,
    mode: DecoderMode,
    operation_id: &str,
    element_index: usize,
    profile: &CompactQ7Profile,
    policy: &ExperimentalCompactQ7Policy,
) -> Result<BoundCompactQ7ClientMaterial, String> {
    let context_digest =
        validate_context(plan, mode, operation_id, element_index, profile, policy)?;
    let (client, program) = prepare_compact_q7_polynomial(profile)?;
    let evaluator_ciphertext_bytes = program.evaluator_ciphertext_bytes()?;
    if evaluator_ciphertext_bytes > policy.max_evaluator_ciphertext_bytes {
        return Err(format!(
            "protected Compact Q7 element requires {evaluator_ciphertext_bytes} ciphertext bytes, exceeding the admitted limit {}",
            policy.max_evaluator_ciphertext_bytes,
        ));
    }
    let binding_digest = canonical_digest(
        MATERIAL_DOMAIN,
        &MaterialContext {
            element_context: &context_digest,
            and_gate_count: program.and_gate_count(),
            evaluator_ciphertext_bytes,
            issuance_id: program.issuance_id(),
        },
    );
    Ok(BoundCompactQ7ClientMaterial {
        context_digest,
        binding_digest,
        client,
        program,
    })
}

impl BoundCompactQ7ClientMaterial {
    pub fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }

    pub fn evaluator_ciphertext_bytes(&self) -> Result<u64, String> {
        self.program.evaluator_ciphertext_bytes()
    }

    /// Input failures also consume and drop the one-use program.
    pub fn encode(
        self,
        value: i16,
    ) -> Result<(BoundCompactQ7Evaluation, BoundCompactQ7Decoder), String> {
        let (inputs, decoder) = self.client.encode(value)?;
        Ok((
            BoundCompactQ7Evaluation {
                context_digest: self.context_digest,
                binding_digest: self.binding_digest,
                program: self.program,
                inputs,
            },
            BoundCompactQ7Decoder(decoder),
        ))
    }
}

impl BoundCompactQ7Evaluation {
    pub fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }

    /// A context mismatch burns this owned program before any evaluation.
    pub fn evaluate(
        self,
        plan: &DecoderPlan,
        mode: DecoderMode,
        operation_id: &str,
        element_index: usize,
        profile: &CompactQ7Profile,
        policy: &ExperimentalCompactQ7Policy,
    ) -> Result<BoundCompactQ7Outputs, String> {
        let expected = validate_context(plan, mode, operation_id, element_index, profile, policy)?;
        if expected != self.context_digest {
            return Err("protected Compact Q7 material differs from its semantic context".into());
        }
        self.program
            .evaluate(self.inputs)
            .map(BoundCompactQ7Outputs)
    }
}

impl BoundCompactQ7Decoder {
    pub fn decode(self, outputs: BoundCompactQ7Outputs) -> Result<(usize, i16), String> {
        self.0.decode(outputs.0)
    }
}
