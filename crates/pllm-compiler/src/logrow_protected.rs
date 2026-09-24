//! Plan-bound SiLU research references for LogRow's fitted Q7 table.
//! These are not Python Experiment slots or protected decoder runtimes.

use crate::{canonical_digest, digest_bytes, lower_model_silu_operation, tensor_elements, Digest};
use pllm_core::{compact::COMPACT_SILU_Q7_PROFILE, CompactQ7Profile};
use pllm_garble::logrow::{
    prepare_compact_q7_logrow, prepare_compact_q7_logrow_elements, CompactQ7LogRowClient,
    CompactQ7LogRowDecoder, CompactQ7LogRowProgram, LogRowInputs, LogRowOutputs,
    COMPACT_Q7_LOGROW_MATERIAL_BYTES,
};
use pllm_models::{DecoderMode, DecoderPlan, ModelOperator};
use serde::Serialize;

const CONTEXT_DOMAIN: &str = "pllm.protected.logrow_q7.element_context.v1";
const MATERIAL_DOMAIN: &str = "pllm.protected.logrow_q7.element_material.v1";
const TENSOR_CONTEXT_DOMAIN: &str = "pllm.protected.logrow_q7.tensor_context.v1";
const TENSOR_MATERIAL_DOMAIN: &str = "pllm.protected.logrow_q7.tensor_material.v1";
const SESSION_ESTIMATE_DOMAIN: &str = "pllm.protected.logrow_q7.session_estimate.v1";
const HARD_MAX_MATERIAL_BYTES: usize = 64 * 1024;
const HARD_MAX_TENSOR_MATERIAL_BYTES: usize = 64 * 1024 * 1024;
const HARD_MAX_SESSION_MATERIAL_BYTES: usize = 512 * 1024 * 1024;

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

/// Per-tensor resource admission for a bounded, in-process research reference.
/// Neither a production-security claim nor an inference component selection.
pub struct ExperimentalLogRowQ7TensorPolicy {
    max_elements: usize,
    max_evaluator_material_bytes: usize,
}

impl ExperimentalLogRowQ7TensorPolicy {
    pub fn acknowledge_unreviewed_public_profile(
        max_elements: usize,
        max_evaluator_material_bytes: usize,
    ) -> Result<Self, String> {
        if max_elements == 0
            || max_evaluator_material_bytes == 0
            || max_evaluator_material_bytes > HARD_MAX_TENSOR_MATERIAL_BYTES
        {
            return Err(
                "protected LogRow Q7 tensor bounds must be positive and at most 64 MiB".into(),
            );
        }
        Ok(Self {
            max_elements,
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

#[derive(Serialize)]
struct TensorContext<'a> {
    plan_digest: Digest,
    mode: DecoderMode,
    operation_id: &'a str,
    shape: &'a [u64],
    profile_id: &'static str,
    profile_digest: [u8; 32],
    artifact_digest: Digest,
    max_elements: usize,
    max_evaluator_material_bytes: usize,
}

#[derive(Serialize)]
struct TensorMaterialContext<'a> {
    tensor_context: &'a Digest,
    issuance_ids: &'a [[u8; 32]],
    evaluator_material_bytes: usize,
}

#[derive(Serialize)]
struct SessionEstimateContext<'a> {
    plan_digest: &'a Digest,
    profile_digest: [u8; 32],
    prefill: &'a [Digest],
    decode: &'a [Digest],
    max_decode_steps: u64,
    max_session_evaluator_material_bytes: usize,
}

pub struct BoundLogRowQ7TensorMaterial {
    context_digest: Digest,
    binding_digest: Digest,
    elements: Vec<(CompactQ7LogRowClient, CompactQ7LogRowProgram)>,
}

pub struct BoundLogRowQ7TensorEvaluation {
    context_digest: Digest,
    binding_digest: Digest,
    elements: Vec<(LogRowInputs, CompactQ7LogRowProgram)>,
}

pub struct BoundLogRowQ7TensorDecoder(Vec<CompactQ7LogRowDecoder>);
pub struct BoundLogRowQ7TensorOutputs(Vec<LogRowOutputs>);

/// Immutable upper bound for all semantic SiLU material in one response.
/// Estimation issues no material and does not authorize a composition.
#[derive(Clone, Debug, Serialize)]
pub struct BoundLogRowQ7SessionEstimate {
    pub schema_version: &'static str,
    pub plan_digest: Digest,
    pub profile_digest: [u8; 32],
    pub max_decode_steps: u64,
    pub prefill_elements: usize,
    pub decode_elements_per_step: usize,
    pub reserved_evaluator_material_bytes: usize,
    pub largest_tensor_material_bytes: usize,
    pub estimate_digest: Digest,
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

fn validate_tensor_context(
    plan: &DecoderPlan,
    mode: DecoderMode,
    operation_id: &str,
    profile: &CompactQ7Profile,
    policy: &ExperimentalLogRowQ7TensorPolicy,
) -> Result<(Digest, usize), String> {
    let (tensor, _) = lower_model_silu_operation(plan, mode, operation_id)?;
    let elements = tensor_elements(&tensor.shape)?;
    if elements > policy.max_elements {
        return Err("protected LogRow Q7 tensor exceeds its element limit".into());
    }
    let bytes = elements
        .checked_mul(COMPACT_Q7_LOGROW_MATERIAL_BYTES)
        .ok_or("protected LogRow Q7 tensor material size overflowed")?;
    if bytes > policy.max_evaluator_material_bytes {
        return Err("protected LogRow Q7 tensor exceeds its material limit".into());
    }
    Ok((
        canonical_digest(
            TENSOR_CONTEXT_DOMAIN,
            &TensorContext {
                plan_digest: plan.digest(),
                mode,
                operation_id,
                shape: &tensor.shape,
                profile_id: COMPACT_SILU_Q7_PROFILE,
                profile_digest: profile.digest(),
                artifact_digest: method_artifact_digest(),
                max_elements: policy.max_elements,
                max_evaluator_material_bytes: policy.max_evaluator_material_bytes,
            },
        ),
        elements,
    ))
}

/// Preflight all semantic SiLU material in a bounded response. This only
/// estimates LogRow bodies; no tensor is issued or live session authorized.
pub fn estimate_bound_logrow_q7_session(
    plan: &DecoderPlan,
    profile: &CompactQ7Profile,
    tensor_policy: &ExperimentalLogRowQ7TensorPolicy,
    max_decode_steps: u64,
    max_session_evaluator_material_bytes: usize,
) -> Result<BoundLogRowQ7SessionEstimate, String> {
    plan.validate().map_err(|error| error.to_string())?;
    if max_decode_steps == 0
        || max_session_evaluator_material_bytes == 0
        || max_session_evaluator_material_bytes > HARD_MAX_SESSION_MATERIAL_BYTES
    {
        return Err("protected LogRow Q7 session limit must be in (0, 512 MiB]".into());
    }
    let final_position = plan
        .prefill
        .query_sequence
        .checked_add(max_decode_steps - 1)
        .ok_or("protected LogRow Q7 session position overflowed")?;
    if final_position > plan.decode.maximum_key_sequence {
        return Err("protected LogRow Q7 decode steps exceed the semantic state bound".into());
    }
    let mut totals = [0usize; 2];
    let mut contexts = [Vec::new(), Vec::new()];
    let mut largest = 0usize;
    for (index, (mode, graph)) in [
        (DecoderMode::Prefill, &plan.prefill),
        (DecoderMode::Decode, &plan.decode),
    ]
    .into_iter()
    .enumerate()
    {
        for operation in graph
            .operations
            .iter()
            .filter(|operation| operation.operator == ModelOperator::Silu)
        {
            let (digest, elements) =
                validate_tensor_context(plan, mode, &operation.id, profile, tensor_policy)?;
            let bytes = elements
                .checked_mul(COMPACT_Q7_LOGROW_MATERIAL_BYTES)
                .ok_or("protected LogRow Q7 tensor material size overflowed")?;
            totals[index] = totals[index]
                .checked_add(elements)
                .ok_or("protected LogRow Q7 phase element count overflowed")?;
            largest = largest.max(bytes);
            contexts[index].push(digest);
        }
    }
    if contexts.iter().any(Vec::is_empty) {
        return Err("protected LogRow Q7 session needs SiLU in both phases".into());
    }
    let decode_total = totals[1]
        .checked_mul(usize::try_from(max_decode_steps).map_err(|_| "decode steps overflowed")?)
        .ok_or("protected LogRow Q7 session element count overflowed")?;
    let session_elements = totals[0]
        .checked_add(decode_total)
        .ok_or("protected LogRow Q7 session element count overflowed")?;
    let reserved_evaluator_material_bytes = session_elements
        .checked_mul(COMPACT_Q7_LOGROW_MATERIAL_BYTES)
        .ok_or("protected LogRow Q7 session material size overflowed")?;
    if reserved_evaluator_material_bytes > max_session_evaluator_material_bytes {
        return Err("protected LogRow Q7 session exceeds its material limit".into());
    }
    let plan_digest = plan.digest();
    let estimate_digest = canonical_digest(
        SESSION_ESTIMATE_DOMAIN,
        &SessionEstimateContext {
            plan_digest: &plan_digest,
            profile_digest: profile.digest(),
            prefill: &contexts[0],
            decode: &contexts[1],
            max_decode_steps,
            max_session_evaluator_material_bytes,
        },
    );
    Ok(BoundLogRowQ7SessionEstimate {
        schema_version: "pllm.logrow_q7_session_estimate.v1",
        plan_digest,
        profile_digest: profile.digest(),
        max_decode_steps,
        prefill_elements: totals[0],
        decode_elements_per_step: totals[1],
        reserved_evaluator_material_bytes,
        largest_tensor_material_bytes: largest,
        estimate_digest,
    })
}

/// Admit the complete semantic tensor before issuing any one-use material.
pub fn prepare_bound_logrow_q7_tensor(
    plan: &DecoderPlan,
    mode: DecoderMode,
    operation_id: &str,
    profile: &CompactQ7Profile,
    policy: &ExperimentalLogRowQ7TensorPolicy,
) -> Result<BoundLogRowQ7TensorMaterial, String> {
    let (context_digest, count) =
        validate_tensor_context(plan, mode, operation_id, profile, policy)?;
    let elements = prepare_compact_q7_logrow_elements(profile, count)?;
    let issuance_ids = elements
        .iter()
        .map(|(_, program)| program.issuance_id())
        .collect::<Vec<_>>();
    let binding_digest = canonical_digest(
        TENSOR_MATERIAL_DOMAIN,
        &TensorMaterialContext {
            tensor_context: &context_digest,
            issuance_ids: &issuance_ids,
            evaluator_material_bytes: count * COMPACT_Q7_LOGROW_MATERIAL_BYTES,
        },
    );
    Ok(BoundLogRowQ7TensorMaterial {
        context_digest,
        binding_digest,
        elements,
    })
}

impl BoundLogRowQ7TensorMaterial {
    pub fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }

    pub fn evaluator_material_bytes(&self) -> usize {
        self.elements.len() * COMPACT_Q7_LOGROW_MATERIAL_BYTES
    }

    pub fn elements(&self) -> usize {
        self.elements.len()
    }

    /// Invalid length or any invalid element burns the *entire* tensor.
    pub fn encode(
        self,
        values: &[i16],
    ) -> Result<(BoundLogRowQ7TensorEvaluation, BoundLogRowQ7TensorDecoder), String> {
        if values.len() != self.elements.len()
            || values.iter().any(|value| !(-128..=128).contains(value))
        {
            return Err("protected LogRow Q7 tensor input shape or domain differs".into());
        }
        let mut encoded = Vec::with_capacity(self.elements.len());
        let mut decoders = Vec::with_capacity(self.elements.len());
        for ((client, program), &value) in self.elements.into_iter().zip(values) {
            let (input, decoder) = client.encode(value)?;
            encoded.push((input, program));
            decoders.push(decoder);
        }
        Ok((
            BoundLogRowQ7TensorEvaluation {
                context_digest: self.context_digest,
                binding_digest: self.binding_digest,
                elements: encoded,
            },
            BoundLogRowQ7TensorDecoder(decoders),
        ))
    }
}

impl BoundLogRowQ7TensorEvaluation {
    pub fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }

    /// Revalidate the complete context before interpreting any row; failure
    /// drops all one-use material, including rows not yet evaluated.
    pub fn evaluate(
        self,
        plan: &DecoderPlan,
        mode: DecoderMode,
        operation_id: &str,
        profile: &CompactQ7Profile,
        policy: &ExperimentalLogRowQ7TensorPolicy,
    ) -> Result<BoundLogRowQ7TensorOutputs, String> {
        let (expected, count) = validate_tensor_context(plan, mode, operation_id, profile, policy)?;
        if expected != self.context_digest || count != self.elements.len() {
            return Err("protected LogRow Q7 tensor differs from its semantic context".into());
        }
        self.elements
            .into_iter()
            .map(|(input, program)| program.evaluate(input))
            .collect::<Result<Vec<_>, _>>()
            .map(BoundLogRowQ7TensorOutputs)
    }
}

impl BoundLogRowQ7TensorDecoder {
    pub fn decode(self, outputs: BoundLogRowQ7TensorOutputs) -> Result<Vec<i16>, String> {
        if self.0.len() != outputs.0.len() {
            return Err("protected LogRow Q7 tensor outputs differ from its decoder".into());
        }
        self.0
            .into_iter()
            .zip(outputs.0)
            .map(|(decoder, output)| decoder.decode(output))
            .collect()
    }
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
