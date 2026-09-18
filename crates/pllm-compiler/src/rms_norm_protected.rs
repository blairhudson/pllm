use crate::{
    canonical_digest, validate_rms_norm_q10_direct_region, Digest, ModelRmsNormQ10DirectRegion,
};
use pllm_models::DecoderPlan;
use serde::Serialize;

const REGION_BINDING_DOMAIN: &str = "pllm.protected.rms_norm_q10.region_binding.v1";
const MATERIAL_BINDING_DOMAIN: &str = "pllm.protected.rms_norm_q10.material_binding.v1";
const MAX_REFERENCE_WIDTH: usize = 64;

pub struct ExperimentalRmsNormQ10Policy {
    max_evaluator_ciphertext_bytes: u64,
}

impl ExperimentalRmsNormQ10Policy {
    /// Acknowledge unreviewed reference cryptography and public-weight topology leakage.
    pub fn acknowledge_unreviewed_public_weights(
        max_evaluator_ciphertext_bytes: u64,
    ) -> Result<Self, String> {
        if max_evaluator_ciphertext_bytes == 0 || max_evaluator_ciphertext_bytes > 4_000_000_000 {
            return Err(
                "protected Q10 RMSNorm evaluator limit must be between 1 and 4000000000 bytes"
                    .into(),
            );
        }
        Ok(Self {
            max_evaluator_ciphertext_bytes,
        })
    }
}

#[derive(Serialize)]
struct RegionBinding<'a> {
    region: &'a ModelRmsNormQ10DirectRegion,
    row_index: usize,
}

#[derive(Serialize)]
struct MaterialBinding<'a> {
    region_binding: &'a Digest,
    weights: &'a [i16],
    and_gate_count: usize,
    evaluator_ciphertext_bytes: u64,
    method_artifact_digest: Digest,
    issuance_id: [u8; 32],
}

pub struct BoundRmsNormQ10ClientMaterial {
    binding_digest: Digest,
    region_binding: Digest,
    width: usize,
    client: pllm_garble::boolean::RmsNormQ10Client,
    program: pllm_garble::boolean::RmsNormQ10Program,
}

pub struct BoundRmsNormQ10Evaluation {
    binding_digest: Digest,
    region_binding: Digest,
    program: pllm_garble::boolean::RmsNormQ10Program,
    inputs: pllm_garble::boolean::RmsNormQ10Inputs,
}

pub struct BoundRmsNormQ10Decoder(pllm_garble::boolean::RmsNormQ10Decoder);

pub struct BoundRmsNormQ10Outputs(pllm_garble::boolean::BooleanCircuitOutputs);

pub fn prepare_bound_rms_norm_q10_row(
    plan: &DecoderPlan,
    region: &ModelRmsNormQ10DirectRegion,
    row_index: usize,
    weights: &[i16],
    policy: &ExperimentalRmsNormQ10Policy,
) -> Result<BoundRmsNormQ10ClientMaterial, String> {
    let (rows, width) = validate_rms_norm_q10_direct_region(plan, region)?;
    if row_index >= rows {
        return Err("protected Q10 RMSNorm row index is outside its semantic tensor".into());
    }
    if weights.len() != width {
        return Err(format!(
            "protected Q10 RMSNorm requires {width} weights, received {}",
            weights.len()
        ));
    }
    if width > MAX_REFERENCE_WIDTH {
        return Err(format!(
            "unreviewed in-memory protected Q10 RMSNorm supports at most {MAX_REFERENCE_WIDTH} values; streaming execution is required for width {width}"
        ));
    }
    let estimate = pllm_garble::boolean::estimate_rms_norm_q10_direct(weights)?;
    if estimate.evaluator_ciphertext_bytes > policy.max_evaluator_ciphertext_bytes {
        return Err(format!(
            "protected Q10 RMSNorm requires {} evaluator bytes, exceeding policy limit {}",
            estimate.evaluator_ciphertext_bytes, policy.max_evaluator_ciphertext_bytes
        ));
    }
    let region_binding =
        canonical_digest(REGION_BINDING_DOMAIN, &RegionBinding { region, row_index });
    let (client, program) = pllm_garble::boolean::prepare_rms_norm_q10_direct(weights)?;
    if u64::try_from(program.and_gate_count()) != Ok(estimate.and_gate_count)
        || program.evaluator_ciphertext_bytes()? != estimate.evaluator_ciphertext_bytes
    {
        return Err("protected Q10 RMSNorm circuit estimate does not match construction".into());
    }
    let binding_digest = canonical_digest(
        MATERIAL_BINDING_DOMAIN,
        &MaterialBinding {
            region_binding: &region_binding,
            weights,
            and_gate_count: program.and_gate_count(),
            evaluator_ciphertext_bytes: program.evaluator_ciphertext_bytes()?,
            method_artifact_digest: crate::protected_rms_norm_q10_artifact_digest(),
            issuance_id: program.issuance_id(),
        },
    );
    Ok(BoundRmsNormQ10ClientMaterial {
        binding_digest,
        region_binding,
        width,
        client,
        program,
    })
}

impl BoundRmsNormQ10ClientMaterial {
    pub fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }

    pub fn and_gate_count(&self) -> usize {
        self.program.and_gate_count()
    }

    pub fn evaluator_ciphertext_bytes(&self) -> Result<u64, String> {
        self.program.evaluator_ciphertext_bytes()
    }

    pub fn encode(
        self,
        input: &[i16],
    ) -> Result<(BoundRmsNormQ10Evaluation, BoundRmsNormQ10Decoder), String> {
        if input.len() != self.width {
            return Err(format!(
                "protected Q10 RMSNorm row requires {} inputs, received {}",
                self.width,
                input.len()
            ));
        }
        let (inputs, decoder) = self.client.encode(input)?;
        Ok((
            BoundRmsNormQ10Evaluation {
                binding_digest: self.binding_digest,
                region_binding: self.region_binding,
                program: self.program,
                inputs,
            },
            BoundRmsNormQ10Decoder(decoder),
        ))
    }
}

impl BoundRmsNormQ10Evaluation {
    pub fn binding_digest(&self) -> &Digest {
        &self.binding_digest
    }

    pub fn evaluate(
        self,
        plan: &DecoderPlan,
        region: &ModelRmsNormQ10DirectRegion,
        row_index: usize,
    ) -> Result<BoundRmsNormQ10Outputs, String> {
        validate_rms_norm_q10_direct_region(plan, region)?;
        let expected =
            canonical_digest(REGION_BINDING_DOMAIN, &RegionBinding { region, row_index });
        if expected != self.region_binding {
            return Err("protected Q10 RMSNorm material differs from its region binding".into());
        }
        self.program
            .evaluate(self.inputs)
            .map(BoundRmsNormQ10Outputs)
    }
}

impl BoundRmsNormQ10Decoder {
    pub fn decode(self, outputs: BoundRmsNormQ10Outputs) -> Result<Vec<i32>, String> {
        self.0.decode(outputs.0)
    }
}
