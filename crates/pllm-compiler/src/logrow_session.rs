//! Offline all-or-nothing LogRow material issuance for bounded semantic SiLU.
//! The evaluator and trusted decoder are collocated: no distributed 2PC claim.

use std::collections::VecDeque;

use pllm_core::{
    compact::{compact_q7_from_f32, compact_q7_to_f32},
    CompactQ7Profile,
};
use pllm_models::{DecoderMode, DecoderPlan, ModelOperator};

use crate::{
    canonical_digest, estimate_bound_logrow_q7_session, prepare_bound_logrow_q7_tensor,
    BoundLogRowQ7SessionEstimate, BoundLogRowQ7TensorMaterial, Digest,
    ExperimentalLogRowQ7TensorPolicy,
};
use serde::Serialize;

const ISSUANCE_DOMAIN: &str = "pllm.protected.logrow_q7.session_issuance.v1";

#[derive(Serialize)]
struct IssuanceContext<'a> {
    estimate_digest: &'a Digest,
    ordered_tensors: &'a [Digest],
}

struct PendingTensor {
    mode: DecoderMode,
    decode_step: u64,
    operation_id: String,
    material: BoundLogRowQ7TensorMaterial,
}

/// Owns every pre-issued one-use tensor in semantic order. Dropping the session
/// burns all remaining material; errors burn the whole remainder immediately.
pub struct BoundLogRowQ7Session {
    plan: DecoderPlan,
    profile: CompactQ7Profile,
    tensor_policy: ExperimentalLogRowQ7TensorPolicy,
    estimate: BoundLogRowQ7SessionEstimate,
    issuance_digest: Digest,
    pending: VecDeque<PendingTensor>,
    aborted: bool,
}

pub fn prepare_bound_logrow_q7_session(
    plan: &DecoderPlan,
    profile: &CompactQ7Profile,
    tensor_policy: &ExperimentalLogRowQ7TensorPolicy,
    max_decode_steps: u64,
    max_session_evaluator_material_bytes: usize,
) -> Result<BoundLogRowQ7Session, String> {
    let estimate = estimate_bound_logrow_q7_session(
        plan,
        profile,
        tensor_policy,
        max_decode_steps,
        max_session_evaluator_material_bytes,
    )?;
    let operation_count = plan
        .prefill
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::Silu)
        .count()
        .checked_add(
            plan.decode
                .operations
                .iter()
                .filter(|operation| operation.operator == ModelOperator::Silu)
                .count()
                .checked_mul(
                    usize::try_from(max_decode_steps).map_err(|_| "decode steps overflowed")?,
                )
                .ok_or("LogRow Q7 operation count overflowed")?,
        )
        .ok_or("LogRow Q7 operation count overflowed")?;
    let mut pending = VecDeque::new();
    pending
        .try_reserve(operation_count)
        .map_err(|_| "LogRow Q7 session allocation failed")?;
    let mut digests = Vec::new();
    digests
        .try_reserve_exact(operation_count)
        .map_err(|_| "LogRow Q7 session digest allocation failed")?;
    for (mode, graph, steps) in [
        (DecoderMode::Prefill, &plan.prefill, 1),
        (DecoderMode::Decode, &plan.decode, max_decode_steps),
    ] {
        for step in 0..steps {
            for operation in graph
                .operations
                .iter()
                .filter(|operation| operation.operator == ModelOperator::Silu)
            {
                let material = prepare_bound_logrow_q7_tensor(
                    plan,
                    mode,
                    &operation.id,
                    profile,
                    tensor_policy,
                )?;
                digests.push(material.binding_digest().clone());
                pending.push_back(PendingTensor {
                    mode,
                    decode_step: step,
                    operation_id: operation.id.clone(),
                    material,
                });
            }
        }
    }
    let issuance_digest = canonical_digest(
        ISSUANCE_DOMAIN,
        &IssuanceContext {
            estimate_digest: &estimate.estimate_digest,
            ordered_tensors: &digests,
        },
    );
    Ok(BoundLogRowQ7Session {
        plan: plan.clone(),
        profile: profile.clone(),
        tensor_policy: *tensor_policy,
        estimate,
        issuance_digest,
        pending,
        aborted: false,
    })
}

impl BoundLogRowQ7Session {
    pub fn estimate(&self) -> &BoundLogRowQ7SessionEstimate {
        &self.estimate
    }

    pub fn issuance_digest(&self) -> &Digest {
        &self.issuance_digest
    }

    pub fn remaining_tensors(&self) -> usize {
        self.pending.len()
    }

    pub fn abort(&mut self) {
        self.pending.clear();
        self.aborted = true;
    }

    /// Caller must supply the next semantic operation. Any mismatch, malformed
    /// input or evaluator failure burns all unconsumed material, including the
    /// current tensor; replay and reordering cannot mint a replacement.
    pub fn evaluate_float32(
        &mut self,
        mode: DecoderMode,
        decode_step: u64,
        operation_id: &str,
        values: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.aborted || self.pending.is_empty() {
            return Err("LogRow Q7 session has no unconsumed material".into());
        }
        let expected = self.pending.front().expect("checked nonempty");
        if expected.mode != mode
            || expected.decode_step != decode_step
            || expected.operation_id != operation_id
        {
            self.abort();
            return Err("LogRow Q7 session operation order differs from its plan".into());
        }
        let current = self.pending.pop_front().expect("checked nonempty");
        let result = (|| {
            if values.len() != current.material.elements() {
                return Err("LogRow Q7 session input shape differs from its plan".into());
            }
            let input = values
                .iter()
                .copied()
                .map(|value| compact_q7_from_f32(value).map_err(|error| error.to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            let (evaluation, decoder) = current.material.encode(&input)?;
            let output = evaluation.evaluate(
                &self.plan,
                mode,
                operation_id,
                &self.profile,
                &self.tensor_policy,
            )?;
            decoder.decode(output).map(|values| {
                values
                    .into_iter()
                    .map(compact_q7_to_f32)
                    .collect::<Vec<_>>()
            })
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }
}
