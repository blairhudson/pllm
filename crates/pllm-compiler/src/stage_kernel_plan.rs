//! Bounded phase-aware device selection for a fixed public decoder schedule.
use crate::{lower_decoder_runtime_schedule, DecoderRuntimeExecutor};
use pllm_models::{DecoderPlan, ModelOperator};
use pllm_types::{canonical_bytes, canonical_digest, Digest};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    weights: Vec<String>,
    cpu_prefill_ns: u64,
    cpu_decode_ns: u64,
    metal_prefill_ns: u64,
    metal_decode_ns: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Evidence {
    schema: String,
    model_plan_digest: Digest,
    schedule_digest: Digest,
    binding_digest: Digest,
    build_digest: Digest,
    prefill_rows: u64,
    decode_steps: u64,
    profiles: Vec<Profile>,
}

#[derive(Clone, Serialize)]
struct Choice {
    weights: Vec<String>,
    prefill: &'static str,
    decode: &'static str,
    device_bytes: u64,
    estimated_ns: u64,
}

#[derive(Clone)]
struct Candidate {
    bytes: u64,
    cost: u64,
    choices: Vec<bool>,
}

pub fn stage_kernel_plan(
    plan: &[u8],
    composition: &[u8],
    evidence: &[u8],
    budget: u64,
) -> Result<Vec<u8>, String> {
    if plan.len() > (8 << 20)
        || composition.len() > (1 << 20)
        || evidence.len() > (1 << 20)
        || budget > (2 << 30)
    {
        return Err("stage kernel planning resource admission exceeded".into());
    }
    let plan: DecoderPlan = serde_json::from_slice(plan).map_err(|e| e.to_string())?;
    let schedule = lower_decoder_runtime_schedule(&plan, composition).map_err(|e| e.to_string())?;
    let profile: Evidence = serde_json::from_slice(evidence).map_err(|e| e.to_string())?;
    if profile.schema != "pllm.stage_kernel_profile.v1"
        || profile.model_plan_digest != plan.digest()
        || profile.schedule_digest != schedule.digest()
        || profile.prefill_rows < 2
        || profile.prefill_rows > plan.prefill.query_sequence
        || profile.decode_steps
            > plan
                .decode
                .maximum_key_sequence
                .saturating_sub(profile.prefill_rows)
        || schedule.nonlinear.is_some()
        || schedule.retained_state.is_some()
        || schedule.mlp_tiling.is_some()
        || schedule.prefill.batch != 1
        || profile.profiles.len() > 512
    {
        return Err("stage kernel evidence source, workload or schedule differs".into());
    }
    let ops: BTreeMap<_, _> = plan
        .prefill
        .operations
        .iter()
        .map(|op| (op.id.as_str(), op))
        .collect();
    let mut groups = BTreeMap::new();
    for step in &schedule.prefill.steps {
        if !step.operators.iter().all(|op| *op == ModelOperator::Linear) {
            continue;
        }
        if step.executor != DecoderRuntimeExecutor::RemoteStage || step.terminal_row_only {
            return Err("stage kernel reference requires unchanged remote dense groups".into());
        }
        let input = step
            .input_ids
            .first()
            .and_then(|id| ops.get(id.as_str()))
            .and_then(|op| op.output_shape.last())
            .copied()
            .ok_or("missing input width")?;
        let output = step
            .outputs
            .iter()
            .try_fold(0u64, |sum, op| sum.checked_add(op.stage_width))
            .ok_or("output width overflow")?;
        let weights = input.checked_mul(output).ok_or("weight size overflow")?;
        // Immutable i8 GPU snapshot, int32 input/output and kernel-bound scratch.
        // Sum scratch conservatively across selected owners rather than hiding it.
        let scratch = input
            .checked_add(output)
            .and_then(|x| x.checked_mul(profile.prefill_rows))
            .and_then(|x| x.checked_mul(8))
            .ok_or("device scratch overflow")?;
        let bytes = weights.checked_add(scratch).ok_or("device size overflow")?;
        if groups.insert(step.weight_ids.clone(), bytes).is_some() {
            return Err("stage kernel reference requires exclusive weight groups".into());
        }
    }
    let decode: BTreeSet<_> = schedule
        .decode
        .steps
        .iter()
        .filter(|step| step.operators.iter().all(|op| *op == ModelOperator::Linear))
        .map(|step| step.weight_ids.clone())
        .collect();
    if groups.is_empty()
        || groups.len() != profile.profiles.len()
        || decode != groups.keys().cloned().collect()
    {
        return Err("stage kernel profile coverage is incomplete".into());
    }
    let mut seen = BTreeSet::new();
    let mut frontier = vec![Candidate {
        bytes: 0,
        cost: 0,
        choices: vec![],
    }];
    let mut max_frontier = 1;
    let mut alternatives = Vec::new();
    for row in &profile.profiles {
        if !seen.insert(row.weights.clone()) {
            return Err("duplicate stage kernel profile".into());
        }
        let bytes = *groups
            .get(&row.weights)
            .ok_or("unknown stage kernel group")?;
        if [
            row.cpu_prefill_ns,
            row.cpu_decode_ns,
            row.metal_prefill_ns,
            row.metal_decode_ns,
        ]
        .iter()
        .any(|x| *x == 0 || *x > 3_600_000_000_000)
        {
            return Err("stage kernel timing is missing or exceeds its bound".into());
        }
        let total = |prefill: u64, decode: u64| {
            prefill
                .checked_add(
                    decode
                        .checked_mul(profile.decode_steps)
                        .ok_or("timing overflow")?,
                )
                .ok_or("timing overflow")
        };
        let cpu = Choice {
            weights: row.weights.clone(),
            prefill: "cpu",
            decode: "cpu",
            device_bytes: 0,
            estimated_ns: total(row.cpu_prefill_ns, row.cpu_decode_ns)?,
        };
        let device = Choice {
            weights: row.weights.clone(),
            prefill: if row.metal_prefill_ns < row.cpu_prefill_ns {
                "metal"
            } else {
                "cpu"
            },
            decode: if profile.decode_steps > 0 && row.metal_decode_ns < row.cpu_decode_ns {
                "metal"
            } else {
                "cpu"
            },
            device_bytes: bytes,
            estimated_ns: total(
                row.cpu_prefill_ns.min(row.metal_prefill_ns),
                row.cpu_decode_ns.min(row.metal_decode_ns),
            )?,
        };
        let mut next = Vec::new();
        for old in frontier {
            for (selected, choice) in [(false, &cpu), (true, &device)] {
                if choice.device_bytes > (512 << 20) {
                    continue;
                }
                let memory = old
                    .bytes
                    .checked_add(choice.device_bytes)
                    .ok_or("memory overflow")?;
                if memory > budget {
                    continue;
                }
                let cost = old
                    .cost
                    .checked_add(choice.estimated_ns)
                    .ok_or("cost overflow")?;
                let mut choices = old.choices.clone();
                choices.push(selected);
                next.push(Candidate {
                    bytes: memory,
                    cost,
                    choices,
                });
            }
        }
        next.sort_by_key(|row| (row.bytes, row.cost));
        let mut best = u64::MAX;
        next.retain(|row| {
            if row.cost < best {
                best = row.cost;
                true
            } else {
                false
            }
        });
        if next.len() > 4096 {
            return Err("stage kernel Pareto frontier exceeds its bound".into());
        }
        max_frontier = max_frontier.max(next.len());
        frontier = next;
        alternatives.push((cpu, device));
    }
    let best = frontier
        .into_iter()
        .min_by_key(|row| (row.cost, row.bytes))
        .ok_or("no feasible stage kernel choice")?;
    let evidence: serde_json::Value =
        serde_json::from_slice(evidence).map_err(|e| e.to_string())?;
    let choices: Vec<_> = alternatives
        .into_iter()
        .zip(best.choices)
        .map(|((cpu, device), selected)| if selected { device } else { cpu })
        .collect();
    Ok(canonical_bytes(
        &json!({"schema": "pllm.stage_kernel_plan.v1", "model_plan_digest": profile.model_plan_digest,
        "schedule_digest": profile.schedule_digest, "binding_digest": profile.binding_digest, "build_digest": profile.build_digest,
        "profile_digest": canonical_digest("pllm.stage_kernel_profile.v1", &evidence), "budget_bytes": budget,
        "device_bytes": best.bytes, "estimated_kernel_ns": best.cost, "prefill_rows": profile.prefill_rows,
        "decode_steps": profile.decode_steps, "maximum_frontier": max_frontier, "choices": choices}),
    ))
}
