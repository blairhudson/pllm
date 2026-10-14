//! Source-locked, load-ready public execution metadata. Tensor kernels stay separate.
use crate::lower_decoder_runtime_schedule;
use pllm_models::DecoderPlan;
use pllm_types::{canonical_bytes, digest_bytes};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

const SCHEMA: &str = "pllm.execution_capsule.v1";
const MAX_BYTES: usize = 8 << 20;
const MAX_STEPS: usize = 16384;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Phase {
    graph: Value,
    schedule: Value,
    state_kinds: BTreeMap<String, String>,
    logits_id: String,
    steps: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCapsule {
    schema: String,
    build: String,
    binding: String,
    model: String,
    composition: String,
    graphs: Value,
    schedule: Value,
    phases: [Phase; 2],
}

fn identity(value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err("execution capsule needs a strict digest".into());
    }
    Ok(())
}

impl ExecutionCapsule {
    pub fn compile(
        plan: &[u8],
        composition: &[u8],
        build: &str,
        binding: &str,
    ) -> Result<Vec<u8>, String> {
        identity(build)?;
        identity(binding)?;
        if plan.len() > MAX_BYTES || composition.len() > (1 << 20) {
            return Err("execution capsule source exceeds admission".into());
        }
        let plan: DecoderPlan = serde_json::from_slice(plan).map_err(|e| e.to_string())?;
        let schedule =
            lower_decoder_runtime_schedule(&plan, composition).map_err(|e| e.to_string())?;
        if schedule.nonlinear.is_some()
            || schedule.private_boundary.is_some()
            || schedule.retained_state.is_some()
            || schedule.mlp_tiling.is_some()
        {
            return Err("execution capsule requires an untransformed schedule".into());
        }
        let mut graphs = serde_json::to_value(&plan).map_err(|e| e.to_string())?;
        let mut schedule_doc = serde_json::to_value(&schedule).map_err(|e| e.to_string())?;
        let mut phases = Vec::new();
        for name in ["prefill", "decode"] {
            let mut graph = graphs
                .as_object_mut()
                .ok_or("invalid graph")?
                .remove(name)
                .ok_or("missing graph")?;
            let mut phase_schedule = schedule_doc
                .as_object_mut()
                .ok_or("invalid schedule")?
                .remove(name)
                .ok_or("missing phase")?;
            let operations = graph
                .as_object_mut()
                .ok_or("invalid graph")?
                .remove("operations")
                .ok_or("missing operations")?;
            let operations = operations.as_array().ok_or("invalid operations")?;
            let by_id: BTreeMap<_, _> = operations
                .iter()
                .map(|op| (op["id"].as_str().unwrap_or(""), op))
                .collect();
            let selections: Vec<_> = operations
                .iter()
                .filter(|op| op["operator"] == "greedy_token_selection")
                .collect();
            if selections.len() != 1
                || selections[0]["inputs"]
                    .as_array()
                    .is_none_or(|a| a.len() != 1)
            {
                return Err("execution capsule requires one logit source".into());
            }
            let logits_id = selections[0]["inputs"][0]
                .as_str()
                .ok_or("invalid logit source")?
                .to_string();
            let mut state_kinds = BTreeMap::new();
            for op in operations {
                if let Some(kind) = op["state_kind"].as_str() {
                    state_kinds.insert(
                        op["id"].as_str().ok_or("invalid state owner")?.to_string(),
                        kind.to_string(),
                    );
                }
            }
            let steps = phase_schedule
                .as_object_mut()
                .ok_or("invalid phase")?
                .remove("steps")
                .ok_or("missing steps")?;
            let steps = steps.as_array().ok_or("invalid steps")?;
            if steps.is_empty() || steps.len() > MAX_STEPS {
                return Err("execution capsule step admission exceeded".into());
            }
            let mut uses = BTreeMap::<String, usize>::new();
            for step in steps {
                for id in step["input_ids"].as_array().ok_or("invalid step inputs")? {
                    *uses
                        .entry(id.as_str().ok_or("invalid input")?.to_string())
                        .or_default() += 1;
                }
            }
            let mut frames = Vec::with_capacity(steps.len());
            for step in steps {
                let mut local = BTreeMap::new();
                for id in step["operation_ids"]
                    .as_array()
                    .ok_or("invalid step operations")?
                {
                    let id = id.as_str().ok_or("invalid operation")?;
                    local.insert(id, *by_id.get(id).ok_or("unresolved step operation")?);
                }
                let mut retire = Vec::new();
                for id in step["input_ids"].as_array().ok_or("invalid step inputs")? {
                    let id = id.as_str().ok_or("invalid input")?;
                    let left = uses.get_mut(id).ok_or("unresolved input lifetime")?;
                    *left -= 1;
                    if *left == 0 && id != logits_id {
                        retire.push(id);
                    }
                }
                frames.push(
                    String::from_utf8(canonical_bytes(
                        &json!({"step": step, "operations": local, "retire": retire}),
                    ))
                    .map_err(|e| e.to_string())?,
                );
            }
            phases.push(Phase {
                graph,
                schedule: phase_schedule,
                state_kinds,
                logits_id,
                steps: frames,
            });
        }
        let capsule = Self {
            schema: SCHEMA.into(),
            build: build.into(),
            binding: binding.into(),
            model: schedule.model_plan_digest.to_string(),
            composition: schedule.composition_digest.to_string(),
            graphs,
            schedule: schedule_doc,
            phases: phases.try_into().map_err(|_| "invalid phases")?,
        };
        let bytes = canonical_bytes(&capsule);
        if bytes.len() > MAX_BYTES {
            return Err("execution capsule payload admission exceeded".into());
        }
        Ok(bytes)
    }

    pub fn digest(bytes: &[u8]) -> String {
        digest_bytes(SCHEMA, bytes).to_string()
    }

    /// The expected digest comes from a trusted offline publisher, never this blob.
    pub fn load(
        bytes: &[u8],
        expected: &str,
        build: &str,
        binding: &str,
        model: &str,
        composition: &str,
    ) -> Result<Self, String> {
        for value in [expected, build, binding, model, composition] {
            identity(value)?;
        }
        if bytes.len() > MAX_BYTES || Self::digest(bytes) != expected {
            return Err("execution capsule artifact differs or exceeds admission".into());
        }
        let capsule: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if capsule.schema != SCHEMA
            || capsule.schedule.get("private_boundary").is_some()
            || capsule.build != build
            || capsule.binding != binding
            || capsule.model != model
            || capsule.composition != composition
        {
            return Err("execution capsule source, build or binding differs".into());
        }
        for phase in &capsule.phases {
            if phase.steps.is_empty()
                || phase.steps.len() > MAX_STEPS
                || phase.steps.iter().any(|s| s.len() > (64 << 10))
            {
                return Err("execution capsule step admission exceeded".into());
            }
        }
        Ok(capsule)
    }

    pub fn metadata(&self) -> Vec<u8> {
        let mut graphs = self.graphs.clone();
        let mut schedule = self.schedule.clone();
        let mut facts = BTreeMap::new();
        for (name, phase) in ["prefill", "decode"].into_iter().zip(&self.phases) {
            graphs[name] = phase.graph.clone();
            schedule[name] = phase.schedule.clone();
            facts.insert(
                name,
                json!({"state_kinds": phase.state_kinds, "logits_id": phase.logits_id}),
            );
        }
        canonical_bytes(&json!({"graphs": graphs, "schedule": schedule, "facts": facts}))
    }

    pub fn step_count(&self, phase: usize) -> Result<usize, String> {
        self.phases
            .get(phase)
            .map(|p| p.steps.len())
            .ok_or_else(|| "invalid capsule phase".into())
    }

    pub fn step(&self, phase: usize, index: usize) -> Result<&[u8], String> {
        self.phases
            .get(phase)
            .and_then(|p| p.steps.get(index))
            .map(|s| s.as_bytes())
            .ok_or_else(|| "invalid capsule step".into())
    }
}
