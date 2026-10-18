//! Native capability and bounded source admission for replicated execution.
use super::{program, Program, WeightMode};
use crate::{ExperimentDeployment, ExperimentDocument, ExperimentPipeline};
use pllm_types::canonical_bytes;

pub const PROTOCOL: &str = "pllm/puma-three-party-decoder/v1";
pub const TOPOLOGY: &str = "pllm/three-party-replicated/v1";
pub const STATE: &str = "pllm/replicated-private-prefix/v1";
pub const NUMERIC: &str = "puma.source-q18.rmsnorm-epsilon1.full-prefix.final-greedy-ids.v1";
pub const PUBLIC_NUMERIC: &str =
    "puma.public-q18.rmsnorm-epsilon1.full-prefix.batched-final-greedy-ids.v1";

fn weights(document: &ExperimentDocument) -> Result<WeightMode, String> {
    let params = &document
        .pipeline
        .components
        .get("linear")
        .ok_or("missing replicated linear component")?
        .params;
    match serde_json::to_value(params).unwrap() {
        value if value == serde_json::json!({}) => Ok(WeightMode::Private),
        value if value == serde_json::json!({"weights":"public"}) => Ok(WeightMode::Public),
        _ => Err("invalid replicated weight visibility".into()),
    }
}

pub(crate) fn selected(pipeline: &ExperimentPipeline) -> bool {
    pipeline
        .components
        .values()
        .any(|v| matches!(v.component.as_str(), PROTOCOL | TOPOLOGY | STATE))
}
pub(crate) fn validate(document: &ExperimentDocument) -> Result<(), String> {
    let weights = weights(document)?;
    let expected = [
        (
            "linear",
            PROTOCOL,
            if weights == WeightMode::Public {
                serde_json::json!({"weights":"public"})
            } else {
                serde_json::json!({})
            },
        ),
        ("topology", TOPOLOGY, serde_json::json!({})),
        ("state", STATE, serde_json::json!({})),
        ("kernels", "pllm/cpu", serde_json::json!({"threads":1})),
    ];
    if document.pipeline.components.len() != expected.len()
        || !matches!(
            document.pipeline.model.kind.as_str(),
            "huggingface" | "safetensors"
        )
        || !matches!(document.deployment, ExperimentDeployment::Local { .. })
        || !(1..=32).contains(&document.budget.max_input_tokens)
        || !(1..=8).contains(&document.budget.max_new_tokens)
        || !(1..=64).contains(&document.budget.requests)
    {
        return Err("replicated source, deployment or workload is unsupported".into());
    }
    for (slot, id, params) in expected {
        let item = document
            .pipeline
            .components
            .get(slot)
            .ok_or("incomplete replicated composition")?;
        if item.component != id || serde_json::to_value(&item.params).unwrap() != params {
            return Err("incompatible replicated component or numeric choice".into());
        }
    }
    Ok(())
}

pub fn source_contract(
    experiment: &[u8],
    config: &[u8],
    rows: usize,
    outputs: usize,
    source: &str,
    tokenizer: &str,
) -> Result<Vec<u8>, String> {
    if experiment.is_empty() || experiment.len() > 32 << 10 {
        return Err("replicated Experiment byte bound".into());
    }
    crate::resolve_experiment(experiment)?;
    let document: ExperimentDocument =
        serde_json::from_slice(experiment).map_err(|_| "invalid replicated Experiment")?;
    validate(&document)?;
    for pin in [source, tokenizer] {
        if pin.len() != 64
            || !pin
                .bytes()
                .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
        {
            return Err("invalid replicated source pin".into());
        }
    }
    if rows == 0
        || rows as u64 > document.budget.max_input_tokens
        || outputs == 0
        || outputs as u64 > document.budget.max_new_tokens
    {
        return Err("replicated request exceeds Experiment".into());
    }
    let weights = weights(&document)?;
    Program::source_shapes_with_weights(
        config,
        document.budget.max_input_tokens as usize,
        document.budget.max_new_tokens as usize,
        weights,
    )?;
    let shapes = Program::source_shapes_with_weights(config, rows, outputs, weights)?;
    let plan = program::lower_with_weights(config, rows, outputs, weights)?;
    let head = plan
        .prefill
        .operations
        .iter()
        .find(|v| v.operator == pllm_models::ModelOperator::OutputHead)
        .ok_or("missing full head")?;
    let width = shapes[head.attributes["weight"]
        .as_str()
        .ok_or("missing head weight")?][1];
    let size = shapes
        .values()
        .map(|v| v.iter().product::<usize>() * 4)
        .sum::<usize>();
    let mut contract = serde_json::json!({"schema_version":"pllm.replicated.sdk-contract.v1",
        "configuration":crate::configuration_digest_bytes(experiment),"composition":crate::pipeline_digest(&document.pipeline),
        "source_lock":source,"tokenizer":tokenizer,"numeric_profile":NUMERIC,"plan":plan,
        "source_shapes":shapes,"source_float32_bytes":size,"input_rows":rows,"decode_steps":outputs-1,
        "selection_prefix_rows":(rows..rows+outputs).collect::<Vec<_>>(),
        "vocabulary":head.output_shape[1],"width":width,"execution_payload_estimate_bytes":192usize<<20});
    if weights == WeightMode::Public {
        contract["weights"] = serde_json::json!("public");
        contract["numeric_profile"] = serde_json::json!(PUBLIC_NUMERIC);
        contract["decoder_resources"] =
            serde_json::json!({"profile":"pllm.puma.public-source-capacity.v1"});
        contract["public_weight_snapshot_bytes_per_worker"] = serde_json::json!(size * 2);
        contract["public_weight_read_buffer_bytes"] = serde_json::json!(65536);
    }
    Ok(canonical_bytes(&contract))
}
