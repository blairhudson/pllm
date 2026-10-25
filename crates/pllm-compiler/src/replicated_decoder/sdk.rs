//! Native capability and bounded source admission for replicated execution.
use super::{program, ExecutionPolicy, PeerTransport, Program, RangePolicy, WeightMode};
use crate::{ExperimentDeployment, ExperimentDocument, ExperimentPipeline};
use pllm_types::canonical_bytes;
use std::sync::Arc;

pub const PROTOCOL: &str = "pllm/puma-three-party-decoder/v1";
pub const TOPOLOGY: &str = "pllm/three-party-replicated/v1";
pub const STATE: &str = "pllm/replicated-private-prefix/v1";
pub const NUMERIC: &str = "puma.source-q18.rmsnorm-epsilon1.full-prefix.final-greedy-ids.v1";
pub const PUBLIC_NUMERIC: &str =
    "puma.public-q18.rmsnorm-epsilon1.full-prefix.batched-final-greedy-ids.v1";
pub const CHECKED_NUMERIC: &str =
    "puma.public-q18.checked-range.rmsnorm-epsilon1.full-prefix.final-greedy-ids.v1";

/// Compiler-native opt-in only. Not a discoverable component, SDK route or a
/// constructor from a compiled Program. Source config lineage is not checkpoint
/// authentication: the actual immutable PublicSource is independently checked.
#[doc(hidden)]
pub struct SourceRequestProgram(std::sync::Arc<Program>);

#[doc(hidden)]
pub use program::source_entry::SourceRequestImport;

/// Private construction choice, not an arithmetic policy or Manifest field.
/// Canonical expanded components omit it, preserving their original identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceStorage {
    Expanded,
    Compact,
}
impl SourceStorage {
    pub(crate) fn from_params(params: &serde_json::Value) -> Result<Self, String> {
        match params.get("storage") {
            None => Ok(Self::Expanded),
            Some(value) if value.as_str() == Some("compact") => {
                if params["weights"].as_str() != Some("public")
                    || params["range_policy"].as_str() != Some("checked")
                    || !params["max_abs_activation"]
                        .as_u64()
                        .is_some_and(|v| (1..=1024).contains(&v))
                    || params.get("lookup").is_some()
                    || params.get("selection").is_some()
                    || params.get("peer_transport").is_some()
                {
                    return Err("unsupported compact replicated storage composition".into());
                }
                Ok(Self::Compact)
            }
            _ => Err("noncanonical replicated storage choice".into()),
        }
    }
    fn component_params(
        self,
        weights: WeightMode,
        execution: ExecutionPolicy,
    ) -> serde_json::Value {
        let mut value = execution.component_params(weights);
        if self == Self::Compact {
            value
                .as_object_mut()
                .expect("component object")
                .insert("storage".into(), serde_json::json!("compact"));
        }
        value
    }
}

fn component_policy(
    mut params: serde_json::Value,
) -> Result<(WeightMode, ExecutionPolicy, SourceStorage), String> {
    let storage = SourceStorage::from_params(&params)?;
    params
        .as_object_mut()
        .ok_or("invalid replicated component")?
        .remove("storage");
    let (weights, execution) = ExecutionPolicy::from_component(params)?;
    Ok((weights, execution, storage))
}

/// Read only the already authenticated/admitted Experiment's construction mode.
/// This grants no source, input, Program or public snapshot authority.
pub(crate) fn source_storage(document: &ExperimentDocument) -> Result<SourceStorage, String> {
    let params = &document
        .pipeline
        .components
        .get("linear")
        .ok_or("missing replicated linear component")?
        .params;
    Ok(
        component_policy(
            serde_json::to_value(params).map_err(|_| "invalid replicated component")?,
        )?
        .2,
    )
}

impl SourceRequestProgram {
    pub(in crate::replicated_decoder) fn snapshot_directory(
        &self,
    ) -> Result<
        &Arc<super::program::source_entry::snapshot_directory::OwnedSnapshotDirectory>,
        super::Error,
    > {
        self.0
            .compact_storage()
            .ok_or(super::Error::Binding)?
            .snapshot_directory()
    }
    // The only compact wrapper constructor consumes the genuine pending source
    // funnel. There is deliberately no constructor taking an arbitrary Program.
    pub(super) fn finish_compact(import: SourceRequestImport) -> Result<Self, super::Error> {
        Ok(Self(
            import.compile_compact_storage()?.into_source_program()?,
        ))
    }
    pub(crate) fn collection_phase(
        &self,
        guard: &mut pllm_garble::shared_allocation::OwnedExternalCharge,
        bytes: usize,
        visits: u64,
    ) -> Result<(), super::Error> {
        self.0
            .compact_storage()
            .ok_or(super::Error::Binding)?
            .collection_phase(guard, bytes, visits)
    }
    pub(crate) fn input_count(&self) -> usize {
        self.0.input_count()
    }
    /// Test-only wrapper lens; production borrows the actual retained Program
    /// during its mutable collection phase to avoid overlapping owner borrows.
    /// Actual retained unpublished header, not a rebuilt descriptor. Publisher
    /// serializes final `program().digest()` in its wire Manifest, not zero here.
    #[cfg(test)]
    pub(crate) fn source_manifest(&self) -> Result<&super::Manifest, super::Error> {
        self.0
            .compact_storage()
            .ok_or(super::Error::Binding)?
            .source_manifest()
    }
    /// No plan copy/JSON roundtrip, guard extraction or manufactured request
    /// permit. The complete call consumes its source/admission even on failure.
    #[allow(clippy::too_many_arguments)]
    pub fn construct(
        original: crate::original_source::OriginalSourcePlan,
        manifest: pllm_garble::shared_allocation::OwnedCharged<super::Manifest>,
        public: std::sync::Arc<super::PublicSource>,
        registry_guard: pllm_garble::shared_allocation::OwnedExternalCharge,
        imports_guard: pllm_garble::shared_allocation::OwnedExternalCharge,
        controller: pllm_garble::shared_allocation::OwnedExternalCharge,
        limits: crate::original_source::Limits,
        cap: usize,
    ) -> Result<Self, super::Error> {
        use super::source_registry::{self, SourceConstructionOwners};
        use std::mem::size_of;
        if !std::ptr::eq(manifest.budget(), original.budget()) {
            return Err(super::Error::Binding);
        }
        let dto = manifest.payload();
        let minimum = std::mem::size_of_val(manifest.payload())
            .checked_add(dto.schema.capacity())
            .and_then(|n| n.checked_add(dto.config.capacity()))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or(super::Error::Bounds)?;
        let minimum = minimum
            .checked_add(source_registry::source_metadata_bytes(
                &dto.tensors,
                &dto.norms,
            )?)
            .ok_or(super::Error::Bounds)?;
        if manifest.charged_bytes() < minimum {
            return Err(super::Error::Bounds);
        }
        let baseline = original
            .budget()
            .used_bytes()
            .checked_sub(registry_guard.payload_bytes())
            .ok_or(super::Error::Bounds)?;
        let owners = SourceConstructionOwners {
            manifest: dto,
            incoming: None,
            public: Some(&public),
            retained_baseline_bytes: baseline,
        };
        let program = program::source_entry::compile(
            original,
            owners,
            std::sync::Arc::clone(&public),
            program::source_entry::EntryGuards {
                registry: registry_guard,
                imports: imports_guard,
                controller,
            },
            limits,
            cap,
        )?;
        Ok(Self(program))
    }

    pub fn program(&self) -> &std::sync::Arc<Program> {
        &self.0
    }

    /// Existing original setup/runtime/PRSS are the only execution constructors.
    /// Consuming this wrapper issues no BODY handoff or transport-drain proof.
    pub fn worker(
        self,
        runtime: super::puma::multiply::PartyRuntime,
        session: [u8; 32],
    ) -> Result<super::Worker, super::Error> {
        self.0.source_request_worker(runtime, session)
    }
}

pub(crate) fn policy(
    document: &ExperimentDocument,
) -> Result<(WeightMode, ExecutionPolicy), String> {
    let params = &document
        .pipeline
        .components
        .get("linear")
        .ok_or("missing replicated linear component")?
        .params;
    let (weights, execution, _) = component_policy(serde_json::to_value(params).unwrap())?;
    Ok((weights, execution))
}

pub(crate) fn selected(pipeline: &ExperimentPipeline) -> bool {
    pipeline
        .components
        .values()
        .any(|v| matches!(v.component.as_str(), PROTOCOL | TOPOLOGY | STATE))
}
pub(crate) fn validate(document: &ExperimentDocument) -> Result<(), String> {
    let (weights, execution) = policy(document)?;
    let storage = source_storage(document)?;
    let expected = [
        (
            "linear",
            PROTOCOL,
            storage.component_params(weights, execution),
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
    let (weights, execution) = policy(&document)?;
    Program::source_shapes_with_policy(
        config,
        document.budget.max_input_tokens as usize,
        document.budget.max_new_tokens as usize,
        weights,
        execution,
    )?;
    let shapes = Program::source_shapes_with_policy(config, rows, outputs, weights, execution)?;
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
    if !execution.is_default() {
        contract["execution"] = serde_json::to_value(execution).unwrap();
    }
    if execution.range_policy == RangePolicy::Checked {
        contract["numeric_profile"] = serde_json::json!(CHECKED_NUMERIC);
    }
    if execution.peer_transport == PeerTransport::Streams {
        contract["peer_stream_workspace_bytes_per_worker"] = serde_json::json!(8usize << 20);
    }
    Ok(canonical_bytes(&contract))
}

#[cfg(test)]
mod storage_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn storage_omission_preserves_canonical_component_bytes() {
        for params in [
            json!({}),
            json!({"weights":"public"}),
            json!({"weights":"public","range_policy":"checked","max_abs_activation":1}),
        ] {
            let before = canonical_bytes(&params);
            let (weights, execution, storage) = component_policy(params.clone()).unwrap();
            assert_eq!(storage, SourceStorage::Expanded);
            assert_eq!(
                canonical_bytes(&storage.component_params(weights, execution)),
                before
            );
        }
    }
    #[test]
    fn compact_is_composition_bound_not_arithmetic_policy() {
        let params = json!({"weights":"public","range_policy":"checked","max_abs_activation":1,"storage":"compact"});
        let (weights, execution, storage) = component_policy(params.clone()).unwrap();
        assert_eq!(storage, SourceStorage::Compact);
        assert_eq!(storage.component_params(weights, execution), params);
        assert!(serde_json::to_value(execution)
            .unwrap()
            .get("storage")
            .is_none());
        assert_ne!(
            canonical_bytes(&params),
            canonical_bytes(&execution.component_params(weights))
        );
    }
    #[test]
    fn unsupported_or_redundant_storage_choices_refuse() {
        let base = json!({"weights":"public","range_policy":"checked","max_abs_activation":1,"storage":"compact"});
        for (key, value) in [
            ("storage", json!("expanded")),
            ("storage", json!(true)),
            ("storage", json!("unknown")),
            ("weights", json!("private")),
            ("range_policy", json!("static")),
            ("lookup", json!("trie")),
            ("selection", json!("tiled")),
            ("peer_transport", json!("streams")),
            ("max_abs_activation", json!(0)),
        ] {
            let mut candidate = base.clone();
            candidate[key] = value;
            assert!(component_policy(candidate).is_err(), "{key}");
        }
    }
}
