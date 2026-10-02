//! Offline placement legality derived exclusively from native composition and
//! complete decoder scheduling. This does not authenticate offers, reserve
//! capacity, prove independence, or authorize live network execution.

use super::{
    lower_decoder_runtime_schedule, validate_decoder_linear_composition, DecoderCompositionKind,
    DecoderRuntimeExecutor, DecoderRuntimeSchedule, ExperimentPipeline,
};
use pllm_models::{DecoderGraph, DecoderPlan, ModelOperation, ModelOperator};
use pllm_types::network::{
    NetworkPlacementRequest, NetworkPlacementRequirements, NetworkRoleRequirement,
    ValidatedNetworkPlacement, NETWORK_CAPABILITIES, NETWORK_MAX_DOCUMENT_BYTES,
    NETWORK_MAX_MODEL_PLAN_BYTES, NETWORK_MAX_PARTIES, NETWORK_MAX_ROLES,
    NETWORK_PLACEMENT_DIGEST_DOMAIN, NETWORK_PLACEMENT_REQUIREMENTS_SCHEMA,
    NETWORK_PLACEMENT_SCHEMA,
};
use pllm_types::{canonical_bytes, canonical_digest, valid_identity, Digest};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

type RoleCapabilities = &'static [(&'static str, &'static str)];
type OperatorSeparation = Option<(&'static str, &'static str)>;

fn bounded(bytes: &[u8], maximum: usize, kind: &str) -> Result<(), String> {
    if bytes.is_empty() || bytes.len() > maximum {
        return Err(format!(
            "network {kind} has an invalid byte length (maximum {maximum})"
        ));
    }
    Ok(())
}

fn nonzero(digest: &Digest) -> bool {
    digest.as_str().bytes().any(|byte| byte != b'0')
}

struct NativePlacementAdmission {
    model_identity: Digest,
    requirements: NetworkPlacementRequirements,
}

/// One admission path for assignment-free inspection and placement validation.
fn admit_network_placement(
    model_plan: &[u8],
    composition: &[u8],
) -> Result<NativePlacementAdmission, String> {
    bounded(model_plan, NETWORK_MAX_MODEL_PLAN_BYTES, "model plan")?;
    bounded(composition, NETWORK_MAX_DOCUMENT_BYTES, "composition")?;
    let plan: DecoderPlan = serde_json::from_slice(model_plan)
        .map_err(|error| format!("invalid network model plan: {error}"))?;
    if canonical_bytes(&plan) != model_plan || !nonzero(&plan.config_digest) {
        return Err("network model plan must be canonical and have a nonzero config digest".into());
    }
    let pipeline: ExperimentPipeline = serde_json::from_slice(composition)
        .map_err(|error| format!("invalid network composition: {error}"))?;
    if canonical_bytes(&pipeline) != composition {
        return Err("network composition must use canonical JSON".into());
    }
    super::validate_experiment_model(&pipeline.model)?;
    // Exact component and privacy-option admission. Pending slots cannot inject roles.
    let kind = validate_decoder_linear_composition(&pipeline)?;
    let (role_capabilities, separation): (RoleCapabilities, OperatorSeparation) = match kind {
        DecoderCompositionKind::MaskedLinear | DecoderCompositionKind::VerifiedMaskedLinear => (
            &[
                ("client", "trusted_client"),
                ("inference", "masked_linear_provider"),
                ("preparation", "trusted_preparation"),
            ],
            Some(("inference", "preparation")),
        ),
        DecoderCompositionKind::ClientOnlyLinear => (&[("client", "trusted_client")], None),
        DecoderCompositionKind::TwoOnlineOffsetLinear => (
            &[
                ("client", "trusted_client"),
                ("worker_a", "public_linear_provider"),
                ("worker_b", "public_linear_provider"),
            ],
            Some(("worker_a", "worker_b")),
        ),
        DecoderCompositionKind::Other => {
            return Err("network composition has no admitted role graph".into())
        }
    };
    let schedule = lower_decoder_runtime_schedule(&plan, composition)?;
    if !schedule.complete {
        return Err("network placement requires a complete native schedule".into());
    }
    let minimum_weight_bytes = owned_weight_floors(
        &plan,
        &schedule,
        kind,
        pipeline.components.contains_key("boundary"),
    )?;
    Ok(NativePlacementAdmission {
        model_identity: plan.config_digest,
        requirements: NetworkPlacementRequirements {
            schema: NETWORK_PLACEMENT_REQUIREMENTS_SCHEMA.to_owned(),
            roles: role_capabilities
                .iter()
                .map(|(role, capability)| NetworkRoleRequirement {
                    role_id: (*role).to_owned(),
                    capability: (*capability).to_owned(),
                })
                .collect(),
            separate_operators: separation
                .into_iter()
                .map(|(left, right)| [left.to_owned(), right.to_owned()])
                .collect(),
            minimum_memory_bytes: minimum_weight_bytes.clone(),
            minimum_weight_bytes,
            schedule_digest: schedule.digest(),
            model_plan_digest: schedule.model_plan_digest,
            composition_digest: schedule.composition_digest,
        },
    })
}

/// Derive canonical requirements without offers, estimates or an assignment.
/// Uses exactly the same native admission path as validate_network_placement.
pub fn network_placement_requirements(
    model_plan: &[u8],
    composition: &[u8],
) -> Result<Vec<u8>, String> {
    Ok(canonical_bytes(
        &admit_network_placement(model_plan, composition)?.requirements,
    ))
}

/// Validate three immutable JSON byte documents and return canonical JSON bytes.
/// Model plan and composition must already use native canonical JSON. Request
/// may use arbitrary whitespace/key order, but cannot contain duplicate fields.
pub fn validate_network_placement(
    model_plan: &[u8],
    composition: &[u8],
    document: &[u8],
) -> Result<Vec<u8>, String> {
    bounded(document, NETWORK_MAX_DOCUMENT_BYTES, "placement document")?;
    let admitted = admit_network_placement(model_plan, composition)?;
    let requirements = admitted.requirements;
    let mut request: NetworkPlacementRequest = serde_json::from_slice(document)
        .map_err(|error| format!("invalid network placement document: {error}"))?;
    validate_request(&request, &requirements, &admitted.model_identity)?;
    request
        .roles
        .sort_by(|left, right| left.role_id.cmp(&right.role_id));
    request
        .parties
        .sort_by(|left, right| left.party_id.cmp(&right.party_id));
    for party in &mut request.parties {
        party.capabilities.sort();
        party.allowed_models.sort();
    }
    #[derive(Serialize)]
    struct Commitment<'a> {
        #[serde(flatten)]
        request: &'a NetworkPlacementRequest,
        schedule_digest: Digest,
        model_plan_digest: Digest,
        composition_digest: Digest,
        minimum_weight_bytes: &'a BTreeMap<String, u64>,
        minimum_memory_bytes: &'a BTreeMap<String, u64>,
    }
    let commitment = Commitment {
        request: &request,
        schedule_digest: requirements.schedule_digest,
        model_plan_digest: requirements.model_plan_digest,
        composition_digest: requirements.composition_digest,
        minimum_weight_bytes: &requirements.minimum_weight_bytes,
        minimum_memory_bytes: &requirements.minimum_memory_bytes,
    };
    let placement_digest = canonical_digest(NETWORK_PLACEMENT_DIGEST_DOMAIN, &commitment);
    Ok(canonical_bytes(&ValidatedNetworkPlacement {
        schedule_digest: commitment.schedule_digest,
        model_plan_digest: commitment.model_plan_digest,
        composition_digest: commitment.composition_digest,
        minimum_memory_bytes: requirements.minimum_memory_bytes,
        minimum_weight_bytes: requirements.minimum_weight_bytes,
        placement_digest,
        request,
    }))
}

fn validate_request(
    request: &NetworkPlacementRequest,
    requirements: &NetworkPlacementRequirements,
    model_identity: &Digest,
) -> Result<(), String> {
    if request.schema != NETWORK_PLACEMENT_SCHEMA
        || !valid_identity(&request.client_party_id)
        || !nonzero(&request.snapshot_digest)
        || request.roles.is_empty()
        || request.roles.len() > NETWORK_MAX_ROLES
        || request.parties.is_empty()
        || request.parties.len() > NETWORK_MAX_PARTIES
    {
        return Err("network placement schema, identity, or record count is invalid".into());
    }
    let expected: BTreeSet<_> = requirements
        .roles
        .iter()
        .map(|role| role.role_id.as_str())
        .collect();
    let roles: BTreeMap<_, _> = request
        .roles
        .iter()
        .map(|role| (role.role_id.as_str(), role.party_id.as_str()))
        .collect();
    if roles.len() != request.roles.len()
        || roles.keys().copied().collect::<BTreeSet<_>>() != expected
        || request
            .required_memory_bytes
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != expected
        || request
            .required_weight_bytes
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != expected
    {
        return Err(
            "network assignments and resource maps must name every native role exactly once".into(),
        );
    }
    if roles["client"] != request.client_party_id {
        return Err("network client must be assigned exactly to client_party_id".into());
    }
    let mut parties = BTreeMap::new();
    for party in &request.parties {
        if !valid_identity(&party.party_id)
            || !valid_identity(&party.operator_id)
            || !nonzero(&party.instance_epoch)
            || parties.insert(party.party_id.as_str(), party).is_some()
        {
            return Err(
                "network offers require distinct valid party IDs, operator IDs, and nonzero epochs"
                    .into(),
            );
        }
        if party.expires_at_ms <= request.evaluated_at_ms {
            return Err(format!("network offer {} is expired", party.party_id));
        }
        if party.active_sessions > party.max_sessions {
            return Err(format!(
                "network offer {} has invalid session capacity",
                party.party_id
            ));
        }
        let caps: BTreeSet<_> = party.capabilities.iter().map(String::as_str).collect();
        if caps.is_empty()
            || caps.len() != party.capabilities.len()
            || caps.iter().any(|cap| !NETWORK_CAPABILITIES.contains(cap))
        {
            return Err(format!(
                "network offer {} has unknown or duplicate capabilities",
                party.party_id
            ));
        }
        let models: BTreeSet<_> = party.allowed_models.iter().map(String::as_str).collect();
        if models.is_empty()
            || models.len() != party.allowed_models.len()
            || models.iter().any(|model| {
                *model != "*"
                    && (model.len() != 64
                        || !model
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                        || model.bytes().all(|byte| byte == b'0'))
            })
        {
            return Err(format!(
                "network offer {} allowed_models requires nonzero config digests or '*'",
                party.party_id
            ));
        }
    }
    // Each role consumes one slot; co-located roles aggregate conservatively.
    let mut budgets = BTreeMap::<&str, (u64, u64, u32)>::new();
    for requirement in &requirements.roles {
        let role = requirement.role_id.as_str();
        let capability = &requirement.capability;
        let party = parties
            .get(roles[role])
            .ok_or_else(|| format!("network role {role} names an unknown party"))?;
        if !party.capabilities.iter().any(|cap| cap == capability) {
            return Err(format!(
                "network role {role} requires capability {capability}"
            ));
        }
        if !party
            .allowed_models
            .iter()
            .any(|model| model == "*" || model == model_identity.as_str())
        {
            return Err(format!(
                "network role {role} model config digest is not allowed"
            ));
        }
        for (name, required, floor) in [
            (
                "required_memory_bytes",
                request.required_memory_bytes[role],
                requirements.minimum_memory_bytes[role],
            ),
            (
                "required_weight_bytes",
                request.required_weight_bytes[role],
                requirements.minimum_weight_bytes[role],
            ),
        ] {
            if required < floor {
                return Err(format!(
                    "network role {role} {name} {required} is below native minimum {floor}"
                ));
            }
        }
        let budget = budgets.entry(&party.party_id).or_default();
        budget.0 = budget
            .0
            .checked_add(request.required_memory_bytes[role])
            .ok_or("network memory resource budget overflow")?;
        budget.1 = budget
            .1
            .checked_add(request.required_weight_bytes[role])
            .ok_or("network weight resource budget overflow")?;
        budget.2 = budget
            .2
            .checked_add(1)
            .ok_or("network session resource budget overflow")?;
        if budget.0 > party.max_memory_bytes
            || budget.1 > party.max_weight_bytes
            || budget.2 > party.max_sessions - party.active_sessions
        {
            return Err(format!(
                "network party {} capacity exceeded",
                party.party_id
            ));
        }
    }
    for [left, right] in &requirements.separate_operators {
        let (left, right) = (left.as_str(), right.as_str());
        if parties[roles[left]].operator_id == parties[roles[right]].operator_id {
            return Err(format!(
                "network roles {left} and {right} require separate declared operators"
            ));
        }
    }
    Ok(())
}

/// Deduplicate explicit native artifact identities, never guessed names/content.
/// One byte per element is a conservative resident lower bound, including local
/// tensors whose actual runtime representation may be wider. The complete native
/// schedule supplies execution ownership; providers still retain every body
/// matrix when selected stages are additionally executed on the client.
fn owned_weight_floors(
    plan: &DecoderPlan,
    schedule: &DecoderRuntimeSchedule,
    kind: DecoderCompositionKind,
    remote_head: bool,
) -> Result<BTreeMap<String, u64>, String> {
    let providers: &[&str] = match kind {
        DecoderCompositionKind::MaskedLinear | DecoderCompositionKind::VerifiedMaskedLinear => {
            &["inference", "preparation"]
        }
        DecoderCompositionKind::TwoOnlineOffsetLinear => &["worker_a", "worker_b"],
        DecoderCompositionKind::ClientOnlyLinear => &[],
        DecoderCompositionKind::Other => {
            return Err("network weight ownership is unsupported".into())
        }
    };
    let mut owned: BTreeMap<&str, BTreeSet<&str>> = std::iter::once("client")
        .chain(providers.iter().copied())
        .map(|role| (role, BTreeSet::new()))
        .collect();
    let mut artifacts = BTreeMap::<&str, Vec<u64>>::new();
    for (graph, phase) in [
        (&plan.prefill, &schedule.prefill),
        (&plan.decode, &schedule.decode),
    ] {
        let operations: BTreeMap<_, _> = graph
            .operations
            .iter()
            .map(|op| (op.id.as_str(), op))
            .collect();
        let head = graph
            .operations
            .iter()
            .find(|op| op.operator == ModelOperator::OutputHead)
            .ok_or("network weights require an output head")?;
        let vocabulary = last_dimension(&head.output_shape)?;
        for step in &phase.steps {
            for operation_id in &step.operation_ids {
                let op = operations
                    .get(operation_id.as_str())
                    .ok_or("network weight operation is missing")?;
                let tensors = operation_tensors(graph, op, vocabulary)?;
                let provider_owned = op.operator == ModelOperator::Linear
                    || (op.operator == ModelOperator::OutputHead && remote_head);
                let client_owned = providers.is_empty()
                    || !provider_owned
                    || step.executor == DecoderRuntimeExecutor::ClientLinear;
                for (id, shape) in tensors {
                    checked_weight_elements(&shape)?;
                    if let Some(previous) = artifacts.get(id) {
                        if previous != &shape {
                            return Err(format!(
                                "network weight artifact {id} has inconsistent shapes"
                            ));
                        }
                    } else {
                        artifacts.insert(id, shape);
                    }
                    if client_owned {
                        owned
                            .get_mut("client")
                            .expect("native client role")
                            .insert(id);
                    }
                    if provider_owned {
                        for role in providers {
                            owned
                                .get_mut(role)
                                .expect("native provider role")
                                .insert(id);
                        }
                    }
                }
            }
        }
    }
    owned
        .into_iter()
        .map(|(role, ids)| {
            let bytes = ids.into_iter().try_fold(0_u64, |sum, id| {
                sum.checked_add(checked_weight_elements(&artifacts[id])?)
                    .ok_or_else(|| "network owned weight byte sum overflow".to_owned())
            })?;
            Ok((role.to_owned(), bytes))
        })
        .collect()
}

fn checked_weight_elements(shape: &[u64]) -> Result<u64, String> {
    if shape.is_empty() || shape.contains(&0) {
        return Err("network weight shape must have nonzero dimensions".into());
    }
    shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| "network weight shape byte product overflow".to_owned())
    })
}

fn last_dimension(shape: &[u64]) -> Result<u64, String> {
    shape
        .last()
        .copied()
        .filter(|width| *width > 0)
        .ok_or_else(|| "network weight use has no nonzero feature dimension".to_owned())
}

fn tensor_id<'a>(attrs: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    attrs
        .get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| format!("network tensor requires a nonempty {key} artifact identity"))
}

/// Operator contracts, not model-family branches. DecoderPlan has no independent
/// `weights` table: shapes come from native semantic edges and typed attributes.
fn operation_tensors<'a>(
    graph: &DecoderGraph,
    op: &'a ModelOperation,
    vocabulary: u64,
) -> Result<Vec<(&'a str, Vec<u64>)>, String> {
    let attrs = &op.attributes;
    let mut tensors = Vec::new();
    let matrix = match op.operator {
        ModelOperator::Linear => {
            let region = super::lower_model_linear_region(graph, graph.mode, op)?;
            Some(vec![
                last_dimension(&region.operation.output.shape)?,
                last_dimension(&region.input.shape)?,
            ])
        }
        ModelOperator::OutputHead => {
            let region = super::lower_model_output_head_region(graph, graph.mode, op)?;
            Some(vec![
                last_dimension(&region.output.shape)?,
                last_dimension(&region.input.shape)?,
            ])
        }
        ModelOperator::TokenLookup => {
            if op.inputs.as_slice() != ["input.tokens"]
                || op.output_shape.len() != 3
                || op.output_shape[0] != graph.batch
                || op.output_shape[1] != graph.query_sequence
            {
                return Err("network token table has an invalid semantic shape or source".into());
            }
            Some(vec![vocabulary, last_dimension(&op.output_shape)?])
        }
        _ => None,
    };
    if let Some(shape) = matrix {
        tensors.push((tensor_id(attrs, "weight")?, shape));
        if attrs.get("bias").is_some_and(|bias| !bias.is_null()) {
            tensors.push((
                tensor_id(attrs, "bias")?,
                vec![last_dimension(&op.output_shape)?],
            ));
        }
        return Ok(tensors);
    }
    match op.operator {
        ModelOperator::RmsNorm | ModelOperator::RmsNormGated => {
            if attrs.get("weight").is_none_or(serde_json::Value::is_null) {
                if op.operator != ModelOperator::RmsNorm
                    || attrs["with_scale"] != false
                    || attrs["output_dtype"] != "bfloat16"
                {
                    return Err(
                        "network unweighted norm requires an explicit native contract".into(),
                    );
                }
            } else {
                tensors.push((
                    tensor_id(attrs, "weight")?,
                    vec![last_dimension(&op.output_shape)?],
                ));
            }
        }
        ModelOperator::GatedDeltaDecay => {
            for key in ["a_log", "dt_bias"] {
                tensors.push((
                    tensor_id(attrs, key)?,
                    vec![last_dimension(&op.output_shape)?],
                ));
            }
        }
        ModelOperator::CausalConvolution | ModelOperator::ConvolutionStateUpdate => {
            let channels = attrs["groups"]
                .as_u64()
                .ok_or("network convolution channels are invalid")?;
            let kernel = attrs["kernel_size"]
                .as_u64()
                .ok_or("network convolution kernel is invalid")?;
            tensors.push((tensor_id(attrs, "weight")?, vec![channels, 1, kernel]));
        }
        ModelOperator::Scale if attrs["factor"]["kind"] == "checkpoint_scalar" => {
            let factor = &attrs["factor"];
            if factor["weight_shape"] != serde_json::json!([1]) {
                return Err("network checkpoint scalar has an invalid shape".into());
            }
            tensors.push((tensor_id(factor, "weight")?, vec![1]));
        }
        _ => {
            if ["weight", "bias", "a_log", "dt_bias"]
                .iter()
                .any(|key| attrs.get(key).is_some_and(|value| !value.is_null()))
            {
                return Err(format!(
                    "network operator {} has an unbound tensor declaration",
                    op.id
                ));
            }
        }
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_models::{lower_model_json, DecoderWorkload};
    use serde_json::{json, Value};

    fn inputs(kind: &str) -> (Vec<u8>, Vec<u8>, Value) {
        let plan = lower_model_json(
            br#"{
            "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
            "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
            "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
            "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
        }"#,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 3,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        let (components, roles) = match kind {
            "client" => (
                json!({
                    "linear": {"component":"pllm/cleartext-linear","params":{}},
                    "topology": {"component":"pllm/client-only/v1","params":{}},
                    "kernels": {"component":"pllm/cpu","params":{"threads":1}}
                }),
                vec![("client", "trusted_client")],
            ),
            "two" => (
                json!({
                    "linear": {"component":"pllm/two-online-offset-linear/v1","params":{}},
                    "topology": {"component":"pllm/two-online-offset-workers/v1","params":{}},
                    "kernels": {"component":"pllm/cpu","params":{"threads":1}}
                }),
                vec![
                    ("client", "trusted_client"),
                    ("worker_a", "public_linear_provider"),
                    ("worker_b", "public_linear_provider"),
                ],
            ),
            _ => (
                json!({
                    "linear": {"component":"pllm/masked-linear","params":{}},
                    "inference": {"component":"pllm/inference","params":{}},
                    "preparation": {"component":"pllm/model-aware-corrections","params":{}},
                    "kernels": {"component":"pllm/cpu","params":{"threads":1}}
                }),
                vec![
                    ("client", "trusted_client"),
                    ("inference", "masked_linear_provider"),
                    ("preparation", "trusted_preparation"),
                ],
            ),
        };
        let mut composition = json!({"model":{"source":"model.fixture"},"components":components});
        if kind == "verified" {
            composition["components"]["verification"] = json!({
                "component":"pllm/freivalds-verify/v1","params":{"target_failure_bits":40}
            });
        }
        let mut request = json!({
            "schema":NETWORK_PLACEMENT_SCHEMA,"snapshot_digest":"a".repeat(64),
            "evaluated_at_ms":100,"client_party_id":"client",
            "roles":[],"parties":[],"required_memory_bytes":{},"required_weight_bytes":{}
        });
        for (role, capability) in roles {
            request["roles"]
                .as_array_mut()
                .unwrap()
                .push(json!({"role_id":role,"party_id":role}));
            request["parties"].as_array_mut().unwrap().push(json!({
                "party_id":role,"operator_id":role,"instance_epoch":"b".repeat(64),
                "expires_at_ms":101,"capabilities":[capability],"max_memory_bytes":10000,
                "max_weight_bytes":10000,"max_sessions":1,"active_sessions":0,"allowed_models":["*"]
            }));
            request["required_memory_bytes"][role] = json!(4000);
            request["required_weight_bytes"][role] = json!(3000);
        }
        (
            canonical_bytes(&plan),
            canonical_bytes(&composition),
            request,
        )
    }

    fn validate(inputs: &(Vec<u8>, Vec<u8>, Value)) -> Result<Vec<u8>, String> {
        validate_network_placement(&inputs.0, &inputs.1, &canonical_bytes(&inputs.2))
    }

    #[test]
    fn admits_exact_native_topologies_and_binds_all_commitments() {
        for kind in ["prepared", "verified", "client", "two"] {
            let mut inputs = inputs(kind);
            let bytes = validate(&inputs).unwrap();
            let mut result: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(canonical_bytes(&result), bytes);
            let placement = result
                .as_object_mut()
                .unwrap()
                .remove("placement_digest")
                .unwrap();
            assert_eq!(
                placement,
                canonical_digest(NETWORK_PLACEMENT_DIGEST_DOMAIN, &result).to_string()
            );
            let plan: DecoderPlan = serde_json::from_slice(&inputs.0).unwrap();
            let schedule = lower_decoder_runtime_schedule(&plan, &inputs.1).unwrap();
            assert_eq!(result["schedule_digest"], schedule.digest().to_string());
            assert_eq!(result["model_plan_digest"], plan.digest().to_string());
            assert_eq!(
                result["composition_digest"],
                schedule.composition_digest.to_string()
            );
            inputs.2["roles"].as_array_mut().unwrap().reverse();
            inputs.2["parties"].as_array_mut().unwrap().reverse();
            assert_eq!(validate(&inputs).unwrap(), bytes);
            inputs.2["parties"][0]["instance_epoch"] = json!("c".repeat(64));
            assert_ne!(validate(&inputs).unwrap(), bytes);
        }
    }

    #[test]
    fn rejects_underseparation_capability_substitution_and_role_injection() {
        for kind in ["prepared", "verified", "two"] {
            let original = inputs(kind);
            let mut changed = original.clone();
            changed.2["parties"][2]["operator_id"] = changed.2["parties"][1]["operator_id"].clone();
            assert!(validate(&changed)
                .unwrap_err()
                .contains("separate declared operators"));
            let mut changed = original.clone();
            changed.2["parties"][1]["capabilities"] = json!(["trusted_client"]);
            assert!(validate(&changed)
                .unwrap_err()
                .contains("requires capability"));
            let mut changed = original.clone();
            changed.2["roles"].as_array_mut().unwrap().pop();
            assert!(validate(&changed).is_err());
            let mut changed = original.clone();
            changed.2["roles"][1]["role_id"] = json!("forged");
            assert!(validate(&changed).is_err());
            let mut changed = original;
            changed.2["roles"][0]["party_id"] = json!("inference");
            assert!(validate(&changed).unwrap_err().contains("client_party_id"));
        }
    }

    #[test]
    fn checks_colocated_aggregate_resources_and_sessions_with_overflow_detection() {
        let mut inputs = inputs("prepared");
        inputs.2["roles"][2]["party_id"] = json!("client");
        inputs.2["parties"][0]["capabilities"] = json!(["trusted_client", "trusted_preparation"]);
        // Preparation may share client's operator, but never inference's.
        assert!(validate(&inputs).unwrap_err().contains("capacity exceeded"));
        inputs.2["parties"][0]["max_sessions"] = json!(2);
        assert!(validate(&inputs).is_ok());
        inputs.2["parties"][0]["max_memory_bytes"] = json!(7999);
        assert!(validate(&inputs).is_err());
        inputs.2["parties"][0]["max_memory_bytes"] = json!(u64::MAX);
        inputs.2["required_memory_bytes"]["client"] = json!(u64::MAX);
        assert!(validate(&inputs).unwrap_err().contains("overflow"));
        inputs.2["required_memory_bytes"]["client"] = json!(4000);
        inputs.2["parties"][0]["max_weight_bytes"] = json!(u64::MAX);
        inputs.2["required_weight_bytes"]["client"] = json!(u64::MAX);
        assert!(validate(&inputs).unwrap_err().contains("overflow"));
    }

    #[test]
    fn model_allowlist_uses_exact_native_config_identity() {
        let mut inputs = inputs("client");
        let plan: DecoderPlan = serde_json::from_slice(&inputs.0).unwrap();
        inputs.2["parties"][0]["allowed_models"] = json!([plan.config_digest]);
        assert!(validate(&inputs).is_ok());
        for denied in ["a".repeat(64), "model.fixture".into(), "0".repeat(64)] {
            inputs.2["parties"][0]["allowed_models"] = json!([denied]);
            assert!(validate(&inputs).is_err());
        }
    }

    #[test]
    fn strict_schema_rejects_malformed_offers_and_duplicate_ids() {
        let original = inputs("prepared");
        for (field, value) in [
            ("instance_epoch", json!("0".repeat(64))),
            ("instance_epoch", json!("A".repeat(64))),
            ("instance_epoch", json!(1)),
            ("operator_id", json!("bad operator")),
            ("expires_at_ms", json!(100)),
            ("expires_at_ms", json!(true)),
            ("capabilities", json!(["unknown"])),
            ("capabilities", json!(["trusted_client", "trusted_client"])),
            ("max_memory_bytes", json!(-1)),
            ("max_weight_bytes", json!(1.5)),
            ("max_sessions", json!(u64::MAX)),
            ("active_sessions", json!(2)),
            ("allowed_models", json!([])),
            ("extra", json!(0)),
        ] {
            let mut changed = original.clone();
            changed.2["parties"][0][field] = value;
            assert!(validate(&changed).is_err(), "accepted {field}");
        }
        for field in ["roles", "parties"] {
            let mut changed = original.clone();
            let duplicate = changed.2[field][0].clone();
            changed.2[field].as_array_mut().unwrap().push(duplicate);
            assert!(validate(&changed).is_err());
        }
        for field in ["schema", "snapshot_digest", "client_party_id"] {
            let mut changed = original.clone();
            changed.2[field] = json!("");
            assert!(validate(&changed).is_err());
        }
        let mut changed = original;
        changed.2["role_graph"] = json!({});
        assert!(validate(&changed).is_err());
    }

    #[test]
    fn rejects_duplicate_keys_oversized_documents_and_bounded_collections() {
        let (plan, composition, request) = inputs("client");
        let raw = String::from_utf8(canonical_bytes(&request)).unwrap();
        for invalid in [
            raw.replace(
                "\"evaluated_at_ms\":100",
                "\"evaluated_at_ms\":100,\"evaluated_at_ms\":100",
            ),
            raw.replace("\"client\":4000", "\"client\":4000,\"client\":4000"),
            raw.replace(
                "\"evaluated_at_ms\":100",
                "\"evaluated_at_ms\":18446744073709551616",
            ),
            raw.replace("\"client\":4000", "\"client\":true"),
        ] {
            assert!(validate_network_placement(&plan, &composition, invalid.as_bytes()).is_err());
        }
        assert!(validate_network_placement(
            &plan,
            &composition,
            &vec![b' '; NETWORK_MAX_DOCUMENT_BYTES + 1]
        )
        .is_err());
        for (field, count) in [("roles", 17), ("parties", 65)] {
            let mut changed = request.clone();
            changed[field] = json!(vec![request[field][0].clone(); count]);
            assert!(
                validate_network_placement(&plan, &composition, &canonical_bytes(&changed))
                    .unwrap_err()
                    .contains("record bound")
            );
        }
    }

    #[test]
    fn pending_compositions_and_incomplete_semantics_fail_closed() {
        let (plan, composition, request) = inputs("prepared");
        let mut changed: Value = serde_json::from_slice(&composition).unwrap();
        changed["components"]["nonlinear"] = json!({"component":"pllm/pending/v1","params":{}});
        assert!(validate_network_placement(
            &plan,
            &canonical_bytes(&changed),
            &canonical_bytes(&request)
        )
        .is_err());
        let mut changed: Value = serde_json::from_slice(&plan).unwrap();
        changed["token_feedback"] = json!(false);
        assert!(validate_network_placement(
            &canonical_bytes(&changed),
            &composition,
            &canonical_bytes(&request)
        )
        .is_err());
        changed["token_feedback"] = json!(true);
        changed["prefill"]["operations"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(validate_network_placement(
            &canonical_bytes(&changed),
            &composition,
            &canonical_bytes(&request)
        )
        .is_err());
    }

    fn exact_floor_request(inputs: &mut (Vec<u8>, Vec<u8>, Value)) -> Value {
        let output: Value =
            serde_json::from_slice(&network_placement_requirements(&inputs.0, &inputs.1).unwrap())
                .unwrap();
        inputs.2["required_weight_bytes"] = output["minimum_weight_bytes"].clone();
        inputs.2["required_memory_bytes"] = output["minimum_memory_bytes"].clone();
        for offer in inputs.2["parties"].as_array_mut().unwrap() {
            let role = offer["party_id"].as_str().unwrap().to_owned();
            offer["max_weight_bytes"] = output["minimum_weight_bytes"][&role].clone();
            offer["max_memory_bytes"] = output["minimum_memory_bytes"][&role].clone();
        }
        output
    }

    #[test]
    fn exact_owned_floors_deduplicate_tied_weights_and_both_phases() {
        // Independent shape oracle: two layers each have 576 matrix elements
        // and 16 bias elements. Local norms have 40; tied token/head table 256.
        for (kind, expected) in [
            (
                "prepared",
                json!({"client":296,"inference":1184,"preparation":1184}),
            ),
            (
                "verified",
                json!({"client":296,"inference":1184,"preparation":1184}),
            ),
            ("two", json!({"client":296,"worker_a":1184,"worker_b":1184})),
            ("client", json!({"client":1480})),
        ] {
            let mut inputs = inputs(kind);
            let output = exact_floor_request(&mut inputs);
            assert_eq!(output["minimum_weight_bytes"], expected);
            assert_eq!(output["minimum_memory_bytes"], expected);
            validate(&inputs).unwrap();
            for field in ["required_weight_bytes", "required_memory_bytes"] {
                for (role, floor) in expected.as_object().unwrap() {
                    for value in [0, floor.as_u64().unwrap() - 1] {
                        let mut changed = inputs.clone();
                        changed.2[field][role] = json!(value);
                        assert!(validate(&changed)
                            .unwrap_err()
                            .contains("below native minimum"));
                    }
                }
            }
        }
    }

    #[test]
    fn exact_floors_enforce_colocated_sum_in_both_capacity_axes() {
        let mut inputs = inputs("prepared");
        exact_floor_request(&mut inputs);
        inputs.2["roles"][2]["party_id"] = json!("client");
        inputs.2["parties"][0]["capabilities"] = json!(["trusted_client", "trusted_preparation"]);
        inputs.2["parties"][0]["max_sessions"] = json!(2);
        for field in ["max_memory_bytes", "max_weight_bytes"] {
            inputs.2["parties"][0][field] = json!(1480);
        }
        validate(&inputs).unwrap();
        for field in ["max_memory_bytes", "max_weight_bytes"] {
            let mut changed = inputs.clone();
            changed.2["parties"][0][field] = json!(1479);
            assert!(validate(&changed)
                .unwrap_err()
                .contains("capacity exceeded"));
        }
    }

    #[test]
    fn client_stage_ownership_adds_floor_without_removing_prepared_body() {
        for (component, params, client_bytes) in [
            (
                "pllm/client-owned-prefix-layers/v1",
                json!({"layers":1}),
                888,
            ),
            (
                "pllm/client-owned-linear-roles/v1",
                json!({"roles":["mlp_down"]}),
                552,
            ),
        ] {
            let mut inputs = inputs("prepared");
            let mut composition: Value = serde_json::from_slice(&inputs.1).unwrap();
            composition["components"]["placement"] = json!({"component":component,"params":params});
            inputs.1 = canonical_bytes(&composition);
            let output = exact_floor_request(&mut inputs);
            assert_eq!(
                output["minimum_weight_bytes"],
                json!({
                    "client":client_bytes,"inference":1184,"preparation":1184
                })
            );
            validate(&inputs).unwrap();
        }
    }

    #[test]
    fn untied_and_remote_heads_charge_actual_artifact_ownership() {
        for remote in [false, true] {
            let mut inputs = inputs("prepared");
            let mut plan: DecoderPlan = serde_json::from_slice(&inputs.0).unwrap();
            for graph in [&mut plan.prefill, &mut plan.decode] {
                let head = graph
                    .operations
                    .iter_mut()
                    .find(|op| op.operator == ModelOperator::OutputHead)
                    .unwrap();
                head.attributes["weight"] = json!("lm_head.weight");
                head.attributes["tied"] = json!(false);
            }
            inputs.0 = canonical_bytes(&plan);
            if remote {
                let mut composition: Value = serde_json::from_slice(&inputs.1).unwrap();
                composition["components"]["boundary"] =
                    json!({"component":"pllm/output-head-at-inference/v1","params":{}});
                inputs.1 = canonical_bytes(&composition);
            }
            let output = exact_floor_request(&mut inputs);
            let expected = if remote {
                json!({"client":296,"inference":1440,"preparation":1440})
            } else {
                json!({"client":552,"inference":1184,"preparation":1184})
            };
            assert_eq!(output["minimum_weight_bytes"], expected);
            validate(&inputs).unwrap();
        }
    }

    #[test]
    fn validated_generic_model_shapes_have_owned_floors_without_family_branches() {
        for config in [
            include_bytes!("../../pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json")
                .as_slice(),
            include_bytes!("../../pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json")
                .as_slice(),
            include_bytes!(
                "../../pllm-models/tests/fixtures/Phi-4-mini-instruct-cfbefac-config.json"
            )
            .as_slice(),
        ] {
            let plan = lower_model_json(
                config,
                DecoderWorkload {
                    batch: 1,
                    max_input_tokens: 3,
                    max_new_tokens: 2,
                },
            )
            .unwrap();
            let inputs = inputs("prepared");
            let schedule = lower_decoder_runtime_schedule(&plan, &inputs.1).unwrap();
            let floors = owned_weight_floors(
                &plan,
                &schedule,
                DecoderCompositionKind::MaskedLinear,
                false,
            )
            .unwrap();
            assert!(floors["client"] > 10000);
            assert!(floors["inference"] > 10000);
            assert_eq!(floors["inference"], floors["preparation"]);
        }
    }

    #[test]
    fn zero_or_overflowing_weight_shapes_and_conflicting_artifacts_fail_closed() {
        for shape in [vec![], vec![0, 8], vec![u64::MAX, 2]] {
            assert!(checked_weight_elements(&shape).is_err());
        }
        let inputs = inputs("prepared");
        let mut plan: DecoderPlan = serde_json::from_slice(&inputs.0).unwrap();
        let schedule = lower_decoder_runtime_schedule(&plan, &inputs.1).unwrap();
        // This tests the cross-use identity guard separately from plan validation.
        let norm = plan
            .decode
            .operations
            .iter_mut()
            .find(|op| op.operator == ModelOperator::RmsNorm)
            .unwrap();
        norm.attributes["weight"] = json!("model.embed_tokens.weight");
        assert!(owned_weight_floors(
            &plan,
            &schedule,
            DecoderCompositionKind::MaskedLinear,
            false
        )
        .unwrap_err()
        .contains("inconsistent shapes"));
        // Public admission also rejects a malformed native matrix shape.
        let mut malformed: Value = serde_json::from_slice(&inputs.0).unwrap();
        let linear = malformed["prefill"]["operations"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|op| op["operator"] == "linear")
            .unwrap();
        linear["output_shape"] = json!([1, 3, 0]);
        assert!(validate_network_placement(
            &canonical_bytes(&malformed),
            &inputs.1,
            &canonical_bytes(&inputs.2)
        )
        .is_err());
    }

    #[test]
    fn native_weight_products_and_owned_sums_cannot_overflow() {
        for (width, expected) in [
            (1_u64 << 32, "weight shape byte product overflow"),
            (1_u64 << 31, "owned weight byte sum overflow"),
        ] {
            let config = json!({
                "model_type":"qwen2","hidden_size":width,"intermediate_size":width,
                "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
                "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
            });
            // Lower metadata only; no tensor allocations or model downloads.
            let plan = lower_model_json(
                &canonical_bytes(&config),
                DecoderWorkload {
                    batch: 1,
                    max_input_tokens: 3,
                    max_new_tokens: 2,
                },
            )
            .unwrap();
            let mut inputs = inputs("prepared");
            inputs.0 = canonical_bytes(&plan);
            assert!(validate(&inputs).unwrap_err().contains(expected));
        }
    }

    #[test]
    fn assignment_free_requirements_match_validation_and_native_topology() {
        for kind in ["prepared", "verified", "client", "two"] {
            let inputs = inputs(kind);
            let bytes = network_placement_requirements(&inputs.0, &inputs.1).unwrap();
            let requirements: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(canonical_bytes(&requirements), bytes);
            assert_eq!(
                requirements["schema"],
                NETWORK_PLACEMENT_REQUIREMENTS_SCHEMA
            );
            assert_eq!(requirements.as_object().unwrap().len(), 8);
            let validated: Value = serde_json::from_slice(&validate(&inputs).unwrap()).unwrap();
            for field in [
                "minimum_weight_bytes",
                "minimum_memory_bytes",
                "schedule_digest",
                "model_plan_digest",
                "composition_digest",
            ] {
                assert_eq!(
                    requirements[field], validated[field],
                    "{kind} drifted at {field}"
                );
            }
            let expected_roles = match kind {
                "client" => json!([{"role_id":"client","capability":"trusted_client"}]),
                "two" => json!([
                    {"role_id":"client","capability":"trusted_client"},
                    {"role_id":"worker_a","capability":"public_linear_provider"},
                    {"role_id":"worker_b","capability":"public_linear_provider"}
                ]),
                _ => json!([
                    {"role_id":"client","capability":"trusted_client"},
                    {"role_id":"inference","capability":"masked_linear_provider"},
                    {"role_id":"preparation","capability":"trusted_preparation"}
                ]),
            };
            assert_eq!(requirements["roles"], expected_roles);
            let separation = match kind {
                "client" => json!([]),
                "two" => json!([["worker_a", "worker_b"]]),
                _ => json!([["inference", "preparation"]]),
            };
            assert_eq!(requirements["separate_operators"], separation);
        }
    }

    #[test]
    fn assignment_free_admission_rejects_the_same_bad_plans_and_compositions() {
        let inputs = inputs("prepared");
        for (field, value) in [
            ("schema_version", json!("pllm.decoder_plan.pending")),
            ("token_feedback", json!(false)),
            ("config_digest", json!("0".repeat(64))),
        ] {
            let mut plan: Value = serde_json::from_slice(&inputs.0).unwrap();
            plan[field] = value;
            let plan = canonical_bytes(&plan);
            assert_eq!(
                network_placement_requirements(&plan, &inputs.1).unwrap_err(),
                validate_network_placement(&plan, &inputs.1, &canonical_bytes(&inputs.2))
                    .unwrap_err()
            );
        }
        for (slot, component) in [
            ("nonlinear", "pllm/pending/v1"),
            ("topology", "pllm/two-online-workers/v1"),
        ] {
            let mut composition: Value = serde_json::from_slice(&inputs.1).unwrap();
            composition["components"][slot] = json!({"component":component,"params":{}});
            let composition = canonical_bytes(&composition);
            assert_eq!(
                network_placement_requirements(&inputs.0, &composition).unwrap_err(),
                validate_network_placement(&inputs.0, &composition, &canonical_bytes(&inputs.2))
                    .unwrap_err()
            );
        }
        let mut composition: Value = serde_json::from_slice(&inputs.1).unwrap();
        composition["model"]["kind"] = json!("unknown");
        assert!(network_placement_requirements(&inputs.0, &canonical_bytes(&composition)).is_err());
        for plan in [b"{}".as_slice(), b"invalid", b""] {
            assert!(network_placement_requirements(plan, &inputs.1).is_err());
        }
        assert!(network_placement_requirements(&inputs.0, b"{}").is_err());
    }
}
