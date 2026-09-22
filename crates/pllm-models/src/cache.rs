//! Cache-policy graph transformations.

use crate::{
    AppliedMethod, DecoderGraph, DecoderPlan, ModelError, ModelOperation, ModelOperator, StateKind,
    StateTensor,
};
use pllm_types::canonical_digest;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

const COMPONENT: &str = "pllm/kv-cache-eviction";
const IMPLEMENTATION: &str = "pllm/importance-kv-cache-eviction/v1";
const METHOD_ID: &str = "R23";
const METHOD_PREFIX: &str = "method.kv_cache_eviction.";
const POLICY_DIGEST_NAMESPACE: &str = "pllm.pass.kv_cache_eviction.policy.v1";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Ratio {
    pub numerator: u64,
    pub denominator: u64,
}

impl Ratio {
    fn validate(&self, name: &str) -> Result<(), MpcacheError> {
        if self.denominator == 0 || self.numerator == 0 || self.numerator > self.denominator {
            return Err(MpcacheError::InvalidPolicy(format!(
                "{name} must be in (0, 1]"
            )));
        }
        Ok(())
    }

    fn ceil(&self, value: u64) -> u64 {
        let product = u128::from(value) * u128::from(self.numerator);
        product.div_ceil(u128::from(self.denominator)) as u64
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MpcachePolicy {
    pub observation_window: Ratio,
    pub static_keep: Ratio,
    pub dynamic_keep: Ratio,
    pub alpha: Ratio,
    pub cluster_sizes: Vec<u64>,
    pub share_adjacent_layers: bool,
}

impl MpcachePolicy {
    pub fn r23_reference_policy() -> Self {
        Self {
            observation_window: Ratio {
                numerator: 1,
                denominator: 5,
            },
            static_keep: Ratio {
                numerator: 3,
                denominator: 10,
            },
            dynamic_keep: Ratio {
                numerator: 1,
                denominator: 4,
            },
            alpha: Ratio {
                numerator: 3,
                denominator: 5,
            },
            cluster_sizes: vec![32, 16],
            share_adjacent_layers: true,
        }
    }

    fn validate(&self) -> Result<(), MpcacheError> {
        self.observation_window.validate("observation_window")?;
        self.static_keep.validate("static_keep")?;
        self.dynamic_keep.validate("dynamic_keep")?;
        self.alpha.validate("alpha")?;
        if self.cluster_sizes.is_empty() || self.cluster_sizes.iter().any(|size| *size == 0) {
            return Err(MpcacheError::InvalidPolicy(
                "cluster_sizes must be nonempty and positive".to_owned(),
            ));
        }
        if self
            .cluster_sizes
            .windows(2)
            .any(|pair| pair[0] <= pair[1] || pair[0] % pair[1] != 0)
        {
            return Err(MpcacheError::InvalidPolicy(
                "cluster_sizes must be a strictly descending divisible hierarchy".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum MpcacheError {
    InvalidPolicy(String),
    InvalidPlan(String),
}

impl fmt::Display for MpcacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy(message) => write!(f, "invalid MPCache policy: {message}"),
            Self::InvalidPlan(message) => write!(f, "invalid MPCache plan: {message}"),
        }
    }
}

impl Error for MpcacheError {}

impl From<ModelError> for MpcacheError {
    fn from(error: ModelError) -> Self {
        Self::InvalidPlan(error.to_string())
    }
}

pub fn optimize(base: &DecoderPlan, policy: MpcachePolicy) -> Result<DecoderPlan, MpcacheError> {
    policy.validate()?;
    base.validate()?;
    if !matches!(base.model_family.as_str(), "qwen2" | "qwen3") {
        return Err(MpcacheError::InvalidPlan(
            "MPCache requires a dense Qwen fixed-shape cache plan".into(),
        ));
    }
    if base
        .transformations
        .iter()
        .any(|method| method.component == COMPONENT)
    {
        return Err(MpcacheError::InvalidPlan(
            "MPCache is already applied".into(),
        ));
    }
    validate_compatibility(base)?;
    let input_digest = base.digest();
    let configuration_digest = canonical_digest(POLICY_DIGEST_NAMESPACE, &policy);
    let static_keep = policy.static_keep.ceil(base.prefill.maximum_key_sequence);
    let generated = base
        .decode
        .maximum_key_sequence
        .checked_sub(base.prefill.maximum_key_sequence)
        .ok_or_else(|| MpcacheError::InvalidPlan("decode bound precedes prefill bound".into()))?;
    let candidate_bound = static_keep
        .checked_add(generated)
        .ok_or_else(|| MpcacheError::InvalidPlan("candidate bound overflowed".into()))?;
    let mut decoder = base.clone();
    decoder.prefill = transform_prefill(&base.prefill, &policy, static_keep)?;
    decoder.decode = transform_decode(&base.decode, &policy, static_keep, candidate_bound)?;
    decoder.transformations.push(AppliedMethod {
        component: COMPONENT.to_owned(),
        implementation: IMPLEMENTATION.to_owned(),
        method_id: METHOD_ID.to_owned(),
        input_digest,
        configuration_digest,
    });
    decoder.validate()?;
    Ok(decoder)
}

fn validate_compatibility(plan: &DecoderPlan) -> Result<(), MpcacheError> {
    let prefill_softmax = softmax_layers(&plan.prefill, "prefill")?;
    let decode_softmax = softmax_layers(&plan.decode, "decode")?;
    if prefill_softmax.is_empty() {
        return Err(MpcacheError::InvalidPlan(
            "incompatible cache topology: plan has no Softmax attention layers".into(),
        ));
    }
    let prefill_layers = prefill_softmax.keys().copied().collect::<BTreeSet<_>>();
    let decode_layers = decode_softmax.keys().copied().collect::<BTreeSet<_>>();
    if prefill_layers != decode_layers {
        return Err(MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: prefill Softmax layers {prefill_layers:?} do not match decode layers {decode_layers:?}"
        )));
    }

    for (layer, prefill_probability) in prefill_softmax {
        let decode_probability = decode_softmax.get(&layer).ok_or_else(|| {
            MpcacheError::InvalidPlan(format!(
                "incompatible cache topology: decode layer {layer} has no Softmax"
            ))
        })?;
        let prefill_key = state_operation(&plan.prefill, layer, StateKind::Key)?;
        let prefill_value = state_operation(&plan.prefill, layer, StateKind::Value)?;
        let decode_key = state_operation(&plan.decode, layer, StateKind::Key)?;
        let decode_value = state_operation(&plan.decode, layer, StateKind::Value)?;
        validate_producer_pair("prefill", layer, prefill_key, prefill_value)?;
        validate_producer_pair("decode", layer, decode_key, decode_value)?;

        let prefill_key_output =
            validate_state_output(&plan.prefill, "prefill", layer, StateKind::Key, prefill_key)?;
        let prefill_value_output = validate_state_output(
            &plan.prefill,
            "prefill",
            layer,
            StateKind::Value,
            prefill_value,
        )?;
        let decode_key_output =
            validate_state_output(&plan.decode, "decode", layer, StateKind::Key, decode_key)?;
        let decode_value_output = validate_state_output(
            &plan.decode,
            "decode",
            layer,
            StateKind::Value,
            decode_value,
        )?;
        let decode_key_input = state_tensor(
            &plan.decode.state_inputs,
            "decode input",
            layer,
            StateKind::Key,
        )?;
        let decode_value_input = state_tensor(
            &plan.decode.state_inputs,
            "decode input",
            layer,
            StateKind::Value,
        )?;

        for (kind, prefill_output, decode_input, decode_output, decode_producer) in [
            (
                StateKind::Key,
                prefill_key_output,
                decode_key_input,
                decode_key_output,
                decode_key,
            ),
            (
                StateKind::Value,
                prefill_value_output,
                decode_value_input,
                decode_value_output,
                decode_value,
            ),
        ] {
            if decode_input.shape != decode_output.shape
                || decode_input.maximum_sequence != decode_output.maximum_sequence
                || !matching_kv_layout(prefill_output, decode_input)
                || decode_producer.inputs.first() != Some(&decode_input.id)
                || decode_producer
                    .attributes
                    .get("state")
                    .and_then(serde_json::Value::as_str)
                    != Some(decode_input.id.as_str())
            {
                return Err(MpcacheError::InvalidPlan(format!(
                    "incompatible cache topology: layer {layer} {kind:?} prefill output and decode input/output do not match"
                )));
            }
        }

        validate_attention_inputs(
            &plan.prefill,
            "prefill",
            layer,
            prefill_probability,
            prefill_key,
            prefill_value,
        )?;
        validate_attention_inputs(
            &plan.decode,
            "decode",
            layer,
            decode_probability,
            decode_key,
            decode_value,
        )?;
    }
    Ok(())
}

fn softmax_layers<'a>(
    graph: &'a DecoderGraph,
    phase: &str,
) -> Result<BTreeMap<u64, &'a ModelOperation>, MpcacheError> {
    let mut layers = BTreeMap::new();
    for operation in graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::Softmax)
    {
        let layer = layer_number(operation)?;
        if layers.insert(layer, operation).is_some() {
            return Err(MpcacheError::InvalidPlan(format!(
                "incompatible cache topology: {phase} layer {layer} has multiple Softmax operations"
            )));
        }
    }
    Ok(layers)
}

fn validate_producer_pair(
    phase: &str,
    layer: u64,
    key: &ModelOperation,
    value: &ModelOperation,
) -> Result<(), MpcacheError> {
    if key.output_shape.len() != 4
        || key.output_shape != value.output_shape
        || key.layer != Some(layer)
        || value.layer != Some(layer)
    {
        return Err(MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: {phase} layer {layer} Key and Value producers must be layer-local matching rank-four caches"
        )));
    }
    Ok(())
}

fn validate_state_output<'a>(
    graph: &'a DecoderGraph,
    phase: &str,
    layer: u64,
    kind: StateKind,
    producer: &ModelOperation,
) -> Result<&'a StateTensor, MpcacheError> {
    let output = state_tensor(
        &graph.state_outputs,
        &format!("{phase} output"),
        layer,
        kind,
    )?;
    if output.id != producer.id || output.shape != producer.output_shape {
        return Err(MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: {phase} layer {layer} {kind:?} state output {} is not its layer-local cache producer {}",
            output.id, producer.id
        )));
    }
    Ok(output)
}

fn state_tensor<'a>(
    states: &'a [StateTensor],
    location: &str,
    layer: u64,
    kind: StateKind,
) -> Result<&'a StateTensor, MpcacheError> {
    let mut matching = states
        .iter()
        .filter(|state| state.layer == Some(layer) && state.kind == kind);
    let state = matching.next().ok_or_else(|| {
        MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: layer {layer} has no {kind:?} {location}"
        ))
    })?;
    if matching.next().is_some() {
        return Err(MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: layer {layer} has multiple {kind:?} {location}s"
        )));
    }
    Ok(state)
}

fn matching_kv_layout(left: &StateTensor, right: &StateTensor) -> bool {
    left.shape.len() == 4
        && right.shape.len() == 4
        && left.shape[0] == right.shape[0]
        && left.shape[1] == right.shape[1]
        && left.shape[3] == right.shape[3]
}

fn validate_attention_inputs(
    graph: &DecoderGraph,
    phase: &str,
    layer: u64,
    probability: &ModelOperation,
    key: &ModelOperation,
    value: &ModelOperation,
) -> Result<(), MpcacheError> {
    let scores = unique_layer_operation(graph, phase, layer, ModelOperator::AttentionScores)?;
    let values = unique_layer_operation(graph, phase, layer, ModelOperator::AttentionValues)?;
    if !consumes_cache(graph, scores.inputs.get(1), key)
        || values.inputs.first() != Some(&probability.id)
        || !consumes_cache(graph, values.inputs.get(1), value)
    {
        return Err(MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: {phase} layer {layer} attention does not consume its layer-local Key and Value producers"
        )));
    }
    Ok(())
}

fn consumes_cache(graph: &DecoderGraph, input: Option<&String>, producer: &ModelOperation) -> bool {
    input.is_some_and(|input| {
        input == &producer.id
            || graph.operations.iter().any(|operation| {
                operation.id == *input
                    && operation.operator == ModelOperator::CacheSuffix
                    && operation.inputs.first() == Some(&producer.id)
            })
    })
}

fn unique_layer_operation<'a>(
    graph: &'a DecoderGraph,
    phase: &str,
    layer: u64,
    operator: ModelOperator,
) -> Result<&'a ModelOperation, MpcacheError> {
    let mut matching = graph
        .operations
        .iter()
        .filter(|operation| operation.layer == Some(layer) && operation.operator == operator);
    let operation = matching.next().ok_or_else(|| {
        MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: {phase} layer {layer} has no {operator:?} operation"
        ))
    })?;
    if matching.next().is_some() {
        return Err(MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: {phase} layer {layer} has multiple {operator:?} operations"
        )));
    }
    Ok(operation)
}

fn transform_prefill(
    graph: &DecoderGraph,
    policy: &MpcachePolicy,
    static_keep: u64,
) -> Result<DecoderGraph, MpcacheError> {
    let mut transformed = graph.clone();
    transformed.operations.clear();
    for operation in &graph.operations {
        transformed.operations.push(operation.clone());
        if operation.operator != ModelOperator::Softmax {
            continue;
        }
        let layer = layer_number(operation)?;
        let prefix = format!("{METHOD_PREFIX}layer.{layer}");
        let importance = format!("{prefix}.static_importance");
        transformed.operations.push(op(
            &importance,
            ModelOperator::AttentionImportance,
            layer,
            vec![operation.id.clone(), "input.sequence_lengths".to_owned()],
            vec![graph.batch, graph.maximum_key_sequence],
            json!({
                "observation_window": policy.observation_window,
                "reduction": "mean_heads_sum_last_queries",
                "valid_lengths_input": "input.sequence_lengths",
                "policy": policy,
            }),
        ));
        let selection_id = format!("{prefix}.static_selection");
        let mut selection = op(
            &selection_id,
            ModelOperator::SecureTopK,
            layer,
            vec![importance],
            vec![graph.batch, static_keep],
            json!({
                "keep": static_keep,
                "keep_ratio": policy.static_keep,
                "index_domain": "global_token_positions",
                "output_order": "ascending",
                "padding": "masked_sentinel",
            }),
        );
        selection.state_kind = Some(StateKind::CacheIndices);
        transformed.operations.push(selection);
        transformed.state_outputs.push(StateTensor {
            id: selection_id,
            layer: Some(layer),
            kind: StateKind::CacheIndices,
            shape: vec![graph.batch, static_keep],
            maximum_sequence: static_keep,
        });
    }
    Ok(transformed)
}

fn transform_decode(
    graph: &DecoderGraph,
    policy: &MpcachePolicy,
    static_keep: u64,
    candidate_bound: u64,
) -> Result<DecoderGraph, MpcacheError> {
    let mut transformed = graph.clone();
    transformed.operations.clear();
    let mut selected = BTreeMap::<u64, (String, u64)>::new();
    for source in &graph.operations {
        let mut operation = source.clone();
        match operation.operator {
            ModelOperator::AttentionScores => {
                let layer = layer_number(&operation)?;
                let prefix = format!("{METHOD_PREFIX}layer.{layer}");
                let key_view = cache_view(graph, layer, StateKind::Key)?;
                let value_view = cache_view(graph, layer, StateKind::Value)?;
                let query = operation.inputs.first().cloned().ok_or_else(|| {
                    MpcacheError::InvalidPlan(format!(
                        "attention scores operation {} has no query input",
                        operation.id
                    ))
                })?;
                let group_size = operation
                    .attributes
                    .get("group_size")
                    .cloned()
                    .unwrap_or(Value::Null);
                let heads = *operation.output_shape.get(1).ok_or_else(|| {
                    MpcacheError::InvalidPlan(format!(
                        "attention scores operation {} must have rank four",
                        operation.id
                    ))
                })?;
                let query_sequence = *operation.output_shape.get(2).ok_or_else(|| {
                    MpcacheError::InvalidPlan(format!(
                        "attention scores operation {} must have rank four",
                        operation.id
                    ))
                })?;
                let head_dim = *key_view.output_shape.get(3).ok_or_else(|| {
                    MpcacheError::InvalidPlan(format!(
                        "layer {layer} Key cache view must have rank four"
                    ))
                })?;

                let state_id = format!("state.layer.{layer}.cache_indices");
                transformed.state_inputs.push(StateTensor {
                    id: state_id.clone(),
                    layer: Some(layer),
                    kind: StateKind::CacheIndices,
                    shape: vec![graph.batch, static_keep],
                    maximum_sequence: static_keep,
                });
                let carry_id = format!("{prefix}.static_selection_state");
                let mut carry = op(
                    &carry_id,
                    ModelOperator::SharedIndices,
                    layer,
                    vec![state_id],
                    vec![graph.batch, static_keep],
                    json!({
                        "mode": "carry",
                        "keep": static_keep,
                        "index_domain": "global_token_positions",
                    }),
                );
                carry.state_kind = Some(StateKind::CacheIndices);
                transformed.operations.push(carry);
                transformed.state_outputs.push(StateTensor {
                    id: carry_id.clone(),
                    layer: Some(layer),
                    kind: StateKind::CacheIndices,
                    shape: vec![graph.batch, static_keep],
                    maximum_sequence: static_keep,
                });

                let share = policy.share_adjacent_layers && layer >= 3 && layer % 2 == 1;
                let (positions_id, selected_tokens) = if share {
                    let previous = layer - 1;
                    let (previous_id, previous_tokens) =
                        selected.get(&previous).cloned().ok_or_else(|| {
                            MpcacheError::InvalidPlan(format!(
                                "layer {layer} cannot share indices: layer {previous} has no previous selection"
                            ))
                        })?;
                    let positions_id = format!("{prefix}.selected_positions");
                    transformed.operations.push(op(
                        &positions_id,
                        ModelOperator::SharedIndices,
                        layer,
                        vec![previous_id],
                        vec![graph.batch, heads, query_sequence, previous_tokens],
                        json!({
                            "mode": "cross_layer",
                            "index_domain": "global_token_positions",
                        }),
                    ));
                    (positions_id, previous_tokens)
                } else {
                    let active_id = format!("{prefix}.active_indices");
                    transformed.operations.push(op(
                        &active_id,
                        ModelOperator::CacheActiveIndices,
                        layer,
                        vec![
                            carry_id.clone(),
                            "input.positions".to_owned(),
                            "input.sequence_lengths".to_owned(),
                        ],
                        vec![graph.batch, candidate_bound],
                        json!({
                            "static_keep": static_keep,
                            "prefill_maximum_sequence": graph.maximum_key_sequence
                                - (candidate_bound - static_keep),
                            "candidate_bound": candidate_bound,
                            "semantics": "static_plus_generated_global_positions",
                        }),
                    ));
                    append_hierarchy(HierarchyPlan {
                        transformed: &mut transformed,
                        layer,
                        prefix: &prefix,
                        active_id: &active_id,
                        key_view_id: &key_view.id,
                        kv_heads: key_view.output_shape[1],
                        head_dim,
                        batch: graph.batch,
                        heads,
                        query_sequence,
                        candidate_bound,
                        policy,
                        query: &query,
                        group_size: &group_size,
                    })?
                };

                let key_gather = format!("{prefix}.dynamic_key_gather");
                let mut key = op(
                    &key_gather,
                    ModelOperator::SecureGather,
                    layer,
                    vec![key_view.id.clone(), positions_id.clone()],
                    vec![
                        graph.batch,
                        heads,
                        query_sequence,
                        selected_tokens,
                        head_dim,
                    ],
                    json!({
                        "index_axis": 2,
                        "index_unit_size": 1,
                        "output_semantics": "per_query_head_cache_window",
                        "group_size": group_size,
                    }),
                );
                key.state_kind = Some(StateKind::Key);
                transformed.operations.push(key);
                let value_gather = format!("{prefix}.dynamic_value_gather");
                let mut value = op(
                    &value_gather,
                    ModelOperator::SecureGather,
                    layer,
                    vec![value_view.id.clone(), positions_id.clone()],
                    vec![
                        graph.batch,
                        heads,
                        query_sequence,
                        selected_tokens,
                        head_dim,
                    ],
                    json!({
                        "index_axis": 2,
                        "index_unit_size": 1,
                        "output_semantics": "per_query_head_cache_window",
                        "group_size": group_size,
                    }),
                );
                value.state_kind = Some(StateKind::Value);
                transformed.operations.push(value);
                selected.insert(layer, (positions_id, selected_tokens));
                operation.inputs[1] = key_gather;
                operation.output_shape[3] = selected_tokens;
                transformed.operations.push(operation);
            }
            ModelOperator::AttentionScale | ModelOperator::Softmax => {
                let layer = layer_number(&operation)?;
                if let Some((_, tokens)) = selected.get(&layer) {
                    if let Some(last) = operation.output_shape.last_mut() {
                        *last = *tokens;
                    }
                }
                transformed.operations.push(operation);
            }
            ModelOperator::CausalMask => {
                let layer = layer_number(&operation)?;
                if let Some((positions, tokens)) = selected.get(&layer) {
                    let selected_positions = positions.clone();
                    operation.inputs = vec![
                        operation.inputs[0].clone(),
                        "input.positions".to_owned(),
                        selected_positions.clone(),
                    ];
                    operation.attributes["cache_positions_input"] = json!(selected_positions);
                    operation.attributes["selection_semantics"] = json!("global_token_positions");
                    if let Some(last) = operation.output_shape.last_mut() {
                        *last = *tokens;
                    }
                }
                transformed.operations.push(operation);
            }
            ModelOperator::AttentionValues => {
                let layer = layer_number(&operation)?;
                if let Some(input) = operation.inputs.get_mut(1) {
                    *input = format!("{METHOD_PREFIX}layer.{layer}.dynamic_value_gather");
                }
                transformed.operations.push(operation);
            }
            _ => transformed.operations.push(operation),
        }
    }
    Ok(transformed)
}

struct HierarchyPlan<'a> {
    transformed: &'a mut DecoderGraph,
    layer: u64,
    prefix: &'a str,
    active_id: &'a str,
    key_view_id: &'a str,
    kv_heads: u64,
    head_dim: u64,
    batch: u64,
    heads: u64,
    query_sequence: u64,
    candidate_bound: u64,
    policy: &'a MpcachePolicy,
    query: &'a str,
    group_size: &'a Value,
}

struct HierarchyLevel {
    cluster_size: u64,
    clusters: u64,
    keep: u64,
    retained: u64,
}

fn hierarchy_levels(candidate_bound: u64, policy: &MpcachePolicy) -> Vec<HierarchyLevel> {
    let mut levels = Vec::with_capacity(policy.cluster_sizes.len());
    let mut candidate_tokens = candidate_bound;
    for (index, cluster_size) in policy.cluster_sizes.iter().copied().enumerate() {
        let clusters = candidate_tokens.div_ceil(cluster_size);
        let last = index + 1 == policy.cluster_sizes.len();
        let keep = if last {
            let wanted = (u128::from(candidate_bound) * u128::from(policy.dynamic_keep.numerator))
                .div_ceil(u128::from(policy.dynamic_keep.denominator) * u128::from(cluster_size));
            wanted.min(u128::from(clusters)) as u64
        } else if u128::from(policy.dynamic_keep.numerator) * 2
            < u128::from(policy.dynamic_keep.denominator)
        {
            clusters.div_ceil(2)
        } else {
            clusters
        };
        let retained = bounded_product(keep, cluster_size, candidate_tokens);
        levels.push(HierarchyLevel {
            cluster_size,
            clusters,
            keep,
            retained,
        });
        candidate_tokens = retained;
    }
    levels
}

fn append_hierarchy(plan: HierarchyPlan<'_>) -> Result<(String, u64), MpcacheError> {
    let HierarchyPlan {
        transformed,
        layer,
        prefix,
        active_id,
        key_view_id,
        kv_heads,
        head_dim,
        batch,
        heads,
        query_sequence,
        candidate_bound,
        policy,
        query,
        group_size,
    } = plan;
    let levels = hierarchy_levels(candidate_bound, policy);
    let mut position_candidates = active_id.to_owned();
    let mut key_candidate = key_view_id.to_owned();
    let mut candidate_axis = 1_u64;
    let mut result = None;
    for (level, spec) in levels.iter().enumerate() {
        let last = level + 1 == levels.len();
        let hierarchy = format!("{prefix}.hierarchy.{level}");
        let (bounds_shape, similarity_group_size) = if level == 0 {
            (
                vec![batch, kv_heads, spec.clusters, head_dim, 2],
                group_size.clone(),
            )
        } else {
            (
                vec![batch, heads, query_sequence, spec.clusters, head_dim, 2],
                json!(1),
            )
        };
        let bounds = format!("{hierarchy}.cluster_bounds");
        transformed.operations.push(op(
            &bounds,
            ModelOperator::ClusterBounds,
            layer,
            vec![key_candidate.clone()],
            bounds_shape,
            json!({"cluster_size": spec.cluster_size, "level": level}),
        ));
        let similarity = format!("{hierarchy}.cluster_similarity");
        transformed.operations.push(op(
            &similarity,
            ModelOperator::ClusterSimilarity,
            layer,
            vec![query.to_owned(), bounds],
            vec![batch, heads, query_sequence, spec.clusters],
            json!({"alpha": policy.alpha, "level": level, "group_size": similarity_group_size}),
        ));
        let selection = format!("{hierarchy}.selection");
        transformed.operations.push(op(
            &selection,
            ModelOperator::SecureTopK,
            layer,
            vec![similarity],
            vec![batch, heads, query_sequence, spec.keep],
            json!({
                "keep": spec.keep,
                "index_domain": "cluster_indices",
                "output_order": "ascending",
            }),
        ));
        let positions = if last {
            format!("{prefix}.selected_positions")
        } else {
            format!("{hierarchy}.selected_positions")
        };
        transformed.operations.push(op(
            &positions,
            ModelOperator::SecureGather,
            layer,
            vec![position_candidates.clone(), selection],
            vec![batch, heads, query_sequence, spec.retained],
            json!({
                "index_axis": candidate_axis,
                "index_unit_size": spec.cluster_size,
                "output_semantics": "global_token_positions",
            }),
        ));
        if !last {
            let key_gather = format!("{hierarchy}.candidate_key_gather");
            transformed.operations.push(op(
                &key_gather,
                ModelOperator::SecureGather,
                layer,
                vec![key_view_id.to_owned(), positions.clone()],
                vec![batch, heads, query_sequence, spec.retained, head_dim],
                json!({
                    "index_axis": 2,
                    "index_unit_size": 1,
                    "output_semantics": "key_candidates_at_global_positions",
                }),
            ));
            key_candidate = key_gather;
        }
        position_candidates = positions.clone();
        candidate_axis = 3;
        if last {
            result = Some((positions, spec.retained));
        }
    }
    result.ok_or_else(|| {
        MpcacheError::InvalidPolicy("cluster_sizes must contain a final hierarchy level".into())
    })
}

fn bounded_product(left: u64, right: u64, bound: u64) -> u64 {
    (u128::from(left) * u128::from(right)).min(u128::from(bound)) as u64
}

fn layer_number(operation: &ModelOperation) -> Result<u64, MpcacheError> {
    operation.layer.ok_or_else(|| {
        MpcacheError::InvalidPlan(format!("operation {} has no layer metadata", operation.id))
    })
}

fn state_operation(
    graph: &DecoderGraph,
    layer: u64,
    kind: StateKind,
) -> Result<&ModelOperation, MpcacheError> {
    let mut matching = graph.operations.iter().filter(|operation| {
        operation.layer == Some(layer)
            && operation.operator == ModelOperator::KvCacheAppend
            && operation.state_kind == Some(kind)
    });
    let operation = matching.next().ok_or_else(|| {
        MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: layer {layer} has no layer-local {kind:?} cache producer"
        ))
    })?;
    if matching.next().is_some() {
        return Err(MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: layer {layer} has multiple layer-local {kind:?} cache producers"
        )));
    }
    Ok(operation)
}

fn cache_view(
    graph: &DecoderGraph,
    layer: u64,
    kind: StateKind,
) -> Result<&ModelOperation, MpcacheError> {
    let mut matching = graph.operations.iter().filter(|operation| {
        operation.layer == Some(layer)
            && operation.operator == ModelOperator::CacheSuffix
            && operation.state_kind == Some(kind)
    });
    let operation = matching.next().ok_or_else(|| {
        MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: layer {layer} has no layer-local {kind:?} cache view"
        ))
    })?;
    if matching.next().is_some() {
        return Err(MpcacheError::InvalidPlan(format!(
            "incompatible cache topology: layer {layer} has multiple layer-local {kind:?} cache views"
        )));
    }
    Ok(operation)
}

fn op(
    id: &str,
    operator: ModelOperator,
    layer: u64,
    inputs: Vec<String>,
    output_shape: Vec<u64>,
    attributes: serde_json::Value,
) -> ModelOperation {
    ModelOperation {
        id: id.to_owned(),
        operator,
        layer: Some(layer),
        state_kind: None,
        inputs,
        output_shape,
        attributes,
    }
}

fn incomplete(message: impl Into<String>) -> ModelError {
    ModelError::Incomplete(message.into())
}

fn method_operation<'a>(
    graph: &'a DecoderGraph,
    id: &str,
) -> Result<&'a ModelOperation, ModelError> {
    graph
        .operations
        .iter()
        .find(|operation| operation.id == id)
        .ok_or_else(|| incomplete(format!("MPCache operation {id} is missing")))
}

fn unique_state(
    states: &[StateTensor],
    layer: u64,
    kind: StateKind,
) -> Result<&StateTensor, ModelError> {
    let mut matching = states
        .iter()
        .filter(|state| state.layer == Some(layer) && state.kind == kind);
    let state = matching.next().ok_or_else(|| {
        incomplete(format!(
            "MPCache layer {layer} has no {kind:?} state tensor"
        ))
    })?;
    if matching.next().is_some() {
        return Err(incomplete(format!(
            "MPCache layer {layer} has multiple {kind:?} state tensors"
        )));
    }
    Ok(state)
}

fn unique_operation(
    graph: &DecoderGraph,
    layer: u64,
    operator: ModelOperator,
) -> Result<&ModelOperation, ModelError> {
    let mut matching = graph
        .operations
        .iter()
        .filter(|operation| operation.layer == Some(layer) && operation.operator == operator);
    let operation = matching.next().ok_or_else(|| {
        incomplete(format!(
            "MPCache layer {layer} has no {operator:?} operation"
        ))
    })?;
    if matching.next().is_some() {
        return Err(incomplete(format!(
            "MPCache layer {layer} has multiple {operator:?} operations"
        )));
    }
    Ok(operation)
}

fn unique_state_view(
    graph: &DecoderGraph,
    layer: u64,
    kind: StateKind,
) -> Result<&ModelOperation, ModelError> {
    let mut matching = graph.operations.iter().filter(|operation| {
        operation.layer == Some(layer)
            && operation.operator == ModelOperator::CacheSuffix
            && operation.state_kind == Some(kind)
    });
    let operation = matching
        .next()
        .ok_or_else(|| incomplete(format!("MPCache layer {layer} has no {kind:?} cache view")))?;
    if matching.next().is_some() {
        return Err(incomplete(format!(
            "MPCache layer {layer} has multiple {kind:?} cache views"
        )));
    }
    Ok(operation)
}

fn recover_policy(plan: &DecoderPlan) -> Result<MpcachePolicy, ModelError> {
    let mut recovered: Option<MpcachePolicy> = None;
    for operation in &plan.prefill.operations {
        if !(operation.id.starts_with(METHOD_PREFIX)
            && operation.operator == ModelOperator::AttentionImportance)
        {
            continue;
        }
        let value = operation.attributes.get("policy").ok_or_else(|| {
            incomplete(format!(
                "MPCache operation {} omits its policy",
                operation.id
            ))
        })?;
        let policy: MpcachePolicy = serde_json::from_value(value.clone()).map_err(|error| {
            incomplete(format!(
                "MPCache operation {} has an unreadable policy: {error}",
                operation.id
            ))
        })?;
        match &recovered {
            None => {
                policy
                    .validate()
                    .map_err(|error| incomplete(error.to_string()))?;
                recovered = Some(policy);
            }
            Some(expected) if *expected == policy => {}
            Some(_) => {
                return Err(incomplete(
                    "MPCache static importance policies differ across layers",
                ));
            }
        }
    }
    recovered.ok_or_else(|| incomplete("MPCache plan has no static importance operation"))
}

pub(crate) fn validate_transformation(plan: &DecoderPlan) -> Result<(), ModelError> {
    let has_method_ops = [&plan.prefill, &plan.decode]
        .iter()
        .flat_map(|graph| graph.operations.iter())
        .any(|operation| operation.id.starts_with(METHOD_PREFIX));
    let has_index_state = [
        &plan.prefill.state_inputs,
        &plan.prefill.state_outputs,
        &plan.decode.state_inputs,
        &plan.decode.state_outputs,
    ]
    .iter()
    .flat_map(|states| states.iter())
    .any(|state| state.kind == StateKind::CacheIndices);
    let methods: Vec<&AppliedMethod> = plan
        .transformations
        .iter()
        .filter(|method| method.component == COMPONENT)
        .collect();
    if methods.is_empty() {
        return if has_method_ops || has_index_state {
            Err(incomplete(
                "MPCache operations or cache-index state exist without an applied method",
            ))
        } else {
            Ok(())
        };
    }
    if methods.len() != 1 || !has_method_ops || !has_index_state {
        return Err(incomplete(
            "MPCache transformation must apply exactly once with its operations and state",
        ));
    }
    let method = methods[0];
    if method.implementation != IMPLEMENTATION || method.method_id != METHOD_ID {
        return Err(incomplete("MPCache method metadata is incomplete"));
    }
    let policy = recover_policy(plan)?;
    if canonical_digest(POLICY_DIGEST_NAMESPACE, &policy) != method.configuration_digest {
        return Err(incomplete(
            "MPCache policy digest differs from the applied method record",
        ));
    }
    validate_static_selection(plan, &policy)?;
    validate_decode_selection(plan, &policy)?;
    validate_method_reachability(plan)?;
    Ok(())
}

fn validate_static_selection(plan: &DecoderPlan, policy: &MpcachePolicy) -> Result<(), ModelError> {
    let graph = &plan.prefill;
    let policy_value =
        serde_json::to_value(policy).map_err(|error| incomplete(error.to_string()))?;
    let observation = serde_json::to_value(&policy.observation_window)
        .map_err(|error| incomplete(error.to_string()))?;
    let keep_ratio =
        serde_json::to_value(&policy.static_keep).map_err(|error| incomplete(error.to_string()))?;
    let mut layers = 0_u64;
    for operation in &graph.operations {
        if operation.operator != ModelOperator::Softmax {
            continue;
        }
        layers += 1;
        let layer = operation.layer.ok_or_else(|| {
            incomplete(format!("softmax operation {} has no layer", operation.id))
        })?;
        let prefix = format!("{METHOD_PREFIX}layer.{layer}");
        let importance = method_operation(graph, &format!("{prefix}.static_importance"))?;
        if importance.operator != ModelOperator::AttentionImportance
            || importance.layer != Some(layer)
            || importance.inputs.as_slice() != [operation.id.as_str(), "input.sequence_lengths"]
            || importance.output_shape != [graph.batch, graph.maximum_key_sequence]
            || importance.attributes.get("policy") != Some(&policy_value)
            || importance.attributes.get("observation_window") != Some(&observation)
            || importance
                .attributes
                .get("reduction")
                .and_then(Value::as_str)
                != Some("mean_heads_sum_last_queries")
            || importance
                .attributes
                .get("valid_lengths_input")
                .and_then(Value::as_str)
                != Some("input.sequence_lengths")
        {
            return Err(incomplete(format!(
                "MPCache layer {layer} has invalid static importance"
            )));
        }
        let selection_id = format!("{prefix}.static_selection");
        let selection = method_operation(graph, &selection_id)?;
        let keep = selection.output_shape.get(1).copied().unwrap_or(0);
        if selection.operator != ModelOperator::SecureTopK
            || selection.layer != Some(layer)
            || selection.state_kind != Some(StateKind::CacheIndices)
            || selection.inputs.as_slice() != [importance.id.as_str()]
            || selection.output_shape.len() != 2
            || selection.output_shape[0] != graph.batch
            || keep == 0
            || selection.attributes.get("keep").and_then(Value::as_u64) != Some(keep)
            || selection.attributes.get("keep_ratio") != Some(&keep_ratio)
            || selection
                .attributes
                .get("index_domain")
                .and_then(Value::as_str)
                != Some("global_token_positions")
            || selection
                .attributes
                .get("output_order")
                .and_then(Value::as_str)
                != Some("ascending")
            || selection.attributes.get("padding").and_then(Value::as_str)
                != Some("masked_sentinel")
        {
            return Err(incomplete(format!(
                "MPCache layer {layer} has invalid static selection"
            )));
        }
        let state = unique_state(&graph.state_outputs, layer, StateKind::CacheIndices)?;
        if state.id != selection_id
            || state.shape != selection.output_shape
            || state.maximum_sequence != keep
        {
            return Err(incomplete(format!(
                "MPCache layer {layer} has invalid static selection state"
            )));
        }
    }
    if graph
        .state_outputs
        .iter()
        .filter(|state| state.kind == StateKind::CacheIndices)
        .count()
        != layers as usize
    {
        return Err(incomplete(
            "MPCache static selection states do not match attention layers",
        ));
    }
    Ok(())
}

fn validate_decode_selection(plan: &DecoderPlan, policy: &MpcachePolicy) -> Result<(), ModelError> {
    let graph = &plan.decode;
    let generated = plan
        .decode
        .maximum_key_sequence
        .checked_sub(plan.prefill.maximum_key_sequence)
        .ok_or_else(|| incomplete("MPCache decode bound precedes prefill bound"))?;
    let prefill_bound = plan.prefill.maximum_key_sequence;
    for operation in &graph.operations {
        if operation.operator != ModelOperator::Softmax {
            continue;
        }
        let layer = operation.layer.ok_or_else(|| {
            incomplete(format!("softmax operation {} has no layer", operation.id))
        })?;
        let prefix = format!("{METHOD_PREFIX}layer.{layer}");
        let input_state = unique_state(&graph.state_inputs, layer, StateKind::CacheIndices)?;
        if input_state.shape.len() != 2 || input_state.maximum_sequence != input_state.shape[1] {
            return Err(incomplete(format!(
                "MPCache layer {layer} has invalid cache-index state input"
            )));
        }
        let static_keep = input_state.shape[1];
        let carry_id = format!("{prefix}.static_selection_state");
        let carry = method_operation(graph, &carry_id)?;
        if carry.operator != ModelOperator::SharedIndices
            || carry.state_kind != Some(StateKind::CacheIndices)
            || carry.inputs.as_slice() != [input_state.id.as_str()]
            || carry.output_shape != input_state.shape
            || carry.attributes.get("mode").and_then(Value::as_str) != Some("carry")
            || carry.attributes.get("keep").and_then(Value::as_u64) != Some(static_keep)
            || carry.attributes.get("index_domain").and_then(Value::as_str)
                != Some("global_token_positions")
        {
            return Err(incomplete(format!(
                "MPCache layer {layer} has invalid static selection carry"
            )));
        }
        let output_state = unique_state(&graph.state_outputs, layer, StateKind::CacheIndices)?;
        if output_state.id != carry_id || output_state.shape != carry.output_shape {
            return Err(incomplete(format!(
                "MPCache layer {layer} has invalid cache-index state output"
            )));
        }
        let key_view = unique_state_view(graph, layer, StateKind::Key)?;
        let value_view = unique_state_view(graph, layer, StateKind::Value)?;
        let scores = unique_operation(graph, layer, ModelOperator::AttentionScores)?;
        let mask = unique_operation(graph, layer, ModelOperator::CausalMask)?;
        let values = unique_operation(graph, layer, ModelOperator::AttentionValues)?;
        let scale = unique_operation(graph, layer, ModelOperator::AttentionScale)?;
        let positions_id = format!("{prefix}.selected_positions");
        let positions = method_operation(graph, &positions_id)?;
        let share = policy.share_adjacent_layers && layer >= 3 && layer % 2 == 1;
        let selected_tokens = if share {
            let previous = layer - 1;
            let previous_id = format!("{METHOD_PREFIX}layer.{previous}.selected_positions");
            let previous_op = method_operation(graph, &previous_id)?;
            if positions.operator != ModelOperator::SharedIndices
                || positions.inputs.as_slice() != [previous_id.as_str()]
                || positions.attributes.get("mode").and_then(Value::as_str) != Some("cross_layer")
                || positions.output_shape != previous_op.output_shape
            {
                return Err(incomplete(format!(
                    "MPCache layer {layer} has invalid cross-layer shared selection"
                )));
            }
            let active_id = format!("{prefix}.active_indices");
            let hierarchy_prefix = format!("{prefix}.hierarchy");
            if graph.operations.iter().any(|operation| {
                operation.id == active_id || operation.id.starts_with(&hierarchy_prefix)
            }) {
                return Err(incomplete(format!(
                    "MPCache shared layer {layer} retains a selection hierarchy"
                )));
            }
            positions.output_shape.get(3).copied().unwrap_or(0)
        } else {
            if positions.operator != ModelOperator::SecureGather {
                return Err(incomplete(format!(
                    "MPCache layer {layer} selected positions are not a gather"
                )));
            }
            let active_id = format!("{prefix}.active_indices");
            let candidate_bound = static_keep
                .checked_add(generated)
                .ok_or_else(|| incomplete("MPCache candidate bound overflowed"))?;
            let active = method_operation(graph, &active_id)?;
            if active.operator != ModelOperator::CacheActiveIndices
                || active.inputs.as_slice()
                    != [
                        carry_id.as_str(),
                        "input.positions",
                        "input.sequence_lengths",
                    ]
                || active.output_shape != [graph.batch, candidate_bound]
                || active.attributes.get("static_keep").and_then(Value::as_u64) != Some(static_keep)
                || active
                    .attributes
                    .get("prefill_maximum_sequence")
                    .and_then(Value::as_u64)
                    != Some(prefill_bound)
                || active
                    .attributes
                    .get("candidate_bound")
                    .and_then(Value::as_u64)
                    != Some(candidate_bound)
                || active.attributes.get("semantics").and_then(Value::as_str)
                    != Some("static_plus_generated_global_positions")
            {
                return Err(incomplete(format!(
                    "MPCache layer {layer} has invalid active cache indices"
                )));
            }
            validate_hierarchy(
                graph,
                layer,
                &prefix,
                &active_id,
                key_view,
                scores,
                candidate_bound,
                policy,
            )?
        };
        for (suffix, view, kind) in [
            ("dynamic_key_gather", key_view, StateKind::Key),
            ("dynamic_value_gather", value_view, StateKind::Value),
        ] {
            let gather = method_operation(graph, &format!("{prefix}.{suffix}"))?;
            if gather.operator != ModelOperator::SecureGather
                || gather.state_kind != Some(kind)
                || gather.inputs.as_slice() != [view.id.as_str(), positions_id.as_str()]
                || gather.output_shape
                    != [
                        graph.batch,
                        scores.output_shape[1],
                        scores.output_shape[2],
                        selected_tokens,
                        view.output_shape[3],
                    ]
                || gather.attributes.get("index_axis").and_then(Value::as_u64) != Some(2)
                || gather
                    .attributes
                    .get("index_unit_size")
                    .and_then(Value::as_u64)
                    != Some(1)
                || gather
                    .attributes
                    .get("output_semantics")
                    .and_then(Value::as_str)
                    != Some("per_query_head_cache_window")
                || gather.attributes.get("group_size") != scores.attributes.get("group_size")
            {
                return Err(incomplete(format!(
                    "MPCache layer {layer} has invalid {suffix}"
                )));
            }
        }
        let key_gather_id = format!("{prefix}.dynamic_key_gather");
        let value_gather_id = format!("{prefix}.dynamic_value_gather");
        if scores.inputs.get(1) != Some(&key_gather_id)
            || scores.output_shape.get(3) != Some(&selected_tokens)
            || values.inputs.get(1) != Some(&value_gather_id)
            || mask.inputs.as_slice()
                != [scale.id.as_str(), "input.positions", positions_id.as_str()]
        {
            return Err(incomplete(format!(
                "MPCache layer {layer} has invalid attention rewiring"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_hierarchy(
    graph: &DecoderGraph,
    layer: u64,
    prefix: &str,
    active_id: &str,
    key_view: &ModelOperation,
    scores: &ModelOperation,
    candidate_bound: u64,
    policy: &MpcachePolicy,
) -> Result<u64, ModelError> {
    let levels = hierarchy_levels(candidate_bound, policy);
    if levels.is_empty() {
        return Err(incomplete("MPCache hierarchy has no levels"));
    }
    let kv_heads = key_view.output_shape.get(1).copied().unwrap_or(0);
    let head_dim = key_view.output_shape.get(3).copied().unwrap_or(0);
    let heads = scores.output_shape.get(1).copied().unwrap_or(0);
    let query_sequence = scores.output_shape.get(2).copied().unwrap_or(0);
    let query = scores.inputs.first().cloned().ok_or_else(|| {
        incomplete(format!(
            "MPCache layer {layer} attention scores has no query input"
        ))
    })?;
    let mut position_candidates = active_id.to_owned();
    let mut key_candidate = key_view.id.clone();
    let mut candidate_axis = 1_u64;
    let mut selected_tokens = 0_u64;
    for (level, spec) in levels.iter().enumerate() {
        let last = level + 1 == levels.len();
        let hierarchy = format!("{prefix}.hierarchy.{level}");
        let (expected_bounds_shape, expected_group_size) = if level == 0 {
            (
                vec![graph.batch, kv_heads, spec.clusters, head_dim, 2],
                scores
                    .attributes
                    .get("group_size")
                    .cloned()
                    .unwrap_or(Value::Null),
            )
        } else {
            (
                vec![
                    graph.batch,
                    heads,
                    query_sequence,
                    spec.clusters,
                    head_dim,
                    2,
                ],
                json!(1),
            )
        };
        let bounds_id = format!("{hierarchy}.cluster_bounds");
        let bounds = method_operation(graph, &bounds_id)?;
        if bounds.operator != ModelOperator::ClusterBounds
            || bounds.inputs.as_slice() != [key_candidate.as_str()]
            || bounds.output_shape != expected_bounds_shape
            || bounds
                .attributes
                .get("cluster_size")
                .and_then(Value::as_u64)
                != Some(spec.cluster_size)
            || bounds.attributes.get("level").and_then(Value::as_u64) != Some(level as u64)
        {
            return Err(incomplete(format!(
                "MPCache layer {layer} hierarchy {level} has invalid cluster bounds"
            )));
        }
        let similarity_id = format!("{hierarchy}.cluster_similarity");
        let similarity = method_operation(graph, &similarity_id)?;
        let alpha =
            serde_json::to_value(&policy.alpha).map_err(|error| incomplete(error.to_string()))?;
        if similarity.operator != ModelOperator::ClusterSimilarity
            || similarity.inputs.as_slice() != [query.as_str(), bounds_id.as_str()]
            || similarity.output_shape != [graph.batch, heads, query_sequence, spec.clusters]
            || similarity.attributes.get("alpha") != Some(&alpha)
            || similarity.attributes.get("level").and_then(Value::as_u64) != Some(level as u64)
            || similarity.attributes.get("group_size") != Some(&expected_group_size)
        {
            return Err(incomplete(format!(
                "MPCache layer {layer} hierarchy {level} has invalid cluster similarity"
            )));
        }
        let selection_id = format!("{hierarchy}.selection");
        let selection = method_operation(graph, &selection_id)?;
        if selection.operator != ModelOperator::SecureTopK
            || selection.inputs.as_slice() != [similarity_id.as_str()]
            || selection.output_shape != [graph.batch, heads, query_sequence, spec.keep]
            || selection.attributes.get("keep").and_then(Value::as_u64) != Some(spec.keep)
            || selection
                .attributes
                .get("index_domain")
                .and_then(Value::as_str)
                != Some("cluster_indices")
            || selection
                .attributes
                .get("output_order")
                .and_then(Value::as_str)
                != Some("ascending")
        {
            return Err(incomplete(format!(
                "MPCache layer {layer} hierarchy {level} has invalid top-k selection"
            )));
        }
        let positions_id = if last {
            format!("{prefix}.selected_positions")
        } else {
            format!("{hierarchy}.selected_positions")
        };
        let positions = method_operation(graph, &positions_id)?;
        if positions.operator != ModelOperator::SecureGather
            || positions.inputs.as_slice() != [position_candidates.as_str(), selection_id.as_str()]
            || positions.output_shape != [graph.batch, heads, query_sequence, spec.retained]
            || positions
                .attributes
                .get("index_axis")
                .and_then(Value::as_u64)
                != Some(candidate_axis)
            || positions
                .attributes
                .get("index_unit_size")
                .and_then(Value::as_u64)
                != Some(spec.cluster_size)
            || positions
                .attributes
                .get("output_semantics")
                .and_then(Value::as_str)
                != Some("global_token_positions")
        {
            return Err(incomplete(format!(
                "MPCache layer {layer} hierarchy {level} has invalid selected positions"
            )));
        }
        if !last {
            let key_gather_id = format!("{hierarchy}.candidate_key_gather");
            let key_gather = method_operation(graph, &key_gather_id)?;
            if key_gather.operator != ModelOperator::SecureGather
                || key_gather.inputs.as_slice() != [key_view.id.as_str(), positions_id.as_str()]
                || key_gather.output_shape
                    != [graph.batch, heads, query_sequence, spec.retained, head_dim]
            {
                return Err(incomplete(format!(
                    "MPCache layer {layer} hierarchy {level} has invalid candidate key gather"
                )));
            }
            key_candidate = key_gather_id;
        }
        position_candidates = positions_id;
        candidate_axis = 3;
        if last {
            selected_tokens = spec.retained;
        }
    }
    Ok(selected_tokens)
}

fn validate_method_reachability(plan: &DecoderPlan) -> Result<(), ModelError> {
    for graph in [&plan.prefill, &plan.decode] {
        let ids: BTreeSet<&str> = graph
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect();
        let mut reachable: BTreeSet<&str> = BTreeSet::new();
        let mut stack: Vec<&str> = graph
            .state_outputs
            .iter()
            .map(|state| state.id.as_str())
            .chain(std::iter::once(graph.output.as_str()))
            .collect();
        while let Some(id) = stack.pop() {
            if !reachable.insert(id) {
                continue;
            }
            if let Some(operation) = graph.operations.iter().find(|operation| operation.id == id) {
                stack.extend(
                    operation
                        .inputs
                        .iter()
                        .map(String::as_str)
                        .filter(|input| ids.contains(input)),
                );
            }
        }
        if let Some(orphan) = graph.operations.iter().find(|operation| {
            operation.id.starts_with(METHOD_PREFIX) && !reachable.contains(operation.id.as_str())
        }) {
            return Err(incomplete(format!(
                "MPCache operation {} is unreachable from attention or state outputs",
                orphan.id
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower_model_json, lower_qwen_decoder, DecoderWorkload, QwenConfig};

    const GEMMA_EXACT_FIXTURE: &str =
        include_str!("../../pllm-models/tests/fixtures/gemma-4-E4B-it-ee0ef602-config.json");
    const QWEN3_EXACT_FIXTURE: &[u8] =
        include_bytes!("../../pllm-models/tests/fixtures/mini-coder-4b-c87892d-config.json");

    fn qwen_plan(layers: u32) -> DecoderPlan {
        let config = QwenConfig::from_json(
            format!(
                r#"{{"model_type":"qwen2","hidden_size":32,"intermediate_size":64,"num_hidden_layers":{layers},"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":128,"max_position_embeddings":512,"hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000.0,"tie_word_embeddings":true}}"#
            )
            .as_bytes(),
        )
        .unwrap();
        lower_qwen_decoder(
            &config,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 100,
                max_new_tokens: 20,
            },
        )
        .unwrap()
    }

    fn operation<'a>(graph: &'a DecoderGraph, id: &str) -> &'a ModelOperation {
        graph
            .operations
            .iter()
            .find(|operation| operation.id == id)
            .unwrap()
    }

    fn kv_states(states: &[StateTensor]) -> Vec<(String, Vec<u64>, u64)> {
        states
            .iter()
            .filter(|state| matches!(state.kind, StateKind::Key | StateKind::Value))
            .map(|state| {
                (
                    state.id.clone(),
                    state.shape.clone(),
                    state.maximum_sequence,
                )
            })
            .collect()
    }

    fn cache_index_states(states: &[StateTensor]) -> Vec<&StateTensor> {
        states
            .iter()
            .filter(|state| state.kind == StateKind::CacheIndices)
            .collect()
    }

    #[test]
    fn transforms_fixed_shape_qwen2_prefill_state_and_decode_attention() {
        let base = qwen_plan(4);
        let original_prefill_kv = kv_states(&base.prefill.state_outputs);
        let original_decode_kv: Vec<(String, Vec<u64>, u64)> = kv_states(&base.decode.state_inputs)
            .into_iter()
            .chain(kv_states(&base.decode.state_outputs))
            .collect();
        let plan = optimize(&base, MpcachePolicy::r23_reference_policy()).unwrap();
        plan.validate().unwrap();
        assert_eq!(
            plan.decode.maximum_key_sequence,
            base.decode.maximum_key_sequence
        );
        assert_eq!(kv_states(&plan.prefill.state_outputs), original_prefill_kv);
        let transformed_decode_kv: Vec<(String, Vec<u64>, u64)> =
            kv_states(&plan.decode.state_inputs)
                .into_iter()
                .chain(kv_states(&plan.decode.state_outputs))
                .collect();
        assert_eq!(transformed_decode_kv, original_decode_kv);
        assert_eq!(cache_index_states(&plan.prefill.state_outputs).len(), 4);
        assert_eq!(
            (
                cache_index_states(&plan.decode.state_inputs).len(),
                cache_index_states(&plan.decode.state_outputs).len()
            ),
            (4, 4)
        );
        for state in cache_index_states(&plan.prefill.state_outputs)
            .into_iter()
            .chain(cache_index_states(&plan.decode.state_inputs))
            .chain(cache_index_states(&plan.decode.state_outputs))
        {
            assert_eq!(state.shape, vec![1, 30]);
            assert_eq!(state.maximum_sequence, 30);
        }
        let scores = operation(&plan.decode, "layer.0.attention_scores");
        assert_eq!(scores.output_shape[3], 16);
        assert_eq!(
            scores.inputs[1],
            "method.kv_cache_eviction.layer.0.dynamic_key_gather"
        );
        let mask = operation(&plan.decode, "layer.0.causal_mask");
        assert_eq!(
            mask.inputs[2],
            "method.kv_cache_eviction.layer.0.selected_positions"
        );
        assert_eq!(
            operation(
                &plan.decode,
                "method.kv_cache_eviction.layer.0.hierarchy.0.cluster_bounds"
            )
            .output_shape,
            vec![1, 2, 2, 8, 2],
        );
        assert_eq!(
            operation(
                &plan.decode,
                "method.kv_cache_eviction.layer.0.hierarchy.0.cluster_similarity"
            )
            .output_shape,
            vec![1, 4, 1, 2],
        );
        assert_eq!(
            operation(
                &plan.decode,
                "method.kv_cache_eviction.layer.0.hierarchy.1.cluster_bounds"
            )
            .output_shape,
            vec![1, 4, 1, 2, 8, 2],
        );
        assert_eq!(
            operation(
                &plan.decode,
                "method.kv_cache_eviction.layer.0.hierarchy.1.cluster_similarity"
            )
            .output_shape,
            vec![1, 4, 1, 2],
        );
    }

    #[test]
    fn transformation_is_deterministic_and_policy_is_bounded() {
        let policy = MpcachePolicy::r23_reference_policy();
        assert_eq!(
            optimize(&qwen_plan(4), policy.clone()).unwrap().digest(),
            optimize(&qwen_plan(4), policy).unwrap().digest()
        );
        let mut invalid = MpcachePolicy::r23_reference_policy();
        invalid.dynamic_keep.numerator = 0;
        assert!(matches!(
            optimize(&qwen_plan(4), invalid),
            Err(MpcacheError::InvalidPolicy(_))
        ));
        let mut invalid = MpcachePolicy::r23_reference_policy();
        invalid.cluster_sizes = vec![24, 16];
        assert!(matches!(
            optimize(&qwen_plan(4), invalid),
            Err(MpcacheError::InvalidPolicy(_))
        ));
    }

    #[test]
    fn exact_qwen3_fixture_transforms_and_validates() {
        let base = lower_model_json(
            QWEN3_EXACT_FIXTURE,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 16,
                max_new_tokens: 4,
            },
        )
        .unwrap();
        let plan = optimize(&base, MpcachePolicy::r23_reference_policy()).unwrap();
        plan.validate().unwrap();
        assert_eq!(plan.transformations.len(), 1);
        assert_eq!(plan.transformations[0].method_id, "R23");
        assert_eq!(
            cache_index_states(&plan.prefill.state_outputs).len(),
            base.prefill.state_outputs.len() / 2
        );
    }

    #[test]
    fn exact_gemma_fixture_is_cleanly_rejected_as_incompatible() {
        let plan = lower_model_json(
            GEMMA_EXACT_FIXTURE.as_bytes(),
            DecoderWorkload {
                batch: 2,
                max_input_tokens: 1_024,
                max_new_tokens: 32,
            },
        )
        .unwrap();
        plan.validate().unwrap();
        let error = optimize(&plan, MpcachePolicy::r23_reference_policy()).unwrap_err();
        assert!(matches!(error, MpcacheError::InvalidPlan(_)));
        assert!(error
            .to_string()
            .contains("MPCache requires a dense Qwen fixed-shape cache plan"));
    }

    #[test]
    fn reapplication_is_rejected_without_mutating_the_plan() {
        let plan = optimize(&qwen_plan(4), MpcachePolicy::r23_reference_policy()).unwrap();
        let digest = plan.digest();
        let error = optimize(&plan, MpcachePolicy::r23_reference_policy()).unwrap_err();
        assert!(matches!(error, MpcacheError::InvalidPlan(_)));
        assert!(error.to_string().contains("MPCache is already applied"));
        assert_eq!(plan.digest(), digest);
    }

    #[test]
    fn tampered_policy_attribute_fails_validation() {
        let mut plan = optimize(&qwen_plan(4), MpcachePolicy::r23_reference_policy()).unwrap();
        plan.validate().unwrap();
        let importance = plan
            .prefill
            .operations
            .iter_mut()
            .find(|operation| operation.id == "method.kv_cache_eviction.layer.0.static_importance")
            .unwrap();
        importance.attributes["policy"]["static_keep"]["numerator"] = json!(1);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn tampered_static_state_shape_fails_validation() {
        let mut plan = optimize(&qwen_plan(4), MpcachePolicy::r23_reference_policy()).unwrap();
        plan.validate().unwrap();
        plan.prefill
            .state_outputs
            .iter_mut()
            .find(|state| state.kind == StateKind::CacheIndices)
            .unwrap()
            .shape[1] += 1;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn tampered_selected_position_input_fails_validation() {
        let mut plan = optimize(&qwen_plan(4), MpcachePolicy::r23_reference_policy()).unwrap();
        plan.validate().unwrap();
        let mask = plan
            .decode
            .operations
            .iter_mut()
            .find(|operation| operation.id == "layer.0.causal_mask")
            .unwrap();
        mask.inputs[2] = "method.kv_cache_eviction.layer.1.selected_positions".to_owned();
        assert!(plan.validate().is_err());
    }

    #[test]
    fn tampered_cross_layer_share_fails_validation() {
        let mut plan = optimize(&qwen_plan(4), MpcachePolicy::r23_reference_policy()).unwrap();
        plan.validate().unwrap();
        let share = plan
            .decode
            .operations
            .iter_mut()
            .find(|operation| operation.id == "method.kv_cache_eviction.layer.3.selected_positions")
            .unwrap();
        share.inputs[0] = "method.kv_cache_eviction.layer.0.selected_positions".to_owned();
        assert!(plan.validate().is_err());
    }

    #[test]
    fn tampered_hierarchy_bounds_shape_fails_validation() {
        let mut plan = optimize(&qwen_plan(4), MpcachePolicy::r23_reference_policy()).unwrap();
        plan.validate().unwrap();
        let bounds = plan
            .decode
            .operations
            .iter_mut()
            .find(|operation| {
                operation.id == "method.kv_cache_eviction.layer.0.hierarchy.1.cluster_bounds"
            })
            .unwrap();
        bounds.output_shape = vec![1, 2, 2, 8, 2];
        assert!(plan.validate().is_err());
    }
}
