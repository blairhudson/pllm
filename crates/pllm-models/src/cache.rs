//! Cache-policy graph transformations.

use crate::{
    AppliedMethod, DecoderGraph, DecoderPlan, ModelError, ModelOperation, ModelOperator, StateKind,
    StateTensor,
};
use pllm_types::canonical_digest;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

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
    pub fn paper_profile() -> Self {
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

pub fn optimize(_base: &DecoderPlan, _policy: MpcachePolicy) -> Result<DecoderPlan, MpcacheError> {
    Err(MpcacheError::InvalidPlan(
        "fixed-shape attention/state handoff is unimplemented for MPCache".into(),
    ))
}

// Retained only to test legacy transformation mechanics. Public application is fail-closed.
#[allow(dead_code)]
fn optimize_legacy_unreachable(
    base: &DecoderPlan,
    policy: MpcachePolicy,
) -> Result<DecoderPlan, MpcacheError> {
    policy.validate()?;
    base.validate()?;
    validate_compatibility(base)?;
    let input_digest = base.digest();
    let configuration_digest = canonical_digest("pllm.method.mpcache.policy.v1", &policy);
    let mut decoder = base.clone();
    decoder.prefill = transform_prefill(&base.prefill, &policy)?;
    let retained = policy.static_keep.ceil(base.prefill.maximum_key_sequence);
    let generated = base
        .decode
        .maximum_key_sequence
        .checked_sub(base.prefill.maximum_key_sequence)
        .ok_or_else(|| MpcacheError::InvalidPlan("decode bound precedes prefill bound".into()))?;
    let maximum_key_sequence = retained
        .checked_add(generated)
        .ok_or_else(|| MpcacheError::InvalidPlan("transformed decode bound overflowed".into()))?;
    decoder.decode = transform_decode(&base.decode, &policy, maximum_key_sequence)?;
    decoder.transformations.push(AppliedMethod {
        component: "pllm/kv-cache-eviction".to_owned(),
        implementation: "pllm/mpcache/v1".to_owned(),
        method_id: "R23".to_owned(),
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
) -> Result<DecoderGraph, MpcacheError> {
    let keep = policy.static_keep.ceil(graph.maximum_key_sequence);
    let mut transformed = graph.clone();
    transformed.operations.clear();
    let mut selections = BTreeMap::<u64, String>::new();
    for operation in &graph.operations {
        transformed.operations.push(operation.clone());
        if operation.operator != ModelOperator::Softmax {
            continue;
        }
        let layer = layer_number(operation)?;
        let prefix = format!("method.mpcache.layer.{layer}");
        let importance = format!("{prefix}.static_importance");
        transformed.operations.push(op(
            &importance,
            ModelOperator::AttentionImportance,
            layer,
            vec![operation.id.clone()],
            vec![graph.batch, graph.maximum_key_sequence],
            json!({"observation_window": policy.observation_window}),
        ));
        let selection = selection_operation(layer, &prefix, &importance, keep, &selections, false)?;
        selections.insert(layer, selection.id.clone());
        transformed.operations.push(selection.clone());
        let key_append = state_operation(graph, layer, StateKind::Key)?;
        let value_append = state_operation(graph, layer, StateKind::Value)?;
        let (key, value) = gather_operations(
            &prefix,
            "static",
            layer,
            (&selection.id, 1),
            keep,
            (&key_append.id, &value_append.id),
            &key_append.output_shape,
        )?;
        transformed.operations.push(key.clone());
        transformed.operations.push(value.clone());
        replace_state_output(
            &mut transformed.state_outputs,
            layer,
            keep,
            &key.id,
            &value.id,
        )?;
    }
    Ok(transformed)
}

fn transform_decode(
    graph: &DecoderGraph,
    policy: &MpcachePolicy,
    maximum_key_sequence: u64,
) -> Result<DecoderGraph, MpcacheError> {
    let final_cluster_size = policy.cluster_sizes.last().copied().ok_or_else(|| {
        MpcacheError::InvalidPolicy("cluster_sizes must be nonempty and positive".into())
    })?;
    let target_tokens = policy.dynamic_keep.ceil(maximum_key_sequence);
    let final_keep = target_tokens.div_ceil(final_cluster_size);
    let final_selected_tokens =
        bounded_product(final_keep, final_cluster_size, maximum_key_sequence);
    let mut resized = graph.clone();
    resize_decode_graph(&mut resized, maximum_key_sequence);
    let mut transformed = resized.clone();
    transformed.operations.clear();
    let mut selections = BTreeMap::<u64, String>::new();
    for source in &resized.operations {
        let mut operation = source.clone();
        if operation.operator == ModelOperator::AttentionScores {
            let layer = layer_number(&operation)?;
            let prefix = format!("method.mpcache.layer.{layer}");
            let key_append = state_operation(&resized, layer, StateKind::Key)?;
            let value_append = state_operation(&resized, layer, StateKind::Value)?;
            let query = operation.inputs.first().cloned().ok_or_else(|| {
                MpcacheError::InvalidPlan(format!(
                    "attention scores operation {} has no query input",
                    operation.id
                ))
            })?;
            let key_shape = key_append.output_shape.as_slice();
            let [_, key_heads, _, key_features] = key_shape else {
                return Err(MpcacheError::InvalidPlan(format!(
                    "layer {layer} Key cache append must have rank four"
                )));
            };
            let share = policy.share_adjacent_layers && layer >= 3 && layer % 2 == 1;
            let (selection, selected_tokens) = if share {
                (
                    selection_operation(layer, &prefix, "", final_keep, &selections, true)?,
                    final_selected_tokens,
                )
            } else {
                let mut candidate_key = key_append.id.clone();
                let mut candidate_value = value_append.id.clone();
                let mut candidate_shape = key_shape.to_vec();
                let mut candidate_tokens = maximum_key_sequence;
                let mut final_result = None;
                for (level, cluster_size) in policy.cluster_sizes.iter().copied().enumerate() {
                    let clusters = candidate_tokens.div_ceil(cluster_size);
                    let last = level + 1 == policy.cluster_sizes.len();
                    let keep = if last {
                        final_keep.min(clusters)
                    } else if u128::from(policy.dynamic_keep.numerator) * 2
                        < u128::from(policy.dynamic_keep.denominator)
                    {
                        clusters.div_ceil(2)
                    } else {
                        clusters
                    };
                    let level_prefix = if last {
                        prefix.clone()
                    } else {
                        format!("{prefix}.hierarchy.{level}")
                    };
                    let bounds = format!("{level_prefix}.cluster_bounds");
                    transformed.operations.push(op(
                        &bounds,
                        ModelOperator::ClusterBounds,
                        layer,
                        vec![candidate_key.clone()],
                        vec![graph.batch, *key_heads, clusters, *key_features, 2],
                        json!({"cluster_size": cluster_size, "level": level}),
                    ));
                    let similarity = format!("{level_prefix}.cluster_similarity");
                    let attention_heads =
                        operation.output_shape.get(1).copied().ok_or_else(|| {
                            MpcacheError::InvalidPlan(format!(
                                "attention scores operation {} must have rank four",
                                operation.id
                            ))
                        })?;
                    transformed.operations.push(op(
                        &similarity,
                        ModelOperator::ClusterSimilarity,
                        layer,
                        vec![query.clone(), bounds],
                        vec![graph.batch, attention_heads, clusters],
                        json!({"alpha": policy.alpha, "level": level}),
                    ));
                    let selection = selection_operation(
                        layer,
                        &level_prefix,
                        &similarity,
                        keep,
                        &selections,
                        false,
                    )?;
                    if last {
                        final_result = Some((
                            selection,
                            bounded_product(keep, cluster_size, candidate_tokens),
                        ));
                        break;
                    }
                    transformed.operations.push(selection.clone());
                    let retained_tokens = bounded_product(keep, cluster_size, candidate_tokens);
                    let (key, value) = gather_operations(
                        &level_prefix,
                        "candidate",
                        layer,
                        (&selection.id, cluster_size),
                        retained_tokens,
                        (&candidate_key, &candidate_value),
                        &candidate_shape,
                    )?;
                    candidate_key = key.id.clone();
                    candidate_value = value.id.clone();
                    candidate_shape = key.output_shape.clone();
                    candidate_tokens = retained_tokens;
                    transformed.operations.push(key);
                    transformed.operations.push(value);
                }
                final_result.ok_or_else(|| {
                    MpcacheError::InvalidPolicy(
                        "cluster_sizes must contain a final hierarchy level".into(),
                    )
                })?
            };
            selections.insert(layer, selection.id.clone());
            transformed.operations.push(selection.clone());
            let (key, value) = gather_operations(
                &prefix,
                "dynamic",
                layer,
                (&selection.id, final_cluster_size),
                selected_tokens,
                (&key_append.id, &value_append.id),
                &key_append.output_shape,
            )?;
            transformed.operations.push(key.clone());
            transformed.operations.push(value.clone());
            *operation.inputs.get_mut(1).ok_or_else(|| {
                MpcacheError::InvalidPlan(format!(
                    "attention scores operation {} has no Key input",
                    operation.id
                ))
            })? = key.id;
            transformed.operations.push(operation);
            continue;
        }
        if operation.operator == ModelOperator::AttentionValues {
            let layer = layer_number(&operation)?;
            *operation.inputs.get_mut(1).ok_or_else(|| {
                MpcacheError::InvalidPlan(format!(
                    "attention values operation {} has no Value input",
                    operation.id
                ))
            })? = format!("method.mpcache.layer.{layer}.dynamic_value_gather");
        }
        transformed.operations.push(operation);
    }
    Ok(transformed)
}

fn selection_operation(
    layer: u64,
    prefix: &str,
    scores: &str,
    keep: u64,
    previous: &BTreeMap<u64, String>,
    share: bool,
) -> Result<ModelOperation, MpcacheError> {
    let operator = if share {
        ModelOperator::SharedIndices
    } else {
        ModelOperator::SecureTopK
    };
    let input = if share {
        let previous_layer = layer.checked_sub(1).ok_or_else(|| {
            MpcacheError::InvalidPlan(format!(
                "layer {layer} cannot share indices without a previous layer"
            ))
        })?;
        previous
            .get(&previous_layer)
            .ok_or_else(|| {
                MpcacheError::InvalidPlan(format!(
                    "layer {layer} cannot share indices: layer {previous_layer} has no previous selection"
                ))
            })?
            .clone()
    } else {
        scores.to_owned()
    };
    Ok(op(
        &format!("{prefix}.selection"),
        operator,
        layer,
        vec![input],
        vec![keep],
        json!({"keep": keep, "global_indices": 1}),
    ))
}

fn gather_operations(
    prefix: &str,
    phase: &str,
    layer: u64,
    selection: (&str, u64),
    tokens: u64,
    state_inputs: (&str, &str),
    key_shape: &[u64],
) -> Result<(ModelOperation, ModelOperation), MpcacheError> {
    let (key_input, value_input) = state_inputs;
    let (selection, unit_size) = selection;
    let mut shape = key_shape.to_vec();
    *shape.get_mut(2).ok_or_else(|| {
        MpcacheError::InvalidPlan(format!("layer {layer} cache producer must have rank four"))
    })? = tokens;
    let key = op(
        &format!("{prefix}.{phase}_key_gather"),
        ModelOperator::SecureGather,
        layer,
        vec![key_input.to_owned(), selection.to_owned()],
        shape.clone(),
        json!({"index_unit_size": unit_size}),
    );
    let value = op(
        &format!("{prefix}.{phase}_value_gather"),
        ModelOperator::SecureGather,
        layer,
        vec![value_input.to_owned(), selection.to_owned()],
        shape,
        json!({"index_unit_size": unit_size}),
    );
    Ok((key, value))
}

fn replace_state_output(
    outputs: &mut [StateTensor],
    layer: u64,
    tokens: u64,
    key: &str,
    value: &str,
) -> Result<(), MpcacheError> {
    let mut found_key = false;
    let mut found_value = false;
    for output in outputs {
        if output.layer == Some(layer) {
            match output.kind {
                StateKind::Key => {
                    output.id = key.to_owned();
                    found_key = true;
                }
                StateKind::Value => {
                    output.id = value.to_owned();
                    found_value = true;
                }
                StateKind::Recurrent | StateKind::Convolution => continue,
            }
            *output.shape.get_mut(2).ok_or_else(|| {
                MpcacheError::InvalidPlan(format!(
                    "layer {layer} {:?} state output must have rank four",
                    output.kind
                ))
            })? = tokens;
            output.maximum_sequence = tokens;
        }
    }
    if !found_key || !found_value {
        return Err(MpcacheError::InvalidPlan(format!(
            "layer {layer} must have Key and Value state outputs"
        )));
    }
    Ok(())
}

fn bounded_product(left: u64, right: u64, bound: u64) -> u64 {
    (u128::from(left) * u128::from(right)).min(u128::from(bound)) as u64
}

fn resize_decode_graph(graph: &mut DecoderGraph, maximum_key_sequence: u64) {
    graph.maximum_key_sequence = maximum_key_sequence;
    for state in graph
        .state_inputs
        .iter_mut()
        .chain(graph.state_outputs.iter_mut())
    {
        if matches!(state.kind, StateKind::Key | StateKind::Value) && state.shape.len() == 4 {
            state.shape[2] = maximum_key_sequence;
            state.maximum_sequence = maximum_key_sequence;
        }
    }
    for operation in &mut graph.operations {
        match operation.operator {
            ModelOperator::KvCacheAppend if operation.output_shape.len() == 4 => {
                operation.output_shape[2] = maximum_key_sequence;
                operation.attributes["state_capacity"] = json!(maximum_key_sequence);
                operation.attributes["attention_domain"]["maximum_sequence"] =
                    json!(maximum_key_sequence);
            }
            ModelOperator::CacheSuffix if operation.output_shape.len() == 4 => {
                operation.output_shape[2] = maximum_key_sequence;
                operation.attributes["maximum_sequence"] = json!(maximum_key_sequence);
            }
            ModelOperator::AttentionScores | ModelOperator::CausalMask | ModelOperator::Softmax => {
                if let Some(last) = operation.output_shape.last_mut() {
                    *last = maximum_key_sequence;
                }
            }
            _ => {}
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower_model_json, lower_qwen_decoder, DecoderWorkload, QwenConfig};

    const GEMMA_EXACT_FIXTURE: &str =
        include_str!("../../pllm-models/tests/fixtures/gemma-4-E4B-it-ee0ef602-config.json");
    const QWEN3_EXACT_FIXTURE: &[u8] =
        include_bytes!("../../pllm-models/tests/fixtures/mini-coder-4b-c87892d-config.json");

    fn base() -> DecoderPlan {
        legacy_plan(4)
    }

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

    fn legacy_plan(layers: u32) -> DecoderPlan {
        let mut plan = qwen_plan(layers);
        plan.model_family = "synthetic_legacy_attention".into();
        plan.adapter = "test.synthetic_legacy_attention.v1".into();
        let decode_inputs = plan.decode.state_inputs.clone();
        for graph in [&mut plan.prefill, &mut plan.decode] {
            if graph.mode == crate::DecoderMode::Prefill {
                graph.state_inputs = decode_inputs.clone();
            }
            let suffixes = graph
                .operations
                .iter()
                .filter(|operation| operation.operator == ModelOperator::CacheSuffix)
                .map(|operation| (operation.id.clone(), operation.inputs[0].clone()))
                .collect::<BTreeMap<_, _>>();
            for operation in &mut graph.operations {
                if matches!(
                    operation.operator,
                    ModelOperator::AttentionScores | ModelOperator::AttentionValues
                ) {
                    if let Some(input) = operation.inputs.get_mut(1) {
                        if let Some(append) = suffixes.get(input) {
                            *input = append.clone();
                        }
                    }
                }
                if operation.operator == ModelOperator::KvCacheAppend {
                    let state = operation.attributes["state"]
                        .as_str()
                        .expect("rich cache has state")
                        .to_owned();
                    let current = operation.inputs
                        [usize::from(graph.mode == crate::DecoderMode::Decode)]
                    .clone();
                    operation.inputs = vec![state.clone(), current];
                    operation.attributes = json!({"state": state});
                }
            }
            graph
                .operations
                .retain(|operation| operation.operator != ModelOperator::CacheSuffix);
        }
        plan.validate().unwrap();
        plan
    }

    fn relabel_layers(plan: &mut DecoderPlan, relabel: impl Fn(u64) -> u64) {
        for graph in [&mut plan.prefill, &mut plan.decode] {
            for operation in &mut graph.operations {
                operation.layer = operation.layer.map(&relabel);
            }
            for state in graph
                .state_inputs
                .iter_mut()
                .chain(graph.state_outputs.iter_mut())
            {
                state.layer = state.layer.map(&relabel);
            }
        }
    }

    #[test]
    fn transforms_prefill_state_and_decode_attention() {
        let plan = optimize_legacy_unreachable(&base(), MpcachePolicy::paper_profile()).unwrap();
        assert_eq!(plan.transformations[0].method_id, "R23");
        assert!(plan
            .prefill
            .state_outputs
            .iter()
            .any(|state| state.id == "method.mpcache.layer.0.static_key_gather"));
        assert!(plan.prefill.operations.iter().any(|operation| {
            operation.id == "method.mpcache.layer.3.selection"
                && operation.operator == ModelOperator::SecureTopK
        }));
        assert!(plan.decode.operations.iter().any(|operation| {
            operation.id == "method.mpcache.layer.3.selection"
                && operation.operator == ModelOperator::SharedIndices
        }));
        assert!(!plan
            .decode
            .operations
            .iter()
            .any(|operation| operation.id.starts_with("method.mpcache.layer.3.cluster_")));
        assert_eq!(plan.prefill.state_outputs[0].maximum_sequence, 30);
        assert_eq!(plan.decode.maximum_key_sequence, 49);
        assert!(plan.decode.operations.iter().any(|operation| {
            operation.id == "method.mpcache.layer.0.hierarchy.0.cluster_similarity"
        }));
        assert!(plan.decode.operations.iter().any(|operation| {
            operation.id == "method.mpcache.layer.0.hierarchy.0.candidate_key_gather"
        }));
        let scores = plan
            .decode
            .operations
            .iter()
            .find(|operation| operation.id == "layer.0.attention_scores")
            .unwrap();
        assert_eq!(
            scores.inputs[1],
            "method.mpcache.layer.0.dynamic_key_gather"
        );
        assert!(plan
            .decode
            .state_outputs
            .iter()
            .any(|state| state.id == "layer.0.key_append"));
    }

    #[test]
    fn transformation_is_deterministic_and_policy_is_bounded() {
        let policy = MpcachePolicy::paper_profile();
        assert_eq!(
            optimize_legacy_unreachable(&base(), policy.clone())
                .unwrap()
                .digest(),
            optimize_legacy_unreachable(&base(), policy)
                .unwrap()
                .digest()
        );
        let mut invalid = MpcachePolicy::paper_profile();
        invalid.dynamic_keep.numerator = 0;
        assert!(matches!(
            optimize_legacy_unreachable(&base(), invalid),
            Err(MpcacheError::InvalidPolicy(_))
        ));
        let mut invalid = MpcachePolicy::paper_profile();
        invalid.cluster_sizes = vec![24, 16];
        assert!(matches!(
            optimize_legacy_unreachable(&base(), invalid),
            Err(MpcacheError::InvalidPolicy(_))
        ));
    }

    #[test]
    fn public_application_rejects_all_plans_before_transformation() {
        let qwen2 = qwen_plan(1);
        let qwen3 = lower_model_json(
            QWEN3_EXACT_FIXTURE,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 2,
                max_new_tokens: 1,
            },
        )
        .unwrap();
        for plan in [base(), qwen2, qwen3] {
            let digest = plan.digest();
            let error = optimize(&plan, MpcachePolicy::paper_profile()).unwrap_err();
            assert_eq!(plan.digest(), digest);
            assert!(plan.transformations.is_empty());
            assert!(error
                .to_string()
                .contains("fixed-shape attention/state handoff is unimplemented"));
        }
    }

    #[test]
    fn exact_gemma_fixture_is_cleanly_rejected_as_incompatible() {
        let mut plan = lower_model_json(
            GEMMA_EXACT_FIXTURE.as_bytes(),
            DecoderWorkload {
                batch: 2,
                max_input_tokens: 1_024,
                max_new_tokens: 32,
            },
        )
        .unwrap();
        plan.validate().unwrap();

        let error = optimize(&plan, MpcachePolicy::paper_profile()).unwrap_err();
        assert!(matches!(error, MpcacheError::InvalidPlan(_)));
        assert!(error
            .to_string()
            .contains("fixed-shape attention/state handoff is unimplemented"));

        plan.model_family = "renamed_fixture".into();
        plan.adapter = "renamed_adapter".into();
        let error = optimize(&plan, MpcachePolicy::paper_profile()).unwrap_err();
        assert!(matches!(error, MpcacheError::InvalidPlan(_)));
        assert!(error
            .to_string()
            .contains("fixed-shape attention/state handoff is unimplemented"));
    }

    #[test]
    fn malformed_layer_order_is_cleanly_rejected() {
        let mut plan = base();
        relabel_layers(&mut plan, |layer| match layer {
            2 => 3,
            3 => 2,
            other => other,
        });
        plan.validate().unwrap();

        let error = optimize_legacy_unreachable(&plan, MpcachePolicy::paper_profile()).unwrap_err();
        assert!(matches!(error, MpcacheError::InvalidPlan(_)));
        assert!(error.to_string().contains("has no previous selection"));
    }

    #[test]
    fn shared_layer_without_previous_index_is_cleanly_rejected() {
        let mut plan = legacy_plan(1);
        relabel_layers(&mut plan, |_| 3);
        plan.validate().unwrap();

        let error = optimize_legacy_unreachable(&plan, MpcachePolicy::paper_profile()).unwrap_err();
        assert!(matches!(error, MpcacheError::InvalidPlan(_)));
        assert!(error.to_string().contains("has no previous selection"));
    }
}
