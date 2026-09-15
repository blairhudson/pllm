use super::{
    DecoderGraph, DecoderMode, DecoderPlan, DecoderWorkload, ModelError, ModelOperation,
    ModelOperator, StateKind, StateTensor,
};
use pllm_types::canonical_digest;
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};

const ADAPTER: &str = "pllm.phi4_mini.v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Phi4Config {
    model_type: String,
    hidden_size: u64,
    intermediate_size: u64,
    num_hidden_layers: u32,
    num_attention_heads: u64,
    num_key_value_heads: u64,
    vocab_size: u64,
    max_position_embeddings: u64,
    original_max_position_embeddings: u64,
    hidden_act: String,
    #[serde(deserialize_with = "decimal_string")]
    rms_norm_eps: String,
    #[serde(deserialize_with = "whole_number")]
    rope_theta: u64,
    #[serde(deserialize_with = "decimal_string")]
    partial_rotary_factor: String,
    tie_word_embeddings: bool,
    use_cache: bool,
    attention_bias: bool,
    #[serde(default)]
    qkv_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    lm_head_bias: bool,
    #[serde(default)]
    input_emb_layernorm: bool,
    #[serde(default)]
    attention_layer_norm: bool,
    #[serde(deserialize_with = "decimal_string")]
    attention_dropout: String,
    #[serde(deserialize_with = "decimal_string")]
    embd_pdrop: String,
    #[serde(deserialize_with = "decimal_string")]
    resid_pdrop: String,
    full_attn_mod: u64,
    sliding_window: u64,
    rope_scaling: LongRopeConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LongRopeConfig {
    #[serde(rename = "type")]
    scaling_type: String,
    #[serde(default)]
    rope_type: Option<String>,
    long_factor: Vec<Value>,
    short_factor: Vec<Value>,
}

impl Phi4Config {
    fn validate(&self) -> Result<(), ModelError> {
        if self.model_type != "phi3" {
            return Err(unsupported(
                "Phi-4 mini outer config must use model_type phi3",
            ));
        }
        if self.hidden_act != "silu" {
            return Err(unsupported("hidden_act must be silu"));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
        {
            return Err(invalid("model dimensions must be positive"));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(invalid(
                "hidden_size must be divisible by num_attention_heads",
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(invalid(
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }
        if self.original_max_position_embeddings == 0
            || self.original_max_position_embeddings > self.max_position_embeddings
        {
            return Err(invalid(
                "original_max_position_embeddings must be within the maximum",
            ));
        }
        if !self.tie_word_embeddings
            || !self.use_cache
            || self.attention_bias
            || self.qkv_bias
            || self.mlp_bias
            || self.lm_head_bias
            || self.input_emb_layernorm
            || self.attention_layer_norm
        {
            return Err(unsupported(
                "Phi-4 mini requires tied embeddings, cache, bias-free projections, and no extra embedding/attention norms",
            ));
        }
        zero(&self.attention_dropout, "attention_dropout")?;
        zero(&self.embd_pdrop, "embd_pdrop")?;
        zero(&self.resid_pdrop, "resid_pdrop")?;
        positive(&self.rms_norm_eps, "rms_norm_eps")?;
        if self.partial_rotary_factor != "0.75" {
            return Err(unsupported("partial_rotary_factor must be 0.75"));
        }
        if self.full_attn_mod != 1 || self.sliding_window < self.max_position_embeddings {
            return Err(unsupported(
                "Phi-4 mini requires full attention for every layer",
            ));
        }
        if self.rope_scaling.scaling_type != "longrope"
            || self
                .rope_scaling
                .rope_type
                .as_deref()
                .is_some_and(|value| value != "longrope")
        {
            return Err(unsupported("rope_scaling must use longrope"));
        }
        let head_dim = self.hidden_size / self.num_attention_heads;
        let rotary_dimensions = head_dim * 3 / 4;
        let expected_factors = usize::try_from(rotary_dimensions / 2)
            .map_err(|_| invalid("rotary dimension does not fit this platform"))?;
        validate_factors(
            &self.rope_scaling.short_factor,
            expected_factors,
            "short_factor",
        )?;
        validate_factors(
            &self.rope_scaling.long_factor,
            expected_factors,
            "long_factor",
        )?;
        Ok(())
    }
}

pub(super) fn lower_phi4_json(
    document: &Value,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    let config: Phi4Config = serde_json::from_value(document.clone())
        .map_err(|error| ModelError::InvalidJson(error.to_string()))?;
    config.validate()?;
    if workload.batch == 0 || workload.max_input_tokens == 0 || workload.max_new_tokens == 0 {
        return Err(invalid("workload bounds must be positive"));
    }
    let total = workload
        .max_input_tokens
        .checked_add(workload.max_new_tokens)
        .ok_or_else(|| invalid("workload token bound overflow"))?;
    if total > config.max_position_embeddings {
        return Err(invalid(
            "workload exceeds max_position_embeddings after decode",
        ));
    }
    let plan = DecoderPlan {
        schema_version: super::DECODER_PLAN_SCHEMA_VERSION.into(),
        model_family: "phi4_mini".into(),
        adapter: ADAPTER.into(),
        config_digest: canonical_digest("pllm.phi4_mini_config.v1", &config),
        transformations: Vec::new(),
        prefill: lower_graph(
            &config,
            workload,
            DecoderMode::Prefill,
            workload.max_input_tokens,
            workload.max_input_tokens,
        ),
        decode: lower_graph(&config, workload, DecoderMode::Decode, 1, total),
        token_feedback: true,
    };
    plan.validate()?;
    Ok(plan)
}

fn lower_graph(
    config: &Phi4Config,
    workload: DecoderWorkload,
    mode: DecoderMode,
    query_sequence: u64,
    maximum_key_sequence: u64,
) -> DecoderGraph {
    let batch = workload.batch;
    let hidden = config.hidden_size;
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = hidden / heads;
    let q_width = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let hidden_shape = vec![batch, query_sequence, hidden];
    let q_shape = vec![batch, heads, query_sequence, head_dim];
    let kv_query_shape = vec![batch, kv_heads, query_sequence, head_dim];
    let kv_state_shape = vec![batch, kv_heads, maximum_key_sequence, head_dim];
    let score_shape = vec![batch, heads, query_sequence, maximum_key_sequence];
    let mut operations = Vec::new();
    let mut state_inputs = Vec::new();
    let mut state_outputs = Vec::new();
    push(
        &mut operations,
        "token_lookup",
        ModelOperator::TokenLookup,
        &["input.tokens"],
        hidden_shape.clone(),
        json!({"weight": "model.embed_tokens.weight"}),
    );
    let mut hidden_input = "token_lookup".to_owned();
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("layer.{layer}");
        let residual = hidden_input.clone();
        let input_norm = format!("{prefix}.input_norm");
        push(
            &mut operations,
            &input_norm,
            ModelOperator::RmsNorm,
            &[&hidden_input],
            hidden_shape.clone(),
            json!({"epsilon": config.rms_norm_eps, "weight": format!("model.layers.{layer}.input_layernorm.weight")}),
        );
        let qkv_linear = format!("{prefix}.qkv_linear");
        push(
            &mut operations,
            &qkv_linear,
            ModelOperator::Linear,
            &[&input_norm],
            vec![batch, query_sequence, q_width + 2 * kv_width],
            json!({"weight": format!("model.layers.{layer}.self_attn.qkv_proj.weight"), "bias": Value::Null}),
        );
        let q_slice = format!("{prefix}.q_slice");
        let k_slice = format!("{prefix}.k_slice");
        let v_slice = format!("{prefix}.v_slice");
        slice(
            &mut operations,
            &q_slice,
            &qkv_linear,
            0,
            q_width,
            vec![batch, query_sequence, q_width],
        );
        slice(
            &mut operations,
            &k_slice,
            &qkv_linear,
            q_width,
            q_width + kv_width,
            vec![batch, query_sequence, kv_width],
        );
        slice(
            &mut operations,
            &v_slice,
            &qkv_linear,
            q_width + kv_width,
            q_width + 2 * kv_width,
            vec![batch, query_sequence, kv_width],
        );
        let q = format!("{prefix}.q_heads");
        let k = format!("{prefix}.k_heads");
        let v = format!("{prefix}.v_heads");
        reshape(&mut operations, &q, &q_slice, q_shape.clone());
        reshape(&mut operations, &k, &k_slice, kv_query_shape.clone());
        reshape(&mut operations, &v, &v_slice, kv_query_shape.clone());
        let rope_q = format!("{prefix}.rope_q");
        let rope_k = format!("{prefix}.rope_k");
        rotary(
            &mut operations,
            &rope_q,
            &q,
            q_shape.clone(),
            config,
            head_dim,
        );
        rotary(
            &mut operations,
            &rope_k,
            &k,
            kv_query_shape.clone(),
            config,
            head_dim,
        );
        let key_state = format!("state.layer.{layer}.key");
        let value_state = format!("state.layer.{layer}.value");
        state_inputs.push(state(
            &key_state,
            layer,
            StateKind::Key,
            kv_state_shape.clone(),
            maximum_key_sequence,
        ));
        state_inputs.push(state(
            &value_state,
            layer,
            StateKind::Value,
            kv_state_shape.clone(),
            maximum_key_sequence,
        ));
        let key_append = format!("{prefix}.key_append");
        let value_append = format!("{prefix}.value_append");
        append(
            &mut operations,
            &key_append,
            &key_state,
            &rope_k,
            kv_state_shape.clone(),
            StateKind::Key,
        );
        append(
            &mut operations,
            &value_append,
            &value_state,
            &v,
            kv_state_shape.clone(),
            StateKind::Value,
        );
        state_outputs.push(state(
            &key_append,
            layer,
            StateKind::Key,
            kv_state_shape.clone(),
            maximum_key_sequence,
        ));
        state_outputs.push(state(
            &value_append,
            layer,
            StateKind::Value,
            kv_state_shape.clone(),
            maximum_key_sequence,
        ));
        let scores = format!("{prefix}.attention_scores");
        let scaled = format!("{prefix}.attention_scale");
        let masked = format!("{prefix}.causal_mask");
        let probabilities = format!("{prefix}.softmax");
        let values = format!("{prefix}.attention_values");
        let attention_hidden = format!("{prefix}.attention_hidden");
        push(
            &mut operations,
            &scores,
            ModelOperator::AttentionScores,
            &[&rope_q, &key_append],
            score_shape.clone(),
            json!({"group_size": heads / kv_heads}),
        );
        push(
            &mut operations,
            &scaled,
            ModelOperator::AttentionScale,
            &[&scores],
            score_shape.clone(),
            json!({"head_dim": head_dim}),
        );
        push(
            &mut operations,
            &masked,
            ModelOperator::CausalMask,
            &[&scaled, "input.positions"],
            score_shape.clone(),
            json!({}),
        );
        push(
            &mut operations,
            &probabilities,
            ModelOperator::Softmax,
            &[&masked],
            score_shape.clone(),
            json!({"axis": -1}),
        );
        push(
            &mut operations,
            &values,
            ModelOperator::AttentionValues,
            &[&probabilities, &value_append],
            q_shape.clone(),
            json!({"group_size": heads / kv_heads}),
        );
        reshape(
            &mut operations,
            &attention_hidden,
            &values,
            hidden_shape.clone(),
        );
        let o = format!("{prefix}.o_proj");
        push(
            &mut operations,
            &o,
            ModelOperator::Linear,
            &[&attention_hidden],
            hidden_shape.clone(),
            json!({"weight": format!("model.layers.{layer}.self_attn.o_proj.weight"), "bias": Value::Null}),
        );
        let attention_residual = format!("{prefix}.attention_residual");
        push(
            &mut operations,
            &attention_residual,
            ModelOperator::ResidualAdd,
            &[&residual, &o],
            hidden_shape.clone(),
            json!({}),
        );
        let post_norm = format!("{prefix}.post_norm");
        push(
            &mut operations,
            &post_norm,
            ModelOperator::RmsNorm,
            &[&attention_residual],
            hidden_shape.clone(),
            json!({"epsilon": config.rms_norm_eps, "weight": format!("model.layers.{layer}.post_attention_layernorm.weight")}),
        );
        let gate_up = format!("{prefix}.gate_up_proj");
        push(
            &mut operations,
            &gate_up,
            ModelOperator::Linear,
            &[&post_norm],
            vec![batch, query_sequence, 2 * config.intermediate_size],
            json!({"weight": format!("model.layers.{layer}.mlp.gate_up_proj.weight"), "bias": Value::Null}),
        );
        let gate = format!("{prefix}.gate");
        let up = format!("{prefix}.up");
        slice(
            &mut operations,
            &gate,
            &gate_up,
            0,
            config.intermediate_size,
            vec![batch, query_sequence, config.intermediate_size],
        );
        slice(
            &mut operations,
            &up,
            &gate_up,
            config.intermediate_size,
            2 * config.intermediate_size,
            vec![batch, query_sequence, config.intermediate_size],
        );
        let activated = format!("{prefix}.silu");
        let multiplied = format!("{prefix}.gated_multiply");
        push(
            &mut operations,
            &activated,
            ModelOperator::Silu,
            &[&gate],
            vec![batch, query_sequence, config.intermediate_size],
            json!({}),
        );
        push(
            &mut operations,
            &multiplied,
            ModelOperator::Multiply,
            &[&activated, &up],
            vec![batch, query_sequence, config.intermediate_size],
            json!({}),
        );
        let down = format!("{prefix}.down_proj");
        push(
            &mut operations,
            &down,
            ModelOperator::Linear,
            &[&multiplied],
            hidden_shape.clone(),
            json!({"weight": format!("model.layers.{layer}.mlp.down_proj.weight"), "bias": Value::Null}),
        );
        let mlp_residual = format!("{prefix}.mlp_residual");
        push(
            &mut operations,
            &mlp_residual,
            ModelOperator::ResidualAdd,
            &[&attention_residual, &down],
            hidden_shape.clone(),
            json!({}),
        );
        hidden_input = mlp_residual;
    }
    push(
        &mut operations,
        "final_norm",
        ModelOperator::RmsNorm,
        &[&hidden_input],
        hidden_shape,
        json!({"epsilon": config.rms_norm_eps, "weight": "model.norm.weight"}),
    );
    push(
        &mut operations,
        "last_hidden",
        ModelOperator::LastToken,
        &["final_norm"],
        vec![batch, hidden],
        json!({"axis": 1}),
    );
    push(
        &mut operations,
        "output_head",
        ModelOperator::OutputHead,
        &["last_hidden"],
        vec![batch, config.vocab_size],
        json!({"weight": "model.embed_tokens.weight", "bias": Value::Null, "tied": true}),
    );
    push(
        &mut operations,
        "token_selection",
        ModelOperator::GreedyTokenSelection,
        &["output_head"],
        vec![batch],
        json!({}),
    );
    push(
        &mut operations,
        "token_feedback",
        ModelOperator::TokenFeedback,
        &["token_selection"],
        vec![batch, 1],
        json!({}),
    );
    DecoderGraph {
        mode,
        batch,
        query_sequence,
        maximum_key_sequence,
        operations,
        state_inputs,
        state_outputs,
        output: "token_feedback".into(),
    }
}

fn rotary(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    shape: Vec<u64>,
    config: &Phi4Config,
    head_dim: u64,
) {
    push(
        operations,
        id,
        ModelOperator::RotaryEmbedding,
        &[input, "input.positions"],
        shape,
        json!({
            "rope_type": "longrope", "theta": config.rope_theta,
            "rotary_dimensions": head_dim * 3 / 4,
            "original_max_position_embeddings": config.original_max_position_embeddings,
            "factor": config.max_position_embeddings / config.original_max_position_embeddings,
            "short_factor": config.rope_scaling.short_factor,
            "long_factor": config.rope_scaling.long_factor,
        }),
    );
}

fn slice(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    start: u64,
    end: u64,
    shape: Vec<u64>,
) {
    push(
        operations,
        id,
        ModelOperator::Slice,
        &[input],
        shape,
        json!({"axis": -1, "start": start, "end": end, "squeeze": false}),
    );
}

fn reshape(operations: &mut Vec<ModelOperation>, id: &str, input: &str, shape: Vec<u64>) {
    push(
        operations,
        id,
        ModelOperator::Reshape,
        &[input],
        shape,
        json!({"layout": "batch_heads_sequence_feature"}),
    );
}

fn append(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    state: &str,
    input: &str,
    shape: Vec<u64>,
    kind: StateKind,
) {
    push(
        operations,
        id,
        ModelOperator::KvCacheAppend,
        &[state, input],
        shape,
        json!({"state": state}),
    );
    operations.last_mut().expect("append was pushed").state_kind = Some(kind);
}

fn state(
    id: &str,
    layer: u32,
    kind: StateKind,
    shape: Vec<u64>,
    maximum_sequence: u64,
) -> StateTensor {
    StateTensor {
        id: id.into(),
        layer: Some(u64::from(layer)),
        kind,
        shape,
        maximum_sequence,
    }
}

fn push(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    operator: ModelOperator,
    inputs: &[&str],
    output_shape: Vec<u64>,
    attributes: Value,
) {
    operations.push(ModelOperation {
        id: id.into(),
        operator,
        layer: id
            .strip_prefix("layer.")
            .and_then(|rest| rest.split('.').next())
            .and_then(|value| value.parse().ok()),
        state_kind: None,
        inputs: inputs.iter().map(|value| (*value).into()).collect(),
        output_shape,
        attributes,
    });
}

fn validate_factors(values: &[Value], expected: usize, field: &str) -> Result<(), ModelError> {
    if values.len() != expected {
        return Err(unsupported(&format!(
            "rope_scaling.{field} must contain {expected} factors"
        )));
    }
    for value in values {
        let number = value
            .as_f64()
            .ok_or_else(|| invalid(&format!("rope_scaling.{field} entries must be numbers")))?;
        if !number.is_finite() || number <= 0.0 {
            return Err(invalid(&format!(
                "rope_scaling.{field} entries must be positive finite numbers"
            )));
        }
    }
    Ok(())
}

fn positive(value: &str, field: &str) -> Result<(), ModelError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid(&format!("{field} must be a positive finite decimal")))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(invalid(&format!(
            "{field} must be a positive finite decimal"
        )));
    }
    Ok(())
}

fn zero(value: &str, field: &str) -> Result<(), ModelError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid(&format!("{field} must be zero")))?;
    if !parsed.is_finite() || parsed != 0.0 {
        return Err(unsupported(&format!("{field} must be zero")));
    }
    Ok(())
}

fn invalid(message: &str) -> ModelError {
    ModelError::InvalidConfig(message.into())
}
fn unsupported(message: &str) -> ModelError {
    ModelError::Unsupported(message.into())
}

fn decimal_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    match Value::deserialize(deserializer)? {
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => Ok(value),
        _ => Err(de::Error::custom("expected a decimal number or string")),
    }
}

fn whole_number<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    match Value::deserialize(deserializer)? {
        Value::Number(value) => value
            .as_u64()
            .or_else(|| {
                value.as_f64().and_then(|item| {
                    (item.is_finite()
                        && item >= 0.0
                        && item < u64::MAX as f64
                        && item.fract() == 0.0)
                        .then_some(item as u64)
                })
            })
            .ok_or_else(|| de::Error::custom("expected a nonnegative integral number")),
        _ => Err(de::Error::custom("expected a nonnegative integral number")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest as _, Sha256};

    const OFFICIAL_CONFIG: &str =
        include_str!("../tests/fixtures/Phi-4-mini-instruct-cfbefac-config.json");

    #[test]
    fn source_lock_matches_pinned_phi4_mini_revision() {
        assert_eq!(
            format!("{:x}", Sha256::digest(OFFICIAL_CONFIG)),
            "ac65d86061d3d0d704ee2511fd0eb8713ef19eb6eedba17c3080a4165d5b933b"
        );
    }

    #[test]
    fn lowers_phi4_mini_fused_projections_and_longrope() {
        let plan = lower_phi4_json(
            &serde_json::from_str(OFFICIAL_CONFIG).unwrap(),
            DecoderWorkload {
                batch: 2,
                max_input_tokens: 128,
                max_new_tokens: 32,
            },
        )
        .unwrap();

        assert_eq!(plan.model_family, "phi4_mini");
        assert_eq!(plan.adapter, ADAPTER);
        assert_eq!(plan.prefill.state_outputs.len(), 64);
        let qkv = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.id == "layer.0.qkv_linear")
            .unwrap();
        assert_eq!(qkv.output_shape, vec![2, 128, 5120]);
        assert_eq!(qkv.attributes["bias"], Value::Null);
        let rotary = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.id == "layer.0.rope_q")
            .unwrap();
        assert_eq!(rotary.attributes["rope_type"], "longrope");
        assert_eq!(rotary.attributes["rotary_dimensions"], 96);
        assert_eq!(
            rotary.attributes["long_factor"].as_array().unwrap().len(),
            48
        );
        assert!(plan
            .prefill
            .operations
            .iter()
            .any(|op| { op.id == "layer.0.silu" && op.operator == ModelOperator::Silu }));
        let head = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.id == "output_head")
            .unwrap();
        assert_eq!(head.attributes["weight"], "model.embed_tokens.weight");
        assert_eq!(head.attributes["tied"], true);
    }

    #[test]
    fn rejects_phi_without_longrope() {
        let mut config: Value = serde_json::from_str(OFFICIAL_CONFIG).unwrap();
        config["rope_scaling"] = Value::Null;
        let error = lower_phi4_json(
            &config,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 4,
                max_new_tokens: 2,
            },
        )
        .unwrap_err();
        assert!(matches!(error, ModelError::InvalidJson(_)));
    }
}
