//! Model configuration validation and model-aware semantic lowering.

use pllm_types::{canonical_digest, Digest};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

mod gemma4;
mod phi4;
mod qwen35;

pub const DECODER_PLAN_SCHEMA_VERSION: &str = "pllm.decoder_plan.v1";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct QwenConfig {
    pub model_type: String,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub vocab_size: u64,
    pub max_position_embeddings: u64,
    pub hidden_act: String,
    #[serde(deserialize_with = "decimal_string")]
    pub rms_norm_eps: String,
    #[serde(deserialize_with = "integral_u64")]
    pub rope_theta: u64,
    pub tie_word_embeddings: bool,
}

impl QwenConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, ModelError> {
        serde_json::from_slice(bytes).map_err(|error| ModelError::InvalidJson(error.to_string()))
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.model_type != "qwen2" {
            return Err(ModelError::Unsupported(format!(
                "model_type {} is not qwen2",
                self.model_type
            )));
        }
        if self.hidden_act != "silu" {
            return Err(ModelError::Unsupported(format!(
                "hidden_act {} is not silu",
                self.hidden_act
            )));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
        {
            return Err(ModelError::InvalidConfig(
                "Qwen dimensions must be nonzero".into(),
            ));
        }
        if self.hidden_size % u64::from(self.num_attention_heads) != 0 {
            return Err(ModelError::InvalidConfig(
                "hidden_size must be divisible by num_attention_heads".into(),
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(ModelError::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads".into(),
            ));
        }
        let epsilon = self.rms_norm_eps.parse::<f64>().map_err(|_| {
            ModelError::InvalidConfig("rms_norm_eps must be a positive finite decimal".into())
        })?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ModelError::InvalidConfig(
                "rms_norm_eps must be a positive finite decimal".into(),
            ));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> u64 {
        self.hidden_size / u64::from(self.num_attention_heads)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Qwen3Config {
    model_type: String,
    hidden_size: u64,
    intermediate_size: u64,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    vocab_size: u64,
    max_position_embeddings: u64,
    head_dim: u64,
    hidden_act: String,
    #[serde(deserialize_with = "decimal_string")]
    rms_norm_eps: String,
    #[serde(deserialize_with = "integral_u64")]
    rope_theta: u64,
    tie_word_embeddings: bool,
    attention_bias: bool,
    #[serde(deserialize_with = "decimal_string")]
    attention_dropout: String,
    rope_scaling: Option<Value>,
    sliding_window: Option<u64>,
    use_sliding_window: bool,
    use_cache: bool,
    max_window_layers: u32,
    layer_types: Vec<String>,
}

impl Qwen3Config {
    fn from_json(bytes: &[u8]) -> Result<Self, ModelError> {
        serde_json::from_slice(bytes).map_err(|error| ModelError::InvalidJson(error.to_string()))
    }

    fn validate(&self) -> Result<(), ModelError> {
        if self.model_type != "qwen3" {
            return Err(ModelError::Unsupported(format!(
                "model_type {} is not qwen3",
                self.model_type
            )));
        }
        if self.hidden_act != "silu" {
            return Err(ModelError::Unsupported(format!(
                "hidden_act {} is not silu",
                self.hidden_act
            )));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
            || self.head_dim == 0
        {
            return Err(ModelError::InvalidConfig(
                "Qwen3 dimensions must be nonzero".into(),
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(ModelError::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads".into(),
            ));
        }
        positive_decimal(&self.rms_norm_eps, "rms_norm_eps")?;
        zero_decimal(&self.attention_dropout, "attention_dropout")?;
        if self.attention_bias {
            return Err(ModelError::Unsupported(
                "Qwen3 attention_bias must be false".into(),
            ));
        }
        if self.rope_scaling.is_some() {
            return Err(ModelError::Unsupported(
                "Qwen3 rope_scaling must be null".into(),
            ));
        }
        if self.sliding_window.is_some() || self.use_sliding_window {
            return Err(ModelError::Unsupported(
                "Qwen3 sliding attention is not supported by this adapter".into(),
            ));
        }
        if !self.use_cache {
            return Err(ModelError::Unsupported(
                "Qwen3 use_cache must be true".into(),
            ));
        }
        if self.max_window_layers != self.num_hidden_layers
            || self.layer_types.len() != self.num_hidden_layers as usize
            || self
                .layer_types
                .iter()
                .any(|layer_type| layer_type != "full_attention")
        {
            return Err(ModelError::Unsupported(
                "Qwen3 adapter requires one full_attention entry per layer".into(),
            ));
        }
        Ok(())
    }
}

struct DenseDecoderConfig<'a> {
    model_family: &'a str,
    adapter: &'a str,
    hidden_size: u64,
    intermediate_size: u64,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    vocab_size: u64,
    max_position_embeddings: u64,
    head_dim: u64,
    rms_norm_eps: &'a str,
    rope_theta: u64,
    tie_word_embeddings: bool,
    attention_bias: bool,
    qk_norm: bool,
    config_digest: Digest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderMode {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelOperator {
    TokenLookup,
    Reshape,
    RmsNorm,
    Linear,
    RotaryEmbedding,
    KvCacheAppend,
    AttentionScores,
    AttentionScale,
    CausalMask,
    Softmax,
    AttentionValues,
    ResidualAdd,
    Silu,
    Multiply,
    OutputHead,
    LastToken,
    GreedyTokenSelection,
    TokenFeedback,
    AttentionImportance,
    SecureTopK,
    SecureGather,
    ClusterBounds,
    ClusterSimilarity,
    SharedIndices,
    Scale,
    Slice,
    GeluTanh,
    Softcap,
    CacheSuffix,
    Permute,
    Sigmoid,
    GatedDeltaDecay,
    CausalConvolution,
    ConvolutionStateUpdate,
    GatedDeltaRule,
    GatedDeltaStateUpdate,
    RmsNormGated,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOperation {
    pub id: String,
    pub operator: ModelOperator,
    pub layer: Option<u64>,
    pub state_kind: Option<StateKind>,
    pub inputs: Vec<String>,
    pub output_shape: Vec<u64>,
    pub attributes: Value,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StateKind {
    Key,
    Value,
    Recurrent,
    Convolution,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateTensor {
    pub id: String,
    pub layer: Option<u64>,
    pub kind: StateKind,
    pub shape: Vec<u64>,
    pub maximum_sequence: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderGraph {
    pub mode: DecoderMode,
    pub batch: u64,
    pub query_sequence: u64,
    pub maximum_key_sequence: u64,
    pub operations: Vec<ModelOperation>,
    pub state_inputs: Vec<StateTensor>,
    pub state_outputs: Vec<StateTensor>,
    pub output: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderPlan {
    pub schema_version: String,
    pub model_family: String,
    pub adapter: String,
    pub config_digest: Digest,
    pub transformations: Vec<AppliedMethod>,
    pub prefill: DecoderGraph,
    pub decode: DecoderGraph,
    pub token_feedback: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AppliedMethod {
    pub component: String,
    pub implementation: String,
    pub method_id: String,
    pub input_digest: Digest,
    pub configuration_digest: Digest,
}

impl DecoderPlan {
    pub fn digest(&self) -> Digest {
        canonical_digest("pllm.decoder_plan.v1", self)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.schema_version != DECODER_PLAN_SCHEMA_VERSION {
            return Err(ModelError::Incomplete(format!(
                "unsupported decoder plan schema_version {}",
                self.schema_version
            )));
        }
        if self.prefill.mode != DecoderMode::Prefill || self.decode.mode != DecoderMode::Decode {
            return Err(ModelError::Incomplete(
                "decoder plan prefill/decode modes are misplaced".into(),
            ));
        }
        validate_graph(&self.prefill)?;
        validate_graph(&self.decode)?;
        if !self.token_feedback {
            return Err(ModelError::Incomplete(
                "decoder plan omits token feedback".into(),
            ));
        }
        for graph in [&self.prefill, &self.decode] {
            if graph.operations.last().map(|operation| operation.operator)
                != Some(ModelOperator::TokenFeedback)
            {
                return Err(ModelError::Incomplete(
                    "decoder graph final operation is not token feedback".into(),
                ));
            }
        }
        if self.model_family == "gemma4_text" {
            validate_state_transition(self)?;
        }
        Ok(())
    }
}

pub fn lower_model_json(
    bytes: &[u8],
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    let document: Value = serde_json::from_slice(bytes)
        .map_err(|error| ModelError::InvalidJson(error.to_string()))?;
    let model_type = document
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| ModelError::InvalidConfig("model_type must be a string".into()))?;
    match model_type {
        "qwen2" => lower_qwen_decoder(&QwenConfig::from_json(bytes)?, workload),
        "qwen3" => lower_qwen3_decoder(&Qwen3Config::from_json(bytes)?, workload),
        "gemma4" => gemma4::lower_gemma4_json(document, workload),
        "phi3" => phi4::lower_phi4_json(&document, workload),
        "qwen3_5" => qwen35::lower_qwen35_json(&document, workload),
        other => Err(ModelError::Unsupported(format!(
            "model_type {other} has no decoder adapter"
        ))),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderWorkload {
    pub batch: u64,
    pub max_input_tokens: u64,
    pub max_new_tokens: u64,
}

pub fn lower_qwen_decoder(
    config: &QwenConfig,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    config.validate()?;
    let digest = canonical_digest("pllm.qwen_config.v1", config);
    lower_dense_decoder(
        &DenseDecoderConfig {
            model_family: &config.model_type,
            adapter: "pllm.qwen2.v1",
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            vocab_size: config.vocab_size,
            max_position_embeddings: config.max_position_embeddings,
            head_dim: config.head_dim(),
            rms_norm_eps: &config.rms_norm_eps,
            rope_theta: config.rope_theta,
            tie_word_embeddings: config.tie_word_embeddings,
            attention_bias: true,
            qk_norm: false,
            config_digest: digest,
        },
        workload,
    )
}

fn lower_qwen3_decoder(
    config: &Qwen3Config,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    config.validate()?;
    let digest = canonical_digest("pllm.qwen3_config.v1", config);
    lower_dense_decoder(
        &DenseDecoderConfig {
            model_family: &config.model_type,
            adapter: "pllm.qwen3.v1",
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            vocab_size: config.vocab_size,
            max_position_embeddings: config.max_position_embeddings,
            head_dim: config.head_dim,
            rms_norm_eps: &config.rms_norm_eps,
            rope_theta: config.rope_theta,
            tie_word_embeddings: config.tie_word_embeddings,
            attention_bias: false,
            qk_norm: true,
            config_digest: digest,
        },
        workload,
    )
}

fn lower_dense_decoder(
    config: &DenseDecoderConfig<'_>,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    if workload.batch == 0 || workload.max_input_tokens == 0 || workload.max_new_tokens == 0 {
        return Err(ModelError::InvalidConfig(
            "decoder workload bounds must be nonzero".into(),
        ));
    }
    let total = workload
        .max_input_tokens
        .checked_add(workload.max_new_tokens)
        .ok_or_else(|| ModelError::InvalidConfig("decoder token bound overflowed".into()))?;
    if total > config.max_position_embeddings {
        return Err(ModelError::InvalidConfig(format!(
            "decoder requires {total} positions but model permits {}",
            config.max_position_embeddings
        )));
    }
    let plan = DecoderPlan {
        schema_version: DECODER_PLAN_SCHEMA_VERSION.into(),
        model_family: config.model_family.into(),
        adapter: config.adapter.into(),
        config_digest: config.config_digest.clone(),
        transformations: Vec::new(),
        prefill: lower_graph(
            config,
            DecoderMode::Prefill,
            workload.batch,
            workload.max_input_tokens,
            workload.max_input_tokens,
        ),
        decode: lower_graph(config, DecoderMode::Decode, workload.batch, 1, total - 1),
        token_feedback: true,
    };
    plan.validate()?;
    Ok(plan)
}

fn lower_graph(
    config: &DenseDecoderConfig<'_>,
    mode: DecoderMode,
    batch: u64,
    query_sequence: u64,
    maximum_key_sequence: u64,
) -> DecoderGraph {
    let hidden = config.hidden_size;
    let heads = u64::from(config.num_attention_heads);
    let kv_heads = u64::from(config.num_key_value_heads);
    let head_dim = config.head_dim;
    let attention_width = heads * head_dim;
    let hidden_shape = vec![batch, query_sequence, hidden];
    let attention_hidden_shape = vec![batch, query_sequence, attention_width];
    let q_shape = vec![batch, heads, query_sequence, head_dim];
    let kv_query_shape = vec![batch, kv_heads, query_sequence, head_dim];
    let kv_state_shape = vec![batch, kv_heads, maximum_key_sequence, head_dim];
    let score_shape = vec![batch, heads, query_sequence, maximum_key_sequence];
    let mut operations = Vec::new();
    push(
        &mut operations,
        "token_lookup",
        ModelOperator::TokenLookup,
        &["input.tokens"],
        hidden_shape.clone(),
        json!({"weight": "model.embed_tokens.weight"}),
    );
    let mut hidden_input = "token_lookup".to_owned();
    let mut state_inputs = Vec::new();
    let mut state_outputs = Vec::new();
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
        let q_linear = format!("{prefix}.q_linear");
        let k_linear = format!("{prefix}.k_linear");
        let v_linear = format!("{prefix}.v_linear");
        let q = format!("{prefix}.q_heads");
        let k = format!("{prefix}.k_heads");
        let v = format!("{prefix}.v_heads");
        linear(
            &mut operations,
            &q_linear,
            &input_norm,
            attention_hidden_shape.clone(),
            layer,
            "q_proj",
            config.attention_bias,
        );
        linear(
            &mut operations,
            &k_linear,
            &input_norm,
            vec![batch, query_sequence, kv_heads * head_dim],
            layer,
            "k_proj",
            config.attention_bias,
        );
        linear(
            &mut operations,
            &v_linear,
            &input_norm,
            vec![batch, query_sequence, kv_heads * head_dim],
            layer,
            "v_proj",
            config.attention_bias,
        );
        push(
            &mut operations,
            &q,
            ModelOperator::Reshape,
            &[&q_linear],
            q_shape.clone(),
            json!({"layout": "batch_heads_sequence_feature"}),
        );
        push(
            &mut operations,
            &k,
            ModelOperator::Reshape,
            &[&k_linear],
            kv_query_shape.clone(),
            json!({"layout": "batch_heads_sequence_feature"}),
        );
        push(
            &mut operations,
            &v,
            ModelOperator::Reshape,
            &[&v_linear],
            kv_query_shape.clone(),
            json!({"layout": "batch_heads_sequence_feature"}),
        );
        let q_rope_input = if config.qk_norm {
            let q_norm = format!("{prefix}.q_norm");
            push(
                &mut operations,
                &q_norm,
                ModelOperator::RmsNorm,
                &[&q],
                q_shape.clone(),
                json!({"epsilon": config.rms_norm_eps, "weight": format!("model.layers.{layer}.self_attn.q_norm.weight")}),
            );
            q_norm
        } else {
            q.clone()
        };
        let k_rope_input = if config.qk_norm {
            let k_norm = format!("{prefix}.k_norm");
            push(
                &mut operations,
                &k_norm,
                ModelOperator::RmsNorm,
                &[&k],
                kv_query_shape.clone(),
                json!({"epsilon": config.rms_norm_eps, "weight": format!("model.layers.{layer}.self_attn.k_norm.weight")}),
            );
            k_norm
        } else {
            k.clone()
        };
        let rope_q = format!("{prefix}.rope_q");
        let rope_k = format!("{prefix}.rope_k");
        push(
            &mut operations,
            &rope_q,
            ModelOperator::RotaryEmbedding,
            &[&q_rope_input, "input.positions"],
            q_shape.clone(),
            json!({"theta": config.rope_theta}),
        );
        push(
            &mut operations,
            &rope_k,
            ModelOperator::RotaryEmbedding,
            &[&k_rope_input, "input.positions"],
            kv_query_shape.clone(),
            json!({"theta": config.rope_theta}),
        );
        let key_state = format!("state.layer.{layer}.key");
        let value_state = format!("state.layer.{layer}.value");
        state_inputs.push(StateTensor {
            id: key_state.clone(),
            layer: Some(u64::from(layer)),
            kind: StateKind::Key,
            shape: kv_state_shape.clone(),
            maximum_sequence: maximum_key_sequence,
        });
        state_inputs.push(StateTensor {
            id: value_state.clone(),
            layer: Some(u64::from(layer)),
            kind: StateKind::Value,
            shape: kv_state_shape.clone(),
            maximum_sequence: maximum_key_sequence,
        });
        let key_append = format!("{prefix}.key_append");
        let value_append = format!("{prefix}.value_append");
        push(
            &mut operations,
            &key_append,
            ModelOperator::KvCacheAppend,
            &[&key_state, &rope_k],
            kv_state_shape.clone(),
            json!({"state": key_state}),
        );
        operations
            .last_mut()
            .expect("key append was pushed")
            .state_kind = Some(StateKind::Key);
        push(
            &mut operations,
            &value_append,
            ModelOperator::KvCacheAppend,
            &[&value_state, &v],
            kv_state_shape.clone(),
            json!({"state": value_state}),
        );
        operations
            .last_mut()
            .expect("value append was pushed")
            .state_kind = Some(StateKind::Value);
        state_outputs.push(StateTensor {
            id: key_append.clone(),
            layer: Some(u64::from(layer)),
            kind: StateKind::Key,
            shape: kv_state_shape.clone(),
            maximum_sequence: maximum_key_sequence,
        });
        state_outputs.push(StateTensor {
            id: value_append.clone(),
            layer: Some(u64::from(layer)),
            kind: StateKind::Value,
            shape: kv_state_shape.clone(),
            maximum_sequence: maximum_key_sequence,
        });
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
        push(
            &mut operations,
            &attention_hidden,
            ModelOperator::Reshape,
            &[&values],
            attention_hidden_shape.clone(),
            json!({"layout": "batch_sequence_hidden"}),
        );
        let o = format!("{prefix}.o_proj");
        linear(
            &mut operations,
            &o,
            &attention_hidden,
            hidden_shape.clone(),
            layer,
            "o_proj",
            false,
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
        let gate = format!("{prefix}.gate_proj");
        let up = format!("{prefix}.up_proj");
        let silu = format!("{prefix}.silu");
        let multiplied = format!("{prefix}.gated_multiply");
        let down = format!("{prefix}.down_proj");
        let intermediate_shape = vec![batch, query_sequence, config.intermediate_size];
        linear(
            &mut operations,
            &gate,
            &post_norm,
            intermediate_shape.clone(),
            layer,
            "gate_proj",
            false,
        );
        linear(
            &mut operations,
            &up,
            &post_norm,
            intermediate_shape.clone(),
            layer,
            "up_proj",
            false,
        );
        push(
            &mut operations,
            &silu,
            ModelOperator::Silu,
            &[&gate],
            intermediate_shape.clone(),
            json!({}),
        );
        push(
            &mut operations,
            &multiplied,
            ModelOperator::Multiply,
            &[&silu, &up],
            intermediate_shape,
            json!({}),
        );
        linear(
            &mut operations,
            &down,
            &multiplied,
            hidden_shape.clone(),
            layer,
            "down_proj",
            false,
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
        vec![batch, config.hidden_size],
        json!({"axis": 1}),
    );
    push(
        &mut operations,
        "output_head",
        ModelOperator::OutputHead,
        &["last_hidden"],
        vec![batch, config.vocab_size],
        json!({"weight": if config.tie_word_embeddings { "model.embed_tokens.weight" } else { "lm_head.weight" }}),
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

fn linear(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    shape: Vec<u64>,
    layer: u32,
    name: &str,
    has_bias: bool,
) {
    let section = if matches!(name, "gate_proj" | "up_proj" | "down_proj") {
        "mlp"
    } else {
        "self_attn"
    };
    let stem = format!("model.layers.{layer}.{section}.{name}");
    let bias = has_bias.then(|| format!("{stem}.bias"));
    push(
        operations,
        id,
        ModelOperator::Linear,
        &[input],
        shape,
        json!({"weight": format!("{stem}.weight"), "bias": bias}),
    );
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

fn validate_graph(graph: &DecoderGraph) -> Result<(), ModelError> {
    if graph.batch == 0 || graph.query_sequence == 0 || graph.maximum_key_sequence == 0 {
        return Err(ModelError::Incomplete(
            "graph bounds must be nonzero".into(),
        ));
    }
    if graph.maximum_key_sequence < graph.query_sequence {
        return Err(ModelError::Incomplete(
            "maximum key sequence is shorter than query sequence".into(),
        ));
    }
    if graph.mode == DecoderMode::Decode && graph.query_sequence != 1 {
        return Err(ModelError::Incomplete(
            "decode graph query sequence must equal one".into(),
        ));
    }
    let mut available: BTreeSet<String> = [
        "input.tokens".into(),
        "input.positions".into(),
        "input.attention_mask".into(),
        "input.sequence_lengths".into(),
    ]
    .into_iter()
    .collect();
    let mut shapes = BTreeMap::<String, Vec<u64>>::new();
    let mut state_input_ids = BTreeSet::new();
    let mut state_input_slots = BTreeSet::new();
    for state in &graph.state_inputs {
        validate_state(state)?;
        validate_state_bounds(graph, state)?;
        if !state_input_ids.insert(state.id.clone())
            || !state_input_slots.insert((state.layer, state.kind))
            || available.contains(&state.id)
        {
            return Err(ModelError::Incomplete(format!(
                "duplicate state input id {}",
                state.id
            )));
        }
        available.insert(state.id.clone());
        shapes.insert(state.id.clone(), state.shape.clone());
    }
    for operation in &graph.operations {
        if operation.id.is_empty() || available.contains(&operation.id) {
            return Err(ModelError::Incomplete(format!(
                "duplicate or empty operation id {}",
                operation.id
            )));
        }
        if operation.output_shape.is_empty() || operation.output_shape.contains(&0) {
            return Err(ModelError::Incomplete(format!(
                "operation {} has an empty output dimension",
                operation.id
            )));
        }
        for input in &operation.inputs {
            if !available.contains(input) {
                return Err(ModelError::Incomplete(format!(
                    "operation {} references unavailable input {input}",
                    operation.id
                )));
            }
        }
        validate_operation(graph.mode, operation, &shapes)?;
        available.insert(operation.id.clone());
        shapes.insert(operation.id.clone(), operation.output_shape.clone());
    }
    let mut state_output_ids = BTreeSet::new();
    let mut state_output_slots = BTreeSet::new();
    for state in &graph.state_outputs {
        validate_state(state)?;
        validate_state_bounds(graph, state)?;
        if state_input_ids.contains(&state.id)
            || !state_output_ids.insert(state.id.clone())
            || !state_output_slots.insert((state.layer, state.kind))
        {
            return Err(ModelError::Incomplete(format!(
                "duplicate state output id {}",
                state.id
            )));
        }
        if !available.contains(&state.id) {
            return Err(ModelError::Incomplete(format!(
                "state output {} is unavailable",
                state.id
            )));
        }
    }
    if !available.contains(&graph.output)
        || graph.operations.last().map(|operation| &operation.id) != Some(&graph.output)
    {
        return Err(ModelError::Incomplete(
            "graph output is not its final operation".into(),
        ));
    }
    Ok(())
}

fn validate_state(state: &StateTensor) -> Result<(), ModelError> {
    if state.id.is_empty() || state.shape.is_empty() || state.shape.contains(&0) {
        return Err(ModelError::Incomplete(format!(
            "state {} has invalid shape or id",
            state.id
        )));
    }
    if matches!(state.kind, StateKind::Key | StateKind::Value) {
        if state.shape.len() != 4 {
            return Err(ModelError::Incomplete(format!(
                "KV state {} must have rank four",
                state.id
            )));
        }
        if state.maximum_sequence > state.shape[2] {
            return Err(ModelError::Incomplete(format!(
                "state {} maximum_sequence exceeds its sequence dimension",
                state.id
            )));
        }
    }
    Ok(())
}

fn validate_state_bounds(graph: &DecoderGraph, state: &StateTensor) -> Result<(), ModelError> {
    if state.shape.first().copied() != Some(graph.batch) {
        return Err(ModelError::Incomplete(format!(
            "state {} exceeds graph bounds",
            state.id
        )));
    }
    Ok(())
}

fn validate_operation(
    graph_mode: DecoderMode,
    operation: &ModelOperation,
    shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<(), ModelError> {
    if !operation.attributes.is_object() {
        return Err(ModelError::Incomplete(format!(
            "operation {} attributes must be an object",
            operation.id
        )));
    }
    let expected_arity = match operation.operator {
        ModelOperator::TokenLookup => Some(1),
        ModelOperator::Reshape
        | ModelOperator::RmsNorm
        | ModelOperator::Linear
        | ModelOperator::AttentionScale
        | ModelOperator::Softmax
        | ModelOperator::Silu
        | ModelOperator::GeluTanh
        | ModelOperator::OutputHead
        | ModelOperator::GreedyTokenSelection
        | ModelOperator::TokenFeedback
        | ModelOperator::Scale
        | ModelOperator::Slice
        | ModelOperator::Softcap
        | ModelOperator::Permute
        | ModelOperator::Sigmoid
        | ModelOperator::GatedDeltaDecay => Some(1),
        ModelOperator::RotaryEmbedding
        | ModelOperator::AttentionScores
        | ModelOperator::AttentionValues
        | ModelOperator::ResidualAdd
        | ModelOperator::Multiply
        | ModelOperator::CausalConvolution
        | ModelOperator::ConvolutionStateUpdate
        | ModelOperator::RmsNormGated => Some(2),
        ModelOperator::GatedDeltaRule | ModelOperator::GatedDeltaStateUpdate => Some(6),
        ModelOperator::LastToken => None,
        ModelOperator::KvCacheAppend | ModelOperator::CausalMask | ModelOperator::CacheSuffix => {
            None
        }
        _ => None,
    };
    if expected_arity.is_some_and(|arity| operation.inputs.len() != arity) {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid input arity",
            operation.id
        )));
    }
    let attributes = operation.attributes.as_object().expect("checked above");
    let required = match operation.operator {
        ModelOperator::RmsNorm => &["epsilon"][..],
        ModelOperator::Linear | ModelOperator::TokenLookup | ModelOperator::OutputHead => {
            &["weight"][..]
        }
        ModelOperator::RotaryEmbedding => &["theta"][..],
        ModelOperator::KvCacheAppend => &["state"][..],
        ModelOperator::AttentionScores | ModelOperator::AttentionValues => &["group_size"][..],
        ModelOperator::AttentionScale => {
            if attributes.contains_key("factor") {
                &["factor"][..]
            } else {
                &["head_dim"][..]
            }
        }
        ModelOperator::Softmax | ModelOperator::LastToken => &["axis"][..],
        ModelOperator::Slice => &["axis", "start", "end", "squeeze"][..],
        ModelOperator::Scale => &["factor"][..],
        ModelOperator::Softcap => &["cap"][..],
        ModelOperator::CacheSuffix => &["axis", "maximum_sequence"][..],
        ModelOperator::Permute => &["permutation"][..],
        ModelOperator::GatedDeltaDecay => &["a_log", "dt_bias", "formula"][..],
        ModelOperator::CausalConvolution | ModelOperator::ConvolutionStateUpdate => {
            &["weight", "kernel_size"][..]
        }
        ModelOperator::GatedDeltaRule | ModelOperator::GatedDeltaStateUpdate => {
            &["qk_l2_normalize", "query_scale"][..]
        }
        ModelOperator::RmsNormGated => &["epsilon", "weight", "activation"][..],
        _ => &[][..],
    };
    if required.iter().any(|key| !attributes.contains_key(*key)) {
        return Err(ModelError::Incomplete(format!(
            "operation {} omits required attributes",
            operation.id
        )));
    }
    if matches!(
        operation.operator,
        ModelOperator::Softmax | ModelOperator::LastToken | ModelOperator::Slice
    ) {
        let axis = attributes
            .get("axis")
            .and_then(Value::as_i64)
            .ok_or_else(|| {
                ModelError::Incomplete(format!("operation {} has invalid axis", operation.id))
            })?;
        let rank = i64::try_from(operation.output_shape.len()).map_err(|_| {
            ModelError::Incomplete(format!("operation {} rank overflowed", operation.id))
        })?;
        if axis < -rank || axis >= rank {
            return Err(ModelError::Incomplete(format!(
                "operation {} axis is out of range",
                operation.id
            )));
        }
    }
    match operation.operator {
        ModelOperator::Reshape => validate_reshape(operation, shapes)?,
        ModelOperator::Permute => validate_permute(operation, shapes)?,
        ModelOperator::KvCacheAppend => validate_cache_update(graph_mode, operation, shapes)?,
        ModelOperator::CacheSuffix => validate_cache_suffix(operation, shapes)?,
        ModelOperator::CausalMask => validate_causal_mask(operation)?,
        ModelOperator::AttentionScores | ModelOperator::AttentionValues => {
            validate_attention_operation(operation, shapes)?;
        }
        ModelOperator::LastToken => {
            if !matches!(operation.inputs.len(), 1 | 2) {
                return Err(ModelError::Incomplete(format!(
                    "operation {} has invalid input arity",
                    operation.id
                )));
            }
            if operation.inputs.len() == 2
                && (operation.inputs[1] != "input.sequence_lengths"
                    || attributes.get("selection").and_then(Value::as_str) != Some("last_valid")
                    || attributes
                        .get("valid_lengths_input")
                        .and_then(Value::as_str)
                        != Some("input.sequence_lengths"))
            {
                return Err(ModelError::Incomplete(format!(
                    "operation {} has invalid length-aware selection",
                    operation.id
                )));
            }
        }
        _ => {}
    }
    if operation.operator == ModelOperator::CacheSuffix {
        let maximum = attributes
            .get("maximum_sequence")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if maximum == 0 || operation.output_shape.len() != 4 || operation.output_shape[2] != maximum
        {
            return Err(ModelError::Incomplete(format!(
                "operation {} has invalid cache suffix bound",
                operation.id
            )));
        }
    }
    if matches!(
        operation.operator,
        ModelOperator::Scale | ModelOperator::AttentionScale
    ) && !attributes.get("factor").is_some_and(Value::is_object)
        && attributes.contains_key("factor")
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid exact scale factor",
            operation.id
        )));
    }
    if operation.operator == ModelOperator::Slice {
        let start = attributes.get("start").and_then(Value::as_u64);
        let end = attributes.get("end").and_then(Value::as_u64);
        if !attributes.get("squeeze").is_some_and(Value::is_boolean)
            || start.zip(end).is_none_or(|(start, end)| start >= end)
        {
            return Err(ModelError::Incomplete(format!(
                "operation {} has invalid slice bounds",
                operation.id
            )));
        }
    }
    if operation.operator == ModelOperator::Softcap
        && attributes
            .get("cap")
            .and_then(Value::as_f64)
            .is_none_or(|cap| cap <= 0.0)
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid softcap",
            operation.id
        )));
    }
    Ok(())
}

fn validate_reshape(
    operation: &ModelOperation,
    shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<(), ModelError> {
    let Some(input_shape) = operation.inputs.first().and_then(|input| shapes.get(input)) else {
        return Ok(());
    };
    let input_elements = input_shape.iter().try_fold(1_u128, |size, dimension| {
        size.checked_mul(u128::from(*dimension))
    });
    let output_elements = operation
        .output_shape
        .iter()
        .try_fold(1_u128, |size, dimension| {
            size.checked_mul(u128::from(*dimension))
        });
    if input_elements.is_none() || input_elements != output_elements {
        return Err(ModelError::Incomplete(format!(
            "operation {} reshape element counts differ or overflow",
            operation.id
        )));
    }
    Ok(())
}

fn validate_permute(
    operation: &ModelOperation,
    shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<(), ModelError> {
    let permutation = operation
        .attributes
        .get("permutation")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ModelError::Incomplete(format!(
                "operation {} has invalid permutation",
                operation.id
            ))
        })?;
    let rank = operation.output_shape.len();
    if permutation.len() != rank {
        return Err(ModelError::Incomplete(format!(
            "operation {} permutation rank differs",
            operation.id
        )));
    }
    let mut axes = BTreeSet::new();
    let parsed = permutation
        .iter()
        .map(|axis| axis.as_u64().and_then(|axis| usize::try_from(axis).ok()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            ModelError::Incomplete(format!(
                "operation {} has invalid permutation",
                operation.id
            ))
        })?;
    if parsed
        .iter()
        .any(|axis| *axis >= rank || !axes.insert(*axis))
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} permutation is not a rank bijection",
            operation.id
        )));
    }
    if let Some(input_shape) = operation.inputs.first().and_then(|input| shapes.get(input)) {
        if input_shape.len() != rank
            || parsed.iter().enumerate().any(|(output_axis, input_axis)| {
                operation.output_shape[output_axis] != input_shape[*input_axis]
            })
        {
            return Err(ModelError::Incomplete(format!(
                "operation {} permutation shape is inconsistent",
                operation.id
            )));
        }
    }
    Ok(())
}

fn validate_attention_operation(
    operation: &ModelOperation,
    shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<(), ModelError> {
    if operation.output_shape.len() != 4 {
        return Err(ModelError::Incomplete(format!(
            "operation {} attention output must have rank four",
            operation.id
        )));
    }
    let Some(left) = operation.inputs.first().and_then(|input| shapes.get(input)) else {
        return Ok(());
    };
    let Some(right) = operation.inputs.get(1).and_then(|input| shapes.get(input)) else {
        return Ok(());
    };
    if left.len() != 4 || !matches!(right.len(), 4 | 5) {
        return Err(ModelError::Incomplete(format!(
            "operation {} attention input ranks are invalid",
            operation.id
        )));
    }
    let valid = match operation.operator {
        ModelOperator::AttentionScores => {
            let key_query = if right.len() == 5 { right[2] } else { left[2] };
            left[0] == right[0]
                && left[2] == key_query
                && left[3] == right[right.len() - 1]
                && operation.output_shape[..3] == left[..3]
        }
        ModelOperator::AttentionValues => {
            let value_query = if right.len() == 5 { right[2] } else { left[2] };
            left[0] == right[0]
                && left[2] == value_query
                && operation.output_shape == [left[0], left[1], left[2], right[right.len() - 1]]
        }
        _ => true,
    };
    if !valid {
        return Err(ModelError::Incomplete(format!(
            "operation {} attention shapes are inconsistent",
            operation.id
        )));
    }
    Ok(())
}

fn validate_cache_update(
    graph_mode: DecoderMode,
    operation: &ModelOperation,
    shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<(), ModelError> {
    let attributes = operation.attributes.as_object().expect("validated object");
    let Some(mode) = attributes.get("mode").and_then(Value::as_str) else {
        return if operation.inputs.len() == 2 {
            Ok(())
        } else {
            Err(ModelError::Incomplete(format!(
                "operation {} legacy cache append has invalid arity",
                operation.id
            )))
        };
    };
    let (expected_mode, expected_arity, current_index) = match graph_mode {
        DecoderMode::Prefill => ("initialize", 4, 0),
        DecoderMode::Decode => ("append", 5, 1),
    };
    if mode != expected_mode
        || operation.inputs.len() != expected_arity
        || operation.inputs[current_index + 1] != "input.positions"
        || operation.inputs[current_index + 2] != "input.attention_mask"
        || operation.inputs[current_index + 3] != "input.sequence_lengths"
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} cache mode or dependencies are invalid",
            operation.id
        )));
    }
    let state = attributes
        .get("state")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ModelError::Incomplete(format!("operation {} omits state slot", operation.id))
        })?;
    if graph_mode == DecoderMode::Decode && operation.inputs[0] != state {
        return Err(ModelError::Incomplete(format!(
            "operation {} does not consume its declared state slot",
            operation.id
        )));
    }
    let current_shape = shapes
        .get(&operation.inputs[current_index])
        .ok_or_else(|| {
            ModelError::Incomplete(format!(
                "operation {} current KV shape is unavailable",
                operation.id
            ))
        })?;
    if current_shape.len() != 4 || !matches!(operation.output_shape.len(), 4 | 5) {
        return Err(ModelError::Incomplete(format!(
            "operation {} cache ranks are invalid",
            operation.id
        )));
    }
    let state_capacity = attributes
        .get("state_capacity")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if state_capacity == 0 {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid state capacity",
            operation.id
        )));
    }
    for key in [
        "absolute_write_positions_input",
        "padding_mask_input",
        "valid_lengths_input",
        "attention_domain",
    ] {
        if !attributes.contains_key(key) {
            return Err(ModelError::Incomplete(format!(
                "operation {} omits cache descriptor {key}",
                operation.id
            )));
        }
    }
    Ok(())
}

fn validate_cache_suffix(
    operation: &ModelOperation,
    shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<(), ModelError> {
    let attributes = operation.attributes.as_object().expect("validated object");
    if operation.inputs.len() != 4
        || operation.inputs[1] != "input.positions"
        || operation.inputs[2] != "input.attention_mask"
        || operation.inputs[3] != "input.sequence_lengths"
        || operation.output_shape.len() != 4
        || operation
            .inputs
            .first()
            .and_then(|input| shapes.get(input))
            .is_none_or(|shape| shape.len() != 5)
        || attributes.get("axis").and_then(Value::as_u64) != Some(3)
        || attributes.get("output_axis").and_then(Value::as_u64) != Some(2)
        || attributes
            .get("absolute_write_positions_input")
            .and_then(Value::as_str)
            != Some("input.positions")
        || attributes.get("padding_mask_input").and_then(Value::as_str)
            != Some("input.attention_mask")
        || attributes
            .get("valid_lengths_input")
            .and_then(Value::as_str)
            != Some("input.sequence_lengths")
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid cache suffix contract",
            operation.id
        )));
    }
    Ok(())
}

fn validate_causal_mask(operation: &ModelOperation) -> Result<(), ModelError> {
    if operation.inputs.len() == 2 {
        return Ok(());
    }
    let attributes = operation.attributes.as_object().expect("validated object");
    if operation.inputs.len() != 4
        || operation.inputs[1] != "input.positions"
        || operation.inputs[2] != "input.attention_mask"
        || operation.inputs[3] != "input.sequence_lengths"
        || attributes.get("padding_mask_input").and_then(Value::as_str)
            != Some("input.attention_mask")
        || attributes
            .get("valid_lengths_input")
            .and_then(Value::as_str)
            != Some("input.sequence_lengths")
        || !attributes.contains_key("cache_validity")
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid causal mask dependencies",
            operation.id
        )));
    }
    Ok(())
}

fn validate_state_transition(plan: &DecoderPlan) -> Result<(), ModelError> {
    let prefill = state_slots(&plan.prefill.state_outputs)?;
    let decode = state_slots(&plan.decode.state_inputs)?;
    if prefill.len() != decode.len()
        || prefill
            .keys()
            .any(|slot| prefill.get(slot) != decode.get(slot))
    {
        return Err(ModelError::Incomplete(
            "prefill outputs and decode inputs have incompatible state slots".into(),
        ));
    }
    if !plan.prefill.state_inputs.is_empty() {
        return Err(ModelError::Incomplete(
            "Gemma prefill must initialize caches without state inputs".into(),
        ));
    }
    for (graph, producer_mode) in [(&plan.prefill, "initialize"), (&plan.decode, "append")] {
        for state in &graph.state_outputs {
            let producer = graph
                .operations
                .iter()
                .find(|operation| operation.id == state.id)
                .ok_or_else(|| {
                    ModelError::Incomplete(format!("state output {} has no producer", state.id))
                })?;
            if producer.layer != state.layer
                || producer.state_kind != Some(state.kind)
                || producer.output_shape != state.shape
            {
                return Err(ModelError::Incomplete(format!(
                    "state output {} producer descriptor differs",
                    state.id
                )));
            }
            let update = if producer.operator == ModelOperator::CacheSuffix {
                graph
                    .operations
                    .iter()
                    .find(|operation| operation.id == producer.inputs[0])
            } else {
                Some(producer)
            }
            .ok_or_else(|| {
                ModelError::Incomplete(format!("state output {} cache update is missing", state.id))
            })?;
            if update.operator != ModelOperator::KvCacheAppend
                || update.attributes.get("mode").and_then(Value::as_str) != Some(producer_mode)
            {
                return Err(ModelError::Incomplete(format!(
                    "state output {} has invalid cache producer mode",
                    state.id
                )));
            }
        }
    }
    for state in &plan.decode.state_inputs {
        if !plan.decode.operations.iter().any(|operation| {
            operation.operator == ModelOperator::KvCacheAppend
                && operation.attributes.get("mode").and_then(Value::as_str) == Some("append")
                && operation.attributes.get("state").and_then(Value::as_str) == Some(&state.id)
                && operation.inputs.first() == Some(&state.id)
        }) {
            return Err(ModelError::Incomplete(format!(
                "decode state input {} is not consumed by its append",
                state.id
            )));
        }
    }
    Ok(())
}

type StateSlotMap = BTreeMap<(Option<u64>, StateKind), (Vec<u64>, u64)>;

fn state_slots(states: &[StateTensor]) -> Result<StateSlotMap, ModelError> {
    let mut slots = BTreeMap::new();
    for state in states {
        if slots
            .insert(
                (state.layer, state.kind),
                (state.shape.clone(), state.maximum_sequence),
            )
            .is_some()
        {
            return Err(ModelError::Incomplete(format!(
                "duplicate state slot at layer {:?} for {:?}",
                state.layer, state.kind
            )));
        }
    }
    Ok(slots)
}

fn positive_decimal(value: &str, field: &str) -> Result<(), ModelError> {
    let parsed = value.parse::<f64>().map_err(|_| {
        ModelError::InvalidConfig(format!("{field} must be a positive finite decimal"))
    })?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(ModelError::InvalidConfig(format!(
            "{field} must be a positive finite decimal"
        )));
    }
    Ok(())
}

fn zero_decimal(value: &str, field: &str) -> Result<(), ModelError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| ModelError::InvalidConfig(format!("{field} must be a finite zero")))?;
    if !parsed.is_finite() || parsed != 0.0 {
        return Err(ModelError::Unsupported(format!("{field} must be zero")));
    }
    Ok(())
}

fn decimal_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    match Value::deserialize(deserializer)? {
        Value::String(value) => Ok(value),
        Value::Number(value) => Ok(value.to_string()),
        _ => Err(de::Error::custom("expected a decimal number or string")),
    }
}

fn integral_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
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

#[derive(Debug, Eq, PartialEq)]
pub enum ModelError {
    Incomplete(String),
    InvalidConfig(String),
    InvalidJson(String),
    Unsupported(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incomplete(message) => write!(formatter, "incomplete decoder graph: {message}"),
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid model configuration: {message}")
            }
            Self::InvalidJson(message) => write!(formatter, "invalid model JSON: {message}"),
            Self::Unsupported(message) => write!(formatter, "unsupported model: {message}"),
        }
    }
}

impl Error for ModelError {}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest as _, Sha256};

    const MINI_CODER_4B_CONFIG: &[u8] =
        include_bytes!("../tests/fixtures/mini-coder-4b-c87892d-config.json");
    const MINI_CODER_4B_CONFIG_SHA256: &str =
        "fdbc9e0615fcb88b2cc37aa2b23fa332b3863068d50092aa6bd1e628fd187c92";

    fn config() -> QwenConfig {
        QwenConfig {
            model_type: "qwen2".into(),
            hidden_size: 896,
            intermediate_size: 4864,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            vocab_size: 151_936,
            max_position_embeddings: 32_768,
            hidden_act: "silu".into(),
            rms_norm_eps: "0.000001".into(),
            rope_theta: 1_000_000,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn lowers_complete_bounded_prefill_and_decode_graphs() {
        let plan = lower_qwen_decoder(
            &config(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 128,
                max_new_tokens: 32,
            },
        )
        .unwrap();
        plan.validate().unwrap();
        assert_eq!(plan.prefill.operations.len(), 26 * 24 + 6);
        assert_eq!(plan.decode.operations.len(), plan.prefill.operations.len());
        assert_eq!(plan.prefill.state_outputs.len(), 48);
        assert_eq!(plan.decode.output, "token_feedback");
        assert!(plan
            .prefill
            .operations
            .iter()
            .any(|operation| operation.operator == ModelOperator::Softmax));
        assert!(plan
            .prefill
            .operations
            .iter()
            .any(|operation| operation.operator == ModelOperator::Silu));
        assert!(plan
            .prefill
            .operations
            .iter()
            .any(|operation| operation.operator == ModelOperator::OutputHead));
    }

    #[test]
    fn digest_changes_with_workload_bounds() {
        let first = lower_qwen_decoder(
            &config(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 128,
                max_new_tokens: 32,
            },
        )
        .unwrap();
        let second = lower_qwen_decoder(
            &config(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 129,
                max_new_tokens: 32,
            },
        )
        .unwrap();
        assert_ne!(first.digest(), second.digest());
    }

    #[test]
    fn rejects_wrong_architecture_and_position_overflow() {
        let mut wrong = config();
        wrong.model_type = "qwen3".into();
        assert!(matches!(wrong.validate(), Err(ModelError::Unsupported(_))));
        assert!(matches!(
            lower_qwen_decoder(
                &config(),
                DecoderWorkload {
                    batch: 1,
                    max_input_tokens: 32_768,
                    max_new_tokens: 1
                }
            ),
            Err(ModelError::InvalidConfig(_))
        ));
    }

    #[test]
    fn parses_realistic_hugging_face_numeric_fields() {
        let document = br#"{
            "model_type":"qwen2","hidden_size":896,"intermediate_size":4864,
            "num_hidden_layers":24,"num_attention_heads":14,"num_key_value_heads":2,
            "vocab_size":151936,"max_position_embeddings":32768,"hidden_act":"silu",
            "rms_norm_eps":1e-6,"rope_theta":1000000.0,"tie_word_embeddings":true,
            "ignored_upstream_field":"allowed"
        }"#;
        let parsed = QwenConfig::from_json(document).unwrap();
        parsed.validate().unwrap();
        assert_eq!(parsed.rms_norm_eps, "1e-6");
        assert_eq!(parsed.rope_theta, 1_000_000);
    }

    #[test]
    fn lowers_pinned_mini_coder_qwen3_configuration() {
        assert_eq!(
            format!("{:x}", Sha256::digest(MINI_CODER_4B_CONFIG)),
            MINI_CODER_4B_CONFIG_SHA256
        );
        let plan = lower_model_json(
            MINI_CODER_4B_CONFIG,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 128,
                max_new_tokens: 32,
            },
        )
        .unwrap();
        plan.validate().unwrap();
        assert_eq!(plan.adapter, "pllm.qwen3.v1");
        assert_eq!(plan.model_family, "qwen3");
        assert_eq!(plan.prefill.state_outputs.len(), 72);
        let operation = |id: &str| {
            plan.prefill
                .operations
                .iter()
                .find(|operation| operation.id == id)
                .unwrap()
        };
        assert_eq!(operation("layer.0.q_linear").output_shape, [1, 128, 4096]);
        assert_eq!(
            operation("layer.0.attention_hidden").output_shape,
            [1, 128, 4096]
        );
        assert_eq!(operation("layer.0.q_norm").operator, ModelOperator::RmsNorm);
        assert_eq!(
            operation("layer.0.q_linear").attributes["bias"],
            Value::Null
        );
    }

    #[test]
    fn qwen3_rejects_unrepresented_attention_semantics() {
        let mut document: Value = serde_json::from_slice(MINI_CODER_4B_CONFIG).unwrap();
        document["rope_scaling"] = json!({"rope_type": "yarn"});
        assert!(matches!(
            lower_model_json(
                serde_json::to_vec(&document).unwrap().as_slice(),
                DecoderWorkload {
                    batch: 1,
                    max_input_tokens: 8,
                    max_new_tokens: 2,
                },
            ),
            Err(ModelError::Unsupported(_))
        ));
    }

    #[test]
    fn graph_validation_rejects_bad_schema_modes_states_and_axes() {
        let mut plan = lower_qwen_decoder(
            &config(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 8,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        plan.schema_version = "pllm.decoder_plan.v2".into();
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = lower_qwen_decoder(
            &config(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 8,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        plan.prefill.mode = DecoderMode::Decode;
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = lower_qwen_decoder(
            &config(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 8,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        plan.prefill.state_inputs[1].id = plan.prefill.state_inputs[0].id.clone();
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = lower_qwen_decoder(
            &config(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 8,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        plan.prefill.state_outputs[0].id = "missing.state.output".into();
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = lower_qwen_decoder(
            &config(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 8,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        plan.prefill
            .operations
            .iter_mut()
            .find(|operation| operation.operator == ModelOperator::Softmax)
            .unwrap()
            .attributes["axis"] = json!(8);
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));
    }
}
