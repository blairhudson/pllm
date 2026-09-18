//! Model configuration validation and model-aware semantic lowering.

pub mod cache;

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
    CacheActiveIndices,
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
    CacheIndices,
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
        let dense_qwen = matches!(self.model_family.as_str(), "qwen2" | "qwen3");
        if self.model_family == "gemma4_text" || dense_qwen {
            validate_state_transition(self)?;
        }
        if dense_qwen {
            validate_dense_qwen_semantics(self)?;
        }
        cache::validate_transformation(self)?;
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
    let state_capacity = total - 1;
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
            state_capacity,
        ),
        decode: lower_graph(
            config,
            DecoderMode::Decode,
            workload.batch,
            1,
            state_capacity,
            state_capacity,
        ),
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
    state_capacity: u64,
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
    let kv_state_shape = vec![batch, kv_heads, state_capacity, head_dim];
    let kv_view_shape = vec![batch, kv_heads, maximum_key_sequence, head_dim];
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
            json!({"epsilon": config.rms_norm_eps, "weight": format!("model.layers.{layer}.input_layernorm.weight"), "weight_offset": 0}),
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
                json!({"epsilon": config.rms_norm_eps, "weight": format!("model.layers.{layer}.self_attn.q_norm.weight"), "weight_offset": 0}),
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
                json!({"epsilon": config.rms_norm_eps, "weight": format!("model.layers.{layer}.self_attn.k_norm.weight"), "weight_offset": 0}),
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
            dense_qwen_rope_attributes(config.rope_theta, head_dim),
        );
        push(
            &mut operations,
            &rope_k,
            ModelOperator::RotaryEmbedding,
            &[&k_rope_input, "input.positions"],
            kv_query_shape.clone(),
            dense_qwen_rope_attributes(config.rope_theta, head_dim),
        );
        let key_state = format!("state.layer.{layer}.key");
        let value_state = format!("state.layer.{layer}.value");
        if mode == DecoderMode::Decode {
            state_inputs.push(StateTensor {
                id: key_state.clone(),
                layer: Some(u64::from(layer)),
                kind: StateKind::Key,
                shape: kv_state_shape.clone(),
                maximum_sequence: state_capacity,
            });
            state_inputs.push(StateTensor {
                id: value_state.clone(),
                layer: Some(u64::from(layer)),
                kind: StateKind::Value,
                shape: kv_state_shape.clone(),
                maximum_sequence: state_capacity,
            });
        }
        let key_append = format!("{prefix}.key_append");
        let value_append = format!("{prefix}.value_append");
        dense_qwen_cache_update(
            &mut operations,
            &key_append,
            &key_state,
            &rope_k,
            kv_state_shape.clone(),
            state_capacity,
            maximum_key_sequence,
            mode,
            StateKind::Key,
        );
        dense_qwen_cache_update(
            &mut operations,
            &value_append,
            &value_state,
            &v,
            kv_state_shape.clone(),
            state_capacity,
            maximum_key_sequence,
            mode,
            StateKind::Value,
        );
        state_outputs.push(StateTensor {
            id: key_append.clone(),
            layer: Some(u64::from(layer)),
            kind: StateKind::Key,
            shape: kv_state_shape.clone(),
            maximum_sequence: state_capacity,
        });
        state_outputs.push(StateTensor {
            id: value_append.clone(),
            layer: Some(u64::from(layer)),
            kind: StateKind::Value,
            shape: kv_state_shape.clone(),
            maximum_sequence: state_capacity,
        });
        let key_view = format!("{prefix}.key_view");
        let value_view = format!("{prefix}.value_view");
        dense_qwen_cache_view(
            &mut operations,
            &key_view,
            &key_append,
            kv_view_shape.clone(),
            maximum_key_sequence,
            StateKind::Key,
        );
        dense_qwen_cache_view(
            &mut operations,
            &value_view,
            &value_append,
            kv_view_shape.clone(),
            maximum_key_sequence,
            StateKind::Value,
        );
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
            &[&rope_q, &key_view],
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
            &[&probabilities, &value_view],
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
            json!({"epsilon": config.rms_norm_eps, "weight": format!("model.layers.{layer}.post_attention_layernorm.weight"), "weight_offset": 0}),
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
        json!({"epsilon": config.rms_norm_eps, "weight": "model.norm.weight", "weight_offset": 0}),
    );
    if mode == DecoderMode::Prefill {
        push(
            &mut operations,
            "last_hidden",
            ModelOperator::LastToken,
            &["final_norm", "input.sequence_lengths"],
            vec![batch, config.hidden_size],
            json!({
                "axis": 1,
                "selection": "last_valid",
                "valid_lengths_input": "input.sequence_lengths"
            }),
        );
    } else {
        push(
            &mut operations,
            "last_hidden",
            ModelOperator::LastToken,
            &["final_norm"],
            vec![batch, config.hidden_size],
            json!({"axis": 1}),
        );
    }
    push(
        &mut operations,
        "output_head",
        ModelOperator::OutputHead,
        &["last_hidden"],
        vec![batch, config.vocab_size],
        json!({
            "weight": if config.tie_word_embeddings { "model.embed_tokens.weight" } else { "lm_head.weight" },
            "tied": config.tie_word_embeddings,
            "input_layout": "batch_hidden",
            "output_layout": "batch_vocabulary",
            "compute_dtype": "model_native",
            "output_dtype": "model_native"
        }),
    );
    push(
        &mut operations,
        "token_selection",
        ModelOperator::GreedyTokenSelection,
        &["output_head"],
        vec![batch],
        json!({"policy": "pllm.greedy.v1", "source": "execution_policy"}),
    );
    push(
        &mut operations,
        "token_feedback",
        ModelOperator::TokenFeedback,
        &["token_selection"],
        vec![batch, 1],
        json!({"policy": "pllm.greedy.v1", "source": "execution_policy"}),
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

fn dense_qwen_rope_attributes(theta: u64, head_dim: u64) -> Value {
    json!({
        "theta": theta,
        "rotary_dimensions": head_dim,
        "pairing": "split_half",
        "position_policy": "sequential_absolute",
        "coefficient_profile": "pllm.numeric.rope.q30.libm.v1",
        "input_layout": "batch_heads_sequence_feature",
        "output_layout": "batch_heads_sequence_feature",
        "tail_policy": "unchanged"
    })
}

#[allow(clippy::too_many_arguments)]
fn dense_qwen_cache_update(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    state: &str,
    current: &str,
    shape: Vec<u64>,
    state_capacity: u64,
    maximum_sequence: u64,
    mode: DecoderMode,
    kind: StateKind,
) {
    let attributes = json!({
        "state": state,
        "mode": if mode == DecoderMode::Prefill { "initialize" } else { "append" },
        "state_capacity": state_capacity,
        "state_layout": "batch_kv_heads_sequence_feature",
        "absolute_write_positions_input": "input.positions",
        "padding_mask_input": "input.attention_mask",
        "valid_lengths_input": "input.sequence_lengths",
        "attention_domain": {
            "layout": "batch_kv_heads_sequence_feature",
            "maximum_sequence": maximum_sequence,
            "includes_current": true
        }
    });
    if mode == DecoderMode::Prefill {
        push(
            operations,
            id,
            ModelOperator::KvCacheAppend,
            &[
                current,
                "input.positions",
                "input.attention_mask",
                "input.sequence_lengths",
            ],
            shape,
            attributes,
        );
    } else {
        push(
            operations,
            id,
            ModelOperator::KvCacheAppend,
            &[
                state,
                current,
                "input.positions",
                "input.attention_mask",
                "input.sequence_lengths",
            ],
            shape,
            attributes,
        );
    }
    operations
        .last_mut()
        .expect("cache update was pushed")
        .state_kind = Some(kind);
}

fn dense_qwen_cache_view(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    shape: Vec<u64>,
    maximum_sequence: u64,
    kind: StateKind,
) {
    push(
        operations,
        id,
        ModelOperator::CacheSuffix,
        &[
            input,
            "input.positions",
            "input.attention_mask",
            "input.sequence_lengths",
        ],
        shape,
        json!({
            "axis": 2,
            "maximum_sequence": maximum_sequence,
            "semantics": "visible_valid_prefix",
            "absolute_write_positions_input": "input.positions",
            "padding_mask_input": "input.attention_mask",
            "valid_lengths_input": "input.sequence_lengths"
        }),
    );
    operations
        .last_mut()
        .expect("cache view was pushed")
        .state_kind = Some(kind);
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
        if state.maximum_sequence != state.shape[2] {
            return Err(ModelError::Incomplete(format!(
                "state {} maximum_sequence differs from its sequence dimension",
                state.id
            )));
        }
    }
    if state.kind == StateKind::CacheIndices
        && (state.shape.len() != 2 || state.maximum_sequence != state.shape[1])
    {
        return Err(ModelError::Incomplete(format!(
            "cache-index state {} must have rank two with a matching keep bound",
            state.id
        )));
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
        | ModelOperator::GatedDeltaDecay
        | ModelOperator::SecureTopK
        | ModelOperator::ClusterBounds
        | ModelOperator::SharedIndices => Some(1),
        ModelOperator::RotaryEmbedding
        | ModelOperator::AttentionScores
        | ModelOperator::AttentionValues
        | ModelOperator::ResidualAdd
        | ModelOperator::Multiply
        | ModelOperator::CausalConvolution
        | ModelOperator::ConvolutionStateUpdate
        | ModelOperator::RmsNormGated
        | ModelOperator::AttentionImportance
        | ModelOperator::ClusterSimilarity
        | ModelOperator::SecureGather => Some(2),
        ModelOperator::CacheActiveIndices => Some(3),
        ModelOperator::GatedDeltaRule | ModelOperator::GatedDeltaStateUpdate => Some(6),
        ModelOperator::LastToken => None,
        ModelOperator::KvCacheAppend | ModelOperator::CausalMask | ModelOperator::CacheSuffix => {
            None
        }
    };
    if expected_arity.is_some_and(|arity| operation.inputs.len() != arity) {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid input arity",
            operation.id
        )));
    }
    let attributes = operation.attributes.as_object().expect("checked above");
    let required = match operation.operator {
        ModelOperator::RmsNorm => &["epsilon", "weight", "weight_offset"][..],
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
        ModelOperator::CacheActiveIndices => &[
            "static_keep",
            "prefill_maximum_sequence",
            "candidate_bound",
            "semantics",
        ][..],
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
        ModelOperator::Softmax | ModelOperator::Slice
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
        ModelOperator::RotaryEmbedding => validate_rotary_embedding(operation, shapes)?,
        ModelOperator::KvCacheAppend => validate_cache_update(graph_mode, operation, shapes)?,
        ModelOperator::CacheSuffix => validate_cache_suffix(operation, shapes)?,
        ModelOperator::CacheActiveIndices => validate_cache_active_indices(operation)?,
        ModelOperator::CausalMask => validate_causal_mask(operation)?,
        ModelOperator::AttentionScores | ModelOperator::AttentionValues => {
            validate_attention_operation(operation, shapes)?;
        }
        ModelOperator::LastToken => validate_last_token(operation, shapes)?,
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

fn validate_rotary_embedding(
    operation: &ModelOperation,
    shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<(), ModelError> {
    let attributes = operation.attributes.as_object().expect("validated object");
    let input_shape = operation
        .inputs
        .first()
        .and_then(|input| shapes.get(input))
        .ok_or_else(|| {
            ModelError::Incomplete(format!(
                "operation {} rotary input shape is unavailable",
                operation.id
            ))
        })?;
    if operation.inputs.get(1).map(String::as_str) != Some("input.positions")
        || input_shape.len() != 4
        || operation.output_shape != *input_shape
        || attributes
            .get("theta")
            .and_then(Value::as_u64)
            .is_none_or(|theta| theta == 0)
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid rotary shape or dependencies",
            operation.id
        )));
    }
    if let Some(rotary_dimensions) = attributes.get("rotary_dimensions") {
        let rotary_dimensions = rotary_dimensions.as_u64().ok_or_else(|| {
            ModelError::Incomplete(format!(
                "operation {} has invalid rotary dimensions",
                operation.id
            ))
        })?;
        if rotary_dimensions == 0
            || rotary_dimensions % 2 != 0
            || rotary_dimensions > input_shape[3]
        {
            return Err(ModelError::Incomplete(format!(
                "operation {} has invalid rotary dimensions",
                operation.id
            )));
        }
    }
    let descriptor_keys = [
        "pairing",
        "position_policy",
        "coefficient_profile",
        "input_layout",
        "output_layout",
        "tail_policy",
    ];
    if descriptor_keys
        .iter()
        .any(|key| attributes.contains_key(*key))
        && (attributes.get("pairing").and_then(Value::as_str) != Some("split_half")
            || attributes.get("position_policy").and_then(Value::as_str)
                != Some("sequential_absolute")
            || attributes
                .get("coefficient_profile")
                .and_then(Value::as_str)
                != Some("pllm.numeric.rope.q30.libm.v1")
            || attributes.get("input_layout").and_then(Value::as_str)
                != Some("batch_heads_sequence_feature")
            || attributes.get("output_layout").and_then(Value::as_str)
                != Some("batch_heads_sequence_feature")
            || attributes.get("tail_policy").and_then(Value::as_str) != Some("unchanged"))
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid rotary descriptors",
            operation.id
        )));
    }
    Ok(())
}

fn validate_last_token(
    operation: &ModelOperation,
    shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<(), ModelError> {
    if !matches!(operation.inputs.len(), 1 | 2) {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid input arity",
            operation.id
        )));
    }
    let input_shape = shapes.get(&operation.inputs[0]).ok_or_else(|| {
        ModelError::Incomplete(format!(
            "operation {} has no declared input shape",
            operation.id
        ))
    })?;
    let axis = operation.attributes["axis"].as_i64().ok_or_else(|| {
        ModelError::Incomplete(format!("operation {} has invalid axis", operation.id))
    })?;
    let rank = i64::try_from(input_shape.len()).map_err(|_| {
        ModelError::Incomplete(format!("operation {} rank overflowed", operation.id))
    })?;
    if rank < 2 || axis < -rank || axis >= rank {
        return Err(ModelError::Incomplete(format!(
            "operation {} axis is out of range",
            operation.id
        )));
    }
    let normalized_axis =
        usize::try_from(if axis < 0 { rank + axis } else { axis }).map_err(|_| {
            ModelError::Incomplete(format!("operation {} has invalid axis", operation.id))
        })?;
    let mut expected_shape = input_shape.clone();
    expected_shape.remove(normalized_axis);
    if operation.output_shape != expected_shape {
        return Err(ModelError::Incomplete(format!(
            "operation {} output shape does not remove its selected axis",
            operation.id
        )));
    }
    let attributes = operation.attributes.as_object().expect("validated above");
    if operation.inputs.len() == 1 {
        if attributes.contains_key("selection") || attributes.contains_key("valid_lengths_input") {
            return Err(ModelError::Incomplete(format!(
                "operation {} has contradictory fixed-last metadata",
                operation.id
            )));
        }
    } else if operation.inputs[1] != "input.sequence_lengths"
        || attributes.get("selection").and_then(Value::as_str) != Some("last_valid")
        || attributes
            .get("valid_lengths_input")
            .and_then(Value::as_str)
            != Some("input.sequence_lengths")
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid length-aware selection",
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
    let attention_domain = attributes
        .get("attention_domain")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ModelError::Incomplete(format!(
                "operation {} has invalid attention domain",
                operation.id
            ))
        })?;
    let attention_maximum = attention_domain
        .get("maximum_sequence")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if attributes.get("state_layout").and_then(Value::as_str)
        != Some("batch_kv_heads_sequence_feature")
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
        || attention_maximum == 0
        || attention_domain
            .get("includes_current")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid cache descriptors",
            operation.id
        )));
    }
    if operation.output_shape.len() == 4 {
        if attention_domain.get("layout").and_then(Value::as_str)
            != Some("batch_kv_heads_sequence_feature")
            || attention_maximum > state_capacity
            || operation.output_shape
                != [
                    current_shape[0],
                    current_shape[1],
                    state_capacity,
                    current_shape[3],
                ]
        {
            return Err(ModelError::Incomplete(format!(
                "operation {} has invalid sequence cache shape",
                operation.id
            )));
        }
    } else if attention_domain.get("layout").and_then(Value::as_str)
        != Some("batch_kv_heads_query_window_feature")
        || operation.output_shape[0] != current_shape[0]
        || operation.output_shape[1] != current_shape[1]
        || operation.output_shape[2] != current_shape[2]
        || operation.output_shape[3] != attention_maximum
        || operation.output_shape[4] != current_shape[3]
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid window cache shape",
            operation.id
        )));
    }
    if graph_mode == DecoderMode::Decode {
        let state_shape = shapes.get(state).ok_or_else(|| {
            ModelError::Incomplete(format!(
                "operation {} state shape is unavailable",
                operation.id
            ))
        })?;
        if state_shape.len() != 4
            || state_shape
                != &[
                    current_shape[0],
                    current_shape[1],
                    state_capacity,
                    current_shape[3],
                ]
        {
            return Err(ModelError::Incomplete(format!(
                "operation {} state shape differs from its capacity",
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
        || operation
            .inputs
            .first()
            .and_then(|input| shapes.get(input))
            .is_none()
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid cache suffix contract",
            operation.id
        )));
    }
    let input_shape = shapes.get(&operation.inputs[0]).expect("validated above");
    let maximum_sequence = attributes
        .get("maximum_sequence")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let valid_shape = if input_shape.len() == 4 {
        attributes.get("axis").and_then(Value::as_u64) == Some(2)
            && attributes.get("semantics").and_then(Value::as_str) == Some("visible_valid_prefix")
            && operation.output_shape
                == [
                    input_shape[0],
                    input_shape[1],
                    maximum_sequence,
                    input_shape[3],
                ]
            && maximum_sequence <= input_shape[2]
    } else if input_shape.len() == 5 {
        attributes.get("axis").and_then(Value::as_u64) == Some(3)
            && attributes.get("output_axis").and_then(Value::as_u64) == Some(2)
            && attributes.get("semantics").and_then(Value::as_str)
                == Some("persist_last_valid_past_tokens")
            && operation.output_shape
                == [
                    input_shape[0],
                    input_shape[1],
                    maximum_sequence,
                    input_shape[4],
                ]
            && maximum_sequence <= input_shape[3]
    } else {
        false
    };
    if maximum_sequence == 0 || !valid_shape {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid cache suffix shape",
            operation.id
        )));
    }
    Ok(())
}

fn validate_cache_active_indices(operation: &ModelOperation) -> Result<(), ModelError> {
    let attributes = operation.attributes.as_object().expect("validated object");
    if operation.inputs[1] != "input.positions"
        || operation.inputs[2] != "input.sequence_lengths"
        || operation.output_shape.len() != 2
        || attributes.get("semantics").and_then(Value::as_str)
            != Some("static_plus_generated_global_positions")
        || attributes.get("candidate_bound").and_then(Value::as_u64)
            != operation.output_shape.get(1).copied()
    {
        return Err(ModelError::Incomplete(format!(
            "operation {} has invalid active cache indices contract",
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
    if operation.inputs.len() == 3 {
        if operation.inputs[1] != "input.positions"
            || attributes
                .get("cache_positions_input")
                .and_then(Value::as_str)
                != operation.inputs.get(2).map(String::as_str)
            || attributes
                .get("selection_semantics")
                .and_then(Value::as_str)
                != Some("global_token_positions")
        {
            return Err(ModelError::Incomplete(format!(
                "operation {} has invalid selected-position causal mask",
                operation.id
            )));
        }
        return Ok(());
    }
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
            "prefill must initialize caches without state inputs".into(),
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
            if state.kind == StateKind::CacheIndices {
                let valid_producer = match graph.mode {
                    DecoderMode::Prefill => producer.operator == ModelOperator::SecureTopK,
                    DecoderMode::Decode => {
                        producer.operator == ModelOperator::SharedIndices
                            && producer.attributes.get("mode").and_then(Value::as_str)
                                == Some("carry")
                    }
                };
                if !valid_producer {
                    return Err(ModelError::Incomplete(format!(
                        "state output {} has invalid cache-index producer",
                        state.id
                    )));
                }
                continue;
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
                || update
                    .attributes
                    .get("state_capacity")
                    .and_then(Value::as_u64)
                    != Some(state.maximum_sequence)
            {
                return Err(ModelError::Incomplete(format!(
                    "state output {} has invalid cache producer mode",
                    state.id
                )));
            }
        }
    }
    for state in &plan.decode.state_inputs {
        if state.kind == StateKind::CacheIndices {
            if !plan.decode.operations.iter().any(|operation| {
                operation.operator == ModelOperator::SharedIndices
                    && operation.state_kind == Some(StateKind::CacheIndices)
                    && operation.attributes.get("mode").and_then(Value::as_str) == Some("carry")
                    && operation.inputs.first() == Some(&state.id)
            }) {
                return Err(ModelError::Incomplete(format!(
                    "decode state input {} is not consumed by its carry",
                    state.id
                )));
            }
            continue;
        }
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

#[derive(Debug, Eq, PartialEq)]
struct DenseQwenOutputContract {
    head_attributes: Value,
    input_shape: Vec<u64>,
    output_shape: Vec<u64>,
    selection_attributes: Value,
    selection_shape: Vec<u64>,
    feedback_attributes: Value,
    feedback_shape: Vec<u64>,
}

fn validate_dense_qwen_semantics(plan: &DecoderPlan) -> Result<(), ModelError> {
    let qwen3 = plan.model_family == "qwen3";
    let epsilon = validate_dense_qwen_epsilon(plan)?;
    let theta = validate_dense_qwen_theta(plan)?;
    let mut output_contract = None;
    for graph in [&plan.prefill, &plan.decode] {
        let layers = graph
            .operations
            .iter()
            .filter_map(|operation| operation.layer)
            .collect::<BTreeSet<_>>();
        if layers.iter().copied().ne(0..layers.len() as u64) {
            return Err(ModelError::Incomplete(
                "dense Qwen layers must be contiguous from zero".into(),
            ));
        }
        let expected_norms = if qwen3 {
            layers.len() * 4 + 1
        } else {
            layers.len() * 2 + 1
        };
        if graph
            .operations
            .iter()
            .filter(|operation| operation.operator == ModelOperator::AttentionScores)
            .count()
            != layers.len()
            || graph
                .operations
                .iter()
                .filter(|operation| operation.operator == ModelOperator::AttentionValues)
                .count()
                != layers.len()
            || graph
                .operations
                .iter()
                .filter(|operation| operation.operator == ModelOperator::RotaryEmbedding)
                .count()
                != layers.len() * 2
            || graph
                .operations
                .iter()
                .filter(|operation| operation.operator == ModelOperator::RmsNorm)
                .count()
                != expected_norms
        {
            return Err(ModelError::Incomplete(
                "dense Qwen graph has incomplete attention or RMSNorm topology".into(),
            ));
        }
        let mut previous_residual = None;
        for layer in &layers {
            let scores = dense_qwen_layer_operation(graph, *layer, ModelOperator::AttentionScores)?;
            let values = dense_qwen_layer_operation(graph, *layer, ModelOperator::AttentionValues)?;
            let (input_norm, mlp_residual) = validate_dense_qwen_attention_layer(
                graph, *layer, scores, values, qwen3, &epsilon, theta,
            )?;
            let source = dense_qwen_input(graph, input_norm, 0)?;
            if let Some(previous) = previous_residual {
                if source.id != previous {
                    return Err(ModelError::Incomplete(format!(
                        "dense Qwen layer {layer} bypasses previous MLP residual"
                    )));
                }
            } else if source.operator != ModelOperator::TokenLookup
                || source.inputs.as_slice() != ["input.tokens"]
                || source.attributes.get("weight").and_then(Value::as_str)
                    != Some("model.embed_tokens.weight")
            {
                return Err(ModelError::Incomplete(
                    "dense Qwen first layer does not consume token lookup".into(),
                ));
            }
            previous_residual = Some(mlp_residual.id.as_str());
        }
        let final_residual = previous_residual.ok_or_else(|| {
            ModelError::Incomplete("dense Qwen graph has no decoder layers".into())
        })?;
        let graph_output_contract =
            validate_dense_qwen_graph_tail(graph, final_residual, &epsilon)?;
        if output_contract
            .as_ref()
            .is_some_and(|expected| expected != &graph_output_contract)
        {
            return Err(ModelError::Incomplete(
                "dense Qwen prefill and decode output-head contracts differ".into(),
            ));
        }
        output_contract = Some(graph_output_contract);
        if !qwen3
            && graph.operations.iter().any(|operation| {
                operation.id.ends_with(".q_norm") || operation.id.ends_with(".k_norm")
            })
        {
            return Err(ModelError::Incomplete(
                "Qwen2 must not contain per-head q_norm or k_norm operations".into(),
            ));
        }
        for operation in &graph.operations {
            match operation.operator {
                ModelOperator::RotaryEmbedding => {
                    let attributes = operation.attributes.as_object().expect("validated object");
                    if attributes.get("rotary_dimensions").and_then(Value::as_u64)
                        != operation.output_shape.last().copied()
                        || attributes.get("pairing").and_then(Value::as_str) != Some("split_half")
                        || attributes.get("position_policy").and_then(Value::as_str)
                            != Some("sequential_absolute")
                        || attributes
                            .get("coefficient_profile")
                            .and_then(Value::as_str)
                            != Some("pllm.numeric.rope.q30.libm.v1")
                        || attributes.get("input_layout").and_then(Value::as_str)
                            != Some("batch_heads_sequence_feature")
                        || attributes.get("output_layout").and_then(Value::as_str)
                            != Some("batch_heads_sequence_feature")
                        || attributes.get("tail_policy").and_then(Value::as_str)
                            != Some("unchanged")
                    {
                        return Err(ModelError::Incomplete(format!(
                            "dense Qwen operation {} omits exact rotary semantics",
                            operation.id
                        )));
                    }
                }
                ModelOperator::KvCacheAppend => {
                    if operation
                        .attributes
                        .get("attention_domain")
                        .and_then(|domain| domain.get("maximum_sequence"))
                        .and_then(Value::as_u64)
                        != Some(graph.maximum_key_sequence)
                    {
                        return Err(ModelError::Incomplete(format!(
                            "dense Qwen operation {} has wrong visible cache bound",
                            operation.id
                        )));
                    }
                }
                ModelOperator::CacheSuffix => {
                    if operation
                        .attributes
                        .get("maximum_sequence")
                        .and_then(Value::as_u64)
                        != Some(graph.maximum_key_sequence)
                        || operation
                            .attributes
                            .get("semantics")
                            .and_then(Value::as_str)
                            != Some("visible_valid_prefix")
                    {
                        return Err(ModelError::Incomplete(format!(
                            "dense Qwen operation {} has wrong cache view semantics",
                            operation.id
                        )));
                    }
                    let update = graph
                        .operations
                        .iter()
                        .find(|candidate| operation.inputs.first() == Some(&candidate.id));
                    if update.is_none_or(|update| {
                        update.operator != ModelOperator::KvCacheAppend
                            || update.state_kind != operation.state_kind
                    }) {
                        return Err(ModelError::Incomplete(format!(
                            "dense Qwen operation {} does not view its cache update",
                            operation.id
                        )));
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn validate_dense_qwen_epsilon(plan: &DecoderPlan) -> Result<String, ModelError> {
    let mut norms = plan
        .prefill
        .operations
        .iter()
        .chain(&plan.decode.operations)
        .filter(|operation| operation.operator == ModelOperator::RmsNorm);
    let epsilon = norms
        .next()
        .and_then(|operation| operation.attributes.get("epsilon"))
        .and_then(Value::as_str)
        .ok_or_else(|| ModelError::Incomplete("dense Qwen RMSNorm epsilon is missing".into()))?
        .to_owned();
    let parsed = epsilon.parse::<f64>().map_err(|_| {
        ModelError::Incomplete("dense Qwen RMSNorm epsilon is not a positive decimal".into())
    })?;
    if !parsed.is_finite()
        || parsed <= 0.0
        || norms.any(|operation| {
            operation.attributes.get("epsilon").and_then(Value::as_str) != Some(epsilon.as_str())
        })
    {
        return Err(ModelError::Incomplete(
            "dense Qwen RMSNorm epsilon differs across layers or phases".into(),
        ));
    }
    Ok(epsilon)
}

fn validate_dense_qwen_theta(plan: &DecoderPlan) -> Result<u64, ModelError> {
    let mut rotary = plan
        .prefill
        .operations
        .iter()
        .chain(&plan.decode.operations)
        .filter(|operation| operation.operator == ModelOperator::RotaryEmbedding);
    let theta = rotary
        .next()
        .and_then(|operation| operation.attributes.get("theta"))
        .and_then(Value::as_u64)
        .filter(|theta| *theta > 0)
        .ok_or_else(|| ModelError::Incomplete("dense Qwen RoPE theta is invalid".into()))?;
    if rotary
        .any(|operation| operation.attributes.get("theta").and_then(Value::as_u64) != Some(theta))
    {
        return Err(ModelError::Incomplete(
            "dense Qwen RoPE theta differs across q/k, layers, or phases".into(),
        ));
    }
    Ok(theta)
}

fn dense_qwen_layer_operation(
    graph: &DecoderGraph,
    layer: u64,
    operator: ModelOperator,
) -> Result<&ModelOperation, ModelError> {
    let mut matching = graph
        .operations
        .iter()
        .filter(|operation| operation.layer == Some(layer) && operation.operator == operator);
    let operation = matching.next().ok_or_else(|| {
        ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has no {operator:?} operation"
        ))
    })?;
    if matching.next().is_some() {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has multiple {operator:?} operations"
        )));
    }
    Ok(operation)
}

fn dense_qwen_input<'a>(
    graph: &'a DecoderGraph,
    operation: &ModelOperation,
    index: usize,
) -> Result<&'a ModelOperation, ModelError> {
    operation
        .inputs
        .get(index)
        .and_then(|input| {
            graph
                .operations
                .iter()
                .find(|candidate| &candidate.id == input)
        })
        .ok_or_else(|| {
            ModelError::Incomplete(format!(
                "dense Qwen operation {} has missing producer at input {index}",
                operation.id
            ))
        })
}

fn dense_qwen_consumer<'a>(
    graph: &'a DecoderGraph,
    layer: u64,
    input: &str,
    operator: ModelOperator,
) -> Result<&'a ModelOperation, ModelError> {
    let mut matching = graph.operations.iter().filter(|operation| {
        operation.layer == Some(layer)
            && operation.operator == operator
            && operation.inputs.iter().any(|candidate| candidate == input)
    });
    let operation = matching.next().ok_or_else(|| {
        ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has no {operator:?} consumer for {input}"
        ))
    })?;
    if matching.next().is_some() {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has multiple {operator:?} consumers for {input}"
        )));
    }
    Ok(operation)
}

struct DenseQwenCachePath<'a> {
    attention_input: &'a ModelOperation,
    view: &'a ModelOperation,
    append: &'a ModelOperation,
    selected_positions: Option<&'a str>,
}

fn validate_dense_qwen_cache_path<'a>(
    graph: &'a DecoderGraph,
    attention: &ModelOperation,
    kind: StateKind,
) -> Result<DenseQwenCachePath<'a>, ModelError> {
    let layer = attention.layer.ok_or_else(|| {
        ModelError::Incomplete(format!(
            "dense Qwen attention {} has no layer",
            attention.id
        ))
    })?;
    let attention_input = attention
        .inputs
        .get(1)
        .and_then(|input| {
            graph
                .operations
                .iter()
                .find(|operation| &operation.id == input)
        })
        .ok_or_else(|| {
            ModelError::Incomplete(format!(
                "dense Qwen attention {} has no cache view",
                attention.id
            ))
        })?;
    let (view, selected_positions) = if attention_input.operator == ModelOperator::SecureGather {
        let view = attention_input
            .inputs
            .first()
            .and_then(|input| {
                graph
                    .operations
                    .iter()
                    .find(|operation| &operation.id == input)
            })
            .ok_or_else(|| {
                ModelError::Incomplete(format!(
                    "dense Qwen attention {} gather has no cache view",
                    attention.id
                ))
            })?;
        (view, attention_input.inputs.get(1).map(String::as_str))
    } else {
        (attention_input, None)
    };
    let append = view
        .inputs
        .first()
        .and_then(|input| {
            graph
                .operations
                .iter()
                .find(|operation| &operation.id == input)
        })
        .ok_or_else(|| {
            ModelError::Incomplete(format!(
                "dense Qwen attention {} has no cache append",
                attention.id
            ))
        })?;
    let expected_mode = if graph.mode == DecoderMode::Prefill {
        "initialize"
    } else {
        "append"
    };
    let expected_state = format!(
        "state.layer.{layer}.{}",
        if kind == StateKind::Key {
            "key"
        } else {
            "value"
        }
    );
    let common_shape = view.output_shape.len() == 4
        && append.output_shape.len() == 4
        && view.output_shape[0] == append.output_shape[0]
        && view.output_shape[1] == append.output_shape[1]
        && view.output_shape[3] == append.output_shape[3];
    let gather = selected_positions.is_some();
    let window = if gather {
        attention_input.output_shape.get(3).copied().unwrap_or(0)
    } else {
        view.output_shape.get(2).copied().unwrap_or(0)
    };
    let gather_shape = !gather
        || (attention_input.output_shape.len() == 5
            && attention_input.layer == Some(layer)
            && attention_input.state_kind == Some(kind)
            && attention_input.inputs.len() == 2
            && attention_input.output_shape[0] == view.output_shape[0]
            && attention_input.output_shape[4] == view.output_shape[3]);
    let attention_shape = if kind == StateKind::Key {
        let query = attention.inputs.first().and_then(|input| {
            graph
                .operations
                .iter()
                .find(|operation| &operation.id == input)
        });
        query.is_some_and(|query| {
            query.output_shape.len() == 4
                && attention.output_shape
                    == [
                        query.output_shape[0],
                        query.output_shape[1],
                        query.output_shape[2],
                        window,
                    ]
                && view.output_shape[0] == query.output_shape[0]
                && view.output_shape[3] == query.output_shape[3]
                && (!gather
                    || (attention_input.output_shape[1] == query.output_shape[1]
                        && attention_input.output_shape[2] == query.output_shape[2]))
        })
    } else {
        let probabilities = attention.inputs.first().and_then(|input| {
            graph
                .operations
                .iter()
                .find(|operation| &operation.id == input)
        });
        probabilities.is_some_and(|probabilities| {
            probabilities.output_shape.len() == 4
                && probabilities.output_shape[3] == window
                && view.output_shape[0] == probabilities.output_shape[0]
                && attention.output_shape
                    == [
                        probabilities.output_shape[0],
                        probabilities.output_shape[1],
                        probabilities.output_shape[2],
                        view.output_shape[3],
                    ]
                && (!gather
                    || (attention_input.output_shape[1] == probabilities.output_shape[1]
                        && attention_input.output_shape[2] == probabilities.output_shape[2]))
        })
    };
    if view.operator != ModelOperator::CacheSuffix
        || view.layer != Some(layer)
        || view.state_kind != Some(kind)
        || append.operator != ModelOperator::KvCacheAppend
        || append.layer != Some(layer)
        || append.state_kind != Some(kind)
        || append.attributes.get("mode").and_then(Value::as_str) != Some(expected_mode)
        || append.attributes.get("state").and_then(Value::as_str) != Some(expected_state.as_str())
        || !common_shape
        || !gather_shape
        || !attention_shape
    {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen attention {} has invalid {:?} cache path",
            attention.id, kind
        )));
    }
    Ok(DenseQwenCachePath {
        attention_input,
        view,
        append,
        selected_positions,
    })
}

fn validate_dense_qwen_attention_layer<'a>(
    graph: &'a DecoderGraph,
    layer: u64,
    scores: &ModelOperation,
    values: &ModelOperation,
    qwen3: bool,
    epsilon: &str,
    theta: u64,
) -> Result<(&'a ModelOperation, &'a ModelOperation), ModelError> {
    let key_path = validate_dense_qwen_cache_path(graph, scores, StateKind::Key)?;
    let value_path = validate_dense_qwen_cache_path(graph, values, StateKind::Value)?;
    if key_path.selected_positions != value_path.selected_positions {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} Key and Value gathers select different positions"
        )));
    }
    let (key_view, key_append) = (key_path.view, key_path.append);
    let (value_view, value_append) = (value_path.view, value_path.append);
    let rope_q = dense_qwen_input(graph, scores, 0)?;
    let current_index = usize::from(graph.mode == DecoderMode::Decode);
    let rope_k = dense_qwen_input(graph, key_append, current_index)?;
    let v_heads = dense_qwen_input(graph, value_append, current_index)?;
    if rope_q.operator != ModelOperator::RotaryEmbedding
        || rope_k.operator != ModelOperator::RotaryEmbedding
        || rope_q.layer != Some(layer)
        || rope_k.layer != Some(layer)
        || rope_q.attributes.get("theta").and_then(Value::as_u64) != Some(theta)
        || rope_k.attributes.get("theta").and_then(Value::as_u64) != Some(theta)
    {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has invalid rotary producers"
        )));
    }

    let q_source = dense_qwen_input(graph, rope_q, 0)?;
    let k_source = dense_qwen_input(graph, rope_k, 0)?;
    let (q_heads, k_heads) = if qwen3 {
        let q_heads = dense_qwen_input(graph, q_source, 0)?;
        let k_heads = dense_qwen_input(graph, k_source, 0)?;
        validate_qwen3_qk_norm(q_source, q_heads, rope_q, layer, "q", epsilon)?;
        validate_qwen3_qk_norm(k_source, k_heads, rope_k, layer, "k", epsilon)?;
        (q_heads, k_heads)
    } else {
        (q_source, k_source)
    };
    for heads in [q_heads, k_heads, v_heads] {
        if heads.operator != ModelOperator::Reshape
            || heads.layer != Some(layer)
            || heads.attributes.get("layout").and_then(Value::as_str)
                != Some("batch_heads_sequence_feature")
        {
            return Err(ModelError::Incomplete(format!(
                "dense Qwen layer {layer} has invalid attention reshape"
            )));
        }
    }
    if rope_q.output_shape != q_heads.output_shape
        || rope_k.output_shape != k_heads.output_shape
        || key_view.inputs.first() != Some(&key_append.id)
        || value_view.inputs.first() != Some(&value_append.id)
    {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has invalid rotary or cache-view dependencies"
        )));
    }

    let q_linear = dense_qwen_input(graph, q_heads, 0)?;
    let k_linear = dense_qwen_input(graph, k_heads, 0)?;
    let v_linear = dense_qwen_input(graph, v_heads, 0)?;
    let norm_id = q_linear.inputs.first().ok_or_else(|| {
        ModelError::Incomplete(format!(
            "dense Qwen layer {layer} q projection has no input"
        ))
    })?;
    if k_linear.inputs.as_slice() != [norm_id.as_str()]
        || v_linear.inputs.as_slice() != [norm_id.as_str()]
    {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} projections do not share input norm"
        )));
    }
    let input_norm = dense_qwen_input(graph, q_linear, 0)?;
    validate_dense_qwen_projection(q_linear, q_heads, input_norm, layer, "q_proj", qwen3)?;
    validate_dense_qwen_projection(k_linear, k_heads, input_norm, layer, "k_proj", qwen3)?;
    validate_dense_qwen_projection(v_linear, v_heads, input_norm, layer, "v_proj", qwen3)?;
    let expected_input_norm_weight = format!("model.layers.{layer}.input_layernorm.weight");
    if input_norm.operator != ModelOperator::RmsNorm
        || input_norm.layer != Some(layer)
        || input_norm.output_shape.len() != 3
        || input_norm.attributes.get("epsilon").and_then(Value::as_str) != Some(epsilon)
        || input_norm
            .attributes
            .get("weight_offset")
            .and_then(Value::as_u64)
            != Some(0)
        || input_norm.attributes.get("weight").and_then(Value::as_str)
            != Some(expected_input_norm_weight.as_str())
    {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has invalid attention input norm"
        )));
    }

    let scale = dense_qwen_consumer(graph, layer, &scores.id, ModelOperator::AttentionScale)?;
    let mask = dense_qwen_consumer(graph, layer, &scale.id, ModelOperator::CausalMask)?;
    let softmax = dense_qwen_consumer(graph, layer, &mask.id, ModelOperator::Softmax)?;
    let mask_inputs: &[&str] = match key_path.selected_positions {
        Some(selected) => &[scale.id.as_str(), "input.positions", selected],
        None => &[scale.id.as_str(), "input.positions"],
    };
    if scale.inputs.as_slice() != [scores.id.as_str()]
        || scale.output_shape != scores.output_shape
        || mask.inputs.as_slice() != mask_inputs
        || mask.output_shape != scale.output_shape
        || softmax.inputs.as_slice() != [mask.id.as_str()]
        || softmax.output_shape != mask.output_shape
        || values.inputs.as_slice() != [softmax.id.as_str(), value_path.attention_input.id.as_str()]
    {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has invalid score-to-value dependency chain"
        )));
    }

    let attention_hidden = dense_qwen_consumer(graph, layer, &values.id, ModelOperator::Reshape)?;
    let o_proj = dense_qwen_consumer(graph, layer, &attention_hidden.id, ModelOperator::Linear)?;
    let residual = dense_qwen_consumer(graph, layer, &o_proj.id, ModelOperator::ResidualAdd)?;
    let residual_source = dense_qwen_input(graph, input_norm, 0)?;
    let attention_width = values.output_shape[1]
        .checked_mul(values.output_shape[3])
        .ok_or_else(|| {
            ModelError::Incomplete(format!(
                "dense Qwen layer {layer} attention width overflowed"
            ))
        })?;
    let expected_hidden = [
        values.output_shape[0],
        values.output_shape[2],
        attention_width,
    ];
    let expected_o_weight = format!("model.layers.{layer}.self_attn.o_proj.weight");
    if attention_hidden.inputs.as_slice() != [values.id.as_str()]
        || attention_hidden.output_shape != expected_hidden
        || attention_hidden
            .attributes
            .get("layout")
            .and_then(Value::as_str)
            != Some("batch_sequence_hidden")
        || o_proj.inputs.as_slice() != [attention_hidden.id.as_str()]
        || o_proj.output_shape != residual_source.output_shape
        || o_proj.attributes.get("weight").and_then(Value::as_str)
            != Some(expected_o_weight.as_str())
        || !o_proj.attributes.get("bias").is_some_and(Value::is_null)
        || residual.inputs.as_slice() != [residual_source.id.as_str(), o_proj.id.as_str()]
        || residual.output_shape != residual_source.output_shape
        || input_norm.output_shape != residual_source.output_shape
    {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has invalid attention output or residual chain"
        )));
    }
    let post_norm = dense_qwen_consumer(graph, layer, &residual.id, ModelOperator::RmsNorm)?;
    let linears = graph
        .operations
        .iter()
        .filter(|operation| {
            operation.layer == Some(layer)
                && operation.operator == ModelOperator::Linear
                && operation.inputs.as_slice() == [post_norm.id.as_str()]
        })
        .collect::<Vec<_>>();
    if linears.len() != 2 {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} must have gate and up projections"
        )));
    }
    let expected_gate_weight = format!("model.layers.{layer}.mlp.gate_proj.weight");
    let expected_up_weight = format!("model.layers.{layer}.mlp.up_proj.weight");
    let gate = linears
        .iter()
        .find(|operation| {
            operation.attributes.get("weight").and_then(Value::as_str)
                == Some(expected_gate_weight.as_str())
        })
        .copied()
        .ok_or_else(|| {
            ModelError::Incomplete(format!(
                "dense Qwen layer {layer} has invalid gate projection"
            ))
        })?;
    let up = linears.iter().find(|operation| {
        operation.attributes.get("weight").and_then(Value::as_str)
            == Some(expected_up_weight.as_str())
    });
    let up = up.copied().ok_or_else(|| {
        ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has invalid up projection"
        ))
    })?;
    let silu = dense_qwen_consumer(graph, layer, &gate.id, ModelOperator::Silu)?;
    let multiplied = dense_qwen_consumer(graph, layer, &silu.id, ModelOperator::Multiply)?;
    let down = dense_qwen_consumer(graph, layer, &multiplied.id, ModelOperator::Linear)?;
    let mlp_residual = dense_qwen_consumer(graph, layer, &down.id, ModelOperator::ResidualAdd)?;
    let expected_post_weight = format!("model.layers.{layer}.post_attention_layernorm.weight");
    let expected_down_weight = format!("model.layers.{layer}.mlp.down_proj.weight");
    if post_norm.inputs.as_slice() != [residual.id.as_str()]
        || post_norm.output_shape != residual.output_shape
        || post_norm.attributes.get("epsilon").and_then(Value::as_str) != Some(epsilon)
        || post_norm.attributes.get("weight").and_then(Value::as_str)
            != Some(expected_post_weight.as_str())
        || post_norm
            .attributes
            .get("weight_offset")
            .and_then(Value::as_u64)
            != Some(0)
        || gate.inputs.as_slice() != [post_norm.id.as_str()]
        || gate.output_shape != up.output_shape
        || gate.attributes.get("weight").and_then(Value::as_str)
            != Some(expected_gate_weight.as_str())
        || !gate.attributes.get("bias").is_some_and(Value::is_null)
        || up.inputs.as_slice() != [post_norm.id.as_str()]
        || !up.attributes.get("bias").is_some_and(Value::is_null)
        || silu.inputs.as_slice() != [gate.id.as_str()]
        || silu.output_shape != gate.output_shape
        || multiplied.inputs.as_slice() != [silu.id.as_str(), up.id.as_str()]
        || multiplied.output_shape != gate.output_shape
        || down.inputs.as_slice() != [multiplied.id.as_str()]
        || down.output_shape != residual.output_shape
        || down.attributes.get("weight").and_then(Value::as_str)
            != Some(expected_down_weight.as_str())
        || !down.attributes.get("bias").is_some_and(Value::is_null)
        || mlp_residual.inputs.as_slice() != [residual.id.as_str(), down.id.as_str()]
        || mlp_residual.output_shape != residual.output_shape
    {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has invalid MLP weight, bias, or residual topology"
        )));
    }
    Ok((input_norm, mlp_residual))
}

fn validate_qwen3_qk_norm(
    norm: &ModelOperation,
    heads: &ModelOperation,
    rope: &ModelOperation,
    layer: u64,
    projection: &str,
    epsilon: &str,
) -> Result<(), ModelError> {
    let expected_weight = format!("model.layers.{layer}.self_attn.{projection}_norm.weight");
    if norm.operator != ModelOperator::RmsNorm
        || norm.layer != Some(layer)
        || norm.inputs.as_slice() != [heads.id.as_str()]
        || norm.output_shape != heads.output_shape
        || norm.attributes.get("weight").and_then(Value::as_str) != Some(expected_weight.as_str())
        || norm.attributes.get("weight_offset").and_then(Value::as_u64) != Some(0)
        || norm.attributes.get("epsilon").and_then(Value::as_str) != Some(epsilon)
        || rope.inputs.first() != Some(&norm.id)
        || rope.output_shape != norm.output_shape
    {
        return Err(ModelError::Incomplete(format!(
            "Qwen3 layer {layer} has invalid {projection}_norm semantics"
        )));
    }
    Ok(())
}

fn validate_dense_qwen_projection(
    linear: &ModelOperation,
    heads: &ModelOperation,
    input_norm: &ModelOperation,
    layer: u64,
    projection: &str,
    qwen3: bool,
) -> Result<(), ModelError> {
    let expected_weight = format!("model.layers.{layer}.self_attn.{projection}.weight");
    let expected_bias = format!("model.layers.{layer}.self_attn.{projection}.bias");
    let valid_bias = if qwen3 {
        linear.attributes.get("bias").is_some_and(Value::is_null)
    } else {
        linear.attributes.get("bias").and_then(Value::as_str) == Some(expected_bias.as_str())
    };
    let shape_matches = linear.output_shape.len() == 3
        && heads.output_shape.len() == 4
        && heads.output_shape[0] == linear.output_shape[0]
        && heads.output_shape[2] == linear.output_shape[1]
        && u128::from(heads.output_shape[1]) * u128::from(heads.output_shape[3])
            == u128::from(linear.output_shape[2]);
    if linear.operator != ModelOperator::Linear
        || linear.layer != Some(layer)
        || linear.inputs.as_slice() != [input_norm.id.as_str()]
        || linear.attributes.get("weight").and_then(Value::as_str) != Some(expected_weight.as_str())
        || !valid_bias
        || heads.inputs.as_slice() != [linear.id.as_str()]
        || !shape_matches
    {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen layer {layer} has invalid {projection} linear-to-reshape weight or bias semantics"
        )));
    }
    Ok(())
}

fn dense_qwen_unlayered_consumer<'a>(
    graph: &'a DecoderGraph,
    input: &str,
    operator: ModelOperator,
) -> Result<&'a ModelOperation, ModelError> {
    let mut matching = graph.operations.iter().filter(|operation| {
        operation.layer.is_none()
            && operation.operator == operator
            && operation.inputs.iter().any(|candidate| candidate == input)
    });
    let operation = matching.next().ok_or_else(|| {
        ModelError::Incomplete(format!(
            "dense Qwen graph has no unlayered {operator:?} consumer for {input}"
        ))
    })?;
    if matching.next().is_some() {
        return Err(ModelError::Incomplete(format!(
            "dense Qwen graph has multiple unlayered {operator:?} consumers for {input}"
        )));
    }
    Ok(operation)
}

fn validate_dense_qwen_graph_tail(
    graph: &DecoderGraph,
    final_residual_id: &str,
    epsilon: &str,
) -> Result<DenseQwenOutputContract, ModelError> {
    let final_residual = graph
        .operations
        .iter()
        .find(|operation| operation.id == final_residual_id)
        .expect("validated final residual");
    let final_norm =
        dense_qwen_unlayered_consumer(graph, final_residual_id, ModelOperator::RmsNorm)?;
    let last_token =
        dense_qwen_unlayered_consumer(graph, &final_norm.id, ModelOperator::LastToken)?;
    let output_head =
        dense_qwen_unlayered_consumer(graph, &last_token.id, ModelOperator::OutputHead)?;
    let selection =
        dense_qwen_unlayered_consumer(graph, &output_head.id, ModelOperator::GreedyTokenSelection)?;
    let feedback =
        dense_qwen_unlayered_consumer(graph, &selection.id, ModelOperator::TokenFeedback)?;
    let length_aware = last_token.inputs.as_slice()
        == [final_norm.id.as_str(), "input.sequence_lengths"]
        && last_token
            .attributes
            .get("selection")
            .and_then(Value::as_str)
            == Some("last_valid")
        && last_token
            .attributes
            .get("valid_lengths_input")
            .and_then(Value::as_str)
            == Some("input.sequence_lengths");
    let physical_decode = graph.mode == DecoderMode::Decode
        && graph.query_sequence == 1
        && final_norm.output_shape.get(1) == Some(&1)
        && last_token.inputs.as_slice() == [final_norm.id.as_str()]
        && last_token.attributes.get("selection").is_none()
        && last_token.attributes.get("valid_lengths_input").is_none();
    let output_weight = output_head.attributes.get("weight").and_then(Value::as_str);
    let tied = output_head.attributes.get("tied").and_then(Value::as_bool);
    let valid_weight_policy = matches!(
        (output_weight, tied),
        (Some("model.embed_tokens.weight"), Some(true)) | (Some("lm_head.weight"), Some(false))
    );
    let expected_head_attributes = output_weight.zip(tied).map(|(weight, tied)| {
        json!({
            "weight": weight,
            "tied": tied,
            "input_layout": "batch_hidden",
            "output_layout": "batch_vocabulary",
            "compute_dtype": "model_native",
            "output_dtype": "model_native"
        })
    });
    if final_norm.inputs.as_slice() != [final_residual.id.as_str()]
        || final_norm.output_shape != final_residual.output_shape
        || final_norm.attributes.get("epsilon").and_then(Value::as_str) != Some(epsilon)
        || final_norm.attributes.get("weight").and_then(Value::as_str) != Some("model.norm.weight")
        || final_norm
            .attributes
            .get("weight_offset")
            .and_then(Value::as_u64)
            != Some(0)
        || last_token.attributes.get("axis").and_then(Value::as_i64) != Some(1)
        || (graph.mode == DecoderMode::Prefill && !length_aware)
        || (graph.mode == DecoderMode::Decode && !length_aware && !physical_decode)
        || last_token.output_shape != [final_norm.output_shape[0], final_norm.output_shape[2]]
        || output_head.inputs.as_slice() != [last_token.id.as_str()]
        || output_head.output_shape.len() != 2
        || output_head.output_shape[0] != last_token.output_shape[0]
        || output_head.output_shape[1] == 0
        || !valid_weight_policy
        || expected_head_attributes.as_ref() != Some(&output_head.attributes)
        || selection.inputs.as_slice() != [output_head.id.as_str()]
        || selection.output_shape != [graph.batch]
        || selection.attributes != json!({"policy": "pllm.greedy.v1", "source": "execution_policy"})
        || feedback.inputs.as_slice() != [selection.id.as_str()]
        || feedback.output_shape != [graph.batch, 1]
        || feedback.attributes != json!({"policy": "pllm.greedy.v1", "source": "execution_policy"})
        || graph.output != feedback.id
    {
        return Err(ModelError::Incomplete(
            "dense Qwen graph tail has invalid norm, token selection, or feedback topology".into(),
        ));
    }
    Ok(DenseQwenOutputContract {
        head_attributes: output_head.attributes.clone(),
        input_shape: last_token.output_shape.clone(),
        output_shape: output_head.output_shape.clone(),
        selection_attributes: selection.attributes.clone(),
        selection_shape: selection.output_shape.clone(),
        feedback_attributes: feedback.attributes.clone(),
        feedback_shape: feedback.output_shape.clone(),
    })
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

    fn qwen2_plan(max_input_tokens: u64, max_new_tokens: u64) -> DecoderPlan {
        lower_qwen_decoder(
            &config(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens,
                max_new_tokens,
            },
        )
        .unwrap()
    }

    fn qwen3_plan(max_input_tokens: u64, max_new_tokens: u64) -> DecoderPlan {
        lower_model_json(
            MINI_CODER_4B_CONFIG,
            DecoderWorkload {
                batch: 1,
                max_input_tokens,
                max_new_tokens,
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

    fn operation_mut<'a>(graph: &'a mut DecoderGraph, id: &str) -> &'a mut ModelOperation {
        graph
            .operations
            .iter_mut()
            .find(|operation| operation.id == id)
            .unwrap()
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
        assert_eq!(plan.prefill.operations.len(), 28 * 24 + 6);
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
    fn dense_qwen_rope_and_cache_contracts_are_explicit() {
        let plan = qwen2_plan(128, 32);
        let rope = operation(&plan.prefill, "layer.0.rope_k");
        assert_eq!(rope.inputs, ["layer.0.k_heads", "input.positions"]);
        assert_eq!(
            rope.attributes,
            json!({
                "theta": 1_000_000,
                "rotary_dimensions": 64,
                "pairing": "split_half",
                "position_policy": "sequential_absolute",
                "coefficient_profile": "pllm.numeric.rope.q30.libm.v1",
                "input_layout": "batch_heads_sequence_feature",
                "output_layout": "batch_heads_sequence_feature",
                "tail_policy": "unchanged"
            })
        );

        assert!(plan.prefill.state_inputs.is_empty());
        assert_eq!(plan.decode.state_inputs.len(), 48);
        for output in &plan.prefill.state_outputs {
            let input = plan
                .decode
                .state_inputs
                .iter()
                .find(|input| input.layer == output.layer && input.kind == output.kind)
                .unwrap();
            assert_eq!(output.shape, input.shape);
            assert_eq!(output.maximum_sequence, 159);
            assert_eq!(input.maximum_sequence, 159);
        }

        let prefill_append = operation(&plan.prefill, "layer.0.key_append");
        assert_eq!(
            prefill_append.inputs,
            [
                "layer.0.rope_k",
                "input.positions",
                "input.attention_mask",
                "input.sequence_lengths"
            ]
        );
        assert_eq!(prefill_append.output_shape, [1, 2, 159, 64]);
        assert_eq!(prefill_append.attributes["mode"], "initialize");
        assert_eq!(prefill_append.attributes["state_capacity"], 159);
        assert_eq!(
            prefill_append.attributes["state_layout"],
            "batch_kv_heads_sequence_feature"
        );
        assert_eq!(
            prefill_append.attributes["attention_domain"],
            json!({
                "layout": "batch_kv_heads_sequence_feature",
                "maximum_sequence": 128,
                "includes_current": true
            })
        );
        let decode_append = operation(&plan.decode, "layer.0.key_append");
        assert_eq!(decode_append.inputs[0], "state.layer.0.key");
        assert_eq!(decode_append.inputs[1], "layer.0.rope_k");
        assert_eq!(decode_append.attributes["mode"], "append");
        assert_eq!(decode_append.attributes["state_capacity"], 159);

        let key_view = operation(&plan.prefill, "layer.0.key_view");
        assert_eq!(key_view.output_shape, [1, 2, 128, 64]);
        assert_eq!(key_view.attributes["axis"], 2);
        assert_eq!(key_view.attributes["maximum_sequence"], 128);
        assert_eq!(key_view.attributes["semantics"], "visible_valid_prefix");
        assert_eq!(
            operation(&plan.prefill, "layer.0.attention_scores").inputs[1],
            "layer.0.key_view"
        );
        assert_eq!(
            operation(&plan.prefill, "layer.0.attention_values").inputs[1],
            "layer.0.value_view"
        );
        assert_eq!(plan.prefill.state_outputs[0].id, "layer.0.key_append");
        assert_eq!(
            operation(&plan.prefill, "last_hidden").inputs,
            ["final_norm", "input.sequence_lengths"]
        );
        assert_eq!(
            operation(&plan.prefill, "last_hidden").attributes,
            json!({
                "axis": 1,
                "selection": "last_valid",
                "valid_lengths_input": "input.sequence_lengths"
            })
        );
        assert_eq!(
            operation(&plan.decode, "last_hidden").inputs,
            ["final_norm"]
        );
        assert_eq!(
            operation(&plan.decode, "last_hidden").attributes,
            json!({"axis": 1})
        );
        let policy = json!({"policy": "pllm.greedy.v1", "source": "execution_policy"});
        for graph in [&plan.prefill, &plan.decode] {
            assert_eq!(operation(graph, "token_selection").attributes, policy);
            assert_eq!(operation(graph, "token_feedback").attributes, policy);
        }
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
        assert_eq!(plan.prefill.operations.len(), 30 * 36 + 6);
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
        assert_eq!(operation("layer.0.q_norm").inputs, ["layer.0.q_heads"]);
        assert_eq!(operation("layer.0.q_norm").output_shape, [1, 32, 128, 128]);
        assert_eq!(
            operation("layer.0.q_norm").attributes,
            json!({
                "epsilon": "1e-6",
                "weight": "model.layers.0.self_attn.q_norm.weight",
                "weight_offset": 0
            })
        );
        assert_eq!(operation("layer.0.k_norm").inputs, ["layer.0.k_heads"]);
        assert_eq!(operation("layer.0.k_norm").output_shape, [1, 8, 128, 128]);
        assert_eq!(
            operation("layer.0.rope_k").inputs,
            ["layer.0.k_norm", "input.positions"]
        );
        assert_eq!(
            operation("layer.0.rope_k").attributes,
            json!({
                "theta": 5_000_000,
                "rotary_dimensions": 128,
                "pairing": "split_half",
                "position_policy": "sequential_absolute",
                "coefficient_profile": "pllm.numeric.rope.q30.libm.v1",
                "input_layout": "batch_heads_sequence_feature",
                "output_layout": "batch_heads_sequence_feature",
                "tail_policy": "unchanged"
            })
        );
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
        plan.decode.state_inputs[1].id = plan.decode.state_inputs[0].id.clone();
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
            .find(|operation| operation.operator == ModelOperator::LastToken)
            .unwrap()
            .attributes["axis"] = json!(3);
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
            .find(|operation| operation.operator == ModelOperator::LastToken)
            .unwrap()
            .output_shape = vec![1, 8];
        assert!(plan.validate().is_err());
    }

    #[test]
    fn dense_qwen_validation_rejects_tampered_semantics_and_shapes() {
        let mut plan = qwen2_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.rope_q").attributes["rotary_dimensions"] =
            json!(63);
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.rope_q").attributes["rotary_dimensions"] =
            json!(62);
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.rope_q").attributes["pairing"] =
            json!("interleaved");
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.rope_q").inputs[1] = "input.tokens".into();
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.key_append").attributes["state_capacity"] =
            json!(8);
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.key_view").output_shape[2] = 7;
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        operation_mut(&mut plan.decode, "layer.0.attention_scores").inputs[1] =
            "layer.0.key_append".into();
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        plan.decode.state_inputs[0].shape[2] = 8;
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.attention_scores").inputs[1] =
            "layer.0.value_view".into();
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.attention_values").inputs[1] =
            "layer.0.key_view".into();
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        let digest = plan.config_digest.clone();
        plan.transformations.push(AppliedMethod {
            component: "unknown/component".into(),
            implementation: "unknown/implementation".into(),
            method_id: "unknown".into(),
            input_digest: digest.clone(),
            configuration_digest: digest,
        });
        operation_mut(&mut plan.prefill, "layer.0.rope_q").attributes["pairing"] =
            json!("interleaved");
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));
    }

    #[test]
    fn dense_qwen_validation_enforces_family_specific_qk_norms() {
        let mut plan = qwen3_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.q_norm").attributes["weight"] =
            json!("wrong.weight");
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen3_plan(8, 2);
        operation_mut(&mut plan.decode, "layer.0.k_norm").attributes["weight_offset"] = json!(1);
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen3_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.q_norm").attributes["epsilon"] =
            json!("0.000002");
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen3_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.k_norm").inputs[0] = "layer.0.q_heads".into();
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        let rope_index = plan
            .prefill
            .operations
            .iter()
            .position(|operation| operation.id == "layer.0.rope_q")
            .unwrap();
        let shape = operation(&plan.prefill, "layer.0.q_heads")
            .output_shape
            .clone();
        plan.prefill.operations.insert(
            rope_index,
            ModelOperation {
                id: "layer.0.q_norm".into(),
                operator: ModelOperator::RmsNorm,
                layer: Some(0),
                state_kind: None,
                inputs: vec!["layer.0.q_heads".into()],
                output_shape: shape,
                attributes: json!({
                    "epsilon": "0.000001",
                    "weight": "model.layers.0.self_attn.q_norm.weight",
                    "weight_offset": 0
                }),
            },
        );
        operation_mut(&mut plan.prefill, "layer.0.rope_q").inputs[0] = "layer.0.q_norm".into();
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));
    }

    #[test]
    fn dense_qwen_validation_rejects_same_shape_attention_rewires() {
        for (operation_id, input_index, replacement) in [
            ("layer.0.k_heads", 0, "layer.0.v_linear"),
            ("layer.0.key_append", 0, "layer.0.v_heads"),
            ("layer.0.value_append", 0, "layer.0.rope_k"),
            ("layer.0.causal_mask", 0, "layer.0.attention_scores"),
            ("layer.0.softmax", 0, "layer.0.attention_scale"),
            ("layer.0.attention_values", 0, "layer.0.causal_mask"),
            ("layer.0.attention_hidden", 0, "layer.0.rope_q"),
            ("layer.0.o_proj", 0, "layer.0.q_linear"),
            ("layer.0.attention_residual", 1, "layer.0.input_norm"),
        ] {
            let mut plan = qwen2_plan(8, 2);
            operation_mut(&mut plan.prefill, operation_id).inputs[input_index] = replacement.into();
            assert!(
                matches!(plan.validate(), Err(ModelError::Incomplete(_))),
                "accepted rewire {operation_id}[{input_index}] -> {replacement}"
            );
        }

        let mut equal_heads = config();
        equal_heads.num_key_value_heads = equal_heads.num_attention_heads;
        let mut plan = lower_qwen_decoder(
            &equal_heads,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 8,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        operation_mut(&mut plan.prefill, "layer.0.attention_scores").inputs[0] =
            "layer.0.rope_k".into();
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));
    }

    #[test]
    fn qwen3_qk_norm_epsilon_tracks_configured_direct_norm_epsilon() {
        let mut document: Value = serde_json::from_slice(MINI_CODER_4B_CONFIG).unwrap();
        document["rms_norm_eps"] = json!("0.000002");
        let mut plan = lower_model_json(
            &serde_json::to_vec(&document).unwrap(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 8,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        assert_eq!(
            operation(&plan.prefill, "layer.0.input_norm").attributes["epsilon"],
            "0.000002"
        );
        assert_eq!(
            operation(&plan.prefill, "layer.0.q_norm").attributes["epsilon"],
            "0.000002"
        );
        plan.validate().unwrap();

        operation_mut(&mut plan.prefill, "layer.0.q_norm").attributes["epsilon"] = json!("2e-6");
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));
    }

    #[test]
    fn dense_qwen_validation_rejects_mlp_interlayer_and_tail_bypasses() {
        for (operation_id, input_index, replacement) in [
            ("layer.0.post_norm", 0, "layer.0.input_norm"),
            ("layer.0.gate_proj", 0, "layer.0.attention_residual"),
            ("layer.0.silu", 0, "layer.0.up_proj"),
            ("layer.0.gated_multiply", 0, "layer.0.up_proj"),
            ("layer.0.down_proj", 0, "layer.0.up_proj"),
            ("layer.0.mlp_residual", 0, "layer.0.input_norm"),
            ("layer.1.input_norm", 0, "layer.0.attention_residual"),
            ("final_norm", 0, "layer.23.attention_residual"),
            ("last_hidden", 0, "layer.23.mlp_residual"),
            ("token_feedback", 0, "output_head"),
        ] {
            let mut plan = qwen2_plan(8, 2);
            operation_mut(&mut plan.prefill, operation_id).inputs[input_index] = replacement.into();
            assert!(
                matches!(plan.validate(), Err(ModelError::Incomplete(_))),
                "accepted rewire {operation_id}[{input_index}] -> {replacement}"
            );
        }

        let mut plan = qwen2_plan(8, 2);
        let last = operation_mut(&mut plan.prefill, "last_hidden");
        last.inputs = vec!["final_norm".into()];
        last.attributes = json!({"axis": 1});
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        let last = operation_mut(&mut plan.decode, "last_hidden");
        last.inputs = vec!["final_norm".into()];
        last.attributes = json!({"axis": 1});
        plan.validate().unwrap();
    }

    #[test]
    fn dense_qwen_validation_rejects_cross_phase_epsilon_and_theta_drift() {
        let mut plan = qwen2_plan(8, 2);
        for operation in &mut plan.decode.operations {
            if operation.operator == ModelOperator::RmsNorm {
                operation.attributes["epsilon"] = json!("0.000002");
            }
        }
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        operation_mut(&mut plan.prefill, "layer.0.rope_k").attributes["theta"] = json!(1_000_001);
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));

        let mut plan = qwen2_plan(8, 2);
        for operation in &mut plan.decode.operations {
            if operation.operator == ModelOperator::RotaryEmbedding {
                operation.attributes["theta"] = json!(1_000_001);
            }
        }
        assert!(matches!(plan.validate(), Err(ModelError::Incomplete(_))));
    }

    #[test]
    fn dense_qwen_validation_rejects_cross_phase_output_head_drift() {
        let plan = qwen2_plan(8, 2);
        assert_eq!(
            operation(&plan.prefill, "output_head").attributes,
            json!({
                "weight": "model.embed_tokens.weight",
                "tied": true,
                "input_layout": "batch_hidden",
                "output_layout": "batch_vocabulary",
                "compute_dtype": "model_native",
                "output_dtype": "model_native"
            })
        );

        let mut tampered = plan.clone();
        let output = operation_mut(&mut tampered.prefill, "output_head");
        output.attributes["weight"] = json!("lm_head.weight");
        output.attributes["tied"] = json!(false);
        assert!(matches!(
            tampered.validate(),
            Err(ModelError::Incomplete(_))
        ));

        let mut tampered = plan.clone();
        operation_mut(&mut tampered.prefill, "output_head").output_shape[1] -= 1;
        assert!(matches!(
            tampered.validate(),
            Err(ModelError::Incomplete(_))
        ));

        let mut tampered = plan;
        operation_mut(&mut tampered.decode, "output_head").attributes["output_layout"] =
            json!("batch_hidden");
        assert!(matches!(
            tampered.validate(),
            Err(ModelError::Incomplete(_))
        ));
    }
}
