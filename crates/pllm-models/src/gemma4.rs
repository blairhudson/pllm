use super::{
    decimal_string, integral_u64, DecoderGraph, DecoderMode, DecoderPlan, DecoderWorkload,
    ModelError, ModelOperation, ModelOperator, StateKind, StateTensor, DECODER_PLAN_SCHEMA_VERSION,
};
use pllm_types::canonical_digest;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const GEMMA4_E4B_SOURCE_REVISION: &str = "ee0ef6023621cff504d758262d4e04895a5af4a2";
pub const GEMMA4_E2B_SOURCE_REVISION: &str = "3e22461f65e89153144f8adb70e3b8c2cc9845a7";

const VOCAB_SIZE: u64 = 262_144;
const NUM_HEADS: u32 = 8;
const LOCAL_HEAD_DIM: u64 = 256;
const GLOBAL_HEAD_DIM: u64 = 512;
const SLIDING_WINDOW: u64 = 512;
const MAX_POSITIONS: u64 = 131_072;
const PLE_DIM: u64 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Gemma4TextVariant {
    adapter: &'static str,
    digest_domain: &'static str,
    source_revision: &'static str,
    hidden_size: u64,
    intermediate_size: u64,
    num_layers: u32,
    num_kv_heads: u32,
    shared_kv_layers: u32,
    full_attention_period: u32,
    double_wide_mlp: bool,
}

const E4B: Gemma4TextVariant = Gemma4TextVariant {
    adapter: "pllm.gemma4_e4b_text.v1",
    digest_domain: "pllm.gemma4_e4b_text_config.v1",
    source_revision: GEMMA4_E4B_SOURCE_REVISION,
    hidden_size: 2_560,
    intermediate_size: 10_240,
    num_layers: 42,
    num_kv_heads: 2,
    shared_kv_layers: 18,
    full_attention_period: 6,
    double_wide_mlp: false,
};

const E2B: Gemma4TextVariant = Gemma4TextVariant {
    adapter: "pllm.gemma4_e2b_text.v1",
    digest_domain: "pllm.gemma4_e2b_text_config.v1",
    source_revision: GEMMA4_E2B_SOURCE_REVISION,
    hidden_size: 1_536,
    intermediate_size: 6_144,
    num_layers: 35,
    num_kv_heads: 1,
    shared_kv_layers: 20,
    full_attention_period: 5,
    double_wide_mlp: true,
};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Gemma4TextConfig {
    pub model_type: String,
    pub vocab_size: u64,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub num_global_key_value_heads: Value,
    pub head_dim: u64,
    pub global_head_dim: u64,
    pub sliding_window: u64,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: u64,
    pub hidden_activation: String,
    #[serde(deserialize_with = "decimal_string")]
    pub rms_norm_eps: String,
    #[serde(deserialize_with = "decimal_string")]
    pub final_logit_softcapping: String,
    pub hidden_size_per_layer_input: u64,
    pub vocab_size_per_layer_input: u64,
    pub num_kv_shared_layers: u32,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub attention_bias: bool,
    #[serde(deserialize_with = "decimal_string")]
    pub attention_dropout: String,
    pub attention_k_eq_v: bool,
    pub enable_moe_block: bool,
    pub num_experts: Value,
    pub top_k_experts: Value,
    pub expert_intermediate_size: Value,
    pub use_double_wide_mlp: bool,
    pub use_bidirectional_attention: Value,
    pub rope_parameters: Gemma4RopeParameters,
    pub dtype: String,
    #[serde(default)]
    pub per_layer_config: Option<Value>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Gemma4RopeParameters {
    pub sliding_attention: Gemma4RopeConfig,
    pub full_attention: Gemma4RopeConfig,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Gemma4RopeConfig {
    pub rope_type: String,
    #[serde(deserialize_with = "integral_u64")]
    pub rope_theta: u64,
    #[serde(default, deserialize_with = "optional_decimal_string")]
    pub partial_rotary_factor: Option<String>,
}

impl Gemma4TextConfig {
    fn validate(&self) -> Result<&'static Gemma4TextVariant, ModelError> {
        if self.model_type != "gemma4_text" {
            return Err(ModelError::Unsupported(format!(
                "model_type {} is not gemma4_text",
                self.model_type
            )));
        }
        let variant = match self.num_hidden_layers {
            42 => &E4B,
            35 => &E2B,
            _ => {
                return Err(unsupported("num_hidden_layers", "35 (E2B) or 42 (E4B)"));
            }
        };
        exact(self.vocab_size, VOCAB_SIZE, "vocab_size")?;
        exact(self.hidden_size, variant.hidden_size, "hidden_size")?;
        exact(
            self.intermediate_size,
            variant.intermediate_size,
            "intermediate_size",
        )?;
        exact(
            self.num_hidden_layers,
            variant.num_layers,
            "num_hidden_layers",
        )?;
        exact(self.num_attention_heads, NUM_HEADS, "num_attention_heads")?;
        exact(
            self.num_key_value_heads,
            variant.num_kv_heads,
            "num_key_value_heads",
        )?;
        exact(self.head_dim, LOCAL_HEAD_DIM, "head_dim")?;
        exact(self.global_head_dim, GLOBAL_HEAD_DIM, "global_head_dim")?;
        exact(self.sliding_window, SLIDING_WINDOW, "sliding_window")?;
        exact(
            self.max_position_embeddings,
            MAX_POSITIONS,
            "max_position_embeddings",
        )?;
        exact(
            self.hidden_size_per_layer_input,
            PLE_DIM,
            "hidden_size_per_layer_input",
        )?;
        exact(
            self.vocab_size_per_layer_input,
            VOCAB_SIZE,
            "vocab_size_per_layer_input",
        )?;
        exact(
            self.num_kv_shared_layers,
            variant.shared_kv_layers,
            "num_kv_shared_layers",
        )?;
        if self.layer_types != default_layer_types(variant) {
            return Err(unsupported(
                "layer_types",
                "the official E2B or E4B attention schedule",
            ));
        }
        if self.hidden_activation != "gelu_pytorch_tanh" {
            return Err(unsupported("hidden_activation", "gelu_pytorch_tanh"));
        }
        decimal_exact(&self.rms_norm_eps, 0.000_001, "rms_norm_eps")?;
        decimal_exact(
            &self.final_logit_softcapping,
            30.0,
            "final_logit_softcapping",
        )?;
        decimal_exact(&self.attention_dropout, 0.0, "attention_dropout")?;
        if !self.tie_word_embeddings {
            return Err(unsupported("tie_word_embeddings", "true"));
        }
        if !self.use_cache {
            return Err(unsupported("use_cache", "true"));
        }
        if self.attention_bias {
            return Err(unsupported("attention_bias", "false"));
        }
        if self.attention_k_eq_v {
            return Err(unsupported("attention_k_eq_v", "false"));
        }
        if self.enable_moe_block
            || !self.num_experts.is_null()
            || !self.top_k_experts.is_null()
            || !self.expert_intermediate_size.is_null()
        {
            return Err(unsupported("MoE configuration", "disabled"));
        }
        exact(
            self.use_double_wide_mlp,
            variant.double_wide_mlp,
            "use_double_wide_mlp",
        )?;
        if !self.use_bidirectional_attention.is_null() {
            return Err(unsupported("use_bidirectional_attention", "null"));
        }
        if !self.num_global_key_value_heads.is_null() {
            return Err(unsupported("num_global_key_value_heads", "null"));
        }
        if self.dtype != "bfloat16" {
            return Err(unsupported("dtype", "bfloat16"));
        }
        if self
            .per_layer_config
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return Err(unsupported("per_layer_config overrides", "absent"));
        }
        const NON_SEMANTIC_FIELDS: &[&str] = &[
            "_name_or_path",
            "architectures",
            "bos_token_id",
            "eos_token_id",
            "id2label",
            "initializer_range",
            "label2id",
            "name_or_path",
            "output_attentions",
            "output_hidden_states",
            "pad_token_id",
            "problem_type",
            "return_dict",
            "torch_dtype",
            "transformers_version",
        ];
        if let Some(field) = self
            .extra
            .keys()
            .find(|field| !NON_SEMANTIC_FIELDS.contains(&field.as_str()))
        {
            return Err(ModelError::Unsupported(format!(
                "unrecognized gemma4_text semantic field {field}"
            )));
        }
        validate_rope(
            &self.rope_parameters.sliding_attention,
            "default",
            10_000,
            None,
            "sliding_attention",
        )?;
        validate_rope(
            &self.rope_parameters.full_attention,
            "proportional",
            1_000_000,
            Some(0.25),
            "full_attention",
        )?;
        Ok(variant)
    }

    fn digest(&self, variant: &Gemma4TextVariant) -> pllm_types::Digest {
        canonical_digest(
            variant.digest_domain,
            &json!({
                "attention_bias": false,
                "attention_dropout": "0",
                "attention_k_eq_v": false,
                "enable_moe_block": false,
                "final_logit_softcapping": 30,
                "global_head_dim": GLOBAL_HEAD_DIM,
                "head_dim": LOCAL_HEAD_DIM,
                "hidden_activation": "gelu_pytorch_tanh",
                "dtype": "bfloat16",
                "hidden_size": variant.hidden_size,
                "hidden_size_per_layer_input": PLE_DIM,
                "intermediate_size": variant.intermediate_size,
                "layer_types": self.layer_types,
                "max_position_embeddings": MAX_POSITIONS,
                "model_type": "gemma4_text",
                "num_attention_heads": NUM_HEADS,
                "num_hidden_layers": variant.num_layers,
                "num_key_value_heads": variant.num_kv_heads,
                "num_kv_shared_layers": variant.shared_kv_layers,
                "rms_norm_eps": "1/1000000",
                "rope_parameters": {
                    "full_attention": {"partial_rotary_factor": "1/4", "rope_theta": 1_000_000, "rope_type": "proportional"},
                    "sliding_attention": {"rope_theta": 10_000, "rope_type": "default"}
                },
                "sliding_window": SLIDING_WINDOW,
                "tie_word_embeddings": true,
                "use_bidirectional_attention": null,
                "use_cache": true,
                "use_double_wide_mlp": variant.double_wide_mlp,
                "vocab_size": VOCAB_SIZE,
                "vocab_size_per_layer_input": VOCAB_SIZE
            }),
        )
    }
}

pub(super) fn lower_gemma4_json(
    document: Value,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    if document.get("model_type").and_then(Value::as_str) != Some("gemma4") {
        return Err(ModelError::Unsupported(
            "pinned Gemma adapter requires outer model_type gemma4".into(),
        ));
    }
    let architectures = document.get("architectures").and_then(Value::as_array);
    if architectures.is_none_or(|architectures| {
        architectures.len() != 1
            || architectures[0].as_str() != Some("Gemma4ForConditionalGeneration")
    }) {
        return Err(unsupported(
            "outer architectures",
            "[Gemma4ForConditionalGeneration]",
        ));
    }
    if document.get("tie_word_embeddings").and_then(Value::as_bool) != Some(true) {
        return Err(unsupported("outer tie_word_embeddings", "true"));
    }
    if document.get("dtype").and_then(Value::as_str) != Some("bfloat16") {
        return Err(unsupported("outer dtype", "bfloat16"));
    }
    let text = document
        .get("text_config")
        .cloned()
        .ok_or_else(|| ModelError::InvalidConfig("gemma4 text_config is missing".into()))?;
    let config: Gemma4TextConfig = serde_json::from_value(text).map_err(|error| {
        ModelError::InvalidConfig(format!("invalid gemma4 text_config: {error}"))
    })?;
    lower(&config, workload)
}

fn lower(config: &Gemma4TextConfig, workload: DecoderWorkload) -> Result<DecoderPlan, ModelError> {
    let variant = config.validate()?;
    if workload.batch == 0 || workload.max_input_tokens == 0 || workload.max_new_tokens == 0 {
        return Err(ModelError::InvalidConfig(
            "decoder workload bounds must be nonzero".into(),
        ));
    }
    let processed_positions = workload
        .max_input_tokens
        .checked_add(workload.max_new_tokens)
        .and_then(|total| total.checked_sub(1))
        .ok_or_else(|| ModelError::InvalidConfig("decoder position bound overflowed".into()))?;
    if processed_positions > MAX_POSITIONS {
        return Err(ModelError::InvalidConfig(format!(
            "decoder requires {processed_positions} processed positions but model permits {MAX_POSITIONS}"
        )));
    }
    let plan = DecoderPlan {
        schema_version: DECODER_PLAN_SCHEMA_VERSION.into(),
        model_family: "gemma4_text".into(),
        adapter: variant.adapter.into(),
        config_digest: config.digest(variant),
        transformations: Vec::new(),
        prefill: lower_graph(
            config,
            variant,
            DecoderMode::Prefill,
            workload.batch,
            workload.max_input_tokens,
            processed_positions,
        ),
        decode: lower_graph(
            config,
            variant,
            DecoderMode::Decode,
            workload.batch,
            1,
            processed_positions,
        ),
        token_feedback: true,
    };
    plan.validate()?;
    Ok(plan)
}

fn lower_graph(
    config: &Gemma4TextConfig,
    variant: &Gemma4TextVariant,
    mode: DecoderMode,
    batch: u64,
    query: u64,
    state_capacity: u64,
) -> DecoderGraph {
    let model = "model.language_model";
    let hidden_shape = vec![batch, query, variant.hidden_size];
    let packed_ple_shape = vec![batch, query, u64::from(variant.num_layers) * PLE_DIM];
    let ple_shape = vec![batch, query, u64::from(variant.num_layers), PLE_DIM];
    let mut operations = Vec::new();
    push(
        &mut operations,
        "main_embedding",
        ModelOperator::TokenLookup,
        &["input.tokens"],
        hidden_shape.clone(),
        json!({
            "weight": format!("{model}.embed_tokens.weight"),
            "checkpoint_layout": "conditional_generation",
            "adapter_reference_revision": variant.source_revision
        }),
    );
    scale(
        &mut operations,
        "main_embedding_scaled",
        "main_embedding",
        hidden_shape.clone(),
        json!({
            "kind": "sqrt",
            "radicand": variant.hidden_size,
            "factor_source_dtype": "float32_buffer",
            "factor_rounding_dtype": "bfloat16",
            "compute_dtype": "bfloat16",
            "output_dtype": "bfloat16"
        }),
    );
    push(
        &mut operations,
        "ple_token_embedding",
        ModelOperator::TokenLookup,
        &["input.tokens"],
        packed_ple_shape.clone(),
        json!({"weight": format!("{model}.embed_tokens_per_layer.weight")}),
    );
    scale(
        &mut operations,
        "ple_token_scaled",
        "ple_token_embedding",
        packed_ple_shape.clone(),
        json!({
            "kind": "sqrt",
            "radicand": PLE_DIM,
            "factor_source_dtype": "float32_buffer",
            "factor_rounding_dtype": "bfloat16",
            "compute_dtype": "bfloat16",
            "output_dtype": "bfloat16"
        }),
    );
    push(
        &mut operations,
        "ple_token_reshape",
        ModelOperator::Reshape,
        &["ple_token_scaled"],
        ple_shape.clone(),
        json!({"layout": "batch_sequence_layer_feature"}),
    );
    linear(
        &mut operations,
        "ple_context_projection",
        "main_embedding_scaled",
        packed_ple_shape.clone(),
        format!("{model}.per_layer_model_projection.weight"),
    );
    scale(
        &mut operations,
        "ple_context_scaled",
        "ple_context_projection",
        packed_ple_shape.clone(),
        json!({
            "kind": "inverse_sqrt",
            "radicand": variant.hidden_size,
            "factor_source_dtype": "python_float64",
            "factor_rounding_dtype": "bfloat16",
            "compute_dtype": "bfloat16",
            "output_dtype": "bfloat16"
        }),
    );
    push(
        &mut operations,
        "ple_context_reshape",
        ModelOperator::Reshape,
        &["ple_context_scaled"],
        ple_shape.clone(),
        json!({"layout": "batch_sequence_layer_feature"}),
    );
    rms_norm(
        &mut operations,
        "ple_context_norm",
        "ple_context_reshape",
        ple_shape.clone(),
        Some(format!("{model}.per_layer_projection_norm.weight")),
    );
    push(
        &mut operations,
        "ple_combined",
        ModelOperator::ResidualAdd,
        &["ple_token_reshape", "ple_context_norm"],
        ple_shape.clone(),
        json!({"semantics": "ple_token_plus_context"}),
    );
    scale(
        &mut operations,
        "ple_combined_scaled",
        "ple_combined",
        ple_shape,
        json!({
            "kind": "inverse_sqrt",
            "radicand": 2,
            "factor_source_dtype": "python_float64",
            "factor_rounding_dtype": "bfloat16",
            "compute_dtype": "bfloat16",
            "output_dtype": "bfloat16"
        }),
    );

    let mut hidden = "main_embedding_scaled".to_owned();
    let mut state_inputs = Vec::new();
    let mut state_outputs = Vec::new();
    let first_shared_layer = variant.num_layers - variant.shared_kv_layers;
    for layer in 0..variant.num_layers {
        let prefix = format!("layer.{layer}");
        let weights = format!("{model}.layers.{layer}");
        let layer_type = &config.layer_types[layer as usize];
        let sliding = layer_type == "sliding_attention";
        let head_dim = if sliding {
            LOCAL_HEAD_DIM
        } else {
            GLOBAL_HEAD_DIM
        };
        let q_projection_shape = vec![batch, query, u64::from(NUM_HEADS), head_dim];
        let q_attention_shape = vec![batch, u64::from(NUM_HEADS), query, head_dim];
        let kv_projection_shape = vec![batch, query, u64::from(variant.num_kv_heads), head_dim];
        let residual = hidden.clone();
        let input_norm = format!("{prefix}.input_norm");
        rms_norm(
            &mut operations,
            &input_norm,
            &hidden,
            hidden_shape.clone(),
            Some(format!("{weights}.input_layernorm.weight")),
        );
        let q_linear = format!("{prefix}.q_linear");
        linear(
            &mut operations,
            &q_linear,
            &input_norm,
            vec![batch, query, u64::from(NUM_HEADS) * head_dim],
            format!("{weights}.self_attn.q_proj.weight"),
        );
        let q_heads = format!("{prefix}.q_heads");
        push(
            &mut operations,
            &q_heads,
            ModelOperator::Reshape,
            &[&q_linear],
            q_projection_shape.clone(),
            json!({"layout": "batch_sequence_heads_feature"}),
        );
        let q_norm = format!("{prefix}.q_norm");
        rms_norm(
            &mut operations,
            &q_norm,
            &q_heads,
            q_projection_shape.clone(),
            Some(format!("{weights}.self_attn.q_norm.weight")),
        );
        let rope_q = format!("{prefix}.rope_q");
        push(
            &mut operations,
            &rope_q,
            ModelOperator::RotaryEmbedding,
            &[&q_norm, "input.positions"],
            q_projection_shape,
            rope_attributes(sliding, head_dim),
        );
        let q_permute = format!("{prefix}.q_permute");
        permute(
            &mut operations,
            &q_permute,
            &rope_q,
            q_attention_shape.clone(),
        );

        let source_layer = if layer < first_shared_layer {
            layer
        } else {
            shared_kv_source(config, first_shared_layer, layer_type)
                .expect("validated schedule has a nonshared source for every attention type")
        };
        let (attention_key, attention_value, attention_bound) = if layer < first_shared_layer {
            producer_kv(
                &mut operations,
                &mut state_inputs,
                &mut state_outputs,
                &prefix,
                &weights,
                layer,
                sliding,
                head_dim,
                batch,
                query,
                state_capacity,
                mode,
                &input_norm,
                &kv_projection_shape,
                variant.num_kv_heads,
            )
        } else {
            (
                format!("layer.{source_layer}.key_append"),
                format!("layer.{source_layer}.value_append"),
                attention_bound(sliding, state_capacity),
            )
        };
        let score_shape = vec![batch, u64::from(NUM_HEADS), query, attention_bound];
        let scores = format!("{prefix}.attention_scores");
        push(
            &mut operations,
            &scores,
            ModelOperator::AttentionScores,
            &[&q_permute, &attention_key],
            score_shape.clone(),
            json!({
                "group_size": NUM_HEADS / variant.num_kv_heads,
                "key_value_source_layer": source_layer,
                "key_layout": if sliding { "batch_kv_heads_query_window_feature" } else { "batch_kv_heads_sequence_feature" }
            }),
        );
        let scaled = format!("{prefix}.attention_scale");
        scale_operator(
            &mut operations,
            &scaled,
            ModelOperator::AttentionScale,
            &scores,
            score_shape.clone(),
            json!({
                "kind": "rational",
                "numerator": 1,
                "denominator": 1,
                "factor_source_dtype": "exact_integer",
                "factor_rounding_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "output_dtype": "bfloat16"
            }),
        );
        let mask = format!("{prefix}.causal_mask");
        push(
            &mut operations,
            &mask,
            ModelOperator::CausalMask,
            &[
                &scaled,
                "input.positions",
                "input.attention_mask",
                "input.sequence_lengths",
            ],
            score_shape.clone(),
            if sliding {
                json!({
                    "kind": "sliding_causal",
                    "sliding_window": SLIDING_WINDOW,
                    "left_context": SLIDING_WINDOW - 1,
                    "includes_current": true,
                    "absolute_positions_input": "input.positions",
                    "padding_mask_input": "input.attention_mask",
                    "valid_lengths_input": "input.sequence_lengths",
                    "cache_validity": "valid_lengths_bounded_suffix",
                    "key_domain": "query_relative_window"
                })
            } else {
                json!({
                    "kind": "full_causal",
                    "maximum_position_embeddings": MAX_POSITIONS,
                    "absolute_positions_input": "input.positions",
                    "padding_mask_input": "input.attention_mask",
                    "valid_lengths_input": "input.sequence_lengths",
                    "cache_validity": "valid_lengths_fixed_capacity",
                    "key_domain": "fixed_capacity"
                })
            },
        );
        let probabilities = format!("{prefix}.softmax");
        push(
            &mut operations,
            &probabilities,
            ModelOperator::Softmax,
            &[&mask],
            score_shape,
            json!({"axis": -1, "compute_dtype": "float32", "output_dtype": "bfloat16"}),
        );
        let values = format!("{prefix}.attention_values");
        push(
            &mut operations,
            &values,
            ModelOperator::AttentionValues,
            &[&probabilities, &attention_value],
            q_attention_shape.clone(),
            json!({
                "group_size": NUM_HEADS / variant.num_kv_heads,
                "value_layout": if sliding { "batch_kv_heads_query_window_feature" } else { "batch_kv_heads_sequence_feature" }
            }),
        );
        let attention_permute = format!("{prefix}.attention_permute");
        permute(
            &mut operations,
            &attention_permute,
            &values,
            vec![batch, query, u64::from(NUM_HEADS), head_dim],
        );
        let attention_hidden = format!("{prefix}.attention_hidden");
        push(
            &mut operations,
            &attention_hidden,
            ModelOperator::Reshape,
            &[&attention_permute],
            vec![batch, query, u64::from(NUM_HEADS) * head_dim],
            json!({"layout": "batch_sequence_hidden"}),
        );
        let o_proj = format!("{prefix}.o_proj");
        linear(
            &mut operations,
            &o_proj,
            &attention_hidden,
            hidden_shape.clone(),
            format!("{weights}.self_attn.o_proj.weight"),
        );
        let post_attention = format!("{prefix}.post_attention_norm");
        rms_norm(
            &mut operations,
            &post_attention,
            &o_proj,
            hidden_shape.clone(),
            Some(format!("{weights}.post_attention_layernorm.weight")),
        );
        let attention_residual = format!("{prefix}.attention_residual");
        push(
            &mut operations,
            &attention_residual,
            ModelOperator::ResidualAdd,
            &[&residual, &post_attention],
            hidden_shape.clone(),
            json!({}),
        );
        let pre_ffn = format!("{prefix}.pre_feedforward_norm");
        rms_norm(
            &mut operations,
            &pre_ffn,
            &attention_residual,
            hidden_shape.clone(),
            Some(format!("{weights}.pre_feedforward_layernorm.weight")),
        );
        let gate = format!("{prefix}.gate_proj");
        let up = format!("{prefix}.up_proj");
        let intermediate_size = if variant.double_wide_mlp && layer >= first_shared_layer {
            variant
                .intermediate_size
                .checked_mul(2)
                .expect("validated Gemma MLP width fits u64")
        } else {
            variant.intermediate_size
        };
        let intermediate_shape = vec![batch, query, intermediate_size];
        linear(
            &mut operations,
            &gate,
            &pre_ffn,
            intermediate_shape.clone(),
            format!("{weights}.mlp.gate_proj.weight"),
        );
        linear(
            &mut operations,
            &up,
            &pre_ffn,
            intermediate_shape.clone(),
            format!("{weights}.mlp.up_proj.weight"),
        );
        let activated = format!("{prefix}.gelu_tanh");
        push(
            &mut operations,
            &activated,
            ModelOperator::GeluTanh,
            &[&gate],
            intermediate_shape.clone(),
            json!({"approximation": "tanh", "compute_dtype": "bfloat16", "output_dtype": "bfloat16"}),
        );
        let gated = format!("{prefix}.gated_multiply");
        push(
            &mut operations,
            &gated,
            ModelOperator::Multiply,
            &[&activated, &up],
            intermediate_shape,
            json!({}),
        );
        let down = format!("{prefix}.down_proj");
        linear(
            &mut operations,
            &down,
            &gated,
            hidden_shape.clone(),
            format!("{weights}.mlp.down_proj.weight"),
        );
        let post_ffn = format!("{prefix}.post_feedforward_norm");
        rms_norm(
            &mut operations,
            &post_ffn,
            &down,
            hidden_shape.clone(),
            Some(format!("{weights}.post_feedforward_layernorm.weight")),
        );
        let ffn_residual = format!("{prefix}.feedforward_residual");
        push(
            &mut operations,
            &ffn_residual,
            ModelOperator::ResidualAdd,
            &[&attention_residual, &post_ffn],
            hidden_shape.clone(),
            json!({}),
        );
        let ple_slice = format!("{prefix}.ple_slice");
        push(
            &mut operations,
            &ple_slice,
            ModelOperator::Slice,
            &["ple_combined_scaled"],
            vec![batch, query, PLE_DIM],
            json!({"axis": 2, "start": layer, "end": layer + 1, "squeeze": true}),
        );
        let ple_gate = format!("{prefix}.ple_gate");
        linear(
            &mut operations,
            &ple_gate,
            &ffn_residual,
            vec![batch, query, PLE_DIM],
            format!("{weights}.per_layer_input_gate.weight"),
        );
        let ple_activated = format!("{prefix}.ple_gelu_tanh");
        push(
            &mut operations,
            &ple_activated,
            ModelOperator::GeluTanh,
            &[&ple_gate],
            vec![batch, query, PLE_DIM],
            json!({"approximation": "tanh", "compute_dtype": "bfloat16", "output_dtype": "bfloat16"}),
        );
        let ple_gated = format!("{prefix}.ple_multiply");
        push(
            &mut operations,
            &ple_gated,
            ModelOperator::Multiply,
            &[&ple_activated, &ple_slice],
            vec![batch, query, PLE_DIM],
            json!({}),
        );
        let ple_projection = format!("{prefix}.ple_projection");
        linear(
            &mut operations,
            &ple_projection,
            &ple_gated,
            hidden_shape.clone(),
            format!("{weights}.per_layer_projection.weight"),
        );
        let ple_norm = format!("{prefix}.ple_norm");
        rms_norm(
            &mut operations,
            &ple_norm,
            &ple_projection,
            hidden_shape.clone(),
            Some(format!("{weights}.post_per_layer_input_norm.weight")),
        );
        let ple_residual = format!("{prefix}.ple_residual");
        push(
            &mut operations,
            &ple_residual,
            ModelOperator::ResidualAdd,
            &[&ffn_residual, &ple_norm],
            hidden_shape.clone(),
            json!({}),
        );
        let layer_scaled = format!("{prefix}.layer_scalar");
        scale(
            &mut operations,
            &layer_scaled,
            &ple_residual,
            hidden_shape.clone(),
            json!({
                "kind": "learned",
                "weight": format!("{weights}.layer_scalar"),
                "factor_source_dtype": "bfloat16",
                "factor_rounding_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "output_dtype": "bfloat16"
            }),
        );
        hidden = layer_scaled;
    }
    rms_norm(
        &mut operations,
        "final_norm",
        &hidden,
        hidden_shape,
        Some(format!("{model}.norm.weight")),
    );
    push(
        &mut operations,
        "last_hidden",
        ModelOperator::LastToken,
        &["final_norm", "input.sequence_lengths"],
        vec![batch, variant.hidden_size],
        json!({"axis": 1, "selection": "last_valid", "valid_lengths_input": "input.sequence_lengths"}),
    );
    push(
        &mut operations,
        "output_head",
        ModelOperator::OutputHead,
        &["last_hidden"],
        vec![batch, VOCAB_SIZE],
        json!({"weight": format!("{model}.embed_tokens.weight"), "tied": true}),
    );
    push(
        &mut operations,
        "logit_softcap",
        ModelOperator::Softcap,
        &["output_head"],
        vec![batch, VOCAB_SIZE],
        json!({"cap": 30, "formula": "cap*tanh(input/cap)", "compute_dtype": "bfloat16", "output_dtype": "bfloat16"}),
    );
    push(
        &mut operations,
        "token_selection",
        ModelOperator::GreedyTokenSelection,
        &["logit_softcap"],
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
        query_sequence: query,
        maximum_key_sequence: state_capacity,
        operations,
        state_inputs,
        state_outputs,
        output: "token_feedback".into(),
    }
}

#[allow(clippy::too_many_arguments)]
fn producer_kv(
    operations: &mut Vec<ModelOperation>,
    state_inputs: &mut Vec<StateTensor>,
    state_outputs: &mut Vec<StateTensor>,
    prefix: &str,
    weights: &str,
    layer: u32,
    sliding: bool,
    head_dim: u64,
    batch: u64,
    query: u64,
    state_capacity: u64,
    mode: DecoderMode,
    input_norm: &str,
    kv_projection_shape: &[u64],
    num_kv_heads: u32,
) -> (String, String, u64) {
    let kv_width = u64::from(num_kv_heads) * head_dim;
    let k_linear = format!("{prefix}.k_linear");
    let v_linear = format!("{prefix}.v_linear");
    linear(
        operations,
        &k_linear,
        input_norm,
        vec![batch, query, kv_width],
        format!("{weights}.self_attn.k_proj.weight"),
    );
    linear(
        operations,
        &v_linear,
        input_norm,
        vec![batch, query, kv_width],
        format!("{weights}.self_attn.v_proj.weight"),
    );
    let k_heads = format!("{prefix}.k_heads");
    let v_heads = format!("{prefix}.v_heads");
    push(
        operations,
        &k_heads,
        ModelOperator::Reshape,
        &[&k_linear],
        kv_projection_shape.to_vec(),
        json!({"layout": "batch_sequence_heads_feature"}),
    );
    push(
        operations,
        &v_heads,
        ModelOperator::Reshape,
        &[&v_linear],
        kv_projection_shape.to_vec(),
        json!({"layout": "batch_sequence_heads_feature"}),
    );
    let k_norm = format!("{prefix}.k_norm");
    let v_norm = format!("{prefix}.v_norm");
    rms_norm(
        operations,
        &k_norm,
        &k_heads,
        kv_projection_shape.to_vec(),
        Some(format!("{weights}.self_attn.k_norm.weight")),
    );
    rms_norm(
        operations,
        &v_norm,
        &v_heads,
        kv_projection_shape.to_vec(),
        None,
    );
    let rope_k = format!("{prefix}.rope_k");
    push(
        operations,
        &rope_k,
        ModelOperator::RotaryEmbedding,
        &[&k_norm, "input.positions"],
        kv_projection_shape.to_vec(),
        rope_attributes(sliding, head_dim),
    );
    let k_permute = format!("{prefix}.k_permute");
    let v_permute = format!("{prefix}.v_permute");
    let kv_attention_shape = vec![batch, u64::from(num_kv_heads), query, head_dim];
    permute(operations, &k_permute, &rope_k, kv_attention_shape.clone());
    permute(operations, &v_permute, &v_norm, kv_attention_shape);
    let bound = attention_bound(sliding, state_capacity);
    let persistent = if sliding {
        SLIDING_WINDOW - 1
    } else {
        state_capacity
    };
    let state_shape = vec![batch, u64::from(num_kv_heads), persistent, head_dim];
    let append_shape = if sliding {
        vec![
            batch,
            u64::from(num_kv_heads),
            query,
            SLIDING_WINDOW,
            head_dim,
        ]
    } else {
        state_shape.clone()
    };
    let key_state = format!("state.layer.{layer}.key");
    let value_state = format!("state.layer.{layer}.value");
    if mode == DecoderMode::Decode {
        state_inputs.push(StateTensor {
            id: key_state.clone(),
            layer: Some(u64::from(layer)),
            kind: StateKind::Key,
            shape: state_shape.clone(),
            maximum_sequence: persistent,
        });
        state_inputs.push(StateTensor {
            id: value_state.clone(),
            layer: Some(u64::from(layer)),
            kind: StateKind::Value,
            shape: state_shape.clone(),
            maximum_sequence: persistent,
        });
    }
    let key_append = format!("{prefix}.key_append");
    let value_append = format!("{prefix}.value_append");
    cache_update(
        operations,
        &key_append,
        &key_state,
        &k_permute,
        append_shape.clone(),
        persistent,
        bound,
        sliding,
        mode,
        StateKind::Key,
    );
    cache_update(
        operations,
        &value_append,
        &value_state,
        &v_permute,
        append_shape,
        persistent,
        bound,
        sliding,
        mode,
        StateKind::Value,
    );
    if sliding {
        let key_suffix = format!("{prefix}.key_suffix");
        let value_suffix = format!("{prefix}.value_suffix");
        cache_suffix(
            operations,
            &key_suffix,
            &key_append,
            state_shape.clone(),
            persistent,
            StateKind::Key,
        );
        cache_suffix(
            operations,
            &value_suffix,
            &value_append,
            state_shape.clone(),
            persistent,
            StateKind::Value,
        );
        state_outputs.push(StateTensor {
            id: key_suffix,
            layer: Some(u64::from(layer)),
            kind: StateKind::Key,
            shape: state_shape.clone(),
            maximum_sequence: persistent,
        });
        state_outputs.push(StateTensor {
            id: value_suffix,
            layer: Some(u64::from(layer)),
            kind: StateKind::Value,
            shape: state_shape,
            maximum_sequence: persistent,
        });
    } else {
        state_outputs.push(StateTensor {
            id: key_append.clone(),
            layer: Some(u64::from(layer)),
            kind: StateKind::Key,
            shape: state_shape.clone(),
            maximum_sequence: persistent,
        });
        state_outputs.push(StateTensor {
            id: value_append.clone(),
            layer: Some(u64::from(layer)),
            kind: StateKind::Value,
            shape: state_shape,
            maximum_sequence: persistent,
        });
    }
    (key_append, value_append, bound)
}

fn attention_bound(sliding: bool, state_capacity: u64) -> u64 {
    if sliding {
        SLIDING_WINDOW
    } else {
        state_capacity
    }
}

#[allow(clippy::too_many_arguments)]
fn cache_update(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    state: &str,
    current: &str,
    shape: Vec<u64>,
    state_capacity: u64,
    attention_capacity: u64,
    sliding: bool,
    mode: DecoderMode,
    kind: StateKind,
) {
    let mut inputs = Vec::new();
    if mode == DecoderMode::Decode {
        inputs.push(state.to_owned());
    }
    inputs.extend([
        current.to_owned(),
        "input.positions".into(),
        "input.attention_mask".into(),
        "input.sequence_lengths".into(),
    ]);
    push_owned(
        operations,
        id,
        ModelOperator::KvCacheAppend,
        inputs,
        shape,
        json!({
            "state": state,
            "mode": if mode == DecoderMode::Prefill { "initialize" } else { "append" },
            "state_capacity": state_capacity,
            "state_layout": "batch_kv_heads_sequence_feature",
            "absolute_write_positions_input": "input.positions",
            "padding_mask_input": "input.attention_mask",
            "valid_lengths_input": "input.sequence_lengths",
            "attention_domain": {
                "layout": if sliding { "batch_kv_heads_query_window_feature" } else { "batch_kv_heads_sequence_feature" },
                "maximum_sequence": attention_capacity,
                "includes_current": true
            }
        }),
    );
    operations
        .last_mut()
        .expect("cache update exists")
        .state_kind = Some(kind);
}

fn cache_suffix(
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
            "axis": 3,
            "output_axis": 2,
            "maximum_sequence": maximum_sequence,
            "semantics": "persist_last_valid_past_tokens",
            "absolute_write_positions_input": "input.positions",
            "padding_mask_input": "input.attention_mask",
            "valid_lengths_input": "input.sequence_lengths"
        }),
    );
    operations
        .last_mut()
        .expect("cache suffix exists")
        .state_kind = Some(kind);
}

fn rope_attributes(sliding: bool, head_dim: u64) -> Value {
    if sliding {
        json!({
            "rope_type": "default",
            "theta": 10_000,
            "head_dim": head_dim,
            "partial_rotary_factor": {"numerator": 1, "denominator": 1},
            "attention_scaling": {"numerator": 1, "denominator": 1},
            "frequency_compute_dtype": "float32",
            "output_dtype": "bfloat16"
        })
    } else {
        json!({
            "rope_type": "proportional",
            "theta": 1_000_000,
            "head_dim": head_dim,
            "partial_rotary_factor": {"numerator": 1, "denominator": 4},
            "attention_scaling": {"numerator": 1, "denominator": 1},
            "frequency_compute_dtype": "float32",
            "output_dtype": "bfloat16"
        })
    }
}

fn rms_norm(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    shape: Vec<u64>,
    weight: Option<String>,
) {
    push(
        operations,
        id,
        ModelOperator::RmsNorm,
        &[input],
        shape,
        json!({"epsilon": "1/1000000", "weight": weight, "weight_offset": 1, "with_scale": weight.is_some(), "compute_dtype": "float32", "output_dtype": "bfloat16"}),
    );
}

fn linear(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    shape: Vec<u64>,
    weight: String,
) {
    push(
        operations,
        id,
        ModelOperator::Linear,
        &[input],
        shape,
        json!({"weight": weight, "bias": null}),
    );
}

fn scale(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    shape: Vec<u64>,
    factor: Value,
) {
    scale_operator(operations, id, ModelOperator::Scale, input, shape, factor);
}

fn scale_operator(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    operator: ModelOperator,
    input: &str,
    shape: Vec<u64>,
    factor: Value,
) {
    push(
        operations,
        id,
        operator,
        &[input],
        shape,
        json!({"factor": factor}),
    );
}

fn permute(operations: &mut Vec<ModelOperation>, id: &str, input: &str, shape: Vec<u64>) {
    push(
        operations,
        id,
        ModelOperator::Permute,
        &[input],
        shape,
        json!({"permutation": [0, 2, 1, 3]}),
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

fn push_owned(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    operator: ModelOperator,
    inputs: Vec<String>,
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
        inputs,
        output_shape,
        attributes,
    });
}

fn layer_type(layer: u32, full_attention_period: u32) -> &'static str {
    if layer % full_attention_period == full_attention_period - 1 {
        "full_attention"
    } else {
        "sliding_attention"
    }
}

fn validate_rope(
    config: &Gemma4RopeConfig,
    rope_type: &str,
    theta: u64,
    partial: Option<f64>,
    name: &str,
) -> Result<(), ModelError> {
    if config.rope_type != rope_type || config.rope_theta != theta {
        return Err(unsupported(
            &format!("rope_parameters.{name}"),
            "the official E4B RoPE",
        ));
    }
    match (&config.partial_rotary_factor, partial) {
        (None, None) => Ok(()),
        (Some(value), Some(expected)) => {
            decimal_exact(value, expected, &format!("{name}.partial_rotary_factor"))
        }
        _ => Err(unsupported(
            &format!("{name}.partial_rotary_factor"),
            "the official E4B value",
        )),
    }
}

fn decimal_exact(value: &str, expected: f64, field: &str) -> Result<(), ModelError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| ModelError::InvalidConfig(format!("{field} must be a finite decimal")))?;
    if parsed.is_finite() && parsed == expected {
        Ok(())
    } else {
        Err(unsupported(field, &expected.to_string()))
    }
}

fn exact<T>(actual: T, expected: T, field: &str) -> Result<(), ModelError>
where
    T: Copy + PartialEq + std::fmt::Display,
{
    if actual == expected {
        Ok(())
    } else {
        Err(unsupported(field, &expected.to_string()))
    }
}

fn unsupported(field: &str, expected: &str) -> ModelError {
    ModelError::Unsupported(format!("{field} must be {expected}"))
}

fn optional_decimal_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    match Value::deserialize(deserializer)? {
        Value::Null => Ok(None),
        Value::String(value) => Ok(Some(value)),
        Value::Number(value) => Ok(Some(value.to_string())),
        _ => Err(D::Error::custom(
            "expected a decimal number, string, or null",
        )),
    }
}

fn default_layer_types(variant: &Gemma4TextVariant) -> Vec<String> {
    (0..variant.num_layers)
        .map(|layer| layer_type(layer, variant.full_attention_period))
        .map(str::to_owned)
        .collect()
}

fn shared_kv_source(
    config: &Gemma4TextConfig,
    first_shared_layer: u32,
    layer_type: &str,
) -> Option<u32> {
    config.layer_types[..first_shared_layer as usize]
        .iter()
        .rposition(|candidate| candidate == layer_type)
        .map(|layer| layer as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest as _, Sha256};

    const OFFICIAL_OUTER: &str =
        include_str!("../tests/fixtures/gemma-4-E4B-it-ee0ef602-config.json");
    const OFFICIAL_E2B_OUTER: &str =
        include_str!("../tests/fixtures/gemma-4-E2B-it-3e22461f-config.json");

    fn workload() -> DecoderWorkload {
        DecoderWorkload {
            batch: 2,
            max_input_tokens: 1_024,
            max_new_tokens: 32,
        }
    }

    fn plan() -> DecoderPlan {
        super::super::lower_model_json(OFFICIAL_OUTER.as_bytes(), workload()).unwrap()
    }

    fn e2b_plan() -> DecoderPlan {
        super::super::lower_model_json(OFFICIAL_E2B_OUTER.as_bytes(), workload()).unwrap()
    }

    fn operation<'a>(graph: &'a DecoderGraph, id: &str) -> &'a ModelOperation {
        graph
            .operations
            .iter()
            .find(|operation| operation.id == id)
            .unwrap_or_else(|| panic!("missing operation {id}"))
    }

    fn operation_mut<'a>(graph: &'a mut DecoderGraph, id: &str) -> &'a mut ModelOperation {
        graph
            .operations
            .iter_mut()
            .find(|operation| operation.id == id)
            .unwrap_or_else(|| panic!("missing operation {id}"))
    }

    #[test]
    fn pinned_fixture_has_verified_raw_sha256() {
        let digest = Sha256::digest(OFFICIAL_OUTER.as_bytes());
        assert_eq!(
            format!("{digest:x}"),
            "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
        );
    }

    #[test]
    fn pinned_e2b_fixture_has_verified_raw_sha256() {
        let digest = Sha256::digest(OFFICIAL_E2B_OUTER.as_bytes());
        assert_eq!(
            format!("{digest:x}"),
            "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330"
        );
    }

    #[test]
    fn lowers_e2b_with_variant_dimensions_sharing_and_double_wide_suffix() {
        let plan = e2b_plan();
        plan.validate().unwrap();
        assert_eq!(plan.model_family, "gemma4_text");
        assert_eq!(plan.adapter, "pllm.gemma4_e2b_text.v1");
        assert_eq!(plan.prefill.state_outputs.len(), 30);
        assert_eq!(plan.decode.state_inputs.len(), 30);
        assert_eq!(
            operation(&plan.prefill, "main_embedding").output_shape,
            [2, 1_024, 1_536]
        );
        assert_eq!(
            operation(&plan.prefill, "ple_token_reshape").output_shape,
            [2, 1_024, 35, 256]
        );
        assert_eq!(
            operation(&plan.prefill, "layer.14.gate_proj").output_shape,
            [2, 1_024, 6_144]
        );
        assert_eq!(
            operation(&plan.prefill, "layer.15.gate_proj").output_shape,
            [2, 1_024, 12_288]
        );
        assert_eq!(
            operation(&plan.prefill, "layer.34.attention_scores").inputs[1],
            "layer.14.key_append"
        );
        assert_eq!(
            operation(&plan.prefill, "main_embedding").attributes["adapter_reference_revision"],
            GEMMA4_E2B_SOURCE_REVISION
        );
    }

    #[test]
    fn lowers_pinned_official_outer_and_ignores_other_modalities() {
        let plan = plan();
        plan.validate().unwrap();
        assert_eq!(plan.model_family, "gemma4_text");
        assert_eq!(plan.adapter, "pllm.gemma4_e4b_text.v1");
        assert!(plan.prefill.state_inputs.is_empty());
        assert_eq!(plan.prefill.state_outputs.len(), 48);
        assert_eq!(plan.decode.state_inputs.len(), 48);
        assert_eq!(plan.decode.state_outputs.len(), 48);
        assert_eq!(
            operation(&plan.prefill, "main_embedding").attributes["weight"],
            "model.language_model.embed_tokens.weight"
        );
        assert_eq!(
            operation(&plan.prefill, "output_head").attributes["weight"],
            "model.language_model.embed_tokens.weight"
        );
        assert_eq!(
            operation(&plan.prefill, "main_embedding").attributes["adapter_reference_revision"],
            GEMMA4_E4B_SOURCE_REVISION
        );
    }

    #[test]
    fn direct_text_config_is_not_a_supported_checkpoint_layout() {
        let outer: Value = serde_json::from_str(OFFICIAL_OUTER).unwrap();
        let direct = serde_json::to_vec(&outer["text_config"]).unwrap();
        assert!(matches!(
            super::super::lower_model_json(&direct, workload()),
            Err(ModelError::Unsupported(_))
        ));
    }

    #[test]
    fn prelude_has_exact_ple_dimensions_order_and_symbolic_scales() {
        let plan = plan();
        let expected = [
            "main_embedding",
            "main_embedding_scaled",
            "ple_token_embedding",
            "ple_token_scaled",
            "ple_token_reshape",
            "ple_context_projection",
            "ple_context_scaled",
            "ple_context_reshape",
            "ple_context_norm",
            "ple_combined",
            "ple_combined_scaled",
        ];
        assert_eq!(
            plan.prefill
                .operations
                .iter()
                .take(expected.len())
                .map(|operation| operation.id.as_str())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            operation(&plan.prefill, "main_embedding").output_shape,
            [2, 1_024, 2_560]
        );
        assert_eq!(
            operation(&plan.prefill, "ple_token_embedding").output_shape,
            [2, 1_024, 10_752]
        );
        assert_eq!(
            operation(&plan.prefill, "ple_token_reshape").output_shape,
            [2, 1_024, 42, 256]
        );
        assert_eq!(
            operation(&plan.prefill, "main_embedding_scaled").attributes["factor"],
            json!({
                "kind":"sqrt",
                "radicand":2560,
                "factor_source_dtype":"float32_buffer",
                "factor_rounding_dtype":"bfloat16",
                "compute_dtype":"bfloat16",
                "output_dtype":"bfloat16"
            })
        );
        for (id, radicand) in [("ple_context_scaled", 2_560), ("ple_combined_scaled", 2)] {
            let factor = &operation(&plan.prefill, id).attributes["factor"];
            assert_eq!(factor["kind"], "inverse_sqrt");
            assert_eq!(factor["radicand"], radicand);
            assert_eq!(factor["factor_source_dtype"], "python_float64");
            assert_eq!(factor["factor_rounding_dtype"], "bfloat16");
            assert_eq!(factor["compute_dtype"], "bfloat16");
            assert_eq!(factor["output_dtype"], "bfloat16");
        }
    }

    #[test]
    fn all_layers_encode_width_rope_mask_norm_and_block_order() {
        let plan = plan();
        for layer in 0..E4B.num_layers {
            let sliding = layer_type(layer, E4B.full_attention_period) == "sliding_attention";
            let head_dim = if sliding { 256 } else { 512 };
            let prefix = format!("layer.{layer}");
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.q_heads")).output_shape,
                [2, 1_024, 8, head_dim]
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.q_permute")).output_shape,
                [2, 8, 1_024, head_dim]
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.q_permute")).attributes["permutation"],
                json!([0, 2, 1, 3])
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.attention_permute")).attributes
                    ["permutation"],
                json!([0, 2, 1, 3])
            );
            let rope = &operation(&plan.prefill, &format!("{prefix}.rope_q")).attributes;
            assert_eq!(
                rope["rope_type"],
                if sliding { "default" } else { "proportional" }
            );
            assert_eq!(rope["theta"], if sliding { 10_000 } else { 1_000_000 });
            assert_eq!(
                rope["partial_rotary_factor"],
                if sliding {
                    json!({"numerator":1,"denominator":1})
                } else {
                    json!({"numerator":1,"denominator":4})
                }
            );
            let factor = &operation(&plan.prefill, &format!("{prefix}.attention_scale")).attributes
                ["factor"];
            assert_eq!(factor["kind"], "rational");
            assert_eq!(factor["numerator"], 1);
            assert_eq!(factor["denominator"], 1);
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.causal_mask")).attributes["kind"],
                if sliding {
                    "sliding_causal"
                } else {
                    "full_causal"
                }
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.causal_mask")).inputs,
                [
                    format!("{prefix}.attention_scale"),
                    "input.positions".into(),
                    "input.attention_mask".into(),
                    "input.sequence_lengths".into()
                ]
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.softmax")).attributes["compute_dtype"],
                "float32"
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.post_attention_norm")).inputs,
                [format!("{prefix}.o_proj")]
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.attention_residual")).inputs,
                [
                    if layer == 0 {
                        "main_embedding_scaled".into()
                    } else {
                        format!("layer.{}.layer_scalar", layer - 1)
                    },
                    format!("{prefix}.post_attention_norm")
                ]
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.pre_feedforward_norm")).inputs,
                [format!("{prefix}.attention_residual")]
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.post_feedforward_norm")).inputs,
                [format!("{prefix}.down_proj")]
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.ple_slice")).attributes,
                json!({"axis":2,"start":layer,"end":layer+1,"squeeze":true})
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.layer_scalar")).operator,
                ModelOperator::Scale
            );
        }
        assert_eq!(
            plan.prefill
                .operations
                .iter()
                .filter(|operation| operation.operator == ModelOperator::GeluTanh)
                .count(),
            84
        );
    }

    #[test]
    fn physical_and_shared_kv_references_are_exact() {
        let plan = plan();
        let first_shared_layer = E4B.num_layers - E4B.shared_kv_layers;
        for layer in 0..first_shared_layer {
            let prefix = format!("layer.{layer}");
            assert!(plan.prefill.operations.iter().any(|operation| {
                operation.id == format!("{prefix}.k_linear")
                    && operation.operator == ModelOperator::Linear
            }));
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.q_norm")).attributes["with_scale"],
                true
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.k_norm")).attributes["with_scale"],
                true
            );
            let value_norm = operation(&plan.prefill, &format!("{prefix}.v_norm"));
            assert_eq!(value_norm.attributes["with_scale"], false);
            assert!(value_norm.attributes["weight"].is_null());
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.key_append")).inputs[0],
                format!("{prefix}.k_permute")
            );
            assert_eq!(
                operation(&plan.prefill, &format!("{prefix}.value_append")).inputs[0],
                format!("{prefix}.v_permute")
            );
            assert_eq!(
                operation(&plan.decode, &format!("{prefix}.key_append")).inputs[0],
                format!("state.layer.{layer}.key")
            );
        }
        for layer in first_shared_layer..E4B.num_layers {
            let source = if layer_type(layer, E4B.full_attention_period) == "sliding_attention" {
                22
            } else {
                23
            };
            assert!(!plan
                .prefill
                .operations
                .iter()
                .any(|operation| operation.id == format!("layer.{layer}.k_linear")));
            assert_eq!(
                operation(&plan.prefill, &format!("layer.{layer}.attention_scores")).inputs[1],
                format!("layer.{source}.key_append")
            );
            assert_eq!(
                operation(&plan.prefill, &format!("layer.{layer}.attention_values")).inputs[1],
                format!("layer.{source}.value_append")
            );
        }
    }

    #[test]
    fn sliding_attention_domain_and_persisted_suffix_are_distinct() {
        let plan = plan();
        let prefill_append = operation(&plan.prefill, "layer.0.key_append");
        let prefill_suffix = operation(&plan.prefill, "layer.0.key_suffix");
        assert_eq!(prefill_append.output_shape, [2, 2, 1_024, 512, 256]);
        assert_eq!(
            prefill_append.attributes["attention_domain"]["maximum_sequence"],
            512
        );
        assert_eq!(prefill_append.attributes["mode"], "initialize");
        assert_eq!(prefill_suffix.output_shape[2], 511);
        assert_eq!(
            prefill_suffix.inputs,
            [
                "layer.0.key_append",
                "input.positions",
                "input.attention_mask",
                "input.sequence_lengths"
            ]
        );
        assert_eq!(
            plan.prefill
                .state_outputs
                .iter()
                .find(|state| state.layer == Some(0) && state.kind == StateKind::Key)
                .unwrap()
                .id,
            "layer.0.key_suffix"
        );
        assert_eq!(
            operation(&plan.decode, "layer.0.key_append").output_shape,
            [2, 2, 1, 512, 256]
        );
        assert_eq!(
            operation(&plan.decode, "layer.0.key_suffix").output_shape[2],
            511
        );
        assert_eq!(
            operation(&plan.prefill, "layer.22.attention_scores").output_shape[3],
            512
        );
        assert_eq!(
            operation(&plan.prefill, "layer.23.attention_scores").output_shape[3],
            1_055
        );
        assert!(plan.prefill.state_inputs.is_empty());
        for output in &plan.prefill.state_outputs {
            let input = plan
                .decode
                .state_inputs
                .iter()
                .find(|input| input.layer == output.layer && input.kind == output.kind)
                .unwrap();
            assert_eq!(output.shape, input.shape);
            assert_eq!(output.maximum_sequence, input.maximum_sequence);
        }
    }

    #[test]
    fn output_is_norm_last_token_tied_head_softcap_greedy_feedback() {
        let plan = plan();
        let suffix = plan
            .prefill
            .operations
            .iter()
            .rev()
            .take(6)
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            suffix,
            [
                "token_feedback",
                "token_selection",
                "logit_softcap",
                "output_head",
                "last_hidden",
                "final_norm"
            ]
        );
        assert_eq!(
            operation(&plan.prefill, "logit_softcap").attributes["cap"],
            30
        );
        assert_eq!(
            operation(&plan.prefill, "output_head").attributes["tied"],
            true
        );
        assert_eq!(
            operation(&plan.prefill, "last_hidden").inputs,
            ["final_norm", "input.sequence_lengths"]
        );
        assert_eq!(
            operation(&plan.prefill, "token_selection").attributes["policy"],
            "pllm.greedy.v1"
        );
    }

    #[test]
    fn validation_rejects_permutation_cache_and_state_transition_drift() {
        let mut malformed = plan();
        operation_mut(&mut malformed.prefill, "layer.0.q_permute").attributes["permutation"] =
            json!([0, 1, 1, 3]);
        assert!(matches!(
            malformed.validate(),
            Err(ModelError::Incomplete(_))
        ));

        let mut malformed = plan();
        operation_mut(&mut malformed.decode, "layer.0.key_append").inputs[2] =
            "input.tokens".into();
        assert!(matches!(
            malformed.validate(),
            Err(ModelError::Incomplete(_))
        ));

        let mut malformed = plan();
        malformed.decode.state_inputs[0].shape[2] = 510;
        assert!(matches!(
            malformed.validate(),
            Err(ModelError::Incomplete(_))
        ));

        let mut malformed = plan();
        operation_mut(&mut malformed.prefill, "layer.0.key_suffix").output_shape[2] = 510;
        assert!(matches!(
            malformed.validate(),
            Err(ModelError::Incomplete(_))
        ));
    }

    #[test]
    fn rejects_overflow_and_unsupported_semantic_variants() {
        let outer: Value = serde_json::from_str(OFFICIAL_OUTER).unwrap();
        let mut variant = outer.clone();
        for (field, value) in [
            ("attention_bias", json!(true)),
            ("attention_k_eq_v", json!(true)),
            ("enable_moe_block", json!(true)),
            ("use_double_wide_mlp", json!(true)),
            ("use_bidirectional_attention", json!("all")),
            ("tie_word_embeddings", json!(false)),
            ("hidden_activation", json!("silu")),
            ("num_kv_shared_layers", json!(17)),
        ] {
            variant["text_config"][field] = value;
            assert!(matches!(
                super::super::lower_model_json(&serde_json::to_vec(&variant).unwrap(), workload()),
                Err(ModelError::Unsupported(_))
            ));
            variant = outer.clone();
        }
        for (field, value) in [
            ("architectures", json!(["Gemma4ForCausalLM"])),
            ("tie_word_embeddings", json!(false)),
            ("dtype", json!("float32")),
        ] {
            variant[field] = value;
            assert!(matches!(
                super::super::lower_model_json(&serde_json::to_vec(&variant).unwrap(), workload()),
                Err(ModelError::Unsupported(_))
            ));
            variant = outer.clone();
        }
        variant["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("head_dim");
        assert!(matches!(
            super::super::lower_model_json(&serde_json::to_vec(&variant).unwrap(), workload()),
            Err(ModelError::InvalidConfig(_))
        ));
        assert!(matches!(
            super::super::lower_model_json(
                OFFICIAL_OUTER.as_bytes(),
                DecoderWorkload {
                    batch: 1,
                    max_input_tokens: u64::MAX,
                    max_new_tokens: 2,
                }
            ),
            Err(ModelError::InvalidConfig(_))
        ));
        super::super::lower_model_json(
            OFFICIAL_OUTER.as_bytes(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: MAX_POSITIONS,
                max_new_tokens: 1,
            },
        )
        .unwrap();
        assert!(matches!(
            super::super::lower_model_json(
                OFFICIAL_OUTER.as_bytes(),
                DecoderWorkload {
                    batch: 1,
                    max_input_tokens: MAX_POSITIONS,
                    max_new_tokens: 2,
                }
            ),
            Err(ModelError::InvalidConfig(_))
        ));
    }
}
