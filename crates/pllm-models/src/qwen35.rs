use super::{
    DecoderGraph, DecoderMode, DecoderPlan, DecoderWorkload, ModelError, ModelOperation,
    ModelOperator, StateKind, StateTensor, DECODER_PLAN_SCHEMA_VERSION,
};
use pllm_types::{canonical_digest, Digest};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};

const ADAPTER: &str = "pllm.qwen3_5_text.v1";
const FAMILY: &str = "qwen3_5_text";

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OuterConfig {
    architectures: Vec<String>,
    model_type: String,
    text_config: TextConfig,
    tie_word_embeddings: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TextConfig {
    attention_bias: bool,
    #[serde(deserialize_with = "decimal_string")]
    attention_dropout: String,
    attn_output_gate: bool,
    full_attention_interval: u32,
    head_dim: u64,
    hidden_act: String,
    hidden_size: u64,
    intermediate_size: u64,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: u64,
    linear_key_head_dim: u64,
    linear_num_key_heads: u64,
    linear_num_value_heads: u64,
    linear_value_head_dim: u64,
    max_position_embeddings: u64,
    mlp_only_layers: Vec<u32>,
    model_type: String,
    mtp_num_hidden_layers: u32,
    mtp_use_dedicated_embeddings: bool,
    num_attention_heads: u64,
    num_hidden_layers: u32,
    num_key_value_heads: u64,
    #[serde(deserialize_with = "decimal_string")]
    rms_norm_eps: String,
    tie_word_embeddings: bool,
    use_cache: bool,
    vocab_size: u64,
    mamba_ssm_dtype: String,
    rope_parameters: RopeParameters,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RopeParameters {
    mrope_interleaved: bool,
    mrope_section: Vec<u64>,
    rope_type: String,
    #[serde(deserialize_with = "integral_u64")]
    rope_theta: u64,
    #[serde(deserialize_with = "decimal_string")]
    partial_rotary_factor: String,
}

pub(super) fn lower_qwen35_json(
    document: &Value,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    let config: OuterConfig = serde_json::from_value(document.clone()).map_err(|error| {
        ModelError::InvalidConfig(format!(
            "Qwen3.5 configuration does not match the adapter: {error}"
        ))
    })?;
    config.validate()?;
    validate_workload(workload, config.text_config.max_position_embeddings)?;
    let digest = canonical_digest("pllm.qwen3_5_outer_config.v1", &config);
    let prefill = lower_graph(&config.text_config, workload, DecoderMode::Prefill)?;
    let decode = lower_graph(&config.text_config, workload, DecoderMode::Decode)?;
    let plan = DecoderPlan {
        schema_version: DECODER_PLAN_SCHEMA_VERSION.into(),
        model_family: FAMILY.into(),
        adapter: ADAPTER.into(),
        config_digest: digest,
        prefill,
        decode,
        token_feedback: true,
        transformations: Vec::new(),
    };
    plan.validate()?;
    Ok(plan)
}

impl OuterConfig {
    fn validate(&self) -> Result<(), ModelError> {
        exact_str(&self.model_type, "qwen3_5", "model_type")?;
        if self.architectures.as_slice() != ["Qwen3_5ForConditionalGeneration"] {
            return Err(unsupported("architectures"));
        }
        if !self.tie_word_embeddings || !self.text_config.tie_word_embeddings {
            return Err(unsupported("tie_word_embeddings"));
        }
        self.text_config.validate()
    }
}

impl TextConfig {
    fn validate(&self) -> Result<(), ModelError> {
        exact_str(&self.model_type, "qwen3_5_text", "text_config.model_type")?;
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
        {
            return Err(unsupported("positive text decoder dimensions"));
        }
        if self.hidden_act != "silu"
            || self.rms_norm_eps != "0.000001"
            || self.attention_dropout != "0"
            || self.attention_bias
            || !self.attn_output_gate
            || !self.use_cache
            || self.mamba_ssm_dtype != "float32"
        {
            return Err(unsupported(
                "normalization, activation, or attention controls",
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0
            || self.linear_num_value_heads % self.linear_num_key_heads != 0
            || self.linear_key_head_dim == 0
            || self.linear_value_head_dim == 0
            || self.linear_conv_kernel_dim == 0
        {
            return Err(unsupported("attention head and convolution dimensions"));
        }
        if self.full_attention_interval == 0
            || self.layer_types.len() != self.num_hidden_layers as usize
            || self.layer_types.iter().enumerate().any(|(layer, kind)| {
                let expected = if (layer + 1) % self.full_attention_interval as usize == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                };
                kind != expected
            })
        {
            return Err(unsupported("hybrid layer schedule"));
        }
        if !self.mlp_only_layers.is_empty()
            || self.mtp_num_hidden_layers != 1
            || self.mtp_use_dedicated_embeddings
        {
            return Err(unsupported("MLP-only or MTP controls"));
        }
        let rope = &self.rope_parameters;
        let rotary_dim = exact_fraction(self.head_dim, &rope.partial_rotary_factor)?;
        if rope.rope_type != "default"
            || !rope.mrope_interleaved
            || rope.rope_theta == 0
            || rotary_dim == 0
            || rotary_dim % 2 != 0
            || rope.mrope_section.iter().sum::<u64>() * 2 != rotary_dim
        {
            return Err(unsupported("multimodal rotary parameters"));
        }
        Ok(())
    }

    fn rotary_dim(&self) -> Result<u64, ModelError> {
        exact_fraction(self.head_dim, &self.rope_parameters.partial_rotary_factor)
    }
}

fn lower_graph(
    config: &TextConfig,
    workload: DecoderWorkload,
    mode: DecoderMode,
) -> Result<DecoderGraph, ModelError> {
    let batch = workload.batch;
    let query = match mode {
        DecoderMode::Prefill => workload.max_input_tokens,
        DecoderMode::Decode => 1,
    };
    let maximum_key_sequence = workload
        .max_input_tokens
        .checked_add(workload.max_new_tokens)
        .ok_or_else(|| ModelError::Unsupported("workload sequence bound overflowed".into()))?;
    let hidden_shape = vec![batch, query, config.hidden_size];
    let mut operations = Vec::new();
    push(
        &mut operations,
        "token_lookup",
        ModelOperator::TokenLookup,
        &["input.tokens"],
        hidden_shape.clone(),
        json!({"weight": "model.embed_tokens.weight"}),
    );
    let mut hidden = "token_lookup".to_owned();
    let mut state_inputs = Vec::new();
    let mut state_outputs = Vec::new();
    for (layer, layer_type) in config.layer_types.iter().enumerate() {
        let layer = u64::try_from(layer)
            .map_err(|_| ModelError::Unsupported("layer index overflowed".into()))?;
        let prefix = format!("layer.{layer}");
        let residual = hidden.clone();
        let input_norm = format!("{prefix}.input_norm");
        rms_norm(
            &mut operations,
            &input_norm,
            &hidden,
            hidden_shape.clone(),
            config,
            format!("model.layers.{layer}.input_layernorm.weight"),
        );
        let mixer = if layer_type == "full_attention" {
            lower_full_attention(
                &mut operations,
                &mut state_inputs,
                &mut state_outputs,
                config,
                mode,
                layer,
                &input_norm,
                batch,
                query,
                maximum_key_sequence,
            )?
        } else {
            lower_linear_attention(
                &mut operations,
                &mut state_inputs,
                &mut state_outputs,
                config,
                mode,
                layer,
                &input_norm,
                batch,
                query,
            )?
        };
        let attention_residual = format!("{prefix}.attention_residual");
        push(
            &mut operations,
            &attention_residual,
            ModelOperator::ResidualAdd,
            &[&residual, &mixer],
            hidden_shape.clone(),
            json!({}),
        );
        let post_norm = format!("{prefix}.post_attention_norm");
        rms_norm(
            &mut operations,
            &post_norm,
            &attention_residual,
            hidden_shape.clone(),
            config,
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
        );
        let gate = format!("{prefix}.gate");
        let up = format!("{prefix}.up");
        let activated = format!("{prefix}.silu");
        let gated = format!("{prefix}.gated");
        let down = format!("{prefix}.down");
        linear(
            &mut operations,
            &gate,
            &post_norm,
            vec![batch, query, config.intermediate_size],
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
        );
        linear(
            &mut operations,
            &up,
            &post_norm,
            vec![batch, query, config.intermediate_size],
            format!("model.layers.{layer}.mlp.up_proj.weight"),
        );
        push(
            &mut operations,
            &activated,
            ModelOperator::Silu,
            &[&gate],
            vec![batch, query, config.intermediate_size],
            json!({}),
        );
        push(
            &mut operations,
            &gated,
            ModelOperator::Multiply,
            &[&activated, &up],
            vec![batch, query, config.intermediate_size],
            json!({}),
        );
        linear(
            &mut operations,
            &down,
            &gated,
            hidden_shape.clone(),
            format!("model.layers.{layer}.mlp.down_proj.weight"),
        );
        let output = format!("{prefix}.output");
        push(
            &mut operations,
            &output,
            ModelOperator::ResidualAdd,
            &[&attention_residual, &down],
            hidden_shape.clone(),
            json!({}),
        );
        hidden = output;
    }
    rms_norm(
        &mut operations,
        "final_norm",
        &hidden,
        hidden_shape.clone(),
        config,
        "model.norm.weight".into(),
    );
    let selected = if mode == DecoderMode::Prefill {
        push(
            &mut operations,
            "last_token",
            ModelOperator::LastToken,
            &["final_norm", "input.sequence_lengths"],
            vec![batch, config.hidden_size],
            json!({"axis": 1, "selection": "last_valid", "valid_lengths_input": "input.sequence_lengths"}),
        );
        "last_token"
    } else {
        "final_norm"
    };
    push(
        &mut operations,
        "output_head",
        ModelOperator::OutputHead,
        &[selected],
        if mode == DecoderMode::Prefill {
            vec![batch, config.vocab_size]
        } else {
            vec![batch, 1, config.vocab_size]
        },
        json!({"weight": "model.embed_tokens.weight", "tied": true}),
    );
    push(
        &mut operations,
        "greedy_token",
        ModelOperator::GreedyTokenSelection,
        &["output_head"],
        if mode == DecoderMode::Prefill {
            vec![batch]
        } else {
            vec![batch, 1]
        },
        json!({"axis": -1}),
    );
    push(
        &mut operations,
        "token_feedback",
        ModelOperator::TokenFeedback,
        &["greedy_token"],
        vec![batch, 1],
        json!({"target": "input.tokens"}),
    );
    Ok(DecoderGraph {
        mode,
        batch,
        query_sequence: query,
        maximum_key_sequence,
        operations,
        state_inputs,
        state_outputs,
        output: "token_feedback".into(),
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_linear_attention(
    operations: &mut Vec<ModelOperation>,
    state_inputs: &mut Vec<StateTensor>,
    state_outputs: &mut Vec<StateTensor>,
    config: &TextConfig,
    mode: DecoderMode,
    layer: u64,
    input: &str,
    batch: u64,
    query_sequence: u64,
) -> Result<String, ModelError> {
    let prefix = format!("layer.{layer}.linear_attention");
    let key_width = config
        .linear_num_key_heads
        .checked_mul(config.linear_key_head_dim)
        .ok_or_else(|| ModelError::Unsupported("linear attention key width overflowed".into()))?;
    let value_width = config
        .linear_num_value_heads
        .checked_mul(config.linear_value_head_dim)
        .ok_or_else(|| ModelError::Unsupported("linear attention value width overflowed".into()))?;
    let conv_width = key_width
        .checked_mul(2)
        .and_then(|value| value.checked_add(value_width))
        .ok_or_else(|| {
            ModelError::Unsupported("linear attention convolution width overflowed".into())
        })?;
    let qkv = format!("{prefix}.qkv_projection");
    let z = format!("{prefix}.z_projection");
    let b = format!("{prefix}.b_projection");
    let a = format!("{prefix}.a_projection");
    linear(
        operations,
        &qkv,
        input,
        vec![batch, query_sequence, conv_width],
        format!("model.layers.{layer}.linear_attn.in_proj_qkv.weight"),
    );
    linear(
        operations,
        &z,
        input,
        vec![batch, query_sequence, value_width],
        format!("model.layers.{layer}.linear_attn.in_proj_z.weight"),
    );
    linear(
        operations,
        &b,
        input,
        vec![batch, query_sequence, config.linear_num_value_heads],
        format!("model.layers.{layer}.linear_attn.in_proj_b.weight"),
    );
    linear(
        operations,
        &a,
        input,
        vec![batch, query_sequence, config.linear_num_value_heads],
        format!("model.layers.{layer}.linear_attn.in_proj_a.weight"),
    );
    let conv_state = format!("state.layer.{layer}.convolution");
    let recurrent_state = format!("state.layer.{layer}.recurrent");
    let conv_state_shape = vec![batch, conv_width, config.linear_conv_kernel_dim];
    let recurrent_state_shape = vec![
        batch,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    ];
    state_inputs.push(StateTensor {
        id: conv_state.clone(),
        layer: Some(layer),
        kind: StateKind::Convolution,
        shape: conv_state_shape.clone(),
        maximum_sequence: config.linear_conv_kernel_dim,
    });
    state_inputs.push(StateTensor {
        id: recurrent_state.clone(),
        layer: Some(layer),
        kind: StateKind::Recurrent,
        shape: recurrent_state_shape.clone(),
        maximum_sequence: 1,
    });
    let convolved = format!("{prefix}.causal_convolution");
    let next_conv_state = format!("{prefix}.convolution_state");
    let convolution_attributes = json!({
        "weight": format!("model.layers.{layer}.linear_attn.conv1d.weight"),
        "bias": Value::Null,
        "kernel_size": config.linear_conv_kernel_dim,
        "groups": conv_width,
        "activation": "silu",
        "mode": match mode { DecoderMode::Prefill => "chunk", DecoderMode::Decode => "recurrent" },
    });
    push(
        operations,
        &next_conv_state,
        ModelOperator::ConvolutionStateUpdate,
        &[&qkv, &conv_state],
        conv_state_shape.clone(),
        convolution_attributes.clone(),
    );
    operations
        .last_mut()
        .expect("state operation exists")
        .state_kind = Some(StateKind::Convolution);
    push(
        operations,
        &convolved,
        ModelOperator::CausalConvolution,
        &[&qkv, &conv_state],
        vec![batch, query_sequence, conv_width],
        convolution_attributes,
    );
    state_outputs.push(StateTensor {
        id: next_conv_state,
        layer: Some(layer),
        kind: StateKind::Convolution,
        shape: conv_state_shape,
        maximum_sequence: config.linear_conv_kernel_dim,
    });
    let query = slice(
        operations,
        &format!("{prefix}.query"),
        &convolved,
        vec![batch, query_sequence, key_width],
        0,
        key_width,
    );
    let key = slice(
        operations,
        &format!("{prefix}.key"),
        &convolved,
        vec![batch, query_sequence, key_width],
        key_width,
        key_width * 2,
    );
    let value = slice(
        operations,
        &format!("{prefix}.value"),
        &convolved,
        vec![batch, query_sequence, value_width],
        key_width * 2,
        conv_width,
    );
    let beta = format!("{prefix}.beta");
    push(
        operations,
        &beta,
        ModelOperator::Sigmoid,
        &[&b],
        vec![batch, query_sequence, config.linear_num_value_heads],
        json!({}),
    );
    let decay = format!("{prefix}.decay");
    push(
        operations,
        &decay,
        ModelOperator::GatedDeltaDecay,
        &[&a],
        vec![batch, query_sequence, config.linear_num_value_heads],
        json!({
            "a_log": format!("model.layers.{layer}.linear_attn.A_log"),
            "dt_bias": format!("model.layers.{layer}.linear_attn.dt_bias"),
            "formula": "-exp(A_log)*softplus(a+dt_bias)",
            "compute_dtype": "float32",
        }),
    );
    let core = format!("{prefix}.gated_delta_rule");
    let next_recurrent = format!("{prefix}.recurrent_state");
    let delta_attributes = json!({
        "qk_l2_normalize": true,
        "normalization_epsilon": "0.000001",
        "query_scale": {"numerator": 1, "sqrt_denominator": config.linear_key_head_dim},
        "key_head_repeats": config.linear_num_value_heads / config.linear_num_key_heads,
        "mode": match mode { DecoderMode::Prefill => "chunked", DecoderMode::Decode => "recurrent" },
        "chunk_size": 64,
        "state_update": "S'=exp(g)*S+k*(v-S^T*k)*beta",
    });
    let delta_inputs = [
        &query[..],
        &key[..],
        &value[..],
        &decay[..],
        &beta[..],
        &recurrent_state[..],
    ];
    push(
        operations,
        &next_recurrent,
        ModelOperator::GatedDeltaStateUpdate,
        &delta_inputs,
        recurrent_state_shape.clone(),
        delta_attributes.clone(),
    );
    operations
        .last_mut()
        .expect("state operation exists")
        .state_kind = Some(StateKind::Recurrent);
    push(
        operations,
        &core,
        ModelOperator::GatedDeltaRule,
        &delta_inputs,
        vec![
            batch,
            query_sequence,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        ],
        delta_attributes,
    );
    state_outputs.push(StateTensor {
        id: next_recurrent,
        layer: Some(layer),
        kind: StateKind::Recurrent,
        shape: recurrent_state_shape,
        maximum_sequence: 1,
    });
    let z_heads = format!("{prefix}.z_heads");
    push(
        operations,
        &z_heads,
        ModelOperator::Reshape,
        &[&z],
        vec![
            batch,
            query_sequence,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        ],
        json!({}),
    );
    let gated_norm = format!("{prefix}.gated_norm");
    push(
        operations,
        &gated_norm,
        ModelOperator::RmsNormGated,
        &[&core, &z_heads],
        vec![
            batch,
            query_sequence,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        ],
        json!({
            "epsilon": config.rms_norm_eps,
            "weight": format!("model.layers.{layer}.linear_attn.norm.weight"),
            "activation": "silu",
            "weight_offset": 0,
            "norm_before_gate": true,
        }),
    );
    let merged = format!("{prefix}.merged");
    push(
        operations,
        &merged,
        ModelOperator::Reshape,
        &[&gated_norm],
        vec![batch, query_sequence, value_width],
        json!({}),
    );
    let output = format!("{prefix}.output");
    linear(
        operations,
        &output,
        &merged,
        vec![batch, query_sequence, config.hidden_size],
        format!("model.layers.{layer}.linear_attn.out_proj.weight"),
    );
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn lower_full_attention(
    operations: &mut Vec<ModelOperation>,
    state_inputs: &mut Vec<StateTensor>,
    state_outputs: &mut Vec<StateTensor>,
    config: &TextConfig,
    _mode: DecoderMode,
    layer: u64,
    input: &str,
    batch: u64,
    query_sequence: u64,
    maximum_key_sequence: u64,
) -> Result<String, ModelError> {
    let prefix = format!("layer.{layer}.self_attention");
    let q_width = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| ModelError::Unsupported("query width overflowed".into()))?;
    let kv_width = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| ModelError::Unsupported("KV width overflowed".into()))?;
    let q_projection = format!("{prefix}.q_projection");
    let k_projection = format!("{prefix}.k_projection");
    let v_projection = format!("{prefix}.v_projection");
    linear(
        operations,
        &q_projection,
        input,
        vec![batch, query_sequence, q_width * 2],
        format!("model.layers.{layer}.self_attn.q_proj.weight"),
    );
    linear(
        operations,
        &k_projection,
        input,
        vec![batch, query_sequence, kv_width],
        format!("model.layers.{layer}.self_attn.k_proj.weight"),
    );
    linear(
        operations,
        &v_projection,
        input,
        vec![batch, query_sequence, kv_width],
        format!("model.layers.{layer}.self_attn.v_proj.weight"),
    );
    let q_flat = slice(
        operations,
        &format!("{prefix}.query_flat"),
        &q_projection,
        vec![batch, query_sequence, q_width],
        0,
        q_width,
    );
    let gate = slice(
        operations,
        &format!("{prefix}.output_gate"),
        &q_projection,
        vec![batch, query_sequence, q_width],
        q_width,
        q_width * 2,
    );
    let q = format!("{prefix}.query_heads");
    let k = format!("{prefix}.key_heads");
    let v = format!("{prefix}.value_heads");
    push(
        operations,
        &q,
        ModelOperator::Reshape,
        &[&q_flat],
        vec![
            batch,
            config.num_attention_heads,
            query_sequence,
            config.head_dim,
        ],
        json!({}),
    );
    push(
        operations,
        &k,
        ModelOperator::Reshape,
        &[&k_projection],
        vec![
            batch,
            config.num_key_value_heads,
            query_sequence,
            config.head_dim,
        ],
        json!({}),
    );
    push(
        operations,
        &v,
        ModelOperator::Reshape,
        &[&v_projection],
        vec![
            batch,
            config.num_key_value_heads,
            query_sequence,
            config.head_dim,
        ],
        json!({}),
    );
    let q_norm = format!("{prefix}.q_norm");
    let k_norm = format!("{prefix}.k_norm");
    rms_norm(
        operations,
        &q_norm,
        &q,
        vec![
            batch,
            config.num_attention_heads,
            query_sequence,
            config.head_dim,
        ],
        config,
        format!("model.layers.{layer}.self_attn.q_norm.weight"),
    );
    rms_norm(
        operations,
        &k_norm,
        &k,
        vec![
            batch,
            config.num_key_value_heads,
            query_sequence,
            config.head_dim,
        ],
        config,
        format!("model.layers.{layer}.self_attn.k_norm.weight"),
    );
    let q_rope = format!("{prefix}.q_rope");
    let k_rope = format!("{prefix}.k_rope");
    let rope_attrs = json!({
        "theta": config.rope_parameters.rope_theta,
        "rope_type": "default",
        "rotary_dimensions": config.rotary_dim()?,
        "partial_rotary_factor": config.rope_parameters.partial_rotary_factor,
        "mrope_interleaved": config.rope_parameters.mrope_interleaved,
        "mrope_section": config.rope_parameters.mrope_section,
    });
    push(
        operations,
        &q_rope,
        ModelOperator::RotaryEmbedding,
        &[&q_norm, "input.positions"],
        vec![
            batch,
            config.num_attention_heads,
            query_sequence,
            config.head_dim,
        ],
        rope_attrs.clone(),
    );
    push(
        operations,
        &k_rope,
        ModelOperator::RotaryEmbedding,
        &[&k_norm, "input.positions"],
        vec![
            batch,
            config.num_key_value_heads,
            query_sequence,
            config.head_dim,
        ],
        rope_attrs,
    );
    let key_state = format!("state.layer.{layer}.key");
    let value_state = format!("state.layer.{layer}.value");
    let kv_shape = vec![
        batch,
        config.num_key_value_heads,
        maximum_key_sequence,
        config.head_dim,
    ];
    for (id, kind) in [
        (&key_state, StateKind::Key),
        (&value_state, StateKind::Value),
    ] {
        state_inputs.push(StateTensor {
            id: id.clone(),
            layer: Some(layer),
            kind,
            shape: kv_shape.clone(),
            maximum_sequence: maximum_key_sequence,
        });
    }
    let key_append = format!("{prefix}.key_append");
    let value_append = format!("{prefix}.value_append");
    push(
        operations,
        &key_append,
        ModelOperator::KvCacheAppend,
        &[&key_state, &k_rope],
        kv_shape.clone(),
        json!({"state": key_state}),
    );
    operations
        .last_mut()
        .expect("state operation exists")
        .state_kind = Some(StateKind::Key);
    push(
        operations,
        &value_append,
        ModelOperator::KvCacheAppend,
        &[&value_state, &v],
        kv_shape.clone(),
        json!({"state": value_state}),
    );
    operations
        .last_mut()
        .expect("state operation exists")
        .state_kind = Some(StateKind::Value);
    state_outputs.push(StateTensor {
        id: key_append.clone(),
        layer: Some(layer),
        kind: StateKind::Key,
        shape: kv_shape.clone(),
        maximum_sequence: maximum_key_sequence,
    });
    state_outputs.push(StateTensor {
        id: value_append.clone(),
        layer: Some(layer),
        kind: StateKind::Value,
        shape: kv_shape,
        maximum_sequence: maximum_key_sequence,
    });
    let scores = format!("{prefix}.scores");
    let scaled = format!("{prefix}.scaled");
    let masked = format!("{prefix}.masked");
    let probabilities = format!("{prefix}.probabilities");
    let values = format!("{prefix}.values");
    let group_size = config.num_attention_heads / config.num_key_value_heads;
    push(
        operations,
        &scores,
        ModelOperator::AttentionScores,
        &[&q_rope, &key_append],
        vec![
            batch,
            config.num_attention_heads,
            query_sequence,
            maximum_key_sequence,
        ],
        json!({"group_size": group_size}),
    );
    push(
        operations,
        &scaled,
        ModelOperator::AttentionScale,
        &[&scores],
        vec![
            batch,
            config.num_attention_heads,
            query_sequence,
            maximum_key_sequence,
        ],
        json!({"head_dim": config.head_dim}),
    );
    push(
        operations,
        &masked,
        ModelOperator::CausalMask,
        &[&scaled, "input.positions"],
        vec![
            batch,
            config.num_attention_heads,
            query_sequence,
            maximum_key_sequence,
        ],
        json!({"maximum_key_sequence": maximum_key_sequence}),
    );
    push(
        operations,
        &probabilities,
        ModelOperator::Softmax,
        &[&masked],
        vec![
            batch,
            config.num_attention_heads,
            query_sequence,
            maximum_key_sequence,
        ],
        json!({"axis": -1, "compute_dtype": "float32"}),
    );
    push(
        operations,
        &values,
        ModelOperator::AttentionValues,
        &[&probabilities, &value_append],
        vec![
            batch,
            config.num_attention_heads,
            query_sequence,
            config.head_dim,
        ],
        json!({"group_size": group_size}),
    );
    let merged = format!("{prefix}.merged");
    push(
        operations,
        &merged,
        ModelOperator::Reshape,
        &[&values],
        vec![batch, query_sequence, q_width],
        json!({}),
    );
    let sigmoid_gate = format!("{prefix}.sigmoid_gate");
    push(
        operations,
        &sigmoid_gate,
        ModelOperator::Sigmoid,
        &[&gate],
        vec![batch, query_sequence, q_width],
        json!({}),
    );
    let gated = format!("{prefix}.gated");
    push(
        operations,
        &gated,
        ModelOperator::Multiply,
        &[&merged, &sigmoid_gate],
        vec![batch, query_sequence, q_width],
        json!({}),
    );
    let output = format!("{prefix}.output");
    linear(
        operations,
        &output,
        &gated,
        vec![batch, query_sequence, config.hidden_size],
        format!("model.layers.{layer}.self_attn.o_proj.weight"),
    );
    Ok(output)
}

fn slice(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    shape: Vec<u64>,
    start: u64,
    end: u64,
) -> String {
    push(
        operations,
        id,
        ModelOperator::Slice,
        &[input],
        shape,
        json!({"axis": -1, "start": start, "end": end, "squeeze": false}),
    );
    id.into()
}

fn rms_norm(
    operations: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    shape: Vec<u64>,
    config: &TextConfig,
    weight: String,
) {
    push(
        operations,
        id,
        ModelOperator::RmsNorm,
        &[input],
        shape,
        json!({"epsilon": config.rms_norm_eps, "weight": weight, "weight_offset": 1}),
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
        json!({"weight": weight, "bias": Value::Null}),
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

fn validate_workload(workload: DecoderWorkload, maximum: u64) -> Result<(), ModelError> {
    let total = workload
        .max_input_tokens
        .checked_add(workload.max_new_tokens)
        .ok_or_else(|| ModelError::Unsupported("workload sequence bound overflowed".into()))?;
    if workload.batch == 0
        || workload.max_input_tokens == 0
        || workload.max_new_tokens == 0
        || total > maximum
    {
        return Err(ModelError::Unsupported(
            "workload must be positive and fit max_position_embeddings".into(),
        ));
    }
    Ok(())
}

fn exact_str(actual: &str, expected: &str, field: &str) -> Result<(), ModelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(unsupported(field))
    }
}

fn unsupported(field: &str) -> ModelError {
    ModelError::Unsupported(format!("Qwen3.5 adapter does not support {field}"))
}

fn exact_fraction(value: u64, decimal: &str) -> Result<u64, ModelError> {
    let (numerator, denominator) = match decimal {
        "0.25" => (1, 4),
        "1" => (1, 1),
        _ => return Err(unsupported("partial_rotary_factor")),
    };
    value
        .checked_mul(numerator)
        .filter(|scaled| scaled % denominator == 0)
        .map(|scaled| scaled / denominator)
        .ok_or_else(|| unsupported("partial_rotary_factor"))
}

fn decimal_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Number(number) => {
            let parsed = number
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| serde::de::Error::custom("expected a finite decimal"))?;
            Ok(if parsed == 0.0 {
                "0".into()
            } else {
                let text = format!("{parsed:.15}");
                text.trim_end_matches('0').trim_end_matches('.').into()
            })
        }
        other => Err(serde::de::Error::custom(format!(
            "expected a decimal, got {other}"
        ))),
    }
}

fn integral_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Number(number) => number
            .as_u64()
            .or_else(|| {
                number
                    .as_f64()
                    .filter(|value| value.is_finite() && value.fract() == 0.0 && *value >= 0.0)
                    .map(|value| value as u64)
            })
            .ok_or_else(|| serde::de::Error::custom("expected a non-negative integer")),
        other => Err(serde::de::Error::custom(format!(
            "expected an integer, got {other}"
        ))),
    }
}

#[allow(dead_code)]
fn _digest_type_is_stable(_: Digest) {}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest as _, Sha256};

    const OFFICIAL_CONFIG: &str = include_str!("../tests/fixtures/Qwen3.5-4B-851bf6e-config.json");

    fn workload() -> DecoderWorkload {
        DecoderWorkload {
            batch: 2,
            max_input_tokens: 128,
            max_new_tokens: 32,
        }
    }

    #[test]
    fn lowers_hybrid_qwen35_text_decoder() {
        let document: Value = serde_json::from_str(OFFICIAL_CONFIG).unwrap();
        let plan = lower_qwen35_json(&document, workload()).unwrap();
        assert_eq!(plan.adapter, ADAPTER);
        assert_eq!(plan.model_family, FAMILY);
        assert_eq!(plan.prefill.state_inputs.len(), 64);
        assert_eq!(plan.prefill.state_outputs.len(), 64);

        let first_linear = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.id == "layer.0.linear_attention.qkv_projection")
            .unwrap();
        assert_eq!(first_linear.output_shape, vec![2, 128, 8192]);
        assert!(plan.prefill.operations.iter().any(|op| {
            op.id == "layer.0.linear_attention.gated_delta_rule"
                && op.operator == ModelOperator::GatedDeltaRule
        }));
        assert!(plan.prefill.operations.iter().any(|op| {
            op.id == "layer.0.linear_attention.convolution_state"
                && op.state_kind == Some(StateKind::Convolution)
        }));

        let full_q = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.id == "layer.3.self_attention.q_projection")
            .unwrap();
        assert_eq!(full_q.output_shape, vec![2, 128, 8192]);
        let rope = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.id == "layer.3.self_attention.q_rope")
            .unwrap();
        assert_eq!(rope.attributes["rotary_dimensions"], 64);
        assert_eq!(rope.attributes["mrope_section"], json!([11, 11, 10]));
        assert!(plan.prefill.operations.iter().any(|op| {
            op.id == "layer.3.self_attention.sigmoid_gate" && op.operator == ModelOperator::Sigmoid
        }));
        assert_eq!(
            format!("{:x}", Sha256::digest(OFFICIAL_CONFIG)),
            "4d02c8384ab8bd5f38244ee3fb5c227e38ac23ba466faa369bd8080af0f9df45"
        );
        let upstream = OFFICIAL_CONFIG
            .strip_suffix('\n')
            .expect("repository fixture adds one trailing newline");
        assert_eq!(
            format!("{:x}", Sha256::digest(upstream)),
            "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
        );
    }

    #[test]
    fn rejects_qwen35_schedule_drift() {
        let mut document: Value = serde_json::from_str(OFFICIAL_CONFIG).unwrap();
        document["text_config"]["layer_types"][0] = json!("full_attention");
        let error = lower_qwen35_json(&document, workload()).unwrap_err();
        assert!(error.to_string().contains("hybrid layer schedule"));
    }
}
