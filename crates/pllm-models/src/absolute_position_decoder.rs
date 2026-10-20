//! Inference-only FP32 semantics: learned positions, affine centered norms,
//! input/output-oriented fused projections, and an ungated tanh GELU MLP.
//! Numeric protocol lowering and disclosure/state execution policy are separate.
use super::source_mappings::{ArtifactLayout, FusedDenseSource, ABSOLUTE_POSITION};
use super::*;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct Config {
    model_type: String,
    n_embd: u64,
    n_head: u64,
    n_layer: u32,
    n_positions: u64,
    vocab_size: u64,
    #[serde(default)]
    n_inner: Option<u64>,
    #[serde(default = "activation")]
    activation_function: String,
    #[serde(default = "epsilon", deserialize_with = "decimal_string")]
    layer_norm_epsilon: String,
    #[serde(default = "yes")]
    tie_word_embeddings: bool,
    #[serde(default = "yes")]
    use_cache: bool,
    #[serde(default = "yes")]
    scale_attn_weights: bool,
    #[serde(default)]
    scale_attn_by_inverse_layer_idx: bool,
    #[serde(default)]
    reorder_and_upcast_attn: bool,
    #[serde(default)]
    add_cross_attention: bool,
    #[serde(default)]
    is_encoder_decoder: bool,
    #[serde(default)]
    pruned_heads: BTreeMap<String, Value>,
}
fn activation() -> String {
    "gelu_new".into()
}
fn epsilon() -> String {
    "0.00001".into()
}
fn yes() -> bool {
    true
}
fn invalid(message: &str) -> ModelError {
    ModelError::InvalidConfig(message.into())
}
fn unsupported(message: &str) -> ModelError {
    ModelError::Unsupported(message.into())
}

impl Config {
    fn inner(&self) -> u64 {
        self.n_inner.unwrap_or(4 * self.n_embd)
    }
    fn validate(&self, source: &FusedDenseSource<'_>) -> Result<(), ModelError> {
        if self.model_type != source.model_type
            || self.activation_function != "gelu_new"
            || !self.use_cache
            || !self.scale_attn_weights
            || self.scale_attn_by_inverse_layer_idx
            || self.reorder_and_upcast_attn
            || self.add_cross_attention
            || self.is_encoder_decoder
            || !self.pruned_heads.is_empty()
        {
            return Err(unsupported("source requires standard full causal attention, gelu_new, and no cross attention, pruning or altered scaling"));
        }
        // Explicit finite source envelope, independent of caller-selected admission.
        if !(1..=128).contains(&self.n_layer)
            || !(1..=16384).contains(&self.n_embd)
            || !(1..=256).contains(&self.n_head)
            || self.n_embd % self.n_head != 0
            || !(1..=1024).contains(&self.n_positions)
            || !(1..=262144).contains(&self.vocab_size)
            || self
                .n_inner
                .is_some_and(|width| width == 0 || width > 65536)
        {
            return Err(invalid("absolute-position source dimensions exceed the supported envelope or heads do not divide hidden width"));
        }
        let eps = self
            .layer_norm_epsilon
            .parse::<f32>()
            .map_err(|_| invalid("invalid LayerNorm epsilon"))?;
        if !eps.is_finite() || eps <= 0.0 {
            return Err(invalid("LayerNorm epsilon must be finite positive FP32"));
        }
        Ok(())
    }
}

fn parse(document: &Value, source: &FusedDenseSource<'_>) -> Result<Config, ModelError> {
    let config: Config = serde_json::from_value(document.clone())
        .map_err(|e| ModelError::InvalidJson(e.to_string()))?;
    config.validate(source)?;
    // Do not silently ignore architecture-changing source extensions. These extra
    // fields are metadata or training/generation knobs, never inference operators.
    let semantic = serde_json::to_value(&config).expect("serializable config");
    let metadata = [
        "architectures",
        "n_ctx",
        "attn_pdrop",
        "embd_pdrop",
        "resid_pdrop",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "initializer_range",
        "summary_activation",
        "summary_first_dropout",
        "summary_proj_to_labels",
        "summary_type",
        "summary_use_proj",
        "task_specific_params",
        "transformers_version",
        "torch_dtype",
        "dtype",
        "_name_or_path",
        "_attn_implementation",
        "return_dict",
        "output_attentions",
        "output_hidden_states",
    ];
    for key in document
        .as_object()
        .ok_or_else(|| invalid("source must be an object"))?
        .keys()
    {
        if semantic.get(key).is_none() && !metadata.contains(&key.as_str()) {
            return Err(unsupported(&format!("unsupported source field {key}")));
        }
    }
    if document
        .get("architectures")
        .is_some_and(|v| v != &json!([source_mappings::ABSOLUTE_POSITION_ARCHITECTURE]))
    {
        return Err(unsupported(
            "source architecture must be a complete causal language model",
        ));
    }
    if document
        .get("n_ctx")
        .is_some_and(|v| v.as_u64() != Some(config.n_positions))
    {
        return Err(invalid("n_ctx must match n_positions"));
    }
    for key in ["attn_pdrop", "embd_pdrop", "resid_pdrop"] {
        if document
            .get(key)
            .is_some_and(|v| v.as_f64().is_none_or(|p| !(0.0..=1.0).contains(&p)))
        {
            return Err(invalid(
                "dropout probabilities must be in [0, 1]; dropout is disabled during inference",
            ));
        }
    }
    for key in ["torch_dtype", "dtype"] {
        if document
            .get(key)
            .is_some_and(|v| !v.is_null() && v != "float32")
        {
            return Err(unsupported(
                "source adapter requires FP32 inference semantics",
            ));
        }
    }
    if document
        .get("_attn_implementation")
        .is_some_and(|v| !v.is_null() && v != "eager")
    {
        return Err(unsupported(
            "source adapter requires eager attention semantics",
        ));
    }
    Ok(config)
}

fn capacity(config: &Config, workload: DecoderWorkload) -> Result<u64, ModelError> {
    if workload.batch == 0 || workload.max_input_tokens == 0 || workload.max_new_tokens == 0 {
        return Err(invalid("workload bounds must be positive"));
    }
    let total = workload
        .max_input_tokens
        .checked_add(workload.max_new_tokens)
        .and_then(|n| n.checked_sub(1))
        .ok_or_else(|| invalid("token capacity overflow"))?;
    if total > config.n_positions {
        return Err(invalid("workload exceeds learned position table"));
    }
    Ok(total)
}

pub(super) fn preflight(
    document: &Value,
    workload: DecoderWorkload,
    budget: DecoderExpansionBudget,
) -> Result<(), ModelError> {
    let c = parse(document, &ABSOLUTE_POSITION)?;
    capacity(&c, workload)?;
    let dense = json!({"num_hidden_layers": c.n_layer, "hidden_size": c.n_embd, "intermediate_size": c.inner(), "num_attention_heads": c.n_head, "vocab_size": c.vocab_size});
    source_budget::preflight_dense_source(
        &dense,
        workload,
        budget,
        source_budget::HeadWidth::HiddenPerHead,
    )?;
    let fused = c
        .n_embd
        .checked_mul(3)
        .ok_or_else(|| invalid("source expansion budget overflow"))?;
    let elements = workload
        .batch
        .checked_mul(workload.max_input_tokens)
        .and_then(|n| n.checked_mul(fused));
    if fused > budget.max_feature_width || elements.is_none_or(|n| n > budget.max_tensor_elements) {
        return Err(invalid(
            "decoder source expansion budget exceeded by fused QKV",
        ));
    }
    Ok(())
}

pub(super) fn lower_source_json(
    document: &Value,
    workload: DecoderWorkload,
    source: &FusedDenseSource<'_>,
) -> Result<DecoderPlan, ModelError> {
    let c = parse(document, source)?;
    let cap = capacity(&c, workload)?;
    let plan = DecoderPlan {
        schema_version: DECODER_PLAN_SCHEMA_VERSION.into(),
        model_family: source.family.into(),
        adapter: source.adapter.into(),
        config_digest: canonical_digest(source.digest_domain, &c),
        transformations: vec![],
        prefill: graph(&c, workload, DecoderMode::Prefill, cap, source.artifacts)?,
        decode: graph(&c, workload, DecoderMode::Decode, cap, source.artifacts)?,
        token_feedback: true,
    };
    plan.validate()?;
    Ok(plan)
}

fn graph(
    c: &Config,
    w: DecoderWorkload,
    mode: DecoderMode,
    cap: u64,
    a: &ArtifactLayout<'_>,
) -> Result<DecoderGraph, ModelError> {
    let (b, h, heads, d) = (w.batch, c.n_embd, c.n_head, c.n_embd / c.n_head);
    let q = if mode == DecoderMode::Prefill {
        w.max_input_tokens
    } else {
        1
    };
    let k = if mode == DecoderMode::Prefill { q } else { cap };
    let hidden = vec![b, q, h];
    let split = vec![b, heads, q, d];
    let scores = vec![b, heads, q, k];
    let state_shape = vec![b, heads, cap, d];
    let mut ops = vec![];
    push(
        &mut ops,
        "token_lookup",
        ModelOperator::TokenLookup,
        &["input.tokens"],
        hidden.clone(),
        json!({"weight":a.token_embedding,"weight_shape":[c.vocab_size,h],"compute_dtype":"float32","semantic_config":c}),
    );
    push(
        &mut ops,
        "position_lookup",
        ModelOperator::TokenLookup,
        &["input.positions"],
        hidden.clone(),
        json!({"weight":a.global("position_embedding")?,"weight_shape":[c.n_positions,h],"index_domain":"absolute_positions","position_policy":"sequential_absolute","out_of_bounds":"reject","compute_dtype":"float32"}),
    );
    push(
        &mut ops,
        "embedding_sum",
        ModelOperator::ResidualAdd,
        &["token_lookup", "position_lookup"],
        hidden.clone(),
        json!({"dropout":"disabled_inference"}),
    );
    let mut current = "embedding_sum".to_owned();
    let mut state_inputs = vec![];
    let mut state_outputs = vec![];
    for layer in 0..c.n_layer {
        let prefix = format!("layer.{layer}");
        let name = |suffix: &str| format!("{prefix}.{suffix}");
        norm(
            &mut ops,
            &name("input_norm"),
            &current,
            hidden.clone(),
            &c.layer_norm_epsilon,
            a.layer(layer.into(), "input_norm")?,
            a.layer(layer.into(), "input_norm_bias")?,
        );
        projection(
            &mut ops,
            &name("qkv_linear"),
            &name("input_norm"),
            vec![b, q, 3 * h],
            h,
            3 * h,
            a.layer(layer.into(), "attention_qkv")?,
            a.layer(layer.into(), "attention_qkv_bias")?,
        );
        for (index, role) in ["q", "k", "v"].iter().enumerate() {
            push(
                &mut ops,
                &name(&format!("{role}_slice")),
                ModelOperator::Slice,
                &[&name("qkv_linear")],
                hidden.clone(),
                json!({"axis":-1,"start":index as u64*h,"end":(index as u64+1)*h,"squeeze":false}),
            );
            push(
                &mut ops,
                &name(&format!("{role}_heads")),
                ModelOperator::Reshape,
                &[&name(&format!("{role}_slice"))],
                split.clone(),
                json!({"layout":"batch_heads_sequence_feature","input_layout":"batch_sequence_hidden"}),
            );
        }
        for (role, kind) in [("key", StateKind::Key), ("value", StateKind::Value)] {
            let state = format!("state.{prefix}.{role}");
            let append = name(&format!("{role}_append"));
            if mode == DecoderMode::Decode {
                state_inputs.push(StateTensor {
                    id: state.clone(),
                    layer: Some(layer.into()),
                    kind,
                    shape: state_shape.clone(),
                    maximum_sequence: cap,
                });
            }
            full_kv_cache_update(
                &mut ops,
                &append,
                &state,
                &name(if kind == StateKind::Key {
                    "k_heads"
                } else {
                    "v_heads"
                }),
                state_shape.clone(),
                cap,
                k,
                mode,
                kind,
            );
            state_outputs.push(StateTensor {
                id: append.clone(),
                layer: Some(layer.into()),
                kind,
                shape: state_shape.clone(),
                maximum_sequence: cap,
            });
            full_kv_cache_view(
                &mut ops,
                &name(&format!("{role}_view")),
                &append,
                vec![b, heads, k, d],
                k,
                kind,
            );
        }
        push(
            &mut ops,
            &name("attention_scores"),
            ModelOperator::AttentionScores,
            &[&name("q_heads"), &name("key_view")],
            scores.clone(),
            json!({"group_size":1}),
        );
        push(
            &mut ops,
            &name("attention_scale"),
            ModelOperator::AttentionScale,
            &[&name("attention_scores")],
            scores.clone(),
            json!({"head_dim":d}),
        );
        push(
            &mut ops,
            &name("causal_mask"),
            ModelOperator::CausalMask,
            &[
                &name("attention_scale"),
                "input.positions",
                "input.attention_mask",
                "input.sequence_lengths",
            ],
            scores.clone(),
            json!({"padding_mask_input":"input.attention_mask","valid_lengths_input":"input.sequence_lengths","cache_validity":"visible_valid_prefix","masked_value":"negative_infinity"}),
        );
        push(
            &mut ops,
            &name("softmax"),
            ModelOperator::Softmax,
            &[&name("causal_mask")],
            scores.clone(),
            json!({"axis":-1,"compute_dtype":"float32"}),
        );
        push(
            &mut ops,
            &name("attention_values"),
            ModelOperator::AttentionValues,
            &[&name("softmax"), &name("value_view")],
            split.clone(),
            json!({"group_size":1}),
        );
        push(
            &mut ops,
            &name("attention_hidden"),
            ModelOperator::Reshape,
            &[&name("attention_values")],
            hidden.clone(),
            json!({"layout":"batch_sequence_hidden","input_layout":"batch_heads_sequence_feature"}),
        );
        projection(
            &mut ops,
            &name("o_proj"),
            &name("attention_hidden"),
            hidden.clone(),
            h,
            h,
            a.layer(layer.into(), "attention_output")?,
            a.layer(layer.into(), "attention_output_bias")?,
        );
        push(
            &mut ops,
            &name("attention_residual"),
            ModelOperator::ResidualAdd,
            &[&current, &name("o_proj")],
            hidden.clone(),
            json!({}),
        );
        norm(
            &mut ops,
            &name("post_norm"),
            &name("attention_residual"),
            hidden.clone(),
            &c.layer_norm_epsilon,
            a.layer(layer.into(), "post_attention_norm")?,
            a.layer(layer.into(), "post_attention_norm_bias")?,
        );
        projection(
            &mut ops,
            &name("up_proj"),
            &name("post_norm"),
            vec![b, q, c.inner()],
            h,
            c.inner(),
            a.layer(layer.into(), "mlp_up")?,
            a.layer(layer.into(), "mlp_up_bias")?,
        );
        push(
            &mut ops,
            &name("gelu"),
            ModelOperator::GeluTanh,
            &[&name("up_proj")],
            vec![b, q, c.inner()],
            json!({"formula":"0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))","compute_dtype":"float32"}),
        );
        projection(
            &mut ops,
            &name("down_proj"),
            &name("gelu"),
            hidden.clone(),
            c.inner(),
            h,
            a.layer(layer.into(), "mlp_down")?,
            a.layer(layer.into(), "mlp_down_bias")?,
        );
        push(
            &mut ops,
            &name("mlp_residual"),
            ModelOperator::ResidualAdd,
            &[&name("attention_residual"), &name("down_proj")],
            hidden.clone(),
            json!({}),
        );
        current = name("mlp_residual");
    }
    norm(
        &mut ops,
        "final_norm",
        &current,
        hidden,
        &c.layer_norm_epsilon,
        a.final_norm.into(),
        a.global("final_norm_bias")?.into(),
    );
    if mode == DecoderMode::Prefill {
        push(
            &mut ops,
            "last_hidden",
            ModelOperator::LastToken,
            &["final_norm", "input.sequence_lengths"],
            vec![b, h],
            json!({"axis":1,"selection":"last_valid","valid_lengths_input":"input.sequence_lengths"}),
        );
    } else {
        push(
            &mut ops,
            "last_hidden",
            ModelOperator::LastToken,
            &["final_norm"],
            vec![b, h],
            json!({"axis":1}),
        );
    }
    push(
        &mut ops,
        "output_head",
        ModelOperator::OutputHead,
        &["last_hidden"],
        vec![b, c.vocab_size],
        json!({"weight":if c.tie_word_embeddings {a.token_embedding} else {a.output_head},"weight_shape":[c.vocab_size,h],"weight_layout":"output_input","bias":null,"tied":c.tie_word_embeddings,"compute_dtype":"float32","output_dtype":"float32","input_layout":"batch_hidden","output_layout":"batch_vocabulary"}),
    );
    push(
        &mut ops,
        "token_selection",
        ModelOperator::GreedyTokenSelection,
        &["output_head"],
        vec![b],
        json!({"tie_break":"lowest_index","policy":"pllm.greedy.v1","source":"execution_policy"}),
    );
    push(
        &mut ops,
        "token_feedback",
        ModelOperator::TokenFeedback,
        &["token_selection"],
        vec![b, 1],
        json!({"policy":"pllm.greedy.v1","source":"execution_policy"}),
    );
    Ok(DecoderGraph {
        mode,
        batch: b,
        query_sequence: q,
        maximum_key_sequence: k,
        operations: ops,
        state_inputs,
        state_outputs,
        output: "token_feedback".into(),
    })
}

#[allow(clippy::too_many_arguments)]
fn norm(
    ops: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    shape: Vec<u64>,
    eps: &str,
    weight: String,
    bias: String,
) {
    let width = *shape.last().expect("nonempty shape");
    push(
        ops,
        id,
        ModelOperator::LayerNorm,
        &[input],
        shape,
        json!({"epsilon":eps,"weight":weight,"bias":bias,"weight_shape":[width],"bias_shape":[width],"axis":-1,"variance":"centered_mean_square","epsilon_placement":"inside_sqrt","affine_order":"normalize_multiply_add","compute_dtype":"float32"}),
    );
}
#[allow(clippy::too_many_arguments)]
fn projection(
    ops: &mut Vec<ModelOperation>,
    id: &str,
    input: &str,
    shape: Vec<u64>,
    input_width: u64,
    output_width: u64,
    weight: String,
    bias: String,
) {
    push(
        ops,
        id,
        ModelOperator::Linear,
        &[input],
        shape,
        json!({"weight":weight,"bias":bias,"weight_shape":[input_width,output_width],"bias_shape":[output_width],"weight_layout":"input_output","formula":"x_matmul_weight_plus_bias","compute_dtype":"float32"}),
    );
}

pub(super) fn validate_layer_norm(
    op: &ModelOperation,
    shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<(), ModelError> {
    let a = &op.attributes;
    let eps = a["epsilon"].as_str().and_then(|v| v.parse::<f32>().ok());
    let width = op.output_shape.last().copied().unwrap_or(0);
    if shapes.get(&op.inputs[0]) != Some(&op.output_shape)
        || a["axis"] != -1
        || a["variance"] != "centered_mean_square"
        || a["compute_dtype"] != "float32"
        || a["epsilon_placement"] != "inside_sqrt"
        || a["affine_order"] != "normalize_multiply_add"
        || eps.is_none_or(|v| !v.is_finite() || v <= 0.0)
        || a["weight"].as_str().is_none_or(str::is_empty)
        || a["bias"].as_str().is_none_or(str::is_empty)
        || a["weight_shape"] != json!([width])
        || a["bias_shape"] != json!([width])
    {
        return Err(ModelError::Incomplete(
            "invalid centered affine LayerNorm contract".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_plan(plan: &DecoderPlan) -> Result<(), ModelError> {
    let source = &ABSOLUTE_POSITION;
    if plan.adapter != source.adapter {
        return Ok(());
    }
    let c: Config = plan
        .prefill
        .operations
        .first()
        .and_then(|op| op.attributes.get("semantic_config"))
        .cloned()
        .ok_or_else(|| ModelError::Incomplete("absolute-position semantic config missing".into()))
        .and_then(|v| {
            serde_json::from_value(v).map_err(|e| ModelError::Incomplete(e.to_string()))
        })?;
    c.validate(source)?;
    let cap = plan.decode.maximum_key_sequence;
    let w = DecoderWorkload {
        batch: plan.prefill.batch,
        max_input_tokens: plan.prefill.query_sequence,
        max_new_tokens: cap
            .checked_sub(plan.prefill.query_sequence)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("invalid state capacity"))?,
    };
    if plan.model_family != source.family
        || plan.config_digest != canonical_digest(source.digest_domain, &c)
        || capacity(&c, w)? != cap
        || !plan.transformations.is_empty()
        || plan.prefill != graph(&c, w, DecoderMode::Prefill, cap, source.artifacts)?
        || plan.decode != graph(&c, w, DecoderMode::Decode, cap, source.artifacts)?
    {
        return Err(ModelError::Incomplete(
            "absolute-position graph differs from its source semantic contract".into(),
        ));
    }
    super::validate_state_transition(plan)
}
