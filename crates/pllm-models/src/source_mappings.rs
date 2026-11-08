//! Source-format discovery. The registry selects a validated architectural
//! capability set; lowering modules never dispatch on a model family name.

use super::source_budget::{
    preflight_dense_readout_source, HeadWidth, ReadoutContract, SelectionFailure,
};
use super::{
    absolute_position_decoder, dense_gated_source, fused_dense_decoder, hybrid_text_decoder,
    lower_qwen3_decoder, lower_qwen_decoder, shared_kv_decoder, DecoderExpansionBudget,
    DecoderPlan, DecoderWorkload, ModelError, Qwen3Config, QwenConfig,
};
use serde::Deserialize;
use serde_json::Value;

/// Forward-only source registry. Source spellings and checkpoint paths remain
/// here; the encoder builder consumes validated geometry and semantic roles.
pub(super) struct EncoderSource {
    model_type: &'static str,
    pub(super) family: &'static str,
    pub(super) adapter: &'static str,
    pub(super) digest_domain: &'static str,
    architectures: [&'static str; 2],
    prefixes: [&'static str; 2],
    globals: &'static [(&'static str, &'static str)],
    layer_prefix: &'static str,
    layers: &'static [(&'static str, &'static str)],
}

const ENCODER_SOURCES: &[EncoderSource] = &[EncoderSource {
    model_type: "bert", family: "bert", adapter: "pllm.bert.encoder.v1",
    digest_domain: "pllm.bert.forward_source_config.v1",
    architectures: ["BertModel", "BertForSequenceClassification"],
    prefixes: ["", "bert."],
    globals: &[
        ("tokens", "embeddings.word_embeddings"),
        ("positions", "embeddings.position_embeddings"),
        ("segments", "embeddings.token_type_embeddings"),
        ("embedding_norm", "embeddings.LayerNorm"),
        ("pooler", "pooler.dense"), ("classifier", "classifier"),
    ],
    layer_prefix: "encoder.layer",
    layers: &[
        ("query", "attention.self.query"), ("key", "attention.self.key"),
        ("value", "attention.self.value"), ("attention_output", "attention.output.dense"),
        ("attention_norm", "attention.output.LayerNorm"),
        ("intermediate", "intermediate.dense"), ("output", "output.dense"),
        ("output_norm", "output.LayerNorm"),
    ],
}];

impl EncoderSource {
    pub(super) fn maximum_artifact_bytes(&self, classifier: bool, layer_digits: usize) -> Result<usize, ModelError> {
        let prefix = self.prefixes[usize::from(classifier)].len();
        let globals = self.globals.iter().map(|(role,path)|
            (if *role == "classifier" { 0 } else { prefix }) + path.len() + ".weight".len()).max().unwrap_or(0);
        let layer = self.layers.iter().map(|(_,path)| path.len()).max().unwrap_or(0);
        prefix.checked_add(self.layer_prefix.len()).and_then(|n| n.checked_add(layer_digits))
            .and_then(|n| n.checked_add(layer)).and_then(|n| n.checked_add("..weight".len()))
            .map(|n| n.max(globals)).ok_or_else(|| crate::forward_plan::invalid("artifact name length overflow"))
    }
    pub(super) fn artifact(&self, classifier: bool, layer: Option<u64>, role: &str, bias: bool)
        -> Result<String, ModelError> {
        let prefix = self.prefixes[usize::from(classifier)];
        let suffix = if bias { "bias" } else { "weight" };
        if let Some(layer) = layer {
            let path = self.layers.iter().find(|(r, _)| *r == role)
                .ok_or_else(|| ModelError::Unsupported("unknown encoder artifact role".into()))?.1;
            Ok(format!("{prefix}{}.{layer}.{path}.{suffix}", self.layer_prefix))
        } else {
            let path = self.globals.iter().find(|(r, _)| *r == role)
                .ok_or_else(|| ModelError::Unsupported("unknown encoder global role".into()))?.1;
            Ok(format!("{}{}.{suffix}", if role == "classifier" { "" } else { prefix }, path))
        }
    }
}

pub(super) const FORWARD_EXACT_DECIMALS: &[&str] = &["layer_norm_eps"];

pub(super) fn forward_geometry(document: &Value, max_sequence: u64)
    -> Result<(&'static EncoderSource, crate::encoder_blocks::Geometry<'_>), ModelError> {
    let source = ENCODER_SOURCES.iter().find(|s| document.get("model_type").and_then(Value::as_str) == Some(s.model_type))
        .ok_or_else(|| ModelError::Unsupported("source has no registered forward factory".into()))?;
    let geometry = parse_encoder_source(document, max_sequence, source)?;
    Ok((source,geometry))
}

fn parse_encoder_source<'a>(document: &'a Value, max_sequence: u64, source: &EncoderSource)
    -> Result<crate::encoder_blocks::Geometry<'a>, ModelError> {
    use crate::forward_plan::invalid;
    let o = document.as_object().ok_or_else(|| invalid("encoder source must be object"))?;
    let positive = |key: &str| o.get(key).and_then(Value::as_u64).filter(|n| *n > 0)
        .ok_or_else(|| invalid(&format!("{key} must be explicit positive JSON integer")));
    let architecture = document.get("architectures").and_then(Value::as_array)
        .filter(|a| a.len() == 1).and_then(|a| a[0].as_str())
        .ok_or_else(|| invalid("one explicit encoder architecture required"))?;
    let classifier = if architecture == source.architectures[0] { false }
        else if architecture == source.architectures[1] { true }
        else { return Err(ModelError::Unsupported("unsupported encoder architecture".into())); };
    let mut g = crate::encoder_blocks::Geometry {
        hidden: positive("hidden_size")?, intermediate: positive("intermediate_size")?,
        layers: positive("num_hidden_layers")?, heads: positive("num_attention_heads")?,
        vocabulary: positive("vocab_size")?, positions: positive("max_position_embeddings")?,
        segments: positive("type_vocab_size")?, epsilon: crate::encoder_blocks::Decimal::Text(""), labels: None,
    };
    let eps = o.get("layer_norm_eps").ok_or_else(|| invalid("explicit layer_norm_eps required"))?;
    g.epsilon = match eps {
        Value::String(s) => crate::encoder_blocks::Decimal::Text(s),
        Value::Number(n) => crate::encoder_blocks::Decimal::Number(n),
        _ => return Err(invalid("invalid layer_norm_eps")),
    };
    g.epsilon.positive_f32()?;
    if o.get("hidden_act").and_then(Value::as_str) != Some("gelu")
        || o.get("position_embedding_type").and_then(Value::as_str) != Some("absolute") {
        return Err(ModelError::Unsupported("encoder needs exact GELU and absolute positions".into()));
    }
    const KEYS: &[&str] = &[
        "model_type", "architectures", "vocab_size", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "intermediate_size", "max_position_embeddings", "type_vocab_size",
        "hidden_act", "layer_norm_eps", "position_embedding_type", "is_decoder", "is_encoder_decoder",
        "add_cross_attention", "pruned_heads", "chunk_size_feed_forward", "use_cache",
        "hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout", "_attn_implementation",
        "torch_dtype", "dtype", "output_attentions", "output_hidden_states", "return_dict", "pad_token_id",
        "initializer_range", "_name_or_path", "transformers_version", "finetuning_task", "problem_type",
        "num_labels", "id2label", "label2id",
    ];
    for key in o.keys() {
        if !KEYS.contains(&key.as_str()) {
            return Err(ModelError::Unsupported(format!("unsupported encoder source field {key}")));
        }
    }
    for key in ["is_decoder", "is_encoder_decoder", "add_cross_attention", "output_attentions", "output_hidden_states"] {
        if o.get(key).is_some_and(|v| v.as_bool() != Some(false)) { return Err(invalid(&format!("{key} must be false"))); }
    }
    for key in ["use_cache", "return_dict"] {
        if o.get(key).is_some_and(|v| !v.is_boolean()) { return Err(invalid(&format!("{key} must be boolean"))); }
    }
    if o.get("pruned_heads").is_some_and(|v| !v.as_object().is_some_and(|o| o.is_empty()))
        || o.get("chunk_size_feed_forward").is_some_and(|v| v.as_u64() != Some(0)) {
        return Err(invalid("pruning and feed-forward chunking unsupported"));
    }
    for key in ["hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout"] {
        if let Some(v) = o.get(key) {
            if key == "classifier_dropout" && v.is_null() { continue; }
            if !v.as_f64().is_some_and(|n| n.is_finite() && (0.0..=1.0).contains(&n)) {
                return Err(invalid("dropout must be finite in [0,1]"));
            }
        }
    }
    if o.get("_attn_implementation").is_some_and(|v| !v.is_null() && v.as_str() != Some("eager")) {
        return Err(invalid("only eager attention supported"));
    }
    for key in ["torch_dtype", "dtype"] {
        if o.get(key).is_some_and(|v| !v.is_null() && v.as_str() != Some("float32")) {
            return Err(invalid("source dtype must be float32"));
        }
    }
    if o.get("pad_token_id").is_some_and(|v| !v.is_null() && !v.as_u64().is_some_and(|n| n < g.vocabulary)) {
        return Err(invalid("pad_token_id outside vocabulary"));
    }
    if o.get("initializer_range").is_some_and(|v| !v.as_f64().is_some_and(|n| n.is_finite() && n >= 0.0)) {
        return Err(invalid("initializer_range must be finite nonnegative"));
    }
    for key in ["_name_or_path", "transformers_version"] {
        if o.get(key).is_some_and(|v| !v.is_string()) { return Err(invalid("source metadata must be string")); }
    }
    if o.get("finetuning_task").is_some_and(|v| !v.is_null() && !v.is_string()) {
        return Err(invalid("finetuning_task must be null or string"));
    }
    if o.get("problem_type").is_some_and(|v| !v.is_null() && !matches!(v.as_str(),
        Some("regression" | "single_label_classification" | "multi_label_classification"))) {
        return Err(invalid("unsupported loss metadata problem_type"));
    }
    let explicit_labels = if o.contains_key("num_labels") { Some(positive("num_labels")?) } else { None };
    // Null maps mean absent metadata, not an implicit class count. A present
    // num_labels must still be a positive integer; classifiers never default C.
    let id2label = o.get("id2label").filter(|v| !v.is_null()).map(|v| v.as_object().ok_or_else(|| invalid("id2label must be object or null"))).transpose()?;
    let label2id = o.get("label2id").filter(|v| !v.is_null()).map(|v| v.as_object().ok_or_else(|| invalid("label2id must be object or null"))).transpose()?;
    let resolved = explicit_labels.or_else(|| id2label.map(|m| m.len() as u64));
    if let Some(labels) = id2label {
        let count = labels.len() as u64;
        if count == 0 || resolved != Some(count) { return Err(invalid("label count mismatch")); }
        for (key,value) in labels {
            let index = key.parse::<u64>().ok().filter(|n| *n < count)
                .filter(|_| key == "0" || !key.starts_with('0'))
                .filter(|_| key.bytes().all(|b| b.is_ascii_digit()))
                .ok_or_else(|| invalid("id2label must use canonical contiguous integer keys"))?;
            let label = value.as_str()
                .ok_or_else(|| invalid("id2label must have canonical contiguous integer keys and string labels"))?;
            // Borrowed comparisons, prepaid by generic source-comparison work.
            // No BTreeSet, formatted index, epsilon clone or graph precedes quote.
            if labels.iter().any(|(other,v)| other < key && v.as_str() == Some(label)) {
                return Err(invalid("id2label labels must be unique"));
            }
            if label2id.is_some_and(|inverse| inverse.get(label).and_then(Value::as_u64) != Some(index)) {
                return Err(invalid("label2id must be exact inverse"));
            }
        }
    }
    if let Some(inverse) = label2id {
        if id2label.is_none_or(|labels| labels.len() != inverse.len()) { return Err(invalid("label2id requires exact id2label inverse")); }
    }
    if classifier { g.labels = Some(resolved.ok_or_else(|| invalid("classifier requires explicit label cardinality"))?); }
    // Geometry/expansion admission precedes any graph allocation and any source
    // canonicalization. Metadata values above never choose execution operators.
    g.validate(max_sequence)?;
    Ok(g)
}

pub(super) const SOURCE_CONTEXT_ALIASES: &[&str] = &[
    "max_position_embeddings",
    "max_sequence_length",
    "seq_length",
    "n_positions",
];

/// Resolve source scalar aliases without supplying an execution default. Every
/// present spelling must be a positive JSON integer and agree with the others.
pub(super) fn resolve_positive_integer_aliases(
    object: &serde_json::Map<String, Value>,
    ordered_aliases: &[&str],
) -> Result<Option<u64>, ModelError> {
    let mut resolved = None;
    for &alias in ordered_aliases {
        if let Some(value) = object.get(alias) {
            let value = value.as_u64().filter(|value| *value > 0).ok_or_else(|| {
                ModelError::InvalidConfig(format!("{alias} must be a positive JSON integer"))
            })?;
            if resolved.is_some_and(|previous| previous != value) {
                return Err(ModelError::InvalidConfig(format!(
                    "contradictory source integer alias {alias}"
                )));
            }
            resolved = Some(value);
        }
    }
    Ok(resolved)
}

/// Maps semantic artifact roles to exact checkpoint names. The decoder graphs
/// depend on roles; only inspected source mappings contain checkpoint paths.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct ArtifactLayout<'a> {
    pub(super) token_embedding: &'static str,
    pub(super) output_head: &'static str,
    pub(super) final_norm: &'static str,
    pub(super) layer_prefix: &'static str,
    pub(super) layer_artifacts: &'a [(&'a str, &'a str)],
    pub(super) global_artifacts: &'a [(&'a str, &'a str)],
}

impl ArtifactLayout<'_> {
    pub(super) fn layer(&self, layer: u64, role: &str) -> Result<String, ModelError> {
        let (_, path) = self
            .layer_artifacts
            .iter()
            .find(|(name, _)| *name == role)
            .ok_or_else(|| {
                ModelError::Unsupported(format!("source has no artifact for semantic role {role}"))
            })?;
        Ok(format!("{}.{layer}.{path}", self.layer_prefix))
    }

    pub(super) fn global(&self, role: &str) -> Result<&str, ModelError> {
        self.global_artifacts
            .iter()
            .find(|(name, _)| *name == role)
            .map(|(_, path)| *path)
            .ok_or_else(|| {
                ModelError::Unsupported(format!(
                    "source has no global artifact for semantic role {role}"
                ))
            })
    }
}

const HYBRID_ARTIFACTS: ArtifactLayout<'static> = ArtifactLayout {
    token_embedding: "model.language_model.embed_tokens.weight",
    output_head: "model.language_model.embed_tokens.weight",
    final_norm: "model.language_model.norm.weight",
    layer_prefix: "model.language_model.layers",
    layer_artifacts: &[
        ("input_norm", "input_layernorm.weight"),
        ("post_attention_norm", "post_attention_layernorm.weight"),
        ("mlp_gate", "mlp.gate_proj.weight"),
        ("mlp_up", "mlp.up_proj.weight"),
        ("mlp_down", "mlp.down_proj.weight"),
        ("recurrent_qkv", "linear_attn.in_proj_qkv.weight"),
        ("recurrent_gate", "linear_attn.in_proj_z.weight"),
        ("recurrent_beta", "linear_attn.in_proj_b.weight"),
        ("recurrent_decay", "linear_attn.in_proj_a.weight"),
        ("recurrent_conv", "linear_attn.conv1d.weight"),
        ("recurrent_a_log", "linear_attn.A_log"),
        ("recurrent_dt_bias", "linear_attn.dt_bias"),
        ("recurrent_norm", "linear_attn.norm.weight"),
        ("recurrent_output", "linear_attn.out_proj.weight"),
        ("attention_query", "self_attn.q_proj.weight"),
        ("attention_key", "self_attn.k_proj.weight"),
        ("attention_value", "self_attn.v_proj.weight"),
        ("attention_query_norm", "self_attn.q_norm.weight"),
        ("attention_key_norm", "self_attn.k_norm.weight"),
        ("attention_output", "self_attn.o_proj.weight"),
    ],
    global_artifacts: &[],
};

type SourceLowering = fn(&[u8], Value, DecoderWorkload) -> Result<DecoderPlan, ModelError>;
type AdmittedLowering = fn(&Value, DecoderWorkload) -> Result<DecoderPlan, ModelError>;

pub(super) fn admitted_lowerer(
    document: &Value,
    workload: DecoderWorkload,
    budget: DecoderExpansionBudget,
) -> Result<AdmittedLowering, ModelError> {
    let (_, lower, preflight) =
        eligible_mapping(document).map_err(SelectionFailure::into_model_error)?;
    preflight.check(document, workload, budget)?;
    Ok(lower)
}

fn eligible_mapping(
    document: &Value,
) -> Result<(&'static SourceMapping, AdmittedLowering, SourcePreflight), SelectionFailure> {
    eligible_mapping_from(document, SOURCE_MAPPINGS)
}

// Keep the first registry entry itself alongside its resolved callbacks. This
// selection core neither checks preflight nor invokes either source factory.
fn eligible_mapping_from(
    document: &Value,
    mappings: &'static [SourceMapping],
) -> Result<(&'static SourceMapping, AdmittedLowering, SourcePreflight), SelectionFailure> {
    let source = document.get("model_type").and_then(Value::as_str);
    let mapping = mappings
        .iter()
        .find(|mapping| Some(mapping.model_type) == source)
        .ok_or(SelectionFailure::NoAdmittedFactory)?;
    let lower = mapping
        .lower_admitted
        .ok_or(SelectionFailure::NoAdmittedFactory)?;
    let preflight = mapping.preflight.ok_or(SelectionFailure::NoExpansionContract)?;
    Ok((mapping, lower, preflight))
}
#[derive(Clone, Copy)]
enum SourcePreflight {
    Dense(HeadWidth, ReadoutContract),
    Other(fn(&Value, DecoderWorkload, DecoderExpansionBudget) -> Result<(), ModelError>),
}
impl SourcePreflight {
    fn check(
        self,
        document: &Value,
        workload: DecoderWorkload,
        budget: DecoderExpansionBudget,
    ) -> Result<(), ModelError> {
        match self {
            Self::Dense(width, readout) => {
                preflight_dense_readout_source(document, workload, budget, width, readout)
            }
            Self::Other(preflight) => preflight(document, workload, budget),
        }
    }
}

pub(super) fn dense_scalar_summary(
    document: &Value,
    workload: DecoderWorkload,
) -> Result<super::source_budget::SourceScalarSummary, super::source_budget::SourceScalarSummaryError>
{
    use super::source_budget::SourceScalarSummaryError;
    let source = document.get("model_type").and_then(Value::as_str);
    let mapping = SOURCE_MAPPINGS
        .iter()
        .find(|m| Some(m.model_type) == source)
        .ok_or(SourceScalarSummaryError::Unsupported)?;
    // Only EXISTING allocation-free admitted dense contracts qualify. GPT's
    // eager adapter and unsupported source formats remain untouched.
    match (mapping.lower_admitted, mapping.preflight) {
        (Some(_), Some(SourcePreflight::Dense(width, readout))) => {
            super::source_budget::dense_scalar_summary(document, workload, width, readout)
        }
        _ => Err(SourceScalarSummaryError::Unsupported),
    }
}

/// Declarative expansion contracts for the builders that opt in to bounded
/// metadata admission. No graph construction or family-specific execution here.
pub(super) fn metadata_expansion(
    document: &Value,
) -> Result<(&'static str, u64, u64, u64), ModelError> {
    metadata_expansion_entry(document).ok_or_else(|| {
        ModelError::Unsupported("source has no bounded metadata expansion contract".into())
    })
}

pub(super) fn metadata_expansion_entry(document: &Value) -> Option<(&'static str, u64, u64, u64)> {
    // (source format, layer field, operations per layer, attribute nodes/layer).
    // Ordinary dense views have six fixed ops, learned-position views eight;
    // each has two KV states per layer. Counts are tested against graph builders.
    const MAPPINGS: &[(&str, &str, u64, u64, u64)] = &[
        ("gpt2", "n_layer", 25, 480, 8),
        ("qwen2", "num_hidden_layers", 28, 352, 6),
        ("qwen3", "num_hidden_layers", 30, 384, 6),
        ("llama", "num_hidden_layers", 28, 352, 6),
    ];
    let source = document.get("model_type").and_then(Value::as_str);
    MAPPINGS
        .iter()
        .find(|m| Some(m.0) == source)
        .map(|m| (m.1, m.2, m.3, m.4))
}

pub(super) struct HybridSource<'a> {
    pub(super) outer_type: &'static str,
    pub(super) outer_architecture: &'static str,
    pub(super) text_type: &'static str,
    pub(super) family: &'static str,
    pub(super) adapter: &'static str,
    pub(super) digest_domain: &'static str,
    pub(super) artifacts: &'a ArtifactLayout<'a>,
}

pub(super) const HYBRID_TEXT: HybridSource<'static> = HybridSource {
    outer_type: "qwen3_5",
    outer_architecture: "Qwen3_5ForConditionalGeneration",
    text_type: "qwen3_5_text",
    family: "qwen3_5_text",
    adapter: "pllm.qwen3_5_text.v1",
    digest_domain: "pllm.qwen3_5_outer_config.v1",
    artifacts: &HYBRID_ARTIFACTS,
};

pub(super) struct FusedDenseSource<'a> {
    pub(super) model_type: &'static str,
    pub(super) family: &'static str,
    pub(super) adapter: &'static str,
    pub(super) digest_domain: &'static str,
    pub(super) artifacts: &'a ArtifactLayout<'a>,
}

const FUSED_DENSE_ARTIFACTS: ArtifactLayout<'static> = ArtifactLayout {
    token_embedding: "model.embed_tokens.weight",
    output_head: "model.embed_tokens.weight",
    final_norm: "model.norm.weight",
    layer_prefix: "model.layers",
    layer_artifacts: &[
        ("input_norm", "input_layernorm.weight"),
        ("attention_qkv", "self_attn.qkv_proj.weight"),
        ("attention_output", "self_attn.o_proj.weight"),
        ("post_attention_norm", "post_attention_layernorm.weight"),
        ("mlp_gate_up", "mlp.gate_up_proj.weight"),
        ("mlp_down", "mlp.down_proj.weight"),
    ],
    global_artifacts: &[],
};

pub(super) const FUSED_DENSE: FusedDenseSource<'static> = FusedDenseSource {
    model_type: "phi3",
    family: "phi4_mini",
    adapter: "pllm.phi4_mini.v1",
    digest_domain: "pllm.phi4_mini_config.v1",
    artifacts: &FUSED_DENSE_ARTIFACTS,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SharedKvVariant<'a> {
    pub(super) outer_model_type: &'static str,
    pub(super) outer_architecture: &'static str,
    pub(super) text_model_type: &'static str,
    pub(super) family: &'static str,
    pub(super) adapter: &'static str,
    pub(super) digest_domain: &'static str,
    pub(super) source_revision: &'static str,
    pub(super) hidden_size: u64,
    pub(super) vocab_size: u64,
    pub(super) attention_heads: u32,
    pub(super) local_head_dim: u64,
    pub(super) global_head_dim: u64,
    pub(super) sliding_window: u64,
    pub(super) max_positions: u64,
    pub(super) per_layer_input_dim: u64,
    pub(super) sliding_rope_type: &'static str,
    pub(super) sliding_rope_theta: u64,
    pub(super) full_rope_type: &'static str,
    pub(super) full_rope_theta: u64,
    pub(super) full_rotary_fraction: (u64, u64),
    pub(super) intermediate_size: u64,
    pub(super) num_layers: u32,
    pub(super) num_kv_heads: u32,
    pub(super) shared_kv_layers: u32,
    pub(super) full_attention_period: u32,
    pub(super) double_wide_mlp: bool,
    pub(super) artifacts: &'a ArtifactLayout<'a>,
}

const SHARED_KV_ARTIFACTS: ArtifactLayout<'static> = ArtifactLayout {
    token_embedding: "model.language_model.embed_tokens.weight",
    output_head: "model.language_model.embed_tokens.weight",
    final_norm: "model.language_model.norm.weight",
    layer_prefix: "model.language_model.layers",
    layer_artifacts: &[
        ("input_norm", "input_layernorm.weight"),
        ("attention_query", "self_attn.q_proj.weight"),
        ("attention_key", "self_attn.k_proj.weight"),
        ("attention_value", "self_attn.v_proj.weight"),
        ("attention_query_norm", "self_attn.q_norm.weight"),
        ("attention_key_norm", "self_attn.k_norm.weight"),
        ("attention_output", "self_attn.o_proj.weight"),
        ("post_attention_norm", "post_attention_layernorm.weight"),
        ("pre_feedforward_norm", "pre_feedforward_layernorm.weight"),
        ("mlp_gate", "mlp.gate_proj.weight"),
        ("mlp_up", "mlp.up_proj.weight"),
        ("mlp_down", "mlp.down_proj.weight"),
        ("post_feedforward_norm", "post_feedforward_layernorm.weight"),
        ("ple_gate", "per_layer_input_gate.weight"),
        ("ple_projection", "per_layer_projection.weight"),
        ("ple_norm", "post_per_layer_input_norm.weight"),
        ("layer_scalar", "layer_scalar"),
    ],
    global_artifacts: &[
        (
            "ple_token_embedding",
            "model.language_model.embed_tokens_per_layer.weight",
        ),
        (
            "ple_context_projection",
            "model.language_model.per_layer_model_projection.weight",
        ),
        (
            "ple_context_norm",
            "model.language_model.per_layer_projection_norm.weight",
        ),
    ],
};

pub(super) const SHARED_KV_E4B: SharedKvVariant<'static> = SharedKvVariant {
    outer_model_type: "gemma4",
    outer_architecture: "Gemma4ForConditionalGeneration",
    text_model_type: "gemma4_text",
    family: "gemma4_text",
    adapter: "pllm.gemma4_e4b_text.v1",
    digest_domain: "pllm.gemma4_e4b_text_config.v1",
    source_revision: "ee0ef6023621cff504d758262d4e04895a5af4a2",
    hidden_size: 2_560,
    vocab_size: 262_144,
    attention_heads: 8,
    local_head_dim: 256,
    global_head_dim: 512,
    sliding_window: 512,
    max_positions: 131_072,
    per_layer_input_dim: 256,
    sliding_rope_type: "default",
    sliding_rope_theta: 10_000,
    full_rope_type: "proportional",
    full_rope_theta: 1_000_000,
    full_rotary_fraction: (1, 4),
    intermediate_size: 10_240,
    num_layers: 42,
    num_kv_heads: 2,
    shared_kv_layers: 18,
    full_attention_period: 6,
    double_wide_mlp: false,
    artifacts: &SHARED_KV_ARTIFACTS,
};

pub(super) const SHARED_KV_E2B: SharedKvVariant<'static> = SharedKvVariant {
    outer_model_type: "gemma4",
    outer_architecture: "Gemma4ForConditionalGeneration",
    text_model_type: "gemma4_text",
    family: "gemma4_text",
    adapter: "pllm.gemma4_e2b_text.v1",
    digest_domain: "pllm.gemma4_e2b_text_config.v1",
    source_revision: "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    hidden_size: 1_536,
    vocab_size: 262_144,
    attention_heads: 8,
    local_head_dim: 256,
    global_head_dim: 512,
    sliding_window: 512,
    max_positions: 131_072,
    per_layer_input_dim: 256,
    sliding_rope_type: "default",
    sliding_rope_theta: 10_000,
    full_rope_type: "proportional",
    full_rope_theta: 1_000_000,
    full_rotary_fraction: (1, 4),
    intermediate_size: 6_144,
    num_layers: 35,
    num_kv_heads: 1,
    shared_kv_layers: 20,
    full_attention_period: 5,
    double_wide_mlp: true,
    artifacts: &SHARED_KV_ARTIFACTS,
};

pub(super) const SHARED_KV_VARIANTS: &[SharedKvVariant<'static>] = &[SHARED_KV_E2B, SHARED_KV_E4B];

struct SourceMapping {
    model_type: &'static str,
    semantic_family: &'static str,
    lower: SourceLowering,
    // Same semantic factory; the immutable Value belongs to an already admitted
    // frontend. This is not a parser, source authenticator, or admission permit.
    lower_admitted: Option<AdmittedLowering>,
    preflight: Option<SourcePreflight>,
}

const SOURCE_MAPPINGS: &[SourceMapping] = &[
    SourceMapping {
        model_type: "gpt2",
        semantic_family: "gpt2",
        lower: |_, document, workload| {
            absolute_position_decoder::lower_source_json(&document, workload, &ABSOLUTE_POSITION)
        },
        // Its existing preflight eagerly parses/clones the configuration and
        // builds a JSON adapter. Do not enter that factory before reservation.
        lower_admitted: None,
        preflight: Some(SourcePreflight::Other(absolute_position_decoder::preflight)),
    },
    SourceMapping {
        model_type: "qwen2",
        semantic_family: "qwen2",
        lower: lower_dense_qwen2,
        lower_admitted: Some(|document, workload| {
            let config = QwenConfig::deserialize(document)
                .map_err(|error| ModelError::InvalidJson(error.to_string()))?;
            lower_qwen_decoder(&config, workload)
        }),
        // lower_qwen_decoder -> shared lower_graph: final_norm -> LastToken -> OutputHead.
        preflight: Some(SourcePreflight::Dense(
            HeadWidth::HiddenPerHead,
            ReadoutContract::LastToken,
        )),
    },
    SourceMapping {
        model_type: "qwen3",
        semantic_family: "qwen3",
        lower: lower_dense_qwen3,
        lower_admitted: Some(|document, workload| {
            let config = Qwen3Config::deserialize(document)
                .map_err(|error| ModelError::InvalidJson(error.to_string()))?;
            lower_qwen3_decoder(&config, workload)
        }),
        // lower_qwen3_decoder uses the same last-token graph, with Q/K normalization.
        preflight: Some(SourcePreflight::Dense(
            HeadWidth::SourceField,
            ReadoutContract::LastToken,
        )),
    },
    SourceMapping {
        model_type: "llama",
        semantic_family: "llama",
        lower: lower_dense_gated,
        lower_admitted: Some(dense_gated_source::lower_admitted_value),
        // dense_gated_source -> lower_dense_decoder -> the same last-token graph.
        preflight: Some(SourcePreflight::Dense(
            HeadWidth::HiddenPerHead,
            ReadoutContract::LastToken,
        )),
    },
    SourceMapping {
        model_type: "gemma4",
        semantic_family: "gemma4_text",
        lower: lower_shared_kv,
        lower_admitted: None,
        preflight: None,
    },
    SourceMapping {
        model_type: "phi3",
        semantic_family: "phi4_mini",
        lower: lower_fused_dense,
        lower_admitted: None,
        preflight: None,
    },
    SourceMapping {
        model_type: "qwen3_5",
        semantic_family: "qwen3_5_text",
        lower: lower_hybrid,
        lower_admitted: None,
        preflight: None,
    },
];

pub(super) const ABSOLUTE_POSITION: FusedDenseSource<'static> = FusedDenseSource {
    model_type: "gpt2",
    family: "gpt2",
    adapter: "pllm.absolute_position_decoder.v1",
    digest_domain: "pllm.absolute_position_source_config.v1",
    artifacts: &ArtifactLayout {
        token_embedding: "transformer.wte.weight",
        output_head: "lm_head.weight",
        final_norm: "transformer.ln_f.weight",
        layer_prefix: "transformer.h",
        layer_artifacts: &[
            ("input_norm", "ln_1.weight"),
            ("input_norm_bias", "ln_1.bias"),
            ("post_attention_norm", "ln_2.weight"),
            ("post_attention_norm_bias", "ln_2.bias"),
            ("attention_qkv", "attn.c_attn.weight"),
            ("attention_qkv_bias", "attn.c_attn.bias"),
            ("attention_output", "attn.c_proj.weight"),
            ("attention_output_bias", "attn.c_proj.bias"),
            ("mlp_up", "mlp.c_fc.weight"),
            ("mlp_up_bias", "mlp.c_fc.bias"),
            ("mlp_down", "mlp.c_proj.weight"),
            ("mlp_down_bias", "mlp.c_proj.bias"),
        ],
        global_artifacts: &[
            ("position_embedding", "transformer.wpe.weight"),
            ("final_norm_bias", "transformer.ln_f.bias"),
        ],
    },
};

pub(super) const ABSOLUTE_POSITION_ARCHITECTURE: &str = "GPT2LMHeadModel";

pub(super) fn lower_json(
    bytes: &[u8],
    workload: DecoderWorkload,
    budget: Option<DecoderExpansionBudget>,
) -> Result<DecoderPlan, ModelError> {
    let document: Value = serde_json::from_slice(bytes)
        .map_err(|error| ModelError::InvalidJson(error.to_string()))?;
    let model_type = document
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| ModelError::InvalidConfig("model_type must be a string".into()))?;
    let mapping = SOURCE_MAPPINGS
        .iter()
        .find(|mapping| mapping.model_type == model_type)
        .ok_or_else(|| {
            ModelError::Unsupported(format!("model_type {model_type} has no decoder adapter"))
        })?;
    lower_mapped(mapping, bytes, document, workload, budget)
}

fn lower_mapped(
    mapping: &SourceMapping,
    bytes: &[u8],
    document: Value,
    workload: DecoderWorkload,
    budget: Option<DecoderExpansionBudget>,
) -> Result<DecoderPlan, ModelError> {
    if let Some(budget) = budget {
        let preflight = mapping.preflight.ok_or_else(|| {
            ModelError::Unsupported("source has no bounded expansion contract".into())
        })?;
        preflight.check(&document, workload, budget)?;
    }
    let plan = (mapping.lower)(bytes, document, workload)?;
    if plan.model_family != mapping.semantic_family {
        return Err(ModelError::Unsupported(
            "source mapping changed semantic family".into(),
        ));
    }
    Ok(plan)
}

fn lower_dense_qwen2(
    bytes: &[u8],
    _: Value,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    lower_qwen_decoder(&QwenConfig::from_json(bytes)?, workload)
}

fn lower_dense_qwen3(
    bytes: &[u8],
    _: Value,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    lower_qwen3_decoder(&Qwen3Config::from_json(bytes)?, workload)
}

fn lower_dense_gated(
    bytes: &[u8],
    _: Value,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    dense_gated_source::lower_json(bytes, workload)
}

fn lower_shared_kv(
    _: &[u8],
    document: Value,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    shared_kv_decoder::lower_source_json(document, workload, SHARED_KV_VARIANTS)
}

fn lower_fused_dense(
    _: &[u8],
    document: Value,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    fused_dense_decoder::lower_source_json(&document, workload, &FUSED_DENSE)
}

fn lower_hybrid(
    _: &[u8],
    document: Value,
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    hybrid_text_decoder::lower_source_json(&document, workload, &HYBRID_TEXT)
}

#[cfg(test)]
#[path = "source_fixed_preflight_tests.rs"]
mod source_fixed_preflight_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::path::Path;

    #[test]
    fn metadata_recipe_counts_cover_valid_registered_views_without_changing_identity() {
        let budget = DecoderExpansionBudget {
            max_layers: 1024,
            max_feature_width: 32768,
            max_attention_heads: 32768,
            max_vocabulary: 32768,
            max_tensor_elements: 32768,
        };
        let workload = DecoderWorkload {
            batch: 1,
            max_input_tokens: 2,
            max_new_tokens: 2,
        };
        for family in ["gpt2", "qwen2", "qwen3", "llama"] {
            for layers in [1, 2, 3] {
                let source = if family == "gpt2" {
                    json!({"model_type":family,"n_layer":layers,"n_embd":4,"n_head":2,
                        "n_inner":8,"n_positions":8,"vocab_size":7})
                } else {
                    json!({"model_type":family,"num_hidden_layers":layers,"hidden_size":4,
                        "intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,
                        "head_dim":2,"vocab_size":7,"max_position_embeddings":8,
                        "hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000,
                        "tie_word_embeddings":true,"attention_bias":false,"attention_dropout":0.0,
                        "use_sliding_window":false,"use_cache":true,"max_window_layers":layers})
                };
                let estimate = budget
                    .preflight_metadata(&source, workload, 1024, 1024, 32768, 1 << 20)
                    .unwrap();
                let bytes = serde_json::to_vec(&source).unwrap();
                let plain = lower_json(&bytes, workload, None).unwrap();
                let bounded = lower_json(&bytes, workload, Some(budget)).unwrap();
                assert_eq!(plain, bounded);
                assert_eq!(plain.digest(), bounded.digest());
                // Independent byte frontend still invokes the original source
                // factories. The opt-in frontend must not serialize a plan, use
                // its digest as provenance, or replace any numeric recipe.
                let mapping = SOURCE_MAPPINGS
                    .iter()
                    .find(|m| m.model_type == family)
                    .unwrap();
                if mapping.lower_admitted.is_some() {
                    let lower = admitted_lowerer(&source, workload, budget).unwrap();
                    let from_value = lower(&source, workload).unwrap();
                    assert_eq!(plain, from_value);
                    assert_eq!(plain.digest(), from_value.digest());
                } else {
                    assert!(admitted_lowerer(&source, workload, budget).is_err());
                }
                assert_eq!(
                    plain.prefill.operations.len(),
                    estimate.operations_per_phase,
                    "{family}"
                );
                assert_eq!(
                    plain.decode.operations.len(),
                    estimate.operations_per_phase,
                    "{family}"
                );
                assert_eq!(plain.prefill.state_outputs.len(), estimate.states_per_phase);
                fn nodes(v: &Value) -> usize {
                    1 + match v {
                        Value::Array(a) => a.iter().map(nodes).sum::<usize>(),
                        Value::Object(m) => m.values().map(nodes).sum::<usize>(),
                        _ => 0,
                    }
                }
                let actual: usize = [&plain.prefill, &plain.decode]
                    .into_iter()
                    .flat_map(|g| &g.operations)
                    .map(|op| nodes(&op.attributes))
                    .sum();
                assert!(
                    actual <= estimate.attribute_nodes,
                    "{family}: {actual} > {}",
                    estimate.attribute_nodes
                );
            }
        }
    }

    fn lowering_sentinel(
        _: &[u8],
        _: Value,
        _: DecoderWorkload,
    ) -> Result<DecoderPlan, ModelError> {
        Err(ModelError::Unsupported("graph lowering reached".into()))
    }

    #[test]
    fn expansion_admission_precedes_the_graph_lowering_callback() {
        let mapping = SourceMapping {
            model_type: "sentinel",
            semantic_family: "sentinel",
            lower: lowering_sentinel,
            lower_admitted: None,
            preflight: Some(SourcePreflight::Dense(
                HeadWidth::SourceField,
                ReadoutContract::FullPrefix,
            )),
        };
        let budget = DecoderExpansionBudget {
            max_layers: 16,
            max_feature_width: 256,
            max_attention_heads: 256,
            max_vocabulary: 256,
            max_tensor_elements: 1 << 18,
        };
        let workload = DecoderWorkload {
            batch: 1,
            max_input_tokens: 3,
            max_new_tokens: 2,
        };
        let source = json!({
            "num_hidden_layers": 2, "hidden_size": 8, "intermediate_size": 16,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 32,
            "head_dim": 4
        });
        for (field, value) in [
            ("num_hidden_layers", u32::MAX as u64),
            ("hidden_size", u64::MAX),
            ("intermediate_size", u64::MAX),
            ("num_attention_heads", u32::MAX as u64),
            ("num_key_value_heads", u32::MAX as u64),
            ("head_dim", u64::MAX),
            ("head_dim", 256), // Each dimension fits; the projection does not.
            ("vocab_size", u64::MAX),
            ("num_hidden_layers", 0),
        ] {
            let mut changed = source.clone();
            changed[field] = json!(value);
            let error = lower_mapped(&mapping, b"", changed, workload, Some(budget)).unwrap_err();
            assert!(
                error.to_string().contains("source expansion budget"),
                "{field}: {error}"
            );
        }
        for workload in [
            DecoderWorkload {
                max_input_tokens: u64::MAX,
                ..workload
            },
            DecoderWorkload {
                batch: 1 << 18,
                ..workload
            },
            DecoderWorkload {
                batch: 0,
                ..workload
            },
        ] {
            let error =
                lower_mapped(&mapping, b"", source.clone(), workload, Some(budget)).unwrap_err();
            assert!(error.to_string().contains("source expansion budget"));
        }
        let error =
            lower_mapped(&mapping, b"", source.clone(), workload, Some(budget)).unwrap_err();
        assert!(error.to_string().contains("graph lowering reached"));
        let missing = SourceMapping {
            preflight: None,
            ..mapping
        };
        let error = lower_mapped(&missing, b"", source, workload, Some(budget)).unwrap_err();
        assert!(error.to_string().contains("no bounded expansion contract"));
    }

    #[test]
    fn expansion_prices_resolved_heads_and_complete_projection_shapes() {
        for (family, head_dim, input, outputs) in [
            ("qwen2", 1, 1, 128), // Ignored source field cannot underprice native-width KV.
            ("qwen3", 128, 2, 1), // Query projection is wider than hidden/KV activations.
        ] {
            let source = json!({
                "model_type": family, "num_hidden_layers": 1, "hidden_size": 8,
                "intermediate_size": 16, "num_attention_heads": 2,
                "num_key_value_heads": 1, "vocab_size": 32, "head_dim": head_dim,
                "max_position_embeddings": 256, "hidden_act": "silu",
                "rms_norm_eps": 0.000001, "rope_theta": 10000,
                "tie_word_embeddings": true, "attention_bias": false,
                "attention_dropout": 0.0, "rope_scaling": null, "sliding_window": null,
                "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
            });
            let workload = DecoderWorkload {
                batch: 1,
                max_input_tokens: input,
                max_new_tokens: outputs,
            };
            let budget = DecoderExpansionBudget {
                max_layers: 16,
                max_feature_width: 256,
                max_attention_heads: 256,
                max_vocabulary: 256,
                max_tensor_elements: 256,
            };
            let registered = SOURCE_MAPPINGS
                .iter()
                .find(|mapping| mapping.model_type == family)
                .unwrap();
            let sentinel = SourceMapping {
                lower: lowering_sentinel,
                ..*registered
            };
            let error =
                lower_mapped(&sentinel, b"", source.clone(), workload, Some(budget)).unwrap_err();
            assert!(error.to_string().contains("source expansion budget"));
            let admitted = DecoderExpansionBudget {
                max_tensor_elements: 512,
                ..budget
            };
            let error =
                lower_mapped(&sentinel, b"", source.clone(), workload, Some(admitted)).unwrap_err();
            assert!(error.to_string().contains("graph lowering reached"));
            let bytes = serde_json::to_vec(&source).unwrap();
            let plan = lower_json(&bytes, workload, Some(admitted)).unwrap();
            assert_eq!(plan, lower_json(&bytes, workload, None).unwrap());
            let mut largest = 0;
            for graph in [&plan.prefill, &plan.decode] {
                for shape in graph
                    .operations
                    .iter()
                    .map(|op| &op.output_shape)
                    .chain(graph.state_inputs.iter().map(|state| &state.shape))
                    .chain(graph.state_outputs.iter().map(|state| &state.shape))
                {
                    largest = largest.max(shape.iter().product::<u64>());
                }
            }
            assert_eq!(largest, 512, "{family}");
        }
    }

    #[test]
    fn executable_decoder_modules_are_named_and_bound_by_architecture() {
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for module in [
            "absolute_position_decoder.rs",
            "hybrid_text_decoder.rs",
            "fused_dense_decoder.rs",
            "shared_kv_decoder.rs",
        ] {
            let contents = fs::read_to_string(source_dir.join(module)).unwrap();
            let implementation = contents.split("#[cfg(test)]").next().unwrap();
            for source_label in ["qwen", "gemma", "phi4", "gpt2", "GPT2"] {
                assert!(
                    !implementation.contains(source_label),
                    "{module} places source identity {source_label:?} in executable graph code"
                );
            }
            for raw_path in ["model.layers.", "model.language_model.layers."] {
                assert!(
                    !implementation.contains(raw_path),
                    "{module} embeds source-specific checkpoint path {raw_path:?}"
                );
            }
        }
        for former_module in ["qwen35.rs", "phi4.rs", "gemma4.rs"] {
            assert!(
                !source_dir.join(former_module).exists(),
                "new source formats must use declarative mappings, not family-named graph files"
            );
        }
    }
}
