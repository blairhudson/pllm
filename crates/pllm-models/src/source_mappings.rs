//! Source-format discovery. The registry selects a validated architectural
//! capability set; lowering modules never dispatch on a model family name.

use super::source_budget::{preflight_dense_source, HeadWidth};
use super::{
    absolute_position_decoder, dense_gated_source, fused_dense_decoder, hybrid_text_decoder,
    lower_qwen3_decoder, lower_qwen_decoder, shared_kv_decoder, DecoderExpansionBudget,
    DecoderPlan, DecoderWorkload, ModelError, Qwen3Config, QwenConfig,
};
use serde::Deserialize;
use serde_json::Value;

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
    let source = document.get("model_type").and_then(Value::as_str);
    let mapping = SOURCE_MAPPINGS
        .iter()
        .find(|mapping| Some(mapping.model_type) == source)
        .ok_or_else(|| ModelError::Unsupported("source has no admitted Value factory".into()))?;
    let lower = mapping
        .lower_admitted
        .ok_or_else(|| ModelError::Unsupported("source has no admitted Value factory".into()))?;
    let preflight = mapping.preflight.ok_or_else(|| {
        ModelError::Unsupported("source has no bounded expansion contract".into())
    })?;
    preflight.check(document, workload, budget)?;
    Ok(lower)
}
#[derive(Clone, Copy)]
enum SourcePreflight {
    Dense(HeadWidth),
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
            Self::Dense(width) => preflight_dense_source(document, workload, budget, width),
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
        (Some(_), Some(SourcePreflight::Dense(width))) => {
            super::source_budget::dense_scalar_summary(document, workload, width)
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
        preflight: Some(SourcePreflight::Dense(HeadWidth::HiddenPerHead)),
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
        preflight: Some(SourcePreflight::Dense(HeadWidth::SourceField)),
    },
    SourceMapping {
        model_type: "llama",
        semantic_family: "llama",
        lower: lower_dense_gated,
        lower_admitted: Some(dense_gated_source::lower_admitted_value),
        preflight: Some(SourcePreflight::Dense(HeadWidth::HiddenPerHead)),
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
            preflight: Some(SourcePreflight::Dense(HeadWidth::SourceField)),
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
