//! Source-format discovery. The registry selects a validated architectural
//! capability set; lowering modules never dispatch on a model family name.

use super::{
    dense_gated_source, fused_dense_decoder, hybrid_text_decoder, lower_qwen3_decoder,
    lower_qwen_decoder, shared_kv_decoder, DecoderPlan, DecoderWorkload, ModelError, Qwen3Config,
    QwenConfig,
};
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
}

const SOURCE_MAPPINGS: &[SourceMapping] = &[
    SourceMapping {
        model_type: "qwen2",
        semantic_family: "qwen2",
        lower: lower_dense_qwen2,
    },
    SourceMapping {
        model_type: "qwen3",
        semantic_family: "qwen3",
        lower: lower_dense_qwen3,
    },
    SourceMapping {
        model_type: "llama",
        semantic_family: "llama",
        lower: lower_dense_gated,
    },
    SourceMapping {
        model_type: "gemma4",
        semantic_family: "gemma4_text",
        lower: lower_shared_kv,
    },
    SourceMapping {
        model_type: "phi3",
        semantic_family: "phi4_mini",
        lower: lower_fused_dense,
    },
    SourceMapping {
        model_type: "qwen3_5",
        semantic_family: "qwen3_5_text",
        lower: lower_hybrid,
    },
];

pub(super) fn lower_json(
    bytes: &[u8],
    workload: DecoderWorkload,
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
    use std::fs;
    use std::path::Path;

    #[test]
    fn executable_decoder_modules_are_named_and_bound_by_architecture() {
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for module in [
            "hybrid_text_decoder.rs",
            "fused_dense_decoder.rs",
            "shared_kv_decoder.rs",
        ] {
            let contents = fs::read_to_string(source_dir.join(module)).unwrap();
            let implementation = contents.split("#[cfg(test)]").next().unwrap();
            for source_label in ["qwen", "gemma", "phi4"] {
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
