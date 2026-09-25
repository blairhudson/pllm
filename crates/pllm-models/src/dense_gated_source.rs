//! Source configuration reader for bounded bias-free dense gated decoders.
//! Execution and validation use the shared semantic operator vocabulary.

use super::{decimal_string, integral_u64, lower_dense_decoder, positive_decimal, rotary_scale};
use super::{DecoderPlan, DecoderWorkload, DenseDecoderConfig, ModelError};
use pllm_types::canonical_digest;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SourceConfig {
    model_type: String,
    hidden_size: u64,
    intermediate_size: u64,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    vocab_size: u64,
    max_position_embeddings: u64,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
    #[serde(default = "default_rms_norm_eps", deserialize_with = "decimal_string")]
    rms_norm_eps: String,
    #[serde(default = "default_rope_theta", deserialize_with = "integral_u64")]
    rope_theta: u64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    head_dim: Option<u64>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    attention_dropout: f64,
    #[serde(default)]
    rope_scaling: Option<Value>,
    #[serde(default)]
    sliding_window: Option<u64>,
    #[serde(default)]
    use_cache: Option<bool>,
    #[serde(default)]
    pretraining_tp: Option<u32>,
}

fn default_hidden_act() -> String {
    "silu".into()
}

fn default_rms_norm_eps() -> String {
    "0.000001".into()
}

fn default_rope_theta() -> u64 {
    10_000
}

impl SourceConfig {
    fn validate(&self) -> Result<(u64, Option<Value>), ModelError> {
        if self.model_type != "llama" {
            return Err(ModelError::Unsupported(
                "dense gated source requires model_type llama".into(),
            ));
        }
        if self.hidden_act != "silu"
            || self.attention_bias
            || self.mlp_bias
            || self.attention_dropout != 0.0
            || self.sliding_window.is_some()
            || self.use_cache == Some(false)
            || self.pretraining_tp.is_some_and(|value| value != 1)
        {
            return Err(ModelError::Unsupported(
                "dense gated source requires unscaled full attention, bias-free projections, and SiLU"
                    .into(),
            ));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == Some(0)
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
            || self.hidden_size % u64::from(self.num_attention_heads) != 0
            || self.num_attention_heads
                % self.num_key_value_heads.unwrap_or(self.num_attention_heads)
                != 0
        {
            return Err(ModelError::InvalidConfig(
                "dense gated decoder dimensions are invalid".into(),
            ));
        }
        positive_decimal(&self.rms_norm_eps, "rms_norm_eps")?;
        let head_dim = self.hidden_size / u64::from(self.num_attention_heads);
        if self.head_dim.is_some_and(|value| value != head_dim) {
            return Err(ModelError::Unsupported(
                "dense gated source requires native head width".into(),
            ));
        }
        if self.rope_theta == 0 {
            return Err(ModelError::InvalidConfig(
                "rope_theta must be positive".into(),
            ));
        }
        let scaling = rotary_scale::source_descriptor(
            self.rope_scaling.as_ref(),
            self.max_position_embeddings,
        )?;
        Ok((head_dim, scaling))
    }
}

pub(super) fn lower_json(
    bytes: &[u8],
    workload: DecoderWorkload,
) -> Result<DecoderPlan, ModelError> {
    let source: SourceConfig = serde_json::from_slice(bytes)
        .map_err(|error| ModelError::InvalidJson(error.to_string()))?;
    let (head_dim, rope_frequency_scaling) = source.validate()?;
    let digest = canonical_digest("pllm.dense_gated_source_config.v1", &source);
    lower_dense_decoder(
        &DenseDecoderConfig {
            model_family: &source.model_type,
            adapter: "pllm.dense_gated_decoder.v1",
            hidden_size: source.hidden_size,
            intermediate_size: source.intermediate_size,
            num_hidden_layers: source.num_hidden_layers,
            num_attention_heads: source.num_attention_heads,
            num_key_value_heads: source
                .num_key_value_heads
                .unwrap_or(source.num_attention_heads),
            vocab_size: source.vocab_size,
            max_position_embeddings: source.max_position_embeddings,
            head_dim,
            rms_norm_eps: &source.rms_norm_eps,
            rope_theta: source.rope_theta,
            rope_frequency_scaling,
            tie_word_embeddings: source.tie_word_embeddings,
            attention_bias: false,
            qk_norm: false,
            config_digest: digest,
        },
        workload,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn source() -> Value {
        json!({
            "model_type": "llama", "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 258,
            "max_position_embeddings": 256, "hidden_act": "silu",
            "rms_norm_eps": 0.000001, "rope_theta": 10000.0,
            "tie_word_embeddings": true, "attention_bias": false,
            "use_cache": true
        })
    }

    fn workload() -> DecoderWorkload {
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 2,
            max_new_tokens: 2,
        }
    }

    #[test]
    fn dense_source_reuses_complete_semantic_decoder_without_attention_bias() {
        let plan = lower_json(&serde_json::to_vec(&source()).unwrap(), workload()).unwrap();
        assert_eq!(plan.model_family, "llama");
        assert_eq!(plan.adapter, "pllm.dense_gated_decoder.v1");
        assert_eq!(
            plan.prefill
                .operations
                .iter()
                .find(|op| op.id == "layer.0.q_linear")
                .unwrap()
                .attributes["bias"],
            Value::Null
        );
        plan.validate().unwrap();
    }

    #[test]
    fn dense_source_rejects_unimplemented_architecture_capabilities() {
        for (key, value) in [
            ("rope_scaling", json!({"rope_type": "llama3"})),
            ("attention_bias", json!(true)),
            ("mlp_bias", json!(true)),
            ("sliding_window", json!(128)),
            ("head_dim", json!(16)),
            ("attention_dropout", json!(0.1)),
            ("use_cache", json!(false)),
            ("pretraining_tp", json!(2)),
        ] {
            let mut config = source();
            config[key] = value;
            assert!(
                lower_json(&serde_json::to_vec(&config).unwrap(), workload()).is_err(),
                "{key}"
            );
        }
    }

    #[test]
    fn bounded_wavelength_rotary_is_shared_by_query_key_and_phases() {
        let mut scaled = source();
        scaled["rope_scaling"] = json!({
            "rope_type": "llama3", "factor": 8.0,
            "original_max_position_embeddings": 32,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0
        });
        let plan = lower_json(&serde_json::to_vec(&scaled).unwrap(), workload()).unwrap();
        let unscaled = lower_json(&serde_json::to_vec(&source()).unwrap(), workload()).unwrap();
        assert_ne!(plan.digest(), unscaled.digest());
        for graph in [&plan.prefill, &plan.decode] {
            let query = graph
                .operations
                .iter()
                .find(|op| op.id == "layer.0.rope_q")
                .unwrap();
            let key = graph
                .operations
                .iter()
                .find(|op| op.id == "layer.0.rope_k")
                .unwrap();
            assert_eq!(query.attributes, key.attributes);
            assert_eq!(
                query.attributes["frequency_scaling"]["kind"],
                "wavelength_transition"
            );
            assert_eq!(
                query.attributes["coefficient_profile"],
                rotary_scale::WAVELENGTH_COEFFICIENT_PROFILE
            );
        }
        let mut forged = plan.clone();
        let key = forged
            .prefill
            .operations
            .iter_mut()
            .find(|op| op.id == "layer.0.rope_k")
            .unwrap();
        key.attributes["frequency_scaling"]["factor"] = json!(4.0);
        assert!(forged.validate().is_err());
        let mut invalid = plan;
        let key = invalid
            .decode
            .operations
            .iter_mut()
            .find(|op| op.id == "layer.0.rope_k")
            .unwrap();
        key.attributes["frequency_scaling"]["factor"] = json!(0.0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn wavelength_source_rejects_noncanonical_and_unsafe_choices() {
        let valid = json!({
            "rope_type": "llama3", "factor": 8.0,
            "original_max_position_embeddings": 32,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0
        });
        for rejected in [
            json!({"rope_type": "linear", "factor": 8.0}),
            json!({"type": "llama3", "factor": 8.0}),
            json!({"rope_type": "llama3", "factor": 8.0}),
            json!({"rope_type": "llama3", "factor": 8.0,
                "original_max_position_embeddings": 32,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0, "extra": 1}),
            json!({"rope_type": "llama3", "factor": 0.5,
                "original_max_position_embeddings": 32,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0}),
            json!({"rope_type": "llama3", "factor": 1.0000000001,
                "original_max_position_embeddings": 32,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0}),
            json!({"rope_type": "llama3", "factor": 8.0,
                "original_max_position_embeddings": 256,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0}),
            json!({"rope_type": "llama3", "factor": 8.0,
                "original_max_position_embeddings": 32,
                "low_freq_factor": 4.0, "high_freq_factor": 1.0}),
            json!({"rope_type": "llama3", "factor": 8.0,
                "original_max_position_embeddings": 32,
                "low_freq_factor": 1e-40, "high_freq_factor": 4.0}),
            json!({"rope_type": "llama3", "factor": 8.0,
                "original_max_position_embeddings": 32,
                "low_freq_factor": 1.0, "high_freq_factor": 1.000000001}),
        ] {
            let mut config = source();
            config["rope_scaling"] = rejected;
            assert!(lower_json(&serde_json::to_vec(&config).unwrap(), workload()).is_err());
        }
        let mut config = source();
        config["rope_scaling"] = valid;
        config["max_position_embeddings"] = json!(1 << 25);
        assert!(lower_json(&serde_json::to_vec(&config).unwrap(), workload()).is_err());
    }
}
