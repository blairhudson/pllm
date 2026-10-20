//! Caller-selected bounds for source expansion before semantic graph allocation.
//! Source mappings opt in only when these dimensions bound their graph builder.

use crate::{DecoderWorkload, ModelError};
use serde_json::Value;

/// Pre-construction logical metadata allowance. These are conservative, checked
/// builder bounds, not allocator observations or a process/RSS reservation.
#[derive(Debug, Clone, Copy)]
pub struct SourceMetadataEstimate {
    pub operations_per_phase: usize,
    pub states_per_phase: usize,
    pub attribute_nodes: usize,
    pub retained_string_bytes: usize,
    pub retained_metadata_bytes: usize,
}

impl DecoderExpansionBudget {
    /// Additional opt-in admission; old lowering entrypoints/digests are unchanged.
    /// Callers still bound config bytes, and must bound serialized output before
    /// materializing a canonical document. Mapping formulas precede expansion.
    pub fn preflight_metadata(
        &self,
        document: &Value,
        workload: DecoderWorkload,
        max_operations: usize,
        max_string: usize,
        max_nodes: usize,
        max_strings: usize,
    ) -> Result<SourceMetadataEstimate, ModelError> {
        fn strings(
            v: &Value,
            depth: usize,
            ceiling: usize,
            nodes: &mut usize,
        ) -> Result<(), ModelError> {
            *nodes = nodes.checked_add(1).ok_or_else(rejected)?;
            if depth > 16 || *nodes > 32768 {
                return Err(rejected());
            }
            match v {
                Value::String(s) if s.len() > ceiling => return Err(rejected()),
                Value::Array(a) => {
                    for v in a {
                        strings(v, depth + 1, ceiling, nodes)?;
                    }
                }
                Value::Object(m) => {
                    for (k, v) in m {
                        if k.len() > ceiling {
                            return Err(rejected());
                        }
                        strings(v, depth + 1, ceiling, nodes)?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
        // In particular, reject a huge valid decimal before each norm clones it.
        let mut config_nodes = 0;
        strings(document, 0, max_string, &mut config_nodes)?;
        let (layer_key, per_layer, attr_per_layer, fixed_operations) =
            super::source_mappings::metadata_expansion(document)?;
        let layers = positive(field(document, layer_key)?, self.max_layers)?;
        if workload.batch == 0 || workload.max_input_tokens == 0 || workload.max_new_tokens == 0 {
            return Err(rejected());
        }
        let operations = layers
            .checked_mul(per_layer)
            .and_then(|n| n.checked_add(fixed_operations))
            .ok_or_else(rejected)?;
        let operations = usize::try_from(operations).map_err(|_| rejected())?;
        if operations > max_operations {
            return Err(rejected());
        }
        let layers = usize::try_from(layers).map_err(|_| rejected())?;
        let nodes = layers
            .checked_mul(attr_per_layer as usize)
            .and_then(|n| n.checked_add(256))
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(config_nodes * 2))
            .ok_or_else(rejected)?;
        let epsilon = document
            .get("rms_norm_eps")
            .or_else(|| document.get("layer_norm_epsilon"))
            .and_then(Value::as_str)
            .map_or(32, str::len);
        // Static paths/IDs/keys are bounded by the registered builder recipes;
        // 4 KiB/layer/view covers their aggregate string payload. Norms may clone
        // epsilon four times/layer/view (Q/K norms), plus the final norm.
        let retained_strings = layers
            .checked_mul(4096)
            .and_then(|n| n.checked_add(4096))
            .and_then(|n| {
                (layers * 4 + 1)
                    .checked_mul(epsilon)
                    .and_then(|e| n.checked_add(e))
            })
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(rejected)?;
        if nodes > max_nodes || retained_strings > max_strings {
            return Err(rejected());
        }
        // Registered recipes bound ordinary op attributes at 8 KiB of map/vector
        // backing (four sparse BTree nodes plus arrays/nested fields). A further
        // 256 KiB/view covers the once-per-view source config. A standard BTree
        // node has 11 key/value slots and 12 edges (<1024 B on this ABI). Leaf
        // headers, op vectors and strings are additional below. The importer
        // separately prices phase overlap, not process allocator/RSS usage.
        let retained = nodes
            .checked_mul(std::mem::size_of::<Value>())
            .and_then(|n| {
                n.checked_add(
                    operations * 2 * (std::mem::size_of::<crate::ModelOperation>() + 512 + 8192),
                )
            })
            .and_then(|n| n.checked_add(2 * (256 << 10)))
            .and_then(|n| n.checked_add(retained_strings * 2))
            .ok_or_else(rejected)?;
        Ok(SourceMetadataEstimate {
            operations_per_phase: operations,
            states_per_phase: layers * 2,
            attribute_nodes: nodes,
            retained_string_bytes: retained_strings,
            retained_metadata_bytes: retained,
        })
    }
}

/// Logical expansion limits, not a process-memory or source-byte allowance.
/// Callers must bound configuration bytes before parsing. Unsupported source
/// mappings reject rather than constructing a graph without this admission.
#[derive(Clone, Copy, Debug)]
pub struct DecoderExpansionBudget {
    pub max_layers: u64,
    pub max_feature_width: u64,
    pub max_attention_heads: u64,
    pub max_vocabulary: u64,
    /// Maximum semantic activation/state elements; source parameter shapes are
    /// bounded by the separate width/vocabulary ceilings, not this field.
    pub max_tensor_elements: u64,
}

#[derive(Clone, Copy)]
pub(super) enum HeadWidth {
    HiddenPerHead,
    SourceField,
}

fn rejected() -> ModelError {
    ModelError::InvalidConfig("decoder source expansion budget exceeded or invalid".into())
}

fn positive(value: u64, ceiling: u64) -> Result<u64, ModelError> {
    if value == 0 || value > ceiling {
        return Err(rejected());
    }
    Ok(value)
}

fn field(document: &Value, key: &str) -> Result<u64, ModelError> {
    document
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(rejected)
}

fn optional_field(document: &Value, key: &str, fallback: u64) -> Result<u64, ModelError> {
    match document.get(key) {
        None | Some(Value::Null) => Ok(fallback),
        Some(value) => value.as_u64().ok_or_else(rejected),
    }
}

fn product(values: &[u64], ceiling: u64) -> Result<(), ModelError> {
    let value = values
        .iter()
        .try_fold(1u64, |total, value| total.checked_mul(*value))
        .ok_or_else(rejected)?;
    positive(value, ceiling)?;
    Ok(())
}

/// Dimension layout shared by the registered ordinary dense source readers.
/// Their existing validators still check architectural and numeric semantics.
pub(super) fn preflight_dense_source(
    document: &Value,
    workload: DecoderWorkload,
    budget: DecoderExpansionBudget,
    head_width: HeadWidth,
) -> Result<(), ModelError> {
    positive(field(document, "num_hidden_layers")?, budget.max_layers)?;
    let hidden = positive(field(document, "hidden_size")?, budget.max_feature_width)?;
    let intermediate = positive(
        field(document, "intermediate_size")?,
        budget.max_feature_width,
    )?;
    let heads = positive(
        field(document, "num_attention_heads")?,
        budget.max_attention_heads,
    )?;
    let kv_heads = positive(
        optional_field(document, "num_key_value_heads", heads)?,
        budget.max_attention_heads,
    )?;
    let head_dim = positive(
        match head_width {
            HeadWidth::HiddenPerHead => hidden / heads,
            HeadWidth::SourceField => field(document, "head_dim")?,
        },
        budget.max_feature_width,
    )?;
    let vocabulary = positive(field(document, "vocab_size")?, budget.max_vocabulary)?;
    product(&[heads, head_dim], budget.max_feature_width)?;
    product(&[kv_heads, head_dim], budget.max_feature_width)?;
    if workload.batch == 0 || workload.max_input_tokens == 0 || workload.max_new_tokens == 0 {
        return Err(rejected());
    }
    let capacity = workload
        .max_input_tokens
        .checked_add(workload.max_new_tokens)
        .and_then(|total| total.checked_sub(1))
        .ok_or_else(rejected)?;
    let batch = workload.batch;
    let rows = workload.max_input_tokens;
    for shape in [
        vec![batch, rows, hidden],
        vec![batch, rows, intermediate],
        vec![batch, rows, vocabulary],
        vec![batch, rows, heads, head_dim],
        vec![batch, heads, rows, rows],
        vec![batch, heads, capacity],
        vec![batch, kv_heads, capacity, head_dim],
    ] {
        product(&shape, budget.max_tensor_elements)?;
    }
    Ok(())
}
