//! Caller-selected bounds for source expansion before semantic graph allocation.
//! Source mappings opt in only when these dimensions bound their graph builder.

use crate::{DecoderWorkload, ModelError};
use serde_json::Value;

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
